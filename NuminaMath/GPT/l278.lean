import Mathlib

namespace honey_teas_l278_278529

-- Definitions corresponding to the conditions
def evening_cups := 2
def evening_servings_per_cup := 2
def morning_cups := 1
def morning_servings_per_cup := 1
def afternoon_cups := 1
def afternoon_servings_per_cup := 1
def servings_per_ounce := 6
def container_ounces := 16

-- Calculation for total servings of honey per day and total days until the container is empty
theorem honey_teas :
  (container_ounces * servings_per_ounce) / 
  (evening_cups * evening_servings_per_cup +
   morning_cups * morning_servings_per_cup +
   afternoon_cups * afternoon_servings_per_cup) = 16 :=
by
  sorry

end honey_teas_l278_278529


namespace find_divisor_l278_278737

theorem find_divisor (n x : ℕ) (h1 : n = 3) (h2 : (n / x : ℝ) * 12 = 9): x = 4 := by
  sorry

end find_divisor_l278_278737


namespace odd_periodic_function_l278_278690

noncomputable def f : ℤ → ℤ := sorry

theorem odd_periodic_function (f_odd : ∀ x : ℤ, f (-x) = -f x)
  (period_f_3x1 : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 3) + 1))
  (f_one : f 1 = -1) : f 2006 = 1 :=
sorry

end odd_periodic_function_l278_278690


namespace Fabian_total_cost_correct_l278_278784

noncomputable def total_spent_by_Fabian (mouse_cost : ℝ) : ℝ :=
  let keyboard_cost := 2 * mouse_cost
  let headphones_cost := mouse_cost + 15
  let usb_hub_cost := 36 - mouse_cost
  let webcam_cost := keyboard_cost / 2
  let total_cost := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost + webcam_cost
  let discounted_total := total_cost * 0.90
  let final_total := discounted_total * 1.05
  final_total

theorem Fabian_total_cost_correct :
  total_spent_by_Fabian 20 = 123.80 :=
by
  sorry

end Fabian_total_cost_correct_l278_278784


namespace range_of_a_l278_278481

theorem range_of_a (x a : ℝ) (p : (x + 1)^2 > 4) (q : x > a) 
  (h : (¬((x + 1)^2 > 4)) → (¬(x > a)))
  (sufficient_but_not_necessary : (¬((x + 1)^2 > 4)) → (¬(x > a))) : a ≥ 1 :=
sorry

end range_of_a_l278_278481


namespace beth_sheep_l278_278144

-- Definition: number of sheep Beth has (B)
variable (B : ℕ)

-- Condition 1: Aaron has 7 times as many sheep as Beth
def Aaron_sheep (B : ℕ) := 7 * B

-- Condition 2: Together, Aaron and Beth have 608 sheep
axiom together_sheep : B + Aaron_sheep B = 608

-- Theorem: Prove that Beth has 76 sheep
theorem beth_sheep : B = 76 :=
sorry

end beth_sheep_l278_278144


namespace total_value_of_item_l278_278736

variable (V : ℝ) -- Total value of the item

def import_tax (V : ℝ) := 0.07 * (V - 1000) -- Definition of import tax

theorem total_value_of_item
  (htax_paid : import_tax V = 112.70) :
  V = 2610 := 
by
  sorry

end total_value_of_item_l278_278736


namespace length_of_picture_frame_l278_278500

theorem length_of_picture_frame (P W : ℕ) (hP : P = 30) (hW : W = 10) : ∃ L : ℕ, 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end length_of_picture_frame_l278_278500


namespace deepak_and_wife_meet_time_l278_278120

noncomputable def deepak_speed_kmph : ℝ := 20
noncomputable def wife_speed_kmph : ℝ := 12
noncomputable def track_circumference_m : ℝ := 1000

noncomputable def speed_to_m_per_min (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 60

noncomputable def deepak_speed_m_per_min : ℝ := speed_to_m_per_min deepak_speed_kmph
noncomputable def wife_speed_m_per_min : ℝ := speed_to_m_per_min wife_speed_kmph

noncomputable def combined_speed_m_per_min : ℝ :=
  deepak_speed_m_per_min + wife_speed_m_per_min

noncomputable def meeting_time_minutes : ℝ :=
  track_circumference_m / combined_speed_m_per_min

theorem deepak_and_wife_meet_time :
  abs (meeting_time_minutes - 1.875) < 0.01 :=
by
  sorry

end deepak_and_wife_meet_time_l278_278120


namespace smallest_positive_angle_l278_278701

theorem smallest_positive_angle (deg : ℤ) (k : ℤ) (h : deg = -2012) : ∃ m : ℤ, m = 148 ∧ 0 ≤ m ∧ m < 360 ∧ (∃ n : ℤ, deg + 360 * n = m) :=
by
  sorry

end smallest_positive_angle_l278_278701


namespace min_direction_changes_l278_278450

theorem min_direction_changes (n : ℕ) : 
  ∀ (path : Finset (ℕ × ℕ)), 
    (path.card = (n + 1) * (n + 2) / 2) → 
    (∀ (v : ℕ × ℕ), v ∈ path) →
    ∃ changes, (changes ≥ n) :=
by sorry

end min_direction_changes_l278_278450


namespace simplify_fraction_150_div_225_l278_278678

theorem simplify_fraction_150_div_225 :
  let a := 150
  let b := 225
  let gcd_ab := Nat.gcd a b
  let num_fact := 2 * 3 * 5^2
  let den_fact := 3^2 * 5^2
  gcd_ab = 75 →
  num_fact = a →
  den_fact = b →
  (a / gcd_ab) / (b / gcd_ab) = (2 / 3) :=
  by
    intros 
    sorry

end simplify_fraction_150_div_225_l278_278678


namespace no_solution_k_l278_278979

theorem no_solution_k (k : ℝ) : 
  (∀ t s : ℝ, 
    ∃ (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (d : ℝ × ℝ), 
      (a = (2, 7)) ∧ 
      (b = (5, -9)) ∧ 
      (c = (4, -3)) ∧ 
      (d = (-2, k)) ∧ 
      (a + t • b ≠ c + s • d)) ↔ k = 18 / 5 := 
by
  sorry

end no_solution_k_l278_278979


namespace find_C_coordinates_l278_278522

open Real

noncomputable def pointC_coordinates (A B : ℝ × ℝ) (hA : A = (-1, 0)) (hB : B = (3, 8)) (hdist : dist A C = 2 * dist C B) : ℝ × ℝ :=
  (⟨7 / 3, 20 / 3⟩)

theorem find_C_coordinates :
  ∀ (A B C : ℝ × ℝ), 
  A = (-1, 0) → B = (3, 8) → dist A C = 2 * dist C B →
  C = (7 / 3, 20 / 3) :=
by 
  intros A B C hA hB hdist
  -- We will use the given conditions and definitions to find the coordinates of C
  sorry

end find_C_coordinates_l278_278522


namespace distinct_solution_count_number_of_solutions_l278_278024

theorem distinct_solution_count (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 := by
  sorry

theorem number_of_solutions : ∃! x : ℝ, |x - 5| = |x + 3| := by
  use 1
  split
  { -- First part: showing that x = 1 is a solution
    exact (fun h : 1 = 1 => by 
      rwa sub_self,
    sorry)
  },
  { -- Second part: showing that x = 1 is the only solution
    assume x hx,
    rw [hx],
    sorry  
  }

end distinct_solution_count_number_of_solutions_l278_278024


namespace a2016_value_l278_278373

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 1 - (1 / a n)

theorem a2016_value : ∃ a : ℕ → ℚ, seq a ∧ a 2016 = 1 / 3 :=
by
  sorry

end a2016_value_l278_278373


namespace point_quadrant_I_or_IV_l278_278184

def is_point_on_line (x y : ℝ) : Prop := 4 * x + 3 * y = 12
def is_equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

def point_in_quadrant_I (x y : ℝ) : Prop := (x > 0 ∧ y > 0)
def point_in_quadrant_IV (x y : ℝ) : Prop := (x > 0 ∧ y < 0)

theorem point_quadrant_I_or_IV (x y : ℝ) 
  (h1 : is_point_on_line x y) 
  (h2 : is_equidistant_from_axes x y) :
  point_in_quadrant_I x y ∨ point_in_quadrant_IV x y :=
sorry

end point_quadrant_I_or_IV_l278_278184


namespace car_gasoline_tank_capacity_l278_278452

theorem car_gasoline_tank_capacity
    (speed : ℝ)
    (usage_rate : ℝ)
    (travel_time : ℝ)
    (fraction_used : ℝ)
    (tank_capacity : ℝ)
    (gallons_used : ℝ)
    (distance_traveled : ℝ) :
  speed = 50 →
  usage_rate = 1 / 30 →
  travel_time = 5 →
  fraction_used = 0.5555555555555556 →
  distance_traveled = speed * travel_time →
  gallons_used = distance_traveled * usage_rate →
  gallon_used = tank_capacity * fraction_used →
  tank_capacity = 15 :=
by
  intros hs hr ht hf hd hu hf
  sorry

end car_gasoline_tank_capacity_l278_278452


namespace total_pages_in_book_l278_278528

theorem total_pages_in_book (pages_monday pages_tuesday total_pages_read total_pages_book : ℝ)
    (h1 : pages_monday = 15.5)
    (h2 : pages_tuesday = 1.5 * pages_monday + 16)
    (h3 : total_pages_read = pages_monday + pages_tuesday)
    (h4 : total_pages_book = 2 * total_pages_read) :
    total_pages_book = 109.5 :=
by
  sorry

end total_pages_in_book_l278_278528


namespace students_taking_both_languages_l278_278045

theorem students_taking_both_languages (total_students students_neither students_french students_german : ℕ) (h1 : total_students = 69)
  (h2 : students_neither = 15) (h3 : students_french = 41) (h4 : students_german = 22) :
  (students_french + students_german - (total_students - students_neither) = 9) :=
by
  sorry

end students_taking_both_languages_l278_278045


namespace intersection_M_N_l278_278661

-- Definitions of sets M and N
def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Proof statement showing the intersection of M and N
theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l278_278661


namespace paint_can_distribution_l278_278133

-- Definitions based on conditions provided in the problem.
def ratio_red := 3
def ratio_white := 2
def ratio_blue := 1
def total_paint := 60
def ratio_sum := ratio_red + ratio_white + ratio_blue

-- Definition of the problem to be proved.
theorem paint_can_distribution :
  (ratio_red * total_paint) / ratio_sum = 30 ∧
  (ratio_white * total_paint) / ratio_sum = 20 ∧
  (ratio_blue * total_paint) / ratio_sum = 10 := 
by
  sorry

end paint_can_distribution_l278_278133


namespace gift_wrapping_combinations_l278_278131

theorem gift_wrapping_combinations 
  (wrapping_varieties : ℕ)
  (ribbon_colors : ℕ)
  (gift_card_types : ℕ)
  (H_wrapping_varieties : wrapping_varieties = 8)
  (H_ribbon_colors : ribbon_colors = 3)
  (H_gift_card_types : gift_card_types = 4) : 
  wrapping_varieties * ribbon_colors * gift_card_types = 96 := 
by
  sorry

end gift_wrapping_combinations_l278_278131


namespace negation_of_universal_proposition_l278_278014

theorem negation_of_universal_proposition {f : ℝ → ℝ} :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by
  sorry

end negation_of_universal_proposition_l278_278014


namespace prime_expression_integer_value_l278_278471

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_expression_integer_value (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ n, (p * q + p^p + q^q) % (p + q) = 0 → n = 3 :=
by
  sorry

end prime_expression_integer_value_l278_278471


namespace perimeter_of_square_l278_278083

theorem perimeter_of_square (s : ℝ) (h : s^2 = s * Real.sqrt 2) (h_ne_zero : s ≠ 0) :
    4 * s = 4 * Real.sqrt 2 := by
  sorry

end perimeter_of_square_l278_278083


namespace geometric_sequence_general_term_l278_278625

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2:ℝ)^(n-1)

theorem geometric_sequence_general_term : 
  ∀ (n : ℕ), 
  (∀ (n : ℕ), 0 < a_n n) ∧ a_n 1 = 1 ∧ (a_n 1 + a_n 2 + a_n 3 = 7) → 
  a_n n = 2^(n-1) :=
by
  sorry

end geometric_sequence_general_term_l278_278625


namespace lines_parallel_l278_278494

def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_parallel : ∀ x1 x2 : ℝ, l1 x1 = l2 x2 → false := 
by
  intros x1 x2 h
  rw [l1, l2] at h
  sorry

end lines_parallel_l278_278494


namespace range_of_m_exacts_two_integers_l278_278043

theorem range_of_m_exacts_two_integers (m : ℝ) :
  (∀ x : ℝ, (x - 2) / 4 < (x - 1) / 3 ∧ 2 * x - m ≤ 2 - x) ↔ -2 ≤ m ∧ m < 1 := 
sorry

end range_of_m_exacts_two_integers_l278_278043


namespace arcsin_one_eq_pi_div_two_l278_278965

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l278_278965


namespace supplement_of_angle_with_given_complement_l278_278039

theorem supplement_of_angle_with_given_complement (θ : ℝ) (h : 90 - θ = 50) : 180 - θ = 140 :=
by sorry

end supplement_of_angle_with_given_complement_l278_278039


namespace simplify_fraction_l278_278398

noncomputable def simplify_expression (x : ℂ) : Prop :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) =
  (x - 3) / (x^2 - 6*x + 8)

theorem simplify_fraction (x : ℂ) : simplify_expression x :=
by
  sorry

end simplify_fraction_l278_278398


namespace average_speed_l278_278574

theorem average_speed (v1 v2 : ℝ) (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0) : 
  2 / (1 / v1 + 1 / v2) = 2 * v1 * v2 / (v1 + v2) :=
by sorry

end average_speed_l278_278574


namespace target_runs_correct_l278_278201

noncomputable def target_runs (run_rate1 : ℝ) (ovs1 : ℕ) (run_rate2 : ℝ) (ovs2 : ℕ) : ℝ :=
  (run_rate1 * ovs1) + (run_rate2 * ovs2)

theorem target_runs_correct : target_runs 4.5 12 8.052631578947368 38 = 360 :=
by
  sorry

end target_runs_correct_l278_278201


namespace smallest_num_rectangles_l278_278728

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l278_278728


namespace triple_overlap_area_correct_l278_278100

-- Define the dimensions of the auditorium and carpets
def auditorium_dim : ℕ × ℕ := (10, 10)
def carpet1_dim : ℕ × ℕ := (6, 8)
def carpet2_dim : ℕ × ℕ := (6, 6)
def carpet3_dim : ℕ × ℕ := (5, 7)

-- The coordinates and dimensions of the overlap regions are derived based on the given positions
-- Here we assume derivations as described in the solution steps without recalculating them

-- Overlap area of the second and third carpets
def overlap23 : ℕ × ℕ := (5, 3)

-- Intersection of this overlap with the first carpet
def overlap_all : ℕ × ℕ := (2, 3)

-- Calculate the area of the region where all three carpets overlap
def triple_overlap_area : ℕ :=
  (overlap_all.1 * overlap_all.2)

theorem triple_overlap_area_correct :
  triple_overlap_area = 6 := by
  -- Expected result should be 6 square meters
  sorry

end triple_overlap_area_correct_l278_278100


namespace John_new_weekly_earnings_l278_278508

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end John_new_weekly_earnings_l278_278508


namespace ram_birthday_l278_278468

theorem ram_birthday
    (L : ℕ) (L1 : ℕ) (Llast : ℕ) (d : ℕ) (languages_learned_per_day : ℕ) (days_in_month : ℕ) :
    (L = 1000) →
    (L1 = 820) →
    (Llast = 1100) →
    (days_in_month = 28 ∨ days_in_month = 29 ∨ days_in_month = 30 ∨ days_in_month = 31) →
    (d = days_in_month - 1) →
    (languages_learned_per_day = (Llast - L1) / d) →
    ∃ n : ℕ, n = 19 :=
by
  intros hL hL1 hLlast hDays hm_d hLearned
  existsi 19
  sorry

end ram_birthday_l278_278468


namespace count_multiples_of_8_between_200_and_400_l278_278862

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278862


namespace mary_can_keep_warm_l278_278067

def sticks_from_chairs (n_c : ℕ) (c_1 : ℕ) : ℕ := n_c * c_1
def sticks_from_tables (n_t : ℕ) (t_1 : ℕ) : ℕ := n_t * t_1
def sticks_from_cabinets (n_cb : ℕ) (cb_1 : ℕ) : ℕ := n_cb * cb_1
def sticks_from_stools (n_s : ℕ) (s_1 : ℕ) : ℕ := n_s * s_1

def total_sticks (n_c n_t n_cb n_s c_1 t_1 cb_1 s_1 : ℕ) : ℕ :=
  sticks_from_chairs n_c c_1
  + sticks_from_tables n_t t_1 
  + sticks_from_cabinets n_cb cb_1 
  + sticks_from_stools n_s s_1

noncomputable def hours (total_sticks r : ℕ) : ℕ :=
  total_sticks / r

theorem mary_can_keep_warm (n_c n_t n_cb n_s : ℕ) (c_1 t_1 cb_1 s_1 r : ℕ) :
  n_c = 25 → n_t = 12 → n_cb = 5 → n_s = 8 → c_1 = 8 → t_1 = 12 → cb_1 = 16 → s_1 = 3 → r = 7 →
  hours (total_sticks n_c n_t n_cb n_s c_1 t_1 cb_1 s_1) r = 64 :=
by
  intros h_nc h_nt h_ncb h_ns h_c1 h_t1 h_cb1 h_s1 h_r
  sorry

end mary_can_keep_warm_l278_278067


namespace certain_number_l278_278433

theorem certain_number (x : ℝ) : 
  0.55 * x = (4/5 : ℝ) * 25 + 2 → 
  x = 40 :=
by
  sorry

end certain_number_l278_278433


namespace bus_stops_for_45_minutes_per_hour_l278_278117

-- Define the conditions
def speed_excluding_stoppages : ℝ := 48 -- in km/hr
def speed_including_stoppages : ℝ := 12 -- in km/hr

-- Define the statement to be proven
theorem bus_stops_for_45_minutes_per_hour :
  let speed_reduction := speed_excluding_stoppages - speed_including_stoppages
  let time_stopped : ℝ := (speed_reduction / speed_excluding_stoppages) * 60
  time_stopped = 45 :=
by
  sorry

end bus_stops_for_45_minutes_per_hour_l278_278117


namespace platform_length_l278_278563

/-- Mathematical proof problem:
The problem is to prove that given the train's length, time taken to cross a signal pole and 
time taken to cross a platform, the length of the platform is 525 meters.
-/
theorem platform_length 
    (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) (P : ℕ) 
    (h_train_length : train_length = 450) (h_time_pole : time_pole = 18) 
    (h_time_platform : time_platform = 39) (h_P : P = 525) : 
    P = 525 := 
  sorry

end platform_length_l278_278563


namespace best_play_wins_probability_l278_278756

/-- Define the conditions and parameters for the problem. -/
variables (n m : ℕ)
variables (C : ℕ → ℕ → ℕ) /- Binomial coefficient -/

/-- Define the probability calculation -/
def probability_best_play_wins : ℚ :=
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t)

/-- The theorem stating that the above calculation represents the probability of the best play winning -/
theorem best_play_wins_probability :
  probability_best_play_wins n m C =
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t) :=
  by
  sorry

end best_play_wins_probability_l278_278756


namespace kate_visits_cost_l278_278378

theorem kate_visits_cost (entrance_fee_first_year : Nat) (monthly_visits : Nat) (next_two_years_fee : Nat) (yearly_visits_next_two_years : Nat) (total_years : Nat) : 
  entrance_fee_first_year = 5 →
  monthly_visits = 12 →
  next_two_years_fee = 7 →
  yearly_visits_next_two_years = 4 →
  total_years = 3 →
  let first_year_cost := entrance_fee_first_year * monthly_visits in
  let subsequent_years_visits := (total_years - 1) * yearly_visits_next_two_years in
  let subsequent_years_cost := next_two_years_fee * subsequent_years_visits in
  let total_cost := first_year_cost + subsequent_years_cost in
  total_cost = 116 :=
begin
  intros h1 h2 h3 h4 h5,
  unfold first_year_cost subsequent_years_visits subsequent_years_cost total_cost,
  simp [h1, h2, h3, h4, h5],
  rfl,
end

end kate_visits_cost_l278_278378


namespace largest_not_sum_of_two_composites_l278_278818

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278818


namespace strings_completely_pass_each_other_l278_278416

-- Define the problem parameters
def d : ℝ := 30    -- distance between A and B in cm
def l1 : ℝ := 151  -- length of string A in cm
def l2 : ℝ := 187  -- length of string B in cm
def v1 : ℝ := 2    -- speed of string A in cm/s
def v2 : ℝ := 3    -- speed of string B in cm/s
def r1 : ℝ := 1    -- burn rate of string A in cm/s
def r2 : ℝ := 2    -- burn rate of string B in cm/s

-- The proof problem statement
theorem strings_completely_pass_each_other : ∀ (T : ℝ), T = 40 :=
by
  sorry

end strings_completely_pass_each_other_l278_278416


namespace donation_amount_l278_278538

theorem donation_amount 
  (total_needed : ℕ) (bronze_amount : ℕ) (silver_amount : ℕ) (raised_so_far : ℕ)
  (bronze_families : ℕ) (silver_families : ℕ) (other_family_donation : ℕ)
  (final_push_needed : ℕ) 
  (h1 : total_needed = 750) 
  (h2 : bronze_amount = 25)
  (h3 : silver_amount = 50)
  (h4 : bronze_families = 10)
  (h5 : silver_families = 7)
  (h6 : raised_so_far = 600)
  (h7 : final_push_needed = 50)
  (h8 : raised_so_far = bronze_families * bronze_amount + silver_families * silver_amount)
  (h9 : total_needed - raised_so_far - other_family_donation = final_push_needed) : 
  other_family_donation = 100 :=
by
  sorry

end donation_amount_l278_278538


namespace solve_abs_quadratic_l278_278400

theorem solve_abs_quadratic :
  ∃ x : ℝ, (|x - 3| + x^2 = 10) ∧ 
  (x = (-1 + Real.sqrt 53) / 2 ∨ x = (1 + Real.sqrt 29) / 2 ∨ x = (1 - Real.sqrt 29) / 2) :=
by sorry

end solve_abs_quadratic_l278_278400


namespace count_divisible_by_8_l278_278861

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278861


namespace dice_probability_is_correct_l278_278159

noncomputable def dice_probability : ℚ :=
  let total_outcomes := (6:ℚ) ^ 7
  let one_pair_no_three_of_a_kind := (6 * 21 * 120 : ℚ)
  let two_pairs_no_three_of_a_kind := (15 * 35 * 6 * 24 : ℚ)
  let three_pairs_one_different := (20 * 7 * 90 * 3 : ℚ)
  let successful_outcomes := one_pair_no_three_of_a_kind + two_pairs_no_three_of_a_kind + three_pairs_one_different
  successful_outcomes / total_outcomes

theorem dice_probability_is_correct : dice_probability = 6426 / 13997 := 
by sorry

end dice_probability_is_correct_l278_278159


namespace solve_for_x_l278_278324

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end solve_for_x_l278_278324


namespace largest_natural_number_not_sum_of_two_composites_l278_278814

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278814


namespace problem_value_expression_l278_278074

theorem problem_value_expression 
  (x y : ℝ)
  (h₁ : x + y = 4)
  (h₂ : x * y = -2) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := 
sorry

end problem_value_expression_l278_278074


namespace betty_watermelons_l278_278151

theorem betty_watermelons :
  ∃ b : ℕ, 
  (b + (b + 10) + (b + 20) + (b + 30) + (b + 40) = 200) ∧
  (b + 40 = 60) :=
by
  sorry

end betty_watermelons_l278_278151


namespace a_range_l278_278334

noncomputable def f (x a : ℝ) : ℝ := |2 * x - 1| + |x - 2 * a|

def valid_a_range (a : ℝ) : Prop :=
∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ 4

theorem a_range (a : ℝ) : valid_a_range a → (1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) := 
sorry

end a_range_l278_278334


namespace total_is_83_l278_278369

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end total_is_83_l278_278369


namespace minimum_green_sticks_l278_278068

def natasha_sticks (m n : ℕ) : ℕ :=
  if (m = 3 ∧ n = 3) then 5 else 0

theorem minimum_green_sticks (m n : ℕ) (grid : m = 3 ∧ n = 3) :
  natasha_sticks m n = 5 :=
by
  sorry

end minimum_green_sticks_l278_278068


namespace max_det_value_l278_278169

theorem max_det_value :
  ∃ θ : ℝ, 
    (1 * ((5 + Real.sin θ) * 9 - 6 * 8) 
     - 2 * (4 * 9 - 6 * (7 + Real.cos θ)) 
     + 3 * (4 * 8 - (5 + Real.sin θ) * (7 + Real.cos θ))) 
     = 93 :=
sorry

end max_det_value_l278_278169


namespace total_lives_l278_278261

theorem total_lives (initial_players additional_players lives_per_player : ℕ) (h1 : initial_players = 4) (h2 : additional_players = 5) (h3 : lives_per_player = 3) :
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l278_278261


namespace probability_all_from_same_tribe_l278_278255

-- Definitions based on the conditions of the problem
def total_people := 24
def tribe_count := 3
def people_per_tribe := 8
def quitters := 3

-- We assume each person has an equal chance of quitting and the quitters are chosen independently
-- The probability that all three people who quit belong to the same tribe

theorem probability_all_from_same_tribe :
  ((3 * (Nat.choose people_per_tribe quitters)) / (Nat.choose total_people quitters) : ℚ) = 1 / 12 := 
  by 
    sorry

end probability_all_from_same_tribe_l278_278255


namespace find_x_l278_278185

theorem find_x (x : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (x, 1)) :
  ((2 * a.fst - x, 2 * a.snd + 1) • b = 0) → x = -1 ∨ x = 3 :=
by
  sorry

end find_x_l278_278185


namespace expression_for_f_minimum_positive_period_of_f_range_of_f_l278_278051

noncomputable def f (x : ℝ) : ℝ :=
  let A := (2, 0) 
  let B := (0, 2)
  let C := (Real.cos (2 * x), Real.sin (2 * x))
  let AB := (B.1 - A.1, B.2 - A.2) 
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.fst * AC.fst + AB.snd * AC.snd 

theorem expression_for_f (x : ℝ) :
  f x = 2 * Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 4 :=
by sorry

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by sorry

theorem range_of_f (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) :
  2 < f x ∧ f x ≤ 4 + 2 * Real.sqrt 2 :=
by sorry

end expression_for_f_minimum_positive_period_of_f_range_of_f_l278_278051


namespace find_first_factor_of_LCM_l278_278082

-- Conditions
def HCF : ℕ := 23
def Y : ℕ := 14
def largest_number : ℕ := 322

-- Statement
theorem find_first_factor_of_LCM
  (A B : ℕ)
  (H : Nat.gcd A B = HCF)
  (max_num : max A B = largest_number)
  (lcm_eq : Nat.lcm A B = HCF * X * Y) :
  X = 23 :=
sorry

end find_first_factor_of_LCM_l278_278082


namespace carrie_hours_per_day_l278_278777

theorem carrie_hours_per_day (h : ℕ) 
  (worked_4_days : ∀ n, n = 4 * h) 
  (paid_per_hour : ℕ := 22)
  (cost_of_supplies : ℕ := 54)
  (profit : ℕ := 122) :
  88 * h - cost_of_supplies = profit → h = 2 := 
by 
  -- Assume problem conditions and solve
  sorry

end carrie_hours_per_day_l278_278777


namespace chosen_number_eq_l278_278442

-- Given a number x, if (x / 2) - 100 = 4, then x = 208.
theorem chosen_number_eq (x : ℝ) (h : (x / 2) - 100 = 4) : x = 208 := 
by
  sorry

end chosen_number_eq_l278_278442


namespace stuart_initially_had_20_l278_278145

variable (B T S : ℕ) -- Initial number of marbles for Betty, Tom, and Susan
variable (S_after : ℕ) -- Number of marbles Stuart has after receiving from Betty

-- Given conditions
axiom betty_initially : B = 150
axiom tom_initially : T = 30
axiom susan_initially : S = 20

axiom betty_to_tom : (0.20 : ℚ) * B = 30
axiom betty_to_susan : (0.10 : ℚ) * B = 15
axiom betty_to_stuart : (0.40 : ℚ) * B = 60
axiom stuart_after_receiving : S_after = 80

-- Theorem to prove Stuart initially had 20 marbles
theorem stuart_initially_had_20 : ∃ S_initial : ℕ, S_after - 60 = S_initial ∧ S_initial = 20 :=
by {
  sorry
}

end stuart_initially_had_20_l278_278145


namespace two_students_solve_all_problems_l278_278104

theorem two_students_solve_all_problems
    (students : Fin 15 → Fin 6 → Prop)
    (h : ∀ (p : Fin 6), (∃ (s1 s2 s3 s4 s5 s6 s7 s8 : Fin 15), 
          students s1 p ∧ students s2 p ∧ students s3 p ∧ students s4 p ∧ 
          students s5 p ∧ students s6 p ∧ students s7 p ∧ students s8 p)) :
    ∃ (s1 s2 : Fin 15), ∀ (p : Fin 6), students s1 p ∨ students s2 p := 
by
    sorry

end two_students_solve_all_problems_l278_278104


namespace player_a_all_opportunities_player_a_probability_distribution_math_expectation_l278_278684

-- Define success probability and failure probability
def success_probability : ℝ := 3 / 5
def failure_probability := 1 - success_probability

-- Define the possible scores and their probabilities
def pmf_x : Pmf ℝ :=
  Pmf.ofFinset {0, 50, 100, 150} (λ x,
    if x = 0 then failure_probability^2 
    else if x = 50 then success_probability * (failure_probability^2) + failure_probability * success_probability * failure_probability
    else if x = 100 then (3.choose 2) * (success_probability^2) * failure_probability
    else success_probability^3)

-- Lean statement to prove both conditions
theorem player_a_all_opportunities :
  (3:ℝ) * (1 - failure_probability^2) =  3 * (21/25) := by
  sorry

theorem player_a_probability_distribution_math_expectation:
  pmf_x.prob 0 = 4/25 ∧ pmf_x.prob 50 = 24/125 ∧ pmf_x.prob 100 = 54/125 ∧ pmf_x.prob 150 = 27/125
  ∧ pmf_x.exp = 85.2 := by
  sorry

end player_a_all_opportunities_player_a_probability_distribution_math_expectation_l278_278684


namespace simplify_polynomial_l278_278918

theorem simplify_polynomial : 
  (5 - 3 * x - 7 * x^2 + 3 + 12 * x - 9 * x^2 - 8 + 15 * x + 21 * x^2) = (5 * x^2 + 24 * x) :=
by 
  sorry

end simplify_polynomial_l278_278918


namespace peanuts_total_correct_l278_278641

def initial_peanuts : ℕ := 4
def added_peanuts : ℕ := 6
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_total_correct : total_peanuts = 10 := by
  sorry

end peanuts_total_correct_l278_278641


namespace solve_quadratic_l278_278526

theorem solve_quadratic : 
  ∀ x : ℝ, (x - 1) ^ 2 = 64 → (x = 9 ∨ x = -7) :=
by
  sorry

end solve_quadratic_l278_278526


namespace puppies_per_dog_l278_278960

/--
Chuck breeds dogs. He has 3 pregnant dogs.
They each give birth to some puppies. Each puppy needs 2 shots and each shot costs $5.
The total cost of the shots is $120. Prove that each pregnant dog gives birth to 4 puppies.
-/
theorem puppies_per_dog :
  let num_dogs := 3
  let cost_per_shot := 5
  let shots_per_puppy := 2
  let total_cost := 120
  let cost_per_puppy := shots_per_puppy * cost_per_shot
  let total_puppies := total_cost / cost_per_puppy
  (total_puppies / num_dogs) = 4 := by
  sorry

end puppies_per_dog_l278_278960


namespace part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l278_278938

-- Part (1)
theorem part1_inequality (m : ℝ) : (∀ x : ℝ, (m^2 + 1)*x^2 - (2*m - 1)*x + 1 > 0) ↔ m > -3/4 := sorry

-- Part (2)
theorem part2_inequality_case1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ x < 1 ∨ x > 1/a) := sorry

theorem part2_inequality_case2 : 
  (∀ x : ℝ, (x - 1)*(0*x - 1) > 0 ↔ x < 1) := sorry

theorem part2_inequality_case3 (a : ℝ) (h : a < 0) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ 1/a < x ∧ x < 1) := sorry

end part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l278_278938


namespace inequality_AM_GM_HM_l278_278008

theorem inequality_AM_GM_HM (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * (a * b) / (a + b) :=
by
  sorry

end inequality_AM_GM_HM_l278_278008


namespace largest_non_summable_composite_l278_278808

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278808


namespace remaining_sausage_meat_l278_278675

-- Define the conditions
def total_meat_pounds : ℕ := 10
def sausage_links : ℕ := 40
def links_eaten_by_Brandy : ℕ := 12
def pounds_to_ounces : ℕ := 16

-- Calculate the remaining sausage meat and prove the correctness
theorem remaining_sausage_meat :
  (total_meat_pounds * pounds_to_ounces - links_eaten_by_Brandy * (total_meat_pounds * pounds_to_ounces / sausage_links)) = 112 :=
by
  sorry

end remaining_sausage_meat_l278_278675


namespace trig_identity_l278_278427

theorem trig_identity : 
  Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + 
  Real.cos (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 :=
by 
  sorry

end trig_identity_l278_278427


namespace largest_m_is_795_l278_278094

noncomputable def largest_possible_m : ℕ :=
  let prime_less_than_10 := [2, 3, 5, 7] in
  let all_combinations := [(2, 3), (5, 3), (7, 3), (3, 7)] in
  all_combinations.foldr (λ (p : ℕ × ℕ) acc,
    let (x, y) := p,
    let candidate := x * y * (10 * x + y) in
    if candidate < 1000 ∧ candidate > acc then candidate else acc) 0

theorem largest_m_is_795 : largest_possible_m = 795 := by
  sorry

end largest_m_is_795_l278_278094


namespace students_selected_milk_l278_278593

noncomputable def selected_soda_percent : ℚ := 50 / 100
noncomputable def selected_milk_percent : ℚ := 30 / 100
noncomputable def selected_soda_count : ℕ := 90
noncomputable def selected_milk_count := selected_milk_percent / selected_soda_percent * selected_soda_count

theorem students_selected_milk :
    selected_milk_count = 54 :=
by
  sorry

end students_selected_milk_l278_278593


namespace walkway_time_against_direction_l278_278575

theorem walkway_time_against_direction (v_p v_w t : ℝ) (h1 : 90 = (v_p + v_w) * 30)
  (h2 : v_p * 48 = 90) 
  (h3 : 90 = (v_p - v_w) * t) :
  t = 120 := by 
  sorry

end walkway_time_against_direction_l278_278575


namespace max_x2_plus_2xy_plus_3y2_l278_278242

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end max_x2_plus_2xy_plus_3y2_l278_278242


namespace largest_non_sum_of_composites_l278_278786

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278786


namespace average_price_mixed_sugar_l278_278586

def average_selling_price_per_kg (weightA weightB weightC costA costB costC : ℕ) := 
  (costA * weightA + costB * weightB + costC * weightC) / (weightA + weightB + weightC : ℚ)

theorem average_price_mixed_sugar : 
  average_selling_price_per_kg 3 2 5 28 20 12 = 18.4 := 
by
  sorry

end average_price_mixed_sugar_l278_278586


namespace correct_forecast_interpretation_l278_278092

/-- The probability of precipitation in the area tomorrow is 80%. -/
def prob_precipitation_tomorrow : ℝ := 0.8

/-- Multiple choice options regarding the interpretation of the probability of precipitation. -/
inductive forecast_interpretation
| A : forecast_interpretation
| B : forecast_interpretation
| C : forecast_interpretation
| D : forecast_interpretation

/-- The correct interpretation is Option C: "There is an 80% chance of rain in the area tomorrow." -/
def correct_interpretation : forecast_interpretation :=
forecast_interpretation.C

theorem correct_forecast_interpretation :
  (prob_precipitation_tomorrow = 0.8) → (correct_interpretation = forecast_interpretation.C) :=
by
  sorry

end correct_forecast_interpretation_l278_278092


namespace cone_base_diameter_l278_278946

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end cone_base_diameter_l278_278946


namespace total_spent_by_mrs_hilt_l278_278391

-- Define the cost per set of tickets for kids.
def cost_per_set_kids : ℕ := 1
-- Define the number of tickets in a set for kids.
def tickets_per_set_kids : ℕ := 4

-- Define the cost per set of tickets for adults.
def cost_per_set_adults : ℕ := 2
-- Define the number of tickets in a set for adults.
def tickets_per_set_adults : ℕ := 3

-- Define the total number of kids' tickets purchased.
def total_kids_tickets : ℕ := 12
-- Define the total number of adults' tickets purchased.
def total_adults_tickets : ℕ := 9

-- Prove that the total amount spent by Mrs. Hilt is $9.
theorem total_spent_by_mrs_hilt :
  (total_kids_tickets / tickets_per_set_kids * cost_per_set_kids) + 
  (total_adults_tickets / tickets_per_set_adults * cost_per_set_adults) = 9 :=
by sorry

end total_spent_by_mrs_hilt_l278_278391


namespace find_x4_l278_278706

theorem find_x4 (x_1 x_2 : ℝ) (h1 : 0 < x_1) (h2 : x_1 < x_2) 
  (P : (ℝ × ℝ)) (Q : (ℝ × ℝ)) (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (R : (ℝ × ℝ)) (x_4 : ℝ) :
  R = ((x_1 + x_2) / 2, (Real.log x_1 + Real.log x_2) / 2) →
  Real.log x_4 = (Real.log x_1 + Real.log x_2) / 2 →
  x_4 = Real.sqrt 1000 :=
by 
  intro hR hT
  sorry

end find_x4_l278_278706


namespace problem_statement_l278_278053

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem problem_statement : same_terminal_side (-510) 210 :=
by
  sorry

end problem_statement_l278_278053


namespace total_amount_l278_278388

def mark_dollars : ℚ := 5 / 8
def carolyn_dollars : ℚ := 2 / 5
def total_dollars : ℚ := mark_dollars + carolyn_dollars

theorem total_amount : total_dollars = 1.025 := by
  sorry

end total_amount_l278_278388


namespace shift_parabola_upwards_l278_278247

theorem shift_parabola_upwards (y x : ℝ) (h : y = x^2) : y + 5 = (x^2 + 5) := by 
  sorry

end shift_parabola_upwards_l278_278247


namespace rowing_distance_l278_278259

theorem rowing_distance (v_b : ℝ) (v_s : ℝ) (t_total : ℝ) (D : ℝ) :
  v_b = 9 → v_s = 1.5 → t_total = 48 → D / (v_b + v_s) + D / (v_b - v_s) = t_total → D = 210 :=
by
  intros
  sorry

end rowing_distance_l278_278259


namespace solve_for_a_l278_278193

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solve_for_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x / ((x + 1) * (x - a)))
  (h_odd : is_odd_function f) :
  a = 1 :=
sorry

end solve_for_a_l278_278193


namespace find_f2_l278_278990

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l278_278990


namespace cost_of_jeans_l278_278431

    variable (J S : ℝ)

    def condition1 := 3 * J + 6 * S = 104.25
    def condition2 := 4 * J + 5 * S = 112.15

    theorem cost_of_jeans (h1 : condition1 J S) (h2 : condition2 J S) : J = 16.85 := by
      sorry
    
end cost_of_jeans_l278_278431


namespace abs_has_min_at_zero_l278_278850

def f (x : ℝ) : ℝ := abs x

theorem abs_has_min_at_zero : ∃ m, (∀ x : ℝ, f x ≥ m) ∧ f 0 = m := by
  sorry

end abs_has_min_at_zero_l278_278850


namespace part1_part2_l278_278908

-- Definitions for part 1
def total_souvenirs := 60
def price_a := 100
def price_b := 60
def total_cost_1 := 4600

-- Definitions for part 2
def max_total_cost := 4500
def twice (m : ℕ) := 2 * m

theorem part1 (x y : ℕ) (hx : x + y = total_souvenirs) (hc : price_a * x + price_b * y = total_cost_1) :
  x = 25 ∧ y = 35 :=
by
  -- You can provide the detailed proof here
  sorry

theorem part2 (m : ℕ) (hm1 : 20 ≤ m) (hm2 : m ≤ 22) (hc2 : price_a * m + price_b * (total_souvenirs - m) ≤ max_total_cost) :
  (m = 20 ∨ m = 21 ∨ m = 22) ∧ 
  ∃ W, W = min (40 * 20 + 3600) (min (40 * 21 + 3600) (40 * 22 + 3600)) ∧ W = 4400 :=
by
  -- You can provide the detailed proof here
  sorry

end part1_part2_l278_278908


namespace lewis_speed_l278_278507

theorem lewis_speed
  (v : ℕ)
  (john_speed : ℕ := 40)
  (distance_AB : ℕ := 240)
  (meeting_distance : ℕ := 160)
  (time_john_to_meeting : ℕ := meeting_distance / john_speed)
  (distance_lewis_traveled : ℕ := distance_AB + (distance_AB - meeting_distance))
  (v_eq : v = distance_lewis_traveled / time_john_to_meeting) :
  v = 80 :=
by
  sorry

end lewis_speed_l278_278507


namespace circle_equation_l278_278175

theorem circle_equation 
  (circle_eq : ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = (x - 3)^2 + (y - 2)^2) 
  (tangent_to_line : ∀ (x y : ℝ), (2*x - y + 5) = 0 → 
    (x = -2 ∧ y = 1))
  (passes_through_N : ∀ (x y : ℝ), (x = 3 ∧ y = 2)) :
  ∀ (x y : ℝ), x^2 + y^2 - 9*x + (9/2)*y - (55/2) = 0 := 
sorry

end circle_equation_l278_278175


namespace determine_a_l278_278189

theorem determine_a (a : ℝ) : (∃ b : ℝ, (3 * (x : ℝ))^2 - 2 * 3 * b * x + b^2 = 9 * x^2 - 27 * x + a) → a = 20.25 :=
by
  sorry

end determine_a_l278_278189


namespace f_at_10_l278_278183

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

-- Prove that f(10) = 756
theorem f_at_10 : f 10 = 756 := by
  sorry

end f_at_10_l278_278183


namespace password_combinations_l278_278232

theorem password_combinations (n r k : ℕ) (hn : n = 5) (hk_fact : k.factorial = 6) (hr : r = 20) : 
  ∃ (password : list char), 
    let combinations := (n.factorial / k.factorial) in 
    combinations = r := 
begin
  sorry
end

end password_combinations_l278_278232


namespace area_of_ADC_l278_278052

theorem area_of_ADC
  (BD DC : ℝ)
  (h_ratio : BD / DC = 2 / 3)
  (area_ABD : ℝ)
  (h_area_ABD : area_ABD = 30) :
  ∃ area_ADC, area_ADC = 45 :=
by {
  sorry
}

end area_of_ADC_l278_278052


namespace majority_vote_is_280_l278_278647

-- Definitions based on conditions from step (a)
def totalVotes : ℕ := 1400
def winningPercentage : ℝ := 0.60
def losingPercentage : ℝ := 0.40

-- Majority computation based on the winning and losing percentages
def majorityVotes : ℝ := totalVotes * winningPercentage - totalVotes * losingPercentage

-- Theorem statement
theorem majority_vote_is_280 : majorityVotes = 280 := by
  sorry

end majority_vote_is_280_l278_278647


namespace total_surface_area_correct_l278_278768

def surface_area_calculation (height_e height_f height_g : ℚ) : ℚ :=
  let top_bottom_area := 4
  let side_area := (height_e + height_f + height_g) * 2
  let front_back_area := 4
  top_bottom_area + side_area + front_back_area

theorem total_surface_area_correct :
  surface_area_calculation (5 / 8) (1 / 4) (9 / 8) = 12 := 
by
  sorry

end total_surface_area_correct_l278_278768


namespace exists_prime_with_composite_sequence_l278_278516

theorem exists_prime_with_composite_sequence (n : ℕ) (hn : n ≠ 0) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬ Nat.Prime (p + k) :=
sorry

end exists_prime_with_composite_sequence_l278_278516


namespace total_is_83_l278_278368

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end total_is_83_l278_278368


namespace hearts_per_card_l278_278548

-- Definitions of the given conditions
def num_suits := 4
def num_cards_total := 52
def num_cards_per_suit := num_cards_total / num_suits
def cost_per_cow := 200
def total_cost := 83200
def num_cows := total_cost / cost_per_cow

-- The mathematical proof problem translated to Lean 4:
theorem hearts_per_card :
    (2 * (num_cards_total / num_suits) = num_cows) → (num_cows = 416) → (num_cards_total / num_suits = 208) :=
by
  intros h1 h2
  sorry

end hearts_per_card_l278_278548


namespace dog_tail_length_l278_278206

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end dog_tail_length_l278_278206


namespace largest_cannot_be_sum_of_two_composites_l278_278792

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278792


namespace units_digit_of_42_pow_3_add_24_pow_3_l278_278268

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end units_digit_of_42_pow_3_add_24_pow_3_l278_278268


namespace total_cartons_packed_l278_278301

-- Define the given conditions
def cans_per_carton : ℕ := 20
def cartons_loaded : ℕ := 40
def cans_left : ℕ := 200

-- Formalize the proof problem
theorem total_cartons_packed : cartons_loaded + (cans_left / cans_per_carton) = 50 := by
  sorry

end total_cartons_packed_l278_278301


namespace ratio_y_x_l278_278636

variable {c x y : ℝ}

-- Conditions stated as assumptions
theorem ratio_y_x (h1 : x = 0.80 * c) (h2 : y = 1.25 * c) : y / x = 25 / 16 :=
by
  sorry

end ratio_y_x_l278_278636


namespace digit_sum_subtraction_l278_278248

theorem digit_sum_subtraction (P Q R S : ℕ) (hQ : Q + P = P) (hP : Q - P = 0) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10) (h4 : S < 10) : S = 0 := by
  sorry

end digit_sum_subtraction_l278_278248


namespace minimum_value_expression_l278_278835

theorem minimum_value_expression {a : ℝ} (h₀ : 1 < a) (h₁ : a < 4) : 
  (∃ m : ℝ, (∀ x : ℝ, 1 < x ∧ x < 4 → m ≤ (x / (4 - x) + 1 / (x - 1))) ∧ m = 2) :=
sorry

end minimum_value_expression_l278_278835


namespace sqrt_six_plus_s_cubed_l278_278062

theorem sqrt_six_plus_s_cubed (s : ℝ) : 
    Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) :=
sorry

end sqrt_six_plus_s_cubed_l278_278062


namespace price_of_70_cans_l278_278097

noncomputable def discounted_price (regular_price : ℝ) (discount_percent : ℝ) : ℝ :=
  regular_price * (1 - discount_percent / 100)

noncomputable def total_price (regular_price : ℝ) (discount_percent : ℝ) (total_cans : ℕ) (cans_per_case : ℕ) : ℝ :=
  let price_per_can := discounted_price regular_price discount_percent
  let full_cases := total_cans / cans_per_case
  let remaining_cans := total_cans % cans_per_case
  full_cases * cans_per_case * price_per_can + remaining_cans * price_per_can

theorem price_of_70_cans :
  total_price 0.55 25 70 24 = 28.875 :=
by
  sorry

end price_of_70_cans_l278_278097


namespace people_own_only_cats_and_dogs_l278_278547

-- Define the given conditions
def total_people : ℕ := 59
def only_dogs : ℕ := 15
def only_cats : ℕ := 10
def cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 29

-- Define the proof problem
theorem people_own_only_cats_and_dogs : ∃ x : ℕ, 15 + 10 + x + 3 + (29 - 3) = 59 ∧ x = 5 :=
by {
  sorry
}

end people_own_only_cats_and_dogs_l278_278547


namespace range_of_a_iff_condition_l278_278502

theorem range_of_a_iff_condition (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3 * a) ↔ (a ≥ -2 ∧ a ≤ 5) :=
by
  sorry

end range_of_a_iff_condition_l278_278502


namespace total_rankings_l278_278646

-- Defines the set of players
inductive Player
| P : Player
| Q : Player
| R : Player
| S : Player

-- Defines a function to count the total number of ranking sequences
def total_possible_rankings (p : Player → Player → Prop) : Nat := 
  4 * 2 * 2

-- Problem statement
theorem total_rankings : ∃ t : Player → Player → Prop, total_possible_rankings t = 16 :=
by
  sorry

end total_rankings_l278_278646


namespace pagoda_lights_l278_278907

/-- From afar, the magnificent pagoda has seven layers, with red lights doubling on each
ascending floor, totaling 381 lights. How many lights are there at the very top? -/
theorem pagoda_lights :
  ∃ x, (1 + 2 + 4 + 8 + 16 + 32 + 64) * x = 381 ∧ x = 3 :=
by
  sorry

end pagoda_lights_l278_278907


namespace express_set_M_l278_278607

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def M : Set ℤ := {m | is_divisor 10 (m + 1)}

theorem express_set_M :
  M = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by
  sorry

end express_set_M_l278_278607


namespace highest_avg_speed_2_to_3_l278_278520

-- Define the time periods and distances traveled in those periods
def distance_8_to_9 : ℕ := 50
def distance_9_to_10 : ℕ := 70
def distance_10_to_11 : ℕ := 60
def distance_2_to_3 : ℕ := 80
def distance_3_to_4 : ℕ := 40

-- Define the average speed calculation for each period
def avg_speed (distance : ℕ) (hours : ℕ) : ℕ := distance / hours

-- Proposition stating that the highest average speed is from 2 pm to 3 pm
theorem highest_avg_speed_2_to_3 : 
  avg_speed distance_2_to_3 1 > avg_speed distance_8_to_9 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_9_to_10 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_10_to_11 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_3_to_4 1 := 
by 
  sorry

end highest_avg_speed_2_to_3_l278_278520


namespace person_picking_number_who_announced_6_is_1_l278_278683

theorem person_picking_number_who_announced_6_is_1
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ)
  (h₁ : a₁₀ + a₂ = 2)
  (h₂ : a₁ + a₃ = 4)
  (h₃ : a₂ + a₄ = 6)
  (h₄ : a₃ + a₅ = 8)
  (h₅ : a₄ + a₆ = 10)
  (h₆ : a₅ + a₇ = 12)
  (h₇ : a₆ + a₈ = 14)
  (h₈ : a₇ + a₉ = 16)
  (h₉ : a₈ + a₁₀ = 18)
  (h₁₀ : a₉ + a₁ = 20) :
  a₆ = 1 :=
by
  sorry

end person_picking_number_who_announced_6_is_1_l278_278683


namespace total_surface_area_of_cylinder_l278_278294

-- Define radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 12

-- Theorem stating the total surface area of the cylinder
theorem total_surface_area_of_cylinder : 2 * real.pi * radius * (radius + height) = 170 * real.pi := by
  sorry

end total_surface_area_of_cylinder_l278_278294


namespace total_noodles_and_pirates_l278_278371

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end total_noodles_and_pirates_l278_278371


namespace expected_worth_is_1_33_l278_278146

noncomputable def expected_worth_of_coin_flip : ℝ :=
  let prob_heads := 2 / 3
  let profit_heads := 5
  let prob_tails := 1 / 3
  let loss_tails := -6
  (prob_heads * profit_heads + prob_tails * loss_tails)

theorem expected_worth_is_1_33 : expected_worth_of_coin_flip = 1.33 := by
  sorry

end expected_worth_is_1_33_l278_278146


namespace real_solutions_l278_278164

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end real_solutions_l278_278164


namespace binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l278_278459

def binary_to_decimal (b : ℕ) : ℕ :=
  32 + 0 + 8 + 4 + 2 + 1 -- Calculated manually for simplicity

def decimal_to_octal (d : ℕ) : ℕ :=
  (5 * 10) + 7 -- Manually converting decimal 47 to octal 57 for simplicity

theorem binary_101111_to_decimal_is_47 : binary_to_decimal 0b101111 = 47 := 
by sorry

theorem decimal_47_to_octal_is_57 : decimal_to_octal 47 = 57 := 
by sorry

end binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l278_278459


namespace ratio_of_four_numbers_exists_l278_278172

theorem ratio_of_four_numbers_exists (A B C D : ℕ) (h1 : A + B + C + D = 1344) (h2 : D = 672) : 
  ∃ rA rB rC rD, rA ≠ 0 ∧ rB ≠ 0 ∧ rC ≠ 0 ∧ rD ≠ 0 ∧ A = rA * k ∧ B = rB * k ∧ C = rC * k ∧ D = rD * k :=
by {
  sorry
}

end ratio_of_four_numbers_exists_l278_278172


namespace min_value_f_l278_278630

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x^2 - m * x

theorem min_value_f (m : ℝ) (h_m : 0 < m) :
  let f := λ x, x * Real.exp x - (m / 2) * x^2 - m * x in
  let I := Set.Icc (1 : ℝ) 2 in
  f ∈ measurable (Set.IntervalIntegrable I) ∧
  (∀ x ∈ I, f x) ∈ lower_bounds ((λ x, x * Real.exp x - (m / 2) * x^2 - m * x) '' I) 
    ∈ {e - (3/2)*m, - (1/2)*m*Real.log m^2, 2*Real.exp 2^2 - 4*m} :=
begin
  sorry
end

end min_value_f_l278_278630


namespace translate_parabola_l278_278264

theorem translate_parabola (x : ℝ) :
  (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ ∀ x: ℝ, y = 2*x^2 → y = 2*(x - h)^2 + k) := 
by
  use 1, 3
  sorry

end translate_parabola_l278_278264


namespace valid_paths_in_grid_with_forbidden_segments_l278_278948

theorem valid_paths_in_grid_with_forbidden_segments :
  let totalPaths := Nat.choose 14 4
  let forbiddenPaths1 := Nat.choose 4 1 * Nat.choose 9 3
  let forbiddenPaths2 := Nat.choose 8 2 * Nat.choose 5 2
  let forbiddenPaths := forbiddenPaths1 + forbiddenPaths2
  let validPaths := totalPaths - forbiddenPaths
  validPaths = 385 :=
by
  sorry

end valid_paths_in_grid_with_forbidden_segments_l278_278948


namespace yvon_combination_l278_278275

theorem yvon_combination :
  let num_notebooks := 4
  let num_pens := 5
  num_notebooks * num_pens = 20 :=
by
  sorry

end yvon_combination_l278_278275


namespace no_solution_exists_l278_278769

theorem no_solution_exists : ¬ ∃ (x : ℕ), (42 + x = 3 * (8 + x) ∧ 42 + x = 2 * (10 + x)) :=
by
  sorry

end no_solution_exists_l278_278769


namespace certain_number_exists_l278_278914

theorem certain_number_exists
  (N : ℕ) 
  (hN : ∀ x, x < N → x % 2 = 1 → ∃ k m, k = 5 * m ∧ x = k ∧ m % 2 = 1) :
  N = 76 := by
  sorry

end certain_number_exists_l278_278914


namespace restaurant_meals_l278_278592

theorem restaurant_meals (k a : ℕ) (ratio_kids_to_adults : k / a = 10 / 7) (kids_meals_sold : k = 70) : a = 49 :=
by
  sorry

end restaurant_meals_l278_278592


namespace distinct_solutions_abs_eq_l278_278020

theorem distinct_solutions_abs_eq (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 :=
by
  -- The proof is omitted
  sorry

end distinct_solutions_abs_eq_l278_278020


namespace solve_logarithmic_system_l278_278854

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_logarithmic_system :
  ∃ x y : ℝ, log_base 2 x + log_base 4 y = 4 ∧ log_base 4 x + log_base 2 y = 5 ∧ x = 4 ∧ y = 16 :=
by
  sorry

end solve_logarithmic_system_l278_278854


namespace customers_in_each_car_l278_278543

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end customers_in_each_car_l278_278543


namespace arcsin_one_eq_pi_div_two_l278_278962

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end arcsin_one_eq_pi_div_two_l278_278962


namespace rectangular_region_area_l278_278136

theorem rectangular_region_area :
  ∀ (s : ℝ), 18 * s * s = (15 * Real.sqrt 2) * (7.5 * Real.sqrt 2) :=
by
  intro s
  have h := 5 ^ 2 = 2 * s ^ 2
  have s := Real.sqrt (25 / 2)
  exact sorry

end rectangular_region_area_l278_278136


namespace songs_per_album_correct_l278_278236

-- Define the number of albums and total number of songs as conditions
def number_of_albums : ℕ := 8
def total_songs : ℕ := 16

-- Define the number of songs per album
def songs_per_album (albums : ℕ) (songs : ℕ) : ℕ := songs / albums

-- The main theorem stating that the number of songs per album is 2
theorem songs_per_album_correct :
  songs_per_album number_of_albums total_songs = 2 :=
by
  unfold songs_per_album
  sorry

end songs_per_album_correct_l278_278236


namespace exponent_property_l278_278351

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end exponent_property_l278_278351


namespace determine_V_300_l278_278612

-- Definitions related to the arithmetic sequence
def arithmetic_sequence {α : Type*} [LinearOrderedField α] (b r : α) (n : ℕ) : α := b + (n - 1) * r

-- Definition of U_n
def U_n {α : Type*} [LinearOrderedField α] (b r : α) (n : ℕ) : α := (n / 2) * (2 * b + (n - 1) * r)

-- Definition of V_n
def V_n {α : Type*} [LinearOrderedField α] (b r : α) (n : ℕ) : α :=
  ∑ i in finset.range (n + 1), U_n b r i

-- Theorem stating the relationship between U_150 and V_300
theorem determine_V_300 {α : Type*} [LinearOrderedField α] (b r : α) (U150 : α) :
  U_150 = 150 * (b + 74.5 * r) →
  ∃ V300, V300 = 25 * 301 * (450 + 224.5 * r) :=
begin
  -- Proof is not required
  sorry,
end

end determine_V_300_l278_278612


namespace num_common_points_of_three_lines_l278_278262

def three_planes {P : Type} [AddCommGroup P] (l1 l2 l3 : Set P) : Prop :=
  let p12 := Set.univ \ (l1 ∪ l2)
  let p13 := Set.univ \ (l1 ∪ l3)
  let p23 := Set.univ \ (l2 ∪ l3)
  ∃ (pl12 pl13 pl23 : Set P), 
    p12 = pl12 ∧ p13 = pl13 ∧ p23 = pl23

theorem num_common_points_of_three_lines (l1 l2 l3 : Set ℝ) 
  (h : three_planes l1 l2 l3) : ∃ n : ℕ, n = 0 ∨ n = 1 := by
  sorry

end num_common_points_of_three_lines_l278_278262


namespace customers_in_each_car_l278_278544

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end customers_in_each_car_l278_278544


namespace largest_not_sum_of_two_composites_l278_278820

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278820


namespace find_number_l278_278124

theorem find_number 
  (x : ℝ)
  (h : (258 / 100 * x) / 6 = 543.95) :
  x = 1265 :=
sorry

end find_number_l278_278124


namespace roger_trips_required_l278_278746

variable (carry_trays_per_trip total_trays : ℕ)

theorem roger_trips_required (h1 : carry_trays_per_trip = 4) (h2 : total_trays = 12) : total_trays / carry_trays_per_trip = 3 :=
by
  -- proof follows
  sorry

end roger_trips_required_l278_278746


namespace find_number_l278_278036

theorem find_number (x : ℝ) (h : 0.4 * x = 15) : x = 37.5 := by
  sorry

end find_number_l278_278036


namespace determine_xyz_l278_278180

-- Define the conditions for the variables x, y, and z
variables (x y z : ℝ)

-- State the problem as a theorem
theorem determine_xyz :
  (x + y + z) * (x * y + x * z + y * z) = 24 ∧
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8 →
  x * y * z = 16 / 3 :=
by
  intros h
  sorry

end determine_xyz_l278_278180


namespace sum_a6_a7_a8_is_32_l278_278219

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l278_278219


namespace midpoint_trajectory_l278_278987

theorem midpoint_trajectory (x y p q : ℝ) (h_parabola : p^2 = 4 * q)
  (h_focus : ∀ (p q : ℝ), p^2 = 4 * q → q = (p/2)^2) 
  (h_midpoint_x : x = (p + 1) / 2)
  (h_midpoint_y : y = q / 2):
  y^2 = 2 * x - 1 :=
by
  sorry

end midpoint_trajectory_l278_278987


namespace probability_selecting_both_types_X_distribution_correct_E_X_correct_l278_278975

section DragonBoatFestival

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The total number of red bean zongzi -/
def red_bean_zongzi : ℕ := 2

/-- The total number of plain zongzi -/
def plain_zongzi : ℕ := 8

/-- The number of zongzi to select -/
def zongzi_to_select : ℕ := 3

/-- Probability of selecting at least one red bean zongzi and at least one plain zongzi -/
def probability_selecting_both : ℚ := 8 / 15

/-- Distribution of the number of red bean zongzi selected (X) -/
def X_distribution : ℕ → ℚ
| 0 => 7 / 15
| 1 => 7 / 15
| 2 => 1 / 15
| _ => 0

/-- Mathematical expectation of the number of red bean zongzi selected (E(X)) -/
def E_X : ℚ := 3 / 5

/-- Theorem stating the probability of selecting both types of zongzi -/
theorem probability_selecting_both_types :
  let p := probability_selecting_both
  p = 8 / 15 :=
by
  let p := probability_selecting_both
  sorry

/-- Theorem stating the probability distribution of the number of red bean zongzi selected -/
theorem X_distribution_correct :
  (X_distribution 0 = 7 / 15) ∧
  (X_distribution 1 = 7 / 15) ∧
  (X_distribution 2 = 1 / 15) :=
by
  sorry

/-- Theorem stating the mathematical expectation of the number of red bean zongzi selected -/
theorem E_X_correct :
  let E := E_X
  E = 3 / 5 :=
by
  let E := E_X
  sorry

end DragonBoatFestival

end probability_selecting_both_types_X_distribution_correct_E_X_correct_l278_278975


namespace log_ab_a2_plus_log_ab_b2_eq_2_l278_278865

theorem log_ab_a2_plus_log_ab_b2_eq_2 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h_distinct : a ≠ b) (h_a_gt_2 : a > 2) (h_b_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 :=
by
  sorry

end log_ab_a2_plus_log_ab_b2_eq_2_l278_278865


namespace evaluate_expression_l278_278467

theorem evaluate_expression : 2 * (2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3) = 24240542 :=
by
  let a := 2009
  let b := 2010
  sorry

end evaluate_expression_l278_278467


namespace prob_exactly_one_hits_prob_at_least_one_hits_l278_278071

noncomputable def prob_A_hits : ℝ := 1 / 2
noncomputable def prob_B_hits : ℝ := 1 / 3
noncomputable def prob_A_misses : ℝ := 1 - prob_A_hits
noncomputable def prob_B_misses : ℝ := 1 - prob_B_hits

theorem prob_exactly_one_hits :
  (prob_A_hits * prob_B_misses) + (prob_A_misses * prob_B_hits) = 1 / 2 :=
by sorry

theorem prob_at_least_one_hits :
  1 - (prob_A_misses * prob_B_misses) = 2 / 3 :=
by sorry

end prob_exactly_one_hits_prob_at_least_one_hits_l278_278071


namespace work_completion_days_l278_278126

theorem work_completion_days (A B C : ℕ) (work_rate_A : A = 4) (work_rate_B : B = 10) (work_rate_C : C = 20 / 3) :
  (1 / A) + (1 / B) + (3 / C) = 1 / 2 :=
by
  sorry

end work_completion_days_l278_278126


namespace CD_expression_l278_278054

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C A1 B1 C1 D : V)
variables (a b c : V)

-- Given conditions
axiom AB_eq_a : A - B = a
axiom AC_eq_b : A - C = b
axiom AA1_eq_c : A - A1 = c
axiom midpoint_D : D = (1/2) • (B1 + C1)

-- We need to show
theorem CD_expression : C - D = (1/2) • a - (1/2) • b + c :=
sorry

end CD_expression_l278_278054


namespace find_y_intercept_l278_278154

theorem find_y_intercept (m : ℝ) 
  (h1 : ∀ x y : ℝ, y = 2 * x + m)
  (h2 : ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = 2 * x + m) : 
  m = -1 := 
sorry

end find_y_intercept_l278_278154


namespace train_crossing_platform_l278_278140

/-- Given a train crosses a 100 m platform in 15 seconds, and the length of the train is 350 m,
    prove that the train takes 20 seconds to cross a second platform of length 250 m. -/
theorem train_crossing_platform (dist1 dist2 l_t t1 t2 : ℝ) (h1 : dist1 = 100) (h2 : dist2 = 250) (h3 : l_t = 350) (h4 : t1 = 15) :
  t2 = 20 :=
sorry

end train_crossing_platform_l278_278140


namespace largest_number_not_sum_of_two_composites_l278_278825

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278825


namespace problem_1_problem_2_l278_278153

def op (x y : ℝ) : ℝ := 3 * x - y

theorem problem_1 (x : ℝ) : op x (op 2 3) = 1 ↔ x = 4 / 3 := by
  -- definitions from conditions
  let def_op_2_3 := op 2 3
  let eq1 := op x def_op_2_3
  -- problem in lean representation
  sorry

theorem problem_2 (x : ℝ) : op (x ^ 2) 2 = 10 ↔ x = 2 ∨ x = -2 := by
  -- problem in lean representation
  sorry

end problem_1_problem_2_l278_278153


namespace alice_paid_percentage_l278_278251

theorem alice_paid_percentage {P : ℝ} (hP : P > 0)
  (hMP : ∀ P, MP = 0.60 * P)
  (hPrice_Alice_Paid : ∀ MP, Price_Alice_Paid = 0.40 * MP) :
  (Price_Alice_Paid / P) * 100 = 24 := by
  sorry

end alice_paid_percentage_l278_278251


namespace necessary_but_not_sufficient_condition_l278_278332

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l278_278332


namespace height_of_spheres_l278_278418

theorem height_of_spheres (R r : ℝ) (h : ℝ) :
  0 < r ∧ r < R → h = R - Real.sqrt ((3 * R^2 - 6 * R * r - r^2) / 3) :=
by
  intros h0
  sorry

end height_of_spheres_l278_278418


namespace committee_probability_l278_278699

def num_boys : ℕ := 10
def num_girls : ℕ := 15
def num_total : ℕ := 25
def committee_size : ℕ := 5

def num_ways_total : ℕ := Nat.choose num_total committee_size
def num_ways_boys_only : ℕ := Nat.choose num_boys committee_size
def num_ways_girls_only : ℕ := Nat.choose num_girls committee_size

def probability_boys_or_girls_only : ℚ :=
  (num_ways_boys_only + num_ways_girls_only) / num_ways_total

def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - probability_boys_or_girls_only

theorem committee_probability :
  probability_at_least_one_boy_and_one_girl = 475 / 506 :=
sorry

end committee_probability_l278_278699


namespace a_minus_b_l278_278692

noncomputable def find_a_b (a b : ℝ) :=
  ∃ k : ℝ, ∀ (x : ℝ) (y : ℝ), 
    (k = 2 + a) ∧ 
    (y = k * x + 1) ∧ 
    (y = x^2 + a * x + b) ∧ 
    (x = 1) ∧ (y = 3)

theorem a_minus_b (a b : ℝ) (h : find_a_b a b) : a - b = -2 := by 
  sorry

end a_minus_b_l278_278692


namespace trains_meet_distance_l278_278116

noncomputable def time_difference : ℝ :=
  5 -- Time difference between two departures in hours

noncomputable def speed_train_a : ℝ :=
  30 -- Speed of Train A in km/h

noncomputable def speed_train_b : ℝ :=
  40 -- Speed of Train B in km/h

noncomputable def distance_train_a : ℝ :=
  speed_train_a * time_difference -- Distance covered by Train A before Train B starts

noncomputable def relative_speed : ℝ :=
  speed_train_b - speed_train_a -- Relative speed of Train B with respect to Train A

noncomputable def catch_up_time : ℝ :=
  distance_train_a / relative_speed -- Time taken for Train B to catch up with Train A

noncomputable def distance_from_delhi : ℝ :=
  speed_train_b * catch_up_time -- Distance from Delhi where the two trains will meet

theorem trains_meet_distance :
  distance_from_delhi = 600 := by
  sorry

end trains_meet_distance_l278_278116


namespace train_length_l278_278588

def train_speed_kmph := 25 -- speed of train in km/h
def man_speed_kmph := 2 -- speed of man in km/h
def crossing_time_sec := 52 -- time to cross in seconds

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph -- relative speed in km/h
  let relative_speed_mps := relative_speed_kmph * (5 / 18) -- convert to m/s
  relative_speed_mps * crossing_time_sec -- length of train in meters

theorem train_length : length_of_train = 390 :=
  by sorry -- proof omitted

end train_length_l278_278588


namespace arcsin_one_eq_pi_div_two_l278_278963

noncomputable def arcsin (x : ℝ) : ℝ :=
classical.some (exists_inverse_sin x)

theorem arcsin_one_eq_pi_div_two : arcsin 1 = π / 2 :=
sorry

end arcsin_one_eq_pi_div_two_l278_278963


namespace sum_four_variables_l278_278221

theorem sum_four_variables 
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + 2 = x)
  (h2 : b + 3 = x)
  (h3 : c + 4 = x)
  (h4 : d + 5 = x)
  (h5 : a + b + c + d + 8 = x) :
  a + b + c + d = -6 :=
by
  sorry

end sum_four_variables_l278_278221


namespace part1_part2_l278_278429

-- Part (I)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, 3 * x - abs (-2 * x + 1) ≥ a ↔ 2 ≤ x) → a = 3 :=
by
  sorry

-- Part (II)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x - abs (x - a) ≤ 1)) → (a ≤ 1 ∨ 3 ≤ a) :=
by
  sorry

end part1_part2_l278_278429


namespace michael_earnings_l278_278230

-- Define variables for pay rates and hours.
def regular_pay_rate : ℝ := 7.00
def overtime_multiplier : ℝ := 2
def regular_hours : ℝ := 40
def overtime_hours (total_hours : ℝ) : ℝ := total_hours - regular_hours

-- Define the earnings functions.
def regular_earnings (hourly_rate : ℝ) (hours : ℝ) : ℝ := hourly_rate * hours
def overtime_earnings (hourly_rate : ℝ) (multiplier : ℝ) (hours : ℝ) : ℝ := hourly_rate * multiplier * hours

-- Total earnings calculation.
def total_earnings (total_hours : ℝ) : ℝ := 
regular_earnings regular_pay_rate regular_hours + 
overtime_earnings regular_pay_rate overtime_multiplier (overtime_hours total_hours)

-- The theorem to prove the correct earnings for 42.857142857142854 hours worked.
theorem michael_earnings : total_earnings 42.857142857142854 = 320 := by
  sorry

end michael_earnings_l278_278230


namespace sum_a6_a7_a8_is_32_l278_278217

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l278_278217


namespace intersection_A_B_complement_A_in_U_complement_B_in_U_l278_278227

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {5, 6, 7, 8}
def B : Set ℕ := {2, 4, 6, 8}

-- Problems to prove
theorem intersection_A_B : A ∩ B = {6, 8} := by
  sorry

theorem complement_A_in_U : U \ A = {1, 2, 3, 4} := by
  sorry

theorem complement_B_in_U : U \ B = {1, 3, 5, 7} := by
  sorry

end intersection_A_B_complement_A_in_U_complement_B_in_U_l278_278227


namespace exactly_two_sports_l278_278365

variables {U : Type} {A B C : Finset U}

def badminton_and_tennis (A B : Finset U) : ℕ := (A ∩ B).card
def badminton_and_soccer (A C : Finset U) : ℕ := (A ∩ C).card
def tennis_and_soccer (B C : Finset U) : ℕ := (B ∩ C).card

theorem exactly_two_sports (badminton tennis soccer : Finset U) 
  (total_members : ℕ) (badminton_card : ℕ) (tennis_card : ℕ) (soccer_card : ℕ) 
  (not_playing_any : ℕ) (badminton_tennis : ℕ) (badminton_soccer : ℕ) (tennis_soccer : ℕ) 
  (no_three_sports : (badminton ∩ tennis ∩ soccer).card = 0)
  (htotal_members : total_members = 60)
  (hbadminton_card : badminton_card = 25)
  (htennis_card : tennis_card = 32)
  (hsoccer_card : soccer_card = 14)
  (hnot_playing_any : not_playing_any = 5)
  (hbadminton_tennis : badminton_tennis = 10)
  (hbadminton_soccer : badminton_soccer = 8)
  (htennis_soccer : tennis_soccer = 6)
  : badminton_and_tennis badminton tennis + badminton_and_soccer badminton soccer + tennis_and_soccer tennis soccer = 24 :=
by
  rw [badminton_and_tennis, badminton_and_soccer, tennis_and_soccer]
  simp
  sorry

end exactly_two_sports_l278_278365


namespace complement_of_A_l278_278853

def A : Set ℝ := { x | x^2 - x ≥ 0 }
def R_complement_A : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem complement_of_A :
  ∀ x : ℝ, x ∈ R_complement_A ↔ x ∉ A :=
sorry

end complement_of_A_l278_278853


namespace cover_square_with_rectangles_l278_278717

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l278_278717


namespace find_constant_l278_278904

theorem find_constant
  (k : ℝ)
  (r : ℝ := 36)
  (C : ℝ := 72 * k)
  (h1 : C = 2 * Real.pi * r)
  : k = Real.pi := by
  sorry

end find_constant_l278_278904


namespace determinant_of_A_is_one_l278_278976

-- Define the matrix
def A (α β γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β * Real.cos γ, -Real.sin α],
    ![-Real.sin β * Real.cos γ, Real.cos β, Real.sin γ],
    ![Real.sin α * Real.cos β, Real.sin α * Real.sin β * Real.cos γ, Real.cos α]
  ]

-- Define the theorem to prove that the determinant is 1
theorem determinant_of_A_is_one (α β γ : ℝ) : Matrix.det (A α β γ) = 1 := 
by 
  sorry

end determinant_of_A_is_one_l278_278976


namespace smallest_number_of_rectangles_l278_278715

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l278_278715


namespace customers_in_each_car_l278_278542

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end customers_in_each_car_l278_278542


namespace positive_multiples_of_6_l278_278421

theorem positive_multiples_of_6 (k a b : ℕ) (h₁ : a = (3 + 3 * k))
  (h₂ : b = 24) (h₃ : a^2 - b^2 = 0) : k = 7 :=
sorry

end positive_multiples_of_6_l278_278421


namespace find_f2_l278_278991

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l278_278991


namespace force_exerted_by_pulley_on_axis_l278_278143

-- Define the basic parameters given in the problem
def m1 : ℕ := 3 -- mass 1 in kg
def m2 : ℕ := 6 -- mass 2 in kg
def g : ℕ := 10 -- acceleration due to gravity in m/s^2

-- From the problem, we know that:
def F1 : ℕ := m1 * g -- gravitational force on mass 1
def F2 : ℕ := m2 * g -- gravitational force on mass 2

-- To find the tension, setup the equations
def a := (F2 - F1) / (m1 + m2) -- solving for acceleration between the masses

def T := (m1 * a) + F1 -- solving for the tension in the rope considering mass 1

-- Define the proof statement to find the force exerted by the pulley on its axis
theorem force_exerted_by_pulley_on_axis : 2 * T = 80 :=
by
  -- Annotations or calculations can go here
  sorry

end force_exerted_by_pulley_on_axis_l278_278143


namespace floor_length_is_twelve_l278_278445

-- Definitions based on the conditions
def floor_width := 10
def strip_width := 3
def rug_area := 24

-- Problem statement
theorem floor_length_is_twelve (L : ℕ) 
  (h1 : rug_area = (L - 2 * strip_width) * (floor_width - 2 * strip_width)) :
  L = 12 := 
sorry

end floor_length_is_twelve_l278_278445


namespace toads_per_acre_l278_278686

theorem toads_per_acre (b g : ℕ) (h₁ : b = 25 * g)
  (h₂ : b / 4 = 50) : g = 8 :=
by
  -- Condition h₁: For every green toad, there are 25 brown toads.
  -- Condition h₂: One-quarter of the brown toads are spotted, and there are 50 spotted brown toads per acre.
  sorry

end toads_per_acre_l278_278686


namespace gamma_donuts_received_l278_278461

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end gamma_donuts_received_l278_278461


namespace Chemistry_marks_l278_278309

theorem Chemistry_marks (english_marks mathematics_marks physics_marks biology_marks : ℕ) (avg_marks : ℝ) (num_subjects : ℕ) (total_marks : ℕ)
  (h1 : english_marks = 72)
  (h2 : mathematics_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : avg_marks = 62.6)
  (h6 : num_subjects = 5)
  (h7 : total_marks = avg_marks * num_subjects) :
  (total_marks - (english_marks + mathematics_marks + physics_marks + biology_marks) = 62) :=
by
  sorry

end Chemistry_marks_l278_278309


namespace first_group_size_l278_278594

theorem first_group_size
  (x : ℕ)
  (h1 : 2 * x + 22 + 16 + 14 = 68) : 
  x = 8 :=
by
  sorry

end first_group_size_l278_278594


namespace lyndee_friends_count_l278_278390

-- Definitions
variables (total_chicken total_garlic_bread : ℕ)
variables (lyndee_chicken lyndee_garlic_bread : ℕ)
variables (friends_large_chicken_count : ℕ)
variables (friends_large_chicken : ℕ)
variables (friend_garlic_bread_per_friend : ℕ)

def remaining_chicken (total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken : ℕ) : ℕ :=
  total_chicken - (lyndee_chicken + friends_large_chicken_count * friends_large_chicken)

def remaining_garlic_bread (total_garlic_bread lyndee_garlic_bread : ℕ) : ℕ :=
  total_garlic_bread - lyndee_garlic_bread

def total_friends (friends_large_chicken_count remaining_chicken remaining_garlic_bread friend_garlic_bread_per_friend : ℕ) : ℕ :=
  friends_large_chicken_count + remaining_chicken + remaining_garlic_bread / friend_garlic_bread_per_friend

-- Theorem statement
theorem lyndee_friends_count : 
  total_chicken = 11 → 
  total_garlic_bread = 15 →
  lyndee_chicken = 1 →
  lyndee_garlic_bread = 1 →
  friends_large_chicken_count = 3 →
  friends_large_chicken = 2 →
  friend_garlic_bread_per_friend = 3 →
  total_friends friends_large_chicken_count 
                (remaining_chicken total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken)
                (remaining_garlic_bread total_garlic_bread lyndee_garlic_bread)
                friend_garlic_bread_per_friend = 7 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof omitted
  sorry

end lyndee_friends_count_l278_278390


namespace sum_of_angles_of_inscribed_quadrilateral_l278_278127

/--
Given a quadrilateral EFGH inscribed in a circle, and the measures of ∠EGH = 50° and ∠GFE = 70°,
then the sum of the angles ∠EFG + ∠EHG is 60°.
-/
theorem sum_of_angles_of_inscribed_quadrilateral
  (E F G H : Type)
  (circumscribed : True) -- This is just a place holder for the circle condition
  (angle_EGH : ℝ) (angle_GFE : ℝ)
  (h1 : angle_EGH = 50)
  (h2 : angle_GFE = 70) :
  ∃ (angle_EFG angle_EHG : ℝ), angle_EFG + angle_EHG = 60 := sorry

end sum_of_angles_of_inscribed_quadrilateral_l278_278127


namespace correct_final_positions_l278_278176

noncomputable def shapes_after_rotation (initial_positions : (String × String) × (String × String) × (String × String)) : (String × String) × (String × String) × (String × String) :=
  match initial_positions with
  | (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) =>
    (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left"))
  | _ => initial_positions

theorem correct_final_positions :
  shapes_after_rotation (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) = (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left")) :=
by
  unfold shapes_after_rotation
  rfl

end correct_final_positions_l278_278176


namespace qu_arrangements_l278_278524

theorem qu_arrangements :
  ∃ n : ℕ, n = 480 ∧ 
    ∀ (s : Finset Char), s = {'e', 'q', 'u', 'a', 't', 'i', 'o', 'n'} → 
    ∃ (t : Finset (Finset Char)), t.card = 1 ∧ t ⊆ s →
    ∃ u : Finset Char, u ⊆ (s \ {'q', 'u'}) ∧ u.card = 3 →
    (factorial 4) * (Finset.choose (s \ {'q', 'u'}) 3).card = n :=
by
  sorry

end qu_arrangements_l278_278524


namespace find_a6_plus_a7_plus_a8_l278_278216

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l278_278216


namespace largest_non_representable_as_sum_of_composites_l278_278801

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278801


namespace ratio_of_boys_to_girls_l278_278645

-- Definitions based on the initial conditions
def G : ℕ := 135
def T : ℕ := 351

-- Noncomputable because it involves division which is not always computable
noncomputable def B : ℕ := T - G

-- Main theorem to prove the ratio
theorem ratio_of_boys_to_girls : (B : ℚ) / G = 8 / 5 :=
by
  -- Here would be the proof, skipped with sorry.
  sorry

end ratio_of_boys_to_girls_l278_278645


namespace apples_fallen_l278_278002

theorem apples_fallen (H1 : ∃ ground_apples : ℕ, ground_apples = 10 + 3)
                      (H2 : ∃ tree_apples : ℕ, tree_apples = 5)
                      (H3 : ∃ total_apples : ℕ, total_apples = ground_apples ∧ total_apples = 10 + 3 + 5)
                      : ∃ fallen_apples : ℕ, fallen_apples = 13 :=
by
  sorry

end apples_fallen_l278_278002


namespace total_carrots_l278_278188

/-- 
  If Pleasant Goat and Beautiful Goat each receive 6 carrots, and the other goats each receive 3 carrots, there will be 6 carrots left over.
  If Pleasant Goat and Beautiful Goat each receive 7 carrots, and the other goats each receive 5 carrots, there will be a shortage of 14 carrots.
  Prove the total number of carrots (n) is 45. 
--/
theorem total_carrots (X n : ℕ) 
  (h1 : n = 3 * X + 18) 
  (h2 : n = 5 * X) : 
  n = 45 := 
by
  sorry

end total_carrots_l278_278188


namespace total_planks_l278_278596

-- Define the initial number of planks
def initial_planks : ℕ := 15

-- Define the planks Charlie got
def charlie_planks : ℕ := 10

-- Define the planks Charlie's father got
def father_planks : ℕ := 10

-- Prove the total number of planks
theorem total_planks : (initial_planks + charlie_planks + father_planks) = 35 :=
by sorry

end total_planks_l278_278596


namespace sector_central_angle_l278_278984

-- The conditions
def r : ℝ := 2
def S : ℝ := 4

-- The question
theorem sector_central_angle : ∃ α : ℝ, |α| = 2 ∧ S = 0.5 * α * r * r :=
by
  sorry

end sector_central_angle_l278_278984


namespace find_initial_music_files_l278_278707

-- Define the initial state before any deletion
def initial_files (music_files : ℕ) (video_files : ℕ) : ℕ := music_files + video_files

-- Define the state after deleting files
def files_after_deletion (initial_files : ℕ) (deleted_files : ℕ) : ℕ := initial_files - deleted_files

-- Theorem to prove that the initial number of music files was 13
theorem find_initial_music_files 
  (video_files : ℕ) (deleted_files : ℕ) (remaining_files : ℕ) 
  (h_videos : video_files = 30) (h_deleted : deleted_files = 10) (h_remaining : remaining_files = 33) : 
  ∃ (music_files : ℕ), initial_files music_files video_files - deleted_files = remaining_files ∧ music_files = 13 :=
by {
  sorry
}

end find_initial_music_files_l278_278707


namespace christine_makes_two_cakes_l278_278456

theorem christine_makes_two_cakes (tbsp_per_egg_white : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (total_tbsp_aquafaba : ℕ)
  (h1 : tbsp_per_egg_white = 2) 
  (h2 : egg_whites_per_cake = 8) 
  (h3 : total_tbsp_aquafaba = 32) : 
  total_tbsp_aquafaba / tbsp_per_egg_white / egg_whites_per_cake = 2 := by 
  sorry

end christine_makes_two_cakes_l278_278456


namespace smallest_circle_area_l278_278109

noncomputable def function_y (x : ℝ) : ℝ := 6 / x - 4 * x / 3

theorem smallest_circle_area :
  ∃ r : ℝ, (∀ x : ℝ, r * r = x^2 + (function_y x)^2) → r^2 * π = 4 * π :=
sorry

end smallest_circle_area_l278_278109


namespace seedling_probability_l278_278125

theorem seedling_probability (germination_rate survival_rate : ℝ)
    (h_germ : germination_rate = 0.9) (h_survival : survival_rate = 0.8) : 
    germination_rate * survival_rate = 0.72 :=
by
  rw [h_germ, h_survival]
  norm_num

end seedling_probability_l278_278125


namespace find_positives_xyz_l278_278317

theorem find_positives_xyz (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0)
    (heq : (1 : ℚ)/x + (1 : ℚ)/y + (1 : ℚ)/z = 4 / 5) :
    (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10) :=
by
  sorry

-- This theorem states that there are only two sets of positive integers (x, y, z)
-- that satisfy the equation (1/x) + (1/y) + (1/z) = 4/5, specifically:
-- (2, 4, 20) and (2, 5, 10).

end find_positives_xyz_l278_278317


namespace set_intersection_complement_l278_278493

open Set

variable (U P Q: Set ℕ)

theorem set_intersection_complement (hU: U = {1, 2, 3, 4}) (hP: P = {1, 2}) (hQ: Q = {2, 3}) :
  P ∩ (U \ Q) = {1} :=
by
  sorry

end set_intersection_complement_l278_278493


namespace cricket_scores_l278_278437

-- Define the conditions
variable (X : ℝ) (A B C D E average10 average6 : ℝ)
variable (matches10 matches6 : ℕ)

-- Set the given constants
axiom average_runs_10 : average10 = 38.9
axiom matches_10 : matches10 = 10
axiom average_runs_6 : average6 = 42
axiom matches_6 : matches6 = 6

-- Define the equations based on the conditions
axiom eq1 : X = average10 * matches10
axiom eq2 : A + B + C + D = X - (average6 * matches6)
axiom eq3 : E = (A + B + C + D) / 4

-- The target statement
theorem cricket_scores : X = 389 ∧ A + B + C + D = 137 ∧ E = 34.25 :=
  by
    sorry

end cricket_scores_l278_278437


namespace largest_percentage_increase_l278_278591

def student_count (year: ℕ) : ℝ :=
  match year with
  | 2010 => 80
  | 2011 => 88
  | 2012 => 95
  | 2013 => 100
  | 2014 => 105
  | 2015 => 112
  | _    => 0  -- Because we only care about 2010-2015

noncomputable def percentage_increase (year1 year2 : ℕ) : ℝ :=
  ((student_count year2 - student_count year1) / student_count year1) * 100

theorem largest_percentage_increase :
  (∀ x y, percentage_increase 2010 2011 ≥ percentage_increase x y) :=
by sorry

end largest_percentage_increase_l278_278591


namespace problem1_problem2_l278_278846

theorem problem1 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : 0 < t ∧ t < 1) :
  x^t - (x-1)^t < (x-2)^t - (x-3)^t :=
sorry

theorem problem2 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : t > 1) :
  x^t - (x-1)^t > (x-2)^t - (x-3)^t :=
sorry

end problem1_problem2_l278_278846


namespace func_value_sum_l278_278851

noncomputable def f (x : ℝ) : ℝ :=
  -x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2 + 1

theorem func_value_sum : f (1/2) + f (-1/2) = 2 :=
by
  sorry

end func_value_sum_l278_278851


namespace range_of_a_l278_278501

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a < -1 ∨ a > 3 :=
by
  sorry

end range_of_a_l278_278501


namespace points_per_member_l278_278141

def numMembersTotal := 12
def numMembersAbsent := 4
def totalPoints := 64

theorem points_per_member (h : numMembersTotal - numMembersAbsent = 12 - 4) :
  (totalPoints / (numMembersTotal - numMembersAbsent)) = 8 := 
  sorry

end points_per_member_l278_278141


namespace arcsin_one_eq_pi_div_two_l278_278966

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l278_278966


namespace mode_of_scores_is_85_l278_278700

-- Define the scores based on the given stem-and-leaf plot
def scores : List ℕ := [50, 55, 55, 62, 62, 68, 70, 71, 75, 79, 81, 81, 83, 85, 85, 85, 92, 96, 96, 98, 100, 100]

-- Define a function to compute the mode
def mode (s : List ℕ) : ℕ :=
  s.foldl (λ acc x => if s.count x > s.count acc then x else acc) 0

-- The theorem to prove that the mode of the scores is 85
theorem mode_of_scores_is_85 : mode scores = 85 :=
by
  -- The proof is omitted
  sorry

end mode_of_scores_is_85_l278_278700


namespace handshakes_mod_500_l278_278046

theorem handshakes_mod_500 : 
  let n := 10
  let k := 3
  let M := 199584 -- total number of ways calculated from the problem
  (n = 10) -> (k = 3) -> (M % 500 = 84) :=
by
  intros
  sorry

end handshakes_mod_500_l278_278046


namespace range_of_m_l278_278328

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = 3 * m + 1)
  (h2 : x + 2 * y = 3)
  (h3 : 2 * x - y < 1) : 
  m < 1 := 
sorry

end range_of_m_l278_278328


namespace cover_square_with_rectangles_l278_278720

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l278_278720


namespace proof_no_solution_l278_278772

noncomputable def no_solution (a b : ℕ) : Prop :=
  2 * a^2 + 1 ≠ 4 * b^2

theorem proof_no_solution (a b : ℕ) : no_solution a b := by
  sorry

end proof_no_solution_l278_278772


namespace color_5x5_grid_excluding_two_corners_l278_278105

-- Define the total number of ways to color a 5x5 grid with each row and column having exactly one colored cell
def total_ways : Nat := 120

-- Define the number of ways to color a 5x5 grid excluding one specific corner cell such that each row and each column has exactly one colored cell
def ways_excluding_one_corner : Nat := 96

-- Prove the number of ways to color the grid excluding two specific corner cells is 78
theorem color_5x5_grid_excluding_two_corners : total_ways - (ways_excluding_one_corner + ways_excluding_one_corner - 6) = 78 := by
  -- We state our given conditions directly as definitions
  -- Now we state our theorem explicitly and use the correct answer we derived
  sorry

end color_5x5_grid_excluding_two_corners_l278_278105


namespace John_new_weekly_earnings_l278_278509

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end John_new_weekly_earnings_l278_278509


namespace find_f_of_given_g_and_odd_l278_278989

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end find_f_of_given_g_and_odd_l278_278989


namespace fred_grew_38_cantaloupes_l278_278003

/-
  Fred grew some cantaloupes. Tim grew 44 cantaloupes.
  Together, they grew a total of 82 cantaloupes.
  Prove that Fred grew 38 cantaloupes.
-/

theorem fred_grew_38_cantaloupes (T F : ℕ) (h1 : T = 44) (h2 : T + F = 82) : F = 38 :=
by
  rw [h1] at h2
  linarith

end fred_grew_38_cantaloupes_l278_278003


namespace molly_age_l278_278231

theorem molly_age : 14 + 6 = 20 := by
  sorry

end molly_age_l278_278231


namespace problem1_problem2_l278_278680

-- Problem (1)
theorem problem1 (a : ℚ) (h : a = -1/2) : 
  a * (a - 4) - (a + 6) * (a - 2) = 16 := by
  sorry

-- Problem (2)
theorem problem2 (x y : ℚ) (hx : x = 8) (hy : y = -8) :
  (x + 2 * y) * (x - 2 * y) - (2 * x - y) * (-2 * x - y) = 0 := by
  sorry

end problem1_problem2_l278_278680


namespace population_difference_l278_278121

variable (A B C : ℝ)

-- Conditions
def population_condition (A B C : ℝ) : Prop := A + B = B + C + 5000

-- The proof statement
theorem population_difference (h : population_condition A B C) : A - C = 5000 :=
by sorry

end population_difference_l278_278121


namespace reduced_price_l278_278277

open Real

noncomputable def original_price : ℝ := 33.33

variables (P R: ℝ) (Q : ℝ)

theorem reduced_price
  (h1 : R = 0.75 * P)
  (h2 : P * 500 / P = 500)
  (h3 : 0.75 * P * (Q + 5) = 500)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  -- The proof will be provided here
  sorry

end reduced_price_l278_278277


namespace range_of_k_l278_278491

theorem range_of_k (x y k : ℝ) (h1 : 2 * x - 3 * y = 5) (h2 : 2 * x - y = k) (h3 : x > y) : k > -5 :=
sorry

end range_of_k_l278_278491


namespace average_cost_is_70_l278_278947

noncomputable def C_before_gratuity (total_bill : ℝ) (gratuity_rate : ℝ) : ℝ :=
  total_bill / (1 + gratuity_rate)

noncomputable def average_cost_per_individual (C : ℝ) (total_people : ℝ) : ℝ :=
  C / total_people

theorem average_cost_is_70 :
  let total_bill := 756
  let gratuity_rate := 0.20
  let total_people := 9
  average_cost_per_individual (C_before_gratuity total_bill gratuity_rate) total_people = 70 :=
by
  sorry

end average_cost_is_70_l278_278947


namespace length_of_first_train_l278_278417

theorem length_of_first_train
  (speed1_kmph : ℝ) (speed2_kmph : ℝ)
  (time_s : ℝ) (length2_m : ℝ)
  (relative_speed_mps : ℝ := (speed1_kmph + speed2_kmph) * 1000 / 3600)
  (total_distance_m : ℝ := relative_speed_mps * time_s)
  (length1_m : ℝ := total_distance_m - length2_m) :
  speed1_kmph = 80 →
  speed2_kmph = 65 →
  time_s = 7.199424046076314 →
  length2_m = 180 →
  length1_m = 110 :=
by
  sorry

end length_of_first_train_l278_278417


namespace evaluate_polynomial_at_2_l278_278605

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + x^2 + 2 * x + 3

theorem evaluate_polynomial_at_2 : polynomial 2 = 67 := by
  sorry

end evaluate_polynomial_at_2_l278_278605


namespace problem1_l278_278748

theorem problem1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a ^ 2 + 1 / a ^ 2 + 3) / (4 * a + 1 / (4 * a)) = 10 * Real.sqrt 5 := sorry

end problem1_l278_278748


namespace largest_cannot_be_sum_of_two_composites_l278_278796

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278796


namespace certain_number_value_l278_278498

theorem certain_number_value :
  ∃ n : ℚ, 9 - (4 / 6) = 7 + (n / 6) ∧ n = 8 := by
sorry

end certain_number_value_l278_278498


namespace find_n_l278_278495

theorem find_n (n : ℕ) (h : 2 * 2^2 * 2^n = 2^10) : n = 7 :=
sorry

end find_n_l278_278495


namespace positive_number_is_25_over_9_l278_278869

variable (a : ℚ) (x : ℚ)

theorem positive_number_is_25_over_9 
  (h1 : 2 * a - 1 = -a + 3)
  (h2 : ∃ r : ℚ, r^2 = x ∧ (r = 2 * a - 1 ∨ r = -a + 3)) : 
  x = 25 / 9 := 
by
  sorry

end positive_number_is_25_over_9_l278_278869


namespace Basel_series_l278_278149

theorem Basel_series :
  (∑' (n : ℕ+), 1 / (n : ℝ)^2) = π^2 / 6 := by sorry

end Basel_series_l278_278149


namespace solution_set_inequality_l278_278475

theorem solution_set_inequality (x : ℝ) (h : x - 3 / x > 2) :
    -1 < x ∧ x < 0 ∨ x > 3 :=
  sorry

end solution_set_inequality_l278_278475


namespace union_of_A_and_B_l278_278842

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
noncomputable def B : Set ℝ := {x : ℝ | 1 < x }

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x} :=
by
  sorry

end union_of_A_and_B_l278_278842


namespace incorrect_statement_d_l278_278114

variable (x : ℝ)
variables (p q : Prop)

-- Proving D is incorrect given defined conditions
theorem incorrect_statement_d :
  ∀ (x : ℝ), (¬ (x = 1) → ¬ (x^2 - 3 * x + 2 = 0)) ∧
  ((x > 2) → (x^2 - 3 * x + 2 > 0) ∧
  (¬ (x^2 + x + 1 = 0))) ∧
  ((p ∨ q) → ¬ (p ∧ q)) :=
by
  -- A detailed proof would be required here
  sorry

end incorrect_statement_d_l278_278114


namespace exponent_property_l278_278350

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end exponent_property_l278_278350


namespace part1_part2_l278_278479

variable (a b c x : ℝ)

-- Condition: lengths of the sides of the triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Quadratic equation
def quadratic_eq (x : ℝ) : ℝ := (a + c) * x^2 - 2 * b * x + (a - c)

-- Proof problem 1: If x = 1 is a root, then triangle ABC is isosceles
theorem part1 (h : quadratic_eq a b c 1 = 0) : a = b :=
by
  sorry

-- Proof problem 2: If triangle ABC is equilateral, then roots of the quadratic equation are 0 and 1
theorem part2 (h_eq : a = b ∧ b = c) :
  (quadratic_eq a a a 0 = 0) ∧ (quadratic_eq a a a 1 = 0) :=
by
  sorry

end part1_part2_l278_278479


namespace roots_quadratic_eq_l278_278843

theorem roots_quadratic_eq (α β : ℝ) (hαβ : ∀ x, x^2 + 2017*x + 1 = (x - α) * (x - β))
  : (1 + 2020 * α + α^2) * (1 + 2020 * β + β^2) = 9 :=
  sorry

end roots_quadratic_eq_l278_278843


namespace total_amount_is_24_l278_278584

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l278_278584


namespace half_radius_of_circle_y_l278_278742

theorem half_radius_of_circle_y
  (r_x r_y : ℝ)
  (hx : π * r_x ^ 2 = π * r_y ^ 2)
  (hc : 2 * π * r_x = 10 * π) :
  r_y / 2 = 2.5 :=
by
  sorry

end half_radius_of_circle_y_l278_278742


namespace find_f_neg_a_l278_278941

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

-- Given condition
axiom h_fa : f a = 11

-- Statement to prove
theorem find_f_neg_a : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l278_278941


namespace find_R_l278_278639

theorem find_R (R : ℝ) (h_diff : ∃ a b : ℝ, a ≠ b ∧ (a - b = 12 ∨ b - a = 12) ∧ a + b = 2 ∧ a * b = -R) : R = 35 :=
by
  obtain ⟨a, b, h_neq, h_diff_12, h_sum, h_prod⟩ := h_diff
  sorry

end find_R_l278_278639


namespace quincy_sold_more_than_jake_l278_278877

variables (T : ℕ) (Jake Quincy : ℕ)

def thors_sales (T : ℕ) := T
def jakes_sales (T : ℕ) := T + 10
def quincys_sales (T : ℕ) := 10 * T

theorem quincy_sold_more_than_jake (h1 : jakes_sales T = Jake) 
  (h2 : quincys_sales T = Quincy) (h3 : Quincy = 200) : 
  Quincy - Jake = 170 :=
by
  sorry

end quincy_sold_more_than_jake_l278_278877


namespace least_sum_of_bases_l278_278406

theorem least_sum_of_bases (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : 4 * a + 7 = 7 * b + 4) (h4 : 4 * a + 3 % 7 = 0) :
  a + b = 24 :=
sorry

end least_sum_of_bases_l278_278406


namespace largest_lcm_value_l278_278266

-- Define the conditions as local constants 
def lcm_18_3 : ℕ := Nat.lcm 18 3
def lcm_18_6 : ℕ := Nat.lcm 18 6
def lcm_18_9 : ℕ := Nat.lcm 18 9
def lcm_18_15 : ℕ := Nat.lcm 18 15
def lcm_18_21 : ℕ := Nat.lcm 18 21
def lcm_18_27 : ℕ := Nat.lcm 18 27

-- Statement to prove
theorem largest_lcm_value : max lcm_18_3 (max lcm_18_6 (max lcm_18_9 (max lcm_18_15 (max lcm_18_21 lcm_18_27)))) = 126 :=
by
  -- We assume the necessary calculations have been made
  have h1 : lcm_18_3 = 18 := by sorry
  have h2 : lcm_18_6 = 18 := by sorry
  have h3 : lcm_18_9 = 18 := by sorry
  have h4 : lcm_18_15 = 90 := by sorry
  have h5 : lcm_18_21 = 126 := by sorry
  have h6 : lcm_18_27 = 54 := by sorry

  -- Using above results to determine the maximum
  exact (by rw [h1, h2, h3, h4, h5, h6]; exact rfl)

end largest_lcm_value_l278_278266


namespace isosceles_triangle_angles_sum_l278_278415

theorem isosceles_triangle_angles_sum (x : ℝ) 
  (h_triangle_sum : ∀ a b c : ℝ, a + b + c = 180)
  (h_isosceles : ∃ a b : ℝ, (a = 50 ∧ b = x) ∨ (a = x ∧ b = 50)) :
  50 + x + (180 - 50 * 2) + 65 + 80 = 195 :=
by
  sorry

end isosceles_triangle_angles_sum_l278_278415


namespace smallest_num_rectangles_l278_278730

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l278_278730


namespace trig_identity_proof_l278_278276

theorem trig_identity_proof (α : ℝ) :
  sin^2 (π / 4 + α) - sin^2 (π / 6 - α) - sin (π / 12) * cos (π / 12 + 2 * α) = sin (2 * α) :=
by sorry

end trig_identity_proof_l278_278276


namespace negation_of_implication_l278_278912

-- Definitions based on the conditions from part (a)
def original_prop (x : ℝ) : Prop := x > 5 → x > 0
def negation_candidate_A (x : ℝ) : Prop := x ≤ 5 → x ≤ 0

-- The goal is to prove that the negation of the original proposition
-- is equivalent to option A, that is:
theorem negation_of_implication (x : ℝ) : (¬ (x > 5 → x > 0)) = (x ≤ 5 → x ≤ 0) :=
by
  sorry

end negation_of_implication_l278_278912


namespace nine_digit_positive_integers_l278_278186

theorem nine_digit_positive_integers :
  (∃ n : Nat, 10^8 * 9 = n ∧ n = 900000000) :=
sorry

end nine_digit_positive_integers_l278_278186


namespace divisors_form_60k_l278_278782

-- Define the conditions in Lean
def is_positive_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def satisfies_conditions (n a b c : ℕ) : Prop :=
  is_positive_divisor n a ∧
  is_positive_divisor n b ∧
  is_positive_divisor n c ∧
  a > b ∧ b > c ∧
  is_positive_divisor n (a^2 - b^2) ∧
  is_positive_divisor n (b^2 - c^2) ∧
  is_positive_divisor n (a^2 - c^2)

-- State the theorem to be proven in Lean
theorem divisors_form_60k (n : ℕ) (a b c : ℕ) (h1 : satisfies_conditions n a b c) : 
  ∃ k : ℕ, n = 60 * k :=
sorry

end divisors_form_60k_l278_278782


namespace least_number_to_add_l278_278270

theorem least_number_to_add (x : ℕ) (h : 53 ∣ x ∧ 71 ∣ x) : 
  ∃ n : ℕ, x = 1357 + n ∧ n = 2406 :=
by sorry

end least_number_to_add_l278_278270


namespace smallest_number_of_rectangles_l278_278714

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l278_278714


namespace rectangle_lengths_l278_278358

theorem rectangle_lengths (side_length : ℝ) (width1 width2: ℝ) (length1 length2 : ℝ) 
  (h1 : side_length = 6) 
  (h2 : width1 = 4) 
  (h3 : width2 = 3)
  (h_area_square : side_length * side_length = 36)
  (h_area_rectangle1 : width1 * length1 = side_length * side_length)
  (h_area_rectangle2 : width2 * length2 = (1 / 2) * (side_length * side_length)) :
  length1 = 9 ∧ length2 = 6 :=
by
  sorry

end rectangle_lengths_l278_278358


namespace unique_non_zero_in_rows_and_cols_l278_278512

variable (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)

theorem unique_non_zero_in_rows_and_cols
  (non_neg_A : ∀ i j, 0 ≤ A i j)
  (non_sing_A : Invertible A)
  (non_neg_A_inv : ∀ i j, 0 ≤ (A⁻¹) i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end unique_non_zero_in_rows_and_cols_l278_278512


namespace unique_square_friendly_l278_278770

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k^2 = n

def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18 * m + c)

theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 := 
sorry

end unique_square_friendly_l278_278770


namespace age_difference_l278_278440

variable (S M : ℕ)

theorem age_difference (hS : S = 28) (hM : M + 2 = 2 * (S + 2)) : M - S = 30 :=
by
  sorry

end age_difference_l278_278440


namespace g_at_50_l278_278535

variable (g : ℝ → ℝ)

axiom g_functional_eq (x y : ℝ) : g (x * y) = x * g y
axiom g_at_1 : g 1 = 40

theorem g_at_50 : g 50 = 2000 :=
by
  -- Placeholder for proof
  sorry

end g_at_50_l278_278535


namespace tory_sold_to_neighbor_l278_278830

def total_cookies : ℕ := 50
def sold_to_grandmother : ℕ := 12
def sold_to_uncle : ℕ := 7
def to_be_sold : ℕ := 26

def sold_to_neighbor : ℕ :=
  total_cookies - to_be_sold - (sold_to_grandmother + sold_to_uncle)

theorem tory_sold_to_neighbor :
  sold_to_neighbor = 5 :=
by
  intros
  sorry

end tory_sold_to_neighbor_l278_278830


namespace green_toads_per_acre_l278_278687

-- Conditions definitions
def brown_toads_per_green_toad : ℕ := 25
def spotted_fraction : ℝ := 1 / 4
def spotted_brown_per_acre : ℕ := 50

-- Theorem statement to prove the main question
theorem green_toads_per_acre :
  (brown_toads_per_green_toad * spotted_brown_per_acre * spotted_fraction).to_nat / brown_toads_per_green_toad = 8 := 
sorry

end green_toads_per_acre_l278_278687


namespace remainder_98_pow_50_mod_100_l278_278710

theorem remainder_98_pow_50_mod_100 :
  (98 : ℤ) ^ 50 % 100 = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l278_278710


namespace largest_n_for_negative_sum_l278_278177

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ} -- common difference of the arithmetic sequence

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

theorem largest_n_for_negative_sum
  (h_arith_seq : is_arithmetic_sequence a d)
  (h_first_term : a 0 < 0)
  (h_sum_2015_2016 : a 2014 + a 2015 > 0)
  (h_product_2015_2016 : a 2014 * a 2015 < 0) :
  (∀ n, sum_of_first_n_terms a n < 0 → n ≤ 4029) ∧ (sum_of_first_n_terms a 4029 < 0) :=
sorry

end largest_n_for_negative_sum_l278_278177


namespace cleaner_flow_rate_after_second_unclogging_l278_278443

theorem cleaner_flow_rate_after_second_unclogging
  (rate1 rate2 : ℕ) (time1 time2 total_time total_cleaner : ℕ)
  (used_cleaner1 used_cleaner2 : ℕ)
  (final_rate : ℕ)
  (H1 : rate1 = 2)
  (H2 : rate2 = 3)
  (H3 : time1 = 15)
  (H4 : time2 = 10)
  (H5 : total_time = 30)
  (H6 : total_cleaner = 80)
  (H7 : used_cleaner1 = rate1 * time1)
  (H8 : used_cleaner2 = rate2 * time2)
  (H9 : used_cleaner1 + used_cleaner2 ≤ total_cleaner)
  (H10 : final_rate = (total_cleaner - (used_cleaner1 + used_cleaner2)) / (total_time - (time1 + time2))) :
  final_rate = 4 := by
  sorry

end cleaner_flow_rate_after_second_unclogging_l278_278443


namespace intersection_complement_eq_l278_278344

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- Define the intersection of A and complement of B
def intersection : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- The theorem to be proved
theorem intersection_complement_eq : (A ∩ complement_B) = intersection :=
sorry

end intersection_complement_eq_l278_278344


namespace count_odd_distinct_digits_numbers_l278_278028

theorem count_odd_distinct_digits_numbers :
  let odd_digits := [1, 3, 5, 7, 9]
  let four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999}
  (number_of_distinct_digit_numbers (four_digit_numbers ∩ {n | ∃ d, odd_digits.includes d ∧ n % 10 = d})) = 2240 :=
sorry

end count_odd_distinct_digits_numbers_l278_278028


namespace no_integer_solutions_for_eq_l278_278235

theorem no_integer_solutions_for_eq {x y : ℤ} : ¬ (∃ x y : ℤ, (x + 7) * (x + 6) = 8 * y + 3) := by
  sorry

end no_integer_solutions_for_eq_l278_278235


namespace find_a_b_find_m_l278_278342

-- Define the parabola and the points it passes through
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- The conditions based on the given problem
def condition1 (a b : ℝ) : Prop := parabola a b 1 = -2
def condition2 (a b : ℝ) : Prop := parabola a b (-2) = 13

-- Part 1: Proof for a and b
theorem find_a_b : ∃ a b : ℝ, condition1 a b ∧ condition2 a b ∧ a = 1 ∧ b = -4 :=
by sorry

-- Part 2: Given y equation and the specific points
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 1

-- Conditions for the second part
def condition3 : Prop := parabola2 5 = 6
def condition4 (m : ℝ) : Prop := parabola2 m = 12 - 6

-- Theorem statement for the second part
theorem find_m : ∃ m : ℝ, condition3 ∧ condition4 m ∧ m = -1 :=
by sorry

end find_a_b_find_m_l278_278342


namespace equation_of_line_through_point_l278_278754

theorem equation_of_line_through_point (a T : ℝ) (h : a ≠ 0 ∧ T ≠ 0) :
  ∃ k : ℝ, (k = T / (a^2)) ∧ (k * x + (2 * T / a)) = (k * x + (2 * T / a)) → 
  (T * x - a^2 * y + 2 * T * a = 0) :=
by
  use T / (a^2)
  sorry

end equation_of_line_through_point_l278_278754


namespace x_intercept_of_line_l278_278168

theorem x_intercept_of_line (x y : ℚ) (h : 4 * x + 6 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by
  sorry

end x_intercept_of_line_l278_278168


namespace james_second_hour_distance_l278_278878

theorem james_second_hour_distance :
  ∃ x : ℝ, 
    x + 1.20 * x + 1.50 * x = 37 ∧ 
    1.20 * x = 12 :=
by
  sorry

end james_second_hour_distance_l278_278878


namespace fraction_simplify_l278_278304

theorem fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by sorry

end fraction_simplify_l278_278304


namespace find_d_l278_278829

theorem find_d (d : ℝ) (h : 4 * (3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5) = 3200.0000000000005) : d = 0.3 :=
by
  sorry

end find_d_l278_278829


namespace max_value_of_m_l278_278514

theorem max_value_of_m {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 20) :
  ∃ m, m = min (a * b) (min (b * c) (c * a)) ∧ m = 12 :=
by
  sorry

end max_value_of_m_l278_278514


namespace range_of_a_l278_278359

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a * x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := 
by 
  sorry

end range_of_a_l278_278359


namespace gamma_received_eight_donuts_l278_278462

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end gamma_received_eight_donuts_l278_278462


namespace solve_arctan_equation_l278_278399

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (1 / x) + Real.arctan (1 / (x^3))

theorem solve_arctan_equation (x : ℝ) (hx : x = (1 + Real.sqrt 5) / 2) :
  f x = Real.pi / 4 :=
by
  rw [hx]
  sorry

end solve_arctan_equation_l278_278399


namespace problem_inequality_l278_278628

variable {a b : ℝ}

theorem problem_inequality 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0)
  (h_a_gt_b : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) := 
by 
  sorry

end problem_inequality_l278_278628


namespace problem_a_lt_c_lt_b_l278_278382

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem problem_a_lt_c_lt_b : a < c ∧ c < b := 
by {
  sorry
}

end problem_a_lt_c_lt_b_l278_278382


namespace range_of_omega_l278_278013

theorem range_of_omega (ω : ℝ) (h_pos : ω > 0) (h_three_high_points : (9 * π / 2) ≤ ω + π / 4 ∧ ω + π / 4 < 6 * π + π / 2) : 
           (17 * π / 4) ≤ ω ∧ ω < (25 * π / 4) :=
  sorry

end range_of_omega_l278_278013


namespace correct_operation_l278_278272

variable (x y : ℝ)

theorem correct_operation : 3 * x * y² - 4 * x * y² = -x * y² :=
by
  sorry

end correct_operation_l278_278272


namespace bike_helmet_cost_increase_l278_278879

open Real

theorem bike_helmet_cost_increase :
  let old_bike_cost := 150
  let old_helmet_cost := 50
  let new_bike_cost := old_bike_cost + 0.10 * old_bike_cost
  let new_helmet_cost := old_helmet_cost + 0.20 * old_helmet_cost
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let total_increase := new_total_cost - old_total_cost
  let percent_increase := (total_increase / old_total_cost) * 100
  percent_increase = 12.5 :=
by
  sorry

end bike_helmet_cost_increase_l278_278879


namespace probability_of_mixed_selection_distribution_of_X_expected_value_of_X_l278_278974

namespace ZongziProblem

open ProbabilityTheory

def total_zongzi : ℕ := 10
def red_bean_zongzi : ℕ := 2
def plain_zongzi : ℕ := 8
def selected_zongzi : ℕ := 3

-- Question 1
theorem probability_of_mixed_selection : 
  let C (n k : ℕ) := nat.choose n k in
  proof_problem := (C(2, 1) * C(8, 2) + C(2, 2) * C(8, 1)) / C(10, 3) = 8 / 15 :=
sorry

-- Question 2
def X : Type := {x : ℕ // x ≤ 2}  -- Representing the number of red bean zongzi selected

def P (x : ℕ) : Rational :=
  if x = 0 then 7 / 15 else
  if x = 1 then 7 / 15 else
  if x = 2 then 1 / 15 else
  0

theorem distribution_of_X : 
  ( ∑ x ∈ {0, 1, 2}, P(x) = 1 ) ∧
  ( P(0) = 7 / 15 ) ∧
  ( P(1) = 7 / 15 ) ∧
  ( P(2) = 1 / 15 ) :=
sorry

theorem expected_value_of_X :
  ∑ x ∈ {0, 1, 2}, x * P(x) = 3 / 5 :=
sorry

end ZongziProblem

end probability_of_mixed_selection_distribution_of_X_expected_value_of_X_l278_278974


namespace temperature_difference_l278_278870

-- Define the temperatures given in the problem.
def T_noon : ℝ := 10
def T_midnight : ℝ := -150

-- State the theorem to prove the temperature difference.
theorem temperature_difference :
  T_noon - T_midnight = 160 :=
by
  -- We skip the proof and add sorry.
  sorry

end temperature_difference_l278_278870


namespace player_catches_ball_in_5_seconds_l278_278085

theorem player_catches_ball_in_5_seconds
    (s_ball : ℕ → ℝ) (s_player : ℕ → ℝ)
    (t_ball : ℕ)
    (t_player : ℕ)
    (d_player_initial : ℝ)
    (d_sideline : ℝ) :
  (∀ t, s_ball t = (4.375 * t - 0.375 * t^2)) →
  (∀ t, s_player t = (3.25 * t + 0.25 * t^2)) →
  (d_player_initial = 10) →
  (d_sideline = 23) →
  t_player = 5 →
  s_player t_player + d_player_initial = s_ball t_player ∧ s_ball t_player < d_sideline := 
by sorry

end player_catches_ball_in_5_seconds_l278_278085


namespace all_edges_same_color_l278_278132

-- Define the vertices in the two pentagons and the set of all vertices
inductive vertex
| A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | B5
open vertex

-- Predicate to identify edges between vertices
def edge (v1 v2 : vertex) : Prop :=
  match (v1, v2) with
  | (A1, A2) | (A2, A3) | (A3, A4) | (A4, A5) | (A5, A1) => true
  | (B1, B2) | (B2, B3) | (B3, B4) | (B4, B5) | (B5, B1) => true
  | (A1, B1) | (A1, B2) | (A1, B3) | (A1, B4) | (A1, B5) => true
  | (A2, B1) | (A2, B2) | (A2, B3) | (A2, B4) | (A2, B5) => true
  | (A3, B1) | (A3, B2) | (A3, B3) | (A3, B4) | (A3, B5) => true
  | (A4, B1) | (A4, B2) | (A4, B3) | (A4, B4) | (A4, B5) => true
  | (A5, B1) | (A5, B2) | (A5, B3) | (A5, B4) | (A5, B5) => true
  | _ => false

-- Edge coloring predicate 'black' or 'white'
inductive color
| black | white
open color

def edge_color (v1 v2 : vertex) : color → Prop :=
  sorry -- Coloring function needs to be defined accordingly

-- Predicate to check for monochrome triangles
def no_monochrome_triangle : Prop :=
  ∀ v1 v2 v3 : vertex,
    (edge v1 v2 ∧ edge v2 v3 ∧ edge v3 v1) →
    ¬ (∃ c : color, edge_color v1 v2 c ∧ edge_color v2 v3 c ∧ edge_color v3 v1 c)

-- Main theorem statement
theorem all_edges_same_color (no_mt : no_monochrome_triangle) :
  ∃ c : color, ∀ v1 v2 : vertex,
    (edge v1 v2 ∧ (v1 = A1 ∨ v1 = A2 ∨ v1 = A3 ∨ v1 = A4 ∨ v1 = A5) ∧
                 (v2 = A1 ∨ v2 = A2 ∨ v2 = A3 ∨ v2 = A4 ∨ v2 = A5) ) →
    edge_color v1 v2 c ∧
    (edge v1 v2 ∧ (v1 = B1 ∨ v1 = B2 ∨ v1 = B3 ∨ v1 = B4 ∨ v1 = B5) ∧
                 (v2 = B1 ∨ v2 = B2 ∨ v2 = B3 ∨ v2 = B4 ∨ v2 = B5) ) →
    edge_color v1 v2 c := sorry

end all_edges_same_color_l278_278132


namespace initial_money_amount_l278_278280

theorem initial_money_amount 
  (X : ℝ) 
  (h : 0.70 * X = 350) : 
  X = 500 := 
sorry

end initial_money_amount_l278_278280


namespace initial_action_figures_l278_278375

theorem initial_action_figures (x : ℕ) (h : x + 2 - 7 = 10) : x = 15 :=
by
  sorry

end initial_action_figures_l278_278375


namespace third_vs_second_plant_relationship_l278_278654

-- Define the constants based on the conditions
def first_plant_tomatoes := 24
def second_plant_tomatoes := 12 + 5  -- Half of 24 plus 5
def total_tomatoes := 60

-- Define the production of the third plant based on the total number of tomatoes
def third_plant_tomatoes := total_tomatoes - (first_plant_tomatoes + second_plant_tomatoes)

-- Define the relationship to be proved
theorem third_vs_second_plant_relationship : 
  third_plant_tomatoes = second_plant_tomatoes + 2 :=
by
  -- Proof not provided, adding sorry to skip
  sorry

end third_vs_second_plant_relationship_l278_278654


namespace base12_remainder_l278_278920

theorem base12_remainder (x : ℕ) (h : x = 2 * 12^3 + 7 * 12^2 + 4 * 12 + 5) : x % 5 = 2 :=
by {
    -- Proof would go here
    sorry
}

end base12_remainder_l278_278920


namespace radius_of_smaller_circle_l278_278200

open Real

-- Definitions based on the problem conditions
def large_circle_radius : ℝ := 10
def pattern := "square"

-- Statement of the problem in Lean 4
theorem radius_of_smaller_circle :
  ∀ (r : ℝ), (large_circle_radius = 10) → (pattern = "square") → r = 5 * sqrt 2 →  ∃ r, r = 5 * sqrt 2 :=
by
  sorry

end radius_of_smaller_circle_l278_278200


namespace total_pieces_l278_278327

-- Define the given conditions
def pieces_eaten_per_person : ℕ := 4
def num_people : ℕ := 3

-- Theorem stating the result
theorem total_pieces (h : num_people > 0) : (num_people * pieces_eaten_per_person) = 12 := 
by
  sorry

end total_pieces_l278_278327


namespace ratio_unit_price_brand_x_to_brand_y_l278_278453

-- Definitions based on the conditions in the problem
def volume_brand_y (v : ℝ) := v
def price_brand_y (p : ℝ) := p
def volume_brand_x (v : ℝ) := 1.3 * v
def price_brand_x (p : ℝ) := 0.85 * p
noncomputable def unit_price (volume : ℝ) (price : ℝ) := price / volume

-- Theorems to prove the ratio of unit price of Brand X to Brand Y is 17/26
theorem ratio_unit_price_brand_x_to_brand_y (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0) : 
  (unit_price (volume_brand_x v) (price_brand_x p)) / (unit_price (volume_brand_y v) (price_brand_y p)) = 17 / 26 := by
  sorry

end ratio_unit_price_brand_x_to_brand_y_l278_278453


namespace distinct_solutions_count_l278_278022

theorem distinct_solutions_count : ∀ (x : ℝ), (|x - 5| = |x + 3| ↔ x = 1) → ∃! x, |x - 5| = |x + 3| :=
by
  intro x h
  existsi 1
  rw h
  sorry 

end distinct_solutions_count_l278_278022


namespace division_problem_l278_278271

theorem division_problem (A : ℕ) (h : 23 = (A * 3) + 2) : A = 7 :=
sorry

end division_problem_l278_278271


namespace segment_ratio_ae_ad_l278_278518

/-- Given points B, C, and E lie on line segment AD, and the following conditions:
  1. The length of segment AB is twice the length of segment BD.
  2. The length of segment AC is 5 times the length of segment CD.
  3. The length of segment BE is one-third the length of segment EC.
Prove that the fraction of the length of segment AD that segment AE represents is 17/24. -/
theorem segment_ratio_ae_ad (AB BD AC CD BE EC AD AE : ℝ)
    (h1 : AB = 2 * BD)
    (h2 : AC = 5 * CD)
    (h3 : BE = (1/3) * EC)
    (h4 : AD = 6 * CD)
    (h5 : AE = 4.25 * CD) :
    AE / AD = 17 / 24 := 
  by 
  sorry

end segment_ratio_ae_ad_l278_278518


namespace largest_natural_number_not_sum_of_two_composites_l278_278812

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278812


namespace find_quotient_l278_278533

theorem find_quotient :
  ∃ q : ℕ, ∀ L S : ℕ, L = 1584 ∧ S = 249 ∧ (L - S = 1335) ∧ (L = S * q + 15) → q = 6 :=
by
  sorry

end find_quotient_l278_278533


namespace perpendicular_angles_l278_278004

theorem perpendicular_angles (α : ℝ) 
  (h1 : 4 * Real.pi < α) 
  (h2 : α < 6 * Real.pi)
  (h3 : ∃ (k : ℤ), α = -2 * Real.pi / 3 + Real.pi / 2 + k * Real.pi) :
  α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6 :=
by
  sorry

end perpendicular_angles_l278_278004


namespace cocktail_cans_l278_278572

theorem cocktail_cans (prev_apple_ratio : ℝ) (prev_grape_ratio : ℝ) 
  (new_apple_cans : ℝ) : ∃ new_grape_cans : ℝ, new_grape_cans = 15 :=
by
  let prev_apple_per_can := 1 / 6
  let prev_grape_per_can := 1 / 10
  let prev_total_per_can := (1 / 6) + (1 / 10)
  let new_apple_per_can := 1 / 5
  let new_grape_per_can := prev_total_per_can - new_apple_per_can
  let result := 1 / new_grape_per_can
  use result
  sorry

end cocktail_cans_l278_278572


namespace rebecca_groups_of_eggs_l278_278237

def eggs : Nat := 16
def group_size : Nat := 2

theorem rebecca_groups_of_eggs : (eggs / group_size) = 8 := by
  sorry

end rebecca_groups_of_eggs_l278_278237


namespace hour_hand_degrees_noon_to_2_30_l278_278566

def degrees_moved (hours: ℕ) : ℝ := (hours * 30)

theorem hour_hand_degrees_noon_to_2_30 :
  degrees_moved 2 + degrees_moved 1 / 2 = 75 :=
sorry

end hour_hand_degrees_noon_to_2_30_l278_278566


namespace ratio_john_maya_age_l278_278158

theorem ratio_john_maya_age :
  ∀ (john drew maya peter jacob : ℕ),
  -- Conditions:
  john = 30 ∧
  drew = maya + 5 ∧
  peter = drew + 4 ∧
  jacob = 11 ∧
  jacob + 2 = (peter + 2) / 2 →
  -- Conclusion:
  john / gcd john maya = 2 ∧ maya / gcd john maya = 1 :=
by
  sorry

end ratio_john_maya_age_l278_278158


namespace textbook_order_total_cost_l278_278953

theorem textbook_order_total_cost :
  let english_quantity := 35
  let geography_quantity := 35
  let mathematics_quantity := 20
  let science_quantity := 30
  let english_price := 7.50
  let geography_price := 10.50
  let mathematics_price := 12.00
  let science_price := 9.50
  (english_quantity * english_price + geography_quantity * geography_price + mathematics_quantity * mathematics_price + science_quantity * science_price = 1155.00) :=
by sorry

end textbook_order_total_cost_l278_278953


namespace darcy_commute_l278_278928

theorem darcy_commute (d w r t x time_walk train_time : ℝ) 
  (h1 : d = 1.5)
  (h2 : w = 3)
  (h3 : r = 20)
  (h4 : train_time = t + x)
  (h5 : time_walk = 15 + train_time)
  (h6 : time_walk = d / w * 60)  -- Time taken to walk in minutes
  (h7 : t = d / r * 60)  -- Time taken on train in minutes
  : x = 10.5 :=
sorry

end darcy_commute_l278_278928


namespace find_integer_pairs_l278_278047

-- Define the plane and lines properties
def horizontal_lines (h : ℕ) : Prop := h > 0
def non_horizontal_lines (s : ℕ) : Prop := s > 0
def non_parallel (s : ℕ) : Prop := s > 0
def no_three_intersect (total_lines : ℕ) : Prop := total_lines > 0

-- Function to calculate regions from the given formula
def calculate_regions (h s : ℕ) : ℕ :=
  h * (s + 1) + 1 + (s * (s + 1)) / 2

-- Prove that the given (h, s) pairs divide the plane into 1992 regions
theorem find_integer_pairs :
  (horizontal_lines 995 ∧ non_horizontal_lines 1 ∧ non_parallel 1 ∧ no_three_intersect (995 + 1) ∧ calculate_regions 995 1 = 1992)
  ∨ (horizontal_lines 176 ∧ non_horizontal_lines 10 ∧ non_parallel 10 ∧ no_three_intersect (176 + 10) ∧ calculate_regions 176 10 = 1992)
  ∨ (horizontal_lines 80 ∧ non_horizontal_lines 21 ∧ non_parallel 21 ∧ no_three_intersect (80 + 21) ∧ calculate_regions 80 21 = 1992) :=
by
  -- Include individual cases to verify correctness of regions calculation
  sorry

end find_integer_pairs_l278_278047


namespace prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l278_278070

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l278_278070


namespace limit_sequence_l278_278775

def sequence (n : ℕ) : ℝ :=
  (sqrt (3 * n - 1) - (125 * n ^ 3 + n) ^ (1 / 3)) /
  (n ^ (1 / 3) - n)

theorem limit_sequence : Filter.Tendsto sequence Filter.atTop (nhds 5) :=
by
  sorry

end limit_sequence_l278_278775


namespace largest_non_summable_composite_l278_278804

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278804


namespace max_value_of_z_l278_278347

variable (x y z : ℝ)

def condition1 : Prop := 2 * x + y ≤ 4
def condition2 : Prop := x ≤ y
def condition3 : Prop := x ≥ 1 / 2
def objective_function : ℝ := 2 * x - y

theorem max_value_of_z :
  (∀ x y, condition1 x y ∧ condition2 x y ∧ condition3 x → z = objective_function x y) →
  z ≤ 4 / 3 :=
sorry

end max_value_of_z_l278_278347


namespace scientific_notation_of_22nm_l278_278401

theorem scientific_notation_of_22nm (h : 22 * 10^(-9) = 0.000000022) : 0.000000022 = 2.2 * 10^(-8) :=
sorry

end scientific_notation_of_22nm_l278_278401


namespace pure_imaginary_complex_number_l278_278485

variable (a : ℝ)

theorem pure_imaginary_complex_number:
  (a^2 + 2*a - 3 = 0) ∧ (a^2 + a - 6 ≠ 0) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l278_278485


namespace min_max_abs_poly_eq_zero_l278_278978

theorem min_max_abs_poly_eq_zero :
  ∃ y : ℝ, (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |x^2 - x^3 * y| ≤ 0) :=
sorry

end min_max_abs_poly_eq_zero_l278_278978


namespace tickets_system_l278_278685

variable (x y : ℕ)

theorem tickets_system (h1 : x + y = 20) (h2 : 2800 * x + 6400 * y = 74000) :
  (x + y = 20) ∧ (2800 * x + 6400 * y = 74000) :=
by {
  exact (And.intro h1 h2)
}

end tickets_system_l278_278685


namespace sum_put_at_simple_interest_l278_278447

theorem sum_put_at_simple_interest (P R : ℝ) 
  (h : ((P * (R + 3) * 2) / 100) - ((P * R * 2) / 100) = 300) : 
  P = 5000 :=
by
  sorry

end sum_put_at_simple_interest_l278_278447


namespace original_radius_of_cylinder_l278_278055

theorem original_radius_of_cylinder (r y : ℝ) 
  (h₁ : 3 * π * ((r + 5)^2 - r^2) = y) 
  (h₂ : 5 * π * r^2 = y)
  (h₃ : 3 > 0) :
  r = 7.5 :=
by
  sorry

end original_radius_of_cylinder_l278_278055


namespace number_of_turns_l278_278752

/-
  Given the cyclist's speed v = 5 m/s, time duration t = 5 s,
  and the circumference of the wheel c = 1.25 m, 
  prove that the number of complete turns n the wheel makes is equal to 20.
-/
theorem number_of_turns (v t c : ℝ) (h_v : v = 5) (h_t : t = 5) (h_c : c = 1.25) : 
  (v * t) / c = 20 :=
by
  sorry

end number_of_turns_l278_278752


namespace g_sum_even_l278_278883

def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5

theorem g_sum_even (a b c d : ℝ) (h : g 42 a b c d = 3) : g 42 a b c d + g (-42) a b c d = 6 := by
  sorry

end g_sum_even_l278_278883


namespace sufficient_but_not_necessary_l278_278937

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 2) (h2 : b > 2) : 
  a + b > 4 ∧ a * b > 4 := 
by
  sorry

end sufficient_but_not_necessary_l278_278937


namespace unique_integral_root_of_equation_l278_278534

theorem unique_integral_root_of_equation :
  ∀ x : ℤ, (x - 9 / (x - 5) = 7 - 9 / (x - 5)) ↔ (x = 7) :=
by
  sorry

end unique_integral_root_of_equation_l278_278534


namespace alex_chairs_l278_278662

theorem alex_chairs (x y z : ℕ) (h : x + y + z = 74) : z = 74 - x - y :=
by
  sorry

end alex_chairs_l278_278662


namespace smallest_num_rectangles_to_cover_square_l278_278735

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l278_278735


namespace min_radius_for_area_l278_278229

theorem min_radius_for_area (A : ℝ) (hA : A = 500) : ∃ r : ℝ, r = 13 ∧ π * r^2 ≥ A :=
by
  sorry

end min_radius_for_area_l278_278229


namespace binom_20_5_l278_278307

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end binom_20_5_l278_278307


namespace find_circle_equation_l278_278992

noncomputable def circle_equation (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) = (-1, 3) ∨ (x, y) = (0, 0) ∨ (x, y) = (0, 2) →
  x^2 + y^2 + D * x + E * y + F = 0

theorem find_circle_equation :
  ∃ D E F : ℝ, circle_equation D E F ∧
               (∀ x y, x^2 + y^2 + D * x + E * y + F = x^2 + y^2 + 4 * x - 2 * y) :=
sorry

end find_circle_equation_l278_278992


namespace problem1_xy_value_problem2_min_value_l278_278673

-- Define the first problem conditions
def problem1 (x y : ℝ) : Prop :=
  x^2 - 2 * x * y + 2 * y^2 + 6 * y + 9 = 0

-- Prove that xy = 9 given the above condition
theorem problem1_xy_value (x y : ℝ) (h : problem1 x y) : x * y = 9 :=
  sorry

-- Define the second problem conditions
def expression (m : ℝ) : ℝ :=
  m^2 + 6 * m + 13

-- Prove that the minimum value of the expression is 4
theorem problem2_min_value : ∃ m, expression m = 4 :=
  sorry

end problem1_xy_value_problem2_min_value_l278_278673


namespace find_z_l278_278617

theorem find_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z * i = 2 - i) : z = -1 - 2 * i := 
by
  sorry

end find_z_l278_278617


namespace binom_odd_n_eq_2_pow_m_minus_1_l278_278001

open Nat

/-- For which n will binom n k be odd for every 0 ≤ k ≤ n?
    Prove that n = 2^m - 1 for some m ≥ 1. -/
theorem binom_odd_n_eq_2_pow_m_minus_1 (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Nat.choose n k % 2 = 1) ↔ (∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1) :=
by
  sorry

end binom_odd_n_eq_2_pow_m_minus_1_l278_278001


namespace swim_team_more_people_l278_278102

theorem swim_team_more_people :
  let car1_people := 5
  let car2_people := 4
  let van1_people := 3
  let van2_people := 3
  let van3_people := 5
  let minibus_people := 10

  let car_max_capacity := 6
  let van_max_capacity := 8
  let minibus_max_capacity := 15

  let actual_people := car1_people + car2_people + van1_people + van2_people + van3_people + minibus_people
  let max_capacity := 2 * car_max_capacity + 3 * van_max_capacity + minibus_max_capacity
  (max_capacity - actual_people : ℕ) = 21 := 
  by
    sorry

end swim_team_more_people_l278_278102


namespace sequence_property_l278_278066

theorem sequence_property : 
  ∀ (a : ℕ → ℝ), 
    a 1 = 1 →
    a 2 = 1 → 
    (∀ n, a (n + 2) = a (n + 1) + 1 / a n) →
    a 180 > 19 :=
by
  intros a h1 h2 h3
  sorry

end sequence_property_l278_278066


namespace neg_mod_eq_1998_l278_278496

theorem neg_mod_eq_1998 {a : ℤ} (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end neg_mod_eq_1998_l278_278496


namespace necessary_not_sufficient_l278_278286

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end necessary_not_sufficient_l278_278286


namespace pattern_continues_for_max_8_years_l278_278559

def is_adult_age (age : ℕ) := 18 ≤ age ∧ age < 40

def fits_pattern (p1 p2 n : ℕ) : Prop := 
  is_adult_age p1 ∧
  is_adult_age p2 ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 
    (k % (p1 + k) = 0 ∨ k % (p2 + k) = 0) ∧ ¬ (k % (p1 + k) = 0 ∧ k % (p2 + k) = 0))

theorem pattern_continues_for_max_8_years (p1 p2 : ℕ) : 
  fits_pattern p1 p2 8 := 
sorry

end pattern_continues_for_max_8_years_l278_278559


namespace stratified_sampling_group_B_l278_278951

theorem stratified_sampling_group_B
  (total_cities : ℕ)
  (group_A_cities : ℕ)
  (group_B_cities : ℕ)
  (group_C_cities : ℕ)
  (total_sampled : ℕ)
  (h_total : total_cities = 24)
  (h_A : group_A_cities = 4)
  (h_B : group_B_cities = 12)
  (h_C : group_C_cities = 8)
  (h_sampled : total_sampled = 6) :
  group_B_cities * total_sampled / total_cities = 3 := 
by
  rw [h_total, h_A, h_B, h_C, h_sampled] 
  -- Provide a simpler proof if necessary, or use algebraic manipulations
  -- onioning the correctness of the statement
  norm_num
  sorry

end stratified_sampling_group_B_l278_278951


namespace deductible_increase_l278_278278

theorem deductible_increase (current_deductible : ℝ) (increase_fraction : ℝ) (next_year_deductible : ℝ) : 
  current_deductible = 3000 ∧ increase_fraction = 2 / 3 ∧ next_year_deductible = (1 + increase_fraction) * current_deductible →
  next_year_deductible - current_deductible = 2000 :=
by
  intros h
  sorry

end deductible_increase_l278_278278


namespace digit_sum_is_twelve_l278_278932

theorem digit_sum_is_twelve (n x y : ℕ) (h1 : n = 10 * x + y) (h2 : 0 ≤ x ∧ x ≤ 9) (h3 : 0 ≤ y ∧ y ≤ 9)
  (h4 : (1 / 2 : ℚ) * n = (1 / 4 : ℚ) * n + 3) : x + y = 12 :=
by
  sorry

end digit_sum_is_twelve_l278_278932


namespace gcd_lcm_product_l278_278474

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 3 * 5^2) (h2 : b = 5^3) : 
  Nat.gcd a b * Nat.lcm a b = 9375 := by
  sorry

end gcd_lcm_product_l278_278474


namespace infinite_not_expressible_as_sum_of_three_squares_l278_278123

theorem infinite_not_expressible_as_sum_of_three_squares :
  ∃ (n : ℕ), ∃ (infinitely_many_n : ℕ → Prop), (∀ m:ℕ, (infinitely_many_n m ↔ m ≡ 7 [MOD 8])) ∧ ∀ a b c : ℕ, n ≠ a^2 + b^2 + c^2 := 
by
  sorry

end infinite_not_expressible_as_sum_of_three_squares_l278_278123


namespace Alice_spent_19_percent_l278_278958

variable (A B A': ℝ)
def Bob_less_money_than_Alice (A B : ℝ) : Prop :=
  B = 0.9 * A

def Alice_less_money_than_Bob (B A' : ℝ) : Prop :=
  A' = 0.9 * B

theorem Alice_spent_19_percent (A B A' : ℝ) 
  (h1 : Bob_less_money_than_Alice A B)
  (h2 : Alice_less_money_than_Bob B A') :
  ((A - A') / A) * 100 = 19 :=
by
  sorry

end Alice_spent_19_percent_l278_278958


namespace max_distance_circle_ellipse_l278_278670

theorem max_distance_circle_ellipse:
  (∀ P Q : ℝ × ℝ, 
     (P.1^2 + (P.2 - 3)^2 = 1 / 4) → 
     (Q.1^2 + 4 * Q.2^2 = 4) → 
     ∃ Q_max : ℝ × ℝ, 
         Q_max = (0, -1) ∧ 
         (∀ P : ℝ × ℝ, P.1^2 + (P.2 - 3)^2 = 1 / 4 →
         |dist P Q_max| = 9 / 2)) := 
sorry

end max_distance_circle_ellipse_l278_278670


namespace find_divisor_l278_278923

theorem find_divisor (n d : ℤ) (k : ℤ)
  (h1 : n % d = 3)
  (h2 : n^2 % d = 4) : d = 5 :=
sorry

end find_divisor_l278_278923


namespace angle_OA_plane_ABC_l278_278302

noncomputable def sphere_radius (A B C : Type*) (O : Type*) : ℝ :=
  let surface_area : ℝ := 48 * Real.pi
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let radius := Real.sqrt (surface_area / (4 * Real.pi))
  radius

noncomputable def length_AC (A B C : Type*) : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let AC := Real.sqrt (AB ^ 2 + BC ^ 2 - 2 * AB * BC * Real.cos angle_ABC)
  AC

theorem angle_OA_plane_ABC 
(A B C O : Type*)
(radius : ℝ)
(AC : ℝ) :
radius = 2 * Real.sqrt 3 ∧
AC = 2 * Real.sqrt 3 ∧ 
(AB : ℝ) = 2 ∧ 
(BC : ℝ) = 4 ∧ 
(angle_ABC : ℝ) = Real.pi / 3
→ ∃ (angle_OA_plane_ABC : ℝ), angle_OA_plane_ABC = Real.arccos (Real.sqrt 3 / 3) :=
by
  intro h
  sorry

end angle_OA_plane_ABC_l278_278302


namespace value_of_f_three_l278_278849

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * Real.cos x - x

theorem value_of_f_three (a b : ℝ) (h : f a b (-3) = 7) : f a b 3 = 1 :=
by
  sorry

end value_of_f_three_l278_278849


namespace max_x2_plus_2xy_plus_3y2_l278_278243

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end max_x2_plus_2xy_plus_3y2_l278_278243


namespace cos_double_beta_alpha_plus_double_beta_l278_278009

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = Real.sqrt 2 / 10)
variable (h2 : Real.sin β = Real.sqrt 10 / 10)

theorem cos_double_beta :
  Real.cos (2 * β) = 4 / 5 := by 
  sorry

theorem alpha_plus_double_beta :
  α + 2 * β = π / 4 := by 
  sorry

end cos_double_beta_alpha_plus_double_beta_l278_278009


namespace part_one_part_two_i_part_two_ii_l278_278182

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem part_one (a b : ℝ) : 
  f (-a / 2 + 1) a b ≤ f (a^2 + 5 / 4) a b :=
sorry

theorem part_two_i (a b : ℝ) : 
  f 1 a b + f 3 a b - 2 * f 2 a b = 2 :=
sorry

theorem part_two_ii (a b : ℝ) : 
  ¬((|f 1 a b| < 1/2) ∧ (|f 2 a b| < 1/2) ∧ (|f 3 a b| < 1/2)) :=
sorry

end part_one_part_two_i_part_two_ii_l278_278182


namespace range_of_a_l278_278894

-- Define the propositions
def Proposition_p (a : ℝ) := ∀ x : ℝ, x > 0 → x + 1/x > a
def Proposition_q (a : ℝ) := ∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0

-- Define the main theorem
theorem range_of_a (a : ℝ) (h1 : ¬ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) 
(h2 : (∀ x : ℝ, x > 0 → x + 1/x > a) ∧ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) :
a ≥ 2 :=
sorry

end range_of_a_l278_278894


namespace vector_dot_product_condition_l278_278366

variables {V : Type*} [InnerProductSpace ℝ V]

def isosceles_triangle (A B C : V) : Prop :=
  (A - B).norm = (A - C).norm

noncomputable def angle_BAC (A B C : V) : ℝ := (A - B) ⬝ (A - C) / 
  (∥A - B∥ * ∥A - C∥)

variables (A B C D E : V)
noncomputable def length_AB : ℝ := (A - B).norm
noncomputable def length_AC : ℝ := (A - C).norm

axiom condition1 : isosceles_triangle A B C
axiom condition2 : angle_BAC A B C = -1 / 2  -- since cos(120°) = -1/2
axiom condition3 : length_AB = 2
axiom condition4 : length_AC = 2
axiom condition5 : B - C = 2 • (B - D)
axiom condition6 : A - C = 3 • (A - E)

theorem vector_dot_product_condition :
  (A - D) ⬝ (B - E) = -2 / 3 :=
sorry

end vector_dot_product_condition_l278_278366


namespace parametric_line_l278_278250

theorem parametric_line (s m : ℤ) :
  (∀ t : ℤ, ∃ x y : ℤ, 
    y = 5 * x - 7 ∧
    x = s + 6 * t ∧ y = 3 + m * t ) → 
  (s = 2 ∧ m = 30) :=
by
  sorry

end parametric_line_l278_278250


namespace quadratic_root_relation_l278_278336

theorem quadratic_root_relation (x₁ x₂ : ℝ) (h₁ : x₁ ^ 2 - 3 * x₁ + 2 = 0) (h₂ : x₂ ^ 2 - 3 * x₂ + 2 = 0) :
  x₁ + x₂ - x₁ * x₂ = 1 := by
sorry

end quadratic_root_relation_l278_278336


namespace weight_mixture_is_correct_l278_278423

noncomputable def weight_mixture_in_kg (weight_a_per_liter weight_b_per_liter : ℝ)
  (ratio_a ratio_b total_volume_liters weight_conversion : ℝ) : ℝ :=
  let total_parts := ratio_a + ratio_b
  let volume_per_part := total_volume_liters / total_parts
  let volume_a := ratio_a * volume_per_part
  let volume_b := ratio_b * volume_per_part
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  total_weight_gm / weight_conversion

theorem weight_mixture_is_correct :
  weight_mixture_in_kg 900 700 3 2 4 1000 = 3.280 :=
by
  -- Calculation should follow from the def
  sorry

end weight_mixture_is_correct_l278_278423


namespace find_real_solutions_l278_278165

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end find_real_solutions_l278_278165


namespace max_area_of_triangle_ABC_l278_278517

-- Definitions for the problem conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (5, 4)
def parabola (x : ℝ) : ℝ := x^2 - 3 * x
def C (r : ℝ) : ℝ × ℝ := (r, parabola r)

-- Function to compute the Shoelace Theorem area of ABC
def shoelace_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Proof statement
theorem max_area_of_triangle_ABC : ∃ (r : ℝ), -2 ≤ r ∧ r ≤ 5 ∧ shoelace_area A B (C r) = 39 := 
  sorry

end max_area_of_triangle_ABC_l278_278517


namespace find_all_solutions_l278_278785

noncomputable def functionalSolutions (f : ℝ → ℝ) :=
  ((∃ A : ℝ, f = λ x, A * x) ∨
   (∃ (A a : ℝ), f = λ x, A * Real.sin (a * x)) ∨
   (∃ (A a : ℝ), f = λ x, A * Real.sinh (a * x)))

theorem find_all_solutions (f : ℝ → ℝ) 
  (h_differentiable : Differentiable ℝ f)
  (h_differentiable2 : Differentiable ℝ (fderiv ℝ f)) 
  (functional_equation : ∀ x y : ℝ, f(x)^2 - f(y)^2 = f(x + y) * f(x - y)) : 
  (∀ x, f x = 0) ∨ functionalSolutions f :=
begin
  sorry
end


end find_all_solutions_l278_278785


namespace joan_balloon_gain_l278_278057

theorem joan_balloon_gain
  (initial_balloons : ℕ)
  (final_balloons : ℕ)
  (h_initial : initial_balloons = 9)
  (h_final : final_balloons = 11) :
  final_balloons - initial_balloons = 2 :=
by {
  sorry
}

end joan_balloon_gain_l278_278057


namespace closest_integer_to_1000E_l278_278150

theorem closest_integer_to_1000E : 
  let n := 2020
  let lambda := 1
  let poisson_pmf := λ k, Mathlib.exp(-lambda) * lambda^k / Mathlib.factorial k
  let E := ∑ i in finset.range (n + 1), 
            (1 - ∑ k in finset.range i, poisson_pmf k)
  in (1000 * E).nat_ceil = 1000 := 
by 
  sorry

end closest_integer_to_1000E_l278_278150


namespace smallest_num_rectangles_l278_278727

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l278_278727


namespace ordered_pair_sol_l278_278093

noncomputable def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![5, d]]

noncomputable def is_inverse_scalar_mul (d k : ℝ) : Prop :=
  (A d)⁻¹ = k • (A d)

theorem ordered_pair_sol (d k : ℝ) :
  is_inverse_scalar_mul d k → (d = -2 ∧ k = 1 / 19) :=
by
  intros h
  sorry

end ordered_pair_sol_l278_278093


namespace problem_solution_l278_278999

theorem problem_solution :
  ∀ (x y z : ℤ),
  4 * x + y + z = 80 →
  3 * x + y - z = 20 →
  x = 20 →
  2 * x - y - z = 40 :=
by
  intros x y z h1 h2 hx
  rw [hx] at h1 h2
  -- Here you could continue solving but we'll use sorry to indicate the end as no proof is requested.
  sorry

end problem_solution_l278_278999


namespace ways_to_select_at_least_one_defective_l278_278954

open Finset

-- Define basic combinatorial selection functions
def combination (n k : ℕ) := Nat.choose n k

-- Given conditions
def total_products : ℕ := 100
def defective_products : ℕ := 6
def selected_products : ℕ := 3
def non_defective_products : ℕ := total_products - defective_products

-- The question to prove: the number of ways to select at least one defective product
theorem ways_to_select_at_least_one_defective :
  (combination total_products selected_products) - (combination non_defective_products selected_products) =
  (combination 100 3) - (combination 94 3) := by
  sorry

end ways_to_select_at_least_one_defective_l278_278954


namespace gamma_received_eight_donuts_l278_278463

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end gamma_received_eight_donuts_l278_278463


namespace tangent_segment_length_l278_278570

-- Setting up the necessary definitions and theorem.
def radius := 10
def seg1 := 4
def seg2 := 2

theorem tangent_segment_length :
  ∃ X : ℝ, X = 8 ∧
  (radius^2 = X^2 + ((X + seg1 + seg2) / 2)^2) :=
by
  sorry

end tangent_segment_length_l278_278570


namespace polygon_interior_angle_increase_l278_278111

theorem polygon_interior_angle_increase (n : ℕ) (h : 3 ≤ n) :
  ((n + 1 - 2) * 180 - (n - 2) * 180 = 180) :=
by sorry

end polygon_interior_angle_increase_l278_278111


namespace maximum_expression_value_l278_278241

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end maximum_expression_value_l278_278241


namespace linear_regression_equation_l278_278260

theorem linear_regression_equation (x y : ℝ) (h : {(1, 2), (2, 3), (3, 4), (4, 5)} ⊆ {(x, y) | y = x + 1}) : 
  (∀ x y, (x = 1 → y = 2) ∧ (x = 2 → y = 3) ∧ (x = 3 → y = 4) ∧ (x = 4 → y = 5)) ↔ (y = x + 1) :=
by
  sorry

end linear_regression_equation_l278_278260


namespace fraction_zero_iff_l278_278871

theorem fraction_zero_iff (x : ℝ) (h₁ : (x - 1) / (2 * x - 4) = 0) (h₂ : 2 * x - 4 ≠ 0) : x = 1 := sorry

end fraction_zero_iff_l278_278871


namespace angle_BMC_not_obtuse_angle_BAC_is_120_l278_278532

theorem angle_BMC_not_obtuse (α β γ : ℝ) (h : α + β + γ = 180) :
  0 < 90 - α / 2 ∧ 90 - α / 2 < 90 :=
sorry

theorem angle_BAC_is_120 (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : 90 - α / 2 = α / 2) : α = 120 :=
sorry

end angle_BMC_not_obtuse_angle_BAC_is_120_l278_278532


namespace crayons_per_unit_l278_278506

theorem crayons_per_unit :
  ∀ (units : ℕ) (cost_per_crayon : ℕ) (total_cost : ℕ),
    units = 4 →
    cost_per_crayon = 2 →
    total_cost = 48 →
    (total_cost / cost_per_crayon) / units = 6 :=
by
  intros units cost_per_crayon total_cost h_units h_cost_per_crayon h_total_cost
  sorry

end crayons_per_unit_l278_278506


namespace symmetric_graph_inverse_l278_278181

def f (x : ℝ) : ℝ := sorry -- We assume f is defined accordingly somewhere, as the inverse of ln.

theorem symmetric_graph_inverse (h : ∀ x, f (f x) = x) : f 2 = Real.exp 2 := by
  sorry

end symmetric_graph_inverse_l278_278181


namespace total_songs_l278_278171

theorem total_songs (h : ℕ) (m : ℕ) (a : ℕ) (t : ℕ) (P : ℕ)
  (Hh : h = 6) (Hm : m = 3) (Ha : a = 5) 
  (Htotal : P = (h + m + a + t) / 3) 
  (Hdiv : (h + m + a + t) % 3 = 0) : P = 6 := by
  sorry

end total_songs_l278_278171


namespace matching_pair_probability_l278_278469

/-- Proof goal: To prove that Emily picking two socks randomly from her drawer, 
which has 12 gray-bottomed socks, 10 white-bottomed socks, and 6 black-bottomed socks, 
gives a probability of 1/3 that she picks a matching pair. -/

theorem matching_pair_probability :
  let gray_count := 12
  let white_count := 10
  let black_count := 6
  let total_socks := gray_count + white_count + black_count
  let total_ways := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_count * (gray_count - 1) / 2
  let white_pairs := white_count * (white_count - 1) / 2
  let black_pairs := black_count * (black_count - 1) / 2
  let matching_pairs := gray_pairs + white_pairs + black_pairs
  let probability := matching_pairs / total_ways
  probability = 1/3 := 
sorry

end matching_pair_probability_l278_278469


namespace cannot_factor_polynomial_l278_278565

theorem cannot_factor_polynomial (a b c d : ℤ) :
  ¬(x^4 + 3 * x^3 + 6 * x^2 + 9 * x + 12 = (x^2 + a * x + b) * (x^2 + c * x + d)) := 
by {
  sorry
}

end cannot_factor_polynomial_l278_278565


namespace last_ball_probability_l278_278434

variables (p q : ℕ)

def probability_white_last_ball (p : ℕ) : ℝ :=
  if p % 2 = 0 then 0 else 1

theorem last_ball_probability :
  ∀ {p q : ℕ},
    probability_white_last_ball p = if p % 2 = 0 then 0 else 1 :=
by
  intros
  sorry

end last_ball_probability_l278_278434


namespace bus_passing_time_l278_278290

noncomputable def time_for_bus_to_pass (bus_length : ℝ) (bus_speed_kph : ℝ) (man_speed_kph : ℝ) : ℝ :=
  let relative_speed_kph := bus_speed_kph + man_speed_kph
  let relative_speed_mps := (relative_speed_kph * (1000/3600))
  bus_length / relative_speed_mps

theorem bus_passing_time :
  time_for_bus_to_pass 15 40 8 = 1.125 :=
by
  sorry

end bus_passing_time_l278_278290


namespace perfect_square_probability_l278_278759

noncomputable def modified_die_faces : List ℕ := [1, 2, 3, 4, 5, 8]

theorem perfect_square_probability (n : ℕ) (H : n = 5) : 
  let possible_faces := {1, 2, 3, 4, 5, 8},
      total_outcomes := 7776,
      favorable_outcomes := 762
  in (total_outcomes = (6^5)) ∧ 
     (favorable_outcomes = ∑ x in Finset.powersetLen n (Finset.ofList modified_die_faces), 
                                 ite (is_perfect_square (x.prod id)) 1 0) ∧
     let probability := (762 : ℚ) / 7776
  in probability = 127 / 1296 ∧ 127 + 1296 = 1423 :=
by
  sorry

end perfect_square_probability_l278_278759


namespace squared_remainder_l278_278038

theorem squared_remainder (N : ℤ) (k : ℤ) :
  (N % 9 = 2 ∨ N % 9 = 7) → 
  (N^2 % 9 = 4) :=
by
  sorry

end squared_remainder_l278_278038


namespace arithmetic_seq_fifth_term_l278_278967

theorem arithmetic_seq_fifth_term (x y : ℝ) 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 2 * x^2 + 3 * y^2) 
  (h2 : a2 = x^2 + 2 * y^2) 
  (h3 : a3 = 2 * x^2 - y^2) 
  (h4 : a4 = x^2 - y^2) 
  (d : ℝ) 
  (hd : d = -x^2 - y^2) 
  (h_arith: ∀ i j k : ℕ, i < j ∧ j < k → a2 - a1 = d ∧ a3 - a2 = d ∧ a4 - a3 = d) : 
  a4 + d = -2 * y^2 := 
by 
  sorry

end arithmetic_seq_fifth_term_l278_278967


namespace units_digit_8421_1287_l278_278972

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_8421_1287 :
  units_digit (8421 ^ 1287) = 1 := 
by
  sorry

end units_digit_8421_1287_l278_278972


namespace min_rectangles_to_cover_square_exactly_l278_278723

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l278_278723


namespace domain_of_function_y_eq_sqrt_2x_3_div_x_2_l278_278971

def domain (x : ℝ) : Prop :=
  (2 * x - 3 ≥ 0) ∧ (x ≠ 2)

theorem domain_of_function_y_eq_sqrt_2x_3_div_x_2 :
  ∀ x : ℝ, domain x ↔ ((x ≥ 3 / 2) ∧ (x ≠ 2)) :=
by
  sorry

end domain_of_function_y_eq_sqrt_2x_3_div_x_2_l278_278971


namespace tiles_covering_the_floor_l278_278135

theorem tiles_covering_the_floor 
  (L W : ℕ) 
  (h1 : (∃ k, L = 10 * k) ∧ (∃ j, W = 10 * j))
  (h2 : W = 2 * L)
  (h3 : (L * L + W * W).sqrt = 45) :
  L * W = 810 :=
sorry

end tiles_covering_the_floor_l278_278135


namespace amino_inequality_l278_278424

theorem amino_inequality
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h : x + y + z = x * y * z) :
  ( (x^2 - 1) / x )^2 + ( (y^2 - 1) / y )^2 + ( (z^2 - 1) / z )^2 ≥ 4 := by
  sorry

end amino_inequality_l278_278424


namespace largest_divisor_of_expression_l278_278981

theorem largest_divisor_of_expression :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
sorry

end largest_divisor_of_expression_l278_278981


namespace probability_m_eq_kn_l278_278130

/- 
Define the conditions and question in Lean 4 -/
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_rolls : Finset (ℕ × ℕ) := Finset.product die_faces die_faces

def events_satisfying_condition : Finset (ℕ × ℕ) :=
  {(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 2), (4, 4), 
   (5, 1), (5, 5), (6, 1), (6, 2), (6, 3), (6, 6)}

theorem probability_m_eq_kn (k : ℕ) (h : k > 0) :
  (events_satisfying_condition.card : ℚ) / (valid_rolls.card : ℚ) = 7/18 := by
  sorry

end probability_m_eq_kn_l278_278130


namespace percent_not_participating_music_sports_l278_278098

theorem percent_not_participating_music_sports
  (total_students : ℕ) 
  (both : ℕ) 
  (music_only : ℕ) 
  (sports_only : ℕ) 
  (not_participating : ℕ)
  (percentage_not_participating : ℝ) :
  total_students = 50 →
  both = 5 →
  music_only = 15 →
  sports_only = 20 →
  not_participating = total_students - (both + music_only + sports_only) →
  percentage_not_participating = (not_participating : ℝ) / (total_students : ℝ) * 100 →
  percentage_not_participating = 20 :=
by
  sorry

end percent_not_participating_music_sports_l278_278098


namespace tan_square_proof_l278_278173

theorem tan_square_proof (θ : ℝ) (h : Real.tan θ = 2) : 
  1 / (Real.sin θ ^ 2 - Real.cos θ ^ 2) = 5 / 3 := by
  sorry

end tan_square_proof_l278_278173


namespace hyperbola_equation_l278_278489

-- Definitions for a given hyperbola
variables {a b : ℝ}
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Definitions for the asymptote condition
axiom point_on_asymptote : (4 : ℝ) = (b / a) * 3

-- Definitions for the focal distance condition
axiom point_circle_intersect : (3 : ℝ)^2 + 4^2 = (a^2 + b^2)

-- The goal is to prove the hyperbola's specific equation
theorem hyperbola_equation : 
  (a^2 = 9 ∧ b^2 = 16) →
  (∃ a b : ℝ, (4 : ℝ)^2 + 3^2 = (a^2 + b^2) ∧ 
               (4 : ℝ) = (b / a) * 3 ∧ 
               ((a^2 = 9) ∧ (b^2 = 16)) ∧ (a > 0) ∧ (b > 0)) :=
sorry

end hyperbola_equation_l278_278489


namespace largest_non_sum_of_composites_l278_278789

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278789


namespace red_ball_probability_l278_278228

-- Definitions based on conditions
def numBallsA := 10
def redBallsA := 5
def greenBallsA := numBallsA - redBallsA

def numBallsBC := 10
def redBallsBC := 7
def greenBallsBC := numBallsBC - redBallsBC

def probSelectContainer := 1 / 3
def probRedBallA := redBallsA / numBallsA
def probRedBallBC := redBallsBC / numBallsBC

-- Theorem statement to be proved
theorem red_ball_probability : (probSelectContainer * probRedBallA) + (probSelectContainer * probRedBallBC) + (probSelectContainer * probRedBallBC) = 4 / 5 := 
sorry

end red_ball_probability_l278_278228


namespace largest_cannot_be_sum_of_two_composites_l278_278793

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278793


namespace factor_polynomial_l278_278314

theorem factor_polynomial :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 = 
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := 
by sorry

end factor_polynomial_l278_278314


namespace polynomial_remainder_l278_278318

theorem polynomial_remainder (x : ℂ) : 
  (3 * x ^ 1010 + x ^ 1000) % (x ^ 2 + 1) * (x - 1) = 3 * x ^ 2 + 1 := 
sorry

end polynomial_remainder_l278_278318


namespace rate_of_grapes_l278_278016

theorem rate_of_grapes (G : ℝ) 
  (h_grapes : 8 * G + 9 * 60 = 1100) : 
  G = 70 := 
by
  sorry

end rate_of_grapes_l278_278016


namespace B_completion_time_l278_278426

theorem B_completion_time (A_days : ℕ) (A_efficiency_multiple : ℝ) (B_days_correct : ℝ) :
  A_days = 15 →
  A_efficiency_multiple = 1.8 →
  B_days_correct = 4 + 1 / 6 →
  B_days_correct = 25 / 6 :=
sorry

end B_completion_time_l278_278426


namespace smallest_num_rectangles_to_cover_square_l278_278731

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l278_278731


namespace max_apartment_size_l278_278956

theorem max_apartment_size (rental_price_per_sqft : ℝ) (budget : ℝ) (h1 : rental_price_per_sqft = 1.20) (h2 : budget = 720) : 
  budget / rental_price_per_sqft = 600 :=
by 
  sorry

end max_apartment_size_l278_278956


namespace winter_sales_l278_278942

theorem winter_sales (spring_sales summer_sales fall_sales : ℕ) (fall_sales_pct : ℝ) (total_sales winter_sales : ℕ) :
  spring_sales = 6 →
  summer_sales = 7 →
  fall_sales = 5 →
  fall_sales_pct = 0.20 →
  fall_sales = ⌊fall_sales_pct * total_sales⌋ →
  total_sales = spring_sales + summer_sales + fall_sales + winter_sales →
  winter_sales = 7 :=
by
  sorry

end winter_sales_l278_278942


namespace dorothy_profit_l278_278604

def cost_to_buy_ingredients : ℕ := 53
def number_of_doughnuts : ℕ := 25
def selling_price_per_doughnut : ℕ := 3

def revenue : ℕ := number_of_doughnuts * selling_price_per_doughnut
def profit : ℕ := revenue - cost_to_buy_ingredients

theorem dorothy_profit : profit = 22 :=
by
  -- calculation steps
  sorry

end dorothy_profit_l278_278604


namespace real_solutions_l278_278163

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end real_solutions_l278_278163


namespace odd_distinct_digit_count_l278_278029

theorem odd_distinct_digit_count : 
  let is_good_number (n : ℕ) : Prop :=
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 2 = 1) ∧ 
    ((toString n).to_list.nodup) 
  in 
  (∃ count : ℕ, count = 2240 ∧ (∀ n : ℕ, is_good_number n → n < count)) :=
sorry

end odd_distinct_digit_count_l278_278029


namespace lucy_groceries_total_l278_278936

theorem lucy_groceries_total (cookies noodles : ℕ) (h1 : cookies = 12) (h2 : noodles = 16) : cookies + noodles = 28 :=
by
  sorry

end lucy_groceries_total_l278_278936


namespace number_is_375_l278_278034

theorem number_is_375 (x : ℝ) (h : (40 / 100) * x = (30 / 100) * 50) : x = 37.5 :=
sorry

end number_is_375_l278_278034


namespace probability_of_next_satisfied_customer_l278_278590

noncomputable def probability_of_satisfied_customer : ℝ :=
  let p := (0.8 : ℝ)
  let q := (0.15 : ℝ)
  let neg_reviews := (60 : ℝ)
  let pos_reviews := (20 : ℝ)
  p / (p + q) * (q / (q + p))

theorem probability_of_next_satisfied_customer :
  probability_of_satisfied_customer = 0.64 :=
sorry

end probability_of_next_satisfied_customer_l278_278590


namespace probability_of_distance_less_than_8000_l278_278403

-- Define distances between cities

noncomputable def distances : List (String × String × ℕ) :=
  [("Bangkok", "Cape Town", 6300),
   ("Bangkok", "Honolulu", 7609),
   ("Bangkok", "London", 5944),
   ("Bangkok", "Tokyo", 2870),
   ("Cape Town", "Honolulu", 11535),
   ("Cape Town", "London", 5989),
   ("Cape Town", "Tokyo", 13400),
   ("Honolulu", "London", 7240),
   ("Honolulu", "Tokyo", 3805),
   ("London", "Tokyo", 5950)]

-- Define the total number of pairs and the pairs with distances less than 8000 miles

noncomputable def total_pairs : ℕ := 10
noncomputable def pairs_less_than_8000 : ℕ := 7

-- Define the statement of the probability being 7/10
theorem probability_of_distance_less_than_8000 :
  pairs_less_than_8000 / total_pairs = 7 / 10 :=
by
  sorry

end probability_of_distance_less_than_8000_l278_278403


namespace like_terms_exponents_l278_278995

theorem like_terms_exponents (m n : ℤ) 
  (h1 : m - 1 = 1) 
  (h2 : m + n = 3) : 
  m = 2 ∧ n = 1 :=
by 
  sorry

end like_terms_exponents_l278_278995


namespace negation_of_p_l278_278626

def p := ∀ x : ℝ, x^2 ≥ 0

theorem negation_of_p : ¬p = (∃ x : ℝ, x^2 < 0) :=
  sorry

end negation_of_p_l278_278626


namespace twelve_star_three_eq_four_star_eight_eq_star_assoc_l278_278107

def star (a b : ℕ) : ℕ := 10^a * 10^b

theorem twelve_star_three_eq : star 12 3 = 10^15 :=
by 
  -- Proof here
  sorry

theorem four_star_eight_eq : star 4 8 = 10^12 :=
by 
  -- Proof here
  sorry

theorem star_assoc (a b c : ℕ) : star (a + b) c = star a (b + c) :=
by 
  -- Proof here
  sorry

end twelve_star_three_eq_four_star_eight_eq_star_assoc_l278_278107


namespace cost_of_building_fence_l278_278935

-- Define the conditions
def area : ℕ := 289
def price_per_foot : ℕ := 60

-- Define the length of one side of the square (since area = side^2)
def side_length (a : ℕ) : ℕ := Nat.sqrt a

-- Define the perimeter of the square (since square has 4 equal sides)
def perimeter (s : ℕ) : ℕ := 4 * s

-- Define the cost of building the fence
def cost (p : ℕ) (ppf : ℕ) : ℕ := p * ppf

-- Prove that the cost of building the fence is Rs. 4080
theorem cost_of_building_fence : cost (perimeter (side_length area)) price_per_foot = 4080 := by
  -- Skip the proof steps
  sorry

end cost_of_building_fence_l278_278935


namespace part_a_part_b_l278_278313

/-- Two equally skilled chess players with p = 0.5, q = 0.5. -/
def p : ℝ := 0.5
def q : ℝ := 0.5

-- Definition for binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial distribution
def P (n k : ℕ) : ℝ := (binomial_coeff n k) * (p^k) * (q^(n-k))

/-- Prove that the probability of winning one out of two games is greater than the probability of winning two out of four games -/
theorem part_a : (P 2 1) > (P 4 2) := sorry

/-- Prove that the probability of winning at least two out of four games is greater than the probability of winning at least three out of five games -/
theorem part_b : (P 4 2 + P 4 3 + P 4 4) > (P 5 3 + P 5 4 + P 5 5) := sorry

end part_a_part_b_l278_278313


namespace largest_non_representable_as_sum_of_composites_l278_278799

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278799


namespace profit_percent_l278_278922

variable (P C : ℝ)
variable (h₁ : (2/3) * P = 0.84 * C)

theorem profit_percent (P C : ℝ) (h₁ : (2/3) * P = 0.84 * C) : 
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_l278_278922


namespace distinct_solutions_count_l278_278021

theorem distinct_solutions_count : ∀ (x : ℝ), (|x - 5| = |x + 3| ↔ x = 1) → ∃! x, |x - 5| = |x + 3| :=
by
  intro x h
  existsi 1
  rw h
  sorry 

end distinct_solutions_count_l278_278021


namespace B_pow_101_eq_B_l278_278657

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]

-- State the theorem
theorem B_pow_101_eq_B : B^101 = B :=
  sorry

end B_pow_101_eq_B_l278_278657


namespace pig_problem_l278_278505

theorem pig_problem (x y : ℕ) (h₁ : y - 100 = 100 * x) (h₂ : y = 90 * x) : x = 10 ∧ y = 900 := 
by
  sorry

end pig_problem_l278_278505


namespace cost_of_first_book_l278_278677

-- Define the initial amount of money Shelby had.
def initial_amount : ℕ := 20

-- Define the cost of the second book.
def cost_of_second_book : ℕ := 4

-- Define the cost of one poster.
def cost_of_poster : ℕ := 4

-- Define the number of posters bought.
def num_posters : ℕ := 2

-- Define the total cost that Shelby had to spend on posters.
def total_cost_of_posters : ℕ := num_posters * cost_of_poster

-- Define the total amount spent on books and posters.
def total_spent (X : ℕ) : ℕ := X + cost_of_second_book + total_cost_of_posters

-- Prove that the cost of the first book is 8 dollars.
theorem cost_of_first_book (X : ℕ) (h : total_spent X = initial_amount) : X = 8 :=
by
  sorry

end cost_of_first_book_l278_278677


namespace negation_proposition_l278_278252

theorem negation_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 + x_0 - 2 < 0) ↔ ∀ x_0 : ℝ, x_0^2 + x_0 - 2 ≥ 0 :=
by
  sorry

end negation_proposition_l278_278252


namespace henry_earnings_correct_l278_278855

-- Define constants for the amounts earned per task
def earn_per_lawn : Nat := 5
def earn_per_leaves : Nat := 10
def earn_per_driveway : Nat := 15

-- Define constants for the number of tasks he actually managed to do
def lawns_mowed : Nat := 5
def leaves_raked : Nat := 3
def driveways_shoveled : Nat := 2

-- Define the expected total earnings calculation
def expected_earnings : Nat :=
  (lawns_mowed * earn_per_lawn) +
  (leaves_raked * earn_per_leaves) +
  (driveways_shoveled * earn_per_driveway)

-- State the theorem that the total earnings are 85 dollars.
theorem henry_earnings_correct : expected_earnings = 85 :=
by
  sorry

end henry_earnings_correct_l278_278855


namespace carla_book_count_l278_278263

theorem carla_book_count (tiles_count books_count : ℕ) 
  (tiles_monday : tiles_count = 38)
  (total_tuesday_count : 2 * tiles_count + 3 * books_count = 301) : 
  books_count = 75 :=
by
  sorry

end carla_book_count_l278_278263


namespace find_f_of_given_g_and_odd_l278_278988

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end find_f_of_given_g_and_odd_l278_278988


namespace line_through_point_trangle_area_line_with_given_slope_l278_278010

theorem line_through_point_trangle_area (k : ℝ) (b : ℝ) : 
  (∃ k, (∀ x y, y = k * (x + 3) + 4 ∧ (1 / 2) * (abs (3 * k + 4) * abs (-4 / k - 3)) = 3)) → 
  (∃ k₁ k₂, k₁ = -2/3 ∧ k₂ = -8/3 ∧ 
    (∀ x y, y = k₁ * (x + 3) + 4 → 2 * x + 3 * y - 6 = 0) ∧ 
    (∀ x y, y = k₂ * (x + 3) + 4 → 8 * x + 3 * y + 12 = 0)) := 
sorry

theorem line_with_given_slope (b : ℝ) : 
  (∀ x y, y = (1 / 6) * x + b) → (1 / 2) * abs (6 * b * b) = 3 → 
  (b = 1 ∨ b = -1) → (∀ x y, (b = 1 → x - 6 * y + 6 = 0 ∧ b = -1 → x - 6 * y - 6 = 0)) := 
sorry

end line_through_point_trangle_area_line_with_given_slope_l278_278010


namespace distinct_solutions_eq_l278_278018

theorem distinct_solutions_eq : ∃! x : ℝ, abs (x - 5) = abs (x + 3) :=
by
  sorry

end distinct_solutions_eq_l278_278018


namespace rectangle_width_l278_278911

theorem rectangle_width (width : ℝ) : 
  ∃ w, w = 14 ∧
  (∀ length : ℝ, length = 10 →
  (2 * (length + width) = 3 * 16)) → 
  width = w :=
by
  sorry

end rectangle_width_l278_278911


namespace max_books_borrowed_l278_278874

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ)
  (two_books : ℕ) (at_least_three_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 35 →
  no_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books_per_student = 2 →
  total_students - (no_books + one_book + two_books) = at_least_three_books →
  ∃ max_books_borrowed_by_individual, max_books_borrowed_by_individual = 8 :=
by
  intros h_total_students h_no_books h_one_book h_two_books h_avg_books_per_student h_remaining_students
  -- Skipping the proof steps
  sorry

end max_books_borrowed_l278_278874


namespace find_real_solutions_l278_278166

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end find_real_solutions_l278_278166


namespace proper_fraction_and_condition_l278_278167

theorem proper_fraction_and_condition (a b : ℤ) (h1 : 1 < a) (h2 : b = 2 * a - 1) :
  0 < a ∧ a < b ∧ (a - 1 : ℚ) / (b - 1) = 1 / 2 :=
by
  sorry

end proper_fraction_and_condition_l278_278167


namespace largest_non_summable_composite_l278_278809

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278809


namespace intersection_complement_l278_278387

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 2}
def complement_U_B : Set ℤ := {x ∈ U | x ∉ B}

theorem intersection_complement :
  A ∩ complement_U_B = {0, 1} :=
by
  sorry

end intersection_complement_l278_278387


namespace sequence_unique_l278_278063

theorem sequence_unique (n : ℕ) (h1 : n > 1)
  (x : ℕ → ℕ)
  (hx1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j < n → x i < x j)
  (hx2 : ∀ i, 1 ≤ i ∧ i < n → x i + x (n - i) = 2 * n)
  (hx3 : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j < n ∧ x i + x j < 2 * n →
    ∃ k, 1 ≤ k ∧ k < n ∧ x i + x j = x k) :
  ∀ k, 1 ≤ k ∧ k < n → x k = 2 * k :=
by
  sorry

end sequence_unique_l278_278063


namespace problem1_problem2_problem3_l278_278546

-- Definitions
def num_students := 8
def num_males := 4
def num_females := 4

-- Problem statements
-- (1) Number of arrangements where male student A and female student B stand next to each other is 10080
theorem problem1 (A B : Fin num_students) (h1 : A < 4) (h2 : 4 ≤ B ∧ B < 8) : 
    ((num_males + num_females) - 1)! * 2 = 10080 :=
by sorry

-- (2) Number of arrangements where the order of male student A and female student B is fixed is 20160
theorem problem2 : 
    num_students! / 2 = 20160 :=
by sorry

-- (3) Number of arrangements where female student A does not stand at either end, and among the 4 male students, exactly two of them stand next to each other is 13824
theorem problem3 (A : Fin num_females) :
    3! * C(num_males, 2) * 2! * (num_males - 1)! * 5 = 13824 :=
by sorry

end problem1_problem2_problem3_l278_278546


namespace symmetric_matrix_diagonal_odd_symmetric_matrix_diagonal_even_l278_278952

theorem symmetric_matrix_diagonal_odd (n : ℕ) (hn : n % 2 = 1) (M : Matrix (Fin n) (Fin n) ℕ) 
  (hM1 : ∀ i j : Fin n, M i j ∈ Fin.succ n) 
  (hM2 : ∀ i j : Fin n, (M i j = M j i))
  (h_rows : ∀ i, (Finset.univ.image (M i)).card = n)
  (h_cols : ∀ j, (Finset.univ.image (fun i => M i j)).card = n) :
  ∀ k : ℕ, k ∈ Fin.succ n → ∃ i : Fin n, M i i = k :=
by sorry

theorem symmetric_matrix_diagonal_even (n : ℕ) (hn : n % 2 = 0) (M : Matrix (Fin n) (Fin n) ℕ) 
  (hM1 : ∀ i j : Fin n, M i j ∈ Fin.succ n) 
  (hM2 : ∀ i j : Fin n, (M i j = M j i))
  (h_rows : ∀ i, (Finset.univ.image (M i)).card = n)
  (h_cols : ∀ j, (Finset.univ.image (fun i => M i j)).card = n) :
  ¬ ∀ k : ℕ, k ∈ Fin.succ n → ∃ i : Fin n, M i i = k :=
by sorry

end symmetric_matrix_diagonal_odd_symmetric_matrix_diagonal_even_l278_278952


namespace no_arithmetic_seq_with_sum_n_cubed_l278_278783

theorem no_arithmetic_seq_with_sum_n_cubed (a1 d : ℕ) :
  ¬ (∀ (n : ℕ), (n > 0) → (n / 2) * (2 * a1 + (n - 1) * d) = n^3) :=
sorry

end no_arithmetic_seq_with_sum_n_cubed_l278_278783


namespace probability_even_sum_of_three_dice_l278_278413

-- Conditions
noncomputable def die_faces : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def even_faces : Set ℕ := {x ∈ die_faces | x % 2 = 0}
def odd_faces : Set ℕ := {x ∈ die_faces | x % 2 = 1}

-- Definitions of probabilities
noncomputable def prob_even : ℚ := 5 / 9
noncomputable def prob_odd : ℚ := 4 / 9

-- Question and Correct Answer
theorem probability_even_sum_of_three_dice : 
  (prob_even^3 + 3 * (prob_odd^2 * prob_even) + 3 * (prob_odd * (prob_even^2))) = 665 / 729 :=
by
  sorry

end probability_even_sum_of_three_dice_l278_278413


namespace least_three_digit_with_product_l278_278267

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_product (n : ℕ) (p : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 * d2 * d3 = p

theorem least_three_digit_with_product (p : ℕ) : ∃ n : ℕ, is_three_digit n ∧ digits_product n p ∧ 
  ∀ m : ℕ, is_three_digit m ∧ digits_product m p → n ≤ m :=
by
  use 116
  sorry

end least_three_digit_with_product_l278_278267


namespace find_a6_plus_a7_plus_a8_l278_278213

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l278_278213


namespace car_cost_l278_278753

/--
A group of six friends planned to buy a car. They plan to share the cost equally. 
They had a car wash to help raise funds, which would be taken out of the total cost. 
The remaining cost would be split between the six friends. At the car wash, they earn $500. 
However, Brad decided not to join in the purchase of the car, and now each friend has to pay $40 more. 
What is the cost of the car?
-/
theorem car_cost 
  (C : ℝ) 
  (h1 : 6 * ((C - 500) / 5) = 5 * (C / 6 + 40)) : 
  C = 4200 := 
by 
  sorry

end car_cost_l278_278753


namespace probability_best_play_wins_l278_278758

noncomputable def prob_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) : ℝ :=
  let C := Nat.choose in
  (1 : ℝ) / (C (2 * n) n * C (2 * n) (2 * m)) * ∑ q in Finset.range (2 * m + 1),
  (C n q * C n (2 * m - q)) * 
  ∑ t in Finset.range (min q m),
  (C q t * C (2 * n - q) (n - t))

-- A theorem statement in Lean to ensure proper type checks and conditions 
theorem probability_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) :
  ∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t)) 
  =
  (∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t) )) * 
  (nat.choose (2 * n) n * nat.choose (2 * n) (2 * m)) :=
sorry

end probability_best_play_wins_l278_278758


namespace sum_of_digits_of_N_l278_278441

theorem sum_of_digits_of_N :
  (∃ N : ℕ, 3 * N * (N + 1) / 2 = 3825 ∧ (N.digits 10).sum = 5) :=
by
  sorry

end sum_of_digits_of_N_l278_278441


namespace floor_sum_eq_55_l278_278380

noncomputable def x : ℝ := 9.42

theorem floor_sum_eq_55 : ∀ (x : ℝ), x = 9.42 → (⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋) = 55 := by
  intros
  sorry

end floor_sum_eq_55_l278_278380


namespace euclidean_remainder_2022_l278_278709

theorem euclidean_remainder_2022 : 
  (2022 ^ (2022 ^ 2022)) % 11 = 5 := 
by sorry

end euclidean_remainder_2022_l278_278709


namespace problem1_solution_problem2_solution_l278_278897

-- Problem 1
theorem problem1_solution (x : ℝ) : (2 * x - 3) * (x + 1) < 0 ↔ (-1 < x) ∧ (x < 3 / 2) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) : (4 * x - 1) / (x + 2) ≥ 0 ↔ (x < -2) ∨ (x >= 1 / 4) :=
sorry

end problem1_solution_problem2_solution_l278_278897


namespace inequality_solution_l278_278901

theorem inequality_solution {x : ℝ} : (x + 1) / x > 1 ↔ x > 0 := 
sorry

end inequality_solution_l278_278901


namespace largest_natural_number_not_sum_of_two_composites_l278_278810

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278810


namespace rectangle_area_l278_278089

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end rectangle_area_l278_278089


namespace supplementary_angles_difference_l278_278695

theorem supplementary_angles_difference 
  (x : ℝ) 
  (h1 : 5 * x + 3 * x = 180) 
  (h2 : 0 < x) : 
  abs (5 * x - 3 * x) = 45 :=
by sorry

end supplementary_angles_difference_l278_278695


namespace joy_remaining_tape_l278_278377

theorem joy_remaining_tape (total_tape length width : ℕ) (h_total_tape : total_tape = 250) (h_length : length = 60) (h_width : width = 20) :
  total_tape - 2 * (length + width) = 90 :=
by
  sorry

end joy_remaining_tape_l278_278377


namespace problem_l278_278866

theorem problem (p q : ℕ) (hp: p > 1) (hq: q > 1) (h1 : (2 * p - 1) % q = 0) (h2 : (2 * q - 1) % p = 0) : p + q = 8 := 
sorry

end problem_l278_278866


namespace perpendicular_iff_zero_dot_product_l278_278346

open Real

def a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem perpendicular_iff_zero_dot_product (m : ℝ) :
  dot_product (a m) (b m) = 0 → m = -1 / 3 :=
by
  sorry

end perpendicular_iff_zero_dot_product_l278_278346


namespace counting_divisibles_by_8_l278_278856

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278856


namespace smallest_rectangle_area_l278_278708

-- Definitions based on conditions
def diameter : ℝ := 10
def length : ℝ := diameter
def width : ℝ := diameter + 2

-- Theorem statement
theorem smallest_rectangle_area : (length * width) = 120 :=
by
  -- The proof would go here, but we provide sorry for now
  sorry

end smallest_rectangle_area_l278_278708


namespace functional_equation_solution_l278_278296

noncomputable def f : ℝ → ℝ := sorry 

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) →
  10 * f 2006 + f 0 = 20071 :=
by
  intros h
  sorry

end functional_equation_solution_l278_278296


namespace sin_subtract_pi_over_6_l278_278212

theorem sin_subtract_pi_over_6 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hcos : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_subtract_pi_over_6_l278_278212


namespace rise_in_water_level_correct_l278_278299

noncomputable def volume_of_rectangular_solid (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def area_of_circular_base (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

noncomputable def rise_in_water_level (solid_volume base_area : ℝ) : ℝ :=
  solid_volume / base_area

theorem rise_in_water_level_correct :
  let l := 10
  let w := 12
  let h := 15
  let d := 18
  let solid_volume := volume_of_rectangular_solid l w h
  let base_area := area_of_circular_base d
  let expected_rise := 7.07
  abs (rise_in_water_level solid_volume base_area - expected_rise) < 0.01 
:= 
by {
  sorry
}

end rise_in_water_level_correct_l278_278299


namespace find_a_l278_278080

theorem find_a (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 20) (h2 : (56831742 - a) % 17 = 0) : a = 2 :=
by
  sorry

end find_a_l278_278080


namespace children_distribution_l278_278157

theorem children_distribution (a b c d N : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : a + b + c + d < 18) 
  (h5 : a * b * c * d = N) : 
  N = 120 ∧ a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 := 
by 
  sorry

end children_distribution_l278_278157


namespace combined_alloy_tin_amount_l278_278430

theorem combined_alloy_tin_amount
  (weight_A weight_B weight_C : ℝ)
  (ratio_lead_tin_A : ℝ)
  (ratio_tin_copper_B : ℝ)
  (ratio_copper_tin_C : ℝ)
  (amount_tin : ℝ) :
  weight_A = 150 → weight_B = 200 → weight_C = 250 →
  ratio_lead_tin_A = 5/3 → ratio_tin_copper_B = 2/3 → ratio_copper_tin_C = 4 →
  amount_tin = ((3/8) * weight_A) + ((2/5) * weight_B) + ((1/5) * weight_C) →
  amount_tin = 186.25 :=
by sorry

end combined_alloy_tin_amount_l278_278430


namespace find_integer_pairs_l278_278316

theorem find_integer_pairs (x y : ℕ) (h : x ^ 5 = y ^ 5 + 10 * y ^ 2 + 20 * y + 1) : (x, y) = (1, 0) :=
  sorry

end find_integer_pairs_l278_278316


namespace assignment_plans_proof_l278_278197

noncomputable def total_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let positions := ["translation", "tour guide", "etiquette", "driver"]
  -- Definitions for eligible volunteers for the first two positions
  let first_positions := ["Xiao Zhang", "Xiao Zhao"]
  let remaining_positions := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Assume the computation for the exact number which results in 36
  36

theorem assignment_plans_proof : total_assignment_plans = 36 := 
  by 
  -- Proof skipped
  sorry

end assignment_plans_proof_l278_278197


namespace least_multiplier_l278_278934

theorem least_multiplier (x: ℕ) (h1: 72 * x % 112 = 0) (h2: ∀ y, 72 * y % 112 = 0 → x ≤ y) : x = 14 :=
sorry

end least_multiplier_l278_278934


namespace sum_of_products_of_three_numbers_l278_278101

theorem sum_of_products_of_three_numbers
    (a b c : ℝ)
    (h1 : a^2 + b^2 + c^2 = 179)
    (h2 : a + b + c = 21) :
  ab + bc + ac = 131 :=
by
  -- Proof goes here
  sorry

end sum_of_products_of_three_numbers_l278_278101


namespace smallest_number_of_rectangles_l278_278713

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l278_278713


namespace floor_sum_proof_l278_278384

noncomputable def floor_sum (x y z w : ℝ) : ℝ :=
  x + y + z + w

theorem floor_sum_proof
  (x y z w : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hw_pos : 0 < w)
  (h1 : x^2 + y^2 = 2010)
  (h2 : z^2 + w^2 = 2010)
  (h3 : x * z = 1008)
  (h4 : y * w = 1008) :
  ⌊floor_sum x y z w⌋ = 126 :=
by
  sorry

end floor_sum_proof_l278_278384


namespace x_squared_y_plus_xy_squared_l278_278354

-- Define the variables and their conditions
variables {x y : ℝ}

-- Define the theorem stating that if xy = 3 and x + y = 5, then x^2y + xy^2 = 15
theorem x_squared_y_plus_xy_squared (h1 : x * y = 3) (h2 : x + y = 5) : x^2 * y + x * y^2 = 15 :=
by {
  sorry
}

end x_squared_y_plus_xy_squared_l278_278354


namespace parabola_focus_distance_l278_278298

theorem parabola_focus_distance
  (p : ℝ) (h : p > 0)
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 = 3 - p / 2) 
  (h2 : x2 = 2 - p / 2)
  (h3 : y1^2 = 2 * p * x1)
  (h4 : y2^2 = 2 * p * x2)
  (h5 : y1^2 / y2^2 = x1 / x2) : 
  p = 12 / 5 := 
sorry

end parabola_focus_distance_l278_278298


namespace miniature_tower_height_l278_278312

-- Definitions of conditions
def actual_tower_height := 60
def actual_dome_volume := 200000 -- in liters
def miniature_dome_volume := 0.4 -- in liters

-- Goal: Prove the height of the miniature tower
theorem miniature_tower_height
  (actual_tower_height: ℝ)
  (actual_dome_volume: ℝ)
  (miniature_dome_volume: ℝ) : 
  actual_tower_height = 60 ∧ actual_dome_volume = 200000 ∧ miniature_dome_volume = 0.4 →
  (actual_tower_height / ( (actual_dome_volume / miniature_dome_volume)^(1/3) )) = 1.2 :=
by
  sorry

end miniature_tower_height_l278_278312


namespace arcsin_one_eq_pi_div_two_l278_278961

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end arcsin_one_eq_pi_div_two_l278_278961


namespace total_number_of_eggs_l278_278545

theorem total_number_of_eggs 
  (cartons : ℕ) 
  (eggs_per_carton_length : ℕ) 
  (eggs_per_carton_width : ℕ)
  (egg_position_from_front : ℕ)
  (egg_position_from_back : ℕ)
  (egg_position_from_left : ℕ)
  (egg_position_from_right : ℕ) :
  cartons = 28 →
  egg_position_from_front = 14 →
  egg_position_from_back = 20 →
  egg_position_from_left = 3 →
  egg_position_from_right = 2 →
  eggs_per_carton_length = egg_position_from_front + egg_position_from_back - 1 →
  eggs_per_carton_width = egg_position_from_left + egg_position_from_right - 1 →
  cartons * (eggs_per_carton_length * eggs_per_carton_width) = 3696 := 
  by 
  intros
  sorry

end total_number_of_eggs_l278_278545


namespace cartons_per_box_l278_278142

open Nat

theorem cartons_per_box (cartons packs sticks brown_boxes total_sticks : ℕ) 
  (h1 : cartons * (packs * sticks) * brown_boxes = total_sticks) 
  (h2 : packs = 5) 
  (h3 : sticks = 3) 
  (h4 : brown_boxes = 8) 
  (h5 : total_sticks = 480) :
  cartons = 4 := 
by 
  sorry

end cartons_per_box_l278_278142


namespace largest_n_satisfies_l278_278982

noncomputable def sin_plus_cos_bound (n : ℕ) (x : ℝ) : Prop :=
  (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt n)

theorem largest_n_satisfies :
  ∃ (n : ℕ), (∀ x : ℝ, sin_plus_cos_bound n x) ∧
  ∀ m : ℕ, (∀ x : ℝ, sin_plus_cos_bound m x) → m ≤ 2 := 
sorry

end largest_n_satisfies_l278_278982


namespace age_ratio_l278_278697

theorem age_ratio (R D : ℕ) (hR : R + 4 = 32) (hD : D = 21) : R / D = 4 / 3 := 
by sorry

end age_ratio_l278_278697


namespace Marty_combinations_l278_278665

theorem Marty_combinations : 
  let colors := 4
  let decorations := 3
  colors * decorations = 12 :=
by
  sorry

end Marty_combinations_l278_278665


namespace problem_proof_l278_278882

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

def num_multiples_of_lt (m bound : ℕ) : ℕ :=
  (bound - 1) / m

-- Definitions for the conditions
def a := num_multiples_of_lt 8 40
def b := num_multiples_of_lt 8 40

-- Proof statement
theorem problem_proof : (a - b)^3 = 0 := by
  sorry

end problem_proof_l278_278882


namespace cover_square_with_rectangles_l278_278718

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l278_278718


namespace point_Q_probability_l278_278576

noncomputable def rect_region : set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

def point_closer_to_p1_than_p2 (p q₁ q₂ : ℝ × ℝ) : Prop :=
  (p.1 - q₁.1)^2 + (p.2 - q₁.2)^2 < (p.1 - q₂.1)^2 + (p.2 - q₂.2)^2

theorem point_Q_probability:
  let Q : pmf (ℝ × ℝ) := pmf.of_finset { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 } (by sorry)
  Q { p | point_closer_to_p1_than_p2 p (1, 1) (4, 2) } = 1 / 2 := 
by 
  sorry

end point_Q_probability_l278_278576


namespace improper_fraction_2012a_div_b_l278_278740

theorem improper_fraction_2012a_div_b
  (a b : ℕ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) :
  2012 * a > b :=
by 
  sorry

end improper_fraction_2012a_div_b_l278_278740


namespace avgPairsTriplets_l278_278554

open Finset

noncomputable def numPairsTriplets (s : Finset ℕ) : ℕ :=
  let pairs := s.subsets.card.filter (λ t, t.card = 2 ∧ (t.max' (by simp [t.nonempty]) - t.min' (by simp [t.nonempty]) = 1))
  let triplets := s.subsets.card.filter (λ t, t.card = 3 ∧ (t.max' (by simp [t.nonempty]) - t.min' (by simp [t.nonempty]) = 2))
  pairs + triplets

theorem avgPairsTriplets : 
  let sets := (powerset (range 1 21)).filter (λ s, s.card = 4) in
  (∑ s in sets, numPairsTriplets s : ℝ) / sets.card = 0.2 :=
  sorry

end avgPairsTriplets_l278_278554


namespace correct_multiple_l278_278355

theorem correct_multiple (n : ℝ) (m : ℝ) (h1 : n = 6) (h2 : m * n - 6 = 2 * n) : m * n = 18 :=
by
  sorry

end correct_multiple_l278_278355


namespace evaluate_expression_l278_278408

theorem evaluate_expression : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end evaluate_expression_l278_278408


namespace solve_sum_of_squares_l278_278179

theorem solve_sum_of_squares
  (k l m n a b c : ℕ)
  (h_cond1 : k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n)
  (h_cond2 : a * k^2 - b * k + c = 0)
  (h_cond3 : a * l^2 - b * l + c = 0)
  (h_cond4 : c * m^2 - 16 * b * m + 256 * a = 0)
  (h_cond5 : c * n^2 - 16 * b * n + 256 * a = 0) :
  k^2 + l^2 + m^2 + n^2 = 325 :=
by
  sorry

end solve_sum_of_squares_l278_278179


namespace hyperbola_condition_l278_278997

theorem hyperbola_condition (k : ℝ) (x y : ℝ) :
  (k ≠ 0 ∧ k ≠ 3 ∧ (x^2 / k + y^2 / (k - 3) = 1)) → 0 < k ∧ k < 3 :=
by
  sorry

end hyperbola_condition_l278_278997


namespace infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l278_278900

theorem infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017 :
  ∀ n : ℕ, ∃ m : ℕ, (m ∈ {x | ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 1}) ∧ 2017 ∣ m :=
by
  sorry

end infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l278_278900


namespace sum_of_first_n_odd_integers_l278_278395

theorem sum_of_first_n_odd_integers (n : ℕ) : (∑ i in Finset.range n, (2 * i + 1)) = n^2 :=
by
  sorry

end sum_of_first_n_odd_integers_l278_278395


namespace decimalToFrac_l278_278265

theorem decimalToFrac : (145 / 100 : ℚ) = 29 / 20 := by
  sorry

end decimalToFrac_l278_278265


namespace circles_intersect_if_and_only_if_l278_278484

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6 * x - 8 * y - 11 = 0) ↔ (1 < m ∧ m < 121) :=
by
  sorry

end circles_intersect_if_and_only_if_l278_278484


namespace a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l278_278844

variable {a b c : ℝ}

theorem a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2 :
  ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) :=
sorry

theorem a_gt_b_necessary_not_sufficient_ac2_gt_bc2 :
  ¬((a > b) → (a * c^2 > b * c^2)) ∧ ((a * c^2 > b * c^2) → (a > b)) :=
sorry

end a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l278_278844


namespace functional_eq_solution_l278_278310

noncomputable def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) * f(x - y) = f(x)^2 + f(y)^2 - 1

theorem functional_eq_solution (f : ℝ → ℝ) (c : ℝ) :
  (functional_eq f) →
  (∀ t : ℝ, has_deriv_at f (f t) t) →
  (∃ c : ℝ, ∀ t : ℝ, deriv (deriv f) t = c^2 * f t ∨ deriv (deriv f) t = -c^2 * f t) →
  (∀ t : ℝ, f t = cos (c * t) ∨ f t = cosh (c * t) ∨ f t = - cos (c * t) ∨ f t = - cosh (c * t)) :=
sorry

end functional_eq_solution_l278_278310


namespace pq_sum_l278_278660

open Real

theorem pq_sum (p q : ℝ) (hp : p^3 - 18 * p^2 + 81 * p - 162 = 0) (hq : 4 * q^3 - 24 * q^2 + 45 * q - 27 = 0) :
    p + q = 8 ∨ p + q = 8 + 6 * sqrt 3 ∨ p + q = 8 - 6 * sqrt 3 :=
sorry

end pq_sum_l278_278660


namespace gcd_m_n_l278_278060

def m : ℕ := 555555555
def n : ℕ := 1111111111

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l278_278060


namespace is_quadratic_equation_l278_278113

open Real

-- Define the candidate equations as statements in Lean 4
def equation_A (x : ℝ) : Prop := 3 * x^2 = 1 - 1 / (3 * x)
def equation_B (x m : ℝ) : Prop := (m - 2) * x^2 - m * x + 3 = 0
def equation_C (x : ℝ) : Prop := (x^2 - 3) * (x - 1) = 0
def equation_D (x : ℝ) : Prop := x^2 = 2

-- Prove that among the given equations, equation_D is the only quadratic equation
theorem is_quadratic_equation (x : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_A x = (a * x^2 + b * x + c = 0)) ∨
  (∃ m a b c : ℝ, a ≠ 0 ∧ equation_B x m = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_C x = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_D x = (a * x^2 + b * x + c = 0)) := by
  sorry

end is_quadratic_equation_l278_278113


namespace anniversary_sale_total_cost_l278_278581

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l278_278581


namespace problem_statement_l278_278000

theorem problem_statement (a b : ℝ) (h : (1 / a + 1 / b) / (1 / a - 1 / b) = 2023) : (a + b) / (a - b) = 2023 :=
by
  sorry

end problem_statement_l278_278000


namespace gas_cost_correct_l278_278536

def cost_to_fill_remaining_quarter (initial_fill : ℚ) (final_fill : ℚ) (added_gas : ℚ) (cost_per_litre : ℚ) : ℚ :=
  let tank_capacity := (added_gas * (1 / (final_fill - initial_fill)))
  let remaining_quarter_cost := (tank_capacity * (1 / 4)) * cost_per_litre
  remaining_quarter_cost

theorem gas_cost_correct :
  cost_to_fill_remaining_quarter (1/8) (3/4) 30 1.38 = 16.56 :=
by
  sorry

end gas_cost_correct_l278_278536


namespace probability_of_negative_product_l278_278704

def set_integers : Set ℤ := { -5, -8, -1, 7, 4, 2, -3 }

noncomputable def probability_negative_product : ℚ :=
  let negs := {a | a ∈ set_integers ∧ a < 0}
  let pos := {a | a ∈ set_integers ∧ a > 0}
  let successful_outcomes := negs.card * pos.card
  let total_outcomes := (set_integers.card.choose 2)
  (successful_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_of_negative_product :
  probability_negative_product = 4 / 7 := 
by
  sorry

end probability_of_negative_product_l278_278704


namespace log_sqrt_pi_simplification_l278_278428

theorem log_sqrt_pi_simplification:
  2 * Real.log 4 + Real.log (5 / 8) + Real.sqrt ((Real.sqrt 3 - Real.pi) ^ 2) = 1 + Real.pi - Real.sqrt 3 :=
sorry

end log_sqrt_pi_simplification_l278_278428


namespace cody_ate_dumplings_l278_278597

theorem cody_ate_dumplings (initial_dumplings remaining_dumplings : ℕ) (h1 : initial_dumplings = 14) (h2 : remaining_dumplings = 7) : initial_dumplings - remaining_dumplings = 7 :=
by
  sorry

end cody_ate_dumplings_l278_278597


namespace company_needs_86_workers_l278_278128

def profit_condition (n : ℕ) : Prop :=
  147 * n > 600 + 140 * n

theorem company_needs_86_workers (n : ℕ) : profit_condition n → n ≥ 86 :=
by
  intro h
  sorry

end company_needs_86_workers_l278_278128


namespace arcsin_one_eq_pi_div_two_l278_278964

noncomputable def arcsin (x : ℝ) : ℝ :=
classical.some (exists_inverse_sin x)

theorem arcsin_one_eq_pi_div_two : arcsin 1 = π / 2 :=
sorry

end arcsin_one_eq_pi_div_two_l278_278964


namespace power_of_two_divides_factorial_iff_l278_278073

theorem power_of_two_divides_factorial_iff (n : ℕ) (k : ℕ) : 2^(n - 1) ∣ n! ↔ n = 2^k := sorry

end power_of_two_divides_factorial_iff_l278_278073


namespace four_distinct_real_solutions_l278_278225

noncomputable def polynomial (a b c d e x : ℝ) : ℝ :=
  (x - a) * (x - b) * (x - c) * (x - d) * (x - e)

noncomputable def derivative (a b c d e x : ℝ) : ℝ :=
  (x - b) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - b) * (x - d) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - d)

theorem four_distinct_real_solutions (a b c d e : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
    (derivative a b c d e x1 = 0 ∧ derivative a b c d e x2 = 0 ∧ derivative a b c d e x3 = 0 ∧ derivative a b c d e x4 = 0) :=
sorry

end four_distinct_real_solutions_l278_278225


namespace smallest_number_of_rectangles_l278_278712

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l278_278712


namespace squirrel_climb_l278_278139

-- Define the problem conditions and the goal
variable (x : ℝ)

-- net_distance_climbed_every_two_minutes
def net_distance_climbed_every_two_minutes : ℝ := x - 2

-- distance_climbed_in_14_minutes
def distance_climbed_in_14_minutes : ℝ := 7 * (x - 2)

-- distance_climbed_in_15th_minute
def distance_climbed_in_15th_minute : ℝ := x

-- total_distance_climbed_in_15_minutes
def total_distance_climbed_in_15_minutes : ℝ := 26

-- Theorem: proving x based on the conditions
theorem squirrel_climb : 
  7 * (x - 2) + x = 26 -> x = 5 := by
  intros h
  sorry

end squirrel_climb_l278_278139


namespace sufficient_but_not_necessary_condition_l278_278747

theorem sufficient_but_not_necessary_condition {α : ℝ} :
  (α = π / 6 → Real.sin α = 1 / 2) ∧ (∃ α', α' ≠ π / 6 ∧ Real.sin α' = 1 / 2) :=
by
  split
  { intro h
    rw [h, Real.sin_pi_div_six] 
  }
  {
    use 5 * π / 6
    split
    { linarith }
    rw [Real.sin_of_real 5π / 6]
    norm_num
  }
  sorry


end sufficient_but_not_necessary_condition_l278_278747


namespace range_of_m_l278_278640

-- Defining the conditions
variable (x m : ℝ)

-- The theorem statement
theorem range_of_m (h : ∀ x : ℝ, x < m → 2*x + 1 < 5) : m ≤ 2 := by
  sorry

end range_of_m_l278_278640


namespace sheena_sewing_hours_weekly_l278_278076

theorem sheena_sewing_hours_weekly
  (hours_per_dress : ℕ)
  (number_of_dresses : ℕ)
  (weeks_to_complete : ℕ)
  (total_sewing_hours : ℕ)
  (hours_per_week : ℕ) :
  hours_per_dress = 12 →
  number_of_dresses = 5 →
  weeks_to_complete = 15 →
  total_sewing_hours = number_of_dresses * hours_per_dress →
  hours_per_week = total_sewing_hours / weeks_to_complete →
  hours_per_week = 4 := by
  intros h1 h2 h3 h4 h5
  sorry

end sheena_sewing_hours_weekly_l278_278076


namespace shoes_cost_l278_278389

theorem shoes_cost (S : ℝ) : 
  let suit := 430
  let discount := 100
  let total_paid := 520
  suit + S - discount = total_paid -> 
  S = 190 :=
by 
  intro h
  sorry

end shoes_cost_l278_278389


namespace gumballs_per_pair_of_earrings_l278_278655

theorem gumballs_per_pair_of_earrings : 
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  (total_gumballs / total_earrings) = 9 :=
by
  -- Definitions
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  -- Theorem statement
  sorry

end gumballs_per_pair_of_earrings_l278_278655


namespace number_of_new_trailer_homes_l278_278326

-- Definitions coming from the conditions
def initial_trailers : ℕ := 30
def initial_avg_age : ℕ := 15
def years_passed : ℕ := 5
def current_avg_age : ℕ := initial_avg_age + years_passed

-- Let 'n' be the number of new trailer homes added five years ago
variable (n : ℕ)

def new_trailer_age : ℕ := years_passed
def total_trailers : ℕ := initial_trailers + n
def total_ages : ℕ := (initial_trailers * current_avg_age) + (n * new_trailer_age)
def combined_avg_age := total_ages / total_trailers

theorem number_of_new_trailer_homes (h : combined_avg_age = 12) : n = 34 := 
sorry

end number_of_new_trailer_homes_l278_278326


namespace minimum_value_l278_278339

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (∃ x, (∀ y, y = (1 / a) + (4 / b) → y ≥ x) ∧ x = 9 / 2) :=
by
  sorry

end minimum_value_l278_278339


namespace sum_of_ages_l278_278449

theorem sum_of_ages (a b c : ℕ) 
  (h1 : a = 18 + b + c) 
  (h2 : a^2 = 2016 + (b + c)^2) : 
  a + b + c = 112 := 
sorry

end sum_of_ages_l278_278449


namespace inequality_proof_l278_278338

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end inequality_proof_l278_278338


namespace count_multiples_of_8_in_range_l278_278860

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278860


namespace min_t_of_BE_CF_l278_278651

theorem min_t_of_BE_CF (A B C E F: ℝ)
  (hE_midpoint_AC : ∃ D, D = (A + C) / 2 ∧ E = D)
  (hF_midpoint_AB : ∃ D, D = (A + B) / 2 ∧ F = D)
  (h_AB_AC_ratio : B - A = 2 / 3 * (C - A)) :
  ∃ t : ℝ, t = 7 / 8 ∧ ∀ (BE CF : ℝ), BE = dist B E ∧ CF = dist C F → BE / CF < t := by
  sorry

end min_t_of_BE_CF_l278_278651


namespace distinct_solutions_eq_l278_278017

theorem distinct_solutions_eq : ∃! x : ℝ, abs (x - 5) = abs (x + 3) :=
by
  sorry

end distinct_solutions_eq_l278_278017


namespace cube_volume_in_pyramid_and_cone_l278_278950

noncomputable def volume_of_cube
  (base_side : ℝ)
  (pyramid_height : ℝ)
  (cone_radius : ℝ)
  (cone_height : ℝ)
  (cube_side_length : ℝ) : ℝ := 
  cube_side_length^3

theorem cube_volume_in_pyramid_and_cone :
  let base_side := 2
  let pyramid_height := Real.sqrt 3
  let cone_radius := Real.sqrt 2
  let cone_height := Real.sqrt 3
  let cube_side_length := (Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3)
  volume_of_cube base_side pyramid_height cone_radius cone_height cube_side_length = (6 * Real.sqrt 6) / 17 :=
by sorry

end cube_volume_in_pyramid_and_cone_l278_278950


namespace complex_z_pow_2017_l278_278482

noncomputable def complex_number_z : ℂ := (1 + Complex.I) / (1 - Complex.I)

theorem complex_z_pow_2017 :
  (complex_number_z * (1 - Complex.I) = 1 + Complex.I) → (complex_number_z ^ 2017 = Complex.I) :=
by
  intro h
  sorry

end complex_z_pow_2017_l278_278482


namespace bacteria_colony_growth_l278_278875

theorem bacteria_colony_growth : 
  ∃ (n : ℕ), n = 4 ∧ 5 * 3 ^ n > 200 ∧ (∀ (m : ℕ), 5 * 3 ^ m > 200 → m ≥ n) :=
by
  sorry

end bacteria_colony_growth_l278_278875


namespace tens_digit_of_seven_times_cubed_is_one_l278_278741

-- Variables and definitions
variables (p : ℕ) (h1 : p < 10)

-- Main theorem statement
theorem tens_digit_of_seven_times_cubed_is_one (hp : p < 10) :
  let N := 11 * p
  let m := 7
  let result := m * N^3
  (result / 10) % 10 = 1 := 
sorry

end tens_digit_of_seven_times_cubed_is_one_l278_278741


namespace geometric_sequence_a6_l278_278202

theorem geometric_sequence_a6 (a : ℕ → ℝ) (r : ℝ)
  (h₁ : a 4 = 7)
  (h₂ : a 8 = 63)
  (h_geom : ∀ n, a n = a 1 * r^(n - 1)) :
  a 6 = 21 :=
sorry

end geometric_sequence_a6_l278_278202


namespace rectangle_length_l278_278190

-- Define the area and width of the rectangle as given
def width : ℝ := 4
def area  : ℝ := 28

-- Prove that the length is 7 cm given the conditions
theorem rectangle_length : ∃ length : ℝ, length = 7 ∧ area = length * width :=
sorry

end rectangle_length_l278_278190


namespace distance_between_andrey_and_valentin_l278_278147

-- Definitions based on conditions
def speeds_relation_andrey_boris (a b : ℝ) := b = 0.94 * a
def speeds_relation_boris_valentin (b c : ℝ) := c = 0.95 * b

theorem distance_between_andrey_and_valentin
  (a b c : ℝ)
  (h1 : speeds_relation_andrey_boris a b)
  (h2 : speeds_relation_boris_valentin b c)
  : 1000 - 1000 * c / a = 107 :=
by
  sorry

end distance_between_andrey_and_valentin_l278_278147


namespace total_lawns_mowed_l278_278674

theorem total_lawns_mowed (earned_per_lawn forgotten_lawns total_earned : ℕ) 
    (h1 : earned_per_lawn = 9) 
    (h2 : forgotten_lawns = 8) 
    (h3 : total_earned = 54) : 
    ∃ (total_lawns : ℕ), total_lawns = 14 :=
by
    sorry

end total_lawns_mowed_l278_278674


namespace smallest_multiple_5_711_l278_278745

theorem smallest_multiple_5_711 : ∃ n : ℕ, n = Nat.lcm 5 711 ∧ n = 3555 := 
by
  sorry

end smallest_multiple_5_711_l278_278745


namespace find_m_range_l278_278331

noncomputable def range_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : Prop :=
  m ≥ 4

-- Here is the theorem statement
theorem find_m_range (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : range_m a b c m h1 h2 h3 :=
sorry

end find_m_range_l278_278331


namespace simplify_expression_l278_278831

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 :=
by
  sorry

end simplify_expression_l278_278831


namespace complex_z_eq_neg_i_l278_278065

theorem complex_z_eq_neg_i (z : ℂ) (i : ℂ) (h1 : i * z = 1) (hi : i^2 = -1) : z = -i :=
sorry

end complex_z_eq_neg_i_l278_278065


namespace comprehensiveInvestigation_is_Census_l278_278751

def comprehensiveInvestigation (s: String) : Prop :=
  s = "Census"

theorem comprehensiveInvestigation_is_Census :
  comprehensiveInvestigation "Census" :=
by
  sorry

end comprehensiveInvestigation_is_Census_l278_278751


namespace distance_from_point_to_x_axis_l278_278689

theorem distance_from_point_to_x_axis (x y : ℝ) (hP : x = 3 ∧ y = -4) : abs(y) = 4 := by
  cases hP with
  | intro _ hy =>
    have : y = -4 := hy
    rw [this]
    simp
    sorry

end distance_from_point_to_x_axis_l278_278689


namespace two_digit_numbers_sum_reversed_l278_278600

theorem two_digit_numbers_sum_reversed (a b : ℕ) (h₁ : 0 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : a + b = 12) :
  ∃ n : ℕ, n = 7 := 
sorry

end two_digit_numbers_sum_reversed_l278_278600


namespace vip_seat_cost_is_65_l278_278300

noncomputable def cost_of_VIP_seat (G V_T V : ℕ) (cost : ℕ) : Prop :=
  G + V_T = 320 ∧
  (15 * G + V * V_T = cost) ∧
  V_T = G - 212 → V = 65

theorem vip_seat_cost_is_65 :
  ∃ (G V_T V : ℕ), cost_of_VIP_seat G V_T V 7500 :=
  sorry

end vip_seat_cost_is_65_l278_278300


namespace average_time_relay_race_l278_278781

theorem average_time_relay_race :
  let dawson_time := 38
  let henry_time := 7
  let total_legs := 2
  (dawson_time + henry_time) / total_legs = 22.5 :=
by
  sorry

end average_time_relay_race_l278_278781


namespace avg_and_var_of_scaled_shifted_data_l278_278341

-- Definitions of average and variance
noncomputable def avg (l: List ℝ) : ℝ := (l.sum) / l.length
noncomputable def var (l: List ℝ) : ℝ := (l.map (λ x => (x - avg l) ^ 2)).sum / l.length

theorem avg_and_var_of_scaled_shifted_data
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_avg : avg (List.ofFn x) = 2)
  (h_var : var (List.ofFn x) = 3) :
  avg (List.ofFn (λ i => 2 * x i + 3)) = 7 ∧ var (List.ofFn (λ i => 2 * x i + 3)) = 12 := by
  sorry

end avg_and_var_of_scaled_shifted_data_l278_278341


namespace integer_solutions_count_l278_278031

theorem integer_solutions_count :
  (∃ (n : ℕ), ∀ (x y : ℤ), x^2 + y^2 = 6 * x + 2 * y + 15 → n = 12) :=
by
  sorry

end integer_solutions_count_l278_278031


namespace not_perfect_square_l278_278222

theorem not_perfect_square (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : ¬ (a^2 - b^2) % 4 = 0) : 
  ¬ ∃ k : ℤ, (a + 3*b) * (5*a + 7*b) = k^2 :=
sorry

end not_perfect_square_l278_278222


namespace perpendicular_slope_l278_278322

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end perpendicular_slope_l278_278322


namespace value_of_m_div_x_l278_278279

variables (a b : ℝ) (k : ℝ)
-- Condition: The ratio of a to b is 4 to 5
def ratio_a_to_b : Prop := a / b = 4 / 5

-- Condition: x equals a increased by 75 percent of a
def x := a + 0.75 * a

-- Condition: m equals b decreased by 80 percent of b
def m := b - 0.80 * b

-- Prove the given question
theorem value_of_m_div_x (h1 : ratio_a_to_b a b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  m / x = 1 / 7 := by
sorry

end value_of_m_div_x_l278_278279


namespace parabola_point_distance_condition_l278_278335

theorem parabola_point_distance_condition (k : ℝ) (p : ℝ) (h_p_gt_0 : p > 0) (focus : ℝ × ℝ) (vertex : ℝ × ℝ) :
  vertex = (0, 0) → focus = (0, p/2) → (k^2 = -2 * p * (-2)) → dist (k, -2) focus = 4 → k = 4 ∨ k = -4 :=
by
  sorry

end parabola_point_distance_condition_l278_278335


namespace CoveredAreaIs84_l278_278007

def AreaOfStrip (length width : ℕ) : ℕ :=
  length * width

def TotalAreaWithoutOverlaps (numStrips areaOfOneStrip : ℕ) : ℕ :=
  numStrips * areaOfOneStrip

def OverlapArea (intersectionArea : ℕ) (numIntersections : ℕ) : ℕ :=
  intersectionArea * numIntersections

def ActualCoveredArea (totalArea overlapArea : ℕ) : ℕ :=
  totalArea - overlapArea

theorem CoveredAreaIs84 :
  let length := 12
  let width := 2
  let numStrips := 6
  let intersectionArea := width * width
  let numIntersections := 15
  let areaOfOneStrip := AreaOfStrip length width
  let totalAreaWithoutOverlaps := TotalAreaWithoutOverlaps numStrips areaOfOneStrip
  let totalOverlapArea := OverlapArea intersectionArea numIntersections
  ActualCoveredArea totalAreaWithoutOverlaps totalOverlapArea = 84 :=
by
  sorry

end CoveredAreaIs84_l278_278007


namespace problem_statement_l278_278515

theorem problem_statement (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) →
  (∀ x : ℝ, x > 1 → f x < 0) →

  -- Conclusion 1: f(1) = 0, f(-1) = 0
  f 1 = 0 ∧ f (-1) = 0 ∧

  -- Conclusion 2: f(x) is an odd function: f(-x) = -f(x)
  (∀ x : ℝ, f (-x) = -f x) ∧

  -- Conclusion 3: f(x) is decreasing on (1, +∞)
  (∀ x1 x2 : ℝ, x1 > 1 → x2 > 1 → x1 < x2 → f x1 < f x2) := sorry

end problem_statement_l278_278515


namespace fraction_integer_solution_l278_278383

theorem fraction_integer_solution (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 8) (h₃ : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = -1 := 
sorry

end fraction_integer_solution_l278_278383


namespace min_rectangles_to_cover_square_exactly_l278_278721

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l278_278721


namespace range_of_a_l278_278635

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) : 0 ≤ a := sorry

end range_of_a_l278_278635


namespace roots_of_quadratic_eval_l278_278838

theorem roots_of_quadratic_eval :
  ∀ x₁ x₂ : ℝ, (x₁^2 + 4 * x₁ + 2 = 0) ∧ (x₂^2 + 4 * x₂ + 2 = 0) ∧ (x₁ + x₂ = -4) ∧ (x₁ * x₂ = 2) →
    x₁^3 + 14 * x₂ + 55 = 7 :=
by
  sorry

end roots_of_quadratic_eval_l278_278838


namespace quadratic_solution_eq_l278_278906

theorem quadratic_solution_eq (c d : ℝ) 
  (h_eq : ∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = c ∨ x = d))
  (h_order : c ≥ d) :
  c + 2*d = 9 - Real.sqrt 23 :=
sorry

end quadratic_solution_eq_l278_278906


namespace tips_multiple_l278_278780

variable (A T : ℝ) (x : ℝ)
variable (h1 : T = 7 * A)
variable (h2 : T / 4 = x * A)

theorem tips_multiple (A T : ℝ) (x : ℝ) (h1 : T = 7 * A) (h2 : T / 4 = x * A) : x = 1.75 := by
  sorry

end tips_multiple_l278_278780


namespace candy_weight_reduction_l278_278292

theorem candy_weight_reduction:
  ∀ (W P : ℝ), (33.333333333333314 / 100) * (P / W) = (P / (W - (1/4) * W)) →
  (1 - (W - (1/4) * W) / W) * 100 = 25 :=
by
  intros W P h
  sorry

end candy_weight_reduction_l278_278292


namespace largest_number_not_sum_of_two_composites_l278_278826

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278826


namespace Ryan_reads_more_l278_278561

theorem Ryan_reads_more 
  (total_pages_Ryan : ℕ)
  (days_in_week : ℕ)
  (pages_per_book_brother : ℕ)
  (books_per_day_brother : ℕ)
  (total_pages_brother : ℕ)
  (Ryan_books : ℕ)
  (Ryan_weeks : ℕ)
  (Brother_weeks : ℕ)
  (days_in_week_def : days_in_week = 7)
  (total_pages_Ryan_def : total_pages_Ryan = 2100)
  (pages_per_book_brother_def : pages_per_book_brother = 200)
  (books_per_day_brother_def : books_per_day_brother = 1)
  (Ryan_weeks_def : Ryan_weeks = 1)
  (Brother_weeks_def : Brother_weeks = 1)
  (total_pages_brother_def : total_pages_brother = pages_per_book_brother * days_in_week)
  : ((total_pages_Ryan / days_in_week) - (total_pages_brother / days_in_week) = 100) :=
by
  -- We provide the proof steps
  sorry

end Ryan_reads_more_l278_278561


namespace slices_per_friend_l278_278881

theorem slices_per_friend (n : ℕ) (h1 : n > 0)
    (h2 : ∀ i : ℕ, i < n → (15 + 18 + 20 + 25) = 78 * n) :
    78 = (15 + 18 + 20 + 25) / n := 
by
  sorry

end slices_per_friend_l278_278881


namespace find_c_l278_278323

noncomputable def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3 * x^2 + c * x - 8

theorem find_c (c : ℝ) : (∀ x, P c (x + 2) = 0) → c = -14 :=
sorry

end find_c_l278_278323


namespace third_quadrant_angle_to_fourth_l278_278187

theorem third_quadrant_angle_to_fourth {α : ℝ} (k : ℤ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  -90 - k * 360 < 180 - α ∧ 180 - α < -k * 360 :=
by
  sorry

end third_quadrant_angle_to_fourth_l278_278187


namespace calculation_correct_l278_278422

theorem calculation_correct : 469111 * 9999 = 4690428889 := 
by sorry

end calculation_correct_l278_278422


namespace scientific_notation_of_360_billion_l278_278282

def number_in_scientific_notation (n : ℕ) : String :=
  match n with
  | 360000000000 => "3.6 × 10^11"
  | _ => "Unknown"

theorem scientific_notation_of_360_billion : 
  number_in_scientific_notation 360000000000 = "3.6 × 10^11" :=
by
  -- insert proof steps here
  sorry

end scientific_notation_of_360_billion_l278_278282


namespace time_before_Car_Y_started_in_minutes_l278_278455

noncomputable def timeBeforeCarYStarted (speedX speedY distanceX : ℝ) : ℝ :=
  let t := distanceX / speedX
  (speedY * t - distanceX) / speedX

theorem time_before_Car_Y_started_in_minutes 
  (speedX speedY distanceX : ℝ)
  (h_speedX : speedX = 35)
  (h_speedY : speedY = 70)
  (h_distanceX : distanceX = 42) : 
  (timeBeforeCarYStarted speedX speedY distanceX) * 60 = 72 :=
by
  sorry

end time_before_Car_Y_started_in_minutes_l278_278455


namespace remainder_when_divided_by_198_l278_278244

-- Define the conditions as Hypotheses
variables (x : ℤ)

-- Hypotheses stating the given conditions
def cond1 : Prop := 2 + x ≡ 9 [ZMOD 8]
def cond2 : Prop := 3 + x ≡ 4 [ZMOD 27]
def cond3 : Prop := 11 + x ≡ 49 [ZMOD 1331]

-- Final statement to prove
theorem remainder_when_divided_by_198 (h1 : cond1 x) (h2 : cond2 x) (h3 : cond3 x) : x ≡ 1 [ZMOD 198] := by
  sorry

end remainder_when_divided_by_198_l278_278244


namespace arithmetic_sequence_30th_term_l278_278086

theorem arithmetic_sequence_30th_term (a1 a2 a3 d a30 : ℤ) 
 (h1 : a1 = 3) (h2 : a2 = 12) (h3 : a3 = 21) 
 (h4 : d = a2 - a1) (h5 : a3 = a1 + 2 * d) 
 (h6 : a30 = a1 + 29 * d) : 
 a30 = 264 :=
by
  sorry

end arithmetic_sequence_30th_term_l278_278086


namespace paint_needed_for_720_statues_l278_278634

noncomputable def paint_for_similar_statues (n : Nat) (h₁ h₂ : ℝ) (p₁ : ℝ) : ℝ :=
  let ratio := (h₂ / h₁) ^ 2
  n * (ratio * p₁)

theorem paint_needed_for_720_statues :
  paint_for_similar_statues 720 12 2 1 = 20 :=
by
  sorry

end paint_needed_for_720_statues_l278_278634


namespace cube_volume_surface_area_x_l278_278921

theorem cube_volume_surface_area_x (x s : ℝ) (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_x_l278_278921


namespace number_of_bricks_is_1800_l278_278129

-- Define the conditions
def rate_first_bricklayer (x : ℕ) : ℕ := x / 8
def rate_second_bricklayer (x : ℕ) : ℕ := x / 12
def combined_reduced_rate (x : ℕ) : ℕ := (rate_first_bricklayer x + rate_second_bricklayer x - 15)

-- Prove that the number of bricks in the wall is 1800
theorem number_of_bricks_is_1800 :
  ∃ x : ℕ, 5 * combined_reduced_rate x = x ∧ x = 1800 :=
by
  use 1800
  sorry

end number_of_bricks_is_1800_l278_278129


namespace original_number_is_106_25_l278_278771

theorem original_number_is_106_25 (x : ℝ) (h : (x + 0.375 * x) - (x - 0.425 * x) = 85) : x = 106.25 := by
  sorry

end original_number_is_106_25_l278_278771


namespace circle_symmetric_about_line_l278_278998

theorem circle_symmetric_about_line :
  ∃ b : ℝ, (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 4 = 0 → y = 2*x + b) → b = 4 :=
by
  sorry

end circle_symmetric_about_line_l278_278998


namespace derivative_of_function_y_l278_278980

noncomputable def function_y (x : ℝ) : ℝ := (x^2) / (x + 3)

theorem derivative_of_function_y (x : ℝ) :
  deriv function_y x = (x^2 + 6 * x) / ((x + 3)^2) :=
by 
  -- sorry since the proof is not required
  sorry

end derivative_of_function_y_l278_278980


namespace infinite_series_sum_eq_l278_278458

theorem infinite_series_sum_eq : 
  (∑' n : ℕ, if n = 0 then 0 else ((1 : ℝ) / (n * (n + 3)))) = (11 / 18 : ℝ) :=
sorry

end infinite_series_sum_eq_l278_278458


namespace rectangle_ratio_l278_278329

theorem rectangle_ratio (s y x : ℝ)
  (h1 : 4 * y * x + s * s = 9 * s * s)
  (h2 : s + y + y = 3 * s)
  (h3 : y = s)
  (h4 : x + s = 3 * s) : 
  (x / y = 2) :=
sorry

end rectangle_ratio_l278_278329


namespace find_b_value_l278_278567

theorem find_b_value (x : ℝ) (h_neg : x < 0) (h_eq : 1 / (x + 1 / (x + 2)) = 2) : 
  x + 7 / 2 = 2 :=
sorry

end find_b_value_l278_278567


namespace possible_values_of_expression_l278_278598

open Matrix

-- Define the matrix
def myMatrix (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^2, b, c], ![b^2, c, a], ![c^2, a, b]]

-- Define the polynomial
def p (a b c x : ℝ) : ℝ := x^3 - (a^2 + b^2 + c^2) * x^2 + (a^4 + b^4 + c^4) * x - a * b * c

-- Conditions
variables a b c : ℝ
#check Real

noncomputable def det_myMatrix : ℝ :=
  det (myMatrix a b c)

theorem possible_values_of_expression (h1 : det_myMatrix = 0) :
  (∃ (v : ℝ), v ∈ {-1, (3 / 2)} ∧
  v = (a^2 / (b^2 + c) + b^2 / (a^2 + c) + c^2 / (a^2 + b^2))) :=
begin
  sorry
end

end possible_values_of_expression_l278_278598


namespace problem1_problem2_l278_278627

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := { x | a - b < x ∧ x < a + b }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- First problem: prove the range of a
theorem problem1 (a : ℝ) (h : A a 1 ⊆ B) : a ≤ -2 ∨ a ≥ 6 := by
  sorry

-- Second problem: prove the range of b
theorem problem2 (b : ℝ) (h : A 1 b ∩ B = ∅) : b ≤ 2 := by
  sorry

end problem1_problem2_l278_278627


namespace bacteria_initial_count_l278_278246

noncomputable def initial_bacteria (b_final : ℕ) (q : ℕ) : ℕ :=
  b_final / 4^q

theorem bacteria_initial_count : initial_bacteria 262144 4 = 1024 := by
  sorry

end bacteria_initial_count_l278_278246


namespace smallest_possible_value_l278_278110

theorem smallest_possible_value (n : ℕ) (h1 : ∀ m, (Nat.lcm 60 m / Nat.gcd 60 m = 24) → m = n) (h2 : ∀ m, (m % 5 = 0) → m = n) : n = 160 :=
sorry

end smallest_possible_value_l278_278110


namespace raft_sticks_total_l278_278397

theorem raft_sticks_total : 
  let S := 45 
  let G := (3/5 * 45 : ℝ)
  let M := 45 + G + 15
  let D := 2 * M - 7
  S + G + M + D = 326 := 
by
  sorry

end raft_sticks_total_l278_278397


namespace largest_natural_number_not_sum_of_two_composites_l278_278811

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278811


namespace total_students_in_school_l278_278118

theorem total_students_in_school (s : ℕ) (below_8 above_8 : ℕ) (students_8 : ℕ)
  (h1 : below_8 = 20 * s / 100) 
  (h2 : above_8 = 2 * students_8 / 3) 
  (h3 : students_8 = 48) 
  (h4 : s = students_8 + above_8 + below_8) : 
  s = 100 := 
by 
  sorry 

end total_students_in_school_l278_278118


namespace exponent_multiplication_l278_278353

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end exponent_multiplication_l278_278353


namespace bill_steps_l278_278192

theorem bill_steps (step_length : ℝ) (total_distance : ℝ) (n_steps : ℕ) 
  (h_step_length : step_length = 1 / 2) 
  (h_total_distance : total_distance = 12) 
  (h_n_steps : n_steps = total_distance / step_length) : 
  n_steps = 24 :=
by sorry

end bill_steps_l278_278192


namespace largest_non_summable_composite_l278_278806

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278806


namespace eval_poly_at_2_l278_278623

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem eval_poly_at_2 :
  f 2 = 123 :=
by
  sorry

end eval_poly_at_2_l278_278623


namespace remainder_130_div_k_l278_278983

theorem remainder_130_div_k (k : ℕ) (h_positive : k > 0)
  (h_remainder : 84 % (k*k) = 20) : 
  130 % k = 2 := 
by sorry

end remainder_130_div_k_l278_278983


namespace total_amount_is_24_l278_278585

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l278_278585


namespace number_of_real_roots_l278_278253

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 6 * x ^ 2 + 9 * x - 10

theorem number_of_real_roots : ∃! x : ℝ, f x = 0 :=
sorry

end number_of_real_roots_l278_278253


namespace sum_min_values_eq_zero_l278_278957

-- Definitions of the polynomials
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c*x + d

-- Main theorem statement
theorem sum_min_values_eq_zero (b d : ℝ) :
  let a := -16
  let c := -8
  (-64 + b = 0) ∧ (-16 + d = 0) → (-64 + b + (-16 + d) = 0) :=
by
  intros
  rw [add_assoc]
  sorry

end sum_min_values_eq_zero_l278_278957


namespace find_m_l278_278015

theorem find_m (m : ℝ) (a b : ℝ × ℝ) (k : ℝ) (ha : a = (1, 1)) (hb : b = (m, 2)) 
  (h_parallel : 2 • a + b = k • a) : m = 2 :=
sorry

end find_m_l278_278015


namespace number_of_divisibles_by_eight_in_range_l278_278859

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278859


namespace minimum_value_expr_l278_278477

theorem minimum_value_expr (a : ℝ) (h₀ : 0 < a) (h₁ : a < 2) : 
  ∃ (m : ℝ), m = (4 / a + 1 / (2 - a)) ∧ m = 9 / 2 :=
by
  sorry

end minimum_value_expr_l278_278477


namespace largest_stamps_per_page_l278_278374

-- Definitions of the conditions
def stamps_book1 : ℕ := 1260
def stamps_book2 : ℕ := 1470

-- Statement to be proven: The largest number of stamps per page (gcd of 1260 and 1470)
theorem largest_stamps_per_page : Nat.gcd stamps_book1 stamps_book2 = 210 :=
by
  sorry

end largest_stamps_per_page_l278_278374


namespace money_left_after_purchase_l278_278968

noncomputable def initial_money : ℝ := 200
noncomputable def candy_bars : ℝ := 25
noncomputable def bags_of_chips : ℝ := 10
noncomputable def soft_drinks : ℝ := 15

noncomputable def cost_per_candy_bar : ℝ := 3
noncomputable def cost_per_bag_of_chips : ℝ := 2.5
noncomputable def cost_per_soft_drink : ℝ := 1.75

noncomputable def discount_candy_bars : ℝ := 0.10
noncomputable def discount_bags_of_chips : ℝ := 0.05
noncomputable def sales_tax : ℝ := 0.06

theorem money_left_after_purchase : initial_money - 
  ( ((candy_bars * cost_per_candy_bar * (1 - discount_candy_bars)) + 
    (bags_of_chips * cost_per_bag_of_chips * (1 - discount_bags_of_chips)) + 
    (soft_drinks * cost_per_soft_drink)) * 
    (1 + sales_tax)) = 75.45 := by
  sorry

end money_left_after_purchase_l278_278968


namespace remainder_777_777_mod_13_l278_278919

theorem remainder_777_777_mod_13 : (777^777) % 13 = 1 := by
  sorry

end remainder_777_777_mod_13_l278_278919


namespace maximize_profit_l278_278885

/-- 
The total number of rooms in the hotel 
-/
def totalRooms := 80

/-- 
The initial rent when the hotel is fully booked 
-/
def initialRent := 160

/-- 
The loss in guests for each increase in rent by 20 yuan 
-/
def guestLossPerIncrease := 3

/-- 
The increase in rent 
-/
def increasePer20Yuan := 20

/-- 
The daily service and maintenance cost per occupied room
-/
def costPerOccupiedRoom := 40

/-- 
Maximize profit given the conditions
-/
theorem maximize_profit : 
  ∃ x : ℕ, x = 360 ∧ 
            ∀ y : ℕ,
              (initialRent - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (x - initialRent) / increasePer20Yuan)
              ≥ (y - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (y - initialRent) / increasePer20Yuan) := 
sorry

end maximize_profit_l278_278885


namespace cassandra_overall_score_l278_278595

theorem cassandra_overall_score 
  (score1_percent : ℤ) (score1_total : ℕ)
  (score2_percent : ℤ) (score2_total : ℕ)
  (score3_percent : ℤ) (score3_total : ℕ) :
  score1_percent = 60 → score1_total = 15 →
  score2_percent = 75 → score2_total = 20 →
  score3_percent = 85 → score3_total = 25 →
  let correct1 := (score1_percent * score1_total) / 100
  let correct2 := (score2_percent * score2_total) / 100
  let correct3 := (score3_percent * score3_total) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := score1_total + score2_total + score3_total
  75 = (100 * total_correct) / total_problems := by
  intros h1 h2 h3 h4 h5 h6
  let correct1 := (60 * 15) / 100
  let correct2 := (75 * 20) / 100
  let correct3 := (85 * 25) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := 15 + 20 + 25
  suffices 75 = (100 * total_correct) / total_problems by sorry
  sorry

end cassandra_overall_score_l278_278595


namespace Quentin_chickens_l278_278523

variable (C S Q : ℕ)

theorem Quentin_chickens (h1 : C = 37)
    (h2 : S = 3 * C - 4)
    (h3 : Q + S + C = 383) :
    (Q = 2 * S + 32) :=
by
  sorry

end Quentin_chickens_l278_278523


namespace total_amount_is_24_l278_278583

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l278_278583


namespace sum_first_100_sum_51_to_100_l278_278959

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem sum_first_100 : sum_natural_numbers 100 = 5050 :=
  sorry

theorem sum_51_to_100 : sum_natural_numbers 100 - sum_natural_numbers 50 = 3775 :=
  sorry

end sum_first_100_sum_51_to_100_l278_278959


namespace largest_natural_number_not_sum_of_two_composites_l278_278813

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278813


namespace surface_area_with_holes_l278_278448

-- Define the cube and holes properties
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def number_faces_cube : ℕ := 6

-- Define areas
def area_face_cube := edge_length_cube ^ 2
def area_face_hole := side_length_hole ^ 2
def original_surface_area := number_faces_cube * area_face_cube
def total_hole_area := number_faces_cube * area_face_hole
def new_exposed_area := number_faces_cube * 4 * area_face_hole

-- Calculate the total surface area including holes
def total_surface_area := original_surface_area - total_hole_area + new_exposed_area

-- Lean statement for the proof
theorem surface_area_with_holes :
  total_surface_area = 168 := by
  sorry

end surface_area_with_holes_l278_278448


namespace largest_non_representable_as_sum_of_composites_l278_278798

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278798


namespace saber_toothed_frog_tails_l278_278540

def tails_saber_toothed_frog (n k : ℕ) (x : ℕ) : Prop :=
  5 * n + 4 * k = 100 ∧ n + x * k = 64

theorem saber_toothed_frog_tails : ∃ x, ∃ n k : ℕ, tails_saber_toothed_frog n k x ∧ x = 3 := 
by
  sorry

end saber_toothed_frog_tails_l278_278540


namespace andrey_valentin_distance_l278_278148

/-- Prove the distance between Andrey and Valentin is 107 meters,
    given that Andrey finishes 60 meters ahead of Boris and Boris finishes 50 meters ahead of Valentin.
    Assumptions:
    - Andrey, Boris, and Valentin participated in a 1 km race.
    - All participants ran at a constant speed. -/
theorem andrey_valentin_distance :
  ∀ (a b c : ℝ), -- Speeds of Andrey, Boris, and Valentin respectively.
  (a ≠ 0) →     -- Non-zero speed
  (b = 0.94 * a) → 
  (c = 0.95 * b) →
  (distance_a := 1000 : ℝ) →
  (distance_valentin : ℝ := c * (distance_a / a)) →
  (distance_andrey_valentin : ℝ := distance_a - distance_valentin) →
  distance_andrey_valentin = 107 :=
by {
  intros a b c ha hb hc distance_a distance_valentin distance_andrey_valentin,
  subst_vars,
  sorry
}

end andrey_valentin_distance_l278_278148


namespace largest_perimeter_l278_278766

noncomputable def triangle_largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : ℕ :=
7 + 8 + y

theorem largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : triangle_largest_perimeter y h1 h2 = 29 :=
sorry

end largest_perimeter_l278_278766


namespace rent_increase_percentage_l278_278245

theorem rent_increase_percentage :
  ∀ (initial_avg new_avg rent : ℝ) (num_friends : ℝ),
    num_friends = 4 →
    initial_avg = 800 →
    new_avg = 850 →
    rent = 800 →
    ((num_friends * new_avg) - (num_friends * initial_avg)) / rent * 100 = 25 :=
by
  intros initial_avg new_avg rent num_friends h_num h_initial h_new h_rent
  sorry

end rent_increase_percentage_l278_278245


namespace num_integer_solutions_of_equation_l278_278030

theorem num_integer_solutions_of_equation : 
  (∃ (x y : ℤ), (x^2 + y^2 = 6*x + 2*y + 15)) = 12 := 
sorry

end num_integer_solutions_of_equation_l278_278030


namespace integer_roots_of_polynomial_l278_278472

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 4 * x^2 - 7 * x + 10 = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_of_polynomial_l278_278472


namespace cost_price_of_computer_table_l278_278696

theorem cost_price_of_computer_table (S : ℝ) (C : ℝ) (h1 : 1.80 * C = S) (h2 : S = 3500) : C = 1944.44 :=
by
  sorry

end cost_price_of_computer_table_l278_278696


namespace Jovana_final_addition_l278_278511

theorem Jovana_final_addition 
  (initial_amount added_initial removed final_amount x : ℕ)
  (h1 : initial_amount = 5)
  (h2 : added_initial = 9)
  (h3 : removed = 2)
  (h4 : final_amount = 28) :
  final_amount = initial_amount + added_initial - removed + x → x = 16 :=
by
  intros h
  sorry

end Jovana_final_addition_l278_278511


namespace inequality_proof_l278_278333

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b :=
sorry

end inequality_proof_l278_278333


namespace range_of_k_for_real_roots_l278_278040

theorem range_of_k_for_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k <= 1 / 12 ∧ k ≠ 0) :=
sorry

end range_of_k_for_real_roots_l278_278040


namespace geometric_series_squares_sum_l278_278249

theorem geometric_series_squares_sum (a : ℝ) (r : ℝ) (h : -1 < r ∧ r < 1) :
  (∑' n : ℕ, (a * r^n)^2) = a^2 / (1 - r^2) :=
by sorry

end geometric_series_squares_sum_l278_278249


namespace machine_x_produces_40_percent_l278_278295

theorem machine_x_produces_40_percent (T X Y : ℝ) 
  (h1 : X + Y = T)
  (h2 : 0.009 * X + 0.004 * Y = 0.006 * T) :
  X = 0.4 * T :=
by
  sorry

end machine_x_produces_40_percent_l278_278295


namespace find_prices_l278_278439

variables (C S : ℕ) -- Using natural numbers to represent rubles

theorem find_prices (h1 : C + S = 2500) (h2 : 4 * C + 3 * S = 8870) :
  C = 1370 ∧ S = 1130 :=
by
  sorry

end find_prices_l278_278439


namespace movie_screening_guests_l278_278549

theorem movie_screening_guests
  (total_guests : ℕ)
  (women_percentage : ℝ)
  (men_count : ℕ)
  (men_left_fraction : ℝ)
  (children_left_percentage : ℝ)
  (children_count : ℕ)
  (people_left : ℕ) :
  total_guests = 75 →
  women_percentage = 0.40 →
  men_count = 25 →
  men_left_fraction = 1/3 →
  children_left_percentage = 0.20 →
  children_count = total_guests - (round (women_percentage * total_guests) + men_count) →
  people_left = (round (men_left_fraction * men_count)) + (round (children_left_percentage * children_count)) →
  (total_guests - people_left) = 63 :=
by
  intros ht hw hm hf hc hc_count hl
  sorry

end movie_screening_guests_l278_278549


namespace largest_number_not_sum_of_two_composites_l278_278824

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278824


namespace algebra_expression_value_l278_278005

theorem algebra_expression_value (a b : ℝ) (h : a - 2 * b = -1) : 1 - 2 * a + 4 * b = 3 :=
by
  sorry

end algebra_expression_value_l278_278005


namespace dig_days_l278_278203

theorem dig_days (m1 m2 : ℕ) (d1 d2 : ℚ) (k : ℚ) 
  (h1 : m1 * d1 = k) (h2 : m2 * d2 = k) : 
  m1 = 30 ∧ d1 = 6 ∧ m2 = 40 → d2 = 4.5 := 
by sorry

end dig_days_l278_278203


namespace distinct_solutions_abs_eq_l278_278025

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 5| = |x + 3|) → ∃! (x : ℝ), x = 1 :=
begin
  sorry
end

end distinct_solutions_abs_eq_l278_278025


namespace limit_sequence_l278_278776

def sequence (n : ℕ) : ℝ :=
  (sqrt (3 * n - 1) - (125 * n ^ 3 + n) ^ (1 / 3)) /
  (n ^ (1 / 3) - n)

theorem limit_sequence : Filter.Tendsto sequence Filter.atTop (nhds 5) :=
by
  sorry

end limit_sequence_l278_278776


namespace perpendicular_vectors_l278_278345

theorem perpendicular_vectors (x : ℝ) : (2 * x + 3 = 0) → (x = -3 / 2) :=
by
  intro h
  sorry

end perpendicular_vectors_l278_278345


namespace number_of_tests_in_series_l278_278761

theorem number_of_tests_in_series (S : ℝ) (n : ℝ) :
  (S + 97) / n = 90 →
  (S + 73) / n = 87 →
  n = 8 :=
by 
  sorry

end number_of_tests_in_series_l278_278761


namespace find_a6_plus_a7_plus_a8_l278_278215

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l278_278215


namespace largest_cannot_be_sum_of_two_composites_l278_278797

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278797


namespace sum_is_seventeen_l278_278667

variable (x y : ℕ)

def conditions (x y : ℕ) : Prop :=
  x > y ∧ x - y = 3 ∧ x * y = 56

theorem sum_is_seventeen (x y : ℕ) (h: conditions x y) : x + y = 17 :=
by
  sorry

end sum_is_seventeen_l278_278667


namespace value_of_x_l278_278863

theorem value_of_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y + 2))  : 
  x = y^2 + 2 * y + 3 := 
by 
  sorry

end value_of_x_l278_278863


namespace largest_non_summable_composite_l278_278805

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278805


namespace line_parameterization_l278_278091

theorem line_parameterization (s m : ℝ) :
  (∃ t : ℝ, ∀ x y : ℝ, (x = s + 2 * t ∧ y = 3 + m * t) ↔ y = 5 * x - 7) →
  s = 2 ∧ m = 10 :=
by
  intro h_conditions
  sorry

end line_parameterization_l278_278091


namespace probability_black_balls_l278_278311

variable {m1 m2 k1 k2 : ℕ}

/-- Given conditions:
  1. The total number of balls in both urns is 25.
  2. The probability of drawing one white ball from each urn is 0.54.
To prove: The probability of both drawn balls being black is 0.04.
-/
theorem probability_black_balls : 
  m1 + m2 = 25 → 
  (k1 * k2) * 50 = 27 * m1 * m2 → 
  ((m1 - k1) * (m2 - k2) : ℚ) / (m1 * m2) = 0.04 :=
by
  intros h1 h2
  sorry

end probability_black_balls_l278_278311


namespace rectangle_area_l278_278090

theorem rectangle_area :
  ∃ (l w : ℝ), l = 4 * w ∧ 2 * l + 2 * w = 200 ∧ l * w = 1600 :=
by
  use [80, 20]
  split; norm_num
  split; norm_num
  sorry

end rectangle_area_l278_278090


namespace find_equation_of_BC_l278_278624

theorem find_equation_of_BC :
  ∃ (BC : ℝ → ℝ → Prop), 
  (∀ x y, (BC x y ↔ 2 * x - y + 5 = 0)) :=
sorry

end find_equation_of_BC_l278_278624


namespace sum_of_coefficients_eq_l278_278361

theorem sum_of_coefficients_eq :
  ∃ n : ℕ, (∀ a b : ℕ, (3 * a + 5 * b)^n = 2^15) → n = 5 :=
by
  sorry

end sum_of_coefficients_eq_l278_278361


namespace path_count_A_to_D_via_B_and_C_l278_278363

def count_paths (start finish : ℕ × ℕ) (h_steps v_steps : ℕ) : ℕ :=
  Nat.choose (h_steps + v_steps) v_steps

theorem path_count_A_to_D_via_B_and_C :
  let A := (0, 5)
  let B := (3, 3)
  let C := (4, 1)
  let D := (6, 0)
  count_paths A B 3 2 * count_paths B C 1 2 * count_paths C D 2 1 = 90 :=
by {
  let A := (0, 5),
  let B := (3, 3),
  let C := (4, 1),
  let D := (6, 0),
  suffices h1 : count_paths A B 3 2 = 10, by {
    suffices h2 : count_paths B C 1 2 = 3, by {
      suffices h3 : count_paths C D 2 1 = 3, by {
        calc
          count_paths A B 3 2 * count_paths B C 1 2 * count_paths C D 2 1
            = 10 * 3 * 3 : by rw [h1, h2, h3]
        ... = 90 : by norm_num
      }
      show count_paths C D 2 1 = 3, sorry
    }
    show count_paths B C 1 2 = 3, sorry
  }
  show count_paths A B 3 2 = 10, sorry
}

end path_count_A_to_D_via_B_and_C_l278_278363


namespace xiaobo_probability_not_home_l278_278115

theorem xiaobo_probability_not_home :
  let r1 := 1 / 2
  let r2 := 1 / 4
  let area_circle := Real.pi
  let area_greater_r1 := area_circle * (1 - r1^2)
  let area_less_r2 := area_circle * r2^2
  let area_favorable := area_greater_r1 + area_less_r2
  let probability_not_home := area_favorable / area_circle
  probability_not_home = 13 / 16 := by
  sorry

end xiaobo_probability_not_home_l278_278115


namespace customers_in_each_car_l278_278541

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end customers_in_each_car_l278_278541


namespace leah_coins_value_l278_278058

theorem leah_coins_value :
  ∃ (p n d : ℕ), 
    p + n + d = 20 ∧
    p = n ∧
    p = d + 4 ∧
    1 * p + 5 * n + 10 * d = 88 :=
by
  sorry

end leah_coins_value_l278_278058


namespace number_of_zeros_of_quadratic_function_l278_278095

-- Given the quadratic function y = x^2 + x - 1
def quadratic_function (x : ℝ) : ℝ := x^2 + x - 1

-- Prove that the number of zeros of the quadratic function y = x^2 + x - 1 is 2
theorem number_of_zeros_of_quadratic_function : 
  ∃ x1 x2 : ℝ, quadratic_function x1 = 0 ∧ quadratic_function x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

end number_of_zeros_of_quadratic_function_l278_278095


namespace degree_measure_cherry_pie_l278_278872

theorem degree_measure_cherry_pie 
  (total_students : ℕ) 
  (chocolate_pie : ℕ) 
  (apple_pie : ℕ) 
  (blueberry_pie : ℕ) 
  (remaining_students : ℕ)
  (remaining_students_eq_div : remaining_students = (total_students - (chocolate_pie + apple_pie + blueberry_pie))) 
  (equal_division : remaining_students / 2 = 5) 
  : (remaining_students / 2 * 360 / total_students = 45) := 
by 
  sorry

end degree_measure_cherry_pie_l278_278872


namespace jerry_charge_per_hour_l278_278653

-- Define the conditions from the problem
def time_painting : ℝ := 8
def time_fixing_counter : ℝ := 3 * time_painting
def time_mowing_lawn : ℝ := 6
def total_time_worked : ℝ := time_painting + time_fixing_counter + time_mowing_lawn
def total_payment : ℝ := 570

-- The proof statement
theorem jerry_charge_per_hour : 
  total_payment / total_time_worked = 15 :=
by
  sorry

end jerry_charge_per_hour_l278_278653


namespace largest_not_sum_of_two_composites_l278_278819

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278819


namespace parabola_vertex_example_l278_278603

noncomputable def parabola_vertex (a b c : ℝ) := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem parabola_vertex_example : parabola_vertex (-4) (-16) (-20) = (-2, -4) :=
by
  sorry

end parabola_vertex_example_l278_278603


namespace quadratic_roots_identity_l278_278929

noncomputable def sum_of_roots (a b : ℝ) : Prop := a + b = -10
noncomputable def product_of_roots (a b : ℝ) : Prop := a * b = 5

theorem quadratic_roots_identity (a b : ℝ)
  (h₁ : sum_of_roots a b)
  (h₂ : product_of_roots a b) :
  (a / b + b / a) = 18 :=
by sorry

end quadratic_roots_identity_l278_278929


namespace sum_a6_a7_a8_is_32_l278_278218

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l278_278218


namespace necessary_but_not_sufficient_l278_278283

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l278_278283


namespace unique_zero_a_neg_l278_278749

noncomputable def f (a x : ℝ) : ℝ := 3 * Real.exp (abs (x - 1)) - a * (2^(x - 1) + 2^(1 - x)) - a^2

theorem unique_zero_a_neg (a : ℝ) (h_unique : ∃! x : ℝ, f a x = 0) (h_neg : a < 0) : a = -3 := 
sorry

end unique_zero_a_neg_l278_278749


namespace total_distance_traveled_l278_278763

/--
A spider is on the edge of a ceiling of a circular room with a radius of 65 feet. 
The spider walks straight across the ceiling to the opposite edge, passing through 
the center. It then walks straight to another point on the edge of the circle but 
not back through the center. The third part of the journey is straight back to the 
original starting point. If the third part of the journey was 90 feet long, then 
the total distance traveled by the spider is 313.81 feet.
-/
theorem total_distance_traveled (r : ℝ) (d1 d2 d3 : ℝ) (h1 : r = 65) (h2 : d1 = 2 * r) (h3 : d3 = 90) :
  d1 + d2 + d3 = 313.81 :=
by
  sorry

end total_distance_traveled_l278_278763


namespace age_difference_l278_278925

theorem age_difference (A B C : ℕ) (h1 : B = 20) (h2 : C = B / 2) (h3 : A + B + C = 52) : A - B = 2 := by
  sorry

end age_difference_l278_278925


namespace initial_owls_l278_278079

theorem initial_owls (n_0 : ℕ) (h : n_0 + 2 = 5) : n_0 = 3 :=
by 
  sorry

end initial_owls_l278_278079


namespace swimming_speed_in_still_water_l278_278760

theorem swimming_speed_in_still_water (v : ℝ) (current_speed : ℝ) (time : ℝ) (distance : ℝ) (effective_speed : current_speed = 10) (time_to_return : time = 6) (distance_to_return : distance = 12) (speed_eq : v - current_speed = distance / time) : v = 12 :=
by
  sorry

end swimming_speed_in_still_water_l278_278760


namespace transformed_quadratic_roots_l278_278362

-- Definitions of the conditions
def quadratic_roots (a b : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + 3 = 0 → (x = -2) ∨ (x = 3)

-- Statement of the theorem
theorem transformed_quadratic_roots (a b : ℝ) :
  quadratic_roots a b →
  ∀ x : ℝ, a * (x + 2)^2 + b * (x + 2) + 3 = 0 → (x = -4) ∨ (x = 1) :=
sorry

end transformed_quadratic_roots_l278_278362


namespace correct_multiplication_l278_278289

theorem correct_multiplication (n : ℕ) (wrong_answer correct_answer : ℕ) 
    (h1 : wrong_answer = 559981)
    (h2 : correct_answer = 987 * n)
    (h3 : ∃ (x y : ℕ), correct_answer = 500000 + x + 901 + y ∧ x ≠ 98 ∧ y ≠ 98 ∧ (wrong_answer - correct_answer) % 10 = 0) :
    correct_answer = 559989 :=
by
  sorry

end correct_multiplication_l278_278289


namespace find_a6_plus_a7_plus_a8_l278_278214

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l278_278214


namespace smallest_rectangles_needed_l278_278557

theorem smallest_rectangles_needed {a b : ℕ} (h1 : a = 3) (h2 : b = 4) :
  ∃ (n : ℕ), ∀ (s : ℕ), square_side s ∧ (∃ m, m * (a * b) = s * s) → n = 12 :=
begin
  sorry

end smallest_rectangles_needed_l278_278557


namespace Mona_grouped_with_one_player_before_in_second_group_l278_278887

/-- Mona plays in groups with four other players, joined 9 groups, and grouped with 33 unique players. 
    One of the groups included 2 players she had grouped with before. 
    Prove that the number of players she had grouped with before in the second group is 1. -/
theorem Mona_grouped_with_one_player_before_in_second_group 
    (total_groups : ℕ) (group_size : ℕ) (unique_players : ℕ) 
    (repeat_players_in_group1 : ℕ) : 
    (total_groups = 9) → (group_size = 5) → (unique_players = 33) → (repeat_players_in_group1 = 2) 
        → ∃ repeat_players_in_group2 : ℕ, repeat_players_in_group2 = 1 :=
by
    sorry

end Mona_grouped_with_one_player_before_in_second_group_l278_278887


namespace simplify_expression_l278_278679

theorem simplify_expression : (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by
  sorry

end simplify_expression_l278_278679


namespace expand_product_eq_l278_278606

theorem expand_product_eq :
  (∀ (x : ℤ), (x^3 - 3 * x^2 + 3 * x - 1) * (x^2 + 3 * x + 3) = x^5 - 3 * x^3 - x^2 + 3 * x) :=
by
  intro x
  sorry

end expand_product_eq_l278_278606


namespace find_root_D_l278_278483

/-- Given C and D are roots of the polynomial k x^2 + 2 x + 5 = 0, 
    and k = -1/4 and C = 10, then D must be -2. -/
theorem find_root_D 
  (k : ℚ) (C D : ℚ)
  (h1 : k = -1/4)
  (h2 : C = 10)
  (h3 : C^2 * k + 2 * C + 5 = 0)
  (h4 : D^2 * k + 2 * D + 5 = 0) : 
  D = -2 :=
by
  sorry

end find_root_D_l278_278483


namespace exists_multiple_l278_278616

theorem exists_multiple (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, a i > 0) 
  (h2 : ∀ i, a i ≤ 2 * n) : 
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
sorry

end exists_multiple_l278_278616


namespace solve_for_a_l278_278629

theorem solve_for_a (a y x : ℝ)
  (h1 : y = 5 * a)
  (h2 : x = 2 * a - 2)
  (h3 : y + 3 = x) :
  a = -5 / 3 :=
by
  sorry

end solve_for_a_l278_278629


namespace no_valid_n_l278_278476

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n.minFac

theorem no_valid_n (n : ℕ) (h1 : n > 1)
  (h2 : is_prime (greatest_prime_factor n))
  (h3 : greatest_prime_factor n = Nat.sqrt n)
  (h4 : is_prime (greatest_prime_factor (n + 36)))
  (h5 : greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) :
  false :=
sorry

end no_valid_n_l278_278476


namespace largest_cannot_be_sum_of_two_composites_l278_278795

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278795


namespace expression_is_nonnegative_l278_278224

noncomputable def expression_nonnegative (a b c d e : ℝ) : Prop :=
  (a - b) * (a - c) * (a - d) * (a - e) +
  (b - a) * (b - c) * (b - d) * (b - e) +
  (c - a) * (c - b) * (c - d) * (c - e) +
  (d - a) * (d - b) * (d - c) * (d - e) +
  (e - a) * (e - b) * (e - c) * (e - d) ≥ 0

theorem expression_is_nonnegative (a b c d e : ℝ) : expression_nonnegative a b c d e := 
by 
  sorry

end expression_is_nonnegative_l278_278224


namespace min_value_expr_l278_278836

theorem min_value_expr (a : ℝ) (h1 : 1 < a) (h2 : a < 4) : (∀ b, (1 < b ∧ b < 4) → (b / (4 - b) + 1 / (b - 1)) ≥ 2) :=
by
  intro b hb1 hb2
  sorry

end min_value_expr_l278_278836


namespace least_cost_planting_l278_278394

theorem least_cost_planting :
  let region1_area := 3 * 1
  let region2_area := 4 * 4
  let region3_area := 7 * 2
  let region4_area := 5 * 4
  let region5_area := 5 * 6
  let easter_lilies_cost_per_sqft := 3.25
  let dahlias_cost_per_sqft := 2.75
  let cannas_cost_per_sqft := 2.25
  let begonias_cost_per_sqft := 1.75
  let asters_cost_per_sqft := 1.25
  region1_area * easter_lilies_cost_per_sqft +
  region2_area * dahlias_cost_per_sqft +
  region3_area * cannas_cost_per_sqft +
  region4_area * begonias_cost_per_sqft +
  region5_area * asters_cost_per_sqft =
  156.75 := 
sorry

end least_cost_planting_l278_278394


namespace sum_first_2009_terms_arith_seq_l278_278876

variable {a : ℕ → ℝ}

-- Given condition a_1004 + a_1005 + a_1006 = 3
axiom H : a 1004 + a 1005 + a 1006 = 3

-- Arithmetic sequence definition
def is_arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem sum_first_2009_terms_arith_seq
  (d : ℝ) (h_arith_seq : is_arith_seq a d)
  : sum_arith_seq a 2009 = 2009 := 
by
  sorry

end sum_first_2009_terms_arith_seq_l278_278876


namespace children_difference_l278_278412

-- Axiom definitions based on conditions
def initial_children : ℕ := 36
def first_stop_got_off : ℕ := 45
def first_stop_got_on : ℕ := 25
def second_stop_got_off : ℕ := 68
def final_children : ℕ := 12

-- Mathematical formulation of the problem and its proof statement
theorem children_difference :
  ∃ (x : ℕ), 
    initial_children - first_stop_got_off + first_stop_got_on - second_stop_got_off + x = final_children ∧ 
    (first_stop_got_off + second_stop_got_off) - (first_stop_got_on + x) = 24 :=
by 
  sorry

end children_difference_l278_278412


namespace ratio_of_Frederick_to_Tyson_l278_278656

-- Definitions of the ages based on given conditions
def Kyle : Nat := 25
def Tyson : Nat := 20
def Julian : Nat := Kyle - 5
def Frederick : Nat := Julian + 20

-- The ratio of Frederick's age to Tyson's age
def ratio : Nat × Nat := (Frederick / Nat.gcd Frederick Tyson, Tyson / Nat.gcd Frederick Tyson)

-- Proving the ratio is 2:1
theorem ratio_of_Frederick_to_Tyson : ratio = (2, 1) := by
  sorry

end ratio_of_Frederick_to_Tyson_l278_278656


namespace find_polynomial_P_l278_278513

noncomputable def P (x : ℝ) : ℝ :=
  - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1

theorem find_polynomial_P 
  (α β γ : ℝ)
  (h_roots : ∀ {x: ℝ}, x^3 - 4 * x^2 + 6 * x + 8 = 0 → x = α ∨ x = β ∨ x = γ)
  (h1 : P α = β + γ)
  (h2 : P β = α + γ)
  (h3 : P γ = α + β)
  (h4 : P (α + β + γ) = -20) :
  P x = - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1 :=
by sorry

end find_polynomial_P_l278_278513


namespace sum_even_minus_sum_odd_l278_278194

theorem sum_even_minus_sum_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 := by
sorry

end sum_even_minus_sum_odd_l278_278194


namespace find_n_for_divisibility_by_33_l278_278613

theorem find_n_for_divisibility_by_33 (n : ℕ) (hn_range : n < 10) (div11 : (12 - n) % 11 = 0) (div3 : (20 + n) % 3 = 0) : n = 1 :=
by {
  -- Proof steps go here
  sorry
}

end find_n_for_divisibility_by_33_l278_278613


namespace largest_non_sum_of_composites_l278_278788

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278788


namespace mean_equals_d_l278_278274

noncomputable def sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

theorem mean_equals_d
  (a b c d e : ℝ)
  (h_a : a = sqrt 2)
  (h_b : b = sqrt 18)
  (h_c : c = sqrt 200)
  (h_d : d = sqrt 32)
  (h_e : e = sqrt 8) :
  d = (a + b + c + e) / 4 := by
  -- We insert proof steps here normally
  sorry

end mean_equals_d_l278_278274


namespace license_plate_probability_l278_278886

theorem license_plate_probability :
  let m := 5
  let n := 104
  Nat.gcd m n = 1 ∧ m + n = 109 := by
  have h : Nat.gcd 5 104 = 1 := by norm_num
  exact ⟨h, by norm_num⟩

end license_plate_probability_l278_278886


namespace find_ccb_l278_278064

theorem find_ccb (a b c : ℕ) 
  (h1: a ≠ b) 
  (h2: a ≠ c) 
  (h3: b ≠ c) 
  (h4: b = 1) 
  (h5: (10 * a + b) ^ 2 = 100 * c + 10 * c + b) 
  (h6: 100 * c + 10 * c + b > 300) : 
  100 * c + 10 * c + b = 441 :=
sorry

end find_ccb_l278_278064


namespace prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l278_278069

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l278_278069


namespace company_employees_after_reduction_l278_278446

theorem company_employees_after_reduction :
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  current_number = 195 :=
by
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  sorry

end company_employees_after_reduction_l278_278446


namespace distinct_solution_count_number_of_solutions_l278_278023

theorem distinct_solution_count (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 := by
  sorry

theorem number_of_solutions : ∃! x : ℝ, |x - 5| = |x + 3| := by
  use 1
  split
  { -- First part: showing that x = 1 is a solution
    exact (fun h : 1 = 1 => by 
      rwa sub_self,
    sorry)
  },
  { -- Second part: showing that x = 1 is the only solution
    assume x hx,
    rw [hx],
    sorry  
  }

end distinct_solution_count_number_of_solutions_l278_278023


namespace solve_equation_l278_278973

theorem solve_equation (x : ℝ) :
  3 * x + 6 = abs (-20 + x^2) →
  x = (3 + Real.sqrt 113) / 2 ∨ x = (3 - Real.sqrt 113) / 2 :=
by
  sorry

end solve_equation_l278_278973


namespace units_digit_of_42_pow_3_add_24_pow_3_l278_278269

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end units_digit_of_42_pow_3_add_24_pow_3_l278_278269


namespace rebecca_swimming_problem_l278_278898

theorem rebecca_swimming_problem :
  ∃ D : ℕ, (D / 4 - D / 5) = 6 → D = 120 :=
sorry

end rebecca_swimming_problem_l278_278898


namespace parallel_lines_slope_l278_278558

-- Define the equations of the lines in Lean
def line1 (x : ℝ) : ℝ := 7 * x + 3
def line2 (c : ℝ) (x : ℝ) : ℝ := (3 * c) * x + 5

-- State the theorem: if the lines are parallel, then c = 7/3
theorem parallel_lines_slope (c : ℝ) :
  (∀ x : ℝ, (7 * x + 3 = (3 * c) * x + 5)) → c = (7/3) :=
by
  sorry

end parallel_lines_slope_l278_278558


namespace solve_for_z_l278_278864

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
  sorry

end solve_for_z_l278_278864


namespace no_int_a_divisible_289_l278_278525

theorem no_int_a_divisible_289 : ¬ ∃ a : ℤ, ∃ k : ℤ, a^2 - 3 * a - 19 = 289 * k :=
by
  sorry

end no_int_a_divisible_289_l278_278525


namespace least_number_of_marbles_l278_278138

def divisible_by (n : ℕ) (d : ℕ) : Prop := n % d = 0

theorem least_number_of_marbles 
  (n : ℕ)
  (h3 : divisible_by n 3)
  (h4 : divisible_by n 4)
  (h5 : divisible_by n 5)
  (h7 : divisible_by n 7)
  (h8 : divisible_by n 8) :
  n = 840 :=
sorry

end least_number_of_marbles_l278_278138


namespace probability_of_earning_exactly_2300_in_3_spins_l278_278650

-- Definitions of the conditions
def spinner_sections : List ℕ := [0, 1000, 200, 7000, 300]
def equal_area_sections : Prop := true  -- Each section has the same area, simple condition

-- Proving the probability of earning exactly $2300 in three spins
theorem probability_of_earning_exactly_2300_in_3_spins :
  ∃ p : ℚ, p = 3 / 125 := sorry

end probability_of_earning_exactly_2300_in_3_spins_l278_278650


namespace intersection_M_N_l278_278490

open Set

def M := { x : ℝ | 0 < x ∧ x < 3 }
def N := { x : ℝ | x^2 - 5 * x + 4 ≥ 0 }

theorem intersection_M_N :
  { x | x ∈ M ∧ x ∈ N } = { x | 0 < x ∧ x ≤ 1 } :=
sorry

end intersection_M_N_l278_278490


namespace polynomial_simplification_l278_278405

theorem polynomial_simplification :
  ∃ A B C D : ℤ,
  (∀ x : ℤ, x ≠ D → (x^3 + 5 * x^2 + 8 * x + 4) / (x + 1) = A * x^2 + B * x + C)
  ∧ (A + B + C + D = 8) :=
sorry

end polynomial_simplification_l278_278405


namespace find_g_neg_five_l278_278618

-- Given function and its properties
variables (g : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom ax2 : ∀ (x : ℝ), g x ≠ 0
axiom ax3 : g 5 = 2

-- Theorem to prove
theorem find_g_neg_five : g (-5) = 8 :=
sorry

end find_g_neg_five_l278_278618


namespace length_of_platform_proof_l278_278297

def convert_speed_to_mps (kmph : Float) : Float := kmph * (5/18)

def distance_covered (speed : Float) (time : Float) : Float := speed * time

def length_of_platform (total_distance : Float) (train_length : Float) : Float := total_distance - train_length

theorem length_of_platform_proof :
  let speed_kmph := 72.0
  let speed_mps := convert_speed_to_mps speed_kmph
  let time_seconds := 36.0
  let train_length := 470.06
  let total_distance := distance_covered speed_mps time_seconds
  length_of_platform total_distance train_length = 249.94 :=
by
  sorry

end length_of_platform_proof_l278_278297


namespace largest_natural_number_not_sum_of_two_composites_l278_278815

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278815


namespace system_infinite_solutions_l278_278681

theorem system_infinite_solutions :
  ∃ (x y : ℚ), (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = 15) ↔ (3 * x - 4 * y = 5) :=
by
  sorry

end system_infinite_solutions_l278_278681


namespace area_H1H2H3_eq_four_l278_278658

section TriangleArea

variables {P D E F H1 H2 H3 : Type*}

-- Definitions of midpoints, centroid, etc. can be implicit in Lean's formalism if necessary
-- We'll represent the area relation directly

-- Assume P is inside triangle DEF
def point_inside_triangle (P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Assume H1, H2, H3 are centroids of triangles PDE, PEF, PFD respectively
def is_centroid (H1 H2 H3 P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Given the area of triangle DEF
def area_DEF : ℝ := 12

-- Define the area function for the triangle formed by specific points
def area_triangle (A B C : Type*) : ℝ :=
sorry  -- Actual computation is abstracted

-- Mathematical statement to be proven
theorem area_H1H2H3_eq_four (P D E F H1 H2 H3 : Type*)
  (h_inside : point_inside_triangle P D E F)
  (h_centroid : is_centroid H1 H2 H3 P D E F)
  (h_area_DEF : area_triangle D E F = area_DEF) :
  area_triangle H1 H2 H3 = 4 :=
sorry

end TriangleArea

end area_H1H2H3_eq_four_l278_278658


namespace every_positive_integer_has_good_multiple_l278_278444

def is_good (n : ℕ) : Prop :=
  ∃ (D : Finset ℕ), (D.sum id = n) ∧ (1 ∈ D) ∧ (∀ d ∈ D, d ∣ n)

theorem every_positive_integer_has_good_multiple (n : ℕ) (hn : n > 0) : ∃ m : ℕ, (m % n = 0) ∧ is_good m :=
  sorry

end every_positive_integer_has_good_multiple_l278_278444


namespace find_angle_A_l278_278503

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c) :
  A = Real.pi / 3 :=
sorry

end find_angle_A_l278_278503


namespace exists_three_sticks_form_triangle_l278_278122

theorem exists_three_sticks_form_triangle 
  (l : Fin 5 → ℝ) 
  (h1 : ∀ i, 2 < l i) 
  (h2 : ∀ i, l i < 8) : 
  ∃ (i j k : Fin 5), i < j ∧ j < k ∧ 
    (l i + l j > l k) ∧ 
    (l j + l k > l i) ∧ 
    (l k + l i > l j) :=
sorry

end exists_three_sticks_form_triangle_l278_278122


namespace seeds_per_watermelon_l278_278465

theorem seeds_per_watermelon (total_seeds : ℕ) (num_watermelons : ℕ) (h : total_seeds = 400 ∧ num_watermelons = 4) : total_seeds / num_watermelons = 100 :=
by
  sorry

end seeds_per_watermelon_l278_278465


namespace tangent_line_circle_l278_278041

theorem tangent_line_circle {m : ℝ} (tangent : ∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 = m → false) : m = 2 :=
sorry

end tangent_line_circle_l278_278041


namespace length_of_AB_l278_278258

theorem length_of_AB (V : ℝ) (r : ℝ) :
  V = 216 * Real.pi →
  r = 3 →
  ∃ (len_AB : ℝ), len_AB = 20 :=
by
  intros hV hr
  have volume_cylinder := V - 36 * Real.pi
  have height_cylinder := volume_cylinder / (Real.pi * r^2)
  exists height_cylinder
  exact sorry

end length_of_AB_l278_278258


namespace number_of_distinct_digit_odd_numbers_l278_278027

theorem number_of_distinct_digit_odd_numbers (a b c d : ℕ) :
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧
  a * 1000 + b * 100 + c * 10 + d ≤ 9999 ∧
  (a * 1000 + b * 100 + c * 10 + d) % 2 = 1 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0
  → ∃ (n : ℕ), n = 2240 :=
by 
  sorry

end number_of_distinct_digit_odd_numbers_l278_278027


namespace mr_llesis_rice_cost_l278_278889

theorem mr_llesis_rice_cost :
  let total_kg : ℚ := 50
  let price_part1 : ℚ := 1.2
  let price_part2 : ℚ := 1.5
  let price_part3 : ℚ := 2
  let kg_part1 : ℚ := 20
  let kg_part2 : ℚ := 25
  let kg_part3 : ℚ := 5
  let total_cost := kg_part1 * price_part1 + kg_part2 * price_part2 + kg_part3 * price_part3
  let kept_kg := 7 / 10 * total_kg
  let given_kg := total_kg - kept_kg
  let cost_kept := kg_part1 * price_part1 + (kg_part2 * (kept_kg - kg_part1) / kg_part2) * price_part2 + kg_part3 * price_part3
  let cost_given := total_cost - cost_kept
in cost_kept - cost_given = 41.5 := sorry

end mr_llesis_rice_cost_l278_278889


namespace minimum_value_l278_278828

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem minimum_value :
  ∃ x₀ : ℝ, (∀ x : ℝ, f x₀ ≤ f x) ∧ f x₀ = -2 := by
  sorry

end minimum_value_l278_278828


namespace nested_fraction_value_l278_278847

theorem nested_fraction_value :
  1 + (1 / (1 + (1 / (2 + (2 / 3))))) = 19 / 11 :=
by sorry

end nested_fraction_value_l278_278847


namespace anniversary_sale_total_cost_l278_278582

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l278_278582


namespace largest_k_divides_2_pow_3_pow_m_add_1_l278_278611

theorem largest_k_divides_2_pow_3_pow_m_add_1 (m : ℕ) : 9 ∣ 2^(3^m) + 1 := sorry

end largest_k_divides_2_pow_3_pow_m_add_1_l278_278611


namespace anna_stops_700th_draw_l278_278703

theorem anna_stops_700th_draw (marbles : Finset (Fin 800)) 
  (colors : Fin 100 → Finset marbles) 
  (h1 : ∀ c, (colors c).card = 8)
  (h2 : ∑ c, (colors c).erase 699.card = 101)
  (h3 : ∑ c, if c.erase 699.card = 1 then 1 else 0 = 99)
  (h4 : ∑ c, if c.erase 699.card = 2 then 1 else 0 = 1) :
  probability (colors (erase 699)).erase 700.card = \frac{99}{101}) :=
sorry

end anna_stops_700th_draw_l278_278703


namespace area_triangle_ABC_is_correct_l278_278550

noncomputable def radius : ℝ := 4

noncomputable def angleABDiameter : ℝ := 30

noncomputable def ratioAM_MB : ℝ := 2 / 3

theorem area_triangle_ABC_is_correct :
  ∃ (area : ℝ), area = (180 * Real.sqrt 3) / 19 :=
by sorry

end area_triangle_ABC_is_correct_l278_278550


namespace mean_equality_l278_278694

theorem mean_equality (y : ℝ) : 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 :=
by
  intros h
  sorry

end mean_equality_l278_278694


namespace sum_first_n_odd_eq_n_squared_l278_278396

theorem sum_first_n_odd_eq_n_squared (n : ℕ) : (Finset.sum (Finset.range n) (fun k => (2 * k + 1)) = n^2) := sorry

end sum_first_n_odd_eq_n_squared_l278_278396


namespace password_combinations_check_l278_278233

theorem password_combinations_check : ∃ (s : Multiset Char), Multiset.card s = 5 ∧ (Multiset.perm s).card = 20 := by
  sorry

end password_combinations_check_l278_278233


namespace comparison_of_abc_l278_278615

noncomputable def a : ℝ := 24 / 7
noncomputable def b : ℝ := Real.log 7
noncomputable def c : ℝ := Real.log (7 / Real.exp 1) / Real.log 3 + 1

theorem comparison_of_abc :
  (a = 24 / 7) →
  (b * Real.exp b = 7 * Real.log 7) →
  (3 ^ (c - 1) = 7 / Real.exp 1) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  sorry

end comparison_of_abc_l278_278615


namespace power_division_identity_l278_278419

theorem power_division_identity : 
  ∀ (a b c : ℕ), a = 3 → b = 12 → c = 2 → (3 ^ 12 / (3 ^ 2) ^ 2 = 6561) :=
by
  intros a b c h1 h2 h3
  sorry

end power_division_identity_l278_278419


namespace no_such_number_exists_l278_278672

theorem no_such_number_exists : ¬ ∃ n : ℕ, 10^(n+1) + 35 ≡ 0 [MOD 63] :=
by {
  sorry 
}

end no_such_number_exists_l278_278672


namespace second_fraction_correct_l278_278905

theorem second_fraction_correct : 
  ∃ x : ℚ, (2 / 3) * x * (1 / 3) * (3 / 8) = 0.07142857142857142 ∧ x = 6 / 7 :=
by
  sorry

end second_fraction_correct_l278_278905


namespace find_coefficient_of_x_l278_278832

theorem find_coefficient_of_x :
  ∃ a : ℚ, ∀ (x y : ℚ),
  (x + y = 19) ∧ (x + 3 * y = 1) ∧ (2 * x + y = 5) →
  (a * x + y = 19) ∧ (a = 7) :=
by
  sorry

end find_coefficient_of_x_l278_278832


namespace domain_of_v_l278_278464

noncomputable def v (x : ℝ) : ℝ := 1 / (x ^ (1/3) + x^2 - 1)

theorem domain_of_v : ∀ x, x ≠ 1 → x ^ (1/3) + x^2 - 1 ≠ 0 :=
by
  sorry

end domain_of_v_l278_278464


namespace largest_non_sum_of_composites_l278_278791

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278791


namespace num_solutions_congruence_l278_278845

-- Define the problem context and conditions
def is_valid_solution (y : ℕ) : Prop :=
  y < 150 ∧ (y + 21) % 46 = 79 % 46

-- Define the proof problem
theorem num_solutions_congruence : ∃ (s : Finset ℕ), s.card = 3 ∧ ∀ y ∈ s, is_valid_solution y := by
  sorry

end num_solutions_congruence_l278_278845


namespace find_base_b_l278_278642

theorem find_base_b (b : ℕ) (h : (3 * b + 4) ^ 2 = b ^ 3 + 2 * b ^ 2 + 9 * b + 6) : b = 10 :=
sorry

end find_base_b_l278_278642


namespace calendar_sum_multiple_of_4_l278_278199

theorem calendar_sum_multiple_of_4 (a : ℕ) : 
  let top_left := a - 1
  let bottom_left := a + 6
  let bottom_right := a + 7
  top_left + a + bottom_left + bottom_right = 4 * (a + 3) :=
by
  sorry

end calendar_sum_multiple_of_4_l278_278199


namespace largest_number_not_sum_of_two_composites_l278_278822

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278822


namespace moses_percentage_l278_278414

theorem moses_percentage (P : ℝ) (T : ℝ) (E : ℝ) (total_amount : ℝ) (moses_more : ℝ)
  (h1 : total_amount = 50)
  (h2 : moses_more = 5)
  (h3 : T = E)
  (h4 : P / 100 * total_amount = E + moses_more)
  (h5 : 2 * E = (1 - P / 100) * total_amount) :
  P = 40 :=
by
  sorry

end moses_percentage_l278_278414


namespace geese_percentage_l278_278210

noncomputable def percentage_of_geese_among_non_swans (geese swans herons ducks : ℝ) : ℝ :=
  (geese / (100 - swans)) * 100

theorem geese_percentage (geese swans herons ducks : ℝ)
  (h1 : geese = 40)
  (h2 : swans = 20)
  (h3 : herons = 15)
  (h4 : ducks = 25) :
  percentage_of_geese_among_non_swans geese swans herons ducks = 50 :=
by
  simp [percentage_of_geese_among_non_swans, h1, h2, h3, h4]
  sorry

end geese_percentage_l278_278210


namespace range_of_a_l278_278631

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) * x^3

theorem range_of_a (a : ℝ) :
  f (Real.logb 2 a) + f (Real.logb 0.5 a) ≤ 2 * f 1 → (1/2 : ℝ) ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l278_278631


namespace map_width_l278_278589

theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) : ∃ (width : ℝ), width = 10 :=
by
  sorry

end map_width_l278_278589


namespace rational_linear_function_l278_278161

theorem rational_linear_function (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
sorry

end rational_linear_function_l278_278161


namespace correct_car_selection_l278_278451

-- Define the production volumes
def production_emgrand : ℕ := 1600
def production_king_kong : ℕ := 6000
def production_freedom_ship : ℕ := 2000

-- Define the total number of cars produced
def total_production : ℕ := production_emgrand + production_king_kong + production_freedom_ship

-- Define the number of cars selected for inspection
def cars_selected_for_inspection : ℕ := 48

-- Calculate the sampling ratio
def sampling_ratio : ℚ := cars_selected_for_inspection / total_production

-- Define the expected number of cars to be selected from each model using the sampling ratio
def cars_selected_emgrand : ℚ := sampling_ratio * production_emgrand
def cars_selected_king_kong : ℚ := sampling_ratio * production_king_kong
def cars_selected_freedom_ship : ℚ := sampling_ratio * production_freedom_ship

theorem correct_car_selection :
  cars_selected_emgrand = 8 ∧ cars_selected_king_kong = 30 ∧ cars_selected_freedom_ship = 10 := by
  sorry

end correct_car_selection_l278_278451


namespace cos_alpha_add_beta_over_2_l278_278348

variable (α β : ℝ)

-- Conditions
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : -π / 2 < β ∧ β < 0)
variables (h3 : Real.cos (π / 4 + α) = 1 / 3)
variables (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Result
theorem cos_alpha_add_beta_over_2 :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_2_l278_278348


namespace integer_product_l278_278156

open Real

theorem integer_product (P Q R S : ℕ) (h1 : P + Q + R + S = 48)
    (h2 : P + 3 = Q - 3) (h3 : P + 3 = R * 3) (h4 : P + 3 = S / 3) :
    P * Q * R * S = 5832 :=
sorry

end integer_product_l278_278156


namespace ramola_rank_last_is_14_l278_278743

-- Define the total number of students
def total_students : ℕ := 26

-- Define Ramola's rank from the start
def ramola_rank_start : ℕ := 14

-- Define a function to calculate the rank from the last given the above conditions
def ramola_rank_from_last (total_students ramola_rank_start : ℕ) : ℕ :=
  total_students - ramola_rank_start + 1

-- Theorem stating that Ramola's rank from the last is 14th
theorem ramola_rank_last_is_14 :
  ramola_rank_from_last total_students ramola_rank_start = 14 :=
by
  -- Proof goes here
  sorry

end ramola_rank_last_is_14_l278_278743


namespace no_such_polynomial_exists_l278_278896

theorem no_such_polynomial_exists :
  ¬ ∃ (P : Polynomial ℤ) (m : ℕ), ∀ (x : ℤ), Polynomial.eval x P(P(x)) = x^m + x + 2 := 
by
  sorry

end no_such_polynomial_exists_l278_278896


namespace quadratic_root_signs_l278_278404

-- Variables representation
variables {x m : ℝ}

-- Given: The quadratic equation with one positive root and one negative root
theorem quadratic_root_signs (h : ∃ a b : ℝ, 2*a*2*b + (m+1)*(a + b) + m = 0 ∧ a > 0 ∧ b < 0) : 
  m < 0 := 
sorry

end quadratic_root_signs_l278_278404


namespace common_difference_of_arithmetic_sequence_l278_278198

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 9)
  (h2 : a 5 = 33)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = 8 :=
sorry

end common_difference_of_arithmetic_sequence_l278_278198


namespace min_rectangles_to_cover_square_exactly_l278_278722

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l278_278722


namespace multiplication_correct_l278_278305

theorem multiplication_correct :
  375680169467 * 4565579427629 = 1715110767607750737263 :=
  by sorry

end multiplication_correct_l278_278305


namespace smithtown_left_handed_women_percentage_l278_278931

theorem smithtown_left_handed_women_percentage
    (x y : ℕ)
    (H1 : 3 * x + x = 4 * x)
    (H2 : 3 * y + 2 * y = 5 * y)
    (H3 : 4 * x = 5 * y) :
    (x / (4 * x)) * 100 = 25 :=
by sorry

end smithtown_left_handed_women_percentage_l278_278931


namespace Laplace_transform_ratio_l278_278211

variables {α : Type*} [MeasureSpace α] {X : ℕ → α → ℝ}
variable  {f : ℝ → ℝ}
variable  (λ : ℝ) {n : ℕ} {x y : ℝ}

-- Random variables X_i are independent and identically distributed
axiom iid_nonneg_random_vars (X : ℕ → α → ℝ) (f : ℝ → ℝ) : (∀ i, MeasureTheory.integrable (X i) ∧ ∀ i, HasPdf (λ a, f (X i a)) (f i)) ∧ ∀ i, 0 ≤ X i := sorry

-- Definitions of Sn and Mn
noncomputable def S_n := (finset.range n).sum (λ i, X i)
noncomputable def M_n := finset.max (finset.range n) (λ i, X i)

-- Definition of φ_n, the Laplace transform of S_n / M_n
noncomputable def φ_n (λ : ℝ) : ℝ :=
  n * real.exp (-λ) * ∫ (x : ℝ) in set.Ici 0, (∫ (y : ℝ) in set.Ioc 0 x, real.exp (-λ * y / x) * f y) ^ (n - 1) * f x

-- Goal statement
theorem Laplace_transform_ratio (h : iid_nonneg_random_vars X f) :
  φ_n λ = φ_n_le := sorry

end Laplace_transform_ratio_l278_278211


namespace initial_volume_of_solution_l278_278291

variable (V : ℝ)

theorem initial_volume_of_solution :
  (0.05 * V + 5.5 = 0.15 * (V + 10)) → (V = 40) :=
by
  intro h
  sorry

end initial_volume_of_solution_l278_278291


namespace smallest_num_rectangles_to_cover_square_l278_278733

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l278_278733


namespace perpendicular_slope_l278_278319

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end perpendicular_slope_l278_278319


namespace orange_bin_count_l278_278425

theorem orange_bin_count (initial_count throw_away add_new : ℕ) 
  (h1 : initial_count = 40) 
  (h2 : throw_away = 37) 
  (h3 : add_new = 7) : 
  initial_count - throw_away + add_new = 10 := 
by 
  sorry

end orange_bin_count_l278_278425


namespace tan_neg_480_eq_sqrt_3_l278_278287

theorem tan_neg_480_eq_sqrt_3 : Real.tan (-8 * Real.pi / 3) = Real.sqrt 3 :=
by
  sorry

end tan_neg_480_eq_sqrt_3_l278_278287


namespace cover_square_with_rectangles_l278_278719

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l278_278719


namespace count_divisibles_l278_278858

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278858


namespace distinct_solutions_abs_eq_l278_278026

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 5| = |x + 3|) → ∃! (x : ℝ), x = 1 :=
begin
  sorry
end

end distinct_solutions_abs_eq_l278_278026


namespace registration_methods_count_l278_278994

theorem registration_methods_count (students : Fin 4) (groups : Fin 3) : (3 : ℕ)^4 = 81 :=
by
  sorry

end registration_methods_count_l278_278994


namespace gcd_40304_30203_eq_1_l278_278473

theorem gcd_40304_30203_eq_1 : Nat.gcd 40304 30203 = 1 := 
by 
  sorry

end gcd_40304_30203_eq_1_l278_278473


namespace tens_digit_of_13_pow_3007_l278_278555

theorem tens_digit_of_13_pow_3007 : 
  (13 ^ 3007 / 10) % 10 = 1 :=
sorry

end tens_digit_of_13_pow_3007_l278_278555


namespace find_number_l278_278943

theorem find_number (x : ℝ) :
  10 * x - 10 = 50 ↔ x = 6 := by
  sorry

end find_number_l278_278943


namespace f_2021_value_l278_278059

def A : Set ℚ := {x | x ≠ -1 ∧ x ≠ 0}

def f (x : ℚ) : ℝ := sorry -- Placeholder for function definition with its properties

axiom f_property : ∀ x ∈ A, f x + f (1 + 1 / x) = 1 / 2 * Real.log (|x|)

theorem f_2021_value : f 2021 = 1 / 2 * Real.log 2021 :=
by
  sorry

end f_2021_value_l278_278059


namespace largest_not_sum_of_two_composites_l278_278817

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278817


namespace digit_makes_divisible_by_nine_l278_278917

theorem digit_makes_divisible_by_nine (A : ℕ) : (7 + A + 4 + 6) % 9 = 0 ↔ A = 1 :=
by
  sorry

end digit_makes_divisible_by_nine_l278_278917


namespace max_min_diff_eq_l278_278691

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 2) - Real.sqrt (x^2 - 3*x + 3)

theorem max_min_diff_eq : 
  (∀ x : ℝ, ∃ max min : ℝ, max = Real.sqrt (8 - Real.sqrt 3) ∧ min = -Real.sqrt (8 - Real.sqrt 3) ∧ 
  (max - min = 2 * Real.sqrt (8 - Real.sqrt 3))) :=
sorry

end max_min_diff_eq_l278_278691


namespace largest_number_not_sum_of_two_composites_l278_278823

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278823


namespace rhombus_diagonal_length_l278_278137

theorem rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) (d1 : ℝ)
  (h_area : area = 432) 
  (h_d2 : d2 = 24) :
  d1 = 36 :=
by
  sorry

end rhombus_diagonal_length_l278_278137


namespace find_a_l278_278012

theorem find_a (a x : ℝ)
    (h1 : 6 * (x + 8) = 18 * x)
    (h2 : 6 * x - 2 * (a - x) = 2 * a + x) :
    a = 7 :=
  sorry

end find_a_l278_278012


namespace vacation_cost_l278_278103

theorem vacation_cost (C : ℝ) (h : C / 3 - C / 4 = 60) : C = 720 := 
by sorry

end vacation_cost_l278_278103


namespace anniversary_sale_total_cost_l278_278580

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l278_278580


namespace largest_non_sum_of_composites_l278_278790

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278790


namespace rectangle_area_l278_278088

noncomputable def length (w : ℝ) : ℝ := 4 * w

noncomputable def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area :
  ∀ (l w : ℝ), 
  l = length w ∧ perimeter l w = 200 → area l w = 1600 :=
by
  intros l w h
  cases h with h1 h2
  rw [length, perimeter, area] at *
  sorry

end rectangle_area_l278_278088


namespace inequality_represents_area_l278_278910

theorem inequality_represents_area (a : ℝ) :
  (if a > 1 then ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y < - (x + 3) / (a - 1)
  else ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y > - (x + 3) / (a - 1)) :=
by sorry

end inequality_represents_area_l278_278910


namespace smallest_number_of_rectangles_needed_l278_278556

theorem smallest_number_of_rectangles_needed :
  ∃ n, (n * 12 = 144) ∧ (∀ k, (k * 12 = 144) → k ≥ n) := by
  sorry

end smallest_number_of_rectangles_needed_l278_278556


namespace evaluate_at_points_l278_278884

noncomputable def f (x : ℝ) : ℝ :=
if x > 3 then x^2 - 3*x + 2
else if -2 ≤ x ∧ x ≤ 3 then -3*x + 5
else 9

theorem evaluate_at_points : f (-3) + f (0) + f (4) = 20 := by
  sorry

end evaluate_at_points_l278_278884


namespace largest_non_representable_as_sum_of_composites_l278_278802

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278802


namespace unit_squares_in_50th_ring_l278_278779

-- Definitions from the conditions
def unit_squares_in_first_ring : ℕ := 12

def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  32 * n - 16

-- Prove the specific instance for the 50th ring
theorem unit_squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1584 :=
by
  sorry

end unit_squares_in_50th_ring_l278_278779


namespace lukas_points_in_5_games_l278_278281

theorem lukas_points_in_5_games (avg_points_per_game : ℕ) (games_played : ℕ) (total_points : ℕ)
  (h_avg : avg_points_per_game = 12) (h_games : games_played = 5) : total_points = 60 :=
by
  sorry

end lukas_points_in_5_games_l278_278281


namespace prob_exactly_one_hits_prob_at_least_one_hits_l278_278072

noncomputable def prob_A_hits : ℝ := 1 / 2
noncomputable def prob_B_hits : ℝ := 1 / 3
noncomputable def prob_A_misses : ℝ := 1 - prob_A_hits
noncomputable def prob_B_misses : ℝ := 1 - prob_B_hits

theorem prob_exactly_one_hits :
  (prob_A_hits * prob_B_misses) + (prob_A_misses * prob_B_hits) = 1 / 2 :=
by sorry

theorem prob_at_least_one_hits :
  1 - (prob_A_misses * prob_B_misses) = 2 / 3 :=
by sorry

end prob_exactly_one_hits_prob_at_least_one_hits_l278_278072


namespace necessary_but_not_sufficient_l278_278284

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l278_278284


namespace antecedent_is_50_l278_278195

theorem antecedent_is_50 (antecedent consequent : ℕ) (h_ratio : 4 * consequent = 6 * antecedent) (h_consequent : consequent = 75) : antecedent = 50 := by
  sorry

end antecedent_is_50_l278_278195


namespace eq_perp_bisector_BC_area_triangle_ABC_l278_278486

section Triangle_ABC

open Real

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the equation of the perpendicular bisector
theorem eq_perp_bisector_BC : ∀ x y : ℝ, 2 * x + y - 4 = 0 :=
sorry

-- Define the area of the triangle ABC
noncomputable def triangle_area : ℝ :=
1 / 2 * (abs ((-1 * 3 + 3 * (-2) + 3 * 4) - (3 * 4 + 1 * (-2) + 3*(-1))))

theorem area_triangle_ABC : triangle_area = 7 :=
sorry

end Triangle_ABC

end eq_perp_bisector_BC_area_triangle_ABC_l278_278486


namespace vacation_days_l278_278056

-- A plane ticket costs $24 for each person
def plane_ticket_cost : ℕ := 24

-- A hotel stay costs $12 for each person per day
def hotel_stay_cost_per_day : ℕ := 12

-- Total vacation cost is $120
def total_vacation_cost : ℕ := 120

-- The number of days they are planning to stay is 3
def number_of_days : ℕ := 3

-- Prove that given the conditions, the number of days (d) they plan to stay satisfies the total vacation cost
theorem vacation_days (d : ℕ) (plane_ticket_cost hotel_stay_cost_per_day total_vacation_cost : ℕ) 
  (h1 : plane_ticket_cost = 24)
  (h2 : hotel_stay_cost_per_day = 12) 
  (h3 : total_vacation_cost = 120) 
  (h4 : 2 * plane_ticket_cost + (2 * hotel_stay_cost_per_day) * d = total_vacation_cost)
  : d = 3 := sorry

end vacation_days_l278_278056


namespace solve_for_q_l278_278470

theorem solve_for_q (q : ℕ) : 16^4 = (8^3 / 2 : ℕ) * 2^(16 * q) → q = 1 / 2 :=
by
  sorry

end solve_for_q_l278_278470


namespace min_x_y_l278_278837

theorem min_x_y
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x + 2 * y + x * y - 7 = 0) :
  x + y ≥ 3 := by
  sorry

end min_x_y_l278_278837


namespace even_iff_a_zero_monotonous_iff_a_range_max_value_l278_278343

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 2

-- (I) Prove that f(x) is even on [-5, 5] if and only if a = 0
theorem even_iff_a_zero (a : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a = f (-x) a) ↔ a = 0 := sorry

-- (II) Prove that f(x) is monotonous on [-5, 5] if and only if a ≥ 10 or a ≤ -10
theorem monotonous_iff_a_range (a : ℝ) : (∀ x y : ℝ, -5 ≤ x ∧ x ≤ y ∧ y ≤ 5 → f x a ≤ f y a) ↔ (a ≥ 10 ∨ a ≤ -10) := sorry

-- (III) Prove the maximum value of f(x) in the interval [-5, 5]
theorem max_value (a : ℝ) : (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ (∀ y : ℝ, -5 ≤ y ∧ y ≤ 5 → f y a ≤ f x a)) ∧  
                           ((a ≥ 0 → f 5 a = 27 + 5 * a) ∧ (a < 0 → f (-5) a = 27 - 5 * a)) := sorry

end even_iff_a_zero_monotonous_iff_a_range_max_value_l278_278343


namespace range_of_b_div_c_l278_278049

theorem range_of_b_div_c (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : b^2 = c^2 + a * c) :
  1 < b / c ∧ b / c < 2 := 
sorry

end range_of_b_div_c_l278_278049


namespace original_selling_price_l278_278744

-- Definitions based on the conditions
def original_price : ℝ := 933.33

-- Given conditions
def discount_rate : ℝ := 0.40
def price_after_discount : ℝ := 560.0

-- Lean theorem statement to prove that original selling price (x) is equal to 933.33
theorem original_selling_price (x : ℝ) 
  (h1 : x * (1 - discount_rate) = price_after_discount) : 
  x = original_price :=
  sorry

end original_selling_price_l278_278744


namespace time_to_groom_rottweiler_l278_278208

theorem time_to_groom_rottweiler
  (R : ℕ)  -- Time to groom a rottweiler
  (B : ℕ)  -- Time to groom a border collie
  (C : ℕ)  -- Time to groom a chihuahua
  (total_time_6R_9B_1C : 6 * R + 9 * B + C = 255)  -- Total time for grooming 6 rottweilers, 9 border collies, and 1 chihuahua
  (time_to_groom_border_collie : B = 10)  -- Time to groom a border collie is 10 minutes
  (time_to_groom_chihuahua : C = 45) :  -- Time to groom a chihuahua is 45 minutes
  R = 20 :=  -- Prove that it takes 20 minutes to groom a rottweiler
by
  sorry

end time_to_groom_rottweiler_l278_278208


namespace smallest_num_rectangles_to_cover_square_l278_278732

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l278_278732


namespace length_of_AB_l278_278257

theorem length_of_AB {L : ℝ} (h : 9 * Real.pi * L + 36 * Real.pi = 216 * Real.pi) : L = 20 :=
sorry

end length_of_AB_l278_278257


namespace number_is_375_l278_278035

theorem number_is_375 (x : ℝ) (h : (40 / 100) * x = (30 / 100) * 50) : x = 37.5 :=
sorry

end number_is_375_l278_278035


namespace sufficient_but_not_necessary_condition_l278_278478

open Real

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (x = y → |x| = |y|) ∧ (|x| = |y| → x = y) = false :=
by
  sorry

end sufficient_but_not_necessary_condition_l278_278478


namespace maximum_expression_value_l278_278240

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end maximum_expression_value_l278_278240


namespace best_play_wins_probability_l278_278755

/-- Define the conditions and parameters for the problem. -/
variables (n m : ℕ)
variables (C : ℕ → ℕ → ℕ) /- Binomial coefficient -/

/-- Define the probability calculation -/
def probability_best_play_wins : ℚ :=
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t)

/-- The theorem stating that the above calculation represents the probability of the best play winning -/
theorem best_play_wins_probability :
  probability_best_play_wins n m C =
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t) :=
  by
  sorry

end best_play_wins_probability_l278_278755


namespace smallest_S_value_l278_278254

theorem smallest_S_value :
  ∃ (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ),
  (∀ x ∈ {a1, a2, a3, b1, b2, b3, c1, c2, c3, d}, x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
  (∀ x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ∃! y, y ∈ {a1, a2, a3, b1, b2, b3, c1, c2, c3, d} ∧ y = x) ∧
  minimal_value a1 a2 a3 b1 b2 b3 c1 c2 c3 d = 609 :=
by
  sorry

def minimal_value (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ) : ℕ :=
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 + d

end smallest_S_value_l278_278254


namespace like_terms_expressions_l278_278996

theorem like_terms_expressions (m n : ℤ) :
  (∀ x y : ℝ, -3 * x ^ (m - 1) * y ^ 3 = 4 * x * y ^ (m + n)) → (m = 2 ∧ n = 1) :=
by
  intro h
  have h_mx_pow : m - 1 = 1 := sorry
  have h_my_pow : 3 = m + n := sorry
  finish

end like_terms_expressions_l278_278996


namespace strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l278_278360

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

theorem strictly_increasing : ∃ A : Set ℝ, A = (Set.Ioo 0 1) ∧ (∀ x y, x ∈ A → y ∈ A → x < y → f x < f y) :=
  sorry

theorem not_gamma_interval : ¬(Set.Icc (1/2) (3/2) ⊆ Set.Ioo 0 1 ∧ 
  (∀ x ∈ Set.Icc (1/2) (3/2), f x ∈ Set.Icc (1/(3/2)) (1/(1/2)))) :=
  sorry

theorem gamma_interval_within_one_inf : ∃ m n : ℝ, 1 ≤ m ∧ m < n ∧ 
  Set.Icc m n = Set.Icc 1 ((1 + Real.sqrt 5) / 2) ∧ 
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (1/n) (1/m)) :=
  sorry

end strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l278_278360


namespace lindas_average_speed_l278_278663

theorem lindas_average_speed
  (dist1 : ℕ) (time1 : ℝ)
  (dist2 : ℕ) (time2 : ℝ)
  (h1 : dist1 = 450)
  (h2 : time1 = 7.5)
  (h3 : dist2 = 480)
  (h4 : time2 = 8) :
  (dist1 + dist2) / (time1 + time2) = 60 :=
by
  sorry

end lindas_average_speed_l278_278663


namespace quadratic_inequality_l278_278622

noncomputable def exists_real_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0

noncomputable def valid_values (a : ℝ) : Prop :=
  a > 5 / 2 ∧ a < 10

theorem quadratic_inequality (a : ℝ) 
  (h1 : exists_real_roots a) 
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0 
  → (1 / x1 + 1 / x2 < -3 / 5)) :
  valid_values a :=
sorry

end quadratic_inequality_l278_278622


namespace marie_saves_money_in_17_days_l278_278664

noncomputable def number_of_days_needed (cash_register_cost revenue tax_rate costs : ℝ) : ℕ := 
  let net_revenue := revenue / (1 + tax_rate) 
  let daily_profit := net_revenue - costs
  Nat.ceil (cash_register_cost / daily_profit)

def marie_problem_conditions : Prop := 
  let bread_daily_revenue := 40 * 2
  let bagels_daily_revenue := 20 * 1.5
  let cakes_daily_revenue := 6 * 12
  let muffins_daily_revenue := 10 * 3
  let daily_revenue := bread_daily_revenue + bagels_daily_revenue + cakes_daily_revenue + muffins_daily_revenue
  let fixed_daily_costs := 20 + 2 + 80 + 30
  fixed_daily_costs = 132 ∧ daily_revenue = 212 ∧ 8 / 100 = 0.08

theorem marie_saves_money_in_17_days : marie_problem_conditions → number_of_days_needed 1040 212 0.08 132 = 17 := 
by 
  intro h
  -- Proof goes here.
  sorry

end marie_saves_money_in_17_days_l278_278664


namespace who_stole_the_broth_l278_278873

-- Define the suspects
inductive Suspect
| MarchHare : Suspect
| MadHatter : Suspect
| Dormouse : Suspect

open Suspect

-- Define the statements
def stole_broth (s : Suspect) : Prop :=
  s = Dormouse

def told_truth (s : Suspect) : Prop :=
  s = Dormouse

-- The March Hare's testimony
def march_hare_testimony : Prop :=
  stole_broth MadHatter

-- Conditions
def condition1 : Prop := ∃! s, stole_broth s
def condition2 : Prop := ∀ s, told_truth s ↔ stole_broth s
def condition3 : Prop := told_truth MarchHare → stole_broth MadHatter

-- Combining conditions into a single proposition to prove
theorem who_stole_the_broth : 
  (condition1 ∧ condition2 ∧ condition3) → stole_broth Dormouse := sorry

end who_stole_the_broth_l278_278873


namespace largest_d_in_range_l278_278602

theorem largest_d_in_range (d : ℝ) (g : ℝ → ℝ) :
  (g x = x^2 - 6x + d) → (∃ x : ℝ, g x = 2) → d ≤ 11 :=
by
  sorry

end largest_d_in_range_l278_278602


namespace sun_city_population_greater_than_twice_roseville_l278_278527

-- Conditions
def willowdale_population : ℕ := 2000
def roseville_population : ℕ := 3 * willowdale_population - 500
def sun_city_population : ℕ := 12000

-- Theorem
theorem sun_city_population_greater_than_twice_roseville :
  sun_city_population = 2 * roseville_population + 1000 :=
by
  -- The proof is omitted as per the problem statement
  sorry

end sun_city_population_greater_than_twice_roseville_l278_278527


namespace extra_apples_proof_l278_278099

def total_apples (red_apples : ℕ) (green_apples : ℕ) : ℕ :=
  red_apples + green_apples

def apples_taken_by_students (students : ℕ) : ℕ :=
  students

def extra_apples (total_apples : ℕ) (apples_taken : ℕ) : ℕ :=
  total_apples - apples_taken

theorem extra_apples_proof
  (red_apples : ℕ) (green_apples : ℕ) (students : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21) :
  extra_apples (total_apples red_apples green_apples) (apples_taken_by_students students) = 35 :=
by
  sorry

end extra_apples_proof_l278_278099


namespace find_a_l278_278868

theorem find_a (a : ℝ) (hne : a ≠ 1) (eq_sets : ∀ x : ℝ, (a-1) * x < a + 5 ↔ 2 * x < 4) : a = 7 :=
sorry

end find_a_l278_278868


namespace roots_quadratic_expression_l278_278381

theorem roots_quadratic_expression :
  ∀ (a b : ℝ), (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) → 
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b * (a + b) = 533 :=
by
  intros a b h
  sorry

end roots_quadratic_expression_l278_278381


namespace perpendicular_slope_l278_278321

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end perpendicular_slope_l278_278321


namespace sum_of_perimeters_l278_278578

theorem sum_of_perimeters (s : ℝ) : (∀ n : ℕ, n >= 0) → 
  (∑' n : ℕ, (4 * s) / (2 ^ n)) = 8 * s :=
by
  sorry

end sum_of_perimeters_l278_278578


namespace Option_C_correct_l278_278273

theorem Option_C_correct (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = - x * y^2 :=
by
  sorry

end Option_C_correct_l278_278273


namespace Nunzio_eats_pizza_every_day_l278_278890

theorem Nunzio_eats_pizza_every_day
  (one_piece_fraction : ℚ := 1/8)
  (total_pizzas : ℕ := 27)
  (total_days : ℕ := 72)
  (pieces_per_pizza : ℕ := 8)
  (total_pieces : ℕ := total_pizzas * pieces_per_pizza)
  : (total_pieces / total_days = 3) :=
by
  -- We assume 1/8 as a fraction for the pieces of pizza is stated in the conditions, therefore no condition here.
  -- We need to show that Nunzio eats 3 pieces of pizza every day given the total pieces and days.
  sorry

end Nunzio_eats_pizza_every_day_l278_278890


namespace largest_non_representable_as_sum_of_composites_l278_278803

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278803


namespace consumer_credit_amount_l278_278303

theorem consumer_credit_amount
  (C A : ℝ)
  (h1 : A = 0.20 * C)
  (h2 : 57 = 1/3 * A) :
  C = 855 := by
  sorry

end consumer_credit_amount_l278_278303


namespace combined_work_time_l278_278191

theorem combined_work_time (W : ℝ) (A B C : ℝ) (ha : A = W / 12) (hb : B = W / 18) (hc : C = W / 9) : 
  1 / (A + B + C) = 4 := 
by sorry

end combined_work_time_l278_278191


namespace incorrect_comparison_l278_278112

theorem incorrect_comparison :
  ¬ (- (2 / 3) < - (4 / 5)) :=
by
  sorry

end incorrect_comparison_l278_278112


namespace arithmetic_sum_sequences_l278_278993

theorem arithmetic_sum_sequences (a b : ℕ → ℕ) (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h2 : ∀ n, b n = b 0 + n * (b 1 - b 0)) (h3 : a 2 + b 2 = 3) (h4 : a 4 + b 4 = 5): a 7 + b 7 = 8 := by
  sorry

end arithmetic_sum_sequences_l278_278993


namespace triangle_proof_problem_l278_278196

-- The conditions and question programmed as a Lean theorem statement
theorem triangle_proof_problem
    (A B C : ℝ)
    (h1 : A > B)
    (S T : ℝ)
    (h2 : A = C)
    (K : ℝ)
    (arc_mid_A : K = A): -- K is midpoint of the arc A
    
    RS = K := sorry

end triangle_proof_problem_l278_278196


namespace probability_A_score_not_less_than_135_l278_278750

/-- A certain school organized a competition with the following conditions:
  - The test has 25 multiple-choice questions, each with 4 options.
  - Each correct answer scores 6 points, each unanswered question scores 2 points, and each wrong answer scores 0 points.
  - Both candidates answered the first 20 questions correctly.
  - Candidate A will attempt only the last 3 questions, and for each, A can eliminate 1 wrong option,
    hence the probability of answering any one question correctly is 1/3.
  - A gives up the last 2 questions.
  - Prove that the probability that A's total score is not less than 135 points is equal to 7/27.
-/
theorem probability_A_score_not_less_than_135 :
  let prob_success := 1 / 3
  let prob_2_successes := (3 * (prob_success^2) * (2/3))
  let prob_3_successes := (prob_success^3)
  prob_2_successes + prob_3_successes = 7 / 27 := 
by
  sorry

end probability_A_score_not_less_than_135_l278_278750


namespace minimum_squares_required_l278_278955

theorem minimum_squares_required (length : ℚ) (width : ℚ) (M N : ℕ) :
  (length = 121 / 2) → (width = 143 / 3) → (M / N = 33 / 26) → (M * N = 858) :=
by
  intros hL hW hMN
  -- Proof skipped
  sorry

end minimum_squares_required_l278_278955


namespace lena_candy_bars_l278_278379

/-- Lena has some candy bars. She needs 5 more candy bars to have 3 times as many as Kevin,
and Kevin has 4 candy bars less than Nicole. Lena has 5 more candy bars than Nicole.
How many candy bars does Lena have? -/
theorem lena_candy_bars (L K N : ℕ) 
  (h1 : L + 5 = 3 * K)
  (h2 : K = N - 4)
  (h3 : L = N + 5) : 
  L = 16 :=
sorry

end lena_candy_bars_l278_278379


namespace volume_third_bottle_is_250_milliliters_l278_278773

-- Define the volumes of the bottles in milliliters
def volume_first_bottle : ℕ := 2 * 1000                        -- 2000 milliliters
def volume_second_bottle : ℕ := 750                            -- 750 milliliters
def total_volume : ℕ := 3 * 1000                               -- 3000 milliliters
def volume_third_bottle : ℕ := total_volume - (volume_first_bottle + volume_second_bottle)

-- The theorem stating the volume of the third bottle
theorem volume_third_bottle_is_250_milliliters :
  volume_third_bottle = 250 :=
by
  sorry

end volume_third_bottle_is_250_milliliters_l278_278773


namespace simplify_expression_l278_278170

theorem simplify_expression (x y z : ℝ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (hx2 : x ≠ 2) (hy3 : y ≠ 3) (hz5 : z ≠ 5) :
  ( ( (x - 2) / (3 - z) * ( (y - 3) / (5 - x) ) * ( (z - 5) / (2 - y) ) ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l278_278170


namespace taco_truck_profit_l278_278765

-- Definitions and conditions
def pounds_of_beef : ℕ := 100
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Desired profit result
def expected_profit : ℝ := 200

-- The proof statement (to be completed)
theorem taco_truck_profit :
  let tacos := pounds_of_beef / beef_per_taco;
  let revenue := tacos * price_per_taco;
  let cost := tacos * cost_per_taco;
  let profit := revenue - cost;
  profit = expected_profit :=
by
  sorry

end taco_truck_profit_l278_278765


namespace x_coordinate_of_point_l278_278915

theorem x_coordinate_of_point (x_1 n : ℝ) 
  (h1 : x_1 = (n / 5) - (2 / 5)) 
  (h2 : x_1 + 3 = ((n + 15) / 5) - (2 / 5)) : 
  x_1 = (n / 5) - (2 / 5) :=
by sorry

end x_coordinate_of_point_l278_278915


namespace largest_not_sum_of_two_composites_l278_278816

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278816


namespace binom_20_5_l278_278306

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end binom_20_5_l278_278306


namespace solve_inequality_l278_278902

theorem solve_inequality (x : ℝ) : 3 * (x + 1) > 9 → x > 2 :=
by sorry

end solve_inequality_l278_278902


namespace area_of_F1_M_F2_l278_278939

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 9 - y^2 / 4 = 1)

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Foci (F1 F2 : Point) (hyper : ∀ p : Point, hyperbola p.x p.y → Prop) : Prop :=
  (on_hyperbola : ∀ p : Point, hyperbola p.x p.y → True)
  (angle_60_deg : ∀ M : Point, hyperbola M.x M.y → 
    let d_F1_M := Real.sqrt ((M.x - F1.x)^2 + (M.y - F1.y)^2) in
    let d_F2_M := Real.sqrt ((M.x - F2.x)^2 + (M.y - F2.y)^2) in
    (d_F1_M^2 + d_F2_M^2 - d_F1_M * d_F2_M = 52) → 
    Real.cos (Real.pi / 3) = 1/2)

noncomputable def Area_of_Triangle (F1 F2 M : Point) : ℝ :=
  let d_F1_M := Real.sqrt ((M.x - F1.x)^2 + (M.y - F1.y)^2) in
  let d_F2_M := Real.sqrt ((M.x - F2.x)^2 + (M.y - F2.y)^2) in
  let d_F1_F2 := Real.sqrt ((F2.x - F1.x)^2 + (F2.y - F1.y)^2) in
  1/2 * d_F1_M * d_F2_M * Real.sin (Real.pi / 3)

theorem area_of_F1_M_F2 {F1 F2 M : Point} (hFoci : Foci F1 F2 hyperbola)
  (hM_on_hyper : hyperbola M.x M.y) (h60 : ∀ M : Point,
    hyperbola M.x M.y → Real.cos (Real.pi / 3) = 1/2) :
  Area_of_Triangle F1 F2 M = 4 * Real.sqrt 3 :=
  sorry

end area_of_F1_M_F2_l278_278939


namespace Oliver_Battle_Gremlins_Card_Count_l278_278668

theorem Oliver_Battle_Gremlins_Card_Count 
  (MonsterClubCards AlienBaseballCards BattleGremlinsCards : ℕ)
  (h1 : MonsterClubCards = 2 * AlienBaseballCards)
  (h2 : BattleGremlinsCards = 3 * AlienBaseballCards)
  (h3 : MonsterClubCards = 32) : 
  BattleGremlinsCards = 48 := by
  sorry

end Oliver_Battle_Gremlins_Card_Count_l278_278668


namespace pipe_fill_time_with_leak_l278_278521

theorem pipe_fill_time_with_leak (A L : ℝ) (hA : A = 1 / 2) (hL : L = 1 / 6) :
  (1 / (A - L)) = 3 :=
by
  sorry

end pipe_fill_time_with_leak_l278_278521


namespace solve_system_of_equations_l278_278078

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ xyz = -16 ↔ 
  (x = 1 ∧ y = 4 ∧ z = -4) ∨ (x = 1 ∧ y = -4 ∧ z = 4) ∨ 
  (x = 4 ∧ y = 1 ∧ z = -4) ∨ (x = 4 ∧ y = -4 ∧ z = 1) ∨ 
  (x = -4 ∧ y = 1 ∧ z = 4) ∨ (x = -4 ∧ y = 4 ∧ z = 1) := 
by
  sorry

end solve_system_of_equations_l278_278078


namespace cyclic_quadrilateral_AC_plus_BD_l278_278880

theorem cyclic_quadrilateral_AC_plus_BD (AB BC CD DA : ℝ) (AC BD : ℝ) (h1 : AB = 5) (h2 : BC = 10) (h3 : CD = 11) (h4 : DA = 14)
  (h5 : AC = Real.sqrt 221) (h6 : BD = 195 / Real.sqrt 221) :
  AC + BD = 416 / Real.sqrt (13 * 17) ∧ (AC = Real.sqrt 221 ∧ BD = 195 / Real.sqrt 221) →
  (AC + BD = 416 / Real.sqrt (13 * 17)) ∧ (AC + BD = 446) :=
by
  sorry

end cyclic_quadrilateral_AC_plus_BD_l278_278880


namespace min_rectangles_to_cover_square_exactly_l278_278725

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l278_278725


namespace number_of_cookies_l278_278392

def candy : ℕ := 63
def brownies : ℕ := 21
def people : ℕ := 7
def dessert_per_person : ℕ := 18

theorem number_of_cookies : 
  (people * dessert_per_person) - (candy + brownies) = 42 := 
by
  sorry

end number_of_cookies_l278_278392


namespace reciprocal_neg_one_over_2011_l278_278256

theorem reciprocal_neg_one_over_2011 : 1 / (- (1 / 2011)) = -2011 :=
by
  sorry

end reciprocal_neg_one_over_2011_l278_278256


namespace probability_white_or_red_l278_278648

theorem probability_white_or_red (a b c : ℕ) : 
  (a + b) / (a + b + c) = (a + b) / (a + b + c) := by
  -- Conditions
  let total_balls := a + b + c
  let white_red_balls := a + b
  -- Goal
  have prob_white_or_red := white_red_balls / total_balls
  exact rfl

end probability_white_or_red_l278_278648


namespace games_bought_l278_278669

def initial_money : ℕ := 35
def spent_money : ℕ := 7
def cost_per_game : ℕ := 4

theorem games_bought : (initial_money - spent_money) / cost_per_game = 7 := by
  sorry

end games_bought_l278_278669


namespace clock_equiv_to_square_l278_278891

theorem clock_equiv_to_square : ∃ h : ℕ, h > 5 ∧ (h^2 - h) % 24 = 0 ∧ h = 9 :=
by 
  let h := 9
  use h
  refine ⟨by decide, by decide, rfl⟩ 

end clock_equiv_to_square_l278_278891


namespace smallest_num_rectangles_l278_278726

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l278_278726


namespace team_members_run_distance_l278_278411

-- Define the given conditions
def total_distance : ℕ := 150
def members : ℕ := 5

-- Prove the question == answer given the conditions
theorem team_members_run_distance :
  total_distance / members = 30 :=
by
  sorry

end team_members_run_distance_l278_278411


namespace cos_alpha_condition_l278_278688

theorem cos_alpha_condition (k : ℤ) (α : ℝ) :
  (α = 2 * k * Real.pi - Real.pi / 4 -> Real.cos α = Real.sqrt 2 / 2) ∧
  (Real.cos α = Real.sqrt 2 / 2 -> ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 4 ∨ α = 2 * k * Real.pi - Real.pi / 4) :=
by
  sorry

end cos_alpha_condition_l278_278688


namespace brown_gumdrops_after_replacement_l278_278573

-- Definitions based on the given conditions.
def total_gumdrops (green_gumdrops : ℕ) : ℕ :=
  (green_gumdrops * 100) / 15

def blue_gumdrops (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 25 / 100

def brown_gumdrops_initial (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 15 / 100

def brown_gumdrops_final (brown_initial : ℕ) (blue_gumdrops : ℕ) : ℕ :=
  brown_initial + blue_gumdrops / 3

-- The main theorem statement based on the proof problem.
theorem brown_gumdrops_after_replacement
  (green_gumdrops : ℕ)
  (h_green : green_gumdrops = 36)
  : brown_gumdrops_final (brown_gumdrops_initial (total_gumdrops green_gumdrops)) 
                         (blue_gumdrops (total_gumdrops green_gumdrops))
    = 56 := 
  by sorry

end brown_gumdrops_after_replacement_l278_278573


namespace squares_in_region_l278_278032

theorem squares_in_region :
  let bounded_region (x y : ℤ) := y ≤ 2 * x ∧ y ≥ -1 ∧ x ≤ 6
  ∃ n : ℕ, ∀ (a b : ℤ), bounded_region a b → n = 118
:= 
  sorry

end squares_in_region_l278_278032


namespace smallest_num_rectangles_l278_278729

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l278_278729


namespace triplet_divisibility_cond_l278_278970

theorem triplet_divisibility_cond (a b c : ℤ) (hac : a ≥ 2) (hbc : b ≥ 2) (hcc : c ≥ 2) :
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ 
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 3 ∧ b = 15 ∧ c = 5) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 2 ∧ b = 8 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end triplet_divisibility_cond_l278_278970


namespace boys_passed_l278_278119

theorem boys_passed (total_boys : ℕ) (avg_marks : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) (P : ℕ) 
    (h1 : total_boys = 120) (h2 : avg_marks = 36) (h3 : avg_passed = 39) (h4 : avg_failed = 15)
    (h5 : P + (total_boys - P) = 120) 
    (h6 : P * avg_passed + (total_boys - P) * avg_failed = total_boys * avg_marks) :
    P = 105 := 
sorry

end boys_passed_l278_278119


namespace batsman_avg_increase_l278_278568

theorem batsman_avg_increase (R : ℕ) (A : ℕ) : 
  (R + 48 = 12 * 26) ∧ (R = 11 * A) → 26 - A = 2 :=
by
  intro h
  have h1 : R + 48 = 312 := h.1
  have h2 : R = 11 * A := h.2
  sorry

end batsman_avg_increase_l278_278568


namespace quadratic_decreasing_on_nonneg_real_l278_278499

theorem quadratic_decreasing_on_nonneg_real (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) : 
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → (a * x^2 + b * x + c) ≥ (a * y^2 + b * y + c) :=
by
  sorry

end quadratic_decreasing_on_nonneg_real_l278_278499


namespace least_additional_squares_needed_for_symmetry_l278_278308

-- Conditions
def grid_size : ℕ := 5
def initial_shaded_squares : List (ℕ × ℕ) := [(1, 5), (3, 3), (5, 1)]

-- Goal statement
theorem least_additional_squares_needed_for_symmetry
  (grid_size : ℕ)
  (initial_shaded_squares : List (ℕ × ℕ)) : 
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℕ), (x, y) ∈ initial_shaded_squares ∨ (grid_size - x + 1, y) ∈ initial_shaded_squares ∨ (x, grid_size - y + 1) ∈ initial_shaded_squares ∨ (grid_size - x + 1, grid_size - y + 1) ∈ initial_shaded_squares) :=
sorry

end least_additional_squares_needed_for_symmetry_l278_278308


namespace compute_expression_l278_278457

theorem compute_expression : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end compute_expression_l278_278457


namespace sam_dads_dimes_l278_278899

theorem sam_dads_dimes (original_dimes new_dimes given_dimes : ℕ) 
  (h1 : original_dimes = 9)
  (h2 : new_dimes = 16)
  (h3 : new_dimes = original_dimes + given_dimes) : 
  given_dimes = 7 := 
by 
  sorry

end sam_dads_dimes_l278_278899


namespace amounts_are_correct_l278_278226

theorem amounts_are_correct (P Q R S : ℕ) 
    (h1 : P + Q + R + S = 10000)
    (h2 : R = 2 * P)
    (h3 : R = 3 * Q)
    (h4 : S = P + Q) :
    P = 1875 ∧ Q = 1250 ∧ R = 3750 ∧ S = 3125 := by
  sorry

end amounts_are_correct_l278_278226


namespace triangle_obtuse_l278_278044

theorem triangle_obtuse 
  (A B : ℝ)
  (hA : 0 < A ∧ A < π/2)
  (hB : 0 < B ∧ B < π/2)
  (h_cosA_gt_sinB : Real.cos A > Real.sin B) :
  π - (A + B) > π/2 ∧ π - (A + B) < π :=
by
  sorry

end triangle_obtuse_l278_278044


namespace expected_games_is_14_l278_278903

-- Define the conditions of the problem
variable (C : Prop)
  [hC : C ↔ (∀ (player : ℕ), player = 2)
            ∧ (∀ (game_result : ℕ), game_result = 1 \/ game_result = 0)
            ∧ (∀ (p_win : ℚ), p_win = 1/2)]

-- Define the expected value of games and end state conditions
noncomputable def expected_games : ℕ := 14

-- Theorem statement: The expected number of games given the conditions is 14
theorem expected_games_is_14 (hC : C) : expected_games = 14 :=
by trivial

end expected_games_is_14_l278_278903


namespace count_valid_arrangements_l278_278435

-- Definitions based on conditions
def total_chairs : Nat := 48

def valid_factor_pairs (n : Nat) : List (Nat × Nat) :=
  [ (2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2) ]

def count_valid_arrays : Nat := valid_factor_pairs total_chairs |>.length

-- The theorem we want to prove
theorem count_valid_arrangements : count_valid_arrays = 8 := 
  by
    -- proof should be provided here
    sorry

end count_valid_arrangements_l278_278435


namespace sum_multiple_of_3_probability_l278_278106

noncomputable def probability_sum_multiple_of_3 (faces : List ℕ) (rolls : ℕ) (multiple : ℕ) : ℚ :=
  if rolls = 3 ∧ multiple = 3 ∧ faces = [1, 2, 3, 4, 5, 6] then 1 / 3 else 0

theorem sum_multiple_of_3_probability :
  probability_sum_multiple_of_3 [1, 2, 3, 4, 5, 6] 3 3 = 1 / 3 :=
by
  sorry

end sum_multiple_of_3_probability_l278_278106


namespace tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l278_278234

open Real

theorem tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence (α β γ : ℝ) 
  (h1 : α + β + γ = π)  -- Assuming α, β, γ are angles in a triangle
  (h2 : tan α + tan γ = 2 * tan β) :
  sin (2 * α) + sin (2 * γ) = 2 * sin (2 * β) :=
by
  sorry

end tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l278_278234


namespace de_morgan_neg_or_l278_278061

theorem de_morgan_neg_or (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by sorry

end de_morgan_neg_or_l278_278061


namespace f_decreasing_l278_278895

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / x^2 + 3

theorem f_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h : x1 < x2) : f x1 > f x2 := 
by
  sorry

end f_decreasing_l278_278895


namespace evaluate_expression_l278_278977

theorem evaluate_expression : (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end evaluate_expression_l278_278977


namespace exists_x_y_for_2021_pow_n_l278_278162

theorem exists_x_y_for_2021_pow_n (n : ℕ) :
  (∃ x y : ℤ, 2021 ^ n = x ^ 4 - 4 * y ^ 4) ↔ ∃ m : ℕ, n = 4 * m := 
sorry

end exists_x_y_for_2021_pow_n_l278_278162


namespace four_leaf_area_l278_278075

theorem four_leaf_area (a : ℝ) : 
  let radius := a / 2
  let semicircle_area := (π * radius ^ 2) / 2
  let triangle_area := (a / 2) * (a / 2) / 2
  let half_leaf_area := semicircle_area - triangle_area
  let leaf_area := 2 * half_leaf_area
  let total_area := 4 * leaf_area
  total_area = a ^ 2 / 2 * (π - 2) := 
by
  sorry

end four_leaf_area_l278_278075


namespace cannot_determine_c_l278_278834

-- Definitions based on conditions
variables {a b c d : ℕ}
axiom h1 : a + b + c = 21
axiom h2 : a + b + d = 27
axiom h3 : a + c + d = 30

-- The statement that c cannot be determined exactly
theorem cannot_determine_c : ¬ (∃ c : ℕ, c = c) :=
by sorry

end cannot_determine_c_l278_278834


namespace candidates_appeared_l278_278930

-- Define the number of appeared candidates in state A and state B
variables (X : ℝ)

-- The conditions given in the problem
def condition1 : Prop := (0.07 * X = 0.06 * X + 83)

-- The claim that needs to be proved
def claim : Prop := (X = 8300)

-- The theorem statement in Lean 4
theorem candidates_appeared (X : ℝ) (h1 : condition1 X) : claim X := by
  -- Proof is omitted
  sorry

end candidates_appeared_l278_278930


namespace f_neg_a_l278_278940

-- Definition of the function f
def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

-- Given condition
variable (a : ℝ)
axiom f_a : f a = 11

-- The goal is to prove f(-a) = -9
theorem f_neg_a : f (-a) = -9 := 
sorry

end f_neg_a_l278_278940


namespace taehyung_math_score_l278_278530

theorem taehyung_math_score
  (avg_before : ℝ)
  (drop_in_avg : ℝ)
  (num_subjects_before : ℕ)
  (num_subjects_after : ℕ)
  (avg_after : ℝ)
  (total_before : ℝ)
  (total_after : ℝ)
  (math_score : ℝ) :
  avg_before = 95 →
  drop_in_avg = 3 →
  num_subjects_before = 3 →
  num_subjects_after = 4 →
  avg_after = avg_before - drop_in_avg →
  total_before = avg_before * num_subjects_before →
  total_after = avg_after * num_subjects_after →
  math_score = total_after - total_before →
  math_score = 83 :=
by
  intros
  sorry

end taehyung_math_score_l278_278530


namespace probability_best_play_wins_l278_278757

noncomputable def prob_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) : ℝ :=
  let C := Nat.choose in
  (1 : ℝ) / (C (2 * n) n * C (2 * n) (2 * m)) * ∑ q in Finset.range (2 * m + 1),
  (C n q * C n (2 * m - q)) * 
  ∑ t in Finset.range (min q m),
  (C q t * C (2 * n - q) (n - t))

-- A theorem statement in Lean to ensure proper type checks and conditions 
theorem probability_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) :
  ∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t)) 
  =
  (∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t) )) * 
  (nat.choose (2 * n) n * nat.choose (2 * n) (2 * m)) :=
sorry

end probability_best_play_wins_l278_278757


namespace cube_volume_l278_278042

-- Define the condition: the surface area of the cube is 54
def surface_area_of_cube (x : ℝ) : Prop := 6 * x^2 = 54

-- Define the theorem that states the volume of the cube given the surface area condition
theorem cube_volume : ∃ (x : ℝ), surface_area_of_cube x ∧ x^3 = 27 := by
  sorry

end cube_volume_l278_278042


namespace sum_midpoints_x_sum_midpoints_y_l278_278913

-- Defining the problem conditions
variables (a b c d e f : ℝ)
-- Sum of the x-coordinates of the triangle vertices is 15
def sum_x_coords (a b c : ℝ) : Prop := a + b + c = 15
-- Sum of the y-coordinates of the triangle vertices is 12
def sum_y_coords (d e f : ℝ) : Prop := d + e + f = 12

-- Proving the sum of x-coordinates of midpoints of sides is 15
theorem sum_midpoints_x (h1 : sum_x_coords a b c) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by  
  sorry

-- Proving the sum of y-coordinates of midpoints of sides is 12
theorem sum_midpoints_y (h2 : sum_y_coords d e f) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := 
by  
  sorry

end sum_midpoints_x_sum_midpoints_y_l278_278913


namespace cat_food_percentage_l278_278927

theorem cat_food_percentage (D C : ℝ) (h1 : 7 * D + 4 * C = 8 * D) (h2 : 4 * C = D) : 
  (C / (7 * D + D)) * 100 = 3.125 := by
  sorry

end cat_food_percentage_l278_278927


namespace largest_non_summable_composite_l278_278807

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278807


namespace symmetric_points_x_axis_l278_278986

theorem symmetric_points_x_axis (a b : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (a + 2, -2))
  (hQ : Q = (4, b))
  (hx : (a + 2) = 4)
  (hy : b = 2) :
  (a^b) = 4 := by
sorry

end symmetric_points_x_axis_l278_278986


namespace chuck_distance_l278_278778

theorem chuck_distance
  (total_time : ℝ) (out_speed : ℝ) (return_speed : ℝ) (D : ℝ)
  (h1 : total_time = 3)
  (h2 : out_speed = 16)
  (h3 : return_speed = 24)
  (h4 : D / out_speed + D / return_speed = total_time) :
  D = 28.80 :=
by
  sorry

end chuck_distance_l278_278778


namespace pages_per_day_difference_l278_278562

theorem pages_per_day_difference :
  ∀ (total_pages_Ryan : ℕ) (days : ℕ) (pages_per_book_brother : ℕ) (books_per_day_brother : ℕ),
    total_pages_Ryan = 2100 →
    days = 7 →
    pages_per_book_brother = 200 →
    books_per_day_brother = 1 →
    (total_pages_Ryan / days) - (books_per_day_brother * pages_per_book_brother) = 100 := 
by
  intros total_pages_Ryan days pages_per_book_brother books_per_day_brother
  intros h_total_pages_Ryan h_days h_pages_per_book_brother h_books_per_day_brother
  have h1 : total_pages_Ryan / days = 300 := by sorry
  have h2 : books_per_day_brother * pages_per_book_brother = 200 := by sorry
  rw [h1, h2]
  exact rfl

end pages_per_day_difference_l278_278562


namespace min_AB_plus_five_thirds_BF_l278_278330

theorem min_AB_plus_five_thirds_BF 
  (A : ℝ × ℝ) (onEllipse : ℝ × ℝ → Prop) (F : ℝ × ℝ)
  (B : ℝ × ℝ) (minFunction : ℝ)
  (hf : F = (-3, 0)) (hA : A = (-2,2))
  (hB : onEllipse B) :
  (∀ B', onEllipse B' → (dist A B' + 5/3 * dist B' F) ≥ minFunction) →
  minFunction = (dist A B + 5/3 * dist B F) →
  B = (-(5 * Real.sqrt 3) / 2, 2) := by
  sorry

def onEllipse (B : ℝ × ℝ) : Prop := (B.1^2) / 25 + (B.2^2) / 16 = 1

end min_AB_plus_five_thirds_BF_l278_278330


namespace cos_beta_l278_278006

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.cos α = 3/5)
  (h2 : Real.cos (α + β) = -5/13) : Real.cos β = 33/65 := 
sorry

end cos_beta_l278_278006


namespace expected_tomato_yield_is_correct_l278_278888

noncomputable def expected_tomato_yield : ℝ :=
let step_length := 2.5
let tomato_yield_per_sqft := 3 / 4
let area1_length := 15 * step_length
let area1_width := 15 * step_length
let area1_area := area1_length * area1_width
let area2_length := 15 * step_length
let area2_width := 5 * step_length
let area2_area := area2_length * area2_width
let total_area := area1_area + area2_area
total_area * tomato_yield_per_sqft

theorem expected_tomato_yield_is_correct :
  expected_tomato_yield = 1406.25 :=
by
  sorry

end expected_tomato_yield_is_correct_l278_278888


namespace negation_correct_l278_278537

variable (x : Real)

def original_proposition : Prop :=
  x > 0 → x^2 > 0

def negation_proposition : Prop :=
  x ≤ 0 → x^2 ≤ 0

theorem negation_correct :
  ¬ original_proposition x = negation_proposition x :=
by 
  sorry

end negation_correct_l278_278537


namespace average_marks_l278_278084

theorem average_marks (avg1 avg2 : ℝ) (n1 n2 : ℕ) 
  (h_avg1 : avg1 = 40) 
  (h_avg2 : avg2 = 60) 
  (h_n1 : n1 = 25) 
  (h_n2 : n2 = 30) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 50.91 := 
by
  sorry

end average_marks_l278_278084


namespace largest_non_representable_as_sum_of_composites_l278_278800

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278800


namespace part1_part2_l278_278848

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≤ 0) : a ≥ 1 / Real.exp 1 :=
  sorry

noncomputable def g (x b : ℝ) : ℝ := Real.log x + 1/2 * x^2 - (b + 1) * x

theorem part2 (b : ℝ) (x1 x2 : ℝ) (h1 : b ≥ 3/2) (h2 : x1 < x2) (hx3 : g x1 b - g x2 b ≥ k) : k ≤ 15/8 - 2 * Real.log 2 :=
  sorry

end part1_part2_l278_278848


namespace sum_f_1_2021_l278_278340

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom equation_f : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom interval_f : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f x = Real.log (1 - x) / Real.log 2

theorem sum_f_1_2021 : (List.sum (List.map f (List.range' 1 2021))) = -1 := sorry

end sum_f_1_2021_l278_278340


namespace range_f_contained_in_0_1_l278_278619

theorem range_f_contained_in_0_1 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := 
by {
  sorry
}

end range_f_contained_in_0_1_l278_278619


namespace farmer_brown_leg_wing_count_l278_278315

theorem farmer_brown_leg_wing_count :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let pigeons := 4
  let kangaroos := 2
  
  let chicken_legs := 2
  let chicken_wings := 2
  let sheep_legs := 4
  let grasshopper_legs := 6
  let grasshopper_wings := 2
  let spider_legs := 8
  let pigeon_legs := 2
  let pigeon_wings := 2
  let kangaroo_legs := 2

  (chickens * (chicken_legs + chicken_wings) +
  sheep * sheep_legs +
  grasshoppers * (grasshopper_legs + grasshopper_wings) +
  spiders * spider_legs +
  pigeons * (pigeon_legs + pigeon_wings) +
  kangaroos * kangaroo_legs) = 172 := 
by
  sorry

end farmer_brown_leg_wing_count_l278_278315


namespace wine_problem_l278_278367

theorem wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + (1 / 3) * y = 33) : x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by
  sorry

end wine_problem_l278_278367


namespace largest_not_sum_of_two_composites_l278_278821

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278821


namespace smallest_num_rectangles_to_cover_square_l278_278734

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l278_278734


namespace box_white_balls_count_l278_278564

/--
A box has exactly 100 balls, and each ball is either red, blue, or white.
Given that the box has 12 more blue balls than white balls,
and twice as many red balls as blue balls,
prove that the number of white balls is 16.
-/
theorem box_white_balls_count (W B R : ℕ) 
  (h1 : B = W + 12) 
  (h2 : R = 2 * B) 
  (h3 : W + B + R = 100) : 
  W = 16 := 
sorry

end box_white_balls_count_l278_278564


namespace sum_of_possible_values_l278_278497

theorem sum_of_possible_values (x : ℝ) (h : x^2 - 4 * x + 4 = 0) : x = 2 :=
sorry

end sum_of_possible_values_l278_278497


namespace simplified_expression_is_one_l278_278454

-- Define the specific mathematical expressions
def expr1 := -1 ^ 2023
def expr2 := (-2) ^ 3
def expr3 := (-2) * (-3)

-- Construct the full expression
def full_expr := expr1 - expr2 - expr3

-- State the theorem that this full expression equals 1
theorem simplified_expression_is_one : full_expr = 1 := by
  sorry

end simplified_expression_is_one_l278_278454


namespace joanna_needs_more_hours_to_finish_book_l278_278702
-- Import the necessary library

-- Define the problem conditions and prove the final answer

theorem joanna_needs_more_hours_to_finish_book :
  let total_pages := 248
  let pages_per_hour := 16
  let hours_monday := 3
  let hours_tuesday := 6.5
  let pages_read_monday := hours_monday * pages_per_hour
  let pages_read_tuesday := hours_tuesday * pages_per_hour
  let total_pages_read := pages_read_monday + pages_read_tuesday
  let pages_left := total_pages - total_pages_read
  let hours_needed := pages_left / pages_per_hour
  in hours_needed = 6 :=
by
  sorry

end joanna_needs_more_hours_to_finish_book_l278_278702


namespace initial_bleach_percentage_l278_278579

-- Define variables and constants
def total_volume : ℝ := 100
def drained_volume : ℝ := 3.0612244898
def desired_percentage : ℝ := 0.05

-- Define the initial percentage (unknown)
variable (P : ℝ)

-- Define the statement to be proved
theorem initial_bleach_percentage :
  ( (total_volume - drained_volume) * P + drained_volume * 1 = total_volume * desired_percentage )
  → P = 0.02 :=
  by
    intro h
    -- skipping the proof as per instructions
    sorry

end initial_bleach_percentage_l278_278579


namespace ellipse_equation_l278_278386

theorem ellipse_equation 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hM : M = (2, real.sqrt 2))
  (hN : N = (real.sqrt 6, 1))
  (hE : ∀ p : ℝ × ℝ, p = M ∨ p = N -> (p.1^2 / a^2 + p.2^2 / b^2 = 1)) :
  (a^2 = 8 ∧ b^2 = 4) ∧ 
  (∃ R : ℝ, R^2 = 8 / 3 ∧ R > 0 ∧ (∀ (k m : ℝ), 
    (m^2 > 2 ∧ 3*m^2 >= 8 ∧ R = m / real.sqrt(1 + k^2)) ∧ 
    (∀ A B : ℝ × ℝ, tangent_to_circle_at_origin R k m A ∧ tangent_to_circle_at_origin R k m B ∧ 
      (A ≠ B ∧ (A.1 * B.1 + A.2 * B.2 = 0) -> 
      (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) > 4 * real.sqrt 6 / 3 ∧ 
       sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) <= 2 * real.sqrt 3)))) :=
begin
  sorry
end

def tangent_to_circle_at_origin (R k m : ℝ) (P : ℝ × ℝ) : Prop :=
P.2 = k * P.1 + m ∧ P.1^2 + P.2^2 = R^2

end ellipse_equation_l278_278386


namespace rectangle_ratio_l278_278833

theorem rectangle_ratio (s x y : ℝ) (h1 : 4 * (x * y) + s * s = 9 * s * s) (h2 : s + 2 * y = 3 * s) (h3 : x + y = 3 * s): x / y = 2 :=
by sorry

end rectangle_ratio_l278_278833


namespace solve_diophantine_l278_278239

theorem solve_diophantine : ∃ (x y : ℕ) (t : ℤ), x = 4 - 43 * t ∧ y = 6 - 65 * t ∧ t ≤ 0 ∧ 65 * x - 43 * y = 2 :=
by
  sorry

end solve_diophantine_l278_278239


namespace speedster_convertibles_approx_l278_278569

-- Definitions corresponding to conditions
def total_inventory : ℕ := 120
def num_non_speedsters : ℕ := 40
def num_speedsters : ℕ := 2 * total_inventory / 3
def num_speedster_convertibles : ℕ := 64

-- Theorem statement
theorem speedster_convertibles_approx :
  2 * total_inventory / 3 - num_non_speedsters + num_speedster_convertibles = total_inventory :=
sorry

end speedster_convertibles_approx_l278_278569


namespace find_numbers_l278_278325

theorem find_numbers (p q x : ℝ) (h : (p ≠ 1)) :
  ((p * x) ^ 2 - x ^ 2) / (p * x + x) = q ↔ x = q / (p - 1) ∧ p * x = (p * q) / (p - 1) := 
by
  sorry

end find_numbers_l278_278325


namespace sum_a6_a7_a8_is_32_l278_278220

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l278_278220


namespace factorize_difference_of_squares_l278_278609

theorem factorize_difference_of_squares :
  ∀ x : ℝ, x^2 - 9 = (x + 3) * (x - 3) :=
by 
  intro x
  have h : x^2 - 9 = x^2 - 3^2 := by rw (show 9 = 3^2, by norm_num)
  have hs : (x^2 - 3^2) = (x + 3) * (x - 3) := by exact (mul_self_sub_mul_self_eq x 3)
  exact Eq.trans h hs

end factorize_difference_of_squares_l278_278609


namespace largest_multiple_of_15_who_negation_greater_than_neg_150_l278_278420

theorem largest_multiple_of_15_who_negation_greater_than_neg_150 : 
  ∃ (x : ℤ), x % 15 = 0 ∧ -x > -150 ∧ ∀ (y : ℤ), y % 15 = 0 ∧ -y > -150 → x ≥ y :=
by
  sorry

end largest_multiple_of_15_who_negation_greater_than_neg_150_l278_278420


namespace probability_ephraim_fiona_same_heads_as_keiko_l278_278209

/-- Define a function to calculate the probability that Keiko, Ephraim, and Fiona get the same number of heads. -/
def probability_same_heads : ℚ :=
  let total_outcomes := (2^2) * (2^3) * (2^3)
  let successful_outcomes := 13
  successful_outcomes / total_outcomes

/-- Theorem stating the problem condition and expected probability. -/
theorem probability_ephraim_fiona_same_heads_as_keiko
  (h_keiko : ℕ := 2) -- Keiko tosses two coins
  (h_ephraim : ℕ := 3) -- Ephraim tosses three coins
  (h_fiona : ℕ := 3) -- Fiona tosses three coins
  -- Expected probability that both Ephraim and Fiona get the same number of heads as Keiko
  : probability_same_heads = 13 / 256 :=
sorry

end probability_ephraim_fiona_same_heads_as_keiko_l278_278209


namespace probability_of_committee_correct_l278_278531

noncomputable def probability_of_committee_with_at_least_one_boy_and_one_girl : ℚ :=
  let total_members := 24
  let boys := 14
  let girls := 10
  let committee_size := 5
  let total_ways := Nat.choose total_members committee_size
  let ways_all_boys := Nat.choose boys committee_size
  let ways_all_girls := Nat.choose girls committee_size
  let ways_unwanted := ways_all_boys + ways_all_girls
  let desirable_ways := total_ways - ways_unwanted in
  (desirable_ways : ℚ) / total_ways

theorem probability_of_committee_correct :
  probability_of_committee_with_at_least_one_boy_and_one_girl = 40250 / 42504 := by
  sorry

end probability_of_committee_correct_l278_278531


namespace non_consecutive_heads_probability_l278_278108

-- Define the total number of basic events (n).
def total_events : ℕ := 2^4

-- Define the number of events where heads do not appear consecutively (m).
def non_consecutive_heads_events : ℕ := 1 + (Nat.choose 4 1) + (Nat.choose 3 2)

-- Define the probability of heads not appearing consecutively.
def probability_non_consecutive_heads : ℚ := non_consecutive_heads_events / total_events

-- The theorem we seek to prove
theorem non_consecutive_heads_probability :
  probability_non_consecutive_heads = 1 / 2 :=
by
  sorry

end non_consecutive_heads_probability_l278_278108


namespace product_mb_gt_one_l278_278087

theorem product_mb_gt_one (m b : ℝ) (hm : m = 3 / 4) (hb : b = 2) : m * b = 3 / 2 := by
  sorry

end product_mb_gt_one_l278_278087


namespace segment_length_calc_l278_278652

noncomputable def segment_length_parallel_to_side
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : ℝ :=
  a * (b + c) / (a + b + c)

theorem segment_length_calc
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  segment_length_parallel_to_side a b c a_pos b_pos c_pos = a * (b + c) / (a + b + c) :=
sorry

end segment_length_calc_l278_278652


namespace correct_mark_l278_278949

theorem correct_mark (x : ℝ) (n : ℝ) (avg_increase : ℝ) :
  n = 40 → avg_increase = 1 / 2 → (83 - x) / n = avg_increase → x = 63 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l278_278949


namespace min_rectangles_to_cover_square_exactly_l278_278724

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l278_278724


namespace number_of_valid_sets_l278_278682

open Set

-- Definitions for elements
constant a1 a2 a3 a4 a5: Type
constant U : Set Type := {a1, a2, a3, a4, a5}

-- Define the properties of M
def is_valid_M (M : Set Type) : Prop :=
  M ⊆ U ∧ M ∩ {a1, a2, a3} = {a1, a2}

theorem number_of_valid_sets : 
  {M : Set Type // is_valid_M M}.card = 4 :=
sorry

end number_of_valid_sets_l278_278682


namespace sum_of_products_two_at_a_time_l278_278409

theorem sum_of_products_two_at_a_time
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 222)
  (h2 : a + b + c = 22) :
  a * b + b * c + c * a = 131 := 
sorry

end sum_of_products_two_at_a_time_l278_278409


namespace oxygen_atoms_in_compound_l278_278436

theorem oxygen_atoms_in_compound (K_weight Br_weight O_weight molecular_weight : ℕ) 
    (hK : K_weight = 39) (hBr : Br_weight = 80) (hO : O_weight = 16) (hMW : molecular_weight = 168) 
    (n : ℕ) :
    168 = 39 + 80 + n * 16 → n = 3 :=
by
  intros h
  sorry

end oxygen_atoms_in_compound_l278_278436


namespace tiffany_total_score_l278_278916

-- Definitions based on conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def treasures_second_level : ℕ := 5

-- The statement we want to prove
theorem tiffany_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 48 := by
  sorry

end tiffany_total_score_l278_278916


namespace remainder_sum_mult_3_zero_mod_18_l278_278840

theorem remainder_sum_mult_3_zero_mod_18
  (p q r s : ℕ)
  (hp : p % 18 = 8)
  (hq : q % 18 = 11)
  (hr : r % 18 = 14)
  (hs : s % 18 = 15) :
  3 * (p + q + r + s) % 18 = 0 :=
by
  sorry

end remainder_sum_mult_3_zero_mod_18_l278_278840


namespace seq_bound_gt_pow_two_l278_278841

theorem seq_bound_gt_pow_two (a : Fin 101 → ℕ) 
  (h1 : a 1 > a 0) 
  (h2 : ∀ n : Fin 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2 ^ 99 :=
sorry

end seq_bound_gt_pow_two_l278_278841


namespace factorization_of_x_squared_minus_nine_l278_278610

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end factorization_of_x_squared_minus_nine_l278_278610


namespace percentage_deficit_of_second_side_l278_278050

theorem percentage_deficit_of_second_side
  (L W : Real)
  (h1 : ∃ (L' : Real), L' = 1.16 * L)
  (h2 : ∃ (W' : Real), (L' * W') = 1.102 * (L * W))
  (h3 : ∃ (x : Real), W' = W * (1 - x / 100)) :
  x = 5 := 
  sorry

end percentage_deficit_of_second_side_l278_278050


namespace sum_of_cubes_form_l278_278671

theorem sum_of_cubes_form (a b : ℤ) (x1 y1 x2 y2 : ℤ)
  (h1 : a = x1^2 + 3 * y1^2) (h2 : b = x2^2 + 3 * y2^2) :
  ∃ x y : ℤ, a^3 + b^3 = x^2 + 3 * y^2 := sorry

end sum_of_cubes_form_l278_278671


namespace profit_is_5000_l278_278238

namespace HorseshoeProfit

-- Defining constants and conditions
def initialOutlay : ℝ := 10000
def costPerSet : ℝ := 20
def sellingPricePerSet : ℝ := 50
def numberOfSets : ℝ := 500

-- Calculating the profit
def profit : ℝ :=
  let revenue := numberOfSets * sellingPricePerSet
  let manufacturingCosts := initialOutlay + (costPerSet * numberOfSets)
  revenue - manufacturingCosts

-- The main theorem: the profit is $5,000
theorem profit_is_5000 : profit = 5000 := by
  sorry

end HorseshoeProfit

end profit_is_5000_l278_278238


namespace quadrilateral_area_is_correct_l278_278767

-- Let's define the situation
structure TriangleDivisions where
  T1_area : ℝ
  T2_area : ℝ
  T3_area : ℝ
  Q_area : ℝ

def triangleDivisionExample : TriangleDivisions :=
  { T1_area := 4,
    T2_area := 9,
    T3_area := 9,
    Q_area := 36 }

-- The statement to prove
theorem quadrilateral_area_is_correct (T : TriangleDivisions) (h1 : T.T1_area = 4) 
  (h2 : T.T2_area = 9) (h3 : T.T3_area = 9) : T.Q_area = 36 :=
by
  sorry

end quadrilateral_area_is_correct_l278_278767


namespace correct_statement_is_C_l278_278924

theorem correct_statement_is_C :
  (∃ x : ℚ, ∀ y : ℚ, x < y) = false ∧
  (∃ x : ℚ, x < 0 ∧ ∀ y : ℚ, y < 0 → x < y) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y) ∧
  (∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → x ≤ y) = false :=
sorry

end correct_statement_is_C_l278_278924


namespace certain_number_is_sixteen_l278_278293

theorem certain_number_is_sixteen (x : ℝ) (h : x ^ 5 = 4 ^ 10) : x = 16 :=
by
  sorry

end certain_number_is_sixteen_l278_278293


namespace sum_2019_l278_278178

noncomputable def a : ℕ → ℝ := sorry
def S (n : ℕ) : ℝ := sorry

axiom prop_1 : (a 2 - 1)^3 + (a 2 - 1) = 2019
axiom prop_2 : (a 2018 - 1)^3 + (a 2018 - 1) = -2019
axiom arithmetic_sequence : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom sum_formula : S 2019 = (2019 * (a 1 + a 2019)) / 2

theorem sum_2019 : S 2019 = 2019 :=
by sorry

end sum_2019_l278_278178


namespace compare_fractions_l278_278738

theorem compare_fractions : (31 : ℚ) / 11 > (17 : ℚ) / 14 := 
by
  sorry

end compare_fractions_l278_278738


namespace range_of_a_min_value_ab_range_of_y_l278_278288
-- Import the necessary Lean library 

-- Problem 1
theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) → (-2 ≤ a ∧ a ≤ 1) := 
sorry

-- Problem 2
theorem min_value_ab (a b : ℝ) (h₁ : a + b = 1) : 
  (∀ x, |x - 1| + |x - 3| ≥ a^2 + a) → 
  (min ((1 : ℝ) / (4 * |b|) + |b| / a) = 3 / 4 ∧ (a = 2)) :=
sorry

-- Problem 3
theorem range_of_y (a : ℝ) (y : ℝ) (h₁ : a ∈ Set.Ici (2 : ℝ)) : 
  y = (2 * a) / (a^2 + 1) → 0 < y ∧ y ≤ (4 / 5) :=
sorry

end range_of_a_min_value_ab_range_of_y_l278_278288


namespace factorization_correct_l278_278160

-- Define the expression
def expression (a b : ℝ) : ℝ := 3 * a^2 - 3 * b^2

-- Define the factorized form of the expression
def factorized (a b : ℝ) : ℝ := 3 * (a + b) * (a - b)

-- The main statement we need to prove
theorem factorization_correct (a b : ℝ) : expression a b = factorized a b :=
by 
  sorry -- Proof to be filled in

end factorization_correct_l278_278160


namespace price_per_glass_first_day_l278_278892

theorem price_per_glass_first_day (O W : ℝ) (P1 P2 : ℝ) 
  (h1 : O = W) 
  (h2 : P2 = 0.40)
  (revenue_eq : 2 * O * P1 = 3 * O * P2) 
  : P1 = 0.60 := 
by 
  sorry

end price_per_glass_first_day_l278_278892


namespace total_students_l278_278644

-- Definitions
def is_half_reading (S : ℕ) (half_reading : ℕ) := half_reading = S / 2
def is_third_playing (S : ℕ) (third_playing : ℕ) := third_playing = S / 3
def is_total_students (S half_reading third_playing homework : ℕ) := half_reading + third_playing + homework = S

-- Homework is given to be 4
def homework : ℕ := 4

-- Total number of students
theorem total_students (S : ℕ) (half_reading third_playing : ℕ)
    (h₁ : is_half_reading S half_reading) 
    (h₂ : is_third_playing S third_playing) 
    (h₃ : is_total_students S half_reading third_playing homework) :
    S = 24 := 
sorry

end total_students_l278_278644


namespace max_students_distributed_equally_l278_278693

theorem max_students_distributed_equally (pens pencils : ℕ) (h1 : pens = 3528) (h2 : pencils = 3920) : 
  Nat.gcd pens pencils = 392 := 
by 
  sorry

end max_students_distributed_equally_l278_278693


namespace radio_loss_percentage_l278_278933

theorem radio_loss_percentage (CP SP : ℝ) (h_CP : CP = 2400) (h_SP : SP = 2100) :
  ((CP - SP) / CP) * 100 = 12.5 :=
by
  -- Given cost price
  have h_CP : CP = 2400 := h_CP
  -- Given selling price
  have h_SP : SP = 2100 := h_SP
  sorry

end radio_loss_percentage_l278_278933


namespace height_difference_l278_278393

def empireStateBuildingHeight : ℕ := 443
def petronasTowersHeight : ℕ := 452

theorem height_difference :
  petronasTowersHeight - empireStateBuildingHeight = 9 := 
sorry

end height_difference_l278_278393


namespace rin_craters_difference_l278_278969

theorem rin_craters_difference (d da r : ℕ) (h1 : d = 35) (h2 : da = d - 10) (h3 : r = 75) :
  r - (d + da) = 15 :=
by
  sorry

end rin_craters_difference_l278_278969


namespace last_house_probability_l278_278410

noncomputable def santa_claus_distribution (A : ℕ) (B : ℕ) : Enat :=
if B ≠ A then 1 / 2013 else 0

theorem last_house_probability (A B : ℕ)
  (h : B ≠ A) :
  santa_claus_distribution A B = 1 / 2013 :=
by
  sorry

end last_house_probability_l278_278410


namespace max_abs_value_l278_278385

open Complex Real

theorem max_abs_value (z : ℂ) (h : abs (z - 8) + abs (z + 6 * I) = 10) : abs z ≤ 8 :=
sorry

example : ∃ z : ℂ, abs (z - 8) + abs (z + 6 * I) = 10 ∧ abs z = 8 :=
sorry

end max_abs_value_l278_278385


namespace total_noodles_and_pirates_l278_278370

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end total_noodles_and_pirates_l278_278370


namespace simplify_and_evaluate_evaluate_when_x_is_zero_l278_278077

def expr_1 (x : ℝ) : ℝ := (1 / (1 - x)) + 1
def expr_2 (x : ℝ) : ℝ := (x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)
def simplified_expr (x : ℝ) : ℝ := (x + 1) / (x - 2)

theorem simplify_and_evaluate (x : ℝ) (h1 : -2 < x) (h2 : x < 3) (h3 : x ≠ 1) (h4 : x ≠ -1) (h5 : x ≠ 2) :
  (expr_1 x) / (expr_2 x) = simplified_expr x :=
by
  sorry

theorem evaluate_when_x_is_zero :
  (expr_1 0) / (expr_2 0) = -1 / 2 :=
by
  sorry

end simplify_and_evaluate_evaluate_when_x_is_zero_l278_278077


namespace find_number_l278_278037

theorem find_number (x : ℝ) (h : 0.4 * x = 15) : x = 37.5 := by
  sorry

end find_number_l278_278037


namespace problem1_problem2_l278_278852

open Real

noncomputable def f (x : ℝ) : ℝ := sin (x + (π / 2))

theorem problem1 : f x = cos x := by
  sorry

theorem problem2 (α : ℝ) (h : tan α + 1 / tan α = 5) :
    (sqrt 2 * cos (2 * α - π / 4) - 1) / (1 - tan α) = 2 / 5 := by
  sorry

end problem1_problem2_l278_278852


namespace dog_tail_length_l278_278207

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end dog_tail_length_l278_278207


namespace find_n_l278_278349

theorem find_n (n : ℕ) (h1 : ∃ k : ℕ, 12 - n = k * k) : n = 11 := 
by sorry

end find_n_l278_278349


namespace no_a_b_exist_no_a_b_c_exist_l278_278926

-- Part (a):
theorem no_a_b_exist (a b : ℕ) (h0 : 0 < a) (h1 : 0 < b) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n = k^2) :=
sorry

-- Part (b):
theorem no_a_b_c_exist (a b c : ℕ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n + c = k^2) :=
sorry

end no_a_b_exist_no_a_b_c_exist_l278_278926


namespace exponent_multiplication_l278_278352

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end exponent_multiplication_l278_278352


namespace set_1234_excellent_no_proper_subset_excellent_l278_278552

open Set

namespace StepLength

def excellent_set (D : Set ℤ) : Prop :=
∀ A : Set ℤ, ∃ a d : ℤ, d ∈ D → ({a - d, a, a + d} ⊆ A ∨ {a - d, a, a + d} ⊆ (univ \ A))

noncomputable def S : Set (Set ℤ) := {{1}, {2}, {3}, {4}}

theorem set_1234_excellent : excellent_set {1, 2, 3, 4} := sorry

theorem no_proper_subset_excellent :
  ¬ (excellent_set {1, 3, 4} ∨ excellent_set {1, 2, 3} ∨ excellent_set {1, 2, 4} ∨ excellent_set {2, 3, 4}) := sorry

end StepLength

end set_1234_excellent_no_proper_subset_excellent_l278_278552


namespace david_money_left_l278_278599

noncomputable def david_trip (S H : ℝ) : Prop :=
  S + H = 3200 ∧ H = 0.65 * S

theorem david_money_left : ∃ H, david_trip 1939.39 H ∧ |H - 1260.60| < 0.01 := by
  sorry

end david_money_left_l278_278599


namespace necessary_not_sufficient_l278_278285

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end necessary_not_sufficient_l278_278285


namespace range_of_function_l278_278649

theorem range_of_function (x : ℝ) : x ≠ 2 ↔ ∃ y, y = x / (x - 2) :=
sorry

end range_of_function_l278_278649


namespace fraction_is_perfect_square_l278_278223

theorem fraction_is_perfect_square (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_perfect_square_l278_278223


namespace find_total_children_l278_278519

-- Define conditions as a Lean structure
structure SchoolDistribution where
  B : ℕ     -- Total number of bananas
  C : ℕ     -- Total number of children
  absent : ℕ := 160      -- Number of absent children (constant)
  bananas_per_child : ℕ := 2 -- Bananas per child originally (constant)
  bananas_extra : ℕ := 2      -- Extra bananas given to present children (constant)

-- Define the theorem we want to prove
theorem find_total_children (dist : SchoolDistribution) 
  (h1 : dist.B = 2 * dist.C) 
  (h2 : dist.B = 4 * (dist.C - dist.absent)) :
  dist.C = 320 := by
  sorry

end find_total_children_l278_278519


namespace largest_angle_in_triangle_l278_278539

theorem largest_angle_in_triangle : 
  ∀ (A B C : ℝ), A + B + C = 180 ∧ A + B = 105 ∧ (A = B + 40)
  → (C = 75) :=
by
  sorry

end largest_angle_in_triangle_l278_278539


namespace smallest_number_of_rectangles_l278_278711

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l278_278711


namespace complete_square_solution_l278_278560

theorem complete_square_solution (x : ℝ) :
  (x^2 + 6 * x - 4 = 0) → ((x + 3)^2 = 13) :=
by
  sorry

end complete_square_solution_l278_278560


namespace josh_money_remaining_l278_278376

theorem josh_money_remaining :
  let initial := 50.00
  let shirt := 7.85
  let meal := 15.49
  let magazine := 6.13
  let friends_debt := 3.27
  let cd := 11.75
  initial - shirt - meal - magazine - friends_debt - cd = 5.51 :=
by
  sorry

end josh_money_remaining_l278_278376


namespace infection_never_covers_grid_l278_278893

theorem infection_never_covers_grid (n : ℕ) (H : n > 0) :
  exists (non_infected_cell : ℕ × ℕ), (non_infected_cell.1 < n ∧ non_infected_cell.2 < n) :=
by
  sorry

end infection_never_covers_grid_l278_278893


namespace combinedHeightOfBuildingsIsCorrect_l278_278510

-- Define the heights to the top floor of the buildings (in feet)
def empireStateBuildingHeightFeet : Float := 1250
def willisTowerHeightFeet : Float := 1450
def oneWorldTradeCenterHeightFeet : Float := 1368

-- Define the antenna heights of the buildings (in feet)
def empireStateBuildingAntennaFeet : Float := 204
def willisTowerAntennaFeet : Float := 280
def oneWorldTradeCenterAntennaFeet : Float := 408

-- Define the conversion factor from feet to meters
def feetToMeters : Float := 0.3048

-- Calculate the total heights of the buildings in meters
def empireStateBuildingTotalHeightMeters : Float := (empireStateBuildingHeightFeet + empireStateBuildingAntennaFeet) * feetToMeters
def willisTowerTotalHeightMeters : Float := (willisTowerHeightFeet + willisTowerAntennaFeet) * feetToMeters
def oneWorldTradeCenterTotalHeightMeters : Float := (oneWorldTradeCenterHeightFeet + oneWorldTradeCenterAntennaFeet) * feetToMeters

-- Calculate the combined total height of the three buildings in meters
def combinedTotalHeightMeters : Float :=
  empireStateBuildingTotalHeightMeters + willisTowerTotalHeightMeters + oneWorldTradeCenterTotalHeightMeters

-- The statement to prove
theorem combinedHeightOfBuildingsIsCorrect : combinedTotalHeightMeters = 1511.8164 := by
  sorry

end combinedHeightOfBuildingsIsCorrect_l278_278510


namespace line_not_in_first_quadrant_l278_278011

theorem line_not_in_first_quadrant (m x : ℝ) (h : mx + 3 = 4) (hx : x = 1) : 
  ∀ x y : ℝ, y = (m - 2) * x - 3 → ¬(0 < x ∧ 0 < y) :=
by
  -- The actual proof would go here
  sorry

end line_not_in_first_quadrant_l278_278011


namespace distinct_solutions_abs_eq_l278_278019

theorem distinct_solutions_abs_eq (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 :=
by
  -- The proof is omitted
  sorry

end distinct_solutions_abs_eq_l278_278019


namespace cover_square_with_rectangles_l278_278716

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l278_278716


namespace ratio_of_ages_l278_278407

variable (F S : ℕ)

-- Condition 1: The product of father's age and son's age is 756
def cond1 := F * S = 756

-- Condition 2: The ratio of their ages after 6 years will be 2
def cond2 := (F + 6) / (S + 6) = 2

-- Theorem statement: The current ratio of the father's age to the son's age is 7:3
theorem ratio_of_ages (h1 : cond1 F S) (h2 : cond2 F S) : F / S = 7 / 3 :=
sorry

end ratio_of_ages_l278_278407


namespace sam_friend_points_l278_278676

theorem sam_friend_points (sam_points total_points : ℕ) (h1 : sam_points = 75) (h2 : total_points = 87) :
  total_points - sam_points = 12 :=
by sorry

end sam_friend_points_l278_278676


namespace paula_karl_age_sum_l278_278432

theorem paula_karl_age_sum :
  ∃ (P K : ℕ), (P - 5 = 3 * (K - 5)) ∧ (P + 6 = 2 * (K + 6)) ∧ (P + K = 54) :=
by
  sorry

end paula_karl_age_sum_l278_278432


namespace max_sum_of_arithmetic_sequence_l278_278839

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → 4 * a (n + 1) = 4 * a n - 7) →
  a 1 = 25 →
  (∀ n : ℕ, S n = (n * (50 - (7/4 : ℚ) * (n - 1))) / 2) →
  ∃ n : ℕ, n = 15 ∧ S n = 765 / 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l278_278839


namespace infinite_series_sum_eq_1_div_432_l278_278152

theorem infinite_series_sum_eq_1_div_432 :
  (∑' n : ℕ, (4 * (n + 1) + 1) / ((4 * (n + 1) - 1)^3 * (4 * (n + 1) + 3)^3)) = (1 / 432) :=
  sorry

end infinite_series_sum_eq_1_div_432_l278_278152


namespace largest_non_sum_of_composites_l278_278787

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278787


namespace average_rst_l278_278033

theorem average_rst (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 :=
sorry

end average_rst_l278_278033


namespace largest_cannot_be_sum_of_two_composites_l278_278794

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278794


namespace max_superior_squares_l278_278621

theorem max_superior_squares (n : ℕ) (h : n > 2004) :
  ∃ superior_squares_count : ℕ, superior_squares_count = n * (n - 2004) := 
sorry

end max_superior_squares_l278_278621


namespace length_of_cable_l278_278372

noncomputable def sphere_radius (x y z : ℝ) : ℝ := 8
noncomputable def plane_distance (v : ℝ) : ℝ := v / real.sqrt 3
noncomputable def circle_radius (R d : ℝ) : ℝ := real.sqrt (R ^ 2 - d ^ 2)

theorem length_of_cable :
  (∃ (x y z : ℝ), x + y + z = 10 ∧ x * y + y * z + x * z = 18) →
  (∃ (l : ℝ), l = 4 * real.pi * real.sqrt (23 / 3)) :=
by
  intros h
  obtain ⟨x, y, z, h₁, h₂⟩ := h
  -- Definition of values based on conditions
  let R := sphere_radius x y z
  let d := plane_distance 10
  let r := circle_radius R d
  -- Expected proof that the calculated length is indeed 4 π √(23/3)
  use 4 * real.pi * r
  sorry

end length_of_cable_l278_278372


namespace problem1_problem2_problem3_l278_278774

theorem problem1 : 128 + 52 / 13 = 132 :=
by
  sorry

theorem problem2 : 132 / 11 * 29 - 178 = 170 :=
by
  sorry

theorem problem3 : 45 * (320 / (4 * 5)) = 720 :=
by
  sorry

end problem1_problem2_problem3_l278_278774


namespace mowing_lawn_time_l278_278666

def maryRate := 1 / 3
def tomRate := 1 / 4
def combinedRate := 7 / 12
def timeMaryAlone := 1
def lawnLeft := 1 - (timeMaryAlone * maryRate)

theorem mowing_lawn_time:
  (7 / 12) * (8 / 7) = (2 / 3) :=
by
  sorry

end mowing_lawn_time_l278_278666


namespace find_y_value_l278_278356

theorem find_y_value (k c x y : ℝ) (h1 : c = 3) 
                     (h2 : ∀ x : ℝ, y = k * x + c)
                     (h3 : ∃ k : ℝ, 15 = k * 5 + 3) :
  y = -21 :=
by 
  sorry

end find_y_value_l278_278356


namespace factorization_of_x_squared_minus_nine_l278_278608

theorem factorization_of_x_squared_minus_nine {x : ℝ} : x^2 - 9 = (x + 3) * (x - 3) :=
by
  -- Introduce the hypothesis to assist Lean in understanding the polynomial
  have h : x^2 - 9 = (x^2 - 3^2), 
  rw [pow_two, pow_two],
  exact factorization_of_x_squared_minus_3_squared _,
end

end factorization_of_x_squared_minus_nine_l278_278608


namespace equal_costs_at_45_students_l278_278643

def ticket_cost_option1 (x : ℕ) : ℝ :=
  x * 30 * 0.8

def ticket_cost_option2 (x : ℕ) : ℝ :=
  (x - 5) * 30 * 0.9

theorem equal_costs_at_45_students : ∀ x : ℕ, ticket_cost_option1 x = ticket_cost_option2 x ↔ x = 45 := 
by
  intro x
  sorry

end equal_costs_at_45_students_l278_278643


namespace penumbra_ring_area_l278_278867

theorem penumbra_ring_area (r_umbra r_penumbra : ℝ) (h_ratio : r_umbra / r_penumbra = 2 / 6) (h_umbra : r_umbra = 40) :
  π * (r_penumbra ^ 2 - r_umbra ^ 2) = 12800 * π := by
  sorry

end penumbra_ring_area_l278_278867


namespace product_of_zero_multiples_is_equal_l278_278096

theorem product_of_zero_multiples_is_equal :
  (6000 * 0 = 0) ∧ (6 * 0 = 0) → (6000 * 0 = 6 * 0) :=
by sorry

end product_of_zero_multiples_is_equal_l278_278096


namespace graph_is_pair_of_straight_lines_l278_278155

theorem graph_is_pair_of_straight_lines : ∀ (x y : ℝ), 9 * x^2 - y^2 - 6 * x = 0 → ∃ a b c : ℝ, (y = 3 * x - 2 ∨ y = 2 - 3 * x) :=
by
  intro x y h
  sorry

end graph_is_pair_of_straight_lines_l278_278155


namespace range_of_a_l278_278487

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ (¬ ∃ x : ℝ, x < 0 ∧ |x| = a * x - a) ↔ (a > 1 ∨ a ≤ -1) :=
sorry

end range_of_a_l278_278487


namespace processing_rates_and_total_cost_l278_278571

variables (products total_days total_days_A total_days_B daily_capacity_A daily_capacity_B total_cost_A total_cost_B : ℝ)

noncomputable def A_processing_rate : ℝ := daily_capacity_A
noncomputable def B_processing_rate : ℝ := daily_capacity_B

theorem processing_rates_and_total_cost
  (h1 : products = 1000)
  (h2 : total_days_A = total_days_B + 10)
  (h3 : daily_capacity_B = 1.25 * daily_capacity_A)
  (h4 : total_cost_A = 100 * total_days_A)
  (h5 : total_cost_B = 125 * total_days_B) :
  (daily_capacity_A = 20) ∧ (daily_capacity_B = 25) ∧ (total_cost_A + total_cost_B = 5000) :=
by
  sorry

end processing_rates_and_total_cost_l278_278571


namespace exists_marked_sum_of_three_l278_278364

theorem exists_marked_sum_of_three (s : Finset ℕ) (h₀ : s.card = 22) (h₁ : ∀ x ∈ s, x ≤ 30) :
  ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, ∃ d ∈ s, a = b + c + d :=
by
  sorry

end exists_marked_sum_of_three_l278_278364


namespace correct_sampling_methods_l278_278504

def reporter_A_sampling : String :=
  "systematic sampling"

def reporter_B_sampling : String :=
  "systematic sampling"

theorem correct_sampling_methods (constant_flow : Prop)
  (A_interview_method : ∀ t : ℕ, t % 10 = 0)
  (B_interview_method : ∀ n : ℕ, n % 1000 = 0) :
  reporter_A_sampling = "systematic sampling" ∧ reporter_B_sampling = "systematic sampling" :=
by
  sorry

end correct_sampling_methods_l278_278504


namespace solve_expression_l278_278553

theorem solve_expression : 6 / 3 - 2 - 8 + 2 * 8 = 8 := 
by 
  sorry

end solve_expression_l278_278553


namespace tail_length_l278_278205

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end tail_length_l278_278205


namespace sum_of_squares_of_geometric_progression_l278_278705

theorem sum_of_squares_of_geometric_progression 
  {b_1 q S_1 S_2 : ℝ} 
  (h1 : |q| < 1) 
  (h2 : S_1 = b_1 / (1 - q))
  (h3 : S_2 = b_1 / (1 + q)) : 
  (b_1^2 / (1 - q^2)) = S_1 * S_2 := 
by
  sorry

end sum_of_squares_of_geometric_progression_l278_278705


namespace model_A_selected_count_l278_278945

def production_A := 1200
def production_B := 6000
def production_C := 2000
def total_selected := 46

def total_production := production_A + production_B + production_C

theorem model_A_selected_count :
  (production_A / total_production) * total_selected = 6 := by
  sorry

end model_A_selected_count_l278_278945


namespace parallel_resistance_example_l278_278944

theorem parallel_resistance_example :
  ∀ (R1 R2 : ℕ), R1 = 3 → R2 = 6 → 1 / (R : ℚ) = 1 / (R1 : ℚ) + 1 / (R2 : ℚ) → R = 2 := by
  intros R1 R2 hR1 hR2 h_formula
  -- Formulation of the resistance equations and assumptions
  sorry

end parallel_resistance_example_l278_278944


namespace some_number_value_l278_278357

theorem some_number_value (a : ℕ) (some_number : ℕ) (h_a : a = 105)
  (h_eq : a ^ 3 = some_number * 25 * 35 * 63) : some_number = 7 := by
  sorry

end some_number_value_l278_278357


namespace gamma_donuts_received_l278_278460

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end gamma_donuts_received_l278_278460


namespace second_supply_cost_is_24_l278_278587

-- Definitions based on the given problem conditions
def cost_first_supply : ℕ := 13
def last_year_remaining : ℕ := 6
def this_year_budget : ℕ := 50
def remaining_budget : ℕ := 19

-- Sum of last year's remaining budget and this year's budget
def total_budget : ℕ := last_year_remaining + this_year_budget

-- Total amount spent on school supplies
def total_spent : ℕ := total_budget - remaining_budget

-- Cost of second school supply
def cost_second_supply : ℕ := total_spent - cost_first_supply

-- The theorem to prove
theorem second_supply_cost_is_24 : cost_second_supply = 24 := by
  sorry

end second_supply_cost_is_24_l278_278587


namespace k_value_if_perfect_square_l278_278638

theorem k_value_if_perfect_square (a k : ℝ) (h : ∃ b : ℝ, a^2 + 2*k*a + 1 = (a + b)^2) : k = 1 ∨ k = -1 :=
sorry

end k_value_if_perfect_square_l278_278638


namespace greatest_sum_of_visible_numbers_l278_278614

/-- Definition of a cube with numbered faces -/
structure Cube where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  face4 : ℕ
  face5 : ℕ
  face6 : ℕ

/-- The cubes face numbers -/
def cube_numbers : List ℕ := [1, 2, 4, 8, 16, 32]

/-- Stacked cubes with maximized visible numbers sum -/
def maximize_visible_sum :=
  let cube1 := Cube.mk 1 2 4 8 16 32
  let cube2 := Cube.mk 1 2 4 8 16 32
  let cube3 := Cube.mk 1 2 4 8 16 32
  let cube4 := Cube.mk 1 2 4 8 16 32
  244

theorem greatest_sum_of_visible_numbers : maximize_visible_sum = 244 := 
  by
    sorry -- Proof to be done

end greatest_sum_of_visible_numbers_l278_278614


namespace powderman_distance_when_hears_explosion_l278_278577

noncomputable def powderman_speed_yd_per_s : ℝ := 10
noncomputable def blast_time_s : ℝ := 45
noncomputable def sound_speed_ft_per_s : ℝ := 1080
noncomputable def powderman_speed_ft_per_s : ℝ := 30

noncomputable def distance_powderman (t : ℝ) : ℝ := powderman_speed_ft_per_s * t
noncomputable def distance_sound (t : ℝ) : ℝ := sound_speed_ft_per_s * (t - blast_time_s)

theorem powderman_distance_when_hears_explosion :
  ∃ t, t > blast_time_s ∧ distance_powderman t = distance_sound t ∧ (distance_powderman t) / 3 = 463 :=
sorry

end powderman_distance_when_hears_explosion_l278_278577


namespace isosceles_triangle_perimeter_l278_278480

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

end isosceles_triangle_perimeter_l278_278480


namespace total_profit_correct_l278_278764

-- Defining the given conditions as constants
def beef_total : ℝ := 100
def beef_per_taco : ℝ := 0.25
def selling_price : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Calculate the number of tacos
def num_tacos := beef_total / beef_per_taco

-- Calculate the profit per taco
def profit_per_taco := selling_price - cost_per_taco

-- Calculate the total profit
def total_profit := num_tacos * profit_per_taco

-- Prove the total profit
theorem total_profit_correct : total_profit = 200 :=
by sorry

end total_profit_correct_l278_278764


namespace largest_d_for_range_l278_278601

theorem largest_d_for_range (d : ℝ) : (∃ x : ℝ, x^2 - 6*x + d = 2) ↔ d ≤ 11 := 
by
  sorry

end largest_d_for_range_l278_278601


namespace perpendicular_slope_l278_278320

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end perpendicular_slope_l278_278320


namespace find_first_number_l278_278081

theorem find_first_number 
  (first_number second_number hcf lcm : ℕ) 
  (hCF_condition : hcf = 12) 
  (lCM_condition : lcm = 396) 
  (one_number_condition : first_number = 99) 
  (relation_condition : first_number * second_number = hcf * lcm) : 
  second_number = 48 :=
by
  sorry

end find_first_number_l278_278081


namespace trapezoid_area_correct_l278_278985

noncomputable def calculate_trapezoid_area : ℕ :=
  let parallel_side_1 := 6
  let parallel_side_2 := 12
  let leg := 5
  let radius := 5
  let height := radius
  let area := (1 / 2) * (parallel_side_1 + parallel_side_2) * height
  area

theorem trapezoid_area_correct :
  calculate_trapezoid_area = 45 :=
by {
  sorry
}

end trapezoid_area_correct_l278_278985


namespace range_f_in_0_1_l278_278620

variable (f : ℝ → ℝ)
variable (cond : ∀ x y : ℝ, x > y → (f x) ^ 2 ≤ f y)

theorem range_f_in_0_1 : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
begin
  sorry
end

end range_f_in_0_1_l278_278620


namespace sum_of_ratios_of_squares_l278_278698

theorem sum_of_ratios_of_squares (r : ℚ) (a b c : ℤ) (h1 : r = 45 / 64) 
  (h2 : r = (a * (Real.sqrt b)) / c) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hc : c = 8) : a + b + c = 16 := 
by
  sorry

end sum_of_ratios_of_squares_l278_278698


namespace side_length_of_largest_square_l278_278438

theorem side_length_of_largest_square (S : ℝ) 
  (h1 : 2 * (S / 2)^2 + 2 * (S / 4)^2 = 810) : S = 36 :=
by
  -- proof steps go here
  sorry

end side_length_of_largest_square_l278_278438


namespace union_complement_equals_set_l278_278492

universe u

variable {I A B : Set ℕ}

def universal_set : Set ℕ := {0, 1, 2, 3, 4}
def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {2, 3, 4}
def complement_B : Set ℕ := { x ∈ universal_set | x ∉ set_B }

theorem union_complement_equals_set :
  set_A ∪ complement_B = {0, 1, 2} := by
  sorry

end union_complement_equals_set_l278_278492


namespace ed_more_marbles_than_doug_initially_l278_278466

noncomputable def ed_initial_marbles := 37
noncomputable def doug_marbles := 5

theorem ed_more_marbles_than_doug_initially :
  ed_initial_marbles - doug_marbles = 32 := by
  sorry

end ed_more_marbles_than_doug_initially_l278_278466


namespace saleswoman_commission_l278_278762

theorem saleswoman_commission (x : ℝ) (h1 : ∀ sale : ℝ, sale = 800) (h2 : (x / 100) * 500 + 0.25 * (800 - 500) = 0.21875 * 800) : x = 20 := by
  sorry

end saleswoman_commission_l278_278762


namespace largest_value_of_m_exists_l278_278659

theorem largest_value_of_m_exists (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 30) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) : 
  ∃ m : ℝ, (m = min (a * b) (min (b * c) (c * a))) ∧ (m = 2) := sorry

end largest_value_of_m_exists_l278_278659


namespace largest_number_not_sum_of_two_composites_l278_278827

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278827


namespace max_area_of_rectangle_l278_278174

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * (x + y) = 60) : x * y ≤ 225 :=
by sorry

end max_area_of_rectangle_l278_278174


namespace inradius_of_triangle_l278_278048

theorem inradius_of_triangle (A p r s : ℝ) (h1 : A = 3 * p) (h2 : A = r * s) (h3 : s = p / 2) :
  r = 6 :=
by
  sorry

end inradius_of_triangle_l278_278048


namespace smallest_number_of_coins_l278_278551

theorem smallest_number_of_coins :
  ∃ pennies nickels dimes quarters half_dollars : ℕ,
    pennies + nickels + dimes + quarters + half_dollars = 6 ∧
    (∀ amount : ℕ, amount < 100 →
      ∃ p n d q h : ℕ,
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧ h ≤ half_dollars ∧
        1 * p + 5 * n + 10 * d + 25 * q + 50 * h = amount) :=
sorry

end smallest_number_of_coins_l278_278551


namespace find_lambda_l278_278633

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

noncomputable def f (x λ : ℝ) : ℝ :=
  dot_product (a x) (b x) - 2 * λ * magnitude (a x + b x)

theorem find_lambda
  (h_cos2x : ∀ x, dot_product (a x) (b x) = Real.cos (2 * x))
  (h_magnitude : ∀ x, magnitude (a x + b x) = 2 * Real.cos x)
  (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ λ : ℝ, (∀ y, f y λ ≥ 3/2) ∧ λ = 1/2 := sorry

end find_lambda_l278_278633


namespace max_value_of_function_l278_278488

open Real

theorem max_value_of_function :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), (x + cos x) ≤ π / 2 ∧ (∃ y ∈ Icc (0 : ℝ) (π / 2), y + cos y = π / 2) := by
  sorry

end max_value_of_function_l278_278488


namespace extremum_problem_l278_278632

def f (x a b : ℝ) := x^3 + a*x^2 + b*x + a^2

def f_prime (x a b : ℝ) := 3*x^2 + 2*a*x + b

theorem extremum_problem (a b : ℝ) 
  (cond1 : f_prime 1 a b = 0)
  (cond2 : f 1 a b = 10) :
  (a, b) = (4, -11) := 
sorry

end extremum_problem_l278_278632


namespace determine_radii_l278_278337

-- Definitions based on conditions from a)
variable (S1 S2 S3 S4 : Type) -- Centers of the circles
variable (dist_S2_S4 : ℝ) (dist_S1_S2 : ℝ) (dist_S2_S3 : ℝ) (dist_S3_S4 : ℝ)
variable (r1 r2 r3 r4 : ℝ) -- Radii of circles k1, k2, k3, and k4
variable (rhombus : Prop) -- Quadrilateral S1S2S3S4 is a rhombus

-- Given conditions
axiom C1 : ∀ t : S1, r1 = 5
axiom C2 : dist_S2_S4 = 24
axiom C3 : rhombus

-- Equivalency to be proven
theorem determine_radii : 
  r2 = 12 ∧ r4 = 12 ∧ r1 = 5 ∧ r3 = 5 :=
sorry

end determine_radii_l278_278337


namespace numbers_divisible_by_8_between_200_and_400_l278_278857

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278857


namespace tail_length_l278_278204

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end tail_length_l278_278204


namespace find_ages_l278_278134

-- Define that f is a polynomial with integer coefficients
noncomputable def f : ℤ → ℤ := sorry

-- Given conditions
axiom f_at_7 : f 7 = 77
axiom f_at_b : ∃ b : ℕ, f b = 85
axiom f_at_c : ∃ c : ℕ, f c = 0

-- Define what we need to prove
theorem find_ages : ∃ b c : ℕ, (b - 7 ∣ 8) ∧ (c - b ∣ 85) ∧ (c - 7 ∣ 77) ∧ (b = 9) ∧ (c = 14) :=
sorry

end find_ages_l278_278134


namespace average_of_data_set_l278_278402

theorem average_of_data_set :
  (7 + 5 + (-2) + 5 + 10) / 5 = 5 :=
by sorry

end average_of_data_set_l278_278402


namespace solve_gcd_problem_l278_278909

def gcd_problem : Prop :=
  gcd 1337 382 = 191

theorem solve_gcd_problem : gcd_problem := 
by 
  sorry

end solve_gcd_problem_l278_278909


namespace fractional_eq_has_positive_root_m_value_l278_278637

-- Define the conditions and the proof goal
theorem fractional_eq_has_positive_root_m_value (m x : ℝ) (h1 : x - 2 ≠ 0) (h2 : 2 - x ≠ 0) (h3 : ∃ x > 0, (m / (x - 2)) = ((1 - x) / (2 - x)) - 3) : m = 1 :=
by
  -- Proof goes here
  sorry

end fractional_eq_has_positive_root_m_value_l278_278637


namespace total_songs_bought_l278_278739

def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7

theorem total_songs_bought :
  (country_albums + pop_albums) * songs_per_album = 70 := by
  sorry

end total_songs_bought_l278_278739
