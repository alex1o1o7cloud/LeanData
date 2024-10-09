import Mathlib

namespace value_of_product_of_sums_of_roots_l337_33738

theorem value_of_product_of_sums_of_roots 
    (a b c : ℂ)
    (h1 : a + b + c = 15)
    (h2 : a * b + b * c + c * a = 22)
    (h3 : a * b * c = 8) :
    (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end value_of_product_of_sums_of_roots_l337_33738


namespace equation_is_hyperbola_l337_33739

theorem equation_is_hyperbola : 
  ∀ x y : ℝ, (x^2 - 25*y^2 - 10*x + 50 = 0) → 
  (∃ a b h k : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (x - h)^2 / a^2 - (y - k)^2 / b^2 = -1)) :=
by
  sorry

end equation_is_hyperbola_l337_33739


namespace triangle_side_length_l337_33710

   theorem triangle_side_length
   (A B C D E F : Type)
   (angle_bac angle_edf : Real)
   (AB AC DE DF : Real)
   (h1 : angle_bac = angle_edf)
   (h2 : AB = 5)
   (h3 : AC = 4)
   (h4 : DE = 2.5)
   (area_eq : (1 / 2) * AB * AC * Real.sin angle_bac = (1 / 2) * DE * DF * Real.sin angle_edf):
   DF = 8 :=
   by
   sorry
   
end triangle_side_length_l337_33710


namespace joan_half_dollars_spent_on_wednesday_l337_33751

variable (x : ℝ)
variable (h1 : x * 0.5 + 14 * 0.5 = 9)

theorem joan_half_dollars_spent_on_wednesday :
  x = 4 :=
by
  -- The proof is not required, hence using sorry
  sorry

end joan_half_dollars_spent_on_wednesday_l337_33751


namespace jason_money_determination_l337_33737

theorem jason_money_determination (fred_last_week : ℕ) (fred_earned : ℕ) (fred_now : ℕ) (jason_last_week : ℕ → Prop)
  (h1 : fred_last_week = 23)
  (h2 : fred_earned = 63)
  (h3 : fred_now = 86) :
  ¬ ∃ x, jason_last_week x :=
by
  sorry

end jason_money_determination_l337_33737


namespace correct_articles_l337_33760

-- Define the given conditions
def specific_experience : Prop := true
def countable_noun : Prop := true

-- Problem statement: given the conditions, choose the correct articles to fill in the blanks
theorem correct_articles (h1 : specific_experience) (h2 : countable_noun) : 
  "the; a" = "the; a" :=
by
  sorry

end correct_articles_l337_33760


namespace seq_nth_term_2009_l337_33727

theorem seq_nth_term_2009 (n x : ℤ) (h : 2 * x - 3 = 5 ∧ 5 * x - 11 = 9 ∧ 3 * x + 1 = 13) :
  n = 502 ↔ 2009 = (2 * x - 3) + (n - 1) * ((5 * x - 11) - (2 * x - 3)) :=
sorry

end seq_nth_term_2009_l337_33727


namespace max_length_polyline_l337_33747

-- Definition of the grid and problem
def grid_rows : ℕ := 6
def grid_cols : ℕ := 10

-- The maximum length of a closed, non-self-intersecting polyline
theorem max_length_polyline (rows cols : ℕ) 
  (h_rows : rows = grid_rows) (h_cols : cols = grid_cols) :
  ∃ length : ℕ, length = 76 :=
by {
  sorry
}

end max_length_polyline_l337_33747


namespace remainder_when_divided_by_22_l337_33731

theorem remainder_when_divided_by_22 (n : ℤ) (h : (2 * n) % 11 = 2) : n % 22 = 1 :=
by
  sorry

end remainder_when_divided_by_22_l337_33731


namespace original_student_count_l337_33761

variable (A B C N D : ℕ)
variable (hA : A = 40)
variable (hB : B = 32)
variable (hC : C = 36)
variable (hD : D = N * A)
variable (hNewSum : D + 8 * B = (N + 8) * C)

theorem original_student_count (hA : A = 40) (hB : B = 32) (hC : C = 36) (hD : D = N * A) (hNewSum : D + 8 * B = (N + 8) * C) : 
  N = 8 :=
by
  sorry

end original_student_count_l337_33761


namespace find_x_solution_l337_33777

theorem find_x_solution (x b c : ℝ) (h_eq : x^2 + c^2 = (b - x)^2):
  x = (b^2 - c^2) / (2 * b) :=
sorry

end find_x_solution_l337_33777


namespace count_squares_below_graph_l337_33756

theorem count_squares_below_graph (x y: ℕ) (h : 5 * x + 195 * y = 975) :
  ∃ n : ℕ, n = 388 ∧ 
  ∀ a b : ℕ, 0 ≤ a ∧ a ≤ 195 ∧ 0 ≤ b ∧ b ≤ 5 →
    1 * a + 1 * b < 195 * 5 →
    n = 388 := 
sorry

end count_squares_below_graph_l337_33756


namespace digit_150th_of_17_div_70_is_7_l337_33763

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l337_33763


namespace totalCorrectQuestions_l337_33742

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end totalCorrectQuestions_l337_33742


namespace chord_length_l337_33798

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : 
  ∃ EF : ℝ, EF = 6 :=
by
  sorry

end chord_length_l337_33798


namespace inequality_lemma_l337_33776

-- Define the conditions: x and y are positive numbers and x > y
variables (x y : ℝ)
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y)

-- State the theorem to be proved
theorem inequality_lemma : 2 * x + 1 / (x^2 - 2*x*y + y^2) >= 2 * y + 3 :=
by
  sorry

end inequality_lemma_l337_33776


namespace valid_triples_l337_33765

theorem valid_triples :
  ∀ (a b c : ℕ), 1 ≤ a → 1 ≤ b → 1 ≤ c →
  (∃ k : ℕ, 32 * a + 3 * b + 48 * c = 4 * k * a * b * c) ↔ 
  (a = 1 ∧ b = 20 ∧ c = 1) ∨ (a = 1 ∧ b = 4 ∧ c = 1) ∨ (a = 3 ∧ b = 4 ∧ c = 1) := 
by
  sorry

end valid_triples_l337_33765


namespace fifth_boy_pays_l337_33770

def problem_conditions (a b c d e : ℝ) : Prop :=
  d = 20 ∧
  a = (1 / 3) * (b + c + d + e) ∧
  b = (1 / 4) * (a + c + d + e) ∧
  c = (1 / 5) * (a + b + d + e) ∧
  a + b + c + d + e = 120 

theorem fifth_boy_pays (a b c d e : ℝ) (h : problem_conditions a b c d e) : 
  e = 35 :=
sorry

end fifth_boy_pays_l337_33770


namespace problem_l337_33783

def g (x : ℝ) (d e f : ℝ) := d * x^2 + e * x + f

theorem problem (d e f : ℝ) (h_vertex : ∀ x : ℝ, g d e f (x + 2) = -1 * (x + 2)^2 + 5) :
  d + e + 3 * f = 14 := 
sorry

end problem_l337_33783


namespace triangle_area_l337_33792

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l337_33792


namespace fred_earnings_l337_33704

-- Conditions as definitions
def initial_amount : ℕ := 23
def final_amount : ℕ := 86

-- Theorem to prove
theorem fred_earnings : final_amount - initial_amount = 63 := by
  sorry

end fred_earnings_l337_33704


namespace width_of_roads_l337_33708

-- Definitions for the conditions
def length_of_lawn := 80 
def breadth_of_lawn := 60 
def total_cost := 5200 
def cost_per_sq_m := 4 

-- Derived condition: total area based on cost
def total_area_by_cost := total_cost / cost_per_sq_m 

-- Statement to prove: width of each road w is 65/7
theorem width_of_roads (w : ℚ) : (80 * w) + (60 * w) = total_area_by_cost → w = 65 / 7 :=
by
  sorry

end width_of_roads_l337_33708


namespace range_of_a_l337_33759

theorem range_of_a (x a : ℝ) (h₀ : x < 0) (h₁ : 2^x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l337_33759


namespace hospital_bed_occupancy_l337_33793

theorem hospital_bed_occupancy 
  (x : ℕ)
  (beds_A := x)
  (beds_B := 2 * x)
  (beds_C := 3 * x)
  (occupied_A := (1 / 3) * x)
  (occupied_B := (1 / 2) * (2 * x))
  (occupied_C := (1 / 4) * (3 * x))
  (max_capacity_B := (3 / 4) * (2 * x))
  (max_capacity_C := (5 / 6) * (3 * x)) :
  (4 / 3 * x) / (2 * x) = 2 / 3 ∧ (3 / 4 * x) / (3 * x) = 1 / 4 := 
  sorry

end hospital_bed_occupancy_l337_33793


namespace largest_divisor_of_n_l337_33786

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 12 ∣ n :=
by sorry

end largest_divisor_of_n_l337_33786


namespace triangle_base_length_l337_33784

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 6) (A_eq : A = 13.5) (area_eq : A = (b * h) / 2) : b = 4.5 :=
by
  sorry

end triangle_base_length_l337_33784


namespace no_real_roots_of_quad_eq_l337_33728

theorem no_real_roots_of_quad_eq (k : ℝ) : ¬(k ≠ 0 ∧ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0) :=
by
  sorry

end no_real_roots_of_quad_eq_l337_33728


namespace gasoline_distribution_impossible_l337_33712

theorem gasoline_distribution_impossible
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = 50)
  (h2 : x1 = x2 + 10)
  (h3 : x3 + 26 = x2) : false :=
by {
  sorry
}

end gasoline_distribution_impossible_l337_33712


namespace value_depletion_rate_l337_33703

theorem value_depletion_rate (V_initial V_final : ℝ) (t : ℝ) (r : ℝ) :
  V_initial = 900 → V_final = 729 → t = 2 → V_final = V_initial * (1 - r)^t → r = 0.1 :=
by sorry

end value_depletion_rate_l337_33703


namespace some_zen_not_cen_l337_33782

variable {Zen Ben Cen : Type}
variables (P Q R : Zen → Prop)

theorem some_zen_not_cen (h1 : ∀ x, P x → Q x)
                        (h2 : ∃ x, Q x ∧ ¬ (R x)) :
  ∃ x, P x ∧ ¬ (R x) :=
  sorry

end some_zen_not_cen_l337_33782


namespace temperature_at_midnight_l337_33729

theorem temperature_at_midnight :
  ∀ (morning_temp noon_rise midnight_drop midnight_temp : ℤ),
    morning_temp = -3 →
    noon_rise = 6 →
    midnight_drop = -7 →
    midnight_temp = morning_temp + noon_rise + midnight_drop →
    midnight_temp = -4 :=
by
  intros
  sorry

end temperature_at_midnight_l337_33729


namespace will_initially_bought_seven_boxes_l337_33722

theorem will_initially_bought_seven_boxes :
  let given_away_pieces := 3 * 4
  let total_initial_pieces := given_away_pieces + 16
  let initial_boxes := total_initial_pieces / 4
  initial_boxes = 7 := 
by
  sorry

end will_initially_bought_seven_boxes_l337_33722


namespace union_of_A_and_B_l337_33725

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := sorry

end union_of_A_and_B_l337_33725


namespace common_point_arithmetic_progression_l337_33721

theorem common_point_arithmetic_progression (a b c : ℝ) (h : 2 * b = a + c) :
  ∃ (x y : ℝ), (∀ x, y = a * x^2 + b * x + c) ∧ x = -2 ∧ y = 0 :=
by
  sorry

end common_point_arithmetic_progression_l337_33721


namespace find_x_l337_33757

-- Definitions based on conditions
variables (A B C M O : Type)
variables (OA OB OC OM : vector_space O)
variables (x : ℚ) -- Rational number type for x

-- Condition (1): M lies in the plane ABC
-- Condition (2): OM = x * OA + 1/3 * OB + 1/2 * OC
axiom H : OM = x • OA + (1 / 3 : ℚ) • OB + (1 / 2 : ℚ) • OC

-- The theorem statement
theorem find_x :
  x = 1 / 6 :=
sorry -- Proof is to be provided

end find_x_l337_33757


namespace average_speed_is_75_l337_33772

-- Define the conditions
def speed_first_hour : ℕ := 90
def speed_second_hour : ℕ := 60
def total_time : ℕ := 2

-- Define the average speed and prove it is equal to the given answer
theorem average_speed_is_75 : 
  (speed_first_hour + speed_second_hour) / total_time = 75 := 
by 
  -- We will skip the proof for now
  sorry

end average_speed_is_75_l337_33772


namespace percentage_increase_l337_33750

theorem percentage_increase (original new : ℕ) (h₀ : original = 60) (h₁ : new = 120) :
  ((new - original) / original) * 100 = 100 := by
  sorry

end percentage_increase_l337_33750


namespace figure_100_squares_l337_33705

theorem figure_100_squares :
  ∀ (f : ℕ → ℕ),
    (f 0 = 1) →
    (f 1 = 6) →
    (f 2 = 17) →
    (f 3 = 34) →
    f 100 = 30201 :=
by
  intros f h0 h1 h2 h3
  sorry

end figure_100_squares_l337_33705


namespace chord_of_ellipse_bisected_by_point_l337_33789

theorem chord_of_ellipse_bisected_by_point :
  ∀ (x y : ℝ),
  (∃ (x₁ x₂ y₁ y₂ : ℝ), 
    ( (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2) ∧ 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1)) →
  (x + 2 * y = 8) :=
by
  sorry

end chord_of_ellipse_bisected_by_point_l337_33789


namespace vector_identity_l337_33755

namespace VectorAddition

variable {V : Type*} [AddCommGroup V]

theorem vector_identity
  (AD DC AB BC : V)
  (h1 : AD + DC = AC)
  (h2 : AC - AB = BC) :
  AD + DC - AB = BC :=
by
  sorry

end VectorAddition

end vector_identity_l337_33755


namespace compound_proposition_p_or_q_l337_33764

theorem compound_proposition_p_or_q : 
  (∃ (n : ℝ), ∀ (m : ℝ), m * n = m) ∨ 
  (∀ (n : ℝ), ∃ (m : ℝ), m^2 < n) := 
by
  sorry

end compound_proposition_p_or_q_l337_33764


namespace emma_time_l337_33724

theorem emma_time (E : ℝ) (h1 : 2 * E + E = 60) : E = 20 :=
sorry

end emma_time_l337_33724


namespace leonardo_sleep_fraction_l337_33732

theorem leonardo_sleep_fraction (h : 60 ≠ 0) : (12 / 60 : ℚ) = (1 / 5 : ℚ) :=
by
  sorry

end leonardo_sleep_fraction_l337_33732


namespace battery_difference_l337_33713

def flashlights_batteries := 2
def toys_batteries := 15
def difference := 13

theorem battery_difference : toys_batteries - flashlights_batteries = difference :=
by
  sorry

end battery_difference_l337_33713


namespace range_of_f_l337_33723

noncomputable def f (x : ℝ) := Real.arcsin (x ^ 2 - x)

theorem range_of_f :
  Set.range f = Set.Icc (-Real.arcsin (1/4)) (Real.pi / 2) :=
sorry

end range_of_f_l337_33723


namespace prove_math_problem_l337_33749

noncomputable def math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : Prop :=
  (x + y = 1) ∧ (x^5 + y^5 = 11)

theorem prove_math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : math_problem x y h1 h2 h3 :=
  sorry

end prove_math_problem_l337_33749


namespace men_wages_l337_33779

def men := 5
def women := 5
def boys := 7
def total_wages := 90
def wage_man := 7.5

theorem men_wages (men women boys : ℕ) (total_wages wage_man : ℝ)
  (h1 : 5 = women) (h2 : women = boys) (h3 : 5 * wage_man + 1 * wage_man + 7 * wage_man = total_wages) :
  5 * wage_man = 37.5 :=
  sorry

end men_wages_l337_33779


namespace power_sum_l337_33773

theorem power_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := 
by
  sorry

end power_sum_l337_33773


namespace Betty_will_pay_zero_l337_33726

-- Definitions of the conditions
def Doug_age : ℕ := 40
def Alice_age (D : ℕ) : ℕ := D / 2
def Betty_age (B D A : ℕ) : Prop := B + D + A = 130
def Cost_of_pack_of_nuts (C B : ℕ) : Prop := C = 2 * B
def Decrease_rate : ℕ := 5
def New_cost (C B A : ℕ) : ℕ := max 0 (C - (B - A) * Decrease_rate)
def Total_cost (packs cost_per_pack: ℕ) : ℕ := packs * cost_per_pack

-- The main proposition
theorem Betty_will_pay_zero :
  ∃ B A C, 
    (C = 2 * B) ∧
    (A = Doug_age / 2) ∧
    (B + Doug_age + A = 130) ∧
    (Total_cost 20 (max 0 (C - (B - A) * Decrease_rate)) = 0) :=
by sorry

end Betty_will_pay_zero_l337_33726


namespace twentieth_common_number_l337_33714

theorem twentieth_common_number : 
  (∃ (m n : ℤ), (4 * m - 1) = (3 * n + 2) ∧ 20 * 12 - 1 = 239) := 
by
  sorry

end twentieth_common_number_l337_33714


namespace list_price_of_article_l337_33736

theorem list_price_of_article
  (P : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (final_price : ℝ)
  (h1 : discount1 = 0.10)
  (h2 : discount2 = 0.01999999999999997)
  (h3 : final_price = 61.74) :
  P = 70 :=
by
  sorry

end list_price_of_article_l337_33736


namespace avg_age_of_14_students_l337_33746

theorem avg_age_of_14_students (avg_age_25 : ℕ) (avg_age_10 : ℕ) (age_25th : ℕ) (total_students : ℕ) (remaining_students : ℕ) :
  avg_age_25 = 25 →
  avg_age_10 = 22 →
  age_25th = 13 →
  total_students = 25 →
  remaining_students = 14 →
  ( (total_students * avg_age_25) - (10 * avg_age_10) - age_25th ) / remaining_students = 28 :=
by
  intros
  sorry

end avg_age_of_14_students_l337_33746


namespace uncle_taller_than_james_l337_33715

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end uncle_taller_than_james_l337_33715


namespace opposite_of_neg_eight_l337_33790

theorem opposite_of_neg_eight (y : ℤ) (h : y + (-8) = 0) : y = 8 :=
by {
  -- proof goes here
  sorry
}

end opposite_of_neg_eight_l337_33790


namespace concentric_circles_ratio_l337_33740

theorem concentric_circles_ratio (d1 d2 d3 : ℝ) (h1 : d1 = 2) (h2 : d2 = 4) (h3 : d3 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let r3 := d3 / 2
  let A_red := π * r1 ^ 2
  let A_middle := π * r2 ^ 2
  let A_large := π * r3 ^ 2
  let A_blue := A_middle - A_red
  let A_green := A_large - A_middle
  (A_green / A_blue) = 5 / 3 := 
by
  sorry

end concentric_circles_ratio_l337_33740


namespace difference_of_integers_l337_33767

theorem difference_of_integers :
  ∀ (x y : ℤ), (x = 32) → (y = 5*x + 2) → (y - x = 130) :=
by
  intros x y hx hy
  sorry

end difference_of_integers_l337_33767


namespace volume_of_new_pyramid_l337_33799

theorem volume_of_new_pyramid (l w h : ℝ) (h_vol : (1 / 3) * l * w * h = 80) :
  (1 / 3) * (3 * l) * w * (1.8 * h) = 432 :=
by
  sorry

end volume_of_new_pyramid_l337_33799


namespace avg_waiting_time_is_1_point_2_minutes_l337_33700

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l337_33700


namespace find_x_satisfying_floor_eq_l337_33718

theorem find_x_satisfying_floor_eq (x : ℝ) (hx: ⌊x⌋ * x = 152) : x = 38 / 3 :=
sorry

end find_x_satisfying_floor_eq_l337_33718


namespace total_yield_l337_33796

theorem total_yield (x y z : ℝ)
  (h1 : 0.4 * z + 0.2 * x = 1)
  (h2 : 0.1 * y - 0.1 * z = -0.5)
  (h3 : 0.1 * x + 0.2 * y = 4) :
  x + y + z = 15 :=
sorry

end total_yield_l337_33796


namespace find_normal_monthly_charge_l337_33717

-- Define the conditions
def normal_monthly_charge (x : ℕ) : Prop :=
  let first_month_charge := x / 3
  let fourth_month_charge := x + 15
  let other_months_charge := 4 * x
  (first_month_charge + fourth_month_charge + other_months_charge = 175)

-- The statement to prove
theorem find_normal_monthly_charge : ∃ x : ℕ, normal_monthly_charge x ∧ x = 30 := by
  sorry

end find_normal_monthly_charge_l337_33717


namespace new_volume_l337_33788

variable (l w h : ℝ)

-- Given conditions
def volume := l * w * h = 5000
def surface_area := l * w + l * h + w * h = 975
def sum_of_edges := l + w + h = 60

-- Statement to prove
theorem new_volume (h1 : volume l w h) (h2 : surface_area l w h) (h3 : sum_of_edges l w h) :
  (l + 2) * (w + 2) * (h + 2) = 7198 :=
by
  sorry

end new_volume_l337_33788


namespace arithmetic_sequence_formula_l337_33754

-- Define the sequence and its properties
def is_arithmetic_sequence (a : ℤ) (u : ℕ → ℤ) : Prop :=
  u 0 = a - 1 ∧ u 1 = a + 1 ∧ u 2 = 2 * a + 3 ∧ ∀ n, u (n + 1) - u n = u 1 - u 0

theorem arithmetic_sequence_formula (a : ℤ) :
  ∃ u : ℕ → ℤ, is_arithmetic_sequence a u ∧ (∀ n, u n = 2 * n - 3) :=
by
  sorry

end arithmetic_sequence_formula_l337_33754


namespace tan_five_pi_over_four_l337_33762

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l337_33762


namespace average_payment_correct_l337_33752

-- Definitions based on conditions in the problem
def first_payments_num : ℕ := 20
def first_payment_amount : ℕ := 450

def second_payments_num : ℕ := 30
def increment_after_first : ℕ := 80

def third_payments_num : ℕ := 40
def increment_after_second : ℕ := 65

def fourth_payments_num : ℕ := 50
def increment_after_third : ℕ := 105

def fifth_payments_num : ℕ := 60
def increment_after_fourth : ℕ := 95

def total_payments : ℕ := first_payments_num + second_payments_num + third_payments_num + fourth_payments_num + fifth_payments_num

-- Function to calculate total paid amount
def total_amount_paid : ℕ :=
  (first_payments_num * first_payment_amount) +
  (second_payments_num * (first_payment_amount + increment_after_first)) +
  (third_payments_num * (first_payment_amount + increment_after_first + increment_after_second)) +
  (fourth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third)) +
  (fifth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third + increment_after_fourth))

-- Function to calculate average payment
def average_payment : ℕ := total_amount_paid / total_payments

-- The theorem to be proved
theorem average_payment_correct : average_payment = 657 := by
  sorry

end average_payment_correct_l337_33752


namespace total_length_correct_l337_33711

-- Definitions for the first area's path length and scale.
def first_area_scale : ℕ := 500
def first_area_path_length_inches : ℕ := 6
def first_area_path_length_feet : ℕ := first_area_scale * first_area_path_length_inches

-- Definitions for the second area's path length and scale.
def second_area_scale : ℕ := 1000
def second_area_path_length_inches : ℕ := 3
def second_area_path_length_feet : ℕ := second_area_scale * second_area_path_length_inches

-- Total length represented by both paths in feet.
def total_path_length_feet : ℕ :=
  first_area_path_length_feet + second_area_path_length_feet

-- The Lean theorem proving that the total length is 6000 feet.
theorem total_length_correct : total_path_length_feet = 6000 := by
  sorry

end total_length_correct_l337_33711


namespace solution_set_inequality_l337_33768

theorem solution_set_inequality (x : ℝ) :
  ((x + (1 / 2)) * ((3 / 2) - x) ≥ 0) ↔ (- (1 / 2) ≤ x ∧ x ≤ (3 / 2)) :=
by sorry

end solution_set_inequality_l337_33768


namespace pow_four_inequality_l337_33766

theorem pow_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x * y * (x + y)^2 :=
by
  sorry

end pow_four_inequality_l337_33766


namespace largest_of_a_b_c_d_e_l337_33781

theorem largest_of_a_b_c_d_e (a b c d e : ℝ)
  (h1 : a - 2 = b + 3)
  (h2 : a - 2 = c - 4)
  (h3 : a - 2 = d + 5)
  (h4 : a - 2 = e - 6) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end largest_of_a_b_c_d_e_l337_33781


namespace percentage_deducted_from_list_price_l337_33745

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 65.97
noncomputable def selling_price : ℝ := 65.97
noncomputable def required_profit_percent : ℝ := 25

theorem percentage_deducted_from_list_price :
  let desired_selling_price := cost_price * (1 + required_profit_percent / 100)
  let discount_percentage := 100 * (1 - desired_selling_price / list_price)
  discount_percentage = 10.02 :=
by
  sorry

end percentage_deducted_from_list_price_l337_33745


namespace tiling_vertex_squares_octagons_l337_33791

theorem tiling_vertex_squares_octagons (m n : ℕ) 
  (h1 : 135 * n + 90 * m = 360) : 
  m = 1 ∧ n = 2 :=
by
  sorry

end tiling_vertex_squares_octagons_l337_33791


namespace cosine_identity_l337_33730

theorem cosine_identity (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (π / 2 + α) = -1 / 3 := by
  sorry

end cosine_identity_l337_33730


namespace complement_intersection_is_correct_l337_33780

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2}
noncomputable def B : Set ℕ := {0, 2, 5}
noncomputable def complementA := (U \ A)

theorem complement_intersection_is_correct :
  complementA ∩ B = {0, 5} :=
by
  sorry

end complement_intersection_is_correct_l337_33780


namespace remaining_nails_after_repairs_l337_33787

def fraction_used (perc : ℤ) (total : ℤ) : ℤ :=
  (total * perc) / 100

def after_kitchen (nails : ℤ) : ℤ :=
  nails - fraction_used 35 nails

def after_fence (nails : ℤ) : ℤ :=
  let remaining := after_kitchen nails
  remaining - fraction_used 75 remaining

def after_table (nails : ℤ) : ℤ :=
  let remaining := after_fence nails
  remaining - fraction_used 55 remaining

def after_floorboard (nails : ℤ) : ℤ :=
  let remaining := after_table nails
  remaining - fraction_used 30 remaining

theorem remaining_nails_after_repairs :
  after_floorboard 400 = 21 :=
by
  sorry

end remaining_nails_after_repairs_l337_33787


namespace solve_purchase_price_problem_l337_33753

def purchase_price_problem : Prop :=
  ∃ P : ℝ, (0.10 * P + 12 = 35) ∧ (P = 230)

theorem solve_purchase_price_problem : purchase_price_problem :=
  by
    sorry

end solve_purchase_price_problem_l337_33753


namespace value_of_k_l337_33775

theorem value_of_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = 2 * a * b) : k = 3 * Real.sqrt 2 :=
by
  sorry

end value_of_k_l337_33775


namespace car_fuel_tanks_l337_33701

theorem car_fuel_tanks {x X p : ℝ}
  (h1 : x + X = 70)            -- Condition: total capacity is 70 liters
  (h2 : x * p = 45)            -- Condition: cost to fill small car's tank
  (h3 : X * (p + 0.29) = 68)   -- Condition: cost to fill large car's tank
  : x = 30 ∧ X = 40            -- Conclusion: capacities of the tanks
  :=
by {
  sorry
}

end car_fuel_tanks_l337_33701


namespace blake_initial_money_l337_33706

theorem blake_initial_money (amount_spent_oranges amount_spent_apples amount_spent_mangoes change_received initial_amount : ℕ)
  (h1 : amount_spent_oranges = 40)
  (h2 : amount_spent_apples = 50)
  (h3 : amount_spent_mangoes = 60)
  (h4 : change_received = 150)
  (h5 : initial_amount = (amount_spent_oranges + amount_spent_apples + amount_spent_mangoes) + change_received) :
  initial_amount = 300 :=
by
  sorry

end blake_initial_money_l337_33706


namespace fran_speed_l337_33769

-- Definitions for conditions
def joann_speed : ℝ := 15 -- in miles per hour
def joann_time : ℝ := 4 -- in hours
def fran_time : ℝ := 2 -- in hours
def joann_distance : ℝ := joann_speed * joann_time -- distance Joann traveled

-- Proof Goal Statement
theorem fran_speed (hf: fran_time ≠ 0) : (joann_speed * joann_time) / fran_time = 30 :=
by
  -- Sorry placeholder skips the proof steps
  sorry

end fran_speed_l337_33769


namespace Lizette_average_above_94_l337_33733

noncomputable def Lizette_new_weighted_average
  (score3: ℝ) (avg3: ℝ) (weight3: ℝ) (score_new1 score_new2: ℝ) (weight_new: ℝ) :=
  let total_points3 := avg3 * 3
  let total_weight3 := 3 * weight3
  let total_points := total_points3 + score_new1 + score_new2
  let total_weight := total_weight3 + 2 * weight_new
  total_points / total_weight

theorem Lizette_average_above_94:
  ∀ (score3 avg3 weight3 score_new1 score_new2 weight_new: ℝ),
  score3 = 92 →
  avg3 = 94 →
  weight3 = 0.15 →
  score_new1 > 94 →
  score_new2 > 94 →
  weight_new = 0.20 →
  Lizette_new_weighted_average score3 avg3 weight3 score_new1 score_new2 weight_new > 94 :=
by
  intros score3 avg3 weight3 score_new1 score_new2 weight_new h1 h2 h3 h4 h5 h6
  sorry

end Lizette_average_above_94_l337_33733


namespace average_inside_time_l337_33743

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end average_inside_time_l337_33743


namespace june_vs_christopher_l337_33734

namespace SwordLength

def christopher_length : ℕ := 15
def jameson_length : ℕ := 3 + 2 * christopher_length
def june_length : ℕ := 5 + jameson_length

theorem june_vs_christopher : june_length - christopher_length = 23 := by
  show 5 + (3 + 2 * christopher_length) - christopher_length = 23
  sorry

end SwordLength

end june_vs_christopher_l337_33734


namespace counties_under_50k_perc_l337_33748

def percentage (s: String) : ℝ := match s with
  | "20k_to_49k" => 45
  | "less_than_20k" => 30
  | _ => 0

theorem counties_under_50k_perc : percentage "20k_to_49k" + percentage "less_than_20k" = 75 := by
  sorry

end counties_under_50k_perc_l337_33748


namespace jason_total_hours_l337_33720

variables (hours_after_school hours_total : ℕ)

def earnings_after_school := 4 * hours_after_school
def earnings_saturday := 6 * 8
def total_earnings := earnings_after_school + earnings_saturday

theorem jason_total_hours :
  4 * hours_after_school + earnings_saturday = 88 →
  hours_total = hours_after_school + 8 →
  total_earnings = 88 →
  hours_total = 18 :=
by
  intros h1 h2 h3
  sorry

end jason_total_hours_l337_33720


namespace garden_borders_length_l337_33795

theorem garden_borders_length 
  (a b c d e : ℕ)
  (h1 : 6 * 7 = a^2 + b^2 + c^2 + d^2 + e^2)
  (h2 : a * a + b * b + c * c + d * d + e * e = 42) -- This is analogous to the condition
    
: 15 = (4*a + 4*b + 4*c + 4*d + 4*e - 2*(6 + 7)) / 2 :=
by sorry

end garden_borders_length_l337_33795


namespace rain_is_random_event_l337_33785

def is_random_event (p : ℝ) : Prop := p > 0 ∧ p < 1

theorem rain_is_random_event (p : ℝ) (h : p = 0.75) : is_random_event p :=
by
  -- Here we will provide the necessary proof eventually.
  sorry

end rain_is_random_event_l337_33785


namespace swimming_pool_length_l337_33771

noncomputable def solveSwimmingPoolLength : ℕ :=
  let w_pool := 22
  let w_deck := 3
  let total_area := 728
  let total_width := w_pool + 2 * w_deck
  let L := (total_area / total_width) - 2 * w_deck
  L

theorem swimming_pool_length : solveSwimmingPoolLength = 20 := 
  by
  -- Proof goes here
  sorry

end swimming_pool_length_l337_33771


namespace sum_possible_values_l337_33741

theorem sum_possible_values (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := 
by
  sorry

end sum_possible_values_l337_33741


namespace penguins_remaining_to_get_fish_l337_33707

def total_penguins : Nat := 36
def fed_penguins : Nat := 19

theorem penguins_remaining_to_get_fish : (total_penguins - fed_penguins = 17) :=
by
  sorry

end penguins_remaining_to_get_fish_l337_33707


namespace weighted_average_of_angles_l337_33794

def triangle_inequality (a b c α β γ : ℝ) : Prop :=
  (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0

noncomputable def angle_sum (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

theorem weighted_average_of_angles (a b c α β γ : ℝ)
  (h1 : triangle_inequality a b c α β γ)
  (h2 : angle_sum α β γ) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 :=
by
  sorry

end weighted_average_of_angles_l337_33794


namespace angle_ABC_is_83_l337_33778

-- Define a structure for the quadrilateral ABCD 
structure Quadrilateral (A B C D : Type) :=
  (angle_BAC : ℝ) -- Measure in degrees
  (angle_CAD : ℝ) -- Measure in degrees
  (angle_ACD : ℝ) -- Measure in degrees
  (side_AB : ℝ) -- Lengths of sides
  (side_AD : ℝ)
  (side_AC : ℝ)

-- Define the conditions from the problem
variable {A B C D : Type}
variable (quad : Quadrilateral A B C D)
variable (h1 : quad.angle_BAC = 60)
variable (h2 : quad.angle_CAD = 60)
variable (h3 : quad.angle_ACD = 23)
variable (h4 : quad.side_AB + quad.side_AD = quad.side_AC)

-- State the theorem to be proved
theorem angle_ABC_is_83 : quad.angle_ACD = 23 → quad.angle_CAD = 60 → 
                           quad.angle_BAC = 60 → quad.side_AB + quad.side_AD = quad.side_AC → 
                           ∃ angle_ABC : ℝ, angle_ABC = 83 := by
  sorry

end angle_ABC_is_83_l337_33778


namespace largest_class_is_28_l337_33735

-- definition and conditions
def largest_class_students (x : ℕ) : Prop :=
  let total_students := x + (x - 2) + (x - 4) + (x - 6) + (x - 8)
  total_students = 120

-- statement to prove
theorem largest_class_is_28 : ∃ x : ℕ, largest_class_students x ∧ x = 28 :=
by
  sorry

end largest_class_is_28_l337_33735


namespace data_a_value_l337_33797

theorem data_a_value (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a + b + c = 96) : a = 12 :=
by
  sorry

end data_a_value_l337_33797


namespace wind_velocity_determination_l337_33716

theorem wind_velocity_determination (ρ : ℝ) (P1 P2 : ℝ) (A1 A2 : ℝ) (V1 V2 : ℝ) (k : ℝ) :
  ρ = 1.2 →
  P1 = 0.75 →
  A1 = 2 →
  V1 = 12 →
  P1 = ρ * k * A1 * V1^2 →
  P2 = 20.4 →
  A2 = 10.76 →
  P2 = ρ * k * A2 * V2^2 →
  V2 = 27 := 
by sorry

end wind_velocity_determination_l337_33716


namespace num_friends_bought_robots_l337_33709

def robot_cost : Real := 8.75
def tax_charged : Real := 7.22
def change_left : Real := 11.53
def initial_amount : Real := 80.0
def friends_bought_robots : Nat := 7

theorem num_friends_bought_robots :
  (initial_amount - (change_left + tax_charged)) / robot_cost = friends_bought_robots := sorry

end num_friends_bought_robots_l337_33709


namespace calculate_total_earnings_l337_33774

theorem calculate_total_earnings :
  let num_floors := 10
  let rooms_per_floor := 20
  let hours_per_room := 8
  let earnings_per_hour := 20
  let total_rooms := num_floors * rooms_per_floor
  let total_hours := total_rooms * hours_per_room
  let total_earnings := total_hours * earnings_per_hour
  total_earnings = 32000 := by sorry

end calculate_total_earnings_l337_33774


namespace arithmetic_sequence_sum_l337_33758

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum (h1 : a 2 + a 3 = 2) (h2 : a 4 + a 5 = 6) : a 5 + a 6 = 8 :=
sorry

end arithmetic_sequence_sum_l337_33758


namespace exists_positive_integer_k_l337_33719

theorem exists_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → ¬ Nat.Prime (2^n * k + 1) ∧ 2^n * k + 1 > 1 :=
by
  sorry

end exists_positive_integer_k_l337_33719


namespace find_k_l337_33702

theorem find_k (k : ℕ) : 5 ^ k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end find_k_l337_33702


namespace swimming_speed_in_still_water_l337_33744

theorem swimming_speed_in_still_water (v : ℝ) 
  (h_current_speed : 2 = 2) 
  (h_time_distance : 7 = 7) 
  (h_effective_speed : v - 2 = 14 / 7) : 
  v = 4 :=
sorry

end swimming_speed_in_still_water_l337_33744
