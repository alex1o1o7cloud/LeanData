import Mathlib

namespace die_rolls_multiple_of_5_l10_10908

open Probability

def probability_product_multiple_of_5 : ℚ :=
  1 - (5/6)^8

theorem die_rolls_multiple_of_5 :
  probability_product_multiple_of_5 = 1288991 / 1679616 :=
by
  sorry

end die_rolls_multiple_of_5_l10_10908


namespace sample_size_survey_l10_10541

theorem sample_size_survey (students_selected : ℕ) (h : students_selected = 200) : students_selected = 200 :=
by
  assumption

end sample_size_survey_l10_10541


namespace difference_of_two_numbers_l10_10434

def nat_sum := 22305
def a := ∃ a: ℕ, 5 ∣ a
def is_b (a b: ℕ) := b = a / 10 + 3

theorem difference_of_two_numbers (a b : ℕ) (h : a + b = nat_sum) (h1 : 5 ∣ a) (h2 : is_b a b) : a - b = 14872 :=
by
  sorry

end difference_of_two_numbers_l10_10434


namespace problem_statement_l10_10941

theorem problem_statement (a b : ℝ) (h : a < b) : a - b < 0 :=
sorry

end problem_statement_l10_10941


namespace mark_age_in_5_years_l10_10408

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end mark_age_in_5_years_l10_10408


namespace violet_needs_water_l10_10439

/-- Violet needs 800 ml of water per hour hiked, her dog needs 400 ml of water per hour,
    and they can hike for 4 hours. We need to prove that Violet needs 4.8 liters of water
    for the hike. -/
theorem violet_needs_water (hiking_hours : ℝ)
  (violet_water_per_hour : ℝ)
  (dog_water_per_hour : ℝ)
  (violet_water_needed : ℝ)
  (dog_water_needed : ℝ)
  (total_water_needed_ml : ℝ)
  (total_water_needed_liters : ℝ) :
  hiking_hours = 4 ∧
  violet_water_per_hour = 800 ∧
  dog_water_per_hour = 400 ∧
  violet_water_needed = 3200 ∧
  dog_water_needed = 1600 ∧
  total_water_needed_ml = 4800 ∧
  total_water_needed_liters = 4.8 →
  total_water_needed_liters = 4.8 :=
by sorry

end violet_needs_water_l10_10439


namespace geometric_sequence_problem_l10_10543

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 :=
sorry

end geometric_sequence_problem_l10_10543


namespace new_boarder_ratio_l10_10723

structure School where
  initial_boarders : ℕ
  day_students : ℕ
  boarders_ratio : ℚ

theorem new_boarder_ratio (S : School) (additional_boarders : ℕ) :
  S.initial_boarders = 60 →
  S.boarders_ratio = 2 / 5 →
  additional_boarders = 15 →
  S.day_students = (60 * 5) / 2 →
  (S.initial_boarders + additional_boarders) / S.day_students = 1 / 2 :=
by
  sorry

end new_boarder_ratio_l10_10723


namespace num_zeros_of_g_l10_10646

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x
else -(x^2 - 2 * -x)

noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem num_zeros_of_g : ∃! x : ℝ, g x = 0 := sorry

end num_zeros_of_g_l10_10646


namespace girls_on_debate_team_l10_10141

def number_of_students (groups: ℕ) (group_size: ℕ) : ℕ :=
  groups * group_size

def total_students_debate_team : ℕ :=
  number_of_students 8 9

def number_of_boys : ℕ := 26

def number_of_girls : ℕ :=
  total_students_debate_team - number_of_boys

theorem girls_on_debate_team :
  number_of_girls = 46 :=
by
  sorry

end girls_on_debate_team_l10_10141


namespace gumball_water_wednesday_l10_10088

variable (water_Mon_Thu_Sat : ℕ)
variable (water_Tue_Fri_Sun : ℕ)
variable (water_total : ℕ)
variable (water_Wed : ℕ)

theorem gumball_water_wednesday 
  (h1 : water_Mon_Thu_Sat = 9) 
  (h2 : water_Tue_Fri_Sun = 8) 
  (h3 : water_total = 60) 
  (h4 : 3 * water_Mon_Thu_Sat + 3 * water_Tue_Fri_Sun + water_Wed = water_total) : 
  water_Wed = 9 := 
by 
  sorry

end gumball_water_wednesday_l10_10088


namespace min_value_f_when_a1_l10_10080

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + |x - a|

theorem min_value_f_when_a1 : ∀ x : ℝ, f x 1 ≥ 3/4 :=
by sorry

end min_value_f_when_a1_l10_10080


namespace area_of_figure_M_l10_10962

noncomputable def figure_M_area : Real :=
  sorry

theorem area_of_figure_M :
  figure_M_area = 3 :=
  sorry

end area_of_figure_M_l10_10962


namespace ferry_tourists_total_l10_10164

theorem ferry_tourists_total 
  (n : ℕ)
  (a d : ℕ)
  (sum_arithmetic_series : ℕ → ℕ → ℕ → ℕ)
  (trip_count : n = 5)
  (first_term : a = 85)
  (common_difference : d = 3) :
  sum_arithmetic_series n a d = 455 :=
by
  sorry

end ferry_tourists_total_l10_10164


namespace fraction_ratio_l10_10377

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end fraction_ratio_l10_10377


namespace inequality_solution_l10_10704

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 2 / (3 * x + 4) < 5) ↔ x ∈ set.Ioo (-5/3 : ℝ) (-4/3) ∪ set.Ioi (-4/3) :=
by
  sorry

end inequality_solution_l10_10704


namespace problem_statement_l10_10365

variable {A B C D E F H : Point}
variable {a b c : ℝ}

-- Assume the conditions
variable (h_triangle : Triangle A B C)
variable (h_acute : AcuteTriangle h_triangle)
variable (h_altitudes : AltitudesIntersectAt h_triangle H A D B E C F)
variable (h_sides : Sides h_triangle BC a AC b AB c)

-- Statement to prove
theorem problem_statement : AH * AD + BH * BE + CH * CF = 1/2 * (a^2 + b^2 + c^2) :=
sorry

end problem_statement_l10_10365


namespace birgit_hiking_time_l10_10858

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end birgit_hiking_time_l10_10858


namespace expand_expression_l10_10634

variable {R : Type _} [CommRing R] (x : R)

theorem expand_expression :
  (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 :=
by
  sorry

end expand_expression_l10_10634


namespace price_of_expensive_feed_l10_10024

theorem price_of_expensive_feed
  (total_weight : ℝ)
  (mix_price_per_pound : ℝ)
  (cheaper_feed_weight : ℝ)
  (cheaper_feed_price_per_pound : ℝ)
  (expensive_feed_price_per_pound : ℝ) :
  total_weight = 27 →
  mix_price_per_pound = 0.26 →
  cheaper_feed_weight = 14.2105263158 →
  cheaper_feed_price_per_pound = 0.17 →
  expensive_feed_price_per_pound = 0.36 :=
by
  intros h1 h2 h3 h4
  sorry

end price_of_expensive_feed_l10_10024


namespace maximum_value_of_d_l10_10635

theorem maximum_value_of_d 
  (d e : ℕ) 
  (h1 : 0 ≤ d ∧ d < 10) 
  (h2: 0 ≤ e ∧ e < 10) 
  (h3 : (18 + d + e) % 3 = 0) 
  (h4 : (15 - (d + e)) % 11 = 0) 
  : d ≤ 0 := 
sorry

end maximum_value_of_d_l10_10635


namespace diagonal_BD_l10_10732

variables {A B C D : Point}
variables {AB BC BE : ℝ}
variables {parallelogram : ABCD A B C D}

-- Conditions
def side_AB : AB = 3 := sorry
def side_BC : BC = 5 := sorry
def intersection_BE : BE = 9 := sorry

-- Goal 
theorem diagonal_BD : ∀ (BD : ℝ), BD = 34 / 9 :=
by sorry

end diagonal_BD_l10_10732


namespace worth_of_stuff_l10_10181

theorem worth_of_stuff (x : ℝ)
  (h1 : 1.05 * x - 8 = 34) :
  x = 40 :=
by
  sorry

end worth_of_stuff_l10_10181


namespace stephanie_fewer_forks_l10_10708

noncomputable def fewer_forks := 
  (60 - 44) / 4

theorem stephanie_fewer_forks : fewer_forks = 4 := by
  sorry

end stephanie_fewer_forks_l10_10708


namespace birgit_time_to_travel_8km_l10_10857

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end birgit_time_to_travel_8km_l10_10857


namespace number_of_points_marked_l10_10249

theorem number_of_points_marked (a₁ a₂ b₁ b₂ : ℕ) 
  (h₁ : a₁ * a₂ = 50) (h₂ : b₁ * b₂ = 56) (h₃ : a₁ + a₂ = b₁ + b₂) : 
  (a₁ + a₂ + 1 = 16) :=
sorry

end number_of_points_marked_l10_10249


namespace prove_unattainable_y_l10_10914

noncomputable def unattainable_y : Prop :=
  ∀ (x y : ℝ), x ≠ -4 / 3 → y = (2 - x) / (3 * x + 4) → y ≠ -1 / 3

theorem prove_unattainable_y : unattainable_y :=
by
  intro x y h1 h2
  sorry

end prove_unattainable_y_l10_10914


namespace store_discount_difference_l10_10613

theorem store_discount_difference 
  (p : ℝ) -- original price
  (p1 : ℝ := p * 0.60) -- price after initial discount
  (p2 : ℝ := p1 * 0.90) -- price after additional discount
  (claimed_discount : ℝ := 0.55) -- store's claimed discount
  (true_discount : ℝ := (p - p2) / p) -- calculated true discount
  (difference : ℝ := claimed_discount - true_discount)
  : difference = 0.09 :=
sorry

end store_discount_difference_l10_10613


namespace smallest_multiple_of_7_greater_than_500_l10_10190

theorem smallest_multiple_of_7_greater_than_500 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n > 500 ∧ n = 504 := 
by
  sorry

end smallest_multiple_of_7_greater_than_500_l10_10190


namespace min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l10_10445

-- Statements for minimum questions required for different number of cards 

theorem min_questions_30_cards (cards : Fin 30 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 10 :=
by
  sorry

theorem min_questions_31_cards (cards : Fin 31 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 11 :=
by
  sorry

theorem min_questions_32_cards (cards : Fin 32 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 12 :=
by
  sorry

theorem min_questions_50_cards_circle (cards : Fin 50 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 50 :=
by
  sorry

end min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l10_10445


namespace pages_for_ten_dollars_l10_10544

theorem pages_for_ten_dollars (p c pages_per_cent : ℕ) (dollars cents : ℕ) (h1 : p = 5) (h2 : c = 10) (h3 : pages_per_cent = p / c) (h4 : dollars = 10) (h5 : cents = 100 * dollars) :
  (cents * pages_per_cent) = 500 :=
by
  sorry

end pages_for_ten_dollars_l10_10544


namespace omega_value_l10_10550

noncomputable def f (ω : ℝ) (k : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x - Real.pi / 6) + k

theorem omega_value (ω k : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, f ω k x ≤ f ω k (Real.pi / 3)) → ω = 8 :=
by sorry

end omega_value_l10_10550


namespace calculation_not_minus_one_l10_10033

theorem calculation_not_minus_one :
  (-1 : ℤ) * 1 ≠ 1 ∧
  (-1 : ℤ) / (-1) = 1 ∧
  (-2015 : ℤ) / 2015 ≠ 1 ∧
  (-1 : ℤ)^9 * (-1 : ℤ)^2 ≠ 1 := by 
  sorry

end calculation_not_minus_one_l10_10033


namespace greatest_value_is_B_l10_10531

def x : Int := -6

def A : Int := 2 + x
def B : Int := 2 - x
def C : Int := x - 1
def D : Int := x
def E : Int := x / 2

theorem greatest_value_is_B :
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  sorry

end greatest_value_is_B_l10_10531


namespace price_reduction_equation_l10_10607

variable (x : ℝ)

theorem price_reduction_equation 
    (original_price : ℝ)
    (final_price : ℝ)
    (two_reductions : original_price * (1 - x) ^ 2 = final_price) :
    100 * (1 - x) ^ 2 = 81 :=
by
  sorry

end price_reduction_equation_l10_10607


namespace cupboard_cost_price_l10_10877

theorem cupboard_cost_price
  (C : ℝ)
  (h1 : ∀ (S : ℝ), S = 0.84 * C) -- Vijay sells a cupboard at 84% of the cost price.
  (h2 : ∀ (S_new : ℝ), S_new = 1.16 * C) -- If Vijay got Rs. 1200 more, he would have made a profit of 16%.
  (h3 : ∀ (S_new S : ℝ), S_new - S = 1200) -- The difference between new selling price and original selling price is Rs. 1200.
  : C = 3750 := 
sorry -- Proof is not required.

end cupboard_cost_price_l10_10877


namespace sum_n_k_l10_10973

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l10_10973


namespace bernoulli_inequality_gt_bernoulli_inequality_lt_l10_10558

theorem bernoulli_inequality_gt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : x > 1 ∨ x < 0) : (1 + h)^x > 1 + h * x := sorry

theorem bernoulli_inequality_lt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : 0 < x) (hx3 : x < 1) : (1 + h)^x < 1 + h * x := sorry

end bernoulli_inequality_gt_bernoulli_inequality_lt_l10_10558


namespace arithmetic_sequence_difference_l10_10384

theorem arithmetic_sequence_difference (a b c : ℤ) (d : ℤ)
  (h1 : 9 - 1 = 4 * d)
  (h2 : c - a = 2 * d) :
  c - a = 4 := by sorry

end arithmetic_sequence_difference_l10_10384


namespace inequality_proof_l10_10676

theorem inequality_proof (a b c A α : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (h_sum : a + b + c = A) (hA : A ≤ 1) (hα : α > 0) :
  ( (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ) ≥ 3 * ( (3 / A) - (A / 3) ) ^ α :=
by
  sorry

end inequality_proof_l10_10676


namespace new_volume_increased_dimensions_l10_10322

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l10_10322


namespace radius_of_circle_formed_by_spherical_coords_l10_10264

theorem radius_of_circle_formed_by_spherical_coords :
  (∃ θ : ℝ, radius_of_circle (1, θ, π / 3) = sqrt 3 / 2) :=
sorry

end radius_of_circle_formed_by_spherical_coords_l10_10264


namespace Chicago_White_Sox_loss_l10_10481

theorem Chicago_White_Sox_loss :
  ∃ (L : ℕ), (99 = L + 36) ∧ (L = 63) :=
by
  sorry

end Chicago_White_Sox_loss_l10_10481


namespace division_remainder_l10_10106

theorem division_remainder 
  (R D Q : ℕ) 
  (h1 : D = 3 * Q)
  (h2 : D = 3 * R + 3)
  (h3 : 113 = D * Q + R) : R = 5 :=
sorry

end division_remainder_l10_10106


namespace probability_multiple_of_4_l10_10178

-- Definition of the problem conditions
def random_integer (n : ℕ) := ∀ i, 0 < i ∧ i ≤ n → Prop

def multiple_of_4 (i : ℕ) : Prop := i % 4 = 0

def count_multiples_of_4 (n : ℕ) : ℕ := (finset.range n).filter (λ x, multiple_of_4 x).card

-- Given problem conditions
def ben_choose_random_integer : Prop :=
  ∃ x y : ℕ, random_integer 60 x ∧ random_integer 60 y

-- Required proof statement
theorem probability_multiple_of_4 :
  (count_multiples_of_4 60 = 15) →
  (ben_choose_random_integer) →
  let probability := 1 - (3/4) * (3/4)
  in probability = 7/16 :=
begin
  intros h_multiples h_ben_choose,
  sorry
end

end probability_multiple_of_4_l10_10178


namespace son_age_l10_10900

theorem son_age {x : ℕ} {father son : ℕ} 
  (h1 : father = 4 * son)
  (h2 : (son - 10) + (father - 10) = 60)
  (h3 : son = x)
  : x = 16 := 
sorry

end son_age_l10_10900


namespace rectangle_area_l10_10321

theorem rectangle_area
  (line : ∀ x, 6 = x * x + 4 * x + 3 → x = -2 + Real.sqrt 7 ∨ x = -2 - Real.sqrt 7)
  (shorter_side : ∃ l, l = 2 * Real.sqrt 7 ∧ ∃ s, s = l + 3) :
  ∃ a, a = 28 + 12 * Real.sqrt 7 :=
by
  sorry

end rectangle_area_l10_10321


namespace find_x_l10_10373

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem find_x (x : ℝ) :
  (0 ≤ x ∧ x ≤ Real.pi)
  ∧ (norm_sq (a x) + norm_sq (b x) + 2 * ((a x).1 * (b x).1 + (a x).2 * (b x).2) = 1)
  → (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) :=
by
  intro h
  sorry

end find_x_l10_10373


namespace find_temperature_on_friday_l10_10745

variable (M T W Th F : ℕ)

def problem_conditions : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 46 ∧
  M = 44

theorem find_temperature_on_friday (h : problem_conditions M T W Th F) : F = 36 := by
  sorry

end find_temperature_on_friday_l10_10745


namespace prove_a_star_b_l10_10263

variable (a b : ℤ)
variable (h1 : a + b = 12)
variable (h2 : a * b = 35)

def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem prove_a_star_b : star a b = 12 / 35 :=
by
  sorry

end prove_a_star_b_l10_10263


namespace find_largest_number_l10_10576

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
  sorry

end find_largest_number_l10_10576


namespace third_derivative_l10_10638

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative (x : ℝ) : (iterated_deriv 3 y) x = 4 / (1 + x^2)^2 :=
by
  sorry

end third_derivative_l10_10638


namespace gcf_360_270_lcm_360_270_l10_10738

def prime_factors_360 := [(2, 3), (3, 2), (5, 1)]
def prime_factors_270 := [(2, 1), (3, 3), (5, 1)]

def GCF (a b: ℕ) : ℕ := 2^1 * 3^2 * 5^1
def LCM (a b: ℕ) : ℕ := 2^3 * 3^3 * 5^1

-- Theorem: The GCF of 360 and 270 is 90
theorem gcf_360_270 : GCF 360 270 = 90 := by
  sorry

-- Theorem: The LCM of 360 and 270 is 1080
theorem lcm_360_270 : LCM 360 270 = 1080 := by
  sorry

end gcf_360_270_lcm_360_270_l10_10738


namespace sin_cos_shift_l10_10025

noncomputable def cos_max := (0 : ℝ, 1 : ℝ)
noncomputable def sin_max := (Real.pi / 2, 1 : ℝ)

theorem sin_cos_shift :
  cos_max.1 + Real.pi / 2 = sin_max.1 ∧ cos_max.2 = sin_max.2 :=
by
  sorry

end sin_cos_shift_l10_10025


namespace max_player_salary_l10_10751

theorem max_player_salary (n : ℕ) (min_salary total_salary : ℕ) (player_count : ℕ)
  (h1 : player_count = 25)
  (h2 : min_salary = 15000)
  (h3 : total_salary = 850000)
  (h4 : n = 24 * min_salary)
  : (total_salary - n) = 490000 := 
by
  -- assumptions ensure that n represents the total minimum salaries paid to 24 players
  sorry

end max_player_salary_l10_10751


namespace units_digit_7_pow_6_l10_10452

theorem units_digit_7_pow_6 : (7 ^ 6) % 10 = 9 := by
  sorry

end units_digit_7_pow_6_l10_10452


namespace product_of_sum_and_reciprocal_ge_four_l10_10557

theorem product_of_sum_and_reciprocal_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
sorry

end product_of_sum_and_reciprocal_ge_four_l10_10557


namespace line_segment_no_intersection_l10_10944

theorem line_segment_no_intersection (a : ℝ) :
  (¬ ∃ t : ℝ, (0 ≤ t ∧ t ≤ 1 ∧ (1 - t) * (3 : ℝ) + t * (1 : ℝ) = 2 ∧ (1 - t) * (1 : ℝ) + t * (2 : ℝ) = (2 - (1 - t) * (3 : ℝ)) / a)) ->
  (a < -1 ∨ a > 0.5) :=
by
  sorry

end line_segment_no_intersection_l10_10944


namespace range_of_x_l10_10640

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) :
  -1 ≤ x ∧ x < 5 / 4 :=
sorry

end range_of_x_l10_10640


namespace solve_fraction_identity_l10_10420

theorem solve_fraction_identity (x : ℝ) (hx : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_identity_l10_10420


namespace smallest_nine_consecutive_sum_l10_10018

theorem smallest_nine_consecutive_sum (n : ℕ) (h : (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) = 2007)) : n = 219 :=
sorry

end smallest_nine_consecutive_sum_l10_10018


namespace cos_B_eq_neg_one_sixth_l10_10825

theorem cos_B_eq_neg_one_sixth
  (a b c S : ℝ)
  (h1 : sin A * sin (2 * A) = (1 - cos A) * (1 - cos (2 * A)))
  (h2 : S = sqrt 3 / 12 * (8 * b^2 - 9 * a^2)) :
  cos B = -1/6 :=
by
  -- sorry placeholder
  sorry

end cos_B_eq_neg_one_sixth_l10_10825


namespace topping_cost_l10_10525

noncomputable def cost_of_topping (ic_cost sundae_cost number_of_toppings: ℝ) : ℝ :=
(sundae_cost - ic_cost) / number_of_toppings

theorem topping_cost
  (ic_cost : ℝ)
  (sundae_cost : ℝ)
  (number_of_toppings : ℕ)
  (h_ic_cost : ic_cost = 2)
  (h_sundae_cost : sundae_cost = 7)
  (h_number_of_toppings : number_of_toppings = 10) :
  cost_of_topping ic_cost sundae_cost number_of_toppings = 0.5 :=
  by
  -- Proof will be here
  sorry

end topping_cost_l10_10525


namespace fish_remain_approximately_correct_l10_10868

noncomputable def remaining_fish : ℝ :=
  let west_initial := 1800
  let east_initial := 3200
  let north_initial := 500
  let south_initial := 2300
  let a := 3
  let b := 4
  let c := 2
  let d := 5
  let e := 1
  let f := 3
  let west_caught := (a / b) * west_initial
  let east_caught := (c / d) * east_initial
  let south_caught := (e / f) * south_initial
  let west_left := west_initial - west_caught
  let east_left := east_initial - east_caught
  let south_left := south_initial - south_caught
  let north_left := north_initial
  west_left + east_left + south_left + north_left

theorem fish_remain_approximately_correct :
  abs (remaining_fish - 4403) < 1 := 
  sorry

end fish_remain_approximately_correct_l10_10868


namespace track_length_l10_10850

theorem track_length (h₁ : ∀ (x : ℕ), (exists y₁ y₂ : ℕ, y₁ = 120 ∧ y₂ = 180 ∧ y₁ + y₂ = x ∧ (y₂ - y₁ = 60) ∧ (y₂ = x - 120))) : 
  ∃ x : ℕ, x = 600 := by
  sorry

end track_length_l10_10850


namespace standard_equation_of_ellipse_l10_10945

theorem standard_equation_of_ellipse
  (a b c : ℝ)
  (h_major_minor : 2 * a = 6 * b)
  (h_focal_distance : 2 * c = 8)
  (h_ellipse_relation : a^2 = b^2 + c^2) :
  (∀ x y : ℝ, (x^2 / 18 + y^2 / 2 = 1) ∨ (y^2 / 18 + x^2 / 2 = 1)) :=
by {
  sorry
}

end standard_equation_of_ellipse_l10_10945


namespace simplify_expr_1_l10_10297

theorem simplify_expr_1 (a : ℝ) : (2 * a - 3) ^ 2 + (2 * a + 3) * (2 * a - 3) = 8 * a ^ 2 - 12 * a :=
by
  sorry

end simplify_expr_1_l10_10297


namespace total_worth_of_travelers_checks_l10_10334

theorem total_worth_of_travelers_checks (x y : ℕ) (h1 : x + y = 30) (h2 : 50 * (x - 18) + 100 * y = 900) : 
  50 * x + 100 * y = 1800 := 
by
  sorry

end total_worth_of_travelers_checks_l10_10334


namespace zoes_apartment_number_units_digit_is_1_l10_10191

-- Defining the conditions as the initial problem does
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_digit_two (n : ℕ) : Prop :=
  n / 10 = 2 ∨ n % 10 = 2

def three_out_of_four (n : ℕ) : Prop :=
  (is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬ has_digit_two n) ∨
  (is_square n ∧ is_odd n ∧ ¬ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (is_square n ∧ ¬ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (¬ is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n)

theorem zoes_apartment_number_units_digit_is_1 : ∃ n : ℕ, is_two_digit_number n ∧ three_out_of_four n ∧ n % 10 = 1 :=
by
  sorry

end zoes_apartment_number_units_digit_is_1_l10_10191


namespace committee_count_is_correct_l10_10388

-- Definitions of the problem conditions
def total_people : ℕ := 10
def committee_size : ℕ := 5
def remaining_people := total_people - 1
def members_to_choose := committee_size - 1

-- The combinatorial function for selecting committee members
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def number_of_ways_to_form_committee : ℕ :=
  binomial remaining_people members_to_choose

-- Statement of the problem to prove the number of ways is 126
theorem committee_count_is_correct :
  number_of_ways_to_form_committee = 126 :=
by
  sorry

end committee_count_is_correct_l10_10388


namespace measure_angle_BEC_l10_10756

-- The following definitions capture the conditions in the problem:
def square_points (A B C D : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  ∠ A B C = π / 2 ∧ ∠ B C D = π / 2 ∧ ∠ C D A = π / 2 ∧ ∠ D A B = π / 2

def circle_of_radius_centered_at (radius : ℝ) (center : ℝ × ℝ) (point : ℝ × ℝ) : Prop :=
  dist center point = radius

def line_extends_to_point (A B C : ℝ × ℝ) : Prop :=
  collinear {A, B, C}

-- Use these definitions to state the theorem corresponding to the problem:
theorem measure_angle_BEC
  (A B C D E : ℝ × ℝ)
  (h1 : square_points A B C D)
  (h2 : circle_of_radius_centered_at 8 C B)
  (h3 : line_extends_to_point B C E)
  (h4 : circle_of_radius_centered_at 8 C E) :
  angle B E C = π / 2 :=
by
  sorry

end measure_angle_BEC_l10_10756


namespace factor_x10_minus_1024_l10_10747

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end factor_x10_minus_1024_l10_10747


namespace A_inter_B_is_correct_l10_10601

def set_A : Set ℤ := { x : ℤ | x^2 - x - 2 ≤ 0 }
def set_B : Set ℤ := { x : ℤ | True }

theorem A_inter_B_is_correct : set_A ∩ set_B = { -1, 0, 1, 2 } := by
  sorry

end A_inter_B_is_correct_l10_10601


namespace monotonic_sufficient_not_necessary_maximum_l10_10998

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function f
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)
def has_max_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∃ M, ∀ x, a ≤ x → x ≤ b → f x ≤ M

theorem monotonic_sufficient_not_necessary_maximum : 
  ∀ f : ℝ → ℝ,
  ∀ a b : ℝ,
  a ≤ b →
  monotonic_on f a b → 
  has_max_on f a b :=
sorry  -- Proof is omitted

end monotonic_sufficient_not_necessary_maximum_l10_10998


namespace plates_remove_proof_l10_10815

noncomputable def total_weight_initial (plates: ℤ) (weight_per_plate: ℤ): ℤ :=
  plates * weight_per_plate

noncomputable def weight_limit (pounds: ℤ) (ounces_per_pound: ℤ): ℤ :=
  pounds * ounces_per_pound

noncomputable def plates_to_remove (initial_weight: ℤ) (limit: ℤ) (weight_per_plate: ℤ): ℤ :=
  (initial_weight - limit) / weight_per_plate

theorem plates_remove_proof :
  let pounds := 20
  let ounces_per_pound := 16
  let plates_initial := 38
  let weight_per_plate := 10
  let initial_weight := total_weight_initial plates_initial weight_per_plate
  let limit := weight_limit pounds ounces_per_pound
  plates_to_remove initial_weight limit weight_per_plate = 6 :=
by
  sorry

end plates_remove_proof_l10_10815


namespace quadratic_function_max_value_l10_10863

theorem quadratic_function_max_value (x : ℝ) : 
  let y := -x^2 - 1 in y ≤ -1 :=
begin
  let y := -x^2 - 1,
  use x,
  show y = -1,
  sorry
end

end quadratic_function_max_value_l10_10863


namespace travel_speed_is_four_l10_10951
-- Import the required library

-- Define the conditions
def jacksSpeed (x : ℝ) : ℝ := x^2 - 13 * x - 26
def jillsDistance (x : ℝ) : ℝ := x^2 - 5 * x - 66
def jillsTime (x : ℝ) : ℝ := x + 8

-- Prove the equivalent statement
theorem travel_speed_is_four (x : ℝ) (h : x = 15) :
  jillsDistance x / jillsTime x = 4 ∧ jacksSpeed x = 4 := 
by sorry

end travel_speed_is_four_l10_10951


namespace verify_triangle_operation_l10_10133

def triangle (a b c : ℕ) : ℕ := a^2 + b^2 + c^2

theorem verify_triangle_operation : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end verify_triangle_operation_l10_10133


namespace min_value_of_f_inequality_a_b_l10_10650

theorem min_value_of_f :
  ∃ m : ℝ, m = 4 ∧ (∀ x : ℝ, |x + 3| + |x - 1| ≥ m) :=
sorry

theorem inequality_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1 / a + 4 / b ≥ 9 / 4) :=
sorry

end min_value_of_f_inequality_a_b_l10_10650


namespace sequence_an_correct_l10_10931

theorem sequence_an_correct (S_n : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S_n n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n, n ≥ 2 → a n = 2 * n - 1) :=
by
  -- We assume S_n is defined such that S_n = n^2 + 1
  -- From this, we have to show that:
  -- for n = 1, a_1 = 2,
  -- and for n ≥ 2, a_n = 2n - 1
  sorry

end sequence_an_correct_l10_10931


namespace salary_increase_l10_10545

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 0.65 * S = 0.5 * S + (P / 100) * (0.5 * S)) : P = 30 := 
by
  -- proof goes here
  sorry

end salary_increase_l10_10545


namespace has_two_roots_l10_10320

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l10_10320


namespace TriangleInscribedAngle_l10_10829

theorem TriangleInscribedAngle
  (x : ℝ)
  (arc_PQ : ℝ := x + 100)
  (arc_QR : ℝ := 2 * x + 50)
  (arc_RP : ℝ := 3 * x - 40)
  (angle_sum_eq_360 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_PQR : ℝ, angle_PQR = 70.84 := 
sorry

end TriangleInscribedAngle_l10_10829


namespace part1_part2_l10_10890

-- Part 1
theorem part1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

-- Part 2
theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := sorry

end part1_part2_l10_10890


namespace obtuse_triangle_probability_l10_10492

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l10_10492


namespace equation_solution_l10_10358

theorem equation_solution (x y : ℝ) (h : x^2 + (1 - y)^2 + (x - y)^2 = (1 / 3)) : 
  x = (1 / 3) ∧ y = (2 / 3) := 
  sorry

end equation_solution_l10_10358


namespace keegan_total_school_time_l10_10675

-- Definition of the conditions
def keegan_classes : Nat := 7
def history_and_chemistry_time : ℝ := 1.5
def other_class_time : ℝ := 1.2

-- The theorem stating that given these conditions, Keegan spends 7.5 hours a day in school.
theorem keegan_total_school_time : 
  (history_and_chemistry_time + 5 * other_class_time) = 7.5 := 
by
  sorry

end keegan_total_school_time_l10_10675


namespace mn_value_l10_10660

theorem mn_value (m n : ℤ) (h1 : m + n = 1) (h2 : m - n + 2 = 1) : m * n = 0 := 
by 
  sorry

end mn_value_l10_10660


namespace operation_equivalence_l10_10444

theorem operation_equivalence :
  (∀ (x : ℝ), (x * (4 / 5) / (2 / 7)) = x * (7 / 5)) :=
by
  sorry

end operation_equivalence_l10_10444


namespace arithmetic_seq_sum_l10_10075

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 3 = 9)
  (h3 : a 5 = 5) :
  S 9 / S 5 = 1 :=
by
  sorry

end arithmetic_seq_sum_l10_10075


namespace rate_of_fuel_consumption_l10_10748

-- Define the necessary conditions
def total_fuel : ℝ := 100
def total_hours : ℝ := 175

-- Prove the rate of fuel consumption per hour
theorem rate_of_fuel_consumption : (total_fuel / total_hours) = 100 / 175 := 
by 
  sorry

end rate_of_fuel_consumption_l10_10748


namespace problem1_solve_eq_l10_10251

theorem problem1_solve_eq (x : ℝ) : x * (x - 5) = 3 * x - 15 ↔ (x = 5 ∨ x = 3) := by
  sorry

end problem1_solve_eq_l10_10251


namespace pepperoni_fraction_covered_by_pepperoni_l10_10960

-- Definitions of conditions
def pizza_diameter : ℝ := 18
def num_pepperoni_diameter_fits : ℕ := 9
def total_pepperoni_circles : ℕ := 36

-- Definitions derived from conditions
def pepperoni_diameter : ℝ := pizza_diameter / num_pepperoni_diameter_fits
def pepperoni_radius : ℝ := pepperoni_diameter / 2
def pepperoni_area : ℝ := π * (pepperoni_radius) ^ 2
def total_pepperoni_area : ℝ := total_pepperoni_circles * pepperoni_area
def pizza_radius : ℝ := pizza_diameter / 2
def pizza_area : ℝ := π * (pizza_radius) ^ 2
def fraction_covered : ℚ := total_pepperoni_area / pizza_area

-- Theorem statement
theorem pepperoni_fraction_covered_by_pepperoni :
  fraction_covered = 4 / 9 := by
  -- Proof omitted
  sorry

end pepperoni_fraction_covered_by_pepperoni_l10_10960


namespace parallel_vectors_k_l10_10361

theorem parallel_vectors_k (k : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2 - k, 3)) (h₂ : b = (2, -6)) (h₃ : a.1 * b.2 = a.2 * b.1) : k = 3 :=
sorry

end parallel_vectors_k_l10_10361


namespace depth_of_first_hole_l10_10892

theorem depth_of_first_hole :
  (45 * 8 * (80 * 6 * 40) / (45 * 8) : ℝ) = 53.33 := by
  -- This is where you would provide the proof, but it will be skipped with 'sorry'
  sorry

end depth_of_first_hole_l10_10892


namespace rectangle_area_eq_l10_10905

theorem rectangle_area_eq (d : ℝ) (w : ℝ) (h1 : w = d / (2 * (5 : ℝ) ^ (1/2))) (h2 : 3 * w = (3 * d) / (2 * (5 : ℝ) ^ (1/2))) : 
  (3 * w^2) = (3 / 10) * d^2 := 
by sorry

end rectangle_area_eq_l10_10905


namespace range_of_a_l10_10551

-- Definitions for the conditions
def p (x : ℝ) := x ≤ 2
def q (x : ℝ) (a : ℝ) := x < a + 2

-- Theorem statement
theorem range_of_a (a : ℝ) : (∀ x : ℝ, q x a → p x) → a ≤ 0 := by
  sorry

end range_of_a_l10_10551


namespace age_solution_l10_10446

noncomputable def age_problem : Prop :=
  ∃ (A B x : ℕ),
    A = B + 5 ∧
    A + B = 13 ∧
    3 * (A + x) = 4 * (B + x) ∧
    x = 11

theorem age_solution : age_problem :=
  sorry

end age_solution_l10_10446


namespace square_field_area_l10_10614

theorem square_field_area (s A : ℝ) (h1 : 10 * 4 * s = 9280) (h2 : A = s^2) : A = 53824 :=
by {
  sorry -- The proof goes here
}

end square_field_area_l10_10614


namespace tim_fewer_apples_l10_10991

theorem tim_fewer_apples (martha_apples : ℕ) (harry_apples : ℕ) (tim_apples : ℕ) (H1 : martha_apples = 68) (H2 : harry_apples = 19) (H3 : harry_apples * 2 = tim_apples) : martha_apples - tim_apples = 30 :=
by
  sorry

end tim_fewer_apples_l10_10991


namespace y1_less_than_y2_l10_10213

noncomputable def y1 : ℝ := 2 * (-5) + 1
noncomputable def y2 : ℝ := 2 * 3 + 1

theorem y1_less_than_y2 : y1 < y2 := by
  sorry

end y1_less_than_y2_l10_10213


namespace joan_exam_time_difference_l10_10400

theorem joan_exam_time_difference :
  (let english_questions := 30
       math_questions := 15
       english_time_hours := 1
       math_time_hours := 1.5
       english_time_minutes := english_time_hours * 60
       math_time_minutes := math_time_hours * 60
       time_per_english_question := english_time_minutes / english_questions
       time_per_math_question := math_time_minutes / math_questions
    in time_per_math_question - time_per_english_question = 4) :=
by
  sorry

end joan_exam_time_difference_l10_10400


namespace value_of_ab_l10_10823

theorem value_of_ab (a b : ℤ) (h1 : ∀ x : ℤ, -1 < x ∧ x < 1 → (2 * x < a + 1) ∧ (x > 2 * b + 3)) :
  (a + 1) * (b - 1) = -6 :=
by
  sorry

end value_of_ab_l10_10823


namespace count_zero_expressions_l10_10205

/-- Given four specific vector expressions, prove that exactly two of them evaluate to the zero vector. --/
theorem count_zero_expressions
(AB BC CA MB BO OM AC BD CD OA OC CO : ℝ × ℝ)
(H1 : AB + BC + CA = 0)
(H2 : AB + (MB + BO + OM) ≠ 0)
(H3 : AB - AC + BD - CD = 0)
(H4 : OA + OC + BO + CO ≠ 0) :
  (∃ count, count = 2 ∧
      ((AB + BC + CA = 0) → count = count + 1) ∧
      ((AB + (MB + BO + OM) = 0) → count = count + 1) ∧
      ((AB - AC + BD - CD = 0) → count = count + 1) ∧
      ((OA + OC + BO + CO = 0) → count = count + 1)) :=
sorry

end count_zero_expressions_l10_10205


namespace find_a_b_l10_10267

theorem find_a_b (a b : ℝ) : 
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) → a = 5 ∧ b = -6 :=
sorry

end find_a_b_l10_10267


namespace leftover_yarn_after_square_l10_10876

theorem leftover_yarn_after_square (total_yarn : ℕ) (side_length : ℕ) (left_yarn : ℕ) :
  total_yarn = 35 →
  (4 * side_length ≤ total_yarn ∧ (∀ s : ℕ, s > side_length → 4 * s > total_yarn)) →
  left_yarn = total_yarn - 4 * side_length →
  left_yarn = 3 :=
by
  sorry

end leftover_yarn_after_square_l10_10876


namespace total_cost_is_225_l10_10967

def total_tickets : ℕ := 29
def cost_7_dollar_ticket : ℕ := 7
def cost_9_dollar_ticket : ℕ := 9
def number_of_9_dollar_tickets : ℕ := 11
def number_of_7_dollar_tickets : ℕ := total_tickets - number_of_9_dollar_tickets
def total_cost : ℕ := (number_of_9_dollar_tickets * cost_9_dollar_ticket) + (number_of_7_dollar_tickets * cost_7_dollar_ticket)

theorem total_cost_is_225 : total_cost = 225 := by
  sorry

end total_cost_is_225_l10_10967


namespace product_of_fractions_l10_10179

theorem product_of_fractions :
  (1 / 2) * (2 / 3) * (3 / 4) * (3 / 2) = 3 / 8 := by
  sorry

end product_of_fractions_l10_10179


namespace geometric_sequence_properties_l10_10642

theorem geometric_sequence_properties 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  (∀ n, a n = 2^(n - 1)) ∧ (S 6 = 63) := 
by 
  sorry

end geometric_sequence_properties_l10_10642


namespace find_f5_l10_10429

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f4_value : f 4 = 5

theorem find_f5 : f 5 = 25 / 4 :=
by
  -- Proof goes here
  sorry

end find_f5_l10_10429


namespace fraction_of_income_from_tips_l10_10742

variable (S T I : ℝ)

-- Conditions
def tips_as_fraction_of_salary : Prop := T = (3/4) * S
def total_income : Prop := I = S + T

-- Theorem stating the proof problem
theorem fraction_of_income_from_tips 
  (h1 : tips_as_fraction_of_salary S T)
  (h2 : total_income S T I) : (T / I) = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l10_10742


namespace hyperbola_eccentricity_l10_10651

variables (a b c e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
          (c_eq : c = 4) (b_eq : b = 2 * Real.sqrt 3)
          (hyperbola_eq : c ^ 2 = a ^ 2 + b ^ 2)
          (projection_cond : 2 < (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ∧ (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ≤ 4)

theorem hyperbola_eccentricity : e = c / a := 
by
  sorry

end hyperbola_eccentricity_l10_10651


namespace problem_1_problem_2_l10_10797

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | abs (x - 1) < a}

-- Define the first problem statement: If A ⊂ B, then a > 2.
theorem problem_1 (a : ℝ) : (A ⊂ B a) → (2 < a) := by
  sorry

-- Define the second problem statement: If B ⊂ A, then a ≤ 0 or (0 < a < 2).
theorem problem_2 (a : ℝ) : (B a ⊂ A) → (a ≤ 0 ∨ (0 < a ∧ a < 2)) := by
  sorry

end problem_1_problem_2_l10_10797


namespace problem_inequality_l10_10530

theorem problem_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 1 < a₁) (h₂ : 1 < a₂) (h₃ : 1 < a₃) (h₄ : 1 < a₄) (h₅ : 1 < a₅) :
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) :=
sorry

end problem_inequality_l10_10530


namespace line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l10_10556

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : ℕ := 1 
def B : ℕ := 2
def C : ℕ := 3

-- Proof Problem 1: Prove that there are 1800 ways to line up 5 people out of 7 given A must be included.
theorem line_up_including_A : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 2: Prove that there are 1800 ways to line up 5 people out of 7 given A, B, and C are not all included.
theorem line_up_excluding_all_ABC : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 3: Prove that there are 144 ways to line up 5 people out of 7 given A, B, and C are all included, A and B are adjacent, and C is not adjacent to A or B.
theorem line_up_adjacent_AB_not_adjacent_C : Finset ℕ → ℕ :=
by
  sorry

end line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l10_10556


namespace find_m_l10_10933

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
by
  sorry

end find_m_l10_10933


namespace more_cats_than_dogs_l10_10996

theorem more_cats_than_dogs (total_animals : ℕ) (cats : ℕ) (h1 : total_animals = 60) (h2 : cats = 40) : (cats - (total_animals - cats)) = 20 :=
by 
  sorry

end more_cats_than_dogs_l10_10996


namespace order_of_a_b_c_l10_10362

noncomputable def ln : ℝ → ℝ := Real.log
noncomputable def a : ℝ := ln 3 / 3
noncomputable def b : ℝ := ln 5 / 5
noncomputable def c : ℝ := ln 6 / 6

theorem order_of_a_b_c : a > b ∧ b > c := by
  sorry

end order_of_a_b_c_l10_10362


namespace new_volume_increased_dimensions_l10_10324

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l10_10324


namespace initial_fliers_l10_10741

variable (F : ℕ) -- Initial number of fliers

-- Conditions
axiom morning_send : F - (1 / 5) * F = (4 / 5) * F
axiom afternoon_send : (4 / 5) * F - (1 / 4) * ((4 / 5) * F) = (3 / 5) * F
axiom final_count : (3 / 5) * F = 600

theorem initial_fliers : F = 1000 := by
  sorry

end initial_fliers_l10_10741


namespace arithmetic_sequence_common_difference_l10_10802

theorem arithmetic_sequence_common_difference (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 3 = 4) (h₂ : S 3 = 3)
  (h₃ : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h₄ : ∀ n, a n = a 1 + (n - 1) * d) :
  ∃ d, d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l10_10802


namespace y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l10_10867

def line_equation (m x1 y1 x y : ℝ) : Prop :=
  y - y1 = m * (x - x1)

theorem y_intercept_of_line_with_slope_3_and_x_intercept_7_0 :
  ∃ b : ℝ, line_equation 3 7 0 0 b ∧ b = -21 :=
by
  sorry

end y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l10_10867


namespace incorrect_proposition_l10_10884

-- Variables and conditions
variable (p q : Prop)
variable (m x a b c : ℝ)
variable (hreal : 1 + 4 * m ≥ 0)

-- Theorem statement
theorem incorrect_proposition :
  ¬ (∀ m > 0, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) :=
sorry

end incorrect_proposition_l10_10884


namespace sum_digits_single_digit_l10_10927

theorem sum_digits_single_digit (n : ℕ) (h : n = 2^100) : (n % 9) = 7 := 
sorry

end sum_digits_single_digit_l10_10927


namespace weight_around_59_3_l10_10414

noncomputable def weight_at_height (height: ℝ) : ℝ := 0.75 * height - 68.2

theorem weight_around_59_3 (x : ℝ) (h : x = 170) : abs (weight_at_height x - 59.3) < 1 :=
by
  sorry

end weight_around_59_3_l10_10414


namespace probability_even_sum_cards_l10_10663

theorem probability_even_sum_cards (cards : Finset ℕ) (h : cards = {1, 2, 3, 3, 4}) :
  let outcomes := (cards.card choose 2) in
  let evens := {2, 4} in
  let odds := {1, 3, 3} in
  let favorable := (evens.card choose 2) + (odds.card choose 2) in
  (favorable : ℚ) / outcomes = 2 / 5 :=
by
  sorry

end probability_even_sum_cards_l10_10663


namespace henry_books_donation_l10_10209

theorem henry_books_donation
  (initial_books : ℕ := 99)
  (room_books : ℕ := 21)
  (coffee_table_books : ℕ := 4)
  (cookbook_books : ℕ := 18)
  (boxes : ℕ := 3)
  (picked_up_books : ℕ := 12)
  (final_books : ℕ := 23) :
  (initial_books - final_books + picked_up_books - (room_books + coffee_table_books + cookbook_books)) / boxes = 15 :=
by
  sorry

end henry_books_donation_l10_10209


namespace total_payment_360_l10_10039

noncomputable def q : ℝ := 12
noncomputable def p_wage : ℝ := 1.5 * q
noncomputable def p_hourly_rate : ℝ := q + 6
noncomputable def h : ℝ := 20
noncomputable def total_payment_p : ℝ := p_wage * h -- The total payment when candidate p is hired
noncomputable def total_payment_q : ℝ := q * (h + 10) -- The total payment when candidate q is hired

theorem total_payment_360 : 
  p_wage = p_hourly_rate ∧ 
  total_payment_p = total_payment_q ∧ 
  total_payment_p = 360 := by
  sorry

end total_payment_360_l10_10039


namespace find_multiple_of_numerator_l10_10256

theorem find_multiple_of_numerator
  (n d k : ℕ)
  (h1 : d = k * n - 1)
  (h2 : (n + 1) / (d + 1) = 3 / 5)
  (h3 : (n : ℚ) / d = 5 / 9) : k = 2 :=
sorry

end find_multiple_of_numerator_l10_10256


namespace girl_speed_l10_10304

theorem girl_speed (distance time : ℝ) (h_distance : distance = 96) (h_time : time = 16) : distance / time = 6 :=
by
  sorry

end girl_speed_l10_10304


namespace addition_subtraction_questions_l10_10730

theorem addition_subtraction_questions (total_questions word_problems answered_questions add_sub_questions : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : answered_questions = total_questions - 7)
  (h4 : add_sub_questions = answered_questions - word_problems) : 
  add_sub_questions = 21 := 
by 
  -- the proof steps are skipped
  sorry

end addition_subtraction_questions_l10_10730


namespace nancy_more_money_l10_10123

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end nancy_more_money_l10_10123


namespace temperature_at_night_l10_10216

theorem temperature_at_night 
  (T_morning : ℝ) 
  (T_rise_noon : ℝ) 
  (T_drop_night : ℝ) 
  (h1 : T_morning = 22) 
  (h2 : T_rise_noon = 6) 
  (h3 : T_drop_night = 10) : 
  (T_morning + T_rise_noon - T_drop_night = 18) :=
by 
  sorry

end temperature_at_night_l10_10216


namespace units_digit_n_l10_10923

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31^8) (h2 : m % 10 = 7) : n % 10 = 3 := 
sorry

end units_digit_n_l10_10923


namespace chord_length_l10_10898

theorem chord_length (r d: ℝ) (h1: r = 5) (h2: d = 4) : ∃ EF, EF = 6 := by
  sorry

end chord_length_l10_10898


namespace range_of_a_l10_10207

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, x^2 - a * x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
by sorry

end range_of_a_l10_10207


namespace charlyn_viewable_area_l10_10475

noncomputable def charlyn_sees_area (side_length viewing_distance : ℝ) : ℝ :=
  let inner_viewable_area := (side_length^2 - (side_length - 2 * viewing_distance)^2)
  let rectangular_area := 4 * (side_length * viewing_distance)
  let circular_corner_area := 4 * ((viewing_distance^2 * Real.pi) / 4)
  inner_viewable_area + rectangular_area + circular_corner_area

theorem charlyn_viewable_area :
  let side_length := 7
  let viewing_distance := 1.5
  charlyn_sees_area side_length viewing_distance = 82 := 
by
  sorry

end charlyn_viewable_area_l10_10475


namespace find_a_l10_10678

noncomputable def f (x a : ℝ) := 2 * real.log x + a / x^2

theorem find_a (a : ℝ) (h1 : 0 < a) 
  (h2 : ∀ x : ℝ, 0 < x → 2 * real.log x + a / x^2 ≥ 2) :
  a ≥ real.exp 1 :=
sorry

end find_a_l10_10678


namespace find_m_interval_l10_10184

def seq (x : ℕ → ℚ) : Prop :=
  (x 0 = 7) ∧ (∀ n : ℕ, x (n + 1) = (x n ^ 2 + 8 * x n + 9) / (x n + 7))

def m_spec (x : ℕ → ℚ) (m : ℕ) : Prop :=
  (x m ≤ 5 + 1 / 2^15)

theorem find_m_interval :
  ∃ (x : ℕ → ℚ) (m : ℕ), seq x ∧ m_spec x m ∧ 81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l10_10184


namespace solve_inequality_l10_10348

theorem solve_inequality : {x : ℝ | 3 * x ^ 2 - 7 * x - 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
sorry

end solve_inequality_l10_10348


namespace number_of_smaller_cubes_l10_10163

theorem number_of_smaller_cubes (N : ℕ) : 
  (∀ a : ℕ, ∃ n : ℕ, n * a^3 = 125) ∧
  (∀ b : ℕ, b ≤ 5 → ∃ m : ℕ, m * b^3 ≤ 125) ∧
  (∃ x y : ℕ, x ≠ y) → 
  N = 118 :=
sorry

end number_of_smaller_cubes_l10_10163


namespace eq_curveE_eq_lineCD_l10_10803

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def curveE (x y : ℝ) : Prop :=
  distance (x, y) (-1, 0) = Real.sqrt 3 * distance (x, y) (1, 0)

theorem eq_curveE (x y : ℝ) : curveE x y ↔ (x - 2)^2 + y^2 = 3 :=
by sorry

variables (m : ℝ)
variables (m_nonzero : m ≠ 0)
variables (A C B D : ℝ × ℝ)
variables (line1_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = A ∨ p = C) → p.1 - m * p.2 - 1 = 0)
variables (line2_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = B ∨ p = D) → m * p.1 + p.2 - m = 0)
variables (CD_slope : (D.2 - C.2) / (D.1 - C.1) = -1)

theorem eq_lineCD (x y : ℝ) : 
  (y = -x ∨ y = -x + 3) :=
by sorry

end eq_curveE_eq_lineCD_l10_10803


namespace intersection_of_M_and_N_l10_10371

def M := {x : ℝ | 3 * x - x^2 > 0}
def N := {x : ℝ | x^2 - 4 * x + 3 > 0}
def I := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = I :=
by
  sorry

end intersection_of_M_and_N_l10_10371


namespace sandwiches_provided_l10_10022

theorem sandwiches_provided (original_count sold_out : ℕ) (h1 : original_count = 9) (h2 : sold_out = 5) : (original_count - sold_out = 4) :=
by
  sorry

end sandwiches_provided_l10_10022


namespace train_length_72kmphr_9sec_180m_l10_10447

/-- Given speed in km/hr and time in seconds, calculate the length of the train in meters -/
theorem train_length_72kmphr_9sec_180m : ∀ (speed_kmph : ℕ) (time_sec : ℕ),
  speed_kmph = 72 → time_sec = 9 → 
  (speed_kmph * 1000 / 3600) * time_sec = 180 :=
by
  intros speed_kmph time_sec h1 h2
  sorry

end train_length_72kmphr_9sec_180m_l10_10447


namespace fraction_irreducible_l10_10699

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by 
  sorry

end fraction_irreducible_l10_10699


namespace sum_n_k_eq_eight_l10_10976

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l10_10976


namespace second_number_value_l10_10433

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end second_number_value_l10_10433


namespace remainder_845307_div_6_l10_10149

theorem remainder_845307_div_6 :
  let n := 845307
  ∃ r : ℕ, n % 6 = r ∧ r = 3 :=
by
  let n := 845307
  have h_div_2 : ¬(n % 2 = 0) := by sorry
  have h_div_3 : n % 3 = 0 := by sorry
  exact ⟨3, by sorry, rfl⟩

end remainder_845307_div_6_l10_10149


namespace inequality_proof_l10_10853

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l10_10853


namespace smallest_positive_multiple_of_45_l10_10286

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ 45 * x = 45 :=
by {
  use 1,
  rw mul_one,
  exact nat.one_pos,
  sorry
}

end smallest_positive_multiple_of_45_l10_10286


namespace sum_n_k_eq_eight_l10_10977

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l10_10977


namespace number_of_students_in_class_l10_10237

theorem number_of_students_in_class
  (total_stickers : ℕ) (stickers_to_friends : ℕ) (stickers_left : ℝ) (students_each : ℕ → ℝ)
  (n_friends : ℕ) (remaining_stickers : ℝ) :
  total_stickers = 300 →
  stickers_to_friends = (n_friends * (n_friends + 1)) / 2 →
  stickers_left = 7.5 →
  ∀ n, n_friends = 10 →
  remaining_stickers = total_stickers - stickers_to_friends - (students_each n_friends) * (n - n_friends - 1) →
  (∃ n : ℕ, remaining_stickers = 7.5 ∧
              total_stickers - (stickers_to_friends + (students_each (n - n_friends - 1) * (n - n_friends - 1))) = 7.5) :=
by
  sorry

end number_of_students_in_class_l10_10237


namespace arithmetic_sequence_sum_l10_10235

-- Definitions based on problem conditions
variable (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) -- terms of the sequence
variable (S_3 S_6 S_9 : ℤ)

-- Given conditions
variable (h1 : S_3 = 3 * a_1 + 3 * (a_2 - a_1))
variable (h2 : S_6 = 6 * a_1 + 15 * (a_2 - a_1))
variable (h3 : S_3 = 9)
variable (h4 : S_6 = 36)

-- Theorem to prove
theorem arithmetic_sequence_sum : S_9 = 81 :=
by
  -- We just state the theorem here and will provide a proof later
  sorry

end arithmetic_sequence_sum_l10_10235


namespace min_value_3_div_a_add_2_div_b_l10_10509

/-- Given positive real numbers a and b, and the condition that the lines
(a + 1)x + 2y - 1 = 0 and 3x + (b - 2)y + 2 = 0 are perpendicular,
prove that the minimum value of 3/a + 2/b is 25, given the condition 3a + 2b = 1. -/
theorem min_value_3_div_a_add_2_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : 3 * a + 2 * b = 1) : 3 / a + 2 / b ≥ 25 :=
sorry

end min_value_3_div_a_add_2_div_b_l10_10509


namespace simple_interest_rate_l10_10035

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ) : 
  T = 6 → I = (7/6) * P - P → I = P * R * T / 100 → R = 100 / 36 :=
by
  intros T_eq I_eq simple_interest_eq
  sorry

end simple_interest_rate_l10_10035


namespace find_positions_l10_10030

def first_column (m : ℕ) : ℕ := 4 + 3*(m-1)

def table_element (m n : ℕ) : ℕ := first_column m + (n-1)*(2*m + 1)

theorem find_positions :
  (∀ m n, table_element m n ≠ 1994) ∧
  (∃ m n, table_element m n = 1995 ∧ ((m = 6 ∧ n = 153) ∨ (m = 153 ∧ n = 6))) :=
by
  sorry

end find_positions_l10_10030


namespace margie_drive_distance_l10_10682

theorem margie_drive_distance
  (miles_per_gallon : ℕ)
  (cost_per_gallon : ℕ)
  (dollar_amount : ℕ)
  (h₁ : miles_per_gallon = 32)
  (h₂ : cost_per_gallon = 4)
  (h₃ : dollar_amount = 20) :
  (dollar_amount / cost_per_gallon) * miles_per_gallon = 160 :=
by
  sorry

end margie_drive_distance_l10_10682


namespace min_value_of_expression_l10_10798

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) : 
  36 ≤ (1/x + 4/y + 9/z) :=
sorry

end min_value_of_expression_l10_10798


namespace Ayla_call_duration_l10_10772

theorem Ayla_call_duration
  (charge_per_minute : ℝ)
  (monthly_bill : ℝ)
  (customers_per_week : ℕ)
  (weeks_in_month : ℕ)
  (calls_duration : ℝ)
  (h_charge : charge_per_minute = 0.05)
  (h_bill : monthly_bill = 600)
  (h_customers : customers_per_week = 50)
  (h_weeks_in_month : weeks_in_month = 4)
  (h_calls_duration : calls_duration = (monthly_bill / charge_per_minute) / (customers_per_week * weeks_in_month)) :
  calls_duration = 60 :=
by 
  sorry

end Ayla_call_duration_l10_10772


namespace sum_of_bases_l10_10828

theorem sum_of_bases (F1 F2 : ℚ) (R1 R2 : ℕ) (hF1_R1 : F1 = (3 * R1 + 7) / (R1^2 - 1) ∧ F2 = (7 * R1 + 3) / (R1^2 - 1))
    (hF1_R2 : F1 = (2 * R2 + 5) / (R2^2 - 1) ∧ F2 = (5 * R2 + 2) / (R2^2 - 1)) : 
    R1 + R2 = 19 := 
sorry

end sum_of_bases_l10_10828


namespace inequality_solution_l10_10703

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 2 / (3 * x + 4) < 5) ↔ x ∈ set.Ioo (-5/3 : ℝ) (-4/3) ∪ set.Ioi (-4/3) :=
by
  sorry

end inequality_solution_l10_10703


namespace largest_multiple_5_6_lt_1000_is_990_l10_10880

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end largest_multiple_5_6_lt_1000_is_990_l10_10880


namespace mark_age_in_5_years_l10_10407

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end mark_age_in_5_years_l10_10407


namespace darts_game_score_l10_10409

variable (S1 S2 S3 : ℕ)
variable (n : ℕ)

theorem darts_game_score :
  n = 8 →
  S2 = 2 * S1 →
  S3 = (3 * S1) →
  S2 = 48 :=
by
  intros h1 h2 h3
  sorry

end darts_game_score_l10_10409


namespace probability_defective_first_box_l10_10134

noncomputable def box1_pieces : ℕ := 5
noncomputable def box1_defective_pieces : ℕ := 2
noncomputable def box2_pieces : ℕ := 10
noncomputable def box2_defective_pieces : ℕ := 3
noncomputable def total_boxes : ℕ := 2

theorem probability_defective_first_box :
  let p_box1 := (1 : ℚ) / total_boxes
      p_def_given_box1 := box1_defective_pieces / box1_pieces
      p_box2 := (1 : ℚ) / total_boxes
      p_def_given_box2 := box2_defective_pieces / box2_pieces
      p_def := p_box1 * p_def_given_box1 + p_box2 * p_def_given_box2
      p_def_and_box1 := p_box1 * p_def_given_box1 in
  (p_def_and_box1 / p_def) = 4 / 7 :=
by
  sorry

end probability_defective_first_box_l10_10134


namespace find_m_value_l10_10011

-- Definitions of the hyperbola and its focus condition
def hyperbola_eq (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / m) - (y^2 / (3 + m)) = 1

def focus_condition (m : ℝ) : Prop :=
  4 = (m) + (3 + m)

-- Theorem stating the value of m
theorem find_m_value (m : ℝ) : hyperbola_eq m → focus_condition m → m = 1 / 2 :=
by
  intros
  sorry

end find_m_value_l10_10011


namespace has_exactly_one_zero_interval_l10_10817

noncomputable def f (a x : ℝ) : ℝ := x^2 - a*x + 1

theorem has_exactly_one_zero_interval (a : ℝ) (h : a > 3) : ∃! x, 0 < x ∧ x < 2 ∧ f a x = 0 :=
sorry

end has_exactly_one_zero_interval_l10_10817


namespace smallest_positive_multiple_of_45_l10_10287

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l10_10287


namespace complement_of_M_l10_10214

open Set

def U : Set ℝ := univ

def M : Set ℝ := { x | x^2 - x ≥ 0 }

theorem complement_of_M :
  compl M = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end complement_of_M_l10_10214


namespace amount_A_received_l10_10626

-- Define the conditions
def total_amount : ℕ := 600
def ratio_a : ℕ := 1
def ratio_b : ℕ := 2

-- Define the total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b

-- Define the value of one part
def value_per_part : ℕ := total_amount / total_parts

-- Define the amount A gets
def amount_A_gets : ℕ := ratio_a * value_per_part

-- Lean statement to prove
theorem amount_A_received : amount_A_gets = 200 := by
  sorry

end amount_A_received_l10_10626


namespace john_age_l10_10894

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end john_age_l10_10894


namespace minimum_value_l10_10234

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ min_value_expr x y :=
  sorry

end minimum_value_l10_10234


namespace fourth_term_geometric_sequence_l10_10428

theorem fourth_term_geometric_sequence :
  let a := (6: ℝ)^(1/2)
  let b := (6: ℝ)^(1/6)
  let c := (6: ℝ)^(1/12)
  b = a * r ∧ c = a * r^2 → (a * r^3) = 1 := 
by
  sorry

end fourth_term_geometric_sequence_l10_10428


namespace vasya_claim_false_l10_10578

theorem vasya_claim_false :
  ∀ (weights : List ℕ), weights = [1, 2, 3, 4, 5, 6, 7] →
  (¬ ∃ (subset : List ℕ), subset.length = 3 ∧ 1 ∈ subset ∧
  ((weights.sum - subset.sum) = 14) ∧ (14 = 14)) :=
by
  sorry

end vasya_claim_false_l10_10578


namespace no_obtuse_triangle_probability_l10_10504

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l10_10504


namespace solve_x_l10_10950

-- Define the structure of the pyramid
def pyramid (x : ℕ) : Prop :=
  let level1 := [x + 4, 12, 15, 18]
  let level2 := [x + 16, 27, 33]
  let level3 := [x + 43, 60]
  let top := x + 103
  top = 120

theorem solve_x : ∃ x : ℕ, pyramid x → x = 17 :=
by
  -- Proof omitted
  sorry

end solve_x_l10_10950


namespace arithmetic_square_root_of_3_neg_2_l10_10054

theorem arithmetic_square_root_of_3_neg_2 : Real.sqrt (3 ^ (-2: Int)) = 1 / 3 := 
by 
  sorry

end arithmetic_square_root_of_3_neg_2_l10_10054


namespace difference_of_numbers_l10_10257

noncomputable def larger_num : ℕ := 1495
noncomputable def quotient : ℕ := 5
noncomputable def remainder : ℕ := 4

theorem difference_of_numbers :
  ∃ S : ℕ, larger_num = quotient * S + remainder ∧ (larger_num - S = 1197) :=
by 
  sorry

end difference_of_numbers_l10_10257


namespace increasing_m_range_l10_10212

noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x - 2 * m * x

theorem increasing_m_range (m : ℝ) : 
  (∀ x > 0, (2 * x + 1 / x - 2 * m ≥ 0)) → m ≤ Real.sqrt 2 :=
by
  intros h
  -- Proof steps would go here
  sorry

end increasing_m_range_l10_10212


namespace break_even_production_volume_l10_10757

theorem break_even_production_volume
  (Q : ℕ) 
  (ATC : ℕ → ℚ)
  (P : ℚ)
  (h1 : ∀ Q, ATC Q = 100 + 100000 / Q)
  (h2 : P = 300) :
  ATC 500 = P :=
by
  sorry

end break_even_production_volume_l10_10757


namespace cube_painting_l10_10466

theorem cube_painting (n : ℕ) (h : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) ↔ (n = 8) :=
by
  sorry

end cube_painting_l10_10466


namespace joan_exam_time_difference_l10_10398

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l10_10398


namespace convert_to_spherical_l10_10346

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then 3 * Real.pi / 2
           else if x > 0 then Real.arctan (y / x)
           else if y >= 0 then Real.arctan (y / x) + Real.pi
           else Real.arctan (y / x) - Real.pi
  (ρ, θ, φ)

theorem convert_to_spherical :
  rectangular_to_spherical (3 * Real.sqrt 2) (-4) 5 =
  (Real.sqrt 59, 2 * Real.pi + Real.arctan ((-4) / (3 * Real.sqrt 2)), Real.arccos (5 / Real.sqrt 59)) :=
by
  sorry

end convert_to_spherical_l10_10346


namespace number_of_mixed_vegetable_plates_l10_10469

theorem number_of_mixed_vegetable_plates :
  ∃ n : ℕ, n * 70 = 1051 - (16 * 6 + 5 * 45 + 6 * 40) ∧ n = 7 :=
by
  sorry

end number_of_mixed_vegetable_plates_l10_10469


namespace max_ab_l10_10296

theorem max_ab (a b : ℝ) (h : 4 * a + b = 1) (ha : a > 0) (hb : b > 0) : ab <= 1 / 16 :=
sorry

end max_ab_l10_10296


namespace conference_total_duration_is_715_l10_10609

structure ConferenceSession where
  hours : ℕ
  minutes : ℕ

def totalDuration (s1 s2 : ConferenceSession): ℕ :=
  (s1.hours * 60 + s1.minutes) + (s2.hours * 60 + s2.minutes)

def session1 : ConferenceSession := { hours := 8, minutes := 15 }
def session2 : ConferenceSession := { hours := 3, minutes := 40 }

theorem conference_total_duration_is_715 :
  totalDuration session1 session2 = 715 := 
sorry

end conference_total_duration_is_715_l10_10609


namespace cycling_sequences_reappear_after_28_cycles_l10_10013

/-- Cycling pattern of letters and digits. Letter cycle length is 7; digit cycle length is 4.
Prove that the LCM of 7 and 4 is 28, which is the first line on which both sequences will reappear -/
theorem cycling_sequences_reappear_after_28_cycles 
  (letters_cycle_length : ℕ) (digits_cycle_length : ℕ) 
  (h_letters : letters_cycle_length = 7) 
  (h_digits : digits_cycle_length = 4) 
  : Nat.lcm letters_cycle_length digits_cycle_length = 28 :=
by
  rw [h_letters, h_digits]
  sorry

end cycling_sequences_reappear_after_28_cycles_l10_10013


namespace mean_weight_correct_l10_10573

def weights := [51, 60, 62, 64, 64, 65, 67, 73, 74, 74, 75, 76, 77, 78, 79]

noncomputable def mean_weight (weights : List ℕ) : ℚ :=
  (weights.sum : ℚ) / weights.length

theorem mean_weight_correct :
  mean_weight weights = 69.27 := by
  sorry

end mean_weight_correct_l10_10573


namespace other_number_is_29_l10_10382

theorem other_number_is_29
    (k : ℕ)
    (some_number : ℕ)
    (h1 : k = 2)
    (h2 : (5 + k) * (5 - k) = some_number - 2^3) :
    some_number = 29 :=
by
  sorry

end other_number_is_29_l10_10382


namespace fraction_of_b_eq_three_tenths_a_l10_10292

theorem fraction_of_b_eq_three_tenths_a (a b : ℝ) (h1 : a + b = 100) (h2 : b = 60) :
  (3 / 10) * a = (1 / 5) * b :=
by 
  have h3 : a = 40 := by linarith [h1, h2]
  rw [h2, h3]
  linarith

end fraction_of_b_eq_three_tenths_a_l10_10292


namespace problem_solution_l10_10516

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 3^x else m - x^2

def p (m : ℝ) : Prop :=
∃ x, f m x = 0

def q (m : ℝ) : Prop :=
m = 1 / 9 → f m (f m (-1)) = 0

theorem problem_solution :
  ¬ (∃ m, m < 0 ∧ p m) ∧ q (1 / 9) :=
by 
  sorry

end problem_solution_l10_10516


namespace maximize_tetrahedron_volume_l10_10041

noncomputable def volume_maximized_ob_length : ℝ :=
  let PA := 4 in
  let PC := PA / 2 in
  let OC := PC in
  let HC := OC in
  let OP := OC in
  OP * real.tan (real.pi / 6)

theorem maximize_tetrahedron_volume :
  let OB := volume_maximized_ob_length
  in OB = 2 * real.sqrt 6 / 3 :=
begin
  sorry
end

end maximize_tetrahedron_volume_l10_10041


namespace equation_of_tangent_circle_l10_10790

-- Define the point and conditional tangency
def center : ℝ × ℝ := (5, 4)
def tangent_to_x_axis : Prop := true -- Placeholder for the tangency condition, which is encoded in our reasoning

-- Define the proof statement
theorem equation_of_tangent_circle :
  (∀ (x y : ℝ), tangent_to_x_axis → 
  (center = (5, 4)) → 
  ((x - 5) ^ 2 + (y - 4) ^ 2 = 16)) := 
sorry

end equation_of_tangent_circle_l10_10790


namespace binomial_arithmetic_sequence_l10_10598

theorem binomial_arithmetic_sequence (n : ℕ) (h : n > 3)
  (C : ℕ → ℕ → ℕ)
  (hC1 : C n 1 = n)
  (hC2 : C n 2 = n * (n - 1) / 2)
  (hC3 : C n 3 = n * (n - 1) * (n - 2) / 6) :
  C n 2 - C n 1 = C n 3 - C n 2 → n = 7 := sorry

end binomial_arithmetic_sequence_l10_10598


namespace expected_waiting_time_approx_l10_10771

noncomputable def expectedWaitingTime : ℚ :=
  (10 * (1/2) + 30 * (1/3) + 50 * (1/36) + 70 * (1/12) + 90 * (1/18))

theorem expected_waiting_time_approx :
  abs (expectedWaitingTime - 27.22) < 1 :=
by
  sorry

end expected_waiting_time_approx_l10_10771


namespace number_of_people_who_selected_dog_l10_10827

theorem number_of_people_who_selected_dog 
  (total : ℕ) 
  (cat : ℕ) 
  (fish : ℕ) 
  (bird : ℕ) 
  (other : ℕ) 
  (h_total : total = 90) 
  (h_cat : cat = 25) 
  (h_fish : fish = 10) 
  (h_bird : bird = 15) 
  (h_other : other = 5) :
  (total - (cat + fish + bird + other) = 35) :=
by
  sorry

end number_of_people_who_selected_dog_l10_10827


namespace max_value_expression_l10_10840

noncomputable def target_expr (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)

theorem max_value_expression (x y z : ℝ) (h : x + y + z = 3) (hxy : x = y) (hxz : 0 ≤ x) (hyz : 0 ≤ y) (hzz : 0 ≤ z) :
  target_expr x y z ≤ 9 / 4 := by
  sorry

end max_value_expression_l10_10840


namespace smallest_angle_of_triangle_l10_10725

theorem smallest_angle_of_triangle :
  ∀ a b c : ℝ, a = 2 * Real.sqrt 10 → b = 3 * Real.sqrt 5 → c = 5 → 
  ∃ α β γ : ℝ, α + β + γ = π ∧ α = 45 * (π / 180) ∧ (a = c → α < β ∧ α < γ) ∧ (b = c → β < α ∧ β < γ) ∧ (c = a → γ < α ∧ γ < β) → 
  α = 45 * (π / 180) := 
sorry

end smallest_angle_of_triangle_l10_10725


namespace volume_after_increase_l10_10325

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l10_10325


namespace total_money_l10_10300

theorem total_money (a b c : ℕ) (h_ratio : (a / 2) / (b / 3) / (c / 4) = 1) (h_c : c = 306) : 
  a + b + c = 782 := 
by sorry

end total_money_l10_10300


namespace compute_g_f_1_l10_10679

def f (x : ℝ) : ℝ := x^3 - 2 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem compute_g_f_1 : g (f 1) = 3 :=
by
  sorry

end compute_g_f_1_l10_10679


namespace fill_pool_time_l10_10460

theorem fill_pool_time :
  let pool_capacity := 36000 in
  let hose_count := 6 in
  let flow_rate_per_hose := 3 in
  let flow_rate_per_minute := hose_count * flow_rate_per_hose in
  let flow_rate_per_hour := flow_rate_per_minute * 60 in
  let fill_time := pool_capacity / flow_rate_per_hour in
  fill_time = 100 / 3 := sorry

end fill_pool_time_l10_10460


namespace ten_percent_of_x_is_17_85_l10_10891

-- Define the conditions and the proof statement
theorem ten_percent_of_x_is_17_85 :
  ∃ x : ℝ, (3 - (1/4) * 2 - (1/3) * 3 - (1/7) * x = 27) ∧ (0.10 * x = 17.85) := sorry

end ten_percent_of_x_is_17_85_l10_10891


namespace find_int_solutions_l10_10352

theorem find_int_solutions (x : ℤ) :
  (∃ p : ℤ, Prime p ∧ 2*x^2 - x - 36 = p^2) ↔ (x = 5 ∨ x = 13) := 
sorry

end find_int_solutions_l10_10352


namespace simplify_expression_l10_10130

variable (a b : ℝ)

theorem simplify_expression :
  -2 * (a^3 - 3 * b^2) + 4 * (-b^2 + a^3) = 2 * a^3 + 2 * b^2 :=
by
  sorry

end simplify_expression_l10_10130


namespace true_supporters_of_rostov_l10_10126

theorem true_supporters_of_rostov
  (knights_liars_fraction : ℕ → ℕ)
  (rostov_support_yes : ℕ)
  (zenit_support_yes : ℕ)
  (lokomotiv_support_yes : ℕ)
  (cska_support_yes : ℕ)
  (h1 : knights_liars_fraction 100 = 10)
  (h2 : rostov_support_yes = 40)
  (h3 : zenit_support_yes = 30)
  (h4 : lokomotiv_support_yes = 50)
  (h5 : cska_support_yes = 0):
  rostov_support_yes - knights_liars_fraction 100 = 30 := 
sorry

end true_supporters_of_rostov_l10_10126


namespace maximize_winning_probability_l10_10729

def ahmet_wins (n : ℕ) : Prop :=
  n = 13

theorem maximize_winning_probability :
  ∃ n ∈ {x : ℕ | x ≥ 1 ∧ x ≤ 25}, ahmet_wins n :=
by
  sorry

end maximize_winning_probability_l10_10729


namespace mean_and_sum_l10_10269

-- Define the sum of five numbers to be 1/3
def sum_of_five_numbers : ℚ := 1 / 3

-- Define the mean of these five numbers
def mean_of_five_numbers : ℚ := sum_of_five_numbers / 5

-- State the theorem
theorem mean_and_sum (h : sum_of_five_numbers = 1 / 3) :
  mean_of_five_numbers = 1 / 15 ∧ (mean_of_five_numbers + sum_of_five_numbers = 2 / 5) :=
by
  sorry

end mean_and_sum_l10_10269


namespace ratio_of_radii_l10_10279

theorem ratio_of_radii (r R : ℝ) (k : ℝ) (h1 : R > r) (h2 : π * R^2 - π * r^2 = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
sorry

end ratio_of_radii_l10_10279


namespace farmer_trees_l10_10044

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end farmer_trees_l10_10044


namespace trig_identity_l10_10359

theorem trig_identity (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1 / 2) : 
  (Real.cos a / (Real.sin a - 1)) = 1 / 2 := by
  -- Proof goes here
  sorry

end trig_identity_l10_10359


namespace quadratic_inequality_solution_l10_10917

theorem quadratic_inequality_solution :
  {x : ℝ | 3 * x^2 + 5 * x < 8} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
sorry

end quadratic_inequality_solution_l10_10917


namespace range_of_a_l10_10142

theorem range_of_a (a : ℝ) : (5 - a > 1) → (a < 4) := 
by
  sorry

end range_of_a_l10_10142


namespace total_weight_is_correct_l10_10236

def siblings_suitcases : Nat := 1 + 2 + 3 + 4 + 5 + 6
def weight_per_sibling_suitcase : Nat := 10
def total_weight_siblings : Nat := siblings_suitcases * weight_per_sibling_suitcase

def parents : Nat := 2
def suitcases_per_parent : Nat := 3
def weight_per_parent_suitcase : Nat := 12
def total_weight_parents : Nat := parents * suitcases_per_parent * weight_per_parent_suitcase

def grandparents : Nat := 2
def suitcases_per_grandparent : Nat := 2
def weight_per_grandparent_suitcase : Nat := 8
def total_weight_grandparents : Nat := grandparents * suitcases_per_grandparent * weight_per_grandparent_suitcase

def other_relatives_suitcases : Nat := 8
def weight_per_other_relatives_suitcase : Nat := 15
def total_weight_other_relatives : Nat := other_relatives_suitcases * weight_per_other_relatives_suitcase

def total_weight_all_suitcases : Nat := total_weight_siblings + total_weight_parents + total_weight_grandparents + total_weight_other_relatives

theorem total_weight_is_correct : total_weight_all_suitcases = 434 := by {
  sorry
}

end total_weight_is_correct_l10_10236


namespace wall_bricks_count_l10_10344

def alice_rate (y : ℕ) : ℕ := y / 8
def bob_rate (y : ℕ) : ℕ := y / 12
def combined_rate (y : ℕ) : ℕ := (5 * y) / 24 - 12
def effective_working_time : ℕ := 6

theorem wall_bricks_count :
  ∃ y : ℕ, (combined_rate y * effective_working_time = y) ∧ y = 288 :=
by
  sorry

end wall_bricks_count_l10_10344


namespace value_of_expression_l10_10691

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 6) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 228498 := by
  sorry

end value_of_expression_l10_10691


namespace number_of_numbers_l10_10711

theorem number_of_numbers (n : ℕ) (S : ℕ) 
  (h1 : (S + 26) / n = 16) 
  (h2 : (S + 46) / n = 18) : 
  n = 10 := 
by 
  -- placeholder for the proof
  sorry

end number_of_numbers_l10_10711


namespace adele_age_fraction_l10_10681

theorem adele_age_fraction 
  (jackson_age : ℕ) 
  (mandy_age : ℕ) 
  (adele_age_fraction : ℚ) 
  (total_age_10_years : ℕ)
  (H1 : jackson_age = 20)
  (H2 : mandy_age = jackson_age + 10)
  (H3 : total_age_10_years = (jackson_age + 10) + (mandy_age + 10) + (jackson_age * adele_age_fraction + 10))
  (H4 : total_age_10_years = 95) : 
  adele_age_fraction = 3 / 4 := 
sorry

end adele_age_fraction_l10_10681


namespace discount_percentage_l10_10005

theorem discount_percentage (M C S : ℝ) (hC : C = 0.64 * M) (hS : S = C * 1.28125) :
  ((M - S) / M) * 100 = 18.08 := 
by
  sorry

end discount_percentage_l10_10005


namespace tina_money_left_l10_10277

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end tina_money_left_l10_10277


namespace bart_earned_14_l10_10773

variable (questions_per_survey money_per_question surveys_monday surveys_tuesday : ℕ → ℝ)
variable (total_surveys total_questions money_earned : ℕ → ℝ)

noncomputable def conditions :=
  let questions_per_survey := 10
  let money_per_question := 0.2
  let surveys_monday := 3
  let surveys_tuesday := 4
  let total_surveys := surveys_monday + surveys_tuesday
  let total_questions := questions_per_survey * total_surveys
  let money_earned := total_questions * money_per_question
  money_earned = 14

theorem bart_earned_14 : conditions :=
by
  -- proof steps
  sorry

end bart_earned_14_l10_10773


namespace machine_a_produces_6_sprockets_per_hour_l10_10680

theorem machine_a_produces_6_sprockets_per_hour : 
  ∀ (A G T : ℝ), 
  (660 = A * (T + 10)) → 
  (660 = G * T) → 
  (G = 1.10 * A) → 
  A = 6 := 
by
  intros A G T h1 h2 h3
  sorry

end machine_a_produces_6_sprockets_per_hour_l10_10680


namespace work_completion_days_l10_10456

theorem work_completion_days
    (A : ℝ) (B : ℝ) (h1 : 1 / A + 1 / B = 1 / 10)
    (h2 : B = 35) :
    A = 14 :=
by
  sorry

end work_completion_days_l10_10456


namespace find_common_ratio_of_geometric_sequence_l10_10809

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem find_common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a n > a (n + 1))
  (h1 : a 1 * a 5 = 9)
  (h2 : a 2 + a 4 = 10) : 
  q = -1/3 :=
sorry

end find_common_ratio_of_geometric_sequence_l10_10809


namespace total_students_l10_10252

theorem total_students (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hshake : (2 * m * n - m - n) = 252) : m * n = 72 :=
  sorry

end total_students_l10_10252


namespace initial_production_rate_l10_10338

variable (x : ℕ) (t : ℝ)

-- Conditions
def produces_initial (x : ℕ) (t : ℝ) : Prop := x * t = 60
def produces_subsequent : Prop := 60 * 1 = 60
def overall_average (t : ℝ) : Prop := 72 = 120 / (t + 1)

-- Goal: Prove the initial production rate
theorem initial_production_rate : 
  (∃ t : ℝ, produces_initial x t ∧ produces_subsequent ∧ overall_average t) → x = 90 := 
  by
    sorry

end initial_production_rate_l10_10338


namespace craig_total_distance_l10_10476

-- Define the distances Craig walked
def dist_school_to_david : ℝ := 0.27
def dist_david_to_home : ℝ := 0.73

-- Prove the total distance walked
theorem craig_total_distance : dist_school_to_david + dist_david_to_home = 1.00 :=
by
  -- Proof goes here
  sorry

end craig_total_distance_l10_10476


namespace largest_consecutive_even_integer_l10_10144

theorem largest_consecutive_even_integer (n : ℕ) (h : 5 * n - 20 = 2 * 15 * 16 / 2) : n = 52 :=
sorry

end largest_consecutive_even_integer_l10_10144


namespace coefficients_sum_l10_10819

theorem coefficients_sum:
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (1+x)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  have h0 : a_0 = 1
  sorry -- proof when x=0
  have h1 : a_1 + a_2 + a_3 + a_4 + a_5 = 31
  sorry -- proof when x=1
  exact h1

end coefficients_sum_l10_10819


namespace distinct_prime_divisors_l10_10114

theorem distinct_prime_divisors (a : ℤ) (n : ℕ) (h₁ : a > 3) (h₂ : Odd a) (h₃ : n > 0) : 
  ∃ (p : Finset ℤ), p.card ≥ n + 1 ∧ ∀ q ∈ p, Prime q ∧ q ∣ (a ^ (2 ^ n) - 1) :=
sorry

end distinct_prime_divisors_l10_10114


namespace probability_at_least_one_multiple_of_4_l10_10176

-- Define the condition
def random_integer_between_1_and_60 : set ℤ := {n : ℤ | 1 ≤ n ∧ n ≤ 60}

-- Define the probability theorems and the proof for probability calculation
theorem probability_at_least_one_multiple_of_4 :
  (∀ (n1 n2 : ℤ), (n1 ∈ random_integer_between_1_and_60) ∧ (n2 ∈ random_integer_between_1_and_60) → 
  (∃ k, n1 = 4 * k ∨ ∃ k, n2 = 4 * k)) ∧ 
  (1 / 60 * 1 / 60) * (15 / 60) ^ 2 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_l10_10176


namespace neg_prop_p_equiv_l10_10367

variable {x : ℝ}

def prop_p : Prop := ∃ x ≥ 0, 2^x = 3

theorem neg_prop_p_equiv : ¬prop_p ↔ ∀ x ≥ 0, 2^x ≠ 3 :=
by sorry

end neg_prop_p_equiv_l10_10367


namespace part1_part2_l10_10999

-- Part 1: Prove that x < -12 given the inequality 2(-3 + x) > 3(x + 2)
theorem part1 (x : ℝ) : 2 * (-3 + x) > 3 * (x + 2) → x < -12 := 
  by
  intro h
  sorry

-- Part 2: Prove that 0 ≤ x < 3 given the system of inequalities
theorem part2 (x : ℝ) : 
    (1 / 2) * (x + 1) < 2 ∧ (x + 2) / 2 ≥ (x + 3) / 3 → 0 ≤ x ∧ x < 3 :=
  by
  intro h
  sorry

end part1_part2_l10_10999


namespace hexagonal_tiles_in_box_l10_10606

theorem hexagonal_tiles_in_box :
  ∃ a b c : ℕ, a + b + c = 35 ∧ 3 * a + 4 * b + 6 * c = 128 ∧ c = 6 :=
by
  sorry

end hexagonal_tiles_in_box_l10_10606


namespace pump1_half_drain_time_l10_10848

-- Definitions and Conditions
def time_to_drain_half_pump1 (t : ℝ) : Prop :=
  ∃ rate1 rate2 : ℝ, 
    rate1 = 1 / (2 * t) ∧
    rate2 = 1 / 1.25 ∧
    rate1 + rate2 = 2

-- Equivalent Proof Problem
theorem pump1_half_drain_time (t : ℝ) : time_to_drain_half_pump1 t → t = 5 / 12 := sorry

end pump1_half_drain_time_l10_10848


namespace candidate_lost_by_2340_votes_l10_10753

theorem candidate_lost_by_2340_votes
  (total_votes : ℝ)
  (candidate_percentage : ℝ)
  (rival_percentage : ℝ)
  (candidate_votes : ℝ)
  (rival_votes : ℝ)
  (votes_difference : ℝ)
  (h1 : total_votes = 7800)
  (h2 : candidate_percentage = 0.35)
  (h3 : rival_percentage = 0.65)
  (h4 : candidate_votes = candidate_percentage * total_votes)
  (h5 : rival_votes = rival_percentage * total_votes)
  (h6 : votes_difference = rival_votes - candidate_votes) :
  votes_difference = 2340 :=
by
  sorry

end candidate_lost_by_2340_votes_l10_10753


namespace has_two_roots_l10_10318

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l10_10318


namespace binomial_np_sum_l10_10510

-- Definitions of variance and expectation for a binomial distribution
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)
def binomial_expectation (n : ℕ) (p : ℚ) : ℚ := n * p

-- Statement of the problem
theorem binomial_np_sum (n : ℕ) (p : ℚ) (h_var : binomial_variance n p = 4) (h_exp : binomial_expectation n p = 12) :
    n + p = 56 / 3 := by
  sorry

end binomial_np_sum_l10_10510


namespace range_of_a_l10_10199

theorem range_of_a (f : ℝ → ℝ) (a : ℝ):
  (∀ x, f x = f (-x)) →
  (∀ x y, 0 ≤ x → x < y → f x ≤ f y) →
  (∀ x, 1/2 ≤ x ∧ x ≤ 1 → f (a * x + 1) ≤ f (x - 2)) →
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l10_10199


namespace range_of_a_l10_10813

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l10_10813


namespace B_C_work_days_l10_10818

noncomputable def days_for_B_and_C {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) : ℝ :=
  30 / 7

theorem B_C_work_days {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) :
  days_for_B_and_C hA hA_B hA_B_C = 30 / 7 :=
sorry

end B_C_work_days_l10_10818


namespace number_of_bottles_l10_10436

-- Define the weights and total weight based on given conditions
def weight_of_two_bags_chips : ℕ := 800
def total_weight_five_bags_and_juices : ℕ := 2200
def weight_difference_chip_Juice : ℕ := 350

-- Considering 1 bag of chips weighs 400 g (derived from the condition)
def weight_of_one_bag_chips : ℕ := 400
def weight_of_one_bottle_juice : ℕ := weight_of_one_bag_chips - weight_difference_chip_Juice

-- Define the proof of the question
theorem number_of_bottles :
  (total_weight_five_bags_and_juices - (5 * weight_of_one_bag_chips)) / weight_of_one_bottle_juice = 4 := by sorry

end number_of_bottles_l10_10436


namespace johns_age_l10_10895

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end johns_age_l10_10895


namespace quadratic_trinomial_has_two_roots_l10_10315

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l10_10315


namespace positive_number_property_l10_10169

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_property : (x^2 / 100) = 9) : x = 30 :=
by
  sorry

end positive_number_property_l10_10169


namespace fraction_difference_l10_10940

variable (x y : ℝ)

theorem fraction_difference (h : x / y = 2) : (x - y) / y = 1 :=
  sorry

end fraction_difference_l10_10940


namespace dad_steps_l10_10623

theorem dad_steps (dad_steps_ratio: ℕ) (masha_steps_ratio: ℕ) (masha_steps: ℕ)
  (masha_and_yasha_steps: ℕ) (total_steps: ℕ)
  (h1: dad_steps_ratio * 3 = masha_steps_ratio * 5)
  (h2: masha_steps * 3 = masha_and_yasha_steps * 5)
  (h3: masha_and_yasha_steps = total_steps)
  (h4: total_steps = 400) :
  dad_steps_ratio * 30 = 90 :=
by
  sorry

end dad_steps_l10_10623


namespace simplify_fraction_l10_10701

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 :=
by
  sorry

end simplify_fraction_l10_10701


namespace petya_cannot_have_equal_coins_l10_10718

def petya_initial_two_kopeck_coins : Nat := 1
def petya_initial_ten_kopeck_coins : Nat := 0
def petya_use_ten_kopeck (T G : Nat) : Nat := G - 1 + T + 5
def petya_use_two_kopeck (T G : Nat) : Nat := T - 1 + G + 5

theorem petya_cannot_have_equal_coins : ¬ (∃ n : Nat, 
  ∃ T G : Nat, 
    T = G ∧ 
    (n = petya_use_ten_kopeck T G ∨ n = petya_use_two_kopeck T G ∨ n = petya_initial_two_kopeck_coins + petya_initial_ten_kopeck_coins)) := 
by
  sorry

end petya_cannot_have_equal_coins_l10_10718


namespace bricklayer_team_size_l10_10333

/-- Problem: Prove the number of bricklayers in the team -/
theorem bricklayer_team_size
  (x : ℕ)
  (h1 : 432 = (432 * (x - 4) / x) + 9 * (x - 4)) :
  x = 16 :=
sorry

end bricklayer_team_size_l10_10333


namespace michael_average_speed_l10_10958

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end michael_average_speed_l10_10958


namespace initial_apples_l10_10952

-- Define the number of initial fruits
def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def fruits_given : ℕ := 40
def fruits_left : ℕ := 15

-- Define the equation for the initial number of fruits
def initial_total_fruits (A : ℕ) : Prop :=
  initial_plums + initial_guavas + A = fruits_left + fruits_given

-- Define the proof problem to find the number of apples
theorem initial_apples : ∃ A : ℕ, initial_total_fruits A ∧ A = 21 :=
  by
    sorry

end initial_apples_l10_10952


namespace max_sum_of_inverses_l10_10954

open Set

theorem max_sum_of_inverses 
  (n : ℕ) (h : n ≥ 5)
  (a : Fin n → ℕ) 
  (ha : Function.Injective a)
  (hb : ∀ A B : Finset (Fin n), A ≠ B → A.nonempty → B.nonempty → (A.sum (λ i => a i) ≠ B.sum (λ i => a i))) :
  (Finset.univ.sum (λ i => (1 : ℚ) / a i) ≤ 2 - 1 / 2 ^ (n - 1)) :=
by
  sorry

end max_sum_of_inverses_l10_10954


namespace xy_equals_twelve_l10_10099

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by
  sorry

end xy_equals_twelve_l10_10099


namespace negation_of_original_prop_l10_10430

variable (a : ℝ)
def original_prop (x : ℝ) : Prop := x^2 + a * x + 1 < 0

theorem negation_of_original_prop :
  ¬ (∃ x : ℝ, original_prop a x) ↔ ∀ x : ℝ, ¬ original_prop a x :=
by sorry

end negation_of_original_prop_l10_10430


namespace price_range_of_book_l10_10590

variable (x : ℝ)

theorem price_range_of_book (h₁ : ¬(x ≥ 15)) (h₂ : ¬(x ≤ 12)) (h₃ : ¬(x ≤ 10)) : 12 < x ∧ x < 15 := 
by
  sorry

end price_range_of_book_l10_10590


namespace find_sum_fusion_number_l10_10385

def sum_fusion_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * (2 * k + 1)

theorem find_sum_fusion_number (n : ℕ) :
  n = 2020 ↔ sum_fusion_number n :=
sorry

end find_sum_fusion_number_l10_10385


namespace number_of_members_l10_10262

-- Definitions based on conditions in the problem
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def cap_cost : ℕ := tshirt_cost

def home_game_cost_per_member : ℕ := sock_cost + tshirt_cost
def away_game_cost_per_member : ℕ := sock_cost + tshirt_cost + cap_cost
def total_cost_per_member : ℕ := home_game_cost_per_member + away_game_cost_per_member

def total_league_cost : ℕ := 4324

-- Statement to be proved
theorem number_of_members (m : ℕ) (h : total_league_cost = m * total_cost_per_member) : m = 85 :=
sorry

end number_of_members_l10_10262


namespace zero_is_neither_positive_nor_negative_l10_10993

theorem zero_is_neither_positive_nor_negative :
  ¬ (0 > 0) ∧ ¬ (0 < 0) :=
by
  sorry

end zero_is_neither_positive_nor_negative_l10_10993


namespace rational_values_of_expressions_l10_10889

theorem rational_values_of_expressions {x : ℚ} :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by
  sorry

end rational_values_of_expressions_l10_10889


namespace monotonically_increasing_intervals_min_and_max_values_l10_10085

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.sin (2 * x + Real.pi / 4) + 1

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, 
    -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi → 
    f (x + 1) ≥ f x := sorry

theorem min_and_max_values :
  ∃ min max, 
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≥ min ∧ f x ≤ max) ∧ 
    (min = 0) ∧ 
    (max = Real.sqrt 2 + 1) := sorry

end monotonically_increasing_intervals_min_and_max_values_l10_10085


namespace who_plays_piano_l10_10619

theorem who_plays_piano 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (hA : A = True)
  (hB : B = False)
  (hC : A = False)
  (only_one_true : (A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C)) : B = True := 
sorry

end who_plays_piano_l10_10619


namespace area_shaded_quad_correct_l10_10023

-- Define the side lengths of the squares
def side_length_small : ℕ := 3
def side_length_middle : ℕ := 5
def side_length_large : ℕ := 7

-- Define the total base length
def total_base_length : ℕ := side_length_small + side_length_middle + side_length_large

-- The height of triangle T3, equal to the side length of the largest square
def height_T3 : ℕ := side_length_large

-- The height-to-base ratio for each triangle
def height_to_base_ratio : ℚ := height_T3 / total_base_length

-- The heights of T1 and T2
def height_T1 : ℚ := side_length_small * height_to_base_ratio
def height_T2 : ℚ := (side_length_small + side_length_middle) * height_to_base_ratio

-- The height of the trapezoid, which is the side length of the middle square
def trapezoid_height : ℕ := side_length_middle

-- The bases of the trapezoid
def base1 : ℚ := height_T1
def base2 : ℚ := height_T2

-- The area of the trapezoid formula
def area_shaded_quad : ℚ := (trapezoid_height * (base1 + base2)) / 2

-- Assertion that the area of the shaded quadrilateral is equal to 77/6
theorem area_shaded_quad_correct : area_shaded_quad = 77 / 6 := by sorry

end area_shaded_quad_correct_l10_10023


namespace sun_radius_scientific_notation_l10_10866

theorem sun_radius_scientific_notation : 
  (369000 : ℝ) = 3.69 * 10^5 :=
by
  sorry

end sun_radius_scientific_notation_l10_10866


namespace dropped_score_l10_10037

variable (A B C D : ℕ)

theorem dropped_score (h1 : A + B + C + D = 180) (h2 : A + B + C = 150) : D = 30 := by
  sorry

end dropped_score_l10_10037


namespace greatest_four_digit_divisible_by_conditions_l10_10597

-- Definitions based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

-- Problem statement: Finding the greatest 4-digit number divisible by 15, 25, 40, and 75
theorem greatest_four_digit_divisible_by_conditions :
  ∃ n, is_four_digit n ∧ is_divisible_by n 15 ∧ is_divisible_by n 25 ∧ is_divisible_by n 40 ∧ is_divisible_by n 75 ∧ n = 9600 :=
  sorry

end greatest_four_digit_divisible_by_conditions_l10_10597


namespace find_floor_l10_10692

-- Define the total number of floors
def totalFloors : ℕ := 9

-- Define the total number of entrances
def totalEntrances : ℕ := 10

-- Each floor has the same number of apartments
-- The claim we are to prove is that for entrance 10 and apartment 333, Petya needs to go to the 3rd floor.

theorem find_floor (apartment_number : ℕ) (entrance_number : ℕ) (floor : ℕ)
  (h1 : entrance_number = 10)
  (h2 : apartment_number = 333)
  (h3 : ∀ (f : ℕ), 0 < f ∧ f ≤ totalFloors)
  (h4 : ∃ (n : ℕ), totalEntrances * totalFloors * n >= apartment_number)
  : floor = 3 :=
  sorry

end find_floor_l10_10692


namespace trapezoid_cd_length_l10_10147

noncomputable def proof_cd_length (AD BC CD : ℝ) (BD : ℝ) (angle_DBA angle_BDC : ℝ) (ratio_BC_AD : ℝ) : Prop :=
  AD > 0 ∧ BC > 0 ∧
  BD = 1 ∧
  angle_DBA = 23 ∧
  angle_BDC = 46 ∧
  ratio_BC_AD = 9 / 5 ∧
  AD / BC = 5 / 9 ∧
  CD = 4 / 5

theorem trapezoid_cd_length
  (AD BC CD : ℝ)
  (BD : ℝ := 1)
  (angle_DBA : ℝ := 23)
  (angle_BDC : ℝ := 46)
  (ratio_BC_AD : ℝ := 9 / 5)
  (h_conditions : proof_cd_length AD BC CD BD angle_DBA angle_BDC ratio_BC_AD) : CD = 4 / 5 :=
sorry

end trapezoid_cd_length_l10_10147


namespace total_parking_spaces_l10_10869

-- Definitions of conditions
def caravan_space : ℕ := 2
def number_of_caravans : ℕ := 3
def spaces_left : ℕ := 24

-- Proof statement
theorem total_parking_spaces :
  (number_of_caravans * caravan_space + spaces_left) = 30 :=
by
  sorry

end total_parking_spaces_l10_10869


namespace price_per_eraser_l10_10728

-- Definitions of the given conditions
def boxes_donated : ℕ := 48
def erasers_per_box : ℕ := 24
def total_money_made : ℝ := 864

-- The Lean statement to prove the price per eraser is $0.75
theorem price_per_eraser : (total_money_made / (boxes_donated * erasers_per_box) = 0.75) := by
  sorry

end price_per_eraser_l10_10728


namespace compound_interest_second_year_l10_10569

theorem compound_interest_second_year
  (P: ℝ) (r: ℝ) (CI_3 : ℝ) (CI_2 : ℝ)
  (h1 : r = 0.06)
  (h2 : CI_3 = 1272)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1200 :=
by
  sorry

end compound_interest_second_year_l10_10569


namespace intersection_is_neg2_l10_10956

-- Define sets A and B
def A : Set ℤ := {-3, -2, -1, 0, 1}
def B : Set ℤ := {x | x * x - 4 = 0}

-- Goal: Prove that the intersection of A and B is {-2}
theorem intersection_is_neg2 : A ∩ B = {-2} :=
by
  sorry

end intersection_is_neg2_l10_10956


namespace map_length_representation_l10_10125

variable (x : ℕ)

theorem map_length_representation :
  (12 : ℕ) * x = 17 * (72 : ℕ) / 12
:=
sorry

end map_length_representation_l10_10125


namespace division_of_negatives_l10_10913

theorem division_of_negatives (x y : Int) (h1 : y ≠ 0) (h2 : -x = 150) (h3 : -y = 25) : (-150) / (-25) = 6 :=
by
  sorry

end division_of_negatives_l10_10913


namespace probability_of_at_least_one_multiple_of_4_l10_10177

open ProbabilityTheory

def prob_at_least_one_multiple_of_4 : ℚ := 7 / 16

theorem probability_of_at_least_one_multiple_of_4 :
  let S := Finset.range 60
  let multiples_of_4 := S.filter (λ x, (x + 1) % 4 = 0)
  let prob (a b : ℕ) := (a : ℚ) / b
  let prob_neither_multiple_4 := (prob (60 - multiples_of_4.card) 60) ^ 2
  1 - prob_neither_multiple_4 = prob_at_least_one_multiple_of_4 := by
  sorry

end probability_of_at_least_one_multiple_of_4_l10_10177


namespace weight_difference_l10_10135

theorem weight_difference (brown black white grey : ℕ) 
  (h_brown : brown = 4)
  (h_white : white = 2 * brown)
  (h_grey : grey = black - 2)
  (avg_weight : (brown + black + white + grey) / 4 = 5): 
  (black - brown) = 1 := by
  sorry

end weight_difference_l10_10135


namespace expression_value_l10_10272

theorem expression_value : (8 * 6) - (4 / 2) = 46 :=
by
  sorry

end expression_value_l10_10272


namespace periodic_function_l10_10231

open Real

theorem periodic_function (f : ℝ → ℝ) 
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func_eq : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) : 
  ∀ x : ℝ, f (x + 1) = f x := 
  sorry

end periodic_function_l10_10231


namespace minimum_dimes_needed_l10_10783

theorem minimum_dimes_needed (n : ℕ) 
  (sneaker_cost : ℝ := 58) 
  (ten_bills : ℝ := 50)
  (five_quarters : ℝ := 1.25) :
  ten_bills + five_quarters + (0.10 * n) ≥ sneaker_cost ↔ n ≥ 68 := 
by 
  sorry

end minimum_dimes_needed_l10_10783


namespace students_in_class_l10_10731

def total_eggs : Nat := 56
def eggs_per_student : Nat := 8
def num_students : Nat := 7

theorem students_in_class :
  total_eggs / eggs_per_student = num_students :=
by
  sorry

end students_in_class_l10_10731


namespace smallest_product_not_factor_60_l10_10280

theorem smallest_product_not_factor_60 : ∃ (a b : ℕ), a ≠ b ∧ a ∣ 60 ∧ b ∣ 60 ∧ ¬ (a * b) ∣ 60 ∧ a * b = 8 := sorry

end smallest_product_not_factor_60_l10_10280


namespace LCM_14_21_l10_10031

theorem LCM_14_21 : Nat.lcm 14 21 = 42 := 
by
  sorry

end LCM_14_21_l10_10031


namespace fraction_simplification_l10_10442

theorem fraction_simplification : 1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 :=
by
  sorry

end fraction_simplification_l10_10442


namespace repeating_decimal_fraction_l10_10920

noncomputable def repeating_decimal := 4.66666 -- Assuming repeating forever

theorem repeating_decimal_fraction : repeating_decimal = 14 / 3 :=
by 
  sorry

end repeating_decimal_fraction_l10_10920


namespace shells_not_red_or_green_l10_10467

theorem shells_not_red_or_green (total_shells : ℕ) (red_shells : ℕ) (green_shells : ℕ) 
  (h_total : total_shells = 291) (h_red : red_shells = 76) (h_green : green_shells = 49) :
  total_shells - (red_shells + green_shells) = 166 :=
by
  sorry

end shells_not_red_or_green_l10_10467


namespace student_A_more_stable_l10_10736

-- Defining the variances of students A and B as constants
def S_A_sq : ℝ := 0.04
def S_B_sq : ℝ := 0.13

-- Statement of the theorem
theorem student_A_more_stable : S_A_sq < S_B_sq → true :=
by
  -- proof will go here
  sorry

end student_A_more_stable_l10_10736


namespace andrew_friends_brought_food_l10_10132

theorem andrew_friends_brought_food (slices_per_friend total_slices : ℕ) (h1 : slices_per_friend = 4) (h2 : total_slices = 16) :
  total_slices / slices_per_friend = 4 :=
by
  sorry

end andrew_friends_brought_food_l10_10132


namespace smallest_value_of_Q_l10_10480

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - 4*x^2 + 2*x - 3

theorem smallest_value_of_Q :
  min (-10) (min 3 (-2)) = -10 :=
by
  -- Skip the proof
  sorry

end smallest_value_of_Q_l10_10480


namespace degree_f_x2_mul_g_x4_l10_10567

open Polynomial

theorem degree_f_x2_mul_g_x4 {f g : Polynomial ℝ} (hf : degree f = 4) (hg : degree g = 5) :
  degree (f.comp (X ^ 2) * g.comp (X ^ 4)) = 28 :=
sorry

end degree_f_x2_mul_g_x4_l10_10567


namespace max_distance_traveled_l10_10577

theorem max_distance_traveled (fare: ℝ) (x: ℝ) :
  fare = 17.2 → 
  x > 3 →
  1.4 * (x - 3) + 6 ≤ fare → 
  x ≤ 11 := by
  sorry

end max_distance_traveled_l10_10577


namespace intersection_point_l10_10058

noncomputable def line1 (x : ℚ) : ℚ := 3 * x
noncomputable def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_point : ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = -1/2 ∧ y = -3/2 :=
by
  -- skipping the actual proof steps
  sorry

end intersection_point_l10_10058


namespace bowling_average_change_l10_10306

theorem bowling_average_change (old_avg : ℝ) (wickets_last : ℕ) (runs_last : ℕ) (wickets_before : ℕ)
  (h_old_avg : old_avg = 12.4)
  (h_wickets_last : wickets_last = 8)
  (h_runs_last : runs_last = 26)
  (h_wickets_before : wickets_before = 175) :
  old_avg - ((old_avg * wickets_before + runs_last)/(wickets_before + wickets_last)) = 0.4 :=
by {
  sorry
}

end bowling_average_change_l10_10306


namespace total_weight_of_rings_l10_10674

-- Conditions
def weight_orange : ℝ := 0.08333333333333333
def weight_purple : ℝ := 0.3333333333333333
def weight_white : ℝ := 0.4166666666666667

-- Goal
theorem total_weight_of_rings : weight_orange + weight_purple + weight_white = 0.8333333333333333 := by
  sorry

end total_weight_of_rings_l10_10674


namespace total_marbles_l10_10118

theorem total_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l10_10118


namespace katrina_cookies_sale_l10_10546

/-- 
Katrina has 120 cookies in the beginning.
She sells 36 cookies in the morning.
She sells 16 cookies in the afternoon.
She has 11 cookies left to take home at the end of the day.
Prove that she sold 57 cookies during the lunch rush.
-/
theorem katrina_cookies_sale :
  let total_cookies := 120
  let morning_sales := 36
  let afternoon_sales := 16
  let cookies_left := 11
  let cookies_sold_lunch_rush := total_cookies - morning_sales - afternoon_sales - cookies_left
  cookies_sold_lunch_rush = 57 :=
by
  sorry

end katrina_cookies_sale_l10_10546


namespace tv_price_reduction_percentage_l10_10016

noncomputable def price_reduction (x : ℝ) : Prop :=
  (1 - x / 100) * 1.80 = 1.44000000000000014

theorem tv_price_reduction_percentage : price_reduction 20 :=
by
  sorry

end tv_price_reduction_percentage_l10_10016


namespace calculate_neg_pow_mul_l10_10473

theorem calculate_neg_pow_mul (a : ℝ) : -a^4 * a^3 = -a^7 := by
  sorry

end calculate_neg_pow_mul_l10_10473


namespace new_class_mean_l10_10386

theorem new_class_mean {X Y : ℕ} {mean_a mean_b : ℚ}
  (hx : X = 30) (hy : Y = 6) 
  (hmean_a : mean_a = 72) (hmean_b : mean_b = 78) :
  (X * mean_a + Y * mean_b) / (X + Y) = 73 := 
by 
  sorry

end new_class_mean_l10_10386


namespace obtuse_triangle_probability_l10_10499

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l10_10499


namespace smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l10_10584

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

end smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l10_10584


namespace probability_meeting_part_a_l10_10548

theorem probability_meeting_part_a :
  ∃ p : ℝ, p = (11 : ℝ) / 36 :=
sorry

end probability_meeting_part_a_l10_10548


namespace average_weight_increase_l10_10969

-- Define the initial conditions as given in the problem
def W_old : ℕ := 53
def W_new : ℕ := 71
def N : ℕ := 10

-- Average weight increase after replacing one oarsman
theorem average_weight_increase : (W_new - W_old : ℝ) / N = 1.8 := by
  sorry

end average_weight_increase_l10_10969


namespace sum_n_k_l10_10972

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l10_10972


namespace correct_operation_l10_10335

variable (a b : ℝ)

theorem correct_operation : (-2 * a ^ 2) ^ 2 = 4 * a ^ 4 := by
  sorry

end correct_operation_l10_10335


namespace altitude_in_scientific_notation_l10_10424

theorem altitude_in_scientific_notation : 
  (389000 : ℝ) = 3.89 * (10 : ℝ) ^ 5 :=
by
  sorry

end altitude_in_scientific_notation_l10_10424


namespace positive_difference_of_solutions_is_14_l10_10722

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 5 * x + 15 = x + 55

-- Define the positive difference between solutions of the quadratic equation
def positive_difference (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem positive_difference_of_solutions_is_14 : 
  ∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ positive_difference a b = 14 :=
by
  sorry

end positive_difference_of_solutions_is_14_l10_10722


namespace central_angle_of_sector_l10_10202

theorem central_angle_of_sector (alpha : ℝ) (l : ℝ) (A : ℝ) (h1 : l = 2 * Real.pi) (h2 : A = 5 * Real.pi) : 
  alpha = 72 :=
by
  sorry

end central_angle_of_sector_l10_10202


namespace rectangular_eq_of_C_slope_of_l_l10_10109

noncomputable section

/-- Parametric equations for curve C -/
def parametric_eq (θ : ℝ) : ℝ × ℝ :=
⟨4 * Real.cos θ, 3 * Real.sin θ⟩

/-- Question 1: Prove that the rectangular coordinate equation of curve C is (x^2)/16 + (y^2)/9 = 1. -/
theorem rectangular_eq_of_C (x y θ : ℝ) (h₁ : x = 4 * Real.cos θ) (h₂ : y = 3 * Real.sin θ) : 
  x^2 / 16 + y^2 / 9 = 1 := 
sorry

/-- Line passing through point M(2, 2) with parametric equations -/
def line_through_M (t α : ℝ) : ℝ × ℝ :=
⟨2 + t * Real.cos α, 2 + t * Real.sin α⟩ 

/-- Question 2: Prove that the slope of line l passing M(2, 2) which intersects curve C at points A and B is -9/16 -/
theorem slope_of_l (t₁ t₂ α : ℝ) (t₁_t₂_sum_zero : (9 * Real.sin α + 36 * Real.cos α) = 0) :
  Real.tan α = -9 / 16 :=
sorry

end rectangular_eq_of_C_slope_of_l_l10_10109


namespace girls_bought_balloons_l10_10053

theorem girls_bought_balloons (initial_balloons boys_bought girls_bought remaining_balloons : ℕ)
  (h1 : initial_balloons = 36)
  (h2 : boys_bought = 3)
  (h3 : remaining_balloons = 21)
  (h4 : initial_balloons - remaining_balloons = boys_bought + girls_bought) :
  girls_bought = 12 := by
  sorry

end girls_bought_balloons_l10_10053


namespace valid_N_count_l10_10639

theorem valid_N_count : 
  (∃ n : ℕ, 0 < n ∧ (49 % (n + 3) = 0) ∧ (49 / (n + 3)) % 2 = 1) → 
  (∃ count : ℕ, count = 2) :=
sorry

end valid_N_count_l10_10639


namespace opposite_exprs_have_value_l10_10586

theorem opposite_exprs_have_value (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) → x = 2 :=
by
  intro h
  sorry

end opposite_exprs_have_value_l10_10586


namespace problem_1_problem_2_l10_10518

-- First problem: Find the solution set for the inequality |x - 1| + |x + 2| ≥ 5
theorem problem_1 (x : ℝ) : (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) :=
sorry

-- Second problem: Find the range of real number a such that |x - a| + |x + 2| ≤ |x + 4| for all x in [0, 1]
theorem problem_2 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x - a| + |x + 2| ≤ |x + 4|) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l10_10518


namespace bill_property_taxes_l10_10343

theorem bill_property_taxes 
  (take_home_salary sales_taxes gross_salary : ℕ)
  (income_tax_rate : ℚ)
  (take_home_salary_eq : take_home_salary = 40000)
  (sales_taxes_eq : sales_taxes = 3000)
  (gross_salary_eq : gross_salary = 50000)
  (income_tax_rate_eq : income_tax_rate = 0.1) :
  let income_taxes := (income_tax_rate * gross_salary) 
  let property_taxes := gross_salary - (income_taxes + sales_taxes + take_home_salary)
  property_taxes = 2000 := by
  sorry

end bill_property_taxes_l10_10343


namespace third_smallest_is_four_probability_l10_10697

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l10_10697


namespace common_ratio_of_geometric_sequence_l10_10542

variables (a : ℕ → ℝ) (q : ℝ)
axiom h1 : a 1 = 2
axiom h2 : ∀ n : ℕ, a (n + 1) - a n ≠ 0 -- Common difference is non-zero
axiom h3 : a 3 = (a 1) * q
axiom h4 : a 11 = (a 1) * q^2
axiom h5 : a 11 = a 1 + 5 * (a 3 - a 1)

theorem common_ratio_of_geometric_sequence : q = 4 := 
by sorry

end common_ratio_of_geometric_sequence_l10_10542


namespace sequence_n_value_l10_10110

theorem sequence_n_value (n : ℤ) : (2 * n^2 - 3 = 125) → (n = 8) := 
by {
    sorry
}

end sequence_n_value_l10_10110


namespace rectangular_prism_sides_multiples_of_5_l10_10432

noncomputable def rectangular_prism_sides_multiples_product_condition 
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) (prod_eq_450 : l * w = 450) : Prop :=
  l ∣ 450 ∧ w ∣ 450

theorem rectangular_prism_sides_multiples_of_5
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) :
  rectangular_prism_sides_multiples_product_condition l w hl hw (by sorry) :=
sorry

end rectangular_prism_sides_multiples_of_5_l10_10432


namespace complement_intersection_l10_10522

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection (hU : ∀ x, x ∈ U) (hA : ∀ x, x ∈ A) (hB : ∀ x, x ∈ B) :
    (U \ A) ∩ (U \ B) = {7, 9} :=
by
  sorry

end complement_intersection_l10_10522


namespace problem_solving_example_l10_10198

theorem problem_solving_example (α β : ℝ) (h1 : α + β = 3) (h2 : α * β = 1) (h3 : α^2 - 3 * α + 1 = 0) (h4 : β^2 - 3 * β + 1 = 0) :
  7 * α^5 + 8 * β^4 = 1448 :=
sorry

end problem_solving_example_l10_10198


namespace cricket_bat_selling_price_l10_10302

theorem cricket_bat_selling_price
    (profit : ℝ)
    (profit_percentage : ℝ)
    (CP : ℝ)
    (SP : ℝ)
    (h_profit : profit = 255)
    (h_profit_percentage : profit_percentage = 42.857142857142854)
    (h_CP : CP = 255 * 100 / 42.857142857142854)
    (h_SP : SP = CP + profit) :
    SP = 850 :=
by
  skip -- This is where the proof would go
  sorry -- Placeholder for the required proof

end cricket_bat_selling_price_l10_10302


namespace find_positive_number_l10_10596

theorem find_positive_number 
  (x : ℝ) (h_pos : x > 0) 
  (h_eq : (2 / 3) * x = (16 / 216) * (1 / x)) : 
  x = 1 / 3 :=
by
  -- This is indicating that we're skipping the actual proof steps
  sorry

end find_positive_number_l10_10596


namespace angle_at_230_is_105_degrees_l10_10283

def degree_between_hands (h m : ℕ) : ℚ :=
  let minute_angle := m * 6 in
  let hour_angle := (h % 12) * 30 + (m / 2) in
  let angle := abs (hour_angle - minute_angle) in
  min angle (360 - angle)

theorem angle_at_230_is_105_degrees :
  degree_between_hands 2 30 = 105 := 
sorry

end angle_at_230_is_105_degrees_l10_10283


namespace frequency_not_equal_probability_l10_10902

theorem frequency_not_equal_probability
  (N : ℕ) -- Total number of trials
  (N1 : ℕ) -- Number of times student A is selected
  (hN : N > 0) -- Ensure the number of trials is positive
  (rand_int_gen : ℕ → ℕ) -- A function generating random integers from 1 to 6
  (h_gen : ∀ n, 1 ≤ rand_int_gen n ∧ rand_int_gen n ≤ 6) -- Generator produces numbers between 1 to 6
: (N1/N : ℚ) ≠ (1/6 : ℚ) := 
sorry

end frequency_not_equal_probability_l10_10902


namespace haley_trees_grown_after_typhoon_l10_10374

def original_trees := 9
def trees_died := 4
def current_trees := 10

theorem haley_trees_grown_after_typhoon (newly_grown_trees : ℕ) :
  (original_trees - trees_died) + newly_grown_trees = current_trees → newly_grown_trees = 5 :=
by
  sorry

end haley_trees_grown_after_typhoon_l10_10374


namespace households_3_houses_proportion_l10_10946

noncomputable def total_households : ℕ := 100000
noncomputable def ordinary_households : ℕ := 99000
noncomputable def high_income_households : ℕ := 1000

noncomputable def sampled_ordinary_households : ℕ := 990
noncomputable def sampled_high_income_households : ℕ := 100

noncomputable def sampled_ordinary_3_houses : ℕ := 40
noncomputable def sampled_high_income_3_houses : ℕ := 80

noncomputable def proportion_3_houses : ℝ := (sampled_ordinary_3_houses / sampled_ordinary_households * ordinary_households + sampled_high_income_3_houses / sampled_high_income_households * high_income_households) / total_households

theorem households_3_houses_proportion : proportion_3_houses = 0.048 := 
by
  sorry

end households_3_houses_proportion_l10_10946


namespace quadruples_solution_l10_10412

noncomputable
def valid_quadruples (x1 x2 x3 x4 : ℝ) : Prop :=
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧ (x3 ≠ 0) ∧ (x4 ≠ 0)

theorem quadruples_solution (x1 x2 x3 x4 : ℝ) :
  valid_quadruples x1 x2 x3 x4 ↔ 
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) := 
by sorry

end quadruples_solution_l10_10412


namespace does_not_pass_through_second_quadrant_l10_10014

def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

theorem does_not_pass_through_second_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ x < 0 ∧ y > 0 :=
sorry

end does_not_pass_through_second_quadrant_l10_10014


namespace point_in_which_quadrant_l10_10529

noncomputable def quadrant_of_point (x y : ℝ) : String :=
if (x > 0) ∧ (y > 0) then
    "First"
else if (x < 0) ∧ (y > 0) then
    "Second"
else if (x < 0) ∧ (y < 0) then
    "Third"
else if (x > 0) ∧ (y < 0) then
    "Fourth"
else
    "On Axis"

theorem point_in_which_quadrant (α : ℝ) (h1 : π / 2 < α) (h2 : α < π) : quadrant_of_point (Real.sin α) (Real.cos α) = "Fourth" :=
by {
    sorry
}

end point_in_which_quadrant_l10_10529


namespace seashells_total_l10_10394

theorem seashells_total (joan_seashells jessica_seashells : ℕ)
  (h_joan : joan_seashells = 6)
  (h_jessica : jessica_seashells = 8) :
  joan_seashells + jessica_seashells = 14 :=
by 
  sorry

end seashells_total_l10_10394


namespace onur_biking_distance_l10_10128

-- Definitions based only on given conditions
def Onur_biking_distance_per_day (O : ℕ) := O
def Hanil_biking_distance_per_day (O : ℕ) := O + 40
def biking_days_per_week := 5
def total_distance_per_week := 2700

-- Mathematically equivalent proof problem
theorem onur_biking_distance (O : ℕ) (cond : 5 * (O + (O + 40)) = 2700) : O = 250 := by
  sorry

end onur_biking_distance_l10_10128


namespace problem1_problem2_l10_10645

open Set

noncomputable def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0}

theorem problem1 (a : ℝ) (h1 : A ⊆ (A ∩ B a)) : (4 / 3 : ℝ) ≤ a ∧ a ≤ 2 :=
sorry

theorem problem2 (a : ℝ) (h2 : A ∩ B a = ∅) : a ≤ (2 / 3 : ℝ) ∨ a ≥ 4 :=
sorry

end problem1_problem2_l10_10645


namespace quadratic_trinomial_has_two_roots_l10_10310

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l10_10310


namespace remainder_equiv_l10_10290

theorem remainder_equiv (x : ℤ) (h : ∃ k : ℤ, x = 95 * k + 31) : ∃ m : ℤ, x = 19 * m + 12 := 
sorry

end remainder_equiv_l10_10290


namespace ben_minimum_test_score_l10_10472

theorem ben_minimum_test_score 
  (scores : List ℕ) 
  (current_avg : ℕ) 
  (desired_increase : ℕ) 
  (lowest_score : ℕ) 
  (required_score : ℕ) 
  (h_scores : scores = [95, 85, 75, 65, 90]) 
  (h_current_avg : current_avg = 82) 
  (h_desired_increase : desired_increase = 5) 
  (h_lowest_score : lowest_score = 65) 
  (h_required_score : required_score = 112) :
  (current_avg + desired_increase) = 87 ∧ 
  (6 * (current_avg + desired_increase)) = 522 ∧ 
  required_score = (522 - (95 + 85 + 75 + 65 + 90)) ∧ 
  (522 - (95 + 85 + 75 + 65 + 90)) > lowest_score :=
by 
  sorry

end ben_minimum_test_score_l10_10472


namespace mass_percentage_of_O_in_CaCO3_l10_10148

-- Assuming the given conditions as definitions
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def formula_CaCO3 : (ℕ × ℝ) := (1, molar_mass_Ca) -- 1 atom of Calcium
def formula_CaCO3_C : (ℕ × ℝ) := (1, molar_mass_C) -- 1 atom of Carbon
def formula_CaCO3_O : (ℕ × ℝ) := (3, molar_mass_O) -- 3 atoms of Oxygen

-- Desired result
def mass_percentage_O_CaCO3 : ℝ := 47.95

-- The theorem statement to be proven
theorem mass_percentage_of_O_in_CaCO3 :
  let molar_mass_CaCO3 := formula_CaCO3.2 + formula_CaCO3_C.2 + (formula_CaCO3_O.1 * formula_CaCO3_O.2)
  let mass_percentage_O := (formula_CaCO3_O.1 * formula_CaCO3_O.2 / molar_mass_CaCO3) * 100
  mass_percentage_O = mass_percentage_O_CaCO3 :=
by
  sorry

end mass_percentage_of_O_in_CaCO3_l10_10148


namespace dabbie_turkey_cost_l10_10478

theorem dabbie_turkey_cost :
  let weight1 := 6
  let weight2 := 9
  let weight3 := 2 * weight2
  let cost_per_kg := 2
  let total_weight := weight1 + weight2 + weight3
  total_weight * cost_per_kg = 66 :=
by
  let weight1 := 6
  let weight2 := 9
  let weight3 := 2 * weight2
  let cost_per_kg := 2
  let total_weight := weight1 + weight2 + weight3
  show total_weight * cost_per_kg = 66
  rw [total_weight, cost_per_kg, 66]
  sorry

end dabbie_turkey_cost_l10_10478


namespace inequality_proof_l10_10842

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

end inequality_proof_l10_10842


namespace length_segment_AB_l10_10127

theorem length_segment_AB (A B : ℝ) (hA : A = -5) (hB : B = 2) : |A - B| = 7 :=
by
  sorry

end length_segment_AB_l10_10127


namespace complement_of_intersection_l10_10406

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end complement_of_intersection_l10_10406


namespace common_ratio_solution_l10_10364

-- Define the problem condition
def geometric_sum_condition (a1 : ℝ) (q : ℝ) : Prop :=
  (a1 * (1 - q^3)) / (1 - q) = 3 * a1

-- Define the theorem we want to prove
theorem common_ratio_solution (a1 : ℝ) (q : ℝ) (h : geometric_sum_condition a1 q) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_solution_l10_10364


namespace seventeen_power_seven_mod_eleven_l10_10253

-- Define the conditions
def mod_condition : Prop := 17 % 11 = 6

-- Define the main goal (to prove the correct answer)
theorem seventeen_power_seven_mod_eleven (h : mod_condition) : (17^7) % 11 = 8 := by
  -- Proof goes here
  sorry

end seventeen_power_seven_mod_eleven_l10_10253


namespace complete_square_eqn_l10_10563

theorem complete_square_eqn (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 → (x + d)^2 = e) → d + e = 5 :=
by
  sorry

end complete_square_eqn_l10_10563


namespace candy_count_l10_10792

theorem candy_count (initial_candy : ℕ) (eaten_candy : ℕ) (received_candy : ℕ) (final_candy : ℕ) :
  initial_candy = 33 → eaten_candy = 17 → received_candy = 19 → final_candy = 35 :=
by
  intros h_initial h_eaten h_received
  sorry

end candy_count_l10_10792


namespace least_positive_integer_for_multiple_of_five_l10_10285

theorem least_positive_integer_for_multiple_of_five (x : ℕ) (h_pos : 0 < x) (h_multiple : (625 + x) % 5 = 0) : x = 5 :=
sorry

end least_positive_integer_for_multiple_of_five_l10_10285


namespace convex_polyhedron_space_diagonals_l10_10301

theorem convex_polyhedron_space_diagonals
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)
  (triangular_faces : ℕ)
  (hexagonal_faces : ℕ)
  (total_faces : faces = triangular_faces + hexagonal_faces)
  (vertices_eq : vertices = 30)
  (edges_eq : edges = 72)
  (triangular_faces_eq : triangular_faces = 32)
  (hexagonal_faces_eq : hexagonal_faces = 12)
  (faces_eq : faces = 44) :
  ((vertices * (vertices - 1)) / 2) - edges - 
  (triangular_faces * 0 + hexagonal_faces * ((6 * (6 - 3)) / 2)) = 255 := by
sorry

end convex_polyhedron_space_diagonals_l10_10301


namespace smallest_possible_AC_l10_10182

-- Constants and assumptions
variables (AC CD : ℕ)
def BD_squared : ℕ := 68

-- Prime number constraint for CD
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Given facts
axiom eq_ab_ac (AB : ℕ) : AB = AC
axiom perp_bd_ac (BD AC : ℕ) : BD^2 = BD_squared
axiom int_ac_cd : AC = (CD^2 + BD_squared) / (2 * CD)

theorem smallest_possible_AC :
  ∃ AC : ℕ, (∃ CD : ℕ, is_prime CD ∧ CD < 10 ∧ AC = (CD^2 + BD_squared) / (2 * CD)) ∧ AC = 18 :=
by
  sorry

end smallest_possible_AC_l10_10182


namespace Will_Had_28_Bottles_l10_10588

-- Definitions based on conditions
-- Let days be the number of days water lasted (4 days)
def days : ℕ := 4

-- Let bottles_per_day be the number of bottles Will drank each day (7 bottles/day)
def bottles_per_day : ℕ := 7

-- Correct answer defined as total number of bottles (28 bottles)
def total_bottles : ℕ := 28

-- The proof statement to show that the total number of bottles is equal to 28
theorem Will_Had_28_Bottles :
  (bottles_per_day * days = total_bottles) :=
by
  sorry

end Will_Had_28_Bottles_l10_10588


namespace seller_loss_l10_10712

/--
Given:
1. The buyer took goods worth 10 rubles (v_goods : Real := 10).
2. The buyer gave 25 rubles (payment : Real := 25).
3. The seller exchanged 25 rubles of genuine currency with the neighbor (exchange : Real := 25).
4. The seller received 25 rubles in counterfeit currency from the neighbor (counterfeit : Real := 25).
5. The seller gave 15 rubles in genuine currency as change (change : Real := 15).
6. The neighbor discovered the counterfeit and the seller returned 25 rubles to the neighbor (returned : Real := 25).

Prove that the net loss incurred by the seller is 30 rubles.
-/
theorem seller_loss :
  let v_goods := 10
  let payment := 25
  let exchange := 25
  let counterfeit := 25
  let change := 15
  let returned := 25
  (exchange + change) - v_goods = 30 :=
by
  sorry

end seller_loss_l10_10712


namespace B_alone_completion_l10_10459

-- Define the conditions:
def A_efficiency_rel_to_B (A B: ℕ → Prop) : Prop :=
  ∀ (x: ℕ), B x → A (2 * x)

def together_job_completion (A B: ℕ → Prop) : Prop :=
  ∀ (t: ℕ), t = 20 → (∃ (x y : ℕ), B x ∧ A y ∧ (1/x + 1/y = 1/t))

-- Define the theorem:
theorem B_alone_completion (A B: ℕ → Prop) (h1 : A_efficiency_rel_to_B A B) (h2 : together_job_completion A B) :
  ∃ (x: ℕ), B x ∧ x = 30 :=
sorry

end B_alone_completion_l10_10459


namespace initial_amount_l10_10885

theorem initial_amount (x : ℝ) (h1 : x = (2*x - 10) / 2) (h2 : x = (4*x - 30) / 2) (h3 : 8*x - 70 = 0) : x = 8.75 :=
by
  sorry

end initial_amount_l10_10885


namespace positive_integer_solution_eq_l10_10916

theorem positive_integer_solution_eq :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (xyz + 2 * x + 3 * y + 6 * z = xy + 2 * xz + 3 * yz) ∧ (x, y, z) = (4, 3, 1) := 
by
  sorry

end positive_integer_solution_eq_l10_10916


namespace fewest_posts_l10_10331

def grazingAreaPosts (length width post_interval rock_wall_length : ℕ) : ℕ :=
  let side1 := width / post_interval + 1
  let side2 := length / post_interval
  side1 + 2 * side2

theorem fewest_posts (length width post_interval rock_wall_length posts : ℕ) :
  length = 70 ∧ width = 50 ∧ post_interval = 10 ∧ rock_wall_length = 150 ∧ posts = 18 →
  grazingAreaPosts length width post_interval rock_wall_length = posts := 
by
  intros h
  obtain ⟨hl, hw, hp, hr, ht⟩ := h
  simp [grazingAreaPosts, hl, hw, hp, hr]
  sorry

end fewest_posts_l10_10331


namespace fraction_is_three_halves_l10_10379

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end fraction_is_three_halves_l10_10379


namespace range_of_a_l10_10082

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l10_10082


namespace peasant_initial_money_l10_10761

theorem peasant_initial_money :
  ∃ (x1 x2 x3 : ℕ), 
    (x1 / 2 + 1 = x2) ∧ 
    (x2 / 2 + 2 = x3) ∧ 
    (x3 / 2 + 1 = 0) ∧ 
    x1 = 18 := 
by
  sorry

end peasant_initial_money_l10_10761


namespace A_investment_l10_10615

theorem A_investment (B_invest C_invest Total_profit A_share : ℝ) 
  (h1 : B_invest = 4200)
  (h2 : C_invest = 10500)
  (h3 : Total_profit = 12100)
  (h4 : A_share = 3630) 
  (h5 : ∀ {x : ℝ}, A_share / Total_profit = x / (x + B_invest + C_invest)) :
  ∃ A_invest : ℝ, A_invest = 6300 :=
by sorry

end A_investment_l10_10615


namespace third_smallest_is_four_probability_l10_10698

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l10_10698


namespace complex_division_example_l10_10512

-- Given conditions
def i : ℂ := Complex.I

-- The statement we need to prove
theorem complex_division_example : (1 + 3 * i) / (1 + i) = 2 + i :=
by
  sorry

end complex_division_example_l10_10512


namespace largest_common_number_in_arithmetic_sequences_l10_10337

theorem largest_common_number_in_arithmetic_sequences (x : ℕ)
  (h1 : x ≡ 2 [MOD 8])
  (h2 : x ≡ 5 [MOD 9])
  (h3 : x < 200) : x = 194 :=
by sorry

end largest_common_number_in_arithmetic_sequences_l10_10337


namespace weight_triangle_correct_weight_l10_10766

noncomputable def area_square (side : ℝ) : ℝ := side ^ 2

noncomputable def area_triangle (side : ℝ) : ℝ := (side ^ 2 * Real.sqrt 3) / 4

noncomputable def weight (area : ℝ) (density : ℝ) := area * density

noncomputable def weight_equilateral_triangle (weight_square : ℝ) (side_square : ℝ) (side_triangle : ℝ) : ℝ :=
  let area_s := area_square side_square
  let area_t := area_triangle side_triangle
  let density := weight_square / area_s
  weight area_t density

theorem weight_triangle_correct_weight :
  weight_equilateral_triangle 8 4 6 = 9 * Real.sqrt 3 / 2 := by sorry

end weight_triangle_correct_weight_l10_10766


namespace intersection_A_B_l10_10076

def set_A (x : ℝ) : Prop := 2 * x^2 + 5 * x - 3 ≤ 0

def set_B (x : ℝ) : Prop := -2 < x

theorem intersection_A_B :
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -2 < x ∧ x ≤ 1/2} := 
by {
  sorry
}

end intersection_A_B_l10_10076


namespace math_vs_english_time_difference_l10_10397

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l10_10397


namespace valid_permutations_count_l10_10830

def num_permutations (seq : List ℕ) : ℕ :=
  -- A dummy implementation, the real function would calculate the number of valid permutations.
  sorry

theorem valid_permutations_count : num_permutations [1, 2, 3, 4, 5, 6] = 32 :=
by
  sorry

end valid_permutations_count_l10_10830


namespace number_of_associates_l10_10608

theorem number_of_associates
  (num_managers : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ)
  (total_employees : ℕ := num_managers + A) -- Adding a placeholder A for the associates
  (total_salary_company : ℝ := (num_managers * avg_salary_managers) + (A * avg_salary_associates)) 
  (average_calculation : avg_salary_company = total_salary_company / total_employees) :
  ∃ A : ℕ, A = 75 :=
by
  let A : ℕ := 75
  sorry

end number_of_associates_l10_10608


namespace false_proposition_C_l10_10192

variable (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x^2 + b * x + c

theorem false_proposition_C 
  (ha : a > 0)
  (x0 : ℝ)
  (hx0 : x0 = -b / (2 * a)) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 :=
by
  sorry

end false_proposition_C_l10_10192


namespace convert_binary_to_decimal_l10_10183

theorem convert_binary_to_decimal : (1 * 2^2 + 1 * 2^1 + 1 * 2^0) = 7 := by
  sorry

end convert_binary_to_decimal_l10_10183


namespace diameter_of_lid_is_2_inches_l10_10629

noncomputable def π : ℝ := 3.14
def C : ℝ := 6.28

theorem diameter_of_lid_is_2_inches (d : ℝ) : d = C / π → d = 2 :=
by
  intro h
  sorry

end diameter_of_lid_is_2_inches_l10_10629


namespace other_solution_of_quadratic_l10_10647

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop :=
  65 * x^2 - 104 * x + 31 = 0

-- Main theorem statement
theorem other_solution_of_quadratic :
  quadratic_eq (6 / 5) → quadratic_eq (5 / 13) :=
by
  intro h
  sorry

end other_solution_of_quadratic_l10_10647


namespace remainder_2519_div_6_l10_10166

theorem remainder_2519_div_6 : ∃ q r, 2519 = 6 * q + r ∧ 0 ≤ r ∧ r < 6 ∧ r = 5 := 
by
  sorry

end remainder_2519_div_6_l10_10166


namespace max_f_value_l10_10357

open Real

noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

theorem max_f_value : ∃ (x₀ y₀ : ℝ), (0 < x₀) ∧ (0 < y₀) ∧ (∀ (x y : ℝ), (0 < x) → (0 < y) → f x y ≤ f x₀ y₀) ∧ f x₀ y₀ = 1 / sqrt 2 :=
by 
  sorry

end max_f_value_l10_10357


namespace max_gcd_14m_plus_4_9m_plus_2_l10_10910

theorem max_gcd_14m_plus_4_9m_plus_2 (m : ℕ) (h : m > 0) : ∃ M, M = 8 ∧ ∀ k, gcd (14 * m + 4) (9 * m + 2) = k → k ≤ M :=
by
  sorry

end max_gcd_14m_plus_4_9m_plus_2_l10_10910


namespace probability_no_obtuse_triangle_correct_l10_10494

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l10_10494


namespace minimum_balls_to_draw_l10_10667

-- Defining the sizes for the different colors of balls
def red_balls : Nat := 40
def green_balls : Nat := 25
def yellow_balls : Nat := 20
def blue_balls : Nat := 15
def purple_balls : Nat := 10
def orange_balls : Nat := 5

-- Given conditions
def max_red_balls_before_18 : Nat := 17
def max_green_balls_before_18 : Nat := 17
def max_yellow_balls_before_18 : Nat := 17
def max_blue_balls_before_18 : Nat := 15
def max_purple_balls_before_18 : Nat := 10
def max_orange_balls_before_18 : Nat := 5

-- Sum of maximum balls of each color that can be drawn without ensuring 18 of any color
def max_balls_without_18 : Nat := 
  max_red_balls_before_18 + 
  max_green_balls_before_18 + 
  max_yellow_balls_before_18 + 
  max_blue_balls_before_18 + 
  max_purple_balls_before_18 + 
  max_orange_balls_before_18

theorem minimum_balls_to_draw {n : Nat} (h : n = max_balls_without_18 + 1) :
  n = 82 := by
  sorry

end minimum_balls_to_draw_l10_10667


namespace inequality_product_lt_zero_l10_10935

theorem inequality_product_lt_zero (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 :=
  sorry

end inequality_product_lt_zero_l10_10935


namespace rectangle_new_area_l10_10862

theorem rectangle_new_area (original_area : ℝ) (new_length_factor : ℝ) (new_width_factor : ℝ) 
  (h1 : original_area = 560) (h2 : new_length_factor = 1.2) (h3 : new_width_factor = 0.85) : 
  new_length_factor * new_width_factor * original_area = 571 := 
by 
  sorry

end rectangle_new_area_l10_10862


namespace find_A_l10_10661

theorem find_A (A : ℕ) (h1 : A < 5) (h2 : (9 * 100 + A * 10 + 7) / 10 * 10 = 930) : A = 3 :=
sorry

end find_A_l10_10661


namespace regression_is_appropriate_l10_10992

-- Definitions for the different analysis methods
inductive AnalysisMethod
| ResidualAnalysis : AnalysisMethod
| RegressionAnalysis : AnalysisMethod
| IsoplethBarChart : AnalysisMethod
| IndependenceTest : AnalysisMethod

-- Relating height and weight with an appropriate analysis method
def appropriateMethod (method : AnalysisMethod) : Prop :=
  method = AnalysisMethod.RegressionAnalysis

-- Stating the theorem that regression analysis is the appropriate method
theorem regression_is_appropriate : appropriateMethod AnalysisMethod.RegressionAnalysis :=
by sorry

end regression_is_appropriate_l10_10992


namespace largest_multiple_5_6_lt_1000_is_990_l10_10879

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end largest_multiple_5_6_lt_1000_is_990_l10_10879


namespace dice_five_prob_l10_10881

-- Define a standard six-sided die probability
def prob_five : ℚ := 1 / 6

-- Define the probability of all four dice showing five
def prob_all_five : ℚ := prob_five * prob_five * prob_five * prob_five

-- State the theorem
theorem dice_five_prob : prob_all_five = 1 / 1296 := by
  sorry

end dice_five_prob_l10_10881


namespace part1_solution_part2_solution_l10_10855

def f (x : ℝ) (a : ℝ) := |x + 1| - |a * x - 1|

-- Statement for part 1
theorem part1_solution (x : ℝ) : (f x 1 > 1) ↔ (x > 1 / 2) := sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (f x a > x) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_solution_part2_solution_l10_10855


namespace pizza_slices_per_pizza_l10_10437

theorem pizza_slices_per_pizza (num_coworkers slices_per_person num_pizzas : ℕ) (h1 : num_coworkers = 12) (h2 : slices_per_person = 2) (h3 : num_pizzas = 3) :
  (num_coworkers * slices_per_person) / num_pizzas = 8 :=
by
  sorry

end pizza_slices_per_pizza_l10_10437


namespace traveler_arrangements_l10_10131

theorem traveler_arrangements :
  let travelers := 6
  let rooms := 3
  ∃ (arrangements : Nat), arrangements = 240 := by
  sorry

end traveler_arrangements_l10_10131


namespace find_ab_l10_10294

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry -- Proof to be provided

end find_ab_l10_10294


namespace polynomial_coeff_sum_l10_10505

theorem polynomial_coeff_sum (a0 a1 a2 a3 : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a3 * x^3 + a2 * x^2 + a1 * x + a0) →
  a0 + a1 + a2 + a3 = 27 :=
by
  sorry

end polynomial_coeff_sum_l10_10505


namespace nancy_more_money_l10_10122

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end nancy_more_money_l10_10122


namespace inverse_value_l10_10381

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 2 * x)

-- Define the goal of the proof
theorem inverse_value {g : ℝ → ℝ}
  (h : ∀ y, g (g⁻¹ y) = y) :
  ((g⁻¹ 5)⁻¹) = -1 :=
by
  sorry

end inverse_value_l10_10381


namespace volume_after_increase_l10_10327

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l10_10327


namespace solve_inequality_l10_10707

theorem solve_inequality {x : ℝ} :
  (3 / (5 - 3 * x) > 1) ↔ (2/3 < x ∧ x < 5/3) :=
by
  sorry

end solve_inequality_l10_10707


namespace weight_jordan_after_exercise_l10_10834

def initial_weight : ℕ := 250
def first_4_weeks_loss : ℕ := 3 * 4
def next_8_weeks_loss : ℕ := 2 * 8
def total_weight_loss : ℕ := first_4_weeks_loss + next_8_weeks_loss
def final_weight : ℕ := initial_weight - total_weight_loss

theorem weight_jordan_after_exercise : final_weight = 222 :=
by 
  sorry

end weight_jordan_after_exercise_l10_10834


namespace jean_pairs_of_pants_l10_10393

theorem jean_pairs_of_pants
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (number_of_pairs : ℝ)
  (h1 : retail_price = 45)
  (h2 : discount_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : total_paid = 396)
  (h5 : number_of_pairs = total_paid / ((retail_price * (1 - discount_rate)) * (1 + tax_rate))) :
  number_of_pairs = 10 :=
by
  sorry

end jean_pairs_of_pants_l10_10393


namespace cheryl_material_used_l10_10448

noncomputable def total_material_needed : ℚ :=
  (5 / 11) + (2 / 3)

noncomputable def material_left : ℚ :=
  25 / 55

noncomputable def material_used : ℚ :=
  total_material_needed - material_left

theorem cheryl_material_used :
  material_used = 22 / 33 :=
by
  sorry

end cheryl_material_used_l10_10448


namespace average_after_31st_inning_l10_10299

-- Define the conditions as Lean definitions
def initial_average (A : ℝ) := A

def total_runs_before_31st_inning (A : ℝ) := 30 * A

def score_in_31st_inning := 105

def new_average (A : ℝ) := A + 3

def total_runs_after_31st_inning (A : ℝ) := total_runs_before_31st_inning A + score_in_31st_inning

-- Define the statement to prove the batsman's average after the 31st inning is 15
theorem average_after_31st_inning (A : ℝ) : total_runs_after_31st_inning A = 31 * (new_average A) → new_average A = 15 := by
  sorry

end average_after_31st_inning_l10_10299


namespace remainder_T_2015_mod_10_l10_10068

-- Define the number of sequences with no more than two consecutive identical letters
noncomputable def T : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 6
| n + 1 => (T n + T (n - 1) + T (n - 2) + T (n - 3))  -- hypothetically following initial conditions pattern

theorem remainder_T_2015_mod_10 : T 2015 % 10 = 6 :=
by 
  sorry

end remainder_T_2015_mod_10_l10_10068


namespace patrick_purchased_pencils_l10_10849

theorem patrick_purchased_pencils (c s : ℝ) : 
  (∀ n : ℝ, n * c = 1.375 * n * s ∧ (n * c - n * s = 30 * s) → n = 80) :=
by sorry

end patrick_purchased_pencils_l10_10849


namespace total_number_of_cards_l10_10953

theorem total_number_of_cards (groups : ℕ) (cards_per_group : ℕ) (h_groups : groups = 9) (h_cards_per_group : cards_per_group = 8) : groups * cards_per_group = 72 := by
  sorry

end total_number_of_cards_l10_10953


namespace part1_A_intersect_B_l10_10805

def setA : Set ℝ := { x | x ^ 2 - 2 * x - 3 ≤ 0 }
def setB (m : ℝ) : Set ℝ := { x | (x - (m - 1)) * (x - (m + 1)) > 0 }

theorem part1_A_intersect_B (m : ℝ) (h : m = 0) : 
  setA ∩ setB m = { x | 1 < x ∧ x ≤ 3 } :=
sorry

end part1_A_intersect_B_l10_10805


namespace nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l10_10175

theorem nat_forms_6n_plus_1_or_5 (x : ℕ) (h1 : ¬ (x % 2 = 0) ∧ ¬ (x % 3 = 0)) :
  ∃ n : ℕ, x = 6 * n + 1 ∨ x = 6 * n + 5 := 
sorry

theorem prod_6n_plus_1 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 1) = 6 * (6 * m * n + m + n) + 1 :=
sorry

theorem prod_6n_plus_5 (m n : ℕ) :
  (6 * m + 5) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + 5 * n + 4) + 1 :=
sorry

theorem prod_6n_plus_1_and_5 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + n) + 5 :=
sorry

end nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l10_10175


namespace union_A_B_l10_10932

-- Define them as sets
def A : Set ℝ := {x | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 3}

-- Statement of the theorem
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l10_10932


namespace inclination_angle_x_eq_one_l10_10943

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end inclination_angle_x_eq_one_l10_10943


namespace cos_value_l10_10508

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 4 - α) = 1 / 3) :
  Real.cos (Real.pi / 4 + α) = 1 / 3 :=
sorry

end cos_value_l10_10508


namespace third_derivative_correct_l10_10637

noncomputable def func (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_correct :
  (deriv^[3] func) x = (4 / (1 + x^2)^2) :=
sorry

end third_derivative_correct_l10_10637


namespace tan_sum_identity_l10_10806

-- Definitions
def quadratic_eq (x : ℝ) : Prop := 6 * x^2 - 5 * x + 1 = 0
def tan_roots (α β : ℝ) : Prop := quadratic_eq (Real.tan α) ∧ quadratic_eq (Real.tan β)

-- Problem statement
theorem tan_sum_identity (α β : ℝ) (hαβ : tan_roots α β) : Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l10_10806


namespace geometric_series_sum_l10_10713

theorem geometric_series_sum :
  2016 * (1 / (1 + (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32))) = 1024 :=
by
  sorry

end geometric_series_sum_l10_10713


namespace square_area_l10_10687

theorem square_area
  (E_on_AD : ∃ E : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ E = (0, s))
  (F_on_extension_BC : ∃ F : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ F = (s, 0))
  (BE_20 : ∃ B E : ℝ × ℝ, ∃ s : ℝ, B = (s, 0) ∧ E = (0, s) ∧ dist B E = 20)
  (EF_25 : ∃ E F : ℝ × ℝ, ∃ s : ℝ, E = (0, s) ∧ F = (s, 0) ∧ dist E F = 25)
  (FD_20 : ∃ F D : ℝ × ℝ, ∃ s : ℝ, F = (s, 0) ∧ D = (s, s) ∧ dist F D = 20) :
  ∃ s : ℝ, s > 0 ∧ s^2 = 400 :=
by
  -- Hypotheses are laid out in conditions as defined above
  sorry

end square_area_l10_10687


namespace distinct_x_intercepts_l10_10094

theorem distinct_x_intercepts : 
  let f (x : ℝ) := ((x - 8) * (x^2 + 4*x + 3))
  (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
by
  sorry

end distinct_x_intercepts_l10_10094


namespace wax_he_has_l10_10846

def total_wax : ℕ := 353
def additional_wax : ℕ := 22

theorem wax_he_has : total_wax - additional_wax = 331 := by
  sorry

end wax_he_has_l10_10846


namespace heidi_more_nail_polishes_l10_10547

theorem heidi_more_nail_polishes :
  ∀ (k h r : ℕ), 
    k = 12 ->
    r = k - 4 ->
    h + r = 25 ->
    h - k = 5 :=
by
  intros k h r hk hr hr_sum
  sorry

end heidi_more_nail_polishes_l10_10547


namespace number_multiplied_by_9_l10_10042

theorem number_multiplied_by_9 (x : ℕ) (h : 50 = x + 26) : 9 * x = 216 := by
  sorry

end number_multiplied_by_9_l10_10042


namespace an_general_formula_Tn_formula_l10_10370

open Nat
open BigOperators

-- Given conditions 
def Sn (n : ℕ) : ℕ := (n * n + n) / 2
def an (n : ℕ) : ℕ := if n = 1 then 1 else (Sn n - Sn (n - 1))
def bn (n : ℕ) : ℕ := an n * 2 ^ an (2 * n)

-- Lean statement for part 1
theorem an_general_formula (n : ℕ) : an n = n :=
by sorry

-- Lean statement for part 2
theorem Tn_formula (n : ℕ) : 
  (∑ k in Finset.range n, bn (k + 1)) = ((n / 3) - (1 / 9)) * 4^(n + 1) + (4 / 9) :=
by sorry

end an_general_formula_Tn_formula_l10_10370


namespace machine_does_not_print_13824_l10_10787

-- Definitions corresponding to the conditions:
def machine_property (S : Set ℕ) : Prop :=
  ∀ n ∈ S, (2 * n) ∉ S ∧ (3 * n) ∉ S

def machine_prints_2 (S : Set ℕ) : Prop :=
  2 ∈ S

-- Statement to be proved
theorem machine_does_not_print_13824 (S : Set ℕ) 
  (H1 : machine_property S) 
  (H2 : machine_prints_2 S) : 
  13824 ∉ S :=
sorry

end machine_does_not_print_13824_l10_10787


namespace abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l10_10230

theorem abs_x_minus_one_sufficient_but_not_necessary_for_quadratic (x : ℝ) :
  (|x - 1| < 2) → (x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l10_10230


namespace sin_cos_special_l10_10793

def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem sin_cos_special (x : ℝ) : 
  special_operation (Real.sin (x / 12)) (Real.cos (x / 12)) = -(1 + 2 * Real.sqrt 3) / 4 :=
  sorry

end sin_cos_special_l10_10793


namespace volume_ratio_l10_10468

theorem volume_ratio (A B C : ℚ) (h1 : (3/4) * A = (2/3) * B) (h2 : (2/3) * B = (1/2) * C) :
  A / C = 2 / 3 :=
sorry

end volume_ratio_l10_10468


namespace additional_time_due_to_leak_l10_10093

theorem additional_time_due_to_leak (fill_time_no_leak: ℝ) (leak_empty_time: ℝ) (fill_rate_no_leak: fill_time_no_leak ≠ 0):
  (fill_time_no_leak = 3) → 
  (leak_empty_time = 12) → 
  (1 / fill_time_no_leak - 1 / leak_empty_time ≠ 0) → 
  ((1 / fill_time_no_leak - 1 / leak_empty_time) / (1 / (1 / fill_time_no_leak - 1 / leak_empty_time)) - fill_time_no_leak = 1) := 
by
  intro h_fill h_leak h_effective_rate
  sorry

end additional_time_due_to_leak_l10_10093


namespace birgit_time_to_travel_8km_l10_10856

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end birgit_time_to_travel_8km_l10_10856


namespace number_of_ways_two_girls_together_l10_10223

theorem number_of_ways_two_girls_together
  (boys girls : ℕ)
  (total_people : ℕ)
  (ways : ℕ) :
  boys = 3 →
  girls = 3 →
  total_people = boys + girls →
  ways = 432 :=
by
  intros
  sorry

end number_of_ways_two_girls_together_l10_10223


namespace eggs_per_hen_l10_10028

theorem eggs_per_hen (total_chickens : ℕ) (num_roosters : ℕ) (non_laying_hens : ℕ) (total_eggs : ℕ) :
  total_chickens = 440 →
  num_roosters = 39 →
  non_laying_hens = 15 →
  total_eggs = 1158 →
  (total_eggs / (total_chickens - num_roosters - non_laying_hens) = 3) :=
by
  intros
  sorry

end eggs_per_hen_l10_10028


namespace joan_exam_time_difference_l10_10399

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l10_10399


namespace simple_interest_rate_l10_10036

theorem simple_interest_rate (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : SI = P / 5)
  (h2 : SI = P * R * T / 100)
  (h3 : T = 7) : 
  R = 20 / 7 :=
by 
  sorry

end simple_interest_rate_l10_10036


namespace product_of_distances_equal_l10_10589

theorem product_of_distances_equal
  (A O B P Q P' Q' : Point)
  (hAOB : angle O A B)
  (hP_on_Perp_to_OA : distance P OA = P')
  (hQ_on_Perp_to_OB : distance Q OB = Q')
  (hEqual_Angles : ∀ O' M N, angle M O A = angle N O B)
  (h_on_rays: ∀ M N, point_on_ray O M P ∧ point_on_ray O N Q)
  :
  (distance P' OA) * (distance Q' OB) = (distance P' Q' O).collab :=
by
  -- Establish intermediate steps using properties of similar triangles
  sorry

end product_of_distances_equal_l10_10589


namespace gcd_of_four_sum_1105_l10_10795

theorem gcd_of_four_sum_1105 (a b c d : ℕ) (h_sum : a + b + c + d = 1105)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_neq_ab : a ≠ b) (h_neq_ac : a ≠ c) (h_neq_ad : a ≠ d)
  (h_neq_bc : b ≠ c) (h_neq_bd : b ≠ d) (h_neq_cd : c ≠ d)
  (h_gcd_ab : gcd a b > 1) (h_gcd_ac : gcd a c > 1) (h_gcd_ad : gcd a d > 1)
  (h_gcd_bc : gcd b c > 1) (h_gcd_bd : gcd b d > 1) (h_gcd_cd : gcd c d > 1) :
  gcd a (gcd b (gcd c d)) = 221 := by
  sorry

end gcd_of_four_sum_1105_l10_10795


namespace integer_values_of_n_satisfy_inequality_l10_10656

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end integer_values_of_n_satisfy_inequality_l10_10656


namespace arc_length_of_polar_curve_l10_10780

noncomputable def arc_length (f : ℝ → ℝ) (df : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt ((f x)^2 + (df x)^2)

theorem arc_length_of_polar_curve :
  arc_length (λ φ => 3 * (1 + Real.sin φ)) (λ φ => 3 * Real.cos φ) (-Real.pi / 6) 0 = 
  6 * (Real.sqrt 3 - Real.sqrt 2) :=
by
  sorry -- Proof goes here

end arc_length_of_polar_curve_l10_10780


namespace factorize_expr1_factorize_expr2_l10_10349

-- Define the expressions
def expr1 (m x y : ℝ) : ℝ := 3 * m * x - 6 * m * y
def expr2 (x : ℝ) : ℝ := 1 - 25 * x^2

-- Define the factorized forms
def factorized_expr1 (m x y : ℝ) : ℝ := 3 * m * (x - 2 * y)
def factorized_expr2 (x : ℝ) : ℝ := (1 + 5 * x) * (1 - 5 * x)

-- Proof problems
theorem factorize_expr1 (m x y : ℝ) : expr1 m x y = factorized_expr1 m x y := sorry
theorem factorize_expr2 (x : ℝ) : expr2 x = factorized_expr2 x := sorry

end factorize_expr1_factorize_expr2_l10_10349


namespace B_div_A_75_l10_10717

noncomputable def find_ratio (A B : ℝ) (x : ℝ) :=
  (A / (x + 3) + B / (x * (x - 9)) = (x^2 - 3*x + 15) / (x * (x + 3) * (x - 9)))

theorem B_div_A_75 :
  ∀ (A B : ℝ), (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → find_ratio A B x) → 
  B/A = 7.5 :=
by
  sorry

end B_div_A_75_l10_10717


namespace tenth_day_of_month_is_monday_l10_10664

theorem tenth_day_of_month_is_monday (Sundays_on_even_dates : ℕ → Prop)
  (h1: Sundays_on_even_dates 2)
  (h2: Sundays_on_even_dates 16)
  (h3: Sundays_on_even_dates 30) :
  ∃ k : ℕ, 10 = k + 2 + 7 * 1 ∧ k.succ.succ.succ.succ.succ.succ.succ.succ.succ.succ = 1 :=
by sorry

end tenth_day_of_month_is_monday_l10_10664


namespace arthur_spent_on_second_day_l10_10770

variable (H D : ℝ)
variable (a1 : 3 * H + 4 * D = 10)
variable (a2 : D = 1)

theorem arthur_spent_on_second_day :
  2 * H + 3 * D = 7 :=
by
  sorry

end arthur_spent_on_second_day_l10_10770


namespace P_xi_gt_30_l10_10413

noncomputable def letter_weight : ℝ → ℂ := sorry -- ξ (weight of a letter in grams)

axiom P_xi_lt_10 : letter_weight(ξ) < 10 = 0.3
axiom P_10_leq_xi_leq_30 : 10 ≤ letter_weight(ξ) ≤ 30 = 0.4

theorem P_xi_gt_30 : letter_weight(ξ) > 30 = 0.3 :=
by 
  have h1 : P(letter_weight(ξ) < 10) + P(10 ≤ letter_weight(ξ) ≤ 30) + P(letter_weight(ξ) > 30) = 1 := sorry
  rw [P_xi_lt_10, P_10_leq_xi_leq_30] at h1
  sorry

end P_xi_gt_30_l10_10413


namespace prob_five_coins_heads_or_one_tail_l10_10926

theorem prob_five_coins_heads_or_one_tail : 
  (∃ (H T : ℚ), H = 1/32 ∧ T = 31/32 ∧ H + T = 1) ↔ 1 = 1 :=
by sorry

end prob_five_coins_heads_or_one_tail_l10_10926


namespace whipped_cream_needed_l10_10633

/- Problem conditions -/
def pies_per_day : ℕ := 3
def days : ℕ := 11
def pies_total : ℕ := pies_per_day * days
def pies_eaten_by_tiffany : ℕ := 4
def pies_remaining : ℕ := pies_total - pies_eaten_by_tiffany
def whipped_cream_per_pie : ℕ := 2

/- Proof statement -/
theorem whipped_cream_needed : whipped_cream_per_pie * pies_remaining = 58 := by
  sorry

end whipped_cream_needed_l10_10633


namespace first_car_made_earlier_l10_10871

def year_first_car : ℕ := 1970
def year_third_car : ℕ := 2000
def diff_third_second : ℕ := 20

theorem first_car_made_earlier : (year_third_car - diff_third_second) - year_first_car = 10 := by
  sorry

end first_car_made_earlier_l10_10871


namespace general_form_of_line_l10_10048

theorem general_form_of_line (x y : ℝ) 
  (passes_through_A : ∃ y, 2 = y)          -- Condition 1: passes through A(-2, 2)
  (same_y_intercept : ∃ y, 6 = y)          -- Condition 2: same y-intercept as y = x + 6
  : 2 * x - y + 6 = 0 := 
sorry

end general_form_of_line_l10_10048


namespace mean_equality_l10_10356

theorem mean_equality (x : ℤ) (h : (8 + 10 + 24) / 3 = (16 + x + 18) / 3) : x = 8 := by 
sorry

end mean_equality_l10_10356


namespace exists_idempotent_l10_10453

-- Definition of the set M as the natural numbers from 1 to 1993
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1993 }

-- Operation * on M
noncomputable def star (a b : ℕ) : ℕ := sorry

-- Hypothesis: * is closed on M and (a * b) * a = b for any a, b in M
axiom star_closed (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star a b ∈ M
axiom star_property (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star (star a b) a = b

-- Goal: Prove that there exists a number a in M such that a * a = a
theorem exists_idempotent : ∃ a ∈ M, star a a = a := by
  sorry

end exists_idempotent_l10_10453


namespace _l10_10305

noncomputable def gear_speeds_relationship (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ) 
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : Prop :=
  ω₁ = (2 * z / x) * ω₃ ∧ ω₂ = (4 * z / (3 * y)) * ω₃

-- Example theorem statement
example (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ)
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : gear_speeds_relationship x y z ω₁ ω₂ ω₃ h1 h2 :=
by sorry

end _l10_10305


namespace break_even_production_volume_l10_10759

theorem break_even_production_volume :
  ∃ Q : ℝ, 300 = 100 + 100000 / Q ∧ Q = 500 :=
by
  use 500
  sorry

end break_even_production_volume_l10_10759


namespace graph_of_equation_is_two_intersecting_lines_l10_10479

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ (x y : ℝ), (x - y)^2 = 3 * x^2 - y^2 ↔ 
  (x = (1 - Real.sqrt 5) / 2 * y) ∨ (x = (1 + Real.sqrt 5) / 2 * y) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l10_10479


namespace solve_equation_l10_10561

theorem solve_equation (x y : ℕ) (h_xy : x ≠ y) : x = 2 ∧ y = 4 ∨ x = 4 ∧ y = 2 :=
by {
  sorry -- Proof skipped
}

end solve_equation_l10_10561


namespace find_t_l10_10686

theorem find_t (t : ℤ) :
  ((t + 1) * (3 * t - 3)) = ((3 * t - 5) * (t + 2) + 2) → 
  t = 5 :=
by
  intros
  sorry

end find_t_l10_10686


namespace sequence_formula_sequence_inequality_l10_10801

open Nat

-- Definition of the sequence based on the given conditions
noncomputable def a : ℕ → ℚ
| 0     => 1                -- 0-indexed for Lean handling convenience, a_1 = 1 is a(0) in Lean
| (n+1) => 2 - 1 / (a n)    -- recurrence relation

-- Proof for part (I) that a_n = (n + 1) / n
theorem sequence_formula (n : ℕ) : a (n + 1) = (n + 2) / (n + 1) := sorry

-- Proof for part (II)
theorem sequence_inequality (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  (1 + a (n + 1)) / a (k + 1) < 2 ∨ (1 + a (k + 1)) / a (n + 1) < 2 := sorry

end sequence_formula_sequence_inequality_l10_10801


namespace spending_less_l10_10308

-- Define the original costs in USD for each category.
def cost_A_usd : ℝ := 520
def cost_B_usd : ℝ := 860
def cost_C_usd : ℝ := 620

-- Define the budget cuts for each category.
def cut_A : ℝ := 0.25
def cut_B : ℝ := 0.35
def cut_C : ℝ := 0.30

-- Conversion rate from USD to EUR.
def conversion_rate : ℝ := 0.85

-- Sales tax rate.
def tax_rate : ℝ := 0.07

-- Calculate the reduced cost after budget cuts for each category.
def reduced_cost_A_usd := cost_A_usd * (1 - cut_A)
def reduced_cost_B_usd := cost_B_usd * (1 - cut_B)
def reduced_cost_C_usd := cost_C_usd * (1 - cut_C)

-- Convert costs from USD to EUR.
def reduced_cost_A_eur := reduced_cost_A_usd * conversion_rate
def reduced_cost_B_eur := reduced_cost_B_usd * conversion_rate
def reduced_cost_C_eur := reduced_cost_C_usd * conversion_rate

-- Calculate the total reduced cost in EUR before tax.
def total_reduced_cost_eur := reduced_cost_A_eur + reduced_cost_B_eur + reduced_cost_C_eur

-- Calculate the tax amount on the reduced cost.
def tax_reduced_cost := total_reduced_cost_eur * tax_rate

-- Total reduced cost in EUR after tax.
def total_reduced_cost_with_tax := total_reduced_cost_eur + tax_reduced_cost

-- Calculate the original costs in EUR without any cuts.
def original_cost_A_eur := cost_A_usd * conversion_rate
def original_cost_B_eur := cost_B_usd * conversion_rate
def original_cost_C_eur := cost_C_usd * conversion_rate

-- Calculate the total original cost in EUR before tax.
def total_original_cost_eur := original_cost_A_eur + original_cost_B_eur + original_cost_C_eur

-- Calculate the tax amount on the original cost.
def tax_original_cost := total_original_cost_eur * tax_rate

-- Total original cost in EUR after tax.
def total_original_cost_with_tax := total_original_cost_eur + tax_original_cost

-- Difference in spending.
def spending_difference := total_original_cost_with_tax - total_reduced_cost_with_tax

-- Prove the company must spend €561.1615 less.
theorem spending_less : spending_difference = 561.1615 := 
by 
  sorry

end spending_less_l10_10308


namespace guilt_proof_l10_10628

variables (E F G : Prop)

theorem guilt_proof
  (h1 : ¬G → F)
  (h2 : ¬E → G)
  (h3 : G → E)
  (h4 : E → ¬F)
  : E ∧ G :=
by
  sorry

end guilt_proof_l10_10628


namespace value_of_r_when_n_is_2_l10_10228

-- Define the given conditions
def s : ℕ := 2 ^ 2 + 1
def r : ℤ := 3 ^ s - s

-- Prove that r equals 238 when n = 2
theorem value_of_r_when_n_is_2 : r = 238 := by
  sorry

end value_of_r_when_n_is_2_l10_10228


namespace find_k_l10_10159

def a : ℕ := 786
def b : ℕ := 74
def c : ℝ := 1938.8

theorem find_k (k : ℝ) : (a * b) / k = c → k = 30 :=
by
  intro h
  sorry

end find_k_l10_10159


namespace value_of_expression_l10_10211

variable {a b c d e f : ℝ}

theorem value_of_expression :
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 1000 →
  d * e * f = 250 →
  (a * f) / (c * d) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_expression_l10_10211


namespace divisible_by_six_of_power_two_l10_10592

theorem divisible_by_six_of_power_two (n a b : ℤ) (h1 : n > 3) (h2 : 2^n = 10 * a + b) (h3 : b < 10) : 6 ∣ (a * b) :=
by
  sorry

end divisible_by_six_of_power_two_l10_10592


namespace find_y_l10_10822

theorem find_y :
  ∃ (x y : ℤ), (x - 5) / 7 = 7 ∧ (x - y) / 10 = 3 ∧ y = 24 :=
by
  sorry

end find_y_l10_10822


namespace units_digit_is_valid_l10_10153

theorem units_digit_is_valid (n : ℕ) : 
  (∃ k : ℕ, (k^3 % 10 = n)) → 
  (n = 2 ∨ n = 3 ∨ n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end units_digit_is_valid_l10_10153


namespace no_obtuse_triangle_probability_l10_10502

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l10_10502


namespace find_n_l10_10167

theorem find_n (x : ℝ) (h1 : x = 596.95) (h2 : ∃ n : ℝ, n + 11.95 - x = 3054) : ∃ n : ℝ, n = 3639 :=
by
  sorry

end find_n_l10_10167


namespace machine_minutes_worked_l10_10620

-- Definitions based on conditions
def shirts_made_yesterday : ℕ := 9
def shirts_per_minute : ℕ := 3

-- The proof problem statement
theorem machine_minutes_worked (shirts_made_yesterday shirts_per_minute : ℕ) : 
  shirts_made_yesterday / shirts_per_minute = 3 := 
by
  sorry

end machine_minutes_worked_l10_10620


namespace min_inverse_sum_l10_10936

theorem min_inverse_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 4) : 1 ≤ (1/a) + (1/b) :=
by
  sorry

end min_inverse_sum_l10_10936


namespace time_3050_minutes_after_midnight_l10_10096

theorem time_3050_minutes_after_midnight :
  let midnight := datetime.mk 2015 1 1 0 0 0 0,
      mins_to_add := 3050,
      added_datetime := time_since midnight (minutes ∷ mins_to_add)
  in added_datetime = datetime.mk 2015 1 3 2 50 0 0 :=
sorry

end time_3050_minutes_after_midnight_l10_10096


namespace find_other_number_l10_10040

theorem find_other_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 83) (h3 : A = 210) (h4 : LCM * HCF = A * B) : B = 913 :=
by
  sorry

end find_other_number_l10_10040


namespace number_of_children_l10_10160

theorem number_of_children (n m : ℕ) (h1 : 11 * (m + 6) + n * m = n^2 + 3 * n - 2) : n = 9 :=
sorry

end number_of_children_l10_10160


namespace john_age_l10_10893

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end john_age_l10_10893


namespace frank_reads_pages_per_day_l10_10928

theorem frank_reads_pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) : pages_per_book / days_per_book = 83 :=
by {
  sorry
}

end frank_reads_pages_per_day_l10_10928


namespace total_acres_cleaned_l10_10165

theorem total_acres_cleaned (A D : ℕ) (h1 : (D - 1) * 90 + 30 = A) (h2 : D * 80 = A) : A = 480 :=
sorry

end total_acres_cleaned_l10_10165


namespace probability_of_red_black_or_white_l10_10605

def numberOfBalls := 12
def redBalls := 5
def blackBalls := 4
def whiteBalls := 2
def greenBalls := 1

def favorableOutcomes : Nat := redBalls + blackBalls + whiteBalls
def totalOutcomes : Nat := numberOfBalls

theorem probability_of_red_black_or_white :
  (favorableOutcomes : ℚ) / (totalOutcomes : ℚ) = 11 / 12 :=
by
  sorry

end probability_of_red_black_or_white_l10_10605


namespace nancy_earns_more_l10_10121

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end nancy_earns_more_l10_10121


namespace seating_arrangements_l10_10389

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 8
  let jwp_together := (Nat.factorial 6) * (Nat.factorial 3)
  total_arrangements - jwp_together = 36000 := by
  sorry

end seating_arrangements_l10_10389


namespace exists_x_gg_eq_3_l10_10971

noncomputable def g (x : ℝ) : ℝ :=
if x < -3 then -0.5 * x^2 + 3
else if x < 2 then 1
else 0.5 * x^2 - 1.5 * x + 3

theorem exists_x_gg_eq_3 : ∃ x : ℝ, x = -5 ∨ x = 5 ∧ g (g x) = 3 :=
by
  sorry

end exists_x_gg_eq_3_l10_10971


namespace work_together_days_l10_10743

theorem work_together_days (a_days : ℕ) (b_days : ℕ) :
  a_days = 10 → b_days = 9 → (1 / ((1 / (a_days : ℝ)) + (1 / (b_days : ℝ)))) = 90 / 19 :=
by
  intros ha hb
  sorry

end work_together_days_l10_10743


namespace range_of_a_l10_10517

noncomputable def f (a x : ℝ) : ℝ := Real.sin x + 0.5 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 0, f a x ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l10_10517


namespace lily_read_total_books_l10_10689

-- Definitions
def books_weekdays_last_month : ℕ := 4
def books_weekends_last_month : ℕ := 4

def books_weekdays_this_month : ℕ := 2 * books_weekdays_last_month
def books_weekends_this_month : ℕ := 3 * books_weekends_last_month

def total_books_last_month : ℕ := books_weekdays_last_month + books_weekends_last_month
def total_books_this_month : ℕ := books_weekdays_this_month + books_weekends_this_month
def total_books_two_months : ℕ := total_books_last_month + total_books_this_month

-- Proof problem statement
theorem lily_read_total_books : total_books_two_months = 28 :=
by
  sorry

end lily_read_total_books_l10_10689


namespace problem_1_problem_2_l10_10810

variables (a b : ℝ)
axiom mag_a : ‖a‖ = 1
axiom mag_b : ‖b‖ = 2
axiom angle_ab : real.angle a b = real.pi / 3

theorem problem_1 : ‖a + 2 * b‖ = real.sqrt 21 := by
  sorry

axiom dot_product_condition : (2 * a - b) • (3 * a + b) = 3

theorem problem_2 : ∃ θ : ℝ, θ = 2 * real.pi / 3 ∧ ∥ a.angle b ∥ = θ := by
  sorry

end problem_1_problem_2_l10_10810


namespace max_b_no_lattice_points_line_l10_10047

theorem max_b_no_lattice_points_line (b : ℝ) (h : ∀ (m : ℝ), 0 < m ∧ m < b → ∀ (x : ℤ), 0 < (x : ℝ) ∧ (x : ℝ) ≤ 150 → ¬∃ (y : ℤ), y = m * x + 5) :
  b ≤ 1 / 151 :=
by sorry

end max_b_no_lattice_points_line_l10_10047


namespace number_of_boys_is_810_l10_10575

theorem number_of_boys_is_810 (B G : ℕ) (h1 : B + G = 900) (h2 : G = B / 900 * 100) : B = 810 :=
by
  sorry

end number_of_boys_is_810_l10_10575


namespace jack_walked_distance_l10_10112

def jack_walking_time: ℝ := 1.25
def jack_walking_rate: ℝ := 3.2
def jack_distance_walked: ℝ := 4

theorem jack_walked_distance:
  jack_walking_rate * jack_walking_time = jack_distance_walked :=
by
  sorry

end jack_walked_distance_l10_10112


namespace unique_prime_p_l10_10837

def f (x : ℤ) : ℤ := x^3 + 7 * x^2 + 9 * x + 10

theorem unique_prime_p (p : ℕ) (hp : p = 5 ∨ p = 7 ∨ p = 11 ∨ p = 13 ∨ p = 17) :
  (∀ a b : ℤ, f a ≡ f b [ZMOD p] → a ≡ b [ZMOD p]) ↔ p = 11 :=
by
  sorry

end unique_prime_p_l10_10837


namespace two_polygons_sum_of_interior_angles_l10_10574

theorem two_polygons_sum_of_interior_angles
  (n1 n2 : ℕ) (h1 : Even n1) (h2 : Even n2) 
  (h_sum : (n1 - 2) * 180 + (n2 - 2) * 180 = 1800):
  (n1 = 4 ∧ n2 = 10) ∨ (n1 = 6 ∧ n2 = 8) :=
by
  sorry

end two_polygons_sum_of_interior_angles_l10_10574


namespace problem_statement_l10_10197

open Set

variable (a : ℕ)
variable (A : Set ℕ := {2, 3, 4})
variable (B : Set ℕ := {a + 2, a})

theorem problem_statement (hB : B ⊆ A) : (A \ B) = {3} :=
sorry

end problem_statement_l10_10197


namespace sum_of_arithmetic_sequence_l10_10369

variable {α : Type*} [LinearOrderedField α]

def sum_arithmetic_sequence (a₁ d : α) (n : ℕ) : α :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence {a₁ d : α}
  (h₁ : sum_arithmetic_sequence a₁ d 10 = 12) :
  (a₁ + 4 * d) + (a₁ + 5 * d) = 12 / 5 :=
by
  sorry

end sum_of_arithmetic_sequence_l10_10369


namespace total_people_attended_l10_10172

theorem total_people_attended (A C : ℕ) (ticket_price_adult ticket_price_child : ℕ) (total_receipts : ℕ) 
  (number_of_children : ℕ) (h_ticket_prices : ticket_price_adult = 60 ∧ ticket_price_child = 25)
  (h_total_receipts : total_receipts = 140 * 100) (h_children : C = 80) 
  (h_equation : ticket_price_adult * A + ticket_price_child * C = total_receipts) : 
  A + C = 280 :=
by
  sorry

end total_people_attended_l10_10172


namespace length_of_square_side_l10_10591

theorem length_of_square_side (length_of_string : ℝ) (num_sides : ℕ) (total_side_length : ℝ) 
  (h1 : length_of_string = 32) (h2 : num_sides = 4) (h3 : total_side_length = length_of_string) : 
  total_side_length / num_sides = 8 :=
by
  sorry

end length_of_square_side_l10_10591


namespace proportion_option_B_true_l10_10380

theorem proportion_option_B_true {a b c d : ℚ} (h : a / b = c / d) : 
  (a + c) / c = (b + d) / d := 
by 
  sorry

end proportion_option_B_true_l10_10380


namespace correct_statements_l10_10994

open MeasureTheory ProbabilityTheory

noncomputable def Binomial : ProbabilityMassFunction ℕ := sorry
noncomputable def Normal : ProbabilityMassFunction ℝ := sorry

variables
  (X : ℕ → ℝ) -- Random variable X for the binomial distribution
  (Y : ℝ → ℝ) -- Random variable Y for the normal distribution

-- Defining the conditions for binomial distribution B(4, 1/3)
axiom binomial_dist : X follows Binomial

-- Defining the conditions for normal distribution N(3, σ^2)
axiom normal_dist : Y follows Normal

-- Probability value P(X ≤ 5) = 0.85 for normal distribution case
axiom normal_prob_5 : P((λ x, x ≤ 5) Y) = 0.85

-- Variance of a random variable X
variable D : ℝ → ℝ

-- Defining the condition about variance
axiom variance_property : D(Y) = D(X)

-- Defining the mutually exclusive condition
axiom mutually_exclusive (A B : Event) : A ∩ B = ∅

-- Defining the theorem
theorem correct_statements :
  B_correct : (∃ σ^2, (P((1 <ᵣ Y ∧ Y ≤ 3)) = 0.35)) ∧ 
  D_correct : mutually_exclusive → complementary events :=
by
  sorry

end correct_statements_l10_10994


namespace a_plus_b_plus_c_at_2_l10_10981

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def maximum_value (a b c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic a b c x = 75

def passes_through (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  quadratic a b c p1.1 = p1.2 ∧ quadratic a b c p2.1 = p2.2

theorem a_plus_b_plus_c_at_2 
  (a b c : ℝ)
  (hmax : maximum_value a b c)
  (hpoints : passes_through a b c (-3, 0) (3, 0))
  (hvertex : ∀ x : ℝ, quadratic a 0 c x ≤ quadratic a (2 * b) c 0) : 
  quadratic a b c 2 = 125 / 3 :=
sorry

end a_plus_b_plus_c_at_2_l10_10981


namespace min_sum_of_dimensions_l10_10137

theorem min_sum_of_dimensions 
  (a b c : ℕ) 
  (h_pos : a > 0) 
  (h_pos_2 : b > 0) 
  (h_pos_3 : c > 0) 
  (h_even : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) 
  (h_vol : a * b * c = 1806) 
  : a + b + c = 56 :=
sorry

end min_sum_of_dimensions_l10_10137


namespace math_proof_equivalent_l10_10602

theorem math_proof_equivalent :
  (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end math_proof_equivalent_l10_10602


namespace find_third_divisor_l10_10488

theorem find_third_divisor 
  (h1 : ∃ (n : ℕ), n = 1014 - 3 ∧ n % 12 = 0 ∧ n % 16 = 0 ∧ n % 21 = 0 ∧ n % 28 = 0) 
  (h2 : 1011 - 3 = 1008) : 
  (∃ d, d = 3 ∧ 1008 % d = 0 ∧ 1008 % 12 = 0 ∧ 1008 % 16 = 0 ∧ 1008 % 21 = 0 ∧ 1008 % 28 = 0) :=
sorry

end find_third_divisor_l10_10488


namespace arithmetic_sequence_a4_a7_div2_eq_10_l10_10831

theorem arithmetic_sequence_a4_a7_div2_eq_10 (a : ℕ → ℝ) (h : a 4 + a 6 = 20) : (a 3 + a 6) / 2 = 10 :=
  sorry

end arithmetic_sequence_a4_a7_div2_eq_10_l10_10831


namespace general_formula_sum_first_n_terms_l10_10074

open BigOperators

def geometric_sequence (a_3 : ℚ) (q : ℚ) : ℕ → ℚ
| 0       => 1 -- this is a placeholder since sequence usually start from 1
| (n + 1) => 1 * q ^ n

def sum_geometric_sequence (a_1 q : ℚ) (n : ℕ) : ℚ :=
  a_1 * (1 - q ^ n) / (1 - q)

theorem general_formula (a_3 : ℚ) (q : ℚ) (n : ℕ) (h_a3 : a_3 = 1 / 4) (h_q : q = -1 / 2) :
  geometric_sequence a_3 q (n + 1) = (-1 / 2) ^ n :=
by
  sorry

theorem sum_first_n_terms (a_1 q : ℚ) (n : ℕ) (h_a1 : a_1 = 1) (h_q : q = -1 / 2) :
  sum_geometric_sequence a_1 q n = 2 / 3 * (1 - (-1 / 2) ^ n) :=
by
  sorry

end general_formula_sum_first_n_terms_l10_10074


namespace no_obtuse_triangle_probability_l10_10503

noncomputable def prob_no_obtuse_triangle : ℝ :=
  9 / 128

theorem no_obtuse_triangle_probability :
  let circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 },
      points : List (ℝ × ℝ) × ℝ := (List.replicate 4 (0, 0), 1) -- assuming uniform rand points chosen
  in 
  ∀ (A₀ A₁ A₂ A₃ : ℝ × ℝ), A₀ ∈ circle → A₁ ∈ circle → A₂ ∈ circle → A₃ ∈ circle →
  (prob_no_obtuse_triangle = 9 / 128) :=
by
  sorry

end no_obtuse_triangle_probability_l10_10503


namespace ice_cream_sandwiches_l10_10874

theorem ice_cream_sandwiches (n : ℕ) (x : ℕ) (h1 : n = 11) (h2 : x = 13) : (n * x = 143) := 
by
  sorry

end ice_cream_sandwiches_l10_10874


namespace find_larger_number_l10_10426

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 10) : L = 1636 := 
by
  sorry

end find_larger_number_l10_10426


namespace red_apples_count_l10_10961

-- Definitions based on conditions
def green_apples : ℕ := 2
def yellow_apples : ℕ := 14
def total_apples : ℕ := 19

-- Definition of red apples as a theorem to be proven
theorem red_apples_count :
  green_apples + yellow_apples + red_apples = total_apples → red_apples = 3 :=
by
  -- You would need to prove this using Lean
  sorry

end red_apples_count_l10_10961


namespace asymptotes_of_hyperbola_l10_10259

theorem asymptotes_of_hyperbola (a b x y : ℝ) (h : a = 5 ∧ b = 2) :
  (x^2 / 25 - y^2 / 4 = 1) → (y = (2 / 5) * x ∨ y = -(2 / 5) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l10_10259


namespace hyperbola_eccentricity_l10_10513

-- Definitions based on the conditions
def hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0)

def distance_from_focus_to_asymptote (a b c : ℝ) : Prop :=
  (b^2 * c) / (a^2 + b^2).sqrt = b ∧ b = 2 * Real.sqrt 3

def minimum_distance_point_to_focus (a c : ℝ) : Prop :=
  c - a = 2

def eccentricity (a c e : ℝ) : Prop :=
  e = c / a

-- Problem statement
theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_hyperbola : hyperbola a b)
  (h_dist_asymptote : distance_from_focus_to_asymptote a b c)
  (h_min_dist_focus : minimum_distance_point_to_focus a c)
  (h_eccentricity : eccentricity a c e) :
  e = 2 :=
sorry

end hyperbola_eccentricity_l10_10513


namespace abs_sum_neq_zero_iff_or_neq_zero_l10_10081

variable {x y : ℝ}

theorem abs_sum_neq_zero_iff_or_neq_zero (x y : ℝ) :
  (|x| + |y| ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end abs_sum_neq_zero_iff_or_neq_zero_l10_10081


namespace value_of_d_l10_10104

theorem value_of_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (4 * y) / 20 + (3 * y) / d = 0.5 * y) : d = 10 :=
by
  sorry

end value_of_d_l10_10104


namespace water_added_is_five_l10_10612

theorem water_added_is_five :
  ∃ W x : ℝ, (4 / 3 = 10 / W) ∧ (4 / 5 = 10 / (W + x)) ∧ x = 5 := by
  sorry

end water_added_is_five_l10_10612


namespace distance_from_apex_to_A_is_zero_l10_10052

theorem distance_from_apex_to_A_is_zero :
  let A : ℝ × ℝ × ℝ := (0, 0, 0)
  let B : ℝ × ℝ × ℝ := (4 * Real.sqrt 3, 0, 0)
  let C : ℝ × ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.sqrt 3, 0)
  let P : ℝ × ℝ × ℝ := (0, 0, 10)
  let R : ℝ × ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.sqrt 3, 6)
  let t := 0 in
  let Point_on_AC := (2 * t * Real.sqrt 3, 2 * t * Real.sqrt 3, 10 - 4 * t) in
  (Point_on_AC.1 - A.1) ^ 2 + (Point_on_AC.2 - A.2) ^ 2 + 0 ^ 2 = 0 :=
by
  -- Proof goes here
  sorry

end distance_from_apex_to_A_is_zero_l10_10052


namespace sum_possible_n_k_l10_10979

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l10_10979


namespace all_numbers_appear_on_diagonal_l10_10763

theorem all_numbers_appear_on_diagonal 
  (n : ℕ) 
  (h_odd : n % 2 = 1)
  (A : Matrix (Fin n) (Fin n) (Fin n.succ))
  (h_elements : ∀ i j, 1 ≤ A i j ∧ A i j ≤ n) 
  (h_unique_row : ∀ i k, ∃! j, A i j = k)
  (h_unique_col : ∀ j k, ∃! i, A i j = k)
  (h_symmetric : ∀ i j, A i j = A j i)
  : ∀ k, 1 ≤ k ∧ k ≤ n → ∃ i, A i i = k := 
by {
  sorry
}

end all_numbers_appear_on_diagonal_l10_10763


namespace parabola_through_point_l10_10963

theorem parabola_through_point (a b : ℝ) (ha : 0 < a) :
  ∃ f : ℝ → ℝ, (∀ x, f x = -a*x^2 + b*x + 1) ∧ f 0 = 1 :=
by
  -- We are given a > 0
  -- We need to show there exists a parabola of the form y = -a*x^2 + b*x + 1 passing through (0,1)
  sorry

end parabola_through_point_l10_10963


namespace carolyn_shared_with_diana_l10_10782

theorem carolyn_shared_with_diana (initial final shared : ℕ) 
    (h_initial : initial = 47) 
    (h_final : final = 5)
    (h_shared : shared = initial - final) : shared = 42 := by
  rw [h_initial, h_final] at h_shared
  exact h_shared

end carolyn_shared_with_diana_l10_10782


namespace complex_point_quadrant_l10_10363

theorem complex_point_quadrant 
  (i : Complex) 
  (h_i_unit : i = Complex.I) : 
  (Complex.re ((i - 3) / (1 + i)) < 0) ∧ (Complex.im ((i - 3) / (1 + i)) > 0) :=
by {
  sorry
}

end complex_point_quadrant_l10_10363


namespace eval_expression_l10_10527

theorem eval_expression (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m - 3 = -1 :=
by
  sorry

end eval_expression_l10_10527


namespace tina_savings_l10_10274

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end tina_savings_l10_10274


namespace foma_wait_time_probability_l10_10342

noncomputable def probability_no_more_than_four_minutes_wait (x y : ℝ) : ℝ :=
if h : 2 < x ∧ x < y ∧ y < 10 ∧ y - x ≤ 4 then
  (1 / 2)
else 0

theorem foma_wait_time_probability :
  ∀ (x y : ℝ), 2 < x → x < y → y < 10 → 
  (probability_no_more_than_four_minutes_wait x y) = 1 / 2 :=
sorry

end foma_wait_time_probability_l10_10342


namespace songs_performed_l10_10218

variable (R L S M : ℕ)
variable (songs_total : ℕ)

def conditions := 
  R = 9 ∧ L = 6 ∧ (6 ≤ S ∧ S ≤ 9) ∧ (6 ≤ M ∧ M ≤ 9) ∧ songs_total = (R + L + S + M) / 3

theorem songs_performed (h : conditions R L S M songs_total) :
  songs_total = 9 ∨ songs_total = 10 ∨ songs_total = 11 :=
sorry

end songs_performed_l10_10218


namespace real_root_uncertainty_l10_10535

noncomputable def f (x m : ℝ) : ℝ := m * x^2 - 2 * (m + 2) * x + m + 5
noncomputable def g (x m : ℝ) : ℝ := (m - 5) * x^2 - 2 * (m + 2) * x + m

theorem real_root_uncertainty (m : ℝ) :
  (∀ x : ℝ, f x m ≠ 0) → 
  (m ≤ 5 → ∃ x : ℝ, g x m = 0 ∧ ∀ y : ℝ, y ≠ x → g y m = 0) ∧
  (m > 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) :=
sorry

end real_root_uncertainty_l10_10535


namespace abs_fraction_eq_sqrt_three_over_two_l10_10532

variable (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b)

theorem abs_fraction_eq_sqrt_three_over_two (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b) : 
  |(a + b) / (a - b)| = Real.sqrt (3 / 2) := by
  sorry

end abs_fraction_eq_sqrt_three_over_two_l10_10532


namespace cities_with_highest_increase_l10_10796

-- Define population changes for each city
def cityF_initial := 30000
def cityF_final := 45000
def cityG_initial := 55000
def cityG_final := 77000
def cityH_initial := 40000
def cityH_final := 60000
def cityI_initial := 70000
def cityI_final := 98000
def cityJ_initial := 25000
def cityJ_final := 37500

-- Function to calculate percentage increase
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / (initial : ℚ) * 100

-- Theorem stating cities F, H, and J had the highest percentage increase
theorem cities_with_highest_increase :
  percentage_increase cityF_initial cityF_final = 50 ∧
  percentage_increase cityH_initial cityH_final = 50 ∧
  percentage_increase cityJ_initial cityJ_final = 50 ∧
  percentage_increase cityG_initial cityG_final < 50 ∧
  percentage_increase cityI_initial cityI_final < 50 :=
by
-- Proof omitted
sorry

end cities_with_highest_increase_l10_10796


namespace jennifer_fruits_left_l10_10833

-- Definitions based on the conditions
def pears : ℕ := 15
def oranges : ℕ := 30
def apples : ℕ := 2 * pears
def cherries : ℕ := oranges / 2
def grapes : ℕ := 3 * apples
def pineapples : ℕ := pears + oranges + apples + cherries + grapes

-- Definitions for the number of fruits given to the sister
def pears_given : ℕ := 3
def oranges_given : ℕ := 5
def apples_given : ℕ := 5
def cherries_given : ℕ := 7
def grapes_given : ℕ := 3

-- Calculations based on the conditions for what's left after giving fruits
def pears_left : ℕ := pears - pears_given
def oranges_left : ℕ := oranges - oranges_given
def apples_left : ℕ := apples - apples_given
def cherries_left : ℕ := cherries - cherries_given
def grapes_left : ℕ := grapes - grapes_given

def remaining_pineapples : ℕ := pineapples - (pineapples / 2)

-- Total number of fruits left
def total_fruits_left : ℕ := pears_left + oranges_left + apples_left + cherries_left + grapes_left + remaining_pineapples

-- Theorem statement
theorem jennifer_fruits_left : total_fruits_left = 247 :=
by
  -- The detailed proof would go here
  sorry

end jennifer_fruits_left_l10_10833


namespace find_m_l10_10086

-- Define the given vectors and the parallel condition
def vectors_parallel (m : ℝ) : Prop :=
  let a := (1, m)
  let b := (3, 1)
  a.1 * b.2 = a.2 * b.1

-- Statement to be proved
theorem find_m (m : ℝ) : vectors_parallel m → m = 1 / 3 :=
by
  sorry

end find_m_l10_10086


namespace intersection_of_M_and_N_l10_10653

def set_M : Set ℝ := {x | -1 < x}
def set_N : Set ℝ := {x | x * (x + 2) ≤ 0}

theorem intersection_of_M_and_N : (set_M ∩ set_N) = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_of_M_and_N_l10_10653


namespace modular_inverse_sum_eq_14_l10_10441

theorem modular_inverse_sum_eq_14 : 
(9 + 13 + 15 + 16 + 12 + 3 + 14) % 17 = 14 := by
  sorry

end modular_inverse_sum_eq_14_l10_10441


namespace required_brick_volume_l10_10046

theorem required_brick_volume :
  let height := 4 / 12 -- in feet
  let length := 6 -- in feet
  let thickness := 4 / 12 -- in feet
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  rounded_volume = 1 := 
by
  let height := 1 / 3
  let length := 6
  let thickness := 1 / 3
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  show rounded_volume = 1
  sorry

end required_brick_volume_l10_10046


namespace red_stripe_area_l10_10767

theorem red_stripe_area (diameter height stripe_width : ℝ) (num_revolutions : ℕ) 
  (diam_pos : 0 < diameter) (height_pos : 0 < height) (width_pos : 0 < stripe_width) (height_eq_80 : height = 80)
  (width_eq_3 : stripe_width = 3) (revolutions_eq_2 : num_revolutions = 2) :
  240 = stripe_width * height := 
by
  sorry

end red_stripe_area_l10_10767


namespace find_x_values_l10_10405

def f (x : ℝ) : ℝ := 3 * x^2 - 8

noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Placeholder for the inverse function

theorem find_x_values:
  ∃ x : ℝ, (f x = f_inv x) ↔ (x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6) := sorry

end find_x_values_l10_10405


namespace intersection_of_M_and_N_l10_10372

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}
def intersection := {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7}

theorem intersection_of_M_and_N : M ∩ N = intersection := by
  sorry

end intersection_of_M_and_N_l10_10372


namespace ax2_x_plus_1_positive_l10_10715

theorem ax2_x_plus_1_positive (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ (a > 1/4) :=
by {
  sorry
}

end ax2_x_plus_1_positive_l10_10715


namespace inequality_solution_l10_10706

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l10_10706


namespace heptagon_diagonals_l10_10899

theorem heptagon_diagonals (n : ℕ) (h : n = 7) : (n * (n - 3)) / 2 = 14 := by
  sorry

end heptagon_diagonals_l10_10899


namespace total_donation_l10_10090

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end total_donation_l10_10090


namespace chemistry_more_than_physics_l10_10145

theorem chemistry_more_than_physics
  (M P C : ℕ)
  (h1 : M + P = 60)
  (h2 : (M + C) / 2 = 35) :
  ∃ x : ℕ, C = P + x ∧ x = 10 := 
by
  sorry

end chemistry_more_than_physics_l10_10145


namespace initial_men_count_l10_10273

theorem initial_men_count (M : ℕ) :
  let total_food := M * 22
  let food_after_2_days := total_food - 2 * M
  let remaining_food := 20 * M
  let new_total_men := M + 190
  let required_food_for_16_days := new_total_men * 16
  (remaining_food = required_food_for_16_days) → M = 760 :=
by
  intro h
  sorry

end initial_men_count_l10_10273


namespace solution_to_g_inv_2_l10_10839

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := 1 / (c * x + d)

theorem solution_to_g_inv_2 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
    ∃ x : ℝ, g x c d = 2 ↔ x = (1 - 2 * d) / (2 * c) :=
by
  sorry

end solution_to_g_inv_2_l10_10839


namespace distance_to_bus_stand_l10_10887

theorem distance_to_bus_stand :
  ∀ D : ℝ, (D / 5 - 0.2 = D / 6 + 0.25) → D = 13.5 :=
by
  intros D h
  sorry

end distance_to_bus_stand_l10_10887


namespace esther_evening_speed_l10_10630

/-- Esther's average speed in the evening was 30 miles per hour -/
theorem esther_evening_speed : 
  let morning_speed := 45   -- miles per hour
  let total_commuting_time := 1 -- hour
  let morning_distance := 18  -- miles
  let evening_distance := 18  -- miles (same route)
  let time_morning := morning_distance / morning_speed
  let time_evening := total_commuting_time - time_morning
  let evening_speed := evening_distance / time_evening
  evening_speed = 30 := 
by sorry

end esther_evening_speed_l10_10630


namespace solve_for_x_l10_10657

theorem solve_for_x (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2 * y = 10) : x = 26 / 3 := by
  sorry

end solve_for_x_l10_10657


namespace monotone_f_find_m_l10_10812

noncomputable def f (x : ℝ) : ℝ := (2 * x - 2) / (x + 2)

theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by
  sorry

theorem find_m (m : ℝ) : 
  (∃ m, (f m - f 1 = 1/2)) ↔ m = 2 :=
by
  sorry

end monotone_f_find_m_l10_10812


namespace arithmetic_geometric_sequence_l10_10511

/-- Given:
  * 1, a₁, a₂, 4 form an arithmetic sequence
  * 1, b₁, b₂, b₃, 4 form a geometric sequence
Prove that:
  (a₁ + a₂) / b₂ = 5 / 2
-/
theorem arithmetic_geometric_sequence (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : 2 * a₁ = 1 + a₂ ∧ 2 * a₂ = a₁ + 4)
  (h_geom : b₁ * b₁ = b₂ ∧ b₁ * b₂ = b₃ ∧ b₂ * b₂ = b₃ * 4) :
  (a₁ + a₂) / b₂ = 5 / 2 :=
sorry

end arithmetic_geometric_sequence_l10_10511


namespace solve_equation_l10_10419

theorem solve_equation (x : ℝ) : 
  (x - 4)^6 + (x - 6)^6 = 64 → x = 4 ∨ x = 6 :=
by
  sorry

end solve_equation_l10_10419


namespace fish_tagging_problem_l10_10826

theorem fish_tagging_problem
  (N : ℕ) (T : ℕ)
  (h1 : N = 1250)
  (h2 : T = N / 25) :
  T = 50 :=
sorry

end fish_tagging_problem_l10_10826


namespace function_identity_l10_10404

theorem function_identity (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, f m + f n ∣ m + n) : ∀ m : ℕ+, f m = m := by
  sorry

end function_identity_l10_10404


namespace distance_between_X_and_Y_l10_10688

theorem distance_between_X_and_Y 
  (b_walked_distance : ℕ) 
  (time_difference : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_rate : ℕ) 
  (time_bob_walked : ℕ) 
  (distance_when_met : ℕ) 
  (bob_walked_8_miles : b_walked_distance = 8) 
  (one_hour_time_difference : time_difference = 1) 
  (yolanda_3_mph : yolanda_rate = 3) 
  (bob_4_mph : bob_rate = 4) 
  (time_bob_2_hours : time_bob_walked = b_walked_distance / bob_rate)
  : 
  distance_when_met = yolanda_rate * (time_bob_walked + time_difference) + bob_rate * time_bob_walked :=
by
  sorry  -- proof steps

end distance_between_X_and_Y_l10_10688


namespace divisibility_3804_l10_10852

theorem divisibility_3804 (n : ℕ) (h : 0 < n) :
    3804 ∣ ((n ^ 3 - n) * (5 ^ (8 * n + 4) + 3 ^ (4 * n + 2))) :=
sorry

end divisibility_3804_l10_10852


namespace nancy_earns_more_l10_10120

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end nancy_earns_more_l10_10120


namespace minimum_elapsed_time_l10_10341

theorem minimum_elapsed_time : 
  let initial_time := 45  -- in minutes
  let final_time := 3 * 60 + 30  -- 3 hours 30 minutes in minutes
  let elapsed_time := final_time - initial_time
  elapsed_time = 2 * 60 + 45 :=
by
  sorry

end minimum_elapsed_time_l10_10341


namespace bound_diff_sqrt_two_l10_10232

theorem bound_diff_sqrt_two (a b k m : ℝ) (h : ∀ x ∈ Set.Icc a b, abs (x^2 - k * x - m) ≤ 1) : b - a ≤ 2 * Real.sqrt 2 := sorry

end bound_diff_sqrt_two_l10_10232


namespace geometric_series_sum_l10_10185

theorem geometric_series_sum:
  let a := 1
  let r := 5
  let n := 5
  (1 - r^n) / (1 - r) = 781 :=
by
  let a := 1
  let r := 5
  let n := 5
  sorry

end geometric_series_sum_l10_10185


namespace consecutive_sum_ways_l10_10948

theorem consecutive_sum_ways (S : ℕ) (hS : S = 385) :
  ∃! n : ℕ, ∃! k : ℕ, n ≥ 2 ∧ S = n * (2 * k + n - 1) / 2 :=
sorry

end consecutive_sum_ways_l10_10948


namespace iced_coffee_cost_correct_l10_10241

-- Definitions based on the conditions 
def coffee_cost_per_day (iced_coffee_cost : ℝ) : ℝ := 3 + iced_coffee_cost
def total_spent (days : ℕ) (iced_coffee_cost : ℝ) : ℝ := days * coffee_cost_per_day iced_coffee_cost

-- Proof statement
theorem iced_coffee_cost_correct (iced_coffee_cost : ℝ) (h : total_spent 20 iced_coffee_cost = 110) : iced_coffee_cost = 2.5 :=
by
  sorry

end iced_coffee_cost_correct_l10_10241


namespace part1_part2_l10_10391

variables {A B C : ℝ} {a b c : ℝ}

-- conditions of the problem
def condition_1 (a b c : ℝ) (C : ℝ) : Prop :=
  a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0

def condition_2 (C : ℝ) : Prop :=
  0 < C ∧ C < Real.pi

-- Part 1: Proving the value of angle A
theorem part1 (a b c C : ℝ) (h1 : condition_1 a b c C) (h2 : condition_2 C) : 
  A = Real.pi / 3 :=
sorry

-- Part 2: Range of possible values for the perimeter, given c = 3
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2

theorem part2 (a b A B C : ℝ) (h1 : condition_1 a b 3 C) (h2 : condition_2 C) 
           (h3 : A = Real.pi / 3) (h4 : is_acute_triangle A B C) :
  ∃ p, p ∈ Set.Ioo ((3 * Real.sqrt 3 + 9) / 2) (9 + 3 * Real.sqrt 3) :=
sorry

end part1_part2_l10_10391


namespace toll_for_18_wheel_truck_l10_10157

-- Define the number of wheels and axles conditions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def number_of_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the toll calculation formula
def toll (x : ℕ) : ℝ := 1.50 + 1.50 * (x - 2)

-- Lean theorem statement asserting that the toll for the given truck is 6 dollars
theorem toll_for_18_wheel_truck : toll number_of_axles = 6 := by
  -- Skipping the actual proof using sorry
  sorry

end toll_for_18_wheel_truck_l10_10157


namespace volume_of_cube_l10_10244

theorem volume_of_cube (a : ℕ) (h : a^3 - (a^3 - 4 * a) = 12) : a^3 = 27 :=
by 
  sorry

end volume_of_cube_l10_10244


namespace negation_proposition_l10_10983

theorem negation_proposition : 
  ¬(∀ x : ℝ, 0 ≤ x → 2^x > x^2) ↔ ∃ x : ℝ, 0 ≤ x ∧ 2^x ≤ x^2 := by
  sorry

end negation_proposition_l10_10983


namespace remainder_when_x150_divided_by_x1_4_l10_10066

noncomputable def remainder_div_x150_by_x1_4 (x : ℝ) : ℝ :=
  x^150 % (x-1)^4

theorem remainder_when_x150_divided_by_x1_4 (x : ℝ) :
  remainder_div_x150_by_x1_4 x = -551300 * x^3 + 1665075 * x^2 - 1667400 * x + 562626 :=
by
  sorry

end remainder_when_x150_divided_by_x1_4_l10_10066


namespace cube_surface_area_l10_10020

open Real

theorem cube_surface_area (V : ℝ) (a : ℝ) (S : ℝ)
  (h1 : V = a ^ 3)
  (h2 : a = 4)
  (h3 : V = 64) :
  S = 6 * a ^ 2 :=
by
  sorry

end cube_surface_area_l10_10020


namespace picture_area_l10_10375

theorem picture_area (x y : ℕ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (3*x + 3) * (y + 2) = 110) : x * y = 28 :=
by {
  sorry
}

end picture_area_l10_10375


namespace circle_equation_l10_10187

-- Defining the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Defining the point
def point : ℝ × ℝ := (1, -1)

-- Proving the equation of the new circle that passes through the intersection points 
-- of the given circles and the given point
theorem circle_equation (x y : ℝ) :
  (circle1 x y ∧ circle2 x y ∧ x = 1 ∧ y = -1) → 9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0 :=
sorry

end circle_equation_l10_10187


namespace lattice_points_equality_l10_10719

-- Definitions based on the problem statement conditions
def f (t q s : ℕ) : ℕ :=
  ∑ x in Finset.range (t + 1), (Int.floor ((s - 1) * x + t) / q) - (Int.ceil ((s + 1) * x - t) / q) + 1

-- Main theorem to prove
theorem lattice_points_equality (t q r s : ℕ) (hq_div : q ∣ (r * s - 1)) :
  f t q r = f t q s := sorry

end lattice_points_equality_l10_10719


namespace negation_exists_l10_10138

theorem negation_exists {x : ℝ} (h : ∀ x, x > 0 → x^2 - x ≤ 0) : ∃ x, x > 0 ∧ x^2 - x > 0 :=
sorry

end negation_exists_l10_10138


namespace tangent_line_to_ellipse_l10_10060

theorem tangent_line_to_ellipse (k : ℝ) :
  (∀ x : ℝ, (x / 2 + 2 * (k * x + 2) ^ 2) = 2) →
  k^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l10_10060


namespace opposite_of_pi_eq_neg_pi_l10_10154

theorem opposite_of_pi_eq_neg_pi (π : Real) (h : π = Real.pi) : -π = -Real.pi :=
by sorry

end opposite_of_pi_eq_neg_pi_l10_10154


namespace cost_of_gravelling_roads_l10_10906

theorem cost_of_gravelling_roads :
  let lawn_length := 70
  let lawn_breadth := 30
  let road_width := 5
  let cost_per_sqm := 4
  let area_road_length := lawn_length * road_width
  let area_road_breadth := lawn_breadth * road_width
  let area_intersection := road_width * road_width
  let total_area_to_be_graveled := (area_road_length + area_road_breadth) - area_intersection
  let total_cost := total_area_to_be_graveled * cost_per_sqm
  total_cost = 1900 :=
by
  sorry

end cost_of_gravelling_roads_l10_10906


namespace building_houses_200_people_l10_10603

-- Define number of floors, apartments per floor, and people per apartment as constants
def numFloors := 25
def apartmentsPerFloor := 4
def peoplePerApartment := 2

-- Define the total number of apartments
def totalApartments := numFloors * apartmentsPerFloor

-- Define the total number of people
def totalPeople := totalApartments * peoplePerApartment

theorem building_houses_200_people : totalPeople = 200 :=
by
  sorry

end building_houses_200_people_l10_10603


namespace find_x_for_fraction_equality_l10_10489

theorem find_x_for_fraction_equality (x : ℝ) : 
  (4 + 2 * x) / (7 + x) = (2 + x) / (3 + x) ↔ (x = -2 ∨ x = 1) := by
  sorry

end find_x_for_fraction_equality_l10_10489


namespace Micheal_work_rate_l10_10238

theorem Micheal_work_rate 
    (M A : ℕ) 
    (h1 : 1 / M + 1 / A = 1 / 20)
    (h2 : 9 / 200 = 1 / A) : M = 200 :=
by
    sorry

end Micheal_work_rate_l10_10238


namespace point_to_line_distance_l10_10258

theorem point_to_line_distance :
  let circle_center : ℝ×ℝ := (0, 1)
  let A : ℝ := -1
  let B : ℝ := 1
  let C : ℝ := -2
  let line_eq (x y : ℝ) := A * x + B * y + C == 0
  ∀ (x0 : ℝ) (y0 : ℝ),
    circle_center = (x0, y0) →
    (|A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)) = (Real.sqrt 2 / 2) := 
by 
  intros
  -- Proof goes here
  sorry -- Placeholder for the proof.

end point_to_line_distance_l10_10258


namespace four_digit_numbers_condition_l10_10095

theorem four_digit_numbers_condition :
  ∃ (N : Nat), (1000 ≤ N ∧ N < 10000) ∧
               (∃ x a : Nat, N = 1000 * a + x ∧ x = 200 * a ∧ 1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end four_digit_numbers_condition_l10_10095


namespace figure_perimeter_l10_10860

theorem figure_perimeter 
  (side_length : ℕ)
  (inner_large_square_sides : ℕ)
  (shared_edge_length : ℕ)
  (rectangle_dimension_1 : ℕ)
  (rectangle_dimension_2 : ℕ) 
  (h1 : side_length = 2)
  (h2 : inner_large_square_sides = 4)
  (h3 : shared_edge_length = 2)
  (h4 : rectangle_dimension_1 = 2)
  (h5 : rectangle_dimension_2 = 1) : 
  let large_square_perimeter := inner_large_square_sides * side_length
  let horizontal_perimeter := large_square_perimeter - shared_edge_length + rectangle_dimension_1 + rectangle_dimension_2
  let vertical_perimeter := large_square_perimeter
  horizontal_perimeter + vertical_perimeter = 33 := 
by
  sorry

end figure_perimeter_l10_10860


namespace num_ints_satisfying_inequality_l10_10655

theorem num_ints_satisfying_inequality : ∃ n : ℕ, ∀ a : ℤ, (-4 ≤ a ∧ a ≤ 4) ∧ (-100 < a^3 ∧ a^3 < 100) → n = 9 :=
begin
  sorry
end

end num_ints_satisfying_inequality_l10_10655


namespace ratio_of_cream_l10_10225

def initial_coffee := 12
def joe_drank := 2
def cream_added := 2
def joann_cream_added := 2
def joann_drank := 2

noncomputable def joe_coffee_after_drink_add := initial_coffee - joe_drank + cream_added
noncomputable def joe_cream := cream_added

noncomputable def joann_initial_mixture := initial_coffee + joann_cream_added
noncomputable def joann_portion_before_drink := joann_cream_added / joann_initial_mixture
noncomputable def joann_remaining_coffee := joann_initial_mixture - joann_drank
noncomputable def joann_cream_after_drink := joann_portion_before_drink * joann_remaining_coffee
noncomputable def joann_cream := joann_cream_after_drink

theorem ratio_of_cream : joe_cream / joann_cream = 7 / 6 :=
by sorry

end ratio_of_cream_l10_10225


namespace volume_of_prism_l10_10002

-- Given dimensions a, b, and c, with the following conditions:
variables (a b c : ℝ)
axiom ab_eq_30 : a * b = 30
axiom ac_eq_40 : a * c = 40
axiom bc_eq_60 : b * c = 60

-- The volume of the prism is given by:
theorem volume_of_prism : a * b * c = 120 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l10_10002


namespace smallest_divisor_28_l10_10922

theorem smallest_divisor_28 : ∃ (d : ℕ), d > 0 ∧ d ∣ 28 ∧ ∀ (d' : ℕ), d' > 0 ∧ d' ∣ 28 → d ≤ d' := by
  sorry

end smallest_divisor_28_l10_10922


namespace least_number_to_add_l10_10151

theorem least_number_to_add (n : ℕ) (h : (1052 + n) % 37 = 0) : n = 19 := by
  sorry

end least_number_to_add_l10_10151


namespace area_of_parallelogram_l10_10486

def vector_u : fin 3 → ℝ := ![2, 4, -1]
def vector_v : fin 3 → ℝ := ![5, -2, 3]

def cross_product (u v : fin 3 → ℝ) : fin 3 → ℝ :=
![u 1 * v 2 - u 2 * v 1,
  u 2 * v 0 - u 0 * v 2,
  u 0 * v 1 - u 1 * v 0]

noncomputable def vector_magnitude (v : fin 3 → ℝ) : ℝ :=
real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)

theorem area_of_parallelogram :
  vector_magnitude (cross_product vector_u vector_v) = real.sqrt 797 :=
by sorry

end area_of_parallelogram_l10_10486


namespace find_third_number_l10_10101

-- Definitions and conditions for the problem
def x : ℚ := 1.35
def third_number := 5
def proportion (a b c d : ℚ) := a * d = b * c 

-- Proposition to prove
theorem find_third_number : proportion 0.75 x third_number 9 := 
by
  -- It's advisable to split the proof steps here, but the proof itself is condensed.
  sorry

end find_third_number_l10_10101


namespace mul_exponent_property_l10_10180

variable (m : ℕ)  -- Assuming m is a natural number for simplicity

theorem mul_exponent_property : m^2 * m^3 = m^5 := 
by {
  sorry
}

end mul_exponent_property_l10_10180


namespace entry_cost_proof_l10_10226

variable (hitting_rate : ℕ → ℝ)
variable (entry_cost : ℝ)
variable (total_hits : ℕ)
variable (money_lost : ℝ)

-- Conditions
axiom hitting_rate_condition : hitting_rate 200 = 0.025
axiom total_hits_condition : total_hits = 300
axiom money_lost_condition : money_lost = 7.5

-- Question: Prove that the cost to enter the contest equals $10.00
theorem entry_cost_proof : entry_cost = 10 := by
  sorry

end entry_cost_proof_l10_10226


namespace chord_angle_measure_l10_10368

theorem chord_angle_measure (AB_ratio : ℕ) (circ : ℝ) (h : AB_ratio = 1 + 5) : 
  ∃ θ : ℝ, θ = (1 / 6) * circ ∧ θ = 60 :=
by
  sorry

end chord_angle_measure_l10_10368


namespace factorize_difference_of_squares_l10_10483

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) :=
by
  sorry

end factorize_difference_of_squares_l10_10483


namespace oil_truck_radius_l10_10901

theorem oil_truck_radius
  (r_stationary : ℝ) (h_stationary : ℝ) (h_drop : ℝ) 
  (h_truck : ℝ)
  (V_pumped : ℝ) (π : ℝ) (r_truck : ℝ) :
  r_stationary = 100 → h_stationary = 25 → h_drop = 0.064 → h_truck = 10 →
  V_pumped = π * r_stationary^2 * h_drop →
  V_pumped = π * r_truck^2 * h_truck →
  r_truck = 8 := 
by 
  intros r_stationary_eq h_stationary_eq h_drop_eq h_truck_eq V_pumped_eq1 V_pumped_eq2
  sorry

end oil_truck_radius_l10_10901


namespace radius_of_given_spherical_circle_l10_10265
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end radius_of_given_spherical_circle_l10_10265


namespace union_sets_l10_10644

-- Definitions of sets A and B
def set_A : Set ℝ := {x | x / (x - 1) < 0}
def set_B : Set ℝ := {x | abs (1 - x) > 1 / 2}

-- The problem: prove that the union of sets A and B is (-∞, 1) ∪ (3/2, ∞)
theorem union_sets :
  set_A ∪ set_B = {x | x < 1} ∪ {x | x > 3 / 2} :=
by
  sorry

end union_sets_l10_10644


namespace find_value_of_expression_l10_10079

variable {a b c d x : ℝ}

-- Conditions
def opposites (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def abs_three (x : ℝ) : Prop := |x| = 3

-- Proof
theorem find_value_of_expression (h1 : opposites a b) (h2 : reciprocals c d) 
  (h3 : abs_three x) : ∃ res : ℝ, (res = 3 ∨ res = -3) ∧ res = 10 * a + 10 * b + c * d * x :=
by
  sorry

end find_value_of_expression_l10_10079


namespace night_shift_hours_l10_10045

theorem night_shift_hours
  (hours_first_guard : ℕ := 3)
  (hours_last_guard : ℕ := 2)
  (hours_each_middle_guard : ℕ := 2) :
  hours_first_guard + 2 * hours_each_middle_guard + hours_last_guard = 9 :=
by 
  sorry

end night_shift_hours_l10_10045


namespace line_passes_through_fixed_point_l10_10519

theorem line_passes_through_fixed_point (k : ℝ) : (k * 2 - 1 + 1 - 2 * k = 0) :=
by
  sorry

end line_passes_through_fixed_point_l10_10519


namespace box_volume_increase_l10_10328

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l10_10328


namespace longest_tape_length_l10_10449

/-!
  Problem: Find the length of the longest tape that can exactly measure the lengths 
  24 m, 36 m, and 54 m in cm.
  
  Solution: Convert the given lengths to the same unit (cm), then find their GCD.
  
  Given: Lengths are 2400 cm, 3600 cm, and 5400 cm.
  To Prove: gcd(2400, 3600, 5400) = 300.
-/

theorem longest_tape_length (a b c : ℕ) : a = 2400 → b = 3600 → c = 5400 → Nat.gcd (Nat.gcd a b) c = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- omitted proof steps
  sorry

end longest_tape_length_l10_10449


namespace quadratic_trinomial_has_two_roots_l10_10316

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l10_10316


namespace function_quadrants_l10_10210

theorem function_quadrants (a b : ℝ) (h_a : a > 1) (h_b : b < -1) :
  (∀ x : ℝ, a^x + b > 0 → ∃ x1 : ℝ, a^x1 + b < 0 → ∃ x2 : ℝ, a^x2 + b < 0) :=
sorry

end function_quadrants_l10_10210


namespace coupons_per_coloring_book_l10_10765

theorem coupons_per_coloring_book 
  (initial_books : ℝ) (books_sold : ℝ) (coupons_used : ℝ)
  (h1 : initial_books = 40) (h2 : books_sold = 20) (h3 : coupons_used = 80) : 
  (coupons_used / (initial_books - books_sold) = 4) :=
by 
  simp [*, sub_eq_add_neg]
  sorry

end coupons_per_coloring_book_l10_10765


namespace randy_brother_ate_l10_10690

-- Definitions
def initial_biscuits : ℕ := 32
def biscuits_from_father : ℕ := 13
def biscuits_from_mother : ℕ := 15
def remaining_biscuits : ℕ := 40

-- Theorem to prove
theorem randy_brother_ate : 
  initial_biscuits + biscuits_from_father + biscuits_from_mother - remaining_biscuits = 20 :=
by
  sorry

end randy_brother_ate_l10_10690


namespace probability_five_correct_l10_10146

theorem probability_five_correct :
  let num_people := 6
  let num_total_permutations := (Nat.factorial num_people)
  let num_five_correct := 0
  P(exactly_five_correct num_people) = num_five_correct / num_total_permutations :=
begin
  -- Definitions
  sorry
end

end probability_five_correct_l10_10146


namespace max_value_of_function_in_interval_l10_10015

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem max_value_of_function_in_interval :
  ∃ x ∈ Icc (-4 : ℝ) (4 : ℝ), f x = 10 ∧ ∀ y ∈ Icc (-4) (4), f y ≤ f x :=
by
  sorry -- Proof omitted

end max_value_of_function_in_interval_l10_10015


namespace find_a7_a8_l10_10222

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (g : geometric_sequence a q)

def sum_1_2 : ℝ := a 1 + a 2
def sum_3_4 : ℝ := a 3 + a 4

theorem find_a7_a8
  (h1 : sum_1_2 = 30)
  (h2 : sum_3_4 = 60)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 7 + a 8 = (a 1 + a 2) * (q ^ 6) := 
sorry

end find_a7_a8_l10_10222


namespace shaded_area_is_correct_l10_10832

-- Define the basic constants and areas
def grid_length : ℝ := 15
def grid_height : ℝ := 5
def total_grid_area : ℝ := grid_length * grid_height

def large_triangle_base : ℝ := 15
def large_triangle_height : ℝ := 3
def large_triangle_area : ℝ := 0.5 * large_triangle_base * large_triangle_height

def small_triangle_base : ℝ := 3
def small_triangle_height : ℝ := 4
def small_triangle_area : ℝ := 0.5 * small_triangle_base * small_triangle_height

-- Define the total shaded area
def shaded_area : ℝ := total_grid_area - large_triangle_area + small_triangle_area

-- Theorem stating that the shaded area is 58.5 square units
theorem shaded_area_is_correct : shaded_area = 58.5 := 
by 
  -- proof will be provided here
  sorry

end shaded_area_is_correct_l10_10832


namespace problem_solution_l10_10966

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)
variables (h3 : 3 * log 101 ((1030301 - a - b) / (3 * a * b)) = 3 - 2 * log 101 (a * b))

theorem problem_solution : 101 - (a)^(1/3) - (b)^(1/3) = 0 :=
by
  sorry

end problem_solution_l10_10966


namespace intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l10_10520

-- Definitions based on the conditions
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x - a * y + 2 = 0
def perpendicular (a : ℝ) : Prop := a = 0
def parallel (a : ℝ) : Prop := a = 1 ∨ a = -1

-- Theorem 1: Intersection point when a = 0 is (-2, 2)
theorem intersection_point_zero_a_0 :
  ∀ x y : ℝ, l₁ 0 x y → l₂ 0 x y → (x, y) = (-2, 2) := 
by
  sorry

-- Theorem 2: Line l₁ always passes through (0, 2)
theorem l₁_passes_through_0_2 :
  ∀ a : ℝ, l₁ a 0 2 := 
by
  sorry

-- Theorem 3: l₁ is perpendicular to l₂ implies a = 0
theorem l₁_perpendicular_l₂ :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ∀ m n, (a * m + (n / a) = 0)) → (a = 0) :=
by
  sorry

-- Theorem 4: l₁ is parallel to l₂ implies a = 1 or a = -1
theorem l₁_parallel_l₂ :
  ∀ a : ℝ, parallel a → (a = 1 ∨ a = -1) :=
by
  sorry

end intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l10_10520


namespace time_saved_correct_l10_10845

-- Define the conditions as constants
def section1_problems : Nat := 20
def section2_problems : Nat := 15

def time_with_calc_sec1 : Nat := 3
def time_without_calc_sec1 : Nat := 8

def time_with_calc_sec2 : Nat := 5
def time_without_calc_sec2 : Nat := 10

-- Calculate the total times
def total_time_with_calc : Nat :=
  (section1_problems * time_with_calc_sec1) +
  (section2_problems * time_with_calc_sec2)

def total_time_without_calc : Nat :=
  (section1_problems * time_without_calc_sec1) +
  (section2_problems * time_without_calc_sec2)

-- The time saved using a calculator
def time_saved : Nat :=
  total_time_without_calc - total_time_with_calc

-- State the proof problem
theorem time_saved_correct :
  time_saved = 175 := by
  sorry

end time_saved_correct_l10_10845


namespace domain_of_f_l10_10084

noncomputable def f (x : ℝ) : ℝ :=
  (x - 4)^0 + Real.sqrt (2 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (1 < x ∧ x < 4) ∨ (4 < x) ↔
    ∃ y : ℝ, f y = f x :=
sorry

end domain_of_f_l10_10084


namespace vec_expression_l10_10938

def vec_a : ℝ × ℝ := (1, -2)
def vec_b : ℝ × ℝ := (3, 5)

theorem vec_expression : 2 • vec_a + vec_b = (5, 1) := by
  sorry

end vec_expression_l10_10938


namespace problem1_problem2_l10_10474

theorem problem1 (a b : ℝ) : (-(2 : ℝ) * a ^ 2 * b) ^ 3 / (-(2 * a * b)) * (1 / 3 * a ^ 2 * b ^ 3) = (4 / 3) * a ^ 7 * b ^ 5 :=
  by
  sorry

theorem problem2 (x : ℝ) : (27 * x ^ 3 + 18 * x ^ 2 - 3 * x) / -3 * x = -9 * x ^ 2 - 6 * x + 1 :=
  by
  sorry

end problem1_problem2_l10_10474


namespace triangle_angles_l10_10281

noncomputable def angle_triangle (E : ℝ) :=
if E = 45 then (90, 45, 45) else if E = 36 then (72, 72, 36) else (0, 0, 0)

theorem triangle_angles (E : ℝ) :
  (∃ E, E = 45 → angle_triangle E = (90, 45, 45))
  ∨
  (∃ E, E = 36 → angle_triangle E = (72, 72, 36)) :=
by
    sorry

end triangle_angles_l10_10281


namespace probability_red_or_white_is_19_over_25_l10_10161

-- Definitions for the conditions
def totalMarbles : ℕ := 50
def blueMarbles : ℕ := 12
def redMarbles : ℕ := 18
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the probability calculation
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- The theorem we need to prove
theorem probability_red_or_white_is_19_over_25 :
  probabilityRedOrWhite = 19 / 25 :=
by
  -- Sorry to skip the proof
  sorry

end probability_red_or_white_is_19_over_25_l10_10161


namespace diana_principal_charge_l10_10918

theorem diana_principal_charge :
  ∃ P : ℝ, P > 0 ∧ (P + P * 0.06 = 63.6) ∧ P = 60 :=
by
  use 60
  sorry

end diana_principal_charge_l10_10918


namespace total_donation_correct_l10_10091

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end total_donation_correct_l10_10091


namespace coin_toss_dice_roll_l10_10733

theorem coin_toss_dice_roll :
  let coin_toss := 2 -- two outcomes for same side coin toss
  let dice_roll := 2 -- two outcomes for multiple of 3 on dice roll
  coin_toss * dice_roll = 4 :=
by
  sorry

end coin_toss_dice_roll_l10_10733


namespace pythagorean_theorem_l10_10000

theorem pythagorean_theorem {a b c p q : ℝ} 
  (h₁ : p * c = a ^ 2) 
  (h₂ : q * c = b ^ 2)
  (h₃ : p + q = c) : 
  c ^ 2 = a ^ 2 + b ^ 2 := 
by 
  sorry

end pythagorean_theorem_l10_10000


namespace equal_area_condition_l10_10108

variable {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2)

theorem equal_area_condition : 2 * θ = (Real.tan θ) * (Real.tan (2 * θ)) :=
by {
  sorry
}

end equal_area_condition_l10_10108


namespace tens_digit_of_7_pow_2011_l10_10882

-- Define the conditions for the problem
def seven_power := 7
def exponent := 2011
def modulo := 100

-- Define the target function to find the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem formally
theorem tens_digit_of_7_pow_2011 : tens_digit (seven_power ^ exponent % modulo) = 4 := by
  sorry

end tens_digit_of_7_pow_2011_l10_10882


namespace kendra_change_is_correct_l10_10721

-- Define the initial conditions
def price_wooden_toy : ℕ := 20
def price_hat : ℕ := 10
def kendra_initial_money : ℕ := 100
def num_wooden_toys : ℕ := 2
def num_hats : ℕ := 3

-- Calculate the total costs
def total_wooden_toys_cost : ℕ := price_wooden_toy * num_wooden_toys
def total_hats_cost : ℕ := price_hat * num_hats
def total_cost : ℕ := total_wooden_toys_cost + total_hats_cost

-- Calculate the change Kendra received
def kendra_change : ℕ := kendra_initial_money - total_cost

theorem kendra_change_is_correct : kendra_change = 30 := by
  sorry

end kendra_change_is_correct_l10_10721


namespace total_cost_of_barbed_wire_l10_10710

noncomputable def cost_of_barbed_wire : ℝ :=
  let area : ℝ := 3136
  let side_length : ℝ := Real.sqrt area
  let perimeter_without_gates : ℝ := 4 * side_length - 2 * 1
  let rate_per_meter : ℝ := 1.10
  perimeter_without_gates * rate_per_meter

theorem total_cost_of_barbed_wire :
  cost_of_barbed_wire = 244.20 :=
sorry

end total_cost_of_barbed_wire_l10_10710


namespace quadratic_equation_root_zero_l10_10658

/-- Given that x = -3 is a root of the quadratic equation x^2 + 3x + k = 0,
    prove that the other root of the equation is 0 and k = 0. -/
theorem quadratic_equation_root_zero (k : ℝ) (h : -3^2 + 3 * -3 + k = 0) :
  (∀ t : ℝ, t^2 + 3 * t + k = 0 → t = 0) ∧ k = 0 :=
sorry

end quadratic_equation_root_zero_l10_10658


namespace camera_pictures_olivia_camera_pictures_l10_10685

theorem camera_pictures (phone_pics : Nat) (albums : Nat) (pics_per_album : Nat) (total_pics : Nat) : Prop :=
  phone_pics = 5 →
  albums = 8 →
  pics_per_album = 5 →
  total_pics = albums * pics_per_album →
  total_pics - phone_pics = 35

-- Here's the statement of the theorem followed by a sorry to indicate that the proof is not provided
theorem olivia_camera_pictures (phone_pics albums pics_per_album total_pics : Nat) (h1 : phone_pics = 5) (h2 : albums = 8) (h3 : pics_per_album = 5) (h4 : total_pics = albums * pics_per_album) : total_pics - phone_pics = 35 :=
by
  sorry

end camera_pictures_olivia_camera_pictures_l10_10685


namespace range_of_x_l10_10072

variable (x y : ℝ)

theorem range_of_x (h1 : 2 * x - y = 4) (h2 : -2 < y ∧ y ≤ 3) :
  1 < x ∧ x ≤ 7 / 2 :=
  sorry

end range_of_x_l10_10072


namespace max_min_K_max_min_2x_plus_y_l10_10201

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

theorem max_min_K (x y : ℝ) (h : circle_equation x y) : 
  - (Real.sqrt 3) / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ (Real.sqrt 3) / 3 :=
by sorry

theorem max_min_2x_plus_y (x y : ℝ) (h : circle_equation x y) :
  1 - Real.sqrt 5 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + Real.sqrt 5 :=
by sorry

end max_min_K_max_min_2x_plus_y_l10_10201


namespace largest_integer_same_cost_l10_10026

def sum_decimal_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_ternary_digits (n : ℕ) : ℕ :=
  n.digits 3 |>.sum

theorem largest_integer_same_cost :
  ∃ n : ℕ, n < 1000 ∧ sum_decimal_digits n = sum_ternary_digits n ∧ ∀ m : ℕ, m < 1000 ∧ sum_decimal_digits m = sum_ternary_digits m → m ≤ n := 
  sorry

end largest_integer_same_cost_l10_10026


namespace has_two_roots_l10_10319

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l10_10319


namespace simplify_fraction_144_12672_l10_10560

theorem simplify_fraction_144_12672 : (144 / 12672 : ℚ) = 1 / 88 :=
by
  sorry

end simplify_fraction_144_12672_l10_10560


namespace division_by_power_of_ten_l10_10600

theorem division_by_power_of_ten (a b : ℕ) (h_a : a = 10^7) (h_b : b = 5 * 10^4) : a / b = 200 := by
  sorry

end division_by_power_of_ten_l10_10600


namespace new_volume_increased_dimensions_l10_10323

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l10_10323


namespace find_x_l10_10360

theorem find_x (x y : ℕ) (h1 : y = 30) (h2 : x / y = 5 / 2) : x = 75 := by
  sorry

end find_x_l10_10360


namespace rajesh_walked_distance_l10_10247

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end rajesh_walked_distance_l10_10247


namespace min_distance_AB_tangent_line_circle_l10_10814

theorem min_distance_AB_tangent_line_circle 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h_tangent : a^2 + b^2 = 1) :
  ∃ A B : ℝ × ℝ, (A = (0, 1/b) ∧ B = (2/a, 0)) ∧ dist A B = 3 :=
by
  sorry

end min_distance_AB_tangent_line_circle_l10_10814


namespace magician_starting_decks_l10_10461

def starting_decks (price_per_deck earned remaining_decks : ℕ) : ℕ :=
  earned / price_per_deck + remaining_decks

theorem magician_starting_decks :
  starting_decks 2 4 3 = 5 :=
by
  sorry

end magician_starting_decks_l10_10461


namespace unique_solution_for_y_l10_10347

def operation (x y : ℝ) : ℝ := 4 * x - 2 * y + x^2 * y

theorem unique_solution_for_y : ∃! (y : ℝ), operation 3 y = 20 :=
by {
  sorry
}

end unique_solution_for_y_l10_10347


namespace cost_of_milk_l10_10929

theorem cost_of_milk (x : ℝ) (h1 : 10 * 0.1 = 1) (h2 : 11 = 1 + x + 3 * x) : x = 2.5 :=
by 
  sorry

end cost_of_milk_l10_10929


namespace bart_earnings_l10_10777

theorem bart_earnings :
  let payment_per_question := 0.2 in
  let questions_per_survey := 10 in
  let surveys_monday := 3 in
  let surveys_tuesday := 4 in
  (surveys_monday * questions_per_survey + surveys_tuesday * questions_per_survey) * payment_per_question = 14 :=
by
  sorry

end bart_earnings_l10_10777


namespace quadratic_trinomial_has_two_roots_l10_10309

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l10_10309


namespace infinitely_many_solutions_eq_l10_10534

theorem infinitely_many_solutions_eq {a b : ℝ} 
  (H : ∀ x : ℝ, a * (a - x) - b * (b - x) = 0) : a = b :=
sorry

end infinitely_many_solutions_eq_l10_10534


namespace average_score_first_10_matches_l10_10425

theorem average_score_first_10_matches (A : ℕ) 
  (h1 : 0 < A) 
  (h2 : 10 * A + 15 * 70 = 25 * 66) : A = 60 :=
by
  sorry

end average_score_first_10_matches_l10_10425


namespace piecewise_function_identity_l10_10785

theorem piecewise_function_identity (x : ℝ) : 
  (3 * x + abs (5 * x - 10)) = if x < 2 then -2 * x + 10 else 8 * x - 10 := by
  sorry

end piecewise_function_identity_l10_10785


namespace circle_through_intersections_and_point_l10_10188

-- Definitions of given circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Given point (1, -1)
def P1 : (ℝ × ℝ) := (1, -1)

-- Proof problem statement
theorem circle_through_intersections_and_point : 
  ∃ (λ : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 4 * y + λ * (x^2 + y^2 - x) = 0) → 
     (C1 x y) ∧ (C2 x y)) ∧
    (let (a, b) := P1 in (a^2 + b^2 + 4 * a - 4 * b + λ * (a^2 + b^2 - a) = 0)) ∧ 
    (9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0) := sorry

end circle_through_intersections_and_point_l10_10188


namespace thirteen_pow_2023_mod_1000_l10_10455

theorem thirteen_pow_2023_mod_1000 :
  (13^2023) % 1000 = 99 :=
sorry

end thirteen_pow_2023_mod_1000_l10_10455


namespace xy_solutions_l10_10070

theorem xy_solutions : 
  ∀ (x y : ℕ), 0 < x → 0 < y →
  (xy ^ 2 + 7) ∣ (x^2 * y + x) →
  (x, y) = (7, 1) ∨ (x, y) = (14, 1) ∨ (x, y) = (35, 1) ∨ (x, y) = (7, 2) ∨ (∃ k : ℕ, x = 7 * k ∧ y = 7) :=
by
  sorry

end xy_solutions_l10_10070


namespace z_gets_amount_per_unit_l10_10465

-- Define the known conditions
variables (x y z : ℝ)
variables (x_share : ℝ)
variables (y_share : ℝ)
variables (z_share : ℝ)
variables (total : ℝ)

-- Assume the conditions given in the problem
axiom h1 : y_share = 54
axiom h2 : total = 234
axiom h3 : (y / x) = 0.45
axiom h4 : total = x_share + y_share + z_share

-- Prove the target statement
theorem z_gets_amount_per_unit : ((z_share / x_share) = 0.50) :=
by
  sorry

end z_gets_amount_per_unit_l10_10465


namespace solve_x_values_l10_10794

theorem solve_x_values (x : ℝ) :
  (5 + x) / (7 + x) = (2 + x^2) / (4 + x) ↔ x = 1 ∨ x = -2 ∨ x = -3 := 
sorry

end solve_x_values_l10_10794


namespace shaded_area_of_rotated_square_is_four_thirds_l10_10332

noncomputable def common_shaded_area_of_rotated_square (β : ℝ) (h1 : 0 < β) (h2 : β < π / 2) (h_cos_beta : Real.cos β = 3 / 5) : ℝ :=
  let side_length := 2
  let area := side_length * side_length / 3 * 2
  area

theorem shaded_area_of_rotated_square_is_four_thirds
  (β : ℝ)
  (h1 : 0 < β)
  (h2 : β < π / 2)
  (h_cos_beta : Real.cos β = 3 / 5) :
  common_shaded_area_of_rotated_square β h1 h2 h_cos_beta = 4 / 3 :=
sorry

end shaded_area_of_rotated_square_is_four_thirds_l10_10332


namespace find_a_l10_10289

theorem find_a (a x : ℝ) (h1: a - 2 ≤ x) (h2: x ≤ a + 1) (h3 : -x^2 + 2 * x + 3 = 3) :
  a = 2 := sorry

end find_a_l10_10289


namespace triangle_side_sum_l10_10735

theorem triangle_side_sum (A B C : Type) [geometry : Triangle A B C] [Angle A = 50] [Angle C = 40]
  (side_opposite_C : length (side B A) = 8 * real.sqrt 3) :
  length (side A B) + length (side B C) = 59.5 :=
by sorry

end triangle_side_sum_l10_10735


namespace dabbies_turkey_cost_l10_10477

noncomputable def first_turkey_weight : ℕ := 6
noncomputable def second_turkey_weight : ℕ := 9
noncomputable def third_turkey_weight : ℕ := 2 * second_turkey_weight
noncomputable def cost_per_kg : ℕ := 2

noncomputable def total_cost : ℕ :=
  first_turkey_weight * cost_per_kg +
  second_turkey_weight * cost_per_kg +
  third_turkey_weight * cost_per_kg

theorem dabbies_turkey_cost : total_cost = 66 :=
by
  sorry

end dabbies_turkey_cost_l10_10477


namespace probability_third_smallest_is_4_l10_10693

theorem probability_third_smallest_is_4 :
  (∃ (integers : Finset ℕ), integers.card = 7 ∧ integers ⊆ (Finset.range 13).erase 0 ∧ 
  ∃ (S : Finset ℕ), S = (Finset.filter (λ x, x < 4) integers) ∧ S.card = 2 ∧ 
  ∃ (T : Finset ℕ), T = (Finset.filter (λ x, 4 < x) integers) ∧ T.card = 5) → 
  let total_ways := Nat.choose 12 7 in
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 5) in
  (favorable_ways) / total_ways.toReal = 7 / 33 :=
by sorry

end probability_third_smallest_is_4_l10_10693


namespace find_m_if_perpendicular_l10_10204

theorem find_m_if_perpendicular 
  (m : ℝ)
  (h : ∀ m (slope1 : ℝ) (slope2 : ℝ), 
    (slope1 = -m) → 
    (slope2 = (-1) / (3 - 2 * m)) → 
    slope1 * slope2 = -1)
  : m = 3 := 
by
  sorry

end find_m_if_perpendicular_l10_10204


namespace midpoint_trajectory_of_intersecting_line_l10_10514

theorem midpoint_trajectory_of_intersecting_line 
    (h₁ : ∀ x y, x^2 + 2 * y^2 = 4) 
    (h₂ : ∀ M: ℝ × ℝ, M = (4, 6)) :
    ∃ x y, (x-2)^2 / 22 + (y-3)^2 / 11 = 1 :=
sorry

end midpoint_trajectory_of_intersecting_line_l10_10514


namespace distance_ratio_l10_10278

variable (d_RB d_BC : ℝ)

theorem distance_ratio
    (h1 : d_RB / 60 + d_BC / 20 ≠ 0)
    (h2 : 36 * (d_RB / 60 + d_BC / 20) = d_RB + d_BC) : 
    d_RB / d_BC = 2 := 
sorry

end distance_ratio_l10_10278


namespace triangle_area_is_correct_l10_10027

-- Define the points
def point1 : (ℝ × ℝ) := (0, 3)
def point2 : (ℝ × ℝ) := (5, 0)
def point3 : (ℝ × ℝ) := (0, 6)
def point4 : (ℝ × ℝ) := (4, 0)

-- Define a function to calculate the area based on the intersection points
noncomputable def area_of_triangle (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let slope1 := (p2.2 - p1.2) / (p2.1 - p1.1)
  let intercept1 := p1.2 - slope1 * p1.1
  let slope2 := (p4.2 - p3.2) / (p4.1 - p3.1)
  let intercept2 := p3.2 - slope2 * p3.1
  let x_intersect := (intercept2 - intercept1) / (slope1 - slope2)
  let y_intersect := slope1 * x_intersect + intercept1
  let base := x_intersect
  let height := y_intersect
  (1 / 2) * base * height

-- The proof problem statement in Lean
theorem triangle_area_is_correct :
  area_of_triangle point1 point2 point3 point4 = 5 / 3 :=
by
  sorry

end triangle_area_is_correct_l10_10027


namespace rotation_matrix_det_75_degrees_l10_10227

open Matrix Real

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ; sin θ, cos θ]

theorem rotation_matrix_det_75_degrees :
  let S := rotation_matrix (75 * π / 180) in det S = 1 :=
by
  let θ := (75 * π / 180) -- defining 75 degrees in radians
  let S := rotation_matrix θ
  have hS : S = !![cos θ, -sin θ; sin θ, cos θ] := rfl
  have hdet : det S = cos θ * cos θ + sin θ * sin θ := Matrix.det_fin_two
  have pythagorean_identity : cos θ * cos θ + sin θ * sin θ = 1 := sorry
  rw [hdet, pythagorean_identity]
  exact rfl

end rotation_matrix_det_75_degrees_l10_10227


namespace fifteen_horses_fifteen_bags_l10_10749

-- Definitions based on the problem
def days_for_one_horse_one_bag : ℝ := 1  -- It takes 1 day for 1 horse to eat 1 bag of grain

-- Theorem statement
theorem fifteen_horses_fifteen_bags {d : ℝ} (h : d = days_for_one_horse_one_bag) :
  d = 1 :=
by
  sorry

end fifteen_horses_fifteen_bags_l10_10749


namespace avg_speed_while_climbing_l10_10411

-- Definitions for conditions
def totalClimbTime : ℝ := 4
def restBreaks : ℝ := 0.5
def descentTime : ℝ := 2
def avgSpeedWholeJourney : ℝ := 1.5
def totalDistance : ℝ := avgSpeedWholeJourney * (totalClimbTime + descentTime)

-- The question: Prove Natasha's average speed while climbing to the top, excluding the rest breaks duration.
theorem avg_speed_while_climbing :
  (totalDistance / 2) / (totalClimbTime - restBreaks) = 1.29 := 
sorry

end avg_speed_while_climbing_l10_10411


namespace selecting_female_probability_l10_10220

theorem selecting_female_probability (female male : ℕ) (total : ℕ)
  (h_female : female = 4)
  (h_male : male = 6)
  (h_total : total = female + male) :
  (female / total : ℚ) = 2 / 5 := 
by
  -- Insert proof steps here
  sorry

end selecting_female_probability_l10_10220


namespace three_pair_probability_l10_10582

theorem three_pair_probability :
  let total_combinations := Nat.choose 52 5
  let three_pair_combinations := 13 * 4 * 12 * 4
  total_combinations = 2598960 ∧ three_pair_combinations = 2496 →
  (three_pair_combinations : ℚ) / total_combinations = 2496 / 2598960 :=
by
  -- Definitions and computations can be added here if necessary
  sorry

end three_pair_probability_l10_10582


namespace actors_in_one_hour_l10_10107

theorem actors_in_one_hour (actors_per_set : ℕ) (minutes_per_set : ℕ) (total_minutes : ℕ) :
  actors_per_set = 5 → minutes_per_set = 15 → total_minutes = 60 →
  (total_minutes / minutes_per_set) * actors_per_set = 20 :=
by
  intros h1 h2 h3
  sorry

end actors_in_one_hour_l10_10107


namespace simplify_expression_l10_10417

open Nat

theorem simplify_expression (x : ℤ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x :=
by
  sorry

end simplify_expression_l10_10417


namespace marcy_drinks_in_250_minutes_l10_10117

-- Define a function to represent that Marcy takes n minutes to drink x liters of water.
def time_to_drink (minutes_per_sip : ℕ) (sip_volume_ml : ℕ) (total_volume_liters : ℕ) : ℕ :=
  let total_volume_ml := total_volume_liters * 1000
  let sips := total_volume_ml / sip_volume_ml
  sips * minutes_per_sip

theorem marcy_drinks_in_250_minutes :
  time_to_drink 5 40 2 = 250 :=
  by
    -- The function definition and its application will show this value holds.
    sorry

end marcy_drinks_in_250_minutes_l10_10117


namespace students_before_Yoongi_l10_10170

theorem students_before_Yoongi (total_students : ℕ) (students_after_Yoongi : ℕ) 
  (condition1 : total_students = 20) (condition2 : students_after_Yoongi = 11) :
  total_students - students_after_Yoongi - 1 = 8 :=
by 
  sorry

end students_before_Yoongi_l10_10170


namespace opposite_direction_of_vectors_l10_10087

theorem opposite_direction_of_vectors
  (x : ℝ)
  (a : ℝ × ℝ := (x, 1))
  (b : ℝ × ℝ := (4, x)) :
  (∃ k : ℝ, k ≠ 0 ∧ a = -k • b) → x = -2 := 
sorry

end opposite_direction_of_vectors_l10_10087


namespace probability_no_obtuse_triangle_correct_l10_10495

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l10_10495


namespace fraction_ratio_l10_10376

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end fraction_ratio_l10_10376


namespace abs_le_and_interval_iff_l10_10454

variable (x : ℝ)

theorem abs_le_and_interval_iff :
  (|x - 2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end abs_le_and_interval_iff_l10_10454


namespace graph_of_f_4_minus_x_l10_10821

theorem graph_of_f_4_minus_x (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 :=
by
  rw [sub_self]
  exact h

end graph_of_f_4_minus_x_l10_10821


namespace sum_of_consecutive_odds_eq_169_l10_10739

theorem sum_of_consecutive_odds_eq_169 : 
  ∃ n : ℕ, (∑ i in Finset.range n, 2 * i + 1) = 169 ∧ (2 * n - 1) = 25 := 
by
  sorry

end sum_of_consecutive_odds_eq_169_l10_10739


namespace satisfy_eqn_l10_10485

/-- 
  Prove that the integer pairs (0, 1), (0, -1), (1, 0), (-1, 0), (2, 2), (-2, -2)
  are the only pairs that satisfy x^5 + y^5 = (x + y)^3
-/
theorem satisfy_eqn (x y : ℤ) : 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (1, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (2, 2) ∨ (x, y) = (-2, -2) ↔ 
  x^5 + y^5 = (x + y)^3 := 
by 
  sorry

end satisfy_eqn_l10_10485


namespace cricket_players_count_l10_10665

theorem cricket_players_count (Hockey Football Softball Total Cricket : ℕ) 
    (hHockey : Hockey = 12)
    (hFootball : Football = 18)
    (hSoftball : Softball = 13)
    (hTotal : Total = 59)
    (hTotalCalculation : Total = Hockey + Football + Softball + Cricket) : 
    Cricket = 16 := by
  sorry

end cricket_players_count_l10_10665


namespace period_of_f_l10_10800

noncomputable def f : ℝ → ℝ := sorry

def functional_equation (f : ℝ → ℝ) := ∀ x y : ℝ, f (2 * x) + f (2 * y) = f (x + y) * f (x - y)

def f_pi_zero (f : ℝ → ℝ) := f (Real.pi) = 0

def f_not_identically_zero (f : ℝ → ℝ) := ∃ x : ℝ, f x ≠ 0

theorem period_of_f (f : ℝ → ℝ)
  (hf_eq : functional_equation f)
  (hf_pi_zero : f_pi_zero f)
  (hf_not_zero : f_not_identically_zero f) : 
  ∀ x : ℝ, f (x + 4 * Real.pi) = f x := sorry

end period_of_f_l10_10800


namespace PQ_ratio_l10_10006

-- Definitions
def hexagon_area : ℕ := 7
def base_of_triangle : ℕ := 4

-- Conditions
def PQ_bisects_area (A : ℕ) : Prop :=
  A = hexagon_area / 2

def area_below_PQ (U T : ℚ) : Prop :=
  U + T = hexagon_area / 2 ∧ U = 1

def triangle_area (T b : ℚ) : ℚ :=
  1/2 * b * (5/4)

def XQ_QY_ratio (XQ QY : ℚ) : ℚ :=
  XQ / QY

-- Theorem Statement
theorem PQ_ratio (XQ QY : ℕ) (h1 : PQ_bisects_area (hexagon_area / 2))
  (h2 : area_below_PQ 1 (triangle_area (5/2) base_of_triangle))
  (h3 : XQ + QY = base_of_triangle) : XQ_QY_ratio XQ QY = 1 := sorry

end PQ_ratio_l10_10006


namespace break_even_production_volume_l10_10758

theorem break_even_production_volume
  (Q : ℕ) 
  (ATC : ℕ → ℚ)
  (P : ℚ)
  (h1 : ∀ Q, ATC Q = 100 + 100000 / Q)
  (h2 : P = 300) :
  ATC 500 = P :=
by
  sorry

end break_even_production_volume_l10_10758


namespace apples_difference_l10_10835

theorem apples_difference 
  (father_apples : ℕ := 8)
  (mother_apples : ℕ := 13)
  (jungkook_apples : ℕ := 7)
  (brother_apples : ℕ := 5) :
  max father_apples (max mother_apples (max jungkook_apples brother_apples)) - 
  min father_apples (min mother_apples (min jungkook_apples brother_apples)) = 8 :=
by
  sorry

end apples_difference_l10_10835


namespace mean_of_three_numbers_l10_10423

theorem mean_of_three_numbers (a : Fin 12 → ℕ) (x y z : ℕ) 
  (h1 : (Finset.univ.sum a) / 12 = 40)
  (h2 : ((Finset.univ.sum a) + x + y + z) / 15 = 50) :
  (x + y + z) / 3 = 90 := 
by
  sorry

end mean_of_three_numbers_l10_10423


namespace sum_n_k_l10_10974

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l10_10974


namespace vector_subtraction_l10_10671

def a : ℝ × ℝ × ℝ := (1, -2, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 2)

theorem vector_subtraction : a - b = (0, -2, -1) := 
by 
  unfold a b
  simp
  sorry

end vector_subtraction_l10_10671


namespace chess_team_selection_l10_10410

theorem chess_team_selection:
  let boys := 10
  let girls := 12
  let team_size := 8     -- total team size
  let boys_selected := 5 -- number of boys to select
  let girls_selected := 3 -- number of girls to select
  ∃ (w : ℕ), 
  (w = Nat.choose boys boys_selected * Nat.choose girls girls_selected) ∧ 
  w = 55440 :=
by
  sorry

end chess_team_selection_l10_10410


namespace cash_price_of_television_l10_10683

variable (DownPayment : ℕ := 120)
variable (MonthlyPayment : ℕ := 30)
variable (NumberOfMonths : ℕ := 12)
variable (Savings : ℕ := 80)

-- Define the total installment cost
def TotalInstallment := DownPayment + MonthlyPayment * NumberOfMonths

-- The main statement to prove
theorem cash_price_of_television : (TotalInstallment - Savings) = 400 := by
  sorry

end cash_price_of_television_l10_10683


namespace range_of_m_l10_10643

theorem range_of_m (m : ℝ) 
  (p : m < 0) 
  (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) : 
  -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l10_10643


namespace floor_plus_ceil_eq_seven_l10_10059

theorem floor_plus_ceil_eq_seven (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end floor_plus_ceil_eq_seven_l10_10059


namespace remainder_of_x_pow_150_div_by_x_minus_1_cubed_l10_10355

theorem remainder_of_x_pow_150_div_by_x_minus_1_cubed :
  (x : ℤ) → (x^150 % (x - 1)^3) = (11175 * x^2 - 22200 * x + 11026) :=
by
  intro x
  sorry

end remainder_of_x_pow_150_div_by_x_minus_1_cubed_l10_10355


namespace find_a5_find_a31_div_a29_l10_10624

noncomputable def geo_diff_seq (a : ℕ → ℕ) (d : ℕ) :=
∀ n : ℕ, n > 0 → (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem find_a5 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 5 = 105 :=
sorry

theorem find_a31_div_a29 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 31 / a 29 = 3363 :=
sorry

end find_a5_find_a31_div_a29_l10_10624


namespace inequality_hold_l10_10554

theorem inequality_hold (n : ℕ) (h1 : n > 1) : 1 + n * 2^((n - 1 : ℕ) / 2) < 2^n :=
by
  sorry

end inequality_hold_l10_10554


namespace sum_possible_n_k_l10_10980

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l10_10980


namespace factorial_power_of_two_l10_10158

theorem factorial_power_of_two solutions (a b c : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_equation : a.factorial + b.factorial = 2^(c.factorial)) :
  solutions = [(1, 1, 1), (2, 2, 2)] :=
sorry

end factorial_power_of_two_l10_10158


namespace smallest_number_of_students_in_debate_club_l10_10984

-- Define conditions
def ratio_8th_to_6th (x₈ x₆ : ℕ) : Prop := 7 * x₆ = 4 * x₈
def ratio_8th_to_7th (x₈ x₇ : ℕ) : Prop := 6 * x₇ = 5 * x₈
def ratio_8th_to_9th (x₈ x₉ : ℕ) : Prop := 9 * x₉ = 2 * x₈

-- Problem statement
theorem smallest_number_of_students_in_debate_club 
  (x₈ x₆ x₇ x₉ : ℕ) 
  (h₁ : ratio_8th_to_6th x₈ x₆) 
  (h₂ : ratio_8th_to_7th x₈ x₇) 
  (h₃ : ratio_8th_to_9th x₈ x₉) : 
  x₈ + x₆ + x₇ + x₉ = 331 := 
sorry

end smallest_number_of_students_in_debate_club_l10_10984


namespace four_digit_integer_product_l10_10788

theorem four_digit_integer_product :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
  a^2 + b^2 + c^2 + d^2 = 65 ∧ a * b * c * d = 140 :=
by
  sorry

end four_digit_integer_product_l10_10788


namespace equal_intercepts_on_both_axes_l10_10515

theorem equal_intercepts_on_both_axes (m : ℝ) :
  (5 - 2 * m ≠ 0) ∧
  (- (5 - 2 * m) / (m^2 - 2 * m - 3) = - (5 - 2 * m) / (2 * m^2 + m - 1)) ↔ m = -2 :=
by sorry

end equal_intercepts_on_both_axes_l10_10515


namespace three_liters_to_gallons_l10_10200

theorem three_liters_to_gallons :
  (0.5 : ℝ) * 3 * 0.1319 = 0.7914 := by
  sorry

end three_liters_to_gallons_l10_10200


namespace alice_oranges_proof_l10_10617

-- Definitions for conditions
def oranges_emily_sold (E : ℕ) := E
def oranges_alice_sold (E : ℕ) := 2 * E
def total_oranges_sold (E : ℕ) := E + 2 * E

-- Proof statement
theorem alice_oranges_proof : ∀ E : ℕ, total_oranges_sold E = 180 → oranges_alice_sold E = 120 :=
begin
  intros E h,
  sorry
end

end alice_oranges_proof_l10_10617


namespace point_on_curve_l10_10847

-- Define the parametric curve equations
def onCurve (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.sin (2 * θ) ∧ y = Real.cos θ + Real.sin θ

-- Define the general form of the curve
def curveEquation (x y : ℝ) : Prop :=
  y^2 = 1 + x

-- The proof statement
theorem point_on_curve : 
  curveEquation (-3/4) (1/2) ∧ ∃ θ : ℝ, onCurve θ (-3/4) (1/2) :=
by
  sorry

end point_on_curve_l10_10847


namespace same_solution_set_l10_10587

theorem same_solution_set :
  (∀ x : ℝ, (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0) :=
sorry

end same_solution_set_l10_10587


namespace math_vs_english_time_difference_l10_10396

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l10_10396


namespace shirts_per_minute_l10_10471

theorem shirts_per_minute (shirts_in_6_minutes : ℕ) (time_minutes : ℕ) (h1 : shirts_in_6_minutes = 36) (h2 : time_minutes = 6) : 
  ((shirts_in_6_minutes / time_minutes) = 6) :=
by
  sorry

end shirts_per_minute_l10_10471


namespace probability_no_obtuse_triangle_correct_l10_10493

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l10_10493


namespace joan_exam_time_difference_l10_10401

theorem joan_exam_time_difference :
  (let english_questions := 30
       math_questions := 15
       english_time_hours := 1
       math_time_hours := 1.5
       english_time_minutes := english_time_hours * 60
       math_time_minutes := math_time_hours * 60
       time_per_english_question := english_time_minutes / english_questions
       time_per_math_question := math_time_minutes / math_questions
    in time_per_math_question - time_per_english_question = 4) :=
by
  sorry

end joan_exam_time_difference_l10_10401


namespace find_x_l10_10100

variable (x y : ℝ)

theorem find_x (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end find_x_l10_10100


namespace post_spacing_change_l10_10029

theorem post_spacing_change :
  ∀ (posts : ℕ → ℝ) (constant_spacing : ℝ), 
  (∀ n, 1 ≤ n ∧ n < 16 → posts (n + 1) - posts n = constant_spacing) →
  posts 16 - posts 1 = 48 → 
  posts 28 - posts 16 = 36 →
  ∃ (k : ℕ), 16 < k ∧ k ≤ 28 ∧ posts (k + 1) - posts k ≠ constant_spacing ∧ posts (k + 1) - posts k = 2.9 ∧ k = 20 := 
  sorry

end post_spacing_change_l10_10029


namespace find_three_digit_number_in_decimal_l10_10254

theorem find_three_digit_number_in_decimal :
  ∃ (A B C : ℕ), ∀ (hA : A ≠ 0 ∧ A < 7) (hB : B ≠ 0 ∧ B < 7) (hC : C ≠ 0 ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (h1 : (7 * A + B) + C = 7 * C)
    (h2 : (7 * A + B) + (7 * B + A) = 7 * B + 6), 
    A * 100 + B * 10 + C = 425 :=
by
  sorry

end find_three_digit_number_in_decimal_l10_10254


namespace max_area_right_triangle_in_semicircle_l10_10533

theorem max_area_right_triangle_in_semicircle :
  ∀ (r : ℝ), r = 1/2 → 
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ y > 0 ∧ 
  (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 ∧ y' > 0 → (1/2) * x * y ≥ (1/2) * x' * y') ∧ 
  (1/2) * x * y = 3 * Real.sqrt 3 / 32 := 
sorry

end max_area_right_triangle_in_semicircle_l10_10533


namespace negation_example_l10_10139

theorem negation_example : ¬(∀ x : ℝ, x^2 + |x| ≥ 0) ↔ ∃ x : ℝ, x^2 + |x| < 0 :=
by
  sorry

end negation_example_l10_10139


namespace find_other_number_l10_10001

theorem find_other_number (A B : ℕ) (H1 : Nat.lcm A B = 2310) (H2 : Nat.gcd A B = 30) (H3 : A = 770) : B = 90 :=
  by
  sorry

end find_other_number_l10_10001


namespace side_length_square_base_l10_10464

theorem side_length_square_base 
  (height : ℕ) (volume : ℕ) (A : ℕ) (s : ℕ) 
  (h_height : height = 8) 
  (h_volume : volume = 288) 
  (h_base_area : A = volume / height) 
  (h_square_base : A = s ^ 2) :
  s = 6 :=
by
  sorry

end side_length_square_base_l10_10464


namespace initial_coins_l10_10418

theorem initial_coins (y : ℚ) 
  (h : y = (81*y - 1200 + 30)/81) : y = 1210 / 81 := 
by 
  sorry

end initial_coins_l10_10418


namespace susan_age_indeterminate_l10_10298

-- Definitions and conditions
def james_age_in_15_years : ℕ := 37
def current_james_age : ℕ := james_age_in_15_years - 15
def james_age_8_years_ago : ℕ := current_james_age - 8
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def current_janet_age : ℕ := janet_age_8_years_ago + 8

-- Problem: Prove that without Janet's age when Susan was born, we cannot determine Susan's age in 5 years.
theorem susan_age_indeterminate (susan_current_age : ℕ) : 
  (∃ janet_age_when_susan_born : ℕ, susan_current_age = current_janet_age - janet_age_when_susan_born) → 
  ¬ (∃ susan_age_in_5_years : ℕ, susan_age_in_5_years = susan_current_age + 5) := 
by
  sorry

end susan_age_indeterminate_l10_10298


namespace value_of_a_l10_10528

theorem value_of_a (x a : ℤ) (h1 : x = 2) (h2 : 3 * x - a = -x + 7) : a = 1 :=
by
  sorry

end value_of_a_l10_10528


namespace fraction_is_three_halves_l10_10378

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end fraction_is_three_halves_l10_10378


namespace white_washing_cost_correct_l10_10295

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

def door_length : ℝ := 6
def door_width : ℝ := 3

def window_length : ℝ := 4
def window_width : ℝ := 3

def cost_per_sq_ft : ℝ := 8

def calculate_white_washing_cost : ℝ :=
  let total_wall_area := 2 * (room_length * room_height) + 2 * (room_width * room_height)
  let door_area := door_length * door_width
  let window_area := 3 * (window_length * window_width)
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sq_ft

theorem white_washing_cost_correct : calculate_white_washing_cost = 7248 := by
  sorry

end white_washing_cost_correct_l10_10295


namespace all_go_together_l10_10769

noncomputable def probability_all_go_together : ℚ :=
  let time_frame := 60
  let v_wait := 15
  let b_wait := 10

  -- Total probability space (any moment within one hour for V and B)
  let total_area := (time_frame : ℚ) * (time_frame : ℚ)

  -- Unsuccessful meeting areas
  let unsuccessful_area := 2 * (b_wait * (time_frame / 2 : ℚ))

  -- Successful meeting area
  let meeting_area := total_area - unsuccessful_area

  -- Probability P(B and V meet)
  let p_meet := meeting_area / total_area

  -- Probability P(A arrives last)
  let p_lia := (1 : ℚ) / 3

  -- Combined probability of all three going together
  p_lia * p_meet

theorem all_go_together (A B V : ℚ) (arrives_between : ∀ x ∈ {A, B, V}, 0 ≤ x ∧ x ≤ 60) :
    probability_all_go_together = 5 / 18 := by
  sorry

end all_go_together_l10_10769


namespace part1_part2_l10_10206

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : f x ≤ x^2 :=
sorry

theorem part2 (x : ℝ) (hx : x > 0) (c : ℝ) (hc : c ≥ -1) : f x ≤ 2 * x + c :=
sorry

end part1_part2_l10_10206


namespace solve_oranges_problem_find_plans_and_max_profit_l10_10339

theorem solve_oranges_problem :
  ∃ (a b : ℕ), 15 * a + 20 * b = 430 ∧ 10 * a + 8 * b = 212 ∧ a = 10 ∧ b = 14 := by
    sorry

theorem find_plans_and_max_profit (a b : ℕ) (h₁ : 15 * a + 20 * b = 430) (h₂ : 10 * a + 8 * b = 212) (ha : a = 10) (hb : b = 14) :
  ∃ (x : ℕ), 58 ≤ x ∧ x ≤ 60 ∧ (10 * x + 14 * (100 - x) ≥ 1160) ∧ (10 * x + 14 * (100 - x) ≤ 1168) ∧ (1000 - 4 * x = 768) :=
    sorry

end solve_oranges_problem_find_plans_and_max_profit_l10_10339


namespace max_value_of_m_l10_10097

theorem max_value_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 8 > 0 → x < m) → m = -2 :=
by
  sorry

end max_value_of_m_l10_10097


namespace oak_trees_initially_in_park_l10_10989

def initialOakTrees (new_oak_trees total_oak_trees_after: ℕ) : ℕ :=
  total_oak_trees_after - new_oak_trees

theorem oak_trees_initially_in_park (new_oak_trees total_oak_trees_after initial_oak_trees : ℕ) 
  (h_new_trees : new_oak_trees = 2) 
  (h_total_after : total_oak_trees_after = 11) 
  (h_correct : initial_oak_trees = 9) : 
  initialOakTrees new_oak_trees total_oak_trees_after = initial_oak_trees := 
by 
  rw [h_new_trees, h_total_after, h_correct]
  sorry

end oak_trees_initially_in_park_l10_10989


namespace largest_integer_among_four_l10_10071

theorem largest_integer_among_four 
  (p q r s : ℤ)
  (h1 : p + q + r = 210)
  (h2 : p + q + s = 230)
  (h3 : p + r + s = 250)
  (h4 : q + r + s = 270) :
  max (max p q) (max r s) = 110 :=
by
  sorry

end largest_integer_among_four_l10_10071


namespace number_of_sheep_l10_10451

theorem number_of_sheep (S H : ℕ) 
  (h1 : S / H = 5 / 7)
  (h2 : H * 230 = 12880) : 
  S = 40 :=
by
  sorry

end number_of_sheep_l10_10451


namespace cats_and_dogs_biscuits_l10_10861

theorem cats_and_dogs_biscuits 
  (d c : ℕ) 
  (h1 : d + c = 10) 
  (h2 : 6 * d + 5 * c = 56) 
  : d = 6 ∧ c = 4 := 
by 
  sorry

end cats_and_dogs_biscuits_l10_10861


namespace time_on_wednesday_is_40_minutes_l10_10239

def hours_to_minutes (h : ℚ) : ℚ := h * 60

def time_monday : ℚ := hours_to_minutes (3 / 4)
def time_tuesday : ℚ := hours_to_minutes (1 / 2)
def time_wednesday (w : ℚ) : ℚ := w
def time_thursday : ℚ := hours_to_minutes (5 / 6)
def time_friday : ℚ := 75
def total_time : ℚ := hours_to_minutes 4

theorem time_on_wednesday_is_40_minutes (w : ℚ) 
    (h1 : time_monday = 45) 
    (h2 : time_tuesday = 30) 
    (h3 : time_thursday = 50) 
    (h4 : time_friday = 75)
    (h5 : total_time = 240) 
    (h6 : total_time = time_monday + time_tuesday + time_wednesday w + time_thursday + time_friday) 
    : w = 40 := 
by 
  sorry

end time_on_wednesday_is_40_minutes_l10_10239


namespace range_of_a_l10_10838

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1/x) + a

theorem range_of_a (a : ℝ) (h : f a 0 = a^2) : (f a 0 = f a 0 -> 0 ≤ a ∧ a ≤ 2) := by
  sorry

end range_of_a_l10_10838


namespace last_two_nonzero_digits_of_70_factorial_are_04_l10_10865

-- Given conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial_are_04 :
  let n := 70;
  ∀ t : ℕ, 
    t = factorial n → t % 100 ≠ 0 → (t % 100) / 10 != 0 → 
    (t % 100) = 04 :=
sorry

end last_two_nonzero_digits_of_70_factorial_are_04_l10_10865


namespace percentage_error_in_calculated_area_l10_10155

theorem percentage_error_in_calculated_area
  (a : ℝ)
  (measured_side_length : ℝ := 1.025 * a) :
  (measured_side_length ^ 2 - a ^ 2) / (a ^ 2) * 100 = 5.0625 :=
by 
  sorry

end percentage_error_in_calculated_area_l10_10155


namespace P_investment_time_l10_10431

noncomputable def investment_in_months 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop)
  (time_Q : ℕ)
  (time_P : ℕ)
  (x : ℕ) : Prop :=
  investment_ratio_PQ 7 5 ∧ 
  profit_ratio_PQ 7 9 ∧ 
  time_Q = 9 ∧ 
  (7 * time_P) / (5 * time_Q) = 7 / 9

theorem P_investment_time 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop) 
  (x : ℕ) : Prop :=
  ∀ (t : ℕ), investment_in_months investment_ratio_PQ profit_ratio_PQ 9 t x → t = 5

end P_investment_time_l10_10431


namespace problem_statement_l10_10115

variable (a b c : ℝ)

theorem problem_statement 
  (h₀ : a * b + b * c + c * a > a + b + c) 
  (h₁ : a + b + c > 0) 
: a + b + c > 3 := 
sorry

end problem_statement_l10_10115


namespace question1_question2_question3_l10_10804

open Set

-- Define sets A and B
def A := { x : ℝ | x^2 + 6 * x + 5 < 0 }
def B := { x : ℝ | -1 ≤ x ∧ x < 1 }

-- Universal set U is implicitly ℝ in Lean

-- Question 1: Prove A ∩ B = ∅
theorem question1 : A ∩ B = ∅ := 
sorry

-- Question 2: Prove complement of A ∪ B in ℝ is (-∞, -5] ∪ [1, ∞)
theorem question2 : compl (A ∪ B) = { x : ℝ | x ≤ -5 } ∪ { x : ℝ | x ≥ 1 } := 
sorry

-- Define set C which depends on parameter a
def C (a: ℝ) := { x : ℝ | x < a }

-- Question 3: Prove if B ∩ C = B, then a ≥ 1
theorem question3 (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := 
sorry

end question1_question2_question3_l10_10804


namespace sin_cos_identity_l10_10526

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := 
by
  sorry

end sin_cos_identity_l10_10526


namespace volume_after_increase_l10_10326

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l10_10326


namespace smallest_positive_multiple_of_45_l10_10288

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ (∀ y : ℕ, y > 0 → 45 * y ≥ 45 * x) ∧ 45 * x = 45 :=
by
  use 1
  split
  · apply Nat.one_pos
  · split
    · intros y hy
      apply mul_le_mul
      · apply Nat.one_le_of_lt hy
      · apply le_refl
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
    · apply rfl
  sorry

end smallest_positive_multiple_of_45_l10_10288


namespace family_travel_distance_l10_10293

noncomputable def distance_travelled(t1 t2 s1 s2 T : ℝ) : ℝ :=
  let D := 2 * T / ((1/s1) + (1/s2)) in D

theorem family_travel_distance : distance_travelled 1 1 35 40 12 = 448 :=
by
  sorry

end family_travel_distance_l10_10293


namespace tan_alpha_value_l10_10073

theorem tan_alpha_value (α : ℝ) (h : Real.sin (α / 2) = 2 * Real.cos (α / 2)) : Real.tan α = -4 / 3 :=
by
  sorry

end tan_alpha_value_l10_10073


namespace optimal_garden_dimensions_l10_10415

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l ≥ w + 20 ∧ l * w = 9600 :=
by
  sorry

end optimal_garden_dimensions_l10_10415


namespace largest_alpha_exists_l10_10065

theorem largest_alpha_exists : 
  ∃ α, (∀ m n : ℕ, 0 < m → 0 < n → (m:ℝ) / (n:ℝ) < Real.sqrt 7 → α / (n^2:ℝ) ≤ 7 - (m^2:ℝ) / (n^2:ℝ)) ∧ α = 3 :=
by
  sorry

end largest_alpha_exists_l10_10065


namespace inequality_solution_l10_10564

theorem inequality_solution :
  {x : ℝ | |2 * x - 3| + |x + 1| < 7 ∧ x ≤ 4} = {x : ℝ | -5 / 3 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_l10_10564


namespace root_quadratic_sum_product_l10_10807

theorem root_quadratic_sum_product (x1 x2 : ℝ) (h1 : x1^2 - 2 * x1 - 5 = 0) (h2 : x2^2 - 2 * x2 - 5 = 0) 
  (h3 : x1 ≠ x2) : (x1 + x2 + 3 * (x1 * x2)) = -13 := 
by 
  sorry

end root_quadratic_sum_product_l10_10807


namespace square_number_increased_decreased_by_five_remains_square_l10_10350

theorem square_number_increased_decreased_by_five_remains_square :
  ∃ x : ℤ, ∃ u v : ℤ, x^2 + 5 = u^2 ∧ x^2 - 5 = v^2 := by
  sorry

end square_number_increased_decreased_by_five_remains_square_l10_10350


namespace minimum_value_of_expression_l10_10193

theorem minimum_value_of_expression
  (a b c : ℝ)
  (h : 2 * a + 2 * b + c = 8) :
  ∃ x, (x = (a - 1)^2 + (b + 2)^2 + (c - 3)^2) ∧ x ≥ (49 / 9) :=
sorry

end minimum_value_of_expression_l10_10193


namespace total_houses_in_neighborhood_l10_10215

-- Definition of the function f
def f (x : ℕ) : ℕ := x^2 + 3*x

-- Given conditions
def x := 40

-- The theorem states that the total number of houses in Mariam's neighborhood is 1760.
theorem total_houses_in_neighborhood : (x + f x) = 1760 :=
by
  sorry

end total_houses_in_neighborhood_l10_10215


namespace ball_hits_ground_time_l10_10009

theorem ball_hits_ground_time :
  ∃ t : ℝ, -20 * t^2 + 30 * t + 60 = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
by 
  sorry

end ball_hits_ground_time_l10_10009


namespace odd_positive_int_divisible_by_24_l10_10964

theorem odd_positive_int_divisible_by_24 (n : ℕ) (hn : n % 2 = 1 ∧ n > 0) : 24 ∣ (n ^ n - n) :=
sorry

end odd_positive_int_divisible_by_24_l10_10964


namespace almonds_received_by_amanda_l10_10566

variable (totalAlmonds : ℚ)
variable (numberOfPiles : ℚ)
variable (pilesForAmanda : ℚ)

-- Conditions
def stephanie_has_almonds := totalAlmonds = 66 / 7
def distribute_equally_into_piles := numberOfPiles = 6
def amanda_receives_piles := pilesForAmanda = 3

-- Conclusion to prove
theorem almonds_received_by_amanda :
  stephanie_has_almonds totalAlmonds →
  distribute_equally_into_piles numberOfPiles →
  amanda_receives_piles pilesForAmanda →
  (totalAlmonds / numberOfPiles) * pilesForAmanda = 33 / 7 :=
by
  sorry

end almonds_received_by_amanda_l10_10566


namespace remainder_of_7_9_power_2008_mod_64_l10_10985

theorem remainder_of_7_9_power_2008_mod_64 :
  (7^2008 + 9^2008) % 64 = 2 := 
sorry

end remainder_of_7_9_power_2008_mod_64_l10_10985


namespace total_games_l10_10786

variable (G R : ℕ)

axiom cond1 : 85 + (1/2 : ℚ) * R = (0.70 : ℚ) * G
axiom cond2 : G = 100 + R

theorem total_games : G = 175 := by
  sorry

end total_games_l10_10786


namespace particular_number_l10_10750

theorem particular_number {x : ℕ} (h : x - 29 + 64 = 76) : x = 41 := by
  sorry

end particular_number_l10_10750


namespace greatest_natural_number_l10_10487

theorem greatest_natural_number (n q r : ℕ) (h1 : n = 91 * q + r)
  (h2 : r = q^2) (h3 : r < 91) : n = 900 :=
sorry

end greatest_natural_number_l10_10487


namespace solve_trigonometric_equation_l10_10727

open Set Real

theorem solve_trigonometric_equation :
  {x : ℝ | 0 < x ∧ x < π ∧ 2 * cos (x - π / 4) = sqrt 2} = {π / 2} :=
by
  sorry

end solve_trigonometric_equation_l10_10727


namespace percentage_of_students_owning_birds_l10_10539

theorem percentage_of_students_owning_birds
    (total_students : ℕ) 
    (students_owning_birds : ℕ) 
    (h_total_students : total_students = 500) 
    (h_students_owning_birds : students_owning_birds = 75) : 
    (students_owning_birds * 100) / total_students = 15 := 
by 
    sorry

end percentage_of_students_owning_birds_l10_10539


namespace exists_acute_triangle_l10_10507

-- Define the segments as a list of positive real numbers
variables (a b c d e : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0) (h3 : d > 0) (h4 : e > 0)
(h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e)

-- Conditions: Any three segments can form a triangle
variables (h_triangle_1 : a + b > c ∧ a + c > b ∧ b + c > a)
variables (h_triangle_2 : a + b > d ∧ a + d > b ∧ b + d > a)
variables (h_triangle_3 : a + b > e ∧ a + e > b ∧ b + e > a)
variables (h_triangle_4 : a + c > d ∧ a + d > c ∧ c + d > a)
variables (h_triangle_5 : a + c > e ∧ a + e > c ∧ c + e > a)
variables (h_triangle_6 : a + d > e ∧ a + e > d ∧ d + e > a)
variables (h_triangle_7 : b + c > d ∧ b + d > c ∧ c + d > b)
variables (h_triangle_8 : b + c > e ∧ b + e > c ∧ c + e > b)
variables (h_triangle_9 : b + d > e ∧ b + e > d ∧ d + e > b)
variables (h_triangle_10 : c + d > e ∧ c + e > d ∧ d + e > c)

-- Prove that there exists an acute-angled triangle 
theorem exists_acute_triangle : ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                                        (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                                        (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
                                        x + y > z ∧ x + z > y ∧ y + z > x ∧ 
                                        x^2 < y^2 + z^2 := 
sorry

end exists_acute_triangle_l10_10507


namespace inequality_solution_l10_10705

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l10_10705


namespace rounds_played_l10_10740

-- Define the given conditions as Lean constants
def totalPoints : ℝ := 378.5
def pointsPerRound : ℝ := 83.25

-- Define the goal as a Lean theorem
theorem rounds_played :
  Int.ceil (totalPoints / pointsPerRound) = 5 := 
by 
  sorry

end rounds_played_l10_10740


namespace find_pairs_l10_10351

theorem find_pairs (p q : ℤ) (a b : ℤ) :
  (p^2 - 4 * q = a^2) ∧ (q^2 - 4 * p = b^2) ↔ 
    (p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9) :=
by
  sorry

end find_pairs_l10_10351


namespace quadratic_pos_in_interval_l10_10291

theorem quadratic_pos_in_interval (m n : ℤ)
  (h2014 : (2014:ℤ)^2 + m * 2014 + n > 0)
  (h2015 : (2015:ℤ)^2 + m * 2015 + n > 0) :
  ∀ x : ℝ, 2014 ≤ x ∧ x ≤ 2015 → (x^2 + (m:ℝ) * x + (n:ℝ)) > 0 :=
by
  sorry

end quadratic_pos_in_interval_l10_10291


namespace ratio_of_patients_l10_10463

def one_in_four_zx (current_patients : ℕ) : ℕ :=
  current_patients / 4

def previous_patients : ℕ :=
  26

def diagnosed_patients : ℕ :=
  13

def current_patients : ℕ :=
  diagnosed_patients * 4

theorem ratio_of_patients : 
  one_in_four_zx current_patients = diagnosed_patients → 
  (current_patients / previous_patients) = 2 := 
by 
  sorry

end ratio_of_patients_l10_10463


namespace parallel_lines_iff_a_eq_neg3_l10_10599

theorem parallel_lines_iff_a_eq_neg3 (a : ℝ) :
  (∀ x y : ℝ, a * x + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 ≠ 0) ↔ a = -3 :=
sorry

end parallel_lines_iff_a_eq_neg3_l10_10599


namespace goods_train_length_l10_10611

noncomputable def length_of_goods_train (speed_first_train_kmph speed_goods_train_kmph time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := speed_first_train_kmph + speed_goods_train_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5.0 / 18.0)
  relative_speed_mps * (time_seconds : ℝ)

theorem goods_train_length
  (speed_first_train_kmph : ℕ) (speed_goods_train_kmph : ℕ) (time_seconds : ℕ) 
  (h1 : speed_first_train_kmph = 50)
  (h2 : speed_goods_train_kmph = 62)
  (h3 : time_seconds = 9) :
  length_of_goods_train speed_first_train_kmph speed_goods_train_kmph time_seconds = 280 :=
  sorry

end goods_train_length_l10_10611


namespace unique_real_solution_l10_10055

theorem unique_real_solution :
  ∀ (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) →
    (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) :=
by
  intro x y z w
  intros h
  have h1 : x = z + w + Real.sqrt (z * w * x) := h.1
  have h2 : y = w + x + Real.sqrt (w * x * y) := h.2.1
  have h3 : z = x + y + Real.sqrt (x * y * z) := h.2.2.1
  have h4 : w = y + z + Real.sqrt (y * z * w) := h.2.2.2
  sorry

end unique_real_solution_l10_10055


namespace negation_proposition_l10_10864

theorem negation_proposition :
  (∀ x : ℝ, 0 < x → x^2 + 1 ≥ 2 * x) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proposition_l10_10864


namespace triangle_area_triangle_perimeter_l10_10105

noncomputable def area_of_triangle (A B C : ℝ) (a b c : ℝ) := 
  1/2 * b * c * (Real.sin A)

theorem triangle_area (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : A = Real.pi / 3) : 
  area_of_triangle A B C a b c = Real.sqrt 3 / 4 := 
  sorry

theorem triangle_perimeter (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : 4 * Real.cos B * Real.cos C - 1 = 0) 
  (h3 : b + c = 2)
  (h4 : a = 1) :
  a + b + c = 3 :=
  sorry

end triangle_area_triangle_perimeter_l10_10105


namespace length_of_first_train_solution_l10_10873

noncomputable def length_of_first_train (speed1_kmph speed2_kmph : ℝ) (length2_m time_s : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (5 / 18)
  let speed2_mps := speed2_kmph * (5 / 18)
  let relative_speed_mps := speed1_mps + speed2_mps
  let combined_length_m := relative_speed_mps * time_s
  combined_length_m - length2_m

theorem length_of_first_train_solution 
  (speed1_kmph : ℝ) 
  (speed2_kmph : ℝ) 
  (length2_m : ℝ) 
  (time_s : ℝ) 
  (h₁ : speed1_kmph = 42) 
  (h₂ : speed2_kmph = 30) 
  (h₃ : length2_m = 120) 
  (h₄ : time_s = 10.999120070394369) : 
  length_of_first_train speed1_kmph speed2_kmph length2_m time_s = 99.98 :=
by 
  sorry

end length_of_first_train_solution_l10_10873


namespace correct_statement_C_l10_10995

def V_m_rho_relation (V m ρ : ℝ) : Prop :=
  V = m / ρ

theorem correct_statement_C (V m ρ : ℝ) (h : ρ ≠ 0) : 
  ((∃ k : ℝ, k = ρ ∧ ∀ V' m' : ℝ, V' = m' / k → V' ≠ V) ∧ 
  (∃ v_var v_var', v_var = V ∧ v_var' = m ∧ V = m / ρ) →
  (∃ ρ_const : ℝ, ρ_const = ρ)) :=
by
  sorry

end correct_statement_C_l10_10995


namespace total_money_earned_l10_10775

def earning_per_question : ℝ := 0.2
def questions_per_survey : ℕ := 10
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4

theorem total_money_earned :
  earning_per_question * (questions_per_survey * (surveys_on_monday + surveys_on_tuesday)) = 14 := by
  sorry

end total_money_earned_l10_10775


namespace max_sides_subdivision_13_max_sides_subdivision_1950_l10_10217

-- Part (a)
theorem max_sides_subdivision_13 (n : ℕ) (h : n = 13) : 
  ∃ p : ℕ, p ≤ n ∧ p = 13 := 
sorry

-- Part (b)
theorem max_sides_subdivision_1950 (n : ℕ) (h : n = 1950) : 
  ∃ p : ℕ, p ≤ n ∧ p = 1950 := 
sorry

end max_sides_subdivision_13_max_sides_subdivision_1950_l10_10217


namespace find_a_value_l10_10982

def quadratic_vertex_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = 2) → (y = 5) →
  a * (x - 2)^2 + 5 = y

def quadratic_passing_point_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = -1) → (y = -20) →
  a * (x - 2)^2 + 5 = y

theorem find_a_value : ∃ a : ℚ, quadratic_vertex_condition a ∧ quadratic_passing_point_condition a ∧ a = (-25)/9 := 
by 
  sorry

end find_a_value_l10_10982


namespace base_k_number_to_decimal_l10_10136

theorem base_k_number_to_decimal (k : ℕ) (h : 4 ≤ k) : 1 * k^2 + 3 * k + 2 = 30 ↔ k = 4 := by
  sorry

end base_k_number_to_decimal_l10_10136


namespace find_e_of_x_l10_10103

noncomputable def x_plus_inv_x_eq_five (x : ℝ) : Prop :=
  x + (1 / x) = 5

theorem find_e_of_x (x : ℝ) (h : x_plus_inv_x_eq_five x) : 
  x^2 + (1 / x)^2 = 23 := sorry

end find_e_of_x_l10_10103


namespace intersecting_lines_sum_l10_10649

theorem intersecting_lines_sum (a b : ℝ) 
  (h1 : a * 1 + 1 + 1 = 0)
  (h2 : 2 * 1 - b * 1 - 1 = 0) : 
  a + b = -1 := 
by 
  have ha : a = -2 := by linarith [h1]
  have hb : b = 1 := by linarith [h2]
  rw [ha, hb]
  exact by norm_num

end intersecting_lines_sum_l10_10649


namespace high_probability_event_is_C_l10_10017

-- Define the probabilities of events A, B, and C
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Statement asserting Event C has the high possibility of occurring
theorem high_probability_event_is_C : prob_C > prob_A ∧ prob_C > prob_B :=
by
  sorry

end high_probability_event_is_C_l10_10017


namespace count_integers_with_factors_12_and_7_l10_10524

theorem count_integers_with_factors_12_and_7 :
  ∃ k : ℕ, k = 4 ∧
    (∀ (n : ℕ), 500 ≤ n ∧ n ≤ 800 ∧ 12 ∣ n ∧ 7 ∣ n ↔ (84 ∣ n ∧
      n = 504 ∨ n = 588 ∨ n = 672 ∨ n = 756)) :=
sorry

end count_integers_with_factors_12_and_7_l10_10524


namespace exists_maximum_value_of_f_l10_10559

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ := (3 * x * y + 1) * Real.exp (-(x^2 + y^2))

-- Maximum value proof statement
theorem exists_maximum_value_of_f :
  ∃ (x y : ℝ), f x y = (3 / 2) * Real.exp (-1 / 3) :=
sorry

end exists_maximum_value_of_f_l10_10559


namespace min_sum_geometric_sequence_l10_10538

noncomputable def sequence_min_value (a : ℕ → ℝ) : ℝ :=
  a 4 + a 3 - 2 * a 2 - 2 * a 1

theorem min_sum_geometric_sequence (a : ℕ → ℝ)
  (h : sequence_min_value a = 6) :
  a 5 + a 6 = 48 := 
by
  sorry

end min_sum_geometric_sequence_l10_10538


namespace bart_earned_14_l10_10774

variable (questions_per_survey money_per_question surveys_monday surveys_tuesday : ℕ → ℝ)
variable (total_surveys total_questions money_earned : ℕ → ℝ)

noncomputable def conditions :=
  let questions_per_survey := 10
  let money_per_question := 0.2
  let surveys_monday := 3
  let surveys_tuesday := 4
  let total_surveys := surveys_monday + surveys_tuesday
  let total_questions := questions_per_survey * total_surveys
  let money_earned := total_questions * money_per_question
  money_earned = 14

theorem bart_earned_14 : conditions :=
by
  -- proof steps
  sorry

end bart_earned_14_l10_10774


namespace peter_total_distance_is_six_l10_10744

def total_distance_covered (d : ℝ) :=
  let first_part_time := (2/3) * d / 4
  let second_part_time := (1/3) * d / 5
  (first_part_time + second_part_time) = 1.4

theorem peter_total_distance_is_six :
  ∃ d : ℝ, total_distance_covered d ∧ d = 6 := 
by
  -- Proof can be filled here
  sorry

end peter_total_distance_is_six_l10_10744


namespace probability_non_obtuse_l10_10497

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l10_10497


namespace box_volume_increase_l10_10330

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l10_10330


namespace Q1_Q2_l10_10709

noncomputable def prob_A_scores_3_out_of_4 (p_A_serves : ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q1 (p_A_serves : ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_A_scores_3_out_of_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 1/3 :=
  by
    -- Proof of the theorem
    sorry

noncomputable def prob_X_lessthan_or_equal_4 (p_A_serves: ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q2 (p_A_serves: ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_X_lessthan_or_equal_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 3/4 :=
  by
    -- Proof of the theorem
    sorry

end Q1_Q2_l10_10709


namespace terminating_decimal_count_l10_10069

def count_terminating_decimals (n: ℕ): ℕ :=
  (n / 17)

theorem terminating_decimal_count : count_terminating_decimals 493 = 29 := by
  sorry

end terminating_decimal_count_l10_10069


namespace mean_of_remaining_l10_10568

variable (a b c : ℝ)
variable (mean_of_four : ℝ := 90)
variable (largest : ℝ := 105)

theorem mean_of_remaining (h1 : (a + b + c + largest) / 4 = mean_of_four) : (a + b + c) / 3 = 85 := by
  sorry

end mean_of_remaining_l10_10568


namespace total_money_earned_l10_10776

def earning_per_question : ℝ := 0.2
def questions_per_survey : ℕ := 10
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4

theorem total_money_earned :
  earning_per_question * (questions_per_survey * (surveys_on_monday + surveys_on_tuesday)) = 14 := by
  sorry

end total_money_earned_l10_10776


namespace farm_horses_cows_l10_10038

variables (H C : ℕ)

theorem farm_horses_cows (H C : ℕ) (h1 : H = 6 * C) (h2 : (H - 15) = 3 * (C + 15)) : (H - 15) - (C + 15) = 70 :=
by {
  sorry
}

end farm_horses_cows_l10_10038


namespace contrapositive_of_square_inequality_l10_10255

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x > y → x^2 > y^2) ↔ (x^2 ≤ y^2 → x ≤ y) :=
sorry

end contrapositive_of_square_inequality_l10_10255


namespace plane_eq_unique_l10_10261

open Int 

def plane_eq (A B C D x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_eq_unique (x y z : ℤ) (A B C D : ℤ)
  (h₁ : x = 8) 
  (h₂ : y = -6) 
  (h₃ : z = 2) 
  (h₄ : A > 0)
  (h₅ : gcd (|A|) (gcd (|B|) (gcd (|C|) (|D|))) = 1) :
  plane_eq 4 (-3) 1 (-52) x y z :=
by
  sorry

end plane_eq_unique_l10_10261


namespace average_value_correct_l10_10063

noncomputable def average_value (k z : ℝ) : ℝ :=
  (k + 2 * k * z + 4 * k * z + 8 * k * z + 16 * k * z) / 5

theorem average_value_correct (k z : ℝ) :
  average_value k z = (k * (1 + 30 * z)) / 5 := by
  sorry

end average_value_correct_l10_10063


namespace stamps_total_l10_10571

theorem stamps_total (x : ℕ) (a_initial : ℕ := 5 * x) (b_initial : ℕ := 4 * x)
                     (a_after : ℕ := a_initial - 5) (b_after : ℕ := b_initial + 5)
                     (h_ratio_initial : a_initial / b_initial = 5 / 4)
                     (h_ratio_final : a_after / b_after = 4 / 5) :
                     a_initial + b_initial = 45 :=
by
  sorry

end stamps_total_l10_10571


namespace contrapositive_of_proposition_is_false_l10_10004

theorem contrapositive_of_proposition_is_false (x y : ℝ) 
  (h₀ : (x + y > 0) → (x > 0 ∧ y > 0)) : 
  ¬ ((x ≤ 0 ∨ y ≤ 0) → (x + y ≤ 0)) :=
by
  sorry

end contrapositive_of_proposition_is_false_l10_10004


namespace only_n_equal_3_exists_pos_solution_l10_10925

theorem only_n_equal_3_exists_pos_solution :
  ∀ (n : ℕ), (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := 
by
  sorry

end only_n_equal_3_exists_pos_solution_l10_10925


namespace max_digit_d_divisible_by_33_l10_10636

theorem max_digit_d_divisible_by_33 (d e : ℕ) (h₀ : 0 ≤ d ∧ d ≤ 9) (h₁ : 0 ≤ e ∧ e ≤ 9) 
  (h₂ : d + e = 4) : d ≤ 4 :=
by {
  sorry
}

example : ∃ d e : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧ d + e = 4 ∧ 
(d = 4) :=
by {
  use [4, 0],
  repeat { split },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial }
}

end max_digit_d_divisible_by_33_l10_10636


namespace complex_number_solution_l10_10195

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i * i = -1) (h : z + z * i = 1 + 5 * i) : z = 3 + 2 * i :=
sorry

end complex_number_solution_l10_10195


namespace solutions_to_deqs_l10_10700

noncomputable def x1 (t : ℝ) : ℝ := -1 / t^2
noncomputable def x2 (t : ℝ) : ℝ := -t * Real.log t

theorem solutions_to_deqs (t : ℝ) (ht : 0 < t) :
  (deriv x1 t = 2 * t * (x1 t)^2) ∧ (deriv x2 t = x2 t / t - 1) :=
by
  sorry

end solutions_to_deqs_l10_10700


namespace trinomial_has_two_roots_l10_10314

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l10_10314


namespace fraction_of_students_received_B_l10_10537

theorem fraction_of_students_received_B {total_students : ℝ}
  (fraction_A : ℝ)
  (fraction_A_or_B : ℝ)
  (h_fraction_A : fraction_A = 0.7)
  (h_fraction_A_or_B : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 :=
by
  rw [h_fraction_A, h_fraction_A_or_B]
  sorry

end fraction_of_students_received_B_l10_10537


namespace total_students_count_l10_10947

variable (T : ℕ)
variable (J : ℕ) (S : ℕ) (F : ℕ) (Sn : ℕ)

-- Given conditions:
-- 1. 26 percent are juniors.
def percentage_juniors (T J : ℕ) : Prop := J = 26 * T / 100
-- 2. 75 percent are not sophomores.
def percentage_sophomores (T S : ℕ) : Prop := S = 25 * T / 100
-- 3. There are 160 seniors.
def seniors_count (Sn : ℕ) : Prop := Sn = 160
-- 4. There are 32 more freshmen than sophomores.
def freshmen_sophomore_relationship (F S : ℕ) : Prop := F = S + 32

-- Question: Prove the total number of students is 800.
theorem total_students_count
  (hJ : percentage_juniors T J)
  (hS : percentage_sophomores T S)
  (hSn : seniors_count Sn)
  (hF : freshmen_sophomore_relationship F S) :
  F + S + J + Sn = T → T = 800 := by
  sorry

end total_students_count_l10_10947


namespace gcd_1729_78945_is_1_l10_10032

theorem gcd_1729_78945_is_1 :
  ∃ m n : ℤ, 1729 * m + 78945 * n = 1 := sorry

end gcd_1729_78945_is_1_l10_10032


namespace angle_BAC_eq_69_l10_10808

-- Definitions and conditions
def AM_Squared_EQ_CM_MN (AM CM MN : ℝ) : Prop := AM^2 = CM * MN
def AM_EQ_MK (AM MK : ℝ) : Prop := AM = MK
def angle_AMN_EQ_CMK (angle_AMN angle_CMK : ℝ) : Prop := angle_AMN = angle_CMK
def angle_B : ℝ := 47
def angle_C : ℝ := 64

-- Final proof statement
theorem angle_BAC_eq_69 (AM CM MN MK : ℝ)
  (h1: AM_Squared_EQ_CM_MN AM CM MN)
  (h2: AM_EQ_MK AM MK)
  (h3: angle_AMN_EQ_CMK 70 70) -- Placeholder angle values since angles must be given/defined
  : ∃ angle_BAC : ℝ, angle_BAC = 69 :=
sorry

end angle_BAC_eq_69_l10_10808


namespace perpendicular_lines_slope_l10_10791

theorem perpendicular_lines_slope :
  ∀ (a : ℚ), (∀ x y : ℚ, y = 3 * x + 5) 
  ∧ (∀ x y : ℚ, 4 * y + a * x = 8) →
  a = 4 / 3 :=
by
  intro a
  intro h
  sorry

end perpendicular_lines_slope_l10_10791


namespace cost_price_toy_l10_10997

theorem cost_price_toy (selling_price_total : ℝ) (total_toys : ℕ) (gain_toys : ℕ) (sp_per_toy : ℝ) (general_cost : ℝ) :
  selling_price_total = 27300 →
  total_toys = 18 →
  gain_toys = 3 →
  sp_per_toy = selling_price_total / total_toys →
  general_cost = sp_per_toy * total_toys - (sp_per_toy * gain_toys / total_toys) →
    general_cost = 1300 := 
by 
  sorry

end cost_price_toy_l10_10997


namespace original_price_after_discount_l10_10102

theorem original_price_after_discount (a x : ℝ) (h : 0.7 * x = a) : x = (10 / 7) * a := 
sorry

end original_price_after_discount_l10_10102


namespace carrie_bought_t_shirts_l10_10912

theorem carrie_bought_t_shirts (total_spent : ℝ) (cost_each : ℝ) (n : ℕ) 
    (h_total : total_spent = 199) (h_cost : cost_each = 9.95) 
    (h_eq : n = total_spent / cost_each) : n = 20 := 
by
sorry

end carrie_bought_t_shirts_l10_10912


namespace largest_constant_D_l10_10353

theorem largest_constant_D (D : ℝ) 
  (h : ∀ (x y : ℝ), x^2 + y^2 + 4 ≥ D * (x + y)) : 
  D ≤ 2 * Real.sqrt 2 :=
sorry

end largest_constant_D_l10_10353


namespace area_of_right_triangle_with_incircle_l10_10755

theorem area_of_right_triangle_with_incircle (a b c r : ℝ) :
  (a = 6 + r) → 
  (b = 7 + r) → 
  (c = 13) → 
  (a^2 + b^2 = c^2) →
  (2 * r^2 + 26 * r = 84) →
  (area = 1/2 * ((6 + r) * (7 + r))) →
  area = 42 := 
by 
  sorry

end area_of_right_triangle_with_incircle_l10_10755


namespace car_second_hour_speed_l10_10143

theorem car_second_hour_speed (s1 s2 : ℕ) (h1 : s1 = 100) (avg : (s1 + s2) / 2 = 80) : s2 = 60 :=
by
  sorry

end car_second_hour_speed_l10_10143


namespace arithmetic_sequence_sum_cubes_l10_10019

theorem arithmetic_sequence_sum_cubes (x : ℤ) (k : ℕ) (h : ∀ i, 0 <= i ∧ i <= k → (x + 2 * i : ℤ)^3 =
  -1331) (hk : k > 3) : k = 6 :=
sorry

end arithmetic_sequence_sum_cubes_l10_10019


namespace bart_earnings_l10_10778

theorem bart_earnings :
  let payment_per_question := 0.2 in
  let questions_per_survey := 10 in
  let surveys_monday := 3 in
  let surveys_tuesday := 4 in
  (surveys_monday * questions_per_survey + surveys_tuesday * questions_per_survey) * payment_per_question = 14 :=
by
  sorry

end bart_earnings_l10_10778


namespace complement_P_eq_Ioo_l10_10208

def U : Set ℝ := Set.univ
def P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_of_P_in_U : Set ℝ := Set.Ioo (-1) 6

theorem complement_P_eq_Ioo :
  (U \ P) = complement_of_P_in_U :=
by sorry

end complement_P_eq_Ioo_l10_10208


namespace inequality_proof_l10_10942

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : (n / m) + (m / n) > 2 :=
sorry

end inequality_proof_l10_10942


namespace volume_of_P_ABC_correct_equation_of_line_through_M_correct_l10_10949

noncomputable def volume_of_triangular_pyramid (AB AC AP : ℝ × ℝ × ℝ) : ℝ :=
  let normal_vector := (1, 2, 0) in
  let h := (|normal_vector.fst * AP.fst + normal_vector.snd * AP.snd + normal_vector.trd * AP.trd|) /
    (Real.sqrt (normal_vector.fst ^ 2 + normal_vector.snd ^ 2 + normal_vector.trd ^ 2)) in
  let cosA := (AB.fst * AC.fst + AB.snd * AC.snd + AB.trd * AC.trd) /
    ((Real.sqrt (AB.fst ^ 2 + AB.snd ^ 2 + AB.trd ^ 2) * Real.sqrt (AC.fst ^ 2 + AC.snd ^ 2 + AC.trd ^ 2))) in
  let sinA := Real.sqrt (1 - cosA ^ 2) in
  let area_ABC := 0.5 * (Real.sqrt (AB.fst ^ 2 + AB.snd ^ 2 + AB.trd ^ 2) * Real.sqrt (AC.fst ^ 2 + AC.snd ^ 2 + AC.trd ^ 2) * sinA) in
  (1/3) * area_ABC * h

theorem volume_of_P_ABC_correct : volume_of_triangular_pyramid (2, -1, 3) (-2, 1, 0) (3, -1, 4) = 1/2 := by
  sorry

noncomputable def equation_of_line_through_M (M AB : ℝ × ℝ × ℝ) : String :=
  "⟨(1 - x) / " ++ toString AB.fst ++ ", (1 - y) / " ++ toString -AB.snd ++ ", (1 - z) / " ++ toString AB.trd ++ "⟩"

theorem equation_of_line_through_M_correct : equation_of_line_through_M (1, 1, 1) (2, -1, 3) = "⟨(1 - x) / 2, (1 - y) / -1, (1 - z) / 3⟩" := by 
  sorry

end volume_of_P_ABC_correct_equation_of_line_through_M_correct_l10_10949


namespace change_received_l10_10720

theorem change_received (price_wooden_toy : ℕ) (price_hat : ℕ) (money_paid : ℕ) (num_wooden_toys : ℕ) (num_hats : ℕ) : 
  price_wooden_toy = 20 → price_hat = 10 → money_paid = 100 → num_wooden_toys = 2 → num_hats = 3 → 
  money_paid - (num_wooden_toys * price_wooden_toy + num_hats * price_hat) = 30 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5] 
  simp 
  norm_num 
  sorry

end change_received_l10_10720


namespace prob_a_greater_than_b_l10_10242

noncomputable def probability_of_team_a_finishing_with_more_points (n_teams : ℕ) (initial_win : Bool) : ℚ :=
  if initial_win ∧ n_teams = 9 then
    39203 / 65536
  else
    0 -- This is a placeholder and not accurate for other cases

theorem prob_a_greater_than_b (n_teams : ℕ) (initial_win : Bool) (hp : initial_win ∧ n_teams = 9) :
  probability_of_team_a_finishing_with_more_points n_teams initial_win = 39203 / 65536 :=
by
  sorry

end prob_a_greater_than_b_l10_10242


namespace joan_carrots_grown_correct_l10_10395

variable (total_carrots : ℕ) (jessica_carrots : ℕ) (joan_carrots : ℕ)

theorem joan_carrots_grown_correct (h1 : total_carrots = 40) (h2 : jessica_carrots = 11) (h3 : total_carrots = joan_carrots + jessica_carrots) : joan_carrots = 29 :=
by
  sorry

end joan_carrots_grown_correct_l10_10395


namespace quadratic_trinomial_has_two_roots_l10_10311

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l10_10311


namespace system_of_equations_solution_l10_10986

theorem system_of_equations_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 3) : 
  x = 4 ∧ y = 1 :=
by
  sorry

end system_of_equations_solution_l10_10986


namespace obtuse_triangle_probability_l10_10500

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l10_10500


namespace parabola_reflection_translation_l10_10904

open Real

noncomputable def f (a b c x : ℝ) : ℝ := a * (x - 4)^2 + b * (x - 4) + c
noncomputable def g (a b c x : ℝ) : ℝ := -a * (x + 4)^2 - b * (x + 4) - c
noncomputable def fg_x (a b c x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation (a b c x : ℝ) (ha : a ≠ 0) :
  fg_x a b c x = -16 * a * x :=
by
  sorry

end parabola_reflection_translation_l10_10904


namespace exists_subset_with_property_l10_10061

theorem exists_subset_with_property :
  ∃ X : Set Int, ∀ n : Int, ∃ (a b : X), a + 2 * b = n ∧ ∀ (a' b' : X), (a + 2 * b = n ∧ a' + 2 * b' = n) → (a = a' ∧ b = b') :=
sorry

end exists_subset_with_property_l10_10061


namespace track_and_field_unit_incorrect_l10_10724

theorem track_and_field_unit_incorrect :
  ∀ (L : ℝ), L = 200 → "mm" ≠ "m" → false :=
by
  intros L hL hUnit
  sorry

end track_and_field_unit_incorrect_l10_10724


namespace statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l10_10034

theorem statement_A_correct :
  (∃ x0 : ℝ, x0^2 + 2 * x0 + 2 < 0) ↔ (¬ ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0) :=
sorry

theorem statement_B_incorrect :
  ¬ (∀ x y : ℝ, x > y → |x| > |y|) :=
sorry

theorem statement_C_incorrect :
  ¬ ∀ x : ℤ, x^2 > 0 :=
sorry

theorem statement_D_correct :
  (∀ m : ℝ, (∃ x1 x2 : ℝ, x1 + x2 = 2 ∧ x1 * x2 = m ∧ x1 * x2 > 0) ↔ m < 0) :=
sorry

end statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l10_10034


namespace solve_for_A_in_terms_of_B_l10_10549

noncomputable def f (A B x : ℝ) := A * x - 2 * B^2
noncomputable def g (B x : ℝ) := B * x

theorem solve_for_A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end solve_for_A_in_terms_of_B_l10_10549


namespace max_viewing_area_l10_10579

theorem max_viewing_area (L W: ℝ) (h1: 2 * L + 2 * W = 420) (h2: L ≥ 100) (h3: W ≥ 60) : 
  (L = 105) ∧ (W = 105) ∧ (L * W = 11025) :=
by
  sorry

end max_viewing_area_l10_10579


namespace first_year_exceeds_threshold_l10_10043

def P (n : ℕ) : ℝ := 40000 * (1 + 0.2) ^ n
def exceeds_threshold (n : ℕ) : Prop := P n > 120000

theorem first_year_exceeds_threshold : ∃ n : ℕ, exceeds_threshold n ∧ 2013 + n = 2020 := 
by
  sorry

end first_year_exceeds_threshold_l10_10043


namespace trinomial_has_two_roots_l10_10313

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l10_10313


namespace exists_n_sum_three_digit_identical_digit_l10_10672

theorem exists_n_sum_three_digit_identical_digit:
  ∃ (n : ℕ), (∃ (k : ℕ), (k ≥ 1 ∧ k ≤ 9) ∧ (n*(n+1)/2 = 111*k)) ∧ n = 36 :=
by
  -- Placeholder for the proof
  sorry

end exists_n_sum_three_digit_identical_digit_l10_10672


namespace arithmetic_sequence_sum_l10_10934

theorem arithmetic_sequence_sum 
  (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n : ℕ, a n = 2 + (n - 5)) 
  (ha5 : a 5 = 2) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9) := 
by 
  sorry

end arithmetic_sequence_sum_l10_10934


namespace system_solution_l10_10987

theorem system_solution (x y: ℝ) 
  (h1: x + y = 2) 
  (h2: 3 * x + y = 4) : 
  x = 1 ∧ y = 1 :=
sorry

end system_solution_l10_10987


namespace geometric_sequence_nine_l10_10670

theorem geometric_sequence_nine (a : ℕ → ℝ) (h_geo : ∀ n, a (n + 1) / a n = a 1 / a 0) 
  (h_a1 : a 1 = 2) (h_a5: a 5 = 4) : a 9 = 8 := 
by
  sorry

end geometric_sequence_nine_l10_10670


namespace least_expensive_trip_is_1627_44_l10_10246

noncomputable def least_expensive_trip_cost : ℝ :=
  let distance_DE := 4500
  let distance_DF := 4000
  let distance_EF := Real.sqrt (distance_DE ^ 2 - distance_DF ^ 2)
  let cost_bus (distance : ℝ) : ℝ := distance * 0.20
  let cost_plane (distance : ℝ) : ℝ := distance * 0.12 + 120
  let cost_DE := min (cost_bus distance_DE) (cost_plane distance_DE)
  let cost_EF := min (cost_bus distance_EF) (cost_plane distance_EF)
  let cost_DF := min (cost_bus distance_DF) (cost_plane distance_DF)
  cost_DE + cost_EF + cost_DF

theorem least_expensive_trip_is_1627_44 :
  least_expensive_trip_cost = 1627.44 := sorry

end least_expensive_trip_is_1627_44_l10_10246


namespace percentage_increase_l10_10784

theorem percentage_increase (D J : ℝ) (hD : D = 480) (hJ : J = 417.39) :
  ((D - J) / J) * 100 = 14.99 := 
by
  sorry

end percentage_increase_l10_10784


namespace inequality_solution_set_l10_10789

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (2 * x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} 
  = {x : ℝ | (0 < x ∧ x ≤ 1/5) ∨ (2 < x ∧ x ≤ 6)} := 
by {
  sorry
}

end inequality_solution_set_l10_10789


namespace building_houses_200_people_l10_10604

theorem building_houses_200_people 
    (num_floors : ℕ)
    (apartments_per_floor : ℕ)
    (people_per_apartment : ℕ) :
    num_floors = 25 →
    apartments_per_floor = 4 →
    people_per_apartment = 2 →
    num_floors * apartments_per_floor * people_per_apartment = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end building_houses_200_people_l10_10604


namespace Cheapest_Taxi_l10_10224

noncomputable def Jim_charge : ℝ := 2.25 + 9 * 0.35
noncomputable def Susan_charge : ℝ := 3.00 + 11 * 0.40
noncomputable def John_charge : ℝ := 1.75 + 15 * 0.30

theorem Cheapest_Taxi :
  Jim_charge < Susan_charge ∧ Jim_charge < John_charge := by
sorry

end Cheapest_Taxi_l10_10224


namespace area_of_quadrilateral_rspy_l10_10111

open Real
open EuclideanGeometry

namespace TriangleArea

noncomputable def triangle_xyz (XYZ : Triangle ℝ) (XY XZ : ℝ) (a_xyz : ℝ) :=
XY = 40 ∧ XZ = 20 ∧ a_xyz = 160

def midpoints (XYZ : Triangle ℝ) (P Q : Point ℝ) : Prop :=
P = midpoint XYZ.X XYZ.Y ∧ Q = midpoint XYZ.X XYZ.Z

noncomputable def area_quadrilateral_rspy (XYZ : Triangle ℝ) (P Q R S Y : Point ℝ) (a_rspy : ℝ) : Prop :=
midpoints XYZ P Q ∧
∃ (bisector : Line ℝ), bisector ∈ angle_bisectors XYZ (XY, XZ) ∧
(R, S) = line_intersections bisector PQ YZ ∧
a_rspy = 120

theorem area_of_quadrilateral_rspy (XYZ : Triangle ℝ) (XY XZ : ℝ) (a_xyz a_rspy : ℝ) (P Q R S Y : Point ℝ) :
  triangle_xyz XYZ XY XZ a_xyz → area_quadrilateral_rspy XYZ P Q R S Y a_rspy :=
begin
  intro h,
  sorry
end

end TriangleArea

end area_of_quadrilateral_rspy_l10_10111


namespace evaluate_f_at_2_l10_10150

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem evaluate_f_at_2 : f 2 = 5 := by
  sorry

end evaluate_f_at_2_l10_10150


namespace find_s_at_3_l10_10116

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := y^2 - (y + 12)

theorem find_s_at_3 : s 3 = -6 :=
by
  sorry

end find_s_at_3_l10_10116


namespace find_share_of_C_l10_10625

-- Definitions and assumptions
def share_in_ratio (x : ℕ) : Prop :=
  let a := 2 * x
  let b := 3 * x
  let c := 4 * x
  a + b + c = 945

-- Statement to prove
theorem find_share_of_C :
  ∃ x : ℕ, share_in_ratio x ∧ 4 * x = 420 :=
by
  -- We skip the proof here.
  sorry

end find_share_of_C_l10_10625


namespace solution_set_of_inequality_l10_10268

theorem solution_set_of_inequality {x : ℝ} :
  {x | |x| * (1 - 2 * x) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x < 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l10_10268


namespace inequality_solution_l10_10702

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ (x > 8)) ↔
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) :=
sorry

end inequality_solution_l10_10702


namespace fifth_bowler_points_l10_10458

variable (P1 P2 P3 P4 P5 : ℝ)
variable (h1 : P1 = (5 / 12) * P3)
variable (h2 : P2 = (5 / 3) * P3)
variable (h3 : P4 = (5 / 3) * P3)
variable (h4 : P5 = (50 / 27) * P3)
variable (h5 : P3 ≤ 500)
variable (total_points : P1 + P2 + P3 + P4 + P5 = 2000)

theorem fifth_bowler_points : P5 = 561 :=
  sorry

end fifth_bowler_points_l10_10458


namespace determine_f_101_l10_10402

theorem determine_f_101 (f : ℕ → ℕ) (h : ∀ m n : ℕ, m * n + 1 ∣ f m * f n + 1) : 
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101 ^ k :=
sorry

end determine_f_101_l10_10402


namespace parabola_focus_directrix_distance_l10_10521

theorem parabola_focus_directrix_distance (a : ℝ) (h_pos : a > 0) (h_dist : 1 / (2 * 2 * a) = 1) : a = 1 / 4 :=
by
  sorry

end parabola_focus_directrix_distance_l10_10521


namespace train_cross_time_approx_24_seconds_l10_10939

open Real

noncomputable def time_to_cross (train_length : ℝ) (train_speed_km_h : ℝ) (man_speed_km_h : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_h * (1000 / 3600)
  let man_speed_m_s := man_speed_km_h * (1000 / 3600)
  let relative_speed := train_speed_m_s - man_speed_m_s
  train_length / relative_speed

theorem train_cross_time_approx_24_seconds : 
  abs (time_to_cross 400 63 3 - 24) < 1 :=
by
  sorry

end train_cross_time_approx_24_seconds_l10_10939


namespace product_of_xyz_is_correct_l10_10383

theorem product_of_xyz_is_correct : 
  ∃ x y z : ℤ, 
    (-3 * x + 4 * y - z = 28) ∧ 
    (3 * x - 2 * y + z = 8) ∧ 
    (x + y - z = 2) ∧ 
    (x * y * z = 2898) :=
by
  sorry

end product_of_xyz_is_correct_l10_10383


namespace circle_radius_tangent_to_semicircles_and_sides_l10_10050

noncomputable def side_length_of_square : ℝ := 4
noncomputable def side_length_of_smaller_square : ℝ := side_length_of_square / 2
noncomputable def radius_of_semicircle : ℝ := side_length_of_smaller_square / 2
noncomputable def distance_from_center_to_tangent_point : ℝ := Real.sqrt (side_length_of_smaller_square^2 + radius_of_semicircle^2)

theorem circle_radius_tangent_to_semicircles_and_sides : 
  ∃ (r : ℝ), r = (Real.sqrt 5 - 1) / 2 :=
by
  have r : ℝ := (Real.sqrt 5 - 1) / 2
  use r
  sorry -- Proof omitted

end circle_radius_tangent_to_semicircles_and_sides_l10_10050


namespace probability_third_smallest_is_4_l10_10694

theorem probability_third_smallest_is_4 :
  (∃ (integers : Finset ℕ), integers.card = 7 ∧ integers ⊆ (Finset.range 13).erase 0 ∧ 
  ∃ (S : Finset ℕ), S = (Finset.filter (λ x, x < 4) integers) ∧ S.card = 2 ∧ 
  ∃ (T : Finset ℕ), T = (Finset.filter (λ x, 4 < x) integers) ∧ T.card = 5) → 
  let total_ways := Nat.choose 12 7 in
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 5) in
  (favorable_ways) / total_ways.toReal = 7 / 33 :=
by sorry

end probability_third_smallest_is_4_l10_10694


namespace obtuse_triangle_probability_l10_10491

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l10_10491


namespace mia_high_school_has_2000_students_l10_10119

variables (M Z : ℕ)

def mia_high_school_students : Prop :=
  M = 4 * Z ∧ M + Z = 2500

theorem mia_high_school_has_2000_students (h : mia_high_school_students M Z) : 
  M = 2000 := by
  sorry

end mia_high_school_has_2000_students_l10_10119


namespace find_volume_of_pyramid_l10_10390

noncomputable def volume_of_pyramid
  (a : ℝ) (α : ℝ)
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) : ℝ :=
  (a ^ 3 * abs (Real.cos α)) / 3

--and the theorem to prove the statement
theorem find_volume_of_pyramid
  (a α : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) :
  volume_of_pyramid a α h1 h2 h3 = (a ^ 3 * abs (Real.cos α)) / 3 :=
sorry

end find_volume_of_pyramid_l10_10390


namespace avg_of_14_23_y_is_21_l10_10968

theorem avg_of_14_23_y_is_21 (y : ℝ) (h : (14 + 23 + y) / 3 = 21) : y = 26 :=
by
  sorry

end avg_of_14_23_y_is_21_l10_10968


namespace min_c_plus_d_l10_10677

theorem min_c_plus_d (c d : ℤ) (h : c * d = 36) : c + d = -37 :=
sorry

end min_c_plus_d_l10_10677


namespace bags_already_made_l10_10581

def bags_per_batch : ℕ := 10
def customer_order : ℕ := 60
def days_to_fulfill : ℕ := 4
def batches_per_day : ℕ := 1

theorem bags_already_made :
  (customer_order - (days_to_fulfill * batches_per_day * bags_per_batch)) = 20 :=
by
  sorry

end bags_already_made_l10_10581


namespace average_salary_l10_10003

theorem average_salary (total_workers technicians other_workers technicians_avg_salary other_workers_avg_salary total_salary : ℝ)
  (h_workers : total_workers = 21)
  (h_technicians : technicians = 7)
  (h_other_workers : other_workers = total_workers - technicians)
  (h_technicians_avg_salary : technicians_avg_salary = 12000)
  (h_other_workers_avg_salary : other_workers_avg_salary = 6000)
  (h_total_technicians_salary : total_salary = (technicians * technicians_avg_salary + other_workers * other_workers_avg_salary))
  (h_total_other_salary : total_salary = 168000) :
  total_salary / total_workers = 8000 := by
    sorry

end average_salary_l10_10003


namespace greatest_number_of_sets_l10_10240

-- We define the number of logic and visual puzzles.
def n_logic : ℕ := 18
def n_visual : ℕ := 9

-- The theorem states that the greatest number of identical sets Mrs. Wilson can create is the GCD of 18 and 9.
theorem greatest_number_of_sets : gcd n_logic n_visual = 9 := by
  sorry

end greatest_number_of_sets_l10_10240


namespace upstream_distance_18_l10_10049

theorem upstream_distance_18 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) : 
  upstream_distance = 18 :=
by
  have v := (downstream_distance / downstream_time) - still_water_speed
  have upstream_distance := (still_water_speed - v) * upstream_time
  sorry

end upstream_distance_18_l10_10049


namespace travel_distance_l10_10553

theorem travel_distance (x t : ℕ) (h : t = 14400) (h_eq : 12 * x + 12 * (2 * x) = t) : x = 400 :=
by
  sorry

end travel_distance_l10_10553


namespace value_of_n_l10_10203

theorem value_of_n (a : ℝ) (n : ℕ) (h : ∃ (k : ℕ), (n - 2 * k = 0) ∧ (k = 4)) : n = 8 :=
sorry

end value_of_n_l10_10203


namespace find_2a_plus_6b_l10_10841

theorem find_2a_plus_6b (a b : ℕ) (n : ℕ)
  (h1 : 3 * a + 5 * b ≡ 19 [MOD n + 1])
  (h2 : 4 * a + 2 * b ≡ 25 [MOD n + 1])
  (hn : n = 96) :
  2 * a + 6 * b = 96 :=
by
  sorry

end find_2a_plus_6b_l10_10841


namespace total_bricks_fill_box_l10_10307

-- Define brick and box volumes based on conditions
def volume_brick1 := 2 * 5 * 8
def volume_brick2 := 2 * 3 * 7
def volume_box := 10 * 11 * 14

-- Define the main proof problem
theorem total_bricks_fill_box (x y : ℕ) (h1 : volume_brick1 * x + volume_brick2 * y = volume_box) :
  x + y = 24 :=
by
  -- Left as an exercise (proof steps are not included per instructions)
  sorry

end total_bricks_fill_box_l10_10307


namespace find_center_radius_l10_10714

noncomputable def circle_center_radius (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y - 6 = 0 → 
  ∃ (h k r : ℝ), (x + 1) * (x + 1) + (y - 2) * (y - 2) = r ∧ h = -1 ∧ k = 2 ∧ r = 11

theorem find_center_radius :
  circle_center_radius x y :=
sorry

end find_center_radius_l10_10714


namespace initial_number_of_men_l10_10965

theorem initial_number_of_men (W : ℝ) (M : ℝ) (h1 : (M * 15) = W / 2) (h2 : ((M - 2) * 25) = W / 2) : M = 5 :=
sorry

end initial_number_of_men_l10_10965


namespace probability_two_draws_l10_10907

def probability_first_red_second_kd (total_cards : ℕ) (red_cards : ℕ) (king_of_diamonds : ℕ) : ℚ :=
  (red_cards / total_cards) * (king_of_diamonds / (total_cards - 1))

theorem probability_two_draws :
  let total_cards := 52
  let red_cards := 26
  let king_of_diamonds := 1
  probability_first_red_second_kd total_cards red_cards king_of_diamonds = 1 / 102 :=
by {
  sorry
}

end probability_two_draws_l10_10907


namespace channel_width_at_top_l10_10570

theorem channel_width_at_top 
  (area : ℝ) (bottom_width : ℝ) (depth : ℝ) 
  (H1 : bottom_width = 6) 
  (H2 : area = 630) 
  (H3 : depth = 70) : 
  ∃ w : ℝ, (∃ H : w + 6 > 0, area = 1 / 2 * (w + bottom_width) * depth) ∧ w = 12 :=
by
  sorry

end channel_width_at_top_l10_10570


namespace expand_expression_l10_10919

theorem expand_expression (y : ℚ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end expand_expression_l10_10919


namespace exists_reals_condition_l10_10484

-- Define the conditions in Lean
theorem exists_reals_condition (n : ℕ) (h₁ : n ≥ 3) : 
  (∃ a : Fin (n + 2) → ℝ, a 0 = a n ∧ a 1 = a (n + 1) ∧ 
  ∀ i : Fin n, a i * a (i + 1) + 1 = a (i + 2)) ↔ 3 ∣ n := 
sorry

end exists_reals_condition_l10_10484


namespace percent_boys_in_class_l10_10387

-- Define the conditions given in the problem
def initial_ratio (b g : ℕ) : Prop := b = 3 * g / 4

def total_students_after_new_girls (total : ℕ) (new_girls : ℕ) : Prop :=
  total = 42 ∧ new_girls = 4

-- Define the percentage calculation correctness
def percentage_of_boys (boys total : ℕ) (percentage : ℚ) : Prop :=
  percentage = (boys : ℚ) / (total : ℚ) * 100

-- State the theorem to be proven
theorem percent_boys_in_class
  (b g : ℕ)   -- Number of boys and initial number of girls
  (total new_girls : ℕ) -- Total students after new girls joined and number of new girls
  (percentage : ℚ) -- The percentage of boys in the class
  (h_initial_ratio : initial_ratio b g)
  (h_total_students : total_students_after_new_girls total new_girls)
  (h_goals : g + new_girls = total - b)
  (h_correct_calc : percentage = 35.71) :
  percentage_of_boys b total percentage :=
by
  sorry

end percent_boys_in_class_l10_10387


namespace evaluate_fraction_sum_l10_10631

theorem evaluate_fraction_sum : (5 / 50) + (4 / 40) + (6 / 60) = 0.3 :=
by
  sorry

end evaluate_fraction_sum_l10_10631


namespace probability_drawing_3_one_color_1_other_l10_10752

theorem probability_drawing_3_one_color_1_other (black white : ℕ) (total_balls drawn_balls : ℕ) 
    (total_ways : ℕ) (ways_3_black_1_white : ℕ) (ways_1_black_3_white : ℕ) :
    black = 10 → white = 5 → total_balls = 15 → drawn_balls = 4 →
    total_ways = Nat.choose total_balls drawn_balls →
    ways_3_black_1_white = Nat.choose black 3 * Nat.choose white 1 →
    ways_1_black_3_white = Nat.choose black 1 * Nat.choose white 3 →
    (ways_3_black_1_white + ways_1_black_3_white) / total_ways = 140 / 273 := 
by
  intros h_black h_white h_total_balls h_drawn_balls h_total_ways h_ways_3_black_1_white h_ways_1_black_3_white
  -- The proof would go here, but is not required for this task.
  sorry

end probability_drawing_3_one_color_1_other_l10_10752


namespace lowry_earnings_l10_10957

def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def small_bonsai_sold : ℕ := 3
def big_bonsai_sold : ℕ := 5

def total_earnings (small_cost : ℕ) (big_cost : ℕ) (small_sold : ℕ) (big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

theorem lowry_earnings :
  total_earnings small_bonsai_cost big_bonsai_cost small_bonsai_sold big_bonsai_sold = 190 := 
by
  sorry

end lowry_earnings_l10_10957


namespace total_cost_eq_16000_l10_10595

theorem total_cost_eq_16000 (F M T : ℕ) (n : ℕ) (hF : F = 12000) (hM : M = 200) (hT : T = 16000) :
  T = F + M * n → n = 20 :=
by
  sorry

end total_cost_eq_16000_l10_10595


namespace trigonometric_identity_l10_10930

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 6 / 5 := 
sorry

end trigonometric_identity_l10_10930


namespace true_weight_of_C_l10_10021

theorem true_weight_of_C (A1 B1 C1 A2 B2 : ℝ) (l1 l2 m1 m2 A B C : ℝ)
  (hA1 : (A + m1) * l1 = (A1 + m2) * l2)
  (hB1 : (B + m1) * l1 = (B1 + m2) * l2)
  (hC1 : (C + m1) * l1 = (C1 + m2) * l2)
  (hA2 : (A2 + m1) * l1 = (A + m2) * l2)
  (hB2 : (B2 + m1) * l1 = (B + m2) * l2) :
  C = (C1 - A1) * Real.sqrt ((A2 - B2) / (A1 - B1)) + 
      (A1 * Real.sqrt (A2 - B2) + A2 * Real.sqrt (A1 - B1)) / 
      (Real.sqrt (A1 - B1) + Real.sqrt (A2 - B2)) :=
sorry

end true_weight_of_C_l10_10021


namespace minimize_expression_10_l10_10435

theorem minimize_expression_10 (n : ℕ) (h : 0 < n) : 
  (∃ m : ℕ, 0 < m ∧ (∀ k : ℕ, 0 < k → (n = k) → (n = 10))) :=
by
  sorry

end minimize_expression_10_l10_10435


namespace choice_first_question_range_of_P2_l10_10540

theorem choice_first_question (P1 P2 a b : ℚ) (hP1 : P1 = 1/2) (hP2 : P2 = 1/3) :
  (P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) > 0) ↔ a > b / 2 :=
sorry

theorem range_of_P2 (a b P1 P2 : ℚ) (ha : a = 10) (hb : b = 20) (hP1 : P1 = 2/5) :
  P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) ≥ 0 ↔ (0 ≤ P2 ∧ P2 ≤ P1 / (2 - P1)) :=
sorry

end choice_first_question_range_of_P2_l10_10540


namespace common_difference_arithmetic_seq_l10_10988

theorem common_difference_arithmetic_seq (S n a1 d : ℕ) (h_sum : S = 650) (h_n : n = 20) (h_a1 : a1 = 4) :
  S = (n / 2) * (2 * a1 + (n - 1) * d) → d = 3 := by
  intros h_formula
  sorry

end common_difference_arithmetic_seq_l10_10988


namespace squirrel_acorns_initial_stash_l10_10764

theorem squirrel_acorns_initial_stash (A : ℕ) 
  (h1 : 3 * (A / 3 - 60) = 30) : A = 210 := 
sorry

end squirrel_acorns_initial_stash_l10_10764


namespace rectangle_area_exceeds_m_l10_10669

theorem rectangle_area_exceeds_m (m : ℤ) (h_m : m > 12) :
  ∃ x y : ℤ, x * y > m ∧ (x - 1) * y < m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_area_exceeds_m_l10_10669


namespace total_investment_sum_l10_10737

-- Definitions of the problem
variable (Raghu Trishul Vishal : ℕ)
variable (h1 : Raghu = 2000)
variable (h2 : Trishul = Nat.div (Raghu * 9) 10)
variable (h3 : Vishal = Nat.div (Trishul * 11) 10)

-- The theorem to prove
theorem total_investment_sum :
  Vishal + Trishul + Raghu = 5780 :=
by
  sorry

end total_investment_sum_l10_10737


namespace total_donation_correct_l10_10092

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end total_donation_correct_l10_10092


namespace abs_inequality_solution_l10_10067

theorem abs_inequality_solution {x : ℝ} (h : |x + 1| < 5) : -6 < x ∧ x < 4 :=
by
  sorry

end abs_inequality_solution_l10_10067


namespace union_sets_l10_10937

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem union_sets :
  M ∪ N = {x | x ≤ 1} :=
by
  sorry

end union_sets_l10_10937


namespace correct_multiplication_l10_10909

theorem correct_multiplication :
  ∃ (n : ℕ), 98765 * n = 888885 ∧ (98765 * n = 867559827931 → n = 9) :=
by
  sorry

end correct_multiplication_l10_10909


namespace obtuse_triangle_probability_l10_10490

/-- 
Given four points chosen uniformly at random on a circle,
prove that the probability that no three of these points form an obtuse triangle with the circle's center is 3/32.
-/
theorem obtuse_triangle_probability : 
  let points := 4 in
  ∀ (P : Fin points → ℝ × ℝ) (random_uniform : ∀ i, random.uniform_on_circle i),
  (probability (no_three_form_obtuse_triangle P) = 3 / 32) :=
sorry

end obtuse_triangle_probability_l10_10490


namespace conditional_probability_B_given_A_l10_10799

/-
Given a box containing 6 balls: 2 red, 2 yellow, and 2 blue.
One ball is drawn with replacement for 3 times.
Let event A be "the color of the ball drawn in the first draw is the same as the color of the ball drawn in the second draw".
Let event B be "the color of the balls drawn in all three draws is the same".
Prove that the conditional probability P(B|A) is 1/3.
-/
noncomputable def total_balls := 6
noncomputable def red_balls := 2
noncomputable def yellow_balls := 2
noncomputable def blue_balls := 2

noncomputable def event_A (n : ℕ) : ℕ := 
  3 * 2 * 2 * total_balls

noncomputable def event_AB (n : ℕ) : ℕ := 
  3 * 2 * 2 * 2

noncomputable def P_B_given_A : ℚ := 
  event_AB total_balls / event_A total_balls

theorem conditional_probability_B_given_A :
  P_B_given_A = 1 / 3 :=
by sorry

end conditional_probability_B_given_A_l10_10799


namespace cinema_total_cost_l10_10245

theorem cinema_total_cost 
  (total_students : ℕ)
  (ticket_cost : ℕ)
  (half_price_interval : ℕ)
  (free_interval : ℕ)
  (half_price_cost : ℕ)
  (free_cost : ℕ)
  (total_cost : ℕ)
  (H_total_students : total_students = 84)
  (H_ticket_cost : ticket_cost = 50)
  (H_half_price_interval : half_price_interval = 12)
  (H_free_interval : free_interval = 35)
  (H_half_price_cost : half_price_cost = ticket_cost / 2)
  (H_free_cost : free_cost = 0)
  (H_total_cost : total_cost = 3925) :
  total_cost = ((total_students / half_price_interval) * half_price_cost +
                (total_students / free_interval) * free_cost +
                (total_students - (total_students / half_price_interval + total_students / free_interval)) * ticket_cost) :=
by 
  sorry

end cinema_total_cost_l10_10245


namespace bowling_ball_surface_area_l10_10162

theorem bowling_ball_surface_area (diameter : ℝ) (h : diameter = 9) :
    let r := diameter / 2
    let surface_area := 4 * Real.pi * r^2
    surface_area = 81 * Real.pi := by
  sorry

end bowling_ball_surface_area_l10_10162


namespace nat_pairs_solution_l10_10921

theorem nat_pairs_solution (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) :=
by
  sorry

end nat_pairs_solution_l10_10921


namespace birgit_hiking_time_l10_10859

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end birgit_hiking_time_l10_10859


namespace find_f_log_l10_10196

def even_function (f : ℝ → ℝ) :=
  ∀ (x : ℝ), f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ (x : ℝ), f (x + p) = f x

theorem find_f_log (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 2)
  (h_condition : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f x = 3 * x + 4 / 9) :
  f (Real.log 5 / Real.log (1 / 3)) = -5 / 9 :=
by
  sorry

end find_f_log_l10_10196


namespace problem_a_problem_b_l10_10632

variable (α : ℝ)

theorem problem_a (hα : 0 < α ∧ α < π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = Real.tan (α / 2) :=
sorry

theorem problem_b (hα : π < α ∧ α < 2 * π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = -Real.tan (α / 2) :=
sorry

end problem_a_problem_b_l10_10632


namespace find_m_values_l10_10062

theorem find_m_values :
  ∃ m : ℝ, (∀ (α β : ℝ), (3 * α^2 + m * α - 4 = 0 ∧ 3 * β^2 + m * β - 4 = 0) ∧ (α * β = -4 / 3) ∧ (α + β = -m / 3) ∧ (α * β = 2 * (α^3 + β^3))) ↔
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
sorry

end find_m_values_l10_10062


namespace drug_price_reduction_l10_10872

theorem drug_price_reduction :
  ∃ x : ℝ, 56 * (1 - x)^2 = 31.5 :=
by
  sorry

end drug_price_reduction_l10_10872


namespace b_contribution_is_correct_l10_10886

-- Definitions based on the conditions
def A_investment : ℕ := 35000
def B_join_after_months : ℕ := 5
def profit_ratio_A_B : ℕ := 2
def profit_ratio_B_A : ℕ := 3
def A_total_months : ℕ := 12
def B_total_months : ℕ := 7
def profit_ratio := (profit_ratio_A_B, profit_ratio_B_A)
def total_investment_time_ratio : ℕ := 12 * 35000 / 7

-- The property to be proven
theorem b_contribution_is_correct (X : ℕ) (h : 35000 * 12 / (X * 7) = 2 / 3) : X = 90000 :=
by
  sorry

end b_contribution_is_correct_l10_10886


namespace clock_chime_time_l10_10051

/-- The proven time it takes for a wall clock to strike 12 times at 12 o'clock -/
theorem clock_chime_time :
  (∃ (interval_time : ℝ), (interval_time = 3) ∧ (∃ (time_12_times : ℝ), (time_12_times = interval_time * (12 - 1)) ∧ (time_12_times = 33))) :=
by
  sorry

end clock_chime_time_l10_10051


namespace polynomial_identity_equals_neg_one_l10_10098

theorem polynomial_identity_equals_neg_one
  (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by
  intro h
  sorry

end polynomial_identity_equals_neg_one_l10_10098


namespace cheese_placement_distinct_ways_l10_10440

theorem cheese_placement_distinct_ways (total_wedges : ℕ) (selected_wedges : ℕ) : 
  total_wedges = 18 ∧ selected_wedges = 6 → 
  ∃ (distinct_ways : ℕ), distinct_ways = 130 :=
by
  sorry

end cheese_placement_distinct_ways_l10_10440


namespace alice_oranges_l10_10618

theorem alice_oranges (E A : ℕ) 
  (h1 : A = 2 * E) 
  (h2 : E + A = 180) : 
  A = 120 :=
by
  sorry

end alice_oranges_l10_10618


namespace solve_for_x_l10_10250

theorem solve_for_x (x : ℝ) : 
  (x - 35) / 3 = (3 * x + 10) / 8 → x = -310 := by
  sorry

end solve_for_x_l10_10250


namespace sum_possible_n_k_l10_10978

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l10_10978


namespace range_of_g_l10_10843

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x) * (arcsin x)

theorem range_of_g :
  ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), 0 ≤ g x ∧ g x ≤ (π ^ 2) / 8 :=
by
  intros x hx
  have h1 : arccos x + arcsin x = π / 2 :=
    arcsin_arccos_add x hx.left hx.right
  sorry

end range_of_g_l10_10843


namespace geometric_sequence_common_ratio_l10_10303

-- Define a sequence as a list of real numbers
def seq : List ℚ := [8, -20, 50, -125]

-- Define the common ratio of a geometric sequence
def common_ratio (l : List ℚ) : ℚ := l.head! / l.tail!.head!

-- The theorem to prove the common ratio is -5/2
theorem geometric_sequence_common_ratio :
  common_ratio seq = -5 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l10_10303


namespace cosine_of_A_l10_10662

theorem cosine_of_A (a b : ℝ) (A B : ℝ) (h1 : b = (5 / 8) * a) (h2 : A = 2 * B) :
  Real.cos A = 7 / 25 :=
by
  sorry

end cosine_of_A_l10_10662


namespace polynomial_degree_l10_10284

noncomputable def polynomial1 : Polynomial ℤ := 3 * Polynomial.monomial 5 1 + 2 * Polynomial.monomial 4 1 - Polynomial.monomial 1 1 + Polynomial.C 5
noncomputable def polynomial2 : Polynomial ℤ := 4 * Polynomial.monomial 11 1 - 2 * Polynomial.monomial 8 1 + 5 * Polynomial.monomial 5 1 - Polynomial.C 9
noncomputable def polynomial3 : Polynomial ℤ := (Polynomial.monomial 2 1 - Polynomial.C 3) ^ 9

theorem polynomial_degree :
  (polynomial1 * polynomial2 - polynomial3).degree = 18 := by
  sorry

end polynomial_degree_l10_10284


namespace sahil_selling_price_l10_10854

noncomputable def sales_tax : ℝ := 0.10 * 18000
noncomputable def initial_cost_with_tax : ℝ := 18000 + sales_tax

noncomputable def broken_part_cost : ℝ := 3000
noncomputable def software_update_cost : ℝ := 4000
noncomputable def total_repair_cost : ℝ := broken_part_cost + software_update_cost
noncomputable def service_tax_on_repair : ℝ := 0.05 * total_repair_cost
noncomputable def total_repair_cost_with_tax : ℝ := total_repair_cost + service_tax_on_repair

noncomputable def transportation_charges : ℝ := 1500
noncomputable def total_cost_before_depreciation : ℝ := initial_cost_with_tax + total_repair_cost_with_tax + transportation_charges

noncomputable def depreciation_first_year : ℝ := 0.15 * total_cost_before_depreciation
noncomputable def value_after_first_year : ℝ := total_cost_before_depreciation - depreciation_first_year

noncomputable def depreciation_second_year : ℝ := 0.15 * value_after_first_year
noncomputable def value_after_second_year : ℝ := value_after_first_year - depreciation_second_year

noncomputable def profit : ℝ := 0.50 * value_after_second_year
noncomputable def selling_price : ℝ := value_after_second_year + profit

theorem sahil_selling_price : selling_price = 31049.44 := by
  sorry

end sahil_selling_price_l10_10854


namespace negate_proposition_p_l10_10392

theorem negate_proposition_p (f : ℝ → ℝ) :
  (¬ ∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) >= 0) ↔ ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
sorry

end negate_proposition_p_l10_10392


namespace quadratic_trinomial_has_two_roots_l10_10317

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l10_10317


namespace valid_propositions_l10_10470

theorem valid_propositions :
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧ (∃ n : ℝ, ∀ m : ℝ, m * n = m) :=
by
  sorry

end valid_propositions_l10_10470


namespace intersection_complement_l10_10844

def real_set_M : Set ℝ := {x | 1 < x}
def real_set_N : Set ℝ := {x | x > 4}

theorem intersection_complement (x : ℝ) : x ∈ (real_set_M ∩ (real_set_Nᶜ)) ↔ 1 < x ∧ x ≤ 4 :=
by
  sorry

end intersection_complement_l10_10844


namespace triangle_side_sum_l10_10734

def sum_of_remaining_sides_of_triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α = 40 ∧ β = 50 ∧ γ = 180 - α - β ∧ c = 8 * Real.sqrt 3 →
  (a + b) = 34.3

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :
  sum_of_remaining_sides_of_triangle A B C a b c α β γ :=
sorry

end triangle_side_sum_l10_10734


namespace onions_total_l10_10555

-- Define the number of onions grown by Sara, Sally, and Fred
def sara_onions : ℕ := 4
def sally_onions : ℕ := 5
def fred_onions : ℕ := 9

-- Define the total onions grown
def total_onions : ℕ := sara_onions + sally_onions + fred_onions

-- Theorem stating the total number of onions grown
theorem onions_total : total_onions = 18 := by
  sorry

end onions_total_l10_10555


namespace more_ones_than_twos_in_first_billion_l10_10129

def digital_root (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

theorem more_ones_than_twos_in_first_billion : 
  ∃ (count_1 count_2 : ℕ), count_1 > count_2 ∧ 
  count_1 = (Finset.Icc 1 1000000000).filter (λ n, digital_root n = 1).card ∧ 
  count_2 = (Finset.Icc 1 1000000000).filter (λ n, digital_root n = 2).card :=
by
  sorry

end more_ones_than_twos_in_first_billion_l10_10129


namespace calculate_expression_l10_10622

theorem calculate_expression : (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end calculate_expression_l10_10622


namespace first_train_cross_time_l10_10816

noncomputable def length_first_train : ℝ := 800
noncomputable def speed_first_train_kmph : ℝ := 120
noncomputable def length_second_train : ℝ := 1000
noncomputable def speed_second_train_kmph : ℝ := 80
noncomputable def length_third_train : ℝ := 600
noncomputable def speed_third_train_kmph : ℝ := 150

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

noncomputable def speed_first_train_mps : ℝ := speed_kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train_mps : ℝ := speed_kmph_to_mps speed_second_train_kmph
noncomputable def speed_third_train_mps : ℝ := speed_kmph_to_mps speed_third_train_kmph

noncomputable def relative_speed_same_direction : ℝ := speed_first_train_mps - speed_second_train_mps
noncomputable def relative_speed_opposite_direction : ℝ := speed_first_train_mps + speed_third_train_mps

noncomputable def time_to_cross_second_train : ℝ := (length_first_train + length_second_train) / relative_speed_same_direction
noncomputable def time_to_cross_third_train : ℝ := (length_first_train + length_third_train) / relative_speed_opposite_direction

noncomputable def total_time_to_cross : ℝ := time_to_cross_second_train + time_to_cross_third_train

theorem first_train_cross_time : total_time_to_cross = 180.67 := by
  sorry

end first_train_cross_time_l10_10816


namespace alice_bob_meet_l10_10616

theorem alice_bob_meet (n : ℕ) (h_n : n = 18) (alice_move : ℕ) (bob_move : ℕ)
  (h_alice : alice_move = 7) (h_bob : bob_move = 13) :
  ∃ k : ℕ, alice_move * k % n = (n - bob_move) * k % n :=
by
  sorry

end alice_bob_meet_l10_10616


namespace clock_angle_230_l10_10282

theorem clock_angle_230 (h12 : ℕ := 12) (deg360 : ℕ := 360) 
  (hour_mark_deg : ℕ := deg360 / h12) (hour_halfway : ℕ := hour_mark_deg / 2)
  (hour_deg_230 : ℕ := hour_mark_deg * 3) (total_angle : ℕ := hour_halfway + hour_deg_230) :
  total_angle = 105 := 
by
  sorry

end clock_angle_230_l10_10282


namespace product_of_numbers_l10_10716

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) : x * y = 26 :=
sorry

end product_of_numbers_l10_10716


namespace trains_meet_time_l10_10746

theorem trains_meet_time :
  (∀ (D : ℝ) (s1 s2 t1 t2 : ℝ),
    D = 155 ∧ 
    s1 = 20 ∧ 
    s2 = 25 ∧ 
    t1 = 7 ∧ 
    t2 = 8 →
    (∃ t : ℝ, 20 * t + 25 * t = D - 20)) →
  8 + 3 = 11 :=
by {
  sorry
}

end trains_meet_time_l10_10746


namespace number_of_men_in_first_group_l10_10820

/-- The number of men in the first group that can complete a piece of work in 5 days alongside 16 boys,
    given that 13 men and 24 boys can complete the same work in 4 days, and the ratio of daily work done 
    by a man to a boy is 2:1, is 12. -/
theorem number_of_men_in_first_group
  (x : ℕ)  -- define x as the amount of work a boy can do in a day
  (m : ℕ)  -- define m as the number of men in the first group
  (h1 : ∀ (x : ℕ), 5 * (m * 2 * x + 16 * x) = 4 * (13 * 2 * x + 24 * x))
  (h2 : 2 * x = x + x) : m = 12 :=
sorry

end number_of_men_in_first_group_l10_10820


namespace perimeter_of_rectangle_l10_10270

-- Define the properties of the rectangle based on the given conditions
variable (l w : ℝ)
axiom h1 : l + w = 7
axiom h2 : 2 * l + w = 9.5

-- Define the function for perimeter of the rectangle
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

-- Formal statement of the proof problem
theorem perimeter_of_rectangle : perimeter l w = 14 := by
  -- Given conditions
  have h3 : l = 2.5 := sorry
  have h4 : w = 4.5 := sorry
  -- Conclusion based on the conditions
  show perimeter l w = 14 from sorry

end perimeter_of_rectangle_l10_10270


namespace sophie_aunt_money_l10_10422

noncomputable def totalMoneyGiven (shirts: ℕ) (shirtCost: ℝ) (trousers: ℕ) (trouserCost: ℝ) (additionalItems: ℕ) (additionalItemCost: ℝ) : ℝ :=
  shirts * shirtCost + trousers * trouserCost + additionalItems * additionalItemCost

theorem sophie_aunt_money : totalMoneyGiven 2 18.50 1 63 4 40 = 260 := 
by
  sorry

end sophie_aunt_money_l10_10422


namespace system_solution_l10_10421

theorem system_solution (x : Fin 1995 → ℤ) :
  (∀ i : (Fin 1995),
    x (i + 1) ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) →
  (∀ n : (Fin 1995),
    (x n = 0 ∧ x (n + 1) = 1 ∧ x (n + 2) = -1) ∨
    (x n = 0 ∧ x (n + 1) = -1 ∧ x (n + 2) = 1)) :=
by sorry

end system_solution_l10_10421


namespace abs_expression_equals_one_l10_10924

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end abs_expression_equals_one_l10_10924


namespace valid_n_values_l10_10565

theorem valid_n_values :
  {n : ℕ | ∀ a : ℕ, a^(n+1) ≡ a [MOD n]} = {1, 2, 6, 42, 1806} :=
sorry

end valid_n_values_l10_10565


namespace box_volume_increase_l10_10329

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l10_10329


namespace projectile_height_at_time_l10_10427

theorem projectile_height_at_time
  (y : ℝ)
  (t : ℝ)
  (h_eq : y = -16 * t ^ 2 + 64 * t) :
  ∃ t₀ : ℝ, t₀ = 3 ∧ y = 49 :=
by sorry

end projectile_height_at_time_l10_10427


namespace find_xyz_l10_10077

variables (x y z s : ℝ)

theorem find_xyz (h₁ : (x + y + z) * (x * y + x * z + y * z) = 12)
    (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
    (hs : x + y + z = s) : xyz = -8 :=
by
  sorry

end find_xyz_l10_10077


namespace total_donation_l10_10089

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end total_donation_l10_10089


namespace abs_eq_iff_x_eq_2_l10_10585

theorem abs_eq_iff_x_eq_2 (x : ℝ) : |x - 1| = |x - 3| → x = 2 := by
  sorry

end abs_eq_iff_x_eq_2_l10_10585


namespace limit_seq_l10_10888

/-- Define the sequence a(n) as given in the problem -/ 
def seq (n : ℕ) := (3 - 4 * n : ℝ)^2 / ((n - 3 : ℝ)^3 - (n + 3 : ℝ)^3)

/-- The limit statement to prove -/
theorem limit_seq : 
  tendsto (fun n => seq n) at_top (𝓝 (-8/9 : ℝ)) :=
sorry

end limit_seq_l10_10888


namespace find_a_plus_b_l10_10648

theorem find_a_plus_b (a b : ℝ) :
  (∀ x y : ℝ, (ax + y + 1 = 0) ∧ (2x - by - 1 = 0) → (x = 1 → y = 1)) → (a + b = -1) :=
by
  intros
  sorry

end find_a_plus_b_l10_10648


namespace radius_of_circle_from_spherical_coords_l10_10266

theorem radius_of_circle_from_spherical_coords :
  ∀ (θ: ℝ), let ρ := 1, φ := π / 3 in
  (√(ρ * sin φ * cos θ)^2 + (ρ * sin φ * sin θ)^2) = √3 / 2 :=
by
  intros θ
  let ρ := 1
  let φ := π / 3
  sorry

end radius_of_circle_from_spherical_coords_l10_10266


namespace not_p_sufficient_not_necessary_for_not_q_l10_10659

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) (h1 : q → p) (h2 : ¬ (p → q)) : 
  (¬p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l10_10659


namespace trinomial_has_two_roots_l10_10312

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l10_10312


namespace probability_of_hitting_target_at_least_twice_l10_10168

theorem probability_of_hitting_target_at_least_twice :
  let p : ℝ := 0.6 in
  let q : ℝ := 0.4 in
  let n : ℕ := 3 in
  ∑ k in {2, 3}, nat.choose n k * p^k * q^(n-k) = 81 / 125 :=
by
  sorry

end probability_of_hitting_target_at_least_twice_l10_10168


namespace probability_third_smallest_is_four_l10_10696

/--
Seven distinct integers are picked at random from the set {1, 2, 3, ..., 12}.
The probability that the third smallest number is 4 is 7/33.
-/
theorem probability_third_smallest_is_four : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.to_finset in
  ∀ s : Finset ℕ, s ⊆ S ∧ s.card = 7 →
  let event := { s | s.nth_le 2 (by simp [s.card_eq_coe] ; norm_num) = 4 }.to_finset in
  (event.card : ℚ) / (S.choose 7).card = 7 / 33 :=
by
  intros S S_prop event
  sorry

end probability_third_smallest_is_four_l10_10696


namespace taller_cycle_shadow_length_l10_10580

theorem taller_cycle_shadow_length 
  (h_taller : ℝ) (h_shorter : ℝ) (shadow_shorter : ℝ) (shadow_taller : ℝ) 
  (h_taller_val : h_taller = 2.5) 
  (h_shorter_val : h_shorter = 2) 
  (shadow_shorter_val : shadow_shorter = 4)
  (similar_triangles : h_taller / shadow_taller = h_shorter / shadow_shorter) :
  shadow_taller = 5 := 
by 
  sorry

end taller_cycle_shadow_length_l10_10580


namespace fraction_arithmetic_l10_10779

-- Definitions for given fractions
def frac1 := 8 / 19
def frac2 := 5 / 57
def frac3 := 1 / 3

-- Theorem statement that needs to be proven
theorem fraction_arithmetic : frac1 - frac2 + frac3 = 2 / 3 :=
by
  -- Lean proof goes here
  sorry

end fraction_arithmetic_l10_10779


namespace cos_alpha_add_beta_div2_l10_10078

open Real 

theorem cos_alpha_add_beta_div2 (α β : ℝ) 
  (h_range : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h_cos1 : cos (π/4 + α) = 1/3)
  (h_cos2 : cos (π/4 - β/2) = sqrt 3 / 3) :
  cos (α + β/2) = 5 * sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_div2_l10_10078


namespace value_v3_at_1_horners_method_l10_10875

def f (x : ℝ) : ℝ := 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem value_v3_at_1_horners_method :
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  v3 = 7.9 :=
by
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  exact sorry

end value_v3_at_1_horners_method_l10_10875


namespace sector_to_cone_ratio_l10_10762

noncomputable def sector_angle : ℝ := 135
noncomputable def sector_area (S1 : ℝ) : ℝ := S1
noncomputable def cone_surface_area (S2 : ℝ) : ℝ := S2

theorem sector_to_cone_ratio (S1 S2 : ℝ) :
  sector_area S1 = (3 / 8) * (π * 1^2) →
  cone_surface_area S2 = (3 / 8) * (π * 1^2) + (9 / 64 * π) →
  (S1 / S2) = (8 / 11) :=
by
  intros h1 h2
  sorry

end sector_to_cone_ratio_l10_10762


namespace find_x_l10_10594

theorem find_x (x : ℝ) (h : 0.90 * 600 = 0.50 * x) : x = 1080 :=
sorry

end find_x_l10_10594


namespace ratio_proof_l10_10186

variables (x y m n : ℝ)

def ratio_equation1 (x y m n : ℝ) : Prop :=
  (5 * x + 7 * y) / (3 * x + 2 * y) = m / n

def target_equation (x y m n : ℝ) : Prop :=
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n)

theorem ratio_proof (x y m n : ℝ) (h: ratio_equation1 x y m n) :
  target_equation x y m n :=
by
  sorry

end ratio_proof_l10_10186


namespace sum_m_n_l10_10851

-- We define the conditions and problem
variables (m n : ℕ)

-- Conditions
def conditions := m > 50 ∧ n > 50 ∧ Nat.lcm m n = 480 ∧ Nat.gcd m n = 12

-- Statement to prove
theorem sum_m_n : conditions m n → m + n = 156 := by sorry

end sum_m_n_l10_10851


namespace find_u_value_l10_10726

theorem find_u_value (h : ∃ n : ℕ, n = 2012) : ∃ u : ℕ, u = 2015 := 
by
  sorry

end find_u_value_l10_10726


namespace pen_tip_movement_l10_10482

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end pen_tip_movement_l10_10482


namespace least_number_to_add_l10_10593

theorem least_number_to_add (n : ℕ) (H : n = 433124) : ∃ k, k = 15 ∧ (n + k) % 17 = 0 := by
  sorry

end least_number_to_add_l10_10593


namespace profit_percent_l10_10903

-- Definitions based on the conditions in the problem
def marked_price_per_pen := ℝ
def total_pens := 52
def cost_equivalent_pens := 46
def discount_percentage := 1 / 100

-- Values calculated from conditions
def cost_price (P : ℝ) := cost_equivalent_pens * P
def selling_price_per_pen (P : ℝ) := P * (1 - discount_percentage)
def total_selling_price (P : ℝ) := total_pens * selling_price_per_pen P

-- The proof statement
theorem profit_percent (P : ℝ) (hP : P > 0) :
  ((total_selling_price P - cost_price P) / (cost_price P)) * 100 = 11.91 := by
    sorry

end profit_percent_l10_10903


namespace find_a_l10_10955

theorem find_a (a : ℕ) : 
  (a >= 100 ∧ a <= 999) ∧ 7 ∣ (504000 + a) ∧ 9 ∣ (504000 + a) ∧ 11 ∣ (504000 + a) ↔ a = 711 :=
by {
  sorry
}

end find_a_l10_10955


namespace cube_face_sum_l10_10627

theorem cube_face_sum
  (a d b e c f : ℕ)
  (pos_a : 0 < a) (pos_d : 0 < d) (pos_b : 0 < b) (pos_e : 0 < e) (pos_c : 0 < c) (pos_f : 0 < f)
  (hd : (a + d) * (b + e) * (c + f) = 2107) :
  a + d + b + e + c + f = 57 :=
sorry

end cube_face_sum_l10_10627


namespace car_travel_distance_l10_10970

-- Definitions of conditions
def speed_kmph : ℝ := 27 -- 27 kilometers per hour
def time_sec : ℝ := 50 -- 50 seconds

-- Equivalent in Lean 4 for car moving distance in meters
theorem car_travel_distance : (speed_kmph * 1000 / 3600) * time_sec = 375 := by
  sorry

end car_travel_distance_l10_10970


namespace range_of_a_l10_10010

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l10_10010


namespace percentage_increase_biking_time_l10_10443

theorem percentage_increase_biking_time
  (time_young_hours : ℕ)
  (distance_young_miles : ℕ)
  (time_now_hours : ℕ)
  (distance_now_miles : ℕ)
  (time_young_minutes : ℕ := time_young_hours * 60)
  (time_now_minutes : ℕ := time_now_hours * 60)
  (time_per_mile_young : ℕ := time_young_minutes / distance_young_miles)
  (time_per_mile_now : ℕ := time_now_minutes / distance_now_miles)
  (increase_in_time_per_mile : ℕ := time_per_mile_now - time_per_mile_young)
  (percentage_increase : ℕ := (increase_in_time_per_mile * 100) / time_per_mile_young) :
  percentage_increase = 100 :=
by
  -- substitution of values for conditions
  have time_young_hours := 2
  have distance_young_miles := 20
  have time_now_hours := 3
  have distance_now_miles := 15
  sorry

end percentage_increase_biking_time_l10_10443


namespace largest_consecutive_integers_sum_to_45_l10_10012

theorem largest_consecutive_integers_sum_to_45 (x n : ℕ) (h : 45 = n * (2 * x + n - 1) / 2) : n ≤ 9 :=
sorry

end largest_consecutive_integers_sum_to_45_l10_10012


namespace total_jumps_l10_10523

theorem total_jumps (hattie_1 : ℕ) (lorelei_1 : ℕ) (hattie_2 : ℕ) (lorelei_2 : ℕ) (hattie_3 : ℕ) (lorelei_3 : ℕ) :
  hattie_1 = 180 →
  lorelei_1 = 3 / 4 * hattie_1 →
  hattie_2 = 2 / 3 * hattie_1 →
  lorelei_2 = hattie_2 + 50 →
  hattie_3 = hattie_2 + 1 / 3 * hattie_2 →
  lorelei_3 = 4 / 5 * lorelei_1 →
  hattie_1 + hattie_2 + hattie_3 + lorelei_1 + lorelei_2 + lorelei_3 = 873 :=
by
  intros h1 l1 h2 l2 h3 l3
  sorry

end total_jumps_l10_10523


namespace pyramid_volume_correct_l10_10870

noncomputable def volume_of_pyramid (l α β : ℝ) (Hα : α = π/8) (Hβ : β = π/4) :=
  (1 / 3) * (l^3 / 24) * Real.sqrt (Real.sqrt 2 + 1)

theorem pyramid_volume_correct :
  ∀ (l : ℝ), l = 6 → volume_of_pyramid l (π/8) (π/4) (rfl) (rfl) = 9 * Real.sqrt (Real.sqrt 2 + 1) :=
by
  intros l hl
  rw [hl]
  norm_num
  sorry

end pyramid_volume_correct_l10_10870


namespace ratio_of_segments_of_hypotenuse_l10_10536

theorem ratio_of_segments_of_hypotenuse (k : Real) :
  let AB := 3 * k
  let BC := 2 * k
  let AC := Real.sqrt (AB^2 + BC^2)
  ∃ D : Real, 
    let BD := (2 / 3) * D
    let AD := (4 / 9) * D
    let CD := D
    ∀ AD CD, AD / CD = 4 / 9 :=
by
  sorry

end ratio_of_segments_of_hypotenuse_l10_10536


namespace find_p_l10_10156

variables (m n p : ℝ)

def line_equation (x y : ℝ) : Prop :=
  x = y / 3 - 2 / 5

theorem find_p
  (h1 : line_equation m n)
  (h2 : line_equation (m + p) (n + 9)) :
  p = 3 :=
by
  sorry

end find_p_l10_10156


namespace num_of_ints_l10_10654

theorem num_of_ints (n : ℤ) (h : -100 < n^3) (h2 : n^3 < 100) : 
    (finset.card (finset.filter (λ x : ℤ, -100 < x^3 ∧ x^3 < 100) (finset.Icc (-4) 4))) = 9 :=
sorry

end num_of_ints_l10_10654


namespace brownies_count_l10_10621

theorem brownies_count {B : ℕ} 
  (h1 : B/2 = (B - B / 2))
  (h2 : B/4 = (B - B / 2) / 2)
  (h3 : B/4 - 2 = B/4 - 2)
  (h4 : B/4 - 2 = 3) : 
  B = 20 := 
by 
  sorry

end brownies_count_l10_10621


namespace alice_winning_strategy_l10_10233

theorem alice_winning_strategy (n : ℕ) (hn : n ≥ 2) : 
  (Alice_has_winning_strategy ↔ n % 4 = 3) :=
sorry

end alice_winning_strategy_l10_10233


namespace tina_money_left_l10_10276

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end tina_money_left_l10_10276


namespace total_volume_collection_l10_10883

-- Define the conditions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def cost_per_box : ℚ := 0.5
def minimum_total_cost : ℚ := 255

-- Define the volume of one box
def volume_of_one_box : ℕ := box_length * box_width * box_height

-- Define the number of boxes needed
def number_of_boxes : ℚ := minimum_total_cost / cost_per_box

-- Define the total volume of the collection
def total_volume : ℚ := volume_of_one_box * number_of_boxes

-- The goal is to prove that the total volume of the collection is as calculated
theorem total_volume_collection :
  total_volume = 3060000 := by
  sorry

end total_volume_collection_l10_10883


namespace four_digit_integer_l10_10008

theorem four_digit_integer (a b c d : ℕ) 
    (h1 : a + b + c + d = 16) 
    (h2 : b + c = 10) 
    (h3 : a - d = 2) 
    (h4 : (a - b + c - d) % 11 = 0) : 
    a = 4 ∧ b = 4 ∧ c = 6 ∧ d = 2 :=
sorry

end four_digit_integer_l10_10008


namespace can_weight_is_two_l10_10673

theorem can_weight_is_two (c : ℕ) (h1 : 100 = 20 * c + 6 * ((100 - 20 * c) / 6)) (h2 : 160 = 10 * ((100 - 20 * c) / 6) + 3 * 20) : c = 2 :=
by
  sorry

end can_weight_is_two_l10_10673


namespace initial_discount_percentage_l10_10457

-- Statement of the problem
theorem initial_discount_percentage (d : ℝ) (x : ℝ)
  (h₁ : d > 0)
  (h_staff_price : d * ((100 - x) / 100) * 0.5 = 0.225 * d) :
  x = 55 := 
sorry

end initial_discount_percentage_l10_10457


namespace probability_non_obtuse_l10_10498

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l10_10498


namespace program_output_l10_10083

theorem program_output (x : ℤ) : 
  (if x < 0 then -1 else if x = 0 then 0 else 1) = 1 ↔ x = 3 :=
by
  sorry

end program_output_l10_10083


namespace sum_n_k_eq_eight_l10_10975

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l10_10975


namespace rajesh_walked_distance_l10_10248

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end rajesh_walked_distance_l10_10248


namespace largest_x_floor_condition_l10_10354

theorem largest_x_floor_condition :
  ∃ x : ℝ, (⌊x⌋ : ℝ) / x = 8 / 9 ∧
      (∀ y : ℝ, (⌊y⌋ : ℝ) / y = 8 / 9 → y ≤ x) →
  x = 63 / 8 :=
by
  sorry

end largest_x_floor_condition_l10_10354


namespace OfficerHoppsTotalTickets_l10_10684

theorem OfficerHoppsTotalTickets : 
  (15 * 8 + (31 - 15) * 5 = 200) :=
  by
    sorry

end OfficerHoppsTotalTickets_l10_10684


namespace exists_close_points_l10_10194

theorem exists_close_points (r : ℝ) (h : r > 0) (points : Fin 5 → EuclideanSpace ℝ (Fin 3)) (hf : ∀ i, dist (points i) (0 : EuclideanSpace ℝ (Fin 3)) = r) :
  ∃ i j : Fin 5, i ≠ j ∧ dist (points i) (points j) ≤ r * Real.sqrt 2 :=
by 
  sorry

end exists_close_points_l10_10194


namespace houses_with_only_one_pet_l10_10219

theorem houses_with_only_one_pet (h_total : ∃ t : ℕ, t = 75)
                                 (h_dogs : ∃ d : ℕ, d = 40)
                                 (h_cats : ∃ c : ℕ, c = 30)
                                 (h_dogs_and_cats : ∃ dc : ℕ, dc = 10)
                                 (h_birds : ∃ b : ℕ, b = 8)
                                 (h_cats_and_birds : ∃ cb : ℕ, cb = 5)
                                 (h_no_dogs_and_birds : ∀ db : ℕ, ¬ (∃ db : ℕ, db = 1)) :
  ∃ n : ℕ, n = 48 :=
by
  have only_dogs := 40 - 10
  have only_cats := 30 - 10 - 5
  have only_birds := 8 - 5
  have result := only_dogs + only_cats + only_birds
  exact ⟨result, sorry⟩

end houses_with_only_one_pet_l10_10219


namespace leaves_blew_away_correct_l10_10959

-- Define the initial number of leaves Mikey had.
def initial_leaves : ℕ := 356

-- Define the number of leaves Mikey has left.
def leaves_left : ℕ := 112

-- Define the number of leaves that blew away.
def leaves_blew_away : ℕ := initial_leaves - leaves_left

-- Prove that the number of leaves that blew away is 244.
theorem leaves_blew_away_correct : leaves_blew_away = 244 :=
by sorry

end leaves_blew_away_correct_l10_10959


namespace monster_perimeter_correct_l10_10668

noncomputable def monster_perimeter (radius : ℝ) (central_angle_missing : ℝ) : ℝ :=
  let full_circle_circumference := 2 * radius * Real.pi
  let arc_length := (1 - central_angle_missing / 360) * full_circle_circumference
  arc_length + 2 * radius

theorem monster_perimeter_correct :
  monster_perimeter 2 90 = 3 * Real.pi + 4 :=
by
  -- The proof would go here
  sorry

end monster_perimeter_correct_l10_10668


namespace arithmetic_sequence_sum_false_statement_l10_10811

theorem arithmetic_sequence_sum_false_statement (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n.succ - a_n n = a_n 1 - a_n 0)
  (h_S : ∀ n, S n = (n + 1) * a_n 0 + (n * (n + 1) * (a_n 1 - a_n 0)) / 2)
  (h1 : S 6 < S 7) (h2 : S 7 = S 8) (h3 : S 8 > S 9) : ¬ (S 10 > S 6) :=
by
  sorry

end arithmetic_sequence_sum_false_statement_l10_10811


namespace cupboard_selling_percentage_l10_10438

theorem cupboard_selling_percentage (CP SP : ℝ) (h1 : CP = 6250) (h2 : SP + 1500 = 6250 * 1.12) :
  ((CP - SP) / CP) * 100 = 12 := by
sorry

end cupboard_selling_percentage_l10_10438


namespace find_parcera_triples_l10_10174

noncomputable def is_prime (n : ℕ) : Prop := sorry
noncomputable def parcera_triple (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧
  p ∣ q^2 - 4 ∧ q ∣ r^2 - 4 ∧ r ∣ p^2 - 4

theorem find_parcera_triples : 
  {t : ℕ × ℕ × ℕ | parcera_triple t.1 t.2.1 t.2.2} = 
  {(2, 2, 2), (5, 3, 7), (7, 5, 3), (3, 7, 5)} :=
sorry

end find_parcera_triples_l10_10174


namespace combination_of_students_l10_10781

-- Define the conditions
def num_boys := 4
def num_girls := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Calculate possible combinations
def two_boys_one_girl : ℕ :=
  combination num_boys 2 * combination num_girls 1

def one_boy_two_girls : ℕ :=
  combination num_boys 1 * combination num_girls 2

-- Total combinations
def total_combinations : ℕ :=
  two_boys_one_girl + one_boy_two_girls

-- Lean statement to be proven
theorem combination_of_students :
  total_combinations = 30 :=
by sorry

end combination_of_students_l10_10781


namespace find_common_ratio_l10_10403

theorem find_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2)
  (h5 : ∀ n : ℕ, a (n+1) = q * a n) : q = 4 := sorry

end find_common_ratio_l10_10403


namespace sequence_prime_bounded_l10_10462

theorem sequence_prime_bounded (c : ℕ) (h : c > 0) : 
  ∀ (p : ℕ → ℕ), (∀ k, Nat.Prime (p k)) → (p 0) = some_prime →
  (∀ k, ∃ q, Nat.Prime q ∧ q ∣ (p k + c) ∧ (∀ i < k, q ≠ p i)) → 
  (∃ N, ∀ m ≥ N, ∀ n ≥ N, p m = p n) :=
by
  sorry

end sequence_prime_bounded_l10_10462


namespace ratio_of_volume_to_surface_area_l10_10056

def volume_of_shape (num_cubes : ℕ) : ℕ :=
  -- Volume is simply the number of unit cubes
  num_cubes

def surface_area_of_shape : ℕ :=
  -- Surface area calculation given in the problem and solution
  12  -- edge cubes (4 cubes) with 3 exposed faces each
  + 16  -- side middle cubes (4 cubes) with 4 exposed faces each
  + 1  -- top face of the central cube in the bottom layer
  + 5  -- middle cube in the column with 5 exposed faces
  + 6  -- top cube in the column with all 6 faces exposed

theorem ratio_of_volume_to_surface_area
  (num_cubes : ℕ)
  (h1 : num_cubes = 9) :
  (volume_of_shape num_cubes : ℚ) / (surface_area_of_shape : ℚ) = 9 / 40 :=
by
  sorry

end ratio_of_volume_to_surface_area_l10_10056


namespace baseball_team_groups_l10_10140

theorem baseball_team_groups
  (new_players : ℕ) 
  (returning_players : ℕ)
  (players_per_group : ℕ)
  (total_players : ℕ := new_players + returning_players) :
  new_players = 48 → 
  returning_players = 6 → 
  players_per_group = 6 → 
  total_players / players_per_group = 9 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end baseball_team_groups_l10_10140


namespace max_unique_sums_l10_10610

-- Define the coin values in cents
def penny := 1
def nickel := 5
def quarter := 25
def half_dollar := 50

-- Define the set of all coins and their counts
structure Coins :=
  (pennies : ℕ := 3)
  (nickels : ℕ := 3)
  (quarters : ℕ := 1)
  (half_dollars : ℕ := 2)

-- Define the list of all possible pairs and their sums
def possible_sums : Finset ℕ :=
  { 2, 6, 10, 26, 30, 51, 55, 75, 100 }

-- Prove that the count of unique sums is 9
theorem max_unique_sums (c : Coins) : c.pennies = 3 → c.nickels = 3 → c.quarters = 1 → c.half_dollars = 2 →
  possible_sums.card = 9 := 
by
  intros
  sorry

end max_unique_sums_l10_10610


namespace exactly_two_pass_probability_l10_10336

theorem exactly_two_pass_probability (PA PB PC : ℚ) (hPA : PA = 2 / 3) (hPB : PB = 3 / 4) (hPC : PC = 2 / 5) :
  ((PA * PB * (1 - PC)) + (PA * (1 - PB) * PC) + ((1 - PA) * PB * PC) = 7 / 15) := by
  sorry

end exactly_two_pass_probability_l10_10336


namespace correct_train_process_l10_10152

-- Define each step involved in the train process
inductive Step
| buy_ticket
| wait_for_train
| check_ticket
| board_train
| repair_train

open Step

-- Define each condition as a list of steps
def process_a : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]
def process_b : List Step := [wait_for_train, buy_ticket, board_train, check_ticket]
def process_c : List Step := [buy_ticket, wait_for_train, board_train, check_ticket]
def process_d : List Step := [repair_train, buy_ticket, check_ticket, board_train]

-- Define the correct process
def correct_process : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]

-- The theorem to prove that process A is the correct representation
theorem correct_train_process : process_a = correct_process :=
by {
  sorry
}

end correct_train_process_l10_10152


namespace quadratic_root_exists_l10_10911

theorem quadratic_root_exists (a b c : ℝ) (ha : a ≠ 0)
  (h1 : a * (0.6 : ℝ)^2 + b * 0.6 + c = -0.04)
  (h2 : a * (0.7 : ℝ)^2 + b * 0.7 + c = 0.19) :
  ∃ x : ℝ, 0.6 < x ∧ x < 0.7 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_exists_l10_10911


namespace tina_savings_l10_10275

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end tina_savings_l10_10275


namespace lisa_eggs_total_l10_10552

def children_mon_tue := 4 * 2 * 2
def husband_mon_tue := 3 * 2 
def lisa_mon_tue := 2 * 2
def total_mon_tue := children_mon_tue + husband_mon_tue + lisa_mon_tue

def children_wed := 4 * 3
def husband_wed := 4
def lisa_wed := 3
def total_wed := children_wed + husband_wed + lisa_wed

def children_thu := 4 * 1
def husband_thu := 2
def lisa_thu := 1
def total_thu := children_thu + husband_thu + lisa_thu

def children_fri := 4 * 2
def husband_fri := 3
def lisa_fri := 2
def total_fri := children_fri + husband_fri + lisa_fri

def total_week := total_mon_tue + total_wed + total_thu + total_fri

def weeks_per_year := 52
def yearly_eggs := total_week * weeks_per_year

def children_holidays := 4 * 2 * 8
def husband_holidays := 2 * 8
def lisa_holidays := 2 * 8
def total_holidays := children_holidays + husband_holidays + lisa_holidays

def total_annual_eggs := yearly_eggs + total_holidays

theorem lisa_eggs_total : total_annual_eggs = 3476 := by
  sorry

end lisa_eggs_total_l10_10552


namespace max_curved_sides_l10_10260

theorem max_curved_sides (n : ℕ) (h : 2 ≤ n) : 
  ∃ m, m = 2 * n - 2 :=
sorry

end max_curved_sides_l10_10260


namespace solve_by_completing_square_l10_10562

noncomputable def d : ℤ := -5
noncomputable def e : ℤ := 10

theorem solve_by_completing_square :
  ∃ d e : ℤ, (x^2 - 10 * x + 15 = 0 ↔ (x + d)^2 = e) ∧ (d + e = 5) :=
by
  use -5, 10
  split
  -- First part: Show the equivalence of equations
  sorry
  -- Second part: Show d + e = 5
  refl

end solve_by_completing_square_l10_10562


namespace aquarium_length_l10_10124

theorem aquarium_length {L : ℝ} (W H : ℝ) (final_volume : ℝ)
  (hW : W = 6) (hH : H = 3) (h_final_volume : final_volume = 54)
  (h_volume_relation : final_volume = 3 * (1/4 * L * W * H)) :
  L = 4 := by
  -- Mathematically translate the problem given conditions and resulting in L = 4.
  sorry

end aquarium_length_l10_10124


namespace hot_dog_cost_l10_10340

theorem hot_dog_cost : 
  ∃ h d : ℝ, (3 * h + 4 * d = 10) ∧ (2 * h + 3 * d = 7) ∧ (d = 1) := 
by 
  sorry

end hot_dog_cost_l10_10340


namespace probability_non_obtuse_l10_10496

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l10_10496


namespace break_even_production_volume_l10_10760

theorem break_even_production_volume :
  ∃ Q : ℝ, 300 = 100 + 100000 / Q ∧ Q = 500 :=
by
  use 500
  sorry

end break_even_production_volume_l10_10760


namespace number_is_7612_l10_10271

-- Definitions of the conditions
def digits_correct_wrong_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10, 
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 ≠ (guess / 1000) % 10 ∧ 
      digits_placed 1 ≠ (guess / 100) % 10 ∧ 
      digits_placed 2 ≠ (guess / 10) % 10 ∧ 
      digits_placed 3 ≠ guess % 10)))

def digits_correct_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 = (guess / 1000) % 10 ∨ 
      digits_placed 1 = (guess / 100) % 10 ∨ 
      digits_placed 2 = (guess / 10) % 10 ∨ 
      digits_placed 3 = guess % 10)))

def digits_not_correct (guess : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → False)

-- The main theorem to prove
theorem number_is_7612 :
  digits_correct_wrong_positions 8765 2 ∧
  digits_correct_wrong_positions 1023 2 ∧
  digits_correct_positions 8642 2 ∧
  digits_not_correct 5430 →
  ∃ (num : Nat), 
    (num / 1000) % 10 = 7 ∧
    (num / 100) % 10 = 6 ∧
    (num / 10) % 10 = 1 ∧
    num % 10 = 2 ∧
    num = 7612 :=
sorry

end number_is_7612_l10_10271


namespace descending_order_of_numbers_l10_10057

theorem descending_order_of_numbers :
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  b > c ∧ c > a ∧ a > d :=
by
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  sorry

end descending_order_of_numbers_l10_10057


namespace least_positive_multiple_of_13_gt_418_l10_10583

theorem least_positive_multiple_of_13_gt_418 : ∃ (n : ℕ), n > 418 ∧ (13 ∣ n) ∧ n = 429 :=
by
  sorry

end least_positive_multiple_of_13_gt_418_l10_10583


namespace maximum_area_of_inscribed_rectangle_l10_10666

theorem maximum_area_of_inscribed_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (A : ℝ), A = (a * b) / 4 :=
by
  sorry -- placeholder for the proof

end maximum_area_of_inscribed_rectangle_l10_10666


namespace koala_food_consumed_l10_10836

theorem koala_food_consumed (x y : ℝ) (h1 : 0.40 * x = 12) (h2 : 0.20 * y = 2) : 
  x = 30 ∧ y = 10 := 
by
  sorry

end koala_food_consumed_l10_10836


namespace find_B_l10_10768

-- Define the polynomial function and its properties
def polynomial (z : ℤ) (A B : ℤ) : ℤ :=
  z^4 - 6 * z^3 + A * z^2 + B * z + 9

-- Prove that B = -9 under the given conditions
theorem find_B (A B : ℤ) (r1 r2 r3 r4 : ℤ)
  (h1 : polynomial r1 A B = 0)
  (h2 : polynomial r2 A B = 0)
  (h3 : polynomial r3 A B = 0)
  (h4 : polynomial r4 A B = 0)
  (h5 : r1 + r2 + r3 + r4 = 6)
  (h6 : r1 > 0)
  (h7 : r2 > 0)
  (h8 : r3 > 0)
  (h9 : r4 > 0) :
  B = -9 :=
by
  sorry

end find_B_l10_10768


namespace greatest_three_digit_number_l10_10878

theorem greatest_three_digit_number : ∃ n : ℕ, n < 1000 ∧ n >= 100 ∧ (n + 1) % 8 = 0 ∧ (n - 4) % 7 = 0 ∧ n = 967 :=
by
  sorry

end greatest_three_digit_number_l10_10878


namespace triangle_area_l10_10450

/-- Given a triangle with a perimeter of 20 cm and an inradius of 2.5 cm,
prove that its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 20) (h2 : inradius = 2.5) :
  area = 25 :=
by
  sorry

end triangle_area_l10_10450


namespace Jake_has_8_peaches_l10_10113

variable (Jake Steven Jill : ℕ)

theorem Jake_has_8_peaches
  (h_steven_peaches : Steven = 15)
  (h_steven_jill : Steven = Jill + 14)
  (h_jake_steven : Jake = Steven - 7) :
  Jake = 8 := by
  sorry

end Jake_has_8_peaches_l10_10113


namespace magic_square_y_minus_x_l10_10221

theorem magic_square_y_minus_x :
  ∀ (x y : ℝ), 
    (x - 2 = 2 * y + y) ∧ (x - 2 = -2 + y + 6) →
    y - x = -6 :=
by 
  intros x y h
  sorry

end magic_square_y_minus_x_l10_10221


namespace sin_alpha_l10_10641

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

theorem sin_alpha (α : ℝ) (h : f α = 1 / 3) : Real.sin α = -7 / 9 :=
by 
  sorry

end sin_alpha_l10_10641


namespace tunnel_length_l10_10171

theorem tunnel_length (x : ℕ) (y : ℕ) 
  (h1 : 300 + x = 60 * y) 
  (h2 : x - 300 = 30 * y) : 
  x = 900 := 
by
  sorry

end tunnel_length_l10_10171


namespace convert_to_base7_l10_10915

theorem convert_to_base7 : 3589 = 1 * 7^4 + 3 * 7^3 + 3 * 7^2 + 1 * 7^1 + 5 * 7^0 :=
by
  sorry

end convert_to_base7_l10_10915


namespace probability_third_smallest_is_four_l10_10695

/--
Seven distinct integers are picked at random from the set {1, 2, 3, ..., 12}.
The probability that the third smallest number is 4 is 7/33.
-/
theorem probability_third_smallest_is_four : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.to_finset in
  ∀ s : Finset ℕ, s ⊆ S ∧ s.card = 7 →
  let event := { s | s.nth_le 2 (by simp [s.card_eq_coe] ; norm_num) = 4 }.to_finset in
  (event.card : ℚ) / (S.choose 7).card = 7 / 33 :=
by
  intros S S_prop event
  sorry

end probability_third_smallest_is_four_l10_10695


namespace four_digit_integer_l10_10007

theorem four_digit_integer (a b c d : ℕ) 
  (h1 : a + b + c + d = 16) 
  (h2 : b + c = 10) 
  (h3 : a - d = 2)
  (h4 : (a - b + c - d) % 11 = 0) 
  : 1000 * a + 100 * b + 10 * c + d = 4642 := 
begin
  sorry
end

end four_digit_integer_l10_10007


namespace car_speed_first_hour_l10_10572

theorem car_speed_first_hour 
  (x : ℝ)  -- Speed of the car in the first hour.
  (s2 : ℝ)  -- Speed of the car in the second hour is fixed at 40 km/h.
  (avg_speed : ℝ)  -- Average speed over two hours is 65 km/h.
  (h1 : s2 = 40)  -- speed in the second hour is 40 km/h.
  (h2 : avg_speed = 65)  -- average speed is 65 km/h
  (h3 : avg_speed = (x + s2) / 2)  -- definition of average speed
  : x = 90 := 
  sorry

end car_speed_first_hour_l10_10572


namespace johns_age_l10_10896

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end johns_age_l10_10896


namespace color_triangle_congruence_l10_10754

theorem color_triangle_congruence :
  ∀ (points : Fin 432 → Prop) (colors : Fin 432 → Fin 4),
    (∀ i, points i) ∧
    (∀ c : Fin 4, ∃ (Ps : Finset (Fin 432)), Ps.card = 108 ∧ ∀ p ∈ Ps, colors p = c) →
    ∃ (chosen : Fin 4 → Finset (Fin 432)), 
      (∀ c, chosen c).card = 3 ∧
      ∀ c, (∀ (p1 p2 p3 ∈ chosen c), 
            dist (Fin (432 : Nat)) p1 p2 = dist (Fin (432 : Nat)) p3 (p3 + (p2 - p1)) :=
sorry

end color_triangle_congruence_l10_10754


namespace visitors_yesterday_l10_10173

-- Definitions based on the given conditions
def visitors_today : ℕ := 583
def visitors_total : ℕ := 829

-- Theorem statement to prove the number of visitors the day before Rachel visited
theorem visitors_yesterday : ∃ v_yesterday: ℕ, v_yesterday = visitors_total - visitors_today ∧ v_yesterday = 246 :=
by
  sorry

end visitors_yesterday_l10_10173


namespace gcd_of_8247_13619_29826_l10_10189

theorem gcd_of_8247_13619_29826 : Nat.gcd (Nat.gcd 8247 13619) 29826 = 3 := 
sorry

end gcd_of_8247_13619_29826_l10_10189


namespace arithmetic_mean_solve_x_l10_10345

theorem arithmetic_mean_solve_x (x : ℚ) :
  (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30 → x = 99 / 7 :=
by 
sorry

end arithmetic_mean_solve_x_l10_10345


namespace find_circle_center_l10_10064

theorem find_circle_center :
  ∀ x y : ℝ,
  (x^2 + 4*x + y^2 - 6*y = 20) →
  (x + 2, y - 3) = (-2, 3) := by
  sorry

end find_circle_center_l10_10064


namespace sum_f_values_l10_10366

noncomputable def f : ℝ → ℝ := sorry

axiom odd_property (x : ℝ) : f (-x) = -f (x)
axiom periodicity (x : ℝ) : f (x) = f (x + 4)
axiom f1 : f 1 = -1

theorem sum_f_values : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end sum_f_values_l10_10366


namespace simplify_and_evaluate_expression_l10_10416

-- Define the condition
def condition (x y : ℝ) := (x - 2) ^ 2 + |y + 1| = 0

-- Define the expression
def expression (x y : ℝ) := 3 * x ^ 2 * y - (2 * x ^ 2 * y - 3 * (2 * x * y - x ^ 2 * y) + 5 * x * y)

-- State the theorem
theorem simplify_and_evaluate_expression (x y : ℝ) (h : condition x y) : expression x y = 6 :=
by
  sorry

end simplify_and_evaluate_expression_l10_10416


namespace parabola_symmetric_points_l10_10824

-- Define the parabola and the symmetry condition
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

def symmetric_points (P Q : ℝ × ℝ) : Prop :=
  P.1 + P.2 = 0 ∧ Q.1 + Q.2 = 0 ∧ P.1 = -Q.1 ∧ P.2 = -Q.2

-- Problem definition: Prove that if there exist symmetric points on the parabola, then a > 3/4
theorem parabola_symmetric_points (a : ℝ) :
  (∃ P Q : ℝ × ℝ, symmetric_points P Q ∧ parabola a P.1 = P.2 ∧ parabola a Q.1 = Q.2) → a > 3 / 4 :=
by
  sorry

end parabola_symmetric_points_l10_10824


namespace geometric_series_sum_l10_10229

theorem geometric_series_sum :
  let a := 6
  let r := - (2 / 5)
  s = ∑' n, (a * r ^ n) ↔ s = 30 / 7 :=
sorry

end geometric_series_sum_l10_10229


namespace platform_length_l10_10897

noncomputable def train_length : ℕ := 1200
noncomputable def time_to_cross_tree : ℕ := 120
noncomputable def time_to_pass_platform : ℕ := 230

theorem platform_length
  (v : ℚ)
  (h1 : v = train_length / time_to_cross_tree)
  (total_distance : ℚ)
  (h2 : total_distance = v * time_to_pass_platform)
  (platform_length : ℚ)
  (h3 : total_distance = train_length + platform_length) :
  platform_length = 1100 := by 
  sorry

end platform_length_l10_10897


namespace hyperbola_eccentricity_l10_10652

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
    (h_asymptote : ∀ x, y = x * 3/4 → y^2 / b^2 - x^2 / a^2 = 1) :
    ∃ e, e = 5/4 :=
by
  sorry

end hyperbola_eccentricity_l10_10652


namespace obtuse_triangle_probability_l10_10501

noncomputable def uniform_circle_distribution (radius : ℝ) : ProbabilitySpace (ℝ × ℝ) := sorry

def no_obtuse_triangle (points : list (ℝ × ℝ)) (center : (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k < 4 →
  ∀ (p₁ p₂ : (ℝ × ℝ)),
  (p₁ = points.nth i ∧ p₂ = points.nth j ∧ points.nth k ∉ set_obtuse_triangles p₁ p₂ center)

theorem obtuse_triangle_probability :
  let points := [⟨cos θ₀, sin θ₀⟩, ⟨cos θ₁, sin θ₁⟩, ⟨cos θ₂, sin θ₂⟩, ⟨cos θ₃, sin θ₃⟩]
  ∀ center : (ℝ × ℝ), 
  ∀ (h : set.center.circle center radius), 
  (no_obtuse_triangle points center) →
  Pr[(points)] = 3 / 32 :=
sorry

end obtuse_triangle_probability_l10_10501


namespace remainder_of_sum_l10_10990

theorem remainder_of_sum (a b c : ℕ) (h₁ : a % 15 = 11) (h₂ : b % 15 = 12) (h₃ : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
by 
  sorry

end remainder_of_sum_l10_10990


namespace percent_motorists_receive_tickets_l10_10243

theorem percent_motorists_receive_tickets (n : ℕ) (h1 : (25 : ℕ) % 100 = 25) (h2 : (20 : ℕ) % 100 = 20) :
  (75 * n / 100) = (20 * n / 100) :=
by
  sorry

end percent_motorists_receive_tickets_l10_10243


namespace arithmetic_seq_common_difference_l10_10506

theorem arithmetic_seq_common_difference (a1 d : ℝ) (h1 : a1 + 2 * d = 10) (h2 : 4 * a1 + 6 * d = 36) : d = 2 :=
by
  sorry

end arithmetic_seq_common_difference_l10_10506
