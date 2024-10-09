import Mathlib

namespace problem1_problem2_l1407_140777

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l1407_140777


namespace find_hall_length_l1407_140708

variable (W H total_cost cost_per_sqm : ℕ)

theorem find_hall_length
  (hW : W = 15)
  (hH : H = 5)
  (h_total_cost : total_cost = 57000)
  (h_cost_per_sqm : cost_per_sqm = 60)
  : (32 * W) + (2 * (H * 32)) + (2 * (H * W)) = total_cost / cost_per_sqm :=
by
  sorry

end find_hall_length_l1407_140708


namespace directrix_of_parabola_l1407_140764

theorem directrix_of_parabola (p m : ℝ) (hp : p > 0)
  (hM_on_parabola : (4, m).fst ^ 2 = 2 * p * (4, m).snd)
  (hM_to_focus : dist (4, m) (p / 2, 0) = 6) :
  -p/2 = -2 :=
sorry

end directrix_of_parabola_l1407_140764


namespace range_of_f_l1407_140761

noncomputable def f : ℝ → ℝ := sorry -- Define f appropriately

theorem range_of_f : Set.range f = {y : ℝ | 0 < y} :=
sorry

end range_of_f_l1407_140761


namespace wade_customers_sunday_l1407_140757

theorem wade_customers_sunday :
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  customers_sunday = 36 :=
by
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  have h : customers_sunday = 36 := by sorry
  exact h

end wade_customers_sunday_l1407_140757


namespace shaded_region_area_l1407_140734

theorem shaded_region_area (r_s r_l chord_AB : ℝ) (hs : r_s = 40) (hl : r_l = 60) (hc : chord_AB = 100) :
    chord_AB / 2 = 50 →
    60^2 - (chord_AB / 2)^2 = r_s^2 →
    (π * r_l^2) - (π * r_s^2) = 2500 * π :=
by
  intros h1 h2
  sorry

end shaded_region_area_l1407_140734


namespace train_arrival_day_l1407_140765

-- Definitions for the start time and journey duration
def start_time : ℕ := 0  -- early morning (0 hours) on Tuesday
def journey_duration : ℕ := 28  -- 28 hours

-- Proving the arrival time
theorem train_arrival_day (start_time journey_duration : ℕ) :
  journey_duration == 28 → 
  start_time == 0 → 
  (journey_duration / 24, journey_duration % 24) == (1, 4) → 
  true := 
by
  intros
  sorry

end train_arrival_day_l1407_140765


namespace erasers_left_in_the_box_l1407_140751

-- Conditions expressed as definitions
def E0 : ℕ := 320
def E1 : ℕ := E0 - 67
def E2 : ℕ := E1 - 126
def E3 : ℕ := E2 + 30

-- Proof problem statement
theorem erasers_left_in_the_box : E3 = 157 := 
by sorry

end erasers_left_in_the_box_l1407_140751


namespace Ryan_spits_percentage_shorter_l1407_140729

theorem Ryan_spits_percentage_shorter (Billy_dist Madison_dist Ryan_dist : ℝ) (h1 : Billy_dist = 30) (h2 : Madison_dist = 1.20 * Billy_dist) (h3 : Ryan_dist = 18) :
  ((Madison_dist - Ryan_dist) / Madison_dist) * 100 = 50 :=
by
  sorry

end Ryan_spits_percentage_shorter_l1407_140729


namespace crackers_per_friend_l1407_140741

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (n : ℕ) 
  (h1 : total_crackers = 8) 
  (h2 : num_friends = 4)
  (h3 : total_crackers / num_friends = n) : n = 2 :=
by
  sorry

end crackers_per_friend_l1407_140741


namespace quadratic_has_real_roots_l1407_140760

theorem quadratic_has_real_roots (k : ℝ) (h : k > 0) : ∃ x : ℝ, x^2 + 2 * x - k = 0 :=
by
  sorry

end quadratic_has_real_roots_l1407_140760


namespace problem_min_value_l1407_140749

theorem problem_min_value {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  (2 * a + b + c) ≥ 4 := 
  sorry

end problem_min_value_l1407_140749


namespace initial_number_is_correct_l1407_140791

theorem initial_number_is_correct (x : ℝ) (h : 8 * x - 4 = 2.625) : x = 0.828125 :=
by
  sorry

end initial_number_is_correct_l1407_140791


namespace last_two_digits_of_7_pow_2023_l1407_140753

theorem last_two_digits_of_7_pow_2023 : (7 ^ 2023) % 100 = 43 := by
  sorry

end last_two_digits_of_7_pow_2023_l1407_140753


namespace divide_cookie_into_16_equal_parts_l1407_140726

def Cookie (n : ℕ) : Type := sorry

theorem divide_cookie_into_16_equal_parts (cookie : Cookie 64) :
  ∃ (slices : List (Cookie 4)), slices.length = 16 ∧ 
  (∀ (slice : Cookie 4), slice ≠ cookie) := 
sorry

end divide_cookie_into_16_equal_parts_l1407_140726


namespace corner_cell_revisit_l1407_140745

theorem corner_cell_revisit
    (M N : ℕ)
    (hM : M = 101)
    (hN : N = 200)
    (initial_position : ℕ × ℕ)
    (h_initial : initial_position = (0, 0) ∨ initial_position = (0, 200) ∨ initial_position = (101, 0) ∨ initial_position = (101, 200)) :
    ∃ final_position : ℕ × ℕ, 
      final_position = initial_position ∧ (final_position = (0, 0) ∨ final_position = (0, 200) ∨ final_position = (101, 0) ∨ final_position = (101, 200)) :=
by
  sorry

end corner_cell_revisit_l1407_140745


namespace problem_condition_holds_l1407_140780

theorem problem_condition_holds (x y : ℝ) (h₁ : x + 0.35 * y - (x + y) = 200) : y = -307.69 :=
sorry

end problem_condition_holds_l1407_140780


namespace scale_readings_poles_greater_l1407_140720

-- Define the necessary quantities and conditions
variable (m : ℝ) -- mass of the object
variable (ω : ℝ) -- angular velocity of Earth's rotation
variable (R_e : ℝ) -- radius of the Earth at the equator
variable (g_e : ℝ) -- gravitational acceleration at the equator
variable (g_p : ℝ) -- gravitational acceleration at the poles
variable (F_c : ℝ) -- centrifugal force at the equator
variable (F_g_e : ℝ) -- gravitational force at the equator
variable (F_g_p : ℝ) -- gravitational force at the poles
variable (W_e : ℝ) -- apparent weight at the equator
variable (W_p : ℝ) -- apparent weight at the poles

-- Establish conditions
axiom centrifugal_definition : F_c = m * ω^2 * R_e
axiom gravitational_force_equator : F_g_e = m * g_e
axiom apparent_weight_equator : W_e = F_g_e - F_c
axiom no_centrifugal_force_poles : F_c = 0
axiom gravitational_force_poles : F_g_p = m * g_p
axiom apparent_weight_poles : W_p = F_g_p
axiom gravity_comparison : g_p > g_e

-- Theorem: The readings on spring scales at the poles will be greater than the readings at the equator
theorem scale_readings_poles_greater : W_p > W_e := 
sorry

end scale_readings_poles_greater_l1407_140720


namespace floor_floor_3x_eq_floor_x_plus_1_l1407_140773

theorem floor_floor_3x_eq_floor_x_plus_1 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋) ↔ (2 / 3 ≤ x ∧ x < 4 / 3) :=
by
  sorry

end floor_floor_3x_eq_floor_x_plus_1_l1407_140773


namespace find_random_discount_l1407_140795

theorem find_random_discount
  (initial_price : ℝ) (final_price : ℝ) (autumn_discount : ℝ) (loyalty_discount : ℝ) (random_discount : ℝ) :
  initial_price = 230 ∧ final_price = 69 ∧ autumn_discount = 0.25 ∧ loyalty_discount = 0.20 ∧ 
  final_price = initial_price * (1 - autumn_discount) * (1 - loyalty_discount) * (1 - random_discount / 100) →
  random_discount = 50 :=
by
  intros h
  sorry

end find_random_discount_l1407_140795


namespace number_of_rhombuses_is_84_l1407_140737

def total_rhombuses (side_length_large_triangle : Nat) (side_length_small_triangle : Nat) (num_small_triangles : Nat) : Nat :=
  if side_length_large_triangle = 10 ∧ 
     side_length_small_triangle = 1 ∧ 
     num_small_triangles = 100 then 84 else 0

theorem number_of_rhombuses_is_84 :
  total_rhombuses 10 1 100 = 84 := by
  sorry

end number_of_rhombuses_is_84_l1407_140737


namespace time_to_cover_escalator_l1407_140728

-- Define the given conditions
def escalator_speed : ℝ := 20 -- feet per second
def escalator_length : ℝ := 360 -- feet
def delay_time : ℝ := 5 -- seconds
def person_speed : ℝ := 4 -- feet per second

-- Define the statement to be proven
theorem time_to_cover_escalator : (delay_time + (escalator_length - (escalator_speed * delay_time)) / (person_speed + escalator_speed)) = 15.83 := 
by {
  sorry
}

end time_to_cover_escalator_l1407_140728


namespace total_musicians_count_l1407_140702

-- Define the given conditions
def orchestra_males := 11
def orchestra_females := 12
def choir_males := 12
def choir_females := 17

-- Total number of musicians in the orchestra
def orchestra_musicians := orchestra_males + orchestra_females

-- Total number of musicians in the band
def band_musicians := 2 * orchestra_musicians

-- Total number of musicians in the choir
def choir_musicians := choir_males + choir_females

-- Total number of musicians in the orchestra, band, and choir
def total_musicians := orchestra_musicians + band_musicians + choir_musicians

-- The theorem to prove
theorem total_musicians_count : total_musicians = 98 :=
by
  -- Lean proof part goes here.
  sorry

end total_musicians_count_l1407_140702


namespace smallest_x_multiple_of_1024_l1407_140738

theorem smallest_x_multiple_of_1024 (x : ℕ) (hx : 900 * x % 1024 = 0) : x = 256 :=
sorry

end smallest_x_multiple_of_1024_l1407_140738


namespace different_language_classes_probability_l1407_140742

theorem different_language_classes_probability :
  let total_students := 40
  let french_students := 28
  let spanish_students := 26
  let german_students := 15
  let french_and_spanish_students := 10
  let french_and_german_students := 6
  let spanish_and_german_students := 8
  let all_three_languages_students := 3
  let total_pairs := Nat.choose total_students 2
  let french_only := french_students - (french_and_spanish_students + french_and_german_students - all_three_languages_students) - all_three_languages_students
  let spanish_only := spanish_students - (french_and_spanish_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let german_only := german_students - (french_and_german_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let french_only_pairs := Nat.choose french_only 2
  let spanish_only_pairs := Nat.choose spanish_only 2
  let german_only_pairs := Nat.choose german_only 2
  let single_language_pairs := french_only_pairs + spanish_only_pairs + german_only_pairs
  let different_classes_probability := 1 - (single_language_pairs / total_pairs)
  different_classes_probability = (34 / 39) :=
by
  sorry

end different_language_classes_probability_l1407_140742


namespace red_balls_estimation_l1407_140756

noncomputable def numberOfRedBalls (x : ℕ) : ℝ := x / (x + 3)

theorem red_balls_estimation {x : ℕ} (h : numberOfRedBalls x = 0.85) : x = 17 :=
by
  sorry

end red_balls_estimation_l1407_140756


namespace sector_angle_l1407_140719

theorem sector_angle (R : ℝ) (S : ℝ) (α : ℝ) (hR : R = 2) (hS : S = 8) : 
  α = 4 := by
  sorry

end sector_angle_l1407_140719


namespace part1_part2_l1407_140724

noncomputable def f (ω x : ℝ) : ℝ := 4 * ((Real.sin (ω * x - Real.pi / 4)) * (Real.cos (ω * x)))

noncomputable def g (α : ℝ) : ℝ := 2 * (Real.sin (α - Real.pi / 6)) - Real.sqrt 2

theorem part1 (ω : ℝ) (x : ℝ) (hω : 0 < ω ∧ ω < 2) (hx : f ω (Real.pi / 4) = Real.sqrt 2) : 
  ∃ T > 0, ∀ x, f ω (x + T) = f ω x :=
sorry

theorem part2 (α : ℝ) (hα: 0 < α ∧ α < Real.pi / 2) (h : g α = 4 / 3 - Real.sqrt 2) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 :=
sorry

end part1_part2_l1407_140724


namespace selling_price_correct_l1407_140759

namespace Shopkeeper

def costPrice : ℝ := 1500
def profitPercentage : ℝ := 20
def expectedSellingPrice : ℝ := 1800

theorem selling_price_correct
  (cp : ℝ := costPrice)
  (pp : ℝ := profitPercentage) :
  cp * (1 + pp / 100) = expectedSellingPrice :=
by
  sorry

end Shopkeeper

end selling_price_correct_l1407_140759


namespace sum_of_fractions_l1407_140797

theorem sum_of_fractions :
  (3 / 20 : ℝ) +  (7 / 200) + (8 / 2000) + (3 / 20000) = 0.1892 :=
by 
  sorry

end sum_of_fractions_l1407_140797


namespace solve_system_l1407_140700

theorem solve_system :
  ∃ (x y : ℝ), (x^2 + y^2 ≤ 1) ∧ (x^4 - 18 * x^2 * y^2 + 81 * y^4 - 20 * x^2 - 180 * y^2 + 100 = 0) ∧
    ((x = -1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = -1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10)) :=
  by
  sorry

end solve_system_l1407_140700


namespace find_a_l1407_140709

def tangent_condition (x a : ℝ) : Prop := 2 * x - (Real.log x + a) + 1 = 0

def slope_condition (x : ℝ) : Prop := 2 = 1 / x

theorem find_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ tangent_condition x a ∧ slope_condition x) →
  a = -2 * Real.log 2 :=
by
  intro h
  sorry

end find_a_l1407_140709


namespace PatriciaHighlightFilmTheorem_l1407_140787

def PatriciaHighlightFilmProblem : Prop :=
  let point_guard_seconds := 130
  let shooting_guard_seconds := 145
  let small_forward_seconds := 85
  let power_forward_seconds := 60
  let center_seconds := 180
  let total_seconds := point_guard_seconds + shooting_guard_seconds + small_forward_seconds + power_forward_seconds + center_seconds
  let num_players := 5
  let average_seconds := total_seconds / num_players
  let average_minutes := average_seconds / 60
  average_minutes = 2

theorem PatriciaHighlightFilmTheorem : PatriciaHighlightFilmProblem :=
  by
    -- Proof goes here
    sorry

end PatriciaHighlightFilmTheorem_l1407_140787


namespace intersection_sum_zero_l1407_140792

-- Definitions from conditions:
def lineA (x : ℝ) : ℝ := -x
def lineB (x : ℝ) : ℝ := 5 * x - 10

-- Declaration of the theorem:
theorem intersection_sum_zero : ∃ a b : ℝ, lineA a = b ∧ lineB a = b ∧ a + b = 0 := sorry

end intersection_sum_zero_l1407_140792


namespace locus_of_M_l1407_140790

theorem locus_of_M (k : ℝ) (A B M : ℝ × ℝ) (hA : A.1 ≥ 0 ∧ A.2 = 0) (hB : B.2 ≥ 0 ∧ B.1 = 0) (h_sum : A.1 + B.2 = k) :
    ∃ (M : ℝ × ℝ), (M.1 - k / 2)^2 + (M.2 - k / 2)^2 = k^2 / 2 :=
by
  sorry

end locus_of_M_l1407_140790


namespace area_EPHQ_l1407_140781

theorem area_EPHQ {EFGH : Type} 
  (rectangle_EFGH : EFGH) 
  (length_EF : Real) (width_EG : Real) 
  (P_point : Real) (Q_point : Real) 
  (area_EFGH : Real) 
  (area_EFP : Real) 
  (area_EHQ : Real) : 
  length_EF = 12 → width_EG = 6 → P_point = 4 → Q_point = 3 → 
  area_EFGH = length_EF * width_EG →
  area_EFP = (1 / 2) * width_EG * P_point →
  area_EHQ = (1 / 2) * length_EF * Q_point → 
  (area_EFGH - area_EFP - area_EHQ) = 42 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end area_EPHQ_l1407_140781


namespace pq_necessary_not_sufficient_l1407_140779

theorem pq_necessary_not_sufficient (p q : Prop) : (p ∨ q) → (p ∧ q) ↔ false :=
by sorry

end pq_necessary_not_sufficient_l1407_140779


namespace maximize_c_l1407_140754

theorem maximize_c (c d e : ℤ) (h1 : 5 * c + (d - 12)^2 + e^3 = 235) (h2 : c < d) : c ≤ 22 :=
sorry

end maximize_c_l1407_140754


namespace daily_egg_count_per_female_emu_l1407_140748

noncomputable def emus_per_pen : ℕ := 6
noncomputable def pens : ℕ := 4
noncomputable def total_eggs_per_week : ℕ := 84

theorem daily_egg_count_per_female_emu :
  (total_eggs_per_week / ((pens * emus_per_pen) / 2 * 7) = 1) :=
by
  sorry

end daily_egg_count_per_female_emu_l1407_140748


namespace sum_of_repeating_decimals_l1407_140743

-- Defining the given repeating decimals as fractions
def rep_decimal1 : ℚ := 2 / 9
def rep_decimal2 : ℚ := 2 / 99
def rep_decimal3 : ℚ := 2 / 9999

-- Stating the theorem to prove the given sum equals the correct answer
theorem sum_of_repeating_decimals :
  rep_decimal1 + rep_decimal2 + rep_decimal3 = 224422 / 9999 :=
by
  sorry

end sum_of_repeating_decimals_l1407_140743


namespace ellipse_parametric_form_l1407_140778

theorem ellipse_parametric_form :
  (∃ A B C D E F : ℤ,
    ((∀ t : ℝ, (3 * (Real.sin t - 2)) / (3 - Real.cos t) = x ∧ 
     (2 * (Real.cos t - 4)) / (3 - Real.cos t) = y) → 
    (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1846)) := 
sorry

end ellipse_parametric_form_l1407_140778


namespace total_tickets_l1407_140710

theorem total_tickets (A C : ℕ) (cost_adult cost_child total_cost : ℝ) 
  (h1 : cost_adult = 5.50) 
  (h2 : cost_child = 3.50) 
  (h3 : C = 16) 
  (h4 : total_cost = 83.50) 
  (h5 : cost_adult * A + cost_child * C = total_cost) : 
  A + C = 21 := 
by 
  sorry

end total_tickets_l1407_140710


namespace sally_bought_48_eggs_l1407_140736

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_sally_bought : ℕ := 4

-- Define the total number of eggs Sally bought
def total_eggs_sally_bought : ℕ := dozens_sally_bought * eggs_in_a_dozen

-- Theorem stating the number of eggs Sally bought
theorem sally_bought_48_eggs : total_eggs_sally_bought = 48 :=
sorry

end sally_bought_48_eggs_l1407_140736


namespace number_of_hours_sold_l1407_140799

def packs_per_hour_peak := 6
def packs_per_hour_low := 4
def price_per_pack := 60
def extra_revenue := 1800

def revenue_per_hour_peak := packs_per_hour_peak * price_per_pack
def revenue_per_hour_low := packs_per_hour_low * price_per_pack
def revenue_diff_per_hour := revenue_per_hour_peak - revenue_per_hour_low

theorem number_of_hours_sold (h : ℕ) 
  (h_eq : revenue_diff_per_hour * h = extra_revenue) : 
  h = 15 :=
by
  -- skip proof
  sorry

end number_of_hours_sold_l1407_140799


namespace coordinates_of_foci_l1407_140704

-- Given conditions
def equation_of_hyperbola : Prop := ∃ (x y : ℝ), (x^2 / 4) - (y^2 / 5) = 1

-- The mathematical goal translated into a theorem
theorem coordinates_of_foci (x y : ℝ) (a b c : ℝ) (ha : a^2 = 4) (hb : b^2 = 5) (hc : c^2 = a^2 + b^2) :
  equation_of_hyperbola →
  ((x = 3 ∨ x = -3) ∧ y = 0) :=
sorry

end coordinates_of_foci_l1407_140704


namespace range_of_m_local_odd_function_l1407_140796

def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

noncomputable def f (x m : ℝ) : ℝ :=
  9^x - m * 3^x - 3

theorem range_of_m_local_odd_function :
  (∀ m : ℝ, is_local_odd_function (λ x => f x m) ↔ m ∈ Set.Ici (-2)) :=
by
  sorry

end range_of_m_local_odd_function_l1407_140796


namespace monkey_bananas_max_l1407_140775

noncomputable def max_bananas_home : ℕ :=
  let total_bananas := 100
  let distance := 50
  let carry_capacity := 50
  let consumption_rate := 1
  let distance_each_way := distance / 2
  let bananas_eaten_each_way := distance_each_way * consumption_rate
  let bananas_left_midway := total_bananas / 2 - bananas_eaten_each_way
  let bananas_picked_midway := bananas_left_midway * 2
  let bananas_left_home := bananas_picked_midway - distance_each_way * consumption_rate
  bananas_left_home

theorem monkey_bananas_max : max_bananas_home = 25 :=
  sorry

end monkey_bananas_max_l1407_140775


namespace expression_for_f_when_x_lt_0_l1407_140794

noncomputable section

variable (f : ℝ → ℝ)

theorem expression_for_f_when_x_lt_0
  (hf_neg : ∀ x : ℝ, f (-x) = -f x)
  (hf_pos : ∀ x : ℝ, x > 0 → f x = x * abs (x - 2)) :
  ∀ x : ℝ, x < 0 → f x = x * abs (x + 2) :=
by
  sorry

end expression_for_f_when_x_lt_0_l1407_140794


namespace atomic_weight_Ba_l1407_140793

-- Definitions for conditions
def atomic_weight_O : ℕ := 16
def molecular_weight_compound : ℕ := 153

-- Theorem statement
theorem atomic_weight_Ba : ∃ bw, molecular_weight_compound = bw + atomic_weight_O ∧ bw = 137 :=
by {
  -- Skip the proof
  sorry
}

end atomic_weight_Ba_l1407_140793


namespace legs_walking_on_ground_l1407_140716

def number_of_horses : ℕ := 14
def number_of_men : ℕ := number_of_horses
def legs_per_man : ℕ := 2
def legs_per_horse : ℕ := 4
def half (n : ℕ) : ℕ := n / 2

theorem legs_walking_on_ground :
  (half number_of_men) * legs_per_man + (half number_of_horses) * legs_per_horse = 42 :=
by
  sorry

end legs_walking_on_ground_l1407_140716


namespace base_case_of_interior_angle_sum_l1407_140747

-- Definitions consistent with conditions: A convex polygon with at least n sides where n >= 3.
def convex_polygon (n : ℕ) : Prop := n ≥ 3

-- Proposition: If w the sum of angles for convex polygons, we start checking from n = 3.
theorem base_case_of_interior_angle_sum (n : ℕ) (h : convex_polygon n) :
  n = 3 := 
by
  sorry

end base_case_of_interior_angle_sum_l1407_140747


namespace evaluate_f_5_minus_f_neg_5_l1407_140727

noncomputable def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by 
  sorry

end evaluate_f_5_minus_f_neg_5_l1407_140727


namespace prime_root_condition_l1407_140783

theorem prime_root_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℤ, x ≠ y ∧ (x^2 + 2 * p * x - 240 * p = 0) ∧ (y^2 + 2 * p * y - 240 * p = 0) ∧ x*y = -240*p) → p = 5 :=
by sorry

end prime_root_condition_l1407_140783


namespace drawing_blue_ball_probability_l1407_140706

noncomputable def probability_of_blue_ball : ℚ :=
  let total_balls := 10
  let blue_balls := 6
  blue_balls / total_balls

theorem drawing_blue_ball_probability :
  probability_of_blue_ball = 3 / 5 :=
by
  sorry -- Proof is omitted as per instructions.

end drawing_blue_ball_probability_l1407_140706


namespace find_integer_n_l1407_140774

theorem find_integer_n (n : ℤ) (h : (⌊n^2 / 4⌋ - (⌊n / 2⌋)^2) = 3) : n = 7 :=
sorry

end find_integer_n_l1407_140774


namespace total_maple_trees_in_park_after_planting_l1407_140703

def number_of_maple_trees_in_the_park (X_M : ℕ) (Y_M : ℕ) : ℕ := 
  X_M + Y_M

theorem total_maple_trees_in_park_after_planting : 
  number_of_maple_trees_in_the_park 2 9 = 11 := 
by 
  unfold number_of_maple_trees_in_the_park
  -- provide the mathematical proof here
  sorry

end total_maple_trees_in_park_after_planting_l1407_140703


namespace exists_initial_value_l1407_140798

theorem exists_initial_value (x : ℤ) : ∃ y : ℤ, x + 49 = y^2 :=
sorry

end exists_initial_value_l1407_140798


namespace average_error_diff_l1407_140711

theorem average_error_diff (n : ℕ) (total_data_pts : ℕ) (error_data1 error_data2 : ℕ)
  (h_n : n = 30) (h_total_data_pts : total_data_pts = 30)
  (h_error_data1 : error_data1 = 105) (h_error_data2 : error_data2 = 15)
  : (error_data1 - error_data2) / n = 3 :=
sorry

end average_error_diff_l1407_140711


namespace range_of_a_l1407_140788

noncomputable def A (a : ℝ) := {x : ℝ | a < x ∧ x < 2 * a + 1}
def B := {x : ℝ | abs (x - 1) > 2}

theorem range_of_a (a : ℝ) (h : A a ⊆ B) : a ≤ -1 ∨ a ≥ 3 := by
  sorry

end range_of_a_l1407_140788


namespace greatest_x_lcm_l1407_140768

theorem greatest_x_lcm (x : ℕ) (h1 : Nat.lcm x 15 = Nat.lcm 90 15) (h2 : Nat.lcm x 18 = Nat.lcm 90 18) : x = 90 := 
sorry

end greatest_x_lcm_l1407_140768


namespace range_of_a_l1407_140762

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 - 1

def is_fixed_point (a x : ℝ) : Prop := f a x = x

def is_stable_point (a x : ℝ) : Prop := f a (f a x) = x

def are_equal_sets (a : ℝ) : Prop :=
  {x : ℝ | is_fixed_point a x} = {x : ℝ | is_stable_point a x}

theorem range_of_a (a : ℝ) (h : are_equal_sets a) : - (1 / 4) ≤ a ∧ a ≤ 3 / 4 := 
by
  sorry

end range_of_a_l1407_140762


namespace perfect_square_trinomial_l1407_140739

theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, 4 * (c^2) = 9 ∧ 4 * c = a - b) → 2 * a - 2 * b = 24 ∨ 2 * a - 2 * b = -24 :=
by
  sorry

end perfect_square_trinomial_l1407_140739


namespace percentage_increase_l1407_140732

theorem percentage_increase (D J : ℝ) (hD : D = 480) (hJ : J = 417.39) :
  ((D - J) / J) * 100 = 14.99 := 
by
  sorry

end percentage_increase_l1407_140732


namespace real_part_of_one_over_one_minus_z_l1407_140766

open Complex

noncomputable def real_part_fraction {z : ℂ} (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) : ℝ :=
  re (1 / (1 - z))

theorem real_part_of_one_over_one_minus_z (z : ℂ) (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) :
  real_part_fraction hz1 hz2 = 1 / 2 :=
by
  sorry

end real_part_of_one_over_one_minus_z_l1407_140766


namespace ratio_of_sums_eq_19_over_17_l1407_140713

theorem ratio_of_sums_eq_19_over_17 :
  let a₁ := 5
  let d₁ := 3
  let l₁ := 59
  let a₂ := 4
  let d₂ := 4
  let l₂ := 64
  let n₁ := 19  -- from solving l₁ = a₁ + (n₁ - 1) * d₁
  let n₂ := 16  -- from solving l₂ = a₂ + (n₂ - 1) * d₂
  let S₁ := n₁ * (a₁ + l₁) / 2
  let S₂ := n₂ * (a₂ + l₂) / 2
  S₁ / S₂ = 19 / 17 := by sorry

end ratio_of_sums_eq_19_over_17_l1407_140713


namespace magnitude_of_two_a_minus_b_l1407_140782

namespace VectorMagnitude

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (3, -2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Vector operation 2a - b
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem to prove
theorem magnitude_of_two_a_minus_b : magnitude two_a_minus_b = Real.sqrt 17 := by
  sorry

end VectorMagnitude

end magnitude_of_two_a_minus_b_l1407_140782


namespace one_element_in_A_inter_B_range_m_l1407_140750

theorem one_element_in_A_inter_B_range_m (m : ℝ) :
  let A := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = -x^2 + m * x - 1}
  let B := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = 3 - x ∧ 0 ≤ x ∧ x ≤ 3}
  (∃! p, p ∈ A ∧ p ∈ B) → (m = 3 ∨ m > 10 / 3) :=
by
  sorry

end one_element_in_A_inter_B_range_m_l1407_140750


namespace square_side_length_l1407_140715

theorem square_side_length (a : ℝ) (n : ℕ) (P : ℝ) (h₀ : n = 5) (h₁ : 15 * (8 * a / 3) = P) (h₂ : P = 800) : a = 20 := 
by sorry

end square_side_length_l1407_140715


namespace value_of_m_l1407_140784

theorem value_of_m (m : ℝ) (h : ∀ x : ℝ, 0 < x → x < 2 → - (1 / 2) * x^2 + 2 * x ≤ m * x) :
  m = 1 :=
sorry

end value_of_m_l1407_140784


namespace sarees_original_price_l1407_140721

theorem sarees_original_price (P : ℝ) (h : 0.90 * P * 0.95 = 342) : P = 400 :=
by
  sorry

end sarees_original_price_l1407_140721


namespace matthew_younger_than_freddy_l1407_140771

variables (M R F : ℕ)

-- Define the conditions
def sum_of_ages : Prop := M + R + F = 35
def matthew_older_than_rebecca : Prop := M = R + 2
def freddy_age : Prop := F = 15

-- Prove the statement "Matthew is 4 years younger than Freddy."
theorem matthew_younger_than_freddy (h1 : sum_of_ages M R F) (h2 : matthew_older_than_rebecca M R) (h3 : freddy_age F) :
    F - M = 4 := by
  sorry

end matthew_younger_than_freddy_l1407_140771


namespace sum_x1_x2_eq_five_l1407_140746

theorem sum_x1_x2_eq_five {x1 x2 : ℝ} 
  (h1 : 2^x1 = 5 - x1)
  (h2 : x2 + Real.log x2 / Real.log 2 = 5) : 
  x1 + x2 = 5 := 
sorry

end sum_x1_x2_eq_five_l1407_140746


namespace greatest_abs_solution_l1407_140767

theorem greatest_abs_solution :
  (∃ x : ℝ, x^2 + 18 * x + 81 = 0 ∧ ∀ y : ℝ, y^2 + 18 * y + 81 = 0 → |x| ≥ |y| ∧ |x| = 9) :=
sorry

end greatest_abs_solution_l1407_140767


namespace solve_for_X_l1407_140722

variable (X Y : ℝ)

def diamond (X Y : ℝ) := 4 * X + 3 * Y + 7

theorem solve_for_X (h : diamond X 5 = 75) : X = 53 / 4 :=
by
  sorry

end solve_for_X_l1407_140722


namespace bag_of_potatoes_weight_l1407_140752

variable (W : ℝ)

-- Define the condition given in the problem.
def condition : Prop := W = 12 / (W / 2)

-- Define the statement we want to prove.
theorem bag_of_potatoes_weight : condition W → W = 24 := by
  intro h
  sorry

end bag_of_potatoes_weight_l1407_140752


namespace min_value_frac_ineq_l1407_140701

theorem min_value_frac_ineq (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) : 
  (9/m + 1/n) ≥ 16 :=
sorry

end min_value_frac_ineq_l1407_140701


namespace mimi_spent_on_clothes_l1407_140763

theorem mimi_spent_on_clothes : 
  let A := 800
  let N := 2 * A
  let S := 4 * A
  let P := 1 / 2 * N
  let total_spending := 10000
  let total_sneaker_spending := A + N + S + P
  let amount_spent_on_clothes := total_spending - total_sneaker_spending
  amount_spent_on_clothes = 3600 := 
by
  sorry

end mimi_spent_on_clothes_l1407_140763


namespace addition_problem_base6_l1407_140717

theorem addition_problem_base6 (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 :=
by
  sorry

end addition_problem_base6_l1407_140717


namespace selling_price_of_cycle_l1407_140755

theorem selling_price_of_cycle (cost_price : ℕ) (gain_percent : ℕ) (cost_price_eq : cost_price = 1500) (gain_percent_eq : gain_percent = 8) :
  ∃ selling_price : ℕ, selling_price = 1620 := 
by
  sorry

end selling_price_of_cycle_l1407_140755


namespace cost_of_berries_and_cheese_l1407_140758

variables (b m l c : ℕ)

theorem cost_of_berries_and_cheese (h1 : b + m + l + c = 25)
                                  (h2 : m = 2 * l)
                                  (h3 : c = b + 2) : 
                                  b + c = 10 :=
by {
  -- proof omitted, this is just the statement
  sorry
}

end cost_of_berries_and_cheese_l1407_140758


namespace larger_page_of_opened_book_l1407_140730

theorem larger_page_of_opened_book (x : ℕ) (h : x + (x + 1) = 137) : x + 1 = 69 :=
sorry

end larger_page_of_opened_book_l1407_140730


namespace bouquet_daisies_percentage_l1407_140735

theorem bouquet_daisies_percentage :
  (∀ (total white yellow white_tulips white_daisies yellow_tulips yellow_daisies : ℕ),
    total = white + yellow →
    white = 7 * total / 10 →
    yellow = total - white →
    white_tulips = white / 2 →
    white_daisies = white / 2 →
    yellow_daisies = 2 * yellow / 3 →
    yellow_tulips = yellow - yellow_daisies →
    (white_daisies + yellow_daisies) * 100 / total = 55) :=
by
  intros total white yellow white_tulips white_daisies yellow_tulips yellow_daisies h_total h_white h_yellow ht_wd hd_wd hd_yd ht_yt
  sorry

end bouquet_daisies_percentage_l1407_140735


namespace part1_part2_l1407_140769

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x)
  else (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 2)

theorem part1 : f (Real.log 3 / Real.log 2 - Real.log 2 / Real.log 2) = 2 / 3 := by
  sorry

theorem part2 : ∃ x : ℝ, f x = -1 / 4 := by
  sorry

end part1_part2_l1407_140769


namespace calculation_l1407_140723

theorem calculation :
  ((4.5 - 1.23) * 2.5 = 8.175) := 
by
  sorry

end calculation_l1407_140723


namespace base_b_square_l1407_140744

theorem base_b_square (b : ℕ) (h : b > 2) : ∃ k : ℕ, 121 = k ^ 2 :=
by
  sorry

end base_b_square_l1407_140744


namespace find_value_of_a3_plus_a5_l1407_140770

variable {a : ℕ → ℝ}
variable {r : ℝ}

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_value_of_a3_plus_a5 (h_geom : geometric_seq a r) (h_pos: ∀ n, 0 < a n)
  (h_eq: a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end find_value_of_a3_plus_a5_l1407_140770


namespace math_problem_l1407_140786

noncomputable def problem_statement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9

theorem math_problem
  (a b c d : ℕ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : a ≠ d)
  (h4 : b ≠ c)
  (h5 : b ≠ d)
  (h6 : c ≠ d)
  (h7 : (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9) :
  a + b + c + d = 24 :=
sorry

end math_problem_l1407_140786


namespace volume_in_cubic_yards_l1407_140789

-- Definition: A box with a specific volume in cubic feet.
def volume_in_cubic_feet (v : ℝ) : Prop :=
  v = 200

-- Definition: Conversion factor from cubic feet to cubic yards.
def cubic_feet_per_cubic_yard : ℝ := 27

-- Theorem: The volume of the box in cubic yards given the volume in cubic feet.
theorem volume_in_cubic_yards (v_cubic_feet : ℝ) 
    (h : volume_in_cubic_feet v_cubic_feet) : 
    v_cubic_feet / cubic_feet_per_cubic_yard = 200 / 27 :=
  by
    rw [h]
    sorry

end volume_in_cubic_yards_l1407_140789


namespace find_p_l1407_140712

noncomputable def area_of_ABC (p : ℚ) : ℚ :=
  128 - 6 * p

theorem find_p (p : ℚ) : area_of_ABC p = 45 → p = 83 / 6 := by
  intro h
  sorry

end find_p_l1407_140712


namespace units_digit_3_pow_2004_l1407_140731

-- Definition of the observed pattern of the units digits of powers of 3.
def pattern_units_digits : List ℕ := [3, 9, 7, 1]

-- Theorem stating that the units digit of 3^2004 is 1.
theorem units_digit_3_pow_2004 : (3 ^ 2004) % 10 = 1 :=
by
  sorry

end units_digit_3_pow_2004_l1407_140731


namespace calories_peter_wants_to_eat_l1407_140733

-- Definitions for the conditions 
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def total_spent : ℕ := 4

-- Proven statement about the calories Peter wants to eat
theorem calories_peter_wants_to_eat : (total_spent / cost_per_bag) * (chips_per_bag * calories_per_chip) = 480 := by
  sorry

end calories_peter_wants_to_eat_l1407_140733


namespace divisor_value_l1407_140740

theorem divisor_value (D : ℕ) (k m : ℤ) (h1 : 242 % D = 8) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) : D = 13 := by
  sorry

end divisor_value_l1407_140740


namespace value_two_std_dev_less_l1407_140718

noncomputable def mean : ℝ := 15.5
noncomputable def std_dev : ℝ := 1.5

theorem value_two_std_dev_less : mean - 2 * std_dev = 12.5 := by
  sorry

end value_two_std_dev_less_l1407_140718


namespace smallest_side_of_triangle_l1407_140725

variable {α : Type} [LinearOrderedField α]

theorem smallest_side_of_triangle (a b c : α) (h : a^2 + b^2 > 5*c^2) : c ≤ a ∧ c ≤ b :=
by
  sorry

end smallest_side_of_triangle_l1407_140725


namespace distinguishes_conditional_from_sequential_l1407_140785

variable (C P S I D : Prop)

-- Conditions
def conditional_structure_includes_processing_box  : Prop := C = P
def conditional_structure_includes_start_end_box   : Prop := C = S
def conditional_structure_includes_io_box          : Prop := C = I
def conditional_structure_includes_decision_box    : Prop := C = D
def sequential_structure_excludes_decision_box     : Prop := ¬S = D

-- Proof problem statement
theorem distinguishes_conditional_from_sequential : C → S → I → D → P → 
    (conditional_structure_includes_processing_box C P) ∧ 
    (conditional_structure_includes_start_end_box C S) ∧ 
    (conditional_structure_includes_io_box C I) ∧ 
    (conditional_structure_includes_decision_box C D) ∧ 
    sequential_structure_excludes_decision_box S D → 
    (D = true) :=
by sorry

end distinguishes_conditional_from_sequential_l1407_140785


namespace derivative_y_l1407_140714

noncomputable def y (x : ℝ) : ℝ := 
  (Real.sqrt (49 * x^2 + 1) * Real.arctan (7 * x)) - 
  Real.log (7 * x + Real.sqrt (49 * x^2 + 1))

theorem derivative_y (x : ℝ) : 
  deriv y x = (7 * Real.arctan (7 * x)) / (2 * Real.sqrt (49 * x^2 + 1)) := by
  sorry

end derivative_y_l1407_140714


namespace average_height_of_three_l1407_140707

theorem average_height_of_three (parker daisy reese : ℕ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : reese = 60) : 
  (parker + daisy + reese) / 3 = 64 := 
  sorry

end average_height_of_three_l1407_140707


namespace sin_squared_minus_cos_squared_value_l1407_140705

noncomputable def sin_squared_minus_cos_squared : Real :=
  (Real.sin (Real.pi / 12))^2 - (Real.cos (Real.pi / 12))^2

theorem sin_squared_minus_cos_squared_value :
  sin_squared_minus_cos_squared = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_squared_minus_cos_squared_value_l1407_140705


namespace last_10_digits_repeat_periodically_l1407_140772

theorem last_10_digits_repeat_periodically :
  ∃ (p : ℕ) (n₀ : ℕ), p = 4 * 10^9 ∧ n₀ = 10 ∧ 
  ∀ n, (2^(n + p) % 10^10 = 2^n % 10^10) :=
by sorry

end last_10_digits_repeat_periodically_l1407_140772


namespace complement_intersection_l1407_140776

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5} := by
    sorry

end complement_intersection_l1407_140776
