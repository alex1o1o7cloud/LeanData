import Mathlib

namespace NUMINAMATH_GPT_water_spilled_l648_64822

theorem water_spilled (x s : ℕ) (h1 : s = x + 7) : s = 8 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_water_spilled_l648_64822


namespace NUMINAMATH_GPT_find_m_l648_64812

theorem find_m (a : ℕ → ℝ) (m : ℕ) (h_pos : m > 0) 
  (h_a0 : a 0 = 37) (h_a1 : a 1 = 72) (h_am : a m = 0)
  (h_rec : ∀ k, 1 ≤ k ∧ k ≤ m - 1 → a (k + 1) = a (k - 1) - 3 / a k) :
  m = 889 :=
sorry

end NUMINAMATH_GPT_find_m_l648_64812


namespace NUMINAMATH_GPT_fuel_tank_capacity_l648_64848

theorem fuel_tank_capacity (C : ℝ) (h1 : 0.12 * 106 + 0.16 * (C - 106) = 30) : C = 214 :=
by
  sorry

end NUMINAMATH_GPT_fuel_tank_capacity_l648_64848


namespace NUMINAMATH_GPT_inequality_1_inequality_2_l648_64835

theorem inequality_1 (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) → x ≤ -3/2 :=
by
  sorry

theorem inequality_2 (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 → x ≥ -2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_l648_64835


namespace NUMINAMATH_GPT_sum_of_A_and_B_l648_64837

theorem sum_of_A_and_B (A B : ℕ) (h1 : A ≠ B) (h2 : A < 10) (h3 : B < 10) :
  (10 * A + B) * 6 = 111 * B → A + B = 11 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_A_and_B_l648_64837


namespace NUMINAMATH_GPT_jenny_stamps_l648_64896

theorem jenny_stamps :
  let num_books := 8
  let pages_per_book := 42
  let stamps_per_page := 6
  let new_stamps_per_page := 10
  let complete_books_in_new_system := 4
  let pages_in_fifth_book := 33
  (num_books * pages_per_book * stamps_per_page) % new_stamps_per_page = 6 :=
by
  sorry

end NUMINAMATH_GPT_jenny_stamps_l648_64896


namespace NUMINAMATH_GPT_lion_cubs_per_month_l648_64858

theorem lion_cubs_per_month
  (initial_lions : ℕ)
  (final_lions : ℕ)
  (months : ℕ)
  (lions_dying_per_month : ℕ)
  (net_increase : ℕ)
  (x : ℕ) : 
  initial_lions = 100 → 
  final_lions = 148 → 
  months = 12 → 
  lions_dying_per_month = 1 → 
  net_increase = 48 → 
  12 * (x - 1) = net_increase → 
  x = 5 := by
  intros initial_lions_eq final_lions_eq months_eq lions_dying_eq net_increase_eq equation
  sorry

end NUMINAMATH_GPT_lion_cubs_per_month_l648_64858


namespace NUMINAMATH_GPT_least_multiplier_l648_64895

theorem least_multiplier (x: ℕ) (h1: 72 * x % 112 = 0) (h2: ∀ y, 72 * y % 112 = 0 → x ≤ y) : x = 14 :=
sorry

end NUMINAMATH_GPT_least_multiplier_l648_64895


namespace NUMINAMATH_GPT_Norine_retire_age_l648_64897

theorem Norine_retire_age:
  ∀ (A W : ℕ),
    (A = 50) →
    (W = 19) →
    (A + W = 85) →
    (A = 50 + 8) :=
by
  intros A W hA hW hAW
  sorry

end NUMINAMATH_GPT_Norine_retire_age_l648_64897


namespace NUMINAMATH_GPT_parallel_lines_condition_l648_64880

theorem parallel_lines_condition {a : ℝ} :
  (∀ x y : ℝ, a * x + 2 * y + 3 * a = 0) ∧ (∀ x y : ℝ, 3 * x + (a - 1) * y = a - 7) ↔ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l648_64880


namespace NUMINAMATH_GPT_arithmetic_sums_l648_64805

theorem arithmetic_sums (d : ℤ) (p q : ℤ) (S : ℤ → ℤ)
  (hS : ∀ n, S n = p * n^2 + q * n)
  (h_eq : S 20 = S 40) : S 60 = 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sums_l648_64805


namespace NUMINAMATH_GPT_unique_solution_l648_64825

noncomputable def f (a b x : ℝ) := 2 * (a + b) * Real.exp (2 * x) + 2 * a * b
noncomputable def g (a b x : ℝ) := 4 * Real.exp (2 * x) + a + b

theorem unique_solution (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃! x, f a b x = ( (a^(1/3) + b^(1/3))/2 )^3 * g a b x :=
sorry

end NUMINAMATH_GPT_unique_solution_l648_64825


namespace NUMINAMATH_GPT_intersection_eq_l648_64881

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : S ∩ T = { x | -2 < x ∧ x ≤ 1 } :=
by
  simp [S, T]
  sorry

end NUMINAMATH_GPT_intersection_eq_l648_64881


namespace NUMINAMATH_GPT_rook_path_exists_l648_64846

theorem rook_path_exists :
  ∃ (path : Finset (Fin 8 × Fin 8)) (s1 s2 : Fin 8 × Fin 8),
  s1 ≠ s2 ∧
  s1.1 % 2 = s2.1 % 2 ∧ s1.2 % 2 = s2.2 % 2 ∧
  ∀ s : Fin 8 × Fin 8, s ∈ path ∧ s ≠ s2 :=
sorry

end NUMINAMATH_GPT_rook_path_exists_l648_64846


namespace NUMINAMATH_GPT_avg_ticket_cost_per_person_l648_64882

-- Define the conditions
def full_price : ℤ := 150
def half_price : ℤ := full_price / 2
def num_full_price_tickets : ℤ := 2
def num_half_price_tickets : ℤ := 2
def free_tickets : ℤ := 1
def total_people : ℤ := 5

-- Prove that the average cost of tickets per person is 90 yuan
theorem avg_ticket_cost_per_person : ((num_full_price_tickets * full_price + num_half_price_tickets * half_price) / total_people) = 90 := 
by 
  sorry

end NUMINAMATH_GPT_avg_ticket_cost_per_person_l648_64882


namespace NUMINAMATH_GPT_polynomial_sum_l648_64820

theorem polynomial_sum :
  let f := (x^3 + 9*x^2 + 26*x + 24) 
  let g := (x + 3)
  let A := 1
  let B := 6
  let C := 8
  let D := -3
  (y = f/g) → (A + B + C + D = 12) :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_sum_l648_64820


namespace NUMINAMATH_GPT_sum_even_numbers_l648_64874

def is_even (n : ℕ) : Prop := n % 2 = 0

def largest_even_less_than_or_equal (n m : ℕ) : ℕ :=
if h : m % 2 = 0 ∧ m ≤ n then m else
if h : m % 2 = 1 ∧ (m - 1) ≤ n then m - 1 else 0

def smallest_even_less_than_or_equal (n : ℕ) : ℕ :=
if h : 2 ≤ n then 2 else 0

theorem sum_even_numbers (n : ℕ) (h : n = 49) :
  largest_even_less_than_or_equal n 48 + smallest_even_less_than_or_equal n = 50 :=
by sorry

end NUMINAMATH_GPT_sum_even_numbers_l648_64874


namespace NUMINAMATH_GPT_range_of_a_l648_64879

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^2 + 2*a*x + a) > 0) → (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l648_64879


namespace NUMINAMATH_GPT_minimum_g7_l648_64863

def is_tenuous (g : ℕ → ℤ) : Prop :=
∀ x y : ℕ, 0 < x → 0 < y → g x + g y > x^2

noncomputable def min_possible_value_g7 (g : ℕ → ℤ) (h : is_tenuous g) 
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) : ℤ :=
g 7

theorem minimum_g7 (g : ℕ → ℤ) (h : is_tenuous g)
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) :
  min_possible_value_g7 g h h_sum = 49 :=
sorry

end NUMINAMATH_GPT_minimum_g7_l648_64863


namespace NUMINAMATH_GPT_actual_size_of_plot_l648_64813

/-
Theorem: The actual size of the plot of land is 61440 acres.
Given:
- The plot of land is a rectangle.
- The map dimensions are 12 cm by 8 cm.
- 1 cm on the map equals 1 mile in reality.
- One square mile equals 640 acres.
-/

def map_length_cm := 12
def map_width_cm := 8
def cm_to_miles := 1 -- 1 cm equals 1 mile
def mile_to_acres := 640 -- 1 square mile is 640 acres

theorem actual_size_of_plot
  (length_cm : ℕ) (width_cm : ℕ) (cm_to_miles : ℕ → ℕ) (mile_to_acres : ℕ → ℕ) :
  length_cm = 12 → width_cm = 8 →
  (cm_to_miles 1 = 1) →
  (mile_to_acres 1 = 640) →
  (length_cm * width_cm * mile_to_acres (cm_to_miles 1 * cm_to_miles 1) = 61440) :=
by
  intros
  sorry

end NUMINAMATH_GPT_actual_size_of_plot_l648_64813


namespace NUMINAMATH_GPT_bicycle_has_four_wheels_l648_64894

variables (Car : Type) (Bicycle : Car) (FourWheeled : Car → Prop)
axiom car_four_wheels : ∀ (c : Car), FourWheeled c

theorem bicycle_has_four_wheels : FourWheeled Bicycle :=
by {
  apply car_four_wheels
}

end NUMINAMATH_GPT_bicycle_has_four_wheels_l648_64894


namespace NUMINAMATH_GPT_min_value_l648_64821

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) : a + 4 * b ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_l648_64821


namespace NUMINAMATH_GPT_age_difference_is_16_l648_64833

-- Variables
variables (y : ℕ) -- y represents the present age of the younger person

-- Conditions from the problem
def elder_present_age := 30
def elder_age_6_years_ago := elder_present_age - 6
def younger_age_6_years_ago := y - 6

-- Given condition 6 years ago:
def condition_6_years_ago := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- The theorem to prove the difference in ages is 16 years
theorem age_difference_is_16
  (h1 : elder_present_age = 30)
  (h2 : condition_6_years_ago) :
  elder_present_age - y = 16 :=
by sorry

end NUMINAMATH_GPT_age_difference_is_16_l648_64833


namespace NUMINAMATH_GPT_exists_special_sequence_l648_64801

open List
open Finset
open BigOperators

theorem exists_special_sequence :
  ∃ s : ℕ → ℕ,
    (∀ n, s n > 0) ∧
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    (∀ k, (∑ i in range (k + 1), s i) % (k + 1) = 0) :=
sorry  -- Proof from the provided solution steps.

end NUMINAMATH_GPT_exists_special_sequence_l648_64801


namespace NUMINAMATH_GPT_total_distance_traveled_l648_64857

def speed := 60  -- Jace drives 60 miles per hour
def first_leg_time := 4  -- Jace drives for 4 hours straight
def break_time := 0.5  -- Jace takes a 30-minute break (0.5 hours)
def second_leg_time := 9  -- Jace drives for another 9 hours straight

def distance (speed : ℕ) (time : ℕ) : ℕ := speed * time  -- Distance formula

theorem total_distance_traveled : 
  distance speed first_leg_time + distance speed second_leg_time = 780 := by
-- Sorry allows us to skip the proof, since only the statement is required.
sorry

end NUMINAMATH_GPT_total_distance_traveled_l648_64857


namespace NUMINAMATH_GPT_percentage_reduction_price_increase_l648_64873

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end NUMINAMATH_GPT_percentage_reduction_price_increase_l648_64873


namespace NUMINAMATH_GPT_surface_area_of_resulting_solid_l648_64887

-- Define the original cube dimensions
def original_cube_surface_area (s : ℕ) := 6 * s * s

-- Define the smaller cube dimensions to be cut
def small_cube_surface_area (s : ℕ) := 3 * s * s

-- Define the proof problem
theorem surface_area_of_resulting_solid :
  original_cube_surface_area 3 - small_cube_surface_area 1 - small_cube_surface_area 2 + (3 * 1 + 3 * 4) = 54 :=
by
  -- The actual proof is to be filled in here
  sorry

end NUMINAMATH_GPT_surface_area_of_resulting_solid_l648_64887


namespace NUMINAMATH_GPT_sumata_family_total_miles_l648_64818

theorem sumata_family_total_miles
  (days : ℝ) (miles_per_day : ℝ)
  (h1 : days = 5.0)
  (h2 : miles_per_day = 250) : 
  miles_per_day * days = 1250 := 
by
  sorry

end NUMINAMATH_GPT_sumata_family_total_miles_l648_64818


namespace NUMINAMATH_GPT_garden_perimeter_l648_64870

theorem garden_perimeter (L B : ℕ) (hL : L = 100) (hB : B = 200) : 
  2 * (L + B) = 600 := by
sorry

end NUMINAMATH_GPT_garden_perimeter_l648_64870


namespace NUMINAMATH_GPT_rachel_colored_pictures_l648_64861

theorem rachel_colored_pictures :
  ∃ b1 b2 : ℕ, b1 = 23 ∧ b2 = 32 ∧ ∃ remaining: ℕ, remaining = 11 ∧ (b1 + b2) - remaining = 44 :=
by
  sorry

end NUMINAMATH_GPT_rachel_colored_pictures_l648_64861


namespace NUMINAMATH_GPT_purchase_costs_10_l648_64883

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end NUMINAMATH_GPT_purchase_costs_10_l648_64883


namespace NUMINAMATH_GPT_arrangement_proof_l648_64838

/-- The Happy Valley Zoo houses 5 chickens, 3 dogs, and 6 cats in a large exhibit area
    with separate but adjacent enclosures. We need to find the number of ways to place
    the 14 animals in a row of 14 enclosures, ensuring all animals of each type are together,
    and that chickens are always placed before cats, but with no restrictions regarding the
    placement of dogs. -/
def number_of_arrangements : ℕ :=
  let chickens := 5
  let dogs := 3
  let cats := 6
  let chicken_permutations := Nat.factorial chickens
  let dog_permutations := Nat.factorial dogs
  let cat_permutations := Nat.factorial cats
  let group_arrangements := 3 -- Chickens-Dogs-Cats, Dogs-Chickens-Cats, Chickens-Cats-Dogs
  group_arrangements * chicken_permutations * dog_permutations * cat_permutations

theorem arrangement_proof : number_of_arrangements = 1555200 :=
by 
  sorry

end NUMINAMATH_GPT_arrangement_proof_l648_64838


namespace NUMINAMATH_GPT_line_circle_intersection_range_l648_64851

theorem line_circle_intersection_range (b : ℝ) :
    (2 - Real.sqrt 2) < b ∧ b < (2 + Real.sqrt 2) ↔
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ ((p1.1 - 2)^2 + p1.2^2 = 1) ∧ ((p2.1 - 2)^2 + p2.2^2 = 1) ∧ (p1.2 = p1.1 - b ∧ p2.2 = p2.1 - b) :=
by
  sorry

end NUMINAMATH_GPT_line_circle_intersection_range_l648_64851


namespace NUMINAMATH_GPT_remainder_b_91_mod_49_l648_64827

def b (n : ℕ) := 12^n + 14^n

theorem remainder_b_91_mod_49 : (b 91) % 49 = 38 := by
  sorry

end NUMINAMATH_GPT_remainder_b_91_mod_49_l648_64827


namespace NUMINAMATH_GPT_Bing_max_games_l648_64890

/-- 
  Jia, Yi, and Bing play table tennis with the following rules: each game is played between two 
  people, and the loser gives way to the third person. If Jia played 10 games and Yi played 
  7 games, then Bing can play at most 13 games; and can win at most 10 games.
-/
theorem Bing_max_games 
  (games_played_Jia : ℕ)
  (games_played_Yi : ℕ)
  (games_played_Bing : ℕ)
  (games_won_Bing  : ℕ)
  (hJia : games_played_Jia = 10)
  (hYi : games_played_Yi = 7) :
  (games_played_Bing ≤ 13) ∧ (games_won_Bing ≤ 10) := 
sorry

end NUMINAMATH_GPT_Bing_max_games_l648_64890


namespace NUMINAMATH_GPT_find_x_l648_64869

theorem find_x (x : ℝ) : 
  3.5 * ( (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x) ) = 2800.0000000000005 → x = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l648_64869


namespace NUMINAMATH_GPT_most_suitable_candidate_l648_64852

-- Definitions for variances
def variance_A := 3.4
def variance_B := 2.1
def variance_C := 2.5
def variance_D := 2.7

-- We start the theorem to state the most suitable candidate based on given variances and average scores.
theorem most_suitable_candidate :
  (variance_A = 3.4) ∧ (variance_B = 2.1) ∧ (variance_C = 2.5) ∧ (variance_D = 2.7) →
  true := 
by
  sorry

end NUMINAMATH_GPT_most_suitable_candidate_l648_64852


namespace NUMINAMATH_GPT_total_seats_round_table_l648_64871

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end NUMINAMATH_GPT_total_seats_round_table_l648_64871


namespace NUMINAMATH_GPT_current_algae_plants_l648_64899

def original_algae_plants : ℕ := 809
def additional_algae_plants : ℕ := 2454

theorem current_algae_plants :
  original_algae_plants + additional_algae_plants = 3263 := by
  sorry

end NUMINAMATH_GPT_current_algae_plants_l648_64899


namespace NUMINAMATH_GPT_evaluate_expression_l648_64845

theorem evaluate_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l648_64845


namespace NUMINAMATH_GPT_distribution_centers_count_l648_64889

theorem distribution_centers_count (n : ℕ) (h : n = 5) : n + (n * (n - 1)) / 2 = 15 :=
by
  subst h -- replace n with 5
  show 5 + (5 * (5 - 1)) / 2 = 15
  have : (5 * 4) / 2 = 10 := by norm_num
  show 5 + 10 = 15
  norm_num

end NUMINAMATH_GPT_distribution_centers_count_l648_64889


namespace NUMINAMATH_GPT_largest_four_digit_number_mod_l648_64892

theorem largest_four_digit_number_mod (n : ℕ) : 
  (n < 10000) → 
  (n % 11 = 2) → 
  (n % 7 = 4) → 
  n ≤ 9973 :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_mod_l648_64892


namespace NUMINAMATH_GPT_park_maple_trees_total_l648_64831

theorem park_maple_trees_total (current_maples planted_maples : ℕ) 
    (h1 : current_maples = 2) (h2 : planted_maples = 9) 
    : current_maples + planted_maples = 11 := 
by
  sorry

end NUMINAMATH_GPT_park_maple_trees_total_l648_64831


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l648_64810

theorem first_term_of_geometric_series (a r S : ℝ)
  (h_sum : S = a / (1 - r))
  (h_r : r = 1/3)
  (h_S : S = 18) :
  a = 12 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l648_64810


namespace NUMINAMATH_GPT_ratio_of_sugar_to_flour_l648_64829

theorem ratio_of_sugar_to_flour
  (F B : ℕ)
  (h1 : F = 10 * B)
  (h2 : F = 8 * (B + 60))
  (sugar : ℕ)
  (hs : sugar = 2000) :
  sugar / F = 5 / 6 :=
by {
  sorry -- proof omitted
}

end NUMINAMATH_GPT_ratio_of_sugar_to_flour_l648_64829


namespace NUMINAMATH_GPT_car_and_truck_arrival_time_simultaneous_l648_64808

theorem car_and_truck_arrival_time_simultaneous {t_car t_truck : ℕ} 
    (h1 : t_car = 8 * 60 + 16) -- Car leaves at 08:16
    (h2 : t_truck = 9 * 60) -- Truck leaves at 09:00
    (h3 : t_car_arrive = 10 * 60 + 56) -- Car arrives at 10:56
    (h4 : t_truck_arrive = 12 * 60 + 20) -- Truck arrives at 12:20
    (h5 : t_truck_exit = t_car_exit + 2) -- Truck leaves tunnel 2 minutes after car
    : (t_car_exit + t_car_tunnel_time = 10 * 60) ∧ (t_truck_exit + t_truck_tunnel_time = 10 * 60) :=
  sorry

end NUMINAMATH_GPT_car_and_truck_arrival_time_simultaneous_l648_64808


namespace NUMINAMATH_GPT_problem_l648_64884

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1) * d) / 2

theorem problem (a1 S3 : ℕ) (a1_eq : a1 = 2) (S3_eq : S3 = 12) : 
  ∃ a6 : ℕ, a6 = 12 := by
  let a2 := (S3 - a1) / 2
  let d := a2 - a1
  let a6 := a1 + 5 * d
  use a6
  sorry

end NUMINAMATH_GPT_problem_l648_64884


namespace NUMINAMATH_GPT_all_numbers_non_positive_l648_64872

theorem all_numbers_non_positive 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → (a (k - 1) - 2 * a k + a (k + 1) ≥ 0)) : 
  ∀ k, 0 ≤ k → k ≤ n → a k ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_all_numbers_non_positive_l648_64872


namespace NUMINAMATH_GPT_general_term_formula_of_a_l648_64809

def S (n : ℕ) : ℚ := (3 / 2) * n^2 - 2 * n

def a (n : ℕ) : ℚ :=
  if n = 1 then (3 / 2) - 2
  else 2 * (3 / 2) * n - (3 / 2) - 2

theorem general_term_formula_of_a :
  ∀ n : ℕ, n > 0 → a n = 3 * n - (7 / 2) :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_general_term_formula_of_a_l648_64809


namespace NUMINAMATH_GPT_will_pages_needed_l648_64807

theorem will_pages_needed :
  let new_cards_2020 := 8
  let old_cards := 10
  let duplicates := 2
  let cards_per_page := 3
  let unique_old_cards := old_cards - duplicates
  let pages_needed_for_2020 := (new_cards_2020 + cards_per_page - 1) / cards_per_page -- ceil(new_cards_2020 / cards_per_page)
  let pages_needed_for_old := (unique_old_cards + cards_per_page - 1) / cards_per_page -- ceil(unique_old_cards / cards_per_page)
  let pages_needed := pages_needed_for_2020 + pages_needed_for_old
  pages_needed = 6 :=
by
  sorry

end NUMINAMATH_GPT_will_pages_needed_l648_64807


namespace NUMINAMATH_GPT_minimize_PA2_plus_PB2_plus_PC2_l648_64817

def PA (x y : ℝ) : ℝ := (x - 3) ^ 2 + (y + 1) ^ 2
def PB (x y : ℝ) : ℝ := (x + 1) ^ 2 + (y - 4) ^ 2
def PC (x y : ℝ) : ℝ := (x - 1) ^ 2 + (y + 6) ^ 2

theorem minimize_PA2_plus_PB2_plus_PC2 :
  ∃ x y : ℝ, (PA x y + PB x y + PC x y) = 64 :=
by
  use 1
  use -1
  simp [PA, PB, PC]
  sorry

end NUMINAMATH_GPT_minimize_PA2_plus_PB2_plus_PC2_l648_64817


namespace NUMINAMATH_GPT_symmetrical_shapes_congruent_l648_64830

theorem symmetrical_shapes_congruent
  (shapes : Type)
  (is_symmetrical : shapes → shapes → Prop)
  (congruent : shapes → shapes → Prop)
  (symmetrical_implies_equal_segments : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (segment : ℝ), segment_s1 = segment_s2)
  (symmetrical_implies_equal_angles : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (angle : ℝ), angle_s1 = angle_s2) :
  ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → congruent s1 s2 :=
by
  sorry

end NUMINAMATH_GPT_symmetrical_shapes_congruent_l648_64830


namespace NUMINAMATH_GPT_eq_irrational_parts_l648_64811

theorem eq_irrational_parts (a b c d : ℝ) (h : a + b * (Real.sqrt 5) = c + d * (Real.sqrt 5)) : a = c ∧ b = d := 
by 
  sorry

end NUMINAMATH_GPT_eq_irrational_parts_l648_64811


namespace NUMINAMATH_GPT_sin_2012_eq_neg_sin_32_l648_64876

theorem sin_2012_eq_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = - Real.sin (32 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_sin_2012_eq_neg_sin_32_l648_64876


namespace NUMINAMATH_GPT_max_value_expression_l648_64888

noncomputable def max_expression (a b c : ℝ) : ℝ :=
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3)

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_expression a b c ≤ 1 / 12 := 
sorry

end NUMINAMATH_GPT_max_value_expression_l648_64888


namespace NUMINAMATH_GPT_square_div_by_144_l648_64802

theorem square_div_by_144 (n : ℕ) (h1 : ∃ (k : ℕ), n = 12 * k) : ∃ (m : ℕ), n^2 = 144 * m :=
by
  sorry

end NUMINAMATH_GPT_square_div_by_144_l648_64802


namespace NUMINAMATH_GPT_problem1_problem2_l648_64865

-- Problem 1
theorem problem1 : 40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12 = 43 :=
by
  sorry

-- Problem 2
theorem problem2 : (-1) ^ 2 * (-5) + ((-3) ^ 2 + 2 * (-5)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l648_64865


namespace NUMINAMATH_GPT_total_eggs_found_l648_64832

def eggs_club_house := 12
def eggs_park := 5
def eggs_town_hall_garden := 3

theorem total_eggs_found : eggs_club_house + eggs_park + eggs_town_hall_garden = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_eggs_found_l648_64832


namespace NUMINAMATH_GPT_vegetables_sold_ratio_l648_64849

def totalMassInstalled (carrots zucchini broccoli : ℕ) : ℕ := carrots + zucchini + broccoli

def massSold (soldMass : ℕ) : ℕ := soldMass

def vegetablesSoldRatio (carrots zucchini broccoli soldMass : ℕ) : ℚ :=
  soldMass / (carrots + zucchini + broccoli)

theorem vegetables_sold_ratio
  (carrots zucchini broccoli soldMass : ℕ)
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8)
  (h_soldMass : soldMass = 18) :
  vegetablesSoldRatio carrots zucchini broccoli soldMass = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_vegetables_sold_ratio_l648_64849


namespace NUMINAMATH_GPT_prime_factors_identity_l648_64839

theorem prime_factors_identity (w x y z k : ℕ) 
    (h : 2^w * 3^x * 5^y * 7^z * 11^k = 900) : 
      2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 20 :=
by
  sorry

end NUMINAMATH_GPT_prime_factors_identity_l648_64839


namespace NUMINAMATH_GPT_larry_result_is_correct_l648_64893

theorem larry_result_is_correct (a b c d e : ℤ) 
  (h1: a = 2) (h2: b = 4) (h3: c = 3) (h4: d = 5) (h5: e = -15) :
  a - (b - (c * (d + e))) = (-17 + e) :=
by 
  rw [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_larry_result_is_correct_l648_64893


namespace NUMINAMATH_GPT_total_octopus_legs_l648_64875

-- Define the number of octopuses Carson saw
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- Define or state the theorem for total number of legs
theorem total_octopus_legs : num_octopuses * legs_per_octopus = 40 := by
  sorry

end NUMINAMATH_GPT_total_octopus_legs_l648_64875


namespace NUMINAMATH_GPT_probability_interval_l648_64877

theorem probability_interval (P_A P_B : ℚ) (h1 : P_A = 5/6) (h2 : P_B = 3/4) :
  ∃ p : ℚ, (5/12 ≤ p ∧ p ≤ 3/4) :=
sorry

end NUMINAMATH_GPT_probability_interval_l648_64877


namespace NUMINAMATH_GPT_correct_statement_l648_64841

theorem correct_statement :
  (∃ (A : Prop), A = (2 * x^3 - 4 * x - 3 ≠ 3)) ∧
  (∃ (B : Prop), B = ((2 + 3) ≠ 6)) ∧
  (∃ (C : Prop), C = (-4 * x^2 * y = -4)) ∧
  (∃ (D : Prop), D = (1 = 1 ∧ 1 = 1 / 8)) →
  (C) :=
by sorry

end NUMINAMATH_GPT_correct_statement_l648_64841


namespace NUMINAMATH_GPT_distance_from_origin_to_line_l648_64891

theorem distance_from_origin_to_line : 
  let A := 1
  let B := 2
  let C := -5
  let x_0 := 0
  let y_0 := 0
  let distance := |A * x_0 + B * y_0 + C| / (Real.sqrt (A ^ 2 + B ^ 2))
  distance = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_to_line_l648_64891


namespace NUMINAMATH_GPT_any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l648_64828
open Int

variable (m n : ℕ) (h : Nat.gcd m n = 1)

theorem any_integer_amount_purchasable (x : ℤ) : 
  ∃ (a b : ℤ), a * n + b * m = x :=
by sorry

theorem amount_over_mn_minus_two_payable (k : ℤ) (hk : k > m * n - 2) : 
  ∃ (a b : ℤ), a * n + b * m = k :=
by sorry

end NUMINAMATH_GPT_any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l648_64828


namespace NUMINAMATH_GPT_smallest_repeating_block_fraction_3_over_11_l648_64855

theorem smallest_repeating_block_fraction_3_over_11 :
  ∃ n : ℕ, (∃ d : ℕ, (3/11 : ℚ) = d / 10^n) ∧ n = 2 :=
sorry

end NUMINAMATH_GPT_smallest_repeating_block_fraction_3_over_11_l648_64855


namespace NUMINAMATH_GPT_stickers_in_either_not_both_l648_64843

def stickers_shared := 12
def emily_total_stickers := 22
def mia_unique_stickers := 10

theorem stickers_in_either_not_both : 
  (emily_total_stickers - stickers_shared) + mia_unique_stickers = 20 :=
by
  sorry

end NUMINAMATH_GPT_stickers_in_either_not_both_l648_64843


namespace NUMINAMATH_GPT_train_speed_l648_64866

theorem train_speed (length : ℝ) (time : ℝ)
  (length_pos : length = 160) (time_pos : time = 8) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l648_64866


namespace NUMINAMATH_GPT_complement_of_A_eq_l648_64862

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x > 1}

theorem complement_of_A_eq {U : Set ℝ} (U_eq : U = Set.univ) {A : Set ℝ} (A_eq : A = {x | x > 1}) :
    U \ A = {x | x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_eq_l648_64862


namespace NUMINAMATH_GPT_baseball_opponents_score_l648_64814

theorem baseball_opponents_score 
  (team_scores : List ℕ)
  (team_lost_scores : List ℕ)
  (team_won_scores : List ℕ)
  (opponent_lost_scores : List ℕ)
  (opponent_won_scores : List ℕ)
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : team_lost_scores = [1, 3, 5, 7, 9, 11])
  (h3 : team_won_scores = [6, 9, 12])
  (h4 : opponent_lost_scores = [3, 5, 7, 9, 11, 13])
  (h5 : opponent_won_scores = [2, 3, 4]) :
  (List.sum opponent_lost_scores + List.sum opponent_won_scores = 57) :=
sorry

end NUMINAMATH_GPT_baseball_opponents_score_l648_64814


namespace NUMINAMATH_GPT_coterminal_angle_l648_64867

theorem coterminal_angle :
  ∀ θ : ℤ, (θ - 60) % 360 = 0 → θ = -300 ∨ θ = -60 ∨ θ = 600 ∨ θ = 1380 :=
by
  sorry

end NUMINAMATH_GPT_coterminal_angle_l648_64867


namespace NUMINAMATH_GPT_deny_evenness_l648_64878

-- We need to define the natural numbers and their parity.
variables {a b c : ℕ}

-- Define what it means for a number to be odd and even.
def is_odd (n : ℕ) := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) := ∃ k, n = 2 * k

-- The Lean theorem statement translating the given problem.
theorem deny_evenness :
  (is_odd a ∧ is_odd b ∧ is_odd c) → ¬(is_even a ∨ is_even b ∨ is_even c) :=
by sorry

end NUMINAMATH_GPT_deny_evenness_l648_64878


namespace NUMINAMATH_GPT_minimum_value_inequality_l648_64804

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x * y * z * (x + y + z) = 1) : (x + y) * (y + z) ≥ 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l648_64804


namespace NUMINAMATH_GPT_football_cost_correct_l648_64842

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end NUMINAMATH_GPT_football_cost_correct_l648_64842


namespace NUMINAMATH_GPT_completing_the_square_l648_64815

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end NUMINAMATH_GPT_completing_the_square_l648_64815


namespace NUMINAMATH_GPT_trains_meet_distance_l648_64824

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

end NUMINAMATH_GPT_trains_meet_distance_l648_64824


namespace NUMINAMATH_GPT_range_y_eq_2cosx_minus_1_range_y_eq_sq_2sinx_minus_1_plus_3_l648_64816

open Real

theorem range_y_eq_2cosx_minus_1 : 
  (∀ x : ℝ, -1 ≤ cos x ∧ cos x ≤ 1) →
  (∀ y : ℝ, y = 2 * (cos x) - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  intros h1 y h2
  sorry

theorem range_y_eq_sq_2sinx_minus_1_plus_3 : 
  (∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1) →
  (∀ y : ℝ, y = (2 * (sin x) - 1)^2 + 3 → 3 ≤ y ∧ y ≤ 12) :=
by
  intros h1 y h2
  sorry

end NUMINAMATH_GPT_range_y_eq_2cosx_minus_1_range_y_eq_sq_2sinx_minus_1_plus_3_l648_64816


namespace NUMINAMATH_GPT_fraction_is_three_halves_l648_64898

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_is_three_halves_l648_64898


namespace NUMINAMATH_GPT_profit_growth_equation_l648_64826

noncomputable def profitApril : ℝ := 250000
noncomputable def profitJune : ℝ := 360000
noncomputable def averageMonthlyGrowth (x : ℝ) : ℝ := 25 * (1 + x) * (1 + x)

theorem profit_growth_equation (x : ℝ) :
  averageMonthlyGrowth x = 36 * 10000 ↔ 25 * (1 + x)^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_profit_growth_equation_l648_64826


namespace NUMINAMATH_GPT_plane_equation_rewriting_l648_64800

theorem plane_equation_rewriting (A B C D x y z p q r : ℝ)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (eq1 : A * x + B * y + C * z + D = 0)
  (hp : p = -D / A) (hq : q = -D / B) (hr : r = -D / C) :
  x / p + y / q + z / r = 1 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_rewriting_l648_64800


namespace NUMINAMATH_GPT_line_circle_no_intersection_l648_64840

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), 3 * x + 4 * y ≠ 12 ∧ x^2 + y^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l648_64840


namespace NUMINAMATH_GPT_Sophia_fraction_finished_l648_64864

/--
Sophia finished a fraction of a book.
She calculated that she finished 90 more pages than she has yet to read.
Her book is 270.00000000000006 pages long.
Prove that the fraction of the book she finished is 2/3.
-/
theorem Sophia_fraction_finished :
  let total_pages : ℝ := 270.00000000000006
  let yet_to_read : ℝ := (total_pages - 90) / 2
  let finished_pages : ℝ := yet_to_read + 90
  finished_pages / total_pages = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_Sophia_fraction_finished_l648_64864


namespace NUMINAMATH_GPT_minimize_average_comprehensive_cost_l648_64885

theorem minimize_average_comprehensive_cost :
  ∀ (f : ℕ → ℝ), (∀ (x : ℕ), x ≥ 10 → f x = 560 + 48 * x + 10800 / x) →
  ∃ x : ℕ, x = 15 ∧ ( ∀ y : ℕ, y ≥ 10 → f y ≥ f 15 ) :=
by
  sorry

end NUMINAMATH_GPT_minimize_average_comprehensive_cost_l648_64885


namespace NUMINAMATH_GPT_lock_combination_l648_64836

def valid_combination (T I D E b : ℕ) : Prop :=
  (T > 0) ∧ (I > 0) ∧ (D > 0) ∧ (E > 0) ∧
  (T ≠ I) ∧ (T ≠ D) ∧ (T ≠ E) ∧ (I ≠ D) ∧ (I ≠ E) ∧ (D ≠ E) ∧
  (T * b^3 + I * b^2 + D * b + E) + 
  (E * b^3 + D * b^2 + I * b + T) + 
  (T * b^3 + I * b^2 + D * b + E) = 
  (D * b^3 + I * b^2 + E * b + T)

theorem lock_combination : ∃ (T I D E b : ℕ), valid_combination T I D E b ∧ (T * 100 + I * 10 + D = 984) :=
sorry

end NUMINAMATH_GPT_lock_combination_l648_64836


namespace NUMINAMATH_GPT_coordinate_of_point_A_l648_64823

theorem coordinate_of_point_A (a b : ℝ) 
    (h1 : |b| = 3) 
    (h2 : |a| = 4) 
    (h3 : a > b) : 
    (a, b) = (4, 3) ∨ (a, b) = (4, -3) :=
by
    sorry

end NUMINAMATH_GPT_coordinate_of_point_A_l648_64823


namespace NUMINAMATH_GPT_radius_of_circle_centered_at_l648_64854

def center : ℝ × ℝ := (3, 4)

def intersects_axes_at_three_points (A : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - r = 0 ∨ A.1 + r = 0) ∧ (A.2 - r = 0 ∨ A.2 + r = 0)

theorem radius_of_circle_centered_at (A : ℝ × ℝ) : 
  (intersects_axes_at_three_points A 4) ∨ (intersects_axes_at_three_points A 5) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_centered_at_l648_64854


namespace NUMINAMATH_GPT_symmetric_circle_eq_l648_64860

theorem symmetric_circle_eq :
  ∀ (x y : ℝ),
  ((x + 2)^2 + y^2 = 5) →
  (x - y + 1 = 0) →
  (∃ (a b : ℝ), ((a + 1)^2 + (b + 1)^2 = 5)) := 
by
  intros x y h_circle h_line
  -- skip the proof
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l648_64860


namespace NUMINAMATH_GPT_amc_inequality_l648_64850

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem amc_inequality : (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_amc_inequality_l648_64850


namespace NUMINAMATH_GPT_eval_expression_l648_64847

theorem eval_expression : (3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_GPT_eval_expression_l648_64847


namespace NUMINAMATH_GPT_circumradius_of_sector_l648_64834

noncomputable def R_circumradius (θ : ℝ) (r : ℝ) := r / (2 * Real.sin (θ / 2))

theorem circumradius_of_sector (r : ℝ) (θ : ℝ) (hθ : θ = 120) (hr : r = 8) :
  R_circumradius θ r = (8 * Real.sqrt 3) / 3 :=
by
  rw [hθ, hr, R_circumradius]
  sorry

end NUMINAMATH_GPT_circumradius_of_sector_l648_64834


namespace NUMINAMATH_GPT_kates_discount_is_8_percent_l648_64803

-- Definitions based on the problem's conditions
def bobs_bill : ℤ := 30
def kates_bill : ℤ := 25
def total_paid : ℤ := 53
def total_without_discount : ℤ := bobs_bill + kates_bill
def discount_received : ℤ := total_without_discount - total_paid
def kates_discount_percentage : ℚ := (discount_received : ℚ) / kates_bill * 100

-- The theorem to prove
theorem kates_discount_is_8_percent : kates_discount_percentage = 8 :=
by
  sorry

end NUMINAMATH_GPT_kates_discount_is_8_percent_l648_64803


namespace NUMINAMATH_GPT_remove_brackets_l648_64886

-- Define the variables a, b, and c
variables (a b c : ℝ)

-- State the theorem
theorem remove_brackets (a b c : ℝ) : a - (b - c) = a - b + c := 
sorry

end NUMINAMATH_GPT_remove_brackets_l648_64886


namespace NUMINAMATH_GPT_cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l648_64868

noncomputable def cos_negative_pi_over_3 : Real :=
  Real.cos (-Real.pi / 3)

theorem cos_neg_pi_over_3_eq_one_half :
  cos_negative_pi_over_3 = 1 / 2 :=
  by
    sorry

noncomputable def solutions_sin_eq_sqrt3_over_2 (x : Real) : Prop :=
  Real.sin x = Real.sqrt 3 / 2 ∧ 0 ≤ x ∧ x < 2 * Real.pi

theorem sin_eq_sqrt3_over_2_solutions :
  {x : Real | solutions_sin_eq_sqrt3_over_2 x} = {Real.pi / 3, 2 * Real.pi / 3} :=
  by
    sorry

end NUMINAMATH_GPT_cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l648_64868


namespace NUMINAMATH_GPT_polynomial_expansion_l648_64819

theorem polynomial_expansion (a_0 a_1 a_2 a_3 a_4 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 = 5^4)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 = 1) :
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l648_64819


namespace NUMINAMATH_GPT_number_of_kids_stayed_home_is_668278_l648_64853

  def number_of_kids_who_stayed_home : Prop :=
    ∃ X : ℕ, X + 150780 = 819058 ∧ X = 668278

  theorem number_of_kids_stayed_home_is_668278 : number_of_kids_who_stayed_home :=
    sorry
  
end NUMINAMATH_GPT_number_of_kids_stayed_home_is_668278_l648_64853


namespace NUMINAMATH_GPT_hikers_count_l648_64844

theorem hikers_count (B H K : ℕ) (h1 : H = B + 178) (h2 : K = B / 2) (h3 : H + B + K = 920) : H = 474 :=
by
  sorry

end NUMINAMATH_GPT_hikers_count_l648_64844


namespace NUMINAMATH_GPT_min_value_sum_inverse_sq_l648_64806

theorem min_value_sum_inverse_sq (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 1) : 
  (39 + 1/x + 4/y + 9/z) ≥ 25 :=
by
    sorry

end NUMINAMATH_GPT_min_value_sum_inverse_sq_l648_64806


namespace NUMINAMATH_GPT_min_races_required_to_determine_top_3_horses_l648_64859

def maxHorsesPerRace := 6
def totalHorses := 30
def possibleConditions := "track conditions and layouts change for each race"

noncomputable def minRacesToDetermineTop3 : Nat :=
  7

-- Problem Statement: Prove that given the conditions on track and race layout changes,
-- the minimum number of races needed to confidently determine the top 3 fastest horses is 7.
theorem min_races_required_to_determine_top_3_horses 
  (maxHorsesPerRace : Nat := 6) 
  (totalHorses : Nat := 30)
  (possibleConditions : String := "track conditions and layouts change for each race") :
  minRacesToDetermineTop3 = 7 :=
  sorry

end NUMINAMATH_GPT_min_races_required_to_determine_top_3_horses_l648_64859


namespace NUMINAMATH_GPT_min_value_f_l648_64856

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem min_value_f {x0 : ℝ} (hx0 : 0 < x0) (hx0_min : ∀ x > 0, f x ≥ f x0) :
  f x0 = x0 + 1 ∧ f x0 < 3 :=
by sorry

end NUMINAMATH_GPT_min_value_f_l648_64856
