import Mathlib

namespace NUMINAMATH_GPT_min_value_condition_l1129_112959

theorem min_value_condition {a b c d e f g h : ℝ} (h1 : a * b * c * d = 16) (h2 : e * f * g * h = 25) :
  (a^2 * e^2 + b^2 * f^2 + c^2 * g^2 + d^2 * h^2) ≥ 160 :=
  sorry

end NUMINAMATH_GPT_min_value_condition_l1129_112959


namespace NUMINAMATH_GPT_diaz_age_twenty_years_later_l1129_112915

theorem diaz_age_twenty_years_later (D S : ℕ) (h₁ : 10 * D - 40 = 10 * S + 20) (h₂ : S = 30) : D + 20 = 56 :=
sorry

end NUMINAMATH_GPT_diaz_age_twenty_years_later_l1129_112915


namespace NUMINAMATH_GPT_length_of_platform_l1129_112928

theorem length_of_platform 
  (speed_kmph : ℕ)
  (time_cross_platform : ℕ)
  (time_cross_man : ℕ)
  (speed_mps : ℕ)
  (length_of_train : ℕ)
  (distance_platform : ℕ)
  (length_of_platform : ℕ) :
  speed_kmph = 72 →
  time_cross_platform = 30 →
  time_cross_man = 16 →
  speed_mps = speed_kmph * 1000 / 3600 →
  length_of_train = speed_mps * time_cross_man →
  distance_platform = speed_mps * time_cross_platform →
  length_of_platform = distance_platform - length_of_train →
  length_of_platform = 280 := by
  sorry

end NUMINAMATH_GPT_length_of_platform_l1129_112928


namespace NUMINAMATH_GPT_hexagon_area_l1129_112953

-- Definition of an equilateral triangle with a given perimeter.
def is_equilateral_triangle (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] :=
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = 42 ∧ ∀ (angle : ℝ), angle = 60

-- Statement of the problem
theorem hexagon_area (P Q R P' Q' R' : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace P'] [MetricSpace Q'] [MetricSpace R']
  (h1 : is_equilateral_triangle P Q R) :
  ∃ (area : ℝ), area = 49 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_hexagon_area_l1129_112953


namespace NUMINAMATH_GPT_equidistant_point_on_y_axis_l1129_112911

theorem equidistant_point_on_y_axis :
  ∃ (y : ℝ), 0 < y ∧ 
  (dist (0, y) (-3, 0) = dist (0, y) (-2, 5)) ∧ 
  y = 2 :=
by
  sorry

end NUMINAMATH_GPT_equidistant_point_on_y_axis_l1129_112911


namespace NUMINAMATH_GPT_train_speed_l1129_112952

theorem train_speed :
  ∀ (length : ℝ) (time : ℝ),
    length = 135 ∧ time = 3.4711508793582233 →
    (length / time) * 3.6 = 140.0004 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1129_112952


namespace NUMINAMATH_GPT_christina_payment_l1129_112974

theorem christina_payment :
  let pay_flowers_per_flower := (8 : ℚ) / 3
  let pay_lawn_per_meter := (5 : ℚ) / 2
  let num_flowers := (9 : ℚ) / 4
  let area_lawn := (7 : ℚ) / 3
  let total_payment := pay_flowers_per_flower * num_flowers + pay_lawn_per_meter * area_lawn
  total_payment = 71 / 6 :=
by
  sorry

end NUMINAMATH_GPT_christina_payment_l1129_112974


namespace NUMINAMATH_GPT_find_principal_amount_l1129_112986

-- Given conditions
def SI : ℝ := 4016.25
def R : ℝ := 0.14
def T : ℕ := 5

-- Question: What is the principal amount P?
theorem find_principal_amount : (SI / (R * T) = 5737.5) :=
sorry

end NUMINAMATH_GPT_find_principal_amount_l1129_112986


namespace NUMINAMATH_GPT_largest_and_smallest_correct_l1129_112923

noncomputable def largest_and_smallest (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) : ℝ × ℝ :=
  if hx_y : x * y > 0 then
    if hx_y_sq : x * y * y > x then
      (x * y, x)
    else
      sorry
  else
    sorry

theorem largest_and_smallest_correct {x y : ℝ} (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  largest_and_smallest x y hx hy = (x * y, x) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_and_smallest_correct_l1129_112923


namespace NUMINAMATH_GPT_houses_with_pools_l1129_112977

theorem houses_with_pools (total G overlap N P : ℕ) 
  (h1 : total = 70) 
  (h2 : G = 50) 
  (h3 : overlap = 35) 
  (h4 : N = 15) 
  (h_eq : total = G + P - overlap + N) : 
  P = 40 := by
  sorry

end NUMINAMATH_GPT_houses_with_pools_l1129_112977


namespace NUMINAMATH_GPT_lcm_18_24_l1129_112992

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end NUMINAMATH_GPT_lcm_18_24_l1129_112992


namespace NUMINAMATH_GPT_monica_tiles_l1129_112964

-- Define the dimensions of the living room
def living_room_length : ℕ := 20
def living_room_width : ℕ := 15

-- Define the size of the border tiles and inner tiles
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Prove the number of tiles used is 44
theorem monica_tiles (border_tile_count inner_tile_count total_tiles : ℕ)
  (h_border : border_tile_count = ((2 * ((living_room_length - 4) / border_tile_size) + 2 * ((living_room_width - 4) / border_tile_size) - 4)))
  (h_inner : inner_tile_count = (176 / (inner_tile_size * inner_tile_size)))
  (h_total : total_tiles = border_tile_count + inner_tile_count) :
  total_tiles = 44 :=
by
  sorry

end NUMINAMATH_GPT_monica_tiles_l1129_112964


namespace NUMINAMATH_GPT_chad_bbq_people_l1129_112969

theorem chad_bbq_people (ice_cost_per_pack : ℝ) (packs_included : ℕ) (total_money_spent : ℝ) (pounds_needed_per_person : ℝ) :
  total_money_spent = 9 → 
  ice_cost_per_pack = 3 → 
  packs_included = 10 → 
  pounds_needed_per_person = 2 → 
  ∃ (people : ℕ), people = 15 :=
by intros; sorry

end NUMINAMATH_GPT_chad_bbq_people_l1129_112969


namespace NUMINAMATH_GPT_range_f_l1129_112990

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4 

theorem range_f : Set.Icc (0 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end NUMINAMATH_GPT_range_f_l1129_112990


namespace NUMINAMATH_GPT_ab_inequality_smaller_than_fourth_sum_l1129_112929

theorem ab_inequality_smaller_than_fourth_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := 
by
  sorry

end NUMINAMATH_GPT_ab_inequality_smaller_than_fourth_sum_l1129_112929


namespace NUMINAMATH_GPT_inequality_abs_l1129_112972

noncomputable def f (x : ℝ) : ℝ := abs (x - 1/2) + abs (x + 1/2)

def M : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem inequality_abs (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + a * b| := 
by
  sorry

end NUMINAMATH_GPT_inequality_abs_l1129_112972


namespace NUMINAMATH_GPT_ant_climbing_floors_l1129_112957

theorem ant_climbing_floors (time_per_floor : ℕ) (total_time : ℕ) (floors_climbed : ℕ) :
  time_per_floor = 15 →
  total_time = 105 →
  floors_climbed = total_time / time_per_floor + 1 →
  floors_climbed = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ant_climbing_floors_l1129_112957


namespace NUMINAMATH_GPT_min_value_of_quadratic_fun_min_value_is_reached_l1129_112973

theorem min_value_of_quadratic_fun (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1) :
  (3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 ≥ (15 / 782)) :=
sorry

theorem min_value_is_reached (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1)
  (h2 : 3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 = (15 / 782)) :
  true :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_fun_min_value_is_reached_l1129_112973


namespace NUMINAMATH_GPT_binom_identity1_binom_identity2_l1129_112900

section Combinatorics

variable (n k m : ℕ)

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

-- Prove the identity: C(n, k) + C(n, k-1) = C(n+1, k)
theorem binom_identity1 : binomial n k + binomial n (k-1) = binomial (n+1) k :=
  sorry

-- Using the identity, prove: C(n, m) + C(n-1, m) + ... + C(n-10, m) = C(n+1, m+1) - C(n-10, m+1)
theorem binom_identity2 :
  (binomial n m + binomial (n-1) m + binomial (n-2) m + binomial (n-3) m
   + binomial (n-4) m + binomial (n-5) m + binomial (n-6) m + binomial (n-7) m
   + binomial (n-8) m + binomial (n-9) m + binomial (n-10) m)
   = binomial (n+1) (m+1) - binomial (n-10) (m+1) :=
  sorry

end Combinatorics

end NUMINAMATH_GPT_binom_identity1_binom_identity2_l1129_112900


namespace NUMINAMATH_GPT_acute_triangle_tangent_difference_range_l1129_112920

theorem acute_triangle_tangent_difference_range {A B C a b c : ℝ} 
    (h1 : a^2 + b^2 > c^2) (h2 : b^2 + c^2 > a^2) (h3 : c^2 + a^2 > b^2)
    (hb2_minus_ha2_eq_ac : b^2 - a^2 = a * c) :
    1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < (2 * Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_tangent_difference_range_l1129_112920


namespace NUMINAMATH_GPT_minimize_b_plus_c_l1129_112941

theorem minimize_b_plus_c (a b c : ℝ) (h1 : 0 < a)
  (h2 : ∀ x, (y : ℝ) = a * x^2 + b * x + c)
  (h3 : ∀ x, (yr : ℝ) = a * (x + 2)^2 + (a - 1)^2) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimize_b_plus_c_l1129_112941


namespace NUMINAMATH_GPT_jane_exercises_40_hours_l1129_112988

-- Define the conditions
def hours_per_day : ℝ := 1
def days_per_week : ℝ := 5
def weeks : ℝ := 8

-- Define total_hours using the conditions
def total_hours : ℝ := (hours_per_day * days_per_week) * weeks

-- The theorem stating the result
theorem jane_exercises_40_hours :
  total_hours = 40 := by
  sorry

end NUMINAMATH_GPT_jane_exercises_40_hours_l1129_112988


namespace NUMINAMATH_GPT_nails_per_plank_l1129_112907

theorem nails_per_plank {total_nails planks : ℕ} (h1 : total_nails = 4) (h2 : planks = 2) :
  total_nails / planks = 2 := by
  sorry

end NUMINAMATH_GPT_nails_per_plank_l1129_112907


namespace NUMINAMATH_GPT_find_r_and_s_l1129_112910

theorem find_r_and_s (r s : ℝ) :
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + 10 * x = m * (x - 10) + 5) ↔ r < m ∧ m < s) →
  r + s = 60 :=
sorry

end NUMINAMATH_GPT_find_r_and_s_l1129_112910


namespace NUMINAMATH_GPT_value_of_abc_l1129_112958

theorem value_of_abc : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (ab + c + 10 = 51) ∧ (bc + a + 10 = 51) ∧ (ac + b + 10 = 51) ∧ (a + b + c = 41) :=
by
  sorry

end NUMINAMATH_GPT_value_of_abc_l1129_112958


namespace NUMINAMATH_GPT_workshop_participants_problem_l1129_112978

variable (WorkshopSize : ℕ) 
variable (LeftHanded : ℕ) 
variable (RockMusicLovers : ℕ) 
variable (RightHandedDislikeRock : ℕ) 
variable (Under25 : ℕ)
variable (RightHandedUnder25RockMusicLovers : ℕ)
variable (y : ℕ)

theorem workshop_participants_problem
  (h1 : WorkshopSize = 30)
  (h2 : LeftHanded = 12)
  (h3 : RockMusicLovers = 18)
  (h4 : RightHandedDislikeRock = 5)
  (h5 : Under25 = 9)
  (h6 : RightHandedUnder25RockMusicLovers = 3)
  (h7 : WorkshopSize = LeftHanded + (WorkshopSize - LeftHanded))
  (h8 : WorkshopSize - LeftHanded = RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + (WorkshopSize - LeftHanded - RightHandedDislikeRock - RightHandedUnder25RockMusicLovers - y))
  (h9 : WorkshopSize - (RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + Under25 - y - (RockMusicLovers - y)) - (LeftHanded - y) = WorkshopSize) :
  y = 5 := by
  sorry

end NUMINAMATH_GPT_workshop_participants_problem_l1129_112978


namespace NUMINAMATH_GPT_west_movement_80_eq_neg_80_l1129_112918

-- Define conditions
def east_movement (distance : ℤ) : ℤ := distance

-- Prove that moving westward is represented correctly
theorem west_movement_80_eq_neg_80 : east_movement (-80) = -80 :=
by
  -- Theorem proof goes here
  sorry

end NUMINAMATH_GPT_west_movement_80_eq_neg_80_l1129_112918


namespace NUMINAMATH_GPT_common_sum_l1129_112998

theorem common_sum (a l : ℤ) (n r c : ℕ) (S x : ℤ) 
  (h_a : a = -18) 
  (h_l : l = 30) 
  (h_n : n = 49) 
  (h_S : S = (n * (a + l)) / 2) 
  (h_r : r = 7) 
  (h_c : c = 7) 
  (h_sum_eq : r * x = S) :
  x = 42 := 
sorry

end NUMINAMATH_GPT_common_sum_l1129_112998


namespace NUMINAMATH_GPT_cuboid_length_l1129_112982

theorem cuboid_length (SA w h : ℕ) (h_SA : SA = 700) (h_w : w = 14) (h_h : h = 7) 
  (h_surface_area : SA = 2 * l * w + 2 * l * h + 2 * w * h) : l = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cuboid_length_l1129_112982


namespace NUMINAMATH_GPT_find_k_l1129_112948

-- Define the function y = kx
def linear_function (k x : ℝ) : ℝ := k * x

-- Define the point P(3,1)
def P : ℝ × ℝ := (3, 1)

theorem find_k (k : ℝ) (h : linear_function k 3 = 1) : k = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1129_112948


namespace NUMINAMATH_GPT_exists_colored_subset_l1129_112971

theorem exists_colored_subset (n : ℕ) (h_positive : n > 0) (colors : ℕ → ℕ) (h_colors : ∀ a b : ℕ, a < b → a + b ≤ n → 
  (colors a = colors b ∨ colors b = colors (a + b) ∨ colors a = colors (a + b))) :
  ∃ c, ∃ s : Finset ℕ, s.card ≥ (2 * n / 5) ∧ ∀ x ∈ s, colors x = c :=
sorry

end NUMINAMATH_GPT_exists_colored_subset_l1129_112971


namespace NUMINAMATH_GPT_max_a_monotonic_f_l1129_112962

theorem max_a_monotonic_f {a : ℝ} (h1 : 0 < a)
  (h2 : ∀ x ≥ 1, 0 ≤ (3 * x^2 - a)) : a ≤ 3 := by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_max_a_monotonic_f_l1129_112962


namespace NUMINAMATH_GPT_total_children_on_playground_l1129_112937

theorem total_children_on_playground (boys girls : ℕ) (hb : boys = 27) (hg : girls = 35) : boys + girls = 62 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_children_on_playground_l1129_112937


namespace NUMINAMATH_GPT_goods_train_length_is_420_l1129_112906

/-- The man's train speed in km/h. -/
def mans_train_speed_kmph : ℝ := 64

/-- The goods train speed in km/h. -/
def goods_train_speed_kmph : ℝ := 20

/-- The time taken for the trains to pass each other in seconds. -/
def passing_time_s : ℝ := 18

/-- The relative speed of two trains traveling in opposite directions in m/s. -/
noncomputable def relative_speed_mps : ℝ := 
  (mans_train_speed_kmph + goods_train_speed_kmph) * 1000 / 3600

/-- The length of the goods train in meters. -/
noncomputable def goods_train_length_m : ℝ := relative_speed_mps * passing_time_s

/-- The theorem stating the length of the goods train is 420 meters. -/
theorem goods_train_length_is_420 :
  goods_train_length_m = 420 :=
sorry

end NUMINAMATH_GPT_goods_train_length_is_420_l1129_112906


namespace NUMINAMATH_GPT_every_positive_integer_displayable_l1129_112951

-- Definitions based on the conditions of the problem
def flip_switch_up (n : ℕ) : ℕ := n + 1
def flip_switch_down (n : ℕ) : ℕ := n - 1
def press_red_button (n : ℕ) : ℕ := n * 3
def press_yellow_button (n : ℕ) : ℕ := if n % 3 = 0 then n / 3 else n
def press_green_button (n : ℕ) : ℕ := n * 5
def press_blue_button (n : ℕ) : ℕ := if n % 5 = 0 then n / 5 else n

-- Prove that every positive integer can appear on the calculator display
theorem every_positive_integer_displayable : ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = n ∧
    (m = flip_switch_up m ∨ m = flip_switch_down m ∨ 
     m = press_red_button m ∨ m = press_yellow_button m ∨ 
     m = press_green_button m ∨ m = press_blue_button m) := 
sorry

end NUMINAMATH_GPT_every_positive_integer_displayable_l1129_112951


namespace NUMINAMATH_GPT_value_for_real_value_for_pure_imaginary_l1129_112933

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def value_conditions (k : ℝ) : ℂ := ⟨k^2 - 3*k - 4, k^2 - 5*k - 6⟩

theorem value_for_real (k : ℝ) : is_real (value_conditions k) ↔ (k = 6 ∨ k = -1) :=
by
  sorry

theorem value_for_pure_imaginary (k : ℝ) : is_pure_imaginary (value_conditions k) ↔ (k = 4) :=
by
  sorry

end NUMINAMATH_GPT_value_for_real_value_for_pure_imaginary_l1129_112933


namespace NUMINAMATH_GPT_Louisa_travel_distance_l1129_112914

variables (D : ℕ)

theorem Louisa_travel_distance : 
  (200 / 50 + 3 = D / 50) → D = 350 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Louisa_travel_distance_l1129_112914


namespace NUMINAMATH_GPT_cubes_sum_eq_zero_l1129_112966

theorem cubes_sum_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_cubes_sum_eq_zero_l1129_112966


namespace NUMINAMATH_GPT_power_identity_l1129_112961

theorem power_identity (x : ℕ) (h : 2^x = 16) : 2^(x + 3) = 128 := 
sorry

end NUMINAMATH_GPT_power_identity_l1129_112961


namespace NUMINAMATH_GPT_part1_k_real_part2_find_k_l1129_112913

-- Part 1: Discriminant condition
theorem part1_k_real (k : ℝ) (h : x^2 + (2*k - 1)*x + k^2 - 1 = 0) : k ≤ 5 / 4 :=
by
  sorry

-- Part 2: Given additional conditions, find k
theorem part2_find_k (x1 x2 k : ℝ) (h_eq : x^2 + (2 * k - 1) * x + k^2 - 1 = 0)
  (h1 : x1 + x2 = 1 - 2 * k) (h2 : x1 * x2 = k^2 - 1) (h3 : x1^2 + x2^2 = 16 + x1 * x2) : k = -2 :=
by
  sorry

end NUMINAMATH_GPT_part1_k_real_part2_find_k_l1129_112913


namespace NUMINAMATH_GPT_remainder_of_7_pow_145_mod_9_l1129_112912

theorem remainder_of_7_pow_145_mod_9 : (7 ^ 145) % 9 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_7_pow_145_mod_9_l1129_112912


namespace NUMINAMATH_GPT_temperature_increase_per_century_l1129_112976

def total_temperature_change_over_1600_years : ℕ := 64
def years_in_a_century : ℕ := 100
def years_overall : ℕ := 1600

theorem temperature_increase_per_century :
  total_temperature_change_over_1600_years / (years_overall / years_in_a_century) = 4 := by
  sorry

end NUMINAMATH_GPT_temperature_increase_per_century_l1129_112976


namespace NUMINAMATH_GPT_sandy_walks_before_meet_l1129_112996

/-
Sandy leaves her home and walks toward Ed's house.
Two hours later, Ed leaves his home and walks toward Sandy's house.
The distance between their homes is 52 kilometers.
Sandy's walking speed is 6 km/h.
Ed's walking speed is 4 km/h.
Prove that Sandy will walk 36 kilometers before she meets Ed.
-/

theorem sandy_walks_before_meet
    (distance_between_homes : ℕ)
    (sandy_speed ed_speed : ℕ)
    (sandy_start_time ed_start_time : ℕ)
    (time_to_meet : ℕ) :
  distance_between_homes = 52 →
  sandy_speed = 6 →
  ed_speed = 4 →
  sandy_start_time = 2 →
  ed_start_time = 0 →
  time_to_meet = 4 →
  (sandy_start_time * sandy_speed + time_to_meet * sandy_speed) = 36 := 
by
  sorry

end NUMINAMATH_GPT_sandy_walks_before_meet_l1129_112996


namespace NUMINAMATH_GPT_soccer_game_points_ratio_l1129_112927

theorem soccer_game_points_ratio :
  ∃ B1 A1 A2 B2 : ℕ,
    A1 = 8 ∧
    B2 = 8 ∧
    A2 = 6 ∧
    (A1 + B1 + A2 + B2 = 26) ∧
    (B1 / A1 = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_soccer_game_points_ratio_l1129_112927


namespace NUMINAMATH_GPT_slices_with_both_toppings_l1129_112968

theorem slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices : ℕ)
    (all_have_topping : total_slices = 24)
    (pepperoni_cond: pepperoni_slices = 14)
    (mushroom_cond: mushroom_slices = 16)
    (at_least_one_topping : total_slices = pepperoni_slices + mushroom_slices - slices_with_both):
    slices_with_both = 6 := by
  sorry

end NUMINAMATH_GPT_slices_with_both_toppings_l1129_112968


namespace NUMINAMATH_GPT_least_m_plus_n_l1129_112995

theorem least_m_plus_n (m n : ℕ) (hmn : Nat.gcd (m + n) 330 = 1) (hm_multiple : m^m % n^n = 0) (hm_not_multiple : ¬ (m % n = 0)) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  m + n = 119 :=
sorry

end NUMINAMATH_GPT_least_m_plus_n_l1129_112995


namespace NUMINAMATH_GPT_ben_weekly_eggs_l1129_112991

-- Definitions for the conditions
def weekly_saly_eggs : ℕ := 10
def weekly_ben_eggs (B : ℕ) : ℕ := B
def weekly_ked_eggs (B : ℕ) : ℕ := B / 2

def weekly_production (B : ℕ) : ℕ :=
  weekly_saly_eggs + weekly_ben_eggs B + weekly_ked_eggs B

def monthly_production (B : ℕ) : ℕ := 4 * weekly_production B

-- Theorem for the proof
theorem ben_weekly_eggs (B : ℕ) (h : monthly_production B = 124) : B = 14 :=
sorry

end NUMINAMATH_GPT_ben_weekly_eggs_l1129_112991


namespace NUMINAMATH_GPT_blue_socks_count_l1129_112919

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end NUMINAMATH_GPT_blue_socks_count_l1129_112919


namespace NUMINAMATH_GPT_graph_forms_l1129_112987

theorem graph_forms (x y : ℝ) :
  x^3 * (2 * x + 2 * y + 3) = y^3 * (2 * x + 2 * y + 3) →
  (∀ x y : ℝ, y ≠ x → y = -x - 3 / 2) ∨ (y = x) :=
sorry

end NUMINAMATH_GPT_graph_forms_l1129_112987


namespace NUMINAMATH_GPT_proof_of_problem_l1129_112994

def problem_statement : Prop :=
  2 * Real.cos (Real.pi / 4) + abs (Real.sqrt 2 - 3)
  - (1 / 3) ^ (-2 : ℤ) + (2021 - Real.pi) ^ 0 = -5

theorem proof_of_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_proof_of_problem_l1129_112994


namespace NUMINAMATH_GPT_largest_base4_to_base10_l1129_112981

theorem largest_base4_to_base10 : 
  (3 * 4^2 + 3 * 4^1 + 3 * 4^0) = 63 := 
by
  -- sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_largest_base4_to_base10_l1129_112981


namespace NUMINAMATH_GPT_graph_is_line_l1129_112950

theorem graph_is_line : {p : ℝ × ℝ | (p.1 - p.2)^2 = 2 * (p.1^2 + p.2^2)} = {p : ℝ × ℝ | p.2 = -p.1} :=
by 
  sorry

end NUMINAMATH_GPT_graph_is_line_l1129_112950


namespace NUMINAMATH_GPT_pizza_problem_l1129_112979

theorem pizza_problem :
  ∃ (x : ℕ), x = 20 ∧ (3 * x ^ 2 = 3 * 14 ^ 2 * 2 + 49) :=
by
  let small_pizza_side := 14
  let large_pizza_cost := 20
  let pool_cost := 60
  let individually_cost := 30
  have total_individual_area := 2 * 3 * (small_pizza_side ^ 2)
  have extra_area := 49
  sorry

end NUMINAMATH_GPT_pizza_problem_l1129_112979


namespace NUMINAMATH_GPT_range_of_real_roots_l1129_112901

theorem range_of_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) ↔
  a >= -1 ∨ a <= -3/2 :=
  sorry

end NUMINAMATH_GPT_range_of_real_roots_l1129_112901


namespace NUMINAMATH_GPT_sequence_terms_are_integers_l1129_112944

theorem sequence_terms_are_integers (a : ℕ → ℕ)
  (h0 : a 0 = 1) 
  (h1 : a 1 = 2) 
  (h_recurrence : ∀ n : ℕ, (n + 3) * a (n + 2) = (6 * n + 9) * a (n + 1) - n * a n) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k := 
by
  -- Initialize the proof
  sorry

end NUMINAMATH_GPT_sequence_terms_are_integers_l1129_112944


namespace NUMINAMATH_GPT_final_amount_after_two_years_l1129_112926

open BigOperators

/-- Given an initial amount A0 and a percentage increase p, calculate the amount after n years -/
def compound_increase (A0 : ℝ) (p : ℝ) (n : ℕ) : ℝ :=
  (A0 * (1 + p)^n)

theorem final_amount_after_two_years (A0 : ℝ) (p : ℝ) (A2 : ℝ) :
  A0 = 1600 ∧ p = 1 / 8 ∧ compound_increase 1600 (1 / 8) 2 = 2025 :=
  sorry

end NUMINAMATH_GPT_final_amount_after_two_years_l1129_112926


namespace NUMINAMATH_GPT_fractional_eq_solution_l1129_112930

theorem fractional_eq_solution (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) →
  k ≠ -3 ∧ k ≠ 5 :=
by 
  sorry

end NUMINAMATH_GPT_fractional_eq_solution_l1129_112930


namespace NUMINAMATH_GPT_triangle_inequality_l1129_112934

variables {A B C P D E F : Type} -- Variables representing points in the plane.
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (PD PE PF PA PB PC : ℝ) -- Distances corresponding to the points.

-- Condition stating P lies inside or on the boundary of triangle ABC
axiom P_in_triangle_ABC : ∀ (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P], 
  (PD > 0 ∧ PE > 0 ∧ PF > 0 ∧ PA > 0 ∧ PB > 0 ∧ PC > 0)

-- Objective statement to prove
theorem triangle_inequality (PD PE PF PA PB PC : ℝ) 
  (h1 : PA ≥ 0) 
  (h2 : PB ≥ 0) 
  (h3 : PC ≥ 0) 
  (h4 : PD ≥ 0) 
  (h5 : PE ≥ 0) 
  (h6 : PF ≥ 0) :
  PA + PB + PC ≥ 2 * (PD + PE + PF) := 
sorry -- Proof to be provided later.

end NUMINAMATH_GPT_triangle_inequality_l1129_112934


namespace NUMINAMATH_GPT_relationship_between_M_and_N_l1129_112975

theorem relationship_between_M_and_N (a b : ℝ) (M N : ℝ) 
  (hM : M = a^2 - a * b) 
  (hN : N = a * b - b^2) : M ≥ N :=
by sorry

end NUMINAMATH_GPT_relationship_between_M_and_N_l1129_112975


namespace NUMINAMATH_GPT_celsius_equals_fahrenheit_l1129_112916

-- Define the temperature scales.
def celsius_to_fahrenheit (T_C : ℝ) : ℝ := 1.8 * T_C + 32

-- The Lean statement for the problem.
theorem celsius_equals_fahrenheit : ∃ (T : ℝ), T = celsius_to_fahrenheit T ↔ T = -40 :=
by
  sorry -- Proof is not required, just the statement.

end NUMINAMATH_GPT_celsius_equals_fahrenheit_l1129_112916


namespace NUMINAMATH_GPT_potato_slice_length_l1129_112947

theorem potato_slice_length (x : ℕ) (h1 : 600 = x + (x + 50)) : x + 50 = 325 :=
by
  sorry

end NUMINAMATH_GPT_potato_slice_length_l1129_112947


namespace NUMINAMATH_GPT_second_negative_integer_l1129_112956

theorem second_negative_integer (n : ℤ) (h : -11 * n + 5 = 93) : n = -8 :=
by
  sorry

end NUMINAMATH_GPT_second_negative_integer_l1129_112956


namespace NUMINAMATH_GPT_number_of_ping_pong_balls_l1129_112942

def sales_tax_rate : ℝ := 0.16

def total_cost_with_tax (B x : ℝ) : ℝ := B * x * (1 + sales_tax_rate)

def total_cost_without_tax (B x : ℝ) : ℝ := (B + 3) * x

theorem number_of_ping_pong_balls
  (B x : ℝ) (h₁ : total_cost_with_tax B x = total_cost_without_tax B x) :
  B = 18.75 := 
sorry

end NUMINAMATH_GPT_number_of_ping_pong_balls_l1129_112942


namespace NUMINAMATH_GPT_commercials_per_hour_l1129_112970

theorem commercials_per_hour (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : ∃ x : ℝ, x = (1 - p) * 60 := 
sorry

end NUMINAMATH_GPT_commercials_per_hour_l1129_112970


namespace NUMINAMATH_GPT_triangle_area_ratio_l1129_112965

theorem triangle_area_ratio (a n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  let area_A := (a^2 : ℝ) / (4 * n^2)
  let area_B := (a^2 : ℝ) / (4 * m^2)
  (area_A / area_B) = (m^2 : ℝ) / (n^2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1129_112965


namespace NUMINAMATH_GPT_vertex_angle_isosceles_l1129_112946

theorem vertex_angle_isosceles (a b c : ℝ)
  (isosceles: (a = b ∨ b = c ∨ c = a))
  (angle_sum : a + b + c = 180)
  (one_angle_is_70 : a = 70 ∨ b = 70 ∨ c = 70) :
  a = 40 ∨ a = 70 ∨ b = 40 ∨ b = 70 ∨ c = 40 ∨ c = 70 :=
by sorry

end NUMINAMATH_GPT_vertex_angle_isosceles_l1129_112946


namespace NUMINAMATH_GPT_complement_of_S_in_U_l1129_112902

variable (U : Set ℕ)
variable (S : Set ℕ)

theorem complement_of_S_in_U (hU : U = {1, 2, 3, 4}) (hS : S = {1, 3}) : U \ S = {2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_of_S_in_U_l1129_112902


namespace NUMINAMATH_GPT_james_total_cost_l1129_112905

def courseCost (units: Nat) (cost_per_unit: Nat) : Nat :=
  units * cost_per_unit

def totalCostForFall : Nat :=
  courseCost 12 60 + courseCost 8 45

def totalCostForSpring : Nat :=
  let science_cost := courseCost 10 60
  let science_scholarship := science_cost / 2
  let humanities_cost := courseCost 10 45
  (science_cost - science_scholarship) + humanities_cost

def totalCostForSummer : Nat :=
  courseCost 6 80 + courseCost 4 55

def totalCostForWinter : Nat :=
  let science_cost := courseCost 6 80
  let science_scholarship := 3 * science_cost / 4
  let humanities_cost := courseCost 4 55
  (science_cost - science_scholarship) + humanities_cost

def totalAmountSpent : Nat :=
  totalCostForFall + totalCostForSpring + totalCostForSummer + totalCostForWinter

theorem james_total_cost: totalAmountSpent = 2870 :=
  by sorry

end NUMINAMATH_GPT_james_total_cost_l1129_112905


namespace NUMINAMATH_GPT_solution_set_f_cos_x_l1129_112967

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 3 then -(x-2)^2 + 1
else if x = 0 then 0
else if -3 < x ∧ x < 0 then (x+2)^2 - 1
else 0 -- Defined as 0 outside the given interval for simplicity

theorem solution_set_f_cos_x :
  {x : ℝ | f x * Real.cos x < 0} = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)} :=
sorry

end NUMINAMATH_GPT_solution_set_f_cos_x_l1129_112967


namespace NUMINAMATH_GPT_proof_triangle_inequality_l1129_112949

noncomputable def proof_statement (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : Prop :=
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c)

-- Proof statement without the proof
theorem proof_triangle_inequality (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : 
  proof_statement a b c h :=
sorry

end NUMINAMATH_GPT_proof_triangle_inequality_l1129_112949


namespace NUMINAMATH_GPT_distance_to_valley_l1129_112908

theorem distance_to_valley (car_speed_kph : ℕ) (time_seconds : ℕ) (sound_speed_mps : ℕ) 
  (car_speed_mps : ℕ) (distance_by_car : ℕ) (distance_by_sound : ℕ) 
  (total_distance_equation : 2 * x + distance_by_car = distance_by_sound) : x = 640 :=
by
  have car_speed_kph := 72
  have time_seconds := 4
  have sound_speed_mps := 340
  have car_speed_mps := car_speed_kph * 1000 / 3600
  have distance_by_car := time_seconds * car_speed_mps
  have distance_by_sound := time_seconds * sound_speed_mps
  have total_distance_equation := (2 * x + distance_by_car = distance_by_sound)
  exact sorry

end NUMINAMATH_GPT_distance_to_valley_l1129_112908


namespace NUMINAMATH_GPT_hours_sunday_correct_l1129_112963

-- Definitions of given conditions
def hours_saturday : ℕ := 6
def total_hours : ℕ := 9

-- The question translated to a proof problem
theorem hours_sunday_correct : total_hours - hours_saturday = 3 := 
by
  -- The proof is skipped and replaced by sorry
  sorry

end NUMINAMATH_GPT_hours_sunday_correct_l1129_112963


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l1129_112985

theorem sum_of_transformed_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l1129_112985


namespace NUMINAMATH_GPT_number_of_real_values_of_p_l1129_112936

theorem number_of_real_values_of_p :
  ∃ p_values : Finset ℝ, (∀ p ∈ p_values, ∀ x, x^2 - 2 * p * x + 3 * p = 0 → (x = p)) ∧ Finset.card p_values = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_values_of_p_l1129_112936


namespace NUMINAMATH_GPT_find_x_l1129_112931

theorem find_x (x : ℝ) (h : x + 2.75 + 0.158 = 2.911) : x = 0.003 :=
sorry

end NUMINAMATH_GPT_find_x_l1129_112931


namespace NUMINAMATH_GPT_largest_integer_divisor_l1129_112999

theorem largest_integer_divisor (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end NUMINAMATH_GPT_largest_integer_divisor_l1129_112999


namespace NUMINAMATH_GPT_trey_total_time_is_two_hours_l1129_112993

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end NUMINAMATH_GPT_trey_total_time_is_two_hours_l1129_112993


namespace NUMINAMATH_GPT_circle_count_2012_l1129_112980

/-
The pattern is defined as follows: 
○●, ○○●, ○○○●, ○○○○●, …
We need to prove that the number of ● in the first 2012 circles is 61.
-/

-- Define the pattern sequence
def circlePattern (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Total number of circles in the first k segments:
def totalCircles (k : ℕ) : ℕ :=
  k * (k + 1) / 2 + k

theorem circle_count_2012 : 
  ∃ (n : ℕ), totalCircles n ≤ 2012 ∧ 2012 < totalCircles (n + 1) ∧ n = 61 :=
by
  sorry

end NUMINAMATH_GPT_circle_count_2012_l1129_112980


namespace NUMINAMATH_GPT_apples_first_year_l1129_112924

theorem apples_first_year (A : ℕ) 
  (second_year_prod : ℕ := 2 * A + 8)
  (third_year_prod : ℕ := 3 * (2 * A + 8) / 4)
  (total_prod : ℕ := A + second_year_prod + third_year_prod) :
  total_prod = 194 → A = 40 :=
by
  sorry

end NUMINAMATH_GPT_apples_first_year_l1129_112924


namespace NUMINAMATH_GPT_inequality_example_l1129_112954

theorem inequality_example (a b c : ℝ) (hac : a ≠ 0) (hbc : b ≠ 0) (hcc : c ≠ 0) :
  (a^4) / (4 * a^4 + b^4 + c^4) + (b^4) / (a^4 + 4 * b^4 + c^4) + (c^4) / (a^4 + b^4 + 4 * c^4) ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_example_l1129_112954


namespace NUMINAMATH_GPT_incorrect_statement_isosceles_trapezoid_l1129_112932

-- Define the properties of an isosceles trapezoid
structure IsoscelesTrapezoid (a b c d : ℝ) :=
  (parallel_bases : a = c ∨ b = d)  -- Bases are parallel
  (equal_diagonals : a = b) -- Diagonals are equal
  (equal_angles : ∀ α β : ℝ, α = β)  -- Angles on the same base are equal
  (axisymmetric : ∀ x : ℝ, x = -x)  -- Is an axisymmetric figure

-- Prove that the statement "The two bases of an isosceles trapezoid are parallel and equal" is incorrect
theorem incorrect_statement_isosceles_trapezoid (a b c d : ℝ) (h : IsoscelesTrapezoid a b c d) :
  ¬ (a = c ∧ b = d) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_isosceles_trapezoid_l1129_112932


namespace NUMINAMATH_GPT_Bulgaria_f_1992_divisibility_l1129_112939

def f (m n : ℕ) : ℕ := m^(3^(4 * n) + 6) - m^(3^(4 * n) + 4) - m^5 + m^3

theorem Bulgaria_f_1992_divisibility (n : ℕ) (m : ℕ) :
  ( ∀ m : ℕ, m > 0 → f m n ≡ 0 [MOD 1992] ) ↔ ( n % 2 = 1 ) :=
by
  sorry

end NUMINAMATH_GPT_Bulgaria_f_1992_divisibility_l1129_112939


namespace NUMINAMATH_GPT_smaller_tablet_diagonal_l1129_112945

theorem smaller_tablet_diagonal :
  ∀ (A_large A_small : ℝ)
    (d : ℝ),
    A_large = (8 / Real.sqrt 2) ^ 2 →
    A_small = (d / Real.sqrt 2) ^ 2 →
    A_large = A_small + 7.5 →
    d = 7
:= by
  intros A_large A_small d h1 h2 h3
  sorry

end NUMINAMATH_GPT_smaller_tablet_diagonal_l1129_112945


namespace NUMINAMATH_GPT_coordinates_of_point_l1129_112955

theorem coordinates_of_point (a : ℝ) (P : ℝ × ℝ) (hy : P = (a^2 - 1, a + 1)) (hx : (a^2 - 1) = 0) :
  P = (0, 2) ∨ P = (0, 0) :=
sorry

end NUMINAMATH_GPT_coordinates_of_point_l1129_112955


namespace NUMINAMATH_GPT_fraction_computation_l1129_112983

noncomputable def compute_fraction : ℚ :=
  (64^4 + 324) * (52^4 + 324) * (40^4 + 324) * (28^4 + 324) * (16^4 + 324) /
  (58^4 + 324) * (46^4 + 324) * (34^4 + 324) * (22^4 + 324) * (10^4 + 324)

theorem fraction_computation :
  compute_fraction = 137 / 1513 :=
by sorry

end NUMINAMATH_GPT_fraction_computation_l1129_112983


namespace NUMINAMATH_GPT_total_spent_is_correct_l1129_112917

def trumpet : ℝ := 149.16
def music_tool : ℝ := 9.98
def song_book : ℝ := 4.14
def trumpet_maintenance_accessories : ℝ := 21.47
def valve_oil_original : ℝ := 8.20
def valve_oil_discount_rate : ℝ := 0.20
def valve_oil_discounted : ℝ := valve_oil_original * (1 - valve_oil_discount_rate)
def band_t_shirt : ℝ := 14.95
def sales_tax_rate : ℝ := 0.065

def total_before_tax : ℝ :=
  trumpet + music_tool + song_book + trumpet_maintenance_accessories + valve_oil_discounted + band_t_shirt

def sales_tax : ℝ := total_before_tax * sales_tax_rate

def total_amount_spent : ℝ := total_before_tax + sales_tax

theorem total_spent_is_correct : total_amount_spent = 219.67 := by
  sorry

end NUMINAMATH_GPT_total_spent_is_correct_l1129_112917


namespace NUMINAMATH_GPT_logic_problem_l1129_112925

variable (p q : Prop)

theorem logic_problem (h₁ : p ∨ q) (h₂ : ¬ p) : ¬ p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_logic_problem_l1129_112925


namespace NUMINAMATH_GPT_equation_has_one_negative_and_one_zero_root_l1129_112904

theorem equation_has_one_negative_and_one_zero_root :
  ∃ x y : ℝ, x < 0 ∧ y = 0 ∧ 3^x + x^2 + 2 * x - 1 = 0 ∧ 3^y + y^2 + 2 * y - 1 = 0 :=
sorry

end NUMINAMATH_GPT_equation_has_one_negative_and_one_zero_root_l1129_112904


namespace NUMINAMATH_GPT_sqrt_sixteen_equals_four_l1129_112984

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sixteen_equals_four_l1129_112984


namespace NUMINAMATH_GPT_neg_P_4_of_P_implication_and_neg_P_5_l1129_112989

variable (P : ℕ → Prop)

theorem neg_P_4_of_P_implication_and_neg_P_5
  (h1 : ∀ k : ℕ, 0 < k → (P k → P (k+1)))
  (h2 : ¬ P 5) :
  ¬ P 4 :=
by
  sorry

end NUMINAMATH_GPT_neg_P_4_of_P_implication_and_neg_P_5_l1129_112989


namespace NUMINAMATH_GPT_total_strawberries_weight_is_72_l1129_112909

-- Define the weights
def Marco_strawberries_weight := 19
def dad_strawberries_weight := Marco_strawberries_weight + 34 

-- The total weight of their strawberries
def total_strawberries_weight := Marco_strawberries_weight + dad_strawberries_weight

-- Prove that the total weight is 72 pounds
theorem total_strawberries_weight_is_72 : total_strawberries_weight = 72 := by
  sorry

end NUMINAMATH_GPT_total_strawberries_weight_is_72_l1129_112909


namespace NUMINAMATH_GPT_largest_possible_a_l1129_112943

theorem largest_possible_a 
  (a b c d : ℕ) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 :=
sorry

end NUMINAMATH_GPT_largest_possible_a_l1129_112943


namespace NUMINAMATH_GPT_christen_potatoes_and_total_time_l1129_112935

-- Variables representing the given conditions
variables (homer_rate : ℕ) (christen_rate : ℕ) (initial_potatoes : ℕ) 
(homer_time_alone : ℕ) (total_time : ℕ)

-- Specific values for the given problem
def homerRate := 4
def christenRate := 6
def initialPotatoes := 60
def homerTimeAlone := 5

-- Function to calculate the number of potatoes peeled by Homer alone
def potatoesPeeledByHomerAlone :=
  homerRate * homerTimeAlone

-- Function to calculate the number of remaining potatoes
def remainingPotatoes :=
  initialPotatoes - potatoesPeeledByHomerAlone

-- Function to calculate the total peeling rate when Homer and Christen are working together
def combinedRate :=
  homerRate + christenRate

-- Function to calculate the time taken to peel the remaining potatoes
def timePeelingTogether :=
  remainingPotatoes / combinedRate

-- Function to calculate the total time spent peeling potatoes
def totalTime :=
  homerTimeAlone + timePeelingTogether

-- Function to calculate the number of potatoes peeled by Christen
def potatoesPeeledByChristen :=
  christenRate * timePeelingTogether

/- The theorem to be proven: Christen peeled 24 potatoes, and it took 9 minutes to peel all the potatoes. -/
theorem christen_potatoes_and_total_time :
  (potatoesPeeledByChristen = 24) ∧ (totalTime = 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_christen_potatoes_and_total_time_l1129_112935


namespace NUMINAMATH_GPT_range_of_y_l1129_112997

noncomputable def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_y :
  (∀ x : ℝ, operation (x - y) (x + y) < 1) ↔ - (1 : ℝ) / 2 < y ∧ y < (3 : ℝ) / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l1129_112997


namespace NUMINAMATH_GPT_number_of_distinct_intersections_of_curves_l1129_112960

theorem number_of_distinct_intersections_of_curves (x y : ℝ) :
  (∀ x y, x^2 - 4*y^2 = 4) ∧ (∀ x y, 4*x^2 + y^2 = 16) → 
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), 
    ((x1, y1) ≠ (x2, y2)) ∧
    ((x1^2 - 4*y1^2 = 4) ∧ (4*x1^2 + y1^2 = 16)) ∧
    ((x2^2 - 4*y2^2 = 4) ∧ (4*x2^2 + y2^2 = 16)) ∧
    ∀ (x' y' : ℝ), 
      ((x'^2 - 4*y'^2 = 4) ∧ (4*x'^2 + y'^2 = 16)) → 
      ((x', y') = (x1, y1) ∨ (x', y') = (x2, y2)) := 
sorry

end NUMINAMATH_GPT_number_of_distinct_intersections_of_curves_l1129_112960


namespace NUMINAMATH_GPT_circumscribed_sphere_surface_area_l1129_112940

noncomputable def surface_area_of_circumscribed_sphere_from_volume (V : ℝ) : ℝ :=
  let s := V^(1/3 : ℝ)
  let d := s * Real.sqrt 3
  4 * Real.pi * (d / 2) ^ 2

theorem circumscribed_sphere_surface_area (V : ℝ) (h : V = 27) : surface_area_of_circumscribed_sphere_from_volume V = 27 * Real.pi :=
by
  rw [h]
  unfold surface_area_of_circumscribed_sphere_from_volume
  sorry

end NUMINAMATH_GPT_circumscribed_sphere_surface_area_l1129_112940


namespace NUMINAMATH_GPT_maximize_profit_constraints_l1129_112922

variable (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

theorem maximize_profit_constraints (a1 a2 b1 b2 d1 d2 c1 c2 x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (a1 * x + a2 * y ≤ c1) ∧ (b1 * x + b2 * y ≤ c2) :=
sorry

end NUMINAMATH_GPT_maximize_profit_constraints_l1129_112922


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l1129_112921

-- Definition of the numbers we are interested in
def a : ℕ := 9118
def b : ℕ := 12173
def c : ℕ := 33182

-- Statement of the problem to prove GCD
theorem gcd_of_three_numbers : Int.gcd (Int.gcd a b) c = 47 := 
sorry  -- Proof skipped

end NUMINAMATH_GPT_gcd_of_three_numbers_l1129_112921


namespace NUMINAMATH_GPT_Mike_age_l1129_112938

-- We define the ages of Mike and Barbara
variables (M B : ℕ)

-- Conditions extracted from the problem
axiom h1 : B = M / 2
axiom h2 : M - B = 8

-- The theorem to prove
theorem Mike_age : M = 16 :=
by sorry

end NUMINAMATH_GPT_Mike_age_l1129_112938


namespace NUMINAMATH_GPT_chalkboard_area_l1129_112903

def width : Float := 3.5
def length : Float := 2.3 * width
def area : Float := length * width

theorem chalkboard_area : area = 28.175 :=
by 
  sorry

end NUMINAMATH_GPT_chalkboard_area_l1129_112903
