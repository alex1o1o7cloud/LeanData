import Mathlib

namespace NUMINAMATH_GPT_original_length_of_field_l1909_190995

theorem original_length_of_field (L W : ℕ) 
  (h1 : L * W = 144) 
  (h2 : (L + 6) * W = 198) : 
  L = 16 := 
by 
  sorry

end NUMINAMATH_GPT_original_length_of_field_l1909_190995


namespace NUMINAMATH_GPT_parabola_constant_term_l1909_190913

theorem parabola_constant_term (b c : ℝ)
  (h1 : 2 * b + c = 8)
  (h2 : -2 * b + c = -4)
  (h3 : 4 * b + c = 24) :
  c = 2 :=
sorry

end NUMINAMATH_GPT_parabola_constant_term_l1909_190913


namespace NUMINAMATH_GPT_apples_initial_count_l1909_190924

theorem apples_initial_count 
  (trees : ℕ)
  (apples_per_tree_picked : ℕ)
  (apples_picked_in_total : ℕ)
  (apples_remaining : ℕ)
  (initial_apples : ℕ) 
  (h1 : trees = 3) 
  (h2 : apples_per_tree_picked = 8) 
  (h3 : apples_picked_in_total = trees * apples_per_tree_picked)
  (h4 : apples_remaining = 9) 
  (h5 : initial_apples = apples_picked_in_total + apples_remaining) : 
  initial_apples = 33 :=
by sorry

end NUMINAMATH_GPT_apples_initial_count_l1909_190924


namespace NUMINAMATH_GPT_problem_proof_l1909_190900

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

end NUMINAMATH_GPT_problem_proof_l1909_190900


namespace NUMINAMATH_GPT_Josh_marbles_count_l1909_190910

-- Definitions of the given conditions
def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

-- The statement we aim to prove
theorem Josh_marbles_count : (initial_marbles - lost_marbles) = 9 :=
by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_Josh_marbles_count_l1909_190910


namespace NUMINAMATH_GPT_shingle_area_l1909_190948

-- Definitions from conditions
def length := 10 -- uncut side length in inches
def width := 7   -- uncut side width in inches
def trapezoid_base1 := 6 -- base of the trapezoid in inches
def trapezoid_height := 2 -- height of the trapezoid in inches

-- Definition derived from conditions
def trapezoid_base2 := length - trapezoid_base1 -- the second base of the trapezoid

-- Required proof in Lean
theorem shingle_area : (length * width - (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height)) = 60 := 
by
  sorry

end NUMINAMATH_GPT_shingle_area_l1909_190948


namespace NUMINAMATH_GPT_children_count_l1909_190923

theorem children_count (C : ℕ) 
    (cons : ℕ := 12)
    (total_cost : ℕ := 76)
    (child_ticket_cost : ℕ := 7)
    (adult_ticket_cost : ℕ := 10)
    (num_adults : ℕ := 5)
    (adult_cost := num_adults * adult_ticket_cost)
    (cost_with_concessions := total_cost - adult_cost )
    (children_cost := cost_with_concessions - cons):
    C = children_cost / child_ticket_cost :=
by
    sorry

end NUMINAMATH_GPT_children_count_l1909_190923


namespace NUMINAMATH_GPT_athlete_runs_entire_track_in_44_seconds_l1909_190937

noncomputable def time_to_complete_track (flags : ℕ) (time_to_4th_flag : ℕ) : ℕ :=
  let distances_between_flags := flags - 1
  let distances_to_4th_flag := 4 - 1
  let time_per_distance := time_to_4th_flag / distances_to_4th_flag
  distances_between_flags * time_per_distance

theorem athlete_runs_entire_track_in_44_seconds :
  time_to_complete_track 12 12 = 44 :=
by
  sorry

end NUMINAMATH_GPT_athlete_runs_entire_track_in_44_seconds_l1909_190937


namespace NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l1909_190901

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  3 * a + 2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l1909_190901


namespace NUMINAMATH_GPT_subtraction_property_l1909_190942

theorem subtraction_property : (12.56 - (5.56 - 2.63)) = (12.56 - 5.56 + 2.63) := 
by 
  sorry

end NUMINAMATH_GPT_subtraction_property_l1909_190942


namespace NUMINAMATH_GPT_minimum_value_exists_l1909_190973

noncomputable def min_value (a b c : ℝ) : ℝ :=
  a / (3 * b^2) + b / (4 * c^3) + c / (5 * a^4)

theorem minimum_value_exists :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → abc = 1 → min_value a b c ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_exists_l1909_190973


namespace NUMINAMATH_GPT_hamburgers_sold_in_winter_l1909_190969

theorem hamburgers_sold_in_winter:
  ∀ (T x : ℕ), 
  (T = 5 * 4) → 
  (5 + 6 + 4 + x = T) →
  (x = 5) :=
by
  intros T x hT hTotal
  sorry

end NUMINAMATH_GPT_hamburgers_sold_in_winter_l1909_190969


namespace NUMINAMATH_GPT_original_paint_intensity_l1909_190983

theorem original_paint_intensity (I : ℝ) (h1 : 0.5 * I + 0.5 * 20 = 15) : I = 10 :=
sorry

end NUMINAMATH_GPT_original_paint_intensity_l1909_190983


namespace NUMINAMATH_GPT_josh_points_l1909_190975

variable (x y : ℕ)
variable (three_point_success_rate two_point_success_rate : ℚ)
variable (total_shots : ℕ)
variable (points : ℚ)

theorem josh_points (h1 : three_point_success_rate = 0.25)
                    (h2 : two_point_success_rate = 0.40)
                    (h3 : total_shots = 40)
                    (h4 : x + y = total_shots) :
                    points = 32 :=
by sorry

end NUMINAMATH_GPT_josh_points_l1909_190975


namespace NUMINAMATH_GPT_isometric_curve_l1909_190977

noncomputable def Q (a b c x y : ℝ) := a * x^2 + 2 * b * x * y + c * y^2

theorem isometric_curve (a b c d e f : ℝ) (h : a * c - b^2 = 0) :
  ∃ (p : ℝ), (Q a b c x y + 2 * d * x + 2 * e * y = f → 
    (y^2 = 2 * p * x) ∨ 
    (∃ c' : ℝ, y^2 = c'^2) ∨ 
    y^2 = 0 ∨ 
    ∀ x y : ℝ, false) :=
sorry

end NUMINAMATH_GPT_isometric_curve_l1909_190977


namespace NUMINAMATH_GPT_derivative_of_f_l1909_190933

noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_f :
  (deriv f) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end NUMINAMATH_GPT_derivative_of_f_l1909_190933


namespace NUMINAMATH_GPT_determine_function_l1909_190917

theorem determine_function (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 := 
sorry

end NUMINAMATH_GPT_determine_function_l1909_190917


namespace NUMINAMATH_GPT_four_dice_min_rolls_l1909_190911

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end NUMINAMATH_GPT_four_dice_min_rolls_l1909_190911


namespace NUMINAMATH_GPT_divisors_of_30_l1909_190922

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end NUMINAMATH_GPT_divisors_of_30_l1909_190922


namespace NUMINAMATH_GPT_circle_passing_through_points_l1909_190986

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end NUMINAMATH_GPT_circle_passing_through_points_l1909_190986


namespace NUMINAMATH_GPT_butterfinger_count_l1909_190932

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end NUMINAMATH_GPT_butterfinger_count_l1909_190932


namespace NUMINAMATH_GPT_find_circle_eqn_range_of_slope_l1909_190968

noncomputable def circle_eqn_through_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) :=
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ {P : ℝ × ℝ | line P.1 P.2} ∧
    dist C M = dist C N ∧
    (∀ (P : ℝ × ℝ), dist P C = r ↔ (P = M ∨ P = N))

noncomputable def circle_standard_eqn (C : ℝ × ℝ) (r : ℝ) :=
  ∀ (P : ℝ × ℝ), dist P C = r ↔ (P.1 - C.1)^2 + P.2^2 = r^2

theorem find_circle_eqn (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (h : circle_eqn_through_points M N line) :
  ∃ r : ℝ, circle_standard_eqn (1, 0) r ∧ r = 5 := 
  sorry

theorem range_of_slope (k : ℝ) :
  0 < k → 8 * k^2 - 15 * k > 0 → k > (15 / 8) :=
  sorry

end NUMINAMATH_GPT_find_circle_eqn_range_of_slope_l1909_190968


namespace NUMINAMATH_GPT_remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l1909_190991

-- Part (a): Remainder of (1989 * 1990 * 1991 + 1992^2) when divided by 7 is 0.
theorem remainder_of_product_and_square_is_zero_mod_7 :
  (1989 * 1990 * 1991 + 1992^2) % 7 = 0 :=
sorry

-- Part (b): Remainder of 9^100 when divided by 8 is 1.
theorem remainder_of_9_pow_100_mod_8 :
  9^100 % 8 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l1909_190991


namespace NUMINAMATH_GPT_smallest_sum_infinite_geometric_progression_l1909_190905

theorem smallest_sum_infinite_geometric_progression :
  ∃ (a q A : ℝ), (a * q = 3) ∧ (0 < q) ∧ (q < 1) ∧ (A = a / (1 - q)) ∧ (A = 12) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_infinite_geometric_progression_l1909_190905


namespace NUMINAMATH_GPT_haley_picked_carrots_l1909_190918

variable (H : ℕ)
variable (mom_carrots : ℕ := 38)
variable (good_carrots : ℕ := 64)
variable (bad_carrots : ℕ := 13)
variable (total_carrots : ℕ := good_carrots + bad_carrots)

theorem haley_picked_carrots : H + mom_carrots = total_carrots → H = 39 := by
  sorry

end NUMINAMATH_GPT_haley_picked_carrots_l1909_190918


namespace NUMINAMATH_GPT_find_d_minus_c_l1909_190949

noncomputable def point_transformed (c d : ℝ) : Prop :=
  let Q := (c, d)
  let R := (2 * 2 - c, 2 * 3 - d)  -- Rotating Q by 180º about (2, 3)
  let S := (d, c)                -- Reflecting Q about the line y = x
  (S.1, S.2) = (2, -1)           -- Result is (2, -1)

theorem find_d_minus_c (c d : ℝ) (h : point_transformed c d) : d - c = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_d_minus_c_l1909_190949


namespace NUMINAMATH_GPT_root_interval_sum_l1909_190927

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 8

def has_root_in_interval (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : Prop :=
  a < b ∧ b - a = 1 ∧ f a < 0 ∧ f b > 0

theorem root_interval_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : has_root_in_interval a b h1 h2) : 
  a + b = 5 :=
sorry

end NUMINAMATH_GPT_root_interval_sum_l1909_190927


namespace NUMINAMATH_GPT_probability_train_or_plane_probability_not_ship_l1909_190956

def P_plane : ℝ := 0.2
def P_ship : ℝ := 0.3
def P_train : ℝ := 0.4
def P_car : ℝ := 0.1
def mutually_exclusive : Prop := P_plane + P_ship + P_train + P_car = 1

theorem probability_train_or_plane : mutually_exclusive → P_train + P_plane = 0.6 := by
  intro h
  sorry

theorem probability_not_ship : mutually_exclusive → 1 - P_ship = 0.7 := by
  intro h
  sorry

end NUMINAMATH_GPT_probability_train_or_plane_probability_not_ship_l1909_190956


namespace NUMINAMATH_GPT_sum_of_first_ten_terms_l1909_190912

variable {α : Type*} [LinearOrderedField α]

-- Defining the arithmetic sequence and sum of the first n terms
def a_n (a d : α) (n : ℕ) : α := a + d * (n - 1)

def S_n (a : α) (d : α) (n : ℕ) : α := n / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_ten_terms (a d : α) (h : a_n a d 3 + a_n a d 8 = 12) : S_n a d 10 = 60 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_ten_terms_l1909_190912


namespace NUMINAMATH_GPT_transform_to_quadratic_l1909_190966

theorem transform_to_quadratic :
  (∀ x : ℝ, (x + 1) ^ 2 + (x - 2) * (x + 2) = 1 ↔ 2 * x ^ 2 + 2 * x - 4 = 0) :=
sorry

end NUMINAMATH_GPT_transform_to_quadratic_l1909_190966


namespace NUMINAMATH_GPT_linear_function_not_in_fourth_quadrant_l1909_190989

theorem linear_function_not_in_fourth_quadrant (a b : ℝ) (h : a = 2 ∧ b = 1) :
  ∀ (x : ℝ), (2 * x + 1 < 0 → x > 0) := 
sorry

end NUMINAMATH_GPT_linear_function_not_in_fourth_quadrant_l1909_190989


namespace NUMINAMATH_GPT_carrie_phone_charges_l1909_190962

def total_miles (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def charges_needed (total_miles charge_miles : ℕ) : ℕ :=
  total_miles / charge_miles + if total_miles % charge_miles = 0 then 0 else 1

theorem carrie_phone_charges :
  let d1 := 135
  let d2 := 135 + 124
  let d3 := 159
  let d4 := 189
  let charge_miles := 106
  charges_needed (total_miles d1 d2 d3 d4) charge_miles = 7 :=
by
  sorry

end NUMINAMATH_GPT_carrie_phone_charges_l1909_190962


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1909_190944

theorem quadratic_inequality_solution 
  (x : ℝ) (b c : ℝ)
  (h : ∀ x, -x^2 + b*x + c < 0 ↔ x < -3 ∨ x > 2) :
  (6 * x^2 + x - 1 > 0) ↔ (x < -1/2 ∨ x > 1/3) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1909_190944


namespace NUMINAMATH_GPT_female_democrats_count_l1909_190947

theorem female_democrats_count (F M : ℕ) (h1 : F + M = 750) 
  (h2 : F / 2 ≠ 0) (h3 : M / 4 ≠ 0) 
  (h4 : F / 2 + M / 4 = 750 / 3) : F / 2 = 125 :=
by
  sorry

end NUMINAMATH_GPT_female_democrats_count_l1909_190947


namespace NUMINAMATH_GPT_fraction_of_male_birds_l1909_190943

theorem fraction_of_male_birds (T : ℕ) (h_cond1 : T ≠ 0) :
  let robins := (2 / 5) * T
  let bluejays := T - robins
  let male_robins := (2 / 3) * robins
  let male_bluejays := (1 / 3) * bluejays
  (male_robins + male_bluejays) / T = 7 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_of_male_birds_l1909_190943


namespace NUMINAMATH_GPT_transformed_point_of_function_l1909_190979

theorem transformed_point_of_function (f : ℝ → ℝ) (h : f 1 = -2) : f (-1) + 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_transformed_point_of_function_l1909_190979


namespace NUMINAMATH_GPT_negation_red_cards_in_deck_l1909_190916

variable (Deck : Type) (is_red : Deck → Prop) (is_in_deck : Deck → Prop)

theorem negation_red_cards_in_deck :
  (¬ ∃ x : Deck, is_red x ∧ is_in_deck x) ↔ (∃ x : Deck, is_red x ∧ is_in_deck x) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_red_cards_in_deck_l1909_190916


namespace NUMINAMATH_GPT_radius_of_circle_with_tangent_parabolas_l1909_190941

theorem radius_of_circle_with_tangent_parabolas (r : ℝ) : 
  (∀ x : ℝ, (x^2 + r = x → ∃ x0 : ℝ, x^2 + r = x0)) → r = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_with_tangent_parabolas_l1909_190941


namespace NUMINAMATH_GPT_range_of_m_l1909_190959

open Set Real

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - (m + 3) * x + m^2 = 0 }

theorem range_of_m (m : ℝ) :
  (A ∪ (univ \ B m)) = univ ↔ m ∈ Iio (-1) ∪ Ici 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1909_190959


namespace NUMINAMATH_GPT_length_of_field_l1909_190925

variable (l w : ℝ)

theorem length_of_field : 
  (l = 2 * w) ∧ (8 * 8 = 64) ∧ ((8 * 8) = (1 / 50) * l * w) → l = 80 :=
by
  sorry

end NUMINAMATH_GPT_length_of_field_l1909_190925


namespace NUMINAMATH_GPT_blueberries_in_blue_box_l1909_190903

theorem blueberries_in_blue_box (B S : ℕ) (h1: S - B = 10) (h2 : 50 = S) : B = 40 := 
by
  sorry

end NUMINAMATH_GPT_blueberries_in_blue_box_l1909_190903


namespace NUMINAMATH_GPT_first_reduction_percentage_l1909_190946

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.70 = P * 0.525 ↔ x = 25 := by
  sorry

end NUMINAMATH_GPT_first_reduction_percentage_l1909_190946


namespace NUMINAMATH_GPT_flowers_total_l1909_190931

def red_roses := 1491
def yellow_carnations := 3025
def white_roses := 1768
def purple_tulips := 2150
def pink_daisies := 3500
def blue_irises := 2973
def orange_marigolds := 4234
def lavender_orchids := 350
def sunflowers := 815
def violet_lilies := 26

theorem flowers_total :
  red_roses +
  yellow_carnations +
  white_roses +
  purple_tulips +
  pink_daisies +
  blue_irises +
  orange_marigolds +
  lavender_orchids +
  sunflowers +
  violet_lilies = 21332 := 
by
  -- Simplify and add up all given numbers
  sorry

end NUMINAMATH_GPT_flowers_total_l1909_190931


namespace NUMINAMATH_GPT_speed_of_second_train_l1909_190935

noncomputable def speed_of_first_train_kmph := 60 -- km/h
noncomputable def speed_of_first_train_mps := (speed_of_first_train_kmph * 1000) / 3600 -- m/s
noncomputable def length_of_first_train := 145 -- m
noncomputable def length_of_second_train := 165 -- m
noncomputable def time_to_cross := 8 -- seconds
noncomputable def total_distance := length_of_first_train + length_of_second_train -- m
noncomputable def relative_speed := total_distance / time_to_cross -- m/s

theorem speed_of_second_train (V : ℝ) :
  V * 1000 / 3600 + 60 * 1000 / 3600 = 38.75 →
  V = 79.5 := by {
  sorry
}

end NUMINAMATH_GPT_speed_of_second_train_l1909_190935


namespace NUMINAMATH_GPT_area_of_regular_octagon_l1909_190958

theorem area_of_regular_octagon (BDEF_is_rectangle : true) (AB : ℝ) (BC : ℝ) 
    (capture_regular_octagon : true) (AB_eq_1 : AB = 1) (BC_eq_2 : BC = 2)
    (octagon_perimeter_touch : ∀ x, x = 1) : 
    ∃ A : ℝ, A = 11 :=
by
  sorry

end NUMINAMATH_GPT_area_of_regular_octagon_l1909_190958


namespace NUMINAMATH_GPT_replace_digits_correct_l1909_190994

def digits_eq (a b c d e : ℕ) : Prop :=
  5 * 10 + a + (b * 100) + (c * 10) + 3 = (d * 1000) + (e * 100) + 1

theorem replace_digits_correct :
  ∃ (a b c d e : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
    digits_eq a b c d e ∧ a = 1 ∧ b = 1 ∧ c = 4 ∧ d = 1 ∧ e = 4 :=
by
  sorry

end NUMINAMATH_GPT_replace_digits_correct_l1909_190994


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l1909_190928

noncomputable def side_length (a : ℝ) := if a = 0 then 0 else (a : ℝ) * (3 : ℝ) / 2

theorem equilateral_triangle_side_length
  (a : ℝ)
  (h1 : a ≠ 0)
  (A := (a, - (1 / 3) * a^2))
  (B := (-a, - (1 / 3) * a^2))
  (Habo : (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2) :
  ∃ s : ℝ, s = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l1909_190928


namespace NUMINAMATH_GPT_monotonic_invertible_function_l1909_190974

theorem monotonic_invertible_function (f : ℝ → ℝ) (c : ℝ) (h_mono : ∀ x y, x < y → f x < f y) (h_inv : ∀ x, f (f⁻¹ x) = x) :
  (∀ x, f x + f⁻¹ x = 2 * x) ↔ ∀ x, f x = x + c :=
sorry

end NUMINAMATH_GPT_monotonic_invertible_function_l1909_190974


namespace NUMINAMATH_GPT_triangle_side_length_l1909_190999

theorem triangle_side_length {x : ℝ} (h1 : 6 + x + x = 20) : x = 7 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_side_length_l1909_190999


namespace NUMINAMATH_GPT_sqrt_pos_condition_l1909_190981

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_pos_condition_l1909_190981


namespace NUMINAMATH_GPT_pounds_of_coffee_bought_l1909_190987

theorem pounds_of_coffee_bought 
  (total_amount_gift_card : ℝ := 70) 
  (cost_per_pound : ℝ := 8.58) 
  (amount_left_on_card : ℝ := 35.68) :
  (total_amount_gift_card - amount_left_on_card) / cost_per_pound = 4 :=
sorry

end NUMINAMATH_GPT_pounds_of_coffee_bought_l1909_190987


namespace NUMINAMATH_GPT_find_t_l1909_190914

variables {t : ℝ}

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (t : ℝ) : ℝ × ℝ := (-2, t)

def are_parallel (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

theorem find_t (h : are_parallel vector_a (vector_b t)) : t = -4 :=
by sorry

end NUMINAMATH_GPT_find_t_l1909_190914


namespace NUMINAMATH_GPT_ratio_of_legs_of_triangles_l1909_190964

theorem ratio_of_legs_of_triangles (s a b : ℝ) (h1 : 0 < s)
  (h2 : a = s / 2)
  (h3 : b = (s * Real.sqrt 7) / 2) :
  b / a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_ratio_of_legs_of_triangles_l1909_190964


namespace NUMINAMATH_GPT_solve_system_eq_l1909_190904

theorem solve_system_eq (a b c x y z : ℝ) (h1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (h2 : x / a + y / b + z / c = a + b + c) (h3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eq_l1909_190904


namespace NUMINAMATH_GPT_solve_for_x_l1909_190953

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1909_190953


namespace NUMINAMATH_GPT_sum_of_ages_l1909_190921

theorem sum_of_ages (a b c : ℕ) 
  (h1 : a = 18 + b + c) 
  (h2 : a^2 = 2016 + (b + c)^2) : 
  a + b + c = 112 := 
sorry

end NUMINAMATH_GPT_sum_of_ages_l1909_190921


namespace NUMINAMATH_GPT_remainder_when_N_divided_by_1000_l1909_190902

def number_of_factors_of_5 (n : Nat) : Nat :=
  if n = 0 then 0 
  else n / 5 + number_of_factors_of_5 (n / 5)

def total_factors_of_5_upto (n : Nat) : Nat := 
  match n with
  | 0 => 0
  | n + 1 => number_of_factors_of_5 (n + 1) + total_factors_of_5_upto n

def product_factorial_5s : Nat := total_factors_of_5_upto 100

def N : Nat := product_factorial_5s

theorem remainder_when_N_divided_by_1000 : N % 1000 = 124 := by
  sorry

end NUMINAMATH_GPT_remainder_when_N_divided_by_1000_l1909_190902


namespace NUMINAMATH_GPT_odd_sol_exists_l1909_190934

theorem odd_sol_exists (n : ℕ) (hn : n > 0) : 
  ∃ (x_n y_n : ℕ), (x_n % 2 = 1) ∧ (y_n % 2 = 1) ∧ (x_n^2 + 7 * y_n^2 = 2^n) := 
sorry

end NUMINAMATH_GPT_odd_sol_exists_l1909_190934


namespace NUMINAMATH_GPT_company_fund_initial_amount_l1909_190997

-- Let n be the number of employees in the company.
variable (n : ℕ)

-- Conditions from the problem.
def initial_fund := 60 * n - 10
def adjusted_fund := 50 * n + 150
def employees_count := 16

-- Given the conditions, prove that the initial fund amount was $950.
theorem company_fund_initial_amount
    (h1 : adjusted_fund n = initial_fund n)
    (h2 : n = employees_count) : 
    initial_fund n = 950 := by
  sorry

end NUMINAMATH_GPT_company_fund_initial_amount_l1909_190997


namespace NUMINAMATH_GPT_quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l1909_190930

structure Point where
  x : ℚ
  y : ℚ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := 5, y := 4 }
def D : Point := { x := 6, y := 1 }

def line_eq_y_eq_kx_plus_b (k b x : ℚ) : ℚ := k * x + b

def intersects (A : Point) (P : Point × Point) (x y : ℚ) : Prop :=
  ∃ k b, P.1.y = line_eq_y_eq_kx_plus_b k b P.1.x ∧ P.2.y = line_eq_y_eq_kx_plus_b k b P.2.x ∧
         y = line_eq_y_eq_kx_plus_b k b x

theorem quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176 :
  ∃ (p q r s : ℚ), 
    gcd p q = 1 ∧ gcd r s = 1 ∧ intersects A (C, D) (p / q) (r / s) ∧
    (p + q + r + s = 176) :=
sorry

end NUMINAMATH_GPT_quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l1909_190930


namespace NUMINAMATH_GPT_wheel_rotation_angle_l1909_190984

-- Define the conditions
def radius : ℝ := 20
def arc_length : ℝ := 40

-- Define the theorem stating the desired proof problem
theorem wheel_rotation_angle (r : ℝ) (l : ℝ) (h_r : r = radius) (h_l : l = arc_length) :
  l / r = 2 := 
by sorry

end NUMINAMATH_GPT_wheel_rotation_angle_l1909_190984


namespace NUMINAMATH_GPT_problem_l1909_190982

noncomputable def f (x : ℝ) : ℝ := 5 * x - 7
noncomputable def g (x : ℝ) : ℝ := x / 5 + 3

theorem problem : ∀ x : ℝ, f (g x) - g (f x) = 6.4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_problem_l1909_190982


namespace NUMINAMATH_GPT_sum_f_values_l1909_190985

theorem sum_f_values (a b c d e f g : ℕ) 
  (h1: 100 * a * b = 100 * d)
  (h2: c * d * e = 100 * d)
  (h3: b * d * f = 100 * d)
  (h4: b * f = 100)
  (h5: 100 * d = 100) : 
  100 + 50 + 25 + 20 + 10 + 5 + 4 + 2 + 1 = 217 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_values_l1909_190985


namespace NUMINAMATH_GPT_find_d_l1909_190993

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_d (d e : ℝ) (h1 : -(-6) / 3 = 2) (h2 : 3 + d + e - 6 = 9) (h3 : -d / 3 = 6) : d = -18 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1909_190993


namespace NUMINAMATH_GPT_volume_comparison_l1909_190954

-- Define the properties for the cube and the cuboid.
def cube_side_length : ℕ := 1 -- in meters
def cuboid_width : ℕ := 50  -- in centimeters
def cuboid_length : ℕ := 50 -- in centimeters
def cuboid_height : ℕ := 20 -- in centimeters

-- Convert cube side length to centimeters.
def cube_side_length_cm := cube_side_length * 100 -- in centimeters

-- Calculate volumes.
def cube_volume : ℕ := cube_side_length_cm ^ 3 -- in cubic centimeters
def cuboid_volume : ℕ := cuboid_width * cuboid_length * cuboid_height -- in cubic centimeters

-- The theorem stating the problem.
theorem volume_comparison : cube_volume / cuboid_volume = 20 :=
by sorry

end NUMINAMATH_GPT_volume_comparison_l1909_190954


namespace NUMINAMATH_GPT_power_expansion_l1909_190940

theorem power_expansion (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := 
by 
  sorry

end NUMINAMATH_GPT_power_expansion_l1909_190940


namespace NUMINAMATH_GPT_ladder_base_distance_l1909_190965

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end NUMINAMATH_GPT_ladder_base_distance_l1909_190965


namespace NUMINAMATH_GPT_max_value_of_f_l1909_190963

def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 5.0625) ∧ (∃ x : ℝ, f x = 5.0625) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1909_190963


namespace NUMINAMATH_GPT_max_sum_abc_min_sum_reciprocal_l1909_190908

open Real

variables {a b c : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 2)

-- Maximum of a + b + c
theorem max_sum_abc : a + b + c ≤ sqrt 6 :=
by sorry

-- Minimum of 1/(a + b) + 1/(b + c) + 1/(c + a)
theorem min_sum_reciprocal : (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 * sqrt 6 / 4 :=
by sorry

end NUMINAMATH_GPT_max_sum_abc_min_sum_reciprocal_l1909_190908


namespace NUMINAMATH_GPT_measure_angle_PQR_given_conditions_l1909_190992

-- Definitions based on conditions
variables {R P Q S : Type} [LinearOrder R] [AddGroup Q] [LinearOrder P] [LinearOrder S]

-- Assume given conditions
def is_straight_line (r s p : ℝ) : Prop := r + p = 2 * s

def is_isosceles_triangle (p s q : ℝ) : Prop := p = q

def angle (q s p : ℝ) := (q - s) - (s - p)

variables (r p q s : ℝ)

-- Define the given angles and equality conditions
def given_conditions : Prop := 
  is_straight_line r s p ∧
  angle q s p = 60 ∧
  is_isosceles_triangle p s q ∧
  r ≠ q 

-- The theorem we want to prove
theorem measure_angle_PQR_given_conditions : given_conditions r p q s → angle p q r = 120 := by
  sorry

end NUMINAMATH_GPT_measure_angle_PQR_given_conditions_l1909_190992


namespace NUMINAMATH_GPT_find_r_x_l1909_190980

open Nat

theorem find_r_x (r n : ℕ) (x : ℕ) (h_r_le_70 : r ≤ 70) (repr_x : x = (10 * r + 6) * (r ^ (2 * n) - 1) / (r ^ 2 - 1))
  (repr_x2 : x^2 = (r ^ (4 * n) - 1) / (r - 1)) :
  (r = 7 ∧ x = 26) :=
by
  sorry

end NUMINAMATH_GPT_find_r_x_l1909_190980


namespace NUMINAMATH_GPT_most_and_least_l1909_190951

variables {Jan Kim Lee Ron Zay : ℝ}

-- Conditions as hypotheses
axiom H1 : Lee < Jan
axiom H2 : Kim < Jan
axiom H3 : Zay < Ron
axiom H4 : Zay < Lee
axiom H5 : Zay < Jan
axiom H6 : Jan < Ron

theorem most_and_least :
  (Ron > Jan) ∧ (Ron > Kim) ∧ (Ron > Lee) ∧ (Ron > Zay) ∧ 
  (Zay < Jan) ∧ (Zay < Kim) ∧ (Zay < Lee) ∧ (Zay < Ron) :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_most_and_least_l1909_190951


namespace NUMINAMATH_GPT_skips_in_one_meter_l1909_190970

variable (p q r s t u : ℕ)

theorem skips_in_one_meter (h1 : p * s * u = q * r * t) : 1 = (p * r * t) / (u * s * q) := by
  sorry

end NUMINAMATH_GPT_skips_in_one_meter_l1909_190970


namespace NUMINAMATH_GPT_minute_first_catch_hour_l1909_190960

theorem minute_first_catch_hour :
  ∃ (t : ℚ), t = 60 * (1 + (5 / 11)) :=
sorry

end NUMINAMATH_GPT_minute_first_catch_hour_l1909_190960


namespace NUMINAMATH_GPT_find_ab_solutions_l1909_190945

theorem find_ab_solutions (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h1 : (a + 1) ∣ (a ^ 3 * b - 1))
  (h2 : (b - 1) ∣ (b ^ 3 * a + 1)) : 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) :=
sorry

end NUMINAMATH_GPT_find_ab_solutions_l1909_190945


namespace NUMINAMATH_GPT_Carson_age_l1909_190950

theorem Carson_age {Aunt_Anna_Age : ℕ} (h1 : Aunt_Anna_Age = 60) 
                   {Maria_Age : ℕ} (h2 : Maria_Age = 2 * Aunt_Anna_Age / 3) 
                   {Carson_Age : ℕ} (h3 : Carson_Age = Maria_Age - 7) : 
                   Carson_Age = 33 := by sorry

end NUMINAMATH_GPT_Carson_age_l1909_190950


namespace NUMINAMATH_GPT_arithmetic_sequence_difference_l1909_190929

theorem arithmetic_sequence_difference :
  ∀ (a d : ℤ), a = -2 → d = 7 →
  |(a + (3010 - 1) * d) - (a + (3000 - 1) * d)| = 70 :=
by
  intros a d a_def d_def
  rw [a_def, d_def]
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_difference_l1909_190929


namespace NUMINAMATH_GPT_percentage_weight_loss_measured_l1909_190988

variable (W : ℝ)

def weight_after_loss (W : ℝ) := 0.85 * W
def weight_with_clothes (W : ℝ) := weight_after_loss W * 1.02

theorem percentage_weight_loss_measured (W : ℝ) :
  ((W - weight_with_clothes W) / W) * 100 = 13.3 := by
  sorry

end NUMINAMATH_GPT_percentage_weight_loss_measured_l1909_190988


namespace NUMINAMATH_GPT_no_solution_inequality_l1909_190955

theorem no_solution_inequality (m : ℝ) : (¬ ∃ x : ℝ, |x + 1| + |x - 5| ≤ m) ↔ m < 6 :=
sorry

end NUMINAMATH_GPT_no_solution_inequality_l1909_190955


namespace NUMINAMATH_GPT_not_even_nor_odd_l1909_190957

def f (x : ℝ) : ℝ := x^2

theorem not_even_nor_odd (x : ℝ) (h₁ : -1 < x) (h₂ : x ≤ 1) : ¬(∀ y, f y = f (-y)) ∧ ¬(∀ y, f y = -f (-y)) :=
by
  sorry

end NUMINAMATH_GPT_not_even_nor_odd_l1909_190957


namespace NUMINAMATH_GPT_range_of_a_l1909_190952

theorem range_of_a (a : ℝ) :
    (∀ x : ℤ, x + 1 > 0 → 3 * x - a ≤ 0 → x = 0 ∨ x = 1 ∨ x = 2) ↔ 6 ≤ a ∧ a < 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1909_190952


namespace NUMINAMATH_GPT_strap_mask_probability_l1909_190907

theorem strap_mask_probability 
  (p_regular_medical : ℝ)
  (p_surgical : ℝ)
  (p_strap_regular : ℝ)
  (p_strap_surgical : ℝ)
  (h_regular_medical : p_regular_medical = 0.8)
  (h_surgical : p_surgical = 0.2)
  (h_strap_regular : p_strap_regular = 0.1)
  (h_strap_surgical : p_strap_surgical = 0.2) :
  (p_regular_medical * p_strap_regular + p_surgical * p_strap_surgical) = 0.12 :=
by
  rw [h_regular_medical, h_surgical, h_strap_regular, h_strap_surgical]
  -- proof will go here
  sorry

end NUMINAMATH_GPT_strap_mask_probability_l1909_190907


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1909_190971

theorem solution_set_of_inequality : {x : ℝ // |x - 2| > x - 2} = {x : ℝ // x < 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1909_190971


namespace NUMINAMATH_GPT_greatest_area_difference_l1909_190961

def first_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 156

def second_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 144

theorem greatest_area_difference : 
  ∃ (l1 w1 l2 w2 : ℕ), 
  first_rectangle_perimeter l1 w1 ∧ 
  second_rectangle_perimeter l2 w2 ∧ 
  (l1 * (78 - l1) - l2 * (72 - l2) = 225) := 
sorry

end NUMINAMATH_GPT_greatest_area_difference_l1909_190961


namespace NUMINAMATH_GPT_average_of_B_and_C_l1909_190939

theorem average_of_B_and_C (x : ℚ) (A B C : ℚ)
  (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  (B + C) / 2 = 93.75 := 
sorry

end NUMINAMATH_GPT_average_of_B_and_C_l1909_190939


namespace NUMINAMATH_GPT_parabola_point_value_l1909_190990

variable {x₀ y₀ : ℝ}

theorem parabola_point_value
  (h₁ : y₀^2 = 4 * x₀)
  (h₂ : (Real.sqrt ((x₀ - 1)^2 + y₀^2) = 5/4 * x₀)) :
  x₀ = 4 := by
  sorry

end NUMINAMATH_GPT_parabola_point_value_l1909_190990


namespace NUMINAMATH_GPT_dodecagon_enclosure_l1909_190936

theorem dodecagon_enclosure (m n : ℕ) (h1 : m = 12) 
  (h2 : ∀ (x : ℕ), x ∈ { k | ∃ p : ℕ, p = n ∧ 12 = k * p}) :
  n = 12 :=
by
  -- begin proof steps here
sorry

end NUMINAMATH_GPT_dodecagon_enclosure_l1909_190936


namespace NUMINAMATH_GPT_speed_of_sound_l1909_190906

theorem speed_of_sound (time_blasts : ℝ) (distance_traveled : ℝ) (time_heard : ℝ) (speed : ℝ) 
  (h_blasts : time_blasts = 30 * 60) -- time between the two blasts in seconds 
  (h_distance : distance_traveled = 8250) -- distance in meters
  (h_heard : time_heard = 30 * 60 + 25) -- time when man heard the second blast
  (h_relationship : speed = distance_traveled / (time_heard - time_blasts)) : 
  speed = 330 :=
sorry

end NUMINAMATH_GPT_speed_of_sound_l1909_190906


namespace NUMINAMATH_GPT_least_integer_sum_of_primes_l1909_190938

-- Define what it means to be prime and greater than a number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greater_than_ten (n : ℕ) : Prop := n > 10

-- Main theorem statement
theorem least_integer_sum_of_primes :
  ∃ n, (∀ p1 p2 p3 p4 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
                        greater_than_ten p1 ∧ greater_than_ten p2 ∧ greater_than_ten p3 ∧ greater_than_ten p4 ∧
                        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
                        n = p1 + p2 + p3 + p4 → n ≥ 60) ∧
        n = 60 :=
  sorry

end NUMINAMATH_GPT_least_integer_sum_of_primes_l1909_190938


namespace NUMINAMATH_GPT_triangle_inequality_l1909_190998

theorem triangle_inequality (a b c : ℝ) (h : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l1909_190998


namespace NUMINAMATH_GPT_plates_arrangement_l1909_190915

theorem plates_arrangement :
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  non_adjacent_green_arrangements = 588 :=
by
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  sorry

end NUMINAMATH_GPT_plates_arrangement_l1909_190915


namespace NUMINAMATH_GPT_aunt_wang_bought_n_lilies_l1909_190919

theorem aunt_wang_bought_n_lilies 
  (cost_rose : ℕ) 
  (cost_lily : ℕ) 
  (total_spent : ℕ) 
  (num_roses : ℕ) 
  (num_lilies : ℕ) 
  (roses_cost : num_roses * cost_rose = 10) 
  (total_spent_cond : total_spent = 55) 
  (cost_conditions : cost_rose = 5 ∧ cost_lily = 9) 
  (spending_eq : total_spent = num_roses * cost_rose + num_lilies * cost_lily) : 
  num_lilies = 5 :=
by 
  sorry

end NUMINAMATH_GPT_aunt_wang_bought_n_lilies_l1909_190919


namespace NUMINAMATH_GPT_total_cost_of_apples_l1909_190996

theorem total_cost_of_apples (cost_per_kg : ℝ) (packaging_fee : ℝ) (weight : ℝ) :
  cost_per_kg = 15.3 →
  packaging_fee = 0.25 →
  weight = 2.5 →
  (weight * (cost_per_kg + packaging_fee) = 38.875) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_cost_of_apples_l1909_190996


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1909_190920

def P (x : ℝ) : Prop := x^3 - x^2 + 1 ≤ 0

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_universal_statement_l1909_190920


namespace NUMINAMATH_GPT_integral_f_eq_34_l1909_190978

noncomputable def f (x : ℝ) := if x ∈ [0, 1] then (1 / Real.pi) * Real.sqrt (1 - x^2) else 2 - x

theorem integral_f_eq_34 :
  ∫ x in (0 : ℝ)..2, f x = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_integral_f_eq_34_l1909_190978


namespace NUMINAMATH_GPT_units_digit_2009_2008_plus_2013_l1909_190909

theorem units_digit_2009_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_2009_2008_plus_2013_l1909_190909


namespace NUMINAMATH_GPT_average_temperature_l1909_190926

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_l1909_190926


namespace NUMINAMATH_GPT_expand_product_l1909_190967

-- Define x as a variable within the real numbers
variable (x : ℝ)

-- Statement of the theorem
theorem expand_product : (x + 3) * (x - 4) = x^2 - x - 12 := 
by 
  sorry

end NUMINAMATH_GPT_expand_product_l1909_190967


namespace NUMINAMATH_GPT_sequence_ab_sum_l1909_190972

theorem sequence_ab_sum (s a b : ℝ) (h1 : 16 * s = 4) (h2 : 1024 * s = a) (h3 : a * s = b) : a + b = 320 := by
  sorry

end NUMINAMATH_GPT_sequence_ab_sum_l1909_190972


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1909_190976

variable {a : ℕ → ℝ} (a2 a5 : ℝ)
variable (h1 : a 2 = 9) (h2 : a 5 = 33)

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 8 := by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1909_190976
