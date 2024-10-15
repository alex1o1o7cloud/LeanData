import Mathlib

namespace NUMINAMATH_GPT_sum_of_ages_l2229_222991

-- Defining the ages of Nathan and his twin sisters.
variables (n t : ℕ)

-- Nathan has two twin younger sisters, and the product of their ages equals 72.
def valid_ages (n t : ℕ) : Prop := t < n ∧ n * t * t = 72

-- Prove that the sum of the ages of Nathan and his twin sisters is 14.
theorem sum_of_ages (n t : ℕ) (h : valid_ages n t) : 2 * t + n = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l2229_222991


namespace NUMINAMATH_GPT_gcf_50_75_l2229_222931

theorem gcf_50_75 : Nat.gcd 50 75 = 25 := by
  sorry

end NUMINAMATH_GPT_gcf_50_75_l2229_222931


namespace NUMINAMATH_GPT_deepak_age_l2229_222938

-- Defining the problem with the given conditions in Lean:
theorem deepak_age (x : ℕ) (rahul_current : ℕ := 4 * x) (deepak_current : ℕ := 3 * x) :
  (rahul_current + 6 = 38) → (deepak_current = 24) :=
by
  sorry

end NUMINAMATH_GPT_deepak_age_l2229_222938


namespace NUMINAMATH_GPT_frustum_volume_fraction_l2229_222959

theorem frustum_volume_fraction {V_original V_frustum : ℚ} 
(base_edge : ℚ) (height : ℚ) 
(h1 : base_edge = 24) (h2 : height = 18) 
(h3 : V_original = (1 / 3) * (base_edge ^ 2) * height)
(smaller_base_edge : ℚ) (smaller_height : ℚ) 
(h4 : smaller_height = (1 / 3) * height) (h5 : smaller_base_edge = base_edge / 3) 
(V_smaller : ℚ) (h6 : V_smaller = (1 / 3) * (smaller_base_edge ^ 2) * smaller_height)
(h7 : V_frustum = V_original - V_smaller) :
V_frustum / V_original = 13 / 27 :=
sorry

end NUMINAMATH_GPT_frustum_volume_fraction_l2229_222959


namespace NUMINAMATH_GPT_largest_integer_less_than_100_div_8_rem_5_l2229_222920

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_div_8_rem_5_l2229_222920


namespace NUMINAMATH_GPT_sum_of_squares_of_coefficients_l2229_222904

theorem sum_of_squares_of_coefficients :
  ∃ a b c d e f : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coefficients_l2229_222904


namespace NUMINAMATH_GPT_money_left_l2229_222943

def initial_money : ℝ := 18
def spent_on_video_games : ℝ := 6
def spent_on_snack : ℝ := 3
def toy_original_cost : ℝ := 4
def toy_discount : ℝ := 0.25

theorem money_left (initial_money spent_on_video_games spent_on_snack toy_original_cost toy_discount : ℝ) :
  initial_money = 18 →
  spent_on_video_games = 6 →
  spent_on_snack = 3 →
  toy_original_cost = 4 →
  toy_discount = 0.25 →
  (initial_money - (spent_on_video_games + spent_on_snack + (toy_original_cost * (1 - toy_discount)))) = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_money_left_l2229_222943


namespace NUMINAMATH_GPT_quadratic_to_standard_form_l2229_222935

theorem quadratic_to_standard_form (a b c : ℝ) (x : ℝ) :
  (20 * x^2 + 240 * x + 3200 = a * (x + b)^2 + c) → (a + b + c = 2506) :=
  sorry

end NUMINAMATH_GPT_quadratic_to_standard_form_l2229_222935


namespace NUMINAMATH_GPT_ball_color_problem_l2229_222976

theorem ball_color_problem
  (n : ℕ)
  (h₀ : ∀ i : ℕ, i ≤ 49 → ∃ r : ℕ, r = 49 ∧ i = 50) 
  (h₁ : ∀ i : ℕ, i > 49 → ∃ r : ℕ, r = 49 + 7 * (i - 50) / 8 ∧ i = n)
  (h₂ : 90 ≤ (49 + (7 * (n - 50) / 8)) * 10 / n) :
  n ≤ 210 := 
sorry

end NUMINAMATH_GPT_ball_color_problem_l2229_222976


namespace NUMINAMATH_GPT_pizza_toppings_combination_l2229_222929

def num_combinations {α : Type} (s : Finset α) (k : ℕ) : ℕ :=
  (s.card.choose k)

theorem pizza_toppings_combination (s : Finset ℕ) (h : s.card = 7) : num_combinations s 3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_pizza_toppings_combination_l2229_222929


namespace NUMINAMATH_GPT_skips_per_meter_l2229_222960

variable (a b c d e f g h : ℕ)

theorem skips_per_meter 
  (hops_skips : a * skips = b * hops)
  (jumps_hops : c * jumps = d * hops)
  (leaps_jumps : e * leaps = f * jumps)
  (leaps_meters : g * leaps = h * meters) :
  1 * skips = (g * b * f * d) / (a * e * h * c) * skips := 
sorry

end NUMINAMATH_GPT_skips_per_meter_l2229_222960


namespace NUMINAMATH_GPT_billy_finished_before_margaret_l2229_222978

-- Define the conditions
def billy_first_laps_time : ℕ := 2 * 60
def billy_next_three_laps_time : ℕ := 4 * 60
def billy_ninth_lap_time : ℕ := 1 * 60
def billy_tenth_lap_time : ℕ := 150
def margaret_total_time : ℕ := 10 * 60

-- The main statement to prove that Billy finished 30 seconds before Margaret
theorem billy_finished_before_margaret :
  (billy_first_laps_time + billy_next_three_laps_time + billy_ninth_lap_time + billy_tenth_lap_time) + 30 = margaret_total_time :=
by
  sorry

end NUMINAMATH_GPT_billy_finished_before_margaret_l2229_222978


namespace NUMINAMATH_GPT_labels_closer_than_distance_l2229_222998

noncomputable def exists_points_with_labels_closer_than_distance (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ |f P - f Q| < dist P Q

-- Statement of the problem
theorem labels_closer_than_distance :
  ∀ (f : ℝ × ℝ → ℝ), exists_points_with_labels_closer_than_distance f :=
sorry

end NUMINAMATH_GPT_labels_closer_than_distance_l2229_222998


namespace NUMINAMATH_GPT_weight_of_B_l2229_222988

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by sorry

end NUMINAMATH_GPT_weight_of_B_l2229_222988


namespace NUMINAMATH_GPT_matrix_identity_l2229_222936

noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![-2, 1]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  N * N = 4 • N + -11 • I :=
by
  sorry

end NUMINAMATH_GPT_matrix_identity_l2229_222936


namespace NUMINAMATH_GPT_jonah_profit_is_correct_l2229_222908

noncomputable def jonah_profit : Real :=
  let pineapples := 6
  let pricePerPineapple := 3
  let pineappleCostWithoutDiscount := pineapples * pricePerPineapple
  let discount := if pineapples > 4 then 0.20 * pineappleCostWithoutDiscount else 0
  let totalCostAfterDiscount := pineappleCostWithoutDiscount - discount
  let ringsPerPineapple := 10
  let totalRings := pineapples * ringsPerPineapple
  let ringsSoldIndividually := 2
  let pricePerIndividualRing := 5
  let revenueFromIndividualRings := ringsSoldIndividually * pricePerIndividualRing
  let ringsLeft := totalRings - ringsSoldIndividually
  let ringsPerSet := 4
  let setsSold := ringsLeft / ringsPerSet -- This should be interpreted as an integer division
  let pricePerSet := 16
  let revenueFromSets := setsSold * pricePerSet
  let totalRevenue := revenueFromIndividualRings + revenueFromSets
  let profit := totalRevenue - totalCostAfterDiscount
  profit
  
theorem jonah_profit_is_correct :
  jonah_profit = 219.60 := by
  sorry

end NUMINAMATH_GPT_jonah_profit_is_correct_l2229_222908


namespace NUMINAMATH_GPT_division_problem_l2229_222914

theorem division_problem : 0.05 / 0.0025 = 20 := 
sorry

end NUMINAMATH_GPT_division_problem_l2229_222914


namespace NUMINAMATH_GPT_exists_rationals_leq_l2229_222900

theorem exists_rationals_leq (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f (a + b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_rationals_leq_l2229_222900


namespace NUMINAMATH_GPT_count_seating_arrangements_l2229_222953

/-
  Definition of the seating problem at the round table:
  - The committee has six members from each of three species: Martians (M), Venusians (V), and Earthlings (E).
  - The table has 18 seats numbered from 1 to 18.
  - Seat 1 is occupied by a Martian, and seat 18 is occupied by an Earthling.
  - Martians cannot sit immediately to the left of Venusians.
  - Venusians cannot sit immediately to the left of Earthlings.
  - Earthlings cannot sit immediately to the left of Martians.
-/
def num_arrangements_valid_seating : ℕ := -- the number of valid seating arrangements
  sorry

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def N : ℕ := 347

theorem count_seating_arrangements :
  num_arrangements_valid_seating = N * (factorial 6)^3 :=
sorry

end NUMINAMATH_GPT_count_seating_arrangements_l2229_222953


namespace NUMINAMATH_GPT_fewest_cookies_l2229_222985

theorem fewest_cookies
  (r a s d1 d2 : ℝ)
  (hr_pos : r > 0)
  (ha_pos : a > 0)
  (hs_pos : s > 0)
  (hd1_pos : d1 > 0)
  (hd2_pos : d2 > 0)
  (h_Alice_cookies : 15 = 15)
  (h_same_dough : true) :
  15 < (15 * (Real.pi * r^2)) / (a^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((3 * Real.sqrt 3 / 2) * s^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((1 / 2) * d1 * d2) :=
by
  sorry

end NUMINAMATH_GPT_fewest_cookies_l2229_222985


namespace NUMINAMATH_GPT_find_a_minus_b_l2229_222942

theorem find_a_minus_b (a b : ℝ) (h1 : a + b = 12) (h2 : a^2 - b^2 = 48) : a - b = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_minus_b_l2229_222942


namespace NUMINAMATH_GPT_find_triples_l2229_222903

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x^2 + y^2 = 3 * 2016^z + 77) :
  (x = 4 ∧ y = 8 ∧ z = 0) ∨ (x = 8 ∧ y = 4 ∧ z = 0) ∨
  (x = 14 ∧ y = 77 ∧ z = 1) ∨ (x = 77 ∧ y = 14 ∧ z = 1) ∨
  (x = 35 ∧ y = 70 ∧ z = 1) ∨ (x = 70 ∧ y = 35 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_find_triples_l2229_222903


namespace NUMINAMATH_GPT_tangent_ellipse_hyperbola_l2229_222912

-- Definitions of the curves
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y+3)^2 = 1

-- Condition for tangency: the curves must meet and the discriminant must be zero
noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Prove the given curves are tangent at some x and y for m = 8/9
theorem tangent_ellipse_hyperbola : 
    (∃ x y : ℝ, ellipse x y ∧ hyperbola x y (8 / 9)) ∧ 
    quadratic_discriminant ((8 / 9) + 9) (6 * (8 / 9)) ((-8/9) * (8 * (8/9)) - 8) = 0 :=
sorry

end NUMINAMATH_GPT_tangent_ellipse_hyperbola_l2229_222912


namespace NUMINAMATH_GPT_expand_expression_l2229_222965

theorem expand_expression (x : ℝ) :
  (2 * x + 3) * (4 * x - 5) = 8 * x^2 + 2 * x - 15 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2229_222965


namespace NUMINAMATH_GPT_john_toy_store_fraction_l2229_222926

theorem john_toy_store_fraction
  (allowance : ℝ)
  (spent_at_arcade_fraction : ℝ)
  (remaining_allowance : ℝ)
  (spent_at_candy_store : ℝ)
  (spent_at_toy_store : ℝ)
  (john_allowance : allowance = 3.60)
  (arcade_fraction : spent_at_arcade_fraction = 3 / 5)
  (arcade_amount : remaining_allowance = allowance - (spent_at_arcade_fraction * allowance))
  (candy_store_amount : spent_at_candy_store = 0.96)
  (remaining_after_candy_store : spent_at_toy_store = remaining_allowance - spent_at_candy_store)
  : spent_at_toy_store / remaining_allowance = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_john_toy_store_fraction_l2229_222926


namespace NUMINAMATH_GPT_sin_cos_power_four_l2229_222948

theorem sin_cos_power_four (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := 
sorry

end NUMINAMATH_GPT_sin_cos_power_four_l2229_222948


namespace NUMINAMATH_GPT_AB_length_l2229_222993

noncomputable def length_of_AB (x y : ℝ) (P_ratio Q_ratio : ℝ × ℝ) (PQ_distance : ℝ) : ℝ :=
    x + y

theorem AB_length (x y : ℝ) (P_ratio : ℝ × ℝ := (3, 5)) (Q_ratio : ℝ × ℝ := (4, 5)) (PQ_distance : ℝ := 3) 
    (h1 : 5 * x = 3 * y) -- P divides AB in the ratio 3:5
    (h2 : 5 * (x + 3) = 4 * (y - 3)) -- Q divides AB in the ratio 4:5 and PQ = 3 units
    : length_of_AB x y P_ratio Q_ratio PQ_distance = 43.2 := 
by sorry

end NUMINAMATH_GPT_AB_length_l2229_222993


namespace NUMINAMATH_GPT_n_must_be_power_of_3_l2229_222971

theorem n_must_be_power_of_3 (n : ℕ) (h1 : 0 < n) (h2 : Prime (4 ^ n + 2 ^ n + 1)) : ∃ k : ℕ, n = 3 ^ k :=
by
  sorry

end NUMINAMATH_GPT_n_must_be_power_of_3_l2229_222971


namespace NUMINAMATH_GPT_inequality_proof_l2229_222992

variable (a b c d : ℝ)
variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

-- Define conditions
def positive (x : ℝ) := x > 0
def unit_circle (x y : ℝ) := x^2 + y^2 = 1

-- Define the main theorem
theorem inequality_proof
  (ha : positive a)
  (hb : positive b)
  (hc : positive c)
  (hd : positive d)
  (habcd : a * b + c * d = 1)
  (hP1 : unit_circle x1 y1)
  (hP2 : unit_circle x2 y2)
  (hP3 : unit_circle x3 y3)
  (hP4 : unit_circle x4 y4)
  : 
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := sorry

end NUMINAMATH_GPT_inequality_proof_l2229_222992


namespace NUMINAMATH_GPT_line_equation_passing_through_point_and_equal_intercepts_l2229_222983

theorem line_equation_passing_through_point_and_equal_intercepts :
    (∃ k: ℝ, ∀ x y: ℝ, (2, 5) = (x, k * x) ∨ x + y = 7) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_passing_through_point_and_equal_intercepts_l2229_222983


namespace NUMINAMATH_GPT_coloring_count_is_2_l2229_222989

noncomputable def count_colorings (initial_color : String) : Nat := 
  if initial_color = "R" then 2 else 0 -- Assumes only the case of initial red color is valid for simplicity

theorem coloring_count_is_2 (h1 : True) (h2 : True) (h3 : True) (h4 : True):
  count_colorings "R" = 2 := by
  sorry

end NUMINAMATH_GPT_coloring_count_is_2_l2229_222989


namespace NUMINAMATH_GPT_solve_pounds_l2229_222913

def price_per_pound_corn : ℝ := 1.20
def price_per_pound_beans : ℝ := 0.60
def price_per_pound_rice : ℝ := 0.80
def total_weight : ℕ := 30
def total_cost : ℝ := 24.00
def equal_beans_rice (b r : ℕ) : Prop := b = r

theorem solve_pounds (c b r : ℕ) (h1 : price_per_pound_corn * ↑c + price_per_pound_beans * ↑b + price_per_pound_rice * ↑r = total_cost)
    (h2 : c + b + r = total_weight) (h3 : equal_beans_rice b r) : c = 6 ∧ b = 12 ∧ r = 12 := by
  sorry

end NUMINAMATH_GPT_solve_pounds_l2229_222913


namespace NUMINAMATH_GPT_initial_percentage_of_grape_juice_l2229_222945

theorem initial_percentage_of_grape_juice (P : ℝ) 
  (h₀ : 10 + 30 = 40)
  (h₁ : 40 * 0.325 = 13)
  (h₂ : 30 * P + 10 = 13) : 
  P = 0.1 :=
  by 
    sorry

end NUMINAMATH_GPT_initial_percentage_of_grape_juice_l2229_222945


namespace NUMINAMATH_GPT_tan_angle_sum_l2229_222905

variable (α β : ℝ)

theorem tan_angle_sum (h1 : Real.tan (α - Real.pi / 6) = 3 / 7)
                      (h2 : Real.tan (Real.pi / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_angle_sum_l2229_222905


namespace NUMINAMATH_GPT_min_area_ABCD_l2229_222958

section Quadrilateral

variables {S1 S2 S3 S4 : ℝ}

-- Define the areas of the triangles
def area_APB := S1
def area_BPC := S2
def area_CPD := S3
def area_DPA := S4

-- Condition: Product of the areas of ΔAPB and ΔCPD is 36
axiom prod_APB_CPD : S1 * S3 = 36

-- We need to prove that the minimum area of the quadrilateral ABCD is 24
theorem min_area_ABCD : S1 + S2 + S3 + S4 ≥ 24 :=
by
  sorry

end Quadrilateral

end NUMINAMATH_GPT_min_area_ABCD_l2229_222958


namespace NUMINAMATH_GPT_Jeff_pays_when_picking_up_l2229_222918

-- Definition of the conditions
def deposit_rate : ℝ := 0.10
def increase_rate : ℝ := 0.40
def last_year_cost : ℝ := 250
def this_year_cost : ℝ := last_year_cost * (1 + increase_rate)
def deposit : ℝ := this_year_cost * deposit_rate

-- Lean statement of the proof
theorem Jeff_pays_when_picking_up : this_year_cost - deposit = 315 := by
  sorry

end NUMINAMATH_GPT_Jeff_pays_when_picking_up_l2229_222918


namespace NUMINAMATH_GPT_a_gt_one_l2229_222915

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 - x - 1

theorem a_gt_one (a : ℝ) :
  (∃! x, 0 < x ∧ x < 1 ∧ f a x = 0) → 1 < a :=
by
  sorry

end NUMINAMATH_GPT_a_gt_one_l2229_222915


namespace NUMINAMATH_GPT_mean_of_five_numbers_l2229_222986

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_l2229_222986


namespace NUMINAMATH_GPT_abc_value_l2229_222955

theorem abc_value (a b c : ℝ) 
  (h0 : (a * (0 : ℝ)^2 + b * (0 : ℝ) + c) = 7) 
  (h1 : (a * (1 : ℝ)^2 + b * (1 : ℝ) + c) = 4) : 
  a + b + 2 * c = 11 :=
by sorry

end NUMINAMATH_GPT_abc_value_l2229_222955


namespace NUMINAMATH_GPT_triangle_perimeter_l2229_222906

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2229_222906


namespace NUMINAMATH_GPT_find_a5_l2229_222919

variable (a_n : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 
  (h1 : is_arithmetic_sequence a_n d)
  (h2 : a_n 3 + a_n 8 = 22)
  (h3 : a_n 6 = 7) :
  a_n 5 = 15 :=
sorry

end NUMINAMATH_GPT_find_a5_l2229_222919


namespace NUMINAMATH_GPT_games_given_away_l2229_222997

/-- Gwen had ninety-eight DS games. 
    After she gave some to her friends she had ninety-one left.
    Prove that she gave away 7 DS games. -/
theorem games_given_away (original_games : ℕ) (games_left : ℕ) (games_given : ℕ) 
  (h1 : original_games = 98) 
  (h2 : games_left = 91) 
  (h3 : games_given = original_games - games_left) : 
  games_given = 7 :=
sorry

end NUMINAMATH_GPT_games_given_away_l2229_222997


namespace NUMINAMATH_GPT_cistern_wet_surface_area_l2229_222987

theorem cistern_wet_surface_area
  (length : ℝ) (width : ℝ) (breadth : ℝ)
  (h_length : length = 9)
  (h_width : width = 6)
  (h_breadth : breadth = 2.25) :
  (length * width + 2 * (length * breadth) + 2 * (width * breadth)) = 121.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cistern_wet_surface_area_l2229_222987


namespace NUMINAMATH_GPT_chocolate_bar_cost_l2229_222949

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l2229_222949


namespace NUMINAMATH_GPT_union_of_sets_eq_A_l2229_222980

noncomputable def A : Set ℝ := {x | x / ((x + 1) * (x - 4)) < 0}
noncomputable def B : Set ℝ := {x | Real.log x < 1}

theorem union_of_sets_eq_A: A ∪ B = A := by
  sorry

end NUMINAMATH_GPT_union_of_sets_eq_A_l2229_222980


namespace NUMINAMATH_GPT_remaining_adults_fed_l2229_222921

theorem remaining_adults_fed 
  (cans : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (remaining_cans : ℕ)
  (remaining_adults : ℕ) :
  (adults_per_can = 4) →
  (children_per_can = 6) →
  (initial_cans = 7) →
  (children_fed = 18) →
  (remaining_cans = initial_cans - children_fed / children_per_can) →
  (remaining_adults = remaining_cans * adults_per_can) →
  remaining_adults = 16 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_remaining_adults_fed_l2229_222921


namespace NUMINAMATH_GPT_path_counts_l2229_222982

    noncomputable def x : ℝ := 2 + Real.sqrt 2
    noncomputable def y : ℝ := 2 - Real.sqrt 2

    theorem path_counts (n : ℕ) :
      ∃ α : ℕ → ℕ, (α (2 * n - 1) = 0) ∧ (α (2 * n) = (1 / Real.sqrt 2) * ((x ^ (n - 1)) - (y ^ (n - 1)))) :=
    by
      sorry
    
end NUMINAMATH_GPT_path_counts_l2229_222982


namespace NUMINAMATH_GPT_cost_of_magazine_l2229_222911

theorem cost_of_magazine (B M : ℝ) 
  (h1 : 2 * B + 2 * M = 26) 
  (h2 : B + 3 * M = 27) : 
  M = 7 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_magazine_l2229_222911


namespace NUMINAMATH_GPT_contrapositive_proof_l2229_222952

theorem contrapositive_proof (m : ℕ) (h_pos : 0 < m) :
  (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

end NUMINAMATH_GPT_contrapositive_proof_l2229_222952


namespace NUMINAMATH_GPT_remaining_water_l2229_222901

def initial_water : ℚ := 3
def water_used : ℚ := 4 / 3

theorem remaining_water : initial_water - water_used = 5 / 3 := 
by sorry -- skipping the proof for now

end NUMINAMATH_GPT_remaining_water_l2229_222901


namespace NUMINAMATH_GPT_min_value_of_a3_l2229_222927

open Real

theorem min_value_of_a3 (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n) (hgeo : ∀ n, a (n + 1) / a n = a 1 / a 0)
    (h : a 1 * a 2 * a 3 = a 1 + a 2 + a 3) : a 2 ≥ sqrt 3 := by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_a3_l2229_222927


namespace NUMINAMATH_GPT_wheat_flour_packets_correct_l2229_222967

-- Define the initial amount of money Victoria had.
def initial_amount : ℕ := 500

-- Define the cost and quantity of rice packets Victoria bought.
def rice_packet_cost : ℕ := 20
def rice_packets : ℕ := 2

-- Define the cost and quantity of soda Victoria bought.
def soda_cost : ℕ := 150
def soda_quantity : ℕ := 1

-- Define the remaining balance after shopping.
def remaining_balance : ℕ := 235

-- Define the cost of one packet of wheat flour.
def wheat_flour_packet_cost : ℕ := 25

-- Define the total amount spent on rice and soda.
def total_spent_on_rice_and_soda : ℕ :=
  (rice_packets * rice_packet_cost) + (soda_quantity * soda_cost)

-- Define the total amount spent on wheat flour.
def total_spent_on_wheat_flour : ℕ :=
  initial_amount - remaining_balance - total_spent_on_rice_and_soda

-- Define the expected number of wheat flour packets bought.
def wheat_flour_packets_expected : ℕ := 3

-- The statement we want to prove: the number of wheat flour packets bought is 3.
theorem wheat_flour_packets_correct : total_spent_on_wheat_flour / wheat_flour_packet_cost = wheat_flour_packets_expected :=
  sorry

end NUMINAMATH_GPT_wheat_flour_packets_correct_l2229_222967


namespace NUMINAMATH_GPT_f_even_f_increasing_f_range_l2229_222934

variables {R : Type*} [OrderedRing R] (f : R → R)

-- Conditions
axiom f_mul : ∀ x y : R, f (x * y) = f x * f y
axiom f_neg1 : f (-1) = 1
axiom f_27 : f 27 = 9
axiom f_lt_1 : ∀ x : R, 0 ≤ x → x < 1 → 0 ≤ f x ∧ f x < 1

-- Questions
theorem f_even (x : R) : f x = f (-x) :=
by sorry

theorem f_increasing (x1 x2 : R) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 < x2) : f x1 < f x2 :=
by sorry

theorem f_range (a : R) (h1 : 0 ≤ a) (h2 : f (a + 1) ≤ 39) : 0 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_f_even_f_increasing_f_range_l2229_222934


namespace NUMINAMATH_GPT_unique_set_of_consecutive_integers_l2229_222990

theorem unique_set_of_consecutive_integers (a b c : ℕ) : 
  (a + b + c = 36) ∧ (b = a + 1) ∧ (c = a + 2) → 
  ∃! a : ℕ, (a = 11 ∧ b = 12 ∧ c = 13) := 
sorry

end NUMINAMATH_GPT_unique_set_of_consecutive_integers_l2229_222990


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2229_222922

variable (a : ℕ → ℝ) (d : ℝ)
-- Conditions
def is_arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def condition : Prop := a 4 + a 8 = 8

-- Question
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a d →
  condition a →
  (11 / 2) * (a 1 + a 11) = 44 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2229_222922


namespace NUMINAMATH_GPT_product_of_points_is_correct_l2229_222973

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map f |> List.sum

def AlexRolls := [6, 4, 3, 2, 1]
def BobRolls := [5, 6, 2, 3, 3]

def AlexPoints := totalPoints AlexRolls
def BobPoints := totalPoints BobRolls

theorem product_of_points_is_correct : AlexPoints * BobPoints = 672 := by
  sorry

end NUMINAMATH_GPT_product_of_points_is_correct_l2229_222973


namespace NUMINAMATH_GPT_car_mpg_city_l2229_222917

theorem car_mpg_city (h c t : ℕ) (H1 : 560 = h * t) (H2 : 336 = c * t) (H3 : c = h - 6) : c = 9 :=
by
  sorry

end NUMINAMATH_GPT_car_mpg_city_l2229_222917


namespace NUMINAMATH_GPT_simple_interest_rate_l2229_222947

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 800) (hSI : SI = 128) (hT : T = 4) : 
  (SI = P * (R : ℝ) * T / 100) → R = 4 := 
by {
  -- Proof goes here.
  sorry
}

end NUMINAMATH_GPT_simple_interest_rate_l2229_222947


namespace NUMINAMATH_GPT_intersection_P_Q_l2229_222951

-- Definitions and Conditions
variable (P Q : Set ℕ)
noncomputable def f (t : ℕ) : ℕ := t ^ 2
axiom hQ : Q = {1, 4}

-- Theorem to Prove
theorem intersection_P_Q (P : Set ℕ) (Q : Set ℕ) (hQ : Q = {1, 4})
  (hf : ∀ t ∈ P, f t ∈ Q) : P ∩ Q = {1} ∨ P ∩ Q = ∅ :=
sorry

end NUMINAMATH_GPT_intersection_P_Q_l2229_222951


namespace NUMINAMATH_GPT_proposition_D_l2229_222969

/-- Lean statement for proving the correct proposition D -/
theorem proposition_D {a b : ℝ} (h : |a| < b) : a^2 < b^2 :=
sorry

end NUMINAMATH_GPT_proposition_D_l2229_222969


namespace NUMINAMATH_GPT_problem1_problem2_part1_problem2_part2_l2229_222944

-- Problem 1
theorem problem1 (x : ℚ) (h : x = 11 / 12) : 
  (2 * x - 5) * (2 * x + 5) - (2 * x - 3) ^ 2 = -23 := 
by sorry

-- Problem 2
theorem problem2_part1 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  a^2 + b^2 = 22 := 
by sorry

theorem problem2_part2 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  (a - b)^2 = 8 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_part1_problem2_part2_l2229_222944


namespace NUMINAMATH_GPT_total_population_l2229_222962

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end NUMINAMATH_GPT_total_population_l2229_222962


namespace NUMINAMATH_GPT_train_speed_l2229_222970

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 250
noncomputable def crossing_time : ℝ := 28.79769618430526

noncomputable def speed_m_per_s : ℝ := (train_length + bridge_length) / crossing_time
noncomputable def speed_kmph : ℝ := speed_m_per_s * 3.6

theorem train_speed : speed_kmph = 50 := by
  sorry

end NUMINAMATH_GPT_train_speed_l2229_222970


namespace NUMINAMATH_GPT_total_trees_in_gray_areas_l2229_222941

theorem total_trees_in_gray_areas (white_region_first : ℕ) (white_region_second : ℕ)
    (total_first : ℕ) (total_second : ℕ)
    (h1 : white_region_first = 82) (h2 : white_region_second = 82)
    (h3 : total_first = 100) (h4 : total_second = 90) :
  (total_first - white_region_first) + (total_second - white_region_second) = 26 := by
  sorry

end NUMINAMATH_GPT_total_trees_in_gray_areas_l2229_222941


namespace NUMINAMATH_GPT_eggs_in_each_basket_l2229_222977

theorem eggs_in_each_basket (n : ℕ) (h₁ : 5 ≤ n) (h₂ : n ∣ 30) (h₃ : n ∣ 42) : n = 6 :=
sorry

end NUMINAMATH_GPT_eggs_in_each_basket_l2229_222977


namespace NUMINAMATH_GPT_digit_encoding_problem_l2229_222928

theorem digit_encoding_problem :
  ∃ (A B : ℕ), 0 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 21 * A + B = 111 * B ∧ A = 5 ∧ B = 5 :=
by
  sorry

end NUMINAMATH_GPT_digit_encoding_problem_l2229_222928


namespace NUMINAMATH_GPT_solve_system_l2229_222909

def inequality1 (x : ℝ) : Prop := 5 / (x + 3) ≥ 1

def inequality2 (x : ℝ) : Prop := x^2 + x - 2 ≥ 0

def solution (x : ℝ) : Prop := (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)

theorem solve_system (x : ℝ) : inequality1 x ∧ inequality2 x → solution x := by
  sorry

end NUMINAMATH_GPT_solve_system_l2229_222909


namespace NUMINAMATH_GPT_area_of_triangle_BQW_l2229_222946

theorem area_of_triangle_BQW (AZ WC AB : ℝ) (h_trap_area : ℝ) (h_eq : AZ = WC) (AZ_val : AZ = 8) (AB_val : AB = 16) (trap_area_val : h_trap_area = 160) : 
  ∃ (BQW_area: ℝ), BQW_area = 48 :=
by
  let h_2 := 2 * h_trap_area / (AZ + AB)
  let h := AZ + h_2
  let BZW_area := h_trap_area - (1 / 2) * AZ * AB
  let BQW_area := 1 / 2 * BZW_area
  have AZ_eq : AZ = 8 := AZ_val
  have AB_eq : AB = 16 := AB_val
  have trap_area_eq : h_trap_area = 160 := trap_area_val
  let h_2_val : ℝ := 10 -- Calculated from h_2 = 2 * 160 / 32
  let h_val : ℝ := AZ + h_2_val -- full height
  let BZW_area_val : ℝ := 96 -- BZW area from 160 - 64
  let BQW_area_val : ℝ := 48 -- Half of BZW
  exact ⟨48, by sorry⟩ -- To complete the theorem

end NUMINAMATH_GPT_area_of_triangle_BQW_l2229_222946


namespace NUMINAMATH_GPT_bronze_medals_l2229_222902

theorem bronze_medals (G S B : ℕ) 
  (h1 : G + S + B = 89) 
  (h2 : G + S = 4 * B - 6) :
  B = 19 :=
sorry

end NUMINAMATH_GPT_bronze_medals_l2229_222902


namespace NUMINAMATH_GPT_find_asymptotes_l2229_222930

def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

def shifted_hyperbola_asymptotes (x y : ℝ) : Prop :=
  y = 4 / 3 * x + 5 ∨ y = -4 / 3 * x + 5

theorem find_asymptotes (x y : ℝ) :
  (∃ y', y = y' + 5 ∧ hyperbola_eq x y')
  ↔ shifted_hyperbola_asymptotes x y :=
by
  sorry

end NUMINAMATH_GPT_find_asymptotes_l2229_222930


namespace NUMINAMATH_GPT_goat_cow_difference_l2229_222925

-- Given the number of pigs (P), cows (C), and goats (G) on a farm
variables (P C G : ℕ)

-- Conditions:
def pig_count := P = 10
def cow_count_relationship := C = 2 * P - 3
def total_animals := P + C + G = 50

-- Theorem: The difference between the number of goats and cows
theorem goat_cow_difference (h1 : pig_count P)
                           (h2 : cow_count_relationship P C)
                           (h3 : total_animals P C G) :
  G - C = 6 := 
  sorry

end NUMINAMATH_GPT_goat_cow_difference_l2229_222925


namespace NUMINAMATH_GPT_difference_in_money_in_nickels_l2229_222961

-- Define the given conditions
def alice_quarters (p : ℕ) : ℕ := 3 * p + 2
def bob_quarters (p : ℕ) : ℕ := 2 * p + 8

-- Define the difference in their money in nickels
def difference_in_nickels (p : ℕ) : ℕ := 5 * (p - 6)

-- The proof problem statement
theorem difference_in_money_in_nickels (p : ℕ) : 
  (5 * (alice_quarters p - bob_quarters p)) = difference_in_nickels p :=
by 
  sorry

end NUMINAMATH_GPT_difference_in_money_in_nickels_l2229_222961


namespace NUMINAMATH_GPT_remaining_volume_of_cube_with_hole_l2229_222940

theorem remaining_volume_of_cube_with_hole : 
  let side_length_cube := 8 
  let side_length_hole := 4 
  let volume_cube := side_length_cube ^ 3 
  let cross_section_hole := side_length_hole ^ 2
  let volume_hole := cross_section_hole * side_length_cube
  let remaining_volume := volume_cube - volume_hole
  remaining_volume = 384 := by {
    sorry
  }

end NUMINAMATH_GPT_remaining_volume_of_cube_with_hole_l2229_222940


namespace NUMINAMATH_GPT_sum_of_areas_of_rectangles_l2229_222932

theorem sum_of_areas_of_rectangles :
  let width := 2
  let lengths := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => l * width)
  let total_area := areas.sum
  total_area = 182 := by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_rectangles_l2229_222932


namespace NUMINAMATH_GPT_infinite_product_eq_four_four_thirds_l2229_222937

theorem infinite_product_eq_four_four_thirds :
  ∏' n : ℕ, (4^(n+1)^(1/(2^(n+1)))) = 4^(4/3) :=
sorry

end NUMINAMATH_GPT_infinite_product_eq_four_four_thirds_l2229_222937


namespace NUMINAMATH_GPT_wolf_hunger_if_eats_11_kids_l2229_222975

variable (p k : ℝ)  -- Define the satiety values of a piglet and a kid.
variable (H : ℝ)    -- Define the satiety threshold for "enough to remove hunger".

-- Conditions from the problem:
def condition1 : Prop := 3 * p + 7 * k < H  -- The wolf feels hungry after eating 3 piglets and 7 kids.
def condition2 : Prop := 7 * p + k > H      -- The wolf suffers from overeating after eating 7 piglets and 1 kid.

-- Statement to prove:
theorem wolf_hunger_if_eats_11_kids (p k H : ℝ) 
  (h1 : condition1 p k H) (h2 : condition2 p k H) : 11 * k < H :=
by
  sorry

end NUMINAMATH_GPT_wolf_hunger_if_eats_11_kids_l2229_222975


namespace NUMINAMATH_GPT_carly_lollipops_total_l2229_222966

theorem carly_lollipops_total (C : ℕ) (h1 : C / 2 = cherry_lollipops)
  (h2 : C / 2 = 3 * 7) : C = 42 :=
by
  sorry

end NUMINAMATH_GPT_carly_lollipops_total_l2229_222966


namespace NUMINAMATH_GPT_clock_palindromes_l2229_222995

theorem clock_palindromes : 
  let valid_hours := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22]
  let valid_minutes := [0, 1, 2, 3, 4, 5]
  let two_digit_palindromes := 9 * 6
  let four_digit_palindromes := 6
  (two_digit_palindromes + four_digit_palindromes) = 60 := 
by
  sorry

end NUMINAMATH_GPT_clock_palindromes_l2229_222995


namespace NUMINAMATH_GPT_find_length_QT_l2229_222956

noncomputable def length_RS : ℝ := 75
noncomputable def length_PQ : ℝ := 36
noncomputable def length_PT : ℝ := 12

theorem find_length_QT :
  ∀ (PQRS : Type)
  (P Q R S T : PQRS)
  (h_RS_perp_PQ : true)
  (h_PQ_perp_RS : true)
  (h_PT_perpendicular_to_PR : true),
  QT = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_length_QT_l2229_222956


namespace NUMINAMATH_GPT_parallelogram_side_length_l2229_222924

theorem parallelogram_side_length (s : ℝ) (h : 3 * s * s * (1 / 2) = 27 * Real.sqrt 3) : 
  s = 3 * Real.sqrt (2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_parallelogram_side_length_l2229_222924


namespace NUMINAMATH_GPT_general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l2229_222923
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 3*n - 1
noncomputable def c_n (n : ℕ) : ℚ := (3*n - 1) / 2^(n-1)

-- 1. Prove that the sequence {a_n} is given by a_n = 2^(n-1) and {b_n} is given by b_n = 3n - 1
theorem general_formulas :
  (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) ∧
  (∀ n : ℕ, n > 0 → b_n n = 3*n - 1) :=
sorry

-- 2. Prove that the values of n for which c_n > 1 are n = 1, 2, 3, 4
theorem values_of_n_for_c_n_gt_one :
  { n : ℕ | n > 0 ∧ c_n n > 1 } = {1, 2, 3, 4} :=
sorry

-- 3. Prove that no three terms from {a_n} can form an arithmetic sequence
theorem no_three_terms_arithmetic_seq :
  ∀ p q r : ℕ, p < q ∧ q < r ∧ p > 0 ∧ q > 0 ∧ r > 0 →
  ¬ (2 * a_n q = a_n p + a_n r) :=
sorry

end NUMINAMATH_GPT_general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l2229_222923


namespace NUMINAMATH_GPT_find_a_l2229_222979

theorem find_a (a : ℝ) :
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 :=
sorry

end NUMINAMATH_GPT_find_a_l2229_222979


namespace NUMINAMATH_GPT_inversions_range_l2229_222933

/-- Given any permutation of 10 elements, 
    the number of inversions (or disorders) in the permutation 
    can take any value from 0 to 45.
-/
theorem inversions_range (perm : List ℕ) (h_length : perm.length = 10):
  ∃ S, 0 ≤ S ∧ S ≤ 45 :=
sorry

end NUMINAMATH_GPT_inversions_range_l2229_222933


namespace NUMINAMATH_GPT_quadratic_transformation_l2229_222950

theorem quadratic_transformation (a b c : ℝ) (h : a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) : 
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end NUMINAMATH_GPT_quadratic_transformation_l2229_222950


namespace NUMINAMATH_GPT_inequality_solution_l2229_222910

theorem inequality_solution (x : ℝ) : 
  (x < -4 ∨ x > 2) ↔ (x^2 + 3 * x - 4) / (x^2 - x - 2) > 0 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2229_222910


namespace NUMINAMATH_GPT_Cole_drive_time_to_work_l2229_222907

theorem Cole_drive_time_to_work :
  ∀ (D T_work T_home : ℝ),
    (T_work = D / 80) →
    (T_home = D / 120) →
    (T_work + T_home = 3) →
    (T_work * 60 = 108) :=
by
  intros D T_work T_home h1 h2 h3
  sorry

end NUMINAMATH_GPT_Cole_drive_time_to_work_l2229_222907


namespace NUMINAMATH_GPT_largest_angle_of_convex_pentagon_l2229_222964

theorem largest_angle_of_convex_pentagon :
  ∀ (x : ℝ), (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  5 * (104 / 3 : ℝ) + 6 = 538 / 3 := 
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_largest_angle_of_convex_pentagon_l2229_222964


namespace NUMINAMATH_GPT_find_sum_u_v_l2229_222939

theorem find_sum_u_v : ∃ (u v : ℚ), 5 * u - 6 * v = 35 ∧ 3 * u + 5 * v = -10 ∧ u + v = -40 / 43 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_u_v_l2229_222939


namespace NUMINAMATH_GPT_mark_sprinted_distance_l2229_222981

def speed := 6 -- miles per hour
def time := 4 -- hours

/-- Mark sprinted exactly 24 miles. -/
theorem mark_sprinted_distance : speed * time = 24 := by
  sorry

end NUMINAMATH_GPT_mark_sprinted_distance_l2229_222981


namespace NUMINAMATH_GPT_clips_ratio_l2229_222999

def clips (April May: Nat) : Prop :=
  April = 48 ∧ April + May = 72 → (48 / (72 - 48)) = 2

theorem clips_ratio : clips 48 (72 - 48) :=
by
  sorry

end NUMINAMATH_GPT_clips_ratio_l2229_222999


namespace NUMINAMATH_GPT_total_jumps_l2229_222957

-- Definitions based on given conditions
def Ronald_jumps : ℕ := 157
def Rupert_jumps : ℕ := Ronald_jumps + 86

-- The theorem we want to prove
theorem total_jumps : Ronald_jumps + Rupert_jumps = 400 :=
by
  sorry

end NUMINAMATH_GPT_total_jumps_l2229_222957


namespace NUMINAMATH_GPT_quadrilateral_side_length_l2229_222974

theorem quadrilateral_side_length (r a b c x : ℝ) (h_radius : r = 100 * Real.sqrt 6) 
    (h_a : a = 100) (h_b : b = 200) (h_c : c = 200) :
    x = 100 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_quadrilateral_side_length_l2229_222974


namespace NUMINAMATH_GPT_sequence_decreasing_l2229_222916

theorem sequence_decreasing : 
  ∀ (n : ℕ), n ≥ 1 → (1 / 2^(n - 1)) > (1 / 2^n) := 
by {
  sorry
}

end NUMINAMATH_GPT_sequence_decreasing_l2229_222916


namespace NUMINAMATH_GPT_intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l2229_222984

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 3

theorem intervals_of_increase_decrease_a_neg1 : 
  ∀ x : ℝ, quadratic_function (-1) x = x^2 - 2 * x + 3 → 
  (∀ x ≥ 1, quadratic_function (-1) x ≥ quadratic_function (-1) 1) ∧ 
  (∀ x ≤ 1, quadratic_function (-1) x ≤ quadratic_function (-1) 1) :=
  sorry

theorem max_min_values_a_neg2 :
  ∃ min : ℝ, min = -1 ∧ (∀ x : ℝ, quadratic_function (-2) x ≥ min) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y > x → quadratic_function (-2) y > quadratic_function (-2) x) :=
  sorry

theorem no_a_for_monotonic_function : 
  ∀ a : ℝ, ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≤ quadratic_function a y) ∧ ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≥ quadratic_function a y) :=
  sorry

end NUMINAMATH_GPT_intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l2229_222984


namespace NUMINAMATH_GPT_geometric_sequence_a5_value_l2229_222994

-- Definition of geometric sequence and the specific condition a_3 * a_7 = 8
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (geom_seq : is_geometric_sequence a)
  (cond : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_value_l2229_222994


namespace NUMINAMATH_GPT_number_in_sequence_l2229_222972

theorem number_in_sequence : ∃ n : ℕ, n * (n + 2) = 99 :=
by
  sorry

end NUMINAMATH_GPT_number_in_sequence_l2229_222972


namespace NUMINAMATH_GPT_bus_people_final_count_l2229_222963

theorem bus_people_final_count (initial_people : ℕ) (people_on : ℤ) (people_off : ℤ) :
  initial_people = 22 → people_on = 4 → people_off = -8 → initial_people + people_on + people_off = 18 :=
by
  intro h_initial h_on h_off
  rw [h_initial, h_on, h_off]
  norm_num

end NUMINAMATH_GPT_bus_people_final_count_l2229_222963


namespace NUMINAMATH_GPT_last_non_zero_digit_of_40_l2229_222954

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def last_non_zero_digit (n : ℕ) : ℕ :=
  let p := factorial n
  let digits : List ℕ := List.filter (λ d => d ≠ 0) (p.digits 10)
  digits.headD 0

theorem last_non_zero_digit_of_40 : last_non_zero_digit 40 = 6 := by
  sorry

end NUMINAMATH_GPT_last_non_zero_digit_of_40_l2229_222954


namespace NUMINAMATH_GPT_percentage_increase_l2229_222968

theorem percentage_increase (M N : ℝ) (h : M ≠ N) : 
  (200 * (M - N) / (M + N) = ((200 : ℝ) * (M - N) / (M + N))) :=
by
  -- Translate the problem conditions into Lean definitions
  let average := (M + N) / 2
  let increase := (M - N)
  let fraction_of_increase_over_average := (increase / average) * 100

  -- Additional annotations and calculations to construct the proof would go here
  sorry

end NUMINAMATH_GPT_percentage_increase_l2229_222968


namespace NUMINAMATH_GPT_number_of_workers_l2229_222996

theorem number_of_workers (supervisors team_leads_per_supervisor workers_per_team_lead : ℕ) 
    (h_supervisors : supervisors = 13)
    (h_team_leads_per_supervisor : team_leads_per_supervisor = 3)
    (h_workers_per_team_lead : workers_per_team_lead = 10):
    supervisors * team_leads_per_supervisor * workers_per_team_lead = 390 :=
by
  -- to avoid leaving the proof section empty and potentially creating an invalid Lean statement
  sorry

end NUMINAMATH_GPT_number_of_workers_l2229_222996
