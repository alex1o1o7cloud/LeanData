import Mathlib

namespace NUMINAMATH_GPT_sum_kml_l2213_221310

theorem sum_kml (k m l : ℤ) (b : ℤ → ℤ)
  (h_seq : ∀ n, ∃ k, b n = k * (Int.floor (Real.sqrt (n + m : ℝ))) + l)
  (h_b1 : b 1 = 2) :
  k + m + l = 3 := by
  sorry

end NUMINAMATH_GPT_sum_kml_l2213_221310


namespace NUMINAMATH_GPT_domain_of_f_l2213_221319

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (-x^2 + 9 * x + 10)) / Real.log (x - 1)

theorem domain_of_f :
  {x : ℝ | -x^2 + 9 * x + 10 ≥ 0 ∧ x - 1 > 0 ∧ Real.log (x - 1) ≠ 0} =
  {x : ℝ | (1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 10)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2213_221319


namespace NUMINAMATH_GPT_katy_summer_reading_total_l2213_221378

def katy_books_in_summer (june_books july_books august_books : ℕ) : ℕ := june_books + july_books + august_books

theorem katy_summer_reading_total (june_books : ℕ) (july_books : ℕ) (august_books : ℕ) 
  (h1 : june_books = 8)
  (h2 : july_books = 2 * june_books)
  (h3 : august_books = july_books - 3) :
  katy_books_in_summer june_books july_books august_books = 37 :=
by
  sorry

end NUMINAMATH_GPT_katy_summer_reading_total_l2213_221378


namespace NUMINAMATH_GPT_quadratic_root_relationship_l2213_221305

theorem quadratic_root_relationship
  (m1 m2 : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h_eq1 : m1 * x1^2 + (1 / 3) * x1 + 1 = 0)
  (h_eq2 : m1 * x2^2 + (1 / 3) * x2 + 1 = 0)
  (h_eq3 : m2 * x3^2 + (1 / 3) * x3 + 1 = 0)
  (h_eq4 : m2 * x4^2 + (1 / 3) * x4 + 1 = 0)
  (h_order : x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ x2 < 0) :
  m2 > m1 ∧ m1 > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_root_relationship_l2213_221305


namespace NUMINAMATH_GPT_find_y_l2213_221361

theorem find_y (x y : ℤ) (h₁ : x = 4) (h₂ : 3 * x + 2 * y = 30) : y = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l2213_221361


namespace NUMINAMATH_GPT_minimum_value_of_m_minus_n_l2213_221327

def f (x : ℝ) : ℝ := (x - 1) ^ 2

theorem minimum_value_of_m_minus_n 
  (f_even : ∀ x : ℝ, f x = f (-x))
  (condition1 : n ≤ f (-2))
  (condition2 : n ≤ f (-1 / 2))
  (condition3 : f (-2) ≤ m)
  (condition4 : f (-1 / 2) ≤ m)
  : ∃ n m, m - n = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_m_minus_n_l2213_221327


namespace NUMINAMATH_GPT_units_digit_G100_l2213_221334

def G (n : ℕ) := 3 * 2 ^ (2 ^ n) + 2

theorem units_digit_G100 : (G 100) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_G100_l2213_221334


namespace NUMINAMATH_GPT_sum_of_solutions_eq_eight_l2213_221353

theorem sum_of_solutions_eq_eight : 
  ∀ x : ℝ, (x^2 - 6 * x + 5 = 2 * x - 7) → (∃ a b : ℝ, (a = 6) ∧ (b = 2) ∧ (a + b = 8)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_eight_l2213_221353


namespace NUMINAMATH_GPT_min_width_of_garden_l2213_221331

theorem min_width_of_garden (w : ℝ) (h : 0 < w) (h1 : w * (w + 20) ≥ 120) : w ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_width_of_garden_l2213_221331


namespace NUMINAMATH_GPT_gumball_machine_total_gumballs_l2213_221336

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end NUMINAMATH_GPT_gumball_machine_total_gumballs_l2213_221336


namespace NUMINAMATH_GPT_lemons_needed_l2213_221352

theorem lemons_needed (initial_lemons : ℝ) (initial_gallons : ℝ) 
  (reduced_ratio : ℝ) (first_gallons : ℝ) (total_gallons : ℝ) :
  initial_lemons / initial_gallons * first_gallons 
  + (initial_lemons / initial_gallons * reduced_ratio) * (total_gallons - first_gallons) = 56.25 :=
by 
  let initial_ratio := initial_lemons / initial_gallons
  let reduced_ratio_amount := initial_ratio * reduced_ratio 
  let lemons_first := initial_ratio * first_gallons
  let lemons_remaining := reduced_ratio_amount * (total_gallons - first_gallons)
  let total_lemons := lemons_first + lemons_remaining
  show total_lemons = 56.25
  sorry

end NUMINAMATH_GPT_lemons_needed_l2213_221352


namespace NUMINAMATH_GPT_M_inter_N_l2213_221382

def M : Set ℝ := {x | abs (x - 1) < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem M_inter_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_l2213_221382


namespace NUMINAMATH_GPT_annulus_area_l2213_221351

variables {R r d : ℝ}
variables (h1 : R > r) (h2 : d < R)

theorem annulus_area :
  π * (R^2 - r^2 - d^2 / (R - r)) = π * ((R - r)^2 - d^2) :=
sorry

end NUMINAMATH_GPT_annulus_area_l2213_221351


namespace NUMINAMATH_GPT_nacho_will_be_three_times_older_in_future_l2213_221320

variable (N D x : ℕ)
variable (h1 : D = 5)
variable (h2 : N + D = 40)
variable (h3 : N + x = 3 * (D + x))

theorem nacho_will_be_three_times_older_in_future :
  x = 10 :=
by {
  -- Given conditions
  sorry
}

end NUMINAMATH_GPT_nacho_will_be_three_times_older_in_future_l2213_221320


namespace NUMINAMATH_GPT_leah_birds_duration_l2213_221365

-- Define the conditions
def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def parrot_weekly_consumption : ℕ := 100
def cockatiel_weekly_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Define the question as a theorem
theorem leah_birds_duration : 
  (boxes_bought + boxes_existing) * grams_per_box / 
  (parrot_weekly_consumption + cockatiel_weekly_consumption) = 12 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_leah_birds_duration_l2213_221365


namespace NUMINAMATH_GPT_sin_2phi_l2213_221337

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_GPT_sin_2phi_l2213_221337


namespace NUMINAMATH_GPT_train_length_is_correct_l2213_221326

-- Define the given conditions and the expected result.
def train_speed_kmph : ℝ := 270
def time_seconds : ℝ := 5
def expected_length_meters : ℝ := 375

-- State the theorem to be proven.
theorem train_length_is_correct :
  (train_speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters := by
  sorry -- Proof is not required, so we use 'sorry'

end NUMINAMATH_GPT_train_length_is_correct_l2213_221326


namespace NUMINAMATH_GPT_people_visited_on_Sunday_l2213_221335

theorem people_visited_on_Sunday (ticket_price : ℕ) 
                                 (people_per_day_week : ℕ) 
                                 (people_on_Saturday : ℕ) 
                                 (total_revenue : ℕ) 
                                 (days_week : ℕ)
                                 (total_days : ℕ) 
                                 (people_per_day_mf : ℕ) 
                                 (people_on_other_days : ℕ) 
                                 (revenue_other_days : ℕ)
                                 (revenue_Sunday : ℕ)
                                 (people_Sunday : ℕ) :
    ticket_price = 3 →
    people_per_day_week = 100 →
    people_on_Saturday = 200 →
    total_revenue = 3000 →
    days_week = 5 →
    total_days = 7 →
    people_per_day_mf = people_per_day_week * days_week →
    people_on_other_days = people_per_day_mf + people_on_Saturday →
    revenue_other_days = people_on_other_days * ticket_price →
    revenue_Sunday = total_revenue - revenue_other_days →
    people_Sunday = revenue_Sunday / ticket_price →
    people_Sunday = 300 := 
by 
  sorry

end NUMINAMATH_GPT_people_visited_on_Sunday_l2213_221335


namespace NUMINAMATH_GPT_minimum_value_expr_l2213_221323

noncomputable def expr (x : ℝ) : ℝ := 9 * x + 3 / (x ^ 3)

theorem minimum_value_expr : (∀ x : ℝ, x > 0 → expr x ≥ 12) ∧ (∃ x : ℝ, x > 0 ∧ expr x = 12) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expr_l2213_221323


namespace NUMINAMATH_GPT_range_of_a_l2213_221358

def f (a x : ℝ) : ℝ := -x^3 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → -3 * x^2 + a ≥ 0) → a ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2213_221358


namespace NUMINAMATH_GPT_power_function_value_l2213_221388

/-- Given a power function passing through a certain point, find the value at a specific point -/
theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h : f x = x ^ α) 
  (h_passes : f (1/4) = 4) : f 2 = 1/2 :=
sorry

end NUMINAMATH_GPT_power_function_value_l2213_221388


namespace NUMINAMATH_GPT_bake_cookies_l2213_221390

noncomputable def scale_factor (original_cookies target_cookies : ℕ) : ℕ :=
  target_cookies / original_cookies

noncomputable def required_flour (original_flour : ℕ) (scale : ℕ) : ℕ :=
  original_flour * scale

noncomputable def adjusted_sugar (original_sugar : ℕ) (scale : ℕ) (reduction_percent : ℚ) : ℚ :=
  original_sugar * scale * (1 - reduction_percent)

theorem bake_cookies 
  (original_cookies : ℕ)
  (target_cookies : ℕ)
  (original_flour : ℕ)
  (original_sugar : ℕ)
  (reduction_percent : ℚ)
  (h_original_cookies : original_cookies = 40)
  (h_target_cookies : target_cookies = 80)
  (h_original_flour : original_flour = 3)
  (h_original_sugar : original_sugar = 1)
  (h_reduction_percent : reduction_percent = 0.25) :
  required_flour original_flour (scale_factor original_cookies target_cookies) = 6 ∧ 
  adjusted_sugar original_sugar (scale_factor original_cookies target_cookies) reduction_percent = 1.5 := by
    sorry

end NUMINAMATH_GPT_bake_cookies_l2213_221390


namespace NUMINAMATH_GPT_arithmetic_sequence_count_l2213_221332

theorem arithmetic_sequence_count :
  ∃! (n a d : ℕ), n ≥ 3 ∧ (n * (2 * a + (n - 1) * d) = 2 * 97^2) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_count_l2213_221332


namespace NUMINAMATH_GPT_cos_triple_angle_l2213_221349

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l2213_221349


namespace NUMINAMATH_GPT_am_gm_inequality_l2213_221379

variable (a : ℝ) (h : a > 0) -- Variables and condition

theorem am_gm_inequality (a : ℝ) (h : a > 0) : a + 1 / a ≥ 2 := 
sorry -- Proof is not provided according to instructions.

end NUMINAMATH_GPT_am_gm_inequality_l2213_221379


namespace NUMINAMATH_GPT_necessary_french_woman_l2213_221387

structure MeetingConditions where
  total_money_women : ℝ
  total_money_men : ℝ
  total_money_french : ℝ
  total_money_russian : ℝ

axiom no_other_representatives : Prop
axiom money_french_vs_russian (conditions : MeetingConditions) : conditions.total_money_french > conditions.total_money_russian
axiom money_women_vs_men (conditions : MeetingConditions) : conditions.total_money_women > conditions.total_money_men

theorem necessary_french_woman (conditions : MeetingConditions) :
  ∃ w_f : ℝ, w_f > 0 ∧ conditions.total_money_french > w_f ∧ w_f + conditions.total_money_men > conditions.total_money_women :=
by
  sorry

end NUMINAMATH_GPT_necessary_french_woman_l2213_221387


namespace NUMINAMATH_GPT_negation_example_l2213_221324

theorem negation_example :
  (¬ (∃ n : ℕ, n^2 ≥ 2^n)) → (∀ n : ℕ, n^2 < 2^n) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l2213_221324


namespace NUMINAMATH_GPT_product_of_triangle_areas_not_end_2014_l2213_221333

theorem product_of_triangle_areas_not_end_2014
  (T1 T2 T3 T4 : ℤ)
  (h1 : T1 > 0)
  (h2 : T2 > 0)
  (h3 : T3 > 0)
  (h4 : T4 > 0) :
  (T1 * T2 * T3 * T4) % 10000 ≠ 2014 := by
sorry

end NUMINAMATH_GPT_product_of_triangle_areas_not_end_2014_l2213_221333


namespace NUMINAMATH_GPT_eduardo_frankie_classes_total_l2213_221348

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end NUMINAMATH_GPT_eduardo_frankie_classes_total_l2213_221348


namespace NUMINAMATH_GPT_div_rule_2701_is_37_or_73_l2213_221372

theorem div_rule_2701_is_37_or_73 (a b x : ℕ) (h1 : 10 * a + b = x) (h2 : a^2 + b^2 = 58) : 
  (x = 37 ∨ x = 73) ↔ 2701 % x = 0 :=
by
  sorry

end NUMINAMATH_GPT_div_rule_2701_is_37_or_73_l2213_221372


namespace NUMINAMATH_GPT_smallest_k_for_factorial_divisibility_l2213_221321

theorem smallest_k_for_factorial_divisibility : 
  ∃ (k : ℕ), (∀ n : ℕ, n < k → ¬(2040 ∣ n!)) ∧ (2040 ∣ k!) ∧ k = 17 :=
by
  -- We skip the actual proof steps and provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_smallest_k_for_factorial_divisibility_l2213_221321


namespace NUMINAMATH_GPT_regular_polygon_sides_l2213_221346

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2213_221346


namespace NUMINAMATH_GPT_diagonal_ratio_of_squares_l2213_221389

theorem diagonal_ratio_of_squares (P d : ℝ) (h : ∃ s S, 4 * S = 4 * s * 4 ∧ P = 4 * s ∧ d = s * Real.sqrt 2) : 
    (∃ D, D = 4 * d) :=
by
  sorry

end NUMINAMATH_GPT_diagonal_ratio_of_squares_l2213_221389


namespace NUMINAMATH_GPT_choir_females_correct_l2213_221329

noncomputable def number_of_females_in_choir : ℕ :=
  let orchestra_males := 11
  let orchestra_females := 12
  let orchestra_musicians := orchestra_males + orchestra_females
  let band_males := 2 * orchestra_males
  let band_females := 2 * orchestra_females
  let band_musicians := 2 * orchestra_musicians
  let total_musicians := 98
  let choir_males := 12
  let choir_musicians := total_musicians - (orchestra_musicians + band_musicians)
  let choir_females := choir_musicians - choir_males
  choir_females

theorem choir_females_correct : number_of_females_in_choir = 17 := by
  sorry

end NUMINAMATH_GPT_choir_females_correct_l2213_221329


namespace NUMINAMATH_GPT_main_theorem_l2213_221318

noncomputable def f : ℝ → ℝ := sorry

axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → 
  (x1 < x2 ↔ (f x2 < f x1))

theorem main_theorem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l2213_221318


namespace NUMINAMATH_GPT_total_red_and_green_peaches_l2213_221344

-- Define the number of red peaches and green peaches.
def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

-- Theorem stating the sum of red and green peaches is 22.
theorem total_red_and_green_peaches : red_peaches + green_peaches = 22 := 
by
  -- Proof would go here but is not required
  sorry

end NUMINAMATH_GPT_total_red_and_green_peaches_l2213_221344


namespace NUMINAMATH_GPT_slower_train_time_to_pass_driver_faster_one_l2213_221311

noncomputable def convert_speed (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def relative_speed (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1 := convert_speed speed1_kmh
  let speed2 := convert_speed speed2_kmh
  speed1 + speed2

noncomputable def time_to_pass (length1_m length2_m speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relative_speed := relative_speed speed1_kmh speed2_kmh
  (length1_m + length2_m) / relative_speed

theorem slower_train_time_to_pass_driver_faster_one :
  ∀ (length1 length2 speed1 speed2 : ℝ),
    length1 = 900 → length2 = 900 →
    speed1 = 45 → speed2 = 30 →
    time_to_pass length1 length2 speed1 speed2 = 86.39 :=
by
  intros
  simp only [time_to_pass, relative_speed, convert_speed]
  sorry

end NUMINAMATH_GPT_slower_train_time_to_pass_driver_faster_one_l2213_221311


namespace NUMINAMATH_GPT_num_distinguishable_octahedrons_l2213_221307

-- Define the given conditions
def num_faces : ℕ := 8
def num_colors : ℕ := 8
def total_permutations : ℕ := Nat.factorial num_colors
def distinct_orientations : ℕ := 24

-- Prove the main statement
theorem num_distinguishable_octahedrons : total_permutations / distinct_orientations = 1680 :=
by
  sorry

end NUMINAMATH_GPT_num_distinguishable_octahedrons_l2213_221307


namespace NUMINAMATH_GPT_music_store_cellos_l2213_221354

/-- 
A certain music store stocks 600 violas. 
There are 100 cello-viola pairs, such that a cello and a viola were both made with wood from the same tree. 
The probability that the two instruments are made with wood from the same tree is 0.00020833333333333335. 
Prove that the store stocks 800 cellos.
-/
theorem music_store_cellos (V : ℕ) (P : ℕ) (Pr : ℚ) (C : ℕ) 
  (h1 : V = 600) 
  (h2 : P = 100) 
  (h3 : Pr = 0.00020833333333333335) 
  (h4 : Pr = P / (C * V)): C = 800 :=
by
  sorry

end NUMINAMATH_GPT_music_store_cellos_l2213_221354


namespace NUMINAMATH_GPT_proposition_C_l2213_221316

-- Given conditions
variables {a b : ℝ}

-- Proposition C is the correct one
theorem proposition_C (h : a^3 > b^3) : a > b := by
  sorry

end NUMINAMATH_GPT_proposition_C_l2213_221316


namespace NUMINAMATH_GPT_star_value_example_l2213_221395

def my_star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

theorem star_value_example : my_star 3 5 = 68 := 
by
  sorry

end NUMINAMATH_GPT_star_value_example_l2213_221395


namespace NUMINAMATH_GPT_construct_line_through_points_l2213_221399

-- Definitions of the conditions
def points_on_sheet (A B : ℝ × ℝ) : Prop := A ≠ B
def tool_constraints (ruler_length compass_max_opening distance_A_B : ℝ) : Prop :=
  distance_A_B > 2 * ruler_length ∧ distance_A_B > 2 * compass_max_opening

-- The main theorem statement
theorem construct_line_through_points (A B : ℝ × ℝ) (ruler_length compass_max_opening : ℝ) 
  (h_points : points_on_sheet A B) 
  (h_constraints : tool_constraints ruler_length compass_max_opening (dist A B)) : 
  ∃ line : ℝ × ℝ → Prop, line A ∧ line B :=
sorry

end NUMINAMATH_GPT_construct_line_through_points_l2213_221399


namespace NUMINAMATH_GPT_range_of_a_l2213_221393

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a = x^2 - x - 1) ↔ -1 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l2213_221393


namespace NUMINAMATH_GPT_sum_of_b_and_c_base7_l2213_221355

theorem sum_of_b_and_c_base7 (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
(h4 : A < 7) (h5 : B < 7) (h6 : C < 7) 
(h7 : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) 
: B + C = 6 ∨ B + C = 12 := sorry

end NUMINAMATH_GPT_sum_of_b_and_c_base7_l2213_221355


namespace NUMINAMATH_GPT_lauras_european_stamps_cost_l2213_221325

def stamp_cost (count : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  count * cost_per_stamp

def total_stamps_cost (stamps80 : ℕ) (stamps90 : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  stamp_cost stamps80 cost_per_stamp + stamp_cost stamps90 cost_per_stamp

def european_stamps_cost_80_90 :=
  total_stamps_cost 10 12 0.09 + total_stamps_cost 18 16 0.07

theorem lauras_european_stamps_cost : european_stamps_cost_80_90 = 4.36 :=
by
  sorry

end NUMINAMATH_GPT_lauras_european_stamps_cost_l2213_221325


namespace NUMINAMATH_GPT_solve_system_of_equations_l2213_221313

theorem solve_system_of_equations (x y : ℚ)
  (h1 : 15 * x + 24 * y = 18)
  (h2 : 24 * x + 15 * y = 63) :
  x = 46 / 13 ∧ y = -19 / 13 := 
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2213_221313


namespace NUMINAMATH_GPT_find_x_l2213_221328

theorem find_x (x : ℝ) : 
  0.65 * x = 0.20 * 682.50 → x = 210 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l2213_221328


namespace NUMINAMATH_GPT_math_problem_l2213_221384

theorem math_problem : -5 * (-6) - 2 * (-3 * (-7) + (-8)) = 4 := 
  sorry

end NUMINAMATH_GPT_math_problem_l2213_221384


namespace NUMINAMATH_GPT_alcohol_water_ratio_l2213_221350

theorem alcohol_water_ratio (V : ℝ) (hV_pos : V > 0) :
  let jar1_alcohol := (2 / 3) * V
  let jar1_water := (1 / 3) * V
  let jar2_alcohol := (3 / 2) * V
  let jar2_water := (1 / 2) * V
  let total_alcohol := jar1_alcohol + jar2_alcohol
  let total_water := jar1_water + jar2_water
  (total_alcohol / total_water) = (13 / 5) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l2213_221350


namespace NUMINAMATH_GPT_coin_flip_sequences_l2213_221370

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end NUMINAMATH_GPT_coin_flip_sequences_l2213_221370


namespace NUMINAMATH_GPT_recycling_points_l2213_221391

-- Define the statement
theorem recycling_points : 
  ∀ (C H L I : ℝ) (points_per_six_pounds : ℝ), 
  C = 28 → H = 4.5 → L = 3.25 → I = 8.75 → points_per_six_pounds = 1 / 6 →
  (⌊ C * points_per_six_pounds ⌋ + ⌊ I * points_per_six_pounds ⌋  + ⌊ H * points_per_six_pounds ⌋ + ⌊ L * points_per_six_pounds ⌋ = 5) :=
by
  intros C H L I pps hC hH hL hI hpps
  rw [hC, hH, hL, hI, hpps]
  simp
  sorry

end NUMINAMATH_GPT_recycling_points_l2213_221391


namespace NUMINAMATH_GPT_initial_seashells_l2213_221339

-- Definitions for the conditions
def seashells_given_to_Tim : ℕ := 13
def seashells_now : ℕ := 36

-- Proving the number of initially found seashells
theorem initial_seashells : seashells_now + seashells_given_to_Tim = 49 :=
by
  -- we omit the proof steps with sorry
  sorry

end NUMINAMATH_GPT_initial_seashells_l2213_221339


namespace NUMINAMATH_GPT_students_play_neither_l2213_221380

-- Define the conditions
def total_students : ℕ := 39
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Define a theorem that states the equivalent proof problem
theorem students_play_neither : 
  total_students - (football_players + long_tennis_players - both_players) = 10 := by
  sorry

end NUMINAMATH_GPT_students_play_neither_l2213_221380


namespace NUMINAMATH_GPT_tan_inequality_l2213_221340

open Real

theorem tan_inequality {x1 x2 : ℝ} 
  (h1 : 0 < x1 ∧ x1 < π / 2) 
  (h2 : 0 < x2 ∧ x2 < π / 2) 
  (h3 : x1 ≠ x2) : 
  (1 / 2 * (tan x1 + tan x2) > tan ((x1 + x2) / 2)) :=
sorry

end NUMINAMATH_GPT_tan_inequality_l2213_221340


namespace NUMINAMATH_GPT_store_owner_oil_l2213_221317

noncomputable def liters_of_oil (volume_per_bottle : ℕ) (number_of_bottles : ℕ) : ℕ :=
  (volume_per_bottle * number_of_bottles) / 1000

theorem store_owner_oil : liters_of_oil 200 20 = 4 := by
  sorry

end NUMINAMATH_GPT_store_owner_oil_l2213_221317


namespace NUMINAMATH_GPT_unfolded_paper_has_four_symmetrical_holes_l2213_221376

structure Paper :=
  (width : ℤ) (height : ℤ) (hole_x : ℤ) (hole_y : ℤ)

structure Fold :=
  (direction : String) (fold_line : ℤ)

structure UnfoldedPaper :=
  (holes : List (ℤ × ℤ))

-- Define the initial paper, folds, and punching
def initial_paper : Paper := {width := 4, height := 6, hole_x := 2, hole_y := 1}
def folds : List Fold := 
  [{direction := "bottom_to_top", fold_line := initial_paper.height / 2}, 
   {direction := "left_to_right", fold_line := initial_paper.width / 2}]
def punch : (ℤ × ℤ) := (initial_paper.hole_x, initial_paper.hole_y)

-- The theorem to prove the resulting unfolded paper
theorem unfolded_paper_has_four_symmetrical_holes (p : Paper) (fs : List Fold) (punch : ℤ × ℤ) :
  UnfoldedPaper :=
  { holes := [(1, 1), (1, 5), (3, 1), (3, 5)] } -- Four symmetrically placed holes.

end NUMINAMATH_GPT_unfolded_paper_has_four_symmetrical_holes_l2213_221376


namespace NUMINAMATH_GPT_heather_walked_distance_l2213_221381

theorem heather_walked_distance {H S : ℝ} (hH : H = 5) (hS : S = H + 1) (total_distance : ℝ) (time_delay_stacy : ℝ) (time_heather_meet : ℝ) :
  (total_distance = 30) → (time_delay_stacy = 0.4) → (time_heather_meet = (total_distance - S * time_delay_stacy) / (H + S)) →
  (H * time_heather_meet = 12.55) :=
by
  sorry

end NUMINAMATH_GPT_heather_walked_distance_l2213_221381


namespace NUMINAMATH_GPT_find_y_l2213_221342

variable (x y z : ℝ)

theorem find_y
    (h₀ : x + y + z = 150)
    (h₁ : x + 10 = y - 10)
    (h₂ : y - 10 = 3 * z) :
    y = 74.29 :=
by
    sorry

end NUMINAMATH_GPT_find_y_l2213_221342


namespace NUMINAMATH_GPT_min_shirts_to_save_l2213_221364

theorem min_shirts_to_save (x : ℕ) :
  (75 + 10 * x < if x < 30 then 15 * x else 14 * x) → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_min_shirts_to_save_l2213_221364


namespace NUMINAMATH_GPT_min_value_of_expression_l2213_221312

theorem min_value_of_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 1 < b) (h₃ : a + b = 2) :
  4 / a + 1 / (b - 1) = 9 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2213_221312


namespace NUMINAMATH_GPT_ratio_p_q_l2213_221343

theorem ratio_p_q 
  (total_amount : ℕ) 
  (amount_r : ℕ) 
  (ratio_q_r : ℕ × ℕ) 
  (total_amount_eq : total_amount = 1210) 
  (amount_r_eq : amount_r = 400) 
  (ratio_q_r_eq : ratio_q_r = (9, 10)) :
  ∃ (amount_p amount_q : ℕ), 
    total_amount = amount_p + amount_q + amount_r ∧ 
    (amount_q : ℕ) = 9 * (amount_r / 10) ∧ 
    (amount_p : ℕ) / (amount_q : ℕ) = 5 / 4 := 
by sorry

end NUMINAMATH_GPT_ratio_p_q_l2213_221343


namespace NUMINAMATH_GPT_real_solutions_l2213_221314

theorem real_solutions (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 7) →
  ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 3) * (x - 7) * (x - 3)) = 1 →
  x = 3 + Real.sqrt 3 ∨ x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l2213_221314


namespace NUMINAMATH_GPT_range_of_m_l2213_221304

def one_root_condition (m : ℝ) : Prop :=
  (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0

theorem range_of_m : {m : ℝ | (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0} = {m | m ≤ -2 ∨ m ≥ 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2213_221304


namespace NUMINAMATH_GPT_inequality_problem_l2213_221330

theorem inequality_problem
  (a b c d e : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : c ≤ d)
  (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l2213_221330


namespace NUMINAMATH_GPT_find_y_in_interval_l2213_221398

theorem find_y_in_interval :
  { y : ℝ | y^2 + 7 * y < 12 } = { y : ℝ | -9 < y ∧ y < 2 } :=
sorry

end NUMINAMATH_GPT_find_y_in_interval_l2213_221398


namespace NUMINAMATH_GPT_Deepak_age_l2213_221359

-- Define the current ages of Arun and Deepak
variable (A D : ℕ)

-- Define the conditions
def ratio_condition := A / D = 4 / 3
def future_age_condition := A + 6 = 26

-- Define the proof statement
theorem Deepak_age (h1 : ratio_condition A D) (h2 : future_age_condition A) : D = 15 :=
  sorry

end NUMINAMATH_GPT_Deepak_age_l2213_221359


namespace NUMINAMATH_GPT_remainder_a83_l2213_221357

def a_n (n : ℕ) : ℕ := 6^n + 8^n

theorem remainder_a83 (n : ℕ) : 
  a_n 83 % 49 = 35 := sorry

end NUMINAMATH_GPT_remainder_a83_l2213_221357


namespace NUMINAMATH_GPT_A_is_sufficient_but_not_necessary_for_D_l2213_221377

variable {A B C D : Prop}

-- Defining the conditions
axiom h1 : A → B
axiom h2 : B ↔ C
axiom h3 : C → D

-- Statement to be proven
theorem A_is_sufficient_but_not_necessary_for_D : (A → D) ∧ ¬(D → A) :=
  by
  sorry

end NUMINAMATH_GPT_A_is_sufficient_but_not_necessary_for_D_l2213_221377


namespace NUMINAMATH_GPT_max_value_l2213_221362

-- Definitions for conditions
variables {a b : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 2)

-- Statement of the theorem
theorem max_value : (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / a) + (1 / b) = 2 ∧ ∀ y : ℝ,
  (1 / y) * ((2 / (y * (3 * y - 1)⁻¹)) + 1) ≤ 25 / 8) :=
sorry

end NUMINAMATH_GPT_max_value_l2213_221362


namespace NUMINAMATH_GPT_find_m_l2213_221300

theorem find_m :
  ∃ m : ℕ, 264 * 391 % 100 = m ∧ 0 ≤ m ∧ m < 100 ∧ m = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2213_221300


namespace NUMINAMATH_GPT_exists_irrationals_floor_neq_l2213_221385

-- Define irrationality of a number
def irrational (x : ℝ) : Prop :=
  ¬ ∃ (r : ℚ), x = r

theorem exists_irrationals_floor_neq :
  ∃ (a b : ℝ), irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ 
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ :=
by
  sorry

end NUMINAMATH_GPT_exists_irrationals_floor_neq_l2213_221385


namespace NUMINAMATH_GPT_nth_equation_l2213_221356

-- Define the product of a list of integers
def prod_list (lst : List ℕ) : ℕ :=
  lst.foldl (· * ·) 1

-- Define the product of first n odd numbers
def prod_odds (n : ℕ) : ℕ :=
  prod_list (List.map (λ i => 2 * i - 1) (List.range n))

-- Define the product of the range from n+1 to 2n
def prod_range (n : ℕ) : ℕ :=
  prod_list (List.range' (n + 1) n)

-- The theorem to prove
theorem nth_equation (n : ℕ) (hn : 0 < n) : prod_range n = 2^n * prod_odds n := 
  sorry

end NUMINAMATH_GPT_nth_equation_l2213_221356


namespace NUMINAMATH_GPT_largest_four_digit_perfect_square_l2213_221397

theorem largest_four_digit_perfect_square :
  ∃ (n : ℕ), n = 9261 ∧ (∃ k : ℕ, k * k = n) ∧ ∀ (m : ℕ), m < 10000 → (∃ x, x * x = m) → m ≤ n := 
by 
  sorry

end NUMINAMATH_GPT_largest_four_digit_perfect_square_l2213_221397


namespace NUMINAMATH_GPT_new_difference_l2213_221367

theorem new_difference (x y a : ℝ) (h : x - y = a) : (x + 0.5) - y = a + 0.5 := 
sorry

end NUMINAMATH_GPT_new_difference_l2213_221367


namespace NUMINAMATH_GPT_seq_b_is_geometric_l2213_221302

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence {a_n} with first term a_1 and common ratio q
def a_n (a₁ q : α) (n : ℕ) : α := a₁ * q^(n-1)

-- Define the sequence {b_n}
def b_n (a₁ q : α) (n : ℕ) : α :=
  a_n a₁ q (3*n - 2) + a_n a₁ q (3*n - 1) + a_n a₁ q (3*n)

-- Theorem stating {b_n} is a geometric sequence with common ratio q^3
theorem seq_b_is_geometric (a₁ q : α) (h : q ≠ 1) :
  ∀ n : ℕ, b_n a₁ q (n + 1) = q^3 * b_n a₁ q n :=
by
  sorry

end NUMINAMATH_GPT_seq_b_is_geometric_l2213_221302


namespace NUMINAMATH_GPT_probability_of_triangle_or_circle_l2213_221366

-- Definitions (conditions)
def total_figures : ℕ := 12
def triangles : ℕ := 4
def circles : ℕ := 3
def squares : ℕ := 5
def figures : ℕ := triangles + circles + squares

-- Probability calculation
def probability_triangle_circle := (triangles + circles) / total_figures

-- Theorem statement (problem)
theorem probability_of_triangle_or_circle : probability_triangle_circle = 7 / 12 :=
by
  -- The proof is omitted, insert the proof here when necessary.
  sorry

end NUMINAMATH_GPT_probability_of_triangle_or_circle_l2213_221366


namespace NUMINAMATH_GPT_exists_linear_function_second_quadrant_l2213_221306

theorem exists_linear_function_second_quadrant (k b : ℝ) (h1 : k > 0) (h2 : b > 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = k * x + b) ∧ (∀ x, x < 0 → f x > 0) :=
by
  -- Prove there exists a linear function of the form f(x) = kx + b with given conditions
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_exists_linear_function_second_quadrant_l2213_221306


namespace NUMINAMATH_GPT_divide_decimals_l2213_221303

theorem divide_decimals : (0.24 / 0.006) = 40 := by
  sorry

end NUMINAMATH_GPT_divide_decimals_l2213_221303


namespace NUMINAMATH_GPT_team_E_has_not_played_against_team_B_l2213_221345

-- We begin by defining the teams as an enumeration
inductive Team
| A | B | C | D | E | F

open Team

-- Define the total number of matches each team has played
def matches_played (t : Team) : Nat :=
  match t with
  | A => 5
  | B => 4
  | C => 3
  | D => 2
  | E => 1
  | F => 0 -- Note: we assume F's matches are not provided; this can be adjusted if needed

-- Prove that team E has not played against team B
theorem team_E_has_not_played_against_team_B :
  ∃ t : Team, matches_played B = 4 ∧ matches_played E < matches_played B ∧
  (t = E) :=
by
  sorry

end NUMINAMATH_GPT_team_E_has_not_played_against_team_B_l2213_221345


namespace NUMINAMATH_GPT_smallest_n_for_three_nested_rectangles_l2213_221308

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  x : ℕ
  y : ℕ
  h1 : 1 ≤ x
  h2 : x ≤ y
  h3 : y ≤ 100

/-- Define the nesting relation between rectangles -/
def nested (R1 R2 : Rectangle) : Prop :=
  R1.x < R2.x ∧ R1.y < R2.y

/-- Prove the smallest n such that there exist 3 nested rectangles out of n rectangles where n = 101 -/
theorem smallest_n_for_three_nested_rectangles (n : ℕ) (h : n ≥ 101) :
  ∀ (rectangles : Fin n → Rectangle), 
    ∃ (R1 R2 R3 : Fin n), nested (rectangles R1) (rectangles R2) ∧ nested (rectangles R2) (rectangles R3) :=
  sorry

end NUMINAMATH_GPT_smallest_n_for_three_nested_rectangles_l2213_221308


namespace NUMINAMATH_GPT_correct_statements_count_l2213_221315

theorem correct_statements_count :
  (∃ n : ℕ, odd_positive_integer = 4 * n + 1 ∨ odd_positive_integer = 4 * n + 3) ∧
  (∀ k : ℕ, k = 3 * m ∨ k = 3 * m + 1 ∨ k = 3 * m + 2) ∧
  (∀ s : ℕ, odd_positive_integer ^ 2 = 8 * p + 1) ∧
  (∀ t : ℕ, perfect_square = 3 * q ∨ perfect_square = 3 * q + 1) →
  num_correct_statements = 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_count_l2213_221315


namespace NUMINAMATH_GPT_no_such_geometric_sequence_exists_l2213_221373

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n

noncomputable def satisfies_conditions (a : ℕ → ℝ) : Prop :=
(a 1 + a 6 = 11) ∧
(a 3 * a 4 = 32 / 9) ∧
(∀ n : ℕ, a (n + 1) > a n) ∧
(∃ m : ℕ, m > 4 ∧ (2 * a m^2 = (2 / 3 * a (m - 1) + (a (m + 1) + 4 / 9))))

theorem no_such_geometric_sequence_exists : 
  ¬ ∃ a : ℕ → ℝ, geometric_sequence a ∧ satisfies_conditions a := 
sorry

end NUMINAMATH_GPT_no_such_geometric_sequence_exists_l2213_221373


namespace NUMINAMATH_GPT_b_share_of_payment_l2213_221347

def work_fraction (d : ℕ) : ℚ := 1 / d

def total_one_day_work (a_days b_days c_days : ℕ) : ℚ :=
  work_fraction a_days + work_fraction b_days + work_fraction c_days

def share_of_work (b_days : ℕ) (total_work : ℚ) : ℚ :=
  work_fraction b_days / total_work

def share_of_payment (total_payment : ℚ) (work_share : ℚ) : ℚ :=
  total_payment * work_share

theorem b_share_of_payment 
  (a_days b_days c_days : ℕ) (total_payment : ℚ):
  a_days = 6 → b_days = 8 → c_days = 12 → total_payment = 1800 →
  share_of_payment total_payment (share_of_work b_days (total_one_day_work a_days b_days c_days)) = 600 :=
by
  intros ha hb hc hp
  unfold total_one_day_work work_fraction share_of_work share_of_payment
  rw [ha, hb, hc, hp]
  -- Simplify the fractions and the multiplication
  sorry

end NUMINAMATH_GPT_b_share_of_payment_l2213_221347


namespace NUMINAMATH_GPT_sequence_monotonic_and_bounded_l2213_221369

theorem sequence_monotonic_and_bounded :
  ∀ (a : ℕ → ℝ), (a 1 = 1 / 2) → (∀ n, a (n + 1) = 1 / 2 + (a n)^2 / 2) →
    (∀ n, a n < 2) ∧ (∀ n, a n < a (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_monotonic_and_bounded_l2213_221369


namespace NUMINAMATH_GPT_double_inequality_l2213_221386

variable (a b c : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem double_inequality (h : triangle_sides a b c) : 
  3 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_double_inequality_l2213_221386


namespace NUMINAMATH_GPT_rectangle_exists_l2213_221301

theorem rectangle_exists (n : ℕ) (h_n : 0 < n)
  (marked : Finset (Fin n × Fin n))
  (h_marked : marked.card ≥ n * (Real.sqrt n + 0.5)) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧ 
    ((r1, c1) ∈ marked ∧ (r1, c2) ∈ marked ∧ (r2, c1) ∈ marked ∧ (r2, c2) ∈ marked) :=
  sorry

end NUMINAMATH_GPT_rectangle_exists_l2213_221301


namespace NUMINAMATH_GPT_lydia_age_when_planted_l2213_221338

-- Definition of the conditions
def years_to_bear_fruit : ℕ := 7
def lydia_age_when_fruit_bears : ℕ := 11

-- Lean 4 statement to prove Lydia's age when she planted the tree
theorem lydia_age_when_planted (a : ℕ) : a = lydia_age_when_fruit_bears - years_to_bear_fruit :=
by
  have : a = 4 := by sorry
  exact this

end NUMINAMATH_GPT_lydia_age_when_planted_l2213_221338


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2213_221394

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x → x + (1 / x) > a) ↔ a < 2 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2213_221394


namespace NUMINAMATH_GPT_cos_C_equal_two_thirds_l2213_221368

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Define the conditions
def condition1 : a > 0 ∧ b > 0 ∧ c > 0 := sorry
def condition2 : (a / b) + (b / a) = 4 * Real.cos C := sorry
def condition3 : Real.cos (A - B) = 1 / 6 := sorry

-- Statement to prove
theorem cos_C_equal_two_thirds 
  (h1: a > 0 ∧ b > 0 ∧ c > 0) 
  (h2: (a / b) + (b / a) = 4 * Real.cos C) 
  (h3: Real.cos (A - B) = 1 / 6) 
  : Real.cos C = 2 / 3 :=
  sorry

end NUMINAMATH_GPT_cos_C_equal_two_thirds_l2213_221368


namespace NUMINAMATH_GPT_seating_arrangement_l2213_221341

def numWaysCableCars (adults children cars capacity : ℕ) : ℕ := 
  sorry 

theorem seating_arrangement :
  numWaysCableCars 4 2 3 3 = 348 :=
by {
  sorry
}

end NUMINAMATH_GPT_seating_arrangement_l2213_221341


namespace NUMINAMATH_GPT_initial_pipes_count_l2213_221371

theorem initial_pipes_count (n : ℕ) (r : ℝ) :
  n * r = 1 / 16 → (n + 15) * r = 1 / 4 → n = 5 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_initial_pipes_count_l2213_221371


namespace NUMINAMATH_GPT_value_of_f_l2213_221375

def B : Set ℚ := {x | x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℝ := sorry

noncomputable def h (x : ℚ) : ℚ :=
  1 / (1 - x)

lemma cyclic_of_h :
  ∀ x ∈ B, h (h (h x)) = x :=
sorry

lemma functional_property (x : ℚ) (hx : x ∈ B) :
  f x + f (h x) = 2 * Real.log (|x|) :=
sorry

theorem value_of_f :
  f 2023 = Real.log 2023 :=
sorry

end NUMINAMATH_GPT_value_of_f_l2213_221375


namespace NUMINAMATH_GPT_f_xh_sub_f_x_l2213_221396

def f (x : ℝ) (k : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + k * x - 4

theorem f_xh_sub_f_x (x h : ℝ) (k : ℝ := -5) : 
    f (x + h) k - f x k = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end NUMINAMATH_GPT_f_xh_sub_f_x_l2213_221396


namespace NUMINAMATH_GPT_balls_in_jar_l2213_221383

theorem balls_in_jar (total_balls initial_blue_balls balls_after_taking_out : ℕ) (probability_blue : ℚ) :
  initial_blue_balls = 6 →
  balls_after_taking_out = initial_blue_balls - 3 →
  probability_blue = 1 / 5 →
  (balls_after_taking_out : ℚ) / (total_balls - 3 : ℚ) = probability_blue →
  total_balls = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_balls_in_jar_l2213_221383


namespace NUMINAMATH_GPT_star_operation_l2213_221309

def star (a b : ℚ) : ℚ := 2 * a - b + 1

theorem star_operation :
  star 1 (star 2 (-3)) = -5 :=
by
  -- Calcualtion follows the steps given in the solution, 
  -- but this line is here just to satisfy the 'rewrite the problem' instruction.
  sorry

end NUMINAMATH_GPT_star_operation_l2213_221309


namespace NUMINAMATH_GPT_composite_rate_proof_l2213_221322

noncomputable def composite_rate (P A : ℝ) (T : ℕ) (X Y Z : ℝ) (R : ℝ) : Prop :=
  let factor := (1 + X / 100) * (1 + Y / 100) * (1 + Z / 100)
  1.375 = factor ∧ (A = P * (1 + R / 100) ^ T)

theorem composite_rate_proof :
  composite_rate 4000 5500 3 X Y Z 11.1 :=
by sorry

end NUMINAMATH_GPT_composite_rate_proof_l2213_221322


namespace NUMINAMATH_GPT_probability_first_number_greater_l2213_221392

noncomputable def probability_first_greater_second : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_first_number_greater :
  probability_first_greater_second = 7 / 16 :=
sorry

end NUMINAMATH_GPT_probability_first_number_greater_l2213_221392


namespace NUMINAMATH_GPT_find_number_l2213_221374

theorem find_number (x : ℝ) (h : x - (3/5) * x = 62) : x = 155 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2213_221374


namespace NUMINAMATH_GPT_exterior_angle_of_regular_octagon_l2213_221360

theorem exterior_angle_of_regular_octagon (sum_of_exterior_angles : ℝ) (n_sides : ℕ) (is_regular : n_sides = 8 ∧ sum_of_exterior_angles = 360) :
  sum_of_exterior_angles / n_sides = 45 := by
  sorry

end NUMINAMATH_GPT_exterior_angle_of_regular_octagon_l2213_221360


namespace NUMINAMATH_GPT_rice_yield_prediction_l2213_221363

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 5 * x + 250

-- Define the specific condition for x = 80
def fertilizer_amount : ℝ := 80

-- State the theorem for the expected rice yield
theorem rice_yield_prediction : regression_line fertilizer_amount = 650 :=
by
  sorry

end NUMINAMATH_GPT_rice_yield_prediction_l2213_221363
