import Mathlib

namespace NUMINAMATH_GPT_grape_ratio_new_new_cans_from_grape_l1490_149013

-- Definitions derived from the problem conditions
def apple_ratio_initial : ℚ := 1 / 6
def grape_ratio_initial : ℚ := 1 / 10
def apple_ratio_new : ℚ := 1 / 5

-- Prove the new grape_ratio
theorem grape_ratio_new : ℚ :=
  let total_volume_per_can := apple_ratio_initial + grape_ratio_initial
  let grape_ratio_new_reciprocal := (total_volume_per_can - apple_ratio_new)
  1 / grape_ratio_new_reciprocal

-- Required final quantity of cans
theorem new_cans_from_grape : 
  (1 / grape_ratio_new) = 15 :=
sorry

end NUMINAMATH_GPT_grape_ratio_new_new_cans_from_grape_l1490_149013


namespace NUMINAMATH_GPT_Ben_total_clothes_l1490_149029

-- Definitions of Alex's clothing items
def Alex_shirts := 4.5
def Alex_pants := 3.0
def Alex_shoes := 2.5
def Alex_hats := 1.5
def Alex_jackets := 2.0

-- Definitions of Joe's clothing items
def Joe_shirts := Alex_shirts + 3.5
def Joe_pants := Alex_pants - 2.5
def Joe_shoes := Alex_shoes
def Joe_hats := Alex_hats + 0.3
def Joe_jackets := Alex_jackets - 1.0

-- Definitions of Ben's clothing items
def Ben_shirts := Joe_shirts + 5.3
def Ben_pants := Alex_pants + 5.5
def Ben_shoes := Joe_shoes - 1.7
def Ben_hats := Alex_hats + 0.5
def Ben_jackets := Joe_jackets + 1.5

-- Statement to prove the total number of Ben's clothing items
def total_Ben_clothing_items := Ben_shirts + Ben_pants + Ben_shoes + Ben_hats + Ben_jackets

theorem Ben_total_clothes : total_Ben_clothing_items = 27.1 :=
by
  sorry

end NUMINAMATH_GPT_Ben_total_clothes_l1490_149029


namespace NUMINAMATH_GPT_gcd_40_56_l1490_149053

theorem gcd_40_56 : Int.gcd 40 56 = 8 :=
by
  sorry

end NUMINAMATH_GPT_gcd_40_56_l1490_149053


namespace NUMINAMATH_GPT_find_x_coordinate_l1490_149044

noncomputable def point_on_plane (x y : ℝ) :=
  (|x + y - 1| / Real.sqrt 2 = |x| ∧
   |x| = |y - 3 * x| / Real.sqrt 10)

theorem find_x_coordinate (x y : ℝ) (h : point_on_plane x y) : 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_find_x_coordinate_l1490_149044


namespace NUMINAMATH_GPT_alexander_first_gallery_pictures_l1490_149005

def pictures_for_new_galleries := 5 * 2
def pencils_for_new_galleries := pictures_for_new_galleries * 4
def total_exhibitions := 1 + 5
def pencils_for_signing := total_exhibitions * 2
def total_pencils := 88
def pencils_for_first_gallery := total_pencils - pencils_for_new_galleries - pencils_for_signing
def pictures_for_first_gallery := pencils_for_first_gallery / 4

theorem alexander_first_gallery_pictures : pictures_for_first_gallery = 9 :=
by
  sorry

end NUMINAMATH_GPT_alexander_first_gallery_pictures_l1490_149005


namespace NUMINAMATH_GPT_train_speed_in_kmh_l1490_149099

/-- Definition of length of the train in meters. -/
def train_length : ℕ := 200

/-- Definition of time taken to cross the electric pole in seconds. -/
def time_to_cross : ℕ := 20

/-- The speed of the train in km/h is 36 given the length of the train and time to cross. -/
theorem train_speed_in_kmh (length : ℕ) (time : ℕ) (h_len : length = train_length) (h_time: time = time_to_cross) : 
  (length / time : ℚ) * 3.6 = 36 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l1490_149099


namespace NUMINAMATH_GPT_further_flight_Gaeun_l1490_149080

theorem further_flight_Gaeun :
  let nana_distance_m := 1.618
  let gaeun_distance_cm := 162.3
  let conversion_factor := 100
  let nana_distance_cm := nana_distance_m * conversion_factor
  gaeun_distance_cm > nana_distance_cm := 
  sorry

end NUMINAMATH_GPT_further_flight_Gaeun_l1490_149080


namespace NUMINAMATH_GPT_seashells_total_l1490_149084

theorem seashells_total (x y z T : ℕ) (m k : ℝ) 
  (h₁ : x = 2) 
  (h₂ : y = 5) 
  (h₃ : z = 9) 
  (h₄ : x + y = T) 
  (h₅ : m * x + k * y = z) : 
  T = 7 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_seashells_total_l1490_149084


namespace NUMINAMATH_GPT_derivative_at_1_l1490_149035

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 * x + (1 / 2) * x^2

theorem derivative_at_1 : deriv f 1 = Real.exp 1 := 
by 
  sorry

end NUMINAMATH_GPT_derivative_at_1_l1490_149035


namespace NUMINAMATH_GPT_game_C_more_likely_than_game_D_l1490_149001

-- Definitions for the probabilities
def p_heads : ℚ := 3 / 4
def p_tails : ℚ := 1 / 4

-- Game C probability
def p_game_C : ℚ := p_heads ^ 4

-- Game D probabilities for each scenario
def p_game_D_scenario1 : ℚ := (p_heads ^ 3) * (p_heads ^ 2)
def p_game_D_scenario2 : ℚ := (p_heads ^ 3) * (p_tails ^ 2)
def p_game_D_scenario3 : ℚ := (p_tails ^ 3) * (p_heads ^ 2)
def p_game_D_scenario4 : ℚ := (p_tails ^ 3) * (p_tails ^ 2)

-- Total probability for Game D
def p_game_D : ℚ :=
  p_game_D_scenario1 + p_game_D_scenario2 + p_game_D_scenario3 + p_game_D_scenario4

-- Proof statement
theorem game_C_more_likely_than_game_D : (p_game_C - p_game_D) = 11 / 256 := by
  sorry

end NUMINAMATH_GPT_game_C_more_likely_than_game_D_l1490_149001


namespace NUMINAMATH_GPT_find_r_l1490_149032

theorem find_r (r : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = 4) → 
  (∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = r^2) →
  (∀ x1 y1 x2 y2: ℝ, 
    (x2 - x1)^2 + (y2 - y1)^2 = 25) →
  (2 + |r| = 5) →
  (r = 3 ∨ r = -3) :=
by
  sorry

end NUMINAMATH_GPT_find_r_l1490_149032


namespace NUMINAMATH_GPT_greatest_integer_b_l1490_149026

theorem greatest_integer_b (b : ℤ) :
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 ≠ -25) → b ≤ 10 :=
by
  intro
  sorry

end NUMINAMATH_GPT_greatest_integer_b_l1490_149026


namespace NUMINAMATH_GPT_john_exactly_three_green_marbles_l1490_149015

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 7
def green_prob : ℚ := 8 / 15
def purple_prob : ℚ := 7 / 15
def binom_coeff : ℕ := Nat.choose 7 3 

-- Theorem Statement
theorem john_exactly_three_green_marbles :
  (binom_coeff : ℚ) * (green_prob^3 * purple_prob^4) = 8604112 / 15946875 :=
by
  sorry

end NUMINAMATH_GPT_john_exactly_three_green_marbles_l1490_149015


namespace NUMINAMATH_GPT_roots_quadratic_l1490_149096

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end NUMINAMATH_GPT_roots_quadratic_l1490_149096


namespace NUMINAMATH_GPT_down_payment_l1490_149019

theorem down_payment {total_loan : ℕ} {monthly_payment : ℕ} {years : ℕ} (h1 : total_loan = 46000) (h2 : monthly_payment = 600) (h3 : years = 5):
  total_loan - (years * 12 * monthly_payment) = 10000 := by
  sorry

end NUMINAMATH_GPT_down_payment_l1490_149019


namespace NUMINAMATH_GPT_katie_roll_probability_l1490_149062

def prob_less_than_five (d : ℕ) : ℚ :=
if d < 5 then 1 else 0

def prob_even (d : ℕ) : ℚ :=
if d % 2 = 0 then 1 else 0

theorem katie_roll_probability :
  (prob_less_than_five 1 + prob_less_than_five 2 + prob_less_than_five 3 + prob_less_than_five 4 +
  prob_less_than_five 5 + prob_less_than_five 6) / 6 *
  (prob_even 1 + prob_even 2 + prob_even 3 + prob_even 4 +
  prob_even 5 + prob_even 6) / 6 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_katie_roll_probability_l1490_149062


namespace NUMINAMATH_GPT_negation_of_universal_l1490_149077

theorem negation_of_universal: (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_universal_l1490_149077


namespace NUMINAMATH_GPT_max_triangle_side_length_l1490_149054

theorem max_triangle_side_length:
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a + b + c = 30 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 14 :=
  sorry

end NUMINAMATH_GPT_max_triangle_side_length_l1490_149054


namespace NUMINAMATH_GPT_number_multiplies_a_l1490_149007

theorem number_multiplies_a (a b x : ℝ) (h₀ : x * a = 8 * b) (h₁ : a ≠ 0 ∧ b ≠ 0) (h₂ : (a / 8) / (b / 7) = 1) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_multiplies_a_l1490_149007


namespace NUMINAMATH_GPT_smallest_sector_angle_l1490_149050

theorem smallest_sector_angle 
  (n : ℕ) (a1 : ℕ) (d : ℕ)
  (h1 : n = 18)
  (h2 : 360 = n * ((2 * a1 + (n - 1) * d) / 2))
  (h3 : ∀ i, 0 < i ∧ i ≤ 18 → ∃ k, 360 / 18 * k = i) :
  a1 = 3 :=
by sorry

end NUMINAMATH_GPT_smallest_sector_angle_l1490_149050


namespace NUMINAMATH_GPT_max_log_sum_l1490_149058

noncomputable def log (x : ℝ) : ℝ := Real.log x

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 
  ∃ L, (∀ x y, x > 0 → y > 0 → x + y = 4 → log x + log y ≤ L) ∧ L = log 4 :=
by
  sorry

end NUMINAMATH_GPT_max_log_sum_l1490_149058


namespace NUMINAMATH_GPT_smallest_solution_eq_l1490_149078

noncomputable def smallest_solution := 4 - Real.sqrt 3

theorem smallest_solution_eq (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 3 / (x - 4)) → x = smallest_solution :=
sorry

end NUMINAMATH_GPT_smallest_solution_eq_l1490_149078


namespace NUMINAMATH_GPT_dispersion_is_variance_l1490_149045

def Mean := "Mean"
def Variance := "Variance"
def Median := "Median"
def Mode := "Mode"

def dispersion_measure := Variance

theorem dispersion_is_variance (A B C D : String) (hA : A = Mean) (hB : B = Variance) (hC : C = Median) (hD : D = Mode) : 
  dispersion_measure = B :=
by
  rw [hB]
  exact sorry

end NUMINAMATH_GPT_dispersion_is_variance_l1490_149045


namespace NUMINAMATH_GPT_part1_part2_l1490_149016

theorem part1 (x : ℝ) : -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 :=
by sorry

theorem part2 (a x : ℝ) (h : 0 < a) :
  (a * x^2 + (a + 3) * x + 3 > 0 ↔
    (
      (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
      (a = 3 ∧ x ≠ -1) ∨
      (a > 3 ∧ (x < -1 ∨ x > -3/a))
    )
  ) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1490_149016


namespace NUMINAMATH_GPT_bowls_per_minute_l1490_149057

def ounces_per_bowl : ℕ := 10
def gallons_of_soup : ℕ := 6
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem bowls_per_minute :
  (gallons_of_soup * ounces_per_gallon / servings_time_minutes) / ounces_per_bowl = 5 :=
by
  sorry

end NUMINAMATH_GPT_bowls_per_minute_l1490_149057


namespace NUMINAMATH_GPT_tangent_line_equation_l1490_149059

theorem tangent_line_equation :
  (∃ l : ℝ → ℝ, 
   (∀ x, l x = (1 / (4 + 2 * Real.sqrt 3)) * x + (2 + Real.sqrt 3) / 2 ∨ 
         l x = (1 / (4 - 2 * Real.sqrt 3)) * x + (2 - Real.sqrt 3) / 2) ∧ 
   (l 1 = 2) ∧ 
   (∀ x, l x = Real.sqrt x)
  ) →
  (∀ x y, 
   (y = (1 / 4 + Real.sqrt 3) * x + (2 + Real.sqrt 3) / 2 ∨ 
    y = (1 / 4 - Real.sqrt 3) * x + (2 - Real.sqrt 3) / 2) ∨ 
   (x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0 ∨ 
    x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)
) :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l1490_149059


namespace NUMINAMATH_GPT_range_of_a_l1490_149064

noncomputable def f (a x : ℝ) := a * x^2 - (2 - a) * x + 1
noncomputable def g (x : ℝ) := x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0 ∨ g x > 0) ↔ (0 ≤ a ∧ a < 4 + 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1490_149064


namespace NUMINAMATH_GPT_ariel_fish_l1490_149036

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end NUMINAMATH_GPT_ariel_fish_l1490_149036


namespace NUMINAMATH_GPT_custom_op_4_3_equals_37_l1490_149046

def custom_op (a b : ℕ) : ℕ := a^2 + a*b + b^2

theorem custom_op_4_3_equals_37 : custom_op 4 3 = 37 := by
  sorry

end NUMINAMATH_GPT_custom_op_4_3_equals_37_l1490_149046


namespace NUMINAMATH_GPT_orange_segments_l1490_149043

noncomputable def total_segments (H S B : ℕ) : ℕ :=
  H + S + B

theorem orange_segments
  (H S B : ℕ)
  (h1 : H = 2 * S)
  (h2 : S = B / 5)
  (h3 : B = S + 8) :
  total_segments H S B = 16 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_orange_segments_l1490_149043


namespace NUMINAMATH_GPT_golden_fish_caught_times_l1490_149047

open Nat

theorem golden_fish_caught_times :
  ∃ (x y z : ℕ), (4 * x + 2 * z = 2000) ∧ (2 * y + z = 800) ∧ (x + y + z = 900) :=
sorry

end NUMINAMATH_GPT_golden_fish_caught_times_l1490_149047


namespace NUMINAMATH_GPT_Hannah_total_spent_l1490_149012

def rides_cost (total_money : ℝ) : ℝ :=
  0.35 * total_money

def games_cost (total_money : ℝ) : ℝ :=
  0.25 * total_money

def food_and_souvenirs_cost : ℝ :=
  7 + 4 + 5 + 6

def total_spent (total_money : ℝ) : ℝ :=
  rides_cost total_money + games_cost total_money + food_and_souvenirs_cost

theorem Hannah_total_spent (total_money : ℝ) (h : total_money = 80) :
  total_spent total_money = 70 :=
by
  rw [total_spent, h, rides_cost, games_cost]
  norm_num
  sorry

end NUMINAMATH_GPT_Hannah_total_spent_l1490_149012


namespace NUMINAMATH_GPT_problem1_1_problem1_2_problem2_l1490_149069

open Set

/-
Given sets U, A, and B, derived from the provided conditions:
  U : Set ℝ
  A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
  B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}
-/

def U : Set ℝ := univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}

theorem problem1_1 (m : ℝ) (h : m = 5) : A ∩ B m = {x | -3 ≤ x ∧ x ≤ 5} :=
sorry

theorem problem1_2 (m : ℝ) (h : m = 5) : (compl A) ∪ B m = univ :=
sorry

theorem problem2 (m : ℝ) : A ⊆ B m → 4 < m :=
sorry

end NUMINAMATH_GPT_problem1_1_problem1_2_problem2_l1490_149069


namespace NUMINAMATH_GPT_sum_of_six_numbers_l1490_149010

theorem sum_of_six_numbers:
  ∃ (A B C D E F : ℕ), 
    A > B ∧ B > C ∧ C > D ∧ D > E ∧ E > F ∧
    E > F ∧ C > F ∧ D > F ∧ A + B + C + D + E + F = 141 := 
sorry

end NUMINAMATH_GPT_sum_of_six_numbers_l1490_149010


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1490_149095

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 1) (h3 : a 4 * a 5 * a 6 = 8) :
  a 2 + a 5 + a 8 + a 11 = 15 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1490_149095


namespace NUMINAMATH_GPT_sin_squared_equiv_cosine_l1490_149009

theorem sin_squared_equiv_cosine :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_GPT_sin_squared_equiv_cosine_l1490_149009


namespace NUMINAMATH_GPT_work_rate_problem_l1490_149094

theorem work_rate_problem (A B : ℚ) (h1 : A + B = 1/8) (h2 : A = 1/12) : B = 1/24 :=
sorry

end NUMINAMATH_GPT_work_rate_problem_l1490_149094


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1490_149006

def f (c a b : ℝ) (x : ℝ) : ℝ := |(c * x + a)| + |(c * x - b)|
def g (c : ℝ) (x : ℝ) : ℝ := |(x - 2)| + c

noncomputable def sol_set_eq1 := {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 2}
noncomputable def range_a_eq2 := {a : ℝ | a ≤ -2 ∨ a ≥ 0}

-- Problem (1)
theorem problem1_solution : ∀ (x : ℝ), f 2 1 3 x - 4 = 0 ↔ x ∈ sol_set_eq1 := 
by
  intro x
  sorry -- Proof to be filled in

-- Problem (2)
theorem problem2_solution : 
  ∀ x_1 : ℝ, ∃ x_2 : ℝ, g 1 x_2 = f 1 0 1 x_1 ↔ a ∈ range_a_eq2 :=
by
  intro x_1
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1490_149006


namespace NUMINAMATH_GPT_jason_fishes_on_day_12_l1490_149066

def initial_fish_count : ℕ := 10

def fish_on_day (n : ℕ) : ℕ :=
  if n = 0 then initial_fish_count else
  (match n with
  | 1 => 10 * 3
  | 2 => 30 * 3
  | 3 => 90 * 3
  | 4 => 270 * 3 * 3 / 5 -- removes fish according to rule
  | 5 => (270 * 3 * 3 / 5) * 3
  | 6 => ((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7 -- removes fish according to rule
  | 7 => (((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3
  | 8 => ((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25
  | 9 => (((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3
  | 10 => ((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)
  | 11 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3
  | 12 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3 + (3 * (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3) + 5
  | _ => 0
  )
 
theorem jason_fishes_on_day_12 : fish_on_day 12 = 1220045 := 
  by sorry

end NUMINAMATH_GPT_jason_fishes_on_day_12_l1490_149066


namespace NUMINAMATH_GPT_complex_number_quadrant_l1490_149038

theorem complex_number_quadrant 
  (i : ℂ) (hi : i.im = 1 ∧ i.re = 0)
  (x y : ℝ) 
  (h : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := 
sorry

end NUMINAMATH_GPT_complex_number_quadrant_l1490_149038


namespace NUMINAMATH_GPT_joan_dimes_l1490_149014

theorem joan_dimes :
  ∀ (total_dimes_jacket : ℕ) (total_money : ℝ) (value_per_dime : ℝ),
    total_dimes_jacket = 15 →
    total_money = 1.90 →
    value_per_dime = 0.10 →
    ((total_money - (total_dimes_jacket * value_per_dime)) / value_per_dime) = 4 :=
by
  intros total_dimes_jacket total_money value_per_dime h1 h2 h3
  sorry

end NUMINAMATH_GPT_joan_dimes_l1490_149014


namespace NUMINAMATH_GPT_desired_cost_per_pound_l1490_149081
-- Importing the necessary library

-- Defining the candy weights and their costs per pound
def weight1 : ℝ := 20
def cost_per_pound1 : ℝ := 8
def weight2 : ℝ := 40
def cost_per_pound2 : ℝ := 5

-- Defining the proof statement
theorem desired_cost_per_pound :
  let total_cost := (weight1 * cost_per_pound1 + weight2 * cost_per_pound2)
  let total_weight := (weight1 + weight2)
  let desired_cost := total_cost / total_weight
  desired_cost = 6 := sorry

end NUMINAMATH_GPT_desired_cost_per_pound_l1490_149081


namespace NUMINAMATH_GPT_sin_add_pi_over_4_eq_l1490_149087

variable (α : Real)
variables (hα1 : 0 < α ∧ α < Real.pi) (hα2 : Real.tan (α - Real.pi / 4) = 1 / 3)

theorem sin_add_pi_over_4_eq : Real.sin (Real.pi / 4 + α) = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_add_pi_over_4_eq_l1490_149087


namespace NUMINAMATH_GPT_eggs_per_group_l1490_149051

-- Define the conditions
def num_eggs : ℕ := 18
def num_groups : ℕ := 3

-- Theorem stating number of eggs per group
theorem eggs_per_group : num_eggs / num_groups = 6 :=
by
  sorry

end NUMINAMATH_GPT_eggs_per_group_l1490_149051


namespace NUMINAMATH_GPT_inequality_proof_l1490_149028
open Nat

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 1) (h4 : n > 0) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1490_149028


namespace NUMINAMATH_GPT_second_root_l1490_149017

variables {a b c x : ℝ}

theorem second_root (h : a * (b + c) * x ^ 2 - b * (c + a) * x + c * (a + b) = 0)
(hroot : a * (b + c) * (-1) ^ 2 - b * (c + a) * (-1) + c * (a + b) = 0) :
  ∃ k : ℝ, k = - c * (a + b) / (a * (b + c)) ∧ a * (b + c) * k ^ 2 - b * (c + a) * k + c * (a + b) = 0 :=
sorry

end NUMINAMATH_GPT_second_root_l1490_149017


namespace NUMINAMATH_GPT_total_ages_l1490_149048

variable (Frank : ℕ) (Gabriel : ℕ)
variables (h1 : Frank = 10) (h2 : Gabriel = Frank - 3)

theorem total_ages (hF : Frank = 10) (hG : Gabriel = Frank - 3) : Frank + Gabriel = 17 :=
by
  rw [hF, hG]
  norm_num
  sorry

end NUMINAMATH_GPT_total_ages_l1490_149048


namespace NUMINAMATH_GPT_percentage_decrease_l1490_149008

noncomputable def original_fraction (N D : ℝ) : Prop := N / D = 0.75
noncomputable def new_fraction (N D x : ℝ) : Prop := (1.15 * N) / (D * (1 - x / 100)) = 15 / 16

theorem percentage_decrease (N D x : ℝ) (h1 : original_fraction N D) (h2 : new_fraction N D x) : 
  x = 22.67 := 
sorry

end NUMINAMATH_GPT_percentage_decrease_l1490_149008


namespace NUMINAMATH_GPT_divisor_of_sum_of_four_consecutive_integers_l1490_149039

theorem divisor_of_sum_of_four_consecutive_integers (n : ℤ) :
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end NUMINAMATH_GPT_divisor_of_sum_of_four_consecutive_integers_l1490_149039


namespace NUMINAMATH_GPT_range_of_a_for_decreasing_exponential_l1490_149082

theorem range_of_a_for_decreasing_exponential :
  ∀ (a : ℝ), (∀ (x1 x2 : ℝ), x1 < x2 → (2 - a)^x1 > (2 - a)^x2) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_decreasing_exponential_l1490_149082


namespace NUMINAMATH_GPT_exists_n_gt_2_divisible_by_1991_l1490_149073

theorem exists_n_gt_2_divisible_by_1991 :
  ∃ n > 2, 1991 ∣ (2 * 10^(n+1) - 9) :=
by
  existsi (1799 : Nat)
  have h1 : 1799 > 2 := by decide
  have h2 : 1991 ∣ (2 * 10^(1799+1) - 9) := sorry
  constructor
  · exact h1
  · exact h2

end NUMINAMATH_GPT_exists_n_gt_2_divisible_by_1991_l1490_149073


namespace NUMINAMATH_GPT_triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l1490_149030

-- Definition of the sides according to Plato's rule
def triangle_sides (p : ℕ) : ℕ × ℕ × ℕ :=
  (2 * p, p^2 - 1, p^2 + 1)

-- Function to check if the given sides form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Theorems to verify the sides of the triangle for given p values
theorem triangle_sides_p2 : triangle_sides 2 = (4, 3, 5) ∧ is_right_triangle 4 3 5 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p3 : triangle_sides 3 = (6, 8, 10) ∧ is_right_triangle 6 8 10 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p4 : triangle_sides 4 = (8, 15, 17) ∧ is_right_triangle 8 15 17 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p5 : triangle_sides 5 = (10, 24, 26) ∧ is_right_triangle 10 24 26 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l1490_149030


namespace NUMINAMATH_GPT_hexagonal_pyramid_edge_length_l1490_149041

noncomputable def hexagonal_pyramid_edge_sum (s h : ℝ) : ℝ :=
  let perimeter := 6 * s
  let center_to_vertex := s * (1 / 2) * Real.sqrt 3
  let slant_height := Real.sqrt (h^2 + center_to_vertex^2)
  let edge_sum := perimeter + 6 * slant_height
  edge_sum

theorem hexagonal_pyramid_edge_length (s h : ℝ) (a : ℝ) :
  s = 8 →
  h = 15 →
  a = 48 + 6 * Real.sqrt 273 →
  hexagonal_pyramid_edge_sum s h = a :=
by
  intros
  sorry

end NUMINAMATH_GPT_hexagonal_pyramid_edge_length_l1490_149041


namespace NUMINAMATH_GPT_calculate_difference_l1490_149092

def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem calculate_difference :
  f (g 5) - g (f 5) = -2 := by
  sorry

end NUMINAMATH_GPT_calculate_difference_l1490_149092


namespace NUMINAMATH_GPT_remainder_is_37_l1490_149085

theorem remainder_is_37
    (d q v r : ℕ)
    (h1 : d = 15968)
    (h2 : q = 89)
    (h3 : v = 179)
    (h4 : d = q * v + r) :
  r = 37 :=
sorry

end NUMINAMATH_GPT_remainder_is_37_l1490_149085


namespace NUMINAMATH_GPT_sum_of_squares_of_coefficients_l1490_149067

theorem sum_of_squares_of_coefficients :
  let poly := 5 * (X^6 + 4 * X^4 + 2 * X^2 + 1)
  let coeffs := [5, 20, 10, 5]
  (coeffs.map (λ c => c * c)).sum = 550 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coefficients_l1490_149067


namespace NUMINAMATH_GPT_used_crayons_l1490_149074

open Nat

theorem used_crayons (N B T U : ℕ) (h1 : N = 2) (h2 : B = 8) (h3 : T = 14) (h4 : T = N + U + B) : U = 4 :=
by
  -- Proceed with the proof here
  sorry

end NUMINAMATH_GPT_used_crayons_l1490_149074


namespace NUMINAMATH_GPT_partial_fraction_series_sum_l1490_149088

theorem partial_fraction_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_partial_fraction_series_sum_l1490_149088


namespace NUMINAMATH_GPT_find_a_maximize_profit_l1490_149061

theorem find_a (a: ℕ) (h: 600 * (a - 110) = 160 * a) : a = 150 :=
sorry

theorem maximize_profit (x y: ℕ) (a: ℕ) 
  (ha: a = 150)
  (hx: x + 5 * x + 20 ≤ 200) 
  (profit_eq: ∀ x, y = 245 * x + 600):
  x = 30 ∧ y = 7950 :=
sorry

end NUMINAMATH_GPT_find_a_maximize_profit_l1490_149061


namespace NUMINAMATH_GPT_tanya_efficiency_increase_l1490_149052

theorem tanya_efficiency_increase 
  (s_efficiency : ℝ := 1 / 10) (t_efficiency : ℝ := 1 / 8) :
  (((t_efficiency - s_efficiency) / s_efficiency) * 100) = 25 := 
by
  sorry

end NUMINAMATH_GPT_tanya_efficiency_increase_l1490_149052


namespace NUMINAMATH_GPT_remaining_budget_is_correct_l1490_149075

def budget := 750
def flasks_cost := 200
def test_tubes_cost := (2 / 3) * flasks_cost
def safety_gear_cost := (1 / 2) * test_tubes_cost
def chemicals_cost := (3 / 4) * flasks_cost
def instruments_min_cost := 50

def total_spent := flasks_cost + test_tubes_cost + safety_gear_cost + chemicals_cost
def remaining_budget_before_instruments := budget - total_spent
def remaining_budget_after_instruments := remaining_budget_before_instruments - instruments_min_cost

theorem remaining_budget_is_correct :
  remaining_budget_after_instruments = 150 := by
  unfold remaining_budget_after_instruments remaining_budget_before_instruments total_spent flasks_cost test_tubes_cost safety_gear_cost chemicals_cost budget
  sorry

end NUMINAMATH_GPT_remaining_budget_is_correct_l1490_149075


namespace NUMINAMATH_GPT_odd_function_condition_l1490_149089

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x) / ((x - a) * (x + 1))

theorem odd_function_condition (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 := 
sorry

end NUMINAMATH_GPT_odd_function_condition_l1490_149089


namespace NUMINAMATH_GPT_ken_ride_time_l1490_149002

variables (x y k t : ℝ)

-- Condition 1: It takes Ken 80 seconds to walk down an escalator when it is not moving.
def condition1 : Prop := 80 * x = y

-- Condition 2: It takes Ken 40 seconds to walk down an escalator when it is moving with a 10-second delay.
def condition2 : Prop := 50 * (x + k) = y

-- Condition 3: There is a 10-second delay before the escalator starts moving.
def condition3 : Prop := t = y / k + 10

-- Related Speed
def condition4 : Prop := k = 0.6 * x

-- Proposition: The time Ken takes to ride the escalator down without walking, including the delay, is 143 seconds.
theorem ken_ride_time {x y k t : ℝ} (h1 : condition1 x y) (h2 : condition2 x y k) (h3 : condition3 y k t) (h4 : condition4 x k) :
  t = 143 :=
by sorry

end NUMINAMATH_GPT_ken_ride_time_l1490_149002


namespace NUMINAMATH_GPT_number_of_days_worked_l1490_149000

-- Define the conditions
def hours_per_day := 8
def total_hours := 32

-- Define the proof statement
theorem number_of_days_worked : total_hours / hours_per_day = 4 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_days_worked_l1490_149000


namespace NUMINAMATH_GPT_inequality1_solution_inequality2_solution_l1490_149027

open Real

-- First problem: proving the solution set for x + |2x + 3| >= 2
theorem inequality1_solution (x : ℝ) : x + abs (2 * x + 3) >= 2 ↔ (x <= -5 ∨ x >= -1/3) := 
sorry

-- Second problem: proving the solution set for |x - 1| - |x - 5| < 2
theorem inequality2_solution (x : ℝ) : abs (x - 1) - abs (x - 5) < 2 ↔ x < 4 :=
sorry

end NUMINAMATH_GPT_inequality1_solution_inequality2_solution_l1490_149027


namespace NUMINAMATH_GPT_total_fish_count_l1490_149097

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end NUMINAMATH_GPT_total_fish_count_l1490_149097


namespace NUMINAMATH_GPT_solve_y_l1490_149004

theorem solve_y (x y : ℤ) (h₁ : x = 3) (h₂ : x^3 - x - 2 = y + 2) : y = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solve_y_l1490_149004


namespace NUMINAMATH_GPT_log2_ratio_squared_l1490_149091

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log2_ratio_squared :
  ∀ (x y : ℝ), x ≠ 1 → y ≠ 1 → log_base 2 x = log_base y 25 → x * y = 81
  → (log_base 2 (x / y))^2 = 5.11 :=
by
  intros x y hx hy hlog hxy
  sorry

end NUMINAMATH_GPT_log2_ratio_squared_l1490_149091


namespace NUMINAMATH_GPT_inequality_proof_l1490_149086

theorem inequality_proof (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt ( (a * x) / (a * x + 8) ) ≤ 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l1490_149086


namespace NUMINAMATH_GPT_sum_equality_l1490_149083

-- Define the conditions and hypothesis
variables (x y z : ℝ)
axiom condition : (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0

-- State the theorem
theorem sum_equality : x + y + z = 21 :=
by sorry

end NUMINAMATH_GPT_sum_equality_l1490_149083


namespace NUMINAMATH_GPT_cube_side_length_l1490_149093

def cube_volume (side : ℝ) : ℝ := side ^ 3

theorem cube_side_length (volume : ℝ) (h : volume = 729) : ∃ (side : ℝ), side = 9 ∧ cube_volume side = volume :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_l1490_149093


namespace NUMINAMATH_GPT_blue_red_difference_l1490_149021

variable (B : ℕ) -- Blue crayons
variable (R : ℕ := 14) -- Red crayons
variable (Y : ℕ := 32) -- Yellow crayons
variable (H : Y = 2 * B - 6) -- Relationship between yellow and blue crayons

theorem blue_red_difference (B : ℕ) (H : (32:ℕ) = 2 * B - 6) : (B - 14 = 5) :=
by
  -- Proof steps goes here
  sorry

end NUMINAMATH_GPT_blue_red_difference_l1490_149021


namespace NUMINAMATH_GPT_probability_one_person_hits_probability_plane_is_hit_l1490_149042
noncomputable def P_A := 0.7
noncomputable def P_B := 0.6

theorem probability_one_person_hits : P_A * (1 - P_B) + (1 - P_A) * P_B = 0.46 :=
by
  sorry

theorem probability_plane_is_hit : 1 - (1 - P_A) * (1 - P_B) = 0.88 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_person_hits_probability_plane_is_hit_l1490_149042


namespace NUMINAMATH_GPT_smaller_solution_of_quadratic_eq_l1490_149063

noncomputable def smaller_solution (a b c : ℝ) : ℝ :=
  if a ≠ 0 then min ((-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
              ((-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
  else if b ≠ 0 then -c / b else 0 

theorem smaller_solution_of_quadratic_eq :
  smaller_solution 1 (-13) (-30) = -2 := 
by
  sorry

end NUMINAMATH_GPT_smaller_solution_of_quadratic_eq_l1490_149063


namespace NUMINAMATH_GPT_multiple_proof_l1490_149070

theorem multiple_proof (n m : ℝ) (h1 : n = 25) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end NUMINAMATH_GPT_multiple_proof_l1490_149070


namespace NUMINAMATH_GPT_avg_age_combined_l1490_149079

-- Define the conditions
def avg_age_roomA : ℕ := 45
def avg_age_roomB : ℕ := 20
def num_people_roomA : ℕ := 8
def num_people_roomB : ℕ := 3

-- Definition of the problem statement
theorem avg_age_combined :
  (num_people_roomA * avg_age_roomA + num_people_roomB * avg_age_roomB) / (num_people_roomA + num_people_roomB) = 38 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_combined_l1490_149079


namespace NUMINAMATH_GPT_unique_zero_function_l1490_149037

theorem unique_zero_function {f : ℕ → ℕ} (h : ∀ m n, f (m + f n) = f m + f n + f (n + 1)) : ∀ n, f n = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_zero_function_l1490_149037


namespace NUMINAMATH_GPT_hamburgers_purchased_l1490_149020

theorem hamburgers_purchased (total_revenue : ℕ) (hamburger_price : ℕ) (additional_hamburgers : ℕ) 
  (target_amount : ℕ) (h1 : total_revenue = 50) (h2 : hamburger_price = 5) (h3 : additional_hamburgers = 4) 
  (h4 : target_amount = 50) :
  (target_amount - (additional_hamburgers * hamburger_price)) / hamburger_price = 6 := 
by 
  sorry

end NUMINAMATH_GPT_hamburgers_purchased_l1490_149020


namespace NUMINAMATH_GPT_equilateral_triangle_bound_l1490_149098

theorem equilateral_triangle_bound (n k : ℕ) (h_n_gt_3 : n > 3) 
  (h_k_triangles : ∃ T : Finset (Finset (ℝ × ℝ)), T.card = k ∧ ∀ t ∈ T, 
  ∃ a b c : (ℝ × ℝ), t = {a, b, c} ∧ dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1) :
  k < (2 * n) / 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_bound_l1490_149098


namespace NUMINAMATH_GPT_correct_equation_l1490_149090

theorem correct_equation (x : ℝ) (h1 : 2000 > 0) (h2 : x > 0) (h3 : x + 40 > 0) :
  (2000 / x) - (2000 / (x + 40)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l1490_149090


namespace NUMINAMATH_GPT_runner_advantage_l1490_149022

theorem runner_advantage (x y z : ℝ) (hx_y: y - x = 0.1) (hy_z: z - y = 0.11111111111111111) :
  z - x = 0.21111111111111111 :=
by
  sorry

end NUMINAMATH_GPT_runner_advantage_l1490_149022


namespace NUMINAMATH_GPT_complement_U_A_l1490_149071

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_U_A : U \ A = {2, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l1490_149071


namespace NUMINAMATH_GPT_tan_passing_through_point_l1490_149060

theorem tan_passing_through_point :
  (∃ ϕ : ℝ, (∀ x : ℝ, y = Real.tan (2 * x + ϕ)) ∧ (Real.tan (2 * (π / 12) + ϕ) = 0)) →
  ϕ = - (π / 6) :=
by
  sorry

end NUMINAMATH_GPT_tan_passing_through_point_l1490_149060


namespace NUMINAMATH_GPT_james_net_income_correct_l1490_149040

def regular_price_per_hour : ℝ := 20
def discount_percent : ℝ := 0.10
def rental_hours_per_day_monday : ℝ := 8
def rental_hours_per_day_wednesday : ℝ := 8
def rental_hours_per_day_friday : ℝ := 6
def rental_hours_per_day_sunday : ℝ := 5
def sales_tax_percent : ℝ := 0.05
def car_maintenance_cost_per_week : ℝ := 35
def insurance_fee_per_day : ℝ := 15

-- Total rental hours
def total_rental_hours : ℝ :=
  rental_hours_per_day_monday + rental_hours_per_day_wednesday + rental_hours_per_day_friday + rental_hours_per_day_sunday

-- Total rental income before discount
def total_rental_income : ℝ := total_rental_hours * regular_price_per_hour

-- Discounted rental income
def discounted_rental_income : ℝ := total_rental_income * (1 - discount_percent)

-- Total income with tax
def total_income_with_tax : ℝ := discounted_rental_income * (1 + sales_tax_percent)

-- Total expenses
def total_expenses : ℝ := car_maintenance_cost_per_week + (insurance_fee_per_day * 4)

-- Net income
def net_income : ℝ := total_income_with_tax - total_expenses

theorem james_net_income_correct : net_income = 415.30 :=
  by
    -- proof omitted
    sorry

end NUMINAMATH_GPT_james_net_income_correct_l1490_149040


namespace NUMINAMATH_GPT_not_age_of_child_digit_l1490_149055

variable {n : Nat}

theorem not_age_of_child_digit : 
  ∀ (ages : List Nat), 
    (∀ x ∈ ages, 5 ≤ x ∧ x ≤ 13) ∧ -- condition 1
    ages.Nodup ∧                    -- condition 2: distinct ages
    ages.length = 9 ∧               -- condition 1: 9 children
    (∃ num : Nat, 
       10000 ≤ num ∧ num < 100000 ∧         -- 5-digit number
       (∀ d : Nat, d ∈ num.digits 10 →     -- condition 3 & 4: each digit appears once and follows a consecutive pattern in increasing order
          1 ≤ d ∧ d ≤ 9) ∧
       (∀ age ∈ ages, num % age = 0)       -- condition 4: number divisible by all children's ages
    ) →
    ¬(9 ∈ ages) :=                         -- question: Prove that '9' is not the age of any child
by
  intro ages h
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_not_age_of_child_digit_l1490_149055


namespace NUMINAMATH_GPT_find_m_l1490_149049

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1490_149049


namespace NUMINAMATH_GPT_cube_split_with_333_l1490_149068

theorem cube_split_with_333 (m : ℕ) (h1 : m > 1)
  (h2 : ∃ k : ℕ, (333 = 2 * k + 1) ∧ (333 + 2 * (k - k) + 2) * k = m^3 ) :
  m = 18 := sorry

end NUMINAMATH_GPT_cube_split_with_333_l1490_149068


namespace NUMINAMATH_GPT_jayden_half_of_ernesto_in_some_years_l1490_149003

theorem jayden_half_of_ernesto_in_some_years :
  ∃ x : ℕ, (4 + x = (1 : ℝ) / 2 * (11 + x)) ∧ x = 3 := by
  sorry

end NUMINAMATH_GPT_jayden_half_of_ernesto_in_some_years_l1490_149003


namespace NUMINAMATH_GPT_sheep_count_l1490_149023

/-- The ratio between the number of sheep and the number of horses at the Stewart farm is 2 to 7.
    Each horse is fed 230 ounces of horse food per day, and the farm needs a total of 12,880 ounces
    of horse food per day. -/
theorem sheep_count (S H : ℕ) (h_ratio : S = (2 / 7) * H)
    (h_food : H * 230 = 12880) : S = 16 :=
sorry

end NUMINAMATH_GPT_sheep_count_l1490_149023


namespace NUMINAMATH_GPT_solve_for_x_y_l1490_149076

theorem solve_for_x_y (x y : ℚ) 
  (h1 : (3 * x + 12 + 2 * y + 18 + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) 
  (h2 : x = 2 * y) : 
  x = 254 / 15 ∧ y = 127 / 15 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_solve_for_x_y_l1490_149076


namespace NUMINAMATH_GPT_find_n_l1490_149018

-- Definitions based on conditions
variables (x n y : ℕ)
variable (h1 : x / n = 3 / 2)
variable (h2 : (7 * x + n * y) / (x - n * y) = 23)

-- Proof that n is equivalent to 1 given the conditions.
theorem find_n : n = 1 :=
sorry

end NUMINAMATH_GPT_find_n_l1490_149018


namespace NUMINAMATH_GPT_sum_of_tens_and_ones_digit_of_7_pow_17_l1490_149034

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tens_and_ones_digit_of_7_pow_17_l1490_149034


namespace NUMINAMATH_GPT_peter_remaining_money_l1490_149031

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end NUMINAMATH_GPT_peter_remaining_money_l1490_149031


namespace NUMINAMATH_GPT_jack_money_proof_l1490_149024

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end NUMINAMATH_GPT_jack_money_proof_l1490_149024


namespace NUMINAMATH_GPT_gcd_102_238_l1490_149011

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end NUMINAMATH_GPT_gcd_102_238_l1490_149011


namespace NUMINAMATH_GPT_sum_of_final_numbers_l1490_149025

theorem sum_of_final_numbers (x y : ℝ) (S : ℝ) (h : x + y = S) : 
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_GPT_sum_of_final_numbers_l1490_149025


namespace NUMINAMATH_GPT_solve_for_y_l1490_149072

theorem solve_for_y : ∃ y : ℝ, (2010 + y)^2 = y^2 ∧ y = -1005 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1490_149072


namespace NUMINAMATH_GPT_train_boxcar_capacity_l1490_149056

theorem train_boxcar_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_boxcar_capacity := 4000
  let blue_boxcar_capacity := 2 * black_boxcar_capacity
  let red_boxcar_capacity := 3 * blue_boxcar_capacity
  (red_boxcars * red_boxcar_capacity + blue_boxcars * blue_boxcar_capacity + black_boxcars * black_boxcar_capacity) = 132000 :=
by
  sorry

end NUMINAMATH_GPT_train_boxcar_capacity_l1490_149056


namespace NUMINAMATH_GPT_carol_seq_last_three_digits_l1490_149065

/-- Carol starts to make a list, in increasing order, of the positive integers that have 
    a first digit of 2. She writes 2, 20, 21, 22, ...
    Prove that the three-digit number formed by the 1198th, 1199th, 
    and 1200th digits she wrote is 218. -/
theorem carol_seq_last_three_digits : 
  (digits_1198th_1199th_1200th = 218) :=
by
  sorry

end NUMINAMATH_GPT_carol_seq_last_three_digits_l1490_149065


namespace NUMINAMATH_GPT_find_smaller_number_l1490_149033

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : y = 28.5 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1490_149033
