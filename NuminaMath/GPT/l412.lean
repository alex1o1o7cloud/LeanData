import Mathlib

namespace NUMINAMATH_GPT_green_or_blue_marble_probability_l412_41219

theorem green_or_blue_marble_probability :
  (4 + 3 : ℝ) / (4 + 3 + 8) = 0.4667 := by
  sorry

end NUMINAMATH_GPT_green_or_blue_marble_probability_l412_41219


namespace NUMINAMATH_GPT_max_apartment_size_l412_41250

/-- Define the rental rate and the maximum rent Michael can afford. -/
def rental_rate : ℝ := 1.20
def max_rent : ℝ := 720

/-- State the problem in Lean: Prove that the maximum apartment size Michael should consider is 600 square feet. -/
theorem max_apartment_size :
  ∃ s : ℝ, rental_rate * s = max_rent ∧ s = 600 := by
  sorry

end NUMINAMATH_GPT_max_apartment_size_l412_41250


namespace NUMINAMATH_GPT_max_expression_value_l412_41204

open Real

theorem max_expression_value : 
  ∃ q : ℝ, ∀ q : ℝ, -3 * q ^ 2 + 18 * q + 5 ≤ 32 ∧ (-3 * (3 ^ 2) + 18 * 3 + 5 = 32) :=
by
  sorry

end NUMINAMATH_GPT_max_expression_value_l412_41204


namespace NUMINAMATH_GPT_twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l412_41217

theorem twelve_plus_four_times_five_minus_five_cubed_equals_twelve :
  12 + 4 * (5 - 10 / 2) ^ 3 = 12 := by
  sorry

end NUMINAMATH_GPT_twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l412_41217


namespace NUMINAMATH_GPT_factorize_x_squared_minus_nine_l412_41284

theorem factorize_x_squared_minus_nine : ∀ (x : ℝ), x^2 - 9 = (x - 3) * (x + 3) :=
by
  intro x
  exact sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_nine_l412_41284


namespace NUMINAMATH_GPT_alice_needs_to_add_stamps_l412_41283

variable (A B E P D : ℕ)
variable (h₁ : B = 4 * E)
variable (h₂ : E = 3 * P)
variable (h₃ : P = 2 * D)
variable (h₄ : D = A + 5)
variable (h₅ : A = 65)

theorem alice_needs_to_add_stamps : (1680 - A = 1615) :=
by
  sorry

end NUMINAMATH_GPT_alice_needs_to_add_stamps_l412_41283


namespace NUMINAMATH_GPT_sunny_lead_l412_41262

-- Define the given conditions as hypotheses
variables (h d : ℝ) (s w : ℝ)
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h)

-- State the theorem we want to prove
theorem sunny_lead (h d : ℝ) (s w : ℝ) 
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h) :
    ∃ distance_ahead_Sunny : ℝ, distance_ahead_Sunny = (2 * d^2) / h :=
sorry

end NUMINAMATH_GPT_sunny_lead_l412_41262


namespace NUMINAMATH_GPT_find_other_number_l412_41238

noncomputable def calculateB (lcm hcf a : ℕ) : ℕ :=
  (lcm * hcf) / a

theorem find_other_number :
  ∃ B : ℕ, (calculateB 76176 116 8128) = 1087 :=
by
  use 1087
  sorry

end NUMINAMATH_GPT_find_other_number_l412_41238


namespace NUMINAMATH_GPT_value_of_D_l412_41282

variable (L E A D : ℤ)

-- given conditions
def LEAD := 41
def DEAL := 45
def ADDED := 53

-- condition that L = 15
axiom hL : L = 15

-- equations from the problem statement
def eq1 := L + E + A + D = 41
def eq2 := D + E + A + L = 45
def eq3 := A + 3 * D + E = 53

-- stating the problem as proving that D = 4 given the conditions
theorem value_of_D : D = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_D_l412_41282


namespace NUMINAMATH_GPT_ordered_sets_equal_l412_41215

theorem ordered_sets_equal
  (n : ℕ) 
  (h_gcd : gcd n 6 = 1) 
  (a b : ℕ → ℕ) 
  (h_order_a : ∀ {i j}, i < j → a i < a j)
  (h_order_b : ∀ {i j}, i < j → b i < b j) 
  (h_sum : ∀ {j k l : ℕ}, 1 ≤ j → j < k → k < l → l ≤ n → a j + a k + a l = b j + b k + b l) : 
  ∀ (j : ℕ), 1 ≤ j → j ≤ n → a j = b j := 
sorry

end NUMINAMATH_GPT_ordered_sets_equal_l412_41215


namespace NUMINAMATH_GPT_find_m_if_purely_imaginary_l412_41280

theorem find_m_if_purely_imaginary : ∀ m : ℝ, (m^2 - 5*m + 6 = 0) → (m = 2) :=
by 
  intro m
  intro h
  sorry

end NUMINAMATH_GPT_find_m_if_purely_imaginary_l412_41280


namespace NUMINAMATH_GPT_remaining_requests_after_7_days_l412_41240

-- Definitions based on the conditions
def dailyRequests : ℕ := 8
def dailyWork : ℕ := 4
def days : ℕ := 7

-- Theorem statement representing our final proof problem
theorem remaining_requests_after_7_days : 
  (dailyRequests * days - dailyWork * days) + dailyRequests * days = 84 := by
  sorry

end NUMINAMATH_GPT_remaining_requests_after_7_days_l412_41240


namespace NUMINAMATH_GPT_octahedron_has_constant_perimeter_cross_sections_l412_41253

structure Octahedron :=
(edge_length : ℝ)

def all_cross_sections_same_perimeter (oct : Octahedron) :=
  ∀ (face1 face2 : ℝ), (face1 = face2)

theorem octahedron_has_constant_perimeter_cross_sections (oct : Octahedron) :
  all_cross_sections_same_perimeter oct :=
  sorry

end NUMINAMATH_GPT_octahedron_has_constant_perimeter_cross_sections_l412_41253


namespace NUMINAMATH_GPT_expression_value_l412_41226

theorem expression_value : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by {
  sorry
}

end NUMINAMATH_GPT_expression_value_l412_41226


namespace NUMINAMATH_GPT_value_of_p_l412_41249

theorem value_of_p (p q : ℝ) (h1 : q = (2 / 5) * p) (h2 : p * q = 90) : p = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_p_l412_41249


namespace NUMINAMATH_GPT_Eric_return_time_l412_41218

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end NUMINAMATH_GPT_Eric_return_time_l412_41218


namespace NUMINAMATH_GPT_possible_triangular_frames_B_l412_41201

-- Define the sides of the triangles and the similarity condition
def similar_triangles (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * b₃ = a₃ * b₁ ∧ a₂ * b₃ = a₃ * b₂

def sides_of_triangle_A := (50, 60, 80)

def is_a_possible_triangle (b₁ b₂ b₃ : ℕ) : Prop :=
  similar_triangles 50 60 80 b₁ b₂ b₃

-- Given conditions
def side_of_triangle_B := 20

-- Theorem to prove
theorem possible_triangular_frames_B :
  ∃ (b₂ b₃ : ℕ), (is_a_possible_triangle 20 b₂ b₃ ∨ is_a_possible_triangle b₂ 20 b₃ ∨ is_a_possible_triangle b₂ b₃ 20) :=
sorry

end NUMINAMATH_GPT_possible_triangular_frames_B_l412_41201


namespace NUMINAMATH_GPT_find_angle_D_l412_41235

noncomputable def measure.angle_A := 80
noncomputable def measure.angle_B := 30
noncomputable def measure.angle_C := 20

def sum_angles_pentagon (A B C : ℕ) := 540 - (A + B + C)

theorem find_angle_D
  (A B C E F : ℕ)
  (hA : A = measure.angle_A)
  (hB : B = measure.angle_B)
  (hC : C = measure.angle_C)
  (h_sum_pentagon : A + B + C + D + E + F = 540)
  (h_triangle : D + E + F = 180) :
  D = 130 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_D_l412_41235


namespace NUMINAMATH_GPT_discriminant_quadratic_eq_l412_41263

theorem discriminant_quadratic_eq : 
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  Δ = 33 :=
by
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  exact sorry

end NUMINAMATH_GPT_discriminant_quadratic_eq_l412_41263


namespace NUMINAMATH_GPT_1_part1_2_part2_l412_41241

/-
Define M and N sets
-/
def M : Set ℝ := {x | x ≥ 1 / 2}
def N : Set ℝ := {y | y ≤ 1}

/-
Theorem 1: Difference set M - N
-/
theorem part1 : (M \ N) = {x | x > 1} := by
  sorry

/-
Define A and B sets and the condition A - B = ∅
-/
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {y | -1 / 2 < y ∧ y ≤ 2}

/-
Theorem 2: Range of values for a
-/
theorem part2 (a : ℝ) (h : A a \ B = ∅) : a ∈ Set.Iio (-12) ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_GPT_1_part1_2_part2_l412_41241


namespace NUMINAMATH_GPT_find_b7_l412_41254

/-- We represent the situation with twelve people in a circle, each with an integer number. The
     average announced by a person is the average of their two immediate neighbors. Given the
     person who announced the average of 7, we aim to find the number they initially chose. --/
theorem find_b7 (b : ℕ → ℕ) (announced_avg : ℕ → ℕ) :
  (announced_avg 1 = (b 12 + b 2) / 2) ∧
  (announced_avg 2 = (b 1 + b 3) / 2) ∧
  (announced_avg 3 = (b 2 + b 4) / 2) ∧
  (announced_avg 4 = (b 3 + b 5) / 2) ∧
  (announced_avg 5 = (b 4 + b 6) / 2) ∧
  (announced_avg 6 = (b 5 + b 7) / 2) ∧
  (announced_avg 7 = (b 6 + b 8) / 2) ∧
  (announced_avg 8 = (b 7 + b 9) / 2) ∧
  (announced_avg 9 = (b 8 + b 10) / 2) ∧
  (announced_avg 10 = (b 9 + b 11) / 2) ∧
  (announced_avg 11 = (b 10 + b 12) / 2) ∧
  (announced_avg 12 = (b 11 + b 1) / 2) ∧
  (announced_avg 7 = 7) →
  b 7 = 12 := 
sorry

end NUMINAMATH_GPT_find_b7_l412_41254


namespace NUMINAMATH_GPT_min_value_expression_l412_41231

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l412_41231


namespace NUMINAMATH_GPT_total_animals_counted_l412_41242

theorem total_animals_counted :
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  show (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605
  sorry

end NUMINAMATH_GPT_total_animals_counted_l412_41242


namespace NUMINAMATH_GPT_ebay_ordered_cards_correct_l412_41278

noncomputable def initial_cards := 4
noncomputable def father_cards := 13
noncomputable def cards_given_to_dexter := 29
noncomputable def cards_kept := 20
noncomputable def bad_cards := 4

theorem ebay_ordered_cards_correct :
  let total_before_ebay := initial_cards + father_cards
  let total_after_giving_and_keeping := cards_given_to_dexter + cards_kept
  let ordered_before_bad := total_after_giving_and_keeping - total_before_ebay
  let ebay_ordered_cards := ordered_before_bad + bad_cards
  ebay_ordered_cards = 36 :=
by
  sorry

end NUMINAMATH_GPT_ebay_ordered_cards_correct_l412_41278


namespace NUMINAMATH_GPT_fraction_value_l412_41272

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem
theorem fraction_value (h : 2 * x = -y) : (x * y) / (x^2 - y^2) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l412_41272


namespace NUMINAMATH_GPT_shells_not_red_or_green_l412_41233

theorem shells_not_red_or_green (total_shells : ℕ) (red_shells : ℕ) (green_shells : ℕ) 
  (h_total : total_shells = 291) (h_red : red_shells = 76) (h_green : green_shells = 49) :
  total_shells - (red_shells + green_shells) = 166 :=
by
  sorry

end NUMINAMATH_GPT_shells_not_red_or_green_l412_41233


namespace NUMINAMATH_GPT_perpendicular_line_passing_point_l412_41270

theorem perpendicular_line_passing_point (x y : ℝ) (hx : 4 * x - 3 * y + 2 = 0) : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ (3 * x + 4 * y + 1 = 0) → l 1 2) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_passing_point_l412_41270


namespace NUMINAMATH_GPT_reese_practice_hours_l412_41274

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end NUMINAMATH_GPT_reese_practice_hours_l412_41274


namespace NUMINAMATH_GPT_arccos_sin_1_5_eq_pi_over_2_minus_1_5_l412_41211

-- Define the problem statement in Lean 4.
theorem arccos_sin_1_5_eq_pi_over_2_minus_1_5 : 
  Real.arccos (Real.sin 1.5) = (Real.pi / 2) - 1.5 :=
by
  sorry

end NUMINAMATH_GPT_arccos_sin_1_5_eq_pi_over_2_minus_1_5_l412_41211


namespace NUMINAMATH_GPT_selling_price_of_mixture_l412_41266

noncomputable def selling_price_per_pound (weight1 weight2 price1 price2 total_weight : ℝ) : ℝ :=
  (weight1 * price1 + weight2 * price2) / total_weight

theorem selling_price_of_mixture :
  selling_price_per_pound 20 10 2.95 3.10 30 = 3.00 :=
by
  -- Skipping the proof part
  sorry

end NUMINAMATH_GPT_selling_price_of_mixture_l412_41266


namespace NUMINAMATH_GPT_man_swims_speed_l412_41247

theorem man_swims_speed (v_m v_s : ℝ) (h_downstream : 28 = (v_m + v_s) * 2) (h_upstream : 12 = (v_m - v_s) * 2) : v_m = 10 := 
by sorry

end NUMINAMATH_GPT_man_swims_speed_l412_41247


namespace NUMINAMATH_GPT_simplify_expression_l412_41246

variables (y : ℝ)

theorem simplify_expression : 
  3 * y + 4 * y^2 - 2 - (8 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 10 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l412_41246


namespace NUMINAMATH_GPT_lemons_for_10_gallons_l412_41299

noncomputable def lemon_proportion : Prop :=
  ∃ x : ℝ, (36 / 48) = (x / 10) ∧ x = 7.5

theorem lemons_for_10_gallons : lemon_proportion :=
by
  sorry

end NUMINAMATH_GPT_lemons_for_10_gallons_l412_41299


namespace NUMINAMATH_GPT_product_of_values_of_t_squared_eq_49_l412_41251

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_product_of_values_of_t_squared_eq_49_l412_41251


namespace NUMINAMATH_GPT_product_of_two_numbers_l412_41271

theorem product_of_two_numbers (x y : ℕ) 
  (h1 : y = 15 * x) 
  (h2 : x + y = 400) : 
  x * y = 9375 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l412_41271


namespace NUMINAMATH_GPT_fourth_student_in_sample_l412_41225

def sample_interval (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

def in_sample (student_number : ℕ) (start : ℕ) (interval : ℕ) (n : ℕ) : Prop :=
  student_number = start + n * interval

theorem fourth_student_in_sample :
  ∀ (total_students sample_size : ℕ) (s1 s2 s3 : ℕ),
    total_students = 52 →
    sample_size = 4 →
    s1 = 7 →
    s2 = 33 →
    s3 = 46 →
    ∃ s4, in_sample s4 s1 (sample_interval total_students sample_size) 1 ∧
           in_sample s2 s1 (sample_interval total_students sample_size) 2 ∧
           in_sample s3 s1 (sample_interval total_students sample_size) 3 ∧
           s4 = 20 := 
by
  sorry

end NUMINAMATH_GPT_fourth_student_in_sample_l412_41225


namespace NUMINAMATH_GPT_javier_initial_games_l412_41248

/--
Javier plays 2 baseball games a week. In each of his first some games, 
he averaged 2 hits. If he has 10 games left, he has to average 5 hits 
a game to bring his average for the season up to 3 hits a game. 
Prove that the number of games Javier initially played is 20.
-/
theorem javier_initial_games (x : ℕ) :
  (2 * x + 5 * 10) / (x + 10) = 3 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_javier_initial_games_l412_41248


namespace NUMINAMATH_GPT_largest_consecutive_odd_nat_divisible_by_3_sum_72_l412_41230

theorem largest_consecutive_odd_nat_divisible_by_3_sum_72
  (a : ℕ)
  (h₁ : a % 3 = 0)
  (h₂ : (a + 6) % 3 = 0)
  (h₃ : (a + 12) % 3 = 0)
  (h₄ : a % 2 = 1)
  (h₅ : (a + 6) % 2 = 1)
  (h₆ : (a + 12) % 2 = 1)
  (h₇ : a + (a + 6) + (a + 12) = 72) :
  a + 12 = 30 :=
by
  sorry

end NUMINAMATH_GPT_largest_consecutive_odd_nat_divisible_by_3_sum_72_l412_41230


namespace NUMINAMATH_GPT_volume_of_first_bottle_l412_41281

theorem volume_of_first_bottle (V_2 V_3 : ℕ) (V_total : ℕ):
  V_2 = 750 ∧ V_3 = 250 ∧ V_total = 3 * 1000 →
  (V_total - V_2 - V_3) / 1000 = 2 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_first_bottle_l412_41281


namespace NUMINAMATH_GPT_daily_production_l412_41210

-- Definitions based on conditions
def weekly_production : ℕ := 3400
def working_days_in_week : ℕ := 5

-- Statement to prove the number of toys produced each day
theorem daily_production : (weekly_production / working_days_in_week) = 680 :=
by
  sorry

end NUMINAMATH_GPT_daily_production_l412_41210


namespace NUMINAMATH_GPT_car_average_speed_l412_41229

theorem car_average_speed
  (d1 d2 t1 t2 : ℕ)
  (h1 : d1 = 85)
  (h2 : d2 = 45)
  (h3 : t1 = 1)
  (h4 : t2 = 1) :
  let total_distance := d1 + d2
  let total_time := t1 + t2
  (total_distance / total_time = 65) :=
by
  sorry

end NUMINAMATH_GPT_car_average_speed_l412_41229


namespace NUMINAMATH_GPT_pentagon_perpendicular_sums_l412_41259

noncomputable def FO := 2
noncomputable def FQ := 2
noncomputable def FR := 2

theorem pentagon_perpendicular_sums :
  FO + FQ + FR = 6 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_perpendicular_sums_l412_41259


namespace NUMINAMATH_GPT_weight_of_b_l412_41292

variable {A B C : ℤ}

def condition1 (A B C : ℤ) : Prop := (A + B + C) / 3 = 45
def condition2 (A B : ℤ) : Prop := (A + B) / 2 = 42
def condition3 (B C : ℤ) : Prop := (B + C) / 2 = 43

theorem weight_of_b (A B C : ℤ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B) 
  (h3 : condition3 B C) : 
  B = 35 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_b_l412_41292


namespace NUMINAMATH_GPT_atm_withdrawal_cost_l412_41256

theorem atm_withdrawal_cost (x y : ℝ)
  (h1 : 221 = x + 40000 * y)
  (h2 : 485 = x + 100000 * y) :
  (x + 85000 * y) = 419 := by
  sorry

end NUMINAMATH_GPT_atm_withdrawal_cost_l412_41256


namespace NUMINAMATH_GPT_contrapositive_quadratic_roots_l412_41264

theorem contrapositive_quadratic_roots (m : ℝ) (h_discriminant : 1 + 4 * m < 0) : m ≤ 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_quadratic_roots_l412_41264


namespace NUMINAMATH_GPT_distance_travelled_l412_41227

def actual_speed : ℝ := 50
def additional_speed : ℝ := 25
def time_difference : ℝ := 0.5

theorem distance_travelled (D : ℝ) : 0.5 = (D / actual_speed) - (D / (actual_speed + additional_speed)) → D = 75 :=
by sorry

end NUMINAMATH_GPT_distance_travelled_l412_41227


namespace NUMINAMATH_GPT_race_speed_ratio_l412_41236

theorem race_speed_ratio (L v_a v_b : ℝ) (h1 : v_a = v_b / 0.84375) :
  v_a / v_b = 32 / 27 :=
by sorry

end NUMINAMATH_GPT_race_speed_ratio_l412_41236


namespace NUMINAMATH_GPT_geometric_sequence_a4_a7_l412_41260

theorem geometric_sequence_a4_a7 (a : ℕ → ℝ) (h1 : ∃ a₁ a₁₀, a₁ * a₁₀ = -6 ∧ a 1 = a₁ ∧ a 10 = a₁₀) :
  a 4 * a 7 = -6 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a4_a7_l412_41260


namespace NUMINAMATH_GPT_cubic_difference_l412_41290

theorem cubic_difference (x y : ℝ) (h1 : x + y = 15) (h2 : 2 * x + y = 20) : x^3 - y^3 = -875 := 
by
  sorry

end NUMINAMATH_GPT_cubic_difference_l412_41290


namespace NUMINAMATH_GPT_painted_faces_of_large_cube_l412_41222

theorem painted_faces_of_large_cube (n : ℕ) (unpainted_cubes : ℕ) :
  n = 9 ∧ unpainted_cubes = 343 → (painted_faces : ℕ) = 3 :=
by
  intros h
  let ⟨h_n, h_unpainted⟩ := h
  sorry

end NUMINAMATH_GPT_painted_faces_of_large_cube_l412_41222


namespace NUMINAMATH_GPT_delivery_meals_l412_41252

theorem delivery_meals (M P : ℕ) 
  (h1 : P = 8 * M) 
  (h2 : M + P = 27) : 
  M = 3 := by
  sorry

end NUMINAMATH_GPT_delivery_meals_l412_41252


namespace NUMINAMATH_GPT_complex_eq_z100_zReciprocal_l412_41265

theorem complex_eq_z100_zReciprocal
  (z : ℂ)
  (h : z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + z⁻¹^100 = -2 * Real.cos (40 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_complex_eq_z100_zReciprocal_l412_41265


namespace NUMINAMATH_GPT_paula_karl_age_sum_l412_41245

theorem paula_karl_age_sum :
  ∃ (P K : ℕ), (P - 5 = 3 * (K - 5)) ∧ (P + 6 = 2 * (K + 6)) ∧ (P + K = 54) :=
by
  sorry

end NUMINAMATH_GPT_paula_karl_age_sum_l412_41245


namespace NUMINAMATH_GPT_product_evaluation_l412_41202

theorem product_evaluation : (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 + 4) = 5040 := by
  -- sorry
  exact rfl  -- This is just a placeholder. The proof would go here.

end NUMINAMATH_GPT_product_evaluation_l412_41202


namespace NUMINAMATH_GPT_pieces_per_box_correct_l412_41205

-- Define the number of boxes Will bought
def total_boxes_bought := 7

-- Define the number of boxes Will gave to his brother
def boxes_given := 3

-- Define the number of pieces left with Will
def pieces_left := 16

-- Define the function to find the pieces per box
def pieces_per_box (total_boxes : Nat) (given_away : Nat) (remaining_pieces : Nat) : Nat :=
  remaining_pieces / (total_boxes - given_away)

-- Prove that each box contains 4 pieces of chocolate candy
theorem pieces_per_box_correct : pieces_per_box total_boxes_bought boxes_given pieces_left = 4 :=
by
  sorry

end NUMINAMATH_GPT_pieces_per_box_correct_l412_41205


namespace NUMINAMATH_GPT_magic_square_proof_l412_41276

theorem magic_square_proof
    (a b c d e S : ℕ)
    (h1 : 35 + e + 27 = S)
    (h2 : 30 + c + d = S)
    (h3 : a + 32 + b = S)
    (h4 : 35 + c + b = S)
    (h5 : a + c + 27 = S)
    (h6 : 35 + c + b = S)
    (h7 : 35 + c + 27 = S)
    (h8 : a + c + d = S) :
  d + e = 35 :=
  sorry

end NUMINAMATH_GPT_magic_square_proof_l412_41276


namespace NUMINAMATH_GPT_find_y_l412_41208

theorem find_y 
  (h : (5 + 8 + 17) / 3 = (12 + y) / 2) : y = 8 :=
sorry

end NUMINAMATH_GPT_find_y_l412_41208


namespace NUMINAMATH_GPT_yanna_afternoon_baking_l412_41279

noncomputable def butter_cookies_in_afternoon (B : ℕ) : Prop :=
  let biscuits_afternoon := 20
  let butter_cookies_morning := 20
  let biscuits_morning := 40
  (biscuits_afternoon = B + 30) → B = 20

theorem yanna_afternoon_baking (h : butter_cookies_in_afternoon 20) : 20 = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_yanna_afternoon_baking_l412_41279


namespace NUMINAMATH_GPT_trigonometric_relationship_l412_41214

noncomputable def a : ℝ := Real.sin (393 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (50 * Real.pi / 180)

theorem trigonometric_relationship : a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_trigonometric_relationship_l412_41214


namespace NUMINAMATH_GPT_how_many_more_red_balls_l412_41296

def r_packs : ℕ := 12
def y_packs : ℕ := 9
def r_balls_per_pack : ℕ := 24
def y_balls_per_pack : ℕ := 20

theorem how_many_more_red_balls :
  (r_packs * r_balls_per_pack) - (y_packs * y_balls_per_pack) = 108 :=
by
  sorry

end NUMINAMATH_GPT_how_many_more_red_balls_l412_41296


namespace NUMINAMATH_GPT_solve_inequality_l412_41257

theorem solve_inequality (x : ℝ) : 
  3*x^2 + 2*x - 3 > 10 - 2*x ↔ x < ( -2 - Real.sqrt 43 ) / 3 ∨ x > ( -2 + Real.sqrt 43 ) / 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l412_41257


namespace NUMINAMATH_GPT_time_for_A_to_complete_race_l412_41286

open Real

theorem time_for_A_to_complete_race (V_A V_B : ℝ) (T_A : ℝ) :
  (V_B = 4) →
  (V_B = 960 / T_A) →
  T_A = 1000 / V_A →
  T_A = 240 := by
  sorry

end NUMINAMATH_GPT_time_for_A_to_complete_race_l412_41286


namespace NUMINAMATH_GPT_find_number_l412_41295

theorem find_number (x : ℕ) (h : x + 8 = 500) : x = 492 :=
by sorry

end NUMINAMATH_GPT_find_number_l412_41295


namespace NUMINAMATH_GPT_gift_bags_needed_l412_41269

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end NUMINAMATH_GPT_gift_bags_needed_l412_41269


namespace NUMINAMATH_GPT_slope_of_dividing_line_l412_41213

/--
Given a rectangle with vertices at (0,0), (0,4), (5,4), (5,2),
and a right triangle with vertices at (5,2), (7,2), (5,0),
prove that the slope of the line through the origin that divides the area
of this L-shaped region exactly in half is 16/11.
-/
theorem slope_of_dividing_line :
  let rectangle_area := 5 * 4
  let triangle_area := (1 / 2) * 2 * 2
  let total_area := rectangle_area + triangle_area
  let half_area := total_area / 2
  let x_division := half_area / 4
  let slope := 4 / x_division
  slope = 16 / 11 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_dividing_line_l412_41213


namespace NUMINAMATH_GPT_intersection_complement_N_l412_41277

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}
def C_U_M : Set ℕ := U \ M

theorem intersection_complement_N : (C_U_M ∩ N) = {4, 6} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_N_l412_41277


namespace NUMINAMATH_GPT_kangaroo_meetings_l412_41293

/-- 
Two kangaroos, A and B, start at point A and jump in specific sequences:
- Kangaroo A jumps in the sequence A, B, C, D, E, F, G, H, I, A, B, C, ... in a loop every 9 jumps.
- Kangaroo B jumps in the sequence A, B, D, E, G, H, A, B, D, ... in a loop every 6 jumps.
They start at point A together. Prove that they will land on the same point 226 times after 2017 jumps.
-/
theorem kangaroo_meetings (n : Nat) (ka : Fin 9 → Fin 9) (kb : Fin 6 → Fin 6)
  (hka : ∀ i, ka i = (i + 1) % 9) (hkb : ∀ i, kb i = (i + 1) % 6) :
  n = 2017 →
  -- Prove that the two kangaroos will meet 226 times after 2017 jumps
  ∃ k, k = 226 :=
by
  sorry

end NUMINAMATH_GPT_kangaroo_meetings_l412_41293


namespace NUMINAMATH_GPT_polynomial_remainder_theorem_l412_41267

open Polynomial

theorem polynomial_remainder_theorem (Q : Polynomial ℝ)
  (h1 : Q.eval 20 = 120)
  (h2 : Q.eval 100 = 40) :
  ∃ R : Polynomial ℝ, R.degree < 2 ∧ Q = (X - 20) * (X - 100) * R + (-X + 140) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_theorem_l412_41267


namespace NUMINAMATH_GPT_correct_answer_l412_41255

def sum_squares_of_three_consecutive_even_integers (n : ℤ) : ℤ :=
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  a * a + b * b + c * c

def T : Set ℤ :=
  {t | ∃ n : ℤ, t = sum_squares_of_three_consecutive_even_integers n}

theorem correct_answer : (∀ t ∈ T, t % 4 = 0) ∧ (∀ t ∈ T, t % 7 ≠ 0) :=
sorry

end NUMINAMATH_GPT_correct_answer_l412_41255


namespace NUMINAMATH_GPT_tetrahedrons_volume_proportional_l412_41203

-- Define the scenario and conditions.
variable 
  (V V' : ℝ) -- Volumes of the tetrahedrons
  (a b c a' b' c' : ℝ) -- Edge lengths emanating from vertices O and O'
  (α : ℝ) -- The angle between vectors OB and OC which is assumed to be congruent

-- Theorem statement.
theorem tetrahedrons_volume_proportional
  (congruent_trihedral_angles_at_O_and_O' : α = α) -- Condition of congruent trihedral angles
  : (V' / V) = (a' * b' * c') / (a * b * c) :=
sorry

end NUMINAMATH_GPT_tetrahedrons_volume_proportional_l412_41203


namespace NUMINAMATH_GPT_original_length_l412_41291

-- Definitions based on conditions
def length_sawed_off : ℝ := 0.33
def remaining_length : ℝ := 0.08

-- The problem statement translated to a Lean 4 theorem
theorem original_length (L : ℝ) (h1 : L = length_sawed_off + remaining_length) : 
  L = 0.41 :=
by
  sorry

end NUMINAMATH_GPT_original_length_l412_41291


namespace NUMINAMATH_GPT_path_length_of_dot_l412_41273

-- Define the edge length of the cube
def edge_length : ℝ := 3

-- Define the conditions of the problem
def cube_condition (l : ℝ) (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) : Prop :=
  l = edge_length ∧ rolling_without_slipping ∧ at_least_two_vertices_touching ∧ dot_at_one_corner ∧ returns_to_original_position

-- Define the theorem to be proven
theorem path_length_of_dot (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) :
  cube_condition edge_length rolling_without_slipping at_least_two_vertices_touching dot_at_one_corner returns_to_original_position →
  ∃ c : ℝ, c = 6 ∧ (c * Real.pi) = 6 * Real.pi :=
by
  intro h
  sorry

end NUMINAMATH_GPT_path_length_of_dot_l412_41273


namespace NUMINAMATH_GPT_find_difference_of_a_and_b_l412_41228

-- Define the conditions
variables (a b : ℝ)
axiom cond1 : 4 * a + 3 * b = 8
axiom cond2 : 3 * a + 4 * b = 6

-- Statement for the proof
theorem find_difference_of_a_and_b : a - b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_of_a_and_b_l412_41228


namespace NUMINAMATH_GPT_determine_angle_range_l412_41289

variable (α : ℝ)

theorem determine_angle_range 
  (h1 : 0 < α) 
  (h2 : α < 2 * π) 
  (h_sin : Real.sin α < 0) 
  (h_cos : Real.cos α > 0) : 
  (3 * π / 2 < α ∧ α < 2 * π) := 
sorry

end NUMINAMATH_GPT_determine_angle_range_l412_41289


namespace NUMINAMATH_GPT_no_matching_option_for_fraction_l412_41294

theorem no_matching_option_for_fraction (m n : ℕ) (h : m = 16 ^ 500) : 
  (m / 8 ≠ 8 ^ 499) ∧ 
  (m / 8 ≠ 4 ^ 999) ∧ 
  (m / 8 ≠ 2 ^ 1998) ∧ 
  (m / 8 ≠ 4 ^ 498) ∧ 
  (m / 8 ≠ 2 ^ 1994) := 
by {
  sorry
}

end NUMINAMATH_GPT_no_matching_option_for_fraction_l412_41294


namespace NUMINAMATH_GPT_intersection_complement_l412_41212

def U : Set ℤ := Set.univ
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≠ x}
def C_U_B : Set ℤ := {x | x ≠ 0 ∧ x ≠ 1}

theorem intersection_complement :
  A ∩ C_U_B = {-1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l412_41212


namespace NUMINAMATH_GPT_bruce_bank_ratio_l412_41261

noncomputable def bruce_aunt : ℝ := 75
noncomputable def bruce_grandfather : ℝ := 150
noncomputable def bruce_bank : ℝ := 45
noncomputable def bruce_total : ℝ := bruce_aunt + bruce_grandfather
noncomputable def bruce_ratio : ℝ := bruce_bank / bruce_total

theorem bruce_bank_ratio :
  bruce_ratio = 1 / 5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_bruce_bank_ratio_l412_41261


namespace NUMINAMATH_GPT_simplify_expression_l412_41285

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l412_41285


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l412_41206

theorem number_of_terms_in_arithmetic_sequence 
  (a : ℕ)
  (d : ℕ)
  (an : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : an = 47) :
  ∃ n : ℕ, an = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l412_41206


namespace NUMINAMATH_GPT_power_multiplication_l412_41234

theorem power_multiplication (a : ℝ) (b : ℝ) (m : ℕ) (n : ℕ) (h1 : a = 0.25) (h2 : b = 4) (h3 : m = 2023) (h4 : n = 2024) : 
  a^m * b^n = 4 := 
by 
  sorry

end NUMINAMATH_GPT_power_multiplication_l412_41234


namespace NUMINAMATH_GPT_solve_exponent_equation_l412_41297

theorem solve_exponent_equation (x y z : ℕ) :
  7^x + 1 = 3^y + 5^z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_exponent_equation_l412_41297


namespace NUMINAMATH_GPT_value_of_t_l412_41275

def vec (x y : ℝ) := (x, y)

def p := vec 3 3
def q := vec (-1) 2
def r := vec 4 1

noncomputable def t := 3

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem value_of_t (t : ℝ) : (dot_product (vec (6 + 4 * t) (6 + t)) q = 0) ↔ t = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_t_l412_41275


namespace NUMINAMATH_GPT_no_unsatisfactory_grades_l412_41223

theorem no_unsatisfactory_grades (total_students : ℕ)
  (top_marks : ℕ) (average_marks : ℕ) (good_marks : ℕ)
  (h1 : top_marks = total_students / 6)
  (h2 : average_marks = total_students / 3)
  (h3 : good_marks = total_students / 2) :
  total_students = top_marks + average_marks + good_marks := by
  sorry

end NUMINAMATH_GPT_no_unsatisfactory_grades_l412_41223


namespace NUMINAMATH_GPT_lcm_ac_is_420_l412_41232

theorem lcm_ac_is_420 (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
    Nat.lcm a c = 420 :=
sorry

end NUMINAMATH_GPT_lcm_ac_is_420_l412_41232


namespace NUMINAMATH_GPT_measure_angle_A_l412_41258

-- Angles A and B are supplementary
def supplementary (A B : ℝ) : Prop :=
  A + B = 180

-- Definition of the problem conditions
def problem_conditions (A B : ℝ) : Prop :=
  supplementary A B ∧ A = 4 * B

-- The measure of angle A
def measure_of_A := 144

-- The statement to prove
theorem measure_angle_A (A B : ℝ) :
  problem_conditions A B → A = measure_of_A := 
by
  sorry

end NUMINAMATH_GPT_measure_angle_A_l412_41258


namespace NUMINAMATH_GPT_total_pencils_bought_l412_41288

theorem total_pencils_bought (x y : ℕ) (y_pos : 0 < y) (initial_cost : y * (x + 10) = 5 * x) (later_cost : (4 * y) * (x + 10) = 20 * x) :
    x = 15 → (40 = x + x + 10) ∨ x = 40 → (90 = x + (x + 10)) :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_bought_l412_41288


namespace NUMINAMATH_GPT_nancy_packs_of_crayons_l412_41287

theorem nancy_packs_of_crayons (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end NUMINAMATH_GPT_nancy_packs_of_crayons_l412_41287


namespace NUMINAMATH_GPT_tangency_of_abs_and_circle_l412_41244

theorem tangency_of_abs_and_circle (a : ℝ) (ha_pos : a > 0) (ha_ne_two : a ≠ 2) :
    (y = abs x ∧ ∀ x, y = abs x → x^2 + (y - a)^2 = 2 * (a - 2)^2)
    → (a = 4/3 ∨ a = 4) := sorry

end NUMINAMATH_GPT_tangency_of_abs_and_circle_l412_41244


namespace NUMINAMATH_GPT_insects_ratio_l412_41216

theorem insects_ratio (total_insects : ℕ) (geckos : ℕ) (gecko_insects : ℕ) (lizards : ℕ)
  (H1 : geckos * gecko_insects + lizards * ((total_insects - geckos * gecko_insects) / lizards) = total_insects)
  (H2 : total_insects = 66)
  (H3 : geckos = 5)
  (H4 : gecko_insects = 6)
  (H5 : lizards = 3) :
  (total_insects - geckos * gecko_insects) / lizards / gecko_insects = 2 :=
by
  sorry

end NUMINAMATH_GPT_insects_ratio_l412_41216


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l412_41298

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l412_41298


namespace NUMINAMATH_GPT_length_of_BC_l412_41243

theorem length_of_BC (AB AC AM : ℝ) (hAB : AB = 5) (hAC : AC = 8) (hAM : AM = 4.5) : 
  ∃ BC, BC = Real.sqrt 97 :=
by
  sorry

end NUMINAMATH_GPT_length_of_BC_l412_41243


namespace NUMINAMATH_GPT_total_votes_cast_l412_41221

theorem total_votes_cast (V : ℕ) (h1 : V > 0) (h2 : ∃ c r : ℕ, c = 40 * V / 100 ∧ r = 40 * V / 100 + 5000 ∧ c + r = V):
  V = 25000 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l412_41221


namespace NUMINAMATH_GPT_fraction_evaluation_l412_41200

theorem fraction_evaluation :
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l412_41200


namespace NUMINAMATH_GPT_compute_alpha_powers_l412_41209

variable (α1 α2 α3 : ℂ)

open Complex

-- Given conditions
def condition1 : Prop := α1 + α2 + α3 = 2
def condition2 : Prop := α1^2 + α2^2 + α3^2 = 6
def condition3 : Prop := α1^3 + α2^3 + α3^3 = 14

-- The required proof statement
theorem compute_alpha_powers (h1 : condition1 α1 α2 α3) (h2 : condition2 α1 α2 α3) (h3 : condition3 α1 α2 α3) :
  α1^7 + α2^7 + α3^7 = 46 := by
  sorry

end NUMINAMATH_GPT_compute_alpha_powers_l412_41209


namespace NUMINAMATH_GPT_sequence_a_2017_l412_41207

theorem sequence_a_2017 :
  (∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2016 * a n / (2014 * a n + 2016)) → a 2017 = 1008 / (1007 * 2017 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_2017_l412_41207


namespace NUMINAMATH_GPT_a_1000_value_l412_41268

open Nat

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), (a 1 = 1010) ∧ (a 2 = 1011) ∧ 
  (∀ n ≥ 1, a n + a (n+1) + a (n+2) = 2 * n) ∧ 
  (a 1000 = 1676) :=
sorry

end NUMINAMATH_GPT_a_1000_value_l412_41268


namespace NUMINAMATH_GPT_pqrs_inequality_l412_41220

theorem pqrs_inequality (p q r : ℝ) (h_condition : ∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - p) * (x - q)) / (x - r) ≥ 0)
  (h_pq : p < q) : p = 28 ∧ q = 32 ∧ r = -6 ∧ p + 2 * q + 3 * r = 78 :=
by
  sorry

end NUMINAMATH_GPT_pqrs_inequality_l412_41220


namespace NUMINAMATH_GPT_simplify_expression_l412_41237

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4 + 3*z^2) = -1 - 8*z^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l412_41237


namespace NUMINAMATH_GPT_golden_ratio_expression_l412_41224

variables (R : ℝ)
noncomputable def divide_segment (R : ℝ) := R^(R^(R^2 + 1/R) + 1/R) + 1/R

theorem golden_ratio_expression :
  (R = (1 / (1 + R))) →
  divide_segment R = 2 :=
by
  sorry

end NUMINAMATH_GPT_golden_ratio_expression_l412_41224


namespace NUMINAMATH_GPT_product_is_48_l412_41239

-- Define the conditions and the target product
def problem (x y : ℝ) := 
  x ≠ y ∧ (x + y) / (x - y) = 7 ∧ (x * y) / (x - y) = 24

-- Prove that the product is 48 given the conditions
theorem product_is_48 (x y : ℝ) (h : problem x y) : x * y = 48 :=
sorry

end NUMINAMATH_GPT_product_is_48_l412_41239
