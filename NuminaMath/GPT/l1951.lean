import Mathlib

namespace factorize_expression_l1951_195199

-- Define that a and b are arbitrary real numbers
variables (a b : ℝ)

-- The theorem statement claiming that 3a²b - 12b equals the factored form 3b(a + 2)(a - 2)
theorem factorize_expression : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) :=
by
  sorry  -- proof omitted

end factorize_expression_l1951_195199


namespace sum_of_three_digit_positive_integers_l1951_195191

noncomputable def sum_of_arithmetic_series (a l n : ℕ) : ℕ :=
  (a + l) / 2 * n

theorem sum_of_three_digit_positive_integers : 
  sum_of_arithmetic_series 100 999 900 = 494550 :=
by
  -- skipping the proof
  sorry

end sum_of_three_digit_positive_integers_l1951_195191


namespace necessary_but_not_sufficient_l1951_195115

def angle_of_inclination (α : ℝ) : Prop :=
  α > Real.pi / 4

def slope_of_line (k : ℝ) : Prop :=
  k > 1

theorem necessary_but_not_sufficient (α k : ℝ) :
  angle_of_inclination α → (slope_of_line k → (k = Real.tan α)) → (angle_of_inclination α → slope_of_line k) ∧ ¬(slope_of_line k → angle_of_inclination α) :=
by
  sorry

end necessary_but_not_sufficient_l1951_195115


namespace ratio_proof_l1951_195176

theorem ratio_proof (a b c d e : ℕ) (h1 : a * 4 = 3 * b) (h2 : b * 9 = 7 * c)
  (h3 : c * 7 = 5 * d) (h4 : d * 13 = 11 * e) : a * 468 = 165 * e :=
by
  sorry

end ratio_proof_l1951_195176


namespace total_investment_sum_l1951_195160

theorem total_investment_sum :
  let R : ℝ := 2200
  let T : ℝ := R - 0.1 * R
  let V : ℝ := T + 0.1 * T
  R + T + V = 6358 := by
  sorry

end total_investment_sum_l1951_195160


namespace probability_at_least_one_consonant_l1951_195187

def letters := ["k", "h", "a", "n", "t", "k", "a", "r"]
def consonants := ["k", "h", "n", "t", "r"]
def vowels := ["a", "a"]

def num_letters := 7
def num_consonants := 5
def num_vowels := 2

def probability_no_consonants : ℚ := (num_vowels / num_letters) * ((num_vowels - 1) / (num_letters - 1))

def complement_rule (p: ℚ) : ℚ := 1 - p

theorem probability_at_least_one_consonant :
  complement_rule probability_no_consonants = 20/21 :=
by
  sorry

end probability_at_least_one_consonant_l1951_195187


namespace part1_solution_part2_solution_l1951_195114

-- Definitions for propositions p and q
def p (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- The actual Lean 4 statements
theorem part1_solution (x : ℝ) (m : ℝ) (hm : m = 1) (hp : p m x) (hq : q x) : 2 ≤ x ∧ x < 3 := by
  sorry

theorem part2_solution (m : ℝ) (hm : m > 0) (hsuff : ∀ x, q x → p m x) : (4 / 3) < m ∧ m < 2 := by
  sorry

end part1_solution_part2_solution_l1951_195114


namespace selling_price_is_80000_l1951_195148

-- Given the conditions of the problem
def purchasePrice : ℕ := 45000
def repairCosts : ℕ := 12000
def profitPercent : ℚ := 40.35 / 100

-- Total cost calculation
def totalCost := purchasePrice + repairCosts

-- Profit calculation
def profit := profitPercent * totalCost

-- Selling price calculation
def sellingPrice := totalCost + profit

-- Statement of the proof problem
theorem selling_price_is_80000 : round sellingPrice = 80000 := by
  sorry

end selling_price_is_80000_l1951_195148


namespace rate_of_grapes_calculation_l1951_195186

theorem rate_of_grapes_calculation (total_cost cost_mangoes cost_grapes : ℕ) (rate_grapes : ℕ):
  total_cost = 1125 →
  cost_mangoes = 9 * 55 →
  cost_grapes = 9 * rate_grapes →
  total_cost = cost_grapes + cost_mangoes →
  rate_grapes = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end rate_of_grapes_calculation_l1951_195186


namespace trigonometric_identity_l1951_195119

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end trigonometric_identity_l1951_195119


namespace megan_markers_l1951_195138

def initial_markers : ℕ := 217
def roberts_gift : ℕ := 109
def sarah_took : ℕ := 35

def final_markers : ℕ := initial_markers + roberts_gift - sarah_took

theorem megan_markers : final_markers = 291 := by
  sorry

end megan_markers_l1951_195138


namespace triangle_A_l1951_195177

variables {a b c : ℝ}
variables (A B C : ℝ) -- Represent vertices
variables (C1 C2 A1 A2 B1 B2 A' B' C' : ℝ)

-- Definition of equilateral triangle
def is_equilateral_trig (x y z : ℝ) : Prop :=
  dist x y = dist y z ∧ dist y z = dist z x

-- Given conditions
axiom ABC_equilateral : is_equilateral_trig A B C
axiom length_cond_1 : dist A1 A2 = a ∧ dist C B1 = a ∧ dist B C2 = a
axiom length_cond_2 : dist B1 B2 = b ∧ dist A C1 = b ∧ dist C A2 = b
axiom length_cond_3 : dist C1 C2 = c ∧ dist B A1 = c ∧ dist A B2 = c

-- Additional constructions
axiom A'_construction : is_equilateral_trig A' B2 C1
axiom B'_construction : is_equilateral_trig B' C2 A1
axiom C'_construction : is_equilateral_trig C' A2 B1

-- The final proof goal
theorem triangle_A'B'C'_equilateral : is_equilateral_trig A' B' C' :=
sorry

end triangle_A_l1951_195177


namespace binomial_square_l1951_195159

theorem binomial_square (a b : ℝ) : (2 * a - 3 * b)^2 = 4 * a^2 - 12 * a * b + 9 * b^2 :=
by
  sorry

end binomial_square_l1951_195159


namespace percentage_increase_in_spending_l1951_195111

variables (P Q : ℝ)
-- Conditions
def price_increase (P : ℝ) := 1.25 * P
def quantity_decrease (Q : ℝ) := 0.88 * Q

-- Mathemtically equivalent proof problem in Lean:
theorem percentage_increase_in_spending (P Q : ℝ) : 
  (price_increase P) * (quantity_decrease Q) / (P * Q) = 1.10 :=
by
  sorry

end percentage_increase_in_spending_l1951_195111


namespace value_expression_l1951_195164

theorem value_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 :=
by
  sorry

end value_expression_l1951_195164


namespace time_to_meet_l1951_195183

-- Definitions based on conditions
def motorboat_speed_Serezha : ℝ := 20 -- km/h
def crossing_time_Serezha : ℝ := 0.5 -- hours (30 minutes)
def running_speed_Dima : ℝ := 6 -- km/h
def running_time_Dima : ℝ := 0.25 -- hours (15 minutes)
def combined_speed : ℝ := running_speed_Dima + running_speed_Dima -- equal speeds running towards each other
def distance_meet : ℝ := (running_speed_Dima * running_time_Dima) -- The distance they need to cover towards each other

-- Prove the time for them to meet
theorem time_to_meet : (distance_meet / combined_speed) = (7.5 / 60) :=
by
  sorry

end time_to_meet_l1951_195183


namespace triangle_right_triangle_of_consecutive_integers_sum_l1951_195182

theorem triangle_right_triangle_of_consecutive_integers_sum (
  m n : ℕ
) (h1 : 0 < m) (h2 : n^2 = 2*m + 1) : 
  n * n + m * m = (m + 1) * (m + 1) := 
sorry

end triangle_right_triangle_of_consecutive_integers_sum_l1951_195182


namespace mitch_family_milk_l1951_195126

variable (total_milk soy_milk regular_milk : ℚ)

-- Conditions
axiom cond1 : total_milk = 0.6
axiom cond2 : soy_milk = 0.1
axiom cond3 : regular_milk + soy_milk = total_milk

-- Theorem statement
theorem mitch_family_milk : regular_milk = 0.5 :=
by
  sorry

end mitch_family_milk_l1951_195126


namespace solve_for_x_l1951_195128

theorem solve_for_x (x : ℝ) (h : x ≠ 2) : (7 * x) / (x - 2) - 5 / (x - 2) = 3 / (x - 2) → x = 8 / 7 :=
by
  sorry

end solve_for_x_l1951_195128


namespace fiona_reaches_pad_thirteen_without_predators_l1951_195127

noncomputable def probability_reach_pad_thirteen : ℚ := sorry

theorem fiona_reaches_pad_thirteen_without_predators :
  probability_reach_pad_thirteen = 3 / 2048 :=
sorry

end fiona_reaches_pad_thirteen_without_predators_l1951_195127


namespace ratio_change_factor_is_5_l1951_195120

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end ratio_change_factor_is_5_l1951_195120


namespace correct_log_conclusions_l1951_195153

variables {x₁ x₂ : ℝ} (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (h_diff : x₁ ≠ x₂)
noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem correct_log_conclusions :
  ¬ (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ¬ ((f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by {
  sorry
}

end correct_log_conclusions_l1951_195153


namespace preimage_of_3_1_l1951_195142

theorem preimage_of_3_1 (a b : ℝ) (f : ℝ × ℝ → ℝ × ℝ) (h : ∀ (a b : ℝ), f (a, b) = (a + 2 * b, 2 * a - b)) :
  f (1, 1) = (3, 1) :=
by {
  sorry
}

end preimage_of_3_1_l1951_195142


namespace minimum_value_of_h_l1951_195179

noncomputable def h (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x))^2)

theorem minimum_value_of_h : (∀ x : ℝ, x > 0 → h x ≥ 2.25) ∧ (h 1 = 2.25) :=
by
  sorry

end minimum_value_of_h_l1951_195179


namespace negation_exists_real_negation_of_quadratic_l1951_195197

theorem negation_exists_real (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

def quadratic (x : ℝ) : Prop := x^2 - 2*x + 3 ≤ 0

theorem negation_of_quadratic :
  (¬ ∀ x : ℝ, quadratic x) ↔ ∃ x : ℝ, ¬ quadratic x :=
by exact negation_exists_real quadratic

end negation_exists_real_negation_of_quadratic_l1951_195197


namespace jason_earned_amount_l1951_195198

theorem jason_earned_amount (init_jason money_jason : ℤ)
    (h0 : init_jason = 3)
    (h1 : money_jason = 63) :
    money_jason - init_jason = 60 := 
by
  sorry

end jason_earned_amount_l1951_195198


namespace slope_of_line_AB_is_pm_4_3_l1951_195171

noncomputable def slope_of_line_AB : ℝ := sorry

theorem slope_of_line_AB_is_pm_4_3 (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 4 * x₁)
  (h₂ : y₂^2 = 4 * x₂)
  (h₃ : (x₁, y₁) ≠ (x₂, y₂))
  (h₄ : (x₁ - 1, y₁) = -4 * (x₂ - 1, y₂)) :
  slope_of_line_AB = 4 / 3 ∨ slope_of_line_AB = -4 / 3 :=
sorry

end slope_of_line_AB_is_pm_4_3_l1951_195171


namespace valentines_given_l1951_195130

-- Let x be the number of boys and y be the number of girls
variables (x y : ℕ)

-- Condition 1: the number of valentines is 28 more than the total number of students.
axiom valentines_eq : x * y = x + y + 28

-- Theorem: Prove that the total number of valentines given is 60.
theorem valentines_given : x * y = 60 :=
by
  sorry

end valentines_given_l1951_195130


namespace lines_non_intersect_l1951_195125

theorem lines_non_intersect (k : ℝ) : 
  (¬∃ t s : ℝ, (1 + 2 * t = -1 + 3 * s ∧ 3 - 5 * t = 4 + k * s)) → 
  k = -15 / 2 :=
by
  intro h
  -- Now left to define proving steps using sorry
  sorry

end lines_non_intersect_l1951_195125


namespace original_number_is_80_l1951_195151

theorem original_number_is_80 (t : ℝ) (h : t * 1.125 - t * 0.75 = 30) : t = 80 := by
  sorry

end original_number_is_80_l1951_195151


namespace sufficient_but_not_necessary_l1951_195161

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 2 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_l1951_195161


namespace problem_statement_l1951_195141

theorem problem_statement (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 :=
by
  sorry

end problem_statement_l1951_195141


namespace g_29_eq_27_l1951_195185

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x : ℝ, g (x + g x) = 3 * g x
axiom initial_condition : g 2 = 9

theorem g_29_eq_27 : g 29 = 27 := by
  sorry

end g_29_eq_27_l1951_195185


namespace paul_needs_score_to_achieve_mean_l1951_195195

theorem paul_needs_score_to_achieve_mean (x : ℤ) :
  (78 + 84 + 76 + 82 + 88 + x) / 6 = 85 → x = 102 :=
by 
  sorry

end paul_needs_score_to_achieve_mean_l1951_195195


namespace radius_of_base_of_cone_correct_l1951_195192

noncomputable def radius_of_base_of_cone (n : ℕ) (r α : ℝ) : ℝ :=
  r * (1 / Real.sin (Real.pi / n) - 1 / Real.tan (Real.pi / 4 + α / 2))

theorem radius_of_base_of_cone_correct :
  radius_of_base_of_cone 11 3 (Real.pi / 6) = 3 / Real.sin (Real.pi / 11) - Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_correct_l1951_195192


namespace geometric_seq_min_value_l1951_195124

theorem geometric_seq_min_value (b : ℕ → ℝ) (s : ℝ) (h1 : b 1 = 1) (h2 : ∀ n : ℕ, b (n + 1) = s * b n) : 
  ∃ s : ℝ, 3 * b 1 + 4 * b 2 = -9 / 16 :=
by
  sorry

end geometric_seq_min_value_l1951_195124


namespace Nancy_shelved_biographies_l1951_195152

def NancyBooks.shelved_books_from_top : Nat := 12 + 8 + 4 -- history + romance + poetry
def NancyBooks.total_books_on_cart : Nat := 46
def NancyBooks.bottom_books_after_top_shelved : Nat := 46 - 24
def NancyBooks.mystery_books_on_bottom : Nat := NancyBooks.bottom_books_after_top_shelved / 2
def NancyBooks.western_novels_on_bottom : Nat := 5
def NancyBooks.biographies : Nat := NancyBooks.bottom_books_after_top_shelved - NancyBooks.mystery_books_on_bottom - NancyBooks.western_novels_on_bottom

theorem Nancy_shelved_biographies : NancyBooks.biographies = 6 := by
  sorry

end Nancy_shelved_biographies_l1951_195152


namespace greatest_divisor_of_420_and_90_l1951_195108

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- Main problem statement
theorem greatest_divisor_of_420_and_90 {d : ℕ} :
  (divides d 420) ∧ (d < 60) ∧ (divides d 90) → d ≤ 30 := 
sorry

end greatest_divisor_of_420_and_90_l1951_195108


namespace present_age_of_son_l1951_195163

theorem present_age_of_son (F S : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 := by
  sorry

end present_age_of_son_l1951_195163


namespace max_cities_l1951_195118

theorem max_cities (n : ℕ) (h1 : ∀ (c : Fin n), ∃ (neighbors : Finset (Fin n)), neighbors.card ≤ 3 ∧ c ∈ neighbors) (h2 : ∀ (c1 c2 : Fin n), c1 ≠ c2 → ∃ c : Fin n, c1 ≠ c ∧ c2 ≠ c) : n ≤ 10 := 
sorry

end max_cities_l1951_195118


namespace value_of_x_l1951_195134

noncomputable def k := 9

theorem value_of_x (y : ℝ) (h1 : y = 3) (h2 : ∀ (x : ℝ), x = 2.25 → x = k / (2 : ℝ)^2) : 
  ∃ (x : ℝ), x = 1 := by
  sorry

end value_of_x_l1951_195134


namespace one_fourth_of_6_8_is_fraction_l1951_195131

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end one_fourth_of_6_8_is_fraction_l1951_195131


namespace angle_between_hands_at_seven_l1951_195170

-- Define the conditions
def clock_parts := 12 -- The clock is divided into 12 parts
def degrees_per_part := 30 -- Each part is 30 degrees

-- Define the position of the hour and minute hands at 7:00 AM
def hour_position_at_seven := 7 -- Hour hand points to 7
def minute_position_at_seven := 0 -- Minute hand points to 12

-- Calculate the number of parts between the two positions
def parts_between_hands := if minute_position_at_seven = 0 then hour_position_at_seven else 12 - hour_position_at_seven

-- Calculate the angle between the hour hand and the minute hand at 7:00 AM
def angle_at_seven := degrees_per_part * parts_between_hands

-- State the theorem
theorem angle_between_hands_at_seven : angle_at_seven = 150 :=
by
  sorry

end angle_between_hands_at_seven_l1951_195170


namespace intersection_points_of_parabolas_l1951_195110

open Real

theorem intersection_points_of_parabolas (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ y1 y2 : ℝ, y1 = c ∧ y2 = (-2 * b^2 / (9 * a)) + c ∧ 
    ((y1 = a * (0)^2 + b * (0) + c) ∧ (y2 = a * (-b / (3 * a))^2 + b * (-b / (3 * a)) + c))) :=
by
  sorry

end intersection_points_of_parabolas_l1951_195110


namespace smallest_n_boxes_cookies_l1951_195135

theorem smallest_n_boxes_cookies (n : ℕ) (h : (17 * n - 1) % 12 = 0) : n = 5 :=
sorry

end smallest_n_boxes_cookies_l1951_195135


namespace football_team_people_count_l1951_195117

theorem football_team_people_count (original_count : ℕ) (new_members : ℕ) (total_count : ℕ) 
  (h1 : original_count = 36) (h2 : new_members = 14) : total_count = 50 :=
by
  -- This is where the proof would go. We write 'sorry' because it is not required.
  sorry

end football_team_people_count_l1951_195117


namespace find_remainder_l1951_195129

variable (x y remainder : ℕ)
variable (h1 : x = 7 * y + 3)
variable (h2 : 2 * x = 18 * y + remainder)
variable (h3 : 11 * y - x = 1)

theorem find_remainder : remainder = 2 := 
by
  sorry

end find_remainder_l1951_195129


namespace multiple_of_7_l1951_195196

theorem multiple_of_7 :
  ∃ k : ℤ, 77 = 7 * k :=
sorry

end multiple_of_7_l1951_195196


namespace Julia_total_payment_l1951_195180

namespace CarRental

def daily_rate : ℝ := 30
def mileage_rate : ℝ := 0.25
def num_days : ℝ := 3
def num_miles : ℝ := 500

def daily_cost : ℝ := daily_rate * num_days
def mileage_cost : ℝ := mileage_rate * num_miles
def total_cost : ℝ := daily_cost + mileage_cost

theorem Julia_total_payment : total_cost = 215 := by
  sorry

end CarRental

end Julia_total_payment_l1951_195180


namespace combined_prism_volume_is_66_l1951_195158

noncomputable def volume_of_combined_prisms
  (length_rect : ℝ) (width_rect : ℝ) (height_rect : ℝ)
  (base_tri : ℝ) (height_tri : ℝ) (length_tri : ℝ) : ℝ :=
  let volume_rect := length_rect * width_rect * height_rect
  let area_tri := (1 / 2) * base_tri * height_tri
  let volume_tri := area_tri * length_tri
  volume_rect + volume_tri

theorem combined_prism_volume_is_66 :
  volume_of_combined_prisms 6 4 2 3 3 4 = 66 := by
  sorry

end combined_prism_volume_is_66_l1951_195158


namespace print_gift_wrap_price_l1951_195116

theorem print_gift_wrap_price (solid_price : ℝ) (total_rolls : ℕ) (total_money : ℝ)
    (print_rolls : ℕ) (solid_rolls_money : ℝ) (print_money : ℝ) (P : ℝ) :
  solid_price = 4 ∧ total_rolls = 480 ∧ total_money = 2340 ∧ print_rolls = 210 ∧
  solid_rolls_money = 270 * 4 ∧ print_money = 1260 ∧
  total_money = solid_rolls_money + print_money ∧ P = print_money / 210 
  → P = 6 :=
by
  sorry

end print_gift_wrap_price_l1951_195116


namespace intersection_product_is_15_l1951_195174

-- Define the first circle equation as a predicate
def first_circle (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 6 * y + 12 = 0

-- Define the second circle equation as a predicate
def second_circle (x y : ℝ) : Prop :=
  x^2 - 10 * x + y^2 - 6 * y + 34 = 0

-- The Lean statement for the proof problem
theorem intersection_product_is_15 :
  ∃ x y : ℝ, first_circle x y ∧ second_circle x y ∧ (x * y = 15) :=
by
  sorry

end intersection_product_is_15_l1951_195174


namespace edward_final_money_l1951_195143

theorem edward_final_money 
  (spring_earnings : ℕ)
  (summer_earnings : ℕ)
  (supplies_cost : ℕ)
  (h_spring : spring_earnings = 2)
  (h_summer : summer_earnings = 27)
  (h_supplies : supplies_cost = 5)
  : spring_earnings + summer_earnings - supplies_cost = 24 := 
sorry

end edward_final_money_l1951_195143


namespace travel_distance_bus_l1951_195193

theorem travel_distance_bus (D P T B : ℝ) 
    (hD : D = 1800)
    (hP : P = D / 3)
    (hT : T = (2 / 3) * B)
    (h_total : P + T + B = D) :
    B = 720 := 
by
    sorry

end travel_distance_bus_l1951_195193


namespace friends_count_l1951_195136

noncomputable def university_students := 1995

theorem friends_count (students : ℕ)
  (knows_each_other : (ℕ → ℕ → Prop))
  (acquaintances : ℕ → ℕ)
  (h_university_students : students = university_students)
  (h_knows_iff_same_acq : ∀ a b, knows_each_other a b ↔ acquaintances a = acquaintances b)
  (h_not_knows_iff_diff_acq : ∀ a b, ¬ knows_each_other a b ↔ acquaintances a ≠ acquaintances b) :
  ∃ a, acquaintances a ≥ 62 ∧ ¬ ∃ a, acquaintances a ≥ 63 :=
sorry

end friends_count_l1951_195136


namespace students_in_class_l1951_195102

theorem students_in_class (n : ℕ) (h1 : (n : ℝ) * 100 = (n * 100 + 60 - 10)) 
  (h2 : (n : ℝ) * 98 = ((n : ℝ) * 100 - 50)) : n = 25 :=
sorry

end students_in_class_l1951_195102


namespace orange_price_l1951_195113

theorem orange_price (initial_apples : ℕ) (initial_oranges : ℕ) 
                     (apple_price : ℝ) (total_earnings : ℝ) 
                     (remaining_apples : ℕ) (remaining_oranges : ℕ)
                     (h1 : initial_apples = 50) (h2 : initial_oranges = 40)
                     (h3 : apple_price = 0.80) (h4 : total_earnings = 49)
                     (h5 : remaining_apples = 10) (h6 : remaining_oranges = 6) :
  ∃ orange_price : ℝ, orange_price = 0.50 :=
by
  sorry

end orange_price_l1951_195113


namespace ab_max_min_sum_l1951_195107

-- Define the conditions
variables {a b : ℝ}
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 4 * b = 4

-- Problem (1)
theorem ab_max : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → a * b ≤ 1 :=
by sorry

-- Problem (2)
theorem min_sum : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → (1 / a) + (4 / b) ≥ 25 / 4 :=
by sorry

end ab_max_min_sum_l1951_195107


namespace multiply_large_numbers_l1951_195181

theorem multiply_large_numbers :
  72519 * 9999 = 724817481 :=
by
  sorry

end multiply_large_numbers_l1951_195181


namespace pow_mod_cycle_l1951_195137

theorem pow_mod_cycle (n : ℕ) : 3^250 % 13 = 3 := 
by
  sorry

end pow_mod_cycle_l1951_195137


namespace exists_polynomial_degree_n_l1951_195175

theorem exists_polynomial_degree_n (n : ℕ) (hn : 0 < n) : 
  ∃ (ω ψ : Polynomial ℤ), ω.degree = n ∧ (ω^2 = (X^2 - 1) * ψ^2 + 1) := 
sorry

end exists_polynomial_degree_n_l1951_195175


namespace solve_system_l1951_195162

def system_of_equations (x y : ℤ) : Prop :=
  (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧
  (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0)

theorem solve_system : system_of_equations (-3) (-1) :=
by {
  -- Proof details are omitted
  sorry
}

end solve_system_l1951_195162


namespace relation_between_a_b_c_l1951_195154

theorem relation_between_a_b_c :
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  a > c ∧ c > b :=
by {
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  sorry
}

end relation_between_a_b_c_l1951_195154


namespace perfect_square_expression_5_l1951_195101

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def expression_1 : ℕ := 3^3 * 4^4 * 7^7
def expression_2 : ℕ := 3^4 * 4^3 * 7^6
def expression_3 : ℕ := 3^5 * 4^6 * 7^5
def expression_4 : ℕ := 3^6 * 4^5 * 7^4
def expression_5 : ℕ := 3^4 * 4^6 * 7^4

theorem perfect_square_expression_5 : is_perfect_square expression_5 :=
sorry

end perfect_square_expression_5_l1951_195101


namespace all_rationals_in_A_l1951_195194

noncomputable def f (n : ℕ) : ℚ := (n-1)/(n+2)

def A : Set ℚ := { q | ∃ (s : Finset ℕ), q = s.sum f }

theorem all_rationals_in_A : A = Set.univ :=
by
  sorry

end all_rationals_in_A_l1951_195194


namespace average_age_of_omi_kimiko_arlette_l1951_195173

theorem average_age_of_omi_kimiko_arlette (Kimiko Omi Arlette : ℕ) (hK : Kimiko = 28) (hO : Omi = 2 * Kimiko) (hA : Arlette = (3 * Kimiko) / 4) : 
  (Omi + Kimiko + Arlette) / 3 = 35 := 
by
  sorry

end average_age_of_omi_kimiko_arlette_l1951_195173


namespace compute_fraction_l1951_195188

theorem compute_fraction (x y z : ℝ) (h : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by 
  sorry

end compute_fraction_l1951_195188


namespace find_a2_plus_b2_l1951_195122

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 :=
by
  sorry

end find_a2_plus_b2_l1951_195122


namespace problem1_problem2_l1951_195167

-- Define the universe U
def U : Set ℝ := Set.univ

-- Define the sets A and B
def A : Set ℝ := { x | -4 < x ∧ x < 4 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

-- Statement of the first proof problem: Prove A ∩ B is equal to the given set
theorem problem1 : A ∩ B = { x | -4 < x ∧ x ≤ 1 ∨ 4 > x ∧ x ≥ 3 } :=
by
  sorry

-- Statement of the second proof problem: Prove the complement of (A ∪ B) in the universe U is ∅
theorem problem2 : Set.compl (A ∪ B) = ∅ :=
by
  sorry

end problem1_problem2_l1951_195167


namespace cube_sum_is_integer_l1951_195112

theorem cube_sum_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^3 + 1/a^3 = m :=
sorry

end cube_sum_is_integer_l1951_195112


namespace speed_of_A_l1951_195139

theorem speed_of_A (B_speed : ℕ) (crossings : ℕ) (H : B_speed = 3 ∧ crossings = 5 ∧ 5 * (1 / (x + B_speed)) = 1) : x = 2 :=
by
  sorry

end speed_of_A_l1951_195139


namespace alex_final_silver_tokens_l1951_195190

-- Define initial conditions
def initial_red_tokens := 100
def initial_blue_tokens := 50

-- Define exchange rules
def booth1_red_cost := 3
def booth1_silver_gain := 2
def booth1_blue_gain := 1

def booth2_blue_cost := 4
def booth2_silver_gain := 1
def booth2_red_gain := 2

-- Define limits where no further exchanges are possible
def red_token_limit := 2
def blue_token_limit := 3

-- Define the number of times visiting each booth
variable (x y : ℕ)

-- Tokens left after exchanges
def remaining_red_tokens := initial_red_tokens - 3 * x + 2 * y
def remaining_blue_tokens := initial_blue_tokens + x - 4 * y

-- Define proof theorem
theorem alex_final_silver_tokens :
  (remaining_red_tokens x y ≤ red_token_limit) ∧
  (remaining_blue_tokens x y ≤ blue_token_limit) →
  (2 * x + y = 113) :=
by
  sorry

end alex_final_silver_tokens_l1951_195190


namespace geometric_seq_min_3b2_7b3_l1951_195133

theorem geometric_seq_min_3b2_7b3 (b_1 b_2 b_3 : ℝ) (r : ℝ) 
  (h_seq : b_1 = 2) (h_geom : b_2 = b_1 * r) (h_geom2 : b_3 = b_1 * r^2) :
  3 * b_2 + 7 * b_3 ≥ -16 / 7 :=
by
  -- Include the necessary definitions to support the setup
  have h_b1 : b_1 = 2 := h_seq
  have h_b2 : b_2 = 2 * r := by rw [h_geom, h_b1]
  have h_b3 : b_3 = 2 * r^2 := by rw [h_geom2, h_b1]
  sorry

end geometric_seq_min_3b2_7b3_l1951_195133


namespace sin_cos_sixth_power_sum_l1951_195144

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 0.8125 :=
by
  sorry

end sin_cos_sixth_power_sum_l1951_195144


namespace slope_tangent_line_at_x1_l1951_195132

def f (x c : ℝ) : ℝ := (x-2)*(x^2 + c)
def f_prime (x c : ℝ) := (x^2 + c) + (x-2) * 2 * x

theorem slope_tangent_line_at_x1 (c : ℝ) (h : f_prime 2 c = 0) : f_prime 1 c = -5 := by
  sorry

end slope_tangent_line_at_x1_l1951_195132


namespace second_alloy_amount_l1951_195146

theorem second_alloy_amount (x : ℝ) :
  (0.12 * 15 + 0.08 * x = 0.092 * (15 + x)) → x = 35 :=
by
  sorry

end second_alloy_amount_l1951_195146


namespace final_cards_l1951_195123

def initial_cards : ℝ := 47.0
def lost_cards : ℝ := 7.0

theorem final_cards : (initial_cards - lost_cards) = 40.0 :=
by
  sorry

end final_cards_l1951_195123


namespace candy_bar_cost_l1951_195121

theorem candy_bar_cost :
  ∃ C : ℕ, (C + 1 = 3) → (C = 2) :=
by
  use 2
  intros h
  linarith

end candy_bar_cost_l1951_195121


namespace calculate_fraction_l1951_195178

theorem calculate_fraction : (2002 - 1999)^2 / 169 = 9 / 169 :=
by
  sorry

end calculate_fraction_l1951_195178


namespace tangent_line_eq_at_P_tangent_lines_through_P_l1951_195165

-- Define the function and point of interest
def f (x : ℝ) : ℝ := x^3
def P : ℝ × ℝ := (1, 1)

-- State the first part: equation of the tangent line at (1, 1)
theorem tangent_line_eq_at_P : 
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ y = f x ∧ x = 1 → y = 3 * x - 2) :=
sorry

-- State the second part: equations of tangent lines passing through (1, 1)
theorem tangent_lines_through_P :
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∀ (x₀ y₀ : ℝ), y₀ = x₀^3 → 
  (x₀ ≠ 1 → ∃ k : ℝ,  k = 3 * (x₀)^2 → 
  (∀ x y : ℝ, y = k * (x - 1) + 1 ∧ y = f x₀ → y = y₀))) → 
  (∃ m b m' b' : ℝ, 
    (¬ ∀ x : ℝ, ∀ y : ℝ, (y = m *x + b ∧ y = 3 * x - 2) → y = m' * x + b') ∧ 
    ((m = 3 ∧ b = -2) ∧ (m' = 3/4 ∧ b' = 1/4))) :=
sorry

end tangent_line_eq_at_P_tangent_lines_through_P_l1951_195165


namespace rhombus_fourth_vertex_l1951_195105

theorem rhombus_fourth_vertex (a b : ℝ) :
  ∃ x y : ℝ, (x, y) = (a - b, a + b) ∧ dist (a, b) (x, y) = dist (-b, a) (x, y) ∧ dist (-b, a) (x, y) = dist (0, 0) (x, y) :=
by
  use (a - b)
  use (a + b)
  sorry

end rhombus_fourth_vertex_l1951_195105


namespace closest_cube_root_l1951_195184

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l1951_195184


namespace factorize_polynomial_l1951_195166

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem factorize_polynomial :
  (zeta^3 = 1) ∧ (zeta^2 + zeta + 1 = 0) → (x : ℂ) → (x^15 + x^10 + x) = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1)
:= sorry

end factorize_polynomial_l1951_195166


namespace solve_system_of_equations_l1951_195106

theorem solve_system_of_equations (x y : ℝ) (hx: x > 0) (hy: y > 0) :
  x * y = 500 ∧ x ^ (Real.log y / Real.log 10) = 25 → (x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100) := by
  sorry

end solve_system_of_equations_l1951_195106


namespace Dan_must_exceed_speed_l1951_195104

theorem Dan_must_exceed_speed (distance : ℝ) (Cara_speed : ℝ) (delay : ℝ) (time_Cara : ℝ) (Dan_time : ℝ) : 
  distance = 120 ∧ Cara_speed = 30 ∧ delay = 1 ∧ time_Cara = distance / Cara_speed ∧ time_Cara = 4 ∧ Dan_time = time_Cara - delay ∧ Dan_time < 4 → 
  (distance / Dan_time) > 40 :=
by
  sorry

end Dan_must_exceed_speed_l1951_195104


namespace parabola_equation_l1951_195168

theorem parabola_equation (a b c : ℝ)
  (h_p : (a + b + c = 1))
  (h_q : (4 * a + 2 * b + c = -1))
  (h_tangent : (4 * a + b = 1)) :
  y = 3 * x^2 - 11 * x + 9 :=
by {
  sorry
}

end parabola_equation_l1951_195168


namespace inequality_solution_l1951_195100

theorem inequality_solution (x: ℝ) (h1: x ≠ -1) (h2: x ≠ 0) :
  (x-2)/(x+1) + (x-3)/(3*x) ≥ 2 ↔ x ∈ Set.Iic (-3) ∪ Set.Icc (-1) (-1/2) :=
by
  sorry

end inequality_solution_l1951_195100


namespace part1_part2_l1951_195169

noncomputable def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 :=
by
  sorry

theorem part2 (m : ℝ) (hx : ∀ x : ℝ, m^2 + 3 * m + 2 * f x ≥ 0) : m ≤ -2 ∨ m ≥ -1 :=
by
  sorry

end part1_part2_l1951_195169


namespace income_expenditure_ratio_l1951_195149

theorem income_expenditure_ratio
  (I : ℕ) (E : ℕ) (S : ℕ)
  (h1 : I = 18000)
  (h2 : S = 3600)
  (h3 : S = I - E) : I / E = 5 / 4 :=
by
  -- The actual proof is skipped.
  sorry

end income_expenditure_ratio_l1951_195149


namespace b_should_pay_360_l1951_195145

theorem b_should_pay_360 :
  let total_cost : ℝ := 870
  let a_horses  : ℝ := 12
  let a_months  : ℝ := 8
  let b_horses  : ℝ := 16
  let b_months  : ℝ := 9
  let c_horses  : ℝ := 18
  let c_months  : ℝ := 6
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  let cost_per_horse_month := total_cost / total_horse_months
  let b_cost := b_horse_months * cost_per_horse_month
  b_cost = 360 :=
by sorry

end b_should_pay_360_l1951_195145


namespace trapezoid_shorter_base_length_l1951_195155

theorem trapezoid_shorter_base_length 
  (a b : ℕ) 
  (mid_segment_length longer_base : ℕ) 
  (h1 : mid_segment_length = 5) 
  (h2 : longer_base = 103) 
  (trapezoid_property : mid_segment_length = (longer_base - a) / 2) : 
  a = 93 := 
sorry

end trapezoid_shorter_base_length_l1951_195155


namespace smallest_s_for_F_l1951_195157

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F :
  ∃ s : ℕ, F s s 2 2 = 65536 ∧ ∀ t : ℕ, F t t 2 2 = 65536 → s ≤ t :=
sorry

end smallest_s_for_F_l1951_195157


namespace area_of_BDOE_l1951_195103

namespace Geometry

noncomputable def areaQuadrilateralBDOE (AE CD AB BC AC : ℝ) : ℝ :=
  if AE = 2 ∧ CD = 11 ∧ AB = 8 ∧ BC = 8 ∧ AC = 6 then
    189 * Real.sqrt 55 / 88
  else
    0

theorem area_of_BDOE :
  areaQuadrilateralBDOE 2 11 8 8 6 = 189 * Real.sqrt 55 / 88 :=
by 
  sorry

end Geometry

end area_of_BDOE_l1951_195103


namespace addition_of_decimals_l1951_195140

theorem addition_of_decimals (a b : ℚ) (h1 : a = 7.56) (h2 : b = 4.29) : a + b = 11.85 :=
by
  -- The proof will be provided here
  sorry

end addition_of_decimals_l1951_195140


namespace lesser_fraction_l1951_195147

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l1951_195147


namespace find_value_of_m_l1951_195150

-- Define the quadratic function and the values in the given table
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
variables (a b c m : ℝ)
variables (h1 : quadratic_function a b c (-1) = m)
variables (h2 : quadratic_function a b c 0 = 2)
variables (h3 : quadratic_function a b c 1 = 1)
variables (h4 : quadratic_function a b c 2 = 2)
variables (h5 : quadratic_function a b c 3 = 5)
variables (h6 : quadratic_function a b c 4 = 10)

-- Theorem stating that the value of m is 5
theorem find_value_of_m : m = 5 :=
by
  sorry

end find_value_of_m_l1951_195150


namespace problem_statement_l1951_195156

variable {S R p a b c : ℝ}
variable (τ τ_a τ_b τ_c : ℝ)

theorem problem_statement
  (h1: S = τ * p)
  (h2: S = τ_a * (p - a))
  (h3: S = τ_b * (p - b))
  (h4: S = τ_c * (p - c))
  (h5: τ = S / p)
  (h6: τ_a = S / (p - a))
  (h7: τ_b = S / (p - b))
  (h8: τ_c = S / (p - c))
  (h9: abc / S = 4 * R) :
  1 / τ^3 - 1 / τ_a^3 - 1 / τ_b^3 - 1 / τ_c^3 = 12 * R / S^2 :=
  sorry

end problem_statement_l1951_195156


namespace garden_area_increase_l1951_195109

theorem garden_area_increase :
  let length := 80
  let width := 20
  let additional_fence := 60
  let original_area := length * width
  let original_perimeter := 2 * (length + width)
  let total_fence := original_perimeter + additional_fence
  let side_of_square := total_fence / 4
  let square_area := side_of_square * side_of_square
  square_area - original_area = 2625 :=
by
  sorry

end garden_area_increase_l1951_195109


namespace subset_bound_l1951_195189

open Finset

variables {α : Type*}

theorem subset_bound (n : ℕ) (S : Finset (Finset (Fin (4 * n)))) (hS : ∀ {s t : Finset (Fin (4 * n))}, s ∈ S → t ∈ S → s ≠ t → (s ∩ t).card ≤ n) (h_card : ∀ s ∈ S, s.card = 2 * n) :
  S.card ≤ 6 ^ ((n + 1) / 2) :=
sorry

end subset_bound_l1951_195189


namespace emily_did_not_sell_bars_l1951_195172

-- Definitions based on conditions
def cost_per_bar : ℕ := 4
def total_bars : ℕ := 8
def total_earnings : ℕ := 20

-- The statement to be proved
theorem emily_did_not_sell_bars :
  (total_bars - (total_earnings / cost_per_bar)) = 3 :=
by
  sorry

end emily_did_not_sell_bars_l1951_195172
