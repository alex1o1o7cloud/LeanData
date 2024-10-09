import Mathlib

namespace average_income_Q_and_R_l1931_193194

variable (P Q R: ℝ)

theorem average_income_Q_and_R:
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 :=
by
  sorry

end average_income_Q_and_R_l1931_193194


namespace find_d_l1931_193154

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) : d = 2 :=
by
  sorry

end find_d_l1931_193154


namespace original_selling_price_is_800_l1931_193122

-- Let CP denote the cost price
variable (CP : ℝ)

-- Condition 1: Selling price with a profit of 25%
def selling_price_with_profit (CP : ℝ) : ℝ := 1.25 * CP

-- Condition 2: Selling price with a loss of 35%
def selling_price_with_loss (CP : ℝ) : ℝ := 0.65 * CP

-- Given selling price with loss is Rs. 416
axiom loss_price_is_416 : selling_price_with_loss CP = 416

-- We need to prove the original selling price (with profit) is Rs. 800
theorem original_selling_price_is_800 : selling_price_with_profit CP = 800 :=
by sorry

end original_selling_price_is_800_l1931_193122


namespace Laura_won_5_games_l1931_193147

-- Define the number of wins and losses for each player
def Peter_wins : ℕ := 5
def Peter_losses : ℕ := 3
def Peter_games : ℕ := Peter_wins + Peter_losses

def Emma_wins : ℕ := 4
def Emma_losses : ℕ := 4
def Emma_games : ℕ := Emma_wins + Emma_losses

def Kyler_wins : ℕ := 2
def Kyler_losses : ℕ := 6
def Kyler_games : ℕ := Kyler_wins + Kyler_losses

-- Define the total number of games played in the tournament
def total_games_played : ℕ := (Peter_games + Emma_games + Kyler_games + 8) / 2

-- Define total wins and losses
def total_wins_losses : ℕ := total_games_played

-- Prove the number of games Laura won
def Laura_wins : ℕ := total_wins_losses - (Peter_wins + Emma_wins + Kyler_wins)

theorem Laura_won_5_games : Laura_wins = 5 := by
  -- The proof will be completed here
  sorry

end Laura_won_5_games_l1931_193147


namespace johns_share_l1931_193181

theorem johns_share (total_amount : ℕ) (r1 r2 r3 : ℕ) (h : total_amount = 6000) (hr1 : r1 = 2) (hr2 : r2 = 4) (hr3 : r3 = 6) :
  let total_ratio := r1 + r2 + r3
  let johns_ratio := r1
  let johns_share := (johns_ratio * total_amount) / total_ratio
  johns_share = 1000 :=
by
  sorry

end johns_share_l1931_193181


namespace factorization_roots_l1931_193179

theorem factorization_roots (x : ℂ) : 
  (x^3 - 2*x^2 - x + 2) * (x - 3) * (x + 1) = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  -- Note: Proof to be completed
  sorry

end factorization_roots_l1931_193179


namespace terminating_decimal_expansion_l1931_193189

theorem terminating_decimal_expansion : (15 / 625 : ℝ) = 0.024 :=
by
  -- Lean requires a justification for non-trivial facts
  -- Provide math reasoning here if necessary
  sorry

end terminating_decimal_expansion_l1931_193189


namespace exponent_sum_l1931_193102

theorem exponent_sum : (-2:ℝ) ^ 4 + (-2:ℝ) ^ (3 / 2) + (-2:ℝ) ^ 1 + 2 ^ 1 + 2 ^ (3 / 2) + 2 ^ 4 = 32 := by
  sorry

end exponent_sum_l1931_193102


namespace sum_of_nine_consecutive_parity_l1931_193161

theorem sum_of_nine_consecutive_parity (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) % 2 = n % 2 := 
  sorry

end sum_of_nine_consecutive_parity_l1931_193161


namespace inequality_condition_l1931_193125

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the main theorem to be proven
theorem inequality_condition (a b : ℝ) (h_a : a > 11 / 4) (h_b : b > 3 / 2) :
  (∀ x : ℝ, |x + 1| < b → |f x + 3| < a) :=
by
  -- We state the required proof without providing the steps
  sorry

end inequality_condition_l1931_193125


namespace articles_selling_price_eq_cost_price_of_50_articles_l1931_193118

theorem articles_selling_price_eq_cost_price_of_50_articles (C S : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * S) (h2 : S = 2 * C) : N = 25 := by
  sorry

end articles_selling_price_eq_cost_price_of_50_articles_l1931_193118


namespace beads_per_bracelet_is_10_l1931_193107

-- Definitions of given conditions
def num_necklaces_Monday : ℕ := 10
def num_necklaces_Tuesday : ℕ := 2
def num_necklaces : ℕ := num_necklaces_Monday + num_necklaces_Tuesday

def beads_per_necklace : ℕ := 20
def beads_necklaces : ℕ := num_necklaces * beads_per_necklace

def num_earrings : ℕ := 7
def beads_per_earring : ℕ := 5
def beads_earrings : ℕ := num_earrings * beads_per_earring

def total_beads_used : ℕ := 325
def beads_used_for_necklaces_and_earrings : ℕ := beads_necklaces + beads_earrings
def beads_remaining_for_bracelets : ℕ := total_beads_used - beads_used_for_necklaces_and_earrings

def num_bracelets : ℕ := 5
def beads_per_bracelet : ℕ := beads_remaining_for_bracelets / num_bracelets

-- Theorem statement to prove
theorem beads_per_bracelet_is_10 : beads_per_bracelet = 10 := by
  sorry

end beads_per_bracelet_is_10_l1931_193107


namespace choose_lines_intersect_l1931_193195

-- We need to define the proof problem
theorem choose_lines_intersect : 
  ∃ (lines : ℕ → ℝ × ℝ → ℝ), 
    (∀ i j, i < 100 ∧ j < 100 ∧ i ≠ j → (lines i = lines j) → ∃ (p : ℕ), p = 2022) :=
sorry

end choose_lines_intersect_l1931_193195


namespace solve_for_x_l1931_193149

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l1931_193149


namespace min_value_fraction_sum_l1931_193139

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (4 / (x + 2) + 1 / (y + 1)) ≥ 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l1931_193139


namespace solve_for_d_l1931_193172

variable (n c b d : ℚ)  -- Alternatively, specify the types if they are required to be specific
variable (H : n = d * c * b / (c - d))

theorem solve_for_d :
  d = n * c / (c * b + n) :=
by
  sorry

end solve_for_d_l1931_193172


namespace complex_quadrant_l1931_193129

theorem complex_quadrant (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : 1 + a * i = (b + i) * (1 + i)) : 
  (a - b * i).re > 0 ∧ (a - b * i).im < 0 :=
by
  have h1 : 1 + a * i = (b - 1) + (b + 1) * i := by sorry
  have h2 : a = b + 1 := by sorry
  have h3 : b - 1 = 1 := by sorry
  have h4 : b = 2 := by sorry
  have h5 : a = 3 := by sorry
  have h6 : (a - b * i).re = 3 := by sorry
  have h7 : (a - b * i).im = -2 := by sorry
  exact ⟨by linarith, by linarith⟩

end complex_quadrant_l1931_193129


namespace detectives_sons_ages_l1931_193104

theorem detectives_sons_ages (x y : ℕ) (h1 : x < 5) (h2 : y < 5) (h3 : x * y = 4) (h4 : (∃ x₁ y₁ : ℕ, (x₁ * y₁ = 4 ∧ x₁ < 5 ∧ y₁ < 5) ∧ x₁ ≠ x ∨ y₁ ≠ y)) :
  (x = 1 ∨ x = 4) ∧ (y = 1 ∨ y = 4) :=
by
  sorry

end detectives_sons_ages_l1931_193104


namespace mary_peter_lucy_chestnuts_l1931_193115

noncomputable def mary_picked : ℕ := 12
noncomputable def peter_picked : ℕ := mary_picked / 2
noncomputable def lucy_picked : ℕ := peter_picked + 2
noncomputable def total_picked : ℕ := mary_picked + peter_picked + lucy_picked

theorem mary_peter_lucy_chestnuts : total_picked = 26 := by
  sorry

end mary_peter_lucy_chestnuts_l1931_193115


namespace eric_containers_l1931_193157

theorem eric_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) 
  (h1 : initial_pencils = 150) (h2 : additional_pencils = 30) (h3 : pencils_per_container = 36) :
  (initial_pencils + additional_pencils) / pencils_per_container = 5 := 
by {
  sorry
}

end eric_containers_l1931_193157


namespace overlapping_area_of_rectangular_strips_l1931_193101

theorem overlapping_area_of_rectangular_strips (theta : ℝ) (h_theta : theta ≠ 0) :
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  area = 2 / Real.sin theta :=
by
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  sorry

end overlapping_area_of_rectangular_strips_l1931_193101


namespace initial_hours_per_day_l1931_193119

-- Definitions capturing the conditions
def num_men_initial : ℕ := 100
def num_men_total : ℕ := 160
def portion_completed : ℚ := 1 / 3
def num_days_total : ℕ := 50
def num_days_half : ℕ := 25
def work_performed_portion : ℚ := 2 / 3
def hours_per_day_additional : ℕ := 10

-- Lean statement to prove the initial number of hours per day worked by the initial employees
theorem initial_hours_per_day (H : ℚ) :
  (num_men_initial * H * num_days_total = work_performed_portion) ∧
  (num_men_total * hours_per_day_additional * num_days_half = portion_completed) →
  H = 1.6 := 
sorry

end initial_hours_per_day_l1931_193119


namespace relationship_between_x_t_G_D_and_x_l1931_193188

-- Definitions
variables {G D : ℝ → ℝ}
variables {t : ℝ}
noncomputable def number_of_boys (x : ℝ) : ℝ := 9000 / x
noncomputable def total_population (x : ℝ) (x_t : ℝ) : Prop := x_t = 15000 / x

-- The proof problem
theorem relationship_between_x_t_G_D_and_x
  (G D : ℝ → ℝ)
  (x : ℝ) (t : ℝ) (x_t : ℝ)
  (h1 : 90 = x / 100 * number_of_boys x)
  (h2 : 0.60 * x_t = number_of_boys x)
  (h3 : 0.40 * x_t > 0)
  (h4 : true) :       -- Placeholder for some condition not used directly
  total_population x x_t :=
by
  -- Proof would go here
  sorry

end relationship_between_x_t_G_D_and_x_l1931_193188


namespace min_value_a_l1931_193111

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 6 > 0 → x > a) ∧
  ¬ (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ↔ a = 3 :=
sorry

end min_value_a_l1931_193111


namespace red_team_score_l1931_193137

theorem red_team_score (C R : ℕ) (h1 : C = 95) (h2 : C - R = 19) : R = 76 :=
by
  sorry

end red_team_score_l1931_193137


namespace books_sold_on_monday_l1931_193108

def InitialStock : ℕ := 800
def BooksNotSold : ℕ := 600
def BooksSoldTuesday : ℕ := 10
def BooksSoldWednesday : ℕ := 20
def BooksSoldThursday : ℕ := 44
def BooksSoldFriday : ℕ := 66

def TotalBooksSold : ℕ := InitialStock - BooksNotSold
def BooksSoldAfterMonday : ℕ := BooksSoldTuesday + BooksSoldWednesday + BooksSoldThursday + BooksSoldFriday

theorem books_sold_on_monday : 
  TotalBooksSold - BooksSoldAfterMonday = 60 := by
  sorry

end books_sold_on_monday_l1931_193108


namespace valerie_needs_21_stamps_l1931_193120

def thank_you_cards : ℕ := 3
def bills : ℕ := 2
def mail_in_rebates : ℕ := bills + 3
def job_applications : ℕ := 2 * mail_in_rebates
def water_bill_stamps : ℕ := 1
def electric_bill_stamps : ℕ := 2

def stamps_for_thank_you_cards : ℕ := thank_you_cards * 1
def stamps_for_bills : ℕ := 1 * water_bill_stamps + 1 * electric_bill_stamps
def stamps_for_rebates : ℕ := mail_in_rebates * 1
def stamps_for_job_applications : ℕ := job_applications * 1

def total_stamps : ℕ :=
  stamps_for_thank_you_cards +
  stamps_for_bills +
  stamps_for_rebates +
  stamps_for_job_applications

theorem valerie_needs_21_stamps : total_stamps = 21 := by
  sorry

end valerie_needs_21_stamps_l1931_193120


namespace range_of_a_l1931_193138

noncomputable def satisfiesInequality (a : ℝ) (x : ℝ) : Prop :=
  x > 1 → a * Real.log x > 1 - 1/x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 1 → satisfiesInequality a x) ↔ a ∈ Set.Ici 1 := 
sorry

end range_of_a_l1931_193138


namespace innovation_contribution_l1931_193128

variable (material : String)
variable (contribution : String → Prop)
variable (A B C D : Prop)

-- Conditions
axiom condA : contribution material → A
axiom condB : contribution material → ¬B
axiom condC : contribution material → ¬C
axiom condD : contribution material → ¬D

-- The problem statement
theorem innovation_contribution :
  contribution material → A :=
by
  -- dummy proof as placeholder
  sorry

end innovation_contribution_l1931_193128


namespace find_n_l1931_193109

theorem find_n : ∀ (n x : ℝ), (3639 + n - x = 3054) → (x = 596.95) → (n = 11.95) :=
by
  intros n x h1 h2
  sorry

end find_n_l1931_193109


namespace find_b_for_integer_a_l1931_193112

theorem find_b_for_integer_a (a : ℤ) (b : ℝ) (h1 : 0 ≤ b) (h2 : b < 1) (h3 : (a:ℝ)^2 = 2 * b * (a + b)) :
  b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 :=
sorry

end find_b_for_integer_a_l1931_193112


namespace total_people_ride_l1931_193127

theorem total_people_ride (people_per_carriage : ℕ) (num_carriages : ℕ) (h1 : people_per_carriage = 12) (h2 : num_carriages = 15) : 
    people_per_carriage * num_carriages = 180 := by
  sorry

end total_people_ride_l1931_193127


namespace sector_area_l1931_193182

theorem sector_area (C : ℝ) (θ : ℝ) (r : ℝ) (S : ℝ)
  (hC : C = (8 * Real.pi / 9) + 4)
  (hθ : θ = (80 * Real.pi / 180))
  (hne : θ * r / 2 + r = C) :
  S = (1 / 2) * θ * r^2 → S = 8 * Real.pi / 9 :=
by
  sorry

end sector_area_l1931_193182


namespace cost_price_of_computer_table_l1931_193158

theorem cost_price_of_computer_table
  (C : ℝ) 
  (S : ℝ := 1.20 * C)
  (S_eq : S = 8600) : 
  C = 7166.67 :=
by
  sorry

end cost_price_of_computer_table_l1931_193158


namespace domain_of_f_l1931_193123

noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 - 1)) / (Real.sqrt (x^2 - x - 2))

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0 ∧ x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end domain_of_f_l1931_193123


namespace first_number_in_list_is_55_l1931_193126

theorem first_number_in_list_is_55 : 
  ∀ (x : ℕ), (55 + 57 + 58 + 59 + 62 + 62 + 63 + 65 + x) / 9 = 60 → x = 65 → 55 = 55 :=
by
  intros x avg_cond x_is_65
  rfl

end first_number_in_list_is_55_l1931_193126


namespace minimum_value_hyperbola_l1931_193141

noncomputable def min_value (a b : ℝ) (h : a > 0) (k : b > 0)
  (eccentricity_eq_two : (2:ℝ) = Real.sqrt (1 + (b/a)^2)) : ℝ :=
  (b^2 + 1) / (3 * a)

theorem minimum_value_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2:ℝ) = Real.sqrt (1 + (b/a)^2) ∧
  min_value a b (by sorry) (by sorry) (by sorry) = (2 * Real.sqrt 3) / 3 :=
sorry

end minimum_value_hyperbola_l1931_193141


namespace two_bedroom_units_l1931_193185

theorem two_bedroom_units {x y : ℕ} 
  (h1 : x + y = 12) 
  (h2 : 360 * x + 450 * y = 4950) : 
  y = 7 := 
by
  sorry

end two_bedroom_units_l1931_193185


namespace graph_of_equation_is_two_intersecting_lines_l1931_193156

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ (x y : ℝ), (x - y)^2 = 3 * x^2 - y^2 ↔ 
  (x = (1 - Real.sqrt 5) / 2 * y) ∨ (x = (1 + Real.sqrt 5) / 2 * y) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l1931_193156


namespace Tom_age_ratio_l1931_193143

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end Tom_age_ratio_l1931_193143


namespace remainder_of_1998_to_10_mod_10k_l1931_193116

theorem remainder_of_1998_to_10_mod_10k : 
  let x := 1998
  let y := 10^4
  x^10 % y = 1024 := 
by
  let x := 1998
  let y := 10^4
  sorry

end remainder_of_1998_to_10_mod_10k_l1931_193116


namespace total_players_count_l1931_193110

theorem total_players_count (M W : ℕ) (h1 : W = M + 4) (h2 : (M : ℚ) / W = 5 / 9) : M + W = 14 :=
sorry

end total_players_count_l1931_193110


namespace three_students_received_A_l1931_193184

variables (A B C E D : Prop)
variables (h1 : A → B) (h2 : B → C) (h3 : C → E) (h4 : E → D)

theorem three_students_received_A :
  (A ∨ ¬A) ∧ (B ∨ ¬B) ∧ (C ∨ ¬C) ∧ (E ∨ ¬E) ∧ (D ∨ ¬D) ∧ (¬A ∧ ¬B) → (C ∧ E ∧ D) ∧ ¬A ∧ ¬B :=
by sorry

end three_students_received_A_l1931_193184


namespace marble_problem_l1931_193186

-- Defining the problem in Lean statement
theorem marble_problem 
  (m : ℕ) (n k : ℕ) (hx : m = 220) (hy : n = 20) : 
  (∀ x : ℕ, (k = n + x) → (m / n = 11) → (m / k = 10)) → (x = 2) :=
by {
  sorry
}

end marble_problem_l1931_193186


namespace crayons_per_child_l1931_193150

theorem crayons_per_child (total_crayons children : ℕ) (h_total : total_crayons = 56) (h_children : children = 7) : (total_crayons / children) = 8 := by
  -- proof will go here
  sorry

end crayons_per_child_l1931_193150


namespace quarters_count_l1931_193170

theorem quarters_count (total_money : ℝ) (value_of_quarter : ℝ) (h1 : total_money = 3) (h2 : value_of_quarter = 0.25) : total_money / value_of_quarter = 12 :=
by sorry

end quarters_count_l1931_193170


namespace sara_bought_cards_l1931_193163

-- Definition of the given conditions
def initial_cards : ℕ := 39
def torn_cards : ℕ := 9
def remaining_cards_after_sale : ℕ := 15

-- Derived definition: Number of good cards before selling to Sara
def good_cards_before_selling : ℕ := initial_cards - torn_cards

-- The statement we need to prove
theorem sara_bought_cards : good_cards_before_selling - remaining_cards_after_sale = 15 :=
by
  sorry

end sara_bought_cards_l1931_193163


namespace mod_pow_sum_7_l1931_193174

theorem mod_pow_sum_7 :
  (45 ^ 1234 + 27 ^ 1234) % 7 = 5 := by
  sorry

end mod_pow_sum_7_l1931_193174


namespace problem1_problem2_l1931_193124

variables {p x1 x2 y1 y2 : ℝ} (h₁ : p > 0) (h₂ : x1 * x2 ≠ 0) (h₃ : y1^2 = 2 * p * x1) (h₄ : y2^2 = 2 * p * x2)

theorem problem1 (h₅ : x1 * x2 + y1 * y2 = 0) :
    ∀ (x y : ℝ), (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 → 
        x^2 + y^2 - (x1 + x2) * x - (y1 + y2) * y = 0 := sorry

theorem problem2 (h₀ : ∀ x y, x = (x1 + x2) / 2 → y = (y1 + y2) / 2 → 
    |((x1 + x2) / 2) - 2 * ((y1 + y2) / 2)| / (Real.sqrt 5) = 2 * (Real.sqrt 5) / 5) :
    p = 2 := sorry

end problem1_problem2_l1931_193124


namespace min_Sn_l1931_193103

variable {a : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (a₄ : ℤ) (d : ℤ) : Prop :=
  a 4 = a₄ ∧ ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

def Sn (a : ℕ → ℤ) (n : ℕ) :=
  n / 2 * (2 * a 1 + (n - 1) * 3)

theorem min_Sn (a : ℕ → ℤ) (h1 : arithmetic_sequence a (-15) 3) :
  ∃ n : ℕ, (Sn a n = -108) :=
sorry

end min_Sn_l1931_193103


namespace LCM_of_numbers_l1931_193121

theorem LCM_of_numbers (a b : ℕ) (h1 : a = 20) (h2 : a / b = 5 / 4): Nat.lcm a b = 80 :=
by
  sorry

end LCM_of_numbers_l1931_193121


namespace mike_coins_value_l1931_193192

theorem mike_coins_value (d q : ℕ)
  (h1 : d + q = 17)
  (h2 : q + 3 = 2 * d) :
  10 * d + 25 * q = 345 :=
by
  sorry

end mike_coins_value_l1931_193192


namespace A_inter_B_l1931_193132

open Set Real

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { y | ∃ x, y = exp x }

theorem A_inter_B :
  A ∩ B = { z | 0 < z ∧ z < 3 } :=
by
  sorry

end A_inter_B_l1931_193132


namespace angle_parallel_result_l1931_193135

theorem angle_parallel_result (A B : ℝ) (h1 : A = 60) (h2 : (A = B ∨ A + B = 180)) : (B = 60 ∨ B = 120) :=
by
  sorry

end angle_parallel_result_l1931_193135


namespace length_AF_l1931_193140

def CE : ℝ := 40
def ED : ℝ := 50
def AE : ℝ := 120
def area_ABCD : ℝ := 7200

theorem length_AF (AF : ℝ) :
  CE = 40 → ED = 50 → AE = 120 → area_ABCD = 7200 →
  AF = 128 :=
by
  intros hCe hEd hAe hArea
  sorry

end length_AF_l1931_193140


namespace ratio_of_expenditures_l1931_193142

-- Let us define the conditions and rewrite the proof problem statement.
theorem ratio_of_expenditures
  (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℝ)
  (H1 : income_P1 / income_P2 = 5 / 4)
  (H2 : income_P1 = 5000)
  (H3 : income_P1 - expenditure_P1 = 2000)
  (H4 : income_P2 - expenditure_P2 = 2000) :
  expenditure_P1 / expenditure_P2 = 3 / 2 :=
sorry

end ratio_of_expenditures_l1931_193142


namespace find_number_l1931_193165

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l1931_193165


namespace number_of_red_balls_l1931_193169

theorem number_of_red_balls (m : ℕ) (h1 : ∃ m : ℕ, (3 / (m + 3) : ℚ) = 1 / 4) : m = 9 :=
by
  obtain ⟨m, h1⟩ := h1
  sorry

end number_of_red_balls_l1931_193169


namespace divisor_of_44404_l1931_193153

theorem divisor_of_44404: ∃ k : ℕ, 2 * 11101 = k ∧ k ∣ (44402 + 2) :=
by
  sorry

end divisor_of_44404_l1931_193153


namespace inverse_proposition_l1931_193175

theorem inverse_proposition (a b : ℝ) (h : ab = 0) : (a = 0 → ab = 0) :=
by
  sorry

end inverse_proposition_l1931_193175


namespace total_hours_worked_l1931_193105

-- Define the number of hours worked on Saturday
def hours_saturday : ℕ := 6

-- Define the number of hours worked on Sunday
def hours_sunday : ℕ := 4

-- Define the total number of hours worked on both days
def total_hours : ℕ := hours_saturday + hours_sunday

-- The theorem to prove the total number of hours worked on Saturday and Sunday
theorem total_hours_worked : total_hours = 10 := by
  sorry

end total_hours_worked_l1931_193105


namespace find_m_l1931_193152

section
variables {R : Type*} [CommRing R]

def f (x : R) : R := 4 * x^2 - 3 * x + 5
def g (x : R) (m : R) : R := x^2 - m * x - 8

theorem find_m (m : ℚ) : 
  f (5 : ℚ) - g (5 : ℚ) m = 20 → m = -53 / 5 :=
by {
  sorry
}

end

end find_m_l1931_193152


namespace octahedron_tetrahedron_volume_ratio_l1931_193199

theorem octahedron_tetrahedron_volume_ratio (s : ℝ) :
  let V_T := (s^3 * Real.sqrt 2) / 12
  let a := s / 2
  let V_O := (a^3 * Real.sqrt 2) / 3
  V_O / V_T = 1 / 2 :=
by
  sorry

end octahedron_tetrahedron_volume_ratio_l1931_193199


namespace parabola_focus_l1931_193114

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l1931_193114


namespace integer_conditions_satisfy_eq_l1931_193177

theorem integer_conditions_satisfy_eq (
  a b c : ℤ 
) : (a > b ∧ b = c → (a * (a - b) + b * (b - c) + c * (c - a) = 2)) ∧
    (¬(a = b - 1 ∧ b = c - 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c + 1 ∧ b = a + 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c ∧ b - 2 = c) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a + b + c = 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) :=
by
sorry

end integer_conditions_satisfy_eq_l1931_193177


namespace recurring_decimal_sum_l1931_193131

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end recurring_decimal_sum_l1931_193131


namespace find_c_l1931_193159

theorem find_c (a c : ℝ) (h1 : x^2 + 80 * x + c = (x + a)^2) (h2 : 2 * a = 80) : c = 1600 :=
sorry

end find_c_l1931_193159


namespace eccentricity_is_sqrt2_div2_l1931_193136

noncomputable def eccentricity_square_ellipse (a b c : ℝ) : ℝ :=
  c / (Real.sqrt (b ^ 2 + c ^ 2))

theorem eccentricity_is_sqrt2_div2 (a b c : ℝ) (h : b = c) : 
  eccentricity_square_ellipse a b c = Real.sqrt 2 / 2 :=
by
  -- The proof will show that the eccentricity calculation is correct given the conditions.
  sorry

end eccentricity_is_sqrt2_div2_l1931_193136


namespace unique_solution_c_min_l1931_193191

theorem unique_solution_c_min (x y : ℝ) (c : ℝ)
  (h1 : 2 * (x+7)^2 + (y-4)^2 = c)
  (h2 : (x+4)^2 + 2 * (y-7)^2 = c) :
  c = 6 :=
sorry

end unique_solution_c_min_l1931_193191


namespace increasing_range_of_a_l1931_193145

noncomputable def f (x : ℝ) (a : ℝ) := 
  if x ≤ 1 then -x^2 + 4*a*x 
  else (2*a + 3)*x - 4*a + 5

theorem increasing_range_of_a :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
sorry

end increasing_range_of_a_l1931_193145


namespace work_problem_l1931_193113

theorem work_problem (W : ℝ) (A B C : ℝ)
  (h1 : B + C = W / 24)
  (h2 : C + A = W / 12)
  (h3 : C = W / 32) : A + B = W / 16 := 
by
  sorry

end work_problem_l1931_193113


namespace scientific_notation_eq_l1931_193134

-- Define the number 82,600,000
def num : ℝ := 82600000

-- Define the scientific notation representation
def sci_not : ℝ := 8.26 * 10^7

-- The theorem to prove that the number is equal to its scientific notation
theorem scientific_notation_eq : num = sci_not :=
by 
  sorry

end scientific_notation_eq_l1931_193134


namespace simplify_expression_l1931_193166

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l1931_193166


namespace rosa_initial_flowers_l1931_193196

-- Definitions derived from conditions
def initial_flowers (total_flowers : ℕ) (given_flowers : ℕ) : ℕ :=
  total_flowers - given_flowers

-- The theorem stating the proof problem
theorem rosa_initial_flowers : initial_flowers 90 23 = 67 :=
by
  -- The proof goes here
  sorry

end rosa_initial_flowers_l1931_193196


namespace area_difference_is_correct_l1931_193155

noncomputable def circumference_1 : ℝ := 264
noncomputable def circumference_2 : ℝ := 352

noncomputable def radius_1 : ℝ := circumference_1 / (2 * Real.pi)
noncomputable def radius_2 : ℝ := circumference_2 / (2 * Real.pi)

noncomputable def area_1 : ℝ := Real.pi * radius_1^2
noncomputable def area_2 : ℝ := Real.pi * radius_2^2

noncomputable def area_difference : ℝ := area_2 - area_1

theorem area_difference_is_correct :
  abs (area_difference - 4305.28) < 1e-2 :=
by
  sorry

end area_difference_is_correct_l1931_193155


namespace max_common_initial_segment_l1931_193173

theorem max_common_initial_segment (m n : ℕ) (h_coprime : Nat.gcd m n = 1) : 
  ∃ L, L = m + n - 2 := 
sorry

end max_common_initial_segment_l1931_193173


namespace extremum_point_of_f_l1931_193162

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem extremum_point_of_f : ∃ x, x = 1 ∧ (∀ y ≠ 1, f y ≥ f x) := 
sorry

end extremum_point_of_f_l1931_193162


namespace five_algorithmic_statements_l1931_193160

-- Define the five types of algorithmic statements in programming languages
inductive AlgorithmicStatement : Type
| input : AlgorithmicStatement
| output : AlgorithmicStatement
| assignment : AlgorithmicStatement
| conditional : AlgorithmicStatement
| loop : AlgorithmicStatement

-- Theorem: Every programming language contains these five basic types of algorithmic statements
theorem five_algorithmic_statements : 
  ∃ (s : List AlgorithmicStatement), 
    (s.length = 5) ∧ 
    ∀ x, x ∈ s ↔
    x = AlgorithmicStatement.input ∨
    x = AlgorithmicStatement.output ∨
    x = AlgorithmicStatement.assignment ∨
    x = AlgorithmicStatement.conditional ∨
    x = AlgorithmicStatement.loop :=
by
  sorry

end five_algorithmic_statements_l1931_193160


namespace total_fruit_weight_l1931_193193

def melon_weight : Real := 0.35
def berries_weight : Real := 0.48
def grapes_weight : Real := 0.29
def pineapple_weight : Real := 0.56
def oranges_weight : Real := 0.17

theorem total_fruit_weight : melon_weight + berries_weight + grapes_weight + pineapple_weight + oranges_weight = 1.85 :=
by
  unfold melon_weight berries_weight grapes_weight pineapple_weight oranges_weight
  sorry

end total_fruit_weight_l1931_193193


namespace colored_pencils_more_than_erasers_l1931_193187

def colored_pencils_initial := 67
def erasers_initial := 38

def colored_pencils_final := 50
def erasers_final := 28

theorem colored_pencils_more_than_erasers :
  colored_pencils_final - erasers_final = 22 := by
  sorry

end colored_pencils_more_than_erasers_l1931_193187


namespace lychees_remaining_l1931_193106
-- Definitions of the given conditions
def initial_lychees : ℕ := 500
def sold_lychees : ℕ := initial_lychees / 2
def home_lychees : ℕ := initial_lychees - sold_lychees
def eaten_lychees : ℕ := (3 * home_lychees) / 5

-- Statement to prove
theorem lychees_remaining : home_lychees - eaten_lychees = 100 := by
  sorry

end lychees_remaining_l1931_193106


namespace tap_emptying_time_l1931_193183

theorem tap_emptying_time
  (F : ℝ := 1 / 3)
  (T_combined : ℝ := 7.5):
  ∃ x : ℝ, x = 5 ∧ (F - (1 / x) = 1 / T_combined) := 
sorry

end tap_emptying_time_l1931_193183


namespace sixth_grade_boys_l1931_193167

theorem sixth_grade_boys (x : ℕ) :
    (1 / 11) * x + (147 - x) = 147 - x → 
    (152 - (x - (1 / 11) * x + (147 - x) - (152 - x - 5))) = x
    → x = 77 :=
by
  intros h1 h2
  sorry

end sixth_grade_boys_l1931_193167


namespace hotdogs_remainder_zero_l1931_193151

theorem hotdogs_remainder_zero :
  25197624 % 6 = 0 :=
by
  sorry -- Proof not required

end hotdogs_remainder_zero_l1931_193151


namespace bus_speed_including_stoppages_l1931_193130

theorem bus_speed_including_stoppages :
  ∀ (s t : ℝ), s = 75 → t = 24 → (s * ((60 - t) / 60)) = 45 :=
by
  intros s t hs ht
  rw [hs, ht]
  sorry

end bus_speed_including_stoppages_l1931_193130


namespace bottles_sold_tuesday_l1931_193133

def initial_inventory : ℕ := 4500
def sold_monday : ℕ := 2445
def sold_days_wed_to_sun : ℕ := 50 * 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

theorem bottles_sold_tuesday : 
  initial_inventory + bottles_delivered_saturday - sold_monday - sold_days_wed_to_sun - final_inventory = 900 := 
by
  sorry

end bottles_sold_tuesday_l1931_193133


namespace difference_in_zits_l1931_193100

variable (avgZitsSwanson : ℕ := 5)
variable (avgZitsJones : ℕ := 6)
variable (numKidsSwanson : ℕ := 25)
variable (numKidsJones : ℕ := 32)
variable (totalZitsSwanson : ℕ := avgZitsSwanson * numKidsSwanson)
variable (totalZitsJones : ℕ := avgZitsJones * numKidsJones)

theorem difference_in_zits :
  totalZitsJones - totalZitsSwanson = 67 := by
  sorry

end difference_in_zits_l1931_193100


namespace Maxim_is_correct_l1931_193171

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end Maxim_is_correct_l1931_193171


namespace angle_bisector_segment_rel_l1931_193117

variable (a b c : ℝ) -- The sides of the triangle
variable (u v : ℝ)   -- The segments into which fa divides side a
variable (fa : ℝ)    -- The length of the angle bisector

-- Statement setting up the given conditions and the proof we need
theorem angle_bisector_segment_rel : 
  (u : ℝ) = a * c / (b + c) → 
  (v : ℝ) = a * b / (b + c) → 
  (fa : ℝ) = 2 * (Real.sqrt (b * s * (s - a) * c)) / (b + c) → 
  fa^2 = b * c - u * v :=
sorry

end angle_bisector_segment_rel_l1931_193117


namespace max_items_with_discount_l1931_193198

theorem max_items_with_discount (total_money items original_price discount : ℕ) 
  (h_orig: original_price = 30)
  (h_discount: discount = 24) 
  (h_limit: items > 5 → (total_money <= 270)) : items ≤ 10 :=
by
  sorry

end max_items_with_discount_l1931_193198


namespace polygon_sum_of_sides_l1931_193144

-- Define the problem conditions and statement
theorem polygon_sum_of_sides :
  ∀ (A B C D E F : ℝ)
    (area_polygon : ℝ)
    (AB BC FA DE horizontal_distance_DF : ℝ),
    area_polygon = 75 →
    AB = 7 →
    BC = 10 →
    FA = 6 →
    DE = AB →
    horizontal_distance_DF = 8 →
    (DE + (2 * area_polygon - AB * BC) / (2 * horizontal_distance_DF) = 8.25) := 
by
  intro A B C D E F area_polygon AB BC FA DE horizontal_distance_DF
  intro h_area_polygon h_AB h_BC h_FA h_DE h_horizontal_distance_DF
  sorry

end polygon_sum_of_sides_l1931_193144


namespace remainder_when_divided_by_5_l1931_193190

theorem remainder_when_divided_by_5 (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3)
  (h3 : k < 41) : k % 5 = 2 :=
sorry

end remainder_when_divided_by_5_l1931_193190


namespace set_inclusion_interval_l1931_193148

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_interval_l1931_193148


namespace proofSmallestM_l1931_193146

def LeanProb (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 2512 →
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ (0 < e) ∧ (0 < f) →
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))))

theorem proofSmallestM (a b c d e f : ℕ) (h1 : a + b + c + d + e + f = 2512) 
(h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < d) (h6 : 0 < e) (h7 : 0 < f) : 
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f))))):=
by
  sorry

end proofSmallestM_l1931_193146


namespace value_of_expression_l1931_193180

theorem value_of_expression : (3023 - 2990) ^ 2 / 121 = 9 := by
  sorry

end value_of_expression_l1931_193180


namespace coin_difference_l1931_193164

-- Define the coin denominations
def coin_denominations : List ℕ := [5, 10, 25, 50]

-- Define the target amount Paul needs to pay
def target_amount : ℕ := 60

-- Define the function to compute the minimum number of coins required
noncomputable def min_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the function to compute the maximum number of coins required
noncomputable def max_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the theorem to state the difference between max and min coins is 10
theorem coin_difference : max_coins target_amount coin_denominations - min_coins target_amount coin_denominations = 10 :=
  sorry

end coin_difference_l1931_193164


namespace theo_selling_price_l1931_193168

theorem theo_selling_price:
  ∀ (maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell: ℕ),
    maddox_price = 20 → 
    theo_cost = 20 → 
    maddox_sell = 28 →
    maddox_profit = (maddox_sell - maddox_price) * 3 →
    (theo_sell - theo_cost) * 3 = (maddox_profit - 15) →
    theo_sell = 23 := by
  intros maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell
  intros maddox_price_eq theo_cost_eq maddox_sell_eq maddox_profit_eq theo_profit_eq

  -- Use given assumptions
  rw [maddox_price_eq, theo_cost_eq, maddox_sell_eq] at *
  simp at *

  -- Final goal
  sorry

end theo_selling_price_l1931_193168


namespace find_amount_l1931_193197

-- Let A be the certain amount.
variable (A x : ℝ)

-- Given conditions
def condition1 (x : ℝ) := 0.65 * x = 0.20 * A
def condition2 (x : ℝ) := x = 150

-- Goal
theorem find_amount (A x : ℝ) (h1 : condition1 A x) (h2 : condition2 x) : A = 487.5 := 
by 
  sorry

end find_amount_l1931_193197


namespace bus_stops_for_18_minutes_l1931_193176

-- Definitions based on conditions
def speed_without_stoppages : ℝ := 50 -- kmph
def speed_with_stoppages : ℝ := 35 -- kmph
def distance_reduced_due_to_stoppage_per_hour : ℝ := speed_without_stoppages - speed_with_stoppages

noncomputable def time_bus_stops_per_hour (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem bus_stops_for_18_minutes :
  time_bus_stops_per_hour distance_reduced_due_to_stoppage_per_hour (speed_without_stoppages / 60) = 18 := by
  sorry

end bus_stops_for_18_minutes_l1931_193176


namespace find_k_l1931_193178

theorem find_k (a k : ℝ) (h : a ≠ 0) (h1 : 3 * a + a = -12)
  (h2 : (3 * a) * a = k) : k = 27 :=
by
  sorry

end find_k_l1931_193178
