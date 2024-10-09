import Mathlib

namespace cos_sum_condition_l1616_161673

theorem cos_sum_condition {x y z : ℝ} (h1 : Real.cos x + Real.cos y + Real.cos z = 1) (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := 
by 
  sorry

end cos_sum_condition_l1616_161673


namespace johnson_family_seating_l1616_161658

-- Defining the total number of children:
def total_children := 8

-- Defining the number of sons and daughters:
def sons := 5
def daughters := 3

-- Factoring in the total number of unrestricted seating arrangements:
def total_seating_arrangements : ℕ := Nat.factorial total_children

-- Factoring in the number of non-adjacent seating arrangements for sons:
def non_adjacent_arrangements : ℕ :=
  (Nat.factorial daughters) * (Nat.factorial sons)

-- The lean proof statement to prove:
theorem johnson_family_seating :
  total_seating_arrangements - non_adjacent_arrangements = 39600 :=
by
  sorry

end johnson_family_seating_l1616_161658


namespace books_bought_l1616_161670

def cost_price_of_books (n : ℕ) (C : ℝ) (S : ℝ) : Prop :=
  n * C = 16 * S

def gain_or_loss_percentage (gain_loss_percent : ℝ) : Prop :=
  gain_loss_percent = 0.5

def loss_selling_price (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) : Prop :=
  S = (1 - gain_loss_percent) * C
  
theorem books_bought (n : ℕ) (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) 
  (h1 : cost_price_of_books n C S) 
  (h2 : gain_or_loss_percentage gain_loss_percent) 
  (h3 : loss_selling_price C S gain_loss_percent) : 
  n = 8 := 
sorry 

end books_bought_l1616_161670


namespace range_of_a_l1616_161600

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, -1 ≤ x → f a x ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by sorry

end range_of_a_l1616_161600


namespace fg_square_diff_l1616_161655

open Real

noncomputable def f (x: ℝ) : ℝ := sorry
noncomputable def g (x: ℝ) : ℝ := sorry

axiom h1 (x: ℝ) (hx : -π / 2 < x ∧ x < π / 2) : f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x))
axiom h2 : ∀ x, f (-x) = -f x
axiom h3 : ∀ x, g (-x) = g x

theorem fg_square_diff (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end fg_square_diff_l1616_161655


namespace g_60_l1616_161662

noncomputable def g : ℝ → ℝ :=
sorry

axiom g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

axiom g_45 : g 45 = 15

theorem g_60 : g 60 = 11.25 :=
by
  sorry

end g_60_l1616_161662


namespace original_price_of_house_l1616_161698

theorem original_price_of_house (P: ℝ) (sold_price: ℝ) (profit: ℝ) (commission: ℝ):
  sold_price = 100000 ∧ profit = 0.20 ∧ commission = 0.05 → P = 86956.52 :=
by
  sorry -- Proof not provided

end original_price_of_house_l1616_161698


namespace remaining_distance_l1616_161693

-- Definitions of conditions
def distance_to_grandmother : ℕ := 300
def speed_per_hour : ℕ := 60
def time_elapsed : ℕ := 2

-- Statement of the proof problem
theorem remaining_distance : distance_to_grandmother - (speed_per_hour * time_elapsed) = 180 :=
by 
  sorry

end remaining_distance_l1616_161693


namespace geometric_series_ratio_l1616_161685

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l1616_161685


namespace fraction_to_terminating_decimal_l1616_161675

theorem fraction_to_terminating_decimal : (21 : ℚ) / 40 = 0.525 := 
by
  sorry

end fraction_to_terminating_decimal_l1616_161675


namespace arithmetic_sum_l1616_161637

variables {a d : ℝ}

theorem arithmetic_sum (h : 15 * a + 105 * d = 90) : 2 * a + 14 * d = 12 :=
sorry

end arithmetic_sum_l1616_161637


namespace percentage_difference_l1616_161686

theorem percentage_difference (n z x y y_decreased : ℝ)
  (h1 : x = 8 * y)
  (h2 : y = 2 * |z - n|)
  (h3 : z = 1.1 * n)
  (h4 : y_decreased = 0.75 * y) :
  (x - y_decreased) / x * 100 = 90.625 := by
sorry

end percentage_difference_l1616_161686


namespace problem_statement_l1616_161646

theorem problem_statement : 6 * (3/2 + 2/3) = 13 :=
by
  sorry

end problem_statement_l1616_161646


namespace value_of_k_l1616_161676

theorem value_of_k (k : ℝ) :
  (5 + ∑' n : ℕ, (5 + k * (2^n / 4^n))) / 4^n = 10 → k = 15 :=
by
  sorry

end value_of_k_l1616_161676


namespace B_visible_from_A_l1616_161654

noncomputable def visibility_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 4 * x - 2 > 2 * x^2

theorem B_visible_from_A (a : ℝ) : visibility_condition a ↔ a < 10 :=
by
  -- sorry statement is used to skip the proof part.
  sorry

end B_visible_from_A_l1616_161654


namespace vasya_birthday_day_l1616_161605

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l1616_161605


namespace total_pens_bought_l1616_161684

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l1616_161684


namespace base_number_is_two_l1616_161667

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^18) (h2 : n = 17) : x = 2 :=
by sorry

end base_number_is_two_l1616_161667


namespace rectangular_area_l1616_161610

theorem rectangular_area (length width : ℝ) (h₁ : length = 0.4) (h₂ : width = 0.22) :
  (length * width = 0.088) :=
by sorry

end rectangular_area_l1616_161610


namespace scientific_notation_of_one_point_six_million_l1616_161615

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end scientific_notation_of_one_point_six_million_l1616_161615


namespace smallest_number_l1616_161648

theorem smallest_number
  (A : ℕ := 2^3 + 2^2 + 2^1 + 2^0)
  (B : ℕ := 2 * 6^2 + 1 * 6)
  (C : ℕ := 1 * 4^3)
  (D : ℕ := 8 + 1) :
  A < B ∧ A < C ∧ A < D :=
by {
  sorry
}

end smallest_number_l1616_161648


namespace triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l1616_161640

theorem triangle_side_square_sum_eq_three_times_centroid_dist_square_sum
  {A B C O : EuclideanSpace ℝ (Fin 2)}
  (h_centroid : O = (1/3 : ℝ) • (A + B + C)) :
  (dist A B)^2 + (dist B C)^2 + (dist C A)^2 =
  3 * ((dist O A)^2 + (dist O B)^2 + (dist O C)^2) :=
sorry

end triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l1616_161640


namespace simplify_and_evaluate_sqrt_log_product_property_l1616_161687

-- Problem I
theorem simplify_and_evaluate_sqrt (a : ℝ) (h : 0 < a) : 
  Real.sqrt (a^(1/4) * Real.sqrt (a * Real.sqrt a)) = Real.sqrt a := 
by
  sorry

-- Problem II
theorem log_product_property : 
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := 
by
  sorry

end simplify_and_evaluate_sqrt_log_product_property_l1616_161687


namespace race_problem_l1616_161635

theorem race_problem 
  (A B C : ℝ) 
  (h1 : A = 100) 
  (h2 : B = 100 - x) 
  (h3 : C = 72) 
  (h4 : B = C + 4)
  : x = 24 := 
by 
  sorry

end race_problem_l1616_161635


namespace armistice_day_is_wednesday_l1616_161691

-- Define the starting date
def start_day : Nat := 5 -- 5 represents Friday if we consider 0 = Sunday

-- Define the number of days after which armistice was signed
def days_after : Nat := 2253

-- Define the target day (Wednesday = 3)
def expected_day : Nat := 3

-- Define the function to calculate the day of the week after a number of days
def day_after_n_days (start_day : Nat) (n : Nat) : Nat :=
  (start_day + n) % 7

-- Define the theorem to prove the equivalent mathematical problem
theorem armistice_day_is_wednesday : day_after_n_days start_day days_after = expected_day := by
  sorry

end armistice_day_is_wednesday_l1616_161691


namespace hyperbola_condition_l1616_161652

theorem hyperbola_condition (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) → m < 0 ∨ m > 2 :=
sorry

end hyperbola_condition_l1616_161652


namespace kaylin_is_younger_by_five_l1616_161695

def Freyja_age := 10
def Kaylin_age := 33
def Eli_age := Freyja_age + 9
def Sarah_age := 2 * Eli_age
def age_difference := Sarah_age - Kaylin_age

theorem kaylin_is_younger_by_five : age_difference = 5 := 
by
  show 5 = Sarah_age - Kaylin_age
  sorry

end kaylin_is_younger_by_five_l1616_161695


namespace complement_union_l1616_161618

namespace SetComplement

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union :
  U \ (A ∪ B) = {1, 2, 6} := by
  sorry

end SetComplement

end complement_union_l1616_161618


namespace ellipse_k_range_ellipse_k_eccentricity_l1616_161604

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) ↔ (1 < k ∧ k < 5 ∨ 5 < k ∧ k < 9) := 
sorry

theorem ellipse_k_eccentricity (k : ℝ) (h : ∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) : 
  eccentricity = Real.sqrt (6/7) → (k = 2 ∨ k = 8) := 
sorry

end ellipse_k_range_ellipse_k_eccentricity_l1616_161604


namespace parabola_equation_l1616_161677

theorem parabola_equation 
  (vertex_x vertex_y : ℝ)
  (a b c : ℝ)
  (h_vertex : vertex_x = 3 ∧ vertex_y = 5)
  (h_point : ∃ x y: ℝ, x = 2 ∧ y = 2 ∧ y = a * (x - vertex_x)^2 + vertex_y)
  (h_vertical_axis : ∃ a b c, a = -3 ∧ b = 18 ∧ c = -22):
  ∀ x: ℝ, x ≠ vertex_x → b^2 - 4 * a * c > 0 := 
    sorry

end parabola_equation_l1616_161677


namespace gain_percentage_for_40_clocks_is_10_l1616_161617

-- Condition: Cost price per clock
def cost_price := 79.99999999999773

-- Condition: Selling price of 50 clocks at a gain of 20%
def selling_price_50 := 50 * cost_price * 1.20

-- Uniform profit condition
def uniform_profit_total := 90 * cost_price * 1.15

-- Given total revenue difference Rs. 40
def total_revenue := uniform_profit_total + 40

-- Question: Prove that selling price of 40 clocks leads to 10% gain
theorem gain_percentage_for_40_clocks_is_10 :
    40 * cost_price * 1.10 = total_revenue - selling_price_50 :=
by
  sorry

end gain_percentage_for_40_clocks_is_10_l1616_161617


namespace arithmetic_geometric_sequence_l1616_161622

theorem arithmetic_geometric_sequence : 
  ∀ (a : ℤ), (∀ n : ℤ, a_n = a + (n-1) * 2) → 
  (a + 4)^2 = a * (a + 6) → 
  (a + 10 = 2) :=
by
  sorry

end arithmetic_geometric_sequence_l1616_161622


namespace convert_mixed_decimals_to_fractions_l1616_161608

theorem convert_mixed_decimals_to_fractions :
  (4.26 = 4 + 13/50) ∧
  (1.15 = 1 + 3/20) ∧
  (3.08 = 3 + 2/25) ∧
  (2.37 = 2 + 37/100) :=
by
  -- Proof omitted
  sorry

end convert_mixed_decimals_to_fractions_l1616_161608


namespace complement_intersection_complement_in_U_l1616_161630

universe u
open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Definitions based on the conditions
def universal_set : Set ℕ := { x ∈ (Set.univ : Set ℕ) | x ≤ 4 }
def set_A : Set ℕ := {1, 4}
def set_B : Set ℕ := {2, 4}

-- Problem to be proven
theorem complement_intersection_complement_in_U :
  (U = universal_set) → (A = set_A) → (B = set_B) →
  compl (A ∩ B) ∩ U = {1, 2, 3} :=
by
  intro hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_complement_in_U_l1616_161630


namespace fifth_inequality_proof_l1616_161664

theorem fifth_inequality_proof :
  (1 + 1 / (2^2 : ℝ) + 1 / (3^2 : ℝ) + 1 / (4^2 : ℝ) + 1 / (5^2 : ℝ) + 1 / (6^2 : ℝ) < 11 / 6) 
  := 
sorry

end fifth_inequality_proof_l1616_161664


namespace certain_amount_l1616_161621

theorem certain_amount (x : ℝ) (h1 : 2 * x = 86 - 54) (h2 : 8 + 3 * 8 = 24) (h3 : 86 - 54 + 32 = 86) : x = 43 := 
by {
  sorry
}

end certain_amount_l1616_161621


namespace sum_of_abc_l1616_161612

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (eq1 : a^2 + b * c = 115) (eq2 : b^2 + a * c = 127) (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 := by
  sorry

end sum_of_abc_l1616_161612


namespace JackEmails_l1616_161679

theorem JackEmails (E : ℕ) (h1 : 10 = E + 7) : E = 3 :=
by
  sorry

end JackEmails_l1616_161679


namespace sum_of_x_values_satisfying_eq_l1616_161692

noncomputable def rational_eq_sum (x : ℝ) : Prop :=
3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

theorem sum_of_x_values_satisfying_eq :
  (∃ (x : ℝ), rational_eq_sum x) ∧ (x ≠ -3 → (x_1 + x_2) = 6) :=
sorry

end sum_of_x_values_satisfying_eq_l1616_161692


namespace profit_sharing_l1616_161649

theorem profit_sharing 
  (total_profit : ℝ) 
  (managing_share_percentage : ℝ) 
  (capital_a : ℝ) 
  (capital_b : ℝ) 
  (managing_partner_share : ℝ)
  (total_capital : ℝ) 
  (remaining_profit : ℝ) 
  (proportion_a : ℝ)
  (share_a_remaining : ℝ)
  (total_share_a : ℝ) : 
  total_profit = 8800 → 
  managing_share_percentage = 0.125 → 
  capital_a = 50000 → 
  capital_b = 60000 → 
  managing_partner_share = managing_share_percentage * total_profit → 
  total_capital = capital_a + capital_b → 
  remaining_profit = total_profit - managing_partner_share → 
  proportion_a = capital_a / total_capital → 
  share_a_remaining = proportion_a * remaining_profit → 
  total_share_a = managing_partner_share + share_a_remaining → 
  total_share_a = 4600 :=
by sorry

end profit_sharing_l1616_161649


namespace obrien_hats_theorem_l1616_161603

-- Define the number of hats Fire Chief Simpson has.
def simpson_hats : ℕ := 15

-- Define the number of hats Policeman O'Brien had before any hats were stolen.
def obrien_hats_before (simpson_hats : ℕ) : ℕ := 2 * simpson_hats + 5

-- Define the number of hats Policeman O'Brien has now, after x hats were stolen.
def obrien_hats_now (x : ℕ) : ℕ := obrien_hats_before simpson_hats - x

-- Define the theorem stating the problem
theorem obrien_hats_theorem (x : ℕ) : obrien_hats_now x = 35 - x :=
by
  sorry

end obrien_hats_theorem_l1616_161603


namespace chicago_bulls_wins_l1616_161601

theorem chicago_bulls_wins (B H : ℕ) (h1 : B + H = 145) (h2 : H = B + 5) : B = 70 :=
by
  sorry

end chicago_bulls_wins_l1616_161601


namespace equivalent_problem_l1616_161643

theorem equivalent_problem : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 := by
  sorry

end equivalent_problem_l1616_161643


namespace value_of_c_infinite_solutions_l1616_161668

theorem value_of_c_infinite_solutions (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ (c = 3) :=
by
  sorry

end value_of_c_infinite_solutions_l1616_161668


namespace polynomial_value_at_2018_l1616_161607

theorem polynomial_value_at_2018 (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f (-x^2 - x - 1) = x^4 + 2*x^3 + 2022*x^2 + 2021*x + 2019) : 
  f 2018 = -2019 :=
sorry

end polynomial_value_at_2018_l1616_161607


namespace savings_calculation_l1616_161627

-- Definitions of the given conditions
def window_price : ℕ := 100
def free_window_offer (purchased : ℕ) : ℕ := purchased / 4

-- Number of windows needed
def dave_needs : ℕ := 7
def doug_needs : ℕ := 8

-- Calculations based on the conditions
def individual_costs : ℕ :=
  (dave_needs - free_window_offer dave_needs) * window_price +
  (doug_needs - free_window_offer doug_needs) * window_price

def together_costs : ℕ :=
  let total_needs := dave_needs + doug_needs
  (total_needs - free_window_offer total_needs) * window_price

def savings : ℕ := individual_costs - together_costs

-- Proof statement
theorem savings_calculation : savings = 100 := by
  sorry

end savings_calculation_l1616_161627


namespace Jason_spent_on_music_store_l1616_161665

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end Jason_spent_on_music_store_l1616_161665


namespace pedestrian_travel_time_l1616_161674

noncomputable def travel_time (d : ℝ) (x y : ℝ) : ℝ :=
  d / x

theorem pedestrian_travel_time
  (d : ℝ)
  (x y : ℝ)
  (h1 : d = 1)
  (h2 : 3 * x = 1 - x - y)
  (h3 : (1 / 2) * (x + y) = 1 - x - y)
  : travel_time d x y = 9 := 
sorry

end pedestrian_travel_time_l1616_161674


namespace joe_bath_shop_bottles_l1616_161678

theorem joe_bath_shop_bottles (b : ℕ) (n : ℕ) (m : ℕ) 
    (h1 : 5 * n = b * m)
    (h2 : 5 * n = 95)
    (h3 : b * m = 95)
    (h4 : b ≠ 1)
    (h5 : b ≠ 95): 
    b = 19 := 
by 
    sorry

end joe_bath_shop_bottles_l1616_161678


namespace greatest_consecutive_integers_sum_120_l1616_161629

def sum_of_consecutive_integers (n : ℤ) (a : ℤ) : ℤ :=
  n * (2 * a + n - 1) / 2

theorem greatest_consecutive_integers_sum_120 (N : ℤ) (a : ℤ) (h1 : sum_of_consecutive_integers N a = 120) : N ≤ 240 :=
by {
  -- Here we would provide the proof, but it's omitted with 'sorry'.
  sorry
}

end greatest_consecutive_integers_sum_120_l1616_161629


namespace min_value_of_m_n_l1616_161697

variable {a b : ℝ}
variable (ab_eq_4 : a * b = 4)
variable (m : ℝ := b + 1 / a)
variable (n : ℝ := a + 1 / b)

theorem min_value_of_m_n (h1 : 0 < a) (h2 : 0 < b) : m + n = 5 :=
sorry

end min_value_of_m_n_l1616_161697


namespace max_value_of_ratio_l1616_161690

theorem max_value_of_ratio (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : 
  ∃ z, z = (x / y) ∧ z ≤ 1 := sorry

end max_value_of_ratio_l1616_161690


namespace sam_bought_new_books_l1616_161696

   def books_question (a m u : ℕ) : ℕ := (a + m) - u

   theorem sam_bought_new_books (a m u : ℕ) (h1 : a = 13) (h2 : m = 17) (h3 : u = 15) :
     books_question a m u = 15 :=
   by sorry
   
end sam_bought_new_books_l1616_161696


namespace continuity_of_f_at_3_l1616_161619

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3*x^2 + 2*x - 4 else b*x + 7

theorem continuity_of_f_at_3 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x b - f 3 b) < ε) ↔ b = 22 / 3 :=
by
  sorry

end continuity_of_f_at_3_l1616_161619


namespace number_of_six_digit_palindromes_l1616_161623

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end number_of_six_digit_palindromes_l1616_161623


namespace simplify_expression_l1616_161656

theorem simplify_expression : (2^8 + 4^5) * ((1^3 - (-1)^3)^8) = 327680 := by
  sorry

end simplify_expression_l1616_161656


namespace max_value_of_f_on_S_l1616_161631

noncomputable def S : Set ℝ := { x | x^4 - 13 * x^2 + 36 ≤ 0 }
noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_of_f_on_S : ∃ x ∈ S, ∀ y ∈ S, f y ≤ f x ∧ f x = 18 :=
by
  sorry

end max_value_of_f_on_S_l1616_161631


namespace extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l1616_161628

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x * exp x - (x + 1) ^ 2

-- Question 1: Extreme value when a = -1
theorem extreme_value_when_a_is_neg_one : 
  f (-1) (-1) = 1 / exp 1 := sorry

-- Question 2: Range of a such that ∀ x ∈ [-1, 1], f(x) ≤ 0
theorem range_of_a_for_f_non_positive :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 4 / exp 1 := sorry

end extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l1616_161628


namespace weight_of_each_bag_is_correct_l1616_161651

noncomputable def weightOfEachBag
    (days1 : ℕ := 60)
    (consumption1 : ℕ := 2)
    (days2 : ℕ := 305)
    (consumption2 : ℕ := 4)
    (ouncesPerPound : ℕ := 16)
    (numberOfBags : ℕ := 17) : ℝ :=
        let totalOunces := (days1 * consumption1) + (days2 * consumption2)
        let totalPounds := totalOunces / ouncesPerPound
        totalPounds / numberOfBags

theorem weight_of_each_bag_is_correct :
  weightOfEachBag = 4.93 :=
by
  sorry

end weight_of_each_bag_is_correct_l1616_161651


namespace older_brother_catches_up_l1616_161614

theorem older_brother_catches_up :
  ∃ (x : ℝ), 0 ≤ x ∧ 6 * x = 2 + 2 * x ∧ x + 1 < 1.75 :=
by
  sorry

end older_brother_catches_up_l1616_161614


namespace gcd_of_256_180_600_l1616_161683

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l1616_161683


namespace mysterious_neighbor_is_13_l1616_161669

variable (x : ℕ) (h1 : x < 15) (h2 : 2 * x * 30 = 780)

theorem mysterious_neighbor_is_13 : x = 13 :=
by {
    sorry 
}

end mysterious_neighbor_is_13_l1616_161669


namespace min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l1616_161661

section
variables {a x : ℝ}

/-- Define the function f(x) = ax^3 - 2x^2 + x + c where c = 1 -/
def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + x + 1

/-- Proposition 1: Minimum value of f when a = 1 and f passes through (0,1) is 1 -/
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f 1 x ≥ 1) := 
by {
  -- Sorry for the full proof
  sorry
}

/-- Proposition 2: If f has no extremum points, then a ≥ 4/3 -/
theorem no_extrema_implies_a_ge_four_thirds (h : ∀ x : ℝ, 3 * a * x^2 - 4 * x + 1 ≠ 0) : 
  a ≥ (4 / 3) :=
by {
  -- Sorry for the full proof
  sorry
}

end

end min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l1616_161661


namespace find_weight_of_a_l1616_161647

variables (a b c d e : ℕ)

-- Conditions
def cond1 : Prop := a + b + c = 252
def cond2 : Prop := a + b + c + d = 320
def cond3 : Prop := e = d + 7
def cond4 : Prop := b + c + d + e = 316

theorem find_weight_of_a (h1 : cond1 a b c) (h2 : cond2 a b c d) (h3 : cond3 d e) (h4 : cond4 b c d e) :
  a = 79 :=
by sorry

end find_weight_of_a_l1616_161647


namespace find_cos_alpha_l1616_161653

theorem find_cos_alpha 
  (α : ℝ) 
  (h₁ : Real.tan (π - α) = 3/4) 
  (h₂ : α ∈ Set.Ioo (π/2) π) 
: Real.cos α = -4/5 :=
sorry

end find_cos_alpha_l1616_161653


namespace n_squared_divides_2n_plus_1_l1616_161620

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end n_squared_divides_2n_plus_1_l1616_161620


namespace jeremy_school_distance_l1616_161642

theorem jeremy_school_distance :
  ∃ d : ℝ, d = 9.375 ∧
  (∃ v : ℝ, (d = v * (15 / 60)) ∧ (d = (v + 25) * (9 / 60))) := by
  sorry

end jeremy_school_distance_l1616_161642


namespace problem_statement_l1616_161641

def P (m n : ℕ) : ℕ :=
  let coeff_x := Nat.choose 4 m
  let coeff_y := Nat.choose 6 n
  coeff_x * coeff_y

theorem problem_statement : P 2 1 + P 1 2 = 96 :=
by
  sorry

end problem_statement_l1616_161641


namespace arithmetic_sequence_a7_l1616_161632

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)) : a 7 = 9 :=
by
  sorry

end arithmetic_sequence_a7_l1616_161632


namespace interest_rate_increase_l1616_161672

-- Define the conditions
def principal (P : ℕ) := P = 1000
def time (t : ℕ) := t = 5
def original_amount (A : ℕ) := A = 1500
def new_amount (A' : ℕ) := A' = 1750

-- Prove that the interest rate increase is 50%
theorem interest_rate_increase
  (P : ℕ) (t : ℕ) (A A' : ℕ)
  (hP : principal P)
  (ht : time t)
  (hA : original_amount A)
  (hA' : new_amount A') :
  (((((A' - P) / (P * t)) - ((A - P) / (P * t))) / ((A - P) / (P * t))) * 100) = 50 := by
  sorry

end interest_rate_increase_l1616_161672


namespace range_of_k_l1616_161602

theorem range_of_k (k : ℝ) (H : ∀ x : ℤ, |(x : ℝ) - 1| < k * x ↔ x ∈ ({1, 2, 3} : Set ℤ)) : 
  (2 / 3 : ℝ) < k ∧ k ≤ (3 / 4 : ℝ) :=
by
  sorry

end range_of_k_l1616_161602


namespace total_fireworks_l1616_161657

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end total_fireworks_l1616_161657


namespace find_some_number_l1616_161638

-- The conditions of the problem
variables (x y : ℝ)
axiom cond1 : 2 * x + y = 7
axiom cond2 : x + 2 * y = 5

-- The "some number" we want to prove exists
def some_number := 3

-- Statement of the problem: the value of 2xy / some_number should equal 2
theorem find_some_number (x y : ℝ) (cond1 : 2 * x + y = 7) (cond2 : x + 2 * y = 5) :
  2 * x * y / some_number = 2 :=
sorry

end find_some_number_l1616_161638


namespace standard_polar_representation_l1616_161689

theorem standard_polar_representation {r θ : ℝ} (hr : r < 0) (hθ : θ = 5 * Real.pi / 6) :
  ∃ (r' θ' : ℝ), r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * Real.pi ∧ (r', θ') = (5, 11 * Real.pi / 6) := 
by {
  sorry
}

end standard_polar_representation_l1616_161689


namespace common_ratio_of_geometric_sequence_is_4_l1616_161616

theorem common_ratio_of_geometric_sequence_is_4 
  (a_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ) 
  (d : ℝ) 
  (h₁ : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h₂ : d ≠ 0)
  (h₃ : (a_n 3)^2 = (a_n 2) * (a_n 7)) :
  b_n 2 / b_n 1 = 4 :=
sorry

end common_ratio_of_geometric_sequence_is_4_l1616_161616


namespace number_of_rabbits_l1616_161663

-- Defining the problem conditions
variables (x y : ℕ)
axiom heads_condition : x + y = 40
axiom legs_condition : 4 * x = 10 * 2 * y - 8

--  Prove the number of rabbits is 33
theorem number_of_rabbits : x = 33 :=
by
  sorry

end number_of_rabbits_l1616_161663


namespace smallest_n_for_fraction_with_digits_439_l1616_161634

theorem smallest_n_for_fraction_with_digits_439 (m n : ℕ) (hmn : Nat.gcd m n = 1) (hmn_pos : 0 < m ∧ m < n) (digits_439 : ∃ X : ℕ, (m : ℚ) / n = (439 + 1000 * X) / 1000) : n = 223 :=
by
  sorry

end smallest_n_for_fraction_with_digits_439_l1616_161634


namespace find_land_area_l1616_161626

variable (L : ℝ) -- cost of land per square meter
variable (B : ℝ) -- cost of bricks per 1000 bricks
variable (R : ℝ) -- cost of roof tiles per tile
variable (numBricks : ℝ) -- number of bricks needed
variable (numTiles : ℝ) -- number of roof tiles needed
variable (totalCost : ℝ) -- total construction cost

theorem find_land_area (h1 : L = 50) 
                       (h2 : B = 100)
                       (h3 : R = 10) 
                       (h4 : numBricks = 10000) 
                       (h5 : numTiles = 500) 
                       (h6 : totalCost = 106000) : 
                       ∃ x : ℝ, 50 * x + (numBricks / 1000) * B + numTiles * R = totalCost ∧ x = 2000 := 
by 
  use 2000
  simp [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end find_land_area_l1616_161626


namespace original_surface_area_l1616_161625

theorem original_surface_area (R : ℝ) (h : 2 * π * R^2 = 4 * π) : 4 * π * R^2 = 8 * π :=
by
  sorry

end original_surface_area_l1616_161625


namespace Mayor_decision_to_adopt_model_A_l1616_161671

-- Define the conditions
def num_people := 17

def radicals_support_model_A := (0 : ℕ)

def socialists_support_model_B (y : ℕ) := y

def republicans_support_model_B (x y : ℕ) := x - y

def independents_support_model_B (x y : ℕ) := (y + (x - y)) / 2

-- The number of individuals supporting model A and model B
def support_model_B (x y : ℕ) := radicals_support_model_A + socialists_support_model_B y + republicans_support_model_B x y + independents_support_model_B x y

def support_model_A (x : ℕ) := 4 * x - support_model_B x x / 2

-- Statement to prove
theorem Mayor_decision_to_adopt_model_A (x : ℕ) (h : x = num_people) : 
  support_model_A x > support_model_B x x := 
by {
  -- Proof goes here
  sorry
}

end Mayor_decision_to_adopt_model_A_l1616_161671


namespace probability_intersection_inside_nonagon_correct_l1616_161699

def nonagon_vertices : ℕ := 9

def total_pairs_of_points := Nat.choose nonagon_vertices 2

def sides_of_nonagon : ℕ := nonagon_vertices

def diagonals_of_nonagon := total_pairs_of_points - sides_of_nonagon

def pairs_of_diagonals := Nat.choose diagonals_of_nonagon 2

def sets_of_intersecting_diagonals := Nat.choose nonagon_vertices 4

noncomputable def probability_intersection_inside_nonagon : ℚ :=
  sets_of_intersecting_diagonals / pairs_of_diagonals

theorem probability_intersection_inside_nonagon_correct :
  probability_intersection_inside_nonagon = 14 / 39 := 
  sorry

end probability_intersection_inside_nonagon_correct_l1616_161699


namespace smallest_value_of_N_l1616_161645

theorem smallest_value_of_N :
  ∃ N : ℕ, ∀ (P1 P2 P3 P4 P5 : ℕ) (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℕ),
    (P1 = 1 ∧ P2 = 2 ∧ P3 = 3 ∧ P4 = 4 ∧ P5 = 5) →
    (x1 = a_1 ∧ x2 = N + a_2 ∧ x3 = 2 * N + a_3 ∧ x4 = 3 * N + a_4 ∧ x5 = 4 * N + a_5) →
    (y1 = 5 * (a_1 - 1) + 1 ∧ y2 = 5 * (a_2 - 1) + 2 ∧ y3 = 5 * (a_3 - 1) + 3 ∧ y4 = 5 * (a_4 - 1) + 4 ∧ y5 = 5 * (a_5 - 1) + 5) →
    (x1 = y2 ∧ x2 = y1 ∧ x3 = y4 ∧ x4 = y5 ∧ x5 = y3) →
    N = 149 :=
sorry

end smallest_value_of_N_l1616_161645


namespace manuscript_pages_l1616_161636

theorem manuscript_pages (P : ℕ) (rate_first : ℕ) (rate_revision : ℕ) 
  (revised_once_pages : ℕ) (revised_twice_pages : ℕ) (total_cost : ℕ) :
  rate_first = 6 →
  rate_revision = 4 →
  revised_once_pages = 35 →
  revised_twice_pages = 15 →
  total_cost = 860 →
  6 * (P - 35 - 15) + 10 * 35 + 14 * 15 = total_cost →
  P = 100 :=
by
  intros h_first h_revision h_once h_twice h_cost h_eq
  sorry

end manuscript_pages_l1616_161636


namespace james_needs_more_marbles_l1616_161611

def number_of_additional_marbles (friends marbles : Nat) : Nat :=
  let required_marbles := (friends * (friends + 1)) / 2
  (if marbles < required_marbles then required_marbles - marbles else 0)

theorem james_needs_more_marbles :
  number_of_additional_marbles 15 80 = 40 := by
  sorry

end james_needs_more_marbles_l1616_161611


namespace boat_speed_in_still_water_l1616_161660

open Real

theorem boat_speed_in_still_water (V_s d t : ℝ) (h1 : V_s = 6) (h2 : d = 72) (h3 : t = 3.6) :
  ∃ (V_b : ℝ), V_b = 14 := by
  have V_d := d / t
  have V_b := V_d - V_s
  use V_b
  sorry

end boat_speed_in_still_water_l1616_161660


namespace triangle_identity_l1616_161694

theorem triangle_identity
  (A B C : ℝ) (a b c: ℝ)
  (h1: A + B + C = Real.pi)
  (h2: a = 2 * R * Real.sin A)
  (h3: b = 2 * R * Real.sin B)
  (h4: c = 2 * R * Real.sin C)
  (h5: Real.sin A = Real.sin B * Real.cos C + Real.cos B * Real.sin C) :
  (b * Real.cos C + c * Real.cos B) / a = 1 := 
  by 
  sorry

end triangle_identity_l1616_161694


namespace sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l1616_161639

-- 1. Sum of the interior angles in a triangle is 180 degrees.
theorem sum_of_angles_in_triangle : ∀ a : ℕ, (∀ x y z : ℕ, x + y + z = 180) → a = 180 := by
  intros a h
  have : a = 180 := sorry
  exact this

-- 2. Sum of interior angles of a regular b-sided polygon is 1080 degrees.
theorem sum_of_angles_in_polygon : ∀ b : ℕ, ((b - 2) * 180 = 1080) → b = 8 := by
  intros b h
  have : b = 8 := sorry
  exact this

-- 3. Exponential equation involving b.
theorem exponential_equation : ∀ p b : ℕ, (8 ^ b = p ^ 21) ∧ (b = 8) → p = 2 := by
  intros p b h
  have : p = 2 := sorry
  exact this

-- 4. Logarithmic equation involving p.
theorem logarithmic_equation : ∀ q p : ℕ, (p = Real.log 81 / Real.log q) ∧ (p = 2) → q = 9 := by
  intros q p h
  have : q = 9 := sorry
  exact this

end sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l1616_161639


namespace cube_surface_area_l1616_161666

noncomputable def volume_of_cube (s : ℝ) := s ^ 3
noncomputable def surface_area_of_cube (s : ℝ) := 6 * (s ^ 2)

theorem cube_surface_area (s : ℝ) (h : volume_of_cube s = 1728) : surface_area_of_cube s = 864 :=
  sorry

end cube_surface_area_l1616_161666


namespace min_adjacent_seat_occupation_l1616_161609

def minOccupiedSeats (n : ℕ) : ℕ :=
  n / 3

theorem min_adjacent_seat_occupation (n : ℕ) (h : n = 150) :
  minOccupiedSeats n = 50 :=
by
  -- Placeholder for proof
  sorry

end min_adjacent_seat_occupation_l1616_161609


namespace suit_price_after_discount_l1616_161613

-- Define the original price of the suit.
def original_price : ℝ := 150

-- Define the increase rate and the discount rate.
def increase_rate : ℝ := 0.20
def discount_rate : ℝ := 0.20

-- Define the increased price after the 20% increase.
def increased_price : ℝ := original_price * (1 + increase_rate)

-- Define the final price after applying the 20% discount.
def final_price : ℝ := increased_price * (1 - discount_rate)

-- Prove that the final price is $144.
theorem suit_price_after_discount : final_price = 144 := by
  sorry  -- Proof to be completed

end suit_price_after_discount_l1616_161613


namespace nine_y_squared_eq_x_squared_z_squared_l1616_161633

theorem nine_y_squared_eq_x_squared_z_squared (x y z : ℝ) (h : x / y = 3 / z) : 9 * y ^ 2 = x ^ 2 * z ^ 2 :=
by
  sorry

end nine_y_squared_eq_x_squared_z_squared_l1616_161633


namespace max_value_E_zero_l1616_161624

noncomputable def E (a b c : ℝ) : ℝ :=
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2)

theorem max_value_E_zero (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≥ b * c^2) (h2 : b ≥ c * a^2) (h3 : c ≥ a * b^2) :
  E a b c ≤ 0 :=
by
  sorry

end max_value_E_zero_l1616_161624


namespace solution_set_f_ge_1_l1616_161650

noncomputable def f (x : ℝ) (a : ℝ) :=
  if x >= 0 then |x - 2| + a else -(|-x - 2| + a)

theorem solution_set_f_ge_1 {a : ℝ} (ha : a = -2) :
  {x : ℝ | f x a ≥ 1} = {x : ℝ | x ≤ -1 ∨ x ≥ 5} :=
by sorry

end solution_set_f_ge_1_l1616_161650


namespace cricket_target_l1616_161606

theorem cricket_target (run_rate_first_10overs run_rate_next_40overs : ℝ) (overs_first_10 next_40_overs : ℕ)
    (h_first : run_rate_first_10overs = 3.2) 
    (h_next : run_rate_next_40overs = 6.25) 
    (h_overs_first : overs_first_10 = 10) 
    (h_overs_next : next_40_overs = 40) 
    : (overs_first_10 * run_rate_first_10overs + next_40_overs * run_rate_next_40overs) = 282 :=
by
  sorry

end cricket_target_l1616_161606


namespace two_digit_sum_divisible_by_17_l1616_161682

theorem two_digit_sum_divisible_by_17 :
  ∃ A : ℕ, A ≥ 10 ∧ A < 100 ∧ ∃ B : ℕ, B = (A % 10) * 10 + (A / 10) ∧ (A + B) % 17 = 0 ↔ A = 89 ∨ A = 98 := 
sorry

end two_digit_sum_divisible_by_17_l1616_161682


namespace simplify_expression_l1616_161644

variable (w : ℝ)

theorem simplify_expression : 3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end simplify_expression_l1616_161644


namespace product_a_b_l1616_161681

variable (a b c : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_pos_c : c > 0)
variable (h_c : c = 3)
variable (h_a : a = b^2)
variable (h_bc : b + c = b * c)

theorem product_a_b : a * b = 27 / 8 :=
by
  -- We need to prove that given the above conditions, a * b = 27 / 8
  sorry

end product_a_b_l1616_161681


namespace thirteen_pow_seven_mod_nine_l1616_161680

theorem thirteen_pow_seven_mod_nine : (13^7 % 9 = 4) :=
by {
  sorry
}

end thirteen_pow_seven_mod_nine_l1616_161680


namespace a1_a2_a3_sum_l1616_161659

-- Given conditions and hypothesis
variables (a0 a1 a2 a3 : ℝ)
axiom H : ∀ x : ℝ, 1 + x + x^2 + x^3 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3

-- Goal statement to be proven
theorem a1_a2_a3_sum : a1 + a2 + a3 = -3 :=
sorry

end a1_a2_a3_sum_l1616_161659


namespace marian_balance_proof_l1616_161688

noncomputable def marian_new_balance : ℝ :=
  let initial_balance := 126.00
  let uk_purchase := 50.0
  let uk_discount := 0.10
  let uk_rate := 1.39
  let france_purchase := 70.0
  let france_discount := 0.15
  let france_rate := 1.18
  let japan_purchase := 10000.0
  let japan_discount := 0.05
  let japan_rate := 0.0091
  let towel_return := 45.0
  let interest_rate := 0.015
  let uk_usd := (uk_purchase * (1 - uk_discount)) * uk_rate
  let france_usd := (france_purchase * (1 - france_discount)) * france_rate
  let japan_usd := (japan_purchase * (1 - japan_discount)) * japan_rate
  let gas_usd := (uk_purchase / 2) * uk_rate
  let balance_before_interest := initial_balance + uk_usd + france_usd + japan_usd + gas_usd - towel_return
  let interest := balance_before_interest * interest_rate
  balance_before_interest + interest

theorem marian_balance_proof :
  abs (marian_new_balance - 340.00) < 1 :=
by
  sorry

end marian_balance_proof_l1616_161688
