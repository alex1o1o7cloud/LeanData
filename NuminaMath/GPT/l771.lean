import Mathlib

namespace rectangle_side_ratio_l771_77190

theorem rectangle_side_ratio
  (s : ℝ) -- side length of inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_square : y = s) -- shorter side aligns to form inner square
  (h_outer_area : (3 * s) ^ 2 = 9 * s ^ 2) -- area of outer square is 9 times the inner square
  (h_outer_side_relation : x + s = 3 * s) -- outer side length relation
  : x / y = 2 := 
by
  sorry

end rectangle_side_ratio_l771_77190


namespace number_of_even_three_digit_numbers_l771_77163

theorem number_of_even_three_digit_numbers : 
  ∃ (count : ℕ), 
  count = 12 ∧ 
  (∀ (d1 d2 : ℕ), (0 ≤ d1 ∧ d1 ≤ 4) ∧ (Even d1) ∧ (0 ≤ d2 ∧ d2 ≤ 4) ∧ (Even d2) ∧ d1 ≠ d2 →
   ∃ (d3 : ℕ), (d3 = 1 ∨ d3 = 3) ∧ 
   ∃ (units tens hundreds : ℕ), 
     (units ∈ [0, 2, 4]) ∧ 
     (tens ∈ [0, 2, 4]) ∧ 
     (hundreds ∈ [1, 3]) ∧ 
     (units ≠ tens) ∧ 
     (units ≠ hundreds) ∧ 
     (tens ≠ hundreds) ∧ 
     ((units + tens * 10 + hundreds * 100) % 2 = 0) ∧ 
     count = 12) :=
sorry

end number_of_even_three_digit_numbers_l771_77163


namespace simplify_fraction_l771_77143

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l771_77143


namespace cannot_be_sum_of_six_consecutive_odds_l771_77110

def is_sum_of_six_consecutive_odds (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (6 * k + 30)

theorem cannot_be_sum_of_six_consecutive_odds :
  ¬ is_sum_of_six_consecutive_odds 198 ∧ ¬ is_sum_of_six_consecutive_odds 390 := 
sorry

end cannot_be_sum_of_six_consecutive_odds_l771_77110


namespace solution_positive_then_opposite_signs_l771_77150

theorem solution_positive_then_opposite_signs
  (a b : ℝ) (h : a ≠ 0) (x : ℝ) (hx : ax + b = 0) (x_pos : x > 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) :=
by
  sorry

end solution_positive_then_opposite_signs_l771_77150


namespace apples_on_tree_l771_77138

-- Defining initial number of apples on the tree
def initial_apples : ℕ := 4

-- Defining apples picked from the tree
def apples_picked : ℕ := 2

-- Defining new apples grown on the tree
def new_apples : ℕ := 3

-- Prove the final number of apples on the tree is 5
theorem apples_on_tree : initial_apples - apples_picked + new_apples = 5 :=
by
  -- This is where the proof would go
  sorry

end apples_on_tree_l771_77138


namespace max_value_exponential_and_power_functions_l771_77140

variable (a b : ℝ)

-- Given conditions
axiom condition : 0 < b ∧ b < a ∧ a < 1

-- Problem statement
theorem max_value_exponential_and_power_functions : 
  a^b = max (max (a^b) (b^a)) (max (a^a) (b^b)) :=
by
  sorry

end max_value_exponential_and_power_functions_l771_77140


namespace find_number_l771_77111

theorem find_number (x : ℝ) (h : x * 2 + (12 + 4) * (1/8) = 602) : x = 300 :=
by
  sorry

end find_number_l771_77111


namespace assistant_stop_time_l771_77129

-- Define the start time for the craftsman
def craftsmanStartTime : Nat := 8 * 60 -- in minutes

-- Craftsman starts at 8:00 AM and stops at 12:00 PM
def craftsmanEndTime : Nat := 12 * 60 -- in minutes

-- Craftsman produces 6 bracelets every 20 minutes
def craftsmanProductionPerMinute : Nat := 6 / 20

-- Assistant starts working at 9:00 AM
def assistantStartTime : Nat := 9 * 60 -- in minutes

-- Assistant produces 8 bracelets every 30 minutes
def assistantProductionPerMinute : Nat := 8 / 30

-- Total production duration for craftsman in minutes
def craftsmanWorkDuration : Nat := craftsmanEndTime - craftsmanStartTime

-- Total bracelets produced by craftsman
def totalBraceletsCraftsman : Nat := craftsmanWorkDuration * craftsmanProductionPerMinute

-- Time it takes for the assistant to produce the same number of bracelets
def assistantWorkDuration : Nat := totalBraceletsCraftsman / assistantProductionPerMinute

-- Time the assistant will stop working
def assistantEndTime : Nat := assistantStartTime + assistantWorkDuration

-- Convert time in minutes to hours and minutes format (output as a string for clarity)
def formatTime (timeInMinutes: Nat) : String :=
  let hours := timeInMinutes / 60
  let minutes := timeInMinutes % 60
  s! "{hours}:{if minutes < 10 then "0" else ""}{minutes}"

-- Proof goal: assistant will stop working at "13:30" (or 1:30 PM)
theorem assistant_stop_time : 
  formatTime assistantEndTime = "13:30" := 
by
  sorry

end assistant_stop_time_l771_77129


namespace cookies_left_l771_77127

def initial_cookies : ℕ := 93
def eaten_cookies : ℕ := 15

theorem cookies_left : initial_cookies - eaten_cookies = 78 := by
  sorry

end cookies_left_l771_77127


namespace distance_between_bus_stops_l771_77114

theorem distance_between_bus_stops (d : ℕ) (unit : String) 
  (h: d = 3000 ∧ unit = "meters") : unit = "C" := 
by 
  sorry

end distance_between_bus_stops_l771_77114


namespace tori_passing_question_l771_77154

def arithmetic_questions : ℕ := 20
def algebra_questions : ℕ := 40
def geometry_questions : ℕ := 40
def total_questions : ℕ := arithmetic_questions + algebra_questions + geometry_questions
def arithmetic_correct_pct : ℕ := 80
def algebra_correct_pct : ℕ := 50
def geometry_correct_pct : ℕ := 70
def passing_grade_pct : ℕ := 65

theorem tori_passing_question (questions_needed_to_pass : ℕ) (arithmetic_correct : ℕ) (algebra_correct : ℕ) (geometry_correct : ℕ) : 
  questions_needed_to_pass = 1 :=
by
  let arithmetic_correct : ℕ := (arithmetic_correct_pct * arithmetic_questions / 100)
  let algebra_correct : ℕ := (algebra_correct_pct * algebra_questions / 100)
  let geometry_correct : ℕ := (geometry_correct_pct * geometry_questions / 100)
  let total_correct : ℕ := arithmetic_correct + algebra_correct + geometry_correct
  let passing_grade : ℕ := (passing_grade_pct * total_questions / 100)
  let questions_needed_to_pass : ℕ := passing_grade - total_correct
  exact sorry

end tori_passing_question_l771_77154


namespace intersection_of_A_and_B_l771_77122

def A : Set ℝ := { x | x^2 - x > 0 }
def B : Set ℝ := { x | Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B : A ∩ B = { x | 1 < x ∧ x < 4 } :=
by sorry

end intersection_of_A_and_B_l771_77122


namespace jim_travel_distance_l771_77192

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end jim_travel_distance_l771_77192


namespace sum_of_four_interior_edges_l771_77151

-- Define the given conditions
def is_two_inch_frame (w : ℕ) := w = 2
def frame_area (A : ℕ) := A = 68
def outer_edge_length (L : ℕ) := L = 15

-- Define the inner dimensions calculation function
def inner_dimensions (outerL outerH frameW : ℕ) := 
  (outerL - 2 * frameW, outerH - 2 * frameW)

-- Define the final question in Lean 4 reflective of the equivalent proof problem
theorem sum_of_four_interior_edges (w A L y : ℕ) 
  (h1 : is_two_inch_frame w) 
  (h2 : frame_area A)
  (h3 : outer_edge_length L)
  (h4 : 15 * y - (15 - 2 * w) * (y - 2 * w) = A)
  : 2 * (15 - 2 * w) + 2 * (y - 2 * w) = 26 := 
sorry

end sum_of_four_interior_edges_l771_77151


namespace phone_price_in_october_l771_77179

variable (a : ℝ) (P_October : ℝ) (r : ℝ)

noncomputable def price_in_january := a
noncomputable def price_in_october (a : ℝ) (r : ℝ) := a * r^9

theorem phone_price_in_october :
  r = 0.97 →
  P_October = price_in_october a r →
  P_October = a * (0.97)^9 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end phone_price_in_october_l771_77179


namespace find_range_of_a_l771_77177

theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ∧ 
  ¬ ((∀ x : ℝ, x^2 - 2 * x > a) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)) → 
  a ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∪ Set.Ici (1:ℝ) :=
sorry

end find_range_of_a_l771_77177


namespace problem_statement_l771_77181

theorem problem_statement (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end problem_statement_l771_77181


namespace complex_division_l771_77131

-- Define i as the imaginary unit
def i : Complex := Complex.I

-- Define the problem statement to prove that 2i / (1 - i) equals -1 + i
theorem complex_division : (2 * i) / (1 - i) = -1 + i :=
by
  -- Since we are focusing on the statement, we use sorry to skip the proof
  sorry

end complex_division_l771_77131


namespace passes_after_6_l771_77183

-- Define the sequence a_n where a_n represents the number of ways the ball is in A's hands after n passes
def passes : ℕ → ℕ
| 0       => 1       -- Initially, the ball is in A's hands (1 way)
| (n + 1) => 2^n - passes n

-- Theorem to prove the number of different passing methods after 6 passes
theorem passes_after_6 : passes 6 = 22 := by
  sorry

end passes_after_6_l771_77183


namespace range_of_b_l771_77182

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem range_of_b (b : ℝ) : 
  (∃ (x1 x2 x3 : ℝ), f x1 = -b ∧ f x2 = -b ∧ f x3 = -b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ (-1 < b ∧ b < 0) :=
by
  sorry

end range_of_b_l771_77182


namespace bouquet_cost_l771_77128

theorem bouquet_cost (c : ℕ) : (c / 25 = 30 / 15) → c = 50 := by
  sorry

end bouquet_cost_l771_77128


namespace expression_not_equal_one_l771_77164

-- Definitions of the variables and the conditions
def a : ℝ := sorry  -- Non-zero real number a
def y : ℝ := sorry  -- Real number y

axiom h1 : a ≠ 0
axiom h2 : y ≠ a
axiom h3 : y ≠ -a

-- The main theorem statement
theorem expression_not_equal_one (h1 : a ≠ 0) (h2 : y ≠ a) (h3 : y ≠ -a) : 
  ( (a / (a - y) + y / (a + y)) / (y / (a - y) - a / (a + y)) ) ≠ 1 :=
sorry

end expression_not_equal_one_l771_77164


namespace triangle_inequality_l771_77133

variable (a b c p : ℝ)
variable (triangle : a + b > c ∧ a + c > b ∧ b + c > a)
variable (h_p : p = (a + b + c) / 2)

theorem triangle_inequality : 2 * Real.sqrt ((p - b) * (p - c)) ≤ a :=
sorry

end triangle_inequality_l771_77133


namespace problem_l771_77115

theorem problem (f : ℝ → ℝ) (h : ∀ x, (x - 3) * (deriv f x) ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := 
sorry

end problem_l771_77115


namespace profit_benny_wants_to_make_l771_77132

noncomputable def pumpkin_pies : ℕ := 10
noncomputable def cherry_pies : ℕ := 12
noncomputable def cost_pumpkin_pie : ℝ := 3
noncomputable def cost_cherry_pie : ℝ := 5
noncomputable def price_per_pie : ℝ := 5

theorem profit_benny_wants_to_make : 5 * (pumpkin_pies + cherry_pies) - (pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) = 20 :=
by
  sorry

end profit_benny_wants_to_make_l771_77132


namespace find_multiplier_l771_77141

theorem find_multiplier (x : ℕ) (h₁ : 3 * x = (26 - x) + 26) (h₂ : x = 13) : 3 = 3 := 
by 
  sorry

end find_multiplier_l771_77141


namespace rod_length_l771_77195

/--
Prove that given the number of pieces that can be cut from the rod is 40 and the length of each piece is 85 cm, the length of the rod is 3400 cm.
-/
theorem rod_length (number_of_pieces : ℕ) (length_of_each_piece : ℕ) (h_pieces : number_of_pieces = 40) (h_length_piece : length_of_each_piece = 85) : number_of_pieces * length_of_each_piece = 3400 := 
by
  -- We need to prove that 40 * 85 = 3400
  sorry

end rod_length_l771_77195


namespace sacks_per_day_proof_l771_77160

-- Definitions based on the conditions in the problem
def totalUnripeOranges : ℕ := 1080
def daysOfHarvest : ℕ := 45

-- Mathematical statement to prove
theorem sacks_per_day_proof : totalUnripeOranges / daysOfHarvest = 24 :=
by sorry

end sacks_per_day_proof_l771_77160


namespace employees_count_l771_77158

theorem employees_count (E M : ℝ) (h1 : M = 0.99 * E) (h2 : M - 299.9999999999997 = 0.98 * E) :
  E = 30000 :=
by sorry

end employees_count_l771_77158


namespace Toph_caught_12_fish_l771_77119

-- Define the number of fish each person caught
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def average_fish : ℕ := 8
def num_people : ℕ := 3

-- The total number of fish based on the average
def total_fish : ℕ := average_fish * num_people

-- Define the number of fish Toph caught
def Toph_fish : ℕ := total_fish - Aang_fish - Sokka_fish

-- Prove that Toph caught the correct number of fish
theorem Toph_caught_12_fish : Toph_fish = 12 := sorry

end Toph_caught_12_fish_l771_77119


namespace prism_volume_l771_77126

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end prism_volume_l771_77126


namespace total_pictures_painted_l771_77188

def pictures_painted_in_june : ℕ := 2
def pictures_painted_in_july : ℕ := 2
def pictures_painted_in_august : ℕ := 9

theorem total_pictures_painted : 
  pictures_painted_in_june + pictures_painted_in_july + pictures_painted_in_august = 13 :=
by
  sorry

end total_pictures_painted_l771_77188


namespace esteban_exercise_each_day_l771_77101

theorem esteban_exercise_each_day (natasha_daily : ℕ) (natasha_days : ℕ) (esteban_days : ℕ) (total_hours : ℕ) :
  let total_minutes := total_hours * 60
  let natasha_total := natasha_daily * natasha_days
  let esteban_total := total_minutes - natasha_total
  esteban_days ≠ 0 →
  natasha_daily = 30 →
  natasha_days = 7 →
  esteban_days = 9 →
  total_hours = 5 →
  esteban_total / esteban_days = 10 := 
by
  intros
  sorry

end esteban_exercise_each_day_l771_77101


namespace tangent_line_at_1_f_geq_x_minus_1_min_value_a_l771_77107

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- 1. Proof that the equation of the tangent line at the point (1, f(1)) is y = x - 1
theorem tangent_line_at_1 :
  ∃ k b, (k = 1 ∧ b = -1 ∧ (∀ x, (f x - k * x - b) = 0)) :=
sorry

-- 2. Proof that f(x) ≥ x - 1 for all x in (0, +∞)
theorem f_geq_x_minus_1 :
  ∀ x, 0 < x → f x ≥ x - 1 :=
sorry

-- 3. Proof that the minimum value of a such that f(x) ≥ ax² + 2/a for all x in (0, +∞) is -e³
theorem min_value_a :
  ∃ a, (∀ x, 0 < x → f x ≥ a * x^2 + 2 / a) ∧ (a = -Real.exp 3) :=
sorry

end tangent_line_at_1_f_geq_x_minus_1_min_value_a_l771_77107


namespace city_division_exists_l771_77100

-- Define the problem conditions and prove the required statement
theorem city_division_exists (squares : Type) (streets : squares → squares → Prop)
  (h_outgoing: ∀ (s : squares), ∃ t u : squares, streets s t ∧ streets s u) :
  ∃ (districts : squares → ℕ), (∀ (s t : squares), districts s ≠ districts t → streets s t ∨ streets t s) ∧
  (∀ (i j : ℕ), i ≠ j → ∀ (s t : squares), districts s = i → districts t = j → streets s t ∨ streets t s) ∧
  (∃ m : ℕ, m = 1014) :=
sorry

end city_division_exists_l771_77100


namespace expression_evaluation_l771_77198

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (2 * a + Real.sqrt 3) * (2 * a - Real.sqrt 3) - 3 * a * (a - 2) + 3 = -7 :=
by
  sorry

end expression_evaluation_l771_77198


namespace biff_break_even_hours_l771_77168

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end biff_break_even_hours_l771_77168


namespace certain_number_calculation_l771_77185

theorem certain_number_calculation (x : ℝ) (h : (15 * x) / 100 = 0.04863) : x = 0.3242 :=
by
  sorry

end certain_number_calculation_l771_77185


namespace combination_multiplication_and_addition_l771_77159

theorem combination_multiplication_and_addition :
  (Nat.choose 10 3) * (Nat.choose 8 3) + (Nat.choose 5 2) = 6730 :=
by
  sorry

end combination_multiplication_and_addition_l771_77159


namespace alpha_in_second_quadrant_l771_77137

theorem alpha_in_second_quadrant (α : ℝ) 
  (h1 : Real.sin α > Real.cos α)
  (h2 : Real.sin α * Real.cos α < 0) : 
  (Real.sin α > 0) ∧ (Real.cos α < 0) :=
by 
  -- Proof omitted
  sorry

end alpha_in_second_quadrant_l771_77137


namespace largest_unattainable_sum_l771_77134

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end largest_unattainable_sum_l771_77134


namespace largest_sum_fraction_l771_77145

theorem largest_sum_fraction :
  max (max (max (max ((1/3) + (1/2)) ((1/3) + (1/4))) ((1/3) + (1/5))) ((1/3) + (1/7))) ((1/3) + (1/9)) = 5/6 :=
by
  sorry

end largest_sum_fraction_l771_77145


namespace trigonometric_identity_l771_77165

theorem trigonometric_identity : 
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  -- Here we assume standard trigonometric identities and basic properties already handled by Mathlib
  sorry

end trigonometric_identity_l771_77165


namespace roots_quadratic_eq_k_l771_77153

theorem roots_quadratic_eq_k (k : ℝ) :
  (∀ x : ℝ, (5 * x^2 + 20 * x + k = 0) ↔ (x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10)) →
  k = 17 := by
  intro h
  sorry

end roots_quadratic_eq_k_l771_77153


namespace find_maximum_value_l771_77191

open Real

noncomputable def maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : ℝ :=
  2 + sqrt 5

theorem find_maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) > maximum_value a b c h₁ h₂ h₃ h₄ :=
by
  sorry

end find_maximum_value_l771_77191


namespace five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l771_77184

noncomputable def count_five_digit_numbers_greater_21035_and_even : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_greater_21035_and_even_correct :
  count_five_digit_numbers_greater_21035_and_even = 39 :=
  sorry

noncomputable def count_five_digit_numbers_even_with_odd_positions : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_even_with_odd_positions_correct :
  count_five_digit_numbers_even_with_odd_positions = 8 :=
  sorry

end five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l771_77184


namespace repeating_decimal_transform_l771_77180

theorem repeating_decimal_transform (n : ℕ) (s : String) (k : ℕ) (m : ℕ)
  (original : s = "2345678") (len : k = 7) (position : n = 2011)
  (effective_position : m = n - 1) (mod_position : m % k = 3) :
  "0.1" ++ s = "0.12345678" :=
sorry

end repeating_decimal_transform_l771_77180


namespace crayons_left_is_4_l771_77121

-- Define initial number of crayons in the drawer
def initial_crayons : Nat := 7

-- Define number of crayons Mary took out
def taken_by_mary : Nat := 3

-- Define the number of crayons left in the drawer
def crayons_left (initial : Nat) (taken : Nat) : Nat :=
  initial - taken

-- Prove the number of crayons left in the drawer is 4
theorem crayons_left_is_4 : crayons_left initial_crayons taken_by_mary = 4 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end crayons_left_is_4_l771_77121


namespace monotonic_increasing_range_l771_77104

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (x + a) / x

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → (∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 < x2 → f x1 a ≤ f x2 a)) ↔ -4 ≤ a ∧ a ≤ 0 :=
sorry

end monotonic_increasing_range_l771_77104


namespace h_evaluation_l771_77120

variables {a b c : ℝ}

-- Definitions and conditions
def p (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c
def h (x : ℝ) : ℝ := sorry -- Definition of h(x) in terms of the roots of p(x)

theorem h_evaluation (ha : a < b) (hb : b < c) : h 2 = (2 + 2 * a + 3 * b + c) / (c^2) :=
sorry

end h_evaluation_l771_77120


namespace simplify_div_expression_evaluate_at_2_l771_77152

variable (a : ℝ)

theorem simplify_div_expression (h0 : a ≠ 0) (h1 : a ≠ 1) :
  (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) :=
by
  sorry

theorem evaluate_at_2 : (1 - 1 / 2) / ((2^2 - 2 * 2 + 1) / 2) = 1 :=
by 
  sorry

end simplify_div_expression_evaluate_at_2_l771_77152


namespace unique_solution_f_l771_77186

theorem unique_solution_f (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y))
  (h2 : f 0 = 0) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_l771_77186


namespace num_perfect_squares_in_range_l771_77194

-- Define the range for the perfect squares
def lower_bound := 75
def upper_bound := 400

-- Define the smallest integer whose square is greater than lower_bound
def lower_int := 9

-- Define the largest integer whose square is less than or equal to upper_bound
def upper_int := 20

-- State the proof problem
theorem num_perfect_squares_in_range : 
  (upper_int - lower_int + 1) = 12 :=
by
  -- Skipping the proof
  sorry

end num_perfect_squares_in_range_l771_77194


namespace product_of_five_integers_l771_77155

theorem product_of_five_integers (E F G H I : ℚ)
  (h1 : E + F + G + H + I = 110)
  (h2 : E / 2 = F / 3 ∧ F / 3 = G * 4 ∧ G * 4 = H * 2 ∧ H * 2 = I - 5) :
  E * F * G * H * I = 623400000 / 371293 := by
  sorry

end product_of_five_integers_l771_77155


namespace number_of_possible_values_l771_77105

-- Define the decimal number s and its representation
def s (e f g h : ℕ) : ℚ := e / 10 + f / 100 + g / 1000 + h / 10000

-- Define the condition that the closest fraction is 2/9
def closest_to_2_9 (s : ℚ) : Prop :=
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 1 / 6)) ∧
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 2 / 11))

-- The main theorem stating the number of possible values for s
theorem number_of_possible_values :
  (∃ e f g h : ℕ, 0 ≤ e ∧ e ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9 ∧ 0 ≤ g ∧ g ≤ 9 ∧ 0 ≤ h ∧ h ≤ 9 ∧
    closest_to_2_9 (s e f g h)) → (∃ n : ℕ, n = 169) :=
by
  sorry

end number_of_possible_values_l771_77105


namespace speed_of_first_plane_l771_77166

theorem speed_of_first_plane
  (v : ℕ)
  (travel_time : ℚ := 44 / 11)
  (relative_speed : ℚ := v + 90)
  (distance : ℚ := 800) :
  (relative_speed * travel_time = distance) → v = 110 :=
by
  sorry

end speed_of_first_plane_l771_77166


namespace circle_diameter_l771_77199

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end circle_diameter_l771_77199


namespace tan_subtraction_l771_77146

theorem tan_subtraction (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 :=
by
  sorry

end tan_subtraction_l771_77146


namespace sufficient_but_not_necessary_condition_l771_77130

theorem sufficient_but_not_necessary_condition (x : ℝ) : (0 < x ∧ x < 5) → |x - 2| < 3 :=
by
  sorry

end sufficient_but_not_necessary_condition_l771_77130


namespace solution_interval_log_eq_l771_77139

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) + x - 3

theorem solution_interval_log_eq (h_mono : ∀ x y, (0 < x ∧ x < y) → f x < f y)
  (h_f2 : f 2 = 0)
  (h_f3 : f 3 > 0) :
  ∃ x, (2 ≤ x ∧ x < 3 ∧ f x = 0) :=
by
  sorry

end solution_interval_log_eq_l771_77139


namespace linear_function_change_l771_77142

-- Define a linear function g
variable (g : ℝ → ℝ)

-- Define and assume the conditions
def linear_function (g : ℝ → ℝ) : Prop := ∀ x y, g (x + y) = g x + g y ∧ g (x - y) = g x - g y
def condition_g_at_points : Prop := g 3 - g (-1) = 20

-- Prove that g(10) - g(2) = 40
theorem linear_function_change (g : ℝ → ℝ) 
  (linear_g : linear_function g) 
  (cond_g : condition_g_at_points g) : 
  g 10 - g 2 = 40 :=
sorry

end linear_function_change_l771_77142


namespace painting_area_l771_77174

theorem painting_area (c t A : ℕ) (h1 : c = 15) (h2 : t = 840) (h3 : c * A = t) : A = 56 := 
by
  sorry -- proof to demonstrate A = 56

end painting_area_l771_77174


namespace cos420_add_sin330_l771_77113

theorem cos420_add_sin330 : Real.cos (420 * Real.pi / 180) + Real.sin (330 * Real.pi / 180) = 0 := 
by
  sorry

end cos420_add_sin330_l771_77113


namespace alcohol_solution_problem_l771_77157

theorem alcohol_solution_problem (x_vol y_vol : ℚ) (x_alcohol y_alcohol target_alcohol : ℚ) (target_vol : ℚ) :
  x_vol = 250 ∧ x_alcohol = 10/100 ∧ y_alcohol = 30/100 ∧ target_alcohol = 25/100 ∧ target_vol = 250 + y_vol →
  (x_alcohol * x_vol + y_alcohol * y_vol = target_alcohol * target_vol) →
  y_vol = 750 :=
by
  sorry

end alcohol_solution_problem_l771_77157


namespace determine_x_l771_77103

variable {x y : ℝ}

theorem determine_x (h : (x - 1) / x = (y^3 + 3 * y^2 - 4) / (y^3 + 3 * y^2 - 5)) : 
  x = y^3 + 3 * y^2 - 5 := 
sorry

end determine_x_l771_77103


namespace area_triangle_PQR_eq_2sqrt2_l771_77170

noncomputable def areaOfTrianglePQR : ℝ :=
  let sideAB := 3
  let altitudeAE := 6
  let EB := Real.sqrt (sideAB^2 + altitudeAE^2)
  let ED := EB
  let EC := Real.sqrt ((sideAB * Real.sqrt 2)^2 + altitudeAE^2)
  let EP := (2 / 3) * EB
  let EQ := EP
  let ER := (1 / 3) * EC
  let PR := Real.sqrt (ER^2 + EP^2 - 2 * ER * EP * (EB^2 + EC^2 - sideAB^2) / (2 * EB * EC))
  let PQ := 2
  let RS := Real.sqrt (PR^2 - (PQ / 2)^2)
  (1 / 2) * PQ * RS

theorem area_triangle_PQR_eq_2sqrt2 : areaOfTrianglePQR = 2 * Real.sqrt 2 :=
  sorry

end area_triangle_PQR_eq_2sqrt2_l771_77170


namespace copper_production_is_correct_l771_77124

-- Define the percentages of copper production for each mine
def percentage_copper_mine_a : ℝ := 0.05
def percentage_copper_mine_b : ℝ := 0.10
def percentage_copper_mine_c : ℝ := 0.15

-- Define the daily production of each mine in tons
def daily_production_mine_a : ℕ := 3000
def daily_production_mine_b : ℕ := 4000
def daily_production_mine_c : ℕ := 3500

-- Define the total copper produced from all mines
def total_copper_produced : ℝ :=
  percentage_copper_mine_a * daily_production_mine_a +
  percentage_copper_mine_b * daily_production_mine_b +
  percentage_copper_mine_c * daily_production_mine_c

-- Prove that the total daily copper production is 1075 tons
theorem copper_production_is_correct :
  total_copper_produced = 1075 := 
sorry

end copper_production_is_correct_l771_77124


namespace circle_k_range_l771_77112

theorem circle_k_range {k : ℝ}
  (h : ∀ x y : ℝ, x^2 + y^2 - 2*x + y + k = 0) :
  k < 5 / 4 :=
sorry

end circle_k_range_l771_77112


namespace part1_part2_1_part2_2_l771_77117

theorem part1 (n : ℚ) :
  (2 / 2 + n / 5 = (2 + n) / 7) → n = -25 / 2 :=
by sorry

theorem part2_1 (m n : ℚ) :
  (m / 2 + n / 5 = (m + n) / 7) → m = -4 / 25 * n :=
by sorry

theorem part2_2 (m n: ℚ) :
  (m = -4 / 25 * n) → (25 * m + n = 6) → (m = 8 / 25 ∧ n = -2) :=
by sorry

end part1_part2_1_part2_2_l771_77117


namespace find_rate_of_current_l771_77173

-- Parameters and definitions
variables (r w : Real)

-- Conditions of the problem
def original_journey := 3 * r^2 - 23 * w^2 = 0
def modified_journey := 6 * r^2 - 2 * w^2 + 40 * w = 0

-- Main theorem to prove
theorem find_rate_of_current (h1 : original_journey r w) (h2 : modified_journey r w) :
  w = 10 / 11 :=
sorry

end find_rate_of_current_l771_77173


namespace side_length_of_cube_l771_77178

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l771_77178


namespace evaluate_expression_l771_77135

noncomputable def w := Complex.exp (2 * Real.pi * Complex.I / 11)

theorem evaluate_expression : (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) = 88573 := 
by 
  sorry

end evaluate_expression_l771_77135


namespace measure_of_RPS_l771_77102

-- Assume the elements of the problem
variables {Q R P S : Type}

-- Angles in degrees
def angle_PQS := 35
def angle_QPR := 80
def angle_PSQ := 40

-- Define the angles and the straight line condition
def QRS_straight_line : Prop := true  -- This definition is trivial for a straight line

-- Measure of angle QPS using sum of angles in triangle
noncomputable def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Measure of angle RPS derived from the previous steps
noncomputable def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The statement of the problem in Lean
theorem measure_of_RPS : angle_RPS = 25 := by
  sorry

end measure_of_RPS_l771_77102


namespace g_at_3_l771_77176

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_at_3 : g 3 = 147 :=
by
  -- Proof omitted for brevity
  sorry

end g_at_3_l771_77176


namespace directrix_of_parabola_l771_77148

theorem directrix_of_parabola :
  ∀ (a h k : ℝ), (a < 0) → (∀ x, y = a * (x - h) ^ 2 + k) → (h = 0) → (k = 0) → 
  (directrix = 1 / (4 * a)) → (directrix = 1 / 4) :=
by
  sorry

end directrix_of_parabola_l771_77148


namespace germs_per_dish_l771_77197

/--
Given:
- the total number of germs is \(5.4 \times 10^6\),
- the number of petri dishes is 10,800,

Prove:
- the number of germs per dish is 500.
-/
theorem germs_per_dish (total_germs : ℝ) (petri_dishes: ℕ) (h₁: total_germs = 5.4 * 10^6) (h₂: petri_dishes = 10800) :
  (total_germs / petri_dishes = 500) :=
sorry

end germs_per_dish_l771_77197


namespace train_overtake_l771_77169

theorem train_overtake :
  let speedA := 30 -- speed of Train A in miles per hour
  let speedB := 38 -- speed of Train B in miles per hour
  let lead_timeA := 2 -- lead time of Train A in hours
  let distanceA := speedA * lead_timeA -- distance traveled by Train A in the lead time
  let t := 7.5 -- time in hours Train B travels to catch up Train A
  let total_distanceB := speedB * t -- total distance traveled by Train B in time t
  total_distanceB = 285 := 
by
  sorry

end train_overtake_l771_77169


namespace find_n_l771_77162

theorem find_n (n : ℕ) (a_n D_n d_n : ℕ) (h1 : n > 5) (h2 : D_n - d_n = a_n) : n = 9 := 
by 
  sorry

end find_n_l771_77162


namespace millennium_run_time_l771_77149

theorem millennium_run_time (M A B : ℕ) (h1 : B = 100) (h2 : B = A + 10) (h3 : A = M - 30) : M = 120 := by
  sorry

end millennium_run_time_l771_77149


namespace boat_speed_in_still_water_l771_77106

-- Definitions of the conditions
def with_stream_speed : ℝ := 36
def against_stream_speed : ℝ := 8

-- Let Vb be the speed of the boat in still water, and Vs be the speed of the stream.
variable (Vb Vs : ℝ)

-- Conditions given in the problem
axiom h1 : Vb + Vs = with_stream_speed
axiom h2 : Vb - Vs = against_stream_speed

-- The statement to prove: the speed of the boat in still water is 22 km/h.
theorem boat_speed_in_still_water : Vb = 22 := by
  sorry

end boat_speed_in_still_water_l771_77106


namespace value_of_x_l771_77196

variable (w x y : ℝ)

theorem value_of_x 
  (h_avg : (w + x) / 2 = 0.5)
  (h_eq : (7 / w) + (7 / x) = 7 / y)
  (h_prod : w * x = y) :
  x = 0.5 :=
sorry

end value_of_x_l771_77196


namespace probability_same_carriage_l771_77172

theorem probability_same_carriage (num_carriages num_people : ℕ) (h1 : num_carriages = 10) (h2 : num_people = 3) : 
  ∃ p : ℚ, p = 7/25 ∧ p = 1 - (10 * 9 * 8) / (10^3) :=
by
  sorry

end probability_same_carriage_l771_77172


namespace find_tents_l771_77123

theorem find_tents (x y : ℕ) (hx : x + y = 600) (hy : 1700 * x + 1300 * y = 940000) : x = 400 ∧ y = 200 :=
by
  sorry

end find_tents_l771_77123


namespace fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l771_77171

-- Conditions
def square_side : ℕ := 1
def area_per_square : ℕ := square_side * square_side
def area_of_stair (n : ℕ) : ℕ := (n * (n + 1)) / 2
def perimeter_of_stair (n : ℕ) : ℕ := 4 * n

-- Part (a)
theorem fifth_stair_area_and_perimeter :
  area_of_stair 5 = 15 ∧ perimeter_of_stair 5 = 20 := by
  sorry

-- Part (b)
theorem stair_for_area_78 :
  ∃ n, area_of_stair n = 78 ∧ n = 12 := by
  sorry

-- Part (c)
theorem stair_for_perimeter_100 :
  ∃ n, perimeter_of_stair n = 100 ∧ n = 25 := by
  sorry

end fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l771_77171


namespace expand_product_l771_77193

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end expand_product_l771_77193


namespace father_current_age_l771_77187

namespace AgeProof

def daughter_age : ℕ := 10
def years_future : ℕ := 20

def father_age (D : ℕ) : ℕ := 4 * D

theorem father_current_age :
  ∀ D : ℕ, ∀ F : ℕ, (F = father_age D) →
  (F + years_future = 2 * (D + years_future)) →
  D = daughter_age →
  F = 40 :=
by
  intro D F h1 h2 h3
  sorry

end AgeProof

end father_current_age_l771_77187


namespace flour_per_cake_l771_77144

theorem flour_per_cake (traci_flour harris_flour : ℕ) (cakes_each : ℕ)
  (h_traci_flour : traci_flour = 500)
  (h_harris_flour : harris_flour = 400)
  (h_cakes_each : cakes_each = 9) :
  (traci_flour + harris_flour) / (2 * cakes_each) = 50 := by
  sorry

end flour_per_cake_l771_77144


namespace unripe_oranges_per_day_l771_77147

/-
Problem: Prove that if after 6 days, they will have 390 sacks of unripe oranges, then the number of sacks of unripe oranges harvested per day is 65.
-/

theorem unripe_oranges_per_day (total_sacks : ℕ) (days : ℕ) (harvest_per_day : ℕ)
  (h1 : days = 6)
  (h2 : total_sacks = 390)
  (h3 : harvest_per_day = total_sacks / days) :
  harvest_per_day = 65 :=
by
  sorry

end unripe_oranges_per_day_l771_77147


namespace smallest_four_digit_divisible_by_53_l771_77156

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l771_77156


namespace relationship_between_xyz_l771_77175

theorem relationship_between_xyz (x y z : ℝ) (h1 : x - z < y) (h2 : x + z > y) : -z < x - y ∧ x - y < z :=
by
  sorry

end relationship_between_xyz_l771_77175


namespace ratio_Sarah_to_Eli_is_2_l771_77116

variable (Kaylin_age : ℕ := 33)
variable (Freyja_age : ℕ := 10)
variable (Eli_age : ℕ := Freyja_age + 9)
variable (Sarah_age : ℕ := Kaylin_age + 5)

theorem ratio_Sarah_to_Eli_is_2 : (Sarah_age : ℚ) / Eli_age = 2 := 
by 
  -- Proof would go here
  sorry

end ratio_Sarah_to_Eli_is_2_l771_77116


namespace ratio_of_jumps_l771_77189

theorem ratio_of_jumps (run_ric: ℕ) (jump_ric: ℕ) (run_mar: ℕ) (extra_dist: ℕ)
    (h1 : run_ric = 20)
    (h2 : jump_ric = 4)
    (h3 : run_mar = 18)
    (h4 : extra_dist = 1) :
    (run_mar + extra_dist - run_ric - jump_ric) / jump_ric = 7 / 4 :=
by
  sorry

end ratio_of_jumps_l771_77189


namespace solve_inequality_l771_77118

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem solve_inequality {x : ℝ} (hx : 0 < x) : 
  f (Real.log x / Real.log 2) < f 2 ↔ (0 < x ∧ x < 1) ∨ (4 < x) :=
by
sorry

end solve_inequality_l771_77118


namespace not_perfect_square_4n_squared_plus_4n_plus_4_l771_77108

theorem not_perfect_square_4n_squared_plus_4n_plus_4 :
  ¬ ∃ m n : ℕ, m^2 = 4 * n^2 + 4 * n + 4 := 
by
  sorry

end not_perfect_square_4n_squared_plus_4n_plus_4_l771_77108


namespace novelists_count_l771_77161

theorem novelists_count (n p : ℕ) (h1 : n / (n + p) = 5 / 8) (h2 : n + p = 24) : n = 15 :=
sorry

end novelists_count_l771_77161


namespace line_through_points_l771_77125

theorem line_through_points (a b : ℝ)
  (h1 : 2 = a * 1 + b)
  (h2 : 14 = a * 5 + b) :
  a - b = 4 := 
  sorry

end line_through_points_l771_77125


namespace distinct_digit_sum_equation_l771_77167

theorem distinct_digit_sum_equation :
  ∃ (F O R T Y S I X : ℕ), 
    F ≠ O ∧ F ≠ R ∧ F ≠ T ∧ F ≠ Y ∧ F ≠ S ∧ F ≠ I ∧ F ≠ X ∧ 
    O ≠ R ∧ O ≠ T ∧ O ≠ Y ∧ O ≠ S ∧ O ≠ I ∧ O ≠ X ∧ 
    R ≠ T ∧ R ≠ Y ∧ R ≠ S ∧ R ≠ I ∧ R ≠ X ∧ 
    T ≠ Y ∧ T ≠ S ∧ T ≠ I ∧ T ≠ X ∧ 
    Y ≠ S ∧ Y ≠ I ∧ Y ≠ X ∧ 
    S ≠ I ∧ S ≠ X ∧ 
    I ≠ X ∧ 
    FORTY = 10000 * F + 1000 * O + 100 * R + 10 * T + Y ∧ 
    TEN = 100 * T + 10 * E + N ∧ 
    SIXTY = 10000 * S + 1000 * I + 100 * X + 10 * T + Y ∧ 
    FORTY + TEN + TEN = SIXTY ∧ 
    SIXTY = 31486 :=
sorry

end distinct_digit_sum_equation_l771_77167


namespace rebus_solution_l771_77109

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end rebus_solution_l771_77109


namespace alexei_loss_per_week_l771_77136

-- Definitions
def aleesia_loss_per_week : ℝ := 1.5
def aleesia_total_weeks : ℕ := 10
def total_loss : ℝ := 35
def alexei_total_weeks : ℕ := 8

-- The statement to prove
theorem alexei_loss_per_week :
  (total_loss - aleesia_loss_per_week * aleesia_total_weeks) / alexei_total_weeks = 2.5 := 
by sorry

end alexei_loss_per_week_l771_77136
