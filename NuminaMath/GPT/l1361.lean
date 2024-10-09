import Mathlib

namespace value_division_l1361_136126

theorem value_division (x : ℝ) (h1 : 54 / x = 54 - 36) : x = 3 := by
  sorry

end value_division_l1361_136126


namespace value_of_a3_a6_a9_l1361_136186

variable (a : ℕ → ℤ) -- Define the sequence a as a function from natural numbers to integers
variable (d : ℤ) -- Define the common difference d as an integer

-- Conditions
axiom h1 : a 1 + a 4 + a 7 = 39
axiom h2 : a 2 + a 5 + a 8 = 33
axiom h3 : ∀ n : ℕ, a (n+1) = a n + d -- This condition ensures the sequence is arithmetic

-- Theorem: We need to prove the value of a_3 + a_6 + a_9 is 27
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 27 :=
by
  sorry

end value_of_a3_a6_a9_l1361_136186


namespace prove_dollar_op_l1361_136153

variable {a b x y : ℝ}

def dollar_op (a b : ℝ) : ℝ := (a - b) ^ 2

theorem prove_dollar_op :
  dollar_op (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) := by
  sorry

end prove_dollar_op_l1361_136153


namespace bryan_books_l1361_136139

theorem bryan_books (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := 
by 
  sorry

end bryan_books_l1361_136139


namespace ratio_of_weights_l1361_136182

variable (x : ℝ)

-- Conditions as definitions in Lean 4
def seth_loss : ℝ := 17.5
def jerome_loss : ℝ := 17.5 * x
def veronica_loss : ℝ := 17.5 + 1.5 -- 19 pounds
def total_loss : ℝ := seth_loss + jerome_loss x + veronica_loss

-- Statement to prove
theorem ratio_of_weights (h : total_loss x = 89) : jerome_loss x / seth_loss = 3 :=
by sorry

end ratio_of_weights_l1361_136182


namespace intersection_correct_l1361_136107

-- Conditions
def M : Set ℤ := { -1, 0, 1, 3, 5 }
def N : Set ℤ := { -2, 1, 2, 3, 5 }

-- Statement to prove
theorem intersection_correct : M ∩ N = { 1, 3, 5 } :=
by
  sorry

end intersection_correct_l1361_136107


namespace real_solutions_quadratic_l1361_136171

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end real_solutions_quadratic_l1361_136171


namespace nature_of_roots_indeterminate_l1361_136123

variable (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nature_of_roots_indeterminate (h : b^2 - 4 * a * c = 0) : 
  ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) = 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) < 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) > 0) :=
sorry

end nature_of_roots_indeterminate_l1361_136123


namespace women_to_total_population_ratio_l1361_136173

/-- original population of Salem -/
def original_population (pop_leesburg : ℕ) : ℕ := 15 * pop_leesburg

/-- new population after people moved out -/
def new_population (orig_pop : ℕ) (moved_out : ℕ) : ℕ := orig_pop - moved_out

/-- ratio of two numbers -/
def ratio (num : ℕ) (denom : ℕ) : ℚ := num / denom

/-- population data -/
structure PopulationData :=
  (pop_leesburg : ℕ)
  (moved_out : ℕ)
  (women : ℕ)

/-- prove ratio of women to the total population in Salem -/
theorem women_to_total_population_ratio (data : PopulationData)
  (pop_leesburg_eq : data.pop_leesburg = 58940)
  (moved_out_eq : data.moved_out = 130000)
  (women_eq : data.women = 377050) : 
  ratio data.women (new_population (original_population data.pop_leesburg) data.moved_out) = 377050 / 754100 :=
by
  sorry

end women_to_total_population_ratio_l1361_136173


namespace system_of_equations_solution_l1361_136179

theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y + 2 * x * y = 11 ∧ 2 * x^2 * y + x * y^2 = 15) ↔
  ((x = 1/2 ∧ y = 5) ∨ (x = 1 ∧ y = 3) ∨ (x = 3/2 ∧ y = 2) ∨ (x = 5/2 ∧ y = 1)) :=
by 
  sorry

end system_of_equations_solution_l1361_136179


namespace certain_number_is_correct_l1361_136152

theorem certain_number_is_correct (x : ℝ) (h : x / 1.45 = 17.5) : x = 25.375 :=
sorry

end certain_number_is_correct_l1361_136152


namespace quadratic_has_two_zeros_l1361_136125

theorem quadratic_has_two_zeros {a b c : ℝ} (h : a * c < 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by
  sorry

end quadratic_has_two_zeros_l1361_136125


namespace total_tickets_sold_l1361_136120

theorem total_tickets_sold 
  (ticket_price : ℕ) 
  (discount_40_percent : ℕ → ℕ) 
  (discount_15_percent : ℕ → ℕ) 
  (revenue : ℕ) 
  (people_10_discount_40 : ℕ) 
  (people_20_discount_15 : ℕ) 
  (people_full_price : ℕ)
  (h_ticket_price : ticket_price = 20)
  (h_discount_40 : ∀ n, discount_40_percent n = n * 12)
  (h_discount_15 : ∀ n, discount_15_percent n = n * 17)
  (h_revenue : revenue = 760)
  (h_people_10_discount_40 : people_10_discount_40 = 10)
  (h_people_20_discount_15 : people_20_discount_15 = 20)
  (h_people_full_price : people_full_price * ticket_price = 300) :
  (people_10_discount_40 + people_20_discount_15 + people_full_price = 45) :=
by
  sorry

end total_tickets_sold_l1361_136120


namespace at_least_100_arcs_of_21_points_l1361_136142

noncomputable def count_arcs (n : ℕ) (θ : ℝ) : ℕ :=
-- Please note this function needs to be defined appropriately, here we assume it computes the number of arcs of θ degrees or fewer between n points on a circle
sorry

theorem at_least_100_arcs_of_21_points :
  ∃ (n : ℕ), n = 21 ∧ count_arcs n (120 : ℝ) ≥ 100 :=
sorry

end at_least_100_arcs_of_21_points_l1361_136142


namespace determine_a_value_l1361_136187

theorem determine_a_value (a : ℝ) :
  (∀ y₁ y₂ : ℝ, ∃ m₁ m₂ : ℝ, (m₁, y₁) = (a, -2) ∧ (m₂, y₂) = (3, -4) ∧ (m₁ = m₂)) → a = 3 :=
by
  sorry

end determine_a_value_l1361_136187


namespace total_age_in_3_years_l1361_136194

theorem total_age_in_3_years (Sam Sue Kendra : ℕ)
  (h1 : Kendra = 18)
  (h2 : Kendra = 3 * Sam)
  (h3 : Sam = 2 * Sue) :
  Sam + Sue + Kendra + 3 * 3 = 36 :=
by
  sorry

end total_age_in_3_years_l1361_136194


namespace mady_balls_2010th_step_l1361_136183

theorem mady_balls_2010th_step :
  let base_5_digits (n : Nat) : List Nat := (Nat.digits 5 n)
  (base_5_digits 2010).sum = 6 := by
  sorry

end mady_balls_2010th_step_l1361_136183


namespace range_of_a_l1361_136113

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, a * x^2 + a * x + 1 > 0) : a ∈ Set.Icc 0 4 :=
sorry

end range_of_a_l1361_136113


namespace friends_total_candies_l1361_136138

noncomputable def total_candies (T S J C V B : ℕ) : ℕ :=
  T + S + J + C + V + B

theorem friends_total_candies :
  let T := 22
  let S := 16
  let J := T / 2
  let C := 2 * S
  let V := J + S
  let B := (T + C) / 2 + 9
  total_candies T S J C V B = 144 := by
  sorry

end friends_total_candies_l1361_136138


namespace volume_of_prism_l1361_136143

variables (a b : ℝ) (α β : ℝ)
  (h1 : a > b)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)

noncomputable def volume_prism : ℝ :=
  (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β

theorem volume_of_prism (a b α β : ℝ) (h1 : a > b) (h2 : 0 < α ∧ α < π / 2) (h3 : 0 < β ∧ β < π / 2) :
  volume_prism a b α β = (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β := by
  sorry

end volume_of_prism_l1361_136143


namespace complex_number_solution_l1361_136154

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (hz : z * (i - 1) = 2 * i) : 
z = 1 - i :=
by 
  sorry

end complex_number_solution_l1361_136154


namespace total_number_of_squares_l1361_136178

theorem total_number_of_squares (n : ℕ) (h : n = 12) : 
  ∃ t, t = 17 :=
by
  -- The proof is omitted here
  sorry

end total_number_of_squares_l1361_136178


namespace subcommittee_combinations_l1361_136144

open Nat

theorem subcommittee_combinations :
  (choose 8 3) * (choose 6 2) = 840 := by
  sorry

end subcommittee_combinations_l1361_136144


namespace oplus_self_twice_l1361_136112

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end oplus_self_twice_l1361_136112


namespace jorge_goals_this_season_l1361_136170

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end jorge_goals_this_season_l1361_136170


namespace volume_of_rect_box_l1361_136195

open Real

/-- Proof of the volume of a rectangular box given its face areas -/
theorem volume_of_rect_box (l w h : ℝ) 
  (A1 : l * w = 40) 
  (A2 : w * h = 10) 
  (A3 : l * h = 8) : 
  l * w * h = 40 * sqrt 2 :=
by
  sorry

end volume_of_rect_box_l1361_136195


namespace total_weight_30_l1361_136169

-- Definitions of initial weights and ratio conditions
variables (a b : ℕ)
def initial_weights (h1 : a = 4 * b) : Prop := True

-- Definitions of transferred weights
def transferred_weights (a' b' : ℕ) (h2 : a' = a - 10) (h3 : b' = b + 10) : Prop := True

-- Definition of the new ratio condition
def new_ratio (a' b' : ℕ) (h4 : 8 * a' = 7 * b') : Prop := True

-- The final proof statement
theorem total_weight_30 (a b a' b' : ℕ)
    (h1 : a = 4 * b) 
    (h2 : a' = a - 10) 
    (h3 : b' = b + 10)
    (h4 : 8 * a' = 7 * b') : a + b = 30 := 
    sorry

end total_weight_30_l1361_136169


namespace find_s_l1361_136168

theorem find_s (a b r1 r2 : ℝ) (h1 : r1 + r2 = -a) (h2 : r1 * r2 = b) :
    let new_root1 := (r1 + r2) * (r1 + r2)
    let new_root2 := (r1 * r2) * (r1 + r2)
    let s := b * a - a * a
    s = ab - a^2 :=
  by
    -- the proof goes here
    sorry

end find_s_l1361_136168


namespace smallest_n_in_T_and_largest_N_not_in_T_l1361_136129

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3 * x + 4) / (x + 3)}

theorem smallest_n_in_T_and_largest_N_not_in_T :
  (∀ n, n = 4 / 3 → n ∈ T) ∧ (∀ N, N = 3 → N ∉ T) :=
by
  sorry

end smallest_n_in_T_and_largest_N_not_in_T_l1361_136129


namespace factorize_polynomial_l1361_136118

theorem factorize_polynomial {x : ℝ} : x^3 + 2 * x^2 - 3 * x = x * (x + 3) * (x - 1) :=
by sorry

end factorize_polynomial_l1361_136118


namespace total_value_is_correct_l1361_136146

-- Define the conditions from the problem
def totalCoins : Nat := 324
def twentyPaiseCoins : Nat := 220
def twentyPaiseValue : Nat := 20
def twentyFivePaiseValue : Nat := 25
def paiseToRupees : Nat := 100

-- Calculate the number of 25 paise coins
def twentyFivePaiseCoins : Nat := totalCoins - twentyPaiseCoins

-- Calculate the total value of 20 paise and 25 paise coins in paise
def totalValueInPaise : Nat :=
  (twentyPaiseCoins * twentyPaiseValue) + 
  (twentyFivePaiseCoins * twentyFivePaiseValue)

-- Convert the total value from paise to rupees
def totalValueInRupees : Nat := totalValueInPaise / paiseToRupees

-- The theorem to be proved
theorem total_value_is_correct : totalValueInRupees = 70 := by
  sorry

end total_value_is_correct_l1361_136146


namespace steve_average_speed_l1361_136136

theorem steve_average_speed 
  (Speed1 Time1 Speed2 Time2 : ℝ) 
  (cond1 : Speed1 = 40) 
  (cond2 : Time1 = 5)
  (cond3 : Speed2 = 80) 
  (cond4 : Time2 = 3) 
: 
(Speed1 * Time1 + Speed2 * Time2) / (Time1 + Time2) = 55 := 
sorry

end steve_average_speed_l1361_136136


namespace sin_identity_l1361_136190

theorem sin_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + π / 4) ^ 2 = 5 / 6 := 
sorry

end sin_identity_l1361_136190


namespace remainder_24_l1361_136119

-- Statement of the problem in Lean 4
theorem remainder_24 (y : ℤ) (h : y % 288 = 45) : y % 24 = 21 :=
by
  sorry

end remainder_24_l1361_136119


namespace gain_amount_is_ten_l1361_136104

theorem gain_amount_is_ten (S : ℝ) (C : ℝ) (g : ℝ) (G : ℝ) 
  (h1 : S = 110) (h2 : g = 0.10) (h3 : S = C + g * C) : G = 10 :=
by 
  sorry

end gain_amount_is_ten_l1361_136104


namespace ratio_of_radii_l1361_136167

open Real

theorem ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 5 * π * a^2) : a / b = 1 / sqrt 6 :=
by
  sorry

end ratio_of_radii_l1361_136167


namespace people_distribution_l1361_136157

theorem people_distribution (x : ℕ) (h1 : x > 5):
  100 / (x - 5) = 150 / x :=
sorry

end people_distribution_l1361_136157


namespace art_class_students_not_in_science_l1361_136176

theorem art_class_students_not_in_science (n S A S_inter_A_only_A : ℕ) 
  (h_n : n = 120) 
  (h_S : S = 85) 
  (h_A : A = 65) 
  (h_union: n = S + A - S_inter_A_only_A) : 
  S_inter_A_only_A = 30 → 
  A - S_inter_A_only_A = 35 :=
by
  intros h
  rw [h]
  sorry

end art_class_students_not_in_science_l1361_136176


namespace pears_left_l1361_136191

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) : 
  jason_pears + keith_pears - mike_ate = 81 := 
by 
  sorry

end pears_left_l1361_136191


namespace simplify_expression_l1361_136137

theorem simplify_expression (a : ℚ) (h : a^2 - a - 7/2 = 0) : 
  a^2 - (a - (2 * a) / (a + 1)) / ((a^2 - 2 * a + 1) / (a^2 - 1)) = 7 / 2 := 
by
  sorry

end simplify_expression_l1361_136137


namespace sequence_next_term_l1361_136189

theorem sequence_next_term (a b c d e : ℕ) (h1 : a = 34) (h2 : b = 45) (h3 : c = 56) (h4 : d = 67) (h5 : e = 78) (h6 : b = a + 11) (h7 : c = b + 11) (h8 : d = c + 11) (h9 : e = d + 11) : e + 11 = 89 :=
by
  sorry

end sequence_next_term_l1361_136189


namespace function_identity_l1361_136158

theorem function_identity {f : ℕ → ℕ} (h₀ : f 1 > 0) 
  (h₁ : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l1361_136158


namespace min_sides_of_polygon_that_overlaps_after_rotation_l1361_136184

theorem min_sides_of_polygon_that_overlaps_after_rotation (θ : ℝ) (n : ℕ) 
  (hθ: θ = 36) (hdiv: 360 % θ = 0) :
    n = 10 :=
by
  sorry

end min_sides_of_polygon_that_overlaps_after_rotation_l1361_136184


namespace books_fraction_sold_l1361_136115

theorem books_fraction_sold (B : ℕ) (h1 : B - 36 * 2 = 144) :
  (B - 36) / B = 2 / 3 := by
  sorry

end books_fraction_sold_l1361_136115


namespace anicka_savings_l1361_136148

theorem anicka_savings (x y : ℕ) (h1 : x + y = 290) (h2 : (1/4 : ℚ) * (2 * y) = (1/3 : ℚ) * x) : 2 * y + x = 406 :=
by
  sorry

end anicka_savings_l1361_136148


namespace first_player_wins_game_l1361_136100

theorem first_player_wins_game :
  ∀ (coins : ℕ), coins = 2019 →
  (∀ (n : ℕ), n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99) →
  (∀ (m : ℕ), m % 2 = 0 ∧ 2 ≤ m ∧ m ≤ 100) →
  ∃ (f : ℕ → ℕ → ℕ), (∀ (c : ℕ), c <= coins → c = 0) :=
by
  sorry

end first_player_wins_game_l1361_136100


namespace range_of_c_l1361_136127

variable {a c : ℝ}

theorem range_of_c (h : a ≥ 1 / 8) (sufficient_but_not_necessary : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 := 
sorry

end range_of_c_l1361_136127


namespace value_of_a2018_l1361_136114

noncomputable def a : ℕ → ℝ
| 0       => 2
| (n + 1) => (1 + a n) / (1 - a n)

theorem value_of_a2018 : a 2017 = -3 := sorry

end value_of_a2018_l1361_136114


namespace greatest_area_difference_l1361_136196

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) 
  (h₁ : 2 * l₁ + 2 * w₁ = 160) 
  (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  1521 = (l₁ * w₁ - l₂ * w₂) → 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 1600 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) ∧ 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 79 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) :=
sorry

end greatest_area_difference_l1361_136196


namespace ratio_initial_to_doubled_l1361_136197

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 5) = 105) : x / (2 * x) = 1 / 2 :=
by
  sorry

end ratio_initial_to_doubled_l1361_136197


namespace kocourkov_coins_l1361_136110

theorem kocourkov_coins :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
  (∀ n > 53, ∃ x y : ℕ, n = x * a + y * b) ∧ 
  ¬ (∃ x y : ℕ, 53 = x * a + y * b) ∧
  ((a = 2 ∧ b = 55) ∨ (a = 3 ∧ b = 28)) :=
by {
  sorry
}

end kocourkov_coins_l1361_136110


namespace final_number_is_correct_l1361_136133

-- Define the problem conditions as Lean definitions/statements
def original_number : ℤ := 4
def doubled_number (x : ℤ) : ℤ := 2 * x
def resultant_number (x : ℤ) : ℤ := doubled_number x + 9
def final_number (x : ℤ) : ℤ := 3 * resultant_number x

-- Formulate the theorem using the conditions
theorem final_number_is_correct :
  final_number original_number = 51 :=
by
  sorry

end final_number_is_correct_l1361_136133


namespace elena_subtracts_99_to_compute_49_squared_l1361_136192

noncomputable def difference_between_squares_50_49 : ℕ := 99

theorem elena_subtracts_99_to_compute_49_squared :
  ∀ (n : ℕ), n = 50 → (n - 1)^2 = n^2 - difference_between_squares_50_49 :=
by
  intro n
  sorry

end elena_subtracts_99_to_compute_49_squared_l1361_136192


namespace area_of_fourth_rectangle_l1361_136130

theorem area_of_fourth_rectangle
  (A1 A2 A3 A_total : ℕ)
  (h1 : A1 = 24)
  (h2 : A2 = 30)
  (h3 : A3 = 18)
  (h_total : A_total = 100) :
  ∃ A4 : ℕ, A1 + A2 + A3 + A4 = A_total ∧ A4 = 28 :=
by
  sorry

end area_of_fourth_rectangle_l1361_136130


namespace number_of_students_l1361_136124

theorem number_of_students (N : ℕ) (h1 : (1/5 : ℚ) * N + (1/4 : ℚ) * N + (1/2 : ℚ) * N + 5 = N) : N = 100 :=
by
  sorry

end number_of_students_l1361_136124


namespace min_value_y_l1361_136164

theorem min_value_y (x : ℝ) (h : x > 1) : 
  ∃ y_min : ℝ, (∀ y, y = (1 / (x - 1) + x) → y ≥ y_min) ∧ y_min = 3 :=
sorry

end min_value_y_l1361_136164


namespace hyperbola_eq_l1361_136145

theorem hyperbola_eq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (hyp_eq : ∀ x y, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1)
  (asymptote : b / a = Real.sqrt 3)
  (focus_parabola : c = 4) : 
  a^2 = 4 ∧ b^2 = 12 := by
sorry

end hyperbola_eq_l1361_136145


namespace no_integer_solution_l1361_136128

theorem no_integer_solution (a b : ℤ) : ¬(a^2 + b^2 = 10^100 + 3) :=
sorry

end no_integer_solution_l1361_136128


namespace specific_value_eq_l1361_136172

def specific_value (x : ℕ) : ℕ := 25 * x

theorem specific_value_eq : specific_value 27 = 675 := by
  sorry

end specific_value_eq_l1361_136172


namespace triangle_angles_in_given_ratio_l1361_136149

theorem triangle_angles_in_given_ratio (x : ℝ) (y : ℝ) (z : ℝ) (h : x + y + z = 180) (r : x / 1 = y / 4 ∧ y / 4 = z / 7) : 
  x = 15 ∧ y = 60 ∧ z = 105 :=
by
  sorry

end triangle_angles_in_given_ratio_l1361_136149


namespace factor_x12_minus_4096_l1361_136135

theorem factor_x12_minus_4096 (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) :=
by
  sorry

end factor_x12_minus_4096_l1361_136135


namespace calculate_max_income_l1361_136147

variables 
  (total_lunch_pasta : ℕ) (total_lunch_chicken : ℕ) (total_lunch_fish : ℕ)
  (sold_lunch_pasta : ℕ) (sold_lunch_chicken : ℕ) (sold_lunch_fish : ℕ)
  (dinner_pasta : ℕ) (dinner_chicken : ℕ) (dinner_fish : ℕ)
  (price_pasta : ℝ) (price_chicken : ℝ) (price_fish : ℝ)
  (discount : ℝ)
  (max_income : ℝ)

def unsold_lunch_pasta := total_lunch_pasta - sold_lunch_pasta
def unsold_lunch_chicken := total_lunch_chicken - sold_lunch_chicken
def unsold_lunch_fish := total_lunch_fish - sold_lunch_fish

def discounted_price (price : ℝ) := price * (1 - discount)

def income_lunch (sold : ℕ) (price : ℝ) := sold * price
def income_dinner (fresh : ℕ) (price : ℝ) := fresh * price
def income_unsold (unsold : ℕ) (price : ℝ) := unsold * discounted_price price

theorem calculate_max_income 
  (h_pasta_total : total_lunch_pasta = 8) (h_chicken_total : total_lunch_chicken = 5) (h_fish_total : total_lunch_fish = 4)
  (h_pasta_sold : sold_lunch_pasta = 6) (h_chicken_sold : sold_lunch_chicken = 3) (h_fish_sold : sold_lunch_fish = 3)
  (h_dinner_pasta : dinner_pasta = 2) (h_dinner_chicken : dinner_chicken = 2) (h_dinner_fish : dinner_fish = 1)
  (h_price_pasta: price_pasta = 12) (h_price_chicken: price_chicken = 15) (h_price_fish: price_fish = 18)
  (h_discount: discount = 0.10) 
  : max_income = 136.80 :=
  sorry

end calculate_max_income_l1361_136147


namespace increasing_sequence_range_of_a_l1361_136185

theorem increasing_sequence_range_of_a (a : ℝ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, a_n n = a * n ^ 2 + n) (increasing : ∀ n : ℕ, a_n (n + 1) > a_n n) : 0 ≤ a :=
by
  sorry

end increasing_sequence_range_of_a_l1361_136185


namespace binom_25_5_l1361_136122

theorem binom_25_5 :
  (Nat.choose 23 3 = 1771) ∧
  (Nat.choose 23 4 = 8855) ∧
  (Nat.choose 23 5 = 33649) → 
  Nat.choose 25 5 = 53130 := by
sorry

end binom_25_5_l1361_136122


namespace circle_equation_correct_l1361_136188

def line_through_fixed_point (a : ℝ) :=
  ∀ x y : ℝ, (x + y - 1) - a * (x + 1) = 0 → x = -1 ∧ y = 2

def equation_of_circle (x y: ℝ) :=
  (x + 1)^2 + (y - 2)^2 = 5

theorem circle_equation_correct (a : ℝ) (h : line_through_fixed_point a) :
  ∀ x y : ℝ, equation_of_circle x y ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
sorry

end circle_equation_correct_l1361_136188


namespace green_notebook_cost_l1361_136159

-- Define the conditions
def num_notebooks : Nat := 4
def num_green_notebooks : Nat := 2
def num_black_notebooks : Nat := 1
def num_pink_notebooks : Nat := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def pink_notebook_cost : ℕ := 10

-- Define what we need to prove: The cost of each green notebook
def green_notebook_cost_each : ℕ := 10

-- The statement that combines the conditions with the goal to prove
theorem green_notebook_cost : 
  num_notebooks = 4 ∧ 
  num_green_notebooks = 2 ∧ 
  num_black_notebooks = 1 ∧ 
  num_pink_notebooks = 1 ∧ 
  total_cost = 45 ∧ 
  black_notebook_cost = 15 ∧ 
  pink_notebook_cost = 10 →
  2 * green_notebook_cost_each = total_cost - (black_notebook_cost + pink_notebook_cost) :=
by
  sorry

end green_notebook_cost_l1361_136159


namespace upper_limit_arun_weight_l1361_136103

theorem upper_limit_arun_weight (x w : ℝ) :
  (65 < w ∧ w < x) ∧
  (60 < w ∧ w < 70) ∧
  (w ≤ 68) ∧
  (w = 67) →
  x = 68 :=
by
  sorry

end upper_limit_arun_weight_l1361_136103


namespace train_cross_pole_time_l1361_136105

noncomputable def time_to_cross_pole : ℝ :=
  let speed_km_hr := 60
  let speed_m_s := speed_km_hr * 1000 / 3600
  let length_of_train := 50
  length_of_train / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole = 3 := 
by
  sorry

end train_cross_pole_time_l1361_136105


namespace length_after_y_months_isabella_hair_length_l1361_136199

-- Define the initial length of the hair
def initial_length : ℝ := 18

-- Define the growth rate of the hair per month
def growth_rate (x : ℝ) : ℝ := x

-- Define the number of months passed
def months_passed (y : ℕ) : ℕ := y

-- Prove the length of the hair after 'y' months
theorem length_after_y_months (x : ℝ) (y : ℕ) : ℝ :=
  initial_length + growth_rate x * y

-- Theorem statement to prove that the length of Isabella's hair after y months is 18 + xy
theorem isabella_hair_length (x : ℝ) (y : ℕ) : length_after_y_months x y = 18 + x * y :=
by sorry

end length_after_y_months_isabella_hair_length_l1361_136199


namespace inverse_proportional_l1361_136193

/-- Given that α is inversely proportional to β and α = -3 when β = -6,
    prove that α = 9/4 when β = 8. --/
theorem inverse_proportional (α β : ℚ) 
  (h1 : α * β = 18)
  (h2 : β = 8) : 
  α = 9 / 4 :=
by
  sorry

end inverse_proportional_l1361_136193


namespace find_value_of_x_l1361_136101

theorem find_value_of_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := 
sorry

end find_value_of_x_l1361_136101


namespace julie_bought_boxes_l1361_136116

-- Definitions for the conditions
def packages_per_box := 5
def sheets_per_package := 250
def sheets_per_newspaper := 25
def newspapers := 100

-- Calculations based on conditions
def total_sheets_needed := newspapers * sheets_per_newspaper
def sheets_per_box := packages_per_box * sheets_per_package

-- The goal: to prove that the number of boxes of paper Julie bought is 2
theorem julie_bought_boxes : total_sheets_needed / sheets_per_box = 2 :=
  by
    sorry

end julie_bought_boxes_l1361_136116


namespace solve_A_solve_area_l1361_136102

noncomputable def angle_A (A : ℝ) : Prop :=
  2 * (Real.cos (A / 2))^2 + Real.cos A = 0

noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 → b + c = 4 → A = 2 * Real.pi / 3 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3

theorem solve_A (A : ℝ) : angle_A A → A = 2 * Real.pi / 3 :=
sorry

theorem solve_area (a b c A S : ℝ) : 
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  A = 2 * Real.pi / 3 →
  area_triangle a b c A →
  S = Real.sqrt 3 :=
sorry

end solve_A_solve_area_l1361_136102


namespace solve_inequality_l1361_136198

theorem solve_inequality (x : ℝ) (h : x ≠ -2 / 3) :
  3 - (1 / (3 * x + 2)) < 5 ↔ (x < -7 / 6 ∨ x > -2 / 3) := by
  sorry

end solve_inequality_l1361_136198


namespace integer_solutions_x2_minus_y2_equals_12_l1361_136155

theorem integer_solutions_x2_minus_y2_equals_12 : 
  ∃! (s : Finset (ℤ × ℤ)), (∀ (xy : ℤ × ℤ), xy ∈ s → xy.1^2 - xy.2^2 = 12) ∧ s.card = 4 :=
sorry

end integer_solutions_x2_minus_y2_equals_12_l1361_136155


namespace tan_lt_neg_one_implies_range_l1361_136140

theorem tan_lt_neg_one_implies_range {x : ℝ} (h1 : 0 < x) (h2 : x < π) (h3 : Real.tan x < -1) :
  (π / 2 < x) ∧ (x < 3 * π / 4) :=
sorry

end tan_lt_neg_one_implies_range_l1361_136140


namespace painting_rate_l1361_136150

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end painting_rate_l1361_136150


namespace determine_value_of_c_l1361_136161

theorem determine_value_of_c (b : ℝ) (h₁ : ∀ x : ℝ, 0 ≤ x^2 + x + b) (h₂ : ∃ m : ℝ, ∀ x : ℝ, x^2 + x + b < c ↔ x = m + 8) : 
    c = 16 :=
sorry

end determine_value_of_c_l1361_136161


namespace largest_of_three_consecutive_integers_sum_18_l1361_136166

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l1361_136166


namespace convex_quadrilateral_probability_l1361_136162

noncomputable def probability_convex_quadrilateral (n : ℕ) : ℚ :=
  if n = 6 then (Nat.choose 6 4 : ℚ) / (Nat.choose 15 4 : ℚ) else 0

theorem convex_quadrilateral_probability :
  probability_convex_quadrilateral 6 = 1 / 91 :=
by
  sorry

end convex_quadrilateral_probability_l1361_136162


namespace max_points_on_circle_l1361_136121

noncomputable def circleMaxPoints (P C : ℝ × ℝ) (r1 r2 d : ℝ) : ℕ :=
  if d = r1 + r2 ∨ d = abs (r1 - r2) then 1 else 
  if d < r1 + r2 ∧ d > abs (r1 - r2) then 2 else 0

theorem max_points_on_circle (P : ℝ × ℝ) (C : ℝ × ℝ) :
  let rC := 5
  let distPC := 9
  let rP := 4
  circleMaxPoints P C rC rP distPC = 1 :=
by sorry

end max_points_on_circle_l1361_136121


namespace proof_for_y_l1361_136131

theorem proof_for_y (x y : ℝ) (h1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0) (h2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 :=
sorry

end proof_for_y_l1361_136131


namespace grocer_pounds_of_bananas_purchased_l1361_136134

/-- 
Given:
1. The grocer purchased bananas at a rate of 3 pounds for $0.50.
2. The grocer sold the entire quantity at a rate of 4 pounds for $1.00.
3. The profit from selling the bananas was $11.00.

Prove that the number of pounds of bananas the grocer purchased is 132. 
-/
theorem grocer_pounds_of_bananas_purchased (P : ℕ) 
    (h1 : ∃ P, (3 * P / 0.5) - (4 * P / 1.0) = 11) : 
    P = 132 := 
sorry

end grocer_pounds_of_bananas_purchased_l1361_136134


namespace smallest_sum_zero_l1361_136181

theorem smallest_sum_zero : ∃ x ∈ ({-1, -2, 1, 2} : Set ℤ), ∀ y ∈ ({-1, -2, 1, 2} : Set ℤ), x + 0 ≤ y + 0 :=
sorry

end smallest_sum_zero_l1361_136181


namespace common_real_solution_unique_y_l1361_136163

theorem common_real_solution_unique_y (x y : ℝ) 
  (h1 : x^2 + y^2 = 16) 
  (h2 : x^2 - 3 * y + 12 = 0) : 
  y = 4 :=
by
  sorry

end common_real_solution_unique_y_l1361_136163


namespace pints_in_5_liters_l1361_136177

-- Define the condition based on the given conversion factor from liters to pints
def conversion_factor : ℝ := 2.1

-- The statement we need to prove
theorem pints_in_5_liters : 5 * conversion_factor = 10.5 :=
by sorry

end pints_in_5_liters_l1361_136177


namespace volleyball_not_basketball_l1361_136109

def class_size : ℕ := 40
def basketball_enjoyers : ℕ := 15
def volleyball_enjoyers : ℕ := 20
def neither_sport : ℕ := 10

theorem volleyball_not_basketball :
  (volleyball_enjoyers - (basketball_enjoyers + volleyball_enjoyers - (class_size - neither_sport))) = 15 :=
by
  sorry

end volleyball_not_basketball_l1361_136109


namespace number_to_remove_l1361_136106

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem number_to_remove (s : List ℕ) (x : ℕ) 
  (h₀ : s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  (h₁ : x ∈ s)
  (h₂ : mean (List.erase s x) = 6.1) : x = 5 := sorry

end number_to_remove_l1361_136106


namespace average_speed_correct_l1361_136132

noncomputable def average_speed (initial_odometer : ℝ) (lunch_odometer : ℝ) (final_odometer : ℝ) (total_time : ℝ) : ℝ :=
  (final_odometer - initial_odometer) / total_time

theorem average_speed_correct :
  average_speed 212.3 372 467.2 6.25 = 40.784 :=
by
  unfold average_speed
  sorry

end average_speed_correct_l1361_136132


namespace gcd_1728_1764_l1361_136156

theorem gcd_1728_1764 : Int.gcd 1728 1764 = 36 := by
  sorry

end gcd_1728_1764_l1361_136156


namespace certain_amount_is_19_l1361_136165

theorem certain_amount_is_19 (x y certain_amount : ℤ) 
  (h1 : x + y = 15)
  (h2 : 3 * x = 5 * y - certain_amount)
  (h3 : x = 7) : 
  certain_amount = 19 :=
by
  sorry

end certain_amount_is_19_l1361_136165


namespace work_rate_solution_l1361_136175

theorem work_rate_solution (x : ℝ) (hA : 60 > 0) (hB : x > 0) (hTogether : 15 > 0) :
  (1 / 60 + 1 / x = 1 / 15) → (x = 20) :=
by 
  sorry -- Proof Placeholder

end work_rate_solution_l1361_136175


namespace ratio_of_p_to_q_l1361_136108

theorem ratio_of_p_to_q (p q : ℝ) (h₁ : (p + q) / (p - q) = 4 / 3) (h₂ : p / q = r) : r = 7 :=
sorry

end ratio_of_p_to_q_l1361_136108


namespace compute_f_at_919_l1361_136151

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 4) = f (x - 2)

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [-3, 0] then 6^(-x) else sorry

-- Lean statement for the proof problem
theorem compute_f_at_919 (f : ℝ → ℝ)
    (h_even : is_even_function f)
    (h_periodic : periodic_function f)
    (h_defined : ∀ x ∈ [-3, 0], f x = 6^(-x)) :
    f 919 = 6 := sorry

end compute_f_at_919_l1361_136151


namespace greatest_number_of_police_officers_needed_l1361_136180

-- Define the conditions within Math City
def number_of_streets : ℕ := 10
def number_of_tunnels : ℕ := 2
def intersections_without_tunnels : ℕ := (number_of_streets * (number_of_streets - 1)) / 2
def intersections_bypassed_by_tunnels : ℕ := number_of_tunnels

-- Define the number of police officers required (which is the same as the number of intersections not bypassed)
def police_officers_needed : ℕ := intersections_without_tunnels - intersections_bypassed_by_tunnels

-- The main theorem: Given the conditions, the greatest number of police officers needed is 43.
theorem greatest_number_of_police_officers_needed : police_officers_needed = 43 := 
by {
  -- Proof would go here, but we'll use sorry to indicate it's not provided.
  sorry
}

end greatest_number_of_police_officers_needed_l1361_136180


namespace range_of_B_l1361_136141

theorem range_of_B (a b c : ℝ) (h : a + c = 2 * b) :
  ∃ B : ℝ, 0 < B ∧ B ≤ π / 3 ∧
  ∃ A C : ℝ, ∃ ha : a = c, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π :=
sorry

end range_of_B_l1361_136141


namespace square_segment_ratio_l1361_136160

theorem square_segment_ratio
  (A B C D E M P Q : ℝ × ℝ)
  (h_square: A = (0, 16) ∧ B = (16, 16) ∧ C = (16, 0) ∧ D = (0, 0))
  (h_E: E = (7, 0))
  (h_midpoint: M = ((0 + 7) / 2, (16 + 0) / 2))
  (h_bisector_P: P = (P.1, 16) ∧ (16 - 8 = (7 / 16) * (P.1 - 3.5)))
  (h_bisector_Q: Q = (Q.1, 0) ∧ (0 - 8 = (7 / 16) * (Q.1 - 3.5)))
  (h_PM: abs (16 - 8) = abs (P.2 - M.2))
  (h_MQ: abs (8 - 0) = abs (M.2 - Q.2)) :
  abs (P.2 - M.2) = abs (M.2 - Q.2) :=
sorry

end square_segment_ratio_l1361_136160


namespace divide_cakes_l1361_136111

/-- Statement: Eleven cakes can be divided equally among six girls without cutting any cake into 
exactly six equal parts such that each girl receives 1 + 1/2 + 1/4 + 1/12 cakes -/
theorem divide_cakes (cakes girls : ℕ) (h_cakes : cakes = 11) (h_girls : girls = 6) :
  ∃ (parts : ℕ → ℝ), (∀ i, parts i = 1 + 1 / 2 + 1 / 4 + 1 / 12) ∧ (cakes = girls * (1 + 1 / 2 + 1 / 4 + 1 / 12)) :=
by
  sorry

end divide_cakes_l1361_136111


namespace line_l_prime_eq_2x_minus_3y_plus_5_l1361_136174

theorem line_l_prime_eq_2x_minus_3y_plus_5 (m : ℝ) (x y : ℝ) : 
  (2 * m + 1) * x + (m + 1) * y + m = 0 →
  (2 * -1 + 1) * (-1) + (1 + 1) * 1 + m = 0 →
  ∀ a b : ℝ, (3 * b, 2 * b) = (3 * 1, 2 * 1) → (a, b) = (-1, 1) → 
  2 * x - 3 * y + 5 = 0 :=
by
  intro h1 h2 a b h3 h4
  sorry

end line_l_prime_eq_2x_minus_3y_plus_5_l1361_136174


namespace find_m_n_l1361_136117

-- Define the set A
def set_A : Set ℝ := {x | |x + 2| < 3}

-- Define the set B in terms of a variable m
def set_B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

-- State the theorem
theorem find_m_n (m n : ℝ) (hA : set_A = {x | -5 < x ∧ x < 1}) (h_inter : set_A ∩ set_B m = {x | -1 < x ∧ x < n}) : 
  m = -1 ∧ n = 1 :=
by
  -- Proof is omitted
  sorry

end find_m_n_l1361_136117
