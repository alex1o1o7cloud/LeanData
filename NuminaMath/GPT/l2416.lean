import Mathlib

namespace directrix_parabola_l2416_241679

theorem directrix_parabola (x y : ℝ) :
  (x^2 = (1/4 : ℝ) * y) → (y = -1/16) :=
sorry

end directrix_parabola_l2416_241679


namespace cafeteria_pies_l2416_241621

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h_initial : initial_apples = 50)
  (h_handed_out : handed_out_apples = 5)
  (h_apples_per_pie : apples_per_pie = 5) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := 
by
  sorry

end cafeteria_pies_l2416_241621


namespace polynomial_transformation_l2416_241634

variable {x y : ℝ}

theorem polynomial_transformation
  (h : y = x + 1/x) 
  (poly_eq_0 : x^4 + x^3 - 5*x^2 + x + 1 = 0) :
  x^2 * (y^2 + y - 7) = 0 :=
sorry

end polynomial_transformation_l2416_241634


namespace min_value_expression_l2416_241618

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (v : ℝ), (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ v) ∧ v = 30 :=
by
  sorry

end min_value_expression_l2416_241618


namespace find_fixed_point_l2416_241665

theorem find_fixed_point (c d k : ℝ) 
(h : ∀ k : ℝ, d = 5 * c^2 + k * c - 3 * k) : (c, d) = (3, 45) :=
sorry

end find_fixed_point_l2416_241665


namespace highest_growth_rate_at_K_div_2_l2416_241638

variable {K : ℝ}

-- Define the population growth rate as a function of the population size.
def population_growth_rate (N : ℝ) : ℝ := sorry

-- Define the S-shaped curve condition of population growth.
axiom s_shaped_curve : ∃ N : ℝ, population_growth_rate N = 0 ∧ population_growth_rate (N/2) > population_growth_rate N

theorem highest_growth_rate_at_K_div_2 (N : ℝ) (hN : N = K/2) :
  population_growth_rate N > population_growth_rate K :=
by
  sorry

end highest_growth_rate_at_K_div_2_l2416_241638


namespace union_P_Q_l2416_241606

noncomputable def P : Set ℤ := {x | x^2 - x = 0}
noncomputable def Q : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

theorem union_P_Q : P ∪ Q = {-1, 0, 1} :=
by 
  sorry

end union_P_Q_l2416_241606


namespace greatest_integer_third_side_of_triangle_l2416_241667

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l2416_241667


namespace proof_problem_l2416_241660

variables {a : ℕ → ℕ} -- sequence a_n is positive integers
variables {b : ℕ → ℕ} -- sequence b_n is integers
variables {q : ℕ} -- ratio for geometric sequence
variables {d : ℕ} -- difference for arithmetic sequence
variables {a1 b1 : ℕ} -- initial terms for the sequences

-- Additional conditions as per the problem statement
def geometric_seq (a : ℕ → ℕ) (a1 q : ℕ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n-1)

def arithmetic_seq (b : ℕ → ℕ) (b1 d : ℕ) : Prop :=
∀ n : ℕ, b n = b1 + (n-1) * d

-- Given conditions
variable (geometric : geometric_seq a a1 q)
variable (arithmetic : arithmetic_seq b b1 d)
variable (equal_term : a 6 = b 7)

-- The proof task
theorem proof_problem : a 3 + a 9 ≥ b 4 + b 10 :=
by sorry

end proof_problem_l2416_241660


namespace vanessa_made_16_l2416_241609

/-
Each chocolate bar in a box costs $4.
There are 11 bars in total in the box.
Vanessa sold all but 7 bars.
Prove that Vanessa made $16.
-/

def cost_per_bar : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost_per_bar

theorem vanessa_made_16 : money_made = 16 :=
by
  sorry

end vanessa_made_16_l2416_241609


namespace find_f_4500_l2416_241630

noncomputable def f : ℕ → ℕ
| 0 => 1
| (n + 3) => f n + 2 * n + 3
| n => sorry  -- This handles all other cases, but should not be called.

theorem find_f_4500 : f 4500 = 6750001 :=
by
  sorry

end find_f_4500_l2416_241630


namespace find_quadratic_function_l2416_241683

theorem find_quadratic_function (g : ℝ → ℝ) 
  (h1 : g 0 = 0) 
  (h2 : g 1 = 1) 
  (h3 : g (-1) = 5) 
  (h_quadratic : ∃ a b, ∀ x, g x = a * x^2 + b * x) : 
  g = fun x => 3 * x^2 - 2 * x := 
by
  sorry

end find_quadratic_function_l2416_241683


namespace find_number_l2416_241632

theorem find_number (x : ℤ) (h : (7 * (x + 10) / 5) - 5 = 44) : x = 25 :=
sorry

end find_number_l2416_241632


namespace opposite_vertices_equal_l2416_241680

-- Define the angles of a regular convex hexagon
variables {α β γ δ ε ζ : ℝ}

-- Regular hexagon condition: The sum of the alternating angles
axiom angle_sum_condition :
  α + γ + ε = β + δ + ε

-- Define the final theorem to prove that the opposite vertices have equal angles
theorem opposite_vertices_equal (h : α + γ + ε = β + δ + ε) :
  α = δ ∧ β = ε ∧ γ = ζ :=
sorry

end opposite_vertices_equal_l2416_241680


namespace correct_option_is_C_l2416_241617

namespace ExponentProof

-- Definitions of conditions
def optionA (a : ℝ) : Prop := a^3 * a^4 = a^12
def optionB (a : ℝ) : Prop := a^3 + a^4 = a^7
def optionC (a : ℝ) : Prop := a^5 / a^3 = a^2
def optionD (a : ℝ) : Prop := (-2 * a)^3 = -6 * a^3

-- Proof problem stating that optionC is the only correct one
theorem correct_option_is_C : ∀ (a : ℝ), ¬ optionA a ∧ ¬ optionB a ∧ optionC a ∧ ¬ optionD a :=
by
  intro a
  sorry

end ExponentProof

end correct_option_is_C_l2416_241617


namespace extra_men_needed_approx_is_60_l2416_241694

noncomputable def extra_men_needed : ℝ :=
  let total_distance := 15.0   -- km
  let total_days := 300.0      -- days
  let initial_workforce := 40.0 -- men
  let completed_distance := 2.5 -- km
  let elapsed_days := 100.0    -- days

  let remaining_distance := total_distance - completed_distance -- km
  let remaining_days := total_days - elapsed_days               -- days

  let current_rate := completed_distance / elapsed_days -- km/day
  let required_rate := remaining_distance / remaining_days -- km/day

  let required_factor := required_rate / current_rate
  let new_workforce := initial_workforce * required_factor
  let extra_men := new_workforce - initial_workforce

  extra_men

theorem extra_men_needed_approx_is_60 :
  abs (extra_men_needed - 60) < 1 :=
sorry

end extra_men_needed_approx_is_60_l2416_241694


namespace determine_exponent_l2416_241697

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

theorem determine_exponent (a : ℝ) (hf : power_function a 4 = 8) : power_function (3/2) = power_function a := by
  sorry

end determine_exponent_l2416_241697


namespace problem1_problem2_l2416_241677

-- Problem 1 Statement
theorem problem1 (a : ℝ) (h : a ≠ 1) : (a^2 / (a - 1) - a - 1) = (1 / (a - 1)) :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) : 
  (2 * x * y / (x^2 - y^2)) / ((1 / (x - y)) + (1 / (x + y))) = y :=
by
  sorry

end problem1_problem2_l2416_241677


namespace width_of_metallic_sheet_is_36_l2416_241625

-- Given conditions
def length_of_metallic_sheet : ℕ := 48
def side_length_of_cutoff_square : ℕ := 8
def volume_of_box : ℕ := 5120

-- Proof statement
theorem width_of_metallic_sheet_is_36 :
  ∀ (w : ℕ), w - 2 * side_length_of_cutoff_square = 36 - 16 →  length_of_metallic_sheet - 2* side_length_of_cutoff_square = 32  →  5120 = 256 * (w - 16)  := sorry

end width_of_metallic_sheet_is_36_l2416_241625


namespace probability_all_from_same_tribe_l2416_241659

-- Definitions based on the conditions of the problem
def total_people := 24
def tribe_count := 3
def people_per_tribe := 8
def quitters := 3

-- We assume each person has an equal chance of quitting and the quitters are chosen independently
-- The probability that all three people who quit belong to the same tribe

theorem probability_all_from_same_tribe :
  ((3 * (Nat.choose people_per_tribe quitters)) / (Nat.choose total_people quitters) : ℚ) = 1 / 12 := 
  by 
    sorry

end probability_all_from_same_tribe_l2416_241659


namespace time_for_nth_mile_l2416_241695

noncomputable def speed (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

noncomputable def time_for_mile (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2 * (n - 1) * (n - 1)

theorem time_for_nth_mile (n : ℕ) (h₁ : ∀ d : ℝ, d ≥ 1 → speed (1/2) d = 1 / (2 * d * d))
  (h₂ : time_for_mile 1 = 1)
  (h₃ : time_for_mile 2 = 2) :
  time_for_mile n = 2 * (n - 1) * (n - 1) := sorry

end time_for_nth_mile_l2416_241695


namespace range_of_a_l2416_241648

theorem range_of_a (a : ℝ) : 
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)) → a > 1 :=
by
  sorry

end range_of_a_l2416_241648


namespace sum_series_eq_260_l2416_241690

theorem sum_series_eq_260 : (2 + 12 + 22 + 32 + 42) + (10 + 20 + 30 + 40 + 50) = 260 := by
  sorry

end sum_series_eq_260_l2416_241690


namespace zero_in_interval_l2416_241627

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

theorem zero_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) → -- f(x) is increasing on (0, +∞)
  f 2 < 0 → -- f(2) < 0
  f 3 > 0 → -- f(3) > 0
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  intros h_increasing h_f2_lt_0 h_f3_gt_0
  sorry

end zero_in_interval_l2416_241627


namespace abs_difference_equality_l2416_241671

theorem abs_difference_equality : (abs (3 - Real.sqrt 2) - abs (Real.sqrt 2 - 2) = 1) :=
  by
    -- Define our conditions as hypotheses
    have h1 : 3 > Real.sqrt 2 := sorry
    have h2 : Real.sqrt 2 < 2 := sorry
    -- The proof itself is skipped in this step
    sorry

end abs_difference_equality_l2416_241671


namespace round_155_628_l2416_241672

theorem round_155_628 :
  round (155.628 : Real) = 156 := by
  sorry

end round_155_628_l2416_241672


namespace problem1_problem2_l2416_241626

theorem problem1 (x : ℝ) (h : 4 * x^2 - 9 = 0) : x = 3/2 ∨ x = -3/2 :=
by
  sorry

theorem problem2 (x : ℝ) (h : 64 * (x-2)^3 - 1 = 0) : x = 2 + 1/4 :=
by
  sorry

end problem1_problem2_l2416_241626


namespace negation_of_proposition_l2416_241653

namespace NegationProp

theorem negation_of_proposition :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - x < 0) ↔
  (∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ x0^2 - x0 ≥ 0) := by sorry

end NegationProp

end negation_of_proposition_l2416_241653


namespace proof_problem_l2416_241608

-- Define the operation
def star (a b : ℝ) : ℝ := (a - b) ^ 2

-- The proof problem as a Lean statement
theorem proof_problem (x y : ℝ) : star ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry

end proof_problem_l2416_241608


namespace faster_current_takes_more_time_l2416_241688

theorem faster_current_takes_more_time (v v1 v2 S : ℝ) (h_v1_gt_v2 : v1 > v2) :
  let t1 := (2 * S * v) / (v^2 - v1^2)
  let t2 := (2 * S * v) / (v^2 - v2^2)
  t1 > t2 :=
by
  sorry

end faster_current_takes_more_time_l2416_241688


namespace positions_after_347_moves_l2416_241643

-- Define the possible positions for the cat
inductive CatPosition
| top_vertex
| right_upper_vertex
| right_lower_vertex
| left_lower_vertex
| left_upper_vertex

-- Define the possible positions for the mouse
inductive MousePosition
| top_left_edge
| left_upper_vertex
| left_middle_edge
| left_lower_vertex
| bottom_edge
| right_lower_vertex
| right_middle_edge
| right_upper_vertex
| top_right_edge
| top_vertex

-- Define the movement function for the cat
def cat_position_after_moves (moves : Nat) : CatPosition :=
  match moves % 5 with
  | 0 => CatPosition.top_vertex
  | 1 => CatPosition.right_upper_vertex
  | 2 => CatPosition.right_lower_vertex
  | 3 => CatPosition.left_lower_vertex
  | 4 => CatPosition.left_upper_vertex
  | _ => CatPosition.top_vertex  -- This case is unreachable due to % 5

-- Define the movement function for the mouse
def mouse_position_after_moves (moves : Nat) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.top_left_edge
  | 1 => MousePosition.left_upper_vertex
  | 2 => MousePosition.left_middle_edge
  | 3 => MousePosition.left_lower_vertex
  | 4 => MousePosition.bottom_edge
  | 5 => MousePosition.right_lower_vertex
  | 6 => MousePosition.right_middle_edge
  | 7 => MousePosition.right_upper_vertex
  | 8 => MousePosition.top_right_edge
  | 9 => MousePosition.top_vertex
  | _ => MousePosition.top_left_edge  -- This case is unreachable due to % 10

-- Prove the positions after 347 moves
theorem positions_after_347_moves :
  cat_position_after_moves 347 = CatPosition.right_upper_vertex ∧
  mouse_position_after_moves 347 = MousePosition.right_middle_edge :=
by
  sorry

end positions_after_347_moves_l2416_241643


namespace meaningful_expression_range_l2416_241623

theorem meaningful_expression_range (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 2 ≠ 0) → x < 2 :=
by
  sorry

end meaningful_expression_range_l2416_241623


namespace total_cups_of_ingredients_l2416_241644

theorem total_cups_of_ingredients
  (ratio_butter : ℕ) (ratio_flour : ℕ) (ratio_sugar : ℕ)
  (flour_cups : ℕ)
  (h_ratio : ratio_butter = 2 ∧ ratio_flour = 3 ∧ ratio_sugar = 5)
  (h_flour : flour_cups = 6) :
  let part_cups := flour_cups / ratio_flour
  let butter_cups := ratio_butter * part_cups
  let sugar_cups := ratio_sugar * part_cups
  let total_cups := butter_cups + flour_cups + sugar_cups
  total_cups = 20 :=
by
  sorry

end total_cups_of_ingredients_l2416_241644


namespace bella_grazing_area_l2416_241628

open Real

theorem bella_grazing_area:
  let leash_length := 5
  let barn_width := 4
  let barn_height := 6
  let sector_fraction := 3 / 4
  let area_circle := π * leash_length^2
  let grazed_area := sector_fraction * area_circle
  grazed_area = 75 / 4 * π := 
by
  sorry

end bella_grazing_area_l2416_241628


namespace gcd_ab_conditions_l2416_241698

theorem gcd_ab_conditions 
  (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) : 
  Nat.gcd (a + b) (a - b) = 1 ∨ Nat.gcd (a + b) (a - b) = 2 := 
sorry

end gcd_ab_conditions_l2416_241698


namespace original_combined_price_l2416_241640

theorem original_combined_price (C S : ℝ) 
  (candy_box_increased : C * 1.25 = 18.75)
  (soda_can_increased : S * 1.50 = 9) : 
  C + S = 21 :=
by
  sorry

end original_combined_price_l2416_241640


namespace solution1_solution2_l2416_241673

open Complex

noncomputable def problem1 : Prop := 
  ((3 - I) / (1 + I)) ^ 2 = -3 - 4 * I

noncomputable def problem2 (z : ℂ) : Prop := 
  z = 1 + I → (2 / z - z = -2 * I)

theorem solution1 : problem1 := 
  by sorry

theorem solution2 : problem2 (1 + I) :=
  by sorry

end solution1_solution2_l2416_241673


namespace number_modulo_conditions_l2416_241676

theorem number_modulo_conditions : 
  ∃ n : ℕ, 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 2519) :=
by
  sorry

end number_modulo_conditions_l2416_241676


namespace helium_balloon_height_l2416_241657

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end helium_balloon_height_l2416_241657


namespace smallest_positive_integer_n_l2416_241663

theorem smallest_positive_integer_n (n : ℕ) (h : 527 * n ≡ 1083 * n [MOD 30]) : n = 2 :=
sorry

end smallest_positive_integer_n_l2416_241663


namespace lines_per_page_l2416_241691

theorem lines_per_page
  (total_words : ℕ)
  (words_per_line : ℕ)
  (words_left : ℕ)
  (pages_filled : ℚ) :
  total_words = 400 →
  words_per_line = 10 →
  words_left = 100 →
  pages_filled = 1.5 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end lines_per_page_l2416_241691


namespace domain_of_function_l2416_241610

def domain_conditions (x : ℝ) : Prop :=
  (1 - x ≥ 0) ∧ (x + 2 > 0)

theorem domain_of_function :
  {x : ℝ | domain_conditions x} = {x : ℝ | -2 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l2416_241610


namespace sqrt_factorial_product_l2416_241602

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l2416_241602


namespace number_of_juniors_l2416_241681

theorem number_of_juniors
  (T : ℕ := 28)
  (hT : T = 28)
  (x y : ℕ)
  (hxy : x = y)
  (J S : ℕ)
  (hx : x = J / 4)
  (hy : y = S / 10)
  (hJS : J + S = T) :
  J = 8 :=
by sorry

end number_of_juniors_l2416_241681


namespace fraction_of_bag_spent_on_lunch_l2416_241649

-- Definitions of conditions based on the problem
def initial_amount : ℕ := 158
def price_of_shoes : ℕ := 45
def price_of_bag : ℕ := price_of_shoes - 17
def amount_left : ℕ := 78
def money_before_lunch := amount_left + price_of_shoes + price_of_bag
def money_spent_on_lunch := initial_amount - money_before_lunch 

-- Statement of the problem in Lean
theorem fraction_of_bag_spent_on_lunch :
  (money_spent_on_lunch : ℚ) / price_of_bag = 1 / 4 :=
by
  -- Conditions decoded to match the solution provided
  have h1 : price_of_bag = 28 := by sorry
  have h2 : money_before_lunch = 151 := by sorry
  have h3 : money_spent_on_lunch = 7 := by sorry
  -- The main theorem statement
  exact sorry

end fraction_of_bag_spent_on_lunch_l2416_241649


namespace initial_girls_count_l2416_241658

variable (p : ℕ) -- total number of people initially in the group
variable (girls_initial : ℕ) -- number of girls initially in the group
variable (girls_after : ℕ) -- number of girls after the change
variable (total_after : ℕ) -- total number of people after the change

/--
Initially, 50% of the group are girls. 
Later, five girls leave and five boys arrive, leading to 40% of the group now being girls.
--/
theorem initial_girls_count :
  (girls_initial = p / 2) →
  (total_after = p) →
  (girls_after = girls_initial - 5) →
  (girls_after = 2 * total_after / 5) →
  girls_initial = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_girls_count_l2416_241658


namespace find_number_l2416_241616

theorem find_number (x : ℕ) :
  ((4 * x) / 8 = 6) ∧ ((4 * x) % 8 = 4) → x = 13 :=
by
  sorry

end find_number_l2416_241616


namespace product_evaluation_l2416_241615

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end product_evaluation_l2416_241615


namespace renata_lottery_winnings_l2416_241675

def initial_money : ℕ := 10
def donation : ℕ := 4
def prize_won : ℕ := 90
def water_cost : ℕ := 1
def lottery_ticket_cost : ℕ := 1
def final_money : ℕ := 94

theorem renata_lottery_winnings :
  ∃ (lottery_winnings : ℕ), 
  initial_money - donation + prize_won 
  - water_cost - lottery_ticket_cost 
  = final_money ∧ 
  lottery_winnings = 2 :=
by
  -- Proof steps will go here
  sorry

end renata_lottery_winnings_l2416_241675


namespace sum_first_8_terms_arithmetic_sequence_l2416_241668

theorem sum_first_8_terms_arithmetic_sequence (a : ℕ → ℝ) (h : a 4 + a 5 = 12) :
    (8 * (a 1 + a 8)) / 2 = 48 :=
by
  sorry

end sum_first_8_terms_arithmetic_sequence_l2416_241668


namespace domain_of_log_function_l2416_241637

theorem domain_of_log_function : 
  { x : ℝ | x < 1 ∨ x > 2 } = { x : ℝ | 0 < x^2 - 3 * x + 2 } :=
by sorry

end domain_of_log_function_l2416_241637


namespace greatest_possible_y_l2416_241684

theorem greatest_possible_y (y : ℕ) (h1 : (y^4 / y^2) < 18) : y ≤ 4 := 
  sorry -- Proof to be filled in later

end greatest_possible_y_l2416_241684


namespace sum_of_three_consecutive_even_numbers_is_162_l2416_241633

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end sum_of_three_consecutive_even_numbers_is_162_l2416_241633


namespace dot_product_parallel_a_b_l2416_241674

noncomputable def a : ℝ × ℝ := (-1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Definition of parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2)

-- Given conditions and result to prove
theorem dot_product_parallel_a_b : ∀ (x : ℝ), parallel a (b x) → x = -2 → (a.1 * (b x).1 + a.2 * (b x).2) = -4 := 
by
  intros x h_parallel h_x
  subst h_x
  sorry

end dot_product_parallel_a_b_l2416_241674


namespace normal_price_of_article_l2416_241639

theorem normal_price_of_article 
  (final_price : ℝ) 
  (d1 d2 d3 : ℝ) 
  (P : ℝ) 
  (h_final_price : final_price = 36) 
  (h_d1 : d1 = 0.15) 
  (h_d2 : d2 = 0.25) 
  (h_d3 : d3 = 0.20) 
  (h_eq : final_price = P * (1 - d1) * (1 - d2) * (1 - d3)) : 
  P = 70.59 := sorry

end normal_price_of_article_l2416_241639


namespace simplify_root_power_l2416_241612

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l2416_241612


namespace crayons_per_pack_l2416_241687

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) : crayons_per_pack = 15 := by
sorry

end crayons_per_pack_l2416_241687


namespace percentage_of_two_is_point_eight_l2416_241678

theorem percentage_of_two_is_point_eight (p : ℝ) : (p / 100) * 2 = 0.8 ↔ p = 40 := 
by
  sorry

end percentage_of_two_is_point_eight_l2416_241678


namespace max_value_of_expression_l2416_241647

noncomputable def maxExpression (x y : ℝ) :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  maxExpression x y ≤ (656^2 / 18) :=
by
  sorry

end max_value_of_expression_l2416_241647


namespace angle_B_value_triangle_perimeter_l2416_241642

open Real

variables {A B C a b c : ℝ}

-- Statement 1
theorem angle_B_value (h1 : a = b * sin A + sqrt 3 * a * cos B) : B = π / 2 := by
  sorry

-- Statement 2
theorem triangle_perimeter 
  (h1 : B = π / 2)
  (h2 : b = 4)
  (h3 : (1 / 2) * a * c = 4) : 
  a + b + c = 4 + 4 * sqrt 2 := by
  sorry


end angle_B_value_triangle_perimeter_l2416_241642


namespace johns_average_speed_l2416_241646

def start_time := 8 * 60 + 15  -- 8:15 a.m. in minutes
def end_time := 14 * 60 + 45   -- 2:45 p.m. in minutes
def break_start := 12 * 60     -- 12:00 p.m. in minutes
def break_duration := 30       -- 30 minutes
def total_distance := 240      -- Total distance in miles

def total_driving_time : ℕ := 
  (break_start - start_time) + (end_time - (break_start + break_duration))

def average_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / (time / 60)  -- converting time from minutes to hours

theorem johns_average_speed :
  average_speed total_distance total_driving_time = 40 :=
by
  sorry

end johns_average_speed_l2416_241646


namespace waiter_customers_l2416_241635

theorem waiter_customers
    (initial_tables : ℝ)
    (left_tables : ℝ)
    (customers_per_table : ℝ)
    (remaining_tables : ℝ) 
    (total_customers : ℝ) 
    (h1 : initial_tables = 44.0)
    (h2 : left_tables = 12.0)
    (h3 : customers_per_table = 8.0)
    (remaining_tables_def : remaining_tables = initial_tables - left_tables)
    (total_customers_def : total_customers = remaining_tables * customers_per_table) :
    total_customers = 256.0 :=
by
  sorry

end waiter_customers_l2416_241635


namespace problem_solution_l2416_241692

theorem problem_solution (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ x y, (2 * y / x) + (8 * x / y) > m^2 + 2 * m) → -4 < m ∧ m < 2 :=
by
  intros h
  sorry

end problem_solution_l2416_241692


namespace probability_females_not_less_than_males_l2416_241619

noncomputable def prob_female_not_less_than_male : ℚ :=
  let total_students := 5
  let females := 2
  let males := 3
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose females 2 + females * males
  favorable_combinations / total_combinations

theorem probability_females_not_less_than_males (total_students females males : ℕ) :
  total_students = 5 → females = 2 → males = 3 →
  prob_female_not_less_than_male = 7 / 10 :=
by intros; sorry

end probability_females_not_less_than_males_l2416_241619


namespace g_13_equals_236_l2416_241689

def g (n : ℕ) : ℕ := n^2 + 2 * n + 41

theorem g_13_equals_236 : g 13 = 236 := sorry

end g_13_equals_236_l2416_241689


namespace inverse_44_mod_53_l2416_241685

theorem inverse_44_mod_53 : (44 * 22) % 53 = 1 :=
by
-- Given condition: 19's inverse modulo 53 is 31
have h: (19 * 31) % 53 = 1 := by sorry
-- We should prove the required statement using the given condition.
sorry

end inverse_44_mod_53_l2416_241685


namespace equivalence_of_expression_l2416_241656

theorem equivalence_of_expression (x y : ℝ) :
  ( (x^2 + y^2 + xy) / (x^2 + y^2 - xy) ) - ( (x^2 + y^2 - xy) / (x^2 + y^2 + xy) ) =
  ( 4 * xy * (x^2 + y^2) ) / ( x^4 + y^4 ) :=
by sorry

end equivalence_of_expression_l2416_241656


namespace how_many_bigger_panda_bears_l2416_241666

-- Definitions for the conditions
def four_small_panda_bears_eat_daily : ℕ := 25
def one_small_panda_bear_eats_daily : ℚ := 25 / 4
def each_bigger_panda_bear_eats_daily : ℚ := 40
def total_bamboo_eaten_weekly : ℕ := 2100
def total_bamboo_eaten_daily : ℚ := 2100 / 7

-- The theorem statement to prove
theorem how_many_bigger_panda_bears :
  ∃ B : ℚ, one_small_panda_bear_eats_daily * 4 + each_bigger_panda_bear_eats_daily * B = total_bamboo_eaten_daily := by
  sorry

end how_many_bigger_panda_bears_l2416_241666


namespace female_voters_percentage_is_correct_l2416_241682

def percentage_of_population_that_are_female_voters
  (female_percentage : ℝ)
  (voter_percentage_of_females : ℝ) : ℝ :=
  female_percentage * voter_percentage_of_females * 100

theorem female_voters_percentage_is_correct :
  percentage_of_population_that_are_female_voters 0.52 0.4 = 20.8 := by
  sorry

end female_voters_percentage_is_correct_l2416_241682


namespace temperature_or_daytime_not_sufficiently_high_l2416_241624

variable (T : ℝ) (Daytime Lively : Prop)
axiom h1 : (T ≥ 75 ∧ Daytime) → Lively
axiom h2 : ¬ Lively

theorem temperature_or_daytime_not_sufficiently_high : T < 75 ∨ ¬ Daytime :=
by
  -- proof steps
  sorry

end temperature_or_daytime_not_sufficiently_high_l2416_241624


namespace polar_circle_equation_l2416_241650

theorem polar_circle_equation (ρ θ : ℝ) (O pole : ℝ) (eq_line : ρ * Real.cos θ + ρ * Real.sin θ = 2) :
  (∃ ρ, ρ = 2 * Real.cos θ) :=
sorry

end polar_circle_equation_l2416_241650


namespace intersection_A_B_l2416_241699

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}
def intersection_of_A_and_B : Set ℕ := {0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = intersection_of_A_and_B :=
by
  sorry

end intersection_A_B_l2416_241699


namespace largest_integer_odd_divides_expression_l2416_241641

theorem largest_integer_odd_divides_expression (x : ℕ) (h_odd : x % 2 = 1) : 
    ∃ k, k = 384 ∧ ∀ m, m ∣ (8*x + 6) * (8*x + 10) * (4*x + 4) → m ≤ k :=
by {
  sorry
}

end largest_integer_odd_divides_expression_l2416_241641


namespace sum_of_squares_inequality_l2416_241620

theorem sum_of_squares_inequality (a b c : ℝ) (h : a + 2 * b + 3 * c = 4) : a^2 + b^2 + c^2 ≥ 8 / 7 := by
  sorry

end sum_of_squares_inequality_l2416_241620


namespace circle_radius_zero_l2416_241669

-- Theorem statement
theorem circle_radius_zero :
  ∀ (x y : ℝ), 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → 
  ∃ (c : ℝ) (r : ℝ), r = 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2 :=
by sorry

end circle_radius_zero_l2416_241669


namespace sqrt_sum_simplify_l2416_241662

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l2416_241662


namespace find_m_value_l2416_241601

noncomputable def is_solution (p q m : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (x^2 - m*x + m^2 - 19 = 0)

theorem find_m_value :
  let A := { x : ℝ | x^2 + 2 * x - 8 = 0 }
  let B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
  ∀ (C : ℝ → Prop), 
  (∃ x, B x ∧ C x) ∧ (¬ ∃ x, A x ∧ C x) → 
  (∃ m, C = { x : ℝ | x^2 - m * x + m^2 - 19 = 0 } ∧ m = -2) :=
by
  sorry

end find_m_value_l2416_241601


namespace nat_exponent_sum_eq_l2416_241661

theorem nat_exponent_sum_eq (n p q : ℕ) : n^p + n^q = n^2010 ↔ (n = 2 ∧ p = 2009 ∧ q = 2009) :=
by
  sorry

end nat_exponent_sum_eq_l2416_241661


namespace decreasing_interval_l2416_241631

def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative function
def f_prime (x : ℝ) : ℝ := 3*x^2 - 3

theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0 :=
by
  intro x h
  have h1: x^2 < 1 := by
    sorry
  have h2: 3*x^2 < 3 := by
    sorry
  have h3: 3*x^2 - 3 < 0 := by
    sorry
  exact h3

end decreasing_interval_l2416_241631


namespace average_writing_speed_time_to_write_10000_words_l2416_241613

-- Definitions based on the problem conditions
def total_words : ℕ := 60000
def total_hours : ℝ := 90.5
def writing_speed : ℝ := 663
def words_to_write : ℕ := 10000
def writing_time : ℝ := 15.08

-- Proposition that the average writing speed is 663 words per hour
theorem average_writing_speed :
  (total_words : ℝ) / total_hours = writing_speed :=
sorry

-- Proposition that the time to write 10,000 words at the given average speed is 15.08 hours
theorem time_to_write_10000_words :
  (words_to_write : ℝ) / writing_speed = writing_time :=
sorry

end average_writing_speed_time_to_write_10000_words_l2416_241613


namespace budget_circle_salaries_degrees_l2416_241696

theorem budget_circle_salaries_degrees :
  let transportation := 20
  let research_development := 9
  let utilities := 5
  let equipment := 4
  let supplies := 2
  let total_percent := 100
  let full_circle_degrees := 360
  let total_allocated_percent := transportation + research_development + utilities + equipment + supplies
  let salaries_percent := total_percent - total_allocated_percent
  let salaries_degrees := (salaries_percent * full_circle_degrees) / total_percent
  salaries_degrees = 216 :=
by
  sorry

end budget_circle_salaries_degrees_l2416_241696


namespace Geli_pushups_total_l2416_241655

variable (x : ℕ)
variable (total_pushups : ℕ)

theorem Geli_pushups_total (h : 10 + (10 + x) + (10 + 2 * x) = 45) : x = 5 :=
by
  sorry

end Geli_pushups_total_l2416_241655


namespace sales_revenue_nonnegative_l2416_241629

def revenue (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

theorem sales_revenue_nonnegative (x : ℝ) (hx : x = 9 ∨ x = 11) : revenue x ≥ 15950 :=
by
  cases hx
  case inl h₁ =>
    sorry -- calculation for x = 9
  case inr h₂ =>
    sorry -- calculation for x = 11

end sales_revenue_nonnegative_l2416_241629


namespace domain_of_function_l2416_241693

noncomputable def domain_is_valid (x z : ℝ) : Prop :=
  1 < x ∧ x < 2 ∧ (|x| - z) ≠ 0

theorem domain_of_function (x z : ℝ) : domain_is_valid x z :=
by
  sorry

end domain_of_function_l2416_241693


namespace petya_wrong_l2416_241686

theorem petya_wrong : ∃ (a b : ℕ), b^2 ∣ a^5 ∧ ¬ (b ∣ a^2) :=
by
  use 4
  use 32
  sorry

end petya_wrong_l2416_241686


namespace beads_counter_representation_l2416_241622

-- Given conditions
variable (a : ℕ) -- a is a natural number representing the beads in the tens place.
variable (h : a ≥ 0) -- Ensure a is non-negative since the number of beads cannot be negative.

-- The main statement to prove
theorem beads_counter_representation (a : ℕ) (h : a ≥ 0) : 10 * a + 4 = (10 * a) + 4 :=
by sorry

end beads_counter_representation_l2416_241622


namespace infinite_consecutive_pairs_l2416_241614

-- Define the relation
def related (x y : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 (x + y)), d = 0 ∨ d = 1

-- Define sets A and B
variable (A B : Set ℕ)

-- Define the conditions
axiom cond1 : ∀ a ∈ A, ∀ b ∈ B, related a b
axiom cond2 : ∀ c, (∀ a ∈ A, related c a) → c ∈ B
axiom cond3 : ∀ c, (∀ b ∈ B, related c b) → c ∈ A

-- Prove that one of the sets contains infinitely many pairs of consecutive numbers
theorem infinite_consecutive_pairs :
  (∃ a ∈ A, ∀ n : ℕ, a + n ∈ A ∧ a + n + 1 ∈ A) ∨ (∃ b ∈ B, ∀ n : ℕ, b + n ∈ B ∧ b + n + 1 ∈ B) :=
sorry

end infinite_consecutive_pairs_l2416_241614


namespace tailoring_business_days_l2416_241600

theorem tailoring_business_days
  (shirts_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (pants_per_day : ℕ)
  (fabric_per_pant : ℕ)
  (total_fabric : ℕ)
  (h1 : shirts_per_day = 3)
  (h2 : fabric_per_shirt = 2)
  (h3 : pants_per_day = 5)
  (h4 : fabric_per_pant = 5)
  (h5 : total_fabric = 93) :
  (total_fabric / (shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pant)) = 3 :=
by {
  sorry
}

end tailoring_business_days_l2416_241600


namespace square_of_99_l2416_241652

theorem square_of_99 : 99 * 99 = 9801 :=
by sorry

end square_of_99_l2416_241652


namespace additional_pairs_of_snakes_l2416_241654

theorem additional_pairs_of_snakes (total_snakes breeding_balls snakes_per_ball additional_snakes_per_pair : ℕ)
  (h1 : total_snakes = 36) 
  (h2 : breeding_balls = 3)
  (h3 : snakes_per_ball = 8) 
  (h4 : additional_snakes_per_pair = 2) :
  (total_snakes - (breeding_balls * snakes_per_ball)) / additional_snakes_per_pair = 6 :=
by
  sorry

end additional_pairs_of_snakes_l2416_241654


namespace binom_28_7_l2416_241607

theorem binom_28_7 (h1 : Nat.choose 26 3 = 2600) (h2 : Nat.choose 26 4 = 14950) (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 :=
by
  sorry

end binom_28_7_l2416_241607


namespace integer_solution_of_inequality_system_l2416_241664

theorem integer_solution_of_inequality_system :
  ∃ x : ℤ, (2 * (x : ℝ) ≤ 1) ∧ ((x : ℝ) + 2 > 1) ∧ (x = 0) :=
by
  sorry

end integer_solution_of_inequality_system_l2416_241664


namespace solve_for_x_l2416_241670

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end solve_for_x_l2416_241670


namespace gcf_60_90_150_l2416_241636

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end gcf_60_90_150_l2416_241636


namespace father_catches_up_l2416_241651

noncomputable def min_steps_to_catch_up : Prop :=
  let x := 30
  let father_steps := 5
  let xiaoming_steps := 8
  let distance_ratio := 2 / 5
  let xiaoming_headstart := 27
  ((xiaoming_headstart + (xiaoming_steps / father_steps) * x) / distance_ratio) = x

theorem father_catches_up : min_steps_to_catch_up :=
  by
  sorry

end father_catches_up_l2416_241651


namespace geometric_sequence_sum_l2416_241611

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l2416_241611


namespace gcd_of_44_54_74_l2416_241645

theorem gcd_of_44_54_74 : gcd (gcd 44 54) 74 = 2 :=
by
    sorry

end gcd_of_44_54_74_l2416_241645


namespace compute_expression_l2416_241604

section
variable (a : ℝ)

theorem compute_expression :
  (-a^2)^3 * a^3 = -a^9 :=
sorry
end

end compute_expression_l2416_241604


namespace second_trial_addition_amount_l2416_241605

variable (optimal_min optimal_max: ℝ) (phi: ℝ)

def method_618 (optimal_min optimal_max phi: ℝ) :=
  let x1 := optimal_min + (optimal_max - optimal_min) * phi
  let x2 := optimal_max + optimal_min - x1
  x2

theorem second_trial_addition_amount:
  optimal_min = 10 ∧ optimal_max = 110 ∧ phi = 0.618 →
  method_618 10 110 0.618 = 48.2 :=
by
  intro h
  simp [method_618, h]
  sorry

end second_trial_addition_amount_l2416_241605


namespace payment_amount_l2416_241603

/-- 
A certain debt will be paid in 52 installments from January 1 to December 31 of a certain year.
Each of the first 25 payments is to be a certain amount; each of the remaining payments is to be $100 more than each of the first payments.
The average (arithmetic mean) payment that will be made on the debt for the year is $551.9230769230769.
Prove that the amount of each of the first 25 payments is $500.
-/
theorem payment_amount (X : ℝ) 
  (h1 : 25 * X + 27 * (X + 100) = 52 * 551.9230769230769) :
  X = 500 :=
sorry

end payment_amount_l2416_241603
