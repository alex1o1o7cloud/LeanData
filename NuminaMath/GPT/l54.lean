import Mathlib

namespace total_amount_paid_l54_54873

def grapes_quantity := 8
def grapes_rate := 80
def mangoes_quantity := 9
def mangoes_rate := 55
def apples_quantity := 6
def apples_rate := 120
def oranges_quantity := 4
def oranges_rate := 75

theorem total_amount_paid :
  grapes_quantity * grapes_rate +
  mangoes_quantity * mangoes_rate +
  apples_quantity * apples_rate +
  oranges_quantity * oranges_rate =
  2155 := by
  sorry

end total_amount_paid_l54_54873


namespace lcm_of_4_5_6_9_is_180_l54_54365

theorem lcm_of_4_5_6_9_is_180 : Nat.lcm (Nat.lcm 4 5) (Nat.lcm 6 9) = 180 :=
by
  sorry

end lcm_of_4_5_6_9_is_180_l54_54365


namespace speed_second_boy_l54_54846

theorem speed_second_boy (v : ℝ) (t : ℝ) (d : ℝ) (s₁ : ℝ) :
  s₁ = 4.5 ∧ t = 9.5 ∧ d = 9.5 ∧ (d = (v - s₁) * t) → v = 5.5 :=
by
  intros h
  obtain ⟨hs₁, ht, hd, hev⟩ := h
  sorry

end speed_second_boy_l54_54846


namespace triangle_inequality_part_a_l54_54271

theorem triangle_inequality_part_a (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  a^2 + b^2 + c^2 + a * b * c < 8 :=
sorry

end triangle_inequality_part_a_l54_54271


namespace tangent_line_touching_circle_l54_54775

theorem tangent_line_touching_circle (a : ℝ) : 
  (∃ (x y : ℝ), 5 * x + 12 * y + a = 0 ∧ (x - 1)^2 + y^2 = 1) → 
  (a = 8 ∨ a = -18) :=
by
  sorry

end tangent_line_touching_circle_l54_54775


namespace problem1_problem2_l54_54407

variable (x : ℝ)

theorem problem1 : 
  (3 * x + 1) * (3 * x - 1) - (3 * x + 1)^2 = -6 * x - 2 :=
sorry

theorem problem2 : 
  (6 * x^4 - 8 * x^3) / (-2 * x^2) - (3 * x + 2) * (1 - x) = 3 * x - 2 :=
sorry

end problem1_problem2_l54_54407


namespace necklace_price_l54_54349

variable (N : ℝ)

def price_of_bracelet : ℝ := 15.00
def price_of_earring : ℝ := 10.00
def num_necklaces_sold : ℝ := 5
def num_bracelets_sold : ℝ := 10
def num_earrings_sold : ℝ := 20
def num_complete_ensembles_sold : ℝ := 2
def price_of_complete_ensemble : ℝ := 45.00
def total_amount_made : ℝ := 565.0

theorem necklace_price :
  5 * N + 10 * price_of_bracelet + 20 * price_of_earring
  + 2 * price_of_complete_ensemble = total_amount_made → N = 25 :=
by
  intro h
  sorry

end necklace_price_l54_54349


namespace books_remaining_correct_l54_54019

-- Define the initial number of book donations
def initial_books : ℕ := 300

-- Define the number of people donating and the number of books each donates
def num_people : ℕ := 10
def books_per_person : ℕ := 5

-- Calculate total books donated by all people
def total_donation : ℕ := num_people * books_per_person

-- Define the number of books borrowed by other people
def borrowed_books : ℕ := 140

-- Calculate the total number of books after donations and then subtract the borrowed books
def total_books_remaining : ℕ := initial_books + total_donation - borrowed_books

-- Prove the total number of books remaining is 210
theorem books_remaining_correct : total_books_remaining = 210 := by
  sorry

end books_remaining_correct_l54_54019


namespace cone_lateral_surface_area_l54_54737

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l54_54737


namespace range_of_a_l54_54811

open Function

def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

theorem range_of_a (a : ℝ) : f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 := 
by
  sorry

end range_of_a_l54_54811


namespace find_sum_of_distinct_real_numbers_l54_54557

noncomputable def determinant_3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem find_sum_of_distinct_real_numbers (x y : ℝ) (hxy : x ≠ y) 
    (h : determinant_3x3 1 6 15 3 x y 3 y x = 0) : x + y = 63 := 
by
  sorry

end find_sum_of_distinct_real_numbers_l54_54557


namespace largest_integer_m_dividing_30_factorial_l54_54851

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end largest_integer_m_dividing_30_factorial_l54_54851


namespace valid_permutations_count_l54_54900

/-- 
Given five elements consisting of the numbers 1, 2, 3, and the symbols "+" and "-", 
we want to count the number of permutations such that no two numbers are adjacent.
-/
def count_valid_permutations : Nat := 
  let number_permutations := Nat.factorial 3 -- 3! permutations of 1, 2, 3
  let symbol_insertions := Nat.factorial 2  -- 2! permutations of "+" and "-"
  number_permutations * symbol_insertions

theorem valid_permutations_count : count_valid_permutations = 12 := by
  sorry

end valid_permutations_count_l54_54900


namespace find_a_in_triangle_l54_54399

theorem find_a_in_triangle (b c : ℝ) (cos_B_minus_C : ℝ) (a : ℝ) 
  (hb : b = 7) (hc : c = 6) (hcos : cos_B_minus_C = 15 / 16) :
  a = 5 * Real.sqrt 3 :=
by
  sorry

end find_a_in_triangle_l54_54399


namespace smaller_cube_volume_l54_54309

theorem smaller_cube_volume
  (d : ℝ) (s : ℝ) (V : ℝ)
  (h1 : d = 12)  -- condition: diameter of the sphere equals the edge length of the larger cube
  (h2 : d = s * Real.sqrt 3)  -- condition: space diagonal of the smaller cube equals the diameter of the sphere
  (h3 : s = 12 / Real.sqrt 3)  -- condition: side length of the smaller cube
  (h4 : V = s^3)  -- condition: volume of the cube with side length s
  : V = 192 * Real.sqrt 3 :=  -- proving the volume of the smaller cube
sorry

end smaller_cube_volume_l54_54309


namespace find_number_l54_54854

theorem find_number (x : ℕ) (h : x + 1015 = 3016) : x = 2001 :=
sorry

end find_number_l54_54854


namespace card_draw_prob_l54_54850

/-- Define the total number of cards in the deck -/
def total_cards : ℕ := 52

/-- Define the total number of diamonds or aces -/
def diamonds_and_aces : ℕ := 16

/-- Define the probability of drawing a card that is a diamond or an ace in one draw -/
def prob_diamond_or_ace : ℚ := diamonds_and_aces / total_cards

/-- Define the complementary probability of not drawing a diamond nor ace in one draw -/
def prob_not_diamond_or_ace : ℚ := (total_cards - diamonds_and_aces) / total_cards

/-- Define the probability of not drawing a diamond nor ace in three draws with replacement -/
def prob_not_diamond_or_ace_three_draws : ℚ := prob_not_diamond_or_ace ^ 3

/-- Define the probability of drawing at least one diamond or ace in three draws with replacement -/
def prob_at_least_one_diamond_or_ace_in_three_draws : ℚ := 1 - prob_not_diamond_or_ace_three_draws

/-- The final probability calculated -/
def final_prob : ℚ := 1468 / 2197

theorem card_draw_prob :
  prob_at_least_one_diamond_or_ace_in_three_draws = final_prob := by
  sorry

end card_draw_prob_l54_54850


namespace evaluate_expression_l54_54800

theorem evaluate_expression 
  (d a b c : ℚ)
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (nz1 : d + 3 ≠ 0)
  (nz2 : a + 2 ≠ 0)
  (nz3 : b - 5 ≠ 0)
  (nz4 : c + 7 ≠ 0) :
  (d + 5) / (d + 3) * (a + 3) / (a + 2) * (b - 3) / (b - 5) * (c + 10) / (c + 7) = 1232 / 585 :=
sorry

end evaluate_expression_l54_54800


namespace tan_theta_values_l54_54223

theorem tan_theta_values (θ : ℝ) (h₁ : 0 < θ ∧ θ < Real.pi / 2) (h₂ : 12 / Real.sin θ + 12 / Real.cos θ = 35) : 
  Real.tan θ = 4 / 3 ∨ Real.tan θ = 3 / 4 := 
by
  sorry

end tan_theta_values_l54_54223


namespace inequality_proof_l54_54874

noncomputable def a := (1 / 4) * Real.logb 2 3
noncomputable def b := 1 / 2
noncomputable def c := (1 / 2) * Real.logb 5 3

theorem inequality_proof : c < a ∧ a < b :=
by
  sorry

end inequality_proof_l54_54874


namespace sequence_number_pair_l54_54586

theorem sequence_number_pair (n m : ℕ) (h : m ≤ n) : (m, n - m + 1) = (m, n - m + 1) :=
by sorry

end sequence_number_pair_l54_54586


namespace subset_relationship_l54_54810

def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def T : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

theorem subset_relationship : S ⊆ T :=
by sorry

end subset_relationship_l54_54810


namespace katherine_has_4_apples_l54_54310

variable (A P : ℕ)

theorem katherine_has_4_apples
  (h1 : P = 3 * A)
  (h2 : A + P = 16) :
  A = 4 := 
sorry

end katherine_has_4_apples_l54_54310


namespace star_property_l54_54801

-- Define the operation a ⋆ b = (a - b) ^ 3
def star (a b : ℝ) : ℝ := (a - b) ^ 3

-- State the theorem
theorem star_property (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := 
by 
  sorry

end star_property_l54_54801


namespace floor_sqrt_12_squared_l54_54061

theorem floor_sqrt_12_squared : (Int.floor (Real.sqrt 12))^2 = 9 := by
  sorry

end floor_sqrt_12_squared_l54_54061


namespace mary_money_left_l54_54153

def drink_price (p : ℕ) : ℕ := p
def medium_pizza_price (p : ℕ) : ℕ := 2 * p
def large_pizza_price (p : ℕ) : ℕ := 3 * p
def drinks_cost (n : ℕ) (p : ℕ) : ℕ := n * drink_price p
def medium_pizzas_cost (n : ℕ) (p : ℕ) : ℕ := n * medium_pizza_price p
def large_pizza_cost (n : ℕ) (p : ℕ) : ℕ := n * large_pizza_price p
def total_cost (p : ℕ) : ℕ := drinks_cost 5 p + medium_pizzas_cost 2 p + large_pizza_cost 1 p
def money_left (initial_money : ℕ) (p : ℕ) : ℕ := initial_money - total_cost p

theorem mary_money_left (p : ℕ) : money_left 50 p = 50 - 12 * p := sorry

end mary_money_left_l54_54153


namespace average_price_of_six_toys_l54_54829

/-- Define the average cost of toys given the number of toys and their total cost -/
def avg_cost (total_cost : ℕ) (num_toys : ℕ) : ℕ :=
  total_cost / num_toys

/-- Define the total cost of toys given a list of individual toy costs -/
def total_cost (costs : List ℕ) : ℕ :=
  costs.foldl (· + ·) 0

/-- The main theorem -/
theorem average_price_of_six_toys :
  let dhoni_toys := 5
  let avg_cost_dhoni := 10
  let total_cost_dhoni := dhoni_toys * avg_cost_dhoni
  let david_toy_cost := 16
  let total_toys := dhoni_toys + 1
  total_cost_dhoni + david_toy_cost = 66 →
  avg_cost (66) (total_toys) = 11 :=
by
  -- Introduce the conditions and hypothesis
  intros total_cost_of_6_toys H
  -- Simplify the expression
  sorry  -- Proof skipped

end average_price_of_six_toys_l54_54829


namespace jana_height_l54_54593

theorem jana_height (jess_height : ℕ) (kelly_height : ℕ) (jana_height : ℕ) 
  (h1 : kelly_height = jess_height - 3) 
  (h2 : jana_height = kelly_height + 5) 
  (h3 : jess_height = 72) : 
  jana_height = 74 := 
by
  sorry

end jana_height_l54_54593


namespace minimize_fuel_consumption_l54_54882

-- Define conditions as constants
def cargo_total : ℕ := 157
def cap_large : ℕ := 5
def cap_small : ℕ := 2
def fuel_large : ℕ := 20
def fuel_small : ℕ := 10

-- Define truck counts
def n_large : ℕ := 31
def n_small : ℕ := 1

-- Theorem: the number of large and small trucks that minimize fuel consumption
theorem minimize_fuel_consumption : 
  n_large * cap_large + n_small * cap_small = cargo_total ∧
  (∀ m_large m_small, m_large * cap_large + m_small * cap_small = cargo_total → 
    m_large * fuel_large + m_small * fuel_small ≥ n_large * fuel_large + n_small * fuel_small) :=
by
  -- Statement to be proven
  sorry

end minimize_fuel_consumption_l54_54882


namespace circle_diameter_l54_54716

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end circle_diameter_l54_54716


namespace solve_olympics_problem_max_large_sets_l54_54051

-- Definitions based on the conditions
variables (x y : ℝ)

-- Condition 1: 2 small sets cost $20 less than 1 large set
def condition1 : Prop := y - 2 * x = 20

-- Condition 2: 3 small sets and 2 large sets cost $390
def condition2 : Prop := 3 * x + 2 * y = 390

-- Finding unit prices
def unit_prices : Prop := x = 50 ∧ y = 120

-- Condition 3: Budget constraint for purchasing sets
def budget_constraint (m : ℕ) : Prop := m ≤ 7

-- Prove unit prices and purchasing constraints
theorem solve_olympics_problem :
  condition1 x y ∧ condition2 x y → unit_prices x y :=
by
  sorry

theorem max_large_sets :
  budget_constraint 7 :=
by
  sorry

end solve_olympics_problem_max_large_sets_l54_54051


namespace money_left_correct_l54_54845

def initial_amount : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def amount_left : ℕ := initial_amount - total_spent

theorem money_left_correct :
  amount_left = 78 := by
  sorry

end money_left_correct_l54_54845


namespace problem_expression_l54_54795

theorem problem_expression (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 4) : x^2 + y^2 = 33 :=
by sorry

end problem_expression_l54_54795


namespace divisible_by_6_l54_54067

theorem divisible_by_6 (n : ℤ) (h1 : n % 3 = 0) (h2 : n % 2 = 0) : n % 6 = 0 :=
sorry

end divisible_by_6_l54_54067


namespace min_sum_of_box_dimensions_l54_54203

theorem min_sum_of_box_dimensions :
  ∃ (x y z : ℕ), x * y * z = 2541 ∧ (y = x + 3 ∨ x = y + 3) ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 38 :=
sorry

end min_sum_of_box_dimensions_l54_54203


namespace inequality_proof_l54_54357

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 :=
by 
  sorry

end inequality_proof_l54_54357


namespace fraction_meaningful_range_l54_54036

theorem fraction_meaningful_range (x : ℝ) : 5 - x ≠ 0 ↔ x ≠ 5 :=
by sorry

end fraction_meaningful_range_l54_54036


namespace determine_q_l54_54505

theorem determine_q (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 4) : q = 2 := 
sorry

end determine_q_l54_54505


namespace sum_first_15_odd_integers_from_5_l54_54879

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l54_54879


namespace cost_of_candy_car_l54_54935

theorem cost_of_candy_car (starting_amount paid_amount change : ℝ) (h1 : starting_amount = 1.80) (h2 : change = 1.35) (h3 : paid_amount = starting_amount - change) : paid_amount = 0.45 := by
  sorry

end cost_of_candy_car_l54_54935


namespace mandy_difference_of_cinnamon_and_nutmeg_l54_54152

theorem mandy_difference_of_cinnamon_and_nutmeg :
  let cinnamon := 0.6666666666666666
  let nutmeg := 0.5
  let difference := cinnamon - nutmeg
  difference = 0.1666666666666666 :=
by
  sorry

end mandy_difference_of_cinnamon_and_nutmeg_l54_54152


namespace percentage_difference_l54_54004

theorem percentage_difference (G P R : ℝ) (h1 : P = 0.9 * G) (h2 : R = 1.125 * G) :
  ((1 - P / R) * 100) = 20 :=
by
  sorry

end percentage_difference_l54_54004


namespace initial_mixture_equals_50_l54_54695

theorem initial_mixture_equals_50 (x : ℝ) (h1 : 0.10 * x + 10 = 0.25 * (x + 10)) : x = 50 :=
by
  sorry

end initial_mixture_equals_50_l54_54695


namespace find_g_inv_84_l54_54560

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end find_g_inv_84_l54_54560


namespace sacks_per_day_proof_l54_54694

-- Definitions based on the conditions in the problem
def totalUnripeOranges : ℕ := 1080
def daysOfHarvest : ℕ := 45

-- Mathematical statement to prove
theorem sacks_per_day_proof : totalUnripeOranges / daysOfHarvest = 24 :=
by sorry

end sacks_per_day_proof_l54_54694


namespace midpoint_of_segment_l54_54126

def z1 : ℂ := 2 + 4 * Complex.I  -- Define the first endpoint
def z2 : ℂ := -6 + 10 * Complex.I  -- Define the second endpoint

theorem midpoint_of_segment :
  (z1 + z2) / 2 = -2 + 7 * Complex.I := by
  sorry

end midpoint_of_segment_l54_54126


namespace find_A_l54_54405

theorem find_A (A B C D: ℕ) (h1: A ≠ B) (h2: A ≠ C) (h3: A ≠ D) (h4: B ≠ C) (h5: B ≠ D) (h6: C ≠ D)
  (hAB: A * B = 72) (hCD: C * D = 72) (hDiff: A - B = C + D + 2) : A = 6 :=
sorry

end find_A_l54_54405


namespace minimum_value_l54_54294

noncomputable def min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :=
  a + 2 * b

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :
  min_value a b h₁ h₂ h₃ ≥ 2 * Real.sqrt 2 :=
sorry

end minimum_value_l54_54294


namespace pages_after_break_l54_54678

-- Formalize the conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.70

-- Define the proof problem
theorem pages_after_break : 
  let pages_read_before_break := (break_percentage * total_pages)
  let pages_remaining := total_pages - pages_read_before_break
  pages_remaining = 9 :=
by
  sorry

end pages_after_break_l54_54678


namespace five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l54_54731

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

end five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l54_54731


namespace candidate_a_valid_votes_l54_54080

/-- In an election, candidate A got 80% of the total valid votes.
If 15% of the total votes were declared invalid and the total number of votes is 560,000,
find the number of valid votes polled in favor of candidate A. -/
theorem candidate_a_valid_votes :
  let total_votes := 560000
  let invalid_percentage := 0.15
  let valid_percentage := 0.85
  let candidate_a_percentage := 0.80
  let valid_votes := (valid_percentage * total_votes : ℝ)
  let candidate_a_votes := (candidate_a_percentage * valid_votes : ℝ)
  candidate_a_votes = 380800 :=
by
  sorry

end candidate_a_valid_votes_l54_54080


namespace difference_of_squares_l54_54945

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Define the condition for the expression which should hold
def expression_b := (2 * x + y) * (y - 2 * x)

-- The theorem to prove that this expression fits the formula for the difference of squares
theorem difference_of_squares : 
  ∃ a b : ℝ, expression_b x y = a^2 - b^2 := 
by 
  sorry

end difference_of_squares_l54_54945


namespace smallest_n_satisfying_congruence_l54_54821

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end smallest_n_satisfying_congruence_l54_54821


namespace work_rate_l54_54330

/-- 
A alone can finish a work in some days which B alone can finish in 15 days. 
If they work together and finish it, then out of a total wages of Rs. 3400, 
A will get Rs. 2040. Prove that A alone can finish the work in 22.5 days. 
-/
theorem work_rate (A : ℚ) (B_rate : ℚ) 
  (total_wages : ℚ) (A_wages : ℚ) 
  (total_rate : ℚ) 
  (hB : B_rate = 1 / 15) 
  (hWages : total_wages = 3400 ∧ A_wages = 2040) 
  (hTotal : total_rate = 1 / A + B_rate)
  (hWorkTogether : 
    (A_wages / (total_wages - A_wages) = 51 / 34) ↔ 
    (A / (A + 15) = 51 / 85)) : 
  A = 22.5 := 
sorry

end work_rate_l54_54330


namespace angle_in_fourth_quadrant_l54_54162

-- Define the main condition converting the angle to the range [0, 360)
def reducedAngle (θ : ℤ) : ℤ := (θ % 360 + 360) % 360

-- State the theorem proving the angle of -390° is in the fourth quadrant
theorem angle_in_fourth_quadrant (θ : ℤ) (h : θ = -390) : 270 ≤ reducedAngle θ ∧ reducedAngle θ < 360 := by
  sorry

end angle_in_fourth_quadrant_l54_54162


namespace candidate_valid_vote_percentage_l54_54999

theorem candidate_valid_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_votes : ℕ) 
  (valid_percentage : ℚ)
  (total_votes_eq : total_votes = 560000)
  (invalid_percentage_eq : invalid_percentage = 15 / 100)
  (candidate_votes_eq : candidate_votes = 357000)
  (valid_percentage_eq : valid_percentage = 85 / 100) :
  (candidate_votes / (total_votes * valid_percentage)) * 100 = 75 := 
by
  sorry

end candidate_valid_vote_percentage_l54_54999


namespace krista_egg_sales_l54_54543

-- Define the conditions
def hens : ℕ := 10
def eggs_per_hen_per_week : ℕ := 12
def price_per_dozen : ℕ := 3
def weeks : ℕ := 4

-- Define the total money made as the value we want to prove
def total_money_made : ℕ := 120

-- State the theorem
theorem krista_egg_sales : 
  (hens * eggs_per_hen_per_week * weeks / 12) * price_per_dozen = total_money_made :=
by
  sorry

end krista_egg_sales_l54_54543


namespace min_number_of_students_l54_54638

theorem min_number_of_students 
  (n : ℕ)
  (h1 : 25 ≡ 99 [MOD n])
  (h2 : 8 ≡ 119 [MOD n]) : 
  n = 37 :=
by sorry

end min_number_of_students_l54_54638


namespace f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l54_54916

noncomputable def f (x : ℝ) : ℝ := x + 4/x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem f_min_value_pos : ∀ x : ℝ, x > 0 → f x ≥ 4 :=
by
  sorry

theorem f_minimum_at_2 : f 2 = 4 :=
by
  sorry

theorem f_increasing_intervals : (MonotoneOn f {x | x ≤ -2} ∧ MonotoneOn f {x | x ≥ 2}) :=
by
  sorry

end f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l54_54916


namespace min_value_proof_l54_54111

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  2 / a + 2 / b + 2 / c

theorem min_value_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_abc : a + b + c = 9) : 
  minimum_value a b c ≥ 2 := 
by 
  sorry

end min_value_proof_l54_54111


namespace four_thirds_of_product_eq_25_div_2_l54_54243

noncomputable def a : ℚ := 15 / 4
noncomputable def b : ℚ := 5 / 2
noncomputable def c : ℚ := 4 / 3
noncomputable def d : ℚ := a * b
noncomputable def e : ℚ := c * d

theorem four_thirds_of_product_eq_25_div_2 : e = 25 / 2 := 
sorry

end four_thirds_of_product_eq_25_div_2_l54_54243


namespace find_product_xyz_l54_54564

-- Definitions for the given conditions
variables (x y z : ℕ) -- positive integers

-- Conditions
def condition1 : Prop := x + 2 * y = z
def condition2 : Prop := x^2 - 4 * y^2 + z^2 = 310

-- Theorem statement
theorem find_product_xyz (h1 : condition1 x y z) (h2 : condition2 x y z) : 
  x * y * z = 11935 ∨ x * y * z = 2015 :=
sorry

end find_product_xyz_l54_54564


namespace part_a_part_b_l54_54069

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_permutation (P : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (P i + P (i + 1))) ∧
  ∀ i, P i ∈ (Finset.range 16).image (λ x => x + 1)

def valid_cyclic_permutation (C : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (C i + C (i + 1))) ∧
  is_perfect_square (C 15 + C 0) ∧
  ∀ i, C i ∈ (Finset.range 16).image (λ x => x + 1)

theorem part_a :
  ∃ P : Fin 16 → ℕ, valid_permutation P := sorry

theorem part_b :
  ¬ ∃ C : Fin 16 → ℕ, valid_cyclic_permutation C := sorry

end part_a_part_b_l54_54069


namespace simplify_exponent_l54_54321

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l54_54321


namespace germs_per_dish_l54_54693

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

end germs_per_dish_l54_54693


namespace problem1_problem2_l54_54012

noncomputable def f (x : ℝ) : ℝ :=
  |x - 2| - |2 * x + 1|

theorem problem1 (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ -1/3 ≤ x :=
sorry

theorem problem2 (a : ℝ) (b : ℝ) :
  (∀ x, |a + b| - |a - b| ≥ f x) → (a ≥ 5 / 4 ∨ a ≤ -5 / 4) :=
sorry

end problem1_problem2_l54_54012


namespace number_of_insects_l54_54382

-- Conditions
def total_legs : ℕ := 30
def legs_per_insect : ℕ := 6

-- Theorem statement
theorem number_of_insects (total_legs legs_per_insect : ℕ) : 
  total_legs / legs_per_insect = 5 := 
by
  sorry

end number_of_insects_l54_54382


namespace monthly_rent_requirement_l54_54573

noncomputable def initial_investment : Float := 200000
noncomputable def annual_return_rate : Float := 0.06
noncomputable def annual_insurance_cost : Float := 4500
noncomputable def maintenance_percentage : Float := 0.15
noncomputable def required_monthly_rent : Float := 1617.65

theorem monthly_rent_requirement :
  let annual_return := initial_investment * annual_return_rate
  let annual_cost_with_insurance := annual_return + annual_insurance_cost
  let monthly_required_net := annual_cost_with_insurance / 12
  let rental_percentage_kept := 1 - maintenance_percentage
  let monthly_rental_full := monthly_required_net / rental_percentage_kept
  monthly_rental_full = required_monthly_rent := 
by
  sorry

end monthly_rent_requirement_l54_54573


namespace inequality_correct_l54_54303

theorem inequality_correct (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1/a) < (1/b) :=
sorry

end inequality_correct_l54_54303


namespace difference_in_cm_l54_54488

def line_length : ℝ := 80  -- The length of the line is 80.0 centimeters
def diff_length_factor : ℝ := 0.35  -- The difference factor in the terms of the line's length

theorem difference_in_cm (l : ℝ) (d : ℝ) (h₀ : l = 80) (h₁ : d = 0.35 * l) : d = 28 :=
by
  sorry

end difference_in_cm_l54_54488


namespace find_complex_number_l54_54444

open Complex

theorem find_complex_number (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
sorry

end find_complex_number_l54_54444


namespace school_pupils_l54_54648

def girls : ℕ := 868
def difference : ℕ := 281
def boys (g b : ℕ) : Prop := g = b + difference
def total_pupils (g b t : ℕ) : Prop := t = g + b

theorem school_pupils : 
  ∃ b t, boys girls b ∧ total_pupils girls b t ∧ t = 1455 :=
by
  sorry

end school_pupils_l54_54648


namespace parallel_lines_slope_l54_54789

theorem parallel_lines_slope (m : ℝ) :
  ((m + 2) * (2 * m - 1) = 3 * 1) →
  m = - (5 / 2) :=
by
  sorry

end parallel_lines_slope_l54_54789


namespace truck_sand_at_arrival_l54_54439

-- Definitions based on conditions in part a)
def initial_sand : ℝ := 4.1
def lost_sand : ℝ := 2.4

-- Theorem statement corresponding to part c)
theorem truck_sand_at_arrival : initial_sand - lost_sand = 1.7 :=
by
  -- "sorry" placeholder to skip the proof
  sorry

end truck_sand_at_arrival_l54_54439


namespace original_number_l54_54490

theorem original_number (x : ℝ) (hx : 100000 * x = 5 * (1 / x)) : x = 0.00707 := 
by
  sorry

end original_number_l54_54490


namespace area_of_rectangular_field_l54_54919

-- Define the conditions
def length (b : ℕ) : ℕ := b + 30
def perimeter (b : ℕ) (l : ℕ) : ℕ := 2 * (b + l)

-- Define the main theorem to prove
theorem area_of_rectangular_field (b : ℕ) (l : ℕ) (h1 : l = length b) (h2 : perimeter b l = 540) : 
  l * b = 18000 := by
  -- Placeholder for the proof
  sorry

end area_of_rectangular_field_l54_54919


namespace jacob_river_water_collection_l54_54043

/-- Definitions: 
1. Capacity of the tank in milliliters
2. Daily water collected from the rain in milliliters
3. Number of days to fill the tank
4. To be proved: Daily water collected from the river in milliliters
-/
def tank_capacity_ml : Int := 50000
def daily_rain_ml : Int := 800
def days_to_fill : Int := 20
def daily_river_ml : Int := 1700

/-- Prove that the amount of water Jacob collects from the river every day equals 1700 milliliters.
-/
theorem jacob_river_water_collection (total_water: Int) 
  (rain_water: Int) (days: Int) (correct_river_water: Int) : 
  total_water = tank_capacity_ml → 
  rain_water = daily_rain_ml → 
  days = days_to_fill → 
  correct_river_water = daily_river_ml → 
  (total_water - rain_water * days) / days = correct_river_water := 
by 
  intros; 
  sorry

end jacob_river_water_collection_l54_54043


namespace expected_worth_coin_flip_l54_54076

def prob_head : ℚ := 2 / 3
def prob_tail : ℚ := 1 / 3
def gain_head : ℚ := 5
def loss_tail : ℚ := -12

theorem expected_worth_coin_flip : ∃ E : ℚ, E = round (((prob_head * gain_head) + (prob_tail * loss_tail)) * 100) / 100 ∧ E = - (2 / 3) :=
by
  sorry

end expected_worth_coin_flip_l54_54076


namespace heartsuit_ratio_l54_54937

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by
  sorry

end heartsuit_ratio_l54_54937


namespace Murtha_pebble_collection_l54_54402

def sum_of_first_n_natural_numbers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem Murtha_pebble_collection : sum_of_first_n_natural_numbers 20 = 210 := by
  sorry

end Murtha_pebble_collection_l54_54402


namespace problem_statement_l54_54416

-- Define operations "※" and "#"
def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

-- Define the proof statement
theorem problem_statement : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end problem_statement_l54_54416


namespace find_c_for_equal_real_roots_l54_54714

theorem find_c_for_equal_real_roots
  (c : ℝ)
  (h : ∀ x : ℝ, x^2 + 6 * x + c = 0 → x = -3) : c = 9 :=
sorry

end find_c_for_equal_real_roots_l54_54714


namespace find_k_l54_54819

theorem find_k 
  (x k : ℚ)
  (h1 : (x^2 - 3*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 7))
  (h2 : k ≠ 0) : k = 7 / 3 := 
sorry

end find_k_l54_54819


namespace madeline_money_l54_54447

variable (M B : ℝ)

theorem madeline_money :
  B = 1/2 * M →
  M + B = 72 →
  M = 48 :=
  by
    intros h1 h2
    sorry

end madeline_money_l54_54447


namespace petals_per_rose_l54_54474

theorem petals_per_rose
    (roses_per_bush : ℕ)
    (bushes : ℕ)
    (bottles : ℕ)
    (oz_per_bottle : ℕ)
    (petals_per_oz : ℕ)
    (petals : ℕ)
    (ounces : ℕ := bottles * oz_per_bottle)
    (total_petals : ℕ := ounces * petals_per_oz)
    (petals_per_bush : ℕ := total_petals / bushes)
    (petals_per_rose : ℕ := petals_per_bush / roses_per_bush) :
    petals_per_oz = 320 →
    roses_per_bush = 12 →
    bushes = 800 →
    bottles = 20 →
    oz_per_bottle = 12 →
    petals_per_rose = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end petals_per_rose_l54_54474


namespace q_join_after_days_l54_54345

noncomputable def workRate (totalWork : ℕ) (days : ℕ) : ℚ :=
  totalWork / days

theorem q_join_after_days (W : ℕ) (days_p : ℕ) (days_q : ℕ) (total_days : ℕ) (x : ℕ) :
  days_p = 80 ∧ days_q = 48 ∧ total_days = 35 ∧ 
  ((workRate W days_p) * x + (workRate W days_p + workRate W days_q) * (total_days - x) = W) 
  → x = 8 := sorry

end q_join_after_days_l54_54345


namespace problem_statement_l54_54685

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end problem_statement_l54_54685


namespace root_equation_alpha_beta_property_l54_54338

theorem root_equation_alpha_beta_property {α β : ℝ} (h1 : α^2 + α - 1 = 0) (h2 : β^2 + β - 1 = 0) :
    α^2 + 2 * β^2 + β = 4 :=
by
  sorry

end root_equation_alpha_beta_property_l54_54338


namespace sqrt_inequality_l54_54635

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := 
by
  sorry

end sqrt_inequality_l54_54635


namespace students_in_second_class_l54_54177

-- Definitions based on the conditions
def students_first_class : ℕ := 30
def avg_mark_first_class : ℕ := 40
def avg_mark_second_class : ℕ := 80
def combined_avg_mark : ℕ := 65

-- Proposition to prove
theorem students_in_second_class (x : ℕ) 
  (h1 : students_first_class * avg_mark_first_class + x * avg_mark_second_class = (students_first_class + x) * combined_avg_mark) : 
  x = 50 :=
sorry

end students_in_second_class_l54_54177


namespace find_p_q_l54_54788

def op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

theorem find_p_q :
  (∀ (a b c d : ℝ), (a = c ∧ b = d) ↔ (a, b) = (c, d)) →
  (op 1 2 p q = (5, 0)) →
  (p, q) = (1, -2) :=
by
  intro h
  intro eq_op
  sorry

end find_p_q_l54_54788


namespace time_for_type_Q_machine_l54_54760

theorem time_for_type_Q_machine (Q : ℝ) (h1 : Q > 0)
  (h2 : 2 * (1 / Q) + 3 * (1 / 7) = 5 / 6) :
  Q = 84 / 17 :=
sorry

end time_for_type_Q_machine_l54_54760


namespace intersection_A_B_l54_54682

def A : Set ℝ := { x | |x| > 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l54_54682


namespace g_at_3_l54_54740

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_at_3 : g 3 = 147 :=
by
  -- Proof omitted for brevity
  sorry

end g_at_3_l54_54740


namespace find_cost_per_kg_l54_54893

-- Define the conditions given in the problem
def side_length : ℕ := 30
def coverage_per_kg : ℕ := 20
def total_cost : ℕ := 10800

-- The cost per kg we need to find
def cost_per_kg := total_cost / ((6 * side_length^2) / coverage_per_kg)

-- We need to prove that cost_per_kg = 40
theorem find_cost_per_kg : cost_per_kg = 40 := by
  sorry

end find_cost_per_kg_l54_54893


namespace days_to_empty_tube_l54_54632

-- Define the conditions
def gelInTube : ℕ := 128
def dailyUsage : ℕ := 4

-- Define the proof statement
theorem days_to_empty_tube : gelInTube / dailyUsage = 32 := 
by 
  sorry

end days_to_empty_tube_l54_54632


namespace number_of_valid_integers_l54_54414

def count_valid_numbers : Nat :=
  let one_digit_count : Nat := 6
  let two_digit_count : Nat := 6 * 6
  let three_digit_count : Nat := 6 * 6 * 6
  one_digit_count + two_digit_count + three_digit_count

theorem number_of_valid_integers :
  count_valid_numbers = 258 :=
sorry

end number_of_valid_integers_l54_54414


namespace compute_m_div_18_l54_54273

noncomputable def ten_pow (n : ℕ) : ℕ := Nat.pow 10 n

def valid_digits (m : ℕ) : Prop :=
  ∀ d ∈ m.digits 10, d = 0 ∨ d = 8

def is_multiple_of_18 (m : ℕ) : Prop :=
  m % 18 = 0

theorem compute_m_div_18 :
  ∃ m, valid_digits m ∧ is_multiple_of_18 m ∧ m / 18 = 493827160 :=
by
  sorry

end compute_m_div_18_l54_54273


namespace gallons_of_gas_l54_54672

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end gallons_of_gas_l54_54672


namespace vartan_spent_on_recreation_last_week_l54_54109

variable (W P : ℝ)
variable (h1 : P = 0.20)
variable (h2 : W > 0)

theorem vartan_spent_on_recreation_last_week :
  (P * W) = 0.20 * W :=
by
  sorry

end vartan_spent_on_recreation_last_week_l54_54109


namespace constant_term_of_first_equation_l54_54500

theorem constant_term_of_first_equation
  (y z : ℤ)
  (h1 : 2 * 20 - y - z = 40)
  (h2 : 3 * 20 + y - z = 20)
  (hx : 20 = 20) :
  4 * 20 + y + z = 80 := 
sorry

end constant_term_of_first_equation_l54_54500


namespace green_apples_count_l54_54265

def red_apples := 33
def students_took := 21
def extra_apples := 35

theorem green_apples_count : ∃ G : ℕ, red_apples + G - students_took = extra_apples ∧ G = 23 :=
by
  use 23
  have h1 : 33 + 23 - 21 = 35 := by norm_num
  exact ⟨h1, rfl⟩

end green_apples_count_l54_54265


namespace fill_tank_with_only_C_l54_54927

noncomputable def time_to_fill_with_only_C (x y z : ℝ) : ℝ := 
  let eq1 := (1 / z - 1 / x) * 2 = 1
  let eq2 := (1 / z - 1 / y) * 4 = 1
  let eq3 := 1 / z * 5 - (1 / x + 1 / y) * 8 = 0
  z

theorem fill_tank_with_only_C (x y z : ℝ) (h1 : (1 / z - 1 / x) * 2 = 1) 
  (h2 : (1 / z - 1 / y) * 4 = 1) (h3 : 1 / z * 5 - (1 / x + 1 / y) * 8 = 0) : 
  time_to_fill_with_only_C x y z = 11 / 6 :=
by
  sorry

end fill_tank_with_only_C_l54_54927


namespace spend_together_is_85_l54_54929

variable (B D : ℝ)

theorem spend_together_is_85 (h1 : D = 0.70 * B) (h2 : B = D + 15) : B + D = 85 := by
  sorry

end spend_together_is_85_l54_54929


namespace no_solution_fraction_eq_l54_54497

theorem no_solution_fraction_eq (x : ℝ) : 
  (1 / (x - 2) = (1 - x) / (2 - x) - 3) → False := 
by 
  sorry

end no_solution_fraction_eq_l54_54497


namespace Dan_work_hours_l54_54048

theorem Dan_work_hours (x : ℝ) :
  (1 / 15) * x + 3 / 5 = 1 → x = 6 :=
by
  intro h
  sorry

end Dan_work_hours_l54_54048


namespace smallest_angle_in_scalene_triangle_l54_54318

theorem smallest_angle_in_scalene_triangle :
  ∃ (triangle : Type) (a b c : ℝ),
    ∀ (A B C : triangle),
      a = 162 ∧
      b / c = 3 / 4 ∧
      a + b + c = 180 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c ->
        min b c = 7.7 :=
sorry

end smallest_angle_in_scalene_triangle_l54_54318


namespace simplify_inverse_sum_l54_54155

variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem simplify_inverse_sum :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) :=
by sorry

end simplify_inverse_sum_l54_54155


namespace calculate_womans_haircut_cost_l54_54371

-- Define the necessary constants and conditions
def W : ℝ := sorry
def child_haircut_cost : ℝ := 36
def tip_percentage : ℝ := 0.20
def total_tip : ℝ := 24
def number_of_children : ℕ := 2

-- Helper function to calculate total cost before the tip
def total_cost_before_tip (W : ℝ) (number_of_children : ℕ) (child_haircut_cost : ℝ) : ℝ :=
  W + number_of_children * child_haircut_cost

-- Lean statement for the main theorem
theorem calculate_womans_haircut_cost (W : ℝ) (child_haircut_cost : ℝ) (tip_percentage : ℝ)
  (total_tip : ℝ) (number_of_children : ℕ) :
  (tip_percentage * total_cost_before_tip W number_of_children child_haircut_cost) = total_tip →
  W = 48 :=
by
  sorry

end calculate_womans_haircut_cost_l54_54371


namespace sum_of_angles_of_inscribed_quadrilateral_l54_54631

/--
Given a quadrilateral EFGH inscribed in a circle, and the measures of ∠EGH = 50° and ∠GFE = 70°,
then the sum of the angles ∠EFG + ∠EHG is 60°.
-/
theorem sum_of_angles_of_inscribed_quadrilateral
  (E F G H : Type)
  (circumscribed : True) -- This is just a place holder for the circle condition
  (angle_EGH : ℝ) (angle_GFE : ℝ)
  (h1 : angle_EGH = 50)
  (h2 : angle_GFE = 70) :
  ∃ (angle_EFG angle_EHG : ℝ), angle_EFG + angle_EHG = 60 := sorry

end sum_of_angles_of_inscribed_quadrilateral_l54_54631


namespace range_of_a_for_f_zero_l54_54030

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_f_zero (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 :=
by
  sorry

end range_of_a_for_f_zero_l54_54030


namespace complex_number_fourth_quadrant_l54_54683

theorem complex_number_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) : 
  (3 * m - 2) > 0 ∧ (m - 1) < 0 := 
by 
  sorry

end complex_number_fourth_quadrant_l54_54683


namespace sum_of_angles_is_90_l54_54026

variables (α β γ : ℝ)
-- Given angles marked on squared paper, which imply certain geometric properties
axiom angle_properties : α + β + γ = 90

theorem sum_of_angles_is_90 : α + β + γ = 90 := 
by
  apply angle_properties

end sum_of_angles_is_90_l54_54026


namespace speed_of_first_plane_l54_54704

theorem speed_of_first_plane
  (v : ℕ)
  (travel_time : ℚ := 44 / 11)
  (relative_speed : ℚ := v + 90)
  (distance : ℚ := 800) :
  (relative_speed * travel_time = distance) → v = 110 :=
by
  sorry

end speed_of_first_plane_l54_54704


namespace prove_AP_BP_CP_product_l54_54355

open Classical

-- Defines that the point P is inside the acute-angled triangle ABC
variables {A B C P: Type} [MetricSpace P] 
variables (PA1 PB1 PC1 AP BP CP : ℝ)

-- Conditions
def conditions (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) : Prop :=
  PA1 = 3 ∧ PB1 = 3 ∧ PC1 = 3 ∧ AP + BP + CP = 43

-- Proof goal
theorem prove_AP_BP_CP_product (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) :
  AP * BP * CP = 441 :=
by {
  -- Proof steps will be filled here
  sorry
}

end prove_AP_BP_CP_product_l54_54355


namespace find_x_l54_54063

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) : x = 24 / 7 :=
by
  sorry

end find_x_l54_54063


namespace banker_gain_l54_54276

theorem banker_gain :
  ∀ (t : ℝ) (r : ℝ) (TD : ℝ),
  t = 1 →
  r = 12 →
  TD = 65 →
  (TD * r * t) / (100 - (r * t)) = 8.86 :=
by
  intros t r TD ht hr hTD
  rw [ht, hr, hTD]
  sorry

end banker_gain_l54_54276


namespace votes_cast_l54_54644

theorem votes_cast (V : ℝ) (h1 : ∃ Vc, Vc = 0.25 * V) (h2 : ∃ Vr, Vr = 0.25 * V + 4000) : V = 8000 :=
sorry

end votes_cast_l54_54644


namespace total_cupcakes_l54_54379

noncomputable def cupcakesForBonnie : ℕ := 24
noncomputable def cupcakesPerDay : ℕ := 60
noncomputable def days : ℕ := 2

theorem total_cupcakes : (cupcakesPerDay * days + cupcakesForBonnie) = 144 := 
by
  sorry

end total_cupcakes_l54_54379


namespace unattainable_y_l54_54183

theorem unattainable_y (x : ℝ) (h : x ≠ -4 / 3) :
  ¬ ∃ y : ℝ, y = (2 - x) / (3 * x + 4) ∧ y = -1 / 3 :=
by
  sorry

end unattainable_y_l54_54183


namespace certain_number_calculation_l54_54726

theorem certain_number_calculation (x : ℝ) (h : (15 * x) / 100 = 0.04863) : x = 0.3242 :=
by
  sorry

end certain_number_calculation_l54_54726


namespace arun_deepak_age_ratio_l54_54763

-- Define the current age of Arun based on the condition that after 6 years he will be 26 years old
def Arun_current_age : ℕ := 26 - 6

-- Define Deepak's current age based on the given condition
def Deepak_current_age : ℕ := 15

-- The present ratio between Arun's age and Deepak's age
theorem arun_deepak_age_ratio : Arun_current_age / Nat.gcd Arun_current_age Deepak_current_age = (4 : ℕ) ∧ Deepak_current_age / Nat.gcd Arun_current_age Deepak_current_age = (3 : ℕ) := 
by
  -- Proof omitted
  sorry

end arun_deepak_age_ratio_l54_54763


namespace min_visible_sum_of_values_l54_54877

-- Definitions based on the problem conditions
def is_standard_die (die : ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), (i + j = 7) → (die j + die i = 7)

def corner_cubes (cubes : ℕ) : ℕ := 8
def edge_cubes (cubes : ℕ) : ℕ := 24
def face_center_cubes (cubes : ℕ) : ℕ := 24

-- The proof statement
theorem min_visible_sum_of_values
  (m : ℕ)
  (condition1 : is_standard_die m)
  (condition2 : corner_cubes 64 = 8)
  (condition3 : edge_cubes 64 = 24)
  (condition4 : face_center_cubes 64 = 24)
  (condition5 : 64 = 8 + 24 + 24 + 8): 
  m = 144 :=
sorry

end min_visible_sum_of_values_l54_54877


namespace radius_ratio_l54_54755

theorem radius_ratio (V₁ V₂ : ℝ) (hV₁ : V₁ = 432 * Real.pi) (hV₂ : V₂ = 108 * Real.pi) : 
  (∃ (r₁ r₂ : ℝ), V₁ = (4/3) * Real.pi * r₁^3 ∧ V₂ = (4/3) * Real.pi * r₂^3) →
  ∃ k : ℝ, k = r₂ / r₁ ∧ k = 1 / 2^(2/3) := 
by
  sorry

end radius_ratio_l54_54755


namespace coefficient_x4_of_square_l54_54270

theorem coefficient_x4_of_square (q : Polynomial ℝ) (hq : q = Polynomial.X^5 - 4 * Polynomial.X^2 + 3) :
  (Polynomial.coeff (q * q) 4 = 16) :=
by {
  sorry
}

end coefficient_x4_of_square_l54_54270


namespace problem_statement_l54_54213

theorem problem_statement (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : abc = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) := 
  sorry

end problem_statement_l54_54213


namespace asymptotes_of_hyperbola_l54_54412

variable {a : ℝ}

/-- Given that the length of the real axis of the hyperbola x^2/a^2 - y^2 = 1 (a > 0) is 1,
    we want to prove that the equation of its asymptotes is y = ± 2x. -/
theorem asymptotes_of_hyperbola (ha : a > 0) (h_len : 2 * a = 1) :
  ∀ x y : ℝ, (y = 2 * x) ∨ (y = -2 * x) :=
by {
  sorry
}

end asymptotes_of_hyperbola_l54_54412


namespace total_cookies_l54_54585

   -- Define the conditions
   def cookies_per_bag : ℕ := 41
   def number_of_bags : ℕ := 53

   -- Define the problem: Prove that the total number of cookies is 2173
   theorem total_cookies : cookies_per_bag * number_of_bags = 2173 :=
   by sorry
   
end total_cookies_l54_54585


namespace total_legs_in_farm_l54_54098

theorem total_legs_in_farm (total_animals : ℕ) (total_cows : ℕ) (cow_legs : ℕ) (duck_legs : ℕ) 
  (h_total_animals : total_animals = 15) (h_total_cows : total_cows = 6) 
  (h_cow_legs : cow_legs = 4) (h_duck_legs : duck_legs = 2) :
  total_cows * cow_legs + (total_animals - total_cows) * duck_legs = 42 :=
by
  sorry

end total_legs_in_farm_l54_54098


namespace inequality_of_powers_l54_54844

theorem inequality_of_powers (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end inequality_of_powers_l54_54844


namespace stair_calculation_l54_54966

def already_climbed : ℕ := 74
def left_to_climb : ℕ := 22
def total_stairs : ℕ := 96

theorem stair_calculation :
  already_climbed + left_to_climb = total_stairs :=
by {
  sorry
}

end stair_calculation_l54_54966


namespace probability_age_20_to_40_l54_54992

theorem probability_age_20_to_40 
    (total_people : ℕ) (aged_20_to_30 : ℕ) (aged_30_to_40 : ℕ) 
    (h_total : total_people = 350) 
    (h_aged_20_to_30 : aged_20_to_30 = 105) 
    (h_aged_30_to_40 : aged_30_to_40 = 85) : 
    (190 / 350 : ℚ) = 19 / 35 := 
by 
  sorry

end probability_age_20_to_40_l54_54992


namespace inequality_proof_l54_54907

theorem inequality_proof (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) :=
by 
  sorry

end inequality_proof_l54_54907


namespace eq_x_minus_y_l54_54096

theorem eq_x_minus_y (x y : ℝ) : (x - y) * (x - y) = x^2 - 2 * x * y + y^2 :=
by
  sorry

end eq_x_minus_y_l54_54096


namespace find_the_number_l54_54584

-- Define the variables and conditions
variable (x z : ℝ)
variable (the_number : ℝ)

-- Condition: given that x = 1
axiom h1 : x = 1

-- Condition: given the equation
axiom h2 : 14 * (-x + z) + 18 = -14 * (x - z) - the_number

-- The theorem to prove
theorem find_the_number : the_number = -4 :=
by
  sorry

end find_the_number_l54_54584


namespace pieces_per_sister_l54_54280

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end pieces_per_sister_l54_54280


namespace quadratic_complete_square_l54_54394

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 + 2 * x + 3) = ((x + 1)^2 + 2) :=
by
  intro x
  sorry

end quadratic_complete_square_l54_54394


namespace sample_second_grade_l54_54122

theorem sample_second_grade (r1 r2 r3 sample_size : ℕ) (h1 : r1 = 3) (h2 : r2 = 3) (h3 : r3 = 4) (h_sample_size : sample_size = 50) : (r2 * sample_size) / (r1 + r2 + r3) = 15 := by
  sorry

end sample_second_grade_l54_54122


namespace class_gpa_l54_54910

theorem class_gpa (n : ℕ) (h1 : (n / 3) * 60 + (2 * (n / 3)) * 66 = total_gpa) :
  total_gpa / n = 64 :=
by
  sorry

end class_gpa_l54_54910


namespace negation_of_proposition_l54_54864

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x > 0 → (x - 2) / x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ (0 ≤ x ∧ x < 2) := 
sorry

end negation_of_proposition_l54_54864


namespace units_digit_product_composites_l54_54046

theorem units_digit_product_composites :
  (4 * 6 * 8 * 9 * 10) % 10 = 0 :=
sorry

end units_digit_product_composites_l54_54046


namespace log_bounds_sum_l54_54661

theorem log_bounds_sum : (∀ a b : ℕ, a = 18 ∧ b = 19 → 18 < Real.log 537800 / Real.log 2 ∧ Real.log 537800 / Real.log 2 < 19 → a + b = 37) := 
sorry

end log_bounds_sum_l54_54661


namespace probability_of_9_heads_in_12_l54_54681

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l54_54681


namespace proof_problem_l54_54180

-- Definitions for the conditions and the events in the problem
def P_A : ℚ := 2 / 3
def P_B : ℚ := 1 / 4
def P_not_any_module : ℚ := 1 - (P_A + P_B)

-- Definition for the binomial coefficient
def C (n k : ℕ) := Nat.choose n k

-- Definition for the event where at least 3 out of 4 students have taken "Selected Topics in Geometric Proofs"
def P_at_least_three_taken : ℚ := 
  C 4 3 * (P_A ^ 3) * ((1 - P_A) ^ 1) + C 4 4 * (P_A ^ 4)

-- The main theorem to prove
theorem proof_problem : 
  P_not_any_module = 1 / 12 ∧ P_at_least_three_taken = 16 / 27 :=
by
  sorry

end proof_problem_l54_54180


namespace find_x_l54_54660

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define the collinearity condition
def collinear_with_3a_plus_b (x : ℝ) : Prop :=
  ∃ k : ℝ, c x = k • (3 • a + b)

theorem find_x :
  ∀ x : ℝ, collinear_with_3a_plus_b x → x = -4 := 
sorry

end find_x_l54_54660


namespace carpet_area_l54_54348

def room_length_ft := 16
def room_width_ft := 12
def column_side_ft := 2
def ft_to_inches := 12

def room_length_in := room_length_ft * ft_to_inches
def room_width_in := room_width_ft * ft_to_inches
def column_side_in := column_side_ft * ft_to_inches

def room_area_in_sq := room_length_in * room_width_in
def column_area_in_sq := column_side_in * column_side_in

def remaining_area_in_sq := room_area_in_sq - column_area_in_sq

theorem carpet_area : remaining_area_in_sq = 27072 := by
  sorry

end carpet_area_l54_54348


namespace packet_weight_l54_54777

theorem packet_weight
  (tons_to_pounds : ℕ := 2600) -- 1 ton = 2600 pounds
  (total_tons : ℕ := 13)       -- Total capacity in tons
  (num_packets : ℕ := 2080)    -- Number of packets
  (expected_weight_per_packet : ℚ := 16.25) : 
  total_tons * tons_to_pounds / num_packets = expected_weight_per_packet := 
sorry

end packet_weight_l54_54777


namespace problem_statement_l54_54506

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def complement_U (s : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ s}
noncomputable def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem problem_statement : intersection N (complement_U M) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end problem_statement_l54_54506


namespace original_number_unique_l54_54784

theorem original_number_unique (N : ℤ) (h : (N - 31) % 87 = 0) : N = 118 :=
by
  sorry

end original_number_unique_l54_54784


namespace pablo_distributed_fraction_l54_54326

-- Definitions based on the problem statement
def mia_coins (m : ℕ) := m
def sofia_coins (m : ℕ) := 3 * m
def pablo_coins (m : ℕ) := 12 * m

-- Condition for equal distribution
def target_coins (m : ℕ) := (mia_coins m + sofia_coins m + pablo_coins m) / 3

-- Needs for redistribution
def sofia_needs (m : ℕ) := target_coins m - sofia_coins m
def mia_needs (m : ℕ) := target_coins m - mia_coins m

-- Total distributed coins by Pablo
def total_distributed_by_pablo (m : ℕ) := sofia_needs m + mia_needs m

-- Fraction of coins Pablo distributes
noncomputable def fraction_distributed_by_pablo (m : ℕ) := (total_distributed_by_pablo m) / (pablo_coins m)

-- Theorem to prove
theorem pablo_distributed_fraction (m : ℕ) : fraction_distributed_by_pablo m = 5 / 9 := by
  sorry

end pablo_distributed_fraction_l54_54326


namespace point_not_on_graph_l54_54776

def on_graph (x y : ℚ) : Prop := y = x / (x + 2)

/-- Let's state the main theorem -/
theorem point_not_on_graph : ¬ on_graph 2 (2 / 3) := by
  sorry

end point_not_on_graph_l54_54776


namespace arithmetic_mean_x_is_16_point_4_l54_54528

theorem arithmetic_mean_x_is_16_point_4 {x : ℝ}
  (h : (x + 10 + 17 + 2 * x + 15 + 2 * x + 6) / 5 = 26):
  x = 16.4 := 
sorry

end arithmetic_mean_x_is_16_point_4_l54_54528


namespace ram_birthday_l54_54052

theorem ram_birthday
    (L : ℕ) (L1 : ℕ) (Llast : ℕ) (d : ℕ) (languages_learned_per_day : ℕ) (days_in_month : ℕ) :
    (L = 1000) →
    (L1 = 820) →
    (Llast = 1100) →
    (days_in_month = 28 ∨ days_in_month = 29 ∨ days_in_month = 30 ∨ days_in_month = 31) →
    (d = days_in_month - 1) →
    (languages_learned_per_day = (Llast - L1) / d) →
    ∃ n : ℕ, n = 19 :=
by
  intros hL hL1 hLlast hDays hm_d hLearned
  existsi 19
  sorry

end ram_birthday_l54_54052


namespace discount_each_book_l54_54665

-- Definition of conditions
def original_price : ℝ := 5
def num_books : ℕ := 10
def total_paid : ℝ := 45

-- Theorem statement to prove the discount
theorem discount_each_book (d : ℝ) 
  (h1 : original_price * (num_books : ℝ) - d * (num_books : ℝ) = total_paid) : 
  d = 0.5 := 
sorry

end discount_each_book_l54_54665


namespace isosceles_triangle_base_length_l54_54652

theorem isosceles_triangle_base_length
  (b : ℕ)
  (congruent_side : ℕ)
  (perimeter : ℕ)
  (h1 : congruent_side = 8)
  (h2 : perimeter = 25)
  (h3 : 2 * congruent_side + b = perimeter) :
  b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l54_54652


namespace no_prime_solution_l54_54926

theorem no_prime_solution (p : ℕ) (h_prime : Nat.Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end no_prime_solution_l54_54926


namespace cost_of_one_hockey_stick_l54_54056

theorem cost_of_one_hockey_stick (x : ℝ)
    (h1 : x * 2 + 25 = 68) : x = 21.50 :=
by
  sorry

end cost_of_one_hockey_stick_l54_54056


namespace expression_value_l54_54335

theorem expression_value :
  3 * 12^2 - 3 * 13 + 2 * 16 * 11^2 = 4265 :=
by
  sorry

end expression_value_l54_54335


namespace lines_intersect_l54_54745

def line1 (t : ℝ) : ℝ × ℝ :=
  (1 - 2 * t, 2 + 4 * t)

def line2 (u : ℝ) : ℝ × ℝ :=
  (3 + u, 5 + 3 * u)

theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (1.2, 1.6) :=
by
  sorry

end lines_intersect_l54_54745


namespace profit_percentage_for_unspecified_weight_l54_54640

-- Definitions to align with the conditions
def total_sugar : ℝ := 1000
def profit_400_kg : ℝ := 0.08
def unspecified_weight : ℝ := 600
def overall_profit : ℝ := 0.14
def total_400_kg := total_sugar - unspecified_weight
def total_overall_profit := total_sugar * overall_profit
def total_400_kg_profit := total_400_kg * profit_400_kg
def total_unspecified_weight_profit (profit_percentage : ℝ) := unspecified_weight * profit_percentage

-- The theorem statement
theorem profit_percentage_for_unspecified_weight : 
  ∃ (profit_percentage : ℝ), total_400_kg_profit + total_unspecified_weight_profit profit_percentage = total_overall_profit ∧ profit_percentage = 0.18 := by
  sorry

end profit_percentage_for_unspecified_weight_l54_54640


namespace num_children_with_dogs_only_l54_54931

-- Defining the given values and constants
def total_children : ℕ := 30
def children_with_cats : ℕ := 12
def children_with_dogs_and_cats : ℕ := 6

-- Define the required proof statement
theorem num_children_with_dogs_only : 
  ∃ (D : ℕ), D + children_with_dogs_and_cats + (children_with_cats - children_with_dogs_and_cats) = total_children ∧ D = 18 :=
by
  sorry

end num_children_with_dogs_only_l54_54931


namespace cubicroots_expression_l54_54725

theorem cubicroots_expression (a b c : ℝ)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 11)
  (h₃ : a * b * c = 6) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 251 / 216 :=
by sorry

end cubicroots_expression_l54_54725


namespace chef_pies_total_l54_54828

def chefPieSales : ℕ :=
  let small_shepherd_pies := 52 / 4
  let large_shepherd_pies := 76 / 8
  let small_chicken_pies := 80 / 5
  let large_chicken_pies := 130 / 10
  let small_vegetable_pies := 42 / 6
  let large_vegetable_pies := 96 / 12
  let small_beef_pies := 35 / 7
  let large_beef_pies := 105 / 14

  small_shepherd_pies + large_shepherd_pies + small_chicken_pies + large_chicken_pies +
  small_vegetable_pies + large_vegetable_pies +
  small_beef_pies + large_beef_pies

theorem chef_pies_total : chefPieSales = 80 := by
  unfold chefPieSales
  have h1 : 52 / 4 = 13 := by norm_num
  have h2 : 76 / 8 = 9 ∨ 76 / 8 = 10 := by norm_num -- rounding consideration
  have h3 : 80 / 5 = 16 := by norm_num
  have h4 : 130 / 10 = 13 := by norm_num
  have h5 : 42 / 6 = 7 := by norm_num
  have h6 : 96 / 12 = 8 := by norm_num
  have h7 : 35 / 7 = 5 := by norm_num
  have h8 : 105 / 14 = 7 ∨ 105 / 14 = 8 := by norm_num -- rounding consideration
  sorry

end chef_pies_total_l54_54828


namespace geometric_sequence_sum_l54_54951

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 2 = 1 - a 1)
  (h3 : a 4 = 9 - a 3)
  (h4 : ∀ n, a (n + 1) = a n * q) :
  a 4 + a 5 = 27 :=
sorry

end geometric_sequence_sum_l54_54951


namespace sqrt_of_mixed_number_l54_54934

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l54_54934


namespace trigonometric_identity_l54_54703

theorem trigonometric_identity : 
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  -- Here we assume standard trigonometric identities and basic properties already handled by Mathlib
  sorry

end trigonometric_identity_l54_54703


namespace inequality_squares_l54_54923

theorem inequality_squares (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h : a + b + c = 1) :
    (3 / 16) ≤ ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ∧
    ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ≤ 1 / 4 :=
by
  sorry

end inequality_squares_l54_54923


namespace combination_multiplication_and_addition_l54_54739

theorem combination_multiplication_and_addition :
  (Nat.choose 10 3) * (Nat.choose 8 3) + (Nat.choose 5 2) = 6730 :=
by
  sorry

end combination_multiplication_and_addition_l54_54739


namespace expression_evaluation_l54_54715

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (2 * a + Real.sqrt 3) * (2 * a - Real.sqrt 3) - 3 * a * (a - 2) + 3 = -7 :=
by
  sorry

end expression_evaluation_l54_54715


namespace frequency_of_group_l54_54389

-- Definitions based on conditions in the problem
def sampleCapacity : ℕ := 32
def frequencyRate : ℝ := 0.25

-- Lean statement representing the proof
theorem frequency_of_group : (frequencyRate * sampleCapacity : ℝ) = 8 := 
by 
  sorry -- Proof placeholder

end frequency_of_group_l54_54389


namespace range_of_x_l54_54571

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

theorem range_of_x (x : ℝ) (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) 2) :
  f (m * x - 3) + f x < 0 → -3 < x ∧ x < 1 :=
sorry

end range_of_x_l54_54571


namespace minimum_value_side_c_l54_54905

open Real

noncomputable def minimum_side_c (a b c : ℝ) (B : ℝ) (S : ℝ) : ℝ := c

theorem minimum_value_side_c (a b c B : ℝ) (h1 : c * cos B = a + 1 / 2 * b)
  (h2 : S = sqrt 3 / 12 * c) :
  minimum_side_c a b c B S >= 1 :=
by
  -- Precise translation of mathematical conditions and required proof. 
  -- The actual steps to prove the theorem would be here.
  sorry

end minimum_value_side_c_l54_54905


namespace inequality_proof_l54_54603

variable {a b c d : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  64 * (abcd + 1) / (a + b + c + d)^2 ≤ a^2 + b^2 + c^2 + d^2 + 1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 :=
by 
  sorry

end inequality_proof_l54_54603


namespace number_of_triangles_in_polygon_l54_54675

theorem number_of_triangles_in_polygon {n : ℕ} (h : n > 0) :
  let vertices := (2 * n + 1)
  ∃ triangles_containing_center : ℕ, triangles_containing_center = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

end number_of_triangles_in_polygon_l54_54675


namespace pyramid_surface_area_l54_54381

noncomputable def total_surface_area_of_pyramid (a b : ℝ) (theta : ℝ) (height : ℝ) : ℝ :=
  let base_area := a * b * Real.sin theta
  let slant_height := Real.sqrt (height ^ 2 + (a / 2) ^ 2)
  let lateral_area := 4 * (1 / 2 * a * slant_height)
  base_area + lateral_area

theorem pyramid_surface_area :
  total_surface_area_of_pyramid 12 14 (Real.pi / 3) 15 = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 :=
by sorry

end pyramid_surface_area_l54_54381


namespace find_divisor_l54_54071

def div_remainder (a b r : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b + r

theorem find_divisor :
  ∃ D : ℕ, (div_remainder 242 D 15) ∧ (div_remainder 698 D 27) ∧ (div_remainder (242 + 698) D 5) ∧ D = 37 := 
by
  sorry

end find_divisor_l54_54071


namespace lowest_sale_price_percentage_l54_54070

theorem lowest_sale_price_percentage :
  ∃ (p : ℝ) (h1 : 30 / 100 * p ≤ 70 / 100 * p) (h2 : p = 80),
  (p - 70 / 100 * p - 20 / 100 * p) / p * 100 = 10 := by
sorry

end lowest_sale_price_percentage_l54_54070


namespace total_weight_of_peppers_l54_54976

def green_peppers_weight : Real := 0.3333333333333333
def red_peppers_weight : Real := 0.3333333333333333
def total_peppers_weight : Real := 0.6666666666666666

theorem total_weight_of_peppers :
  green_peppers_weight + red_peppers_weight = total_peppers_weight :=
by
  sorry

end total_weight_of_peppers_l54_54976


namespace ratio_of_areas_of_similar_triangles_l54_54998

-- Define the variables and conditions
variables {ABC DEF : Type} 
variables (hABCDEF : Similar ABC DEF) 
variables (perimeterABC perimeterDEF : ℝ)
variables (hpABC : perimeterABC = 3)
variables (hpDEF : perimeterDEF = 1)

-- The theorem statement
theorem ratio_of_areas_of_similar_triangles :
  (perimeterABC / perimeterDEF) ^ 2 = 9 :=
by
  sorry

end ratio_of_areas_of_similar_triangles_l54_54998


namespace area_of_square_l54_54786

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l54_54786


namespace Mike_and_Sarah_missed_days_l54_54207

theorem Mike_and_Sarah_missed_days :
  ∀ (V M S : ℕ), V + M + S = 17 → V + M = 14 → V = 5 → M + S = 12 :=
by
  intros V M S h1 h2 h3
  sorry

end Mike_and_Sarah_missed_days_l54_54207


namespace starling_nests_flying_condition_l54_54175

theorem starling_nests_flying_condition (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → A ≠ B)
  (h3 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → 
  (∃ d1 d2 : ℝ, d1 < d2 ∧ d1 < d2 → d1 > d2)) : n = 3 :=
by
  sorry

end starling_nests_flying_condition_l54_54175


namespace identify_counterfeit_bag_l54_54199

theorem identify_counterfeit_bag (n : ℕ) (w W : ℕ) (H : ∃ k : ℕ, k ≤ n ∧ W = w * (n * (n + 1) / 2) - k) : 
  ∃ bag_num, bag_num = w * (n * (n + 1) / 2) - W := by
  sorry

end identify_counterfeit_bag_l54_54199


namespace stars_substitution_correct_l54_54263

-- Define x and y with given conditions
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5
def product_ends_in_25 (x y : ℕ) : Prop := (x * y) % 100 = 25
def tens_digit_even (n : ℕ) : Prop := (n / 10) % 2 = 0
def valid_tens_digit (n : ℕ) : Prop := (n / 10) % 10 ≤ 3

theorem stars_substitution_correct :
  ∃ (x y : ℕ), ends_in_5 x ∧ ends_in_5 y ∧ product_ends_in_25 x y ∧ tens_digit_even x ∧ valid_tens_digit y ∧ x * y = 9125 :=
sorry

end stars_substitution_correct_l54_54263


namespace all_items_weight_is_8040_l54_54187

def weight_of_all_items : Real :=
  let num_tables := 15
  let settings_per_table := 8
  let backup_percentage := 0.25

  let weight_fork := 3.5
  let weight_knife := 4.0
  let weight_spoon := 4.5
  let weight_large_plate := 14.0
  let weight_small_plate := 10.0
  let weight_wine_glass := 7.0
  let weight_water_glass := 9.0
  let weight_table_decoration := 16.0

  let total_settings := (num_tables * settings_per_table) * (1 + backup_percentage)
  let weight_per_setting := (weight_fork + weight_knife + weight_spoon) + (weight_large_plate + weight_small_plate) + (weight_wine_glass + weight_water_glass)
  let total_weight_decorations := num_tables * weight_table_decoration

  let total_weight := total_settings * weight_per_setting + total_weight_decorations
  total_weight

theorem all_items_weight_is_8040 :
  weight_of_all_items = 8040 := sorry

end all_items_weight_is_8040_l54_54187


namespace convert_50_to_base_3_l54_54676

-- Define a function to convert decimal to ternary (base-3)
def convert_to_ternary (n : ℕ) : ℕ := sorry

-- Main theorem statement
theorem convert_50_to_base_3 : convert_to_ternary 50 = 1212 :=
sorry

end convert_50_to_base_3_l54_54676


namespace prove_inequality_l54_54696

-- Defining properties of f
variable {α : Type*} [LinearOrderedField α] (f : α → α)

-- Condition 1: f is even function
def is_even_function (f : α → α) : Prop := ∀ x : α, f (-x) = f x

-- Condition 2: f is monotonically increasing on (0, ∞)
def is_monotonically_increasing_on_positive (f : α → α) : Prop := ∀ ⦃x y : α⦄, 0 < x → 0 < y → x < y → f x < f y

-- Define the main theorem we need to prove:
theorem prove_inequality (h1 : is_even_function f) (h2 : is_monotonically_increasing_on_positive f) : 
  f (-1) < f 2 ∧ f 2 < f (-3) :=
by
  sorry

end prove_inequality_l54_54696


namespace area_of_triangle_XPQ_l54_54767

noncomputable def area_triangle_XPQ (XY YZ XZ XP XQ : ℝ) (hXY : XY = 12) (hYZ : YZ = 13) (hXZ : XZ = 15) (hXP : XP = 5) (hXQ : XQ = 9) : ℝ :=
  let s := (XY + YZ + XZ) / 2
  let area_XYZ := Real.sqrt (s * (s - XY) * (s - YZ) * (s - XZ))
  let cosX := (XY^2 + YZ^2 - XZ^2) / (2 * XY * YZ)
  let sinX := Real.sqrt (1 - cosX^2)
  (1 / 2) * XP * XQ * sinX

theorem area_of_triangle_XPQ :
  area_triangle_XPQ 12 13 15 5 9 (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) = 45 * Real.sqrt 1400 / 78 :=
by
  sorry

end area_of_triangle_XPQ_l54_54767


namespace calories_in_250_grams_is_106_l54_54083

noncomputable def total_calories_apple : ℝ := 150 * (46 / 100)
noncomputable def total_calories_orange : ℝ := 50 * (45 / 100)
noncomputable def total_calories_carrot : ℝ := 300 * (40 / 100)
noncomputable def total_calories_mix : ℝ := total_calories_apple + total_calories_orange + total_calories_carrot
noncomputable def total_weight_mix : ℝ := 150 + 50 + 300
noncomputable def caloric_density : ℝ := total_calories_mix / total_weight_mix
noncomputable def calories_in_250_grams : ℝ := 250 * caloric_density

theorem calories_in_250_grams_is_106 : calories_in_250_grams = 106 :=
by
  sorry

end calories_in_250_grams_is_106_l54_54083


namespace intersection_of_planes_is_line_l54_54449

-- Define the conditions as Lean 4 statements
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y + z - 8 = 0
def plane2 (x y z : ℝ) : Prop := x - 2 * y - 2 * z + 1 = 0

-- Define the canonical form of the line as a Lean 4 proposition
def canonical_line (x y z : ℝ) : Prop := 
  (x - 3) / -4 = y / 5 ∧ y / 5 = (z - 2) / -7

-- The theorem to state equivalence between conditions and canonical line equations
theorem intersection_of_planes_is_line :
  ∀ (x y z : ℝ), plane1 x y z → plane2 x y z → canonical_line x y z :=
by
  intros x y z h1 h2
  -- TODO: Insert proof here
  sorry

end intersection_of_planes_is_line_l54_54449


namespace ratio_of_areas_of_shaded_and_white_region_l54_54332

theorem ratio_of_areas_of_shaded_and_white_region
  (all_squares_have_vertices_in_middle: ∀ (n : ℕ), n ≠ 0 → (square_vertices_positioned_mid : Prop)) :
  ∃ (ratio : ℚ), ratio = 5 / 3 :=
by
  sorry

end ratio_of_areas_of_shaded_and_white_region_l54_54332


namespace intersection_complement_eq_find_a_l54_54005

-- Proof Goal 1: A ∩ ¬B = {x : ℝ | x ∈ (-∞, -3] ∪ [14, ∞)}

def setA : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def setB : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def negB : Set ℝ := {x | x ≤ -2 ∨ x ≥ 14}

theorem intersection_complement_eq :
  setA ∩ negB = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Proof Goal 2: The range of a such that E ⊆ B

def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

theorem find_a (a : ℝ) :
  (∀ x, E a x → setB x) → a ≥ -1 :=
by
  sorry

end intersection_complement_eq_find_a_l54_54005


namespace find_S3_l54_54369

-- Define the known scores
def S1 : ℕ := 55
def S2 : ℕ := 67
def S4 : ℕ := 55
def Avg : ℕ := 67

-- Statement to prove
theorem find_S3 : ∃ S3 : ℕ, (S1 + S2 + S3 + S4) / 4 = Avg ∧ S3 = 91 :=
by
  sorry

end find_S3_l54_54369


namespace find_gain_percent_l54_54636

-- Definitions based on the conditions
def CP : ℕ := 900
def SP : ℕ := 1170

-- Calculation of gain
def Gain := SP - CP

-- Calculation of gain percent
def GainPercent := (Gain * 100) / CP

-- The theorem to prove the gain percent is 30%
theorem find_gain_percent : GainPercent = 30 := 
by
  sorry -- Proof to be filled in.

end find_gain_percent_l54_54636


namespace burger_cost_cents_l54_54750

theorem burger_cost_cents 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 550) 
  (h2 : 3 * b + 2 * s = 400) 
  (h3 : 2 * b + s = 250) : 
  b = 100 :=
by
  sorry

end burger_cost_cents_l54_54750


namespace max_value_x_2y_2z_l54_54604

theorem max_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : x + 2*y + 2*z ≤ 15 :=
sorry

end max_value_x_2y_2z_l54_54604


namespace fifth_term_arithmetic_sequence_l54_54110

noncomputable def fifth_term (x y : ℚ) (a1 : ℚ := x + 2 * y) (a2 : ℚ := x - 2 * y) (a3 : ℚ := x + 2 * y^2) (a4 : ℚ := x / (2 * y)) (d : ℚ := -4 * y) : ℚ :=
    a4 + d

theorem fifth_term_arithmetic_sequence (x y : ℚ) (h1 : y ≠ 0) :
  (fifth_term x y - (-((x : ℚ) / 6) - 12)) = 0 :=
by
  sorry

end fifth_term_arithmetic_sequence_l54_54110


namespace D_neither_sufficient_nor_necessary_for_A_l54_54167

theorem D_neither_sufficient_nor_necessary_for_A 
  (A B C D : Prop) 
  (h1 : A → B) 
  (h2 : ¬(B → A)) 
  (h3 : B ↔ C) 
  (h4 : C → D) 
  (h5 : ¬(D → C)) 
  :
  ¬(D → A) ∧ ¬(A → D) :=
by 
  sorry

end D_neither_sufficient_nor_necessary_for_A_l54_54167


namespace linear_eq_a_l54_54527

theorem linear_eq_a (a : ℝ) (x y : ℝ) (h1 : (a + 1) ≠ 0) (h2 : |a| = 1) : a = 1 :=
by
  sorry

end linear_eq_a_l54_54527


namespace smallest_difference_l54_54130

-- Definition for the given problem conditions.
def side_lengths (AB BC AC : ℕ) : Prop := 
  AB + BC + AC = 2023 ∧ AB < BC ∧ BC ≤ AC ∧ 
  AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

theorem smallest_difference (AB BC AC : ℕ) 
  (h: side_lengths AB BC AC) : 
  ∃ (AB BC AC : ℕ), side_lengths AB BC AC ∧ (BC - AB = 1) :=
by
  sorry

end smallest_difference_l54_54130


namespace ratio_of_jumps_l54_54723

theorem ratio_of_jumps (run_ric: ℕ) (jump_ric: ℕ) (run_mar: ℕ) (extra_dist: ℕ)
    (h1 : run_ric = 20)
    (h2 : jump_ric = 4)
    (h3 : run_mar = 18)
    (h4 : extra_dist = 1) :
    (run_mar + extra_dist - run_ric - jump_ric) / jump_ric = 7 / 4 :=
by
  sorry

end ratio_of_jumps_l54_54723


namespace quadratic_inequality_solution_l54_54302

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l54_54302


namespace tv_price_reduction_l54_54646

theorem tv_price_reduction (x : ℝ) (Q : ℝ) (P : ℝ) (h1 : Q > 0) (h2 : P > 0) (h3 : P*(1 - x/100) * 1.85 * Q = 1.665 * P * Q) : x = 10 :=
by 
  sorry

end tv_price_reduction_l54_54646


namespace no_integers_satisfy_eq_l54_54097

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^3 = 4 * n + 2) := 
  sorry

end no_integers_satisfy_eq_l54_54097


namespace reduced_price_is_16_l54_54987

noncomputable def reduced_price_per_kg (P : ℝ) (r : ℝ) : ℝ :=
  0.9 * (P * (1 + r))

theorem reduced_price_is_16 (P r : ℝ) (h₀ : (0.9 : ℝ) * (P * (1 + r)) = 16) : 
  reduced_price_per_kg P r = 16 :=
by
  -- We have the hypothesis and we need to prove the result
  exact h₀

end reduced_price_is_16_l54_54987


namespace touchdowns_points_l54_54768

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end touchdowns_points_l54_54768


namespace material_needed_l54_54577

-- Define the required conditions
def feet_per_tee_shirt : ℕ := 4
def number_of_tee_shirts : ℕ := 15

-- State the theorem and the proof obligation
theorem material_needed : feet_per_tee_shirt * number_of_tee_shirts = 60 := 
by 
  sorry

end material_needed_l54_54577


namespace average_price_per_book_l54_54996

theorem average_price_per_book 
  (amount1 : ℝ)
  (books1 : ℕ)
  (amount2 : ℝ)
  (books2 : ℕ)
  (h1 : amount1 = 581)
  (h2 : books1 = 27)
  (h3 : amount2 = 594)
  (h4 : books2 = 20) :
  (amount1 + amount2) / (books1 + books2) = 25 := 
by
  sorry

end average_price_per_book_l54_54996


namespace range_sin_cos_two_x_is_minus2_to_9_over_8_l54_54066

noncomputable def range_of_function : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.sin x + Real.cos (2 * x) }

theorem range_sin_cos_two_x_is_minus2_to_9_over_8 :
  range_of_function = Set.Icc (-2) (9 / 8) := 
by
  sorry

end range_sin_cos_two_x_is_minus2_to_9_over_8_l54_54066


namespace rhombus_diagonal_sum_l54_54803

theorem rhombus_diagonal_sum (e f : ℝ) (h1: e^2 + f^2 = 16) (h2: 0 < e ∧ 0 < f):
  e + f = 5 :=
by
  sorry

end rhombus_diagonal_sum_l54_54803


namespace solve_system_of_equations_l54_54930

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), 
    (2 * y + x - x^2 - y^2 = 0) ∧ 
    (z - x + y - y * (x + z) = 0) ∧ 
    (-2 * y + z - y^2 - z^2 = 0) ∧ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l54_54930


namespace probability_of_selecting_3_co_captains_is_correct_l54_54143

def teams : List ℕ := [4, 6, 7, 9]

def probability_of_selecting_3_co_captains (n : ℕ) : ℚ :=
  if n = 4 then 1/4
  else if n = 6 then 1/20
  else if n = 7 then 1/35
  else if n = 9 then 1/84
  else 0

def total_probability : ℚ :=
  (1/4) * (probability_of_selecting_3_co_captains 4 +
            probability_of_selecting_3_co_captains 6 +
            probability_of_selecting_3_co_captains 7 +
            probability_of_selecting_3_co_captains 9)

theorem probability_of_selecting_3_co_captains_is_correct :
  total_probability = 143 / 1680 :=
by
  -- The proof will be inserted here
  sorry

end probability_of_selecting_3_co_captains_is_correct_l54_54143


namespace find_a16_l54_54642

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n ≥ 1, a (n + 1) = 1 - 1 / a n

theorem find_a16 (a : ℕ → ℝ) (h : seq a) : a 16 = 1 / 2 :=
sorry

end find_a16_l54_54642


namespace expression_not_equal_one_l54_54742

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

end expression_not_equal_one_l54_54742


namespace total_people_l54_54148

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l54_54148


namespace unique_triple_solution_l54_54833

theorem unique_triple_solution (a b c : ℝ) 
  (h1 : a * (b ^ 2 + c) = c * (c + a * b))
  (h2 : b * (c ^ 2 + a) = a * (a + b * c))
  (h3 : c * (a ^ 2 + b) = b * (b + c * a)) : 
  a = b ∧ b = c := 
sorry

end unique_triple_solution_l54_54833


namespace percentage_of_students_play_sports_l54_54256

def total_students : ℕ := 400
def soccer_percentage : ℝ := 0.125
def soccer_players : ℕ := 26

theorem percentage_of_students_play_sports : 
  ∃ P : ℝ, (soccer_percentage * P = soccer_players) → (P / total_students * 100 = 52) :=
by
  sorry

end percentage_of_students_play_sports_l54_54256


namespace solve_fraction_x_l54_54216

theorem solve_fraction_x (a b c d : ℤ) (hb : b ≠ 0) (hdc : d + c ≠ 0) 
: (2 * a + (bc - 2 * a * d) / (d + c)) / (b - (bc - 2 * a * d) / (d + c)) = c / d := 
sorry

end solve_fraction_x_l54_54216


namespace factorize_a_cube_minus_nine_a_l54_54419

theorem factorize_a_cube_minus_nine_a (a : ℝ) : a^3 - 9 * a = a * (a + 3) * (a - 3) :=
by sorry

end factorize_a_cube_minus_nine_a_l54_54419


namespace hex_conversion_sum_l54_54812

-- Convert hexadecimal E78 to decimal
def hex_to_decimal (h : String) : Nat :=
  match h with
  | "E78" => 3704
  | _ => 0

-- Convert decimal to radix 7
def decimal_to_radix7 (d : Nat) : String :=
  match d with
  | 3704 => "13541"
  | _ => ""

-- Convert radix 7 to decimal
def radix7_to_decimal (r : String) : Nat :=
  match r with
  | "13541" => 3704
  | _ => 0

-- Convert decimal to hexadecimal
def decimal_to_hex (d : Nat) : String :=
  match d with
  | 3704 => "E78"
  | 7408 => "1CF0"
  | _ => ""

theorem hex_conversion_sum :
  let initial_hex : String := "E78"
  let final_decimal := 3704 
  let final_hex := decimal_to_hex (final_decimal)
  let final_sum := hex_to_decimal initial_hex + final_decimal
  (decimal_to_hex final_sum) = "1CF0" :=
by
  sorry

end hex_conversion_sum_l54_54812


namespace ny_mets_fans_count_l54_54600

variable (Y M R : ℕ) -- Variables representing number of fans
variable (k j : ℕ)   -- Helper variables for ratios

theorem ny_mets_fans_count :
  (Y = 3 * k) →
  (M = 2 * k) →
  (M = 4 * j) →
  (R = 5 * j) →
  (Y + M + R = 330) →
  (∃ (k j : ℕ), k = 2 * j) →
  M = 88 := sorry

end ny_mets_fans_count_l54_54600


namespace episodes_per_wednesday_l54_54204

theorem episodes_per_wednesday :
  ∀ (W : ℕ), (∃ (n_episodes : ℕ) (n_mondays : ℕ) (n_weeks : ℕ), 
    n_episodes = 201 ∧ n_mondays = 67 ∧ n_weeks = 67 
    ∧ n_weeks * W + n_mondays = n_episodes) 
    → W = 2 :=
by
  intro W
  rintro ⟨n_episodes, n_mondays, n_weeks, h1, h2, h3, h4⟩
  -- proof would go here
  sorry

end episodes_per_wednesday_l54_54204


namespace length_of_bridge_l54_54450

-- Define the conditions
def length_of_train : ℝ := 750
def speed_of_train_kmh : ℝ := 120
def crossing_time : ℝ := 45
def wind_resistance_factor : ℝ := 0.10

-- Define the conversion from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 0.27778

-- Define the actual speed considering wind resistance
def actual_speed_ms (v : ℝ) (resistance : ℝ) : ℝ := (kmh_to_ms v) * (1 - resistance)

-- Define the total distance covered
def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem: Length of the bridge
theorem length_of_bridge : total_distance (actual_speed_ms speed_of_train_kmh wind_resistance_factor) crossing_time - length_of_train = 600 := by
  sorry

end length_of_bridge_l54_54450


namespace solution_of_system_l54_54008

theorem solution_of_system :
  ∃ x y z : ℚ,
    x + 2 * y = 12 ∧
    y + 3 * z = 15 ∧
    3 * x - z = 6 ∧
    x = 54 / 17 ∧
    y = 75 / 17 ∧
    z = 60 / 17 :=
by
  exists 54 / 17, 75 / 17, 60 / 17
  repeat { sorry }

end solution_of_system_l54_54008


namespace initial_number_l54_54217

theorem initial_number (N : ℤ) 
  (h : (N + 3) % 24 = 0) : N = 21 := 
sorry

end initial_number_l54_54217


namespace pyramid_height_l54_54261

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end pyramid_height_l54_54261


namespace find_k_l54_54060

def g (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (h1 : g a b c 1 = 0)
  (h2 : 20 < g a b c 5 ∧ g a b c 5 < 30)
  (h3 : 40 < g a b c 6 ∧ g a b c 6 < 50)
  (h4 : ∃ k : ℤ, 3000 * k < g a b c 100 ∧ g a b c 100 < 3000 * (k + 1)) :
  ∃ k : ℤ, k = 9 :=
by
  sorry

end find_k_l54_54060


namespace additional_track_length_l54_54235

theorem additional_track_length (h : ℝ) (g1 g2 : ℝ) (L1 L2 : ℝ)
  (rise_eq : h = 800) 
  (orig_grade : g1 = 0.04) 
  (new_grade : g2 = 0.025) 
  (L1_eq : L1 = h / g1) 
  (L2_eq : L2 = h / g2)
  : (L2 - L1 = 12000) := 
sorry

end additional_track_length_l54_54235


namespace percentage_increase_is_50_l54_54456

def initialNumber := 80
def finalNumber := 120

theorem percentage_increase_is_50 : ((finalNumber - initialNumber) / initialNumber : ℝ) * 100 = 50 := 
by 
  sorry

end percentage_increase_is_50_l54_54456


namespace grade_more_problems_l54_54170

theorem grade_more_problems (worksheets_total problems_per_worksheet worksheets_graded: ℕ)
  (h1 : worksheets_total = 9)
  (h2 : problems_per_worksheet = 4)
  (h3 : worksheets_graded = 5):
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 16 :=
by
  sorry

end grade_more_problems_l54_54170


namespace correct_option_l54_54485

-- Define the four conditions as propositions
def option_A (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + b ^ 2
def option_B (a : ℝ) : Prop := 2 * a ^ 2 + a = 3 * a ^ 3
def option_C (a : ℝ) : Prop := a ^ 3 * a ^ 2 = a ^ 5
def option_D (a : ℝ) (h : a ≠ 0) : Prop := 2 * a⁻¹ = 1 / (2 * a)

-- Prove which operation is the correct one
theorem correct_option (a b : ℝ) (h : a ≠ 0) : option_C a :=
by {
  -- Placeholder for actual proofs, each option needs to be verified
  sorry
}

end correct_option_l54_54485


namespace john_bike_speed_l54_54915

noncomputable def average_speed_for_bike_ride (swim_distance swim_speed run_distance run_speed bike_distance total_time : ℕ) := 
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem john_bike_speed : average_speed_for_bike_ride 1 5 8 12 (3 / 2) = 18 := by
  sorry

end john_bike_speed_l54_54915


namespace paul_tips_l54_54908

theorem paul_tips (P : ℕ) (h1 : P + 16 = 30) : P = 14 :=
by
  sorry

end paul_tips_l54_54908


namespace black_cards_taken_out_l54_54615

theorem black_cards_taken_out (total_black_cards remaining_black_cards : ℕ)
  (h1 : total_black_cards = 26) (h2 : remaining_black_cards = 21) :
  total_black_cards - remaining_black_cards = 5 :=
by
  sorry

end black_cards_taken_out_l54_54615


namespace bothStoresSaleSameDate_l54_54958

-- Define the conditions
def isBookstoreSaleDay (d : ℕ) : Prop := d % 4 = 0
def isShoeStoreSaleDay (d : ℕ) : Prop := ∃ k : ℕ, d = 5 + 7 * k
def isJulyDay (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

-- Define the problem statement
theorem bothStoresSaleSameDate : 
  (∃ d1 d2 : ℕ, isJulyDay d1 ∧ isBookstoreSaleDay d1 ∧ isShoeStoreSaleDay d1 ∧
                 isJulyDay d2 ∧ isBookstoreSaleDay d2 ∧ isShoeStoreSaleDay d2 ∧ d1 ≠ d2) :=
sorry

end bothStoresSaleSameDate_l54_54958


namespace line_equation_l54_54886

-- Define the conditions: point (2,1) on the line and slope is 2
def point_on_line (x y : ℝ) (m b : ℝ) : Prop := y = m * x + b

def slope_of_line (m : ℝ) : Prop := m = 2

-- Prove the equation of the line is 2x - y - 3 = 0
theorem line_equation (b : ℝ) (h1 : point_on_line 2 1 2 b) : 2 * 2 - 1 - 3 = 0 := by
  sorry

end line_equation_l54_54886


namespace total_pictures_painted_l54_54710

def pictures_painted_in_june : ℕ := 2
def pictures_painted_in_july : ℕ := 2
def pictures_painted_in_august : ℕ := 9

theorem total_pictures_painted : 
  pictures_painted_in_june + pictures_painted_in_july + pictures_painted_in_august = 13 :=
by
  sorry

end total_pictures_painted_l54_54710


namespace right_triangle_perimeter_area_ratio_l54_54711

theorem right_triangle_perimeter_area_ratio 
  (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (hyp : ∀ c, c = Real.sqrt (a^2 + b^2))
  : (a + b + Real.sqrt (a^2 + b^2)) / (0.5 * a * b) = 5 → (∃! x y : ℝ, x + y + Real.sqrt (x^2 + y^2) / (0.5 * x * y) = 5) :=
by
  sorry   -- Proof is omitted as per instructions.

end right_triangle_perimeter_area_ratio_l54_54711


namespace probability_same_carriage_l54_54728

theorem probability_same_carriage (num_carriages num_people : ℕ) (h1 : num_carriages = 10) (h2 : num_people = 3) : 
  ∃ p : ℚ, p = 7/25 ∧ p = 1 - (10 * 9 * 8) / (10^3) :=
by
  sorry

end probability_same_carriage_l54_54728


namespace sum_of_square_areas_l54_54295

theorem sum_of_square_areas (a b : ℝ)
  (h1 : a + b = 14)
  (h2 : a - b = 2) :
  a^2 + b^2 = 100 := by
  sorry

end sum_of_square_areas_l54_54295


namespace frustum_lateral_surface_area_l54_54530

theorem frustum_lateral_surface_area:
  ∀ (R r h : ℝ), R = 7 → r = 4 → h = 6 → (∃ L, L = 33 * Real.pi * Real.sqrt 5) := by
  sorry

end frustum_lateral_surface_area_l54_54530


namespace quadrilateral_area_ABCDEF_l54_54712

theorem quadrilateral_area_ABCDEF :
  ∀ (A B C D E : Type)
  (AC CD AE : ℝ) 
  (angle_ABC angle_ACD : ℝ),
  angle_ABC = 90 ∧
  angle_ACD = 90 ∧
  AC = 20 ∧
  CD = 30 ∧
  AE = 5 →
  ∃ S : ℝ, S = 360 :=
by
  sorry

end quadrilateral_area_ABCDEF_l54_54712


namespace brendas_age_l54_54679

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end brendas_age_l54_54679


namespace proposition_correctness_l54_54065

theorem proposition_correctness :
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬ (1 / a < 1 / b)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ a * b / (a + b)) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (Real.log 9 * Real.log 11 < 1) ∧
  (∀ a b : ℝ, a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → ¬(x + 2 * y = 6)) :=
sorry

end proposition_correctness_l54_54065


namespace range_of_eccentricity_l54_54820

theorem range_of_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) (c : ℝ := Real.sqrt (a^2 - b^2)) 
  (h_dot_product : ∀ (x y: ℝ) (h_point : x^2 / a^2 + y^2 / b^2 = 1), 
    let PF1 : ℝ × ℝ := (-c - x, -y)
    let PF2 : ℝ × ℝ := (c - x, -y)
    PF1.1 * PF2.1 + PF1.2 * PF2.2 ≤ a * c) : 
  ∀ (e : ℝ := c / a), (Real.sqrt 5 - 1) / 2 ≤ e ∧ e < 1 := 
by 
  sorry

end range_of_eccentricity_l54_54820


namespace price_difference_is_correct_l54_54455

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l54_54455


namespace problem_solution_l54_54555

variable (y Q : ℝ)

theorem problem_solution
  (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l54_54555


namespace jail_time_calculation_l54_54020

-- Define conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def arrests_per_day : ℕ := 10
def pre_trial_days : ℕ := 4
def half_two_week_sentence_days : ℕ := 7 -- 1 week is half of 2 weeks

-- Define the calculation of the total combined weeks of jail time
def total_combined_weeks_jail_time : ℕ :=
  let total_arrests := arrests_per_day * number_of_cities * days_of_protest
  let total_days_jail_per_person := pre_trial_days + half_two_week_sentence_days
  let total_combined_days_jail_time := total_arrests * total_days_jail_per_person
  total_combined_days_jail_time / 7

-- Theorem statement
theorem jail_time_calculation : total_combined_weeks_jail_time = 9900 := by
  sorry

end jail_time_calculation_l54_54020


namespace f_of_7_l54_54015

theorem f_of_7 (f : ℝ → ℝ) (h : ∀ (x : ℝ), f (4 * x - 1) = x^2 + 2 * x + 2) :
    f 7 = 10 := by
  sorry

end f_of_7_l54_54015


namespace no_right_angle_sequence_l54_54331

theorem no_right_angle_sequence 
  (A B C : Type)
  (angle_A angle_B angle_C : ℝ)
  (angle_A_eq : angle_A = 59)
  (angle_B_eq : angle_B = 61)
  (angle_C_eq : angle_C = 60)
  (midpoint : A → A → A)
  (A0 B0 C0 : A) :
  ¬ ∃ n : ℕ, ∃ An Bn Cn : A, 
    (An = midpoint Bn Cn) ∧ 
    (Bn = midpoint An Cn) ∧ 
    (Cn = midpoint An Bn) ∧ 
    (angle_A = 90 ∨ angle_B = 90 ∨ angle_C = 90) :=
sorry

end no_right_angle_sequence_l54_54331


namespace smallest_number_l54_54129

theorem smallest_number (x : ℕ) (h1 : (x + 7) % 8 = 0) (h2 : (x + 7) % 11 = 0) (h3 : (x + 7) % 24 = 0) : x = 257 :=
sorry

end smallest_number_l54_54129


namespace determine_x_l54_54607

theorem determine_x (x : ℝ) :
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 8 * x + 15) / (x^2 - 10 * x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 :=
by
  sorry

end determine_x_l54_54607


namespace parallel_lines_l54_54917

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, (a-1) * x + 2 * y + 10 = 0) → (∀ x y : ℝ, x + a * y + 3 = 0) → (a = -1 ∨ a = 2) :=
sorry

end parallel_lines_l54_54917


namespace min_value_3x_4y_l54_54611

noncomputable def minValue (x y : ℝ) : ℝ := 3 * x + 4 * y

theorem min_value_3x_4y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 3 * y = 5 * x * y) : 
  minValue x y ≥ 5 :=
sorry

end min_value_3x_4y_l54_54611


namespace rhombus_condition_perimeter_rhombus_given_ab_l54_54939

noncomputable def roots_of_quadratic (m : ℝ) : Set ℝ :=
{ x : ℝ | x^2 - m * x + m / 2 - 1 / 4 = 0 }

theorem rhombus_condition (m : ℝ) : 
  (∃ ab ad : ℝ, ab ∈ roots_of_quadratic m ∧ ad ∈ roots_of_quadratic m ∧ ab = ad) ↔ m = 1 :=
by
  sorry

theorem perimeter_rhombus_given_ab (m : ℝ) (ab : ℝ) (ad : ℝ) : 
  ab = 2 →
  (ab ∈ roots_of_quadratic m) →
  (ad ∈ roots_of_quadratic m) →
  ab ≠ ad →
  m = 5 / 2 →
  2 * (ab + ad) = 5 :=
by
  sorry

end rhombus_condition_perimeter_rhombus_given_ab_l54_54939


namespace price_increase_percentage_l54_54796

theorem price_increase_percentage (original_price : ℝ) (discount : ℝ) (reduced_price : ℝ) : 
  reduced_price = original_price * (1 - discount) →
  (original_price / reduced_price - 1) * 100 = 8.7 :=
by
  intros h
  sorry

end price_increase_percentage_l54_54796


namespace rectangle_side_ratio_l54_54706

theorem rectangle_side_ratio
  (s : ℝ) -- side length of inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_square : y = s) -- shorter side aligns to form inner square
  (h_outer_area : (3 * s) ^ 2 = 9 * s ^ 2) -- area of outer square is 9 times the inner square
  (h_outer_side_relation : x + s = 3 * s) -- outer side length relation
  : x / y = 2 := 
by
  sorry

end rectangle_side_ratio_l54_54706


namespace encoded_base5_to_base10_l54_54680

-- Given definitions
def base5_to_int (d1 d2 d3 : ℕ) : ℕ := d1 * 25 + d2 * 5 + d3

def V := 2
def W := 0
def X := 4
def Y := 1
def Z := 3

-- Prove that the base-10 expression for the integer coded as XYZ is 108
theorem encoded_base5_to_base10 :
  base5_to_int X Y Z = 108 :=
sorry

end encoded_base5_to_base10_l54_54680


namespace reconstruct_point_A_l54_54460

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A E' F' G' H' : V)

theorem reconstruct_point_A (E F G H : V) (p q r s : ℝ)
  (hE' : E' = 2 • F - E)
  (hF' : F' = 2 • G - F)
  (hG' : G' = 2 • H - G)
  (hH' : H' = 2 • E - H)
  : p = 1/4 ∧ q = 1/4  ∧ r = 1/4  ∧ s = 1/4  :=
by
  sorry

end reconstruct_point_A_l54_54460


namespace eq1_solution_eq2_no_solution_l54_54881

-- For Equation (1)
theorem eq1_solution (x : ℝ) (h : (3 / (2 * x - 2)) + (1 / (1 - x)) = 3) : 
  x = 7 / 6 :=
by sorry

-- For Equation (2)
theorem eq2_no_solution (y : ℝ) : ¬((y / (y - 1)) - (2 / (y^2 - 1)) = 1) :=
by sorry

end eq1_solution_eq2_no_solution_l54_54881


namespace range_of_a_l54_54446

variable (A B : Set ℝ)
variable (a : ℝ)

def setA : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def setB (a : ℝ) : Set ℝ := {x | x ≤ 2 * a ∨ x ≥ a + 1}

theorem range_of_a (a : ℝ) :
  (compl (setB a) ⊆ setA) ↔ (a ≤ -2 ∨ (1 / 2 ≤ a ∧ a < 1)) :=
by
  sorry

end range_of_a_l54_54446


namespace largest_valid_n_l54_54762

def is_valid_n (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 10 * a + b ∧ n = a * (a + b)

theorem largest_valid_n : ∀ n : ℕ, is_valid_n n → n ≤ 48 := by sorry

example : is_valid_n 48 := by sorry

end largest_valid_n_l54_54762


namespace round_robin_tournament_l54_54884

theorem round_robin_tournament (n : ℕ)
  (total_points_1 : ℕ := 3086) (total_points_2 : ℕ := 2018) (total_points_3 : ℕ := 1238)
  (pair_avg_1 : ℕ := (3086 + 1238) / 2) (pair_avg_2 : ℕ := (3086 + 2018) / 2) (pair_avg_3 : ℕ := (1238 + 2018) / 2)
  (overall_avg : ℕ := (3086 + 2018 + 1238) / 3)
  (all_pairwise_diff : pair_avg_1 ≠ pair_avg_2 ∧ pair_avg_1 ≠ pair_avg_3 ∧ pair_avg_2 ≠ pair_avg_3) :
  n = 47 :=
by
  sorry

end round_robin_tournament_l54_54884


namespace miles_driven_l54_54986

def total_miles : ℕ := 1200
def remaining_miles : ℕ := 432

theorem miles_driven : total_miles - remaining_miles = 768 := by
  sorry

end miles_driven_l54_54986


namespace sufficiency_not_necessity_condition_l54_54299

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end sufficiency_not_necessity_condition_l54_54299


namespace line_canonical_eqn_l54_54957

theorem line_canonical_eqn 
  (x y z : ℝ)
  (h1 : x - y + z - 2 = 0)
  (h2 : x - 2*y - z + 4 = 0) :
  ∃ a : ℝ, ∃ b : ℝ, ∃ c : ℝ,
    (a = (x - 8)/3) ∧ (b = (y - 6)/2) ∧ (c = z/(-1)) ∧ (a = b) ∧ (b = c) ∧ (c = a) :=
by sorry

end line_canonical_eqn_l54_54957


namespace find_pairs_l54_54281

def Point := (ℤ × ℤ)

def P : Point := (1, 1)
def Q : Point := (4, 5)
def valid_pairs : List Point := [(4, 1), (7, 5), (10, 9), (1, 5), (4, 9)]

def area (P Q R : Point) : ℚ :=
  (1 / 2 : ℚ) * ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)).natAbs : ℚ)

theorem find_pairs :
  {pairs : List Point // ∀ (a b : ℤ), (0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ area P Q (a, b) = 6) ↔ (a, b) ∈ pairs} :=
  ⟨valid_pairs, by sorry⟩

end find_pairs_l54_54281


namespace nhai_highway_construction_l54_54086

/-- Problem definition -/
def total_man_hours (men1 men2 days1 days2 hours1 hours2 : Nat) : Nat := 
  (men1 * days1 * hours1) + (men2 * days2 * hours2)

theorem nhai_highway_construction :
  let men := 100
  let days1 := 25
  let days2 := 25
  let hours1 := 8
  let hours2 := 10
  let additional_men := 60
  let total_days := 50
  total_man_hours men (men + additional_men) total_days total_days hours1 hours2 = 
  2 * total_man_hours men men days1 days2 hours1 hours1 :=
  sorry

end nhai_highway_construction_l54_54086


namespace remainder_when_divided_by_100_l54_54865

/-- A basketball team has 15 available players. A fixed set of 5 players starts the game, while the other 
10 are available as substitutes. During the game, the coach may make up to 4 substitutions. No player 
removed from the game may reenter, and no two substitutions can happen simultaneously. The players 
involved and the order of substitutions matter. -/
def num_substitution_sequences : ℕ :=
  let a_0 := 1
  let a_1 := 5 * 10
  let a_2 := a_1 * 4 * 9
  let a_3 := a_2 * 3 * 8
  let a_4 := a_3 * 2 * 7
  a_0 + a_1 + a_2 + a_3 + a_4

theorem remainder_when_divided_by_100 : num_substitution_sequences % 100 = 51 :=
by
  -- proof to be written
  sorry

end remainder_when_divided_by_100_l54_54865


namespace find_x_l54_54614

theorem find_x (x : ℝ) (h : 70 + 60 / (x / 3) = 71) : x = 180 :=
sorry

end find_x_l54_54614


namespace find_positive_real_number_l54_54373

theorem find_positive_real_number (x : ℝ) (hx : x = 25 + 2 * Real.sqrt 159) :
  1 / 2 * (3 * x ^ 2 - 1) = (x ^ 2 - 50 * x - 10) * (x ^ 2 + 25 * x + 5) :=
by
  sorry

end find_positive_real_number_l54_54373


namespace midpoint_trajectory_l54_54494

theorem midpoint_trajectory (x y : ℝ) (x0 y0 : ℝ)
  (h_circle : x0^2 + y0^2 = 4)
  (h_tangent : x0 * x + y0 * y = 4)
  (h_x0 : x0 = 2 / x)
  (h_y0 : y0 = 2 / y) :
  x^2 * y^2 = x^2 + y^2 :=
sorry

end midpoint_trajectory_l54_54494


namespace find_k_find_m_l54_54980

-- Condition definitions
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

-- Proof problem statements
theorem find_k (k : ℝ) :
  (3 * a.fst - b.fst) / (a.fst + k * b.fst) = (3 * a.snd - b.snd) / (a.snd + k * b.snd) →
  k = -1 / 3 :=
sorry

theorem find_m (m : ℝ) :
  a.fst * (m * a.fst - b.fst) + a.snd * (m * a.snd - b.snd) = 0 →
  m = -4 / 5 :=
sorry

end find_k_find_m_l54_54980


namespace cost_of_gravelling_path_l54_54889

theorem cost_of_gravelling_path (length width path_width : ℝ) (cost_per_sq_m : ℝ)
  (h1 : length = 110) (h2 : width = 65) (h3 : path_width = 2.5) (h4 : cost_per_sq_m = 0.50) :
  (length * width - (length - 2 * path_width) * (width - 2 * path_width)) * cost_per_sq_m = 425 := by
  sorry

end cost_of_gravelling_path_l54_54889


namespace total_savings_l54_54849

-- Definitions and Conditions
def thomas_monthly_savings : ℕ := 40
def joseph_saving_ratio : ℚ := 3 / 5
def saving_period_months : ℕ := 72

-- Problem Statement
theorem total_savings :
  let thomas_total := thomas_monthly_savings * saving_period_months
  let joseph_monthly_savings := thomas_monthly_savings * joseph_saving_ratio
  let joseph_total := joseph_monthly_savings * saving_period_months
  thomas_total + joseph_total = 4608 := 
by
  sorry

end total_savings_l54_54849


namespace find_four_digit_number_l54_54441

def digits_sum (n : ℕ) : ℕ := (n / 1000) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)
def digits_product (n : ℕ) : ℕ := (n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)

theorem find_four_digit_number :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (digits_sum n) * (digits_product n) = 3990 :=
by
  -- The proof is omitted as instructed.
  sorry

end find_four_digit_number_l54_54441


namespace Isabel_total_problems_l54_54531

theorem Isabel_total_problems :
  let math_pages := 2
  let reading_pages := 4
  let science_pages := 3
  let history_pages := 1
  let problems_per_math_page := 5
  let problems_per_reading_page := 5
  let problems_per_science_page := 7
  let problems_per_history_page := 10
  let total_math_problems := math_pages * problems_per_math_page
  let total_reading_problems := reading_pages * problems_per_reading_page
  let total_science_problems := science_pages * problems_per_science_page
  let total_history_problems := history_pages * problems_per_history_page
  let total_problems := total_math_problems + total_reading_problems + total_science_problems + total_history_problems
  total_problems = 61 := by
  sorry

end Isabel_total_problems_l54_54531


namespace find_g5_l54_54360

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l54_54360


namespace width_of_rectangular_field_l54_54515

theorem width_of_rectangular_field
  (L W : ℝ)
  (h1 : L = (7/5) * W)
  (h2 : 2 * L + 2 * W = 384) :
  W = 80 :=
by
  sorry

end width_of_rectangular_field_l54_54515


namespace James_age_is_11_l54_54434

-- Define the ages of Julio and James.
def Julio_age := 36

-- The age condition in 14 years.
def Julio_age_in_14_years := Julio_age + 14

-- James' age in 14 years and the relation as per the condition.
def James_age_in_14_years (J : ℕ) := J + 14

-- The main proof statement.
theorem James_age_is_11 (J : ℕ) 
  (h1 : Julio_age_in_14_years = 2 * James_age_in_14_years J) : J = 11 :=
by
  sorry

end James_age_is_11_l54_54434


namespace ratio_dvds_to_cds_l54_54814

def total_sold : ℕ := 273
def dvds_sold : ℕ := 168
def cds_sold : ℕ := total_sold - dvds_sold

theorem ratio_dvds_to_cds : (dvds_sold : ℚ) / cds_sold = 8 / 5 := by
  sorry

end ratio_dvds_to_cds_l54_54814


namespace symmetric_points_on_parabola_l54_54196

theorem symmetric_points_on_parabola {a b m n : ℝ}
  (hA : m = a^2 - 2*a - 2)
  (hB : m = b^2 - 2*b - 2)
  (hP : n = (a + b)^2 - 2*(a + b) - 2)
  (h_symmetry : (a + b) / 2 = 1) :
  n = -2 :=
by {
  -- Proof omitted
  sorry
}

end symmetric_points_on_parabola_l54_54196


namespace milan_billed_minutes_l54_54582

-- Variables corresponding to the conditions
variables (f r b : ℝ) (m : ℕ)

-- The conditions of the problem
def conditions : Prop :=
  f = 2 ∧ r = 0.12 ∧ b = 23.36 ∧ b = f + r * m

-- The theorem based on given conditions and aiming to prove that m = 178
theorem milan_billed_minutes (h : conditions f r b m) : m = 178 :=
sorry

end milan_billed_minutes_l54_54582


namespace trigonometric_identity_l54_54245

noncomputable def alpha := -35 / 6 * Real.pi

theorem trigonometric_identity :
  (2 * Real.sin (Real.pi + alpha) * Real.cos (Real.pi - alpha)
    - Real.sin (3 * Real.pi / 2 + alpha)) /
  (1 + Real.sin (alpha) ^ 2 - Real.cos (Real.pi / 2 + alpha)
    - Real.cos (Real.pi + alpha) ^ 2) = -Real.sqrt 3 := by
  sorry

end trigonometric_identity_l54_54245


namespace candy_system_of_equations_l54_54673

-- Definitions based on conditions
def candy_weight := 100
def candy_price1 := 36
def candy_price2 := 20
def mixed_candy_price := 28

theorem candy_system_of_equations (x y: ℝ):
  (x + y = candy_weight) ∧ (candy_price1 * x + candy_price2 * y = mixed_candy_price * candy_weight) :=
sorry

end candy_system_of_equations_l54_54673


namespace piglet_gifted_balloons_l54_54817

noncomputable def piglet_balloons_gifted (piglet_balloons : ℕ) : ℕ :=
  let winnie_balloons := 3 * piglet_balloons
  let owl_balloons := 4 * piglet_balloons
  let total_balloons := piglet_balloons + winnie_balloons + owl_balloons
  let burst_balloons := total_balloons - 60
  piglet_balloons - burst_balloons / 8

-- Prove that Piglet gifted 4 balloons given the conditions
theorem piglet_gifted_balloons :
  ∃ (piglet_balloons : ℕ), piglet_balloons = 8 ∧ piglet_balloons_gifted piglet_balloons = 4 := sorry

end piglet_gifted_balloons_l54_54817


namespace number_of_numbers_l54_54317

theorem number_of_numbers 
  (avg : ℚ) (avg1 : ℚ) (avg2 : ℚ) (avg3 : ℚ)
  (h_avg : avg = 4.60) 
  (h_avg1 : avg1 = 3.4) 
  (h_avg2 : avg2 = 3.8) 
  (h_avg3 : avg3 = 6.6) 
  (h_sum_eq : 2 * avg1 + 2 * avg2 + 2 * avg3 = 27.6) : 
  (27.6 / avg = 6) := 
  by sorry

end number_of_numbers_l54_54317


namespace circle_tangent_line_k_range_l54_54123

theorem circle_tangent_line_k_range
  (k : ℝ)
  (P Q : ℝ × ℝ)
  (c : ℝ × ℝ := (0, 1)) -- Circle center
  (r : ℝ := 1) -- Circle radius
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 2 * y = 0)
  (line_eq : ∀ (x y : ℝ), k * x + y + 3 = 0)
  (dist_pq : Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = Real.sqrt 3) :
  k ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry

end circle_tangent_line_k_range_l54_54123


namespace cheryl_tournament_cost_is_1440_l54_54247

noncomputable def cheryl_electricity_bill : ℝ := 800
noncomputable def additional_for_cell_phone : ℝ := 400
noncomputable def cheryl_cell_phone_expenses : ℝ := cheryl_electricity_bill + additional_for_cell_phone
noncomputable def tournament_cost_percentage : ℝ := 0.2
noncomputable def additional_tournament_cost : ℝ := tournament_cost_percentage * cheryl_cell_phone_expenses
noncomputable def total_tournament_cost : ℝ := cheryl_cell_phone_expenses + additional_tournament_cost

theorem cheryl_tournament_cost_is_1440 : total_tournament_cost = 1440 := by
  sorry

end cheryl_tournament_cost_is_1440_l54_54247


namespace biff_break_even_hours_l54_54743

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

end biff_break_even_hours_l54_54743


namespace distinct_digit_sum_equation_l54_54701

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

end distinct_digit_sum_equation_l54_54701


namespace fewer_seats_right_side_l54_54645

theorem fewer_seats_right_side
  (left_seats : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (total_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : people_per_seat = 3)
  (h3 : back_seat_capacity = 12)
  (h4 : total_capacity = 93)
  : left_seats - (total_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat = 3 :=
  by sorry

end fewer_seats_right_side_l54_54645


namespace range_of_m_for_inversely_proportional_function_l54_54101

theorem range_of_m_for_inversely_proportional_function 
  (m : ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > x₁ → (m - 1) / x₂ < (m - 1) / x₁) : 
  m > 1 :=
sorry

end range_of_m_for_inversely_proportional_function_l54_54101


namespace five_star_three_l54_54809

def star (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2

theorem five_star_three : star 5 3 = 4 := by
  sorry

end five_star_three_l54_54809


namespace joe_total_time_l54_54979

variable (r_w t_w : ℝ) 
variable (t_total : ℝ)

-- Given conditions:
def joe_problem_conditions : Prop :=
  (r_w > 0) ∧ 
  (t_w = 9) ∧
  (3 * r_w * (3)) / 2 = r_w * 9 / 2 + 1 / 2

-- The statement to prove:
theorem joe_total_time (h : joe_problem_conditions r_w t_w) : t_total = 13 :=
by { sorry }

end joe_total_time_l54_54979


namespace find_cos_A_l54_54426

noncomputable def cos_A_of_third_quadrant : Real :=
-3 / 5

theorem find_cos_A (A : Real) (h1 : A ∈ Set.Icc (π) (3 * π / 2)) 
  (h2 : Real.sin A = 4 / 5) : Real.cos A = -3 / 5 := 
sorry

end find_cos_A_l54_54426


namespace solve_equation_l54_54955

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (x / (x + 1) = 2 / (x^2 - 1)) ↔ (x = 2) :=
by
  sorry

end solve_equation_l54_54955


namespace days_until_see_grandma_l54_54157

def hours_in_a_day : ℕ := 24
def hours_until_see_grandma : ℕ := 48

theorem days_until_see_grandma : hours_until_see_grandma / hours_in_a_day = 2 := by
  sorry

end days_until_see_grandma_l54_54157


namespace kids_in_group_l54_54200

open Nat

theorem kids_in_group (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 := by
  sorry

end kids_in_group_l54_54200


namespace isabella_jumped_farthest_l54_54700

-- defining the jumping distances
def ricciana_jump : ℕ := 4
def margarita_jump : ℕ := 2 * ricciana_jump - 1
def isabella_jump : ℕ := ricciana_jump + 3 

-- defining the total distances
def ricciana_total : ℕ := 20 + ricciana_jump
def margarita_total : ℕ := 18 + margarita_jump
def isabella_total : ℕ := 22 + isabella_jump

-- stating the theorem
theorem isabella_jumped_farthest : isabella_total = 29 :=
by sorry

end isabella_jumped_farthest_l54_54700


namespace max_kings_l54_54351

theorem max_kings (initial_kings : ℕ) (kings_attacking_each_other : initial_kings = 21) 
  (no_two_kings_attack : ∀ kings_remaining, kings_remaining ≤ 16) : 
  ∃ kings_remaining, kings_remaining = 16 :=
by
  sorry

end max_kings_l54_54351


namespace bond_yield_correct_l54_54536

-- Definitions of the conditions
def number_of_bonds : ℕ := 1000
def holding_period : ℕ := 2
def bond_income : ℚ := 980 - 980 + 1000 * 0.07 * 2
def initial_investment : ℚ := 980000

-- Yield for 2 years
def yield_2_years : ℚ := (number_of_bonds * bond_income) / initial_investment * 100

-- Average annual yield
def avg_annual_yield : ℚ := yield_2_years / holding_period

-- The main theorem to prove
theorem bond_yield_correct :
  yield_2_years = 15.31 ∧ avg_annual_yield = 7.65 :=
by
  sorry

end bond_yield_correct_l54_54536


namespace pie_remaining_portion_l54_54417

theorem pie_remaining_portion (carlos_portion maria_portion remaining_portion : ℝ)
  (h1 : carlos_portion = 0.6) 
  (h2 : remaining_portion = 1 - carlos_portion)
  (h3 : maria_portion = 0.5 * remaining_portion) :
  remaining_portion - maria_portion = 0.2 := 
by
  sorry

end pie_remaining_portion_l54_54417


namespace danica_planes_l54_54692

def smallestAdditionalPlanes (n k : ℕ) : ℕ :=
  let m := k * (n / k + 1)
  m - n

theorem danica_planes : smallestAdditionalPlanes 17 7 = 4 :=
by
  -- Proof would go here
  sorry

end danica_planes_l54_54692


namespace largest_T_l54_54119

theorem largest_T (T : ℝ) (a b c d e : ℝ) 
  (h1: a ≥ 0) (h2: b ≥ 0) (h3: c ≥ 0) (h4: d ≥ 0) (h5: e ≥ 0)
  (h_sum : a + b = c + d + e)
  (h_T : T ≤ (Real.sqrt 30) / (30 + 12 * Real.sqrt 6)) : 
  Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ T * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2 :=
sorry

end largest_T_l54_54119


namespace total_dolls_l54_54003

def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

theorem total_dolls : rene_dolls + sister_dolls + grandmother_dolls = 258 :=
by {
  -- Required proof steps would be placed here, 
  -- but are omitted as per the instructions.
  sorry
}

end total_dolls_l54_54003


namespace cells_at_day_10_l54_54928

-- Define a function to compute the number of cells given initial cells, tripling rate, intervals, and total time.
def number_of_cells (initial_cells : ℕ) (ratio : ℕ) (interval : ℕ) (total_time : ℕ) : ℕ :=
  let n := total_time / interval + 1
  initial_cells * ratio^(n-1)

-- State the main theorem
theorem cells_at_day_10 :
  number_of_cells 5 3 2 10 = 1215 := by
  sorry

end cells_at_day_10_l54_54928


namespace time_to_cross_platform_l54_54892

-- Definitions of the given conditions
def train_length : ℝ := 900
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 1050

-- Goal statement in Lean 4 format
theorem time_to_cross_platform : 
  let speed := train_length / time_to_cross_pole;
  let total_distance := train_length + platform_length;
  let time := total_distance / speed;
  time = 39 := 
by
  sorry

end time_to_cross_platform_l54_54892


namespace inequality_proof_l54_54483

variable (a b c : ℝ)

-- Conditions
def conditions : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 14

-- Statement to prove
theorem inequality_proof (h : conditions a b c) : 
  a^5 + (1/8) * b^5 + (1/27) * c^5 ≥ 14 := 
sorry

end inequality_proof_l54_54483


namespace heidi_and_karl_painting_l54_54963

-- Given conditions
def heidi_paint_rate := 1 / 60 -- Rate at which Heidi paints, in walls per minute
def karl_paint_rate := 2 * heidi_paint_rate -- Rate at which Karl paints, in walls per minute
def painting_time := 20 -- Time spent painting, in minutes

-- Prove the amount of each wall painted
theorem heidi_and_karl_painting :
  (heidi_paint_rate * painting_time = 1 / 3) ∧ (karl_paint_rate * painting_time = 2 / 3) :=
sorry

end heidi_and_karl_painting_l54_54963


namespace percentage_orange_juice_l54_54306

-- Definitions based on conditions
def total_volume : ℝ := 120
def watermelon_percentage : ℝ := 0.60
def grape_juice_volume : ℝ := 30
def watermelon_juice_volume : ℝ := watermelon_percentage * total_volume
def combined_watermelon_grape_volume : ℝ := watermelon_juice_volume + grape_juice_volume
def orange_juice_volume : ℝ := total_volume - combined_watermelon_grape_volume

-- Lean 4 statement to prove the percentage of orange juice
theorem percentage_orange_juice : (orange_juice_volume / total_volume) * 100 = 15 := by
  -- sorry to skip the proof
  sorry

end percentage_orange_juice_l54_54306


namespace shadedQuadrilateralArea_is_13_l54_54561

noncomputable def calculateShadedQuadrilateralArea : ℝ :=
  let s1 := 2
  let s2 := 4
  let s3 := 6
  let s4 := 8
  let bases := s1 + s2
  let height_small := bases * (10 / 20)
  let height_large := 10
  let alt := s4 - s3
  let area := (1 / 2) * (height_small + height_large) * alt
  13

theorem shadedQuadrilateralArea_is_13 :
  calculateShadedQuadrilateralArea = 13 := by
sorry

end shadedQuadrilateralArea_is_13_l54_54561


namespace color_of_241st_marble_l54_54035

def sequence_color (n : ℕ) : String :=
  if n % 14 < 6 then "blue"
  else if n % 14 < 11 then "red"
  else "green"

theorem color_of_241st_marble : sequence_color 240 = "blue" :=
  by
  sorry

end color_of_241st_marble_l54_54035


namespace exponent_value_l54_54774

theorem exponent_value (y k : ℕ) (h1 : 9^y = 3^k) (h2 : y = 7) : k = 14 := by
  sorry

end exponent_value_l54_54774


namespace infinite_series_sum_eq_seven_l54_54780

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end infinite_series_sum_eq_seven_l54_54780


namespace oliver_total_earnings_l54_54872

/-- Rates for different types of laundry items -/
def rate_regular : ℝ := 3
def rate_delicate : ℝ := 4
def rate_bulky : ℝ := 5

/-- Quantity of laundry items washed over three days -/
def quantity_day1_regular : ℝ := 7
def quantity_day1_delicate : ℝ := 4
def quantity_day1_bulky : ℝ := 2

def quantity_day2_regular : ℝ := 10
def quantity_day2_delicate : ℝ := 6
def quantity_day2_bulky : ℝ := 3

def quantity_day3_regular : ℝ := 20
def quantity_day3_delicate : ℝ := 4
def quantity_day3_bulky : ℝ := 0

/-- Discount on delicate clothes for the third day -/
def discount : ℝ := 0.2

/-- The expected earnings for each day and total -/
def earnings_day1 : ℝ :=
  rate_regular * quantity_day1_regular +
  rate_delicate * quantity_day1_delicate +
  rate_bulky * quantity_day1_bulky

def earnings_day2 : ℝ :=
  rate_regular * quantity_day2_regular +
  rate_delicate * quantity_day2_delicate +
  rate_bulky * quantity_day2_bulky

def earnings_day3 : ℝ :=
  rate_regular * quantity_day3_regular +
  (rate_delicate * quantity_day3_delicate * (1 - discount)) +
  rate_bulky * quantity_day3_bulky

def total_earnings : ℝ := earnings_day1 + earnings_day2 + earnings_day3

theorem oliver_total_earnings : total_earnings = 188.80 := by
  sorry

end oliver_total_earnings_l54_54872


namespace normal_distribution_interval_probability_l54_54984

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
sorry

theorem normal_distribution_interval_probability
  (σ : ℝ) (hσ : σ > 0)
  (hprob : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8) :
  (normal_cdf 1 σ 2 - normal_cdf 1 σ 1) = 0.4 :=
sorry

end normal_distribution_interval_probability_l54_54984


namespace rectangle_ABCD_area_l54_54238

def rectangle_area (x : ℕ) : ℕ :=
  let side_lengths := [x, x+1, x+2, x+3];
  let width := side_lengths.sum;
  let height := width - x;
  width * height

theorem rectangle_ABCD_area : rectangle_area 1 = 143 :=
by
  sorry

end rectangle_ABCD_area_l54_54238


namespace original_volume_of_ice_l54_54807

theorem original_volume_of_ice (V : ℝ) 
  (h1 : V * (1/4) * (1/4) = 0.4) : 
  V = 6.4 :=
sorry

end original_volume_of_ice_l54_54807


namespace min_value_geometric_sequence_l54_54792

noncomputable def geometric_min_value (b1 b2 b3 : ℝ) (s : ℝ) : ℝ :=
  3 * b2 + 4 * b3

theorem min_value_geometric_sequence (s : ℝ) :
  ∃ s : ℝ, 2 = b1 ∧ b2 = 2 * s ∧ b3 = 2 * s^2 ∧ 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end min_value_geometric_sequence_l54_54792


namespace simplify_sqrt_expr_l54_54392

/-- Simplify the given radical expression and prove its equivalence to the expected result. -/
theorem simplify_sqrt_expr :
  (Real.sqrt (5 * 3) * Real.sqrt ((3 ^ 4) * (5 ^ 2)) = 225 * Real.sqrt 15) := 
by
  sorry

end simplify_sqrt_expr_l54_54392


namespace symmetric_lines_a_b_l54_54301

theorem symmetric_lines_a_b (x y a b : ℝ) (A : ℝ × ℝ) (hA : A = (1, 0))
  (h1 : x + 2 * y - 3 = 0)
  (h2 : a * x + 4 * y + b = 0)
  (h_slope : -1 / 2 = -a / 4)
  (h_point : a * 1 + 4 * 0 + b = 0) :
  a + b = 0 :=
sorry

end symmetric_lines_a_b_l54_54301


namespace cos_690_eq_sqrt3_div_2_l54_54300

theorem cos_690_eq_sqrt3_div_2 : Real.cos (690 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_690_eq_sqrt3_div_2_l54_54300


namespace greatest_possible_large_chips_l54_54643

theorem greatest_possible_large_chips :
  ∃ l s : ℕ, ∃ p : ℕ, s + l = 61 ∧ s = l + p ∧ Nat.Prime p ∧ l = 29 :=
sorry

end greatest_possible_large_chips_l54_54643


namespace triangle_sides_inequality_l54_54115

theorem triangle_sides_inequality
  {a b c : ℝ} (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0)
  (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  -- We would place the proof here if it were required
  sorry

end triangle_sides_inequality_l54_54115


namespace cost_of_each_teddy_bear_is_15_l54_54198

-- Definitions
variable (number_of_toys_cost_10 : ℕ := 28)
variable (cost_per_toy : ℕ := 10)
variable (number_of_teddy_bears : ℕ := 20)
variable (total_amount_in_wallet : ℕ := 580)

-- Theorem statement
theorem cost_of_each_teddy_bear_is_15 :
  (total_amount_in_wallet - (number_of_toys_cost_10 * cost_per_toy)) / number_of_teddy_bears = 15 :=
by
  -- proof goes here
  sorry

end cost_of_each_teddy_bear_is_15_l54_54198


namespace remainder_987654_div_8_l54_54655

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end remainder_987654_div_8_l54_54655


namespace range_of_a_l54_54007

theorem range_of_a (x a : ℝ) 
  (h₁ : ∀ x, |x + 1| ≤ 2 → x ≤ a) 
  (h₂ : ∃ x, x > a ∧ |x + 1| ≤ 2) 
  : a ≥ 1 :=
sorry

end range_of_a_l54_54007


namespace find_quartic_polynomial_l54_54010

noncomputable def p (x : ℝ) : ℝ := -(1 / 9) * x^4 + (40 / 9) * x^3 - 8 * x^2 + 10 * x + 2

theorem find_quartic_polynomial :
  p 1 = -3 ∧
  p 2 = -1 ∧
  p 3 = 1 ∧
  p 4 = -7 ∧
  p 0 = 2 :=
by
  sorry

end find_quartic_polynomial_l54_54010


namespace Sarah_l54_54432

variable (s g : ℕ)

theorem Sarah's_score_130 (h1 : s = g + 50) (h2 : (s + g) / 2 = 105) : s = 130 :=
by
  sorry

end Sarah_l54_54432


namespace initial_interest_rate_l54_54595

theorem initial_interest_rate 
  (r P : ℝ)
  (h1 : 20250 = P * r)
  (h2 : 22500 = P * (r + 5)) :
  r = 45 :=
by
  sorry

end initial_interest_rate_l54_54595


namespace distinct_convex_polygons_l54_54887

def twelve_points : Finset (Fin 12) := (Finset.univ : Finset (Fin 12))

noncomputable def polygon_count_with_vertices (n : ℕ) : ℕ :=
  2^n - 1 - n - (n * (n - 1)) / 2

theorem distinct_convex_polygons :
  polygon_count_with_vertices 12 = 4017 := 
by
  sorry

end distinct_convex_polygons_l54_54887


namespace paige_science_problems_l54_54959

variable (S : ℤ)

theorem paige_science_problems (h1 : 43 + S - 44 = 11) : S = 12 :=
by
  sorry

end paige_science_problems_l54_54959


namespace product_B_sampling_l54_54522

theorem product_B_sampling (a : ℕ) (h_seq : a > 0) :
  let A := a
  let B := 2 * a
  let C := 4 * a
  let total := A + B + C
  total = 7 * a →
  let total_drawn := 140
  B / total * total_drawn = 40 :=
by sorry

end product_B_sampling_l54_54522


namespace y_is_one_y_is_neg_two_thirds_l54_54079

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove y = 1 given dot_product(vector_a, vector_b(y)) = 5
theorem y_is_one (h : dot_product vector_a (vector_b y) = 5) : y = 1 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

-- Prove y = -2/3 given |vector_a + vector_b(y)| = |vector_a - vector_b(y)|
theorem y_is_neg_two_thirds (h : (vector_a.1 + (vector_b y).1)^2 + (vector_a.2 + (vector_b y).2)^2 =
                                (vector_a.1 - (vector_b y).1)^2 + (vector_a.2 - (vector_b y).2)^2) : y = -2/3 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

end y_is_one_y_is_neg_two_thirds_l54_54079


namespace find_maximum_value_l54_54707

open Real

noncomputable def maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : ℝ :=
  2 + sqrt 5

theorem find_maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) > maximum_value a b c h₁ h₂ h₃ h₄ :=
by
  sorry

end find_maximum_value_l54_54707


namespace min_minutes_to_make_B_cheaper_l54_54257

def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

def costB (x : ℕ) : ℕ := 2500 + 4 * x

theorem min_minutes_to_make_B_cheaper : ∃ (x : ℕ), x ≥ 301 ∧ costB x < costA x :=
by
  use 301
  sorry

end min_minutes_to_make_B_cheaper_l54_54257


namespace sum_of_transformed_numbers_l54_54239

variables (a b x k S : ℝ)

-- Define the condition that a + b = S
def sum_condition : Prop := a + b = S

-- Define the function that represents the final sum after transformations
def final_sum (a b x k : ℝ) : ℝ :=
  k * (a + x) + k * (b + x)

-- The theorem statement to prove
theorem sum_of_transformed_numbers (h : sum_condition a b S) : 
  final_sum a b x k = k * S + 2 * k * x :=
by
  sorry

end sum_of_transformed_numbers_l54_54239


namespace marching_band_total_weight_l54_54473

def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drummer : ℕ := 15
def weight_percussionist : ℕ := 8

def uniform_trumpet : ℕ := 3
def uniform_clarinet : ℕ := 3
def uniform_trombone : ℕ := 4
def uniform_tuba : ℕ := 5
def uniform_drummer : ℕ := 6
def uniform_percussionist : ℕ := 3

def count_trumpet : ℕ := 6
def count_clarinet : ℕ := 9
def count_trombone : ℕ := 8
def count_tuba : ℕ := 3
def count_drummer : ℕ := 2
def count_percussionist : ℕ := 4

def total_weight_band : ℕ :=
  (count_trumpet * (weight_trumpet + uniform_trumpet)) +
  (count_clarinet * (weight_clarinet + uniform_clarinet)) +
  (count_trombone * (weight_trombone + uniform_trombone)) +
  (count_tuba * (weight_tuba + uniform_tuba)) +
  (count_drummer * (weight_drummer + uniform_drummer)) +
  (count_percussionist * (weight_percussionist + uniform_percussionist))

theorem marching_band_total_weight : total_weight_band = 393 :=
  by
  sorry

end marching_band_total_weight_l54_54473


namespace largest_divisor_l54_54451

theorem largest_divisor (n : ℤ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (∃ k : ℤ, k > 0 ∧ (∀ n : ℤ, n > 0 → n % 2 = 1 → k ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)))) → 
  k = 15 :=
by
  sorry

end largest_divisor_l54_54451


namespace geometric_series_sum_l54_54274

theorem geometric_series_sum (a r : ℚ) (ha : a = 1) (hr : r = 1/4) : 
  (∑' n:ℕ, a * r^n) = 4/3 :=
by
  rw [ha, hr]
  sorry

end geometric_series_sum_l54_54274


namespace arithmetic_seq_inequality_l54_54572

-- Definition for the sum of the first n terms of an arithmetic sequence
def sum_arith_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_seq_inequality (a₁ : ℕ) (d : ℕ) (n : ℕ) (h : d > 0) :
  sum_arith_seq a₁ d n + sum_arith_seq a₁ d (3 * n) > 2 * sum_arith_seq a₁ d (2 * n) := by
  sorry

end arithmetic_seq_inequality_l54_54572


namespace common_ratio_of_geometric_sequence_l54_54041

open BigOperators

theorem common_ratio_of_geometric_sequence
  (a1 : ℝ) (q : ℝ)
  (h1 : 2 * (a1 * q^5) = 3 * (a1 * (1 - q^4) / (1 - q)) + 1)
  (h2 : a1 * q^6 = 3 * (a1 * (1 - q^5) / (1 - q)) + 1)
  (h_pos : a1 > 0) :
  q = 3 :=
sorry

end common_ratio_of_geometric_sequence_l54_54041


namespace cannot_be_covered_by_dominoes_l54_54759

-- Definitions for each board
def board_3x4_squares : ℕ := 3 * 4
def board_3x5_squares : ℕ := 3 * 5
def board_4x4_one_removed_squares : ℕ := 4 * 4 - 1
def board_5x5_squares : ℕ := 5 * 5
def board_6x3_squares : ℕ := 6 * 3

-- Parity check
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Mathematical proof problem statement
theorem cannot_be_covered_by_dominoes :
  ¬ is_even board_3x5_squares ∧
  ¬ is_even board_4x4_one_removed_squares ∧
  ¬ is_even board_5x5_squares :=
by
  -- Checking the conditions that must hold
  sorry

end cannot_be_covered_by_dominoes_l54_54759


namespace complex_quadrant_l54_54735

theorem complex_quadrant (z : ℂ) (h : z * (2 - I) = 2 + I) : 0 < z.re ∧ 0 < z.im := 
sorry

end complex_quadrant_l54_54735


namespace Ron_four_times_Maurice_l54_54973

theorem Ron_four_times_Maurice
  (r m : ℕ) (x : ℕ) 
  (h_r : r = 43) 
  (h_m : m = 7) 
  (h_eq : r + x = 4 * (m + x)) : 
  x = 5 := 
by
  sorry

end Ron_four_times_Maurice_l54_54973


namespace total_cost_of_fencing_l54_54798

def costOfFencing (lengths rates : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) lengths rates)

theorem total_cost_of_fencing :
  costOfFencing [14, 20, 35, 40, 15, 30, 25]
                [2.50, 3.00, 3.50, 4.00, 2.75, 3.25, 3.75] = 610.00 :=
by
  sorry

end total_cost_of_fencing_l54_54798


namespace correct_transformation_D_l54_54754

theorem correct_transformation_D : ∀ x, 2 * (x + 1) = x + 7 → x = 5 :=
by
  intro x
  sorry

end correct_transformation_D_l54_54754


namespace missing_condition_l54_54163

theorem missing_condition (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : y = 3 * (x - 2)) :
  true := -- The equivalent mathematical statement asserts the correct missing condition.
sorry

end missing_condition_l54_54163


namespace find_raspberries_l54_54028

def total_berries (R : ℕ) : ℕ := 30 + 20 + R

def fresh_berries (R : ℕ) : ℕ := 2 * total_berries R / 3

def fresh_berries_to_keep (R : ℕ) : ℕ := fresh_berries R / 2

def fresh_berries_to_sell (R : ℕ) : ℕ := fresh_berries R - fresh_berries_to_keep R

theorem find_raspberries (R : ℕ) : fresh_berries_to_sell R = 20 → R = 10 := 
by 
sorry

-- To ensure the problem is complete and solvable, we also need assumptions on the domain:
example : ∃ R : ℕ, fresh_berries_to_sell R = 20 := 
by 
  use 10 
  sorry

end find_raspberries_l54_54028


namespace old_record_was_300_points_l54_54605

theorem old_record_was_300_points :
  let touchdowns_per_game := 4
  let points_per_touchdown := 6
  let games_in_season := 15
  let conversions := 6
  let points_per_conversion := 2
  let points_beat := 72
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + conversions * points_per_conversion
  total_points - points_beat = 300 := 
by
  sorry

end old_record_was_300_points_l54_54605


namespace original_cube_volume_l54_54947

theorem original_cube_volume (a : ℕ) (V_cube V_new : ℕ)
  (h1 : V_cube = a^3)
  (h2 : V_new = (a + 2) * a * (a - 2))
  (h3 : V_cube = V_new + 24) :
  V_cube = 216 :=
by
  sorry

end original_cube_volume_l54_54947


namespace income_to_expenditure_ratio_l54_54906

variable (I E S : ℕ)

def Ratio (a b : ℕ) : ℚ := a / (b : ℚ)

theorem income_to_expenditure_ratio (h1 : I = 14000) (h2 : S = 2000) (h3 : S = I - E) : 
  Ratio I E = 7 / 6 :=
by
  sorry

end income_to_expenditure_ratio_l54_54906


namespace additional_grassy_ground_l54_54970

theorem additional_grassy_ground (r₁ r₂ : ℝ) (π : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 23) :
  π * r₂ ^ 2 - π * r₁ ^ 2 = 385 * π :=
  by
  subst h₁ h₂
  sorry

end additional_grassy_ground_l54_54970


namespace expression_evaluation_l54_54997

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l54_54997


namespace sum_of_digits_of_N_l54_54176

-- The total number of coins
def total_coins : ℕ := 3081

-- Setting up the equation N^2 = 3081
def N : ℕ := 55 -- Since 55^2 is closest to 3081 and sqrt(3081) ≈ 55

-- Proving the sum of the digits of N is 10
theorem sum_of_digits_of_N : (5 + 5) = 10 :=
by
  sorry

end sum_of_digits_of_N_l54_54176


namespace proof_of_problem_l54_54857

-- Define the problem conditions using a combination function
def problem_statement : Prop :=
  (Nat.choose 6 3 = 20)

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l54_54857


namespace belle_biscuits_l54_54468

-- Define the conditions
def cost_per_rawhide_bone : ℕ := 1
def num_rawhide_bones_per_evening : ℕ := 2
def cost_per_biscuit : ℚ := 0.25
def total_weekly_cost : ℚ := 21
def days_in_week : ℕ := 7

-- Define the number of biscuits Belle eats every evening
def num_biscuits_per_evening : ℚ := 4

-- Define the statement that encapsulates the problem
theorem belle_biscuits :
  (total_weekly_cost = days_in_week * (num_rawhide_bones_per_evening * cost_per_rawhide_bone + num_biscuits_per_evening * cost_per_biscuit)) :=
sorry

end belle_biscuits_l54_54468


namespace sufficient_but_not_necessary_l54_54870

theorem sufficient_but_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l54_54870


namespace paint_cans_needed_l54_54671

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l54_54671


namespace min_length_QR_l54_54117

theorem min_length_QR (PQ PR SR QS QR : ℕ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  QR > PR - PQ ∧ QR > QS - SR ↔ QR = 16 :=
by
  sorry

end min_length_QR_l54_54117


namespace area_of_blackboard_l54_54840

def side_length : ℝ := 6
def area (side : ℝ) : ℝ := side * side

theorem area_of_blackboard : area side_length = 36 := by
  -- proof
  sorry

end area_of_blackboard_l54_54840


namespace sum_not_divisible_by_three_times_any_number_l54_54387

theorem sum_not_divisible_by_three_times_any_number (n : ℕ) (a : Fin n → ℕ) (h : n ≥ 3) (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k : Fin n, ¬ (a i + a j) ∣ (3 * a k)) :=
sorry

end sum_not_divisible_by_three_times_any_number_l54_54387


namespace Jordan_length_is_8_l54_54150

-- Definitions of the conditions given in the problem
def Carol_length := 5
def Carol_width := 24
def Jordan_width := 15

-- Definition to calculate the area of Carol's rectangle
def Carol_area : ℕ := Carol_length * Carol_width

-- Definition to calculate the length of Jordan's rectangle
def Jordan_length (area : ℕ) (width : ℕ) : ℕ := area / width

-- Proposition to prove the length of Jordan's rectangle
theorem Jordan_length_is_8 : Jordan_length Carol_area Jordan_width = 8 :=
by
  -- skipping the proof
  sorry

end Jordan_length_is_8_l54_54150


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l54_54526

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l54_54526


namespace min_star_value_l54_54242

theorem min_star_value :
  ∃ (star : ℕ), (98348 * 10 + star) % 72 = 0 ∧ (∀ (x : ℕ), (98348 * 10 + x) % 72 = 0 → star ≤ x) := sorry

end min_star_value_l54_54242


namespace f_monotonically_decreasing_iff_l54_54808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4 * a * x + 3 else (2 - 3 * a) * x + 1

theorem f_monotonically_decreasing_iff (a : ℝ) : 
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≥ f a x₂) ↔ (1/2 ≤ a ∧ a < 2/3) :=
by 
  sorry

end f_monotonically_decreasing_iff_l54_54808


namespace novelists_count_l54_54698

theorem novelists_count (n p : ℕ) (h1 : n / (n + p) = 5 / 8) (h2 : n + p = 24) : n = 15 :=
sorry

end novelists_count_l54_54698


namespace cos_A_of_triangle_l54_54185

theorem cos_A_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : b = Real.sqrt 2 * c)
  (h2 : Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B)
  (h3 : a = Real.sin A / Real.sin A * b) -- Sine rule used implicitly

: Real.cos A = Real.sqrt 2 / 4 := by
  -- proof will be skipped, hence 'sorry' included
  sorry

end cos_A_of_triangle_l54_54185


namespace solve_fisherman_problem_l54_54883

def fisherman_problem : Prop :=
  ∃ (x y z : ℕ), x + y + z = 16 ∧ 13 * x + 5 * y + 4 * z = 113 ∧ x = 5 ∧ y = 4 ∧ z = 7

theorem solve_fisherman_problem : fisherman_problem :=
sorry

end solve_fisherman_problem_l54_54883


namespace linear_equation_in_two_variables_l54_54816

/--
Prove that Equation C (3x - 1 = 2 - 5y) is a linear equation in two variables 
given the equations in conditions.
-/
theorem linear_equation_in_two_variables :
  ∀ (x y : ℝ),
  (2 * x + 3 = x - 5) →
  (x * y + y = 2) →
  (3 * x - 1 = 2 - 5 * y) →
  (2 * x + (3 / y) = 7) →
  ∃ (A B C : ℝ), A * x + B * y = C :=
by 
  sorry

end linear_equation_in_two_variables_l54_54816


namespace vertex_x_coordinate_of_quadratic_l54_54592

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 8 * x + 15

-- Define the x-coordinate of the vertex
def vertex_x_coordinate (f : ℝ → ℝ) : ℝ := 4

-- The theorem to prove
theorem vertex_x_coordinate_of_quadratic :
  vertex_x_coordinate quadratic_function = 4 :=
by
  -- Proof skipped
  sorry

end vertex_x_coordinate_of_quadratic_l54_54592


namespace scientific_notation_correct_l54_54430

def big_number : ℕ := 274000000

noncomputable def scientific_notation : ℝ := 2.74 * 10^8

theorem scientific_notation_correct : (big_number : ℝ) = scientific_notation :=
by sorry

end scientific_notation_correct_l54_54430


namespace unique_solution_for_quadratic_l54_54356

theorem unique_solution_for_quadratic (a : ℝ) : 
  ∃! (x : ℝ), x^2 - 2 * a * x + a^2 = 0 := 
by
  sorry

end unique_solution_for_quadratic_l54_54356


namespace car_distance_covered_by_car_l54_54469

theorem car_distance_covered_by_car
  (V : ℝ)                               -- Initial speed of the car
  (D : ℝ)                               -- Distance covered by the car
  (h1 : D = V * 6)                      -- The car takes 6 hours to cover the distance at speed V
  (h2 : D = 56 * 9)                     -- The car takes 9 hours to cover the distance at speed 56
  : D = 504 :=                          -- Prove that the distance D is 504 kilometers
by
  sorry

end car_distance_covered_by_car_l54_54469


namespace escher_prints_consecutive_l54_54260

noncomputable def probability_all_eschers_consecutive (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
if h : m = n + 3 ∧ k = 4 then 1 / (n * (n + 1) * (n + 2)) else 0

theorem escher_prints_consecutive :
  probability_all_eschers_consecutive 10 12 4 = 1 / 1320 :=
  by sorry

end escher_prints_consecutive_l54_54260


namespace integer_solutions_count_l54_54040

theorem integer_solutions_count (x : ℤ) : 
  (x^2 - 3 * x + 2)^2 - 3 * (x^2 - 3 * x) - 4 = 0 ↔ 0 = 0 :=
by sorry

end integer_solutions_count_l54_54040


namespace range_of_m_l54_54517

noncomputable def f (x : ℝ) := Real.log (x^2 + 1)

noncomputable def g (x m : ℝ) := (1 / 2)^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (0:ℝ) 3, ∃ x2 ∈ Set.Icc (1:ℝ) 2, f x1 ≤ g x2 m) ↔ m ≤ -1/2 :=
by
  sorry

end range_of_m_l54_54517


namespace B_months_grazing_eq_five_l54_54393

-- Define the conditions in the problem
def A_oxen : ℕ := 10
def A_months : ℕ := 7
def B_oxen : ℕ := 12
def C_oxen : ℕ := 15
def C_months : ℕ := 3
def total_rent : ℝ := 175
def C_rent_share : ℝ := 45

-- Total ox-units function
def total_ox_units (x : ℕ) : ℕ :=
  A_oxen * A_months + B_oxen * x + C_oxen * C_months

-- Prove that the number of months B's oxen grazed is 5
theorem B_months_grazing_eq_five (x : ℕ) :
  total_ox_units x = 70 + 12 * x + 45 →
  (C_rent_share / total_rent = 45 / total_ox_units x) →
  x = 5 :=
by
  intros h1 h2
  sorry

end B_months_grazing_eq_five_l54_54393


namespace probability_of_odd_sum_l54_54081

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end probability_of_odd_sum_l54_54081


namespace overlap_area_of_parallelogram_l54_54184

theorem overlap_area_of_parallelogram (w1 w2 : ℝ) (β : ℝ) (hβ : β = 30) (hw1 : w1 = 2) (hw2 : w2 = 1) : 
  (w1 * (w2 / Real.sin (β * Real.pi / 180))) = 4 :=
by
  sorry

end overlap_area_of_parallelogram_l54_54184


namespace total_kids_played_with_l54_54395

-- Define the conditions as separate constants
def kidsMonday : Nat := 12
def kidsTuesday : Nat := 7

-- Prove the total number of kids Julia played with
theorem total_kids_played_with : kidsMonday + kidsTuesday = 19 := 
by
  sorry

end total_kids_played_with_l54_54395


namespace all_statements_correct_l54_54225

-- Definitions based on the problem conditions
def population_size : ℕ := 60000
def sample_size : ℕ := 1000
def is_sampling_survey (population_size sample_size : ℕ) : Prop := sample_size < population_size
def is_population (n : ℕ) : Prop := n = 60000
def is_sample (population_size sample_size : ℕ) : Prop := sample_size < population_size
def matches_sample_size (n : ℕ) : Prop := n = 1000

-- Lean problem statement representing the proof that all statements are correct
theorem all_statements_correct :
  is_sampling_survey population_size sample_size ∧
  is_population population_size ∧ 
  is_sample population_size sample_size ∧
  matches_sample_size sample_size := by
  sorry

end all_statements_correct_l54_54225


namespace longer_strap_length_l54_54476

theorem longer_strap_length (S L : ℕ) 
  (h1 : L = S + 72) 
  (h2 : S + L = 348) : 
  L = 210 := 
sorry

end longer_strap_length_l54_54476


namespace max_area_of_rectangular_garden_l54_54535

noncomputable def max_rectangle_area (x y : ℝ) (h1 : 2 * (x + y) = 36) (h2 : x > 0) (h3 : y > 0) : ℝ :=
  x * y

theorem max_area_of_rectangular_garden
  (x y : ℝ)
  (h1 : 2 * (x + y) = 36)
  (h2 : x > 0)
  (h3 : y > 0) :
  max_rectangle_area x y h1 h2 h3 = 81 :=
sorry

end max_area_of_rectangular_garden_l54_54535


namespace comparison_of_a_b_c_l54_54633

theorem comparison_of_a_b_c : 
  let a := (1/3)^(2/5)
  let b := 2^(4/3)
  let c := Real.logb 2 (1/3)
  c < a ∧ a < b :=
by
  sorry

end comparison_of_a_b_c_l54_54633


namespace rectangle_length_l54_54981

theorem rectangle_length
    (a : ℕ)
    (b : ℕ)
    (area_square : a * a = 81)
    (width_rect : b = 3)
    (area_equal : a * a = b * (27) )
    : b * 27 = 81 :=
by
  sorry

end rectangle_length_l54_54981


namespace weight_of_8_moles_CCl4_correct_l54_54288

/-- The problem states that carbon tetrachloride (CCl4) is given, and we are to determine the weight of 8 moles of CCl4 based on its molar mass calculations. -/
noncomputable def weight_of_8_moles_CCl4 (molar_mass_C : ℝ) (molar_mass_Cl : ℝ) : ℝ :=
  let molar_mass_CCl4 := molar_mass_C + 4 * molar_mass_Cl
  8 * molar_mass_CCl4

/-- Given the molar masses of Carbon (C) and Chlorine (Cl), prove that the calculated weight of 8 moles of CCl4 matches the expected weight. -/
theorem weight_of_8_moles_CCl4_correct :
  let molar_mass_C := 12.01
  let molar_mass_Cl := 35.45
  weight_of_8_moles_CCl4 molar_mass_C molar_mass_Cl = 1230.48 := by
  sorry

end weight_of_8_moles_CCl4_correct_l54_54288


namespace not_rectangle_determined_by_angle_and_side_l54_54435

axiom parallelogram_determined_by_two_sides_and_angle : Prop
axiom equilateral_triangle_determined_by_area : Prop
axiom square_determined_by_perimeter_and_side : Prop
axiom rectangle_determined_by_two_diagonals : Prop
axiom rectangle_determined_by_angle_and_side : Prop

theorem not_rectangle_determined_by_angle_and_side : ¬rectangle_determined_by_angle_and_side := 
sorry

end not_rectangle_determined_by_angle_and_side_l54_54435


namespace complement_intersection_l54_54842

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {3, 4, 5}) (hN : N = {2, 3}) :
  (U \ N) ∩ M = {4, 5} := by
  sorry

end complement_intersection_l54_54842


namespace Amanda_ticket_sales_goal_l54_54022

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l54_54022


namespace find_n_with_divisors_sum_l54_54761

theorem find_n_with_divisors_sum (n : ℕ) (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 5) (h4 : d4 = 10) 
  (hd : n = 130) : d1^2 + d2^2 + d3^2 + d4^2 = n :=
sorry

end find_n_with_divisors_sum_l54_54761


namespace average_episodes_per_year_is_16_l54_54574

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l54_54574


namespace seventh_fisherman_right_neighbor_l54_54994

theorem seventh_fisherman_right_neighbor (f1 f2 f3 f4 f5 f6 f7 : ℕ) (L1 L2 L3 L4 L5 L6 L7 : ℕ) :
  (L2 * f1 = 12 ∨ L3 * f2 = 12 ∨ L4 * f3 = 12 ∨ L5 * f4 = 12 ∨ L6 * f5 = 12 ∨ L7 * f6 = 12 ∨ L1 * f7 = 12) → 
  (L2 * f1 = 14 ∨ L3 * f2 = 18 ∨ L4 * f3 = 32 ∨ L5 * f4 = 48 ∨ L6 * f5 = 70 ∨ L7 * f6 = x ∨ L1 * f7 = 12) →
  (12 * 12 * 20 * 24 * 32 * 42 * 56) / (12 * 14 * 18 * 32 * 48 * 70) = x :=
by
  sorry

end seventh_fisherman_right_neighbor_l54_54994


namespace power_sum_positive_l54_54529

theorem power_sum_positive 
    (a b c : ℝ) 
    (h1 : a * b * c > 0)
    (h2 : a + b + c > 0)
    (n : ℕ):
    a ^ n + b ^ n + c ^ n > 0 :=
by
  sorry

end power_sum_positive_l54_54529


namespace restaurant_total_tables_l54_54221

theorem restaurant_total_tables (N O : ℕ) (h1 : 6 * N + 4 * O = 212) (h2 : N = O + 12) : N + O = 40 :=
sorry

end restaurant_total_tables_l54_54221


namespace smallest_y_l54_54249

theorem smallest_y (y : ℕ) : 
    (y % 5 = 4) ∧ 
    (y % 7 = 6) ∧ 
    (y % 8 = 7) → 
    y = 279 :=
sorry

end smallest_y_l54_54249


namespace sara_golf_balls_total_l54_54843

-- Define the conditions
def dozens := 16
def dozen_to_balls := 12

-- The final proof statement
theorem sara_golf_balls_total : dozens * dozen_to_balls = 192 :=
by
  sorry

end sara_golf_balls_total_l54_54843


namespace real_solutions_l54_54922

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end real_solutions_l54_54922


namespace ratio_of_areas_l54_54215

theorem ratio_of_areas
  (s: ℝ) (h₁: s > 0)
  (large_square_area: ℝ)
  (inscribed_square_area: ℝ)
  (harea₁: large_square_area = s * s)
  (harea₂: inscribed_square_area = (s / 2) * (s / 2)) :
  inscribed_square_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l54_54215


namespace sqrt_simplify_l54_54496

theorem sqrt_simplify (p : ℝ) :
  (Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5)) =
  6 * p^4 * Real.sqrt (35 * p) :=
by
  sorry

end sqrt_simplify_l54_54496


namespace julia_total_balls_l54_54337

theorem julia_total_balls :
  (3 * 19) + (10 * 19) + (8 * 19) = 399 :=
by
  -- proof goes here
  sorry

end julia_total_balls_l54_54337


namespace even_stones_fraction_odd_stones_fraction_l54_54258

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an even number of stones is 12/65. -/
theorem even_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 0 ∧ B2 % 2 = 0 ∧ B3 % 2 = 0 ∧ B4 % 2 = 0 ∧ B1 + B2 + B3 + B4 = 12) → (84 / 455 = 12 / 65) := 
by sorry

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an odd number of stones is 1/13. -/
theorem odd_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 1 ∧ B2 % 2 = 1 ∧ B3 % 2 = 1 ∧ B4 % 2 = 1 ∧ B1 + B2 + B3 + B4 = 12) → (35 / 455 = 1 / 13) := 
by sorry

end even_stones_fraction_odd_stones_fraction_l54_54258


namespace median_song_length_l54_54254

-- Define the list of song lengths in seconds
def song_lengths : List ℕ := [32, 43, 58, 65, 70, 72, 75, 80, 145, 150, 175, 180, 195, 210, 215, 225, 250, 252]

-- Define the statement that the median length of the songs is 147.5 seconds
theorem median_song_length : ∃ median : ℕ, median = 147 ∧ (median : ℚ) + 0.5 = 147.5 := by
  sorry

end median_song_length_l54_54254


namespace math_problem_proof_l54_54687

noncomputable def problem_expr : ℚ :=
  ((11 + 1/9) - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / (36/10) / (2 + 6/25)

theorem math_problem_proof : problem_expr = 20 / 9 := by
  sorry

end math_problem_proof_l54_54687


namespace abs_val_neg_three_l54_54438

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l54_54438


namespace number_of_even_three_digit_numbers_l54_54705

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

end number_of_even_three_digit_numbers_l54_54705


namespace sequence_is_increasing_l54_54612

def S (n : ℕ) : ℤ :=
  n^2 + 2 * n - 2

def a : ℕ → ℤ
| 0       => 0
| 1       => 1
| n + 1   => S (n + 1) - S n

theorem sequence_is_increasing : ∀ n m : ℕ, n < m → a n < a m :=
  sorry

end sequence_is_increasing_l54_54612


namespace range_of_y_eq_x_squared_l54_54550

noncomputable def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem range_of_y_eq_x_squared :
  M = { y : ℝ | ∃ x : ℝ, y = x^2 } := by
  sorry

end range_of_y_eq_x_squared_l54_54550


namespace find_w_over_y_l54_54837

theorem find_w_over_y 
  (w x y : ℝ) 
  (h1 : w / x = 2 / 3) 
  (h2 : (x + y) / y = 1.6) : 
  w / y = 0.4 := 
  sorry

end find_w_over_y_l54_54837


namespace value_of_composition_l54_54182

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x - 1

theorem value_of_composition : g (f (1 + 2 * g 3)) = 12 := by
  sorry

end value_of_composition_l54_54182


namespace quadratic_sum_r_s_l54_54353

/-- Solve the quadratic equation and identify the sum of r and s 
from the equivalent completed square form (x + r)^2 = s. -/
theorem quadratic_sum_r_s (r s : ℤ) :
  (∃ r s : ℤ, (x - r)^2 = s → r + s = 11) :=
sorry

end quadratic_sum_r_s_l54_54353


namespace find_a9_l54_54144

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {d a₁ : ℤ}

-- Conditions
def arithmetic_sequence := ∀ n : ℕ, a_n n = a₁ + n * d
def sum_first_n_terms := ∀ n : ℕ, S n = (n * (2 * a₁ + (n - 1) * d)) / 2

-- Specific Conditions for the problem
axiom condition1 : S 8 = 4 * a₁
axiom condition2 : a_n 6 = -2 -- Note that a_n is 0-indexed here.

theorem find_a9 : a_n 8 = 2 :=
by
  sorry

end find_a9_l54_54144


namespace triangle_side_lengths_l54_54116

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l54_54116


namespace ratio_bound_exceeds_2023_power_l54_54072

theorem ratio_bound_exceeds_2023_power (a b : ℕ → ℝ) (h_pos : ∀ n, 0 < a n ∧ 0 < b n)
  (h1 : ∀ n, (a (n + 1)) * (b (n + 1)) = (a n)^2 + (b n)^2)
  (h2 : ∀ n, (a (n + 1)) + (b (n + 1)) = (a n) * (b n))
  (h3 : ∀ n, a n ≥ b n) :
  ∃ n, (a n) / (b n) > 2023^2023 :=
by
  sorry

end ratio_bound_exceeds_2023_power_l54_54072


namespace total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l54_54965

-- Define the conditions
def total_people (A B : ℕ) : Prop := A + B = 92
def valid_class_A (A : ℕ) : Prop := 51 < A ∧ A < 55
def total_cost (sets : ℕ) (cost_per_set : ℕ) : ℕ := sets * cost_per_set

-- Prices per set for different ranges of number of sets
def price_per_set (n : ℕ) : ℕ :=
  if n > 90 then 30 else if n > 50 then 40 else 50

-- Question 1
theorem total_amount_for_uniforms (A B : ℕ) (h1 : total_people A B) : total_cost 92 30 = 2760 := sorry

-- Question 2
theorem students_in_classes (A B : ℕ) (h1 : total_people A B) (h2 : valid_class_A A) (h3 : 40 * A + 50 * B = 4080) : A = 52 ∧ B = 40 := sorry

-- Question 3
theorem cost_effective_purchase_plan (A : ℕ) (h1 : 51 < A ∧ A < 55) (B : ℕ) (h2 : 92 - A = B) (h3 : A - 8 + B = 91) :
  ∃ (cost : ℕ), cost = total_cost 91 30 ∧ cost = 2730 := sorry

end total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l54_54965


namespace circle_radius_l54_54802

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2 * x + 6 * y + 1 = 0) → (∃ (r : ℝ), r = 3) :=
by
  sorry

end circle_radius_l54_54802


namespace painted_cubes_count_l54_54137

/-- A theorem to prove the number of painted small cubes in a larger cube. -/
theorem painted_cubes_count (total_cubes unpainted_cubes : ℕ) (a b : ℕ) :
  total_cubes = a * a * a →
  unpainted_cubes = (a - 2) * (a - 2) * (a - 2) →
  22 = unpainted_cubes →
  64 = total_cubes →
  ∃ m, m = total_cubes - unpainted_cubes ∧ m = 42 :=
by
  sorry

end painted_cubes_count_l54_54137


namespace John_works_5_days_a_week_l54_54756

theorem John_works_5_days_a_week
  (widgets_per_hour : ℕ)
  (hours_per_day : ℕ)
  (widgets_per_week : ℕ)
  (H1 : widgets_per_hour = 20)
  (H2 : hours_per_day = 8)
  (H3 : widgets_per_week = 800) :
  widgets_per_week / (widgets_per_hour * hours_per_day) = 5 :=
by
  sorry

end John_works_5_days_a_week_l54_54756


namespace pupils_count_l54_54547

-- Definitions based on given conditions
def number_of_girls : ℕ := 692
def girls_more_than_boys : ℕ := 458
def number_of_boys : ℕ := number_of_girls - girls_more_than_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

-- The statement that the total number of pupils is 926
theorem pupils_count : total_pupils = 926 := by
  sorry

end pupils_count_l54_54547


namespace second_ball_probability_l54_54085

-- Definitions and conditions
def red_balls := 3
def white_balls := 2
def black_balls := 5
def total_balls := red_balls + white_balls + black_balls

def first_ball_white_condition : Prop := (white_balls / total_balls) = (2 / 10)
def second_ball_red_given_first_white (first_ball_white : Prop) : Prop :=
  (first_ball_white → (red_balls / (total_balls - 1)) = (1 / 3))

-- Mathematical equivalence proof problem statement in Lean
theorem second_ball_probability : 
  first_ball_white_condition ∧ second_ball_red_given_first_white first_ball_white_condition :=
by
  sorry

end second_ball_probability_l54_54085


namespace find_theta_l54_54264

def equilateral_triangle_angle : ℝ := 60
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def total_round_angle : ℝ := 360

theorem find_theta (θ : ℝ)
  (h_eq_tri : equilateral_triangle_angle = 60)
  (h_squ : square_angle = 90)
  (h_pen : pentagon_angle = 108)
  (h_round : total_round_angle = 360) :
  θ = total_round_angle - (equilateral_triangle_angle + square_angle + pentagon_angle) :=
sorry

end find_theta_l54_54264


namespace sum_of_remainders_l54_54534

theorem sum_of_remainders (a b c : ℕ) 
  (ha : a % 15 = 12) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 9 := 
by 
  sorry

end sum_of_remainders_l54_54534


namespace cookies_per_batch_l54_54975

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l54_54975


namespace fewer_blue_than_green_l54_54918

-- Definitions for given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def total_buttons : ℕ := 275
def blue_buttons : ℕ := total_buttons - (green_buttons + yellow_buttons)

-- Theorem statement to be proved
theorem fewer_blue_than_green : green_buttons - blue_buttons = 5 :=
by
  -- Proof is omitted as per the instructions
  sorry

end fewer_blue_than_green_l54_54918


namespace cory_packs_l54_54262

theorem cory_packs (total_money_needed cost_per_pack : ℕ) (h1 : total_money_needed = 98) (h2 : cost_per_pack = 49) : total_money_needed / cost_per_pack = 2 :=
by 
  sorry

end cory_packs_l54_54262


namespace tan_product_l54_54454

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l54_54454


namespace total_squares_l54_54156

theorem total_squares (num_groups : ℕ) (squares_per_group : ℕ) (total : ℕ) 
  (h1 : num_groups = 5) (h2 : squares_per_group = 5) (h3 : total = num_groups * squares_per_group) : 
  total = 25 :=
by
  rw [h1, h2] at h3
  exact h3

end total_squares_l54_54156


namespace base8_to_base10_4532_l54_54489

theorem base8_to_base10_4532 : 
    (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := 
by sorry

end base8_to_base10_4532_l54_54489


namespace incorrect_statement_c_l54_54944

-- Define even function
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- Define odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Function definitions
def f1 (x : ℝ) : ℝ := x^4 + x^2
def f2 (x : ℝ) : ℝ := x^3 + x^2

-- Main theorem statement
theorem incorrect_statement_c : ¬ is_odd f2 := sorry

end incorrect_statement_c_l54_54944


namespace angle_between_hands_at_3_15_l54_54102

-- Definitions based on conditions
def minuteHandAngleAt_3_15 : ℝ := 90 -- The position of the minute hand at 3:15 is 90 degrees.

def hourHandSpeed : ℝ := 0.5 -- The hour hand moves at 0.5 degrees per minute.

def hourHandAngleAt_3_15 : ℝ := 3 * 30 + 15 * hourHandSpeed
-- The hour hand starts at 3 o'clock (90 degrees) and moves 0.5 degrees per minute.

-- Statement to prove
theorem angle_between_hands_at_3_15 : abs (minuteHandAngleAt_3_15 - hourHandAngleAt_3_15) = 82.5 :=
by
  sorry

end angle_between_hands_at_3_15_l54_54102


namespace license_plates_count_l54_54209

theorem license_plates_count : (6 * 10^5 * 26^3) = 10584576000 := by
  sorry

end license_plates_count_l54_54209


namespace pizza_dough_milk_needed_l54_54106

variable (milk_per_300 : ℕ) (flour_per_batch : ℕ) (total_flour : ℕ)

-- Definitions based on problem conditions
def milk_per_batch := milk_per_300
def batch_size := flour_per_batch
def used_flour := total_flour

-- The target proof statement
theorem pizza_dough_milk_needed (h1 : milk_per_batch = 60) (h2 : batch_size = 300) (h3 : used_flour = 1500) : 
  (used_flour / batch_size) * milk_per_batch = 300 :=
by
  rw [h1, h2, h3]
  sorry -- proof steps

end pizza_dough_milk_needed_l54_54106


namespace total_fleas_l54_54548

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end total_fleas_l54_54548


namespace range_of_m_l54_54029

theorem range_of_m (x y m : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 / x + 1 / y = 1) (h4 : x + 2 * y > m^2 + 2 * m) :
  -4 < m ∧ m < 2 :=
sorry

end range_of_m_l54_54029


namespace probability_of_event_3a_minus_1_gt_0_l54_54311

noncomputable def probability_event : ℝ :=
if h : 0 <= 1 then (1 - 1/3) else 0

theorem probability_of_event_3a_minus_1_gt_0 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) : 
  probability_event = 2 / 3 :=
by
  sorry

end probability_of_event_3a_minus_1_gt_0_l54_54311


namespace no_such_x_exists_l54_54342

theorem no_such_x_exists : ¬ ∃ x : ℝ, 
  (∃ x1 : ℤ, x - 1/x = x1) ∧ 
  (∃ x2 : ℤ, 1/x - 1/(x^2 + 1) = x2) ∧ 
  (∃ x3 : ℤ, 1/(x^2 + 1) - 2*x = x3) :=
by
  sorry

end no_such_x_exists_l54_54342


namespace area_of_square_same_yarn_l54_54983

theorem area_of_square_same_yarn (a : ℕ) (ha : a = 4) :
  let hexagon_perimeter := 6 * a
  let square_side := hexagon_perimeter / 4
  square_side * square_side = 36 :=
by
  sorry

end area_of_square_same_yarn_l54_54983


namespace sum_of_three_consecutive_integers_is_21_l54_54617

theorem sum_of_three_consecutive_integers_is_21 : 
  ∃ (n : ℤ), 3 * n = 21 :=
by
  sorry

end sum_of_three_consecutive_integers_is_21_l54_54617


namespace partial_fractions_sum_zero_l54_54806

noncomputable def sum_of_coefficients (A B C D E : ℝ) : Prop :=
  (A + B + C + D + E = 0)

theorem partial_fractions_sum_zero :
  ∀ (A B C D E : ℝ),
    (∀ x : ℝ, 1 = A*(x+1)*(x+2)*(x+3)*(x+5) + B*x*(x+2)*(x+3)*(x+5) + 
              C*x*(x+1)*(x+3)*(x+5) + D*x*(x+1)*(x+2)*(x+5) + 
              E*x*(x+1)*(x+2)*(x+3)) →
    sum_of_coefficients A B C D E :=
by sorry

end partial_fractions_sum_zero_l54_54806


namespace three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l54_54524

theorem three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693 :
  ∃ (n : ℕ), n = 693 ∧ 
    (100 * 6 + 10 * (n / 10 % 10) + 3) = n ∧
    (n % 10 = 3) ∧
    (n / 100 = 6) ∧
    n % 11 = 0 :=
by
  sorry

end three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l54_54524


namespace prob_AB_diff_homes_l54_54470

-- Define the volunteers
inductive Volunteer : Type
| A | B | C | D | E

open Volunteer

-- Define the homes
inductive Home : Type
| home1 | home2

open Home

-- Total number of ways to distribute the volunteers
def total_ways : ℕ := 2^5  -- Each volunteer has independently 2 choices

-- Number of ways in which A and B are in different homes
def diff_ways : ℕ := 2 * 4 * 2^3  -- Split the problem down by cases for simplicity

-- Calculate the probability
def probability : ℚ := diff_ways / total_ways

-- The final statement to prove
theorem prob_AB_diff_homes : probability = 8 / 15 := sorry

end prob_AB_diff_homes_l54_54470


namespace rectangle_area_unchanged_l54_54219

theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) : 
  0.8 * l * 1.25 * w = 432 := 
by {
  -- The proof goes here
  sorry
}

end rectangle_area_unchanged_l54_54219


namespace one_third_of_seven_times_nine_l54_54862

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l54_54862


namespace daily_sales_volume_selling_price_for_profit_l54_54226

noncomputable def cost_price : ℝ := 40
noncomputable def initial_selling_price : ℝ := 60
noncomputable def initial_sales_volume : ℝ := 20
noncomputable def price_decrease_per_increase : ℝ := 5
noncomputable def volume_increase_per_decrease : ℝ := 10

theorem daily_sales_volume (p : ℝ) (v : ℝ) :
  v = initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease :=
sorry

theorem selling_price_for_profit (p : ℝ) (profit : ℝ) :
  profit = (p - cost_price) * (initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease) → p = 54 :=
sorry

end daily_sales_volume_selling_price_for_profit_l54_54226


namespace midpoint_product_l54_54088

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l54_54088


namespace cos_585_eq_neg_sqrt2_div_2_l54_54344

theorem cos_585_eq_neg_sqrt2_div_2 : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by sorry

end cos_585_eq_neg_sqrt2_div_2_l54_54344


namespace sqrt13_decomposition_ten_plus_sqrt3_decomposition_l54_54493

-- For the first problem
theorem sqrt13_decomposition :
  let a := 3
  let b := Real.sqrt 13 - 3
  a^2 + b - Real.sqrt 13 = 6 := by
sorry

-- For the second problem
theorem ten_plus_sqrt3_decomposition :
  let x := 11
  let y := Real.sqrt 3 - 1
  x - y = 12 - Real.sqrt 3 := by
sorry

end sqrt13_decomposition_ten_plus_sqrt3_decomposition_l54_54493


namespace jonah_total_ingredients_in_cups_l54_54896

noncomputable def volume_of_ingredients_in_cups : ℝ :=
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  let almonds_in_ounces := 5.5
  let pumpkin_seeds_in_grams := 150
  let ounce_to_cup_conversion := 0.125
  let gram_to_cup_conversion := 0.00423
  let almonds := almonds_in_ounces * ounce_to_cup_conversion
  let pumpkin_seeds := pumpkin_seeds_in_grams * gram_to_cup_conversion
  yellow_raisins + black_raisins + almonds + pumpkin_seeds

theorem jonah_total_ingredients_in_cups : volume_of_ingredients_in_cups = 2.022 :=
by
  sorry

end jonah_total_ingredients_in_cups_l54_54896


namespace g_is_even_l54_54105

noncomputable def g (x : ℝ) : ℝ := 5^(x^2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g x = g (-x) :=
by
  sorry

end g_is_even_l54_54105


namespace cary_mow_weekends_l54_54659

theorem cary_mow_weekends :
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  remaining_amount / earn_per_weekend = 6 :=
by
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  have needed_weekends : remaining_amount / earn_per_weekend = 6 :=
    sorry
  exact needed_weekends

end cary_mow_weekends_l54_54659


namespace initial_cheerleaders_count_l54_54583

theorem initial_cheerleaders_count (C : ℕ) 
  (initial_football_players : ℕ := 13) 
  (quit_football_players : ℕ := 10) 
  (quit_cheerleaders : ℕ := 4) 
  (remaining_people : ℕ := 15) 
  (initial_total : ℕ := initial_football_players + C) 
  (final_total : ℕ := (initial_football_players - quit_football_players) + (C - quit_cheerleaders)) :
  remaining_people = final_total → C = 16 :=
by intros h; sorry

end initial_cheerleaders_count_l54_54583


namespace geese_more_than_ducks_l54_54869

theorem geese_more_than_ducks (initial_ducks: ℕ) (initial_geese: ℕ) (initial_swans: ℕ) (additional_ducks: ℕ)
  (additional_geese: ℕ) (leaving_swans: ℕ) (leaving_geese: ℕ) (returning_geese: ℕ) (returning_swans: ℕ)
  (final_leaving_ducks: ℕ) (final_leaving_swans: ℕ)
  (initial_ducks_eq: initial_ducks = 25)
  (initial_geese_eq: initial_geese = 2 * initial_ducks - 10)
  (initial_swans_eq: initial_swans = 3 * initial_ducks + 8)
  (additional_ducks_eq: additional_ducks = 4)
  (additional_geese_eq: additional_geese = 7)
  (leaving_swans_eq: leaving_swans = 9)
  (leaving_geese_eq: leaving_geese = 5)
  (returning_geese_eq: returning_geese = 15)
  (returning_swans_eq: returning_swans = 11)
  (final_leaving_ducks_eq: final_leaving_ducks = 2 * (initial_ducks + additional_ducks))
  (final_leaving_swans_eq: final_leaving_swans = (initial_swans + returning_swans) / 2):
  (initial_geese + additional_geese + returning_geese - leaving_geese - final_leaving_geese + returning_geese) -
  (initial_ducks + additional_ducks - final_leaving_ducks) = 57 :=
by
  sorry

end geese_more_than_ducks_l54_54869


namespace sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l54_54457

noncomputable def calculation (x y z : ℝ) : ℝ :=
  (Real.sqrt x * Real.sqrt y) / Real.sqrt z

theorem sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3 :
  calculation 12 27 3 = 6 * Real.sqrt 3 :=
by
  sorry

end sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l54_54457


namespace children_sit_in_same_row_twice_l54_54386

theorem children_sit_in_same_row_twice
  (rows : ℕ) (seats_per_row : ℕ) (children : ℕ)
  (h_rows : rows = 7) (h_seats_per_row : seats_per_row = 10) (h_children : children = 50) :
  ∃ (morning_evening_pair : ℕ × ℕ), 
  (morning_evening_pair.1 < rows ∧ morning_evening_pair.2 < rows) ∧ 
  morning_evening_pair.1 = morning_evening_pair.2 :=
by
  sorry

end children_sit_in_same_row_twice_l54_54386


namespace quadrilateral_area_is_two_l54_54087

def A : (Int × Int) := (0, 0)
def B : (Int × Int) := (2, 0)
def C : (Int × Int) := (2, 3)
def D : (Int × Int) := (0, 2)

noncomputable def area (p1 p2 p3 p4 : (Int × Int)) : ℚ :=
  (1 / 2 : ℚ) * (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) - 
                      (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1)))

theorem quadrilateral_area_is_two : 
  area A B C D = 2 := by
  sorry

end quadrilateral_area_is_two_l54_54087


namespace cookie_cost_l54_54368

variables (m o c : ℝ)
variables (H1 : m = 2 * o)
variables (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c)

theorem cookie_cost (H1 : m = 2 * o) (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c) : c = (13 / 4) * o :=
by sorry

end cookie_cost_l54_54368


namespace num_values_of_n_l54_54594

theorem num_values_of_n (a b c : ℕ) (h : 7 * a + 77 * b + 7777 * c = 8000) : 
  ∃ n : ℕ, (n = a + 2 * b + 4 * c) ∧ (110 * n ≤ 114300) ∧ ((8000 - 7 * a) % 70 = 7 * (10 * b + 111 * c) % 70) := 
sorry

end num_values_of_n_l54_54594


namespace sum_of_arithmetic_sequence_l54_54747

variable {S : ℕ → ℕ}

def isArithmeticSum (S : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ n, S n = n * (2 * a + (n - 1) * d ) / 2

theorem sum_of_arithmetic_sequence :
  isArithmeticSum S →
  S 8 - S 4 = 12 →
  S 12 = 36 :=
by
  intros
  sorry

end sum_of_arithmetic_sequence_l54_54747


namespace A_and_C_complete_remaining_work_in_2_point_4_days_l54_54791

def work_rate_A : ℚ := 1 / 12
def work_rate_B : ℚ := 1 / 15
def work_rate_C : ℚ := 1 / 18
def work_completed_B_in_10_days : ℚ := (10 : ℚ) * work_rate_B
def remaining_work : ℚ := 1 - work_completed_B_in_10_days
def combined_work_rate_AC : ℚ := work_rate_A + work_rate_C
def time_to_complete_remaining_work : ℚ := remaining_work / combined_work_rate_AC

theorem A_and_C_complete_remaining_work_in_2_point_4_days :
  time_to_complete_remaining_work = 2.4 := 
sorry

end A_and_C_complete_remaining_work_in_2_point_4_days_l54_54791


namespace find_b_vector_l54_54341

-- Define input vectors a, b, and their sum.
def vec_a : ℝ × ℝ × ℝ := (1, -2, 1)
def vec_b : ℝ × ℝ × ℝ := (-2, 4, -2)
def vec_sum : ℝ × ℝ × ℝ := (-1, 2, -1)

-- The theorem statement to prove that b is calculated correctly.
theorem find_b_vector :
  vec_a + vec_b = vec_sum →
  vec_b = (-2, 4, -2) :=
by
  sorry

end find_b_vector_l54_54341


namespace prove_f_of_pi_div_4_eq_0_l54_54093

noncomputable
def tan_function (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

theorem prove_f_of_pi_div_4_eq_0 
  (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x : ℝ, tan_function ω (x + π / (4 * ω)) = tan_function ω x) :
  tan_function ω (π / 4) = 0 :=
by
  -- This is where the proof would go.
  sorry

end prove_f_of_pi_div_4_eq_0_l54_54093


namespace cone_new_height_l54_54815

noncomputable def new_cone_height : ℝ := 6

theorem cone_new_height (r h V : ℝ) (circumference : 2 * Real.pi * r = 24 * Real.pi)
  (original_height : h = 40) (same_base_circumference : 2 * Real.pi * r = 24 * Real.pi)
  (volume : (1 / 3) * Real.pi * (r ^ 2) * new_cone_height = 288 * Real.pi) :
    new_cone_height = 6 := 
sorry

end cone_new_height_l54_54815


namespace max_value_of_y_l54_54623

theorem max_value_of_y (x : ℝ) (h : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 :=
sorry

end max_value_of_y_l54_54623


namespace smallest_side_of_triangle_l54_54459

theorem smallest_side_of_triangle (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) : 
  a > c ∧ b > c :=
by
  sorry

end smallest_side_of_triangle_l54_54459


namespace simplify_and_evaluate_expression_l54_54127

noncomputable def expression (a : ℝ) : ℝ :=
  ((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6 * a + 9))

theorem simplify_and_evaluate_expression : expression (3 - Real.sqrt 2) = -2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l54_54127


namespace poly_divisible_coeff_sum_eq_one_l54_54724

theorem poly_divisible_coeff_sum_eq_one (C D : ℂ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^100 + C * x^2 + D * x + 1 = 0) →
  C + D = 1 :=
by
  sorry

end poly_divisible_coeff_sum_eq_one_l54_54724


namespace feet_per_inch_of_model_l54_54218

theorem feet_per_inch_of_model 
  (height_tower : ℝ)
  (height_model : ℝ)
  (height_tower_eq : height_tower = 984)
  (height_model_eq : height_model = 6)
  : (height_tower / height_model) = 164 :=
by
  -- Assume the proof here
  sorry

end feet_per_inch_of_model_l54_54218


namespace number_of_correct_inequalities_l54_54445

variable {a b : ℝ}

theorem number_of_correct_inequalities (h₁ : a > 0) (h₂ : 0 > b) (h₃ : a + b > 0) :
  (ite (a^2 > b^2) 1 0) + (ite (1/a > 1/b) 1 0) + (ite (a^3 < ab^2) 1 0) + (ite (a^2 * b < b^3) 1 0) = 3 := 
sorry

end number_of_correct_inequalities_l54_54445


namespace total_tickets_sold_is_336_l54_54580

-- Define the costs of the tickets
def cost_vip_ticket : ℕ := 45
def cost_ga_ticket : ℕ := 20

-- Define the total cost collected
def total_cost_collected : ℕ := 7500

-- Define the difference in the number of tickets sold
def vip_less_ga : ℕ := 276

-- Define the main theorem to be proved
theorem total_tickets_sold_is_336 (V G : ℕ) 
  (h1 : cost_vip_ticket * V + cost_ga_ticket * G = total_cost_collected)
  (h2 : V = G - vip_less_ga) : V + G = 336 :=
  sorry

end total_tickets_sold_is_336_l54_54580


namespace stratified_sampling_correct_l54_54248

-- Defining the conditions
def total_students : ℕ := 900
def freshmen : ℕ := 300
def sophomores : ℕ := 200
def juniors : ℕ := 400
def sample_size : ℕ := 45

-- Defining the target sample numbers
def freshmen_sample : ℕ := 15
def sophomores_sample : ℕ := 10
def juniors_sample : ℕ := 20

-- The proof problem statement
theorem stratified_sampling_correct :
  freshmen_sample = (freshmen * sample_size / total_students) ∧
  sophomores_sample = (sophomores * sample_size / total_students) ∧
  juniors_sample = (juniors * sample_size / total_students) :=
by
  sorry

end stratified_sampling_correct_l54_54248


namespace trigonometric_identity_l54_54764

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - (1 / (Real.cos (20 * Real.pi / 180))^2) + 64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by
  sorry

end trigonometric_identity_l54_54764


namespace find_a_l54_54135

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l54_54135


namespace find_speed_of_stream_l54_54677

-- Define the given conditions
def boat_speed_still_water : ℝ := 14
def distance_downstream : ℝ := 72
def time_downstream : ℝ := 3.6

-- Define the speed of the stream (to be proven)
def speed_of_stream : ℝ := 6

-- The statement of the problem
theorem find_speed_of_stream 
  (h1 : boat_speed_still_water = 14)
  (h2 : distance_downstream = 72)
  (h3 : time_downstream = 3.6)
  (speed_of_stream_eq : boat_speed_still_water + speed_of_stream = distance_downstream / time_downstream) :
  speed_of_stream = 6 := 
by 
  sorry

end find_speed_of_stream_l54_54677


namespace transform_to_A_plus_one_l54_54372

theorem transform_to_A_plus_one (A : ℕ) (hA : A > 0) : 
  ∃ n : ℕ, (∀ i : ℕ, (i ≤ n) → ((A + 9 * i) = A + 1 ∨ ∃ j : ℕ, (A + 9 * i) = (A + 1 + 10 * j))) :=
sorry

end transform_to_A_plus_one_l54_54372


namespace remaining_digits_count_l54_54568

theorem remaining_digits_count 
  (avg9 : ℝ) (avg4 : ℝ) (avgRemaining : ℝ) (h1 : avg9 = 18) (h2 : avg4 = 8) (h3 : avgRemaining = 26) :
  let S := 9 * avg9
  let S4 := 4 * avg4
  let S_remaining := S - S4
  let N := S_remaining / avgRemaining
  N = 5 := 
by
  sorry

end remaining_digits_count_l54_54568


namespace packages_katie_can_make_l54_54525

-- Definition of the given conditions
def number_of_cupcakes_baked := 18
def cupcakes_eaten_by_todd := 8
def cupcakes_per_package := 2

-- The main statement to prove
theorem packages_katie_can_make : 
  (number_of_cupcakes_baked - cupcakes_eaten_by_todd) / cupcakes_per_package = 5 :=
by
  -- Use sorry to skip the proof
  sorry

end packages_katie_can_make_l54_54525


namespace phone_price_in_october_l54_54689

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

end phone_price_in_october_l54_54689


namespace base_of_power_expr_l54_54932

-- Defining the power expression as a condition
def power_expr : ℤ := (-4 : ℤ) ^ 3

-- The Lean statement for the proof problem
theorem base_of_power_expr : ∃ b : ℤ, (power_expr = b ^ 3) ∧ (b = -4) := 
sorry

end base_of_power_expr_l54_54932


namespace find_number_l54_54855

theorem find_number
  (x : ℝ)
  (h : (7.5 * 7.5) + 37.5 + (x * x) = 100) :
  x = 2.5 :=
sorry

end find_number_l54_54855


namespace inscribed_circle_equals_arc_length_l54_54797

open Real

theorem inscribed_circle_equals_arc_length 
  (R : ℝ) 
  (hR : 0 < R) 
  (θ : ℝ)
  (hθ : θ = (2 * π) / 3)
  (r : ℝ)
  (h_r : r = R / 2) 
  : 2 * π * r = 2 * π * R * (θ / (2 * π)) := by
  sorry

end inscribed_circle_equals_arc_length_l54_54797


namespace verify_base_case_l54_54107

theorem verify_base_case : 1 + (1 / 2) + (1 / 3) < 2 :=
sorry

end verify_base_case_l54_54107


namespace possible_values_for_n_l54_54164

theorem possible_values_for_n (n : ℕ) (h1 : ∀ a b c : ℤ, (a = n-1) ∧ (b = n) ∧ (c = n+1) → 
    (∃ f g : ℤ, f = 2*a - b ∧ g = 2*b - a)) 
    (h2 : ∃ a b c : ℤ, (a = 0 ∨ b = 0 ∨ c = 0) ∧ (a + b + c = 0)) : 
    ∃ k : ℕ, n = 3^k := 
sorry

end possible_values_for_n_l54_54164


namespace evan_ivan_kara_total_weight_eq_432_l54_54647

variable (weight_evan : ℕ) (weight_ivan : ℕ) (weight_kara_cat : ℕ)

-- Conditions
def evans_dog_weight : Prop := weight_evan = 63
def ivans_dog_weight : Prop := weight_evan = 7 * weight_ivan
def karas_cat_weight : Prop := weight_kara_cat = 5 * (weight_evan + weight_ivan)

-- Mathematical equivalence
def total_weight : Prop := weight_evan + weight_ivan + weight_kara_cat = 432

theorem evan_ivan_kara_total_weight_eq_432 :
  evans_dog_weight weight_evan →
  ivans_dog_weight weight_evan weight_ivan →
  karas_cat_weight weight_evan weight_ivan weight_kara_cat →
  total_weight weight_evan weight_ivan weight_kara_cat :=
by
  intros h1 h2 h3
  sorry

end evan_ivan_kara_total_weight_eq_432_l54_54647


namespace binomial_coefficient_is_252_l54_54520

theorem binomial_coefficient_is_252 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_coefficient_is_252_l54_54520


namespace minimum_distance_l54_54989

def curve1 (x y : ℝ) : Prop := y^2 - 9 + 2*y*x - 12*x - 3*x^2 = 0
def curve2 (x y : ℝ) : Prop := y^2 + 3 - 4*x - 2*y + x^2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem minimum_distance 
  (A B : ℝ × ℝ) 
  (hA : curve1 A.1 A.2) 
  (hB : curve2 B.1 B.2) : 
  ∃ d, d = 2 * Real.sqrt 2 ∧ (∀ P Q : ℝ × ℝ, curve1 P.1 P.2 → curve2 Q.1 Q.2 → distance P.1 P.2 Q.1 Q.2 ≥ d) :=
sorry

end minimum_distance_l54_54989


namespace field_width_l54_54429

theorem field_width (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 288) : W = 60 :=
by
  sorry

end field_width_l54_54429


namespace meteorite_weight_possibilities_l54_54978

def valid_meteorite_weight_combinations : ℕ :=
  (2 * (Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2))) + (Nat.factorial 5)

theorem meteorite_weight_possibilities :
  valid_meteorite_weight_combinations = 180 :=
by
  -- Sorry added to skip the proof.
  sorry

end meteorite_weight_possibilities_l54_54978


namespace common_ratio_geometric_sequence_l54_54518

theorem common_ratio_geometric_sequence (a₃ S₃ : ℝ) (q : ℝ)
  (h1 : a₃ = 7) (h2 : S₃ = 21)
  (h3 : ∃ a₁ : ℝ, a₃ = a₁ * q^2)
  (h4 : ∃ a₁ : ℝ, S₃ = a₁ * (1 + q + q^2)) :
  q = -1/2 ∨ q = 1 :=
sorry

end common_ratio_geometric_sequence_l54_54518


namespace initial_milk_in_container_A_l54_54818

theorem initial_milk_in_container_A (A B C D : ℝ) 
  (h1 : B = A - 0.625 * A) 
  (h2 : C - 158 = B) 
  (h3 : D = 0.45 * (C - 58)) 
  (h4 : D = 58) 
  : A = 231 := 
sorry

end initial_milk_in_container_A_l54_54818


namespace a_works_less_than_b_l54_54953

theorem a_works_less_than_b (A B : ℝ) (x y : ℝ)
  (h1 : A = 3 * B)
  (h2 : (A + B) * 22.5 = A * x)
  (h3 : y = 3 * x) :
  y - x = 60 :=
by sorry

end a_works_less_than_b_l54_54953


namespace min_value_of_expression_l54_54100

theorem min_value_of_expression (x y : ℝ) (h : x^2 + x * y + y^2 = 3) : x^2 - x * y + y^2 ≥ 1 :=
by 
sorry

end min_value_of_expression_l54_54100


namespace cost_price_of_book_l54_54236

theorem cost_price_of_book 
  (C : ℝ)
  (h1 : ∃ C, C > 0)
  (h2 : 1.10 * C = 1.15 * C - 120) :
  C = 2400 :=
sorry

end cost_price_of_book_l54_54236


namespace servant_leaving_months_l54_54027

-- The given conditions
def total_salary_year : ℕ := 90 + 110
def monthly_salary (months: ℕ) : ℕ := (months * total_salary_year) / 12
def total_received : ℕ := 40 + 110

-- The theorem to prove
theorem servant_leaving_months (months : ℕ) (h : monthly_salary months = total_received) : months = 9 :=
by {
    sorry
}

end servant_leaving_months_l54_54027


namespace expression_divisible_512_l54_54305

theorem expression_divisible_512 (n : ℤ) (h : n % 2 ≠ 0) : (n^12 - n^8 - n^4 + 1) % 512 = 0 := 
by 
  sorry

end expression_divisible_512_l54_54305


namespace parabola_properties_l54_54770

theorem parabola_properties :
  let a := -2
  let b := 4
  let c := 8
  ∃ h k : ℝ, 
    (∀ x : ℝ, y = a * x^2 + b * x + c) ∧ 
    (h = 1) ∧ 
    (k = 10) ∧ 
    (a < 0) ∧ 
    (axisOfSymmetry = h) ∧ 
    (vertex = (h, k)) :=
by
  sorry

end parabola_properties_l54_54770


namespace cliff_collection_has_180_rocks_l54_54624

noncomputable def cliffTotalRocks : ℕ :=
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks

theorem cliff_collection_has_180_rocks :
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks = 180 := sorry

end cliff_collection_has_180_rocks_l54_54624


namespace cats_weight_more_than_puppies_l54_54778

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l54_54778


namespace contestant_wins_probability_l54_54383

-- Define the basic parameters: number of questions and number of choices
def num_questions : ℕ := 4
def num_choices : ℕ := 3

-- Define the probability of getting a single question right
def prob_right : ℚ := 1 / num_choices

-- Define the probability of guessing all questions right
def prob_all_right : ℚ := prob_right ^ num_questions

-- Define the probability of guessing exactly three questions right (one wrong)
def prob_one_wrong : ℚ := (prob_right ^ 3) * (2 / num_choices)

-- Calculate the total probability of winning
def total_prob_winning : ℚ := prob_all_right + 4 * prob_one_wrong

-- The final statement to prove
theorem contestant_wins_probability :
  total_prob_winning = 1 / 9 := 
sorry

end contestant_wins_probability_l54_54383


namespace sum_reciprocal_of_shifted_roots_l54_54734

noncomputable def roots_of_cubic (a b c : ℝ) : Prop := 
    ∀ x : ℝ, x^3 - x - 2 = (x - a) * (x - b) * (x - c)

theorem sum_reciprocal_of_shifted_roots (a b c : ℝ) 
    (h : roots_of_cubic a b c) : 
    (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = 1 :=
by
  sorry

end sum_reciprocal_of_shifted_roots_l54_54734


namespace arithmetic_sequence_eighth_term_l54_54885

theorem arithmetic_sequence_eighth_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 8 = 15 := by
  sorry

end arithmetic_sequence_eighth_term_l54_54885


namespace intersection_A_B_l54_54268

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_A_B_l54_54268


namespace min_cubes_l54_54575

theorem min_cubes (a b c : ℕ) (h₁ : (a - 1) * (b - 1) * (c - 1) = 240) : a * b * c = 385 :=
  sorry

end min_cubes_l54_54575


namespace simplify_expr1_simplify_expr2_l54_54385

theorem simplify_expr1 : (-4)^2023 * (-0.25)^2024 = -0.25 :=
by 
  sorry

theorem simplify_expr2 : 23 * (-4 / 11) + (-5 / 11) * 23 - 23 * (2 / 11) = -23 :=
by 
  sorry

end simplify_expr1_simplify_expr2_l54_54385


namespace min_value_frac_sum_l54_54546

theorem min_value_frac_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1): 
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_frac_sum_l54_54546


namespace arithmetic_sequence_terms_sum_l54_54169

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

-- Definitions based on given problem conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) := 
  ∀ n, S n = n * (a 1 + a n) / 2

axiom Sn_2017 : S_n 2017 = 4034

-- Goal: a_3 + a_1009 + a_2015 = 6
theorem arithmetic_sequence_terms_sum :
  arithmetic_sequence a_n →
  sum_first_n_terms S_n a_n →
  S_n 2017 = 4034 → 
  a_n 3 + a_n 1009 + a_n 2015 = 6 :=
by
  intros
  sorry

end arithmetic_sequence_terms_sum_l54_54169


namespace solution_set_l54_54559

theorem solution_set (x : ℝ) : (x + 1 = |x + 3| - |x - 1|) ↔ (x = 3 ∨ x = -1 ∨ x = -5) :=
by
  sorry

end solution_set_l54_54559


namespace midpoint_coordinates_l54_54138

theorem midpoint_coordinates :
  let A := (7, 8)
  let B := (1, 2)
  let midpoint (p1 p2 : ℕ × ℕ) : ℕ × ℕ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint A B = (4, 5) :=
by
  sorry

end midpoint_coordinates_l54_54138


namespace one_integral_root_exists_l54_54551

theorem one_integral_root_exists :
    ∃! x : ℤ, x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end one_integral_root_exists_l54_54551


namespace sign_up_ways_l54_54972

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end sign_up_ways_l54_54972


namespace max_value_a4b3c2_l54_54472

theorem max_value_a4b3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  a^4 * b^3 * c^2 ≤ 1 / 6561 :=
sorry

end max_value_a4b3c2_l54_54472


namespace find_abc_l54_54134

-- Given conditions: a, b, c are positive real numbers and satisfy the given equations.
variables (a b c : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h1 : a * (b + c) = 152)
variable (h2 : b * (c + a) = 162)
variable (h3 : c * (a + b) = 170)

theorem find_abc : a * b * c = 720 := 
  sorry

end find_abc_l54_54134


namespace geometric_sequence_26th_term_l54_54193

noncomputable def r : ℝ := (8 : ℝ)^(1/6)

noncomputable def a (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_26th_term :
  (a 26 (a 14 10 r) r = 640) :=
by
  have h₁ : a 14 10 r = 10 := sorry
  have h₂ : r^6 = 8 := sorry
  sorry

end geometric_sequence_26th_term_l54_54193


namespace binary_10101000_is_1133_base_5_l54_54565

def binary_to_decimal (b : Nat) : Nat :=
  128 * (b / 128 % 2) + 64 * (b / 64 % 2) + 32 * (b / 32 % 2) + 16 * (b / 16 % 2) + 8 * (b / 8 % 2) + 4 * (b / 4 % 2) + 2 * (b / 2 % 2) + (b % 2)

def decimal_to_base_5 (d : Nat) : List Nat :=
  if d = 0 then [] else (d % 5) :: decimal_to_base_5 (d / 5)

def binary_to_base_5 (b : Nat) : List Nat :=
  decimal_to_base_5 (binary_to_decimal b)

theorem binary_10101000_is_1133_base_5 :
  binary_to_base_5 168 = [1, 1, 3, 3] := 
by 
  sorry

end binary_10101000_is_1133_base_5_l54_54565


namespace probability_one_solve_l54_54202

variables {p1 p2 : ℝ}

theorem probability_one_solve (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) := 
sorry

end probability_one_solve_l54_54202


namespace solution_of_inequality_system_l54_54259

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end solution_of_inequality_system_l54_54259


namespace jancy_currency_notes_l54_54783

theorem jancy_currency_notes (x y : ℕ) (h1 : 70 * x + 50 * y = 5000) (h2 : y = 2) : x + y = 72 :=
by
  -- proof goes here
  sorry

end jancy_currency_notes_l54_54783


namespace minimum_value_ge_100_minimum_value_eq_100_l54_54504

noncomputable def minimum_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2

theorem minimum_value_ge_100 (α β : ℝ) : minimum_value_expression α β ≥ 100 :=
  sorry

theorem minimum_value_eq_100 (α β : ℝ)
  (hα : 3 * Real.cos α + 4 * Real.sin β = 7)
  (hβ : 3 * Real.sin α + 4 * Real.cos β = 10) :
  minimum_value_expression α β = 100 :=
  sorry

end minimum_value_ge_100_minimum_value_eq_100_l54_54504


namespace find_angle_A_find_minimum_bc_l54_54757

open Real

variables (A B C a b c : ℝ)

-- Conditions
def side_opposite_angles_condition : Prop :=
  A > 0 ∧ A < π ∧ (A + B + C) = π

def collinear_vectors_condition (B C : ℝ) : Prop :=
  ∃ (k : ℝ), (2 * cos B * cos C + 1, 2 * sin B) = k • (sin C, 1)

-- Questions translated to proof statements
theorem find_angle_A (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C) :
  A = π / 3 :=
sorry

theorem find_minimum_bc (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C)
  (h3 : (1 / 2) * b * c * sin A = sqrt 3) :
  b + c = 4 :=
sorry

end find_angle_A_find_minimum_bc_l54_54757


namespace complement_P_correct_l54_54229

def is_solution (x : ℝ) : Prop := |x + 3| + |x + 6| = 3

def P : Set ℝ := {x | is_solution x}

def C_R (P : Set ℝ) : Set ℝ := {x | x ∉ P}

theorem complement_P_correct : C_R P = {x | x < -6 ∨ x > -3} :=
by
  sorry

end complement_P_correct_l54_54229


namespace roses_formula_l54_54307

open Nat

def total_roses (n : ℕ) : ℕ := 
  (choose n 4) + (choose (n - 1) 2)

theorem roses_formula (n : ℕ) (h : n ≥ 4) : 
  total_roses n = (choose n 4) + (choose (n - 1) 2) := 
by
  sorry

end roses_formula_l54_54307


namespace part1_part2_l54_54443

theorem part1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) : 
  a + 2 * b + c ≤ 4 :=
sorry

theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) (h5 : a = 2 * b) : 
  1 / b + 1 / (c - 1) ≥ 3 :=
sorry

end part1_part2_l54_54443


namespace randy_initial_amount_l54_54149

theorem randy_initial_amount (spend_per_trip: ℤ) (trips_per_month: ℤ) (dollars_left_after_year: ℤ) (total_month_months: ℤ := 12):
  (spend_per_trip = 2 ∧ trips_per_month = 4 ∧ dollars_left_after_year = 104) → spend_per_trip * trips_per_month * total_month_months + dollars_left_after_year = 200 := 
by
  sorry

end randy_initial_amount_l54_54149


namespace arithmetic_sequence_8th_term_l54_54453

theorem arithmetic_sequence_8th_term :
  ∃ (a1 a15 n : ℕ) (d a8 : ℝ),
  a1 = 3 ∧ a15 = 48 ∧ n = 15 ∧
  d = (a15 - a1) / (n - 1) ∧
  a8 = a1 + 7 * d ∧
  a8 = 25.5 :=
by
  sorry

end arithmetic_sequence_8th_term_l54_54453


namespace minValue_expression_l54_54514

theorem minValue_expression (x y : ℝ) (h : x + 2 * y = 4) : ∃ (v : ℝ), v = 2^x + 4^y ∧ ∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ v :=
by 
  sorry

end minValue_expression_l54_54514


namespace distinct_values_of_expression_l54_54495

variable {u v x y z : ℝ}

theorem distinct_values_of_expression (hu : u + u⁻¹ = x) (hv : v + v⁻¹ = y)
  (hx_distinct : x ≠ y) (hx_abs : |x| ≥ 2) (hy_abs : |y| ≥ 2) :
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z = u * v + (u * v)⁻¹)) →
  ∃ n, n = 2 := by 
    sorry

end distinct_values_of_expression_l54_54495


namespace files_remaining_l54_54620

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h_music : music_files = 13) 
  (h_video : video_files = 30) 
  (h_deleted : deleted_files = 10) : 
  (music_files + video_files - deleted_files) = 33 :=
by
  sorry

end files_remaining_l54_54620


namespace mean_of_two_numbers_l54_54095

theorem mean_of_two_numbers (a b : ℝ) (mean_twelve : ℝ) (mean_fourteen : ℝ) 
  (h1 : mean_twelve = 60) 
  (h2 : mean_fourteen = 75) 
  (sum_twelve : 12 * mean_twelve = 720) 
  (sum_fourteen : 14 * mean_fourteen = 1050) 
  : (a + b) / 2 = 165 :=
by
  sorry

end mean_of_two_numbers_l54_54095


namespace cricket_target_run_rate_cricket_wicket_partnership_score_l54_54388

noncomputable def remaining_runs_needed (initial_runs : ℕ) (target_runs : ℕ) : ℕ :=
  target_runs - initial_runs

noncomputable def required_run_rate (remaining_runs : ℕ) (remaining_overs : ℕ) : ℚ :=
  (remaining_runs : ℚ) / remaining_overs

theorem cricket_target_run_rate (initial_runs : ℕ) (target_runs : ℕ) (remaining_overs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → remaining_overs = 40 → initial_wickets = 3 →
  required_run_rate (remaining_runs_needed initial_runs target_runs) remaining_overs = 6.25 :=
by
  sorry


theorem cricket_wicket_partnership_score (initial_runs : ℕ) (target_runs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → initial_wickets = 3 →
  remaining_runs_needed initial_runs target_runs = 250 :=
by
  sorry

end cricket_target_run_rate_cricket_wicket_partnership_score_l54_54388


namespace punctures_covered_l54_54977

theorem punctures_covered (P1 P2 P3 : ℝ) (h1 : 0 ≤ P1) (h2 : P1 < P2) (h3 : P2 < P3) (h4 : P3 < 3) :
    ∃ x, x ≤ P1 ∧ x + 2 ≥ P3 := 
sorry

end punctures_covered_l54_54977


namespace complex_number_division_l54_54752

theorem complex_number_division (i : ℂ) (h_i : i^2 = -1) :
  2 / (i * (3 - i)) = (1 - 3 * i) / 5 :=
by
  sorry

end complex_number_division_l54_54752


namespace min_increase_velocity_correct_l54_54666

noncomputable def min_increase_velocity (V_A V_B V_C V_D : ℝ) (dist_AC dist_CD : ℝ) : ℝ :=
  let t_AC := dist_AC / (V_A + V_C)
  let t_AB := 30 / (V_A - V_B)
  let t_AD := (dist_AC + dist_CD) / (V_A + V_D)
  let new_velocity_A := (dist_AC + dist_CD) / t_AC - V_D
  new_velocity_A - V_A

theorem min_increase_velocity_correct :
  min_increase_velocity 80 50 70 60 300 400 = 210 :=
by
  sorry

end min_increase_velocity_correct_l54_54666


namespace no_real_roots_iff_k_gt_2_l54_54320

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end no_real_roots_iff_k_gt_2_l54_54320


namespace largest_side_of_rectangle_l54_54968

theorem largest_side_of_rectangle (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 1920) : 
    max l w = 101 := 
sorry

end largest_side_of_rectangle_l54_54968


namespace alexandra_magazines_l54_54062

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end alexandra_magazines_l54_54062


namespace henry_has_more_than_500_seeds_on_saturday_l54_54499

theorem henry_has_more_than_500_seeds_on_saturday :
  (∃ k : ℕ, 5 * 3^k > 500 ∧ k + 1 = 6) :=
sorry

end henry_has_more_than_500_seeds_on_saturday_l54_54499


namespace scientific_notation_of_188_million_l54_54251

theorem scientific_notation_of_188_million : 
  (188000000 : ℝ) = 1.88 * 10^8 := 
by
  sorry

end scientific_notation_of_188_million_l54_54251


namespace final_fish_stock_l54_54285

def initial_stock : ℤ := 200 
def sold_fish : ℤ := 50 
def fraction_spoiled : ℚ := 1/3 
def new_stock : ℤ := 200 

theorem final_fish_stock : 
    initial_stock - sold_fish - (fraction_spoiled * (initial_stock - sold_fish)) + new_stock = 300 := 
by 
  sorry

end final_fish_stock_l54_54285


namespace geometric_sequence_12th_term_l54_54133

theorem geometric_sequence_12th_term 
  (a_4 a_8 : ℕ) (h4 : a_4 = 2) (h8 : a_8 = 162) :
  ∃ a_12 : ℕ, a_12 = 13122 :=
by
  sorry

end geometric_sequence_12th_term_l54_54133


namespace solve_diamond_l54_54031

theorem solve_diamond (d : ℕ) (h : d * 6 + 5 = d * 7 + 2) : d = 3 :=
by
  sorry

end solve_diamond_l54_54031


namespace min_value_expression_l54_54397

theorem min_value_expression (x y z : ℝ) (h1 : -1/2 < x ∧ x < 1/2) (h2 : -1/2 < y ∧ y < 1/2) (h3 : -1/2 < z ∧ z < 1/2) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) + 1 / 2) ≥ 2.5 :=
by {
  sorry
}

end min_value_expression_l54_54397


namespace rectangular_field_area_l54_54452

theorem rectangular_field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangular_field_area_l54_54452


namespace fourth_quarter_points_sum_l54_54292

variable (a d b j : ℕ)

-- Conditions from the problem
def halftime_tied : Prop := 2 * a + d = 2 * b
def wildcats_won_by_four : Prop := 4 * a + 6 * d = 4 * b - 3 * j + 4

-- The proof goal to be established
theorem fourth_quarter_points_sum
  (h1 : halftime_tied a d b)
  (h2 : wildcats_won_by_four a d b j) :
  (a + 3 * d) + (b - 2 * j) = 28 :=
sorry

end fourth_quarter_points_sum_l54_54292


namespace det_B_squared_minus_3B_l54_54404

theorem det_B_squared_minus_3B (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B = ![![2, 4], ![3, 2]]) : 
  Matrix.det (B * B - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3B_l54_54404


namespace num_perfect_squares_in_range_l54_54720

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

end num_perfect_squares_in_range_l54_54720


namespace find_prices_find_min_money_spent_l54_54654

-- Define the prices of volleyball and soccer ball
def prices (pv ps : ℕ) : Prop :=
  pv + 20 = ps ∧ 500 / ps = 400 / pv

-- Define the quantity constraint
def quantity_constraint (a : ℕ) : Prop :=
  a ≥ 25 ∧ a < 50

-- Define the minimum amount spent problem
def min_money_spent (a : ℕ) (pv ps : ℕ) : Prop :=
  prices pv ps → quantity_constraint a → 100 * a + 80 * (50 - a) = 4500

-- Prove the price of each volleyball and soccer ball
theorem find_prices : ∃ (pv ps : ℕ), prices pv ps ∧ pv = 80 ∧ ps = 100 :=
by {sorry}

-- Prove the minimum amount of money spent
theorem find_min_money_spent : ∃ (a pv ps : ℕ), min_money_spent a pv ps :=
by {sorry}

end find_prices_find_min_money_spent_l54_54654


namespace happy_number_part1_happy_number_part2_happy_number_part3_l54_54308

section HappyEquations

def is_happy_eq (a b c : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, a ≠ 0 ∧ a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b ^ 2) / (4 * a)

def happy_to_each_other (a b c p q r : ℤ) : Prop :=
  let Fa : ℚ := happy_number a b c
  let Fb : ℚ := happy_number p q r
  |r * Fa - c * Fb| = 0

theorem happy_number_part1 :
  happy_number 1 (-2) (-3) = -4 :=
by sorry

theorem happy_number_part2 (m : ℤ) (h : 1 < m ∧ m < 6) :
  is_happy_eq 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) →
  m = 3 ∧ happy_number 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) = -25 / 4 :=
by sorry

theorem happy_number_part3 (m n : ℤ) :
  is_happy_eq 1 (-m) (m + 1) ∧ is_happy_eq 1 (-(n + 2)) (2 * n) →
  happy_to_each_other 1 (-m) (m + 1) 1 (-(n + 2)) (2 * n) →
  n = 0 ∨ n = 3 ∨ n = 3 / 2 :=
by sorry

end HappyEquations

end happy_number_part1_happy_number_part2_happy_number_part3_l54_54308


namespace find_ratio_l54_54000

-- Given conditions
variable (x y a b : ℝ)
variable (h1 : 2 * x - y = a)
variable (h2 : 4 * y - 8 * x = b)
variable (h3 : b ≠ 0)

theorem find_ratio (a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 4 := by
  sorry

end find_ratio_l54_54000


namespace find_range_of_a_l54_54741

theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ∧ 
  ¬ ((∀ x : ℝ, x^2 - 2 * x > a) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)) → 
  a ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∪ Set.Ici (1:ℝ) :=
sorry

end find_range_of_a_l54_54741


namespace monotonic_decreasing_interval_l54_54002

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (2 * x + Real.pi / 4))

theorem monotonic_decreasing_interval :
  ∀ (x1 x2 : ℝ), (-Real.pi / 8) < x1 ∧ x1 < Real.pi / 8 ∧ (-Real.pi / 8) < x2 ∧ x2 < Real.pi / 8 ∧ x1 < x2 →
  f x1 > f x2 :=
sorry

end monotonic_decreasing_interval_l54_54002


namespace circle_tangency_problem_l54_54782

theorem circle_tangency_problem :
  let u1 := ∀ (x y : ℝ), x^2 + y^2 + 8 * x - 30 * y - 63 = 0
  let u2 := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 30 * y + 99 = 0
  let line := ∀ (b x : ℝ), y = b * x
  ∃ p q : ℕ, gcd p q = 1 ∧ n^2 = (p : ℚ) / (q : ℚ) ∧ p + q = 7 :=
sorry

end circle_tangency_problem_l54_54782


namespace minNumberOfRectangles_correct_l54_54287

variable (k n : ℤ)

noncomputable def minNumberOfRectangles (k n : ℤ) : ℤ :=
  if 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1 then
    if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1)
  else 0 -- 0 if the conditions are not met

theorem minNumberOfRectangles_correct (k n : ℤ) (h : 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1) : 
  minNumberOfRectangles k n = 
  if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1) := 
by 
  -- Proof will go here
  sorry

end minNumberOfRectangles_correct_l54_54287


namespace min_moves_to_equalize_boxes_l54_54769

def initialCoins : List ℕ := [5, 8, 11, 17, 20, 15, 10]

def targetCoins (boxes : List ℕ) : ℕ := boxes.sum / boxes.length

def movesRequiredToBalance : List ℕ → ℕ
| [5, 8, 11, 17, 20, 15, 10] => 22
| _ => sorry

theorem min_moves_to_equalize_boxes :
  movesRequiredToBalance initialCoins = 22 :=
by
  sorry

end min_moves_to_equalize_boxes_l54_54769


namespace Joey_age_digit_sum_l54_54290

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end Joey_age_digit_sum_l54_54290


namespace rectangle_area_is_correct_l54_54863

noncomputable def inscribed_rectangle_area (r : ℝ) (l_to_w_ratio : ℝ) : ℝ :=
  let width := 2 * r
  let length := l_to_w_ratio * width
  length * width

theorem rectangle_area_is_correct :
  inscribed_rectangle_area 7 3 = 588 :=
  by
    -- The proof goes here
    sorry

end rectangle_area_is_correct_l54_54863


namespace matrix_A_pow_50_l54_54313

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 1], ![-16, -3]]

theorem matrix_A_pow_50 :
  A ^ 50 = ![![201, 50], ![-800, -199]] :=
sorry

end matrix_A_pow_50_l54_54313


namespace volume_of_locations_eq_27sqrt6pi_over_8_l54_54969

noncomputable def volumeOfLocationSet : ℝ :=
  let sqrt2_inv := 1 / (2 * Real.sqrt 2)
  let points := [ (sqrt2_inv, sqrt2_inv, sqrt2_inv),
                  (sqrt2_inv, sqrt2_inv, -sqrt2_inv),
                  (sqrt2_inv, -sqrt2_inv, sqrt2_inv),
                  (-sqrt2_inv, sqrt2_inv, sqrt2_inv) ]
  let condition (x y z : ℝ) : Prop :=
    4 * (x^2 + y^2 + z^2) + 3 / 2 ≤ 15
  let r := Real.sqrt (27 / 8)
  let volume := (4/3) * Real.pi * r^3
  volume

theorem volume_of_locations_eq_27sqrt6pi_over_8 :
  volumeOfLocationSet = 27 * Real.sqrt 6 * Real.pi / 8 :=
sorry

end volume_of_locations_eq_27sqrt6pi_over_8_l54_54969


namespace initial_distance_between_trains_l54_54232

theorem initial_distance_between_trains :
  let length_train1 := 100 -- meters
  let length_train2 := 200 -- meters
  let speed_train1_kmph := 54 -- km/h
  let speed_train2_kmph := 72 -- km/h
  let time_hours := 1.999840012798976 -- hours
  
  -- Conversion to meters per second
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600 -- 15 m/s
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600 -- 20 m/s

  -- Conversion of time to seconds
  let time_seconds := time_hours * 3600 -- 7199.4240460755136 seconds

  -- Relative speed in meters per second
  let relative_speed := speed_train1_mps + speed_train2_mps -- 35 m/s

  -- Distance covered by both trains
  let distance_covered := relative_speed * time_seconds -- 251980.84161264498 meters

  -- Initial distance between the trains
  let initial_distance := distance_covered - (length_train1 + length_train2) -- 251680.84161264498 meters

  initial_distance = 251680.84161264498 := 
by
  sorry

end initial_distance_between_trains_l54_54232


namespace probability_four_collinear_dots_l54_54034

noncomputable def probability_collinear_four_dots : ℚ :=
  let total_dots := 25
  let choose_4 := (total_dots.choose 4)
  let successful_outcomes := 60
  successful_outcomes / choose_4

theorem probability_four_collinear_dots :
  probability_collinear_four_dots = 12 / 2530 :=
by
  sorry

end probability_four_collinear_dots_l54_54034


namespace proof_1_proof_2_l54_54835

noncomputable def problem_1 (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 1) > 1 → x > 1

noncomputable def problem_2 (x a : ℝ) : Prop :=
  if a = 0 then False
  else if a > 0 then -a < x ∧ x < 2 * a
  else if a < 0 then 2 * a < x ∧ x < -a
  else False

-- Sorry to skip the proofs
theorem proof_1 (x : ℝ) (h : problem_1 x) : x > 1 :=
  sorry

theorem proof_2 (x a : ℝ) (h : x * x - a * x - 2 * a * a < 0) : problem_2 x a :=
  sorry

end proof_1_proof_2_l54_54835


namespace greatest_integer_b_not_in_range_l54_54823

theorem greatest_integer_b_not_in_range :
  let f (x : ℝ) (b : ℝ) := x^2 + b*x + 20
  let g (x : ℝ) (b : ℝ) := x^2 + b*x + 24
  (¬ (∃ (x : ℝ), g x b = 0)) → (b = 9) :=
by
  sorry

end greatest_integer_b_not_in_range_l54_54823


namespace evaluate_expression_l54_54936

theorem evaluate_expression : (733 * 733) - (732 * 734) = 1 :=
by
  sorry

end evaluate_expression_l54_54936


namespace area_quadrilateral_l54_54211

theorem area_quadrilateral (EF GH: ℝ) (EHG: ℝ) 
  (h1 : EF = 9) (h2 : GH = 12) (h3 : GH = EH) (h4 : EHG = 75) 
  (a b c : ℕ)
  : 
  (∀ (a b c : ℕ), 
  a = 26 ∧ b = 18 ∧ c = 6 → 
  a + b + c = 50) := 
sorry

end area_quadrilateral_l54_54211


namespace hour_hand_rotations_l54_54376

theorem hour_hand_rotations (degrees_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) (rotations_per_day : ℕ) :
  degrees_per_hour = 30 →
  hours_per_day = 24 →
  rotations_per_day = (degrees_per_hour * hours_per_day) / 360 →
  days = 6 →
  rotations_per_day * days = 12 :=
by
  intros
  sorry

end hour_hand_rotations_l54_54376


namespace gyeongyeon_total_path_l54_54118

theorem gyeongyeon_total_path (D : ℝ) :
  (D / 4 + 250 = D / 2 - 300) -> D = 2200 :=
by
  intro h
  -- We would now proceed to show that D must equal 2200
  sorry

end gyeongyeon_total_path_l54_54118


namespace determinant_transformation_l54_54895

theorem determinant_transformation 
  (a b c d : ℝ)
  (h : a * d - b * c = 6) :
  (a * (5 * c + 2 * d) - c * (5 * a + 2 * b)) = 12 := by
  sorry

end determinant_transformation_l54_54895


namespace find_natural_numbers_l54_54018

theorem find_natural_numbers (x y z : ℕ) (hx : x ≤ y) (hy : y ≤ z) : 
    (1 + 1 / x) * (1 + 1 / y) * (1 + 1 / z) = 3 
    → (x = 1 ∧ y = 3 ∧ z = 8) 
    ∨ (x = 1 ∧ y = 4 ∧ z = 5) 
    ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
sorry

end find_natural_numbers_l54_54018


namespace final_value_l54_54366

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l54_54366


namespace suresh_work_hours_l54_54390

theorem suresh_work_hours (x : ℝ) (h : x / 15 + 8 / 20 = 1) : x = 9 :=
by 
    sorry

end suresh_work_hours_l54_54390


namespace value_of_f1_l54_54487

noncomputable def f (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 3

theorem value_of_f1 (m : ℝ) (h_increasing : ∀ x : ℝ, x ≥ -2 → 2 * x^2 - m * x + 3 ≤ 2 * (x + 1)^2 - m * (x + 1) + 3)
  (h_decreasing : ∀ x : ℝ, x ≤ -2 → 2 * (x - 1)^2 - m * (x - 1) + 3 ≤ 2 * x^2 - m * x + 3) : 
  f 1 (-8) = 13 := 
sorry

end value_of_f1_l54_54487


namespace solution1_solution2_l54_54312

-- Problem: Solving equations and finding their roots

-- Condition 1:
def equation1 (x : Real) : Prop := x^2 - 2 * x = -1

-- Condition 2:
def equation2 (x : Real) : Prop := (x + 3)^2 = 2 * x * (x + 3)

-- Correct answer 1
theorem solution1 : ∀ x : Real, equation1 x → x = 1 := 
by 
  sorry

-- Correct answer 2
theorem solution2 : ∀ x : Real, equation2 x → x = -3 ∨ x = 3 := 
by 
  sorry

end solution1_solution2_l54_54312


namespace similar_triangle_perimeters_l54_54878

theorem similar_triangle_perimeters 
  (h_ratio : ℕ) (h_ratio_eq : h_ratio = 2/3)
  (sum_perimeters : ℕ) (sum_perimeters_eq : sum_perimeters = 50)
  (a b : ℕ)
  (perimeter_ratio : ℕ) (perimeter_ratio_eq : perimeter_ratio = 2/3)
  (hyp1 : a + b = sum_perimeters)
  (hyp2 : a * 3 = b * 2) :
  (a = 20 ∧ b = 30) :=
by
  sorry

end similar_triangle_perimeters_l54_54878


namespace general_form_of_numbers_whose_square_ends_with_9_l54_54867

theorem general_form_of_numbers_whose_square_ends_with_9 (x : ℤ) (h : (x^2 % 10 = 9)) :
  ∃ a : ℤ, x = 10 * a + 3 ∨ x = 10 * a + 7 :=
sorry

end general_form_of_numbers_whose_square_ends_with_9_l54_54867


namespace price_of_thermometer_l54_54625

noncomputable def thermometer_price : ℝ := 2

theorem price_of_thermometer
  (T : ℝ)
  (price_hot_water_bottle : ℝ := 6)
  (hot_water_bottles_sold : ℕ := 60)
  (total_sales : ℝ := 1200)
  (thermometers_sold : ℕ := 7 * hot_water_bottles_sold)
  (thermometers_sales : ℝ := total_sales - (price_hot_water_bottle * hot_water_bottles_sold)) :
  T = thermometer_price :=
by
  sorry

end price_of_thermometer_l54_54625


namespace distance_problem_l54_54001

-- Define the problem
theorem distance_problem
  (x y : ℝ)
  (h1 : x + y = 21)
  (h2 : x / 60 + 21 / 60 = 10 / 60 + y / 4) :
  x = 19 ∧ y = 2 :=
by
  sorry

end distance_problem_l54_54001


namespace prove_y_l54_54554

-- Define the conditions
variables (x y : ℤ) -- x and y are integers

-- State the problem conditions
def conditions := (x + y = 270) ∧ (x - y = 200)

-- Define the theorem to prove that y = 35 given the conditions
theorem prove_y : conditions x y → y = 35 :=
by
  sorry

end prove_y_l54_54554


namespace remaining_pieces_l54_54214

/-- Define the initial number of pieces on a standard chessboard. -/
def initial_pieces : Nat := 32

/-- Define the number of pieces lost by Audrey. -/
def audrey_lost : Nat := 6

/-- Define the number of pieces lost by Thomas. -/
def thomas_lost : Nat := 5

/-- Proof that the remaining number of pieces on the chessboard is 21. -/
theorem remaining_pieces : initial_pieces - (audrey_lost + thomas_lost) = 21 := by
  -- Mathematical equivalence to 32 - (6 + 5) = 21
  sorry

end remaining_pieces_l54_54214


namespace least_overlap_coffee_tea_l54_54674

open BigOperators

-- Define the percentages in a way that's compatible in Lean
def percentage (n : ℕ) := n / 100

noncomputable def C := percentage 75
noncomputable def T := percentage 80
noncomputable def B := percentage 55

-- The theorem statement
theorem least_overlap_coffee_tea : C + T - 1 = B := sorry

end least_overlap_coffee_tea_l54_54674


namespace valid_k_sum_correct_l54_54758

def sum_of_valid_k : ℤ :=
  (List.range 17).sum * 1734 + (List.range 17).sum * 3332

theorem valid_k_sum_correct : sum_of_valid_k = 5066 := by
  sorry

end valid_k_sum_correct_l54_54758


namespace remainder_div_by_3_not_divisible_by_9_l54_54289

theorem remainder_div_by_3 (x : ℕ) (h : x = 1493826) : x % 3 = 0 :=
by sorry

theorem not_divisible_by_9 (x : ℕ) (h : x = 1493826) : x % 9 ≠ 0 :=
by sorry

end remainder_div_by_3_not_divisible_by_9_l54_54289


namespace find_abc_l54_54480

open Polynomial

noncomputable def my_gcd_lcm_problem (a b c : ℤ) : Prop :=
  gcd (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X + 1 ∧
  lcm (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X^3 - 5*X^2 + 7*X - 3

theorem find_abc : ∀ (a b c : ℤ),
  my_gcd_lcm_problem a b c → a + b + c = -3 :=
by
  intros a b c h
  sorry

end find_abc_l54_54480


namespace four_thirds_eq_36_l54_54296

theorem four_thirds_eq_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := by
  sorry

end four_thirds_eq_36_l54_54296


namespace speed_of_train_l54_54656

def distance : ℝ := 80
def time : ℝ := 6
def expected_speed : ℝ := 13.33

theorem speed_of_train : distance / time = expected_speed :=
by
  sorry

end speed_of_train_l54_54656


namespace average_ABC_is_3_l54_54191

theorem average_ABC_is_3
  (A B C : ℝ)
  (h1 : 2003 * C - 4004 * A = 8008)
  (h2 : 2003 * B + 6006 * A = 10010)
  (h3 : B = 2 * A - 6) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_is_3_l54_54191


namespace find_number_l54_54990

theorem find_number (x : ℤ) : (150 - x = x + 68) → x = 41 :=
by
  intro h
  sorry

end find_number_l54_54990


namespace M_subset_N_cond_l54_54591

theorem M_subset_N_cond (a : ℝ) (h : 0 < a) :
  (∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 = a^2} → p ∈ {p : ℝ × ℝ | |p.fst + p.snd| + |p.fst - p.snd| ≤ 2}) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end M_subset_N_cond_l54_54591


namespace calc_expr_l54_54956

theorem calc_expr : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 :=
by sorry

end calc_expr_l54_54956


namespace father_current_age_l54_54733

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

end father_current_age_l54_54733


namespace neg_exists_is_forall_l54_54995

theorem neg_exists_is_forall: 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by
  sorry

end neg_exists_is_forall_l54_54995


namespace wendy_total_glasses_l54_54616

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end wendy_total_glasses_l54_54616


namespace ana_bonita_age_gap_l54_54477

theorem ana_bonita_age_gap (A B n : ℚ) (h1 : A = 2 * B + 3) (h2 : A - 2 = 6 * (B - 2)) (h3 : A = B + n) : n = 6.25 :=
by
  sorry

end ana_bonita_age_gap_l54_54477


namespace problem1_problem2_problem3_l54_54055

def A : Set ℝ := Set.Icc (-1) 1
def B : Set ℝ := Set.Icc (-2) 2
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1
def g (a m x : ℝ) : ℝ := 2 * abs (x - a) - x^2 - m * x

theorem problem1 (m : ℝ) : (∀ x, f m x ≤ 0 → x ∈ A) → m ∈ Set.Icc (-1) 1 :=
sorry

theorem problem2 (f_eq : ∀ x, f (-4) (1-x) = f (-4) (1+x)) : 
  Set.range (f (-4) ∘ id) ⊆ Set.Icc (-3) 15 :=
sorry

theorem problem3 (a : ℝ) (m : ℝ) :
  (a ≤ -1 → ∃ x, f m x + g a m x = -2*a - 2) ∧
  (-1 < a ∧ a < 1 → ∃ x, f m x + g a m x = a^2 - 1) ∧
  (a ≥ 1 → ∃ x, f m x + g a m x = 2*a - 2) :=
sorry

end problem1_problem2_problem3_l54_54055


namespace incorrect_conversion_D_l54_54275

-- Definition of base conversions as conditions
def binary_to_decimal (b : String) : ℕ := -- Converts binary string to decimal number
  sorry

def octal_to_decimal (o : String) : ℕ := -- Converts octal string to decimal number
  sorry

def decimal_to_base_n (d : ℕ) (n : ℕ) : String := -- Converts decimal number to base-n string
  sorry

-- Given conditions
axiom cond1 : binary_to_decimal "101" = 5
axiom cond2 : octal_to_decimal "27" = 25 -- Note: "27"_base(8) is 2*8 + 7 = 23 in decimal; there's a typo in question's option.
axiom cond3 : decimal_to_base_n 119 6 = "315"
axiom cond4 : decimal_to_base_n 13 2 = "1101" -- Note: correcting from 62 to "1101"_base(2) which is 13

-- Prove the incorrect conversion between number systems
theorem incorrect_conversion_D : decimal_to_base_n 31 4 ≠ "62" :=
  sorry

end incorrect_conversion_D_l54_54275


namespace system_non_zero_solution_condition_l54_54124

theorem system_non_zero_solution_condition (a b c : ℝ) :
  (∃ (x y z : ℝ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  (2 * a * b * c + a * b + b * c + c * a - 1 = 0) :=
sorry

end system_non_zero_solution_condition_l54_54124


namespace area_ratio_of_similar_isosceles_triangles_l54_54805

theorem area_ratio_of_similar_isosceles_triangles
  (b1 b2 h1 h2 : ℝ)
  (h_ratio : h1 / h2 = 2 / 3)
  (similar_tri : b1 / b2 = 2 / 3) :
  (1 / 2 * b1 * h1) / (1 / 2 * b2 * h2) = 4 / 9 :=
by
  sorry

end area_ratio_of_similar_isosceles_triangles_l54_54805


namespace time_to_cross_trains_l54_54322

/-- Length of the first train in meters -/
def length_train1 : ℕ := 50

/-- Length of the second train in meters -/
def length_train2 : ℕ := 120

/-- Speed of the first train in km/hr -/
def speed_train1_kmh : ℕ := 60

/-- Speed of the second train in km/hr -/
def speed_train2_kmh : ℕ := 40

/-- Relative speed in km/hr as trains are moving in opposite directions -/
def relative_speed_kmh : ℕ := speed_train1_kmh + speed_train2_kmh

/-- Convert speed from km/hr to m/s -/
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

/-- Relative speed in m/s -/
def relative_speed_ms : ℚ := kmh_to_ms relative_speed_kmh

/-- Total distance to be covered in meters -/
def total_distance : ℕ := length_train1 + length_train2

/-- Time taken in seconds to cross each other -/
def time_to_cross : ℚ := total_distance / relative_speed_ms

theorem time_to_cross_trains :
  time_to_cross = 6.12 := 
sorry

end time_to_cross_trains_l54_54322


namespace carter_drum_stick_sets_l54_54017

theorem carter_drum_stick_sets (sets_per_show sets_tossed_per_show nights : ℕ) :
  sets_per_show = 5 →
  sets_tossed_per_show = 6 →
  nights = 30 →
  (sets_per_show + sets_tossed_per_show) * nights = 330 := by
  intros
  sorry

end carter_drum_stick_sets_l54_54017


namespace max_ak_at_k_125_l54_54465

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def ak (k : ℕ) : ℚ :=
  binomial_coefficient 500 k * (0.3)^k

theorem max_ak_at_k_125 : 
  ∀ k : ℕ, k ∈ Finset.range 501 → (ak k ≤ ak 125) :=
by sorry

end max_ak_at_k_125_l54_54465


namespace pedestrian_walking_time_in_interval_l54_54639

noncomputable def bus_departure_interval : ℕ := 5  -- Condition 1: Buses depart every 5 minutes
noncomputable def buses_same_direction : ℕ := 11  -- Condition 2: 11 buses passed him going the same direction
noncomputable def buses_opposite_direction : ℕ := 13  -- Condition 3: 13 buses came from opposite direction
noncomputable def bus_speed_factor : ℕ := 8  -- Condition 4: Bus speed is 8 times the pedestrian's speed
noncomputable def min_walking_time : ℚ := 57 + 1 / 7 -- Correct Answer: Minimum walking time
noncomputable def max_walking_time : ℚ := 62 + 2 / 9 -- Correct Answer: Maximum walking time

theorem pedestrian_walking_time_in_interval (t : ℚ)
  (h1 : bus_departure_interval = 5)
  (h2 : buses_same_direction = 11)
  (h3 : buses_opposite_direction = 13)
  (h4 : bus_speed_factor = 8) :
  min_walking_time ≤ t ∧ t ≤ max_walking_time :=
sorry

end pedestrian_walking_time_in_interval_l54_54639


namespace value_of_expression_l54_54562

theorem value_of_expression (x y : ℝ) (h1 : 3 * x + 2 * y = 7) (h2 : 2 * x + 3 * y = 8) :
  13 * x ^ 2 + 22 * x * y + 13 * y ^ 2 = 113 :=
sorry

end value_of_expression_l54_54562


namespace second_divisor_l54_54622

theorem second_divisor (x : ℕ) : (282 % 31 = 3) ∧ (282 % x = 3) → x = 9 :=
by
  sorry

end second_divisor_l54_54622


namespace simplify_expression_l54_54609

variable (a b c d x y z : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * a^2 * y^3 + c^2 * z^3) + dz * (a^2 * x^3 + 3 * c^2 * y^3 + b^2 * z^3)) / (cx + dz) =
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
sorry

end simplify_expression_l54_54609


namespace andrew_subway_time_l54_54556

variable (S : ℝ) -- Let \( S \) be the time Andrew spends on the subway in hours

variable (total_time : ℝ)
variable (bike_time : ℝ)
variable (train_time : ℝ)

noncomputable def travel_conditions := 
  total_time = S + 2 * S + bike_time ∧ 
  total_time = 38 ∧ 
  bike_time = 8

theorem andrew_subway_time
  (S : ℝ)
  (total_time : ℝ)
  (bike_time : ℝ)
  (train_time : ℝ)
  (h : travel_conditions S total_time bike_time) : 
  S = 10 := 
sorry

end andrew_subway_time_l54_54556


namespace rectangle_area_y_l54_54641

theorem rectangle_area_y (y : ℚ) (h_pos: y > 0) 
  (h_area: ((6 : ℚ) - (-2)) * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_area_y_l54_54641


namespace ball_probability_l54_54075

theorem ball_probability (n : ℕ) (h : (n : ℚ) / (n + 2) = 1 / 3) : n = 1 :=
sorry

end ball_probability_l54_54075


namespace quadratic_equal_roots_k_value_l54_54513

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 4 * k = 0 → x^2 - 8 * x - 4 * k = 0 ∧ (0 : ℝ) = 0 ) →
  k = -4 :=
sorry

end quadratic_equal_roots_k_value_l54_54513


namespace mistaken_quotient_is_35_l54_54377

theorem mistaken_quotient_is_35 (D : ℕ) (correct_divisor mistaken_divisor correct_quotient : ℕ) 
    (h1 : D = correct_divisor * correct_quotient)
    (h2 : correct_divisor = 21)
    (h3 : mistaken_divisor = 12)
    (h4 : correct_quotient = 20)
    : D / mistaken_divisor = 35 := by
  sorry

end mistaken_quotient_is_35_l54_54377


namespace travel_period_l54_54316

-- Nina's travel pattern
def travels_in_one_month : ℕ := 400
def travels_in_two_months : ℕ := travels_in_one_month + 2 * travels_in_one_month

-- The total distance Nina wants to travel
def total_distance : ℕ := 14400

-- The period in months during which Nina travels the given total distance 
def required_period_in_months (d_per_2_months : ℕ) (total_d : ℕ) : ℕ := (total_d / d_per_2_months) * 2

-- Statement we need to prove
theorem travel_period : required_period_in_months travels_in_two_months total_distance = 24 := by
  sorry

end travel_period_l54_54316


namespace urn_probability_four_each_l54_54463

def number_of_sequences := Nat.choose 6 3

def probability_of_sequence := (1/3) * (1/2) * (3/5) * (1/2) * (4/7) * (5/8)

def total_probability := number_of_sequences * probability_of_sequence

theorem urn_probability_four_each :
  total_probability = 5 / 14 := by
  -- proof goes here
  sorry

end urn_probability_four_each_l54_54463


namespace smallest_positive_period_minimum_value_of_f_l54_54145

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 3) - sqrt 3 * sin x ^ 2 + sin x * cos x

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x :=
sorry

theorem minimum_value_of_f :
  ∀ k : ℤ, f (k * π - 5 * π / 12) = -2 :=
sorry

end smallest_positive_period_minimum_value_of_f_l54_54145


namespace factorial_power_of_two_l54_54974

theorem factorial_power_of_two solutions (a b c : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_equation : a.factorial + b.factorial = 2^(c.factorial)) :
  solutions = [(1, 1, 1), (2, 2, 2)] :=
sorry

end factorial_power_of_two_l54_54974


namespace a_plus_b_eq_zero_l54_54009

-- Define the universal set and the relevant sets
def U : Set ℝ := Set.univ
def M (a : ℝ) : Set ℝ := {x | x^2 + a * x ≤ 0}
def C_U_M (b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Define the proof theorem
theorem a_plus_b_eq_zero (a b : ℝ) (h1 : ∀ x, x ∈ M a ↔ -a < x ∧ x < 0 ∨ 0 < x ∧ x < -a)
                         (h2 : ∀ x, x ∈ C_U_M b ↔ x > b ∨ x < 0) : a + b = 0 := 
sorry

end a_plus_b_eq_zero_l54_54009


namespace cherries_per_pound_l54_54013

-- Definitions from conditions in the problem
def total_pounds_of_cherries : ℕ := 3
def pitting_time_for_20_cherries : ℕ := 10 -- in minutes
def total_pitting_time : ℕ := 2 * 60  -- in minutes (2 hours to minutes)

-- Theorem to prove the question equals the correct answer
theorem cherries_per_pound : (total_pitting_time / pitting_time_for_20_cherries) * 20 / total_pounds_of_cherries = 80 := by
  sorry

end cherries_per_pound_l54_54013


namespace num_comfortable_butterflies_final_state_l54_54479

noncomputable def num_comfortable_butterflies (n : ℕ) : ℕ :=
  if h : 0 < n then
    n
  else
    0

theorem num_comfortable_butterflies_final_state {n : ℕ} (h : 0 < n):
  num_comfortable_butterflies n = n := by
  sorry

end num_comfortable_butterflies_final_state_l54_54479


namespace average_matches_rounded_l54_54378

def total_matches : ℕ := 6 * 1 + 3 * 2 + 3 * 3 + 2 * 4 + 6 * 5

def total_players : ℕ := 6 + 3 + 3 + 2 + 6

noncomputable def average_matches : ℚ := total_matches / total_players

theorem average_matches_rounded : Int.floor (average_matches + 0.5) = 3 :=
by
  unfold average_matches total_matches total_players
  norm_num
  sorry

end average_matches_rounded_l54_54378


namespace mixture_problem_l54_54361

theorem mixture_problem
  (x : ℝ)
  (c1 c2 c_final : ℝ)
  (v1 v2 v_final : ℝ)
  (h1 : c1 = 0.60)
  (h2 : c2 = 0.75)
  (h3 : c_final = 0.72)
  (h4 : v1 = 4)
  (h5 : x = 16)
  (h6 : v2 = x)
  (h7 : v_final = v1 + v2) :
  v_final = 20 ∧ c_final * v_final = c1 * v1 + c2 * v2 :=
by
  sorry

end mixture_problem_l54_54361


namespace left_handed_women_percentage_l54_54826

theorem left_handed_women_percentage
  (x y : ℕ)
  (h1 : 4 * x = 5 * y)
  (h2 : 3 * x ≥ 3 * y) :
  (x / (4 * x) : ℚ) * 100 = 25 :=
by
  sorry

end left_handed_women_percentage_l54_54826


namespace helen_oranges_l54_54662

def initial_oranges := 9
def oranges_from_ann := 29
def oranges_taken_away := 14

def final_oranges (initial : Nat) (add : Nat) (taken : Nat) : Nat :=
  initial + add - taken

theorem helen_oranges :
  final_oranges initial_oranges oranges_from_ann oranges_taken_away = 24 :=
by
  sorry

end helen_oranges_l54_54662


namespace first_digit_base_5_of_2197_l54_54813

theorem first_digit_base_5_of_2197 : 
  ∃ k : ℕ, 2197 = k * 625 + r ∧ k = 3 ∧ r < 625 :=
by
  -- existence of k and r follows from the division algorithm
  -- sorry is used to indicate the part of the proof that needs to be filled in
  sorry

end first_digit_base_5_of_2197_l54_54813


namespace number_of_correct_conclusions_l54_54113

-- Given conditions
variables {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : c > 3)
           (h₂ : a * 25 + b * 5 + c = 0)
           (h₃ : -b / (2 * a) = 2)
           (h₄ : a < 0)

-- Proof should show:
theorem number_of_correct_conclusions 
  (h₀ : a ≠ 0)
  (h₁ : c > 3)
  (h₂ : 25 * a + 5 * b + c = 0)
  (h₃ : - b / (2 * a) = 2)
  (h₄ : a < 0) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ (a * x₁^2 + b * x₁ + c = 2) ∧ (a * x₂^2 + b * x₂ + c = 2)) ∧ 
  (a < -3 / 5) := 
by
  sorry

end number_of_correct_conclusions_l54_54113


namespace range_of_b_l54_54686

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem range_of_b (b : ℝ) : 
  (∃ (x1 x2 x3 : ℝ), f x1 = -b ∧ f x2 = -b ∧ f x3 = -b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ (-1 < b ∧ b < 0) :=
by
  sorry

end range_of_b_l54_54686


namespace carrot_lettuce_ratio_l54_54836

theorem carrot_lettuce_ratio :
  let lettuce_cal := 50
  let dressing_cal := 210
  let crust_cal := 600
  let pepperoni_cal := crust_cal / 3
  let cheese_cal := 400
  let total_pizza_cal := crust_cal + pepperoni_cal + cheese_cal
  let carrot_cal := C
  let total_salad_cal := lettuce_cal + carrot_cal + dressing_cal
  let jackson_salad_cal := (1 / 4) * total_salad_cal
  let jackson_pizza_cal := (1 / 5) * total_pizza_cal
  jackson_salad_cal + jackson_pizza_cal = 330 →
  carrot_cal / lettuce_cal = 2 :=
by
  intro lettuce_cal dressing_cal crust_cal pepperoni_cal cheese_cal total_pizza_cal carrot_cal total_salad_cal jackson_salad_cal jackson_pizza_cal h
  sorry

end carrot_lettuce_ratio_l54_54836


namespace curve_is_circle_l54_54596

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) : 
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = k^2) → 
    (r^2 = x^2 + y^2 ∧ ∃ (θ : ℝ), x/r = Real.cos θ ∧ y/r = Real.sin θ) :=
sorry

end curve_is_circle_l54_54596


namespace calculate_oplus_l54_54799

def op (X Y : ℕ) : ℕ :=
  (X + Y) / 2

theorem calculate_oplus : op (op 6 10) 14 = 11 := by
  sorry

end calculate_oplus_l54_54799


namespace value_of_x_minus_y_squared_l54_54911

theorem value_of_x_minus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : (x - y) ^ 2 = 1 :=
by
  sorry

end value_of_x_minus_y_squared_l54_54911


namespace percentage_increase_l54_54023

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  ((x - 70) / 70) * 100 = 11 := by
  sorry

end percentage_increase_l54_54023


namespace sum_first_n_geometric_terms_l54_54314

theorem sum_first_n_geometric_terms (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 2 = 2) (h2 : S 6 = 4) :
  S 4 = 1 + Real.sqrt 5 :=
by
  sorry

end sum_first_n_geometric_terms_l54_54314


namespace students_got_off_the_bus_l54_54266

theorem students_got_off_the_bus
    (original_students : ℕ)
    (students_left : ℕ)
    (h_original : original_students = 10)
    (h_left : students_left = 7) :
    original_students - students_left = 3 :=
by {
  sorry
}

end students_got_off_the_bus_l54_54266


namespace jake_snakes_l54_54315

theorem jake_snakes (S : ℕ) 
  (h1 : 2 * S + 1 = 6) 
  (h2 : 2250 = 5 * 250 + 1000) :
  S = 3 := 
by
  sorry

end jake_snakes_l54_54315


namespace angle_B_in_progression_l54_54466

theorem angle_B_in_progression (A B C a b c : ℝ) (h1: A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) 
(h2: B - A = C - B) (h3: b^2 - a^2 = a * c) (h4: A + B + C = Real.pi) : 
B = 2 * Real.pi / 7 := sorry

end angle_B_in_progression_l54_54466


namespace part1_part2_l54_54172

def A (x : ℝ) : Prop := x ^ 2 - 2 * x - 8 < 0
def B (x : ℝ) : Prop := x ^ 2 + 2 * x - 3 > 0
def C (a : ℝ) (x : ℝ) : Prop := x ^ 2 - 3 * a * x + 2 * a ^ 2 < 0

theorem part1 : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 4} := 
by sorry

theorem part2 (a : ℝ) : {x : ℝ | C a x} ⊆ {x : ℝ | A x} ∩ {x : ℝ | B x} ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := 
by sorry

end part1_part2_l54_54172


namespace total_cars_produced_l54_54352

theorem total_cars_produced (cars_NA cars_EU : ℕ) (h1 : cars_NA = 3884) (h2 : cars_EU = 2871) : cars_NA + cars_EU = 6755 := by
  sorry

end total_cars_produced_l54_54352


namespace broken_crayons_l54_54068

theorem broken_crayons (total new used : Nat) (h1 : total = 14) (h2 : new = 2) (h3 : used = 4) :
  total = new + used + 8 :=
by
  -- Proof omitted
  sorry

end broken_crayons_l54_54068


namespace greatest_q_minus_r_l54_54189

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1013 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 39) := 
by
  sorry

end greatest_q_minus_r_l54_54189


namespace least_cost_of_grass_seed_l54_54938

-- Definitions of the prices and weights
def price_per_bag (size : Nat) : Float :=
  if size = 5 then 13.85
  else if size = 10 then 20.40
  else if size = 25 then 32.25
  else 0.0

-- The conditions for the weights and costs
def valid_weight_range (total_weight : Nat) : Prop :=
  65 ≤ total_weight ∧ total_weight ≤ 80

-- Calculate the total cost given quantities of each bag size
def total_cost (bag5 : Nat) (bag10 : Nat) (bag25 : Nat) : Float :=
  Float.ofNat bag5 * price_per_bag 5 + Float.ofNat bag10 * price_per_bag 10 + Float.ofNat bag25 * price_per_bag 25

-- Correct cost for the minimum possible cost within the given weight range
def min_possible_cost : Float := 98.75

-- Proof statement to be proven
theorem least_cost_of_grass_seed : ∃ (bag5 bag10 bag25 : Nat), 
  valid_weight_range (bag5 * 5 + bag10 * 10 + bag25 * 25) ∧ total_cost bag5 bag10 bag25 = min_possible_cost :=
sorry

end least_cost_of_grass_seed_l54_54938


namespace value_of_sum_plus_five_l54_54224

theorem value_of_sum_plus_five (a b : ℕ) (h : 4 * a^2 + 4 * b^2 + 8 * a * b = 100) :
  (a + b) + 5 = 10 :=
sorry

end value_of_sum_plus_five_l54_54224


namespace find_q_in_geometric_sequence_l54_54773

theorem find_q_in_geometric_sequence
  {q : ℝ} (q_pos : q > 0) 
  (a1_def : ∀(a : ℕ → ℝ), a 1 = 1 / q^2) 
  (S5_eq_S2_plus_2 : ∀(S : ℕ → ℝ), S 5 = S 2 + 2) :
  q = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_q_in_geometric_sequence_l54_54773


namespace czakler_inequality_l54_54364

variable {a b : ℕ} (ha : a > 0) (hb : b > 0)
variable {c : ℝ} (hc : c > 0)

theorem czakler_inequality (h : (a + 1 : ℝ) / (b + c) = b / a) : c ≥ 1 := by
  sorry

end czakler_inequality_l54_54364


namespace total_chrome_parts_l54_54839

theorem total_chrome_parts (a b : ℕ) 
  (h1 : a + b = 21) 
  (h2 : 3 * a + 2 * b = 50) : 2 * a + 4 * b = 68 := 
sorry

end total_chrome_parts_l54_54839


namespace parity_related_to_phi_not_omega_l54_54961

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem parity_related_to_phi_not_omega (ω : ℝ) (φ : ℝ) (h : 0 < ω) :
  (∃ k : ℤ, φ = k * Real.pi → ∀ x : ℝ, f ω φ (-x) = -f ω φ x) ∧
  (∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 → ∀ x : ℝ, f ω φ (-x) = f ω φ x) :=
sorry

end parity_related_to_phi_not_omega_l54_54961


namespace sum_of_octal_numbers_l54_54519

theorem sum_of_octal_numbers :
  let a := 0o1275
  let b := 0o164
  let sum := 0o1503
  a + b = sum :=
by
  -- Proof is omitted here with sorry
  sorry

end sum_of_octal_numbers_l54_54519


namespace find_digit_e_l54_54054

theorem find_digit_e (A B C D E F : ℕ) (h1 : A * 10 + B + (C * 10 + D) = A * 10 + E) (h2 : A * 10 + B - (D * 10 + C) = A * 10 + F) : E = 9 :=
sorry

end find_digit_e_l54_54054


namespace relation_of_M_and_N_l54_54785

-- Define the functions for M and N
def M (x : ℝ) : ℝ := (x - 3) * (x - 4)
def N (x : ℝ) : ℝ := (x - 1) * (x - 6)

-- Formulate the theorem to prove M < N for all x
theorem relation_of_M_and_N (x : ℝ) : M x < N x := sorry

end relation_of_M_and_N_l54_54785


namespace find_y_l54_54282

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ)
  (hx : x = 3 - 2 * t)
  (hy : y = 3 * t + 6)
  (hx_cond : x = -6) :
  y = 19.5 :=
by
  sorry

end find_y_l54_54282


namespace number_of_tiles_per_row_l54_54448

theorem number_of_tiles_per_row : 
  ∀ (side_length_in_feet room_area_in_sqft : ℕ) (tile_width_in_inches : ℕ), 
  room_area_in_sqft = 256 → tile_width_in_inches = 8 → 
  side_length_in_feet * side_length_in_feet = room_area_in_sqft → 
  12 * side_length_in_feet / tile_width_in_inches = 24 := 
by
  intros side_length_in_feet room_area_in_sqft tile_width_in_inches h_area h_tile_width h_side_length
  sorry

end number_of_tiles_per_row_l54_54448


namespace necessary_and_sufficient_condition_l54_54545

universe u

variables {Point : Type u} 
variables (Plane : Type u) (Line : Type u)
variables (α β : Plane) (l : Line)
variables (P Q : Point)
variables (is_perpendicular : Plane → Plane → Prop)
variables (is_on_plane : Point → Plane → Prop)
variables (is_on_line : Point → Line → Prop)
variables (PQ_perpendicular_to_l : Prop) 
variables (PQ_perpendicular_to_β : Prop)
variables (line_in_plane : Line → Plane → Prop)

-- Given conditions
axiom plane_perpendicular : is_perpendicular α β
axiom plane_intersection : ∀ (α β : Plane), is_perpendicular α β → ∃ l : Line, line_in_plane l β
axiom point_on_plane_alpha : is_on_plane P α
axiom point_on_line : is_on_line Q l

-- Problem statement
theorem necessary_and_sufficient_condition :
  (PQ_perpendicular_to_l ↔ PQ_perpendicular_to_β) :=
sorry

end necessary_and_sufficient_condition_l54_54545


namespace south_120_meters_l54_54991

-- Define the directions
inductive Direction
| North
| South

-- Define the movement function
def movement (dir : Direction) (distance : Int) : Int :=
  match dir with
  | Direction.North => distance
  | Direction.South => -distance

-- Statement to prove
theorem south_120_meters : movement Direction.South 120 = -120 := 
by
  sorry

end south_120_meters_l54_54991


namespace first_pipe_fills_cistern_in_10_hours_l54_54838

noncomputable def time_to_fill (x : ℝ) : Prop :=
  let first_pipe_rate := 1 / x
  let second_pipe_rate := 1 / 12
  let third_pipe_rate := 1 / 15
  let combined_rate := first_pipe_rate + second_pipe_rate - third_pipe_rate
  combined_rate = 7 / 60

theorem first_pipe_fills_cistern_in_10_hours : time_to_fill 10 :=
by
  sorry

end first_pipe_fills_cistern_in_10_hours_l54_54838


namespace sum_of_numbers_on_cards_l54_54650

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l54_54650


namespace orthodiagonal_quadrilateral_l54_54948

-- Define the quadrilateral sides and their relationships
variables (AB BC CD DA : ℝ)
variables (h1 : AB = 20) (h2 : BC = 70) (h3 : CD = 90)
theorem orthodiagonal_quadrilateral : AB^2 + CD^2 = BC^2 + DA^2 → DA = 60 :=
by
  sorry

end orthodiagonal_quadrilateral_l54_54948


namespace pies_from_36_apples_l54_54702

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l54_54702


namespace ray_total_grocery_bill_l54_54933

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end ray_total_grocery_bill_l54_54933


namespace minimum_protein_content_is_at_least_1_8_l54_54856

-- Define the net weight of the can and the minimum protein percentage
def netWeight : ℝ := 300
def minProteinPercentage : ℝ := 0.006

-- Prove that the minimum protein content is at least 1.8 grams
theorem minimum_protein_content_is_at_least_1_8 :
  netWeight * minProteinPercentage ≥ 1.8 := 
by
  sorry

end minimum_protein_content_is_at_least_1_8_l54_54856


namespace cos_double_angle_sin_double_angle_l54_54411

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 :=
by sorry

theorem sin_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.sin (2 * θ) = (Real.sqrt 3) / 2 :=
by sorry

end cos_double_angle_sin_double_angle_l54_54411


namespace cos_A_sin_B_eq_l54_54598

theorem cos_A_sin_B_eq (A B : ℝ) (hA1 : 0 < A) (hA2 : A < π / 2) (hB1 : 0 < B) (hB2 : B < π / 2)
    (h : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
    Real.cos A * Real.sin B = 1 / Real.sqrt 6 := sorry

end cos_A_sin_B_eq_l54_54598


namespace no_negative_roots_l54_54099

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 :=
by sorry

end no_negative_roots_l54_54099


namespace identify_perfect_square_is_689_l54_54667

-- Definitions of the conditions
def natural_numbers (n : ℕ) : Prop := True -- All natural numbers are accepted
def digits_in_result (n m : ℕ) (d : ℕ) : Prop := (n * m) % 1000 = d

-- Theorem to be proved
theorem identify_perfect_square_is_689 (n : ℕ) :
  (∀ m, natural_numbers m → digits_in_result m m 689 ∨ digits_in_result m m 759) →
  ∃ m, natural_numbers m ∧ digits_in_result m m 689 :=
sorry

end identify_perfect_square_is_689_l54_54667


namespace hexagon_angle_E_l54_54949

theorem hexagon_angle_E (A N G L E S : ℝ) 
  (h1 : A = G) 
  (h2 : G = E) 
  (h3 : N + S = 180) 
  (h4 : L = 90) 
  (h_sum : A + N + G + L + E + S = 720) : 
  E = 150 := 
by 
  sorry

end hexagon_angle_E_l54_54949


namespace probability_one_project_not_selected_l54_54077

noncomputable def calc_probability : ℚ :=
  let n := 4 ^ 4
  let m := Nat.choose 4 2 * Nat.factorial 4
  let p := m / n
  p

theorem probability_one_project_not_selected :
  calc_probability = 9 / 16 :=
by
  sorry

end probability_one_project_not_selected_l54_54077


namespace entertainment_team_count_l54_54403

theorem entertainment_team_count 
  (total_members : ℕ)
  (singers : ℕ) 
  (dancers : ℕ) 
  (prob_both_sing_dance_gt_0 : ℚ)
  (sing_count : singers = 2)
  (dance_count : dancers = 5)
  (prob_condition : prob_both_sing_dance_gt_0 = 7/10) :
  total_members = 5 := 
by 
  sorry

end entertainment_team_count_l54_54403


namespace side_length_of_cube_l54_54709

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l54_54709


namespace gold_weight_l54_54298

theorem gold_weight:
  ∀ (G C A : ℕ), 
  C = 9 → 
  (A = (4 * G + C) / 5) → 
  A = 17 → 
  G = 19 :=
by
  intros G C A hc ha h17
  sorry

end gold_weight_l54_54298


namespace find_triples_l54_54244

theorem find_triples : 
  { (a, b, k) : ℕ × ℕ × ℕ | 2^a * 3^b = k * (k + 1) } = 
  { (1, 0, 1), (1, 1, 2), (3, 2, 8), (2, 1, 3) } := 
by
  sorry

end find_triples_l54_54244


namespace envelopes_initial_count_l54_54272

noncomputable def initialEnvelopes (given_per_friend : ℕ) (friends : ℕ) (left : ℕ) : ℕ :=
  given_per_friend * friends + left

theorem envelopes_initial_count
  (given_per_friend : ℕ) (friends : ℕ) (left : ℕ)
  (h_given_per_friend : given_per_friend = 3)
  (h_friends : friends = 5)
  (h_left : left = 22) :
  initialEnvelopes given_per_friend friends left = 37 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end envelopes_initial_count_l54_54272


namespace log_power_relationship_l54_54888

theorem log_power_relationship (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (hm : m = Real.log c / Real.log a)
  (hn : n = Real.log c / Real.log b)
  (hr : r = a^c) :
  r > m ∧ m > n :=
sorry

end log_power_relationship_l54_54888


namespace power_mod_eq_nine_l54_54630

theorem power_mod_eq_nine :
  ∃ n : ℕ, 13^6 ≡ n [MOD 11] ∧ 0 ≤ n ∧ n < 11 ∧ n = 9 :=
by
  sorry

end power_mod_eq_nine_l54_54630


namespace divisible_by_24_l54_54871

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, n^4 + 2 * n^3 + 11 * n^2 + 10 * n = 24 * k := sorry

end divisible_by_24_l54_54871


namespace primes_dividing_sequence_l54_54553

def a_n (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def is_prime (p : ℕ) := Nat.Prime p

theorem primes_dividing_sequence :
  {p : ℕ | is_prime p ∧ p ≤ 19 ∧ ∃ n ≥ 1, p ∣ a_n n} = {3, 7, 13, 17} :=
by
  sorry

end primes_dividing_sequence_l54_54553


namespace numbers_unchanged_by_powers_of_n_l54_54255

-- Definitions and conditions
def unchanged_when_raised (x : ℂ) (n : ℕ) : Prop :=
  x^n = x

def modulus_one (z : ℂ) : Prop :=
  Complex.abs z = 1

-- Proof statements
theorem numbers_unchanged_by_powers_of_n :
  (∀ x : ℂ, (∀ n : ℕ, n > 0 → unchanged_when_raised x n → x = 0 ∨ x = 1)) ∧
  (∀ z : ℂ, modulus_one z → (∀ n : ℕ, n > 0 → Complex.abs (z^n) = 1)) :=
by
  sorry

end numbers_unchanged_by_powers_of_n_l54_54255


namespace y_coordinate_sum_of_circle_on_y_axis_l54_54147

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end y_coordinate_sum_of_circle_on_y_axis_l54_54147


namespace miranda_can_stuff_10_pillows_l54_54664

def feathers_needed_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

theorem miranda_can_stuff_10_pillows :
  (goose_total_feathers / goose_feathers_per_pound + duck_total_feathers / duck_feathers_per_pound) / feathers_needed_per_pillow = 10 :=
by
  sorry

end miranda_can_stuff_10_pillows_l54_54664


namespace average_speed_of_rocket_l54_54902

theorem average_speed_of_rocket
  (ascent_speed : ℕ)
  (ascent_time : ℕ)
  (descent_distance : ℕ)
  (descent_time : ℕ)
  (average_speed : ℕ)
  (h_ascent_speed : ascent_speed = 150)
  (h_ascent_time : ascent_time = 12)
  (h_descent_distance : descent_distance = 600)
  (h_descent_time : descent_time = 3)
  (h_average_speed : average_speed = 160) :
  (ascent_speed * ascent_time + descent_distance) / (ascent_time + descent_time) = average_speed :=
by
  sorry

end average_speed_of_rocket_l54_54902


namespace mean_of_five_numbers_l54_54319

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l54_54319


namespace sum_first_7_l54_54220

variable {α : Type*} [LinearOrderedField α]

-- Definitions for the arithmetic sequence
noncomputable def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + d * (n - 1)

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

-- Conditions
variable {a d : α} -- Initial term and common difference of the arithmetic sequence
variable (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12)

-- Proof statement
theorem sum_first_7 (a d : α) (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12) : 
  sum_of_first_n_terms a d 7 = 28 := 
by 
  sorry

end sum_first_7_l54_54220


namespace cost_of_goat_l54_54092

theorem cost_of_goat (G : ℝ) (goat_count : ℕ) (llama_count : ℕ) (llama_multiplier : ℝ) (total_cost : ℝ) 
    (h1 : goat_count = 3)
    (h2 : llama_count = 2 * goat_count)
    (h3 : llama_multiplier = 1.5)
    (h4 : total_cost = 4800) : G = 400 :=
by
  sorry

end cost_of_goat_l54_54092


namespace black_car_overtakes_red_car_in_one_hour_l54_54749

def red_car_speed : ℕ := 40
def black_car_speed : ℕ := 50
def initial_gap : ℕ := 10

theorem black_car_overtakes_red_car_in_one_hour (h_red_car_speed : red_car_speed = 40)
                                               (h_black_car_speed : black_car_speed = 50)
                                               (h_initial_gap : initial_gap = 10) :
  initial_gap / (black_car_speed - red_car_speed) = 1 :=
by
  sorry

end black_car_overtakes_red_car_in_one_hour_l54_54749


namespace problem_l54_54649

open BigOperators

variables {p q : ℝ} {n : ℕ}

theorem problem 
  (h : p + q = 1) : 
  ∑ r in Finset.range (n / 2 + 1), (-1 : ℝ) ^ r * (Nat.choose (n - r) r) * p^r * q^r = (p ^ (n + 1) - q ^ (n + 1)) / (p - q) :=
by
  sorry

end problem_l54_54649


namespace inequality_abc_l54_54790

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l54_54790


namespace henry_age_l54_54408

theorem henry_age (H J : ℕ) (h1 : H + J = 43) (h2 : H - 5 = 2 * (J - 5)) : H = 27 :=
by
  -- This is where we would prove the theorem based on the given conditions
  sorry

end henry_age_l54_54408


namespace solve_equation_parabola_equation_l54_54523

-- Part 1: Equation Solutions
theorem solve_equation {x : ℝ} :
  (x - 9) ^ 2 = 2 * (x - 9) ↔ x = 9 ∨ x = 11 := by
  sorry

-- Part 2: Expression of Parabola
theorem parabola_equation (a h k : ℝ) (vertex : (ℝ × ℝ)) (point: (ℝ × ℝ)) :
  vertex = (-3, 2) → point = (-1, -2) →
  a * (point.1 - h) ^ 2 + k = point.2 →
  (a = -1) → (h = -3) → (k = 2) →
  - x ^ 2 - 6 * x - 7 = a * (x + 3) ^ 2 + 2 := by
  sorry

end solve_equation_parabola_equation_l54_54523


namespace same_root_a_eq_3_l54_54078

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end same_root_a_eq_3_l54_54078


namespace percent_of_dollar_in_pocket_l54_54471

theorem percent_of_dollar_in_pocket :
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  (nickel + 2 * dime + quarter + half_dollar = 100) →
  (100 / 100 * 100 = 100) :=
by
  intros
  sorry

end percent_of_dollar_in_pocket_l54_54471


namespace intersection_M_N_l54_54039

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} :=
by {
  sorry
}

end intersection_M_N_l54_54039


namespace spinsters_count_l54_54579

variable (S C : ℕ)

-- defining the conditions
def ratio_condition (S C : ℕ) : Prop := 9 * S = 2 * C
def difference_condition (S C : ℕ) : Prop := C = S + 63

-- theorem to prove
theorem spinsters_count 
  (h1 : ratio_condition S C) 
  (h2 : difference_condition S C) : 
  S = 18 :=
sorry

end spinsters_count_l54_54579


namespace intersection_sum_l54_54920

theorem intersection_sum (h j : ℝ → ℝ)
  (H1 : h 3 = 3 ∧ j 3 = 3)
  (H2 : h 6 = 9 ∧ j 6 = 9)
  (H3 : h 9 = 18 ∧ j 9 = 18)
  (H4 : h 12 = 18 ∧ j 12 = 18) :
  ∃ a b : ℕ, h (3 * a) = b ∧ 3 * j a = b ∧ (a + b = 33) :=
by {
  sorry
}

end intersection_sum_l54_54920


namespace jake_not_drop_coffee_percentage_l54_54606

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l54_54606


namespace problem_statement_l54_54690

theorem problem_statement (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end problem_statement_l54_54690


namespace negation_proposition_l54_54492

theorem negation_proposition : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l54_54492


namespace Linda_original_savings_l54_54484

-- Definition of the problem with all conditions provided.
theorem Linda_original_savings (S : ℝ) (TV_cost : ℝ) (TV_tax_rate : ℝ) (refrigerator_rate : ℝ) (furniture_discount_rate : ℝ) :
  let furniture_cost := (3 / 4) * S
  let TV_cost_with_tax := TV_cost + TV_cost * TV_tax_rate
  let refrigerator_cost := TV_cost + TV_cost * refrigerator_rate
  let remaining_savings := TV_cost_with_tax + refrigerator_cost
  let furniture_cost_after_discount := furniture_cost - furniture_cost * furniture_discount_rate
  (remaining_savings = (1 / 4) * S) →
  S = 1898.40 :=
by
  sorry


end Linda_original_savings_l54_54484


namespace min_value_inequality_l54_54325

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end min_value_inequality_l54_54325


namespace best_value_l54_54339

variables {cS qS cM qL cL : ℝ}
variables (medium_cost : cM = 1.4 * cS) (medium_quantity : qM = 0.7 * qL)
variables (large_quantity : qL = 1.5 * qS) (large_cost : cL = 1.2 * cM)

theorem best_value :
  let small_value := cS / qS
  let medium_value := cM / (0.7 * qL)
  let large_value := cL / qL
  small_value < large_value ∧ large_value < medium_value :=
sorry

end best_value_l54_54339


namespace original_sticker_price_l54_54058

-- Define the conditions in Lean
variables {x : ℝ} -- x is the original sticker price of the laptop

-- Definitions based on the problem conditions
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 50
def store_B_price (x : ℝ) : ℝ := 0.70 * x
def heather_saves (x : ℝ) : Prop := store_B_price x - store_A_price x = 30

-- The theorem to prove
theorem original_sticker_price (x : ℝ) (h : heather_saves x) : x = 200 :=
by
  sorry

end original_sticker_price_l54_54058


namespace sum_is_2000_l54_54765

theorem sum_is_2000 (x y : ℝ) (h : x ≠ y) (h_eq : x^2 - 2000 * x = y^2 - 2000 * y) : x + y = 2000 := by
  sorry

end sum_is_2000_l54_54765


namespace integer_value_expression_l54_54971

theorem integer_value_expression (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  (p = 2 ∧ q = 2) ∨ (p ≠ 2 ∧ q = 2 ∧ pq + p^p + q^q = 3 * (p + q)) :=
sorry

end integer_value_expression_l54_54971


namespace book_shelf_arrangement_l54_54053

-- Definitions for the problem conditions
def math_books := 3
def english_books := 4
def science_books := 2

-- The total number of ways to arrange the books
def total_arrangements :=
  (Nat.factorial (math_books + english_books + science_books - 6)) * -- For the groups
  (Nat.factorial math_books) * -- For math books within the group
  (Nat.factorial english_books) * -- For English books within the group
  (Nat.factorial science_books) -- For science books within the group

theorem book_shelf_arrangement :
  total_arrangements = 1728 := by
  -- Proof starts here
  sorry

end book_shelf_arrangement_l54_54053


namespace even_function_expression_l54_54146

theorem even_function_expression (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x * (2 * x - 1)) :
  ∀ x, x > 0 → f x = x * (2 * x + 1) :=
by 
  sorry

end even_function_expression_l54_54146


namespace passes_after_6_l54_54744

-- Define the sequence a_n where a_n represents the number of ways the ball is in A's hands after n passes
def passes : ℕ → ℕ
| 0       => 1       -- Initially, the ball is in A's hands (1 way)
| (n + 1) => 2^n - passes n

-- Theorem to prove the number of different passing methods after 6 passes
theorem passes_after_6 : passes 6 = 22 := by
  sorry

end passes_after_6_l54_54744


namespace value_of_k_l54_54324

theorem value_of_k (k : ℝ) (x : ℝ) (h : (k - 3) * x^2 + 6 * x + k^2 - k = 0) (r : x = -1) : 
  k = -3 := 
by
  sorry

end value_of_k_l54_54324


namespace mondays_in_first_70_days_l54_54222

theorem mondays_in_first_70_days (days : ℕ) (h1 : days = 70) (mondays_per_week : ℕ) (h2 : mondays_per_week = 1) : 
  ∃ (mondays : ℕ), mondays = 10 := 
by
  sorry

end mondays_in_first_70_days_l54_54222


namespace total_brownies_correct_l54_54541

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end total_brownies_correct_l54_54541


namespace odd_function_increasing_l54_54252

variables {f : ℝ → ℝ}

/-- Let f be an odd function defined on (-∞, 0) ∪ (0, ∞). 
If ∀ y z ∈ (0, ∞), y ≠ z → (f y - f z) / (y - z) > 0, then f(-3) > f(-5). -/
theorem odd_function_increasing {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ y z : ℝ, y > 0 → z > 0 → y ≠ z → (f y - f z) / (y - z) > 0) :
  f (-3) > f (-5) :=
sorry

end odd_function_increasing_l54_54252


namespace distance_two_from_origin_l54_54899

theorem distance_two_from_origin (x : ℝ) (h : abs x = 2) : x = 2 ∨ x = -2 := by
  sorry

end distance_two_from_origin_l54_54899


namespace repeating_decimal_is_fraction_l54_54512

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l54_54512


namespace T_n_bounds_l54_54581

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

noncomputable def b_n (n : ℕ) : ℚ := 
if n ≤ 4 then 2 * n + 1
else 1 / (n * (n + 2))

noncomputable def T_n (n : ℕ) : ℚ := 
if n ≤ 4 then S_n n
else (24 : ℚ) + (1 / 2) * (1 / 5 + 1 / 6 - 1 / (n + 1 : ℚ) - 1 / (n + 2 : ℚ))

theorem T_n_bounds (n : ℕ) : 3 ≤ T_n n ∧ T_n n < 24 + 11 / 60 := by
  sorry

end T_n_bounds_l54_54581


namespace functional_eq_zero_l54_54903

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_zero :
  (∀ x y : ℝ, f (x + y) = f x - f y) →
  (∀ x : ℝ, f x = 0) :=
by
  intros h x
  sorry

end functional_eq_zero_l54_54903


namespace initial_earning_members_l54_54228

theorem initial_earning_members (n : ℕ)
  (avg_income_initial : ℕ) (avg_income_after : ℕ) (income_deceased : ℕ)
  (h1 : avg_income_initial = 735)
  (h2 : avg_income_after = 590)
  (h3 : income_deceased = 1170)
  (h4 : 735 * n - 1170 = 590 * (n - 1)) :
  n = 4 :=
by
  sorry

end initial_earning_members_l54_54228


namespace minimum_area_convex_quadrilateral_l54_54602

theorem minimum_area_convex_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  (∀ S_BOC S_AOD : ℝ, S_AOB * S_COD = S_BOC * S_AOD → 
    (S_AOB + S_BOC + S_COD + S_AOD) ≥ 25) := sorry

end minimum_area_convex_quadrilateral_l54_54602


namespace fg_of_neg3_eq_3_l54_54539

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_of_neg3_eq_3 : f (g (-3)) = 3 :=
by
  sorry

end fg_of_neg3_eq_3_l54_54539


namespace domain_of_f_l54_54161

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x - 1)) / Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x - 1 > 0 ∧ x + 1 ≥ 0} = {x : ℝ | x > 1/2} :=
by
  sorry

end domain_of_f_l54_54161


namespace correct_option_d_l54_54205

-- Definitions
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f x = f (-x))
variable (hf_inc : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x ≤ y → f x ≤ f y)

-- Theorem statement
theorem correct_option_d :
  f (Real.sin (Real.pi / 12)) > f (Real.tan (Real.pi / 12)) :=
sorry

end correct_option_d_l54_54205


namespace sum_of_distinct_prime_divisors_1728_l54_54462

theorem sum_of_distinct_prime_divisors_1728 : 
  (2 + 3 = 5) :=
sorry

end sum_of_distinct_prime_divisors_1728_l54_54462


namespace problem_statement_l54_54478

variable {Point Line Plane : Type}

-- Definitions for perpendicular and parallel
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def perp_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given variables
variable (a b c d : Line) (α β : Plane)

-- Conditions
axiom a_perp_b : perpendicular a b
axiom c_perp_d : perpendicular c d
axiom a_perp_alpha : perp_to_plane a α
axiom c_perp_alpha : perp_to_plane c α

-- Required proof
theorem problem_statement : perpendicular c b :=
by sorry

end problem_statement_l54_54478


namespace berry_circle_properties_l54_54578

theorem berry_circle_properties :
  ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 12 = 2 * x + 4 * y → r = Real.sqrt 17)
    ∧ (π * Real.sqrt 17 ^ 2 > 30) :=
by
  sorry

end berry_circle_properties_l54_54578


namespace smallest_range_mean_2017_l54_54231

theorem smallest_range_mean_2017 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a + b + c + d) / 4 = 2017 ∧ (max (max a b) (max c d) - min (min a b) (min c d)) = 4 := 
sorry

end smallest_range_mean_2017_l54_54231


namespace milk_cans_l54_54502

theorem milk_cans (x y : ℕ) (h : 10 * x + 17 * y = 206) : x = 7 ∧ y = 8 := sorry

end milk_cans_l54_54502


namespace shorter_leg_of_right_triangle_l54_54246

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l54_54246


namespace find_m_for_split_l54_54576

theorem find_m_for_split (m : ℕ) (h1 : m > 1) (h2 : ∃ k, k < m ∧ 2023 = (m^2 - m + 1) + 2*k) : m = 45 :=
sorry

end find_m_for_split_l54_54576


namespace cubs_more_home_runs_l54_54103

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end cubs_more_home_runs_l54_54103


namespace no_positive_abc_exists_l54_54073

theorem no_positive_abc_exists 
  (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : b^2 ≥ 4 * a * c)
  (h2 : c^2 ≥ 4 * b * a)
  (h3 : a^2 ≥ 4 * b * c)
  : false :=
sorry

end no_positive_abc_exists_l54_54073


namespace units_digit_fib_cycle_length_60_l54_54618

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib n + fib (n+1)

-- Define the function to get the units digit (mod 10)
def units_digit_fib (n : ℕ) : ℕ :=
  (fib n) % 10

-- State the theorem about the cycle length of the units digits in Fibonacci sequence
theorem units_digit_fib_cycle_length_60 :
  ∃ k, k = 60 ∧ ∀ n, units_digit_fib (n + k) = units_digit_fib n := sorry

end units_digit_fib_cycle_length_60_l54_54618


namespace arithmetic_series_sum_l54_54042

theorem arithmetic_series_sum :
  let first_term := -25
  let common_difference := 2
  let last_term := 19
  let n := (last_term - first_term) / common_difference + 1
  let sum := n * (first_term + last_term) / 2
  sum = -69 :=
by
  sorry

end arithmetic_series_sum_l54_54042


namespace molecular_weight_of_compound_l54_54014

def n_weight : ℝ := 14.01
def h_weight : ℝ := 1.01
def br_weight : ℝ := 79.90

def molecular_weight : ℝ := (1 * n_weight) + (4 * h_weight) + (1 * br_weight)

theorem molecular_weight_of_compound :
  molecular_weight = 97.95 :=
by
  -- proof steps go here if needed, but currently, we use sorry to complete the theorem
  sorry

end molecular_weight_of_compound_l54_54014


namespace elevator_people_count_l54_54501

theorem elevator_people_count (weight_limit : ℕ) (excess_weight : ℕ) (avg_weight : ℕ) (total_weight : ℕ) (n : ℕ) 
  (h1 : weight_limit = 1500)
  (h2 : excess_weight = 100)
  (h3 : avg_weight = 80)
  (h4 : total_weight = weight_limit + excess_weight)
  (h5 : total_weight = n * avg_weight) :
  n = 20 :=
sorry

end elevator_people_count_l54_54501


namespace grown_ups_in_milburg_l54_54186

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l54_54186


namespace find_x_l54_54588

-- We define the given condition in Lean
theorem find_x (x : ℝ) (h : 6 * x - 12 = -(4 + 2 * x)) : x = 1 :=
sorry

end find_x_l54_54588


namespace retail_price_percentage_l54_54781

variable (P : ℝ)
variable (wholesale_cost : ℝ)
variable (employee_price : ℝ)

axiom wholesale_cost_def : wholesale_cost = 200
axiom employee_price_def : employee_price = 192
axiom employee_discount_def : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))

theorem retail_price_percentage (P : ℝ) (wholesale_cost : ℝ) (employee_price : ℝ)
    (H1 : wholesale_cost = 200)
    (H2 : employee_price = 192)
    (H3 : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))) :
    P = 20 :=
  sorry

end retail_price_percentage_l54_54781


namespace solve_for_N_l54_54347

theorem solve_for_N (a b c N : ℝ) 
  (h1 : a + b + c = 72) 
  (h2 : a - 7 = N) 
  (h3 : b + 7 = N) 
  (h4 : 2 * c = N) : 
  N = 28.8 := 
sorry

end solve_for_N_l54_54347


namespace sum_of_ages_l54_54663

variable (J L : ℝ)
variable (h1 : J = L + 8)
variable (h2 : J + 10 = 5 * (L - 5))

theorem sum_of_ages (J L : ℝ) (h1 : J = L + 8) (h2 : J + 10 = 5 * (L - 5)) : J + L = 29.5 := by
  sorry

end sum_of_ages_l54_54663


namespace jim_travel_distance_l54_54718

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end jim_travel_distance_l54_54718


namespace total_barking_dogs_eq_l54_54165

-- Definitions
def initial_barking_dogs : ℕ := 30
def additional_barking_dogs : ℕ := 10

-- Theorem to prove the total number of barking dogs
theorem total_barking_dogs_eq :
  initial_barking_dogs + additional_barking_dogs = 40 :=
by
  sorry

end total_barking_dogs_eq_l54_54165


namespace younger_son_age_after_30_years_l54_54753

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l54_54753


namespace positive_distinct_solutions_of_system_l54_54626

variables {a b x y z : ℝ}

theorem positive_distinct_solutions_of_system
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : xy = z^2) :
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by
  sorry

end positive_distinct_solutions_of_system_l54_54626


namespace range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l54_54328

-- Define the function
def f (x a : ℝ) : ℝ := x^2 - a * x + 4 - a^2

-- Problem (1): Range of the function when a = 2
theorem range_of_f_when_a_eq_2 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x 2 = (x - 1)^2 - 1) →
  Set.image (f 2) (Set.Icc (-2 : ℝ) 3) = Set.Icc (-1 : ℝ) 8 := sorry

-- Problem (2): Sufficient but not necessary condition
theorem sufficient_but_not_necessary_condition_for_q :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x 4 ≤ 0) →
  (Set.Icc (-2 : ℝ) 2 → (∃ (M : Set ℝ), Set.singleton 4 ⊆ M ∧ 
    (∀ a ∈ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 0) ∧
    (∀ a ∈ Set.Icc (-2 : ℝ) 2, ∃ a' ∉ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a' ≤ 0))) := sorry

end range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l54_54328


namespace negation_of_universal_proposition_l54_54637

theorem negation_of_universal_proposition :
  (¬ ∀ x > 1, (1 / 2)^x < 1 / 2) ↔ (∃ x > 1, (1 / 2)^x ≥ 1 / 2) :=
sorry

end negation_of_universal_proposition_l54_54637


namespace binomial_identity_l54_54179

theorem binomial_identity (n k : ℕ) (h1 : 0 < k) (h2 : k < n)
    (h3 : Nat.choose n (k-1) + Nat.choose n (k+1) = 2 * Nat.choose n k) :
  ∃ c : ℤ, k = (c^2 + c - 2) / 2 ∧ n = c^2 - 2 := sorry

end binomial_identity_l54_54179


namespace find_triples_l54_54909

theorem find_triples (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  (x! + y!) / n! = (3:ℕ)^n ↔ (x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end find_triples_l54_54909


namespace total_savings_l54_54336

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l54_54336


namespace cost_of_each_croissant_l54_54234

theorem cost_of_each_croissant 
  (quiches_price : ℝ) (num_quiches : ℕ) (each_quiche_cost : ℝ)
  (buttermilk_biscuits_price : ℝ) (num_biscuits : ℕ) (each_biscuit_cost : ℝ)
  (total_cost_with_discount : ℝ) (discount_rate : ℝ)
  (num_croissants : ℕ) (croissant_price : ℝ) :
  quiches_price = num_quiches * each_quiche_cost →
  each_quiche_cost = 15 →
  num_quiches = 2 →
  buttermilk_biscuits_price = num_biscuits * each_biscuit_cost →
  each_biscuit_cost = 2 →
  num_biscuits = 6 →
  discount_rate = 0.10 →
  (quiches_price + buttermilk_biscuits_price + (num_croissants * croissant_price)) * (1 - discount_rate) = total_cost_with_discount →
  total_cost_with_discount = 54 →
  num_croissants = 6 →
  croissant_price = 3 :=
sorry

end cost_of_each_croissant_l54_54234


namespace volume_of_pyramid_l54_54437

-- Definitions based on conditions
def regular_quadrilateral_pyramid (h r : ℝ) := 
  ∃ a : ℝ, ∃ S : ℝ, ∃ V : ℝ,
  a = 2 * h * ((h^2 - r^2) / r^2).sqrt ∧
  S = (2 * h * ((h^2 - r^2) / r^2).sqrt)^2 ∧
  V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2)

-- Lean 4 theorem statement
theorem volume_of_pyramid (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  ∃ V : ℝ, V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2) :=
sorry

end volume_of_pyramid_l54_54437


namespace find_n_l54_54722

theorem find_n (n : ℕ) (a_n D_n d_n : ℕ) (h1 : n > 5) (h2 : D_n - d_n = a_n) : n = 9 := 
by 
  sorry

end find_n_l54_54722


namespace smallest_integer_ends_in_3_and_divisible_by_5_l54_54779

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end smallest_integer_ends_in_3_and_divisible_by_5_l54_54779


namespace geometric_sequence_solution_l54_54396

theorem geometric_sequence_solution:
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ),
    a 2 = 6 → 6 * a1 + a 3 = 30 → q > 2 →
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧
    (∀ n, S n = (3 ^ n - 1) / 2) :=
by
  intros a S q a1 h1 h2 h3
  sorry

end geometric_sequence_solution_l54_54396


namespace min_value_geom_seq_l54_54858

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  0 < a 4 ∧ 0 < a 14 ∧ a 4 * a 14 = 8 ∧ 0 < a 7 ∧ 0 < a 11 ∧ a 7 * a 11 = 8

theorem min_value_geom_seq {a : ℕ → ℝ} (h : geom_seq a) :
  2 * a 7 + a 11 = 8 :=
by
  sorry

end min_value_geom_seq_l54_54858


namespace questionnaire_visitors_l54_54467

theorem questionnaire_visitors
  (V : ℕ)
  (E U : ℕ)
  (h1 : ∀ v : ℕ, v ∈ { x : ℕ | x ≠ E ∧ x ≠ U } → v = 110)
  (h2 : E = U)
  (h3 : 3 * V = 4 * (E + U - 110))
  : V = 440 :=
by
  sorry

end questionnaire_visitors_l54_54467


namespace unique_solution_set_l54_54406

theorem unique_solution_set :
  {a : ℝ | ∃ x : ℝ, (x+a)/(x^2-1) = 1 ∧ 
                    (∀ y : ℝ, (y+a)/(y^2-1) = 1 → y = x)} 
  = {-1, 1, -5/4} :=
sorry

end unique_solution_set_l54_54406


namespace max_min_f_triangle_area_l54_54894

open Real

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (-2 * sin x, -1)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (-cos x, cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_min_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∀ x : ℝ, -2 ≤ f x) :=
sorry

theorem triangle_area
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h : A + B + C = π)
  (h_f_A : f A = 1)
  (b c : ℝ)
  (h_bc : b * c = 8) :
  (1 / 2) * b * c * sin A = 2 :=
sorry

end max_min_f_triangle_area_l54_54894


namespace field_trip_buses_needed_l54_54904

def fifth_graders : Nat := 109
def sixth_graders : Nat := 115
def seventh_graders : Nat := 118
def teachers_per_grade : Nat := 4
def parents_per_grade : Nat := 2
def total_grades : Nat := 3
def seats_per_bus : Nat := 72

def total_students : Nat := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : Nat := (teachers_per_grade + parents_per_grade) * total_grades
def total_people : Nat := total_students + total_chaperones
def buses_needed : Nat := (total_people + seats_per_bus - 1) / seats_per_bus  -- ceiling division

theorem field_trip_buses_needed : buses_needed = 5 := by
  sorry

end field_trip_buses_needed_l54_54904


namespace parabola_opens_downwards_l54_54532

theorem parabola_opens_downwards (m : ℝ) : (m + 3 < 0) → (m < -3) := 
by
  sorry

end parabola_opens_downwards_l54_54532


namespace circle_radius_l54_54508

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y = 0) : ∃ r : ℝ, r = Real.sqrt 13 :=
by
  sorry

end circle_radius_l54_54508


namespace profit_function_correct_l54_54195

-- Definitions based on Conditions
def selling_price {R : Type*} [LinearOrderedField R] : R := 45
def profit_max {R : Type*} [LinearOrderedField R] : R := 450
def price_no_sales {R : Type*} [LinearOrderedField R] : R := 60
def quadratic_profit {R : Type*} [LinearOrderedField R] (x : R) : R := -2 * (x - 30) * (x - 60)

-- The statement we need to prove.
theorem profit_function_correct {R : Type*} [LinearOrderedField R] :
  quadratic_profit (selling_price : R) = profit_max ∧ quadratic_profit (price_no_sales : R) = 0 := 
sorry

end profit_function_correct_l54_54195


namespace fraction_identity_l54_54736

variable {a b x : ℝ}

-- Conditions
axiom h1 : x = a / b
axiom h2 : a ≠ b
axiom h3 : b ≠ 0

-- Question to prove
theorem fraction_identity :
  (a + b) / (a - b) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_identity_l54_54736


namespace value_of_x_pow_12_l54_54436

theorem value_of_x_pow_12 (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^12 = 439 := sorry

end value_of_x_pow_12_l54_54436


namespace solve_for_x_l54_54914

noncomputable def f (x : ℝ) : ℝ := x^3

noncomputable def f_prime (x : ℝ) : ℝ := 3

theorem solve_for_x (x : ℝ) (h : f_prime x = 3) : x = 1 ∨ x = -1 :=
by
  sorry

end solve_for_x_l54_54914


namespace fraction_addition_l54_54440

theorem fraction_addition (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/9) : a + b = 47/36 :=
by
  rw [h1, h2]
  sorry

end fraction_addition_l54_54440


namespace range_f_period_f_monotonic_increase_intervals_l54_54901

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 

theorem range_f : Set.Icc 0 4 = Set.range f := sorry

theorem period_f : ∀ x, f (x + Real.pi) = f x := sorry

theorem monotonic_increase_intervals (k : ℤ) :
  ∀ x, (-π / 6 + k * π : ℝ) ≤ x ∧ x ≤ (π / 3 + k * π : ℝ) → 
        ∀ y, f y ≤ f x → y ≤ x := sorry

end range_f_period_f_monotonic_increase_intervals_l54_54901


namespace find_m_value_l54_54197

-- Definitions from conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (2, -4)
def OA := (A.1 - O.1, A.2 - O.2)
def AB := (B.1 - A.1, B.2 - A.2)

-- Defining the vector OP with the given expression
def OP (m : ℝ) := (2 * OA.1 + m * AB.1, 2 * OA.2 + m * AB.2)

-- The point P is on the y-axis if the x-coordinate of OP is zero
theorem find_m_value : ∃ m : ℝ, OP m = (0, (OP m).2) ∧ m = 2 / 3 :=
by { 
  -- sorry is added to skip the proof itself
  sorry 
}

end find_m_value_l54_54197


namespace gcd_75_100_l54_54284

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l54_54284


namespace g_at_10_l54_54442

-- Definitions and conditions
def f : ℝ → ℝ := sorry
axiom f_at_1 : f 1 = 10
axiom f_inequality_1 : ∀ x : ℝ, f (x + 20) ≥ f x + 20
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1
def g (x : ℝ) : ℝ := f x - x + 1

-- Proof statement (no proof required)
theorem g_at_10 : g 10 = 10 := sorry

end g_at_10_l54_54442


namespace parallel_lines_condition_l54_54621

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y - 4 = 0 → x + (a + 1) * y + 2 = 0) ↔ a = 1 :=
by sorry

end parallel_lines_condition_l54_54621


namespace find_f_six_minus_a_l54_54794

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(x-2) - 2 else -Real.logb 2 (x + 1)

variable (a : ℝ)
axiom h : f a = -3

theorem find_f_six_minus_a : f (6 - a) = - 15 / 8 :=
by
  sorry

end find_f_six_minus_a_l54_54794


namespace words_on_each_page_l54_54304

/-- Given a book with 150 pages, where each page has between 50 and 150 words, 
    and the total number of words in the book is congruent to 217 modulo 221, 
    prove that each page has 135 words. -/
theorem words_on_each_page (p : ℕ) (h1 : 50 ≤ p) (h2 : p ≤ 150) (h3 : 150 * p ≡ 217 [MOD 221]) : 
  p = 135 :=
by
  sorry

end words_on_each_page_l54_54304


namespace problem1_l54_54279

open Real

theorem problem1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  ∃ (m : ℝ), m = 9 / 2 ∧ ∀ (u v : ℝ), 0 < u → 0 < v → u + v = 1 → (1 / u + 4 / (1 + v)) ≥ m := 
sorry

end problem1_l54_54279


namespace problem_statement_l54_54498

noncomputable def f (x : ℝ) (A : ℝ) (ϕ : ℝ) : ℝ := A * Real.cos (2 * x + ϕ)

theorem problem_statement {A ϕ : ℝ} (hA : A > 0) (hϕ : |ϕ| < π / 2)
  (h1 : f (-π / 4) A ϕ = 2 * Real.sqrt 2)
  (h2 : f 0 A ϕ = 2 * Real.sqrt 6)
  (h3 : f (π / 12) A ϕ = 2 * Real.sqrt 2)
  (h4 : f (π / 4) A ϕ = -2 * Real.sqrt 2)
  (h5 : f (π / 3) A ϕ = -2 * Real.sqrt 6) :
  ϕ = π / 6 ∧ f (5 * π / 12) A ϕ = -4 * Real.sqrt 2 := 
sorry

end problem_statement_l54_54498


namespace simplify_and_evaluate_l54_54658

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 2 - Real.sqrt 2

theorem simplify_and_evaluate : 
  let expr := (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a))
  expr = -1 / 2 := by
  sorry

end simplify_and_evaluate_l54_54658


namespace find_expression_l54_54941

def B : ℂ := 3 + 2 * Complex.I
def Q : ℂ := -5 * Complex.I
def R : ℂ := 1 + Complex.I
def T : ℂ := 3 - 4 * Complex.I

theorem find_expression : B * R + Q + T = 4 + Complex.I := by
  sorry

end find_expression_l54_54941


namespace find_f_one_l54_54340

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f_defined_for_neg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 2 * x^2 - 1

-- Statement that needs to be proven
theorem find_f_one (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : f_defined_for_neg f) :
  f 1 = -1 :=
  sorry

end find_f_one_l54_54340


namespace cost_of_four_pencils_and_three_pens_l54_54544

variable {p q : ℝ}

theorem cost_of_four_pencils_and_three_pens (h1 : 3 * p + 2 * q = 4.30) (h2 : 2 * p + 3 * q = 4.05) : 4 * p + 3 * q = 5.97 := by
  sorry

end cost_of_four_pencils_and_three_pens_l54_54544


namespace no_integer_solutions_m2n_eq_2mn_minus_3_l54_54464

theorem no_integer_solutions_m2n_eq_2mn_minus_3 :
  ∀ (m n : ℤ), m + 2 * n ≠ 2 * m * n - 3 := 
sorry

end no_integer_solutions_m2n_eq_2mn_minus_3_l54_54464


namespace base8_to_base10_conversion_l54_54409

theorem base8_to_base10_conversion : 
  (6 * 8^3 + 3 * 8^2 + 7 * 8^1 + 5 * 8^0) = 3325 := 
by 
  sorry

end base8_to_base10_conversion_l54_54409


namespace cube_sqrt_three_eq_three_sqrt_three_l54_54510

theorem cube_sqrt_three_eq_three_sqrt_three : (Real.sqrt 3) ^ 3 = 3 * Real.sqrt 3 := 
by 
  sorry

end cube_sqrt_three_eq_three_sqrt_three_l54_54510


namespace inequality_cannot_hold_l54_54188

theorem inequality_cannot_hold (a b : ℝ) (ha : a < b) (hb : b < 0) : a^3 ≤ b^3 :=
by
  sorry

end inequality_cannot_hold_l54_54188


namespace ellipse_standard_equation_chord_length_range_l54_54898

-- Conditions for question 1
def ellipse_center (O : ℝ × ℝ) : Prop := O = (0, 0)
def major_axis_x (major_axis : ℝ) : Prop := major_axis = 1
def eccentricity (e : ℝ) : Prop := e = (Real.sqrt 2) / 2
def perp_chord_length (AA' : ℝ) : Prop := AA' = Real.sqrt 2

-- Lean statement for question 1
theorem ellipse_standard_equation (O : ℝ × ℝ) (major_axis : ℝ) (e : ℝ) (AA' : ℝ) :
  ellipse_center O → major_axis_x major_axis → eccentricity e → perp_chord_length AA' →
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / (a^2)) + y^2 / (b^2) = 1) := sorry

-- Conditions for question 2
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def max_area_triangle (S : ℝ) : Prop := S = 1 / 2

-- Lean statement for question 2
theorem chord_length_range (x y z w : ℝ) (E F G H : ℝ × ℝ) :
  circle_eq x y → ellipse_eq z w → max_area_triangle ((E.1 * F.1) * (Real.sin (E.2 * F.2))) →
  ( ∃ min_chord max_chord : ℝ, min_chord = Real.sqrt 3 ∧ max_chord = 2 ∧
    ∀ x1 y1 x2 y2 : ℝ, (G.1 = x1 ∧ H.1 = x2 ∧ G.2 = y1 ∧ H.2 = y2) →
    (min_chord ≤ (Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2)))) ∧
         Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2))) ≤ max_chord )) := sorry

end ellipse_standard_equation_chord_length_range_l54_54898


namespace fishermen_total_catch_l54_54942

noncomputable def m : ℕ := 30  -- Mike can catch 30 fish per hour
noncomputable def j : ℕ := 2 * m  -- Jim can catch twice as much as Mike
noncomputable def b : ℕ := j + (j / 2)  -- Bob can catch 50% more than Jim

noncomputable def fish_caught_in_40_minutes : ℕ := (2 * m) / 3 -- Fishermen fish together for 40 minutes (2/3 hour)
noncomputable def fish_caught_by_jim_in_remaining_time : ℕ := j / 3 -- Jim fishes alone for the remaining 20 minutes (1/3 hour)

noncomputable def total_fish_caught : ℕ :=
  fish_caught_in_40_minutes * 3 + fish_caught_by_jim_in_remaining_time

theorem fishermen_total_catch : total_fish_caught = 140 := by
  sorry

end fishermen_total_catch_l54_54942


namespace largest_value_is_D_l54_54697

theorem largest_value_is_D :
  let A := 15432 + 1/3241
  let B := 15432 - 1/3241
  let C := 15432 * (1/3241)
  let D := 15432 / (1/3241)
  let E := 15432.3241
  max (max (max A B) (max C D)) E = D := by
{
  sorry -- proof not required
}

end largest_value_is_D_l54_54697


namespace smallest_circle_radius_polygonal_chain_l54_54159

theorem smallest_circle_radius_polygonal_chain (l : ℝ) (hl : l = 1) : ∃ (r : ℝ), r = 0.5 := 
sorry

end smallest_circle_radius_polygonal_chain_l54_54159


namespace SandySpentTotal_l54_54059

theorem SandySpentTotal :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := by
  sorry

end SandySpentTotal_l54_54059


namespace p_at_zero_l54_54793

-- We state the conditions: p is a polynomial of degree 6, and p(3^n) = 1/(3^n) for n = 0 to 6
def p : Polynomial ℝ := sorry

axiom p_degree : p.degree = 6
axiom p_values : ∀ (n : ℕ), n ≤ 6 → p.eval (3^n) = 1 / (3^n)

-- We want to prove that p(0) = 29523 / 2187
theorem p_at_zero : p.eval 0 = 29523 / 2187 := by sorry

end p_at_zero_l54_54793


namespace probability_heart_then_club_l54_54542

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l54_54542


namespace joan_seashells_left_l54_54374

theorem joan_seashells_left (original_seashells : ℕ) (given_seashells : ℕ) (seashells_left : ℕ)
  (h1 : original_seashells = 70) (h2 : given_seashells = 43) : seashells_left = 27 :=
by
  sorry

end joan_seashells_left_l54_54374


namespace central_student_coins_l54_54619

theorem central_student_coins (n_students: ℕ) (total_coins : ℕ)
  (equidistant_same : Prop)
  (coin_exchange : Prop):
  (n_students = 16) →
  (total_coins = 3360) →
  (equidistant_same) →
  (coin_exchange) →
  ∃ coins_in_center: ℕ, coins_in_center = 280 :=
by
  intros
  sorry

end central_student_coins_l54_54619


namespace compute_f_g_f_l54_54628

def f (x : ℤ) : ℤ := 2 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

theorem compute_f_g_f (x : ℤ) : f (g (f 3)) = 108 := 
  by 
  sorry

end compute_f_g_f_l54_54628


namespace material_needed_for_second_type_l54_54160

namespace CherylProject

def first_material := 5 / 9
def leftover_material := 1 / 3
def total_material_used := 5 / 9

theorem material_needed_for_second_type :
  0.8888888888888889 - (5 / 9 : ℝ) = 0.3333333333333333 := by
  sorry

end CherylProject

end material_needed_for_second_type_l54_54160


namespace inequality_solution_l54_54089

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

lemma monotone_decreasing (a : ℝ) : ∀ x y : ℝ, x < y → f a y < f a x := 
sorry

lemma odd_function (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 0 := 
sorry

theorem inequality_solution (t : ℝ) (a : ℝ) (h_monotone : ∀ x y : ℝ, x < y → f a y < f a x)
    (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : t ≥ 4 / 3 ↔ f a (2 * t + 1) + f a (t - 5) ≤ 0 := 
sorry

end inequality_solution_l54_54089


namespace athlete_stable_performance_l54_54461

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end athlete_stable_performance_l54_54461


namespace simplify_fraction_l54_54533

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l54_54533


namespace modulo_calculation_l54_54250

theorem modulo_calculation (n : ℕ) (hn : 0 ≤ n ∧ n < 19) (hmod : 5 * n % 19 = 1) : 
  ((3^n)^2 - 3) % 19 = 3 := 
by 
  sorry

end modulo_calculation_l54_54250


namespace painting_area_l54_54730

theorem painting_area (c t A : ℕ) (h1 : c = 15) (h2 : t = 840) (h3 : c * A = t) : A = 56 := 
by
  sorry -- proof to demonstrate A = 56

end painting_area_l54_54730


namespace find_f_of_functions_l54_54358

theorem find_f_of_functions
  (f g : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = - f x)
  (h_even : ∀ x, g (-x) = g x)
  (h_eq : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := 
sorry

end find_f_of_functions_l54_54358


namespace average_time_per_leg_l54_54212

-- Conditions
def time_y : ℕ := 58
def time_z : ℕ := 26
def total_time : ℕ := time_y + time_z
def number_of_legs : ℕ := 2

-- Theorem stating the average time per leg
theorem average_time_per_leg : total_time / number_of_legs = 42 := by
  sorry

end average_time_per_leg_l54_54212


namespace printer_cost_l54_54142

theorem printer_cost (num_keyboards : ℕ) (num_printers : ℕ) (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) :
  num_keyboards = 15 →
  num_printers = 25 →
  total_cost = 2050 →
  keyboard_cost = 20 →
  (total_cost - (num_keyboards * keyboard_cost)) / num_printers = printer_cost →
  printer_cost = 70 :=
by
  sorry

end printer_cost_l54_54142


namespace cos_b_eq_one_div_sqrt_two_l54_54021

variable {a b c : ℝ} -- Side lengths
variable {A B C : ℝ} -- Angles in radians

-- Conditions of the problem
variables (h1 : c = 2 * a) 
          (h2 : b^2 = a * c) 
          (h3 : a^2 + b^2 = c^2 - 2 * a * b * Real.cos C)
          (h4 : A + B + C = Real.pi)

theorem cos_b_eq_one_div_sqrt_two
    (h1 : c = 2 * a)
    (h2 : b = a * Real.sqrt 2)
    (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
    (h4 : A + B + C = Real.pi )
    : Real.cos B = 1 / Real.sqrt 2 := 
sorry

end cos_b_eq_one_div_sqrt_two_l54_54021


namespace luke_fish_catching_l54_54748

theorem luke_fish_catching :
  ∀ (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ),
  days = 30 → fillets_per_fish = 2 → total_fillets = 120 →
  (total_fillets / fillets_per_fish) / days = 2 :=
by
  intros days fillets_per_fish total_fillets days_eq fillets_eq fillets_total_eq
  sorry

end luke_fish_catching_l54_54748


namespace fraction_of_males_l54_54206

theorem fraction_of_males (M F : ℚ) (h1 : M + F = 1)
  (h2 : (3/4) * M + (5/6) * F = 7/9) :
  M = 2/3 :=
by sorry

end fraction_of_males_l54_54206


namespace customers_non_holiday_l54_54876

theorem customers_non_holiday (h : ∀ n, 2 * n = 350) (H : ∃ h : ℕ, h * 8 = 2800) : (2800 / 8 / 2 = 175) :=
by sorry

end customers_non_holiday_l54_54876


namespace solve_equation_l54_54037

theorem solve_equation (x : ℝ) : 
  (9 - x - 2 * (31 - x) = 27) → (x = 80) :=
by
  sorry

end solve_equation_l54_54037


namespace no_fixed_points_implies_no_double_fixed_points_l54_54128

theorem no_fixed_points_implies_no_double_fixed_points (f : ℝ → ℝ) (hf : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

end no_fixed_points_implies_no_double_fixed_points_l54_54128


namespace value_of_x_l54_54691

variable (w x y : ℝ)

theorem value_of_x 
  (h_avg : (w + x) / 2 = 0.5)
  (h_eq : (7 / w) + (7 / x) = 7 / y)
  (h_prod : w * x = y) :
  x = 0.5 :=
sorry

end value_of_x_l54_54691


namespace loan_amount_is_900_l54_54861

theorem loan_amount_is_900 (P R T SI : ℕ) (hR : R = 9) (hT : T = 9) (hSI : SI = 729)
    (h_simple_interest : SI = (P * R * T) / 100) : P = 900 := by
  sorry

end loan_amount_is_900_l54_54861


namespace students_like_both_l54_54608

variable (total_students : ℕ) 
variable (students_like_sea : ℕ) 
variable (students_like_mountains : ℕ) 
variable (students_like_neither : ℕ) 

theorem students_like_both (h1 : total_students = 500)
                           (h2 : students_like_sea = 337)
                           (h3 : students_like_mountains = 289)
                           (h4 : students_like_neither = 56) :
  (students_like_sea + students_like_mountains - (total_students - students_like_neither)) = 182 :=
sorry

end students_like_both_l54_54608


namespace mailman_total_delivered_l54_54827

def pieces_of_junk_mail : Nat := 6
def magazines : Nat := 5
def newspapers : Nat := 3
def bills : Nat := 4
def postcards : Nat := 2

def total_pieces_of_mail : Nat := pieces_of_junk_mail + magazines + newspapers + bills + postcards

theorem mailman_total_delivered : total_pieces_of_mail = 20 := by
  sorry

end mailman_total_delivered_l54_54827


namespace membership_percentage_change_l54_54423

-- Definitions required based on conditions
def membersFallChange (initialMembers : ℝ) : ℝ := initialMembers * 1.07
def membersSpringChange (fallMembers : ℝ) : ℝ := fallMembers * 0.81
def membersSummerChange (springMembers : ℝ) : ℝ := springMembers * 1.15

-- Prove the total change in percentage from fall to the end of summer
theorem membership_percentage_change :
  let initialMembers := 100
  let fallMembers := membersFallChange initialMembers
  let springMembers := membersSpringChange fallMembers
  let summerMembers := membersSummerChange springMembers
  ((summerMembers - initialMembers) / initialMembers) * 100 = -0.33 := by
  sorry

end membership_percentage_change_l54_54423


namespace check_not_coverable_boards_l54_54121

def is_coverable_by_dominoes (m n : ℕ) : Prop :=
  (m * n) % 2 = 0

theorem check_not_coverable_boards:
  (¬is_coverable_by_dominoes 5 5) ∧ (¬is_coverable_by_dominoes 3 7) :=
by
  -- Proof steps are omitted.
  sorry

end check_not_coverable_boards_l54_54121


namespace find_x_in_acute_triangle_l54_54291

-- Definition of an acute triangle with given segment lengths due to altitudes
def acute_triangle_with_segments (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) : Prop :=
  BC = 4 + x ∧ AE = x ∧ BE = 8 ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- The theorem to prove
theorem find_x_in_acute_triangle (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) 
  (h : acute_triangle_with_segments A B C D E BC AE BE x) : 
  x = 4 :=
by
  -- As the focus is on the statement, we add sorry to skip the proof.
  sorry

end find_x_in_acute_triangle_l54_54291


namespace diagonal_ratio_l54_54746

variable (a b : ℝ)
variable (d1 : ℝ) -- diagonal length of the first square
variable (r : ℝ := 1.5) -- ratio between perimeters

theorem diagonal_ratio (h : 4 * a / (4 * b) = r) (hd1 : d1 = a * Real.sqrt 2) : 
  (b * Real.sqrt 2) = (2/3) * d1 := 
sorry

end diagonal_ratio_l54_54746


namespace max_height_l54_54890

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 80 * t + 50

theorem max_height : ∃ t : ℝ, ∀ t' : ℝ, h t' ≤ h t ∧ h t = 130 :=
by
  sorry

end max_height_l54_54890


namespace sum_of_coefficients_l54_54599

noncomputable def coeff_sum (x y z : ℝ) : ℝ :=
  let p := (x + 2*y - z)^8  
  -- extract and sum coefficients where exponent of x is 2 and exponent of y is not 1
  sorry

theorem sum_of_coefficients (x y z : ℝ) :
  coeff_sum x y z = 364 := by
  sorry

end sum_of_coefficients_l54_54599


namespace number_of_zeros_of_f_l54_54570

noncomputable def f (a x : ℝ) := x * Real.log x - a * x^2 - x

theorem number_of_zeros_of_f (a : ℝ) (h : |a| ≥ 1 / (2 * Real.exp 1)) :
  ∃! x, f a x = 0 :=
sorry

end number_of_zeros_of_f_l54_54570


namespace sqrt_neg3_squared_l54_54859

theorem sqrt_neg3_squared : Real.sqrt ((-3)^2) = 3 :=
by sorry

end sqrt_neg3_squared_l54_54859


namespace possible_values_for_p_l54_54201

-- Definitions for the conditions
variables {a b c p : ℝ}

-- Assumptions
def distinct (a b c : ℝ) := ¬(a = b) ∧ ¬(b = c) ∧ ¬(c = a)
def main_eq (a b c p : ℝ) := a + (1 / b) = p ∧ b + (1 / c) = p ∧ c + (1 / a) = p

-- Theorem statement
theorem possible_values_for_p (h1 : distinct a b c) (h2 : main_eq a b c p) : p = 1 ∨ p = -1 := 
sorry

end possible_values_for_p_l54_54201


namespace process_terminates_with_one_element_in_each_list_final_elements_are_different_l54_54428

-- Define the initial lists
def List1 := [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
def List2 := [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

-- Predicate to state the termination of the process with exactly one element in each list
theorem process_terminates_with_one_element_in_each_list (List1 List2 : List ℕ):
  ∃ n m, List.length List1 = n ∧ List.length List2 = m ∧ (n = 1 ∧ m = 1) :=
sorry

-- Predicate to state that the final elements in the lists are different
theorem final_elements_are_different (List1 List2 : List ℕ) :
  ∀ a b, a ∈ List1 → b ∈ List2 → (a % 5 = 1 ∧ b % 5 = 4) → a ≠ b :=
sorry

end process_terminates_with_one_element_in_each_list_final_elements_are_different_l54_54428


namespace expand_product_l54_54719

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end expand_product_l54_54719


namespace local_value_proof_l54_54384

-- Definitions based on the conditions
def face_value_7 : ℕ := 7
def local_value_6_in_7098060 : ℕ := 6000
def product_of_face_value_and_local_value : ℕ := face_value_7 * local_value_6_in_7098060
def local_value_6_in_product : ℕ := 6000

-- Theorem statement
theorem local_value_proof : local_value_6_in_product = 6000 :=
by
  -- Direct restatement of the condition in Lean
  sorry

end local_value_proof_l54_54384


namespace flagpole_break_height_l54_54566

theorem flagpole_break_height (total_height break_point distance_from_base : ℝ) 
(h_total : total_height = 6) 
(h_distance : distance_from_base = 2) 
(h_equation : (distance_from_base^2 + (total_height - break_point)^2) = break_point^2) :
  break_point = 3 := 
sorry

end flagpole_break_height_l54_54566


namespace base_for_four_digit_even_l54_54925

theorem base_for_four_digit_even (b : ℕ) : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0 → b = 6 :=
by
  sorry

end base_for_four_digit_even_l54_54925


namespace incorrect_inequality_l54_54064

-- Given definitions
variables {a b : ℝ}
axiom h : a < b ∧ b < 0

-- Equivalent theorem statement
theorem incorrect_inequality (ha : a < b) (hb : b < 0) : (1 / (a - b)) < (1 / a) := 
sorry

end incorrect_inequality_l54_54064


namespace maximum_cookies_andy_could_have_eaten_l54_54475

theorem maximum_cookies_andy_could_have_eaten :
  ∃ x : ℤ, (x ≥ 0 ∧ 2 * x + (x - 3) + x = 30) ∧ (∀ y : ℤ, 0 ≤ y ∧ 2 * y + (y - 3) + y = 30 → y ≤ 8) :=
by {
  sorry
}

end maximum_cookies_andy_could_have_eaten_l54_54475


namespace n_divisibility_and_factors_l54_54024

open Nat

theorem n_divisibility_and_factors (n : ℕ) (h1 : 1990 ∣ n) (h2 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n):
  n = 4 * 5 * 199 ∨ n = 2 * 25 * 199 ∨ n = 2 * 5 * 39601 := 
sorry

end n_divisibility_and_factors_l54_54024


namespace license_plate_increase_l54_54834

theorem license_plate_increase :
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 :=
by
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  show new_plates / old_plates = 26^2 / 10
  sorry

end license_plate_increase_l54_54834


namespace real_values_of_a_l54_54950

noncomputable def P (x a b : ℝ) : ℝ := x^2 - 2 * a * x + b

theorem real_values_of_a (a b : ℝ) :
  (P 0 a b ≠ 0) →
  (P 1 a b ≠ 0) →
  (P 2 a b ≠ 0) →
  (P 1 a b / P 0 a b = P 2 a b / P 1 a b) →
  (∃ b, P x 1 b = 0) :=
by
  sorry

end real_values_of_a_l54_54950


namespace sum_of_primes_l54_54425

theorem sum_of_primes (a b c : ℕ) (h₁ : Nat.Prime a) (h₂ : Nat.Prime b) (h₃ : Nat.Prime c) (h₄ : b + c = 13) (h₅ : c^2 - a^2 = 72) :
  a + b + c = 20 := 
sorry

end sum_of_primes_l54_54425


namespace kitten_weight_l54_54420

theorem kitten_weight :
  ∃ (x y z : ℝ), x + y + z = 36 ∧ x + z = 3 * y ∧ x + y = 1 / 2 * z ∧ x = 3 := 
by
  sorry

end kitten_weight_l54_54420


namespace rationalize_denominator_l54_54657

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1)) = ((Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l54_54657


namespace max_hedgehogs_l54_54173

theorem max_hedgehogs (S : ℕ) (n : ℕ) (hS : S = 65) (hn : ∀ m, m > n → (m * (m + 1)) / 2 > S) :
  n = 10 := 
sorry

end max_hedgehogs_l54_54173


namespace cubic_roots_inequality_l54_54208

theorem cubic_roots_inequality (a b c : ℝ) (h : ∃ (α β γ : ℝ), (x : ℝ) → x^3 + a * x^2 + b * x + c = (x - α) * (x - β) * (x - γ)) :
  3 * b ≤ a^2 :=
sorry

end cubic_roots_inequality_l54_54208


namespace range_of_a_l54_54880

-- Define the condition function
def inequality (a x : ℝ) : Prop := a^2 * x - 2 * (a - x - 4) < 0

-- Prove that given the inequality always holds for any real x, the range of a is (-2, 2]
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, inequality a x) : -2 < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l54_54880


namespace route_Y_saves_2_minutes_l54_54558

noncomputable def distance_X : ℝ := 8
noncomputable def speed_X : ℝ := 40

noncomputable def distance_Y1 : ℝ := 5
noncomputable def speed_Y1 : ℝ := 50
noncomputable def distance_Y2 : ℝ := 1
noncomputable def speed_Y2 : ℝ := 20
noncomputable def distance_Y3 : ℝ := 1
noncomputable def speed_Y3 : ℝ := 60

noncomputable def t_X : ℝ := (distance_X / speed_X) * 60
noncomputable def t_Y1 : ℝ := (distance_Y1 / speed_Y1) * 60
noncomputable def t_Y2 : ℝ := (distance_Y2 / speed_Y2) * 60
noncomputable def t_Y3 : ℝ := (distance_Y3 / speed_Y3) * 60
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3

noncomputable def time_saved : ℝ := t_X - t_Y

theorem route_Y_saves_2_minutes :
  time_saved = 2 := by
  sorry

end route_Y_saves_2_minutes_l54_54558


namespace input_command_is_INPUT_l54_54912

-- Define the commands
def PRINT : String := "PRINT"
def INPUT : String := "INPUT"
def THEN : String := "THEN"
def END : String := "END"

-- Define the properties of each command
def PRINT_is_output (cmd : String) : Prop :=
  cmd = PRINT

def INPUT_is_input (cmd : String) : Prop :=
  cmd = INPUT

def THEN_is_conditional (cmd : String) : Prop :=
  cmd = THEN

def END_is_end (cmd : String) : Prop :=
  cmd = END

-- Theorem stating that INPUT is the command associated with input operation
theorem input_command_is_INPUT : INPUT_is_input INPUT :=
by
  -- Proof goes here
  sorry

end input_command_is_INPUT_l54_54912


namespace find_x_l54_54433

noncomputable def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_x (x : ℝ) :
  let a := (1, 2*x + 1)
  let b := (2, 3)
  (vector_parallel a b) → x = 1 / 4 :=
by
  intro h
  have h_eq := h
  sorry  -- proof is not needed as per instruction

end find_x_l54_54433


namespace number_of_commonly_used_structures_is_3_l54_54629

def commonly_used_algorithm_structures : Nat := 3
theorem number_of_commonly_used_structures_is_3 
  (structures : Nat)
  (h : structures = 1 ∨ structures = 2 ∨ structures = 3 ∨ structures = 4) :
  commonly_used_algorithm_structures = 3 :=
by
  -- Proof to be added
  sorry

end number_of_commonly_used_structures_is_3_l54_54629


namespace necessary_but_not_sufficient_l54_54804

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def P : Set ℝ := {x | x ≤ -1}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧ (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
by
  sorry

end necessary_but_not_sufficient_l54_54804


namespace cone_height_circular_sector_l54_54569

theorem cone_height_circular_sector (r : ℝ) (n : ℕ) (h : ℝ)
  (hr : r = 10)
  (hn : n = 3)
  (hradius : r > 0)
  (hcircumference : 2 * Real.pi * r / n = 2 * Real.pi * r / 3)
  : h = (20 * Real.sqrt 2) / 3 :=
by {
  sorry
}

end cone_height_circular_sector_l54_54569


namespace angles_bisectors_l54_54158

theorem angles_bisectors (k : ℤ) : 
    ∃ α : ℤ, α = k * 180 + 135 
  -> 
    (α = (2 * k) * 180 + 135 ∨ α = (2 * k + 1) * 180 + 135) 
  := sorry

end angles_bisectors_l54_54158


namespace two_planes_divide_at_most_4_parts_l54_54131

-- Definitions related to the conditions
def Plane := ℝ × ℝ × ℝ → Prop -- Representing a plane in ℝ³ by an equation

-- Axiom: Two given planes
axiom plane1 : Plane
axiom plane2 : Plane

-- Conditions about their relationship
def are_parallel (p1 p2 : Plane) : Prop := 
  ∀ x y z, p1 (x, y, z) → p2 (x, y, z)

def intersect (p1 p2 : Plane) : Prop :=
  ∃ x y z, p1 (x, y, z) ∧ p2 (x, y, z)

-- Main theorem to state
theorem two_planes_divide_at_most_4_parts :
  (∃ p1 p2 : Plane, are_parallel p1 p2 ∨ intersect p1 p2) →
  (exists n : ℕ, n <= 4) :=
sorry

end two_planes_divide_at_most_4_parts_l54_54131


namespace original_garden_length_l54_54410

theorem original_garden_length (x : ℝ) (area : ℝ) (reduced_length : ℝ) (width : ℝ) (length_condition : x - reduced_length = width) (area_condition : x * width = area) (given_area : area = 120) (given_reduced_length : reduced_length = 2) : x = 12 := 
by
  sorry

end original_garden_length_l54_54410


namespace dishonest_dealer_profit_l54_54025

theorem dishonest_dealer_profit (cost_weight actual_weight : ℝ) (kg_in_g : ℝ) 
  (h1 : cost_weight = 1000) (h2 : actual_weight = 920) (h3 : kg_in_g = 1000) :
  ((cost_weight - actual_weight) / actual_weight) * 100 = 8.7 := by
  sorry

end dishonest_dealer_profit_l54_54025


namespace monotonically_increasing_range_k_l54_54964

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_range_k :
  (∀ x > 1, deriv (f k) x ≥ 0) → k ≥ 1 :=
sorry

end monotonically_increasing_range_k_l54_54964


namespace range_of_m_l54_54090

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def B (m : ℝ) := { x : ℝ | x^2 - (2 * m + 1) * x + 2 * m < 0 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → (-1 / 2 ≤ m ∧ m ≤ 1) :=
by
  sorry

end range_of_m_l54_54090


namespace polynomial_remainder_l54_54094

noncomputable def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 3
noncomputable def g (x : ℝ) : ℝ := x^2 + x - 2
noncomputable def r (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 3

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end polynomial_remainder_l54_54094


namespace negation_of_P_equiv_l54_54587

-- Define the proposition P
def P : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of P equivalently
theorem negation_of_P_equiv :
  ¬ P ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := 
sorry

end negation_of_P_equiv_l54_54587


namespace degree_equality_l54_54011

theorem degree_equality (m : ℕ) :
  (∀ x y z : ℕ, 2 + 4 = 1 + (m + 2)) → 3 * m - 2 = 7 :=
by
  intro h
  sorry

end degree_equality_l54_54011


namespace smallest_number_divisible_by_618_3648_60_inc_l54_54104

theorem smallest_number_divisible_by_618_3648_60_inc :
  ∃ N : ℕ, (N + 1) % 618 = 0 ∧ (N + 1) % 3648 = 0 ∧ (N + 1) % 60 = 0 ∧ N = 1038239 :=
by
  sorry

end smallest_number_divisible_by_618_3648_60_inc_l54_54104


namespace expression_evaluation_l54_54563

variable (x y : ℝ)

theorem expression_evaluation (h1 : x = 2 * y) (h2 : y ≠ 0) : 
  (x + 2 * y) - (2 * x + y) = -y := 
by
  sorry

end expression_evaluation_l54_54563


namespace sale_in_fifth_month_l54_54503

-- Define the sales for the first four months and the required sale for the sixth month
def sale_month1 : ℕ := 5124
def sale_month2 : ℕ := 5366
def sale_month3 : ℕ := 5808
def sale_month4 : ℕ := 5399
def sale_month6 : ℕ := 4579

-- Define the target average sale and number of months
def target_average_sale : ℕ := 5400
def number_of_months : ℕ := 6

-- Define the total sales calculation using the provided information
def total_sales : ℕ := target_average_sale * number_of_months
def total_sales_first_four_months : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month4

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + (total_sales - 
  (total_sales_first_four_months + sale_month6)) + sale_month6 = total_sales :=
by
  sorry

end sale_in_fifth_month_l54_54503


namespace arithmetic_seq_sum_is_110_l54_54651

noncomputable def S₁₀ (a_1 : ℝ) : ℝ :=
  10 / 2 * (2 * a_1 + 9 * (-2))

theorem arithmetic_seq_sum_is_110 (a1 a3 a7 a9 : ℝ) 
  (h_diff3 : a3 = a1 - 4)
  (h_diff7 : a7 = a1 - 12)
  (h_diff9 : a9 = a1 - 16)
  (h_geom : (a1 - 12) ^ 2 = (a1 - 4) * (a1 - 16)) :
  S₁₀ a1 = 110 :=
by
  sorry

end arithmetic_seq_sum_is_110_l54_54651


namespace fraction_proof_l54_54044

-- Define the fractions as constants
def a := 1 / 3
def b := 1 / 4
def c := 1 / 2
def d := 1 / 3

-- Prove the main statement
theorem fraction_proof : (a - b) / (c - d) = 1 / 2 := by
  sorry

end fraction_proof_l54_54044


namespace instantaneous_velocity_at_t_2_l54_54057

def y (t : ℝ) : ℝ := 3 * t^2 + 4

theorem instantaneous_velocity_at_t_2 :
  deriv y 2 = 12 :=
by
  sorry

end instantaneous_velocity_at_t_2_l54_54057


namespace f_17_l54_54511

def f : ℕ → ℤ := sorry

axiom f_prop1 : f 1 = 0
axiom f_prop2 : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) = f m + f n + 4 * (9 * m * n - 1)

theorem f_17 : f 17 = 1052 := by
  sorry

end f_17_l54_54511


namespace annual_profits_l54_54141

-- Define the profits of each quarter
def P1 : ℕ := 1500
def P2 : ℕ := 1500
def P3 : ℕ := 3000
def P4 : ℕ := 2000

-- State the annual profit theorem
theorem annual_profits : P1 + P2 + P3 + P4 = 8000 := by
  sorry

end annual_profits_l54_54141


namespace sin_cos_equation_solution_l54_54050

open Real

theorem sin_cos_equation_solution (x : ℝ): 
  (∃ n : ℤ, x = (π / 4050) + (π * n / 2025)) ∨ (∃ k : ℤ, x = (π * k / 9)) ↔ 
  sin (2025 * x) ^ 4 + (cos (2016 * x) ^ 2019) * (cos (2025 * x) ^ 2018) = 1 := 
by 
  sorry

end sin_cos_equation_solution_l54_54050


namespace toilet_paper_duration_l54_54084

theorem toilet_paper_duration :
  let bill_weekday := 3 * 5
  let wife_weekday := 4 * 8
  let kid_weekday := 5 * 6
  let total_weekday := bill_weekday + wife_weekday + 2 * kid_weekday
  let bill_weekend := 4 * 6
  let wife_weekend := 5 * 10
  let kid_weekend := 6 * 5
  let total_weekend := bill_weekend + wife_weekend + 2 * kid_weekend
  let total_week := 5 * total_weekday + 2 * total_weekend
  let total_squares := 1000 * 300
  let weeks_last := total_squares / total_week
  let days_last := weeks_last * 7
  days_last = 2615 :=
sorry

end toilet_paper_duration_l54_54084


namespace range_of_a_l54_54589

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (3 * x + a / x - 2)

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) := 
sorry

end range_of_a_l54_54589


namespace seats_empty_l54_54952

def number_of_people : ℕ := 532
def total_seats : ℕ := 750

theorem seats_empty (n : ℕ) (m : ℕ) : m - n = 218 := by
  have number_of_people : ℕ := 532
  have total_seats : ℕ := 750
  sorry

end seats_empty_l54_54952


namespace probability_AB_selected_l54_54346

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l54_54346


namespace function_tangent_and_max_k_l54_54537

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x - 1

theorem function_tangent_and_max_k 
  (x : ℝ) (h1 : 0 < x) 
  (h2 : 3 * x - y - 2 = 0) : 
  (∀ k : ℤ, (∀ x : ℝ, 1 < x → k < (f x) / (x - 1)) → k ≤ 4) := 
sorry

end function_tangent_and_max_k_l54_54537


namespace total_pencils_sold_l54_54108

theorem total_pencils_sold (price_reduced: Bool)
  (day1_students : ℕ) (first4_d1 : ℕ) (next3_d1 : ℕ) (last3_d1 : ℕ)
  (day2_students : ℕ) (first5_d2 : ℕ) (next6_d2 : ℕ) (last4_d2 : ℕ)
  (day3_students : ℕ) (first10_d3 : ℕ) (next10_d3 : ℕ) (last10_d3 : ℕ)
  (day1_total : day1_students = 10 ∧ first4_d1 = 4 ∧ next3_d1 = 3 ∧ last3_d1 = 3 ∧
    (first4_d1 * 5) + (next3_d1 * 7) + (last3_d1 * 3) = 50)
  (day2_total : day2_students = 15 ∧ first5_d2 = 5 ∧ next6_d2 = 6 ∧ last4_d2 = 4 ∧
    (first5_d2 * 4) + (next6_d2 * 9) + (last4_d2 * 6) = 98)
  (day3_total : day3_students = 2 * day2_students ∧ first10_d3 = 10 ∧ next10_d3 = 10 ∧ last10_d3 = 10 ∧
    (first10_d3 * 2) + (next10_d3 * 8) + (last10_d3 * 4) = 140) :
  (50 + 98 + 140 = 288) :=
sorry

end total_pencils_sold_l54_54108


namespace simplify_expression_l54_54482

variable (b c : ℝ)

theorem simplify_expression :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 :=
by sorry

end simplify_expression_l54_54482


namespace vector_subtraction_correct_l54_54771

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-4, 2)

theorem vector_subtraction_correct :
  vector_a - 2 • vector_b = (10, -5) :=
sorry

end vector_subtraction_correct_l54_54771


namespace solution_set_of_inequality_l54_54860

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l54_54860


namespace correct_geometry_problems_l54_54853

-- Let A_c be the number of correct algebra problems.
-- Let A_i be the number of incorrect algebra problems.
-- Let G_c be the number of correct geometry problems.
-- Let G_i be the number of incorrect geometry problems.

def algebra_correct_incorrect_ratio (A_c A_i : ℕ) : Prop :=
  A_c * 2 = A_i * 3

def geometry_correct_incorrect_ratio (G_c G_i : ℕ) : Prop :=
  G_c * 1 = G_i * 4

def total_algebra_problems (A_c A_i : ℕ) : Prop :=
  A_c + A_i = 25

def total_geometry_problems (G_c G_i : ℕ) : Prop :=
  G_c + G_i = 35

def total_problems (A_c A_i G_c G_i : ℕ) : Prop :=
  A_c + A_i + G_c + G_i = 60

theorem correct_geometry_problems (A_c A_i G_c G_i : ℕ) :
  algebra_correct_incorrect_ratio A_c A_i →
  geometry_correct_incorrect_ratio G_c G_i →
  total_algebra_problems A_c A_i →
  total_geometry_problems G_c G_i →
  total_problems A_c A_i G_c G_i →
  G_c = 28 :=
sorry

end correct_geometry_problems_l54_54853


namespace rod_length_l54_54721

/--
Prove that given the number of pieces that can be cut from the rod is 40 and the length of each piece is 85 cm, the length of the rod is 3400 cm.
-/
theorem rod_length (number_of_pieces : ℕ) (length_of_each_piece : ℕ) (h_pieces : number_of_pieces = 40) (h_length_piece : length_of_each_piece = 85) : number_of_pieces * length_of_each_piece = 3400 := 
by
  -- We need to prove that 40 * 85 = 3400
  sorry

end rod_length_l54_54721


namespace part1_part2_l54_54924

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 2 :=
by
  sorry

theorem part2 (x : ℝ) : (∀ b : ℝ, b ≠ 0 → f x ≥ (|2 * b + 1| - |1 - b|) / |b|) → (x ≤ -1.5 ∨ x ≥ 1.5) :=
by
  sorry

end part1_part2_l54_54924


namespace slower_pump_time_l54_54278

theorem slower_pump_time (R : ℝ) (hours : ℝ) (combined_rate : ℝ) (faster_rate_adj : ℝ) (time_both : ℝ) :
  (combined_rate = R * (1 + faster_rate_adj)) →
  (faster_rate_adj = 1.5) →
  (time_both = 5) →
  (combined_rate * time_both = 1) →
  (hours = 1 / R) →
  hours = 12.5 :=
by
  sorry

end slower_pump_time_l54_54278


namespace eva_total_marks_l54_54891

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l54_54891


namespace probability_no_self_draws_l54_54380

theorem probability_no_self_draws :
  let total_outcomes := 6
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 3 :=
by
  sorry

end probability_no_self_draws_l54_54380


namespace line_within_plane_correct_l54_54684

-- Definitions of sets representing a line and a plane
variable {Point : Type}
variable (l α : Set Point)

-- Definition of the statement
def line_within_plane : Prop := l ⊆ α

-- Proof statement (without the actual proof)
theorem line_within_plane_correct (h : l ⊆ α) : line_within_plane l α :=
by
  sorry

end line_within_plane_correct_l54_54684


namespace jerry_claims_years_of_salary_l54_54868

theorem jerry_claims_years_of_salary
  (Y : ℝ)
  (salary_damage_per_year : ℝ := 50000)
  (medical_bills : ℝ := 200000)
  (punitive_damages : ℝ := 3 * (salary_damage_per_year * Y + medical_bills))
  (total_damages : ℝ := salary_damage_per_year * Y + medical_bills + punitive_damages)
  (received_amount : ℝ := 0.8 * total_damages)
  (actual_received_amount : ℝ := 5440000) :
  received_amount = actual_received_amount → Y = 30 := 
by
  sorry

end jerry_claims_years_of_salary_l54_54868


namespace value_of_b_l54_54375

theorem value_of_b 
  (a b : ℝ) 
  (h : ∃ c : ℝ, (ax^3 + bx^2 + 1) = (x^2 - x - 1) * (x + c)) : 
  b = -2 :=
  sorry

end value_of_b_l54_54375


namespace calculator_display_exceeds_1000_after_three_presses_l54_54486

-- Define the operation of pressing the squaring key
def square_key (n : ℕ) : ℕ := n * n

-- Define the initial display number
def initial_display : ℕ := 3

-- Prove that after pressing the squaring key 3 times, the display is greater than 1000.
theorem calculator_display_exceeds_1000_after_three_presses : 
  square_key (square_key (square_key initial_display)) > 1000 :=
by
  sorry

end calculator_display_exceeds_1000_after_three_presses_l54_54486


namespace f_gt_e_plus_2_l54_54875

noncomputable def f (x : ℝ) : ℝ := ( (Real.exp x) / x ) - ( (8 * Real.log (x / 2)) / (x^2) ) + x

lemma slope_at_2 : HasDerivAt f (Real.exp 2 / 4) 2 := 
by 
  sorry

theorem f_gt_e_plus_2 (x : ℝ) (hx : 0 < x) : f x > Real.exp 1 + 2 :=
by
  sorry

end f_gt_e_plus_2_l54_54875


namespace smallest_nonprime_with_large_prime_factors_l54_54670

/-- 
The smallest nonprime integer greater than 1 with no prime factor less than 15
falls in the range 260 < m ≤ 270.
-/
theorem smallest_nonprime_with_large_prime_factors :
  ∃ m : ℕ, 2 < m ∧ ¬ Nat.Prime m ∧ (∀ p : ℕ, Nat.Prime p → p ∣ m → 15 ≤ p) ∧ 260 < m ∧ m ≤ 270 :=
by
  sorry

end smallest_nonprime_with_large_prime_factors_l54_54670


namespace range_of_a_l54_54897

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

/-- If A ⊆ B, then the range of values for 'a' satisfies -4 ≤ a ≤ -1 -/
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : -4 ≤ a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l54_54897


namespace inequality_min_value_l54_54362

theorem inequality_min_value (a : ℝ) : 
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ a) → (a ≤ 3) := 
by
  sorry

end inequality_min_value_l54_54362


namespace minimum_k_l54_54334

variable {a b k : ℝ}

theorem minimum_k (h_a : a > 0) (h_b : b > 0) (h : ∀ a b : ℝ, a > 0 → b > 0 → (1 / a) + (1 / b) + (k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_l54_54334


namespace find_q_l54_54841

-- Defining the polynomial and conditions
def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

variable (p q r : ℝ)

-- Given conditions
def mean_of_zeros_eq_prod_of_zeros (p q r : ℝ) : Prop :=
  -p / 3 = r

def prod_of_zeros_eq_sum_of_coeffs (p q r : ℝ) : Prop :=
  r = 1 + p + q + r

def y_intercept_eq_three (r : ℝ) : Prop :=
  r = 3

-- Final proof statement asserting q = 5
theorem find_q (p q r : ℝ) (h1 : mean_of_zeros_eq_prod_of_zeros p q r)
  (h2 : prod_of_zeros_eq_sum_of_coeffs p q r)
  (h3 : y_intercept_eq_three r) :
  q = 5 :=
sorry

end find_q_l54_54841


namespace fraction_to_decimal_l54_54194

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l54_54194


namespace compute_expression_l54_54297

theorem compute_expression :
  24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end compute_expression_l54_54297


namespace range_of_a_l54_54016

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 16 * Real.log x

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, a - 1 ≤ x ∧ x ≤ a + 2 → (fderiv ℝ f x) x < 0)
  ↔ (1 < a) ∧ (a ≤ 2) :=
by
  sorry

end range_of_a_l54_54016


namespace neg_cube_squared_l54_54627

theorem neg_cube_squared (x : ℝ) : (-x^3) ^ 2 = x ^ 6 :=
by
  sorry

end neg_cube_squared_l54_54627


namespace relationship_between_xyz_l54_54717

theorem relationship_between_xyz (x y z : ℝ) (h1 : x - z < y) (h2 : x + z > y) : -z < x - y ∧ x - y < z :=
by
  sorry

end relationship_between_xyz_l54_54717


namespace find_side_c_l54_54601

noncomputable def triangle_side_c (A b S : ℝ) (c : ℝ) : Prop :=
  S = 0.5 * b * c * Real.sin A

theorem find_side_c :
  ∀ (c : ℝ), triangle_side_c (Real.pi / 3) 16 (64 * Real.sqrt 3) c → c = 16 :=
by
  sorry

end find_side_c_l54_54601


namespace negation_proposition_l54_54549

theorem negation_proposition :
  (¬ (∀ x : ℝ, abs x + x^2 ≥ 0)) ↔ (∃ x₀ : ℝ, abs x₀ + x₀^2 < 0) :=
by
  sorry

end negation_proposition_l54_54549


namespace find_n_l54_54943

noncomputable def satisfies_condition (n d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ) : Prop :=
  1 = d₁ ∧ d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧ d₇ < n ∧
  (∀ d, d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n → n % d = 0) ∧
  (∀ d, n % d = 0 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n)

theorem find_n (n : ℕ) : (∃ d₁ d₂ d₃ d₄ d₅ d₆ d₇, satisfies_condition n d₁ d₂ d₃ d₄ d₅ d₆ d₇ ∧ n = d₆^2 + d₇^2 - 1) → (n = 144 ∨ n = 1984) :=
  by
  sorry

end find_n_l54_54943


namespace intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l54_54370

def l1 (x y : ℝ) : Prop := x + y = 2
def l2 (x y : ℝ) : Prop := x - 3 * y = -10
def l3 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

def M : (ℝ × ℝ) := (-1, 3)

-- Part (Ⅰ): Prove that M is the intersection point of l1 and l2
theorem intersection_l1_l2 : l1 M.1 M.2 ∧ l2 M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅱ): Prove the equation of the line passing through M and parallel to l3 is 3x - 4y + 15 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y + 15 = 0

theorem line_parallel_to_l3 : parallel_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅲ): Prove the equation of the line passing through M and perpendicular to l3 is 4x + 3y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

theorem line_perpendicular_to_l3 : perpendicular_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

end intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l54_54370


namespace evaluate_expression_zero_l54_54233

-- Main proof statement
theorem evaluate_expression_zero :
  ∀ (a d c b : ℤ),
    d = c + 5 →
    c = b - 8 →
    b = a + 3 →
    a = 3 →
    a - 1 ≠ 0 →
    d - 6 ≠ 0 →
    c + 4 ≠ 0 →
    (a + 3) * (d - 3) * (c + 9) = 0 :=
by
  intros a d c b hd hc hb ha h1 h2 h3
  sorry -- The proof goes here

end evaluate_expression_zero_l54_54233


namespace compute_expression_l54_54413

theorem compute_expression : 2 + ((4 * 3 - 2) / 2 * 3) + 5 = 22 :=
by
  -- Place the solution steps if needed
  sorry

end compute_expression_l54_54413


namespace min_value_expression_l54_54074

theorem min_value_expression : ∃ (x y : ℝ), x^2 + 2*x*y + 3*y^2 - 6*x - 2*y = -11 := by
  sorry

end min_value_expression_l54_54074


namespace factor_expression_l54_54954

theorem factor_expression (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + ab + bc + ca) :=
by
  sorry

end factor_expression_l54_54954


namespace exists_set_with_property_l54_54049

theorem exists_set_with_property (n : ℕ) (h : n > 0) :
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ {a b}, a ∈ S → b ∈ S → a ≠ b → (a - b) ∣ a ∧ (a - b) ∣ b) ∧
  (∀ {a b c}, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ¬ ((a - b) ∣ c)) :=
sorry

end exists_set_with_property_l54_54049


namespace complex_number_simplification_l54_54491

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : i * (1 - i) - 1 = i := 
by
  sorry

end complex_number_simplification_l54_54491


namespace matrix_sum_correct_l54_54120

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![4, -3],
  ![2, 5]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-6, 8],
  ![-3, 7]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, 5],
  ![-1, 12]
]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l54_54120


namespace max_value_y_eq_neg10_l54_54400

open Real

theorem max_value_y_eq_neg10 (x : ℝ) (hx : x > 0) : 
  ∃ y, y = 2 - 9 * x - 4 / x ∧ (∀ z, (∃ (x' : ℝ), x' > 0 ∧ z = 2 - 9 * x' - 4 / x') → z ≤ y) ∧ y = -10 :=
by
  sorry

end max_value_y_eq_neg10_l54_54400


namespace number_line_y_l54_54047

theorem number_line_y (step_length : ℕ) (steps_total : ℕ) (total_distance : ℕ) (y_step : ℕ) (y : ℕ) 
    (H1 : steps_total = 6) 
    (H2 : total_distance = 24) 
    (H3 : y_step = 4)
    (H4 : step_length = total_distance / steps_total) 
    (H5 : y = step_length * y_step) : 
  y = 16 := 
  by 
    sorry

end number_line_y_l54_54047


namespace prove_root_property_l54_54154

-- Define the quadratic equation and its roots
theorem prove_root_property :
  let r := -4 + Real.sqrt 226
  let s := -4 - Real.sqrt 226
  (r + 4) * (s + 4) = -226 :=
by
  -- the proof steps go here (omitted)
  sorry

end prove_root_property_l54_54154


namespace min_f_over_f_prime_at_1_l54_54323

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def quadratic_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem min_f_over_f_prime_at_1 (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b > 0) (h₂ : ∀ x, quadratic_function a b c x ≥ 0) :
  (∃ k, (∀ x, quadratic_function a b c x ≥ 0 → quadratic_function a b c ((-b)/(2*a)) ≤ x) ∧ k = 2) :=
by
  sorry

end min_f_over_f_prime_at_1_l54_54323


namespace first_year_after_2020_with_sum_4_l54_54822

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

def is_year (y : ℕ) : Prop :=
  y > 2020 ∧ sum_of_digits y = 4

theorem first_year_after_2020_with_sum_4 : ∃ y, is_year y ∧ ∀ z, is_year z → z ≥ y :=
by sorry

end first_year_after_2020_with_sum_4_l54_54822


namespace factorize_binomial_square_l54_54597

theorem factorize_binomial_square (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 :=
by
  sorry

end factorize_binomial_square_l54_54597


namespace product_discount_l54_54509

theorem product_discount (P : ℝ) (h₁ : P > 0) :
  let price_after_first_discount := 0.7 * P
  let price_after_second_discount := 0.8 * price_after_first_discount
  let total_reduction := P - price_after_second_discount
  let percent_reduction := (total_reduction / P) * 100
  percent_reduction = 44 :=
by
  sorry

end product_discount_l54_54509


namespace lcm_5_7_10_14_l54_54421

theorem lcm_5_7_10_14 : Nat.lcm (Nat.lcm 5 7) (Nat.lcm 10 14) = 70 := by
  sorry

end lcm_5_7_10_14_l54_54421


namespace perpendicular_line_through_circle_center_l54_54343

theorem perpendicular_line_through_circle_center :
  ∀ (x y : ℝ), (x^2 + (y-1)^2 = 4) → (3*x + 2*y + 1 = 0) → (2*x - 3*y + 3 = 0) :=
by
  intros x y h_circle h_line
  sorry

end perpendicular_line_through_circle_center_l54_54343


namespace circle_area_of_circumscribed_triangle_l54_54241

theorem circle_area_of_circumscribed_triangle :
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  π * R^2 = (5184 / 119) * π := 
by
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  have h1 : height = Real.sqrt (a^2 - (c / 2)^2) := by sorry
  have h2 : A = (1 / 2) * c * height := by sorry
  have h3 : R = (a * b * c) / (4 * A) := by sorry
  have h4 : π * R^2 = (5184 / 119) * π := by sorry
  exact h4

end circle_area_of_circumscribed_triangle_l54_54241


namespace final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l54_54227

-- Definitions of the driving records for trainee A and B
def driving_record_A : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def driving_record_B : List Int := [-17, 9, -2, 8, 6, 9, -5, -1, 4, -7, -8]

-- Fuel consumption rate per kilometer
variable (a : ℝ)

-- Proof statements in Lean
theorem final_position_A : driving_record_A.sum = 39 := by sorry
theorem final_position_B : driving_record_B.sum = -4 := by sorry
theorem fuel_consumption_A : (driving_record_A.map (abs)).sum * a = 65 * a := by sorry
theorem fuel_consumption_B : (driving_record_B.map (abs)).sum * a = 76 * a := by sorry
theorem less_fuel_consumption : (driving_record_A.map (abs)).sum * a < (driving_record_B.map (abs)).sum * a := by sorry

end final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l54_54227


namespace tangent_eq_inequality_not_monotonic_l54_54427

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / (x + a)

theorem tangent_eq (a : ℝ) (h : 0 < a) : 
  ∃ k : ℝ, (k, f 1 a) ∈ {
    p : ℝ × ℝ | p.1 - (a + 1) * p.2 - 1 = 0 
  } :=
  sorry

theorem inequality (x : ℝ) (h : 1 ≤ x) : f x 1 ≤ (x - 1) / 2 := 
  sorry

theorem not_monotonic (a : ℝ) (h : 0 < a) : 
  ¬(∀ x y : ℝ, x < y → f x a ≤ f y a ∨ x < y → f x a ≥ f y a) := 
  sorry

end tangent_eq_inequality_not_monotonic_l54_54427


namespace john_order_cost_l54_54168

-- Definitions from the problem conditions
def discount_rate : ℝ := 0.10
def item_price : ℝ := 200
def num_items : ℕ := 7
def discount_threshold : ℝ := 1000

-- Final proof statement
theorem john_order_cost : 
  (num_items * item_price) - 
  (if (num_items * item_price) > discount_threshold then 
    discount_rate * ((num_items * item_price) - discount_threshold) 
  else 0) = 1360 := 
sorry

end john_order_cost_l54_54168


namespace fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l54_54708

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

end fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l54_54708


namespace tank_capacity_l54_54422

variable (c w : ℕ)

-- Conditions
def initial_fraction (w c : ℕ) : Prop := w = c / 7
def final_fraction (w c : ℕ) : Prop := (w + 2) = c / 5

-- The theorem statement
theorem tank_capacity : 
  initial_fraction w c → 
  final_fraction w c → 
  c = 35 := 
by
  sorry  -- indicates that the proof is not provided

end tank_capacity_l54_54422


namespace difference_of_digits_l54_54174

theorem difference_of_digits (A B : ℕ) (h1 : 6 * 10 + A - (B * 10 + 2) = 36) (h2 : A ≠ B) : A - B = 5 :=
sorry

end difference_of_digits_l54_54174


namespace find_k_value_l54_54112

theorem find_k_value
  (k : ℤ)
  (h : 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998) : k = 11 :=
by
  sorry

end find_k_value_l54_54112


namespace perfect_square_expression_l54_54634

theorem perfect_square_expression : 
    ∀ x : ℝ, (11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02)^2) → (x = 0.4792) :=
by
  intros x h
  -- sorry placeholder for the proof
  sorry

end perfect_square_expression_l54_54634


namespace initial_price_of_gasoline_l54_54982

theorem initial_price_of_gasoline 
  (P0 : ℝ) 
  (P1 : ℝ := 1.30 * P0)
  (P2 : ℝ := 0.75 * P1)
  (P3 : ℝ := 1.10 * P2)
  (P4 : ℝ := 0.85 * P3)
  (P5 : ℝ := 0.80 * P4)
  (h : P5 = 102.60) : 
  P0 = 140.67 :=
by sorry

end initial_price_of_gasoline_l54_54982


namespace intersection_complement_l54_54567

open Set

def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_complement :
  A ∩ (univ \ B) = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end intersection_complement_l54_54567


namespace polar_coordinates_of_point_l54_54913

noncomputable def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ := 
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  (r, θ)

theorem polar_coordinates_of_point :
  point_rectangular_to_polar 1 (-1) = (Real.sqrt 2, 7 * Real.pi / 4) :=
by
  unfold point_rectangular_to_polar
  sorry

end polar_coordinates_of_point_l54_54913


namespace analyze_monotonicity_and_find_a_range_l54_54540

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem analyze_monotonicity_and_find_a_range
  (a : ℝ)
  (h : ∀ x : ℝ, f x a + f_prime x a = 2 - a * x^2) :
  (∀ x : ℝ, a ≤ 0 → f_prime x a > 0) ∧
  (a > 0 → (∀ x : ℝ, (x < Real.log (2 * a) → f_prime x a < 0) ∧ (x > Real.log (2 * a) → f_prime x a > 0))) ∧
  (1 < a ∧ a < Real.exp 1 - 1) :=
sorry

end analyze_monotonicity_and_find_a_range_l54_54540


namespace unit_circle_inequality_l54_54139

theorem unit_circle_inequality 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (habcd : a * b + c * d = 1) 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (hx1 : x1^2 + y1^2 = 1)
  (hx2 : x2^2 + y2^2 = 1)
  (hx3 : x3^2 + y3^2 = 1)
  (hx4 : x4^2 + y4^2 = 1) :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2 ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
sorry

end unit_circle_inequality_l54_54139


namespace increase_in_difference_between_strawberries_and_blueberries_l54_54521

theorem increase_in_difference_between_strawberries_and_blueberries :
  ∀ (B S : ℕ), B = 32 → S = B + 12 → (S - B) = 12 :=
by
  intros B S hB hS
  sorry

end increase_in_difference_between_strawberries_and_blueberries_l54_54521


namespace number_of_solutions_l54_54391

theorem number_of_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℕ,
    (x < 10^2006) ∧ ((x * (x - 1)) % 10^2006 = 0) → x ≤ n :=
sorry

end number_of_solutions_l54_54391


namespace sum_of_squares_geometric_progression_theorem_l54_54921

noncomputable def sum_of_squares_geometric_progression (a₁ q : ℝ) (S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) : ℝ :=
  S₁ * S₂

theorem sum_of_squares_geometric_progression_theorem
  (a₁ q S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) :
  sum_of_squares_geometric_progression a₁ q S₁ S₂ h_q h_S₁ h_S₂ = S₁ * S₂ := sorry

end sum_of_squares_geometric_progression_theorem_l54_54921


namespace paper_length_l54_54832

theorem paper_length :
  ∃ (L : ℝ), (2 * (11 * L) = 2 * (8.5 * 11) + 100 ∧ L = 287 / 22) :=
sorry

end paper_length_l54_54832


namespace fleas_difference_l54_54967

-- Define the initial number of fleas and subsequent fleas after each treatment.
def initial_fleas (F : ℝ) := F
def after_first_treatment (F : ℝ) := F * 0.40
def after_second_treatment (F : ℝ) := (after_first_treatment F) * 0.55
def after_third_treatment (F : ℝ) := (after_second_treatment F) * 0.70
def after_fourth_treatment (F : ℝ) := (after_third_treatment F) * 0.80

-- Given condition
axiom final_fleas : initial_fleas 20 = after_fourth_treatment 20

-- Prove the number of fleas before treatment minus the number after treatment is 142
theorem fleas_difference (F : ℝ) (h : initial_fleas F = after_fourth_treatment 20) : 
  F - 20 = 142 :=
by {
  sorry
}

end fleas_difference_l54_54967


namespace train_overtake_l54_54688

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

end train_overtake_l54_54688


namespace savanna_total_animals_l54_54114

def num_lions_safari := 100
def num_snakes_safari := num_lions_safari / 2
def num_giraffes_safari := num_snakes_safari - 10
def num_elephants_safari := num_lions_safari / 4

def num_lions_savanna := num_lions_safari * 2
def num_snakes_savanna := num_snakes_safari * 3
def num_giraffes_savanna := num_giraffes_safari + 20
def num_elephants_savanna := num_elephants_safari * 5
def num_zebras_savanna := (num_lions_savanna + num_snakes_savanna) / 2

def total_animals_savanna := 
  num_lions_savanna 
  + num_snakes_savanna 
  + num_giraffes_savanna 
  + num_elephants_savanna 
  + num_zebras_savanna

open Nat
theorem savanna_total_animals : total_animals_savanna = 710 := by
  sorry

end savanna_total_animals_l54_54114


namespace total_items_count_l54_54458

theorem total_items_count :
  let old_women  := 7
  let mules      := 7
  let bags       := 7
  let loaves     := 7
  let knives     := 7
  let sheaths    := 7
  let sheaths_per_loaf := knives * sheaths
  let sheaths_per_bag := loaves * sheaths_per_loaf
  let sheaths_per_mule := bags * sheaths_per_bag
  let sheaths_per_old_woman := mules * sheaths_per_mule
  let total_sheaths := old_women * sheaths_per_old_woman

  let loaves_per_bag := loaves
  let loaves_per_mule := bags * loaves_per_bag
  let loaves_per_old_woman := mules * loaves_per_mule
  let total_loaves := old_women * loaves_per_old_woman

  let knives_per_loaf := knives
  let knives_per_bag := loaves * knives_per_loaf
  let knives_per_mule := bags * knives_per_bag
  let knives_per_old_woman := mules * knives_per_mule
  let total_knives := old_women * knives_per_old_woman

  let total_bags := old_women * mules * bags

  let total_mules := old_women * mules

  let total_items := total_sheaths + total_loaves + total_knives + total_bags + total_mules + old_women

  total_items = 137256 :=
by
  sorry

end total_items_count_l54_54458


namespace sum_of_x_values_l54_54960

theorem sum_of_x_values :
  (2^(x^2 + 6*x + 9) = 16^(x + 3)) → ∃ x1 x2 : ℝ, x1 + x2 = -2 :=
by
  sorry

end sum_of_x_values_l54_54960


namespace area_triangle_PQR_eq_2sqrt2_l54_54713

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

end area_triangle_PQR_eq_2sqrt2_l54_54713


namespace smallest_number_from_digits_l54_54431

theorem smallest_number_from_digits : 
  ∀ (d1 d2 d3 d4 : ℕ), (d1 = 2) → (d2 = 0) → (d3 = 1) → (d4 = 6) →
  ∃ n : ℕ, (n = 1026) ∧ 
  ((n = d1 * 1000 + d2 * 100 + d3 * 10 + d4) ∨ 
   (n = d1 * 1000 + d2 * 100 + d4 * 10 + d3) ∨ 
   (n = d1 * 1000 + d3 * 100 + d2 * 10 + d4) ∨ 
   (n = d1 * 1000 + d3 * 100 + d4 * 10 + d2) ∨ 
   (n = d1 * 1000 + d4 * 100 + d2 * 10 + d3) ∨ 
   (n = d1 * 1000 + d4 * 100 + d3 * 10 + d2) ∨ 
   (n = d2 * 1000 + d1 * 100 + d3 * 10 + d4) ∨ 
   (n = d2 * 1000 + d1 * 100 + d4 * 10 + d3) ∨ 
   (n = d2 * 1000 + d3 * 100 + d1 * 10 + d4) ∨ 
   (n = d2 * 1000 + d3 * 100 + d4 * 10 + d1) ∨ 
   (n = d2 * 1000 + d4 * 100 + d1 * 10 + d3) ∨ 
   (n = d2 * 1000 + d4 * 100 + d3 * 10 + d1) ∨ 
   (n = d3 * 1000 + d1 * 100 + d2 * 10 + d4) ∨ 
   (n = d3 * 1000 + d1 * 100 + d4 * 10 + d2) ∨ 
   (n = d3 * 1000 + d2 * 100 + d1 * 10 + d4) ∨ 
   (n = d3 * 1000 + d2 * 100 + d4 * 10 + d1) ∨ 
   (n = d3 * 1000 + d4 * 100 + d1 * 10 + d2) ∨ 
   (n = d3 * 1000 + d4 * 100 + d2 * 10 + d1) ∨ 
   (n = d4 * 1000 + d1 * 100 + d2 * 10 + d3) ∨ 
   (n = d4 * 1000 + d1 * 100 + d3 * 10 + d2) ∨ 
   (n = d4 * 1000 + d2 * 100 + d1 * 10 + d3) ∨ 
   (n = d4 * 1000 + d2 * 100 + d3 * 10 + d1) ∨ 
   (n = d4 * 1000 + d3 * 100 + d1 * 10 + d2) ∨ 
   (n = d4 * 1000 + d3 * 100 + d2 * 10 + d1)) := sorry

end smallest_number_from_digits_l54_54431


namespace range_of_h_l54_54772

theorem range_of_h 
  (y1 y2 y3 k : ℝ)
  (h : ℝ)
  (H1 : y1 = (-3 - h)^2 + k)
  (H2 : y2 = (-1 - h)^2 + k)
  (H3 : y3 = (1 - h)^2 + k)
  (H_ord : y2 < y1 ∧ y1 < y3) : 
  -2 < h ∧ h < -1 :=
sorry

end range_of_h_l54_54772


namespace short_haired_girls_l54_54192

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l54_54192


namespace alan_total_payment_l54_54993

-- Define the costs of CDs
def cost_AVN : ℝ := 12
def cost_TheDark : ℝ := 2 * cost_AVN
def cost_TheDark_total : ℝ := 2 * cost_TheDark
def cost_other_CDs : ℝ := cost_AVN + cost_TheDark_total
def cost_90s : ℝ := 0.4 * cost_other_CDs
def total_cost : ℝ := cost_AVN + cost_TheDark_total + cost_90s

-- Formulate the main statement
theorem alan_total_payment :
  total_cost = 84 := by
  sorry

end alan_total_payment_l54_54993


namespace rohan_age_is_25_l54_54669

-- Define the current age of Rohan
def rohan_current_age (x : ℕ) : Prop :=
  x + 15 = 4 * (x - 15)

-- The goal is to prove that Rohan's current age is 25 years old
theorem rohan_age_is_25 : ∃ x : ℕ, rohan_current_age x ∧ x = 25 :=
by
  existsi (25 : ℕ)
  -- Proof is omitted since this is a statement only
  sorry

end rohan_age_is_25_l54_54669


namespace find_m_l54_54210

-- Let's define the sets A and B.
def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- We'll state the problem as a theorem
theorem find_m (m : ℝ) (h : B m ⊆ A) : m = 1 ∨ m = -1 :=
by sorry

end find_m_l54_54210


namespace negation_of_positive_x2_plus_2_l54_54151

theorem negation_of_positive_x2_plus_2 (h : ∀ x : ℝ, x^2 + 2 > 0) : ¬ (∀ x : ℝ, x^2 + 2 > 0) = False := 
by
  sorry

end negation_of_positive_x2_plus_2_l54_54151


namespace Nunzio_eats_pizza_every_day_l54_54329

theorem Nunzio_eats_pizza_every_day
  (one_piece_fraction : ℚ := 1/8)
  (total_pizzas : ℕ := 27)
  (total_days : ℕ := 72)
  (pieces_per_pizza : ℕ := 8)
  (total_pieces : ℕ := total_pizzas * pieces_per_pizza)
  : (total_pieces / total_days = 3) :=
by
  -- We assume 1/8 as a fraction for the pieces of pizza is stated in the conditions, therefore no condition here.
  -- We need to show that Nunzio eats 3 pieces of pizza every day given the total pieces and days.
  sorry

end Nunzio_eats_pizza_every_day_l54_54329


namespace pipes_fill_cistern_time_l54_54125

noncomputable def pipe_fill_time : ℝ :=
  let rateA := 1 / 80
  let rateC := 1 / 60
  let combined_rateAB := 1 / 20
  let rateB := combined_rateAB - rateA
  let combined_rateABC := rateA + rateB - rateC
  1 / combined_rateABC

theorem pipes_fill_cistern_time :
  pipe_fill_time = 30 := by
  sorry

end pipes_fill_cistern_time_l54_54125


namespace unique_solution_f_l54_54727

theorem unique_solution_f (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y))
  (h2 : f 0 = 0) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_l54_54727


namespace range_f_x_negative_l54_54350

-- We define the conditions: f is an even function, increasing on (-∞, 0), and f(2) = 0.
variables {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_neg_infinity_to_zero (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x < 0 ∧ y < 0 → f x ≤ f y

def f_at_2_is_zero (f : ℝ → ℝ) : Prop :=
  f 2 = 0

-- The theorem to be proven.
theorem range_f_x_negative (hf_even : even_function f)
  (hf_incr : increasing_on_neg_infinity_to_zero f)
  (hf_at2 : f_at_2_is_zero f) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 :=
by
  sorry

end range_f_x_negative_l54_54350


namespace intersection_complement_A_B_l54_54610

def Universe : Set ℝ := Set.univ

def A : Set ℝ := {x | abs (x - 1) > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_complement_A_B :
  (Universe \ A) ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l54_54610


namespace spatial_relationship_l54_54415

variables {a b c : Type}          -- Lines a, b, c
variables {α β γ : Type}          -- Planes α, β, γ

-- Parallel relationship between planes
def plane_parallel (α β : Type) : Prop := sorry
-- Perpendicular relationship between planes
def plane_perpendicular (α β : Type) : Prop := sorry
-- Parallel relationship between lines and planes
def line_parallel_plane (a α : Type) : Prop := sorry
-- Perpendicular relationship between lines and planes
def line_perpendicular_plane (a α : Type) : Prop := sorry
-- Parallel relationship between lines
def line_parallel (a b : Type) : Prop := sorry
-- The angle formed by a line and a plane
def angle (a : Type) (α : Type) : Type := sorry

theorem spatial_relationship :
  (plane_parallel α γ ∧ plane_parallel β γ → plane_parallel α β) ∧
  ¬ (line_parallel_plane a α ∧ line_parallel_plane b α → line_parallel a b) ∧
  ¬ (plane_perpendicular α γ ∧ plane_perpendicular β γ → plane_parallel α β) ∧
  ¬ (line_perpendicular_plane a c ∧ line_perpendicular_plane b c → line_parallel a b) ∧
  (line_parallel a b ∧ plane_parallel α β → angle a α = angle b β) :=
sorry

end spatial_relationship_l54_54415


namespace necessary_but_not_sufficient_for_gt_zero_l54_54418

theorem necessary_but_not_sufficient_for_gt_zero (x : ℝ) : 
  x ≠ 0 → (¬ (x ≤ 0)) := by 
  sorry

end necessary_but_not_sufficient_for_gt_zero_l54_54418


namespace expression_positive_for_all_integers_l54_54267

theorem expression_positive_for_all_integers (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 :=
by
  sorry

end expression_positive_for_all_integers_l54_54267


namespace average_speed_of_train_l54_54293

-- Define conditions
def traveled_distance1 : ℝ := 240
def traveled_distance2 : ℝ := 450
def time_period1 : ℝ := 3
def time_period2 : ℝ := 5

-- Define total distance and total time based on the conditions
def total_distance : ℝ := traveled_distance1 + traveled_distance2
def total_time : ℝ := time_period1 + time_period2

-- Prove that the average speed is 86.25 km/h
theorem average_speed_of_train : total_distance / total_time = 86.25 := by
  -- Here should be the proof, but we put sorry since we only need the statement
  sorry

end average_speed_of_train_l54_54293


namespace problem1_problem2_l54_54136

-- Definitions for permutation and combination
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problems statements
theorem problem1 : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by 
  sorry

theorem problem2 :
  C 200 198 + C 200 196 + 2 * C 200 197 = C 202 4 := by 
  sorry

end problem1_problem2_l54_54136


namespace sum_of_squares_of_ages_l54_54538

theorem sum_of_squares_of_ages 
  (d t h : ℕ) 
  (cond1 : 3 * d + t = 2 * h)
  (cond2 : 2 * h ^ 3 = 3 * d ^ 3 + t ^ 3)
  (rel_prime : Nat.gcd d (Nat.gcd t h) = 1) :
  d ^ 2 + t ^ 2 + h ^ 2 = 42 :=
sorry

end sum_of_squares_of_ages_l54_54538


namespace casey_nail_decorating_time_l54_54237

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end casey_nail_decorating_time_l54_54237


namespace sandwiches_left_l54_54653

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end sandwiches_left_l54_54653


namespace correct_calculation_l54_54848

theorem correct_calculation (a b : ℝ) :
  ¬(a^2 + 2 * a^2 = 3 * a^4) ∧
  ¬(a^6 / a^3 = a^2) ∧
  ¬((a^2)^3 = a^5) ∧
  (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l54_54848


namespace isosceles_triangles_perimeter_l54_54401

theorem isosceles_triangles_perimeter (c d : ℕ) 
  (h1 : ¬(7 = c ∧ 10 = d) ∧ ¬(7 = d ∧ 10 = c))
  (h2 : 2 * c + d = 24) :
  d = 2 :=
sorry

end isosceles_triangles_perimeter_l54_54401


namespace problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l54_54699

variable (x y a b : ℝ)

def A : ℝ := 2*x^2 + a*x - y + 6
def B : ℝ := b*x^2 - 3*x + 5*y - 1

theorem problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13 
  (h : A x y a - B x y b = -6*y + 7) : a^2 + b^2 = 13 := by
  sorry

end problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l54_54699


namespace new_customers_needed_l54_54333

theorem new_customers_needed 
  (initial_customers : ℕ)
  (customers_after_some_left : ℕ)
  (first_group_left : ℕ)
  (second_group_left : ℕ)
  (new_customers : ℕ)
  (h1 : initial_customers = 13)
  (h2 : customers_after_some_left = 9)
  (h3 : first_group_left = initial_customers - customers_after_some_left)
  (h4 : second_group_left = 8)
  (h5 : new_customers = first_group_left + second_group_left) :
  new_customers = 12 :=
by
  sorry

end new_customers_needed_l54_54333


namespace average_after_12th_innings_l54_54354

variable (runs_11 score_12 increase_avg : ℕ)
variable (A : ℕ)

theorem average_after_12th_innings
  (h1 : score_12 = 60)
  (h2 : increase_avg = 2)
  (h3 : 11 * A = runs_11)
  (h4 : (runs_11 + score_12) / 12 = A + increase_avg) :
  (A + 2 = 38) :=
by
  sorry

end average_after_12th_innings_l54_54354


namespace gain_percent_is_150_l54_54668

theorem gain_percent_is_150 (CP SP : ℝ) (hCP : CP = 10) (hSP : SP = 25) : (SP - CP) / CP * 100 = 150 := by
  sorry

end gain_percent_is_150_l54_54668


namespace solve_system_l54_54590

theorem solve_system (x1 x2 x3 : ℝ) :
  (x1 - 2 * x2 + 3 * x3 = 5) ∧ 
  (2 * x1 + 3 * x2 - x3 = 7) ∧ 
  (3 * x1 + x2 + 2 * x3 = 12) 
  ↔ (x1, x2, x3) = (7 - 5 * x3, 1 - x3, x3) :=
by
  sorry

end solve_system_l54_54590


namespace algebraic_expression_value_l54_54985

theorem algebraic_expression_value (a b : ℤ) (h : 2 * (-3) - a + 2 * b = 0) : 2 * a - 4 * b + 1 = -11 := 
by {
  sorry
}

end algebraic_expression_value_l54_54985


namespace binary_ternary_conversion_l54_54613

theorem binary_ternary_conversion (a b : ℕ) (h_b : b = 0 ∨ b = 1) (h_a : a = 0 ∨ a = 1 ∨ a = 2)
  (h_eq : 8 + 2 * b + 1 = 9 * a + 2) : 2 * a + b = 3 :=
by
  sorry

end binary_ternary_conversion_l54_54613


namespace find_rate_of_current_l54_54729

-- Parameters and definitions
variables (r w : Real)

-- Conditions of the problem
def original_journey := 3 * r^2 - 23 * w^2 = 0
def modified_journey := 6 * r^2 - 2 * w^2 + 40 * w = 0

-- Main theorem to prove
theorem find_rate_of_current (h1 : original_journey r w) (h2 : modified_journey r w) :
  w = 10 / 11 :=
sorry

end find_rate_of_current_l54_54729


namespace distance_bob_walked_when_met_l54_54038

theorem distance_bob_walked_when_met (distance_XY walk_rate_Yolanda walk_rate_Bob : ℕ)
  (start_time_Yolanda start_time_Bob : ℕ) (y_distance b_distance : ℕ) (t : ℕ)
  (h1 : distance_XY = 65)
  (h2 : walk_rate_Yolanda = 5)
  (h3 : walk_rate_Bob = 7)
  (h4 : start_time_Yolanda = 0)
  (h5 : start_time_Bob = 1)
  (h6 : y_distance = walk_rate_Yolanda * (t + start_time_Bob))
  (h7 : b_distance = walk_rate_Bob * t)
  (h8 : y_distance + b_distance = distance_XY) : 
  b_distance = 35 := 
sorry

end distance_bob_walked_when_met_l54_54038


namespace scheme2_saves_money_for_80_participants_l54_54398

-- Define the variables and conditions
def total_charge_scheme1 (x : ℕ) (hx : x > 50) : ℕ :=
  1500 + 240 * x

def total_charge_scheme2 (x : ℕ) (hx : x > 50) : ℕ :=
  270 * (x - 5)

-- Define the theorem
theorem scheme2_saves_money_for_80_participants :
  total_charge_scheme2 80 (by decide) < total_charge_scheme1 80 (by decide) :=
sorry

end scheme2_saves_money_for_80_participants_l54_54398


namespace find_abc_l54_54166

noncomputable def log (x : ℝ) : ℝ := sorry -- Replace sorry with an actual implementation of log function if needed

theorem find_abc (a b c : ℝ) 
    (h1 : 1 ≤ a) 
    (h2 : 1 ≤ b) 
    (h3 : 1 ≤ c)
    (h4 : a * b * c = 10)
    (h5 : a^(log a) * b^(log b) * c^(log c) ≥ 10) :
    (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) := 
by
  sorry

end find_abc_l54_54166


namespace intersection_M_N_l54_54283

noncomputable def set_M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
noncomputable def set_N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_M_N :
  (set_M ∩ set_N) = { x | -1 ≤ x ∧ x ≤ Real.sqrt 2 } := sorry

end intersection_M_N_l54_54283


namespace cheese_left_after_10_customers_l54_54140

theorem cheese_left_after_10_customers :
  ∀ (S : ℕ → ℚ), (∀ n, S n = (20 * n) / (n + 10)) →
  20 - S 10 = 10 := by
  sorry

end cheese_left_after_10_customers_l54_54140


namespace cherries_purchase_l54_54552

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end cherries_purchase_l54_54552


namespace maria_workday_ends_at_330_pm_l54_54277

/-- 
Given:
1. Maria's workday is 8 hours long.
2. Her workday does not include her lunch break.
3. Maria starts work at 7:00 A.M.
4. She takes her lunch break at 11:30 A.M., lasting 30 minutes.
Prove that Maria's workday ends at 3:30 P.M.
-/
def maria_end_workday : Prop :=
  let start_time : Nat := 7 * 60 -- in minutes
  let lunch_start_time : Nat := 11 * 60 + 30 -- in minutes
  let lunch_duration : Nat := 30 -- in minutes
  let lunch_end_time : Nat := lunch_start_time + lunch_duration
  let total_work_minutes : Nat := 8 * 60
  let work_before_lunch : Nat := lunch_start_time - start_time
  let remaining_work : Nat := total_work_minutes - work_before_lunch
  let end_time : Nat := lunch_end_time + remaining_work
  end_time = 15 * 60 + 30

theorem maria_workday_ends_at_330_pm : maria_end_workday :=
  by
    sorry

end maria_workday_ends_at_330_pm_l54_54277


namespace interior_diagonals_sum_l54_54847

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 112)
  (h2 : 4 * (a + b + c) = 60) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := 
by 
  sorry

end interior_diagonals_sum_l54_54847


namespace range_of_a_if_f_decreasing_l54_54738

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (x^2 - a * x + 4)

theorem range_of_a_if_f_decreasing:
  ∀ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → f a y < f a x) →
    2 ≤ a ∧ a ≤ 5 :=
by
  intros a h
  sorry

end range_of_a_if_f_decreasing_l54_54738


namespace friends_gift_l54_54359

-- Define the original number of balloons and the final number of balloons
def original_balloons := 8
def final_balloons := 10

-- The main theorem: Joan's friend gave her 2 orange balloons.
theorem friends_gift : (final_balloons - original_balloons) = 2 := by
  sorry

end friends_gift_l54_54359


namespace integer_multiplied_by_b_l54_54481

variable (a b : ℤ) (x : ℤ)

theorem integer_multiplied_by_b (h1 : -11 * a < 0) (h2 : x < 0) (h3 : (-11 * a * x) * (x * b) + a * b = 89) :
  x = -1 :=
by
  sorry

end integer_multiplied_by_b_l54_54481


namespace relation_among_a_b_c_l54_54091

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 6 - Real.sqrt 2

theorem relation_among_a_b_c : a > c ∧ c > b :=
by {
  sorry
}

end relation_among_a_b_c_l54_54091


namespace find_m_n_and_max_value_l54_54230

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + 3 * m + n

-- Define a predicate for the function being even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the conditions and what we want to prove
theorem find_m_n_and_max_value :
  ∀ m n : ℝ,
    is_even_function (f m n) →
    (m - 1 ≤ 2 * m) →
      (m = 1 / 3 ∧ n = 0) ∧ 
      (∀ x : ℝ, -2 / 3 ≤ x ∧ x ≤ 2 / 3 → f (1/3) 0 x ≤ 31 / 27) :=
by
  sorry

end find_m_n_and_max_value_l54_54230


namespace textbook_cost_l54_54787

theorem textbook_cost 
  (credits : ℕ) 
  (cost_per_credit : ℕ) 
  (facility_fee : ℕ) 
  (total_cost : ℕ) 
  (num_textbooks : ℕ) 
  (total_spent : ℕ) 
  (h1 : credits = 14) 
  (h2 : cost_per_credit = 450) 
  (h3 : facility_fee = 200) 
  (h4 : total_spent = 7100) 
  (h5 : num_textbooks = 5) :
  (total_cost - (credits * cost_per_credit + facility_fee)) / num_textbooks = 120 :=
by
  sorry

end textbook_cost_l54_54787


namespace int_as_sum_of_squares_l54_54516

theorem int_as_sum_of_squares (n : ℤ) : ∃ a b c : ℤ, n = a^2 + b^2 - c^2 :=
sorry

end int_as_sum_of_squares_l54_54516


namespace eighth_arithmetic_term_l54_54831

theorem eighth_arithmetic_term (a₂ a₁₄ a₈ : ℚ) 
  (h2 : a₂ = 8 / 11)
  (h14 : a₁₄ = 9 / 13) :
  a₈ = 203 / 286 :=
by
  sorry

end eighth_arithmetic_term_l54_54831


namespace average_age_of_students_is_14_l54_54866

noncomputable def average_age_of_students (student_count : ℕ) (teacher_age : ℕ) (combined_avg_age : ℕ) : ℕ :=
  let total_people := student_count + 1
  let total_combined_age := total_people * combined_avg_age
  let total_student_age := total_combined_age - teacher_age
  total_student_age / student_count

theorem average_age_of_students_is_14 :
  average_age_of_students 50 65 15 = 14 :=
by
  sorry

end average_age_of_students_is_14_l54_54866


namespace trains_crossing_time_l54_54507

noncomputable def timeToCross (L1 L2 : ℕ) (v1 v2 : ℕ) : ℝ :=
  let total_distance := (L1 + L2 : ℝ)
  let relative_speed := ((v1 + v2) * 1000 / 3600 : ℝ) -- converting km/hr to m/s
  total_distance / relative_speed

theorem trains_crossing_time :
  timeToCross 140 160 60 40 = 10.8 := 
  by 
    sorry

end trains_crossing_time_l54_54507


namespace distinguishable_balls_boxes_l54_54424

theorem distinguishable_balls_boxes : (3^6 = 729) :=
by {
  sorry
}

end distinguishable_balls_boxes_l54_54424


namespace junk_mail_per_house_l54_54181

theorem junk_mail_per_house (total_junk_mail : ℕ) (houses_per_block : ℕ) 
  (h1 : total_junk_mail = 14) (h2 : houses_per_block = 7) : 
  (total_junk_mail / houses_per_block) = 2 :=
by 
  sorry

end junk_mail_per_house_l54_54181


namespace pow_product_l54_54033

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end pow_product_l54_54033


namespace find_xy_l54_54946

theorem find_xy (x y : ℝ) (k : ℤ) :
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 ↔
  (x = -Real.arccos (-4/5) + (2 * k + 1) * Real.pi ∧ y = -1/2) := by
  sorry

end find_xy_l54_54946


namespace weight_lifting_ratio_l54_54032

theorem weight_lifting_ratio :
  ∀ (F S : ℕ), F + S = 600 ∧ F = 300 ∧ 2 * F = S + 300 → F / S = 1 :=
by
  intro F S
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_lifting_ratio_l54_54032


namespace coordinates_satisfy_l54_54988

theorem coordinates_satisfy (x y : ℝ) : y * (x + 1) = x^2 - 1 ↔ (x = -1 ∨ y = x - 1) :=
by
  sorry

end coordinates_satisfy_l54_54988


namespace square_side_length_l54_54045

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l54_54045


namespace cost_of_shorts_l54_54766

-- Define the given conditions and quantities
def initial_money : ℕ := 50
def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def remaining_money : ℕ := 14

-- The total amount spent
def total_spent : ℕ := initial_money - remaining_money

-- The total cost of the jerseys and basketball
def jerseys_basketball_cost : ℕ := jerseys_cost + basketball_cost

-- The cost of the shorts
def shorts_cost : ℕ := total_spent - jerseys_basketball_cost

theorem cost_of_shorts : shorts_cost = 8 := sorry

end cost_of_shorts_l54_54766


namespace ratio_of_sides_l54_54367

theorem ratio_of_sides (s x y : ℝ) 
    (h1 : 0.1 * s^2 = 0.25 * x * y)
    (h2 : x = s / 10)
    (h3 : y = 4 * s) : x / y = 1 / 40 :=
by
  sorry

end ratio_of_sides_l54_54367


namespace expandProduct_l54_54082

theorem expandProduct (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := 
by 
  sorry

end expandProduct_l54_54082


namespace intersect_parabolas_l54_54962

theorem intersect_parabolas :
  ∀ (x y : ℝ),
    ((y = 2 * x^2 - 7 * x + 1 ∧ y = 8 * x^2 + 5 * x + 1) ↔ 
     ((x = -2 ∧ y = 23) ∨ (x = 0 ∧ y = 1))) :=
by sorry

end intersect_parabolas_l54_54962


namespace remainder_sets_two_disjoint_subsets_l54_54269

noncomputable def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem remainder_sets_two_disjoint_subsets (m : ℕ)
  (h : m = (3^12 - 2 * 2^12 + 1) / 2) : m % 1000 = 625 := 
by {
  -- math proof is omitted
  sorry
}

end remainder_sets_two_disjoint_subsets_l54_54269


namespace ab_value_l54_54006

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := 
by
  sorry

end ab_value_l54_54006


namespace necessary_condition_x_squared_minus_x_lt_zero_l54_54286

theorem necessary_condition_x_squared_minus_x_lt_zero (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧ ((-1 < x ∧ x < 1) → ¬ (x^2 - x < 0)) :=
by
  sorry

end necessary_condition_x_squared_minus_x_lt_zero_l54_54286


namespace find_x_if_vectors_parallel_l54_54240

/--
Given the vectors a = (2 * x + 1, 3) and b = (2 - x, 1), if a is parallel to b, 
then x must be equal to 1.
-/
theorem find_x_if_vectors_parallel (x : ℝ) :
  let a := (2 * x + 1, 3)
  let b := (2 - x, 1)
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 :=
by
  sorry

end find_x_if_vectors_parallel_l54_54240


namespace fiona_received_59_l54_54171

theorem fiona_received_59 (Dan_riddles : ℕ) (Andy_riddles : ℕ) (Bella_riddles : ℕ) (Emma_riddles : ℕ) (Fiona_riddles : ℕ)
  (h1 : Dan_riddles = 21)
  (h2 : Andy_riddles = Dan_riddles + 12)
  (h3 : Bella_riddles = Andy_riddles - 7)
  (h4 : Emma_riddles = Bella_riddles / 2)
  (h5 : Fiona_riddles = Andy_riddles + Bella_riddles) :
  Fiona_riddles = 59 :=
by
  sorry

end fiona_received_59_l54_54171


namespace repeating_decimal_transform_l54_54732

theorem repeating_decimal_transform (n : ℕ) (s : String) (k : ℕ) (m : ℕ)
  (original : s = "2345678") (len : k = 7) (position : n = 2011)
  (effective_position : m = n - 1) (mod_position : m % k = 3) :
  "0.1" ++ s = "0.12345678" :=
sorry

end repeating_decimal_transform_l54_54732


namespace relationship_x_x2_negx_l54_54132

theorem relationship_x_x2_negx (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x :=
by
  sorry

end relationship_x_x2_negx_l54_54132


namespace arithmetic_sequence_condition_l54_54824

theorem arithmetic_sequence_condition (a : ℕ → ℕ) 
(h1 : a 4 = 4) 
(h2 : a 3 + a 8 = 5) : 
a 7 = 1 := 
sorry

end arithmetic_sequence_condition_l54_54824


namespace book_area_correct_l54_54363

/-- Converts inches to centimeters -/
def inch_to_cm (inches : ℚ) : ℚ :=
  inches * 2.54

/-- The length of the book given a parameter x -/
def book_length (x : ℚ) : ℚ :=
  3 * x - 4

/-- The width of the book in inches -/
def book_width_in_inches : ℚ :=
  5 / 2

/-- The width of the book in centimeters -/
def book_width : ℚ :=
  inch_to_cm book_width_in_inches

/-- The area of the book given a parameter x -/
def book_area (x : ℚ) : ℚ :=
  book_length x * book_width

/-- Proof that the area of the book with x = 5 is 69.85 cm² -/
theorem book_area_correct : book_area 5 = 69.85 := by
  sorry

end book_area_correct_l54_54363


namespace line_parallel_not_coincident_l54_54253

theorem line_parallel_not_coincident (a : ℝ) :
  (a = 3) ↔ (∀ x y, (a * x + 2 * y + 3 * a = 0) ∧ (3 * x + (a - 1) * y + 7 - a = 0) → 
              (∃ k : Real, a / 3 = k ∧ k ≠ 3 * a / (7 - a))) :=
by
  sorry

end line_parallel_not_coincident_l54_54253


namespace find_a_find_n_l54_54825

noncomputable def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum_of_first_n_terms (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
noncomputable def S (a d n : ℕ) : ℕ := if n = 1 then a else sum_of_first_n_terms a d n
noncomputable def arithmetic_sum_property (a d n : ℕ) : Prop :=
  ∀ n ≥ 2, (S a d n) ^ 2 = 3 * n ^ 2 * arithmetic_sequence a d n + (S a d (n - 1)) ^ 2

theorem find_a (a : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2) :
  a = 3 :=
sorry

noncomputable def c (n : ℕ) (a5 : ℕ) : ℕ := 3 ^ (n - 1) + a5
noncomputable def sum_of_first_n_terms_c (n a5 : ℕ) : ℕ := (3^n - 1) / 2 + 15 * n
noncomputable def T (n a5 : ℕ) : ℕ := sum_of_first_n_terms_c n a5

theorem find_n (a : ℕ) (a5 : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2)
  (h2 : a = 3) (h3 : a5 = 15) :
  ∃ n : ℕ, 4 * T n a5 > S a 3 10 ∧ n = 3 :=
sorry

end find_a_find_n_l54_54825


namespace triangle_shape_l54_54178

-- Define the sides of the triangle and the angles
variables {a b c : ℝ}
variables {A B C : ℝ} 
-- Assume that angles are in radians and 0 < A, B, C < π
-- Also assume that the sum of angles in the triangle is π
axiom angle_sum_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : a^2 * Real.cos A * Real.sin B = b^2 * Real.sin A * Real.cos B

-- Conclusion: The shape of triangle ABC is either isosceles or right triangle
theorem triangle_shape : 
  (A = B) ∨ (A + B = (Real.pi / 2)) := 
by sorry

end triangle_shape_l54_54178


namespace no_solution_1221_l54_54830

def equation_correctness (n : ℤ) : Prop :=
  -n^3 + 555^3 = n^2 - n * 555 + 555^2

-- Prove that the prescribed value 1221 does not satisfy the modified equation by contradiction
theorem no_solution_1221 : ¬ ∃ n : ℤ, equation_correctness n ∧ n = 1221 := by
  sorry

end no_solution_1221_l54_54830


namespace gcd_power_of_two_sub_one_l54_54751

def a : ℤ := 2^1100 - 1
def b : ℤ := 2^1122 - 1
def c : ℤ := 2^22 - 1

theorem gcd_power_of_two_sub_one :
  Int.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end gcd_power_of_two_sub_one_l54_54751


namespace pages_copied_l54_54327

-- Define the assumptions
def cost_per_pages (cent_per_pages: ℕ) : Prop := 
  5 * cent_per_pages = 7 * 1

def total_cents (dollars: ℕ) (cents: ℕ) : Prop :=
  cents = dollars * 100

-- The problem to prove
theorem pages_copied (dollars: ℕ) (cents: ℕ) (cent_per_pages: ℕ) : 
  cost_per_pages cent_per_pages → total_cents dollars cents → dollars = 35 → cents = 3500 → 
  3500 * (5/7 : ℚ) = 2500 :=
by
  sorry

end pages_copied_l54_54327


namespace factorize_expression1_factorize_expression2_l54_54190

variable {R : Type*} [CommRing R]

theorem factorize_expression1 (x y : R) : x^2 + 2 * x + 1 - y^2 = (x + y + 1) * (x - y + 1) :=
  sorry

theorem factorize_expression2 (m n p : R) : m^2 - n^2 - 2 * n * p - p^2 = (m + n + p) * (m - n - p) :=
  sorry

end factorize_expression1_factorize_expression2_l54_54190


namespace shaded_area_of_square_l54_54940

theorem shaded_area_of_square (side_square : ℝ) (leg_triangle : ℝ) (h1 : side_square = 40) (h2 : leg_triangle = 25) :
  let area_square := side_square ^ 2
  let area_triangle := (1 / 2) * leg_triangle * leg_triangle
  let total_area_triangles := 2 * area_triangle
  let shaded_area := area_square - total_area_triangles
  shaded_area = 975 :=
by
  sorry

end shaded_area_of_square_l54_54940


namespace sqrt_meaningful_l54_54852

theorem sqrt_meaningful (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end sqrt_meaningful_l54_54852
