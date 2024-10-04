import Mathlib

namespace probability_of_3_black_2_white_l154_154766

def total_balls := 15
def black_balls := 10
def white_balls := 5
def drawn_balls := 5
def drawn_black_balls := 3
def drawn_white_balls := 2

noncomputable def probability_black_white_draw : ℝ :=
  (Nat.choose black_balls drawn_black_balls * Nat.choose white_balls drawn_white_balls : ℝ) /
  (Nat.choose total_balls drawn_balls : ℝ)

theorem probability_of_3_black_2_white :
  probability_black_white_draw = 400 / 1001 := by
  sorry

end probability_of_3_black_2_white_l154_154766


namespace find_x_l154_154593

theorem find_x (x : ℕ) 
  (h : (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750) : 
  x = 1255 := 
sorry

end find_x_l154_154593


namespace circle_area_irrational_of_rational_radius_l154_154971

theorem circle_area_irrational_of_rational_radius (r : ℚ) : ¬ ∃ A : ℚ, A = π * (r:ℝ) * (r:ℝ) :=
by sorry

end circle_area_irrational_of_rational_radius_l154_154971


namespace ratio_of_ages_l154_154295

theorem ratio_of_ages (x m : ℕ) 
  (mother_current_age : ℕ := 41) 
  (daughter_current_age : ℕ := 23) 
  (age_diff : ℕ := mother_current_age - daughter_current_age) 
  (eq : (mother_current_age - x) = m * (daughter_current_age - x)) : 
  (41 - x) / (23 - x) = m :=
by
  -- Proof not required
  sorry

end ratio_of_ages_l154_154295


namespace alcohol_water_ratio_l154_154616

theorem alcohol_water_ratio (A W A_new W_new : ℝ) (ha1 : A / W = 4 / 3) (ha2: A = 5) (ha3: W_new = W + 7) : A / W_new = 1 / 2.15 :=
by
  sorry

end alcohol_water_ratio_l154_154616


namespace xiao_li_profit_l154_154980

noncomputable def original_price_per_share : ℝ := 21 / 1.05
noncomputable def closing_price_first_day : ℝ := original_price_per_share * 0.94
noncomputable def selling_price_second_day : ℝ := closing_price_first_day * 1.10
noncomputable def total_profit : ℝ := (selling_price_second_day - 21) * 5000

theorem xiao_li_profit :
  total_profit = 600 := sorry

end xiao_li_profit_l154_154980


namespace investment_period_l154_154117

theorem investment_period (x t : ℕ) (p_investment q_investment q_time : ℕ) (profit_ratio : ℚ):
  q_investment = 5 * x →
  p_investment = 7 * x →
  q_time = 16 →
  profit_ratio = 7 / 10 →
  7 * x * t = profit_ratio * 5 * x * q_time →
  t = 8 := sorry

end investment_period_l154_154117


namespace rita_remaining_money_l154_154876

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l154_154876


namespace functional_equation_solution_l154_154631

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2) →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 :=
by
  sorry

end functional_equation_solution_l154_154631


namespace loss_recorded_as_negative_l154_154069

-- Define the condition that a profit of 100 yuan is recorded as +100 yuan
def recorded_profit (p : ℤ) : Prop :=
  p = 100

-- Define the condition about how a profit is recorded
axiom profit_condition : recorded_profit 100

-- Define the function for recording profit or loss
def record (x : ℤ) : ℤ :=
  if x > 0 then x
  else -x

-- Theorem: If a profit of 100 yuan is recorded as +100 yuan, then a loss of 50 yuan is recorded as -50 yuan.
theorem loss_recorded_as_negative : ∀ x: ℤ, (x < 0) → record x = -x :=
by
  intros x h
  unfold record
  simp [h]
  -- sorry indicates the proof is not provided
  sorry

end loss_recorded_as_negative_l154_154069


namespace geometric_sequence_sum_n5_l154_154667

def geometric_sum (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_n5 (a₁ q : ℕ) (n : ℕ) (h₁ : a₁ = 3) (h₂ : q = 4) (h₃ : n = 5) : 
  geometric_sum a₁ q n = 1023 :=
by
  sorry

end geometric_sequence_sum_n5_l154_154667


namespace cubic_root_expression_l154_154238

theorem cubic_root_expression (p q r : ℝ) (h1 : p + q + r = 0) (h2 : p * q + p * r + q * r = -2) (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -24 :=
sorry

end cubic_root_expression_l154_154238


namespace rectangle_area_increase_l154_154750

-- Definitions to match the conditions
variables {l w : ℝ}

-- The statement 
theorem rectangle_area_increase (h1 : l > 0) (h2 : w > 0) :
  (((1.15 * l) * (1.2 * w) - (l * w)) / (l * w)) * 100 = 38 :=
by
  sorry

end rectangle_area_increase_l154_154750


namespace lion_room_is_3_l154_154143

/-!
  A lion is hidden in one of three rooms. A note on the door of room 1 reads "The lion is here".
  A note on the door of room 2 reads "The lion is not here". A note on the door of room 3 reads "2+3=2×3".
  Only one of these notes is true. Prove that the lion is in room 3.
-/

def note1 (lion_room : ℕ) : Prop := lion_room = 1
def note2 (lion_room : ℕ) : Prop := lion_room ≠ 2
def note3 (lion_room : ℕ) : Prop := 2 + 3 = 2 * 3
def lion_is_in_room3 : Prop := ∀ lion_room, (note1 lion_room ∨ note2 lion_room ∨ note3 lion_room) ∧
  (note1 lion_room → note2 lion_room = false) ∧ (note1 lion_room → note3 lion_room = false) ∧
  (note2 lion_room → note1 lion_room = false) ∧ (note2 lion_room → note3 lion_room = false) ∧
  (note3 lion_room → note1 lion_room = false) ∧ (note3 lion_room → note2 lion_room = false) → lion_room = 3

theorem lion_room_is_3 : lion_is_in_room3 := 
  by
  sorry

end lion_room_is_3_l154_154143


namespace isosceles_triangle_perimeter_l154_154341

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end isosceles_triangle_perimeter_l154_154341


namespace prime_divides_3np_minus_3n1_l154_154092

theorem prime_divides_3np_minus_3n1 (p n : ℕ) (hp : Prime p) : p ∣ (3^(n + p) - 3^(n + 1)) :=
sorry

end prime_divides_3np_minus_3n1_l154_154092


namespace opposite_of_neg_one_third_l154_154575

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end opposite_of_neg_one_third_l154_154575


namespace complex_expression_equality_l154_154330

open Complex

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := 
sorry

end complex_expression_equality_l154_154330


namespace solve_equation_l154_154721

theorem solve_equation (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2)
  (h₃ : (3 * x + 6)/(x^2 + 5 * x + 6) = (3 - x)/(x - 2)) :
  x = 3 ∨ x = -3 :=
sorry

end solve_equation_l154_154721


namespace smallest_composite_proof_l154_154648

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l154_154648


namespace find_F_l154_154359

theorem find_F (F C : ℝ) (h1 : C = 35) (h2 : C = (7/12) * (F - 40)) : F = 100 :=
by
  sorry

end find_F_l154_154359


namespace intersection_eq_l154_154859

def A : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}
def B : Set ℝ := {y | y ≤ 1}

theorem intersection_eq : A ∩ {y | y ∈ Set.Icc (-2 : ℤ) 1} = {-2, -1, 0, 1} := by
  sorry

end intersection_eq_l154_154859


namespace algebra_inequality_l154_154949

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end algebra_inequality_l154_154949


namespace unique_solution_for_exponential_eq_l154_154938

theorem unique_solution_for_exponential_eq (a y : ℕ) (h_a : a ≥ 1) (h_y : y ≥ 1) :
  3^(2*a-1) + 3^a + 1 = 7^y ↔ (a = 1 ∧ y = 1) := by
  sorry

end unique_solution_for_exponential_eq_l154_154938


namespace sawing_time_determination_l154_154720

variable (totalLength pieceLength sawTime : Nat)

theorem sawing_time_determination
  (h1 : totalLength = 10)
  (h2 : pieceLength = 2)
  (h3 : sawTime = 10) :
  (totalLength / pieceLength - 1) * sawTime = 40 := by
  sorry

end sawing_time_determination_l154_154720


namespace russian_needed_goals_equals_tunisian_scored_goals_l154_154882

-- Define the total goals required by each team
def russian_goals := 9
def tunisian_goals := 5

-- Statement: there exists a moment where the Russian remaining goals equal the Tunisian scored goals
theorem russian_needed_goals_equals_tunisian_scored_goals :
  ∃ n : ℕ, n ≤ russian_goals ∧ (russian_goals - n) = (tunisian_goals) := by
  sorry

end russian_needed_goals_equals_tunisian_scored_goals_l154_154882


namespace martin_improved_lap_time_l154_154711

def initial_laps := 15
def initial_time := 45 -- in minutes
def final_laps := 18
def final_time := 42 -- in minutes

noncomputable def initial_lap_time := initial_time / initial_laps
noncomputable def final_lap_time := final_time / final_laps
noncomputable def improvement := initial_lap_time - final_lap_time

theorem martin_improved_lap_time : improvement = 2 / 3 := by 
  sorry

end martin_improved_lap_time_l154_154711


namespace geometric_sequence_common_ratio_l154_154347

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 2)
  (h3 : a 5 = 1/4) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l154_154347


namespace inverse_variation_with_constant_l154_154969

theorem inverse_variation_with_constant
  (k : ℝ)
  (x y : ℝ)
  (h1 : y = (3 * k) / x)
  (h2 : x = 4)
  (h3 : y = 8) :
  (y = (3 * (32 / 3)) / -16) := by
sorry

end inverse_variation_with_constant_l154_154969


namespace true_propositions_l154_154155

theorem true_propositions :
  (¬ (∀ (A B : Type) (f : A → B) (x y : A), f x = f y → x = y)) ∧   -- (1)
  ((∃ (x : ℝ), x = Real.pi ^ Real.sqrt 2)) ∧                        -- (2)
  (¬ (∃ (x : ℝ), (x^2 + 2 * x + 3 = 0))) ∧                          -- (3)
  (¬ (∀ (x y : ℝ), x^2 ≠ y^2 ↔ x ≠ y ∨ x ≠ -y)) ∧                    -- (4)
  (¬ (∀ (a b : ℕ), (a % 2 = 0 ∧ b % 2 = 0) → ((a + b) % 2 = 0) → (a + b) % 2 ≠ 1)) ∧  -- (5)
  (¬ (∀ (p q : Prop), ¬ (p ∨ q) ↔ ¬ p ∧ ¬ q)) ∧                    -- (6)
  (¬ (∀ (a b c : ℝ), (∀ (x : ℝ), ¬ (a * x^2 + b * x + c ≤ 0)) → a > 0 ∧ b^2 - 4 * a * c < 0))  -- (7)
  := by
    sorry

end true_propositions_l154_154155


namespace birds_flew_up_count_l154_154763

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end birds_flew_up_count_l154_154763


namespace range_of_a_if_f_increasing_l154_154072

theorem range_of_a_if_f_increasing (a : ℝ) :
  (∀ x : ℝ, 3*x^2 + 3*a ≥ 0) → (a ≥ 0) :=
sorry

end range_of_a_if_f_increasing_l154_154072


namespace sue_received_votes_l154_154265

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end sue_received_votes_l154_154265


namespace stratified_sampling_group_C_l154_154150

theorem stratified_sampling_group_C
  (total_cities : ℕ)
  (cities_group_A : ℕ)
  (cities_group_B : ℕ)
  (cities_group_C : ℕ)
  (total_selected : ℕ)
  (C_subset_correct: total_cities = cities_group_A + cities_group_B + cities_group_C)
  (total_cities_correct: total_cities = 48)
  (cities_group_A_correct: cities_group_A = 8)
  (cities_group_B_correct: cities_group_B = 24)
  (total_selected_correct: total_selected = 12)
  : (total_selected * cities_group_C) / total_cities = 4 :=
by 
  sorry

end stratified_sampling_group_C_l154_154150


namespace problem_l154_154070

theorem problem (a b : ℤ) (h1 : |a - 2| = 5) (h2 : |b| = 9) (h3 : a + b < 0) :
  a - b = 16 ∨ a - b = 6 := 
sorry

end problem_l154_154070


namespace total_stones_l154_154879

theorem total_stones (sent_away kept total : ℕ) (h1 : sent_away = 63) (h2 : kept = 15) (h3 : total = sent_away + kept) : total = 78 :=
by
  sorry

end total_stones_l154_154879


namespace problem1_problem2_l154_154785

-- Proof problem 1
theorem problem1 : 
  (real.cbrt 2 * real.sqrt 3) ^ 6 + (real.sqrt (2 * real.sqrt 2)) ^ (4 / 3) - 
  4 * (16 / 49) ^ (-1 / 2) - real.root 4 2 * 8 ^ (1 / 4) - (-2017) ^ 0 = 100 :=
by
  sorry

-- Proof problem 2
theorem problem2 :
  real.log 2.5 6.25 + real.lg 0.01 + real.ln (real.sqrt real.exp 1) - 2 ^ (1 + real.log 2 3) = -11 / 2 :=
by
  sorry

end problem1_problem2_l154_154785


namespace quadrilateral_area_beih_correct_l154_154592

-- Definitions based on conditions in the problem
def point := (ℝ × ℝ)

noncomputable def square_vertices : point → point → point → point → Prop :=
λ A B C D, A = (0, 3) ∧ B = (0, 0) ∧ C = (3, 0) ∧ D = (3, 3)

noncomputable def midpoint (A B : point) : point :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_eq : point → point → (ℝ × ℝ) :=
λ P Q, let m := (Q.2 - P.2) / (Q.1 - P.1) in (m, P.2 - m * P.1)

noncomputable def intersect (l1 l2 : ℝ × ℝ) : point :=
let x := (l2.2 - l1.2) / (l1.1 - l2.1) in (x, l1.1 * x + l1.2)

noncomputable def area (points : list point) : ℝ :=
|points.head.1 * points.last.2 - points.last.1 * points.head.2 + list.sum (list.map₂ (λ p1 p2, p1.1 * p2.2 - p2.1 * p1.2) points (points.tail ++ [points.head]))| / 2

-- Proof statement
theorem quadrilateral_area_beih_correct {A B C D E F I H : point}
  (h_square : square_vertices A B C D)
  (h_midpt_E : E = midpoint A B)
  (h_midpt_F : F = midpoint B C)
  (h_intersect_I : I = intersect (line_eq A F) (line_eq D E))
  (h_intersect_H : H = intersect (line_eq B D) (line_eq A F)) :
  area [B, E, I, H] = 1.35 :=
sorry

end quadrilateral_area_beih_correct_l154_154592


namespace calculate_dani_pants_l154_154626

theorem calculate_dani_pants : ∀ (initial_pants number_years pairs_per_year pants_per_pair : ℕ), 
  initial_pants = 50 →
  number_years = 5 →
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants + (number_years * (pairs_per_year * pants_per_pair)) = 90 :=
by
  intros initial_pants number_years pairs_per_year pants_per_pair
  intro h_initial_pants h_number_years h_pairs_per_year h_pants_per_pair
  rw [h_initial_pants, h_number_years, h_pairs_per_year, h_pants_per_pair]
  norm_num
  sorry

end calculate_dani_pants_l154_154626


namespace distance_from_dormitory_to_city_l154_154284

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h1 : D = (1/2) * D + (1/4) * D + 6) : D = 24 := 
  sorry

end distance_from_dormitory_to_city_l154_154284


namespace temperature_problem_l154_154311

theorem temperature_problem
  (M L N : ℝ)
  (h1 : M = L + N)
  (h2 : M - 9 = M - 9)
  (h3 : L + 5 = L + 5)
  (h4 : abs (M - 9 - (L + 5)) = 1) :
  (N = 15 ∨ N = 13) → (N = 15 ∧ N = 13 → 15 * 13 = 195) :=
by
  sorry

end temperature_problem_l154_154311


namespace probability_of_three_primes_out_of_six_l154_154925

-- Define the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

-- Given six 12-sided fair dice
def total_dice : ℕ := 6
def sides : ℕ := 12

-- Probability of rolling a prime number on one die
def prime_probability : ℚ := 5 / 12

-- Probability of rolling a non-prime number on one die
def non_prime_probability : ℚ := 7 / 12

-- Number of ways to choose 3 dice from 6 to show a prime number
def combination (n k : ℕ) : ℕ := n.choose k
def choose_3_out_of_6 : ℕ := combination total_dice 3

-- Combined probability for exactly 3 primes and 3 non-primes
def combined_probability : ℚ :=
  (prime_probability ^ 3) * (non_prime_probability ^ 3)

-- Total probability
def total_probability : ℚ :=
  choose_3_out_of_6 * combined_probability

-- Main theorem statement
theorem probability_of_three_primes_out_of_six :
  total_probability = 857500 / 5177712 :=
by
  sorry

end probability_of_three_primes_out_of_six_l154_154925


namespace sean_final_cost_l154_154544

noncomputable def totalCost (sodaCount soupCount sandwichCount saladCount : ℕ)
                            (pricePerSoda pricePerSoup pricePerSandwich pricePerSalad : ℚ)
                            (discountRate taxRate : ℚ) : ℚ :=
  let totalCostBeforeDiscount := (sodaCount * pricePerSoda) +
                                (soupCount * pricePerSoup) +
                                (sandwichCount * pricePerSandwich) +
                                (saladCount * pricePerSalad)
  let discountedTotal := totalCostBeforeDiscount * (1 - discountRate)
  let finalCost := discountedTotal * (1 + taxRate)
  finalCost

theorem sean_final_cost : 
  totalCost 4 3 2 1 
            1 (2 * 1) (4 * (2 * 1)) (2 * (4 * (2 * 1)))
            0.1 0.05 = 39.69 := 
by
  sorry

end sean_final_cost_l154_154544


namespace perimeter_ratio_l154_154255

/-- Suppose we have a square piece of paper, 6 inches on each side, folded in half horizontally. 
The paper is then cut along the fold, and one of the halves is subsequently cut again horizontally 
through all layers. This results in one large rectangle and two smaller identical rectangles. 
Find the ratio of the perimeter of one smaller rectangle to the perimeter of the larger rectangle. -/
theorem perimeter_ratio (side_length : ℝ) (half_side_length : ℝ) (double_half_side_length : ℝ) :
    side_length = 6 →
    half_side_length = side_length / 2 →
    double_half_side_length = 1.5 * 2 →
    (2 * (half_side_length / 2 + side_length)) / (2 * (half_side_length + side_length)) = (5 / 6) :=
by
    -- Declare the side lengths
    intros h₁ h₂ h₃
    -- Insert the necessary algebra (proven manually earlier)
    sorry

end perimeter_ratio_l154_154255


namespace forty_percent_of_n_l154_154377

theorem forty_percent_of_n (N : ℝ) (h : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10) : 0.40 * N = 120 := by
  sorry

end forty_percent_of_n_l154_154377


namespace fraction_zero_x_eq_2_l154_154224

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end fraction_zero_x_eq_2_l154_154224


namespace books_fill_shelf_l154_154231

theorem books_fill_shelf
  (A H S M E : ℕ)
  (h1 : A ≠ H) (h2 : S ≠ M) (h3 : M ≠ H) (h4 : E > 0)
  (Eq1 : A > 0) (Eq2 : H > 0) (Eq3 : S > 0) (Eq4 : M > 0)
  (h5 : A ≠ S) (h6 : E ≠ A) (h7 : E ≠ H) (h8 : E ≠ S) (h9 : E ≠ M) :
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end books_fill_shelf_l154_154231


namespace unique_solution_in_z3_l154_154545

theorem unique_solution_in_z3 (x y z : ℤ) (h : x^3 + 2 * y^3 = 4 * z^3) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end unique_solution_in_z3_l154_154545


namespace all_initial_rectangles_are_squares_l154_154520

theorem all_initial_rectangles_are_squares (n : ℕ) (total_squares : ℕ) (h_prime : Nat.Prime total_squares) 
  (cut_rect_into_squares : ℕ → ℕ → ℕ → Prop) :
  ∀ (a b : ℕ), (∀ i, i < n → cut_rect_into_squares a b total_squares) → a = b :=
by 
  sorry

end all_initial_rectangles_are_squares_l154_154520


namespace total_elephants_in_two_parks_l154_154385

theorem total_elephants_in_two_parks (n1 n2 : ℕ) (h1 : n1 = 70) (h2 : n2 = 3 * n1) : n1 + n2 = 280 := by
  sorry

end total_elephants_in_two_parks_l154_154385


namespace jovana_added_pounds_l154_154085

noncomputable def initial_amount : ℕ := 5
noncomputable def final_amount : ℕ := 28

theorem jovana_added_pounds : final_amount - initial_amount = 23 := by
  sorry

end jovana_added_pounds_l154_154085


namespace range_of_a_l154_154827

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- Prove that if f(x) is decreasing on ℝ, then a must be less than or equal to -3
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (3 * a * x^2 + 6 * x - 1) < 0 ) → a ≤ -3 :=
sorry

end range_of_a_l154_154827


namespace find_fraction_of_original_flow_rate_l154_154781

noncomputable def fraction_of_original_flow_rate (f : ℚ) : Prop :=
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  reduced_flow_rate = f * original_flow_rate - 1

theorem find_fraction_of_original_flow_rate : ∃ (f : ℚ), fraction_of_original_flow_rate f ∧ f = 3 / 5 :=
by
  sorry

end find_fraction_of_original_flow_rate_l154_154781


namespace smallest_four_digit_mod_8_l154_154434

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l154_154434


namespace num_terms_in_expansion_eq_3_pow_20_l154_154700

-- Define the expression 
def expr (x y : ℝ) := (1 + x + y) ^ 20

-- Statement of the problem
theorem num_terms_in_expansion_eq_3_pow_20 (x y : ℝ) : (3 : ℝ)^20 = (1 + x + y) ^ 20 :=
by sorry

end num_terms_in_expansion_eq_3_pow_20_l154_154700


namespace jacob_age_proof_l154_154974

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end jacob_age_proof_l154_154974


namespace even_integers_count_l154_154682

theorem even_integers_count (n : ℤ) (m : ℤ) (total_even : ℤ) 
  (h1 : m = 45) (h2 : total_even = 10) (h3 : m % 2 = 1) :
  (∃ k : ℤ, ∀ x : ℤ, 0 ≤ x ∧ x < total_even → k = n + 2 * x) ∧ (n = 26) :=
by
  sorry

end even_integers_count_l154_154682


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154393

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154393


namespace remainder_T2015_mod_12_eq_8_l154_154944

-- Define sequences of length n consisting of the letters A and B,
-- with no more than two A's in a row and no more than two B's in a row
def T : ℕ → ℕ :=
  sorry  -- Definition for T(n) must follow the given rules

-- Theorem to prove that T(2015) modulo 12 equals 8
theorem remainder_T2015_mod_12_eq_8 :
  (T 2015) % 12 = 8 :=
  sorry

end remainder_T2015_mod_12_eq_8_l154_154944


namespace students_per_class_l154_154768

variable (c : ℕ) (s : ℕ)

def books_per_month := 6
def months_per_year := 12
def books_per_year := books_per_month * months_per_year
def total_books_read := 72

theorem students_per_class : (s * c = 1 ∧ s * books_per_year = total_books_read) → s = 1 := by
  intros h
  have h1: books_per_year = total_books_read := by
    calc
      books_per_year = books_per_month * months_per_year := rfl
      _ = 6 * 12 := rfl
      _ = 72 := rfl
  sorry

end students_per_class_l154_154768


namespace division_by_n_minus_1_squared_l154_154381

theorem division_by_n_minus_1_squared (n : ℕ) (h : n > 2) : (n ^ (n - 1) - 1) % ((n - 1) ^ 2) = 0 :=
sorry

end division_by_n_minus_1_squared_l154_154381


namespace symmetric_line_x_axis_l154_154559

theorem symmetric_line_x_axis (x y : ℝ) : 
  let P := (x, y)
  let P' := (x, -y)
  (3 * x - 4 * y + 5 = 0) →  
  (3 * x + 4 * -y + 5 = 0) :=
by 
  sorry

end symmetric_line_x_axis_l154_154559


namespace complement_union_l154_154203

def is_pos_int_less_than_9 (x : ℕ) : Prop := x > 0 ∧ x < 9

def U : Set ℕ := {x | is_pos_int_less_than_9 x}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union :
  (U \ (M ∪ N)) = {2, 4, 8} :=
by
  sorry

end complement_union_l154_154203


namespace tickets_sold_in_total_l154_154584

def total_tickets
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ) : ℕ :=
  adult_tickets + student_tickets

theorem tickets_sold_in_total 
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ)
    (h1 : adult_price = 6)
    (h2 : student_price = 3)
    (h3 : total_revenue = 3846)
    (h4 : adult_tickets = 410)
    (h5 : student_tickets = 436) :
  total_tickets adult_price student_price total_revenue adult_tickets student_tickets = 846 :=
by
  sorry

end tickets_sold_in_total_l154_154584


namespace smallest_arithmetic_mean_divisible_product_l154_154400

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l154_154400


namespace Jenny_has_6_cards_l154_154232

variable (J : ℕ)

noncomputable def Jenny_number := J
noncomputable def Orlando_number := J + 2
noncomputable def Richard_number := 3 * (J + 2)
noncomputable def Total_number := J + (J + 2) + 3 * (J + 2)

theorem Jenny_has_6_cards
  (h1 : Orlando_number J = J + 2)
  (h2 : Richard_number J = 3 * (J + 2))
  (h3 : Total_number J = 38) : J = 6 :=
by
  sorry

end Jenny_has_6_cards_l154_154232


namespace total_value_is_155_l154_154233

def coin_count := 20
def silver_coin_count := 10
def silver_coin_value_total := 30
def gold_coin_count := 5
def regular_coin_value := 1

def silver_coin_value := silver_coin_value_total / 4
def gold_coin_value := 2 * silver_coin_value

def total_silver_value := silver_coin_count * silver_coin_value
def total_gold_value := gold_coin_count * gold_coin_value
def regular_coin_count := coin_count - (silver_coin_count + gold_coin_count)
def total_regular_value := regular_coin_count * regular_coin_value

def total_collection_value := total_silver_value + total_gold_value + total_regular_value

theorem total_value_is_155 : total_collection_value = 155 := 
by
  sorry

end total_value_is_155_l154_154233


namespace intercepts_equal_l154_154513

theorem intercepts_equal (m : ℝ) :
  (∃ x y: ℝ, mx - y - 3 - m = 0 ∧ y ≠ 0 ∧ (x = 3 + m ∧ y = -(3 + m))) ↔ (m = -3 ∨ m = -1) :=
by 
  sorry

end intercepts_equal_l154_154513


namespace simple_interest_borrowed_rate_l154_154149

theorem simple_interest_borrowed_rate
  (P_borrowed P_lent : ℝ)
  (n_years : ℕ)
  (gain_per_year : ℝ)
  (simple_interest_lent_rate : ℝ)
  (SI_lending : ℝ := P_lent * simple_interest_lent_rate * n_years / 100)
  (total_gain : ℝ := gain_per_year * n_years) :
  SI_lending = 1000 →
  total_gain = 100 →
  ∀ (SI_borrowing : ℝ), SI_borrowing = SI_lending - total_gain →
  ∀ (R_borrowed : ℝ), SI_borrowing = P_borrowed * R_borrowed * n_years / 100 →
  R_borrowed = 9 := 
by
  sorry

end simple_interest_borrowed_rate_l154_154149


namespace find_interval_for_a_l154_154174

-- Define the system of equations as a predicate
def system_of_equations (a x y z : ℝ) : Prop := 
  x + y + z = 0 ∧ x * y + y * z + a * z * x = 0

-- Define the condition that (0, 0, 0) is the only solution
def unique_solution (a : ℝ) : Prop :=
  ∀ x y z : ℝ, system_of_equations a x y z → x = 0 ∧ y = 0 ∧ z = 0

-- Rewrite the proof problem as a Lean statement
theorem find_interval_for_a :
  ∀ a : ℝ, unique_solution a ↔ 0 < a ∧ a < 4 :=
by
  sorry

end find_interval_for_a_l154_154174


namespace number_of_zookeepers_12_l154_154594

theorem number_of_zookeepers_12 :
  let P := 30 -- number of penguins
  let Zr := 22 -- number of zebras
  let T := 8 -- number of tigers
  let A_heads := P + Zr + T -- total number of animal heads
  let A_feet := (2 * P) + (4 * Zr) + (4 * T) -- total number of animal feet
  ∃ Z : ℕ, -- number of zookeepers
  (A_heads + Z) + 132 = A_feet + (2 * Z) → Z = 12 :=
by
  sorry

end number_of_zookeepers_12_l154_154594


namespace license_plate_count_l154_154065

def num_license_plates : Nat :=
  let letters := 26 -- choices for each of the first two letters
  let primes := 4 -- choices for prime digits
  let composites := 4 -- choices for composite digits
  letters * letters * (primes * composites * 2)

theorem license_plate_count : num_license_plates = 21632 :=
  by
  sorry

end license_plate_count_l154_154065


namespace necessarily_negative_l154_154719

theorem necessarily_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := 
sorry

end necessarily_negative_l154_154719


namespace trajectory_of_P_distance_EF_l154_154230

section Exercise

-- Define the curve C in polar coordinates
def curve_C (ρ' θ: ℝ) : Prop :=
  ρ' * Real.cos (θ + Real.pi / 4) = 1

-- Define the relationship between OP and OQ
def product_OP_OQ (ρ ρ' : ℝ) : Prop :=
  ρ * ρ' = Real.sqrt 2

-- Define the trajectory of point P (C1) as the goal
theorem trajectory_of_P (ρ θ: ℝ) (hC: curve_C ρ' θ) (hPQ: product_OP_OQ ρ ρ') :
  ρ = Real.cos θ - Real.sin θ :=
sorry

-- Define the coordinates and the curve C2
def curve_C2 (x y t: ℝ) : Prop :=
  x = 0.5 - Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

-- Define the line l in Cartesian coordinates that needs to be converted to polar
def line_l (x y: ℝ) : Prop :=
  y = -Real.sqrt 3 * x

-- Define the distance |EF| to be proved
theorem distance_EF (θ ρ_1 ρ_2: ℝ) (hx: curve_C2 (0.5 - Real.sqrt 2 / 2 * t) (Real.sqrt 2 / 2 * t) t)
  (hE: θ = 2 * Real.pi / 3 ∨ θ = -Real.pi / 3)
  (hρ1: ρ_1 = Real.cos (-Real.pi / 3) - Real.sin (-Real.pi / 3))
  (hρ2: ρ_2 = 0.5 * (Real.sqrt 3 + 1)) :
  |ρ_1 + ρ_2| = Real.sqrt 3 + 1 :=
sorry

end Exercise

end trajectory_of_P_distance_EF_l154_154230


namespace businessmen_drink_neither_l154_154310

theorem businessmen_drink_neither : 
  ∀ (total coffee tea both : ℕ), 
    total = 30 → 
    coffee = 15 → 
    tea = 13 → 
    both = 8 → 
    total - (coffee - both + tea - both + both) = 10 := 
by 
  intros total coffee tea both h_total h_coffee h_tea h_both
  sorry

end businessmen_drink_neither_l154_154310


namespace two_digit_subtraction_pattern_l154_154111

theorem two_digit_subtraction_pattern (a b : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) := 
by
  sorry

end two_digit_subtraction_pattern_l154_154111


namespace correct_reaction_equation_l154_154518

noncomputable def reaction_equation (vA vB vC : ℝ) : Prop :=
  vB = 3 * vA ∧ 3 * vC = 2 * vB

theorem correct_reaction_equation (vA vB vC : ℝ) (h : reaction_equation vA vB vC) :
  ∃ (α β γ : ℕ), α = 1 ∧ β = 3 ∧ γ = 2 :=
sorry

end correct_reaction_equation_l154_154518


namespace log_evaluation_l154_154022

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l154_154022


namespace pq_square_eq_169_div_4_l154_154992

-- Defining the quadratic equation and the condition on solutions p and q.
def quadratic_eq (x : ℚ) : Prop := 2 * x^2 + 7 * x - 15 = 0

-- Defining the specific solutions p and q.
def p : ℚ := 3 / 2
def q : ℚ := -5

-- The main theorem stating that (p - q)^2 = 169 / 4 given the conditions.
theorem pq_square_eq_169_div_4 (hp : quadratic_eq p) (hq : quadratic_eq q) : (p - q) ^ 2 = 169 / 4 :=
by
  -- Proof is omitted using sorry
  sorry

end pq_square_eq_169_div_4_l154_154992


namespace log_base_3_of_9_cubed_l154_154023

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l154_154023


namespace probability_of_rain_at_least_once_l154_154188

theorem probability_of_rain_at_least_once 
  (P_sat : ℝ) (P_sun : ℝ) (P_mon : ℝ)
  (h_sat : P_sat = 0.30)
  (h_sun : P_sun = 0.60)
  (h_mon : P_mon = 0.50) :
  (1 - (1 - P_sat) * (1 - P_sun) * (1 - P_mon)) * 100 = 86 :=
by
  rw [h_sat, h_sun, h_mon]
  sorry

end probability_of_rain_at_least_once_l154_154188


namespace gcd_five_pentagonal_and_n_plus_one_l154_154187

-- Definition of the nth pentagonal number
def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n - 1)) / 2

-- Proof statement
theorem gcd_five_pentagonal_and_n_plus_one (n : ℕ) (h : 0 < n) : 
  Nat.gcd (5 * pentagonal_number n) (n + 1) = 1 :=
sorry

end gcd_five_pentagonal_and_n_plus_one_l154_154187


namespace complex_ratio_real_l154_154661

theorem complex_ratio_real (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ z : ℂ, z = a + b * Complex.I ∧ (z * (1 - 2 * Complex.I)).im = 0) :
  a / b = 1 / 2 :=
sorry

end complex_ratio_real_l154_154661


namespace number_of_child_workers_l154_154908

-- Define the conditions
def number_of_male_workers : ℕ := 20
def number_of_female_workers : ℕ := 15
def wage_per_male : ℕ := 35
def wage_per_female : ℕ := 20
def wage_per_child : ℕ := 8
def average_wage : ℕ := 26

-- Define the proof goal
theorem number_of_child_workers (C : ℕ) : 
  ((number_of_male_workers * wage_per_male +
    number_of_female_workers * wage_per_female +
    C * wage_per_child) /
   (number_of_male_workers + number_of_female_workers + C) = average_wage) → 
  C = 5 :=
by 
  sorry

end number_of_child_workers_l154_154908


namespace average_rst_l154_154684

variable (r s t : ℝ)

theorem average_rst
  (h : (4 / 3) * (r + s + t) = 12) :
  (r + s + t) / 3 = 3 :=
sorry

end average_rst_l154_154684


namespace max_value_m_l154_154193

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end max_value_m_l154_154193


namespace sum_p_q_r_l154_154165

def b (n : ℕ) : ℕ :=
if n < 1 then 0 else
if n < 2 then 2 else
if n < 4 then 4 else
if n < 7 then 6
else 6 -- Continue this pattern for illustration; an infinite structure would need proper handling for all n.

noncomputable def p := 2
noncomputable def q := 0
noncomputable def r := 0

theorem sum_p_q_r : p + q + r = 2 :=
by sorry

end sum_p_q_r_l154_154165


namespace range_of_f_l154_154790

noncomputable def f (x y z : ℝ) := ((x * y + y * z + z * x) * (x + y + z)) / ((x + y) * (y + z) * (z + x))

theorem range_of_f :
  ∃ x y z : ℝ, (0 < x ∧ 0 < y ∧ 0 < z) ∧ f x y z = r ↔ 1 ≤ r ∧ r ≤ 9 / 8 :=
sorry

end range_of_f_l154_154790


namespace area_of_overlap_l154_154457

theorem area_of_overlap 
  (len1 len2 : ℕ) (area_left only_left_area : ℚ) (area_right only_right_area : ℚ) (w : ℚ)
  (h_len1 : len1 = 9) (h_len2 : len2 = 7) (h_only_left_area : only_left_area = 27) 
  (h_only_right_area : only_right_area = 18) (h_w : w > 0)
  (h_area_left : area_left = only_left_area + (w * 1))
  (h_area_right : area_right = only_right_area + (w * 1))
  (h_ratio : (w * len1) / (w * len2) = 9 / 7) : 
  (13.5) :=
by
  sorry

end area_of_overlap_l154_154457


namespace bus_full_people_could_not_take_l154_154293

-- Definitions of the given conditions
def bus_capacity : ℕ := 80
def first_pickup_people : ℕ := (3 / 5) * bus_capacity
def people_exit_at_second_pickup : ℕ := 25
def people_waiting_at_second_pickup : ℕ := 90

-- The Lean statement to prove the number of people who could not take the bus
theorem bus_full_people_could_not_take (h1 : bus_capacity = 80)
                                       (h2 : first_pickup_people = 48)
                                       (h3 : people_exit_at_second_pickup = 25)
                                       (h4 : people_waiting_at_second_pickup = 90) :
  90 - (80 - (48 - 25)) = 33 :=
by
  sorry

end bus_full_people_could_not_take_l154_154293


namespace triangle_incircle_ratio_l154_154299

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end triangle_incircle_ratio_l154_154299


namespace probability_point_above_cubic_curve_l154_154555

theorem probability_point_above_cubic_curve :
  let single_digit_positives := {n : ℕ | 1 ≤ n ∧ n ≤ 9}
  let is_above_cubic_curve (a b : ℕ) : Prop :=
    ∀ x : ℕ, b > a * x^3 - b * x^2
  let valid_points : ℕ :=
    finset.choose (λ (a b : ℕ), a ∈ single_digit_positives ∧ b ∈ single_digit_positives ∧ is_above_cubic_curve a b) 81
  valid_points = 16 :=
sorry

end probability_point_above_cubic_curve_l154_154555


namespace abc_inequality_l154_154718

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) :=
by sorry

end abc_inequality_l154_154718


namespace ancient_china_pentatonic_scale_l154_154081

theorem ancient_china_pentatonic_scale (a : ℝ) (h : a * (2/3) * (4/3) * (2/3) = 32) : a = 54 :=
by
  sorry

end ancient_china_pentatonic_scale_l154_154081


namespace coordinates_of_B_l154_154079

/--
Given point A with coordinates (2, -3) and line segment AB parallel to the x-axis,
and the length of AB being 4, prove that the coordinates of point B are either (-2, -3)
or (6, -3).
-/
theorem coordinates_of_B (x1 y1 : ℝ) (d : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : d = 4) (hx : 0 ≤ d) :
  ∃ x2 : ℝ, ∃ y2 : ℝ, (y2 = y1) ∧ ((x2 = x1 + d) ∨ (x2 = x1 - d)) :=
by
  sorry

end coordinates_of_B_l154_154079


namespace square_rem_1_mod_9_l154_154214

theorem square_rem_1_mod_9 (N : ℤ) (h : N % 9 = 1 ∨ N % 9 = 8) : (N * N) % 9 = 1 :=
by sorry

end square_rem_1_mod_9_l154_154214


namespace min_faces_n2_min_faces_n3_l154_154540

noncomputable def minimum_faces (n : ℕ) : ℕ := 
  if n = 2 then 2 
  else if n = 3 then 12 
  else sorry 

theorem min_faces_n2 : minimum_faces 2 = 2 := 
  by 
  simp [minimum_faces]

theorem min_faces_n3 : minimum_faces 3 = 12 := 
  by 
  simp [minimum_faces]

end min_faces_n2_min_faces_n3_l154_154540


namespace find_B_l154_154066

structure Point where
  x : Int
  y : Int

def vector_sub (p1 p2 : Point) : Point :=
  ⟨p1.x - p2.x, p1.y - p2.y⟩

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-1, 2⟩
def BA : Point := ⟨3, 3⟩
def B : Point := ⟨-4, -1⟩

theorem find_B :
  vector_sub A BA = B :=
by
  sorry

end find_B_l154_154066


namespace log_diff_lt_one_l154_154662

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_diff_lt_one
  (b c x : ℝ)
  (h_eq_sym : ∀ (t : ℝ), (t - 2)^2 + b * (t - 2) + c = (t + 2)^2 + b * (t + 2) + c)
  (h_f_zero_pos : (0)^2 + b * (0) + c > 0)
  (m n : ℝ)
  (h_fm_0 : m^2 + b * m + c = 0)
  (h_fn_0 : n^2 + b * n + c = 0)
  (h_m_ne_n : m ≠ n)
  : log_base 4 m - log_base (1/4) n < 1 :=
  sorry

end log_diff_lt_one_l154_154662


namespace additional_hours_q_l154_154447

variable (P Q : ℝ)

theorem additional_hours_q (h1 : P = 1.5 * Q) 
                           (h2 : P = Q + 8) 
                           (h3 : 480 / P = 20):
  (480 / Q) - (480 / P) = 10 :=
by
  sorry

end additional_hours_q_l154_154447


namespace adrian_water_amount_l154_154153

theorem adrian_water_amount
  (O S W : ℕ) 
  (h1 : S = 3 * O)
  (h2 : W = 5 * S)
  (h3 : O = 4) : W = 60 :=
by
  sorry

end adrian_water_amount_l154_154153


namespace inequality_proof_l154_154867

theorem inequality_proof (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^2 - b * c) / (2 * a^2 + b * c) + (b^2 - c * a) / (2 * b^2 + c * a) + (c^2 - a * b) / (2 * c^2 + a * b) ≤ 0 :=
sorry

end inequality_proof_l154_154867


namespace isosceles_triangle_perimeter_l154_154340

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end isosceles_triangle_perimeter_l154_154340


namespace circle_equation_l154_154051

variable (x y : ℝ)

def center : ℝ × ℝ := (4, -6)
def radius : ℝ := 3

theorem circle_equation : (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x - 4)^2 + (y + 6)^2 = 9 :=
by
  sorry

end circle_equation_l154_154051


namespace friedahops_l154_154660

theorem friedahops (
    P : ℝ :=
    let p := 1/5 in
    let S := { (i, j) | (i = 2 ∧ j = 1) ∨ (i = 2 ∧ j = 2) ∨ (i = 2 ∧ j = 3) } in
    let E := { (i, j) | i = 1 ∨ i = 4 ∨ j = 1 ∨ j = 4 } in
    1/4 * (p + p + p + 0 + 0) ^ 5) =
  605/625 :=
sorry

end friedahops_l154_154660


namespace find_number_l154_154077

theorem find_number (x : ℝ) (h : 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30) : x = 66 :=
by
  sorry

end find_number_l154_154077


namespace pizza_fraction_eaten_l154_154130

-- The total fractional part of the pizza eaten after six trips
theorem pizza_fraction_eaten : 
  ∑ i in (finset.range 6), (1 / 3) ^ (i + 1) = 364 / 729 :=
by
  sorry

end pizza_fraction_eaten_l154_154130


namespace unique_solution_for_k_l154_154472

theorem unique_solution_for_k : 
  ∃! k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, (x + 3) / (k * x - 2) = x ↔ x = -2) :=
by
  sorry

end unique_solution_for_k_l154_154472


namespace lcm_gcd_pairs_l154_154906

theorem lcm_gcd_pairs (a b : ℕ) :
  (lcm a b + gcd a b = (a * b) / 5) ↔
  (a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6) :=
sorry

end lcm_gcd_pairs_l154_154906


namespace hyperbola_asymptotes_l154_154503

theorem hyperbola_asymptotes :
  ∀ x y : ℝ, x^2 - y^2 / 4 = 1 → (y = 2 * x ∨ y = -2 * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l154_154503


namespace rancher_cows_l154_154305

theorem rancher_cows (H C : ℕ) (h1 : C = 5 * H) (h2 : C + H = 168) : C = 140 := by
  sorry

end rancher_cows_l154_154305


namespace willy_days_to_finish_series_l154_154281

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def days_to_finish (total_episodes : ℕ) (episodes_per_day : ℕ) : ℕ :=
  total_episodes / episodes_per_day

theorem willy_days_to_finish_series : 
  total_episodes 3 20 = 60 → 
  days_to_finish 60 2 = 30 :=
by
  intros h1
  rw [h1]
  rfl

end willy_days_to_finish_series_l154_154281


namespace product_is_even_l154_154862

-- Definitions are captured from conditions

noncomputable def is_permutation {α : Type*} [DecidableEq α] (l1 l2 : List α) : Prop :=
  l1.perm l2

theorem product_is_even (a : ℕ → ℕ) :
  (is_permutation (List.range 2015) (List.ofFn (λ i, a i + 2014))) →
  Even (Finset.univ.prod (λ i : Fin 2015, a i - i.val.succ)) :=
by
  sorry

end product_is_even_l154_154862


namespace dads_strawberries_l154_154536

variable (M D : ℕ)

theorem dads_strawberries (h1 : M + D = 22) (h2 : M = 36) (h3 : D ≤ 22) :
  D + 30 = 46 :=
by
  sorry

end dads_strawberries_l154_154536


namespace find_ending_number_l154_154470

def ending_number (n : ℕ) : Prop :=
  18 < n ∧ n % 7 = 0 ∧ ((21 + n) / 2 : ℝ) = 38.5

theorem find_ending_number : ending_number 56 :=
by
  unfold ending_number
  sorry

end find_ending_number_l154_154470


namespace smallest_arithmetic_mean_l154_154412

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l154_154412


namespace count_three_element_subsets_l154_154477

-- Define the context and required definitions based on conditions
def is_arithmetic_mean (a b c : Nat) : Prop :=
  b = (a + c) / 2

-- Define the main function that calculates a_n
def a_n (n : Nat) : Nat :=
  if n < 3 then 0 else
  let floor_half := Nat.floor (n/2 : ℚ) in
  ((n - 1) * n / 4) - floor_half / 2 + 1

-- State the theorem that needs to be proved
theorem count_three_element_subsets (n : Nat) (h : n ≥ 3) :
  a_n n = (1/2) * ((n-1) * n / 2 - (Nat.floor (n/2 : ℚ))) + 1 :=
sorry

end count_three_element_subsets_l154_154477


namespace final_cost_correct_l154_154162

def dozen_cost : ℝ := 18
def num_dozen : ℝ := 2.5
def discount_rate : ℝ := 0.15

def cost_before_discount : ℝ := num_dozen * dozen_cost
def discount_amount : ℝ := discount_rate * cost_before_discount

def final_cost : ℝ := cost_before_discount - discount_amount

theorem final_cost_correct : final_cost = 38.25 := by
  -- The proof would go here, but we just provide the statement.
  sorry

end final_cost_correct_l154_154162


namespace smallest_positive_period_of_f_minimum_value_of_f_in_interval_l154_154204

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def f (x m : ℝ) : ℝ := (vec_a x).1 * vec_b.1 + (vec_a x).2 * vec_b.2 + m

theorem smallest_positive_period_of_f :
  ∀ (x : ℝ) (m : ℝ), ∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) m = f x m) → p = Real.pi := 
sorry

theorem minimum_value_of_f_in_interval :
  ∀ (x m : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → ∃ m : ℝ, (∀ x : ℝ, f x m ≥ 5) ∧ m = 5 + Real.sqrt 3 :=
sorry

end smallest_positive_period_of_f_minimum_value_of_f_in_interval_l154_154204


namespace intervals_of_monotonicity_min_max_of_g_l154_154673

noncomputable def f (x : ℝ) : ℝ :=
  1 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * Real.sin x ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x - (Real.pi / 6))

theorem intervals_of_monotonicity (k : ℤ) :
  (∃ a b : ℝ, k * Real.pi - (Real.pi / 3) ≤ a ∧ a ≤ k * Real.pi + (Real.pi / 6) ∧ 
              k * Real.pi + (Real.pi / 6) ≤ b ∧ b ≤ k * Real.pi + (Real.pi / 3) ∧ 
              ∀ x, a ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 6) → increasing_on f [a, k * Real.pi + (Real.pi / 6)] ∧ 
                      k * Real.pi + (Real.pi / 6) ≤ x ∧ x ≤ b → decreasing_on f [k * Real.pi + (Real.pi / 6), b]) :=
sorry

theorem min_max_of_g :
  ∀ x ∈ Icc (-Real.pi / 2) 0, g x ∈ Icc (-2 : ℝ) 1 :=
sorry

end intervals_of_monotonicity_min_max_of_g_l154_154673


namespace red_pairs_count_l154_154366

def num_green_students : Nat := 63
def num_red_students : Nat := 69
def total_pairs : Nat := 66
def num_green_pairs : Nat := 27

theorem red_pairs_count : 
  (num_red_students - (num_green_students - num_green_pairs * 2)) / 2 = 30 := 
by sorry

end red_pairs_count_l154_154366


namespace opposite_of_neg_one_third_l154_154576

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end opposite_of_neg_one_third_l154_154576


namespace new_person_age_l154_154595

theorem new_person_age (T : ℕ) : 
  (T / 10) = ((T - 46 + A) / 10) + 3 → (A = 16) :=
by
  sorry

end new_person_age_l154_154595


namespace smallest_arithmetic_mean_divisible_1111_l154_154405

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l154_154405


namespace sum_of_m_and_n_l154_154337

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6 * m + 10 * n + 34 = 0) : m + n = -2 := 
sorry

end sum_of_m_and_n_l154_154337


namespace fraction_ratio_l154_154967

variable (M Q P N R : ℝ)

theorem fraction_ratio (h1 : M = 0.40 * Q)
                       (h2 : Q = 0.25 * P)
                       (h3 : N = 0.40 * R)
                       (h4 : R = 0.75 * P) :
  M / N = 1 / 3 := 
by
  -- proof steps can be provided here
  sorry

end fraction_ratio_l154_154967


namespace half_vector_AB_l154_154335

-- Define vectors MA and MB
def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

-- Define the proof statement 
theorem half_vector_AB : (1 / 2 : ℝ) • (MB - MA) = (2, 1) :=
by sorry

end half_vector_AB_l154_154335


namespace consistent_values_for_a_l154_154054

def eq1 (x a : ℚ) : Prop := 10 * x^2 + x - a - 11 = 0
def eq2 (x a : ℚ) : Prop := 4 * x^2 + (a + 4) * x - 3 * a - 8 = 0

theorem consistent_values_for_a : ∃ x, (eq1 x 0 ∧ eq2 x 0) ∨ (eq1 x (-2) ∧ eq2 x (-2)) ∨ (eq1 x (54) ∧ eq2 x (54)) :=
by
  sorry

end consistent_values_for_a_l154_154054


namespace min_value_l154_154182

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end min_value_l154_154182


namespace triangle_side_a_l154_154338

theorem triangle_side_a (a : ℝ) (h1 : 4 < a) (h2 : a < 10) : a = 8 :=
  by
  sorry

end triangle_side_a_l154_154338


namespace smallest_composite_proof_l154_154649

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l154_154649


namespace find_ordered_pair_l154_154481

theorem find_ordered_pair (x y : ℝ) (h : (x - 2 * y)^2 + (y - 1)^2 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end find_ordered_pair_l154_154481


namespace number_of_valid_three_digit_numbers_l154_154683

def valid_three_digit_numbers : Nat :=
  -- Proving this will be the task: showing that there are precisely 24 such numbers
  24

theorem number_of_valid_three_digit_numbers : valid_three_digit_numbers = 24 :=
by
  -- Proof would go here.
  sorry

end number_of_valid_three_digit_numbers_l154_154683


namespace rita_remaining_money_l154_154873

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l154_154873


namespace expression_equals_answer_l154_154186

noncomputable def evaluate_expression : ℚ :=
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 +
  (2013^2 * 2014 - 2015) / Nat.factorial 2014

theorem expression_equals_answer :
  evaluate_expression = 
  1 / Nat.factorial 2009 + 
  1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 
  1 / Nat.factorial 2014 :=
by
  sorry

end expression_equals_answer_l154_154186


namespace solve_for_x_l154_154002

theorem solve_for_x : ∃ x : ℝ, (6 * x) / 1.5 = 3.8 ∧ x = 0.95 := by
  use 0.95
  exact ⟨by norm_num, by norm_num⟩

end solve_for_x_l154_154002


namespace find_a_2013_l154_154524

def sequence_a (n : ℕ) : ℤ :=
  if n = 0 then 2
  else if n = 1 then 5
  else sequence_a (n - 1) - sequence_a (n - 2)

theorem find_a_2013 :
  sequence_a 2013 = 3 :=
sorry

end find_a_2013_l154_154524


namespace power_mul_eq_l154_154163

variable (a : ℝ)

theorem power_mul_eq :
  (-a)^2 * a^4 = a^6 :=
by sorry

end power_mul_eq_l154_154163


namespace minimize_travel_time_l154_154158

-- Definitions and conditions
def grid_size : ℕ := 7
def mid_point : ℕ := (grid_size + 1) / 2
def is_meeting_point (p : ℕ × ℕ) : Prop := 
  p = (mid_point, mid_point)

-- Main theorem statement to be proven
theorem minimize_travel_time : 
  ∃ (p : ℕ × ℕ), is_meeting_point p ∧
  (∀ (q : ℕ × ℕ), is_meeting_point q → p = q) :=
sorry

end minimize_travel_time_l154_154158


namespace geom_seq_general_formula_sum_first_n_terms_formula_l154_154952

namespace GeometricArithmeticSequences

def geom_seq_general (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
  a_n 1 = 1 ∧ (2 * a_n 3 = a_n 2) → a_n n = 1 / (2 ^ (n - 1))

def sum_first_n_terms (a_n b_n : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (n : ℕ) : Prop :=
  b_n 1 = 2 ∧ S_n 3 = b_n 2 + 6 → 
  T_n n = 6 - (n + 3) / (2 ^ (n - 1))

theorem geom_seq_general_formula :
  ∀ a_n : ℕ → ℝ, ∀ n : ℕ, geom_seq_general a_n n :=
by sorry

theorem sum_first_n_terms_formula :
  ∀ a_n b_n : ℕ → ℝ, ∀ S_n T_n : ℕ → ℝ, ∀ n : ℕ, sum_first_n_terms a_n b_n S_n T_n n :=
by sorry

end GeometricArithmeticSequences

end geom_seq_general_formula_sum_first_n_terms_formula_l154_154952


namespace Billie_has_2_caps_l154_154101

-- Conditions as definitions in Lean
def Sammy_caps : ℕ := 8
def Janine_caps : ℕ := Sammy_caps - 2
def Billie_caps : ℕ := Janine_caps / 3

-- Problem statement to prove
theorem Billie_has_2_caps : Billie_caps = 2 := by
  sorry

end Billie_has_2_caps_l154_154101


namespace strips_overlap_area_l154_154456

theorem strips_overlap_area 
  (length_total : ℝ) 
  (length_left : ℝ) 
  (length_right : ℝ) 
  (area_only_left : ℝ) 
  (area_only_right : ℝ) 
  (length_total_eq : length_total = 16) 
  (length_left_eq : length_left = 9) 
  (length_right_eq : length_right = 7) 
  (area_only_left_eq : area_only_left = 27) 
  (area_only_right_eq : area_only_right = 18) :
  ∃ S : ℝ, (27 + S) / (18 + S) = (9 / 7) ∧ 2 * S = 27 :=
begin
  use 13.5,
  split,
  {
    -- Show proportional relationship holds
    sorry
  },
  {
    -- Show 2 * S = 27
    sorry
  }
end

end strips_overlap_area_l154_154456


namespace trig_identity_tan_solutions_l154_154095

open Real

theorem trig_identity_tan_solutions :
  ∃ α β : ℝ, (tan α) * (tan β) = -3 ∧ (tan α) + (tan β) = 3 ∧
  abs (sin (α + β) ^ 2 - 3 * sin (α + β) * cos (α + β) - 3 * cos (α + β) ^ 2) = 3 :=
by
  have: ∀ x : ℝ, x^2 - 3*x - 3 = 0 → x = (3 + sqrt 21) / 2 ∨ x = (3 - sqrt 21) / 2 := sorry
  sorry

end trig_identity_tan_solutions_l154_154095


namespace valentines_cards_count_l154_154376

theorem valentines_cards_count (x y : ℕ) (h1 : x * y = x + y + 30) : x * y = 64 :=
by {
    sorry
}

end valentines_cards_count_l154_154376


namespace bells_toll_together_l154_154450

noncomputable def LCM (a b : Nat) : Nat := (a * b) / (Nat.gcd a b)

theorem bells_toll_together :
  let intervals := [2, 4, 6, 8, 10, 12]
  let lcm := intervals.foldl LCM 1
  lcm = 120 →
  let duration := 30 * 60 -- 1800 seconds
  let tolls := duration / lcm
  tolls + 1 = 16 :=
by
  sorry

end bells_toll_together_l154_154450


namespace other_continents_passengers_l154_154844

def passengers_from_other_continents (T N_A E A As : ℕ) : ℕ := T - (N_A + E + A + As)

theorem other_continents_passengers :
  passengers_from_other_continents 108 (108 / 12) (108 / 4) (108 / 9) (108 / 6) = 42 :=
by
  -- Proof goes here
  sorry

end other_continents_passengers_l154_154844


namespace exercise_habits_related_to_gender_l154_154522

variable (n : Nat) (a b c d : Nat)
variable (alpha : Real)
variable (k_0 : Real)

-- Assume the conditions from the problem
axiom h_n : n = 100
axiom h1 : a + b + c + d = n
axiom h2 : a = 35
axiom h3 : b = 15
axiom h4 : c = 25
axiom h5 : d = 25
axiom h6 : alpha = 0.1
axiom h7 : k_0 = 2.706

-- Define the chi-square statistic
def chi_square : Real := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Prove that the students' regular exercise habits are related to gender with 90% confidence
theorem exercise_habits_related_to_gender :
  chi_square ≥ k_0 :=
by
  sorry

end exercise_habits_related_to_gender_l154_154522


namespace sin_15_deg_eq_l154_154171

theorem sin_15_deg_eq : 
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
by
  -- conditions
  have h1 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  have h4 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  
  -- proof
  sorry

end sin_15_deg_eq_l154_154171


namespace find_constants_l154_154480

theorem find_constants (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 10 ∧ x ≠ -5 → (8 * x - 3) / (x^2 - 5 * x - 50) = A / (x - 10) + B / (x + 5)) 
  → (A = 77 / 15 ∧ B = 43 / 15) := by 
  sorry

end find_constants_l154_154480


namespace not_function_age_height_l154_154012

theorem not_function_age_height (f : ℕ → ℝ) :
  ¬(∀ (a b : ℕ), a = b → f a = f b) := sorry

end not_function_age_height_l154_154012


namespace part_one_part_two_l154_154814
-- Import the Mathlib library for necessary definitions and theorems.

-- Define the conditions as hypotheses.
variables {a b c : ℝ} (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (1): State the inequality involving sums of reciprocals.
theorem part_one : (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 3 / 2 := 
by
  sorry

-- Part (2): Define the range for m in terms of the inequality condition.
theorem part_two : ∃m: ℝ, (∀a b c : ℝ, a + b + c = 3 → 0 < a → 0 < b → 0 < c → (-x^2 + m*x + 2 ≤ a^2 + b^2 + c^2)) ↔ (-2 ≤ m) ∧ (m ≤ 2) :=
by 
  sorry

end part_one_part_two_l154_154814


namespace roots_of_cubic_equation_l154_154531

theorem roots_of_cubic_equation 
  (k m : ℝ) 
  (h : ∀r1 r2 r3: ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 + r2 + r3 = 7 ∧ r1 * r2 * r3 = m ∧ (r1 * r2 + r2 * r3 + r1 * r3) = k) : 
  k + m = 22 := sorry

end roots_of_cubic_equation_l154_154531


namespace smallest_composite_no_prime_factors_less_than_15_l154_154636

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l154_154636


namespace john_twice_sam_in_years_l154_154527

noncomputable def current_age_sam : ℕ := 9
noncomputable def current_age_john : ℕ := 27

theorem john_twice_sam_in_years (Y : ℕ) :
  (current_age_john + Y = 2 * (current_age_sam + Y)) → Y = 9 := 
by 
  sorry

end john_twice_sam_in_years_l154_154527


namespace g_pi_over_4_eq_neg_sqrt2_over_4_l154_154094

noncomputable def g (x : Real) : Real := 
  Real.sqrt (5 * (Real.sin x)^4 + 4 * (Real.cos x)^2) - 
  Real.sqrt (6 * (Real.cos x)^4 + 4 * (Real.sin x)^2)

theorem g_pi_over_4_eq_neg_sqrt2_over_4 :
  g (Real.pi / 4) = - (Real.sqrt 2) / 4 := 
sorry

end g_pi_over_4_eq_neg_sqrt2_over_4_l154_154094


namespace polynomial_square_b_value_l154_154386

theorem polynomial_square_b_value (a b : ℚ) (h : ∃ (p q : ℚ), x^4 + 3 * x^3 + x^2 + a * x + b = (x^2 + p * x + q)^2) : 
  b = 25/64 := 
by 
  -- Proof steps go here
  sorry

end polynomial_square_b_value_l154_154386


namespace count_numbers_with_cube_root_lt_8_l154_154836

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l154_154836


namespace log_evaluation_l154_154021

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l154_154021


namespace x_y_ge_two_l154_154528

open Real

theorem x_y_ge_two (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : 
  x + y ≥ 2 ∧ (x + y = 2 → x = 1 ∧ y = 1) :=
by {
 sorry
}

end x_y_ge_two_l154_154528


namespace solve_fractional_equation_l154_154271

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end solve_fractional_equation_l154_154271


namespace adults_at_zoo_l154_154759

theorem adults_at_zoo (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : A = 51 :=
sorry

end adults_at_zoo_l154_154759


namespace factorize_expr_l154_154794

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end factorize_expr_l154_154794


namespace sum_series_fraction_l154_154479

open BigOperators

theorem sum_series_fraction :
  (∑ n in finset.range 6, 1 / (n + 1) / (n + 2)^2) = 204 / 1225 := sorry

end sum_series_fraction_l154_154479


namespace abs_value_sum_l154_154118

noncomputable def sin_theta_in_bounds (θ : ℝ) : Prop :=
  -1 ≤ Real.sin θ ∧ Real.sin θ ≤ 1

noncomputable def x_satisfies_log_eq (θ x : ℝ) : Prop :=
  Real.log x / Real.log 3 = 1 + Real.sin θ

theorem abs_value_sum (θ x : ℝ) (h1 : x_satisfies_log_eq θ x) (h2 : sin_theta_in_bounds θ) :
  |x - 1| + |x - 9| = 8 :=
sorry

end abs_value_sum_l154_154118


namespace highest_possible_N_l154_154519

/--
In a football tournament with 15 teams, each team played exactly once against every other team.
A win earns 3 points, a draw earns 1 point, and a loss earns 0 points.
We need to prove that the highest possible integer \( N \) such that there are at least 6 teams with at least \( N \) points is 34.
-/
theorem highest_possible_N : 
  ∃ (N : ℤ) (teams : Fin 15 → ℤ) (successfulTeams : Fin 6 → Fin 15),
    (∀ i j, i ≠ j → teams i + teams j ≤ 207) ∧ 
    (∀ k, k < 6 → teams (successfulTeams k) ≥ 34) ∧ 
    (∀ k, 0 ≤ teams k) ∧ 
    N = 34 := sorry

end highest_possible_N_l154_154519


namespace income_of_m_l154_154596

theorem income_of_m (M N O : ℝ)
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (M + O) / 2 = 5200) :
  M = 4000 :=
by
  -- sorry is used to skip the actual proof.
  sorry

end income_of_m_l154_154596


namespace general_term_min_sum_Sn_l154_154089

-- (I) Prove the general term formula for the arithmetic sequence
theorem general_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10) 
  (geometric_cond : (a 2 + 10) * (a 4 + 6) = (a 3 + 8) ^ 2) : 
  ∃ n : ℕ, a n = 2 * n - 12 :=
by
  sorry

-- (II) Prove the minimum value of the sum of the first n terms
theorem min_sum_Sn (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10)
  (general_term : ∀ n, a n = 2 * n - 12) : 
  ∃ n, S n = n * n - 11 * n ∧ S n = -30 :=
by
  sorry

end general_term_min_sum_Sn_l154_154089


namespace find_c_l154_154003

theorem find_c 
  (b c : ℝ) 
  (h1 : 4 = 2 * (1:ℝ)^2 + b * (1:ℝ) + c)
  (h2 : 4 = 2 * (5:ℝ)^2 + b * (5:ℝ) + c) : 
  c = 14 := 
sorry

end find_c_l154_154003


namespace message_forwarding_time_l154_154970

theorem message_forwarding_time :
  ∃ n : ℕ, (∀ m : ℕ, (∀ p : ℕ, (∀ q : ℕ, 1 + (2 * (2 ^ n)) - 1 = 2047)) ∧ n = 10) :=
sorry

end message_forwarding_time_l154_154970


namespace rita_remaining_amount_l154_154878

theorem rita_remaining_amount :
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  remaining_amount = initial_amount - total_cost
  in remaining_amount = 139 :=
by
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  let remaining_amount := initial_amount - total_cost
  show remaining_amount = 139
  sorry

end rita_remaining_amount_l154_154878


namespace average_salary_techs_l154_154851

noncomputable def total_salary := 20000
noncomputable def average_salary_all := 750
noncomputable def num_technicians := 5
noncomputable def average_salary_non_tech := 700
noncomputable def total_workers := 20

theorem average_salary_techs :
  (20000 - (num_technicians + average_salary_non_tech * (total_workers - num_technicians))) / num_technicians = 900 := by
  sorry

end average_salary_techs_l154_154851


namespace student_number_choice_l154_154462

theorem student_number_choice (x : ℤ) (h : 3 * x - 220 = 110) : x = 110 :=
by sorry

end student_number_choice_l154_154462


namespace dessert_menu_count_l154_154770

def Dessert : Type := {d : String // d = "cake" ∨ d = "pie" ∨ d = "ice cream" ∨ d = "pudding"}

def valid_menu (menu : Fin 7 → Dessert) : Prop :=
  (menu 0).1 ≠ (menu 1).1 ∧
  menu 1 = ⟨"ice cream", Or.inr (Or.inr (Or.inl rfl))⟩ ∧
  (menu 1).1 ≠ (menu 2).1 ∧
  (menu 2).1 ≠ (menu 3).1 ∧
  (menu 3).1 ≠ (menu 4).1 ∧
  (menu 4).1 ≠ (menu 5).1 ∧
  menu 5 = ⟨"cake", Or.inl rfl⟩ ∧
  (menu 5).1 ≠ (menu 6).1

def total_valid_menus : Nat :=
  4 * 1 * 3 * 3 * 3 * 1 * 3

theorem dessert_menu_count : ∃ (count : Nat), count = 324 ∧ count = total_valid_menus :=
  sorry

end dessert_menu_count_l154_154770


namespace probability_of_supporting_law_l154_154078

def prob_supports (p : ℝ) (k n : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_of_supporting_law : 
  prob_supports 0.6 2 5 = 0.2304 :=
sorry

end probability_of_supporting_law_l154_154078


namespace range_of_m_l154_154220

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < 3) ↔ (x / 3 < 1 - (x - 3) / 6 ∧ x < m)) → m ≥ 3 :=
by
  sorry

end range_of_m_l154_154220


namespace smallest_n_for_identity_l154_154184

noncomputable def rot_matrix := 
  ![![Real.cos (160 * Real.pi / 180), -Real.sin (160 * Real.pi / 180)], 
    ![Real.sin (160 * Real.pi / 180), Real.cos (160 * Real.pi / 180)]]  -- Define the rotation matrix for 160 degrees

def is_identity {n : ℕ} (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = Matrix.identity 2  -- Check if the matrix is the identity matrix

theorem smallest_n_for_identity :
  ∃ n : ℕ, 0 < n ∧ is_identity (rot_matrix ^ n) ∧ n = 9 := 
sorry

end smallest_n_for_identity_l154_154184


namespace ones_digit_of_prime_in_arithmetic_sequence_l154_154742

theorem ones_digit_of_prime_in_arithmetic_sequence (p q r : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) 
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4)
  (h : p > 5) : 
    (p % 10 = 3 ∨ p % 10 = 9) :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_l154_154742


namespace smallest_x_multiple_of_53_l154_154801

theorem smallest_x_multiple_of_53 :
  ∃ (x : ℕ), (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
sorry

end smallest_x_multiple_of_53_l154_154801


namespace box_contains_1_8_grams_child_ingests_0_1_grams_l154_154602

-- Define the conditions
def packet_weight : ℝ := 0.2
def packets_in_box : ℕ := 9
def half_a_packet : ℝ := 0.5

-- Prove that a box contains 1.8 grams of "acetaminophen"
theorem box_contains_1_8_grams : packets_in_box * packet_weight = 1.8 :=
by
  sorry

-- Prove that a child will ingest 0.1 grams of "acetaminophen" if they take half a packet
theorem child_ingests_0_1_grams : half_a_packet * packet_weight = 0.1 :=
by
  sorry

end box_contains_1_8_grams_child_ingests_0_1_grams_l154_154602


namespace round_robin_chess_l154_154226

/-- 
In a round-robin chess tournament, two boys and several girls participated. 
The boys together scored 8 points, while all the girls scored an equal number of points.
We are to prove that the number of girls could have participated in the tournament is 7 or 14,
given that a win is 1 point, a draw is 0.5 points, and a loss is 0 points.
-/
theorem round_robin_chess (n : ℕ) (x : ℚ) (h : 2 * n * x + 16 = n ^ 2 + 3 * n + 2) : n = 7 ∨ n = 14 :=
sorry

end round_robin_chess_l154_154226


namespace dexter_filled_fewer_boxes_with_football_cards_l154_154172

-- Conditions
def boxes_with_basketball_cards : ℕ := 9
def cards_per_basketball_box : ℕ := 15
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255

-- Definition of the main problem statement
def fewer_boxes_with_football_cards : Prop :=
  let basketball_cards := boxes_with_basketball_cards * cards_per_basketball_box
  let football_cards := total_cards - basketball_cards
  let boxes_with_football_cards := football_cards / cards_per_football_box
  boxes_with_basketball_cards - boxes_with_football_cards = 3

theorem dexter_filled_fewer_boxes_with_football_cards : fewer_boxes_with_football_cards :=
by
  sorry

end dexter_filled_fewer_boxes_with_football_cards_l154_154172


namespace quotient_of_numbers_l154_154048

noncomputable def larger_number : ℕ := 22
noncomputable def smaller_number : ℕ := 8

theorem quotient_of_numbers : (larger_number.toFloat / smaller_number.toFloat) = 2.75 := by
  sorry

end quotient_of_numbers_l154_154048


namespace rhombus_side_length_l154_154617

theorem rhombus_side_length (a b s K : ℝ)
  (h1 : b = 3 * a)
  (h2 : K = (1 / 2) * a * b)
  (h3 : s ^ 2 = (a / 2) ^ 2 + (3 * a / 2) ^ 2) :
  s = Real.sqrt (5 * K / 3) :=
by
  sorry

end rhombus_side_length_l154_154617


namespace new_person_weight_l154_154257

theorem new_person_weight (W : ℝ) (old_weight : ℝ) (increase_per_person : ℝ) (num_persons : ℕ)
  (h1 : old_weight = 68)
  (h2 : increase_per_person = 5.5)
  (h3 : num_persons = 5)
  (h4 : W = old_weight + increase_per_person * num_persons) :
  W = 95.5 :=
by
  sorry

end new_person_weight_l154_154257


namespace factorize_expr_l154_154793

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end factorize_expr_l154_154793


namespace find_principal_amount_l154_154258

noncomputable def principal_amount (difference : ℝ) (rate : ℝ) : ℝ :=
  let ci := rate / 2
  let si := rate
  difference / (ci ^ 2 - 1 - si)

theorem find_principal_amount :
  principal_amount 4.25 0.10 = 1700 :=
by 
  sorry

end find_principal_amount_l154_154258


namespace convert_4512_base8_to_base10_l154_154788

-- Definitions based on conditions
def base8_to_base10 (n : Nat) : Nat :=
  let d3 := 4 * 8^3
  let d2 := 5 * 8^2
  let d1 := 1 * 8^1
  let d0 := 2 * 8^0
  d3 + d2 + d1 + d0

-- The proof statement
theorem convert_4512_base8_to_base10 :
  base8_to_base10 4512 = 2378 :=
by
  -- proof goes here
  sorry

end convert_4512_base8_to_base10_l154_154788


namespace min_colors_for_G_l154_154164

open Nat

-- Define the graph as a structure with vertices and edge condition
structure Graph :=
  (V : Finset ℕ)
  (E : ℕ → ℕ → Prop)
  (edge_cond : ∀ i j ∈ V, E i j ↔ (i ∣ j .))

-- Define a specific graph G with vertices 1 to 1000 and edge condition
def G : Graph :=
  { V := (Finset.range 1000).map Nat.succ,
    E := λ i j, (i ∣ j),
    edge_cond := λ i j, Iff.rfl }

-- Define the proposition for the minimum number of colors required
def min_colors_needed (G : Graph) (n : ℕ) : Prop :=
  ∀ f : ∀ v ∈ G.V, ℕ, (∀ v1 v2 ∈ G.V, G.E v1 v2 → f v1 ≠ f v2) → ∃ (m ≤ n),
    (∀ v ∈ G.V, f v < m)

-- State the proof as a theorem statement
theorem min_colors_for_G : min_colors_needed G 10 :=
sorry

end min_colors_for_G_l154_154164


namespace saving_time_for_downpayment_l154_154242

def annual_salary : ℚ := 150000
def saving_rate : ℚ := 0.10
def house_cost : ℚ := 450000
def downpayment_rate : ℚ := 0.20

theorem saving_time_for_downpayment : 
  (downpayment_rate * house_cost) / (saving_rate * annual_salary) = 6 :=
by
  sorry

end saving_time_for_downpayment_l154_154242


namespace largest_side_l154_154086

-- Definitions of conditions from part (a)
def perimeter_eq (l w : ℝ) : Prop := 2 * l + 2 * w = 240
def area_eq (l w : ℝ) : Prop := l * w = 2880

-- The main proof statement
theorem largest_side (l w : ℝ) (h1 : perimeter_eq l w) (h2 : area_eq l w) : l = 72 ∨ w = 72 :=
by
  sorry

end largest_side_l154_154086


namespace solution_in_quadrant_I_l154_154931

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 
    3 * x - 4 * y = 6 ∧ 
    k * x + 2 * y = 8 ∧ 
    0 < x ∧ 
    0 < y) ↔ 
  -3 / 2 < k ∧ k < 4 :=
sorry

end solution_in_quadrant_I_l154_154931


namespace probability_X_eq_Y_l154_154619

open Real

theorem probability_X_eq_Y :
  let s := -15 * π / 2
  let t := 15 * π / 2
  (∀ (X Y : ℝ), s ≤ X ∧ X ≤ t ∧ s ≤ Y ∧ Y ≤ t ∧ (cos (sin X) = cos (sin Y)) → X = Y) →
  (∀ (X Y : ℝ), s ≤ X ∧ X ≤ t ∧ s ≤ Y ∧ Y ≤ t → set.prob (set_of (λ p : ℝ × ℝ, p.fst = p.snd)) (set.prod (Icc s t) (Icc s t)) = 15 / (225 * π^2)) :=
sorry

end probability_X_eq_Y_l154_154619


namespace problem_solution_l154_154372

theorem problem_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := 
  sorry

end problem_solution_l154_154372


namespace Clarissa_needs_to_bring_photos_l154_154322

variable (Cristina John Sarah Clarissa Total_slots : ℕ)

def photo_album_problem := Cristina = 7 ∧ John = 10 ∧ Sarah = 9 ∧ Total_slots = 40 ∧
  (Clarissa + Cristina + John + Sarah = Total_slots)

theorem Clarissa_needs_to_bring_photos (h : photo_album_problem 7 10 9 14 40) : Clarissa = 14 := by
  cases h with _ h, cases h with _ h, cases h with _ h, cases h with _ h, cases h
  sorry

end Clarissa_needs_to_bring_photos_l154_154322


namespace smallest_arithmetic_mean_divisible_1111_l154_154408

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l154_154408


namespace problem_l154_154091

theorem problem (a b c k : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hk : k ≠ 0)
  (h1 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 :=
by
  sorry

end problem_l154_154091


namespace Steve_bakes_more_apple_pies_l154_154552

def Steve_bakes (days_apple days_cherry pies_per_day : ℕ) : ℕ :=
  (days_apple * pies_per_day) - (days_cherry * pies_per_day)

theorem Steve_bakes_more_apple_pies :
  Steve_bakes 3 2 12 = 12 :=
by
  sorry

end Steve_bakes_more_apple_pies_l154_154552


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154403

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154403


namespace evaluate_expression_l154_154047

theorem evaluate_expression (b : ℚ) (h : b = 4 / 3) :
  (6 * b ^ 2 - 17 * b + 8) * (3 * b - 4) = 0 :=
by 
  -- Proof goes here
  sorry

end evaluate_expression_l154_154047


namespace tank_capacity_correct_l154_154378

-- Define rates and times for each pipe
def rate_a : ℕ := 200 -- in liters per minute
def rate_b : ℕ := 50 -- in liters per minute
def rate_c : ℕ := 25 -- in liters per minute

def time_a : ℕ := 1 -- pipe A open time in minutes
def time_b : ℕ := 2 -- pipe B open time in minutes
def time_c : ℕ := 2 -- pipe C open time in minutes

def cycle_time : ℕ := time_a + time_b + time_c -- total time for one cycle in minutes
def total_time : ℕ := 40 -- total time to fill the tank in minutes

-- Net water added in one cycle
def net_water_in_cycle : ℕ :=
  (rate_a * time_a) + (rate_b * time_b) - (rate_c * time_c)

-- Number of cycles needed to fill the tank
def number_of_cycles : ℕ :=
  total_time / cycle_time

-- Total capacity of the tank
def tank_capacity : ℕ :=
  number_of_cycles * net_water_in_cycle

-- The hypothesis to prove
theorem tank_capacity_correct :
  tank_capacity = 2000 :=
  by
    sorry

end tank_capacity_correct_l154_154378


namespace expand_polynomial_l154_154328

theorem expand_polynomial (x : ℝ) : 
  3 * (x - 2) * (x^2 + x + 1) = 3 * x^3 - 3 * x^2 - 3 * x - 6 :=
by
  sorry

end expand_polynomial_l154_154328


namespace minimum_distance_l154_154201

theorem minimum_distance (m n : ℝ) (a : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ 4) 
  (h3 : m * Real.sqrt (Real.log a - 1 / 4) + 2 * a + 1 / 2 * n = 0) : 
  Real.sqrt (m^2 + n^2) = 4 * Real.sqrt (Real.log 2) / Real.log 2 :=
sorry

end minimum_distance_l154_154201


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154414

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154414


namespace Clarissa_photos_needed_l154_154321

theorem Clarissa_photos_needed :
  (7 + 10 + 9 <= 40) → 40 - (7 + 10 + 9) = 14 :=
by
  sorry

end Clarissa_photos_needed_l154_154321


namespace compute_abc_l154_154990

theorem compute_abc (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_sum : a + b + c = 30) (h_frac : (1 : ℚ) / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) : a * b * c = 1920 :=
by sorry

end compute_abc_l154_154990


namespace log_three_nine_cubed_l154_154038

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l154_154038


namespace boys_less_than_two_fifths_total_l154_154907

theorem boys_less_than_two_fifths_total
  (n b g n1 n2 b1 b2 : ℕ)
  (h_total: n = b + g)
  (h_first_trip: b1 < 2 * n1 / 5)
  (h_second_trip: b2 < 2 * n2 / 5)
  (h_participation: b ≤ b1 + b2)
  (h_total_participants: n ≤ n1 + n2) :
  b < 2 * n / 5 := 
sorry

end boys_less_than_two_fifths_total_l154_154907


namespace universal_negation_example_l154_154263

theorem universal_negation_example :
  (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) →
  (¬ (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) = (∃ x : ℝ, x^2 - 3 * x + 1 > 0)) :=
by
  intro h
  sorry

end universal_negation_example_l154_154263


namespace distinct_triangles_in_3x3_grid_l154_154832

theorem distinct_triangles_in_3x3_grid : 
  let num_points := 9 
  let total_combinations := Nat.choose num_points 3 
  let degenerate_cases := 8
  total_combinations - degenerate_cases = 76 := 
by
  sorry

end distinct_triangles_in_3x3_grid_l154_154832


namespace opposite_neg_inv_three_l154_154569

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end opposite_neg_inv_three_l154_154569


namespace Q1_Intersection_Q1_Union_Q2_l154_154819

namespace Example

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Question 1: 
theorem Q1_Intersection (a : ℝ) (ha : a = -1) : 
  A ∩ B a = {x | -2 ≤ x ∧ x ≤ -1} :=
sorry

theorem Q1_Union (a : ℝ) (ha : a = -1) :
  A ∪ B a = {x | x ≤ 1 ∨ x ≥ 5} :=
sorry

-- Question 2:
theorem Q2 (a : ℝ) :
  (A ∩ B a = B a) ↔ (a ≤ -3 ∨ a > 2) :=
sorry

end Example

end Q1_Intersection_Q1_Union_Q2_l154_154819


namespace responses_needed_l154_154689

noncomputable def Q : ℝ := 461.54
noncomputable def percentage : ℝ := 0.65
noncomputable def required_responses : ℝ := percentage * Q

theorem responses_needed : required_responses = 300 := by
  sorry

end responses_needed_l154_154689


namespace part1_profit_in_april_part2_price_reduction_l154_154902

-- Given conditions
def cost_per_bag : ℕ := 16
def original_price_per_bag : ℕ := 30
def reduction_amount : ℕ := 5
def increase_in_sales_rate : ℕ := 20
def original_sales_volume : ℕ := 200
def target_profit : ℕ := 2860

-- Part 1: When the price per bag of noodles is reduced by 5 yuan
def profit_in_april_when_reduced_by_5 (cost_per_bag original_price_per_bag reduction_amount increase_in_sales_rate original_sales_volume : ℕ) : ℕ := 
  let new_price := original_price_per_bag - reduction_amount
  let new_sales_volume := original_sales_volume + (increase_in_sales_rate * reduction_amount)
  let profit_per_bag := new_price - cost_per_bag
  profit_per_bag * new_sales_volume

theorem part1_profit_in_april :
  profit_in_april_when_reduced_by_5 16 30 5 20 200 = 2700 :=
sorry

-- Part 2: Determine the price reduction for a specific target profit
def price_reduction_for_profit (cost_per_bag original_price_per_bag increase_in_sales_rate original_sales_volume target_profit : ℕ) : ℕ :=
  let x := (target_profit - (original_sales_volume * (original_price_per_bag - cost_per_bag))) / (increase_in_sales_rate * (original_price_per_bag - cost_per_bag) - increase_in_sales_rate - original_price_per_bag)
  x

theorem part2_price_reduction :
  price_reduction_for_profit 16 30 20 200 2860 = 3 :=
sorry

end part1_profit_in_april_part2_price_reduction_l154_154902


namespace distance_in_scientific_notation_l154_154710

-- Definition for the number to be expressed in scientific notation
def distance : ℝ := 55000000

-- Expressing the number in scientific notation
def scientific_notation : ℝ := 5.5 * (10 ^ 7)

-- Theorem statement asserting the equality
theorem distance_in_scientific_notation : distance = scientific_notation :=
  by
  -- Proof not required here, so we leave it as sorry
  sorry

end distance_in_scientific_notation_l154_154710


namespace find_circular_permutations_l154_154635

def alpha := (1 + Real.sqrt 5) / 2
def beta := (1 - Real.sqrt 5) / 2
def fib : ℕ → ℝ
| 0 := 0
| 1 := 1
| (n + 2) := fib n + fib (n + 1)

def b_n (n : ℕ) : ℝ := alpha^n + beta^n + 2

theorem find_circular_permutations (n : ℕ) : b_n n = alpha^n + beta^n + 2 :=
sorry

end find_circular_permutations_l154_154635


namespace min_value_of_squares_l154_154090

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧ (∃ (a' b' c' : ℝ), a' = b' ∧ b' = c' ∧ a' + b' + c' = t ∧ a'^2 + b'^2 + c'^2 = t^2 / 3) := 
by
  sorry

end min_value_of_squares_l154_154090


namespace tangent_line_eq_l154_154482

noncomputable def f (x : ℝ) : ℝ := x + Real.log x

theorem tangent_line_eq :
  ∃ (m b : ℝ), (m = (deriv f 1)) ∧ (b = (f 1 - m * 1)) ∧
   (∀ (x y : ℝ), y = m * (x - 1) + b ↔ y = 2 * x - 1) :=
by sorry

end tangent_line_eq_l154_154482


namespace A_takes_200_seconds_l154_154903

/-- 
  A can give B a start of 50 meters or 10 seconds in a kilometer race.
  How long does A take to complete the race?
-/
theorem A_takes_200_seconds (v_A : ℝ) (distance : ℝ) (start_meters : ℝ) (start_seconds : ℝ) :
  (start_meters = 50) ∧ (start_seconds = 10) ∧ (distance = 1000) ∧ 
  (v_A = start_meters / start_seconds) → distance / v_A = 200 :=
by
  sorry

end A_takes_200_seconds_l154_154903


namespace grace_age_is_60_l154_154205

def Grace : ℕ := 60
def motherAge : ℕ := 80
def grandmotherAge : ℕ := 2 * motherAge
def graceAge : ℕ := (3 / 8) * grandmotherAge

theorem grace_age_is_60 : graceAge = Grace := by
  sorry

end grace_age_is_60_l154_154205


namespace extra_bananas_each_child_gets_l154_154243

theorem extra_bananas_each_child_gets
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (absent_children : ℕ)
  (present_children : ℕ)
  (total_bananas : ℕ)
  (bananas_each_present_child_gets : ℕ)
  (extra_bananas : ℕ) :
  total_children = 840 ∧
  bananas_per_child = 2 ∧
  absent_children = 420 ∧
  present_children = total_children - absent_children ∧
  total_bananas = total_children * bananas_per_child ∧
  bananas_each_present_child_gets = total_bananas / present_children ∧
  extra_bananas = bananas_each_present_child_gets - bananas_per_child →
  extra_bananas = 2 :=
by
  sorry

end extra_bananas_each_child_gets_l154_154243


namespace find_value_of_expression_l154_154807

theorem find_value_of_expression
  (a b : ℝ)
  (h₁ : a = 4 + Real.sqrt 15)
  (h₂ : b = 4 - Real.sqrt 15)
  (h₃ : ∀ x : ℝ, (x^3 - 9 * x^2 + 9 * x = 1) → (x = a ∨ x = b ∨ x = 1))
  : (a / b) + (b / a) = 62 := sorry

end find_value_of_expression_l154_154807


namespace Alina_messages_comparison_l154_154782

theorem Alina_messages_comparison 
  (lucia_day1 : ℕ) (alina_day1 : ℕ) (lucia_day2 : ℕ) (alina_day2 : ℕ) (lucia_day3 : ℕ) (alina_day3 : ℕ)
  (h1 : lucia_day1 = 120)
  (h2 : alina_day1 = lucia_day1 - 20)
  (h3 : lucia_day2 = lucia_day1 / 3)
  (h4 : lucia_day3 = lucia_day1)
  (h5 : alina_day3 = alina_day1)
  (h6 : lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3 = 680) :
  alina_day2 = alina_day1 + 100 :=
sorry

end Alina_messages_comparison_l154_154782


namespace cos_pi_minus_2alpha_l154_154811

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 := 
by
  sorry

end cos_pi_minus_2alpha_l154_154811


namespace initial_men_in_hostel_l154_154911

theorem initial_men_in_hostel (x : ℕ) (h1 : 36 * x = 45 * (x - 50)) : x = 250 := 
  sorry

end initial_men_in_hostel_l154_154911


namespace expression_simplifies_to_32_l154_154104

noncomputable def simplified_expression (a : ℝ) : ℝ :=
  8 / (1 + a^8) + 4 / (1 + a^4) + 2 / (1 + a^2) + 1 / (1 + a) + 1 / (1 - a)

theorem expression_simplifies_to_32 :
  simplified_expression (2^(-1/16 : ℝ)) = 32 :=
by
  sorry

end expression_simplifies_to_32_l154_154104


namespace solve_for_2a_plus_b_l154_154208

variable (a b : ℝ)

theorem solve_for_2a_plus_b (h1 : 4 * a ^ 2 - b ^ 2 = 12) (h2 : 2 * a - b = 4) : 2 * a + b = 3 := 
by
  sorry

end solve_for_2a_plus_b_l154_154208


namespace problem_1_problem_2_l154_154962

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem problem_1 (h₁ : ∀ x, x > 0 → x ≠ 1 → f x = x / Real.log x) :
  (∀ x, 1 < x ∧ x < Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) ∧
  (∀ x, x > Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) :=
sorry

theorem problem_2 (h₁ : f x₁ = 1) (h₂ : f x₂ = 1) (h₃ : x₁ ≠ x₂) (h₄ : x₁ > 0) (h₅ : x₂ > 0):
  x₁ + x₂ > 2 * Real.exp 1 :=
sorry

end problem_1_problem_2_l154_154962


namespace sum_of_solutions_is_24_l154_154973

theorem sum_of_solutions_is_24 (a : ℝ) (x1 x2 : ℝ) 
    (h1 : abs (x1 - a) = 100) (h2 : abs (x2 - a) = 100)
    (sum_eq : x1 + x2 = 24) : a = 12 :=
sorry

end sum_of_solutions_is_24_l154_154973


namespace recurring_product_is_14_over_41_l154_154431

-- Conditions translating into Lean
def recurring_63_to_frac : ℚ := 63 / 99
def recurring_54_to_frac : ℚ := 54 / 99

theorem recurring_product_is_14_over_41 :
  recurring_63_to_frac * recurring_54_to_frac = 14 / 41 := 
by
  sorry

end recurring_product_is_14_over_41_l154_154431


namespace line_does_not_pass_through_point_l154_154671

theorem line_does_not_pass_through_point 
  (m : ℝ) (h : (2*m + 1)^2 - 4*(m^2 + 4) > 0) : 
  ¬((2*m - 3)*(-2) - 4*m + 7 = 1) :=
by
  sorry

end line_does_not_pass_through_point_l154_154671


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154404

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154404


namespace hyperbola_properties_l154_154829

-- Define the conditions and the final statements we need to prove
theorem hyperbola_properties (a : ℝ) (ha : a > 2) (E : ℝ → ℝ → Prop)
  (hE : ∀ x y, E x y ↔ (x^2 / a^2 - y^2 / (a^2 - 4) = 1))
  (e : ℝ) (he : e = (Real.sqrt (a^2 + (a^2 - 4))) / a) :
  (∃ E' : ℝ → ℝ → Prop,
   ∀ x y, E' x y ↔ (x^2 / 9 - y^2 / 5 = 1)) ∧
  (∃ foci line: ℝ → ℝ → Prop,
   (∀ P : ℝ × ℝ, (E P.1 P.2) →
    (∃ Q : ℝ × ℝ, (P.1 - Q.1) * (P.1 + (Real.sqrt (2*a^2-4))) = 0 ∧ Q.2=0 ∧ 
     line (P.1) (P.2) ↔ P.1 - P.2 = 2))) :=
by
  sorry

end hyperbola_properties_l154_154829


namespace parabola_directrix_l154_154176

theorem parabola_directrix (x : ℝ) : ∃ d : ℝ, (∀ x : ℝ, 4 * x ^ 2 - 3 = d) → d = -49 / 16 :=
by
  sorry

end parabola_directrix_l154_154176


namespace quadratic_equiv_original_correct_transformation_l154_154096

theorem quadratic_equiv_original :
  (5 + 3*Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3 = 
  (7 + 4 * Real.sqrt 3) * x^2 + (2 + Real.sqrt 3) * x - 2 :=
sorry

theorem correct_transformation :
  ∃ r : ℝ, r = (9 / 7) - (4 * Real.sqrt 2 / 7) ∧ 
  ((5 + 3 * Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3) = 0 :=
sorry

end quadratic_equiv_original_correct_transformation_l154_154096


namespace evaluate_log_l154_154025

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l154_154025


namespace find_n_square_divides_exponential_plus_one_l154_154173

theorem find_n_square_divides_exponential_plus_one :
  ∀ n : ℕ, (n^2 ∣ 2^n + 1) → (n = 1) :=
by
  sorry

end find_n_square_divides_exponential_plus_one_l154_154173


namespace sqrt_expr_is_599_l154_154318

theorem sqrt_expr_is_599 : Real.sqrt ((26 * 25 * 24 * 23) + 1) = 599 := by
  sorry

end sqrt_expr_is_599_l154_154318


namespace smallest_composite_proof_l154_154652

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l154_154652


namespace solve_inequality_prove_inequality_l154_154289

open Real

-- Problem 1: Solve the inequality
theorem solve_inequality (x : ℝ) : (x - 1) / (2 * x + 1) ≤ 0 ↔ (-1 / 2) < x ∧ x ≤ 1 :=
sorry

-- Problem 2: Prove the inequality given positive a, b, and c
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) * (1 / a + 1 / (b + c)) ≥ 4 :=
sorry

end solve_inequality_prove_inequality_l154_154289


namespace work_fraction_completed_after_first_phase_l154_154141

-- Definitions based on conditions
def total_work := 1 -- Assume total work as 1 unit
def initial_days := 100
def initial_people := 10
def first_phase_days := 20
def fired_people := 2
def remaining_days := 75
def remaining_people := initial_people - fired_people

-- Hypothesis about the rate of work initially and after firing people
def initial_rate := total_work / initial_days
def first_phase_work := first_phase_days * initial_rate
def remaining_work := total_work - first_phase_work
def remaining_rate := remaining_work / remaining_days

-- Proof problem statement: 
theorem work_fraction_completed_after_first_phase :
  (first_phase_work / total_work) = (15 / 64) :=
by
  -- This is the place where the actual formal proof should be written.
  sorry

end work_fraction_completed_after_first_phase_l154_154141


namespace nonstudent_ticket_cost_l154_154053

theorem nonstudent_ticket_cost :
  ∃ x : ℝ, (530 * 2 + (821 - 530) * x = 1933) ∧ x = 3 :=
by 
  sorry

end nonstudent_ticket_cost_l154_154053


namespace digit_100th_place_of_8_over_11_l154_154685

/-- Decimal expansion of 8/11 -/
def decimal_expansion_8_over_11 : ℕ → ℕ
| n := if n % 2 = 0 then 7 else 2

/-- The 100th digit of the decimal expansion of 8/11 is 2 -/
theorem digit_100th_place_of_8_over_11 : decimal_expansion_8_over_11 99 = 2 :=
sorry

end digit_100th_place_of_8_over_11_l154_154685


namespace smallest_composite_no_prime_factors_below_15_correct_l154_154656

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l154_154656


namespace find_k_and_direction_l154_154356

open Vector

-- Define vectors a, b, c, and d
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def d : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Define parallel condition as scalar multiple of each other
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, u.1 = λ * v.1 ∧ u.2 = λ * v.2

theorem find_k_and_direction (k : ℝ) :
  parallel (c k) d → k = -1 ∧ (c k).1 = -(d.1) ∧ (c k).2 = -(d.2) :=
by
  intro h
  sorry

end find_k_and_direction_l154_154356


namespace hyperbola_equation_l154_154194

open Real

theorem hyperbola_equation (e e' : ℝ) (h₁ : 2 * x^2 + y^2 = 2) (h₂ : e * e' = 1) :
  y^2 - x^2 = 2 :=
sorry

end hyperbola_equation_l154_154194


namespace expression_value_correct_l154_154125

theorem expression_value_correct (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : -a - b^3 + a * b = -11 := by
  sorry

end expression_value_correct_l154_154125


namespace Jacob_age_is_3_l154_154976

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end Jacob_age_is_3_l154_154976


namespace smallest_composite_no_prime_factors_less_than_15_l154_154637

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l154_154637


namespace smallest_four_digit_mod_8_l154_154433

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l154_154433


namespace maximize_profit_l154_154613

-- Define the relationships and constants
def P (x : ℝ) : ℝ := -750 * x + 15000
def material_cost_per_unit : ℝ := 4
def fixed_cost : ℝ := 7000

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - material_cost_per_unit) * P x - fixed_cost

-- The statement of the problem, proving the maximization condition
theorem maximize_profit :
  ∃ x : ℝ, x = 12 ∧ profit 12 = 41000 := by
  sorry

end maximize_profit_l154_154613


namespace lyle_friends_sandwich_and_juice_l154_154998

theorem lyle_friends_sandwich_and_juice : 
  ∀ (sandwich_cost juice_cost lyle_money : ℝ),
    sandwich_cost = 0.30 → 
    juice_cost = 0.20 → 
    lyle_money = 2.50 → 
    (⌊lyle_money / (sandwich_cost + juice_cost)⌋.toNat - 1) = 4 :=
by
  intros sandwich_cost juice_cost lyle_money hc_sandwich hc_juice hc_money
  have cost_one_set := sandwich_cost + juice_cost
  have number_of_sets := lyle_money / cost_one_set
  have friends := (number_of_sets.toNat - 1)
  have friends_count := 4
  sorry

end lyle_friends_sandwich_and_juice_l154_154998


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154413

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154413


namespace angle_ABC_is_83_l154_154502

-- Definitions of angles and the quadrilateral
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (angleBAC angleCAD angleACD : ℝ)
variables (AB AD AC : ℝ)

-- Conditions as hypotheses
axiom h1 : angleBAC = 60
axiom h2 : angleCAD = 60
axiom h3 : AB + AD = AC
axiom h4 : angleACD = 23

-- The theorem to prove
theorem angle_ABC_is_83 (h1 : angleBAC = 60) (h2 : angleCAD = 60) (h3 : AB + AD = AC) (h4 : angleACD = 23) : 
  ∃ angleABC : ℝ, angleABC = 83 :=
sorry

end angle_ABC_is_83_l154_154502


namespace max_actors_chess_tournament_l154_154699

-- Definitions based on conditions
variable {α : Type} [Fintype α] [DecidableEq α]

-- Each actor played with every other actor exactly once.
def played_with_everyone (R : α → α → ℝ) : Prop :=
  ∀ a b, a ≠ b → (R a b = 1 ∨ R a b = 0.5 ∨ R a b = 0)

-- Among every three participants, one earned exactly 1.5 solidus in matches against the other two.
def condition_1_5_solidi (R : α → α → ℝ) : Prop :=
  ∀ a b c, a ≠ b → b ≠ c → a ≠ c → 
   (R a b + R a c = 1.5 ∨ R b a + R b c = 1.5 ∨ R c a + R c b = 1.5)

-- Prove the maximum number of such participants is 5
theorem max_actors_chess_tournament (actors : Finset α) (R : α → α → ℝ) 
  (h_played : played_with_everyone R) (h_condition : condition_1_5_solidi R) :
  actors.card ≤ 5 :=
  sorry

end max_actors_chess_tournament_l154_154699


namespace inequality_solution_l154_154332

theorem inequality_solution (x : ℝ) : (-3 * x^2 - 9 * x - 6 ≥ -12) ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

end inequality_solution_l154_154332


namespace cubic_roots_identity_l154_154987

noncomputable def roots_of_cubic (a b c : ℝ) : Prop :=
  (5 * a^3 - 2019 * a + 4029 = 0) ∧ 
  (5 * b^3 - 2019 * b + 4029 = 0) ∧ 
  (5 * c^3 - 2019 * c + 4029 = 0)

theorem cubic_roots_identity (a b c : ℝ) (h_roots : roots_of_cubic a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087 / 5 :=
by 
  -- proof steps
  sorry

end cubic_roots_identity_l154_154987


namespace increasing_sequences_count_with_modulo_l154_154786

theorem increasing_sequences_count_with_modulo : 
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sequences_count % mod_value = k :=
by
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sorry

end increasing_sequences_count_with_modulo_l154_154786


namespace value_range_neg_x_squared_l154_154583

theorem value_range_neg_x_squared:
  (∀ y, (-9 ≤ y ∧ y ≤ 0) ↔ ∃ x, (-3 ≤ x ∧ x ≤ 1) ∧ y = -x^2) :=
by
  sorry

end value_range_neg_x_squared_l154_154583


namespace calc_7_op_4_minus_4_op_7_l154_154167

def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

theorem calc_7_op_4_minus_4_op_7 : (op 7 4) - (op 4 7) = -12 := by
  sorry

end calc_7_op_4_minus_4_op_7_l154_154167


namespace imag_part_z_is_3_l154_154496

namespace ComplexMultiplication

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := (1 + 2 * i) * (2 - i)

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℂ := Complex.im z

-- Statement to prove: The imaginary part of z = 3
theorem imag_part_z_is_3 : imag_part z = 3 := by
  sorry

end ComplexMultiplication

end imag_part_z_is_3_l154_154496


namespace wade_customers_l154_154430

theorem wade_customers (F : ℕ) (h1 : 2 * F + 6 * F + 72 = 296) : F = 28 := 
by 
  sorry

end wade_customers_l154_154430


namespace area_ratio_of_squares_l154_154421

theorem area_ratio_of_squares (R x y : ℝ) (hx : x^2 = (4/5) * R^2) (hy : y = R * Real.sqrt 2) :
  x^2 / y^2 = 2 / 5 :=
by sorry

end area_ratio_of_squares_l154_154421


namespace arithmetic_mean_correct_l154_154324

-- Define the expressions
def expr1 (x : ℤ) := x + 12
def expr2 (y : ℤ) := y
def expr3 (x : ℤ) := 3 * x
def expr4 := 18
def expr5 (x : ℤ) := 3 * x + 6

-- The condition as a hypothesis
def condition (x y : ℤ) : Prop := (expr1 x + expr2 y + expr3 x + expr4 + expr5 x) / 5 = 30

-- The theorem to prove
theorem arithmetic_mean_correct : condition 6 72 :=
sorry

end arithmetic_mean_correct_l154_154324


namespace possible_values_of_x_l154_154506

theorem possible_values_of_x (x : ℕ) (h1 : ∃ k : ℕ, k * k = 8 - x) (h2 : 1 ≤ x ∧ x ≤ 8) :
  x = 4 ∨ x = 7 ∨ x = 8 :=
by
  sorry

end possible_values_of_x_l154_154506


namespace value_of_g_neg2_l154_154730

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem value_of_g_neg2 : g (-2) = -1 := by
  sorry

end value_of_g_neg2_l154_154730


namespace steve_correct_operations_l154_154881

theorem steve_correct_operations (x : ℕ) (h1 : x / 8 - 20 = 12) : ((x * 8) + 20) = 2068 :=
by
  sorry

end steve_correct_operations_l154_154881


namespace initial_deposit_l154_154712

theorem initial_deposit (A r : ℝ) (n t : ℕ) (hA : A = 169.40) 
  (hr : r = 0.20) (hn : n = 2) (ht : t = 1) :
  ∃ P : ℝ, P = 140 ∧ A = P * (1 + r / n)^(n * t) :=
by
  sorry

end initial_deposit_l154_154712


namespace part1_zero_of_f_a_neg1_part2_range_of_a_l154_154960

noncomputable def f (a x : ℝ) := a * x^2 + 2 * x - 2 - a

theorem part1_zero_of_f_a_neg1 : 
  f (-1) 1 = 0 :=
by 
  sorry

theorem part2_range_of_a (a : ℝ) :
  a ≤ 0 →
  (∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x = 0 → x = 1) ↔ 
  (-1 ≤ a ∧ a ≤ 0) ∨ (a ≤ -2) :=
by 
  sorry

end part1_zero_of_f_a_neg1_part2_range_of_a_l154_154960


namespace history_book_pages_l154_154113

-- Conditions
def science_pages : ℕ := 600
def novel_pages (science: ℕ) : ℕ := science / 4
def history_pages (novel: ℕ) : ℕ := novel * 2

-- Theorem to prove
theorem history_book_pages : history_pages (novel_pages science_pages) = 300 :=
by
  sorry

end history_book_pages_l154_154113


namespace hats_per_yard_of_velvet_l154_154857

theorem hats_per_yard_of_velvet
  (H : ℕ)
  (velvet_for_cloak : ℕ := 3)
  (total_velvet : ℕ := 21)
  (number_of_cloaks : ℕ := 6)
  (number_of_hats : ℕ := 12)
  (yards_for_6_cloaks : ℕ := number_of_cloaks * velvet_for_cloak)
  (remaining_yards_for_hats : ℕ := total_velvet - yards_for_6_cloaks)
  (hats_per_remaining_yard : ℕ := number_of_hats / remaining_yards_for_hats)
  : H = hats_per_remaining_yard :=
  by
  sorry

end hats_per_yard_of_velvet_l154_154857


namespace inequality_proof_l154_154087

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = a * b) : 
  (a / (b^2 + 4) + b / (a^2 + 4) >= 1 / 2) := 
  sorry

end inequality_proof_l154_154087


namespace part1_part2_l154_154501

-- Part (1)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < -1 / 2 → (ax - 1) * (x + 1) > 0) →
  a = -2 :=
sorry

-- Part (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ,
    ((a < -1 ∧ -1 < x ∧ x < 1/a) ∨
     (a = -1 ∧ ∀ x : ℝ, false) ∨
     (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
     (a = 0 ∧ x < -1) ∨
     (a > 0 ∧ (x < -1 ∨ x > 1/a))) →
    (ax - 1) * (x + 1) > 0) :=
sorry

end part1_part2_l154_154501


namespace mixture_replacement_l154_154913

theorem mixture_replacement
  (A B : ℕ)
  (hA : A = 48)
  (h_ratio1 : A / B = 4)
  (x : ℕ)
  (h_ratio2 : A / (B + x) = 2 / 3) :
  x = 60 :=
by
  sorry

end mixture_replacement_l154_154913


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154417

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154417


namespace sqrt_x_minus_2_meaningful_l154_154508

theorem sqrt_x_minus_2_meaningful (x : ℝ) (h : 0 ≤ x - 2) : 2 ≤ x :=
by sorry

end sqrt_x_minus_2_meaningful_l154_154508


namespace Ned_washed_shirts_l154_154870

-- Definitions based on conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts
def not_washed_shirts : ℕ := 1
def washed_shirts : ℕ := total_shirts - not_washed_shirts

-- Statement to prove
theorem Ned_washed_shirts : washed_shirts = 29 := by
  sorry

end Ned_washed_shirts_l154_154870


namespace percentage_respondents_liked_B_l154_154612

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l154_154612


namespace arithmetic_neg3_plus_4_l154_154623

theorem arithmetic_neg3_plus_4 : -3 + 4 = 1 :=
by
  sorry

end arithmetic_neg3_plus_4_l154_154623


namespace product_of_coordinates_of_D_l154_154346

theorem product_of_coordinates_of_D (D : ℝ × ℝ) (N : ℝ × ℝ) (C : ℝ × ℝ) 
  (hN : N = (4, 3)) (hC : C = (5, -1)) (midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 * D.2 = 21 :=
by
  sorry

end product_of_coordinates_of_D_l154_154346


namespace arnold_protein_intake_l154_154157

def protein_in_collagen_powder (scoops : ℕ) : ℕ := if scoops = 1 then 9 else 18

def protein_in_protein_powder (scoops : ℕ) : ℕ := 21 * scoops

def protein_in_steak : ℕ := 56

def protein_in_greek_yogurt : ℕ := 15

def protein_in_almonds (cups : ℕ) : ℕ := 6 * cups

theorem arnold_protein_intake :
  protein_in_collagen_powder 1 + 
  protein_in_protein_powder 2 + 
  protein_in_steak + 
  protein_in_greek_yogurt + 
  protein_in_almonds 2 = 134 :=
by
  -- Sorry, the proof is omitted intentionally
  sorry

end arnold_protein_intake_l154_154157


namespace largest_number_l154_154274

theorem largest_number (a b c : ℝ) (h1 : a + b + c = 67) (h2 : c - b = 7) (h3 : b - a = 5) : c = 86 / 3 := 
by sorry

end largest_number_l154_154274


namespace triangle_incircle_ratio_l154_154298

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end triangle_incircle_ratio_l154_154298


namespace range_of_m_l154_154361

theorem range_of_m (m : ℝ) (h1 : 0 < m) (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |m * x^3 - Real.log x| ≥ 1) : m ≥ (1 / 3) * Real.exp 2 :=
sorry

end range_of_m_l154_154361


namespace algebraic_expression_value_l154_154075

theorem algebraic_expression_value (p q : ℝ) 
  (h : p * 3^3 + q * 3 + 1 = 2015) : 
  p * (-3)^3 + q * (-3) + 1 = -2013 :=
by 
  sorry

end algebraic_expression_value_l154_154075


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154394

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154394


namespace problem_system_of_equations_l154_154504

-- Define the problem as a theorem in Lean 4
theorem problem_system_of_equations (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 :=
by
  -- The proof is omitted
  sorry

end problem_system_of_equations_l154_154504


namespace solve_fractional_equation_l154_154270

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end solve_fractional_equation_l154_154270


namespace area_square_given_diagonal_l154_154893

theorem area_square_given_diagonal (d : ℝ) (h : d = 16) : (∃ A : ℝ, A = 128) :=
by 
  sorry

end area_square_given_diagonal_l154_154893


namespace geometric_sequence_a4_range_l154_154771

theorem geometric_sequence_a4_range
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : 0 < a 1 ∧ a 1 < 1)
  (h2 : 1 < a 1 * q ∧ a 1 * q < 2)
  (h3 : 2 < a 1 * q^2 ∧ a 1 * q^2 < 3) :
  ∃ a4 : ℝ, a4 = a 1 * q^3 ∧ 2 * Real.sqrt 2 < a4 ∧ a4 < 9 := 
sorry

end geometric_sequence_a4_range_l154_154771


namespace probability_sum_3_correct_l154_154247

noncomputable def probability_of_sum_3 : ℚ := 2 / 36

theorem probability_sum_3_correct :
  probability_of_sum_3 = 1 / 18 :=
by
  sorry

end probability_sum_3_correct_l154_154247


namespace find_valid_numbers_l154_154454

-- Define n as a four-digit number ABCD where A, B, C, and D are digits
def valid_digits : List ℕ := [1, 3, 5, 7, 9]

-- Examine all conditions and correct answers
theorem find_valid_numbers :
  {n | ∃ A B C D,
       n = 1000 * A + 100 * B + 10 * C + D ∧
       A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
       A ∈ valid_digits ∧ B ∈ valid_digits ∧ C ∈ valid_digits ∧ D ∈ valid_digits ∧
       n % A = 0 ∧ n % B = 0 ∧ n % C = 0 ∧ n % D = 0}
  = {1395, 1935, 3195, 3915, 9135, 9315} :=
by
  sorry

end find_valid_numbers_l154_154454


namespace candies_left_l154_154621

-- Defining the given conditions
def initial_candies : Nat := 30
def eaten_candies : Nat := 23

-- Define the target statement to prove
theorem candies_left : initial_candies - eaten_candies = 7 := by
  sorry

end candies_left_l154_154621


namespace tangent_line_at_P_l154_154709

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x

theorem tangent_line_at_P 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : P.1 + P.2 = 0)
  (h2 : f' P.1 a = -1) 
  (h3 : P.2 = f P.1 a) 
  : P = (1, -1) ∨ P = (-1, 1) := 
  sorry

end tangent_line_at_P_l154_154709


namespace range_of_m_l154_154675

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m
  (m : ℝ)
  (hθ : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2)
  (h : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) :
  m < 1 :=
by
  sorry

end range_of_m_l154_154675


namespace smallest_arithmetic_mean_divisible_1111_l154_154407

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l154_154407


namespace largest_value_is_E_l154_154442

theorem largest_value_is_E :
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  E > A ∧ E > B ∧ E > C ∧ E > D := 
by
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  sorry

end largest_value_is_E_l154_154442


namespace converse_proposition_l154_154124

-- Define a proposition for vertical angles
def vertical_angles (α β : ℕ) : Prop := α = β

-- Define the converse of the vertical angle proposition
def converse_vertical_angles (α β : ℕ) : Prop := β = α

-- Prove that the converse of "Vertical angles are equal" is 
-- "Angles that are equal are vertical angles"
theorem converse_proposition (α β : ℕ) : vertical_angles α β ↔ converse_vertical_angles α β :=
by
  sorry

end converse_proposition_l154_154124


namespace perfect_play_winner_l154_154160

theorem perfect_play_winner (A B : ℕ) :
    (A = B → (∃ f : ℕ → ℕ, ∀ n, 0 < f n ∧ f n ≤ B ∧ f n = B - A → false)) ∧
    (A ≠ B → (∃ g : ℕ → ℕ, ∀ n, 0 < g n ∧ g n ≤ B ∧ g n = A - B → false)) :=
sorry

end perfect_play_winner_l154_154160


namespace amanda_car_round_trip_time_l154_154011

theorem amanda_car_round_trip_time :
  let bus_time := 40
  let bus_distance := 120
  let detour := 15
  let reduced_time := 5
  let amanda_trip_one_way_time := bus_time - reduced_time
  let amanda_round_trip_distance := (bus_distance * 2) + (detour * 2)
  let required_time := amanda_round_trip_distance * amanda_trip_one_way_time / bus_distance
  required_time = 79 :=
by
  sorry

end amanda_car_round_trip_time_l154_154011


namespace fraction_zero_solution_l154_154221

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end fraction_zero_solution_l154_154221


namespace k_range_proof_l154_154698

/- Define points in the Cartesian plane as ordered pairs. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/- Define two points P and Q. -/
def P : Point := { x := -1, y := 1 }
def Q : Point := { x := 2, y := 2 }

/- Define the line equation. -/
def line_equation (k : ℝ) (x : ℝ) : ℝ :=
  k * x - 1

/- Define the range of k. -/
def k_range (k : ℝ) : Prop :=
  1 / 3 < k ∧ k < 3 / 2

/- Theorem statement. -/
theorem k_range_proof (k : ℝ) (intersects_PQ_extension : ∀ k : ℝ, ∀ x : ℝ, ((P.y ≤ line_equation k x ∧ line_equation k x ≤ Q.y) ∧ line_equation k x ≠ Q.y) → k_range k) :
  ∀ k, k_range k :=
by
  sorry

end k_range_proof_l154_154698


namespace betta_fish_count_l154_154084

theorem betta_fish_count 
  (total_guppies_per_day : ℕ) 
  (moray_eel_consumption : ℕ) 
  (betta_fish_consumption : ℕ) 
  (betta_fish_count : ℕ) 
  (h_total : total_guppies_per_day = 55)
  (h_eel : moray_eel_consumption = 20)
  (h_betta : betta_fish_consumption = 7) 
  (h_eq : total_guppies_per_day - moray_eel_consumption = betta_fish_consumption * betta_fish_count) : 
  betta_fish_count = 5 :=
by 
  sorry

end betta_fish_count_l154_154084


namespace problem_statement_l154_154068

theorem problem_statement (x y : ℝ) (h1 : |x| + x - y = 16) (h2 : x - |y| + y = -8) : x + y = -8 := sorry

end problem_statement_l154_154068


namespace find_overlapping_area_l154_154458

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end find_overlapping_area_l154_154458


namespace minimum_value_l154_154240

theorem minimum_value : 
  ∀ a b : ℝ, 0 < a → 0 < b → a + 2 * b = 3 → (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
by
  sorry

end minimum_value_l154_154240


namespace measure_of_angleA_l154_154669

theorem measure_of_angleA (A B : ℝ) 
  (h1 : ∀ (x : ℝ), x ≠ A → x ≠ B → x ≠ (3 * B - 20) → (3 * x - 20 ≠ A)) 
  (h2 : A = 3 * B - 20) :
  A = 10 ∨ A = 130 :=
by
  sorry

end measure_of_angleA_l154_154669


namespace prob_B_is_0_352_prob_C_is_0_3072_l154_154581

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define event probabilities
def prob_pA : ℝ := 0.4
def prob_pA0 : ℝ := 0.16
def prob_pA1 : ℝ := 0.48
def prob_pB0 : ℝ := 0.36
def prob_pB1 : ℝ := 0.48
def prob_pB2 : ℝ := 0.16
def prob_pA2 : ℝ := 0.36

-- Define the events
def event_A (ω : Ω) : Prop := sorry 
def event_A0 (ω : Ω) : Prop := sorry 
def event_A1 (ω : Ω) : Prop := sorry 
def event_A2 (ω : Ω) : Prop := sorry 
def event_B0 (ω : Ω) : Prop := sorry 
def event_B1 (ω : Ω) : Prop := sorry 
def event_B2 (ω : Ω) : Prop := sorry 

-- Define the compound events as per the problem
def event_B (ω : Ω) : Prop := event_A0 ω ∧ event_A ω ∨ event_A1 ω ∧ ¬event_A ω
def event_C (ω : Ω) : Prop := event_A1 ω ∧ event_B2 ω ∨ event_A2 ω ∧ event_B1 ω ∨ event_A2 ω ∧ event_B2 ω

-- Defined required proofs
theorem prob_B_is_0_352 :
  P {ω | event_B ω} = 0.352 :=
by sorry

theorem prob_C_is_0_3072 :
  P {ω | event_C ω} = 0.3072 :=
by sorry

end prob_B_is_0_352_prob_C_is_0_3072_l154_154581


namespace exists_integers_not_all_zero_l154_154704

-- Given conditions
variables (a b c : ℝ)
variables (ab bc ca : ℚ)
variables (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca)
variables (x y z : ℤ)

-- The theorem to prove
theorem exists_integers_not_all_zero (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca):
  ∃ (x y z : ℤ), (¬ (x = 0 ∧ y = 0 ∧ z = 0)) ∧ (a * x + b * y + c * z = 0) :=
sorry

end exists_integers_not_all_zero_l154_154704


namespace average_of_numbers_l154_154846

theorem average_of_numbers (x : ℝ) (h : (5 + -1 + -2 + x) / 4 = 1) : x = 2 :=
by
  sorry

end average_of_numbers_l154_154846


namespace right_triangle_legs_l154_154116

theorem right_triangle_legs (a b : ℝ) (r R : ℝ) (hypotenuse : ℝ) (h_ab : a + b = 14) (h_c : hypotenuse = 10)
  (h_leg: a * b = a + b + 10) (h_Pythag : a^2 + b^2 = hypotenuse^2) 
  (h_inradius : r = 2) (h_circumradius : R = 5) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
by
  sorry

end right_triangle_legs_l154_154116


namespace determine_b_l154_154219

noncomputable def f (x b : ℝ) : ℝ := x^3 - b * x^2 + 1/2

theorem determine_b (b : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 b = 0 ∧ f x2 b = 0) → b = 3/2 :=
by
  sorry

end determine_b_l154_154219


namespace babjis_height_less_by_20_percent_l154_154309

variable (B A : ℝ) (h : A = 1.25 * B)

theorem babjis_height_less_by_20_percent : ((A - B) / A) * 100 = 20 := by
  sorry

end babjis_height_less_by_20_percent_l154_154309


namespace smallest_composite_proof_l154_154651

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l154_154651


namespace albert_runs_track_l154_154465

theorem albert_runs_track (x : ℕ) (track_distance : ℕ) (total_distance : ℕ) (additional_laps : ℕ) 
(h1 : track_distance = 9)
(h2 : total_distance = 99)
(h3 : additional_laps = 5)
(h4 : total_distance = track_distance * x + track_distance * additional_laps) :
x = 6 :=
by
  sorry

end albert_runs_track_l154_154465


namespace find_k_l154_154050

open Nat

def S (n : ℕ) : ℕ :=
  Integer.toNat $ (n.toBinary).count 1 -- toBinary converts n to its binary representation and count counts 1's in that binary list

def v (n : ℕ) : ℕ :=
  n - S n

theorem find_k (k : ℕ) : (∀ n : ℕ, n > 0 → 2 ^ ((k - 1) * n + 1) ∣ factorial (k * n) / factorial n) ↔ (∃ m : ℕ, k = 2 ^ m) :=
by
  sorry

end find_k_l154_154050


namespace rook_placements_5x5_l154_154845

/-- The number of ways to place five distinct rooks on a 
  5x5 chess board such that each column and row of the 
  board contains exactly one rook is 120. -/
theorem rook_placements_5x5 : 
  ∃! (f : Fin 5 → Fin 5), Function.Bijective f :=
by
  sorry

end rook_placements_5x5_l154_154845


namespace Steve_bakes_more_apple_pies_l154_154551

def Steve_bakes (days_apple days_cherry pies_per_day : ℕ) : ℕ :=
  (days_apple * pies_per_day) - (days_cherry * pies_per_day)

theorem Steve_bakes_more_apple_pies :
  Steve_bakes 3 2 12 = 12 :=
by
  sorry

end Steve_bakes_more_apple_pies_l154_154551


namespace opposite_of_neg_one_third_l154_154573

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end opposite_of_neg_one_third_l154_154573


namespace molecular_weight_Dinitrogen_pentoxide_l154_154312

theorem molecular_weight_Dinitrogen_pentoxide :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_formula := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_formula = 108.02 :=
by
  sorry

end molecular_weight_Dinitrogen_pentoxide_l154_154312


namespace log_base_three_of_nine_cubed_l154_154029

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l154_154029


namespace geometric_series_sum_first_six_terms_l154_154926

theorem geometric_series_sum_first_six_terms :
  let a := (3 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := (6 : ℕ)
  (a * (1 - r ^ n) / (1 - r)) = (364 / 81 : ℚ) :=
by
  let a := 3
  let r := (1 / 3 : ℚ)
  let n := 6
  have h1 : 1 - r ^ n = 1 - (1 / 3) ^ n := sorry
  have h2 : 1 - r = 1 - (1 / 3) := sorry
  rw [h1, h2]
  sorry

end geometric_series_sum_first_six_terms_l154_154926


namespace min_trips_to_fill_hole_l154_154009

def hole_filling_trips (initial_gallons : ℕ) (required_gallons : ℕ) (capacity_2gallon : ℕ)
  (capacity_5gallon : ℕ) (capacity_8gallon : ℕ) (time_limit : ℕ) (time_per_trip : ℕ) : ℕ :=
  if initial_gallons < required_gallons then
    let remaining_gallons := required_gallons - initial_gallons
    let num_8gallon := remaining_gallons / capacity_8gallon
    let remaining_after_8gallon := remaining_gallons % capacity_8gallon
    let num_2gallon := if remaining_after_8gallon = 3 then 1 else 0
    let num_5gallon := if remaining_after_8gallon = 3 then 1 else remaining_after_8gallon / capacity_5gallon
    let total_trips := num_8gallon + num_2gallon + num_5gallon
    if total_trips <= time_limit / time_per_trip then
      total_trips
    else
      sorry -- If calculations overflow time limit
  else
    0

theorem min_trips_to_fill_hole : 
  hole_filling_trips 676 823 2 5 8 45 1 = 20 :=
by rfl

end min_trips_to_fill_hole_l154_154009


namespace min_value_of_expression_l154_154093

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, x = 6 * (12 : ℝ)^(1/6) ∧
  (∀ a b c, 0 < a ∧ 0 < b ∧ 0 < c → 
  x ≤ (a + 2 * b) / c + (2 * a + c) / b + (b + 3 * c) / a) :=
sorry

end min_value_of_expression_l154_154093


namespace determine_k_l154_154787

theorem determine_k (a b c k : ℝ) (h : a + b + c = 1) (h_eq : k * (a + bc) = (a + b) * (a + c)) : k = 1 :=
sorry

end determine_k_l154_154787


namespace larger_page_sum_137_l154_154792

theorem larger_page_sum_137 (x y : ℕ) (h1 : x + y = 137) (h2 : y = x + 1) : y = 69 :=
sorry

end larger_page_sum_137_l154_154792


namespace acute_angles_theorem_l154_154666

open Real

variable (α β : ℝ)

-- Given conditions
def conditions : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  tan α = 1 / 7 ∧
  sin β = sqrt 10 / 10

-- Proof goal
def proof_goal : Prop :=
  α + 2 * β = π / 4

-- The final theorem
theorem acute_angles_theorem (h : conditions α β) : proof_goal α β :=
  sorry

end acute_angles_theorem_l154_154666


namespace petya_vasya_same_result_l154_154741

theorem petya_vasya_same_result (a b : ℤ) 
  (h1 : b = a + 1)
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) :
  (a / b) = 1 := 
by
  sorry

end petya_vasya_same_result_l154_154741


namespace distance_home_to_school_l154_154754

-- Define the variables and conditions
variables (D T : ℝ)
def boy_travel_5km_hr_late := 5 * (T + 5 / 60) = D
def boy_travel_10km_hr_early := 10 * (T - 10 / 60) = D

-- State the theorem to prove
theorem distance_home_to_school 
    (H1 : boy_travel_5km_hr_late D T) 
    (H2 : boy_travel_10km_hr_early D T) : 
  D = 2.5 :=
by
  sorry

end distance_home_to_school_l154_154754


namespace mayoral_election_votes_l154_154365

theorem mayoral_election_votes (Y Z : ℕ) 
  (h1 : 22500 = Y + Y / 2) 
  (h2 : 15000 = Z - Z / 5 * 2)
  : Z = 25000 := 
  sorry

end mayoral_election_votes_l154_154365


namespace expression_value_l154_154896

theorem expression_value (x y z : ℤ) (h1: x = 2) (h2: y = -3) (h3: z = 1) :
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := 
by
  sorry

end expression_value_l154_154896


namespace positive_whole_numbers_cube_root_less_than_eight_l154_154833

theorem positive_whole_numbers_cube_root_less_than_eight : 
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by sorry

end positive_whole_numbers_cube_root_less_than_eight_l154_154833


namespace probability_Cecilia_rolls_4_given_win_l154_154849

noncomputable def P_roll_Cecilia_4_given_win : ℚ :=
  let P_C1_4 := 1/6
  let P_W_C := 1/5
  let P_W_C_given_C1_4 := (4/6)^4
  let P_C1_4_and_W_C := P_C1_4 * P_W_C_given_C1_4
  let P_C1_4_given_W_C := P_C1_4_and_W_C / P_W_C
  P_C1_4_given_W_C

theorem probability_Cecilia_rolls_4_given_win :
  P_roll_Cecilia_4_given_win = 256 / 1555 :=
by 
  -- Here the proof would go, but we include sorry for now.
  sorry

end probability_Cecilia_rolls_4_given_win_l154_154849


namespace find_f_2012_l154_154955

noncomputable def f : ℤ → ℤ := sorry

axiom even_function : ∀ x : ℤ, f (-x) = f x
axiom f_1 : f 1 = 1
axiom f_2011_ne_1 : f 2011 ≠ 1
axiom max_property : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)

theorem find_f_2012 : f 2012 = 1 := sorry

end find_f_2012_l154_154955


namespace find_integer_l154_154920

-- Definition of the given conditions
def conditions (x : ℤ) (r : ℤ) : Prop :=
  (0 ≤ r ∧ r < 7) ∧ ((x - 77) * 8 = 259 + r)

-- Statement of the theorem to be proved
theorem find_integer : ∃ x : ℤ, ∃ r : ℤ, conditions x r ∧ (x = 110) :=
by
  sorry

end find_integer_l154_154920


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154420

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154420


namespace laura_running_speed_l154_154985

theorem laura_running_speed (x : ℝ) (hx : 3 * x + 1 > 0) : 
    (30 / (3 * x + 1)) + (10 / x) = 31 / 12 → x = 7.57 := 
by 
  sorry

end laura_running_speed_l154_154985


namespace ratio_of_segments_l154_154956

theorem ratio_of_segments (a b : ℕ) (ha : a = 200) (hb : b = 40) : a / b = 5 :=
by sorry

end ratio_of_segments_l154_154956


namespace johns_grandpa_money_l154_154984

theorem johns_grandpa_money :
  ∃ G : ℝ, (G + 3 * G = 120) ∧ (G = 30) := 
by
  sorry

end johns_grandpa_money_l154_154984


namespace votes_for_sue_l154_154267

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end votes_for_sue_l154_154267


namespace CD_eq_CE_l154_154379

theorem CD_eq_CE {Point : Type*} [MetricSpace Point]
  (A B C D E : Point) (m : Set Point)
  (hAm : A ∈ m) (hBm : B ∈ m) (hCm : C ∈ m)
  (hDm : D ∉ m) (hEm : E ∉ m) 
  (hAD_AE : dist A D = dist A E)
  (hBD_BE : dist B D = dist B E) :
  dist C D = dist C E :=
sorry

end CD_eq_CE_l154_154379


namespace quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l154_154818

theorem quadratic_eq_real_roots_m_ge_neg1 (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0) →
  m ≥ -1 :=
sorry

theorem quadratic_eq_real_roots_cond (m : ℝ) (x1 x2 : ℝ) :
  x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0 ∧
  (x1 - x2)^2 = 16 - x1 * x2 →
  m = 1 :=
sorry

end quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l154_154818


namespace miles_driven_l154_154856

theorem miles_driven (years_driving : ℕ) (miles_per_four_months : ℕ) (four_month_groups_per_year : ℕ) : 
  years_driving = 9 ∧ miles_per_four_months = 37000 ∧ four_month_groups_per_year = 3 → 
  let miles_per_year := miles_per_four_months * four_month_groups_per_year in
  let total_miles := miles_per_year * years_driving in
  total_miles = 999000 :=
begin
  intros h,
  rcases h with ⟨h1, h2, h3⟩,
  let miles_per_year := miles_per_four_months * four_month_groups_per_year,
  let total_miles := miles_per_year * years_driving,
  sorry
end

end miles_driven_l154_154856


namespace simplify_expression_l154_154752

theorem simplify_expression :
  (4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75 / 1 + 53/68 / ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 :=
by sorry

end simplify_expression_l154_154752


namespace maria_total_cost_l154_154241

-- Define the conditions as variables in the Lean environment
def daily_rental_rate : ℝ := 35
def mileage_rate : ℝ := 0.25
def rental_days : ℕ := 3
def miles_driven : ℕ := 500

-- Now, state the theorem that Maria’s total payment should be $230
theorem maria_total_cost : (daily_rental_rate * rental_days) + (mileage_rate * miles_driven) = 230 := 
by
  -- no proof required, just state as sorry
  sorry

end maria_total_cost_l154_154241


namespace annual_income_before_tax_l154_154515

theorem annual_income_before_tax (I : ℝ) (h1 : 0.42 * I - 0.28 * I = 4830) : I = 34500 :=
sorry

end annual_income_before_tax_l154_154515


namespace triangle_perimeter_l154_154918

theorem triangle_perimeter
  (x : ℝ) 
  (h : x^2 - 6 * x + 8 = 0)
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = x)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 10 := 
sorry

end triangle_perimeter_l154_154918


namespace arithmetic_sequence_sum_l154_154348

-- Definitions used in the conditions
variable (a : ℕ → ℕ)
variable (n : ℕ)
variable (a_seq : Prop)
-- Declaring the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

noncomputable def a_5_is_2 : Prop := a 5 = 2

-- The statement we need to prove
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith_seq : is_arithmetic_sequence a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 := by
sorry

end arithmetic_sequence_sum_l154_154348


namespace smallest_composite_no_prime_under_15_correct_l154_154647

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l154_154647


namespace Ivan_cannot_cut_off_all_heads_l154_154982

-- Defining the number of initial heads
def initial_heads : ℤ := 100

-- Effect of the first sword: Removes 21 heads
def first_sword_effect : ℤ := 21

-- Effect of the second sword: Removes 4 heads and adds 2006 heads
def second_sword_effect : ℤ := 2006 - 4

-- Proving Ivan cannot reduce the number of heads to zero
theorem Ivan_cannot_cut_off_all_heads :
  (∀ n : ℤ, n % 7 = initial_heads % 7 → n ≠ 0) :=
by
  sorry

end Ivan_cannot_cut_off_all_heads_l154_154982


namespace eve_stamp_collection_worth_l154_154327

def total_value_of_collection (stamps_value : ℕ) (num_stamps : ℕ) (set_size : ℕ) (set_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let value_per_stamp := set_value / set_size
  let total_value := value_per_stamp * num_stamps
  let num_complete_sets := num_stamps / set_size
  let total_bonus := num_complete_sets * bonus_per_set
  total_value + total_bonus

theorem eve_stamp_collection_worth :
  total_value_of_collection 21 21 7 28 5 = 99 := by
  rfl

end eve_stamp_collection_worth_l154_154327


namespace weight_of_replaced_person_l154_154725

/-- The weight of the person who was replaced is calculated given the average weight increase for 8 persons and the weight of the new person. --/
theorem weight_of_replaced_person
  (avg_weight_increase : ℝ)
  (num_persons : ℕ)
  (weight_new_person : ℝ) :
  avg_weight_increase = 3 → 
  num_persons = 8 →
  weight_new_person = 89 →
  weight_new_person - avg_weight_increase * num_persons = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end weight_of_replaced_person_l154_154725


namespace english_score_is_96_l154_154701

variable (Science_score : ℕ) (Social_studies_score : ℕ) (English_score : ℕ)

/-- Jimin's social studies score is 6 points higher than his science score -/
def social_studies_score_condition := Social_studies_score = Science_score + 6

/-- The science score is 87 -/
def science_score_condition := Science_score = 87

/-- The average score for science, social studies, and English is 92 -/
def average_score_condition := (Science_score + Social_studies_score + English_score) / 3 = 92

theorem english_score_is_96
  (h1 : social_studies_score_condition Science_score Social_studies_score)
  (h2 : science_score_condition Science_score)
  (h3 : average_score_condition Science_score Social_studies_score English_score) :
  English_score = 96 :=
  by
    sorry

end english_score_is_96_l154_154701


namespace problem_l154_154665

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end problem_l154_154665


namespace quadratic_always_real_roots_rhombus_area_when_m_minus_7_l154_154953

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Statement 1: For any real number m, the quadratic equation always has real roots.
theorem quadratic_always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
by {
  -- Proof omitted
  sorry
}

-- Statement 2: When m = -7, the area of the rhombus whose diagonals are the roots of the quadratic equation is 7/4.
theorem rhombus_area_when_m_minus_7 : (∃ x1 x2 : ℝ, quadratic_eq (-7) x1 = 0 ∧ quadratic_eq (-7) x2 = 0 ∧ (1 / 2) * x1 * x2 = 7 / 4) :=
by {
  -- Proof omitted
  sorry
}

end quadratic_always_real_roots_rhombus_area_when_m_minus_7_l154_154953


namespace simplify_fraction_product_l154_154471

theorem simplify_fraction_product :
  (2 / 3) * (4 / 7) * (9 / 13) = 24 / 91 := by
  sorry

end simplify_fraction_product_l154_154471


namespace proof_problem_l154_154813

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))

theorem proof_problem (
  a : ℝ
) (h1 : a > 1) :
  (∀ x, f a x = (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))) ∧
  (∀ x, f a (-x) = -f a x) ∧
  (∀ x1 x2, x1 < x2 → f a x1 < f a x2) ∧
  (∀ m, -1 < 1 - m ∧ 1 - m < m^2 - 1 ∧ m^2 - 1 < 1 → 1 < m ∧ m < Real.sqrt 2)
  :=
sorry

end proof_problem_l154_154813


namespace correct_operations_result_l154_154248

-- Define conditions and the problem statement
theorem correct_operations_result (x : ℝ) (h1: x / 8 - 12 = 18) : (x * 8) * 12 = 23040 :=
by
  sorry

end correct_operations_result_l154_154248


namespace complement_union_eq_l154_154190

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_union_eq : (U \ (A ∪ B)) = {6,8} := by
  sorry

end complement_union_eq_l154_154190


namespace number_of_divisors_of_square_l154_154213

theorem number_of_divisors_of_square {n : ℕ} (h : ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q) : Nat.totient (n^2) = 9 :=
sorry

end number_of_divisors_of_square_l154_154213


namespace carlotta_tantrum_time_l154_154658

theorem carlotta_tantrum_time :
  (∀ (T P S : ℕ), 
   S = 6 ∧ T + P + S = 54 ∧ P = 3 * S → T = 5 * S) :=
by
  intro T P S
  rintro ⟨hS, hTotal, hPractice⟩
  sorry

end carlotta_tantrum_time_l154_154658


namespace problem_quadratic_radicals_l154_154062

theorem problem_quadratic_radicals (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 :=
sorry

end problem_quadratic_radicals_l154_154062


namespace smallest_arithmetic_mean_l154_154411

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l154_154411


namespace cos_angle_subtraction_l154_154493

open Real

theorem cos_angle_subtraction (A B : ℝ) (h1 : sin A + sin B = 3 / 2) (h2 : cos A + cos B = 1) :
  cos (A - B) = 5 / 8 :=
sorry

end cos_angle_subtraction_l154_154493


namespace mathNotRebusOrLogic_l154_154463

open Set

variables (B : Type) [Fintype B]
variables (R M L : Set B) -- Sets of brainiacs who like each type of teaser
variables (brainiacsSurveyed : Set B) (neither : Set B)

-- Given conditions
def totalSurveyed : Fintype.card brainiacsSurveyed = 500 := sorry
def neitherTeasers : Fintype.card neither = 20 := sorry
def twiceAsManyRebus : Fintype.card R = 2 * Fintype.card M := sorry
def equalLogicMath : Fintype.card L = Fintype.card M := sorry
def bothRebusMath : Fintype.card (R ∩ M) = 72 := sorry
def bothRebusLogic : Fintype.card (R ∩ L) = 40 := sorry
def bothMathLogic : Fintype.card (M ∩ L) = 36 := sorry
def allThree : Fintype.card (R ∩ M ∩ L) = 10 := sorry

-- Prove the number of brainiacs who like math teasers but not rebus or logic teasers is 54
theorem mathNotRebusOrLogic :
  Fintype.card M - Fintype.card (R ∩ M) - Fintype.card (M ∩ L) + Fintype.card (R ∩ M ∩ L) = 54 := sorry

end mathNotRebusOrLogic_l154_154463


namespace remaining_black_area_after_five_changes_l154_154156

-- Define a function that represents the change process
noncomputable def remaining_black_area (iterations : ℕ) : ℚ :=
  (3 / 4) ^ iterations

-- Define the original problem statement as a theorem in Lean
theorem remaining_black_area_after_five_changes :
  remaining_black_area 5 = 243 / 1024 :=
by
  sorry

end remaining_black_area_after_five_changes_l154_154156


namespace log_base_3_of_9_cubed_l154_154040

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l154_154040


namespace custom_mul_4_3_l154_154210

-- Define the binary operation a*b = a^2 - ab + b^2
def custom_mul (a b : ℕ) : ℕ := a^2 - a*b + b^2

-- State the theorem to prove that 4 * 3 = 13
theorem custom_mul_4_3 : custom_mul 4 3 = 13 := by
  sorry -- Proof will be filled in here

end custom_mul_4_3_l154_154210


namespace birds_flew_up_count_l154_154762

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end birds_flew_up_count_l154_154762


namespace liked_product_B_l154_154607

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l154_154607


namespace smallest_composite_no_prime_factors_below_15_correct_l154_154654

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l154_154654


namespace findB_coords_l154_154229

namespace ProofProblem

-- Define point A with its coordinates.
def A : ℝ × ℝ := (-3, 2)

-- Define a property that checks if a line segment AB is parallel to the x-axis.
def isParallelToXAxis (A B : (ℝ × ℝ)) : Prop :=
  A.2 = B.2

-- Define a property that checks if the length of line segment AB is 4.
def hasLengthFour (A B : (ℝ × ℝ)) : Prop :=
  abs (A.1 - B.1) = 4

-- The proof problem statement.
theorem findB_coords :
  ∃ B : ℝ × ℝ, isParallelToXAxis A B ∧ hasLengthFour A B ∧ (B = (-7, 2) ∨ B = (1, 2)) :=
  sorry

end ProofProblem

end findB_coords_l154_154229


namespace houses_with_garage_l154_154693

theorem houses_with_garage (P GP N : ℕ) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) 
    (total_houses : P + GP - GP + N = 65) : 
    P + 65 - P - GP + GP - N = 50 :=
by
  sorry

end houses_with_garage_l154_154693


namespace perpendicular_lines_l154_154820

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x + y + 1 = 0) ∧ (∀ x y : ℝ, x + a * y + 3 = 0) ∧ (∀ A1 B1 A2 B2 : ℝ, A1 * A2 + B1 * B2 = 0) →
  a = -2 :=
by
  intros h
  sorry

end perpendicular_lines_l154_154820


namespace root_difference_l154_154169

theorem root_difference (p : ℝ) (r s : ℝ) 
  (h₁ : r + s = 2 * p) 
  (h₂ : r * s = (p^2 - 4) / 3) : 
  r - s = 2 * (Real.sqrt 3) / 3 :=
by
  sorry

end root_difference_l154_154169


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l154_154840

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l154_154840


namespace height_percentage_difference_l154_154152

theorem height_percentage_difference (A B : ℝ) (h : B = A * (4/3)) : 
  (A * (1/3) / B) * 100 = 25 := by
  sorry

end height_percentage_difference_l154_154152


namespace dart_game_solution_l154_154751

theorem dart_game_solution (x y z : ℕ) (h_x : 8 * x + 9 * y + 10 * z = 100) (h_y : x + y + z > 11) :
  (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0) :=
by
  sorry

end dart_game_solution_l154_154751


namespace jaden_toy_cars_left_l154_154083

-- Definitions for each condition
def initial_toys : ℕ := 14
def purchased_toys : ℕ := 28
def birthday_toys : ℕ := 12
def given_to_sister : ℕ := 8
def given_to_vinnie : ℕ := 3
def traded_lost : ℕ := 5
def traded_received : ℕ := 7

-- The final number of toy cars Jaden has
def final_toys : ℕ :=
  initial_toys + purchased_toys + birthday_toys - given_to_sister - given_to_vinnie + (traded_received - traded_lost)

theorem jaden_toy_cars_left : final_toys = 45 :=
by
  -- The proof will be filled in here 
  sorry

end jaden_toy_cars_left_l154_154083


namespace infinite_sum_equals_one_fourth_l154_154014

theorem infinite_sum_equals_one_fourth :
  ∑' n : ℕ, (3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
sorry

end infinite_sum_equals_one_fourth_l154_154014


namespace enclosed_area_eq_two_l154_154329

noncomputable def enclosed_area : ℝ :=
  -∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem enclosed_area_eq_two : enclosed_area = 2 := 
  sorry

end enclosed_area_eq_two_l154_154329


namespace shape_with_congruent_views_is_sphere_l154_154972

def is_congruent_views (shape : Type) : Prop :=
  ∀ (front_view left_view top_view : shape), 
  (front_view = left_view) ∧ (left_view = top_view) ∧ (front_view = top_view)

noncomputable def is_sphere (shape : Type) : Prop := 
  ∀ (s : shape), true -- Placeholder definition for a sphere, as recognizing a sphere is outside Lean's scope

theorem shape_with_congruent_views_is_sphere (shape : Type) :
  is_congruent_views shape → is_sphere shape :=
by
  intro h
  sorry

end shape_with_congruent_views_is_sphere_l154_154972


namespace log_base_3_of_9_cubed_l154_154034

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l154_154034


namespace smallest_composite_no_prime_factors_less_than_15_l154_154642

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l154_154642


namespace problem_solution_set_l154_154965

variable {a b c : ℝ}

theorem problem_solution_set (h_condition : ∀ x, 1 ≤ x → x ≤ 2 → a * x^2 - b * x + c ≥ 0) : 
  { x : ℝ | c * x^2 + b * x + a ≤ 0 } = { x : ℝ | x ≤ -1 } ∪ { x | -1/2 ≤ x } :=
by 
  sorry

end problem_solution_set_l154_154965


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154419

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154419


namespace pants_after_5_years_l154_154628

theorem pants_after_5_years (initial_pants : ℕ) (pants_per_year : ℕ) (years : ℕ) :
  initial_pants = 50 → pants_per_year = 8 → years = 5 → (initial_pants + pants_per_year * years) = 90 :=
by
  intros initial_cond pants_per_year_cond years_cond
  rw [initial_cond, pants_per_year_cond, years_cond]
  norm_num
  done

end pants_after_5_years_l154_154628


namespace niki_money_l154_154537

variables (N A : ℕ)

def condition1 (N A : ℕ) : Prop := N = 2 * A + 15
def condition2 (N A : ℕ) : Prop := N - 30 = (A + 30) / 2

theorem niki_money : condition1 N A ∧ condition2 N A → N = 55 :=
by
  sorry

end niki_money_l154_154537


namespace exists_pos_int_n_l154_154106

def sequence_x (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n + 2) = x n + (x (n + 1))^2

def sequence_y (y : ℕ → ℝ) : Prop :=
  ∀ n, y (n + 2) = y n^2 + y (n + 1)

def positive_initial_conditions (x y : ℕ → ℝ) : Prop :=
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

theorem exists_pos_int_n (x y : ℕ → ℝ) (hx : sequence_x x) (hy : sequence_y y) 
  (ini : positive_initial_conditions x y) : ∃ n, x n > y n := 
sorry

end exists_pos_int_n_l154_154106


namespace royWeight_l154_154703

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end royWeight_l154_154703


namespace camp_organizer_needs_more_bottles_l154_154139

variable (cases : ℕ) (bottles_per_case : ℕ) (cases_bought : ℕ)
variable (children_group1 : ℕ) (children_group2 : ℕ) (children_group3 : ℕ)
variable (bottles_per_day : ℕ) (days : ℕ)

noncomputable def bottles_needed := 
  let children_group4 := (children_group1 + children_group2 + children_group3) / 2
  let total_children := children_group1 + children_group2 + children_group3 + children_group4
  let total_bottles_needed := total_children * bottles_per_day * days
  let total_bottles_purchased := cases_bought * bottles_per_case
  total_bottles_needed - total_bottles_purchased

theorem camp_organizer_needs_more_bottles :
  cases = 13 →
  bottles_per_case = 24 →
  cases_bought = 13 →
  children_group1 = 14 →
  children_group2 = 16 →
  children_group3 = 12 →
  bottles_per_day = 3 →
  days = 3 →
  bottles_needed cases bottles_per_case cases_bought children_group1 children_group2 children_group3 bottles_per_day days = 255 := by
  sorry

end camp_organizer_needs_more_bottles_l154_154139


namespace radius_ratio_l154_154142

noncomputable def volume_large_sphere : ℝ := 432 * Real.pi

noncomputable def volume_small_sphere : ℝ := 0.08 * volume_large_sphere

noncomputable def radius_large_sphere : ℝ :=
  (3 * volume_large_sphere / (4 * Real.pi)) ^ (1 / 3)

noncomputable def radius_small_sphere : ℝ :=
  (3 * volume_small_sphere / (4 * Real.pi)) ^ (1 / 3)

theorem radius_ratio (V_L V_s : ℝ) (hL : V_L = 432 * Real.pi) (hS : V_s = 0.08 * V_L) :
  (radius_small_sphere / radius_large_sphere) = (2/5)^(1/3) :=
by
  sorry

end radius_ratio_l154_154142


namespace smallest_composite_no_prime_factors_lt_15_l154_154640

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l154_154640


namespace part1_part2_l154_154822

-- Definitions of propositions P and q
def P (t : ℝ) : Prop := (4 - t > t - 1 ∧ t - 1 > 0)
def q (a t : ℝ) : Prop := t^2 - (a+3)*t + (a+2) < 0

-- Part 1: If P is true, find the range of t.
theorem part1 (t : ℝ) (hP : P t) : 1 < t ∧ t < 5/2 :=
by sorry

-- Part 2: If P is a sufficient but not necessary condition for q, find the range of a.
theorem part2 (a : ℝ) 
  (hP_q : ∀ t, P t → q a t) 
  (hsubset : ∀ t, 1 < t ∧ t < 5/2 → q a t) 
  : a > 1/2 :=
by sorry

end part1_part2_l154_154822


namespace y_axis_symmetry_l154_154218

theorem y_axis_symmetry (x y : ℝ) (P : ℝ × ℝ) (hx : P = (-5, 3)) : 
  (P.1 = -5 ∧ P.2 = 3) → (P.1 * -1, P.2) = (5, 3) :=
by
  intro h
  rw [hx]
  simp [Neg.neg, h]
  sorry

end y_axis_symmetry_l154_154218


namespace calculate_expression_l154_154705

def f (x : ℕ) : ℕ := x^2 - 3*x + 4
def g (x : ℕ) : ℕ := 2*x + 1

theorem calculate_expression : f (g 3) - g (f 3) = 23 := by
  sorry

end calculate_expression_l154_154705


namespace tan_addition_formula_l154_154067

theorem tan_addition_formula (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_addition_formula_l154_154067


namespace limit_series_product_eq_l154_154166

variable (a r s : ℝ)

noncomputable def series_product_sum_limit : ℝ :=
∑' n : ℕ, (a * r^n) * (a * s^n)

theorem limit_series_product_eq :
  |r| < 1 → |s| < 1 → series_product_sum_limit a r s = a^2 / (1 - r * s) :=
by
  intro hr hs
  sorry

end limit_series_product_eq_l154_154166


namespace M_inter_N_eq_l154_154964

open Set

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem M_inter_N_eq : M ∩ N = {3, 4} := 
by 
  sorry

end M_inter_N_eq_l154_154964


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154401

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154401


namespace opposite_of_neg_one_third_l154_154574

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end opposite_of_neg_one_third_l154_154574


namespace determine_f_36_l154_154373

def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n, f (n + 1) > f n

def multiplicative (f : ℕ → ℕ) : Prop :=
  ∀ m n, f (m * n) = f m * f n

def special_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n, m > n → m^m = n^n → f m = n

theorem determine_f_36 (f : ℕ → ℕ)
  (H1: strictly_increasing f)
  (H2: multiplicative f)
  (H3: special_condition f)
  : f 36 = 1296 := 
sorry

end determine_f_36_l154_154373


namespace units_digit_of_8_pow_47_l154_154279

theorem units_digit_of_8_pow_47 : (8 ^ 47) % 10 = 2 := by
  sorry

end units_digit_of_8_pow_47_l154_154279


namespace log_base_3_l154_154032

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l154_154032


namespace symmetry_condition_l154_154333

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

theorem symmetry_condition (ϕ : ℝ) (hϕ : |ϕ| ≤ π / 2)
    (hxy: ∀ x : ℝ, f (x + ϕ) = f (-x + ϕ)) : ϕ = π / 6 :=
by
  -- Since the problem specifically asks for the statement only and not the proof steps,
  -- a "sorry" is used to skip the proof content.
  sorry

end symmetry_condition_l154_154333


namespace arithmetic_geometric_seq_l154_154534

noncomputable def a (n : ℕ) : ℤ := 2 * n - 4 -- General form of the arithmetic sequence

def is_geometric_sequence (s : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, (n > 1) → s (n+1) * s (n-1) = s n ^ 2

theorem arithmetic_geometric_seq:
  (∃ (d : ℤ) (a : ℕ → ℤ), a 5 = 6 ∧ 
  (∀ n, a n = 6 + (n - 5) * d) ∧ a (3) * a (11) = a (5) ^ 2 ∧
  (∀ k, 5 < k → is_geometric_sequence (fun n => a (k + n - 1)))) → 
  ∃ t : ℕ, ∀ n : ℕ, n <= 2015 → 
  (a n = 2 * n - 4 →  n = 7) := 
sorry

end arithmetic_geometric_seq_l154_154534


namespace probability_auntie_em_can_park_l154_154147

/-- A parking lot has 20 spaces in a row. -/
def total_spaces : ℕ := 20

/-- Fifteen cars arrive, each requiring one parking space, and their drivers choose spaces at random from among the available spaces. -/
def cars : ℕ := 15

/-- Auntie Em's SUV requires 3 adjacent empty spaces. -/
def required_adjacent_spaces : ℕ := 3

/-- Calculate the probability that there are 3 consecutive empty spaces among the 5 remaining spaces after 15 cars are parked in 20 spaces.
Expected answer is (12501 / 15504) -/
theorem probability_auntie_em_can_park : 
    (1 - (↑(Nat.choose 15 5) / ↑(Nat.choose 20 5))) = (12501 / 15504) := 
sorry

end probability_auntie_em_can_park_l154_154147


namespace students_between_jimin_yuna_l154_154764

theorem students_between_jimin_yuna 
  (total_students : ℕ) 
  (jimin_position : ℕ) 
  (yuna_position : ℕ) 
  (h1 : total_students = 32) 
  (h2 : jimin_position = 27) 
  (h3 : yuna_position = 11) 
  : (jimin_position - yuna_position - 1) = 15 := 
by
  sorry

end students_between_jimin_yuna_l154_154764


namespace value_of_expression_l154_154586

-- Definitions based on the conditions
def a : ℕ := 15
def b : ℕ := 3

-- The theorem to prove
theorem value_of_expression : a^2 + 2 * a * b + b^2 = 324 := by
  -- Skipping the proof as per instructions
  sorry

end value_of_expression_l154_154586


namespace min_value_l154_154181

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end min_value_l154_154181


namespace smallest_arithmetic_mean_divisible_product_l154_154399

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l154_154399


namespace positive_whole_numbers_cube_root_less_than_eight_l154_154834

theorem positive_whole_numbers_cube_root_less_than_eight : 
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by sorry

end positive_whole_numbers_cube_root_less_than_eight_l154_154834


namespace smallest_arithmetic_mean_l154_154409

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l154_154409


namespace arithmetic_sequence_a6_l154_154367

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1)
  (h_a1 : a 1 = 5) (h_a5 : a 5 = 1) : a 6 = 0 :=
by
  -- Definitions derived from conditions in the problem:
  -- 1. a : ℕ → ℤ : Sequence defined on ℕ with integer values.
  -- 2. h_arith : ∀ n, a (n+1) - a n = a 2 - a 1 : Arithmetic sequence property
  -- 3. h_a1 : a 1 = 5 : First term of the sequence is 5.
  -- 4. h_a5 : a 5 = 1 : Fifth term of the sequence is 1.
  sorry

end arithmetic_sequence_a6_l154_154367


namespace survivor_quit_probability_l154_154885

noncomputable def probability_both_quit_same_tribe (n : ℕ) (tribe_size : ℕ) (quits : ℕ) : ℚ :=
  (2 * (nat.choose tribe_size quits : ℚ) / nat.choose n quits)

theorem survivor_quit_probability :
  let n := 16
  let tribe_size := 8
  let quits := 2
  probability_both_quit_same_tribe n tribe_size quits = 7 / 15 :=
by
  sorry

end survivor_quit_probability_l154_154885


namespace sum_of_first_5_terms_of_geometric_sequence_l154_154668

theorem sum_of_first_5_terms_of_geometric_sequence :
  let a₁ := 3
  let q := 4
  let n := 5
  let Sₙ := λ n : ℕ, (a₁ * (1 - q^n)) / (1 - q)
  Sₙ 5 = 1023 :=
by
  sorry

end sum_of_first_5_terms_of_geometric_sequence_l154_154668


namespace trigonometric_ratio_sum_l154_154993

open Real

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h₁ : sin x / sin y = 2) 
  (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 41 / 57 := 
by
  sorry

end trigonometric_ratio_sum_l154_154993


namespace power_of_11_in_expression_l154_154485

-- Define the mathematical context
def prime_factors_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n + a + b

-- Given conditions
def count_factors_of_2 : ℕ := 22
def count_factors_of_7 : ℕ := 5
def total_prime_factors : ℕ := 29

-- Theorem stating that power of 11 in the expression is 2
theorem power_of_11_in_expression : 
  ∃ n : ℕ, prime_factors_count n count_factors_of_2 count_factors_of_7 = total_prime_factors ∧ n = 2 :=
by
  sorry

end power_of_11_in_expression_l154_154485


namespace cos_squared_diff_tan_l154_154198

theorem cos_squared_diff_tan (α : ℝ) (h : Real.tan α = 3) :
  Real.cos (α + π/4) ^ 2 - Real.cos (α - π/4) ^ 2 = -3 / 5 :=
by
  sorry

end cos_squared_diff_tan_l154_154198


namespace largest_integer_of_four_l154_154659

theorem largest_integer_of_four (a b c d : ℤ) 
  (h1 : a + b + c = 160) 
  (h2 : a + b + d = 185) 
  (h3 : a + c + d = 205) 
  (h4 : b + c + d = 230) : 
  max (max a (max b c)) d = 100 := 
by
  sorry

end largest_integer_of_four_l154_154659


namespace tan_ratio_l154_154235

theorem tan_ratio (α β : ℝ) 
  (h1 : Real.sin (α + β) = (Real.sqrt 3) / 2) 
  (h2 : Real.sin (α - β) = (Real.sqrt 2) / 2) : 
  (Real.tan α) / (Real.tan β) = (5 + 2 * Real.sqrt 6) / (5 - 2 * Real.sqrt 6) :=
by
  sorry

end tan_ratio_l154_154235


namespace solve_for_x_l154_154530

def delta (x : ℝ) : ℝ := 5 * x + 9
def phi (x : ℝ) : ℝ := 7 * x + 6

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -4) : x = -43 / 35 :=
by
  sorry

end solve_for_x_l154_154530


namespace totalCupsOfLiquid_l154_154314

def amountOfOil : ℝ := 0.17
def amountOfWater : ℝ := 1.17

theorem totalCupsOfLiquid : amountOfOil + amountOfWater = 1.34 := by
  sorry

end totalCupsOfLiquid_l154_154314


namespace Lyle_friends_sandwich_juice_l154_154996

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_l154_154996


namespace find_angle4_l154_154810

theorem find_angle4
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 70)
  (h2 : angle2 = 110)
  (h3 : angle3 = 40)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 30 := 
  sorry

end find_angle4_l154_154810


namespace fraction_zero_solution_l154_154222

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end fraction_zero_solution_l154_154222


namespace log_base_3_l154_154031

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l154_154031


namespace probability_both_red_is_one_fourth_l154_154778

noncomputable def probability_of_both_red (total_cards : ℕ) (red_cards : ℕ) (draws : ℕ) : ℚ :=
  (red_cards / total_cards) ^ draws

theorem probability_both_red_is_one_fourth :
  probability_of_both_red 52 26 2 = 1/4 :=
by
  sorry

end probability_both_red_is_one_fourth_l154_154778


namespace problem1_problem2_problem3_l154_154831

open Set
open Finset

-- Define the universal set
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define sets A, B, and C
def A : Finset ℕ := {x | x^2 - 3*x + 2 = 0}
def B : Finset ℕ := {x | 1 ≤ x ∧ x ≤ 5 ∧ x ∈ ℕ}
def C : Finset ℕ := {x | 2 < x ∧ x < 9 ∧ x ∈ ℕ}

-- Define complements with respect to U
def complement_U (S : Finset ℕ) : Finset ℕ := U \ S

-- Proof problem 1
theorem problem1 : A ∩ B = {1, 2} := by
  sorry

-- Proof problem 2
theorem problem2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5} := by
  sorry

-- Proof problem 3
theorem problem3 : complement_U B ∪ complement_U C = {1, 2, 6, 7, 8} := by
  sorry

end problem1_problem2_problem3_l154_154831


namespace log_base_3_of_9_cubed_l154_154036

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l154_154036


namespace geom_progression_vertex_ad_l154_154663

theorem geom_progression_vertex_ad
  (a b c d : ℝ)
  (geom_prog : a * c = b * b ∧ b * d = c * c)
  (vertex : (b, c) = (1, 3)) :
  a * d = 3 :=
sorry

end geom_progression_vertex_ad_l154_154663


namespace fraction_stamp_collection_l154_154681

theorem fraction_stamp_collection (sold_amount total_value : ℝ) (sold_for : sold_amount = 28) (total : total_value = 49) : sold_amount / total_value = 4 / 7 :=
by
  sorry

end fraction_stamp_collection_l154_154681


namespace unique_prime_solution_l154_154168

-- Define the problem in terms of prime numbers and checking the conditions
open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_solution (p : ℕ) (hp : is_prime p) (h1 : is_prime (p^2 - 6)) (h2 : is_prime (p^2 + 6)) : p = 5 := 
sorry

end unique_prime_solution_l154_154168


namespace ratio_of_areas_l154_154979

theorem ratio_of_areas
  (PQ QR RP : ℝ)
  (PQ_pos : 0 < PQ)
  (QR_pos : 0 < QR)
  (RP_pos : 0 < RP)
  (s t u : ℝ)
  (s_pos : 0 < s)
  (t_pos : 0 < t)
  (u_pos : 0 < u)
  (h1 : s + t + u = 3 / 4)
  (h2 : s^2 + t^2 + u^2 = 1 / 2)
  : (1 - (s * (1 - u) + t * (1 - s) + u * (1 - t))) = 7 / 32 := by
  sorry

end ratio_of_areas_l154_154979


namespace percentage_liked_B_l154_154605

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l154_154605


namespace Jacob_age_is_3_l154_154977

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end Jacob_age_is_3_l154_154977


namespace votes_for_sue_l154_154268

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end votes_for_sue_l154_154268


namespace product_n_equals_7200_l154_154326

theorem product_n_equals_7200 :
  (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 ^ 2 + 1) = 7200 := by
  sorry

end product_n_equals_7200_l154_154326


namespace eq_solution_l154_154272

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end eq_solution_l154_154272


namespace find_other_number_l154_154183

def smallest_multiple_of_711 (n : ℕ) : ℕ := Nat.lcm n 711

theorem find_other_number (n : ℕ) : smallest_multiple_of_711 n = 3555 → n = 5 := by
  sorry

end find_other_number_l154_154183


namespace intersection_A_B_l154_154823

open Set

def A : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}
def B : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 5} := by
  sorry

end intersection_A_B_l154_154823


namespace triangle_perimeter_l154_154342

-- Define the triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the predicate that checks if the triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the predicate that calculates the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- State the problem
theorem triangle_perimeter : 
  ∃ (t : Triangle), isIsosceles t ∧ (    (t.a = 6 ∧ t.b = 9 ∧ perimeter t = 24)
                                       ∨ (t.b = 6 ∧ t.a = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.a = 9 ∧ perimeter t = 21)
                                       ∨ (t.a = 6 ∧ t.c = 9 ∧ perimeter t = 21)
                                       ∨ (t.b = 6 ∧ t.c = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.b = 9 ∧ perimeter t = 21)
                                    ) :=
sorry

end triangle_perimeter_l154_154342


namespace intersection_of_M_and_N_l154_154677

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end intersection_of_M_and_N_l154_154677


namespace value_of_x_div_y_l154_154994

theorem value_of_x_div_y (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, x = t * y ∧ t = -2 := 
sorry

end value_of_x_div_y_l154_154994


namespace man_average_interest_rate_l154_154001

noncomputable def average_rate_of_interest (total_investment : ℝ) (rate1 rate2 rate_average : ℝ) 
    (x : ℝ) (same_return : (rate1 * (total_investment - x) = rate2 * x)) : Prop :=
  (rate_average = ((rate1 * (total_investment - x) + rate2 * x) / total_investment))

theorem man_average_interest_rate
    (total_investment : ℝ) 
    (rate1 : ℝ)
    (rate2 : ℝ)
    (rate_average : ℝ)
    (x : ℝ)
    (same_return : rate1 * (total_investment - x) = rate2 * x) :
    total_investment = 4500 ∧ rate1 = 0.04 ∧ rate2 = 0.06 ∧ x = 1800 ∧ rate_average = 0.048 → 
    average_rate_of_interest total_investment rate1 rate2 rate_average x same_return := 
by
  sorry

end man_average_interest_rate_l154_154001


namespace percentage_liked_B_l154_154604

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l154_154604


namespace min_expression_value_l154_154360

theorem min_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ x : ℝ, x = 5 ∧ ∀ y, (y = (b / (3 * a) + 3 / b)) → x ≤ y :=
by
  sorry

end min_expression_value_l154_154360


namespace mileage_in_scientific_notation_l154_154469

noncomputable def scientific_notation_of_mileage : Prop :=
  let mileage := 42000
  mileage = 4.2 * 10^4

theorem mileage_in_scientific_notation :
  scientific_notation_of_mileage :=
by
  sorry

end mileage_in_scientific_notation_l154_154469


namespace roots_expression_value_l154_154495

theorem roots_expression_value {a b : ℝ} 
  (h₁ : a^2 + a - 3 = 0) 
  (h₂ : b^2 + b - 3 = 0) 
  (ha_ne_hb : a ≠ b) : 
  a * b - 2023 * a - 2023 * b = 2020 :=
by 
  sorry

end roots_expression_value_l154_154495


namespace symmetric_with_origin_l154_154114

-- Define the original point P
def P : ℝ × ℝ := (2, -3)

-- Define the function for finding the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Prove that the symmetric point of P with respect to the origin is (-2, 3)
theorem symmetric_with_origin :
  symmetric_point P = (-2, 3) :=
by
  -- Placeholders for proof
  sorry

end symmetric_with_origin_l154_154114


namespace study_time_in_minutes_l154_154858

theorem study_time_in_minutes :
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60 = 540 :=
by
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  sorry

end study_time_in_minutes_l154_154858


namespace smallest_composite_no_prime_factors_less_than_15_l154_154643

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l154_154643


namespace quadratic_inequality_solution_l154_154886

theorem quadratic_inequality_solution :
  {x : ℝ | 2*x^2 - 3*x - 2 ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2} :=
sorry

end quadratic_inequality_solution_l154_154886


namespace intersection_M_N_l154_154063

def M : Set ℤ := { x | x^2 > 1 }
def N : Set ℤ := { -2, -1, 0, 1, 2 }

theorem intersection_M_N : (M ∩ N) = { -2, 2 } :=
sorry

end intersection_M_N_l154_154063


namespace transform_polynomial_l154_154687

theorem transform_polynomial (x y : ℝ) (h1 : y = x + 1 / x) (h2 : x^4 - x^3 - 6 * x^2 - x + 1 = 0) :
  x^2 * (y^2 - y - 6) = 0 := 
  sorry

end transform_polynomial_l154_154687


namespace smallest_four_digit_mod_8_l154_154432

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l154_154432


namespace fraction_studying_japanese_l154_154445

variable (J S : ℕ)
variable (hS : S = 3 * J)

def fraction_of_seniors_studying_japanese := (1 / 3 : ℚ) * S
def fraction_of_juniors_studying_japanese := (3 / 4 : ℚ) * J

def total_students := S + J

theorem fraction_studying_japanese (J S : ℕ) (hS : S = 3 * J) :
  ((1 / 3 : ℚ) * S + (3 / 4 : ℚ) * J) / (S + J) = 7 / 16 :=
by {
  -- proof to be filled in
  sorry
}

end fraction_studying_japanese_l154_154445


namespace problem_statement_l154_154847

/-- 
  Theorem: If the solution set of the inequality (ax-1)(x+2) > 0 is -3 < x < -2, 
  then a equals -1/3 
--/
theorem problem_statement (a : ℝ) :
  (forall x, (ax-1)*(x+2) > 0 -> -3 < x ∧ x < -2) → a = -1/3 := 
by
  sorry

end problem_statement_l154_154847


namespace polynomial_integer_root_l154_154937

theorem polynomial_integer_root (b : ℤ) :
  (∃ x : ℤ, x^3 + 5 * x^2 + b * x + 9 = 0) ↔ b = -127 ∨ b = -74 ∨ b = -27 ∨ b = -24 ∨ b = -15 ∨ b = -13 :=
by
  sorry

end polynomial_integer_root_l154_154937


namespace find_a_l154_154349
-- Import the entire Mathlib to ensure all necessary primitives and theorems are available.

-- Define a constant equation representing the conditions.
def equation (x a : ℝ) := 3 * x + 2 * a

-- Define a theorem to prove the condition => result structure.
theorem find_a (h : equation 2 a = 0) : a = -3 :=
by sorry

end find_a_l154_154349


namespace Xiao_Ming_max_notebooks_l154_154128

-- Definitions of the given conditions
def total_yuan : ℝ := 30
def total_books : ℕ := 30
def notebook_cost : ℝ := 4
def exercise_book_cost : ℝ := 0.4

-- Definition of the variables used in the inequality
def x (max_notebooks : ℕ) : ℝ := max_notebooks
def exercise_books (max_notebooks : ℕ) : ℝ := total_books - x max_notebooks

-- Definition of the total cost inequality
def total_cost (max_notebooks : ℕ) : ℝ :=
  x max_notebooks * notebook_cost + exercise_books max_notebooks * exercise_book_cost

theorem Xiao_Ming_max_notebooks (max_notebooks : ℕ) : total_cost max_notebooks ≤ total_yuan → max_notebooks ≤ 5 :=
by
  -- Proof goes here
  sorry

end Xiao_Ming_max_notebooks_l154_154128


namespace imaginary_unit_power_l154_154511

def i := Complex.I

theorem imaginary_unit_power :
  ∀ a : ℝ, (2 - i + a * i ^ 2011).im = 0 → i ^ 2011 = i :=
by
  intro a
  intro h
  sorry

end imaginary_unit_power_l154_154511


namespace liked_product_B_l154_154609

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l154_154609


namespace Winnie_the_Pooh_honey_consumption_l154_154127

theorem Winnie_the_Pooh_honey_consumption (W0 W1 W2 W3 W4 : ℝ) (pot_empty : ℝ) 
  (h1 : W1 = W0 / 2)
  (h2 : W2 = W1 / 2)
  (h3 : W3 = W2 / 2)
  (h4 : W4 = W3 / 2)
  (h5 : W4 = 200)
  (h6 : pot_empty = 200) : 
  W0 - 200 = 3000 := by
  sorry

end Winnie_the_Pooh_honey_consumption_l154_154127


namespace divisibility_ac_bd_l154_154250

-- Conditions definitions
variable (a b c d : ℕ)
variable (hab : a ∣ b)
variable (hcd : c ∣ d)

-- Goal
theorem divisibility_ac_bd : (a * c) ∣ (b * d) :=
  sorry

end divisibility_ac_bd_l154_154250


namespace max_isosceles_triangles_l154_154489

theorem max_isosceles_triangles 
  {A B C D P : ℝ} 
  (h_collinear: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D)
  (h_non_collinear: P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D)
  : (∀ a b c : ℝ, (a = P ∨ a = A ∨ a = B ∨ a = C ∨ a = D) ∧ (b = P ∨ b = A ∨ b = B ∨ b = C ∨ b = D) ∧ (c = P ∨ c = A ∨ c = B ∨ c = C ∨ c = D) 
    ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((a - b)^2 + (b - c)^2 = (a - c)^2 ∨ (a - c)^2 + (b - c)^2 = (a - b)^2 ∨ (a - b)^2 + (a - c)^2 = (b - c)^2)) → 
    isosceles_triangle_count = 6 :=
sorry

end max_isosceles_triangles_l154_154489


namespace min_b_minus_2c_over_a_l154_154966

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (h1 : a ≤ b + c ∧ b + c ≤ 3 * a)
variable (h2 : 3 * b^2 ≤ a * (a + c) ∧ a * (a + c) ≤ 5 * b^2)

theorem min_b_minus_2c_over_a : (∃ u : ℝ, (u = (b - 2 * c) / a) ∧ (∀ v : ℝ, (v = (b - 2 * c) / a) → u ≤ v)) :=
  sorry

end min_b_minus_2c_over_a_l154_154966


namespace initial_weight_of_cheese_l154_154135

theorem initial_weight_of_cheese :
  let initial_weight : Nat := 850
  -- final state after 3 bites
  let final_weight1 : Nat := 25
  let final_weight2 : Nat := 25
  -- third state
  let third_weight1 : Nat := final_weight1 + final_weight2
  let third_weight2 : Nat := final_weight1
  -- second state
  let second_weight1 : Nat := third_weight1 + third_weight2
  let second_weight2 : Nat := third_weight1
  -- first state
  let first_weight1 : Nat := second_weight1 + second_weight2
  let first_weight2 : Nat := second_weight1
  -- initial state
  let initial_weight1 : Nat := first_weight1 + first_weight2
  let initial_weight2 : Nat := first_weight1
  initial_weight = initial_weight1 + initial_weight2 :=
by
  sorry

end initial_weight_of_cheese_l154_154135


namespace evaluate_expression_l154_154745

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 :=
by
  sorry

end evaluate_expression_l154_154745


namespace inequality_log_equality_log_l154_154380

theorem inequality_log (x : ℝ) (hx : x < 0 ∨ x > 0) :
  max 0 (Real.log (|x|)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) := 
sorry

theorem equality_log (x : ℝ) :
  (max 0 (Real.log (|x|)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) := 
sorry

end inequality_log_equality_log_l154_154380


namespace days_b_worked_l154_154294

theorem days_b_worked (A_days B_days A_remaining_days : ℝ) (A_work_rate B_work_rate total_work : ℝ)
  (hA_rate : A_work_rate = 1 / A_days)
  (hB_rate : B_work_rate = 1 / B_days)
  (hA_days : A_days = 9)
  (hB_days : B_days = 15)
  (hA_remaining : A_remaining_days = 3)
  (h_total_work : ∀ x : ℝ, (x * B_work_rate + A_remaining_days * A_work_rate = total_work)) :
  ∃ x : ℝ, x = 10 :=
by
  sorry

end days_b_worked_l154_154294


namespace shares_difference_l154_154466

noncomputable def Faruk_share (V : ℕ) : ℕ := (3 * (V / 5))
noncomputable def Ranjith_share (V : ℕ) : ℕ := (7 * (V / 5))

theorem shares_difference {V : ℕ} (hV : V = 1500) : 
  Ranjith_share V - Faruk_share V = 1200 :=
by
  rw [Faruk_share, Ranjith_share]
  subst hV
  -- It's just a declaration of the problem and sorry is used to skip the proof.
  sorry

end shares_difference_l154_154466


namespace skillful_hands_wire_cut_l154_154287

theorem skillful_hands_wire_cut :
  ∃ x : ℕ, (1000 = 15 * x) ∧ (1040 = 15 * x) ∧ x = 66 :=
by
  sorry

end skillful_hands_wire_cut_l154_154287


namespace inverse_proposition_l154_154562

theorem inverse_proposition (a b c : ℝ) : (a > b → a + c > b + c) → (a + c > b + c → a > b) :=
sorry

end inverse_proposition_l154_154562


namespace meaningful_sqrt_l154_154686

theorem meaningful_sqrt (a : ℝ) (h : a ≥ 4) : a = 6 ↔ ∃ x ∈ ({-1, 0, 2, 6} : Set ℝ), x = 6 := 
by
  sorry

end meaningful_sqrt_l154_154686


namespace rita_remaining_amount_l154_154877

theorem rita_remaining_amount :
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  remaining_amount = initial_amount - total_cost
  in remaining_amount = 139 :=
by
  let num_dresses := 5
  let num_pants := 3
  let num_jackets := 4
  let cost_per_dress := 20
  let cost_per_pants := 12
  let cost_per_jacket := 30
  let additional_cost := 5
  let initial_amount := 400
  let total_cost := num_dresses * cost_per_dress + num_pants * cost_per_pants + num_jackets * cost_per_jacket + additional_cost
  let remaining_amount := initial_amount - total_cost
  show remaining_amount = 139
  sorry

end rita_remaining_amount_l154_154877


namespace triangle_perimeter_l154_154343

-- Define the triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the predicate that checks if the triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the predicate that calculates the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- State the problem
theorem triangle_perimeter : 
  ∃ (t : Triangle), isIsosceles t ∧ (    (t.a = 6 ∧ t.b = 9 ∧ perimeter t = 24)
                                       ∨ (t.b = 6 ∧ t.a = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.a = 9 ∧ perimeter t = 21)
                                       ∨ (t.a = 6 ∧ t.c = 9 ∧ perimeter t = 21)
                                       ∨ (t.b = 6 ∧ t.c = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.b = 9 ∧ perimeter t = 21)
                                    ) :=
sorry

end triangle_perimeter_l154_154343


namespace log_base_3_of_9_cubed_l154_154035

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l154_154035


namespace range_of_k_l154_154848

-- Definitions to use in statement
variable (k : ℝ)

-- Statement: Proving the range of k
theorem range_of_k (h : ∀ x : ℝ, k * x^2 - k * x - 1 < 0) : -4 < k ∧ k ≤ 0 :=
  sorry

end range_of_k_l154_154848


namespace inverse_contrapositive_l154_154731

theorem inverse_contrapositive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : a^2 + b^2 ≠ 0 :=
sorry

end inverse_contrapositive_l154_154731


namespace smallest_composite_no_prime_factors_lt_15_l154_154639

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l154_154639


namespace most_likely_event_is_C_l154_154300

open Classical

noncomputable def total_events : ℕ := 6 * 6

noncomputable def P_A : ℚ := 7 / 36
noncomputable def P_B : ℚ := 18 / 36
noncomputable def P_C : ℚ := 1
noncomputable def P_D : ℚ := 0

theorem most_likely_event_is_C :
  P_C > P_A ∧ P_C > P_B ∧ P_C > P_D := by
  sorry

end most_likely_event_is_C_l154_154300


namespace spherical_to_rectangular_coordinates_l154_154320

noncomputable
def convert_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  let ρ1 := 10
  let θ1 := Real.pi / 4
  let φ1 := Real.pi / 6
  let ρ2 := 15
  let θ2 := 5 * Real.pi / 4
  let φ2 := Real.pi / 3
  convert_to_cartesian ρ1 θ1 φ1 = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  ∧ convert_to_cartesian ρ2 θ2 φ2 = (-15 * Real.sqrt 6 / 4, -15 * Real.sqrt 6 / 4, 7.5) := 
by
  sorry

end spherical_to_rectangular_coordinates_l154_154320


namespace present_age_of_son_l154_154304

variable (S M : ℕ)

-- Conditions
def age_difference : Prop := M = S + 40
def age_relation_in_seven_years : Prop := M + 7 = 3 * (S + 7)

-- Theorem to prove
theorem present_age_of_son : age_difference S M → age_relation_in_seven_years S M → S = 13 := by
  sorry

end present_age_of_son_l154_154304


namespace cats_joined_l154_154453

theorem cats_joined (c : ℕ) (h : 1 + c + 2 * c + 6 * c = 37) : c = 4 :=
sorry

end cats_joined_l154_154453


namespace series_value_l154_154016

theorem series_value :
  ∑ n in Finset.range 120, (-1) ^ n * (n^3 + (n - 1)^3) = 1728000 :=
by
  sorry

end series_value_l154_154016


namespace football_team_birthday_collision_moscow_birthday_collision_l154_154904

theorem football_team_birthday_collision (n : ℕ) (k : ℕ) (h1 : n ≥ 11) (h2 : k = 7) : 
  ∃ (d : ℕ) (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ n ∧ p2 ≤ n ∧ d ≤ k :=
by sorry

theorem moscow_birthday_collision (population : ℕ) (days : ℕ) (h1 : population > 10000000) (h2 : days = 366) :
  ∃ (day : ℕ) (count : ℕ), count ≥ 10000 ∧ count ≤ population / days :=
by sorry

end football_team_birthday_collision_moscow_birthday_collision_l154_154904


namespace smallest_arithmetic_mean_l154_154410

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l154_154410


namespace number_of_birds_flew_up_correct_l154_154760

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end number_of_birds_flew_up_correct_l154_154760


namespace woman_age_multiple_l154_154308

theorem woman_age_multiple (S : ℕ) (W : ℕ) (k : ℕ) 
  (h1 : S = 27)
  (h2 : W + S = 84)
  (h3 : W = k * S + 3) :
  k = 2 :=
by
  sorry

end woman_age_multiple_l154_154308


namespace log_base_3_of_9_cubed_l154_154039

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l154_154039


namespace shirt_cost_l154_154446

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 :=
by
  sorry

end shirt_cost_l154_154446


namespace yujin_wire_length_is_correct_l154_154371

def junhoe_wire_length : ℝ := 134.5
def multiplicative_factor : ℝ := 1.06
def yujin_wire_length (junhoe_length : ℝ) (factor : ℝ) : ℝ := junhoe_length * factor

theorem yujin_wire_length_is_correct : 
  yujin_wire_length junhoe_wire_length multiplicative_factor = 142.57 := 
by 
  sorry

end yujin_wire_length_is_correct_l154_154371


namespace problem_statement_l154_154056

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : (x - y - z) ^ 2002 = 0 :=
sorry

end problem_statement_l154_154056


namespace ratio_of_pieces_l154_154601

theorem ratio_of_pieces (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 ∧ shorter_piece = 20 → shorter_piece / (total_length - shorter_piece) = 1 / 2 :=
by
  sorry

end ratio_of_pieces_l154_154601


namespace total_weight_full_bucket_l154_154747

theorem total_weight_full_bucket (x y p q : ℝ)
  (h1 : x + (3 / 4) * y = p)
  (h2 : x + (1 / 3) * y = q) :
  x + y = (8 * p - 11 * q) / 5 :=
by
  sorry

end total_weight_full_bucket_l154_154747


namespace scientific_notation_l154_154769

def significant_digits : ℝ := 4.032
def exponent : ℤ := 11
def original_number : ℝ := 403200000000

theorem scientific_notation : original_number = significant_digits * 10 ^ exponent := 
by
  sorry

end scientific_notation_l154_154769


namespace trigonometric_expression_l154_154191

theorem trigonometric_expression (x : ℝ) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := 
sorry

end trigonometric_expression_l154_154191


namespace largest_circle_area_l154_154269

theorem largest_circle_area (PQ QR PR : ℝ)
  (h_right_triangle: PR^2 = PQ^2 + QR^2)
  (h_circle_areas_sum: π * (PQ/2)^2 + π * (QR/2)^2 + π * (PR/2)^2 = 338 * π) :
  π * (PR/2)^2 = 169 * π :=
by
  sorry

end largest_circle_area_l154_154269


namespace log_three_nine_cubed_l154_154037

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l154_154037


namespace chimes_in_a_day_l154_154064

-- Definitions for the conditions
def strikes_in_12_hours : ℕ :=
  (1 + 12) * 12 / 2

def strikes_in_24_hours : ℕ :=
  2 * strikes_in_12_hours

def half_hour_strikes : ℕ :=
  24 * 2

def total_chimes_in_a_day : ℕ :=
  strikes_in_24_hours + half_hour_strikes

-- Statement to prove
theorem chimes_in_a_day : total_chimes_in_a_day = 204 :=
by 
  -- The proof would be placed here
  sorry

end chimes_in_a_day_l154_154064


namespace log_pow_evaluation_l154_154042

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l154_154042


namespace ellipse_eccentricity_l154_154353

theorem ellipse_eccentricity (a b c : ℝ) (h_eq : a * a = 16) (h_b : b * b = 12) (h_c : c * c = a * a - b * b) :
  c / a = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l154_154353


namespace sum_abc_l154_154275

theorem sum_abc (A B C : ℕ) (hposA : 0 < A) (hposB : 0 < B) (hposC : 0 < C) (hgcd : Nat.gcd A (Nat.gcd B C) = 1)
  (hlog : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : A + B + C = 5 :=
sorry

end sum_abc_l154_154275


namespace log_pow_evaluation_l154_154041

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l154_154041


namespace cos_B_in_triangle_l154_154192

theorem cos_B_in_triangle
  (A B C a b c : ℝ)
  (h1 : Real.sin A = 2 * Real.sin C)
  (h2 : b^2 = a * c)
  (h3 : 0 < b)
  (h4 : 0 < c)
  (h5 : a = 2 * c)
  : Real.cos B = 3 / 4 := 
sorry

end cos_B_in_triangle_l154_154192


namespace test_average_score_l154_154383

theorem test_average_score (A : ℝ) (h : 0.90 * A + 5 = 86) : A = 90 := 
by
  sorry

end test_average_score_l154_154383


namespace opposite_of_minus_one_third_l154_154567

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_minus_one_third_l154_154567


namespace hall_breadth_is_12_l154_154301

/-- Given a hall with length 15 meters, if the sum of the areas of the floor and the ceiling 
    is equal to the sum of the areas of the four walls and the volume of the hall is 1200 
    cubic meters, then the breadth of the hall is 12 meters. -/
theorem hall_breadth_is_12 (b h : ℝ) (h1 : 15 * b * h = 1200)
  (h2 : 2 * (15 * b) = 2 * (15 * h) + 2 * (b * h)) : b = 12 :=
sorry

end hall_breadth_is_12_l154_154301


namespace find_multiplier_l154_154899

theorem find_multiplier (x : ℝ) (h : (9 / 6) * x = 18) : x = 12 := sorry

end find_multiplier_l154_154899


namespace cats_given_by_Mr_Sheridan_l154_154097

-- Definitions of the initial state and final state
def initial_cats : Nat := 17
def total_cats : Nat := 31

-- Proof statement that Mr. Sheridan gave her 14 cats
theorem cats_given_by_Mr_Sheridan : total_cats - initial_cats = 14 := by
  sorry

end cats_given_by_Mr_Sheridan_l154_154097


namespace units_digit_expression_l154_154932

lemma units_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 := sorry

lemma units_digit_5_pow_2024 : (5 ^ 2024) % 10 = 5 := sorry

lemma units_digit_11_pow_2025 : (11 ^ 2025) % 10 = 1 := sorry

theorem units_digit_expression : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by 
  have h1 := units_digit_2_pow_2023
  have h2 := units_digit_5_pow_2024
  have h3 := units_digit_11_pow_2025
  sorry

end units_digit_expression_l154_154932


namespace cyclic_quadrilateral_iff_condition_l154_154443

theorem cyclic_quadrilateral_iff_condition
  (α β γ δ : ℝ)
  (h : α + β + γ + δ = 2 * π) :
  (α * β + α * δ + γ * β + γ * δ = π^2) ↔ (α + γ = π ∧ β + δ = π) :=
by
  sorry

end cyclic_quadrilateral_iff_condition_l154_154443


namespace quadrilateral_sum_of_squares_l154_154872

theorem quadrilateral_sum_of_squares
  (a b c d m n t : ℝ) : 
  a^2 + b^2 + c^2 + d^2 = m^2 + n^2 + 4 * t^2 :=
sorry

end quadrilateral_sum_of_squares_l154_154872


namespace ship_length_correct_l154_154151

noncomputable def ship_length : ℝ :=
  let speed_kmh := 24
  let speed_mps := speed_kmh * 1000 / 3600
  let time := 202.48
  let bridge_length := 900
  let total_distance := speed_mps * time
  total_distance - bridge_length

theorem ship_length_correct : ship_length = 450.55 :=
by
  -- This is where the proof would be written, but we're skipping the proof as per instructions
  sorry

end ship_length_correct_l154_154151


namespace suzanna_history_book_pages_l154_154256

theorem suzanna_history_book_pages (H G M S : ℕ) 
  (h_geography : G = H + 70)
  (h_math : M = (1 / 2) * (H + H + 70))
  (h_science : S = 2 * H)
  (h_total : H + G + M + S = 905) : 
  H = 160 := 
by
  sorry

end suzanna_history_book_pages_l154_154256


namespace arithmetic_sequence_problem_l154_154195

variables {a : ℕ → ℕ} (d a1 : ℕ)

def arithmetic_sequence (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : arithmetic_sequence 1 + arithmetic_sequence 3 + arithmetic_sequence 9 = 20) :
  4 * arithmetic_sequence 5 - arithmetic_sequence 7 = 20 :=
by
  sorry

end arithmetic_sequence_problem_l154_154195


namespace nine_consecutive_arithmetic_mean_divisible_1111_l154_154389

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l154_154389


namespace line_passes_through_center_l154_154073

theorem line_passes_through_center (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end line_passes_through_center_l154_154073


namespace chord_length_of_intersection_l154_154800

theorem chord_length_of_intersection 
  (A B C : ℝ) (x0 y0 r : ℝ)
  (line_eq : A * x0 + B * y0 + C = 0)
  (circle_eq : (x0 - 1)^2 + (y0 - 3)^2 = r^2) 
  (A_line : A = 4) (B_line : B = -3) (C_line : C = 0) 
  (x0_center : x0 = 1) (y0_center : y0 = 3) (r_circle : r^2 = 10) :
  2 * (Real.sqrt (r^2 - ((A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2)))^2)) = 6 :=
by
  sorry

end chord_length_of_intersection_l154_154800


namespace interval_monotonically_decreasing_l154_154799

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 3)

theorem interval_monotonically_decreasing :
  ∀ x y : ℝ, 1 < x → x < 3 → 1 < y → y < 3 → x < y → f y < f x := 
by sorry

end interval_monotonically_decreasing_l154_154799


namespace percent_profit_l154_154461

theorem percent_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) (final_profit_percent : ℝ)
  (h1 : cost = 50)
  (h2 : markup_percent = 30)
  (h3 : discount_percent = 10)
  (h4 : final_profit_percent = 17)
  : (markup_percent / 100 * cost - discount_percent / 100 * (cost + markup_percent / 100 * cost)) / cost * 100 = final_profit_percent := 
by
  sorry

end percent_profit_l154_154461


namespace product_is_even_l154_154863

-- Definitions are captured from conditions

noncomputable def is_permutation {α : Type*} [DecidableEq α] (l1 l2 : List α) : Prop :=
  l1.perm l2

theorem product_is_even (a : ℕ → ℕ) :
  (is_permutation (List.range 2015) (List.ofFn (λ i, a i + 2014))) →
  Even (Finset.univ.prod (λ i : Fin 2015, a i - i.val.succ)) :=
by
  sorry

end product_is_even_l154_154863


namespace range_of_m_l154_154516

-- Conditions:
def is_opposite_sides_of_line (p1 p2 : ℝ × ℝ) (a b m : ℝ) : Prop :=
  let l1 := a * p1.1 + b * p1.2 + m
  let l2 := a * p2.1 + b * p2.2 + m
  l1 * l2 < 0

-- Point definitions:
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-4, -2)

-- Line definition with coefficients
def a : ℝ := 2
def b : ℝ := 1

-- Proof Goal:
theorem range_of_m (m : ℝ) : is_opposite_sides_of_line point1 point2 a b m ↔ -5 < m ∧ m < 10 :=
by sorry

end range_of_m_l154_154516


namespace chord_length_circle_l154_154108

theorem chord_length_circle {x y : ℝ} :
  (x - 1)^2 + (y - 1)^2 = 2 →
  (exists (p q : ℝ), (p-1)^2 = 1 ∧ (q-1)^2 = 1 ∧ p ≠ q ∧ abs (p - q) = 2) :=
by
  intro h
  use (2 : ℝ)
  use (0 : ℝ)
  -- Formal proof omitted
  sorry

end chord_length_circle_l154_154108


namespace positive_y_equals_32_l154_154170

theorem positive_y_equals_32 (y : ℝ) (h : y^2 = 1024) (hy : 0 < y) : y = 32 :=
sorry

end positive_y_equals_32_l154_154170


namespace A1_lies_on_nine_point_circle_l154_154981

open EuclideanGeometry

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def reflection (P O : Point) : Point := sorry
noncomputable def foot_of_altitude (A B C : Point) : Point := sorry
noncomputable def lies_on_circle (P O R : Point) : Prop := sorry

theorem A1_lies_on_nine_point_circle (A B C : Point) :
  let H := orthocenter A B C
  let O := circumcenter A B C
  let H' := reflection H O
  let A1 := foot_of_altitude A B C
  let nine_point_circle := ninePointCircle A B C
  H' ∈ line_through B C →
  lies_on_circle A1 nine_point_circle :=
by
  intros H H' O A1 nine_point_circle hH'_on_BC
  sorry

end A1_lies_on_nine_point_circle_l154_154981


namespace largest_divisor_l154_154441

theorem largest_divisor (A B : ℕ) (h : 24 = A * B + 4) : A ≤ 20 :=
sorry

end largest_divisor_l154_154441


namespace proof_large_long_brown_dogs_l154_154364

open Finset

variable (D : Finset α) (L B La S : Finset α)

variables (hD : card D = 60)
variables (hL : card L = 35)
variables (hB : card B = 25)
variables (hNeitherLB : card (D \ (L ∪ B)) = 10)
variables (hLa : card La = 30)
variables (hS : card S = 30)
variables (hSmallBrown : card (S ∩ B) = 14)
variables (hLargeLongNotBrown : card (La ∩ L \ B) = 7)

theorem proof_large_long_brown_dogs : card (La ∩ L ∩ B) = 6 :=
  sorry

end proof_large_long_brown_dogs_l154_154364


namespace actual_miles_traveled_l154_154303

def skipped_digits_odometer (digits : List ℕ) : Prop :=
  digits = [0, 1, 2, 3, 6, 7, 8, 9]

theorem actual_miles_traveled (odometer_reading : String) (actual_miles : ℕ) :
  skipped_digits_odometer [0, 1, 2, 3, 6, 7, 8, 9] →
  odometer_reading = "000306" →
  actual_miles = 134 :=
by
  intros
  sorry

end actual_miles_traveled_l154_154303


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l154_154435

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l154_154435


namespace count_three_element_arithmetic_mean_subsets_l154_154476
open Nat

theorem count_three_element_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
    ∃ a_n : ℕ, a_n = (n / 2) * ((n - 1) / 2) :=
by
  sorry

end count_three_element_arithmetic_mean_subsets_l154_154476


namespace avoid_mistakes_l154_154898

-- Definitions for conditions
def minions_shared_info (m : string) (website : string) := m = "Gru's phone number" ∧ website = "untrusted website"

def minions_downloaded_exe (file : string) := file = "banana_cocktail.pdf.exe"

-- Assumption representing the mistakes
axiom minions_mistake (m : string) (w : string) (f : string) :
  minions_shared_info m w ∧ minions_downloaded_exe f

-- Theorem to prove the correct answer
theorem avoid_mistakes (share_info : ∀ (m : string) (w : string), ¬ (minions_shared_info m w))
  (download_file : ∀ (f : string), ¬ (minions_downloaded_exe f)) :
  ∀ (m : string) (w : string) (f : string), ¬ (minions_shared_info m w ∧ minions_downloaded_exe f) :=
by
  intros
  sorry

end avoid_mistakes_l154_154898


namespace bake_four_pans_l154_154525

-- Define the conditions
def bake_time_one_pan : ℕ := 7
def total_bake_time (n : ℕ) : ℕ := 28

-- Define the theorem statement
theorem bake_four_pans : total_bake_time 4 = 28 :=
by
  -- Proof is omitted
  sorry

end bake_four_pans_l154_154525


namespace cans_per_person_day1_l154_154624

theorem cans_per_person_day1
  (initial_cans : ℕ)
  (people_day1 : ℕ)
  (restock_day1 : ℕ)
  (people_day2 : ℕ)
  (cans_per_person_day2 : ℕ)
  (total_cans_given_away : ℕ) :
  initial_cans = 2000 →
  people_day1 = 500 →
  restock_day1 = 1500 →
  people_day2 = 1000 →
  cans_per_person_day2 = 2 →
  total_cans_given_away = 2500 →
  (total_cans_given_away - (people_day2 * cans_per_person_day2)) / people_day1 = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- condition trivially holds
  sorry

end cans_per_person_day1_l154_154624


namespace comprehensive_score_l154_154779

variable (regularAssessmentScore : ℕ)
variable (finalExamScore : ℕ)
variable (regularAssessmentWeighting : ℝ)
variable (finalExamWeighting : ℝ)

theorem comprehensive_score 
  (h1 : regularAssessmentScore = 95)
  (h2 : finalExamScore = 90)
  (h3 : regularAssessmentWeighting = 0.20)
  (h4 : finalExamWeighting = 0.80) :
  (regularAssessmentScore * regularAssessmentWeighting + finalExamScore * finalExamWeighting) = 91 :=
sorry

end comprehensive_score_l154_154779


namespace John_leftover_money_l154_154702

variables (q : ℝ)

def drinks_price (q : ℝ) : ℝ := 4 * q
def small_pizza_price (q : ℝ) : ℝ := q
def large_pizza_price (q : ℝ) : ℝ := 4 * q
def total_cost (q : ℝ) : ℝ := drinks_price q + small_pizza_price q + 2 * large_pizza_price q
def John_initial_money : ℝ := 50
def John_money_left (q : ℝ) : ℝ := John_initial_money - total_cost q

theorem John_leftover_money : John_money_left q = 50 - 13 * q :=
by
  sorry

end John_leftover_money_l154_154702


namespace sqrt_meaningful_implies_x_ge_2_l154_154510

theorem sqrt_meaningful_implies_x_ge_2 (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 := 
sorry

end sqrt_meaningful_implies_x_ge_2_l154_154510


namespace range_of_a_l154_154344

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, (0 < x) ∧ (x + 1/x < a)) ↔ a ≤ 2 :=
by {
  sorry
}

end range_of_a_l154_154344


namespace max_value_of_g_on_interval_l154_154178

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g_on_interval : ∃ x : ℝ, (0 ≤ x ∧ x ≤ Real.sqrt 2) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ Real.sqrt 2) → g y ≤ g x) ∧ g x = 25 / 8 := by
  sorry

end max_value_of_g_on_interval_l154_154178


namespace bianca_ate_candies_l154_154657

-- Definitions based on the conditions
def total_candies : ℕ := 32
def pieces_per_pile : ℕ := 5
def number_of_piles : ℕ := 4

-- The statement to prove
theorem bianca_ate_candies : 
  total_candies - (pieces_per_pile * number_of_piles) = 12 := 
by 
  sorry

end bianca_ate_candies_l154_154657


namespace friends_meeting_time_l154_154889

noncomputable def speed_B (t : ℕ) : ℝ := 4 + 0.75 * (t - 1)

noncomputable def distance_B (t : ℕ) : ℝ :=
  t * 4 + (0.375 * t * (t - 1))

noncomputable def distance_A (t : ℕ) : ℝ := 5 * t

theorem friends_meeting_time :
  ∃ t : ℝ, 5 * t + (t / 2) * (7.25 + 0.75 * t) = 120 ∧ t = 8 :=
by
  sorry

end friends_meeting_time_l154_154889


namespace smallest_composite_no_prime_factors_below_15_correct_l154_154655

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l154_154655


namespace words_per_page_l154_154138

theorem words_per_page (p : ℕ) (hp : p ≤ 120) (h : 150 * p ≡ 210 [MOD 221]) : p = 98 := by
  sorry

end words_per_page_l154_154138


namespace find_a_l154_154227

theorem find_a (a b : ℝ) (h₀ : b = 4) (h₁ : (4, b) ∈ {p | p.snd = 0.75 * p.fst + 1}) 
  (h₂ : (a, 5) ∈ {p | p.snd = 0.75 * p.fst + 1}) (h₃ : (a, b+1) ∈ {p | p.snd = 0.75 * p.fst + 1}) : 
  a = 5.33 :=
by 
  sorry

end find_a_l154_154227


namespace probability_normal_distribution_l154_154352

theorem probability_normal_distribution (ξ : ℝ → ℝ) (δ : ℝ) 
  (H1 : ξ ∼ Normal 2 δ^2)
  (H2 : ∫ (x : ℝ) in -∞..3, ξ x = 0.8413) : 
  ∫ (x : ℝ) in -∞..1, ξ x = 0.1587 := 
sorry

end probability_normal_distribution_l154_154352


namespace problems_completed_l154_154842

theorem problems_completed (p t : ℕ) (h1 : p ≥ 15) (h2 : p * t = (2 * p - 10) * (t - 1)) : p * t = 60 := sorry

end problems_completed_l154_154842


namespace train_crossing_time_l154_154008

def train_length : ℕ := 1000
def train_speed_km_per_h : ℕ := 18
def train_speed_m_per_s := train_speed_km_per_h * 1000 / 3600

theorem train_crossing_time :
  train_length / train_speed_m_per_s = 200 := by
sorry

end train_crossing_time_l154_154008


namespace log_base_three_of_nine_cubed_l154_154030

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l154_154030


namespace age_ratio_l154_154582

variable (A B : ℕ)
variable (k : ℕ)

-- Define the conditions
def sum_of_ages : Prop := A + B = 60
def multiple_of_age : Prop := A = k * B

-- Theorem to prove the ratio of ages
theorem age_ratio (h_sum : sum_of_ages A B) (h_multiple : multiple_of_age A B k) : A = 12 * B :=
by
  sorry

end age_ratio_l154_154582


namespace opposite_neg_inv_three_l154_154571

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end opposite_neg_inv_three_l154_154571


namespace problem_part1_problem_part2_problem_part3_l154_154200

-- Define the context for the problem
variables {x : ℝ} (n : ℕ)

-- Define the known property of the problem
noncomputable def binomial_ratio_condition (n : ℕ) : Prop :=
  (∑ i in finset.range(n + 1), ((-2) ^ i) * (nat.choose n i) : ℝ)

-- The first proof statement
theorem problem_part1 (h: (nat.choose (n - 3) 3) / (nat.choose (n - 2) 2) = 8 / 3)
  : n = 10 :=
sorry

-- The second proof statement
theorem problem_part2 (h : n = 10) 
  : (-2)^2 * (nat.choose 10 2) = 180 :=
sorry

-- The third proof statement
theorem problem_part3 
  : (∑ i in finset.range(11), ((-2) ^ i) * (nat.choose 10 i)) = 1 :=
sorry

end problem_part1_problem_part2_problem_part3_l154_154200


namespace steve_pie_difference_l154_154553

-- Definitions of conditions
def apple_pie_days : Nat := 3
def cherry_pie_days : Nat := 2
def pies_per_day : Nat := 12

-- Theorem statement
theorem steve_pie_difference : 
  (apple_pie_days * pies_per_day) - (cherry_pie_days * pies_per_day) = 12 := 
by
  sorry

end steve_pie_difference_l154_154553


namespace distance_to_school_is_correct_l154_154423

-- Define the necessary constants, variables, and conditions
def distance_to_market : ℝ := 2
def total_weekly_mileage : ℝ := 44
def school_trip_miles (x : ℝ) : ℝ := 16 * x
def market_trip_miles : ℝ := 2 * distance_to_market
def total_trip_miles (x : ℝ) : ℝ := school_trip_miles x + market_trip_miles

-- Prove that the distance from Philip's house to the children's school is 2.5 miles
theorem distance_to_school_is_correct (x : ℝ) (h : total_trip_miles x = total_weekly_mileage) :
  x = 2.5 :=
by
  -- Insert necessary proof steps starting with the provided hypothesis
  sorry

end distance_to_school_is_correct_l154_154423


namespace trigonometric_identity_l154_154334

-- Define the problem conditions and formulas
variables (α : Real) (h : Real.cos (Real.pi / 6 + α) = Real.sqrt 3 / 3)

-- State the theorem
theorem trigonometric_identity : Real.cos (5 * Real.pi / 6 - α) = - (Real.sqrt 3 / 3) :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_identity_l154_154334


namespace team_matches_per_season_l154_154692

theorem team_matches_per_season (teams_count total_games : ℕ) (h1 : teams_count = 50) (h2 : total_games = 4900) : 
  ∃ n : ℕ, n * (teams_count - 1) * teams_count / 2 = total_games ∧ n = 2 :=
by
  sorry

end team_matches_per_season_l154_154692


namespace probability_green_ball_l154_154319

theorem probability_green_ball :
  let p_container_A := 1 / 3
  let p_container_B := 1 / 3
  let p_container_c := 1 / 3
  let p_green_A := 5 / 10
  let p_green_B := 3 / 10
  let p_green_C := 4 / 10
  let total_probability := p_container_A * p_green_A + p_container_B * p_green_B + p_container_C * p_green_C
  total_probability = 2 / 5 :=
by
  sorry

end probability_green_ball_l154_154319


namespace smallest_composite_no_prime_factors_less_than_15_l154_154638

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l154_154638


namespace jacob_age_proof_l154_154975

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end jacob_age_proof_l154_154975


namespace tangent_line_at_x_neg1_l154_154560

-- Definition of the curve.
def curve (x : ℝ) : ℝ := 2*x - x^3

-- Definition of the point of tangency.
def point_of_tangency_x : ℝ := -1

-- Definition of the point of tangency.
def point_of_tangency_y : ℝ := curve point_of_tangency_x

-- Definition of the derivative of the curve.
def derivative (x : ℝ) : ℝ := -3*x^2 + 2

-- Slope of the tangent at the point of tangency.
def slope_at_tangency : ℝ := derivative point_of_tangency_x

-- Equation of the tangent line function.
def tangent_line (x y : ℝ) := x + y + 2 = 0

theorem tangent_line_at_x_neg1 :
  tangent_line point_of_tangency_x point_of_tangency_y :=
by
  -- Here we will perform the proof, which is omitted for the purposes of this task.
  sorry

end tangent_line_at_x_neg1_l154_154560


namespace find_a_even_function_l154_154883

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_domain : ∀ x, 2 * a + 1 ≤ x ∧ x ≤ a + 5) :
  a = -2 :=
sorry

end find_a_even_function_l154_154883


namespace smallest_integer_solution_l154_154585

theorem smallest_integer_solution (n : ℤ) (h : n^3 - 12 * n^2 + 44 * n - 48 ≤ 0) : n = 2 :=
sorry

end smallest_integer_solution_l154_154585


namespace garden_length_l154_154074

def PerimeterLength (P : ℕ) (length : ℕ) (breadth : ℕ) : Prop :=
  P = 2 * (length + breadth)

theorem garden_length
  (P : ℕ)
  (breadth : ℕ)
  (h1 : P = 480)
  (h2 : breadth = 100):
  ∃ length : ℕ, PerimeterLength P length breadth ∧ length = 140 :=
by
  use 140
  sorry

end garden_length_l154_154074


namespace divisors_of_n_squared_l154_154212

-- Definition for a number having exactly 4 divisors
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^3

-- Theorem statement
theorem divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) : 
  Nat.divisors_count (n^2) = 7 :=
by
  sorry

end divisors_of_n_squared_l154_154212


namespace circle_area_difference_l154_154358

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (A1 A2 diff : ℝ) 
  (hr1 : r1 = 30)
  (hd2 : 2 * r2 = 30)
  (hA1 : A1 = π * r1^2)
  (hA2 : A2 = π * r2^2)
  (hdiff : diff = A1 - A2) :
  diff = 675 * π :=
by 
  sorry

end circle_area_difference_l154_154358


namespace find_p_of_probability_l154_154159

-- Define the conditions and the problem statement
theorem find_p_of_probability
  (A_red_prob : ℚ := 1/3) -- probability of drawing a red ball from bag A
  (A_to_B_ratio : ℚ := 1/2) -- ratio of number of balls in bag A to bag B
  (combined_red_prob : ℚ := 2/5) -- total probability of drawing a red ball after combining balls
  : p = 13 / 30 := by
  sorry

end find_p_of_probability_l154_154159


namespace number_of_feasible_networks_10_l154_154620

-- Definitions based on conditions
def feasible_networks (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 ^ (n - 1)

-- The proof problem statement
theorem number_of_feasible_networks_10 : feasible_networks 10 = 512 := by
  -- proof goes here
  sorry

end number_of_feasible_networks_10_l154_154620


namespace arithmetic_sequence_solution_l154_154492

theorem arithmetic_sequence_solution (x : ℝ) (h : 2 * (x + 1) = 2 * x + (x + 2)) : x = 0 :=
by {
  -- To avoid actual proof steps, we add sorry.
  sorry 
}

end arithmetic_sequence_solution_l154_154492


namespace number_of_birds_flew_up_correct_l154_154761

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end number_of_birds_flew_up_correct_l154_154761


namespace isosceles_triangle_perimeter_l154_154059

theorem isosceles_triangle_perimeter (a b : ℝ)
  (h1 : b = 7)
  (h2 : a^2 - 8 * a + 15 = 0)
  (h3 : a * 2 > b)
  : 2 * a + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l154_154059


namespace largest_difference_l154_154529

def A := 3 * 1005^1006
def B := 1005^1006
def C := 1004 * 1005^1005
def D := 3 * 1005^1005
def E := 1005^1005
def F := 1005^1004

theorem largest_difference : 
  A - B > B - C ∧ 
  A - B > C - D ∧ 
  A - B > D - E ∧ 
  A - B > E - F :=
by
  sorry

end largest_difference_l154_154529


namespace monkey_reaches_top_in_19_minutes_l154_154444

theorem monkey_reaches_top_in_19_minutes (pole_height : ℕ) (ascend_first_min : ℕ) (slip_every_alternate_min : ℕ) 
    (total_minutes : ℕ) (net_gain_two_min : ℕ) : 
    pole_height = 10 ∧ ascend_first_min = 2 ∧ slip_every_alternate_min = 1 ∧ net_gain_two_min = 1 ∧ total_minutes = 19 →
    (net_gain_two_min * (total_minutes - 1) / 2 + ascend_first_min = pole_height) := 
by
    intros
    sorry

end monkey_reaches_top_in_19_minutes_l154_154444


namespace remainder_problem_l154_154483

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 2) (h2 : n = 197) : 197 % 16 = 5 := by
  sorry

end remainder_problem_l154_154483


namespace polynomial_simplification_simplify_expression_evaluate_expression_l154_154136

-- Prove that the correct simplification of 6mn - 2m - 3(m + 2mn) results in -5m.
theorem polynomial_simplification (m n : ℤ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m :=
by {
  sorry
}

-- Prove that simplifying a^2b^3 - 1/2(4ab + 6a^2b^3 - 1) + 2(ab - a^2b^3) results in -4a^2b^3 + 1/2.
theorem simplify_expression (a b : ℝ) :
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -4 * a^2 * b^3 + 1/2 :=
by {
  sorry
}

-- Prove that evaluating the expression -4a^2b^3 + 1/2 at a = 1/2 and b = 3 results in -26.5
theorem evaluate_expression :
  -4 * (1/2) ^ 2 * 3 ^ 3 + 1/2 = -26.5 :=
by {
  sorry
}

end polynomial_simplification_simplify_expression_evaluate_expression_l154_154136


namespace sum_of_eight_numbers_l154_154217

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l154_154217


namespace max_area_inscribed_triangle_l154_154494

/-- Let ΔABC be an inscribed triangle in the ellipse given by the equation
    (x^2 / 9) + (y^2 / 4) = 1, where the line segment AB passes through the 
    point (1, 0). Prove that the maximum area of ΔABC is (16 * sqrt 2) / 3. --/
theorem max_area_inscribed_triangle
  (A B C : ℝ × ℝ) 
  (hA : (A.1 ^ 2) / 9 + (A.2 ^ 2) / 4 = 1)
  (hB : (B.1 ^ 2) / 9 + (B.2 ^ 2) / 4 = 1)
  (hC : (C.1 ^ 2) / 9 + (C.2 ^ 2) / 4 = 1)
  (hAB : ∃ n : ℝ, ∀ x y : ℝ, (x, y) ∈ [A, B] → x = n * y + 1)
  : ∃ S : ℝ, S = ((16 : ℝ) * Real.sqrt 2) / 3 :=
sorry

end max_area_inscribed_triangle_l154_154494


namespace speed_equation_l154_154543

theorem speed_equation
  (dA dB : ℝ)
  (sB : ℝ)
  (sA : ℝ)
  (time_difference : ℝ)
  (h1 : dA = 800)
  (h2 : dB = 400)
  (h3 : sA = 1.2 * sB)
  (h4 : time_difference = 4) :
  (dA / sA - dB / sB = time_difference) :=
by
  sorry

end speed_equation_l154_154543


namespace ratio_of_inscribed_circle_segments_l154_154296

/-- A circle is inscribed in a triangle with side lengths 9, 12, and 15.
Let the segments of the side of length 9, made by a point of tangency, be r and s, with r < s.
Prove that the ratio r:s is 1:2. -/
theorem ratio_of_inscribed_circle_segments (r s : ℕ) (h : r < s) 
  (triangle_sides : r + s = 9) (point_of_tangency_15 : s + (12 - r) = 9) : 
  r / s = 1 / 2 := 
sorry

end ratio_of_inscribed_circle_segments_l154_154296


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154418

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154418


namespace festival_second_day_attendance_l154_154000

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end festival_second_day_attendance_l154_154000


namespace eval_expression_l154_154629

-- Definitions based on the conditions and problem statement
def x (b : ℕ) : ℕ := b + 9

-- The theorem to prove
theorem eval_expression (b : ℕ) : x b - b + 5 = 14 := by
    sorry

end eval_expression_l154_154629


namespace find_fraction_l154_154532

variable (x y z : ℂ) -- All complex numbers
variable (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) -- Non-zero conditions
variable (h2 : x + y + z = 10) -- Sum condition
variable (h3 : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) -- Given equation condition

theorem find_fraction 
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
    (h2 : x + y + z = 10)
    (h3 : 2 * ((x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2) = x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := 
sorry -- Proof yet to be completed

end find_fraction_l154_154532


namespace purchasing_plans_count_l154_154618

theorem purchasing_plans_count :
  ∃ n : ℕ, n = 2 ∧ (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = 35) :=
sorry

end purchasing_plans_count_l154_154618


namespace one_prime_p_10_14_l154_154841

theorem one_prime_p_10_14 :
  ∃! (p : ℕ), Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end one_prime_p_10_14_l154_154841


namespace math_equivalent_proof_l154_154140

-- Define the probabilities given the conditions
def P_A1 := 3 / 4
def P_A2 := 2 / 3
def P_A3 := 1 / 2
def P_B1 := 3 / 5
def P_B2 := 2 / 5

-- Define events
def P_C : ℝ := (P_A1 * P_B1 * (1 - P_A2)) + (P_A1 * P_B1 * P_A2 * P_B2 * (1 - P_A3))

-- Probability distribution of X
def P_X_0 : ℝ := (1 - P_A1) + P_C
def P_X_600 : ℝ := P_A1 * (1 - P_B1)
def P_X_1500 : ℝ := P_A1 * P_B1 * P_A2 * (1 - P_B2)
def P_X_3000 : ℝ := P_A1 * P_B1 * P_A2 * P_B2 * P_A3

-- Expected value of X
def E_X : ℝ := 600 * P_X_600 + 1500 * P_X_1500 + 3000 * P_X_3000

-- Statement to prove P(C) and expected value E(X)
theorem math_equivalent_proof :
  P_C = 21 / 100 ∧ 
  P_X_0 = 23 / 50 ∧
  P_X_600 = 3 / 10 ∧
  P_X_1500 = 9 / 50 ∧
  P_X_3000 = 3 / 50 ∧ 
  E_X = 630 := 
by 
  sorry

end math_equivalent_proof_l154_154140


namespace eq_solution_l154_154273

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end eq_solution_l154_154273


namespace ratio_of_area_of_shaded_square_l154_154455

theorem ratio_of_area_of_shaded_square 
  (large_square : Type) 
  (smaller_squares : Finset large_square) 
  (area_large_square : ℝ) 
  (area_smaller_square : ℝ) 
  (h_division : smaller_squares.card = 25)
  (h_equal_area : ∀ s ∈ smaller_squares, area_smaller_square = (area_large_square / 25))
  (shaded_square : Finset large_square)
  (h_shaded_sub : shaded_square ⊆ smaller_squares)
  (h_shaded_card : shaded_square.card = 5) :
  (5 * area_smaller_square) / area_large_square = 1 / 5 := 
by
  sorry

end ratio_of_area_of_shaded_square_l154_154455


namespace triangle_angle_sum_depends_on_parallel_postulate_l154_154103

-- Definitions of conditions
def triangle_angle_sum_theorem (A B C : ℝ) : Prop :=
  A + B + C = 180

def parallel_postulate : Prop :=
  ∀ (l : ℝ) (P : ℝ), ∃! (m : ℝ), m ≠ l ∧ ∀ (Q : ℝ), Q ≠ P → (Q = l ∧ Q = m)

-- Theorem statement: proving the dependence of the triangle_angle_sum_theorem on the parallel_postulate
theorem triangle_angle_sum_depends_on_parallel_postulate: 
  ∀ (A B C : ℝ), parallel_postulate → triangle_angle_sum_theorem A B C :=
sorry

end triangle_angle_sum_depends_on_parallel_postulate_l154_154103


namespace final_price_hat_final_price_tie_l154_154916

theorem final_price_hat (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (h_initial : initial_price = 20) 
    (h_first : first_discount = 0.25) 
    (h_second : second_discount = 0.20) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 12 := 
by 
  rw [h_initial, h_first, h_second]
  norm_num

theorem final_price_tie (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (t_initial : initial_price = 15) 
    (t_first : first_discount = 0.10) 
    (t_second : second_discount = 0.30) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 9.45 := 
by 
  rw [t_initial, t_first, t_second]
  norm_num

end final_price_hat_final_price_tie_l154_154916


namespace plane_through_points_and_perpendicular_l154_154797

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane_eq (A B C D : ℝ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

def vector_sub (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def is_perpendicular (normal1 normal2 : Point3D) : Prop :=
  normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z = 0

theorem plane_through_points_and_perpendicular
  (P1 P2 : Point3D)
  (A B C D : ℝ)
  (n_perp : Point3D)
  (normal1_eq : n_perp = ⟨2, -1, 4⟩)
  (eqn_given : plane_eq 2 (-1) 4 7 P1)
  (vec := vector_sub P1 P2)
  (n := cross_product vec n_perp)
  (eqn : plane_eq 11 (-10) (-9) (-33) P1) :
  (plane_eq 11 (-10) (-9) (-33) P2 ∧ is_perpendicular n n_perp) :=
sorry

end plane_through_points_and_perpendicular_l154_154797


namespace collinear_example_l154_154355

structure Vector2D where
  x : ℝ
  y : ℝ

def collinear (u v : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.x = k * u.x ∧ v.y = k * u.y

def a : Vector2D := ⟨1, 2⟩
def b : Vector2D := ⟨2, 4⟩

theorem collinear_example :
  collinear a b :=
by
  sorry

end collinear_example_l154_154355


namespace first_player_win_boards_l154_154277

-- Define what it means for a player to guarantee a win
def first_player_guarantees_win (n m : ℕ) : Prop :=
  ¬(n % 2 = 1 ∧ m % 2 = 1)

-- The main theorem that matches the math proof problem
theorem first_player_win_boards : (first_player_guarantees_win 6 7) ∧
                                  (first_player_guarantees_win 6 8) ∧
                                  (first_player_guarantees_win 7 8) ∧
                                  (first_player_guarantees_win 8 8) ∧
                                  ¬(first_player_guarantees_win 7 7) := 
by 
sorry

end first_player_win_boards_l154_154277


namespace find_sum_of_natural_numbers_l154_154795

theorem find_sum_of_natural_numbers :
  ∃ (square triangle : ℕ), square^2 + 12 = triangle^2 ∧ square + triangle = 6 :=
by
  sorry

end find_sum_of_natural_numbers_l154_154795


namespace equal_student_distribution_l154_154739

theorem equal_student_distribution
  (students_bus1_initial : ℕ)
  (students_bus2_initial : ℕ)
  (students_to_move : ℕ)
  (students_bus1_final : ℕ)
  (students_bus2_final : ℕ)
  (total_students : ℕ) :
  students_bus1_initial = 57 →
  students_bus2_initial = 31 →
  total_students = students_bus1_initial + students_bus2_initial →
  students_to_move = 13 →
  students_bus1_final = students_bus1_initial - students_to_move →
  students_bus2_final = students_bus2_initial + students_to_move →
  students_bus1_final = 44 ∧ students_bus2_final = 44 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end equal_student_distribution_l154_154739


namespace equal_charge_at_250_l154_154307

/-- Define the monthly fee for Plan A --/
def planA_fee (x : ℕ) : ℝ :=
  0.4 * x + 50

/-- Define the monthly fee for Plan B --/
def planB_fee (x : ℕ) : ℝ :=
  0.6 * x

/-- Prove that the charges for Plan A and Plan B are equal when the call duration is 250 minutes --/
theorem equal_charge_at_250 : planA_fee 250 = planB_fee 250 :=
by
  sorry

end equal_charge_at_250_l154_154307


namespace inequality_proof_l154_154948

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end inequality_proof_l154_154948


namespace min_sum_dimensions_l154_154558

theorem min_sum_dimensions (a b c : ℕ) (h : a * b * c = 2310) : a + b + c ≥ 52 :=
sorry

end min_sum_dimensions_l154_154558


namespace sample_size_l154_154082

theorem sample_size (total_employees : ℕ) (male_employees : ℕ) (sampled_males : ℕ) (sample_size : ℕ) 
  (h1 : total_employees = 120) (h2 : male_employees = 80) (h3 : sampled_males = 24) : 
  sample_size = 36 :=
by
  sorry

end sample_size_l154_154082


namespace female_officers_on_duty_percentage_l154_154539

   def percentage_of_females_on_duty (total_on_duty : ℕ) (female_on_duty : ℕ) (total_females : ℕ) : ℕ :=
   (female_on_duty * 100) / total_females
  
   theorem female_officers_on_duty_percentage
     (total_on_duty : ℕ) (h1 : total_on_duty = 180)
     (female_on_duty : ℕ) (h2 : female_on_duty = total_on_duty / 2)
     (total_females : ℕ) (h3 : total_females = 500) :
     percentage_of_females_on_duty total_on_duty female_on_duty total_females = 18 :=
   by
     rw [h1, h2, h3]
     sorry
   
end female_officers_on_duty_percentage_l154_154539


namespace R_depends_on_d_and_n_l154_154259

def arith_seq_sum (a d n : ℕ) (S1 S2 S3 : ℕ) : Prop := 
  (S1 = n * (a + (n - 1) * d / 2)) ∧ 
  (S2 = n * (2 * a + (2 * n - 1) * d)) ∧ 
  (S3 = 3 * n * (a + (3 * n - 1) * d / 2))

theorem R_depends_on_d_and_n (a d n S1 S2 S3 : ℕ) 
  (hS1 : S1 = n * (a + (n - 1) * d / 2))
  (hS2 : S2 = n * (2 * a + (2 * n - 1) * d))
  (hS3 : S3 = 3 * n * (a + (3 * n - 1) * d / 2)) 
  : S3 - S2 - S1 = 2 * n^2 * d  :=
by
  sorry

end R_depends_on_d_and_n_l154_154259


namespace find_k_l154_154517

-- Identifying conditions from the problem
def point (x : ℝ) : ℝ × ℝ := (x, x^3)  -- A point on the curve y = x^3
def tangent_slope (x : ℝ) : ℝ := 3 * x^2  -- The slope of the tangent to the curve y = x^3 at point (x, x^3)
def tangent_line (x k : ℝ) : ℝ := k * x + 2  -- The given tangent line equation

-- Question as a proof problem
theorem find_k (x : ℝ) (k : ℝ) (h : tangent_line x k = x^3) : k = 3 :=
by
  sorry

end find_k_l154_154517


namespace ratio_of_perimeter_to_length_XY_l154_154696

noncomputable def XY : ℝ := 17
noncomputable def XZ : ℝ := 8
noncomputable def YZ : ℝ := 15
noncomputable def ZD : ℝ := 240 / 17

-- Defining the perimeter P
noncomputable def P : ℝ := 17 + 2 * (240 / 17)

-- Finally, the statement with the ratio in the desired form
theorem ratio_of_perimeter_to_length_XY : 
  (P / XY) = (654 / 289) :=
by
  sorry

end ratio_of_perimeter_to_length_XY_l154_154696


namespace intersection_of_M_and_N_l154_154678

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end intersection_of_M_and_N_l154_154678


namespace man_speed_in_still_water_l154_154773

theorem man_speed_in_still_water 
  (V_u : ℕ) (V_d : ℕ) 
  (hu : V_u = 34) 
  (hd : V_d = 48) : 
  V_s = (V_u + V_d) / 2 :=
by
  sorry

end man_speed_in_still_water_l154_154773


namespace range_of_m_l154_154363

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0) → (2*x + 1 > 3) → (x > 1)) → (m ≤ 1) :=
by
  intros h
  sorry

end range_of_m_l154_154363


namespace subset_B_of_A_l154_154345

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem subset_B_of_A : B ⊆ A :=
by
  sorry

end subset_B_of_A_l154_154345


namespace find_specific_linear_function_l154_154541

-- Define the linear function with given conditions
def linear_function (k b : ℝ) (x : ℝ) := k * x + b

-- Define the condition that the point lies on the line
def passes_through (k b : ℝ) (x y : ℝ) := y = linear_function k b x

-- Define the condition that slope is negative
def slope_negative (k : ℝ) := k < 0

-- The specific function we want to prove
def specific_linear_function (x : ℝ) := -x + 1

-- The theorem to prove
theorem find_specific_linear_function : 
  ∃ (k b : ℝ), slope_negative k ∧ passes_through k b 0 1 ∧ 
  ∀ x, linear_function k b x = specific_linear_function x :=
by
  sorry

end find_specific_linear_function_l154_154541


namespace max_alpha_beta_square_l154_154843

theorem max_alpha_beta_square (k : ℝ) (α β : ℝ)
  (h1 : α^2 - (k - 2) * α + (k^2 + 3 * k + 5) = 0)
  (h2 : β^2 - (k - 2) * β + (k^2 + 3 * k + 5) = 0)
  (h3 : α ≠ β) :
  (α^2 + β^2) ≤ 18 :=
sorry

end max_alpha_beta_square_l154_154843


namespace num_black_squares_in_37th_row_l154_154765

-- Define the total number of squares in the n-th row
def total_squares_in_row (n : ℕ) : ℕ := 2 * n - 1

-- Define the number of black squares in the n-th row
def black_squares_in_row (n : ℕ) : ℕ := (total_squares_in_row n - 1) / 2

theorem num_black_squares_in_37th_row : black_squares_in_row 37 = 36 :=
by
  sorry

end num_black_squares_in_37th_row_l154_154765


namespace relationship_between_m_and_n_l154_154055

theorem relationship_between_m_and_n
  (a : ℝ) (b : ℝ) (ha : a > 2) (hb : b ≠ 0)
  (m : ℝ := a + 1 / (a - 2))
  (n : ℝ := 2^(2 - b^2)) :
  m > n :=
sorry

end relationship_between_m_and_n_l154_154055


namespace interval_length_condition_l154_154732

theorem interval_length_condition (c : ℝ) (x : ℝ) (H1 : 3 ≤ 5 * x - 4) (H2 : 5 * x - 4 ≤ c) 
                                  (H3 : (c + 4) / 5 - 7 / 5 = 15) : c - 3 = 75 := 
sorry

end interval_length_condition_l154_154732


namespace domain_ln_l154_154478

def domain_of_ln (x : ℝ) : Prop := x^2 - x > 0

theorem domain_ln (x : ℝ) :
  domain_of_ln x ↔ (x < 0 ∨ x > 1) :=
by sorry

end domain_ln_l154_154478


namespace evaluate_expression_l154_154746

noncomputable def a : ℕ := 3^2 + 5^2 + 7^2
noncomputable def b : ℕ := 2^2 + 4^2 + 6^2

theorem evaluate_expression : (a / b : ℚ) - (b / a : ℚ) = 3753 / 4656 :=
by
  sorry

end evaluate_expression_l154_154746


namespace line_equation_with_slope_angle_135_and_y_intercept_neg1_l154_154632

theorem line_equation_with_slope_angle_135_and_y_intercept_neg1 :
  ∃ k b : ℝ, k = -1 ∧ b = -1 ∧ ∀ x y : ℝ, y = k * x + b ↔ y = -x - 1 :=
by
  sorry

end line_equation_with_slope_angle_135_and_y_intercept_neg1_l154_154632


namespace sum_of_common_ratios_is_five_l154_154991

theorem sum_of_common_ratios_is_five {k p r : ℝ} 
  (h1 : p ≠ r)                       -- different common ratios
  (h2 : k ≠ 0)                       -- non-zero k
  (a2 : ℝ := k * p)                  -- term a2
  (a3 : ℝ := k * p^2)                -- term a3
  (b2 : ℝ := k * r)                  -- term b2
  (b3 : ℝ := k * r^2)                -- term b3
  (h3 : a3 - b3 = 5 * (a2 - b2))     -- given condition
  : p + r = 5 := 
by
  sorry

end sum_of_common_ratios_is_five_l154_154991


namespace average_eq_y_value_l154_154557

theorem average_eq_y_value :
  (y : ℤ) → (h : (15 + 25 + y) / 3 = 20) → y = 20 :=
by
  intro y h
  sorry

end average_eq_y_value_l154_154557


namespace find_min_value_l154_154180

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end find_min_value_l154_154180


namespace simplify_expression_l154_154880

-- Define the problem context
variables {x y : ℝ} {i : ℂ}

-- The mathematical simplification problem
theorem simplify_expression :
  (x ^ 2 + i * y) ^ 3 * (x ^ 2 - i * y) ^ 3 = x ^ 12 - 9 * x ^ 8 * y ^ 2 - 9 * x ^ 4 * y ^ 4 - y ^ 6 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_expression_l154_154880


namespace find_x_l154_154254

theorem find_x (x : ℝ) (a b : ℝ) (h₀ : a * b = 4 * a - 2 * b)
  (h₁ : 3 * (6 * x) = -2) :
  x = 17 / 2 :=
by
  sorry

end find_x_l154_154254


namespace symmetric_line_eq_l154_154676

-- Given lines
def line₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def mirror_line (x y : ℝ) : Prop := y = -x

-- Definition of symmetry about the line y = -x
def symmetric_about (l₁ l₂: ℝ → ℝ → Prop) : Prop :=
∀ x y, l₁ x y ↔ l₂ y (-x)

-- Definition of line l₂ that is symmetric to line₁ about the mirror_line
def line₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem stating that the symmetric line to line₁ about y = -x is line₂
theorem symmetric_line_eq :
  symmetric_about line₁ line₂ :=
sorry

end symmetric_line_eq_l154_154676


namespace percentage_liked_B_l154_154606

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l154_154606


namespace convert_ternary_to_octal_2101211_l154_154475

def ternaryToOctal (n : List ℕ) : ℕ := 
  sorry

theorem convert_ternary_to_octal_2101211 :
  ternaryToOctal [2, 1, 0, 1, 2, 1, 1] = 444
  := sorry

end convert_ternary_to_octal_2101211_l154_154475


namespace sequence_is_constant_l154_154197

theorem sequence_is_constant
  (a : ℕ+ → ℝ)
  (S : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, S n + S (n + 1) = a (n + 1))
  : ∀ n : ℕ+, a n = 0 :=
by
  sorry

end sequence_is_constant_l154_154197


namespace ratio_of_inscribed_circle_segments_l154_154297

/-- A circle is inscribed in a triangle with side lengths 9, 12, and 15.
Let the segments of the side of length 9, made by a point of tangency, be r and s, with r < s.
Prove that the ratio r:s is 1:2. -/
theorem ratio_of_inscribed_circle_segments (r s : ℕ) (h : r < s) 
  (triangle_sides : r + s = 9) (point_of_tangency_15 : s + (12 - r) = 9) : 
  r / s = 1 / 2 := 
sorry

end ratio_of_inscribed_circle_segments_l154_154297


namespace income_is_108000_l154_154755

theorem income_is_108000 (S I : ℝ) (h1 : S / I = 5 / 9) (h2 : 48000 = I - S) : I = 108000 :=
by
  sorry

end income_is_108000_l154_154755


namespace opposite_of_lime_is_black_l154_154253

-- Given colors of the six faces
inductive Color
| Purple | Cyan | Magenta | Silver | Lime | Black

-- Hinged squares forming a cube
structure Cube :=
(top : Color) (bottom : Color) (front : Color) (back : Color) (left : Color) (right : Color)

-- Condition: Magenta is on the top
def magenta_top (c : Cube) : Prop := c.top = Color.Magenta

-- Problem statement: Prove the color opposite to Lime is Black
theorem opposite_of_lime_is_black (c : Cube) (HM : magenta_top c) (HL : c.front = Color.Lime)
    (HBackFace : c.back = Color.Black) : c.back = Color.Black := 
sorry

end opposite_of_lime_is_black_l154_154253


namespace square_area_dimensions_l154_154425

theorem square_area_dimensions (x : ℝ) (n : ℝ) : 
  (x^2 + (x + 12)^2 = 2120) → 
  (n = x + 12) → 
  (x = 26) → 
  (n = 38) := 
by
  sorry

end square_area_dimensions_l154_154425


namespace number_of_cookies_on_the_fifth_plate_l154_154120

theorem number_of_cookies_on_the_fifth_plate
  (c : ℕ → ℕ)
  (h1 : c 1 = 5)
  (h2 : c 2 = 7)
  (h3 : c 3 = 10)
  (h4 : c 4 = 14)
  (h6 : c 6 = 25)
  (h_diff : ∀ n, c (n + 1) - c n = c (n + 2) - c (n + 1) + 1) :
  c 5 = 19 :=
by
  sorry

end number_of_cookies_on_the_fifth_plate_l154_154120


namespace probability_odd_even_draw_l154_154292

theorem probability_odd_even_draw :
  let balls := {1, 2, 3, 4, 5}
  let odd_balls := {1, 3, 5}
  let even_balls := {2, 4}
  let total_balls := 5
  let first_draw_odd := (odd_balls.card : ℚ) / total_balls
  let remaining_after_odd := total_balls - 1
  let second_draw_even := (even_balls.card : ℚ) / remaining_after_odd
  (first_draw_odd * second_draw_even = 3 / 10) :=
by
  intros
  rw [Set.card_image_of_injective _ (Set.pairwise_injective_of_fintype _), Set.to_finite.card,
      Set.card_image_of_injective _ (Set.pairwise_injective_of_fintype _), Set.to_finite.card]
  simp [first_draw_odd, second_draw_even, div_eq_mul_inv]
  norm_num
  sorry

end probability_odd_even_draw_l154_154292


namespace find_line_AB_l154_154680

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 16

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Proof statement: Line AB is the correct line through the intersection points of the two circles
theorem find_line_AB :
  (∃ x y, circle1 x y ∧ circle2 x y) →
  (∀ x y, (circle1 x y ∧ circle2 x y) ↔ lineAB x y) :=
by
  sorry

end find_line_AB_l154_154680


namespace polygon_sides_l154_154452

theorem polygon_sides (n : ℕ) (h : (n-3) * 180 < 2008 ∧ 2008 < (n-1) * 180) : 
  n = 14 :=
sorry

end polygon_sides_l154_154452


namespace opposite_of_neg_one_third_l154_154577

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end opposite_of_neg_one_third_l154_154577


namespace sum_possible_x_coordinates_l154_154004

-- Define the vertices of the parallelogram
def A := (1, 2)
def B := (3, 8)
def C := (4, 1)

-- Definition of what it means to be a fourth vertex that forms a parallelogram
def is_fourth_vertex (D : ℤ × ℤ) : Prop :=
  (D = (6, 7)) ∨ (D = (2, -5)) ∨ (D = (0, 9))

-- The sum of possible x-coordinates for the fourth vertex
def sum_x_coordinates : ℤ :=
  6 + 2 + 0

theorem sum_possible_x_coordinates :
  (∃ D, is_fourth_vertex D) → sum_x_coordinates = 8 :=
by
  -- Sorry is used to skip the detailed proof steps
  sorry

end sum_possible_x_coordinates_l154_154004


namespace taxi_fare_relationship_taxi_fare_relationship_simplified_l154_154225

variable (x : ℝ) (y : ℝ)

-- Conditions
def starting_fare : ℝ := 14
def additional_fare_per_km : ℝ := 2.4
def initial_distance : ℝ := 3
def total_distance (x : ℝ) := x
def total_fare (x : ℝ) (y : ℝ) := y
def distance_condition (x : ℝ) := x > 3

-- Theorem Statement
theorem taxi_fare_relationship (h : distance_condition x) :
  total_fare x y = additional_fare_per_km * (total_distance x - initial_distance) + starting_fare :=
by
  sorry

-- Simplified Theorem Statement
theorem taxi_fare_relationship_simplified (h : distance_condition x) :
  y = 2.4 * x + 6.8 :=
by
  sorry

end taxi_fare_relationship_taxi_fare_relationship_simplified_l154_154225


namespace divisibility_condition_l154_154936

theorem divisibility_condition (M C D U A q1 q2 q3 r1 r2 r3 : ℕ)
  (h1 : 10 = A * q1 + r1)
  (h2 : 10 * r1 = A * q2 + r2)
  (h3 : 10 * r2 = A * q3 + r3) :
  (U + D * r1 + C * r2 + M * r3) % A = 0 ↔ (1000 * M + 100 * C + 10 * D + U) % A = 0 :=
sorry

end divisibility_condition_l154_154936


namespace pyramid_height_l154_154146

def height_of_pyramid (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem pyramid_height (n : ℕ) : height_of_pyramid n = 2 * (n - 1) :=
by
  -- The proof would typically go here
  sorry

end pyramid_height_l154_154146


namespace total_spent_is_140_l154_154930

-- Define the original prices and discounts
def original_price_shoes : ℕ := 50
def original_price_dress : ℕ := 100
def discount_shoes : ℕ := 40
def discount_dress : ℕ := 20

-- Define the number of items purchased
def number_of_shoes : ℕ := 2
def number_of_dresses : ℕ := 1

-- Define the calculation of discounted prices
def discounted_price_shoes (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

def discounted_price_dress (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

-- Define the total cost calculation
def total_cost : ℕ :=
  discounted_price_shoes original_price_shoes discount_shoes number_of_shoes +
  discounted_price_dress original_price_dress discount_dress number_of_dresses

-- The theorem to prove
theorem total_spent_is_140 : total_cost = 140 := by
  sorry

end total_spent_is_140_l154_154930


namespace total_people_on_hike_l154_154387

-- Definitions of the conditions
def n_cars : ℕ := 3
def n_people_per_car : ℕ := 4
def n_taxis : ℕ := 6
def n_people_per_taxi : ℕ := 6
def n_vans : ℕ := 2
def n_people_per_van : ℕ := 5

-- Statement of the problem
theorem total_people_on_hike : 
  n_cars * n_people_per_car + n_taxis * n_people_per_taxi + n_vans * n_people_per_van = 58 :=
by sorry

end total_people_on_hike_l154_154387


namespace least_positive_integer_reducible_fraction_l154_154633

-- Define gcd function as used in the problem
def is_reducible_fraction (a b : ℕ) : Prop := Nat.gcd a b > 1

-- Define the conditions and the proof problem
theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ is_reducible_fraction (n - 27) (7 * n + 4) ∧
  ∀ m : ℕ, (0 < m → is_reducible_fraction (m - 27) (7 * m + 4) → n ≤ m) :=
sorry

end least_positive_integer_reducible_fraction_l154_154633


namespace constant_term_of_expansion_l154_154015

open BigOperators

noncomputable def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_of_expansion :
  ∑ r in Finset.range (6 + 1), binomialCoeff 6 r * (2^r * (x : ℚ)^r) / (x^3 : ℚ) = 160 :=
by
  sorry

end constant_term_of_expansion_l154_154015


namespace number_of_sides_of_polygon_l154_154688

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)
noncomputable def sum_known_angles : ℝ := 3780

theorem number_of_sides_of_polygon
  (n : ℕ)
  (h1 : sum_known_angles + missing_angle = sum_of_interior_angles n)
  (h2 : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a = 3 * c ∧ b = 3 * c ∧ a + b + c ≤ sum_known_angles) :
  n = 23 :=
sorry

end number_of_sides_of_polygon_l154_154688


namespace sum_first_2500_terms_eq_zero_l154_154459

theorem sum_first_2500_terms_eq_zero
  (b : ℕ → ℤ)
  (h1 : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2))
  (h2 : (Finset.range 1800).sum b = 2023)
  (h3 : (Finset.range 2023).sum b = 1800) :
  (Finset.range 2500).sum b = 0 :=
sorry

end sum_first_2500_terms_eq_zero_l154_154459


namespace neg_p_necessary_not_sufficient_for_neg_p_or_q_l154_154196

variables (p q : Prop)

theorem neg_p_necessary_not_sufficient_for_neg_p_or_q :
  (¬ p → ¬ (p ∨ q)) ∧ (¬ (p ∨ q) → ¬ p) :=
by {
  sorry
}

end neg_p_necessary_not_sufficient_for_neg_p_or_q_l154_154196


namespace algebra_inequality_l154_154950

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end algebra_inequality_l154_154950


namespace find_Z_l154_154061

open Complex

-- Definitions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_Z (Z : ℂ) (h1 : abs Z = 3) (h2 : is_pure_imaginary (Z + (3 * Complex.I))) : Z = 3 * Complex.I :=
by
  sorry

end find_Z_l154_154061


namespace estimated_prob_is_0_9_l154_154426

section GerminationProbability

-- Defining the experiment data
structure ExperimentData :=
  (totalSeeds : ℕ)
  (germinatedSeeds : ℕ)
  (germinationRate : ℝ)

def experiments : List ExperimentData := [
  ⟨100, 91, 0.91⟩, 
  ⟨400, 358, 0.895⟩, 
  ⟨800, 724, 0.905⟩,
  ⟨1400, 1264, 0.903⟩,
  ⟨3500, 3160, 0.903⟩,
  ⟨7000, 6400, 0.914⟩
]

-- Hypothesis based on the given problem's observation
def estimated_germination_probability (experiments : List ExperimentData) : ℝ :=
  /- Fictively calculating the stable germination rate here; however, logically we should use 
     some weighted average or similar statistical stability method. -/
  0.9  -- Rounded and concluded estimated value based on observation

theorem estimated_prob_is_0_9 :
  estimated_germination_probability experiments = 0.9 :=
  sorry

end GerminationProbability

end estimated_prob_is_0_9_l154_154426


namespace inv_3i_minus_2inv_i_eq_neg_inv_5i_l154_154806

-- Define the imaginary unit i such that i^2 = -1
def i : ℂ := Complex.I
axiom i_square : i^2 = -1

-- Proof statement
theorem inv_3i_minus_2inv_i_eq_neg_inv_5i : (3 * i - 2 * (1 / i))⁻¹ = -i / 5 :=
by
  -- Replace these steps with the corresponding actual proofs
  sorry

end inv_3i_minus_2inv_i_eq_neg_inv_5i_l154_154806


namespace percentage_problem_l154_154600

theorem percentage_problem
  (a b c : ℚ) :
  (8 = (2 / 100) * a) →
  (2 = (8 / 100) * b) →
  (c = b / a) →
  c = 1 / 16 :=
by
  sorry

end percentage_problem_l154_154600


namespace profit_percentage_is_correct_l154_154919

noncomputable def CP : ℝ := 47.50
noncomputable def SP : ℝ := 74.21875
noncomputable def MP : ℝ := SP / 0.8
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercentage : ℝ := (Profit / CP) * 100

theorem profit_percentage_is_correct : ProfitPercentage = 56.25 := by
  -- Proof steps to be filled in
  sorry

end profit_percentage_is_correct_l154_154919


namespace seven_distinct_numbers_with_reversed_digits_l154_154080

theorem seven_distinct_numbers_with_reversed_digits (x y : ℕ) :
  (∃ a b c d e f g : ℕ, 
  (10 * a + b + 18 = 10 * b + a) ∧ (10 * c + d + 18 = 10 * d + c) ∧ 
  (10 * e + f + 18 = 10 * f + e) ∧ (10 * g + y + 18 = 10 * y + g) ∧ 
  a ≠ c ∧ a ≠ e ∧ a ≠ g ∧ 
  c ≠ e ∧ c ≠ g ∧ 
  e ≠ g ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧
  (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧
  (1 ≤ e ∧ e <= 9) ∧ (1 ≤ f ∧ f <= 9) ∧
  (1 ≤ g ∧ g <= 9) ∧ (1 ≤ y ∧ y <= 9)) :=
sorry

end seven_distinct_numbers_with_reversed_digits_l154_154080


namespace Lyle_can_buy_for_his_friends_l154_154997

theorem Lyle_can_buy_for_his_friends
  (cost_sandwich : ℝ) (cost_juice : ℝ) (total_money : ℝ)
  (h1 : cost_sandwich = 0.30)
  (h2 : cost_juice = 0.20)
  (h3 : total_money = 2.50) :
  (total_money / (cost_sandwich + cost_juice)).toNat - 1 = 4 :=
by
  sorry

end Lyle_can_buy_for_his_friends_l154_154997


namespace positive_whole_numbers_with_cube_root_less_than_8_l154_154838

theorem positive_whole_numbers_with_cube_root_less_than_8 :
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by
  sorry

end positive_whole_numbers_with_cube_root_less_than_8_l154_154838


namespace opposite_neg_inv_three_l154_154570

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end opposite_neg_inv_three_l154_154570


namespace fractional_pizza_eaten_after_six_trips_l154_154129

def pizza_eaten : ℚ := (1/3) * (1 - (2/3)^6) / (1 - 2/3)

theorem fractional_pizza_eaten_after_six_trips : pizza_eaten = 665 / 729 :=
by
  -- proof will go here
  sorry

end fractional_pizza_eaten_after_six_trips_l154_154129


namespace maximum_value_abs_difference_l154_154598

theorem maximum_value_abs_difference (x y : ℝ) 
  (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : 
  |x - y + 1| ≤ 2 :=
sorry

end maximum_value_abs_difference_l154_154598


namespace friends_behind_Yuna_l154_154290

def total_friends : ℕ := 6
def friends_in_front_of_Yuna : ℕ := 2

theorem friends_behind_Yuna : total_friends - friends_in_front_of_Yuna = 4 :=
by
  -- Proof goes here
  sorry

end friends_behind_Yuna_l154_154290


namespace log3_of_9_to_3_l154_154044

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l154_154044


namespace math_team_count_l154_154119

open Nat

theorem math_team_count :
  let girls := 7
  let boys := 12
  let total_team := 16
  let count_ways (n k : ℕ) := choose n k
  (count_ways girls 3) * (count_ways boys 5) * (count_ways (girls - 3 + boys - 5) 8) = 456660 :=
by
  sorry

end math_team_count_l154_154119


namespace number_of_wheels_l154_154850

theorem number_of_wheels (V : ℕ) (W_2 : ℕ) (n : ℕ) 
  (hV : V = 16) 
  (h_eq : 2 * W_2 + 16 * n = 66) : 
  n = 4 := 
by 
  sorry

end number_of_wheels_l154_154850


namespace smallest_four_digit_mod_8_l154_154439

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l154_154439


namespace area_of_inscribed_square_in_ellipse_l154_154460

open Real

noncomputable def inscribed_square_area : ℝ := 32

theorem area_of_inscribed_square_in_ellipse :
  ∀ (x y : ℝ),
  (x^2 / 4 + y^2 / 8 = 1) →
  (x = t - t) ∧ (y = (t + t) / sqrt 2) ∧ 
  (t = sqrt 4) → inscribed_square_area = 32 :=
  sorry

end area_of_inscribed_square_in_ellipse_l154_154460


namespace stones_required_correct_l154_154772

/- 
Given:
- The hall measures 36 meters long and 15 meters broad.
- Each stone measures 6 decimeters by 5 decimeters.

We need to prove that the number of stones required to pave the hall is 1800.
-/
noncomputable def stones_required 
  (hall_length_m : ℕ) 
  (hall_breadth_m : ℕ) 
  (stone_length_dm : ℕ) 
  (stone_breadth_dm : ℕ) : ℕ :=
  (hall_length_m * 10) * (hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm)

theorem stones_required_correct : 
  stones_required 36 15 6 5 = 1800 :=
by 
  -- Placeholder for proof
  sorry

end stones_required_correct_l154_154772


namespace rita_remaining_money_l154_154874

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l154_154874


namespace Trisha_total_distance_l154_154538

theorem Trisha_total_distance :
  let d1 := 0.11  -- hotel to postcard shop
  let d2 := 0.11  -- postcard shop back to hotel
  let d3 := 1.52  -- hotel to T-shirt shop
  let d4 := 0.45  -- T-shirt shop to hat shop
  let d5 := 0.87  -- hat shop to purse shop
  let d6 := 2.32  -- purse shop back to hotel
  d1 + d2 + d3 + d4 + d5 + d6 = 5.38 :=
by
  sorry

end Trisha_total_distance_l154_154538


namespace spherical_to_rectangular_coordinates_l154_154789

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 10 ∧ θ = 3 * Real.pi / 4 ∧ φ = Real.pi / 6 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  :=
by
  intros ρ θ φ h
  rcases h with ⟨hρ, hθ, hφ⟩
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coordinates_l154_154789


namespace ernie_income_ratio_l154_154018

-- Define constants and properties based on the conditions
def previous_income := 6000
def jack_income := 2 * previous_income
def combined_income := 16800

-- Lean proof statement that the ratio of Ernie's current income to his previous income is 2/3
theorem ernie_income_ratio (current_income : ℕ) (h1 : current_income + jack_income = combined_income) :
    current_income / previous_income = 2 / 3 :=
sorry

end ernie_income_ratio_l154_154018


namespace simplify_expression_l154_154547

theorem simplify_expression (x : ℝ) : 
  (2 * x - 3 * (2 + x) + 4 * (2 - x) - 5 * (2 + 3 * x)) = -20 * x - 8 :=
by
  sorry

end simplify_expression_l154_154547


namespace product_and_sum_of_roots_l154_154941

theorem product_and_sum_of_roots :
  let a := 24
  let b := 60
  let c := -600
  (c / a = -25) ∧ (-b / a = -2.5) := 
by
  sorry

end product_and_sum_of_roots_l154_154941


namespace smallest_four_digit_mod_8_l154_154440

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l154_154440


namespace find_ab_l154_154963

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 2 →
  (3 * x - 2 < a + 1 ∧ 6 - 2 * x < b + 2)) →
  a = 3 ∧ b = 6 :=
by
  sorry

end find_ab_l154_154963


namespace problem_1_problem_2_l154_154808

noncomputable def f (x : ℝ) : ℝ := (1 / (9 * (Real.sin x)^2)) + (4 / (9 * (Real.cos x)^2))

theorem problem_1 (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : f x ≥ 1 := 
sorry

theorem problem_2 (x : ℝ) : x^2 + |x-2| + 1 ≥ 3 ↔ (x ≤ 0 ∨ x ≥ 1) :=
sorry

end problem_1_problem_2_l154_154808


namespace anne_carries_total_weight_l154_154921

-- Definitions for the conditions
def weight_female_cat : ℕ := 2
def weight_male_cat : ℕ := 2 * weight_female_cat

-- Problem statement
theorem anne_carries_total_weight : weight_female_cat + weight_male_cat = 6 :=
by
  sorry

end anne_carries_total_weight_l154_154921


namespace sector_area_l154_154726

theorem sector_area (θ r : ℝ) (hθ : θ = 2) (hr : r = 1) :
  (1 / 2) * r^2 * θ = 1 :=
by
  -- Conditions are instantiated
  rw [hθ, hr]
  -- Simplification is left to the proof
  sorry

end sector_area_l154_154726


namespace simplify_exponent_expression_l154_154548

theorem simplify_exponent_expression (n : ℕ) :
  (3^(n+4) - 3 * 3^n) / (3 * 3^(n+3)) = 26 / 9 := by
  sorry

end simplify_exponent_expression_l154_154548


namespace find_dimensions_l154_154729

theorem find_dimensions (x y : ℝ) 
  (h1 : 90 = (2 * x + y) * (2 * y))
  (h2 : x * y = 10) : x = 2 ∧ y = 5 :=
by
  sorry

end find_dimensions_l154_154729


namespace value_of_m_l154_154946

theorem value_of_m 
  (m : ℤ) 
  (h : ∀ x : ℤ, x^2 - 2 * (m + 1) * x + 16 = (x - 4)^2) : 
  m = 3 := 
sorry

end value_of_m_l154_154946


namespace samatha_routes_l154_154100

-- Definitions based on the given conditions
def blocks_from_house_to_southwest_corner := 4
def blocks_through_park := 1
def blocks_from_northeast_corner_to_school := 4
def blocks_from_school_to_library := 3

-- Number of ways to arrange movements
def number_of_routes_house_to_southwest : ℕ :=
  Nat.choose blocks_from_house_to_southwest_corner 1

def number_of_routes_through_park : ℕ := blocks_through_park

def number_of_routes_northeast_to_school : ℕ :=
  Nat.choose blocks_from_northeast_corner_to_school 1

def number_of_routes_school_to_library : ℕ :=
  Nat.choose blocks_from_school_to_library 1

-- Total number of different routes
def total_number_of_routes : ℕ :=
  number_of_routes_house_to_southwest *
  number_of_routes_through_park *
  number_of_routes_northeast_to_school *
  number_of_routes_school_to_library

theorem samatha_routes (n : ℕ) (h : n = 48) :
  total_number_of_routes = n :=
  by
    -- Proof is skipped
    sorry

end samatha_routes_l154_154100


namespace train_combined_distance_l154_154278

/-- Prove that the combined distance covered by three trains is 3480 km,
    given their respective speeds and travel times. -/
theorem train_combined_distance : 
  let speed_A := 150 -- Speed of Train A in km/h
  let time_A := 8     -- Time Train A travels in hours
  let speed_B := 180 -- Speed of Train B in km/h
  let time_B := 6     -- Time Train B travels in hours
  let speed_C := 120 -- Speed of Train C in km/h
  let time_C := 10    -- Time Train C travels in hours
  let distance_A := speed_A * time_A -- Distance covered by Train A
  let distance_B := speed_B * time_B -- Distance covered by Train B
  let distance_C := speed_C * time_C -- Distance covered by Train C
  let combined_distance := distance_A + distance_B + distance_C -- Combined distance covered by all trains
  combined_distance = 3480 :=
by
  sorry

end train_combined_distance_l154_154278


namespace tangent_line_eq_l154_154177

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x^2 + 1) : 
  (x = -1 ∧ y = 3) → (4 * x + y + 1 = 0) :=
by
  intros
  sorry

end tangent_line_eq_l154_154177


namespace integral_cos_from_0_to_pi_div_2_l154_154313

open Real

theorem integral_cos_from_0_to_pi_div_2 : (∫ x in 0..(pi/2), cos x) = 1 := 
by
  -- Proof goes here.
  sorry

end integral_cos_from_0_to_pi_div_2_l154_154313


namespace opposite_neg_inv_three_l154_154572

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end opposite_neg_inv_three_l154_154572


namespace percent_time_in_meetings_l154_154375

theorem percent_time_in_meetings
  (work_day_minutes : ℕ := 8 * 60)
  (first_meeting_minutes : ℕ := 30)
  (second_meeting_minutes : ℕ := 3 * 30) :
  (first_meeting_minutes + second_meeting_minutes) / work_day_minutes * 100 = 25 :=
by
  -- sorry to skip the actual proof
  sorry

end percent_time_in_meetings_l154_154375


namespace largest_among_numbers_l154_154791

theorem largest_among_numbers :
  ∀ (a b c d e : ℝ), 
  a = 0.997 ∧ b = 0.9799 ∧ c = 0.999 ∧ d = 0.9979 ∧ e = 0.979 →
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by intros a b c d e habcde
   rcases habcde with ⟨ha, hb, hc, hd, he⟩
   simp [ha, hb, hc, hd, he]
   sorry

end largest_among_numbers_l154_154791


namespace k_value_l154_154758

theorem k_value (m n k : ℤ) (h₁ : m + 2 * n + 5 = 0) (h₂ : (m + 2) + 2 * (n + k) + 5 = 0) : k = -1 :=
by sorry

end k_value_l154_154758


namespace only_linear_equation_with_two_variables_l154_154126

def is_linear_equation_with_two_variables (eqn : String) : Prop :=
  eqn = "4x-5y=5"

def equation_A := "4x-5y=5"
def equation_B := "xy-y=1"
def equation_C := "4x+5y"
def equation_D := "2/x+5/y=1/7"

theorem only_linear_equation_with_two_variables :
  is_linear_equation_with_two_variables equation_A ∧
  ¬ is_linear_equation_with_two_variables equation_B ∧
  ¬ is_linear_equation_with_two_variables equation_C ∧
  ¬ is_linear_equation_with_two_variables equation_D :=
by
  sorry

end only_linear_equation_with_two_variables_l154_154126


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154395

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154395


namespace find_length_of_shop_l154_154564

noncomputable def length_of_shop (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  (monthly_rent * 12) / annual_rent_per_sqft / width

theorem find_length_of_shop
  (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ)
  (h_monthly_rent : monthly_rent = 3600)
  (h_width : width = 20)
  (h_annual_rent_per_sqft : annual_rent_per_sqft = 120) 
  : length_of_shop monthly_rent width annual_rent_per_sqft = 18 := 
sorry

end find_length_of_shop_l154_154564


namespace liked_product_B_l154_154608

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l154_154608


namespace cubic_sum_of_roots_l154_154817

theorem cubic_sum_of_roots :
  ∀ (r s : ℝ), (r + s = 5) → (r * s = 6) → (r^3 + s^3 = 35) :=
by
  intros r s h₁ h₂
  sorry

end cubic_sum_of_roots_l154_154817


namespace sufficient_condition_perpendicular_l154_154088

variables {Plane Line : Type}
variables (l : Line) (α β : Plane)

-- Definitions for perpendicularity and parallelism
def perp (l : Line) (α : Plane) : Prop := sorry
def parallel (α β : Plane) : Prop := sorry

theorem sufficient_condition_perpendicular
  (h1 : perp l α) 
  (h2 : parallel α β) : 
  perp l β :=
sorry

end sufficient_condition_perpendicular_l154_154088


namespace height_of_water_a_height_of_water_b_height_of_water_c_l154_154909

noncomputable def edge_length : ℝ := 10  -- Edge length of the cube in cm.
noncomputable def angle_deg : ℝ := 20   -- Angle in degrees.

noncomputable def volume_a : ℝ := 100  -- Volume in cm^3 for case a)
noncomputable def height_a : ℝ := 2.53  -- Height in cm for case a)

noncomputable def volume_b : ℝ := 450  -- Volume in cm^3 for case b)
noncomputable def height_b : ℝ := 5.94  -- Height in cm for case b)

noncomputable def volume_c : ℝ := 900  -- Volume in cm^3 for case c)
noncomputable def height_c : ℝ := 10.29  -- Height in cm for case c)

theorem height_of_water_a :
  ∀ (edge_length angle_deg volume_a : ℝ), volume_a = 100 → height_a = 2.53 := by 
  sorry

theorem height_of_water_b :
  ∀ (edge_length angle_deg volume_b : ℝ), volume_b = 450 → height_b = 5.94 := by 
  sorry

theorem height_of_water_c :
  ∀ (edge_length angle_deg volume_c : ℝ), volume_c = 900 → height_c = 10.29 := by 
  sorry

end height_of_water_a_height_of_water_b_height_of_water_c_l154_154909


namespace smallest_arithmetic_mean_divisible_product_l154_154398

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l154_154398


namespace smallest_composite_no_prime_factors_less_than_15_l154_154644

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l154_154644


namespace total_earnings_is_correct_l154_154374

def lloyd_normal_hours : ℝ := 7.5
def lloyd_rate : ℝ := 4.5
def lloyd_overtime_rate : ℝ := 2.0
def lloyd_hours_worked : ℝ := 10.5

def casey_normal_hours : ℝ := 8
def casey_rate : ℝ := 5
def casey_overtime_rate : ℝ := 1.5
def casey_hours_worked : ℝ := 9.5

def lloyd_earnings : ℝ := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours_worked - lloyd_normal_hours) * lloyd_rate * lloyd_overtime_rate)

def casey_earnings : ℝ := (casey_normal_hours * casey_rate) + ((casey_hours_worked - casey_normal_hours) * casey_rate * casey_overtime_rate)

def total_earnings : ℝ := lloyd_earnings + casey_earnings

theorem total_earnings_is_correct : total_earnings = 112 := by
  sorry

end total_earnings_is_correct_l154_154374


namespace geometric_sequence_problem_l154_154816

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : q ≠ 1)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 6)
  (h_sum_squares : a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 = 18)
  (h_geom_seq : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 1 - a 2 + a 3 - a 4 + a 5 - a 6 + a 7 = 3 :=
by sorry

end geometric_sequence_problem_l154_154816


namespace log_three_pow_nine_pow_three_eq_six_l154_154028

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l154_154028


namespace product_even_permutation_l154_154865

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end product_even_permutation_l154_154865


namespace nine_consecutive_arithmetic_mean_divisible_1111_l154_154390

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l154_154390


namespace cauchy_schwarz_example_l154_154249

theorem cauchy_schwarz_example (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end cauchy_schwarz_example_l154_154249


namespace dani_pants_after_5_years_l154_154627

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end dani_pants_after_5_years_l154_154627


namespace minions_mistake_score_l154_154897

theorem minions_mistake_score :
  (minions_left_phone_on_untrusted_website ∧
   downloaded_file_from_untrusted_source ∧
   guidelines_by_cellular_operators ∧
   avoid_sharing_personal_info ∧
   unverified_files_may_be_harmful ∧
   double_extensions_signify_malicious_software) →
  score = 21 :=
by
  -- Here we would provide the proof steps which we skip with sorry
  sorry

end minions_mistake_score_l154_154897


namespace smallest_four_digit_mod_8_l154_154438

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l154_154438


namespace no_partition_of_integers_l154_154251

theorem no_partition_of_integers (A B C : Set ℕ) :
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (∀ a b, a ∈ A ∧ b ∈ B → (a^2 - a * b + b^2) ∈ C) ∧
  (∀ a b, a ∈ B ∧ b ∈ C → (a^2 - a * b + b^2) ∈ A) ∧
  (∀ a b, a ∈ C ∧ b ∈ A → (a^2 - a * b + b^2) ∈ B) →
  False := 
sorry

end no_partition_of_integers_l154_154251


namespace max_a_condition_slope_condition_exponential_inequality_l154_154237

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 1)
noncomputable def g (x a : ℝ) := f x a + a / Real.exp x

theorem max_a_condition (a : ℝ) (h_pos : a > 0) 
  (h_nonneg : ∀ x : ℝ, f x a ≥ 0) : a ≤ 1 := sorry

theorem slope_condition (a m : ℝ) 
  (ha : a ≤ -1) 
  (h_slope : ∀ x1 x2 : ℝ, x1 ≠ x2 → 
    (g x2 a - g x1 a) / (x2 - x1) > m) : m ≤ 3 := sorry

theorem exponential_inequality (n : ℕ) (hn : n > 0) : 
  (2 * (Real.exp n - 1)) / (Real.exp 1 - 1) ≥ n * (n + 1) := sorry

end max_a_condition_slope_condition_exponential_inequality_l154_154237


namespace smartphones_discount_l154_154007

theorem smartphones_discount
  (discount : ℝ)
  (cost_per_iphone : ℝ)
  (total_saving : ℝ)
  (num_people : ℕ)
  (num_iphones : ℕ)
  (total_cost : ℝ)
  (required_num : ℕ) :
  discount = 0.05 →
  cost_per_iphone = 600 →
  total_saving = 90 →
  num_people = 3 →
  num_iphones = 3 →
  total_cost = num_iphones * cost_per_iphone →
  required_num = num_iphones →
  required_num * cost_per_iphone * discount = total_saving →
  required_num = 3 :=
by
  intros
  sorry

end smartphones_discount_l154_154007


namespace moving_circle_passes_focus_l154_154145

noncomputable def parabola (x : ℝ) : Set (ℝ × ℝ) := {p | p.2 ^ 2 = 8 * p.1}
def is_tangent (c : ℝ × ℝ) (r : ℝ) : Prop := c.1 = -2 ∨ c.1 = -2 + 2 * r

theorem moving_circle_passes_focus
  (center : ℝ × ℝ) (H1 : center ∈ parabola center.1)
  (H2 : is_tangent center 2) :
  ∃ focus : ℝ × ℝ, focus = (2, 0) ∧ ∃ r : ℝ, ∀ p ∈ parabola center.1, dist center p = r := sorry

end moving_circle_passes_focus_l154_154145


namespace english_only_students_l154_154283

theorem english_only_students (T B G_total : ℕ) (hT : T = 40) (hB : B = 12) (hG_total : G_total = 22) :
  (T - (G_total - B) - B) = 18 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end english_only_students_l154_154283


namespace coordinates_of_point_l154_154690

theorem coordinates_of_point (a : ℝ) (P : ℝ × ℝ) (hy : P = (a^2 - 1, a + 1)) (hx : (a^2 - 1) = 0) :
  P = (0, 2) ∨ P = (0, 0) :=
sorry

end coordinates_of_point_l154_154690


namespace hike_people_count_l154_154388

theorem hike_people_count :
  let cars := 3
  let taxis := 6
  let vans := 2
  let people_per_car := 4
  let people_per_taxi := 6
  let people_per_van := 5
  let total_people := (cars * people_per_car) + (taxis * people_per_taxi) + (vans * people_per_van)
  in total_people = 58 :=
by
  -- Proof steps will go here
  sorry

end hike_people_count_l154_154388


namespace value_of_x_for_zero_expression_l154_154486

theorem value_of_x_for_zero_expression (x : ℝ) (h : (x-5 = 0)) (h2 : (6*x - 12 ≠ 0)) :
  x = 5 :=
by {
  sorry
}

end value_of_x_for_zero_expression_l154_154486


namespace solve_system_of_equations_l154_154550

theorem solve_system_of_equations (x y z : ℝ) :
  (2 * x^2 / (1 + x^2) = y) →
  (2 * y^2 / (1 + y^2) = z) →
  (2 * z^2 / (1 + z^2) = x) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l154_154550


namespace sum_of_two_integers_is_22_l154_154422

noncomputable def a_and_b_sum_to_S : Prop :=
  ∃ (a b S : ℕ), 
    a + b = S ∧ 
    a^2 - b^2 = 44 ∧ 
    a * b = 120 ∧ 
    S = 22

theorem sum_of_two_integers_is_22 : a_and_b_sum_to_S :=
by {
  sorry
}

end sum_of_two_integers_is_22_l154_154422


namespace muffin_banana_ratio_l154_154107

variable {R : Type} [LinearOrderedField R]

-- Define the costs of muffins and bananas
variables {m b : R}

-- Susie's cost
def susie_cost (m b : R) := 4 * m + 5 * b

-- Calvin's cost for three times Susie's items
def calvin_cost_tripled (m b : R) := 12 * m + 15 * b

-- Calvin's actual cost
def calvin_cost_actual (m b : R) := 2 * m + 12 * b

theorem muffin_banana_ratio (m b : R) (h : calvin_cost_tripled m b = calvin_cost_actual m b) : m = (3 / 10) * b :=
by sorry

end muffin_banana_ratio_l154_154107


namespace cardinal_transitivity_l154_154679

variable {α β γ : Cardinal}

theorem cardinal_transitivity (h1 : α < β) (h2 : β < γ) : α < γ :=
  sorry

end cardinal_transitivity_l154_154679


namespace largest_triangle_perimeter_with_7_9_x_l154_154780

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def triangle_side_x_valid (x : ℕ) : Prop :=
  is_divisible_by_3 x ∧ 2 < x ∧ x < 16

theorem largest_triangle_perimeter_with_7_9_x (x : ℕ) (h : triangle_side_x_valid x) : 
  ∃ P : ℕ, P = 7 + 9 + x ∧ P = 31 :=
by
  sorry

end largest_triangle_perimeter_with_7_9_x_l154_154780


namespace probability_of_two_correct_deliveries_l154_154809

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

def probability_exactly_two_correct_deliveries : ℚ :=
  let n := 4
  let k := 2
  let pairs := choose n k
  let prob_of_specific_two_correct := (1 / 4 : ℚ) * (1 / 3) * (1 / 2)
  let total_prob := pairs * prob_of_specific_two_correct
  total_prob

theorem probability_of_two_correct_deliveries :
  probability_exactly_two_correct_deliveries = 1 / 4 :=
by
  sorry

end probability_of_two_correct_deliveries_l154_154809


namespace percentage_error_l154_154285

theorem percentage_error (x : ℝ) : ((x * 3 - x / 5) / (x * 3) * 100) = 93.33 := 
  sorry

end percentage_error_l154_154285


namespace total_tape_length_is_230_l154_154207

def tape_length (n : ℕ) (len_piece : ℕ) (overlap : ℕ) : ℕ :=
  len_piece + (n - 1) * (len_piece - overlap)

theorem total_tape_length_is_230 :
  tape_length 15 20 5 = 230 := 
    sorry

end total_tape_length_is_230_l154_154207


namespace find_k_of_quadratic_polynomial_l154_154362

variable (k : ℝ)

theorem find_k_of_quadratic_polynomial (h1 : (k - 2) = 0) (h2 : k ≠ 0) : k = 2 :=
by
  -- proof omitted
  sorry

end find_k_of_quadratic_polynomial_l154_154362


namespace simplify_expression_l154_154252

variable (a b : ℝ)

theorem simplify_expression (a b : ℝ) :
  (6 * a^5 * b^2) / (3 * a^3 * b^2) + ((2 * a * b^3)^2) / ((-b^2)^3) = -2 * a^2 :=
by 
  sorry

end simplify_expression_l154_154252


namespace product_of_three_numbers_l154_154887

theorem product_of_three_numbers (x y z n : ℝ)
  (h_sum : x + y + z = 180)
  (h_n_eq_8x : n = 8 * x)
  (h_n_eq_y_minus_10 : n = y - 10)
  (h_n_eq_z_plus_10 : n = z + 10) :
  x * y * z = (180 / 17) * ((1440 / 17) ^ 2 - 100) := by
  sorry

end product_of_three_numbers_l154_154887


namespace sum_of_squares_of_roots_l154_154803

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end sum_of_squares_of_roots_l154_154803


namespace rectangular_container_volume_l154_154280

theorem rectangular_container_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 :=
by
  sorry

end rectangular_container_volume_l154_154280


namespace primes_with_prime_remainders_l154_154505

namespace PrimePuzzle

open Nat

def primes_between (a b : Nat) : List Nat :=
  (List.range' (a + 1) (b - a)).filter Nat.Prime

def prime_remainders (lst : List Nat) (m : Nat) : List Nat :=
  (lst.map (λ n => n % m)).filter Nat.Prime

theorem primes_with_prime_remainders : 
  primes_between 40 85 = [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] ∧ 
  prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12 = [5, 7, 7, 11, 11, 7, 11] ∧ 
  (prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12).toFinset.card = 9 := 
by 
  sorry

end PrimePuzzle

end primes_with_prime_remainders_l154_154505


namespace max_true_statements_l154_154861

theorem max_true_statements (a b : ℝ) :
  ((a < b) → (b < 0) → (a < 0) → ¬(1 / a < 1 / b)) ∧
  ((a < b) → (b < 0) → (a < 0) → ¬(a^2 < b^2)) →
  3 = 3
:=
by
  intros
  sorry

end max_true_statements_l154_154861


namespace part1_part2_l154_154354

open Real

def f (x a : ℝ) : ℝ :=
  x^2 + a * x + 3

theorem part1 (x : ℝ) (h : x^2 - 4 * x + 3 < 0) :
  1 < x ∧ x < 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a > 0) :
  -2 * sqrt 3 < a ∧ a < 2 * sqrt 3 :=
  sorry

end part1_part2_l154_154354


namespace flower_problem_solution_l154_154888

/-
Given the problem conditions:
1. There are 88 flowers.
2. Each flower was visited by at least one bee.
3. Each bee visited exactly 54 flowers.

Prove that bitter flowers exceed sweet flowers by 14.
-/

noncomputable def flower_problem : Prop :=
  ∃ (s g : ℕ), 
    -- Condition: The total number of flowers
    s + g + (88 - s - g) = 88 ∧ 
    -- Condition: Total number of visits by bees
    3 * 54 = 162 ∧ 
    -- Proof goal: Bitter flowers exceed sweet flowers by 14
    g - s = 14

theorem flower_problem_solution : flower_problem :=
by
  sorry

end flower_problem_solution_l154_154888


namespace fraction_of_board_shaded_is_one_fourth_l154_154852

def totalArea : ℕ := 16
def shadedTopLeft : ℕ := 4
def shadedBottomRight : ℕ := 4
def fractionShaded (totalArea shadedTopLeft shadedBottomRight : ℕ) : ℚ :=
  (shadedTopLeft + shadedBottomRight) / totalArea

theorem fraction_of_board_shaded_is_one_fourth :
  fractionShaded totalArea shadedTopLeft shadedBottomRight = 1 / 4 := by
  sorry

end fraction_of_board_shaded_is_one_fourth_l154_154852


namespace fraction_zero_x_eq_2_l154_154223

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end fraction_zero_x_eq_2_l154_154223


namespace math_problem_l154_154708

variable {R : Type} [LinearOrderedField R]

theorem math_problem
  (a b : R) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
by
  sorry

end math_problem_l154_154708


namespace triangles_area_possibilities_unique_l154_154542

noncomputable def triangle_area_possibilities : ℕ :=
  -- Define lengths of segments on the first line
  let AB := 1
  let BC := 2
  let CD := 3
  -- Sum to get total lengths
  let AC := AB + BC -- 3
  let AD := AB + BC + CD -- 6
  -- Define length of the segment on the second line
  let EF := 2
  -- GH is a segment not parallel to the first two lines
  let GH := 1
  -- The number of unique possible triangle areas
  4

theorem triangles_area_possibilities_unique :
  triangle_area_possibilities = 4 := 
sorry

end triangles_area_possibilities_unique_l154_154542


namespace gcd_294_84_l154_154891

theorem gcd_294_84 : gcd 294 84 = 42 :=
by
  sorry

end gcd_294_84_l154_154891


namespace range_of_a_l154_154497

theorem range_of_a (x a : ℝ) (p : |x - 2| < 3) (q : 0 < x ∧ x < a) :
  (0 < a ∧ a ≤ 5) := 
sorry

end range_of_a_l154_154497


namespace red_before_green_probability_l154_154917

open Classical

noncomputable def probability_red_before_green (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  let total_arrangements := (Nat.choose (total_chips - 1) green_chips)
  let favorable_arrangements := Nat.choose (total_chips - red_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem red_before_green_probability :
  probability_red_before_green 8 4 3 = 3 / 7 :=
sorry

end red_before_green_probability_l154_154917


namespace sum_of_squares_of_roots_of_quadratic_l154_154804

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end sum_of_squares_of_roots_of_quadratic_l154_154804


namespace minimum_trains_needed_l154_154464

theorem minimum_trains_needed (n : ℕ) (h : 50 * n >= 645) : n = 13 :=
by
  sorry

end minimum_trains_needed_l154_154464


namespace smallest_composite_no_prime_factors_lt_15_l154_154641

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l154_154641


namespace solve_exp_l154_154905

theorem solve_exp (x : ℕ) : 8^x = 2^9 → x = 3 :=
by
  sorry

end solve_exp_l154_154905


namespace find_k_value_l154_154830

theorem find_k_value (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ∧
  (∀ x1 x2 : ℝ, (k * x1^2 + 4 * x1 + 4 = 0 ∧ k * x2^2 + 4 * x2 + 4 = 0) → x1 = x2) →
  (k = 0 ∨ k = 1) :=
by
  intros h
  sorry

end find_k_value_l154_154830


namespace function_relationship_l154_154533

theorem function_relationship (f : ℝ → ℝ)
  (h₁ : ∀ x, f (x + 1) = f (-x + 1))
  (h₂ : ∀ x, x ≥ 1 → f x = (1 / 2) ^ x - 1) :
  f (2 / 3) > f (3 / 2) ∧ f (3 / 2) > f (1 / 3) :=
by sorry

end function_relationship_l154_154533


namespace find_k_for_perfect_square_l154_154630

theorem find_k_for_perfect_square :
  ∃ k : ℤ, (k = 12 ∨ k = -12) ∧ (∀ n : ℤ, ∃ a b : ℤ, 4 * n^2 + k * n + 9 = (a * n + b)^2) :=
sorry

end find_k_for_perfect_square_l154_154630


namespace area_BEIH_l154_154591

noncomputable def quadrilateral_area : ℝ := 
  let A := (0, 3)
  let B := (0, 0)
  let C := (3, 0)
  let D := (3, 3)
  let E := (0, 1.5)
  let F := (1.5, 0)
  let I := (0.6, 1.8)
  let H := (1, 1)
  let vertices := [B, E, I, H]
  (0.5 * (abs ((fst B * snd E + fst E * snd I + fst I * snd H + fst H * snd B) -
                 (snd B * fst E + snd E * fst I + snd I * fst H + snd H * fst B))))

theorem area_BEIH :
  quadrilateral_area = 3 / 5 := 
sorry

end area_BEIH_l154_154591


namespace arithmetic_sequence_term_2018_l154_154199

theorem arithmetic_sequence_term_2018 
  (a : ℕ → ℤ)  -- sequence a_n with ℕ index and ℤ values
  (S9 : Σ (n : ℕ), 9 + 9 * (1 / 2) * (fin n) = 27)  -- sum of the first 9 terms is 27
  (a10 : Σ (n : ℕ), 10 = fin n)  -- rich sum value 

: a 2018 = 2016 := by
  sorry

end arithmetic_sequence_term_2018_l154_154199


namespace probability_real_part_greater_imag_part_l154_154890

noncomputable def probability_real_gt_imag (x y : Fin 6) : ℚ :=
  if (x + 1 : ℕ) > (y + 1 : ℕ) then 1 else 0

theorem probability_real_part_greater_imag_part :
  let outcomes := Fin 6 × Fin 6;
  let favorable_outcomes := (outcomes.filter $ λ (x y : Fin 6), (x.val + 1) > (y.val + 1));
  let probability := (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)
  probability = 5 / 12 :=
by
  sorry

end probability_real_part_greater_imag_part_l154_154890


namespace doctors_assignment_l154_154934

theorem doctors_assignment :
  ∃ (assignments : Finset (Fin 3 → Finset (Fin 5))),
    (∀ h ∈ assignments, (∀ i, ∃ j ∈ h i, True) ∧
      ¬(∃ i j, (A ∈ h i ∧ B ∈ h j ∨ A ∈ h j ∧ B ∈ h i)) ∧
      ¬(∃ i j, (C ∈ h i ∧ D ∈ h j ∨ C ∈ h j ∧ D ∈ h i))) ∧
    assignments.card = 84 :=
sorry

end doctors_assignment_l154_154934


namespace find_y_coordinate_of_C_l154_154821

def point (x : ℝ) (y : ℝ) : Prop := y^2 = x + 4

def perp_slope (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) * (y3 - y2) / (x3 - x2) = -1

def valid_y_coordinate_C (x0 : ℝ) : Prop :=
  x0 ≤ 0 ∨ 4 ≤ x0

theorem find_y_coordinate_of_C (x0 : ℝ) :
  (∀ (x y : ℝ), point x y) →
  (∃ (x2 y2 x3 y3 : ℝ), point x2 y2 ∧ point x3 y3 ∧ perp_slope 0 2 x2 y2 x3 y3) →
  valid_y_coordinate_C x0 :=
sorry

end find_y_coordinate_of_C_l154_154821


namespace complex_pow_i_2019_l154_154500

theorem complex_pow_i_2019 : (Complex.I)^2019 = -Complex.I := 
by
  sorry

end complex_pow_i_2019_l154_154500


namespace sum_third_column_l154_154017

variable (a b c d e f g h i : ℕ)

theorem sum_third_column :
  (a + b + c = 24) →
  (d + e + f = 26) →
  (g + h + i = 40) →
  (a + d + g = 27) →
  (b + e + h = 20) →
  (c + f + i = 43) :=
by
  intros
  sorry

end sum_third_column_l154_154017


namespace periodic_function_implies_rational_ratio_l154_154815

noncomputable def g (i : ℕ) (a ω θ x : ℝ) : ℝ := 
  a * Real.sin (ω * x + θ)

theorem periodic_function_implies_rational_ratio 
  (a1 a2 ω1 ω2 θ1 θ2 : ℝ) (h1 : a1 * ω1 ≠ 0) (h2 : a2 * ω2 ≠ 0)
  (h3 : |ω1| ≠ |ω2|) 
  (hf_periodic : ∃ T : ℝ, ∀ x : ℝ, g 1 a1 ω1 θ1 (x + T) + g 2 a2 ω2 θ2 (x + T) = g 1 a1 ω1 θ1 x + g 2 a2 ω2 θ2 x) :
  ∃ m n : ℤ, n ≠ 0 ∧ ω1 / ω2 = m / n :=
sorry

end periodic_function_implies_rational_ratio_l154_154815


namespace log_three_pow_nine_pow_three_eq_six_l154_154027

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l154_154027


namespace inequality_solution_l154_154382

theorem inequality_solution (x : ℝ) :
  (x > -4 ∧ x < -5 / 3) ↔ 
  (2 * x + 3) / (3 * x + 5) > (4 * x + 1) / (x + 4) := 
sorry

end inequality_solution_l154_154382


namespace intersection_A_complement_B_l154_154957

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 > 0}
def comR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

theorem intersection_A_complement_B : A ∩ (comR B) = {x | -1 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_A_complement_B_l154_154957


namespace speed_of_first_train_l154_154744

theorem speed_of_first_train
  (v : ℝ)
  (d : ℝ)
  (distance_between_stations : ℝ := 450)
  (speed_of_second_train : ℝ := 25)
  (additional_distance_first_train : ℝ := 50)
  (meet_time_condition : d / v = (d - additional_distance_first_train) / speed_of_second_train)
  (total_distance_condition : d + (d - additional_distance_first_train) = distance_between_stations) :
  v = 31.25 :=
by {
  sorry
}

end speed_of_first_train_l154_154744


namespace contrapositive_ex_l154_154384

theorem contrapositive_ex (x y : ℝ)
  (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) :
  ¬ (x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0 :=
by
  sorry

end contrapositive_ex_l154_154384


namespace Louisa_traveled_240_miles_first_day_l154_154245

noncomputable def distance_first_day (h : ℕ) := 60 * (h - 3)

theorem Louisa_traveled_240_miles_first_day :
  ∃ h : ℕ, 420 = 60 * h ∧ distance_first_day h = 240 :=
by
  sorry

end Louisa_traveled_240_miles_first_day_l154_154245


namespace remainder_when_divided_by_15_l154_154756

theorem remainder_when_divided_by_15 (N : ℤ) (k : ℤ) 
  (h : N = 45 * k + 31) : (N % 15) = 1 := by
  sorry

end remainder_when_divided_by_15_l154_154756


namespace problem_part1_problem_part2_l154_154674

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * Real.pi / 3) + sqrt 3 * Real.sin (2 * x)

theorem problem_part1 : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f(x + T) = f(x) ∧ T = Real.pi) ∧ 
  (∀ x : ℝ, -1 <= f(x) ∧ f(x) <= 1) ∧ 
  (∃ x : ℝ, f(x) = 1) :=
sorry

theorem problem_part2 (A B C : ℝ) (hC : 0 < C ∧ C < Real.pi) (hAC : AC = 1) (hBC : BC = 3)
  (h_f : f (C / 2) = -1 / 2) :
  ∃ s : ℝ, s = (3 * sqrt 21) / 14 :=
sorry

end problem_part1_problem_part2_l154_154674


namespace divisor_of_subtracted_number_l154_154940

theorem divisor_of_subtracted_number (n : ℕ) (m : ℕ) (h : n = 5264 - 11) : Nat.gcd n 5264 = 5253 :=
by
  sorry

end divisor_of_subtracted_number_l154_154940


namespace gcd_g_50_52_l154_154866

/-- Define the polynomial function g -/
def g (x : ℤ) : ℤ := x^2 - 3 * x + 2023

/-- The theorem stating the gcd of g(50) and g(52) -/
theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g_50_52_l154_154866


namespace dennis_floor_l154_154323

theorem dennis_floor :
  ∃ d c b f e: ℕ, 
  (d = c + 2) ∧ 
  (c = b + 1) ∧ 
  (c = f / 4) ∧ 
  (f = 16) ∧ 
  (e = d / 2) ∧ 
  (d = 6) :=
by
  sorry

end dennis_floor_l154_154323


namespace vet_fees_cat_result_l154_154783

-- Given conditions
def vet_fees_dog : ℕ := 15
def families_dogs : ℕ := 8
def families_cats : ℕ := 3
def vet_donation : ℕ := 53

-- Mathematical equivalency in Lean
noncomputable def vet_fees_cat (C : ℕ) : Prop :=
  (1 / 3 : ℚ) * (families_dogs * vet_fees_dog + families_cats * C) = vet_donation

-- Prove the vet fees for cats are 13 using above conditions
theorem vet_fees_cat_result : ∃ (C : ℕ), vet_fees_cat C ∧ C = 13 :=
by {
  use 13,
  sorry
}

end vet_fees_cat_result_l154_154783


namespace socks_selection_l154_154144

theorem socks_selection :
  ∀ (R Y G B O : ℕ), 
    R = 80 → Y = 70 → G = 50 → B = 60 → O = 40 →
    (∃ k, k = 38 ∧ ∀ (N : ℕ → ℕ), (N R + N Y + N G + N B + N O ≥ k)
          → (exists (pairs : ℕ), pairs ≥ 15 ∧ pairs = (N R / 2) + (N Y / 2) + (N G / 2) + (N B / 2) + (N O / 2) )) :=
by
  sorry

end socks_selection_l154_154144


namespace equation_of_line_perpendicular_and_passing_point_l154_154915

theorem equation_of_line_perpendicular_and_passing_point :
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = -1 ∧
  (∀ (x y : ℝ), (2 * x - 3 * y + 4 = 0 → y = (2 / 3) * x + 4 / 3) →
  (∀ (x1 y1 : ℝ), x1 = -1 ∧ y1 = 2 →
  (a * x1 + b * y1 + c = 0) ∧
  (∀ (x y : ℝ), (-3 / 2) * (x + 1) + 2 = y) →
  (a * x + b * y + c = 0))) :=
sorry

end equation_of_line_perpendicular_and_passing_point_l154_154915


namespace manufacturing_department_degrees_l154_154723

def percentage_of_circle (percentage : ℕ) (total_degrees : ℕ) : ℕ :=
  (percentage * total_degrees) / 100

theorem manufacturing_department_degrees :
  percentage_of_circle 30 360 = 108 :=
by
  sorry

end manufacturing_department_degrees_l154_154723


namespace permutations_count_l154_154264

-- Define the conditions
variable (n : ℕ)
variable (a : Fin n → ℕ)

-- Define the main proposition
theorem permutations_count (hn : 2 ≤ n) (h_perm : ∀ k : Fin n, a k ≥ k.val - 2) :
  ∃! L, L = 2 * 3 ^ (n - 2) :=
by
  sorry

end permutations_count_l154_154264


namespace shaded_area_proof_l154_154306

noncomputable def total_shaded_area (side_length: ℝ) (large_square_ratio: ℝ) (small_square_ratio: ℝ): ℝ := 
  let S := side_length / large_square_ratio
  let T := S / small_square_ratio
  let large_square_area := S ^ 2
  let small_square_area := T ^ 2
  large_square_area + 12 * small_square_area

theorem shaded_area_proof
  (h1: ∀ side_length, side_length = 15)
  (h2: ∀ large_square_ratio, large_square_ratio = 5)
  (h3: ∀ small_square_ratio, small_square_ratio = 4)
  : total_shaded_area 15 5 4 = 15.75 :=
by
  sorry

end shaded_area_proof_l154_154306


namespace smallest_composite_no_prime_under_15_correct_l154_154645

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l154_154645


namespace rectangle_area_l154_154428

theorem rectangle_area
    (w l : ℕ)
    (h₁ : 28 = 2 * (l + w))
    (h₂ : w = 6) : l * w = 48 :=
by
  sorry

end rectangle_area_l154_154428


namespace calculate_roots_l154_154927

noncomputable def cube_root (x : ℝ) := x^(1/3 : ℝ)
noncomputable def square_root (x : ℝ) := x^(1/2 : ℝ)

theorem calculate_roots : cube_root (-8) + square_root 9 = 1 :=
by
  sorry

end calculate_roots_l154_154927


namespace pyramid_volume_l154_154261

-- Define the conditions
def height_vertex_to_center_base := 12 -- cm
def side_of_square_base := 10 -- cm
def base_area := side_of_square_base * side_of_square_base -- cm²
def volume := (1 / 3) * base_area * height_vertex_to_center_base -- cm³

-- State the theorem
theorem pyramid_volume : volume = 400 := 
by
  -- Placeholder for the proof
  sorry

end pyramid_volume_l154_154261


namespace log_base_3_of_9_cubed_l154_154019
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l154_154019


namespace value_of_b_is_one_l154_154988

open Complex

theorem value_of_b_is_one (a b : ℝ) (h : (1 + I) / (1 - I) = a + b * I) : b = 1 := 
by
  sorry

end value_of_b_is_one_l154_154988


namespace train_pass_bridge_in_50_seconds_l154_154757

def length_of_train : ℕ := 360
def length_of_bridge : ℕ := 140
def speed_of_train_kmh : ℕ := 36
def total_distance : ℕ := length_of_train + length_of_bridge
def speed_of_train_ms : ℚ := (speed_of_train_kmh * 1000 : ℚ) / 3600 -- we use ℚ to avoid integer division issues
def time_to_pass_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge = 50 := by
  sorry

end train_pass_bridge_in_50_seconds_l154_154757


namespace sum_of_first_9_primes_l154_154131

theorem sum_of_first_9_primes : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23) = 100 := 
by
  sorry

end sum_of_first_9_primes_l154_154131


namespace steve_pie_difference_l154_154554

-- Definitions of conditions
def apple_pie_days : Nat := 3
def cherry_pie_days : Nat := 2
def pies_per_day : Nat := 12

-- Theorem statement
theorem steve_pie_difference : 
  (apple_pie_days * pies_per_day) - (cherry_pie_days * pies_per_day) = 12 := 
by
  sorry

end steve_pie_difference_l154_154554


namespace probability_of_selecting_green_ball_l154_154599

def container_I :  ℕ × ℕ := (5, 5) -- (red balls, green balls)
def container_II : ℕ × ℕ := (3, 3) -- (red balls, green balls)
def container_III : ℕ × ℕ := (4, 2) -- (red balls, green balls)
def container_IV : ℕ × ℕ := (6, 6) -- (red balls, green balls)

def total_containers : ℕ := 4

def probability_of_green_ball (red_green : ℕ × ℕ) : ℚ :=
  let (red, green) := red_green
  green / (red + green)

noncomputable def combined_probability_of_green_ball : ℚ :=
  (1 / total_containers) *
  (probability_of_green_ball container_I +
   probability_of_green_ball container_II +
   probability_of_green_ball container_III +
   probability_of_green_ball container_IV)

theorem probability_of_selecting_green_ball : 
  combined_probability_of_green_ball = 11 / 24 :=
sorry

end probability_of_selecting_green_ball_l154_154599


namespace rita_remaining_money_l154_154875

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l154_154875


namespace factor_theorem_q_value_l154_154132

theorem factor_theorem_q_value (q : ℤ) (m : ℤ) :
  (∀ m, (m - 8) ∣ (m^2 - q * m - 24)) → q = 5 :=
by
  sorry

end factor_theorem_q_value_l154_154132


namespace tangent_line_to_ex_l154_154514

theorem tangent_line_to_ex (b : ℝ) : (∃ x0 : ℝ, (∀ x : ℝ, (e^x - e^x0 - (x - x0) * e^x0 = 0) ↔ y = x + b)) → b = 1 :=
by
  sorry

end tangent_line_to_ex_l154_154514


namespace power_sums_l154_154189

-- Definitions as per the given conditions
variables (m n a b : ℕ)
variables (hm : 0 < m) (hn : 0 < n)
variables (ha : 2^m = a) (hb : 2^n = b)

-- The theorem statement
theorem power_sums (hmn : 0 < m + n) : 2^(m + n) = a * b :=
by
  sorry

end power_sums_l154_154189


namespace barbara_total_candies_l154_154784

theorem barbara_total_candies :
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855 := 
by
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  show boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855
  sorry

end barbara_total_candies_l154_154784


namespace company_picnic_attendance_l154_154691

theorem company_picnic_attendance :
  ∀ (employees men women men_attending women_attending : ℕ)
  (h_employees : employees = 100)
  (h_men : men = 55)
  (h_women : women = 45)
  (h_men_attending: men_attending = 11)
  (h_women_attending: women_attending = 18),
  (100 * (men_attending + women_attending) / employees) = 29 := 
by
  intros employees men women men_attending women_attending 
         h_employees h_men h_women h_men_attending h_women_attending
  sorry

end company_picnic_attendance_l154_154691


namespace quadratic_passes_through_l154_154112

def quadratic_value_at_point (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_passes_through (a b c : ℝ) :
  quadratic_value_at_point a b c 1 = 5 ∧ 
  quadratic_value_at_point a b c 3 = n ∧ 
  a * (-2)^2 + b * (-2) + c = -8 ∧ 
  (-4*a + b = 0) → 
  n = 253/9 := 
sorry

end quadratic_passes_through_l154_154112


namespace problem_condition_l154_154951

variable {f : ℝ → ℝ}
variable {a b : ℝ}

noncomputable def fx_condition (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x + x * (deriv f x) < 0

theorem problem_condition {f : ℝ → ℝ} {a b : ℝ} (h1 : fx_condition f) (h2 : a < b) :
  a * f a > b * f b :=
sorry

end problem_condition_l154_154951


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154396

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154396


namespace acute_angle_proof_l154_154499

theorem acute_angle_proof
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : Real.cos (α + β) = Real.sin (α - β)) : α = π / 4 :=
  sorry

end acute_angle_proof_l154_154499


namespace percentage_respondents_liked_B_l154_154610

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l154_154610


namespace alvin_age_l154_154010

theorem alvin_age (A S : ℕ) (h_s : S = 10) (h_cond : S = 1/2 * A - 5) : A = 30 := by
  sorry

end alvin_age_l154_154010


namespace distance_from_house_to_work_l154_154728

-- Definitions for the conditions
variables (D : ℝ) (speed_to_work speed_back_work : ℝ) (time_to_work time_back_work total_time : ℝ)

-- Specific conditions in the problem
noncomputable def conditions : Prop :=
  (speed_back_work = 20) ∧
  (speed_to_work = speed_back_work / 2) ∧
  (time_to_work = D / speed_to_work) ∧
  (time_back_work = D / speed_back_work) ∧
  (total_time = 6) ∧
  (time_to_work + time_back_work = total_time)

-- The statement to prove the distance D is 40 km given the conditions
theorem distance_from_house_to_work (h : conditions D speed_to_work speed_back_work time_to_work time_back_work total_time) : D = 40 :=
sorry

end distance_from_house_to_work_l154_154728


namespace smallest_arithmetic_mean_divisible_product_l154_154397

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l154_154397


namespace example_proof_l154_154828

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom axiom1 (x y : ℝ) : f (x - y) = f x * g y - g x * f y
axiom axiom2 (x : ℝ) : f x ≠ 0
axiom axiom3 : f 1 = f 2

theorem example_proof : g (-1) + g 1 = 1 := by
  sorry

end example_proof_l154_154828


namespace avg_and_exp_val_l154_154978

noncomputable def x : Fin 10 → ℝ
| ⟨0, _⟩ => 38
| ⟨1, _⟩ => 41
| ⟨2, _⟩ => 44
| ⟨3, _⟩ => 51
| ⟨4, _⟩ => 54
| ⟨5, _⟩ => 56
| ⟨6, _⟩ => 58
| ⟨7, _⟩ => 64
| ⟨8, _⟩ => 74
| ⟨9, _⟩ => 80
| _ => 0 -- This case should never happen

def average (x : Fin 10 → ℝ) : ℝ :=
  (Finset.univ.sum x) / 10

def variance (x : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (x i - mean) ^ 2)) / 10

theorem avg_and_exp_val:
  let mean := average x in
  mean = 56 ∧
  let var := variance x mean in
  var = 169 ∧ 
  let mu := mean in
  let sigma := Real.sqrt var in
  let p := 0.9545 in
  ∃ Y : ℕ → ℝ, 
    (binomial 100 p).expected_value Y = 95.45 :=
by 
  sorry

end avg_and_exp_val_l154_154978


namespace max_t_for_real_root_l154_154670

theorem max_t_for_real_root (t : ℝ) (x : ℝ) 
  (h : 0 < x ∧ x < π ∧ (t+1) * Real.cos x - t * Real.sin x = t + 2) : t = -1 :=
sorry

end max_t_for_real_root_l154_154670


namespace number_of_men_first_group_l154_154767

theorem number_of_men_first_group :
  (∃ M : ℕ, 30 * 3 * (M : ℚ) * (84 / 30) / 3 = 112 / 6) → ∃ M : ℕ, M = 20 := 
by
  sorry

end number_of_men_first_group_l154_154767


namespace smallest_arithmetic_mean_divisible_1111_l154_154406

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l154_154406


namespace num_even_3digit_nums_lt_700_l154_154429

theorem num_even_3digit_nums_lt_700 
  (digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}) 
  (even_digits : Finset ℕ := {2, 4, 6}) 
  (h1 : ∀ n ∈ digits, n < 10)
  (h2 : 0 ∉ digits) : 
  ∃ n, n = 126 ∧ ∀ d, d ∈ digits → 
  (d < 10) ∧ ∀ u, u ∈ even_digits → 
  (u < 10) 
:=
  sorry

end num_even_3digit_nums_lt_700_l154_154429


namespace prime_pairs_l154_154796

-- Define the predicate to check whether a number is a prime.
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the main theorem.
theorem prime_pairs (p q : Nat) (hp : is_prime p) (hq : is_prime q) : 
  (p^3 - q^5 = (p + q)^2) → (p = 7 ∧ q = 3) :=
by
  sorry

end prime_pairs_l154_154796


namespace no_positive_integer_solutions_l154_154246

theorem no_positive_integer_solutions (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  x^3 + 2 * y^3 ≠ 4 * z^3 :=
by
  sorry

end no_positive_integer_solutions_l154_154246


namespace sufficient_not_necessary_condition_l154_154664

-- Definition of the proposition p
def prop_p (m : ℝ) := ∀ x : ℝ, x^2 - 4 * x + 2 * m ≥ 0

-- Statement of the proof problem
theorem sufficient_not_necessary_condition (m : ℝ) : 
  (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 → m ≥ 2) ∧ (m ≥ 2 → prop_p m) → (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 ↔ prop_p m) :=
sorry

end sufficient_not_necessary_condition_l154_154664


namespace sum_of_solutions_of_quadratic_l154_154943

theorem sum_of_solutions_of_quadratic :
  ∀ a b c x₁ x₂ : ℝ, a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (∃ s : ℝ, s = x₁ + x₂ ∧ -b / a = s) :=
by
  sorry

end sum_of_solutions_of_quadratic_l154_154943


namespace new_three_digit_number_l154_154071

theorem new_three_digit_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) :
  let original := 10 * t + u
  let new_number := (original * 10) + 2
  new_number = 100 * t + 10 * u + 2 :=
by
  sorry

end new_three_digit_number_l154_154071


namespace leaked_before_fixing_l154_154468

def total_leaked_oil := 6206
def leaked_while_fixing := 3731

theorem leaked_before_fixing :
  total_leaked_oil - leaked_while_fixing = 2475 := by
  sorry

end leaked_before_fixing_l154_154468


namespace sum_of_squares_of_roots_l154_154802

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end sum_of_squares_of_roots_l154_154802


namespace percentage_respondents_liked_B_l154_154611

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l154_154611


namespace connie_total_markers_l154_154473

theorem connie_total_markers (red_markers : ℕ) (blue_markers : ℕ) 
                              (h1 : red_markers = 41)
                              (h2 : blue_markers = 64) : 
                              red_markers + blue_markers = 105 := by
  sorry

end connie_total_markers_l154_154473


namespace house_transaction_l154_154912

variable (initial_value : ℝ) (loss_rate : ℝ) (gain_rate : ℝ) (final_loss : ℝ)

theorem house_transaction
  (h_initial : initial_value = 12000)
  (h_loss : loss_rate = 0.15)
  (h_gain : gain_rate = 0.15)
  (h_final_loss : final_loss = 270) :
  let selling_price := initial_value * (1 - loss_rate)
  let buying_price := selling_price * (1 + gain_rate)
  (initial_value - buying_price) = final_loss :=
by
  simp only [h_initial, h_loss, h_gain, h_final_loss]
  sorry

end house_transaction_l154_154912


namespace number_of_solutions_l154_154206

theorem number_of_solutions :
  {p : ℝ × ℝ // p.1 + 2 * p.2 = 4 ∧ | |p.1| - |p.2| | = sin (Real.pi / 4)}.set.card = 2 :=
begin
  sorry

end number_of_solutions_l154_154206


namespace range_of_a_l154_154110

noncomputable def has_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * a * 9^(Real.sin x) + 4 * a * 3^(Real.sin x) + a - 8 = 0

theorem range_of_a : ∀ a : ℝ,
  (has_solutions a ↔ (8 / 31 <= a ∧ a <= 72 / 23)) := sorry

end range_of_a_l154_154110


namespace math_problem_proof_l154_154339

noncomputable def ellipse_equation : Prop := 
  let e := (Real.sqrt 2) / 2
  ∃ (a b : ℝ), 0 < a ∧ a > b ∧ e = (Real.sqrt 2) / 2 ∧ 
    (∀ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ x^2 / 2 + y^2 = 1)

noncomputable def fixed_point_exist : Prop :=
  let S := (0, 1/3) 
  ∀ k : ℝ, ∃ A B : ℝ × ℝ, 
    let M := (0, 1)
    ( 
        (A.1, A.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (B.1, B.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (S.2 = k * S.1 - 1 / 3) ∧ 
        ((A.1 - M.1)^2 + (A.2 - M.2)^2) + ((B.1 - M.1)^2 + (B.2 - M.2)^2) = ((A.1 - B.1)^2 + (A.2 - M.2)^2) / 2)

theorem math_problem_proof : ellipse_equation ∧ fixed_point_exist := sorry

end math_problem_proof_l154_154339


namespace count_numbers_with_cube_root_lt_8_l154_154835

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l154_154835


namespace log_base_3_of_9_cubed_l154_154033

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l154_154033


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l154_154436

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l154_154436


namespace max_min_y_l154_154615

def g (t : ℝ) : ℝ := 80 - 2 * t

def f (t : ℝ) : ℝ := 20 - |t - 10|

def y (t : ℝ) : ℝ := g t * f t

theorem max_min_y (t : ℝ) (h : 0 ≤ t ∧ t ≤ 20) :
  (y t = 1200 → t = 10) ∧ (y t = 400 → t = 20) :=
by
  sorry

end max_min_y_l154_154615


namespace max_value_inequality_l154_154202

theorem max_value_inequality (a x₁ x₂ : ℝ) (h_a : a < 0)
  (h_sol : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x₁ < x ∧ x < x₂) :
    x₁ + x₂ + a / (x₁ * x₂) ≤ - 4 * Real.sqrt 3 / 3 := by
  sorry

end max_value_inequality_l154_154202


namespace lap_time_improvement_l154_154935

theorem lap_time_improvement:
  let initial_lap_time := (30 : ℚ) / 15
  let current_lap_time := (33 : ℚ) / 18
  initial_lap_time - current_lap_time = 1 / 6 :=
by
  let initial_lap_time := (30 : ℚ) / 15
  let current_lap_time := (33 : ℚ) / 18
  calc
    initial_lap_time - current_lap_time
        = 2 - 11 / 6 : by rw [initial_lap_time, current_lap_time]
    ... = 1 / 6 : sorry

end lap_time_improvement_l154_154935


namespace derivative_f_l154_154109

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end derivative_f_l154_154109


namespace cd_value_l154_154133

theorem cd_value (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (ab ac bd : ℝ) 
  (h_ab : ab = 2) (h_ac : ac = 5) (h_bd : bd = 6) :
  ∃ (cd : ℝ), cd = 3 :=
by sorry

end cd_value_l154_154133


namespace find_g_2_l154_154561

variable (g : ℝ → ℝ)

-- Function satisfying the given conditions
axiom g_functional : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom g_nonzero : ∀ (x : ℝ), g x ≠ 0

-- The proof statement
theorem find_g_2 : g 2 = 1 := by
  sorry

end find_g_2_l154_154561


namespace length_of_field_l154_154597

variable (w : ℕ)   -- Width of the rectangular field
variable (l : ℕ)   -- Length of the rectangular field
variable (pond_side : ℕ)  -- Side length of the square pond
variable (pond_area field_area : ℕ)  -- Areas of the pond and field
variable (cond1 : l = 2 * w)  -- Condition 1: Length is double the width
variable (cond2 : pond_side = 4)  -- Condition 2: Side of the pond is 4 meters
variable (cond3 : pond_area = pond_side * pond_side)  -- Condition 3: Area of square pond
variable (cond4 : pond_area = (1 / 8) * field_area)  -- Condition 4: Area of pond is 1/8 of the area of the field

theorem length_of_field :
  pond_area = pond_side * pond_side →
  pond_area = (1 / 8) * (l * w) →
  l = 2 * w →
  w = 8 →
  l = 16 :=
by
  intro h1 h2 h3 h4
  sorry

end length_of_field_l154_154597


namespace greatest_possible_median_l154_154134

theorem greatest_possible_median (k m r s t : ℕ) (h_avg : (k + m + r + s + t) / 5 = 10) (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) (h_t : t = 20) : r = 8 :=
by
  sorry

end greatest_possible_median_l154_154134


namespace find_a_for_inequality_l154_154738

theorem find_a_for_inequality (a : ℚ) :
  (∀ x : ℚ, (ax / (x - 1)) < 1 ↔ (x < 1 ∨ x > 2)) → a = 1/2 :=
by
  sorry

end find_a_for_inequality_l154_154738


namespace quadratic_roots_conditions_l154_154490

-- Definitions of the given conditions.
variables (a b c : ℝ)  -- Coefficients of the quadratic trinomial
variable (h : b^2 - 4 * a * c ≥ 0)  -- Given condition that the discriminant is non-negative

-- Statement to prove:
theorem quadratic_roots_conditions (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) :
  ¬(∀ x : ℝ, a^2 * x^2 + b^2 * x + c^2 = 0) ∧ (∀ x : ℝ, a^3 * x^2 + b^3 * x + c^3 = 0 → b^6 - 4 * a^3 * c^3 ≥ 0) :=
by
  sorry

end quadratic_roots_conditions_l154_154490


namespace sqrt_x_minus_2_meaningful_l154_154507

theorem sqrt_x_minus_2_meaningful (x : ℝ) (h : 0 ≤ x - 2) : 2 ≤ x :=
by sorry

end sqrt_x_minus_2_meaningful_l154_154507


namespace total_geese_l154_154910

/-- Definition of the number of geese that remain flying after each lake, 
    based on the given conditions. -/
def geese_after_lake (G : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then G else 2^(n : ℕ) - 1

/-- Main theorem stating the total number of geese in the flock. -/
theorem total_geese (n : ℕ) : ∃ (G : ℕ), geese_after_lake G n = 2^n - 1 :=
by
  sorry

end total_geese_l154_154910


namespace rectangle_width_is_nine_l154_154733

theorem rectangle_width_is_nine (w l : ℝ) (h1 : l = 2 * w)
  (h2 : l * w = 3 * 2 * (l + w)) : 
  w = 9 :=
by
  sorry

end rectangle_width_is_nine_l154_154733


namespace smallest_composite_proof_l154_154650

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l154_154650


namespace linear_eq_value_abs_sum_l154_154824

theorem linear_eq_value_abs_sum (a m : ℤ)
  (h1: m^2 - 9 = 0)
  (h2: m ≠ 3)
  (h3: |a| ≤ 3) : 
  |a + m| + |a - m| = 6 :=
by
  sorry

end linear_eq_value_abs_sum_l154_154824


namespace intersecting_lines_l154_154262

theorem intersecting_lines (c d : ℝ) 
  (h1 : 3 = (1/3 : ℝ) * 0 + c)
  (h2 : 0 = (1/3 : ℝ) * 3 + d) :
  c + d = 2 := 
by {
  sorry
}

end intersecting_lines_l154_154262


namespace periodic_functions_exist_l154_154986

theorem periodic_functions_exist (p1 p2 : ℝ) (h1 : p1 > 0) (h2 : p2 > 0) :
    ∃ (f1 f2 : ℝ → ℝ), (∀ x, f1 (x + p1) = f1 x) ∧ (∀ x, f2 (x + p2) = f2 x) ∧ ∃ T > 0, ∀ x, (f1 - f2) (x + T) = (f1 - f2) x :=
sorry

end periodic_functions_exist_l154_154986


namespace fraction_of_planted_area_l154_154695

-- Definitions of the conditions
def right_triangle (a b : ℕ) : Prop :=
  a * a + b * b = (Int.sqrt (a ^ 2 + b ^ 2))^2

def unplanted_square_distance (dist : ℕ) : Prop :=
  dist = 3

-- The main theorem to be proved
theorem fraction_of_planted_area (a b : ℕ) (dist : ℕ) (h_triangle : right_triangle a b) (h_square_dist : unplanted_square_distance dist) :
  (a = 5) → (b = 12) → ((a * b - dist ^ 2) / (a * b) = 412 / 1000) :=
by
  sorry

end fraction_of_planted_area_l154_154695


namespace log_base_3_of_9_cubed_l154_154046

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l154_154046


namespace probability_sum_7_is_1_over_3_l154_154743

def odd_die : Set ℕ := {1, 3, 5}
def even_die : Set ℕ := {2, 4, 6}

noncomputable def total_outcomes : ℕ := 6 * 6

noncomputable def favorable_outcomes : ℕ := 4 + 4 + 4

noncomputable def probability_sum_7 : ℚ := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_sum_7_is_1_over_3 :
  probability_sum_7 = 1 / 3 :=
by
  sorry

end probability_sum_7_is_1_over_3_l154_154743


namespace smallest_value_x_abs_eq_32_l154_154185

theorem smallest_value_x_abs_eq_32 : ∃ x : ℚ, (x = -29 / 5) ∧ (|5 * x - 3| = 32) ∧ 
  (∀ y : ℚ, (|5 * y - 3| = 32) → (x ≤ y)) :=
by
  sorry

end smallest_value_x_abs_eq_32_l154_154185


namespace integer_solutions_l154_154868

theorem integer_solutions (x y : ℤ) (h₁ : x + y ≠ 0) :
  (x^2 + y^2) / (x + y) = 10 ↔
  (x, y) ∈ {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)} :=
by
  sorry

end integer_solutions_l154_154868


namespace sam_fish_count_l154_154871

/-- Let S be the number of fish Sam has. -/
def num_fish_sam : ℕ := sorry

/-- Joe has 8 times as many fish as Sam, which gives 8S fish. -/
def num_fish_joe (S : ℕ) : ℕ := 8 * S

/-- Harry has 4 times as many fish as Joe, hence 32S fish. -/
def num_fish_harry (S : ℕ) : ℕ := 32 * S

/-- Harry has 224 fish. -/
def harry_fish : ℕ := 224

/-- Prove that Sam has 7 fish given the conditions above. -/
theorem sam_fish_count : num_fish_harry num_fish_sam = harry_fish → num_fish_sam = 7 := by
  sorry

end sam_fish_count_l154_154871


namespace sufficient_but_not_necessary_condition_for_q_l154_154958

variable (p q r : Prop)

theorem sufficient_but_not_necessary_condition_for_q (hp : p → r) (hq1 : r → q) (hq2 : ¬(q → r)) : 
  (p → q) ∧ ¬(q → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_q_l154_154958


namespace log_base_3_of_9_cubed_l154_154045

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l154_154045


namespace red_pairs_l154_154923

theorem red_pairs (total_students green_students red_students total_pairs green_pairs : ℕ) 
  (h1 : total_students = green_students + red_students)
  (h2 : green_students = 67)
  (h3 : red_students = 89)
  (h4 : total_pairs = 78)
  (h5 : green_pairs = 25)
  (h6 : 2 * green_pairs ≤ green_students ∧ 2 * green_pairs ≤ red_students ∧ 2 * green_pairs ≤ 2 * total_pairs) :
  ∃ red_pairs : ℕ, red_pairs = 36 := by
    sorry

end red_pairs_l154_154923


namespace opposite_of_minus_one_third_l154_154568

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_minus_one_third_l154_154568


namespace problem_HMMT_before_HMT_l154_154148
noncomputable def probability_of_sequence (seq: List Char) : ℚ := sorry
def probability_H : ℚ := 1 / 3
def probability_M : ℚ := 1 / 3
def probability_T : ℚ := 1 / 3

theorem problem_HMMT_before_HMT : probability_of_sequence ['H', 'M', 'M', 'T'] = 1 / 4 :=
sorry

end problem_HMMT_before_HMT_l154_154148


namespace find_x_squared_plus_inv_squared_l154_154286

theorem find_x_squared_plus_inv_squared (x : ℝ) (hx : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := 
by
sorry

end find_x_squared_plus_inv_squared_l154_154286


namespace sum_of_fractions_l154_154587

theorem sum_of_fractions : 
  (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := 
by
  sorry

end sum_of_fractions_l154_154587


namespace box_dimensions_sum_l154_154777

theorem box_dimensions_sum (A B C : ℝ)
  (h1 : A * B = 18)
  (h2 : A * C = 32)
  (h3 : B * C = 50) :
  A + B + C = 57.28 := 
sorry

end box_dimensions_sum_l154_154777


namespace sum_reciprocal_squares_l154_154302

open Real

theorem sum_reciprocal_squares (a : ℝ) (A B C D E F : ℝ)
    (square_ABCD : A = 0 ∧ B = a ∧ D = a ∧ C = a)
    (line_intersects : A = 0 ∧ E ≥ 0 ∧ E ≤ a ∧ F ≥ 0 ∧ F ≤ a) 
    (phi : ℝ) : 
    (cos phi * (a/cos phi))^2 + (sin phi * (a/sin phi))^2 = (1/a^2) := 
sorry 

end sum_reciprocal_squares_l154_154302


namespace boat_speed_in_still_water_l154_154753

theorem boat_speed_in_still_water:
  ∀ (V_b : ℝ) (V_s : ℝ) (D : ℝ),
    V_s = 3 → 
    (D = (V_b + V_s) * 1) → 
    (D = (V_b - V_s) * 1.5) → 
    V_b = 15 :=
by
  intros V_b V_s D V_s_eq H_downstream H_upstream
  sorry

end boat_speed_in_still_water_l154_154753


namespace min_points_to_guarantee_victory_l154_154521

noncomputable def points_distribution (pos : ℕ) : ℕ :=
  match pos with
  | 1 => 7
  | 2 => 4
  | 3 => 2
  | _ => 0

def max_points_per_race : ℕ := 7
def num_races : ℕ := 3

theorem min_points_to_guarantee_victory : ∃ min_points, min_points = 18 ∧ 
  (∀ other_points, other_points < 18) := 
by {
  sorry
}

end min_points_to_guarantee_victory_l154_154521


namespace min_abs_sum_l154_154884

theorem min_abs_sum (x y z : ℝ) (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 4) 
  (hy_eq : y^2 = x^2 + 2) (hz_eq : z^2 = y^2 + 2) : 
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_abs_sum_l154_154884


namespace smallest_three_digit_divisible_by_3_and_6_l154_154895

theorem smallest_three_digit_divisible_by_3_and_6 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999 ∧ n % 3 = 0 ∧ n % 6 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 3 = 0 ∧ m % 6 = 0 → n ≤ m) ∧ n = 102 := 
by {sorry}

end smallest_three_digit_divisible_by_3_and_6_l154_154895


namespace line_perpendicular_to_two_planes_parallel_l154_154945

-- Declare lines and planes
variables {Line Plane : Type}

-- Define the perpendicular and parallel relationships
variables (perpendicular : Line → Plane → Prop)
variables (parallel : Plane → Plane → Prop)

-- Given conditions
variables (m n : Line) (α β : Plane)
-- The known conditions are:
-- 1. m is perpendicular to α
-- 2. m is perpendicular to β
-- We want to prove:
-- 3. α is parallel to β

theorem line_perpendicular_to_two_planes_parallel (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β :=
sorry

end line_perpendicular_to_two_planes_parallel_l154_154945


namespace nine_consecutive_arithmetic_mean_divisible_1111_l154_154392

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l154_154392


namespace sequence_an_formula_l154_154853

theorem sequence_an_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 :=
by
  sorry

end sequence_an_formula_l154_154853


namespace intersecting_graphs_value_l154_154449

theorem intersecting_graphs_value (a b c d : ℝ) 
  (h1 : 5 = -|2 - a| + b) 
  (h2 : 3 = -|8 - a| + b) 
  (h3 : 5 = |2 - c| + d) 
  (h4 : 3 = |8 - c| + d) : 
  a + c = 10 :=
sorry

end intersecting_graphs_value_l154_154449


namespace speed_of_ship_with_two_sails_l154_154013

noncomputable def nautical_mile : ℝ := 1.15
noncomputable def land_miles_traveled : ℝ := 345
noncomputable def time_with_one_sail : ℝ := 4
noncomputable def time_with_two_sails : ℝ := 4
noncomputable def speed_with_one_sail : ℝ := 25

theorem speed_of_ship_with_two_sails :
  ∃ S : ℝ, 
    (S * time_with_two_sails + speed_with_one_sail * time_with_one_sail = land_miles_traveled / nautical_mile) → 
    S = 50  :=
by
  sorry

end speed_of_ship_with_two_sails_l154_154013


namespace hoot_difference_l154_154716

def owl_hoot_rate : ℕ := 5
def heard_hoots_per_min : ℕ := 20
def owls_count : ℕ := 3

theorem hoot_difference :
  heard_hoots_per_min - (owls_count * owl_hoot_rate) = 5 := by
  sorry

end hoot_difference_l154_154716


namespace log_base_3_of_9_cubed_l154_154024

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l154_154024


namespace eccentricity_range_of_isosceles_right_triangle_l154_154350

theorem eccentricity_range_of_isosceles_right_triangle
  (a : ℝ) (e : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + y^2 = 1)
  (h_a_gt_1 : a > 1)
  (B C : ℝ × ℝ)
  (isosceles_right_triangle : ∀ (A B C : ℝ × ℝ), ∃ k : ℝ, k > 0 ∧ 
    B = (-(2*k*a^2)/(1 + a^2*k^2), 0) ∧ 
    C = ((2*k*a^2)/(a^2 + k^2), 0) ∧ 
    (B.1^2 + B.2^2 = C.1^2 + C.2^2 + 1))
  (unique_solution : ∀ (k : ℝ), ∃! k', k' = 1)
  : 0 < e ∧ e ≤ (Real.sqrt 6) / 3 :=
sorry

end eccentricity_range_of_isosceles_right_triangle_l154_154350


namespace shepherd_boys_equation_l154_154556

theorem shepherd_boys_equation (x : ℕ) :
  6 * x + 14 = 8 * x - 2 :=
by sorry

end shepherd_boys_equation_l154_154556


namespace sqrt_meaningful_implies_x_ge_2_l154_154509

theorem sqrt_meaningful_implies_x_ge_2 (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 := 
sorry

end sqrt_meaningful_implies_x_ge_2_l154_154509


namespace cos_F_l154_154697

theorem cos_F (DE EF : ℝ) (h1 : DE = 21) (h2 : EF = 28) (h3 : ∠D = 90): cos (F : ℝ) = 4 / 5 := by
  sorry

end cos_F_l154_154697


namespace inequality_proof_l154_154947

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end inequality_proof_l154_154947


namespace solve_system1_solve_system2_l154_154722

theorem solve_system1 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : x = 2 * y + 1) : x = 3 ∧ y = 1 := 
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x - y = 6) (h2 : 3 * x + 2 * y = 2) : x = 2 ∧ y = -2 := 
by sorry

end solve_system1_solve_system2_l154_154722


namespace min_dist_tangent_to_circle_l154_154933

/-!
# Tangent to a Circle Problem
-/ 

open Real
open Function
open Set

variables {P M : EuclideanSpace ℝ (Fin 2)} {O : EuclideanSpace ℝ (Fin 2)} 

theorem min_dist_tangent_to_circle (h : dist (P, O) = dist (P, M)) (circle_eq : ∀ (P : EuclideanSpace ℝ (Fin 2)), (∥P + ![1, -2]∥ ^ 2 = 1)) : 
  dist (P, O) = 2 * sqrt 5 / 5 :=
by
  -- Proof skipped
  sorry

end min_dist_tangent_to_circle_l154_154933


namespace positive_whole_numbers_with_cube_root_less_than_8_l154_154837

theorem positive_whole_numbers_with_cube_root_less_than_8 :
  { n : ℕ | n > 0 ∧ n < 512 }.card = 511 :=
by
  sorry

end positive_whole_numbers_with_cube_root_less_than_8_l154_154837


namespace problem_statement_l154_154211

-- Define the statement for positive integers m and n
def div_equiv (m n : ℕ) : Prop :=
  19 ∣ (11 * m + 2 * n) ↔ 19 ∣ (18 * m + 5 * n)

-- The final theorem statement
theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : div_equiv m n :=
by
  sorry

end problem_statement_l154_154211


namespace nine_consecutive_arithmetic_mean_divisible_1111_l154_154391

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l154_154391


namespace time_to_empty_tank_by_leakage_l154_154512

theorem time_to_empty_tank_by_leakage (R_t R_l : ℝ) (h1 : R_t = 1 / 12) (h2 : R_t - R_l = 1 / 18) :
  (1 / R_l) = 36 :=
by
  sorry

end time_to_empty_tank_by_leakage_l154_154512


namespace trigonometric_identity_l154_154812

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π + α) = -1/3) : Real.sin (2 * α) / Real.cos α = 2 / 3 := by
  sorry

end trigonometric_identity_l154_154812


namespace two_digit_sum_condition_l154_154939

theorem two_digit_sum_condition (x y : ℕ) (hx : 1 ≤ x) (hx9 : x ≤ 9) (hy : 0 ≤ y) (hy9 : y ≤ 9)
    (h : (x + 1) + (y + 2) - 10 = 2 * (x + y)) :
    (x = 6 ∧ y = 8) ∨ (x = 5 ∧ y = 9) :=
sorry

end two_digit_sum_condition_l154_154939


namespace percentage_of_part_over_whole_l154_154137

theorem percentage_of_part_over_whole (Part Whole : ℕ) (h1 : Part = 120) (h2 : Whole = 50) :
  (Part / Whole : ℚ) * 100 = 240 := by
  sorry

end percentage_of_part_over_whole_l154_154137


namespace probability_of_drawing_three_white_marbles_l154_154914

noncomputable def probability_of_three_white_marbles : ℚ :=
  let total_marbles := 5 + 7 + 15
  let prob_first_white := 15 / total_marbles
  let prob_second_white := 14 / (total_marbles - 1)
  let prob_third_white := 13 / (total_marbles - 2)
  prob_first_white * prob_second_white * prob_third_white

theorem probability_of_drawing_three_white_marbles :
  probability_of_three_white_marbles = 2 / 13 := 
by 
  sorry

end probability_of_drawing_three_white_marbles_l154_154914


namespace teacher_discount_l154_154526

-- Definitions that capture the conditions in Lean
def num_students : ℕ := 30
def num_pens_per_student : ℕ := 5
def num_notebooks_per_student : ℕ := 3
def num_binders_per_student : ℕ := 1
def num_highlighters_per_student : ℕ := 2
def cost_per_pen : ℚ := 0.50
def cost_per_notebook : ℚ := 1.25
def cost_per_binder : ℚ := 4.25
def cost_per_highlighter : ℚ := 0.75
def amount_spent : ℚ := 260

-- Compute the total cost without discount
def total_cost : ℚ :=
  (num_students * num_pens_per_student) * cost_per_pen +
  (num_students * num_notebooks_per_student) * cost_per_notebook +
  (num_students * num_binders_per_student) * cost_per_binder +
  (num_students * num_highlighters_per_student) * cost_per_highlighter

-- The main theorem to prove the applied teacher discount
theorem teacher_discount :
  total_cost - amount_spent = 100 := by
  sorry

end teacher_discount_l154_154526


namespace sum_abc_l154_154448

theorem sum_abc (a b c: ℝ) 
  (h1 : ∃ x: ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + b * x + c = 0)
  (h2 : ∃ x: ℝ, x^2 + x + a = 0 ∧ x^2 + c * x + b = 0) :
  a + b + c = -3 := 
sorry

end sum_abc_l154_154448


namespace compute_xy_l154_154427

variable (x y : ℝ)

-- Conditions from the problem
def condition1 : Prop := x + y = 10
def condition2 : Prop := x^3 + y^3 = 172

-- Theorem statement to prove the answer
theorem compute_xy (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 41.4 :=
sorry

end compute_xy_l154_154427


namespace gcd_294_84_l154_154892

-- Define the numbers for the GCD calculation
def a : ℕ := 294
def b : ℕ := 84

-- Define the greatest common divisor function using Euclidean algorithm
def gcd_euclidean : ℕ → ℕ → ℕ
| x, 0 => x
| x, y => gcd_euclidean y (x % y)

-- Theorem stating that the GCD of 294 and 84 is 42
theorem gcd_294_84 : gcd_euclidean a b = 42 :=
by
  -- Proof is omitted
  sorry

end gcd_294_84_l154_154892


namespace cistern_depth_l154_154451

theorem cistern_depth (h : ℝ) :
  (6 * 4 + 2 * (h * 6) + 2 * (h * 4) = 49) → (h = 1.25) :=
by
  sorry

end cistern_depth_l154_154451


namespace problem_l154_154706

theorem problem (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, f (x + 2) = -f x) :
  f 4 = 0 ∧ (∀ x, f (x + 4) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) :=
sorry

end problem_l154_154706


namespace atomic_weight_Ba_l154_154634

-- Definitions for conditions
def atomic_weight_O : ℕ := 16
def molecular_weight_compound : ℕ := 153

-- Theorem statement
theorem atomic_weight_Ba : ∃ bw, molecular_weight_compound = bw + atomic_weight_O ∧ bw = 137 :=
by {
  -- Skip the proof
  sorry
}

end atomic_weight_Ba_l154_154634


namespace Joe_total_time_correct_l154_154983

theorem Joe_total_time_correct :
  ∀ (distance : ℝ) (walk_rate : ℝ) (bike_rate : ℝ) (walk_time bike_time : ℝ),
    (walk_time = 9) →
    (bike_rate = 5 * walk_rate) →
    (walk_rate * walk_time = distance / 3) →
    (bike_rate * bike_time = 2 * distance / 3) →
    (walk_time + bike_time = 12.6) := 
by
  intros distance walk_rate bike_rate walk_time bike_time
  intro walk_time_cond
  intro bike_rate_cond
  intro walk_distance_cond
  intro bike_distance_cond
  sorry

end Joe_total_time_correct_l154_154983


namespace lines_are_skew_l154_154175

def line1 (a t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 1 + 4 * t, a + 5 * t)
  
def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 3 + 3 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) : (∀ t u : ℝ, line1 a t ≠ line2 u) ↔ a ≠ -4/5 :=
sorry

end lines_are_skew_l154_154175


namespace customers_who_bought_four_paintings_each_l154_154122

/-- Tracy's art fair conditions:
- 20 people came to look at the art
- Four customers bought two paintings each
- Twelve customers bought one painting each
- Tracy sold a total of 36 paintings

We need to prove the number of customers who bought four paintings each. -/
theorem customers_who_bought_four_paintings_each:
  let total_customers := 20
  let customers_bought_two_paintings := 4
  let customers_bought_one_painting := 12
  let total_paintings_sold := 36
  let paintings_per_customer_buying_two := 2
  let paintings_per_customer_buying_one := 1
  let paintings_per_customer_buying_four := 4
  (customers_bought_two_paintings * paintings_per_customer_buying_two +
   customers_bought_one_painting * paintings_per_customer_buying_one +
   x * paintings_per_customer_buying_four = total_paintings_sold) →
  (customers_bought_two_paintings + customers_bought_one_painting + x = total_customers) →
  x = 4 :=
by
  intro h1 h2
  sorry

end customers_who_bought_four_paintings_each_l154_154122


namespace strawberry_picking_l154_154588

theorem strawberry_picking 
  (e : ℕ) (n : ℕ) (p : ℕ) (A : ℕ) (w : ℕ) 
  (h1 : e = 4) 
  (h2 : n = 3) 
  (h3 : p = 20) 
  (h4 : A = 128) 
  : w = 7 :=
by 
  -- proof steps to be filled in
  sorry

end strawberry_picking_l154_154588


namespace christine_wander_time_l154_154316

-- Definitions based on conditions
def distance : ℝ := 50.0
def speed : ℝ := 6.0

-- The statement to prove
theorem christine_wander_time : (distance / speed) = 8 + 20/60 :=
by
  sorry

end christine_wander_time_l154_154316


namespace smallest_composite_proof_l154_154653

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l154_154653


namespace quadratic_even_coeff_l154_154123

theorem quadratic_even_coeff (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∃ r s : ℚ, r * s + b * r + c = 0) : (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0) := by
  sorry

end quadratic_even_coeff_l154_154123


namespace a_5_eq_neg1_l154_154498

-- Given conditions
def S (n : ℕ) : ℤ := n^2 - 10 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem a_5_eq_neg1 : a 5 = -1 :=
by sorry

end a_5_eq_neg1_l154_154498


namespace find_lengths_of_DE_and_HJ_l154_154368

noncomputable def lengths_consecutive_segments (BD DE EF FG GH HJ : ℝ) (BC : ℝ) : Prop :=
  BD = 5 ∧ EF = 11 ∧ FG = 7 ∧ GH = 3 ∧ BC = 29 ∧ BD + DE + EF + FG + GH + HJ = BC ∧ DE = HJ

theorem find_lengths_of_DE_and_HJ (x : ℝ) : lengths_consecutive_segments 5 x 11 7 3 x 29 → x = 1.5 :=
by
  intros h
  sorry

end find_lengths_of_DE_and_HJ_l154_154368


namespace original_cost_of_horse_l154_154774

theorem original_cost_of_horse (x : ℝ) (h : x - x^2 / 100 = 24) : x = 40 ∨ x = 60 := 
by 
  sorry

end original_cost_of_horse_l154_154774


namespace solve_system_of_equations_solve_algebraic_equation_l154_154105

-- Problem 1: System of Equations
theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 3) (h2 : 2 * x - y = 1) : x = 1 ∧ y = 1 :=
sorry

-- Problem 2: Algebraic Equation
theorem solve_algebraic_equation (x : ℝ) (h : 1 / (x - 1) + 2 = 5 / (1 - x)) : x = -2 :=
sorry

end solve_system_of_equations_solve_algebraic_equation_l154_154105


namespace father_walk_time_l154_154282

-- Xiaoming's cycling speed is 4 times his father's walking speed.
-- Xiaoming continues for another 18 minutes to reach B after meeting his father.
-- Prove that Xiaoming's father needs 288 minutes to walk from the meeting point to A.
theorem father_walk_time {V : ℝ} (h₁ : V > 0) (h₂ : ∀ t : ℝ, t > 0 → 18 * V = (V / 4) * t) :
  288 = 4 * 72 :=
by
  sorry

end father_walk_time_l154_154282


namespace cost_price_approx_l154_154006

noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

theorem cost_price_approx :
  ∀ (selling_price profit_percent : ℝ),
  selling_price = 2552.36 →
  profit_percent = 6 →
  abs (cost_price selling_price profit_percent - 2407.70) < 0.01 :=
by
  intros selling_price profit_percent h1 h2
  sorry

end cost_price_approx_l154_154006


namespace lottery_profit_l154_154776

-- Definitions

def Prob_A := (1:ℚ) / 5
def Prob_B := (4:ℚ) / 15
def Prob_C := (1:ℚ) / 5
def Prob_D := (2:ℚ) / 15
def Prob_E := (1:ℚ) / 5

def customers := 300

def first_prize_value := 9
def second_prize_value := 3
def third_prize_value := 1

-- Proof Problem Statement

theorem lottery_profit : 
  (first_prize_category == "D") ∧ 
  (second_prize_category == "B") ∧ 
  (300 * 3 - ((300 * Prob_D) * 9 + (300 * Prob_B) * 3 + (300 * (Prob_A + Prob_C + Prob_E)) * 1)) == 120 :=
by 
  -- Insert mathematical proof here using given probabilities and conditions
  sorry

end lottery_profit_l154_154776


namespace fish_caught_l154_154901

theorem fish_caught (x y : ℕ) 
  (h1 : y - 2 = 4 * (x + 2))
  (h2 : y - 6 = 2 * (x + 6)) :
  x = 4 ∧ y = 26 :=
by
  sorry

end fish_caught_l154_154901


namespace opposite_of_minus_one_third_l154_154566

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_minus_one_third_l154_154566


namespace greatest_possible_x_for_equation_l154_154894

theorem greatest_possible_x_for_equation :
  ∃ x, (x = (9 : ℝ) / 5) ∧ 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 := by
  sorry

end greatest_possible_x_for_equation_l154_154894


namespace h_at_3_l154_154474

theorem h_at_3 :
  ∃ h : ℤ → ℤ,
    (∀ x, (x^7 - 1) * h x = (x+1) * (x^2 + 1) * (x^4 + 1) - (x-1)) →
    h 3 = 3 := 
sorry

end h_at_3_l154_154474


namespace total_red_cards_l154_154005

theorem total_red_cards (num_standard_decks : ℕ) (num_special_decks : ℕ)
  (red_standard_deck : ℕ) (additional_red_special_deck : ℕ)
  (total_decks : ℕ) (h1 : num_standard_decks = 5)
  (h2 : num_special_decks = 10)
  (h3 : red_standard_deck = 26)
  (h4 : additional_red_special_deck = 4)
  (h5 : total_decks = num_standard_decks + num_special_decks) :
  num_standard_decks * red_standard_deck +
  num_special_decks * (red_standard_deck + additional_red_special_deck) = 430 := by
  -- Proof is omitted.
  sorry

end total_red_cards_l154_154005


namespace bottle_caps_shared_l154_154999

theorem bottle_caps_shared (initial_bottle_caps : ℕ) (remaining_bottle_caps : ℕ) : 
  initial_bottle_caps = 51 → remaining_bottle_caps = 15 → initial_bottle_caps - remaining_bottle_caps = 36 :=
by
  intros h1 h2
  rw h1
  rw h2
  simp

end bottle_caps_shared_l154_154999


namespace not_all_odd_l154_154748

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def divides (a b c d : ℕ) : Prop := a = b * c + d ∧ 0 ≤ d ∧ d < b

theorem not_all_odd (a b c d : ℕ) 
  (h_div : divides a b c d)
  (h_odd_a : is_odd a)
  (h_odd_b : is_odd b)
  (h_odd_c : is_odd c)
  (h_odd_d : is_odd d) :
  False :=
sorry

end not_all_odd_l154_154748


namespace minimum_hexagon_perimeter_l154_154740

-- Define the conditions given in the problem
def small_equilateral_triangle (side_length : ℝ) (triangle_count : ℕ) :=
  triangle_count = 57 ∧ side_length = 1

def hexagon_with_conditions (angle_condition : ℝ → Prop) :=
  ∀ θ, angle_condition θ → θ ≤ 180 ∧ θ > 0

-- State the main problem as a theorem
theorem minimum_hexagon_perimeter : ∀ n : ℕ, ∃ p : ℕ,
  (small_equilateral_triangle 1 57) → 
  (∃ angle_condition, hexagon_with_conditions angle_condition) →
  (n = 57) →
  p = 19 :=
by
  sorry

end minimum_hexagon_perimeter_l154_154740


namespace lucca_bread_fraction_l154_154535

theorem lucca_bread_fraction 
  (total_bread : ℕ)
  (initial_fraction_eaten : ℚ)
  (final_pieces : ℕ)
  (bread_first_day : ℚ)
  (bread_second_day : ℚ)
  (bread_third_day : ℚ)
  (remaining_pieces_after_first_day : ℕ)
  (remaining_pieces_after_second_day : ℕ)
  (remaining_pieces_after_third_day : ℕ) :
  total_bread = 200 →
  initial_fraction_eaten = 1/4 →
  bread_first_day = initial_fraction_eaten * total_bread →
  remaining_pieces_after_first_day = total_bread - bread_first_day →
  bread_second_day = (remaining_pieces_after_first_day * bread_second_day) →
  remaining_pieces_after_second_day = remaining_pieces_after_first_day - bread_second_day →
  bread_third_day = 1/2 * remaining_pieces_after_second_day →
  remaining_pieces_after_third_day = remaining_pieces_after_second_day - bread_third_day →
  remaining_pieces_after_third_day = 45 →
  bread_second_day = 2/5 :=
by
  sorry

end lucca_bread_fraction_l154_154535


namespace find_C_plus_D_l154_154672

theorem find_C_plus_D (C D : ℝ) (h : ∀ x : ℝ, (Cx - 20) / (x^2 - 3 * x - 10) = D / (x + 2) + 4 / (x - 5)) :
  C + D = 4.7 :=
sorry

end find_C_plus_D_l154_154672


namespace point_P_in_Quadrant_II_l154_154959

noncomputable def α : ℝ := (5 * Real.pi) / 8

theorem point_P_in_Quadrant_II : (Real.sin α > 0) ∧ (Real.tan α < 0) := sorry

end point_P_in_Quadrant_II_l154_154959


namespace intercepts_of_line_l154_154331

theorem intercepts_of_line (x y : ℝ) 
  (h : 2 * x + 7 * y = 35) :
  (y = 5 → x = 0) ∧ (x = 17.5 → y = 0)  :=
by
  sorry

end intercepts_of_line_l154_154331


namespace log_base_3_of_9_cubed_l154_154020
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l154_154020


namespace quadratic_max_m_l154_154954

theorem quadratic_max_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (m * x^2 - 2 * m * x + 2) ≤ 4) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ (m * x^2 - 2 * m * x + 2) = 4) ∧ 
  m ≠ 0 → 
  (m = 2 / 3 ∨ m = -2) := 
by
  sorry

end quadratic_max_m_l154_154954


namespace pos_int_solutions_l154_154115

-- defining the condition for a positive integer solution to the equation
def is_pos_int_solution (x y : Int) : Prop :=
  5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0

-- stating the theorem for positive integer solutions of the equation
theorem pos_int_solutions : 
  ∃ x y : Int, is_pos_int_solution x y ∧ ((x = 1 ∧ y = 10) ∨ (x = 3 ∧ y = 5)) :=
by
  sorry

end pos_int_solutions_l154_154115


namespace minimum_toothpicks_for_5_squares_l154_154244

theorem minimum_toothpicks_for_5_squares :
  let single_square_toothpicks := 4
  let additional_shared_side_toothpicks := 3
  ∃ n, n = single_square_toothpicks + 4 * additional_shared_side_toothpicks ∧ n = 15 :=
by
  sorry

end minimum_toothpicks_for_5_squares_l154_154244


namespace smallest_value_of_c_l154_154734

/-- The polynomial x^3 - cx^2 + dx - 2550 has three positive integer roots,
    and the product of the roots is 2550. Prove that the smallest possible value of c is 42. -/
theorem smallest_value_of_c :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2550 ∧ c = a + b + c) → c = 42 :=
sorry

end smallest_value_of_c_l154_154734


namespace reciprocal_of_neg_seven_l154_154737

theorem reciprocal_of_neg_seven : (1 : ℚ) / (-7 : ℚ) = -1 / 7 :=
by
  sorry

end reciprocal_of_neg_seven_l154_154737


namespace infinite_divisibility_of_2n_plus_n2_by_100_l154_154098

theorem infinite_divisibility_of_2n_plus_n2_by_100 :
  ∃ᶠ n in at_top, 100 ∣ (2^n + n^2) :=
sorry

end infinite_divisibility_of_2n_plus_n2_by_100_l154_154098


namespace find_n_l154_154215

theorem find_n (n : ℤ) 
  (h : (3 + 16 + 33 + (n + 1)) / 4 = 20) : n = 27 := 
by
  sorry

end find_n_l154_154215


namespace expression_evaluation_l154_154622

-- Define the numbers and operations
def expr : ℚ := 10 * (1 / 2) * 3 / (1 / 6)

-- Formalize the proof problem
theorem expression_evaluation : expr = 90 := 
by 
  -- Start the proof, which is not required according to the instruction, so we replace it with 'sorry'
  sorry

end expression_evaluation_l154_154622


namespace students_at_end_of_year_l154_154228

-- Define the initial number of students
def initial_students : Nat := 10

-- Define the number of students who left during the year
def students_left : Nat := 4

-- Define the number of new students who arrived during the year
def new_students : Nat := 42

-- Proof problem: the number of students at the end of the year
theorem students_at_end_of_year : initial_students - students_left + new_students = 48 := by
  sorry

end students_at_end_of_year_l154_154228


namespace workers_appointment_l154_154424

theorem workers_appointment (F T V : ℕ) (hF : F = 5) (hT : T = 4) (hV : V = 2) : 
  let ways := (choose 5 4) * (choose 4 3) * (choose 2 1) +
              (choose 5 3) * (choose 4 4) * (choose 2 1) +
              (choose 5 4) * (choose 4 2) * (choose 2 2) +
              (choose 5 3) * (choose 4 3) * (choose 2 2) +
              (choose 5 3) * (choose 4 2) * (choose 2 2) in
  ways = 190 :=
by
  intros
  have h1 : (choose 5 4) * (choose 4 3) * (choose 2 1) = 40 := by sorry
  have h2 : (choose 5 3) * (choose 4 4) * (choose 2 1) = 20 := by sorry
  have h3 : (choose 5 4) * (choose 4 2) * (choose 2 2) = 30 := by sorry
  have h4 : (choose 5 3) * (choose 4 3) * (choose 2 2) = 40 := by sorry
  have h5 : (choose 5 3) * (choose 4 2) * (choose 2 2) = 60 := by sorry
  let ways := 40 + 20 + 30 + 40 + 60
  show ways = 190, by sorry

end workers_appointment_l154_154424


namespace sum_of_eight_numbers_l154_154216

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l154_154216


namespace g_neither_even_nor_odd_l154_154854

noncomputable def g (x : ℝ) : ℝ := 3 ^ (x^2 - 3) - |x| + Real.sin x

theorem g_neither_even_nor_odd : ∀ x : ℝ, g x ≠ g (-x) ∧ g x ≠ -g (-x) := 
by
  intro x
  sorry

end g_neither_even_nor_odd_l154_154854


namespace a_plus_b_values_l154_154057

theorem a_plus_b_values (a b : ℝ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a + b = -2 ∨ a + b = -8 :=
sorry

end a_plus_b_values_l154_154057


namespace can_spend_all_money_l154_154749

theorem can_spend_all_money (n : Nat) (h : n > 7) : 
  ∃ (x y : Nat), 3 * x + 5 * y = n :=
by
  sorry

end can_spend_all_money_l154_154749


namespace opposite_of_neg_one_third_l154_154579

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end opposite_of_neg_one_third_l154_154579


namespace problem_statement_l154_154049

open Nat

theorem problem_statement (k : ℕ) (hk : k > 0) : 
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * (factorial (k*n) / factorial n) ≤ (factorial (k*n) / factorial n))
  ↔ ∃ a : ℕ, k = 2^a := 
sorry

end problem_statement_l154_154049


namespace squares_circles_intersections_l154_154929

noncomputable def number_of_intersections (p1 p2 : (ℤ × ℤ)) (square_side : ℚ) (circle_radius : ℚ) : ℕ :=
sorry -- function definition placeholder

theorem squares_circles_intersections :
  let p1 := (0, 0)
  let p2 := (1009, 437)
  let square_side := (1 : ℚ) / 4
  let circle_radius := (1 : ℚ) / 8
  (number_of_intersections p1 p2 square_side circle_radius) = 526 := by
  sorry

end squares_circles_intersections_l154_154929


namespace sqrt_expression_simplification_l154_154928

theorem sqrt_expression_simplification :
  (Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 - |2 - Real.sqrt 6|) = 2 :=
by
  sorry

end sqrt_expression_simplification_l154_154928


namespace product_even_permutation_l154_154864

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end product_even_permutation_l154_154864


namespace min_a_3b_l154_154860

theorem min_a_3b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) : 
  a + 3*b ≥ 12 + 16*Real.sqrt 3 :=
by sorry

end min_a_3b_l154_154860


namespace lifting_to_bodyweight_ratio_l154_154161

variable (t : ℕ) (w : ℕ) (p : ℕ) (delta_w : ℕ)

def lifting_total_after_increase (t : ℕ) (p : ℕ) : ℕ :=
  t + (t * p / 100)

def bodyweight_after_increase (w : ℕ) (delta_w : ℕ) : ℕ :=
  w + delta_w

theorem lifting_to_bodyweight_ratio (h_t : t = 2200) (h_w : w = 245) (h_p : p = 15) (h_delta_w : delta_w = 8) :
  lifting_total_after_increase t p / bodyweight_after_increase w delta_w = 10 :=
  by
    -- Use the given conditions
    rw [h_t, h_w, h_p, h_delta_w]
    -- Calculation steps are omitted, directly providing the final assertion
    sorry

end lifting_to_bodyweight_ratio_l154_154161


namespace unique_peg_placement_l154_154714

theorem unique_peg_placement :
  ∃! f : Fin 6 → Fin 6 → Option (Fin 6), ∀ i j k, 
    (∃ c, f i k = some c) →
    (∃ c, f j k = some c) →
    i = j ∧ match f i j with
    | some c => f j k ≠ some c
    | none => True :=
  sorry

end unique_peg_placement_l154_154714


namespace opposite_of_neg_one_third_l154_154578

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end opposite_of_neg_one_third_l154_154578


namespace loads_of_laundry_l154_154291

theorem loads_of_laundry (families : ℕ) (days : ℕ) (adults_per_family : ℕ) (children_per_family : ℕ)
  (adult_towels_per_day : ℕ) (child_towels_per_day : ℕ) (initial_capacity : ℕ) (reduced_capacity : ℕ)
  (initial_days : ℕ) (remaining_days : ℕ) : 
  families = 7 → days = 12 → adults_per_family = 2 → children_per_family = 4 → 
  adult_towels_per_day = 2 → child_towels_per_day = 1 → initial_capacity = 8 → 
  reduced_capacity = 6 → initial_days = 6 → remaining_days = 6 → 
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * initial_days / initial_capacity) +
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * remaining_days / reduced_capacity) = 98 :=
by 
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end loads_of_laundry_l154_154291


namespace probability_blackboard_empty_k_l154_154924

-- Define the conditions for the problem
def Ben_blackboard_empty_probability (n : ℕ) : ℚ :=
  if h : n = 2013 then (2 * (2013 / 3) + 1) / 2^(2013 / 3 * 2) else 0 / 1

-- Define the theorem that Ben's blackboard is empty after 2013 flips, and determine k
theorem probability_blackboard_empty_k :
  ∃ (u v k : ℕ), Ben_blackboard_empty_probability 2013 = (2 * u + 1) / (2^k * (2 * v + 1)) ∧ k = 1336 :=
by sorry

end probability_blackboard_empty_k_l154_154924


namespace sum_of_squares_of_roots_of_quadratic_l154_154805

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end sum_of_squares_of_roots_of_quadratic_l154_154805


namespace largest_percentage_increase_is_2013_to_2014_l154_154694

-- Defining the number of students in each year as constants
def students_2010 : ℕ := 50
def students_2011 : ℕ := 56
def students_2012 : ℕ := 62
def students_2013 : ℕ := 68
def students_2014 : ℕ := 77
def students_2015 : ℕ := 81

-- Defining the percentage increase between consecutive years
def percentage_increase (a b : ℕ) : ℚ := ((b - a) : ℚ) / (a : ℚ)

-- Calculating all the percentage increases
def pi_2010_2011 := percentage_increase students_2010 students_2011
def pi_2011_2012 := percentage_increase students_2011 students_2012
def pi_2012_2013 := percentage_increase students_2012 students_2013
def pi_2013_2014 := percentage_increase students_2013 students_2014
def pi_2014_2015 := percentage_increase students_2014 students_2015

-- The theorem stating the largest percentage increase is between 2013 and 2014
theorem largest_percentage_increase_is_2013_to_2014 :
  max (pi_2010_2011) (max (pi_2011_2012) (max (pi_2012_2013) (max (pi_2013_2014) (pi_2014_2015)))) = pi_2013_2014 :=
sorry

end largest_percentage_increase_is_2013_to_2014_l154_154694


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154415

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154415


namespace magazines_in_third_pile_l154_154590

-- Define the number of magazines in each pile.
def pile1 := 3
def pile2 := 4
def pile4 := 9
def pile5 := 13

-- Define the differences between the piles.
def diff2_1 := pile2 - pile1  -- Difference between second and first pile
def diff4_2 := pile4 - pile2  -- Difference between fourth and second pile

-- Assume the pattern continues with differences increasing by 4.
def diff3_2 := diff2_1 + 4    -- Difference between third and second pile

-- Define the number of magazines in the third pile.
def pile3 := pile2 + diff3_2

-- Theorem stating the number of magazines in the third pile.
theorem magazines_in_third_pile : pile3 = 9 := by sorry

end magazines_in_third_pile_l154_154590


namespace probability_shaded_region_l154_154614

def triangle_game :=
  let total_regions := 6
  let shaded_regions := 3
  shaded_regions / total_regions

theorem probability_shaded_region:
  triangle_game = 1 / 2 := by
  sorry

end probability_shaded_region_l154_154614


namespace gcd_gx_x_l154_154351

-- Condition: x is a multiple of 7263
def isMultipleOf7263 (x : ℕ) : Prop := ∃ k : ℕ, x = 7263 * k

-- Definition of g(x)
def g (x : ℕ) : ℕ := (3*x + 4) * (9*x + 5) * (17*x + 11) * (x + 17)

-- Statement to be proven
theorem gcd_gx_x (x : ℕ) (h : isMultipleOf7263 x) : Nat.gcd (g x) x = 1 := by
  sorry

end gcd_gx_x_l154_154351


namespace smallest_composite_no_prime_under_15_correct_l154_154646

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l154_154646


namespace books_still_to_read_l154_154121

-- Define the given conditions
def total_books : ℕ := 22
def books_read : ℕ := 12

-- State the theorem to be proven
theorem books_still_to_read : total_books - books_read = 10 := 
by
  -- skipping the proof
  sorry

end books_still_to_read_l154_154121


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l154_154839

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l154_154839


namespace sum_of_abs_first_10_terms_l154_154826

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ := n^2 - 5 * n + 2

theorem sum_of_abs_first_10_terms : 
  let S := sum_of_first_n_terms 10
  let S3 := sum_of_first_n_terms 3
  (S - 2 * S3) = 60 := 
by
  sorry

end sum_of_abs_first_10_terms_l154_154826


namespace find_a_l154_154467

theorem find_a (a k : ℝ) (h1 : ∀ x, a * x^2 + 3 * x - k = 0 → x = 7) (h2 : k = 119) : a = 2 :=
by
  sorry

end find_a_l154_154467


namespace diamond_comm_not_assoc_l154_154487

def diamond (a b : ℤ) : ℤ := (a * b + 5) / (a + b)

-- Lemma: Verify commutativity of the diamond operation
lemma diamond_comm (a b : ℤ) (ha : a > 1) (hb : b > 1) : 
  diamond a b = diamond b a := by
  sorry

-- Lemma: Verify non-associativity of the diamond operation
lemma diamond_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  sorry

-- Theorem: The diamond operation is commutative but not associative
theorem diamond_comm_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond a b = diamond b a ∧ diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  apply And.intro
  · apply diamond_comm
    apply ha
    apply hb
  · apply diamond_not_assoc
    apply ha
    apply hb
    apply hc

end diamond_comm_not_assoc_l154_154487


namespace sine_cosine_fraction_l154_154968

theorem sine_cosine_fraction (θ : ℝ) (h : Real.tan θ = 2) : 
    (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
by 
  sorry

end sine_cosine_fraction_l154_154968


namespace reciprocal_neg_7_l154_154736

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_neg_7 : reciprocal (-7) = -1 / 7 := 
by
  sorry

end reciprocal_neg_7_l154_154736


namespace max_intersections_l154_154325

theorem max_intersections 
  (X : Fin 8) (Y : Fin 6) :
  let segments := List.product (List.ofFn (fun _ => X)) (List.ofFn (fun _ => Y))
  let count_intersections := Nat.choose 8 2 * Nat.choose 6 2
  count_intersections = 420 :=
by
  sorry

end max_intersections_l154_154325


namespace max_value_q_l154_154239

open Nat

theorem max_value_q (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 24 :=
sorry

end max_value_q_l154_154239


namespace pyramid_coloring_methods_l154_154058

theorem pyramid_coloring_methods : 
  ∀ (P A B C D : ℕ),
    (P ≠ A) ∧ (P ≠ B) ∧ (P ≠ C) ∧ (P ≠ D) ∧
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
    (P < 5) ∧ (A < 5) ∧ (B < 5) ∧ (C < 5) ∧ (D < 5) →
  ∃! (num_methods : ℕ), num_methods = 420 :=
by
  sorry

end pyramid_coloring_methods_l154_154058


namespace kaleb_sold_books_l154_154234

theorem kaleb_sold_books (initial_books sold_books purchased_books final_books : ℕ)
  (H_initial : initial_books = 34)
  (H_purchased : purchased_books = 7)
  (H_final : final_books = 24) :
  sold_books = 17 :=
by
  have H_equation : (initial_books - sold_books) + purchased_books = final_books,
    by sorry
  rw [H_initial, H_purchased, H_final] at H_equation,
  sorry

end kaleb_sold_books_l154_154234


namespace seq_arithmetic_l154_154995

noncomputable def f (x n : ℝ) : ℝ := (x - 1)^2 + n

def a_n (n : ℝ) : ℝ := n
def b_n (n : ℝ) : ℝ := n + 4
def c_n (n : ℝ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem seq_arithmetic (n : ℕ) (hn : 0 < n) :
  ∃ d, d ≠ 0 ∧ ∀ n, c_n (↑n : ℝ) = c_n (↑n + 1 : ℝ) - d := 
sorry

end seq_arithmetic_l154_154995


namespace initial_marbles_l154_154315

variable (C_initial : ℕ)
variable (marbles_given : ℕ := 42)
variable (marbles_left : ℕ := 5)

theorem initial_marbles :
  C_initial = marbles_given + marbles_left :=
sorry

end initial_marbles_l154_154315


namespace evaluate_log_l154_154026

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l154_154026


namespace candy_cost_55_cents_l154_154099

theorem candy_cost_55_cents
  (paid: ℕ) (change: ℕ) (num_coins: ℕ)
  (coin1 coin2 coin3 coin4: ℕ)
  (h1: paid = 100)
  (h2: num_coins = 4)
  (h3: coin1 = 25)
  (h4: coin2 = 10)
  (h5: coin3 = 10)
  (h6: coin4 = 0)
  (h7: change = coin1 + coin2 + coin3 + coin4) :
  paid - change = 55 :=
by
  -- The proof can be provided here.
  sorry

end candy_cost_55_cents_l154_154099


namespace color_complete_graph_l154_154625

open SimpleGraph

-- Definitions used in conditions
def K9 : SimpleGraph (Fin 9) := completeGraph (Fin 9)

def edgeColoring (c : Symmetric (Fin 9) × (Fin 9) → Fin 2) : Prop := 
  ∀ e : Symmetric (Fin 9) × (Fin 9), e ∈ K9.edgeSet → c e ∈ ({0, 1} : Finset (Fin 2))

-- Main theorem statement
theorem color_complete_graph (c : Symmetric (Fin 9) × (Fin 9) → Fin 2) (hc : edgeColoring c) :
  ∃ (S : Finset (Fin 9)), (S.card = 4 ∧ S.pairwise (λ u v, c ⟨u, v⟩ = 0)) ∨ (S.card = 3 ∧ S.pairwise (λ u v, c ⟨u, v⟩ = 1)) := 
sorry

end color_complete_graph_l154_154625


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l154_154437

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l154_154437


namespace no_positive_integer_satisfies_conditions_l154_154488

theorem no_positive_integer_satisfies_conditions :
  ¬∃ (n : ℕ), n > 1 ∧ (∃ (p1 : ℕ), Prime p1 ∧ n = p1^2) ∧ (∃ (p2 : ℕ), Prime p2 ∧ 3 * n + 16 = p2^2) :=
by
  sorry

end no_positive_integer_satisfies_conditions_l154_154488


namespace parentheses_removal_correct_l154_154589

theorem parentheses_removal_correct (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 :=
by
  sorry

end parentheses_removal_correct_l154_154589


namespace functions_equiv_l154_154900

noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x)
noncomputable def g_D (x : ℝ) : ℝ := (1/2) * Real.log x

theorem functions_equiv : ∀ x : ℝ, x > 0 → f_D x = g_D x := by
  intro x h
  sorry

end functions_equiv_l154_154900


namespace quadratic_trinomial_positive_c_l154_154735

theorem quadratic_trinomial_positive_c
  (a b c : ℝ)
  (h1 : b^2 < 4 * a * c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_trinomial_positive_c_l154_154735


namespace geometric_sequence_property_l154_154523

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∀ m n : ℕ, a (m + n) = a m * a n / a 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h : geometric_sequence a) 
    (h4 : a 4 = 5) 
    (h8 : a 8 = 6) : 
    a 2 * a 10 = 30 :=
by
  sorry

end geometric_sequence_property_l154_154523


namespace unique_two_digit_perfect_square_divisible_by_5_l154_154357

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The statement to prove: there is exactly 1 two-digit perfect square that is divisible by 5
theorem unique_two_digit_perfect_square_divisible_by_5 :
  ∃! n : ℕ, is_perfect_square n ∧ two_digit n ∧ divisible_by_5 n :=
sorry

end unique_two_digit_perfect_square_divisible_by_5_l154_154357


namespace basketball_price_l154_154102

variable (P : ℝ)

def coachA_cost : ℝ := 10 * P
def coachB_baseball_cost : ℝ := 14 * 2.5
def coachB_bat_cost : ℝ := 18
def coachB_total_cost : ℝ := coachB_baseball_cost + coachB_bat_cost
def coachA_excess_cost : ℝ := 237

theorem basketball_price (h : coachA_cost P = coachB_total_cost + coachA_excess_cost) : P = 29 :=
by
  sorry

end basketball_price_l154_154102


namespace weekly_diesel_spending_l154_154869

-- Conditions
def cost_per_gallon : ℝ := 3
def fuel_used_in_two_weeks : ℝ := 24

-- Question: Prove that Mr. Alvarez spends $36 on diesel fuel each week.
theorem weekly_diesel_spending : (fuel_used_in_two_weeks / 2) * cost_per_gallon = 36 := by
  sorry

end weekly_diesel_spending_l154_154869


namespace line_plane_intersection_l154_154798

theorem line_plane_intersection 
  (t : ℝ)
  (x_eq : ∀ t: ℝ, x = 5 - t)
  (y_eq : ∀ t: ℝ, y = -3 + 5 * t)
  (z_eq : ∀ t: ℝ, z = 1 + 2 * t)
  (plane_eq : 3 * x + 7 * y - 5 * z - 11 = 0)
  : x = 4 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end line_plane_intersection_l154_154798


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154416

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l154_154416


namespace smallest_arithmetic_mean_divisible_by_1111_l154_154402

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l154_154402


namespace arithmetic_sqrt_of_49_l154_154724

theorem arithmetic_sqrt_of_49 : ∃ x : ℕ, x^2 = 49 ∧ x = 7 :=
by
  sorry

end arithmetic_sqrt_of_49_l154_154724


namespace rearrange_cards_l154_154549

theorem rearrange_cards :
  (∀ (arrangement : List ℕ), arrangement = [3, 1, 2, 4, 5, 6] ∨ arrangement = [1, 2, 4, 5, 6, 3] →
  (∀ card, card ∈ arrangement → List.erase arrangement card = [1, 2, 4, 5, 6] ∨
                                        List.erase arrangement card = [3, 1, 2, 4, 5]) →
  List.length arrangement = 6) →
  (∃ n, n = 10) :=
by
  sorry

end rearrange_cards_l154_154549


namespace third_number_hcf_lcm_l154_154260

theorem third_number_hcf_lcm (N : ℕ) 
  (HCF : Nat.gcd (Nat.gcd 136 144) N = 8)
  (LCM : Nat.lcm (Nat.lcm 136 144) N = 2^4 * 3^2 * 17 * 7) : 
  N = 7 := 
  sorry

end third_number_hcf_lcm_l154_154260


namespace simplify_expression_l154_154546

theorem simplify_expression (x : ℝ) : 
  (2 * x - 3 * (2 + x) + 4 * (2 - x) - 5 * (2 + 3 * x)) = -20 * x - 8 :=
by
  sorry

end simplify_expression_l154_154546


namespace opposite_of_neg_one_third_l154_154580

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end opposite_of_neg_one_third_l154_154580


namespace inequality_solution_subset_l154_154825

theorem inequality_solution_subset {x a : ℝ} : (∀ x, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 :=
by sorry

end inequality_solution_subset_l154_154825


namespace maximum_value_of_x_plus_2y_l154_154060

theorem maximum_value_of_x_plus_2y (x y : ℝ) (h : x^2 - 2 * x + 4 * y = 5) : ∃ m, m = x + 2 * y ∧ m ≤ 9/2 := by
  sorry

end maximum_value_of_x_plus_2y_l154_154060


namespace sum_a_c_eq_l154_154336

theorem sum_a_c_eq
  (a b c d : ℝ)
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) :
  a + c = 8.4 :=
by
  sorry

end sum_a_c_eq_l154_154336


namespace miles_driven_l154_154855

-- Definitions based on the conditions
def years : ℕ := 9
def months_in_a_year : ℕ := 12
def months_in_a_period : ℕ := 4
def miles_per_period : ℕ := 37000

-- The proof statement
theorem miles_driven : years * months_in_a_year / months_in_a_period * miles_per_period = 999000 := 
sorry

end miles_driven_l154_154855


namespace log3_of_9_to_3_l154_154043

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l154_154043


namespace arithmetic_geometric_l154_154491

theorem arithmetic_geometric (a_n : ℕ → ℤ) (h1 : ∀ n, a_n n = a_n 0 + n * 2)
  (h2 : ∃ a, a = a_n 0 ∧ (a_n 0 + 4)^2 = a_n 0 * (a_n 0 + 6)) : a_n 0 = -8 := by
  sorry

end arithmetic_geometric_l154_154491


namespace Paige_recycled_pounds_l154_154717

-- Definitions based on conditions from step a)
def points_per_pound := 1 / 4
def friends_pounds_recycled := 2
def total_points := 4

-- The proof statement (no proof required)
theorem Paige_recycled_pounds :
  let total_pounds_recycled := total_points * 4
  let paige_pounds_recycled := total_pounds_recycled - friends_pounds_recycled
  paige_pounds_recycled = 14 :=
by
  sorry

end Paige_recycled_pounds_l154_154717


namespace cody_books_second_week_l154_154317

noncomputable def total_books := 54
noncomputable def books_first_week := 6
noncomputable def books_weeks_after_second := 9
noncomputable def total_weeks := 7

theorem cody_books_second_week :
  let b2 := total_books - (books_first_week + books_weeks_after_second * (total_weeks - 2))
  b2 = 3 :=
by
  sorry

end cody_books_second_week_l154_154317


namespace focus_of_parabola_l154_154727

theorem focus_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 16 * x^2) : 
    ∃ p, p = (0, 1/64) := 
by
    existsi (0, 1/64)
    -- The proof would go here, but we are adding sorry to skip it 
    sorry

end focus_of_parabola_l154_154727


namespace calculate_total_selling_price_l154_154775

noncomputable def total_selling_price (cost_price1 cost_price2 cost_price3 profit_percent1 profit_percent2 profit_percent3 : ℝ) : ℝ :=
  let sp1 := cost_price1 + (profit_percent1 / 100 * cost_price1)
  let sp2 := cost_price2 + (profit_percent2 / 100 * cost_price2)
  let sp3 := cost_price3 + (profit_percent3 / 100 * cost_price3)
  sp1 + sp2 + sp3

theorem calculate_total_selling_price :
  total_selling_price 550 750 1000 30 25 20 = 2852.5 :=
by
  -- proof omitted
  sorry

end calculate_total_selling_price_l154_154775


namespace back_wheel_revolutions_l154_154715

-- Defining relevant distances and conditions
def front_wheel_radius : ℝ := 3 -- radius in feet
def back_wheel_radius : ℝ := 0.5 -- radius in feet
def front_wheel_revolutions : ℕ := 120

-- The target theorem
theorem back_wheel_revolutions :
  let front_wheel_circumference := 2 * Real.pi * front_wheel_radius
  let total_distance := front_wheel_circumference * (front_wheel_revolutions : ℝ)
  let back_wheel_circumference := 2 * Real.pi * back_wheel_radius
  let back_wheel_revs := total_distance / back_wheel_circumference
  back_wheel_revs = 720 :=
by
  sorry

end back_wheel_revolutions_l154_154715


namespace nursing_home_received_boxes_l154_154713

-- Each condition will be a definition in Lean 4.
def vitamins := 472
def supplements := 288
def total_boxes := 760

-- Statement of the proof problem in Lean
theorem nursing_home_received_boxes : vitamins + supplements = total_boxes := by
  sorry

end nursing_home_received_boxes_l154_154713


namespace arithmetic_sequence_inequality_l154_154236

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

-- All terms are positive
def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem arithmetic_sequence_inequality
  (h_arith_seq : is_arithmetic_sequence a a1 d)
  (h_non_zero_diff : d ≠ 0)
  (h_positive : all_positive a) :
  (a 1) * (a 8) < (a 4) * (a 5) :=
by
  sorry

end arithmetic_sequence_inequality_l154_154236


namespace angle_between_hands_230_pm_l154_154922

def hour_hand_position (hour minute : ℕ) : ℕ := hour % 12 * 5 + minute / 12
def minute_hand_position (minute : ℕ) : ℕ := minute
def divisions_to_angle (divisions : ℕ) : ℕ := divisions * 30

theorem angle_between_hands_230_pm :
    hour_hand_position 2 30 = 2 * 5 + 30 / 12 ∧
    minute_hand_position 30 = 30 ∧
    divisions_to_angle (minute_hand_position 30 / 5 - hour_hand_position 2 30 / 5) = 105 :=
by {
    sorry
}

end angle_between_hands_230_pm_l154_154922


namespace smallest_a_for_nonprime_l154_154942

theorem smallest_a_for_nonprime (a : ℕ) : (∀ x : ℤ, ∃ d : ℤ, d ∣ (x^4 + a^4) ∧ d ≠ 1 ∧ d ≠ (x^4 + a^4)) ↔ a = 3 := by
  sorry

end smallest_a_for_nonprime_l154_154942


namespace minimum_value_of_expression_l154_154989

theorem minimum_value_of_expression 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : 3 * a + 4 * b + 2 * c = 3) : 
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) = 1.5 :=
sorry

end minimum_value_of_expression_l154_154989


namespace trigonometric_identity_l154_154154

theorem trigonometric_identity :
  (1 / 2 - (Real.cos (15 * Real.pi / 180)) ^ 2) = - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_identity_l154_154154


namespace problem1_l154_154288

theorem problem1 (a : ℝ) (x : ℝ) (h : a > 0) : |x - (1/a)| + |x + a| ≥ 2 :=
sorry

end problem1_l154_154288


namespace flooring_area_already_installed_l154_154276

variable (living_room_length : ℕ) (living_room_width : ℕ) 
variable (flooring_sqft_per_box : ℕ)
variable (remaining_boxes_needed : ℕ)
variable (already_installed : ℕ)

theorem flooring_area_already_installed 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_sqft_per_box = 10)
  (h4 : remaining_boxes_needed = 7)
  (h5 : living_room_length * living_room_width = 320)
  (h6 : already_installed = 320 - remaining_boxes_needed * flooring_sqft_per_box) : 
  already_installed = 250 :=
by
  sorry

end flooring_area_already_installed_l154_154276


namespace smallest_whole_number_inequality_l154_154484

theorem smallest_whole_number_inequality (x : ℕ) (h : 3 * x + 4 > 11 - 2 * x) : x ≥ 2 :=
sorry

end smallest_whole_number_inequality_l154_154484


namespace f_neg_one_f_eq_half_l154_154961

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^(-x) else Real.log x / Real.log 2

theorem f_neg_one : f (-1) = 2 := by
  sorry

theorem f_eq_half (x : ℝ) : f x = 1 / 2 ↔ x = Real.sqrt 2 := by
  sorry

end f_neg_one_f_eq_half_l154_154961


namespace opposite_of_minus_one_third_l154_154565

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_minus_one_third_l154_154565


namespace standing_arrangements_l154_154369

theorem standing_arrangements : ∃ (arrangements : ℕ), arrangements = 2 :=
by
  -- Given that Jia, Yi, Bing, and Ding are four distinct people standing in a row
  -- We need to prove that there are exactly 2 different ways for them to stand such that Jia is not at the far left and Yi is not at the far right
  sorry

end standing_arrangements_l154_154369


namespace find_min_value_l154_154179

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end find_min_value_l154_154179


namespace subtract_square_l154_154209

theorem subtract_square (n : ℝ) (h : n = 68.70953354520753) : (n^2 - 20^2) = 4321.000000000001 := by
  sorry

end subtract_square_l154_154209


namespace complementary_event_l154_154076

def car_a_selling_well : Prop := sorry
def car_b_selling_poorly : Prop := sorry

def event_A : Prop := car_a_selling_well ∧ car_b_selling_poorly
def event_complement (A : Prop) : Prop := ¬A

theorem complementary_event :
  event_complement event_A = (¬car_a_selling_well ∨ ¬car_b_selling_poorly) :=
by
  sorry

end complementary_event_l154_154076


namespace multiplication_to_squares_l154_154370

theorem multiplication_to_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 :=
by
  sorry

end multiplication_to_squares_l154_154370


namespace sue_received_votes_l154_154266

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end sue_received_votes_l154_154266


namespace subtracted_amount_l154_154603

theorem subtracted_amount (N A : ℝ) (h1 : 0.30 * N - A = 20) (h2 : N = 300) : A = 70 :=
by
  sorry

end subtracted_amount_l154_154603


namespace minimum_surface_area_l154_154563

def small_cuboid_1_length := 3 -- Edge length of small cuboid
def small_cuboid_2_length := 4 -- Edge length of small cuboid
def small_cuboid_3_length := 5 -- Edge length of small cuboid

def num_small_cuboids := 24 -- Number of small cuboids used to build the large cuboid

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def large_cuboid_length := 15 -- Corrected length dimension
def large_cuboid_width := 10  -- Corrected width dimension
def large_cuboid_height := 16 -- Corrected height dimension

theorem minimum_surface_area : surface_area large_cuboid_length large_cuboid_width large_cuboid_height = 788 := by
  sorry -- Proof to be completed

end minimum_surface_area_l154_154563


namespace championship_outcomes_l154_154052

theorem championship_outcomes (students events : ℕ) (hs : students = 5) (he : events = 3) :
  ∃ outcomes : ℕ, outcomes = 5 ^ 3 := by
  sorry

end championship_outcomes_l154_154052


namespace smallest_q_exists_l154_154707

theorem smallest_q_exists (p q : ℕ) (h : 0 < q) (h_eq : (p : ℚ) / q = 123456789 / 100000000000) :
  q = 10989019 :=
sorry

end smallest_q_exists_l154_154707
