import Mathlib

namespace NUMINAMATH_GPT_line_equation_l2335_233565

-- Given a point and a direction vector
def point : ℝ × ℝ := (3, 4)
def direction_vector : ℝ × ℝ := (-2, 1)

-- Equation of the line passing through the given point with the given direction vector
theorem line_equation (x y : ℝ) : 
  (x = 3 ∧ y = 4) → ∃a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -11 ∧ a*x + b*y + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l2335_233565


namespace NUMINAMATH_GPT_geoff_initial_percent_l2335_233545

theorem geoff_initial_percent (votes_cast : ℕ) (win_percent : ℝ) (needed_more_votes : ℕ) (initial_votes : ℕ)
  (h1 : votes_cast = 6000)
  (h2 : win_percent = 50.5)
  (h3 : needed_more_votes = 3000)
  (h4 : initial_votes = 31) :
  (initial_votes : ℝ) / votes_cast * 100 = 0.52 :=
by
  sorry

end NUMINAMATH_GPT_geoff_initial_percent_l2335_233545


namespace NUMINAMATH_GPT_rainy_days_l2335_233506

theorem rainy_days (n R NR : ℕ): (n * R + 3 * NR = 20) ∧ (3 * NR = n * R + 10) ∧ (R + NR = 7) → R = 2 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_rainy_days_l2335_233506


namespace NUMINAMATH_GPT_least_number_of_people_l2335_233549

-- Conditions
def first_caterer_cost (x : ℕ) : ℕ := 120 + 18 * x
def second_caterer_cost (x : ℕ) : ℕ := 250 + 15 * x

-- Proof Statement
theorem least_number_of_people (x : ℕ) (h : x ≥ 44) : first_caterer_cost x > second_caterer_cost x :=
by sorry

end NUMINAMATH_GPT_least_number_of_people_l2335_233549


namespace NUMINAMATH_GPT_number_of_possible_tower_heights_l2335_233569

-- Axiom for the possible increment values when switching brick orientations
def possible_increments : Set ℕ := {4, 7}

-- Base height when all bricks contribute the smallest dimension
def base_height (num_bricks : ℕ) (smallest_side : ℕ) : ℕ :=
  num_bricks * smallest_side

-- Check if a given height can be achieved by changing orientations of the bricks
def can_achieve_height (h : ℕ) (n : ℕ) (increments : Set ℕ) : Prop :=
  ∃ m k : ℕ, h = base_height n 2 + m * 4 + k * 7

-- Final proof statement
theorem number_of_possible_tower_heights :
  (50 : ℕ) = 50 →
  (∀ k : ℕ, (100 + k * 4 <= 450) → can_achieve_height (100 + k * 4) 50 possible_increments) →
  ∃ (num_possible_heights : ℕ), num_possible_heights = 90 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_tower_heights_l2335_233569


namespace NUMINAMATH_GPT_negation_P_l2335_233583

variable (P : Prop) (P_def : ∀ x : ℝ, Real.sin x ≤ 1)

theorem negation_P : ¬P ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end NUMINAMATH_GPT_negation_P_l2335_233583


namespace NUMINAMATH_GPT_map_distance_l2335_233529

variable (map_distance_km : ℚ) (map_distance_inches : ℚ) (actual_distance_km: ℚ)

theorem map_distance (h1 : actual_distance_km = 136)
                     (h2 : map_distance_inches = 42)
                     (h3 : map_distance_km = 18.307692307692307) :
  (actual_distance_km * map_distance_inches / map_distance_km = 312) :=
by sorry

end NUMINAMATH_GPT_map_distance_l2335_233529


namespace NUMINAMATH_GPT_boxes_of_toothpicks_needed_l2335_233525

def total_cards : Nat := 52
def unused_cards : Nat := 23
def cards_used : Nat := total_cards - unused_cards

def toothpicks_wall_per_card : Nat := 64
def windows_per_card : Nat := 3
def doors_per_card : Nat := 2
def toothpicks_per_window_or_door : Nat := 12
def roof_toothpicks : Nat := 1250
def box_capacity : Nat := 750

def toothpicks_for_walls : Nat := cards_used * toothpicks_wall_per_card
def toothpicks_per_card_windows_doors : Nat := (windows_per_card + doors_per_card) * toothpicks_per_window_or_door
def toothpicks_for_windows_doors : Nat := cards_used * toothpicks_per_card_windows_doors
def total_toothpicks_needed : Nat := toothpicks_for_walls + toothpicks_for_windows_doors + roof_toothpicks

def boxes_needed := Nat.ceil (total_toothpicks_needed / box_capacity)

theorem boxes_of_toothpicks_needed : boxes_needed = 7 := by
  -- Proof should be done here
  sorry

end NUMINAMATH_GPT_boxes_of_toothpicks_needed_l2335_233525


namespace NUMINAMATH_GPT_middle_part_is_28_4_over_11_l2335_233535

theorem middle_part_is_28_4_over_11 (x : ℚ) :
  let part1 := x
  let part2 := (1/2) * x
  let part3 := (1/3) * x
  part1 + part2 + part3 = 104
  ∧ part2 = 28 + 4/11 := by
  sorry

end NUMINAMATH_GPT_middle_part_is_28_4_over_11_l2335_233535


namespace NUMINAMATH_GPT_consecutive_negatives_product_to_sum_l2335_233550

theorem consecutive_negatives_product_to_sum :
  ∃ (n : ℤ), n * (n + 1) = 2184 ∧ n + (n + 1) = -95 :=
by {
  sorry
}

end NUMINAMATH_GPT_consecutive_negatives_product_to_sum_l2335_233550


namespace NUMINAMATH_GPT_musketeer_statements_triplets_count_l2335_233576

-- Definitions based on the conditions
def musketeers : Type := { x : ℕ // x < 3 }

def is_guilty (m : musketeers) : Prop := sorry  -- Placeholder for the property of being guilty

def statement (m1 m2 : musketeers) : Prop := sorry  -- Placeholder for the statement made by one musketeer about another

-- Condition that each musketeer makes one statement
def made_statement (m : musketeers) : Prop := sorry

-- Condition that exactly one musketeer lied
def exactly_one_lied : Prop := sorry

-- The final proof problem statement:
theorem musketeer_statements_triplets_count : ∃ n : ℕ, n = 99 :=
  sorry

end NUMINAMATH_GPT_musketeer_statements_triplets_count_l2335_233576


namespace NUMINAMATH_GPT_lcm_of_ratio_and_hcf_l2335_233510

theorem lcm_of_ratio_and_hcf (a b : ℕ) (x : ℕ) (h_ratio : a = 3 * x ∧ b = 4 * x) (h_hcf : Nat.gcd a b = 4) : Nat.lcm a b = 48 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_ratio_and_hcf_l2335_233510


namespace NUMINAMATH_GPT_find_number_69_3_l2335_233509

theorem find_number_69_3 (x : ℝ) (h : (x * 0.004) / 0.03 = 9.237333333333334) : x = 69.3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_69_3_l2335_233509


namespace NUMINAMATH_GPT_gf_neg3_eq_1262_l2335_233595

def f (x : ℤ) : ℤ := x^3 + 6
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 2

theorem gf_neg3_eq_1262 : g (f (-3)) = 1262 := by
  sorry

end NUMINAMATH_GPT_gf_neg3_eq_1262_l2335_233595


namespace NUMINAMATH_GPT_max_value_of_a_l2335_233531

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_of_a_l2335_233531


namespace NUMINAMATH_GPT_consecutive_integers_sum_l2335_233516

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 := by
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l2335_233516


namespace NUMINAMATH_GPT_find_smallest_c_l2335_233599

theorem find_smallest_c (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (graph_eq : ∀ x, (a * Real.sin (b * x + c) + d) = 5 → x = (π / 6))
    (amplitude_eq : a = 3) : c = π / 2 :=
sorry

end NUMINAMATH_GPT_find_smallest_c_l2335_233599


namespace NUMINAMATH_GPT_opposite_of_neg_five_l2335_233522

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end NUMINAMATH_GPT_opposite_of_neg_five_l2335_233522


namespace NUMINAMATH_GPT_mod_remainder_l2335_233520

theorem mod_remainder (a b c d : ℕ) (h1 : a = 11) (h2 : b = 9) (h3 : c = 7) (h4 : d = 7) :
  (a^d + b^(d + 1) + c^(d + 2)) % d = 1 := 
by 
  sorry

end NUMINAMATH_GPT_mod_remainder_l2335_233520


namespace NUMINAMATH_GPT_smallest_number_of_cookies_proof_l2335_233592

def satisfies_conditions (a : ℕ) : Prop :=
  (a % 6 = 5) ∧ (a % 8 = 6) ∧ (a % 10 = 9) ∧ (∃ n : ℕ, a = n * n)

def smallest_number_of_cookies : ℕ :=
  2549

theorem smallest_number_of_cookies_proof :
  satisfies_conditions smallest_number_of_cookies :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_cookies_proof_l2335_233592


namespace NUMINAMATH_GPT_radius_of_circle_eq_zero_l2335_233586

theorem radius_of_circle_eq_zero :
  ∀ x y: ℝ, (x^2 + 8 * x + y^2 - 10 * y + 41 = 0) → (0 : ℝ) = 0 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_radius_of_circle_eq_zero_l2335_233586


namespace NUMINAMATH_GPT_annie_bought_figurines_l2335_233566

theorem annie_bought_figurines:
  let televisions := 5
  let cost_per_television := 50
  let total_spent := 260
  let cost_per_figurine := 1
  let cost_of_televisions := televisions * cost_per_television
  let remaining_money := total_spent - cost_of_televisions
  remaining_money / cost_per_figurine = 10 :=
by
  sorry

end NUMINAMATH_GPT_annie_bought_figurines_l2335_233566


namespace NUMINAMATH_GPT_equation_solution_l2335_233563

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + (2 / 5) = 0 ↔ 
  a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_equation_solution_l2335_233563


namespace NUMINAMATH_GPT_no_12_term_geometric_seq_in_1_to_100_l2335_233572

theorem no_12_term_geometric_seq_in_1_to_100 :
  ¬ ∃ (s : Fin 12 → Set ℕ),
    (∀ i, ∃ (a q : ℕ), (s i = {a * q^n | n : ℕ}) ∧ (∀ x ∈ s i, 1 ≤ x ∧ x ≤ 100)) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ i, n ∈ s i) := 
sorry

end NUMINAMATH_GPT_no_12_term_geometric_seq_in_1_to_100_l2335_233572


namespace NUMINAMATH_GPT_no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l2335_233517

theorem no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49 :
  ∀ n : ℕ, ¬ (∃ k : ℤ, (n^2 + 5 * n + 1) = 49 * k) :=
by
  sorry

end NUMINAMATH_GPT_no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l2335_233517


namespace NUMINAMATH_GPT_fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l2335_233584

-- Define the sequences as functions
def row1 (n : ℕ) : ℤ := (-2)^n
def row2 (n : ℕ) : ℤ := row1 n + 2
def row3 (n : ℕ) : ℤ := (-1) * (-2)^n

-- Theorems to be proven

-- (1) Prove the fourth term in row ① is 16 
theorem fourth_term_row1_is_16 : row1 4 = 16 := sorry

-- (1) Prove the nth term in row ① is (-2)^n
theorem nth_term_row1 (n : ℕ) : row1 n = (-2)^n := sorry

-- (2) Let the nth number in row ① be a, prove the nth number in row ② is a + 2
theorem nth_term_row2 (n : ℕ) : row2 n = row1 n + 2 := sorry

-- (3) If the sum of three consecutive numbers in row ③ is -192, find these numbers
theorem sum_three_consecutive_row3 : ∃ n : ℕ, row3 n + row3 (n + 1) + row3 (n + 2) = -192 ∧ 
  row3 n  = -64 ∧ row3 (n + 1) = 128 ∧ row3 (n + 2) = -256 := sorry

end NUMINAMATH_GPT_fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l2335_233584


namespace NUMINAMATH_GPT_final_middle_pile_cards_l2335_233570

-- Definitions based on conditions
def initial_cards_per_pile (n : ℕ) (h : n ≥ 2) := n

def left_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n - 2
def middle_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n + 2
def right_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n

def right_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n - 1
def middle_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n + 3

def left_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := n
def middle_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := (n + 3) - n

-- The proof problem to solve
theorem final_middle_pile_cards (n : ℕ) (h : n ≥ 2) : middle_pile_after_step_4 n h = 5 :=
sorry

end NUMINAMATH_GPT_final_middle_pile_cards_l2335_233570


namespace NUMINAMATH_GPT_celery_cost_l2335_233577

noncomputable def supermarket_problem
  (total_money : ℕ)
  (price_cereal discount_cereal price_bread : ℕ)
  (price_milk discount_milk price_potato num_potatoes : ℕ)
  (leftover_money : ℕ) 
  (total_cost : ℕ) 
  (cost_of_celery : ℕ) :=
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) + 
   leftover_money = total_money ∧
  total_cost = total_money - leftover_money ∧
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) = total_cost - cost_of_celery

theorem celery_cost (total_money : ℕ := 60) 
  (price_cereal : ℕ := 12) 
  (discount_cereal : ℕ := 50) 
  (price_bread : ℕ := 8) 
  (price_milk : ℕ := 10) 
  (discount_milk : ℕ := 90) 
  (price_potato : ℕ := 1) 
  (num_potatoes : ℕ := 6) 
  (leftover_money : ℕ := 26) 
  (total_cost : ℕ := 34) :
  supermarket_problem total_money price_cereal discount_cereal price_bread price_milk discount_milk price_potato num_potatoes leftover_money total_cost 5 :=
by
  sorry

end NUMINAMATH_GPT_celery_cost_l2335_233577


namespace NUMINAMATH_GPT_find_m_minus_n_l2335_233590

theorem find_m_minus_n (x y m n : ℤ) (h1 : x = -2) (h2 : y = 1) 
  (h3 : 3 * x + 2 * y = m) (h4 : n * x - y = 1) : m - n = -3 :=
by sorry

end NUMINAMATH_GPT_find_m_minus_n_l2335_233590


namespace NUMINAMATH_GPT_exists_mod_inv_l2335_233526

theorem exists_mod_inv (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h : ¬ a ∣ p) : ∃ b : ℕ, a * b ≡ 1 [MOD p] :=
by
  sorry

end NUMINAMATH_GPT_exists_mod_inv_l2335_233526


namespace NUMINAMATH_GPT_solve_equation_l2335_233574

theorem solve_equation (x : ℝ) : (x + 3)^4 + (x + 1)^4 = 82 → x = 0 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2335_233574


namespace NUMINAMATH_GPT_maximize_profit_l2335_233547

noncomputable def profit (x : ℝ) : ℝ :=
  let selling_price := 10 + 0.5 * x
  let sales_volume := 200 - 10 * x
  (selling_price - 8) * sales_volume

theorem maximize_profit : ∃ x : ℝ, x = 8 → profit x = profit 8 ∧ (∀ y : ℝ, profit y ≤ profit 8) := 
  sorry

end NUMINAMATH_GPT_maximize_profit_l2335_233547


namespace NUMINAMATH_GPT_range_of_m_l2335_233512

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = -3) (h3 : x + y > 0) : m > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2335_233512


namespace NUMINAMATH_GPT_positive_integer_pairs_l2335_233552

theorem positive_integer_pairs (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (∃ k : ℕ, k > 0 ∧ k = a^2 / (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, 0 < l ∧ 
    ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_pairs_l2335_233552


namespace NUMINAMATH_GPT_mod_equivalence_l2335_233513

theorem mod_equivalence (x y m : ℤ) (h1 : x ≡ 25 [ZMOD 60]) (h2 : y ≡ 98 [ZMOD 60]) (h3 : m = 167) :
  x - y ≡ m [ZMOD 60] :=
sorry

end NUMINAMATH_GPT_mod_equivalence_l2335_233513


namespace NUMINAMATH_GPT_quadratic_solution_l2335_233524

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_solution_l2335_233524


namespace NUMINAMATH_GPT_completing_the_square_x_squared_minus_4x_plus_1_eq_0_l2335_233558

theorem completing_the_square_x_squared_minus_4x_plus_1_eq_0 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x
  intros h
  sorry

end NUMINAMATH_GPT_completing_the_square_x_squared_minus_4x_plus_1_eq_0_l2335_233558


namespace NUMINAMATH_GPT_gcf_of_lcm_9_21_and_10_22_eq_one_l2335_233521

theorem gcf_of_lcm_9_21_and_10_22_eq_one :
  Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 :=
sorry

end NUMINAMATH_GPT_gcf_of_lcm_9_21_and_10_22_eq_one_l2335_233521


namespace NUMINAMATH_GPT_correct_calculation_l2335_233573

theorem correct_calculation (a b : ℝ) : 
  ¬(3 * a + b = 3 * a * b) ∧ 
  ¬(a^2 + a^2 = a^4) ∧ 
  ¬((a - b)^2 = a^2 - b^2) ∧ 
  ((-3 * a)^2 = 9 * a^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2335_233573


namespace NUMINAMATH_GPT_range_of_x_l2335_233560

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_x (x m : ℝ) (hx : x > -2 ∧ x < 2/3) (hm : m ≥ -2 ∧ m ≤ 2) :
    f (m * x - 2) + f x < 0 := sorry

end NUMINAMATH_GPT_range_of_x_l2335_233560


namespace NUMINAMATH_GPT_inequality_solution_sets_l2335_233532

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 - 5 * x + b > 0 ↔ x < -1 / 3 ∨ x > 1 / 2) →
  (∀ x : ℝ, bx^2 - 5 * x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_sets_l2335_233532


namespace NUMINAMATH_GPT_ratio_of_prices_l2335_233505

-- Define the problem
theorem ratio_of_prices (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.2 * CP) 
  (h2 : SP2 = CP - 0.2 * CP) : 
  SP2 / SP1 = 2 / 3 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_ratio_of_prices_l2335_233505


namespace NUMINAMATH_GPT_roots_poly_sum_l2335_233533

noncomputable def Q (z : ℂ) (a b c : ℝ) : ℂ := z^3 + (a:ℂ)*z^2 + (b:ℂ)*z + (c:ℂ)

theorem roots_poly_sum (a b c : ℝ) (u : ℂ)
  (h1 : u.im = 0) -- Assuming u is a real number
  (h2 : Q (u + 5 * Complex.I) a b c = 0)
  (h3 : Q (u + 15 * Complex.I) a b c = 0)
  (h4 : Q (2 * u - 6) a b c = 0) :
  a + b + c = -196 := by
  sorry

end NUMINAMATH_GPT_roots_poly_sum_l2335_233533


namespace NUMINAMATH_GPT_triangle_problem_l2335_233503

noncomputable def length_of_side_c (a : ℝ) (cosB : ℝ) (C : ℝ) : ℝ :=
  a * (Real.sqrt 2 / 2) / (Real.sqrt (1 - cosB^2))

noncomputable def cos_A_minus_pi_over_6 (cosB : ℝ) (cosA : ℝ) (sinA : ℝ) : ℝ :=
  cosA * (Real.sqrt 3 / 2) + sinA * (1 / 2)

theorem triangle_problem (a : ℝ) (cosB : ℝ) (C : ℝ) 
  (ha : a = 6) (hcosB : cosB = 4/5) (hC : C = Real.pi / 4) : 
  (length_of_side_c a cosB C = 5 * Real.sqrt 2) ∧ 
  (cos_A_minus_pi_over_6 cosB (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2)))) (Real.sqrt (1 - (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2))))^2)) = (7 * Real.sqrt 2 - Real.sqrt 6) / 20) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_problem_l2335_233503


namespace NUMINAMATH_GPT_intersection_point_ordinate_interval_l2335_233580

theorem intersection_point_ordinate_interval:
  ∃ m : ℤ, ∀ x : ℝ, e ^ x = 5 - x → 3 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_GPT_intersection_point_ordinate_interval_l2335_233580


namespace NUMINAMATH_GPT_repeating_decimal_divisible_by_2_or_5_l2335_233541

theorem repeating_decimal_divisible_by_2_or_5 
    (m n : ℕ) 
    (x : ℝ) 
    (r s : ℕ) 
    (a b k p q u : ℕ)
    (hmn_coprime : Nat.gcd m n = 1)
    (h_rep_decimal : x = (m:ℚ) / (n:ℚ))
    (h_non_repeating_part: 0 < r) :
  n % 2 = 0 ∨ n % 5 = 0 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_divisible_by_2_or_5_l2335_233541


namespace NUMINAMATH_GPT_max_x_lcm_15_21_105_l2335_233567

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end NUMINAMATH_GPT_max_x_lcm_15_21_105_l2335_233567


namespace NUMINAMATH_GPT_probability_of_2_red_1_black_l2335_233578

theorem probability_of_2_red_1_black :
  let P_red := 4 / 7
  let P_black := 3 / 7 
  let prob_RRB := P_red * P_red * P_black 
  let prob_RBR := P_red * P_black * P_red 
  let prob_BRR := P_black * P_red * P_red 
  let total_prob := 3 * prob_RRB
  total_prob = 144 / 343 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_2_red_1_black_l2335_233578


namespace NUMINAMATH_GPT_limit_example_l2335_233596

theorem limit_example (ε : ℝ) (hε : 0 < ε) :
  ∃ δ : ℝ, 0 < δ ∧ 
  (∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε) :=
sorry -- The proof is not provided

end NUMINAMATH_GPT_limit_example_l2335_233596


namespace NUMINAMATH_GPT_find_square_digit_l2335_233538

-- Define the known sum of the digits 4, 7, 6, and 9
def sum_known_digits := 4 + 7 + 6 + 9

-- Define the condition that the number 47,69square must be divisible by 6
def is_multiple_of_6 (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∧ (sum_known_digits + d) % 3 = 0

-- Theorem statement that verifies both the conditions and finds possible values of square
theorem find_square_digit (d : ℕ) (h : is_multiple_of_6 d) : d = 4 ∨ d = 8 :=
by sorry

end NUMINAMATH_GPT_find_square_digit_l2335_233538


namespace NUMINAMATH_GPT_angle_B_eq_pi_over_3_range_of_area_l2335_233562

-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- And given vectors m and n represented as stated and are collinear
-- Prove that angle B is π/3
theorem angle_B_eq_pi_over_3
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (a b c A B C : ℝ)
  (m := (2 * Real.sin (A + C), - Real.sqrt 3))
  (n := (Real.cos (2 * B), 2 * Real.cos (B / 2) ^ 2 - 1))
  (collinear_m_n : m.1 * n.2 = m.2 * n.1) :
  B = Real.pi / 3 :=
sorry

-- Given side b = 1, find the range of the area S of triangle ABC
theorem range_of_area
  (a c A B C : ℝ)
  (triangle_area : ℝ)
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (hB : B = Real.pi / 3)
  (hb : b = 1)
  (cosine_theorem : 1 = a^2 + c^2 - a*c)
  (area_formula : triangle_area = (1/2) * a * c * Real.sin B) :
  0 < triangle_area ∧ triangle_area ≤ (Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_GPT_angle_B_eq_pi_over_3_range_of_area_l2335_233562


namespace NUMINAMATH_GPT_sqrt_of_expression_l2335_233515

theorem sqrt_of_expression :
  Real.sqrt (4^4 * 9^2) = 144 :=
sorry

end NUMINAMATH_GPT_sqrt_of_expression_l2335_233515


namespace NUMINAMATH_GPT_quadratic_solution_l2335_233544

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l2335_233544


namespace NUMINAMATH_GPT_alice_probability_at_least_one_multiple_of_4_l2335_233501

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end NUMINAMATH_GPT_alice_probability_at_least_one_multiple_of_4_l2335_233501


namespace NUMINAMATH_GPT_find_a_div_b_l2335_233556

theorem find_a_div_b (a b : ℝ) (h_distinct : a ≠ b) 
  (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_div_b_l2335_233556


namespace NUMINAMATH_GPT_area_of_ADC_l2335_233528

theorem area_of_ADC
  (BD DC : ℝ)
  (h_ratio : BD / DC = 2 / 3)
  (area_ABD : ℝ)
  (h_area_ABD : area_ABD = 30) :
  ∃ area_ADC, area_ADC = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_ADC_l2335_233528


namespace NUMINAMATH_GPT_book_sale_total_amount_l2335_233511

noncomputable def total_amount_received (total_books price_per_book : ℕ → ℝ) : ℝ :=
  price_per_book 80

theorem book_sale_total_amount (B : ℕ)
  (h1 : (1/3 : ℚ) * B = 40)
  (h2 : ∀ (n : ℕ), price_per_book n = 3.50) :
  total_amount_received B price_per_book = 280 := 
by
  sorry

end NUMINAMATH_GPT_book_sale_total_amount_l2335_233511


namespace NUMINAMATH_GPT_work_completion_in_days_l2335_233546

noncomputable def work_days_needed : ℕ :=
  let A_rate := 1 / 9
  let B_rate := 1 / 18
  let C_rate := 1 / 12
  let D_rate := 1 / 24
  let AB_rate := A_rate + B_rate
  let CD_rate := C_rate + D_rate
  let two_day_work := AB_rate + CD_rate
  let total_cycles := 24 / 7
  let total_days := (if total_cycles % 1 = 0 then total_cycles else total_cycles + 1) * 2
  total_days

theorem work_completion_in_days :
  work_days_needed = 8 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_in_days_l2335_233546


namespace NUMINAMATH_GPT_f_n_f_n_eq_n_l2335_233564

def f : ℕ → ℕ := sorry
axiom f_def1 : f 1 = 1
axiom f_def2 : ∀ n ≥ 2, f n = n - f (f (n - 1))

theorem f_n_f_n_eq_n (n : ℕ) (hn : 0 < n) : f (n + f n) = n :=
by sorry

end NUMINAMATH_GPT_f_n_f_n_eq_n_l2335_233564


namespace NUMINAMATH_GPT_min_value_fraction_l2335_233507

theorem min_value_fraction {x : ℝ} (h : x > 8) : 
    ∃ c : ℝ, (∀ y : ℝ, y = (x^2) / ((x - 8)^2) → c ≤ y) ∧ c = 1 := 
sorry

end NUMINAMATH_GPT_min_value_fraction_l2335_233507


namespace NUMINAMATH_GPT_find_x_value_l2335_233543

theorem find_x_value (x : ℝ) (h : 150 + 90 + x + 90 = 360) : x = 30 := by
  sorry

end NUMINAMATH_GPT_find_x_value_l2335_233543


namespace NUMINAMATH_GPT_find_an_from_sums_l2335_233548

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end NUMINAMATH_GPT_find_an_from_sums_l2335_233548


namespace NUMINAMATH_GPT_probability_triangle_side_decagon_l2335_233534

theorem probability_triangle_side_decagon (total_vertices : ℕ) (choose_vertices : ℕ)
  (total_triangles : ℕ) (favorable_outcomes : ℕ)
  (triangle_formula : total_vertices = 10)
  (choose_vertices_formula : choose_vertices = 3)
  (total_triangle_count_formula : total_triangles = 120)
  (favorable_outcome_count_formula : favorable_outcomes = 70)
  : (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_probability_triangle_side_decagon_l2335_233534


namespace NUMINAMATH_GPT_recurrence_sequence_a5_l2335_233559

theorem recurrence_sequence_a5 :
  ∃ a : ℕ → ℚ, (a 1 = 5 ∧ (∀ n, a (n + 1) = 1 + 1 / a n) ∧ a 5 = 28 / 17) :=
  sorry

end NUMINAMATH_GPT_recurrence_sequence_a5_l2335_233559


namespace NUMINAMATH_GPT_sharp_sharp_sharp_20_l2335_233594

def sharp (N : ℝ) : ℝ := (0.5 * N)^2 + 1

theorem sharp_sharp_sharp_20 : sharp (sharp (sharp 20)) = 1627102.64 :=
by
  sorry

end NUMINAMATH_GPT_sharp_sharp_sharp_20_l2335_233594


namespace NUMINAMATH_GPT_compute_expression_l2335_233519

theorem compute_expression :
  120 * 2400 - 20 * 2400 - 100 * 2400 = 0 :=
sorry

end NUMINAMATH_GPT_compute_expression_l2335_233519


namespace NUMINAMATH_GPT_no_integer_roots_l2335_233539

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 11 * x + 20 ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l2335_233539


namespace NUMINAMATH_GPT_smallest_nat_number_l2335_233553

theorem smallest_nat_number : ∃ a : ℕ, (a % 3 = 2) ∧ (a % 5 = 4) ∧ (a % 7 = 4) ∧ (∀ b : ℕ, (b % 3 = 2) ∧ (b % 5 = 4) ∧ (b % 7 = 4) → a ≤ b) ∧ a = 74 := 
sorry

end NUMINAMATH_GPT_smallest_nat_number_l2335_233553


namespace NUMINAMATH_GPT_probability_prime_ball_l2335_233542

open Finset

theorem probability_prime_ball :
  let balls := {1, 2, 3, 4, 5, 6, 8, 9}
  let total := card balls
  let primes := {2, 3, 5}
  let primes_count := card primes
  (total = 8) → (primes ⊆ balls) → 
  primes_count = 3 → 
  primes_count / total = 3 / 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_prime_ball_l2335_233542


namespace NUMINAMATH_GPT_jake_eats_papayas_in_one_week_l2335_233582

variable (J : ℕ)
variable (brother_eats : ℕ := 5)
variable (father_eats : ℕ := 4)
variable (total_papayas_in_4_weeks : ℕ := 48)

theorem jake_eats_papayas_in_one_week (h : 4 * (J + brother_eats + father_eats) = total_papayas_in_4_weeks) : J = 3 :=
by
  sorry

end NUMINAMATH_GPT_jake_eats_papayas_in_one_week_l2335_233582


namespace NUMINAMATH_GPT_probability_of_exactly_one_success_probability_of_at_least_one_success_l2335_233568

variable (PA : ℚ := 1/2)
variable (PB : ℚ := 2/5)
variable (P_A_bar : ℚ := 1 - PA)
variable (P_B_bar : ℚ := 1 - PB)

theorem probability_of_exactly_one_success :
  PA * P_B_bar + PB * P_A_bar = 1/2 :=
sorry

theorem probability_of_at_least_one_success :
  1 - (P_A_bar * P_A_bar * P_B_bar * P_B_bar) = 91/100 :=
sorry

end NUMINAMATH_GPT_probability_of_exactly_one_success_probability_of_at_least_one_success_l2335_233568


namespace NUMINAMATH_GPT_yangyang_helps_mom_for_5_days_l2335_233571

-- Defining the conditions
def quantity_of_rice_in_warehouses_are_same : Prop := sorry
def dad_transports_all_rice_in : ℕ := 10
def mom_transports_all_rice_in : ℕ := 12
def yangyang_transports_all_rice_in : ℕ := 15
def dad_and_mom_start_at_same_time : Prop := sorry
def yangyang_helps_mom_then_dad : Prop := sorry
def finish_transporting_at_same_time : Prop := sorry

-- The theorem to prove
theorem yangyang_helps_mom_for_5_days (h1 : quantity_of_rice_in_warehouses_are_same) 
    (h2 : dad_and_mom_start_at_same_time) 
    (h3 : yangyang_helps_mom_then_dad) 
    (h4 : finish_transporting_at_same_time) : 
    yangyang_helps_mom_then_dad :=
sorry

end NUMINAMATH_GPT_yangyang_helps_mom_for_5_days_l2335_233571


namespace NUMINAMATH_GPT_diagonals_of_square_equal_proof_l2335_233554

-- Let us define the conditions
def square (s : Type) : Prop := True -- Placeholder for the actual definition of square
def parallelogram (p : Type) : Prop := True -- Placeholder for the actual definition of parallelogram
def diagonals_equal (q : Type) : Prop := True -- Placeholder for the property that diagonals are equal

-- Given conditions
axiom square_is_parallelogram {s : Type} (h1 : square s) : parallelogram s
axiom diagonals_of_parallelogram_equal {p : Type} (h2 : parallelogram p) : diagonals_equal p
axiom diagonals_of_square_equal {s : Type} (h3 : square s) : diagonals_equal s

-- Proof statement
theorem diagonals_of_square_equal_proof (s : Type) (h1 : square s) : diagonals_equal s :=
by
  apply diagonals_of_square_equal h1

end NUMINAMATH_GPT_diagonals_of_square_equal_proof_l2335_233554


namespace NUMINAMATH_GPT_gcd_le_two_l2335_233555

theorem gcd_le_two (a m n : ℕ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : Odd n) :
  Nat.gcd (a^n - 1) (a^m + 1) ≤ 2 := 
sorry

end NUMINAMATH_GPT_gcd_le_two_l2335_233555


namespace NUMINAMATH_GPT_l_shaped_tile_rectangle_multiple_of_8_l2335_233575

theorem l_shaped_tile_rectangle_multiple_of_8 (m n : ℕ) 
  (h : ∃ k : ℕ, 4 * k = m * n) : ∃ s : ℕ, m * n = 8 * s :=
by
  sorry

end NUMINAMATH_GPT_l_shaped_tile_rectangle_multiple_of_8_l2335_233575


namespace NUMINAMATH_GPT_min_value_x_l2335_233527

theorem min_value_x (x : ℝ) (h : ∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) : x ≥ -1 := 
sorry

end NUMINAMATH_GPT_min_value_x_l2335_233527


namespace NUMINAMATH_GPT_group_B_fluctuates_less_l2335_233523

-- Conditions
def mean_A : ℝ := 80
def mean_B : ℝ := 90
def variance_A : ℝ := 10
def variance_B : ℝ := 5

-- Goal
theorem group_B_fluctuates_less :
  variance_B < variance_A :=
  by
    sorry

end NUMINAMATH_GPT_group_B_fluctuates_less_l2335_233523


namespace NUMINAMATH_GPT_medical_team_combinations_l2335_233593

-- Number of combinations function
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem medical_team_combinations :
  let maleDoctors := 6
  let femaleDoctors := 5
  let requiredMale := 2
  let requiredFemale := 1
  choose maleDoctors requiredMale * choose femaleDoctors requiredFemale = 75 :=
by
  sorry

end NUMINAMATH_GPT_medical_team_combinations_l2335_233593


namespace NUMINAMATH_GPT_calculate_sheep_l2335_233581

-- Conditions as definitions
def cows : Nat := 24
def goats : Nat := 113
def total_animals_to_transport (groups size_per_group : Nat) : Nat := groups * size_per_group
def cows_and_goats (cows goats : Nat) : Nat := cows + goats

-- The problem statement: Calculate the number of sheep such that the total number of animals matches the target.
theorem calculate_sheep
  (groups : Nat) (size_per_group : Nat) (cows goats : Nat) (transportation_total animals_present : Nat) 
  (h1 : groups = 3) (h2 : size_per_group = 48) (h3 : cows = 24) (h4 : goats = 113) 
  (h5 : animals_present = cows + goats) (h6 : transportation_total = groups * size_per_group) :
  transportation_total - animals_present = 7 :=
by 
  -- To be proven 
  sorry

end NUMINAMATH_GPT_calculate_sheep_l2335_233581


namespace NUMINAMATH_GPT_no_four_consecutive_lucky_numbers_l2335_233551

def is_lucky (n : ℕ) : Prop :=
  let digits := n.digits 10
  n > 999999 ∧ n < 10000000 ∧ (∀ d ∈ digits, d ≠ 0) ∧ 
  n % (digits.foldl (λ x y => x * y) 1) = 0

theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end NUMINAMATH_GPT_no_four_consecutive_lucky_numbers_l2335_233551


namespace NUMINAMATH_GPT_find_whole_number_M_l2335_233589

theorem find_whole_number_M (M : ℕ) (h : 8 < M / 4 ∧ M / 4 < 9) : M = 33 :=
sorry

end NUMINAMATH_GPT_find_whole_number_M_l2335_233589


namespace NUMINAMATH_GPT_proof_a_in_S_l2335_233508

def S : Set ℤ := {n : ℤ | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem proof_a_in_S (a : ℤ) (h1 : 3 * a ∈ S) : a ∈ S :=
sorry

end NUMINAMATH_GPT_proof_a_in_S_l2335_233508


namespace NUMINAMATH_GPT_anna_cupcakes_remaining_l2335_233537

theorem anna_cupcakes_remaining :
  let total_cupcakes := 60
  let cupcakes_given_away := (4 / 5 : ℝ) * total_cupcakes
  let cupcakes_after_giving := total_cupcakes - cupcakes_given_away
  let cupcakes_eaten := 3
  let cupcakes_left := cupcakes_after_giving - cupcakes_eaten
  cupcakes_left = 9 :=
by
  sorry

end NUMINAMATH_GPT_anna_cupcakes_remaining_l2335_233537


namespace NUMINAMATH_GPT_julie_read_yesterday_l2335_233585

variable (x : ℕ)
variable (y : ℕ := 2 * x)
variable (remaining_pages_after_two_days : ℕ := 120 - (x + y))

theorem julie_read_yesterday :
  (remaining_pages_after_two_days / 2 = 42) -> (x = 12) :=
by
  sorry

end NUMINAMATH_GPT_julie_read_yesterday_l2335_233585


namespace NUMINAMATH_GPT_bananas_per_truck_l2335_233561

theorem bananas_per_truck (total_apples total_bananas apples_per_truck : ℝ) 
  (h_total_apples: total_apples = 132.6)
  (h_apples_per_truck: apples_per_truck = 13.26)
  (h_total_bananas: total_bananas = 6.4) :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 :=
by
  sorry

end NUMINAMATH_GPT_bananas_per_truck_l2335_233561


namespace NUMINAMATH_GPT_cos_alpha_plus_5pi_over_4_eq_16_over_65_l2335_233518

theorem cos_alpha_plus_5pi_over_4_eq_16_over_65
  (α β : ℝ)
  (hα : -π / 4 < α ∧ α < 0)
  (hβ : π / 2 < β ∧ β < π)
  (hcos_sum : Real.cos (α + β) = -4/5)
  (hcos_diff : Real.cos (β - π / 4) = 5/13) :
  Real.cos (α + 5 * π / 4) = 16/65 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_5pi_over_4_eq_16_over_65_l2335_233518


namespace NUMINAMATH_GPT_total_baseball_cards_l2335_233530

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_baseball_cards_l2335_233530


namespace NUMINAMATH_GPT_fraction_subtraction_l2335_233514

theorem fraction_subtraction (a b : ℚ) (h_a: a = 5/9) (h_b: b = 1/6) : a - b = 7/18 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l2335_233514


namespace NUMINAMATH_GPT_point_outside_circle_l2335_233587

theorem point_outside_circle (a : ℝ) :
  (a > 1) → (a, a) ∉ {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 2 * a * p.1 + a^2 - a = 0} :=
by sorry

end NUMINAMATH_GPT_point_outside_circle_l2335_233587


namespace NUMINAMATH_GPT_pow_div_mul_pow_eq_l2335_233536

theorem pow_div_mul_pow_eq (a b c d : ℕ) (h_a : a = 8) (h_b : b = 5) (h_c : c = 2) (h_d : d = 6) :
  (a^b / a^c) * (4^6) = 2^21 := by
  sorry

end NUMINAMATH_GPT_pow_div_mul_pow_eq_l2335_233536


namespace NUMINAMATH_GPT_total_students_in_classes_l2335_233557

theorem total_students_in_classes (t1 t2 x y: ℕ) (h1 : t1 = 273) (h2 : t2 = 273) (h3 : (x - 1) * 7 = t1) (h4 : (y - 1) * 13 = t2) : x + y = 62 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_classes_l2335_233557


namespace NUMINAMATH_GPT_casper_candy_problem_l2335_233504

theorem casper_candy_problem (o y gr : ℕ) (n : ℕ) (h1 : 10 * o = 16 * y) (h2 : 16 * y = 18 * gr) (h3 : 18 * gr = 18 * n) :
    n = 40 :=
by
  sorry

end NUMINAMATH_GPT_casper_candy_problem_l2335_233504


namespace NUMINAMATH_GPT_geometric_series_sum_l2335_233588

def first_term : ℤ := 3
def common_ratio : ℤ := -2
def last_term : ℤ := -1536
def num_terms : ℕ := 10
def sum_of_series (a r : ℤ) (n : ℕ) : ℤ := a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum :
  sum_of_series first_term common_ratio num_terms = -1023 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2335_233588


namespace NUMINAMATH_GPT_range_of_m_for_nonnegative_quadratic_l2335_233502

-- The statement of the proof problem in Lean
theorem range_of_m_for_nonnegative_quadratic {x m : ℝ} : 
  (∀ x, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_nonnegative_quadratic_l2335_233502


namespace NUMINAMATH_GPT_steve_berry_picking_strategy_l2335_233597

def berry_picking_goal_reached (monday_earnings tuesday_earnings total_goal: ℕ) : Prop :=
  monday_earnings + tuesday_earnings >= total_goal

def optimal_thursday_strategy (remaining_goal payment_per_pound total_capacity : ℕ) : ℕ :=
  if remaining_goal = 0 then 0 else total_capacity

theorem steve_berry_picking_strategy :
  let monday_lingonberries := 8
  let monday_cloudberries := 10
  let monday_blueberries := 30 - monday_lingonberries - monday_cloudberries
  let tuesday_lingonberries := 3 * monday_lingonberries
  let tuesday_cloudberries := 2 * monday_cloudberries
  let tuesday_blueberries := 5
  let lingonberry_rate := 2
  let cloudberry_rate := 3
  let blueberry_rate := 5
  let max_capacity := 30
  let total_goal := 150

  let monday_earnings := (monday_lingonberries * lingonberry_rate) + 
                         (monday_cloudberries * cloudberry_rate) + 
                         (monday_blueberries * blueberry_rate)
                         
  let tuesday_earnings := (tuesday_lingonberries * lingonberry_rate) + 
                          (tuesday_cloudberries * cloudberry_rate) +
                          (tuesday_blueberries * blueberry_rate)

  let total_earnings := monday_earnings + tuesday_earnings

  berry_picking_goal_reached monday_earnings tuesday_earnings total_goal ∧
  optimal_thursday_strategy (total_goal - total_earnings) blueberry_rate max_capacity = 30 
:= by {
  sorry
}

end NUMINAMATH_GPT_steve_berry_picking_strategy_l2335_233597


namespace NUMINAMATH_GPT_minimize_quadratic_l2335_233591

theorem minimize_quadratic : ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, (y - 6)^2 ≥ (6 - 6)^2 := by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l2335_233591


namespace NUMINAMATH_GPT_total_marbles_l2335_233579

variable (r : ℝ) -- number of red marbles
variable (b g y : ℝ) -- number of blue, green, and yellow marbles

-- Conditions
axiom h1 : r = 1.3 * b
axiom h2 : g = 1.5 * r
axiom h3 : y = 0.8 * g

/-- Theorem: The total number of marbles in the collection is 4.47 times the number of red marbles -/
theorem total_marbles (r b g y : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.5 * r) (h3 : y = 0.8 * g) :
  b + r + g + y = 4.47 * r :=
sorry

end NUMINAMATH_GPT_total_marbles_l2335_233579


namespace NUMINAMATH_GPT_min_y_squared_isosceles_trapezoid_l2335_233540

theorem min_y_squared_isosceles_trapezoid:
  ∀ (EF GH y : ℝ) (circle_center : ℝ)
    (isosceles_trapezoid : Prop)
    (tangent_EH : Prop)
    (tangent_FG : Prop),
  isosceles_trapezoid ∧ EF = 72 ∧ GH = 45 ∧ EH = y ∧ FG = y ∧
  (∃ (circle : ℝ), circle_center = (EF / 2) ∧ tangent_EH ∧ tangent_FG)
  → y^2 = 486 :=
by sorry

end NUMINAMATH_GPT_min_y_squared_isosceles_trapezoid_l2335_233540


namespace NUMINAMATH_GPT_sample_older_employees_count_l2335_233598

-- Definitions of known quantities
def N := 400
def N_older := 160
def N_no_older := 240
def n := 50

-- The proof statement showing that the number of employees older than 45 in the sample equals 20
theorem sample_older_employees_count : 
  let proportion_older := (N_older:ℝ) / (N:ℝ)
  let n_older := proportion_older * (n:ℝ)
  n_older = 20 := by
  sorry

end NUMINAMATH_GPT_sample_older_employees_count_l2335_233598


namespace NUMINAMATH_GPT_sum_series_l2335_233500

noncomputable def f (n : ℕ) : ℝ :=
  (6 * (n : ℝ)^3 - 3 * (n : ℝ)^2 + 2 * (n : ℝ) - 1) / 
  ((n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ)^2 + (n : ℝ) + 1) * ((n : ℝ)^2 - (n : ℝ) + 1))

theorem sum_series:
  (∑' n, if h : 2 ≤ n then f n else 0) = 1 := 
by
  sorry

end NUMINAMATH_GPT_sum_series_l2335_233500
