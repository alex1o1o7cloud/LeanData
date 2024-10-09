import Mathlib

namespace min_value_expression_l1603_160349

theorem min_value_expression (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 21) ∧ 
           (∀ z : ℝ, (z = (x + 18) / Real.sqrt (x - 3)) → y ≤ z) := 
sorry

end min_value_expression_l1603_160349


namespace health_risk_factor_prob_l1603_160388

noncomputable def find_p_q_sum (p q: ℕ) : ℕ :=
if h1 : p.gcd q = 1 then
  31
else 
  sorry

theorem health_risk_factor_prob (p q : ℕ) (h1 : p.gcd q = 1) 
                                (h2 : (p : ℚ) / q = 5 / 26) :
  find_p_q_sum p q = 31 :=
sorry

end health_risk_factor_prob_l1603_160388


namespace find_cd_product_l1603_160361

open Complex

theorem find_cd_product :
  let u : ℂ := -3 + 4 * I
  let v : ℂ := 2 - I
  let c : ℂ := -5 + 5 * I
  let d : ℂ := -5 - 5 * I
  c * d = 50 :=
by
  sorry

end find_cd_product_l1603_160361


namespace distance_between_foci_l1603_160322

theorem distance_between_foci (x y : ℝ)
    (h : 2 * x^2 - 12 * x - 8 * y^2 + 16 * y = 100) :
    2 * Real.sqrt 68.75 =
    2 * Real.sqrt (55 + 13.75) :=
by
  sorry

end distance_between_foci_l1603_160322


namespace perimeter_of_square_l1603_160311

/-- The perimeter of a square with side length 15 cm is 60 cm -/
theorem perimeter_of_square (side_length : ℝ) (area : ℝ) (h1 : side_length = 15) (h2 : area = 225) :
  (4 * side_length = 60) :=
by
  -- Proof steps would go here (omitted)
  sorry

end perimeter_of_square_l1603_160311


namespace value_of_a1_plus_a3_l1603_160338

theorem value_of_a1_plus_a3 (a a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) →
  a1 + a3 = 8 :=
by
  sorry

end value_of_a1_plus_a3_l1603_160338


namespace problem_1_problem_2_problem_3_l1603_160317

-- Definition for question 1:
def gcd_21n_4_14n_3 (n : ℕ) : Prop := (Nat.gcd (21 * n + 4) (14 * n + 3)) = 1

-- Definition for question 2:
def gcd_n_factorial_plus_1 (n : ℕ) : Prop := (Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1)) = 1

-- Definition for question 3:
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1
def gcd_fermat_numbers (m n : ℕ) (h : m ≠ n) : Prop := (Nat.gcd (fermat_number m) (fermat_number n)) = 1

-- Theorem statements
theorem problem_1 (n : ℕ) (h_pos : 0 < n) : gcd_21n_4_14n_3 n := sorry

theorem problem_2 (n : ℕ) (h_pos : 0 < n) : gcd_n_factorial_plus_1 n := sorry

theorem problem_3 (m n : ℕ) (h_pos1 : 0 ≠ m) (h_pos2 : 0 ≠ n) (h_neq : m ≠ n) : gcd_fermat_numbers m n h_neq := sorry

end problem_1_problem_2_problem_3_l1603_160317


namespace function_symmetry_extremum_l1603_160304

noncomputable def f (x θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

theorem function_symmetry_extremum {θ : ℝ} (H : ∀ x : ℝ, f x θ = f (2 - x) θ) : 
  f 1 θ = 3 ∨ f 1 θ = -3 :=
by
  sorry

end function_symmetry_extremum_l1603_160304


namespace probability_difference_l1603_160399

theorem probability_difference (red_marbles black_marbles : ℤ) (h_red : red_marbles = 1500) (h_black : black_marbles = 1500) :
  |(22485 / 44985 : ℚ) - (22500 / 44985 : ℚ)| = 15 / 44985 := 
by {
  sorry
}

end probability_difference_l1603_160399


namespace maggie_earnings_correct_l1603_160352

def subscriptions_sold_to_parents : ℕ := 4
def subscriptions_sold_to_grandfather : ℕ := 1
def subscriptions_sold_to_next_door_neighbor : ℕ := 2
def subscriptions_sold_to_another_neighbor : ℕ := 2 * subscriptions_sold_to_next_door_neighbor
def price_per_subscription : ℕ := 5
def family_bonus_per_subscription : ℕ := 2
def neighbor_bonus_per_subscription : ℕ := 1
def base_bonus_threshold : ℕ := 10
def base_bonus : ℕ := 10
def extra_bonus_per_subscription : ℝ := 0.5

-- Define total subscriptions sold
def total_subscriptions_sold : ℕ := 
  subscriptions_sold_to_parents + subscriptions_sold_to_grandfather + 
  subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor

-- Define earnings from subscriptions
def earnings_from_subscriptions : ℕ := total_subscriptions_sold * price_per_subscription

-- Define bonuses
def family_bonus : ℕ :=
  (subscriptions_sold_to_parents + subscriptions_sold_to_grandfather) * family_bonus_per_subscription

def neighbor_bonus : ℕ := 
  (subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor) * neighbor_bonus_per_subscription

def total_bonus : ℕ := family_bonus + neighbor_bonus

-- Define additional boss bonus
def additional_boss_bonus : ℝ := 
  if total_subscriptions_sold > base_bonus_threshold then 
    base_bonus + extra_bonus_per_subscription * (total_subscriptions_sold - base_bonus_threshold) 
  else 0

-- Define total earnings
def total_earnings : ℝ :=
  earnings_from_subscriptions + total_bonus + additional_boss_bonus

-- Theorem statement
theorem maggie_earnings_correct : total_earnings = 81.5 :=
by
  unfold total_earnings
  unfold earnings_from_subscriptions
  unfold total_bonus
  unfold family_bonus
  unfold neighbor_bonus
  unfold additional_boss_bonus
  unfold total_subscriptions_sold
  simp
  norm_cast
  sorry

end maggie_earnings_correct_l1603_160352


namespace tangent_line_to_circle_l1603_160371

-- Definitions derived directly from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0
def passes_through_point (l : ℝ → ℝ → Prop) : Prop := l (-1) 6

-- The statement to be proven
theorem tangent_line_to_circle :
  ∃ (l : ℝ → ℝ → Prop), passes_through_point l ∧ 
    ((∀ x y, l x y ↔ 3*x - 4*y + 27 = 0) ∨ 
     (∀ x y, l x y ↔ x + 1 = 0)) :=
sorry

end tangent_line_to_circle_l1603_160371


namespace mushroom_collection_l1603_160327

variable (a b v g : ℕ)

theorem mushroom_collection : 
  (a / 2 + 2 * b = v + g) ∧ (a + b = v / 2 + 2 * g) → (v = 2 * b) ∧ (a = 2 * g) :=
by
  sorry

end mushroom_collection_l1603_160327


namespace rojas_speed_l1603_160337

theorem rojas_speed (P R : ℝ) (h1 : P = 3) (h2 : 4 * (R + P) = 28) : R = 4 :=
by
  sorry

end rojas_speed_l1603_160337


namespace symmetrical_point_of_P_is_correct_l1603_160315

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the function to get the symmetric point with respect to the origin
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Prove that the symmetrical point of P with respect to the origin is (1, -2)
theorem symmetrical_point_of_P_is_correct : symmetrical_point P = (1, -2) :=
  sorry

end symmetrical_point_of_P_is_correct_l1603_160315


namespace regular_octagon_interior_angle_l1603_160351

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l1603_160351


namespace problem_l1603_160346

theorem problem (d r : ℕ) (a b c : ℕ) (ha : a = 1059) (hb : b = 1417) (hc : c = 2312)
  (h1 : d ∣ (b - a)) (h2 : d ∣ (c - a)) (h3 : d ∣ (c - b)) (hd : d > 1)
  (hr : r = a % d):
  d - r = 15 := sorry

end problem_l1603_160346


namespace problem_statement_l1603_160339

theorem problem_statement (a b : ℝ) (h : 3 * a - 2 * b = -1) : 3 * a - 2 * b + 2024 = 2023 :=
by
  sorry

end problem_statement_l1603_160339


namespace cost_of_each_box_of_pencils_l1603_160335

-- Definitions based on conditions
def cartons_of_pencils : ℕ := 20
def boxes_per_carton_of_pencils : ℕ := 10
def cartons_of_markers : ℕ := 10
def boxes_per_carton_of_markers : ℕ := 5
def cost_per_carton_of_markers : ℕ := 4
def total_spent : ℕ := 600

-- Variable to define cost per box of pencils
variable (P : ℝ)

-- Main theorem to prove
theorem cost_of_each_box_of_pencils :
  cartons_of_pencils * boxes_per_carton_of_pencils * P + 
  cartons_of_markers * cost_per_carton_of_markers = total_spent → 
  P = 2.80 :=
by
  sorry

end cost_of_each_box_of_pencils_l1603_160335


namespace solve_quadratic_l1603_160355

theorem solve_quadratic (x : ℝ) (h1 : 2 * x ^ 2 = 9 * x - 4) (h2 : x ≠ 4) : 2 * x = 1 :=
by
  -- The proof will go here
  sorry

end solve_quadratic_l1603_160355


namespace total_cost_alex_had_to_pay_l1603_160302

def baseCost : ℝ := 30
def costPerText : ℝ := 0.04 -- 4 cents in dollars
def textsSent : ℕ := 150
def costPerMinuteOverLimit : ℝ := 0.15 -- 15 cents in dollars
def hoursUsed : ℝ := 26
def freeHours : ℝ := 25

def totalCost : ℝ :=
  baseCost + (costPerText * textsSent) + (costPerMinuteOverLimit * (hoursUsed - freeHours) * 60)

theorem total_cost_alex_had_to_pay :
  totalCost = 45 := by
  sorry

end total_cost_alex_had_to_pay_l1603_160302


namespace determine_common_ratio_l1603_160368

-- Definition of geometric sequence and sum of first n terms
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_geometric_sequence (a : ℕ → ℝ) : ℕ → ℝ
  | 0       => a 0
  | (n + 1) => a (n + 1) + sum_geometric_sequence a n

-- Main theorem
theorem determine_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : is_geometric_sequence a q)
  (h3 : ∀ n, S n = sum_geometric_sequence a n)
  (h4 : 3 * (S 2 + a 2 + a 1 * q^2) = 8 * a 1 * q + 5 * a 1) :
  q = 2 :=
by 
  sorry

end determine_common_ratio_l1603_160368


namespace Andrews_age_l1603_160300

theorem Andrews_age (a g : ℝ) (h1 : g = 15 * a) (h2 : g - a = 55) : a = 55 / 14 :=
by
  /- proof will go here -/
  sorry

end Andrews_age_l1603_160300


namespace Karlson_max_candies_l1603_160379

theorem Karlson_max_candies (f : Fin 25 → ℕ) (g : Fin 25 → Fin 25 → ℕ) :
  (∀ i, f i = 1) →
  (∀ i j, g i j = f i * f j) →
  (∃ (S : ℕ), S = 300) :=
by
  intros h1 h2
  sorry

end Karlson_max_candies_l1603_160379


namespace remainder_three_n_l1603_160391

theorem remainder_three_n (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 :=
by
  sorry

end remainder_three_n_l1603_160391


namespace paul_tickets_left_l1603_160396

theorem paul_tickets_left (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) :
  initial_tickets = 11 → spent_tickets = 3 → remaining_tickets = initial_tickets - spent_tickets → remaining_tickets = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end paul_tickets_left_l1603_160396


namespace range_of_m_l1603_160362

theorem range_of_m (m : ℝ) : (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x < 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y < 0) ↔ 0 < m ∧ m < 2 :=
by sorry

end range_of_m_l1603_160362


namespace fixed_point_of_transformed_logarithmic_function_l1603_160341

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_a (a : ℝ) (x : ℝ) : ℝ := 1 + log_a a (x - 1)

theorem fixed_point_of_transformed_logarithmic_function
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1) : f_a a 2 = 1 :=
by
  -- Prove the theorem using given conditions
  sorry

end fixed_point_of_transformed_logarithmic_function_l1603_160341


namespace janes_score_l1603_160386

theorem janes_score (jane_score tom_score : ℕ) (h1 : jane_score = tom_score + 50) (h2 : (jane_score + tom_score) / 2 = 90) :
  jane_score = 115 :=
sorry

end janes_score_l1603_160386


namespace arithmetic_seq_sum_ratio_l1603_160330

theorem arithmetic_seq_sum_ratio
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : S 25 / a 23 = 5)
  (h3 : S 45 / a 33 = 25) :
  S 65 / a 43 = 45 :=
by sorry

end arithmetic_seq_sum_ratio_l1603_160330


namespace set_equality_l1603_160381

open Set

namespace Proof

variables (U M N : Set ℕ) 
variables (U_univ : U = {1, 2, 3, 4, 5, 6})
variables (M_set : M = {2, 3})
variables (N_set : N = {1, 3})

theorem set_equality :
  {4, 5, 6} = (U \ M) ∩ (U \ N) :=
by
  rw [U_univ, M_set, N_set]
  sorry

end Proof

end set_equality_l1603_160381


namespace four_pow_expression_l1603_160384

theorem four_pow_expression : 4 ^ (3 ^ 2) / (4 ^ 3) ^ 2 = 64 := by
  sorry

end four_pow_expression_l1603_160384


namespace regular_polygon_sides_l1603_160343

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l1603_160343


namespace age_of_B_l1603_160377

-- Define the ages of A and B
variables (A B : ℕ)

-- The conditions given in the problem
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 9

theorem age_of_B (A B : ℕ) (h1 : condition1 A B) (h2 : condition2 A B) : B = 39 :=
by
  sorry

end age_of_B_l1603_160377


namespace thieves_cloth_equation_l1603_160319

theorem thieves_cloth_equation (x y : ℤ) 
  (h1 : y = 6 * x + 5)
  (h2 : y = 7 * x - 8) :
  6 * x + 5 = 7 * x - 8 :=
by
  sorry

end thieves_cloth_equation_l1603_160319


namespace find_m_l1603_160393

theorem find_m (m : ℤ) (h0 : -90 ≤ m) (h1 : m ≤ 90) (h2 : Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180)) : m = -10 :=
sorry

end find_m_l1603_160393


namespace parabola_min_value_incorrect_statement_l1603_160380

theorem parabola_min_value_incorrect_statement
  (m : ℝ)
  (A B : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (parabola : ℝ → ℝ)
  (on_parabola : ∀ (x : ℝ), parabola x = x^2 - 2*m*x + m^2 - 9)
  (A_intersects_x_axis : A.2 = 0)
  (B_intersects_x_axis : B.2 = 0)
  (A_on_parabola : parabola A.1 = A.2)
  (B_on_parabola : parabola B.1 = B.2)
  (P_on_parabola : parabola P.1 = P.2)
  (Q_on_parabola : parabola Q.1 = Q.2)
  (P_coordinates : P = (m + 1, parabola (m + 1)))
  (Q_coordinates : Q = (m - 3, parabola (m - 3))) :
  ∃ (min_y : ℝ), min_y = -9 ∧ min_y ≠ m^2 - 9 := 
sorry

end parabola_min_value_incorrect_statement_l1603_160380


namespace roots_of_quadratic_l1603_160324

theorem roots_of_quadratic (a b c : ℝ) (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ¬ ∃ (x : ℝ), x^2 + (a + b + c) * x + a^2 + b^2 + c^2 = 0 :=
by
  sorry

end roots_of_quadratic_l1603_160324


namespace example_theorem_l1603_160397

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l1603_160397


namespace boat_travel_l1603_160326

theorem boat_travel (T_against T_with : ℝ) (V_b D V_c : ℝ) 
  (hT_against : T_against = 10) 
  (hT_with : T_with = 6) 
  (hV_b : V_b = 12)
  (hD1 : D = (V_b - V_c) * T_against)
  (hD2 : D = (V_b + V_c) * T_with) :
  V_c = 3 ∧ D = 90 :=
by
  sorry

end boat_travel_l1603_160326


namespace park_cycling_time_l1603_160331

def length_breadth_ratio (L B : ℕ) : Prop := L / B = 1 / 3
def area_of_park (L B : ℕ) : Prop := L * B = 120000
def speed_of_cyclist : ℕ := 200 -- meters per minute
def perimeter (L B : ℕ) : ℕ := 2 * L + 2 * B
def time_to_complete_round (P v : ℕ) : ℕ := P / v

theorem park_cycling_time
  (L B : ℕ)
  (h_ratio : length_breadth_ratio L B)
  (h_area : area_of_park L B)
  : time_to_complete_round (perimeter L B) speed_of_cyclist = 8 :=
by
  sorry

end park_cycling_time_l1603_160331


namespace animal_shelter_kittens_count_l1603_160333

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end animal_shelter_kittens_count_l1603_160333


namespace unique_positive_real_solution_l1603_160357

-- Define the function
def f (x : ℝ) : ℝ := x^11 + 9 * x^10 + 19 * x^9 + 2023 * x^8 - 1421 * x^7 + 5

-- Prove the statement
theorem unique_positive_real_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end unique_positive_real_solution_l1603_160357


namespace sum_cubes_identity_l1603_160398

theorem sum_cubes_identity (x y z : ℝ) (h1 : x + y + z = 10) (h2 : xy + yz + zx = 20) :
    x^3 + y^3 + z^3 - 3 * x * y * z = 400 := by
  sorry

end sum_cubes_identity_l1603_160398


namespace geom_seq_a1_l1603_160321

-- Define a geometric sequence.
def geom_seq (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q ^ n

-- Given conditions
def a2 (a : ℕ → ℝ) : Prop := a 1 = 2 -- because a2 = a(1) in zero-indexed
def a5 (a : ℕ → ℝ) : Prop := a 4 = -54 -- because a5 = a(4) in zero-indexed

-- Prove that a1 = -2/3
theorem geom_seq_a1 (a : ℕ → ℝ) (a1 q : ℝ) (h_geom : geom_seq a a1 q)
  (h_a2 : a2 a) (h_a5 : a5 a) : a1 = -2 / 3 :=
by
  sorry

end geom_seq_a1_l1603_160321


namespace books_returned_wednesday_correct_l1603_160305

def initial_books : Nat := 250
def books_taken_out_Tuesday : Nat := 120
def books_taken_out_Thursday : Nat := 15
def books_remaining_after_Thursday : Nat := 150

def books_after_tuesday := initial_books - books_taken_out_Tuesday
def books_before_thursday := books_remaining_after_Thursday + books_taken_out_Thursday
def books_returned_wednesday := books_before_thursday - books_after_tuesday

theorem books_returned_wednesday_correct : books_returned_wednesday = 35 := by
  sorry

end books_returned_wednesday_correct_l1603_160305


namespace calculate_f_of_f_of_f_l1603_160395

def f (x : ℤ) : ℤ := 5 * x - 4

theorem calculate_f_of_f_of_f (h : f (f (f 3)) = 251) : f (f (f 3)) = 251 := 
by sorry

end calculate_f_of_f_of_f_l1603_160395


namespace negation_of_forall_statement_l1603_160367

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l1603_160367


namespace x_squared_plus_y_squared_l1603_160325

theorem x_squared_plus_y_squared (x y : ℝ) (h₁ : x - y = 18) (h₂ : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end x_squared_plus_y_squared_l1603_160325


namespace ferry_tourists_total_l1603_160358

def series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem ferry_tourists_total :
  let t_0 := 90
  let d := -2
  let n := 9
  series_sum t_0 d n = 738 :=
by
  sorry

end ferry_tourists_total_l1603_160358


namespace type_C_count_l1603_160348

theorem type_C_count (A B C C1 C2 : ℕ) (h1 : A + B + C = 25) (h2 : A + B + C2 = 17) (h3 : B + C2 = 12) (h4 : C2 = 8) (h5: B = 4) (h6: A = 5) : C = 16 :=
by {
  -- Directly use the given hypotheses.
  sorry
}

end type_C_count_l1603_160348


namespace game_goal_impossible_l1603_160382

-- Definition for initial setup
def initial_tokens : ℕ := 2013
def initial_piles : ℕ := 1

-- Definition for the invariant
def invariant (tokens piles : ℕ) : ℕ := tokens + piles

-- Initial value of the invariant constant
def initial_invariant : ℕ :=
  invariant initial_tokens initial_piles

-- Goal is to check if the final configuration is possible
theorem game_goal_impossible (n : ℕ) :
  (invariant (3 * n) n = initial_invariant) → false :=
by
  -- The invariant states 4n = initial_invariant which is 2014.
  -- Thus, we need to check if 2014 / 4 results in an integer.
  have invariant_expr : 4 * n = 2014 := by sorry
  have n_is_integer : 2014 % 4 = 0 := by sorry
  sorry

end game_goal_impossible_l1603_160382


namespace probability_more_heads_than_tails_l1603_160354

-- Define the total number of outcomes when flipping 10 coins
def total_outcomes : ℕ := 2 ^ 10

-- Define the number of ways to get exactly 5 heads out of 10 flips (combination)
def combinations_5_heads : ℕ := Nat.choose 10 5

-- Define the probability of getting exactly 5 heads out of 10 flips
def probability_5_heads := (combinations_5_heads : ℚ) / total_outcomes

-- Define the probability of getting more heads than tails (x)
def probability_more_heads := (1 - probability_5_heads) / 2

-- Theorem stating the probability of getting more heads than tails
theorem probability_more_heads_than_tails : probability_more_heads = 193 / 512 :=
by
  -- Proof skipped using sorry
  sorry

end probability_more_heads_than_tails_l1603_160354


namespace expression_positive_l1603_160332

theorem expression_positive (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) : 
  5 * x^2 + 5 * y^2 + 5 * z^2 + 6 * x * y - 8 * x * z - 8 * y * z > 0 := 
sorry

end expression_positive_l1603_160332


namespace total_mileage_pay_l1603_160372

-- Conditions
def distance_first_package : ℕ := 10
def distance_second_package : ℕ := 28
def distance_third_package : ℕ := distance_second_package / 2
def total_miles_driven : ℕ := distance_first_package + distance_second_package + distance_third_package
def pay_per_mile : ℕ := 2

-- Proof statement
theorem total_mileage_pay (X : ℕ) : 
  X + (total_miles_driven * pay_per_mile) = X + 104 := by
sorry

end total_mileage_pay_l1603_160372


namespace problem_inequality_l1603_160334

theorem problem_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n :=
sorry

end problem_inequality_l1603_160334


namespace jordan_no_quiz_probability_l1603_160389

theorem jordan_no_quiz_probability (P_quiz : ℚ) (h : P_quiz = 5 / 9) :
  1 - P_quiz = 4 / 9 :=
by
  rw [h]
  exact sorry

end jordan_no_quiz_probability_l1603_160389


namespace maximize_expression_l1603_160356

theorem maximize_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 :=
by
  sorry

end maximize_expression_l1603_160356


namespace bob_walks_more_l1603_160392

def street_width : ℝ := 30
def length_side1 : ℝ := 500
def length_side2 : ℝ := 300

def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

def alice_perimeter : ℝ := perimeter (length_side1 + 2 * street_width) (length_side2 + 2 * street_width)
def bob_perimeter : ℝ := perimeter (length_side1 + 4 * street_width) (length_side2 + 4 * street_width)

theorem bob_walks_more :
  bob_perimeter - alice_perimeter = 240 :=
by
  sorry

end bob_walks_more_l1603_160392


namespace minimum_value_of_expression_l1603_160318

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    ∃ (c : ℝ), (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x^3 + y^3 - 5 * x * y ≥ c) ∧ c = -125 / 27 :=
by
  sorry

end minimum_value_of_expression_l1603_160318


namespace cows_and_sheep_bushels_l1603_160344

theorem cows_and_sheep_bushels (bushels_per_chicken: Int) (total_bushels: Int) (num_chickens: Int) 
  (bushels_chickens: Int) (bushels_cows_sheep: Int) (num_cows: Int) (num_sheep: Int):
  bushels_per_chicken = 3 ∧ total_bushels = 35 ∧ num_chickens = 7 ∧
  bushels_chickens = num_chickens * bushels_per_chicken ∧ bushels_chickens = 21 ∧ bushels_cows_sheep = total_bushels - bushels_chickens → 
  bushels_cows_sheep = 14 := by
  sorry

end cows_and_sheep_bushels_l1603_160344


namespace lcm_of_coprimes_eq_product_l1603_160385

theorem lcm_of_coprimes_eq_product (a b c : ℕ) (h_coprime_ab : Nat.gcd a b = 1) (h_coprime_bc : Nat.gcd b c = 1) (h_coprime_ca : Nat.gcd c a = 1) (h_product : a * b * c = 7429) :
  Nat.lcm (Nat.lcm a b) c = 7429 :=
by 
  sorry

end lcm_of_coprimes_eq_product_l1603_160385


namespace hall_area_l1603_160369

theorem hall_area (L : ℝ) (B : ℝ) (A : ℝ) (h1 : B = (2/3) * L) (h2 : L = 60) (h3 : A = L * B) : A = 2400 := 
by 
sorry

end hall_area_l1603_160369


namespace circle_radius_l1603_160306

theorem circle_radius : 
  ∀ (x y : ℝ), x^2 + y^2 + 12 = 10 * x - 6 * y → ∃ r : ℝ, r = Real.sqrt 22 :=
by
  intros x y h
  -- Additional steps to complete the proof will be added here
  sorry

end circle_radius_l1603_160306


namespace B_and_C_finish_in_22_857_days_l1603_160360

noncomputable def work_rate_A := 1 / 40
noncomputable def work_rate_B := 1 / 60
noncomputable def work_rate_C := 1 / 80

noncomputable def work_done_by_A : ℚ := 10 * work_rate_A
noncomputable def work_done_by_B : ℚ := 5 * work_rate_B

noncomputable def remaining_work : ℚ := 1 - (work_done_by_A + work_done_by_B)

noncomputable def combined_work_rate_BC : ℚ := work_rate_B + work_rate_C

noncomputable def days_BC_to_finish_remaining_work : ℚ := remaining_work / combined_work_rate_BC

theorem B_and_C_finish_in_22_857_days : days_BC_to_finish_remaining_work = 160 / 7 :=
by
  -- Proof is omitted
  sorry

end B_and_C_finish_in_22_857_days_l1603_160360


namespace initial_markup_percentage_l1603_160323

theorem initial_markup_percentage (C M : ℝ) 
  (h1 : C > 0) 
  (h2 : (1 + M) * 1.25 * 0.92 = 1.38) :
  M = 0.2 :=
sorry

end initial_markup_percentage_l1603_160323


namespace find_number_l1603_160340

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l1603_160340


namespace jerry_showers_l1603_160353

variable (water_allowance : ℕ) (drinking_cooking : ℕ) (water_per_shower : ℕ) (pool_length : ℕ) 
  (pool_width : ℕ) (pool_height : ℕ) (gallons_per_cubic_foot : ℕ)

/-- Jerry can take 15 showers in July given the conditions. -/
theorem jerry_showers :
  water_allowance = 1000 →
  drinking_cooking = 100 →
  water_per_shower = 20 →
  pool_length = 10 →
  pool_width = 10 →
  pool_height = 6 →
  gallons_per_cubic_foot = 1 →
  (water_allowance - (drinking_cooking + (pool_length * pool_width * pool_height) * gallons_per_cubic_foot)) / water_per_shower = 15 :=
by
  intros h_water_allowance h_drinking_cooking h_water_per_shower h_pool_length h_pool_width h_pool_height h_gallons_per_cubic_foot
  sorry

end jerry_showers_l1603_160353


namespace correct_calculation_only_A_l1603_160387

-- Definitions of the expressions
def exprA (a : ℝ) : Prop := 3 * a + 2 * a = 5 * a
def exprB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def exprC (a : ℝ) : Prop := 3 * a * 2 * a = 6 * a
def exprD (a : ℝ) : Prop := 3 * a / (2 * a) = (3 / 2) * a

-- The theorem stating that only exprA is correct
theorem correct_calculation_only_A (a : ℝ) :
  exprA a ∧ ¬exprB a ∧ ¬exprC a ∧ ¬exprD a :=
by
  sorry

end correct_calculation_only_A_l1603_160387


namespace calc_3_op_2_op_4_op_1_l1603_160394

def op (a b : ℕ) : ℕ :=
match a, b with
| 1, 1 => 2 | 1, 2 => 3 | 1, 3 => 4 | 1, 4 => 1
| 2, 1 => 3 | 2, 2 => 1 | 2, 3 => 2 | 2, 4 => 4
| 3, 1 => 4 | 3, 2 => 2 | 3, 3 => 1 | 3, 4 => 3
| 4, 1 => 1 | 4, 2 => 4 | 4, 3 => 3 | 4, 4 => 2
| _, _  => 0 -- default case, though won't be used

theorem calc_3_op_2_op_4_op_1 : op (op 3 2) (op 4 1) = 3 :=
by
  sorry

end calc_3_op_2_op_4_op_1_l1603_160394


namespace find_k_l1603_160373

-- Define the lines l1 and l2
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the fact that the quadrilateral formed by l1, l2, and the positive halves of the axes
-- has a circumscribed circle.
def has_circumscribed_circle (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), line1 x1 y1 ∧ line2 k x2 y2 ∧
  x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0 ∧
  (x1 - x2 = 0 ∨ y1 - y2 = 0) ∧
  (x1 = 0 ∨ y1 = 0 ∨ x2 = 0 ∨ y2 = 0)

-- The statement we need to prove
theorem find_k : ∀ k : ℝ, has_circumscribed_circle k → k = 3 := by
  sorry

end find_k_l1603_160373


namespace inequality_true_l1603_160363

theorem inequality_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end inequality_true_l1603_160363


namespace system_exactly_two_solutions_l1603_160375

theorem system_exactly_two_solutions (a : ℝ) : 
  (∃ x y : ℝ, |y + x + 8| + |y - x + 8| = 16 ∧ (|x| - 15)^2 + (|y| - 8)^2 = a) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, |y₁ + x₁ + 8| + |y₁ - x₁ + 8| = 16 ∧ (|x₁| - 15)^2 + (|y₁| - 8)^2 = a → 
                      |y₂ + x₂ + 8| + |y₂ - x₂ + 8| = 16 ∧ (|x₂| - 15)^2 + (|y₂| - 8)^2 = a → 
                      x₁ = x₂ ∧ y₁ = y₂) → 
  (a = 49 ∨ a = 289) :=
sorry

end system_exactly_two_solutions_l1603_160375


namespace function_at_neg_one_zero_l1603_160328

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l1603_160328


namespace trapezoid_third_largest_angle_l1603_160307

theorem trapezoid_third_largest_angle (a d : ℝ)
  (h1 : 2 * a + 3 * d = 200)      -- Condition: 2a + 3d = 200°
  (h2 : a + d = 70) :             -- Condition: a + d = 70°
  a + 2 * d = 130 :=              -- Question: Prove a + 2d = 130°
by
  sorry

end trapezoid_third_largest_angle_l1603_160307


namespace triangle_PQR_PR_value_l1603_160308

theorem triangle_PQR_PR_value (PQ QR PR : ℕ) (h1 : PQ = 7) (h2 : QR = 20) (h3 : 13 < PR) (h4 : PR < 27) : PR = 21 :=
by sorry

end triangle_PQR_PR_value_l1603_160308


namespace num_factors_of_60_l1603_160301

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l1603_160301


namespace grape_juice_amount_l1603_160390

theorem grape_juice_amount (total_juice : ℝ)
  (orange_juice_percent : ℝ) (watermelon_juice_percent : ℝ)
  (orange_juice_amount : ℝ) (watermelon_juice_amount : ℝ)
  (grape_juice_amount : ℝ) :
  orange_juice_percent = 0.25 →
  watermelon_juice_percent = 0.40 →
  total_juice = 200 →
  orange_juice_amount = total_juice * orange_juice_percent →
  watermelon_juice_amount = total_juice * watermelon_juice_percent →
  grape_juice_amount = total_juice - orange_juice_amount - watermelon_juice_amount →
  grape_juice_amount = 70 :=
by
  sorry

end grape_juice_amount_l1603_160390


namespace perimeter_is_22_l1603_160383

-- Definitions based on the conditions
def side_lengths : List ℕ := [2, 3, 2, 6, 2, 4, 3]

-- Statement of the problem
theorem perimeter_is_22 : side_lengths.sum = 22 := 
  sorry

end perimeter_is_22_l1603_160383


namespace smallest_k_for_positive_roots_5_l1603_160336

noncomputable def smallest_k_for_positive_roots : ℕ := 5

theorem smallest_k_for_positive_roots_5
  (k p q : ℕ) 
  (hk : k = smallest_k_for_positive_roots)
  (hq_pos : 0 < q)
  (h_distinct_pos_roots : ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
    k * x₁ * x₂ = q ∧ k * x₁ + k * x₂ > p ∧ k * x₁ * x₂ < q * ( 1 / (x₁*(1 - x₁) * x₂ * (1 - x₂)))) :
  k = 5 :=
by
  sorry

end smallest_k_for_positive_roots_5_l1603_160336


namespace surface_area_of_rectangular_solid_l1603_160316

-- Conditions
variables {a b c : ℕ}
variables (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c)
variables (h_volume : a * b * c = 308)

-- Question and Proof Problem
theorem surface_area_of_rectangular_solid :
  2 * (a * b + b * c + c * a) = 226 :=
sorry

end surface_area_of_rectangular_solid_l1603_160316


namespace distance_to_airport_l1603_160342

theorem distance_to_airport:
  ∃ (d t: ℝ), 
    (d = 35 * (t + 1)) ∧
    (d - 35 = 50 * (t - 1.5)) ∧
    d = 210 := 
by 
  sorry

end distance_to_airport_l1603_160342


namespace operation_commutative_operation_associative_l1603_160347

def my_operation (a b : ℝ) : ℝ := a * b + a + b

theorem operation_commutative (a b : ℝ) : my_operation a b = my_operation b a := by
  sorry

theorem operation_associative (a b c : ℝ) : my_operation (my_operation a b) c = my_operation a (my_operation b c) := by
  sorry

end operation_commutative_operation_associative_l1603_160347


namespace pieces_per_pan_of_brownies_l1603_160310

theorem pieces_per_pan_of_brownies (total_guests guests_ala_mode additional_guests total_scoops_per_tub total_tubs_eaten total_pans guests_per_pan second_pan_percentage consumed_pans : ℝ)
    (h1 : total_guests = guests_ala_mode + additional_guests)
    (h2 : total_scoops_per_tub * total_tubs_eaten = guests_ala_mode * 2)
    (h3 : consumed_pans = 1 + second_pan_percentage)
    (h4 : second_pan_percentage = 0.75)
    (h5 : total_guests = guests_per_pan * consumed_pans)
    (h6 : guests_per_pan = 28)
    : total_guests / consumed_pans = 16 :=
by
  have h7 : total_scoops_per_tub * total_tubs_eaten = 48 := by sorry
  have h8 : guests_ala_mode = 24 := by sorry
  have h9 : total_guests = 28 := by sorry
  have h10 : consumed_pans = 1.75 := by sorry
  have h11 : guests_per_pan = 28 := by sorry
  sorry


end pieces_per_pan_of_brownies_l1603_160310


namespace goals_per_player_is_30_l1603_160309

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end goals_per_player_is_30_l1603_160309


namespace divisor_of_1076_plus_least_addend_l1603_160329

theorem divisor_of_1076_plus_least_addend (a d : ℕ) (h1 : 1076 + a = 1081) (h2 : d ∣ 1081) (ha : a = 5) : d = 13 := 
sorry

end divisor_of_1076_plus_least_addend_l1603_160329


namespace union_of_sets_complement_intersection_of_sets_l1603_160374

def setA : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def setB : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_sets :
  setA ∪ setB = {x | 2 < x ∧ x < 10} :=
sorry

theorem complement_intersection_of_sets :
  (setAᶜ) ∩ setB = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

end union_of_sets_complement_intersection_of_sets_l1603_160374


namespace hypotenuse_length_l1603_160376

def triangle_hypotenuse := ∃ (a b c : ℚ) (x : ℚ), 
  a = 9 ∧ b = 3 * x + 6 ∧ c = x + 15 ∧ 
  a + b + c = 45 ∧ 
  a^2 + b^2 = c^2 ∧ 
  x = 15 / 4 ∧ 
  c = 75 / 4

theorem hypotenuse_length : triangle_hypotenuse :=
sorry

end hypotenuse_length_l1603_160376


namespace determine_a_for_nonnegative_function_l1603_160312

def function_positive_on_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → a * x^3 - 3 * x + 1 ≥ 0

theorem determine_a_for_nonnegative_function :
  ∀ (a : ℝ), function_positive_on_interval a ↔ a = 4 :=
by
  sorry

end determine_a_for_nonnegative_function_l1603_160312


namespace min_value_of_1_over_a_plus_2_over_b_l1603_160345

theorem min_value_of_1_over_a_plus_2_over_b (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  (1 / a + 2 / b) ≥ 9 := 
sorry

end min_value_of_1_over_a_plus_2_over_b_l1603_160345


namespace range_of_x_l1603_160350

theorem range_of_x (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
  sorry

end range_of_x_l1603_160350


namespace find_reflection_line_l1603_160320

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end find_reflection_line_l1603_160320


namespace area_of_field_l1603_160364

-- Definitions based on the conditions
def length_uncovered (L : ℝ) := L = 20
def fencing_required (W : ℝ) (L : ℝ) := 2 * W + L = 76

-- Statement of the theorem to be proved
theorem area_of_field (L W : ℝ) (hL : length_uncovered L) (hF : fencing_required W L) : L * W = 560 := by
  sorry

end area_of_field_l1603_160364


namespace smallest_positive_n_l1603_160314

theorem smallest_positive_n
  (a x y : ℤ)
  (h1 : x ≡ a [ZMOD 9])
  (h2 : y ≡ -a [ZMOD 9]) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 9 = 0 ∧ n = 6 :=
by
  sorry

end smallest_positive_n_l1603_160314


namespace logical_equivalence_l1603_160370

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) :=
by
  sorry

end logical_equivalence_l1603_160370


namespace geometric_sequence_problem_l1603_160303

noncomputable def geometric_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem
  (a1 q : ℝ) (a2 : ℝ := a1 * q) (a5 : ℝ := a1 * q^4)
  (S2 : ℝ := geometric_sum a1 q 2) (S4 : ℝ := geometric_sum a1 q 4)
  (h1 : 8 * a2 + a5 = 0) :
  S4 / S2 = 5 :=
by
  sorry

end geometric_sequence_problem_l1603_160303


namespace dawn_monthly_savings_l1603_160366

variable (annual_income : ℕ)
variable (months : ℕ)
variable (tax_deduction_percent : ℚ)
variable (variable_expense_percent : ℚ)
variable (savings_percent : ℚ)

def calculate_monthly_savings (annual_income months : ℕ) 
    (tax_deduction_percent variable_expense_percent savings_percent : ℚ) : ℚ :=
  let monthly_income := (annual_income : ℚ) / months;
  let after_tax_income := monthly_income * (1 - tax_deduction_percent);
  let after_expenses_income := after_tax_income * (1 - variable_expense_percent);
  after_expenses_income * savings_percent

theorem dawn_monthly_savings : 
    calculate_monthly_savings 48000 12 0.20 0.30 0.10 = 224 := 
  by 
    sorry

end dawn_monthly_savings_l1603_160366


namespace at_least_one_ge_one_l1603_160378

theorem at_least_one_ge_one (x y : ℝ) (h : x + y ≥ 2) : x ≥ 1 ∨ y ≥ 1 :=
sorry

end at_least_one_ge_one_l1603_160378


namespace compare_exponent_inequality_l1603_160365

theorem compare_exponent_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 :=
sorry

end compare_exponent_inequality_l1603_160365


namespace range_of_a_l1603_160359

variable (a : ℝ)

def p := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0
def r := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a
  (h₀ : p a ∨ q a)
  (h₁ : ¬ (p a ∧ q a)) :
  r a :=
sorry

end range_of_a_l1603_160359


namespace magician_earnings_l1603_160313

noncomputable def total_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (end_decks : ℕ) (promotion_price : ℕ) (exchange_rate_start : ℚ) (exchange_rate_mid : ℚ) (foreign_sales_1 : ℕ) (domestic_sales : ℕ) (foreign_sales_2 : ℕ) : ℕ :=
  let foreign_earnings_1 := (foreign_sales_1 / 2) * promotion_price
  let foreign_earnings_2 := foreign_sales_2 * price_per_deck
  (domestic_sales / 2) * promotion_price + foreign_earnings_1 + foreign_earnings_2
  

-- Given conditions:
-- price_per_deck = 2
-- initial_decks = 5
-- end_decks = 3
-- promotion_price = 3
-- exchange_rate_start = 1
-- exchange_rate_mid = 1.5
-- foreign_sales_1 = 4
-- domestic_sales = 2
-- foreign_sales_2 = 1

theorem magician_earnings :
  total_earnings 2 5 3 3 1 1.5 4 2 1 = 11 :=
by
   sorry

end magician_earnings_l1603_160313
