import Mathlib

namespace NUMINAMATH_GPT_four_pow_expression_l2069_206979

theorem four_pow_expression : 4 ^ (3 ^ 2) / (4 ^ 3) ^ 2 = 64 := by
  sorry

end NUMINAMATH_GPT_four_pow_expression_l2069_206979


namespace NUMINAMATH_GPT_purple_cars_count_l2069_206913

theorem purple_cars_count
    (P R G : ℕ)
    (h1 : R = P + 6)
    (h2 : G = 4 * R)
    (h3 : P + R + G = 312) :
    P = 47 :=
by 
  sorry

end NUMINAMATH_GPT_purple_cars_count_l2069_206913


namespace NUMINAMATH_GPT_solve_quadratic_l2069_206959

theorem solve_quadratic (x : ℝ) (h1 : 2 * x ^ 2 = 9 * x - 4) (h2 : x ≠ 4) : 2 * x = 1 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2069_206959


namespace NUMINAMATH_GPT_find_number_l2069_206939

theorem find_number (N : ℝ) (h : 0.60 * N = 0.50 * 720) : N = 600 :=
sorry

end NUMINAMATH_GPT_find_number_l2069_206939


namespace NUMINAMATH_GPT_game_goal_impossible_l2069_206954

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

end NUMINAMATH_GPT_game_goal_impossible_l2069_206954


namespace NUMINAMATH_GPT_infinite_very_good_pairs_l2069_206928

-- Defining what it means for a pair to be "good"
def is_good (m n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ m ↔ p ∣ n)

-- Defining what it means for a pair to be "very good"
def is_very_good (m n : ℕ) : Prop :=
  is_good m n ∧ is_good (m + 1) (n + 1)

-- The theorem to prove: infiniteness of very good pairs
theorem infinite_very_good_pairs : Infinite {p : ℕ × ℕ | is_very_good p.1 p.2} :=
  sorry

end NUMINAMATH_GPT_infinite_very_good_pairs_l2069_206928


namespace NUMINAMATH_GPT_rolling_a_6_on_10th_is_random_event_l2069_206940

-- Definition of what it means for an event to be "random"
def is_random_event (event : ℕ → Prop) : Prop := 
  ∃ n : ℕ, event n

-- Condition: A die roll outcome for getting a 6
def die_roll_getting_6 (roll : ℕ) : Prop := 
  roll = 6

-- The main theorem to state the problem and the conclusion
theorem rolling_a_6_on_10th_is_random_event (event : ℕ → Prop) 
  (h : ∀ n, event n = die_roll_getting_6 n) : 
  is_random_event (event) := 
  sorry

end NUMINAMATH_GPT_rolling_a_6_on_10th_is_random_event_l2069_206940


namespace NUMINAMATH_GPT_tangent_line_to_circle_l2069_206952

-- Definitions derived directly from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0
def passes_through_point (l : ℝ → ℝ → Prop) : Prop := l (-1) 6

-- The statement to be proven
theorem tangent_line_to_circle :
  ∃ (l : ℝ → ℝ → Prop), passes_through_point l ∧ 
    ((∀ x y, l x y ↔ 3*x - 4*y + 27 = 0) ∨ 
     (∀ x y, l x y ↔ x + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_tangent_line_to_circle_l2069_206952


namespace NUMINAMATH_GPT_health_risk_factor_prob_l2069_206990

noncomputable def find_p_q_sum (p q: ℕ) : ℕ :=
if h1 : p.gcd q = 1 then
  31
else 
  sorry

theorem health_risk_factor_prob (p q : ℕ) (h1 : p.gcd q = 1) 
                                (h2 : (p : ℚ) / q = 5 / 26) :
  find_p_q_sum p q = 31 :=
sorry

end NUMINAMATH_GPT_health_risk_factor_prob_l2069_206990


namespace NUMINAMATH_GPT_janek_favorite_number_l2069_206904

theorem janek_favorite_number (S : Set ℕ) (n : ℕ) :
  S = {6, 8, 16, 22, 32} →
  n / 2 ∈ S →
  (n + 6) ∈ S →
  (n - 10) ∈ S →
  2 * n ∈ S →
  n = 16 := by
  sorry

end NUMINAMATH_GPT_janek_favorite_number_l2069_206904


namespace NUMINAMATH_GPT_union_of_sets_complement_intersection_of_sets_l2069_206950

def setA : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def setB : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_sets :
  setA ∪ setB = {x | 2 < x ∧ x < 10} :=
sorry

theorem complement_intersection_of_sets :
  (setAᶜ) ∩ setB = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

end NUMINAMATH_GPT_union_of_sets_complement_intersection_of_sets_l2069_206950


namespace NUMINAMATH_GPT_correct_calculation_only_A_l2069_206989

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

end NUMINAMATH_GPT_correct_calculation_only_A_l2069_206989


namespace NUMINAMATH_GPT_ratio_of_potatoes_l2069_206930

def total_potatoes : ℕ := 24
def number_of_people : ℕ := 3
def potatoes_per_person : ℕ := 8
def total_each_person : ℕ := potatoes_per_person * number_of_people

theorem ratio_of_potatoes :
  total_potatoes = total_each_person → (potatoes_per_person : ℚ) / (potatoes_per_person : ℚ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_potatoes_l2069_206930


namespace NUMINAMATH_GPT_upstream_speed_proof_l2069_206935

-- Definitions based on the conditions in the problem
def speed_in_still_water : ℝ := 25
def speed_downstream : ℝ := 35

-- The speed of the man rowing upstream
def speed_upstream : ℝ := speed_in_still_water - (speed_downstream - speed_in_still_water)

theorem upstream_speed_proof : speed_upstream = 15 := by
  -- Proof is omitted by using sorry
  sorry

end NUMINAMATH_GPT_upstream_speed_proof_l2069_206935


namespace NUMINAMATH_GPT_maggie_earnings_correct_l2069_206970

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

end NUMINAMATH_GPT_maggie_earnings_correct_l2069_206970


namespace NUMINAMATH_GPT_investment_percentage_l2069_206933

theorem investment_percentage (x : ℝ) :
  (4000 * (x / 100) + 3500 * 0.04 + 2500 * 0.064 = 500) ↔ (x = 5) :=
by
  sorry

end NUMINAMATH_GPT_investment_percentage_l2069_206933


namespace NUMINAMATH_GPT_find_m_l2069_206949

theorem find_m (m : ℤ) (h0 : -90 ≤ m) (h1 : m ≤ 90) (h2 : Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180)) : m = -10 :=
sorry

end NUMINAMATH_GPT_find_m_l2069_206949


namespace NUMINAMATH_GPT_probability_more_heads_than_tails_l2069_206957

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

end NUMINAMATH_GPT_probability_more_heads_than_tails_l2069_206957


namespace NUMINAMATH_GPT_find_k_l2069_206983

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

end NUMINAMATH_GPT_find_k_l2069_206983


namespace NUMINAMATH_GPT_tables_difference_l2069_206915

theorem tables_difference (N O : ℕ) (h1 : N + O = 40) (h2 : 6 * N + 4 * O = 212) : N - O = 12 :=
sorry

end NUMINAMATH_GPT_tables_difference_l2069_206915


namespace NUMINAMATH_GPT_largest_divisor_of_m_square_minus_n_square_l2069_206918

theorem largest_divisor_of_m_square_minus_n_square (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k : ℤ, k = 8 ∧ ∀ a b : ℤ, a % 2 = 1 → b % 2 = 1 → a > b → 8 ∣ (a^2 - b^2) := 
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_m_square_minus_n_square_l2069_206918


namespace NUMINAMATH_GPT_lcm_of_coprimes_eq_product_l2069_206980

theorem lcm_of_coprimes_eq_product (a b c : ℕ) (h_coprime_ab : Nat.gcd a b = 1) (h_coprime_bc : Nat.gcd b c = 1) (h_coprime_ca : Nat.gcd c a = 1) (h_product : a * b * c = 7429) :
  Nat.lcm (Nat.lcm a b) c = 7429 :=
by 
  sorry

end NUMINAMATH_GPT_lcm_of_coprimes_eq_product_l2069_206980


namespace NUMINAMATH_GPT_four_cells_same_color_rectangle_l2069_206916

theorem four_cells_same_color_rectangle (color : Fin 3 → Fin 7 → Bool) :
  ∃ (r₁ r₂ r₃ r₄ : Fin 3) (c₁ c₂ c₃ c₄ : Fin 7), 
    r₁ ≠ r₂ ∧ r₃ ≠ r₄ ∧ c₁ ≠ c₂ ∧ c₃ ≠ c₄ ∧ 
    r₁ = r₃ ∧ r₂ = r₄ ∧ c₁ = c₃ ∧ c₂ = c₄ ∧
    color r₁ c₁ = color r₁ c₂ ∧ color r₂ c₁ = color r₂ c₂ := sorry

end NUMINAMATH_GPT_four_cells_same_color_rectangle_l2069_206916


namespace NUMINAMATH_GPT_trajectory_equation_equation_of_line_l2069_206902

-- Define the parabola and the trajectory
def parabola (x y : ℝ) := y^2 = 16 * x
def trajectory (x y : ℝ) := y^2 = 4 * x

-- Define the properties of the point P and the line l
def is_midpoint (P A B : ℝ × ℝ) :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_through_point (x y k : ℝ) := 
  k * x + y = 1

-- Proof problem (Ⅰ): trajectory of the midpoints of segments perpendicular to the x-axis from points on the parabola
theorem trajectory_equation : ∀ (M : ℝ × ℝ), 
  (∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ is_midpoint M P (P.1, 0)) → 
  trajectory M.1 M.2 :=
sorry

-- Proof problem (Ⅱ): equation of line l
theorem equation_of_line : ∀ (A B P : ℝ × ℝ), 
  trajectory A.1 A.2 → trajectory B.1 B.2 → 
  P = (3,2) → is_midpoint P A B → 
  ∃ k, line_through_point (A.1 - B.1) (A.2 - B.2) k :=
sorry

end NUMINAMATH_GPT_trajectory_equation_equation_of_line_l2069_206902


namespace NUMINAMATH_GPT_at_least_one_ge_one_l2069_206951

theorem at_least_one_ge_one (x y : ℝ) (h : x + y ≥ 2) : x ≥ 1 ∨ y ≥ 1 :=
sorry

end NUMINAMATH_GPT_at_least_one_ge_one_l2069_206951


namespace NUMINAMATH_GPT_hall_area_l2069_206977

theorem hall_area (L : ℝ) (B : ℝ) (A : ℝ) (h1 : B = (2/3) * L) (h2 : L = 60) (h3 : A = L * B) : A = 2400 := 
by 
sorry

end NUMINAMATH_GPT_hall_area_l2069_206977


namespace NUMINAMATH_GPT_parallel_lines_a_l2069_206920

theorem parallel_lines_a (a : ℝ) :
  ((∃ k : ℝ, (a + 2) / 6 = k ∧ (a + 3) / (2 * a - 1) = k) ∧ 
   ¬ ((-5 / -5) = ((a + 2) / 6)) ∧ ((a + 3) / (2 * a - 1) = (-5 / -5))) →
  a = -5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_l2069_206920


namespace NUMINAMATH_GPT_system_exactly_two_solutions_l2069_206986

theorem system_exactly_two_solutions (a : ℝ) : 
  (∃ x y : ℝ, |y + x + 8| + |y - x + 8| = 16 ∧ (|x| - 15)^2 + (|y| - 8)^2 = a) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, |y₁ + x₁ + 8| + |y₁ - x₁ + 8| = 16 ∧ (|x₁| - 15)^2 + (|y₁| - 8)^2 = a → 
                      |y₂ + x₂ + 8| + |y₂ - x₂ + 8| = 16 ∧ (|x₂| - 15)^2 + (|y₂| - 8)^2 = a → 
                      x₁ = x₂ ∧ y₁ = y₂) → 
  (a = 49 ∨ a = 289) :=
sorry

end NUMINAMATH_GPT_system_exactly_two_solutions_l2069_206986


namespace NUMINAMATH_GPT_compare_exponent_inequality_l2069_206995

theorem compare_exponent_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 :=
sorry

end NUMINAMATH_GPT_compare_exponent_inequality_l2069_206995


namespace NUMINAMATH_GPT_smallest_n_good_sequence_2014_l2069_206923

-- Define the concept of a "good sequence"
def good_sequence (a : ℕ → ℝ) : Prop :=
  a 0 > 0 ∧
  ∀ i, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

-- Define the smallest n such that a good sequence reaches 2014 at a_n
theorem smallest_n_good_sequence_2014 :
  ∃ (n : ℕ), (∀ a, good_sequence a → a n = 2014) ∧
  ∀ (m : ℕ), m < n → ∀ a, good_sequence a → a m ≠ 2014 :=
sorry

end NUMINAMATH_GPT_smallest_n_good_sequence_2014_l2069_206923


namespace NUMINAMATH_GPT_range_of_m_l2069_206901

def set_A : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) : Set ℝ := { x : ℝ | (2 * m - 1) ≤ x ∧ x ≤ (2 * m + 1) }

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ (-1 / 2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2069_206901


namespace NUMINAMATH_GPT_ferry_tourists_total_l2069_206953

def series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem ferry_tourists_total :
  let t_0 := 90
  let d := -2
  let n := 9
  series_sum t_0 d n = 738 :=
by
  sorry

end NUMINAMATH_GPT_ferry_tourists_total_l2069_206953


namespace NUMINAMATH_GPT_pizza_eaten_after_six_trips_l2069_206943

theorem pizza_eaten_after_six_trips
  (initial_fraction: ℚ)
  (next_fraction : ℚ -> ℚ)
  (S: ℚ)
  (H0: initial_fraction = 1 / 4)
  (H1: ∀ (n: ℕ), next_fraction n = 1 / 2 ^ (n + 2))
  (H2: S = initial_fraction + (next_fraction 1) + (next_fraction 2) + (next_fraction 3) + (next_fraction 4) + (next_fraction 5)):
  S = 125 / 128 :=
by
  sorry

end NUMINAMATH_GPT_pizza_eaten_after_six_trips_l2069_206943


namespace NUMINAMATH_GPT_problem_l2069_206988

theorem problem (d r : ℕ) (a b c : ℕ) (ha : a = 1059) (hb : b = 1417) (hc : c = 2312)
  (h1 : d ∣ (b - a)) (h2 : d ∣ (c - a)) (h3 : d ∣ (c - b)) (hd : d > 1)
  (hr : r = a % d):
  d - r = 15 := sorry

end NUMINAMATH_GPT_problem_l2069_206988


namespace NUMINAMATH_GPT_mode_of_data_set_l2069_206934

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_mode_of_data_set_l2069_206934


namespace NUMINAMATH_GPT_area_of_field_l2069_206972

-- Definitions based on the conditions
def length_uncovered (L : ℝ) := L = 20
def fencing_required (W : ℝ) (L : ℝ) := 2 * W + L = 76

-- Statement of the theorem to be proved
theorem area_of_field (L W : ℝ) (hL : length_uncovered L) (hF : fencing_required W L) : L * W = 560 := by
  sorry

end NUMINAMATH_GPT_area_of_field_l2069_206972


namespace NUMINAMATH_GPT_distance_to_airport_l2069_206991

theorem distance_to_airport:
  ∃ (d t: ℝ), 
    (d = 35 * (t + 1)) ∧
    (d - 35 = 50 * (t - 1.5)) ∧
    d = 210 := 
by 
  sorry

end NUMINAMATH_GPT_distance_to_airport_l2069_206991


namespace NUMINAMATH_GPT_maximize_expression_l2069_206960

theorem maximize_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_GPT_maximize_expression_l2069_206960


namespace NUMINAMATH_GPT_luncheon_tables_needed_l2069_206903

theorem luncheon_tables_needed (invited : ℕ) (no_show : ℕ) (people_per_table : ℕ) (people_attended : ℕ) (tables_needed : ℕ) :
  invited = 47 →
  no_show = 7 →
  people_per_table = 5 →
  people_attended = invited - no_show →
  tables_needed = people_attended / people_per_table →
  tables_needed = 8 := by {
  -- Proof here
  sorry
}

end NUMINAMATH_GPT_luncheon_tables_needed_l2069_206903


namespace NUMINAMATH_GPT_Karlson_max_candies_l2069_206975

theorem Karlson_max_candies (f : Fin 25 → ℕ) (g : Fin 25 → Fin 25 → ℕ) :
  (∀ i, f i = 1) →
  (∀ i j, g i j = f i * f j) →
  (∃ (S : ℕ), S = 300) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Karlson_max_candies_l2069_206975


namespace NUMINAMATH_GPT_regular_polygon_sides_l2069_206982

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) / n = 150) : n = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2069_206982


namespace NUMINAMATH_GPT_type_C_count_l2069_206961

theorem type_C_count (A B C C1 C2 : ℕ) (h1 : A + B + C = 25) (h2 : A + B + C2 = 17) (h3 : B + C2 = 12) (h4 : C2 = 8) (h5: B = 4) (h6: A = 5) : C = 16 :=
by {
  -- Directly use the given hypotheses.
  sorry
}

end NUMINAMATH_GPT_type_C_count_l2069_206961


namespace NUMINAMATH_GPT_fraction_equation_solution_l2069_206910

theorem fraction_equation_solution (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_equation_solution_l2069_206910


namespace NUMINAMATH_GPT_jordan_no_quiz_probability_l2069_206993

theorem jordan_no_quiz_probability (P_quiz : ℚ) (h : P_quiz = 5 / 9) :
  1 - P_quiz = 4 / 9 :=
by
  rw [h]
  exact sorry

end NUMINAMATH_GPT_jordan_no_quiz_probability_l2069_206993


namespace NUMINAMATH_GPT_age_of_B_l2069_206984

-- Define the ages of A and B
variables (A B : ℕ)

-- The conditions given in the problem
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 9

theorem age_of_B (A B : ℕ) (h1 : condition1 A B) (h2 : condition2 A B) : B = 39 :=
by
  sorry

end NUMINAMATH_GPT_age_of_B_l2069_206984


namespace NUMINAMATH_GPT_dawn_monthly_savings_l2069_206964

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

end NUMINAMATH_GPT_dawn_monthly_savings_l2069_206964


namespace NUMINAMATH_GPT_interval_satisfaction_l2069_206938

theorem interval_satisfaction (a : ℝ) :
  (4 ≤ a / (3 * a - 6)) ∧ (a / (3 * a - 6) > 12) → a < 72 / 35 := 
by
  sorry

end NUMINAMATH_GPT_interval_satisfaction_l2069_206938


namespace NUMINAMATH_GPT_perimeter_is_22_l2069_206997

-- Definitions based on the conditions
def side_lengths : List ℕ := [2, 3, 2, 6, 2, 4, 3]

-- Statement of the problem
theorem perimeter_is_22 : side_lengths.sum = 22 := 
  sorry

end NUMINAMATH_GPT_perimeter_is_22_l2069_206997


namespace NUMINAMATH_GPT_count_total_kids_in_lawrence_l2069_206906

namespace LawrenceCountyKids

/-- Number of kids who went to camp from Lawrence county -/
def kids_went_to_camp : ℕ := 610769

/-- Number of kids who stayed home -/
def kids_stayed_home : ℕ := 590796

/-- Total number of kids in Lawrence county -/
def total_kids_in_county : ℕ := 1201565

/-- Proof statement -/
theorem count_total_kids_in_lawrence :
  kids_went_to_camp + kids_stayed_home = total_kids_in_county :=
sorry

end LawrenceCountyKids

end NUMINAMATH_GPT_count_total_kids_in_lawrence_l2069_206906


namespace NUMINAMATH_GPT_range_of_x_l2069_206956

theorem range_of_x (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
  sorry

end NUMINAMATH_GPT_range_of_x_l2069_206956


namespace NUMINAMATH_GPT_traveling_distance_l2069_206917

/-- Let D be the total distance from the dormitory to the city in kilometers.
Given the following conditions:
1. The student traveled 1/3 of the way by foot.
2. The student traveled 3/5 of the way by bus.
3. The remaining portion of the journey was covered by car, which equals 2 kilometers.
We need to prove that the total distance D is 30 kilometers. -/ 
theorem traveling_distance (D : ℕ) 
  (h1 : (1 / 3 : ℚ) * D + (3 / 5 : ℚ) * D + 2 = D) : D = 30 := 
sorry

end NUMINAMATH_GPT_traveling_distance_l2069_206917


namespace NUMINAMATH_GPT_calc_3_op_2_op_4_op_1_l2069_206946

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

end NUMINAMATH_GPT_calc_3_op_2_op_4_op_1_l2069_206946


namespace NUMINAMATH_GPT_min_value_expression_l2069_206955

theorem min_value_expression (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 21) ∧ 
           (∀ z : ℝ, (z = (x + 18) / Real.sqrt (x - 3)) → y ≤ z) := 
sorry

end NUMINAMATH_GPT_min_value_expression_l2069_206955


namespace NUMINAMATH_GPT_remainder_three_n_l2069_206974

theorem remainder_three_n (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_three_n_l2069_206974


namespace NUMINAMATH_GPT_find_number_1920_find_number_60_l2069_206941

theorem find_number_1920 : 320 * 6 = 1920 :=
by sorry

theorem find_number_60 : (1920 / 7 = 60) :=
by sorry

end NUMINAMATH_GPT_find_number_1920_find_number_60_l2069_206941


namespace NUMINAMATH_GPT_unique_positive_real_solution_l2069_206996

-- Define the function
def f (x : ℝ) : ℝ := x^11 + 9 * x^10 + 19 * x^9 + 2023 * x^8 - 1421 * x^7 + 5

-- Prove the statement
theorem unique_positive_real_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_real_solution_l2069_206996


namespace NUMINAMATH_GPT_find_Q_div_P_l2069_206937

variable (P Q : ℚ)
variable (h_eq : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
  P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))

theorem find_Q_div_P : Q / P = -6 / 13 := by
  sorry

end NUMINAMATH_GPT_find_Q_div_P_l2069_206937


namespace NUMINAMATH_GPT_jerry_showers_l2069_206971

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

end NUMINAMATH_GPT_jerry_showers_l2069_206971


namespace NUMINAMATH_GPT_max_min_diff_c_l2069_206914

theorem max_min_diff_c {a b c : ℝ} 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 15) : 
  (∃ c_max c_min, 
    (∀ a b c, a + b + c = 3 ∧ a^2 + b^2 + c^2 = 15 → c_min ≤ c ∧ c ≤ c_max) ∧ 
    c_max - c_min = 16 / 3) :=
sorry

end NUMINAMATH_GPT_max_min_diff_c_l2069_206914


namespace NUMINAMATH_GPT_total_mileage_pay_l2069_206945

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

end NUMINAMATH_GPT_total_mileage_pay_l2069_206945


namespace NUMINAMATH_GPT_regular_octagon_interior_angle_l2069_206969

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end NUMINAMATH_GPT_regular_octagon_interior_angle_l2069_206969


namespace NUMINAMATH_GPT_person_walking_speed_on_escalator_l2069_206911

theorem person_walking_speed_on_escalator 
  (v : ℝ) 
  (escalator_speed : ℝ := 15) 
  (escalator_length : ℝ := 180) 
  (time_taken : ℝ := 10)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) : 
  v = 3 := 
by 
  -- The proof steps will be filled in if required
  sorry

end NUMINAMATH_GPT_person_walking_speed_on_escalator_l2069_206911


namespace NUMINAMATH_GPT_arrow_hits_apple_l2069_206942

noncomputable def time_to_hit (L V0 : ℝ) (α β : ℝ) : ℝ :=
  (L / V0) * (Real.sin β / Real.sin (α + β))

theorem arrow_hits_apple (g : ℝ) (L V0 : ℝ) (α β : ℝ) (h : (L / V0) * (Real.sin β / Real.sin (α + β)) = 3 / 4) 
  : time_to_hit L V0 α β = 3 / 4 := 
  by
  sorry

end NUMINAMATH_GPT_arrow_hits_apple_l2069_206942


namespace NUMINAMATH_GPT_total_charge_for_trip_l2069_206924

noncomputable def calc_total_charge (initial_fee : ℝ) (additional_charge : ℝ) (miles : ℝ) (increment : ℝ) :=
  initial_fee + (additional_charge * (miles / increment))

theorem total_charge_for_trip :
  calc_total_charge 2.35 0.35 3.6 (2 / 5) = 8.65 :=
by
  sorry

end NUMINAMATH_GPT_total_charge_for_trip_l2069_206924


namespace NUMINAMATH_GPT_range_of_m_l2069_206967

theorem range_of_m (m : ℝ) : (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x < 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y < 0) ↔ 0 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l2069_206967


namespace NUMINAMATH_GPT_range_of_a_l2069_206998

variable (a : ℝ)

def p := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0
def r := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a
  (h₀ : p a ∨ q a)
  (h₁ : ¬ (p a ∧ q a)) :
  r a :=
sorry

end NUMINAMATH_GPT_range_of_a_l2069_206998


namespace NUMINAMATH_GPT_negation_of_forall_statement_l2069_206985

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end NUMINAMATH_GPT_negation_of_forall_statement_l2069_206985


namespace NUMINAMATH_GPT_find_cd_product_l2069_206958

open Complex

theorem find_cd_product :
  let u : ℂ := -3 + 4 * I
  let v : ℂ := 2 - I
  let c : ℂ := -5 + 5 * I
  let d : ℂ := -5 - 5 * I
  c * d = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_cd_product_l2069_206958


namespace NUMINAMATH_GPT_citizen_income_l2069_206926

theorem citizen_income (total_tax : ℝ) (income : ℝ) :
  total_tax = 15000 →
  (income ≤ 20000 → total_tax = income * 0.10) ∧
  (20000 < income ∧ income ≤ 50000 → total_tax = (20000 * 0.10) + ((income - 20000) * 0.15)) ∧
  (50000 < income ∧ income ≤ 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + ((income - 50000) * 0.20)) ∧
  (income > 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + (40000 * 0.20) + ((income - 90000) * 0.25)) →
  income = 92000 :=
by
  sorry

end NUMINAMATH_GPT_citizen_income_l2069_206926


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_odd_numbers_l2069_206931

theorem largest_of_seven_consecutive_odd_numbers (a b c d e f g : ℤ) 
  (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) 
  (h5: e % 2 = 1) (h6: f % 2 = 1) (h7: g % 2 = 1)
  (h8 : a + b + c + d + e + f + g = 105)
  (h9 : b = a + 2) (h10 : c = a + 4) (h11 : d = a + 6)
  (h12 : e = a + 8) (h13 : f = a + 10) (h14 : g = a + 12) :
  g = 21 :=
by 
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_odd_numbers_l2069_206931


namespace NUMINAMATH_GPT_lindas_daughters_and_granddaughters_no_daughters_l2069_206905

def number_of_people_with_no_daughters (total_daughters total_descendants daughters_with_5_daughters : ℕ) : ℕ :=
  total_descendants - (5 * daughters_with_5_daughters - total_daughters + daughters_with_5_daughters)

theorem lindas_daughters_and_granddaughters_no_daughters
  (total_daughters : ℕ)
  (total_descendants : ℕ)
  (daughters_with_5_daughters : ℕ)
  (H1 : total_daughters = 8)
  (H2 : total_descendants = 43)
  (H3 : 5 * daughters_with_5_daughters = 35)
  : number_of_people_with_no_daughters total_daughters total_descendants daughters_with_5_daughters = 36 :=
by
  -- Code to check the proof goes here.
  sorry

end NUMINAMATH_GPT_lindas_daughters_and_granddaughters_no_daughters_l2069_206905


namespace NUMINAMATH_GPT_B_and_C_finish_in_22_857_days_l2069_206948

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

end NUMINAMATH_GPT_B_and_C_finish_in_22_857_days_l2069_206948


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l2069_206999

-- Definitions based on the initial conditions
def G : ℕ := 135
def T : ℕ := 351

-- Noncomputable because it involves division which is not always computable
noncomputable def B : ℕ := T - G

-- Main theorem to prove the ratio
theorem ratio_of_boys_to_girls : (B : ℚ) / G = 8 / 5 :=
by
  -- Here would be the proof, skipped with sorry.
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l2069_206999


namespace NUMINAMATH_GPT_ways_to_distribute_balls_l2069_206922

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_distribute_balls_l2069_206922


namespace NUMINAMATH_GPT_hypotenuse_length_l2069_206987

def triangle_hypotenuse := ∃ (a b c : ℚ) (x : ℚ), 
  a = 9 ∧ b = 3 * x + 6 ∧ c = x + 15 ∧ 
  a + b + c = 45 ∧ 
  a^2 + b^2 = c^2 ∧ 
  x = 15 / 4 ∧ 
  c = 75 / 4

theorem hypotenuse_length : triangle_hypotenuse :=
sorry

end NUMINAMATH_GPT_hypotenuse_length_l2069_206987


namespace NUMINAMATH_GPT_cows_and_sheep_bushels_l2069_206966

theorem cows_and_sheep_bushels (bushels_per_chicken: Int) (total_bushels: Int) (num_chickens: Int) 
  (bushels_chickens: Int) (bushels_cows_sheep: Int) (num_cows: Int) (num_sheep: Int):
  bushels_per_chicken = 3 ∧ total_bushels = 35 ∧ num_chickens = 7 ∧
  bushels_chickens = num_chickens * bushels_per_chicken ∧ bushels_chickens = 21 ∧ bushels_cows_sheep = total_bushels - bushels_chickens → 
  bushels_cows_sheep = 14 := by
  sorry

end NUMINAMATH_GPT_cows_and_sheep_bushels_l2069_206966


namespace NUMINAMATH_GPT_selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l2069_206908

section ProofProblems

-- Definitions and constants
def num_males := 6
def num_females := 4
def total_athletes := 10
def num_selections := 5
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- 1. Number of selection methods for 3 males and 2 females
theorem selection_3m2f : binom 6 3 * binom 4 2 = 120 := by sorry

-- 2. Number of selection methods with at least one captain
theorem selection_at_least_one_captain :
  2 * binom 8 4 + binom 8 3 = 196 := by sorry

-- 3. Number of selection methods with at least one female athlete
theorem selection_at_least_one_female :
  binom 10 5 - binom 6 5 = 246 := by sorry

-- 4. Number of selection methods with both a captain and at least one female athlete
theorem selection_captain_and_female :
  binom 9 4 + binom 8 4 - binom 5 4 = 191 := by sorry

end ProofProblems

end NUMINAMATH_GPT_selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l2069_206908


namespace NUMINAMATH_GPT_inequality_true_l2069_206968

theorem inequality_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end NUMINAMATH_GPT_inequality_true_l2069_206968


namespace NUMINAMATH_GPT_total_people_wearing_hats_l2069_206936

variable (total_adults : ℕ) (total_children : ℕ)
variable (half_adults : ℕ) (women : ℕ) (men : ℕ)
variable (women_with_hats : ℕ) (men_with_hats : ℕ)
variable (children_with_hats : ℕ)
variable (total_with_hats : ℕ)

-- Given conditions
def conditions : Prop :=
  total_adults = 1800 ∧
  total_children = 200 ∧
  half_adults = total_adults / 2 ∧
  women = half_adults ∧
  men = half_adults ∧
  women_with_hats = (25 * women) / 100 ∧
  men_with_hats = (12 * men) / 100 ∧
  children_with_hats = (10 * total_children) / 100 ∧
  total_with_hats = women_with_hats + men_with_hats + children_with_hats

-- Proof goal
theorem total_people_wearing_hats : conditions total_adults total_children half_adults women men women_with_hats men_with_hats children_with_hats total_with_hats → total_with_hats = 353 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_total_people_wearing_hats_l2069_206936


namespace NUMINAMATH_GPT_minimize_distance_AP_BP_l2069_206925

theorem minimize_distance_AP_BP :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = -1 ∧
    ∀ P' : ℝ × ℝ, P'.1 = 0 → 
      (dist (3, 2) P + dist (1, -2) P) ≤ (dist (3, 2) P' + dist (1, -2) P') := by
sorry

end NUMINAMATH_GPT_minimize_distance_AP_BP_l2069_206925


namespace NUMINAMATH_GPT_max_x_add_2y_l2069_206909

theorem max_x_add_2y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + 2 * y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_x_add_2y_l2069_206909


namespace NUMINAMATH_GPT_factor_polynomial_l2069_206919

theorem factor_polynomial :
  ∀ (x : ℤ), 9 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 5 * x^2 = (x^2 + 4) * (9 * x^2 + 22 * x + 342) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_polynomial_l2069_206919


namespace NUMINAMATH_GPT_max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l2069_206912

variable {m x x0 : ℝ}

def proposition_p (m : ℝ) : Prop := ∀ x > -2, x + 49 / (x + 2) ≥ 6 * Real.sqrt 2 * m
def proposition_q (m : ℝ) : Prop := ∃ x0 : ℝ, x0 ^ 2 - m * x0 + 1 = 0

theorem max_val_of_m_if_p_true (h : proposition_p m) : m ≤ Real.sqrt 2 := by
  sorry

theorem range_of_m_if_one_prop_true_one_false (hp : proposition_p m) (hq : ¬ proposition_q m) : (-2 < m ∧ m ≤ Real.sqrt 2) ∨ (2 ≤ m) := by
  sorry

theorem range_of_m_if_one_prop_false_one_true (hp : ¬ proposition_p m) (hq : proposition_q m) : (m ≥ 2) := by
  sorry

end NUMINAMATH_GPT_max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l2069_206912


namespace NUMINAMATH_GPT_parabola_min_value_incorrect_statement_l2069_206976

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

end NUMINAMATH_GPT_parabola_min_value_incorrect_statement_l2069_206976


namespace NUMINAMATH_GPT_janes_score_l2069_206981

theorem janes_score (jane_score tom_score : ℕ) (h1 : jane_score = tom_score + 50) (h2 : (jane_score + tom_score) / 2 = 90) :
  jane_score = 115 :=
sorry

end NUMINAMATH_GPT_janes_score_l2069_206981


namespace NUMINAMATH_GPT_bob_walks_more_l2069_206994

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

end NUMINAMATH_GPT_bob_walks_more_l2069_206994


namespace NUMINAMATH_GPT_determine_common_ratio_l2069_206992

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

end NUMINAMATH_GPT_determine_common_ratio_l2069_206992


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_a_l2069_206900

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_a_l2069_206900


namespace NUMINAMATH_GPT_logical_equivalence_l2069_206978

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) :=
by
  sorry

end NUMINAMATH_GPT_logical_equivalence_l2069_206978


namespace NUMINAMATH_GPT_calculate_f_of_f_of_f_l2069_206963

def f (x : ℤ) : ℤ := 5 * x - 4

theorem calculate_f_of_f_of_f (h : f (f (f 3)) = 251) : f (f (f 3)) = 251 := 
by sorry

end NUMINAMATH_GPT_calculate_f_of_f_of_f_l2069_206963


namespace NUMINAMATH_GPT_grape_juice_amount_l2069_206973

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

end NUMINAMATH_GPT_grape_juice_amount_l2069_206973


namespace NUMINAMATH_GPT_change_color_while_preserving_friendship_l2069_206921

-- Definitions
def children := Fin 10000
def colors := Fin 7
def friends (a b : children) : Prop := sorry -- mutual and exactly 11 friends per child
def refuses_to_change (c : children) : Prop := sorry -- only 100 specified children refuse to change color

theorem change_color_while_preserving_friendship :
  ∃ c : children, ¬refuses_to_change c ∧
    ∃ new_color : colors, 
      (∀ friend : children, friends c friend → 
      (∃ current_color current_friend_color : colors, current_color ≠ current_friend_color)) :=
sorry

end NUMINAMATH_GPT_change_color_while_preserving_friendship_l2069_206921


namespace NUMINAMATH_GPT_min_value_of_1_over_a_plus_2_over_b_l2069_206947

theorem min_value_of_1_over_a_plus_2_over_b (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  (1 / a + 2 / b) ≥ 9 := 
sorry

end NUMINAMATH_GPT_min_value_of_1_over_a_plus_2_over_b_l2069_206947


namespace NUMINAMATH_GPT_surface_area_of_box_l2069_206932

variable {l w h : ℝ}

def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * h + w * h + l * w)

theorem surface_area_of_box (l w h : ℝ) : surfaceArea l w h = 2 * (l * h + w * h + l * w) :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_box_l2069_206932


namespace NUMINAMATH_GPT_set_equality_l2069_206962

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

end NUMINAMATH_GPT_set_equality_l2069_206962


namespace NUMINAMATH_GPT_hyperbola_k_range_l2069_206927

theorem hyperbola_k_range (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (k + 2) - y^2 / (5 - k) = 1)) → (-2 < k ∧ k < 5) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_k_range_l2069_206927


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l2069_206907

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem quadratic_distinct_roots (k : ℝ) :
  (k ≠ 0) ∧ (1 > k) ↔ has_two_distinct_real_roots k (-6) 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l2069_206907


namespace NUMINAMATH_GPT_average_salary_of_managers_l2069_206944

theorem average_salary_of_managers (m_avg : ℝ) (assoc_avg : ℝ) (company_avg : ℝ)
  (managers : ℕ) (associates : ℕ) (total_employees : ℕ)
  (h_assoc_avg : assoc_avg = 30000) (h_company_avg : company_avg = 40000)
  (h_managers : managers = 15) (h_associates : associates = 75) (h_total_employees : total_employees = 90)
  (h_total_employees_def : total_employees = managers + associates)
  (h_total_salary_managers : ∀ m_avg, total_employees * company_avg = managers * m_avg + associates * assoc_avg) :
  m_avg = 90000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_of_managers_l2069_206944


namespace NUMINAMATH_GPT_operation_commutative_operation_associative_l2069_206965

def my_operation (a b : ℝ) : ℝ := a * b + a + b

theorem operation_commutative (a b : ℝ) : my_operation a b = my_operation b a := by
  sorry

theorem operation_associative (a b c : ℝ) : my_operation (my_operation a b) c = my_operation a (my_operation b c) := by
  sorry

end NUMINAMATH_GPT_operation_commutative_operation_associative_l2069_206965


namespace NUMINAMATH_GPT_find_divisor_l2069_206929

theorem find_divisor (x : ℤ) : 83 = 9 * x + 2 → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l2069_206929
