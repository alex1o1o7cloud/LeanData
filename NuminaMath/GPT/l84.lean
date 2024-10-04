import Mathlib

namespace solve_for_y_l84_84530

theorem solve_for_y (y : ℝ) : 4 * y + 6 * y = 450 - 10 * (y - 5) → y = 25 :=
by
  sorry

end solve_for_y_l84_84530


namespace math_proof_problem_l84_84467

noncomputable def f (x : ℝ) := Real.log (Real.sin x) * Real.log (Real.cos x)

def domain (k : ℤ) : Set ℝ := { x | 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi / 2 }

def is_even_shifted : Prop :=
  ∀ x, f (x + Real.pi / 4) = f (- (x + Real.pi / 4))

def has_unique_maximum : Prop :=
  ∃! x, 0 < x ∧ x < Real.pi / 2 ∧ ∀ y, 0 < y ∧ y < Real.pi / 2 → f y ≤ f x

theorem math_proof_problem (k : ℤ) :
  (∀ x, x ∈ domain k → f x ∈ domain k) ∧
  ¬ (∀ x, f (-x) = f x) ∧
  is_even_shifted ∧
  has_unique_maximum :=
by
  sorry

end math_proof_problem_l84_84467


namespace arrange_consecutive_integers_no_common_divisors_l84_84014

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l84_84014


namespace lcm_technicians_schedule_l84_84789

theorem lcm_technicians_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := 
sorry

end lcm_technicians_schedule_l84_84789


namespace incorrect_statement_trajectory_of_P_l84_84249

noncomputable def midpoint_of_points (x1 x2 y1 y2 : ℝ) : ℝ × ℝ :=
((x1 + x2) / 2, (y1 + y2) / 2)

theorem incorrect_statement_trajectory_of_P (p k x0 y0 : ℝ) (hp : p > 0)
    (A B : ℝ × ℝ)
    (hA : A.1 * A.1 + 2 * p * A.2 = 0)
    (hB : B.1 * B.1 + 2 * p * B.2 = 0)
    (hMid : (x0, y0) = midpoint_of_points A.1 B.1 A.2 B.2)
    (hLine : A.2 = k * (A.1 - p / 2))
    (hLineIntersection : B.2 = k * (B.1 - p / 2)) : y0 ^ 2 ≠ 4 * p * (x0 - p / 2) :=
by
  sorry

end incorrect_statement_trajectory_of_P_l84_84249


namespace dice_circle_probability_l84_84582

theorem dice_circle_probability :
  ∀ (d : ℕ), (2 ≤ d ∧ d ≤ 432) ∧
  ((∃ (x y : ℕ), (1 ≤ x ∧ x ≤ 6) ∧ (1 ≤ y ∧ y <= 6) ∧ d = x^3 + y^3)) →
  ((d * (d - 4) < 0) ↔ (d = 2)) →
  (∃ (P : ℚ), P = 1 / 36) :=
by
  sorry

end dice_circle_probability_l84_84582


namespace tan_eleven_pi_over_four_eq_neg_one_l84_84930

noncomputable def tan_of_eleven_pi_over_four : Real := 
  let to_degrees (x : Real) : Real := x * 180 / Real.pi
  let angle := to_degrees (11 * Real.pi / 4)
  let simplified := angle - 360 * Real.floor (angle / 360)
  if simplified < 0 then
    simplified := simplified + 360
  if simplified = 135 then -1
  else
    undefined

theorem tan_eleven_pi_over_four_eq_neg_one :
  tan (11 * Real.pi / 4) = -1 := 
by
  sorry

end tan_eleven_pi_over_four_eq_neg_one_l84_84930


namespace largest_common_term_l84_84864

theorem largest_common_term (a : ℕ) (k l : ℕ) (hk : a = 4 + 5 * k) (hl : a = 5 + 10 * l) (h : a < 300) : a = 299 :=
by {
  sorry
}

end largest_common_term_l84_84864


namespace items_priced_at_9_yuan_l84_84184

theorem items_priced_at_9_yuan (equal_number_items : ℕ)
  (total_cost : ℕ)
  (price_8_yuan : ℕ)
  (price_9_yuan : ℕ)
  (price_8_yuan_count : ℕ)
  (price_9_yuan_count : ℕ) :
  equal_number_items * 2 = price_8_yuan_count + price_9_yuan_count ∧
  (price_8_yuan_count * price_8_yuan + price_9_yuan_count * price_9_yuan = total_cost) ∧
  (price_8_yuan = 8) ∧
  (price_9_yuan = 9) ∧
  (total_cost = 172) →
  price_9_yuan_count = 12 :=
by
  sorry

end items_priced_at_9_yuan_l84_84184


namespace greatest_fifty_supportive_X_l84_84993

def fifty_supportive (X : ℝ) : Prop :=
∀ (a : Fin 50 → ℝ),
  (∑ i, a i).floor = (∑ i, a i) →
  ∃ i, |a i - 0.5| ≥ X

theorem greatest_fifty_supportive_X :
  ∀ X : ℝ, fifty_supportive X ↔ X ≤ 0.01 := sorry

end greatest_fifty_supportive_X_l84_84993


namespace route_B_no_quicker_l84_84683

noncomputable def time_route_A (distance_A : ℕ) (speed_A : ℕ) : ℕ :=
(distance_A * 60) / speed_A

noncomputable def time_route_B (distance_B : ℕ) (speed_B1 : ℕ) (speed_B2 : ℕ) : ℕ :=
  let distance_B1 := distance_B - 1
  let distance_B2 := 1
  (distance_B1 * 60) / speed_B1 + (distance_B2 * 60) / speed_B2

theorem route_B_no_quicker : time_route_A 8 40 = time_route_B 6 50 10 :=
by
  sorry

end route_B_no_quicker_l84_84683


namespace problem1_l84_84111

variable {a b : ℝ}

theorem problem1 (ha : a > 0) (hb : b > 0) : 
  (1 / (a + b) ≤ 1 / 4 * (1 / a + 1 / b)) :=
sorry

end problem1_l84_84111


namespace part1_part2_l84_84452

namespace Problem

open Set

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Part (1)
theorem part1 : A ∩ (B ∩ C) = {3} := by 
  sorry

-- Part (2)
theorem part2 : A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0} := by 
  sorry

end Problem

end part1_part2_l84_84452


namespace value_of_a_plus_b_l84_84806

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end value_of_a_plus_b_l84_84806


namespace carla_correct_questions_l84_84978

theorem carla_correct_questions :
  ∀ (Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct : ℕ), 
    Drew_correct = 20 →
    Drew_wrong = 6 →
    Carla_wrong = 2 * Drew_wrong →
    Total_questions = 52 →
    Carla_correct = Total_questions - Carla_wrong →
    Carla_correct = 40 :=
by
  intros Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end carla_correct_questions_l84_84978


namespace determine_h_f_l84_84254

variable (e f g h : ℕ)
variable (u v : ℕ)
variable (h_e : e = u^4)
variable (h_f : f = u^5)
variable (h_g : g = v^2)
variable (h_h : h = v^3)
variable (h_cond1 : e^5 = f^4)
variable (h_cond2 : g^3 = h^2)
variable (h_cond3 : g - e = 31)

theorem determine_h_f : h - f = 971 := by
  sorry

end determine_h_f_l84_84254


namespace n_times_s_eq_2023_l84_84351

noncomputable def S := { x : ℝ | x > 0 }

-- Function f: S → ℝ
def f (x : ℝ) : ℝ := sorry

-- Condition: f(x) f(y) = f(xy) + 2023 * (2/x + 2/y + 2022) for all x, y > 0
axiom f_property (x y : ℝ) (hx : x > 0) (hy : y > 0) : f x * f y = f (x * y) + 2023 * (2 / x + 2 / y + 2022)

-- Theorem: Prove n × s = 2023 where n is the number of possible values of f(2) and s is the sum of all possible values of f(2)
theorem n_times_s_eq_2023 (n s : ℕ) : n * s = 2023 :=
sorry

end n_times_s_eq_2023_l84_84351


namespace max_min_value_of_f_l84_84323

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end max_min_value_of_f_l84_84323


namespace initial_cd_count_l84_84749

variable (X : ℕ)

theorem initial_cd_count (h1 : (2 / 3 : ℝ) * X + 8 = 22) : X = 21 :=
by
  sorry

end initial_cd_count_l84_84749


namespace square_area_divided_into_rectangles_l84_84585

theorem square_area_divided_into_rectangles (l w : ℝ) 
  (h1 : 2 * (l + w) = 120)
  (h2 : l = 5 * w) :
  (5 * w * w)^2 = 2500 := 
by {
  -- Sorry placeholder for proof
  sorry
}

end square_area_divided_into_rectangles_l84_84585


namespace total_nails_to_cut_l84_84274

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l84_84274


namespace find_k_l84_84291

theorem find_k (x y z k : ℝ) (h1 : 5 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 9 / (z - y)) : k = 14 :=
by
  sorry

end find_k_l84_84291


namespace least_number_remainder_l84_84557

theorem least_number_remainder (N k : ℕ) (h : N = 18 * k + 4) : N = 256 :=
by
  sorry

end least_number_remainder_l84_84557


namespace sum_of_consecutive_integers_product_336_l84_84710

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l84_84710


namespace gcf_lcm_360_210_l84_84556

theorem gcf_lcm_360_210 :
  let factorization_360 : ℕ × ℕ × ℕ × ℕ := (3, 2, 1, 0) -- Prime exponents for 2, 3, 5, 7
  let factorization_210 : ℕ × ℕ × ℕ × ℕ := (1, 1, 1, 1) -- Prime exponents for 2, 3, 5, 7
  gcd (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 30 ∧
  lcm (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 2520 :=
by {
  let factorization_360 := (3, 2, 1, 0)
  let factorization_210 := (1, 1, 1, 1)
  sorry
}

end gcf_lcm_360_210_l84_84556


namespace board_total_length_l84_84406

-- Definitions based on conditions
def S : ℝ := 2
def L : ℝ := 2 * S

-- Define the total length of the board
def T : ℝ := S + L

-- The theorem asserting the total length of the board is 6 ft
theorem board_total_length : T = 6 := 
by
  sorry

end board_total_length_l84_84406


namespace completing_the_square_result_l84_84087

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l84_84087


namespace part1_part2_l84_84669

-- Definitions from condition part
def f (a x : ℝ) := a * x^2 + (1 + a) * x + a

-- Part (1) Statement
theorem part1 (a : ℝ) : 
  (a ≥ -1/3) → (∀ x : ℝ, f a x ≥ 0) :=
sorry

-- Part (2) Statement
theorem part2 (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, f a x < a - 1) → 
  ((0 < a ∧ a < 1) → (-1/a < x ∧ x < -1) ∨ 
   (a = 1) → False ∨
   (a > 1) → (-1 < x ∧ x < -1/a)) :=
sorry

end part1_part2_l84_84669


namespace extra_men_needed_l84_84762

theorem extra_men_needed (total_length : ℝ) (total_days : ℕ) (initial_men : ℕ) (completed_length : ℝ) (days_passed : ℕ) 
  (remaining_length := total_length - completed_length)
  (remaining_days := total_days - days_passed)
  (current_rate := completed_length / days_passed)
  (required_rate := remaining_length / remaining_days)
  (rate_increase := required_rate / current_rate)
  (total_men_needed := initial_men * rate_increase)
  (extra_men_needed := ⌈total_men_needed⌉ - initial_men) :
  total_length = 15 → 
  total_days = 300 → 
  initial_men = 35 → 
  completed_length = 2.5 → 
  days_passed = 100 → 
  extra_men_needed = 53 :=
by
-- Prove that given the conditions, the number of extra men needed is 53
sorry

end extra_men_needed_l84_84762


namespace problem_statement_l84_84982

-- Definitions based on conditions
def position_of_3_in_8_063 := "thousandths"
def representation_of_3_in_8_063 : ℝ := 3 * 0.001
def unit_in_0_48 : ℝ := 0.01

theorem problem_statement :
  (position_of_3_in_8_063 = "thousandths") ∧
  (representation_of_3_in_8_063 = 3 * 0.001) ∧
  (unit_in_0_48 = 0.01) :=
sorry

end problem_statement_l84_84982


namespace four_digit_3_or_6_l84_84644

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l84_84644


namespace greatest_50_supportive_X_correct_l84_84992

noncomputable def greatest_50_supportive_X (a : Fin 50 → ℝ) : ℝ :=
if ∃ X, ∀ a : Fin 50 → ℝ, (∑ i : Fin 50, a i).floor = ∑ i : Fin 50, a i ∧ 
  (∃ i, |a i - 0.5| >= X) 
then 0.01 else 0

theorem greatest_50_supportive_X_correct :
  ∀ a : Fin 50 → ℝ, (∑ i : Fin 50, a i).floor = ∑ i : Fin 50, a i → 
  (∃ i, |a i - 0.5| >= 0.01) :=
sorry

end greatest_50_supportive_X_correct_l84_84992


namespace solve_quadratic_eq_solve_equal_squares_l84_84043

theorem solve_quadratic_eq (x : ℝ) : 
    (4 * x^2 - 2 * x - 1 = 0) ↔ 
    (x = (1 + Real.sqrt 5) / 4 ∨ x = (1 - Real.sqrt 5) / 4) := 
by
  sorry

theorem solve_equal_squares (y : ℝ) :
    ((y + 1)^2 = (3 * y - 1)^2) ↔ 
    (y = 1 ∨ y = 0) := 
by
  sorry

end solve_quadratic_eq_solve_equal_squares_l84_84043


namespace extra_time_needed_l84_84583

variable (S : ℝ) (d : ℝ) (T T' : ℝ)

-- Original conditions
def original_speed_at_time_distance (S : ℝ) (T : ℝ) (d : ℝ) : Prop :=
  S * T = d

def decreased_speed (original_S : ℝ) : ℝ :=
  0.80 * original_S

def decreased_speed_time (T' : ℝ ) (decreased_S : ℝ) (d : ℝ) : Prop :=
  decreased_S * T' = d

theorem extra_time_needed
  (h1 : original_speed_at_time_distance S T d)
  (h2 : T = 40)
  (h3 : decreased_speed S = 0.80 * S)
  (h4 : decreased_speed_time T' (decreased_speed S) d) :
  T' - T = 10 :=
by
  sorry

end extra_time_needed_l84_84583


namespace minimum_selling_price_l84_84264

def monthly_sales : ℕ := 50
def base_cost : ℕ := 1200
def shipping_cost : ℕ := 20
def store_fee : ℕ := 10000
def repair_fee : ℕ := 5000
def profit_margin : ℕ := 20

def total_monthly_expenses : ℕ := store_fee + repair_fee
def total_cost_per_machine : ℕ := base_cost + shipping_cost + total_monthly_expenses / monthly_sales
def min_selling_price : ℕ := total_cost_per_machine * (1 + profit_margin / 100)

theorem minimum_selling_price : min_selling_price = 1824 := 
by
  sorry 

end minimum_selling_price_l84_84264


namespace distance_from_point_to_directrix_l84_84307

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l84_84307


namespace range_of_m_l84_84954

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + Real.log x

theorem range_of_m (e l : ℝ) (alpha : ℝ) :
  (∀ (α : ℝ), α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 1 ^ 2) → 
  ∀ (x : ℝ), x ∈ Set.Icc l e → f alpha x < m) →
  m ∈ Set.Ioi (1 + 2 * Real.exp 1 ^ 2) := sorry

end range_of_m_l84_84954


namespace temperature_reaches_90_at_17_l84_84831

def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem temperature_reaches_90_at_17 :
  ∃ t : ℝ, temperature t = 90 ∧ t = 17 :=
by
  exists 17
  dsimp [temperature]
  norm_num
  sorry

end temperature_reaches_90_at_17_l84_84831


namespace problem_l84_84192

    theorem problem (a b c : ℝ) : 
        a < b → 
        (∀ x : ℝ, (x ≤ -2 ∨ |x - 30| < 2) ↔ (0 ≤ (x - a) * (x - b) / (x - c))) → 
        a + 2 * b + 3 * c = 86 := by 
    sorry

end problem_l84_84192


namespace gcd_280_2155_l84_84699

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := 
sorry

end gcd_280_2155_l84_84699


namespace number_of_girls_l84_84523

theorem number_of_girls (total_children boys girls : ℕ) 
    (total_children_eq : total_children = 60)
    (boys_eq : boys = 22)
    (compute_girls : girls = total_children - boys) : 
    girls = 38 :=
by
    rw [total_children_eq, boys_eq] at compute_girls
    simp at compute_girls
    exact compute_girls

end number_of_girls_l84_84523


namespace alices_favorite_number_l84_84251

theorem alices_favorite_number :
  ∃ n : ℕ, 80 < n ∧ n ≤ 130 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ ((n / 100) + (n % 100 / 10) + (n % 10)) % 4 = 0 ∧ n = 130 :=
by
  sorry

end alices_favorite_number_l84_84251


namespace solve_for_x_l84_84480

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l84_84480


namespace compare_a_b_c_l84_84614

noncomputable def a : ℝ := Real.log (Real.sqrt 2)
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := 1 / Real.exp 1

theorem compare_a_b_c : a < b ∧ b < c := by
  -- Proof will be done here
  sorry

end compare_a_b_c_l84_84614


namespace problem_statement_l84_84207

theorem problem_statement
  (x y : ℝ)
  (h1 : x * y = 1 / 9)
  (h2 : x * (y + 1) = 7 / 9)
  (h3 : y * (x + 1) = 5 / 18) :
  (x + 1) * (y + 1) = 35 / 18 :=
by
  sorry

end problem_statement_l84_84207


namespace smallest_multiple_of_36_with_digit_product_divisible_by_9_l84_84757

theorem smallest_multiple_of_36_with_digit_product_divisible_by_9 :
  ∃ n : ℕ, n > 0 ∧ n % 36 = 0 ∧ (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 * d2 * d3) % 9 = 0) ∧ n = 936 := 
by
  sorry

end smallest_multiple_of_36_with_digit_product_divisible_by_9_l84_84757


namespace mild_numbers_with_mild_squares_count_l84_84596

def is_mild (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 3, d = 0 ∨ d = 1

theorem mild_numbers_with_mild_squares_count :
  ∃ count : ℕ, count = 7 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → is_mild n → is_mild (n * n)) → count = 7 := by
  sorry

end mild_numbers_with_mild_squares_count_l84_84596


namespace inequality_system_solution_l84_84544

theorem inequality_system_solution (x : ℝ) (h1 : 5 - 2 * x ≤ 1) (h2 : x - 4 < 0) : 2 ≤ x ∧ x < 4 :=
  sorry

end inequality_system_solution_l84_84544


namespace quadratic_coefficients_l84_84384

theorem quadratic_coefficients :
  ∀ (a b c : ℤ), (2 * a * a - b * a - 5 = 0) → (a = 2 ∧ b = -1) :=
by
  intros a b c H
  sorry

end quadratic_coefficients_l84_84384


namespace find_x_l84_84486

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l84_84486


namespace original_price_l84_84589

variables (q r : ℝ) (h1 : 0 ≤ q) (h2 : 0 ≤ r)

theorem original_price (h : (2 : ℝ) = (1 + q / 100) * (1 - r / 100) * x) :
  x = 200 / (100 + q - r - (q * r) / 100) :=
by
  sorry

end original_price_l84_84589


namespace cos_double_angle_sub_pi_six_l84_84944

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π / 3)
variable (h2 : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5)

theorem cos_double_angle_sub_pi_six :
  Real.cos (2 * α - π / 6) = 4 / 5 :=
by
  sorry

end cos_double_angle_sub_pi_six_l84_84944


namespace parabola_equation_l84_84800

def parabola_condition (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 = a * x) ∨ (x^2 = b * y)

def point_on_parabola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (4, -2) ∧ parabola_condition P.1 P.2 a b

theorem parabola_equation :
  ∃ (a b : ℝ), point_on_parabola (4, -2) a b :=
by
  existsi (1)
  existsi (-8)
  split
  · exact rfl
  · left
    exact rfl
    sorry

end parabola_equation_l84_84800


namespace rational_t_l84_84241

variable (A B t : ℚ)

theorem rational_t (A B : ℚ) (hA : A = 2 * t / (1 + t^2)) (hB : B = (1 - t^2) / (1 + t^2)) : ∃ t' : ℚ, t = t' :=
by
  sorry

end rational_t_l84_84241


namespace bugs_eat_total_flowers_l84_84337

theorem bugs_eat_total_flowers :
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  total = 17 :=
by
  -- Applying given values to compute the total flowers eaten
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  
  -- Verify the total is 17
  have h_total : total = 17 := 
    by
    sorry

  -- Proving the final result
  exact h_total

end bugs_eat_total_flowers_l84_84337


namespace domain_of_log_function_l84_84052

-- Define the problematic quadratic function
def quadratic_fn (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Define the domain condition for our function
def domain_condition (x : ℝ) : Prop := quadratic_fn x > 0

-- The actual statement to prove, stating that the domain is (1, 3)
theorem domain_of_log_function :
  {x : ℝ | domain_condition x} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end domain_of_log_function_l84_84052


namespace expected_earnings_per_hour_l84_84415

def earnings_per_hour (words_per_minute earnings_per_word : ℝ) (earnings_per_article num_articles total_hours : ℕ) : ℝ :=
  let minutes_in_hour := 60
  let total_time := total_hours * minutes_in_hour
  let total_words := total_time * words_per_minute
  let word_earnings := total_words * earnings_per_word
  let article_earnings := earnings_per_article * num_articles
  (word_earnings + article_earnings) / total_hours

theorem expected_earnings_per_hour :
  earnings_per_hour 10 0.1 60 3 4 = 105 := by
  sorry

end expected_earnings_per_hour_l84_84415


namespace flag_yellow_area_percentage_l84_84901

theorem flag_yellow_area_percentage (s w : ℝ) (h_flag_area : s > 0)
  (h_width_positive : w > 0) (h_cross_area : 4 * s * w - 3 * w^2 = 0.49 * s^2) :
  (w^2 / s^2) * 100 = 12.25 :=
by
  sorry

end flag_yellow_area_percentage_l84_84901


namespace soda_cost_l84_84667

variable (b s f : ℝ)

noncomputable def keegan_equation : Prop :=
  3 * b + 2 * s + f = 975

noncomputable def alex_equation : Prop :=
  2 * b + 3 * s + f = 900

theorem soda_cost (h1 : keegan_equation b s f) (h2 : alex_equation b s f) : s = 18.75 :=
by
  sorry

end soda_cost_l84_84667


namespace simplify_radical_l84_84689

theorem simplify_radical (x : ℝ) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) :=
by
  sorry

end simplify_radical_l84_84689


namespace price_per_liter_l84_84372

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end price_per_liter_l84_84372


namespace completing_square_result_l84_84094

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l84_84094


namespace find_p_l84_84136

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p : ℕ) (h : is_prime p) (hpgt1 : 1 < p) :
  8 * p^4 - 3003 = 1997 ↔ p = 5 :=
by
  sorry

end find_p_l84_84136


namespace largest_divisible_n_l84_84601

/-- Largest positive integer n for which n^3 + 10 is divisible by n + 1 --/
theorem largest_divisible_n (n : ℕ) :
  n = 0 ↔ ∀ m : ℕ, (m > n) → ¬ ((m^3 + 10) % (m + 1) = 0) :=
by
  sorry

end largest_divisible_n_l84_84601


namespace three_digit_int_one_less_than_lcm_mult_l84_84398

theorem three_digit_int_one_less_than_lcm_mult : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n + 1) % Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 9 = 0 :=
sorry

end three_digit_int_one_less_than_lcm_mult_l84_84398


namespace chairs_to_exclude_l84_84889

theorem chairs_to_exclude (chairs : ℕ) (h : chairs = 765) : 
  ∃ n, n^2 ≤ chairs ∧ chairs - n^2 = 36 := 
by 
  sorry

end chairs_to_exclude_l84_84889


namespace four_digit_integer_l84_84693

theorem four_digit_integer (a b c d : ℕ) (h1 : a + b + c + d = 18)
  (h2 : b + c = 11) (h3 : a - d = 1) (h4 : 11 ∣ (1000 * a + 100 * b + 10 * c + d)) :
  1000 * a + 100 * b + 10 * c + d = 4653 :=
by sorry

end four_digit_integer_l84_84693


namespace smallest_of_three_numbers_l84_84911

theorem smallest_of_three_numbers : ∀ (a b c : ℕ), (a = 5) → (b = 8) → (c = 4) → min (min a b) c = 4 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  sorry

end smallest_of_three_numbers_l84_84911


namespace tan_11_pi_over_4_l84_84926

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l84_84926


namespace remainder_67pow67_add_67_div_68_l84_84107

-- Lean statement starting with the question and conditions translated to Lean

theorem remainder_67pow67_add_67_div_68 : 
  (67 ^ 67 + 67) % 68 = 66 := 
by
  -- Condition: 67 ≡ -1 mod 68
  have h : 67 % 68 = -1 % 68 := by norm_num
  sorry

end remainder_67pow67_add_67_div_68_l84_84107


namespace cost_per_chicken_l84_84686

-- Definitions for conditions
def totalBirds : ℕ := 15
def ducks : ℕ := totalBirds / 3
def chickens : ℕ := totalBirds - ducks
def feed_cost : ℕ := 20

-- Theorem stating the cost per chicken
theorem cost_per_chicken : (feed_cost / chickens) = 2 := by
  sorry

end cost_per_chicken_l84_84686


namespace ribbon_tying_length_l84_84066

theorem ribbon_tying_length :
  let l1 := 36
  let l2 := 42
  let l3 := 48
  let cut1 := l1 / 6
  let cut2 := l2 / 6
  let cut3 := l3 / 6
  let rem1 := l1 - cut1
  let rem2 := l2 - cut2
  let rem3 := l3 - cut3
  let total_rem := rem1 + rem2 + rem3
  let final_length := 97
  let tying_length := total_rem - final_length
  tying_length = 8 :=
by
  sorry

end ribbon_tying_length_l84_84066


namespace sequence_diff_l84_84456

theorem sequence_diff (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hSn : ∀ n, S n = n^2)
  (hS1 : a 1 = S 1)
  (ha_n : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 3 - a 2 = 2 := sorry

end sequence_diff_l84_84456


namespace mean_of_remaining_two_numbers_l84_84040

theorem mean_of_remaining_two_numbers :
  let n1 := 1871
  let n2 := 1997
  let n3 := 2023
  let n4 := 2029
  let n5 := 2113
  let n6 := 2125
  let n7 := 2137
  let total_sum := n1 + n2 + n3 + n4 + n5 + n6 + n7
  let known_mean := 2100
  let mean_of_other_two := 1397.5
  total_sum = 13295 →
  5 * known_mean = 10500 →
  total_sum - 10500 = 2795 →
  2795 / 2 = mean_of_other_two :=
by
  intros
  sorry

end mean_of_remaining_two_numbers_l84_84040


namespace probability_region_D_l84_84418

noncomputable def P_A : ℝ := 1 / 4
noncomputable def P_B : ℝ := 1 / 3
noncomputable def P_C : ℝ := 1 / 6

theorem probability_region_D (P_D : ℝ) (h : P_A + P_B + P_C + P_D = 1) : P_D = 1 / 4 :=
by
  sorry

end probability_region_D_l84_84418


namespace f_of_6_l84_84894

noncomputable def f (u : ℝ) : ℝ := 
  let x := (u + 2) / 4
  x^3 - x + 2

theorem f_of_6 : f 6 = 8 :=
by
  sorry

end f_of_6_l84_84894


namespace cost_of_filling_all_pots_l84_84752

def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny_per_plant : ℝ := 4.00
def num_creeping_jennies : ℝ := 4
def cost_geranium_per_plant : ℝ := 3.50
def num_geraniums : ℝ := 4
def cost_elephant_ear_per_plant : ℝ := 7.00
def num_elephant_ears : ℝ := 2
def cost_purple_fountain_grass_per_plant : ℝ := 6.00
def num_purple_fountain_grasses : ℝ := 3
def num_pots : ℝ := 4

def total_cost_per_pot : ℝ := 
  cost_palm_fern +
  (num_creeping_jennies * cost_creeping_jenny_per_plant) +
  (num_geraniums * cost_geranium_per_plant) +
  (num_elephant_ears * cost_elephant_ear_per_plant) +
  (num_purple_fountain_grasses * cost_purple_fountain_grass_per_plant)

def total_cost : ℝ := total_cost_per_pot * num_pots

theorem cost_of_filling_all_pots : total_cost = 308.00 := by
  sorry

end cost_of_filling_all_pots_l84_84752


namespace common_divisors_9240_8820_l84_84963

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l84_84963


namespace solve_percentage_increase_length_l84_84055

def original_length (L : ℝ) : Prop := true
def original_breadth (B : ℝ) : Prop := true

def new_breadth (B' : ℝ) (B : ℝ) : Prop := B' = 1.25 * B

def new_length (L' : ℝ) (L : ℝ) (x : ℝ) : Prop := L' = L * (1 + x / 100)

def original_area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

def new_area (A' : ℝ) (A : ℝ) : Prop := A' = 1.375 * A

def percentage_increase_length (x : ℝ) : Prop := x = 10

theorem solve_percentage_increase_length (L B A A' L' B' x : ℝ)
  (hL : original_length L)
  (hB : original_breadth B)
  (hB' : new_breadth B' B)
  (hL' : new_length L' L x)
  (hA : original_area L B A)
  (hA' : new_area A' A)
  (h_eqn : L' * B' = A') :
  percentage_increase_length x :=
by
  sorry

end solve_percentage_increase_length_l84_84055


namespace athletes_in_camp_hours_l84_84895

theorem athletes_in_camp_hours (initial_athletes : ℕ) (left_rate : ℕ) (left_hours : ℕ) (arrived_rate : ℕ) 
  (difference : ℕ) (hours : ℕ) 
  (h_initial: initial_athletes = 300) 
  (h_left_rate: left_rate = 28) 
  (h_left_hours: left_hours = 4) 
  (h_arrived_rate: arrived_rate = 15) 
  (h_difference: difference = 7) 
  (h_left: left_rate * left_hours = 112) 
  (h_equation: initial_athletes - (left_rate * left_hours) + (arrived_rate * hours) = initial_athletes - difference) : 
  hours = 7 :=
by
  sorry

end athletes_in_camp_hours_l84_84895


namespace find_radius_of_circle_l84_84620

theorem find_radius_of_circle
  (a b R : ℝ)
  (h1 : R^2 = a * b) :
  R = Real.sqrt (a * b) :=
by
  sorry

end find_radius_of_circle_l84_84620


namespace probability_of_meeting_l84_84552

theorem probability_of_meeting (α x y : ℝ) (hα : α = 10) (hx : 0 ≤ x ∧ x ≤ 60) (hy : 0 ≤ y ∧ y ≤ 60) :
  let p := 1 - ((1 - α / 60) ^ 2) in p = 11 / 36 :=
by
  have h1 : α / 60 = 10 / 60 := by rw hα
  have h2 : 1 - 10 / 60 = 5 / 6 := by norm_num
  have h3 : (5 / 6) ^ 2 = 25 / 36 := by norm_num
  have h4 : 1 - 25 / 36 = 11 / 36 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num

end probability_of_meeting_l84_84552


namespace cost_of_pen_is_51_l84_84042

-- Definitions of variables and conditions
variables {p q : ℕ}
variables (h1 : 6 * p + 2 * q = 348)
variables (h2 : 3 * p + 4 * q = 234)

-- Goal: Prove the cost of a pen (p) is 51 cents
theorem cost_of_pen_is_51 : p = 51 :=
by
  -- placeholder for the proof
  sorry

end cost_of_pen_is_51_l84_84042


namespace distance_from_A_to_directrix_l84_84312

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l84_84312


namespace average_weight_increase_l84_84536

variable (A N X : ℝ)

theorem average_weight_increase (hN : N = 135.5) (h_avg : A + X = (9 * A - 86 + N) / 9) : 
  X = 5.5 :=
by
  sorry

end average_weight_increase_l84_84536


namespace number_of_cats_l84_84775

def cats_on_ship (C S : ℕ) : Prop :=
  (C + S + 2 = 16) ∧ (4 * C + 2 * S + 3 = 45)

theorem number_of_cats (C S : ℕ) (h : cats_on_ship C S) : C = 7 :=
by
  sorry

end number_of_cats_l84_84775


namespace perfect_game_points_l84_84778

theorem perfect_game_points (points_per_game games_played total_points : ℕ) 
  (h1 : points_per_game = 21) 
  (h2 : games_played = 11) 
  (h3 : total_points = points_per_game * games_played) : 
  total_points = 231 := 
by 
  sorry

end perfect_game_points_l84_84778


namespace arithmetic_sequence_sixth_term_l84_84870

variables (a d : ℤ)

theorem arithmetic_sequence_sixth_term :
  a + (a + d) + (a + 2 * d) = 12 →
  a + 3 * d = 0 →
  a + 5 * d = -4 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sixth_term_l84_84870


namespace find_n_l84_84019

noncomputable def positive_geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : positive_geometric_sequence) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def conditions (a : positive_geometric_sequence) :=
  is_geometric_sequence a ∧
  a 0 * a 1 * a 2 = 4 ∧
  a 3 * a 4 * a 5 = 12 ∧
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324

theorem find_n (a : positive_geometric_sequence) (h : conditions a) : ∃ n : ℕ, n = 14 :=
by
  sorry

end find_n_l84_84019


namespace milk_cans_l84_84120

theorem milk_cans (x y : ℕ) (h : 10 * x + 17 * y = 206) : x = 7 ∧ y = 8 := sorry

end milk_cans_l84_84120


namespace number_of_blue_parrots_l84_84035

noncomputable def total_parrots : ℕ := 108
def fraction_blue_parrots : ℚ := 1 / 6

theorem number_of_blue_parrots : (fraction_blue_parrots * total_parrots : ℚ) = 18 := 
by
  sorry

end number_of_blue_parrots_l84_84035


namespace doves_eggs_l84_84943

theorem doves_eggs (initial_doves total_doves : ℕ) (fraction_hatched : ℚ) (E : ℕ)
  (h_initial_doves : initial_doves = 20)
  (h_total_doves : total_doves = 65)
  (h_fraction_hatched : fraction_hatched = 3/4)
  (h_after_hatching : total_doves = initial_doves + fraction_hatched * E * initial_doves) :
  E = 3 :=
by
  -- The proof would go here.
  sorry

end doves_eggs_l84_84943


namespace find_original_number_l84_84081

theorem find_original_number (x : ℤ) (h : (x + 19) % 25 = 0) : x = 6 :=
sorry

end find_original_number_l84_84081


namespace stratified_sampling_BA3_count_l84_84772

-- Defining the problem parameters
def num_Om_BA1 : ℕ := 60
def num_Om_BA2 : ℕ := 20
def num_Om_BA3 : ℕ := 40
def total_sample_size : ℕ := 30

-- Proving using stratified sampling
theorem stratified_sampling_BA3_count : 
  (total_sample_size * num_Om_BA3 / (num_Om_BA1 + num_Om_BA2 + num_Om_BA3)) = 10 :=
by
  -- Since Lean doesn't handle reals and integers simplistically,
  -- we need to translate the division and multiplication properly.
  sorry

end stratified_sampling_BA3_count_l84_84772


namespace remainder_modulo_9_l84_84937

noncomputable def power10 := 10^15
noncomputable def power3  := 3^15

theorem remainder_modulo_9 : (7 * power10 + power3) % 9 = 7 := by
  -- Define the conditions given in the problem
  have h1 : (10 % 9 = 1) := by 
    norm_num
  have h2 : (3^2 % 9 = 0) := by 
    norm_num
  
  -- Utilize these conditions to prove the statement
  sorry

end remainder_modulo_9_l84_84937


namespace common_divisors_9240_8820_l84_84964

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l84_84964


namespace count_odd_perfect_squares_less_than_16000_l84_84000

theorem count_odd_perfect_squares_less_than_16000 : 
  ∃ n : ℕ, n = 31 ∧ ∀ k < 16000, 
    ∃ b : ℕ, b = 2 * n + 1 ∧ k = (4 * n + 3) ^ 2 ∧ (∃ m : ℕ, m = b + 1 ∧ m % 2 = 0) := 
sorry

end count_odd_perfect_squares_less_than_16000_l84_84000


namespace consecutive_integers_product_l84_84721

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l84_84721


namespace sum_bases_exponents_max_product_l84_84668

theorem sum_bases_exponents_max_product (A : ℕ) (hA : A = 3 ^ 670 * 2 ^ 2) : 
    (3 + 2 + 670 + 2 = 677) := by
  sorry

end sum_bases_exponents_max_product_l84_84668


namespace num_four_digit_36_combinations_l84_84636

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l84_84636


namespace y_in_terms_of_x_l84_84491

theorem y_in_terms_of_x (p x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) := 
by 
  sorry

end y_in_terms_of_x_l84_84491


namespace remainder_7_pow_63_mod_8_l84_84560

theorem remainder_7_pow_63_mod_8 : 7^63 % 8 = 7 :=
by sorry

end remainder_7_pow_63_mod_8_l84_84560


namespace containers_per_truck_l84_84738

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l84_84738


namespace compute_expression_l84_84190

noncomputable def quadratic_roots (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α * β = -2) ∧ (α + β = -p) ∧ (γ * δ = -2) ∧ (γ + δ = -q)

theorem compute_expression (p q α β γ δ : ℝ) 
  (h₁ : quadratic_roots p q α β γ δ) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) :=
by
  -- We will provide the proof here
  sorry

end compute_expression_l84_84190


namespace value_of_a_plus_b_l84_84655

theorem value_of_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 3 ↔ ax^2 + bx + 3 < 0) :
  a + b = -3 :=
sorry

end value_of_a_plus_b_l84_84655


namespace melanie_phil_ages_l84_84007

theorem melanie_phil_ages (A B : ℕ) 
  (h : (A + 10) * (B + 10) = A * B + 400) :
  (A + 6) + (B + 6) = 42 :=
by
  sorry

end melanie_phil_ages_l84_84007


namespace exists_infinitely_many_natural_numbers_factors_l84_84365

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l84_84365


namespace differential_solution_l84_84205

noncomputable def solve_differential_equation (C1 C2 : ℝ) : C∞ ≃ (ℝ → ℝ) :=
  λ x: ℝ, C1 * Real.cos (2 * x) + C2 * Real.sin (2 * x) + (2 * Real.cos (2 * x) + 8 * Real.sin (2 * x)) * x + (1/2 : ℝ) * Real.exp (2 * x)

theorem differential_solution (y : ℝ → ℝ)
    (h : ∀ x, deriv (deriv y x) + 4 * y x = -8 * Real.sin (2 * x) + 32 * Real.cos (2 * x) + 4 * Real.exp (2 * x)) : 
    ∃ C1 C2 : ℝ, y = solve_differential_equation C1 C2 :=
by
  sorry

end differential_solution_l84_84205


namespace find_AB_l84_84979

variables {AB CD AD BC AP PD APD PQ Q: ℝ}

def is_rectangle (ABCD : Prop) := ABCD

variables (P_on_BC : Prop)
variable (BP CP: ℝ)
variable (tan_angle_APD: ℝ)

theorem find_AB (ABCD : Prop) (P_on_BC : Prop) (BP CP: ℝ) (tan_angle_APD: ℝ) : 
  is_rectangle ABCD →
  P_on_BC →
  BP = 24 →
  CP = 12 →
  tan_angle_APD = 2 →
  AB = 27 := 
by
  sorry

end find_AB_l84_84979


namespace Sarah_skateboard_speed_2160_mph_l84_84197

-- Definitions based on the conditions
def miles_to_inches (miles : ℕ) : ℕ := miles * 63360
def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

/-- Pete walks backwards 3 times faster than Susan walks forwards --/
def Susan_walks_forwards_speed (pete_walks_hands_speed : ℕ) : ℕ := pete_walks_hands_speed / 3

/-- Tracy does cartwheels twice as fast as Susan walks forwards --/
def Tracy_cartwheels_speed (susan_walks_forwards_speed : ℕ) : ℕ := susan_walks_forwards_speed * 2

/-- Mike swims 8 times faster than Tracy does cartwheels --/
def Mike_swims_speed (tracy_cartwheels_speed : ℕ) : ℕ := tracy_cartwheels_speed * 8

/-- Pete can walk on his hands at 1/4 the speed Tracy can do cartwheels --/
def Pete_walks_hands_speed : ℕ := 2

/-- Pete rides his bike 5 times faster than Mike swims --/
def Pete_rides_bike_speed (mike_swims_speed : ℕ) : ℕ := mike_swims_speed * 5

/-- Patty can row 3 times faster than Pete walks backwards (in feet per hour) --/
def Patty_rows_speed (pete_walks_backwards_speed : ℕ) : ℕ := pete_walks_backwards_speed * 3

/-- Sarah can skateboard 6 times faster than Patty rows (in miles per minute) --/
def Sarah_skateboards_speed (patty_rows_speed_ft_per_hr : ℕ) : ℕ := (patty_rows_speed_ft_per_hr * 6 * 60) * 63360 * 60

theorem Sarah_skateboard_speed_2160_mph : Sarah_skateboards_speed (Patty_rows_speed (Pete_walks_hands_speed * 3)) = 2160 * 63360 * 60 :=
by
  sorry

end Sarah_skateboard_speed_2160_mph_l84_84197


namespace real_solutions_count_is_two_l84_84796

def equation_has_two_real_solutions (a b c : ℝ) : Prop :=
  (3*a^2 - 8*b + 2 = c) → (∀ x : ℝ, 3*x^2 - 8*x + 2 = 0) → ∃! x₁ x₂ : ℝ, (3*x₁^2 - 8*x₁ + 2 = 0) ∧ (3*x₂^2 - 8*x₂ + 2 = 0)

theorem real_solutions_count_is_two : equation_has_two_real_solutions (3 : ℝ) (-8 : ℝ) (2 : ℝ) := by
  sorry

end real_solutions_count_is_two_l84_84796


namespace inequality_proof_l84_84848

open Real

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l84_84848


namespace volume_of_cone_l84_84171

theorem volume_of_cone (r l h V : ℝ) (r_eq : π * r^2 = π)
  (l_eq : l = 2 * r)
  (h_eq : h = sqrt (l^2 - r^2))
  (V_eq : V = (1 / 3) * π * r^2 * h) :
  V = (sqrt 3 * π / 3) :=
by
  have r_val : r = 1,
  {
    sorry
  },
  have l_val : l = 2,
  {
    sorry
  },
  have h_val : h = sqrt 3,
  {
    sorry
  },
  have V_val : V = (sqrt 3 * π / 3),
  {
    sorry
  },
  exact V_val

end volume_of_cone_l84_84171


namespace probability_three_friends_same_lunch_group_l84_84512

noncomputable def probability_three_friends_same_group : ℝ :=
  let groups := 4
  let probability := (1 / groups) * (1 / groups)
  probability

theorem probability_three_friends_same_lunch_group :
  probability_three_friends_same_group = 1 / 16 :=
by
  unfold probability_three_friends_same_group
  sorry

end probability_three_friends_same_lunch_group_l84_84512


namespace max_marks_l84_84856

theorem max_marks (M : ℕ) (h_pass : 55 / 100 * M = 510) : M = 928 :=
sorry

end max_marks_l84_84856


namespace subtract_fractions_correct_l84_84292

theorem subtract_fractions_correct :
  (3 / 8 + 5 / 12 - 1 / 6) = (5 / 8) := by
sorry

end subtract_fractions_correct_l84_84292


namespace shortest_distance_l84_84187

noncomputable def shortestDistanceCD (c : ℝ) : ℝ :=
  abs (c^2 - 7*c + 12) / sqrt 10

theorem shortest_distance : ∃ c : ℝ, c = 3.5 ∧
  C = (c, c^2 - 4*c + 7) ∧
  D = (c, 3*c - 5) ∧
  shortestDistanceCD c = 0.25 / sqrt 10 :=
sorry

end shortest_distance_l84_84187


namespace ellipse_major_axis_min_length_l84_84321

theorem ellipse_major_axis_min_length (a b c : ℝ) 
  (h1 : b * c = 2)
  (h2 : a^2 = b^2 + c^2) 
  : 2 * a ≥ 4 :=
sorry

end ellipse_major_axis_min_length_l84_84321


namespace cassie_nails_l84_84278

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l84_84278


namespace price_of_one_liter_l84_84374

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end price_of_one_liter_l84_84374


namespace find_k_value_l84_84495

theorem find_k_value (k : ℝ) : 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0 ∧
    (x1^2 - 1) * (x1^2 - 4) = k ∧
    (x2^2 - 1) * (x2^2 - 4) = k ∧
    (x3^2 - 1) * (x3^2 - 4) = k ∧
    (x4^2 - 1) * (x4^2 - 4) = k ∧
    x1 ≠ x2 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
    x4 - x3 = x3 - x2 ∧ x2 - x1 = x4 - x3) → 
  k = 7/4 := 
by
  sorry

end find_k_value_l84_84495


namespace expression_equals_100_l84_84403

-- Define the terms in the numerator and their squares
def num1 := 0.02
def num2 := 0.52
def num3 := 0.035

def num1_sq := num1^2
def num2_sq := num2^2
def num3_sq := num3^2

-- Define the terms in the denominator and their squares
def denom1 := 0.002
def denom2 := 0.052
def denom3 := 0.0035

def denom1_sq := denom1^2
def denom2_sq := denom2^2
def denom3_sq := denom3^2

-- Define the sums of the squares
def sum_numerator := num1_sq + num2_sq + num3_sq
def sum_denominator := denom1_sq + denom2_sq + denom3_sq

-- Define the final expression
def expression := sum_numerator / sum_denominator

-- Prove the expression equals the correct answer
theorem expression_equals_100 : expression = 100 := by sorry

end expression_equals_100_l84_84403


namespace consecutive_integers_sum_l84_84707

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l84_84707


namespace deepak_age_is_21_l84_84727

noncomputable def DeepakCurrentAge (x : ℕ) : Prop :=
  let Rahul := 4 * x
  let Deepak := 3 * x
  let Karan := 5 * x
  Rahul + 6 = 34 ∧
  (Rahul + 6) / 7 = (Deepak + 6) / 5 ∧ (Rahul + 6) / 7 = (Karan + 6) / 9 → 
  Deepak = 21

theorem deepak_age_is_21 : ∃ x : ℕ, DeepakCurrentAge x :=
by
  use 7
  sorry

end deepak_age_is_21_l84_84727


namespace smallest_four_digit_divisible_by_34_l84_84756

/-- Define a four-digit number. -/
def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

/-- Define a number to be divisible by another number. -/
def divisible_by (n k : ℕ) : Prop :=
  k ∣ n

/-- Prove that the smallest four-digit number divisible by 34 is 1020. -/
theorem smallest_four_digit_divisible_by_34 : ∃ n : ℕ, is_four_digit n ∧ divisible_by n 34 ∧ 
    (∀ m : ℕ, is_four_digit m → divisible_by m 34 → n ≤ m) :=
  sorry

end smallest_four_digit_divisible_by_34_l84_84756


namespace factorize1_factorize2_factorize3_l84_84288

theorem factorize1 (x : ℝ) : x^3 + 6 * x^2 + 9 * x = x * (x + 3)^2 := 
  sorry

theorem factorize2 (x y : ℝ) : 16 * x^2 - 9 * y^2 = (4 * x - 3 * y) * (4 * x + 3 * y) := 
  sorry

theorem factorize3 (x y : ℝ) : (3 * x + y)^2 - (x - 3 * y) * (3 * x + y) = 2 * (3 * x + y) * (x + 2 * y) := 
  sorry

end factorize1_factorize2_factorize3_l84_84288


namespace hyperbola_asymptotes_l84_84380

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 16 * x^2 - 9 * y^2 = -144 → (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intros x y h1
  sorry

end hyperbola_asymptotes_l84_84380


namespace base7_addition_problem_l84_84127

theorem base7_addition_problem
  (X Y : ℕ) :
  (5 * 7^1 + X * 7^0 + Y * 7^0 + 0 * 7^2 + 6 * 7^1 + 2 * 7^0) = (6 * 7^1 + 4 * 7^0 + X * 7^0 + 0 * 7^2) →
  X + 6 = 1 * 7 + 4 →
  Y + 2 = X →
  X + Y = 8 :=
by
  intro h1 h2 h3
  sorry

end base7_addition_problem_l84_84127


namespace gcd_gx_x_is_450_l84_84813

def g (x : ℕ) : ℕ := (3 * x + 2) * (8 * x + 3) * (14 * x + 5) * (x + 15)

noncomputable def gcd_gx_x (x : ℕ) (h : 49356 ∣ x) : ℕ :=
  Nat.gcd (g x) x

theorem gcd_gx_x_is_450 (x : ℕ) (h : 49356 ∣ x) : gcd_gx_x x h = 450 := by
  sorry

end gcd_gx_x_is_450_l84_84813


namespace sonya_fell_times_l84_84862

theorem sonya_fell_times (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) :
  steven_falls = 3 →
  stephanie_falls = steven_falls + 13 →
  sonya_falls = 6 →
  sonya_falls = (stephanie_falls / 2) - 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at *
  sorry

end sonya_fell_times_l84_84862


namespace triangle_angles_l84_84341

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180

theorem triangle_angles (x : ℝ) (hA : A = x) (hB : B = 2 * A) (hC : C + A + B = 180) :
  A = x ∧ B = 2 * x ∧ C = 180 - 3 * x := by
  -- proof goes here
  sorry

end triangle_angles_l84_84341


namespace find_number_of_olives_l84_84047

theorem find_number_of_olives (O : ℕ)
  (lettuce_choices : 2 = 2)
  (tomato_choices : 3 = 3)
  (soup_choices : 2 = 2)
  (total_combos : 2 * 3 * O * 2 = 48) :
  O = 4 :=
by
  sorry

end find_number_of_olives_l84_84047


namespace Jake_weight_loss_l84_84005

variables (J K x : ℕ)

theorem Jake_weight_loss : 
  J = 198 ∧ J + K = 293 ∧ J - x = 2 * K → x = 8 := 
by {
  sorry
}

end Jake_weight_loss_l84_84005


namespace geom_seq_product_l84_84662

-- Given conditions
variables (a : ℕ → ℝ)
variable (r : ℝ)
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom a1_eq_1 : a 1 = 1
axiom a10_eq_3 : a 10 = 3

-- Proof goal
theorem geom_seq_product : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 :=  
sorry

end geom_seq_product_l84_84662


namespace common_divisors_9240_8820_l84_84961

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l84_84961


namespace pq_plus_four_mul_l84_84189

open Real

theorem pq_plus_four_mul {p q : ℝ} (h1 : (x - 4) * (3 * x + 11) = x ^ 2 - 19 * x + 72) 
  (hpq1 : 2 * p ^ 2 + 18 * p - 116 = 0) (hpq2 : 2 * q ^ 2 + 18 * q - 116 = 0) (hpq_ne : p ≠ q) : 
  (p + 4) * (q + 4) = -78 := 
sorry

end pq_plus_four_mul_l84_84189


namespace math_proof_l84_84161

open Real

noncomputable def function (a b x : ℝ): ℝ := a * x^3 + b * x^2

theorem math_proof (a b : ℝ) :
  (function a b 1 = 3) ∧
  (deriv (function a b) 1 = 0) ∧
  (∃ (a b : ℝ), a = -6 ∧ b = 9 ∧ 
    function a b = -6 * (x^3) + 9 * (x^2)) ∧
  (∀ x, (0 < x ∧ x < 1) → deriv (function a b) x > 0) ∧
  (∀ x, (x < 0 ∨ x > 1) → deriv (function a b) x < 0) ∧
  (min (function a b (-2)) (function a b 2) = (-12)) ∧
  (max (function a b (-2)) (function a b 2) = 84) :=
by
  sorry

end math_proof_l84_84161


namespace find_number_l84_84763

theorem find_number (n : ℝ) (h : (1/2) * n + 5 = 11) : n = 12 :=
by
  sorry

end find_number_l84_84763


namespace infinite_natural_numbers_with_factored_polynomial_l84_84371

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l84_84371


namespace evaluate_expression_l84_84438

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end evaluate_expression_l84_84438


namespace consecutive_integers_product_l84_84724

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l84_84724


namespace opposite_sides_of_line_l84_84649

theorem opposite_sides_of_line (m : ℝ) (h : (2 * (-2 : ℝ) + m - 2) * (2 * m + 4 - 2) < 0) : -1 < m ∧ m < 6 :=
sorry

end opposite_sides_of_line_l84_84649


namespace deaths_during_operation_l84_84795

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end deaths_during_operation_l84_84795


namespace find_a_l84_84305

theorem find_a (x y a : ℕ) (h₁ : x = 2) (h₂ : y = 3) (h₃ : a * x + 3 * y = 13) : a = 2 :=
by 
  sorry

end find_a_l84_84305


namespace find_first_factor_of_lcm_l84_84700

theorem find_first_factor_of_lcm (hcf : ℕ) (A : ℕ) (X : ℕ) (B : ℕ) (lcm_val : ℕ) 
  (h_hcf : hcf = 59)
  (h_A : A = 944)
  (h_lcm_val : lcm_val = 59 * X * 16)
  (h_A_lcm : A = lcm_val) :
  X = 1 := 
by
  sorry

end find_first_factor_of_lcm_l84_84700


namespace largest_garden_is_candace_and_difference_is_100_l84_84785

-- Define the dimensions of the gardens
def area_alice : Nat := 30 * 50
def area_bob : Nat := 35 * 45
def area_candace : Nat := 40 * 40

-- The proof goal
theorem largest_garden_is_candace_and_difference_is_100 :
  area_candace > area_alice ∧ area_candace > area_bob ∧ area_candace - area_alice = 100 := by
    sorry

end largest_garden_is_candace_and_difference_is_100_l84_84785


namespace azalea_wool_price_l84_84784

noncomputable def sheep_count : ℕ := 200
noncomputable def wool_per_sheep : ℕ := 10
noncomputable def shearing_cost : ℝ := 2000
noncomputable def profit : ℝ := 38000

-- Defining total wool and total revenue based on these definitions
noncomputable def total_wool : ℕ := sheep_count * wool_per_sheep
noncomputable def total_revenue : ℝ := profit + shearing_cost
noncomputable def price_per_pound : ℝ := total_revenue / total_wool

-- Problem statement: Proving that the price per pound of wool is equal to $20
theorem azalea_wool_price :
  price_per_pound = 20 := 
sorry

end azalea_wool_price_l84_84784


namespace sales_discount_percentage_l84_84409

theorem sales_discount_percentage :
  ∀ (P N : ℝ) (D : ℝ),
  (N * 1.12 * (P * (1 - D / 100)) = P * N * (1 + 0.008)) → D = 10 :=
by
  intros P N D h
  sorry

end sales_discount_percentage_l84_84409


namespace q_poly_correct_l84_84045

open Polynomial

noncomputable def q : Polynomial ℚ := 
  -(C 1) * X^6 + C 4 * X^4 + C 21 * X^3 + C 15 * X^2 + C 14 * X + C 3

theorem q_poly_correct : 
  ∀ x : Polynomial ℚ,
  q + (X^6 + 4 * X^4 + 5 * X^3 + 12 * X) = 
  (8 * X^4 + 26 * X^3 + 15 * X^2 + 26 * X + C 3) := by sorry

end q_poly_correct_l84_84045


namespace machine_output_l84_84823

theorem machine_output (input : ℕ) (output : ℕ) (h : input = 26) (h_out : output = input + 15 - 6) : output = 35 := 
by 
  sorry

end machine_output_l84_84823


namespace vehicles_count_l84_84986

theorem vehicles_count (T : ℕ) : 
    2 * T + 3 * (2 * T) + (T / 2) + T = 180 → 
    T = 19 ∧ 2 * T = 38 ∧ 3 * (2 * T) = 114 ∧ (T / 2) = 9 := 
by 
    intros h
    sorry

end vehicles_count_l84_84986


namespace average_burning_time_probability_l84_84854

variables (N : ℕ) (n : ℕ) (Xbar : ℝ) (Xtilde : ℝ) (var: ℝ)

def sample_properties 
  (hN : N = 5000) 
  (hn : n = 300) 
  (hXtilde : Xtilde = 1450) 
  (hvar : var = 40000) : Prop :=
  let sem := real.sqrt (var / n * (1 - n / N)) in
  P (1410 < Xbar ∧ Xbar < 1490) = 0.99964

theorem average_burning_time_probability (hN : N = 5000) 
  (hn : n = 300) 
  (hXtilde : Xtilde = 1450) 
  (hvar : var = 40000) :
  sample_properties N n Xbar Xtilde var :=
sorry

end average_burning_time_probability_l84_84854


namespace derivative_of_log_base_3_derivative_of_exp_base_2_l84_84759

noncomputable def log_base_3_deriv (x : ℝ) : ℝ := (Real.log x / Real.log 3)
noncomputable def exp_base_2_deriv (x : ℝ) : ℝ := Real.exp (x * Real.log 2)

theorem derivative_of_log_base_3 (x : ℝ) (h : x > 0) :
  (log_base_3_deriv x) = (1 / (x * Real.log 3)) :=
by
  sorry

theorem derivative_of_exp_base_2 (x : ℝ) :
  (exp_base_2_deriv x) = (Real.exp (x * Real.log 2) * Real.log 2) :=
by
  sorry

end derivative_of_log_base_3_derivative_of_exp_base_2_l84_84759


namespace volume_of_cone_l84_84170

theorem volume_of_cone
  (r h l : ℝ) -- declaring variables
  (base_area : ℝ) (lateral_surface_is_semicircle : ℝ) 
  (h_eq : h = Real.sqrt (l^2 - r^2))
  (base_area_eq : π * r^2 = π)
  (lateral_surface_eq : π * l = 2 * π * r) : 
  (∀ (V : ℝ), V = (1 / 3) * π * r^2 * h → V = (Real.sqrt 3) / 3 * π) :=
by
  sorry

end volume_of_cone_l84_84170


namespace Isabel_earning_l84_84664

-- Define the number of bead necklaces sold
def bead_necklaces : ℕ := 3

-- Define the number of gem stone necklaces sold
def gemstone_necklaces : ℕ := 3

-- Define the cost of each necklace
def cost_per_necklace : ℕ := 6

-- Calculate the total number of necklaces sold
def total_necklaces : ℕ := bead_necklaces + gemstone_necklaces

-- Calculate the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings is 36 dollars
theorem Isabel_earning : total_earnings = 36 := by
  sorry

end Isabel_earning_l84_84664


namespace books_on_each_shelf_l84_84167

theorem books_on_each_shelf (M P x : ℕ) (h1 : 3 * M + 5 * P = 72) (h2 : M = x) (h3 : P = x) : x = 9 :=
by
  sorry

end books_on_each_shelf_l84_84167


namespace fraction_division_l84_84083

theorem fraction_division (a b c d : ℚ) (h1 : a = 3) (h2 : b = 8) (h3 : c = 5) (h4 : d = 12) :
  (a / b) / (c / d) = 9 / 10 :=
by
  sorry

end fraction_division_l84_84083


namespace probability_of_shaded_shape_l84_84505

   def total_shapes : ℕ := 4
   def shaded_shapes : ℕ := 1

   theorem probability_of_shaded_shape : shaded_shapes / total_shapes = 1 / 4 := 
   by
     sorry
   
end probability_of_shaded_shape_l84_84505


namespace interval_monotonically_decreasing_l84_84162

-- Definitions based on conditions
def power_function (α : ℝ) (x : ℝ) : ℝ := x^α
def passes_through_point (α : ℝ) : Prop := power_function α 2 = 4

-- Statement of the problem as a theorem
theorem interval_monotonically_decreasing : ∀ α, passes_through_point α → ∀ x : ℝ, x < 0 → deriv (power_function α) x < 0 := sorry

end interval_monotonically_decreasing_l84_84162


namespace curve_intersection_l84_84663

noncomputable def C1 (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (2 * t + 2 * a, -t)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ, 1 + 2 * Real.cos θ)

theorem curve_intersection (a : ℝ) :
  (∃ t θ : ℝ, C1 t a = C2 θ) ↔ 1 - Real.sqrt 5 ≤ a ∧ a ≤ 1 + Real.sqrt 5 :=
sorry

end curve_intersection_l84_84663


namespace correct_option_D_l84_84230

theorem correct_option_D (defect_rate_products : ℚ)
                         (rain_probability : ℚ)
                         (cure_rate_hospital : ℚ)
                         (coin_toss_heads_probability : ℚ)
                         (coin_toss_tails_probability : ℚ):
  defect_rate_products = 1/10 →
  rain_probability = 0.9 →
  cure_rate_hospital = 0.1 →
  coin_toss_heads_probability = 0.5 →
  coin_toss_tails_probability = 0.5 →
  coin_toss_tails_probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5
  exact h5

end correct_option_D_l84_84230


namespace remainder_ab_div_48_is_15_l84_84881

noncomputable def remainder_ab_div_48 (a b : ℕ) (ha : a % 8 = 3) (hb : b % 6 = 5) : ℕ :=
  (a * b) % 48

theorem remainder_ab_div_48_is_15 {a b : ℕ} (ha : a % 8 = 3) (hb : b % 6 = 5) : remainder_ab_div_48 a b ha hb = 15 :=
  sorry

end remainder_ab_div_48_is_15_l84_84881


namespace minimum_value_of_function_l84_84946

theorem minimum_value_of_function (x : ℝ) (hx : x > 5 / 4) : 
  ∃ y, y = 4 * x + 1 / (4 * x - 5) ∧ y = 7 :=
sorry

end minimum_value_of_function_l84_84946


namespace pythagorean_set_A_l84_84100

theorem pythagorean_set_A : 
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  x^2 + y^2 = z^2 := 
by
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  sorry

end pythagorean_set_A_l84_84100


namespace books_more_than_movies_l84_84218

-- Define the number of movies and books in the "crazy silly school" series.
def num_movies : ℕ := 14
def num_books : ℕ := 15

-- State the theorem to prove there is 1 more book than movies.
theorem books_more_than_movies : num_books - num_movies = 1 :=
by 
  -- Proof is omitted.
  sorry

end books_more_than_movies_l84_84218


namespace no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l84_84602

theorem no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018 (m n : ℕ) : ¬ (m^2 = n^2 + 2018) :=
sorry

end no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l84_84602


namespace line_positional_relationship_l84_84464

variables {Point Line Plane : Type}

-- Definitions of the conditions
def is_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
def is_within_plane (b : Line) (α : Plane) : Prop := sorry
def no_common_point (a b : Line) : Prop := sorry
def parallel_or_skew (a b : Line) : Prop := sorry

-- Proof statement in Lean
theorem line_positional_relationship
  (a b : Line) (α : Plane)
  (h₁ : is_parallel_to_plane a α)
  (h₂ : is_within_plane b α)
  (h₃ : no_common_point a b) :
  parallel_or_skew a b :=
sorry

end line_positional_relationship_l84_84464


namespace candidate_percentage_l84_84243

theorem candidate_percentage (P : ℝ) (h : (P / 100) * 7800 + 2340 = 7800) : P = 70 :=
sorry

end candidate_percentage_l84_84243


namespace combinatorial_identity_l84_84267

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end combinatorial_identity_l84_84267


namespace scientific_notation_l84_84908

theorem scientific_notation : (0.000000005 : ℝ) = 5 * 10^(-9 : ℤ) := 
by
  sorry

end scientific_notation_l84_84908


namespace bird_wings_l84_84985

theorem bird_wings (birds wings_per_bird : ℕ) (h1 : birds = 13) (h2 : wings_per_bird = 2) : birds * wings_per_bird = 26 := by
  sorry

end bird_wings_l84_84985


namespace total_number_of_students_l84_84871

theorem total_number_of_students 
    (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat) 
    (h1 : group1 = 5) (h2 : group2 = 8) (h3 : group3 = 7) (h4 : group4 = 4) : 
    group1 + group2 + group3 + group4 = 24 := 
by
  sorry

end total_number_of_students_l84_84871


namespace power_relationship_l84_84451

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end power_relationship_l84_84451


namespace sum_of_three_consecutive_integers_product_336_l84_84716

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l84_84716


namespace number_of_days_at_Tom_house_l84_84874

-- Define the constants and conditions
def total_people := 6
def plates_per_person_per_day := 6
def total_plates := 144

-- Prove that the number of days they were at Tom's house is 4
theorem number_of_days_at_Tom_house : total_plates / (total_people * plates_per_person_per_day) = 4 :=
  sorry

end number_of_days_at_Tom_house_l84_84874


namespace dodecahedron_edge_probability_l84_84746

def dodecahedron_vertices : ℕ := 20

def vertex_connections : ℕ := 3

theorem dodecahedron_edge_probability :
  (∃ (u v : fin dodecahedron_vertices), u ≠ v ∧ u.1 < vertex_connections → 
    (Pr (u, v) = 3 / (dodecahedron_vertices - 1))) :=
sorry

end dodecahedron_edge_probability_l84_84746


namespace custom_deck_card_selection_l84_84579

theorem custom_deck_card_selection :
  let cards := 60
  let suits := 4
  let cards_per_suit := 15
  let red_suits := 2
  let black_suits := 2
  -- Total number of ways to pick two cards with the second of a different color
  ∃ (ways : ℕ), ways = 60 * 30 ∧ ways = 1800 := by
  sorry

end custom_deck_card_selection_l84_84579


namespace find_number_l84_84777

theorem find_number (x : ℤ) (h : x = 5 * (x - 4)) : x = 5 :=
by {
  sorry
}

end find_number_l84_84777


namespace gcd_polynomial_l84_84295

theorem gcd_polynomial (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^5 + 125) (n + 5) = if n % 5 = 0 then 5 else 1 :=
by
  sorry

end gcd_polynomial_l84_84295


namespace power_multiplication_l84_84266

variable (a : ℝ)

theorem power_multiplication : (-a)^3 * a^2 = -a^5 := 
sorry

end power_multiplication_l84_84266


namespace probability_same_gender_l84_84603

theorem probability_same_gender :
  let males := 3
  let females := 2
  let total := males + females
  let total_ways := Nat.choose total 2
  let male_ways := Nat.choose males 2
  let female_ways := Nat.choose females 2
  let same_gender_ways := male_ways + female_ways
  let probability := (same_gender_ways : ℚ) / total_ways
  probability = 2 / 5 :=
by
  sorry

end probability_same_gender_l84_84603


namespace number_of_rectangles_l84_84445

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 4) : 
  (nat.choose H 2) * (nat.choose V 2) = 60 := by
  rw [hH, hV]
  norm_num

end number_of_rectangles_l84_84445


namespace total_nails_to_cut_l84_84273

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l84_84273


namespace shaded_area_of_pattern_l84_84201

theorem shaded_area_of_pattern (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) : 
  d = 3 → 
  L = 24 → 
  n = 16 → 
  r = 3 / 2 → 
  (A = 18 * Real.pi) :=
by
  intro hd
  intro hL
  intro hn
  intro hr
  sorry

end shaded_area_of_pattern_l84_84201


namespace triangle_side_length_c_l84_84659

theorem triangle_side_length_c
  (a b A B C : ℝ)
  (ha : a = Real.sqrt 3)
  (hb : b = 1)
  (hA : A = 2 * B)
  (hAngleSum : A + B + C = Real.pi) :
  ∃ c : ℝ, c = 2 := 
by
  sorry

end triangle_side_length_c_l84_84659


namespace roberts_test_score_l84_84995

structure ClassState where
  num_students : ℕ
  avg_19_students : ℕ
  class_avg_20_students : ℕ

def calculate_roberts_score (s : ClassState) : ℕ :=
  let total_19_students := s.num_students * s.avg_19_students
  let total_20_students := (s.num_students + 1) * s.class_avg_20_students
  total_20_students - total_19_students

theorem roberts_test_score 
  (state : ClassState) 
  (h1 : state.num_students = 19) 
  (h2 : state.avg_19_students = 74)
  (h3 : state.class_avg_20_students = 75) : 
  calculate_roberts_score state = 94 := by
  sorry

end roberts_test_score_l84_84995


namespace probability_divisible_by_15_l84_84492

open Finset

def is_prime_digit (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (∀ d, d ∈ n.digits 10 → is_prime_digit d)

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

theorem probability_divisible_by_15 : 
  let S := {n ∈ (Ico 100 1000) | is_valid_three_digit n}
  let favorable := {n ∈ S | divisible_by_15 n}
  (favorable.card : ℚ) / S.card = 1 / 16 :=
by
  sorry

end probability_divisible_by_15_l84_84492


namespace kanul_machinery_expense_l84_84349

theorem kanul_machinery_expense :
  let Total := 93750
  let RawMaterials := 35000
  let Cash := 0.20 * Total
  let Machinery := Total - (RawMaterials + Cash)
  Machinery = 40000 := by
sorry

end kanul_machinery_expense_l84_84349


namespace number_of_days_l84_84903

variables (S Wx Wy : ℝ)

-- Given conditions
def condition1 : Prop := S = 36 * Wx
def condition2 : Prop := S = 45 * Wy

-- The lean statement to prove the number of days D = 20
theorem number_of_days (h1 : condition1 S Wx) (h2 : condition2 S Wy) : 
  S / (Wx + Wy) = 20 :=
by
  sorry

end number_of_days_l84_84903


namespace hexagon_area_l84_84983

open Complex EuclideanGeometry

-- Definitions for sides of triangle ABC
def AB := 13
def BC := 14
def CA := 15

-- Main theorem statement
theorem hexagon_area : 
  ∃ (A B C : ℂ), 
    abs (A - B) = AB ∧ 
    abs (B - C) = BC ∧ 
    abs (C - A) = CA ∧ 
    ∃ (A_5 A_6 B_5 B_6 C_5 C_6 : ℂ), 
      hexagon_constructed_correctly A B C A_5 A_6 B_5 B_6 C_5 C_6 →
      area (hexagon A_5 A_6 B_5 B_6 C_5 C_6) = 19444 := 
sorry

end hexagon_area_l84_84983


namespace man_climbing_out_of_well_l84_84232

theorem man_climbing_out_of_well (depth climb slip : ℕ) (h1 : depth = 30) (h2 : climb = 4) (h3 : slip = 3) : 
  let effective_climb_per_day := climb - slip
  let total_days := if depth % effective_climb_per_day = 0 then depth / effective_climb_per_day else depth / effective_climb_per_day + 1
  total_days = 30 :=
by
  sorry

end man_climbing_out_of_well_l84_84232


namespace scientific_notation_correct_l84_84021

theorem scientific_notation_correct :
  1200000000 = 1.2 * 10^9 := 
by
  sorry

end scientific_notation_correct_l84_84021


namespace smith_family_service_providers_combinations_l84_84133

theorem smith_family_service_providers_combinations :
  ∏ i in (finset.range 5).map (λ k, k + 21), i = 5103000 := 
by
  sorry

end smith_family_service_providers_combinations_l84_84133


namespace strawberries_per_jar_l84_84591

-- Let's define the conditions
def betty_strawberries : ℕ := 16
def matthew_strawberries : ℕ := betty_strawberries + 20
def natalie_strawberries : ℕ := matthew_strawberries / 2
def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
def jars_of_jam : ℕ := 40 / 4

-- Now we need to prove that the number of strawberries used in one jar of jam is 7.
theorem strawberries_per_jar : total_strawberries / jars_of_jam = 7 := by
  sorry

end strawberries_per_jar_l84_84591


namespace quadratic_root_and_coefficient_l84_84816

theorem quadratic_root_and_coefficient (k : ℝ) :
  (∃ x : ℝ, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ x₁ : ℝ, (5 * x₁^2 + k * x₁ - 6 = 0 ∧ x₁ ≠ 2) ∧ x₁ = -3/5 ∧ k = -7) :=
by
  sorry

end quadratic_root_and_coefficient_l84_84816


namespace largest_even_integer_sum_l84_84386

theorem largest_even_integer_sum :
  let sum_first_30_even := 2 * (List.sum (List.range 30).map(λ n, n + 1)) in
  let n := (sum_first_30_even + 20) / 5 in
  n = 190 :=
by
  sorry

end largest_even_integer_sum_l84_84386


namespace base8_arithmetic_l84_84073

def base8_to_base10 (n : Nat) : Nat :=
  sorry -- Placeholder for base 8 to base 10 conversion

def base10_to_base8 (n : Nat) : Nat :=
  sorry -- Placeholder for base 10 to base 8 conversion

theorem base8_arithmetic (n m : Nat) (h1 : base8_to_base10 45 = n) (h2 : base8_to_base10 76 = m) :
  base10_to_base8 ((n * 2) - m) = 14 :=
by
  sorry

end base8_arithmetic_l84_84073


namespace max_value_fraction_l84_84143

theorem max_value_fraction (x : ℝ) : 
  (∃ x, (x^4 / (x^8 + 4 * x^6 - 8 * x^4 + 16 * x^2 + 64)) = (1 / 24)) := 
sorry

end max_value_fraction_l84_84143


namespace original_fraction_eq_2_5_l84_84914

theorem original_fraction_eq_2_5 (a b : ℤ) (h : (a + 4) * b = a * (b + 10)) : (a / b) = (2 / 5) := by
  sorry

end original_fraction_eq_2_5_l84_84914


namespace mary_hourly_wage_l84_84195

-- Defining the conditions as given in the problem
def hours_per_day_MWF : ℕ := 9
def hours_per_day_TTh : ℕ := 5
def days_MWF : ℕ := 3
def days_TTh : ℕ := 2
def weekly_earnings : ℕ := 407

-- Total hours worked in a week by Mary
def total_hours_worked : ℕ := (days_MWF * hours_per_day_MWF) + (days_TTh * hours_per_day_TTh)

-- The hourly wage calculation
def hourly_wage : ℕ := weekly_earnings / total_hours_worked

-- The statement to prove
theorem mary_hourly_wage : hourly_wage = 11 := by
  sorry

end mary_hourly_wage_l84_84195


namespace marble_group_size_l84_84102

-- Define the conditions
def num_marbles : ℕ := 220
def future_people (x : ℕ) : ℕ := x + 2
def marbles_per_person (x : ℕ) : ℕ := num_marbles / x
def marbles_if_2_more (x : ℕ) : ℕ := num_marbles / future_people x

-- Statement of the theorem
theorem marble_group_size (x : ℕ) :
  (marbles_per_person x - 1 = marbles_if_2_more x) ↔ x = 20 :=
sorry

end marble_group_size_l84_84102


namespace total_bees_count_l84_84237

-- Definitions
def initial_bees : ℕ := 16
def additional_bees : ℕ := 7

-- Problem statement to prove
theorem total_bees_count : initial_bees + additional_bees = 23 := by
  -- The proof will be given here
  sorry

end total_bees_count_l84_84237


namespace mod_remainder_l84_84936

theorem mod_remainder (a b c : ℕ) : 
  (7 * 10 ^ 20 + 1 ^ 20) % 11 = 8 := by
  -- Lean proof will be written here
  sorry

end mod_remainder_l84_84936


namespace price_of_pants_l84_84678

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end price_of_pants_l84_84678


namespace sliding_window_sash_translation_l84_84400

def is_translation (movement : Type) : Prop := sorry

def ping_pong_ball_movement : Type := sorry
def sliding_window_sash_movement : Type := sorry
def kite_flight_movement : Type := sorry
def basketball_movement : Type := sorry

axiom ping_pong_not_translation : ¬ is_translation ping_pong_ball_movement
axiom kite_not_translation : ¬ is_translation kite_flight_movement
axiom basketball_not_translation : ¬ is_translation basketball_movement
axiom window_sash_is_translation : is_translation sliding_window_sash_movement

theorem sliding_window_sash_translation :
  is_translation sliding_window_sash_movement :=
by 
  exact window_sash_is_translation

end sliding_window_sash_translation_l84_84400


namespace travel_between_cities_maintained_l84_84743

open Finset Function

/-- Given a graph representing cities and airlines from different companies,
assure that connectivity is maintained even after canceling N-1 airlines.--/
theorem travel_between_cities_maintained
  (V : Type) [Fintype V]
  (E : set (V × V))
  (N : ℕ)
  (company_edges : Fin N → set (V × V))
  (h1 : ∀ i, is_perfect_matching (company_edges i) V) -- each company connects cities perfectly
  (h2 : connected (induce E)) -- initial graph is connected
  (h3 : E = ⋃ i, company_edges i) -- E is union of edges of all companies
  (h4 : ∃ (S : Finset (Fin N)), S.card = N - 1) -- select N-1 edges to cancel
  (h5 : let T := E ∖ ⋃ i ∈ S, company_edges i in ¬connected (induce T)) -- assume resulting graph disconnected
  : false :=
by sorry

end travel_between_cities_maintained_l84_84743


namespace reporter_earnings_per_hour_l84_84417

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end reporter_earnings_per_hour_l84_84417


namespace player_A_wins_4_points_game_game_ends_after_5_points_l84_84692

def prob_A_winning_when_serving : ℚ := 2 / 3
def prob_A_winning_when_B_serving : ℚ := 1 / 4
def prob_A_winning_in_4_points : ℚ := 1 / 12
def prob_game_ending_after_5_points : ℚ := 19 / 216

theorem player_A_wins_4_points_game :
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) = prob_A_winning_in_4_points := 
  sorry

theorem game_ends_after_5_points : 
  ((1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (prob_A_winning_when_serving)) + 
  ((prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (1 - prob_A_winning_when_serving)) = 
  prob_game_ending_after_5_points :=
  sorry

end player_A_wins_4_points_game_game_ends_after_5_points_l84_84692


namespace find_x_l84_84484

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l84_84484


namespace find_x_l84_84489

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l84_84489


namespace students_not_in_biology_l84_84572

theorem students_not_in_biology (total_students : ℕ) (percentage_in_biology : ℚ) 
  (h1 : total_students = 880) (h2 : percentage_in_biology = 27.5 / 100) : 
  total_students - (total_students * percentage_in_biology) = 638 := 
by
  sorry

end students_not_in_biology_l84_84572


namespace number_of_prime_divisors_of_50_fac_l84_84647

-- Define the finite set of prime numbers up to 50
def primes_up_to_50 : finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Main theorem statement
theorem number_of_prime_divisors_of_50_fac :
  (primes_up_to_50.filter prime).card = 15 := 
sorry

end number_of_prime_divisors_of_50_fac_l84_84647


namespace consecutive_integers_product_l84_84722

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l84_84722


namespace shaded_area_of_pattern_l84_84200

theorem shaded_area_of_pattern (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) : 
  d = 3 → 
  L = 24 → 
  n = 16 → 
  r = 3 / 2 → 
  (A = 18 * Real.pi) :=
by
  intro hd
  intro hL
  intro hn
  intro hr
  sorry

end shaded_area_of_pattern_l84_84200


namespace bake_sale_comparison_l84_84263

theorem bake_sale_comparison :
  let tamara_small_brownies := 4 * 2
  let tamara_large_brownies := 12 * 3
  let tamara_cookies := 36 * 1.5
  let tamara_total := tamara_small_brownies + tamara_large_brownies + tamara_cookies

  let sarah_muffins := 24 * 1.75
  let sarah_choco_cupcakes := 7 * 2.5
  let sarah_vanilla_cupcakes := 8 * 2
  let sarah_strawberry_cupcakes := 15 * 2.75
  let sarah_total := sarah_muffins + sarah_choco_cupcakes + sarah_vanilla_cupcakes + sarah_strawberry_cupcakes

  sarah_total - tamara_total = 18.75 := by
  sorry

end bake_sale_comparison_l84_84263


namespace same_solution_sets_l84_84496

theorem same_solution_sets (a : ℝ) :
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := 
by
  sorry

end same_solution_sets_l84_84496


namespace sale_price_with_50_percent_profit_l84_84214

theorem sale_price_with_50_percent_profit (CP SP₁ SP₃ : ℝ) 
(h1 : SP₁ - CP = CP - 448) 
(h2 : SP₃ = 1.5 * CP) 
(h3 : SP₃ = 1020) : 
SP₃ = 1020 := 
by 
  sorry

end sale_price_with_50_percent_profit_l84_84214


namespace find_n_for_geom_sum_l84_84216

-- Define the first term and the common ratio
def first_term := 1
def common_ratio := 1 / 2

-- Define the sum function of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℚ := first_term * (1 - (common_ratio)^n) / (1 - common_ratio)

-- Define the target sum
def target_sum := 31 / 16

-- State the theorem to prove
theorem find_n_for_geom_sum : ∃ n : ℕ, geom_sum n = target_sum := 
    by
    sorry

end find_n_for_geom_sum_l84_84216


namespace stamp_arrangements_equals_76_l84_84436

-- Define the conditions of the problem
def stamps_available : List (ℕ × ℕ) := 
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), 
   (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), 
   (17, 17), (18, 18), (19, 19)]

-- Define a function to compute the number of different arrangements
noncomputable def count_stamp_arrangements : ℕ :=
  -- This is a placeholder for the actual implementation
  sorry

-- State the theorem to be proven
theorem stamp_arrangements_equals_76 : count_stamp_arrangements = 76 :=
sorry

end stamp_arrangements_equals_76_l84_84436


namespace find_x_l84_84488

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l84_84488


namespace necessary_condition_not_sufficient_condition_l84_84866

variable (a b : ℝ)
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0
def proposition_p (a : ℝ) : Prop := a = 0

theorem necessary_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : isPureImaginary z → proposition_p a := sorry

theorem not_sufficient_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : proposition_p a → ¬isPureImaginary z := sorry

end necessary_condition_not_sufficient_condition_l84_84866


namespace jake_snakes_l84_84024

theorem jake_snakes (S : ℕ) 
  (h1 : 2 * S + 1 = 6) 
  (h2 : 2250 = 5 * 250 + 1000) :
  S = 3 := 
by
  sorry

end jake_snakes_l84_84024


namespace interest_rate_l84_84379

theorem interest_rate (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) : 
  P = 8000.000000000171 → t = 2 → d = 20 →
  (P * (1 + r/100)^2 - P - (P * r * t / 100) = d) → r = 5 :=
by
  intros hP ht hd heq
  sorry

end interest_rate_l84_84379


namespace train_a_speed_54_l84_84875

noncomputable def speed_of_train_A (length_A length_B : ℕ) (speed_B : ℕ) (time_to_cross : ℕ) : ℕ :=
  let total_distance := length_A + length_B
  let relative_speed := total_distance / time_to_cross
  let relative_speed_km_per_hr := relative_speed * 36 / 10
  let speed_A := relative_speed_km_per_hr - speed_B
  speed_A

theorem train_a_speed_54 
  (length_A length_B : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (h_length_A : length_A = 150)
  (h_length_B : length_B = 150)
  (h_speed_B : speed_B = 36)
  (h_time_to_cross : time_to_cross = 12) :
  speed_of_train_A length_A length_B speed_B time_to_cross = 54 := by
  sorry

end train_a_speed_54_l84_84875


namespace proposition_2_proposition_4_l84_84989

variable {m n : Line}
variable {α β : Plane}

-- Define predicates for perpendicularity, parallelism, and containment
axiom line_parallel_plane (n : Line) (α : Plane) : Prop
axiom line_perp_plane (n : Line) (α : Plane) : Prop
axiom plane_perp_plane (α β : Plane) : Prop
axiom line_in_plane (m : Line) (β : Plane) : Prop

-- State the correct propositions
theorem proposition_2 (m n : Line) (α β : Plane)
  (h1 : line_perp_plane m n)
  (h2 : line_perp_plane n α)
  (h3 : line_perp_plane m β) :
  plane_perp_plane α β := sorry

theorem proposition_4 (n : Line) (α β : Plane)
  (h1 : line_perp_plane n β)
  (h2 : plane_perp_plane α β) :
  line_parallel_plane n α ∨ line_in_plane n α := sorry

end proposition_2_proposition_4_l84_84989


namespace factor_is_given_sum_l84_84972

theorem factor_is_given_sum (P Q : ℤ)
  (h1 : ∀ x : ℝ, (x^2 + 3 * x + 7) * (x^2 + (-3) * x + 7) = x^4 + P * x^2 + Q) :
  P + Q = 54 := 
sorry

end factor_is_given_sum_l84_84972


namespace nonagon_diagonals_l84_84632

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nonagon_diagonals : number_of_diagonals 9 = 27 := 
by
  sorry

end nonagon_diagonals_l84_84632


namespace exponent_fraction_law_l84_84285

theorem exponent_fraction_law :
  (2 ^ 2017 + 2 ^ 2013) / (2 ^ 2017 - 2 ^ 2013) = 17 / 15 :=
  sorry

end exponent_fraction_law_l84_84285


namespace solve_for_x_l84_84204

theorem solve_for_x : ∃ x k l : ℕ, (3 * 22 = k) ∧ (66 + l = 90) ∧ (160 * 3 / 4 = x - l) → x = 144 :=
by
  sorry

end solve_for_x_l84_84204


namespace expression_eval_l84_84287

noncomputable def a : ℕ := 2001
noncomputable def b : ℕ := 2003

theorem expression_eval : 
  b^3 - a * b^2 - a^2 * b + a^3 = 8 :=
by sorry

end expression_eval_l84_84287


namespace greatest_number_of_bouquets_l84_84688

def sara_red_flowers : ℕ := 16
def sara_yellow_flowers : ℕ := 24

theorem greatest_number_of_bouquets : Nat.gcd sara_red_flowers sara_yellow_flowers = 8 := by
  rfl

end greatest_number_of_bouquets_l84_84688


namespace tan_11_pi_over_4_l84_84927

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l84_84927


namespace Lexie_age_proof_l84_84116

variables (L B S : ℕ)

def condition1 : Prop := L = B + 6
def condition2 : Prop := S = 2 * L
def condition3 : Prop := S - B = 14

theorem Lexie_age_proof (h1 : condition1 L B) (h2 : condition2 S L) (h3 : condition3 S B) : L = 8 :=
by
  sorry

end Lexie_age_proof_l84_84116


namespace evaluate_expression_l84_84920

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

end evaluate_expression_l84_84920


namespace distance_from_A_to_directrix_of_parabola_l84_84315

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l84_84315


namespace greatest_k_value_l84_84220

-- Define a type for triangle and medians intersecting at centroid
structure Triangle :=
(medianA : ℝ)
(medianB : ℝ)
(medianC : ℝ)
(angleA : ℝ)
(angleB : ℝ)
(angleC : ℝ)
(centroid : ℝ)

-- Define a function to determine if the internal angles formed by medians 
-- are greater than 30 degrees
def angle_greater_than_30 (θ : ℝ) : Prop :=
  θ > 30

-- A proof statement that given a triangle and its medians dividing an angle
-- into six angles, the greatest possible number of these angles greater than 30° is 3.
theorem greatest_k_value (T : Triangle) : ∃ k : ℕ, k = 3 ∧ 
  (∀ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ, 
    (angle_greater_than_30 θ₁ ∨ angle_greater_than_30 θ₂ ∨ angle_greater_than_30 θ₃ ∨ 
     angle_greater_than_30 θ₄ ∨ angle_greater_than_30 θ₅ ∨ angle_greater_than_30 θ₆) → 
    k = 3) := 
sorry

end greatest_k_value_l84_84220


namespace stationery_problem_l84_84124

variables (S E : ℕ)

theorem stationery_problem
  (h1 : S - E = 30)
  (h2 : 4 * E = S) :
  S = 40 :=
by
  sorry

end stationery_problem_l84_84124


namespace completing_square_result_l84_84092

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l84_84092


namespace paula_bought_two_shirts_l84_84685

-- Define the conditions
def total_money : Int := 109
def shirt_cost : Int := 11
def pants_cost : Int := 13
def remaining_money : Int := 74

-- Calculate the expenditure on shirts and pants
def expenditure : Int := total_money - remaining_money

-- Define the number of shirts bought
def number_of_shirts (S : Int) : Prop := expenditure = shirt_cost * S + pants_cost

-- The theorem stating that Paula bought 2 shirts
theorem paula_bought_two_shirts : number_of_shirts 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end paula_bought_two_shirts_l84_84685


namespace pumpkins_total_weight_l84_84681

-- Define the weights of the pumpkins as given in the conditions
def first_pumpkin_weight : ℝ := 4
def second_pumpkin_weight : ℝ := 8.7

-- Prove that the total weight of the two pumpkins is 12.7 pounds
theorem pumpkins_total_weight : first_pumpkin_weight + second_pumpkin_weight = 12.7 := by
  sorry

end pumpkins_total_weight_l84_84681


namespace johann_mail_l84_84842

def pieces_of_mail_total : ℕ := 180
def pieces_of_mail_friends : ℕ := 41
def friends : ℕ := 2
def pieces_of_mail_johann : ℕ := pieces_of_mail_total - (pieces_of_mail_friends * friends)

theorem johann_mail : pieces_of_mail_johann = 98 := by
  sorry

end johann_mail_l84_84842


namespace smallest_possible_Y_l84_84352

def digits (n : ℕ) : List ℕ := -- hypothetical function to get the digits of a number
  sorry

def is_divisible (n d : ℕ) : Prop := d ∣ n

theorem smallest_possible_Y :
  ∃ (U : ℕ), (∀ d ∈ digits U, d = 0 ∨ d = 1) ∧ is_divisible U 18 ∧ U / 18 = 61728395 :=
by
  sorry

end smallest_possible_Y_l84_84352


namespace cassie_nail_cutting_l84_84272

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l84_84272


namespace annika_current_age_l84_84343

-- Define the conditions
def hans_age_current : ℕ := 8
def hans_age_in_4_years : ℕ := hans_age_current + 4
def annika_age_in_4_years : ℕ := 3 * hans_age_in_4_years

-- lean statement to prove Annika's current age
theorem annika_current_age (A : ℕ) (hyp : A + 4 = annika_age_in_4_years) : A = 32 :=
by
  -- Skipping the proof
  sorry

end annika_current_age_l84_84343


namespace distinct_three_digit_numbers_count_l84_84958

theorem distinct_three_digit_numbers_count : 
  ∃! n : ℕ, n = 5 * 4 * 3 :=
by
  use 60
  sorry

end distinct_three_digit_numbers_count_l84_84958


namespace high_sulfur_oil_samples_l84_84345

/-- The number of high-sulfur oil samples in a container with the given conditions. -/
theorem high_sulfur_oil_samples (total_samples : ℕ) 
    (heavy_oil_freq : ℚ) (light_low_sulfur_freq : ℚ)
    (no_heavy_low_sulfur: true) (almost_full : total_samples = 198)
    (heavy_oil_freq_value : heavy_oil_freq = 1 / 9)
    (light_low_sulfur_freq_value : light_low_sulfur_freq = 11 / 18) :
    (22 + 68) = 90 := 
by
  sorry

end high_sulfur_oil_samples_l84_84345


namespace radian_measure_of_sector_l84_84157

theorem radian_measure_of_sector
  (perimeter : ℝ) (area : ℝ) (radian_measure : ℝ)
  (h1 : perimeter = 8)
  (h2 : area = 4) :
  radian_measure = 2 :=
sorry

end radian_measure_of_sector_l84_84157


namespace probability_endpoints_of_edge_l84_84748

noncomputable def num_vertices : ℕ := 12
noncomputable def edges_per_vertex : ℕ := 3

theorem probability_endpoints_of_edge :
  let total_ways := Nat.choose num_vertices 2,
      total_edges := (num_vertices * edges_per_vertex) / 2,
      probability := total_edges / total_ways in
  probability = 3 / 11 := by
  sorry

end probability_endpoints_of_edge_l84_84748


namespace files_remaining_l84_84751

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h_music : music_files = 13) 
  (h_video : video_files = 30) 
  (h_deleted : deleted_files = 10) : 
  (music_files + video_files - deleted_files) = 33 :=
by
  sorry

end files_remaining_l84_84751


namespace middle_part_proportional_l84_84768

theorem middle_part_proportional (x : ℚ) (s : ℚ) (h : s = 120) 
    (proportional : (2 * x) + (1/2 * x) + (1/4 * x) = s) : 
    (1/2 * x) = 240/11 := 
by
  sorry

end middle_part_proportional_l84_84768


namespace intersect_sum_l84_84817

noncomputable def intersect_points (y : ℝ → ℝ) (k : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  y x1 = k ∧ y x2 = k ∧ y x3 = k ∧ x1 < x2 ∧ x2 < x3

theorem intersect_sum (x1 x2 x3 : ℝ) (h1 : x1 + x2 = π / 3) (h2 : x2 + x3 = 4 * π / 3) :
  x1 + 2 * x2 + x3 = 5 * π / 3 := by
  sorry

end intersect_sum_l84_84817


namespace linear_equation_m_not_eq_4_l84_84696

theorem linear_equation_m_not_eq_4 (m x y : ℝ) :
  (m * x + 3 * y = 4 * x - 1) → m ≠ 4 :=
by
  sorry

end linear_equation_m_not_eq_4_l84_84696


namespace sum_of_consecutive_integers_product_336_l84_84714

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l84_84714


namespace probability_heads_exactly_2_times_three_tosses_uniform_coin_l84_84550

noncomputable def probability_heads_exactly_2_times (n k : ℕ) (p : ℚ) : ℚ :=
(n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_heads_exactly_2_times_three_tosses_uniform_coin :
  probability_heads_exactly_2_times 3 2 (1/2) = 3 / 8 :=
by
  sorry

end probability_heads_exactly_2_times_three_tosses_uniform_coin_l84_84550


namespace meeting_point_l84_84855

theorem meeting_point :
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  (Paul_start.1 + Lisa_start.1) / 2 = -2 ∧ (Paul_start.2 + Lisa_start.2) / 2 = 3 :=
by
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  have x_coord : (Paul_start.1 + Lisa_start.1) / 2 = -2 := sorry
  have y_coord : (Paul_start.2 + Lisa_start.2) / 2 = 3 := sorry
  exact ⟨x_coord, y_coord⟩

end meeting_point_l84_84855


namespace car_win_probability_l84_84180

theorem car_win_probability :
  let P_A := (1 : ℚ) / 8
  let P_B := (1 : ℚ) / 12
  let P_C := (1 : ℚ) / 15
  let P_D := (1 : ℚ) / 18
  let P_E := (1 : ℚ) / 20
  P_A + P_B + P_C + P_D + P_E = 137 / 360 :=
by
  let P_A := (1 : ℚ) / 8
  let P_B := (1 : ℚ) / 12
  let P_C := (1 : ℚ) / 15
  let P_D := (1 : ℚ) / 18
  let P_E := (1 : ℚ) / 20
  calc
    P_A + P_B + P_C + P_D + P_E
    = 1 / 8 + 1 / 12 + 1 / 15 + 1 / 18 + 1 / 20 : by refl
    ... = 45 / 360 + 30 / 360 + 24 / 360 + 20 / 360 + 18 / 360 : by sorry
    ... = 137 / 360 : by sorry

end car_win_probability_l84_84180


namespace binomial_square_solution_l84_84606

variable (t u b : ℝ)

theorem binomial_square_solution (h1 : 2 * t * u = 12) (h2 : u^2 = 9) : b = t^2 → b = 4 :=
by
  sorry

end binomial_square_solution_l84_84606


namespace trajectory_of_point_A_l84_84809

theorem trajectory_of_point_A (m : ℝ) (A B C : ℝ × ℝ) (hBC : B = (-1, 0) ∧ C = (1, 0)) (hBC_dist : dist B C = 2)
  (hRatio : dist A B / dist A C = m) :
  (m = 1 → ∀ x y : ℝ, A = (x, y) → x = 0) ∧
  (m = 0 → ∀ x y : ℝ, A = (x, y) → x^2 + y^2 - 2 * x + 1 = 0) ∧
  (m ≠ 0 ∧ m ≠ 1 → ∀ x y : ℝ, A = (x, y) → (x + (1 + m^2) / (1 - m^2))^2 + y^2 = (2 * m / (1 - m^2))^2) := 
sorry

end trajectory_of_point_A_l84_84809


namespace sides_of_triangle_inequality_l84_84147

theorem sides_of_triangle_inequality {a b c : ℝ} (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
by
  sorry

end sides_of_triangle_inequality_l84_84147


namespace intersection_A_B_l84_84174

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l84_84174


namespace prod_eq_diff_squares_l84_84907

variable (a b : ℝ)

theorem prod_eq_diff_squares :
  ( (1 / 4 * a + b) * (b - 1 / 4 * a) = b^2 - (1 / 16 * a^2) ) :=
by
  sorry

end prod_eq_diff_squares_l84_84907


namespace redistribute_oil_l84_84742

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l84_84742


namespace cost_of_grapes_and_watermelon_l84_84851

theorem cost_of_grapes_and_watermelon (p g w f : ℝ)
  (h1 : p + g + w + f = 30)
  (h2 : f = 2 * p)
  (h3 : p - g = w) :
  g + w = 7.5 :=
by
  sorry

end cost_of_grapes_and_watermelon_l84_84851


namespace inequality_proof_l84_84670

open Real

theorem inequality_proof (x y : ℝ) (hx : x > 1/2) (hy : y > 1) : 
  (4 * x^2) / (y - 1) + (y^2) / (2 * x - 1) ≥ 8 := 
by
  sorry

end inequality_proof_l84_84670


namespace volume_expression_correct_l84_84590

variable (x : ℝ)

def volume (x : ℝ) := x * (30 - 2 * x) * (20 - 2 * x)

theorem volume_expression_correct (h : x < 10) :
  volume x = 4 * x^3 - 100 * x^2 + 600 * x :=
by sorry

end volume_expression_correct_l84_84590


namespace price_per_glass_first_day_l84_84852

theorem price_per_glass_first_day (O P2 P1: ℝ) (H1 : O > 0) (H2 : P2 = 0.2) (H3 : 2 * O * P1 = 3 * O * P2) : P1 = 0.3 :=
by
  sorry

end price_per_glass_first_day_l84_84852


namespace original_polygon_sides_l84_84897

theorem original_polygon_sides {n : ℕ} 
    (hn : (n - 2) * 180 = 1080) : n = 7 ∨ n = 8 ∨ n = 9 :=
sorry

end original_polygon_sides_l84_84897


namespace sum_of_consecutive_integers_product_336_l84_84712

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l84_84712


namespace completing_the_square_result_l84_84084

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l84_84084


namespace leak_empties_tank_in_10_hours_l84_84884

theorem leak_empties_tank_in_10_hours :
  (∀ (A L : ℝ), (A = 1/5) → (A - L = 1/10) → (1 / L = 10)) 
  := by
  intros A L hA hAL
  sorry

end leak_empties_tank_in_10_hours_l84_84884


namespace sin_double_angle_ratio_l84_84968

theorem sin_double_angle_ratio (α : ℝ) (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 :=
by 
  sorry

end sin_double_angle_ratio_l84_84968


namespace completing_the_square_result_l84_84085

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l84_84085


namespace max_tied_teams_l84_84340

theorem max_tied_teams (n : ℕ) (h_n : n = 8) (tournament : Fin n → Fin n → Prop)
  (h_symmetric : ∀ i j, tournament i j ↔ tournament j i)
  (h_antisymmetric : ∀ i j, tournament i j → ¬ tournament j i)
  (h_total : ∀ i j, i ≠ j → tournament i j ∨ tournament j i) :
  ∃ (k : ℕ), k = 7 ∧ ∀ (wins : Fin n → ℕ), 
  (∀ i, wins i = 4 → ∃! j, i ≠ j ∧ tournament i j) → True :=
by sorry

end max_tied_teams_l84_84340


namespace value_of_expression_l84_84395

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end value_of_expression_l84_84395


namespace sum_two_smallest_prime_factors_l84_84878

theorem sum_two_smallest_prime_factors (n : ℕ) (h : n = 462) : 
  (2 + 3) = 5 := 
by {
  sorry
}

end sum_two_smallest_prime_factors_l84_84878


namespace polynomial_solution_l84_84934

noncomputable def Q (x : ℝ) : ℝ := x^2 + 2*x

theorem polynomial_solution {Q : ℝ → ℝ} 
  (h : ∀ x, Q(Q(x)) = (x^2 + 2*x + 2) * Q(x)) :
  ∃ a b c : ℝ, a ≠ 0 ∧ Q x = a*x^2 + b*x + c ∧ Q x = x^2 + 2*x :=
by 
  existsi [1, 2, 0]
  simp
  sorry

end polynomial_solution_l84_84934


namespace tan_sin_sum_eq_sqrt3_l84_84731

theorem tan_sin_sum_eq_sqrt3 (tan20 sin20 : ℝ) (h1 : tan 20 = sin 20 / cos 20) (h2 : sin20 = sin 20) :
  tan20 + 4 * sin20 = sqrt 3 := by
  sorry

end tan_sin_sum_eq_sqrt3_l84_84731


namespace cleaning_time_l84_84185

def lara_rate := 1 / 4
def chris_rate := 1 / 6
def combined_rate := lara_rate + chris_rate

theorem cleaning_time (t : ℝ) : 
  (combined_rate * (t - 2) = 1) ↔ (t = 22 / 5) :=
by
  sorry

end cleaning_time_l84_84185


namespace consecutive_integers_sum_l84_84708

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l84_84708


namespace sum_of_three_consecutive_integers_product_336_l84_84719

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l84_84719


namespace gain_percent_of_cost_selling_relation_l84_84104

theorem gain_percent_of_cost_selling_relation (C S : ℕ) (h : 50 * C = 45 * S) : 
  (S > C) ∧ ((S - C) / C * 100 = 100 / 9) :=
by
  sorry

end gain_percent_of_cost_selling_relation_l84_84104


namespace geom_sequence_ratio_l84_84504

-- Definitions and assumptions for the problem
noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_ratio (a : ℕ → ℝ) (r : ℝ) 
  (h_geom: geom_seq a)
  (h_r: 0 < r ∧ r < 1)
  (h_seq: ∀ n : ℕ, a (n + 1) = a n * r)
  (ha1: a 7 * a 14 = 6)
  (ha2: a 4 + a 17 = 5) :
  (a 5 / a 18) = (3 / 2) :=
sorry

end geom_sequence_ratio_l84_84504


namespace approx_sum_l84_84113

-- Definitions of the costs
def cost_bicycle : ℕ := 389
def cost_fan : ℕ := 189

-- Definition of the approximations
def approx_bicycle : ℕ := 400
def approx_fan : ℕ := 200

-- The statement to prove
theorem approx_sum (h₁ : cost_bicycle = 389) (h₂ : cost_fan = 189) : 
  approx_bicycle + approx_fan = 600 := 
by 
  sorry

end approx_sum_l84_84113


namespace selling_price_750_max_daily_profit_l84_84502

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 10) * (-10 * x + 300)

theorem selling_price_750 (x : ℝ) : profit x = 750 ↔ (x = 15 ∨ x = 25) :=
by sorry

theorem max_daily_profit : (∀ x : ℝ, profit x ≤ 1000) ∧ (profit 20 = 1000) :=
by sorry

end selling_price_750_max_daily_profit_l84_84502


namespace halfway_between_l84_84290

-- Definitions based on given conditions
def a : ℚ := 1 / 7
def b : ℚ := 1 / 9

-- Theorem that needs to be proved
theorem halfway_between (h : True) : (a + b) / 2 = 8 / 63 := by
  sorry

end halfway_between_l84_84290


namespace quadratic_real_roots_range_l84_84653

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end quadratic_real_roots_range_l84_84653


namespace chord_line_equation_l84_84465

theorem chord_line_equation 
  (x y : ℝ)
  (ellipse_eq : x^2 / 4 + y^2 / 3 = 1)
  (midpoint_condition : ∃ x1 y1 x2 y2 : ℝ, (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1
   ∧ (x1^2 / 4 + y1^2 / 3 = 1) ∧ (x2^2 / 4 + y2^2 / 3 = 1))
  : 3 * x - 4 * y + 7 = 0 :=
sorry

end chord_line_equation_l84_84465


namespace abs_ab_cd_leq_one_fourth_l84_84990

theorem abs_ab_cd_leq_one_fourth (a b c d : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  |a * b - c * d| ≤ 1 / 4 :=
sorry

end abs_ab_cd_leq_one_fourth_l84_84990


namespace redistribute_oil_l84_84740

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l84_84740


namespace percentage_fewer_than_50000_l84_84538

def percentage_lt_20000 : ℝ := 35
def percentage_20000_to_49999 : ℝ := 45
def percentage_lt_50000 : ℝ := 80

theorem percentage_fewer_than_50000 :
  percentage_lt_20000 + percentage_20000_to_49999 = percentage_lt_50000 := 
by
  sorry

end percentage_fewer_than_50000_l84_84538


namespace count_four_digit_integers_with_3_and_6_l84_84634

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l84_84634


namespace treasure_chest_total_value_l84_84904

def base7_to_base10 (n : Nat) : Nat :=
  let rec convert (n acc base : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * base) (base * 7)
  convert n 0 1

theorem treasure_chest_total_value :
  base7_to_base10 5346 + base7_to_base10 6521 + base7_to_base10 320 = 4305 :=
by
  sorry

end treasure_chest_total_value_l84_84904


namespace num_squares_sharing_two_vertices_l84_84847

-- Define the isosceles triangle and condition AB = AC
structure IsoscelesTriangle (A B C : Type) :=
  (AB AC : ℝ)
  (h_iso : AB = AC)

-- Define the problem statement in Lean
theorem num_squares_sharing_two_vertices 
  (A B C : Type) 
  (iso_tri : IsoscelesTriangle A B C) 
  (planeABC : ∀ P Q R : Type, P ≠ Q ∧ Q ≠ R ∧ P ≠ R) :
  ∃ n : ℕ, n = 4 := sorry

end num_squares_sharing_two_vertices_l84_84847


namespace necessary_but_not_sufficient_for_gt_zero_l84_84110

theorem necessary_but_not_sufficient_for_gt_zero (x : ℝ) : 
  x ≠ 0 → (¬ (x ≤ 0)) := by 
  sorry

end necessary_but_not_sufficient_for_gt_zero_l84_84110


namespace dream_clock_time_condition_l84_84499

theorem dream_clock_time_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 1)
  (h3 : (120 + 0.5 * 60 * x) = (240 - 6 * 60 * x)) :
  (4 + x) = 4 + 36 + 12 / 13 := by sorry

end dream_clock_time_condition_l84_84499


namespace ten_men_ten_boys_work_time_l84_84888

theorem ten_men_ten_boys_work_time :
  (∀ (total_work : ℝ) (man_work_rate boy_work_rate : ℝ),
    15 * 10 * man_work_rate = total_work ∧
    20 * 15 * boy_work_rate = total_work →
    (10 * man_work_rate + 10 * boy_work_rate) * 10 = total_work) :=
by
  sorry

end ten_men_ten_boys_work_time_l84_84888


namespace probability_of_four_digit_number_divisible_by_3_l84_84146

def digits : List ℕ := [0, 1, 2, 3, 4, 5]

def count_valid_four_digit_numbers : Int :=
  let all_digits := digits
  let total_four_digit_numbers := 180
  let valid_four_digit_numbers := 96
  total_four_digit_numbers

def probability_divisible_by_3 : ℚ :=
  (96 : ℚ) / (180 : ℚ)

theorem probability_of_four_digit_number_divisible_by_3 :
  probability_divisible_by_3 = 8 / 15 :=
by
  sorry

end probability_of_four_digit_number_divisible_by_3_l84_84146


namespace passing_marks_l84_84578

theorem passing_marks :
  ∃ P T : ℝ, (0.2 * T = P - 40) ∧ (0.3 * T = P + 20) ∧ P = 160 :=
by
  sorry

end passing_marks_l84_84578


namespace grid_satisfies_conditions_l84_84017

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l84_84017


namespace largest_B_181_l84_84921

noncomputable def binom (n k : ℕ) : ℚ := Nat.choose n k
def B (n k : ℕ) (p : ℚ) := binom n k * p^k

theorem largest_B_181 : ∃ k, B 2000 181 (1 / 10) = arg_max k (B 2000 k (1 / 10)) where
  arg_max (k : ℕ) (f : ℕ → ℚ) := k ≤ 2000 ∧ ∀ j, j ≤ 2000 → f j ≤ f k := sorry

end largest_B_181_l84_84921


namespace expression_divisible_by_1968_l84_84858

theorem expression_divisible_by_1968 (n : ℕ) : 
  ( -1 ^ (2 * n) +  9 ^ (4 * n) - 6 ^ (8 * n) + 8 ^ (16 * n) ) % 1968 = 0 :=
by
  sorry

end expression_divisible_by_1968_l84_84858


namespace find_pq_l84_84971

theorem find_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hline : ∀ x y : ℝ, px + qy = 24) 
  (harea : (1 / 2) * (24 / p) * (24 / q) = 48) : p * q = 12 :=
by
  sorry

end find_pq_l84_84971


namespace races_condition_possible_l84_84770

variables {τ : Type} [LinearOrder τ]

def beats (x y : τ) (races : list (list τ)) :=
  (list.countp (λ race, race.index_of x < race.index_of y) races) > (races.length / 2)

def condition (A B C : τ) (races : list (list τ)) := 
  beats A B races ∧ beats B C races ∧ beats C A races

theorem races_condition_possible (A B C : τ) :
  ∃ races : list (list τ), condition A B C races :=
begin
  -- proof would go here
  sorry
end

end races_condition_possible_l84_84770


namespace num_four_digit_pos_integers_l84_84638

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l84_84638


namespace LCM_of_18_and_27_l84_84080

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end LCM_of_18_and_27_l84_84080


namespace trajectory_of_P_l84_84815

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x = 15

structure Point :=
  (x : ℝ)
  (y : ℝ)

def moving_point_on_circle (M : Point) : Prop :=
  equation_of_circle M.x M.y

def N_fixed_point : Point :=
  { x := 1, y := 0 }

theorem trajectory_of_P (P : Point) : 
  (∃ M : Point, moving_point_on_circle M ∧ 
    (perpendicular_bisector_intersects P M N_fixed_point ∧ 
     intersects_CM P M)) →
  (P.x^2 / 4 + P.y^2 / 3 = 1) :=
sorry

-- Auxiliary Definitions: Assume these definitions exist in the library or can be defined.
def perpendicular_bisector_intersects (P M N : Point) : Prop := sorry
def intersects_CM (P M : Point) : Prop := sorry

end trajectory_of_P_l84_84815


namespace solve_for_a_l84_84330

theorem solve_for_a (a : ℝ) (h : (a + 3)^(a + 1) = 1) : a = -2 ∨ a = -1 :=
by {
  -- proof here
  sorry
}

end solve_for_a_l84_84330


namespace x5_y5_z5_value_is_83_l84_84846

noncomputable def find_x5_y5_z5_value (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧ 
  (x^3 + y^3 + z^3 = 15) ∧
  (x^4 + y^4 + z^4 = 35) ∧
  (x^2 + y^2 + z^2 < 10) →
  x^5 + y^5 + z^5 = 83

theorem x5_y5_z5_value_is_83 (x y z : ℝ) :
  find_x5_y5_z5_value x y z :=
  sorry

end x5_y5_z5_value_is_83_l84_84846


namespace b_is_some_even_number_l84_84604

noncomputable def factorable_b (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    (m * p = 15 ∧ n * q = 15) ∧ 
    (b = m * q + n * p)

theorem b_is_some_even_number (b : ℤ) 
  (h : factorable_b b) : ∃ k : ℤ, b = 2 * k := 
by
  sorry

end b_is_some_even_number_l84_84604


namespace factorize_expression_l84_84605

theorem factorize_expression (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l84_84605


namespace find_x_minus_y_l84_84006

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3 / 2 :=
by
  sorry

end find_x_minus_y_l84_84006


namespace solve_equation_floor_l84_84206

theorem solve_equation_floor (x : ℚ) :
  (⌊(5 + 6 * x) / 8⌋ : ℚ) = (15 * x - 7) / 5 ↔ x = 7 / 15 ∨ x = 4 / 5 :=
by
  sorry

end solve_equation_floor_l84_84206


namespace problem_statement_l84_84887

theorem problem_statement :
  (2 * 3 * 4) * (1/2 + 1/3 + 1/4) = 26 := by
  sorry

end problem_statement_l84_84887


namespace distance_between_lines_l84_84695

noncomputable def distance_between_parallel_lines
  (a b m n : ℝ) : ℝ :=
  |m - n| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines
  (a b m n : ℝ) :
  distance_between_parallel_lines a b m n = 
  |m - n| / Real.sqrt (a^2 + b^2) :=
by
  sorry

end distance_between_lines_l84_84695


namespace total_distance_traveled_l84_84521

/-- The total distance traveled by Mr. and Mrs. Hugo over three days. -/
theorem total_distance_traveled :
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  first_day + second_day + third_day = 525 := by
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  have h1 : first_day + second_day + third_day = 525 := by
    sorry
  exact h1

end total_distance_traveled_l84_84521


namespace all_defective_is_impossible_l84_84252

def total_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem all_defective_is_impossible :
  ∀ (products : Finset ℕ),
  products.card = selected_products →
  ∀ (product_ids : Finset ℕ),
  product_ids.card = defective_products →
  products ⊆ product_ids → False :=
by
  sorry

end all_defective_is_impossible_l84_84252


namespace distance_to_directrix_l84_84310

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l84_84310


namespace triangle_inequality_l84_84203

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end triangle_inequality_l84_84203


namespace tan_of_11pi_over_4_is_neg1_l84_84928

noncomputable def tan_periodic : Real := 2 * Real.pi

theorem tan_of_11pi_over_4_is_neg1 :
  Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Angle normalization using periodicity of tangent function
  have h1 : Real.tan (11 * Real.pi / 4) = Real.tan (11 * Real.pi / 4 - 2 * Real.pi) := 
    by rw [Real.tan_periodic]
  -- Further normalization
  have h2 : 11 * Real.pi / 4 - 2 * Real.pi = 3 * Real.pi / 4 := sorry
  -- Evaluate tangent at the simplified angle
  have h3 : Real.tan (3 * Real.pi / 4) = -Real.tan (Real.pi / 4) := sorry
  -- Known value of tangent at common angle
  have h4 : Real.tan (Real.pi / 4) = 1 := by simpl tan
  rw [h2, h3, h4]
  norm_num

end tan_of_11pi_over_4_is_neg1_l84_84928


namespace range_of_f_intersection_points_l84_84587

def digit_of_pi_at (n : ℕ) : ℕ :=
  -- Let's assume we have a function that accurately computes the digit of Pi at position n
  sorry

def f (n : ℕ) : ℕ := digit_of_pi_at n

theorem range_of_f : set.range f = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
sorry

theorem intersection_points : {n : ℕ | f n = n^2}.finite ∧
  ({n : ℕ | f n = n^2}.to_finset.card = 2) :=
sorry

end range_of_f_intersection_points_l84_84587


namespace school_basketballs_l84_84215

theorem school_basketballs (n_classes n_basketballs_per_class total_basketballs : ℕ)
  (h1 : n_classes = 7)
  (h2 : n_basketballs_per_class = 7)
  (h3 : total_basketballs = n_classes * n_basketballs_per_class) :
  total_basketballs = 49 :=
sorry

end school_basketballs_l84_84215


namespace each_persons_tip_l84_84845

theorem each_persons_tip
  (cost_julie cost_letitia cost_anton : ℕ)
  (H1 : cost_julie = 10)
  (H2 : cost_letitia = 20)
  (H3 : cost_anton = 30)
  (total_people : ℕ)
  (H4 : total_people = 3)
  (tip_percentage : ℝ)
  (H5 : tip_percentage = 0.20) :
  ∃ tip_per_person : ℝ, tip_per_person = 4 := 
by
  sorry

end each_persons_tip_l84_84845


namespace duckweed_quarter_covered_l84_84898

theorem duckweed_quarter_covered (N : ℕ) (h1 : N = 64) (h2 : ∀ n : ℕ, n < N → (n + 1 < N) → ∃ k, k = n + 1) :
  N - 2 = 62 :=
by
  sorry

end duckweed_quarter_covered_l84_84898


namespace completing_the_square_equation_l84_84099

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l84_84099


namespace solve_system_equations_l84_84471

theorem solve_system_equations (a b c x y z : ℝ) (h1 : x + y + z = 0)
(h2 : c * x + a * y + b * z = 0)
(h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
(x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = a - b ∧ y = b - c ∧ z = c - a) := 
sorry

end solve_system_equations_l84_84471


namespace basketball_third_quarter_points_l84_84179

noncomputable def teamA_points (a r : ℕ) : ℕ :=
a + a*r + a*r^2 + a*r^3

noncomputable def teamB_points (b d : ℕ) : ℕ :=
b + (b + d) + (b + 2*d) + (b + 3*d)

theorem basketball_third_quarter_points (a b d : ℕ) (r : ℕ) 
    (h1 : r > 1) (h2 : d > 0) (h3 : a * (r^4 - 1) / (r - 1) = 4 * b + 6 * d + 3)
    (h4 : a * (r^4 - 1) / (r - 1) ≤ 100) (h5 : 4 * b + 6 * d ≤ 100) :
    a * r^2 + b + 2 * d = 60 :=
sorry

end basketball_third_quarter_points_l84_84179


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l84_84262

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l84_84262


namespace find_a_l84_84441

theorem find_a :
  ∃ a : ℝ, (∀ t1 t2 : ℝ, t1 + t2 = -a ∧ t1 * t2 = -2017 ∧ 2 * t1 = 4) → a = 1006.5 :=
by
  sorry

end find_a_l84_84441


namespace equation_of_parallel_plane_l84_84140

theorem equation_of_parallel_plane {A B C D : ℤ} (hA : A = 3) (hB : B = -2) (hC : C = 4) (hD : D = -16)
    (point : ℝ × ℝ × ℝ) (pass_through : point = (2, -3, 1)) (parallel_plane : A * 2 + B * (-3) + C * 1 + D = 0)
    (gcd_condition : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) :
    A * 2 + B * (-3) + C + D = 0 ∧ A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by
  sorry

end equation_of_parallel_plane_l84_84140


namespace rhombus_area_l84_84211

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 120 :=
by
  sorry

end rhombus_area_l84_84211


namespace roberto_starting_salary_l84_84038

-- Given conditions as Lean definitions
def current_salary : ℝ := 134400
def previous_salary (S : ℝ) : ℝ := 1.40 * S

-- The proof problem statement
theorem roberto_starting_salary (S : ℝ) 
    (h1 : current_salary = 1.20 * previous_salary S) : 
    S = 80000 :=
by
  -- We will insert the proof here
  sorry

end roberto_starting_salary_l84_84038


namespace smallest_n_power_2013_ends_001_l84_84561

theorem smallest_n_power_2013_ends_001 :
  ∃ n : ℕ, n > 0 ∧ 2013^n % 1000 = 1 ∧ ∀ m : ℕ, m > 0 ∧ 2013^m % 1000 = 1 → n ≤ m := 
sorry

end smallest_n_power_2013_ends_001_l84_84561


namespace arithmetic_seq_sum_l84_84302

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end arithmetic_seq_sum_l84_84302


namespace winning_vote_majority_l84_84834

theorem winning_vote_majority (h1 : 0.70 * 900 = 630)
                             (h2 : 0.30 * 900 = 270) :
  630 - 270 = 360 :=
by
  sorry

end winning_vote_majority_l84_84834


namespace perimeter_of_square_l84_84209

theorem perimeter_of_square (area : ℝ) (h : area = 392) : 
  ∃ (s : ℝ), 4 * s = 56 * Real.sqrt 2 :=
by 
  use (Real.sqrt 392)
  sorry

end perimeter_of_square_l84_84209


namespace AM_bisects_BAC_l84_84779

open Real

namespace Geometry

theorem AM_bisects_BAC
  (a b : ℝ) 
  (h_a_pos : 0 < a) 
  (h_b_pos : 0 < b)
  (A B C D K M : ℝ × ℝ)
  (h_A : A = (0, 0))
  (h_B : B = (a, 0))
  (h_C : C = (a, b))
  (h_D : D = (0, b))
  (h_K_def : K = (0, b + sqrt (a^2 + b^2)))
  (h_M_def : M = (a/2, (b + sqrt (a^2 + b^2)) / 2)) :
  angle_bisector A B C M :=
sorry

end Geometry

end AM_bisects_BAC_l84_84779


namespace simplify_expression_l84_84916

theorem simplify_expression : 1 + 3 / (2 + 5 / 6) = 35 / 17 := 
  sorry

end simplify_expression_l84_84916


namespace range_of_a_l84_84284

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a-1)*x^2 + a*x + 1 ≥ 0) : a ≥ 1 :=
by {
  sorry
}

end range_of_a_l84_84284


namespace y_coordinate_eq_l84_84074

theorem y_coordinate_eq (y : ℝ) : 
  (∃ y, (√(9 + y^2) = √(4 + (5 - y)^2))) ↔ (y = 2) :=
by
  sorry

end y_coordinate_eq_l84_84074


namespace total_operation_time_correct_l84_84186

def accessories_per_doll := 2 + 3 + 1 + 5
def number_of_dolls := 12000
def time_per_doll := 45
def time_per_accessory := 10
def total_accessories := number_of_dolls * accessories_per_doll
def time_for_dolls := number_of_dolls * time_per_doll
def time_for_accessories := total_accessories * time_per_accessory
def total_combined_time := time_for_dolls + time_for_accessories

theorem total_operation_time_correct :
  total_combined_time = 1860000 :=
by
  sorry

end total_operation_time_correct_l84_84186


namespace factorize_difference_of_squares_l84_84924

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end factorize_difference_of_squares_l84_84924


namespace new_population_difference_l84_84782

def population_eagles : ℕ := 150
def population_falcons : ℕ := 200
def population_hawks : ℕ := 320
def population_owls : ℕ := 270
def increase_rate : ℕ := 10

theorem new_population_difference :
  let least_populous := min population_eagles (min population_falcons (min population_hawks population_owls))
  let most_populous := max population_eagles (max population_falcons (max population_hawks population_owls))
  let increased_least_populous := least_populous + least_populous * increase_rate / 100
  most_populous - increased_least_populous = 155 :=
by
  sorry

end new_population_difference_l84_84782


namespace rectangle_enclosed_by_lines_l84_84448

theorem rectangle_enclosed_by_lines :
  ∀ (H : ℕ) (V : ℕ), H = 5 → V = 4 → (nat.choose H 2) * (nat.choose V 2) = 60 :=
by
  intros H V h_eq H_eq
  rw [h_eq, H_eq]
  simp
  sorry

end rectangle_enclosed_by_lines_l84_84448


namespace price_of_pants_l84_84680

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end price_of_pants_l84_84680


namespace value_of_x_l84_84475

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l84_84475


namespace part1_part2_l84_84240

theorem part1 (n : ℕ) (hn : 0 < n) : (3^(2 * n) - 8 * n - 1) % 64 = 0 :=
sorry

theorem part2 : (2^30 - 3) % 7 = 5 :=
sorry

end part1_part2_l84_84240


namespace y_intercept_range_l84_84458

-- Define the points A and B
def pointA : ℝ × ℝ := (-1, -2)
def pointB : ℝ × ℝ := (2, 3)

-- We define the predicate for the line intersection condition
def line_intersects_segment (c : ℝ) : Prop :=
  let x_val_a := -1
  let y_val_a := -2
  let x_val_b := 2
  let y_val_b := 3
  -- Line equation at point A
  let eqn_a := x_val_a + y_val_a - c
  -- Line equation at point B
  let eqn_b := x_val_b + y_val_b - c
  -- We assert that the line must intersect the segment AB
  eqn_a ≤ 0 ∧ eqn_b ≥ 0 ∨ eqn_a ≥ 0 ∧ eqn_b ≤ 0

-- The main theorem to prove the range of c
theorem y_intercept_range : 
  ∃ c_min c_max : ℝ, c_min = -3 ∧ c_max = 5 ∧
  ∀ c, line_intersects_segment c ↔ c_min ≤ c ∧ c ≤ c_max :=
by
  existsi -3
  existsi 5
  sorry

end y_intercept_range_l84_84458


namespace peanuts_added_l84_84872

theorem peanuts_added (initial final added : ℕ) (h1 : initial = 4) (h2 : final = 8) (h3 : final = initial + added) : added = 4 :=
by
  rw [h1] at h3
  rw [h2] at h3
  sorry

end peanuts_added_l84_84872


namespace sum_of_consecutive_integers_product_336_l84_84709

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l84_84709


namespace completing_square_result_l84_84093

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l84_84093


namespace ratio_shorter_to_longer_l84_84890

theorem ratio_shorter_to_longer (total_length shorter_length longer_length : ℕ) (h1 : total_length = 40) 
(h2 : shorter_length = 16) (h3 : longer_length = total_length - shorter_length) : 
(shorter_length / Nat.gcd shorter_length longer_length) / (longer_length / Nat.gcd shorter_length longer_length) = 2 / 3 :=
by
  sorry

end ratio_shorter_to_longer_l84_84890


namespace no_valid_number_l84_84402

theorem no_valid_number (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 9) : ¬ ∃ (y : ℕ), (x * 100 + 3 * 10 + y) % 11 = 0 :=
by
  sorry

end no_valid_number_l84_84402


namespace area_of_yard_l84_84539

def length {w : ℝ} : ℝ := 2 * w + 30

def perimeter {w l : ℝ} (cond_len : l = 2 * w + 30) : Prop := 2 * w + 2 * l = 700

theorem area_of_yard {w l A : ℝ} 
  (cond_len : l = 2 * w + 30) 
  (cond_perim : 2 * w + 2 * l = 700) : 
  A = w * l := 
  sorry

end area_of_yard_l84_84539


namespace solve_for_x_l84_84481

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l84_84481


namespace hexagon_angles_l84_84988

theorem hexagon_angles
  (AB CD EF BC DE FA : ℝ)
  (F A B C D E : Type*)
  (FAB ABC EFA CDE : ℝ)
  (h1 : AB = CD)
  (h2 : AB = EF)
  (h3 : BC = DE)
  (h4 : BC = FA)
  (h5 : FAB + ABC = 240)
  (h6 : FAB + EFA = 240) :
  FAB + CDE = 240 :=
sorry

end hexagon_angles_l84_84988


namespace max_sequence_sum_l84_84294

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a1 d : α) (n : ℕ) : α :=
  a1 + d * n

noncomputable def sequenceSum (a1 d : α) (n : ℕ) : α :=
  n * (a1 + (a1 + d * (n - 1))) / 2

theorem max_sequence_sum (a1 d : α) (n : ℕ) (hn : 5 ≤ n ∧ n ≤ 10)
    (h1 : d < 0) (h2 : sequenceSum a1 d 5 = sequenceSum a1 d 10) :
    n = 7 ∨ n = 8 :=
  sorry

end max_sequence_sum_l84_84294


namespace intersect_sets_l84_84472

open Set

noncomputable def P : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}

theorem intersect_sets (U : Set ℝ) (P : Set ℝ) (Q : Set ℝ) :
  U = univ → P = {x : ℝ | x^2 - 2 * x ≤ 0} → Q = {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x} →
  P ∩ Q = Icc (0 : ℝ) (2 : ℝ) :=
by
  intros
  sorry

end intersect_sets_l84_84472


namespace completing_the_square_equation_l84_84097

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l84_84097


namespace prime_divisors_of_50_fact_eq_15_l84_84646

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end prime_divisors_of_50_fact_eq_15_l84_84646


namespace odd_n_iff_exists_non_integer_rationals_l84_84802

theorem odd_n_iff_exists_non_integer_rationals
  (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ a.denom ≠ 1 ∧ b.denom ≠ 1 ∧ (a + b).denom = 1 ∧ (a^n + b^n).denom = 1) ↔ n % 2 = 1 := 
sorry

end odd_n_iff_exists_non_integer_rationals_l84_84802


namespace josh_initial_marbles_l84_84843

def marbles_initial (lost : ℕ) (left : ℕ) : ℕ := lost + left

theorem josh_initial_marbles :
  marbles_initial 5 4 = 9 :=
by sorry

end josh_initial_marbles_l84_84843


namespace parallel_lines_necessary_not_sufficient_l84_84600

variables {a1 b1 a2 b2 c1 c2 : ℝ}

def determinant (a1 b1 a2 b2 : ℝ) : ℝ := a1 * b2 - a2 * b1

theorem parallel_lines_necessary_not_sufficient
  (h1 : a1^2 + b1^2 ≠ 0)
  (h2 : a2^2 + b2^2 ≠ 0)
  : (determinant a1 b1 a2 b2 = 0) → 
    (a1 * x + b1 * y + c1 = 0 ∧ a2 * x + b2 * y + c2 =0 → exists k : ℝ, (a1 = k ∧ b1 = k)) ∧ 
    (determinant a1 b1 a2 b2 = 0 → (a2 * x + b2 * y + c2 = a1 * x + b1 * y + c1 → false)) :=
sorry

end parallel_lines_necessary_not_sufficient_l84_84600


namespace total_customers_l84_84905

def initial_customers : ℝ := 29.0    -- 29.0 initial customers
def lunch_rush_customers : ℝ := 20.0 -- Adds 20.0 customers during lunch rush
def additional_customers : ℝ := 34.0 -- Adds 34.0 more customers

theorem total_customers : (initial_customers + lunch_rush_customers + additional_customers) = 83.0 :=
by
  sorry

end total_customers_l84_84905


namespace no_integer_solution_for_equation_l84_84860

theorem no_integer_solution_for_equation :
    ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x * y * z - 1 :=
by
  sorry

end no_integer_solution_for_equation_l84_84860


namespace min_value_ineq_l84_84674

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end min_value_ineq_l84_84674


namespace total_number_of_shells_l84_84355

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end total_number_of_shells_l84_84355


namespace problem_statement_l84_84617

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, 8 < x → f (x) > f (x + 1))
variable (h2 : ∀ x, f (x + 8) = f (-x + 8))

theorem problem_statement : f 7 > f 10 := by
  sorry

end problem_statement_l84_84617


namespace number_of_boys_l84_84388

theorem number_of_boys (girls boys : ℕ) (total_books books_girls books_boys books_per_student : ℕ)
  (h1 : girls = 15)
  (h2 : total_books = 375)
  (h3 : books_girls = 225)
  (h4 : total_books = books_girls + books_boys)
  (h5 : books_girls = girls * books_per_student)
  (h6 : books_boys = boys * books_per_student)
  (h7 : books_per_student = 15) :
  boys = 10 :=
by
  sorry

end number_of_boys_l84_84388


namespace gcf_120_180_300_l84_84876

theorem gcf_120_180_300 : Nat.gcd (Nat.gcd 120 180) 300 = 60 := 
by eval_gcd 120 180 300

end gcf_120_180_300_l84_84876


namespace value_of_x_l84_84497

theorem value_of_x (x : ℝ) (h : x = 90 + (11 / 100) * 90) : x = 99.9 :=
by {
  sorry
}

end value_of_x_l84_84497


namespace sum_of_powers_eight_l84_84357

variable {a b : ℝ}

theorem sum_of_powers_eight :
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^8 + b^8 = 47 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  -- Proof to be filled in
  sorry

end sum_of_powers_eight_l84_84357


namespace cost_of_chlorine_l84_84942

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end cost_of_chlorine_l84_84942


namespace mike_games_l84_84519

theorem mike_games (initial_money spent_money game_cost remaining_games : ℕ)
  (h1 : initial_money = 101)
  (h2 : spent_money = 47)
  (h3 : game_cost = 6)
  (h4 : remaining_games = (initial_money - spent_money) / game_cost) :
  remaining_games = 9 := by
  sorry

end mike_games_l84_84519


namespace reduction_amount_is_250_l84_84058

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end reduction_amount_is_250_l84_84058


namespace compute_expression_value_l84_84935

theorem compute_expression_value (x y : ℝ) (hxy : x ≠ y) 
  (h : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (xy + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (xy + 1) = 2 :=
by
  sorry

end compute_expression_value_l84_84935


namespace qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l84_84525

variable (m : Int)

theorem qiqi_initial_batteries (m : Int) : 
  let Qiqi_initial := 2 * m - 2
  Qiqi_initial = 2 * m - 2 := sorry

theorem qiqi_jiajia_difference_after_transfer (m : Int) : 
  let Qiqi_after := 2 * m - 2 - 2
  let Jiajia_after := m + 2
  Qiqi_after - Jiajia_after = m - 6 := sorry

end qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l84_84525


namespace ab_value_l84_84333

theorem ab_value (a b : ℤ) (h : 48 * a * b = 65 * a * b) : a * b = 0 :=
  sorry

end ab_value_l84_84333


namespace find_growth_rate_calculate_fourth_day_donation_l84_84554

-- Define the conditions
def first_day_donation : ℝ := 3000
def third_day_donation : ℝ := 4320
def growth_rate (x : ℝ) : Prop := (1 + x)^2 = third_day_donation / first_day_donation

-- Since the problem states growth rate for second and third day is the same,
-- we need to find that rate which is equivalent to solving the above proposition for x.

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.2 := by
  sorry

-- Calculate the fourth day's donation based on the growth rate found.
def fourth_day_donation (third_day : ℝ) (growth_rate : ℝ) : ℝ :=
  third_day * (1 + growth_rate)

theorem calculate_fourth_day_donation : 
  ∀ x : ℝ, growth_rate x → x = 0.2 → fourth_day_donation third_day_donation x = 5184 := by 
  sorry

end find_growth_rate_calculate_fourth_day_donation_l84_84554


namespace smallest_B_for_divisibility_by_4_l84_84507

theorem smallest_B_for_divisibility_by_4 : 
  ∃ (B : ℕ), B < 10 ∧ (4 * 1000000 + B * 100000 + 80000 + 3961) % 4 = 0 ∧ ∀ (B' : ℕ), (B' < B ∧ B' < 10) → ¬ ((4 * 1000000 + B' * 100000 + 80000 + 3961) % 4 = 0) := 
sorry

end smallest_B_for_divisibility_by_4_l84_84507


namespace population_net_increase_in_one_day_l84_84339

-- Define the problem conditions
def birth_rate : ℕ := 6 / 2  -- births per second
def death_rate : ℕ := 3 / 2  -- deaths per second
def seconds_in_a_day : ℕ := 60 * 60 * 24

-- Define the assertion we want to prove
theorem population_net_increase_in_one_day : 
  ( (birth_rate - death_rate) * seconds_in_a_day ) = 259200 := by
  -- Since 6/2 = 3 and 3/2 = 1.5 is not an integer in Lean, we use ratios directly
  sorry  -- Proof is not required

end population_net_increase_in_one_day_l84_84339


namespace area_relationship_l84_84902

theorem area_relationship (a b c : ℝ) (h : a^2 + b^2 = c^2) : (a + b)^2 = a^2 + 2*a*b + b^2 := 
by sorry

end area_relationship_l84_84902


namespace limit_seq_l84_84592

open Real

noncomputable def seq_limit : ℕ → ℝ :=
  λ n => (sqrt (n^5 - 8) - n * sqrt (n * (n^2 + 5))) / sqrt n

theorem limit_seq : tendsto seq_limit atTop (𝓝 (-5/2)) :=
  sorry

end limit_seq_l84_84592


namespace average_rainfall_correct_l84_84660

/-- In July 1861, 366 inches of rain fell in Cherrapunji, India. -/
def total_rainfall : ℤ := 366

/-- July has 31 days. -/
def days_in_july : ℤ := 31

/-- Each day has 24 hours. -/
def hours_per_day : ℤ := 24

/-- The total number of hours in July -/
def total_hours_in_july : ℤ := days_in_july * hours_per_day

/-- The average rainfall in inches per hour during July 1861 in Cherrapunji, India -/
def average_rainfall_per_hour : ℤ := total_rainfall / total_hours_in_july

/-- Proof that the average rainfall in inches per hour is 366 / (31 * 24) -/
theorem average_rainfall_correct : average_rainfall_per_hour = 366 / (31 * 24) :=
by
  /- We skip the proof as it is not required. -/
  sorry

end average_rainfall_correct_l84_84660


namespace sum_of_three_consecutive_integers_product_336_l84_84717

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l84_84717


namespace exists_equal_subinterval_l84_84031

open Set Metric Function

variable {a b : ℝ}
variable {f : ℕ → ℝ → ℝ}
variable {n m : ℕ}

-- Define the conditions
def continuous_on_interval (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ n, ContinuousOn (f n) (Icc a b)

def root_cond (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ x ∈ Icc a b, ∃ m n, m ≠ n ∧ f m x = f n x

-- The main theorem statement
theorem exists_equal_subinterval (f : ℕ → ℝ → ℝ) (a b : ℝ) 
  (h_cont : continuous_on_interval f a b) 
  (h_root : root_cond f a b) : 
  ∃ (c d : ℝ), c < d ∧ Icc c d ⊆ Icc a b ∧ ∃ m n, m ≠ n ∧ ∀ x ∈ Icc c d, f m x = f n x := 
sorry

end exists_equal_subinterval_l84_84031


namespace number_of_special_four_digit_integers_l84_84640

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l84_84640


namespace percent_increase_hypotenuse_l84_84422

theorem percent_increase_hypotenuse :
  let l1 := 3
  let l2 := 1.25 * l1
  let l3 := 1.25 * l2
  let l4 := 1.25 * l3
  let h1 := l1 * Real.sqrt 2
  let h4 := l4 * Real.sqrt 2
  ((h4 - h1) / h1) * 100 = 95.3 :=
by
  sorry

end percent_increase_hypotenuse_l84_84422


namespace probability_at_most_one_success_in_three_attempts_l84_84892

noncomputable def basket_probability := (2 : ℚ) / 3

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_most_one_success_in_three_attempts :
  binomial_probability 3 0 basket_probability + binomial_probability 3 1 basket_probability = 7 / 27 :=
by
  sorry

end probability_at_most_one_success_in_three_attempts_l84_84892


namespace defective_items_count_l84_84684

variables 
  (total_items : ℕ)
  (total_video_games : ℕ)
  (total_DVDs : ℕ)
  (total_books : ℕ)
  (working_video_games : ℕ)
  (working_DVDs : ℕ)

theorem defective_items_count
  (h1 : total_items = 56)
  (h2 : total_video_games = 30)
  (h3 : total_DVDs = 15)
  (h4 : total_books = total_items - total_video_games - total_DVDs)
  (h5 : working_video_games = 20)
  (h6 : working_DVDs = 10)
  : (total_video_games - working_video_games) + (total_DVDs - working_DVDs) = 15 :=
sorry

end defective_items_count_l84_84684


namespace find_PB_l84_84404

variables (P A B C D : Point) (PA PD PC PB : ℝ)
-- Assume P is interior to rectangle ABCD
-- Conditions
axiom hPA : PA = 3
axiom hPD : PD = 4
axiom hPC : PC = 5

-- The main statement to prove
theorem find_PB (P A B C D : Point) (PA PD PC PB : ℝ)
  (hPA : PA = 3) (hPD : PD = 4) (hPC : PC = 5) : PB = 3 * Real.sqrt 2 :=
by
  sorry

end find_PB_l84_84404


namespace Paul_work_time_l84_84611

def work_completed (rate: ℚ) (time: ℚ) : ℚ := rate * time

noncomputable def George_work_rate : ℚ := 3 / 5 / 9

noncomputable def combined_work_rate : ℚ := 2 / 5 / 4

noncomputable def Paul_work_rate : ℚ := combined_work_rate - George_work_rate

theorem Paul_work_time :
  (work_completed Paul_work_rate 30) = 1 :=
by
  have h_george_rate : George_work_rate = 1 / 15 :=
    by norm_num [George_work_rate]
  have h_combined_rate : combined_work_rate = 1 / 10 :=
    by norm_num [combined_work_rate]
  have h_paul_rate : Paul_work_rate = 1 / 30 :=
    by norm_num [Paul_work_rate, h_combined_rate, h_george_rate]
  sorry -- Complete proof statement here

end Paul_work_time_l84_84611


namespace Brandy_energy_drinks_l84_84382

theorem Brandy_energy_drinks 
  (maximum_safe_amount : ℕ)
  (caffeine_per_drink : ℕ)
  (extra_safe_caffeine : ℕ)
  (x : ℕ)
  (h1 : maximum_safe_amount = 500)
  (h2 : caffeine_per_drink = 120)
  (h3 : extra_safe_caffeine = 20)
  (h4 : caffeine_per_drink * x + extra_safe_caffeine = maximum_safe_amount) :
  x = 4 :=
by
  sorry

end Brandy_energy_drinks_l84_84382


namespace players_quit_game_l84_84069

variable (total_players initial num_lives players_left players_quit : Nat)
variable (each_player_lives : Nat)

theorem players_quit_game :
  (initial = 8) →
  (each_player_lives = 3) →
  (num_lives = 15) →
  players_left = num_lives / each_player_lives →
  players_quit = initial - players_left →
  players_quit = 3 :=
by
  intros h_initial h_each_player_lives h_num_lives h_players_left h_players_quit
  sorry

end players_quit_game_l84_84069


namespace min_expr_l84_84191

theorem min_expr (a b c d : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) (hd : Odd d) (a_pos: 0 < a) (b_pos: 0 < b) (c_pos: 0 < c) (d_pos: 0 < d)
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) = 34 := 
sorry

end min_expr_l84_84191


namespace units_digit_p_plus_5_l84_84571

theorem units_digit_p_plus_5 (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 = 6) (h3 : (p^3 % 10) - (p^2 % 10) = 0) : (p + 5) % 10 = 1 :=
by
  sorry

end units_digit_p_plus_5_l84_84571


namespace not_p_or_not_q_implies_p_and_q_and_p_or_q_l84_84975

variable (p q : Prop)

theorem not_p_or_not_q_implies_p_and_q_and_p_or_q (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
sorry

end not_p_or_not_q_implies_p_and_q_and_p_or_q_l84_84975


namespace inverse_of_A_is_zeroMatrix_l84_84608

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 8], ![-2, -4]]
def zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![![0, 0], ![0, 0]]

theorem inverse_of_A_is_zeroMatrix (h : A.det = 0) : A⁻¹ = zeroMatrix := by
  sorry

end inverse_of_A_is_zeroMatrix_l84_84608


namespace not_simplifiable_by_difference_of_squares_l84_84565

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end not_simplifiable_by_difference_of_squares_l84_84565


namespace ratio_eq_one_l84_84826

theorem ratio_eq_one (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
sorry

end ratio_eq_one_l84_84826


namespace total_drums_hit_l84_84844

/-- 
Given the conditions of the problem, Juanita hits 4500 drums in total. 
-/
theorem total_drums_hit (entry_fee cost_per_drum_hit earnings_per_drum_hit_beyond_200_double
                         net_loss: ℝ) 
                         (first_200_drums hits_after_200: ℕ) :
  entry_fee = 10 → 
  cost_per_drum_hit = 0.02 →
  earnings_per_drum_hit_beyond_200_double = 0.025 →
  net_loss = -7.5 →
  hits_after_200 = 4300 →
  first_200_drums = 200 →
  (-net_loss = entry_fee + (first_200_drums * cost_per_drum_hit) +
   (hits_after_200 * (earnings_per_drum_hit_beyond_200_double - cost_per_drum_hit))) →
  first_200_drums + hits_after_200 = 4500 :=
by
  intro h_entry_fee h_cost_per_drum_hit h_earnings_per_drum_hit_beyond_200_double h_net_loss h_hits_after_200
       h_first_200_drums h_loss_equation
  sorry

end total_drums_hit_l84_84844


namespace new_car_travel_distance_l84_84119

theorem new_car_travel_distance
  (old_distance : ℝ)
  (new_distance : ℝ)
  (h1 : old_distance = 150)
  (h2 : new_distance = 1.30 * old_distance) : 
  new_distance = 195 := 
by 
  /- include required assumptions and skip the proof. -/
  sorry

end new_car_travel_distance_l84_84119


namespace bank_check_problem_l84_84412

theorem bank_check_problem :
  ∃ (x y : ℕ), (0 ≤ y ∧ y ≤ 99) ∧ (y + (x : ℚ) / 100 - 0.05 = 2 * (x + (y : ℚ) / 100)) ∧ x = 31 ∧ y = 63 :=
by
  -- Definitions and Conditions
  sorry

end bank_check_problem_l84_84412


namespace number_of_ones_and_zeros_not_perfect_square_l84_84361

open Int

theorem number_of_ones_and_zeros_not_perfect_square (k : ℕ) : 
  let N := (10^k) * (10^300 - 1) / 9
  ¬ ∃ m : ℤ, m^2 = N :=
by
  sorry

end number_of_ones_and_zeros_not_perfect_square_l84_84361


namespace cubic_function_value_l84_84625

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

theorem cubic_function_value (p q r s : ℝ) (h : g (-3) p q r s = -2) :
  12 * p - 6 * q + 3 * r - s = 2 :=
sorry

end cubic_function_value_l84_84625


namespace smallest_n_mod_equiv_l84_84394

theorem smallest_n_mod_equiv (n : ℕ) (h : 5 * n ≡ 4960 [MOD 31]) : n = 31 := by 
  sorry

end smallest_n_mod_equiv_l84_84394


namespace books_remaining_after_second_day_l84_84410

variable (x a b c d : ℕ)

theorem books_remaining_after_second_day :
  let books_borrowed_first_day := a * b
  let books_borrowed_second_day := c
  let books_returned_second_day := (d * books_borrowed_first_day) / 100
  x - books_borrowed_first_day - books_borrowed_second_day + books_returned_second_day =
  x - (a * b) - c + ((d * (a * b)) / 100) :=
sorry

end books_remaining_after_second_day_l84_84410


namespace expand_product_l84_84134

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := 
by
  sorry

end expand_product_l84_84134


namespace arrangement_is_correct_l84_84009

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l84_84009


namespace quadratic_real_roots_range_l84_84651

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end quadratic_real_roots_range_l84_84651


namespace trigonometric_transform_l84_84698

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := f (x - 3)
noncomputable def g (x : ℝ) : ℝ := 3 * h (x / 3)

theorem trigonometric_transform (x : ℝ) : g x = 3 * Real.sin (x / 3 - 3) := by
  sorry

end trigonometric_transform_l84_84698


namespace minimum_value_inequality_l84_84303

theorem minimum_value_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 = 1) :
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3 / 2) :=
sorry

end minimum_value_inequality_l84_84303


namespace find_y_l84_84473

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 :=
by
  sorry

end find_y_l84_84473


namespace translate_parabola_l84_84221

open Real 

theorem translate_parabola (x y : ℝ) :
  (y = x^2 + 2*x - 1) ↔ (y = (x - 1)^2 - 3) :=
by
  sorry

end translate_parabola_l84_84221


namespace jacob_has_5_times_more_l84_84424

variable (A J D : ℕ)
variable (hA : A = 75)
variable (hAJ : A = J / 2)
variable (hD : D = 30)

theorem jacob_has_5_times_more (hA : A = 75) (hAJ : A = J / 2) (hD : D = 30) : J / D = 5 :=
sorry

end jacob_has_5_times_more_l84_84424


namespace cost_price_equal_l84_84250

theorem cost_price_equal (total_selling_price : ℝ) (profit_percent_first profit_percent_second : ℝ) (length_first_segment length_second_segment : ℝ) (C : ℝ) :
  total_selling_price = length_first_segment * (1 + profit_percent_first / 100) * C + length_second_segment * (1 + profit_percent_second / 100) * C →
  C = 15360 / (66 + 72) :=
by {
  sorry
}

end cost_price_equal_l84_84250


namespace trajectory_of_moving_circle_l84_84149

noncomputable def trajectory_equation_of_moving_circle_center 
  (x y : Real) : Prop :=
  (∃ r : Real, 
    ((x + 5)^2 + y^2 = 16) ∧ 
    ((x - 5)^2 + y^2 = 16)
  ) → (x > 0 → x^2 / 16 - y^2 / 9 = 1)

-- here's the statement of the proof problem
theorem trajectory_of_moving_circle
  (h₁ : ∀ x y : Real, (x + 5)^2 + y^2 = 16)
  (h₂ : ∀ x y : Real, (x - 5)^2 + y^2 = 16) :
  ∀ x y : Real, trajectory_equation_of_moving_circle_center x y :=
sorry

end trajectory_of_moving_circle_l84_84149


namespace running_speed_l84_84576

theorem running_speed (side : ℕ) (time_seconds : ℕ) (speed_result : ℕ) 
  (h1 : side = 50) (h2 : time_seconds = 60) (h3 : speed_result = 12) : 
  (4 * side * 3600) / (time_seconds * 1000) = speed_result :=
by
  sorry

end running_speed_l84_84576


namespace smallest_angle_in_right_triangle_l84_84976

-- Given conditions
def angle_α := 90 -- The right-angle in degrees
def angle_β := 55 -- The given angle in degrees

-- Goal: Prove that the smallest angle is 35 degrees.
theorem smallest_angle_in_right_triangle (a b c : ℕ) (h1 : a = angle_α) (h2 : b = angle_β) (h3 : c = 180 - a - b) : c = 35 := 
by {
  -- use sorry to skip the proof steps
  sorry
}

end smallest_angle_in_right_triangle_l84_84976


namespace tan_eleven_pi_over_four_eq_neg_one_l84_84931

noncomputable def tan_of_eleven_pi_over_four : Real := 
  let to_degrees (x : Real) : Real := x * 180 / Real.pi
  let angle := to_degrees (11 * Real.pi / 4)
  let simplified := angle - 360 * Real.floor (angle / 360)
  if simplified < 0 then
    simplified := simplified + 360
  if simplified = 135 then -1
  else
    undefined

theorem tan_eleven_pi_over_four_eq_neg_one :
  tan (11 * Real.pi / 4) = -1 := 
by
  sorry

end tan_eleven_pi_over_four_eq_neg_one_l84_84931


namespace infinite_natural_numbers_factorable_polynomial_l84_84370

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l84_84370


namespace coprime_3x3_grid_l84_84016

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l84_84016


namespace probability_at_least_one_boy_and_girl_l84_84257
-- Necessary imports

-- Defining the probability problem in Lean 4
theorem probability_at_least_one_boy_and_girl (n : ℕ) (hn : n = 4)
    (p : ℚ) (hp : p = 1 / 2) :
    let prob_all_same := (p ^ n) + (p ^ n) in
    (1 - prob_all_same) = 7 / 8 := by
  -- Include the proof steps here
  sorry

end probability_at_least_one_boy_and_girl_l84_84257


namespace jogs_per_day_l84_84283

-- Definitions of conditions
def weekdays_per_week : ℕ := 5
def total_weeks : ℕ := 3
def total_miles : ℕ := 75

-- Define the number of weekdays in total weeks
def total_weekdays : ℕ := total_weeks * weekdays_per_week

-- Theorem to prove Damien jogs 5 miles per day on weekdays
theorem jogs_per_day : total_miles / total_weekdays = 5 := by
  sorry

end jogs_per_day_l84_84283


namespace find_extrema_l84_84325

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end find_extrema_l84_84325


namespace machine_p_takes_longer_l84_84994

variable (MachineP MachineQ MachineA : Type)
variable (s_prockets_per_hr : MachineA → ℝ)
variable (time_produce_s_prockets : MachineP → ℝ → ℝ)

noncomputable def machine_a_production : ℝ := 3
noncomputable def machine_q_production : ℝ := machine_a_production + 0.10 * machine_a_production

noncomputable def machine_q_time : ℝ := 330 / machine_q_production
noncomputable def additional_time : ℝ := sorry -- Since L is undefined

axiom machine_p_time : ℝ
axiom machine_p_time_eq_machine_q_time_plus_additional : machine_p_time = machine_q_time + additional_time

theorem machine_p_takes_longer : machine_p_time > machine_q_time := by
  rw [machine_p_time_eq_machine_q_time_plus_additional]
  exact lt_add_of_pos_right machine_q_time sorry  -- Need the exact L to conclude


end machine_p_takes_longer_l84_84994


namespace containers_per_truck_l84_84739

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l84_84739


namespace LCM_of_18_and_27_l84_84079

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end LCM_of_18_and_27_l84_84079


namespace smallest_d0_l84_84609

theorem smallest_d0 (r : ℕ) (hr : r ≥ 3) : ∃ d₀, d₀ = 2^(r - 2) ∧ (7^d₀ ≡ 1 [MOD 2^r]) :=
by
  sorry

end smallest_d0_l84_84609


namespace vertex_x_coord_l84_84053

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Conditions based on given points
def conditions (a b c : ℝ) : Prop :=
  quadratic a b c 2 = 4 ∧
  quadratic a b c 8 =4 ∧
  quadratic a b c 10 = 13

-- Statement to prove the x-coordinate of the vertex is 5
theorem vertex_x_coord (a b c : ℝ) (h : conditions a b c) : 
  (-(b) / (2 * a)) = 5 :=
by
  sorry

end vertex_x_coord_l84_84053


namespace remainder_of_3056_div_78_l84_84082

-- Define the necessary conditions and the statement
theorem remainder_of_3056_div_78 : (3056 % 78) = 14 :=
by
  sorry

end remainder_of_3056_div_78_l84_84082


namespace probability_A_given_B_l84_84594

namespace DiceProbability

noncomputable def P_A_given_B : ℚ :=
  let favorable_outcomes := 5
  let total_outcomes := 11
  favorable_outcomes / total_outcomes

theorem probability_A_given_B :
  let A := {outcome : ℕ × ℕ // outcome.1 ≠ outcome.2}
  let B := {outcome : ℕ × ℕ // outcome.1 = 6 ∨ outcome.2 = 6}
  P_A_given_B = 5 / 11 :=
by
  sorry

end DiceProbability

end probability_A_given_B_l84_84594


namespace math_problem_proof_l84_84793

noncomputable def problem_expr : ℚ :=
  ((11 + 1/9) - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / (36/10) / (2 + 6/25)

theorem math_problem_proof : problem_expr = 20 / 9 := by
  sorry

end math_problem_proof_l84_84793


namespace range_of_a_l84_84173

theorem range_of_a
  (a : ℝ)
  (h : ∀ (x : ℝ), 1 < x ∧ x < 4 → x^2 - 3 * x - 2 - a > 0) :
  a < 2 :=
sorry

end range_of_a_l84_84173


namespace john_not_stronger_than_ivan_l84_84358

-- Define strength relations
axiom stronger (a b : Type) : Prop

variable (whiskey liqueur vodka beer : Type)

axiom whiskey_stronger_than_vodka : stronger whiskey vodka
axiom liqueur_stronger_than_beer : stronger liqueur beer

-- Define types for cocktails and their strengths
variable (John_cocktail Ivan_cocktail : Type)

axiom John_mixed_whiskey_liqueur : John_cocktail
axiom Ivan_mixed_vodka_beer : Ivan_cocktail

-- Prove that it can't be asserted that John's cocktail is stronger
theorem john_not_stronger_than_ivan :
  ¬ (stronger John_cocktail Ivan_cocktail) :=
sorry

end john_not_stronger_than_ivan_l84_84358


namespace unit_digit_15_pow_100_l84_84562

theorem unit_digit_15_pow_100 : ((15^100) % 10) = 5 := 
by sorry

end unit_digit_15_pow_100_l84_84562


namespace p_is_necessary_but_not_sufficient_for_q_l84_84460

-- Conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0
def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- Proof target
theorem p_is_necessary_but_not_sufficient_for_q : 
  (∀ a : ℝ, p a → q a) ∧ ¬(∀ a : ℝ, q a → p a) :=
sorry

end p_is_necessary_but_not_sufficient_for_q_l84_84460


namespace rectangular_prism_diagonals_l84_84899

structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (length : ℝ)
  (height : ℝ)
  (width : ℝ)
  (length_ne_height : length ≠ height)
  (height_ne_width : height ≠ width)
  (width_ne_length : width ≠ length)

def diagonals (rp : RectangularPrism) : ℕ :=
  let face_diagonals := 12
  let space_diagonals := 4
  face_diagonals + space_diagonals

theorem rectangular_prism_diagonals (rp : RectangularPrism) :
  rp.faces = 6 →
  rp.edges = 12 →
  rp.vertices = 8 →
  diagonals rp = 16 ∧ 4 = 4 :=
by
  intros
  sorry

end rectangular_prism_diagonals_l84_84899


namespace andrew_age_proof_l84_84788

def andrew_age_problem : Prop :=
  ∃ (a g : ℚ), g = 15 * a ∧ g - a = 60 ∧ a = 30 / 7

theorem andrew_age_proof : andrew_age_problem :=
by
  sorry

end andrew_age_proof_l84_84788


namespace solve_trig_problem_l84_84616

noncomputable def trig_problem (α : ℝ) : Prop :=
  α ∈ (Set.Ioo 0 (Real.pi / 2)) ∪ Set.Ioo (Real.pi / 2) Real.pi ∧
  ∃ r : ℝ, r ≠ 0 ∧ Real.sin α * r = Real.sin (2 * α) ∧ Real.sin (2 * α) * r = Real.sin (4 * α)

theorem solve_trig_problem (α : ℝ) (h : trig_problem α) : α = 2 * Real.pi / 3 :=
by
  sorry

end solve_trig_problem_l84_84616


namespace completing_the_square_equation_l84_84096

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l84_84096


namespace smallest_possible_product_l84_84524

def digits : Set ℕ := {2, 4, 5, 8}

def is_valid_pair (a b : ℤ) : Prop :=
  let (d1, d2, d3, d4) := (a / 10, a % 10, b / 10, b % 10)
  {d1.toNat, d2.toNat, d3.toNat, d4.toNat} ⊆ digits ∧ {d1.toNat, d2.toNat, d3.toNat, d4.toNat} = digits

def smallest_product : ℤ :=
  1200

theorem smallest_possible_product :
  ∀ (a b : ℤ), is_valid_pair a b → a * b ≥ smallest_product :=
by
  intro a b h
  sorry

end smallest_possible_product_l84_84524


namespace sum_of_roots_of_quadratic_l84_84002

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, (x - 1) * (x + 4) = 18 -> (∃ a b c : ℝ, a = 1 ∧ b = 3 ∧ c = -22 ∧ ((a * x^2 + b * x + c = 0) ∧ (-b / a = -3))) :=
by
  sorry

end sum_of_roots_of_quadratic_l84_84002


namespace gcf_120_180_300_l84_84877

theorem gcf_120_180_300 : Nat.gcd (Nat.gcd 120 180) 300 = 60 := 
by eval_gcd 120 180 300

end gcf_120_180_300_l84_84877


namespace value_of_6z_l84_84498

theorem value_of_6z (x y z : ℕ) (h1 : 6 * z = 2 * x) (h2 : x + y + z = 26) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 6 * z = 36 :=
by
  sorry

end value_of_6z_l84_84498


namespace haleys_car_distance_l84_84061

theorem haleys_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) (distance_covered : ℕ) 
   (h_ratio : fuel_ratio = 4) (h_distance_ratio : distance_ratio = 7) (h_fuel_used : fuel_used = 44) :
   distance_covered = 77 := by
  -- Proof to be filled in
  sorry

end haleys_car_distance_l84_84061


namespace max_value_of_expr_l84_84296

noncomputable def max_expr_value (x : ℝ) : ℝ :=
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64)

theorem max_value_of_expr : ∀ x : ℝ, max_expr_value x ≤ 1/26 :=
by
  sorry

end max_value_of_expr_l84_84296


namespace Maria_score_l84_84354

theorem Maria_score (x : ℝ) (y : ℝ) (h1 : x = y + 50) (h2 : (x + y) / 2 = 105) : x = 130 :=
by
  sorry

end Maria_score_l84_84354


namespace percentage_increase_in_length_is_10_l84_84056

variables (L B : ℝ) -- original length and breadth
variables (length_increase_percentage breadth_increase_percentage area_increase_percentage : ℝ)

noncomputable def new_length (x : ℝ) : ℝ := L * (1 + x / 100)
noncomputable def new_breadth : ℝ := B * 1.25
noncomputable def new_area (x : ℝ) : ℝ := new_length L B x * new_breadth B
noncomputable def increased_area : ℝ := L * B * 1.375

theorem percentage_increase_in_length_is_10 :
 (breadth_increase_percentage = 25) →
 (area_increase_percentage = 37.5) →
 (new_area L B 10 = increased_area L B) → length_increase_percentage = 10
:= by
  sorry

end percentage_increase_in_length_is_10_l84_84056


namespace minimum_combinations_to_open_safe_l84_84540

open Fin

-- Definition of the conditions
def wheels := 3
def positions := 8

def opens_if_two_correct (x y z : Fin 8) : Prop :=
  x = y ∨ y = z ∨ z = x

-- The theorem statement
theorem minimum_combinations_to_open_safe : ∃ (n : ℕ), n = 32 ∧ ∀ (attempts : Fin (positions ^ wheels) → Fin (positions × positions × positions)), 
  (∀ t : Fin (positions ^ wheels), ∃ x y z, opens_if_two_correct (attempts t).1 (attempts t).2.1 (attempts t).2.2) → n = 32 :=
sorry

end minimum_combinations_to_open_safe_l84_84540


namespace rectangle_area_error_percentage_l84_84020

theorem rectangle_area_error_percentage (L W : ℝ) : 
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 0.7 := 
by
  sorry

end rectangle_area_error_percentage_l84_84020


namespace xiao_hong_home_to_school_distance_l84_84774

-- Definition of conditions
def distance_from_drop_to_school := 1000 -- in meters
def time_from_home_to_school_walking := 22.5 -- in minutes
def time_from_home_to_school_biking := 40 -- in minutes
def walking_speed := 80 -- in meters per minute
def bike_speed_slowdown := 800 -- in meters per minute

-- The main theorem statement
theorem xiao_hong_home_to_school_distance :
  ∃ d : ℝ, d = 12000 ∧ 
            distance_from_drop_to_school = 1000 ∧
            time_from_home_to_school_walking = 22.5 ∧
            time_from_home_to_school_biking = 40 ∧
            walking_speed = 80 ∧
            bike_speed_slowdown = 800 := 
sorry

end xiao_hong_home_to_school_distance_l84_84774


namespace range_of_square_root_l84_84176

theorem range_of_square_root (x : ℝ) : x + 4 ≥ 0 → x ≥ -4 :=
by
  intro h
  linarith

end range_of_square_root_l84_84176


namespace trig_inequality_l84_84109

open Real

theorem trig_inequality (x : ℝ) (n m : ℕ) (hx : 0 < x ∧ x < π / 2) (hnm : n > m) : 
  2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) := 
sorry

end trig_inequality_l84_84109


namespace flower_growth_l84_84036

theorem flower_growth (total_seeds : ℕ) (seeds_per_bed : ℕ) (max_grow_per_bed : ℕ) (h1 : total_seeds = 55) (h2 : seeds_per_bed = 15) (h3 : max_grow_per_bed = 60) : total_seeds ≤ 55 :=
by
  -- use the given conditions
  have h4 : total_seeds = 55 := h1
  sorry -- Proof goes here, omitted as instructed

end flower_growth_l84_84036


namespace math_equivalence_example_l84_84803

theorem math_equivalence_example :
  ((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2 = 494.09014144 := 
by
  sorry

end math_equivalence_example_l84_84803


namespace maximum_B_k_at_181_l84_84922

open Nat

theorem maximum_B_k_at_181 :
  let B : ℕ → ℝ := λ k, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  ∃ k : ℕ, k ≤ 2000 ∧ (∀ m : ℕ, m ≤ 2000 → B m ≤ B 181) :=
by
  let B := λ k : ℕ, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  use 181
  split
  · linarith
  · intro m hm
    sorry

end maximum_B_k_at_181_l84_84922


namespace number_of_lines_determined_by_12_points_l84_84449

theorem number_of_lines_determined_by_12_points : 
  ∀ (P : Finset (Fin 12)), (∀ p₁ p₂ p₃ ∈ P, p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ → ¬ (collinear ℝ {p₁, p₂, p₃})) → 
  (P.card.choose 2) = 66 :=
by
  intros P hP
  have h_card : P.card = 12 := by sorry
  rw [Finset.card_choose_two, h_card]
  norm_num
  -- exact proof we'll write specifics here 

end number_of_lines_determined_by_12_points_l84_84449


namespace odd_c_perfect_square_no_even_c_infinitely_many_solutions_l84_84949

open Nat

/-- Problem (1): prove that if c is an odd number, then c is a perfect square given 
    c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem odd_c_perfect_square (a b c : ℕ) (h_eq : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) (h_odd : Odd c) : ∃ k : ℕ, c = k^2 :=
  sorry

/-- Problem (2): prove that there does not exist an even number c that satisfies 
    c(a c + 1)^2 = (5c + 2b)(2c + b) for some a and b -/
theorem no_even_c (a b : ℕ) : ∀ c : ℕ, Even c → ¬ (c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) :=
  sorry

/-- Problem (3): prove that there are infinitely many solutions of positive integers 
    (a, b, c) that satisfy c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem infinitely_many_solutions (n : ℕ) : ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b) :=
  sorry

end odd_c_perfect_square_no_even_c_infinitely_many_solutions_l84_84949


namespace max_profit_l84_84408

noncomputable def profit (x : ℕ) : ℝ := -0.15 * (x : ℝ)^2 + 3.06 * (x : ℝ) + 30

theorem max_profit :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 ∧ ∀ y : ℕ, 0 ≤ y ∧ y ≤ 15 → profit y ≤ profit x :=
by
  sorry

end max_profit_l84_84408


namespace Alyssa_number_of_quarters_l84_84906

def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25
def num_pennies : ℕ := 7
def total_money : ℝ := 3.07

def num_quarters (q : ℕ) : Prop :=
  total_money - (num_pennies * value_penny) = q * value_quarter

theorem Alyssa_number_of_quarters : ∃ q : ℕ, num_quarters q ∧ q = 12 :=
by
  sorry

end Alyssa_number_of_quarters_l84_84906


namespace find_k_l84_84248

theorem find_k (k : ℝ) :
    (1 - 7) * (k - 3) = (3 - k) * (7 - 1) → k = 6.5 :=
by
sorry

end find_k_l84_84248


namespace distance_from_A_to_directrix_l84_84314

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l84_84314


namespace quadrilateral_AB_length_l84_84028

/-- Let ABCD be a quadrilateral with BC = CD = DA = 1, ∠DAB = 135°, and ∠ABC = 75°. 
    Prove that AB = (√6 - √2) / 2.
-/
theorem quadrilateral_AB_length (BC CD DA : ℝ) (angle_DAB angle_ABC : ℝ) (h1 : BC = 1)
    (h2 : CD = 1) (h3 : DA = 1) (h4 : angle_DAB = 135) (h5 : angle_ABC = 75) :
    AB = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
    sorry

end quadrilateral_AB_length_l84_84028


namespace probability_red_or_blue_l84_84548

noncomputable def total_marbles : ℕ := 100

noncomputable def probability_white : ℚ := 1 / 4

noncomputable def probability_green : ℚ := 1 / 5

theorem probability_red_or_blue :
  (1 - (probability_white + probability_green)) = 11 / 20 :=
by
  -- Proof is omitted
  sorry

end probability_red_or_blue_l84_84548


namespace number_of_shelves_l84_84885

-- Define the initial conditions and required values
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Define the result we want to prove
theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 4 :=
by
    -- Proof steps go here
    sorry

end number_of_shelves_l84_84885


namespace james_vegetable_consumption_l84_84984

def vegetable_consumption_weekdays (asparagus broccoli cauliflower spinach : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + spinach

def vegetable_consumption_weekend (asparagus broccoli cauliflower other_veg : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + other_veg

def total_vegetable_consumption (
  wd_asparagus wd_broccoli wd_cauliflower wd_spinach : ℚ)
  (sat_asparagus sat_broccoli sat_cauliflower sat_other : ℚ)
  (sun_asparagus sun_broccoli sun_cauliflower sun_other : ℚ) : ℚ :=
  5 * vegetable_consumption_weekdays wd_asparagus wd_broccoli wd_cauliflower wd_spinach +
  vegetable_consumption_weekend sat_asparagus sat_broccoli sat_cauliflower sat_other +
  vegetable_consumption_weekend sun_asparagus sun_broccoli sun_cauliflower sun_other

theorem james_vegetable_consumption :
  total_vegetable_consumption 0.5 0.75 0.875 0.5 0.3 0.4 0.6 1 0.3 0.4 0.6 0.5 = 17.225 :=
sorry

end james_vegetable_consumption_l84_84984


namespace backyard_area_l84_84044

theorem backyard_area {length width : ℝ} 
  (h1 : 30 * length = 1500) 
  (h2 : 12 * (2 * (length + width)) = 1500) : 
  length * width = 625 :=
by
  sorry

end backyard_area_l84_84044


namespace completing_the_square_equation_l84_84098

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l84_84098


namespace evaluate_expression_l84_84397

theorem evaluate_expression (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + (a⁻¹ / 3)) / a = 10 / 27 :=
by 
  sorry

end evaluate_expression_l84_84397


namespace lilly_fish_count_l84_84193

-- Define the number of fish Rosy has
def rosy_fish : ℕ := 9

-- Define the total number of fish
def total_fish : ℕ := 19

-- Define the statement that Lilly has 10 fish given the conditions
theorem lilly_fish_count : rosy_fish + lilly_fish = total_fish → lilly_fish = 10 := by
  intro h
  sorry

end lilly_fish_count_l84_84193


namespace tan_20_plus_4_sin_20_eq_sqrt_3_l84_84732

theorem tan_20_plus_4_sin_20_eq_sqrt_3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end tan_20_plus_4_sin_20_eq_sqrt_3_l84_84732


namespace no_x_satisfies_inequalities_l84_84607

theorem no_x_satisfies_inequalities : ¬ ∃ x : ℝ, 4 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 5 :=
sorry

end no_x_satisfies_inequalities_l84_84607


namespace jordans_greatest_average_speed_l84_84909

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s.reverse = s

theorem jordans_greatest_average_speed :
  ∃ (v : ℕ), 
  ∃ (d : ℕ), 
  ∃ (end_reading : ℕ), 
  is_palindrome 72327 ∧ 
  is_palindrome end_reading ∧ 
  72327 < end_reading ∧ 
  end_reading - 72327 = d ∧ 
  d ≤ 240 ∧ 
  end_reading ≤ 72327 + 240 ∧ 
  v = d / 4 ∧ 
  v = 50 :=
sorry

end jordans_greatest_average_speed_l84_84909


namespace interest_calculation_l84_84233

theorem interest_calculation :
  ∃ n : ℝ, 
  (1000 * 0.03 * n + 1400 * 0.05 * n = 350) →
  n = 3.5 := 
by 
  sorry

end interest_calculation_l84_84233


namespace solve_for_x_l84_84482

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l84_84482


namespace tickets_sold_correctly_l84_84761

theorem tickets_sold_correctly :
  let total := 620
  let cost_per_ticket := 4
  let tickets_sold := 155
  total / cost_per_ticket = tickets_sold :=
by
  sorry

end tickets_sold_correctly_l84_84761


namespace find_smaller_root_l84_84913

theorem find_smaller_root :
  ∀ x : ℝ, (x - 2 / 3) ^ 2 + (x - 2 / 3) * (x - 1 / 3) = 0 → x = 1 / 2 :=
by
  sorry

end find_smaller_root_l84_84913


namespace average_revenue_per_hour_l84_84516

theorem average_revenue_per_hour 
    (sold_A_hour1 : ℕ) (sold_B_hour1 : ℕ) (sold_A_hour2 : ℕ) (sold_B_hour2 : ℕ)
    (price_A_hour1 : ℕ) (price_A_hour2 : ℕ) (price_B_constant : ℕ) : 
    (sold_A_hour1 = 10) ∧ (sold_B_hour1 = 5) ∧ (sold_A_hour2 = 2) ∧ (sold_B_hour2 = 3) ∧
    (price_A_hour1 = 3) ∧ (price_A_hour2 = 4) ∧ (price_B_constant = 2) →
    (54 / 2 = 27) :=
by
  intros
  sorry

end average_revenue_per_hour_l84_84516


namespace arrange_consecutive_integers_no_common_divisors_l84_84013

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l84_84013


namespace area_of_highest_points_l84_84863

noncomputable def highest_point_area (u g : ℝ) : ℝ :=
  let x₁ := u^2 / (2 * g)
  let x₂ := 2 * u^2 / g
  (1/4) * ((x₂^2) - (x₁^2))

theorem area_of_highest_points (u g : ℝ) : highest_point_area u g = 3 * u^4 / (4 * g^2) :=
by
  sorry

end area_of_highest_points_l84_84863


namespace trucks_have_160_containers_per_truck_l84_84736

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l84_84736


namespace sum_of_ages_is_14_l84_84987

/-- Kiana has two older twin brothers and the product of their three ages is 72.
    Prove that the sum of their three ages is 14. -/
theorem sum_of_ages_is_14 (kiana_age twin_age : ℕ) (htwins : twin_age > kiana_age) (h_product : kiana_age * twin_age * twin_age = 72) :
  kiana_age + twin_age + twin_age = 14 :=
sorry

end sum_of_ages_is_14_l84_84987


namespace exists_infinite_N_l84_84368

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l84_84368


namespace find_x_l84_84487

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l84_84487


namespace spinner_final_direction_north_l84_84841

def start_direction := "north"
def clockwise_revolutions := (7 : ℚ) / 2
def counterclockwise_revolutions := (5 : ℚ) / 2
def net_revolutions := clockwise_revolutions - counterclockwise_revolutions

theorem spinner_final_direction_north :
  net_revolutions = 1 → start_direction = "north" → 
  start_direction = "north" :=
by
  intro h1 h2
  -- Here you would prove that net_revolutions of 1 full cycle leads back to start
  exact h2 -- Skipping proof

end spinner_final_direction_north_l84_84841


namespace sum_of_consecutive_integers_product_336_l84_84711

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l84_84711


namespace binary_to_base5_conversion_l84_84281

theorem binary_to_base5_conversion : ∀ (b : ℕ), b = 1101 → (13 : ℕ) % 5 = 3 ∧ (13 / 5) % 5 = 2 → b = 1101 → (1101 : ℕ) = 13 → 13 = 23 :=
by
  sorry

end binary_to_base5_conversion_l84_84281


namespace speed_of_slower_train_is_36_l84_84070

-- Definitions used in the conditions
def length_of_train := 25 -- meters
def combined_length_of_trains := 2 * length_of_train -- meters
def time_to_pass := 18 -- seconds
def speed_of_faster_train := 46 -- km/hr
def conversion_factor := 1000 / 3600 -- to convert from km/hr to m/s

-- Prove that speed of the slower train is 36 km/hr
theorem speed_of_slower_train_is_36 :
  ∃ v : ℕ, v = 36 ∧ ((combined_length_of_trains : ℝ) = ((speed_of_faster_train - v) * conversion_factor * time_to_pass)) :=
sorry

end speed_of_slower_train_is_36_l84_84070


namespace price_per_liter_l84_84373

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end price_per_liter_l84_84373


namespace greatest_number_of_groups_l84_84246

theorem greatest_number_of_groups (s a t b n : ℕ) (hs : s = 10) (ha : a = 15) (ht : t = 12) (hb : b = 18) :
  (∀ n, n ≤ n ∧ n ∣ s ∧ n ∣ a ∧ n ∣ t ∧ n ∣ b ∧ n > 1 → 
  (s / n < (a / n) + (t / n) + (b / n))
  ∧ (∃ groups, groups = n)) → n = 3 :=
sorry

end greatest_number_of_groups_l84_84246


namespace solve_square_l84_84344

theorem solve_square:
  ∃ (square: ℚ), 
    ((13/5) - ((17/2) - square) / (7/2)) / (1 / ((61/20) + (89/20))) = 2 → 
    square = 1/3 :=
  sorry

end solve_square_l84_84344


namespace jeremie_friends_l84_84508

-- Define the costs as constants.
def ticket_cost : ℕ := 18
def snack_cost : ℕ := 5
def total_cost : ℕ := 92
def per_person_cost : ℕ := ticket_cost + snack_cost

-- Define the number of friends Jeremie is going with (to be solved/proven).
def number_of_friends (total_cost : ℕ) (per_person_cost : ℕ) : ℕ :=
  let total_people := total_cost / per_person_cost
  total_people - 1

-- The statement that we want to prove.
theorem jeremie_friends : number_of_friends total_cost per_person_cost = 3 := by
  sorry

end jeremie_friends_l84_84508


namespace vermont_clicked_ads_l84_84393

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end vermont_clicked_ads_l84_84393


namespace _l84_84313

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l84_84313


namespace merchant_gross_profit_l84_84896

theorem merchant_gross_profit :
  ∃ S : ℝ, (42 + 0.30 * S = S) ∧ ((0.80 * S) - 42 = 6) :=
by
  sorry

end merchant_gross_profit_l84_84896


namespace common_divisors_9240_8820_l84_84962

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l84_84962


namespace kitten_food_consumption_l84_84041

-- Definitions of the given conditions
def k : ℕ := 4  -- Number of kittens
def ac : ℕ := 3  -- Number of adult cats
def f : ℕ := 7  -- Initial cans of food
def af : ℕ := 35  -- Additional cans of food needed
def days : ℕ := 7  -- Total number of days

-- Definition of the food consumption per adult cat per day
def food_per_adult_cat_per_day : ℕ := 1

-- Definition of the correct answer: food per kitten per day
def food_per_kitten_per_day : ℚ := 0.75

-- Proof statement
theorem kitten_food_consumption (k : ℕ) (ac : ℕ) (f : ℕ) (af : ℕ) (days : ℕ) (food_per_adult_cat_per_day : ℕ) :
  (ac * food_per_adult_cat_per_day * days + k * food_per_kitten_per_day * days = f + af) → 
  food_per_kitten_per_day = 0.75 :=
sorry

end kitten_food_consumption_l84_84041


namespace sum_of_reciprocals_l84_84159

variable {x y : ℝ}

theorem sum_of_reciprocals (h1 : x + y = 4 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x + 1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l84_84159


namespace cubic_no_maximum_value_l84_84558

theorem cubic_no_maximum_value (x : ℝ) : ¬ ∃ M, ∀ x : ℝ, 3 * x^2 + 6 * x^3 + 27 * x + 100 ≤ M := 
by
  sorry

end cubic_no_maximum_value_l84_84558


namespace polynomial_coeff_sum_eq_four_l84_84967

theorem polynomial_coeff_sum_eq_four (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) :
  (∀ x : ℤ, (2 * x - 1)^6 * (x + 1)^2 = a * x ^ 8 + a1 * x ^ 7 + a2 * x ^ 6 + a3 * x ^ 5 + 
                      a4 * x ^ 4 + a5 * x ^ 3 + a6 * x ^ 2 + a7 * x + a8) →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 := by
  sorry

end polynomial_coeff_sum_eq_four_l84_84967


namespace quadratic_real_roots_range_l84_84652

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end quadratic_real_roots_range_l84_84652


namespace option_c_is_always_odd_l84_84824

theorem option_c_is_always_odd (n : ℤ) : ∃ (q : ℤ), n^2 + n + 5 = 2*q + 1 := by
  sorry

end option_c_is_always_odd_l84_84824


namespace sum_of_three_consecutive_integers_product_336_l84_84718

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l84_84718


namespace max_min_value_of_f_l84_84324

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end max_min_value_of_f_l84_84324


namespace harriet_travel_time_l84_84573

theorem harriet_travel_time (D : ℝ) (h : (D / 90 + D / 160 = 5)) : (D / 90) * 60 = 192 := 
by sorry

end harriet_travel_time_l84_84573


namespace percentage_increase_sale_l84_84758

theorem percentage_increase_sale (P S : ℝ) (hP : 0 < P) (hS : 0 < S) :
  let new_price := 0.65 * P
  let original_revenue := P * S
  let new_revenue := 1.17 * original_revenue
  let percentage_increase := 80 / 100
  let new_sales := S * (1 + percentage_increase)
  new_price * new_sales = new_revenue :=
by
  sorry

end percentage_increase_sale_l84_84758


namespace decimal_division_l84_84555

theorem decimal_division : (0.05 : ℝ) / (0.005 : ℝ) = 10 := 
by 
  sorry

end decimal_division_l84_84555


namespace parabola_equation_l84_84799

/--
Given a point P (4, -2) on a parabola, prove that the equation of the parabola is either:
1) y^2 = x or
2) x^2 = -8y.
-/
theorem parabola_equation (p : ℝ) (x y : ℝ) (h1 : (4 : ℝ) = 4) (h2 : (-2 : ℝ) = -2) :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ 4 = 4 ∧ y = -2) ∨ (∃ p : ℝ, x^2 = 2 * p * y ∧ 4 = 4 ∧ x = 4) :=
sorry

end parabola_equation_l84_84799


namespace trucks_have_160_containers_per_truck_l84_84735

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l84_84735


namespace number_of_regions_on_sphere_l84_84405

theorem number_of_regions_on_sphere (n : ℕ) (h : ∀ {a b c: ℤ}, a ≠ b → b ≠ c → a ≠ c → True) : 
  ∃ a_n, a_n = n^2 - n + 2 := 
by
  sorry

end number_of_regions_on_sphere_l84_84405


namespace shaded_region_area_l84_84198

theorem shaded_region_area (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) (T : ℝ):
  d = 3 → L = 24 → L = n * d → n * 2 = 16 → r = d / 2 → 
  A = (1 / 2) * π * r ^ 2 → T = 16 * A → T = 18 * π :=
  by
  intros d_eq L_eq Ln_eq semicircle_count r_eq A_eq T_eq_total
  sorry

end shaded_region_area_l84_84198


namespace directrix_of_parabola_l84_84138

-- Define the given condition:
def parabola_eq (x : ℝ) : ℝ := 8 * x^2 + 4 * x + 2

-- State the theorem:
theorem directrix_of_parabola :
  (∀ x : ℝ, parabola_eq x = 8 * (x + 1/4)^2 + 1) → (y = 31 / 32) :=
by
  -- We'll prove this later
  sorry

end directrix_of_parabola_l84_84138


namespace andrew_ruined_planks_l84_84423

variable (b L k g h leftover plank_total ruin_bedroom ruin_guest : ℕ)

-- Conditions
def bedroom_planks := b
def living_room_planks := L
def kitchen_planks := k
def guest_bedroom_planks := g
def hallway_planks := h
def planks_leftover := leftover

-- Values
axiom bedroom_planks_val : bedroom_planks = 8
axiom living_room_planks_val : living_room_planks = 20
axiom kitchen_planks_val : kitchen_planks = 11
axiom guest_bedroom_planks_val : guest_bedroom_planks = bedroom_planks - 2
axiom hallway_planks_val : hallway_planks = 4
axiom planks_leftover_val : planks_leftover = 6

-- Total planks used and total planks had
def total_planks_used := bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + (2 * hallway_planks)
def total_planks_had := total_planks_used + planks_leftover

-- Planks ruined
def planks_ruined_in_bedroom := ruin_bedroom
def planks_ruined_in_guest_bedroom := ruin_guest

-- Theorem to be proven
theorem andrew_ruined_planks :
  (planks_ruined_in_bedroom = total_planks_had - total_planks_used) ∧
  (planks_ruined_in_guest_bedroom = planks_ruined_in_bedroom) :=
by
  sorry

end andrew_ruined_planks_l84_84423


namespace numeral_diff_local_face_value_l84_84559

theorem numeral_diff_local_face_value (P : ℕ) :
  7 * (10 ^ P - 1) = 693 → P = 2 ∧ (N = 700) :=
by
  intro h
  -- The actual proof is not required hence we insert sorry
  sorry

end numeral_diff_local_face_value_l84_84559


namespace logically_follows_l84_84522

-- Define the predicates P and Q
variables {Student : Type} {P Q : Student → Prop}

-- The given condition
axiom Turner_statement : ∀ (x : Student), P x → Q x

-- The statement that necessarily follows
theorem logically_follows : (∀ (x : Student), ¬ Q x → ¬ P x) :=
sorry

end logically_follows_l84_84522


namespace tan_double_angle_l84_84613

theorem tan_double_angle (α : Real) (h1 : Real.sin α - Real.cos α = 4 / 3) (h2 : α ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4)) :
  Real.tan (2 * α) = (7 * Real.sqrt 2) / 8 :=
by
  sorry

end tan_double_angle_l84_84613


namespace distance_from_point_A_to_directrix_C_l84_84316

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l84_84316


namespace part1_part2_l84_84454

variable {f : ℝ → ℝ}

theorem part1 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : f 1 = 0 :=
by sorry

theorem part2 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f (-x) + f (3 - x) ≥ 2 :=
by sorry

end part1_part2_l84_84454


namespace lcm_18_27_l84_84078

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end lcm_18_27_l84_84078


namespace y_coord_equidistant_l84_84075

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end y_coord_equidistant_l84_84075


namespace constant_term_in_expansion_l84_84289

theorem constant_term_in_expansion (x : ℂ) : 
  (2 - (3 / x)) * (x ^ 2 + 2 / x) ^ 5 = 0 := 
sorry

end constant_term_in_expansion_l84_84289


namespace distance_to_directrix_l84_84317

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l84_84317


namespace regression_value_l84_84940

theorem regression_value (x : ℝ) (y : ℝ) (h : y = 4.75 * x + 2.57) (hx : x = 28) : y = 135.57 :=
by
  sorry

end regression_value_l84_84940


namespace fourth_number_of_expression_l84_84244

theorem fourth_number_of_expression (x : ℝ) (h : 0.3 * 0.8 + 0.1 * x = 0.29) : x = 0.5 :=
by
  sorry

end fourth_number_of_expression_l84_84244


namespace range_of_a_l84_84575

open Real

theorem range_of_a (a : ℝ) :
  (∀ x, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧ ¬ (∀ x, (x + 2) * (x + a) < 0 → |x - 1| < 3) →
  a < -4 :=
by
  sorry

end range_of_a_l84_84575


namespace increasing_interval_minimum_value_in_interval_l84_84631

noncomputable def f (x : ℝ) : ℝ :=
  1 * sin x + (-sqrt 3 * sin (x / 2)) * (2 * sin (x / 2)) + sqrt 3

theorem increasing_interval (k : ℤ) : 
  ∃ (a b : ℝ), [2 * Real.pi * k - (5 * Real.pi / 6), 2 * Real.pi * k + (Real.pi / 6)] = Icc a b := sorry

theorem minimum_value_in_interval :
  ∀ x ∈ Icc (0 : ℝ) (2 * Real.pi / 3), f x ≥ 0 := sorry

end increasing_interval_minimum_value_in_interval_l84_84631


namespace mixed_number_subtraction_l84_84755

theorem mixed_number_subtraction :
  2 + 5 / 6 - (1 + 1 / 3) = 3 / 2 := by
sorry

end mixed_number_subtraction_l84_84755


namespace completing_square_correct_l84_84089

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l84_84089


namespace consecutive_integers_sum_l84_84704

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l84_84704


namespace price_of_pants_l84_84677

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end price_of_pants_l84_84677


namespace prime_condition_l84_84797

theorem prime_condition (p : ℕ) (h_prime: Nat.Prime p) :
  (∃ m n : ℤ, p = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) ↔ p = 2 ∨ p = 5 :=
by
  sorry

end prime_condition_l84_84797


namespace part1_part2_part3_l84_84867

noncomputable def sequence_a : ℕ → ℝ
| 0       := 1/2
| (n + 1) := sequence_a n / (1 + sequence_a n)^2

noncomputable def sequence_b (n : ℕ) : ℝ := 1 / sequence_a n

theorem part1 (n : ℕ) (hn : 1 < n) : sequence_b n > 2 * n := sorry

theorem part2 : tendsto (λ n, (1 : ℝ) / n * ∑ i in finset.range n, sequence_a (i + 1)) at_top (𝓝 0) := sorry

theorem part3 : tendsto (λ n, n * sequence_a n) at_top (𝓝 (1 / 2)) := sorry

end part1_part2_part3_l84_84867


namespace cost_of_chlorine_l84_84941

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end cost_of_chlorine_l84_84941


namespace expected_length_after_2012_repetitions_l84_84880

noncomputable def expected_length_remaining (n : ℕ) := (11/18 : ℚ)^n

theorem expected_length_after_2012_repetitions :
  expected_length_remaining 2012 = (11 / 18 : ℚ) ^ 2012 :=
by
  sorry

end expected_length_after_2012_repetitions_l84_84880


namespace dodecahedron_edge_probability_l84_84747

theorem dodecahedron_edge_probability :
  ∀ (V E : ℕ), 
  V = 20 → 
  ((∀ v ∈ finset.range V, 3 = 3) → -- condition representing each of the 20 vertices is connected to 3 other vertices
  ∃ (p : ℚ), p = 3 / 19) :=
begin
  intros,
  use 3 / 19,
  split,
  sorry
end

end dodecahedron_edge_probability_l84_84747


namespace smallest_integer_greater_than_power_l84_84231

theorem smallest_integer_greater_than_power (sqrt3 sqrt2 : ℝ) (h1 : (sqrt3 + sqrt2)^6 = 485 + 198 * Real.sqrt 6)
(h2 : (sqrt3 - sqrt2)^6 = 485 - 198 * Real.sqrt 6)
(h3 : 0 < (sqrt3 - sqrt2)^6 ∧ (sqrt3 - sqrt2)^6 < 1) : 
  ⌈(sqrt3 + sqrt2)^6⌉ = 970 := 
sorry

end smallest_integer_greater_than_power_l84_84231


namespace cassie_nails_l84_84276

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l84_84276


namespace sequence_value_2_l84_84977

/-- 
Given the following sequence:
1 = 6
3 = 18
4 = 24
5 = 30

The sequence follows the pattern that for all n ≠ 6, n is mapped to n * 6.
Prove that the value of the 2nd term in the sequence is 12.
-/

theorem sequence_value_2 (a : ℕ → ℕ) 
  (h1 : a 1 = 6) 
  (h3 : a 3 = 18) 
  (h4 : a 4 = 24) 
  (h5 : a 5 = 30) 
  (h_pattern : ∀ n, n ≠ 6 → a n = n * 6) :
  a 2 = 12 :=
by
  sorry

end sequence_value_2_l84_84977


namespace line_tangent_to_parabola_l84_84156

theorem line_tangent_to_parabola (k : ℝ) (x₀ y₀ : ℝ) 
  (h₁ : y₀ = k * x₀ - 2) 
  (h₂ : x₀^2 = 4 * y₀) 
  (h₃ : ∀ x y, (x = x₀ ∧ y = y₀) → (k = (1/2) * x₀)) :
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := 
sorry

end line_tangent_to_parabola_l84_84156


namespace limit_calculation_l84_84593

open real

theorem limit_calculation :
  tendsto (λ n: ℕ, (sqrt (↑n ^ 5 - 8) - ↑n * sqrt (↑n * (↑n ^ 2 + 5))) / sqrt (↑n)) at_top (𝓝 (-5 / 2)) :=
sorry

end limit_calculation_l84_84593


namespace expr_B_not_simplified_using_difference_of_squares_l84_84567

def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

theorem expr_B_not_simplified_using_difference_of_squares (x y : ℝ) :
  ∃ x y, ¬ ∃ a b, expr_B x y = a^2 - b^2 :=
sorry

end expr_B_not_simplified_using_difference_of_squares_l84_84567


namespace number_of_special_four_digit_integers_l84_84639

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l84_84639


namespace geometric_mean_a_b_l84_84951

theorem geometric_mean_a_b : ∀ (a b : ℝ), a > 0 → b > 0 → Real.sqrt 3 = Real.sqrt (3^a * 3^b) → a + b = 1 :=
by
  intros a b ha hb hgeo
  sorry

end geometric_mean_a_b_l84_84951


namespace completing_the_square_result_l84_84086

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l84_84086


namespace trapezoid_PR_length_l84_84551

noncomputable def PR_length (PQ RS QS PR : ℝ) (angle_QSP angle_SRP : ℝ) : Prop :=
  PQ < RS ∧ 
  QS = 2 ∧ 
  angle_QSP = 30 ∧ 
  angle_SRP = 60 ∧ 
  RS / PQ = 7 / 3 ∧ 
  PR = 8 / 3

theorem trapezoid_PR_length (PQ RS QS PR : ℝ) 
  (angle_QSP angle_SRP : ℝ) 
  (h1 : PQ < RS) 
  (h2 : QS = 2) 
  (h3 : angle_QSP = 30) 
  (h4 : angle_SRP = 60) 
  (h5 : RS / PQ = 7 / 3) :
  PR = 8 / 3 := 
by
  sorry

end trapezoid_PR_length_l84_84551


namespace amount_sharpened_off_l84_84346

-- Defining the initial length of the pencil
def initial_length : ℕ := 31

-- Defining the length of the pencil after sharpening
def after_sharpening_length : ℕ := 14

-- Proving the amount sharpened off the pencil
theorem amount_sharpened_off : initial_length - after_sharpening_length = 17 := 
by 
  -- Here we would insert the proof steps, 
  -- but as instructed we leave it as sorry.
  sorry

end amount_sharpened_off_l84_84346


namespace determine_ratio_l84_84112

-- Definition of the given conditions.
def total_length : ℕ := 69
def longer_length : ℕ := 46
def ratio_of_lengths (shorter_length longer_length : ℕ) : ℕ := longer_length / shorter_length

-- The theorem we need to prove.
theorem determine_ratio (x : ℕ) (m : ℕ) (h1 : longer_length = m * x) (h2 : x + longer_length = total_length) : 
  ratio_of_lengths x longer_length = 2 :=
by
  sorry

end determine_ratio_l84_84112


namespace rectangle_lines_combinations_l84_84444

theorem rectangle_lines_combinations : 5.choose 2 * 4.choose 2 = 60 := by
  sorry

end rectangle_lines_combinations_l84_84444


namespace exhibition_adult_child_ratio_l84_84048

theorem exhibition_adult_child_ratio (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 1 := by
  -- Prove the result
  sorry

end exhibition_adult_child_ratio_l84_84048


namespace max_value_of_squares_l84_84029

theorem max_value_of_squares (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  a^2 + b^2 + c^2 + d^2 ≤ 4 :=
sorry

end max_value_of_squares_l84_84029


namespace fraction_zero_l84_84563

theorem fraction_zero (x : ℝ) (h : (x^2 - 1) / (x + 1) = 0) : x = 1 := 
sorry

end fraction_zero_l84_84563


namespace grid_satisfies_conditions_l84_84018

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l84_84018


namespace choose_4_from_15_l84_84501

theorem choose_4_from_15 : (Nat.choose 15 4) = 1365 :=
by
  sorry

end choose_4_from_15_l84_84501


namespace remainder_product_mod_5_l84_84729

theorem remainder_product_mod_5 : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end remainder_product_mod_5_l84_84729


namespace sum_even_integers_eq_930_l84_84387

theorem sum_even_integers_eq_930 :
  let sum_first_30_even := 2 * (30 * (30 + 1) / 2)
  let sum_consecutive_even (n : ℤ) := (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∀ n : ℤ, sum_first_30_even = 930 → sum_consecutive_even n = 930 → n = 190 :=
by
  intros sum_first_30_even sum_consecutive_even n h1 h2
  sorry

end sum_even_integers_eq_930_l84_84387


namespace find_power_y_l84_84938

theorem find_power_y 
  (y : ℕ) 
  (h : (12 : ℝ)^y * (6 : ℝ)^3 / (432 : ℝ) = 72) : 
  y = 2 :=
by
  sorry

end find_power_y_l84_84938


namespace rides_total_l84_84126

theorem rides_total (rides_day1 rides_day2 : ℕ) (h1 : rides_day1 = 4) (h2 : rides_day2 = 3) : rides_day1 + rides_day2 = 7 := 
by 
  sorry

end rides_total_l84_84126


namespace savings_increase_l84_84235

variable (I : ℝ) -- Initial income
variable (E : ℝ) -- Initial expenditure
variable (S : ℝ) -- Initial savings
variable (I_new : ℝ) -- New income
variable (E_new : ℝ) -- New expenditure
variable (S_new : ℝ) -- New savings

theorem savings_increase (h1 : E = 0.75 * I) 
                         (h2 : I_new = 1.20 * I) 
                         (h3 : E_new = 1.10 * E) : 
                         (S_new - S) / S * 100 = 50 :=
by 
  have h4 : S = 0.25 * I := by sorry
  have h5 : E_new = 0.825 * I := by sorry
  have h6 : S_new = 0.375 * I := by sorry
  have increase : (S_new - S) / S * 100 = 50 := by sorry
  exact increase

end savings_increase_l84_84235


namespace age_of_b_l84_84570

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_of_b_l84_84570


namespace eight_p_plus_one_is_composite_l84_84037

theorem eight_p_plus_one_is_composite (p : ℕ) (hp : Nat.Prime p) (h8p1 : Nat.Prime (8 * p - 1)) : ¬ Nat.Prime (8 * p + 1) :=
by
  sorry

end eight_p_plus_one_is_composite_l84_84037


namespace quadratic_one_solution_set_l84_84175

theorem quadratic_one_solution_set (a : ℝ) :
  (∃ x : ℝ, ax^2 + x + 1 = 0 ∧ (∀ y : ℝ, ax^2 + x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1 / 4) :=
by sorry

end quadratic_one_solution_set_l84_84175


namespace find_principal_l84_84586

variable (R P : ℝ)
variable (h1 : ∀ (R P : ℝ), (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400)

theorem find_principal (h1 : ∀ (R P : ℝ), (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400) :
  P = 800 := 
sorry

end find_principal_l84_84586


namespace lightning_distance_l84_84286

/--
Linus observed a flash of lightning and then heard the thunder 15 seconds later.
Given:
- speed of sound: 1088 feet/second
- 1 mile = 5280 feet
Prove that the distance from Linus to the lightning strike is 3.25 miles.
-/
theorem lightning_distance (time_seconds : ℕ) (speed_sound : ℕ) (feet_per_mile : ℕ) (distance_miles : ℚ) :
  time_seconds = 15 →
  speed_sound = 1088 →
  feet_per_mile = 5280 →
  distance_miles = 3.25 :=
by
  sorry

end lightning_distance_l84_84286


namespace part1_part2_l84_84163

def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem part1 (h : m = 2) : P ∪ S m = {x | -2 < x ∧ x ≤ 3} :=
  by sorry

theorem part2 (h : ∀ x, x ∈ S m → x ∈ P) : 0 ≤ m ∧ m ≤ 1 :=
  by sorry

end part1_part2_l84_84163


namespace family_of_four_children_includes_one_boy_one_girl_l84_84260

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l84_84260


namespace number_of_sevens_l84_84238

theorem number_of_sevens (n : ℕ) : ∃ (k : ℕ), k < n ∧ ∃ (f : ℕ → ℕ), (∀ i, f i = 7) ∧ (7 * ((77 - 7) / 7) ^ 14 - 1) / (7 + (7 + 7)/7) = 7^(f k) :=
by sorry

end number_of_sevens_l84_84238


namespace remainder_43_pow_43_plus_43_mod_44_l84_84227

theorem remainder_43_pow_43_plus_43_mod_44 :
  let n := 43
  let m := 44
  (n^43 + n) % m = 42 :=
by 
  let n := 43
  let m := 44
  sorry

end remainder_43_pow_43_plus_43_mod_44_l84_84227


namespace integer_solution_count_l84_84857

theorem integer_solution_count {a b c d : ℤ} (h : a ≠ b) :
  (∀ x y : ℤ, (x + a * y + c) * (x + b * y + d) = 2 →
    ∃ a b : ℤ, (|a - b| = 1 ∨ (|a - b| = 2 ∧ (d - c) % 2 = 1))) :=
sorry

end integer_solution_count_l84_84857


namespace division_of_cubics_l84_84072

theorem division_of_cubics (a b c : ℕ) (h_a : a = 7) (h_b : b = 6) (h_c : c = 1) :
  (a^3 + b^3) / (a^2 - a * b + b^2 + c) = 559 / 44 :=
by
  rw [h_a, h_b, h_c]
  -- After these substitutions, the problem is reduced to proving
  -- (7^3 + 6^3) / (7^2 - 7 * 6 + 6^2 + 1) = 559 / 44
  sorry

end division_of_cubics_l84_84072


namespace find_cost_price_l84_84051

theorem find_cost_price (SP : ℝ) (loss_percent : ℝ) (CP : ℝ) (h1 : SP = 1260) (h2 : loss_percent = 16) : CP = 1500 :=
by
  sorry

end find_cost_price_l84_84051


namespace grid_is_valid_l84_84011

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l84_84011


namespace arithmetic_sequence_a4_l84_84945

theorem arithmetic_sequence_a4 (S n : ℕ) (a : ℕ → ℕ) (h1 : S = 28) (h2 : S = 7 * a 4) : a 4 = 4 :=
by sorry

end arithmetic_sequence_a4_l84_84945


namespace inverse_proportion_indeterminate_l84_84814

theorem inverse_proportion_indeterminate (k : ℝ) (x1 x2 y1 y2 : ℝ) (h1 : x1 < x2)
  (h2 : y1 = k / x1) (h3 : y2 = k / x2) : 
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0) ∨ (y1 * y2 < 0) → false :=
sorry

end inverse_proportion_indeterminate_l84_84814


namespace fraction_to_decimal_l84_84282

theorem fraction_to_decimal : (5 / 50) = 0.10 := 
by
  sorry

end fraction_to_decimal_l84_84282


namespace geometric_sequence_arithmetic_median_l84_84463

theorem geometric_sequence_arithmetic_median 
  (a : ℕ → ℝ) 
  (hpos : ∀ n, 0 < a n) 
  (h_arith : 2 * a 1 + a 2 = 2 * a 3) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 :=
sorry

end geometric_sequence_arithmetic_median_l84_84463


namespace eight_percent_is_64_l84_84918

-- Definition of the condition
variable (x : ℝ)

-- The theorem that states the problem to be proven
theorem eight_percent_is_64 (h : (8 / 100) * x = 64) : x = 800 :=
sorry

end eight_percent_is_64_l84_84918


namespace N_prime_iff_k_eq_2_l84_84145

/-- Define the number N for a given k -/
def N (k : ℕ) : ℕ := (10 ^ (2 * k) - 1) / 99

/-- Statement: Prove that N is prime if and only if k = 2 -/
theorem N_prime_iff_k_eq_2 (k : ℕ) : Prime (N k) ↔ k = 2 := by
  sorry

end N_prime_iff_k_eq_2_l84_84145


namespace profit_percentage_example_l84_84569

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℕ) (sp_total : ℝ) (sp_count : ℕ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

theorem profit_percentage_example : profit_percentage 25 15 33 12 = 65 :=
by
  sorry

end profit_percentage_example_l84_84569


namespace min_value_of_function_l84_84615

theorem min_value_of_function (x : ℝ) (h : x > 5 / 4) : 
  ∃ ymin : ℝ, ymin = 7 ∧ ∀ y : ℝ, y = 4 * x + 1 / (4 * x - 5) → y ≥ ymin := 
sorry

end min_value_of_function_l84_84615


namespace problem_a_correct_answer_l84_84234

def initial_digit_eq_six (n : ℕ) : Prop :=
∃ k a : ℕ, n = 6 * 10^k + a ∧ a = n / 25

theorem problem_a_correct_answer :
  ∀ n : ℕ, initial_digit_eq_six n ↔ ∃ m : ℕ, n = 625 * 10^m :=
by
  sorry

end problem_a_correct_answer_l84_84234


namespace value_of_y_l84_84648

theorem value_of_y (y : ℕ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
-- Since we are only required to state the theorem, we leave the proof out for now.
sorry

end value_of_y_l84_84648


namespace max_sum_first_n_terms_is_S_5_l84_84182

open Nat

-- Define the arithmetic sequence and the conditions.
variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n}
variable {d : ℝ} -- The common difference of the arithmetic sequence
variable {S : ℕ → ℝ} -- The sum of the first n terms of the sequence a

-- Hypotheses corresponding to the conditions in the problem
lemma a_5_positive : a 5 > 0 := sorry
lemma a_4_plus_a_7_negative : a 4 + a 7 < 0 := sorry

-- Statement to prove that the maximum value of the sum of the first n terms is S_5 given the conditions
theorem max_sum_first_n_terms_is_S_5 :
  (∀ (n : ℕ), S n ≤ S 5) :=
sorry

end max_sum_first_n_terms_is_S_5_l84_84182


namespace value_of_x_l84_84476

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l84_84476


namespace inequality_holds_iff_m_range_l84_84801

theorem inequality_holds_iff_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ (-3 < m ∧ m ≤ 0) :=
by
  sorry

end inequality_holds_iff_m_range_l84_84801


namespace distance_to_water_source_l84_84419

theorem distance_to_water_source (d : ℝ) :
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 5)) → 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_water_source_l84_84419


namespace tan_sum_pi_div_12_l84_84859

theorem tan_sum_pi_div_12 (h1 : Real.tan (Real.pi / 12) ≠ 0) (h2 : Real.tan (5 * Real.pi / 12) ≠ 0) :
  Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 := 
by
  sorry

end tan_sum_pi_div_12_l84_84859


namespace range_of_a_for_three_distinct_real_roots_l84_84466

theorem range_of_a_for_three_distinct_real_roots (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x, f x = x^3 - 3*x^2 - a ∧ ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end range_of_a_for_three_distinct_real_roots_l84_84466


namespace part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l84_84998

-- Definitions for the sequences
def first_row (n : ℕ) : ℤ := (-3) ^ n
def second_row (n : ℕ) : ℤ := (-3) ^ n - 3
def third_row (n : ℕ) : ℤ := -((-3) ^ n) - 1

-- Statement for part 1
theorem part1_fifth_numbers:
  first_row 5 = -243 ∧ second_row 5 = -246 ∧ third_row 5 = 242 := sorry

-- Statement for part 2
theorem part2_three_adjacent_sum :
  ∃ n : ℕ, first_row (n-1) + first_row n + first_row (n+1) = -1701 ∧
           first_row (n-1) = -243 ∧ first_row n = 729 ∧ first_row (n+1) = -2187 := sorry

-- Statement for part 3
def sum_nth (n : ℕ) : ℤ := first_row n + second_row n + third_row n
theorem part3_difference_largest_smallest (n : ℕ) (m : ℤ) (hn : sum_nth n = m) :
  (∃ diff, (n % 2 = 1 → diff = -2 * m - 6) ∧ (n % 2 = 0 → diff = 2 * m + 9)) := sorry

end part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l84_84998


namespace total_snails_and_frogs_l84_84389

-- Define the number of snails and frogs in the conditions.
def snails : Nat := 5
def frogs : Nat := 2

-- State the problem: proving that the total number of snails and frogs equals 7.
theorem total_snails_and_frogs : snails + frogs = 7 := by
  -- Proof is omitted as the user requested only the statement.
  sorry

end total_snails_and_frogs_l84_84389


namespace determine_M_l84_84434

theorem determine_M : ∃ M : ℕ, 36^2 * 75^2 = 30^2 * M^2 ∧ M = 90 := 
by
  sorry

end determine_M_l84_84434


namespace divisibility_of_expression_l84_84363

open Int

theorem divisibility_of_expression (a b : ℤ) (ha : Prime a) (hb : Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) :=
sorry

end divisibility_of_expression_l84_84363


namespace compute_sixth_power_sum_l84_84514

theorem compute_sixth_power_sum (ζ1 ζ2 ζ3 : ℂ) 
  (h1 : ζ1 + ζ2 + ζ3 = 2)
  (h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5)
  (h3 : ζ1^4 + ζ2^4 + ζ3^4 = 29) :
  ζ1^6 + ζ2^6 + ζ3^6 = 101.40625 := 
by
  sorry

end compute_sixth_power_sum_l84_84514


namespace natural_numbers_solution_l84_84433

theorem natural_numbers_solution :
  ∃ (a b c d : ℕ), 
    ab = c + d ∧ a + b = cd ∧
    ((a, b, c, d) = (2, 2, 2, 2) ∨ (a, b, c, d) = (2, 3, 5, 1) ∨ 
     (a, b, c, d) = (3, 2, 5, 1) ∨ (a, b, c, d) = (2, 2, 1, 5) ∨ 
     (a, b, c, d) = (3, 2, 1, 5) ∨ (a, b, c, d) = (2, 3, 1, 5)) :=
by
  sorry

end natural_numbers_solution_l84_84433


namespace geometric_sum_eight_terms_l84_84865

theorem geometric_sum_eight_terms (a_1 : ℕ) (S_4 : ℕ) (r : ℕ) (S_8 : ℕ) 
    (h1 : r = 2) (h2 : S_4 = a_1 * (1 + r + r^2 + r^3)) (h3 : S_4 = 30) :
    S_8 = a_1 * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) → S_8 = 510 := 
by sorry

end geometric_sum_eight_terms_l84_84865


namespace sufficient_cond_l84_84300

theorem sufficient_cond (x : ℝ) (h : 1/x > 2) : x < 1/2 := 
by {
  sorry 
}

end sufficient_cond_l84_84300


namespace division_identity_l84_84574

theorem division_identity :
  (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 :=
by
  -- TODO: Provide the proof here
  sorry

end division_identity_l84_84574


namespace conditions_for_k_b_l84_84622

theorem conditions_for_k_b (k b : ℝ) :
  (∀ x : ℝ, (x - (kx + b) + 2) * (2) > 0) →
  (k = 1) ∧ (b < 2) :=
by
  intros h
  sorry

end conditions_for_k_b_l84_84622


namespace pin_probability_l84_84754

theorem pin_probability :
  let total_pins := 9 * 10^5
  let valid_pins := 10^4
  ∃ p : ℚ, p = valid_pins / total_pins ∧ p = 1 / 90 := by
  sorry

end pin_probability_l84_84754


namespace solution_set_of_quadratic_inequality_l84_84293

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) := 
sorry

end solution_set_of_quadratic_inequality_l84_84293


namespace shaded_region_area_l84_84199

theorem shaded_region_area (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) (T : ℝ):
  d = 3 → L = 24 → L = n * d → n * 2 = 16 → r = d / 2 → 
  A = (1 / 2) * π * r ^ 2 → T = 16 * A → T = 18 * π :=
  by
  intros d_eq L_eq Ln_eq semicircle_count r_eq A_eq T_eq_total
  sorry

end shaded_region_area_l84_84199


namespace sequence_an_properties_l84_84807

theorem sequence_an_properties
(S : ℕ → ℝ) (a : ℕ → ℝ)
(h_mean : ∀ n, 2 * a n = S n + 2) :
a 1 = 2 ∧ a 2 = 4 ∧ ∀ n, a n = 2 ^ n :=
by
  sorry

end sequence_an_properties_l84_84807


namespace ordering_of_powers_l84_84076

theorem ordering_of_powers :
  (3:ℕ)^15 < 10^9 ∧ 10^9 < (5:ℕ)^13 :=
by
  sorry

end ordering_of_powers_l84_84076


namespace distance_between_fourth_and_work_l84_84676

theorem distance_between_fourth_and_work (x : ℝ) (h₁ : x > 0) :
  let total_distance := x + 0.5 * x + 2 * x
  let to_fourth := (1 / 3) * total_distance
  let total_to_fourth := total_distance + to_fourth
  3 * total_to_fourth = 14 * x :=
by
  sorry

end distance_between_fourth_and_work_l84_84676


namespace max_value_of_function_neg_x_l84_84383

theorem max_value_of_function_neg_x (x : ℝ) (h : x < 0) : 
  ∃ y, (y = 2 * x + 2 / x) ∧ y ≤ -4 := sorry

end max_value_of_function_neg_x_l84_84383


namespace cost_of_sandwiches_and_smoothies_l84_84034

-- Define the cost of sandwiches and smoothies
def sandwich_cost := 4
def smoothie_cost := 3

-- Define the discount applicable
def sandwich_discount := 1
def total_sandwiches := 6
def total_smoothies := 7

-- Calculate the effective cost per sandwich considering discount
def effective_sandwich_cost := if total_sandwiches > 4 then sandwich_cost - sandwich_discount else sandwich_cost

-- Calculate the total cost for sandwiches
def sandwiches_cost := total_sandwiches * effective_sandwich_cost

-- Calculate the total cost for smoothies
def smoothies_cost := total_smoothies * smoothie_cost

-- Calculate the total cost
def total_cost := sandwiches_cost + smoothies_cost

-- The main statement to prove
theorem cost_of_sandwiches_and_smoothies : total_cost = 39 := by
  -- skip the proof
  sorry

end cost_of_sandwiches_and_smoothies_l84_84034


namespace intersection_A_B_l84_84820

def A : Set ℝ := {x | abs x < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l84_84820


namespace price_of_one_liter_l84_84375

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end price_of_one_liter_l84_84375


namespace minimum_pencils_l84_84105

-- Define the given conditions
def red_pencils : ℕ := 15
def blue_pencils : ℕ := 13
def green_pencils : ℕ := 8

-- Define the requirement for pencils to ensure the conditions are met
def required_red : ℕ := 1
def required_blue : ℕ := 2
def required_green : ℕ := 3

-- The minimum number of pencils Constanza should take out
noncomputable def minimum_pencils_to_ensure : ℕ := 21 + 1

theorem minimum_pencils (red_pencils blue_pencils green_pencils : ℕ)
    (required_red required_blue required_green minimum_pencils_to_ensure : ℕ) :
    red_pencils = 15 →
    blue_pencils = 13 →
    green_pencils = 8 →
    required_red = 1 →
    required_blue = 2 →
    required_green = 3 →
    minimum_pencils_to_ensure = 22 :=
by
    intros h1 h2 h3 h4 h5 h6
    sorry

end minimum_pencils_l84_84105


namespace cassie_nails_l84_84277

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l84_84277


namespace latest_time_temperature_84_l84_84178

noncomputable def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem latest_time_temperature_84 :
  ∃ t_max : ℝ, temperature t_max = 84 ∧ ∀ t : ℝ, temperature t = 84 → t ≤ t_max ∧ t_max = 11 :=
by
  sorry

end latest_time_temperature_84_l84_84178


namespace intersection_of_A_and_B_l84_84950

def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := 
by sorry

end intersection_of_A_and_B_l84_84950


namespace find_angle_C_find_sum_a_b_l84_84336

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = 7 / 2 ∧
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 ∧
  (Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1))

theorem find_angle_C (a b c A B C : ℝ) (h : triangle_condition a b c A B C) : C = Real.pi / 3 :=
  sorry

theorem find_sum_a_b (a b c A B C : ℝ) (h : triangle_condition a b c A B C) (hC : C = Real.pi / 3) : a + b = 11 / 2 :=
  sorry

end find_angle_C_find_sum_a_b_l84_84336


namespace completing_square_correct_l84_84088

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l84_84088


namespace arithmetic_sequence_sum_l84_84188

theorem arithmetic_sequence_sum (a1 d : ℝ)
  (h1 : a1 + 11 * d = -8)
  (h2 : 9 / 2 * (a1 + (a1 + 8 * d)) = -9) :
  16 / 2 * (a1 + (a1 + 15 * d)) = -72 := by
  sorry

end arithmetic_sequence_sum_l84_84188


namespace fraction_of_rotten_fruits_l84_84242

theorem fraction_of_rotten_fruits (a p : ℕ) (rotten_apples_eq_rotten_pears : (2 / 3) * a = (3 / 4) * p)
    (rotten_apples_fraction : 2 / 3 = 2 / 3)
    (rotten_pears_fraction : 3 / 4 = 3 / 4) :
    (4 * a) / (3 * (a + (4 / 3) * (2 * a) / 3)) = 12 / 17 :=
by
  sorry

end fraction_of_rotten_fruits_l84_84242


namespace circles_symmetric_sin_cos_l84_84974

noncomputable def sin_cos_product (θ : Real) : Real := Real.sin θ * Real.cos θ

theorem circles_symmetric_sin_cos (a θ : Real) 
(h1 : ∃ x1 y1, x1 = -a / 2 ∧ y1 = 0 ∧ 2*x1 - y1 - 1 = 0) 
(h2 : ∃ x2 y2, x2 = -a ∧ y2 = -Real.tan θ / 2 ∧ 2*x2 - y2 - 1 = 0) :
sin_cos_product θ = -2 / 5 := 
sorry

end circles_symmetric_sin_cos_l84_84974


namespace coin_flip_probability_l84_84399

theorem coin_flip_probability
    (P_A_heads : ℚ := 1 / 3)
    (P_B_heads : ℚ := 1 / 2)
    (P_C_heads : ℚ := 2 / 3)
    (P_select_coin : ℚ := 1 / 3):
    let P_3_heads_1_tail_given_A := (4 * (P_A_heads^3) * ((1 - P_A_heads))) in
    let P_3_heads_1_tail_given_B := (4 * (P_B_heads^3) * ((1 - P_B_heads))) in
    let P_3_heads_1_tail_given_C := (4 * (P_C_heads^3) * ((1 - P_C_heads))) in
    let total_P_3_heads_1_tail := (P_3_heads_1_tail_given_A * P_select_coin +
                                   P_3_heads_1_tail_given_B * P_select_coin +
                                   P_3_heads_1_tail_given_C * P_select_coin) in
    let P_A_given_3_heads_1_tail := (P_3_heads_1_tail_given_A * P_select_coin) / total_P_3_heads_1_tail in
    let numerator := nat.gcd 32 273 in
    let denominator := 273 / numerator in
    numerator + denominator = 273 :=
sorry

end coin_flip_probability_l84_84399


namespace coprime_3x3_grid_l84_84015

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l84_84015


namespace marbles_difference_l84_84025

def lostMarbles : ℕ := 8
def foundMarbles : ℕ := 10

theorem marbles_difference (lostMarbles foundMarbles : ℕ) : foundMarbles - lostMarbles = 2 := 
by
  sorry

end marbles_difference_l84_84025


namespace range_of_m_l84_84470

open Set

noncomputable def A (m : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 + m * x - y + 2 = 0} 

noncomputable def B : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x - y + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ B ≠ ∅) → (m ≤ -1 ∨ m ≥ 3) := 
sorry

end range_of_m_l84_84470


namespace no_real_numbers_x_l84_84297

theorem no_real_numbers_x (x : ℝ) : ¬ (-(x^2 + x + 1) ≥ 0) := sorry

end no_real_numbers_x_l84_84297


namespace num_four_digit_pos_integers_l84_84637

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l84_84637


namespace contrapositive_l84_84378

theorem contrapositive (p q : Prop) : (p → q) → (¬q → ¬p) :=
by
  sorry

end contrapositive_l84_84378


namespace trig_identity_example_l84_84919

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end trig_identity_example_l84_84919


namespace find_Z_l84_84142

theorem find_Z (Z : ℝ) (h : (100 + 20 / Z) * Z = 9020) : Z = 90 :=
sorry

end find_Z_l84_84142


namespace bob_plate_price_correct_l84_84610

-- Assuming units and specific values for the problem
def anne_plate_area : ℕ := 20 -- in square units
def bob_clay_usage : ℕ := 600 -- total clay used by Bob in square units
def bob_number_of_plates : ℕ := 15
def anne_plate_price : ℕ := 50 -- in cents
def anne_number_of_plates : ℕ := 30
def total_anne_earnings : ℕ := anne_number_of_plates * anne_plate_price

-- Condition
def bob_plate_area : ℕ := bob_clay_usage / bob_number_of_plates

-- Prove the price of one of Bob's plates
theorem bob_plate_price_correct : bob_number_of_plates * bob_plate_area = bob_clay_usage →
                                  bob_number_of_plates * 100 = total_anne_earnings :=
by
  intros 
  sorry

end bob_plate_price_correct_l84_84610


namespace completing_square_correct_l84_84091

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l84_84091


namespace find_missing_percentage_l84_84103

theorem find_missing_percentage (P : ℝ) : (P * 50 = 2.125) → (P * 100 = 4.25) :=
by
  sorry

end find_missing_percentage_l84_84103


namespace intersection_of_A_and_B_l84_84165

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_A_and_B_l84_84165


namespace roots_condition_l84_84154

theorem roots_condition (r1 r2 p : ℝ) (h_eq : ∀ x : ℝ, x^2 + p * x + 12 = 0 → (x = r1 ∨ x = r2))
(h_distinct : r1 ≠ r2) (h_vieta1 : r1 + r2 = -p) (h_vieta2 : r1 * r2 = 12) : 
|r1| > 3 ∨ |r2| > 3 :=
by
  sorry

end roots_condition_l84_84154


namespace zhao_estimate_larger_l84_84181

theorem zhao_estimate_larger (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2 * ε) > x - y :=
by
  sorry

end zhao_estimate_larger_l84_84181


namespace ganesh_average_speed_l84_84108

variable (D : ℝ) -- the distance between towns X and Y

theorem ganesh_average_speed :
  let time_x_to_y := D / 43
  let time_y_to_x := D / 34
  let total_distance := 2 * D
  let total_time := time_x_to_y + time_y_to_x
  let avg_speed := total_distance / total_time
  avg_speed = 37.97 := by
    sorry

end ganesh_average_speed_l84_84108


namespace addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l84_84917

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l84_84917


namespace voucher_placement_l84_84247

/-- A company wants to popularize the sweets they market by hiding prize vouchers in some of the boxes.
The management believes the promotion is effective and the cost is bearable if a customer who buys 10 boxes has approximately a 50% chance of finding at least one voucher.
We aim to determine how often vouchers should be placed in the boxes to meet this requirement. -/
theorem voucher_placement (n : ℕ) (h_positive : n > 0) :
  (1 - (1 - 1/n)^10) ≥ 1/2 → n ≤ 15 :=
sorry

end voucher_placement_l84_84247


namespace min_value_f_range_of_a_l84_84148

-- Define the function f(x) with parameter a.
def f (x a : ℝ) := |x + a| + |x - a|

-- (Ⅰ) Statement: Prove that for a = 1, the minimum value of f(x) is 2.
theorem min_value_f (x : ℝ) : f x 1 ≥ 2 :=
  by sorry

-- (Ⅱ) Statement: Prove that if f(2) > 5, then the range of values for a is (-∞, -5/2) ∪ (5/2, +∞).
theorem range_of_a (a : ℝ) : f 2 a > 5 → a < -5 / 2 ∨ a > 5 / 2 :=
  by sorry

end min_value_f_range_of_a_l84_84148


namespace travel_time_second_bus_l84_84694

def distance_AB : ℝ := 100 -- kilometers
def passengers_first : ℕ := 20
def speed_first : ℝ := 60 -- kilometers per hour
def breakdown_time : ℝ := 0.5 -- hours
def passengers_second_initial : ℕ := 22
def speed_second_initial : ℝ := 50 -- kilometers per hour
def additional_passengers_speed_decrease : ℝ := 1 -- speed decrease for every additional 2 passengers
def passenger_factor : ℝ := 2
def additional_passengers : ℕ := 20
def total_time_second_bus : ℝ := 2.35 -- hours

theorem travel_time_second_bus :
  let distance_first_half := (breakdown_time * speed_first)
  let remaining_distance := distance_AB - distance_first_half
  let time_to_reach_breakdown := distance_first_half / speed_second_initial
  let new_speed_second_bus := speed_second_initial - (additional_passengers / passenger_factor) * additional_passengers_speed_decrease
  let time_from_breakdown_to_B := remaining_distance / new_speed_second_bus
  total_time_second_bus = time_to_reach_breakdown + time_from_breakdown_to_B := 
sorry

end travel_time_second_bus_l84_84694


namespace sparse_real_nums_l84_84222

noncomputable def is_sparse (r : ℝ) : Prop :=
  ∃n > 0, ∀s : ℝ, s^n = r → s = 1 ∨ s = -1 ∨ s = 0

theorem sparse_real_nums (r : ℝ) : is_sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := 
by
  sorry

end sparse_real_nums_l84_84222


namespace find_rate_per_kg_of_mangoes_l84_84791

theorem find_rate_per_kg_of_mangoes (r : ℝ) 
  (total_units_paid : ℝ) (grapes_kg : ℝ) (grapes_rate : ℝ)
  (mangoes_kg : ℝ) (total_grapes_cost : ℝ)
  (total_mangoes_cost : ℝ) (total_cost : ℝ) :
  grapes_kg = 8 →
  grapes_rate = 70 →
  mangoes_kg = 10 →
  total_units_paid = 1110 →
  total_grapes_cost = grapes_kg * grapes_rate →
  total_mangoes_cost = total_units_paid - total_grapes_cost →
  r = total_mangoes_cost / mangoes_kg →
  r = 55 := by
  intros
  sorry

end find_rate_per_kg_of_mangoes_l84_84791


namespace sum_of_digits_1_to_1000_l84_84141

/--  sum_of_digits calculates the sum of digits of a given number n -/
def sum_of_digits(n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- sum_of_digits_in_range calculates the sum of the digits 
of all numbers in the inclusive range from 1 to m -/
def sum_of_digits_in_range (m : ℕ) : ℕ :=
  (Finset.range (m + 1)).sum sum_of_digits

theorem sum_of_digits_1_to_1000 : sum_of_digits_in_range 1000 = 13501 :=
by
  sorry

end sum_of_digits_1_to_1000_l84_84141


namespace most_stable_machine_l84_84067

noncomputable def var_A : ℝ := 10.3
noncomputable def var_B : ℝ := 6.9
noncomputable def var_C : ℝ := 3.5

theorem most_stable_machine :
  (var_C < var_B) ∧ (var_C < var_A) :=
by
  sorry

end most_stable_machine_l84_84067


namespace choir_row_lengths_l84_84115

theorem choir_row_lengths : 
  ∃ s : Finset ℕ, (∀ d ∈ s, d ∣ 90 ∧ 6 ≤ d ∧ d ≤ 15) ∧ s.card = 4 := by
  sorry

end choir_row_lengths_l84_84115


namespace marble_problem_solution_l84_84891

noncomputable def probability_two_marbles (red_marble_initial white_marble_initial total_drawn : ℕ) : ℚ :=
  let total_initial := red_marble_initial + white_marble_initial
  let probability_first_white := (white_marble_initial : ℚ) / total_initial
  let red_marble_after_first_draw := red_marble_initial
  let total_after_first_draw := total_initial - 1
  let probability_second_red := (red_marble_after_first_draw : ℚ) / total_after_first_draw
  probability_first_white * probability_second_red

theorem marble_problem_solution :
  probability_two_marbles 4 6 2 = 4 / 15 := by
  sorry

end marble_problem_solution_l84_84891


namespace problem_l84_84160

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.log x + (a + 1) * (1 / x - 2)

theorem problem (a x : ℝ) (ha_pos : a > 0) :
  f a x > - (a^2 / (a + 1)) - 2 :=
sorry

end problem_l84_84160


namespace product_of_three_numbers_l84_84545

theorem product_of_three_numbers (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x = 4 * (y + z)) 
  (h3 : y = 7 * z) :
  x * y * z = 28 := 
by 
  sorry

end product_of_three_numbers_l84_84545


namespace algebraic_expression_value_l84_84564

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 23 - 1) : x^2 + 2 * x + 2 = 24 :=
by
  -- Start of the proof
  sorry -- Proof is omitted as per instructions

end algebraic_expression_value_l84_84564


namespace gcd_119_34_l84_84381

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end gcd_119_34_l84_84381


namespace strawberries_for_mom_l84_84518

-- Define the conditions as Lean definitions
def dozen : ℕ := 12
def strawberries_picked : ℕ := 2 * dozen
def strawberries_eaten : ℕ := 6

-- Define the statement to be proven
theorem strawberries_for_mom : (strawberries_picked - strawberries_eaten) = 18 := by
  sorry

end strawberries_for_mom_l84_84518


namespace sum_of_consecutive_integers_product_336_l84_84713

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l84_84713


namespace intersection_complement_correct_l84_84304

open Set

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := { x | True }

theorem intersection_complement_correct :
  (A ∩ (U \ B)) = {x | x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5} :=
by
  sorry

end intersection_complement_correct_l84_84304


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l84_84261

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l84_84261


namespace response_rate_is_60_percent_l84_84893

-- Definitions based on conditions
def responses_needed : ℕ := 900
def questionnaires_mailed : ℕ := 1500

-- Derived definition
def response_rate_percentage : ℚ := (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

-- The theorem stating the problem
theorem response_rate_is_60_percent :
  response_rate_percentage = 60 := 
sorry

end response_rate_is_60_percent_l84_84893


namespace estimate_fitness_population_l84_84549

theorem estimate_fitness_population :
  ∀ (sample_size total_population : ℕ) (sample_met_standards : Nat) (percentage_met_standards estimated_met_standards : ℝ),
  sample_size = 1000 →
  total_population = 1200000 →
  sample_met_standards = 950 →
  percentage_met_standards = (sample_met_standards : ℝ) / (sample_size : ℝ) →
  estimated_met_standards = percentage_met_standards * (total_population : ℝ) →
  estimated_met_standards = 1140000 := by sorry

end estimate_fitness_population_l84_84549


namespace beach_trip_time_l84_84023

noncomputable def totalTripTime (driveTime eachWay : ℝ) (beachTimeFactor : ℝ) : ℝ :=
  let totalDriveTime := eachWay * 2
  totalDriveTime + (totalDriveTime * beachTimeFactor)

theorem beach_trip_time :
  totalTripTime 2 2 2.5 = 14 := 
by
  sorry

end beach_trip_time_l84_84023


namespace positive_rational_as_sum_of_cubes_l84_84362

theorem positive_rational_as_sum_of_cubes (q : ℚ) (h_q_pos : q > 0) : 
  ∃ (a b c d : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = ((a^3 + b^3) / (c^3 + d^3)) :=
sorry

end positive_rational_as_sum_of_cubes_l84_84362


namespace area_of_ellipse_l84_84442

theorem area_of_ellipse (x y : ℝ) (h : x^2 + 6 * x + 4 * y^2 - 8 * y + 9 = 0) : 
  area = 2 * Real.pi :=
sorry

end area_of_ellipse_l84_84442


namespace smallest_number_diminished_by_16_divisible_l84_84106

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end smallest_number_diminished_by_16_divisible_l84_84106


namespace average_distance_per_day_l84_84973

def miles_monday : ℕ := 12
def miles_tuesday : ℕ := 18
def miles_wednesday : ℕ := 21
def total_days : ℕ := 3

def total_distance : ℕ := miles_monday + miles_tuesday + miles_wednesday

theorem average_distance_per_day : total_distance / total_days = 17 := by
  sorry

end average_distance_per_day_l84_84973


namespace solve_for_x_l84_84479

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l84_84479


namespace population_present_l84_84828

variable (P : ℝ)

theorem population_present (h1 : P * 0.90 = 450) : P = 500 :=
by
  sorry

end population_present_l84_84828


namespace prime_pair_probability_even_sum_l84_84657

open Finset

-- Conditions given in the problem
def firstEightPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Main statement of the problem to prove
theorem prime_pair_probability_even_sum : 
  let total_pairs := (firstEightPrimes.choose 2).card in
  let odd_pairs := (firstEightPrimes.filter (λ x, x ≠ 2)).card in
  total_pairs > 0 → 
  (total_pairs - odd_pairs) / total_pairs = 3 / 4 :=
by
  intros total_pairs odd_pairs h
  sorry

end prime_pair_probability_even_sum_l84_84657


namespace solution_for_x_l84_84671

theorem solution_for_x : ∀ (x : ℚ), (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) → x = 1 / 5 :=
by
  sorry

end solution_for_x_l84_84671


namespace regular_price_coffee_l84_84584

theorem regular_price_coffee (y : ℝ) (h1 : 0.4 * y / 4 = 4) : y = 40 :=
by
  sorry

end regular_price_coffee_l84_84584


namespace vector_v_satisfies_conditions_l84_84513

open Matrix
open Matrix.SpecialLinearGroup

def vector_a : Vector 3 ℝ := ![2, 1, 1]
def vector_b : Vector 3 ℝ := ![3, -1, 0]
def vector_v : Vector 3 ℝ := ![5, 0, 1]

theorem vector_v_satisfies_conditions :
  (crossProduct vector_v vector_a = crossProduct vector_b vector_a) ∧
  (crossProduct vector_v vector_b = crossProduct vector_a vector_b) :=
by
  sorry

end vector_v_satisfies_conditions_l84_84513


namespace rory_more_jellybeans_l84_84039

-- Definitions based on the conditions
def G : ℕ := 15 -- Gigi has 15 jellybeans
def LorelaiConsumed (R G : ℕ) : ℕ := 3 * (R + G) -- Lorelai has already eaten three times the total number of jellybeans

theorem rory_more_jellybeans {R : ℕ} (h1 : LorelaiConsumed R G = 180) : (R - G) = 30 :=
  by
    -- we can skip the proof here with sorry, as we are only interested in the statement for now
    sorry

end rory_more_jellybeans_l84_84039


namespace repeating_decimal_sum_l84_84135

theorem repeating_decimal_sum :
  (0.3333333333 : ℚ) + (0.0404040404 : ℚ) + (0.005005005 : ℚ) + (0.000600060006 : ℚ) = 3793 / 9999 := by
sorry

end repeating_decimal_sum_l84_84135


namespace sufficient_but_not_necessary_condition_for_monotonicity_l84_84868

theorem sufficient_but_not_necessary_condition_for_monotonicity
  (a : ℕ → ℝ)
  (h_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2)
  (h_initial : a 1 = 2) :
  (∀ n : ℕ, n > 0 → a n > a 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_monotonicity_l84_84868


namespace number_of_different_partitions_l84_84827

variable {α : Type*} [DecidableEq α]

def different_partitions (A : Finset α) : Finset (Finset α × Finset α) :=
  (A.powerset.product A.powerset).filter (λ p, p.1 ∪ p.2 = A)

theorem number_of_different_partitions (A : Finset α) [Fintype α] (hA : A = {1, 2, 3}) :
  (different_partitions A).card = 27 := sorry

end number_of_different_partitions_l84_84827


namespace distance_from_A_to_directrix_on_parabola_l84_84308

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l84_84308


namespace length_PQ_is_5_l84_84155

/-
Given:
- Point P with coordinates (3, 4, 5)
- Point Q is the projection of P onto the xOy plane

Show:
- The length of the segment PQ is 5
-/

def P : ℝ × ℝ × ℝ := (3, 4, 5)
def Q : ℝ × ℝ × ℝ := (3, 4, 0)

theorem length_PQ_is_5 : dist P Q = 5 := by
  sorry

end length_PQ_is_5_l84_84155


namespace min_avg_score_less_than_record_l84_84804

theorem min_avg_score_less_than_record
  (old_record_avg : ℝ := 287.5)
  (players : ℕ := 6)
  (rounds : ℕ := 12)
  (total_points_11_rounds : ℝ := 19350.5)
  (bonus_points_9_rounds : ℕ := 300) :
  ∀ final_round_avg : ℝ, (final_round_avg = (old_record_avg * players * rounds - total_points_11_rounds + bonus_points_9_rounds) / players) →
  old_record_avg - final_round_avg = 12.5833 :=
by {
  sorry
}

end min_avg_score_less_than_record_l84_84804


namespace no_two_or_more_consecutive_sum_30_l84_84001

theorem no_two_or_more_consecutive_sum_30 :
  ∀ (a n : ℕ), n ≥ 2 → (n * (2 * a + n - 1) = 60) → false :=
by
  intro a n hn h
  sorry

end no_two_or_more_consecutive_sum_30_l84_84001


namespace range_of_x_l84_84969

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) :
  x > 1/3 ∨ x < -1/2 :=
sorry

end range_of_x_l84_84969


namespace initial_number_proof_l84_84114

-- Definitions for the given problem
def to_add : ℝ := 342.00000000007276
def multiple_of_412 (n : ℤ) : ℝ := 412 * n

-- The initial number
def initial_number : ℝ := 412 - to_add

-- The proof problem statement
theorem initial_number_proof (n : ℤ) (h : multiple_of_412 n = initial_number + to_add) : 
  ∃ x : ℝ, initial_number = x := 
sorry

end initial_number_proof_l84_84114


namespace wall_number_of_bricks_l84_84392

theorem wall_number_of_bricks (x : ℝ) :
  (∃ x, 6 * ((x / 7) + (x / 11) - 12) = x) →  x = 179 :=
by
  sorry

end wall_number_of_bricks_l84_84392


namespace percentage_error_formula_l84_84421

noncomputable def percentage_error_in_area (a b : ℝ) (x y : ℝ) :=
  let actual_area := a * b
  let measured_area := a * (1 + x / 100) * b * (1 + y / 100)
  let error_percentage := ((measured_area - actual_area) / actual_area) * 100
  error_percentage

theorem percentage_error_formula (a b x y : ℝ) :
  percentage_error_in_area a b x y = x + y + (x * y / 100) :=
by
  sorry

end percentage_error_formula_l84_84421


namespace triangle_BC_60_l84_84177

theorem triangle_BC_60 {A B C X : Type}
    (AB AC BX CX : ℕ) (h1 : AB = 70) (h2 : AC = 80) 
    (h3 : AB^2 - BX^2 = CX*(CX + BX)) 
    (h4 : BX % 7 = 0)
    (h5 : BX + CX = (BC : ℕ)) 
    (h6 : BC = 60) :
  BC = 60 := 
sorry

end triangle_BC_60_l84_84177


namespace kim_total_water_drank_l84_84511

noncomputable def total_water_kim_drank : Float :=
  let water_from_bottle := 1.5 * 32
  let water_from_can := 12
  let shared_bottle := (3 / 5) * 32
  water_from_bottle + water_from_can + shared_bottle

theorem kim_total_water_drank :
  total_water_kim_drank = 79.2 :=
by
  -- Proof skipped
  sorry

end kim_total_water_drank_l84_84511


namespace distance_from_A_to_directrix_of_C_l84_84311

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l84_84311


namespace a4_is_5_l84_84970

-- Define the condition x^5 = a_n + a_1(x-1) + a_2(x-1)^2 + a_3(x-1)^3 + a_4(x-1)^4 + a_5(x-1)^5
noncomputable def polynomial_identity (x a_n a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5

-- Define the theorem statement
theorem a4_is_5 (x a_n a_1 a_2 a_3 a_5 : ℝ) (h : polynomial_identity x a_n a_1 a_2 a_3 5 a_5) : a_4 = 5 :=
 by
 sorry

end a4_is_5_l84_84970


namespace morgan_total_pens_l84_84682

def initial_red_pens : Nat := 65
def initial_blue_pens : Nat := 45
def initial_black_pens : Nat := 58
def initial_green_pens : Nat := 36
def initial_purple_pens : Nat := 27

def red_pens_given_away : Nat := 15
def blue_pens_given_away : Nat := 20
def green_pens_given_away : Nat := 10

def black_pens_bought : Nat := 12
def purple_pens_bought : Nat := 5

def final_red_pens : Nat := initial_red_pens - red_pens_given_away
def final_blue_pens : Nat := initial_blue_pens - blue_pens_given_away
def final_black_pens : Nat := initial_black_pens + black_pens_bought
def final_green_pens : Nat := initial_green_pens - green_pens_given_away
def final_purple_pens : Nat := initial_purple_pens + purple_pens_bought

def total_pens : Nat := final_red_pens + final_blue_pens + final_black_pens + final_green_pens + final_purple_pens

theorem morgan_total_pens : total_pens = 203 := 
by
  -- final_red_pens = 50
  -- final_blue_pens = 25
  -- final_black_pens = 70
  -- final_green_pens = 26
  -- final_purple_pens = 32
  -- Therefore, total_pens = 203
  sorry

end morgan_total_pens_l84_84682


namespace minimum_greeting_pairs_l84_84065

def minimum_mutual_greetings (n: ℕ) (g: ℕ) : ℕ :=
  (n * g - (n * (n - 1)) / 2)

theorem minimum_greeting_pairs :
  minimum_mutual_greetings 400 200 = 200 :=
by 
  sorry

end minimum_greeting_pairs_l84_84065


namespace johns_raise_percentage_increase_l84_84883

theorem johns_raise_percentage_increase (original_amount new_amount : ℝ) (h_original : original_amount = 60) (h_new : new_amount = 70) :
  ((new_amount - original_amount) / original_amount) * 100 = 16.67 := 
  sorry

end johns_raise_percentage_increase_l84_84883


namespace quadratic_real_roots_range_l84_84650

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end quadratic_real_roots_range_l84_84650


namespace number_of_students_in_line_l84_84691

-- Definitions for the conditions
def yoojung_last (n : ℕ) : Prop :=
  n = 14

def eunjung_position : ℕ := 5

def students_between (n : ℕ) : Prop :=
  n = 8

noncomputable def total_students : ℕ := 14

-- The theorem to be proven
theorem number_of_students_in_line 
  (last : yoojung_last total_students) 
  (eunjung_pos : eunjung_position = 5) 
  (between : students_between 8) :
  total_students = 14 := by
  sorry

end number_of_students_in_line_l84_84691


namespace find_a_and_c_l84_84158

theorem find_a_and_c (a c : ℝ) (h : ∀ x : ℝ, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c < 0) :
  a = 12 ∧ c = -2 :=
by {
  sorry
}

end find_a_and_c_l84_84158


namespace tangent_of_11pi_over_4_l84_84932

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l84_84932


namespace sum_of_coefficients_of_expansion_l84_84435

theorem sum_of_coefficients_of_expansion (x y : ℝ) :
  (3*x - 4*y) ^ 20 = 1 :=
by 
  sorry

end sum_of_coefficients_of_expansion_l84_84435


namespace number_of_yellow_marbles_l84_84836

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end number_of_yellow_marbles_l84_84836


namespace designated_time_to_B_l84_84407

theorem designated_time_to_B (s v : ℝ) (x : ℝ) (V' : ℝ)
  (h1 : s / 2 = (x + 2) * V')
  (h2 : s / (2 * V') + 1 + s / (2 * (V' + v)) = x) :
  x = (v + Real.sqrt (9 * v ^ 2 + 6 * v * s)) / v :=
by
  sorry

end designated_time_to_B_l84_84407


namespace no_solution_inequality_system_l84_84335

theorem no_solution_inequality_system (m : ℝ) :
  (¬ ∃ x : ℝ, 2 * x - 1 < 3 ∧ x > m) ↔ m ≥ 2 :=
by
  sorry

end no_solution_inequality_system_l84_84335


namespace geometric_sequence_sum_5_l84_84952

theorem geometric_sequence_sum_5 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q, ∀ n, a (n + 1) = a n * q) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  (a 1 * (1 - (2:ℝ)^5) / (1 - (2:ℝ))) = 31 := 
by
  sorry

end geometric_sequence_sum_5_l84_84952


namespace cos_neg_three_pi_over_two_eq_zero_l84_84925

noncomputable def cos_neg_three_pi_over_two : ℝ :=
  Real.cos (-3 * Real.pi / 2)

theorem cos_neg_three_pi_over_two_eq_zero :
  cos_neg_three_pi_over_two = 0 :=
by
  -- Using trigonometric identities and periodicity of cosine function
  sorry

end cos_neg_three_pi_over_two_eq_zero_l84_84925


namespace probability_at_least_one_boy_and_one_girl_l84_84255

theorem probability_at_least_one_boy_and_one_girl :
  let P := (1 - (1/16 + 1/16)) = 7 / 8,
  (∀ (N: ℕ), (N = 4) → 
    let prob_all_boys := (1 / N) ^ N,
    let prob_all_girls := (1 / N) ^ N,
    let prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)
  in prob_at_least_one_boy_and_one_girl = P) :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l84_84255


namespace probability_of_digit_six_l84_84787

theorem probability_of_digit_six :
  let total_numbers := 90
  let favorable_numbers := 18
  0 < total_numbers ∧ 0 < favorable_numbers →
  (favorable_numbers / total_numbers : ℚ) = 1 / 5 :=
by
  intros total_numbers favorable_numbers h
  sorry

end probability_of_digit_six_l84_84787


namespace final_ratio_l84_84338

-- Define initial conditions
def initial_milk_ratio : ℕ := 1
def initial_water_ratio : ℕ := 5
def total_parts : ℕ := initial_milk_ratio + initial_water_ratio
def can_capacity : ℕ := 8
def additional_milk : ℕ := 2
def initial_volume : ℕ := can_capacity - additional_milk
def part_volume : ℕ := initial_volume / total_parts

-- Define initial quantities
def initial_milk_quantity : ℕ := part_volume * initial_milk_ratio
def initial_water_quantity : ℕ := part_volume * initial_water_ratio

-- Define final quantities
def final_milk_quantity : ℕ := initial_milk_quantity + additional_milk
def final_water_quantity : ℕ := initial_water_quantity

-- Hypothesis: final ratios of milk and water
def final_ratio_of_milk_to_water : ℕ × ℕ := (final_milk_quantity, final_water_quantity)

-- Final ratio should be 3:5
theorem final_ratio (h : final_ratio_of_milk_to_water = (3, 5)) : final_ratio_of_milk_to_water = (3, 5) :=
  by
  sorry

end final_ratio_l84_84338


namespace annika_hike_distance_l84_84425

-- Define the conditions as definitions
def hiking_rate : ℝ := 10  -- rate of 10 minutes per kilometer
def total_minutes : ℝ := 35 -- total available time in minutes
def total_distance_east : ℝ := 3 -- total distance hiked east

-- Define the statement to prove
theorem annika_hike_distance : ∃ (x : ℝ), (x / hiking_rate) + ((total_distance_east - x) / hiking_rate) = (total_minutes - 30) / hiking_rate :=
by
  sorry

end annika_hike_distance_l84_84425


namespace closest_weight_total_shortfall_total_selling_price_l84_84390

-- Definitions
def standard_weight : ℝ := 25
def weights : List ℝ := [1.5, -3, 2, -0.5, 1, -2, -2.5, -2]
def price_per_kg : ℝ := 2.6

-- Assertions
theorem closest_weight : ∃ w ∈ weights, abs w = 0.5 ∧ 25 + w = 24.5 :=
by sorry

theorem total_shortfall : (weights.sum = -5.5) :=
by sorry

theorem total_selling_price : (8 * standard_weight + weights.sum) * price_per_kg = 505.7 :=
by sorry

end closest_weight_total_shortfall_total_selling_price_l84_84390


namespace derivative_at_one_l84_84322

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_at_one : deriv f 1 = 2 + Real.exp 1 := by
  sorry

end derivative_at_one_l84_84322


namespace large_circle_diameter_proof_l84_84437

noncomputable def large_circle_diameter (r: ℝ) (count: ℕ) : ℝ :=
  let s := 2 * r
  s / (2 * Real.sin (Real.pi / count))

theorem large_circle_diameter_proof :
  large_circle_diameter 4 8 ≈ 20.94 :=
by
  let r := 4
  let count := 8
  let diameter := 2 * large_circle_diameter r count
  have : diameter ≈ 10.47 * 2 :=
    by sorry
  exact this

end large_circle_diameter_proof_l84_84437


namespace abs_neg_eight_plus_three_pow_zero_eq_nine_l84_84595

theorem abs_neg_eight_plus_three_pow_zero_eq_nine :
  |-8| + 3^0 = 9 :=
by
  sorry

end abs_neg_eight_plus_three_pow_zero_eq_nine_l84_84595


namespace find_x_l84_84490

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l84_84490


namespace infinite_N_for_factorization_l84_84366

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l84_84366


namespace multiple_of_4_and_8_l84_84533

theorem multiple_of_4_and_8 (a b : ℤ) (h1 : ∃ k1 : ℤ, a = 4 * k1) (h2 : ∃ k2 : ℤ, b = 8 * k2) :
  (∃ k3 : ℤ, b = 4 * k3) ∧ (∃ k4 : ℤ, a - b = 4 * k4) :=
by
  sorry

end multiple_of_4_and_8_l84_84533


namespace number_of_candies_bought_on_Tuesday_l84_84132

theorem number_of_candies_bought_on_Tuesday (T : ℕ) 
  (thursday_candies : ℕ := 5) 
  (friday_candies : ℕ := 2) 
  (candies_left : ℕ := 4) 
  (candies_eaten : ℕ := 6) 
  (total_initial_candies : T + thursday_candies + friday_candies = candies_left + candies_eaten) 
  : T = 3 := by
  sorry

end number_of_candies_bought_on_Tuesday_l84_84132


namespace possible_values_of_5x_plus_2_l84_84332

theorem possible_values_of_5x_plus_2 (x : ℝ) :
  (x - 4) * (5 * x + 2) = 0 →
  (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by
  intro h
  sorry

end possible_values_of_5x_plus_2_l84_84332


namespace students_neither_class_l84_84118

theorem students_neither_class : 
  let total_students := 1500
  let music_students := 300
  let art_students := 200
  let dance_students := 100
  let theater_students := 50
  let music_art_students := 80
  let music_dance_students := 40
  let music_theater_students := 30
  let art_dance_students := 25
  let art_theater_students := 20
  let dance_theater_students := 10
  let music_art_dance_students := 50
  let music_art_theater_students := 30
  let art_dance_theater_students := 20
  let music_dance_theater_students := 10
  let all_four_students := 5
  total_students - 
    (music_students + 
     art_students + 
     dance_students + 
     theater_students - 
     (music_art_students + 
      music_dance_students + 
      music_theater_students + 
      art_dance_students + 
      art_theater_students + 
      dance_theater_students) + 
     (music_art_dance_students + 
      music_art_theater_students + 
      art_dance_theater_students + 
      music_dance_theater_students) - 
     all_four_students) = 950 :=
sorry

end students_neither_class_l84_84118


namespace subtract_23_result_l84_84228

variable {x : ℕ}

theorem subtract_23_result (h : x + 30 = 55) : x - 23 = 2 :=
sorry

end subtract_23_result_l84_84228


namespace symmetric_point_with_respect_to_x_axis_l84_84537

-- Definition of point M
def point_M : ℝ × ℝ := (3, -4)

-- Define the symmetry condition with respect to the x-axis
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Statement that the symmetric point to point M with respect to the x-axis is (3, 4)
theorem symmetric_point_with_respect_to_x_axis : symmetric_x point_M = (3, 4) :=
by
  -- This is the statement of the theorem; the proof will be added here.
  sorry

end symmetric_point_with_respect_to_x_axis_l84_84537


namespace grid_is_valid_l84_84012

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l84_84012


namespace units_digit_of_3_pow_2009_l84_84153

noncomputable def units_digit (n : ℕ) : ℕ :=
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 9
  else if n % 4 = 3 then 7
  else 1

theorem units_digit_of_3_pow_2009 : units_digit (2009) = 3 :=
by
  -- Skipping the proof as instructed
  sorry

end units_digit_of_3_pow_2009_l84_84153


namespace inequality_of_products_l84_84543

theorem inequality_of_products
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_of_products_l84_84543


namespace min_value_g_l84_84654

variables {ℝ : Type*} [linear_ordered_field ℝ] {f g : ℝ → ℝ}

-- Definitions of odd and even functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Given conditions
variables (h1 : is_odd f)
variables (h2 : is_even g)
variables (h3 : ∀ x, f x + g x = 2^x)

-- Prove statement
theorem min_value_g : (∀ x, f x + g x = 2^x) → (∀ x, is_odd f) → (∀ x, is_even g) → ∃ x, g x = 1 :=
by 
  sorry

end min_value_g_l84_84654


namespace fraction_to_decimal_l84_84439

theorem fraction_to_decimal :
  (17 : ℚ) / (2^2 * 5^4) = 0.0068 :=
by
  sorry

end fraction_to_decimal_l84_84439


namespace completing_square_result_l84_84095

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l84_84095


namespace area_of_region_l84_84753

theorem area_of_region (x y : ℝ) : (x^2 + y^2 + 6 * x - 8 * y = 1) → (π * 26) = 26 * π :=
by
  intro h
  sorry

end area_of_region_l84_84753


namespace geometric_series_sum_l84_84597

theorem geometric_series_sum :
  let a := 2
  let r := -2
  let n := 10
  let Sn := (a : ℚ) * (r^n - 1) / (r - 1)
  Sn = 2050 / 3 :=
by
  sorry

end geometric_series_sum_l84_84597


namespace angle_ZQY_l84_84528

-- Definitions
noncomputable def length_XY : ℝ := 2 * r

noncomputable def radius_XY : ℝ := r

noncomputable def radius_YZ : ℝ := r / 2

noncomputable def area_large_semicircle : ℝ := (1 / 2) * real.pi * r^2

noncomputable def area_small_semicircle : ℝ := (1 / 2) * real.pi * (r / 2)^2

noncomputable def total_area : ℝ := area_large_semicircle + area_small_semicircle

noncomputable def split_area : ℝ := total_area / 2

noncomputable def angle_theta : ℝ := 360 * (split_area / area_large_semicircle)

-- The Lean statement
theorem angle_ZQY : angle_theta = 112.5 := 
by sorry

end angle_ZQY_l84_84528


namespace max_sum_of_factors_l84_84500

theorem max_sum_of_factors (A B C : ℕ) (h1 : A * B * C = 2310) (h2 : A ≠ B) (h3 : B ≠ C) (h4 : A ≠ C) (h5 : 0 < A) (h6 : 0 < B) (h7 : 0 < C) : 
  A + B + C ≤ 42 := 
sorry

end max_sum_of_factors_l84_84500


namespace max_value_of_a_l84_84306

theorem max_value_of_a 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 - a * x) 
  (h2 : ∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x ≤ f y) : 
  a ≤ 3 :=
sorry

end max_value_of_a_l84_84306


namespace geometric_sequence_sum_l84_84461

-- Define the relations for geometric sequences
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (m n p q : ℕ), m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n, a n > 0)
  (h_cond : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 :=
sorry

end geometric_sequence_sum_l84_84461


namespace MapleLeafHigh_points_l84_84008

def MapleLeafHigh (x y : ℕ) : Prop :=
  (1/3 * x + 3/8 * x + 18 + y = x) ∧ (10 ≤ y) ∧ (y ≤ 30)

theorem MapleLeafHigh_points : ∃ y, MapleLeafHigh 104 y ∧ y = 21 := 
by
  use 21
  sorry

end MapleLeafHigh_points_l84_84008


namespace sum_of_possible_values_l84_84329

theorem sum_of_possible_values (x : ℝ) :
  (x + 3) * (x - 4) = 20 →
  ∃ a b, (a ≠ b) ∧ 
         ((x = a) ∨ (x = b)) ∧ 
         (x^2 - x - 32 = 0) ∧ 
         (a + b = 1) :=
by
  sorry

end sum_of_possible_values_l84_84329


namespace num_prime_divisors_of_50_fac_l84_84645

-- Define the set of all prime numbers less than or equal to 50.
def primes_le_50 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Define the factorial function.
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the number of prime divisors of n.
noncomputable def num_prime_divisors (n : ℕ) : ℕ :=
(set.count (λ p, p ∣ n) primes_le_50)

-- The theorem statement.
theorem num_prime_divisors_of_50_fac : num_prime_divisors (factorial 50) = 15 :=
by
  sorry

end num_prime_divisors_of_50_fac_l84_84645


namespace determine_conflicting_pairs_l84_84064

structure EngineerSetup where
  n : ℕ
  barrels : Fin (2 * n) → Reactant
  conflicts : Fin n → (Reactant × Reactant)

def testTubeBurst (r1 r2 : Reactant) (conflicts : Fin n → (Reactant × Reactant)) : Prop :=
  ∃ i, conflicts i = (r1, r2) ∨ conflicts i = (r2, r1)

theorem determine_conflicting_pairs (setup : EngineerSetup) :
  ∃ pairs : Fin n → (Reactant × Reactant),
  (∀ i, pairs i ∈ { p | ∃ j, setup.conflicts j = p ∨ setup.conflicts j = (p.snd, p.fst) }) ∧
  (∀ i j, i ≠ j → pairs i ≠ pairs j) := 
sorry

end determine_conflicting_pairs_l84_84064


namespace find_xy_l84_84629

theorem find_xy (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : (x - 10)^2 + (y - 10)^2 = 18) : 
  x * y = 91 := 
by {
  sorry
}

end find_xy_l84_84629


namespace range_of_f_l84_84301

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem range_of_f :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (f x) ∈ Set.Icc (-1 : ℝ) 2) :=
by
  sorry

end range_of_f_l84_84301


namespace intersection_points_of_curve_and_line_max_distance_condition_l84_84503

open Real

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 * cos θ, sin θ)

noncomputable def line (a t : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

theorem intersection_points_of_curve_and_line (a : ℝ) (h₁ : a = -1) :
  (∃ t θ,
    curve θ = line a t) ↔
  (∃ t θ,
    (curve θ = (3, 0) ∨ curve θ = (-21/25, 24/25))) :=
by
  sorry

theorem max_distance_condition (a : ℝ) (h₂ : ∃ θ d,
  d = √17 ∧ 
  d = abs (3 * cos θ + 4 * sin θ - a - 4) / √17) :
  a = -16 ∨ a = 8 :=
by 
  sorry

end intersection_points_of_curve_and_line_max_distance_condition_l84_84503


namespace find_x_l84_84483

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l84_84483


namespace number_exceeds_by_35_l84_84882

theorem number_exceeds_by_35 (x : ℤ) (h : x = (3 / 8 : ℚ) * x + 35) : x = 56 :=
by
  sorry

end number_exceeds_by_35_l84_84882


namespace part_1_part_2a_part_2b_l84_84027

namespace InequalityProofs

-- Definitions extracted from the problem
def quadratic_function (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

-- Lean statement for Part 1
theorem part_1 (m : ℝ) : (∀ x : ℝ, quadratic_function m x ≥ -2) ↔ m ∈ Set.Ici (1 / 3) :=
sorry

-- Lean statement for Part 2, breaking into separate theorems for different ranges of m
theorem part_2a (m : ℝ) (h : m < -1) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic (-1 / m) ∪ Set.Ici 1)) :=
sorry

theorem part_2b (m : ℝ) (h : -1 < m ∧ m < 0) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic 1 ∪ Set.Ici (-1 / m))) :=
sorry

end InequalityProofs

end part_1_part_2a_part_2b_l84_84027


namespace seventh_term_geometric_seq_l84_84697

theorem seventh_term_geometric_seq (a r : ℝ) (h_pos: 0 < r) (h_fifth: a * r^4 = 16) (h_ninth: a * r^8 = 4) : a * r^6 = 8 := by
  sorry

end seventh_term_geometric_seq_l84_84697


namespace x_coordinate_point_P_l84_84494

theorem x_coordinate_point_P (x y : ℝ) (h_on_parabola : y^2 = 4 * x) 
  (h_distance : dist (x, y) (1, 0) = 3) : x = 2 :=
sorry

end x_coordinate_point_P_l84_84494


namespace total_spent_on_toys_l84_84786

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_spent : ℝ := 12.30

theorem total_spent_on_toys : football_cost + marbles_cost = total_spent :=
by sorry

end total_spent_on_toys_l84_84786


namespace consecutive_integers_sum_l84_84703

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l84_84703


namespace completing_square_correct_l84_84090

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l84_84090


namespace number_of_yellow_marbles_l84_84838

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end number_of_yellow_marbles_l84_84838


namespace min_value_of_derivative_l84_84626

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1 / a) * x

noncomputable def f' (a : ℝ) : ℝ := 3 * 2^2 + 4 * a * 2 + (1 / a)

theorem min_value_of_derivative (a : ℝ) (h : a > 0) : 
  f' a ≥ 12 + 8 * Real.sqrt 2 :=
sorry

end min_value_of_derivative_l84_84626


namespace probability_even_sum_from_primes_l84_84656

theorem probability_even_sum_from_primes : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19] in
  let pairs := [(a, b) | a <- primes, b <- primes, a ≠ b] in
  let even_sum_pairs := [(a, b) | (a, b) <- pairs, (a + b) % 2 = 0] in
  (pairs.length > 0) → 
  ((even_sum_pairs.length : ℚ) / (pairs.length : ℚ) = 1) :=
by
  sorry

end probability_even_sum_from_primes_l84_84656


namespace calculate_expression_l84_84792

theorem calculate_expression:
  500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end calculate_expression_l84_84792


namespace apples_taken_from_each_basket_l84_84665

theorem apples_taken_from_each_basket (total_apples : ℕ) (baskets : ℕ) (remaining_apples_per_basket : ℕ) 
(h1 : total_apples = 64) (h2 : baskets = 4) (h3 : remaining_apples_per_basket = 13) : 
(total_apples - (remaining_apples_per_basket * baskets)) / baskets = 3 :=
sorry

end apples_taken_from_each_basket_l84_84665


namespace value_of_N_l84_84063

theorem value_of_N (a b c N : ℚ) (h1 : a + b + c = 120) (h2 : a - 10 = N) (h3 : 10 * b = N) (h4 : c - 10 = N) : N = 1100 / 21 := 
sorry

end value_of_N_l84_84063


namespace expected_value_two_point_distribution_l84_84515

theorem expected_value_two_point_distribution (X : Type) [Fintype X] 
  (p0 p1 : ℝ) (h0 : p0 + p1 = 1) (h1 : p1 - p0 = 0.4) : 
  ∑ x in ({1, 0} : Finset X), (if x = 1 then p1 else p0) * (x : ℝ) = 0.7 :=
by
  sorry

end expected_value_two_point_distribution_l84_84515


namespace slope_of_AB_l84_84818

theorem slope_of_AB (k : ℝ) (y1 y2 x1 x2 : ℝ) 
  (hP : (1, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1})
  (hPA_eq : ∀ x, (x, y1) ∈ {p : ℝ × ℝ | p.2 = k * p.1 - k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hPB_eq : ∀ x, (x, y2) ∈ {p : ℝ × ℝ | p.2 = -k * p.1 + k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hx1 : y1 = k * x1 - k + Real.sqrt 2) 
  (hx2 : y2 = -k * x2 + k + Real.sqrt 2) :
  ((y2 - y1) / (x2 - x1)) = -2 - 2 * Real.sqrt 2 :=
by
  sorry

end slope_of_AB_l84_84818


namespace find_x_l84_84485

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l84_84485


namespace probability_plane_contains_points_inside_octahedron_l84_84780

noncomputable def enhanced_octahedron_probability : ℚ :=
  let total_vertices := 18
  let total_ways := Nat.choose total_vertices 3
  let faces := 8
  let triangles_per_face := 4
  let unfavorable_ways := faces * triangles_per_face
  total_ways - unfavorable_ways

theorem probability_plane_contains_points_inside_octahedron :
  enhanced_octahedron_probability / (816 : ℚ) = 49 / 51 :=
sorry

end probability_plane_contains_points_inside_octahedron_l84_84780


namespace value_of_a_plus_b_l84_84805

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end value_of_a_plus_b_l84_84805


namespace monotonically_increasing_range_k_l84_84915

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_range_k :
  (∀ x > 1, deriv (f k) x ≥ 0) → k ≥ 1 :=
sorry

end monotonically_increasing_range_k_l84_84915


namespace number_of_permutations_l84_84821

def total_letters : ℕ := 10
def freq_s : ℕ := 3
def freq_t : ℕ := 2
def freq_i : ℕ := 2
def freq_a : ℕ := 1
def freq_c : ℕ := 1

theorem number_of_permutations : 
  (total_letters.factorial / (freq_s.factorial * freq_t.factorial * freq_i.factorial * freq_a.factorial * freq_c.factorial)) = 75600 :=
by
  sorry

end number_of_permutations_l84_84821


namespace rectangle_enclosed_by_lines_l84_84446

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l84_84446


namespace probability_of_team_with_2_girls_2_boys_l84_84068

open Nat

-- Define the combinatorics function for binomial coefficients
def binomial (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_of_team_with_2_girls_2_boys :
  let total_women := 8
  let total_men := 6
  let team_size := 4
  let ways_to_choose_2_girls := binomial total_women 2
  let ways_to_choose_2_boys := binomial total_men 2
  let total_ways_to_form_team := binomial (total_women + total_men) team_size
  let favorable_outcomes := ways_to_choose_2_girls * ways_to_choose_2_boys
  (favorable_outcomes : ℚ) / total_ways_to_form_team = 60 / 143 := 
by sorry

end probability_of_team_with_2_girls_2_boys_l84_84068


namespace shopper_total_payment_l84_84900

theorem shopper_total_payment :
  let original_price := 150
  let discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price := price_after_coupon * (1 + sales_tax_rate)
  final_price = 112.75 := by
{
  sorry
}

end shopper_total_payment_l84_84900


namespace tangent_of_11pi_over_4_l84_84933

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l84_84933


namespace multiples_of_7_between_20_and_150_l84_84822

def number_of_multiples_of_7_between (a b : ℕ) : ℕ :=
  (b / 7) - (a / 7) + (if a % 7 = 0 then 1 else 0)

theorem multiples_of_7_between_20_and_150 : number_of_multiples_of_7_between 21 147 = 19 := by
  sorry

end multiples_of_7_between_20_and_150_l84_84822


namespace verify_other_root_l84_84819

variable {a b c x : ℝ}

-- Given conditions
axiom distinct_non_zero_constants : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

axiom root_two : a * 2^2 - (a + b + c) * 2 + (b + c) = 0

-- Function under test
noncomputable def other_root (a b c : ℝ) : ℝ :=
  (b + c - a) / a

-- The goal statement
theorem verify_other_root :
  ∀ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) → (a * 2^2 - (a + b + c) * 2 + (b + c) = 0) → 
  (∀ x, (a * x^2 - (a + b + c) * x + (b + c) = 0) → (x = 2 ∨ x = (b + c - a) / a)) :=
by
  intros a b c h1 h2 x h3
  sorry

end verify_other_root_l84_84819


namespace find_integer_n_l84_84798

theorem find_integer_n : ∃ n, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ n = 9 :=   
by 
  -- The proof will be written here.
  sorry

end find_integer_n_l84_84798


namespace tan_of_11pi_over_4_is_neg1_l84_84929

noncomputable def tan_periodic : Real := 2 * Real.pi

theorem tan_of_11pi_over_4_is_neg1 :
  Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Angle normalization using periodicity of tangent function
  have h1 : Real.tan (11 * Real.pi / 4) = Real.tan (11 * Real.pi / 4 - 2 * Real.pi) := 
    by rw [Real.tan_periodic]
  -- Further normalization
  have h2 : 11 * Real.pi / 4 - 2 * Real.pi = 3 * Real.pi / 4 := sorry
  -- Evaluate tangent at the simplified angle
  have h3 : Real.tan (3 * Real.pi / 4) = -Real.tan (Real.pi / 4) := sorry
  -- Known value of tangent at common angle
  have h4 : Real.tan (Real.pi / 4) = 1 := by simpl tan
  rw [h2, h3, h4]
  norm_num

end tan_of_11pi_over_4_is_neg1_l84_84929


namespace cassie_nail_cutting_l84_84271

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l84_84271


namespace hexagon_chord_problem_l84_84117

-- Define the conditions of the problem
structure Hexagon :=
  (circumcircle : Type*)
  (inscribed : Prop)
  (AB BC CD : ℕ)
  (DE EF FA : ℕ)
  (chord_length_fraction_form : ℚ) 

-- Define the unique problem from given conditions and correct answer
theorem hexagon_chord_problem (hex : Hexagon) 
  (h1 : hex.inscribed)
  (h2 : hex.AB = 3) (h3 : hex.BC = 3) (h4 : hex.CD = 3)
  (h5 : hex.DE = 5) (h6 : hex.EF = 5) (h7 : hex.FA = 5)
  (h8 : hex.chord_length_fraction_form = 360 / 49) :
  let m := 360
  let n := 49
  m + n = 409 :=
by
  sorry

end hexagon_chord_problem_l84_84117


namespace sum_seven_consecutive_l84_84532

theorem sum_seven_consecutive (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 :=
by
  sorry

end sum_seven_consecutive_l84_84532


namespace polynomial_factorization_l84_84440

noncomputable def factorize_polynomial (a b : ℝ) : ℝ :=
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3

theorem polynomial_factorization (a b : ℝ) : 
  factorize_polynomial a b = -3 * a * b * (a - b)^2 := 
by
  sorry

end polynomial_factorization_l84_84440


namespace average_speed_is_correct_l84_84869

-- Definitions for the conditions
def speed_first_hour : ℕ := 140
def speed_second_hour : ℕ := 40
def total_distance : ℕ := speed_first_hour + speed_second_hour
def total_time : ℕ := 2

-- The statement we need to prove
theorem average_speed_is_correct : total_distance / total_time = 90 := by
  -- We would place the proof here
  sorry

end average_speed_is_correct_l84_84869


namespace cubic_roots_nature_l84_84599

-- Define the cubic polynomial function
def cubic_poly (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x - 4

-- Define the statement about the roots of the polynomial
theorem cubic_roots_nature :
  ∃ a b c : ℝ, cubic_poly a = 0 ∧ cubic_poly b = 0 ∧ cubic_poly c = 0 
  ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end cubic_roots_nature_l84_84599


namespace inequality_l84_84360

theorem inequality (A B : ℝ) (n : ℕ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hn : 1 ≤ n) : (A + B)^n ≤ 2^(n - 1) * (A^n + B^n) := 
  sorry

end inequality_l84_84360


namespace number_representation_fewer_sevens_exists_l84_84239

def representable_using_fewer_sevens (n : ℕ) : Prop :=
  ∃ (N : ℕ), let num := 7 * (10 ^ n - 1) / 9 in 
  N < n ∧ num = N

theorem number_representation_fewer_sevens_exists : ∃ (n : ℕ), representable_using_fewer_sevens n :=
sorry

end number_representation_fewer_sevens_exists_l84_84239


namespace arrangement_is_correct_l84_84010

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l84_84010


namespace number_of_yellow_marbles_l84_84837

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end number_of_yellow_marbles_l84_84837


namespace difference_of_squares_l84_84546

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) : a^2 - b^2 = 1200 := 
sorry

end difference_of_squares_l84_84546


namespace remainder_of_7_pow_205_mod_12_l84_84225

theorem remainder_of_7_pow_205_mod_12 : (7^205) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_205_mod_12_l84_84225


namespace curve_is_segment_l84_84210

noncomputable def parametric_curve := {t : ℝ // 0 ≤ t ∧ t ≤ 5}

def x (t : parametric_curve) : ℝ := 3 * t.val ^ 2 + 2
def y (t : parametric_curve) : ℝ := t.val ^ 2 - 1

def line_equation (x y : ℝ) := x - 3 * y - 5 = 0

theorem curve_is_segment :
  ∀ (t : parametric_curve), line_equation (x t) (y t) ∧ 
  2 ≤ x t ∧ x t ≤ 77 :=
by
  sorry

end curve_is_segment_l84_84210


namespace ratio_goats_sold_to_total_l84_84853

-- Define the conditions
variables (G S : ℕ) (total_revenue goat_sold : ℕ)
-- The ratio of goats to sheep is 5:7
axiom ratio_goats_to_sheep : G = (5/7) * S
-- The total number of sheep and goats is 360
axiom total_animals : G + S = 360
-- Mr. Mathews makes $7200 from selling some goats and 2/3 of the sheep
axiom selling_conditions : 40 * goat_sold + 30 * (2/3) * S = 7200

-- Prove the ratio of the number of goats sold to the total number of goats
theorem ratio_goats_sold_to_total : goat_sold / G = 1 / 2 := by
  sorry

end ratio_goats_sold_to_total_l84_84853


namespace expected_value_abs_diff_HT_l84_84517

noncomputable def expected_abs_diff_HT : ℚ :=
  let F : ℕ → ℚ := sorry -- Recurrence relation omitted for brevity
  F 0

theorem expected_value_abs_diff_HT :
  expected_abs_diff_HT = 24 / 7 :=
sorry

end expected_value_abs_diff_HT_l84_84517


namespace charles_total_money_l84_84428

-- Definitions based on the conditions in step a)
def number_of_pennies : ℕ := 6
def number_of_nickels : ℕ := 3
def value_of_penny : ℕ := 1
def value_of_nickel : ℕ := 5

-- Calculations in Lean terms
def total_pennies_value : ℕ := number_of_pennies * value_of_penny
def total_nickels_value : ℕ := number_of_nickels * value_of_nickel
def total_money : ℕ := total_pennies_value + total_nickels_value

-- The final proof statement based on step c)
theorem charles_total_money : total_money = 21 := by
  sorry

end charles_total_money_l84_84428


namespace count_k_for_lcm_problem_l84_84328

theorem count_k_for_lcm_problem :
  let k_values := {k : ℕ | ∃ (a b : ℕ), a ≤ 18 ∧ b = 36 ∧ k = 2^a * 3^b} in
  ∃! k_values, 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k_values) :=
by
  sorry

end count_k_for_lcm_problem_l84_84328


namespace black_marbles_count_l84_84547

theorem black_marbles_count :
  ∀ (white_marbles total_marbles : ℕ), 
  white_marbles = 19 → total_marbles = 37 → total_marbles - white_marbles = 18 :=
by
  intros white_marbles total_marbles h_white h_total
  sorry

end black_marbles_count_l84_84547


namespace option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l84_84183

noncomputable def triangle (A B C : ℝ) := A + B + C = 180

-- Define the conditions for options A, B, C, and D
def option_a := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = 3 * C
def option_b := ∀ A B C : ℝ, triangle A B C → A + B = C
def option_c := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = (1/2) * C
def option_d := ∀ A B C : ℝ, triangle A B C → ∃ x : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x

-- Define that option A does not form a right triangle
theorem option_a_not_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_a → A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 :=
sorry

-- Check that options B, C, and D do form right triangles
theorem option_b_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_b → C = 90 :=
sorry

theorem option_c_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_c → C = 90 :=
sorry

theorem option_d_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_d → C = 90 :=
sorry

end option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l84_84183


namespace penthouse_units_l84_84776

theorem penthouse_units (total_floors : ℕ) (regular_units_per_floor : ℕ) (penthouse_floors : ℕ) (total_units : ℕ) :
  total_floors = 23 →
  regular_units_per_floor = 12 →
  penthouse_floors = 2 →
  total_units = 256 →
  (total_units - (total_floors - penthouse_floors) * regular_units_per_floor) / penthouse_floors = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end penthouse_units_l84_84776


namespace fish_remaining_correct_l84_84666

def remaining_fish (jordan_caught : ℕ) (total_catch_lost_fraction : ℚ) : ℕ :=
  let perry_caught := 2 * jordan_caught
  let total_catch := jordan_caught + perry_caught
  let lost_catch := total_catch * total_catch_lost_fraction
  let remaining := total_catch - lost_catch
  remaining.nat_abs

theorem fish_remaining_correct : (remaining_fish 4 (1/4)) = 9 :=
by 
  sorry

end fish_remaining_correct_l84_84666


namespace blue_balls_in_box_l84_84771

theorem blue_balls_in_box (total_balls : ℕ) (p_two_blue : ℚ) (b : ℕ) 
  (h1 : total_balls = 12) (h2 : p_two_blue = 1/22) 
  (h3 : (↑b / 12) * (↑(b-1) / 11) = p_two_blue) : b = 3 :=
by {
  sorry
}

end blue_balls_in_box_l84_84771


namespace negative_represents_backward_l84_84050

-- Definitions based on conditions
def forward (distance : Int) : Int := distance
def backward (distance : Int) : Int := -distance

-- The mathematical equivalent proof problem
theorem negative_represents_backward
  (distance : Int)
  (h : forward distance = 5) :
  backward distance = -5 :=
sorry

end negative_represents_backward_l84_84050


namespace rectangles_from_lines_l84_84447

theorem rectangles_from_lines : 
  (∃ (h_lines : ℕ) (v_lines : ℕ), (h_lines = 5 ∧ v_lines = 4) ∧ (nat.choose h_lines 2 * nat.choose v_lines 2 = 60)) :=
by
  let h_lines := 5
  let v_lines := 4
  have h_choice := nat.choose h_lines 2
  have v_choice := nat.choose v_lines 2
  have answer := h_choice * v_choice
  exact ⟨h_lines, v_lines, ⟨rfl, rfl⟩, rfl⟩

end rectangles_from_lines_l84_84447


namespace prove_zero_l84_84542

variable {a b c : ℝ}

theorem prove_zero (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
by
  sorry

end prove_zero_l84_84542


namespace remainder_21_pow_2051_mod_29_l84_84224

theorem remainder_21_pow_2051_mod_29 :
  ∀ (a : ℤ), (21^4 ≡ 1 [MOD 29]) -> (2051 = 4 * 512 + 3) -> (21^3 ≡ 15 [MOD 29]) -> (21^2051 ≡ 15 [MOD 29]) :=
by
  intros a h1 h2 h3
  sorry

end remainder_21_pow_2051_mod_29_l84_84224


namespace union_of_A_and_B_l84_84334

open Set

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end union_of_A_and_B_l84_84334


namespace none_of_these_true_l84_84624

def op_star (a b : ℕ) := b ^ a -- Define the binary operation

theorem none_of_these_true :
  ¬ (∀ a b : ℕ, 0 < a ∧ 0 < b → op_star a b = op_star b a) ∧
  ¬ (∀ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c → op_star a (op_star b c) = op_star (op_star a b) c) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → (op_star a b) ^ n = op_star n (op_star a b)) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → op_star a (b ^ n) = op_star n (op_star b a)) :=
sorry

end none_of_these_true_l84_84624


namespace sufficient_but_not_necessary_condition_l84_84385

noncomputable def f (a x : ℝ) := x^2 + 2 * a * x - 2

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, x ≤ -2 → deriv (f a) x ≤ 0) ↔ a = 2 :=
sorry

end sufficient_but_not_necessary_condition_l84_84385


namespace sequence_term_l84_84150

theorem sequence_term (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, (n + 1) * a n = 2 * n * a (n + 1)) : 
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
by
  sorry

end sequence_term_l84_84150


namespace rectangle_breadth_l84_84701

theorem rectangle_breadth (sq_area : ℝ) (rect_area : ℝ) (radius_rect_relation : ℝ → ℝ) 
  (rect_length_relation : ℝ → ℝ) (breadth_correct: ℝ) : 
  (sq_area = 3600) →
  (rect_area = 240) →
  (forall r, radius_rect_relation r = r) →
  (forall r, rect_length_relation r = (2/5) * r) →
  breadth_correct = 10 :=
by
  intros h_sq_area h_rect_area h_radius_rect h_rect_length
  sorry

end rectangle_breadth_l84_84701


namespace find_missing_ratio_l84_84049

def compounded_ratio (x y : ℚ) : ℚ := (x / y) * (6 / 11) * (11 / 2)

theorem find_missing_ratio (x y : ℚ) (h : compounded_ratio x y = 2) :
  x / y = 2 / 3 :=
sorry

end find_missing_ratio_l84_84049


namespace factorial_fraction_eq_l84_84269

theorem factorial_fraction_eq :
  (15.factorial / (6.factorial * 9.factorial) = 5005) := 
sorry

end factorial_fraction_eq_l84_84269


namespace consecutive_integers_product_l84_84726

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l84_84726


namespace num_clients_visited_garage_l84_84781

theorem num_clients_visited_garage :
  ∃ (num_clients : ℕ), num_clients = 24 ∧
    ∀ (num_cars selections_per_car selections_per_client : ℕ),
        num_cars = 16 → selections_per_car = 3 → selections_per_client = 2 →
        (num_cars * selections_per_car) / selections_per_client = num_clients :=
by
  sorry

end num_clients_visited_garage_l84_84781


namespace find_a_range_l84_84468

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := 3 * Real.exp x + a

theorem find_a_range (a : ℝ) :
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x > g x a) → a < Real.exp 2 :=
by
  sorry

end find_a_range_l84_84468


namespace part1_part2_l84_84071

def custom_op (a b : ℤ) : ℤ := a^2 - b + a * b

theorem part1  : custom_op (-3) (-2) = 17 := by
  sorry

theorem part2 : custom_op (-2) (custom_op (-3) (-2)) = -47 := by
  sorry

end part1_part2_l84_84071


namespace find_num_officers_l84_84342

noncomputable def num_officers (O : ℕ) : Prop :=
  let avg_salary_all := 120
  let avg_salary_officers := 440
  let avg_salary_non_officers := 110
  let num_non_officers := 480
  let total_salary :=
    avg_salary_all * (O + num_non_officers)
  let salary_officers :=
    avg_salary_officers * O
  let salary_non_officers :=
    avg_salary_non_officers * num_non_officers
  total_salary = salary_officers + salary_non_officers

theorem find_num_officers : num_officers 15 :=
sorry

end find_num_officers_l84_84342


namespace sum_of_three_consecutive_integers_product_336_l84_84720

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l84_84720


namespace dressing_q_vinegar_percentage_l84_84527

/-- 
Given:
1. P is 30% vinegar and 70% oil.
2. Q is V% vinegar and the rest is oil.
3. The new dressing is produced from 10% of P and 90% of Q and is 12% vinegar.
Prove:
The percentage of vinegar in dressing Q is 10%.
-/
theorem dressing_q_vinegar_percentage (V : ℝ) (h : 0.10 * 0.30 + 0.90 * V = 0.12) : V = 0.10 :=
by 
    sorry

end dressing_q_vinegar_percentage_l84_84527


namespace trucks_have_160_containers_per_truck_l84_84734

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l84_84734


namespace sin_25_over_6_pi_l84_84443

noncomputable def sin_value : ℝ :=
  Real.sin (25 / 6 * Real.pi)

theorem sin_25_over_6_pi : sin_value = 1 / 2 := by
  sorry

end sin_25_over_6_pi_l84_84443


namespace total_nails_to_cut_l84_84275

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l84_84275


namespace lucy_picked_more_l84_84850

variable (Mary Peter Lucy : ℕ)
variable (Mary_amt Peter_amt Lucy_amt : ℕ)

-- Conditions
def mary_amount : Mary_amt = 12 := sorry
def twice_as_peter : Mary_amt = 2 * Peter_amt := sorry
def total_picked : Mary_amt + Peter_amt + Lucy_amt = 26 := sorry

-- Statement to Prove
theorem lucy_picked_more (h1: Mary_amt = 12) (h2: Mary_amt = 2 * Peter_amt) (h3: Mary_amt + Peter_amt + Lucy_amt = 26) :
  Lucy_amt - Peter_amt = 2 := 
sorry

end lucy_picked_more_l84_84850


namespace area_of_region_l84_84999

def plane_region (x y : ℝ) : Prop := |x| ≤ 1 ∧ |y| ≤ 1

def inequality_holds (a b : ℝ) : Prop := ∀ x y : ℝ, plane_region x y → a * x - 2 * b * y ≤ 2

theorem area_of_region (a b : ℝ) (h : inequality_holds a b) : 
  (-2 ≤ a ∧ a ≤ 2) ∧ (-1 ≤ b ∧ b ≤ 1) ∧ (4 * 2 = 8) :=
sorry

end area_of_region_l84_84999


namespace scientific_notation_9600000_l84_84054

theorem scientific_notation_9600000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 9600000 = a * 10 ^ n ∧ a = 9.6 ∧ n = 6 :=
by
  exists 9.6
  exists 6
  simp
  sorry

end scientific_notation_9600000_l84_84054


namespace problem1_problem2_problem3_problem4_l84_84886

-- Problem 1
theorem problem1 : -9 + 5 - 11 + 16 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : -9 + 5 - (-6) - 18 / (-3) = 8 :=
by
  sorry

-- Problem 3
theorem problem3 : -2^2 - ((-3) * (-4 / 3) - (-2)^3) = -16 :=
by
  sorry

-- Problem 4
theorem problem4 : (59 - (7 / 9 - 11 / 12 + 1 / 6) * (-6)^2) / (-7)^2 = 58 / 49 :=
by
  sorry

end problem1_problem2_problem3_problem4_l84_84886


namespace parametric_to_line_segment_l84_84956

theorem parametric_to_line_segment :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 5 →
  ∃ x y : ℝ, x = 3 * t^2 + 2 ∧ y = t^2 - 1 ∧ (x - 3 * y = 5) ∧ (-1 ≤ y ∧ y ≤ 24) :=
by
  sorry

end parametric_to_line_segment_l84_84956


namespace largest_sphere_radius_on_torus_l84_84783

theorem largest_sphere_radius_on_torus :
  ∀ r : ℝ, 16 + (r - 1)^2 = (r + 2)^2 → r = 13 / 6 :=
by
  intro r
  intro h
  sorry

end largest_sphere_radius_on_torus_l84_84783


namespace poly_factorable_l84_84369

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l84_84369


namespace remainder_of_98_times_102_divided_by_9_l84_84226

theorem remainder_of_98_times_102_divided_by_9 : (98 * 102) % 9 = 6 :=
by
  sorry

end remainder_of_98_times_102_divided_by_9_l84_84226


namespace fewest_keystrokes_One_to_410_l84_84577

noncomputable def fewest_keystrokes (start : ℕ) (target : ℕ) : ℕ :=
if target = 410 then 10 else sorry

theorem fewest_keystrokes_One_to_410 : fewest_keystrokes 1 410 = 10 :=
by
  sorry

end fewest_keystrokes_One_to_410_l84_84577


namespace johns_pace_l84_84509

variable {J : ℝ} -- John's pace during his final push

theorem johns_pace
  (steve_speed : ℝ := 3.8)
  (initial_gap : ℝ := 15)
  (finish_gap : ℝ := 2)
  (time : ℝ := 42.5)
  (steve_covered : ℝ := steve_speed * time)
  (john_covered : ℝ := steve_covered + initial_gap + finish_gap)
  (johns_pace_equation : J * time = john_covered) :
  J = 4.188 :=
by
  sorry

end johns_pace_l84_84509


namespace domain_of_f_l84_84223

def denominator (x : ℝ) : ℝ := x^2 - 4 * x + 3

def is_defined (x : ℝ) : Prop := denominator x ≠ 0

theorem domain_of_f :
  {x : ℝ // is_defined x} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l84_84223


namespace one_over_m_add_one_over_n_l84_84730

theorem one_over_m_add_one_over_n (m n : ℕ) (h_sum : m + n = 80) (h_hcf : Nat.gcd m n = 6) (h_lcm : Nat.lcm m n = 210) : 
  1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 15.75 :=
by
  sorry

end one_over_m_add_one_over_n_l84_84730


namespace swimming_class_attendance_l84_84733

theorem swimming_class_attendance (total_students : ℕ) (chess_percentage : ℝ) (swimming_percentage : ℝ) 
  (H1 : total_students = 1000) 
  (H2 : chess_percentage = 0.20) 
  (H3 : swimming_percentage = 0.10) : 
  200 * 0.10 = 20 := 
by sorry

end swimming_class_attendance_l84_84733


namespace distance_to_directrix_l84_84948

theorem distance_to_directrix (x y d : ℝ) (a b c : ℝ) (F1 F2 M : ℝ × ℝ)
  (h_ellipse : x^2 / 25 + y^2 / 9 = 1)
  (h_a : a = 5)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_M_on_ellipse : M.snd^2 / (a^2) + M.fst^2 / (b^2) = 1)
  (h_dist_F1M : dist M F1 = 8) :
  d = 5 / 2 :=
by
  sorry

end distance_to_directrix_l84_84948


namespace compute_sum_of_squares_roots_l84_84912

-- p, q, and r are roots of 3*x^3 - 2*x^2 + 6*x + 15 = 0.
def P (x : ℝ) : Prop := 3*x^3 - 2*x^2 + 6*x + 15 = 0

theorem compute_sum_of_squares_roots :
  ∀ p q r : ℝ, P p ∧ P q ∧ P r → p^2 + q^2 + r^2 = -32 / 9 :=
by
  intros p q r h
  sorry

end compute_sum_of_squares_roots_l84_84912


namespace triangle_angle_solution_exists_l84_84767

noncomputable def possible_angles (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (A = 120 ∨ B = 120 ∨ C = 120) ∧
  (
    ((A = 40 ∧ B = 20) ∨ (A = 20 ∧ B = 40)) ∨
    ((A = 45 ∧ B = 15) ∨ (A = 15 ∧ B = 45))
  )
  
theorem triangle_angle_solution_exists :
  ∃ A B C : ℝ, possible_angles A B C :=
sorry

end triangle_angle_solution_exists_l84_84767


namespace middle_number_is_eight_l84_84217

theorem middle_number_is_eight
    (x y z : ℕ)
    (h1 : x + y = 14)
    (h2 : x + z = 20)
    (h3 : y + z = 22) :
    y = 8 := by
  sorry

end middle_number_is_eight_l84_84217


namespace air_conditioned_rooms_fraction_l84_84359

theorem air_conditioned_rooms_fraction (R A : ℝ) (h1 : 3/4 * R = 3/4 * R - 1/4 * R)
                                        (h2 : 2/3 * A = 2/3 * A - 1/3 * A)
                                        (h3 : 1/3 * A = 0.8 * 1/4 * R) :
    A / R = 3 / 5 :=
by
  -- Proof content goes here
  sorry

end air_conditioned_rooms_fraction_l84_84359


namespace set_complement_union_l84_84991

-- Definitions of the sets
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

-- The statement to prove
theorem set_complement_union : (U \ A) ∪ (U \ B) = {1, 4, 5} :=
by sorry

end set_complement_union_l84_84991


namespace find_original_price_l84_84581

-- Define the conditions for the problem
def original_price (P : ℝ) : Prop :=
  0.90 * P = 1620

-- Prove the original price P
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1800 :=
by
  -- The proof goes here
  sorry

end find_original_price_l84_84581


namespace yoongi_age_l84_84658

theorem yoongi_age (Y H : ℕ) (h1 : Y + H = 16) (h2 : Y = H + 2) : Y = 9 :=
by
  sorry

end yoongi_age_l84_84658


namespace solve_quadratic_eq_l84_84861

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2 * x = 1) : x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end solve_quadratic_eq_l84_84861


namespace min_value_ineq_l84_84675

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end min_value_ineq_l84_84675


namespace sum_of_numbers_l84_84059

noncomputable def sum_two_numbers (x y : ℝ) : ℝ :=
  x + y

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  sum_two_numbers x y = (16 * Real.sqrt 3) / 3 := 
by 
  sorry

end sum_of_numbers_l84_84059


namespace consecutive_integers_sum_l84_84706

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l84_84706


namespace value_of_x_l84_84478

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l84_84478


namespace exists_infinitely_many_N_l84_84367

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l84_84367


namespace bernardo_larger_than_silvia_l84_84790

theorem bernardo_larger_than_silvia :
  let bernardo_candidates := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
      silvia_candidates := {1, 2, 3, 4, 5, 6, 7}
      bernardo_possibilities := choose 10 3
      silvia_possibilities := choose 7 3
      favorable_cases := 25 in
  (favorable_cases : ℚ) / (bernardo_possibilities : ℚ) = 5 / 12 :=
sorry

end bernardo_larger_than_silvia_l84_84790


namespace nonnegative_solution_exists_l84_84453

theorem nonnegative_solution_exists
  (a b c d n : ℕ)
  (h_npos : 0 < n)
  (h_gcd_abc : Nat.gcd (Nat.gcd a b) c = 1)
  (h_gcd_ab : Nat.gcd a b = d)
  (h_conds : n > a * b / d + c * d - a - b - c) :
  ∃ x y z : ℕ, a * x + b * y + c * z = n := 
by
  sorry

end nonnegative_solution_exists_l84_84453


namespace factorial_fraction_value_l84_84268

theorem factorial_fraction_value :
  (15.factorial / (6.factorial * 9.factorial) = 5005) :=
by
  sorry

end factorial_fraction_value_l84_84268


namespace lcm_18_27_l84_84077

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end lcm_18_27_l84_84077


namespace restore_original_expression_l84_84661

-- Define the altered product and correct restored products
def original_expression_1 := 4 * 5 * 4 * 7 * 4
def original_expression_2 := 4 * 7 * 4 * 5 * 4
def altered_product := 2247
def corrected_product := 2240

-- Statement that proves the corrected restored product given the altered product
theorem restore_original_expression :
  (4 * 5 * 4 * 7 * 4 = corrected_product ∨ 4 * 7 * 4 * 5 * 4 = corrected_product) :=
sorry

end restore_original_expression_l84_84661


namespace trajectory_equation_find_m_value_l84_84947

def point (α : Type) := (α × α)
def fixed_points (α : Type) := point α

noncomputable def slopes (x y : ℝ) : ℝ := y / x

theorem trajectory_equation (x y : ℝ) (P : point ℝ) (A B : fixed_points ℝ)
  (k1 k2 : ℝ) (hk : k1 * k2 = -1/4) :
  A = (-2, 0) → B = (2, 0) →
  P = (x, y) → 
  slopes (x + 2) y * slopes (x - 2) y = -1/4 →
  (x^2 / 4) + y^2 = 1 :=
sorry

theorem find_m_value (m x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) (hx : (4 * k^2) + 1 - m^2 > 0)
  (hroots_sum : x₁ + x₂ = -((8 * k * m) / ((4 * k^2) + 1)))
  (hroots_prod : x₁ * x₂ = (4 * m^2 - 4) / ((4 * k^2) + 1))
  (hperp : x₁ * x₂ + y₁ * y₂ = 0) :
  y₁ = k * x₁ + m → y₂ = k * x₂ + m →
  m^2 = 4/5 * (k^2 + 1) →
  m = 2 ∨ m = -2 :=
sorry

end trajectory_equation_find_m_value_l84_84947


namespace geometric_series_sum_correct_l84_84598

def geometric_series_sum (a r n : ℕ) : ℤ :=
  a * ((Int.pow r n - 1) / (r - 1))

theorem geometric_series_sum_correct :
  geometric_series_sum 2 (-2) 11 = 1366 := by
  sorry

end geometric_series_sum_correct_l84_84598


namespace solve_expression_l84_84474

theorem solve_expression (x : ℝ) (h : 3 * x - 5 = 10 * x + 9) : 4 * (x + 7) = 20 :=
by
  sorry

end solve_expression_l84_84474


namespace simplify_and_compute_l84_84430

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_l84_84430


namespace find_two_digit_number_l84_84414

def is_positive (n : ℕ) := n > 0
def is_even (n : ℕ) := n % 2 = 0
def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def product_of_digits_is_square (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  ∃ k : ℕ, (tens * units) = k * k

theorem find_two_digit_number (N : ℕ) 
  (h_pos : is_positive N) 
  (h_ev : is_even N) 
  (h_mult_9 : is_multiple_of_9 N)
  (h_prod_square : product_of_digits_is_square N) 
: N = 90 := by 
  sorry

end find_two_digit_number_l84_84414


namespace find_g_of_one_fifth_l84_84032

variable {g : ℝ → ℝ}

theorem find_g_of_one_fifth (h₀ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1)
    (h₁ : g 0 = 0)
    (h₂ : ∀ {x y}, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y)
    (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x)
    (h₄ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2) :
  g (1 / 5) = 1 / 4 :=
by
  sorry

end find_g_of_one_fifth_l84_84032


namespace sum_of_three_consecutive_integers_product_336_l84_84715

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l84_84715


namespace redistribute_oil_l84_84741

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l84_84741


namespace find_value_of_a_l84_84164

theorem find_value_of_a (a b : ℝ) (h1 : ∀ x, (2 < x ∧ x < 4) ↔ (a - b < x ∧ x < a + b)) : a = 3 := by
  sorry

end find_value_of_a_l84_84164


namespace number_of_common_divisors_l84_84959

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l84_84959


namespace max_z_value_l84_84130

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z = 13 / 3 := 
sorry

end max_z_value_l84_84130


namespace simplify_and_evaluate_expression_l84_84690

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  ( (x^2 - 1) / (x^2 - 6 * x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) ) = - (Real.sqrt 2 / 2) :=
  sorry

end simplify_and_evaluate_expression_l84_84690


namespace sum_first_3m_terms_l84_84062

variable (m : ℕ) (a₁ d : ℕ)

def S (n : ℕ) := n * a₁ + (n * (n - 1)) / 2 * d

-- Given conditions
axiom sum_first_m_terms : S m = 0
axiom sum_first_2m_terms : S (2 * m) = 0

-- Theorem to be proved
theorem sum_first_3m_terms : S (3 * m) = 210 :=
by
  sorry

end sum_first_3m_terms_l84_84062


namespace boat_speed_in_still_water_l84_84128

theorem boat_speed_in_still_water (v s : ℝ) (h1 : v + s = 15) (h2 : v - s = 7) : v = 11 := 
by
  sorry

end boat_speed_in_still_water_l84_84128


namespace base_rate_second_telephone_company_l84_84750

theorem base_rate_second_telephone_company : 
  ∃ B : ℝ, (11 + 20 * 0.25 = B + 20 * 0.20) ∧ B = 12 := by
  sorry

end base_rate_second_telephone_company_l84_84750


namespace river_width_after_30_seconds_l84_84526

noncomputable def width_of_river (initial_width : ℝ) (width_increase_rate : ℝ) (rowing_rate : ℝ) (time_taken : ℝ) : ℝ :=
  initial_width + (time_taken * rowing_rate * (width_increase_rate / 10))

theorem river_width_after_30_seconds :
  width_of_river 50 2 5 30 = 80 :=
by
  -- it suffices to check the calculations here
  sorry

end river_width_after_30_seconds_l84_84526


namespace value_of_expression_l84_84396

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end value_of_expression_l84_84396


namespace probability_two_vertices_endpoints_l84_84745

theorem probability_two_vertices_endpoints (V E : Type) [Fintype V] [DecidableEq V] 
  (dodecahedron : Graph V E) (h1 : Fintype.card V = 20)
  (h2 : ∀ v : V, Fintype.card (dodecahedron.neighbors v) = 3)
  (h3 : Fintype.card E = 30) :
  (∃ A B : V, A ≠ B ∧ (A, B) ∈ dodecahedron.edgeSet) → 
  (∃ p : ℚ, p = 3/19) := 
sorry

end probability_two_vertices_endpoints_l84_84745


namespace negation_correct_l84_84541

-- Definitions needed from the conditions:
def is_positive (m : ℝ) : Prop := m > 0
def square (m : ℝ) : ℝ := m * m

-- The original proposition
def original_proposition (m : ℝ) : Prop := is_positive m → square m > 0

-- The negation of the proposition
def negated_proposition (m : ℝ) : Prop := ¬is_positive m → ¬(square m > 0)

-- The theorem to prove that the negated proposition is the negation of the original proposition
theorem negation_correct (m : ℝ) : (original_proposition m) ↔ (negated_proposition m) :=
by
  sorry

end negation_correct_l84_84541


namespace problem_1_problem_2_l84_84469

-- Proposition p
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2 * a * x + 2 - a)

-- Proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2 * x + a ≥ 0

-- Problem 1: Prove that if p is true then a ≤ -2 or a ≥ 1
theorem problem_1 (a : ℝ) (hp : p a) : a ≤ -2 ∨ a ≥ 1 := sorry

-- Problem 2: Prove that if p ∨ q is true then a ≤ -2 or a ≥ 0
theorem problem_2 (a : ℝ) (hpq : p a ∨ q a) : a ≤ -2 ∨ a ≥ 0 := sorry

end problem_1_problem_2_l84_84469


namespace length_of_leg_of_isosceles_right_triangle_l84_84702

def is_isosceles_right_triangle (a b h : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = h^2

def median_to_hypotenuse (m h : ℝ) : Prop :=
  m = h / 2

theorem length_of_leg_of_isosceles_right_triangle (m : ℝ) (h a : ℝ)
  (h1 : median_to_hypotenuse m h)
  (h2 : h = 2 * m)
  (h3 : is_isosceles_right_triangle a a h) :
  a = 15 * Real.sqrt 2 :=
by
  -- Skipping the proof
  sorry

end length_of_leg_of_isosceles_right_triangle_l84_84702


namespace find_a_l84_84953

noncomputable def geometric_sum_expression (n : ℕ) (a : ℝ) : ℝ :=
  3 * 2^n + a

theorem find_a (a : ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = geometric_sum_expression n a) → a = -3 :=
by
  sorry

end find_a_l84_84953


namespace team_A_leading_after_3_games_prob_team_B_wins_3_2_prob_l84_84208

noncomputable def prob_team_A_wins_game : ℝ := 0.60
def prob_team_B_wins_game : ℝ := 1 - prob_team_A_wins_game

def prob_team_A_leading_after_3_games : ℝ :=
  (prob_team_A_wins_game ^ 3) * prob_team_B_wins_game + 
  (choose 3 2) * (prob_team_A_wins_game ^ 2) * prob_team_B_wins_game

theorem team_A_leading_after_3_games_prob :
  prob_team_A_leading_after_3_games = 0.648 :=
by sorry

def prob_team_B_wins_3_2 : ℝ :=
  (choose 4 2) * (prob_team_A_wins_game ^ 2) * (prob_team_B_wins_game ^ 2) * prob_team_B_wins_game

theorem team_B_wins_3_2_prob :
  prob_team_B_wins_3_2 = 0.138 :=
by sorry

end team_A_leading_after_3_games_prob_team_B_wins_3_2_prob_l84_84208


namespace nonzero_roots_ratio_l84_84213

theorem nonzero_roots_ratio (m : ℝ) (h : m ≠ 0) :
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ r + s = 4 ∧ r * s = m) → m = 3 :=
by 
  intro h_exists
  obtain ⟨r, s, hr_ne_zero, hs_ne_zero, h_ratio, h_sum, h_prod⟩ := h_exists
  sorry

end nonzero_roots_ratio_l84_84213


namespace game_cost_l84_84879

theorem game_cost
    (total_earnings : ℕ)
    (expenses : ℕ)
    (games_bought : ℕ)
    (remaining_money := total_earnings - expenses)
    (cost_per_game := remaining_money / games_bought)
    (h1 : total_earnings = 104)
    (h2 : expenses = 41)
    (h3 : games_bought = 7) :
    cost_per_game = 9 := by
  sorry

end game_cost_l84_84879


namespace find_x_l84_84169

variable (a b x : ℝ)

def condition1 : Prop := a / b = 5 / 4
def condition2 : Prop := (4 * a + x * b) / (4 * a - x * b) = 4

theorem find_x (h1 : condition1 a b) (h2 : condition2 a b x) : x = 3 :=
  sorry

end find_x_l84_84169


namespace eccentricity_of_hyperbola_l84_84955

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : ℝ :=
  (3 * Real.sqrt 7) / 7

-- Ensure the function returns the correct eccentricity
theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : hyperbola_eccentricity a b c ha hb h = (3 * Real.sqrt 7) / 7 :=
sorry

end eccentricity_of_hyperbola_l84_84955


namespace num_four_digit_36_combinations_l84_84635

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l84_84635


namespace distance_from_point_to_parabola_directrix_l84_84319

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l84_84319


namespace difference_of_sums_l84_84202

open Finset

def set_A : Finset ℕ := filter (λ x, x % 2 = 0) (range (50 + 1)) ∩ Icc 2 50
def set_B : Finset ℕ := filter (λ x, x % 2 = 0) (range (150 + 1)) ∩ Icc 102 150

noncomputable def sum_set_A : ℕ := set_A.sum id
noncomputable def sum_set_B : ℕ := set_B.sum id

theorem difference_of_sums : sum_set_B - sum_set_A = 2500 :=
by
  unfold set_A set_B sum_set_A sum_set_B
  sorry

end difference_of_sums_l84_84202


namespace intersection_line_eq_l84_84139

-- Definitions of the circles
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*y - 6 = 0

-- The theorem stating that the equation of the line passing through their intersection points is x = y
theorem intersection_line_eq (x y : ℝ) :
  (circle1 x y → circle2 x y → x = y) := 
by
  intro h1 h2
  sorry

end intersection_line_eq_l84_84139


namespace line_equation_through_point_line_equation_sum_of_intercepts_l84_84022

theorem line_equation_through_point (x y : ℝ) (h : y = 2 * x + 5)
  (hx : x = -2) (hy : y = 1) : 2 * x - y + 5 = 0 :=
by {
  sorry
}

theorem line_equation_sum_of_intercepts (x y : ℝ) (h : y = 2 * x + 6)
  (hx : x = -3) (hy : y = 3) : 2 * x - y + 6 = 0 :=
by {
  sorry
}

end line_equation_through_point_line_equation_sum_of_intercepts_l84_84022


namespace no_polynomials_exist_l84_84131

open Polynomial

theorem no_polynomials_exist
  (a b : Polynomial ℂ) (c d : Polynomial ℂ) :
  ¬ (∀ x y : ℂ, 1 + x * y + x^2 * y^2 = a.eval x * c.eval y + b.eval x * d.eval y) :=
sorry

end no_polynomials_exist_l84_84131


namespace complement_intersection_example_l84_84166

open Set

theorem complement_intersection_example
  (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 4})
  (hB : B = {2, 3}) :
  (U \ A) ∩ B = {2} :=
by
  sorry

end complement_intersection_example_l84_84166


namespace markup_percentage_l84_84413

-- Define the purchase price and the gross profit
def purchase_price : ℝ := 54
def gross_profit : ℝ := 18

-- Define the sale price after discount
def sale_discount : ℝ := 0.8

-- Given that the sale price after the discount is purchase_price + gross_profit
theorem markup_percentage (M : ℝ) (SP : ℝ) : 
  SP = purchase_price * (1 + M / 100) → -- selling price as function of markup
  (SP * sale_discount = purchase_price + gross_profit) → -- sale price after 20% discount
  M = 66.67 := 
by
  -- sorry to skip the proof
  sorry

end markup_percentage_l84_84413


namespace simplify_and_compute_l84_84429

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_l84_84429


namespace perimeter_correct_l84_84376

-- Definitions based on the conditions
def large_rectangle_area : ℕ := 12 * 12
def shaded_rectangle_area : ℕ := 6 * 4
def non_shaded_area : ℕ := large_rectangle_area - shaded_rectangle_area
def perimeter_of_non_shaded_region : ℕ := 2 * ((12 - 6) + (12 - 4))

-- The theorem to prove
theorem perimeter_correct (large_rectangle_area_eq : large_rectangle_area = 144) :
  perimeter_of_non_shaded_region = 28 :=
by
  sorry

end perimeter_correct_l84_84376


namespace calculate_sequences_l84_84151

-- Definitions of sequences and constants
def a (n : ℕ) := 2 * n + 1
def b (n : ℕ) := 3 ^ n
def S (n : ℕ) := n * (n + 2)
def T (n : ℕ) := (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))

-- Hypotheses and proofs
theorem calculate_sequences (d : ℕ) (a1 : ℕ) (h_d : d = 2) (h_a1 : a1 = 3) :
  ∀ n, (a n = 2 * n + 1) ∧ (b 1 = a 1) ∧ (b 2 = a 4) ∧ (b 3 = a 13) ∧ (b n = 3 ^ n) ∧
  (S n = n * (n + 2)) ∧ (T n = (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by
  intros
  -- Skipping proof steps with sorry
  sorry

end calculate_sequences_l84_84151


namespace sufficient_not_necessary_l84_84825

variable (x : ℝ)
def p := x^2 > 4
def q := x > 2

theorem sufficient_not_necessary : (∀ x, q x -> p x) ∧ ¬ (∀ x, p x -> q x) :=
by sorry

end sufficient_not_necessary_l84_84825


namespace odd_and_monotonic_l84_84420

-- Definitions based on the conditions identified
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_monotonic_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x ≤ f y
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement without the proof
theorem odd_and_monotonic :
  is_odd f ∧ is_monotonic_increasing f :=
sorry

end odd_and_monotonic_l84_84420


namespace negation_of_p_l84_84459

theorem negation_of_p :
  ¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_p_l84_84459


namespace find_constants_l84_84137

theorem find_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ -2 →
    (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) = 
    P / (x - 1) + Q / (x - 4) + R / (x + 2))
  → (P = 2 / 3 ∧ Q = 8 / 9 ∧ R = -5 / 9) :=
by
  sorry

end find_constants_l84_84137


namespace simon_age_in_2010_l84_84347

theorem simon_age_in_2010 :
  ∀ (s j : ℕ), (j = 16 → (j + 24 = s) → j + (2010 - 2005) + 24 = 45) :=
by 
  intros s j h1 h2 
  sorry

end simon_age_in_2010_l84_84347


namespace total_number_of_shells_l84_84356

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end total_number_of_shells_l84_84356


namespace total_votes_cast_l84_84353

theorem total_votes_cast (S : ℝ) (x : ℝ) (h1 : S = 120) (h2 : S = 0.72 * x - 0.28 * x) : x = 273 := by
  sorry

end total_votes_cast_l84_84353


namespace count_four_digit_integers_with_3_and_6_l84_84633

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l84_84633


namespace boyden_family_tickets_l84_84520

theorem boyden_family_tickets (child_ticket_cost : ℕ) (adult_ticket_cost : ℕ) (total_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  adult_ticket_cost = child_ticket_cost + 6 →
  total_cost = 77 →
  adult_ticket_cost = 19 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost →
  num_adults + num_children = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end boyden_family_tickets_l84_84520


namespace polygon_eq_quadrilateral_l84_84830

theorem polygon_eq_quadrilateral (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 := 
sorry

end polygon_eq_quadrilateral_l84_84830


namespace correct_power_functions_l84_84229

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, x ≠ 0 → f x = k * x^n

def f1 (x : ℝ) : ℝ := x^2 + 2
def f2 (x : ℝ) : ℝ := x^(1 / 2)
def f3 (x : ℝ) : ℝ := 2 * x^3
def f4 (x : ℝ) : ℝ := x^(3 / 4)
def f5 (x : ℝ) : ℝ := x^(1 / 3) + 1

theorem correct_power_functions :
  {f2, f4} = {f : ℝ → ℝ | is_power_function f} ∩ {f2, f4, f1, f3, f5} :=
by
  sorry

end correct_power_functions_l84_84229


namespace positive_distinct_solutions_conditons_l84_84849

-- Definitions corresponding to the conditions in the problem
variables {x y z a b : ℝ}

-- The statement articulates the condition
theorem positive_distinct_solutions_conditons (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = b^2) (h3 : xy = z^2) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x ≠ y) (h8 : y ≠ z) (h9 : x ≠ z) : 
  b^2 ≥ a^2 / 2 :=
sorry

end positive_distinct_solutions_conditons_l84_84849


namespace length_first_train_l84_84553

noncomputable def length_second_train : ℝ := 200
noncomputable def speed_first_train_kmh : ℝ := 42
noncomputable def speed_second_train_kmh : ℝ := 30
noncomputable def time_seconds : ℝ := 14.998800095992321

noncomputable def speed_first_train_ms : ℝ := speed_first_train_kmh * 1000 / 3600
noncomputable def speed_second_train_ms : ℝ := speed_second_train_kmh * 1000 / 3600

noncomputable def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms
noncomputable def combined_length : ℝ := relative_speed * time_seconds

theorem length_first_train : combined_length - length_second_train = 99.9760019198464 :=
by
  sorry

end length_first_train_l84_84553


namespace find_xy_l84_84462

variable (x y : ℚ)

theorem find_xy (h1 : 1/x + 3/y = 1/2) (h2 : 1/y - 3/x = 1/3) : 
    x = -20 ∧ y = 60/11 := 
by
  sorry

end find_xy_l84_84462


namespace find_a6_l84_84455

-- Define the geometric sequence conditions
noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the specific sequence with given initial conditions and sum of first three terms
theorem find_a6 : 
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (0 < q) ∧ (q ≠ 1) ∧ geom_seq a q ∧ 
    a 1 = 96 ∧ 
    (a 1 + a 2 + a 3 = 168) ∧
    a 6 = 3 := 
by
  sorry

end find_a6_l84_84455


namespace compute_expression_l84_84030

theorem compute_expression (p q : ℝ) (h1 : p + q = 6) (h2 : p * q = 10) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + p * q^3 + p^5 * q^3 = 38676 := by
  -- Proof goes here
  sorry

end compute_expression_l84_84030


namespace book_total_pages_l84_84265

theorem book_total_pages (P : ℕ) (days_read : ℕ) (pages_per_day : ℕ) (fraction_read : ℚ) 
  (total_pages_read : ℕ) :
  (days_read = 15 ∧ pages_per_day = 12 ∧ fraction_read = 3 / 4 ∧ total_pages_read = 180 ∧ 
    total_pages_read = days_read * pages_per_day ∧ total_pages_read = fraction_read * P) → 
    P = 240 :=
by
  intros h
  sorry

end book_total_pages_l84_84265


namespace hyperbola_asymptotes_l84_84627

open Real

noncomputable def hyperbola (x y m : ℝ) : Prop := (x^2 / 9) - (y^2 / m) = 1

noncomputable def on_line (x y : ℝ) : Prop := x + y = 5

theorem hyperbola_asymptotes (m : ℝ) (hm : 9 + m = 25) :
    (∃ x y : ℝ, hyperbola x y m ∧ on_line x y) →
    (∀ x : ℝ, on_line x ((4 / 3) * x) ∧ on_line x (-(4 / 3) * x)) :=
by
  sorry

end hyperbola_asymptotes_l84_84627


namespace dinitrogen_monoxide_molecular_weight_l84_84910

def atomic_weight_N : Real := 14.01
def atomic_weight_O : Real := 16.00

def chemical_formula_N2O_weight : Real :=
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

theorem dinitrogen_monoxide_molecular_weight :
  chemical_formula_N2O_weight = 44.02 :=
by
  sorry

end dinitrogen_monoxide_molecular_weight_l84_84910


namespace colbert_planks_needed_to_buy_l84_84794

variables (total_planks : ℕ) (planks_from_storage : ℕ) 
          (planks_from_parents : ℕ) (planks_from_friends : ℕ)

def planks_needed_from_store := 
  total_planks - (planks_from_storage + planks_from_parents + planks_from_friends)

theorem colbert_planks_needed_to_buy : 
  total_planks = 200 → planks_from_storage = total_planks / 4 → 
  planks_from_parents = total_planks / 2 → planks_from_friends = 20 → 
  planks_needed_from_store total_planks planks_from_storage planks_from_parents planks_from_friends = 30 :=
by
  -- proof steps here
  sorry

end colbert_planks_needed_to_buy_l84_84794


namespace num_men_in_first_group_l84_84531

variable {x m w : ℝ}

theorem num_men_in_first_group (h1 : x * m + 8 * w = 6 * m + 2 * w)
  (h2 : 2 * m + 3 * w = 0.5 * (x * m + 8 * w)) : 
  x = 3 :=
sorry

end num_men_in_first_group_l84_84531


namespace find_a_if_even_function_l84_84172

-- Problem statement in Lean 4
theorem find_a_if_even_function (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 - 2 * (a + 1) * x + 1) 
  (hf_even : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_if_even_function_l84_84172


namespace recorder_price_new_l84_84996

theorem recorder_price_new (a b : ℕ) (h1 : 10 * a + b < 50) (h2 : 10 * b + a = (10 * a + b) * 12 / 10) :
  10 * b + a = 54 :=
by
  sorry

end recorder_price_new_l84_84996


namespace proof_problem_l84_84981

def consistent_system (x y : ℕ) : Prop :=
  x + y = 99 ∧ 3 * x + 1 / 3 * y = 97

theorem proof_problem : ∃ (x y : ℕ), consistent_system x y := sorry

end proof_problem_l84_84981


namespace gcd_of_ropes_l84_84279

theorem gcd_of_ropes : Nat.gcd (Nat.gcd 45 75) 90 = 15 := 
by
  sorry

end gcd_of_ropes_l84_84279


namespace annual_interest_rate_l84_84123

theorem annual_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) 
  (hP : P = 700) 
  (hA : A = 771.75) 
  (hn : n = 2) 
  (ht : t = 1) 
  (h : A = P * (1 + r / n) ^ (n * t)) : 
  r = 0.10 := 
by 
  -- Proof steps go here
  sorry

end annual_interest_rate_l84_84123


namespace sequence_geometric_l84_84506

theorem sequence_geometric (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 = 1)
  (h_geom : ∀ k : ℕ, a (k + 1) - a k = (1 / 3) ^ k) :
  a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by
  sorry

end sequence_geometric_l84_84506


namespace Simon_age_2010_l84_84348

variable (Jorge Simon : ℕ)

axiom age_difference : Jorge + 24 = Simon
axiom Jorge_age_2005 : Jorge = 16

theorem Simon_age_2010 : Simon = 45 := by
  have Simon_age_2005 : Simon = Jorge + 24 := age_difference
  rw [Jorge_age_2005] at Simon_age_2005
  have Simon_age_2005 : Simon = 16 + 24 := Simon_age_2005
  have Simon_age_2005 : Simon = 40 := by norm_num
  have Simon_age_2010 : Simon = 40 + 5 := by
    rw Simon_age_2005
    norm_num
  exact by norm_num at Simon_age_2010

end Simon_age_2010_l84_84348


namespace probability_at_least_one_boy_and_girl_l84_84258
-- Necessary imports

-- Defining the probability problem in Lean 4
theorem probability_at_least_one_boy_and_girl (n : ℕ) (hn : n = 4)
    (p : ℚ) (hp : p = 1 / 2) :
    let prob_all_same := (p ^ n) + (p ^ n) in
    (1 - prob_all_same) = 7 / 8 := by
  -- Include the proof steps here
  sorry

end probability_at_least_one_boy_and_girl_l84_84258


namespace trigonometric_identity_l84_84810

theorem trigonometric_identity
  (α : Real)
  (hcos : Real.cos α = -4/5)
  (hquad : π/2 < α ∧ α < π) :
  (-Real.sin (2 * α) / Real.cos α) = -6/5 := 
by
  sorry

end trigonometric_identity_l84_84810


namespace cassie_nail_cutting_l84_84270

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l84_84270


namespace four_digit_3_or_6_l84_84643

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l84_84643


namespace compute_expression_at_4_l84_84432

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end compute_expression_at_4_l84_84432


namespace compute_cd_l84_84033

-- Define the variables c and d as real numbers
variables (c d : ℝ)

-- Define the conditions
def condition1 : Prop := c + d = 10
def condition2 : Prop := c^3 + d^3 = 370

-- State the theorem we need to prove
theorem compute_cd (h1 : condition1 c d) (h2 : condition2 c d) : c * d = 21 :=
by
  sorry

end compute_cd_l84_84033


namespace num_four_digit_integers_with_3_and_6_l84_84642

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l84_84642


namespace simplify_fraction_1_210_plus_17_35_l84_84529

theorem simplify_fraction_1_210_plus_17_35 :
  1 / 210 + 17 / 35 = 103 / 210 :=
by sorry

end simplify_fraction_1_210_plus_17_35_l84_84529


namespace corn_cobs_each_row_l84_84580

theorem corn_cobs_each_row (x : ℕ) 
  (h1 : 13 * x + 16 * x = 116) : 
  x = 4 :=
by sorry

end corn_cobs_each_row_l84_84580


namespace tan_sum_angle_identity_l84_84003

theorem tan_sum_angle_identity
  (α β : ℝ)
  (h1 : Real.tan (α + 2 * β) = 2)
  (h2 : Real.tan β = -3) :
  Real.tan (α + β) = -1 := sorry

end tan_sum_angle_identity_l84_84003


namespace arithmetic_mean_median_l84_84535

theorem arithmetic_mean_median (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : a = 0) (h4 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end arithmetic_mean_median_l84_84535


namespace fraction_of_students_with_buddy_l84_84833

theorem fraction_of_students_with_buddy (t s : ℕ) (h1 : (t / 4) = (3 * s / 5)) :
  (t / 4 + 3 * s / 5) / (t + s) = 6 / 17 :=
by
  sorry

end fraction_of_students_with_buddy_l84_84833


namespace number_of_a_l84_84619

theorem number_of_a (h : ∃ a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2) : 
  ∃! a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2 :=
sorry

end number_of_a_l84_84619


namespace incorrect_statement_D_l84_84401

def ordinate_of_x_axis_is_zero (p : ℝ × ℝ) : Prop :=
  p.2 = 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A_properties (a b : ℝ) : Prop :=
  let x := - a^2 - 1
  let y := abs b
  x < 0 ∧ y ≥ 0

theorem incorrect_statement_D (a b : ℝ) : 
  ∃ (x y : ℝ), point_A_properties a b ∧ (x = -a^2 - 1 ∧ y = abs b ∧ (x < 0 ∧ y = 0)) :=
by {
  sorry
}

end incorrect_statement_D_l84_84401


namespace minimum_value_expression_l84_84673

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end minimum_value_expression_l84_84673


namespace solve_for_nabla_l84_84168

theorem solve_for_nabla : (∃ (nabla : ℤ), 5 * (-3) + 4 = nabla + 7) → (∃ (nabla : ℤ), nabla = -18) :=
by
  sorry

end solve_for_nabla_l84_84168


namespace composite_sum_of_ab_l84_84057

theorem composite_sum_of_ab (a b : ℕ) (h : 31 * a = 54 * b) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ a + b = k * l :=
sorry

end composite_sum_of_ab_l84_84057


namespace int_cubed_bound_l84_84457

theorem int_cubed_bound (a : ℤ) (h : 0 < a^3 ∧ a^3 < 9) : a = 1 ∨ a = 2 :=
sorry

end int_cubed_bound_l84_84457


namespace original_price_of_cycle_l84_84411

theorem original_price_of_cycle (SP : ℕ) (P : ℕ) (h1 : SP = 1800) (h2 : SP = 9 * P / 10) : P = 2000 :=
by
  have hSP_eq : SP = 1800 := h1
  have hSP_def : SP = 9 * P / 10 := h2
  -- Now we need to combine these to prove P = 2000
  sorry

end original_price_of_cycle_l84_84411


namespace earnings_per_hour_l84_84416

-- Define the conditions and the respective constants
def words_per_minute : ℕ := 10
def earnings_per_word : ℝ := 0.1
def earnings_per_article : ℝ := 60
def number_of_articles : ℕ := 3
def total_hours : ℕ := 4
def minutes_per_hour : ℕ := 60

theorem earnings_per_hour :
  let total_words := words_per_minute * minutes_per_hour * total_hours in
  let earnings_from_words := earnings_per_word * total_words in
  let earnings_from_articles := earnings_per_article * number_of_articles in
  let total_earnings := earnings_from_words + earnings_from_articles in
  let expected_earnings_per_hour := total_earnings / total_hours in
  expected_earnings_per_hour = 105 := 
  sorry

end earnings_per_hour_l84_84416


namespace num_four_digit_integers_with_3_and_6_l84_84641

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l84_84641


namespace worksheets_already_graded_eq_5_l84_84121

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def remaining_problems : ℕ := 16

def total_problems := total_worksheets * problems_per_worksheet
def graded_problems := total_problems - remaining_problems
def graded_worksheets := graded_problems / problems_per_worksheet

theorem worksheets_already_graded_eq_5 :
  graded_worksheets = 5 :=
by 
  sorry

end worksheets_already_graded_eq_5_l84_84121


namespace distance_from_point_to_directrix_l84_84309

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l84_84309


namespace percentage_passed_in_both_l84_84765

def percentage_of_students_failing_hindi : ℝ := 30
def percentage_of_students_failing_english : ℝ := 42
def percentage_of_students_failing_both : ℝ := 28

theorem percentage_passed_in_both (P_H_E: percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both = 44) : 
  100 - (percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both) = 56 := by
  sorry

end percentage_passed_in_both_l84_84765


namespace total_puppies_count_l84_84760

def first_week_puppies : Nat := 20
def second_week_puppies : Nat := 2 * first_week_puppies / 5
def third_week_puppies : Nat := 3 * second_week_puppies / 8
def fourth_week_puppies : Nat := 2 * second_week_puppies
def fifth_week_puppies : Nat := first_week_puppies + 10
def sixth_week_puppies : Nat := 2 * third_week_puppies - 5
def seventh_week_puppies : Nat := 2 * sixth_week_puppies
def eighth_week_puppies : Nat := 5 * seventh_week_puppies / 6 / 1 -- Assuming rounding down to nearest whole number

def total_puppies : Nat :=
  first_week_puppies + second_week_puppies + third_week_puppies +
  fourth_week_puppies + fifth_week_puppies + sixth_week_puppies +
  seventh_week_puppies + eighth_week_puppies

theorem total_puppies_count : total_puppies = 81 := by
  sorry

end total_puppies_count_l84_84760


namespace infinite_nat_N_polynomial_l84_84364

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l84_84364


namespace power_relationship_l84_84450

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end power_relationship_l84_84450


namespace expr_B_not_simplified_using_difference_of_squares_l84_84568

def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

theorem expr_B_not_simplified_using_difference_of_squares (x y : ℝ) :
  ∃ x y, ¬ ∃ a b, expr_B x y = a^2 - b^2 :=
sorry

end expr_B_not_simplified_using_difference_of_squares_l84_84568


namespace hand_towels_in_set_l84_84957

theorem hand_towels_in_set {h : ℕ}
  (hand_towel_sets : ℕ)
  (bath_towel_sets : ℕ)
  (hand_towel_sold : h * hand_towel_sets = 102)
  (bath_towel_sold : 6 * bath_towel_sets = 102)
  (same_sets_sold : hand_towel_sets = bath_towel_sets) :
  h = 17 := 
sorry

end hand_towels_in_set_l84_84957


namespace pure_gala_trees_l84_84101

variable (T F G : ℝ)

theorem pure_gala_trees (h1 : F + 0.1 * T = 170) (h2 : F = 0.75 * T): G = T - F -> G = 50 :=
by
  sorry

end pure_gala_trees_l84_84101


namespace other_root_l84_84493

theorem other_root (m : ℤ) (h : (∀ x : ℤ, x^2 - x + m = 0 → (x = 2))) : (¬ ∃ y : ℤ, (y^2 - y + m = 0 ∧ y ≠ 2 ∧ y ≠ -1) ) := 
by {
  sorry
}

end other_root_l84_84493


namespace part1_part2_l84_84618

variable (a : ℝ)
variable (x y : ℝ)
variable (P Q : ℝ × ℝ)

-- Part (1)
theorem part1 (hP : P = (2 * a - 2, a + 5)) (h_y : y = 0) : P = (-12, 0) :=
sorry

-- Part (2)
theorem part2 (hP : P = (2 * a - 2, a + 5)) (hQ : Q = (4, 5)) 
    (h_parallel : 2 * a - 2 = 4) : P = (4, 8) ∧ quadrant = "first" :=
sorry

end part1_part2_l84_84618


namespace haley_car_distance_l84_84060

theorem haley_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used_gallons : ℕ) (distance_covered_miles : ℕ) :
  fuel_ratio = 4 → distance_ratio = 7 → fuel_used_gallons = 44 → distance_covered_miles = (distance_ratio * fuel_used_gallons / fuel_ratio)
  → distance_covered_miles = 77 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  dsimp at h4
  linarith

end haley_car_distance_l84_84060


namespace fraction_equation_solution_l84_84939

theorem fraction_equation_solution (x : ℝ) (h : x ≠ 3) : (2 - x) / (x - 3) + 3 = 2 / (3 - x) ↔ x = 5 / 2 := by
  sorry

end fraction_equation_solution_l84_84939


namespace minimum_problems_45_l84_84840

-- Define the types for problems and their corresponding points
structure Problem :=
(points : ℕ)

def isValidScore (s : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s

def minimumProblems (s : ℕ) (min_problems : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s ∧ x + y + z = min_problems

-- Main statement
theorem minimum_problems_45 : minimumProblems 45 6 :=
by 
  sorry

end minimum_problems_45_l84_84840


namespace calculate_expr_eq_two_l84_84427

def calculate_expr : ℕ :=
  3^(0^(2^8)) + (3^0^2)^8

theorem calculate_expr_eq_two : calculate_expr = 2 := 
by
  sorry

end calculate_expr_eq_two_l84_84427


namespace average_book_width_correct_l84_84194

noncomputable def average_book_width 
  (widths : List ℚ) (number_of_books : ℕ) : ℚ :=
(widths.sum) / number_of_books

theorem average_book_width_correct :
  average_book_width [5, 3/4, 1.5, 3, 7.25, 12] 6 = 59 / 12 := 
  by 
  sorry

end average_book_width_correct_l84_84194


namespace intersection_A_B_l84_84621

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l84_84621


namespace containers_per_truck_l84_84737

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l84_84737


namespace person_B_winning_strategy_l84_84196

-- Definitions for the problem conditions
def winning_strategy_condition (L a b : ℕ) : Prop := 
  b = 2 * a ∧ ∃ k : ℕ, L = k * a

-- Lean theorem statement for the given problem
theorem person_B_winning_strategy (L a b : ℕ) (hL_pos : 0 < L) (ha_lt_hb : a < b) 
(hpos_a : 0 < a) (hpos_b : 0 < b) : 
  (∃ B_strat : Type, winning_strategy_condition L a b) :=
sorry

end person_B_winning_strategy_l84_84196


namespace percent_university_diploma_no_job_choice_l84_84835

theorem percent_university_diploma_no_job_choice
    (total_people : ℕ)
    (P1 : 10 * total_people / 100 = total_people / 10)
    (P2 : 20 * total_people / 100 = total_people / 5)
    (P3 : 30 * total_people / 100 = 3 * total_people / 10) :
  25 = (20 * total_people / (80 * total_people / 100)) :=
by
  sorry

end percent_university_diploma_no_job_choice_l84_84835


namespace original_price_l84_84122

theorem original_price (P : ℝ) (h : 0.684 * P = 6800) : P = 10000 :=
by
  sorry

end original_price_l84_84122


namespace largest_pentagon_angle_is_179_l84_84773

-- Define the interior angles of the pentagon
def angle1 (x : ℝ) := x + 2
def angle2 (x : ℝ) := 2 * x + 3
def angle3 (x : ℝ) := 3 * x - 5
def angle4 (x : ℝ) := 4 * x + 1
def angle5 (x : ℝ) := 5 * x - 1

-- Define the sum of the interior angles of a pentagon
def pentagon_angle_sum := angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36

-- Define the largest angle function
def largest_angle (x : ℝ) := 5 * x - 1

-- The main theorem stating the largest angle measure
theorem largest_pentagon_angle_is_179 (h : angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36 = 540) :
  largest_angle 36 = 179 :=
sorry

end largest_pentagon_angle_is_179_l84_84773


namespace consecutive_integers_product_l84_84725

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l84_84725


namespace ned_initially_had_games_l84_84997

variable (G : ℕ)

theorem ned_initially_had_games (h1 : (3 / 4) * (2 / 3) * G = 6) : G = 12 := by
  sorry

end ned_initially_had_games_l84_84997


namespace master_parts_per_hour_l84_84212

variable (x : ℝ)

theorem master_parts_per_hour (h1 : 300 / x = 100 / (40 - x)) : 300 / x = 100 / (40 - x) :=
sorry

end master_parts_per_hour_l84_84212


namespace fibonacci_invariant_abs_difference_l84_84687

-- Given the sequence defined by the recurrence relation
def mArithmetical_fibonacci (u_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u_n n = u_n (n - 2) + u_n (n - 1)

theorem fibonacci_invariant_abs_difference (u : ℕ → ℤ) 
  (h : mArithmetical_fibonacci u) :
  ∃ c : ℤ, ∀ n : ℕ, |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c := 
sorry

end fibonacci_invariant_abs_difference_l84_84687


namespace family_of_four_children_includes_one_boy_one_girl_l84_84259

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l84_84259


namespace range_of_m_l84_84152

def p (m : ℝ) : Prop :=
  let Δ := m^2 - 4
  Δ > 0 ∧ -m < 0

def q (m : ℝ) : Prop :=
  let Δ := 16*(m-2)^2 - 16
  Δ < 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ ((1 < m ∧ m ≤ 2) ∨ 3 ≤ m) :=
by {
  sorry
}

end range_of_m_l84_84152


namespace find_a_if_lines_perpendicular_l84_84623

-- Define the lines and the statement about their perpendicularity
theorem find_a_if_lines_perpendicular 
    (a : ℝ)
    (h_perpendicular : (2 * a) / (3 * (a - 1)) = 1) :
    a = 3 :=
by
  sorry

end find_a_if_lines_perpendicular_l84_84623


namespace distance_from_A_to_directrix_l84_84318

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l84_84318


namespace binary_to_decimal_1100_l84_84129

-- Define the binary number 1100
def binary_1100 : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0

-- State the theorem that we need to prove
theorem binary_to_decimal_1100 : binary_1100 = 12 := by
  rw [binary_1100]
  sorry

end binary_to_decimal_1100_l84_84129


namespace find_d_l84_84026

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x + 1

theorem find_d (c d : ℝ) (hx : ∀ x, f (g x c) c = 15 * x + d) : d = 8 :=
sorry

end find_d_l84_84026


namespace number_of_common_divisors_l84_84960

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l84_84960


namespace store_profit_l84_84245

variable (m n : ℝ)
variable (h_mn : m > n)

theorem store_profit : 10 * (m - n) > 0 :=
by
  sorry

end store_profit_l84_84245


namespace no_intersection_tangent_graph_l84_84829

theorem no_intersection_tangent_graph (k : ℝ) (m : ℤ) : 
  (∀ x: ℝ, x = (k * Real.pi) / 2 → (¬ 4 * k ≠ 4 * m + 1)) → 
  (-1 ≤ k ∧ k ≤ 1) →
  (k = 1 / 4 ∨ k = -3 / 4) :=
sorry

end no_intersection_tangent_graph_l84_84829


namespace common_divisors_l84_84965

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l84_84965


namespace price_of_pants_l84_84679

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end price_of_pants_l84_84679


namespace max_area_2017_2018_l84_84253

noncomputable def max_area_of_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem max_area_2017_2018 :
  max_area_of_triangle 2017 2018 = 2035133 := by
  sorry

end max_area_2017_2018_l84_84253


namespace consecutive_integers_sum_l84_84705

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l84_84705


namespace consecutive_integers_product_l84_84723

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l84_84723


namespace find_a_l84_84299

def A (x : ℝ) : Set ℝ := {1, 2, x^2 - 5 * x + 9}
def B (x a : ℝ) : Set ℝ := {3, x^2 + a * x + a}

theorem find_a (a x : ℝ) (hxA : A x = {1, 2, 3}) (h2B : 2 ∈ B x a) :
  a = -2/3 ∨ a = -7/4 :=
by sorry

end find_a_l84_84299


namespace volume_ratio_of_spheres_l84_84728

theorem volume_ratio_of_spheres 
  (r1 r2 : ℝ) 
  (h_surface_area : (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 16) : 
  (4 / 3 * Real.pi * r1^3) / (4 / 3 * Real.pi * r2^3) = 1 / 64 :=
by 
  sorry

end volume_ratio_of_spheres_l84_84728


namespace compute_expression_at_4_l84_84431

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end compute_expression_at_4_l84_84431


namespace percent_of_dollar_is_37_l84_84612

variable (coins_value_in_cents : ℕ)
variable (percent_of_one_dollar : ℕ)

def value_of_pennies : ℕ := 2 * 1
def value_of_nickels : ℕ := 3 * 5
def value_of_dimes : ℕ := 2 * 10

def total_coin_value : ℕ := value_of_pennies + value_of_nickels + value_of_dimes

theorem percent_of_dollar_is_37
  (h1 : total_coin_value = coins_value_in_cents)
  (h2 : percent_of_one_dollar = (coins_value_in_cents * 100) / 100) : 
  percent_of_one_dollar = 37 := 
by
  sorry

end percent_of_dollar_is_37_l84_84612


namespace probability_allison_greater_l84_84588

/-- Allison, Brian, and Noah each roll a die.
- Allison's die always rolls a 5.
- Brian's die has faces 1, 2, 3, 4, 4, 5, 5, 6.
- Noah's die has faces 2, 2, 6, 6, 3, 3, 7, 7.
The statement below proves that the probability that Allison's roll is greater than both Brian's and Noah's rolls is 5/16. -/
theorem probability_allison_greater (A B N : ℕ) (PB : Equiv.Perm [1, 2, 3, 4, 4, 5, 5, 6])
  (PN : Equiv.Perm [2, 2, 6, 6, 3, 3, 7, 7]) :
  (probability (fun (_ : ℕ × ℕ × ℕ) => (A = 5) ∧ (B ∈ {1, 2, 3, 4}) ∧ (N ∈ {2, 3}))  
       (A, B, N) (fun _ => by sorry)) = 5 / 16 :=
sorry

end probability_allison_greater_l84_84588


namespace segment_length_aa_prime_l84_84391

/-- Given points A, B, and C, and their reflections, show that the length of AA' is 8 -/
theorem segment_length_aa_prime
  (A : ℝ × ℝ) (A_reflected : ℝ × ℝ)
  (x₁ y₁ y₁_neg : ℝ) :
  A = (x₁, y₁) →
  A_reflected = (x₁, y₁_neg) →
  y₁_neg = -y₁ →
  y₁ = 4 →
  x₁ = 2 →
  |y₁ - y₁_neg| = 8 :=
sorry

end segment_length_aa_prime_l84_84391


namespace no_values_of_b_l84_84298

def f (b x : ℝ) := x^2 + b * x - 1

theorem no_values_of_b : ∀ b : ℝ, ∃ x : ℝ, f b x = 3 :=
by
  intro b
  use 0  -- example, needs actual computation
  sorry

end no_values_of_b_l84_84298


namespace cos_beta_value_l84_84811

open Real

theorem cos_beta_value (α β : ℝ) (h1 : sin α = sqrt 5 / 5) (h2 : sin (α - β) = - sqrt 10 / 10) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) : cos β = sqrt 2 / 2 :=
by
sorry

end cos_beta_value_l84_84811


namespace square_root_condition_l84_84377

-- Define the condition under which the square root of an expression is defined
def is_square_root_defined (x : ℝ) : Prop := (x + 3) ≥ 0

-- Prove that the condition for the square root of x + 3 to be defined is x ≥ -3
theorem square_root_condition (x : ℝ) : is_square_root_defined x ↔ x ≥ -3 := 
sorry

end square_root_condition_l84_84377


namespace smallest_y_76545_l84_84764

theorem smallest_y_76545 (y : ℕ) (h1 : ∀ z : ℕ, 0 < z → (76545 * z = k ^ 2 → (3 ∣ z ∨ 5 ∣ z) → z = y)) : y = 7 :=
sorry

end smallest_y_76545_l84_84764


namespace alice_min_speed_exceeds_45_l84_84236

theorem alice_min_speed_exceeds_45 
  (distance : ℕ)
  (bob_speed : ℕ)
  (alice_delay : ℕ)
  (alice_speed : ℕ)
  (bob_time : ℕ)
  (expected_speed : ℕ) 
  (distance_eq : distance = 180)
  (bob_speed_eq : bob_speed = 40)
  (alice_delay_eq : alice_delay = 1/2)
  (bob_time_eq : bob_time = distance / bob_speed)
  (expected_speed_eq : expected_speed = distance / (bob_time - alice_delay)) :
  alice_speed > expected_speed := 
sorry

end alice_min_speed_exceeds_45_l84_84236


namespace range_of_a_l84_84628

-- Definitions capturing the given conditions
variables (a b c : ℝ)

-- Conditions are stated as assumptions
def condition1 := a^2 - b * c - 8 * a + 7 = 0
def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

-- The mathematically equivalent proof problem
theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
sorry

end range_of_a_l84_84628


namespace common_divisors_l84_84966

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l84_84966


namespace dogs_in_academy_l84_84125

noncomputable def numberOfDogs : ℕ :=
  let allSit := 60
  let allStay := 35
  let allFetch := 40
  let allRollOver := 45
  let sitStay := 20
  let sitFetch := 15
  let sitRollOver := 18
  let stayFetch := 10
  let stayRollOver := 13
  let fetchRollOver := 12
  let sitStayFetch := 11
  let sitStayFetchRoll := 8
  let none := 15
  118 -- final count of dogs in the academy

theorem dogs_in_academy : numberOfDogs = 118 :=
by
  sorry

end dogs_in_academy_l84_84125


namespace minimum_value_expression_l84_84672

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end minimum_value_expression_l84_84672


namespace hollow_circles_in_2001_pattern_l84_84219

theorem hollow_circles_in_2001_pattern :
  let pattern_length := 9
  let hollow_in_pattern := 3
  let total_circles := 2001
  let complete_patterns := total_circles / pattern_length
  let remaining_circles := total_circles % pattern_length
  let hollow_in_remaining := if remaining_circles >= 3 then 1 else 0
  let total_hollow := complete_patterns * hollow_in_pattern + hollow_in_remaining
  total_hollow = 667 :=
by
  sorry

end hollow_circles_in_2001_pattern_l84_84219


namespace third_root_of_polynomial_l84_84744

theorem third_root_of_polynomial (a b : ℚ) 
  (h₁ : a*(-1)^3 + (a + 3*b)*(-1)^2 + (2*b - 4*a)*(-1) + (10 - a) = 0)
  (h₂ : a*(4)^3 + (a + 3*b)*(4)^2 + (2*b - 4*a)*(4) + (10 - a) = 0) :
  ∃ (r : ℚ), r = -24 / 19 :=
by
  sorry

end third_root_of_polynomial_l84_84744


namespace orangeade_price_l84_84766

theorem orangeade_price (O W : ℝ) (h1 : O = W) (price_day1 : ℝ) (price_day2 : ℝ) 
    (volume_day1 : ℝ) (volume_day2 : ℝ) (revenue_day1 : ℝ) (revenue_day2 : ℝ) : 
    volume_day1 = 2 * O ∧ volume_day2 = 3 * O ∧ revenue_day1 = revenue_day2 ∧ price_day1 = 0.82 
    → price_day2 = 0.55 :=
by
    intros
    sorry

end orangeade_price_l84_84766


namespace distance_from_A_to_directrix_l84_84320

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l84_84320


namespace tan_difference_l84_84812

theorem tan_difference (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h₁ : Real.sin α = 3 / 5) (h₂ : Real.cos β = 12 / 13) : 
    Real.tan (α - β) = 16 / 63 := 
by
  sorry

end tan_difference_l84_84812


namespace not_simplifiable_by_difference_of_squares_l84_84566

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end not_simplifiable_by_difference_of_squares_l84_84566


namespace production_steps_use_process_flowchart_l84_84873

def describe_production_steps (task : String) : Prop :=
  task = "describe production steps of a certain product in a factory"

def correct_diagram (diagram : String) : Prop :=
  diagram = "Process Flowchart"

theorem production_steps_use_process_flowchart (task : String) (diagram : String) :
  describe_production_steps task → correct_diagram diagram :=
sorry

end production_steps_use_process_flowchart_l84_84873


namespace problem_statement_l84_84331

theorem problem_statement (a b c x : ℝ) (h1 : a + x^2 = 2015) (h2 : b + x^2 = 2016)
    (h3 : c + x^2 = 2017) (h4 : a * b * c = 24) :
    (a / (b * c) + b / (a * c) + c / (a * b) - (1 / a) - (1 / b) - (1 / c) = 1 / 8) :=
by
  sorry

end problem_statement_l84_84331


namespace max_Bk_l84_84923

theorem max_Bk (k : ℕ) (h0 : 0 ≤ k) (h1 : k ≤ 2000) : k = 181 ↔ 
  ∀ k' : ℕ, (0 ≤ k' ∧ k' ≤ 2000) → B k ≤ B k' :=
sorry

def B (k : ℕ) : ℝ :=
  if (0 ≤ k ∧ k ≤ 2000) then ((nat.choose 2000 k) : ℝ) * (0.1 ^ k) else 0

end max_Bk_l84_84923


namespace number_of_yellow_marbles_l84_84839

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end number_of_yellow_marbles_l84_84839


namespace fifth_term_of_sequence_is_31_l84_84808

namespace SequenceProof

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem fifth_term_of_sequence_is_31 :
  ∃ a : ℕ → ℕ, sequence a ∧ a 5 = 31 :=
by
  sorry

end SequenceProof

end fifth_term_of_sequence_is_31_l84_84808


namespace total_marks_scored_l84_84510

theorem total_marks_scored :
  let Keith_score := 3.5
  let Larry_score := Keith_score * 3.2
  let Danny_score := Larry_score + 5.7
  let Emma_score := (Danny_score * 2) - 1.2
  let Fiona_score := (Keith_score + Larry_score + Danny_score + Emma_score) / 4
  Keith_score + Larry_score + Danny_score + Emma_score + Fiona_score = 80.25 :=
by
  sorry

end total_marks_scored_l84_84510


namespace rackets_packed_l84_84046

theorem rackets_packed (total_cartons : ℕ) (cartons_3 : ℕ) (cartons_2 : ℕ) 
  (h1 : total_cartons = 38) 
  (h2 : cartons_3 = 24) 
  (h3 : cartons_2 = total_cartons - cartons_3) :
  3 * cartons_3 + 2 * cartons_2 = 100 := 
by
  -- The proof is omitted
  sorry

end rackets_packed_l84_84046


namespace Gyeongyeon_cookies_l84_84327

def initial_cookies : ℕ := 20
def cookies_given : ℕ := 7
def cookies_received : ℕ := 5

def final_cookies (initial : ℕ) (given : ℕ) (received : ℕ) : ℕ :=
  initial - given + received

theorem Gyeongyeon_cookies :
  final_cookies initial_cookies cookies_given cookies_received = 18 :=
by
  sorry

end Gyeongyeon_cookies_l84_84327


namespace value_of_x_l84_84477

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l84_84477


namespace find_x_l84_84630

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ × ℝ := (2, 1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_x (x : ℝ) (h : dot_product (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) = 0) : x = -3 :=
  sorry

end find_x_l84_84630


namespace probability_at_least_one_boy_and_one_girl_l84_84256

theorem probability_at_least_one_boy_and_one_girl :
  let P := (1 - (1/16 + 1/16)) = 7 / 8,
  (∀ (N: ℕ), (N = 4) → 
    let prob_all_boys := (1 / N) ^ N,
    let prob_all_girls := (1 / N) ^ N,
    let prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)
  in prob_at_least_one_boy_and_one_girl = P) :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l84_84256


namespace sufficient_but_not_necessary_condition_l84_84769

theorem sufficient_but_not_necessary_condition (h1 : 1^2 - 1 = 0) (h2 : ∀ x, x^2 - 1 = 0 → (x = 1 ∨ x = -1)) :
  (∀ x, x = 1 → x^2 - 1 = 0) ∧ ¬ (∀ x, x^2 - 1 = 0 → x = 1) := by
  sorry

end sufficient_but_not_necessary_condition_l84_84769


namespace num_clerks_l84_84832

def manager_daily_salary := 5
def clerk_daily_salary := 2
def num_managers := 2
def total_daily_salary := 16

theorem num_clerks (c : ℕ) (h1 : num_managers * manager_daily_salary + c * clerk_daily_salary = total_daily_salary) : c = 3 :=
by 
  sorry

end num_clerks_l84_84832


namespace greatest_x_l84_84534

-- Define x as a positive multiple of 4.
def is_positive_multiple_of_four (x : ℕ) : Prop :=
  x > 0 ∧ ∃ k : ℕ, x = 4 * k

-- Statement of the equivalent proof problem
theorem greatest_x (x : ℕ) (h1: is_positive_multiple_of_four x) (h2: x^3 < 4096) : x ≤ 12 :=
by {
  sorry
}

end greatest_x_l84_84534


namespace find_extrema_l84_84326

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end find_extrema_l84_84326


namespace two_abc_square_l84_84350

variable {R : Type*} [Ring R] [Fintype R]

-- Given condition: For any a, b ∈ R, ∃ c ∈ R such that a^2 + b^2 = c^2.
axiom ring_property (a b : R) : ∃ c : R, a^2 + b^2 = c^2

-- We need to prove: For any a, b, c ∈ R, ∃ d ∈ R such that 2abc = d^2.
theorem two_abc_square (a b c : R) : ∃ d : R, 2 * (a * b * c) = d^2 :=
by
  sorry

end two_abc_square_l84_84350


namespace trigonometric_identity_l84_84280

theorem trigonometric_identity :
  (1 - Real.sin (Real.pi / 6)) * (1 - Real.sin (5 * Real.pi / 6)) = 1 / 4 :=
by
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by sorry
  have h2 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  sorry

end trigonometric_identity_l84_84280


namespace number_of_cakes_sold_l84_84426

namespace Bakery

variables (cakes pastries sold_cakes sold_pastries : ℕ)

-- Defining the conditions
def pastries_sold := 154
def more_pastries_than_cakes := 76

-- Defining the problem statement
theorem number_of_cakes_sold (h1 : sold_pastries = pastries_sold) 
                             (h2 : sold_pastries = sold_cakes + more_pastries_than_cakes) : 
                             sold_cakes = 78 :=
by {
  sorry
}

end Bakery

end number_of_cakes_sold_l84_84426


namespace problem_part1_problem_part2_problem_part3_l84_84144

section
variables (a b : ℚ)

-- Define the operation
def otimes (a b : ℚ) : ℚ := a * b + abs a - b

-- Prove the three statements
theorem problem_part1 : otimes (-5) 4 = -19 :=
sorry

theorem problem_part2 : otimes (otimes 2 (-3)) 4 = -7 :=
sorry

theorem problem_part3 : otimes 3 (-2) > otimes (-2) 3 :=
sorry
end

end problem_part1_problem_part2_problem_part3_l84_84144


namespace slope_of_line_through_PQ_is_4_l84_84004

theorem slope_of_line_through_PQ_is_4
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a4 : a 4 = 15)
  (h_a9 : a 9 = 55) :
  let a3 := a 3
  let a8 := a 8
  (a 9 - a 4) / (9 - 4) = 8 → (a 8 - a 3) / (13 - 3) = 4 := by
  sorry

end slope_of_line_through_PQ_is_4_l84_84004


namespace tan_diff_eq_rat_l84_84980

theorem tan_diff_eq_rat (A : ℝ × ℝ) (B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (5, 1))
  (α β : ℝ)
  (hα : Real.tan α = 2) (hβ : Real.tan β = 1 / 5) :
  Real.tan (α - β) = 9 / 7 := by
  sorry

end tan_diff_eq_rat_l84_84980
