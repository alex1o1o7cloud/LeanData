import Mathlib

namespace find_original_speed_l747_74755

theorem find_original_speed :
  ∀ (v T : ℝ), 
    (300 = 212 + 88) →
    (T + 2/3 = 212 / v + 88 / (v - 50)) →
    v = 110 :=
by
  intro v T h_dist h_trip
  sorry

end find_original_speed_l747_74755


namespace width_of_channel_at_bottom_l747_74701

theorem width_of_channel_at_bottom
    (top_width : ℝ)
    (area : ℝ)
    (depth : ℝ)
    (b : ℝ)
    (H1 : top_width = 12)
    (H2 : area = 630)
    (H3 : depth = 70)
    (H4 : area = 0.5 * (top_width + b) * depth) :
    b = 6 := 
sorry

end width_of_channel_at_bottom_l747_74701


namespace trader_gain_percentage_l747_74717

structure PenType :=
  (pens_sold : ℕ)
  (cost_per_pen : ℕ)

def total_cost (pen : PenType) : ℕ :=
  pen.pens_sold * pen.cost_per_pen

def gain (pen : PenType) (multiplier : ℕ) : ℕ :=
  multiplier * pen.cost_per_pen

def weighted_average_gain_percentage (penA penB penC : PenType) (gainA gainB gainC : ℕ) : ℚ :=
  (((gainA + gainB + gainC):ℚ) / ((total_cost penA + total_cost penB + total_cost penC):ℚ)) * 100

theorem trader_gain_percentage :
  ∀ (penA penB penC : PenType)
  (gainA gainB gainC : ℕ),
  penA.pens_sold = 60 →
  penA.cost_per_pen = 2 →
  penB.pens_sold = 40 →
  penB.cost_per_pen = 3 →
  penC.pens_sold = 50 →
  penC.cost_per_pen = 4 →
  gainA = 20 * penA.cost_per_pen →
  gainB = 15 * penB.cost_per_pen →
  gainC = 10 * penC.cost_per_pen →
  weighted_average_gain_percentage penA penB penC gainA gainB gainC = 28.41 := 
by
  intros
  sorry

end trader_gain_percentage_l747_74717


namespace value_of_a_plus_b_l747_74768

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end value_of_a_plus_b_l747_74768


namespace rice_mixture_price_l747_74712

-- Defining the costs per kg for each type of rice
def rice_cost1 : ℝ := 16
def rice_cost2 : ℝ := 24

-- Defining the given ratio
def mixing_ratio : ℝ := 3

-- Main theorem stating the problem
theorem rice_mixture_price
  (x : ℝ)  -- The common measure of quantity in the ratio
  (h1 : 3 * x * rice_cost1 + x * rice_cost2 = 72 * x)
  (h2 : 3 * x + x = 4 * x) :
  (3 * x * rice_cost1 + x * rice_cost2) / (3 * x + x) = 18 :=
by
  sorry

end rice_mixture_price_l747_74712


namespace max_value_2x_plus_y_l747_74732

def max_poly_value : ℝ :=
  sorry

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 
  2 * x + y ≤ 6 :=
sorry

example (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 2 * x + y = 6 
  ↔ x = 3 ∧ y = 0 :=
by exact sorry

end max_value_2x_plus_y_l747_74732


namespace arithmetic_mean_of_set_l747_74709

theorem arithmetic_mean_of_set {x : ℝ} (mean_eq_12 : (8 + 16 + 20 + x + 12) / 5 = 12) : x = 4 :=
by
  sorry

end arithmetic_mean_of_set_l747_74709


namespace log_579_between_consec_ints_l747_74730

theorem log_579_between_consec_ints (a b : ℤ) (h₁ : 2 < Real.log 579 / Real.log 10) (h₂ : Real.log 579 / Real.log 10 < 3) : a + b = 5 :=
sorry

end log_579_between_consec_ints_l747_74730


namespace math_problem_l747_74704

theorem math_problem : 
  (Real.sqrt 4) * (4 ^ (1 / 2: ℝ)) + (16 / 4) * 2 - (8 ^ (1 / 2: ℝ)) = 12 - 2 * Real.sqrt 2 :=
by
  sorry

end math_problem_l747_74704


namespace problem_divisible_by_factors_l747_74748

theorem problem_divisible_by_factors (n : ℕ) (x : ℝ) : 
  ∃ k : ℝ, (x + 1)^(2 * n) - x^(2 * n) - 2 * x - 1 = k * x * (x + 1) * (2 * x + 1) :=
by
  sorry

end problem_divisible_by_factors_l747_74748


namespace transformed_polynomial_roots_l747_74731

theorem transformed_polynomial_roots (a b c d : ℝ) 
  (h1 : a + b + c + d = 0)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h3 : a * b * c * d ≠ 0)
  (h4 : Polynomial.eval a (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h5 : Polynomial.eval b (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h6 : Polynomial.eval c (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h7 : Polynomial.eval d (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0):
  Polynomial.eval (-2 / d^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / c^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / b^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / a^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 :=
sorry

end transformed_polynomial_roots_l747_74731


namespace arithmetic_sequence_common_difference_l747_74765

theorem arithmetic_sequence_common_difference (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 3 = 4) (h₂ : S 3 = 3)
  (h₃ : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h₄ : ∀ n, a n = a 1 + (n - 1) * d) :
  ∃ d, d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l747_74765


namespace correct_transformation_l747_74700

structure Point :=
  (x : ℝ)
  (y : ℝ)

def rotate180 (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def is_rotation_180 (p p' : Point) : Prop :=
  rotate180 p = p'

theorem correct_transformation (C D : Point) (C' D' : Point) 
  (hC : C = Point.mk 3 (-2)) 
  (hC' : C' = Point.mk (-3) 2)
  (hD : D = Point.mk 2 (-5)) 
  (hD' : D' = Point.mk (-2) 5) :
  is_rotation_180 C C' ∧ is_rotation_180 D D' :=
by
  sorry

end correct_transformation_l747_74700


namespace trapezoid_height_l747_74725

-- Definitions of the problem conditions
def is_isosceles_trapezoid (a b : ℝ) : Prop :=
  ∃ (AB CD BM CN h : ℝ), a = 24 ∧ b = 10 ∧ AB = 25 ∧ CD = 25 ∧ BM = h ∧ CN = h ∧
  BM ^ 2 + ((24 - 10) / 2) ^ 2 = AB ^ 2

-- The theorem to prove
theorem trapezoid_height (a b : ℝ) (h : ℝ) 
  (H : is_isosceles_trapezoid a b) : h = 24 :=
sorry

end trapezoid_height_l747_74725


namespace value_of_f_log3_54_l747_74760

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem value_of_f_log3_54
  (h1 : is_odd f)
  (h2 : ∀ x, f (x + 2) = -1 / f x)
  (h3 : ∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) :
  f (Real.log 54 / Real.log 3) = -3 / 2 := sorry

end value_of_f_log3_54_l747_74760


namespace part_a_part_b_l747_74716

-- Define the function with the given conditions
variable {f : ℝ → ℝ}
variable (h_nonneg : ∀ x, 0 ≤ x → 0 ≤ f x)
variable (h_f1 : f 1 = 1)
variable (h_subadditivity : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Part (a): Prove that f(x) ≤ 2x for all x ∈ [0, 1]
theorem part_a : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
by
  sorry -- Proof required.

-- Part (b): Prove that it is not true that f(x) ≤ 1.9x for all x ∈ [0,1]
theorem part_b : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ 1.9 * x < f x :=
by
  sorry -- Proof required.

end part_a_part_b_l747_74716


namespace rectangle_area_increase_l747_74771

theorem rectangle_area_increase :
  let l := 33.333333333333336
  let b := l / 2
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 4
  let A_new := l_new * b_new
  A_new - A_original = 30 :=
by
  sorry

end rectangle_area_increase_l747_74771


namespace range_of_a_l747_74726

theorem range_of_a (a : ℝ) (x : ℝ) :
  ((a < x ∧ x < a + 2) → x > 3) ∧ ¬(∀ x, (x > 3) → (a < x ∧ x < a + 2)) → a ≥ 3 :=
by
  sorry

end range_of_a_l747_74726


namespace part1_arithmetic_sequence_part2_general_term_part3_max_m_l747_74758

-- Part (1)
theorem part1_arithmetic_sequence (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : a 1 + a 2 = 2 * m) : 
  m = 9 / 8 := 
sorry

-- Part (2)
theorem part2_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2) : 
  ∀ n, a n = 8 ^ (1 - 2 ^ (n - 1)) := 
sorry

-- Part (3)
theorem part3_max_m (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : ∀ n, a n < 4) : 
  m ≤ 2 := 
sorry

end part1_arithmetic_sequence_part2_general_term_part3_max_m_l747_74758


namespace silvia_escalator_time_l747_74714

noncomputable def total_time_standing (v s : ℝ) : ℝ := 
  let d := 80 * v
  d / s

theorem silvia_escalator_time (v s t : ℝ) (h1 : 80 * v = 28 * (v + s)) (h2 : t = total_time_standing v s) : 
  t = 43 := by
  sorry

end silvia_escalator_time_l747_74714


namespace gcd_g_x_1155_l747_74774

def g (x : ℕ) := (4 * x + 5) * (5 * x + 3) * (6 * x + 7) * (3 * x + 11)

theorem gcd_g_x_1155 (x : ℕ) (h : x % 18711 = 0) : Nat.gcd (g x) x = 1155 := by
  sorry

end gcd_g_x_1155_l747_74774


namespace find_point_A_l747_74721

-- Define the point -3, 4
def pointP : ℝ × ℝ := (-3, 4)

-- Define the point 0, 2
def pointB : ℝ × ℝ := (0, 2)

-- Define the coordinates of point A
def pointA (x : ℝ) : ℝ × ℝ := (x, 0)

-- The hypothesis using the condition derived from the problem
def ray_reflection_condition (x : ℝ) : Prop :=
  4 / (x + 3) = -2 / x

-- The main theorem we need to prove that the coordinates of point A are (-1, 0)
theorem find_point_A :
  ∃ x : ℝ, ray_reflection_condition x ∧ pointA x = (-1, 0) :=
sorry

end find_point_A_l747_74721


namespace problem_I_problem_II_l747_74770

def f (x : ℝ) : ℝ := abs (x - 1)

theorem problem_I (x : ℝ) : f (2 * x) + f (x + 4) ≥ 8 ↔ x ≤ -10 / 3 ∨ x ≥ 2 := by
  sorry

variable {a b : ℝ}
theorem problem_II (ha : abs a < 1) (hb : abs b < 1) (h_neq : a ≠ 0) : 
  (abs (a * b - 1) / abs a) > abs ((b / a) - 1) :=
by
  sorry

end problem_I_problem_II_l747_74770


namespace alice_bob_same_point_after_3_turns_l747_74792

noncomputable def alice_position (t : ℕ) : ℕ := (15 + 4 * t) % 15

noncomputable def bob_position (t : ℕ) : ℕ :=
  if t < 2 then 15
  else (15 - 11 * (t - 2)) % 15

theorem alice_bob_same_point_after_3_turns :
  ∃ t, t = 3 ∧ alice_position t = bob_position t :=
by
  exists 3
  simp only [alice_position, bob_position]
  norm_num
  -- Alice's position after 3 turns
  -- alice_position 3 = (15 + 4 * 3) % 15
  -- bob_position 3 = (15 - 11 * (3 - 2)) % 15
  -- Therefore,
  -- alice_position 3 = 12
  -- bob_position 3 = 12
  sorry

end alice_bob_same_point_after_3_turns_l747_74792


namespace baker_price_l747_74738

theorem baker_price
  (P : ℝ)
  (h1 : 8 * P = 320)
  (h2 : 10 * (0.80 * P) = 320)
  : P = 40 := sorry

end baker_price_l747_74738


namespace log_comparison_l747_74702

theorem log_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < e) (h3 : 0 < b) (h4 : b < e) (h5 : a < b) :
  a * Real.log b > b * Real.log a := sorry

end log_comparison_l747_74702


namespace ursula_initial_money_l747_74713

def cost_per_hot_dog : ℝ := 1.50
def number_of_hot_dogs : ℕ := 5
def cost_per_salad : ℝ := 2.50
def number_of_salads : ℕ := 3
def change_received : ℝ := 5.00

def total_cost_of_hot_dogs : ℝ := number_of_hot_dogs * cost_per_hot_dog
def total_cost_of_salads : ℝ := number_of_salads * cost_per_salad
def total_cost : ℝ := total_cost_of_hot_dogs + total_cost_of_salads
def amount_paid : ℝ := total_cost + change_received

theorem ursula_initial_money : amount_paid = 20.00 := by
  /- Proof here, which is not required for the task -/
  sorry

end ursula_initial_money_l747_74713


namespace length_of_lawn_l747_74795

-- Definitions based on conditions
def area_per_bag : ℝ := 250
def width : ℝ := 36
def num_bags : ℝ := 4
def extra_area : ℝ := 208

-- Statement to prove
theorem length_of_lawn :
  (num_bags * area_per_bag + extra_area) / width = 33.56 := by
  sorry

end length_of_lawn_l747_74795


namespace f_even_l747_74719

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  sorry

end f_even_l747_74719


namespace fair_prize_division_l747_74799

theorem fair_prize_division (eq_chance : ∀ (game : ℕ), 0.5 ≤ 1 ∧ 1 ≤ 0.5)
  (first_to_six : ∀ (p1_wins p2_wins : ℕ), (p1_wins = 6 ∨ p2_wins = 6) → (p1_wins + p2_wins) ≤ 11)
  (current_status : 5 + 3 = 8) :
  (7 : ℝ) / 8 = 7 / (8 : ℝ) :=
by
  sorry

end fair_prize_division_l747_74799


namespace students_playing_both_l747_74791

theorem students_playing_both (T F L N B : ℕ)
  (hT : T = 39)
  (hF : F = 26)
  (hL : L = 20)
  (hN : N = 10)
  (hTotal : (F + L - B) + N = T) :
  B = 17 :=
by
  sorry

end students_playing_both_l747_74791


namespace sufficient_but_not_necessary_condition_l747_74729

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) (h : x > 0) : (∃ y : ℝ, (y < -3 ∨ y > -1) ∧ y > 0) := by
  sorry

end sufficient_but_not_necessary_condition_l747_74729


namespace ratio_of_x_to_y_l747_74723

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) : 
  x / y = Real.sqrt (17 / 8) :=
by
  sorry

end ratio_of_x_to_y_l747_74723


namespace no_groups_of_six_l747_74737

theorem no_groups_of_six (x y z : ℕ) 
  (h1 : (2 * x + 6 * y + 10 * z) / (x + y + z) = 5)
  (h2 : (2 * x + 30 * y + 90 * z) / (2 * x + 6 * y + 10 * z) = 7) : 
  y = 0 := 
sorry

end no_groups_of_six_l747_74737


namespace sum_octal_eq_1021_l747_74793

def octal_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let r1 := n / 10
  let d1 := r1 % 10
  let r2 := r1 / 10
  let d2 := r2 % 10
  (d2 * 64) + (d1 * 8) + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let d0 := n % 8
  let r1 := n / 8
  let d1 := r1 % 8
  let r2 := r1 / 8
  let d2 := r2 % 8
  d2 * 100 + d1 * 10 + d0

theorem sum_octal_eq_1021 :
  decimal_to_octal (octal_to_decimal 642 + octal_to_decimal 157) = 1021 := by
  sorry

end sum_octal_eq_1021_l747_74793


namespace exp_ineq_solution_set_l747_74733

theorem exp_ineq_solution_set (e : ℝ) (h : e = Real.exp 1) :
  {x : ℝ | e^(2*x - 1) < 1} = {x : ℝ | x < 1 / 2} :=
sorry

end exp_ineq_solution_set_l747_74733


namespace compute_fg_neg1_l747_74778

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem compute_fg_neg1 : f (g (-1)) = 3 := by
  sorry

end compute_fg_neg1_l747_74778


namespace prove_functions_same_l747_74753

theorem prove_functions_same (u v : ℝ) (huv : u = v) : 
  (u > 1) → (v > 1) → (Real.sqrt ((u + 1) / (u - 1)) = Real.sqrt ((v + 1) / (v - 1))) :=
by
  sorry

end prove_functions_same_l747_74753


namespace brendan_weekly_capacity_l747_74711

/-- Brendan can cut 8 yards of grass per day on flat terrain under normal weather conditions. Bought a lawnmower that improved his cutting speed by 50 percent on flat terrain. On uneven terrain, his speed is reduced by 35 percent. Rain reduces his cutting capacity by 20 percent. Extreme heat reduces his cutting capacity by 10 percent. The conditions for each day of the week are given and we want to prove that the total yards Brendan can cut in a week is 65.46 yards.
  Monday: Flat terrain, normal weather
  Tuesday: Flat terrain, rain
  Wednesday: Uneven terrain, normal weather
  Thursday: Flat terrain, extreme heat
  Friday: Uneven terrain, rain
  Saturday: Flat terrain, normal weather
  Sunday: Uneven terrain, extreme heat
-/
def brendan_cutting_capacity : ℝ :=
  let base_capacity := 8.0
  let flat_terrain_boost := 1.5
  let uneven_terrain_penalty := 0.65
  let rain_penalty := 0.8
  let extreme_heat_penalty := 0.9
  let monday_capacity := base_capacity * flat_terrain_boost
  let tuesday_capacity := monday_capacity * rain_penalty
  let wednesday_capacity := monday_capacity * uneven_terrain_penalty
  let thursday_capacity := monday_capacity * extreme_heat_penalty
  let friday_capacity := wednesday_capacity * rain_penalty
  let saturday_capacity := monday_capacity
  let sunday_capacity := wednesday_capacity * extreme_heat_penalty
  monday_capacity + tuesday_capacity + wednesday_capacity + thursday_capacity + friday_capacity + saturday_capacity + sunday_capacity

theorem brendan_weekly_capacity : brendan_cutting_capacity = 65.46 := 
by 
  sorry

end brendan_weekly_capacity_l747_74711


namespace geom_seq_sum_l747_74798

variable (a : ℕ → ℝ) (r : ℝ) (a1 a4 : ℝ)

theorem geom_seq_sum :
  (∀ n : ℕ, a (n + 1) = a n * r) → r = 2 → a 2 + a 3 = 4 → a 1 + a 4 = 6 :=
by
  sorry

end geom_seq_sum_l747_74798


namespace simplify_and_evaluate_l747_74705

noncomputable def x := Real.tan (Real.pi / 4) + Real.cos (Real.pi / 6)

theorem simplify_and_evaluate :
  ((x / (x ^ 2 - 1)) * ((x - 1) / x - 2)) = - (2 * Real.sqrt 3) / 3 := 
sorry

end simplify_and_evaluate_l747_74705


namespace animal_count_l747_74763

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end animal_count_l747_74763


namespace more_supermarkets_in_us_l747_74745

-- Definitions based on conditions
def total_supermarkets : ℕ := 84
def us_supermarkets : ℕ := 47
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

-- Prove that the number of more FGH supermarkets in the US than in Canada is 10
theorem more_supermarkets_in_us : us_supermarkets - canada_supermarkets = 10 :=
by
  -- adding 'sorry' as the proof
  sorry

end more_supermarkets_in_us_l747_74745


namespace speed_of_man_cycling_l747_74715

theorem speed_of_man_cycling (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : B = 3 * L)
  (h3 : L * B = 30000) (h4 : ∀ t : ℝ, t = 4 / 60): 
  ( (2 * L + 2 * B) / (4 / 60) ) = 12000 :=
by
  -- Assume given conditions
  sorry

end speed_of_man_cycling_l747_74715


namespace impossible_all_matches_outside_own_country_l747_74718

theorem impossible_all_matches_outside_own_country (n : ℕ) (h_teams : n = 16) : 
  ¬ ∀ (T : Fin n → Fin n → Prop), (∀ i j, i ≠ j → T i j) ∧ 
  (∀ i, ∀ j, i ≠ j → T i j → T j i) ∧ 
  (∀ i, T i i = false) → 
  ∀ i, ∃ j, T i j ∧ i ≠ j :=
by
  intro H
  sorry

end impossible_all_matches_outside_own_country_l747_74718


namespace polynomial_divisibility_l747_74790

theorem polynomial_divisibility (C D : ℝ) (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C * x + D = 0) :
  C + D = 2 := 
sorry

end polynomial_divisibility_l747_74790


namespace solve_some_number_l747_74782

theorem solve_some_number (n : ℝ) (h : (n * 10) / 100 = 0.032420000000000004) : n = 0.32420000000000004 :=
by
  -- The proof steps are omitted with 'sorry' here.
  sorry

end solve_some_number_l747_74782


namespace equal_lengths_l747_74727

noncomputable def F (x y z : ℝ) := (x+y+z) * (x+y-z) * (y+z-x) * (x+z-y)

variables {a b c d e f : ℝ}

axiom acute_angled_triangle (x y z : ℝ) : Prop

axiom altitudes_sum_greater (x y z : ℝ) : Prop

axiom cond1 : acute_angled_triangle a b c
axiom cond2 : acute_angled_triangle b d f
axiom cond3 : acute_angled_triangle a e f
axiom cond4 : acute_angled_triangle e c d

axiom cond5 : altitudes_sum_greater a b c
axiom cond6 : altitudes_sum_greater b d f
axiom cond7 : altitudes_sum_greater a e f
axiom cond8 : altitudes_sum_greater e c d

axiom cond9 : F a b c = F b d f
axiom cond10 : F a e f = F e c d

theorem equal_lengths : a = d ∧ b = e ∧ c = f := by
  sorry -- Proof not required.

end equal_lengths_l747_74727


namespace max_value_y_l747_74783

noncomputable def y (x : ℝ) : ℝ := x * (3 - 2 * x)

theorem max_value_y : ∃ x, 0 < x ∧ x < (3:ℝ) / 2 ∧ y x = 9 / 8 :=
by
  sorry

end max_value_y_l747_74783


namespace largest_sum_of_base8_digits_l747_74724

theorem largest_sum_of_base8_digits (a b c y : ℕ) (h1 : a < 8) (h2 : b < 8) (h3 : c < 8) (h4 : 0 < y ∧ y ≤ 16) (h5 : (a * 64 + b * 8 + c) * y = 512) :
  a + b + c ≤ 5 :=
sorry

end largest_sum_of_base8_digits_l747_74724


namespace tunnel_length_l747_74764

/-- A train travels at 80 kmph, enters a tunnel at 5:12 am, and leaves at 5:18 am.
    The length of the train is 1 km. Prove the length of the tunnel is 7 km. -/
theorem tunnel_length 
(speed : ℕ) (enter_time leave_time : ℕ) (train_length : ℕ) 
(h_enter : enter_time = 5 * 60 + 12) 
(h_leave : leave_time = 5 * 60 + 18) 
(h_speed : speed = 80) 
(h_train_length : train_length = 1) 
: ∃ tunnel_length : ℕ, tunnel_length = 7 :=
sorry

end tunnel_length_l747_74764


namespace original_plan_was_to_produce_125_sets_per_day_l747_74706

-- We state our conditions
def plans_to_complete_in_days : ℕ := 30
def produces_sets_per_day : ℕ := 150
def finishes_days_ahead_of_schedule : ℕ := 5

-- Calculations based on conditions
def actual_days_used : ℕ := plans_to_complete_in_days - finishes_days_ahead_of_schedule
def total_production : ℕ := produces_sets_per_day * actual_days_used
def original_planned_production_per_day : ℕ := total_production / plans_to_complete_in_days

-- Claim we want to prove
theorem original_plan_was_to_produce_125_sets_per_day :
  original_planned_production_per_day = 125 :=
by
  sorry

end original_plan_was_to_produce_125_sets_per_day_l747_74706


namespace maple_taller_than_pine_l747_74754

theorem maple_taller_than_pine :
  let pine_tree := 24 + 1/4
  let maple_tree := 31 + 2/3
  (maple_tree - pine_tree) = 7 + 5/12 :=
by
  sorry

end maple_taller_than_pine_l747_74754


namespace part1_part2_part3_l747_74734

-- Problem Definitions
def air_conditioner_cost (A B : ℕ → ℕ) :=
  A 3 + B 2 = 39000 ∧ 4 * A 1 - 5 * B 1 = 6000

def possible_schemes (A B : ℕ → ℕ) :=
  ∀ a b, a ≥ b / 2 ∧ 9000 * a + 6000 * b ≤ 217000 ∧ a + b = 30

def minimize_cost (A B : ℕ → ℕ) :=
  ∃ a, (a = 10 ∧ 9000 * a + 6000 * (30 - a) = 210000) ∧
  ∀ b, b ≥ 10 → b ≤ 12 → 9000 * b + 6000 * (30 - b) ≥ 210000

-- Theorem Statements
theorem part1 (A B : ℕ → ℕ) : air_conditioner_cost A B → A 1 = 9000 ∧ B 1 = 6000 :=
by sorry

theorem part2 (A B : ℕ → ℕ) : air_conditioner_cost A B →
  possible_schemes A B :=
by sorry

theorem part3 (A B : ℕ → ℕ) : air_conditioner_cost A B ∧ possible_schemes A B →
  minimize_cost A B :=
by sorry

end part1_part2_part3_l747_74734


namespace clerk_daily_salary_l747_74752

theorem clerk_daily_salary (manager_salary : ℝ) (num_managers num_clerks : ℕ) (total_salary : ℝ) (clerk_salary : ℝ)
  (h1 : manager_salary = 5)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16) :
  clerk_salary = 2 :=
by
  sorry

end clerk_daily_salary_l747_74752


namespace find_m_value_l747_74740

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end find_m_value_l747_74740


namespace value_of_a4_l747_74749

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem value_of_a4 {a : ℕ → ℝ} {S : ℕ → ℝ} (h1 : arithmetic_sequence a)
  (h2 : sum_of_arithmetic_sequence S a) (h3 : S 7 = 28) :
  a 4 = 4 := 
  sorry

end value_of_a4_l747_74749


namespace area_of_right_triangle_l747_74787

theorem area_of_right_triangle (a b c : ℝ) 
  (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 30 :=
by sorry

end area_of_right_triangle_l747_74787


namespace find_value_of_expr_l747_74786

variables (a b : ℝ)

def condition1 : Prop := a^2 + a * b = -2
def condition2 : Prop := b^2 - 3 * a * b = -3

theorem find_value_of_expr (h1 : condition1 a b) (h2 : condition2 a b) : a^2 + 4 * a * b - b^2 = 1 :=
sorry

end find_value_of_expr_l747_74786


namespace train_speed_l747_74775

theorem train_speed
  (distance_meters : ℝ := 400)
  (time_seconds : ℝ := 12)
  (distance_kilometers : ℝ := distance_meters / 1000)
  (time_hours : ℝ := time_seconds / 3600) :
  distance_kilometers / time_hours = 120 := by
  sorry

end train_speed_l747_74775


namespace ab_equals_six_l747_74710

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l747_74710


namespace y_share_l747_74767

theorem y_share (total_amount : ℝ) (x_share y_share z_share : ℝ)
  (hx : x_share = 1) (hy : y_share = 0.45) (hz : z_share = 0.30)
  (h_total : total_amount = 105) :
  (60 * y_share) = 27 :=
by
  have h_cycle : 1 + y_share + z_share = 1.75 := by sorry
  have h_num_cycles : total_amount / 1.75 = 60 := by sorry
  sorry

end y_share_l747_74767


namespace logician1_max_gain_l747_74722

noncomputable def maxCoinsDistribution (logician1 logician2 logician3 : ℕ) := (logician1, logician2, logician3)

theorem logician1_max_gain 
  (total_coins : ℕ) 
  (coins1 coins2 coins3 : ℕ) 
  (H : total_coins = 10)
  (H1 : ¬ (coins1 = 9 ∧ coins2 = 0 ∧ coins3 = 1) → coins1 = 2):
  maxCoinsDistribution coins1 coins2 coins3 = (9, 0, 1) :=
by
  sorry

end logician1_max_gain_l747_74722


namespace width_of_room_l747_74703

theorem width_of_room (length room_area cost paving_rate : ℝ) 
  (H_length : length = 5.5) 
  (H_cost : cost = 17600)
  (H_paving_rate : paving_rate = 800)
  (H_area : room_area = cost / paving_rate) :
  room_area = length * 4 :=
by
  -- sorry to skip proof
  sorry

end width_of_room_l747_74703


namespace moles_of_NaHSO4_l747_74735

def react_eq (naoh h2so4 nahso4 h2o : ℕ) : Prop :=
  naoh + h2so4 = nahso4 + h2o

theorem moles_of_NaHSO4
  (naoh h2so4 : ℕ)
  (h : 2 = naoh ∧ 2 = h2so4)
  (react : react_eq naoh h2so4 2 2):
  2 = 2 :=
by
  sorry

end moles_of_NaHSO4_l747_74735


namespace value_modulo_7_l747_74759

theorem value_modulo_7 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := 
  by 
  sorry

end value_modulo_7_l747_74759


namespace no_integer_n_exists_l747_74757

theorem no_integer_n_exists : ∀ (n : ℤ), n ^ 2022 - 2 * n ^ 2021 + 3 * n ^ 2019 ≠ 2020 :=
by sorry

end no_integer_n_exists_l747_74757


namespace ratio_of_democrats_l747_74797

theorem ratio_of_democrats (F M : ℕ) 
  (h1 : F + M = 990) 
  (h2 : (1 / 2 : ℚ) * F = 165) 
  (h3 : (1 / 4 : ℚ) * M = 165) : 
  (165 + 165) / 990 = 1 / 3 := 
by
  sorry

end ratio_of_democrats_l747_74797


namespace complement_intersection_eq_4_l747_74777

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection_eq_4 (hU : U = {0, 1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4}) :
  ((U \ A) ∩ B) = {4} :=
by {
  -- Proof goes here
  exact sorry
}

end complement_intersection_eq_4_l747_74777


namespace zero_in_tens_place_l747_74780

variable {A B : ℕ} {m : ℕ}

-- Define the conditions
def condition1 (A : ℕ) (B : ℕ) (m : ℕ) : Prop :=
  ∀ A B : ℕ, ∀ m : ℕ, A * 10^(m+1) + B = 9 * (A * 10^m + B)

theorem zero_in_tens_place (A B : ℕ) (m : ℕ) :
  condition1 A B m → m = 1 :=
by
  intro h
  sorry

end zero_in_tens_place_l747_74780


namespace find_x_l747_74751

noncomputable section

variable (x : ℝ)
def vector_v : ℝ × ℝ := (x, 4)
def vector_w : ℝ × ℝ := (5, 2)
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let num := (v.1 * w.1 + v.2 * w.2)
  let den := (w.1 * w.1 + w.2 * w.2)
  (num / den * w.1, num / den * w.2)

theorem find_x (h : projection (vector_v x) (vector_w) = (3, 1.2)) : 
  x = 47 / 25 :=
by
  sorry

end find_x_l747_74751


namespace length_of_train_l747_74776

-- Conditions
variable (L E T : ℝ)
axiom h1 : 300 * E = L + 300 * T
axiom h2 : 90 * E = L - 90 * T

-- The statement to be proved
theorem length_of_train : L = 200 * E :=
by
  sorry

end length_of_train_l747_74776


namespace last_child_loses_l747_74707

-- Definitions corresponding to conditions
def num_children := 11
def child_sequence := List.range' 1 num_children
def valid_two_digit_numbers := 90
def invalid_digit_sum_6 := 6
def invalid_digit_sum_9 := 9
def valid_numbers := valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9
def complete_cycles := valid_numbers / num_children
def remaining_numbers := valid_numbers % num_children

-- Statement to be proven
theorem last_child_loses (h1 : num_children = 11)
                         (h2 : valid_two_digit_numbers = 90)
                         (h3 : invalid_digit_sum_6 = 6)
                         (h4 : invalid_digit_sum_9 = 9)
                         (h5 : valid_numbers = valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9)
                         (h6 : remaining_numbers = valid_numbers % num_children) :
  (remaining_numbers = 9) ∧ (num_children - remaining_numbers = 2) :=
by
  sorry

end last_child_loses_l747_74707


namespace line_ellipse_common_points_l747_74708

def point (P : Type*) := P → ℝ × ℝ

theorem line_ellipse_common_points
  (m n : ℝ)
  (no_common_points_with_circle : ∀ (x y : ℝ), mx + ny - 3 = 0 → x^2 + y^2 ≠ 3) :
  ∀ (Px Py : ℝ), (Px = m ∧ Py = n) →
  (∃ (x1 y1 x2 y2 : ℝ), ((x1^2 / 7) + (y1^2 / 3) = 1 ∧ (x2^2 / 7) + (y2^2 / 3) = 1) ∧ (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end line_ellipse_common_points_l747_74708


namespace factor_expression_l747_74756

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l747_74756


namespace ages_of_three_persons_l747_74779

theorem ages_of_three_persons (y m e : ℕ) 
  (h1 : e = m + 16)
  (h2 : m = y + 8)
  (h3 : e - 6 = 3 * (y - 6))
  (h4 : e - 6 = 2 * (m - 6)) :
  y = 18 ∧ m = 26 ∧ e = 42 := 
by 
  sorry

end ages_of_three_persons_l747_74779


namespace range_a_two_zeros_l747_74785

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- The theorem statement about the range of a
theorem range_a_two_zeros (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 1 ≤ a ∧ a ≤ 5 := sorry

end range_a_two_zeros_l747_74785


namespace perimeter_of_nonagon_l747_74762

-- Definitions based on the conditions
def sides := 9
def side_length : ℝ := 2

-- The problem statement in Lean
theorem perimeter_of_nonagon : sides * side_length = 18 := 
by sorry

end perimeter_of_nonagon_l747_74762


namespace union_of_sets_l747_74720

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end union_of_sets_l747_74720


namespace total_fires_l747_74772

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end total_fires_l747_74772


namespace perpendicular_lines_b_eq_neg9_l747_74784

-- Definitions for the conditions.
def eq1 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def eq2 (b x y : ℝ) : Prop := b * x + 3 * y + 4 = 0

-- The problem statement
theorem perpendicular_lines_b_eq_neg9 (b : ℝ) : 
  (∀ x y, eq1 x y → eq2 b x y) ∧ (∀ x y, eq2 b x y → eq1 x y) → b = -9 :=
by
  sorry

end perpendicular_lines_b_eq_neg9_l747_74784


namespace avg_of_numbers_l747_74742

theorem avg_of_numbers (a b c d : ℕ) (avg : ℕ) (h₁ : a = 6) (h₂ : b = 16) (h₃ : c = 8) (h₄ : d = 22) (h₅ : avg = 13) :
  (a + b + c + d) / 4 = avg := by
  -- Proof here
  sorry

end avg_of_numbers_l747_74742


namespace karen_packs_cookies_l747_74743

-- Conditions stated as definitions
def school_days := 5
def peanut_butter_days := 2
def ham_sandwich_days := school_days - peanut_butter_days
def cake_days := 1
def probability_ham_and_cake := 0.12

-- Lean theorem statement
theorem karen_packs_cookies : 
  (school_days - cake_days - peanut_butter_days) = 2 :=
by
  sorry

end karen_packs_cookies_l747_74743


namespace sequence_odd_l747_74761

theorem sequence_odd (a : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = 7)
  (hr : ∀ n ≥ 2, -1 < (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ∧ (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ≤ 1) :
  ∀ n > 1, Odd (a n) := 
  sorry

end sequence_odd_l747_74761


namespace necessary_but_not_sufficient_l747_74744

theorem necessary_but_not_sufficient (x y : ℕ) : x + y = 3 → (x = 1 ∧ y = 2) ↔ (¬ (x = 0 ∧ y = 3)) := by
  sorry

end necessary_but_not_sufficient_l747_74744


namespace cow_manure_plant_height_l747_74794

theorem cow_manure_plant_height
  (control_plant_height : ℝ)
  (bone_meal_ratio : ℝ)
  (cow_manure_ratio : ℝ)
  (h1 : control_plant_height = 36)
  (h2 : bone_meal_ratio = 1.25)
  (h3 : cow_manure_ratio = 2) :
  (control_plant_height * bone_meal_ratio * cow_manure_ratio) = 90 :=
sorry

end cow_manure_plant_height_l747_74794


namespace number_of_machines_sold_l747_74788

-- Define the parameters and conditions given in the problem
def commission_of_first_150 (sale_price : ℕ) : ℕ := 150 * (sale_price * 3 / 100)
def commission_of_next_100 (sale_price : ℕ) : ℕ := 100 * (sale_price * 4 / 100)
def commission_of_after_250 (sale_price : ℕ) (x : ℕ) : ℕ := x * (sale_price * 5 / 100)

-- Define the total commission using these commissions
def total_commission (x : ℕ) : ℕ :=
  commission_of_first_150 10000 + 
  commission_of_next_100 9500 + 
  commission_of_after_250 9000 x

-- The main statement we want to prove
theorem number_of_machines_sold (x : ℕ) (total_commission : ℕ) : x = 398 ↔ total_commission = 150000 :=
by
  sorry

end number_of_machines_sold_l747_74788


namespace range_f_l747_74736

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_f : (Set.range f) = (Set.univ \ { -27 }) :=
by
  sorry

end range_f_l747_74736


namespace sixth_graders_count_l747_74746

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end sixth_graders_count_l747_74746


namespace oranges_taken_by_susan_l747_74750

-- Defining the conditions
def original_number_of_oranges_in_box : ℕ := 55
def oranges_left_in_box_after_susan_takes : ℕ := 20

-- Statement to prove:
theorem oranges_taken_by_susan :
  original_number_of_oranges_in_box - oranges_left_in_box_after_susan_takes = 35 :=
by
  sorry

end oranges_taken_by_susan_l747_74750


namespace union_of_sets_l747_74769

def setA := {x : ℝ | x^2 < 4}
def setB := {y : ℝ | ∃ x ∈ setA, y = x^2 - 2 * x - 1}

theorem union_of_sets : (setA ∪ setB) = {x : ℝ | -2 ≤ x ∧ x < 7} :=
by sorry

end union_of_sets_l747_74769


namespace eight_digit_number_min_max_l747_74796

theorem eight_digit_number_min_max (Amin Amax B : ℕ) 
  (hAmin: Amin = 14444446) 
  (hAmax: Amax = 99999998) 
  (hB_coprime: Nat.gcd B 12 = 1) 
  (hB_length: 44444444 < B) 
  (h_digits: ∀ (b : ℕ), b < 10 → ∃ (A : ℕ), A = 10^7 * b + (B - b) / 10 ∧ A < 100000000) :
  (∃ b, Amin = 10^7 * b + (44444461 - b) / 10 ∧ Nat.gcd 44444461 12 = 1 ∧ 44444444 < 44444461) ∧
  (∃ b, Amax = 10^7 * b + (999999989 - b) / 10 ∧ Nat.gcd 999999989 12 = 1 ∧ 44444444 < 999999989) :=
  sorry

end eight_digit_number_min_max_l747_74796


namespace problem_l747_74747

open Real

noncomputable def f (x : ℝ) : ℝ := x + 1

theorem problem (f : ℝ → ℝ)
  (h : ∀ x, 2 * f x - f (-x) = 3 * x + 1) :
  f 1 = 2 :=
by
  sorry

end problem_l747_74747


namespace inv_func_eval_l747_74741

theorem inv_func_eval (a : ℝ) (h : 8^(1/3) = a) : (fun y => (Real.log y / Real.log 8)) (a + 2) = 2/3 :=
by
  sorry

end inv_func_eval_l747_74741


namespace balance_four_heartsuits_with_five_circles_l747_74781

variables (x y z : ℝ)

-- Given conditions
axiom condition1 : 4 * x + 3 * y = 12 * z
axiom condition2 : 2 * x = y + 3 * z

-- Statement to prove
theorem balance_four_heartsuits_with_five_circles : 4 * y = 5 * z :=
by sorry

end balance_four_heartsuits_with_five_circles_l747_74781


namespace domain_of_f_l747_74728

open Real

noncomputable def f (x : ℝ) : ℝ := log (log x)

theorem domain_of_f : { x : ℝ | 1 < x } = { x : ℝ | ∃ y > 1, x = y } :=
by
  sorry

end domain_of_f_l747_74728


namespace infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l747_74789

theorem infinite_n_square_plus_one_divides_factorial :
  ∃ (infinitely_many n : ℕ), (n^2 + 1) ∣ (n!) := sorry

theorem infinite_n_square_plus_one_not_divide_factorial :
  ∃ (infinitely_many n : ℕ), ¬((n^2 + 1) ∣ (n!)) := sorry

end infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l747_74789


namespace ishas_pencil_initial_length_l747_74773

theorem ishas_pencil_initial_length (l : ℝ) (h1 : l - 4 = 18) : l = 22 :=
by
  sorry

end ishas_pencil_initial_length_l747_74773


namespace cos_value_in_second_quadrant_l747_74766

theorem cos_value_in_second_quadrant {B : ℝ} (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = - (12 / 13) :=
sorry

end cos_value_in_second_quadrant_l747_74766


namespace find_x_l747_74739

theorem find_x (x : ℕ) (hx1 : 1 ≤ x) (hx2 : x ≤ 100) (hx3 : (31 + 58 + 98 + 3 * x) / 6 = 2 * x) : x = 21 :=
by
  sorry

end find_x_l747_74739
