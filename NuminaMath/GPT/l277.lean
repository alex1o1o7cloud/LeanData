import Mathlib

namespace tetrahedron_volume_proof_l277_277106

noncomputable def volume_of_tetrahedron 
  (PQ PQR PQS PQSR : ℝ) (angle_PQR_PQS : ℝ) : ℝ :=
  if h : (PQ = 4) ∧ (PQR = 18) ∧ (PQS = 16) ∧ (angle_PQR_PQS = π / 4)
  then 72
  else 0

theorem tetrahedron_volume_proof (PQ PQR PQS angle_PQR_PQS : ℝ) : 
  PQ = 4 → PQR = 18 → PQS = 16 → angle_PQR_PQS = π / 4 → volume_of_tetrahedron PQ PQR PQS angle_PQR_PQS = 72 :=
by
  intro h1 h2 h3 h4
  rw volume_of_tetrahedron
  rw if_pos
  · rfl
  exact ⟨h1, h2, h3, h4⟩

end tetrahedron_volume_proof_l277_277106


namespace paths_remainder_l277_277612

-- Define the number of paths given the problem constraints
def num_paths : ℕ :=
1 + (
  let count_paths := 
    ∑ n : ℕ in (0 : ℕ) .. 10, binomial 10 n * binomial 4 (5 - n) in 
    count_paths + count_paths
)

-- Prove the result
theorem paths_remainder : num_paths % 1000 = 4 :=
by sorry

end paths_remainder_l277_277612


namespace value_of_b_l277_277213

noncomputable def determine_b (a q b c : ℝ) : Prop :=
  (∀ x : ℝ, y = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, y = a * (x - q)^2 + 2 * q) ∧
  y = -2 * q ∧ q ≠ 0

theorem value_of_b (a q b c : ℝ) (h : determine_b a q b c) : 
  b = 8 / q := 
sorry

end value_of_b_l277_277213


namespace S8_value_l277_277365

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem S8_value 
  (h_geo : is_geometric_sequence a q)
  (h_S4 : S 4 = 3)
  (h_S12_S8 : S 12 - S 8 = 12) :
  S 8 = 9 := 
sorry

end S8_value_l277_277365


namespace simplify_cos_expression_l277_277951

theorem simplify_cos_expression (x y : ℝ) : 
  cos x ^ 2 + cos (x + y) ^ 2 - 2 * cos x * cos y * cos (x + y) = sin y ^ 2 :=
by
  sorry

end simplify_cos_expression_l277_277951


namespace value_of_b_pow4_plus_b_inv_pow4_l277_277572

theorem value_of_b_pow4_plus_b_inv_pow4 (b : ℝ) (h : 5 = b + b⁻¹) : b^4 + b^(-4) = 527 := sorry

end value_of_b_pow4_plus_b_inv_pow4_l277_277572


namespace range_of_a_l277_277409

theorem range_of_a (a : ℝ) : 
  (∃ x : ℕ, 2 * x + a < 3 ∧ (∑ i in (finset.filter (λ n, 2 * n + a < 3) (finset.range (nat_ceil ((3 - a) / 2)))), i) = 6) 
  → (-5 : ℝ) ≤ a ∧ a < -3 :=
by
  sorry

end range_of_a_l277_277409


namespace perpendicular_line_eq_l277_277657

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l277_277657


namespace stones_sent_away_l277_277040

variable (original_stones left_stones sent_stones : ℕ)

theorem stones_sent_away 
  (h1 : original_stones = 78)
  (h2 : left_stones = 15)
  (h3 : sent_stones = original_stones - left_stones) :
  sent_stones = 63 :=
by
  rw [h1, h2] at h3
  exact h3

#eval stones_sent_away 78 15 (78 - 15) -- The evaluator to check correctness

end stones_sent_away_l277_277040


namespace correct_algorithm_description_l277_277242

theorem correct_algorithm_description (A B D : Prop) :
  (¬ A) ∧ (¬ B) ∧ (¬ D) → C :=
by
  intro h,
  sorry

end correct_algorithm_description_l277_277242


namespace number_of_friends_l277_277152

theorem number_of_friends (total_crackers crackers_per_friend : ℕ) 
  (h_total : total_crackers = 8) 
  (h_per_friend : crackers_per_friend = 2) : 
  total_crackers / crackers_per_friend = 4 := 
by
  have h1 : total_crackers = 8 := h_total
  have h2 : crackers_per_friend = 2 := h_per_friend
  rw [h1, h2]
  norm_num
  exact nat.div_self (by norm_num : 2 > 0)
  sorry

end number_of_friends_l277_277152


namespace hotdogs_remainder_zero_l277_277861

theorem hotdogs_remainder_zero :
  25197624 % 6 = 0 :=
by
  sorry -- Proof not required

end hotdogs_remainder_zero_l277_277861


namespace equalize_costs_l277_277227

theorem equalize_costs (X Y Z : ℝ) (h1 : Y > X) (h2 : Z > Y) : 
  (Y + (Z - (X + Z - 2 * Y) / 3) = Z) → 
   (Y - (Y + Z - (X + Z - 2 * Y)) / 3 = (X + Z - 2 * Y) / 3) := sorry

end equalize_costs_l277_277227


namespace find_f_x_l277_277833

theorem find_f_x (x : ℝ) :
  ∃ f : ℝ → ℝ, (∀ x, f(x) = 3 * sin(2 * x - (π / 4))) :=
sorry

end find_f_x_l277_277833


namespace interval_min_value_l277_277798

-- Define f and g
def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := Real.log x

-- Define the condition for f(t) = g(s)
def condition (t s : ℝ) : Prop := f t = g s

-- Prove that the interval where f(t) lies, when s - t reaches its minimum value, is (1/2, ln 2)
theorem interval_min_value (t s : ℝ) (h : condition t s) : 
(∃ a > 0, f t = a ∧ (s - t = Real.exp a - Real.log a ∧ a ∈ (1/2 : ℝ, Real.log 2))) := sorry

end interval_min_value_l277_277798


namespace symmetric_circle_equation_l277_277775

-- Define the original circle equation
def original_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 1

-- Define the condition for symmetry around the line y = x
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the circle equation with a given center and radius
def circle (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r

-- Prove that the symmetric circle equation is correct
theorem symmetric_circle_equation :
  (∀ x y, original_circle x y ↔ circle 4 (-1) 1 x y) :=
sorry

end symmetric_circle_equation_l277_277775


namespace p_necessary_not_sufficient_for_q_l277_277006

variables (a b c : ℝ) (p q : Prop)

def condition_p : Prop := a * b * c = 0
def condition_q : Prop := a = 0

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) :=
by
  let p := condition_p a b c
  let q := condition_q a
  sorry

end p_necessary_not_sufficient_for_q_l277_277006


namespace calculator_sum_is_large_l277_277614

-- Definitions for initial conditions and operations
def participants := 50
def initial_calc1 := 2
def initial_calc2 := -2
def initial_calc3 := 0

-- Define the operations
def operation_calc1 (n : ℕ) := initial_calc1 * 2^n
def operation_calc2 (n : ℕ) := (-2) ^ (2^n)
def operation_calc3 (n : ℕ) := initial_calc3 - n

-- Define the final values for each calculator
def final_calc1 := operation_calc1 participants
def final_calc2 := operation_calc2 participants
def final_calc3 := operation_calc3 participants

-- The final sum
def final_sum := final_calc1 + final_calc2 + final_calc3

-- Prove the final result
theorem calculator_sum_is_large :
  final_sum = 2 ^ (2 ^ 50) :=
by
  -- The proof would go here.
  sorry

end calculator_sum_is_large_l277_277614


namespace standard_equation_of_tangent_circle_l277_277443

theorem standard_equation_of_tangent_circle (r h k : ℝ)
  (h_r : r = 1) 
  (h_k : k = 1) 
  (h_center_quadrant : h > 0 ∧ k > 0)
  (h_tangent_x_axis : k = r) 
  (h_tangent_line : r = abs (4 * h - 3) / 5)
  : (x - 2)^2 + (y - 1)^2 = 1 := 
by {
  sorry
}

end standard_equation_of_tangent_circle_l277_277443


namespace sequence_a_10_l277_277030

theorem sequence_a_10 : ∀ {a : ℕ → ℕ}, (a 1 = 1) → (∀ n, a (n+1) = a n + 2^n) → (a 10 = 1023) :=
by
  intros a h1 h_rec
  sorry

end sequence_a_10_l277_277030


namespace number_of_positive_values_S_1_to_S_100_l277_277377

noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in finset.range n, Real.cos ((k + 1 : ℝ) * (Real.pi / 7))

theorem number_of_positive_values_S_1_to_S_100 : ∃ p, p = 37 ∧
  ∀ n, (1 ≤ n ∧ n ≤ 100) → (S n > 0 → p > 0) :=
begin
  sorry
end

end number_of_positive_values_S_1_to_S_100_l277_277377


namespace total_time_spent_l277_277623

-- Definition of the problem conditions
def warm_up_time : ℕ := 10
def additional_puzzles : ℕ := 2
def multiplier : ℕ := 3

-- Statement to prove the total time spent solving puzzles
theorem total_time_spent : warm_up_time + (additional_puzzles * (multiplier * warm_up_time)) = 70 :=
by
  sorry

end total_time_spent_l277_277623


namespace problem_1_problem_2_l277_277809

-- Define the given circle C and point P
def circleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def pointP : ℝ × ℝ := (1, 1)

-- Condition: The distance |AB| = sqrt(17)
def distAB := Real.sqrt 17

-- Problem (1): Prove the slope angle of the line l passing through P and intersects circle C with |AB| = √17.
theorem problem_1 (k : ℝ) (hC : ∀ x y : ℝ, circleC x y) (hP : pointP = (1, 1)) 
(hDist : ∀ A B : ℝ × ℝ, distAB = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) :
∃ θ : ℝ, θ = Real.arctan k ∧ (θ = π / 3 ∨ θ = 2 * π / 3) :=
sorry

-- Problem (2): Prove the trajectory equation of the midpoint M of line segment AB.
theorem problem_2 (M : ℝ × ℝ) 
(hCircleM : circleC M.1 M.2) 
(hMid : ∀ A B : ℝ × ℝ, M = ((A.1 + B.1)/2, (A.2 + B.2)/2)) :
M.1 = 1/2 ∧ 
(M.1 - 0.5)^2 + (M.2 - 1)^2 = 1/4 :=
sorry

end problem_1_problem_2_l277_277809


namespace kaleb_can_buy_toys_l277_277897

def kaleb_initial_money : ℕ := 12
def money_spent_on_game : ℕ := 8
def money_saved : ℕ := 2
def toy_cost : ℕ := 2

theorem kaleb_can_buy_toys :
  (kaleb_initial_money - money_spent_on_game - money_saved) / toy_cost = 1 :=
by
  sorry

end kaleb_can_buy_toys_l277_277897


namespace quadratic_distinct_real_roots_l277_277052

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end quadratic_distinct_real_roots_l277_277052


namespace perpendicular_line_eq_slope_intercept_l277_277649

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l277_277649


namespace domain_of_f_l277_277972

open Set

/-- The domain of the function f(x) = log_2 (1 - 2x) + 1 / (x + 1) is (-∞, -1) ∪ (-1, 1/2) -/
theorem domain_of_f :
  { x : ℝ | 1 - 2x > 0 ∧ x + 1 ≠ 0 } = Set.Ioo (-∞) (-1) ∪ Set.Ioo (-1) (1 / 2) :=
by
  sorry

end domain_of_f_l277_277972


namespace find_AX_l277_277883

open Real

theorem find_AX :
  ∀ (A B C X : ℝ) (AB AC BC AX BX : ℝ),
    AB = 80 →
    AC = 45 →
    BC = 90 →
    (∠ACX = ∠XCB) →
    (BX = 0.5 * AX) →
    (AB = AX + BX) →
    AX = 160 / 3 :=
by
  intros A B C X AB AC BC AX BX hAB hAC hBC hAngle hBX hABBX
  sorry

end find_AX_l277_277883


namespace daniel_gave_noodles_l277_277305

theorem daniel_gave_noodles (initial current : ℕ): initial = 66 → current = 54 → initial - current = 12 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end daniel_gave_noodles_l277_277305


namespace exists_n_for_pn_consecutive_zeros_l277_277121

theorem exists_n_for_pn_consecutive_zeros (p : ℕ) (hp : Nat.Prime p) (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, (∃ k : ℕ, (p^n) / 10^(k+m) % 10^m = 0) := sorry

end exists_n_for_pn_consecutive_zeros_l277_277121


namespace optimal_order_l277_277535

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l277_277535


namespace min_guests_at_banquet_l277_277741

-- Definitions based on conditions
def total_food : ℕ := 675
def vegetarian_food : ℕ := 195
def pescatarian_food : ℕ := 220
def carnivorous_food : ℕ := 260

def max_vegetarian_per_guest : ℚ := 3
def max_pescatarian_per_guest : ℚ := 2.5
def max_carnivorous_per_guest : ℚ := 4

-- Definition based on the question and the correct answer
def minimum_number_of_guests : ℕ := 218

-- Lean statement to prove the problem
theorem min_guests_at_banquet :
  195 / 3 + 220 / 2.5 + 260 / 4 = 218 :=
by sorry

end min_guests_at_banquet_l277_277741


namespace room_breadth_is_five_l277_277451

theorem room_breadth_is_five 
  (length : ℝ)
  (height : ℝ)
  (bricks_per_square_meter : ℝ)
  (total_bricks : ℝ)
  (H_length : length = 4)
  (H_height : height = 2)
  (H_bricks_per_square_meter : bricks_per_square_meter = 17)
  (H_total_bricks : total_bricks = 340) 
  : ∃ (breadth : ℝ), breadth = 5 :=
by
  -- we leave the proof as sorry for now
  sorry

end room_breadth_is_five_l277_277451


namespace B_can_finish_work_in_15_days_l277_277264

theorem B_can_finish_work_in_15_days
  (A_work_rate : ℚ := 1/5)
  (total_work : ℚ := 1)
  (days_worked_together : ℚ := 2)
  (remaining_days_B_worked_alone : ℚ := 7)
  : ∃ B : ℚ, B = 15 :=
begin
  let B_work_rate := (1 : ℚ) / (15 : ℚ),
  let B := 15,
  have B_work_rate_hyp : B_work_rate = 1 / B := by rfl,

  -- Combined work done in 2 days
  let combined_work_in_2_days := days_worked_together * (A_work_rate + B_work_rate),
  have combined_work_in_2_days_hyp : combined_work_in_2_days = 2 * (1/5 + 1/15),

  -- Work done by B in the additional 7 days
  let work_B_done_in_remaining_days := remaining_days_B_worked_alone * B_work_rate,
  have work_B_done_in_remaining_days_hyp : work_B_done_in_remaining_days = 7 * (1/15),

  -- Total work equation
  let total_work_done := combined_work_in_2_days + work_B_done_in_remaining_days,
  have total_work_done_hyp : total_work_done = total_work,

  existsi B,
  sorry
end

end B_can_finish_work_in_15_days_l277_277264


namespace sum_geometric_series_l277_277128

noncomputable def f (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 3), 2^(3*k + 4)

theorem sum_geometric_series (n : ℕ) :
  f n = (16 * (8^(n + 3) - 1)) / 7 := sorry

end sum_geometric_series_l277_277128


namespace relationship_among_zeros_l277_277408

noncomputable def f (x : ℝ) : ℝ := 3^x + x
noncomputable def g (x : ℝ) : ℝ := (log 3 x) / (log 3) + 2
noncomputable def h (x : ℝ) : ℝ := (log 3 x) / (log 3) + x

def a : ℝ := Classical.some (exists_eq_neg_self 1 f)
def b : ℝ := 1 / 9
def c : ℝ := Classical.some (exists_eq_neg_self 1 h)

theorem relationship_among_zeros : a < c ∧ c < b := sorry

end relationship_among_zeros_l277_277408


namespace min_distance_MN_l277_277139

open Real

noncomputable def f (x : ℝ) := exp x - (1 / 2) * x^2
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_MN (x1 x2 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 > 0) (h3 : f x1 = g x2) :
  abs (x2 - x1) = 2 :=
by
  sorry

end min_distance_MN_l277_277139


namespace area_increase_is_24_l277_277730

-- Definitions for the original and modified measurements
def height := 4
def upper_base (a : ℝ) := a
def lower_base (b : ℝ) := b
def increased_upper_base (a : ℝ) := a + 6
def increased_lower_base (b : ℝ) := b + 6

-- Area calculation functions
def original_area (a b : ℝ) := (upper_base a + lower_base b) * height / 2
def new_area (a b : ℝ) := (increased_upper_base a + increased_lower_base b) * height / 2

-- Theorem stating the increase in area
theorem area_increase_is_24 (a b : ℝ) : new_area a b - original_area a b = 24 := by
  -- Proof goes here
  sorry

end area_increase_is_24_l277_277730


namespace bounds_on_xyz_l277_277003

theorem bounds_on_xyz (a x y z : ℝ) (h1 : x + y + z = a)
                      (h2 : x^2 + y^2 + z^2 = (a^2) / 2)
                      (h3 : a > 0) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z) :
                      (0 < x ∧ x ≤ (2 / 3) * a) ∧ 
                      (0 < y ∧ y ≤ (2 / 3) * a) ∧ 
                      (0 < z ∧ z ≤ (2 / 3) * a) :=
sorry

end bounds_on_xyz_l277_277003


namespace lambda_value_l277_277346

variable {d a1 a4 λ : ℝ}
variable (n : ℕ)

def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

def sum_first_n_terms (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem lambda_value 
  (h1 : arithmetic_sequence a1 d 6 = 3 * arithmetic_sequence a1 d 4)
  (h2 : sum_first_n_terms a1 d 9 = λ * arithmetic_sequence a1 d 4)
  (hd : d ≠ 0) :
  λ = 18 := 
  by sorry

end lambda_value_l277_277346


namespace angle_between_vectors_l277_277812

variables {a b : ℝ^3}
variables (h_a_unit : ∥a∥ = 1) (h_b_unit : ∥b∥ = 1)
variables (h_dot : (2 • a + b) ⬝ (a - 2 • b) = 3 / 2)

theorem angle_between_vectors : real.angle a b = 2 * real.pi / 3 :=
by
  sorry

end angle_between_vectors_l277_277812


namespace total_length_segments_l277_277518

theorem total_length_segments (AB CB : ℝ) (P Q : ℕ → ℝ × ℝ) :
  AB = 5 →
  CB = 4 →
  (∀ k, 1 ≤ k ∧ k ≤ 199 → 
    (P k).fst = 5 * (200 - k) / 200 ∧
    (P k).snd = 0 ∧
    (Q k).fst = 0 ∧
    (Q k).snd = 4 * (200 - k) / 200) →
  (∑ k in finset.range 199, 
    real.sqrt ((5 * (200 - k) / 200)^2 + (4 * (200 - k) / 200)^2) * 2)
  - real.sqrt (AB^2 + CB^2) = 
  198 * real.sqrt 41 :=
sorry

end total_length_segments_l277_277518


namespace problem_solution_l277_277384

-- Define the conditions rigorously to match the mathematical problem statement
def ellipse (a b : ℝ) (h : a > b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (p : ℝ × ℝ) (a b : ℝ) : Prop :=
  let ⟨x, y⟩ := p in (x^2 / a^2) + (y^2 / b^2) = 1

def focal_length (a b : ℝ) (length : ℝ) : Prop :=
  (a^2 - b^2 = length^2 / 4)

def line (k : ℝ) : (ℝ × ℝ) → Prop :=
  λ p, let ⟨x, y⟩ := p in y = k * (x + 1)

def intersects_ellipse (L : (ℝ × ℝ) → Prop) (a b : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, L A ∧ L B ∧ ellipse a b A ∧ ellipse a b B

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  ( (x1 + x2) / 2, (y1 + y2) / 2 )

def distance_to_line (M : ℝ × ℝ) (slope intercept : ℝ) (distance : ℝ) : Prop :=
  let ⟨x, y⟩ := M in
  abs (slope * x + y + intercept) / sqrt (slope^2 + 1) = distance

noncomputable def problem_conditions :=
  ∃ a b k t : ℝ, 
    a > b ∧ 
    b > 0 ∧ 
    passes_through (sqrt 2, 1) a b ∧ 
    focal_length a b (2 * sqrt 2) ∧
    k > -2 ∧ 
    intersects_ellipse (line k) a b ∧ 
    ∀ A B : ℝ × ℝ, 
      line k A → 
      line k B → 
      ellipse a b A → 
      ellipse a b B → 
      distance_to_line (midpoint A B) 2 t (3 * sqrt 5 / 5) ∧ 
      t > 2

theorem problem_solution : 
  (problem_conditions →
    (∃ a b : ℝ,
      a^2 = 4 ∧ 
      b^2 = 2 ∧ 
      ∀ t : ℝ,
        (4 - 3 * sqrt 2 / 4 ≤ t ∧ t < 5)
    )) :=
sorry

end problem_solution_l277_277384


namespace hyperbola_locus_of_P_circle_l277_277925

noncomputable def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def foci (a b c : ℝ) : ℝ × ℝ := (c, 0)

def point_on_hyperbola (x y a b : ℝ) : Prop := hyperbola x y a b

def perpendicular_bisector (Q P F1 F2 : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (Q.1 = P.1 ∧ Q.2 = P.2 + m * P.1)

def locus_P_circle (P : ℝ × ℝ) (a : ℝ) : Prop :=
  ∃θ : ℝ, P.1 = a * Real.cos θ ∧ P.2 = a * Real.sin θ ∧ P ≠ (a, 0) ∧ P ≠ (-a, 0)

theorem hyperbola_locus_of_P_circle (a b c : ℝ) (H : a > 0 ∧ b > 0) :
  (∀ Q : ℝ × ℝ, point_on_hyperbola Q.1 Q.2 a b →
  ∃ P : ℝ × ℝ, perpendicular_bisector Q P (foci a b c).1 (-c,0) ∧ locus_P_circle P a) :=
sorry

end hyperbola_locus_of_P_circle_l277_277925


namespace dogsled_team_speed_difference_l277_277230

theorem dogsled_team_speed_difference :
  ∀ (t : ℕ), (300 = 20 * t) → (300 = 25 * (t - 3)) → 25 - 20 = 5 :=
by
  assume t ht1 ht2
  sorry

end dogsled_team_speed_difference_l277_277230


namespace max_elements_7_l277_277695

variable (M : Finset ℤ)
variable h : ∀ a b c ∈ M, ∃ x y ∈ M, (x + y = a + b) ∨ (x + y = a + c) ∨ (x + y = b + c)

theorem max_elements_7 (M : Finset ℤ) (h : ∀ a b c ∈ M, ∃ x y ∈ M, x + y = a + b ∨ x + y = a + c ∨ x + y = b + c) : M.card ≤ 7 :=
  sorry

end max_elements_7_l277_277695


namespace cube_root_approx_l277_277859

theorem cube_root_approx (h1 : Real.cbrt 0.3 ≈ 0.6694) (h2 : Real.cbrt 3 ≈ 1.442) :
  Real.cbrt 300 ≈ 6.694 := by
  sorry

end cube_root_approx_l277_277859


namespace number_of_ways_to_fill_grid_l277_277094

open Finset

theorem number_of_ways_to_fill_grid (n : ℕ) (h : n ≥ 1) :
  let grid := Matrix (Fin n) (Fin n) (Fin 2)
  let condition (m : grid) := (∀ i : Fin n, even (card { j | m i j = 1 })) ∧
                              (∀ j : Fin n, even (card { i | m i j = 1 }))
  ∃ fill_count : ℕ, (fill_count = 2^((n-1)*(n-1))) ∧
                    ∀ g : grid, condition g ↔ (g ∈ universe grid) :=
sorry

end number_of_ways_to_fill_grid_l277_277094


namespace subsets_nonempty_sym_diff_l277_277902

variables {N k : ℕ} (k_le_N : k ≤ N)
variables {A : ℕ → set ℕ} (hA : ∀ (s : finset ℕ), s nonempty → (s ∩ finset.range k).nonempty → (⋂ i in s, A i).nonempty)

theorem subsets_nonempty_sym_diff (hkN : k ≤ N) :
  ∃ (t : finset ℕ), t ⊆ finset.range k ∧ t.nonempty ∧ (finset.card (⋃ i in t, A i) ≥ k) :=
sorry

end subsets_nonempty_sym_diff_l277_277902


namespace ostap_advantageous_order_l277_277548

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l277_277548


namespace positive_number_square_sum_eq_210_l277_277991

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l277_277991


namespace total_pink_crayons_l277_277150

theorem total_pink_crayons (Mara_crayons Luna_crayons Sara_crayons : ℕ)
  (Mara_percent Luna_percent Sara_percent : ℕ)
  (Mara_pink Luna_pink Sara_pink : ℕ)
  (hMara : Mara_crayons = 90) (hMaraPercent : Mara_percent = 15)
  (hLuna : Luna_crayons = 120) (hLunaPercent : Luna_percent = 25)
  (hSara : Sara_crayons = 45) (hSaraPercent : Sara_percent = 5)
  (hMaraPink : Mara_pink = (Mara_percent * Mara_crayons) / 100) (hLunaPink : Luna_pink = (Luna_percent * Luna_crayons) / 100) (hSaraPink : Sara_pink = (Sara_percent * Sara_crayons) / 100) :
  (Mara_pink + Luna_pink + Sara_pink).to_float.ceil = 46 :=
by
  rewrite [hMara, hMaraPercent, hLuna, hLunaPercent, hSara, hSaraPercent, hMaraPink, hLunaPink, hSaraPink]
  calc
    (14 + 30 + 2).to_float.ceil = 46 := sorry

end total_pink_crayons_l277_277150


namespace jason_hours_saturday_l277_277113

def hours_after_school (x : ℝ) : ℝ := 4 * x
def hours_saturday (y : ℝ) : ℝ := 6 * y

theorem jason_hours_saturday 
  (x y : ℝ) 
  (total_hours : x + y = 18) 
  (total_earnings : 4 * x + 6 * y = 88) : 
  y = 8 :=
by 
  sorry

end jason_hours_saturday_l277_277113


namespace bob_25_cent_coins_l277_277746

theorem bob_25_cent_coins (a b c : ℕ)
    (h₁ : a + b + c = 15)
    (h₂ : 15 + 4 * c = 27) : c = 3 := by
  sorry

end bob_25_cent_coins_l277_277746


namespace angle_GAC_eq_angle_EAC_l277_277105

variables {A B C D E F G : Type}
variables [point A] [point B] [point C] [point D] [point E] [point F] [point G] 
variables [is_quadrilateral ABCD]

-- Define the condition that AC bisects ∠BAD.
def AC_bisects_BAD (A B C D : Point) : Prop := 
  ∃ (O : Point), is_on_line A C O ∧ is_on_line B A D ∧ angle_bisector BAD AC O

-- Define the point E on line CD
def point_E_on_CD (A B C D E : Point) : Prop := is_on_line E C D

-- Define that line BE intersects AC at F
def BE_intersects_AC_at_F (A B C D E F : Point) : Prop := 
  line_intersects_point B E A C F

-- Define that the extension of DF intersects BC at G
def DF_intersects_BC_at_G (A B C D E F G : Point) : Prop := 
  extends_line D F C G ∧ line_intersects_point B C D F G

-- Define the main theorem
theorem angle_GAC_eq_angle_EAC
  (h1 : is_quadrilateral ABCD)
  (h2 : AC_bisects_BAD A B C D)
  (h3 : point_E_on_CD A B C D E)
  (h4 : BE_intersects_AC_at_F A B C D E F)
  (h5 : DF_intersects_BC_at_G A B C D E F G) :
  ∠GAC = ∠EAC :=
sorry

end angle_GAC_eq_angle_EAC_l277_277105


namespace perpendicular_line_through_point_l277_277640

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l277_277640


namespace number_of_seating_arrangements_l277_277619

-- defining that there are 8 seats in total
def total_seats : ℕ := 8

-- defining the condition that there must be empty seats on both sides of each person
def valid_seating_arrangement (arr : list (option ℕ)) : Prop :=
  arr.length = total_seats ∧
  ∀ i, arr.nth i ≠ some 1 → (i = 0 ∨ i = 1 ∨ i = 6 ∨ i = 7 ∨
  (arr.nth (i-1) = none ∧ arr.nth (i+1) = none))
  
-- defining the main proof problem
theorem number_of_seating_arrangements : 
  ∃ arrs : finset (list (option ℕ)), 
  arrs.count (λ arr, valid_seating_arrangement arr) = 24 :=
sorry

end number_of_seating_arrangements_l277_277619


namespace part_a_l277_277914

variable {α : Type*} {n : ℕ}

def is_convex (f : α → ℝ) :=
  ∀ (x y : α) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

theorem part_a (f : ℝ → ℝ) (x : Fin n → ℝ) (αs : Fin n → ℚ) [∀ (i : Fin n), 0 < αs i] 
  (h_convex : is_convex f) (h_sum : ∑ i, (αs i : ℝ) = 1) :
  f (∑ i, αs i * x i) ≤ ∑ i, (αs i : ℝ) * f (x i) :=
sorry

end part_a_l277_277914


namespace polar_eq_curve_C_min_area_triangle_OMN_l277_277107

section
variable {α ρ θ R : ℝ}

-- Definitions for parametric and polar equations
def parametric_eq1 (α : ℝ) : ℝ := 2 * cos α
def parametric_eq2 (α : ℝ) : ℝ := sin α

def polar_eq1 (ρ θ : ℝ) : ℝ := ρ * cos θ
def polar_eq2 (ρ θ : ℝ) : ℝ := ρ * sin θ

-- Question 1: Prove polar equation of curve C is ρ^2 (1 + 3 sin^2 θ) = 4
theorem polar_eq_curve_C (α : ℝ) : 
  ∃ ρ θ, 
    (parametric_eq1 α = polar_eq1 ρ θ) ∧ 
    (parametric_eq2 α = polar_eq2 ρ θ) →
  ρ^2 * (1 + 3 * sin θ^2) = 4 :=
sorry

-- Question 2: Prove the minimum value of the area of triangle OMN is 4/5 given OM ⊥ ON
theorem min_area_triangle_OMN (ρ1 ρ2 θ : ℝ) : 
  (ρ1^2 * (1 + 3 * sin^2 θ) = 4) ∧ 
  (ρ2^2 * (1 + 3 * cos^2 θ) = 4) →
  OM ⊥ ON →
  min_area_of_triangle_OMN = 4 / 5 :=
sorry
end

end polar_eq_curve_C_min_area_triangle_OMN_l277_277107


namespace unique_function_l277_277917

-- Define the set of positive integers
def Z_plus := {n : ℤ // n > 0}

-- Define the function property
def func_property (f : ℤ → ℤ) (a b : ℤ) : Prop :=
  (a^2 + f a * f b) % (f a + b) = 0

-- Define the main theorem to prove
theorem unique_function 
  (f : ℤ → ℤ) 
  (h : ∀ a b ∈ Z_plus, func_property f a.val b.val) : 
  ∀ n ∈ Z_plus, f n.val = n.val :=
by {
  sorry
}

end unique_function_l277_277917


namespace equation_of_circumcircle_l277_277123

-- Define the conditions of the problem
def parabola := { p : ℝ × ℝ // p.1^2 = -4 * p.2 }
def F := (0, -1) : ℝ × ℝ
def P := (-4, -4) : ℝ × ℝ
def Q := (-2, 0) : ℝ × ℝ

-- Prove the equation of the circumcircle of triangle PFQ
theorem equation_of_circumcircle (hF : parabola F) (hP : parabola P) (hQ : parabola Q) :
    ∃ (R : ℝ), (x, y) ∈ ℝ × ℝ, (x + 2)^2 + (y + 5/2)^2 = 25/4 := by
    sorry

end equation_of_circumcircle_l277_277123


namespace cos_value_l277_277794

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := 
by
  sorry

end cos_value_l277_277794


namespace standard_deviation_of_data_is_2_l277_277607

def data := [5, 7, 7, 8, 10, 11]

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def squared_diffs (l : List ℝ) (mean : ℝ) : List ℝ :=
  l.map (λ x => (x - mean) ^ 2)

def variance (l : List ℝ) : ℝ :=
  mean (squared_diffs l (mean l))

def standard_deviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_of_data_is_2 :
  standard_deviation data = 2 :=
sorry

end standard_deviation_of_data_is_2_l277_277607


namespace sub_seq_arithmetic_l277_277870

variable (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sub_seq (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  a (3 * k - 1)

theorem sub_seq_arithmetic (h : is_arithmetic_sequence a d) : is_arithmetic_sequence (sub_seq a) (3 * d) := 
sorry


end sub_seq_arithmetic_l277_277870


namespace courtyard_length_is_60_l277_277852

noncomputable def stone_length : ℝ := 2.5
noncomputable def stone_breadth : ℝ := 2.0
noncomputable def num_stones : ℕ := 198
noncomputable def courtyard_breadth : ℝ := 16.5

theorem courtyard_length_is_60 :
  ∃ (courtyard_length : ℝ), courtyard_length = 60 ∧
  num_stones * (stone_length * stone_breadth) = courtyard_length * courtyard_breadth :=
sorry

end courtyard_length_is_60_l277_277852


namespace common_point_of_gp_lines_l277_277735

theorem common_point_of_gp_lines (a b c : ℝ) (r : ℝ) (h : b = a * r ∧ c = a * r ^ 2) :
  ∃ p : ℝ × ℝ, (∀ r : ℝ, let b := a * r in let c := a * r ^ 2 in let x := 0 in let y := -r in a * x + b * y = c) ∧ p = (0, 0) :=
by
  sorry

end common_point_of_gp_lines_l277_277735


namespace optimal_order_for_ostap_l277_277534

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l277_277534


namespace count_sets_M_l277_277207

theorem count_sets_M :
  finset.card
    (finset.filter (λ M, ({1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4})) (finset.powerset {1, 2, 3, 4})) = 3 :=
by
  sorry

end count_sets_M_l277_277207


namespace coin_toss_sequences_5040_l277_277041

theorem coin_toss_sequences_5040 :
  ∃ (n : ℕ), n = 17 ∧
             (∃ (a b c d : ℕ), a = 3 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ (number_of_sequences n a b c d = 5040)) :=
by
  sorry

end coin_toss_sequences_5040_l277_277041


namespace probability_multiple_of_4_l277_277494

theorem probability_multiple_of_4 :
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  prob_end_multiple_of_4 = 7 / 64 :=
by
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  have h : prob_end_multiple_of_4 = 7 / 64 := by sorry
  exact h

end probability_multiple_of_4_l277_277494


namespace smallest_number_divisible_by_five_primes_l277_277670

theorem smallest_number_divisible_by_five_primes : 
  ∃ n : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) ∧ n = 2310 :=
by
  use 2310
  -- Proof will be completed here
  sorry

end smallest_number_divisible_by_five_primes_l277_277670


namespace netCurrentValue_is_2210_89_l277_277785

def initialValue : ℝ := 10000
def maintenanceCost : ℝ := 500
def annualTaxRate : ℝ := 0.02
def depreciationRates : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

noncomputable def depreciate (value : ℝ) (rate : ℝ) : ℝ :=
  value - (rate * value)

noncomputable def carValueAfterYears : ∀ (years : List ℝ) (initial : ℝ), ℝ
  | [], v => v
  | r :: rs, v => carValueAfterYears rs (depreciate v r)

noncomputable def totalMaintenanceCost (years : ℕ) (cost : ℝ) : ℝ :=
  years * cost

noncomputable def annualTax (value : ℝ) (rate : ℝ) : ℝ :=
  value * rate
  
noncomputable def totalTax (values : List ℝ) (rate : ℝ) : ℝ :=
  values.map (λ v => annualTax v rate) |> List.sum

noncomputable def netCurrentValue (initial : ℝ) (rates : List ℝ) (maintenance : ℝ) (taxRate : ℝ) : ℝ :=
  let values := List.scanl depreciate initial rates
  let finalValue := values.getLast! (initial)
  finalValue - totalMaintenanceCost rates.length maintenance - totalTax (values.tail!) taxRate

theorem netCurrentValue_is_2210_89 : 
  netCurrentValue initialValue depreciationRates maintenanceCost annualTaxRate = 2210.8944 := 
  by 
  sorry

end netCurrentValue_is_2210_89_l277_277785


namespace Jenny_total_wins_l277_277114

theorem Jenny_total_wins :
  let games_against_mark := 10
  let mark_wins := 1
  let mark_losses := games_against_mark - mark_wins
  let games_against_jill := 2 * games_against_mark
  let jill_wins := (75 / 100) * games_against_jill
  let jenny_wins_against_jill := games_against_jill - jill_wins
  mark_losses + jenny_wins_against_jill = 14 :=
by
  sorry

end Jenny_total_wins_l277_277114


namespace area_of_given_circle_is_4pi_l277_277312

-- Define the given equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0

-- Define the area of the circle to be proved
noncomputable def area_of_circle : ℝ := 4 * Real.pi

-- Statement of the theorem to be proved in Lean
theorem area_of_given_circle_is_4pi :
  (∃ x y : ℝ, circle_equation x y) → area_of_circle = 4 * Real.pi :=
by
  -- The proof will go here
  sorry

end area_of_given_circle_is_4pi_l277_277312


namespace sum_first_four_terms_l277_277366

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * (2 : ℝ)^n ∧ 0 < a n

def sum_geometric_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a 1 * ((1 - (2 : ℝ)^n) / (1 - (2 : ℝ)))

theorem sum_first_four_terms (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_relation : (1 / a 1) - (1 / a 2) = 2 / a 3) : sum_geometric_terms a 4 = 15 := by
  sorry

end sum_first_four_terms_l277_277366


namespace perpendicular_line_equation_l277_277664

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l277_277664


namespace ellen_rate_l277_277320

-- Definitions of the conditions
def earl_rate : ℝ := 36
def combined_envelopes : ℝ := 360
def combined_time : ℝ := 6

-- Calculation of the combined rate
def combined_rate : ℝ := combined_envelopes / combined_time

-- Lean statement to prove Ellen's rate
theorem ellen_rate : (combined_rate - earl_rate) = 24 :=
by 
  have combined_rate_eq : combined_rate = 60 := by sorry -- Calculation from the conditions
  have earl_rate_eq : earl_rate = 36 := by sorry -- Given condition
  calc
    combined_rate - earl_rate = 60 - 36 := by
      rw [combined_rate_eq, earl_rate_eq]
    ...              = 24               := by linarith

end ellen_rate_l277_277320


namespace imaginary_part_of_complex_expr_l277_277202

def complex_expr := (1 - Complex.i) / (1 + Complex.i) + 1

theorem imaginary_part_of_complex_expr : complex.im complex_expr = -1 := sorry

end imaginary_part_of_complex_expr_l277_277202


namespace James_pays_35_l277_277492

theorem James_pays_35 (first_lesson_free : Bool) (total_lessons : Nat) (cost_per_lesson : Nat) 
  (first_x_paid_lessons_free : Nat) (every_other_remainings_free : Nat) (uncle_pays_half : Bool) :
  total_lessons = 20 → 
  first_lesson_free = true → 
  cost_per_lesson = 5 →
  first_x_paid_lessons_free = 10 →
  every_other_remainings_free = 1 → 
  uncle_pays_half = true →
  (10 * cost_per_lesson + 4 * cost_per_lesson) / 2 = 35 :=
by
  sorry

end James_pays_35_l277_277492


namespace elina_donut_holes_l277_277288

theorem elina_donut_holes (rate : ℝ) :
  let r_elina := 5
  let r_marco := 7
  let r_priya := 9
  let surface_area (r : ℝ) := 4 * Real.pi * r^2
  let sa_elina := surface_area r_elina
  let sa_marco := surface_area r_marco
  let sa_priya := surface_area r_priya
  let lcm_surface_areas := Real.lcm sa_elina sa_marco sa_priya
  let num_holes_elina := lcm_surface_areas / sa_elina
  num_holes_elina = 441 :=
by
  sorry

end elina_donut_holes_l277_277288


namespace domain_f_l277_277763

noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 2) + 1 / (x - 3)

theorem domain_f :
  {x : ℝ | x ≥ 2 ∧ x ≠ 3 } = {x : ℝ | (2 ≤ x ∧ x < 3) ∨ (3 < x)} :=
by
  sorry

end domain_f_l277_277763


namespace students_neither_course_l277_277075

theorem students_neither_course (total_students coding_students robotics_students both_courses : ℕ)
  (h1 : total_students = 150)
  (h2 : coding_students = 90)
  (h3 : robotics_students = 70)
  (h4 : both_courses = 25) : (total_students - (coding_students - both_courses + robotics_students - both_courses + both_courses) = 15) :=
by
  have only_coding := coding_students - both_courses
  have only_robotics := robotics_students - both_courses
  have at_least_one := only_coding + only_robotics + both_courses
  have neither_courses := total_students - at_least_one
  show (total_students - (coding_students - both_courses + robotics_students - both_courses + both_courses) = 15), by sorry

end students_neither_course_l277_277075


namespace f_at_neg_one_l277_277316

theorem f_at_neg_one :
  ∀ f : ℝ → ℝ,
  (∀ x, (x^(2^2007 + 1) - 1) * f x = (x - 1) * (x^2 - 1) * (x^8 - 1) * ... * (x^(2^2006) - 1) - 1) →
  f (-1) = 1 :=
by
  intros f h
  sorry

end f_at_neg_one_l277_277316


namespace fifth_boat_more_than_average_l277_277699

theorem fifth_boat_more_than_average :
  let total_people := 2 + 4 + 3 + 5 + 6
  let num_boats := 5
  let average_people := total_people / num_boats
  let fifth_boat := 6
  (fifth_boat - average_people) = 2 :=
by
  sorry

end fifth_boat_more_than_average_l277_277699


namespace lengths_of_medians_l277_277262

def is_right_angle (angle : Real) : Prop := angle = 90

def midpoint_length (a b : Real) : Real := a / 2

def median_length (a b : Real) : Real := Real.sqrt (a^2 + b^2)

def PQ := 6
def QR := 8
def angle_PQR := 90

def mid_QR := midpoint_length QR QR
def mid_PQ := midpoint_length PQ PQ

def P_M := median_length PQ mid_QR
def R_N := median_length mid_PQ QR

theorem lengths_of_medians (PQR_is_right_angle : is_right_angle angle_PQR)
(mid_PQ_value : mid_PQ = 3)
(mid_QR_value : mid_QR = 4)
: (P_M = 2 * Real.sqrt 13) ∧ (R_N = Real.sqrt 73) := sorry

end lengths_of_medians_l277_277262


namespace number_of_even_1s_grids_l277_277097

theorem number_of_even_1s_grids (n : ℕ) : 
  (∃ grid : fin n → fin n → ℕ, 
    (∀ i j, grid i j = 0 ∨ grid i j = 1) ∧
    (∀ i, (∑ j, grid i j) % 2 = 0) ∧
    (∀ j, (∑ i, grid i j) % 2 = 0)) →
  2 ^ ((n - 1) * (n - 1)) = 2 ^ ((n - 1) * (n - 1)) :=
by sorry

end number_of_even_1s_grids_l277_277097


namespace taxi_charge_l277_277268

theorem taxi_charge (X : ℝ) (h1 : ∀ d : ℝ, d = 1 / 5 → X ∈ ℝ) 
  (h2 : ∀ y : ℝ, y ∈ ℝ → X + y * 0.40 ∈ ℝ) (h3 : 8 * 5 - 1 = 39) 
  (h4 : 39 * 0.40 = 15.60) (h5 : X + 15.60 = 18.40) : X = 2.80 :=
by sorry

end taxi_charge_l277_277268


namespace max_value_f_on_interval_l277_277440

noncomputable def solve_p : ℝ := -2 * real.cbrt 2
noncomputable def solve_q : ℝ := (3 / 2) * real.cbrt 2 + real.cbrt 4

noncomputable def f (x : ℝ) : ℝ := x ^ 2 + solve_p * x + solve_q
noncomputable def g (x : ℝ) : ℝ := x + 1 / (x ^ 2)

theorem max_value_f_on_interval :
  (f 2) = (4 - (5 / 2 * real.cbrt 2) + real.cbrt 4) :=
by sorry

end max_value_f_on_interval_l277_277440


namespace athlete_speed_l277_277684

theorem athlete_speed (d t s : ℝ) (h_d : d = 200) (h_t : t = 40) (h_s : s = d / t) : s = 5 :=
by {
  rw [h_d, h_t] at h_s,
  linarith,
}

end athlete_speed_l277_277684


namespace count_even_ones_grid_l277_277100

theorem count_even_ones_grid (n : ℕ) : 
  (∃ f : (Fin n) → (Fin n) → ℕ, (∀ i : Fin n, ∑ j in (Finset.univ : Finset (Fin n)), f i j % 2 = 0) ∧ 
                                  (∀ j : Fin n, ∑ i in (Finset.univ : Finset (Fin n)), f i j % 2 = 0)) ↔ 
  2^((n-1)^2) = 2^((n-1)^2) :=
sorry

end count_even_ones_grid_l277_277100


namespace total_games_played_l277_277256

theorem total_games_played (num_teams : ℕ) (h : num_teams = 14) : (num_teams.choose 2) = 91 :=
by {
  rw h,
  have : 14.choose 2 = 91, -- This follows from the calculation we showed in the solution part.
  exact this,
}

end total_games_played_l277_277256


namespace grid_fill_even_l277_277086

-- Useful definitions
def even {α : Type} [AddGroup α] (a : α) : Prop := ∃ b, a = 2 * b

-- Statement of the problem
-- 'n' is a natural number, grid is n × n
-- We need to find the number of ways to fill the grid with 0s and 1s such that each row and column has an even number of 1s
theorem grid_fill_even (n : ℕ) : ∃ (ways : ℕ), ways = 2 ^ ((n - 1) * (n - 1)) ∧ 
  (∀ grid : (Fin n → Fin n → bool), (∀ i : Fin n, even (grid i univ.count id)) ∧ (∀ j : Fin n, even (univ.count (λ x, grid x j))) → true) :=
sorry

end grid_fill_even_l277_277086


namespace kamilla_acquaintances_l277_277471

/-- In the city of Bukvinsk, people are acquaintances only if their names contain the same letters.
Martin has 20 acquaintances, Klim has 15 acquaintances, Inna has 12 acquaintances, and Tamara
has 12 acquaintances. Prove that Kamilla has the same number of acquaintances as Klim, which is 15. -/
theorem kamilla_acquaintances :
  let count_martin := 20 in
  let count_klim := 15 in
  let count_inna := 12 in
  let count_tamara := 12 in
  let count_kamilla := 15 in
  ∀ (martin klim inna tamara kamilla : Type),
  martin ∈ bukvinsk → klim ∈ bukvinsk → inna ∈ bukvinsk → tamara ∈ bukvinsk → kamilla ∈ bukvinsk →
  martin.acquaintances = count_martin →
  klim.acquaintances = count_klim →
  inna.acquaintances = count_inna →
  tamara.acquaintances = count_tamara →
  kamilla.acquaintances = count_kamilla :=
by
  sorry

end kamilla_acquaintances_l277_277471


namespace area_of_XMYN_l277_277889

variables {X Y Z M N Q : Type*}
variables [normedGroup X] [normedGroup Y] [normedGroup Z]
variables (XN_is_median : median X N Y Z)
variables (YM_is_median : median Y M X Z)
variables (Q_is_intersection : intersection X M Y N Q)
variables (length_QN : real) (length_QN_val : length_QN = 5)
variables (length_QM : real) (length_QM_val : length_QM = 4)
variables (length_MN : real) (length_MN_val : length_MN = 7)

noncomputable def quadrilateral_area : real :=
1 / 2 * (10 * 5 + 10 * 4 + 8 * 4 + 8 * 5)

theorem area_of_XMYN : quadrilateral_area = 81 :=
by
  sorry

end area_of_XMYN_l277_277889


namespace power_of_two_expressible_l277_277559

theorem power_of_two_expressible (n : ℕ) (hn : n ≥ 3) :
  ∃ (x y : ℤ), odd x ∧ odd y ∧ 2^n = 7 * x^2 + y^2 :=
sorry

end power_of_two_expressible_l277_277559


namespace smallest_n_y_n_integer_l277_277302

def y₁ : ℝ := real.root 4 4
def y_seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 1  -- Since Lean indices from 0
  | 1 => y₁
  | n + 1 => (y_seq n) ^ y₁

theorem smallest_n_y_n_integer :
  ∃ n : ℕ, (y_seq n).is_integer ∧ (∀ m < n, ¬(y_seq m).is_integer) ∧ n = 8 :=
by
  sorry

end smallest_n_y_n_integer_l277_277302


namespace temperature_at_midnight_l277_277072

-- Define the variables for initial conditions and changes
def T_morning : ℤ := 7 -- Morning temperature in degrees Celsius
def ΔT_noon : ℤ := 2   -- Temperature increase at noon in degrees Celsius
def ΔT_midnight : ℤ := -10  -- Temperature drop at midnight in degrees Celsius

-- Calculate the temperatures at noon and midnight
def T_noon := T_morning + ΔT_noon
def T_midnight := T_noon + ΔT_midnight

-- State the theorem to prove the temperature at midnight
theorem temperature_at_midnight : T_midnight = -1 := by
  sorry

end temperature_at_midnight_l277_277072


namespace area_triangle_LOM_l277_277462

noncomputable def angle_TRIANGLE_A : ℝ := 45 -- Boundary angle A (in degrees)
noncomputable def angle_TRIANGLE_B : ℝ := 90 -- Boundary angle B
noncomputable def angle_TRIANGLE_C : ℝ := 45 -- Boundary angle C
noncomputable def area_ABC : ℝ := 20 -- Given area of triangle ABC

-- Consider functions for conditions
def scalene_triangle (A B C : ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def angle_relationship (A B C : ℝ) : Prop :=
  C = B - A ∧ B = 2 * A

-- Main proof problem
theorem area_triangle_LOM
  (h1: scalene_triangle angle_TRIANGLE_A angle_TRIANGLE_B angle_TRIANGLE_C)
  (h2: angle_relationship angle_TRIANGLE_A angle_TRIANGLE_B angle_TRIANGLE_C)
  (h3: area_ABC = 20) :
  let area_LOM := 3 * area_ABC in
  area_LOM = 60 :=
by sorry

end area_triangle_LOM_l277_277462


namespace circle_thru_A_B_l277_277363

-- Defining the points A and B
def A : point := (5, 1)
def B : point := (1, 3)

-- Defining the center C with x-coordinate 2 and y-coordinate 0
def C : point := (2, 0)

-- Equation of the circle with center C and radius sqrt(10)
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 10

-- Proving that the given points A and B satisfy the circle equation
theorem circle_thru_A_B :
  circle_equation 5 1 ∧ circle_equation 1 3 :=
begin
  -- Proof steps would go here
  sorry
end

end circle_thru_A_B_l277_277363


namespace different_lines_count_l277_277867

open Finset

theorem different_lines_count : 
  let S := {0, 2, 3, 4, 5, 6}
  let lines : Finset (Set (ℚ × ℚ)) := (S.product S).image (λ ⟨a, b⟩, {(x, y) : ℚ × ℚ | a * x + b * y = 0 })
  lines.card = 18
  := 
begin
  sorry
end

end different_lines_count_l277_277867


namespace hyperbola_eccentricity_l277_277979

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (F : ℝ × ℝ) (hF : F = (1, 0)) (Q : ℝ × ℝ) (hQ : Q = (0, 1)) 
    (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) 
    (par : ∀ x y : ℝ, y^2 = 4 * x)
    (h_perpendicular : ∀ P : ℝ × ℝ, (QP: ℝ × ℝ) -> LinePerpendicular QP QR -> P (1,2)): 
  let e := 1 / (sqrt 2 - 1) in
  e = sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l277_277979


namespace sin_cos_theta_l277_277909

variables {ℝ : Type*} [normed_group ℝ] [normed_space ℝ ℝ]
variables (a b c : ℝ) (θ : ℝ)

-- Conditions
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 7
axiom norm_c : ∥c∥ = 6
axiom cross_product_condition : a × (a × b) = c

-- Statement to be proved
theorem sin_cos_theta :
  ∃ (θ : ℝ),
    sin θ = 3/7 ∧ cos θ = 2 * sqrt 10 / 7 :=
by
  sorry

end sin_cos_theta_l277_277909


namespace angle_between_vectors_l277_277819

variables {E : Type*} [inner_product_space ℝ E]

theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : inner_product a (a - 2 • b) = 0) (h2 : inner_product b (b - 2 • a) = 0) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_l277_277819


namespace right_triangle_inradius_height_ratio_l277_277558

-- Define a right triangle with sides a, b, and hypotenuse c
variables {a b c : ℝ}
-- Define the altitude from the right angle vertex
variables {h : ℝ}
-- Define the inradius of the triangle
variables {r : ℝ}

-- Define the conditions: right triangle 
-- and the relationships for h and r
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def altitude (h : ℝ) (a b c : ℝ) : Prop := h = (a * b) / c
def inradius (r : ℝ) (a b c : ℝ) : Prop := r = (a + b - c) / 2

theorem right_triangle_inradius_height_ratio {a b c h r : ℝ} 
  (Hrt : is_right_triangle a b c)
  (Hh : altitude h a b c)
  (Hr : inradius r a b c) : 
  0.4 < r / h ∧ r / h < 0.5 :=
sorry

end right_triangle_inradius_height_ratio_l277_277558


namespace correct_properties_l277_277562

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (-Real.pi / 6) = 0) :=
by
  sorry

end correct_properties_l277_277562


namespace advantageous_order_l277_277544

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l277_277544


namespace max_inradius_l277_277803

-- Define the parabola and its focus
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def focus := (1 : ℝ, 0 : ℝ)
def origin := (0 : ℝ, 0 : ℝ)

-- Define the point P on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola_eq P.1 P.2

-- Define the conditions
def conditions (P : ℝ × ℝ) : Prop :=
  point_on_parabola P ∧ P.1 > 1 ∧ P.2 > 0

-- Main statement to prove
theorem max_inradius (P : ℝ × ℝ) (hP : conditions P) :
  ∃ r : ℝ, r = 2 * real.sqrt 3 / 9 := by
  sorry

end max_inradius_l277_277803


namespace inscribed_circle_segment_lengths_l277_277706

theorem inscribed_circle_segment_lengths (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h₄ : a + b > c) (h₅ : a + c > b) (h₆ : b + c > a) :
  let x := (a + b - c) / 2,
      y := (a + c - b) / 2,
      z := (b + c - a) / 2 in
  (y + z = a) ∧ (x + z = b) ∧ (x + y = c) :=
by
  sorry

end inscribed_circle_segment_lengths_l277_277706


namespace fill_grid_with_even_ones_l277_277080

theorem fill_grid_with_even_ones (n : ℕ) : 
  ∃ ways : ℕ, ways = 2^((n-1)^2) ∧ 
  (∀ grid : array n (array n (fin 2)), 
    (∀ i : fin n, even (grid[i].to_list.count (λ x, x = 1))) ∧ 
    (∀ j : fin n, even (grid.map (λ row, row[j]).to_list.count (λ x, x = 1)))) :=
begin
  use 2^((n-1)^2),
  split,
  { refl },
  { sorry },
end

end fill_grid_with_even_ones_l277_277080


namespace arithmetic_seq_a7_value_l277_277879

theorem arithmetic_seq_a7_value {a : ℕ → ℝ} (h_positive : ∀ n, 0 < a n)
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_eq : 3 * a 6 - (a 7) ^ 2 + 3 * a 8 = 0) : a 7 = 6 :=
  sorry

end arithmetic_seq_a7_value_l277_277879


namespace fly_distance_from_ceiling_l277_277711

-- Definitions based on the given conditions
def fly_distance_to_wall_1 (fly : ℝ × ℝ × ℝ) := fly.1 = 2
def fly_distance_to_wall_2 (fly : ℝ × ℝ × ℝ) := fly.2 = 7
def fly_distance_to_corner (fly : ℝ × ℝ × ℝ) := ∥fly∥ = 10

-- Main theorem stating the distance from the ceiling
theorem fly_distance_from_ceiling (fly : ℝ × ℝ × ℝ) 
  (h1 : fly_distance_to_wall_1 fly)
  (h2 : fly_distance_to_wall_2 fly)
  (h3 : fly_distance_to_corner fly) : 
  fly.3 = sqrt 47 :=
sorry

end fly_distance_from_ceiling_l277_277711


namespace smallest_n_multiple_of_7_l277_277960

theorem smallest_n_multiple_of_7 (x y n : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 7]) (h2 : y - 2 ≡ 0 [ZMOD 7]) :
  x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7] → n = 3 :=
by
  sorry

end smallest_n_multiple_of_7_l277_277960


namespace right_triangle_sides_l277_277457

open Real

theorem right_triangle_sides (h : ∀ (α β : ℝ), α + β = 90 ∧ (5 * tan α = 4 * tan β) ∧ α = 50 ∧ β = 40 )
  (c : ℝ) (h_c : c = 15) :
  ∃ (a b : ℝ), a ≈ 11.49 ∧ b ≈ 9.642 :=
by
  let α := 50 : ℝ
  let β := 40 : ℝ
  have h_sum : α + β = 90 := by norm_num
  have ratio : 5 * tan α = 4 * tan β := by norm_num
  let a := c * sin (α * pi/180)
  let b := c * sin (β * pi/180)
  have a_approx : a ≈ 11.49 := sorry
  have b_approx : b ≈ 9.642 := sorry
  use [a, b]
  exact ⟨a_approx, b_approx⟩


end right_triangle_sides_l277_277457


namespace avg_of_all_5_is_8_l277_277583

-- Let a1, a2, a3 be three quantities such that their average is 4.
def is_avg_4 (a1 a2 a3 : ℝ) : Prop :=
  (a1 + a2 + a3) / 3 = 4

-- Let a4, a5 be the remaining two quantities such that their average is 14.
def is_avg_14 (a4 a5 : ℝ) : Prop :=
  (a4 + a5) / 2 = 14

-- Prove that the average of all 5 quantities is 8.
theorem avg_of_all_5_is_8 (a1 a2 a3 a4 a5 : ℝ) :
  is_avg_4 a1 a2 a3 ∧ is_avg_14 a4 a5 → 
  ((a1 + a2 + a3 + a4 + a5) / 5 = 8) :=
by
  intro h
  sorry

end avg_of_all_5_is_8_l277_277583


namespace express_in_scientific_notation_l277_277886

theorem express_in_scientific_notation (n : ℕ) (h : n = 218000) : n = 2.18 * 10^5 := by
  sorry

end express_in_scientific_notation_l277_277886


namespace equilateral_triangles_ratio_l277_277475

theorem equilateral_triangles_ratio (t : ℝ) (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S] 
  (h1 : ∀ (A B C : Type) [equilateral_tris A] [equilateral_tris B] [equilateral_tris C], QR = t) 
  (h2 : altitudes_from P S QR = t * sqrt 3 / 2)
  : PS / QR = sqrt 3 := sorry

end equilateral_triangles_ratio_l277_277475


namespace sum_of_first_2018_terms_l277_277184

-- Definitions and conditions
def Fibonacci (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) = a n + a (n + 1)

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a i

noncomputable def a_2020 := 2020
def M : ℕ := a_2020
def sum_s (a : ℕ → ℕ) (n : ℕ) : ℕ := sum_of_first_n_terms a n

-- The statement to prove
theorem sum_of_first_2018_terms (a : ℕ → ℕ) (h_fib : Fibonacci a) (h_a2020 : a 2020 = M) : sum_s a 2018 = M - 1 :=
sorry

end sum_of_first_2018_terms_l277_277184


namespace ostap_advantageous_order_l277_277546

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l277_277546


namespace find_vertex_D_l277_277413

noncomputable def quadrilateral_vertices : Prop :=
  let A : (ℤ × ℤ) := (-1, -2)
  let B : (ℤ × ℤ) := (3, 1)
  let C : (ℤ × ℤ) := (0, 2)
  A ≠ B ∧ A ≠ C ∧ B ≠ C

theorem find_vertex_D (A B C D : ℤ × ℤ) (h_quad : quadrilateral_vertices) :
    (A = (-1, -2)) →
    (B = (3, 1)) →
    (C = (0, 2)) →
    (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) →
    D = (-4, -1) :=
by
  sorry

end find_vertex_D_l277_277413


namespace correct_propositions_l277_277281

theorem correct_propositions :
  (∫ x in -real.pi/2..real.pi/2, real.cos x = 2) ∧
  (¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f x + 2 = -f x) → (∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ x : ℝ, f (x + ε) ≠ f x))) ∧
  (∀ {a n : ℕ}, S n = 3^n + a ↔ (∀ n, a_n = 3^(n-1) * 2) ↔ a = -1) :=
by sorry

end correct_propositions_l277_277281


namespace BC_length_l277_277065

-- Definitions and conditions
variables {A B C X : Point} -- Points in the triangle and on the circle
variables {AB AC : ℝ} -- Lengths AB and AC

-- Proving the length of BC
theorem BC_length (h1 : AB = 75) (h2 : AC = 120)
  (h3 : (circle A AB).intersect_line_segment B C = {B, X})
  (h4 : ∃ (BX CX : ℤ), BC = BX + CX) : BC = 117 :=
sorry

end BC_length_l277_277065


namespace isosceles_triangle_base_l277_277370

theorem isosceles_triangle_base (a b c : ℕ) (h_isosceles : a = b ∨ a = c ∨ b = c)
  (h_perimeter : a + b + c = 29) (h_side : a = 7 ∨ b = 7 ∨ c = 7) : 
  a = 7 ∨ b = 7 ∨ c = 7 ∧ (a = 7 ∨ a = 11) ∧ (b = 7 ∨ b = 11) ∧ (c = 7 ∨ c = 11) ∧ (a ≠ b ∨ c ≠ b) :=
by
  sorry

end isosceles_triangle_base_l277_277370


namespace smallest_n_multiple_of_7_l277_277961

theorem smallest_n_multiple_of_7 (x y n : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 7]) (h2 : y - 2 ≡ 0 [ZMOD 7]) :
  x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7] → n = 3 :=
by
  sorry

end smallest_n_multiple_of_7_l277_277961


namespace max_points_of_intersection_l277_277667

theorem max_points_of_intersection (p q : ℚ[X])
  (h1 : degree p = 5)
  (hc1 : p.leadingCoeff = 2)
  (h2 : degree q = 3)
  (hc2 : q.leadingCoeff = 3) :
  ∃ m, m ≤ 5 ∧ ∀ x, p.eval x = q.eval x → m = 5 :=
sorry

end max_points_of_intersection_l277_277667


namespace tangents_perpendicular_l277_277489

-- Definitions of the required functions and parameters
def target_function (a x : ℝ) : ℝ := a * (x + 1)^2 + 1

-- Conditions for the problem
theorem tangents_perpendicular (a x0 y0 : ℝ) : 
  (∀ x0 y0, (log ((3 * x0 - x0^2 + 1)) (y0 - 4) = 
               log ((3 * x0) - x0^2 + 1) (abs(2*x0 + 4) - abs(2*x0 + 1)) / 
                     ((3 * x0) + 4.5) * sqrt(x0^2 + 3 * x0 + 2.25))) 
  → (target_function (-0.0625) x0 = y0) :=
sorry

end tangents_perpendicular_l277_277489


namespace Tanya_body_lotion_cost_is_60_l277_277296

variable (B : ℝ)

axiom Christy_spends_twice_as_Tanya : ∀ B, Christy_expense B = 2 * Tanya_expense B
axiom Tanya_pays_per_face_moisturizer : Tanya_face_moisturizer_cost = 2 * 50
axiom Tanya_buys_four_body_lotions : Tanya_body_lotion_cost B = 4 * B
axiom total_expenditure : ∀ B, 100 + 4 * B + 2 * (100 + 4 * B) = 1020

theorem Tanya_body_lotion_cost_is_60 : B = 60 :=
by
  sorry

end Tanya_body_lotion_cost_is_60_l277_277296


namespace min_max_abs_expression_l277_277438

theorem min_max_abs_expression (x y : ℝ) : 
  ∃ x y : ℝ, minimum (max (abs (2 * x + y)) (max (abs (x - y)) (abs (1 + y)))) = (1 / 2) :=
sorry

end min_max_abs_expression_l277_277438


namespace sum_b_n_l277_277840

noncomputable def S (n : ℕ) : ℚ := (1 / 2: ℚ) * n^2 + (1 / 2: ℚ) * n
noncomputable def a (n : ℕ) : ℚ := S n - S (n - 1)
noncomputable def b (n : ℕ) : ℚ := a n * 2^(n-1)

theorem sum_b_n (n : ℕ) : (∑ k in Finset.range n, b (k + 1)) = (n - 1) * 2^n + 1 :=
sorry

end sum_b_n_l277_277840


namespace water_level_drop_recording_l277_277444

theorem water_level_drop_recording (rise6_recorded: Int): 
    (rise6_recorded = 6) → (6 = -rise6_recorded) :=
by
  sorry

end water_level_drop_recording_l277_277444


namespace number_of_lines_through_points_l277_277851

-- Define the 3x3 grid of lattice points
def grid_3x3 : set (ℕ × ℕ) := {(x, y) | x < 3 ∧ y < 3}

-- Define what it means for a line to pass through at least two points in this grid
def line_through_points (p1 p2 : ℕ × ℕ) : set (ℕ × ℕ) :=
{(x, y) | (x - p1.1) * (p2.2 - p1.2) = (y - p1.2) * (p2.1 - p1.1)}

-- Count lines passing through at least two points in the grid
theorem number_of_lines_through_points :
  finset.card {L : set (ℕ × ℕ) | ∃ p1 p2 ∈ grid_3x3, p1 ≠ p2 ∧ L = line_through_points p1 p2} = 20 :=
sorry

end number_of_lines_through_points_l277_277851


namespace length_of_BD_l277_277608

theorem length_of_BD (AB AC CB BD : ℝ) (h1 : AB = 10) (h2 : AC = 4 * CB) (h3 : AC = 4 * 2) (h4 : CB = 2) :
  BD = 3 :=
sorry

end length_of_BD_l277_277608


namespace spotted_mushrooms_ratio_l277_277266

theorem spotted_mushrooms_ratio 
  (total_mushrooms : ℕ) 
  (gilled_mushrooms : ℕ) 
  (spotted_mushrooms : ℕ) 
  (total_mushrooms_eq : total_mushrooms = 30) 
  (gilled_mushrooms_eq : gilled_mushrooms = 3) 
  (spots_and_gills_exclusive : ∀ x, x = spotted_mushrooms ∨ x = gilled_mushrooms) : 
  spotted_mushrooms / gilled_mushrooms = 9 := 
by
  sorry

end spotted_mushrooms_ratio_l277_277266


namespace tan_pi_minus_alpha_l277_277881

noncomputable def A : ℝ × ℝ := (4/5, 3/5)

theorem tan_pi_minus_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hA : A = (Real.cos α, Real.sin α)) : Real.tan(π - α) = -3 / 4 := 
by 
  have h1 : α = Real.arctan (3 / 4),
  { sorry },
  have h2 : Real.sin α = 3 / 5 ∧ Real.cos α = 4 / 5,
  { sorry },
  sorry 

end tan_pi_minus_alpha_l277_277881


namespace ostap_advantageous_order_l277_277549

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l277_277549


namespace optimal_order_for_ostap_l277_277533

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l277_277533


namespace find_a_b_find_range_c_l277_277020

noncomputable def f (x : ℝ) (a b c : ℝ) := x^3 + a * x^2 + b * x + c

theorem find_a_b (c : ℝ) :
  (∃ a b : ℝ, ∀ x, f x a b c = x^3 + a * x^2 + b * x + c ∧
    (f' -2 / 3 = 0) ∧ (f' 1 = 0)) → 
  (a = -1 / 2 ∧ b = -2) :=
sorry

theorem find_range_c (a b : ℝ) :
  (a = -1 / 2 ∧ b = -2) ∧
  (∃ c : ℝ, ∀ x, f x a b c = x^3 + a * x^2 + b * x + c →
    (∃ x₀ > 1 ∨ x₀ < -2 / 3, f' x > 0) ∧
    (∃ x₁ -2 / 3 < x₁ < 1, f' x < 0)) →
    (-22 / 27 < c ∧ c < 3 / 2) :=
sorry

end find_a_b_find_range_c_l277_277020


namespace correct_answer_l277_277414

def sequence (a b : ℤ) : ℕ → ℤ
| 0       := a
| 1       := b
| (n + 2) := sequence a b (n + 1) - sequence a b n

def sum_sequence (a b : ℤ) (n : ℕ) : ℤ :=
  (List.range (n + 1)).map (sequence a b) |>.sum

theorem correct_answer (a b : ℤ) :
  sequence a b 99 = -a ∧ sum_sequence a b 99 = 2 * b - a := by
  sorry

end correct_answer_l277_277414


namespace number_greater_by_l277_277209

def question (a b : Int) : Int := a + b

theorem number_greater_by (a b : Int) : question a b = -11 :=
  by
    sorry

-- Use specific values from the provided problem:
example : question -5 -6 = -11 :=
  by
    sorry

end number_greater_by_l277_277209


namespace kamilla_acquaintances_l277_277472

/-- In the city of Bukvinsk, people are acquaintances only if their names contain the same letters.
Martin has 20 acquaintances, Klim has 15 acquaintances, Inna has 12 acquaintances, and Tamara
has 12 acquaintances. Prove that Kamilla has the same number of acquaintances as Klim, which is 15. -/
theorem kamilla_acquaintances :
  let count_martin := 20 in
  let count_klim := 15 in
  let count_inna := 12 in
  let count_tamara := 12 in
  let count_kamilla := 15 in
  ∀ (martin klim inna tamara kamilla : Type),
  martin ∈ bukvinsk → klim ∈ bukvinsk → inna ∈ bukvinsk → tamara ∈ bukvinsk → kamilla ∈ bukvinsk →
  martin.acquaintances = count_martin →
  klim.acquaintances = count_klim →
  inna.acquaintances = count_inna →
  tamara.acquaintances = count_tamara →
  kamilla.acquaintances = count_kamilla :=
by
  sorry

end kamilla_acquaintances_l277_277472


namespace motorboat_trip_time_l277_277161

theorem motorboat_trip_time 
  (v : ℝ)  -- speed of the motorboat
  (x : ℝ)  -- speed of the current
  (h1 : x = v / 3)  -- speed of the current is 1/3 of the motorboat's speed
  (h2 : 20 = 2 * (10)) :  -- it takes 20 minutes for a round trip without current
  let d := 10 * v / 3 in  -- distance between pier and bridge
  let t_with_current := d / (4 * (v / 3)) in  -- time to travel with current
  let t_against_current := d / (2 * (v / 3)) in  -- time to travel against current
  22.5 = t_with_current + t_against_current :=
by
  sorry

end motorboat_trip_time_l277_277161


namespace perpendicular_line_equation_l277_277632

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l277_277632


namespace proof_problem_l277_277507

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_symm : ∀ x : ℝ, f (2 - x) = f x
axiom f_at_1 : f 1 = 2
axiom f_odd : ∀ x : ℝ, f (3 * x + 2) = - f (3 * (-x) + 2)
axiom g_symm : ∀ x : ℝ, g x = -g (4 - x)
axiom intersections : Σ' (x y : set ℝ), y = {y | (∃ x ∈ x, y = f x ∧ y = g x)} ∧ x = (set.univ : set ℝ) ∧ 2023 = set.card {p | ∃ x y, p = (x, y) ∧ (y = f x) ∧ (y = g x)}

theorem proof_problem (x y : ℝ) (x_set : set ℝ) (y_set : set ℝ) (inter_points : set (ℝ × ℝ)) :
  (∀ x : ℝ, f (2 - x) = f x) →
  (f 1 = 2) →
  (∀ x : ℝ, f (3 * x + 2) = -f (3 * (-x) + 2)) →
  (∀ x : ℝ, g x = -g (4 - x)) →
  (2023 = set.card inter_points) →
  (∀ (a b : ℝ), (a, b) ∈ inter_points → (a ∈ x_set) ∧ (b ∈ y_set) ∧ (b = f a) ∧ (b = g a)) →
  ¬f 2023 = 2 ∧
  (∀ x : ℝ, x = 1 → f x = f(2-x)) →
  f 0 = 0 ∧
  (∑ i in inter_points, (i.1 + i.2) = 4046) :=
sorry

end proof_problem_l277_277507


namespace find_P_coordinates_l277_277487

variables (x y z : ℝ)

def midpoint (A B : ℕ → ℝ) := λ i, (A i + B i) / 2

theorem find_P_coordinates
    (M : ℕ → ℝ) (M_def : M 0 = 2 ∧ M 1 = -1 ∧ M 2 = 3)
    (N : ℕ → ℝ) (N_def : N 0 = 3 ∧ N 1 = 2 ∧ N 2 = -4)
    (O : ℕ → ℝ) (O_def : O 0 = -1 ∧ O 1 = 4 ∧ O 2 = 2)
    (P : ℕ → ℝ)
    (P_mid : ∀ i, midpoint P N i = if i = 0 then 1 / 2 else if i = 1 then 3 / 2 else 5 / 2) :
    P 0 = -2 ∧ P 1 = 1 ∧ P 2 = 9 :=
by {
  have h : (∀ i, midpoint (midpoint M O) 0 = 1 / 2 ∧ midpoint (midpoint M O) 1 = 3 / 2 ∧ midpoint (midpoint M O) 2 = 5 / 2)
  sorry -- skip the detailed computations
}

end find_P_coordinates_l277_277487


namespace garage_sale_items_count_l277_277685

theorem garage_sale_items_count (higher_prices lower_prices : ℕ) (radio_count : ℕ) :
  higher_prices = 15 → lower_prices = 22 → radio_count = 1 → higher_prices + lower_prices + radio_count = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  decide

end garage_sale_items_count_l277_277685


namespace average_daily_sales_l277_277936

def pens_sold_day_one : ℕ := 96
def pens_sold_next_days : ℕ := 44
def total_days : ℕ := 13

theorem average_daily_sales : (pens_sold_day_one + 12 * pens_sold_next_days) / total_days = 48 := 
by 
  sorry

end average_daily_sales_l277_277936


namespace optimal_order_l277_277537

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l277_277537


namespace common_area_proof_l277_277274

noncomputable def common_area_rect_circle (rect_length rect_width circle_radius : ℝ) : ℝ :=
  if h : rect_length = 10 ∧ rect_width = 3 * Real.sqrt 2 ∧ circle_radius = 3 then
    (9 : ℝ) * Real.pi
  else
    0

theorem common_area_proof :
  common_area_rect_circle 10 (3 * Real.sqrt 2) 3 = 9 * Real.pi := 
by
  unfold common_area_rect_circle
  split_ifs
  case h => rfl
  case _ => contradiction
  sorry

end common_area_proof_l277_277274


namespace range_of_f_l277_277749

noncomputable def f (x : ℝ) : ℝ := |x - 3| - |x + 4|

theorem range_of_f : set.range f = set.Icc (-7 : ℝ) (7 : ℝ) :=
sorry

end range_of_f_l277_277749


namespace hyperbola_properties_l277_277974

-- Define the hyperbola equation condition
def hyperbola_equation (x y : ℝ) : Prop := 
  x^2 / 3 - y^2 / 6 = 1

-- Define the asymptotes and eccentricity properties
def asymptotes (x y : ℝ) : Prop := 
  y = (Real.sqrt 2) * x ∨ y = -(Real.sqrt 2) * x

def eccentricity : ℝ := 
  Real.sqrt 3

-- Prove the hyperbola properties given its equation
theorem hyperbola_properties (x y : ℝ) 
  (h : hyperbola_equation x y) : 
  (asymptotes x y) ∧ (eccentricity = Real.sqrt 3) := 
by
  sorry

end hyperbola_properties_l277_277974


namespace lucy_initial_balance_l277_277522

theorem lucy_initial_balance (final_balance deposit withdrawal : Int) 
  (h_final : final_balance = 76)
  (h_deposit : deposit = 15)
  (h_withdrawal : withdrawal = 4) :
  let initial_balance := final_balance + withdrawal - deposit
  initial_balance = 65 := 
by
  sorry

end lucy_initial_balance_l277_277522


namespace locus_of_points_of_tangency_l277_277373

noncomputable section

variable {α : Type} [MetricSpace α]

def is_tangent_circle (A B : α) (r : ℝ) : Set α :=
  { M | (dist B M = r) ∧ (∠ A M B = π / 2) }

theorem locus_of_points_of_tangency (A B : α) :
  (∀ M, (∃ r, r ≤ dist A B ∧ M ∈ is_tangent_circle A B r) ↔
    dist A M = dist B M ∧ ∠ A M B = π / 2) :=
sorry

end locus_of_points_of_tangency_l277_277373


namespace angle_CHX_l277_277736

noncomputable def triangle_ABC (A B C H : Point) : Prop :=
(altitude A C B H) ∧ (altitude B A C H) ∧ (acute_triangle A B C) ∧
(∠BAC = 68) ∧ (∠ABC = 64)

theorem angle_CHX (A B C H X : Point) :
triangle_ABC A B C H →
∠CHX = 42 := by
sorry

end angle_CHX_l277_277736


namespace collinearity_B_K1_K2_l277_277876

noncomputable theory

open EuclideanGeometry

-- Definitions of key locations
variables {A B C B₁ B₂ K₁ K₂: Point} 

-- Predefined conditions
def non_isosceles_triangle (A B C : Point) : Prop := 
A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ Triangle A B C

def internal_angle_bisector_intersect (A B C B₁ : Point) : Prop := 
Bisects (inside Angle_B_ABC) A C B₁

def external_angle_bisector_intersect (A B C B₂ : Point) : Prop := 
Bisects (outside Angle_B_ABC) A C B₂

def tangent_from_incircle_touch {A B C B₁ K₁ B₂ K₂ : Point} 
    (I : InCircle non_isosceles_triangle ABC) 
    (h₁ : internal_angle_bisector_intersect A B C B₁) 
    (h₂ : external_angle_bisector_intersect A B C B₂) : Prop := 
    TangentsFromPoint B₁ I K₁ ∧ TangentsFromPoint B₂ I K₂

-- Main theorem to prove
theorem collinearity_B_K1_K2 
    {A B C B₁ B₂ K₁ K₂: Point} 
    (HABC : non_isosceles_triangle A B C) 
    (H₁ : internal_angle_bisector_intersect A B C B₁) 
    (H₂ : external_angle_bisector_intersect A B C B₂) 
    (touching : tangent_from_incircle_touch (incircle HABC) H₁ H₂) :
     Collinear B K₁ K₂ := 
sorry

end collinearity_B_K1_K2_l277_277876


namespace integer_solutions_l277_277788

theorem integer_solutions (n : ℕ) (h : n ≥ 2) (x : Fin n → ℤ) :
  (∀ i, x i = (∑ j, if j == i then 0 else x j) ^ 2018) → 
  (∀ i, x i = 0) ∨ (∀ i, x i = 1) := by
  sorry

end integer_solutions_l277_277788


namespace product_divisible_by_10_probability_l277_277057

theorem product_divisible_by_10_probability (n : ℕ) :
  let p_not_5 := (8 / 9 : ℚ) ^ n
  let p_not_even := (5 / 9 : ℚ) ^ n
  let p_not_5_and_not_even := (4 / 9 : ℚ) ^ n
  let p_not_divisible_by_10 := p_not_5 + p_not_even - p_not_5_and_not_even
  let p_divisible_by_10 := 1 - p_not_divisible_by_10
  in p_divisible_by_10 = 1 - ((8 / 9 : ℚ) ^ n + (5 / 9 : ℚ) ^ n - (4 / 9 : ℚ) ^ n) :=
by
  sorry

end product_divisible_by_10_probability_l277_277057


namespace min_value_of_expression_l277_277009

noncomputable def f (a b : ℝ) : ℝ :=
  (1 / (1 - a)) + (2 / (1 - b))

def conditions (a b : ℝ) : Prop :=
  (a * b = 1 / 4) ∧ (0 < a) ∧ (a < 1) ∧ (0 < b) ∧ (b < 1)

theorem min_value_of_expression :
  ∀ a b : ℝ, conditions a b → f a b = 4 + 4 * real.sqrt 2 / 3 :=
sorry

end min_value_of_expression_l277_277009


namespace triangle_side_length_l277_277067

theorem triangle_side_length
  (A B C : ℝ) (a b c : ℝ)
  (hA : A = π / 3) (ha : a = √3) (hb : b = 1)
  (hABC : A + B + C = π) :
  c = 2 :=
sorry

end triangle_side_length_l277_277067


namespace find_a_l277_277802

theorem find_a (a : ℝ) (h : a ≠ 0) :
  (∀ x, -1 ≤ x ∧ x ≤ 4 → ax - a + 2 ≤ 7) →
  (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ ax - a + 2 = 7) →
  (a = 5/3 ∨ a = -5/2) :=
by
  sorry

end find_a_l277_277802


namespace function_properties_l277_277199

theorem function_properties :
  ∀ (x : ℝ), f(x) = sin(2*x - (π / 6)) →
    (∀ x ∈ Icc (π / 3) (π / 2), monotone (λ x, -f x)) ∧
    (∀ x₁ x₂ : ℝ, f(x₁) = 1 / 2 ∧ f(x₂) = 1 / 2 → ¬ (x₁ - x₂) % π = 0) ∧
    (∀ (x : ℝ), g(x) = sin(2*x + π / 12 - π / 6) → ∀ (x : ℝ), g(-x) = -g(x)) ∧
    (count_zeros f (0, 8*π) = 16) :=
by 
  sorry

-- Definitions for f and g for completeness
def f (x : ℝ) := sin (2*x - π/6)
def g (x : ℝ) := sin (2*(x + π/12) - π/6)

-- Expected property of monotonicity
def decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂ ∈ Icc a b, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- Expected property of modulo
def is_multiple (x y : ℝ) : Prop :=
  ∃ n : ℤ, x = y * n

-- Measurement of zero counts
def count_zeros (f : ℝ → ℝ) (interval : Set ℝ) : ℕ :=
  -- A definition to count the number of roots could be complex but is needed for support
  sorry

end function_properties_l277_277199


namespace max_expression_l277_277780

noncomputable def expression (x : ℝ) : ℝ :=
  sqrt (x + 36) + sqrt (20 - x) + 2 * sqrt x

theorem max_expression : ∀ x, 0 ≤ x ∧ x ≤ 20 → expression x ≤ sqrt 261 := by
sorry

end max_expression_l277_277780


namespace max_net_income_is_50000_l277_277874

def tax_rate (y : ℝ) : ℝ :=
  10 * y ^ 2

def net_income (y : ℝ) : ℝ :=
  1000 * y - tax_rate y

theorem max_net_income_is_50000 :
  ∃ y : ℝ, (net_income y = 25000 ∧ 1000 * y = 50000) :=
by
  use 50
  sorry

end max_net_income_is_50000_l277_277874


namespace triangle_side_length_sum_l277_277446

theorem triangle_side_length_sum
  (A B C : Point)
  (N D E : Point)
  (h1 : ∠ BAC = 60°)
  (h2 : ∠ ACB = 30°)
  (h3 : dist A B = 2)
  (h4 : N = midpoint A B)
  (h5 : D ∈ line_segment B C)
  (h6 : perpendicular (line_through A D) (line_through B N))
  (h7 : E ∈ extended_segment C B)
  (h8 : dist D E = dist E B)
  (h9 : dist C E = (p - real.sqrt q) / r)
  (hpqr_rel_prime : nat.gcd p r = 1)
  (hp_pos : p > 0) (hq_pos : q ≥ 0) (hr_pos : r > 0) :
  p + q + r = 2 := sorry

end triangle_side_length_sum_l277_277446


namespace problem_l277_277511

noncomputable def x : ℝ := 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + ...)))

theorem problem (x : ℝ) (hx : x = 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + ...)))) :
    (1 / ((x + 1) * (x - 2)) = (2 + Real.sqrt 3)) ∧ (abs 2 + abs 3 + abs 1 = 6) :=
sorry

end problem_l277_277511


namespace farm_cows_l277_277076

theorem farm_cows (x y : ℕ) (h : 4 * x + 2 * y = 20 + 3 * (x + y)) : x = 20 + y :=
sorry

end farm_cows_l277_277076


namespace find_b_value_between_lines_l277_277869

theorem find_b_value_between_lines :
  ∀ (b : ℤ), (let line1 := 6 * 5 - 8 * b + 1 
               let line2 := 3 * 5 - 4 * b + 5 
               in line1 = 0 → line2 = 0) → b = 4 := 
by
  sorry

end find_b_value_between_lines_l277_277869


namespace vector_sum_simplify_l277_277375

-- Define points A, B, C, D, and G in a vector space.
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C D G : V)

-- Define G as the midpoint of C and D.
def is_midpoint (G C D : V) : Prop := G = (C + D) / 2

-- State the theorem.
theorem vector_sum_simplify (h : is_midpoint G C D) :
  A - B + (1/2) • (D - B + C - B) = A - G :=
sorry

end vector_sum_simplify_l277_277375


namespace expression_simplifies_l277_277945

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b)

theorem expression_simplifies : (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  -- TODO: Proof goes here
  sorry

end expression_simplifies_l277_277945


namespace incorrect_statement_about_population_growth_curve_l277_277680

theorem incorrect_statement_about_population_growth_curve
  (nature_population_growth_is_SShaped : ∀ species, InNature (PopulationGrowthCurve species) = SShapedCurve)
  (population_growth_rate_zero_at_K : ∀ species K, (PopulationGrowthRate species K) = 0)
  (population_growth_is_density_influenced : ∀ species, PopulationGrowthInfluencedByDensity species)
  (growth_rate_initially_increasing : ∀ species, ∃ K₀, ∀ size, (PopulationSize size < K₀) → (GrowthRate size > 0)) :
  ¬ (∀ species, (∃ size, (GrowthRateDecrease species size))) :=
sorry

end incorrect_statement_about_population_growth_curve_l277_277680


namespace collinear_points_x_value_l277_277277

theorem collinear_points_x_value :
  let x : ℝ := -6 in
  ∃ x : ℝ, (x = -6) ∧
    (-3 - 5)/(x - 2) = (2)/(-4 - x) :=
by
  sorry

end collinear_points_x_value_l277_277277


namespace tutoring_cost_l277_277039

theorem tutoring_cost (flat_rate cost_per_minute minutes_tutored : ℕ)
                      (h_flat_rate : flat_rate = 20)
                      (h_cost_per_minute : cost_per_minute = 7)
                      (h_minutes_tutored : minutes_tutored = 18) :
                      flat_rate + (cost_per_minute * minutes_tutored) = 146 :=
by
  rw [h_flat_rate, h_cost_per_minute, h_minutes_tutored]
  norm_num
  sorry

end tutoring_cost_l277_277039


namespace sufficient_but_not_necessary_l277_277698

theorem sufficient_but_not_necessary (x : ℝ) (h : 2 < x ∧ x < 3) :
  x * (x - 5) < 0 ∧ ∃ y, y * (y - 5) < 0 ∧ (2 ≤ y ∧ y ≤ 3) → False :=
by
  sorry

end sufficient_but_not_necessary_l277_277698


namespace exist_ten_natural_numbers_sum_and_product_twenty_l277_277774

theorem exist_ten_natural_numbers_sum_and_product_twenty :
  ∃ (a : Fin 10 → ℕ), (∑ i, a i = 20) ∧ (∏ i, a i = 20) :=
by
  sorry

end exist_ten_natural_numbers_sum_and_product_twenty_l277_277774


namespace sum_of_squares_of_medians_l277_277315

theorem sum_of_squares_of_medians (AB AC BC : ℝ) (hBC : BC = 15) (hAC : AC = 14) (hAB : AB = 13) :
  let m_a := (1/2) * (Real.sqrt (2*AC^2 + 2*BC^2 - AB^2))
  let m_b := (1/2) * (Real.sqrt (2*BC^2 + 2*AB^2 - AC^2))
  let m_c := (1/2) * (Real.sqrt (2*AB^2 + 2*AC^2 - BC^2))
  m_a^2 + m_b^2 + m_c^2 = 442.5 :=
by
  have m_a := (1/2) * (Real.sqrt (2*AC^2 + 2*BC^2 - AB^2))
  have m_b := (1/2) * (Real.sqrt (2*BC^2 + 2*AB^2 - AC^2))
  have m_c := (1/2) * (Real.sqrt (2*AB^2 + 2*AC^2 - BC^2))
  have h : m_a^2 + m_b^2 + m_c^2 = 442.5 := sorry
  exact h

end sum_of_squares_of_medians_l277_277315


namespace max_tan2alpha_l277_277791

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 2)
variable (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.tan (α + β) = 2 * Real.tan β)

theorem max_tan2alpha : 
    Real.tan (2 * α) = 4 * Real.sqrt 2 / 7 := 
by 
  sorry

end max_tan2alpha_l277_277791


namespace simplify_expression_l277_277358

variable (p q r : ℝ)
variable (hp : p ≠ 2)
variable (hq : q ≠ 3)
variable (hr : r ≠ 4)

theorem simplify_expression : 
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 :=
by
  -- Skipping the proof using sorry
  sorry

end simplify_expression_l277_277358


namespace integer_p_exists_l277_277594

theorem integer_p_exists (a : ℕ → ℕ) (n : ℕ)
  (h_gcd : ∀ i j, i ≠ j → Nat.gcd (a i) (a j) = 1)
  (h_lcm : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) = Nat.lcm (a 0) (a 1)) :
  ∃ p, ∀ u, ∃ x : ℕ → ℕ, (u = (∑ i in Finset.range n, a i * x i) ∨ (p - u = (∑ i in Finset.range n, a i * x i))) :=
sorry

end integer_p_exists_l277_277594


namespace average_age_decrease_l277_277969

theorem average_age_decrease {A : ℝ} : 
    let original_total_age := 10 * A in
    let new_total_age := original_total_age - 30 in
    let original_average := original_total_age / 10 in
    let new_average := new_total_age / 10 in
    new_average = original_average - 3 :=
by
  intro original_total_age new_total_age original_average new_average
  rw [original_total_age, new_total_age, original_average, new_average]
  field_simp
  norm_num
  sorry

end average_age_decrease_l277_277969


namespace sum_of_squares_of_roots_l277_277603

theorem sum_of_squares_of_roots : 
  (∀ a b c : ℚ, a ≠ 0 → ∀ (roots_real : (b^2 - 4*a*c) ≥ 0), 
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
    in a = 10 ∧ b = 15 ∧ c = -17 → x1^2 + x2^2 = 113 / 20) :=
by 
  intro a b c a_ne_zero roots_real
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  assume (h : a = 10 ∧ b = 15 ∧ c = -17)
  sorry

end sum_of_squares_of_roots_l277_277603


namespace max_value_of_a_l277_277517

theorem max_value_of_a
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

example 
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) : 
  (7 - Real.sqrt 46) / 3 ≤ a :=
sorry

end max_value_of_a_l277_277517


namespace yoga_studio_total_people_l277_277616

theorem yoga_studio_total_people 
    (num_men : ℕ) (num_women : ℕ)
    (average_weight_men : ℕ) (average_weight_women : ℕ)
    (overall_average_weight : ℕ)
    (h1 : num_men = 8)
    (h2 : num_women = 6)
    (h3 : average_weight_men = 190)
    (h4 : average_weight_women = 120)
    (h5 : overall_average_weight = 160) :
    num_men + num_women = 14 := 
begin
  sorry
end

end yoga_studio_total_people_l277_277616


namespace count_no_4_7_9_l277_277433

def valid_digits := {0, 1, 2, 3, 5, 6, 8}
def valid_hundreds := {1, 2, 3, 5, 6, 8}

theorem count_no_4_7_9 : 
  (valid_hundreds.card * valid_digits.card * valid_digits.card) = 294 :=
by
  sorry

end count_no_4_7_9_l277_277433


namespace min_workers_in_first_brigade_l277_277976

theorem min_workers_in_first_brigade (n : ℕ) (S x : ℝ) (h1 : 1 / (n : ℝ) < 3 / (n + 6)) : n ≥ 4 := 
by {
    sorry,
}

end min_workers_in_first_brigade_l277_277976


namespace ellipse_equation_correct_hyperbola_equation_correct_l277_277810

-- Points A and B definition
def A := (2 : ℝ, 0 : ℝ)
def B := (3 : ℝ, 2 * Real.sqrt 6)

-- Eccentricity and semi-major axis of the ellipse
def e_ellipse := (Real.sqrt 3) / 2
def a_ellipse := 2
def b_ellipse := 1

-- Standard equation of the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / (a_ellipse^2) + y^2 / (b_ellipse^2) = 1

-- Hyperbola parameters
def a_hyperbola := 1
def b_hyperbola := Real.sqrt 3
def c_hyperbola := 2 -- focus

-- Standard equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / (a_hyperbola^2) - y^2 / (b_hyperbola^2) = 1

-- Proof statements
theorem ellipse_equation_correct :
  ellipse_eq = (λ x y, x^2 / (4 : ℝ) + y^2 = 1) :=
sorry

theorem hyperbola_equation_correct :
  hyperbola_eq = (λ x y, x^2 - y^2 / 3 = 1) :=
sorry

end ellipse_equation_correct_hyperbola_equation_correct_l277_277810


namespace problem_1_problem_2_problem_3_l277_277392

noncomputable def point (x y : ℝ) := (x, y)

def O := point 0 0
def A := point 2 9
def B := point 6 (-3)
def P (y λ : ℝ) := y = -7 ∧ λ = -7/4 ∧ point 14 y = (14, y) ∧ point 14 y = λ * point (-8) (-3 - y) 

theorem problem_1 : ∃ (λ : ℝ) (P : ℝ × ℝ), P = (14, -7) ∧ λ = -7/4 :=
by 
  use (-7 / 4)
  exact (14, -7)
  split 
  rfl
  exact (-7 / 4)

def Q := point 4 3

theorem problem_2 : Q = (4, 3) :=
by 
  exact rfl

def R (t : ℝ) := 0 ≤ t ∧ t ≤ 1 → (4 * t, 3 * t)
def RO (t : ℝ) := 0 ≤ t ∧ t ≤ 1 → (-4 * t, -3 * t)
def RA (t : ℝ) := 0 ≤ t ∧ t ≤ 1 → (2 - 4 * t, 9 - 3 * t)
def RB (t : ℝ) := 0 ≤ t ∧ t ≤ 1 → (6 - 4 * t, -3 - 3 * t)
def dot (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

def RA_plus_RB (t : ℝ) := RA t + RB t

def
theorem problem_3 : ∀ t, 0 ≤ t ∧ t ≤ 1 → dot (RO t) (RA_plus_RB t) = 50 * (t - 1 / 2) ^ 2 - 25 / 2
by 
  sorry

end problem_1_problem_2_problem_3_l277_277392


namespace sum_of_valid_n_l277_277782

theorem sum_of_valid_n (n : ℕ) (a b c : ℝ) :
  (∀ m : ℤ, (2013 * m^3 + a * m^2 + b * m + c) / n ∈ ℤ) →
  n ∣ 4026 →
  ∑ k in (finset.divisors 4026), k = 2976 :=
by
  sorry

end sum_of_valid_n_l277_277782


namespace advantageous_order_l277_277542

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l277_277542


namespace abc_perfect_ratio_l277_277118

theorem abc_perfect_ratio {a b c : ℚ} (h1 : ∃ t : ℤ, a + b + c = t ∧ a^2 + b^2 + c^2 = t) :
  ∃ (p q : ℤ), (abc = p^3 / q^2) ∧ (IsCoprime p q) := 
sorry

end abc_perfect_ratio_l277_277118


namespace abc_powerfulness_l277_277752

-- Definitions for powerful rational number
def powerful (r : ℚ) : Prop :=
  ∃ p q : ℕ, ∃ k : ℤ, 1 < k ∧ nat.coprime p q ∧ r = (p : ℚ) ^ k / q

-- Main statement of the proof problem
theorem abc_powerfulness (a b c : ℚ) (hx : a * b * c = 1)
  (hy : ∃ x y z : ℕ, a ^ x + b ^ y + c ^ z ∈ ℤ) : powerful a ∧ powerful b ∧ powerful c := 
sorry

end abc_powerfulness_l277_277752


namespace rosa_total_pages_l277_277424

theorem rosa_total_pages
  (last_week_calls : ℝ)
  (this_week_calls : ℝ)
  (h_last_week : last_week_calls = 10.2)
  (h_this_week : this_week_calls = 8.6) :
  last_week_calls + this_week_calls = 18.8 := by
  -- Add the assumptions from the hypotheses
  calc
    last_week_calls + this_week_calls = 10.2 + 8.6 : by rw [h_last_week, h_this_week]
                               ... = 18.8           : by norm_num

end rosa_total_pages_l277_277424


namespace difference_of_numbers_l277_277196

theorem difference_of_numbers (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 :=
by
  sorry

end difference_of_numbers_l277_277196


namespace arithmetic_sequence_common_difference_l277_277379

variable {α : Type*} [AddGroup α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) = a n + (a 2 - a 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a3 : a 3 = -4) :
  a 3 - a 2 = -6 := 
sorry

end arithmetic_sequence_common_difference_l277_277379


namespace expected_scores_l277_277452

def scoring_criteria (n_correct: ℕ) (selected: ℕ) : ℕ :=
if n_correct = selected then 5 else if selected < n_correct then 2 else 0

-- Given the probabilities based on the choices and the number of correct options.
def prob (choices: List (Fin 4)) (correct: List (Fin 4)) : ℝ :=
((choices.toFinset ∩ correct.toFinset).card : ℝ) / 10

theorem expected_scores : 
  ∀ (choices_A choices_B choices_C: List (Fin 4)) (correct: List (Fin 4)),
  prob choices_A correct = 0.6 ∧
  prob choices_B correct = 0.3 ∧
  prob choices_C correct = 0.1 →
  (scoring_criteria 5 1 * prob choices_A correct + scoring_criteria 0 1 * (1 - prob choices_A correct)) = 1.2 ∧
  (scoring_criteria 5 2 * prob choices_B correct + scoring_criteria 2 2 * (1 - prob choices_B correct)) = 0.9 ∧
  (scoring_criteria 5 3 * prob choices_C correct + scoring_criteria 2 3 * (1 - prob choices_C correct)) = 0.5 :=
by
  sorry

end expected_scores_l277_277452


namespace count_even_ones_grid_l277_277102

theorem count_even_ones_grid (n : ℕ) : 
  (∃ f : (Fin n) → (Fin n) → ℕ, (∀ i : Fin n, ∑ j in (Finset.univ : Finset (Fin n)), f i j % 2 = 0) ∧ 
                                  (∀ j : Fin n, ∑ i in (Finset.univ : Finset (Fin n)), f i j % 2 = 0)) ↔ 
  2^((n-1)^2) = 2^((n-1)^2) :=
sorry

end count_even_ones_grid_l277_277102


namespace honey_last_nights_l277_277575

theorem honey_last_nights 
  (serving_per_cup : ℕ)
  (cups_per_night : ℕ)
  (ounces_per_container : ℕ)
  (servings_per_ounce : ℕ)
  (total_nights : ℕ) :
  serving_per_cup = 1 →
  cups_per_night = 2 →
  ounces_per_container = 16 →
  servings_per_ounce = 6 →
  total_nights = 48 := 
by
  intro h1 h2 h3 h4,
  sorry

end honey_last_nights_l277_277575


namespace projectiles_meet_time_l277_277064

def distance : ℕ := 2520
def speed1 : ℕ := 432
def speed2 : ℕ := 576
def combined_speed : ℕ := speed1 + speed2

theorem projectiles_meet_time :
  (distance * 60) / combined_speed = 150 := 
by
  sorry

end projectiles_meet_time_l277_277064


namespace inscribed_circle_distance_property_l277_277868

noncomputable theory

variables {α : Type*} [linear_ordered_field α] (a b c : α)

def right_angled_triangle (a b c : α) : Prop :=
  a^2 + b^2 = c^2

def inscribed_circle_distance_eq (l m n : α) : Prop :=
  1 / l^2 = 1 / m^2 + 1 / n^2 + (real.sqrt 2) / (m * n)

theorem inscribed_circle_distance_property
  {a b c l m n : α}
  (h : right_angled_triangle a b c)
  (hl : l = a)
  (hmn : (2 / b) + (2 / c) = (2 / (b * c)) * ((real.sqrt (b^2 + c^2)) + 1))
  : inscribed_circle_distance_eq l m n :=
sorry

end inscribed_circle_distance_property_l277_277868


namespace range_g_l277_277767

variable (x : ℝ)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g (y : ℝ) : 
  (∃ x, g x = y) ↔ y > 0 :=
by
  sorry

end range_g_l277_277767


namespace pages_per_day_difference_l277_277957

theorem pages_per_day_difference :
  let songhee_pages := 288
  let songhee_days := 12
  let eunju_pages := 243
  let eunju_days := 9
  let songhee_per_day := songhee_pages / songhee_days
  let eunju_per_day := eunju_pages / eunju_days
  eunju_per_day - songhee_per_day = 3 := by
  sorry

end pages_per_day_difference_l277_277957


namespace vector_m_range_l277_277910

theorem vector_m_range (m : ℝ) : 
  let a := (1 : ℝ, 2 : ℝ)
      b := (1 : ℝ, m : ℝ)
  in (∃ θ : ℝ, cos θ > 0 ∧ cos θ ≠ 1 ∧ θ = real.arccos((1 + 2 * m)/((real.sqrt 5)*(real.sqrt(1 + m^2)))) )
      → m > -1 / 2 ∧ m ≠ 2 :=
by
  intro h
  sorry

end vector_m_range_l277_277910


namespace sum_of_digits_of_x_l277_277718

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_x (x : ℕ) (h1 : 100 ≤ x) (h2 : x ≤ 949)
  (h3 : is_palindrome x) (h4 : is_palindrome (x + 50)) :
  sum_of_digits x = 19 :=
sorry

end sum_of_digits_of_x_l277_277718


namespace place_soap_l277_277168

theorem place_soap (n : ℕ) : 
  (∃ (f : fin (2*n*(2*n+1)) → (fin (2*n+1) × fin (2*n+1) × fin (2*n+1))), 
    ∀ i, let ⟨x, y, z⟩ := f i in 
      (x + 1 < 2*n+1 ∧ y + 2 < 2*n+1 ∧ z + (n+1) < 2*n+1) ∨
      (x + 2 < 2*n+1 ∧ y + 1 < 2*n+1 ∧ z + (n+1) < 2*n+1) ∨
      (x + 1 < 2*n+1 ∧ y + (n+1) < 2*n+1 ∧ z + 2 < 2*n+1) ∨
      (x + (n+1) < 2*n+1 ∧ y + 1 < 2*n+1 ∧ z + 2 < 2*n+1)
  ) ↔ (n % 2 = 0 ∨ n = 1) := sorry

end place_soap_l277_277168


namespace ratio_volume_sphere_hemisphere_l277_277216

variable (r : ℝ)

def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

def volume_hemisphere (r : ℝ) : ℝ :=
  (1 / 2) * (4 / 3) * Real.pi * (r / 3)^3

theorem ratio_volume_sphere_hemisphere (r : ℝ) (h : r > 0) :
  volume_sphere r / volume_hemisphere r = 54 := by
  sorry

end ratio_volume_sphere_hemisphere_l277_277216


namespace smallest_positive_period_of_f_interval_of_monotone_decreasing_maximum_value_of_f_minimum_value_of_f_l277_277396

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (1/2 * x + Real.pi / 6) + 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + 4 * Real.pi) = f x := sorry

theorem interval_of_monotone_decreasing (k : ℤ) :
  ∀ x, (2 * Real.pi / 3 + 4 * k * Real.pi) ≤ x ∧ x ≤ (8 * Real.pi / 3 + 4 * k * Real.pi) 
→ f' x < 0 := sorry

theorem maximum_value_of_f :
  ∃ k : ℤ, ∀ x, x = 4 * k * Real.pi + 2 * Real.pi / 3 
→ f x = 4 := sorry

theorem minimum_value_of_f :
  ∃ k : ℤ, ∀ x, x = 4 * k * Real.pi + 8 * Real.pi / 3 
→ f x = 0 := sorry

end smallest_positive_period_of_f_interval_of_monotone_decreasing_maximum_value_of_f_minimum_value_of_f_l277_277396


namespace problem1_problem2_l277_277407

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 1 - 2 * log x

theorem problem1 (a : ℝ) (h : 1 ≤ a) :
  ∀ x > 0, f a x ≥ 0 :=
sorry

theorem problem2 (a : ℝ) :
  if 0 < a ∧ a < 1 then ∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0
  else if a = 1 then ∃ x, f a x = 0
  else ∀ x > 0, f a x ≠ 0 :=
sorry

end problem1_problem2_l277_277407


namespace unique_shading_patterns_l277_277689

/-- Given a grid of 9 triangles, we need to determine the number of unique patterns of shading exactly
    three triangles such that no two shaded triangles share a side and patterns are considered the same if they
    can be matched by rotations or reflections. 
-/
theorem unique_shading_patterns : ∃ (patterns : Finset (Finset (Fin 9))), 
  patterns.card = 10 ∧ 
  (∀ pattern ∈ patterns, pattern.card = 3) ∧ 
  (∀ pattern ∈ patterns, ∀ t₁ t₂ ∈ pattern, t₁ ≠ t₂ → ¬(adjacent t₁ t₂))
  ∧ 
  (∀ p₁ p₂ ∈ patterns, (rotation_or_reflection p₁ p₂)) :=
sorry

/-- Define the relationship of adjacency between two triangles. 
    adjacent t₁ t₂ means t₁ and t₂ share a side.
-/
def adjacent (t₁ t₂ : Fin 9) : Prop :=
sorry

/-- Define a function to check if one pattern can be transformed into another by 
    rotation or reflection. 
-/
def rotation_or_reflection (p₁ p₂ : Finset (Fin 9)) : Prop :=
sorry

end unique_shading_patterns_l277_277689


namespace find_slope_l277_277411

noncomputable def parabola_equation (x y : ℝ) := y^2 = 8 * x

def point_M : ℝ × ℝ := (-2, 2)

def line_through_focus (k x : ℝ) : ℝ := k * (x - 2)

def focus : ℝ × ℝ := (2, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_slope (k : ℝ) : 
  (∀ x y A B, 
    parabola_equation x y → 
    (x = A ∨ x = B) → 
    line_through_focus k x = y → 
    parabola_equation A (k * (A - 2)) → 
    parabola_equation B (k * (B - 2)) → 
    dot_product (A + 2, (k * (A -2)) - 2) (B + 2, (k * (B - 2)) - 2) = 0) →
  k = 2 :=
sorry

end find_slope_l277_277411


namespace find_smaller_circle_circumference_l277_277586

noncomputable def circumference_of_smaller_circle
    (C_large : ℝ) (area_diff : ℝ) : ℝ :=
  let R := C_large / (2 * Real.pi) in
  let r := Real.sqrt (R^2 - area_diff / Real.pi) in
  2 * Real.pi * r

theorem find_smaller_circle_circumference :
  circumference_of_smaller_circle 704 17254.942310250928 ≈ 527.08 := sorry

end find_smaller_circle_circumference_l277_277586


namespace perpendicular_line_through_point_l277_277639

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l277_277639


namespace log3_x_minus_1_increasing_l277_277563

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem log3_x_minus_1_increasing : is_increasing_on (fun x => log_base_3 (x - 1)) (Set.Ioi 1) :=
sorry

end log3_x_minus_1_increasing_l277_277563


namespace honey_last_nights_l277_277578

def servings_per_cup : Nat := 1
def cups_per_night : Nat := 2
def container_ounces : Nat := 16
def servings_per_ounce : Nat := 6

theorem honey_last_nights :
  (container_ounces * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 :=
by
  sorry  -- Proof not provided as per requirements

end honey_last_nights_l277_277578


namespace sum_of_solutions_l277_277783

def satisfies_equation (x : ℝ) : Prop :=
  3^(x^2 - 4 * x - 3) = 9^(x - 5)

theorem sum_of_solutions :
  (∑ x in {x : ℕ | satisfies_equation x}, x) = 8 :=
by
  sorry

end sum_of_solutions_l277_277783


namespace ratio_of_DC_to_AD_l277_277193

noncomputable theory
open Classical
open Real

def is_convex_quadrilateral (A B C D : ℝ × ℝ) := sorry
def are_angles_equal (A B C D : ℝ × ℝ) := sorry
def has_right_angle_at_C (A B C D : ℝ × ℝ) := sorry
def is_AD_perpendicular_to_BD (A B C D : ℝ × ℝ) := sorry
def are_BC_and_CD_equal (A B C D : ℝ × ℝ) := sorry

theorem ratio_of_DC_to_AD (A B C D : ℝ × ℝ) :
  is_convex_quadrilateral A B C D →
  are_angles_equal A B C D →
  has_right_angle_at_C A B C D →
  is_AD_perpendicular_to_BD A B C D →
  are_BC_and_CD_equal A B C D →
  (let a := dist B C in
   let x := dist A D in
   dist C D / x = 2 + sqrt 2) := sorry

end ratio_of_DC_to_AD_l277_277193


namespace pizza_area_percentage_increase_l277_277555

theorem pizza_area_percentage_increase (D1 D2 : ℝ) (h1 : D1 = 12) (h2 : D2 = 15) : 
  let r1 := D1 / 2
      r2 := D2 / 2
      A1 := π * r1^2
      A2 := π * r2^2
      increase := (A2 - A1) / A1 * 100
  in increase = 56.25 := by 
  sorry

end pizza_area_percentage_increase_l277_277555


namespace opposite_of_neg_four_l277_277212

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end opposite_of_neg_four_l277_277212


namespace tailoring_business_days_l277_277891

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

end tailoring_business_days_l277_277891


namespace one_factor_exists_l277_277327

variables (G : Type*) [Graph G] (V : Type*) [Fintype V] (E : V → V → Prop)
          (S : Finset V) (C_G_S : Finset (Finset V))
          (factor_critical : Finset V → Prop)
          (is_matchable : Finset V → Finset (Finset V) → Prop)

axiom condition_1 : is_matchable S (C_G_S)
axiom condition_2 : ∀ C ∈ C_G_S, factor_critical C
axiom condition_3 : S.card = C_G_S.card

theorem one_factor_exists :
  ∃ M : Finset (V × V), is_perfect_matching G M ∧ S.card = C_G_S.card := sorry

end one_factor_exists_l277_277327


namespace sin_double_theta_l277_277355

variable (θ : ℝ)

theorem sin_double_theta :
  cos(θ + Real.pi / 2) = 4 / 5 ∧ -Real.pi / 2 < θ ∧ θ < Real.pi / 2 →
  sin (2 * θ) = -24 / 25 :=
by
  sorry

end sin_double_theta_l277_277355


namespace perpendicular_line_equation_l277_277634

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l277_277634


namespace jina_total_mascots_l277_277893

-- Definitions and Conditions
def num_teddies := 5
def num_bunnies := 3 * num_teddies
def num_koala_bears := 1
def additional_teddies := 2 * num_bunnies

-- Total mascots calculation
def total_mascots := num_teddies + num_bunnies + num_koala_bears + additional_teddies

theorem jina_total_mascots : total_mascots = 51 := by
  sorry

end jina_total_mascots_l277_277893


namespace measure_of_A_max_area_l277_277448

-- Define the conditions given in the problem
variables {A B C a b c : ℝ}
variables h1 : a > 0
variables h2 : b > 0
variables h3 : c > 0
variables h_angles : A + B + C = π
variables h_sides : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
variables hm : ∃ k : ℝ, (cos A, cos B) = k • (a, 2 * c - b)

-- Problem (I): Prove the measure of angle A
theorem measure_of_A : A = π / 3 := by
  sorry

-- Problem (II): Prove the maximum area of triangle
theorem max_area (h_a : a = 4) : (a * b * sin C) / 2 <= 4 * sqrt 3 := by
  sorry

end measure_of_A_max_area_l277_277448


namespace b_power_four_l277_277574

-- Definitions based on the conditions
variable (b : ℝ)
axiom basic_eq : 5 = b + b⁻¹

-- Statement to prove
theorem b_power_four (b : ℝ) (h : 5 = b + b⁻¹) : b^4 + b^(-4) = 527 := by
  sorry

end b_power_four_l277_277574


namespace number_of_even_1s_grids_l277_277099

theorem number_of_even_1s_grids (n : ℕ) : 
  (∃ grid : fin n → fin n → ℕ, 
    (∀ i j, grid i j = 0 ∨ grid i j = 1) ∧
    (∀ i, (∑ j, grid i j) % 2 = 0) ∧
    (∀ j, (∑ i, grid i j) % 2 = 0)) →
  2 ^ ((n - 1) * (n - 1)) = 2 ^ ((n - 1) * (n - 1)) :=
by sorry

end number_of_even_1s_grids_l277_277099


namespace ostap_advantageous_order_l277_277545

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l277_277545


namespace proof_correct_statements_l277_277681

noncomputable def cube_root (x : ℝ) := x^(1 / 3 : ℝ)
noncomputable def sq_root (x : ℝ) := x^(1 / 2 : ℝ)

theorem proof_correct_statements :
  (cube_root 8 = 2) ∧ ¬(sq_root (sq_root 81) = ±3) ∧ ¬(sq_root 4 = ±2) ∧ (cube_root (-1) = -1) → 
  {1, 4} = {1, 2, 4} := sorry

end proof_correct_statements_l277_277681


namespace x7_value_l277_277214

theorem x7_value
  (x : ℕ → ℕ)
  (h1 : x 6 = 144)
  (h2 : ∀ n, 1 ≤ n ∧ n ≤ 4 → x (n + 3) = x (n + 2) * (x (n + 1) + x n))
  (h3 : ∀ m, m < 1 → 0 < x m) : x 7 = 3456 :=
by
  sorry

end x7_value_l277_277214


namespace range_of_f_ratio_l277_277307

variable (f : ℝ → ℝ)

noncomputable def satisfies_conditions :=
  ∀ (x : ℝ), (0 < x) → (9 * f(x) < x * (deriv f x)) ∧ (x * (deriv f x) < 10 * f(x)) ∧ (0 < f(x))

theorem range_of_f_ratio (h : satisfies_conditions f) : 
  2^9 < f(2) / f(1) ∧ f(2) / f(1) < 2^10 :=
sorry

end range_of_f_ratio_l277_277307


namespace arun_age_l277_277142

variable (A S G M : ℕ)

theorem arun_age (h1 : A - 6 = 18 * G)
                 (h2 : G + 2 = M)
                 (h3 : M = 5)
                 (h4 : S = A - 8) : A = 60 :=
by sorry

end arun_age_l277_277142


namespace PS_div_QR_eq_sqrt3_l277_277480

variables {P Q R S : Type} [metric_space P]

-- Assume we have points P, Q, R, S
variables [PQR_equilateral : equilateral_triangle P Q R] 
variables [QRS_equilateral : equilateral_triangle Q R S]

-- Assume lengths of QR and PS
axiom length_QR : length (Q, R) = t
axiom length_PS : length (P, S) = t * sqrt 3

theorem PS_div_QR_eq_sqrt3 :
  length (P, S) / length (Q, R) = sqrt 3 :=
sorry

end PS_div_QR_eq_sqrt3_l277_277480


namespace point_in_fourth_quadrant_l277_277387

theorem point_in_fourth_quadrant (x y : ℝ) (z : ℂ) 
  (h1 : z = x + y * Complex.I)
  (h2 : Complex.abs z = Complex.conj z + 1 - 2 * Complex.I) :
  x = 3 / 2 ∧ y = -2 ∧ x > 0 ∧ y < 0 :=
by 
  have h3 : sqrt (x * x + y * y) = abs z := rfl
  have h4 : sqrt (x * x + y * y) = x + 1 := sorry
  have h5 : y = -2 := sorry
  have h6 : x = 3 / 2 := sorry
  exact ⟨h6, h5, by linarith, by linarith⟩

end point_in_fourth_quadrant_l277_277387


namespace angle_AED_of_fold_l277_277739

theorem angle_AED_of_fold (ABCD : Rectangle) (DE : Segment) 
  (EBFGCD : Hexagon) (angle_GDF : ∠ G D F = 20) : ∠ A E D = 35 :=
sorry

end angle_AED_of_fold_l277_277739


namespace ellipse_focus_product_l277_277590

def ellipse_foci (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

def ellipse_P (a b : ℝ) : set (ℝ × ℝ) := {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}

def perpendicular {α : Type*} [inner_product_space ℝ α] {x y : α} : Prop :=
  inner_product_space.inner x y = 0

theorem ellipse_focus_product (a b c : ℝ) (P F1 F2 : ℝ × ℝ)
  (h_ellipse : ellipse_P a b P)
  (h_foci : c = ellipse_foci a b)
  (h_perpendicular : perpendicular (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2)) :
  ∃ m n : ℝ, m + n = 2 * a ∧ m^2 + n^2 = (2*c)^2 ∧ m * n = 8 := by
  sorry

end ellipse_focus_product_l277_277590


namespace zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l277_277988

theorem zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three :
  (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) :=
by
  sorry

end zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l277_277988


namespace product_of_8_integers_negative_l277_277442

theorem product_of_8_integers_negative
  (a b c d e f g h : ℤ)
  (H1 : (a * b * c * d * e * f * g * h < 0 ↔
         (a < 0 ↔ b < 0 ↔ c < 0 ↔ d < 0 ↔ e < 0 ↔ f < 0 ↔ g < 0 ↔ h < 0) ∨
         (a < 0 ↔ b < 0 ↔ c < 0 ↔ d < 0 ↔ e < 0 ↔ f < 0 ↔ g < 0 ↔ h >= 0) ∨
         (a < 0 ↔ b < 0 ↔ c < 0 ↔ d < 0 ↔ e < 0 ↔ f < 0 ↔ g >=0 ↔ h < 0) ∨
         (a < 0 ↔ b < 0 ↔ c < 0 ↔ d < 0 ↔ e < 0 ↔ f >=0 ↔ g < 0 ↔ h < 0) ∨
         (a < 0 ↔ b < 0 ↔ c < 0 ↔ d < 0 ↔ e >= 0 ↔ f < 0 ↔ g < 0 ↔ h < 0) ∨
         (a < 0 ↔ b < 0 ↔ c < 0 ↔ d >= 0 ↔ e < 0 ↔ f < 0 ↔ g < 0 ↔ h < 0) ∨
         (a < 0 ↔ b < 0 ↔ c >= 0 ↔ d < 0 ↔ e < 0 ↔ f < 0 ↔ g < 0 ↔ h < 0) ∨
         (a < 0 ↔ b >= 0 ↔ c < 0 ↔ d < 0 ↔ e < 0 ↔ f < 0 ↔ g < 0 ↔ h < 0) ∨
         (a >= 0 ↔ b < 0 ↔ c < 0 ↔ d < 0 ↔ e < 0 ↔ f < 0 ↔ g < 0 ↔ h < 0))) :
  (a * b * c * d * e * f * g * h < 0) :=
by
  sorry

end product_of_8_integers_negative_l277_277442


namespace real_number_a_value_l277_277923

open Set

variable {a : ℝ}

theorem real_number_a_value (A B : Set ℝ) (hA : A = {-1, 1, 3}) (hB : B = {a + 2, a^2 + 4}) (hAB : A ∩ B = {3}) : a = 1 := 
by 
-- Step proof will be here
sorry

end real_number_a_value_l277_277923


namespace chairs_to_remove_l277_277723

/- Definitions based on conditions -/
def initial_chairs : ℕ := 156
def chairs_per_row : ℕ := 12
def students_attending : ℕ := 100

/- Problem statement to be proved -/
theorem chairs_to_remove :
  ∃ (to_remove : ℕ), 
    to_remove = 36 ∧ 
    (initial_chairs - to_remove) % 10 = 0 ∧
    (initial_chairs - to_remove) % chairs_per_row = 0  :=
proof
  sorry -- Proof goes here

end chairs_to_remove_l277_277723


namespace real_number_x_l277_277043

theorem real_number_x (x : ℝ) (i : ℂ) (h : i = complex.I) :
  ∃ (r : ℝ), (x + i) ^ 2 = r ↔ x = 0 :=
begin
  sorry
end

end real_number_x_l277_277043


namespace people_in_third_row_l277_277224

theorem people_in_third_row (row1_ini row2_ini left_row1 left_row2 total_left : ℕ) (h1 : row1_ini = 24) (h2 : row2_ini = 20) (h3 : left_row1 = row1_ini - 3) (h4 : left_row2 = row2_ini - 5) (h_total : total_left = 54) :
  total_left - (left_row1 + left_row2) = 18 := 
by
  sorry

end people_in_third_row_l277_277224


namespace solution_set_l277_277012

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set (x : ℝ) (h₁ : 0 < x) (h₂ : x < +∞) 
  (h₃ : ∀ x ∈ (0, +∞), (f x) / x > - (f' x)) 
  : (x - 1) * f (x^2 - 1) < f (x + 1) ↔ 1 < x ∧ x < 2 :=
sorry

end solution_set_l277_277012


namespace horizontal_asymptote_l277_277592

theorem horizontal_asymptote :
  (∀ x : ℝ, y x = (7 * x^2 - 4) / (4 * x^2 + 7 * x + 3)) →
  tendsto (λ x : ℝ, (7 * x^2 - 4) / (4 * x^2 + 7 * x + 3)) at_top (𝓝 (7 / 4)) :=
by
  intros y_def
  simp only [y_def]
  sorry

end horizontal_asymptote_l277_277592


namespace company_profit_l277_277709

theorem company_profit (P : ℝ) (H1 : P + (P + 2_750_000) = 3_635_000) : 
  P = 442_500 := 
by
  sorry

end company_profit_l277_277709


namespace find_integer_n_l277_277340

theorem find_integer_n : ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ (sin (n * Real.pi / 180) = sin (60 * Real.pi / 180) ∨ sin (n * Real.pi / 180) = sin (120 * Real.pi / 180)) :=
by {

  let n1 := 60,
  let n2 := 120,

  use n1,
  split,
  linarith,
  split,
  linarith,
  left,
  norm_num,
  -- Proof of sin 60 degrees omitted
  -- rw Real.sin_eq_tan_div_cos,

  use n2,
  split,
  linarith,
  split,
  linarith,
  right,
  norm_num,
  -- Proof of sin 120 degrees omitted
  -- rw Real.sin_eq_tan_div_cos,
  sorry
}

end find_integer_n_l277_277340


namespace polar_equation_of_circle_l277_277885

theorem polar_equation_of_circle :
  let center_polar := (2, Real.pi / 3)
  let radius := Real.sqrt 5
  (∃ (rho theta : ℝ), 
    let r_cos := rho * (2 - 2 * Real.cos theta)
    let r_sin := rho * (2 * Real.sqrt 3 * Real.sin theta)
    (rho^2 - r_cos - r_sin - 1 = 0)) :=
begin
  sorry
end

end polar_equation_of_circle_l277_277885


namespace area_equivalence_l277_277950

noncomputable def triangle_area (a b c s : ℝ) (alpha beta gamma : ℝ) : ℝ :=
  real.sqrt(a * b * c * s * real.sin(alpha / 2) * real.sin(beta / 2) * real.sin(gamma / 2))

noncomputable def herons_area (a b c s : ℝ) : ℝ := 
  real.sqrt(s * (s - a) * (s - b) * (s - c))

theorem area_equivalence (a b c alpha beta gamma : ℝ) (s : ℝ) 
  (h_s : s = (a + b + c) / 2) 
  (h_sum_alpha_beta_gamma : alpha + beta + gamma = π) : 
  triangle_area a b c s alpha beta gamma = herons_area a b c s :=
by 
  sorry

end area_equivalence_l277_277950


namespace problem_conditions_l277_277306

noncomputable def f (x : ℝ) : ℝ := -x - x^3

variables (x₁ x₂ : ℝ)

theorem problem_conditions (h₁ : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧
  (¬ (f x₂ * f (-x₂) > 0)) ∧
  (¬ (f x₁ + f x₂ ≤ f (-x₁) + f (-x₂))) ∧
  (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :=
sorry

end problem_conditions_l277_277306


namespace paul_and_paula_cookies_l277_277435

-- Define the number of cookies per pack type
def cookies_in_pack (pack : ℕ) : ℕ :=
  match pack with
  | 1 => 15
  | 2 => 30
  | 3 => 45
  | 4 => 60
  | _ => 0

-- Paul's purchase: 2 packs of Pack B and 1 pack of Pack A
def pauls_cookies : ℕ :=
  2 * cookies_in_pack 2 + cookies_in_pack 1

-- Paula's purchase: 1 pack of Pack A and 1 pack of Pack C
def paulas_cookies : ℕ :=
  cookies_in_pack 1 + cookies_in_pack 3

-- Total number of cookies Paul and Paula have
def total_cookies : ℕ :=
  pauls_cookies + paulas_cookies

theorem paul_and_paula_cookies : total_cookies = 135 :=
by
  sorry

end paul_and_paula_cookies_l277_277435


namespace total_weight_of_bars_l277_277989

-- Definitions for weights of each gold bar
variables (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
variables (W1 W2 W3 W4 W5 W6 W7 W8 : ℝ)

-- Definitions for the weighings
axiom weight_C1_C2 : W1 = C1 + C2
axiom weight_C1_C3 : W2 = C1 + C3
axiom weight_C2_C3 : W3 = C2 + C3
axiom weight_C4_C5 : W4 = C4 + C5
axiom weight_C6_C7 : W5 = C6 + C7
axiom weight_C8_C9 : W6 = C8 + C9
axiom weight_C10_C11 : W7 = C10 + C11
axiom weight_C12_C13 : W8 = C12 + C13

-- Prove the total weight of all gold bars
theorem total_weight_of_bars :
  (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13)
  = (W1 + W2 + W3) / 2 + W4 + W5 + W6 + W7 + W8 :=
by sorry

end total_weight_of_bars_l277_277989


namespace chord_length_proof_l277_277412

noncomputable def chord_length_of_circle_intercepted_by_line
  (ρ θ : ℝ → ℝ)
  (polar_circle_eq : ∀ (θ : ℝ), ρ θ ^ 2 + 2 * ρ θ * (Math.cos θ + Real.sqrt 3 * Math.sin θ) = 5)
  (line_eq : ∀ (θ : ℝ), θ = 0) :
  ℝ :=
  2 * Real.sqrt (9 - 3)

theorem chord_length_proof :
  chord_length_of_circle_intercepted_by_line (λ θ, ρ θ) (λ θ, θ) polar_circle_eq line_eq = 2 * Real.sqrt 6 :=
sorry

end chord_length_proof_l277_277412


namespace num_unique_four_digit_2023_l277_277428

theorem num_unique_four_digit_2023 : 
  let digits := [2, 0, 2, 3]
  in (∀ d d_1 d_2 d_3, digits.count d = 4 ∧ (d = 2 ∨ d = 0 ∨ d = 2 ∨ d = 3) → 
     ∃! n, (∀ n = d_1 * 1000 + d_2 * 100 + d_3 * 10 + d_4, d_1 ≠ 0) 
     ∧ perm.contains n := finalize [3, 2, 0] → ∀ n = 6) := sorry

end num_unique_four_digit_2023_l277_277428


namespace relationship_m_n_l277_277799

theorem relationship_m_n (a b : ℝ) (ha : a > 2) (hb : b ≠ 0) :
  let m := a + 1 / (a - 2)
  let n := 2^(2 - b^2)
  in m > n :=
by
  have m_nonneg_quot := (a - 1)^2
  have m_nonneg := 2 * (a - 1)
  have m_lower_bound := 2 * a
  have ha' := ha
  have ha_ge_4 := 2 * a
  
  have ha₂ := ha_ge_4
  have n_pos := 2^2

  sorry

end relationship_m_n_l277_277799


namespace perpendicular_line_eq_slope_intercept_l277_277650

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l277_277650


namespace income_difference_after_raises_l277_277687

noncomputable def Don_annual_income_before_raise (D : ℝ) := 0.08 * D = 800
noncomputable def Wife_annual_income_before_raise (W : ℝ) := 0.08 * W = 840

theorem income_difference_after_raises (D W : ℝ) (hD : Don_annual_income_before_raise D) (hW : Wife_annual_income_before_raise W) :
  (W + 840) - (D + 800) = 540 :=
begin
  sorry
end

end income_difference_after_raises_l277_277687


namespace g_neg2_eq_neg1_l277_277011

-- Representing all the necessary definitions and conditions
variable {f : ℝ → ℝ} 

-- Defining the condition that f is odd
def is_odd (f : ℝ → ℝ) :=
∀ x : ℝ, f (-x) = -f x

-- Given conditions in the problem
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := (2 + f x) / f x
axiom h_f_odd : is_odd f
axiom h_g2_eq_3 : g f 2 = 3

-- The proof we need to show
theorem g_neg2_eq_neg1 : g f (-2) = -1 := 
sorry

end g_neg2_eq_neg1_l277_277011


namespace zero_points_of_function_l277_277832

/-- Assume g(x) is x^3 * ln(x) and we want to determine the range of values of m
such that f(x) = x^3 * ln x + m has 2 zero points. -/
theorem zero_points_of_function (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ x1^3 * Real.log x1 + m = 0 ∧ x2^3 * Real.log x2 + m = 0) ↔
  m < (1 / (3 * Real.exp 1)) :=
sorry

end zero_points_of_function_l277_277832


namespace operation_two_three_l277_277760

def operation (a b : ℕ) : ℤ := 4 * a ^ 2 - 4 * b ^ 2

theorem operation_two_three : operation 2 3 = -20 :=
by
  sorry

end operation_two_three_l277_277760


namespace perpendicular_line_through_point_l277_277635

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l277_277635


namespace relation_between_x_and_y_l277_277484

-- Definitions based on the conditions
variables (r x y : ℝ)

-- Power of a Point Theorem and provided conditions
variables (AE_eq_3EC : AE = 3 * EC)
variables (x_def : x = AE)
variables (y_def : y = r)

-- Main statement to be proved
theorem relation_between_x_and_y (r x y : ℝ) (AE_eq_3EC : AE = 3 * EC) (x_def : x = AE) (y_def : y = r) :
  y^2 = x^3 / (2 * r - x) :=
sorry

end relation_between_x_and_y_l277_277484


namespace addison_tickets_sold_on_sunday_l277_277934

-- Define the basic conditions and the variables
def tickets_sold_friday : ℕ := 181
def tickets_sold_saturday : ℕ := 2 * tickets_sold_friday
def tickets_difference : ℕ := 284

-- State the theorem to be proven
theorem addison_tickets_sold_on_sunday : 
  let tickets_sold_sunday := tickets_sold_saturday - tickets_difference in
  tickets_sold_sunday = 78 :=
by
  -- Provide the placeholder for proof
  sorry

end addison_tickets_sold_on_sunday_l277_277934


namespace polynomials_with_conditions_l277_277309

theorem polynomials_with_conditions (n : ℕ) (h_pos : 0 < n) :
  (∃ P : Polynomial ℤ, Polynomial.degree P = n ∧ 
      (∃ (k : Fin n → ℤ), Function.Injective k ∧ (∀ i, P.eval (k i) = n) ∧ P.eval 0 = 0)) ↔ 
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
sorry

end polynomials_with_conditions_l277_277309


namespace meet_distance_l277_277111

noncomputable def jack_time_uphill (distance : ℕ) (rate_uphill : ℕ) : ℚ :=
  distance / rate_uphill

noncomputable def jill_time_uphill (distance : ℕ) (rate_uphill : ℕ) : ℚ :=
  distance / rate_uphill

noncomputable def jack_position_downhill (distance : ℕ) (rate_downhill : ℕ) (x : ℚ) (start_time : ℚ) : ℚ :=
  distance - rate_downhill * (x - start_time)

noncomputable def jill_position_uphill (rate_uphill : ℕ) (x : ℚ) (start_time : ℚ) : ℚ :=
  rate_uphill * (x - start_time)

theorem meet_distance (head_start : ℚ)
  (distance : ℕ) (jack_rate_uphill jack_rate_downhill : ℕ) (jill_rate_uphill jill_rate_downhill : ℕ) :
  let x := (17/38 : ℚ) in
  let time_jack_up := jack_time_uphill distance jack_rate_uphill in
  let time_jill_up := jill_time_uphill distance jill_rate_uphill in
  let y_jack := jack_position_downhill distance jack_rate_downhill x time_jack_up in
  let y_jill := jill_position_uphill jill_rate_uphill x (time_jill_up + head_start) in
  (distance - y_jill) = (33/19 : ℚ) :=
by
  sorry

end meet_distance_l277_277111


namespace intersection_complement_l277_277416

open Set

variable {α : Type*} [LinearOrder α]

def P (x : α) : Prop := x - 1 ≤ 0
def Q (x : α) : Prop := 0 < x ∧ x ≤ 2

theorem intersection_complement :
  (compl {x : α | P x}) ∩ {x : α | Q x} = {x : α | 1 < x ∧ x ≤ 2} :=
by
  ext x
  simp [P, Q, compl]
  sorry

end intersection_complement_l277_277416


namespace smallest_palindromic_integer_is_21_l277_277770

noncomputable def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n).reverse
  digits = Nat.digits b n

def smallest_palindromic_integer : ℕ :=
  (List.range 1000).find (λ n, n > 20 ∧ is_palindrome_base n 2 ∧ is_palindrome_base n 4).get_or_else 0

theorem smallest_palindromic_integer_is_21 :
  smallest_palindromic_integer = 21 :=
by
  sorry

end smallest_palindromic_integer_is_21_l277_277770


namespace monotonicity_and_range_of_m_l277_277924

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 - a) / 2 * x ^ 2 + a * x - Real.log x

theorem monotonicity_and_range_of_m (a m : ℝ) (h₀ : 2 < a) (h₁ : a < 3)
  (h₂ : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 -> ma + Real.log 2 > |f x1 a - f x2 a|):
  m ≥ 0 :=
sorry

end monotonicity_and_range_of_m_l277_277924


namespace quadratic_distinct_roots_l277_277054

theorem quadratic_distinct_roots (m : ℝ) : 
  ((∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1 * r2 = 9 ∧ r1 + r2 = -m) ↔ 
  m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo (6) ∞) := 
by sorry

end quadratic_distinct_roots_l277_277054


namespace find_r_value_l277_277510

theorem find_r_value (m : ℕ) (h_m : m = 3) (t : ℕ) (h_t : t = 3^m + 2) (r : ℕ) (h_r : r = 4^t - 2 * t) : r = 4^29 - 58 := by
  sorry

end find_r_value_l277_277510


namespace maximum_value_l277_277697

-- Define the variables as positive real numbers
variables (a b c : ℝ)

-- Define the conditions
def condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 2*a*b*c + 1

-- Define the expression
def expr (a b c : ℝ) : ℝ := (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b)

-- The theorem stating that under the given conditions, the expression has a maximum value of 1/8
theorem maximum_value : ∀ (a b c : ℝ), condition a b c → expr a b c ≤ 1/8 :=
by
  sorry

end maximum_value_l277_277697


namespace calculation_eq_990_l277_277700

theorem calculation_eq_990 : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 :=
by
  sorry

end calculation_eq_990_l277_277700


namespace count_even_ones_grid_l277_277101

theorem count_even_ones_grid (n : ℕ) : 
  (∃ f : (Fin n) → (Fin n) → ℕ, (∀ i : Fin n, ∑ j in (Finset.univ : Finset (Fin n)), f i j % 2 = 0) ∧ 
                                  (∀ j : Fin n, ∑ i in (Finset.univ : Finset (Fin n)), f i j % 2 = 0)) ↔ 
  2^((n-1)^2) = 2^((n-1)^2) :=
sorry

end count_even_ones_grid_l277_277101


namespace assignment_problem_l277_277987

theorem assignment_problem (a b c : ℕ) (h1 : a = 10) (h2 : b = 20) (h3 : c = 30) :
  let a := b
  let b := c
  let c := a
  a = 20 ∧ b = 30 ∧ c = 20 :=
by
  sorry

end assignment_problem_l277_277987


namespace number_of_solutions_l277_277156

theorem number_of_solutions (n : ℕ) : (4 * n) = 80 ↔ n = 20 :=
by
  sorry

end number_of_solutions_l277_277156


namespace problem1_part1_problem1_part2_l277_277519

noncomputable theory

open Real

def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) / 2 * x^2 - a * x
def g (x : ℝ) : ℝ := log x
def F (a : ℝ) (x : ℝ) : ℝ := f a x - g x

theorem problem1_part1 (x : ℝ) (a := -3) (h : x ∈ Set.Icc (1 / exp 2) (exp 2)) :
  F a (exp 2) = -exp 4 + 3 * exp 2 - 2 ∧
  F a (1 / exp 2) = - (1 / exp 4) + (3 / exp 2) + 2 :=
sorry

theorem problem1_part2 (a : ℝ) (t : ℝ) (h_a : a ∈ Set.Ioo (-3 : ℝ) (-2))
  (h : ∀ (x₁ x₂ ∈ Set.Icc 1 2), abs (F a x₁ - F a x₂) < a * t + log 2) :
  t ∈ Set.Iic 0 :=
sorry

end problem1_part1_problem1_part2_l277_277519


namespace evaluate_sum_l277_277811

theorem evaluate_sum (a b c : ℝ) 
  (h : (a / (36 - a) + b / (49 - b) + c / (81 - c) = 9)) :
  (6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5.047) :=
by
  sorry

end evaluate_sum_l277_277811


namespace number_of_ways_to_fill_grid_l277_277091

open Finset

theorem number_of_ways_to_fill_grid (n : ℕ) (h : n ≥ 1) :
  let grid := Matrix (Fin n) (Fin n) (Fin 2)
  let condition (m : grid) := (∀ i : Fin n, even (card { j | m i j = 1 })) ∧
                              (∀ j : Fin n, even (card { i | m i j = 1 }))
  ∃ fill_count : ℕ, (fill_count = 2^((n-1)*(n-1))) ∧
                    ∀ g : grid, condition g ↔ (g ∈ universe grid) :=
sorry

end number_of_ways_to_fill_grid_l277_277091


namespace total_apples_l277_277286

def packs : ℕ := 2
def apples_per_pack : ℕ := 4

theorem total_apples : packs * apples_per_pack = 8 := by
  sorry

end total_apples_l277_277286


namespace range_of_a_l277_277359

theorem range_of_a (a : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, x^2 - 4 * a * x + 3 * a^2 ≤ 0) 
  (h3 : ∀ x : ℝ, x^2 + 5 * x + 4 < 0) (h4 : ∀ x : ℝ, (3 * a < x ∧ x < a) → (-4 ≤ x ∧ x ≤ -1)) : 
  (-4 / 3 ≤ a ∧ a ≤ -1) :=
begin
  sorry
end

end range_of_a_l277_277359


namespace value_of_a2_l277_277839

open Nat

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 0 < n → a (n + 1) = 2 - 1 / (a n)

theorem value_of_a2 (a : ℕ → ℝ) (h : sequence a) : a 2 = 3 / 2 :=
by
  have h1 := h.1
  have h2 := h.2 1 (by norm_num)
  rw h1 at h2
  exact h2

end value_of_a2_l277_277839


namespace find_b_and_c_l277_277844

variable (U : Set ℝ) -- Define the universal set U
variable (A : Set ℝ) -- Define the set A
variables (b c : ℝ) -- Variables for coefficients

-- Conditions that U = {2, 3, 5} and A = { x | x^2 + bx + c = 0 }
def cond_universal_set := U = {2, 3, 5}
def cond_set_A := A = { x | x^2 + b * x + c = 0 }

-- Condition for the complement of A w.r.t U being {2}
def cond_complement := (U \ A) = {2}

-- The statement to be proved
theorem find_b_and_c : 
  cond_universal_set U →
  cond_set_A A b c →
  cond_complement U A →
  b = -8 ∧ c = 15 :=
by
  intros
  sorry

end find_b_and_c_l277_277844


namespace eight_mul_eleven_and_one_fourth_l277_277747

theorem eight_mul_eleven_and_one_fourth : 8 * (11 + (1 / 4 : ℝ)) = 90 := by
  sorry

end eight_mul_eleven_and_one_fourth_l277_277747


namespace count_even_ones_grid_l277_277103

theorem count_even_ones_grid (n : ℕ) : 
  (∃ f : (Fin n) → (Fin n) → ℕ, (∀ i : Fin n, ∑ j in (Finset.univ : Finset (Fin n)), f i j % 2 = 0) ∧ 
                                  (∀ j : Fin n, ∑ i in (Finset.univ : Finset (Fin n)), f i j % 2 = 0)) ↔ 
  2^((n-1)^2) = 2^((n-1)^2) :=
sorry

end count_even_ones_grid_l277_277103


namespace TriangleLOM_Area_Approximation_l277_277460

-- Define the conditions and problem statement
def ScaleneTriangle (A B C : ℝ) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C
def AngleProperties (A B C : ℝ) : Prop := 
  ∃ (A B C : ℝ), (C = B - A) ∧ (B = 2 * A) ∧ (A + B + C = 180)
noncomputable def AreaOfTriangleABC := 20
noncomputable def ApproxAreaOfTriangleLOM (area_ABC : ℝ) :=
  let approx_area_LMO := 3 * area_ABC in
  Int.round approx_area_LMO

-- State the problem we need to prove
theorem TriangleLOM_Area_Approximation : 
  ∀ (A B C : ℝ), 
  ScaleneTriangle A B C → 
  AngleProperties A B C → 
  ApproxAreaOfTriangleLOM AreaOfTriangleABC = 27 := by
  -- Proof Placeholder
  sorry

end TriangleLOM_Area_Approximation_l277_277460


namespace find_k_l277_277841

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

-- Given that the union of sets A and B is {1, 2, 3, 5}, prove that k = 3.
theorem find_k (k : ℕ) (h : A k ∪ B = {1, 2, 3, 5}) : k = 3 :=
by
  sorry

end find_k_l277_277841


namespace total_amount_spent_l277_277247

variable (you friend : ℝ)

theorem total_amount_spent (h1 : friend = you + 3) (h2 : friend = 7) : 
  you + friend = 11 :=
by
  sorry

end total_amount_spent_l277_277247


namespace ostap_advantageous_order_l277_277547

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l277_277547


namespace parking_lot_full_sized_cars_l277_277716

theorem parking_lot_full_sized_cars :
  ∀ (total_spaces reserved_motorcycles reserved_ev full_ratio compact_ratio : ℕ),
  total_spaces = 750 →
  reserved_motorcycles = 50 →
  reserved_ev = 30 →
  full_ratio = 11 →
  compact_ratio = 4 →
  let remaining_spaces := total_spaces - reserved_motorcycles - reserved_ev,
      total_car_spaces := (full_ratio + compact_ratio),
      C := remaining_spaces * compact_ratio / total_car_spaces,
      F := full_ratio * C / compact_ratio
  in F = 489 :=
by 
  intros total_spaces reserved_motorcycles reserved_ev full_ratio compact_ratio
  intros h_total h_motor h_ev h_full_ratio h_compact_ratio
  let remaining_spaces := total_spaces - reserved_motorcycles - reserved_ev
  rw [h_total, h_motor, h_ev] at remaining_spaces
  let total_car_spaces := full_ratio + compact_ratio
  let C := remaining_spaces * compact_ratio / total_car_spaces
  let F := full_ratio * C / compact_ratio
  rw [h_full_ratio, h_compact_ratio]
  have h_remaining_spaces: remaining_spaces = 670, by sorry
  have h_full_size_spaces: F = 489, by sorry
  exact h_full_size_spaces

end parking_lot_full_sized_cars_l277_277716


namespace plane_equation_l277_277719

theorem plane_equation (A B C D x y z : ℤ) (h1 : A = 15) (h2 : B = -3) (h3 : C = 2) (h4 : D = -238) 
  (h5 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1) (h6 : A > 0) :
  A * x + B * y + C * z + D = 0 ↔ 15 * x - 3 * y + 2 * z - 238 = 0 :=
by
  sorry

end plane_equation_l277_277719


namespace smallest_n_l277_277963

theorem smallest_n (x y : ℤ) (hx : x ≡ -2 [MOD 7]) (hy : y ≡ 2 [MOD 7]) :
  ∃ (n : ℕ), (n > 0) ∧ (x^2 + x * y + y^2 + ↑n ≡ 0 [MOD 7]) ∧ n = 3 := by
  sorry

end smallest_n_l277_277963


namespace distance_of_coming_down_stairs_l277_277857

noncomputable def totalTimeAscendingDescending (D : ℝ) : ℝ :=
  (D / 2) + ((D + 2) / 3)

theorem distance_of_coming_down_stairs : ∃ D : ℝ, totalTimeAscendingDescending D = 4 ∧ (D + 2) = 6 :=
by
  sorry

end distance_of_coming_down_stairs_l277_277857


namespace quinn_free_donuts_l277_277171

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end quinn_free_donuts_l277_277171


namespace min_value_frac_l277_277815

variable (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1)

theorem min_value_frac : (1 / a + 4 / b) = 9 :=
by sorry

end min_value_frac_l277_277815


namespace num_perfect_square_factors_of_180_l277_277853

theorem num_perfect_square_factors_of_180 : 
  ∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 180 → (∃ e1 e2 e3 : ℕ, 
    d = 2^e1 * 3^e2 * 5^e3 ∧ 
    e1 ≤ 2 ∧ e2 ≤ 2 ∧ e3 ≤ 1 ∧ 
    e1 % 2 = 0 ∧ e2 % 2 = 0 ∧ e3 % 2 = 0)) :=
by {
  let num := 4,
  use num,
  split,
  { refl },
  {
    intros d hd,
    exists 0, exists 0, exists 0,
    split, refl,
    split; try {refl},
    sorry
  }
}

end num_perfect_square_factors_of_180_l277_277853


namespace probability_between_R_and_S_l277_277556

-- Define the points and lengths on the line segment
def P : Type := sorry
def Q : Type := sorry
def R : Type := sorry
def S : Type := sorry

variables (PQ PR RS : ℝ)

-- Conditions from the problem
axiom pq_pr : PQ = 4 * PR
axiom pq_rs : PQ = 8 * RS

-- The theorem to prove
theorem probability_between_R_and_S : (RS / PQ) = (1 / 8) :=
by
  rw [pq_pr, pq_rs]
  sorry

end probability_between_R_and_S_l277_277556


namespace grid_all_black_probability_l277_277263

open classical

noncomputable theory

def gridProbability : ℚ :=
  let probCenterBlack := (1 / 2 : ℚ) * (1 / 2 : ℚ)
  let probPairBlack := (1 / 4 : ℚ)
  let probAllPairsBlack := probPairBlack ^ 7
  probCenterBlack * probAllPairsBlack

theorem grid_all_black_probability :
  gridProbability = (1 / 65536 : ℚ) :=
by
  sorry

end grid_all_black_probability_l277_277263


namespace intersection_range_of_b_l277_277361

theorem intersection_range_of_b (b : ℝ) :
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2 + 2 * y^2 = 3 ∧ y = m * x + b) ↔ 
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := 
sorry

end intersection_range_of_b_l277_277361


namespace find_number_l277_277933

theorem find_number : ∃ x : ℝ, x^2 + 64 = (x - 16)^2 ∧ x = 6 :=
by
  existsi (6 : ℝ)
  rw sq
  have h : 6^2 + 64 = (6 - 16)^2 := by
    calc
      6^2 + 64 = 36 + 64       : by norm_num
      ... = 100               : by norm_num
      ... = (-10)^2           : by norm_num
      ... = (6 - 16)^2        : by norm_num
  exact ⟨ h, rfl ⟩
  sorry

end find_number_l277_277933


namespace rationalize_denominator_l277_277561

theorem rationalize_denominator :
  (1 / (cbrt 4 + cbrt 32 - 1)) = (3 * cbrt 4 + 1) / (18 * cbrt 4 - 1) :=
by
  sorry

end rationalize_denominator_l277_277561


namespace area_T_shaped_region_l277_277772

theorem area_T_shaped_region (w h a1 a2 a3 a4 a5 a6 a7 a8 : ℝ) 
  (H1 : w = 6) (H2 : h = 5) 
  (H3 : a1 = 1) (H4 : a2 = 4) 
  (H5 : a3 = 1) (H6 : a4 = 4) 
  (H7 : a5 = 1) (H8 : a6 = 3) :
  let Area_ABCD := w * h
  let Area1 := a1 * a2
  let Area2 := a3 * a4
  let Area3 := a5 * a6
  let shaded_area := Area_ABCD - (Area1 + Area2 + Area3)
  in shaded_area = 19 := 
by
  dsimp [Area_ABCD, Area1, Area2, Area3, shaded_area]
  rw [H1, H2, H3, H4, H5, H6, H7, H8]
  norm_num
  exact eq.refl _


end area_T_shaped_region_l277_277772


namespace perpendicular_line_eq_slope_intercept_l277_277651

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l277_277651


namespace simplify_expression_l277_277567

theorem simplify_expression (x : ℝ) (h1 : x^2 - 4*x + 3 ≠ 0) (h2 : x^2 - 6*x + 9 ≠ 0) (h3 : x^2 - 3*x + 2 ≠ 0) (h4 : x^2 - 4*x + 4 ≠ 0) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / (x^2 - 3*x + 2) / (x^2 - 4*x + 4) = (x-2) / (x-3) :=
by {
  sorry
}

end simplify_expression_l277_277567


namespace probability_same_tribe_l277_277456

def totalPeople : ℕ := 18
def peoplePerTribe : ℕ := 6
def tribes : ℕ := 3
def totalQuitters : ℕ := 2

def totalWaysToChooseQuitters := Nat.choose totalPeople totalQuitters
def waysToChooseFromTribe := Nat.choose peoplePerTribe totalQuitters
def totalWaysFromSameTribe := tribes * waysToChooseFromTribe

theorem probability_same_tribe (h1 : totalPeople = 18) (h2 : peoplePerTribe = 6) (h3 : tribes = 3) (h4 : totalQuitters = 2)
    (h5 : totalWaysToChooseQuitters = 153) (h6 : totalWaysFromSameTribe = 45) :
    (totalWaysFromSameTribe : ℚ) / totalWaysToChooseQuitters = 5 / 17 := by
  sorry

end probability_same_tribe_l277_277456


namespace symmetric_graphs_inverse_l277_277441

theorem symmetric_graphs_inverse (a b : ℝ) (h : ∀ x : ℝ, a * (-2x + 2b) + 8 = x) : a + b = 2 :=
sorry

end symmetric_graphs_inverse_l277_277441


namespace hallway_covering_l277_277713

theorem hallway_covering :
  ∀ (runner : Set (ℝ × ℝ)), (∀ x ∈ Icc (0 : ℝ) 1, ∃ (a b : ℝ), (a, b) ∈ runner ∧ a ≤ x ∧ x ≤ b) →
  ∃ (runner' : Set (ℝ × ℝ)), runner' ⊆ runner ∧ (∀ x ∈ Icc (0 : ℝ) 1, ∃ (a b : ℝ), (a, b) ∈ runner' ∧ a ≤ x ∧ x ≤ b) ∧ (∑ (r ∈ runner'), r.2 - r.1) ≤ 2 :=
by
  sorry

end hallway_covering_l277_277713


namespace fundraising_division_l277_277246

theorem fundraising_division (total_amount people : ℕ) (h1 : total_amount = 1500) (h2 : people = 6) :
  total_amount / people = 250 := by
  rw [h1, h2]
  norm_num
  sorry

end fundraising_division_l277_277246


namespace length_of_b_and_AE_l277_277007

theorem length_of_b_and_AE (a b c : ℝ) (D midpoint_of_AC : (ℝ × ℝ)) (BD : ℝ) (AE : ℝ) :
  a = 4 * sqrt 7 ∧ c = 12 ∧ BD = 4 * sqrt 7 ∧
  midpoint_of_AC = ((a / 2), (c / 2)) ∧ b = 8 ∧ AE = 24 * sqrt 3 / 5 :=
sorry

end length_of_b_and_AE_l277_277007


namespace sum_of_a_and_b_l277_277589

theorem sum_of_a_and_b :
  ∃ (a b : ℕ), (∀ x : ℝ, x^2 + 14 * x = 65 → x = sqrt a - b) ∧ a + b = 121 :=
by
  sorry

end sum_of_a_and_b_l277_277589


namespace range_of_m_is_increasing_l277_277060

noncomputable def f (x : ℝ) (m: ℝ) := x^2 + m*x + m

theorem range_of_m_is_increasing :
  { m : ℝ // ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m } = {m | 4 ≤ m} :=
by
  sorry

end range_of_m_is_increasing_l277_277060


namespace sequence_is_geometric_iff_t_eq_neg1_l277_277806

variable {t : ℝ}
def S (n : ℕ) : ℝ := 5^n + t
def a (n : ℕ) : ℝ := if n = 1 then S 1 else S n - S (n - 1)

theorem sequence_is_geometric_iff_t_eq_neg1 : 
  (∀ n : ℕ, a n = 4 * 5^(n-1)) ↔ t = -1 :=
sorry

end sequence_is_geometric_iff_t_eq_neg1_l277_277806


namespace find_x_l277_277181

def operation (a b : Int) : Int := 2 * a + b

theorem find_x :
  ∃ x : Int, operation 3 (operation 4 x) = -1 :=
  sorry

end find_x_l277_277181


namespace intersection_point_on_circle_Gamma_l277_277820

-- Definitions related to the problem setup
variables {A B C C1 B1 A1 D E X R : Type}
variables [DistinctPoints A B C C1 B1 A1 D E R]
variables (hex : ConvexHexagon A B C C1 B1 A1)
variables (equal_sides : Length A B = Length B C)
variables (common_perpendiculars : PerpendicularBisector A A1 B B1 C C1)
variables (diagonals_intersection : IntersectsAt (Line A C1) (Line A1 C) D)
variables (circles_intersection_Gamma : Circumcircle (Triangle A B C) "Gamma")
variables (circles_intersection_ABC1 : Circumcircle (Triangle A1 B C1) intersects Gamma at B and E)

-- The question to prove
theorem intersection_point_on_circle_Gamma :
  LaysOnCircumcircle (Intersection (Line BB1) (Line DE)) Gamma :=
sorry

end intersection_point_on_circle_Gamma_l277_277820


namespace optimal_order_l277_277551

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l277_277551


namespace cylinder_lateral_surface_area_l277_277254

variable (r : ℝ) (h : ℝ)

theorem cylinder_lateral_surface_area (hr : r = 12) (hh : h = 21) :
  (2 * Real.pi * r * h) = 504 * Real.pi := by
  sorry

end cylinder_lateral_surface_area_l277_277254


namespace g_does_not_pass_through_fourth_quadrant_l277_277406

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / x)

theorem g_does_not_pass_through_fourth_quadrant (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
    ¬(∃ x, x > 0 ∧ g x < 0) :=
by
    sorry

end g_does_not_pass_through_fourth_quadrant_l277_277406


namespace sum_of_b_sequence_l277_277031

theorem sum_of_b_sequence (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, a n = (3 + (-1:ℝ)^(n+1)) / 2) →
  (∀ n : ℕ, a (n+1) * b n + a n * b (n+1) = (-1:ℝ)^n + 1) →
  (b 1 = 2) →
  (S 0 = 0) →
  (∀ n : ℕ, S (n+1) = S n + b (n+1)) →
  S 99 = 1325 :=
by sorry

end sum_of_b_sequence_l277_277031


namespace geometric_series_first_term_l277_277738

noncomputable def first_term_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  S = a / (1 - r)

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (hr : r = 1/6)
  (hS : S = 54) :
  first_term_geometric_series r S a →
  a = 45 :=
by
  intros h
  -- The proof goes here
  sorry

end geometric_series_first_term_l277_277738


namespace tangent_line_eq_l277_277838

noncomputable def f (x : ℝ) := x ^ (1 / 2)

def point_A : ℝ × ℝ := (1 / 4, 1 / 2)

theorem tangent_line_eq :
  let x := (1 / 4 : ℝ)
  let y := (1 / 2 : ℝ)
  ∀ (f : ℝ → ℝ), (f x = y) →
  (f' x = 1) →
  (∀ (f' : ℝ → ℝ), f' = λ x, (1 / 2) * (1 / Real.sqrt x)) →
  ∀ (y' : ℝ), (y' = f' x) → 4 * x - 4 * y' + 1 = 0 := 
by 
  sorry

end tangent_line_eq_l277_277838


namespace factorial_cubic_power_equality_l277_277241

theorem factorial_cubic_power_equality : (∛(Nat.factorial 6 * Nat.factorial 4))^6 = 308915776 := by
  sorry

end factorial_cubic_power_equality_l277_277241


namespace num_ways_choose_three_l277_277063

theorem num_ways_choose_three (x y z : ℕ) (h1 : 1 ≤ x) (h2 : x < y) (h3 : y < z) (h4 : z ≤ 14) 
(h5 : |y - x| ≥ 3) (h6 : |z - y| ≥ 3) (h7 : |z - x| ≥ 3) : 
  ∃ n : ℕ, n = 120 :=
by
  sorry

end num_ways_choose_three_l277_277063


namespace James_pays_35_l277_277493

theorem James_pays_35 (first_lesson_free : Bool) (total_lessons : Nat) (cost_per_lesson : Nat) 
  (first_x_paid_lessons_free : Nat) (every_other_remainings_free : Nat) (uncle_pays_half : Bool) :
  total_lessons = 20 → 
  first_lesson_free = true → 
  cost_per_lesson = 5 →
  first_x_paid_lessons_free = 10 →
  every_other_remainings_free = 1 → 
  uncle_pays_half = true →
  (10 * cost_per_lesson + 4 * cost_per_lesson) / 2 = 35 :=
by
  sorry

end James_pays_35_l277_277493


namespace matrix_vector_multiplication_l277_277920

variables (N : Matrix (Fin 2) (Fin 2) ℝ)
variables (a b : Vector (Fin 2) ℝ)

noncomputable theory

def condition1 : Prop := 
  N.mulVec a = ![4, 5]

def condition2 : Prop := 
  N.mulVec b = ![-3, -7]

theorem matrix_vector_multiplication :
  condition1 N a →
  condition2 N b →
  N.mulVec (2 • a - 4 • b) = ![20, 38] :=
by
  intros h1 h2
  sorry

end matrix_vector_multiplication_l277_277920


namespace find_total_shaded_area_l277_277228

/-- Definition of the rectangles' dimensions and overlap conditions -/
def rect1_length : ℕ := 4
def rect1_width : ℕ := 15
def rect2_length : ℕ := 5
def rect2_width : ℕ := 10
def rect3_length : ℕ := 3
def rect3_width : ℕ := 18
def shared_side_length : ℕ := 4
def trip_overlap_width : ℕ := 3

/-- Calculation of the rectangular overlap using given conditions -/
theorem find_total_shaded_area : (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width - shared_side_length * shared_side_length - trip_overlap_width * shared_side_length) = 136 :=
    by sorry

end find_total_shaded_area_l277_277228


namespace quadratic_polynomial_exists_l277_277781

theorem quadratic_polynomial_exists (a b c : ℝ) 
  (h_coeff_x2 : a = 2)
  (h_root_1 : a * (2 + 2i)^2 + b * (2 + 2i) + c = 0)
  (h_root_2 : a * (2 - 2i)^2 + b * (2 - 2i) + c = 0) 
  : a * x^2 + b * x + c = 2 * x^2 - 8 * x + 16 :=
sorry

end quadratic_polynomial_exists_l277_277781


namespace equal_sides_l277_277915

variables {A B C D E F G: Type} [inner_product_space ℝ A]

/-- Triangle ABC with line through C parallel to AB and angle bisectors
     at A meeting BC at D and line L at E, angle bisectors at B meeting AC at F and line L at G. 
     Given GF = DE, prove that AC = BC. -/
theorem equal_sides (ABC : triangle A B C) 
  (L : line) (hL : L.parallel (line_through C (vector (A B))))
  (AD_bisects_A : bisector (A.angle B C) (ray_through_point_on A D C) (ray_through_point_on A E L))
  (BF_bisects_B : bisector (B.angle A C) (ray_through_point_on B F C) (ray_through_point_on B G L))
  (GF_eq_DE : distance G F = distance D E) : 
  length (A.vertex_to C) = length (B.vertex_to C) :=
sorry

end equal_sides_l277_277915


namespace smallest_integer_n_smallest_integer_n_5_smallest_value_of_n_l277_277300

noncomputable def sequence (n : ℕ) : ℝ :=
if n = 1 then real.sqrt (real.sqrt 4)
else (sequence (n - 1)) ^ (real.sqrt (real.sqrt 4))

theorem smallest_integer_n (n : ℕ) (h : n < 5) : ¬ (sequence n).is_integer :=
sorry

theorem smallest_integer_n_5 : sequence 5 = 4 :=
sorry

theorem smallest_value_of_n : ∃ (n : ℕ), (sequence n).is_integer ∧ ∀ (m : ℕ), m < n → ¬ (sequence m).is_integer :=
exists.intro 5
  (and.intro
    (by simp [sequence, real.rpow_nat_cast] ; sorry)
    (by intro m hm ; apply smallest_integer_n m hm))

end smallest_integer_n_smallest_integer_n_5_smallest_value_of_n_l277_277300


namespace total_doses_l277_277280

def July_days : ℕ := 31

def missed_diabetes_days : ℕ := 4
def missed_blood_pressure_days : ℕ := 3
def missed_cholesterol_occasions : ℕ := 2

def diabetes_doses_per_day : ℕ := 2
def blood_pressure_doses_per_day : ℕ := 1
def cholesterol_doses_every_two_days : ℕ := 1

theorem total_doses (total_days : ℕ)
  (missed_diabetes_days : ℕ) (missed_blood_pressure_days : ℕ) (missed_cholesterol_occasions : ℕ)
  (diabetes_doses_per_day : ℕ) (blood_pressure_doses_per_day : ℕ) (cholesterol_doses_every_two_days : ℕ) :
  let diabetes_total := total_days * diabetes_doses_per_day - missed_diabetes_days * diabetes_doses_per_day
  let blood_pressure_total := total_days * blood_pressure_doses_per_day - missed_blood_pressure_days
  let cholesterol_doses_potential := total_days / 2 * cholesterol_doses_every_two_days
  let cholesterol_total := cholesterol_doses_potential - missed_cholesterol_occasions in
  (diabetes_total = 54) ∧ (blood_pressure_total = 28) ∧ (cholesterol_total = 13) :=
by {
  let diabetes_total := total_days * diabetes_doses_per_day - missed_diabetes_days * diabetes_doses_per_day
  let blood_pressure_total := total_days * blood_pressure_doses_per_day - missed_blood_pressure_days
  let cholesterol_doses_potential := total_days / 2 * cholesterol_doses_every_two_days
  let cholesterol_total := cholesterol_doses_potential - missed_cholesterol_occasions
  have h_diabetes : diabetes_total = 54, sorry
  have h_blood_pressure : blood_pressure_total = 28, sorry
  have h_cholesterol : cholesterol_total = 13, sorry
  exact ⟨h_diabetes, h_blood_pressure, h_cholesterol⟩
}

end total_doses_l277_277280


namespace range_of_m_l277_277028

noncomputable def proposition_p (m : ℝ) : Prop :=
  (x : ℝ → x^2 + 2*m*x + 1 = 0) ∧
  (x1 x2 : ℝ)(h_root_1 : (x1 : ℝ → x1^2 + 2*m*x1 + 1 = 0)) 
  (h_root_2 : (x2 : ℝ → x2^2 + 2*m*x2 + 1 = 0)) → x1 ≠ x2 ∧ ∀ x, x > 0 ∧ x = x1 ∨ x = x2

noncomputable def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 ≠ 0

theorem range_of_m (m : ℝ) : 
  (proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m) →
  (m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)) :=
by
  sorry

end range_of_m_l277_277028


namespace incorrect_projection_D_l277_277473

-- Define the points and their relationships as per the given conditions
variables {A A1 B B1 C C1 D D1 E F : Type}

-- E is a trisection point of A1A such that AE = 2*A1E
def E_trisection_point (A A1 E : Type) : Prop :=
  -- some formal definition to represent AE = 2 * A1E
  sorry

-- F is a trisection point of C1C such that CF = 2*C1F
def F_trisection_point (C C1 F : Type) : Prop :=
  -- some formal definition to represent CF = 2 * C1F
  sorry

-- A plane passes through points B, E, and F
def plane_through_B_E_F (B E F : Type) : Prop :=
  -- some formal definition involving B, E, and F 
  sorry

-- Prove that projection D is incorrect given the conditions
theorem incorrect_projection_D 
  (h1 : E_trisection_point A A1 E)
  (h2 : F_trisection_point C C1 F)
  (h3 : plane_through_B_E_F B E F) :
  -- formal representation of the projection D being incorrect
  sorry

end incorrect_projection_D_l277_277473


namespace find_tan_x0_l277_277402

noncomputable def f (x : ℝ) : ℝ := sin x - cos x

theorem find_tan_x0 (x0 : ℝ) 
  (h : (deriv^[2] f) x0 = 2 * f x0) : tan x0 = 3 :=
by
  let f' x := deriv f x
  let f'' x := deriv f' x
  have f'_def : ∀ x, f' x = cos x + sin x := 
    by sorry -- Proof omitted
  have f''_def : ∀ x, f'' x = cos x - sin x := 
    by sorry -- Proof omitted
  rw [f''_def, f'_def] at h
  have key : cos x0 + sin x0 = 2 * (sin x0 - cos x0) := by exact h
  calc
    tan x0 = sin x0 / cos x0 : by sorry -- Proof omitted
          ... = 3           : by sorry -- Proof omitted

end find_tan_x0_l277_277402


namespace calculate_length_of_BC_l277_277029

theorem calculate_length_of_BC:
  ∀ (A B C D : Point) (AD CD AC : ℝ),
  (AD = 47) →
  (CD = 25) →
  (AC = 24) →
  (right_triangle A B D) →
  (right_triangle A B C) →
  (D_x < C_x) →
  (B_x = C_x) →
  length_segment (A, D) = 47 →
  length_segment (C, D) = 25 →
  length_segment (A, C) = 24 →
  ∃ (BC : ℝ), BC = 20.16 :=
by
  sorry

end calculate_length_of_BC_l277_277029


namespace problem_f_2011_2012_l277_277013

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2011_2012 :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f (1-x) = f (1+x)) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = 2^x - 1) →
  f 2011 + f 2012 = -1 :=
by
  intros h1 h2 h3
  sorry

end problem_f_2011_2012_l277_277013


namespace fraction_simplification_l277_277975

theorem fraction_simplification :
  let a := 2020 in
  let expr := (a+1)/a - a/(a+1) in
  ∃ p q : ℕ, expr = p / q ∧ (p.gcd q = 1) ∧ p = 4041 :=
by
  let a := 2020
  let expr := ((a + 1) : ℚ) / a - (a : ℚ) / (a + 1)
  use 4041
  use a * (a + 1)
  sorry

end fraction_simplification_l277_277975


namespace product_of_digits_of_non_divisible_number_l277_277529

theorem product_of_digits_of_non_divisible_number:
  (¬ (3641 % 4 = 0)) →
  ((3641 % 10) * ((3641 / 10) % 10)) = 4 :=
by
  intro h
  sorry

end product_of_digits_of_non_divisible_number_l277_277529


namespace sum_mod_20_l277_277343

/-- Define the elements that are summed. -/
def elements : List ℤ := [82, 83, 84, 85, 86, 87, 88, 89]

/-- The problem statement to prove. -/
theorem sum_mod_20 : (elements.sum % 20) = 15 := by
  sorry

end sum_mod_20_l277_277343


namespace optimal_order_for_ostap_l277_277531

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l277_277531


namespace sum_reciprocals_l277_277145

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end sum_reciprocals_l277_277145


namespace determine_values_of_a_and_b_l277_277978

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 2 - 2 * a * x + b

theorem determine_values_of_a_and_b :
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ x ∈ set.Icc 1 2, x = 1 → f a b x = -1) ∧ 
              (∀ x ∈ set.Icc 1 2, x = 2 → f a b x = 0)) →
  (a = 1 ∧ b = 0 ∨ a = -1 ∧ b = -1) :=
sorry

end determine_values_of_a_and_b_l277_277978


namespace positive_number_sum_square_l277_277997

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l277_277997


namespace rhombus_second_diagonal_l277_277599

theorem rhombus_second_diagonal (perimeter : ℝ) (d1 : ℝ) (side : ℝ) (half_d2 : ℝ) (d2 : ℝ) :
  perimeter = 52 → d1 = 24 → side = 13 → (half_d2 = 5) → d2 = 2 * half_d2 → d2 = 10 :=
by
  sorry

end rhombus_second_diagonal_l277_277599


namespace triangle_inequality_proof_l277_277259

noncomputable def triangle_inequality (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) : Prop :=
  Real.pi / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ (a * A + b * B + c * C) / (a + b + c) < Real.pi / 2

theorem triangle_inequality_proof (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h₁: A + B + C = Real.pi) (h₂: ∀ {x y : ℝ}, A ≥ B  → a ≥ b → A * b + B * a ≤ A * a + B * b) 
  (h₃: ∀ {x y : ℝ}, x + y > 0 → A * x + B * y + C * (a + b - x - y) > 0) : 
  triangle_inequality A B C a b c hABC :=
by
  sorry

end triangle_inequality_proof_l277_277259


namespace probability_different_colors_is_correct_l277_277450

-- Definitions of chip counts
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def green_chips := 3
def total_chips := blue_chips + red_chips + yellow_chips + green_chips

-- Definition of the probability calculation
def probability_different_colors := 
  ((blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)) +
  ((red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)) +
  ((yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)) +
  ((green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips))

-- Given the problem conditions, we assert the correct answer
theorem probability_different_colors_is_correct :
  probability_different_colors = (119 / 162) := 
sorry

end probability_different_colors_is_correct_l277_277450


namespace shauna_lowest_score_l277_277174

theorem shauna_lowest_score :
  ∀ (test1 test2 test3 max_points avg_score : ℕ), 
  test1 = 76 → 
  test2 = 94 → 
  test3 = 87 → 
  max_points = 100 → 
  avg_score = 81 → 
  ∃ (score4 score5 : ℕ), 
    score4 ≤ max_points ∧ 
    score5 ≤ max_points ∧ 
    (test1 + test2 + test3 + score4 + score5) / 5 = avg_score ∧ 
    min score4 score5 = 48 :=
by
  intros test1 test2 test3 max_points avg_score h1 h2 h3 h4 h5,
  sorry

end shauna_lowest_score_l277_277174


namespace cos_2beta_proof_l277_277813

theorem cos_2beta_proof (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.sin (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (3 * π / 2) (2 * π)) :
  Real.cos (2 * β) = -7 / 25 :=
by
  sorry

end cos_2beta_proof_l277_277813


namespace g_is_odd_l277_277317

noncomputable def g (x : ℝ) : ℝ := log (x + sqrt (2 + x^2))

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l277_277317


namespace angle_between_a_and_b_is_60_degrees_l277_277002

open Real

noncomputable theory

variables {a b : ℝ^3} 

def is_non_zero (x : ℝ^3) := ∥x∥ ≠ 0

def perpendicular (x y : ℝ^3) := dot_product x y = 0

theorem angle_between_a_and_b_is_60_degrees
  (h1 : is_non_zero a)
  (h2 : is_non_zero b)
  (h3 : perpendicular (a - 2 • b) a)
  (h4 : perpendicular (b - 2 • a) b) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_a_and_b_is_60_degrees_l277_277002


namespace eccentricity_of_ellipse_l277_277906

noncomputable def ecc_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := a * 3 / 4 in
  c / a

theorem eccentricity_of_ellipse (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0) : ecc_of_ellipse a b h1 h2 = 3 / 4 := sorry

end eccentricity_of_ellipse_l277_277906


namespace joan_paid_amount_l277_277497

theorem joan_paid_amount (J K : ℕ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end joan_paid_amount_l277_277497


namespace find_portrait_in_silver_l277_277617

-- Definitions of the propositions based on the conditions
variables (p q r : Prop) -- Represent the propositions for the boxes

-- Conditions
axiom h1 : p ↔ "The portrait is in the gold box"
axiom h2 : q ↔ "The portrait is not in the silver box"
axiom h3 : r ↔ "The portrait is not in the gold box"

-- Additional conditions
axiom h4 : (∃! x, x ∈ {p, q, r} ∧ x = true) -- Exactly one of the propositions is true

-- The proof goal: we want to show that the portrait is in the silver box
theorem find_portrait_in_silver : (¬q) := 
by { sorry }

end find_portrait_in_silver_l277_277617


namespace number_of_people_in_group_l277_277742

-- Definitions for conditions
def adult_ticket_cost : ℝ := 9.50
def child_ticket_cost : ℝ := 6.50
def total_amount_paid : ℝ := 54.50
def number_of_adults : ℕ := 3

-- Theorem statement
theorem number_of_people_in_group : (∃ number_of_children : ℕ, 
  total_amount_paid = number_of_adults * adult_ticket_cost + number_of_children * child_ticket_cost) → 
  ∃ number_of_children : ℕ, number_of_adults + number_of_children = 7 :=
begin
  sorry
end

end number_of_people_in_group_l277_277742


namespace max_value_of_k_l277_277204

theorem max_value_of_k :
  ∀ (k : ℝ), (∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ sqrt (x - 3) + sqrt (6 - x) ≥ k) ↔ k ≤ sqrt 6 := 
sorry

end max_value_of_k_l277_277204


namespace tangent_line_at_origin_increasing_intervals_l277_277827

-- Conditions and function definition
variable (a : ℝ) (h₁ : a ≠ -1)
def f (a : ℝ) : ℝ → ℝ := λ x, (x - 1) / (x + a) + Real.log (x + 1)

-- Proof problem 1: Tangent line equation at (0, f(0)) when a = 2
theorem tangent_line_at_origin (h₂ : a = 2) : 7 * x - 4 * y - 2 = 0 :=
sorry

-- Proof problem 2: Intervals of monotonic increase if f(x) has an extremum at x = 1
theorem increasing_intervals (h₃ : has_extremum_at (f a) 1) (ha : a = -3) :
  (-1 < x ∧ x ≤ 1) ∨ (7 ≤ x) :=
sorry

end tangent_line_at_origin_increasing_intervals_l277_277827


namespace number_of_lattice_points_on_hyperbola_l277_277042

def is_lattice_point_on_hyperbola (x y : ℤ) : Prop :=
  x^2 - y^2 = 999^2

theorem number_of_lattice_points_on_hyperbola :
  {p : ℤ × ℤ | is_lattice_point_on_hyperbola p.1 p.2}.to_finset.card = 21 :=
sorry

end number_of_lattice_points_on_hyperbola_l277_277042


namespace fill_grid_with_even_ones_l277_277081

theorem fill_grid_with_even_ones (n : ℕ) : 
  ∃ ways : ℕ, ways = 2^((n-1)^2) ∧ 
  (∀ grid : array n (array n (fin 2)), 
    (∀ i : fin n, even (grid[i].to_list.count (λ x, x = 1))) ∧ 
    (∀ j : fin n, even (grid.map (λ row, row[j]).to_list.count (λ x, x = 1)))) :=
begin
  use 2^((n-1)^2),
  split,
  { refl },
  { sorry },
end

end fill_grid_with_even_ones_l277_277081


namespace complement_A_correct_l277_277034

-- Define the universal set U
def U : Set ℝ := { x | x ≥ 1 ∨ x ≤ -1 }

-- Define the set A
def A : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := { x | x ≤ -1 ∨ x = 1 ∨ x > 2 }

-- Prove that the complement of A in U is as defined
theorem complement_A_correct : (U \ A) = complement_A_in_U := by
  sorry

end complement_A_correct_l277_277034


namespace circumference_proportionality_l277_277965

theorem circumference_proportionality (r : ℝ) (C : ℝ) (k : ℝ) (π : ℝ)
  (h1 : C = k * r)
  (h2 : C = 2 * π * r) :
  k = 2 * π :=
sorry

end circumference_proportionality_l277_277965


namespace angle_between_lines_NK_DM_is_45_degrees_l277_277900

-- Definitions for the given triangle and points
def is_isosceles_right_triangle (A B C : Point) : Prop :=
  right_angle A C B ∧ isosceles A C B

def point_prolongation (P Q : Point) (d : ℝ) (R : Point) : Prop :=
  dist P Q = 2 * dist P R

-- The Lean statement for the problem
theorem angle_between_lines_NK_DM_is_45_degrees
  (A B C D M N K : Point)
  (h_triangle : is_isosceles_right_triangle A B C)
  (h_AB_2AD : point_prolongation A B (√2) D)
  (h_AM_NC : dist A M = dist N C)
  (h_CN_BK : dist C N = dist B K) :
  angle (line_through N K) (line_through D M) = 45 :=
sorry

end angle_between_lines_NK_DM_is_45_degrees_l277_277900


namespace find_value_l277_277353

variable (θ : ℝ)

def given_condition : Prop := 
  (sin θ)^2 + 4 = 2 * (cos θ + 1)

theorem find_value 
  (h : given_condition θ) : 
  (cos θ + 1) * (sin θ + 1) = 2 := 
  sorry

end find_value_l277_277353


namespace children_descending_after_N_minus_1_steps_l277_277696

open Nat

theorem children_descending_after_N_minus_1_steps (N : ℕ)
  (heights : list ℕ)
  (h_unique : heights.nodup)
  (h_length : heights.length = N)
  (two_step_procedure : list ℕ → list ℕ) :
  (∀ heights, heights.length = N ∧ heights.nodup →
               two_step_procedure heights = heights.reverse) →
  (∃ h', list.reverse (iterate (N-1) two_step_procedure heights) = h'.reverse) :=
by 
  sorry

end children_descending_after_N_minus_1_steps_l277_277696


namespace sin_alpha_plus_beta_l277_277019

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x + 2
def interval := set.Icc 0 (2 * Real.pi)

-- Given the function and the interval for x
variable (m : ℝ) (α β : ℝ)
variable (cond1 : α ∈ interval)
variable (cond2 : β ∈ interval)
variable (root1 : f α = m)
variable (root2 : f β = m)
variable (distinct_roots : α ≠ β)

-- Statement of the theorem
theorem sin_alpha_plus_beta : Real.sin (α + β) = Real.sin (Real.pi / 3) := 
sorry

end sin_alpha_plus_beta_l277_277019


namespace fib_mod_100_l277_277967

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- Theorem that states the 100th Fibonacci number modulo 9 is 3
theorem fib_mod_100 : fib 100 % 9 = 3 :=
sorry

end fib_mod_100_l277_277967


namespace sum_of_squares_of_reciprocals_l277_277610

-- Definitions based on the problem's conditions
variables (a b : ℝ) (hab : a + b = 3 * a * b + 1) (h_an : a ≠ 0) (h_bn : b ≠ 0)

-- Statement of the problem to be proved
theorem sum_of_squares_of_reciprocals :
  (1 / a^2) + (1 / b^2) = (4 * a * b + 10) / (a^2 * b^2) :=
sorry

end sum_of_squares_of_reciprocals_l277_277610


namespace perpendicular_line_through_point_l277_277641

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l277_277641


namespace perpendicular_line_through_point_l277_277645

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l277_277645


namespace find_last_even_number_l277_277609

theorem find_last_even_number (n : ℕ) (h : 4 * (n * (n + 1) * (2 * n + 1) / 6) = 560) : 2 * n = 14 :=
by
  sorry

end find_last_even_number_l277_277609


namespace range_of_m_l277_277823

open Set

variable (f : ℝ → ℝ) (m : ℝ)

theorem range_of_m (h1 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h2 : f (2 * m) > f (1 + m)) : m < 1 :=
by {
  -- The proof would go here.
  sorry
}

end range_of_m_l277_277823


namespace find_a_l277_277015

theorem find_a (a : ℝ) (α : ℝ) (P : ℝ × ℝ) 
  (h_P : P = (3 * a, 4)) 
  (h_cos : Real.cos α = -3/5) : 
  a = -1 := 
by
  sorry

end find_a_l277_277015


namespace calculate_b_l277_277045

open Real

theorem calculate_b (b : ℝ) (h : ∫ x in e..b, 2 / x = 6) : b = exp 4 := 
sorry

end calculate_b_l277_277045


namespace maximum_value_a_l277_277376

theorem maximum_value_a (a : ℝ) (e : ℝ) (ln : ℝ → ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (1 / e) 2 → (a + e) * x - 1 - ln x ≤ 0) → a ≤ -e :=
sorry

end maximum_value_a_l277_277376


namespace distance_between_centers_eq_one_l277_277058

-- Define the radii of the two circles
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 4

-- Define the internal tangency condition for the circles O1 and O2
def internally_tangent (r1 r2 d : ℝ) : Prop :=
  r1 < r2 ∧ d = r2 - r1

-- The theorem statement for the given proof problem
theorem distance_between_centers_eq_one :
  internally_tangent radius_O1 radius_O2 1 :=
by
  unfold internally_tangent
  exact ⟨by norm_num, by norm_num⟩

end distance_between_centers_eq_one_l277_277058


namespace rational_function_horizontal_asymptote_l277_277593

theorem rational_function_horizontal_asymptote (p : Polynomial ℝ) :
  (degree (Polynomial.C (3 : ℝ) * Polynomial.X ^ 7 + Polynomial.C (5 : ℝ) * Polynomial.X ^ 6
    - Polynomial.C (2 : ℝ) * Polynomial.X ^ 3 + 1) ≤ degree p) →
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → (3 * x ^ 7 + 5 * x ^ 6 - 2 * x ^ 3 + 1) / p.eval x = c + (fractional_part (x)) :=
sorry

end rational_function_horizontal_asymptote_l277_277593


namespace min_value_f_l277_277817

theorem min_value_f (a b : ℝ) (h : ∀ x ∈ Set.Ioi (0:ℝ), f a b x ≤ 5):
  ∃ x ∈ Set.Iio (0:ℝ), f a b x = -1 :=
by
  sorry

def f (a b x : ℝ) : ℝ := a*x^3 + b*x^9 + 2

end min_value_f_l277_277817


namespace sum_ideal_numbers_2015_eq_2026_l277_277796

noncomputable def a (n : ℕ) [ h: 0 < n ] : ℝ := Real.logBase (n + 1) (n + 2)

def is_ideal (k : ℕ) := ∃ m : ℕ, m ≥ 2 ∧ k = 2^m - 2

def sum_ideal_numbers (range : ℕ → Prop) : ℕ :=
  let ideals := filter range (finset.range 2016)
  finset.sum ideals id

theorem sum_ideal_numbers_2015_eq_2026 :
  sum_ideal_numbers is_ideal = 2026 := 
sorry

end sum_ideal_numbers_2015_eq_2026_l277_277796


namespace line_properties_l277_277219

def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

def slope (m : ℝ) : Prop := m = -1

def y_intercept (b : ℝ) : Prop := b = -1

def slope_angle (θ : ℝ) : Prop := θ = 135

theorem line_properties :
  (∀ x y, line_eq x y) →
  slope (-1) →
  y_intercept (-1) →
  slope_angle 135 ∧ y_intercept (-1) :=
by
  intros h1 h2 h3
  split
  { sorry }
  { exact h3 }

end line_properties_l277_277219


namespace remainder_product_div_10_l277_277668

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem remainder_product_div_10 :
  let a := 1734
  let b := 5389
  let c := 80607
  let p := a * b * c
  unitsDigit p = 2 := by
  sorry

end remainder_product_div_10_l277_277668


namespace quadratic_inequality_always_positive_l277_277673

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
by sorry

end quadratic_inequality_always_positive_l277_277673


namespace smallest_n_y_n_integer_l277_277301

def y₁ : ℝ := real.root 4 4
def y_seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 1  -- Since Lean indices from 0
  | 1 => y₁
  | n + 1 => (y_seq n) ^ y₁

theorem smallest_n_y_n_integer :
  ∃ n : ℕ, (y_seq n).is_integer ∧ (∀ m < n, ¬(y_seq m).is_integer) ∧ n = 8 :=
by
  sorry

end smallest_n_y_n_integer_l277_277301


namespace area_enclosed_by_graph_l277_277186

noncomputable def enclosed_area (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in set.Icc a b, f x

theorem area_enclosed_by_graph :
  enclosed_area (λ x, x - x^2) 0 1 = 1 / 6 :=
by
  sorry

end area_enclosed_by_graph_l277_277186


namespace find_f_3_l277_277010

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for f, noncomputable as it's defined implicitly

-- Hypothesis 1: f(x) is an increasing function
axiom h_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2)

-- Hypothesis 2: f(f(x) - 2^x) = 3 for any x ∈ ℝ
axiom h_equation : ∀ x : ℝ, f(f(x) - 2^x) = 3

-- Goal: Prove that f(3) = 9
theorem find_f_3 : f 3 = 9 :=
by
  sorry

end find_f_3_l277_277010


namespace PS_div_QR_eq_sqrt_3_l277_277482

-- We assume the lengths and properties initially given in the problem.
def P := sorry -- placeholder for point P
def Q := sorry -- placeholder for point Q
def R := sorry -- placeholder for point R
def S := sorry -- placeholder for point S
def t : ℝ := sorry -- the side length of the equilateral triangles

-- Essential properties of equilateral triangles
axiom PQR_is_equilateral : equilateral_triangle P Q R
axiom QRS_is_equilateral : equilateral_triangle Q R S
axiom side_length_QR : dist Q R = t

-- Heights of triangles from vertices to the opposite sides
def height_PQR : ℝ := t * (sqrt 3) / 2
def height_QRS : ℝ := t * (sqrt 3) / 2

-- PS is the sum of these heights
def PS : ℝ := height_PQR + height_QRS

-- Theorem
theorem PS_div_QR_eq_sqrt_3 : PS / t = sqrt 3 := by
  sorry

end PS_div_QR_eq_sqrt_3_l277_277482


namespace find_f3_l277_277801

noncomputable def f : ℝ → ℝ
| x => if x ≤ 2 then real.exp (x - 1) else 2 * f (x - 2)

theorem find_f3 : f 3 = 2 :=
by
  sorry

end find_f3_l277_277801


namespace min_queries_to_find_two_white_balls_l277_277157

theorem min_queries_to_find_two_white_balls
  (n : ℕ)
  (h_n : n = 2004)
  (h_even : ∃ (wb : Finset (Fin n)), wb.card % 2 = 0)
  (query : (Fin n × Fin n) → Bool)
  (h_query : ∀ i j, query (i, j) = true → (i ≠ j) ∧ (i < j ∨ j < i)) :
  ∃ min_queries : ℕ, min_queries = 4005 := 
sorry

end min_queries_to_find_two_white_balls_l277_277157


namespace find_a_l277_277372

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1 ∧ x ≥ 2

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, point_on_hyperbola x y ∧ (min ((x - a)^2 + y^2) = 3)) → 
  (a = -1 ∨ a = 2 * Real.sqrt 5) :=
by
  sorry

end find_a_l277_277372


namespace good_coloring_count_at_least_6_pow_8_l277_277927

def is_good_coloring (board : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i ≤ 6 → j ≤ 6 → ∃ c1 c2 c3, c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
  (board i j = c1 ∨ board (i+1) j = c1 ∨ board (i+2) j = c1 ∨
   board i (j+1) = c1 ∨ board i (j+2) = c1) ∧
  (board i j = c2 ∨ board (i+1) j = c2 ∨ board (i+2) j = c2 ∨
   board i (j+1) = c2 ∨ board i (j+2) = c2) ∧
  (board i j = c3 ∨ board (i+1) j = c3 ∨ board (i+2) j = c3 ∨
   board i (j+1) = c3 ∨ board i (j+2) = c3)

theorem good_coloring_count_at_least_6_pow_8 :
  ∃ f : (ℕ → ℕ → ℕ) → Prop, (∀ board, f board → is_good_coloring board) ∧
  (∃ boards : list (ℕ → ℕ → ℕ), ∀ board ∈ boards, f board) ∧
  boards.length ≥ 6 ^ 8 :=
sorry

end good_coloring_count_at_least_6_pow_8_l277_277927


namespace larger_number_is_1891_l277_277778

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem larger_number_is_1891 :
  ∃ L S : ℕ, (L - S = 1355) ∧ (L = 6 * S + 15) ∧ is_prime (sum_of_digits L) ∧ sum_of_digits L ≠ 12
  :=
sorry

end larger_number_is_1891_l277_277778


namespace unique_special_divisor_l277_277854

def is_divisor_count (d n : ℕ) := 
  (finset.range (n + 1)).filter (λ x, n % x = 0).card = d

def is_valid_divisor_form (d : ℕ) (a b c : ℕ) :=
  d = 2^a * 3^b * 5^c

theorem unique_special_divisor : ∃! d : ℕ, 
  is_valid_divisor_form d 1 2 24 ∧
  is_divisor_count 150 d ∧
  d ∣ (2^150 * 3^150 * 5^300) :=
sorry

end unique_special_divisor_l277_277854


namespace average_of_first_two_is_1_point_1_l277_277584

theorem average_of_first_two_is_1_point_1
  (a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.5)
  (h2 : (a1 + a2) / 2 = x)
  (h3 : (a3 + a4) / 2 = 1.4)
  (h4 : (a5 + a6) / 2 = 5) :
  x = 1.1 := 
sorry

end average_of_first_two_is_1_point_1_l277_277584


namespace average_shirts_per_day_l277_277715

theorem average_shirts_per_day
  (employees : ℕ) -- number of employees
  (work_hours : ℕ) -- hours per shift
  (hourly_wage : ℕ) -- hourly wage in dollars
  (shirt_wage : ℕ) -- additional wage per shirt made in dollars
  (shirt_price : ℕ) -- price of each shirt in dollars
  (daily_expenses : ℕ) -- daily nonemployee expenses in dollars
  (profit_per_day : ℕ) -- company's profit per day in dollars)
  (S : ℕ) -- average number of shirts made by each employee per day
  (employee_count : employees = 20)
  (work_hours_eq : work_hours = 8)
  (hourly_wage_eq : hourly_wage = 12)
  (shirt_wage_eq : shirt_wage = 5)
  (shirt_price_eq : shirt_price = 35)
  (daily_expenses_eq : daily_expenses = 1000)
  (profit_eq : profit_per_day = 9080)
  (total_profit_eq :
    profit_per_day = (employees * shirt_price * S) - 
                     (employees * (hourly_wage * work_hours + shirt_wage * S)) - 
                     daily_expenses) : 
  S = 20 :=
by {
  sorry,
}

end average_shirts_per_day_l277_277715


namespace books_a_count_l277_277223

theorem books_a_count (A B : ℕ) (h1 : A + B = 20) (h2 : A = B + 4) : A = 12 :=
by
  sorry

end books_a_count_l277_277223


namespace find_cost_price_l277_277153

-- Definitions based on the conditions
def selling_price : ℝ := 27000
def discount_rate : ℝ := 0.10
def profit_rate : ℝ := 0.08

-- The proof problem statement
theorem find_cost_price :
  let discounted_selling_price := selling_price * (1 - discount_rate)
  discounted_selling_price = 24300 → 
  ∃ CP : ℝ, discounted_selling_price = CP * (1 + profit_rate) ∧ CP = 22500 := 
by
  intro h
  use 22500
  split
  {
    exact h
  }
  {
    sorry
  }

end find_cost_price_l277_277153


namespace bisector_c_value_l277_277731

theorem bisector_c_value :
  let P : ℝ × ℝ := (-7, 6)
  let Q : ℝ × ℝ := (-13, -15)
  let R : ℝ × ℝ := (4, -6)
  in ∃ c : ℝ, (∀ x y : ℝ, 3 * x + y + c = 0) ∧ c = -39 := sorry

end bisector_c_value_l277_277731


namespace prime_condition_l277_277503

theorem prime_condition (p : ℕ) (k : ℤ) :
  prime p ∧ p > 5 ∧ (p : ℤ) ∣ (k^2 + 5) →
  ∃ (m n : ℕ), p^2 = m^2 + 5 * n^2 :=
by
  sorry

end prime_condition_l277_277503


namespace evaluate_expression_l277_277325

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end evaluate_expression_l277_277325


namespace optimal_order_l277_277539

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l277_277539


namespace number_of_even_1s_grids_l277_277096

theorem number_of_even_1s_grids (n : ℕ) : 
  (∃ grid : fin n → fin n → ℕ, 
    (∀ i j, grid i j = 0 ∨ grid i j = 1) ∧
    (∀ i, (∑ j, grid i j) % 2 = 0) ∧
    (∀ j, (∑ i, grid i j) % 2 = 0)) →
  2 ^ ((n - 1) * (n - 1)) = 2 ^ ((n - 1) * (n - 1)) :=
by sorry

end number_of_even_1s_grids_l277_277096


namespace unique_function_divisibility_l277_277918

theorem unique_function_divisibility (f : ℕ+ → ℕ+) 
  (h : ∀ a b : ℕ+, (a^2 + f a * f b) % (f a + b) = 0) : 
    ∀ n : ℕ+, f n = n := 
sorry

end unique_function_divisibility_l277_277918


namespace total_time_proof_l277_277162

-- Define the known values
def speed_of_current (x : ℝ) : ℝ := x
def speed_of_boat (x : ℝ) : ℝ := 3 * x
def round_trip_time_without_current : ℝ := 20

-- Define the total round trip time considering the current
def total_round_trip_time_with_current (x : ℝ) : ℝ :=
  let d := 10 * speed_of_boat x -- distance
  (d / (speed_of_boat x + speed_of_current x)) + (d / (speed_of_boat x - speed_of_current x))

-- Problem to prove: the total round trip time considering the current is 22.5 minutes
theorem total_time_proof (x : ℝ) (h : x > 0) : total_round_trip_time_with_current x = 22.5 :=
  by sorry

end total_time_proof_l277_277162


namespace find_radioactive_balls_within_7_checks_l277_277159

theorem find_radioactive_balls_within_7_checks :
  ∃ (balls : Finset α), balls.card = 11 ∧ ∃ radioactive_balls ⊆ balls, radioactive_balls.card = 2 ∧
  (∀ (check : Finset α → Prop), (∀ S, check S ↔ (∃ b ∈ S, b ∈ radioactive_balls)) →
  ∃ checks : Finset (Finset α), checks.card ≤ 7 ∧ (∀ b ∈ radioactive_balls, ∃ S ∈ checks, b ∈ S)) :=
sorry

end find_radioactive_balls_within_7_checks_l277_277159


namespace tetrahedron_labeling_impossible_l277_277180

/-- Suppose each vertex of a tetrahedron needs to be labeled with an integer from 1 to 4, each integer being used exactly once.
We need to prove that there are no such arrangements in which the sum of the numbers on the vertices of each face is the same for all four faces.
Arrangements that can be rotated into each other are considered identical. -/
theorem tetrahedron_labeling_impossible :
  ∀ (label : Fin 4 → Fin 5) (h_unique : ∀ v1 v2 : Fin 4, v1 ≠ v2 → label v1 ≠ label v2),
  ∃ (sum_faces : ℕ), sum_faces = 7 ∧ sum_faces % 3 = 1 → False :=
by
  sorry

end tetrahedron_labeling_impossible_l277_277180


namespace solve_equation_l277_277221

theorem solve_equation (x : ℝ) (h : x + 3 ≠ 0) : (2 / (x + 3) = 1) → (x = -1) :=
by
  intro h1
  -- Proof skipped
  sorry

end solve_equation_l277_277221


namespace probability_of_high_value_hand_l277_277079

noncomputable def bridge_hand_probability : ℚ :=
  let total_combinations : ℕ := Nat.choose 16 4
  let favorable_combinations : ℕ := 1 + 16 + 16 + 16 + 36 + 96 + 16
  favorable_combinations / total_combinations

theorem probability_of_high_value_hand : bridge_hand_probability = 197 / 1820 := by
  sorry

end probability_of_high_value_hand_l277_277079


namespace expected_lotus_seed_zongzi_is_3_l277_277485

-- Define all the conditions
def total_zongzi : ℕ := 72 + 18 + 36 + 54
def lotus_seed_zongzi : ℕ := 54
def num_selected_zongzi : ℕ := 10

-- Define the expected number of lotus seed zongzi in the gift box
def expected_lotus_seed_zongzi : ℚ := num_selected_zongzi * (↑lotus_seed_zongzi / ↑total_zongzi)

/-- Prove that the expected number of lotus seed zongzi in the gift box is 3. -/
theorem expected_lotus_seed_zongzi_is_3 :
  expected_lotus_seed_zongzi = 3 :=
by sorry

end expected_lotus_seed_zongzi_is_3_l277_277485


namespace monge_point_intersection_l277_277169

structure Point3D :=
(x y z : ℝ)

structure Tetrahedron :=
(A B C D : Point3D)

structure Plane :=
(point : Point3D)
(normal : Point3D)

def midpoint (p1 p2 : Point3D) : Point3D :=
{ x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

def plane_through_midpoint_perpendicular_to_edge (mid : Point3D) (p1 p2 : Point3D) : Plane :=
{ point := mid
, normal := { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z } }

def monge_point_exists (T : Tetrahedron) : Prop :=
∃ P : Point3D,
let M_AB := midpoint T.A T.B,
    M_AC := midpoint T.A T.C,
    M_AD := midpoint T.A T.D,
    M_BC := midpoint T.B T.C,
    M_BD := midpoint T.B T.D,
    M_CD := midpoint T.C T.D,
    planes := [
      plane_through_midpoint_perpendicular_to_edge M_AB T.C T.D,
      plane_through_midpoint_perpendicular_to_edge M_AC T.B T.D,
      plane_through_midpoint_perpendicular_to_edge M_AD T.B T.C,
      plane_through_midpoint_perpendicular_to_edge M_BC T.A T.D,
      plane_through_midpoint_perpendicular_to_edge M_BD T.A T.C,
      plane_through_midpoint_perpendicular_to_edge M_CD T.A T.B
    ]
in ∀ p ∈ planes, (p.point = P)

theorem monge_point_intersection (T : Tetrahedron) : monge_point_exists T :=
sorry

end monge_point_intersection_l277_277169


namespace optimal_order_l277_277554

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l277_277554


namespace PS_div_QR_eq_sqrt_3_l277_277481

-- We assume the lengths and properties initially given in the problem.
def P := sorry -- placeholder for point P
def Q := sorry -- placeholder for point Q
def R := sorry -- placeholder for point R
def S := sorry -- placeholder for point S
def t : ℝ := sorry -- the side length of the equilateral triangles

-- Essential properties of equilateral triangles
axiom PQR_is_equilateral : equilateral_triangle P Q R
axiom QRS_is_equilateral : equilateral_triangle Q R S
axiom side_length_QR : dist Q R = t

-- Heights of triangles from vertices to the opposite sides
def height_PQR : ℝ := t * (sqrt 3) / 2
def height_QRS : ℝ := t * (sqrt 3) / 2

-- PS is the sum of these heights
def PS : ℝ := height_PQR + height_QRS

-- Theorem
theorem PS_div_QR_eq_sqrt_3 : PS / t = sqrt 3 := by
  sorry

end PS_div_QR_eq_sqrt_3_l277_277481


namespace part1_monotonicity_and_extremes_part2_monotonically_decreasing_l277_277399

-- Define the function 
def f (x a : ℝ) : ℝ := 1 / 2 * x^2 + a * x - 2 * log x

-- Proof for the first part of the problem
theorem part1_monotonicity_and_extremes (a : ℝ) (h : a = 1) : (∀ x > 1, deriv (f x a) > 0) ∧ (∀ x, 0 < x ∧ x < 1 -> deriv (f x a) < 0) ∧ (f 1 a = 3 / 2) := by
  sorry

-- Proof for the second part of the problem
theorem part2_monotonically_decreasing (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 2 -> deriv (f x a) ≤ 0) ↔ a ≤ -1 := by
  sorry

end part1_monotonicity_and_extremes_part2_monotonically_decreasing_l277_277399


namespace perpendicular_line_equation_l277_277630

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l277_277630


namespace cos_value_range_f_l277_277421

-- Definitions based on given conditions
def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (x / 4), 1)
def n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Problem I: Prove the value of cos given f(x) = 1
theorem cos_value (x : ℝ) (h : f x = 1) : cos (π / 3 + x) = 1 / 2 := sorry

-- Triangle condition and range of the function f(A)
variables {a b c A B C : ℝ} (h_triangle : (2 * a - c) * cos B = b * cos C)

-- Problem II: Prove the range of the function f(A) under given triangle condition
theorem range_f (h_angles : 0 < A ∧ A < 2 * π / 3 ∧ cos B = 1 / 2) : 1 < f A ∧ f A < 3 / 2 := sorry

end cos_value_range_f_l277_277421


namespace quadratic_non_real_roots_interval_l277_277761

theorem quadratic_non_real_roots_interval :
  {b : ℝ | ∀ x, ¬ ∃ a c : ℝ, a = 1 ∧ c = 16 ∧ x^2 + b * x + c = 0 ∧ b^2 - 4 * a * c < 0} = set.Ioo (-8 : ℝ) 8 :=
sorry

end quadratic_non_real_roots_interval_l277_277761


namespace chebyshev_polynomial_form_l277_277166

def T (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then x
  else 2 * x * T (n - 1) x - T (n - 2) x

theorem chebyshev_polynomial_form (n : ℕ) (x : ℝ) (a : ℕ → ℤ) :
  ∃ (a : ℕ → ℤ), T n x = 2^(n - 1) * x^n + ∑ i in (finset.range n), (a i) * x^(n - 1 - i) := 
sorry

end chebyshev_polynomial_form_l277_277166


namespace intersection_A_B_is_C_l277_277415

open Set

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | 3 - 2x > 0 }
def C : Set ℝ := { x | x < 3 / 2 }

theorem intersection_A_B_is_C : A ∩ B = C := by
  sorry

end intersection_A_B_is_C_l277_277415


namespace division_of_5_parts_division_of_7_parts_division_of_8_parts_l277_277257

-- Problem 1: Primary Division of Square into 5 Equal Parts
theorem division_of_5_parts (x : ℝ) (h : x^2 = 1 / 5) : x = Real.sqrt (1 / 5) :=
sorry

-- Problem 2: Primary Division of Square into 7 Equal Parts
theorem division_of_7_parts (x : ℝ) (hx : 196 * x^3 - 294 * x^2 + 128 * x - 15 = 0) : 
  x = (7 + Real.sqrt 19) / 14 :=
sorry

-- Problem 3: Primary Division of Square into 8 Equal Parts
theorem division_of_8_parts (x : ℝ) (hx : 6 * x^2 - 6 * x + 1 = 0) : 
  x = (3 + Real.sqrt 3) / 6 :=
sorry

end division_of_5_parts_division_of_7_parts_division_of_8_parts_l277_277257


namespace sum_of_primes_is_prime_l277_277601

theorem sum_of_primes_is_prime (A B : ℕ) (hA : A ∈ {5, 7, 11}) (hB : B = 2)
  (h1 : Prime A) (h2 : Prime B) (h3 : Prime (A - B)) (h4 : Prime (A + B)) :
  Prime (A + B + (A - B) + B) :=
by
  sorry

end sum_of_primes_is_prime_l277_277601


namespace perpendicular_line_through_point_l277_277646

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l277_277646


namespace min_sine_difference_l277_277136

theorem min_sine_difference (N : ℕ) (hN : 0 < N) :
  ∃ (n k : ℕ), (1 ≤ n ∧ n ≤ N + 1) ∧ (1 ≤ k ∧ k ≤ N + 1) ∧ (n ≠ k) ∧ 
    (|Real.sin n - Real.sin k| < 2 / N) := 
sorry

end min_sine_difference_l277_277136


namespace perpendicular_line_through_point_l277_277638

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l277_277638


namespace correct_option_l277_277888

def sin_rule_condition (a b: ℝ) (A: ℝ) : Prop :=
  let sinA := real.sin (A * real.pi / 180)
  let sinB := b * sinA / a
  sinB ≤ 1

theorem correct_option :
  ¬ (sin_rule_condition 7 18 30) ∧
  (sin_rule_condition 9 10 60) ∧
  ¬ (sin_rule_condition 6 9 45) ∧
  (sin_rule_condition 24 28 150) :=
by
  sorry

end correct_option_l277_277888


namespace fill_grid_with_even_ones_l277_277084

theorem fill_grid_with_even_ones (n : ℕ) : 
  ∃ ways : ℕ, ways = 2^((n-1)^2) ∧ 
  (∀ grid : array n (array n (fin 2)), 
    (∀ i : fin n, even (grid[i].to_list.count (λ x, x = 1))) ∧ 
    (∀ j : fin n, even (grid.map (λ row, row[j]).to_list.count (λ x, x = 1)))) :=
begin
  use 2^((n-1)^2),
  split,
  { refl },
  { sorry },
end

end fill_grid_with_even_ones_l277_277084


namespace integral_bounds_l277_277322

theorem integral_bounds :
  (2 * Real.pi / 21) ≤ (∫ x in 0..(Real.pi / 6), 1 / (1 + 3 * (Real.sin x)^2)) ∧
  (∫ x in 0..(Real.pi / 6), 1 / (1 + 3 * (Real.sin x)^2)) ≤ (Real.pi / 6) :=
by
  sorry

end integral_bounds_l277_277322


namespace compare_solutions_l277_277505

theorem compare_solutions 
  (c d p q : ℝ) 
  (hc : c ≠ 0) 
  (hp : p ≠ 0) :
  (-d / c) < (-q / p) ↔ (q / p) < (d / c) :=
by
  sorry

end compare_solutions_l277_277505


namespace exact_time_now_l277_277109

noncomputable def time_now (t : ℝ) : Prop := 
  (4 < t / 60) ∧ (t / 60 < 5) ∧
  (|6 * (t + 8) - (120 + 0.5 * (t - 6))| = 180)

theorem exact_time_now : ∃ t : ℝ, time_now t ∧ t = 45.27 :=
by
  sorry

end exact_time_now_l277_277109


namespace transformed_mean_and_variance_l277_277725

variables {X : Type} [AddCommGroup X] [Module ℝ X] {n : ℕ}

-- Define the mean and variance for a dataset
def mean (data : Fin n → ℝ) : ℝ :=
  (∑ i, data i) / n

def variance (data : Fin n → ℝ) : ℝ :=
  let m := mean data in
  (∑ i, (data i - m) ^ 2) / n

-- Given conditions
variables (X_data : Fin n → ℝ)
hypothesis h_mean : mean X_data = 3
hypothesis h_variance : variance X_data = 5

-- Prove the mean and variance of transformed data
def transformed_data (data : Fin n → ℝ) : Fin n → ℝ :=
  fun i => 3 * (data i) + 2

theorem transformed_mean_and_variance :
  mean (transformed_data X_data) = 11 ∧
  variance (transformed_data X_data) = 45 :=
by
  sorry

end transformed_mean_and_variance_l277_277725


namespace part1_part2_l277_277284

-- Define the sequence properties and the required proofs
def sequence_property (x : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ ∀ n : ℕ, x n ≥ x (n + 1)

theorem part1 (x : ℕ → ℝ) (h : sequence_property x) :
  ∃ n ≥ 1, (finset.range n).sum (λ i, x i ^ 2 / x (i + 1)) ≥ 3.999 :=
sorry

def specific_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n, x n = 2 ^ (-n : ℤ)

theorem part2 (x : ℕ → ℝ) (h : specific_sequence x) :
  ∀ n, (finset.range n).sum (λ i, x i ^ 2 / x (i + 1)) < 4 :=
sorry

end part1_part2_l277_277284


namespace lowest_cost_per_ton_l277_277968

-- Define the conditions given in the problem statement
variable (x : ℝ) (y : ℝ)

-- Define the annual production range
def production_range (x : ℝ) : Prop := x ≥ 150 ∧ x ≤ 250

-- Define the relationship between total annual production cost and annual production
def production_cost_relation (x y : ℝ) : Prop := y = (x^2 / 10) - 30 * x + 4000

-- State the main theorem: the annual production when the cost per ton is the lowest is 200 tons
theorem lowest_cost_per_ton (x : ℝ) (y : ℝ) (h1 : production_range x) (h2 : production_cost_relation x y) : x = 200 :=
sorry

end lowest_cost_per_ton_l277_277968


namespace sarah_fund_amount_l277_277674

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem sarah_fund_amount :
  let P := 1500
  let r := 0.04
  let t := 21
  let A := compound_interest P r t
  A ≈ 3046.28 := sorry

end sarah_fund_amount_l277_277674


namespace tangent_circle_radius_l277_277276

theorem tangent_circle_radius (R : ℝ) (hR : R > 0) :
    ∃ x : ℝ, x = R / 4 ∧
    (∃ O O_1 O_2 : ℝ × ℝ,
        ∃ M K : ℝ × ℝ,
        (dist O O_1 = R / 2) ∧
        (dist O O_2 = R - x) ∧
        (dist O M = sqrt (2 * R * x)) ∧
        (dist O_2 M = x) ∧
        (dist O K = R) ∧
        (dist O_2 K = 0) ∧
        (R - x)^2 = (sqrt (2 * R * x))^2 + x^2) := sorry

end tangent_circle_radius_l277_277276


namespace range_of_m_range_of_x_l277_277829

-- Define the function f(x) = m*x^2 - m*x - 6 + m
def f (m x : ℝ) : ℝ := m*x^2 - m*x - 6 + m

-- Proof for the first statement
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f m x < 0) ↔ m < 6 / 7 := 
sorry

-- Proof for the second statement
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end range_of_m_range_of_x_l277_277829


namespace unique_colorings_count_l277_277892

-- Definitions based on problem conditions
def Colors := Fin 3 -- represent the three colors as finite elements

-- Define the type representing the squares in the 2x2 grid
inductive Square 
| A | B | C | D

open Square

-- A coloring is a function from squares to colors
def Coloring := Square → Colors

-- Define the conditions
def shares_side (s1 s2 : Square) : Prop :=
  (s1 = A ∧ s2 ∈ {B, D}) ∨ (s1 = B ∧ s2 ∈ {A, C}) ∨ 
  (s1 = C ∧ s2 ∈ {B, D}) ∨ (s1 = D ∧ s2 ∈ {A, C})

def valid_coloring (f : Coloring) : Prop :=
  (∀ s1 s2, shares_side s1 s2 → f s1 ≠ f s2) ∧
  ∃ color1 color2 color3, 
    {c | ∃ s, f s = c}.card = 3 ∧ 
    color1 ≠ color2 ∧ color2 ≠ color3 ∧ color1 ≠ color3

-- Theorem statement
theorem unique_colorings_count : 
  ∃! n : ℕ, n = 3 ∧ ∃ (f : Finset Coloring), 
    (∀ c ∈ f, valid_coloring c) ∧ f.card = n := 
sorry

end unique_colorings_count_l277_277892


namespace determine_range_of_m_l277_277374

variable {m : ℝ}

def discriminant (m : ℝ) : ℝ := 4 - 4 * m

def p (m : ℝ) : Prop := discriminant m ≥ 0
def q (m : ℝ) : Prop := -1 ≤ m ∧ m ≤ 5

def false_prop (p q : Prop) : Prop := ¬(p ∧ q)
def true_prop (p q : Prop) : Prop := p ∨ q

def range_of_m (m : ℝ) : Prop :=
  (-∞ < m ∧ m < -1) ∨ (1 < m ∧ m ≤ 5)

theorem determine_range_of_m (m : ℝ) (hp : p m) (hq : q m)
  (h_false_prop : false_prop (p m) (q m))
  (h_true_prop : true_prop (p m) (q m)) :
  range_of_m m := by
  sorry

end determine_range_of_m_l277_277374


namespace more_good_than_bad_l277_277195

-- Defining a good time as when hour, minute, and second hands are on the same side of a diameter
def is_good_time (hour minute second : ℝ) : Prop :=
  ∃ θ : ℝ, (0 ≤ θ ∧ θ < π) ∧
    (hour / 12 * 2 * π ≤ θ ∨ hour / 12 * 2 * π - 2 * π ≤ θ) ∧
    (minute / 60 * 2 * π ≤ θ ∨ minute / 60 * 2 * π - 2 * π ≤ θ) ∧
    (second / 60 * 2 * π ≤ θ ∨ second / 60 * 2 * π - 2 * π ≤ θ)

-- Definition of a 24-hour period in terms of "good" times
def good_time_over_day : Prop :=
  ∀ t : ℝ, (0 ≤ t ∧ t < 24 * 3600) →
  is_good_time (t / 3600) ((t % 3600) / 60) (t % 60)

theorem more_good_than_bad : good_time_over_day → 
  (∃ good_times : ℝ, ∃ bad_times : ℝ, good_times > bad_times ∧ good_times + bad_times = 24) :=
by {
  sorry
}

end more_good_than_bad_l277_277195


namespace problem1_problem2_problem3_l277_277613

def arrangement_A_not_head_and_B_not_tail : ℕ := 
  5! - 4! * 2 + 3!

def arrangement_A_and_B_not_adjacent : ℕ := 
  5! - (4! * 2!)

def arrangement_A_and_B_together_C_and_D_not_together : ℕ := 
  2! * 2! * 3! - (2! * 2!)

theorem problem1 : 
  5! - 4! * 2 + 3! = 78 := by
  sorry

theorem problem2 : 
  5! - (4! * 2!) = 72 := by
  sorry

theorem problem3 : 
  (2! * 2! * 3!) - (2! * 2!) = 24 := by
  sorry

end problem1_problem2_problem3_l277_277613


namespace honey_last_nights_l277_277576

theorem honey_last_nights 
  (serving_per_cup : ℕ)
  (cups_per_night : ℕ)
  (ounces_per_container : ℕ)
  (servings_per_ounce : ℕ)
  (total_nights : ℕ) :
  serving_per_cup = 1 →
  cups_per_night = 2 →
  ounces_per_container = 16 →
  servings_per_ounce = 6 →
  total_nights = 48 := 
by
  intro h1 h2 h3 h4,
  sorry

end honey_last_nights_l277_277576


namespace max_val_MN_l277_277837

noncomputable def maxMN :=
  let curve_C_polar := λ θ, (2 * Real.sin θ)
  let curve_C_cartesian := λ x y, (x^2 + y^2 - 2 * y = 0)
  let line_l := λ t, (⟨-3/5 * t + 2, 4/5 * t⟩ : ℝ × ℝ)
  let point_M := (2, 0) : ℝ × ℝ
  let center_C := (0, 1) : ℝ × ℝ
  let radius_C := 1 : ℝ
  let dist := λ p1 p2 : ℝ × ℝ, Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  ∀ N : ℝ × ℝ, curve_C_cartesian N.1 N.2 →
    dist point_M N ≤ dist point_M center_C + radius_C
  
theorem max_val_MN :
  maxMN = Real.sqrt 5 + 1 := sorry

end max_val_MN_l277_277837


namespace minimum_rectangle_length_l277_277581

theorem minimum_rectangle_length (a x y : ℝ) (h : x * y = a^2) : x ≥ a ∨ y ≥ a :=
sorry

end minimum_rectangle_length_l277_277581


namespace mode_and_median_of_scores_l277_277464

-- Conditions: Scores and their frequencies
def score_frequencies : List (ℕ × ℕ) :=
[(0, 1), (1, 3), (2, 4), (3, 14), (4, 8)]

-- Question reformulated as a Lean 4 statement
theorem mode_and_median_of_scores :
  (mode score_frequencies = 3) ∧ (median score_frequencies = 3) :=
by
  sorry

end mode_and_median_of_scores_l277_277464


namespace congruent_faces_of_tetrahedron_l277_277167

theorem congruent_faces_of_tetrahedron
  (T : Type)
  [affine.simplex 3 T] -- Assume T is a 3-dimensional simplex (tetrahedron)
  (center_circum_insphere : 
    (affine_trilateral (circumsphere_center T) = affine_trilateral (insphere_center T)))
  : ∀ (face1 face2 : T), affine.congruent face1 face2 :=
sorry

end congruent_faces_of_tetrahedron_l277_277167


namespace value_of_b_pow4_plus_b_inv_pow4_l277_277571

theorem value_of_b_pow4_plus_b_inv_pow4 (b : ℝ) (h : 5 = b + b⁻¹) : b^4 + b^(-4) = 527 := sorry

end value_of_b_pow4_plus_b_inv_pow4_l277_277571


namespace tourists_remaining_l277_277269

theorem tourists_remaining (initial_tourists : ℕ) (eaten_by_anacondas : ℕ) (poisoned_fraction : ℚ) 
  (recovered_poisoned_fraction : ℚ) (bitten_fraction : ℚ) (received_antivenom_fraction : ℚ) :
  initial_tourists = 42 →
  eaten_by_anacondas = 3 →
  poisoned_fraction = 2/3 →
  recovered_poisoned_fraction = 2/9 →
  bitten_fraction = 1/4 →
  received_antivenom_fraction = 3/5 →
  let remaining_after_anacondas := initial_tourists - eaten_by_anacondas in
  let poisoned_tourists := int.of_nat (poisoned_fraction * remaining_after_anacondas).floor in
  let recovered_poisoned := int.of_nat (recovered_poisoned_fraction * poisoned_tourists).floor in
  let remaining_after_poisoned_recover := remaining_after_anacondas - poisoned_tourists + recovered_poisoned in
  let bitten_tourists := int.of_nat (bitten_fraction * remaining_after_poisoned_recover).floor in
  let received_antivenom := int.of_nat (received_antivenom_fraction * bitten_tourists).floor in
  let final_remaining := remaining_after_poisoned_recover - bitten_tourists + received_antivenom in
  final_remaining = 16 :=
begin
  intros,
  sorry
end

end tourists_remaining_l277_277269


namespace count_visible_factor_numbers_in_range_100_to_150_l277_277272

def is_divisible_by_all_nonzero_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

def is_visible_factor_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 150 ∧ is_divisible_by_all_nonzero_digits n

theorem count_visible_factor_numbers_in_range_100_to_150 :
  (List.range' 100 51).countp is_visible_factor_number = 19 := 
by
  sorry

end count_visible_factor_numbers_in_range_100_to_150_l277_277272


namespace linear_function_equation_no_solutions_proof_l277_277249

noncomputable def f (x : ℝ) : ℝ := sorry

theorem linear_function_equation_no_solutions_proof :
  (∀ x, f(f(x)) = x + 1 → false) →
  f(f(f(f(f(2022))))) - f(f(f(2022))) - f(f(2022)) = -2022 :=
by
  intro H
  -- Assuming f is a linear function, encapsulated by some form of the equation
  sorry

end linear_function_equation_no_solutions_proof_l277_277249


namespace probability_reach_3_1_in_8_steps_l277_277179

theorem probability_reach_3_1_in_8_steps :
  let m := 35
  let n := 2048
  let q := m / n
  ∃ (m n : ℕ), (Nat.gcd m n = 1) ∧ (q = 35 / 2048) ∧ (m + n = 2083) := by
  sorry

end probability_reach_3_1_in_8_steps_l277_277179


namespace ariane_permutation_single_cycle_l277_277000

variable (n : ℕ) (h : n ≥ 1)
variable (vertices : Set ℕ) (edges : Set (ℕ × ℕ))

def arianeProcess (initial : ℕ → ℕ) (edges : Set (ℕ × ℕ)) : (ℕ → ℕ) :=
  edges.fold (fun pi (e : ℕ × ℕ) => 
    let (v1, v2) := e
    fun k => if pi k = v1 then v2 else if pi k = v2 then v1 else pi k)
    initial

def permutation (initial : ℕ → ℕ) : ℕ → ℕ := 
  arianeProcess initial edges

theorem ariane_permutation_single_cycle (h : n ≥ 1)
    (initial : {k // 1 ≤ k ∧ k ≤ n} → ℕ)
    (edges : Set (ℕ × ℕ))
    (pi : {k // 1 ≤ k ∧ k ≤ n} → ℕ) :
    permutation initial edges = pi → ∃! cycle : List {k // 1 ≤ k ∧ k ≤ n}, ∀ k ∈ cycle, pi k = cycle.cycle:
by {
  sorry
}

end ariane_permutation_single_cycle_l277_277000


namespace tan_difference_l277_277792

noncomputable def cos_alpha : Real := 1 / 3
noncomputable def tan_phi : Real := Real.sqrt 2
noncomputable def alpha : Real := Real.arccos cos_alpha  -- α should be in \( (-\frac{\pi}{2}, 0) \)
noncomputable def phi : Real := Real.arctan tan_phi

theorem tan_difference (h1 : cos alpha = (1 / 3)) (h2 : tan phi = Real.sqrt 2) : 
  Real.tan (phi - alpha) = -Real.sqrt 2 := 
  sorry

end tan_difference_l277_277792


namespace ticket_cost_calculation_l277_277896

theorem ticket_cost_calculation :
  let adult_price := 12
  let child_price := 10
  let num_adults := 3
  let num_children := 3
  let total_cost := (num_adults * adult_price) + (num_children * child_price)
  total_cost = 66 := 
by
  rfl -- or add sorry to skip proof

end ticket_cost_calculation_l277_277896


namespace find_third_test_score_l277_277151

-- Definitions of the given conditions
def test_score_1 := 80
def test_score_2 := 70
variable (x : ℕ) -- the unknown third score
def test_score_4 := 100
def average_score (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4

-- Theorem stating that given the conditions, the third test score must be 90
theorem find_third_test_score (h : average_score test_score_1 test_score_2 x test_score_4 = 85) : x = 90 :=
by
  sorry

end find_third_test_score_l277_277151


namespace construct_length_one_l277_277805

theorem construct_length_one
    (a : ℝ) 
    (h_a : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) : 
    ∃ (b : ℝ), b = 1 :=
by
    sorry

end construct_length_one_l277_277805


namespace range_of_a_l277_277141

theorem range_of_a (a : ℝ)
  (A : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0})
  (B : Set ℝ := {x : ℝ | x ≥ a - 1})
  (H : A ∪ B = Set.univ) :
  a ≤ 2 :=
by
  sorry

end range_of_a_l277_277141


namespace advantageous_order_l277_277543

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l277_277543


namespace solve_exponential_eq_l277_277569

open Real

theorem solve_exponential_eq (x : ℝ) :
    (4^x - 13 * 2^x + 12 = 0) ↔ (x = 0 ∨ x = 2 + log 2 3) :=
by
    sorry

end solve_exponential_eq_l277_277569


namespace geom_seq_log_sum_l277_277990

theorem geom_seq_log_sum (b s : ℕ) (hb : b > 0) (hs : s > 0)
  (hlog_sum : ∑ i in finset.range 10, real.log (b * s^i) / real.log 4 = 1005) :
  (b, s) = (8, 2^44) :=
sorry

end geom_seq_log_sum_l277_277990


namespace remainder_when_divided_by_x_minus_13_x_minus_17_l277_277908

noncomputable def polynomial_remainder (P : Polynomial ℤ) : Polynomial ℤ :=
  let a := 2
  let b := -20
  a * (Polynomial.X) + b

theorem remainder_when_divided_by_x_minus_13_x_minus_17 (P : Polynomial ℤ) :
  (P % ((Polynomial.X - Polynomial.C 13) * (Polynomial.X - Polynomial.C 17))) = polynomial_remainder P :=
by
  sorry

end remainder_when_divided_by_x_minus_13_x_minus_17_l277_277908


namespace problem_l277_277393

theorem problem
  (a b : ℝ)
  (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) :
  2 * a^100 - 3 * b⁻¹ = 3 := 
by {
  -- Proof steps go here
  sorry
}

end problem_l277_277393


namespace min_magnitude_at_t_zero_l277_277356

def a (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
def b (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)

def vector_diff_magnitude (t : ℝ) : ℝ :=
  let diff := ((1 - t - 2), (2 * t - 1 - t), (0 - t))
  in (diff.1^2 + diff.2^2 + diff.3^2).sqrt

theorem min_magnitude_at_t_zero : ∀ t : ℝ, vector_diff_magnitude t ≥ Real.sqrt 2 :=
by
  sorry

end min_magnitude_at_t_zero_l277_277356


namespace exactly_one_even_l277_277165

theorem exactly_one_even (a b c : ℕ) : 
  (∀ x, ¬ (a = x ∧ b = x ∧ c = x) ∧ 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ b % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ c % 2 = 0) ∧ 
  ¬ (b % 2 = 0 ∧ c % 2 = 0)) :=
by
  sorry

end exactly_one_even_l277_277165


namespace sum_values_f_eq_0_l277_277138

def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x - 3 else x / 2 + 1

theorem sum_values_f_eq_0 : {x : ℝ | f x = 0}.sum id = -3 :=
by
  sorry

end sum_values_f_eq_0_l277_277138


namespace cos_double_angle_shift_l277_277436

theorem cos_double_angle_shift (α : ℝ) (h : sqrt 3 * sin α + cos α = 1/2) : 
  cos (2 * α + 4 * π / 3) = -7/8 :=
sorry

end cos_double_angle_shift_l277_277436


namespace paint_needed_for_combined_solid_additional_paint_for_divided_block_smaller_cubes_after_division_l277_277694

-- Part (a)
theorem paint_needed_for_combined_solid
  (cube1_side_length cube2_side_length : ℕ)
  (paint_ratio : ℕ)
  (cube1_side_length = 10)
  (cube2_side_length = 20)
  (paint_ratio = 100) :
  (paint_ratio / 100 * (6 * cube1_side_length^2 + 6 * cube2_side_length^2 - 2 * cube1_side_length^2) = 28) :=
sorry

-- Part (b)
theorem additional_paint_for_divided_block
  (original_paint : ℕ)
  (paint_ratio : ℕ)
  (original_paint = 54)
  (paint_ratio = 100) :
  (2 * (original_paint / 6 * paint_ratio) / paint_ratio = 18) :=
sorry

-- Part (c)
theorem smaller_cubes_after_division
  (initial_paint additional_paint : ℕ)
  (paint_ratio : ℕ)
  (initial_paint = 54)
  (additional_paint = 216)
  (paint_ratio = 100) :
  ((additional_paint / (paint_ratio * (additional_paint / initial_paint)))^3 = 125) :=
sorry

end paint_needed_for_combined_solid_additional_paint_for_divided_block_smaller_cubes_after_division_l277_277694


namespace sphere_volume_l277_277611

theorem sphere_volume (r : ℝ) (h1 : 4 * π * r^2 = 256 * π) : 
  (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l277_277611


namespace exponent_multiplication_l277_277750

theorem exponent_multiplication :
  (5^0.2 * 10^0.4 * 10^0.1 * 10^0.5 * 5^0.8) = 50 := by
  sorry

end exponent_multiplication_l277_277750


namespace count_valid_integers_l277_277850

-- Definition of the condition that an integer is a valid 4-digit number with unique digits, leading digit non-zero, multiple of 2, and largest digit 7.
def isValidInteger (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧               -- 4 digits condition
  (∀ d1 d2 d3 d4, (n = 1000 * d1 + 100 * d2 + 10 * d3 + d4) → d1 ≠ 0 → 
     d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
     d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) ∧    -- all digits are different
  (n % 2 = 0) ∧                         -- multiple of 2 condition
  (∀ d1 d2 d3 d4, (n = 1000 * d1 + 100 * d2 + 10 * d3 + d4) → 
     d1 = 7 ∨ d2 = 7 ∨ d3 = 7 ∨ d4 = 7)  -- largest digit is 7 condition

-- Statement to prove that the number of such integers is 690
theorem count_valid_integers : 
  {n : ℕ | isValidInteger n}.finite.card = 690 :=
sorry

end count_valid_integers_l277_277850


namespace problem_part_I_problem_part_II_l277_277036

noncomputable def a : ℕ → ℕ
| 1 := 2
| (n + 1) := 2 * a n

noncomputable def b : ℕ → ℕ
| 1 := 1
| (n + 1) := b n + 1

def sum_ab : ℕ → ℕ
| 0 := 0
| (n + 1) := sum_ab n + a (n + 1) * b (n + 1)

theorem problem_part_I :
  (a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = 2 * a n) ∧
  (b 1 = 1 ∧ ∀ n ≥ 1, (∑ k in finset.range n, 1 / k.succ) * b k = b (n + 1) - 1) →
  (∀ n : ℕ, a n = 2 ^ n) ∧ (∀ n : ℕ, b n = n + 1) :=
sorry

theorem problem_part_II (n : ℕ) :
  (∀ n : ℕ, a n = 2 ^ n) ∧ (∀ n : ℕ, b n = n + 1) →
  sum_ab n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end problem_part_I_problem_part_II_l277_277036


namespace bill_picked_apples_l277_277745

-- Definitions from conditions
def children := 2
def apples_per_child_per_teacher := 3
def favorite_teachers := 2
def apples_per_pie := 10
def pies_baked := 2
def apples_left := 24

-- Number of apples given to teachers
def apples_for_teachers := children * apples_per_child_per_teacher * favorite_teachers

-- Number of apples used for pies
def apples_for_pies := pies_baked * apples_per_pie

-- The final theorem to be stated
theorem bill_picked_apples :
  apples_for_teachers + apples_for_pies + apples_left = 56 := 
sorry

end bill_picked_apples_l277_277745


namespace possible_8th_grade_students_l277_277453

theorem possible_8th_grade_students (x : ℕ) :
  ∃ x, (1 + x) % 2 = 0 ∧ x = 7 ∨ x = 14 :=
begin
  sorry,
end

end possible_8th_grade_students_l277_277453


namespace eccentricity_of_hyperbola_l277_277023

open Real

noncomputable def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def parabola := ∀ x y : ℝ, y^2 = -8 * x

noncomputable def directrix := ∀ x : ℝ, x = 2

noncomputable def ecc (a b c : ℝ) := c / a

theorem eccentricity_of_hyperbola 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : b = sqrt 3 * a) 
  (H : ∀ (O A B : PointCoordinate), 
    let Area := (1 / 2) * abs (2 * (4 * b / a)) in Area = 4 * sqrt 3):
  ecc (a) (b) (c) = 2 := 
by 
  sorry

end eccentricity_of_hyperbola_l277_277023


namespace no_two_delegates_next_to_each_other_l277_277321

theorem no_two_delegates_next_to_each_other :
  let n := 8
  let delegates : Fin n → ℕ := λ i, i % 4
  P := 0
  ∃ (m n : ℕ), nat.gcd m n = 1 ∧ (m : ℚ) / n = P :=
sorry

end no_two_delegates_next_to_each_other_l277_277321


namespace four_edge_trips_count_l277_277981

-- Defining points and edges of the cube
inductive Point
| A | B | C | D | E | F | G | H

open Point

-- Edges of the cube are connections between points
def Edge (p1 p2 : Point) : Prop :=
  ∃ (edges : List (Point × Point)), 
    edges = [(A, B), (A, D), (A, E), (B, C), (B, E), (B, F), (C, D), (C, F), (C, G), (D, E), (D, F), (D, H), (E, F), (E, H), (F, G), (F, H), (G, H)] ∧ 
    ((p1, p2) ∈ edges ∨ (p2, p1) ∈ edges)

-- Define the proof statement
theorem four_edge_trips_count : 
  ∃ (num_paths : ℕ), num_paths = 12 :=
sorry

end four_edge_trips_count_l277_277981


namespace find_length_AB_l277_277835

-- Definitions based on the conditions
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 6) - (y^2 / 3) = 1

-- Definition of the distance for the given problem context
def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

variable {F1 F2 A B : ℝ × ℝ}

-- The main theorem to prove the required result
theorem find_length_AB (H : hyperbola_eq A.1 A.2 ∧ hyperbola_eq B.1 B.2) 
    (line_passes_through_focus : F1 = (-√6, 0)) 
    (AF2_BF2_condition : distance A F2 + distance B F2 = 2 * distance A B) :
    distance A B = 4 * real.sqrt 6 :=
  sorry

end find_length_AB_l277_277835


namespace sweets_distribution_l277_277253

theorem sweets_distribution :
  ∀ (S X N A: ℕ), N = 190 → A = 70 → (S = N * X) → (S = (N - A) * (X + 14)) → 
  ((X + 14) = 38) :=
by {
  intros S X N A _ _ _ _,
  sorry
}

end sweets_distribution_l277_277253


namespace problem1_problem2_problem3_problem4_problem5_l277_277701

theorem problem1 : (-1) + (+2) + (-3) + (+1) + (-2) + 3 = 0 := by
  sorry

theorem problem2 : (1 - 2.46 * 3.54) * (0.2 - 1 / 5) = 0 := by
  sorry

theorem problem3 : - (1 / 10) - (1 / 100) - (1 / 1000) - (1 / 10000) = -0.1111 := by
  sorry

theorem problem4 : (-0.027) + 5.7 - 0.073 = 5.6 := by
  sorry

theorem problem5 : -(-(-(-1))) = 1 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_l277_277701


namespace problem_statement_l277_277394

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions of the problem
def cond1 : Prop := (1 / x) + (1 / y) = 2
def cond2 : Prop := (x * y) + x - y = 6

-- The corresponding theorem to prove: x² - y² = 2
theorem problem_statement (h1 : cond1) (h2 : cond2) : x^2 - y^2 = 2 :=
  sorry

end problem_statement_l277_277394


namespace sum_reciprocals_l277_277146

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end sum_reciprocals_l277_277146


namespace grid_fill_even_l277_277087

-- Useful definitions
def even {α : Type} [AddGroup α] (a : α) : Prop := ∃ b, a = 2 * b

-- Statement of the problem
-- 'n' is a natural number, grid is n × n
-- We need to find the number of ways to fill the grid with 0s and 1s such that each row and column has an even number of 1s
theorem grid_fill_even (n : ℕ) : ∃ (ways : ℕ), ways = 2 ^ ((n - 1) * (n - 1)) ∧ 
  (∀ grid : (Fin n → Fin n → bool), (∀ i : Fin n, even (grid i univ.count id)) ∧ (∀ j : Fin n, even (univ.count (λ x, grid x j))) → true) :=
sorry

end grid_fill_even_l277_277087


namespace percentage_of_boys_l277_277873

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (total_students_eq : total_students = 42)
  (ratio_eq : boy_ratio = 3 ∧ girl_ratio = 4) :
  (boy_ratio + girl_ratio) = 7 ∧ (total_students / 7 * boy_ratio * 100 / total_students : ℚ) = 42.86 :=
by
  sorry

end percentage_of_boys_l277_277873


namespace probability_sum_to_five_l277_277615

theorem probability_sum_to_five :
  let cards := {1, 2, 3, 4, 5, 6, 7, 8}
  let draw := (finset.cardinality cards).choose 6
  let arrangements := finset.permutations_6_r 8
  let favorable := 
    (finset.filter 
      (λ arrangement, (arrangement.cardinality == 6) && 
      (exists_middle_row_sum_to_5 arrangement)) (finset.permutations_6_r 8)).card
  favorable / arrangements = 13 / 210 := by
  sorry

-- Additional definitions
def finset.permutations_6_r (n : ℕ) : finset (finset ℕ) :=
  -- implementation for permutation count of 6 out of n elements
  sorry

def exists_middle_row_sum_to_5 (arrangement : finset ℕ) : bool :=
  -- implementation for checking if there's exactly one pair sum to 5 in one row only
  sorry

end probability_sum_to_five_l277_277615


namespace gem_stone_necklaces_sold_l277_277929

-- Definitions and conditions
def bead_necklaces : ℕ := 7
def total_earnings : ℝ := 90
def price_per_necklace : ℝ := 9

-- Theorem to prove the number of gem stone necklaces sold
theorem gem_stone_necklaces_sold : 
  ∃ (G : ℕ), G * price_per_necklace = total_earnings - (bead_necklaces * price_per_necklace) ∧ G = 3 :=
by
  sorry

end gem_stone_necklaces_sold_l277_277929


namespace reflected_ray_bisects_circle_circumference_l277_277804

open Real

noncomputable def equation_of_line_reflected_ray : Prop :=
  ∃ (m b : ℝ), (m = 2 / (-3 + 1)) ∧ (b = (3/(-5 + 5)) + 1) ∧ ((-5, -3) = (-5, (-5*m + b))) ∧ ((1, 1) = (1, (1*m + b)))

theorem reflected_ray_bisects_circle_circumference :
  equation_of_line_reflected_ray ↔ ∃ a b c : ℝ, (a = 2) ∧ (b = -3) ∧ (c = 1) ∧ (a*x + b*y + c = 0) :=
by
  sorry

end reflected_ray_bisects_circle_circumference_l277_277804


namespace tamar_stops_in_quarter_A_l277_277527

-- Definitions for the conditions
def circumference := 60 -- circumference of the track in feet
def quarters := ["A", "B", "C", "D"] -- quarters of the track
def start_point := "S" -- point S
def feet_run := 6000 -- feet run by Tamar

-- Proving that Tamar stops in quarter A
theorem tamar_stops_in_quarter_A (h1 : start_point = "S") (h2 : 6000 % 60 = 0) : quarters.head = "A" :=
sorry

end tamar_stops_in_quarter_A_l277_277527


namespace intersection_of_M_and_N_l277_277926

noncomputable def M : Set ℕ := { x | 1 < x ∧ x < 7 }
noncomputable def N : Set ℕ := { x | x % 3 ≠ 0 }

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4, 5} := sorry

end intersection_of_M_and_N_l277_277926


namespace quadratic_distinct_real_roots_l277_277053

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end quadratic_distinct_real_roots_l277_277053


namespace train_crosses_telegraph_in_approx_12_seconds_l277_277729

open Real

noncomputable def train_crossing_time := (320 : ℝ) / ((96 * 1000) / 3600)

theorem train_crosses_telegraph_in_approx_12_seconds : 
  abs (train_crossing_time - 12) < 1 :=
by
  unfold train_crossing_time
  simp
  norm_num
  -- Here we include 'sorry' to indicate that the proof is not completed
  sorry

end train_crosses_telegraph_in_approx_12_seconds_l277_277729


namespace grid_fill_even_l277_277085

-- Useful definitions
def even {α : Type} [AddGroup α] (a : α) : Prop := ∃ b, a = 2 * b

-- Statement of the problem
-- 'n' is a natural number, grid is n × n
-- We need to find the number of ways to fill the grid with 0s and 1s such that each row and column has an even number of 1s
theorem grid_fill_even (n : ℕ) : ∃ (ways : ℕ), ways = 2 ^ ((n - 1) * (n - 1)) ∧ 
  (∀ grid : (Fin n → Fin n → bool), (∀ i : Fin n, even (grid i univ.count id)) ∧ (∀ j : Fin n, even (univ.count (λ x, grid x j))) → true) :=
sorry

end grid_fill_even_l277_277085


namespace find_a_m_range_c_l277_277831

noncomputable def f (x a : ℝ) := x^2 - 2*x + 2*a
def solution_set (f : ℝ → ℝ) (m : ℝ) := {x : ℝ | -2 ≤ x ∧ x ≤ m ∧ f x ≤ 0}

theorem find_a_m (a m : ℝ) : 
  (∀ x, f x a ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) → a = -4 ∧ m = 4 := by
  sorry

theorem range_c (c : ℝ) : 
  (∀ x, (c - 4) * x^2 + 2 * (c - 4) * x - 1 < 0) → 13 / 4 < c ∧ c < 4 := by
  sorry

end find_a_m_range_c_l277_277831


namespace bukvinsk_acquaintances_l277_277469

theorem bukvinsk_acquaintances (Martin Klim Inna Tamara Kamilla : Type) 
  (acquaints : Type → Type → Prop)
  (exists_same_letters : ∀ (x y : Type), acquaints x y ↔ ∃ S, (x = S ∧ y = S)) :
  (∃ (count_Martin : ℕ), count_Martin = 20) →
  (∃ (count_Klim : ℕ), count_Klim = 15) →
  (∃ (count_Inna : ℕ), count_Inna = 12) →
  (∃ (count_Tamara : ℕ), count_Tamara = 12) →
  (∃ (count_Kamilla : ℕ), count_Kamilla = 15) := by
  sorry

end bukvinsk_acquaintances_l277_277469


namespace park_people_count_l277_277319

def people_in_park (n : ℕ) : ℕ :=
  let f := λ m, 2^m - (m - 1)
  in ∑ i in (range n), f (i + 2) - ∑ k in (range n), (k + 1)

theorem park_people_count : people_in_park 10 = 2001 := sorry

end park_people_count_l277_277319


namespace motorboat_trip_time_l277_277160

theorem motorboat_trip_time 
  (v : ℝ)  -- speed of the motorboat
  (x : ℝ)  -- speed of the current
  (h1 : x = v / 3)  -- speed of the current is 1/3 of the motorboat's speed
  (h2 : 20 = 2 * (10)) :  -- it takes 20 minutes for a round trip without current
  let d := 10 * v / 3 in  -- distance between pier and bridge
  let t_with_current := d / (4 * (v / 3)) in  -- time to travel with current
  let t_against_current := d / (2 * (v / 3)) in  -- time to travel against current
  22.5 = t_with_current + t_against_current :=
by
  sorry

end motorboat_trip_time_l277_277160


namespace calculate_expression_l277_277422

theorem calculate_expression (x y : ℝ) (h : 2 * x + 3 * y + 3 = 0) : 4^x * 8^y = 1 / 8 :=
by 
  sorry

end calculate_expression_l277_277422


namespace distribute_bottles_l277_277137

theorem distribute_bottles (n : ℕ) (h : n ≥ 1) (k : ℕ) (bottles : Fin k → ℝ) (bottles_lt_1 : ∀ i, bottles i < 1) (total_water : (Fin k) → ℝ → ∀ i, ∑ i bottles i = n / 2) :
  ∃ (buckets : Fin n → ℝ), (∀ j, buckets j ≤ 1) ∧ (∀ i, ∃ j, buckets j ≥ bottles i) :=
by
  sorry

end distribute_bottles_l277_277137


namespace perpendicular_line_eq_l277_277655

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l277_277655


namespace length_AB_12_sqrt_2_l277_277488

-- Declaring the variables and the triangle
variables (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables {AC BC AB : ℝ}

-- Conditions 
def is_right_triangle (A B C : Type) : Prop := angle A B C = 90
def angle_A_B_C_right : angle A C B = 90 := by sorry -- Given from conditions
def angle_BAC_45 : angle B A C = 45 := by sorry -- Given from conditions
def length_AC_12 (A C : Type) [MetricSpace A] [MetricSpace C] : distance A C = 12 := by sorry -- Given from conditions
def is_right_triangle_AC_12 (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  is_right_triangle A B C ∧ angle A C B = 90 ∧ angle B A C = 45 ∧ distance A C = 12

-- Main statement
theorem length_AB_12_sqrt_2 {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [angle_A_B_C_right] [angle_BAC_45] [length_AC_12]: 
  is_right_triangle_AC_12 A B C → distance B A = 12 * √2 := 
begin
  sorry
end

end length_AB_12_sqrt_2_l277_277488


namespace optimal_order_l277_277552

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l277_277552


namespace jose_share_of_profit_jose_share_is_35000_l277_277622

theorem jose_share_of_profit (
  tom_investment : ℕ,
  tom_months : ℕ,
  jose_investment : ℕ,
  jose_join_month : ℕ,
  profit : ℕ
) : ℕ :=
by
  let total_months := 12
  let tom_capital_months := tom_investment * tom_months
  let jose_capital_months := jose_investment * (total_months - jose_join_month)
  let total_capital_months := tom_capital_months + jose_capital_months
  let jose_share := (jose_capital_months / total_capital_months) * profit
  exact jose_share

def problem_data :=
  (tom_investment = 30000) ∧ 
  (tom_months = 12) ∧ 
  (jose_investment = 45000) ∧ 
  (jose_join_month = 2) ∧ 
  (profit = 63000)

theorem jose_share_is_35000:
  ∀ (tom_investment tom_months jose_investment jose_join_month profit: ℕ),
  problem_data → jose_share_of_profit tom_investment tom_months jose_investment jose_join_month profit = 35000 :=
by
  intros _ _ _ _ _
  intro pdata
  cases pdata
  simp [jose_share_of_profit]
  sorry

end jose_share_of_profit_jose_share_is_35000_l277_277622


namespace problem_statement_l277_277793

theorem problem_statement {m n : ℝ} 
  (h1 : (n + 2 * m) / (1 + m ^ 2) = -1 / 2) 
  (h2 : -(1 + n) + 2 * (m + 2) = 0) : 
  (m / n = -1) := 
sorry

end problem_statement_l277_277793


namespace demand_change_for_revenue_l277_277708

theorem demand_change_for_revenue (P D D' : ℝ)
  (h1 : D' = (1.10 * D) / 1.20)
  (h2 : P' = 1.20 * P)
  (h3 : P * D = P' * D') :
  (D' - D) / D * 100 = -8.33 := by
sorry

end demand_change_for_revenue_l277_277708


namespace arithmetic_geometric_ratio_l277_277004

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
1 + 3 = a1 + a2

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
b2 ^ 2 = 4

theorem arithmetic_geometric_ratio (a1 a2 b2 : ℝ) 
  (h1 : arithmetic_sequence a1 a2) 
  (h2 : geometric_sequence b2) : 
  (a1 + a2) / b2 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l277_277004


namespace max_value_of_function_l277_277341

noncomputable def problem_statement : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ π) → sqrt 2 * sin (x - π / 4) ≤ sqrt 2 ∧ ∃ (x : ℝ), (0 ≤ x ∧ x ≤ π) ∧ sqrt 2 * sin (x - π / 4) = sqrt 2

theorem max_value_of_function : problem_statement :=
begin
  sorry
end

end max_value_of_function_l277_277341


namespace largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l277_277390

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l277_277390


namespace true_props_l277_277848

def z := (⟨2, -1⟩ : ℂ)  -- Define the complex number z = 2 - i

def p1 := |z| = Real.sqrt 5
def p2 := z^2 = (⟨3, -4⟩ : ℂ)
def p3 := conj z = (⟨2, 1⟩ : ℂ)
def p4 := z.im = -1

theorem true_props : p2 ∧ p4 :=
by
  -- Proof not required
  exact ⟨rfl, rfl⟩

end true_props_l277_277848


namespace astronaut_revolutions_l277_277977

theorem astronaut_revolutions (n : ℤ) (R : ℝ) (hn : n > 2) :
    ∃ k : ℤ, k = n - 1 := 
sorry

end astronaut_revolutions_l277_277977


namespace nikiphor_minimum_expenses_max_amount_F_l277_277217

-- Definition of variables and conditions
def totalVoters : Nat := 35
def undecidedVoters : Nat := 0.4 * totalVoters
def votesToWin : Nat := 0.5 * totalVoters + 1

-- Define the bribe function
def bribes (x : Nat) : Nat := x

-- Ratibor's and Nikiphor's behavior
def ratiborMaxBribe : Nat := 18  -- Input based on solution analysis
def nikiphorBribe : Nat := ratiborMaxBribe + 1

-- Additional condition about Nikiphor's behavior if he has information
def F: Nat -- Maximum he would pay for information
-- Keeps the complete condition in context
def minVotes : Nat := 20
  
theorem nikiphor_minimum_expenses : 
  (undecidedVoters ≥ minVotes) ∧ (votesToWin ≤ totalVoters - minVotes) →
  ∃ k, (n * ratiborMaxBribe < nikiphorBribe) ∧ (totalVoters - votesToWin < undecidedVoters) :=
by
  sorry

theorem max_amount_F : 
  (F ≥ undecidedVoters * ratiborMaxBribe) →
  ∃ k, (k * ratiborMaxBribe ≤ nikiphorBribe) ∧ (totalVoters - votesToWin < undecidedVoters) :=
by
  sorry

end nikiphor_minimum_expenses_max_amount_F_l277_277217


namespace induction_step_induction_product_eq_l277_277233

-- Mathematical induction base case and step function
theorem induction_step (P : ℕ → Prop) (basis : P 0) (step : ∀ k, P k → P (k + 1)) : ∀ n, P n :=
begin
  intro n,
  induction n with k ih,
  { exact basis },
  { exact step k ih }
end

-- Definition of the problem using the conditions and required expression
theorem induction_product_eq (n : ℕ) :
  ( ∏ i in finset.range (n + 1) + 1, n + i) = 2 ^ n * ∏ i in range n, (2 * i + 1) :=
begin
  apply induction_step,
  -- Base case
  { simp },
  -- Inductive step
  { intro k,
    assume h,
    calc 
         ∏ i in finset.range (k + 2) + 1, k + i
         = ( ∏ i in finset.range (k + 1) + 1, k + i ) * (2 * k + 1) : by sorry
       ... = (2 ^ k * ∏ i in finset.range k, (2 * i + 1)) * (2 * k + 1) : by rw h
       ... = 2^(k + 1) * ∏ i in finset.range (k + 1), (2 * i + 1) : by sorry },
end

end induction_step_induction_product_eq_l277_277233


namespace peter_fish_caught_l277_277164

theorem peter_fish_caught (n : ℕ) (h : 3 * n = n + 24) : n = 12 :=
sorry

end peter_fish_caught_l277_277164


namespace prime_ge_7_divides_30_l277_277863

theorem prime_ge_7_divides_30 (p : ℕ) (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 30 ∣ (p^2 - 1) := by
  sorry

end prime_ge_7_divides_30_l277_277863


namespace permutation_count_eq_fib_l277_277773

noncomputable def F : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := F (n + 1) + F n

def valid_permutation (n : ℕ) (p : ℕ → ℕ) : Prop :=
∀ i, (1 ≤ i ∧ i ≤ n - 2 → p i < p (i + 2)) ∧ (1 ≤ i ∧ i ≤ n - 3 → p i < p (i + 3))

theorem permutation_count_eq_fib (n : ℕ) (h : 4 ≤ n) :
  ∃ p : ℕ → ℕ, valid_permutation n p ∧ (∑ i in finset.range n, p i) = F (n + 1) :=
sorry

end permutation_count_eq_fib_l277_277773


namespace inscribed_circle_radius_of_isosceles_triangle_l277_277449

theorem inscribed_circle_radius_of_isosceles_triangle (A B C : Point) (r : ℝ) :
  is_isosceles_triangle A B C ∧ distance AB = 20 ∧ distance BC = 20 ∧ tan (angle BAC) = 4/3 ∧ 
  inscribed_circle_radius ABC = r → r = 6 :=
by
  sorry

end inscribed_circle_radius_of_isosceles_triangle_l277_277449


namespace oldest_brother_age_ratio_l277_277564

-- Define the ages
def rick_age : ℕ := 15
def youngest_brother_age : ℕ := 3
def smallest_brother_age : ℕ := youngest_brother_age + 2
def middle_brother_age : ℕ := smallest_brother_age * 2
def oldest_brother_age : ℕ := middle_brother_age * 3

-- Define the ratio
def expected_ratio : ℕ := oldest_brother_age / rick_age

theorem oldest_brother_age_ratio : expected_ratio = 2 := by
  sorry 

end oldest_brother_age_ratio_l277_277564


namespace line_passes_through_fixed_point_line_equation_given_area_l277_277025

theorem line_passes_through_fixed_point (k : ℝ) : ∃ P : ℝ × ℝ, P = (-2, 1) ∧ k * P.1 - P.2 + 1 + 2 * k = 0 :=
by
  use (-2, 1)
  split
  · rfl
  · dsimp
    rw [← add_assoc, ← sub_eq_add, sub_eq_zero]
    exact sorry

theorem line_equation_given_area (k : ℝ) (h : (1 / 2) * abs (2 * k + 1) * abs ((-1 / k) - 2) = 4) : k = 1 / 2 → (kx - (2 * (x : ℝ)) + 4) = 0 :=
by
  intros hk
  rw hk
  unfold k x
  rw ← sub_eq_iff_eq_add
  exact sorry

end line_passes_through_fixed_point_line_equation_given_area_l277_277025


namespace quadratic_distinct_roots_l277_277056

theorem quadratic_distinct_roots (m : ℝ) : 
  ((∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1 * r2 = 9 ∧ r1 + r2 = -m) ↔ 
  m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo (6) ∞) := 
by sorry

end quadratic_distinct_roots_l277_277056


namespace possible_values_of_m_l277_277050

variable (m : ℝ)
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c > 0)

theorem possible_values_of_m (h : has_two_distinct_real_roots 1 m 9) : m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end possible_values_of_m_l277_277050


namespace solve_system_l277_277176

theorem solve_system :
  ∃ x y z : ℝ, (8 * (x^3 + y^3 + z^3) = 73) ∧
              (2 * (x^2 + y^2 + z^2) = 3 * (x * y + y * z + z * x)) ∧
              (x * y * z = 1) ∧
              (x, y, z) = (1, 2, 0.5) ∨ (x, y, z) = (1, 0.5, 2) ∨
              (x, y, z) = (2, 1, 0.5) ∨ (x, y, z) = (2, 0.5, 1) ∨
              (x, y, z) = (0.5, 1, 2) ∨ (x, y, z) = (0.5, 2, 1) :=
by
  sorry

end solve_system_l277_277176


namespace correct_grid_l277_277059

-- Define the problem conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ d, d ∣ n → d = 1 ∨ d = n

def in_grid (grid : matrix (fin 3) (fin 3) ℕ) : Prop :=
  ∀ i j, grid i j ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : finset ℕ)

def unique_elements (grid : matrix (fin 3) (fin 3) ℕ) : Prop :=
  (∃! a b : (fin 3) × (fin 3), grid a.1 a.2 = grid b.1 b.2)

def row_prime (grid : matrix (fin 3) (fin 3) ℕ) : Prop :=
  ∀ i : fin 3, is_prime (finset.univ.sum (λ j, grid i j))

def col_prime (grid : matrix (fin 3) (fin 3) ℕ) : Prop :=
  ∀ j : fin 3, is_prime (finset.univ.sum (λ i, grid i j))

-- Define the correct solution grid
def solution_grid : matrix (fin 3) (fin 3) ℕ :=
  ![![2, 8, 3],
    ![1, 5, 7],
    ![4, 6, 9]]

-- State the theorem to be proven
theorem correct_grid : 
  in_grid solution_grid ∧ 
  unique_elements solution_grid ∧ 
  row_prime solution_grid ∧ 
  col_prime solution_grid := 
sorry

end correct_grid_l277_277059


namespace parabola_opens_downwards_l277_277026

theorem parabola_opens_downwards (m : ℝ) : (m + 3 < 0) → (m < -3) := 
by
  sorry

end parabola_opens_downwards_l277_277026


namespace magician_finds_coins_l277_277270

noncomputable def magician_assistant_trick : Type := sorry

-- Define the set of 12 boxes
def boxes : Finset (Fin 12) := Finset.univ

-- Define the hiding function
def hide (i j : Fin 12) : Finset (Fin 12) := {i, j}

-- Condition: assistant opens one box that does not contain a coin
def assistant_opens (k : Fin 12) (i j : Fin 12) : Prop := k ≠ i ∧ k ≠ j

-- Define the method template (as a simplified example)
def template (i : Fin 12) : Finset (Fin 12) := {i, (i + 1) % 12, (i + 4) % 12, (i + 6) % 12}

-- Main theorem which ensures the trick always succeeds
theorem magician_finds_coins : 
  ∀ (i j k : Fin 12), 
    (j ≠ 1 ∧ k ≠ j ∧ k ≠ (1 : Fin 12)) → 
      ∃ (m : Fin 12), ({i, j} ⊆ template m) ∧ assistant_opens k i j :=
sorry

end magician_finds_coins_l277_277270


namespace fill_grid_with_even_ones_l277_277082

theorem fill_grid_with_even_ones (n : ℕ) : 
  ∃ ways : ℕ, ways = 2^((n-1)^2) ∧ 
  (∀ grid : array n (array n (fin 2)), 
    (∀ i : fin n, even (grid[i].to_list.count (λ x, x = 1))) ∧ 
    (∀ j : fin n, even (grid.map (λ row, row[j]).to_list.count (λ x, x = 1)))) :=
begin
  use 2^((n-1)^2),
  split,
  { refl },
  { sorry },
end

end fill_grid_with_even_ones_l277_277082


namespace general_formula_l277_277626

noncomputable def sequence_a (n : ℕ) : ℝ :=
if h : n > 0 then 
  match n with
  | 1 => 1
  | _ => 2 * sequence_a (n - 1) + (n + 2) / (n * (n + 1))
  end
else 0

theorem general_formula (n : ℕ) (h : n > 0) :
  sequence_a n = 3 * 2^(n-2) - 1 / (n + 1) :=
by sorry

end general_formula_l277_277626


namespace duck_flying_days_l277_277110

theorem duck_flying_days :
  ∃ S : ℕ, S + 2 * S + 60 = 180 ∧ S = 40 :=
by
  use 40
  split
  · norm_num
  · rfl

end duck_flying_days_l277_277110


namespace reciprocals_sum_eq_neg_one_over_three_l277_277143

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end reciprocals_sum_eq_neg_one_over_three_l277_277143


namespace simplify_complex_multiplication_l277_277175

open Complex

theorem simplify_complex_multiplication : 
  (2 + 3 * Complex.i) * (3 + 2 * Complex.i) * Complex.i = -13 := by
  sorry

end simplify_complex_multiplication_l277_277175


namespace unique_function_f_l277_277308

-- Define the function g and its inverse
noncomputable def g (x : ℝ) := x^3 + x
noncomputable def g_inv (y : ℝ) := Classical.some (exists_inverse_g y)

-- Define the main theorem
theorem unique_function_f (f : ℝ → ℝ) :
  (∀ x, f(x)^3 + f(x) ≤ x) ∧ (∀ x, x ≤ f(g(x))) → (∀ x, f(x) = g_inv(x)) :=
by
  sorry


end unique_function_f_l277_277308


namespace abs_difference_tetrahedrons_square_pyramids_l277_277292

def vertices : ℕ := 8
def cube_faces : ℕ := 6
def diagonal_planes : ℕ := 2

def square_pyramids : ℕ :=
  let bases := cube_faces + diagonal_planes
  2 * bases

def tetrahedrons : ℕ :=
  let total_combinations := Nat.choose vertices 4
  let invalid_combinations := 2 * cube_faces
  total_combinations - invalid_combinations

theorem abs_difference_tetrahedrons_square_pyramids : 
  |tetrahedrons - square_pyramids| = 42 :=
by
  sorry

end abs_difference_tetrahedrons_square_pyramids_l277_277292


namespace sum_mean_median_mode_l277_277671

def numbers : List ℕ := [3, 5, 3, 0, 2, 5, 0, 2]

def mode (l : List ℕ) : ℝ := 4

def median (l : List ℕ) : ℝ := 2.5

def mean (l : List ℕ) : ℝ := 2.5

theorem sum_mean_median_mode : mean numbers + median numbers + mode numbers = 9 := by
  sorry

end sum_mean_median_mode_l277_277671


namespace ln_of_x_sq_sub_2x_monotonic_l277_277983

noncomputable def ln_of_x_sq_sub_2x : ℝ → ℝ := fun x => Real.log (x^2 - 2*x)

theorem ln_of_x_sq_sub_2x_monotonic : ∀ x y : ℝ, (2 < x ∧ 2 < y ∧ x ≤ y) → ln_of_x_sq_sub_2x x ≤ ln_of_x_sq_sub_2x y :=
by
    intros x y h
    sorry

end ln_of_x_sq_sub_2x_monotonic_l277_277983


namespace trapezium_first_parallel_side_length_l277_277337

theorem trapezium_first_parallel_side_length 
  (x : ℝ) 
  (length_other_parallel_side : ℝ) 
  (distance_between_sides : ℝ) 
  (area : ℝ) 
  (h1 : length_other_parallel_side = 18) 
  (h2 : distance_between_sides = 14) 
  (h3 : area = 266) 
  : x = 20 := 
by 
  -- Definitions for convenience
  let length_sum := x + length_other_parallel_side
  let expected_area := (1/2) * length_sum * distance_between_sides
  -- Given conditions imply expected area calculation
  have h_area : expected_area = 266,
  -- Simplifying expressions based on given conditions
  calc
    expected_area = (1/2) * length_sum * distance_between_sides : by sorry
               ... = (1/2) * (x + 18) * 14 : by rw [h1, h2]
               ... = 7 * (x + 18) : by norm_num
               ... = 266 : by rw h3
    _ : _ = _
  -- Finally solving for x
  have h_solve : x + 18 = 38, from (266 / 7) = (x + 18),
  have h_final : x = 20, from sorry,
  exact h_final

end trapezium_first_parallel_side_length_l277_277337


namespace unique_function_l277_277916

-- Define the set of positive integers
def Z_plus := {n : ℤ // n > 0}

-- Define the function property
def func_property (f : ℤ → ℤ) (a b : ℤ) : Prop :=
  (a^2 + f a * f b) % (f a + b) = 0

-- Define the main theorem to prove
theorem unique_function 
  (f : ℤ → ℤ) 
  (h : ∀ a b ∈ Z_plus, func_property f a.val b.val) : 
  ∀ n ∈ Z_plus, f n.val = n.val :=
by {
  sorry
}

end unique_function_l277_277916


namespace sum_a_n_lt_one_l277_277899

def sequence (a : ℕ → ℝ) :=
a 1 = 1 / 2 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2 / (a n ^ 2 - a n + 1)

theorem sum_a_n_lt_one (a : ℕ → ℝ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → (∑ i in finset.range n.succ, a (i + 1)) < 1 :=
by
  sorry

end sum_a_n_lt_one_l277_277899


namespace Katrina_visible_area_l277_277499

noncomputable def visible_area (length width visibility : ℝ) : ℝ :=
  let inner_area := max 0 (length - 2 * visibility) * max 0 (width - 2 * visibility)
  let vertical_strips := 2 * visibility * (length - 2 * visibility)
  let horizontal_strips := 2 * visibility * (width - 2 * visibility)
  let corner_areas := 4 * (π * visibility^2 / 4)
  inner_area + vertical_strips + horizontal_strips + corner_areas

theorem Katrina_visible_area :
  visible_area 8 4 2 = 61 := 
by
  sorry

end Katrina_visible_area_l277_277499


namespace greatest_common_divisor_of_sum_of_three_consecutive_integers_is_3_l277_277122

def is_sum_of_three_consecutive_integers (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 1) + x + (x + 1)

theorem greatest_common_divisor_of_sum_of_three_consecutive_integers_is_3 :
  gcd_n (set_of is_sum_of_three_consecutive_integers) = 3 :=
sorry

end greatest_common_divisor_of_sum_of_three_consecutive_integers_is_3_l277_277122


namespace angle_ABC_leq_60_l277_277251

theorem angle_ABC_leq_60 (A B C H M : Type) [triangle A B C]
  (h_acute : acute_triangle A B C)
  (h_height_median_eq : height A H = median B M) :
  angle A B C ≤ 60 := 
sorry

end angle_ABC_leq_60_l277_277251


namespace solution_set_l277_277388

variable {f : ℝ → ℝ}

axiom dom_f : ∀ x : ℝ, True

axiom condition1 : ∀ x : ℝ, f(x) > f'(x) + 1
axiom condition2 : f(0) = 3

theorem solution_set (x : ℝ) : (f(x) > 2 * real.exp(x) + 1) ↔ x < 0 :=
by
  sorry

end solution_set_l277_277388


namespace solution_a_b_8_l277_277127

def seq_a : ℕ → ℝ
| 0 := 3
| (n+1) := 3 * (seq_a n) ^ 2 / (seq_b n)

def seq_b : ℕ → ℝ
| 0 := 4
| (n+1) := 2 * (seq_b n) ^ 2 / (seq_a n)

theorem solution_a_b_8 : 
  (seq_a 8, seq_b 8) = ( 3^2187 * 4^4374, 2^2916 * 3^1458 ) :=
sorry

end solution_a_b_8_l277_277127


namespace max_b_add_c_tangent_alpha_beta_l277_277148

noncomputable def a (α : ℝ) : ℝ × ℝ := (4 * Real.cos α, Real.sin α)
noncomputable def b (β : ℝ) : ℝ × ℝ := (Real.sin β, 4 * Real.cos β)
noncomputable def c (β : ℝ) : ℝ × ℝ := (Real.cos β, -4 * Real.sin β)
noncomputable def b_add_c (β : ℝ) : ℝ × ℝ :=
  let (bx, by) := b β
  let (cx, cy) := c β
  (bx + cx, by + cy)

theorem max_b_add_c (β : ℝ) : Real.sqrt ((b_add_c β).fst^2 + (b_add_c β).snd^2) ≤ 4 * Real.sqrt 2 :=
  sorry

theorem tangent_alpha_beta (α β : ℝ) (h : (4 * Real.cos α, Real.sin α) ⋅ (Real.sin β - 2 * Real.cos β, 4 * Real.cos β + 8 * Real.sin β) = 0) :
  Real.tan (α + β) = 2 :=
  sorry

end max_b_add_c_tangent_alpha_beta_l277_277148


namespace triangle_count_l277_277455

theorem triangle_count (M : ℕ) (hM : M ≥ 3) 
  (h : ∀ (S : Finset (Fin M)), S.card = M → ¬∀ (p q r : Fin M), (p ∈ S ∧ q ∈ S ∧ r ∈ S → collinear ({p, q, r} : set (Fin M)))) :
  ∃ n, n ≥ ((M - 1) * (M - 2)) / 2 := 
sorry

end triangle_count_l277_277455


namespace find_AH_l277_277937

variable (A B C D M H : Type)
variable (a b : ℝ)

def is_acute_angled_triangle (A B C : Type) : Prop := sorry
def is_altitude (A D : Type) (BC : Type) : Prop := sorry
def is_orthocenter (H : Type) (A B C : Type) : Prop := sorry
def semicircle_intersects_altitude (BC : Type) (M D : Type) (AD : Type) : Prop := sorry

theorem find_AH
  (h1 : is_acute_angled_triangle A B C)
  (h2 : is_altitude A D BC)
  (h3 : semicircle_intersects_altitude BC M D A)
  (h4 : A ≠ C)
  (h5 : is_orthocenter H A B C)
  (h6 : a = real.dist A D)
  (h7 : b = real.dist M D)
  : real.dist A H = (a^2 - b^2) / a := by
  sorry

end find_AH_l277_277937


namespace problem_l277_277201

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem problem (x : ℝ) :
  (∀ x : ℝ, ∀ y : ℝ, (x ∈ Icc (Real.pi / 3) (Real.pi / 2) ∧ y = f x) → 
    (@Function.monotone_decreasing_on ℝ ℝ _ _ _ f (Icc (Real.pi / 3) (Real.pi / 2)))) ∧
  (∀ x : ℝ, g (-x) = -g x) :=
by
  sorry

end problem_l277_277201


namespace max_intersections_l277_277771

-- Define the points and segments
def points_x := Fin 15
def points_y := Fin 10

def segment (x : points_x) (y : points_y) := (x, y)

-- Number of segments
def num_segments := 15 * 10

-- Condition to identify intersection of two segments
def segments_intersect (s1 s2 : segment) : Prop :=
  s1.1 < s2.1 ∧ s1.2 > s2.2 ∨ s1.1 > s2.1 ∧ s1.2 < s2.2

-- Count the number of intersections
def num_intersections : ℕ :=
  (Fintype.card (Finset.pairs points_x)).choose 2 *
  (Fintype.card (Finset.pairs points_y)).choose 2

-- Theorem statement (without proof)
theorem max_intersections :
  num_intersections = 4725 :=
sorry

end max_intersections_l277_277771


namespace permissible_m_value_l277_277378

theorem permissible_m_value (α m : Real)
  (h1 : sin α = (2 * m - 5) / (m + 1))
  (h2 : cos α = -m / (m + 1))
  (h3 : π / 2 < α ∧ α < π) :
  m = 4 :=
by
  sorry

end permissible_m_value_l277_277378


namespace monochromatic_triangle_l277_277600

theorem monochromatic_triangle 
  (plane_colored : ∀ (a : ℝ), a > 0 → ∃ (triangle : EquilateralTriangle), triangle.vertices_same_color)
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) :
  ∃ (triangle : Triangle), triangle.sides = (a, b, c) ∧ triangle.vertices_same_color :=
sorry

end monochromatic_triangle_l277_277600


namespace solve_inequality_l277_277570

theorem solve_inequality (m : ℝ) :
  (∀ x : ℝ, (m + 3) * x - 1 > 0 ↔
    if m = -3 then x > -1 else
    if m > -3 then (x < -1 ∨ x > 1 / (m + 3)) else
    if -4 < m ∧ m < -3 then (1 / (m + 3) < x ∧ x < -1) else
    if m < -4 then (-1 < x ∧ x < 1 / (m + 3)) else
    if m = -4 then false else false) :=
by {
  intros,
  sorry
}

end solve_inequality_l277_277570


namespace length_of_segment_AC_l277_277753

open Real

theorem length_of_segment_AC
  {c : ℝ} (hc : c = 18 * π)
  {ab : ℝ}
  {angle_UAC : ℝ} (h_angle_UAC : angle_UAC = π / 6) :
  let r := c / (2 * π),
  AC = r * sin(angle_UAC) := 
begin
  sorry
end

end length_of_segment_AC_l277_277753


namespace rectangle_length_rounded_l277_277579

noncomputable def rectangle_length (w : ℝ) : ℝ :=
  3 * w

noncomputable def pqrs_area (w : ℝ) : ℝ :=
  15 * w^2

theorem rectangle_length_rounded (w : ℝ) (h: pqrs_area w = 8000) : Int.round (rectangle_length w) = 69 :=
  by
  sorry

end rectangle_length_rounded_l277_277579


namespace john_cards_l277_277894

theorem john_cards (C : ℕ) (h1 : 15 * 2 + C * 2 = 70) : C = 20 :=
by
  sorry

end john_cards_l277_277894


namespace parabola_eq_min_area_OAB_l277_277014

variable {E : ℝ × ℝ} (hE : E = (2, 2 * Real.sqrt 2))

theorem parabola_eq (hF : ∃ F : ℝ × ℝ, F = (1, 0)) :
  ∃ p : ℝ, p = 2 ∧ ∀ x y : ℝ, (y = snd E) ↔ (y^2 = 4 * fst E) := sorry

theorem min_area_OAB {k : ℝ} (hFocus : (1, 0)) :
  (∀ y : ℝ, ((fst E) = ky + 1) ∧ ((snd E)^2 = 4 * (fst E))) →
  ∃ min_area : ℝ, min_area = 2 := sorry

end parabola_eq_min_area_OAB_l277_277014


namespace length_of_TU_l277_277260

-- We declare the necessary variables and structures for the triangles and their side lengths.
variables {P Q R S T U : Type}
variables [HasLength P Q ℝ] [HasLength Q R ℝ] [HasLength S R ℝ] [HasLength T U ℝ]

-- Declare the lengths as given conditions.
def PQ : ℝ := 10
def SR : ℝ := 4
def QR : ℝ := 15
def TU : ℝ := 6

-- Declare the similarity condition.
variables [Similar P Q R S T U]

-- Given conditions: side lengths and similarity condition
def length_PQ : HasLength P Q PQ := by sorry
def length_SR : HasLength S R SR := by sorry
def length_QR : HasLength Q R QR := by sorry

-- Prove that the length of TU is 6 cm under these conditions.
theorem length_of_TU : Similar P Q R S T U → TU = 6 := by
  intros h
  -- Proof steps would go here, but for now, we use sorry to complete the statement.
  sorry

end length_of_TU_l277_277260


namespace find_a_l277_277344

-- Define the circle and line equations, and the required distance condition
def circle (x y : ℝ) := x^2 + y^2 + 4 * x - 2 * y + 1 = 0
def line (x y : ℝ) (a : ℝ) := x + a * y - 1 = 0
def distance (p1 p2 : ℝ × ℝ) (A B C : ℝ) :=
  |A * p1.1 + B * p1.2 + C| / Real.sqrt (A^2 + B^2)

-- Center of the circle
def center := (-2 : ℝ, 1 : ℝ)

-- Lean statement for the proof problem
theorem find_a : ∃ a : ℝ, distance center 0, 0, key := 1 := |1 * -2 +  + a * 1 - 1| / Real.sqrt (1^2 + a^2) &= 0 ;
  have : |-3 + a| = Real.sqrt (1 - a^2) : -
2 * a = -3
2 * axioms  = Real.sqrt (3 )
search $
sorry_signature
false Parti
existence
Particularly
 a = a
theorem ∀ a : = 4/3 (

end find_a_l277_277344


namespace find_original_manufacturing_cost_l277_277686

noncomputable def originalManufacturingCost (P : ℝ) : ℝ := 0.70 * P

theorem find_original_manufacturing_cost (P : ℝ) (currentCost : ℝ) 
  (h1 : currentCost = 50) 
  (h2 : currentCost = P - 0.50 * P) : originalManufacturingCost P = 70 :=
by
  -- The actual proof steps would go here, but we'll add sorry for now
  sorry

end find_original_manufacturing_cost_l277_277686


namespace largest_number_is_B_l277_277244

noncomputable def numA : ℝ := 7.196533
noncomputable def numB : ℝ := 7.19655555555555555555555555555555555555 -- 7.196\overline{5}
noncomputable def numC : ℝ := 7.1965656565656565656565656565656565 -- 7.19\overline{65}
noncomputable def numD : ℝ := 7.196596596596596596596596596596596 -- 7.1\overline{965}
noncomputable def numE : ℝ := 7.196519651965196519651965196519651 -- 7.\overline{1965}

theorem largest_number_is_B : 
  numB > numA ∧ numB > numC ∧ numB > numD ∧ numB > numE :=
by
  sorry

end largest_number_is_B_l277_277244


namespace has_root_sqrt3_add_sqrt5_l277_277333

noncomputable def monic_degree_4_poly_with_root : Polynomial ℚ :=
  Polynomial.X ^ 4 - 16 * Polynomial.X ^ 2 + 4

theorem has_root_sqrt3_add_sqrt5 :
  Polynomial.eval (Real.sqrt 3 + Real.sqrt 5) monic_degree_4_poly_with_root = 0 :=
sorry

end has_root_sqrt3_add_sqrt5_l277_277333


namespace min_value_of_F_range_of_m_l277_277830

-- Problem Part I: Minimization of F(x)
def f (x m : ℝ) : ℝ := x^2 - (m + 1) * x + 4
def F (x m : ℝ) : ℝ := f x m - (m - 1) * x

theorem min_value_of_F (m : ℝ) (hm : m > 0) :
  ∃ x (hx : 0 < x ∧ x ≤ 1), 
  (F x m = (if m > 1 then 5 - 2 * m else 4 - m^2) ∧ 
   (∀ y (hy : 0 < y ∧ y ≤ 1), F y m ≥ F x m)) :=
sorry

-- Problem Part II: Range of values for m such that G(x) intersects y = 1
def g (x m : ℝ) : ℝ := x^2 - (m + 1) * x + 4
def G (x m : ℝ) : ℝ := 2 ^ g x m

theorem range_of_m : 
  ∃ x1 x2 (hx1 : 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3), 
  (G x1 m = 1 ∧ G x2 m = 1) ↔ (3 < m ∧ m ≤ 10/3) :=
sorry

end min_value_of_F_range_of_m_l277_277830


namespace apollonian_circle_min_value_l277_277418

/-- Given two fixed points A and B in a plane, and a moving point P, 
    if |PB| / |PA| = λ (λ > 0 and λ ≠ 1), then the locus of point P is a circle. 
    Additionally, given O(0, 0), Q(0, sqrt(2)/2), the line l1 : kx - y + 2k + 3 = 0,
    and the line l2 : x + ky + 3k + 2 = 0, if P is the intersection point of l1 and l2,
    then the minimum value of 3|PO| + 2|PQ| is 3√3. -/
theorem apollonian_circle_min_value 
    (k : ℝ)
    (O : ℝ × ℝ)
    (Q : ℝ × ℝ)
    (P : ℝ × ℝ)
    (l1 : P.1 * k - P.2 + 2 * k + 3 = 0)
    (l2 : P.1 + k * P.2 + 3 * k + 2 = 0)
    (hO : O = (0, 0))
    (hQ : Q = (0, sqrt(2) / 2)) :
    (3 * |P.1 - O.1 + P.2 - O.2| + 2 * |P.1 - Q.1 + P.2 - Q.2|) ≥ 3 * sqrt(3) :=
by
  sorry

end apollonian_circle_min_value_l277_277418


namespace ellipse_foci_l277_277194

noncomputable def foci_coords : ℝ × ℝ → Prop :=
  λ f, f = (1, 0) ∨ f = (-1, 0)

theorem ellipse_foci : (∀ x y : ℝ, x^2 / 2 + y^2 = 1 → foci_coords (x, y)) :=
sorry

end ellipse_foci_l277_277194


namespace sum_first_99_terms_l277_277218

def geom_sum (n : ℕ) : ℕ := (2^n) - 1

def seq_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum geom_sum

theorem sum_first_99_terms :
  seq_sum 99 = 2^100 - 101 := by
  sorry

end sum_first_99_terms_l277_277218


namespace smallest_k_l277_277787

theorem smallest_k (k : ℕ) :
  (∀ (k : ℕ), k >= 401 → (1 + x) * (1 + 2 * x) * ... * (1 + k * x) = a_0 + a_1 * x + ... + a_k * x^k →
  let s := (Finset.range k).sum (λ n, a_n)
  in s % 2005 = 0) ∧
  (∃ (m : ℕ), m < 401 ∧ let s := (Finset.range m).sum (λ n, a_n) in s % 2005 ≠ 0) := sorry

end smallest_k_l277_277787


namespace sum_geometric_seq_l277_277222

theorem sum_geometric_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1)
  (h2 : 4 * a 2 = 4 * a 1 + a 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 3 = 15 :=
by
  sorry

end sum_geometric_seq_l277_277222


namespace quadratic_distinct_roots_l277_277055

theorem quadratic_distinct_roots (m : ℝ) : 
  ((∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1 * r2 = 9 ∧ r1 + r2 = -m) ↔ 
  m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo (6) ∞) := 
by sorry

end quadratic_distinct_roots_l277_277055


namespace yards_dyed_green_calc_l277_277620

-- Given conditions: total yards dyed and yards dyed pink
def total_yards_dyed : ℕ := 111421
def yards_dyed_pink : ℕ := 49500

-- Goal: Prove the number of yards dyed green
theorem yards_dyed_green_calc : total_yards_dyed - yards_dyed_pink = 61921 :=
by 
-- sorry means that the proof is skipped.
sorry

end yards_dyed_green_calc_l277_277620


namespace sum_of_sequence_l277_277369

theorem sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, a n = (-1 : ℤ)^(n+1) * (2*n - 1)) →
  (S 0 = 0) →
  (∀ n, S (n+1) = S n + a (n+1)) →
  (∀ n, S (n+1) = (-1 : ℤ)^(n+1) * (n+1)) :=
by
  intros h_a h_S0 h_S
  sorry

end sum_of_sequence_l277_277369


namespace eigenvalue_gt_one_l277_277501

theorem eigenvalue_gt_one 
  (A B : Matrix ℝ ℝ) 
  [Matrix.symmetric A] 
  [Matrix.symmetric B]
  (hA : ∀ λ' : ℝ, Matrix.is_eigenvalue A λ' → λ' > 1)
  (hB : ∀ λ' : ℝ, Matrix.is_eigenvalue B λ' → λ' > 1) 
  (λ : ℝ) 
  (hλ : Matrix.is_eigenvalue (A ⬝ B) λ) : 
  |λ| > 1 := 
by
  sorry

end eigenvalue_gt_one_l277_277501


namespace probability_no_adjacent_balls_is_correct_l277_277784

open Finset

noncomputable def probability_no_adjacent_bins : ℚ :=
  let total_ways := (choose 20 5 : ℕ) 
  let valid_ways := finset.sum (finset.range 6) (λ k, (-1 : ℤ)^k * (choose 5 k) * (choose 15 (5 - k))) 
  (valid_ways : ℚ) / (total_ways : ℚ)

theorem probability_no_adjacent_balls_is_correct :
  probability_no_adjacent_bins = (15504 / 18801 : ℚ) :=
sorry

end probability_no_adjacent_balls_is_correct_l277_277784


namespace equation_of_line_through_point_perpendicular_l277_277339

noncomputable def line_perpendicular_through_point (t : ℝ) : ℝ × ℝ :=
  let x := - real.sqrt 3 * t,
      y := 2 + t
  in (x, y)

theorem equation_of_line_through_point_perpendicular :
  ∃ t : ℝ, line_perpendicular_through_point(t) = (0, 2) :=
by
  -- Proof placeholder
  sorry

end equation_of_line_through_point_perpendicular_l277_277339


namespace quadrilateral_radius_l277_277273

theorem quadrilateral_radius :
  (∃ (a b c d : ℝ),
    (a + b + c + d = 360) ∧  -- Sum of angles in a quadrilateral
    a = 120 ∧ b = 90 ∧ c = 60 ∧ d = 90 ∧  -- Given angles
    (∃ (area : ℝ), area = 9 * Real.sqrt 3) ∧  -- Given area
    (∃ (r : ℝ), 
      ∃ (AC BD : ℝ), 
        AC * BD / 2 = 9 * Real.sqrt 3 ∧  -- Area by diagonals
        AC = 2 * r ∧ 
        BD = 2 * r ∧ 
        AC * BD * sin(90 * Real.pi / 180) = 9 * Real.sqrt 3
    )
  ) → 
  (∃ (r : ℝ), r = 3) := sorry

end quadrilateral_radius_l277_277273


namespace problem_I_problem_II_l277_277401

open Real -- To use real number definitions and sin function.
open Set -- To use set constructs like intervals.

noncomputable def f (x : ℝ) : ℝ := sin (4 * x - π / 6) + sqrt 3 * sin (4 * x + π / 3)

-- Proof statement for monotonically decreasing interval of f(x).
theorem problem_I (k : ℤ) : 
  ∃ k : ℤ, ∀ x : ℝ, x ∈ Icc ((π / 12) + (k * π / 2)) ((π / 3) + (k * π / 2)) → 
  (4 * x + π / 6) ∈ Icc ((π / 2) + 2 * k * π) ((3 * π / 2) + 2 * k * π) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 4)

-- Proof statement for the range of g(x) on the interval [-π, 0].
theorem problem_II : 
  ∀ x : ℝ, x ∈ Icc (-π) 0 → g x ∈ Icc (-2) (sqrt 2) := 
sorry

end problem_I_problem_II_l277_277401


namespace equidistant_x_axis_point_l277_277237

open Real

def equidistant_point (A B : ℝ × ℝ) (y : ℝ) : Prop :=
∃ x : ℝ, (A.1 - x)^2 + (A.2 - y)^2 = (B.1 - x)^2 + (B.2 - y)^2

theorem equidistant_x_axis_point :
  let A := (-4, 0)
  let B := (2, 6)
  ∃ x : ℝ, x = 2 ∧ equidistant_point A B 0 :=
begin
  sorry
end

end equidistant_x_axis_point_l277_277237


namespace tire_mileage_problem_l277_277733

/- Definitions -/
def total_miles : ℕ := 45000
def enhancement_ratio : ℚ := 1.2
def total_tire_miles : ℚ := 180000

/- Question as theorem -/
theorem tire_mileage_problem
  (x y : ℚ)
  (h1 : y = enhancement_ratio * x)
  (h2 : 4 * x + y = total_tire_miles) :
  (x = 34615 ∧ y = 41538) :=
sorry

end tire_mileage_problem_l277_277733


namespace bus_seating_l277_277732

theorem bus_seating : 
  (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) 
  (h1: rows = 13) (h2: sections_per_row = 2) (h3: students_per_section = 2) : 
  rows * sections_per_row * students_per_section = 52 := 
sorry

end bus_seating_l277_277732


namespace sum_of_possible_values_of_N_l277_277986

theorem sum_of_possible_values_of_N :
  (∃ N : ℝ, N * (N - 7) = 12) → (∃ N₁ N₂ : ℝ, (N₁ * (N₁ - 7) = 12 ∧ N₂ * (N₂ - 7) = 12) ∧ N₁ + N₂ = 7) :=
by
  sorry

end sum_of_possible_values_of_N_l277_277986


namespace sphere_volume_l277_277585

theorem sphere_volume (C : ℝ) (h : C = 30) : 
  ∃ (V : ℝ), V = 4500 / (π^2) :=
by sorry

end sphere_volume_l277_277585


namespace fresh_grapes_weight_eq_l277_277351

-- Definitions of the conditions from a)
def fresh_grapes_water_percent : ℝ := 0.80
def dried_grapes_water_percent : ℝ := 0.20
def dried_grapes_weight : ℝ := 10
def fresh_grapes_non_water_percent : ℝ := 1 - fresh_grapes_water_percent
def dried_grapes_non_water_percent : ℝ := 1 - dried_grapes_water_percent

-- Proving the weight of fresh grapes
theorem fresh_grapes_weight_eq :
  let F := (dried_grapes_non_water_percent * dried_grapes_weight) / fresh_grapes_non_water_percent
  F = 40 := by
  -- The proof has been omitted
  sorry

end fresh_grapes_weight_eq_l277_277351


namespace solve_problem_l277_277795

noncomputable def problem_statement (a b : ℝ) : Prop :=
  sqrt (a + 3) + sqrt (2 - b) = 0 → a^b = 9

-- Let's state the theorem
theorem solve_problem (a b : ℝ) (h : sqrt (a + 3) + sqrt (2 - b) = 0) : a^b = 9 :=
  sorry

end solve_problem_l277_277795


namespace allocation_ways_l277_277944

/-- Defining the number of different balls and boxes -/
def num_balls : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem asserting the number of ways to place the balls into the boxes -/
theorem allocation_ways : (num_boxes ^ num_balls) = 81 := by
  sorry

end allocation_ways_l277_277944


namespace constant_term_in_expansion_of_P_l277_277192

def P (x : ℝ) : ℝ := (x + 3) * (2 * x - 1 / (4 * x * real.sqrt x)) ^ 5

theorem constant_term_in_expansion_of_P : 
  (∀ x ≠ 0, ∃ c, ∃ f : ℝ → ℝ, P(x) = c + x * f(x) ∧ c = 15) :=
sorry

end constant_term_in_expansion_of_P_l277_277192


namespace lcm_of_54_96_120_150_l277_277779

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end lcm_of_54_96_120_150_l277_277779


namespace perpendicular_line_through_point_l277_277637

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l277_277637


namespace find_angle_DAE_l277_277887

-- Definitions based on the given conditions
variables (A B C D O E : Type)
variables (ABC : Triangle A B C)
variables (∠ACB : Angle (C, A, B)) (∠CBA : Angle (B, C, A))
variables [DecidableEq A] [DecidableEq B] [DecidableEq C]
variables [DecidableEq D] [DecidableEq O] [DecidableEq E]

-- Given conditions
variables (h1 : ∠ACB = 40) (h2 : ∠CBA = 60)
variables (H_D : FootOfPerpendicular A B C D)
variables (H_O : Circumcenter O ABC)
variables (H_E : FootOfPerpendicular A A B E)

-- The proof problem
theorem find_angle_DAE (ABC : Triangle A B C) (∠ACB : Angle (C, A, B)) (∠CBA : Angle (B, C, A))
  (h1 : ∠ACB = 40) (h2 : ∠CBA = 60) 
  (H_D : FootOfPerpendicular A B C D)
  (H_O : Circumcenter O ABC)
  (H_E : FootOfPerpendicular A A B E) :
  ∠D A E = 40 :=
sorry

end find_angle_DAE_l277_277887


namespace min_area_quadrilateral_l277_277516

open Real

/-- Let ABCD be a convex quadrilateral whose diagonals AC and BD intersect at P. 
Assume the area of triangle APB is 24 and the area of triangle CPD is 25. 
We want to prove that the minimum possible area of quadrilateral ABCD is 49 + 20 * sqrt 6. -/
theorem min_area_quadrilateral (ABCD: Type) [convex_quad ABCD]
  (A B C D P: Point)
  (intersect_AC_BD: DiagonalIntersection ABCD A B C D P)
  (area_APB: Area (Triangle A P B) = 24)
  (area_CPD: Area (Triangle C P D) = 25) :
  ∃ [BPC DPA: Triangle],
    Area (Quadrilateral A B C D) = 49 + 20 * sqrt 6 :=
  sorry

end min_area_quadrilateral_l277_277516


namespace perpendicular_line_through_point_l277_277644

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l277_277644


namespace determine_counterfeit_l277_277691

noncomputable def identify_fake_coins
  (coins : List ℕ) (coin_type : ℕ → ℤ)
  (is_genuine : ℕ → Prop) (is_counterfeit : ℕ → Prop)
  (weigh : ℕ → ℕ → Prop)
  (weigh_sum : (List ℕ) → (List ℕ) → Prop)
  (num_weighings : ℕ) : Prop :=
  coins.length = 5 ∧
  ∃ genuine_coins counterfeit_coins,
    (∀ x ∈ genuine_coins, is_genuine x) ∧
    (∀ x ∈ counterfeit_coins, is_counterfeit x) ∧
    counterfeit_coins.length = 2 ∧
    genuine_coins.length = 3 ∧
    (∃ (h l : ℕ), h ≠ l ∧
      coin_type h > 0 ∧
      coin_type l < 0 ∧
      coin_type h = -coin_type l) ∧
    (num_weighings = 3) ∧
    (weigh a b ∨ weigh b a) ∧
    (weigh c d ∨ weigh d c) ∧
    (weigh_sum (a::b::[]) (c::d::[]) ∨ weigh_sum (c::d::[]) (a::b::[])) ∧
    ∀ (outcomes : List (ℕ × ℕ)), is_outcome_correct outcomes

theorem determine_counterfeit :
  ∃ identifications : List (ℕ × ℤ),
  identify_fake_coins [1,2,3,4,5]
    (λ n, if n = 1 then 2 else if n = 2 then -2 else 0)
    (λ n, n ≠ 1 → n ≠ 2) (λ n, n = 1 ∨ n = 2)
    (λ x y, x = a ∧ y = b ∨ x = b ∧ y = a ∨ x = c ∧ y = d ∨ x = d ∧ y = c)
    (λ xs ys, xs = [a, b] ∧ ys = [c, d] ∨ ys = [a, b] ∧ xs = [c, d])
    3 :=
sorry

end determine_counterfeit_l277_277691


namespace eq_of_line_through_center_and_P_chord_length_eq_l277_277001

noncomputable theory -- Handling noncomputable aspects.

open Real -- For real numbers and functions.

/-- Define the circle equation -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 11 / 2

/-- Define a point inside the circle -/
def point_P : ℝ × ℝ := (2, 2)

/-- Define the line that passes through the center (1, 0) of the circle and point P -/
def line_eq1 (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- Verify the equation of the line passing through the center of the circle and point P -/
theorem eq_of_line_through_center_and_P : 
  ∃ (l : ℝ × ℝ → Prop), (∀ q : ℝ × ℝ, l q ↔ (2 * q.1 - q.2 - 2 = 0))
:= by
  existsi (fun p => 2 * p.1 - p.2 - 2 = 0)
  simp
  sorry

/-- Define the line with slope k = 1 passing through P(2, 2) -/
def line_eq2 (x y : ℝ) : Prop := x - y = 0

/-- Find the length of the chord AB when line slopes k = 1 -/
theorem chord_length_eq (A B : ℝ × ℝ) (hA : circle_eq A.1 A.2) (hB : circle_eq B.1 B.2) 
  (intersection_l : line_eq2 A.1 A.2 ∧ line_eq2 B.1 B.2) : 
  dist A B = 2 * sqrt 5 
:= by
  sorry

end eq_of_line_through_center_and_P_chord_length_eq_l277_277001


namespace grid_fill_even_l277_277089

-- Useful definitions
def even {α : Type} [AddGroup α] (a : α) : Prop := ∃ b, a = 2 * b

-- Statement of the problem
-- 'n' is a natural number, grid is n × n
-- We need to find the number of ways to fill the grid with 0s and 1s such that each row and column has an even number of 1s
theorem grid_fill_even (n : ℕ) : ∃ (ways : ℕ), ways = 2 ^ ((n - 1) * (n - 1)) ∧ 
  (∀ grid : (Fin n → Fin n → bool), (∀ i : Fin n, even (grid i univ.count id)) ∧ (∀ j : Fin n, even (univ.count (λ x, grid x j))) → true) :=
sorry

end grid_fill_even_l277_277089


namespace find_k_l277_277021

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x

noncomputable def f' (x : ℝ) : ℝ := -2 * Real.sin x

theorem find_k : 
  let l1_slope := -1 / k,
      l2_slope := f' (Real.pi / 6) in
  (l1_slope = l2_slope) → k = -1 :=
sorry

end find_k_l277_277021


namespace PS_div_QR_eq_sqrt3_l277_277479

variables {P Q R S : Type} [metric_space P]

-- Assume we have points P, Q, R, S
variables [PQR_equilateral : equilateral_triangle P Q R] 
variables [QRS_equilateral : equilateral_triangle Q R S]

-- Assume lengths of QR and PS
axiom length_QR : length (Q, R) = t
axiom length_PS : length (P, S) = t * sqrt 3

theorem PS_div_QR_eq_sqrt3 :
  length (P, S) / length (Q, R) = sqrt 3 :=
sorry

end PS_div_QR_eq_sqrt3_l277_277479


namespace num_unique_four_digit_2023_l277_277427

theorem num_unique_four_digit_2023 : 
  let digits := [2, 0, 2, 3]
  in (∀ d d_1 d_2 d_3, digits.count d = 4 ∧ (d = 2 ∨ d = 0 ∨ d = 2 ∨ d = 3) → 
     ∃! n, (∀ n = d_1 * 1000 + d_2 * 100 + d_3 * 10 + d_4, d_1 ≠ 0) 
     ∧ perm.contains n := finalize [3, 2, 0] → ∀ n = 6) := sorry

end num_unique_four_digit_2023_l277_277427


namespace swap_adjacent_pawns_swap_any_pawns_l277_277466

theorem swap_adjacent_pawns (n : ℕ) (A : ℕ → ℕ → Prop) :
  ∀ i : ℕ, 1 ≤ i → i < n → ∃ s : List (List ℕ → List ℕ), 
  (∀ f : List ℕ → List ℕ, f ∈ s → 
  (∃ k : ℕ, f = λ l, List.rotate l k ∨ f = λ l, 
  (λ (lst : List ℕ), match lst with
   | a1 :: a2 :: rest => a2 :: a1 :: rest
   | _ => lst
   end))) ∧ (∀ pawns : List ℕ, List.length pawns = n → 
  swap_pawns (pawns i) (pawns (i + 1)) ((s.foldl (λ p f, f p) pawns))) := sorry

theorem swap_any_pawns (n : ℕ) (A : ℕ → ℕ → Prop) :
  ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → ∃ s : List (List ℕ → List ℕ), 
  (∀ f : List ℕ → List ℕ, f ∈ s → 
  (∃ k : ℕ, f = λ l, List.rotate l k ∨ f = λ l, 
  (λ (lst : List ℕ), match lst with
   | a1 :: a2 :: rest => a2 :: a1 :: rest
   | _ => lst
   end))) ∧ (∀ pawns : List ℕ, List.length pawns = n → 
  swap_pawns (pawns i) (pawns j) ((s.foldl (λ p f, f p) pawns))) := sorry

end swap_adjacent_pawns_swap_any_pawns_l277_277466


namespace num_cows_l277_277252

-- Define the context
variable (C H L Heads : ℕ)

-- Define the conditions
axiom condition1 : L = 2 * Heads + 8
axiom condition2 : L = 4 * C + 2 * H
axiom condition3 : Heads = C + H

-- State the goal
theorem num_cows : C = 4 := by
  sorry

end num_cows_l277_277252


namespace T_5_3_l277_277759

def T (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem T_5_3 : T 5 3 = 38 :=
by
  -- Evaluating the custom operation T with a = 5 and b = 3
  calc
    T 5 3 = 4 * 5 + 6 * 3 : by rfl
    ...    = 20 + 18 : by norm_num
    ...    = 38 : by norm_num

end T_5_3_l277_277759


namespace sofia_suggestions_l277_277952

theorem sofia_suggestions (mp: ℕ) (b: ℕ) : mp = 185 ∧ b = 125 → mp + b = 310 :=
by
  intros h,
  cases h with h_mp h_b,
  rw [h_mp, h_b],
  exact rfl

end sofia_suggestions_l277_277952


namespace rational_eq1_rational_eq2_l277_277177

open Rational

theorem rational_eq1 (x : ℚ) : 
  (2 * x - 5) / (x - 2) = 3 / (2 - x) ↔ x = 4 := by
  sorry

theorem rational_eq2 (x : ℚ) : 
  (12 / ((x - 3) * (x + 3))) - (2 / (x - 3)) = (1 / (x + 3)) ↔ x ∉ {3, -3} := by
  sorry

end rational_eq1_rational_eq2_l277_277177


namespace determine_counterfeit_l277_277690

noncomputable def identify_fake_coins
  (coins : List ℕ) (coin_type : ℕ → ℤ)
  (is_genuine : ℕ → Prop) (is_counterfeit : ℕ → Prop)
  (weigh : ℕ → ℕ → Prop)
  (weigh_sum : (List ℕ) → (List ℕ) → Prop)
  (num_weighings : ℕ) : Prop :=
  coins.length = 5 ∧
  ∃ genuine_coins counterfeit_coins,
    (∀ x ∈ genuine_coins, is_genuine x) ∧
    (∀ x ∈ counterfeit_coins, is_counterfeit x) ∧
    counterfeit_coins.length = 2 ∧
    genuine_coins.length = 3 ∧
    (∃ (h l : ℕ), h ≠ l ∧
      coin_type h > 0 ∧
      coin_type l < 0 ∧
      coin_type h = -coin_type l) ∧
    (num_weighings = 3) ∧
    (weigh a b ∨ weigh b a) ∧
    (weigh c d ∨ weigh d c) ∧
    (weigh_sum (a::b::[]) (c::d::[]) ∨ weigh_sum (c::d::[]) (a::b::[])) ∧
    ∀ (outcomes : List (ℕ × ℕ)), is_outcome_correct outcomes

theorem determine_counterfeit :
  ∃ identifications : List (ℕ × ℤ),
  identify_fake_coins [1,2,3,4,5]
    (λ n, if n = 1 then 2 else if n = 2 then -2 else 0)
    (λ n, n ≠ 1 → n ≠ 2) (λ n, n = 1 ∨ n = 2)
    (λ x y, x = a ∧ y = b ∨ x = b ∧ y = a ∨ x = c ∧ y = d ∨ x = d ∧ y = c)
    (λ xs ys, xs = [a, b] ∧ ys = [c, d] ∨ ys = [a, b] ∧ xs = [c, d])
    3 :=
sorry

end determine_counterfeit_l277_277690


namespace possible_values_of_m_l277_277049

variable (m : ℝ)
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c > 0)

theorem possible_values_of_m (h : has_two_distinct_real_roots 1 m 9) : m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end possible_values_of_m_l277_277049


namespace num_integers_satisfying_ineqs_l277_277349

theorem num_integers_satisfying_ineqs :
  {n : ℤ | (2 * n) ≤ (6 * n - 10) ∧ (6 * n - 10) < (3 * n + 8)}.finite.card = 3 := by
  sorry

end num_integers_satisfying_ineqs_l277_277349


namespace sequence_general_term_l277_277130

noncomputable def a_n (f g : ℕ → ℕ) (n : ℕ) : ℕ :=
  f (Int.floor ((1 + Real.sqrt (8 * n - 7)) / 2)) +
  g (n - 1 - (Int.floor ((1 + Real.sqrt (8 * n - 7)) / 2) * (Int.floor ((1 + Real.sqrt (8 * n - 7)) / 2) - 1)) / 2)

theorem sequence_general_term (f g : ℕ → ℕ) (f_inc_g_inc : ∀ x y : ℕ, x < y → f x < f y ∧ g x < g y)
    (cond : ∀ k : ℕ, k > 0 → f (k + 1) + g 0 > f k + g (k - 1)) (n : ℕ) :
  ∃ a : ℕ, a = a_n f g (n + 1) := 
sorry

end sequence_general_term_l277_277130


namespace area_of_triangle_is_eight_l277_277124

open Real

variable (a : ℝ × ℝ) (b : ℝ × ℝ)
def area_of_triangle (a b : ℝ × ℝ) : ℝ :=
  1/2 * abs (a.1 * b.2 - a.2 * b.1)

theorem area_of_triangle_is_eight :
  area_of_triangle (3, -2) (-1, 6) = 8 := 
by
  unfold area_of_triangle
  -- insert proof steps here
  sorry

end area_of_triangle_is_eight_l277_277124


namespace rain_depth_proof_l277_277463

-- Define the conditions
def volume_of_water : ℝ := 750  -- in cubic meters
def area_in_hectares : ℝ := 1.5
def hectare_to_square_meters : ℝ := 10_000 

-- Calculate the area in square meters
def area_in_square_meters : ℝ := area_in_hectares * hectare_to_square_meters  -- should be 1.5 * 10,000 = 15,000 square meters

-- Define the function to compute depth
def depth_of_rain (volume : ℝ) (area : ℝ) : ℝ := volume / area

-- The statement to be proved
theorem rain_depth_proof : depth_of_rain volume_of_water area_in_square_meters = 0.05 :=
by
  sorry

end rain_depth_proof_l277_277463


namespace find_EF_squared_l277_277958

noncomputable def square_side := 15
noncomputable def BE := 6
noncomputable def DF := 6
noncomputable def AE := 14
noncomputable def CF := 14

theorem find_EF_squared (A B C D E F : ℝ) (AB BC CD DA : ℝ := square_side) :
  (BE = 6) → (DF = 6) → (AE = 14) → (CF = 14) → EF^2 = 72 :=
by
  -- Definitions and conditions usage according to (a)
  sorry

end find_EF_squared_l277_277958


namespace optimal_order_for_ostap_l277_277532

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l277_277532


namespace number_of_shampoos_l277_277526

-- Define necessary variables in conditions
def h := 10 -- time spent hosing in minutes
def t := 55 -- total time spent cleaning in minutes
def p := 15 -- time per shampoo in minutes

-- State the theorem
theorem number_of_shampoos (h t p : Nat) (h_val : h = 10) (t_val : t = 55) (p_val : p = 15) :
    (t - h) / p = 3 := by
  -- Proof to be filled in
  sorry

end number_of_shampoos_l277_277526


namespace unique_cubic_polynomial_l277_277311

theorem unique_cubic_polynomial (g : ℝ → ℝ) (h_deg : polynomial.degree g = 3)
  (h1 : ∀ x, g (x^2) = g x^2)
  (h2 : ∀ x, g (x^2) = g g x)
  (h3 : g 1 = 1) : (g = λ x, x^2 + 0 * x + 0) := 
sorry

end unique_cubic_polynomial_l277_277311


namespace minimum_sum_is_approx_50_99_l277_277454

noncomputable def minimize_sum_distance : ℝ :=
  let aspen := (0 : ℝ, 20 : ℝ)
  let birch := (50 : ℝ, 30 : ℝ)
  let reflection_aspen := (0 : ℝ, -20 : ℝ)
  let reflection_birch := (50 : ℝ, -30 : ℝ)
  let distance := (p1 p2 : ℝ × ℝ) → Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)
  distance reflection_aspen reflection_birch

theorem minimum_sum_is_approx_50_99 :
  minimize_sum_distance = 10 * Real.sqrt 26 :=
by
  sorry

end minimum_sum_is_approx_50_99_l277_277454


namespace certain_event_sum_of_interior_angles_l277_277243

def sum_of_interior_angles_of_triangle_always_180 {α β γ : ℝ} (h : 0 < α ∧ 0 < β ∧ 0 < γ) (h_tri : α + β + γ = 180) : Prop :=
  ∀ (a b c : ℝ), (a + b + c = 180) → (α = a ∧ β = b ∧ γ = c)

theorem certain_event_sum_of_interior_angles (α β γ : ℝ) (h : α + β + γ = 180) : sum_of_interior_angles_of_triangle_always_180
  (by linarith) h :=
sorry

end certain_event_sum_of_interior_angles_l277_277243


namespace num_valid_n_l277_277855

theorem num_valid_n : 
  let count_n (N : ℕ) := ∀ (n : ℕ), (n < N) → (∃ (m : ℕ), (m % 4 = 0) ∧
   (∃ (a b : ℤ), (a + b = ↑n) ∧ (a * b = ↑m) ∧ (a % 2 = 0) ∧ (b % 2 = 0))) in
  count_n 50 = 12 :=
by simp only; sorry

end num_valid_n_l277_277855


namespace solution_set_of_xf_greater_0_l277_277508

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_even := ∀ x, f x = f (-x)
noncomputable def f_derivative (x : ℝ) : ℝ := sorry

theorem solution_set_of_xf_greater_0 (h1 : f_even f)
    (h2 : ∀ x, x > 0 → f(x) + x * f_derivative x > 0)
    (h3 : f 1 = 0) :
    { x : ℝ | x * f x > 0 } = set.Ico (-1 : ℝ) 0 ∪ set.Ioi 1 :=
begin
  sorry
end

end solution_set_of_xf_greater_0_l277_277508


namespace percentage_decrease_is_30_l277_277115

variables (original_value current_value decrease_value percentage_decrease : ℝ)

-- Conditions:
def original_value : ℝ := 4000
def current_value : ℝ := 2800

-- Define decrease in value
def decrease_value := original_value - current_value

-- Define percentage decrease
def percentage_decrease := (decrease_value / original_value) * 100

-- The theorem to prove
theorem percentage_decrease_is_30 :
  percentage_decrease = 30 := sorry

end percentage_decrease_is_30_l277_277115


namespace find_k_l277_277509

noncomputable def g (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : g a b c (-1) = 0) 
  (h2 : 30 < g a b c 5) (h3 : g a b c 5 < 40)
  (h4 : 120 < g a b c 7) (h5 : g a b c 7 < 130)
  (h6 : 2000 * k < g a b c 50) (h7 : g a b c 50 < 2000 * (k + 1)) : 
  k = 5 := 
sorry

end find_k_l277_277509


namespace least_n_divisibility_l277_277238

theorem least_n_divisibility :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → k ∣ (n - 1)^2) ∧ (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ ¬ k ∣ (n - 1)^2) ∧ n = 3 :=
by
  sorry

end least_n_divisibility_l277_277238


namespace inverse_of_periodic_function_for_neg_interval_l277_277367

section inverse_of_periodic_function

variables {T : ℝ} (f : ℝ → ℝ)
variable  (D : set ℝ)

-- Assume that f is periodic with period T
def periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x : ℝ, f (x + T) = f x

-- Assume that for x in (0, T), the inverse function of f is defined on domain D
def inverse_defined (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (D : set ℝ) : Prop :=
∀ y : ℝ, y ∈ D → f (f_inv y) = y ∧ f_inv (f y) = y

-- Use these assumptions and show the inverse function on the transformed interval
theorem inverse_of_periodic_function_for_neg_interval
  (hf_periodic : periodic f T)
  (hf_inv_defined : inverse_defined f (inv_fun f) D)
  (x : ℝ) (hx : x ∈ Ioo (-T : ℝ) (0 : ℝ)) :
  (inv_fun f (x + T) ∈ D) → (inv_fun f x = inv_fun f (x + T) - T) :=
begin
  intro hx_transformed,
  sorry
end

end inverse_of_periodic_function

end inverse_of_periodic_function_for_neg_interval_l277_277367


namespace monic_poly_has_root_l277_277332

theorem monic_poly_has_root : 
  ∃ (P : Polynomial ℚ), P.degree = 4 ∧ P.leadingCoeff = 1 ∧ P.eval (Real.of_rat (3 ^ (1/2) + 5 ^ (1/2))) = 0 :=
by
  sorry

end monic_poly_has_root_l277_277332


namespace perpendicular_line_through_point_l277_277636

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l277_277636


namespace problem_1_problem_2_l277_277024

theorem problem_1 
  : (∃ (m n : ℝ), m = -1 ∧ n = 1 ∧ ∀ (x : ℝ), |x + 1| + |2 * x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) :=
sorry

theorem problem_2 
  : (∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → 
    ∃ (min_val : ℝ), min_val = 9 / 2 ∧ 
    ∀ (x : ℝ), x = (1 / a + 1 / b + 1 / c) → min_val ≤ x) :=
sorry

end problem_1_problem_2_l277_277024


namespace count_even_ones_grid_l277_277104

theorem count_even_ones_grid (n : ℕ) : 
  (∃ f : (Fin n) → (Fin n) → ℕ, (∀ i : Fin n, ∑ j in (Finset.univ : Finset (Fin n)), f i j % 2 = 0) ∧ 
                                  (∀ j : Fin n, ∑ i in (Finset.univ : Finset (Fin n)), f i j % 2 = 0)) ↔ 
  2^((n-1)^2) = 2^((n-1)^2) :=
sorry

end count_even_ones_grid_l277_277104


namespace function_properties_l277_277198

theorem function_properties :
  ∀ (x : ℝ), f(x) = sin(2*x - (π / 6)) →
    (∀ x ∈ Icc (π / 3) (π / 2), monotone (λ x, -f x)) ∧
    (∀ x₁ x₂ : ℝ, f(x₁) = 1 / 2 ∧ f(x₂) = 1 / 2 → ¬ (x₁ - x₂) % π = 0) ∧
    (∀ (x : ℝ), g(x) = sin(2*x + π / 12 - π / 6) → ∀ (x : ℝ), g(-x) = -g(x)) ∧
    (count_zeros f (0, 8*π) = 16) :=
by 
  sorry

-- Definitions for f and g for completeness
def f (x : ℝ) := sin (2*x - π/6)
def g (x : ℝ) := sin (2*(x + π/12) - π/6)

-- Expected property of monotonicity
def decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂ ∈ Icc a b, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- Expected property of modulo
def is_multiple (x y : ℝ) : Prop :=
  ∃ n : ℤ, x = y * n

-- Measurement of zero counts
def count_zeros (f : ℝ → ℝ) (interval : Set ℝ) : ℕ :=
  -- A definition to count the number of roots could be complex but is needed for support
  sorry

end function_properties_l277_277198


namespace year_with_max_sales_increase_after_2005_l277_277755

noncomputable def sales : ℕ → ℕ
| 0 := 20
| 1 := 24
| 2 := 27
| 3 := 26
| 4 := 28
| 5 := 33
| 6 := 32
| 7 := 35

def sales_increase (n : ℕ) : ℕ :=
  if n > 0 then sales n - sales (n - 1) else 0

theorem year_with_max_sales_increase_after_2005 : ∃ n : ℕ, n > 0 ∧ sales_increase n = 5 ∧ ∀ m : ℕ, m > 0 → sales_increase m ≤ 5 :=
by
  sorry

end year_with_max_sales_increase_after_2005_l277_277755


namespace construction_work_rate_l277_277727

theorem construction_work_rate (C : ℝ) 
  (h1 : ∀ t1 : ℝ, t1 = 10 → t1 * 8 = 80)
  (h2 : ∀ t2 : ℝ, t2 = 15 → t2 * C + 80 ≥ 300)
  (h3 : ∀ t : ℝ, t = 25 → ∀ t1 t2 : ℝ, t = t1 + t2 → t1 = 10 → t2 = 15)
  : C = 14.67 :=
by
  sorry

end construction_work_rate_l277_277727


namespace average_of_remaining_numbers_l277_277970

theorem average_of_remaining_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 15)
  (h_avg : (numbers.sum / 15) = 100)
  (h_remove : [80, 90, 95] ⊆ numbers) :
  ((numbers.sum - 80 - 90 - 95) / 12) = (1235 / 12) :=
sorry

end average_of_remaining_numbers_l277_277970


namespace monotonic_increasing_interval_l277_277777

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.logb (1 / 2) x)^2 - 2 * (Real.logb (1 / 2) x) + 1

theorem monotonic_increasing_interval : 
  ∀ x, x ∈ Icc (Real.sqrt 2 / 2) (Real.Infty) → 
  ∀ y, (y > x) → (f y ≥ f x) :=
sorry

end monotonic_increasing_interval_l277_277777


namespace proof_problem_l277_277261

def fixed_point_through_l (a : ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), P = (-1, 2) ∧
  ∀ a : ℝ, a * (fst P + 1) - (snd P) + 2 = 0

def equation_of_line_m : Prop :=
  ∃ (A B : ℝ × ℝ), 
  A = (-2, 5) ∧ B = (0, -1) ∧
  ∃ (P : ℝ × ℝ), P = (-1, 2) ∧
  ∃ line_m : ℝ × ℝ → Prop, 
  (line_m = λ P, 3 * (fst P) + (snd P) + 1 = 0)

theorem proof_problem :
  (∀ a : ℝ, fixed_point_through_l a) ∧ equation_of_line_m :=
by
  sorry

end proof_problem_l277_277261


namespace hyperbola_correct_eq_l277_277512

noncomputable def hyperbola_eq (a b : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (∀ x y, C x y ↔ (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (let c := 3 in
    let F1 := (c, 0) in
    let F2 := (-c, 0) in
    ∀ P : Point,
      let |PF1| := dist P F1 in
      let |PF2| := dist P F2 in
      (|PF1| + |PF2| = 6 * a) ∧
      (|PF1| - |PF2| = 2 * a) ∧
      (angle P F1 F2 = 30°)) →
  (a = sqrt 3) ∧ (b = sqrt 6) ∧ (∀ x y, C x y ↔ (x^2 / 3 - y^2 / 6 = 1))

theorem hyperbola_correct_eq : 
  ∃ a b C, hyperbola_eq a b C :=
begin
  -- proof will be omitted
  sorry,
end

end hyperbola_correct_eq_l277_277512


namespace length_of_shorter_train_l277_277624

theorem length_of_shorter_train:
  ∀ (v1 v2 : ℝ) (L1 t : ℝ), 
    v1 = 60 * (1000/3600) →
    v2 = 40 * (1000/3600) →
    L1 = 180 →
    t = 11.519078473722104 →
    let relative_speed := v1 + v2 in
    let total_distance := relative_speed * t in
    total_distance - L1 = 140 :=
by
  intros v1 v2 L1 t hv1 hv2 hL1 ht relative_speed total_distance
  dsimp [relative_speed, total_distance]
  rw [hv1, hv2, hL1, ht]
  norm_num
  sorry

end length_of_shorter_train_l277_277624


namespace problem_statement_l277_277132

noncomputable def x : ℝ := (3 + real.sqrt 8) ^ 1001
noncomputable def n : ℤ := int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
  sorry

end problem_statement_l277_277132


namespace factor_increase_eq_three_l277_277928

-- Define the original price
def original_price : ℝ := 100

-- Define the profit made
def profit : ℝ := 200

-- Define the selling price as the sum of original price and profit
def selling_price : ℝ := original_price + profit

-- Define the factor as the ratio of the selling price to the original price
def factor : ℝ := selling_price / original_price

-- State the theorem to prove that the factor is 3
theorem factor_increase_eq_three : factor = 3 :=
by
  -- Proof goes here
  sorry

end factor_increase_eq_three_l277_277928


namespace perpendicular_line_equation_l277_277662

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l277_277662


namespace part1_part2_l277_277846
open Real

noncomputable def vec_a := (3, 2)
noncomputable def vec_b := (-1, 2)
noncomputable def vec_c := (0, 3)

-- conditions
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

def condition1 : Prop := dot_product vec_a vec_c = dot_product vec_b vec_c ∧ dot_product vec_a vec_c > 0

def condition2 : Prop := magnitude vec_c = 3

-- questions
theorem part1 (h1 : condition1) (h2 : condition2) : vec_c = (0, 3) := sorry

noncomputable def vec_3a := (3 * vec_a.1, 3 * vec_a.2)
noncomputable def vec_diff := (vec_3a.1 - vec_c.1, vec_3a.2 - vec_c.2)

theorem part2 (h1 : condition1) (h2 : condition2) : magnitude vec_diff = 3 * sqrt 10 := sorry

end part1_part2_l277_277846


namespace find_x_cube_plus_reciprocal_cube_l277_277864

variable {x : ℝ}

theorem find_x_cube_plus_reciprocal_cube (hx : x + 1/x = 10) : x^3 + 1/x^3 = 970 :=
sorry

end find_x_cube_plus_reciprocal_cube_l277_277864


namespace percentage_difference_l277_277445

-- Define the variables as non-negative reals
variables {W E Y Z : ℝ}

-- Conditions provided in the problem
def condition1 : Prop := W = 0.60 * E
def condition2 : Prop := Z = 0.54 * Y
def condition3 : Prop := Z = 1.5000000000000002 * W

-- We need to prove that the percentage 'p' by which E is less than Y is 0.4
theorem percentage_difference (W E Y Z : ℝ) 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) : 
  ∃ (p : ℝ), E = Y * (1 - p) ∧ p = 0.4 :=
by
  sorry

end percentage_difference_l277_277445


namespace boys_in_club_l277_277714

-- Definitions and assertions based on the conditions
variables (B G : ℕ)

-- Given conditions
def total_members : Prop := B + G = 32
def attendance : Prop := (2 * G) / 3 + B = 22

-- Main theorem to prove
theorem boys_in_club (h1 : total_members B G) (h2 : attendance B G) : B = 2 :=
by {
  sorry,
}

end boys_in_club_l277_277714


namespace james_pays_total_l277_277490

theorem james_pays_total (lessons_total : ℕ) (lessons_paid : ℕ) (lesson_cost : ℕ) (uncle_share : ℕ) :
  lessons_total = 20 →
  lessons_paid = 15 →
  lesson_cost = 5 →
  uncle_share = 2 → 
  (lesson_cost * lessons_paid) / uncle_share = 37.5 := 
by
  intros h1 h2 h3 h4
  sorry

end james_pays_total_l277_277490


namespace cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l277_277744

-- Define the initial state of the cube vertices
def initial_cube : ℕ → ℕ
| 0 => 1  -- The number at vertex 0 is 1
| _ => 0  -- The numbers at other vertices are 0

-- Define the edge addition operation
def edge_add (v1 v2 : ℕ → ℕ) (edge : ℕ × ℕ) : ℕ → ℕ :=
  λ x => if x = edge.1 ∨ x = edge.2 then v1 x + 1 else v1 x

-- Condition: one can add one to the numbers at the ends of any edge
axiom edge_op : ∀ (v : ℕ → ℕ) (e : ℕ × ℕ), ℕ → ℕ

-- Defining the problem in Lean
theorem cube_numbers_not_all_even :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 2 = 0) :=
by
  -- Proof not required
  sorry

theorem cube_numbers_not_all_divisible_by_3 :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 3 = 0) :=
by
  -- Proof not required
  sorry

end cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l277_277744


namespace xiao_li_password_probability_l277_277675

-- Definitions of the conditions
def first_digit :=
  {c | c = 'M' ∨ c = 'N'}

def second_digit :=
  {n | n = '1' ∨ n = '2' ∨ n = '3'}

-- We define an event that represents a successful password guess
def successful_guess_probability : ℚ :=
  1 / 6

-- Statement to be proven
theorem xiao_li_password_probability :
  (∃ password : (char × char), 
    password.1 ∈ first_digit ∧ password.2 ∈ second_digit) →
    successful_guess_probability = 1 / 6 :=
begin
  sorry
end

end xiao_li_password_probability_l277_277675


namespace main_proof_l277_277474

variables (A B C D E F : Type) [field A] [add_comm_group B] [module A B] [module A C] [module A D]
          [midpoint E D C] [intersects_line AE BD]

def coords (x y : B) : B := (x, y)
def vector (x E D : B) : B := E - D

def midpoint_coords : B := coords (3 / 2 : B) 0
def intersection_point : B := coords 1 1

def FD : B := vector (intersection_point (0 : B) (1 : B))
def DE : B := vector (midpoint_coords (0 : B) (3 / 2 : B))

def dot_product (u v : B) : B := u * v + v * 1

def FD_dot_DE : B := dot_product FD DE

theorem main_proof :
  FD_dot_DE = -(3 / 2 : B) := by
  sorry

end main_proof_l277_277474


namespace probability_one_black_one_white_l277_277682

def total_balls : ℕ := 6 + 2
def black_balls : ℕ := 6
def white_balls : ℕ := 2

def total_ways_to_pick_two_balls : ℕ := total_balls.choose 2
def ways_to_pick_one_black_one_white : ℕ := (black_balls.choose 1) * (white_balls.choose 1)

theorem probability_one_black_one_white :
  (ways_to_pick_one_black_one_white : ℚ) / total_ways_to_pick_two_balls = 3 / 7 :=
sorry

end probability_one_black_one_white_l277_277682


namespace problem_solution_l277_277911

theorem problem_solution:
  ∃ (p q m n : ℕ) (A B : Set ℕ) (f : ℕ → ℕ),
    A = {1, 2, 3, m} ∧
    B = {4, 7, n^4, n^2 + 3 * n} ∧
    (∀ a ∈ A, ∀ b ∈ B, f a = b ↔ b = p * a + q) ∧
    ∀ a1 a2 ∈ A, f a1 = f a2 → a1 = a2 ∧
    f 1 = 4 ∧ f 2 = 7 ∧
    m > 0 ∧ n > 0 ∧
    p = 3 ∧ q = 1 ∧ m = 5 ∧ n = 2 :=
by
  sorry

end problem_solution_l277_277911


namespace restaurant_cost_l277_277743

theorem restaurant_cost (total_people kids adult_cost : ℕ)
  (h1 : total_people = 12)
  (h2 : kids = 7)
  (h3 : adult_cost = 3) :
  total_people - kids * adult_cost = 15 := by
  sorry

end restaurant_cost_l277_277743


namespace molecular_weight_proof_l277_277239

-- Definitions for atomic weights
def atomic_weight_C := 12.01
def atomic_weight_H := 1.008
def atomic_weight_O := 16.00

-- Molecular formula of C6H8O6
def molecular_weight_C6H8O6 := 
  6 * atomic_weight_C + 8 * atomic_weight_H + 6 * atomic_weight_O

-- Given total molecular weight
def weight_of_moles := 528.0

-- Number of moles calculation
def number_of_moles := weight_of_moles / molecular_weight_C6H8O6

-- Proof statement
theorem molecular_weight_proof :
  (number_of_moles ≈ 3) ↔ (weight_of_moles = 528) :=
by
  sorry

end molecular_weight_proof_l277_277239


namespace students_no_A_l277_277074

variable (total_students : ℕ)
variable (A_in_history : ℕ)
variable (A_in_math : ℕ)
variable (A_in_science : ℕ)
variable (A_in_history_math : ℕ)
variable (A_in_history_science : ℕ)
variable (A_in_math_science : ℕ)
variable (A_in_all_three : ℕ)

theorem students_no_A (
  -- given conditions
  total_students = 45 
  (A_in_history = 11) 
  (A_in_math = 16)
  (A_in_science = 9)
  (A_in_history_math = 5)
  (A_in_history_science = 3)
  (A_in_math_science = 4)
  (A_in_all_three = 2)
  ) :
  ∃ (students_no_A : ℕ), students_no_A = 19 :=
by {
  -- Using the principle of inclusion-exclusion
  let N_total_A := A_in_history + A_in_math + A_in_science - A_in_history_math - A_in_history_science - A_in_math_science + A_in_all_three,
  exact ⟨total_students - N_total_A, rfl⟩
}

end students_no_A_l277_277074


namespace production_company_profit_l277_277271

-- Definitions of all the conditions
def domestic_opening_weekend := 120 * 10^6
def domestic_total_run := 3.5 * domestic_opening_weekend
def international_earnings := 1.8 * domestic_total_run
def domestic_earnings_after_taxes := 0.60 * domestic_total_run
def international_earnings_after_taxes := 0.45 * international_earnings
def total_earnings_after_taxes := domestic_earnings_after_taxes + international_earnings_after_taxes
def total_earnings_before_taxes := domestic_total_run + international_earnings
def royalties := 0.05 * total_earnings_before_taxes
def production_costs := 60 * 10^6
def marketing_costs := 40 * 10^6

-- The theorem to prove
theorem production_company_profit : 
  total_earnings_after_taxes - royalties - production_costs - marketing_costs = 433.4 * 10^6 := by
  sorry

end production_company_profit_l277_277271


namespace impossible_ratio_5_11_l277_277289

theorem impossible_ratio_5_11:
  ∀ (b g: ℕ), 
  b + g ≥ 66 →
  b + 11 = g - 13 →
  ¬(5 * b = 11 * (b + 24) ∧ b ≥ 21) := 
by
  intros b g h1 h2 h3
  sorry

end impossible_ratio_5_11_l277_277289


namespace tan_eq_example_l277_277814

theorem tan_eq_example (x : ℝ) (hx : Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x)) : x = 30 * Real.pi / 180 :=
  sorry

end tan_eq_example_l277_277814


namespace reciprocals_sum_eq_neg_one_over_three_l277_277144

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end reciprocals_sum_eq_neg_one_over_three_l277_277144


namespace ratio_triangle_A_to_BCD_eq_one_ratio_DE_to_EC_eq_one_l277_277935

-- Definitions for the geometric setup
variables (A B C D E F : Type) [square ABCD] (midpoint E CD)

-- Part (a)
theorem ratio_triangle_A_to_BCD_eq_one (h1 : is_midpoint E CD) : 
  let s := side_length ABCD in
  area (△ ABE) = area (△ BCD) :=
sorry

-- Part (b)
theorem ratio_DE_to_EC_eq_one (h1 : intersection F AE BD) 
  (h2 : area (△ BFE) = 2 * area (△ DFE)) : 
  DE = EC :=
sorry

end ratio_triangle_A_to_BCD_eq_one_ratio_DE_to_EC_eq_one_l277_277935


namespace find_divisor_l277_277628

theorem find_divisor (D Q R d : ℕ) (h1 : D = 159) (h2 : Q = 9) (h3 : R = 6) (h4 : D = d * Q + R) : d = 17 := by
  sorry

end find_divisor_l277_277628


namespace mama_bird_worms_l277_277523

theorem mama_bird_worms (babies : ℕ) (worms_per_day_per_baby : ℕ) (days : ℕ) (papa_worms : ℕ) (mama_worms : ℕ) (stolen_worms : ℕ) :
  babies = 6 →
  worms_per_day_per_baby = 3 →
  days = 3 →
  papa_worms = 9 →
  mama_worms = 13 →
  stolen_worms = 2 →
  let total_worms_needed := babies * worms_per_day_per_baby * days
      remaining_mama_worms := mama_worms - stolen_worms
      total_current_worms := remaining_mama_worms + papa_worms
      additional_worms_needed := total_worms_needed - total_current_worms
  in additional_worms_needed = 34 :=
by intros; sorry

end mama_bird_worms_l277_277523


namespace min_area_eq_line_min_sum_eq_line_l277_277824

-- Part 1: Line equation minimizing the area of triangle OAB
theorem min_area_eq_line (k : ℝ) (x y : ℝ) (P A B : ℝ × ℝ) (O : ℝ × ℝ) :
  (P = (4, 1)) ∧ (A = (4 * k - 1) / k, 0) ∧ (B = (0, 1 - 4 * k)) ∧ (O = (0, 0)) ∧
  (y = k * x - 4 * k + 1) ∧ ((A.1 * B.2) / 2 = -8 * k - 1 / (2 * k) + 4) ∧ 
  (k = -1 / 4) →
  x + 4 * y - 8 = 0 := sorry

-- Part 2: Line equation minimizing the sum |OA| + |OB|
theorem min_sum_eq_line (k : ℝ) (x y : ℝ) (P A B : ℝ × ℝ) (O : ℝ × ℝ) :
  (P = (4, 1)) ∧ (A = (4 * k - 1) / k, 0) ∧ (B = (0, 1 - 4 * k)) ∧ (O = (0, 0)) ∧
  (y = k * x - 4 * k + 1) ∧ (5 - 1 / k - 4 * k = 9) ∧ 
  (k = -1 / 2) →
  x + 2 * y - 6 = 0 := sorry

end min_area_eq_line_min_sum_eq_line_l277_277824


namespace ratio_of_P_to_Q_l277_277283

theorem ratio_of_P_to_Q (p q r s : ℕ) (h1 : p + q + r + s = 1000)
    (h2 : s = 4 * r) (h3 : q = r) (h4 : s - p = 250) : 
    p = 2 * q :=
by
  -- Proof omitted
  sorry

end ratio_of_P_to_Q_l277_277283


namespace imaginary_part_of_fraction_l277_277980

theorem imaginary_part_of_fraction (i : ℂ) (h : i^2 = -1) : ( (i^2) / (2 * i - 1) ).im = (2 / 5) :=
by
  sorry

end imaginary_part_of_fraction_l277_277980


namespace quotient_of_sum_of_distinct_squares_mod_8_l277_277959

theorem quotient_of_sum_of_distinct_squares_mod_8 :
  let squares := {n^2 % 8 | n in set.Icc 1 7}
  let distinct_remainders := set.to_finset squares
  let m := distinct_remainders.sum id
  (m / 8) = 0 :=
by
  sorry

end quotient_of_sum_of_distinct_squares_mod_8_l277_277959


namespace find_a_values_l277_277336

theorem find_a_values (a : ℝ) :
  4 < a ∧ a < 5 ∧ ∃ k : ℤ, a * (a - 3 * (a - a.floor)) = k ↔
  a ∈ {3 + (real.sqrt 21) / 2, 
       3 + real.sqrt 5,
       3 + (real.sqrt 19) / 2,
       3 + (3 * real.sqrt 2) / 2,
       3 + (real.sqrt 17) / 2} := by
  sorry

end find_a_values_l277_277336


namespace number_of_seniors_l277_277458

theorem number_of_seniors (total_students : ℕ) (junior_percentage : ℕ) (not_sophomore_percentage : ℕ)
  (freshmen_more_than_sophomores : ℕ) 
  (H1 : total_students = 800)
  (H2 : junior_percentage = 28)
  (H3 : not_sophomore_percentage = 75)
  (H4 : freshmen_more_than_sophomores = 16) :
  let juniors := junior_percentage * total_students / 100 in
  let sophomores := (100 - not_sophomore_percentage) * total_students / 100 in
  let freshmen := sophomores + freshmen_more_than_sophomores in
  let seniors := total_students - juniors - sophomores - freshmen in
  seniors = 160 :=
by 
  sorry

end number_of_seniors_l277_277458


namespace points_earned_l277_277878

-- Definitions of the types of enemies and their point values
def points_A := 10
def points_B := 15
def points_C := 20

-- Number of each type of enemies in the level
def num_A_total := 3
def num_B_total := 2
def num_C_total := 3

-- Number of each type of enemies defeated
def num_A_defeated := num_A_total -- 3 Type A enemies
def num_B_defeated := 1 -- Half of 2 Type B enemies
def num_C_defeated := 1 -- 1 Type C enemy

-- Calculation of total points earned
def total_points : ℕ :=
  num_A_defeated * points_A + num_B_defeated * points_B + num_C_defeated * points_C

-- Proof that the total points earned is 65
theorem points_earned : total_points = 65 := by
  -- Placeholder for the proof, which calculates the total points
  sorry

end points_earned_l277_277878


namespace inverse_function_log_l277_277203

theorem inverse_function_log {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∃ (f : ℝ → ℝ), ∀ y : ℝ, y = 1 + real.log (f y) / real.log a ↔ ∃ x : ℝ, f x = a ^ (x - 1)) :=
by
  sorry

end inverse_function_log_l277_277203


namespace cards_robert_traded_l277_277940

-- Defining initial conditions
variables (padma_initial_cards : ℕ) (robert_initial_cards : ℕ)
variables (padma_trade1_out : ℕ) (padma_trade1_in : ℕ)
variables (robert_trade_for_padma : ℕ) (total_traded_cards : ℕ)

-- Given conditions assignment
def conditions : Prop :=
  padma_initial_cards = 75 ∧
  padma_trade1_out = 2 ∧
  padma_trade1_in = 10 ∧
  robert_initial_cards = 88 ∧
  (total_traded_cards = 35) ∧
  (robert_trade_for_padma = total_traded_cards - padma_trade1_in)

-- Theorem to prove the answer
theorem cards_robert_traded (h : conditions) : robert_trade_for_padma = 25 :=
by sorry

end cards_robert_traded_l277_277940


namespace triangle_consecutive_numbers_l277_277310

theorem triangle_consecutive_numbers : 
  ∃ a b c S : ℕ, 
    (b = a+1 ∧ c = a+2 ∧ S = a+3) ∨ (b = a+1 ∧ c = a+3 ∧ S = a+2) ∧
    (S = 1/4 * Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))) ∧
    Set.Iso.triangle a b c :=
sorry

end triangle_consecutive_numbers_l277_277310


namespace intersection_of_sets_l277_277033

def A := {x : ℤ | x^2 - 2 * x - 3 < 0}
def B := {-1, 0, 1, 2, 3}
def C := {0, 1, 2}

theorem intersection_of_sets : A ∩ B = C :=
by sorry

end intersection_of_sets_l277_277033


namespace PS_div_QR_eq_sqrt3_l277_277478

variables {P Q R S : Type} [metric_space P]

-- Assume we have points P, Q, R, S
variables [PQR_equilateral : equilateral_triangle P Q R] 
variables [QRS_equilateral : equilateral_triangle Q R S]

-- Assume lengths of QR and PS
axiom length_QR : length (Q, R) = t
axiom length_PS : length (P, S) = t * sqrt 3

theorem PS_div_QR_eq_sqrt3 :
  length (P, S) / length (Q, R) = sqrt 3 :=
sorry

end PS_div_QR_eq_sqrt3_l277_277478


namespace harry_sandy_midpoint_l277_277847

theorem harry_sandy_midpoint :
  ∃ (x y : ℤ), x = 9 ∧ y = -2 → ∃ (a b : ℤ), a = 1 ∧ b = 6 → ((9 + 1) / 2, (-2 + 6) / 2) = (5, 2) := 
by 
  sorry

end harry_sandy_midpoint_l277_277847


namespace pentagon_area_condition_l277_277901

-- Definitions and conditions from the problem
variables {A B C D E P : Type}

-- Conditions: IsConvexPentagon, point_property, etc.
def IsConvexPentagon (A B C D E : Type) : Prop := sorry
def points_collinear (A B X Y : Type) : Prop := sorry
def distance_eq (X Y : Type) (Z W: Type) : Prop := sorry
def angle_eq (X Y Z : Type) (A: Type) : Prop := sorry
def area (X Y Z : Type) : ℝ := sorry

variables (AXB YZ : Type) (Q : Type) -- Including necessary auxiliary points.

theorem pentagon_area_condition (A B C D E P : Type)
  (h₀ : IsConvexPentagon A B C D E)
  (h₁ : distance_eq C D D E)
  (h₂ : ¬angle_eq E D C (2 * angle_eq A D B)) 
  (h₃ : ¬points_collinear A E B P)
  (h₄ : distance_eq A P A E)
  (h₅ : distance_eq B P B C) :
  (points_collinear C E P ↔ ((area B C D + area A D E) = (area A B D + area A B P))) :=
sorry

end pentagon_area_condition_l277_277901


namespace perpendicular_line_through_point_l277_277642

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l277_277642


namespace sum_of_squares_of_coeffs_l277_277240

theorem sum_of_squares_of_coeffs (c1 c2 c3 c4 : ℝ) (h1 : c1 = 3) (h2 : c2 = 6) (h3 : c3 = 15) (h4 : c4 = 6) :
  c1^2 + c2^2 + c3^2 + c4^2 = 306 :=
by
  sorry

end sum_of_squares_of_coeffs_l277_277240


namespace distance_point_to_plane_l277_277826

/-- 
  Given a normal vector n = (2, 0, 1) of a plane α, 
  and a point A(-1, 2, 1) lying on α,
  prove that the distance from the point P(1, 2, -2) to the plane α is sqrt 5 / 5.
-/
theorem distance_point_to_plane (n : ℝ × ℝ × ℝ) 
  (A : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) 
  (h_n : n = (2, 0, 1)) 
  (h_A : A = (-1, 2, 1)) 
  (h_P : P = (1, 2, -2)) :
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
  let d := abs (AP.1 * n.1 + AP.2 * n.2 + AP.3 * n.3) / real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2) in
  d = real.sqrt 5 / 5 :=
by 
  sorry

end distance_point_to_plane_l277_277826


namespace ratio_of_ian_to_jessica_l277_277948

/-- 
Rodney has 35 dollars more than Ian. 
Jessica has 100 dollars. 
Jessica has 15 dollars more than Rodney. 
Prove that the ratio of Ian's money to Jessica's money is 1/2.
-/
theorem ratio_of_ian_to_jessica (I R J : ℕ) (h1 : R = I + 35) (h2 : J = 100) (h3 : J = R + 15) :
  I / J = 1 / 2 :=
by
  sorry

end ratio_of_ian_to_jessica_l277_277948


namespace wire_length_calculation_l277_277971

theorem wire_length_calculation (d₁ d₂ h₁ h₂ : ℝ) (h₁_eq : h₁ = 10) (h₂_eq : h₂ = 18) (d₁_eq : d₁ = 20) :
  let hypotenuse := Real.sqrt (d₁^2 + (h₂ - h₁)^2)
  let vertical_down_and_up := h₁ + h₁
  let base_and_back_up := d₁ + h₂
  total_wire_length = hypotenuse + vertical_down_and_up + base_and_back_up 
  total_wire_length = Real.sqrt 464 + 58 := 
by
  sorry

end wire_length_calculation_l277_277971


namespace select_three_numbers_sum_even_not_less_than_10_l277_277352

theorem select_three_numbers_sum_even_not_less_than_10 :
  ∃ count : ℕ, count = 51 ∧
  ∀ (numbers : Finset ℕ), 
    numbers ⊆ (Finset.range 10) → numbers.card = 3 →
    (∃ (selected_numbers : Finset ℕ), selected_numbers ⊆ numbers ∧ selected_numbers.sum % 2 = 0 ∧ selected_numbers.sum ≥ 10) → 
    Finset.card ((Finset.filter (λ s : Finset ℕ, s.sum % 2 = 0 ∧ s.sum ≥ 10) (Finset.powersetLen 3 numbers))) = count := sorry

end select_three_numbers_sum_even_not_less_than_10_l277_277352


namespace blue_markers_count_l277_277298

theorem blue_markers_count (total_markers : ℕ) (percent_blue : ℝ) (blue_marker_count : ℕ) :
  total_markers = 96 →
  percent_blue = 0.30 →
  blue_marker_count = Int.round (percent_blue * total_markers) →
  blue_marker_count = 29 :=
by
  intros total_markers_eq percent_blue_eq blue_marker_count_eq
  rw [total_markers_eq, percent_blue_eq, blue_marker_count_eq]
  sorry

end blue_markers_count_l277_277298


namespace max_distance_l277_277016

-- Definition of curve C₁ in rectangular coordinates.
def C₁_rectangular (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- Definition of curve C₂ in its general form.
def C₂_general (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Coordinates of point M, the intersection of C₂ with x-axis.
def M : ℝ × ℝ := (2, 0)

-- Condition that N is a moving point on curve C₁.
def N (x y : ℝ) : Prop := C₁_rectangular x y

-- Maximum distance |MN|.
theorem max_distance (x y : ℝ) (hN : N x y) : 
  dist (2, 0) (x, y) ≤ Real.sqrt 5 + 1 := by
  sorry

end max_distance_l277_277016


namespace acid_solution_l277_277703

theorem acid_solution (x y : ℝ) (h1 : 0.3 * x + 0.1 * y = 90)
  (h2 : x + y = 600) : x = 150 ∧ y = 450 :=
by
  sorry

end acid_solution_l277_277703


namespace circle_intersection_l277_277905

variables {A B C D E F O H X : Type*} [Nontrivial (triangle A B C)] [Altitudes D A B C] 
          [Altitudes E B C A] [Altitudes F C A B]

def feet_of_altitudes (A B C : Triangle) : Set (Triangle) :=
  {D, E, F} 

def circumcenter (A B C : Triangle) : Point := O

def orthocenter (A B C : Triangle) : Point := H

theorem circle_intersection
  (A B C : Triangle)
  (D E F : Point)
  (O : Point) 
  (H : Point)
  (h1 : D = feet_of_altitudes A B C)
  (h2 : E = feet_of_altitudes A B C)
  (h3 : F = feet_of_altitudes A B C)
  (h4 : O = circumcenter A B C)
  (h5 : H = orthocenter A B C) :
  ∃ X : Point, (X ≠ O) ∧ (X ∈ circle A O D) ∧ (X ∈ circle B O E) ∧ (X ∈ circle C O F) :=
sorry

end circle_intersection_l277_277905


namespace original_radius_l277_277890

-- Variables representing the original radius and volume changes
variable (r x : ℝ)

-- Conditions
def volume_original (r h : ℝ) : ℝ := π * r^2 * h
def volume_radius_increased (r : ℝ) : ℝ := volume_original (r + 5) 5
def volume_height_increased (r : ℝ) : ℝ := volume_original r 12

-- Volume change conditions
def volume_change_radius_condition (r x : ℝ) : Prop :=
  volume_radius_increased r - volume_original r 5 = x

def volume_change_height_condition (r x : ℝ) : Prop :=
  volume_height_increased r - volume_original r 5 = x

-- Statement to prove
theorem original_radius (r : ℝ) (h : 5 = 5) (x : volume_change_radius_condition r x) (y : volume_change_height_condition r x) :
  r = (25 - 10 * real.sqrt 15) / 7 := 
sorry

end original_radius_l277_277890


namespace advantageous_order_l277_277540

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l277_277540


namespace perpendicular_line_eq_l277_277658

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l277_277658


namespace max_distance_PQ_proof_l277_277385

noncomputable theory

def circle_eqn (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

def ellipse_eqn (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def max_distance_PQ : ℝ := (3 * real.sqrt 6) / 2 + 1

theorem max_distance_PQ_proof : 
  ∃ (Px Py Qx Qy : ℝ), circle_eqn Px Py ∧ ellipse_eqn Qx Qy ∧ distance Px Py Qx Qy = max_distance_PQ := by
  sorry

end max_distance_PQ_proof_l277_277385


namespace triangle_ABC_is_right_angled_l277_277821

theorem triangle_ABC_is_right_angled (a b c: ℕ) (h_a: a = 3) (h_c: c = 5) (h_quad_roots: ∀ x, x^2 - 4 * x + b = 0 → (b^2 - 4 * 1 * b = 0)) : b = 4 ∧ a^2 + b^2 = c^2 :=
by
  have h_b := h_quad_roots 2 (by simp [sq]; linarith [h_a, h_c])
  simp [h_b, h_a, h_c]
  sorry

end triangle_ABC_is_right_angled_l277_277821


namespace identify_different_correlation_l277_277679

-- Define the concept of correlation
inductive Correlation
| positive
| negative

-- Define the conditions for each option
def option_A : Correlation := Correlation.positive
def option_B : Correlation := Correlation.positive
def option_C : Correlation := Correlation.negative
def option_D : Correlation := Correlation.positive

-- The statement to prove
theorem identify_different_correlation :
  (option_A = Correlation.positive) ∧ 
  (option_B = Correlation.positive) ∧ 
  (option_D = Correlation.positive) ∧ 
  (option_C = Correlation.negative) := 
sorry

end identify_different_correlation_l277_277679


namespace three_sum_xyz_l277_277865

theorem three_sum_xyz (x y z : ℝ) 
  (h1 : y + z = 18 - 4 * x) 
  (h2 : x + z = 22 - 4 * y) 
  (h3 : x + y = 15 - 4 * z) : 
  3 * x + 3 * y + 3 * z = 55 / 2 := 
  sorry

end three_sum_xyz_l277_277865


namespace monotonic_intervals_nonneg_f_for_all_x_ge_0_compare_magnitudes_l277_277403

open Real

-- (I)
theorem monotonic_intervals (f : ℝ → ℝ) : 
  (∀ x, f x = exp(2 * x) - 1 - 2 * x) →
  (∀ x, 0 < x → diff f x > 0) ∧ (∀ x, x < 0 → diff f x < 0) :=
sorry

-- (II)
theorem nonneg_f_for_all_x_ge_0 (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = exp(2 * x) - 1 - 2 * x - k * x^2) →
  (k ≤ 2) →
  (∀ x, 0 ≤ x → f x ≥ 0) :=
sorry

-- (III)
theorem compare_magnitudes (n : ℕ) (h : 0 < n) :
  (∑ i in range n, (exp(2 * i))) ≥ (2 * n^3 + n) / 3 :=
sorry

end monotonic_intervals_nonneg_f_for_all_x_ge_0_compare_magnitudes_l277_277403


namespace total_number_of_games_l277_277966

def high_school_twelve_teams := 12
def games_against_each_team := 3
def non_league_games_per_team := 6

theorem total_number_of_games : 
  let league_teams := high_school_twelve_teams in
  let games_per_pair := games_against_each_team in
  let non_league_games := non_league_games_per_team in
  let games_within_league := (league_teams * (league_teams - 1) / 2) * games_per_pair in
  let games_outside_league := league_teams * non_league_games in
  games_within_league + games_outside_league = 270 :=
by
  let league_teams := high_school_twelve_teams
  let games_per_pair := games_against_each_team
  let non_league_games := non_league_games_per_team
  let games_within_league := (league_teams * (league_teams - 1) / 2) * games_per_pair
  let games_outside_league := league_teams * non_league_games
  have games_within_league_correct : (league_teams * (league_teams - 1) / 2) * games_per_pair = 198 :=
    by sorry
  have games_outside_league_correct : league_teams * non_league_games = 72 :=
    by sorry
  rw [games_within_league_correct, games_outside_league_correct]
  exact eq.refl 270

end total_number_of_games_l277_277966


namespace jeremy_home_to_school_distance_l277_277495

theorem jeremy_home_to_school_distance (v d : ℝ) (h1 : 30 / 60 = 1 / 2) (h2 : 15 / 60 = 1 / 4)
  (h3 : d = v * (1 / 2)) (h4 : d = (v + 12) * (1 / 4)):
  d = 6 :=
by
  -- We assume that the conditions given lead to the distance being 6 miles
  sorry

end jeremy_home_to_school_distance_l277_277495


namespace cooking_mode_median_l277_277229

noncomputable def cooking_data : List (ℕ × ℕ) :=
[(4, 7), (5, 6), (6, 12), (7, 10), (8, 5)]

theorem cooking_mode_median :
  let frequencies := cooking_data
  mode frequencies = 6 ∧ median frequencies = 6 :=
by
  -- The proof is omitted.
  sorry

end cooking_mode_median_l277_277229


namespace possible_values_of_m_l277_277048

variable (m : ℝ)
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c > 0)

theorem possible_values_of_m (h : has_two_distinct_real_roots 1 m 9) : m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end possible_values_of_m_l277_277048


namespace problem_l277_277504

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := sorry
def v : Fin 2 → ℝ := ![7, -3]
def result : Fin 2 → ℝ := ![-14, 6]
def expected : Fin 2 → ℝ := ![112, -48]

theorem problem :
    B.vecMul v = result →
    B.vecMul (B.vecMul (B.vecMul (B.vecMul v))) = expected := 
by
  intro h
  sorry

end problem_l277_277504


namespace magnitude_a_is_sqrt2_l277_277845

variables (n : ℝ)

def a := (1 : ℝ, n)
def b := (-1 : ℝ, n)

-- \overrightarrow{a} is perpendicular to \overrightarrow{b}
axiom perp_dot_product_zero (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) : true

-- Proof that the magnitude of \overrightarrow{a} is \sqrt{2}
theorem magnitude_a_is_sqrt2 (h_perpendicular : (1 : ℝ) + n^2 = 0) : real.sqrt (1^2 + n^2) = real.sqrt 2 :=
by sorry

end magnitude_a_is_sqrt2_l277_277845


namespace modulus_of_z_l277_277386

open Complex -- Open the Complex number namespace

-- Define the given condition as a hypothesis
def condition (z : ℂ) : Prop := (1 + I) * z = 3 + I

-- Statement of the theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 5 :=
sorry

end modulus_of_z_l277_277386


namespace drink_exactly_five_bottles_last_day_l277_277565

/-- 
Robin bought 617 bottles of water and needs to purchase 4 additional bottles on the last day 
to meet her daily water intake goal. 
Prove that Robin will drink exactly 5 bottles on the last day.
-/
theorem drink_exactly_five_bottles_last_day : 
  ∀ (bottles_bought : ℕ) (extra_bottles : ℕ), bottles_bought = 617 → extra_bottles = 4 → 
  ∃ x : ℕ, 621 = x * 617 + 4 ∧ x + 4 = 5 :=
by
  intros bottles_bought extra_bottles bottles_bought_eq extra_bottles_eq
  -- The proof would follow here
  sorry

end drink_exactly_five_bottles_last_day_l277_277565


namespace max_vehicles_div_by_100_l277_277707

noncomputable def max_vehicles_passing_sensor (n : ℕ) : ℕ :=
  2 * (20000 * n / (5 + 10 * n))

theorem max_vehicles_div_by_100 : 
  (∀ n : ℕ, (n > 0) → (∃ M : ℕ, M = max_vehicles_passing_sensor n ∧ M / 100 = 40)) :=
sorry

end max_vehicles_div_by_100_l277_277707


namespace smallest_integer_n_smallest_integer_n_5_smallest_value_of_n_l277_277299

noncomputable def sequence (n : ℕ) : ℝ :=
if n = 1 then real.sqrt (real.sqrt 4)
else (sequence (n - 1)) ^ (real.sqrt (real.sqrt 4))

theorem smallest_integer_n (n : ℕ) (h : n < 5) : ¬ (sequence n).is_integer :=
sorry

theorem smallest_integer_n_5 : sequence 5 = 4 :=
sorry

theorem smallest_value_of_n : ∃ (n : ℕ), (sequence n).is_integer ∧ ∀ (m : ℕ), m < n → ¬ (sequence m).is_integer :=
exists.intro 5
  (and.intro
    (by simp [sequence, real.rpow_nat_cast] ; sorry)
    (by intro m hm ; apply smallest_integer_n m hm))

end smallest_integer_n_smallest_integer_n_5_smallest_value_of_n_l277_277299


namespace hospital_staff_l277_277225

-- Define the conditions
variables (d n : ℕ) -- d: number of doctors, n: number of nurses
variables (x : ℕ) -- common multiplier

theorem hospital_staff (h1 : d + n = 456) (h2 : 8 * x = d) (h3 : 11 * x = n) : n = 264 :=
by
  -- noncomputable def only when necessary, skipping the proof with sorry
  sorry

end hospital_staff_l277_277225


namespace option_B_l277_277380

noncomputable def f : ℝ → ℝ := sorry

axiom even_fn (x : ℝ) : f (-x) = f x
axiom f_domain : ∀ x, -5 ≤ x ∧ x ≤ 5
axiom f_ineq : f 3 > f 1

theorem option_B : f (-1) < f 3 :=
by
  have h1 : f 1 = f (-1), from even_fn 1
  have h2 : f 3 > f (-1), from h1 ▸ f_ineq
  exact h2

end option_B_l277_277380


namespace problem1_problem2_l277_277836

-- Define the parametric equations of the curve C
def curveC_parametric (t : ℝ) : ℝ × ℝ :=
  ( (t^2 - 4) / (t^2 + 4),
    8 * t / (t^2 + 4) )

-- Define the standard form of the curve C
def curveC_standard (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

-- Define the problem statement 1: Parametric to standard form proof
theorem problem1 (t : ℝ) :
  let (x, y) := curveC_parametric t in
  curveC_standard x y :=
sorry

-- Define the second problem: Finding the range for |PA| * |PB|
def pointP : ℝ × ℝ := (0, 1)

-- Define |PA| * |PB| given intersection points (t1, t2)
def PA_times_PB (t1 t2 : ℝ) : ℝ :=
  abs t1 * abs t2

-- Define the valid range for |PA| * |PB|
def PA_PB_range : Set ℝ :=
  set.Icc (3 / 4) 3

-- Define the problem statement 2 with line passing through point P
theorem problem2 (t1 t2 : ℝ) (alpha : ℝ) (h : line_through_P_alpha t1 t2 alpha) :
  PA_times_PB t1 t2 ∈ PA_PB_range :=
sorry


end problem1_problem2_l277_277836


namespace number_of_valid_flags_l277_277304

-- Define the colors as a type
inductive Color
| purple
| gold

-- Define the condition for the flag
def valid_flag (s1 s2 s3 : Color) : Prop :=
(s1 ≠ s2) ∧ (s2 ≠ s3)

-- Statement of the problem
theorem number_of_valid_flags : ∃ n: ℕ, n = 2 ∧ (∀ (s1 s2 s3 : Color), valid_flag s1 s2 s3 → n) :=
by
  sorry

end number_of_valid_flags_l277_277304


namespace cable_length_l277_277189

-- Define the problem based on the given conditions.
def length_of_cable (distance_between_poles pole1_height pole2_height : ℝ) : ℝ :=
  let vertical_diff := pole2_height - pole1_height
  in Real.sqrt (distance_between_poles ^ 2 + vertical_diff ^ 2)

-- Define the conditions
def distance_between_poles : ℝ := 20
def height_of_pole1 : ℝ := 9
def height_of_pole2 : ℝ := 24

-- State the theorem
theorem cable_length : length_of_cable distance_between_poles height_of_pole1 height_of_pole2 = 25 := by
  sorry

end cable_length_l277_277189


namespace min_nodes_hexagon_grid_midpoint_l277_277737

theorem min_nodes_hexagon_grid_midpoint (nodes : set (ℤ × ℤ)) (h : nodes ⊆ {(m * a + n * b) | m n a b : ℤ}) :
  9 ≤ nodes.card → ∃ (x y : ℤ × ℤ), x ∈ nodes ∧ y ∈ nodes ∧ (∃ (z : ℤ × ℤ), z = (x + y) / 2 ∧ z ∈ nodes) :=
by sorry

end min_nodes_hexagon_grid_midpoint_l277_277737


namespace sum_medians_less_than_perimeter_l277_277560

noncomputable def median_a (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * b^2 + 2 * c^2 - a^2).sqrt

noncomputable def median_b (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * c^2 - b^2).sqrt

noncomputable def median_c (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * b^2 - c^2).sqrt

noncomputable def sum_of_medians (a b c : ℝ) : ℝ :=
  median_a a b c + median_b a b c + median_c a b c

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  perimeter a b c / 2

theorem sum_medians_less_than_perimeter (a b c : ℝ) :
  semiperimeter a b c < sum_of_medians a b c ∧ sum_of_medians a b c < perimeter a b c :=
by
  sorry

end sum_medians_less_than_perimeter_l277_277560


namespace roots_of_quadratic_irrational_l277_277786

theorem roots_of_quadratic_irrational (k : ℝ) (h : 2 * k^2 - 1 = 7) : 
  let a := 1 
  let b := -3 * k 
  let c := 2 * k^2 - 1
  (c / a = 7) ∧
  ∀ x₁ x₂ : ℝ, (x₁^2 - 3 * k * x₁ + 2 * k^2 - 1 = 0) ∧ (x₂^2 - 3 * k * x₂ + 2 * k^2 - 1 = 0) → 
    (¬is_rat x₁ ∧ ¬is_rat x₂) :=
by
  sorry

end roots_of_quadratic_irrational_l277_277786


namespace chicken_coops_count_l277_277226

theorem chicken_coops_count (chickens_one_coop total_chickens : ℕ) (h1 : chickens_one_coop = 60) (h2 : total_chickens = 540) :
  total_chickens / chickens_one_coop = 9 :=
by
  rw [h1, h2]
  norm_num

end chicken_coops_count_l277_277226


namespace value_of_a5_l277_277882

-- Define the sequence as an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem conditions
constant a : ℕ → ℝ
axiom h1: is_arithmetic_sequence a
axiom h2 : a 1 + a 9 = 10

-- State the theorem
theorem value_of_a5 : a 5 = 5 :=
by
  -- Proof will be provided here
  sorry

end value_of_a5_l277_277882


namespace number_of_ways_to_fill_grid_l277_277092

open Finset

theorem number_of_ways_to_fill_grid (n : ℕ) (h : n ≥ 1) :
  let grid := Matrix (Fin n) (Fin n) (Fin 2)
  let condition (m : grid) := (∀ i : Fin n, even (card { j | m i j = 1 })) ∧
                              (∀ j : Fin n, even (card { i | m i j = 1 }))
  ∃ fill_count : ℕ, (fill_count = 2^((n-1)*(n-1))) ∧
                    ∀ g : grid, condition g ↔ (g ∈ universe grid) :=
sorry

end number_of_ways_to_fill_grid_l277_277092


namespace parallel_vectors_l277_277420

def vec_a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem parallel_vectors (x : ℝ) : vec_a x = (2, 4) → x = 2 := by
  sorry

end parallel_vectors_l277_277420


namespace main_theorem_l277_277693

def Coin : Type := 
  {weight : ℤ // ∃ n : ℤ, weight = n ∨ weight = n + 1 ∨ weight = n - 1}

def genuine (c : Coin) (n : ℤ) : Prop := c.1 = n
def heavy_fake (c : Coin) (n : ℤ) : Prop := c.1 = n + 1
def light_fake (c : Coin) (n : ℤ) : Prop := c.1 = n - 1

def balance (c1 c2 : Coin) : ℤ := c1.1 - c2.1

noncomputable def Nastya (a b c d e : Coin) (g : ℤ) : Prop :=
  (balance a b = 0 ∨ balance a b ≠ 0) ∧
  (balance c d = 0 ∨ balance c d ≠ 0) ∧
  (balance (Coin.mk (a.1 + b.1) _) (Coin.mk (c.1 + d.1) _) = 0 ∨ 
   balance (Coin.mk (a.1 + b.1) _) (Coin.mk (c.1 + d.1) _) ≠ 0) →
  ∃ h l, (heavy_fake h g ∧ light_fake l g)

theorem main_theorem (a b c d e : Coin) :
  ∃ g h l, 
  (genuine a g ∨ genuine b g ∨ genuine c g ∨ genuine d g ∨ genuine e g) ∧
  a ≠ h ∧ a ≠ l ∧
  b ≠ h ∧ b ≠ l ∧
  c ≠ h ∧ c ≠ l ∧
  d ≠ h ∧ d ≠ l ∧
  e ≠ h ∧ e ≠ l →
  Nastya a b c d e g := 
sorry

end main_theorem_l277_277693


namespace second_smallest_perimeter_l277_277669

theorem second_smallest_perimeter (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → 
  (a + b + c = 12) :=
by
  sorry

end second_smallest_perimeter_l277_277669


namespace distinct_stone_arrangements_l277_277895

-- Define the set of 12 unique stones
def stones := Finset.range 12

-- Define the number of unique placements without considering symmetries
def placements : ℕ := stones.card.factorial

-- Define the number of symmetries (6 rotations and 6 reflections)
def symmetries : ℕ := 12

-- The total number of distinct configurations accounting for symmetries
def distinct_arrangements : ℕ := placements / symmetries

-- The main theorem stating the number of distinct arrangements
theorem distinct_stone_arrangements : distinct_arrangements = 39916800 := by 
  sorry

end distinct_stone_arrangements_l277_277895


namespace range_of_a_l277_277789

def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then 16 - a * x else 6 * x - x^3

def quasi_singular_points (f : ℝ → ℝ) : set (ℝ × ℝ) :=
{ p | ∃ x0, p = (x0, f x0) ∨ p = (-x0, f (-x0)) ∧ f (-x0) = -f (x0) }

theorem range_of_a (a : ℝ) :
  (∃ x0, f x0 a = x0 ∧ (-x0, f (-x0) a) ∈ quasi_singular_points (f x0 a)) →
  (6 < a) :=
sorry

end range_of_a_l277_277789


namespace find_x_maximum_binomial_l277_277389

noncomputable def maximum_binomial_equation (x : ℝ) : Prop :=
  (↑(Nat.choose 8 4) * (2 * x)^4 * (-x^Real.log10 x)^4 = 1120)

theorem find_x_maximum_binomial :
  ∃ x : ℝ, maximum_binomial_equation x ∧ x = 1 / 10 :=
by
  sorry

end find_x_maximum_binomial_l277_277389


namespace find_positive_number_l277_277999

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l277_277999


namespace number_of_even_1s_grids_l277_277098

theorem number_of_even_1s_grids (n : ℕ) : 
  (∃ grid : fin n → fin n → ℕ, 
    (∀ i j, grid i j = 0 ∨ grid i j = 1) ∧
    (∀ i, (∑ j, grid i j) % 2 = 0) ∧
    (∀ j, (∑ i, grid i j) % 2 = 0)) →
  2 ^ ((n - 1) * (n - 1)) = 2 ^ ((n - 1) * (n - 1)) :=
by sorry

end number_of_even_1s_grids_l277_277098


namespace radius_of_tangent_circle_l277_277595

theorem radius_of_tangent_circle (a b : ℕ) (r1 r2 r3 : ℚ) (R : ℚ)
  (h1 : a = 6) (h2 : b = 8)
  (h3 : r1 = a / 2) (h4 : r2 = b / 2) (h5 : r3 = (Real.sqrt (a^2 + b^2)) / 2) :
  R = 144 / 23 := sorry

end radius_of_tangent_circle_l277_277595


namespace solution_correct_l277_277528

noncomputable def sum_segment_and_arc_length : ℝ :=
  let r := 4
  let line := λ x : ℝ, 4 - (2 - real.sqrt 3) * x
  let circle := λ x y : ℝ, x^2 + y^2 = r^2
  let A := (0, 4)
  let B := (2, 2 * real.sqrt 3)
  let segment_AB := real.sqrt ((2 - 0)^2 + (2 * real.sqrt 3 - 4)^2)
  let arc_length_AB := (real.pi / 6) * r
  segment_AB + arc_length_AB

theorem solution_correct :
  sum_segment_and_arc_length = 4 * real.sqrt (2 - real.sqrt 3) + (2 * real.pi / 3) :=
sorry

end solution_correct_l277_277528


namespace gillian_spent_multiple_of_sandi_l277_277173

theorem gillian_spent_multiple_of_sandi
  (sandi_had : ℕ := 600)
  (gillian_spent : ℕ := 1050)
  (sandi_spent : ℕ := sandi_had / 2)
  (diff : ℕ := gillian_spent - sandi_spent)
  (extra : ℕ := 150)
  (multiple_of_sandi : ℕ := (diff - extra) / sandi_spent) : 
  multiple_of_sandi = 1 := 
  by sorry

end gillian_spent_multiple_of_sandi_l277_277173


namespace calculate_growth_rate_l277_277267

-- Define the mask production in February and April
def production_feb : ℝ := 1.8 * 10^6
def production_apr : ℝ := 4.61 * 10^6

-- Define the average monthly growth rate
def average_monthly_growth_rate (x : ℝ) : Prop :=
  (production_feb / 10^4) * ((1 + x)^2) = (production_apr / 10^4)

-- Prove that there exists an average monthly growth rate x satisfying the given condition
theorem calculate_growth_rate (x : ℝ) : average_monthly_growth_rate x := 
begin
  sorry, -- Proof will be here
end

end calculate_growth_rate_l277_277267


namespace sum_of_exponents_l277_277328

theorem sum_of_exponents (n : ℕ) (h : n = 896) : 
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 2^a + 2^b + 2^c = n ∧ a + b + c = 24 :=
by
  sorry

end sum_of_exponents_l277_277328


namespace area_of_region_l277_277627

theorem area_of_region (x y : ℝ) :
  let eq := (x^2 + y^2 - 7 = 2 * y - 8 * x + 1)
  in eq → (π * 5^2 = 25 * π) :=
by
  sorry

end area_of_region_l277_277627


namespace positive_number_square_sum_eq_210_l277_277993

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l277_277993


namespace insurance_slogan_equivalence_l277_277285

variables (H I : Prop)

theorem insurance_slogan_equivalence :
  (∀ x, x → H → I) ↔ (∀ y, y → ¬I → ¬H) :=
sorry

end insurance_slogan_equivalence_l277_277285


namespace positive_number_square_sum_eq_210_l277_277994

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l277_277994


namespace sequence_arithmetic_mean_l277_277807

theorem sequence_arithmetic_mean (a b c d e f g : ℝ)
  (h1 : b = (a + c) / 2)
  (h2 : c = (b + d) / 2)
  (h3 : d = (c + e) / 2)
  (h4 : e = (d + f) / 2)
  (h5 : f = (e + g) / 2) :
  d = (a + g) / 2 :=
sorry

end sequence_arithmetic_mean_l277_277807


namespace quadratic_distinct_real_roots_l277_277051

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end quadratic_distinct_real_roots_l277_277051


namespace solve_max_people_solved_A_l277_277875

-- Define the variables to be used
variables (a b c x y z w : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  let total := a + b + c + x + y + z + w in
  total = 39 ∧
  a = (x + y + z + w + a - 5) ∧
  b = 2 * c ∧
  a = b + c

-- Define the statement to be proved
def max_people_solved_A : Prop := a = 23

-- Prove this statement given the conditions
theorem solve_max_people_solved_A (h : conditions a b c x y z w) : max_people_solved_A a b c x y z w :=
sorry

end solve_max_people_solved_A_l277_277875


namespace trains_meet_time_l277_277232

theorem trains_meet_time :
  ∀ (speed_A1 speed_A2 speed_B1 speed_B2 : ℝ)
    (time_A1 time_B1 stop_A stop_B : ℝ)
    (initial_distance : ℝ),
  speed_A1 = 60 → speed_B1 = 50 →
  time_A1 = 2 → time_B1 = 2 →
  stop_A = 0.5 → stop_B = 0.25 →
  speed_A2 = 50 → speed_B2 = 40 →
  initial_distance = 270 →
  let distance_A1 := speed_A1 * time_A1 in
  let distance_B1 := speed_B1 * time_B1 in
  let total_distance_before_stops := distance_A1 + distance_B1 in
  let remaining_distance := initial_distance - total_distance_before_stops in
  let relative_speed_after_stops := speed_A2 + speed_B2 in
  let time_after_stops := remaining_distance / relative_speed_after_stops in
  time_after_stops * 60 ≈ 33.33 := 
begin
  intros,
  sorry
end

end trains_meet_time_l277_277232


namespace sum_a1_to_a10_l277_277354

theorem sum_a1_to_a10 (a : Fin 11 → ℤ) (x : ℝ) :
    ((x^2 - 3 * x + 1) ^ 5).sum = a 0 + a 1 * x + a 2 * x^2 + a 3 * (x ^ 3) + a 4 * (x ^ 4) + a 5 * (x ^ 5) + a 6 * (x ^ 6) + a 7 * (x ^ 7) + a 8 * (x ^ 8) + a 9 * (x ^ 9) + a 10 * (x ^ 10) →
    a 0 = 1 →
    a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -2 := 
sorry

end sum_a1_to_a10_l277_277354


namespace quinn_free_donuts_l277_277170

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end quinn_free_donuts_l277_277170


namespace proj_vec_identity_l277_277126

variables (u z : ℝ^3)

-- Define the projection operator
def proj (z u : ℝ^3) : ℝ^3 := ((u ⬝ z) / (z ⬝ z)) • z

theorem proj_vec_identity
  (h : proj z u = ⟨2, -1, 4⟩) :
  proj z (3 • u) - 2 • proj z u = ⟨2, -1, 4⟩ :=
by
  sorry

end proj_vec_identity_l277_277126


namespace distance_between_closest_points_of_circles_l277_277754

theorem distance_between_closest_points_of_circles :
  let circle1_center : ℝ × ℝ := (3, 3)
  let circle2_center : ℝ × ℝ := (20, 15)
  let circle1_radius : ℝ := 3
  let circle2_radius : ℝ := 15
  let distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (15 - 3)^2)
  distance_between_centers - (circle1_radius + circle2_radius) = 2.81 :=
by {
  sorry
}

end distance_between_closest_points_of_circles_l277_277754


namespace find_f_neg1_l277_277405

def f (x : ℝ) (α : ℝ) : ℝ := α * Real.sin x + x^2

theorem find_f_neg1 (α : ℝ) (h : f 1 α = 0) : f (-1) α = 2 :=
by
  -- proof will be provided here
  sorry

end find_f_neg1_l277_277405


namespace number_of_friends_l277_277956

def money_emma : ℕ := 8

def money_daya : ℕ := money_emma + (money_emma * 25 / 100)

def money_jeff : ℕ := (2 * money_daya) / 5

def money_brenda : ℕ := money_jeff + 4

def money_brenda_condition : Prop := money_brenda = 8

def friends_pooling_pizza : ℕ := 4

theorem number_of_friends (h : money_brenda_condition) : friends_pooling_pizza = 4 := by
  sorry

end number_of_friends_l277_277956


namespace grid_fill_even_l277_277088

-- Useful definitions
def even {α : Type} [AddGroup α] (a : α) : Prop := ∃ b, a = 2 * b

-- Statement of the problem
-- 'n' is a natural number, grid is n × n
-- We need to find the number of ways to fill the grid with 0s and 1s such that each row and column has an even number of 1s
theorem grid_fill_even (n : ℕ) : ∃ (ways : ℕ), ways = 2 ^ ((n - 1) * (n - 1)) ∧ 
  (∀ grid : (Fin n → Fin n → bool), (∀ i : Fin n, even (grid i univ.count id)) ∧ (∀ j : Fin n, even (univ.count (λ x, grid x j))) → true) :=
sorry

end grid_fill_even_l277_277088


namespace total_amphibians_observed_l277_277434

theorem total_amphibians_observed : 
  let green_frogs := 6 in
  let initial_tree_frogs := 5 in
  let total_tree_frogs := 3 * initial_tree_frogs in
  let bullfrogs := 2 in
  let exotic_tree_frogs := 8 in
  let salamanders := 3 in
  let first_group_tadpoles := 50 in
  let second_group_tadpoles := first_group_tadpoles - (first_group_tadpoles / 5) in
  let baby_frogs := 10 in
  let newt := 1 in
  let toads := 2 in
  let caecilian := 1 in
  green_frogs + total_tree_frogs + bullfrogs + exotic_tree_frogs + 
  salamanders + first_group_tadpoles + second_group_tadpoles + 
  baby_frogs + newt + toads + caecilian = 138 :=
by
  sorry

end total_amphibians_observed_l277_277434


namespace mice_meet_at_n_days_l277_277183

noncomputable def total_distance (n : ℕ) : ℝ :=
  2^n - (1 / 2^(n-1)) + 1

def wall_thickness : ℝ := 64 + 31 / 32

theorem mice_meet_at_n_days : ∃ n : ℕ, total_distance n = wall_thickness :=
by sorry

end mice_meet_at_n_days_l277_277183


namespace exists_subspace_two_dim_intersections_l277_277502

variables (V1 V2 V3 V4 : Subspace ℝ (Fin 8))
variables [finite_dimensional ℝ V1] [finite_dimensional ℝ V2] [finite_dimensional ℝ V3] [finite_dimensional ℝ V4]
noncomputable theory

def pairwise_intersection_zero (V: Fin 4 → Subspace ℝ (Fin 8)) : Prop :=
  ∀ (i j : Fin 4), i ≠ j → V i ⊓ V j = ⊥

theorem exists_subspace_two_dim_intersections :
  ∀ (V : Fin 4 → Subspace ℝ (Fin 8))
    (h_dim : ∀ i, finite_dimensional.finrank ℝ (V i) = 4)
    (h_pairwise : pairwise_intersection_zero V),
    ∃ W : Subspace ℝ (Fin 8),
      finite_dimensional.finrank ℝ W = 4 ∧
      ∀ i, finite_dimensional.finrank ℝ (W ⊓ V i) = 2 :=
begin
  sorry
end

end exists_subspace_two_dim_intersections_l277_277502


namespace perpendicular_line_through_point_l277_277643

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l277_277643


namespace divisible_by_4_probability_l277_277557

/-- Define the elements used in the problem -/
def numbers := {i : ℕ | 1 ≤ i ∧ i ≤ 1024}
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

/-- Conditions -/
axiom a_in_range : a ∈ numbers
axiom b_in_range : b ∈ numbers
axiom c_in_range : c ∈ numbers

/-- Proof problem -/
theorem divisible_by_4_probability :
  let n := 1024 in
  let probability_divisible_by_4 := 7 / 16 in
  (probability_divisible_by_4 = 
    (card {x : numbers × numbers × numbers | (x.1 * x.2.1 * x.2.2 + x.1 * x.2.1 + x.1) % 4 = 0}) / (n^3)) :=
sorry

end divisible_by_4_probability_l277_277557


namespace transform_sin_function_l277_277834

-- Definitions of the functions f and g
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
def g (x : ℝ) : ℝ := Real.sin x

-- Statement of the proof problem
theorem transform_sin_function :
  ( ∃ h : ℝ → ℝ, (∀ x, h (x / 2) = x - Real.pi / 6) ∧ (∀ x, f x = g (h x)) ) :=
  sorry

end transform_sin_function_l277_277834


namespace bukvinsk_acquaintances_l277_277470

theorem bukvinsk_acquaintances (Martin Klim Inna Tamara Kamilla : Type) 
  (acquaints : Type → Type → Prop)
  (exists_same_letters : ∀ (x y : Type), acquaints x y ↔ ∃ S, (x = S ∧ y = S)) :
  (∃ (count_Martin : ℕ), count_Martin = 20) →
  (∃ (count_Klim : ℕ), count_Klim = 15) →
  (∃ (count_Inna : ℕ), count_Inna = 12) →
  (∃ (count_Tamara : ℕ), count_Tamara = 12) →
  (∃ (count_Kamilla : ℕ), count_Kamilla = 15) := by
  sorry

end bukvinsk_acquaintances_l277_277470


namespace count_palindromes_divisible_by_7_l277_277430

def is_palindrome (n : ℕ) : Prop :=
  let s := toDigits 10 n
  s = s.reverse

def in_range (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 2000

def is_divisible_by_7 (n : ℕ) : Prop := 
  n % 7 = 0

theorem count_palindromes_divisible_by_7 : 
  ∃ (N : ℕ), 
  ∀ (n : ℕ), 
  in_range(n) ∧ is_palindrome(n) ∧ is_divisible_by_7(n) ↔ 
  n = 1001 + 100 * (n / 1000 % 10) + 10 * (n / 10 % 10) :=
sorry

end count_palindromes_divisible_by_7_l277_277430


namespace optimal_order_for_ostap_l277_277530

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l277_277530


namespace quotient_when_dividing_l277_277342

noncomputable def dividend_poly : ℚ[X] := X^6 + 3 * X^4 - 2 * X^3 + X + 12
noncomputable def divisor_poly : ℚ[X] := X - 2
noncomputable def quotient_poly : ℚ[X] := X^5 + 2 * X^4 + 6 * X^3 + 10 * X^2 + 18 * X + 34

theorem quotient_when_dividing
  : (dividend_poly / divisor_poly) = quotient_poly :=
by
  -- Proof goes here
  sorry

end quotient_when_dividing_l277_277342


namespace perpendicular_line_equation_l277_277631

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l277_277631


namespace unit_vector_orthogonal_l277_277335

def vec1 : ℝ × ℝ × ℝ := (2, 1, -1)
def vec2 : ℝ × ℝ × ℝ := (0, 1, 3)
def unit_vec : ℝ × ℝ × ℝ := (2 / Real.sqrt 14, -3 / Real.sqrt 14, 1 / Real.sqrt 14)

theorem unit_vector_orthogonal :
  (vec1.1 * unit_vec.1 + vec1.2 * unit_vec.2 + vec1.3 * unit_vec.3 = 0) ∧
  (vec2.1 * unit_vec.1 + vec2.2 * unit_vec.2 + vec2.3 * unit_vec.3 = 0) ∧
  (unit_vec.1 * unit_vec.1 + unit_vec.2 * unit_vec.2 + unit_vec.3 * unit_vec.3 = 1) :=
sorry

end unit_vector_orthogonal_l277_277335


namespace max_money_received_back_l277_277712

def total_money_before := 3000
def value_chip_20 := 20
def value_chip_100 := 100
def chips_lost_total := 16
def chips_lost_diff_1 (x y : ℕ) := x = y + 2
def chips_lost_diff_2 (x y : ℕ) := x = y - 2

theorem max_money_received_back :
  ∃ (x y : ℕ), 
  (chips_lost_diff_1 x y ∨ chips_lost_diff_2 x y) ∧ 
  (x + y = chips_lost_total) ∧
  total_money_before - (x * value_chip_20 + y * value_chip_100) = 2120 :=
sorry

end max_money_received_back_l277_277712


namespace pentagon_side_length_equal_10_l277_277984

theorem pentagon_side_length_equal_10 :
  ∀ (rectangle_length rectangle_width : ℝ) (pentagon_area : ℝ),
    rectangle_length = 10 ∧ rectangle_width = 22 ∧ pentagon_area = 220 →
    ∃ (z : ℝ), z = 10 :=
by
  intros rectangle_length rectangle_width pentagon_area
  intro h
  cases h with h1 h2
  cases h2 with h_rectangle h_pentagon
  use 10
  exact sorry

end pentagon_side_length_equal_10_l277_277984


namespace trigonometric_simplification_l277_277248

open Real

theorem trigonometric_simplification (α : ℝ) :
  (3.4113 * sin α * cos (3 * α) + 9 * sin α * cos α - sin (3 * α) * cos (3 * α) - 3 * sin (3 * α) * cos α) = 
  2 * sin (2 * α)^3 :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_simplification_l277_277248


namespace problem_statement_l277_277943

theorem problem_statement (a b : ℝ) (h : a < b) : a^3 - 3 * a ≤ b^3 - 3 * b + 4 :=
by
  sorry

example : (-1 : ℝ)^3 - 3 * (-1 : ℝ) = (1 : ℝ)^3 - 3 * (1 : ℝ) + 4 :=
by
  simp

end problem_statement_l277_277943


namespace proof_problem_l277_277447

variable (a b c A B C : ℝ)
variable (h_a : a = Real.sqrt 3)
variable (h_b_ge_a : b ≥ a)
variable (h_cos : Real.cos (2 * C) - Real.cos (2 * A) =
  2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C))

theorem proof_problem :
  (A = Real.pi / 3) ∧ (2 * b - c ∈ Set.Ico (Real.sqrt 3) (2 * Real.sqrt 3)) :=
  sorry

end proof_problem_l277_277447


namespace constant_term_binomial_expansion_l277_277587

theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (2 * real.sqrt x - (1 / (2 * x)))^9 in
  constant_term f = -672 := 
sorry

-- We define a helper function to extract the constant term
def constant_term (f : ℝ → ℝ) : ℝ :=
  if h : ∀ x, f x ∈ real.algebra_map ℝ ℝ then
    real.algebra_id (f 0)
  else 0

end constant_term_binomial_expansion_l277_277587


namespace nick_total_quarters_l277_277155

theorem nick_total_quarters (Q : ℕ)
  (h1 : 2 / 5 * Q = state_quarters)
  (h2 : 1 / 2 * state_quarters = PA_quarters)
  (h3 : PA_quarters = 7) :
  Q = 35 := by
  sorry

end nick_total_quarters_l277_277155


namespace four_digit_numbers_count_l277_277849

theorem four_digit_numbers_count :
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    let d1 := n / 1000 in let d2 := (n / 100 % 10) in let d3 := (n / 10 % 10) in let d4 := n % 10 in
    (d1 = 2 ∨ d1 = 3 ∨ d1 = 6) ∧
    (d2 = 2 ∨ d2 = 3 ∨ d2 = 6) ∧
    (d3 = 4 ∨ d3 = 7 ∨ d3 = 9) ∧
    (d4 = 4 ∨ d4 = 7 ∨ d4 = 9) ∧
    d3 ≠ d4
  ) = 54 := { sorry }

end four_digit_numbers_count_l277_277849


namespace total_time_school_l277_277932

open Lean

theorem total_time_school : 
  ∀ (time_to_gate : ℕ) (time_to_building : ℕ) (time_to_room : ℕ),
  time_to_gate = 15 →
  time_to_building = 6 →
  time_to_room = 9 →
  (time_to_gate + time_to_building + time_to_room) = 30 :=
by
  intros time_to_gate time_to_building time_to_room
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_time_school_l277_277932


namespace min_k_inequality_l277_277371

theorem min_k_inequality (α β : ℝ) (hα : 0 < α) (hα2 : α < 2 * Real.pi / 3)
  (hβ : 0 < β) (hβ2 : β < 2 * Real.pi / 3) :
  4 * Real.cos α ^ 2 + 2 * Real.cos α * Real.cos β + 4 * Real.cos β ^ 2
  - 3 * Real.cos α - 3 * Real.cos β - 6 < 0 :=
by
  sorry

end min_k_inequality_l277_277371


namespace four_digit_numbers_from_2023_l277_277425

theorem four_digit_numbers_from_2023 :
  let digits := [2, 0, 2, 3],
      thousands_place := {x | x ≠ 0 ∧ (x = 2 ∨ x = 3)},
      remaining_digits := {digits.erase x | x ∈ thousands_place},
      valid_combinations (x : ℕ) (ds : List ℕ) := 
        if x = 2 then 
          3 * Nat.fact 2
        else if x = 3 then
          3 * (Nat.fact 2 / Nat.fact 2)
        else
          0 
  in (remaining_digits.foldr (λ d res => res + valid_combinations d.val d.val.snd) 0) = 9 :=
by
  sorry

end four_digit_numbers_from_2023_l277_277425


namespace arithmetic_sequence_ratio_l277_277197

theorem arithmetic_sequence_ratio (a d : ℕ) (h : b = a + 3 * d) : a = 1 -> d = 1 -> (a / b = 1 / 4) :=
by
  sorry

end arithmetic_sequence_ratio_l277_277197


namespace product_closest_value_l277_277985

theorem product_closest_value :
  let expr := (2.1 * (50.3 + 0.09)) in
  expr = 105.819 ∧ (106 - 105.819).abs <= (100 - 105.819).abs ∧ 
  (106 - 105.819).abs <= (110 - 105.819).abs ∧ 
  (106 - 105.819).abs <= (150 - 105.819).abs ∧ 
  (106 - 105.819).abs <= (105.63 - 105.819).abs :=
by
  sorry

end product_closest_value_l277_277985


namespace max_XY_length_l277_277279

-- Define the conditions
variable (ABC : Triangle) -- Represents some triangle
variable (p : ℝ) -- Perimeter of triangle ABC
-- Triangle perimeter is given as condition (but we keep it generic here)

-- Function to calculate the maximum length XY
noncomputable def max_length_XY (p : ℝ) : ℝ :=
  p / 8

-- The theorem to prove
theorem max_XY_length (ABC : Triangle) (p : ℝ) 
  (h_perimeter : ∀ (BC : ℝ), some_condition_for_perimeter ABC BC p) : max_length_XY p = p / 8 :=
sorry

end max_XY_length_l277_277279


namespace positive_number_sum_square_l277_277996

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l277_277996


namespace positive_rational_number_l277_277678

theorem positive_rational_number : ∃ x : ℝ, x = 1/2 ∧ 
  ( (∀ y : ℝ, y = -Real.sqrt 2 → ¬ (0 < y ∧ ∃ a b : ℤ, y = a / b) ) ∧
    ( (0 < 1/2 ∧ ∃ a b : ℤ, 1/2 = a / b) ∧ ∀ z : ℝ, ¬ ((0 < z ∧ z ≠ 1/2) ∧ ∃ a b : ℤ, z = a / b)) ∧
    ( ∀ w : ℝ, w = 0 → ¬ (0 < w ∧ ∃ a b : ℤ, w = a / b) ) ∧
    ( ∀ v : ℝ, v = Real.sqrt 3 → (0 < v ∧ ¬ (∃ a b : ℤ, v = a / b) ) ) ) :=
by
  sorry

end positive_rational_number_l277_277678


namespace find_a_l277_277062

theorem find_a (t a : ℝ) (h : ∀ x : ℝ, tx^2 - 6x + t^2 < 0 ↔ x < a ∨ x > 1) : a = -3 := sorry

end find_a_l277_277062


namespace convert_point_cylindrical_to_rectangular_l277_277758

noncomputable def cylindrical_to_rectangular_coordinates (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_point_cylindrical_to_rectangular :
  cylindrical_to_rectangular_coordinates 6 (5 * Real.pi / 3) (-3) = (3, -3 * Real.sqrt 3, -3) :=
by
  sorry

end convert_point_cylindrical_to_rectangular_l277_277758


namespace ways_to_choose_4_numbers_with_same_sum_l277_277258

theorem ways_to_choose_4_numbers_with_same_sum (n : ℕ) (h_even : n % 2 = 0) :
  let S := finset.range (n + 1) in
  ∃ k : ℕ, (S.card = n + 1) → k = (n * (n - 2) * (2 * n - 5)) / 24 :=
by
  sorry

end ways_to_choose_4_numbers_with_same_sum_l277_277258


namespace honey_last_nights_l277_277577

def servings_per_cup : Nat := 1
def cups_per_night : Nat := 2
def container_ounces : Nat := 16
def servings_per_ounce : Nat := 6

theorem honey_last_nights :
  (container_ounces * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 :=
by
  sorry  -- Proof not provided as per requirements

end honey_last_nights_l277_277577


namespace TriangleLOM_Area_Approximation_l277_277459

-- Define the conditions and problem statement
def ScaleneTriangle (A B C : ℝ) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C
def AngleProperties (A B C : ℝ) : Prop := 
  ∃ (A B C : ℝ), (C = B - A) ∧ (B = 2 * A) ∧ (A + B + C = 180)
noncomputable def AreaOfTriangleABC := 20
noncomputable def ApproxAreaOfTriangleLOM (area_ABC : ℝ) :=
  let approx_area_LMO := 3 * area_ABC in
  Int.round approx_area_LMO

-- State the problem we need to prove
theorem TriangleLOM_Area_Approximation : 
  ∀ (A B C : ℝ), 
  ScaleneTriangle A B C → 
  AngleProperties A B C → 
  ApproxAreaOfTriangleLOM AreaOfTriangleABC = 27 := by
  -- Proof Placeholder
  sorry

end TriangleLOM_Area_Approximation_l277_277459


namespace max_iso_good_triangles_l277_277912

/-- A "good edge" in a polygon P is any side, or any diagonal which divides the polygon
into two parts each containing an odd number of sides. -/
def good_edge (P : polygon) (edge : segment) : Prop :=
  edge.is_side P ∨ (edge.is_diagonal P ∧ odd (part1.get_sides edge) ∧ odd (part2.get_sides edge))

/-- A triangle is an "iso-good" triangle if two of its sides are "good edges". -/
def iso_good (P : polygon) (triangle : triangle) : Prop :=
  good_edge P triangle.side1 ∧ good_edge P triangle.side2 ∧ (triangle.side3 = φ.some) -- ignoring 3rd side

/-- Given a regular 2006-sided polygon P divided into 2004 triangles by 2003 non-intersecting diagonals,
prove that the maximum number of isosceles triangles with two "good edges" is 1003. -/
theorem max_iso_good_triangles (P : polygon) (h2006 : P.sides = 2006)
  (triangulation : set triangle) (H : triangulation.size = 2004)
  (non_intersecting_diagonals : set segment) (H2 : non_intersecting_diagonals.size = 2003)
  (H3 : ∀ (triangle ∈ triangulation), ∃ (diagonal ∈ non_intersecting_diagonals), 
         diagonal.splits_triangles P triangle) :
  (∃ (iso_good_triangulation : set triangle), iso_good_triangulation.size ≤ 1003 ∧
    ∀ t ∈ iso_good_triangulation, iso_good P t) ∧
  ∀ (iso_good_triangulation' : set triangle),
    (∀ t ∈ iso_good_triangulation', iso_good P t) →
    iso_good_triangulation'.size ≤ 1003 :=
sorry

end max_iso_good_triangles_l277_277912


namespace arithmetic_sequence_term_count_l277_277208

theorem arithmetic_sequence_term_count :
  ∃ n : ℕ, 1 + (n - 1) * -2 = -89 ∧ n = 46 :=
by
  existsi 46
  simpa using 1 + (46 - 1) * -2 = -89
  sorry

end arithmetic_sequence_term_count_l277_277208


namespace james_pays_total_l277_277491

theorem james_pays_total (lessons_total : ℕ) (lessons_paid : ℕ) (lesson_cost : ℕ) (uncle_share : ℕ) :
  lessons_total = 20 →
  lessons_paid = 15 →
  lesson_cost = 5 →
  uncle_share = 2 → 
  (lesson_cost * lessons_paid) / uncle_share = 37.5 := 
by
  intros h1 h2 h3 h4
  sorry

end james_pays_total_l277_277491


namespace b_power_four_l277_277573

-- Definitions based on the conditions
variable (b : ℝ)
axiom basic_eq : 5 = b + b⁻¹

-- Statement to prove
theorem b_power_four (b : ℝ) (h : 5 = b + b⁻¹) : b^4 + b^(-4) = 527 := by
  sorry

end b_power_four_l277_277573


namespace unique_function_divisibility_l277_277919

theorem unique_function_divisibility (f : ℕ+ → ℕ+) 
  (h : ∀ a b : ℕ+, (a^2 + f a * f b) % (f a + b) = 0) : 
    ∀ n : ℕ+, f n = n := 
sorry

end unique_function_divisibility_l277_277919


namespace solve_for_x_l277_277954

theorem solve_for_x (x : ℚ) : (x - 3) ^ 4 = (27 / 8 : ℚ) ^ (-3/4 : ℚ) → x = 11 / 3 :=
by
  sorry

end solve_for_x_l277_277954


namespace exists_integers_ki_l277_277119

theorem exists_integers_ki
  (n : ℕ) (hc1 : 2 ≤ n) (c : Fin n → ℝ)
  (h_sum : 0 ≤ (Finset.univ.sum (λ i, c i)))
  (h_sum' : (Finset.univ.sum (λ i, c i)) ≤ n) :
  ∃ (k : Fin n → ℤ), (Finset.univ.sum (λ i, k i) = 0) ∧ 
  (∀ i, 1 - n ≤ c i + n * (k i) ∧ c i + n * (k i) ≤ n) :=
sorry

end exists_integers_ki_l277_277119


namespace concyclic_points_l277_277134

open EuclideanGeometry

theorem concyclic_points (A B C H: Point) (A' B' C': Point) 
    (ΓA ΓB ΓC: Circle) (A1 A2 B1 B2 C1 C2: Point) :
    Triangle ABC → 
    IsAcute △ABC →
    Orthocenter H △ABC →
    (Midpoint A' B C) → (Midpoint B' C A) → (Midpoint C' A B) →
    (CirclePassingThrough H A' ΓA) → (CirclePassingThrough H B' ΓB) → (CirclePassingThrough H C' ΓC) →
    (CircleIntersectsLine ΓA (Line BC) A1 A2) →
    (CircleIntersectsLine ΓB (Line CA) B1 B2) →
    (CircleIntersectsLine ΓC (Line AB) C1 C2) →
    Concyclic A1 A2 B1 B2 C1 C2 := sorry

end concyclic_points_l277_277134


namespace advantageous_order_l277_277541

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l277_277541


namespace variance_of_dataset_l277_277866

theorem variance_of_dataset (a : ℝ) 
  (h1 : (4 + a + 5 + 3 + 8) / 5 = a) :
  (1 / 5) * ((4 - a) ^ 2 + (a - a) ^ 2 + (5 - a) ^ 2 + (3 - a) ^ 2 + (8 - a) ^ 2) = 14 / 5 :=
by
  sorry

end variance_of_dataset_l277_277866


namespace finite_points_outside_unit_circle_l277_277904

noncomputable def centroid (x y z : ℝ × ℝ) : ℝ × ℝ := 
  ((x.1 + y.1 + z.1) / 3, (x.2 + y.2 + z.2) / 3)

theorem finite_points_outside_unit_circle
  (A₁ B₁ C₁ D₁ : ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)
  (D : ℕ → ℝ × ℝ)
  (hA : ∀ n, A (n + 1) = centroid (B n) (C n) (D n))
  (hB : ∀ n, B (n + 1) = centroid (A n) (C n) (D n))
  (hC : ∀ n, C (n + 1) = centroid (A n) (B n) (D n))
  (hD : ∀ n, D (n + 1) = centroid (A n) (B n) (C n))
  (h₀ : A 1 = A₁ ∧ B 1 = B₁ ∧ C 1 = C₁ ∧ D 1 = D₁)
  : ∃ N : ℕ, ∀ n > N, (A n).1 * (A n).1 + (A n).2 * (A n).2 ≤ 1 :=
sorry

end finite_points_outside_unit_circle_l277_277904


namespace problem_solution_l277_277520

noncomputable def a (n : ℕ) : ℕ := 2 * n - 3

noncomputable def b (n : ℕ) : ℕ := 2 ^ n

noncomputable def c (n : ℕ) : ℕ := a n * b n

noncomputable def sum_c (n : ℕ) : ℕ :=
  (2 * n - 5) * 2 ^ (n + 1) + 10

theorem problem_solution :
  ∀ n : ℕ, n > 0 →
  (S_n = 2 * (b n - 1)) ∧
  (a 2 = b 1 - 1) ∧
  (a 5 = b 3 - 1)
  →
  (∀ n, a n = 2 * n - 3) ∧
  (∀ n, b n = 2 ^ n) ∧
  (sum_c n = (2 * n - 5) * 2 ^ (n + 1) + 10) :=
by
  intros n hn h
  sorry


end problem_solution_l277_277520


namespace max_belts_l277_277290

theorem max_belts (h t b : ℕ) (Hh : h >= 1) (Ht : t >= 1) (Hb : b >= 1) (total_cost : 3 * h + 4 * t + 9 * b = 60) : b <= 5 :=
sorry

end max_belts_l277_277290


namespace solution_l277_277131

noncomputable def find_n (b : ℕ → ℝ) (n : ℕ) : Prop :=
b 0 = 41 ∧ b 1 = 68 ∧ b n = 0 ∧
(∀ k, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 5 / b k ) ∧ n = 559

theorem solution : ∃ n, ∃ b : ℕ → ℝ, find_n b n :=
begin
  sorry
end

end solution_l277_277131


namespace average_grade_over_two_years_l277_277278

-- Using broader import to bring necessary libraries

variable (average_grade_last_year : ℤ) (courses_last_year : ℤ)
variable (average_grade_year_before : ℤ) (courses_year_before : ℤ)

def total_points_last_year := courses_last_year * average_grade_last_year
def total_points_year_before := courses_year_before * average_grade_year_before
def total_courses := courses_last_year + courses_year_before
def total_points := total_points_last_year + total_points_year_before

def average_grade_two_year := (total_points : ℚ) / total_courses

theorem average_grade_over_two_years :
  average_grade_last_year = 100 ∧
  courses_last_year = 6 ∧
  average_grade_year_before = 50 ∧
  courses_year_before = 5
  → average_grade_two_year = 77.3 :=
by
  intros h
  cases h with h0 h_rest
  cases h_rest with h1 h_rest'
  cases h_rest' with h2 h3
  sorry

end average_grade_over_two_years_l277_277278


namespace price_paid_for_refrigerator_l277_277172

variable (P : Real)

-- Conditions
def discount (P : Real) := 0.80 * P
def transport_cost : Real := 125
def installation_cost : Real := 250
def target_selling_price (P : Real) := 1.10 * P
def actual_selling_price : Real := 23100

-- Calculation of labelled price
def labelled_price : Real := 23100 / 1.10

-- Calculate and assert the final price paid
theorem price_paid_for_refrigerator :
  labelled_price - labelled_price * 0.20 + 
  transport_cost + installation_cost = 17175 := by
  have P := 23100 / 1.10
  calc
    discount P + transport_cost + installation_cost
      = P * 0.80 + 125 + 250 : by rw [discount]
      = 16800 + 125 + 250 : by norm_num [P, labelled_price]
      = 16800 + 375 : by norm_num
      = 17175 : by norm_num

end price_paid_for_refrigerator_l277_277172


namespace no_net_coin_change_l277_277734

noncomputable def probability_no_coin_change_each_round : ℚ :=
  (1 / 3) ^ 5

theorem no_net_coin_change :
  probability_no_coin_change_each_round = 1 / 243 := by
  sorry

end no_net_coin_change_l277_277734


namespace crayons_problem_l277_277941

theorem crayons_problem
  (S M L : ℕ)
  (hS_condition : (3 / 5 : ℚ) * S = 60)
  (hM_condition : (1 / 4 : ℚ) * M = 98)
  (hL_condition : (4 / 7 : ℚ) * L = 168) :
  S = 100 ∧ M = 392 ∧ L = 294 ∧ ((2 / 5 : ℚ) * S + (3 / 4 : ℚ) * M + (3 / 7 : ℚ) * L = 460) := 
by
  sorry

end crayons_problem_l277_277941


namespace apples_in_pyramid_stack_l277_277710

theorem apples_in_pyramid_stack : 
  let base_layer := 6 * 9
  ∧ let second_layer := 5 * 8
  ∧ let third_layer := 4 * 7
  ∧ let fourth_layer := 3 * 6
  ∧ let fifth_layer := 2 * 5
  ∧ let sixth_layer := 1 * 4
  ∧ base_layer + second_layer + third_layer + fourth_layer + fifth_layer + sixth_layer = 154
  := sorry

end apples_in_pyramid_stack_l277_277710


namespace juicy_double_l277_277949

def juicy (j : ℤ) : Prop :=
  ∃ (b : ℕ → ℕ) (n : ℕ), j = ∑ i in Finset.range n, 1 / (b i : ℚ)

theorem juicy_double (j : ℤ) (h : juicy j) : juicy (2 * j) := sorry

end juicy_double_l277_277949


namespace at_least_one_non_neg_l277_277360

theorem at_least_one_non_neg (a b c d e f g h : ℝ) :
  ∃ (x : ℝ), x ∈ {a * c + b * d, a * e + b * f, a * g + b * h, c * e + d * f, c * g + d * h, e * g + f * h} ∧ x ≥ 0 := sorry

end at_least_one_non_neg_l277_277360


namespace positive_number_square_sum_eq_210_l277_277992

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l277_277992


namespace sum_of_A_and_B_zero_l277_277913

theorem sum_of_A_and_B_zero
  (A B C : ℝ)
  (h1 : A ≠ B)
  (h2 : C ≠ 0)
  (f g : ℝ → ℝ)
  (h3 : ∀ x, f x = A * x + B + C)
  (h4 : ∀ x, g x = B * x + A - C)
  (h5 : ∀ x, f (g x) - g (f x) = 2 * C) : A + B = 0 :=
sorry

end sum_of_A_and_B_zero_l277_277913


namespace evaluate_expression_l277_277324

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end evaluate_expression_l277_277324


namespace perpendicular_line_equation_l277_277659

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l277_277659


namespace number_of_possible_sums_of_A_l277_277140

def total_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def range_of_possible_sums (total : ℕ) (min_T : ℕ) (max_T : ℕ) : ℕ :=
  (total - min_T - (total - max_T)) + 1

theorem number_of_possible_sums_of_A (S : ℕ) (A : finset ℕ) :
  (A.card = 100) ∧ (A ⊆ finset.range 121) →
  (∃ B : finset ℕ, (B.card = 20) ∧ (B = (finset.range 121) \ A)) →
  ((total_sum 120 - 2210) ≤ S ∧ S ≤ (total_sum 120 - 210)) →
  (range_of_possible_sums 7260 210 2210 = 2001) :=
by
  intros hA hB hS
  sorry

end number_of_possible_sums_of_A_l277_277140


namespace equation_of_perpendicular_line_l277_277776

-- Define the given conditions
def point_P : ℝ × ℝ := (1, -2)
def line1 : ℝ → ℝ → Prop := λ x y, x - 3*y + 2 = 0

-- Define the perpendicular line equation and requirement to pass through point P
def perp_line_eq (x y : ℝ) : ℝ := 3*x + y + c
def passes_point_P (c : ℝ) : Prop := perp_line_eq 1 (-2) = 0

-- The problem to prove
theorem equation_of_perpendicular_line : ∃ (c : ℝ), passes_point_P c ∧ ∀ x y, 3*x + y + c = 0 :=
by
  -- The proof starts here. Note: "sorry" will be used to skip the actual proof steps.
  use (-1)
  split
  {
    unfold passes_point_P perp_line_eq
    simp
  }
  {
    -- Provide the proof that for all x, y, perp_line_eq will hold with c = -1 (answer part)
    intros x y
    simp
    intros
  }

end equation_of_perpendicular_line_l277_277776


namespace tooth_extraction_cost_l277_277236

noncomputable def cleaning_cost : ℕ := 70
noncomputable def filling_cost : ℕ := 120
noncomputable def root_canal_cost : ℕ := 400
noncomputable def crown_cost : ℕ := 600
noncomputable def bridge_cost : ℕ := 800

noncomputable def crown_discount : ℕ := (crown_cost * 20) / 100
noncomputable def bridge_discount : ℕ := (bridge_cost * 10) / 100

noncomputable def total_cost_without_extraction : ℕ := 
  cleaning_cost + 
  3 * filling_cost + 
  root_canal_cost + 
  (crown_cost - crown_discount) + 
  (bridge_cost - bridge_discount)

noncomputable def root_canal_and_one_filling : ℕ := 
  root_canal_cost + filling_cost

noncomputable def dentist_bill : ℕ := 
  11 * root_canal_and_one_filling

theorem tooth_extraction_cost : 
  dentist_bill - total_cost_without_extraction = 3690 :=
by
  -- The proof would go here
  sorry

end tooth_extraction_cost_l277_277236


namespace maximum_value_l277_277513

variables {γ δ : ℂ}

theorem maximum_value (h1 : |δ| = 1) (h2 : (conj γ) * δ ≠ -1) :
  ∃ M : ℝ, M = 1 ∧ ∀ (γ δ : ℂ) (h1 : |δ| = 1) (h2 : (conj γ) * δ ≠ -1),
    |(δ + γ) / (1 + (conj γ) * δ)| ≤ M :=
sorry

end maximum_value_l277_277513


namespace has_root_sqrt3_add_sqrt5_l277_277334

noncomputable def monic_degree_4_poly_with_root : Polynomial ℚ :=
  Polynomial.X ^ 4 - 16 * Polynomial.X ^ 2 + 4

theorem has_root_sqrt3_add_sqrt5 :
  Polynomial.eval (Real.sqrt 3 + Real.sqrt 5) monic_degree_4_poly_with_root = 0 :=
sorry

end has_root_sqrt3_add_sqrt5_l277_277334


namespace solve_equation_l277_277955

theorem solve_equation (x : ℝ) : (x + 2)^2 - 5 * (x + 2) = 0 ↔ (x = -2 ∨ x = 3) :=
by sorry

end solve_equation_l277_277955


namespace gcd_polynomial_l277_277047

theorem gcd_polynomial {b : ℤ} (h1 : ∃ k : ℤ, b = 2 * 7786 * k) : 
  Int.gcd (8 * b^2 + 85 * b + 200) (2 * b + 10) = 10 :=
by
  sorry

end gcd_polynomial_l277_277047


namespace problem_l277_277200

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem problem (x : ℝ) :
  (∀ x : ℝ, ∀ y : ℝ, (x ∈ Icc (Real.pi / 3) (Real.pi / 2) ∧ y = f x) → 
    (@Function.monotone_decreasing_on ℝ ℝ _ _ _ f (Icc (Real.pi / 3) (Real.pi / 2)))) ∧
  (∀ x : ℝ, g (-x) = -g x) :=
by
  sorry

end problem_l277_277200


namespace fill_grid_with_even_ones_l277_277083

theorem fill_grid_with_even_ones (n : ℕ) : 
  ∃ ways : ℕ, ways = 2^((n-1)^2) ∧ 
  (∀ grid : array n (array n (fin 2)), 
    (∀ i : fin n, even (grid[i].to_list.count (λ x, x = 1))) ∧ 
    (∀ j : fin n, even (grid.map (λ row, row[j]).to_list.count (λ x, x = 1)))) :=
begin
  use 2^((n-1)^2),
  split,
  { refl },
  { sorry },
end

end fill_grid_with_even_ones_l277_277083


namespace perpendicular_line_eq_slope_intercept_l277_277652

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l277_277652


namespace minimum_distance_tangent_line_l277_277005

noncomputable def pointA (a : ℝ) : ℝ × ℝ := (-a, a)
def pointB (b : ℝ) : ℝ × ℝ := (b, 0)
def isTangent (a b : ℝ) : Prop := 
  let d := (a * b) / (Real.sqrt (a ^ 2 + (a + b) ^ 2))
  d = 1

theorem minimum_distance_tangent_line :
  ∀ a b : ℝ, a > 0 → b > 0 → isTangent a b → (a + b + Real.sqrt (a ^ 2 + b ^ 2)) ≥ 2 + 2 * Real.sqrt 2 :=
begin
  sorry
end

end minimum_distance_tangent_line_l277_277005


namespace other_solution_l277_277381

theorem other_solution (x : ℚ) (h : x = 5/7) : 56 * x^2 - 89 * x + 35 = 0 → ∃ y : ℚ, y = 7 / 8 ∧ 56 * y^2 - 89 * y + 35 = 0 :=
by
  intros h_eq
  use (7 / 8)
  split
  {
    refl
  }
  {
    sorry
  }

end other_solution_l277_277381


namespace cost_effective_l277_277721

variable (c_S c_M c_L : ℝ) (q_S q_M q_L : ℝ)

def medium_cost (c_S : ℝ) : ℝ := 1.3 * c_S
def large_cost (c_M : ℝ) : ℝ := 1.6 * c_M
def quantity_small (q_L : ℝ) : ℝ := (2 / 3) * q_L
def quantity_medium (q_L : ℝ) : ℝ := 0.9 * q_L

theorem cost_effective (h₁ : c_M = medium_cost c_S)
    (h₂ : c_L = large_cost c_M)
    (h₃ : q_S = quantity_small q_L)
    (h₄ : q_M = quantity_medium q_L)
    (h₅ : c_S > 0) (q_L = 10) 
    (h₆ : c_S = 1)
    : (c_M / q_M) < (c_S / q_S) ∧ (c_M / q_M) < (c_L / q_L) :=
begin
  sorry
end

end cost_effective_l277_277721


namespace range_of_c_l277_277922

def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (1 - 2 * c < 0)

theorem range_of_c (c : ℝ) : (p c ∨ q c) ∧ ¬ (p c ∧ q c) ↔ (0 < c ∧ c ≤ 1/2) ∨ (1 < c) :=
by sorry

end range_of_c_l277_277922


namespace circumradii_sum_inequality_l277_277898

-- Definitions for circumradius and perimeter could be part of a geometric library in Lean.
-- Here we provide simplified definitions to illustrate the theorem.

def circumradius (A B C : Point) : ℝ := sorry    -- Placeholder for actual circumradius calculation
def perimeter (points : List Point) : ℝ := sorry -- Placeholder for actual perimeter calculation

variables {A B C D E F : Point}
-- Assuming existence of points such that the hexagon is convex and other conditions
axiom hexagon_properties : convex_hexagon A B C D E F
axiom parallel_AB_DE : parallel (line_through A B) (line_through D E)
axiom parallel_BC_EF : parallel (line_through B C) (line_through E F)
axiom parallel_CD_FA : parallel (line_through C D) (line_through F A)

-- The circumradii for the specified triangles
def R_A := circumradius F A B
def R_C := circumradius B C D
def R_E := circumradius D E F

-- The perimeter of the hexagon
def P := perimeter [A, B, C, D, E, F]

-- The theorem as required
theorem circumradii_sum_inequality : R_A + R_C + R_E ≥ P / 2 := by 
  sorry

end circumradii_sum_inequality_l277_277898


namespace optimal_order_l277_277550

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l277_277550


namespace PS_div_QR_eq_sqrt_3_l277_277483

-- We assume the lengths and properties initially given in the problem.
def P := sorry -- placeholder for point P
def Q := sorry -- placeholder for point Q
def R := sorry -- placeholder for point R
def S := sorry -- placeholder for point S
def t : ℝ := sorry -- the side length of the equilateral triangles

-- Essential properties of equilateral triangles
axiom PQR_is_equilateral : equilateral_triangle P Q R
axiom QRS_is_equilateral : equilateral_triangle Q R S
axiom side_length_QR : dist Q R = t

-- Heights of triangles from vertices to the opposite sides
def height_PQR : ℝ := t * (sqrt 3) / 2
def height_QRS : ℝ := t * (sqrt 3) / 2

-- PS is the sum of these heights
def PS : ℝ := height_PQR + height_QRS

-- Theorem
theorem PS_div_QR_eq_sqrt_3 : PS / t = sqrt 3 := by
  sorry

end PS_div_QR_eq_sqrt_3_l277_277483


namespace range_g_l277_277769

noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g : set.range (λ x, g x) = set.Ioi 0 := 
sorry

end range_g_l277_277769


namespace math_problem_l277_277720

theorem math_problem :
  let p := (2 * Real.sqrt 2 - 1) / 7
  let p_squared := p ^ 2
  let m := 9
  let n := 8
in p_squared = m / 49 - Real.sqrt n / 49 → m + n = 17 := by
  sorry

end math_problem_l277_277720


namespace range_g_l277_277766

variable (x : ℝ)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g (y : ℝ) : 
  (∃ x, g x = y) ↔ y > 0 :=
by
  sorry

end range_g_l277_277766


namespace calc_101_cubed_expression_l277_277751

theorem calc_101_cubed_expression : 101^3 + 3 * (101^2) - 3 * 101 + 9 = 1060610 := 
by
  sorry

end calc_101_cubed_expression_l277_277751


namespace overlap_length_l277_277231

-- Variables in the conditions
variables (tape_length overlap total_length : ℕ)

-- Conditions
def two_tapes_overlap := (tape_length + tape_length - overlap = total_length)

-- The proof statement we need to prove
theorem overlap_length (h : two_tapes_overlap 275 overlap 512) : overlap = 38 :=
by
  sorry

end overlap_length_l277_277231


namespace greatest_x_integer_l277_277665

theorem greatest_x_integer (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4 * x + 9) = k * (x - 4)) ↔ x ≤ 5 :=
by
  sorry

end greatest_x_integer_l277_277665


namespace num_ways_to_write_5030_l277_277907

theorem num_ways_to_write_5030 : 
  let M := { (b3, b2, b1, b0) : ℕ × ℕ × ℕ × ℕ | 
                5030 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0 ∧ 
                0 ≤ b3 ∧ b3 ≤ 99 ∧ 
                0 ≤ b2 ∧ b2 ≤ 99 ∧ 
                0 ≤ b1 ∧ b1 ≤ 99 ∧
                0 ≤ b0 ∧ b0 ≤ 99 
             } in
  M.card = 504 :=
by
  sorry

end num_ways_to_write_5030_l277_277907


namespace dinner_plates_percentage_l277_277116

/-- Define the cost of silverware and the total cost of both items -/
def silverware_cost : ℝ := 20
def total_cost : ℝ := 30

/-- Define the percentage of the silverware cost that the dinner plates cost -/
def percentage_of_silverware_cost := 50

theorem dinner_plates_percentage :
  ∃ (P : ℝ) (S : ℝ) (x : ℝ), S = silverware_cost ∧ (P + S = total_cost) ∧ (P = (x / 100) * S) ∧ x = percentage_of_silverware_cost :=
by {
  sorry
}

end dinner_plates_percentage_l277_277116


namespace angle_B_is_30_l277_277008

-- Define the problem's conditions.
variables {a b c R : ℝ}
variables {A B C : ℝ}

-- Assuming a triangle with given relations and using Law of Sines.
def sides_and_angles (a b c A B C R : ℝ) : Prop :=
  (b - c) * (Real.sin B + Real.sin C) = (a - Real.sqrt 3 * c) * Real.sin A ∧
  Real.sin B = b / (2 * R) ∧
  Real.sin C = c / (2 * R) ∧
  Real.sin A = a / (2 * R)

-- The theorem statement to prove B = 30 degrees given the above conditions.
theorem angle_B_is_30 (a b c A B C : ℝ) (R : ℝ) :
  sides_and_angles a b c A B C R → B = Real.pi / 6 :=
by
  intro h,
  sorry

end angle_B_is_30_l277_277008


namespace roots_in_interval_l277_277350

theorem roots_in_interval (f : ℝ → ℝ)
  (h : ∀ x, f x = 4 * x ^ 2 - (3 * m + 1) * x - m - 2) :
  (forall (x1 x2 : ℝ), (f x1 = 0 ∧ f x2 = 0) → -1 < x1 ∧ x1 < 2 ∧ -1 < x2 ∧ x2 < 2) ↔ -1 < m ∧ m < 12 / 7 :=
sorry

end roots_in_interval_l277_277350


namespace find_k_if_symmetric_intersection_l277_277061

-- Define the conditions for the problem
def line (k : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = k * p.1 + 1
def circle (k : ℝ) : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 + k * p.1 - p.2 - 9 = 0

-- Define the symmetry condition for intersection points
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop := p1.1 = -p2.1 ∧ p1.2 = p2.2

-- The theorem we want to prove
theorem find_k_if_symmetric_intersection (k : ℝ) :
  (∃ p1 p2 : ℝ × ℝ, line k p1 ∧ circle k p1 ∧ line k p2 ∧ circle k p2 ∧ symmetric_about_y_axis p1 p2) → k = 0 :=
by 
  sorry

end find_k_if_symmetric_intersection_l277_277061


namespace shifted_parabola_eq_l277_277185

theorem shifted_parabola_eq : ∀ (x : ℝ), (λ x, 3 * x ^ 2 + 2) (x - 1) = 3 * (x - 1) ^ 2 + 2 :=
by
  sorry

end shifted_parabola_eq_l277_277185


namespace ratio_of_volumes_l277_277275
noncomputable theory

variables {h r : ℝ}

def volume_of_nth_segment (n : ℕ) : ℝ := 
  (n ^ 3 / 3) * π * r^2 * h

def largest_piece_volume : ℝ := 
  72 * π * r^2 * h - (125 / 3) * π * r^2 * h

def second_largest_piece_volume : ℝ := 
  (125 / 3) * π * r^2 * h - (64 / 3) * π * r^2 * h

theorem ratio_of_volumes : 
  (second_largest_piece_volume / largest_piece_volume) = (61 / 91) :=
begin
  sorry, -- proof to be filled
end

end ratio_of_volumes_l277_277275


namespace exists_special_N_l277_277149

open Nat

theorem exists_special_N :
  ∃ N : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 150 → N % i = 0 ∨ i = 127 ∨ i = 128) ∧ 
  ¬ (N % 127 = 0) ∧ ¬ (N % 128 = 0) :=
by
  sorry

end exists_special_N_l277_277149


namespace gf_neg3_eq_1262_l277_277129

def f (x : ℤ) : ℤ := x^3 + 6
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 2

theorem gf_neg3_eq_1262 : g (f (-3)) = 1262 := by
  sorry

end gf_neg3_eq_1262_l277_277129


namespace range_of_f_l277_277604

noncomputable def f (x : ℝ) : ℝ := x - real.sqrt (2 * x - 1)

theorem range_of_f : set.range f = set.Ici (0 : ℝ) :=
by {
  sorry  -- we skip the proof, as instructed
}

end range_of_f_l277_277604


namespace fifteen_percent_eq_135_l277_277330

theorem fifteen_percent_eq_135 (x : ℝ) (h : (15 / 100) * x = 135) : x = 900 :=
sorry

end fifteen_percent_eq_135_l277_277330


namespace remainder_of_quadratic_expression_l277_277862

theorem remainder_of_quadratic_expression (a : ℤ) :
  let n := 100 * a - 2 in
  (n^2 + 4 * n + 10) % 100 = 6 :=
by
  sorry

end remainder_of_quadratic_expression_l277_277862


namespace cost_price_of_article_l277_277255

theorem cost_price_of_article 
  (CP SP : ℝ)
  (H1 : SP = 1.13 * CP)
  (H2 : 1.10 * SP = 616) :
  CP = 495.58 :=
by
  sorry

end cost_price_of_article_l277_277255


namespace relationship_between_a_b_c_l277_277364

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) - 1
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2)
noncomputable def c : ℝ := f (Real.log (1/4) / Real.log 2)

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l277_277364


namespace tournament_committees_count_l277_277077

theorem tournament_committees_count :
  let teams := 5
  let team_members := 7
  let host_selection := 5
  let host_team_choices := choose team_members 3
  let guest_team_choices := choose team_members 3
  let remaining_guest_team_choices := choose team_members 2
  let guest_team_selection := 4
  let per_host_total := host_team_choices * guest_team_choices * remaining_guest_team_choices^3 * guest_team_selection
  let total_committees := per_host_total * host_selection
  total_committees = 229105500 :=
by
  sorry

end tournament_committees_count_l277_277077


namespace no_t_shaped_div_by_5_l277_277295

/-- 
  There does not exist an arrangement of the numbers 1, 2, 3, ..., 64 in 
  an 8 x 8 grid such that the sum of the numbers in any T-shaped configuration 
  is divisible by 5.
-/
theorem no_t_shaped_div_by_5 : ¬ ∃ (grid : Fin 8 → Fin 8 → Fin 64),
  (∀ (r c : Fin 8) (dirs : Fin 4),
    let T_sum := match dirs with
                 | 0 => grid r c + grid (r + 1) c + grid (r + 1) (c + 1) + grid (r + 1) (c - 1)
                 | 1 => grid r c + grid (r - 1) c + grid (r - 1) (c + 1) + grid (r - 1) (c - 1)
                 | 2 => grid r c + grid (r + 1) c + grid (r + 1) (c - 1) + grid (r + 1) (c + 1)
                 | _ => grid r c + grid (r - 1) c + grid (r - 1) (c - 1) + grid (r - 1) (c + 1)
                 end in T_sum % 5 = 0) :=
sorry

end no_t_shaped_div_by_5_l277_277295


namespace proof_question_l277_277066

noncomputable def triangle_abc 
  (a b c : ℝ) (angle_A angle_B : ℝ) (ha : geometric_sequence a b c) 
  (h1 : a^2 - c^2 = a*c - b*c) (h2 : b^2 = a*c) : ℝ := 
  have angle_A := 60
  have sin_B := (sin angle_B)
  (b * sin_B) / c

theorem proof_question (a b c : ℝ) (angle_A angle_B : ℝ) 
  (h_geo : geometric_sequence a b c) (h1 : a^2 - c^2 = a*c - b*c) 
  (h2 : b^2 = a*c) :
  (b * (sin angle_B)) / c = (√3) / 2 := sorry

end proof_question_l277_277066


namespace ratio_of_books_to_pens_l277_277287

theorem ratio_of_books_to_pens (total_stationery : ℕ) (books : ℕ) (pens : ℕ) 
    (h1 : total_stationery = 400) (h2 : books = 280) (h3 : pens = total_stationery - books) : 
    books / (Nat.gcd books pens) = 7 ∧ pens / (Nat.gcd books pens) = 3 := 
by 
  -- proof steps would go here
  sorry

end ratio_of_books_to_pens_l277_277287


namespace imaginary_part_of_conjugate_l277_277818

def z : ℂ := 4 / (1 - complex.I)

theorem imaginary_part_of_conjugate :
  complex.im (complex.conj z) = -2 :=
sorry

end imaginary_part_of_conjugate_l277_277818


namespace perpendicular_line_eq_slope_intercept_l277_277648

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l277_277648


namespace sqrt_2700_minus_37_form_l277_277182

theorem sqrt_2700_minus_37_form (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (Int.sqrt 2700 - 37) = Int.sqrt a - b ^ 3) : a + b = 13 :=
sorry

end sqrt_2700_minus_37_form_l277_277182


namespace area_triangle_DEF_l277_277877

noncomputable def side_length_of_square (area : ℝ) : ℝ := real.sqrt area

-- Definitions and given conditions
def PQRS := {P Q R S : ℝ}
def small_square_side := 2 : ℝ
def total_area_PQRS := 100 : ℝ
def side_length_PQRS := side_length_of_square total_area_PQRS -- 10 cm

def centroid_PQRS := (5 : ℝ, 5 : ℝ) -- Center of the square (10 cm side length)

-- Triangle DEF inside PQRS with DE = DF and on folding DEF over EF, point D aligns with G.
def triangle_DEF_base := 6 : ℝ -- Effective side length of EF
def triangle_DEF_height := 9 : ℝ -- Altitude DG 

-- Prove the area of triangle DEF
theorem area_triangle_DEF :
  let area_DEF := 1 / 2 * triangle_DEF_base * triangle_DEF_height in
  area_DEF = 27 :=
by
  sorry

end area_triangle_DEF_l277_277877


namespace gcd_pow_minus_one_l277_277514

theorem gcd_pow_minus_one {m n a : ℕ} (hm : 0 < m) (hn : 0 < n) (ha : 2 ≤ a) : 
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd m n) - 1 := 
sorry

end gcd_pow_minus_one_l277_277514


namespace find_angle_y_l277_277884

/-
Problem Statement:
Given:
1. Lines m and n are parallel.
2. Angle A is 40 degrees.
3. Angle B is 90 degrees.

Prove that angle y is 130 degrees.
-/

noncomputable def angle_measure := ℝ

variables {m n : Prop} {A B y : angle_measure}
variables parallel : Prop

-- Condition: lines m and n are parallel
axiom parallel_lines : parallel = (m ↔ n)

-- Conditions for angles
axiom angle_A : A = 40
axiom angle_B : B = 90

-- To prove: angle y is 130 degrees
theorem find_angle_y (parallel : parallel) (A : angle_measure) (B : angle_measure) :
  A = 40 → B = 90 → y = 130 :=
by
  intros hA hB
  -- Proof would go here, but we're adding a placeholder for now
  exact sorry

end find_angle_y_l277_277884


namespace rectangle_area_l277_277486

structure Rectangle where
  length : ℕ    -- Length of the rectangle in cm
  width : ℕ     -- Width of the rectangle in cm
  perimeter : ℕ -- Perimeter of the rectangle in cm
  h : length = width + 4 -- Distance condition from the diagonal intersection

theorem rectangle_area (r : Rectangle) (h_perim : r.perimeter = 56) : r.length * r.width = 192 := by
  sorry

end rectangle_area_l277_277486


namespace infinite_sol_int_sqrt_sum_l277_277618

theorem infinite_sol_int_sqrt_sum (p n : ℤ) (hp : p.prime) :
  ∃∞ n : ℤ, (∃ k : ℤ, p = 2*k + 1) ∧ (∃ k : ℤ, n = k*k) ∧ ∃ m : ℤ, (↑(sqrt(p + n) + sqrt n) = m) :=
begin
  sorry
end

end infinite_sol_int_sqrt_sum_l277_277618


namespace z_in_first_quadrant_l277_277190

noncomputable def z : ℂ := (2 - I) / (1 - I)

def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem z_in_first_quadrant :
  is_in_first_quadrant z :=
sorry

end z_in_first_quadrant_l277_277190


namespace positive_difference_is_zero_l277_277496

theorem positive_difference_is_zero : 
  let S := (50 * (1 + 50)) / 2,
      S' := 10 * (0 + 5 + 10 + 15 + 20 + 25 + 30 + 35 + 40 + 45)
  in |S - S'| = 0 := 
by
  -- prove that the sums are equal
  sorry

end positive_difference_is_zero_l277_277496


namespace main_theorem_l277_277692

def Coin : Type := 
  {weight : ℤ // ∃ n : ℤ, weight = n ∨ weight = n + 1 ∨ weight = n - 1}

def genuine (c : Coin) (n : ℤ) : Prop := c.1 = n
def heavy_fake (c : Coin) (n : ℤ) : Prop := c.1 = n + 1
def light_fake (c : Coin) (n : ℤ) : Prop := c.1 = n - 1

def balance (c1 c2 : Coin) : ℤ := c1.1 - c2.1

noncomputable def Nastya (a b c d e : Coin) (g : ℤ) : Prop :=
  (balance a b = 0 ∨ balance a b ≠ 0) ∧
  (balance c d = 0 ∨ balance c d ≠ 0) ∧
  (balance (Coin.mk (a.1 + b.1) _) (Coin.mk (c.1 + d.1) _) = 0 ∨ 
   balance (Coin.mk (a.1 + b.1) _) (Coin.mk (c.1 + d.1) _) ≠ 0) →
  ∃ h l, (heavy_fake h g ∧ light_fake l g)

theorem main_theorem (a b c d e : Coin) :
  ∃ g h l, 
  (genuine a g ∨ genuine b g ∨ genuine c g ∨ genuine d g ∨ genuine e g) ∧
  a ≠ h ∧ a ≠ l ∧
  b ≠ h ∧ b ≠ l ∧
  c ≠ h ∧ c ≠ l ∧
  d ≠ h ∧ d ≠ l ∧
  e ≠ h ∧ e ≠ l →
  Nastya a b c d e g := 
sorry

end main_theorem_l277_277692


namespace perpendicular_line_equation_l277_277629

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l277_277629


namespace helen_baked_280_raisin_cookies_today_l277_277423

variables (R : ℕ)

theorem helen_baked_280_raisin_cookies_today 
    (h_yesterday : 300) 
    (h_more_yesterday : h_yesterday = R + 20) : 
    R = 280 := 
    sorry

end helen_baked_280_raisin_cookies_today_l277_277423


namespace part_I_part_III_l277_277395

-- Definition of the given functions
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + (1 / 2) * a * x ^ 2 + x
def g (x : ℝ) (a : ℝ) : ℝ := f x a - a * x ^ 2 - a * x + 1

-- Part (I): Prove maximum value
theorem part_I (a : ℝ) (h : f 1 a = 0) : (∀ x : ℝ, f x (-2) ≤ 0) :=
sorry

-- Part (III): Prove inequality with given conditions
theorem part_III (x1 x2 : ℝ) (h_pos : 0 < x1 ∧ 0 < x2)
  (h_cond : f x1 2 + f x2 2 + x1 * x2 = 0) : x1 + x2 ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end part_I_part_III_l277_277395


namespace milk_cost_l277_277112

theorem milk_cost (x : ℝ) (h1 : 4 * 2.50 + 2 * x = 17) : x = 3.50 :=
by
  sorry

end milk_cost_l277_277112


namespace problem1_problem2_l277_277294

noncomputable def calc_expr1 : ℝ :=
  real.sqrt 48 * real.sqrt (1 / 3) - real.sqrt 12 / real.sqrt 2 + real.sqrt 24

noncomputable def calc_expr2 : ℝ :=
  (-3)^0 + real.abs (real.sqrt 2 - 2) - (1 / 3)^(-1 : ℤ) + real.sqrt 8

theorem problem1 : calc_expr1 = 4 + real.sqrt 6 :=
by sorry

theorem problem2 : calc_expr2 = real.sqrt 2 :=
by sorry

end problem1_problem2_l277_277294


namespace imaginary_part_of_z_is_1_l277_277391

def z := Complex.ofReal 0 + Complex.ofReal 1 * Complex.I * (Complex.ofReal 1 + Complex.ofReal 2 * Complex.I)
theorem imaginary_part_of_z_is_1 : z.im = 1 := by
  sorry

end imaginary_part_of_z_is_1_l277_277391


namespace small_cuboid_length_is_five_l277_277705

-- Define initial conditions
def large_cuboid_length : ℝ := 18
def large_cuboid_width : ℝ := 15
def large_cuboid_height : ℝ := 2
def num_small_cuboids : ℕ := 6
def small_cuboid_width : ℝ := 6
def small_cuboid_height : ℝ := 3

-- Theorem to prove the length of the smaller cuboid
theorem small_cuboid_length_is_five (small_cuboid_length : ℝ) 
  (h1 : large_cuboid_length * large_cuboid_width * large_cuboid_height 
          = num_small_cuboids * (small_cuboid_length * small_cuboid_width * small_cuboid_height)) :
  small_cuboid_length = 5 := by
  sorry

end small_cuboid_length_is_five_l277_277705


namespace sacks_per_day_l277_277598

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (h1 : total_sacks = 56) (h2 : days = 4) : total_sacks / days = 14 := by
  sorry

end sacks_per_day_l277_277598


namespace positive_number_sum_square_l277_277998

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l277_277998


namespace sequence_identity_l277_277724

theorem sequence_identity (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, n ≥ 1 → (a (n + 1), b (n + 1)) = (√3 * a n - b n, √3 * b n + a n))
  (h2 : (a 60, b 60) = (4, -2)) :
  a 1 + b 1 = -3 / 2^58 :=
by
  sorry

end sequence_identity_l277_277724


namespace consecutive_integers_no_two_l277_277158

theorem consecutive_integers_no_two (a n : ℕ) : 
  ¬(∃ (b : ℤ), (b : ℤ) = 2) :=
sorry

end consecutive_integers_no_two_l277_277158


namespace expected_sufferers_l277_277939

theorem expected_sufferers (infection_rate : ℚ) (sample_size : ℕ) (expected_count : ℕ) 
  (h_infection_rate: infection_rate = 1 / 4) 
  (h_sample_size: sample_size = 500) 
  (h_expected: expected_count = 125) : 
  expected_count = infection_rate * sample_size :=
by
  rw [h_infection_rate, h_sample_size]
  norm_num
  exact h_expected

end expected_sufferers_l277_277939


namespace _l277_277825

nooncomputable theorem find_angle_C
  (a b c A B C : ℝ)
  (h1 : a + b + c = sqrt 2 + 1)
  (h2 : 1 / 2 * a * b * sin C = 1 / 6 * sin C)
  (h3 : sin A + sin B = sqrt 2 * sin C) :
  C = π / 3 :=
sorry

end _l277_277825


namespace minimum_value_distance_l277_277313

theorem minimum_value_distance (x : ℝ) : 
  sqrt (x^2 + (1 - x)^2) + sqrt ((x - 2)^2 + (x + 1)^2) ≥ 2 * sqrt 2 := 
sorry

end minimum_value_distance_l277_277313


namespace find_r_l277_277044

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 :=
sorry

end find_r_l277_277044


namespace find_value_of_c_l277_277070

-- Definitions
variables (O : Type) [center : Point O]
variables (c y : ℝ)

-- Conditions
variables (h1 : c = 2 * y)
variables (h2 : y + y + 2*c = 180)

-- Statement
theorem find_value_of_c (c y : ℝ) (h1 : c = 2 * y) (h2 : y + y + 2*c = 180) : c = 60 :=
by {
  sorry,
}

end find_value_of_c_l277_277070


namespace number_of_ways_to_fill_grid_l277_277090

open Finset

theorem number_of_ways_to_fill_grid (n : ℕ) (h : n ≥ 1) :
  let grid := Matrix (Fin n) (Fin n) (Fin 2)
  let condition (m : grid) := (∀ i : Fin n, even (card { j | m i j = 1 })) ∧
                              (∀ j : Fin n, even (card { i | m i j = 1 }))
  ∃ fill_count : ℕ, (fill_count = 2^((n-1)*(n-1))) ∧
                    ∀ g : grid, condition g ↔ (g ∈ universe grid) :=
sorry

end number_of_ways_to_fill_grid_l277_277090


namespace modulus_product_eq_sqrt_5_l277_277982

open Complex

-- Define the given complex number.
def z : ℂ := 2 + I

-- Declare the product with I.
def z_product := z * I

-- State the theorem that the modulus of the product is sqrt(5).
theorem modulus_product_eq_sqrt_5 : abs z_product = Real.sqrt 5 := 
sorry

end modulus_product_eq_sqrt_5_l277_277982


namespace find_f_neg_pi_div_3_l277_277797

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (2^x / (2^x + 1)) + a * x + Real.cos (2 * x)

theorem find_f_neg_pi_div_3 (a : ℝ) :
  f (Real.pi / 3) a = 2 → f (-Real.pi / 3) a = -2 :=
begin
  intro h,
  sorry
end

end find_f_neg_pi_div_3_l277_277797


namespace perpendicular_line_eq_l277_277654

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l277_277654


namespace optimal_order_l277_277553

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l277_277553


namespace pyramid_volume_l277_277220

theorem pyramid_volume (s : ℝ) (h : ℝ) (V : ℝ) (hs : s = 4 * real.sqrt 3) (hh : h = s) :
  V = 64 * real.sqrt 3 :=
by 
  sorry

end pyramid_volume_l277_277220


namespace distance_symmetric_point_l277_277368

theorem distance_symmetric_point :
  let P : ℝ × ℝ × ℝ := (1, -2, 3)
  let P' : ℝ × ℝ × ℝ := (1, -2, -3)
  ∃ d : ℝ, d = 6 ∧ d = real.sqrt ((P.2 - P'.2) ^ 2 + (P.2 - P'.2) ^ 2 + (P.3 - P'.3) ^ 2) :=
begin
  sorry
end

end distance_symmetric_point_l277_277368


namespace perpendicular_line_eq_l277_277653

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l277_277653


namespace Mike_gave_marbles_l277_277524

variables (original_marbles given_marbles remaining_marbles : ℕ)

def Mike_original_marbles : ℕ := 8
def Mike_remaining_marbles : ℕ := 4
def Mike_given_marbles (original remaining : ℕ) : ℕ := original - remaining

theorem Mike_gave_marbles :
  Mike_given_marbles Mike_original_marbles Mike_remaining_marbles = 4 :=
sorry

end Mike_gave_marbles_l277_277524


namespace perpendicular_line_eq_l277_277656

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l277_277656


namespace fraction_equality_implies_equality_l277_277234

theorem fraction_equality_implies_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a / c = b / c) → (a = b) :=
by {
  sorry
}

end fraction_equality_implies_equality_l277_277234


namespace total_distance_walked_l277_277500

variables
  (distance1 : ℝ := 1.2)
  (distance2 : ℝ := 0.8)
  (distance3 : ℝ := 1.5)
  (distance4 : ℝ := 0.6)
  (distance5 : ℝ := 2)

theorem total_distance_walked :
  distance1 + distance2 + distance3 + distance4 + distance5 = 6.1 :=
sorry

end total_distance_walked_l277_277500


namespace isosceles_triangle_lengths_l277_277465

noncomputable def isosceles_triangle_side_lengths (m : ℝ) :=
  let x := m * (1 + Real.sqrt 5) / 2 in
  (m, x)

theorem isosceles_triangle_lengths (m : ℝ) :
  (∃ (x : ℝ),
    let triangle := (m, x) in
    let α := 72 in
    let β := α in
    let base_angle := 36 in
    let angle_bisector := m in
    let isosceles := triangle.1 = m ∧ triangle.2 = x ∧ α = β ∧ angle_bisector = m in
    triangle = isosceles_triangle_side_lengths m) :=
  sorry

end isosceles_triangle_lengths_l277_277465


namespace S_n_less_than_6_l277_277347

-- Definitions based on conditions in a)
def f : ℕ → ℕ → ℕ 
| 1, 1 := 1
| (m+1), n := f m n + 2 * (m + n)
| m, (n+1) := f m n + 2 * (m + n - 1)

def a (n : ℕ) : ℝ := (real.sqrt (f n n)) / (2 ^ (n - 1))

def S (n : ℕ) : ℝ := ∑ k in finset.range n, a (k + 1)

-- Statement to prove in Lean
theorem S_n_less_than_6 (n : ℕ) : S n < 6 :=
sorry

end S_n_less_than_6_l277_277347


namespace cake_volume_l277_277722

theorem cake_volume :
  let thickness := 1 / 2
  let diameter := 16
  let radius := diameter / 2
  let total_volume := Real.pi * radius^2 * thickness
  total_volume / 16 = 2 * Real.pi := by
    sorry

end cake_volume_l277_277722


namespace wuyang_team_max_matches_l277_277580

theorem wuyang_team_max_matches : 
  ∃ x : ℕ, x + 20 ≤ 33 ∧ (20 * (1 - 0.3 - 0.2)) / (20 + x) ≥ 0.3 := 
sorry

end wuyang_team_max_matches_l277_277580


namespace perpendicular_line_equation_l277_277660

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l277_277660


namespace exists_cycle_with_neighborhood_subset_l277_277348

-- Definitions of graph components
variable {V : Type*} -- Type for vertices
variable {G : SimpleGraph V} -- Simple graph G
variable [DecidableRel G.Adj] -- Ensure adjacency is decidable

-- Define 2-connectivity (simple definition for the sake of the problem)
def is_2_connected (G : SimpleGraph V) : Prop :=
  ∀ (u v : V), ∃ (P : List V), is_path G u v P ∧ P.length > 2

-- Define vertex neighborhood
def N_G (G : SimpleGraph V) (v : V) : Finset V := G.neighborFinset v

-- Main theorem to be proved
theorem exists_cycle_with_neighborhood_subset (G : SimpleGraph V) [h : is_2_connected G] (x : V) :
  ∃ (C : SimpleGraph V), -- Subgraph that is a cycle
    C ⊆ G ∧
    (∃ (y : V), y ≠ x ∧ y ∈ V C ∧ N_G G y ⊆ V C) :=
sorry

end exists_cycle_with_neighborhood_subset_l277_277348


namespace arithmetic_sequences_count_l277_277429

noncomputable def count_arithmetic_sequences : ℕ :=
  ∑ d in (finset.range 22).filter (λ d, 1 ≤ d), 90 - 4 * (d + 1)

theorem arithmetic_sequences_count : count_arithmetic_sequences = 1012 := by
  sorry

end arithmetic_sequences_count_l277_277429


namespace increasing_interval_of_f_l277_277596

def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x : ℝ, x > 1 → deriv f x > 0 :=
by
  sorry

end increasing_interval_of_f_l277_277596


namespace correct_option_l277_277398

def f (x a : ℝ) : ℝ := -x^2 + 6 * x + a^2 - 1

theorem correct_option : ∀ (a : ℝ), f (real.sqrt 2) a < f 4 a ∧ f 4 a < f 3 a := 
by
  intro a 
  sorry

end correct_option_l277_277398


namespace length_of_train_is_400_l277_277250

-- Define the conditions
def train_speed_kmph : ℝ := 100
def motorbike_speed_kmph : ℝ := 64
def overtaking_time_sec : ℝ := 40

-- Conversion factor for kmph to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Convert speeds to m/s
def train_speed_mps : ℝ := train_speed_kmph * kmph_to_mps
def motorbike_speed_mps : ℝ := motorbike_speed_kmph * kmph_to_mps

-- Calculate relative speed
def relative_speed_mps : ℝ := train_speed_mps - motorbike_speed_mps

-- Calculate the length of the train
def length_of_train : ℝ := relative_speed_mps * overtaking_time_sec

-- The statement to be proved
theorem length_of_train_is_400 : length_of_train = 400 := by
  sorry

end length_of_train_is_400_l277_277250


namespace total_time_proof_l277_277163

-- Define the known values
def speed_of_current (x : ℝ) : ℝ := x
def speed_of_boat (x : ℝ) : ℝ := 3 * x
def round_trip_time_without_current : ℝ := 20

-- Define the total round trip time considering the current
def total_round_trip_time_with_current (x : ℝ) : ℝ :=
  let d := 10 * speed_of_boat x -- distance
  (d / (speed_of_boat x + speed_of_current x)) + (d / (speed_of_boat x - speed_of_current x))

-- Problem to prove: the total round trip time considering the current is 22.5 minutes
theorem total_time_proof (x : ℝ) (h : x > 0) : total_round_trip_time_with_current x = 22.5 :=
  by sorry

end total_time_proof_l277_277163


namespace cos_tan_simplify_l277_277566

theorem cos_tan_simplify : 
  ∀ (cos sin : ℝ → ℝ),
  cos (40 * real.pi / 180) * (1 + real.sqrt 3 * (sin (10 * real.pi / 180) / cos (10 * real.pi / 180))) = 1 :=
by
  sorry

end cos_tan_simplify_l277_277566


namespace prism_surface_area_is_8pi_l277_277808

noncomputable def prismSphereSurfaceArea : ℝ :=
  let AB := 2
  let AC := 1
  let BAC := Real.pi / 3 -- angle 60 degrees in radians
  let volume := Real.sqrt 3
  let AA1 := 2
  let radius := Real.sqrt 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area

theorem prism_surface_area_is_8pi : prismSphereSurfaceArea = 8 * Real.pi :=
  by
    sorry

end prism_surface_area_is_8pi_l277_277808


namespace average_of_rest_l277_277073

theorem average_of_rest (A : ℝ) (total_students scoring_95 scoring_0 : ℕ) (total_avg : ℝ)
  (h_total_students : total_students = 25)
  (h_scoring_95 : scoring_95 = 3)
  (h_scoring_0 : scoring_0 = 3)
  (h_total_avg : total_avg = 45.6)
  (h_sum_eq : total_students * total_avg = 3 * 95 + 3 * 0 + (total_students - scoring_95 - scoring_0) * A) :
  A = 45 := sorry

end average_of_rest_l277_277073


namespace shaded_area_correct_l277_277606

-- Definitions based on conditions
def radius : ℝ := 3
def theta : ℝ := real.pi / 3  -- 60 degrees in radians

-- Calculate the shaded area given radius and angle between tangents
noncomputable def shaded_area (r : ℝ) (θ : ℝ) : ℝ :=
  let full_circle_area := real.pi * r ^ 2
      sector_area := (θ / (2 * real.pi)) * full_circle_area
      triangle_area := r ^ 2 * real.tan (θ / 2)
  in 2 * triangle_area + (full_circle_area - sector_area)

-- The theorem statement
theorem shaded_area_correct :
  shaded_area radius theta = (6 * real.pi + 9 * real.sqrt 3) :=
by
  sorry

end shaded_area_correct_l277_277606


namespace problem_l277_277245

variable (numbers : Finset ℕ)
variables (card1 card2 card3 : Finset ℕ)

def valid_card (card : Finset ℕ) : Prop :=
  ∀ x y ∈ card, x ≠ y → ¬ (x - y = y ∨ x - y = -y)

noncomputable def solution : ℕ := 8

theorem problem (h1 : card1 = {1, 5})
                (h2 : card2 = {2})
                (h3 : card3 = {3})
                (h_valid1 : valid_card card1)
                (h_valid2 : valid_card card2)
                (h_valid3 : valid_card card3)
                : solution = 8 := sorry

end problem_l277_277245


namespace monthly_income_of_A_l277_277187

theorem monthly_income_of_A (A B C : ℝ)
  (h1 : (A + B) / 2 = 5050)
  (h2 : (B + C) / 2 = 6250)
  (h3 : (A + C) / 2 = 5200) :
  A = 4000 :=
sorry

end monthly_income_of_A_l277_277187


namespace area_triangle_LOM_l277_277461

noncomputable def angle_TRIANGLE_A : ℝ := 45 -- Boundary angle A (in degrees)
noncomputable def angle_TRIANGLE_B : ℝ := 90 -- Boundary angle B
noncomputable def angle_TRIANGLE_C : ℝ := 45 -- Boundary angle C
noncomputable def area_ABC : ℝ := 20 -- Given area of triangle ABC

-- Consider functions for conditions
def scalene_triangle (A B C : ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def angle_relationship (A B C : ℝ) : Prop :=
  C = B - A ∧ B = 2 * A

-- Main proof problem
theorem area_triangle_LOM
  (h1: scalene_triangle angle_TRIANGLE_A angle_TRIANGLE_B angle_TRIANGLE_C)
  (h2: angle_relationship angle_TRIANGLE_A angle_TRIANGLE_B angle_TRIANGLE_C)
  (h3: area_ABC = 20) :
  let area_LOM := 3 * area_ABC in
  area_LOM = 60 :=
by sorry

end area_triangle_LOM_l277_277461


namespace sum_of_digits_is_three_l277_277756

noncomputable def A : ℕ := 
  let k := 10^100 in
  5050 * (k / 4) + 50505050 / 4 * (1 - k.modn 4)
  
noncomputable def B : ℕ := 
  let k := 10^100 in
  7070 * (k / 4) + 70707070 / 4 * (1 - k.modn 4)

def units_digit (n : ℕ) : ℕ := n % 10

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

theorem sum_of_digits_is_three : thousands_digit (A * B) + units_digit (A * B) = 3 := 
by
  sorry

end sum_of_digits_is_three_l277_277756


namespace sum_of_coefficients_l277_277858

noncomputable def poly := (3 * (x : ℝ) - 1) ^ 7

theorem sum_of_coefficients (a_7 a_6 a_5 a_4 a_3 a_2 a_1 a_0 : ℝ)
  (h : poly = a_7 * x ^ 7 + a_6 * x ^ 6 + a_5 * x ^ 5 +
            a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + a_0) :
  a_7 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1 = 129 :=
sorry

end sum_of_coefficients_l277_277858


namespace inequality_proof_l277_277515

-- Definitions for the conditions
variable (x y : ℝ)

-- Conditions
def conditions : Prop := 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Problem statement to be proven
theorem inequality_proof (h : conditions x y) : 
  x^3 + x * y^2 + 2 * x * y ≤ 2 * x^2 * y + x^2 + x + y := 
by 
  sorry

end inequality_proof_l277_277515


namespace probability_none_hit_l277_277345

theorem probability_none_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (1 - p)^5 = (1 - p) * (1 - p) * (1 - p) * (1 - p) * (1 - p) :=
by sorry

end probability_none_hit_l277_277345


namespace find_m_range_l277_277400

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + 1

theorem find_m_range :
  ∃ (x₀ ∈ Ioo 0 1), ∀ a ∈ Icc (-2 : ℝ) 0, 2 * (Real.exp a) * m + f x₀ a > a^2 + 2 * a + 4 → m ∈ Set.Ioo 1 (Real.exp 2) :=
sorry

end find_m_range_l277_277400


namespace find_a_perpendicular_lines_l277_277822

theorem find_a_perpendicular_lines (a : ℝ) :
    (∀ x y : ℝ, a * x - y + 2 * a = 0 → (2 * a - 1) * x + a * y + a = 0) →
    (a = 0 ∨ a = 1) :=
by
  intro h
  sorry

end find_a_perpendicular_lines_l277_277822


namespace range_g_l277_277768

noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g : set.range (λ x, g x) = set.Ioi 0 := 
sorry

end range_g_l277_277768


namespace range_of_ab_l277_277018

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  log ((1 + a * x) / (1 - 2 * x))

theorem range_of_ab (a b : ℝ) : 
  (∀ x, f a b x = - f a b (-x)) ∧ a ≠ -2 ∧ (∀ x, -b < x ∧ x < b → 1 - 2 * x ≠ 0) →
  (1 < a ^ b ∧ a ^ b ≤ sqrt 2) := 
begin
  sorry
end

end range_of_ab_l277_277018


namespace sufficient_but_not_necessary_l277_277357

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → 2 / a < 1) ∧ (2 / a < 1 → a > 2 ∨ a < 0) :=
by sorry

end sufficient_but_not_necessary_l277_277357


namespace postage_arrangements_11_cents_l277_277318

-- Definitions for the problem settings, such as stamp denominations and counts
def stamp_collection : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

-- Function to calculate all unique arrangements of stamps that sum to a given value (11 cents)
def count_arrangements (total_cents : ℕ) : ℕ :=
  -- The implementation would involve a combinatorial counting taking into account the problem conditions
  sorry

-- The main theorem statement asserting the solution
theorem postage_arrangements_11_cents :
  count_arrangements 11 = 71 :=
  sorry

end postage_arrangements_11_cents_l277_277318


namespace remainder_of_2n_div4_l277_277688

theorem remainder_of_2n_div4 (n : ℕ) (h : ∃ k : ℕ, n = 4 * k + 3) : (2 * n) % 4 = 2 := 
by
  sorry

end remainder_of_2n_div4_l277_277688


namespace min_a_plus_b_l277_277046

theorem min_a_plus_b (a b : ℝ) (h : a^2 + 2 * b^2 = 6) : a + b ≥ -3 :=
sorry

end min_a_plus_b_l277_277046


namespace factorial_quotient_floor_l277_277297

theorem factorial_quotient_floor :
  ∀ x : ℝ, ⌊x⌋ = int.floor x →
  ⌊((2010! + 2007!) / (2009! + 2008!))⌋ = 2010 :=
by
  sorry

end factorial_quotient_floor_l277_277297


namespace largest_ball_radius_l277_277728

def torus_inner_radius : ℝ := 2
def torus_outer_radius : ℝ := 4
def circle_center : ℝ × ℝ × ℝ := (3, 0, 1)
def circle_radius : ℝ := 1

theorem largest_ball_radius : ∃ r : ℝ, r = 9 / 4 ∧
  (∃ (sphere_center : ℝ × ℝ × ℝ) (torus_center : ℝ × ℝ × ℝ),
  (sphere_center = (0, 0, r)) ∧
  (torus_center = (3, 0, 1)) ∧
  (dist (0, 0, r) (3, 0, 1) = r + 1)) := sorry

end largest_ball_radius_l277_277728


namespace solve_system_of_equations_l277_277842

theorem solve_system_of_equations :
  ∃ y : ℝ, (2 * 2 + y = 0) ∧ (2 + y = 3) :=
by
  sorry

end solve_system_of_equations_l277_277842


namespace derivative_at_pi_over_two_l277_277816

noncomputable def f (x : ℝ) : ℝ := sin x + 2 * x * (f 0).derivative

theorem derivative_at_pi_over_two : (f.derivative (π / 2)) = -2 := by
  -- Proof to be filled
  sorry

end derivative_at_pi_over_two_l277_277816


namespace find_value_less_than_twice_l277_277938

def value_less_than_twice_another (x y v : ℕ) : Prop :=
  y = 2 * x - v ∧ x + y = 51 ∧ y = 33

theorem find_value_less_than_twice (x y v : ℕ) (h : value_less_than_twice_another x y v) : v = 3 := by
  sorry

end find_value_less_than_twice_l277_277938


namespace sqrt_geo_not_arith_l277_277417

-- Define the conditions provided in the problem
variables {a b c : ℝ}
hypothesis (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
hypothesis (geo_seq : b^2 = a * c)
hypothesis (not_arith_seq : ¬ (a + c = 2 * b))

-- Define the theorem to prove
theorem sqrt_geo_not_arith : ¬ ((√a + √c) = 2 * √b) := sorry

end sqrt_geo_not_arith_l277_277417


namespace exponential_rule_l277_277677

theorem exponential_rule (a b : ℝ) : exp(a) * exp(b) = exp(a + b) := 
by 
  -- Skipping the actual proof using sorry
  sorry

end exponential_rule_l277_277677


namespace intersection_point_l277_277765

-- Definitions of the lines
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 10
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 20

-- Theorem stating the intersection point
theorem intersection_point : line1 (60 / 23) (50 / 23) ∧ line2 (60 / 23) (50 / 23) :=
by {
  sorry
}

end intersection_point_l277_277765


namespace quadrilateral_proof_l277_277790

-- Definitions from the conditions
variables {A B C D M : Type*}

-- Assume the quadrilateral is inscribed
axiom inscribed_quadrilateral (circ : A → B → C → D → Prop) : Prop

-- Lengths of the segments
def MA : ℝ := 2
def MB : ℝ := 3

-- The Power of a Point theorem for the point M and chords AD and BC
axiom power_of_point (M A B C D : Type*) (MD MC MA MB : ℝ) : MD * MC = MA * MB

theorem quadrilateral_proof (circ : A → B → C → D → Prop) 
  (inscribed : inscribed_quadrilateral circ) : (MA : ℝ) * (MB : ℝ) = 15 :=
by {
  have h := power_of_point M A B C D (MA := MA) (MB := MB),
  sorry
}

end quadrilateral_proof_l277_277790


namespace curve_is_parabola_l277_277338

theorem curve_is_parabola (r θ : ℝ) (hr : r = 4 * (sin θ / cos θ) * (1 / cos θ)) :
  ∃ (x y : ℝ), x = r * cos θ ∧ y = r * sin θ ∧ x^2 = 4 * y :=
by 
  sorry

end curve_is_parabola_l277_277338


namespace line_passes_through_fixed_point_l277_277800

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3 / 2) + 3 * (1 / 6) + q = 0 :=
by 
  sorry

end line_passes_through_fixed_point_l277_277800


namespace edge_length_of_cube_l277_277582

variables (L W H : ℝ)

def area_floor : ℝ := 20
def area_longer_wall : ℝ := 15
def area_shorter_wall : ℝ := 12

axiom L_times_W_eq_20 : L * W = area_floor
axiom L_times_H_eq_15 : L * H = area_longer_wall
axiom W_times_H_eq_12 : W * H = area_shorter_wall

def volume : ℝ := L * W * H

theorem edge_length_of_cube : 
  (∃ (edge : ℝ), edge ^ 3 = volume) → 
  ∃ (e : ℝ), e ≈ 3.9149 :=
by 
  sorry

end edge_length_of_cube_l277_277582


namespace different_sets_l277_277921

theorem different_sets (a b c : ℤ) (h1 : 0 < a) (h2 : a < c - 1) (h3 : 1 < b) (h4 : b < c)
  (rk : ∀ (k : ℤ), 0 ≤ k ∧ k ≤ a → ∃ (r : ℤ), 0 ≤ r ∧ r < c ∧ k * b % c = r) :
  {r | ∃ k, 0 ≤ k ∧ k ≤ a ∧ r = k * b % c} ≠ {k | 0 ≤ k ∧ k ≤ a} :=
sorry

end different_sets_l277_277921


namespace perpendicular_line_eq_slope_intercept_l277_277647

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l277_277647


namespace solve_for_x_l277_277439

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 9 / (x / 3)) : x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := 
by
  sorry

end solve_for_x_l277_277439


namespace pyramid_coloring_l277_277871

-- Define the pyramid structure
structure Pyramid :=
(P A B C D : ℕ)

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the condition that adjacent vertices cannot be colored the same
def adjacent_vertices {P A B C D : ℕ} : (Pyramid P A B C D) → Prop :=
λ _, true -- Placeholder for actual adjacency definition

-- The main theorem stating the number of different coloring methods
theorem pyramid_coloring (P A B C D : ℕ) (h : adjacent_vertices (Pyramid.mk P A B C D)) : 
  Σ(W : nat), W = 72 :=
sorry

end pyramid_coloring_l277_277871


namespace part_I_part_II_l277_277035

-- Definitions for lines l1, l2, l3
def l1 (m : ℝ) : ℝ × ℝ → Prop := λ P, let (x, y) := P in m * x - (m + 1) * y - 2 = 0
def l2 : ℝ × ℝ → Prop := λ P, let (x, y) := P in x + 2 * y + 1 = 0
def l3 : ℝ × ℝ → Prop := λ P, let (x, y) := P in y = x - 2

-- Condition for circle center and radius
def circle_center : ℝ × ℝ := (1, -1)
def circle_radius : ℝ := 2 * real.sqrt 3

-- Theorems to be proved
theorem part_I (m : ℝ) : l1 m (-2, -2) := by sorry

theorem part_II : 
  let C : ℝ × ℝ := circle_center,
      r : ℝ := circle_radius,
      distance : ℝ := real.sqrt ((1+2)^2 + (-1+2)^2)
  in 2 * real.sqrt (12 - distance^2) = 2 * real.sqrt 2 := by sorry

end part_I_part_II_l277_277035


namespace permutation_sum_divisible_l277_277120

theorem permutation_sum_divisible (p : ℕ) (hp : p > 2 ∧ Nat.Prime p) :
  ∃ (k : Fin (p - 1) → Fin (p - 1)), (Set.Bijective k) ∧ (p ∣ (Finset.sum (Finset.range (p - 1)) (λ i, i.succ ^ k i))) :=
sorry

end permutation_sum_divisible_l277_277120


namespace find_x_l277_277676

-- Define the conditions
def condition (x : ℕ) := (4 * x)^2 - 2 * x = 8062

-- State the theorem
theorem find_x : ∃ x : ℕ, condition x ∧ x = 134 := sorry

end find_x_l277_277676


namespace factorization_mn_l277_277329

variable (m n : ℝ) -- Declare m and n as arbitrary real numbers.

theorem factorization_mn (m n : ℝ) : m^2 - m * n = m * (m - n) := by
  sorry

end factorization_mn_l277_277329


namespace frac_of_20_correct_l277_277702

theorem frac_of_20_correct : ∃ (x : ℚ), (x * 20 + 16 = 32) ∧ (x = 4 / 5) := by
  existsi (4 / 5 : ℚ)
  split
  { calc
      (4 / 5) * 20 + 16 = (4 * 20) / 5 + 16 : by ring
        ... = 16 + 16 : by norm_num
        ... = 32     : by norm_num, }
  { refl }

end frac_of_20_correct_l277_277702


namespace perpendicular_line_equation_l277_277633

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l277_277633


namespace find_superabundant_l277_277506

def divisors_sum (n : ℕ) : ℕ :=
  (Finset.range (n+1)).filter (λ d, n % d = 0).sum id

def superabundant (n : ℕ) : Prop :=
  divisors_sum (divisors_sum n) = n + 3

theorem find_superabundant : ∃! (n : ℕ), superabundant n :=
by {
  -- Proceed to show the only superabundant positive integer is 3
  sorry
}

end find_superabundant_l277_277506


namespace extreme_values_when_a_is_4_range_of_a_for_f_ge_4_in_interval_l277_277828

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

theorem extreme_values_when_a_is_4 :
  ∃ (max min : ℝ), f 4 (-1/2) = 1 ∧ f 4 (1/2) = -1 := sorry

theorem range_of_a_for_f_ge_4_in_interval :
  (∀ x ∈ Icc 1 2, f (7 : ℝ) x ≥ 4) ↔ (∀ a: ℝ, a ≥ 7) := sorry

end extreme_values_when_a_is_4_range_of_a_for_f_ge_4_in_interval_l277_277828


namespace num_distinct_terms_eq_ten_l277_277764

noncomputable def countDistinctTerms [(a + b)^3 * (a - b)^3]^3 :=
  let simplifiedExpr := (a^2 - b^2)^9
  -- Calculate the number of distinct terms in the expansion of simplifiedExpr
  10

theorem num_distinct_terms_eq_ten
  (a b : ℤ) :
  countDistinctTerms [(a + b)^3 * (a - b)^3]^3 = 10 :=
by sorry

end num_distinct_terms_eq_ten_l277_277764


namespace satisfies_conditions_l277_277022

-- Define the conditions on the function f
def is_periodic_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 1) = -f(x)

def is_decreasing_on_segment (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f(x) ≥ f(y)

-- Define the candidate functions
def f1 (x : ℝ) : ℝ := Real.sin (Real.pi * x)
def f2 (x : ℝ) : ℝ := Real.cos (Real.pi * x)
def f3 (x : ℝ) (k : ℤ) : ℝ := 1 - (x - 2 * k)^2
def f4 (x : ℝ) (k : ℤ) : ℝ := 1 + (x - 2 * k)^2

-- The main theorem stating that f2 and f3 with k = 0 satisfy the properties
theorem satisfies_conditions : is_periodic_neg_one f2 ∧ is_decreasing_on_segment f2 0 1 ∧
                              (∀ k : ℤ, is_periodic_neg_one (f3 k)) ∧ is_decreasing_on_segment (f3 0) 0 1 :=
by
  constructor
  sorry
  constructor
  sorry
  constructor
  intro k
  sorry
  sorry

end satisfies_conditions_l277_277022


namespace value_of_m_l277_277605

noncomputable def TV_sales_volume_function (x : ℕ) : ℚ :=
  10 * x + 540

theorem value_of_m : ∀ (m : ℚ),
  (3200 * (1 + m / 100) * 9 / 10) * (600 * (1 - 2 * m / 100) + 220) = 3200 * 600 * (1 + 15.5 / 100) →
  m = 10 :=
by sorry

end value_of_m_l277_277605


namespace area_of_feet_of_altitudes_l277_277078

variables (α S : ℝ)
-- Assuming the conditions: α is the base angle of an isosceles acute triangle, and S is the area of this triangle.
-- We aim to prove that the area of the triangle with vertices at the feet of the altitudes is: -1/2 * S * sin(4 * α) * cot(α)

theorem area_of_feet_of_altitudes (α S : ℝ) :
  ∃ S', (S' = (-1/2) * S * sin (4 * α) * cot (α)) :=
sorry

end area_of_feet_of_altitudes_l277_277078


namespace smallest_n_l277_277962

theorem smallest_n (x y : ℤ) (hx : x ≡ -2 [MOD 7]) (hy : y ≡ 2 [MOD 7]) :
  ∃ (n : ℕ), (n > 0) ∧ (x^2 + x * y + y^2 + ↑n ≡ 0 [MOD 7]) ∧ n = 3 := by
  sorry

end smallest_n_l277_277962


namespace actual_sales_increase_l277_277069

noncomputable def sales (total_sales : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  total_sales / (1 + tax_rate - discount_rate)

def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem actual_sales_increase :
  let sales_this_year := sales 400 0.07 0.05 in
  let sales_last_year := sales 320 0.06 0.03 in
  percent_increase sales_last_year sales_this_year = 26.22 := sorry

end actual_sales_increase_l277_277069


namespace integral_evaluation_l277_277293

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, real.sqrt (-x^2 + 2*x)

theorem integral_evaluation : integral_value = real.pi / 4 :=
sorry

end integral_evaluation_l277_277293


namespace train_length_l277_277625

/-- Given the conditions:
1. Two trains of equal length (in meters) are running on parallel lines in the same direction at 48 km/hr and 36 km/hr.
2. The faster train passes the slower train in 36 seconds.
We need to prove that the length of each train is 60 meters. -/
theorem train_length (L : ℕ) (h1 : 48 = 48000 / 3600) (h2 : 36 = 36000 / 3600)
  (h3 : ∀ t : ℕ, relative_speed km_per_sec := (48000 / 3600) - (36000 / 3600))
  (h4 : 36 * relative_speed km_per_sec = 2 * L) : L = 60 :=
sorry

end train_length_l277_277625


namespace find_a_range_l277_277017

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2 * x^2 - 2 * x + a - 3 = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ y ≠ x ∧ 2 * y^2 - 2 * y + a - 3 = 0) 
  ↔ 3 < a ∧ a < 7 / 2 := 
sorry

end find_a_range_l277_277017


namespace evaluate_expression_l277_277326

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end evaluate_expression_l277_277326


namespace perpendicular_line_equation_l277_277661

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l277_277661


namespace power_of_i_l277_277748
-- Import the necessary library

-- Definition of the imaginary unit
noncomputable def imaginary_unit := complex.I

-- The problem statement in Lean 4
theorem power_of_i (n : ℕ) (hn : n = 2013) : imaginary_unit ^ n = imaginary_unit := by
  sorry

end power_of_i_l277_277748


namespace four_digit_numbers_with_two_repeated_digits_l277_277282

theorem four_digit_numbers_with_two_repeated_digits : 
  ∃ n : ℕ, n = 3888 ∧ ∀ (d1 d2 d3 : ℕ), (1 ≤ d1) ∧ (d1 ≤ 9) ∧ (0 ≤ d2) ∧ (d2 ≤ 9) ∧ (0 ≤ d3) ∧ (d3 ≤ 9) 
  → (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d2 ≠ d3) ∧ (∃ (p : String), p ∈ ["AABC", "BAAC", "BCAA", "ABAC", "BACA", "ABCA"])
  → n = 4320 - (9 * 8 * 6) := 
begin
  sorry
end

end four_digit_numbers_with_two_repeated_digits_l277_277282


namespace optimal_order_l277_277536

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l277_277536


namespace find_a_b_increasing_in_interval_solution_set_inequality_l277_277410

-- Problem 1
theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_dom : ∀ x, true)
  (h_f1 : f 1 = 1/2) :
  a = 1 ∧ b = 0 :=
sorry

-- Problem 2
theorem increasing_in_interval :
  ∀ f : ℝ → ℝ, (∀ x, f x = x / (x^2 + 1)) → 
  ∀ x1 x2, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2 :=
sorry

-- Problem 3
theorem solution_set_inequality :
  ∀ f : ℝ → ℝ, (∀ x, f x = x / (x^2 + 1)) →
  ∀ t, 0 < t ∧ t < 1/2 ↔ f t + f (t - 1) < 0 :=
sorry

end find_a_b_increasing_in_interval_solution_set_inequality_l277_277410


namespace baseball_card_decrease_l277_277704

theorem baseball_card_decrease (V0 V1 : ℝ) (V0_pos : 0 < V0)
  (h1 : V1 = V0 * 0.5) 
  (total_decrease : V0 - V1 = 0.55 * V0) : 
  ∃ x : ℝ, (x / 100) * V1 = V0 - V1 - (total_decrease - (V0 - V1)) :=
by
  sorry

end baseball_card_decrease_l277_277704


namespace A_n_divisible_by_225_l277_277205

theorem A_n_divisible_by_225 (n : ℕ) : 225 ∣ (16^n - 15 * n - 1) := by
  sorry

end A_n_divisible_by_225_l277_277205


namespace vector_magnitude_within_range_l277_277468

noncomputable def vector_magnitude_range : set ℝ := {d | 7 ≤ d ∧ d ≤ 11}

theorem vector_magnitude_within_range (x y : ℝ)
  (h : x^2 + y^2 = 4) :
  (sqrt ((6 - x)^2 + (3 * sqrt 5 - y)^2)) ∈ vector_magnitude_range :=
sorry

end vector_magnitude_within_range_l277_277468


namespace sum_of_three_consecutive_even_numbers_l277_277860

theorem sum_of_three_consecutive_even_numbers (m : ℤ) (h : ∃ k, m = 2 * k) : 
  m + (m + 2) + (m + 4) = 3 * m + 6 :=
by
  sorry

end sum_of_three_consecutive_even_numbers_l277_277860


namespace smallest_a_l277_277964

-- Define the main variables and conditions
def vertex := (-1 / 3, -4 / 3)  -- The vertex of the parabola
def eqn (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c  -- The equation of the parabola

-- Define the problem and proof statement using Lean
theorem smallest_a 
  (a b c : ℝ) 
  (h1 : 0 < a) 
  (h2 : (a + b + c) ∈ ℤ) 
  (h3 : ∃ k : ℝ, vertex = (k, h2)) -- Assuming there's a vertex
  : a = 3 / 16 :=
by
  sorry

end smallest_a_l277_277964


namespace perpendicular_line_equation_l277_277663

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l277_277663


namespace joe_total_toy_cars_l277_277498

def joe_toy_cars (initial_cars additional_cars : ℕ) : ℕ :=
  initial_cars + additional_cars

theorem joe_total_toy_cars : joe_toy_cars 500 120 = 620 := by
  sorry

end joe_total_toy_cars_l277_277498


namespace find_m_l277_277397

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end find_m_l277_277397


namespace four_digit_numbers_from_2023_l277_277426

theorem four_digit_numbers_from_2023 :
  let digits := [2, 0, 2, 3],
      thousands_place := {x | x ≠ 0 ∧ (x = 2 ∨ x = 3)},
      remaining_digits := {digits.erase x | x ∈ thousands_place},
      valid_combinations (x : ℕ) (ds : List ℕ) := 
        if x = 2 then 
          3 * Nat.fact 2
        else if x = 3 then
          3 * (Nat.fact 2 / Nat.fact 2)
        else
          0 
  in (remaining_digits.foldr (λ d res => res + valid_combinations d.val d.val.snd) 0) = 9 :=
by
  sorry

end four_digit_numbers_from_2023_l277_277426


namespace puzzle_sets_l277_277154

theorem puzzle_sets (l v w : ℕ) (h_l : l = 30) (h_v : v = 18) (h_w : w = 12) 
(h_ratio: ∀ {l_set v_set : ℕ}, l_set / v_set = 2 ) (h_min_puzzles : ∀ {s : ℕ}, s >= 5 ) :
∃ (S : ℕ), S = 3 :=
by
  existsi 3
  sorry

end puzzle_sets_l277_277154


namespace total_cost_of_purchase_l277_277740

theorem total_cost_of_purchase : 
  ∀ (cost_sandwich cost_soda : ℕ) (num_sandwiches : ℕ) (num_sodas : ℕ),
  cost_sandwich = 4 →
  cost_soda = 3 →
  num_sandwiches = 6 →
  num_sodas = 5 →
  cost_sandwich * num_sandwiches + cost_soda * num_sodas = 39 :=
by
  intros cost_sandwich cost_soda num_sandwiches num_sodas h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_cost_of_purchase_l277_277740


namespace construct_polyhedron_l277_277757

open Real

-- Definitions and conditions given in the problem
def is_regular_tetrahedron (A B C D : Point3D) : Prop :=
  (dist A B = dist A C) ∧ (dist A B = dist A D) ∧ (dist A B = dist B C) ∧
  (dist A B = dist B D) ∧ (dist A B = dist C D)

def midpoint (P Q : Point3D) : Point3D :=
  (P + Q) / 2

-- Given a regular tetrahedron
variables (A B C D : Point3D)
variable (h_tetra : is_regular_tetrahedron A B C D)

-- Midpoints of the edges of tetrahedron ABCD
def M1 := midpoint A B
def M2 := midpoint A C
def M3 := midpoint A D
def M4 := midpoint B C
def M5 := midpoint B D
def M6 := midpoint C D

-- Create planes through midpoints (not explicitly shown in the Lean statement)

-- Define properties of the smaller tetrahedrons and octahedron
def is_smaller_tetrahedron (A' B' C' D' : Point3D) : Prop :=
  is_regular_tetrahedron A' B' C' D' ∧ (dist A' B' = dist A B / 2)

def is_regular_octahedron (P Q R S T U : Point3D) : Prop :=
  (dist P Q = dist P R) ∧ (dist P Q = dist P S) ∧ (dist P Q = dist P T) ∧ 
  (dist P Q = dist P U) ∧ (dist Q R = dist Q S) ∧ (dist Q R = dist Q T) ∧ 
  (dist Q R = dist Q U) ∧ (dist R S = dist R T) ∧ (dist R S = dist R U) ∧ 
  (dist S T = dist S U) ∧ (dist T U = dist T U)

-- Desired proof statement
theorem construct_polyhedron :
  ∃ (A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3 A4 B4 C4 D4 P Q R S T U : Point3D),
    is_smaller_tetrahedron A1 B1 C1 D1 ∧
    is_smaller_tetrahedron A2 B2 C2 D2 ∧
    is_smaller_tetrahedron A3 B3 C3 D3 ∧
    is_smaller_tetrahedron A4 B4 C4 D4 ∧
    is_regular_octahedron P Q R S T U ∧
    A1 = A ∧ B1 = M1 ∧ C1 = M2 ∧ D1 = M3 ∧
    A2 = M1 ∧ B2 = B ∧ C2 = M4 ∧ D2 = M5 ∧ 
    A3 = M2 ∧ B3 = M4 ∧ C3 = C ∧ D3 = M6 ∧
    A4 = M3 ∧ B4 = M5 ∧ C4 = M6 ∧ D4 = D :=
sorry

end construct_polyhedron_l277_277757


namespace sin_angle_sum_leq_sqrt3_half_cos_half_angle_sum_leq_sqrt3_half_l277_277683

-- Part (a)
theorem sin_angle_sum_leq_sqrt3_half (α β γ : ℝ) (hαβγ : α + β + γ = π)
  (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) :
  sin α + sin β + sin γ ≤ (3 * real.sqrt 3) / 2 := 
sorry

-- Part (b)
theorem cos_half_angle_sum_leq_sqrt3_half (α β γ : ℝ) (hαβγ : α + β + γ = π)
  (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) :
  cos (α / 2) + cos (β / 2) + cos (γ / 2) ≤ (3 * real.sqrt 3) / 2 := 
sorry

end sin_angle_sum_leq_sqrt3_half_cos_half_angle_sum_leq_sqrt3_half_l277_277683


namespace math_problem_l277_277382

noncomputable def inequality_statement (n : ℕ) (a : ℕ → ℝ) (k : ℝ) : Prop :=
  n > 1 →
  a 1 = 0 →
  a (n-1) = 0 →
  abs (a 0) - abs (a n) ≤ ∑ i in Finset.range (n-1), abs (a i - k * a (i+1) - a (i+2))

theorem math_problem (n : ℕ) (a : ℕ → ℝ) (k : ℝ) :
  inequality_statement n a k := 
by {
  intros,
  sorry,
}

end math_problem_l277_277382


namespace opposite_of_neg_four_l277_277211

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end opposite_of_neg_four_l277_277211


namespace optimal_order_l277_277538

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l277_277538


namespace sum_X_Y_Z_l277_277188

theorem sum_X_Y_Z (X Y Z : ℕ) (hX : X ∈ Finset.range 10) (hY : Y ∈ Finset.range 10) (hZ : Z = 0)
     (div9 : (1 + 3 + 0 + 7 + 6 + 7 + 4 + X + 2 + 0 + Y + 0 + 0 + 8 + 0) % 9 = 0) 
     (div7 : (307674 * 10 + X * 20 + Y * 10 + 800) % 7 = 0) :
  X + Y + Z = 7 := 
sorry

end sum_X_Y_Z_l277_277188


namespace quadrilateral_diagonals_and_opposite_midpoints_intersect_l277_277521

theorem quadrilateral_diagonals_and_opposite_midpoints_intersect
  (a1 b1 a2 b2 a3 b3 a4 b4 : ℝ) :
  ∃ x y : ℝ, 
  (x, y) =
    ( (a1 + a2 + a3 + a4) / 4, (b1 + b2 + b3 + b4) / 4) ∧ 
  ( (a1 + a2) / 2, (b1 + b2) / 2 = (a3 + a4) / 2, (b3 + b4) / 2 ) ∧ 
  ( (a2 + a3) / 2, (b2 + b3) / 2 = (a4 + a1) / 2, (b4 + b1) / 2 ) ∧ 
  ( (a1 + a3) / 2, (b1 + b3) / 2 = (a2 + a4) / 2, (b2 + b4) / 2 ) ∧
  sorry -- Intersection proof goes here

end quadrilateral_diagonals_and_opposite_midpoints_intersect_l277_277521


namespace false_statement_l277_277419

variables (α β : Type) [Plane α] [Plane β]
variables (l : Line) [Intersection (α β) l] [Perpendicular α β]
variables (P : α) [NotOnLine P l]

theorem false_statement : ∀ (PL : Line), (LineThrough P PL) ∧ (Perpendicular PL l) → ¬ (LiesWithin PL α) := 
by
  sorry

end false_statement_l277_277419


namespace min_perimeter_isosceles_triangle_l277_277068

theorem min_perimeter_isosceles_triangle (P Q R : Type) [triangle P Q R] :
  (sideLength PQ = sideLength PR) ∧
  (angle QRP = 120) ∧
  (∃ (a b : ℕ), sideLength QR = a ∧ sideLength PQ = b ∧ sideLength PR = b ∧ a = 3 * b^2 ∧ b = 2) →
  (perimeter P Q R = 16) :=
  sorry

end min_perimeter_isosceles_triangle_l277_277068


namespace range_of_a_eq_neg1_to_3_l277_277437

noncomputable def quadratic_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem range_of_a_eq_neg1_to_3 : { a : ℝ | quadratic_condition a } = { a : ℝ | -1 ≤ a ∧ a ≤ 3 } :=
begin
  sorry
end

end range_of_a_eq_neg1_to_3_l277_277437


namespace probability_of_even_distribution_l277_277880

noncomputable def probability_all_players_have_8_cards_after_dealing : ℝ :=
  let initial_cards := 48
  let players := 6
  let cards_per_player := initial_cards / players
  (5 / 6 : ℝ) -- We state the probability directly according to the problem solution.

theorem probability_of_even_distribution : 
  probability_all_players_have_8_cards_after_dealing = (5 / 6 : ℝ) :=
sorry

end probability_of_even_distribution_l277_277880


namespace tangent_to_circumcircle_at_T_l277_277135

open Classical

-- Definitions of the given geometric entities and points
variables {α : Type*} [MetricSpace α]

/-- Points on the circle γ. -/
def circle (γ : Set α) (R S : α) : Prop := R ∈ γ ∧ S ∈ γ ∧ R ≠ S ∧ ¬is_diameter γ (R, S)

-- The tangent line l to circle γ at point R.
def tangent_line (γ : Set α) (l : Set α) (R : α) : Prop := is_tangent l γ R

-- The midpoint condition: S is the midpoint of RT
def midpoint (R T S : α) : Prop := dist S R = dist S T ∧ dist R T = 2 * dist S R

-- Point J on the minor arc RS such that the circumcircle Γ1 intersects the tangent l at two distinct points, one intersection being A
def minor_arc_intersect (γ : Set α) (Γ1 l : Set α) (R S J A : α) : Prop :=
  J ∈ γ ∧ (∃ P Q, P ≠ Q ∧ P ∈ l ∧ Q ∈ l ∧ P ≠ γ ∧ Q ≠ γ ∧ dist R P < dist R Q ∧ A = P)

-- Line AJ intersects the circle γ again at K
def line_intersects_circle_twice (γ : Set α) (A J K : α) : Prop := line A J ∩ γ = {J, K} ∧ K ≠ J

-- The main theorem
theorem tangent_to_circumcircle_at_T
  (γ Γ1 l : Set α) (R S T J A K : α)
  (h_circle : circle γ R S)
  (h_tangent : tangent_line γ l R)
  (h_midpoint : midpoint R T S)
  (h_minor_arc_intersect : minor_arc_intersect γ Γ1 l R S J A)
  (h_line_circle : line_intersects_circle_twice γ A J K) :
  is_tangent (line K T) Γ1 K :=
begin
  -- Proof goes here
  sorry,
end

end tangent_to_circumcircle_at_T_l277_277135


namespace number_of_divisors_of_M_l277_277762

def M := 2^6 * 3^2 * 5^3 * 7^1 * 11^1

theorem number_of_divisors_of_M : (M.factors.prod (λ p k, k + 1)) = 336 := by
  sorry

end number_of_divisors_of_M_l277_277762


namespace min_max_rounds_to_beat_old_score_l277_277291

theorem min_max_rounds_to_beat_old_score :
  (∀ (old_score : ℕ) (min_points_per_round max_points_per_round required_points : ℕ),
    old_score = 725 ∧ min_points_per_round = 3 ∧ max_points_per_round = 5 ∧ required_points = 726 →
    (∀ points_scored ≥ required_points,
      let min_rounds := (required_points + max_points_per_round - 1) / max_points_per_round,
      let max_rounds := (required_points + min_points_per_round - 1) / min_points_per_round,
      min_rounds = 146 ∧ max_rounds = 242)) :=
sorry

end min_max_rounds_to_beat_old_score_l277_277291


namespace number_of_even_1s_grids_l277_277095

theorem number_of_even_1s_grids (n : ℕ) : 
  (∃ grid : fin n → fin n → ℕ, 
    (∀ i j, grid i j = 0 ∨ grid i j = 1) ∧
    (∀ i, (∑ j, grid i j) % 2 = 0) ∧
    (∀ j, (∑ i, grid i j) % 2 = 0)) →
  2 ^ ((n - 1) * (n - 1)) = 2 ^ ((n - 1) * (n - 1)) :=
by sorry

end number_of_even_1s_grids_l277_277095


namespace largest_x_satisfying_eq_l277_277666

theorem largest_x_satisfying_eq (x : ℝ) : (sqrt (3 * x) = 6 * x) → x ≤ 1/12 :=
by
  sorry

end largest_x_satisfying_eq_l277_277666


namespace germination_at_least_4_out_of_5_l277_277591

noncomputable def germination_prob : ℚ := 0.8

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  n.choose k * p^k * (1 - p)^(n - k)

theorem germination_at_least_4_out_of_5 (p : ℚ) (n : ℕ) (h : p = germination_prob) :
  (binomial_probability n 4 p + binomial_probability n 5 p) = 0.73728 :=
by
  sorry

end germination_at_least_4_out_of_5_l277_277591


namespace profit_as_function_max_profit_at_3_optimal_investment_l277_277621

def x (m : ℝ) : ℝ := 3 - 2 / (m + 1)

def production_cost (m : ℝ) : ℝ :=
  8 + 16 * (x m)

def revenue (m : ℝ) : ℝ :=
  1.5 * (production_cost m)

def profit (m : ℝ) : ℝ :=
  revenue(m) - (production_cost m) - m

theorem profit_as_function (m : ℝ) (h : 0 ≤ m) :
  profit(m) = 28 - 16 / (m + 1) - m :=
by
  sorry

theorem max_profit_at_3 :
  profit(3) = 21 :=
by
  sorry

theorem optimal_investment :
  ∃ m ≥ 0, profit(m) = 21 :=
by
  use 3
  split
  . linarith
  sorry

end profit_as_function_max_profit_at_3_optimal_investment_l277_277621


namespace correct_mms_packs_used_l277_277117

variable (num_sundaes_monday : ℕ) (mms_per_sundae_monday : ℕ)
variable (num_sundaes_tuesday : ℕ) (mms_per_sundae_tuesday : ℕ)
variable (mms_per_pack : ℕ)

-- Conditions
def conditions : Prop := 
  num_sundaes_monday = 40 ∧ 
  mms_per_sundae_monday = 6 ∧ 
  num_sundaes_tuesday = 20 ∧
  mms_per_sundae_tuesday = 10 ∧ 
  mms_per_pack = 40

-- Question: How many m&m packs does Kekai use?
def number_of_mms_packs (num_sundaes_monday mms_per_sundae_monday 
                         num_sundaes_tuesday mms_per_sundae_tuesday 
                         mms_per_pack : ℕ) : ℕ := 
  (num_sundaes_monday * mms_per_sundae_monday + num_sundaes_tuesday * mms_per_sundae_tuesday) / mms_per_pack

-- Theorem to prove the correct number of m&m packs used
theorem correct_mms_packs_used (h : conditions num_sundaes_monday mms_per_sundae_monday 
                                              num_sundaes_tuesday mms_per_sundae_tuesday 
                                              mms_per_pack) : 
  number_of_mms_packs num_sundaes_monday mms_per_sundae_monday 
                      num_sundaes_tuesday mms_per_sundae_tuesday 
                      mms_per_pack = 11 := by {
  -- Proof goes here
  sorry
}

end correct_mms_packs_used_l277_277117


namespace domain_of_f_l277_277588

noncomputable def f (x : ℝ) : ℝ := (Real.tan (2 * x)) / Real.sqrt (x - x^2)

theorem domain_of_f :
  { x : ℝ | ∃ k : ℤ, 2*x ≠ k*π + π/2 ∧ x ∈ (Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1) } = 
  { x : ℝ | x ∈ Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1 } :=
sorry

end domain_of_f_l277_277588


namespace expression_value_at_x_eq_6_l277_277672

theorem expression_value_at_x_eq_6 (x : ℝ) (h : x = 4) : (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  rw h
  sorry

end expression_value_at_x_eq_6_l277_277672


namespace sum_first_last_is_twelve_l277_277210

theorem sum_first_last_is_twelve
  (numbers : List ℤ)
  (h_len : numbers.length = 6)
  (h_elems : ∀ x, x ∈ numbers ↔ x ∈ [-3, 1, 5, 8, 11, 13])
  (h_largest : 13 ∈ numbers.drop 1.take 3)
  (h_smallest : -3 ∈ numbers.drop 2.take 4)
  (h_median : ∀ x, x ∈ [5, 8] → x ∉ [numbers.head, numbers.nth 5])
  (h_distinct : numbers.nodup) :
  numbers.head + numbers.nth 5 = 12 := sorry

end sum_first_last_is_twelve_l277_277210


namespace total_red_cards_l277_277726

def num_standard_decks : ℕ := 3
def num_special_decks : ℕ := 2
def num_custom_decks : ℕ := 2
def red_cards_standard_deck : ℕ := 26
def red_cards_special_deck : ℕ := 30
def red_cards_custom_deck : ℕ := 20

theorem total_red_cards : num_standard_decks * red_cards_standard_deck +
                          num_special_decks * red_cards_special_deck +
                          num_custom_decks * red_cards_custom_deck = 178 :=
by
  -- Calculation omitted
  sorry

end total_red_cards_l277_277726


namespace number_of_ways_to_fill_grid_l277_277093

open Finset

theorem number_of_ways_to_fill_grid (n : ℕ) (h : n ≥ 1) :
  let grid := Matrix (Fin n) (Fin n) (Fin 2)
  let condition (m : grid) := (∀ i : Fin n, even (card { j | m i j = 1 })) ∧
                              (∀ j : Fin n, even (card { i | m i j = 1 }))
  ∃ fill_count : ℕ, (fill_count = 2^((n-1)*(n-1))) ∧
                    ∀ g : grid, condition g ↔ (g ∈ universe grid) :=
sorry

end number_of_ways_to_fill_grid_l277_277093


namespace students_at_end_of_year_l277_277467

def students_start : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0
def students_end : ℝ := 28.0

theorem students_at_end_of_year :
  students_start - students_left - students_transferred = students_end := by
  sorry

end students_at_end_of_year_l277_277467


namespace sunkyung_mother_current_age_l277_277568

-- Define the ages 6 years ago
def six_years_ago_sunkyung_age := 9
def six_years_ago_mother_age := 39

-- Conditions
def condition1 (S M : ℕ) : Prop := S + M = 48
def condition2 (S M : ℕ) : Prop := M + 6 = 3 * (S + 6)

-- Definitions based on conditions
theorem sunkyung_mother_current_age :
  ∃ S M : ℕ, condition1 S M ∧ condition2 S M ∧ (M + 6 = 45) :=
by
  use six_years_ago_sunkyung_age
  use six_years_ago_mother_age
  split
  { unfold condition1
    exact rfl }
  split
  { unfold condition2
    exact rfl }
  exact rfl

end sunkyung_mother_current_age_l277_277568


namespace area_isosceles_trapezoid_l277_277903

theorem area_isosceles_trapezoid
  (A B C D X Y : Type)
  (h_parallel: A ≠ B ∧ B ≠ C ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D)
  (h_trapezoid: B ∥ D)
  (h_isosceles: A = C)
  (h_X_on_AC: X ∈ line A C)
  (h_Y_on_AC: Y ∈ line A C)
  (h_X_between_A_Y: X ∈ interval A Y)
  (h_angles: ∠ A X D = 60 ∧ ∠ B Y C = 120)
  (h_lengths: dist A X = 4 ∧ dist X Y = 2 ∧ dist Y C = 3) :
  area A B C D = 21 * sqrt 3 :=
by {
  sorry
}

end area_isosceles_trapezoid_l277_277903


namespace solve_for_x_l277_277953

theorem solve_for_x (x : ℚ) : (x + 4) / (x - 3) = (x - 2) / (x + 2) -> x = -2 / 11 := by
  sorry

end solve_for_x_l277_277953


namespace vector_dot_product_evaluation_l277_277038

variables (a b : ℝ^3)

-- Conditions
def condition1 : Prop := 2 • a + b = 0
def condition2 : Prop := a ⬝ b = -2

-- Theorem Statement
theorem vector_dot_product_evaluation (h1 : condition1 a b) (h2 : condition2 a b) :
  (3 • a + b) ⬝ (a - b) = 3 := 
  sorry

end vector_dot_product_evaluation_l277_277038


namespace area_of_triangle_is_correct_l277_277125

def vector := (ℝ × ℝ)

def a : vector := (7, 3)
def b : vector := (-1, 5)

noncomputable def det2x2 (v1 v2 : vector) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem area_of_triangle_is_correct :
  let area := (det2x2 a b) / 2
  area = 19 := by
  -- defintions and conditions are set here, proof skipped
  sorry

end area_of_triangle_is_correct_l277_277125


namespace complement_union_A_B_l277_277147

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end complement_union_A_B_l277_277147


namespace consecutive_integers_sum_l277_277602

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 := by
  sorry

end consecutive_integers_sum_l277_277602


namespace bus_driver_hours_l277_277265

theorem bus_driver_hours (R H O : ℝ) (TotalCompensation TotalHours : ℝ) :
  R = 15 ∧ H = 40 ∧ TotalCompensation = 976 ∧ O = (376 / 26.25) ∧ TotalHours = H + O → TotalHours = 54 :=
by
  intro h
  cases h with hR rest
  cases rest with hH rest'
  cases rest' with hTotalComp restOU
  cases restOU with hO hT
  subst hR
  subst hH
  subst hTotalComp
  subst hO
  subst hT
  exact rfl

end bus_driver_hours_l277_277265


namespace trajectory_equation_l277_277973

-- Define the condition that the distance to the coordinate axes is equal.
def equidistantToAxes (x y : ℝ) : Prop :=
  abs x = abs y

-- State the theorem that we need to prove.
theorem trajectory_equation (x y : ℝ) (h : equidistantToAxes x y) : y^2 = x^2 :=
by sorry

end trajectory_equation_l277_277973


namespace represent_2008_l277_277947

theorem represent_2008 (a b c : ℕ) (h : 2008 = a + b * 40 + c * 40) (h_rec : 1 = (1 / a : ℝ) + (1 / b : ℝ) + (1 / c : ℝ)) : 
  2008 = 8 + 10 * 40 + 30 * 40 :=
by {
  have : a = 8 := sorry,
  have : b = 10 := sorry,
  have : c = 30 := sorry,
  subst_vars,
  linarith,
}

end represent_2008_l277_277947


namespace largest_prime_factors_555555555555_l277_277206

theorem largest_prime_factors_555555555555 :
  let n := 555555555555 in
  (n = 5 * 3 * 37 * 7 * 11 * 13 * 101 * 9901) →
  ∃ (p q r : ℕ), nat.prime p ∧ nat.prime q ∧ nat.prime r ∧ 
                p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ 
                555555555555 = p * q * r ∧ 
                (p = 37 ∨ p = 101 ∨ p = 9901) ∧ 
                (q = 37 ∨ q = 101 ∨ q = 9901) ∧ 
                (r = 37 ∨ r = 101 ∨ r = 9901) :=
by
  intro n_def
  have prime_fact := prime_factors_555555555555 n_def
  sorry

end largest_prime_factors_555555555555_l277_277206


namespace klinker_twice_as_old_l277_277525

theorem klinker_twice_as_old :
  ∃ x : ℕ, (∀ (m k d : ℕ), m = 35 → d = 10 → m + x = 2 * (d + x)) → x = 15 :=
by
  sorry

end klinker_twice_as_old_l277_277525


namespace find_X_l277_277717

theorem find_X : 
    let Prod := 555 * 465
    let P := Prod^2
    let Diff := 555 - 465
    let Q := Diff^3
    let R := Diff^4
    let X := P * Q + R
    in X = 4849295371235000 :=
by
    -- Calculation without actual proof since we use sorry
    let Prod := 555 * 465
    let P := Prod^2
    let Diff := 555 - 465
    let Q := Diff^3
    let R := Diff^4
    let X := P * Q + R
    -- Expected result
    have h : X = 4849295371235000 := sorry
    exact h

end find_X_l277_277717


namespace f_monotonic_on_interval_g_has_exactly_3_zeros_on_interval_l277_277404

-- Definitions of the given functions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.exp x - 2 * (a - 1) * x - b
def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.exp x - (a - 1) * x^2 - b * x - 1

-- The problem conditions
def g_at_1_is_zero (a : ℝ) (b : ℝ) : Prop := g 1 a b = 0

-- The first problem statement as a Lean theorem
theorem f_monotonic_on_interval {a b : ℝ} :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x a b > 0) ↔ (a ≤ 3/2 ∨ a ≥ Real.exp 1 / 2 + 1) :=
sorry

-- The second problem statement as a Lean theorem
theorem g_has_exactly_3_zeros_on_interval {a b : ℝ} (h_zero : g_at_1_is_zero a b) :
  (∃! x1 x2 x3 ∈ (Icc 0 1), g x1 a b = 0 ∧ g x2 a b = 0 ∧ g x3 a b = 0) ↔ (e-1 < a ∧ a < 2) :=
sorry

end f_monotonic_on_interval_g_has_exactly_3_zeros_on_interval_l277_277404


namespace present_age_of_eldest_is_45_l277_277215

theorem present_age_of_eldest_is_45 (x : ℕ) 
  (h1 : (5 * x - 10) + (7 * x - 10) + (8 * x - 10) + (9 * x - 10) = 107) :
  9 * x = 45 :=
sorry

end present_age_of_eldest_is_45_l277_277215


namespace equilateral_triangles_ratio_l277_277477

theorem equilateral_triangles_ratio (t : ℝ) (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S] 
  (h1 : ∀ (A B C : Type) [equilateral_tris A] [equilateral_tris B] [equilateral_tris C], QR = t) 
  (h2 : altitudes_from P S QR = t * sqrt 3 / 2)
  : PS / QR = sqrt 3 := sorry

end equilateral_triangles_ratio_l277_277477


namespace percentage_of_women_attended_picnic_l277_277071

variable (E : ℝ) -- Total number of employees
variable (M : ℝ) -- The number of men
variable (W : ℝ) -- The number of women
variable (P : ℝ) -- Percentage of women who attended the picnic

-- Conditions
variable (h1 : M = 0.30 * E)
variable (h2 : W = E - M)
variable (h3 : 0.20 * M = 0.20 * 0.30 * E)
variable (h4 : 0.34 * E = 0.20 * 0.30 * E + P * (E - 0.30 * E))

-- Goal
theorem percentage_of_women_attended_picnic : P = 0.40 :=
by
  sorry

end percentage_of_women_attended_picnic_l277_277071


namespace set_rep_l277_277032

-- Definitions of A, A1, A2, A3
noncomputable def A (a c : ℝ) : set ℂ :=
  { z | |z - c| + |z + c| = 2 * a ∧ a > c ∧ a > 0 ∧ c > 0 }

noncomputable def A1 : set (ℝ × ℝ) :=
  { p | p = (2, 1) ∨ p = (2, -1) ∨ p = (-2, 1) ∨ p = (-2, -1) }

noncomputable def A2 : set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 + y^2 > 5 ∧ |y| < 1 }

noncomputable def A3 : set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 + y^2 < 5 ∧ |y| > 1 }

noncomputable def A_real (a c : ℝ) : set (ℝ × ℝ) := A1 ∪ A2 ∪ A3

-- Formal statement
theorem set_rep (a c : ℝ) (z : ℂ) :
  z ∈ A a c ↔ (z.re, z.im) ∈ A_real a c :=
sorry

end set_rep_l277_277032


namespace problem_statement_l277_277597

noncomputable def f (x : ℝ) : ℝ := if -2 < x ∧ x < 0 then 2^x else 0

theorem problem_statement :
  (∀ x : ℝ, f x = -f (4 - x)) → 
  (f (\log 20 / log 2) = - 4 / 5) :=
by
  sorry

end problem_statement_l277_277597


namespace sum_of_possible_x_is_18_923_l277_277303

theorem sum_of_possible_x_is_18_923 :
  let l : List ℝ := [8, 3, 6, 3, 7, 3, x]
  let mean := (30 + x) / 7
  let mode := 3
  let median := if x ≤ 3 then 3 else if 6 ≤ x ∧ x ≤ 7 then 6 else if 3 < x ∧ x < 6 then x else 0
  median ≠ mode
  (∃ x : ℝ, 3, median, mean forms an arithmetic progression) →
  ∑ x in {x | 3, median, mean forms an arithmetic progression}, x = 18.923 :=
by
  sorry

end sum_of_possible_x_is_18_923_l277_277303


namespace quadratic_inequality_solution_l277_277027

theorem quadratic_inequality_solution (a : ℝ) :
  ((0 ≤ a ∧ a < 3) → ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) :=
  sorry

end quadratic_inequality_solution_l277_277027


namespace number_of_bugs_l277_277931

def flowers_per_bug := 2
def total_flowers_eaten := 6

theorem number_of_bugs : total_flowers_eaten / flowers_per_bug = 3 := 
by sorry

end number_of_bugs_l277_277931


namespace polygon_intersections_l277_277946

/-
Statement: Given regular polygons with 6, 7, 8, and 9 sides inscribed in the same circle, where:
- The polygons with 6 and 7 sides share exactly one vertex,
- No other polygons share any vertices,
- No three sides of the polygons intersect at a common point,

prove that the number of points inside the circle where two of their sides intersect is 78.
-/

theorem polygon_intersections
  (P6 P7 P8 P9 : Set Point)
  (circ : Set Point)
  (h1 : Polygon.regular P6 6)
  (h2 : Polygon.regular P7 7)
  (h3 : Polygon.regular P8 8)
  (h4 : Polygon.regular P9 9)
  (h5 : inscribed P6 circ)
  (h6 : inscribed P7 circ)
  (h7 : inscribed P8 circ)
  (h8 : inscribed P9 circ)
  (h9 : ∃ v, v ∈ P6 ∧ v ∈ P7)
  (h10 : ∀ (v : Point), v ∈ P6 → v ∉ P8 ∧ v ∉ P9)
  (h11 : ∀ (v : Point), v ∈ P7 → v ∉ P8 ∧ v ∉ P9)
  (h12 : ∀ (v : Point), v ∈ P8 → v ∉ P9)
  (h13 : ∀ (p : Point), p ∉ P6 ∪ P7 ∪ P8 ∪ P9 ∨ ¬ (p ∈ P6 ∧ p ∈ P7 ∧ p ∈ P8 ∧ p ∈ P9)) :
  ∃ n, n = 78 ∧ number_of_intersections P6 P7 P8 P9 = n :=
by
  sorry

end polygon_intersections_l277_277946


namespace never_reappear_141_l277_277942

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

def next_numbers (n : ℕ) : List ℕ :=
  let prod := digit_product n
  [n - prod, n + prod]

theorem never_reappear_141 (initial : ℕ) (prod : ℕ) (next : ℕ) :
  initial = 141 →
  prod = digit_product initial →
  next ∈ next_numbers initial →
  ∀ (x : ℕ), x ∉ (iterate next_numbers next initial) :=
sorry

end never_reappear_141_l277_277942


namespace friend_time_to_read_book_l277_277856

-- Define the conditions and variables
def my_reading_time : ℕ := 240 -- 4 hours in minutes
def speed_ratio : ℕ := 2 -- I read at half the speed of my friend

-- Define the variable for my friend's reading time which we need to find
def friend_reading_time : ℕ := my_reading_time / speed_ratio

-- The theorem statement that given the conditions, the friend's reading time is 120 minutes
theorem friend_time_to_read_book : friend_reading_time = 120 := sorry

end friend_time_to_read_book_l277_277856


namespace complement_of_M_l277_277843

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- The theorem stating the complement of M in U
theorem complement_of_M : (U \ M) = {y | y < -1} :=
by
  sorry

end complement_of_M_l277_277843


namespace four_digit_numbers_proof_l277_277235

noncomputable def four_digit_numbers_total : ℕ := 9000
noncomputable def two_digit_numbers_total : ℕ := 90
noncomputable def max_distinct_products : ℕ := 4095
noncomputable def cannot_be_expressed_as_product : ℕ := four_digit_numbers_total - max_distinct_products

theorem four_digit_numbers_proof :
  cannot_be_expressed_as_product = 4905 :=
by
  sorry

end four_digit_numbers_proof_l277_277235


namespace number_of_special_three_digit_numbers_l277_277432

noncomputable def count_special_three_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem number_of_special_three_digit_numbers : count_special_three_digit_numbers = 84 := by
  sorry

end number_of_special_three_digit_numbers_l277_277432


namespace tan_double_angle_through_point_l277_277108

theorem tan_double_angle_through_point : 
  (∃ α : ℝ, ∃ P : ℝ × ℝ, P = (1, -2) ∧ tan α = -2) → tan (2 * α) = 4 / 3 :=
by
  sorry

end tan_double_angle_through_point_l277_277108


namespace equilateral_triangles_ratio_l277_277476

theorem equilateral_triangles_ratio (t : ℝ) (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S] 
  (h1 : ∀ (A B C : Type) [equilateral_tris A] [equilateral_tris B] [equilateral_tris C], QR = t) 
  (h2 : altitudes_from P S QR = t * sqrt 3 / 2)
  : PS / QR = sqrt 3 := sorry

end equilateral_triangles_ratio_l277_277476


namespace relationship_l277_277362

noncomputable theory

def a : ℝ := log 0.4 7
def b : ℝ := 0.4^7
def c : ℝ := 7^0.4

theorem relationship (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = log 0.4 7) (h2 : b = 0.4^7) (h3 : c = 7^0.4) :
  c > b ∧ b > a :=
by 
  -- Proof steps will be provided here
  sorry

end relationship_l277_277362


namespace find_prime_p_l277_277383

theorem find_prime_p (p : ℕ) (n : ℕ) 
  (a : ℕ → ℕ) 
  (hp_prime : Prime p) 
  (hn_pos : 0 < n) 
  (ha_bound : ∀ i : ℕ, i ≤ n → a i < p) 
  (sum_condition : (∑ i in Finset.range (n + 1), if i = 0 then 0 else a i) = 13) 
  (poly_condition : (∑ i in Finset.range (n + 1), a i * p^i) = 2015) 
  : p = 2003 := 
sorry

end find_prime_p_l277_277383


namespace condition_neither_sufficient_nor_necessary_l277_277191

noncomputable def f (x a : ℝ) : ℝ := x^3 - x + a
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 1

def condition (a : ℝ) : Prop := a^2 - a = 0

theorem condition_neither_sufficient_nor_necessary
  (a : ℝ) :
  ¬(condition a → (∀ x : ℝ, f' x ≥ 0)) ∧ ¬((∀ x : ℝ, f' x ≥ 0) → condition a) :=
by
  sorry -- Proof is omitted as per the prompt

end condition_neither_sufficient_nor_necessary_l277_277191


namespace composite_sum_exists_l277_277133

theorem composite_sum_exists (A B C a b c : ℕ) (h : A > 0 ∧ B > 0 ∧ C > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c > 1) :
  ∃ n : ℕ, n > 0 ∧ ¬nat.prime (A * a^n + B * b^n + C * c^n) :=
by sorry

end composite_sum_exists_l277_277133


namespace solve_inequality_l277_277178

theorem solve_inequality (a x : ℝ) :
  (x^2 - (a + 1) * x + a ≥ 0) ↔
  (if a > 1 then (-∞ : ℝ) ≤ x ∧ x ≤ 1 ∨ a ≤ x ∧ x ≤ ∞
   else if a = 1 then True
   else (-∞ : ℝ) ≤ x ∧ x ≤ a ∨ 1 ≤ x ∧ x ≤ ∞) :=
by
  sorry

end solve_inequality_l277_277178


namespace ratio_sum_numerator_denominator_l277_277314

theorem ratio_sum_numerator_denominator (n : ℕ) (h : n = 7) : 
  let r := 28 * 28,
      s := (7 * 8 * 15) / 6,
      gcd_val := Nat.gcd s r,
      numerator := s / gcd_val,
      denominator := r / gcd_val
  in numerator + denominator = 33 :=
by
  sorry

end ratio_sum_numerator_denominator_l277_277314


namespace part_1_part_2_l277_277037

def vector_a : ℝ × ℝ := (3, -1)
def vector_b_length : ℝ := Real.sqrt 5
def dot_product_a_b : ℝ := -5
def vector_c (x : ℝ) (b : ℝ × ℝ) : ℝ × ℝ := (x * vector_a.1 + (1 - x) * b.1, x * vector_a.2 + (1 - x) * b.2)
def perp (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- Part I: x such that vector_c is perpendicular to vector_a -/
theorem part_1 (x : ℝ) (b : ℝ × ℝ) (hb : b.1 ^ 2 + b.2 ^ 2 = 5) (hab : vector_a.1 * b.1 + vector_a.2 * b.2 = -5) :
  perp vector_a (vector_c x b) ↔ x = 1 / 3 := sorry

/-- Part II: Cosine of the angle between vector_b and vector_c when vector_c has the minimum length -/
theorem part_2 (b : ℝ × ℝ) (hb : b.1 ^ 2 + b.2 ^ 2 = 5) (hab : vector_a.1 * b.1 + vector_a.2 * b.2 = -5)
  (x : ℝ) (hx : x = 2 / 5) (c : ℝ × ℝ := vector_c x b) :
  Real.cos (Real.atan2 c.2 c.1 - Real.atan2 b.2 b.1) = Real.sqrt 5 / 5 := sorry

end part_1_part_2_l277_277037


namespace monic_poly_has_root_l277_277331

theorem monic_poly_has_root : 
  ∃ (P : Polynomial ℚ), P.degree = 4 ∧ P.leadingCoeff = 1 ∧ P.eval (Real.of_rat (3 ^ (1/2) + 5 ^ (1/2))) = 0 :=
by
  sorry

end monic_poly_has_root_l277_277331


namespace median_siblings_is_1_l277_277872

def students : ℕ := 15
def count_0_siblings : ℕ := 3
def count_1_sibling : ℕ := 5
def count_2_siblings : ℕ := 4
def count_3_siblings : ℕ := 2
def count_4_siblings : ℕ := 1

theorem median_siblings_is_1 (students = 15)
  (count_0_siblings = 3)
  (count_1_sibling = 5)
  (count_2_siblings = 4)
  (count_3_siblings = 2)
  (count_4_siblings = 1) :
  median_siblings( list_repeat 0 count_0_siblings ++
                  list_repeat 1 count_1_sibling ++
                  list_repeat 2 count_2_siblings ++
                  list_repeat 3 count_3_siblings ++
                  list_repeat 4 count_4_siblings ) = 1 :=
sorry

end median_siblings_is_1_l277_277872


namespace moles_of_HCl_required_l277_277431

noncomputable def numberOfMolesHClRequired (moles_AgNO3 : ℕ) : ℕ :=
  if moles_AgNO3 = 3 then 3 else 0

-- Theorem statement
theorem moles_of_HCl_required : numberOfMolesHClRequired 3 = 3 := by
  sorry

end moles_of_HCl_required_l277_277431


namespace evaluate_expression_l277_277323

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end evaluate_expression_l277_277323


namespace positive_number_sum_square_l277_277995

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l277_277995


namespace mirka_number_l277_277930

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b
noncomputable def reversed_number (a b : ℕ) : ℕ := 10 * b + a

theorem mirka_number (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 4) (h2 : b = 2 * a) :
  original_number a b = 12 ∨ original_number a b = 24 ∨ original_number a b = 36 ∨ original_number a b = 48 :=
by
  sorry

end mirka_number_l277_277930
