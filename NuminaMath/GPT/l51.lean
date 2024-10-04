import Mathlib

namespace min_value_of_f_l51_51026

noncomputable def f (x a : ℝ) := x^2 + x + a

theorem min_value_of_f (a : ℝ) (h_max : ∀ x ∈ set.Icc (-1:ℝ) (1:ℝ), f x a ≤ 2) :
  ∃ x ∈ set.Icc (-1:ℝ) (1:ℝ), f x a = -1/4 := 
begin
  have ha : a = 0,
  { 
    have := h_max 1 (by norm_num),
    simp [f, pow_two] at this,
    linarith,
  },
  refine ⟨-1/2, by norm_num, _⟩,
  simp [f, ha],
  norm_num,
end

end min_value_of_f_l51_51026


namespace joan_balloons_l51_51492

theorem joan_balloons (initial_balloons : ℕ) (gained_balloons : ℕ) (initial_balloons_eq : initial_balloons = 9) (gained_balloons_eq : gained_balloons = 2) : initial_balloons + gained_balloons = 11 :=
by
  -- Using the conditions from the problem to prove the statement
  rw [initial_balloons_eq, gained_balloons_eq]
  -- Now proving the simple arithmetic
  exact Nat.add_comm 9 2

end joan_balloons_l51_51492


namespace simplify_complex_expression_l51_51616

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l51_51616


namespace initial_peaches_proof_l51_51288

noncomputable def initial_peaches : ℕ :=
  let ripe_peaches (day : ℕ) : ℕ :=
    if day = 1 then 4 + 2
    else if day = 2 then 6 + 2
    else if day = 3 then 8 + 2 - 3
    else if day = 4 then 7 + 2
    else if day = 5 then 9 + 2
    else 0
  let unripe_peaches_at_end := ripe_peaches 5 - 7
  unripe_peaches_at_end + 3 + 2 * 5

theorem initial_peaches_proof : initial_peaches = 17 :=
by
  have h1 : ripe_peaches 1 = 6 := rfl
  have h2 : ripe_peaches 2 = 8 := rfl
  have h3 : ripe_peaches 3 = 7 := rfl
  have h4 : ripe_peaches 4 = 9 := rfl
  have h5 : ripe_peaches 5 = 11 := rfl
  let U := ripe_peaches 5 - 7
  have hU : U = 4 := by linarith
  have total_peaches_initially := U + 3 + 2 * 5
  have h_total_peaches : total_peaches_initially = 17 := by linarith
  exact h_total_peaches

end initial_peaches_proof_l51_51288


namespace zookeeper_feeding_ways_l51_51752

/-- We define the total number of ways the zookeeper can feed all the animals following the rules. -/
def feed_animal_ways : ℕ :=
  6 * 5^2 * 4^2 * 3^2 * 2^2 * 1^2

/-- Theorem statement: The number of ways to feed all the animals is 86400. -/
theorem zookeeper_feeding_ways : feed_animal_ways = 86400 :=
by
  sorry

end zookeeper_feeding_ways_l51_51752


namespace inequality_proof_l51_51447

theorem inequality_proof (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
    (x * y * z) / ((1 + 5 * x) * (4 * x + 3 * y) * (5 * y + 6 * z) * (z + 18)) ≤ (1 : ℝ) / 5120 := 
by
  sorry

end inequality_proof_l51_51447


namespace tiling_impossible_l51_51486

theorem tiling_impossible (m n k l : ℕ) (r : ℕ) : 
  m = 11 → n = 12 → k = 19 → l = 6 → r = 7 → 
  ¬ ∃ (f : ℕ × ℕ → ℕ × ℕ), (∀ (p : ℕ × ℕ), 
  let (x, y) := p in f p = (x + l mod m, y + r mod n) ∨ f p = (x + r mod m, y + l mod n)) :=
by 
  intros h_m h_n h_k h_l h_r
  sorry

end tiling_impossible_l51_51486


namespace prob_center_in_tetrahedron_is_eighth_l51_51843

noncomputable def prob_center_in_tetrahedron (A B C D : point) : ℝ :=
  if center_in_tetrahedron A B C D then 1/8 else 0

theorem prob_center_in_tetrahedron_is_eighth (A B C D : point) :
  prob_center_in_tetrahedron A B C D = 1/8 := by
  sorry

end prob_center_in_tetrahedron_is_eighth_l51_51843


namespace circumcenter_AXY_on_circumcircle_ABC_l51_51100

-- Definitions representing points and conditions
variables {A B C X Y : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space Y]
variables (AX : dist A X) (BX : dist B X) (BA : dist B A) (CY : dist C Y) (CA : dist C A)

-- Triangle similarity condition
axiom similar_BAX_CAY : ∀ (BAX CAY : triangle), ∀ (B X A C Y : point),
  similar BAX CAY ↔ (dist B X = dist B A) ∧ (dist C Y = dist C A) ∧ (B, C are on the same side of line AB and not AC)

-- Problem statement: Prove the circumcenter of triangle AXY lies on the circumcircle of triangle ABC
theorem circumcenter_AXY_on_circumcircle_ABC :
  (∃ (AXY_triangle : triangle), circumcenter AXY_triangle ∈ circumcircle (triangle A B C)) :=
sorry

end circumcenter_AXY_on_circumcircle_ABC_l51_51100


namespace forty_percent_of_thirty_percent_l51_51906

theorem forty_percent_of_thirty_percent (x : ℝ) 
  (h : 0.3 * 0.4 * x = 48) : 0.4 * 0.3 * x = 48 :=
by
  sorry

end forty_percent_of_thirty_percent_l51_51906


namespace closest_perfect_square_to_350_l51_51646

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l51_51646


namespace Kahi_memorized_today_l51_51942

variable (total_words : ℕ) (memorized_yesterday_fraction : ℚ) (memorized_today_fraction : ℚ)

def remaining_words (total : ℕ) (fraction_yesterday : ℚ) : ℕ :=
  total - (fraction_yesterday * total).toNat

def memorized_today (remaining : ℕ) (fraction_today : ℚ) : ℕ :=
  (fraction_today * remaining).toNat

theorem Kahi_memorized_today :
  remaining_words 810 (1 / 9) * (1 / 4) = 180 := 
by
  sorry

end Kahi_memorized_today_l51_51942


namespace angle_between_a_b_l51_51396

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Hypotheses
def h1 : a + b + c = 0 :=
sorry

def h2 : ∥a∥ = 1 :=
sorry

def h3 : ∥b∥ = 2 :=
sorry

def h4 : ∥c∥ = sqrt 7 :=
sorry

-- Goal
theorem angle_between_a_b : 
  let θ := real.arccos ((inner a b) / (∥a∥ * ∥b∥)) in
  θ = real.pi / 3 :=
sorry

end angle_between_a_b_l51_51396


namespace num_correct_statements_l51_51950

variables {m n : Line} {α β : Plane} {A : Point}

-- Assume the necessary conditions for the problem
axiom diff_lines (hmn : m ≠ n)
axiom diff_planes (hαβ : α ≠ β)

-- Statements as conditions
axiom s1 (hmα : m ∥ α) (hmβ : m ∥ β) : ¬(α ∥ β)
axiom s2 (hαβ : α ∥ β) (hmα : m ⊆ α) (hnβ : n ⊆ β) : ¬(m ∥ n)
axiom s3 (hαβ : α ∥ β) (hmn : m ∥ n) (hmα : m ∥ α) : ¬(n ∥ β)
axiom s4 (hmα : m ∥ α) (hmβ : m ⊆ β) (hαβ : α ∩ β = n) : m ∥ n
axiom s5 (hmα : m ⊆ α) (hnA : n ∩ α = {A}) (hA : A ∉ m) : ¬coplanar m n

-- Prove the number of correct statements is 2
theorem num_correct_statements : 2 = 5 - ((¬(α ∥ β)) + (¬(m ∥ n)) + (¬(n ∥ β))) :=
sorry

end num_correct_statements_l51_51950


namespace probability_not_orange_not_white_l51_51041

theorem probability_not_orange_not_white (num_orange num_black num_white : ℕ)
    (h_orange : num_orange = 8) (h_black : num_black = 7) (h_white : num_white = 6) :
    (num_black : ℚ) / (num_orange + num_black + num_white : ℚ) = 1 / 3 :=
  by
    -- Solution will be here.
    sorry

end probability_not_orange_not_white_l51_51041


namespace heptagram_convex_quadrilaterals_l51_51044

theorem heptagram_convex_quadrilaterals : ∀ (heptagram : Type), (number_of_convex_quadrilaterals heptagram) = 7 :=
sorry

end heptagram_convex_quadrilaterals_l51_51044


namespace simplify_complex_expression_l51_51617

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l51_51617


namespace jose_profit_share_l51_51699

theorem jose_profit_share (investment_tom : ℕ) (months_tom : ℕ) 
                         (investment_jose : ℕ) (months_jose : ℕ) 
                         (total_profit : ℕ) :
                         investment_tom = 30000 →
                         months_tom = 12 →
                         investment_jose = 45000 →
                         months_jose = 10 →
                         total_profit = 63000 →
                         (investment_jose * months_jose / 
                         (investment_tom * months_tom + investment_jose * months_jose)) * total_profit = 35000 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  norm_num
  sorry

end jose_profit_share_l51_51699


namespace Vintik_received_more_than_850_l51_51477

variables (Sh K V : ℕ)
variables (ka : ℝ) -- coefficient for votes ratio of Vintik to Shpuntik.

-- Given conditions
def condition_1 := ∀ S : ℕ, (0.46 * (Sh + V + K) = (ka * Sh) / (1 + ka)) + ((Sh + K) / (Sh + ka * Sh + K)) * (Sh + ka * Sh + K)
def condition_2 := Sh > 1000 -- Shpuntik received more than 1000 votes.

-- Statement to prove
theorem Vintik_received_more_than_850 (Sh V K : ℕ) (ka : ℝ) :
  condition_1 Sh K V ka → condition_2 Sh → V > 850 :=
sorry

end Vintik_received_more_than_850_l51_51477


namespace kolya_screen_display_limit_l51_51083

theorem kolya_screen_display_limit : 
  let n := 1024
  let screen_limit := 10^16
  (∀ d ∈ list.divisors n, d = (2^0) ∨ d = (2^1) ∨ d = (2^2) ∨ d = (2^3) ∨ d = (2^4) ∨ d = (2^5) ∨ d = (2^6) ∨ d = (2^7) ∨ d = (2^8) ∨ d = (2^9) ∨ d = (2^10)) →
  ((list.prod (list.divisors n)) > screen_limit) :=
begin
  sorry
end

end kolya_screen_display_limit_l51_51083


namespace number_of_proper_subsets_of_A_l51_51455

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def complement_U_A : Set ℕ := {1, 2, 3}
def A : Set ℕ := {0, 4, 5}

theorem number_of_proper_subsets_of_A :
  (finset.powerset (finset.of_set A)).card - 1 = 7 :=
by sorry

end number_of_proper_subsets_of_A_l51_51455


namespace simplify_and_substitute_expr_l51_51152

theorem simplify_and_substitute_expr (x : ℕ) (h : x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 2) : 
  ( (x^2 - 2*x + 1)/(x^2 - 1) ∈ (1 - 3/(x+1)) = (x-1)/(x-2) ) ∧
  ( x = 3 → (x-1)/(x-2) = 2 ) :=
by sorry

end simplify_and_substitute_expr_l51_51152


namespace last_digit_decimal_expansion_l51_51629

theorem last_digit_decimal_expansion :
  let x : ℚ := 1 / 3^15
  in (x.to_decimal.last_digit_unit = 5) :=
by
  sorry

end last_digit_decimal_expansion_l51_51629


namespace reconstruct_point_A_l51_51723

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

end reconstruct_point_A_l51_51723


namespace three_dimensional_pythagorean_theorem_l51_51612

def trirectangular_tetrahedron (x y z : ℝ) : Prop :=
  ∃ (a b c d : ℝ × ℝ × ℝ),
    a = (0, 0, 0) ∧
    b = (x, 0, 0) ∧
    c = (0, y, 0) ∧
    d = (0, 0, z)

theorem three_dimensional_pythagorean_theorem (x y z : ℝ)
  (h : trirectangular_tetrahedron x y z) :
  let A := (1/2) * x * y
      B := (1/2) * y * z
      C := (1/2) * x * z
      D := (1/2) * real.sqrt ((y*z)^2 + (x*z)^2 + (x*y)^2)
  in (D ^ 2) = (A ^ 2) + (B ^ 2) + (C ^ 2) :=
by
  sorry

end three_dimensional_pythagorean_theorem_l51_51612


namespace find_c_for_Q_l51_51643

noncomputable def Q (c : ℚ) (x : ℚ) : ℚ := x^3 + 3*x^2 + c*x + 8

theorem find_c_for_Q (c : ℚ) : 
  (Q c 3 = 0) ↔ (c = -62 / 3) := by
  sorry

end find_c_for_Q_l51_51643


namespace investment_amount_l51_51350

noncomputable def annual_income (investment : ℝ) (percent_stock : ℝ) (market_price : ℝ) : ℝ :=
  (investment * percent_stock / 100) / market_price * market_price

theorem investment_amount (annual_income_value : ℝ) (percent_stock : ℝ) (market_price : ℝ) (investment : ℝ) :
  annual_income investment percent_stock market_price = annual_income_value →
  investment = 6800 :=
by
  intros
  sorry

end investment_amount_l51_51350


namespace max_omega_satisfying_conditions_l51_51417

theorem max_omega_satisfying_conditions : 
  ∃ ω > 0, (∀ φ, (|φ| ≤ π / 2) ∧ 
  (∀ x, (x = -π / 4 → sin (ω * x + φ) = 0)) ∧ 
  (∀ x, (x = π / 4 → sin (ω * x + φ) = sin (ω * x + φ))) ∧ 
  (∀ x, (π / 18 < x ∧ x < 5 * π / 36 → 
         (∀ y z, (π / 18 < y ∧ y < 5 * π / 36 ∧ y < z ∧ z < 5 * π / 36 → (sin (ω * y + φ) < sin (ω * z + φ))))))) → 
  ω = 9 := sorry

end max_omega_satisfying_conditions_l51_51417


namespace rain_prob_three_days_l51_51595

-- Define the events and their probabilities
def P (A : Prop) : ℝ := sorry  -- Assume probabilities are real numbers

axiom P_A : P (rain_on_friday) = 0.4
axiom P_B : P (rain_on_saturday) = 0.7
axiom P_C : P (rain_on_sunday) = 0.3
axiom independence : ∀ (A B : Prop), P (A ∧ B) = P (A) * P (B) -- Independence axiom

-- Define the events as propositions
def rain_on_friday : Prop := sorry
def rain_on_saturday : Prop := sorry
def rain_on_sunday : Prop := sorry

theorem rain_prob_three_days : P (rain_on_friday ∧ rain_on_saturday ∧ rain_on_sunday) = 0.084 :=
by 
  have rain_friday_saturday := independence rain_on_friday rain_on_saturday,
  -- The probability for Friday and Saturday rain
  have part_one : P (rain_on_friday ∧ rain_on_saturday) = 0.4 * 0.7 :=
    by rw [P_A, P_B, rain_friday_saturday],
  -- The probability for all three days
  have rain_all_days := independence (rain_on_friday ∧ rain_on_saturday) rain_on_sunday,
  -- Combine the result with Sunday
  have part_two : P (rain_on_friday ∧ rain_on_saturday ∧ rain_on_sunday) = (P (rain_on_friday ∧ rain_on_saturday) * 0.3) :=
    by rw [update part_one P_C],
  -- Simplify the final result
  show 0.084 = ((0.4 * 0.7) * 0.3) by 
    rw [part_two, mul_assoc],
  sorry

end rain_prob_three_days_l51_51595


namespace some_number_is_105_l51_51448

def find_some_number (a : ℕ) (num : ℕ) : Prop :=
  a ^ 3 = 21 * 25 * num * 7

theorem some_number_is_105 (a : ℕ) (num : ℕ) (h : a = 105) (h_eq : find_some_number a num) : num = 105 :=
by
  sorry

end some_number_is_105_l51_51448


namespace hyperbola_correct_eq_l51_51810

-- Define the circle's equation modified from x^2 + y^2 - 4x + 3 = 0
def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

-- Define the ellipse's equation
def ellipse_eq (x y : ℝ) : Prop :=
  4 * x^2 + y^2 = 4

-- Define the foci of the ellipse
def is_focus (p : ℝ × ℝ) : Prop :=
  p = (0, real.sqrt 3) ∨ p = (0, -real.sqrt 3)

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 3 - x^2 / 9 = 1

-- State the main proof problem
theorem hyperbola_correct_eq : 
  (∀ (x y : ℝ), (∃ k : ℝ, y = k * x ∧ circle_eq x y) → circle_eq x y) → 
  (∀ (x y : ℝ), ellipse_eq x y → is_focus (x, y)) → 
  hyperbola_eq 0 (real.sqrt 3) ∧ 
  hyperbola_eq 0 (-real.sqrt 3) → 
  (∀ (x y : ℝ), hyperbola_eq x y) :=
by
  intros hyp_tangent hyp_foci hyp_foci_correct
  sorry

end hyperbola_correct_eq_l51_51810


namespace complex_polynomial_bound_l51_51384

noncomputable def polynomial (n : ℕ) (c : Fin n → ℂ) (z : ℂ) : ℂ :=
∑ i in Finset.range n, c i * z ^ (n - i)

theorem complex_polynomial_bound
  (n : ℕ)
  (c : Fin (n+1) → ℂ) :
  ∃ (z_0 : ℂ), |z_0| ≤ 1 ∧ |polynomial n c z_0| ≥ |c 0| + |c n| :=
sorry

end complex_polynomial_bound_l51_51384


namespace sum_of_abc_is_12_l51_51120

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end sum_of_abc_is_12_l51_51120


namespace find_n_l51_51180

-- Definitions and conditions based on the problem
variables {m n : ℝ}
def line (x y : ℝ) := x = m * y + n
def passes_through_A (x y : ℝ) := line x y = line 4 4
def circumcircle_radius : ℝ := 4
def n_positive := n > 0

-- Theorem statement that n is 4 given the conditions
theorem find_n :
  passes_through_A 4 4 ∧ circumcircle_radius = 4 ∧ n_positive →
  n = 4 :=
sorry

end find_n_l51_51180


namespace football_team_lineup_l51_51721

theorem football_team_lineup : 
  let total_players := 12 in
  let offensive_lineman_choices := 4 in
  let remaining_players_after_ol := total_players - offensive_lineman_choices in
  let quarterback_choices := remaining_players_after_ol in
  let running_back_choices := remaining_players_after_ol - 1 in
  let wide_receiver_choices := remaining_players_after_ol - 2 in
  let tight_end_choices := remaining_players_after_ol - 3 in
  4 * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 31680 :=
by
  let total_players := 12
  let offensive_lineman_choices := 4
  let remaining_players = total_players - offensive_lineman_choices
  let quarterback_choices = remaining_players 
  let running_back_choices = remaining_players - 1
  let wide_receiver_choices = remaining_players - 2
  let tight_end_choices = remaining_players - 3 
  show 4 * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 31680 from sorry

end football_team_lineup_l51_51721


namespace total_cost_l51_51321

-- Define the conditions
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount : ℕ := 10
def total_items (a b : ℕ) : ℕ := a + b
def qualifies_for_discount (n : ℕ) : Prop := n > 10

-- Define the parameters for the problem
def sandwiches_bought : ℕ := 7
def sodas_bought : ℕ := 6

def cost_before_discount (s : ℕ) (n : ℕ) : ℕ := (sandwich_cost * s) + (soda_cost * n)
def cost_after_discount (total_cost : ℕ) (qualifies : Prop) : ℕ := if qualifies then total_cost - discount else total_cost

-- Define the problem statement
theorem total_cost : cost_after_discount (cost_before_discount sandwiches_bought sodas_bought) (qualifies_for_discount (total_items sandwiches_bought sodas_bought)) = 36 := by sorry

end total_cost_l51_51321


namespace problem_part1_problem_part2_l51_51009

open Real

def vector_a (ω x : ℝ) : ℝ × ℝ := (sqrt 3 * cos (ω * x), cos (ω * x))
def vector_b (ω x : ℝ) : ℝ × ℝ := (sin (ω * x), -cos (ω * x))
def f (ω x : ℝ) : ℝ := (vector_a ω x).1 * (vector_b ω x).1 + (vector_a ω x).2 * (vector_b ω x).2
def G (m x : ℝ) : ℝ := m + 1 - sqrt 2 * f 1 (x / 2)

theorem problem_part1 :
  f 1 (π / 3) = 1 / 2 :=
sorry

theorem problem_part2 :
  (∀ x ∈ Icc 0 π, G m x = 0 → ∃! x' ∈ Icc 0 π, G m x' = 0) →
  m ∈ Icc (-1) ((sqrt 2 / 2) - 1) :=
sorry

end problem_part1_problem_part2_l51_51009


namespace infinite_matrices_with_square_zero_l51_51946

open Matrix

variable {R : Type*} [CommRing R]

theorem infinite_matrices_with_square_zero :
  ∃ (infinite : Set (Matrix (Fin 2) (Fin 2) R)),
  (∀ (B : Matrix (Fin 2) (Fin 2) R),
     B ^ 2 = 0 ↔ B ∈ infinite) ∧
  infinite.infinite :=
  sorry

end infinite_matrices_with_square_zero_l51_51946


namespace find_s_t_l51_51927

theorem find_s_t 
  (FG GH EH : ℝ)
  (angleE angleF : ℝ)
  (h1 : FG = 10)
  (h2 : GH = 15)
  (h3 : EH = 12)
  (h4 : angleE = 45)
  (h5 : angleF = 45)
  (s t : ℕ)
  (h6 : 12 + 7.5 * Real.sqrt 2 = s + Real.sqrt t) :
  s + t = 5637 :=
sorry

end find_s_t_l51_51927


namespace total_pages_in_book_l51_51132

theorem total_pages_in_book 
    (pages_read : ℕ) (pages_left : ℕ) 
    (h₁ : pages_read = 11) 
    (h₂ : pages_left = 6) : 
    pages_read + pages_left = 17 := 
by 
    sorry

end total_pages_in_book_l51_51132


namespace inequality_proof_l51_51837

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51837


namespace energy_loss_per_bounce_l51_51287

theorem energy_loss_per_bounce
  (h : ℝ) (t : ℝ) (g : ℝ) (y : ℝ)
  (h_conds : h = 0.2)
  (t_conds : t = 18)
  (g_conds : g = 10)
  (model : t = Real.sqrt (2 * h / g) + 2 * (Real.sqrt (2 * h * y / g)) / (1 - Real.sqrt y)) :
  1 - y = 0.36 :=
by
  sorry

end energy_loss_per_bounce_l51_51287


namespace exists_z_between_x_y_l51_51954

noncomputable def S := {z | ∃ (m n : ℕ), z = (m * n) / (m ^ 2 + n ^ 2)}

theorem exists_z_between_x_y (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) : 
  ∃ (z : ℝ), z ∈ S ∧ x < z ∧ z < y := 
sorry

end exists_z_between_x_y_l51_51954


namespace nine_b_value_l51_51023

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : a = b - 3) : 
  9 * b = 216 / 11 :=
by
  sorry

end nine_b_value_l51_51023


namespace last_digit_decimal_expansion_l51_51627

theorem last_digit_decimal_expansion :
  let x : ℚ := 1 / 3^15
  in (x.to_decimal.last_digit_unit = 5) :=
by
  sorry

end last_digit_decimal_expansion_l51_51627


namespace cylinder_height_l51_51584

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h_cond : SA = 2 * π * r^2 + 2 * π * r * h) 
  (r_eq : r = 3) (SA_eq : SA = 27 * π) : h = 3 / 2 :=
by
  sorry

end cylinder_height_l51_51584


namespace original_triangle_area_l51_51409

-- Define the problem conditions and assumptions
variables (a : ℝ)

-- Define the orthographic projection area
def area_projection := (√3 / 4) * a^2

-- Define the scale factor for the orthographic projection
def scale_factor := √2 / 4

-- Define the original area in terms of the projection area and scale factor
noncomputable def area_original := area_projection / scale_factor

-- State the theorem to be proved
theorem original_triangle_area : 
  area_original a = (√6 / 2) * a^2 := by
  sorry

end original_triangle_area_l51_51409


namespace CoreyCandies_l51_51998

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end CoreyCandies_l51_51998


namespace math_problem_part1_math_problem_part2_l51_51710

-- Definitions and conditions from part (a)
def system_H (α : Type) (H : α → Set ℤ) : Prop :=
  ∀ (α1 α2 : α), α1 ≠ α2 → (H α1) ∩ (H α2) ≠ {} ∧ Finite ((H α1) ∩ (H α2))
  
def distinct_system (α : Type) (H : α → Set ℤ) : Prop :=
  ∀ (α1 α2 : α), α1 ≠ α2 → (H α1) ≠ (H α2)

def system_with_cardinality_continuum (H : Type → Set ℤ) : Prop :=
  ∃ (α : Type) [Huge : Cardinality α = Cardinality.continuum], 
  system_H α H ∧ distinct_system α H

def system_with_intersection_lt_K (H : Type → Set ℤ) (K : ℕ) : Prop :=
  ∀ (α1 α2 : H), α1 ≠ α2 → (H α1) ∩ (H α2) ≠ {} ∧ 
  ∃ (s : Set ℤ), Finset.length s ≤ K ∧ s ∈ ((H α1) ∩ (H α2))

-- Problem Definition in Lean 4 Statement.
theorem math_problem_part1 : 
  ∃ (H : Type → Set ℤ), system_with_cardinality_continuum H := 
sorry

theorem math_problem_part2 (H : Type → Set ℤ) (K : ℕ) : 
  system_with_intersection_lt_K H K → ∃ (α : Type), Cardinality α ≠ Cardinality.continuum := 
sorry

end math_problem_part1_math_problem_part2_l51_51710


namespace decimal_to_fraction_l51_51346

theorem decimal_to_fraction : 0.4 -- meaning 0.4 repeating,
  + decimalFraction 0.03 == (71 / 165 : ℚ) := 
begin
  sorry
end

end decimal_to_fraction_l51_51346


namespace rhombus_area_l51_51179

theorem rhombus_area (d₁ d₂ : ℕ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 24 := 
by
  sorry

end rhombus_area_l51_51179


namespace closest_perfect_square_to_350_l51_51667

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l51_51667


namespace domain_of_f_2x_minus_2_l51_51032

noncomputable def f_domain_eq (x : ℝ) : Bool :=
  if x + 1 ∈ set.Icc 0 1 then True else False

theorem domain_of_f_2x_minus_2 :
  ∀ (x : ℝ),
  (f_domain_eq (2^x - 2) = True) ↔ (log 2 3 ≤ x ∧ x ≤ 2) := 
sorry

end domain_of_f_2x_minus_2_l51_51032


namespace marble_probability_l51_51367

-- Problem Statement
theorem marble_probability :
  let total_marbles := 9
  let chosen_marbles := 4
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3 in
  let total_ways := (total_marbles.choose chosen_marbles) in
  let red_ways := (red_marbles.choose 1) in
  let blue_ways := (blue_marbles.choose 1) in
  let green_ways := (green_marbles.choose 2) in
  (red_ways * blue_ways * green_ways) / total_ways = 3 / 14 := by
  sorry

end marble_probability_l51_51367


namespace greatest_area_difference_l51_51609

def first_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 156

def second_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 144

theorem greatest_area_difference : 
  ∃ (l1 w1 l2 w2 : ℕ), 
  first_rectangle_perimeter l1 w1 ∧ 
  second_rectangle_perimeter l2 w2 ∧ 
  (l1 * (78 - l1) - l2 * (72 - l2) = 225) := 
sorry

end greatest_area_difference_l51_51609


namespace first_three_decimal_digits_A1_same_decimal_digits_l51_51821

-- Define the sequences A_1 and A_n
def A1 := Real.sqrt (49 * 1^2 + 0.35 * 1)
def An (n : ℕ) := Real.sqrt (49 * n^2 + 0.35 * n)

-- Definitions of the first three decimal digits to be compared
def first_three_decimal_digits (x : ℝ) : ℕ := Nat.floor ((x - Real.floor x) * 1000)

theorem first_three_decimal_digits_A1 :
  first_three_decimal_digits A1 = 24 := by sorry

theorem same_decimal_digits (n : ℕ) (hn : 0 < n) :
  first_three_decimal_digits (An n) = first_three_decimal_digits A1 := by sorry

end first_three_decimal_digits_A1_same_decimal_digits_l51_51821


namespace total_students_in_high_school_l51_51215

theorem total_students_in_high_school (sample_size first_year third_year second_year : ℕ) (total_students : ℕ) 
  (h1 : sample_size = 45) 
  (h2 : first_year = 20) 
  (h3 : third_year = 10) 
  (h4 : second_year = 300)
  (h5 : sample_size = first_year + third_year + (sample_size - first_year - third_year)) :
  total_students = 900 :=
by
  sorry

end total_students_in_high_school_l51_51215


namespace decimal_to_binary_3_l51_51337

theorem decimal_to_binary_3 : Nat.toDigits 2 3 = [1, 1] := by {
  sorry
}

end decimal_to_binary_3_l51_51337


namespace infinite_solutions_l51_51016

theorem infinite_solutions (x : ℕ) :
  15 < 2 * x + 10 ↔ ∃ n : ℕ, x = n + 3 :=
by {
  sorry
}

end infinite_solutions_l51_51016


namespace find_radius_of_new_circle_l51_51212

variable (π : ℝ) (r1 r2 : ℝ)
noncomputable def shaded_area (r1 r2 : ℝ) : ℝ :=
  π * (r2^2 - r1^2)

theorem find_radius_of_new_circle (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 25) (h2 : r2 = 35) : 
  ∃ R3 > 0, π * R3^2 = shaded_area π r1 r2 ∧ R3 = 10 * real.sqrt 6 :=
by
  sorry

end find_radius_of_new_circle_l51_51212


namespace thirty_two_not_sum_consecutive_natural_l51_51559

theorem thirty_two_not_sum_consecutive_natural (n k : ℕ) : 
  (n > 0) → (32 ≠ (n * (2 * k + n - 1)) / 2) :=
by
  sorry

end thirty_two_not_sum_consecutive_natural_l51_51559


namespace tangent_secan_power_of_a_point_problems_l51_51504

variable (P O T A B : Type) 
variable [PA PB PT : ℝ]  -- PA, PB, and PT are real numbers

-- Conditions
variable (PA_eq : PA = 5)
variable (PT_eq : PT = (AB - 2 * PA))

-- Power of a point theorem
theorem tangent_secan_power_of_a_point_problems:
  PA * PB = PT * PT → 
  PA < PB →
  PB = 25 :=
by
  -- Proof goes here
  sorry

end tangent_secan_power_of_a_point_problems_l51_51504


namespace f_at_1_f_monotone_on_positive_reals_f_zeros_l51_51849

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then
    1 - (1 / x)
  else
    (a - 1) * x + 1

theorem f_at_1 (a : ℝ) : f a 1 = 0 :=
by {
  -- proof here 
  sorry
}

theorem f_monotone_on_positive_reals (a : ℝ) : 
  MonotoneOn (fun x => f a x) (Set.Ioi 0) :=
by {
  -- proof here 
  sorry
}

theorem f_zeros (a : ℝ) : 
  (a > 1 → (f a 1 = 0 ∧ f a (1 / (1 - a)) = 0)) ∧ 
  (a ≤ 1 → f a 1 = 0) :=
by {
  -- proof here 
  sorry
}

end f_at_1_f_monotone_on_positive_reals_f_zeros_l51_51849


namespace more_than_half_good_l51_51611

open Finset

def is_good_permutation {n : ℕ} (p : Perm (Fin (2 * n))) : Prop :=
  ∃ i : Fin (2 * n - 1), abs (p i).val - (p (i + 1)).val = n

noncomputable def count_good_permutations {n : ℕ} : Nat :=
  (univ.perm.card) / 2

theorem more_than_half_good {n : ℕ} (h : 0 < n) :
  ↑(count_good_permutations) < (2 * n)! :=
sorry

end more_than_half_good_l51_51611


namespace minimum_additional_squares_to_symmetry_l51_51921

def initially_shaded_squares : list (ℕ × ℕ) := [(1, 1), (1, 6), (6, 1), (3, 4)]

def has_symmetry (squares : list (ℕ × ℕ)) : Prop :=
  ∀ x y, (x, y) ∈ squares →
    ((7 - x, y) ∈ squares ∧ (x, 7 - y) ∈ squares ∧ (7 - x, 7 - y) ∈ squares)

theorem minimum_additional_squares_to_symmetry :
  ∃ additional_squares : list (ℕ × ℕ),
    has_symmetry (initially_shaded_squares ++ additional_squares)
    ∧ additional_squares.length = 4 :=
sorry

end minimum_additional_squares_to_symmetry_l51_51921


namespace total_population_l51_51460

-- Definitions based on given conditions
variables (b g t : ℕ)
variables (h1 : b = 4 * g) (h2 : g = 8 * t)

-- Theorem statement
theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * t :=
by
  sorry

end total_population_l51_51460


namespace compare_sums_l51_51683

noncomputable def sum1 : ℝ := ∑ i in (Set.Icc 50 150).toFinset, 1 / i
noncomputable def sum2 : ℝ := ∑ i in (Set.Icc 150 250).toFinset, 2 / i

theorem compare_sums : sum1 > sum2 := 
by
  sorry 

end compare_sums_l51_51683


namespace three_pow_2023_mod_17_l51_51253

theorem three_pow_2023_mod_17 : (3 ^ 2023) % 17 = 7 := by
  sorry

end three_pow_2023_mod_17_l51_51253


namespace calc_expression_l51_51327

theorem calc_expression (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end calc_expression_l51_51327


namespace initial_number_of_persons_l51_51578

-- Definitions based on conditions
def average_weight_increase (N : ℕ) : Prop := 10 * N + 40 = 90

theorem initial_number_of_persons :
  ∃ N : ℕ, average_weight_increase N ∧ N = 5 :=
begin
  have h : average_weight_increase 5,
  {
    unfold average_weight_increase,
    norm_num,
  },
  use 5,
  split,
  exact h,
  refl,
end

end initial_number_of_persons_l51_51578


namespace first_nonzero_digit_right_decimal_l51_51230

/--
  To prove that the first nonzero digit to the right of the decimal point of the fraction 1/137 is 9
-/
theorem first_nonzero_digit_right_decimal (n : ℕ) (h1 : n = 137) :
  ∃ d, d = 9 ∧ (∀ k, 10 ^ k * 1 / 137 < 10^(k+1)) → the_first_nonzero_digit_right_of_decimal_is 9 := 
sorry

end first_nonzero_digit_right_decimal_l51_51230


namespace closest_perfect_square_to_350_l51_51655

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l51_51655


namespace positive_integer_solution_l51_51594

theorem positive_integer_solution (n : ℕ) (H : 1 + 3 + 5 + ... + (2 * n - 1) = n^2) (H2 : 2 + 4 + 6 + ... + 2 * n = n * (n + 1)) : n = 115 :=
by
  sorry

end positive_integer_solution_l51_51594


namespace relationship_a_b_l51_51302

theorem relationship_a_b (a b : ℝ) :
  (∃ (P : ℝ × ℝ), P ∈ {Q : ℝ × ℝ | Q.snd = -3 * Q.fst + b} ∧
                   ∃ (R : ℝ × ℝ), R ∈ {S : ℝ × ℝ | S.snd = -a * S.fst + 3} ∧
                   R = (-P.snd, -P.fst)) →
  a = 1 / 3 ∧ b = -9 :=
by
  intro h
  sorry

end relationship_a_b_l51_51302


namespace smallest_k_round_table_l51_51469

variable {n : ℕ}
variable (a : ℕ → ℕ)
variable (n_pos : n > 2)
variable (sum_a_squared : ∑ i in Finset.range n, (a i) ^ 2 = n^2 - n)

theorem smallest_k_round_table (h : ∀ i j, i ≠ j → ¬ (a i = n - 1) ∧ ∃ k, k ≠ i ∧ k ≠ j ∧ (a k > 0)) :
  5 = 5 :=
begin
  sorry,
end

end smallest_k_round_table_l51_51469


namespace find_r_neg_five_l51_51736

noncomputable def p (x : ℝ) : ℝ := sorry -- Assume p(x) meets the given conditions

def r (x : ℝ) : ℝ :=
  let a : ℝ := -2 / 27
  let b : ℝ := -37 / 27
  let c : ℝ := 19 / 27
  a * x^2 + b * x + c

theorem find_r_neg_five : r (-5) = 154 / 27 := 
by sorry

end find_r_neg_five_l51_51736


namespace find_a2_find_bn_find_an_l51_51871

open Nat

noncomputable def a₁ : ℕ := 2

def S (n : ℕ) : Real := ∑ i in range n, (1 : Real) -- This is a placeholder, correct definition not given

axiom cond_1 (n : ℕ) (hn : 0 < n) : (1 / (a n) - 1 / (a (n + 1))) = 2 / (4 * S n - 1)

-- Statement 1: Prove a₂ = 14/3 given the conditions
theorem find_a2 : a 2 = 14 / 3 := by
  sorry

-- Definitions for sequences aₓ and bₓ
noncomputable def a : ℕ → ℝ 
| 1 := a₁
| n + 1 := sorry -- Placeholder, needs proper recurrence definition

noncomputable def b (n : ℕ) : ℝ := (a n) / ((a (n + 1)) - (a n))

-- Statement 2: Prove the general term of bₙ
theorem find_bn (n : ℕ) (hn : 0 < n) : b n = (4 * n - 1) / 4 := by
  sorry

-- Statement 3: Prove the general term of aₙ
theorem find_an (n : ℕ) (hn : 0 < n) : a n = (8 * n - 2) / 3 := by
  sorry

end find_a2_find_bn_find_an_l51_51871


namespace total_seashells_l51_51541

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end total_seashells_l51_51541


namespace even_degree_graph_has_cycle_l51_51605

-- Definition of a graph structure
structure Graph (V : Type) :=
  (edges : V → V → Prop)
  (symm : symmetric edges)  -- This ensures that if an edge exists from u to v, it also exists from v to u.

-- Definition of even degree for vertices in a graph
def even_degree {V : Type} (G : Graph V) (v : V) :=
  ∃ k : ℕ, count (G.edges v) = 2 * k

-- The theorem we need to prove
theorem even_degree_graph_has_cycle {V : Type} (G : Graph V) :
  (∀ v : V, even_degree G v) → ∃ cycle : set V, is_cycle G cycle := sorry

end even_degree_graph_has_cycle_l51_51605


namespace repair_cost_l51_51148

theorem repair_cost (purchase_price transport_charges selling_price profit_percentage R : ℝ)
  (h1 : purchase_price = 10000)
  (h2 : transport_charges = 1000)
  (h3 : selling_price = 24000)
  (h4 : profit_percentage = 0.5)
  (h5 : selling_price = (1 + profit_percentage) * (purchase_price + R + transport_charges)) :
  R = 5000 :=
by
  sorry

end repair_cost_l51_51148


namespace roots_of_polynomial_l51_51803

noncomputable def polynomial : Polynomial ℝ := Polynomial.mk [6, -11, 6, -1]

theorem roots_of_polynomial :
  (∃ r1 r2 r3 : ℝ, polynomial = (X - C r1) * (X - C r2) * (X - C r3) ∧ {r1, r2, r3} = {1, 2, 3}) :=
sorry

end roots_of_polynomial_l51_51803


namespace max_BM2_minus_AM1_l51_51472

-- Conditions
variables (ω : Type) [metric_space ω]
variables (P M1 M2 A B : ω)
variable [inhabited ω]

-- Given distances
constants (PM1 PM2 : ℝ)
axiom hPM1 : PM1 = 15
axiom hPM2 : PM2 = 20

-- Orthogonality condition of the chords
axiom perpendicular_chords : P ≠ M1 → P ≠ M2 → ∃ O : ω, M1 ≠ P ∧ M2 ≠ P ∧ O ≠ P ∧ O ≠ M1 ∧ O ≠ M2

-- Line intersections
axiom line_intersections : ∃ L : ω → ω, L(M1) ≠ L(M2) ∧ L(M1) = A ∧ L(M2) = B

-- Main statement to prove
theorem max_BM2_minus_AM1 : BM2 - AM1 = 7 := sorry

end max_BM2_minus_AM1_l51_51472


namespace triangle_area_proof_l51_51795

theorem triangle_area_proof :
  let radius_of_circle := 3
  let side_of_equilateral_triangle := 3 * Real.sqrt 3
  let AD := 15
  let AE := 13
  let parallelogram_base := side_of_equilateral_triangle
  let area_EAF := (1 / 2) * AE * AD * (Real.sqrt 3 / 2) -- Calculating the area of EAF

  let BC := side_of_equilateral_triangle
  let AF := Real.sqrt (AD^2 + AE^2) -- Using Pythagorean theorem
  let area_GBC := (BC^2 / AF^2) * area_EAF

  let p := 5265
  let q := 3
  let r := 1576
  let expected_area := (p * Real.sqrt q) / r
  let sum_pqr := p + q + r
in
area_GBC = expected_area ∧ sum_pqr = 6844 :=
by
  unfold radius_of_circle
  unfold side_of_equilateral_triangle
  unfold AD
  unfold AE
  unfold parallelogram_base
  unfold area_EAF
  unfold BC
  unfold AF
  unfold area_GBC
  unfold p
  unfold q
  unfold r
  unfold expected_area
  unfold sum_pqr
  sorry

end triangle_area_proof_l51_51795


namespace euler_lines_concurrent_or_parallel_l51_51112

noncomputable theory

variables {P A B C : Type*} [metric_space P] [metric_space A] [metric_space B] [metric_space C]
[normed_add_comm_group P] [normed_space ℝ P]
[normed_add_comm_group A] [normed_space ℝ A]
[normed_add_comm_group B] [normed_space ℝ B]
[normed_add_comm_group C] [normed_space ℝ C]

def equilateral_triangle (A B C : P) : Prop :=
(dist A B = dist B C ∧ dist B C = dist C A)

theorem euler_lines_concurrent_or_parallel (A B C P: P) (h_eq: equilateral_triangle A B C) :
  ∃ (L₁ L₂ L₃ : set (affine_space ℝ P)), 
  (L₁ = euler_line (triangle.mk P A B) ∧ 
   L₂ = euler_line (triangle.mk P B C) ∧ 
   L₃ = euler_line (triangle.mk P C A)) ∧ 
  (concurrent L₁ L₂ L₃ ∨ parallel L₁ L₂ L₃) :=
sorry

end euler_lines_concurrent_or_parallel_l51_51112


namespace part_a_part_b_l51_51459

-- Define variables for the elements in the 3x3 matrix and the magic constant S
variables {a b c d e f g h i S : ℤ}

-- Define the hypothesis for the magic square condition
def is_magic_square (a b c d e f g h i S : ℤ) : Prop :=
  a + b + c = S ∧ d + e + f = S ∧ g + h + i = S ∧
  a + d + g = S ∧ b + e + h = S ∧ c + f + i = S ∧
  a + e + i = S ∧ c + e + g = S

-- Statement of theorem (a)
theorem part_a (h : is_magic_square a b c d e f g h i S) :
  2 * (a + c + g + i) = b + d + f + h + 4 * e :=
begin
  sorry
end

-- Statement of theorem (b)
theorem part_b (h : is_magic_square a b c d e f g h i S) :
  2 * (a^3 + c^3 + g^3 + i^3) = b^3 + d^3 + f^3 + h^3 + 4 * e^3 :=
begin
  sorry
end

end part_a_part_b_l51_51459


namespace sum_difference_l51_51488

def digit_replacement (n : ℕ) : ℕ :=
  let s := n.toString in
  let s' := s.map (fun ch => if ch = '3' then '2' else ch) in
  s'.toNat

def Jane_sum : ℕ :=
  (List.range' 1 50).sum

def Liam_sum : ℕ :=
  (List.range' 1 50).map digit_replacement |>.sum

theorem sum_difference : Jane_sum - Liam_sum = 105 :=
  sorry

end sum_difference_l51_51488


namespace divisors_product_exceeds_16_digits_l51_51098

theorem divisors_product_exceeds_16_digits :
  let n := 1024
  let screen_digits := 16
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  (String.length (Natural.toString divisors_product) > screen_digits) := 
by {
  let n := 1024
  let screen_digits := 16
  let divisors := List.range (11)
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  show (String.length (Natural.toString divisors_product) > screen_digits),
  sorry
}

end divisors_product_exceeds_16_digits_l51_51098


namespace find_d_l51_51963

variable (c d v : ℝ → ℝ → ℝ → ℝ) 
variable (a b e : ℝ)

def is_parallel (v: ℝ × ℝ × ℝ) (w: ℝ × ℝ × ℝ) : Prop :=
  ∃ (k: ℝ), v = k • w

def is_orthogonal (v: ℝ × ℝ × ℝ) (w: ℝ × ℝ × ℝ) : Prop :=
  (v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0)

theorem find_d :
  ∃ d : ℝ × ℝ × ℝ,
    let M : ℝ × ℝ × ℝ := (8, -4, -8) in
    let R : ℝ × ℝ × ℝ := (1, 1, 1) in
    ∃ c : ℝ × ℝ × ℝ, 
      (c.1 + d.1 = M.1 ∧ c.2 + d.2 = M.2 ∧ c.3 + d.3 = M.3)
      ∧ is_parallel c R
      ∧ is_orthogonal d R
      ∧ d = (28 / 3, -8 / 3, -20 / 3) := by
  sorry

end find_d_l51_51963


namespace complement_U_A_correct_l51_51430

-- Define the universal set U and set A
def U : Set Int := {-1, 0, 2}
def A : Set Int := {-1, 0}

-- Define the complement of A in U
def complement_U_A : Set Int := {x | x ∈ U ∧ x ∉ A}

-- Theorem stating the required proof
theorem complement_U_A_correct : complement_U_A = {2} :=
by
  sorry -- Proof will be filled in

end complement_U_A_correct_l51_51430


namespace value_of_b_l51_51641

theorem value_of_b (b : ℝ) : 
  (∀ x : ℝ, -x ^ 2 + b * x + 7 < 0 ↔ x < -2 ∨ x > 3) → b = 1 :=
by
  sorry

end value_of_b_l51_51641


namespace function_value_f_3_l51_51880

def f : ℕ → ℕ
| x := if x < 2 then 2^x else f (x-1) + 1

theorem function_value_f_3 : f 3 = 4 :=
by
  sorry

end function_value_f_3_l51_51880


namespace term_count_in_expansion_l51_51020

theorem term_count_in_expansion :
  let a := 1 
  let b := 1 
  let c := 1 
  let d := 1 
  let e := 1 
  let f := 1 
  let g := 1 
  let h := 1 
  let i := 1 
  (a + b + c + d + e + f + g + h + i) ^ 2 = 81 → 45 = ∑ k in Ico 0 9, 1=True := sorry

end term_count_in_expansion_l51_51020


namespace true_proposition_l51_51142

variable (α β : ℝ)   -- α and β are real numbers
variable (a b : ℝ)   -- a and b are real numbers representing vectors' coordinates

def p := (a * b < 0) → (acos (a / sqrt (a^2 + b^2)) = π / 2)
def q := (cos α * cos β = 1) → (sin (α + β) = 0)

theorem true_proposition : (p α β a b ∨ q α β) :=
by
  -- because p is false
  have h1 : ¬p α β a b := sorry
  -- and q is true
  have h2 : q α β := sorry
  -- thus p ∨ q is true
  exact Or.inr h2

end true_proposition_l51_51142


namespace candidate_vote_percentage_l51_51716

theorem candidate_vote_percentage
    (lost_by_votes : ℕ)
    (total_votes : ℕ)
    (hc1 : lost_by_votes = 2370)
    (hc2 : total_votes = 7900) :
  ∃ P : ℚ, P = 35 ∧ (2 * (P / 100) * total_votes + 2370 = 7900) :=
by
  use 35
  split
  { refl }
  { sorry }

end candidate_vote_percentage_l51_51716


namespace sum_of_ages_is_50_l51_51189

def youngest_child_age : ℕ := 4

def age_intervals : ℕ := 3

def ages_sum (n : ℕ) : ℕ :=
  youngest_child_age + (youngest_child_age + age_intervals) +
  (youngest_child_age + 2 * age_intervals) +
  (youngest_child_age + 3 * age_intervals) +
  (youngest_child_age + 4 * age_intervals)

theorem sum_of_ages_is_50 : ages_sum 5 = 50 :=
by
  sorry

end sum_of_ages_is_50_l51_51189


namespace part_a_part_b_l51_51709

-- Definitions for the conditions from (a) and (b)
structure RationalNum where
  val : ℚ 
  is_between_0_and_1 : 0 < val ∧ val < 1

def periodicDecimalRepresentation (r : RationalNum) : Prop :=
  ∃ p : List ℕ, (0 ≤ p.all? id ∧ p.size > 0) → r.val = convertToFraction p

-- Statement for part (a)
theorem part_a (p q : RationalNum) (hp : periodicDecimalRepresentation p) (hq : periodicDecimalRepresentation q) :
  ∃ α : ℝ, irrational α ∧ ∀ i : ℕ, (digitOf α i) = (digitOf p.val i) ∨ (digitOf α i) = (digitOf q.val i) :=
  sorry

-- Statement for part (b)
theorem part_b (s : RationalNum) (hs : periodicDecimalRepresentation s) :
  ∃ β : ℝ, irrational β ∧ ∀ N ≥ 2017, (countDiffs s.val β N) ≤ N / 2017 :=
  sorry

end part_a_part_b_l51_51709


namespace YZ_length_l51_51483

noncomputable def sum_of_possible_values_YZ (XY XZ : ℝ) (angle_Y : ℝ) :=
  let YZ1 := YZ_value XY XZ angle_Y 30 105
  let YZ2 := 0 -- assuming no other valid triangles
  YZ1 + YZ2

theorem YZ_length {XY XZ : ℝ} (angle_Y : ℝ) :
  XY = 100 → XZ = 100 * Real.sqrt 2 → angle_Y = 45 → sum_of_possible_values_YZ XY XZ angle_Y = 200 :=
by
  intros hXY hXZ hY
  -- Further proof steps would be filled here
  sorry

end YZ_length_l51_51483


namespace susan_more_cats_than_bob_l51_51994

-- Given problem: Initial and transaction conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def susan_additional_cats : ℕ := 5
def bob_additional_cats : ℕ := 7
def susan_gives_bob_cats : ℕ := 4

-- Declaration to find the difference between Susan's and Bob's cats
def final_susan_cats (initial : ℕ) (additional : ℕ) (given : ℕ) : ℕ := initial + additional - given
def final_bob_cats (initial : ℕ) (additional : ℕ) (received : ℕ) : ℕ := initial + additional + received

-- The proof statement which we need to show
theorem susan_more_cats_than_bob : 
  final_susan_cats susan_initial_cats susan_additional_cats susan_gives_bob_cats - 
  final_bob_cats bob_initial_cats bob_additional_cats susan_gives_bob_cats = 8 := by
  sorry

end susan_more_cats_than_bob_l51_51994


namespace stick_horisontal_fall_position_l51_51018

-- Definitions based on the conditions
def stick_length : ℝ := 120 -- length of the stick in cm
def projection_distance : ℝ := 70 -- distance between projections of the ends of the stick on the floor

-- The main theorem to prove
theorem stick_horisontal_fall_position :
  ∀ (L d : ℝ), L = stick_length ∧ d = projection_distance → 
  ∃ x : ℝ, x = 25 :=
by
  intros L d h
  have h1 : L = stick_length := h.1
  have h2 : d = projection_distance := h.2
  -- The detailed proof steps will be here
  sorry

end stick_horisontal_fall_position_l51_51018


namespace red_chip_count_l51_51603

def num_chips : ℕ := 60
def fraction_blue : ℚ := 1/6
def num_green_chips : ℕ := 16

def num_blue_chips : ℕ := (num_chips * fraction_blue).toNat
def num_red_chips : ℕ := num_chips - num_blue_chips - num_green_chips

theorem red_chip_count :
  num_red_chips = 34 := by
  sorry

end red_chip_count_l51_51603


namespace smallest_number_conditions_l51_51638

theorem smallest_number_conditions :
  ∃ (n : ℕ), n = 4091 ∧ ¬ Prime n ∧ ∀ p : ℕ, Prime p → p ∣ n → 60 < p ∧ ¬ ∃ m : ℕ, m * m = n :=
begin
  sorry
end

end smallest_number_conditions_l51_51638


namespace circle_tangent_secant_problem_l51_51505

theorem circle_tangent_secant_problem 
  (O : Type) [circle O] (P T A B : O)
  (hPoutside: ¬P ∈ O) 
  (h_tangent: tangent_segment P T O)
  (h_secant: secant_segment P A B O)
  (h_dist_PA_PB: distance P A < distance P B)
  (PA_eq_5: distance P A = 5)
  (PT_eq: distance P T = distance A B - 2 * distance P A) :
  distance P B = 20 :=
by
  sorry

end circle_tangent_secant_problem_l51_51505


namespace tan_P_tan_R_l51_51064

noncomputable theory

-- Definitions and conditions
def triangle (P Q R: Type _) := 
  -- Suppose this represents the vertices of the triangle PQR
  true

def orthocenter_divides_altitude (H T Q: Type _) (HT HQ: ℝ) :=
  -- Condition that the orthocenter divides the altitude with given lengths
  HT = 8 ∧ HQ = 24

-- Main statement to prove
theorem tan_P_tan_R {P Q R H T: Type _} [inhabited P] [inhabited Q] [inhabited R]
  (HT HQ: ℝ) [orthocenter_divides_altitude H T Q HT HQ] :
  ∃ (P Q R: Type _), orthocenter_divides_altitude H T Q HT HQ → tan_P_tan_R = 4 :=
by
  sorry

end tan_P_tan_R_l51_51064


namespace rick_division_steps_l51_51986

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l51_51986


namespace closest_perfect_square_to_350_l51_51668

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l51_51668


namespace inequality_proof_l51_51829

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51829


namespace solve_for_y_l51_51991

theorem solve_for_y (y : ℕ) : 8^4 = 2^y → y = 12 :=
by
  sorry

end solve_for_y_l51_51991


namespace required_hours_per_week_l51_51315

noncomputable def hourly_wage_summer (total_earnings : ℝ) (weeks : ℝ) (hours_per_week : ℝ) : ℝ :=
  total_earnings / (weeks * hours_per_week)

noncomputable def new_hourly_wage (hourly_wage : ℝ) (reduction_factor : ℝ) : ℝ :=
  hourly_wage * reduction_factor

noncomputable def hours_per_week (required_earnings : ℝ) (weeks : ℝ) (new_hourly_wage : ℝ) : ℝ :=
  required_earnings / (weeks * new_hourly_wage)

theorem required_hours_per_week
  (summer_earnings : ℝ) (summer_weeks : ℝ) (summer_hours_per_week : ℝ)
  (reduction_factor : ℝ) (spring_weeks : ℝ) (spring_earnings : ℝ) :
  hours_per_week spring_earnings spring_weeks
    (new_hourly_wage (hourly_wage_summer summer_earnings summer_weeks summer_hours_per_week) reduction_factor) ≈ 34 :=
by
  -- Condition definitions
  let summer_earnings := 2250
  let summer_weeks := 15
  let summer_hours_per_week := 36
  let reduction_factor := 0.8
  let spring_weeks := 24
  let spring_earnings := 2700

  sorry

end required_hours_per_week_l51_51315


namespace cubes_passed_through_diagonal_l51_51712

theorem cubes_passed_through_diagonal (a b c : ℕ) : 
  a = 200 → b = 325 → c = 376 →
  let g_ab := Int.gcd 200 325;
  let g_bc := Int.gcd 325 376;
  let g_ca := Int.gcd 376 200;
  let g_abc := Int.gcd (Int.gcd 200 325) 376;
  200 + 325 + 376 - (g_ab + g_bc + g_ca) + g_abc = 868 :=
by
  intros h1 h2 h3;
  rw [h1, h2, h3];
  simp [Int.gcd];
  sorry

end cubes_passed_through_diagonal_l51_51712


namespace g_200_eq_zero_l51_51284

-- Define the function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- Given conditions
axiom g_prop : ∀ (x y : ℝ), 0 < x → 0 < y → x * g(y) - y * g(x) = g(x / y) + g(x + y)

-- Goal: Prove that g(200) = 0
theorem g_200_eq_zero : g 200 = 0 := by
  -- placeholder for the proof
  sorry

end g_200_eq_zero_l51_51284


namespace find_t_u_l51_51890

variable {V : Type*} [InnerProductSpace ℝ V]

-- Definitions for the vectors a, b, and p
variables (a b p : V)

-- Main theorem statement
theorem find_t_u (h : ∥p - b∥ = 3 * ∥p - a∥) :
  ∃ (t u : ℝ), ∀ p, ∥p - (t • a + u • b)∥ = ∥p - (t • a + u • b)∥ ∧ (t, u) = (9 / 8, -1 / 8) :=
sorry

end find_t_u_l51_51890


namespace problem_n_times_s_l51_51945

def S : Set ℝ := {x | x ≠ 0}

def f (x : ℝ) : ℝ := sorry

axiom functional_eq (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x + y ≠ 0) :
  f(x) + f(y) = f(x * y * f(x + y))

theorem problem_n_times_s :
  ∃ n s : ℝ, 
  (∀ t, (t ≠ (1 / 4)) → f(4) = t → False) ∧
  (∀ t, f(4) = t → t = (1 / 4)) ∧
  n = 1 ∧ 
  s = (1 / 4) ∧ 
  n * s = (1 / 4) :=
begin
  sorry
end

end problem_n_times_s_l51_51945


namespace exists_line_no_lattice_point_l51_51933

theorem exists_line_no_lattice_point : ∃ (m c : ℝ), ∀ (a b : ℤ), (b: ℝ) ≠ m * (a: ℝ) + c :=
by
  let m := 1
  let c := 1 / 2
  use m, c
  intros a b h
  have : (b: ℝ) = a + 1 / 2, from h
  have h1 : (b: ℝ) - a = 1 / 2, by simp [this]
  have h2 : (b - a : ℝ) = 1 / 2, from h1
  have h3 : ((b - a : ℤ) : ℝ) = (1 / 2 : ℝ), by simp [h2]
  norm_cast at h3
  exact absurd h3 (by norm_num)

end exists_line_no_lattice_point_l51_51933


namespace eighth_monomial_l51_51742

theorem eighth_monomial (a : ℤ) : 
  let seq (n : ℕ) := (-1) ^ n * n^2 * a^(n+1) in 
  seq 8 = 64 * a^9 :=
by 
  sorry

end eighth_monomial_l51_51742


namespace sum_of_first_n_terms_sequence_l51_51889

def sequence (a : ℕ) : ℕ → ℕ
| 1     := a
| (n+1) := 2 * sequence (n+1) (n+2) - n + 1

def b (a : ℕ) (n : ℕ) : ℕ := sequence a n - n

noncomputable def S_n (a : ℕ) (n : ℕ) : ℕ :=
(∑ k in range n, sequence a k)

theorem sum_of_first_n_terms_sequence (n : ℕ) :
  S_n 2 n = 2^n - 1 + n * (n + 1) / 2 := sorry

end sum_of_first_n_terms_sequence_l51_51889


namespace polar_equations_and_angle_cosine_l51_51050

-- Define the Cartesian equations for line l and circle C
def parametric_line_eq (t : ℝ) : ℝ × ℝ :=
  (-1 + (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Define the polar equations derived from the conditions and prove the results
theorem polar_equations_and_angle_cosine :
  (∀ t, let (x, y) := parametric_line_eq t in 
    let ρ := Real.sqrt (x^2 + y^2) in
    let θ := Real.atan2 y x in
      ρ * Real.sin θ = ρ * Real.cos θ + 2 ∧
      ρ = 4 * Real.cos θ + 2 * Real.sin θ) ∧
  (∃ θ₁ θ₂, 
    θ₁ = Real.pi / 2 ∧
    θ₂ = Real.atan (3 : ℝ) ∧
    Real.cos (θ₁ - θ₂) = 3 * Real.sqrt 10 / 10) :=
sorry

end polar_equations_and_angle_cosine_l51_51050


namespace maximize_sum_l51_51057

def a_n (n : ℕ): ℤ := 11 - 2 * (n - 1)

theorem maximize_sum (n : ℕ) (S : ℕ → ℤ → Prop) :
  (∀ n, S n (a_n n)) → (a_n n ≥ 0) → n = 6 :=
by
  sorry

end maximize_sum_l51_51057


namespace tan_double_angle_given_tan_l51_51372

theorem tan_double_angle_given_tan : ∀ x : ℝ, tan x = 3 → tan (2 * x) = -3 / 4 :=
by
  intro x hx
  sorry

end tan_double_angle_given_tan_l51_51372


namespace original_number_count_l51_51988

theorem original_number_count (k S : ℕ) (M : ℚ)
  (hk : k > 0)
  (hM : M = S / k)
  (h_add15 : (S + 15) / (k + 1) = M + 2)
  (h_add1 : (S + 16) / (k + 2) = M + 1) :
  k = 6 :=
by
  -- Proof will go here
  sorry

end original_number_count_l51_51988


namespace convert_210_deg_to_rad_l51_51788

-- Definition of degree to radian conversion
def deg_to_rad (θ_deg : ℕ) : ℝ := θ_deg * (Real.pi / 180)

-- Main statement of the problem
theorem convert_210_deg_to_rad : deg_to_rad 210 = (7 * Real.pi) / 6 :=
  by
    -- placeholder for the actual proof
    sorry

end convert_210_deg_to_rad_l51_51788


namespace common_divisors_count_l51_51898

-- Definitions of the numbers involved
def n1 : ℕ := 9240
def n2 : ℕ := 10800

-- Prime factorizations based on the conditions
def factor_n1 : Prop := n1 = 2^3 * 3^1 * 5^1 * 7 * 11
def factor_n2 : Prop := n2 = 2^3 * 3^3 * 5^2

-- GCD as defined in the conditions
def gcd_value : ℕ := Nat.gcd n1 n2

-- Proof problem: prove the number of positive divisors of the gcd of n1 and n2 is 16
theorem common_divisors_count (h1 : factor_n1) (h2 : factor_n2) : Nat.divisors (Nat.gcd n1 n2).card = 16 := by
  sorry

end common_divisors_count_l51_51898


namespace jaclyns_polynomial_constant_term_l51_51964

def monic (p : Polynomial ℤ) : Prop := p.leadingCoeff = 1

theorem jaclyns_polynomial_constant_term :
  ∃ p q : Polynomial ℤ,
    monic p ∧ monic q ∧
    degree p = 3 ∧ degree q = 3 ∧
    (∃ a0 : ℤ, 0 < a0 ∧ p.coeff 0 = a0 ∧ q.coeff 0 = a0) ∧
    p.coeff 1 = q.coeff 1 ∧
    p * q = Polynomial.Coeff x^6 + 2*x^5 + 5*x^4 + 8*x^3 + 5*x^2 + 2*x + 9 →
    (∃ a0, a0 = 3) :=
by
  sorry

end jaclyns_polynomial_constant_term_l51_51964


namespace monotone_increasing_interval_l51_51589

-- Define the function f(x) = ln(x) - x
def f (x : ℝ) : ℝ := Real.log x - x

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 1 / x - 1

theorem monotone_increasing_interval :
  ∀ x, (0 < x ∧ x < 1) ↔ f' x > 0 := by
  sorry

end monotone_increasing_interval_l51_51589


namespace not_all_sets_of_10_segments_form_triangle_l51_51066

theorem not_all_sets_of_10_segments_form_triangle :
  ¬ ∀ (segments : Fin 10 → ℝ), ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (segments a + segments b > segments c) ∧
    (segments a + segments c > segments b) ∧
    (segments b + segments c > segments a) :=
by
  sorry

end not_all_sets_of_10_segments_form_triangle_l51_51066


namespace boat_speed_in_still_water_l51_51713

def downstream_distance : ℝ := 10
def downstream_time : ℝ := 3
def upstream_distance : ℝ := 10
def upstream_time : ℝ := 6

def downstream_speed := downstream_distance / downstream_time
def upstream_speed := upstream_distance / upstream_time
def speed_in_still_water := (downstream_speed + upstream_speed) / 2

theorem boat_speed_in_still_water :
  speed_in_still_water = 2.5 :=
sorry

end boat_speed_in_still_water_l51_51713


namespace volume_of_extended_parallelepiped_l51_51787

theorem volume_of_extended_parallelepiped 
  (a b c : ℕ) (ha : a = 2) (hb : b = 6) (hc : c = 7) :
  let m := 660
  let n := 49
  let p := 3
  let V := (m + n * Real.pi) / p
  V = 220 + (49 * Real.pi / 3) → m + n + p = 712 := 
by {
  intros,
  sorry
}

end volume_of_extended_parallelepiped_l51_51787


namespace find_tan_theta_l51_51440

theorem find_tan_theta
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h2 : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 :=
sorry

end find_tan_theta_l51_51440


namespace num_of_diff_is_six_l51_51434

-- Define the set of numbers
def numSet : Set ℕ := {1, 3, 4, 7, 10}

-- Define the set of differences
def diffSet : Set ℕ := {x - y | x ∈ numSet, y ∈ numSet, x ≠ y}

-- The main theorem to prove
theorem num_of_diff_is_six : diffSet.erase 0 = {1, 3, 4, 6, 7, 9} ↔ diffSet.erase 0.card = 6 := by
  sorry

end num_of_diff_is_six_l51_51434


namespace right_triangle_side_length_l51_51205

theorem right_triangle_side_length (side1 side2 : ℝ) (h1 : side1 = 4) (h2 : side2 = 5) :
  (∃ side3 : ℝ, (side3 = √41 ∨ side3 = 3) ∧ 
    (side3^2 + side1^2 = side2^2 ∨ side3^2 + side2^2 = side1^2 ∨ side1^2 + side2^2 = side3^2)) :=
by
  sorry

end right_triangle_side_length_l51_51205


namespace no_such_quadruples_exist_l51_51812

def matrix_inverse_condition (a b c d : ℝ) : Prop :=
  matrix.inv ![![a, b], ![c, d]] = ![![1/d, 1/c], ![1/b, 1/a]]

theorem no_such_quadruples_exist :
  ¬ ∃ (a b c d : ℝ), matrix_inverse_condition a b c d :=
by
  intro h
  sorry

end no_such_quadruples_exist_l51_51812


namespace valid_votes_other_candidate_l51_51696

theorem valid_votes_other_candidate (total_votes : ℕ)
  (invalid_percentage valid_percentage candidate1_percentage candidate2_percentage : ℕ)
  (h_invalid_valid_sum : invalid_percentage + valid_percentage = 100)
  (h_candidates_sum : candidate1_percentage + candidate2_percentage = 100)
  (h_invalid_percentage : invalid_percentage = 20)
  (h_candidate1_percentage : candidate1_percentage = 55)
  (h_total_votes : total_votes = 7500)
  (h_valid_percentage_eq : valid_percentage = 100 - invalid_percentage)
  (h_candidate2_percentage_eq : candidate2_percentage = 100 - candidate1_percentage) :
  ( ( candidate2_percentage * ( valid_percentage * total_votes / 100) ) / 100 ) = 2700 :=
  sorry

end valid_votes_other_candidate_l51_51696


namespace olivia_pays_in_dollars_l51_51547

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l51_51547


namespace elsa_cookie_baking_time_l51_51364

theorem elsa_cookie_baking_time :
  ∀ (baking_time white_icing_time chocolate_icing_time total_time : ℕ), 
  baking_time = 15 →
  white_icing_time = 30 →
  chocolate_icing_time = 30 →
  total_time = 120 →
  (total_time - (baking_time + white_icing_time + chocolate_icing_time)) = 45 :=
by
  intros baking_time white_icing_time chocolate_icing_time total_time
  intros h_baking_time h_white_icing_time h_chocolate_icing_time h_total_time
  rw [h_baking_time, h_white_icing_time, h_chocolate_icing_time, h_total_time]
  have : 15 + 30 + 30 = 75 := rfl
  rw this
  show 120 - 75 = 45
  rfl

end elsa_cookie_baking_time_l51_51364


namespace card_distribution_problem_l51_51138

theorem card_distribution_problem :
  let cards := {1, 2, 3, 4, 5, 6}
  let envelopes := {a, b, c}
  ∃ (f : cards → envelopes), 
    (∀ (c : cards), f c ∈ envelopes) ∧
    (f 1 = f 2) ∧
    (∃ (disjoint_pairs : {A | A ⊆ cards ∧ A.card = 2} → {B | B ⊆ envelopes ∧ B.card = 3}),
       (∀ (A : {A | A ⊆ cards ∧ A.card = 2}), disjoint_pairs A ∈ envelopes)
       → 18) :=
begin
  sorry
end

end card_distribution_problem_l51_51138


namespace find_f_5_l51_51374

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^3 + b * Real.sin x + c / x + 4

theorem find_f_5 (a b c m : ℝ) (h₁ : f (-5) a b c = m) : f 5 a b c = 8 - m := 
by 
  sorry

end find_f_5_l51_51374


namespace simplify_complex_expression_l51_51615

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l51_51615


namespace product_of_n_satisfying_condition_l51_51813

theorem product_of_n_satisfying_condition :
  ∃ (ns : List ℕ), (∀ n ∈ ns, (∃ q : ℕ, Nat.Prime q ∧ n^2 - 45 * n + 396 = q)) ∧ (0 < ∀ n ∈ ns) ∧ ns.prod = 396 :=
sorry

end product_of_n_satisfying_condition_l51_51813


namespace three_digit_number_property_l51_51349

theorem three_digit_number_property :
  (∃ a b c : ℕ, 100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c = (a + b + c)^3) ↔
  (∃ a b c : ℕ, a = 5 ∧ b = 1 ∧ c = 2 ∧ 100 * a + 10 * b + c = 512) := sorry

end three_digit_number_property_l51_51349


namespace positive_operation_l51_51686

def operation_a := 1 + (-2)
def operation_b := 1 - (-2)
def operation_c := 1 * (-2)
def operation_d := 1 / (-2)

theorem positive_operation : operation_b > 0 ∧ 
  (operation_a <= 0) ∧ (operation_c <= 0) ∧ (operation_d <= 0) := by
  sorry

end positive_operation_l51_51686


namespace max_obtuse_in_convex_quadrilateral_l51_51893

-- Definition and problem statement
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

theorem max_obtuse_in_convex_quadrilateral (a b c d : ℝ) :
  convex_quadrilateral a b c d →
  (is_obtuse a → (is_obtuse b → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse b → (is_obtuse a → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse c → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse d))) →
  (is_obtuse d → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse c))) :=
by
  intros h_convex h1 h2 h3 h4
  sorry

end max_obtuse_in_convex_quadrilateral_l51_51893


namespace total_money_earned_l51_51495

theorem total_money_earned (initial_earn: ℕ) (refer_earn: ℕ) (friends_day_one: ℕ) (friends_week: ℕ):
  initial_earn = 5 ∧ refer_earn = 5 ∧ friends_day_one = 5 ∧ friends_week = 7 →
  let katrina_earn := initial_earn + (friends_day_one + friends_week) * refer_earn in
  let friends_earn := (friends_day_one + friends_week) * refer_earn in
  katrina_earn + friends_earn = 125 :=
by
  sorry

end total_money_earned_l51_51495


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51825

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51825


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51824

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51824


namespace first_digit_one_over_137_l51_51221

-- Define the main problem in terms of first nonzero digit.
def first_nonzero_digit_right_of_decimal (n : ℕ) : ℕ :=
  let frac := 1 / (Rat.of_int n)
  let shifted_frac := frac * 10 ^ 3
  let integer_part := shifted_frac.to_nat
  integer_part % 10

theorem first_digit_one_over_137 :
  first_nonzero_digit_right_of_decimal 137 = 7 :=
by
  sorry

end first_digit_one_over_137_l51_51221


namespace cookies_fit_in_box_l51_51013

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end cookies_fit_in_box_l51_51013


namespace max_parallelograms_in_hexagon_l51_51895

noncomputable def parallelogram_fitting_in_hexagon : Prop :=
  let parallelogram_area := 4 * (Real.sqrt 3 / 4)
  let hexagon_area := 6 * (9 * (Real.sqrt 3 / 4))
  let max_parallelograms := hexagon_area / parallelogram_area
  ∃ (n : ℕ), n = 12 ∧ Real.floor max_parallelograms = n

theorem max_parallelograms_in_hexagon : parallelogram_fitting_in_hexagon :=
sorry

end max_parallelograms_in_hexagon_l51_51895


namespace number_of_valid_sequences_l51_51901

-- Definitions for conditions
def digit := Fin 10 -- Digit can be any number from 0 to 9
def is_odd (n : digit) : Prop := n.val % 2 = 1
def is_even (n : digit) : Prop := n.val % 2 = 0

def valid_sequence (s : Fin 8 → digit) : Prop :=
  ∀ i : Fin 7, (is_odd (s i) ↔ is_even (s (i+1)))

-- Theorem statement
theorem number_of_valid_sequences : 
  ∃ n, n = 781250 ∧ 
    ∃ s : (Fin 8 → digit), valid_sequence s :=
sorry -- Proof is not required

end number_of_valid_sequences_l51_51901


namespace line_equation_l51_51277

theorem line_equation (a T : ℝ) (h : 0 < a ∧ 0 < T) :
  ∃ (x y : ℝ), (2 * T * x - a^2 * y + 2 * a * T = 0) :=
by
  sorry

end line_equation_l51_51277


namespace area_of_sector_is_correct_l51_51740

noncomputable def calculate_area_of_sector (arc_length : ℝ) (height : ℝ) : ℝ :=
  let radius := arc_length / (2 * Real.pi)
  let slant_height := Real.sqrt (height^2 + radius^2)
  (1 / 2) * arc_length * slant_height

theorem area_of_sector_is_correct (arc_length : ℝ) (height : ℝ) (area : ℝ) : 
  arc_length = 16 * Real.pi → height = 6 → area = 80 * Real.pi → 
  calculate_area_of_sector arc_length height = area := 
by
  intros h_arc h_height h_area
  rw [calculate_area_of_sector, h_arc, h_height, h_area]
  let r := 16 * Real.pi / (2 * Real.pi)
  have hr : r = 8 := by norm_num [r]
  let l := Real.sqrt (6^2 + 8^2)
  have hl : l = 10 := by norm_num [l]
  simp [calculate_area_of_sector, hr, hl]
  sorry

end area_of_sector_is_correct_l51_51740


namespace kolya_product_divisors_1024_l51_51085

theorem kolya_product_divisors_1024 :
  let number := 1024
  let screen_limit := 10^16
  ∃ (divisors : List ℕ), 
    (∀ d ∈ divisors, d ∣ number) ∧ 
    (∏ d in divisors, d) = 2^55 ∧ 
    2^55 > screen_limit :=
by
  sorry

end kolya_product_divisors_1024_l51_51085


namespace interest_rate_l51_51167

theorem interest_rate (P CI SI: ℝ) (r: ℝ) : P = 5100 → CI = P * (1 + r)^2 - P → SI = P * r * 2 → (CI - SI = 51) → r = 0.1 :=
by
  intros
  -- skipping the proof
  sorry

end interest_rate_l51_51167


namespace ratio_of_largest_to_sum_others_l51_51336

theorem ratio_of_largest_to_sum_others :
  let s : Finset ℕ := Finset.range 11
  let elements := s.map (λ n, 3 * 10^n)
  let largest := 3 * 10^10
  let others_sum := (s.filter (λ n, n ≠ 10)).sum (λ n, 3 * 10^n)
  largest / others_sum = 27 := by
  sorry

end ratio_of_largest_to_sum_others_l51_51336


namespace zeros_in_square_of_nines_l51_51439

theorem zeros_in_square_of_nines : 
  let n := 10
  in (10^n - 1)^2 = 10^20 - 2*10^n + 1 →
     (∃ z : ℕ, z = 10) :=
by 
  sorry

end zeros_in_square_of_nines_l51_51439


namespace books_problem_l51_51965

variable (L W : ℕ) -- L for Li Ming's initial books, W for Wang Hong's initial books

theorem books_problem (h1 : L = W + 26) (h2 : L - 14 = W + 14 - 2) : 14 = 14 :=
by
  sorry

end books_problem_l51_51965


namespace closest_perfect_square_to_350_l51_51663

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l51_51663


namespace dimitri_weekly_calories_l51_51794

-- Define the calories for each type of burger
def calories_burger_a : ℕ := 350
def calories_burger_b : ℕ := 450
def calories_burger_c : ℕ := 550

-- Define the daily consumption of each type of burger
def daily_consumption_a : ℕ := 2
def daily_consumption_b : ℕ := 1
def daily_consumption_c : ℕ := 3

-- Define the duration in days
def duration_in_days : ℕ := 7

-- Define the total number of calories Dimitri consumes in a week
noncomputable def total_weekly_calories : ℕ :=
  (daily_consumption_a * calories_burger_a +
   daily_consumption_b * calories_burger_b +
   daily_consumption_c * calories_burger_c) * duration_in_days

theorem dimitri_weekly_calories : total_weekly_calories = 19600 := 
by 
  sorry

end dimitri_weekly_calories_l51_51794


namespace volume_tetrahedron_OMNB1_l51_51768

-- Define points in the unit cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (1, 1, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def A1 : ℝ × ℝ × ℝ := (0, 0, 1)
def B1 : ℝ × ℝ × ℝ := (1, 0, 1)
def C1 : ℝ × ℝ × ℝ := (1, 1, 1)
def D1 : ℝ × ℝ × ℝ := (0, 1, 1)
def O : ℝ × ℝ × ℝ := (0.5, 0.5, 0)
def M : ℝ × ℝ × ℝ := (0, 0.5, 1)
def N : ℝ × ℝ × ℝ := (1, 1, 2/3)

-- Compute the vectors
def vector_OB1 := (1 - 0.5, 0 - 0.5, 1)
def vector_ON := (1 - 0.5, 1 - 0.5, 2/3)
def vector_OM := (0 - 0.5, 0.5 - 0.5, 1)

-- Assert the volume of the tetrahedron
theorem volume_tetrahedron_OMNB1 : 
  let cross_product := (
    vector_OB1.2 * vector_ON.3 - vector_OB1.3 * vector_ON.2, 
    vector_OB1.3 * vector_ON.1 - vector_OB1.1 * vector_ON.3, 
    vector_OB1.1 * vector_ON.2 - vector_OB1.2 * vector_ON.1
  )
  let dot_product := cross_product.1 * vector_OM.1 + cross_product.2 * vector_OM.2 + cross_product.3 * vector_OM.3
  let volume := (1 / 6) * abs dot_product
  volume = 11 / 72 :=
by
  sorry

end volume_tetrahedron_OMNB1_l51_51768


namespace four_digit_sum_of_digits_divisible_by_101_l51_51342

theorem four_digit_sum_of_digits_divisible_by_101 (a b c d : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 1 ≤ b ∧ b ≤ 9)
  (h3 : 1 ≤ c ∧ c ≤ 9)
  (h4 : 1 ≤ d ∧ d ≤ 9)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_div : (1001 * a + 110 * b + 110 * c + 1001 * d) % 101 = 0) :
  (a + d) % 101 = (b + c) % 101 :=
by
  sorry

end four_digit_sum_of_digits_divisible_by_101_l51_51342


namespace shirt_cost_is_six_l51_51487

-- Define constants and conditions
def shirts_bought : ℕ := 10
def pants_bought : ℕ := shirts_bought / 2 -- half as many pants as shirts
def pants_cost : ℝ := 8 -- cost of each pant
def total_cost : ℝ := 100 -- total cost of shirts and pants

-- Main theorem: proving the cost of each shirt
theorem shirt_cost_is_six (S : ℝ) (h_shirts_bought : shirts_bought = 10)
  (h_pants_bought : pants_bought = 5)
  (h_pants_cost : pants_cost = 8)
  (h_total_cost : total_cost = 100) :
  10 * S + (pants_bought * pants_cost) = total_cost → S = 6 :=
by
  intro h
  have h1 : 10 * S + 40 = 100 := by rw [←h_pants_cost, ←h_pants_bought] at h; assumption
  have h2 : 10 * S = 60 := by linarith[h1]
  have h3 : S = 6 := by linarith[h2]
  exact h3

end shirt_cost_is_six_l51_51487


namespace find_p_power_l51_51867

theorem find_p_power (p : ℕ) (h1 : p % 2 = 0) (h2 : (p + 1) % 10 = 7) : 
  (p % 10)^3 % 10 = (p % 10)^1 % 10 :=
by
  sorry

end find_p_power_l51_51867


namespace Lisa_eats_all_candies_in_4_weeks_l51_51529

theorem Lisa_eats_all_candies_in_4_weeks:
  ∀ (candies : ℕ) (days_eat_2 : ℕ) (days_eat_1 : ℕ),
    candies = 36 →
    days_eat_2 = 2 →
    days_eat_1 = 5 →
    (2 * days_eat_2 + days_eat_1) = 9 →
    (candies / (2 * days_eat_2 + days_eat_1)) = 4 :=
by 
  intros candies days_eat_2 days_eat_1 h_candies h_days2 h_days1 h_total
  rw [h_candies, h_days2, h_days1, h_total]
  norm_num
  sorry

end Lisa_eats_all_candies_in_4_weeks_l51_51529


namespace marble_problem_l51_51318

variables (a b : ℝ)
variable h_b : b = 6

theorem marble_problem 
  (h1 : 88 * a - 6 = 240)
  (h_b : b = 6):
  a = 123 / 44 :=
by
  intro h1
  have ha := h1
  sorry

end marble_problem_l51_51318


namespace bill_bought_60_rats_l51_51325

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end bill_bought_60_rats_l51_51325


namespace summation_eq_16_implies_x_eq_3_over_4_l51_51022

theorem summation_eq_16_implies_x_eq_3_over_4 (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x^n = 16) : x = 3 / 4 :=
sorry

end summation_eq_16_implies_x_eq_3_over_4_l51_51022


namespace det_scaled_matrices_l51_51870

variable (a b c d : ℝ)

-- Given condition: determinant of the original matrix
def det_A : ℝ := Matrix.det ![![a, b], ![c, d]]

-- Problem statement: determinants of the scaled matrices
theorem det_scaled_matrices
    (h: det_A a b c d = 3) :
  Matrix.det ![![3 * a, 3 * b], ![3 * c, 3 * d]] = 27 ∧
  Matrix.det ![![4 * a, 2 * b], ![4 * c, 2 * d]] = 24 :=
by
  sorry

end det_scaled_matrices_l51_51870


namespace area_of_triangle_ABE_l51_51478

-- Definitions based on conditions
def triangle_ABC_right_angles (B C : ℤ) : Prop := B = 90 ∧ C = 90
def triangle_angles (AEB BEC : ℤ) : Prop := AEB = 45 ∧ BEC = 45
def AE_value : ℝ := 30

-- Mathematically equivalent proof problem
theorem area_of_triangle_ABE : 
  ∀ (AB BE : ℝ), 
  triangle_ABC_right_angles 90 90 → 
  triangle_angles 45 45 → 
  AE_value = 30 → 
  AB = (AE_value / (Real.sqrt 2)) → 
  BE = (AE_value / (Real.sqrt 2)) → 
  (1 / 2) * AB * BE = 225 :=
by 
  intros AB BE right_angles angle_measures AE_val_eq AB_eq BE_eq,
  -- Proof steps would go here, omitted for this task
  sorry

end area_of_triangle_ABE_l51_51478


namespace michael_saves_more_with_promotion_a_l51_51719

theorem michael_saves_more_with_promotion_a :
  let first_pair := 50
  let second_pair := 40
  let promo_a_cost := first_pair + second_pair / 2
  let promo_b_cost := first_pair + second_pair - 15
  promo_b_cost - promo_a_cost = 5 :=
by
  let first_pair := 50
  let second_pair := 40
  let promo_a_cost := first_pair + second_pair / 2
  let promo_b_cost := first_pair + second_pair - 15
  have h : promo_b_cost - promo_a_cost = 5
  exact h

end michael_saves_more_with_promotion_a_l51_51719


namespace closest_perfect_square_to_350_l51_51678

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l51_51678


namespace kolya_cannot_see_full_result_l51_51089

theorem kolya_cannot_see_full_result : let n := 1024 in 
                                      let num_decimals := 16 in 
                                      (num_divisors_product n > 10^num_decimals) := 
begin
    let n := 1024,
    let num_decimals := 16,
    have product := (∑ i in range (nat.log2 n), i),
    have sum_of_powers := (2^product),
    have required_digits := (10^num_decimals),
    exact sum_of_powers > required_digits,
end

end kolya_cannot_see_full_result_l51_51089


namespace max_right_angles_2022_l51_51557

axiom points_2022 (A : Fin 2022 → Point) : ∀ i j k : Fin 2022, 
  i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬Collinear ({A i, A j, A k} : Set Point)

theorem max_right_angles_2022 {A : Fin 2022 → Point} 
  (h_no_collinear : ∀ i j k : Fin 2022, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬Collinear ({A i, A j, A k} : Set Point)) : 
  ∃ max_count : ℕ, max_count = 2_042_220 ∧ ∀ other_count : ℕ, other_count ≤ 2_042_220 := 
begin
  sorry
end

end max_right_angles_2022_l51_51557


namespace maximum_ab_value_l51_51441

noncomputable def max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 6) : ℝ :=
  by
    apply sqrt (max (9 : ℝ) 0)

theorem maximum_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 6) :
  ab = 9 :=
  sorry

end maximum_ab_value_l51_51441


namespace desired_gain_percentage_l51_51296

theorem desired_gain_percentage :
  (∀ (C : ℝ), (0.96 * C = 1) →
  ∃ (G : ℝ), (12 / 16 * C) * (1 + G / 100) = 1 ∧ G = 28) :=
by 
  intro C hC,
  use 28,
  split,
  {
    have h : C = 1 / 0.96, from calc
      C = C : by rfl
      ... = 1 / 0.96 : by linarith [hC],
    linarith,
  },
  {
    linarith,
  }

end desired_gain_percentage_l51_51296


namespace find_cd_l51_51746

theorem find_cd (c d : ℕ) (h1: 0 ≤ c ∧ c < 10) (h2: 0 ≤ d ∧ d < 10) :
  (99 * (2 + (c / 10) + (d / 100)) - 99 * (2 + 0.0cd)) = 0.75 → c = 7 ∧ d = 5 :=
sorry

end find_cd_l51_51746


namespace last_digit_of_decimal_expansion_of_fraction_l51_51634

theorem last_digit_of_decimal_expansion_of_fraction : 
  let x := (1 : ℚ) / (3 ^ 15) 
  in ∃ digit : ℕ, (digit < 10) ∧ last_digit (decimal_expansion x) = 7 :=
sorry

end last_digit_of_decimal_expansion_of_fraction_l51_51634


namespace no_perpendicular_AC_BC_constant_chord_length_l51_51049

-- Define the curve and its intersection with the x-axis
def curve := λ m x : ℝ, x^2 + m * x - 2

-- Define conditions given in the problem
axiom A (m x1 x2 : ℝ) (h : A = curve m):
  x1 * x2 = -2 

-- Define point C
def C : ℝ × ℝ := (0, 1)

-- Prove that there cannot be a situation where AC is perpendicular to BC
theorem no_perpendicular_AC_BC (m x1 x2 : ℝ) (h : x1 * x2 = -2) : 
  ¬(0 - x1) * (0 - x2) = -1 := 
sorry

-- Prove that the chord length on the y-axis is always 3
theorem constant_chord_length (m : ℝ) : 
  ∀ (x y : ℝ), (x^2 + y^2 + m*x + y - 2 = 0) → y ∈ {1, -2} ∧ abs (-2 - 1) = 3 :=
sorry

end no_perpendicular_AC_BC_constant_chord_length_l51_51049


namespace incorrect_statements_about_frequency_and_probability_l51_51257

theorem incorrect_statements_about_frequency_and_probability :
  let A := ∀ (P : ℕ → Prop) (p : ℕ → ℝ), (∀ n, 0 ≤ p n ∧ p n ≤ 1) →
    (∃ p_b : ℝ, 0 ≤ p_b ∧ p_b ≤ 1 ∧ ∀ n, p_b = p n) →
    (∃ n, P n ∧ p n = 0.1) → 
    ¬(∀ k, P k → (∃ (selected : list ℕ), selected.length = 100 ∧ (selected.count P = 10)))

  let B := ∀ (flip : ℕ → Prop), (∀ n, 0 ≤ (flip n : ℝ) ∧ (flip n : ℝ) ≤ 1) →
    (∃ k, (flip k : ℝ) = 0.5) →
    ∀ n, (∃ (selected : list ℕ), selected.length = 7 ∧ (selected.count flip = 3) →
      ¬(∃ (freq : ℝ), freq = (3 : ℝ) / 7))

  let C := ∀ (E : Prop) (frequency probability : ℝ), 
    0 ≤ probability ∧ probability ≤ 1 →
    frequency = probability → 
    ¬ (frequency = probability)

  let D := ∀ (n : ℕ) (trials : list ℕ) (E : ℕ → Prop),
    10000 ≤ n →
    (∀ m, m ∈ trials → E m) →
    (∃ frequency probability : ℝ, 0 ≤ probability ∧ probability ≤ 1 ∧
      abs (frequency - probability) < (1 / n)) →
    ¬ (∀ (m : ℕ), m ∈ trials → (abs ((trials.count E : ℝ) / n - probability) < (1 / n)))

  A ∧ B ∧ C ∧ D :=
by
  let A := ∀ (P : ℕ → Prop) (p : ℕ → ℝ), (∀ n, 0 ≤ p n ∧ p n ≤ 1) →
    (∃ p_b : ℝ, 0 ≤ p_b ∧ p_b ≤ 1 ∧ ∀ n, p_b = p n) →
    (∃ n, P n ∧ p n = 0.1) → 
    ¬(∀ k, P k → (∃ (selected : list ℕ), selected.length = 100 ∧ (selected.count P = 10)))

  let B := ∀ (flip : ℕ → Prop), (∀ n, 0 ≤ (flip n : ℝ) ∧ (flip n : ℝ) ≤ 1) →
    (∃ k, (flip k : ℝ) = 0.5) →
    ∀ n, (∃ (selected : list ℕ), selected.length = 7 ∧ (selected.count flip = 3) →
      ¬(∃ (freq : ℝ), freq = (3 : ℝ) / 7))

  let C := ∀ (E : Prop) (frequency probability : ℝ), 
    0 ≤ probability ∧ probability ≤ 1 →
    frequency = probability → 
    ¬ (frequency = probability)

  let D := ∀ (n : ℕ) (trials : list ℕ) (E : ℕ → Prop),
    10000 ≤ n →
    (∀ m, m ∈ trials → E m) →
    (∃ frequency probability : ℝ, 0 ≤ probability ∧ probability ≤ 1 ∧
      abs (frequency - probability) < (1 / n)) →
    ¬ (∀ (m : ℕ), m ∈ trials → (abs ((trials.count E : ℝ) / n - probability) < (1 / n)))

  exact ⟨sorry, sorry, sorry, sorry⟩

end incorrect_statements_about_frequency_and_probability_l51_51257


namespace sin_sum_triangle_max_l51_51449

theorem sin_sum_triangle_max :
  (∀ x y z : ℝ, (0 < x ∧ x < π) ∧ (0 < y ∧ y < π) ∧ (0 < z ∧ z < π) ∧ (x + y + z = π) → 
  (f : ℝ → ℝ) (hf : ∀ a b c : ℝ, f x = sin x → convex_on (0,π) f ∧ (f x + f y + f z) / 3 ≤ f((x + y + z) / 3)) → 
  sin x + sin y + sin z ≤ 3 * (sqrt 3 / 2)) :=
begin
  sorry
end

end sin_sum_triangle_max_l51_51449


namespace infinite_gcd_one_l51_51978

theorem infinite_gcd_one : ∃ᶠ n in at_top, Int.gcd n ⌊Real.sqrt 2 * n⌋ = 1 := sorry

end infinite_gcd_one_l51_51978


namespace minimize_abs_difference_and_product_l51_51756

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end minimize_abs_difference_and_product_l51_51756


namespace find_number_l51_51270

theorem find_number (x : ℚ) (h : 0.15 * 0.30 * 0.50 * x = 108) : x = 4800 :=
by
  sorry

end find_number_l51_51270


namespace minimum_temperature_of_hottest_stars_l51_51170

-- Define the luminosity formula
def luminosity (y : ℝ) : ℝ := 5 * 10^7 * y^(-2)

-- Define the minimum luminosity condition
def minimum_luminosity : ℝ := 2000

-- The minimum temperature for the hottest 50 stars with at least 2000 lumens
theorem minimum_temperature_of_hottest_stars (y : ℝ) (h : luminosity(y) ≥ minimum_luminosity) : y ≥ 158 := by 
    -- Skip the proof
    sorry

end minimum_temperature_of_hottest_stars_l51_51170


namespace evaluate_expression_l51_51621

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l51_51621


namespace kolya_screen_display_limit_l51_51082

theorem kolya_screen_display_limit : 
  let n := 1024
  let screen_limit := 10^16
  (∀ d ∈ list.divisors n, d = (2^0) ∨ d = (2^1) ∨ d = (2^2) ∨ d = (2^3) ∨ d = (2^4) ∨ d = (2^5) ∨ d = (2^6) ∨ d = (2^7) ∨ d = (2^8) ∨ d = (2^9) ∨ d = (2^10)) →
  ((list.prod (list.divisors n)) > screen_limit) :=
begin
  sorry
end

end kolya_screen_display_limit_l51_51082


namespace kolya_screen_display_limit_l51_51079

theorem kolya_screen_display_limit : 
  let n := 1024
  let screen_limit := 10^16
  (∀ d ∈ list.divisors n, d = (2^0) ∨ d = (2^1) ∨ d = (2^2) ∨ d = (2^3) ∨ d = (2^4) ∨ d = (2^5) ∨ d = (2^6) ∨ d = (2^7) ∨ d = (2^8) ∨ d = (2^9) ∨ d = (2^10)) →
  ((list.prod (list.divisors n)) > screen_limit) :=
begin
  sorry
end

end kolya_screen_display_limit_l51_51079


namespace parabola_problem_l51_51423

noncomputable def parabola_properties (p x0 : ℝ) (M : ℝ × ℝ) (A F : ℝ × ℝ) : Prop :=
y^2 = 2 * p * x ∧ p > 0 ∧ M = (x0, 2 * real.sqrt 2) ∧ x0 > p / 2 ∧
(line_intersects_segment M F A) ∧
(chord_length (x = p / 2) (circle M) = real.sqrt 3 * distance M A) ∧
(distance M A / distance A F = 2) → (distance A F = 1)

-- Top-level theorem
theorem parabola_problem :
  ∀ (p x0 : ℝ) (M A F : ℝ × ℝ), parabola_properties p x0 M A F → distance A F = 1 :=
by
  sorry

end parabola_problem_l51_51423


namespace find_x_l51_51145

-- Definitions corresponding to conditions a)
def rectangle (AB CD BC AD x : ℝ) := AB = 2 ∧ CD = 2 ∧ BC = 1 ∧ AD = 1 ∧ x = 0

-- Define the main statement to be proven
theorem find_x (AB CD BC AD x k m: ℝ) (h: rectangle AB CD BC AD x) : 
  x = (0 : ℝ) ∧ k = 0 ∧ m = 0 ∧ x = (Real.sqrt k - m) ∧ k + m = 0 :=
by
  cases h
  sorry

end find_x_l51_51145


namespace wrongly_written_height_l51_51577

theorem wrongly_written_height 
  (n : ℕ) (wrong_avg actual_avg : ℝ) (actual_height : ℝ) (error_height : ℝ) (n_val : n = 35) 
  (wrong_avg_val : wrong_avg = 180) (actual_avg_val : actual_avg = 178)
  (actual_height_val : actual_height = 106)
  (calc_wrong_height : ∑ i in finset.range n, if i = 0 then error_height else wrong_avg = n * wrong_avg)
  (calc_actual_height : ∑ i in finset.range n, if i = 0 then actual_height else actual_avg = n * actual_avg)
  (error_val : error_height = actual_height + (n * wrong_avg - n * actual_avg)) :
  error_height = 176 :=
by
  sorry

end wrongly_written_height_l51_51577


namespace range_of_a_l51_51910

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 4^x - (a + 3) * 2^x + 1 = 0) → a ≥ -1 := sorry

end range_of_a_l51_51910


namespace partition_subset_disjoint_l51_51517

open Nat

theorem partition_subset_disjoint 
  (n : ℕ) (h_n : 0 < n)
  (S : Finset (Fin (n^2 + n - 1))) 
  (part : Finset (Fin (n^2 + n - 1)).subsets n → Bool) :
  ∃ P : Bool, ∃ F : Finset (Fin (n^2 + n - 1).subsets n), 
  (∀ A ∈ F, part A = P) ∧ (F.card ≥ n) ∧ (∀ A B ∈ F, A ∩ B = ∅) :=
sorry

end partition_subset_disjoint_l51_51517


namespace range_a_real_numbers_l51_51790

theorem range_a_real_numbers (x a : ℝ) : 
  (∀ x : ℝ, (x - a) * (1 - (x + a)) < 1) → (a ∈ Set.univ) :=
by
  sorry

end range_a_real_numbers_l51_51790


namespace tangent_secan_power_of_a_point_problems_l51_51503

variable (P O T A B : Type) 
variable [PA PB PT : ℝ]  -- PA, PB, and PT are real numbers

-- Conditions
variable (PA_eq : PA = 5)
variable (PT_eq : PT = (AB - 2 * PA))

-- Power of a point theorem
theorem tangent_secan_power_of_a_point_problems:
  PA * PB = PT * PT → 
  PA < PB →
  PB = 25 :=
by
  -- Proof goes here
  sorry

end tangent_secan_power_of_a_point_problems_l51_51503


namespace value_of_each_baseball_card_l51_51969

theorem value_of_each_baseball_card (x : ℝ) (h : 2 * x + 3 = 15) : x = 6 := by
  sorry

end value_of_each_baseball_card_l51_51969


namespace find_point_P_l51_51868

noncomputable def pointP (P : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), P = (x, y) ∧ x = Real.sqrt(15) / 2 ∧ y = 1

theorem find_point_P :
  ∀ (P : ℝ × ℝ),
  (P.1 > 0) →
  (P.1^2 / 5 + P.2^2 / 4 = 1) →
  (abs (P.2) * 1 = 1) →
  pointP P :=
by
  intros P h1 h2 h3
  sorry

end find_point_P_l51_51868


namespace find_Sierra_age_l51_51285

noncomputable def Sierra_age : ℕ := 30

theorem find_Sierra_age:
  ∃ S D: ℕ, (10 * D - 40 = 10 * S + 20) ∧ (D + 20 = 56) ∧ (S = Sierra_age) :=
begin
  sorry

end find_Sierra_age_l51_51285


namespace kolya_screen_display_limit_l51_51080

theorem kolya_screen_display_limit : 
  let n := 1024
  let screen_limit := 10^16
  (∀ d ∈ list.divisors n, d = (2^0) ∨ d = (2^1) ∨ d = (2^2) ∨ d = (2^3) ∨ d = (2^4) ∨ d = (2^5) ∨ d = (2^6) ∨ d = (2^7) ∨ d = (2^8) ∨ d = (2^9) ∨ d = (2^10)) →
  ((list.prod (list.divisors n)) > screen_limit) :=
begin
  sorry
end

end kolya_screen_display_limit_l51_51080


namespace discount_percent_l51_51297

theorem discount_percent (CP MP SP : ℝ) (markup profit: ℝ) (h1 : CP = 100) (h2 : MP = CP + (markup * CP))
  (h3 : SP = CP + (profit * CP)) (h4 : markup = 0.75) (h5 : profit = 0.225) : 
  (MP - SP) / MP * 100 = 30 :=
by
  sorry

end discount_percent_l51_51297


namespace part1_part2_l51_51373

noncomputable def f (x : ℝ) := x^2 - x - 6
noncomputable def g (x : ℝ) (b : ℝ) := b*x - 10

-- Prove that the range of x for f(x) > 0 is x < -2 or x > 3.
theorem part1 : ∀ x : ℝ, f x > 0 ↔ x < -2 ∨ x > 3 :=
begin
  sorry
end

-- Prove that if f(x) > g(x) for all x, then the range of b is b < -5 or b > 3.
theorem part2 (b : ℝ) : (∀ x : ℝ, f x > g x b) ↔ b < -5 ∨ b > 3 :=
begin
  sorry
end

end part1_part2_l51_51373


namespace inequality_proof_l51_51832

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l51_51832


namespace number_of_two_digit_x_with_sum_of_sum_of_digits_eq_5_l51_51507

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_two_digit (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

theorem number_of_two_digit_x_with_sum_of_sum_of_digits_eq_5 :
  (finset.filter (λ x, sum_of_digits (sum_of_digits x) = 5) 
                 (finset.filter is_two_digit (finset.Icc 10 99))).card = 5 := 
  sorry

end number_of_two_digit_x_with_sum_of_sum_of_digits_eq_5_l51_51507


namespace lines_concurrent_l51_51855

noncomputable theory

open_locale classical

variables {A B C D E F K G J H I : Type*}
variables [projective_space A] [projective_space B] [projective_space C]
variables [projective_space D] [projective_space E] [projective_space F]
variables [projective_space K] [projective_space G] [projective_space J]
variables [projective_space H] [projective_space I]

-- Given a triangle ABC and a point D inside the triangle
variables (A B C D : point)

-- E and F are the intersections of BD and CD with AC and AB respectively
variables (E : point) (hE : collinear A C E ∧ line_through B D E)
variables (F : point) (hF : collinear A B F ∧ line_through C D F)

-- Line DK intersects the line FE at point K
variables (K : point) (hK : ∃ (l1 l2 : line), line_through D K ∧ line_through F E ∧ K ∈ l1 ∧ K ∈ l2)

-- G and J are the intersection points of the line CK with the lines AB and BE respectively
variables (G : point) (hG : collinear C K G ∧ line_through A B G)
variables (J : point) (hJ : collinear C K J ∧ line_through B E J)

-- H and I are the intersection points of the line BK with the lines CF and CA respectively
variables (H : point) (hH : collinear B K H ∧ line_through C F H)
variables (I : point) (hI : collinear B K I ∧ line_through C A I)

-- Prove that the lines BC, EF, HG, and IJ are concurrent
theorem lines_concurrent : ∃ P : point, P ∈ (line_through B C) ∧ P ∈ (line_through E F) ∧ P ∈ (line_through H G) ∧ P ∈ (line_through I J) :=
sorry

end lines_concurrent_l51_51855


namespace motorcycles_per_month_l51_51292

-- Define the conditions for car production and sales
def car_material_cost : ℤ := 100
def cars_per_month : ℤ := 4
def car_selling_price : ℤ := 50

-- Calculate the profit from selling cars
def car_revenue : ℤ := cars_per_month * car_selling_price
def car_profit : ℤ := car_revenue - car_material_cost

-- Define the conditions for motorcycle production and sales
def motorcycle_material_cost : ℤ := 250
def motorcycle_selling_price : ℤ := 50
def profit_difference : ℤ := 50

-- Calculate the profit from selling motorcycles
def motorcycle_profit : ℤ := car_profit + profit_difference

-- Define the number of motorcycles sold per month
def motorcycles_sold (x : ℤ) : ℤ := x * motorcycle_selling_price - motorcycle_material_cost

-- Problem statement: Prove the factory sells 8 motorcycles per month
theorem motorcycles_per_month : ∃ x : ℤ, motorcycles_sold x = motorcycle_profit → x = 8 :=
begin
  sorry
end

end motorcycles_per_month_l51_51292


namespace nala_seashells_l51_51543

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end nala_seashells_l51_51543


namespace sum_diff_square_cube_l51_51453

/-- If the sum of two numbers is 25 and the difference between them is 15,
    then the difference between the square of the larger number and the cube of the smaller number is 275. -/
theorem sum_diff_square_cube (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x - y = 15) :
  x^2 - y^3 = 275 :=
sorry

end sum_diff_square_cube_l51_51453


namespace quadratic_equation_with_means_l51_51909

theorem quadratic_equation_with_means (α β : ℝ) 
  (h_am : (α + β) / 2 = 8) 
  (h_gm : Real.sqrt (α * β) = 15) : 
  (Polynomial.X^2 - Polynomial.C (α + β) * Polynomial.X + Polynomial.C (α * β) = 0) := 
by
  have h1 : α + β = 16 := by linarith
  have h2 : α * β = 225 := by sorry
  rw [h1, h2]
  sorry

end quadratic_equation_with_means_l51_51909


namespace rover_can_explore_planet_l51_51728

noncomputable def equatorial_length : ℝ := 400
noncomputable def total_path_length : ℝ := 600
noncomputable def max_distance_from_path : ℝ := 50

def can_fully_explore_planet (equatorial_length : ℝ) (max_distance_from_path : ℝ) (total_path_length : ℝ) : Prop :=
  ∀ (p : ℝ × ℝ × ℝ), ∃ (q : ℝ × ℝ × ℝ), 
    (dist p q < max_distance_from_path) ∧ 
    (total_distance_traveled ≤ total_path_length)

-- Now we state the theorem to be proven:
theorem rover_can_explore_planet :
  can_fully_explore_planet equatorial_length max_distance_from_path total_path_length :=
sorry

end rover_can_explore_planet_l51_51728


namespace a_minus_2_values_l51_51972

theorem a_minus_2_values (a : ℝ) (h : |a| = 3) : a - 2 = 1 ∨ a - 2 = -5 :=
by {
  -- the theorem states that given the absolute value condition, a - 2 can be 1 or -5
  sorry
}

end a_minus_2_values_l51_51972


namespace find_x0_l51_51415

noncomputable def f (a c : ℝ) (x : ℝ) := a * x^2 + c

noncomputable def integral_f (a c : ℝ) : ℝ := (∫ x in (0:ℝ)..1, f a c x)

variable (a c x0 : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (x0_domain : 0 ≤ x0 ∧ x0 ≤ 1)
variable (int_f_eq_f_x0 : integral_f a c = f a c x0)

theorem find_x0 : x0 = real.sqrt (1/3) := sorry

end find_x0_l51_51415


namespace pigeonhole_principle_subsets_l51_51237

theorem pigeonhole_principle_subsets :
  ∀ (S : Finset ℕ), S = ({1,2,3,4,5,6,7,8,9} : Finset ℕ) →
  ∀ (sets : Finset (Finset ℕ)), sets.card = 6 →
  (∀ A ∈ sets, A.card = 5) →
  ∃ (k : ℕ), k ≥ 4 ∧ (∃ (subset_k : Finset (Finset ℕ)), subset_k.card = k ∧ (∃ e ∈ S, ∀ A ∈ subset_k, e ∈ A)) :=
begin
  sorry
end

end pigeonhole_principle_subsets_l51_51237


namespace area_of_triangle_CBD_l51_51370

theorem area_of_triangle_CBD (AC AB : ℝ) (area_ABC : ℝ) (h_ratio : AC / AB = 2 / 3) (h_area : area_ABC = 20) :
  let area_CBD := 25 in
  area_CBD = 25 := by
  sorry

end area_of_triangle_CBD_l51_51370


namespace nines_in_house_numbers_l51_51744

-- Define the range of house numbers.
def houses : List Nat := List.range 60 |>.map (· + 1)

-- Define the condition for counting 9s in a given list of numbers.
def count_nines (l : List Nat) : Nat := 
  l.map (λ n => (n.digits 10).count (· == 9)).sum

-- Prove that the number of 9s painted is 6.
theorem nines_in_house_numbers : count_nines houses = 6 := 
by
  sorry

end nines_in_house_numbers_l51_51744


namespace circle_tangent_secant_problem_l51_51506

theorem circle_tangent_secant_problem 
  (O : Type) [circle O] (P T A B : O)
  (hPoutside: ¬P ∈ O) 
  (h_tangent: tangent_segment P T O)
  (h_secant: secant_segment P A B O)
  (h_dist_PA_PB: distance P A < distance P B)
  (PA_eq_5: distance P A = 5)
  (PT_eq: distance P T = distance A B - 2 * distance P A) :
  distance P B = 20 :=
by
  sorry

end circle_tangent_secant_problem_l51_51506


namespace sum_of_squares_constant_l51_51319

noncomputable def A : Type := Point3d
noncomputable def B : Type := Point3d
noncomputable def C : Type := Point3d
noncomputable def P : Type := Point3d

variable (A B C P : Type)

-- Assume ABC is an equilateral triangle
-- and P is any point on the incircle of the equilateral triangle ABC.
def equilateral (A B C : Point3d) := dist A B = dist B C ∧ dist B C = dist C A

axiom is_incircle (P : Point3d) (A B C : Point3d) : ∃ r : ℝ, dist P (circumcenter A B C) = r

theorem sum_of_squares_constant (A B C P : Point3d) (r : ℝ) 
  (h1 : equilateral A B C)
  (h2 : is_incircle P A B C) :
    dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2 = 15 * r^2 :=
sorry

end sum_of_squares_constant_l51_51319


namespace second_number_is_255_l51_51588

theorem second_number_is_255 (x : ℝ) (n : ℝ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
  (h2 : (128 + n + 511 + 1023 + x) / 5 = 423) : 
  n = 255 :=
sorry

end second_number_is_255_l51_51588


namespace find_k_l51_51727

-- Definitions of the points and the slope condition
def point1 : ℝ × ℝ := (5, -12)
def point2 (k : ℝ) : ℝ × ℝ := (k, 23)
def slope_of_line_through_points (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
def slope_of_given_line : ℝ := -2 / 3

-- The proof problem
theorem find_k (k : ℝ) :
  slope_of_line_through_points point1 (point2 k) = slope_of_given_line → 
  k = -47.5 :=
by
  sorry

end find_k_l51_51727


namespace max_equal_products_of_sums_l51_51159

noncomputable def exists_equal_products (nums : Fin 10 → ℝ) : Prop :=
  ∃ P, ∃ (k : ℕ), k = 4 ∧ (finset.univ.pair_combinations nums).image (λ p, p.1 * p.2).count P = k

theorem max_equal_products_of_sums
  (nums : Fin 10 → ℝ)
  (h1 : function.injective nums)
  (h2 : ∃ S, ∃ (k : ℕ), k = 5 ∧ (finset.univ.pair_combinations nums).image (λ p, p.1 + p.2).count S = k) :
  exists_equal_products nums :=
sorry

end max_equal_products_of_sums_l51_51159


namespace move_negative_one_impossible_l51_51062

theorem move_negative_one_impossible (vertices : Fin 12 → ℤ) 
  (h₀ : ∀ i, vertices i = 1 ∨ vertices i = -1)
  (h₁ : ∃ i, vertices i = -1 ∧ (∀ j, j ≠ i → vertices j = 1))
  (h₂ : ∀ s, s.card = 6 → ∀ v : ℤ, v ∈ s → (vertices v) = -vertices v) :
  ¬ ∃ i j, vertices j = -1 ∧ (vertices i = -1 ∧ (i = j + 1 ∨ i = j - 1)) :=
sorry

end move_negative_one_impossible_l51_51062


namespace total_weekly_sleep_correct_l51_51718

-- Definition of the weekly sleep time for cougar, zebra, and lion
def cougar_sleep_even_days : Nat := 4
def cougar_sleep_odd_days : Nat := 6
def zebra_sleep_even_days := (cougar_sleep_even_days + 2)
def zebra_sleep_odd_days := (cougar_sleep_odd_days + 2)
def lion_sleep_even_days := (zebra_sleep_even_days - 3)
def lion_sleep_odd_days := (cougar_sleep_odd_days + 1)

def total_weekly_sleep_time : Nat :=
  (4 * cougar_sleep_odd_days + 3 * cougar_sleep_even_days) + -- Cougar's total sleep in a week
  (4 * zebra_sleep_odd_days + 3 * zebra_sleep_even_days) + -- Zebra's total sleep in a week
  (4 * lion_sleep_odd_days + 3 * lion_sleep_even_days) -- Lion's total sleep in a week

theorem total_weekly_sleep_correct : total_weekly_sleep_time = 123 := 
by
  -- Total for the week according to given conditions
  sorry -- Proof is omitted, only the statement is required

end total_weekly_sleep_correct_l51_51718


namespace nonneg_solutions_eq_zero_l51_51436

theorem nonneg_solutions_eq_zero :
  (∀ x : ℝ, x^2 + 6 * x + 9 = 0 → x ≥ 0) = 0 :=
by
  sorry

end nonneg_solutions_eq_zero_l51_51436


namespace mrs_hilt_total_spent_l51_51133

def kids_ticket_usual_cost : ℕ := 1 -- $1 for 4 tickets
def adults_ticket_usual_cost : ℕ := 2 -- $2 for 3 tickets

def kids_ticket_deal_cost : ℕ := 4 -- $4 for 20 tickets
def adults_ticket_deal_cost : ℕ := 8 -- $8 for 15 tickets

def kids_tickets_purchased : ℕ := 24
def adults_tickets_purchased : ℕ := 18

def total_kids_ticket_cost : ℕ :=
  let kids_deal_tickets := kids_ticket_deal_cost
  let remaining_kids_tickets := kids_ticket_usual_cost
  kids_deal_tickets + remaining_kids_tickets

def total_adults_ticket_cost : ℕ :=
  let adults_deal_tickets := adults_ticket_deal_cost
  let remaining_adults_tickets := adults_ticket_usual_cost
  adults_deal_tickets + remaining_adults_tickets

def total_cost (kids_cost adults_cost : ℕ) : ℕ :=
  kids_cost + adults_cost

theorem mrs_hilt_total_spent : total_cost total_kids_ticket_cost total_adults_ticket_cost = 15 := by
  sorry

end mrs_hilt_total_spent_l51_51133


namespace probability_of_B_l51_51184

theorem probability_of_B (P : Set ℕ → ℝ) (A B : Set ℕ) (hA : P A = 0.25) (hAB : P (A ∩ B) = 0.15) (hA_complement_B_complement : P (Aᶜ ∩ Bᶜ) = 0.5) : P B = 0.4 :=
by
  sorry

end probability_of_B_l51_51184


namespace simplify_and_evaluate_l51_51990

-- Define the constants
def a : ℤ := -1
def b : ℤ := 2

-- Declare the expression
def expr : ℤ := 7 * a ^ 2 * b + (-4 * a ^ 2 * b + 5 * a * b ^ 2) - (2 * a ^ 2 * b - 3 * a * b ^ 2)

-- Declare the final evaluated result
def result : ℤ := 2 * ((-1 : ℤ) ^ 2) + 8 * (-1) * (2 : ℤ) ^ 2 

-- The theorem we want to prove
theorem simplify_and_evaluate : expr = result :=
by
  sorry

end simplify_and_evaluate_l51_51990


namespace soda_preference_respondents_l51_51045

noncomputable def fraction_of_soda (angle_soda : ℝ) (total_angle : ℝ) : ℝ :=
  angle_soda / total_angle

noncomputable def number_of_soda_preference (total_people : ℕ) (fraction : ℝ) : ℝ :=
  total_people * fraction

theorem soda_preference_respondents (total_people : ℕ) (angle_soda : ℝ) (total_angle : ℝ) : 
  total_people = 520 → angle_soda = 298 → total_angle = 360 → 
  number_of_soda_preference total_people (fraction_of_soda angle_soda total_angle) = 429 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold fraction_of_soda number_of_soda_preference
  -- further calculation steps
  sorry

end soda_preference_respondents_l51_51045


namespace sum_SHE_equals_6_l51_51156

-- Definitions for conditions
variables {S H E : ℕ}

-- Conditions as stated in the problem
def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ H ∧ H ≠ E ∧ S ≠ E ∧ 1 ≤ S ∧ S < 8 ∧ 1 ≤ H ∧ H < 8 ∧ 1 ≤ E ∧ E < 8

-- Base 8 addition problem
def addition_holds_in_base8 (S H E : ℕ) : Prop :=
  (E + H + (S + E + H) / 8) % 8 = S ∧    -- First column carry
  (H + S + (E + H + S) / 8) % 8 = E ∧    -- Second column carry
  (S + E + (H + S + E) / 8) % 8 = H      -- Third column carry

-- Final statement
theorem sum_SHE_equals_6 :
  distinct_non_zero_digits S H E → addition_holds_in_base8 S H E → S + H + E = 6 :=
by sorry

end sum_SHE_equals_6_l51_51156


namespace rick_books_division_l51_51984

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l51_51984


namespace smallest_positive_period_of_f_range_of_transformed_f_l51_51414

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x - 8 * (Real.cos (x / 4))^4

-- Question (I): Prove that the smallest positive period of f(x) is 4π.
theorem smallest_positive_period_of_f : ∃ T > 0, T = 4 * Real.pi ∧ ∀ x, f (x + T) = f x :=
sorry

noncomputable def transformed_f (x : ℝ) : ℝ :=
  f (2 * x - Real.pi / 6)

-- Question (II): Prove that the range of transformed_f(x) for x ∈ [-π/6, π/4] is [-5, -4].
theorem range_of_transformed_f : 
  Set.range (λ (x : ℝ), transformed_f x) = Set.Icc (-5:ℝ) (-4:ℝ) :=
sorry

end smallest_positive_period_of_f_range_of_transformed_f_l51_51414


namespace smallest_n_condition_l51_51943

noncomputable def distance_origin_to_point (n : ℕ) : ℝ := Real.sqrt (n)

noncomputable def radius_Bn (n : ℕ) : ℝ := distance_origin_to_point n - 1

def condition_Bn_contains_point_with_coordinate_greater_than_2 (n : ℕ) : Prop :=
  radius_Bn n > 2

theorem smallest_n_condition : ∃ n : ℕ, n ≥ 10 ∧ condition_Bn_contains_point_with_coordinate_greater_than_2 n :=
  sorry

end smallest_n_condition_l51_51943


namespace convert_55C_to_F_l51_51275

variable (c : ℝ) (f : ℝ)

def celsius_to_fahrenheit (c : ℝ) : ℝ := (c * (9 / 5)) + 32

theorem convert_55C_to_F : celsius_to_fahrenheit 55 = 131 := by
  sorry

end convert_55C_to_F_l51_51275


namespace closest_perfect_square_to_350_l51_51654

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l51_51654


namespace dodecagon_enclosure_l51_51304

theorem dodecagon_enclosure (m n : ℕ) (h1 : m = 12) 
  (h2 : ∀ (x : ℕ), x ∈ { k | ∃ p : ℕ, p = n ∧ 12 = k * p}) :
  n = 12 :=
by
  -- begin proof steps here
sorry

end dodecagon_enclosure_l51_51304


namespace number_of_girls_in_group_l51_51307

open Finset

/-- Given that a tech group consists of 6 students, and 3 people are to be selected to visit an exhibition,
    if there are at least 1 girl among the selected, the number of different selection methods is 16,
    then the number of girls in the group is 2. -/
theorem number_of_girls_in_group :
  ∃ n : ℕ, (n ≥ 1 ∧ n ≤ 6 ∧ 
            (Nat.choose 6 3 - Nat.choose (6 - n) 3 = 16)) → n = 2 :=
by
  sorry

end number_of_girls_in_group_l51_51307


namespace identify_conic_section_hyperbola_l51_51019

-- Defining the variables and constants in the Lean environment
variable (x y : ℝ)

-- The given equation in function form
def conic_section_eq : Prop := (x - 3) ^ 2 = 4 * (y + 2) ^ 2 + 25

-- The expected type of conic section (Hyperbola)
def is_hyperbola : Prop := 
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^2 - b * y^2 + c * x + d * y + e = f

-- The theorem statement to prove
theorem identify_conic_section_hyperbola (h : conic_section_eq x y) : is_hyperbola x y := by
  sorry

end identify_conic_section_hyperbola_l51_51019


namespace winter_mowing_l51_51494

def spring_mowing : ℕ := 8
def summer_mowing : ℕ := 5
def fall_mowing : ℕ := 12
def average_mowing : ℕ := 7
def seasons : ℕ := 4

theorem winter_mowing :
  spring_mowing + summer_mowing + fall_mowing + ?w = seasons * average_mowing → ?w = 3 :=
by
  intro h
  have total_mowing := seasons * average_mowing
  have lawn_before_winter := spring_mowing + summer_mowing + fall_mowing
  have winter_mowing := total_mowing - lawn_before_winter
  rw h at winter_mowing
  exact winter_mowing

end winter_mowing_l51_51494


namespace cylinder_height_l51_51171

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h₀ : r = 3) (h₁ : SA = 36 * Real.pi) (h₂ : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : h = 3 :=
by
  -- The proof will be constructed here
  sorry

end cylinder_height_l51_51171


namespace rick_group_division_l51_51979

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l51_51979


namespace proof_inequality_l51_51207

theorem proof_inequality (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≤ 0 → (a^2 + b^2 - 1 - a^2 * b^2) ≤ 0 :=
by
  intro h
  have h1 : (a^2 + b^2 - 1 - a^2 * b^2) = (a^2 - 1) * (1 - b^2),
  {
    -- Proof of equality transformation goes here (skipped)
    sorry,
  }
  rw [h1]
  linarith 

end proof_inequality_l51_51207


namespace geometry_problem_l51_51874

noncomputable def vertices_on_hyperbola (A B C : ℝ × ℝ) : Prop :=
  (∃ x1 y1, A = (x1, y1) ∧ 2 * x1^2 - y1^2 = 4) ∧
  (∃ x2 y2, B = (x2, y2) ∧ 2 * x2^2 - y2^2 = 4) ∧
  (∃ x3 y3, C = (x3, y3) ∧ 2 * x3^2 - y3^2 = 4)

noncomputable def midpoints (A B C M N P : ℝ × ℝ) : Prop :=
  (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
  (N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
  (P = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))

noncomputable def slopes (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 = M.2 / M.1 ∧ k2 = N.2 / N.1 ∧ k3 = P.2 / P.1

noncomputable def sum_of_slopes (A B C : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  ((A.2 - B.2) / (A.1 - B.1) +
   (B.2 - C.2) / (B.1 - C.1) +
   (C.2 - A.2) / (C.1 - A.1)) = -1

theorem geometry_problem 
  (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) 
  (h1 : vertices_on_hyperbola A B C)
  (h2 : midpoints A B C M N P) 
  (h3 : slopes A B C M N P k1 k2 k3) 
  (h4 : sum_of_slopes A B C k1 k2 k3) :
  1/k1 + 1/k2 + 1/k3 = -1 / 2 :=
sorry

end geometry_problem_l51_51874


namespace corey_candies_l51_51996

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end corey_candies_l51_51996


namespace mike_reaches_office_time_l51_51338

-- Define the given conditions
def dave_steps_per_minute : ℕ := 80
def dave_step_length_cm : ℕ := 85
def dave_time_min : ℕ := 20

def mike_steps_per_minute : ℕ := 95
def mike_step_length_cm : ℕ := 70

-- Define Dave's walking speed
def dave_speed_cm_per_min : ℕ := dave_steps_per_minute * dave_step_length_cm

-- Define the total distance to the office
def distance_to_office_cm : ℕ := dave_speed_cm_per_min * dave_time_min

-- Define Mike's walking speed
def mike_speed_cm_per_min : ℕ := mike_steps_per_minute * mike_step_length_cm

-- Define the time it takes Mike to walk to the office
noncomputable def mike_time_to_office_min : ℚ := distance_to_office_cm / mike_speed_cm_per_min

-- State the theorem to prove
theorem mike_reaches_office_time :
  mike_time_to_office_min = 20.45 :=
sorry

end mike_reaches_office_time_l51_51338


namespace area_of_triangle_ABC_is_25_l51_51123

/-- Define the coordinates of points A, B, C given OA and the angle BAC.
    Calculate the area of triangle ABC -/
noncomputable def area_of_triangle_ABC : ℝ :=
  let OA := real.cbrt 50 in
  let A := (OA, 0, 0) in
  let b := 1 in
  let c := 1 in
  let B := (0, b, 0) in
  let C := (0, 0, c) in
  let angle_BAC := real.pi / 4 in
  let AB := real.sqrt ((OA)^2 + (b)^2) in
  let AC := real.sqrt ((OA)^2 + (c)^2) in
  let cos_BAC := real.cos angle_BAC in
  let sin_BAC := real.sin angle_BAC in
  0.5 * AB * AC * sin_BAC

theorem area_of_triangle_ABC_is_25 : area_of_triangle_ABC = 25 :=
by sorry

end area_of_triangle_ABC_is_25_l51_51123


namespace lines_parallel_if_perpendicular_to_common_plane_l51_51431

variables {Line Plane : Type*} 
variables (m n : Line) (α : Plane)
variables [Parallel : ∀ {m n : Line}, Prop]
variables [Perpendicular : ∀ {m α : Line}, Prop]

theorem lines_parallel_if_perpendicular_to_common_plane 
  (h1 : Perpendicular m α) 
  (h2 : Perpendicular n α) : Parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_common_plane_l51_51431


namespace math_club_total_members_l51_51053

   theorem math_club_total_members:
     ∀ (num_females num_males total_members : ℕ),
     num_females = 6 →
     num_males = 2 * num_females →
     total_members = num_females + num_males →
     total_members = 18 :=
   by
     intros num_females num_males total_members
     intros h_females h_males h_total
     rw [h_females, h_males] at h_total
     exact h_total
   
end math_club_total_members_l51_51053


namespace closest_perfect_square_to_350_l51_51673

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l51_51673


namespace maximum_ab_value_l51_51442

noncomputable def max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 6) : ℝ :=
  by
    apply sqrt (max (9 : ℝ) 0)

theorem maximum_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 6) :
  ab = 9 :=
  sorry

end maximum_ab_value_l51_51442


namespace sum_abcd_l51_51106

theorem sum_abcd :
  ∀ (a b c d : ℝ),
  a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7 →
  a + b + c + d = -14 / 3 :=
by
  intros a b c d h
  cases h with h₁ h₂
  sorry

end sum_abcd_l51_51106


namespace polynomial_identity_l51_51339

def d (n : ℤ) : ℕ := -- This should define the biggest prime divisor of |n| > 1
  sorry  -- Actual implementation skipped

def P (n : ℤ) : ℤ := n  -- Given answer P(n) = n

theorem polynomial_identity (n : ℤ) (hn : |n| > 1) (hp : P(n) > 1) :
  P(n + d(n)) = n + d(P(n)) :=
by
  sorry

end polynomial_identity_l51_51339


namespace total_money_earned_l51_51496

theorem total_money_earned (initial_earn: ℕ) (refer_earn: ℕ) (friends_day_one: ℕ) (friends_week: ℕ):
  initial_earn = 5 ∧ refer_earn = 5 ∧ friends_day_one = 5 ∧ friends_week = 7 →
  let katrina_earn := initial_earn + (friends_day_one + friends_week) * refer_earn in
  let friends_earn := (friends_day_one + friends_week) * refer_earn in
  katrina_earn + friends_earn = 125 :=
by
  sorry

end total_money_earned_l51_51496


namespace largest_sum_is_C_l51_51793

def A : ℝ := 2010 / 2009 + 2010 / 2011
def B : ℝ := 2010 / 2011 + 2012 / 2011
def C : ℝ := 2011 / 2010 + 2011 / 2012 + 1 / 2011

theorem largest_sum_is_C : C > A ∧ C > B :=
by {
  sorry
}

end largest_sum_is_C_l51_51793


namespace sin_cos_fraction_eq_two_l51_51847

theorem sin_cos_fraction_eq_two (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 :=
sorry

end sin_cos_fraction_eq_two_l51_51847


namespace cookies_in_box_l51_51010

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end cookies_in_box_l51_51010


namespace whitney_money_leftover_l51_51689

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def posters : ℕ := 2
def notebooks : ℕ := 3
def bookmarks : ℕ := 2

def initial_money : ℕ := 2 * 20

def total_cost : ℕ := posters * poster_cost + notebooks * notebook_cost + bookmarks * bookmark_cost

def money_left_over : ℕ := initial_money - total_cost

theorem whitney_money_leftover : money_left_over = 14 := by
  sorry

end whitney_money_leftover_l51_51689


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51823

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51823


namespace more_sons_or_daughters_prob_l51_51131

-- Probability that Mr. Jones has more sons than daughters or more daughters than sons
theorem more_sons_or_daughters_prob :
  let n := 6 in
  let total_ways := 2 ^ n in
  let ways_equal := Nat.choose n (n / 2) in
  let favorable_ways := total_ways - ways_equal in
  let probability := favorable_ways.toRat / total_ways.toRat in
  probability = 11 / 16 :=
by
  sorry

end more_sons_or_daughters_prob_l51_51131


namespace tangent_line_equation_perpendicular_to_neg_x_l51_51157

theorem tangent_line_equation_perpendicular_to_neg_x :
  ∀ (x : ℝ), 0 < x → (∃ (a b c : ℝ), y = x → y = ln x → y' = 1 → x - y - 1 = 0) :=
by
  sorry

end tangent_line_equation_perpendicular_to_neg_x_l51_51157


namespace find_a_for_tangency_l51_51886

-- Definitions of line and parabola
def line (x y : ℝ) : Prop := x - y - 1 = 0
def parabola (x y : ℝ) (a : ℝ) : Prop := y = a * x^2

-- The tangency condition for quadratic equations
def tangency_condition (a : ℝ) : Prop := 1 - 4 * a = 0

theorem find_a_for_tangency (a : ℝ) :
  (∀ x y, line x y → parabola x y a → tangency_condition a) → a = 1/4 :=
by
  -- Proof omitted
  sorry

end find_a_for_tangency_l51_51886


namespace kolya_cannot_see_full_result_l51_51090

theorem kolya_cannot_see_full_result : let n := 1024 in 
                                      let num_decimals := 16 in 
                                      (num_divisors_product n > 10^num_decimals) := 
begin
    let n := 1024,
    let num_decimals := 16,
    have product := (∑ i in range (nat.log2 n), i),
    have sum_of_powers := (2^product),
    have required_digits := (10^num_decimals),
    exact sum_of_powers > required_digits,
end

end kolya_cannot_see_full_result_l51_51090


namespace total_blue_pairs_correct_l51_51966

def lisa_pairs := 15
def percentage_blue_lisa := 0.60
def blue_pairs_lisa := percentage_blue_lisa * lisa_pairs

def sandra_pairs := 25
def percentage_blue_sandra := 0.40
def blue_pairs_sandra := percentage_blue_sandra * sandra_pairs

def cousin_pairs := (sandra_pairs / 3).floor
def percentage_blue_cousin := 0.50
def blue_pairs_cousin := percentage_blue_cousin * cousin_pairs

def mom_pairs := 4 * lisa_pairs + 12
def percentage_blue_mom := 0.30
def blue_pairs_mom := (percentage_blue_mom * mom_pairs).floor

def total_blue_pairs := blue_pairs_lisa + blue_pairs_sandra + blue_pairs_cousin + blue_pairs_mom

theorem total_blue_pairs_correct : total_blue_pairs = 44 := by
  sorry

end total_blue_pairs_correct_l51_51966


namespace kolya_product_divisors_1024_l51_51084

theorem kolya_product_divisors_1024 :
  let number := 1024
  let screen_limit := 10^16
  ∃ (divisors : List ℕ), 
    (∀ d ∈ divisors, d ∣ number) ∧ 
    (∏ d in divisors, d) = 2^55 ∧ 
    2^55 > screen_limit :=
by
  sorry

end kolya_product_divisors_1024_l51_51084


namespace total_spears_is_78_l51_51968

-- Define the spear production rates for each type of wood
def spears_from_sapling := 3
def spears_from_log := 9
def spears_from_bundle := 7
def spears_from_trunk := 15

-- Define the quantity of each type of wood
def saplings := 6
def logs := 1
def bundles := 3
def trunks := 2

-- Prove that the total number of spears is 78
theorem total_spears_is_78 : (saplings * spears_from_sapling) + (logs * spears_from_log) + (bundles * spears_from_bundle) + (trunks * spears_from_trunk) = 78 :=
by 
  -- Calculation can be filled here
  sorry

end total_spears_is_78_l51_51968


namespace largest_3_digit_sum_l51_51397

theorem largest_3_digit_sum : ∃ A B : ℕ, A ≠ B ∧ A < 10 ∧ B < 10 ∧ 100 ≤ 111 * A + 12 * B ∧ 111 * A + 12 * B = 996 := by
  sorry

end largest_3_digit_sum_l51_51397


namespace quadratic_roots_reciprocal_sum_l51_51853

theorem quadratic_roots_reciprocal_sum :
  ∀ (x₁ x₂ : ℝ), (x₁ ^ 2 - 3 * x₁ + 2 = 0) ∧ (x₂ ^ 2 - 3 * x₂ + 2 = 0) → (x₁ ≠ x₂) → (1 / x₁ + 1 / x₂ = 3 / 2) :=
begin
  intros x₁ x₂ h h_diff,
  sorry
end

end quadratic_roots_reciprocal_sum_l51_51853


namespace probability_at_least_eight_people_stay_l51_51970

theorem probability_at_least_eight_people_stay
  (n : ℕ) (k : ℕ) (p : ℚ) (certain_attendees : ℕ) (unsure_probability : ℚ) :
  n = 9 →
  k = 5 →
  certain_attendees = 4 →
  unsure_probability = 3/7 →
  p = (certain_attendees * (unsure_probability ^ certain_attendees) * (1 - unsure_probability)) * ↑(nat.choose k 4) + (unsure_probability ^ k) →
  p = 2511 / 16807 :=
by
  intros n_eq k_eq certain_attendees_eq unsure_probability_eq p_eq
  sorry

end probability_at_least_eight_people_stay_l51_51970


namespace distinct_triangles_in_regular_ngon_l51_51562

theorem distinct_triangles_in_regular_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, t = n * (n-1) * (n-2) / 6 := 
sorry

end distinct_triangles_in_regular_ngon_l51_51562


namespace sin_alpha_eq_sqrt_5_div_3_l51_51371

variable (α : ℝ)

theorem sin_alpha_eq_sqrt_5_div_3
  (hα : 0 < α ∧ α < Real.pi)
  (h : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := 
by 
  sorry

end sin_alpha_eq_sqrt_5_div_3_l51_51371


namespace z_not_root_of_int_polynomial_l51_51769

noncomputable def z (a : ℕ → ℕ) : ℝ := 
  ∑' m : ℕ, (a m) / 10^(nat.factorial m)

theorem z_not_root_of_int_polynomial (a : ℕ → ℕ) (h : ∀ m, 0 < a m ∧ a m < 10) :
  ¬ ∃ (f : polynomial ℤ), polynomial.eval (z a) f = 0 :=
sorry

end z_not_root_of_int_polynomial_l51_51769


namespace average_of_odd_numbers_l51_51809

theorem average_of_odd_numbers {l : List ℕ} (h1 : l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) :
  let odd_numbers := l.filter (λ x => x % 2 = 1) in
  let odd_numbers_lt_6 := odd_numbers.filter (λ x => x < 6) in
  list.sum odd_numbers_lt_6 / odd_numbers_lt_6.length = 3 :=
by
  have h2 : l.filter (λ x => x % 2 = 1) = [1, 3, 5],
    by simp [h1]
  have h3 : [1, 3, 5].filter (λ x => x < 6) = [1, 3, 5],
    by simp
  simp [list.sum, list.length]
  sorry

end average_of_odd_numbers_l51_51809


namespace problem1_problem2_l51_51995

variable {Ω : Type}

-- Probabilities for the first firing
variables (A1 A2 A3 : Ω → Prop)
variables (P1 : ProbabilisticMeasure Ω) [P1.IsProbabilityMeasure]
variables (PA1 PA2 PA3 : ℝ)

axiom pa1_def : PA1 = 1 / 2
axiom pa2_def : PA2 = 4 / 5
axiom pa3_def : PA3 = 3 / 5

-- Equivalence of the first problem
theorem problem1 :
  P1[A1] = PA1 → P1[A2] = PA2 → P1[A3] = PA3 →
  P1[A1 ∧ ¬A2 ∧ ¬A3] + P1[¬A1 ∧ A2 ∧ ¬A3] + P1[¬A1 ∧ ¬A2 ∧ A3] = 13 / 50 :=
by
  intros h1 h2 h3
  have PA1_not_children := P1[¬A2] * P1[¬A3]
  have P1_children := focused_prob h1 h2 h3
  calc
    P1[A1 ∧ ¬A2 ∧ ¬A3] + P1[¬A1 ∧ A2 ∧ ¬A3] + P1[¬A1 ∧ ¬A2 ∧ A3]
        = 13 / 50 := sorry

-- Probabilities after both firings
variables (A1' A2' A3' : Ω → Prop)
variables (PA1' PA2' PA3' : ℝ)

axiom pa1p_def : PA1' = 4 / 5
axiom pa2p_def : PA2' = 1 / 2
axiom pa3p_def : PA3' = 2 / 3

-- Equivalence of the second problem
theorem problem2 :
  P1[A1 → A1'] = PA1' → P1[A2 → A2'] = PA2' → P1[A3 → A3'] = PA3' →
  let p := 2 / 5 in 
  let n := 3 in
  (n * p = 1.2) :=
by
  intros h1' h2' h3'
  let p := 2 / 5
  let n := 3
  have calc := expected_val lean_exp 
  show EqProp (n * p = 1.2) := sorry

end problem1_problem2_l51_51995


namespace b_car_usage_hours_l51_51268

theorem b_car_usage_hours (h : ℕ) (total_cost_a_b_c : ℕ) 
  (a_usage : ℕ) (b_payment : ℕ) (c_usage : ℕ) 
  (total_cost : total_cost_a_b_c = 720)
  (usage_a : a_usage = 9) 
  (usage_c : c_usage = 13)
  (payment_b : b_payment = 225) 
  (cost_per_hour : ℝ := total_cost_a_b_c / (a_usage + h + c_usage)) :
  b_payment = cost_per_hour * h → h = 10 := 
by
  sorry

end b_car_usage_hours_l51_51268


namespace sum_first_50_terms_l51_51426

noncomputable def sequence_a : ℕ → ℚ
| 0       := 1
| (n + 1) := ((3 * sequence_a n + 4) / (2 * sequence_a n + 3)) - 2

def sequence_b (n : ℕ) : ℚ := (sequence_a n + 1) / 2

def sum_bn_bn1 (n : ℕ) : ℚ :=
∑ i in finset.range n, sequence_b i * sequence_b (i + 1)

theorem sum_first_50_terms :
  sum_bn_bn1 50 = 50 / 201 :=
sorry

end sum_first_50_terms_l51_51426


namespace expected_flips_is_200_a_plus_b_is_201_l51_51067

noncomputable def expected_flips_for_points (n : ℕ) : ℕ := (4 * n + 2)

-- Sum from n = 0 to n = 9 of expected_flips_for_points
def expected_flips_10_points : ℕ := finset.sum (finset.range 10) expected_flips_for_points

theorem expected_flips_is_200 : expected_flips_10_points = 200 :=
  by
  sorry

-- a = 200, b = 1
def a_b_sum : ℕ := 200 + 1

theorem a_plus_b_is_201 : a_b_sum = 201 :=
  by
  sorry

end expected_flips_is_200_a_plus_b_is_201_l51_51067


namespace geometric_sequence_b_sum_first_n_terms_nb_l51_51597

-- Definition: sum of first n terms S_n and requirement S_n + a_n
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n + a n = -0.5 * n^2 - 1.5 * n + 1

-- Definition: b_n = a_n + n
def b_sequence (a b : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, b n = a n + n

-- Problem (1): Prove {b_n} is geometric if sum condition holds
theorem geometric_sequence_b {a b : ℕ → ℝ} {S : ℕ → ℝ} (h1 : sum_of_first_n_terms a S) (h2 : b_sequence a b) : 
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) = r * b n := 
sorry

-- Problem (2): Prove T_n = 2 - (n + 2) / 2^n
def sum_nb (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = (finset.range n).sum (λ k, (k+1) * b (k+1))

theorem sum_first_n_terms_nb {a b : ℕ → ℝ} {S : ℕ → ℝ} {T : ℕ → ℝ}
  (h1 : sum_of_first_n_terms a S) (h2 : b_sequence a b) (h3 : ∀ n : ℕ, n > 0 → b n = 1 / 2^n) :
  ∀ n : ℕ, T n = 2 - (n + 2) / (2^n) :=
sorry

end geometric_sequence_b_sum_first_n_terms_nb_l51_51597


namespace inequality_proof_l51_51827

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51827


namespace main_problem_l51_51888

def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)

def p : Prop := ∀ (x y : ℝ), x < y → f x < f y

def q : Prop := ∃ x ∈ set.univ ℝ, x^2 - x + 2 < 0

theorem main_problem : p ∧ ¬q :=
by
  sorry

end main_problem_l51_51888


namespace rick_books_division_l51_51982

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l51_51982


namespace beetle_can_return_l51_51278

-- We define the grid and the beetle's starting cell
structure Grid (Cell : Type) :=
  (adjacent : Cell → Cell → Prop) -- adjacency relation

structure Beetle (Cell : Type) :=
  (start : Cell) -- starting cell

-- Define the conditions: initially closed doors, bug opens doors, doors stay open as described
structure Door (Cell : Type) :=
  (open : Cell → Cell → Prop) -- door open in one direction

-- main theorem proving the beetle can return to its starting cell eventually
theorem beetle_can_return {Cell : Type} (G : Grid Cell) (B : Beetle Cell) (D : Door Cell) :
  (∀ c c', ¬ D.open c c') → -- initially all doors are closed
  (∀ c c', G.adjacent c c' → (¬ D.open c c') → (D.open c c')) → -- opening a door in the direction of movement
  (∀ c c', D.open c c' ↔ ¬ D.open c' c) → -- door remains open in the direction of movement
  (∀ c, c = B.start ∨ ∃ p, G.adjacent c p ∧ D.open p c) → -- the bug can only pass through an open door in the direction it was opened
  ∃ p, p = B.start := -- the beetle can eventually return to its starting cell
sorry

end beetle_can_return_l51_51278


namespace cost_difference_is_35_l51_51695

noncomputable theory

def costX := 1.20
def costY := 1.70
def n := 70

theorem cost_difference_is_35 :
  (n * costY) - (n * costX) = 35 := sorry

end cost_difference_is_35_l51_51695


namespace min_value_l51_51173

theorem min_value (a m n : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) (hm : m > 0) (hn : n > 0)
  (hA : 1 = 2 * m + n) : (∃ x : ℝ, x = 2) ∧ (y = log a (x - 1) + 1) →
  (∃ y : ℝ, y = 4 + (n / m) + (4 * m / n) → y = 8) := 
sorry

end min_value_l51_51173


namespace probability_multiple_of_4_l51_51491

-- Defining the conditions of the problem.
def card_range : finset ℕ := finset.range 15
def spinner_probs : finset ℚ := { 1/3, 2/3 }

-- Function to compute the final position on the number line.
def move (start: ℕ) (spin: ℕ) : ℕ :=
  match spin with
  | 0 => start - 1  -- move 1 space left
  | 1 => start + 2  -- move 2 spaces right
  | 2 => start + 2  -- move 2 spaces right
  | _ => start      -- not used due to fixed spin values

def final_position (start: ℕ) (spins: list ℕ) : ℕ :=
  spins.foldl move start

def is_multiple_of_4 (n: ℕ) : Prop := 
  n % 4 = 0

-- Main theorem: Probability of final position being a multiple of 4.
theorem probability_multiple_of_4 : 
  (∑ start in card_range, 
    (1/15) * 
    (∑ s1 in spinner_probs, 
      (∑ s2 in spinner_probs, 
        if is_multiple_of_4 (final_position start [s1, s2]) 
        then 1 else 0))) / 
  (card_range.card * spinner_probs.card * spinner_probs.card) = 7/27 := 
by sorry

end probability_multiple_of_4_l51_51491


namespace curve_cartesian_equation_chord_length_l51_51059
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * θ.cos, ρ * θ.sin)

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + 1/2 * t, (Real.sqrt 3) / 2 * t)

theorem curve_cartesian_equation :
  ∀ (ρ θ : ℝ), 
    ρ * θ.sin * θ.sin = 8 * θ.cos →
    (ρ * θ.cos) ^ 2 + (ρ * θ.sin) ^ 2 = 
    8 * (ρ * θ.cos) :=
by sorry

theorem chord_length :
  ∀ (t₁ t₂ : ℝ),
    (3 * t₁^2 - 16 * t₁ - 64 = 0) →
    (3 * t₂^2 - 16 * t₂ - 64 = 0) →
    |t₁ - t₂| = (32 / 3) :=
by sorry

end curve_cartesian_equation_chord_length_l51_51059


namespace total_annual_income_percentage_l51_51324

-- Definitions for the conditions
def investment1 : ℝ := 2400
def rate1 : ℝ := 0.05
def investment2 : ℝ := 599.9999999999999  -- approximately 600
def rate2 : ℝ := 0.10
def total_investment : ℝ := investment1 + investment2
def annual_income1 : ℝ := investment1 * rate1
def annual_income2 : ℝ := investment2 * rate2
def total_annual_income : ℝ := annual_income1 + annual_income2
def percentage (income : ℝ) (investment : ℝ) : ℝ := (income / investment) * 100

-- The statement to prove
theorem total_annual_income_percentage : percentage total_annual_income total_investment = 6 := 
by
  sorry

end total_annual_income_percentage_l51_51324


namespace value_of_T_l51_51213

-- Define the main variables and conditions
variables {M T : ℝ}

-- State the conditions given in the problem
def condition1 (M T : ℝ) := 2 * M + T = 7000
def condition2 (M T : ℝ) := M + 2 * T = 9800

-- State the theorem to be proved
theorem value_of_T : 
  ∀ (M T : ℝ), condition1 M T ∧ condition2 M T → T = 4200 :=
by 
  -- Proof would go here; for now, we use "sorry" to skip it
  sorry

end value_of_T_l51_51213


namespace find_a_tangent_slope_at_point_l51_51869

theorem find_a_tangent_slope_at_point :
  ∃ (a : ℝ), (∃ (y : ℝ), y = (fun (x : ℝ) => x^4 + a * x^2 + 1) (-1) ∧ (∃ (y' : ℝ), y' = (fun (x : ℝ) => 4 * x^3 + 2 * a * x) (-1) ∧ y' = 8)) ∧ a = -6 :=
by
  -- Used to skip the proof
  sorry

end find_a_tangent_slope_at_point_l51_51869


namespace rowing_time_ratio_l51_51729

-- Let Vm be the speed of man in still water
def Vm := 4.5

-- Let Vc be the speed of current
def Vc := 1.5

-- Effective speed upstream
def Vu := Vm - Vc

-- Effective speed downstream
def Vd := Vm + Vc

-- Time to row upstream
def Tu (D : ℝ) := D / Vu

-- Time to row downstream
def Td (D : ℝ) := D / Vd

-- Main theorem stating the ratio of time to row upstream to downstream is 2:1
theorem rowing_time_ratio (D : ℝ) (h : D > 0) : Tu D / Td D = 2 := by
  sorry

end rowing_time_ratio_l51_51729


namespace calc_f_five_times_l51_51523

def f (x : ℕ) : ℕ := if x % 2 = 0 then x / 2 else 5 * x + 1

theorem calc_f_five_times : f (f (f (f (f 5)))) = 166 :=
by 
  sorry

end calc_f_five_times_l51_51523


namespace sum_of_distances_inequality_l51_51101

variables (A B C S A1 B1 C1 : Type) [MetricSpace S]

-- Definition of the incenter and the intersections with the circumcircle
axiom is_incenter : ∀ {A B C S}, incenter S A B C
axiom on_circumcircle : ∀ {A B C Ai Si}, intersect_circumcircle Ai Si A B C

-- Statement of the inequality we need to prove
theorem sum_of_distances_inequality 
  (h_incenter : is_incenter S)
  (h_A1 : on_circumcircle A1 (line_through A S) A B C)
  (h_B1 : on_circumcircle B1 (line_through B S) A B C)
  (h_C1 : on_circumcircle C1 (line_through C S) A B C) :
  dist S A1 + dist S B1 + dist S C1 ≥ dist S A + dist S B + dist S C := 
sorry

end sum_of_distances_inequality_l51_51101


namespace rick_group_division_l51_51980

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l51_51980


namespace factor_expression_l51_51347

theorem factor_expression (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) :=
  by
    sorry

end factor_expression_l51_51347


namespace function_decreasing_on_interval_l51_51960

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - 12 * x + b

theorem function_decreasing_on_interval (b : ℝ) :
  ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (f x b) ≤ (f (-2) b) :=
begin
  sorry
end

end function_decreasing_on_interval_l51_51960


namespace sum_reciprocals_no_zero_digits_leq_90_l51_51783

-- Define a predicate to check if a number's decimal representation has no zero digits
def no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 n) → d ≠ 0

-- Prove that the sum of the reciprocals of any finite set of natural numbers without zero digits does not exceed 90
theorem sum_reciprocals_no_zero_digits_leq_90 (M : Finset ℕ) (hM : ∀ n ∈ M, no_zero_digits n) :
  (∑ n in M, (1 : ℝ) / n) ≤ 90 := 
sorry

end sum_reciprocals_no_zero_digits_leq_90_l51_51783


namespace set_intersection_l51_51429

variable (R : Type) [LinearOrderedField R]

def U : Set R := Set.univ
def A : Set R := { x : R | x > -1 }
def B : Set R := { x : R | x > 2 }
def CuB : Set R := { x : R | x ≤ 2 }

theorem set_intersection :
  A ∩ CuB = { x : R | -1 < x ∧ x ≤ 2 } :=
by sorry

end set_intersection_l51_51429


namespace original_bananas_total_l51_51692

theorem original_bananas_total (willie_bananas : ℝ) (charles_bananas : ℝ) : willie_bananas = 48.0 → charles_bananas = 35.0 → willie_bananas + charles_bananas = 83.0 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end original_bananas_total_l51_51692


namespace neg_proposition_equiv_exists_gt_one_l51_51425

def proposition (x : ℝ) : Prop := cos x ≤ 1

theorem neg_proposition_equiv_exists_gt_one :
  ¬ (∀ x : ℝ, proposition x) ↔ ∃ x : ℝ, cos x > 1 :=
by
  sorry

end neg_proposition_equiv_exists_gt_one_l51_51425


namespace math_club_total_members_l51_51051

   theorem math_club_total_members:
     ∀ (num_females num_males total_members : ℕ),
     num_females = 6 →
     num_males = 2 * num_females →
     total_members = num_females + num_males →
     total_members = 18 :=
   by
     intros num_females num_males total_members
     intros h_females h_males h_total
     rw [h_females, h_males] at h_total
     exact h_total
   
end math_club_total_members_l51_51051


namespace dice_probability_sum_ten_l51_51312

theorem dice_probability_sum_ten : 
  (∃ n : ℕ, (∑ i in (finset.range 7), x i = 10) → (P sum_10 = n / 6^7)) → (n = 84) :=
sorry

end dice_probability_sum_ten_l51_51312


namespace sum_of_abc_is_12_l51_51121

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end sum_of_abc_is_12_l51_51121


namespace slope_of_line_with_angle_120_deg_is_neg_sqrt_3_l51_51908

theorem slope_of_line_with_angle_120_deg_is_neg_sqrt_3 :
    ∀ (θ : ℝ), θ = 120 → Real.tan θ = -Real.sqrt 3 :=
by
  assume θ h
  sorry

end slope_of_line_with_angle_120_deg_is_neg_sqrt_3_l51_51908


namespace annual_rent_per_square_foot_l51_51698

theorem annual_rent_per_square_foot (length width : ℕ) (monthly_rent : ℕ)
  (h_length : length = 20) (h_width : width = 15) (h_monthly_rent : monthly_rent = 3600) :
  let area := length * width
  let annual_rent := monthly_rent * 12
  let annual_rent_per_sq_ft := annual_rent / area
  annual_rent_per_sq_ft = 144 := by
  sorry

end annual_rent_per_square_foot_l51_51698


namespace ratio_of_sides_of_rectangles_l51_51468

theorem ratio_of_sides_of_rectangles (s x y : ℝ) 
  (hsx : x + s = 2 * s) 
  (hsy : s + 2 * y = 2 * s)
  (houter_inner_area : (2 * s) ^ 2 = 4 * s ^ 2) : 
  x / y = 2 :=
by
  -- Assuming the conditions hold, we are interested in proving that the ratio x / y = 2
  -- The proof will be provided here
  sorry

end ratio_of_sides_of_rectangles_l51_51468


namespace line_intersects_y_axis_l51_51323

theorem line_intersects_y_axis :
  ∃ y : ℝ, (2 * y - 5 * 0 = 10) ∧ (0, y) = (0, 5) :=
begin
  use 5,
  split,
  {
    calc 2 * 5 - 5 * 0 = 2 * 5 - 0  : by ring
                 ...  = 10          : by norm_num,
  },
  refl,
end

end line_intersects_y_axis_l51_51323


namespace number_of_triangles_l51_51262

theorem number_of_triangles :
  let lengths : set ℕ := {1, 2, 3, 4, 5},
      valid_triangle (a b c : ℕ) : Prop :=
        a + b > c ∧ b + c > a ∧ c + a > b in
  {comb | ∃ (a b c : ℕ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ a ≤ b ∧ b ≤ c ∧ valid_triangle a b c}.card = 22 := sorry

end number_of_triangles_l51_51262


namespace alpha_minus_beta_eq_pi_div_4_l51_51864

open Real

theorem alpha_minus_beta_eq_pi_div_4 (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 4) 
(h : tan α = (cos β + sin β) / (cos β - sin β)) : α - β = π / 4 :=
sorry

end alpha_minus_beta_eq_pi_div_4_l51_51864


namespace minimum_days_bacteria_count_exceeds_500_l51_51043

theorem minimum_days_bacteria_count_exceeds_500 :
  ∃ n : ℕ, 4 * 3^n > 500 ∧ ∀ m : ℕ, m < n → 4 * 3^m ≤ 500 :=
by
  sorry

end minimum_days_bacteria_count_exceeds_500_l51_51043


namespace tank_capacity_ratio_l51_51272

noncomputable def volume (h : ℝ) (c : ℝ) : ℝ := 
  let r := c / (2 * Real.pi)
  Real.pi * r^2 * h

theorem tank_capacity_ratio (hA hB cA cB : ℝ)
  (hA_eq : hA = 7) (cA_eq : cA = 8)
  (hB_eq : hB = 8) (cB_eq : cB = 10) :
  (volume hA cA / volume hB cB) * 100 = 56 :=
by 
  have rA : ℝ := cA / (2 * Real.pi),
  have rB : ℝ := cB / (2 * Real.pi),
  have vA : ℝ := Real.pi * rA^2 * hA,
  have vB : ℝ := Real.pi * rB^2 * hB,
  rw [hA_eq, hB_eq, cA_eq, cB_eq] at rA rB,
  simp [volume, rA, rB, vA, vB],
  norm_num,
  sorry

end tank_capacity_ratio_l51_51272


namespace gulliver_total_kefir_bottles_l51_51892

theorem gulliver_total_kefir_bottles :
  let initial_rubles := 7000000
  let kefir_cost_initial := 7
  let initial_bottles := initial_rubles / kefir_cost_initial
  let second_round_bottles := initial_bottles / 2
  let third_round_bottles := second_round_bottles / 2
  let sum_geometric_series := initial_bottles / (1 - (1 / 2))
  in sum_geometric_series = 2000000 := 
  sorry

end gulliver_total_kefir_bottles_l51_51892


namespace john_paid_l51_51939

def price_of_dress_shirts : ℝ := 25
def price_of_pants : ℝ := 35
def number_of_dress_shirts : ℝ := 4
def number_of_pants : ℝ := 2
def discount_dress_shirts : ℝ := 0.15
def discount_pants : ℝ := 0.20
def tax : ℝ := 0.10

def total_paid : ℝ :=
  let cost_of_dress_shirts := number_of_dress_shirts * price_of_dress_shirts in
  let discount_on_dress_shirts := discount_dress_shirts * cost_of_dress_shirts in
  let cost_after_discount_dress_shirts := cost_of_dress_shirts - discount_on_dress_shirts in
  let cost_of_pants := number_of_pants * price_of_pants in
  let discount_on_pants := discount_pants * cost_of_pants in
  let cost_after_discount_pants := cost_of_pants - discount_on_pants in
  let subtotal := cost_after_discount_dress_shirts + cost_after_discount_pants in
  let tax_amount := tax * subtotal in
  subtotal + tax_amount

theorem john_paid : total_paid = 155.10 := by
  sorry

end john_paid_l51_51939


namespace calc_dot_product_l51_51008

noncomputable def vector_dot_product (c d : ℝ) (φ : ℝ) : ℝ :=
  c * d * Real.cos φ

theorem calc_dot_product :
  let c_norm := 8
  let d_norm := 12
  let φ := Real.pi / 4
  vector_dot_product c_norm d_norm φ = 48 * Real.sqrt 2 :=
by
  simp [vector_dot_product, Real.cos, Real.sqrt]
  sorry

end calc_dot_product_l51_51008


namespace total_distance_proof_l51_51313

-- Definitions for distances covered in each segment
def distance_flat (speed_flat time_flat : ℝ) : ℝ := speed_flat * time_flat
def distance_uphill (speed_uphill time_uphill : ℝ) : ℝ := speed_uphill * time_uphill
def distance_downhill (speed_downhill time_downhill : ℝ) : ℝ := speed_downhill * time_downhill
def distance_walking (walking_distance : ℝ) : ℝ := walking_distance

-- Given conditions
def speed_flat := 20 -- in miles per hour
def time_flat := 4.5 -- in hours
def speed_uphill := 12 -- in miles per hour
def time_uphill := 2.5 -- in hours
def speed_downhill := 24 -- in miles per hour
def time_downhill := 1.5 -- in hours
def walking_distance := 8 -- in miles

-- Proof problem statement
theorem total_distance_proof : 
  distance_flat speed_flat time_flat + distance_uphill speed_uphill time_uphill + distance_downhill speed_downhill time_downhill + distance_walking walking_distance = 164 :=
by
  sorry

end total_distance_proof_l51_51313


namespace midpoint_of_five_points_on_grid_l51_51971

theorem midpoint_of_five_points_on_grid 
    (points : Fin 5 → ℤ × ℤ) :
    ∃ i j : Fin 5, i ≠ j ∧ ((points i).fst + (points j).fst) % 2 = 0 
    ∧ ((points i).snd + (points j).snd) % 2 = 0 :=
by sorry

end midpoint_of_five_points_on_grid_l51_51971


namespace estimated_number_of_fish_l51_51544

variable {x : ℕ}
variable (initial_catch_marked : ℕ := 100)
variable (second_catch_total : ℕ := 100)
variable (second_catch_marked : ℕ := 2)

theorem estimated_number_of_fish : x ≈ 5000 :=
  by
  -- Definition of the probabilities
  have prob_first_catch := initial_catch_marked / (x : ℝ)
  have prob_second_catch := second_catch_marked / (second_catch_total : ℝ)
  
  -- Relationship between the probabilities
  have probability_relation : prob_second_catch = initial_catch_marked / (x : ℝ) := by
    sorry

  -- Solve for x
  have solution_for_x : x = 5000 := by
    sorry

  -- Conclusion
  exact solution_for_x

end estimated_number_of_fish_l51_51544


namespace common_divisors_count_l51_51897

def prime_exponents (n : Nat) : List (Nat × Nat) :=
  if n = 9240 then [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
  else if n = 10800 then [(2, 4), (3, 3), (5, 2)]
  else []

def gcd_prime_exponents (exps1 exps2 : List (Nat × Nat)) : List (Nat × Nat) :=
  exps1.filterMap (fun (p1, e1) =>
    match exps2.find? (fun (p2, _) => p1 = p2) with
    | some (p2, e2) => if e1 ≤ e2 then some (p1, e1) else some (p1, e2)
    | none => none
  )

def count_divisors (exps : List (Nat × Nat)) : Nat :=
  exps.foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem common_divisors_count :
  count_divisors (gcd_prime_exponents (prime_exponents 9240) (prime_exponents 10800)) = 16 :=
by
  sorry

end common_divisors_count_l51_51897


namespace final_result_l51_51389

variable {α : Type*}

def sequence : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := if h : (sequence n) * (sequence (n + 1)) ≠ 1 then (sequence n) + (sequence (n + 1)) else 0

axiom recurrence_relation (n : ℕ) : 
  (sequence n) * (sequence (n + 1)) * (sequence (n + 2)) = (sequence n) + (sequence (n + 1)) + (sequence (n + 2))

axiom condition_non_prod_eq_1 (n : ℕ) : 
  (sequence n) * (sequence (n + 1)) ≠ 1

def S2012 : ℕ := 
  670 * (sequence 0 + sequence 1 + (sequence 2)) + sequence 0 + sequence 1

theorem final_result : S2012 = 4023 := by
  -- proof goes here
  sorry

end final_result_l51_51389


namespace dog_bones_initial_count_l51_51539

theorem dog_bones_initial_count (buried : ℝ) (final : ℝ) : buried = 367.5 → final = -860 → (buried + (final + 367.5) + 860) = 367.5 :=
by
  intros h1 h2
  sorry

end dog_bones_initial_count_l51_51539


namespace factorial_product_inequality_l51_51280

theorem factorial_product_inequality (N : ℕ) (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ)
    (sum_a : (∑ i, a i) = N) (sum_b : (∑ j, b j) = N) :
    (∏ i, (a i)!) * (∏ j, (b j)!) ≤ N! := 
by
  sorry

end factorial_product_inequality_l51_51280


namespace distance_at_true_anomaly_90_l51_51182

theorem distance_at_true_anomaly_90 
  (perigee distance from sun : ℝ)
  (apogee distance from sun : ℝ)
  (true anomaly : ℝ)
  (a : ℝ)
  (c : ℝ)
  (b : ℝ)
  (PF : ℝ) : 
  perigee distance from sun = 5 → 
  apogee distance from sun = 20 → 
  true anomaly = 90 → 
  a = (perigee distance from sun + apogee distance from sun) / 2 → 
  c = a - perigee distance from sun → 
  b = sqrt (a^2 - c^2) → 
  PF = sqrt (b^2 + c^2) → 
  PF = 12.5 := 
sorry

end distance_at_true_anomaly_90_l51_51182


namespace find_integer_n_l51_51613

theorem find_integer_n (n : ℤ) : (0 ≤ n ∧ n < 103) ∧ (99 * n ≡ 65 [MOD 103]) → (n ≡ 68 [MOD 103]) :=
by
  sorry

end find_integer_n_l51_51613


namespace expected_value_is_correct_l51_51702

-- Define the probability density function (pdf) of the random variable X
def pdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 2 then (3 * x^2) / 8
  else 0

-- Define the expected value (expectation) of the random variable X
def expected_value := ∫ (x : ℝ) in 0..2, x * pdf x

-- State the theorem that the expected value of X is 1.5
theorem expected_value_is_correct : expected_value = 1.5 := by sorry

end expected_value_is_correct_l51_51702


namespace not_lt_neg_version_l51_51259

theorem not_lt_neg_version (a b : ℝ) (h : a < b) : ¬ (-3 * a < -3 * b) :=
by 
  -- This is where the proof would go
  sorry

end not_lt_neg_version_l51_51259


namespace solve_modular_equation_l51_51153

theorem solve_modular_equation (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 ↔ x % 6 = 1 % 6 := by
  sorry

end solve_modular_equation_l51_51153


namespace find_minimal_product_l51_51758

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end find_minimal_product_l51_51758


namespace men_in_second_group_l51_51711

theorem men_in_second_group (M : ℕ) (h1 : 14 * 25 = 350) (h2 :  M * 17.5 = 350) : M = 20 :=
sorry

end men_in_second_group_l51_51711


namespace S2_side_length_656_l51_51565

noncomputable def S1_S2_S3_side_lengths (l1 l2 a b c : ℕ) (total_length : ℕ) : Prop :=
  l1 + l2 + a + b + c = total_length

theorem S2_side_length_656 :
  ∃ (l1 l2 a c : ℕ), S1_S2_S3_side_lengths l1 l2 a 656 c 3322 :=
by
  sorry

end S2_side_length_656_l51_51565


namespace paint_cost_per_quart_l51_51031

theorem paint_cost_per_quart
  (total_cost : ℝ)
  (coverage_per_quart : ℝ)
  (side_length : ℝ)
  (cost_per_quart : ℝ) 
  (h1 : total_cost = 192)
  (h2 : coverage_per_quart = 10)
  (h3 : side_length = 10) 
  (h4 : cost_per_quart = total_cost / ((6 * side_length ^ 2) / coverage_per_quart))
  : cost_per_quart = 3.20 := 
by 
  sorry

end paint_cost_per_quart_l51_51031


namespace inequality_proof_l51_51838

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51838


namespace consecutive_sum_divisible_l51_51107

/-- Given n arbitrary numbers, there exist consecutive numbers whose sum is divisible by n -/
theorem consecutive_sum_divisible {n : ℕ} (h : n ≥ 1) (a : ℕ → ℤ) :
  ∃ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n ∧ (∑ k in finset.range (j - i + 1), a (i + k)) % n = 0 := 
sorry

end consecutive_sum_divisible_l51_51107


namespace number_approximation_l51_51600

-- Definitions based on the conditions
def number (d : ℝ) : ℝ := 4.9 * 9.97 / 55

-- Lean 4 statement of the problem, we prove that the number with given conditions is approximately 0.9.
theorem number_approximation : |number 4.9 - 0.9| < 0.1 :=
by
  sorry

end number_approximation_l51_51600


namespace geometric_sequence_ratio_l51_51191

theorem geometric_sequence_ratio (a₁ : ℝ) : 
  let q := 3 in 
  let a₄ := a₁ * q^(4 - 1) in 
  let S₄ := a₁ * (1 - q^4) / (1 - q) in 
  S₄ / a₄ = 40/27 :=
by
  let q := 3
  let a₄ := a₁ * q^(4 - 1)
  let S₄ := a₁ * (1 - q^4) / (1 - q)
  show S₄ / a₄ = 40 / 27
  sorry

end geometric_sequence_ratio_l51_51191


namespace arctan_double_inequality_l51_51143

open Real

theorem arctan_double_inequality (t : ℝ) (h : t > 0) :
  ∀ n : ℕ,
    (t - ∑ i in finset.range n, -1^(i+1) * t^(2*i + 1) / (2*i + 1)) <
    arctan t ∧
    arctan t <
    (t - ∑ i in finset.range n, -1^i * t^(2*i + 1) / (2*i + 1)) :=
sorry

end arctan_double_inequality_l51_51143


namespace closest_perfect_square_to_350_l51_51650

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l51_51650


namespace find_OC_l51_51556

theorem find_OC (O A B C : Point) (h_orthocenter : Orthocenter O A B C) (h_AB : dist A B = 4)
  (h_sinC : sin (angle B A C) = 5 / 13) : dist O C = 9.6 :=
by
  sorry

end find_OC_l51_51556


namespace sum_f_values_l51_51381

noncomputable def f : ℝ → ℝ := sorry

def non_decreasing (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ x1 x2 ∈ D, x1 < x2 → f x1 ≤ f x2

def interval_0_1 := set.Icc 0 1

axiom f_non_decreasing : non_decreasing f interval_0_1
axiom f_at_0 : f 0 = 0
axiom f_scaling : ∀ x ∈ interval_0_1, f (x / 3) = (1 / 2) * f x
axiom f_symmetric : ∀ x ∈ interval_0_1, f (1 - x) = 1 - f x

theorem sum_f_values :
  f 1 + f (1/2) + f (1/3) + f (1/6) + f (1/7) + f (1/8) = 11 / 4 :=
sorry

end sum_f_values_l51_51381


namespace journalist_selection_ways_l51_51771

def num_journalists_selection_ways : ℕ := 260

theorem journalist_selection_ways 
  (d_j : ℕ) (f_j : ℕ)
  (d_choose_2 : ∀ n : ℕ, ℕ) 
  (f_choose_1 : ∀ n : ℕ, ℕ) 
  (d_choose_1 : ∀ n : ℕ, ℕ) 
  (f_choose_2 : ∀ n : ℕ, ℕ) 
  (arrange_2 : ℕ → ℕ) 
  (arrange_3 : ℕ → ℕ) 
  (condition_not_consecutive : bool) :
  d_j = 5 ∧ f_j = 4 ∧ condition_not_consecutive = true →
  (d_choose_2 d_j * f_choose_1 f_j * arrange_2 2) + 
  (d_choose_1 d_j * f_choose_2 f_j * arrange_3 3) = num_journalists_selection_ways := by
  sorry

end journalist_selection_ways_l51_51771


namespace evaluate_expression_l51_51193

theorem evaluate_expression : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := 
sorry

end evaluate_expression_l51_51193


namespace nonneg_solutions_eq_zero_l51_51435

theorem nonneg_solutions_eq_zero :
  (∀ x : ℝ, x^2 + 6 * x + 9 = 0 → x ≥ 0) = 0 :=
by
  sorry

end nonneg_solutions_eq_zero_l51_51435


namespace percentage_increase_lift_10m_l51_51935

-- Definitions for the conditions
def initial_weight_20m := 300
def weight_increase_20m := 50
def final_weight_20m := initial_weight_20m + weight_increase_20m
def straps_increase := 1.20
def weight_with_straps_10m := 546

-- Definition of the proof problem statement
theorem percentage_increase_lift_10m :
  (final_weight_20m * (1 + percentage_increase_lift_10m) * straps_increase = weight_with_straps_10m) → 
  1 + percentage_increase_lift_10m = 1.3 :=
sorry

end percentage_increase_lift_10m_l51_51935


namespace text_messages_not_intended_for_john_l51_51493

theorem text_messages_not_intended_for_john 
    (texts_per_day_before : ℕ)
    (texts_per_day_now : ℕ)
    (days_per_week : ℕ)
    (texts_per_day_from_friends_no_change : texts_per_day_before) :
    texts_per_day_before = 20 →
    texts_per_day_now = 55 →
    days_per_week = 7 →
    (texts_per_day_now - texts_per_day_before) * days_per_week = 245 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end text_messages_not_intended_for_john_l51_51493


namespace sector_angle_maximized_l51_51741

noncomputable def central_angle_maximized (r : ℝ) := (20 - 2 * r) / r

noncomputable def sector_area (r : ℝ) (θ : ℝ) := (1 / 2) * r^2 * θ

theorem sector_angle_maximized :
  ∀ r θ : ℝ, 2 * r + r * θ = 20 → 
             θ = central_angle_maximized r → 
             ∃ (area_deriv : ℝ), 
               area_deriv = deriv (λ r, sector_area r (central_angle_maximized r)) r ∧ 
               area_deriv = 0 → 
               θ = 2 :=
by
  intros r θ h1 h2
  sorry

end sector_angle_maximized_l51_51741


namespace phi_range_for_function_l51_51877

theorem phi_range_for_function (
  f : ℝ → ℝ,
  φ : ℝ,
  h1 : ∀ x, f x = 2 * Real.sin (2 * x + φ) + 1,
  h2 : ∀ x, x ∈ Set.Ioc (-π / 12) (π / 3) → f x > 1
) : φ ∈ Set.Icc (π / 6) (π / 3) :=
sorry

end phi_range_for_function_l51_51877


namespace smaller_cube_volume_l51_51725

theorem smaller_cube_volume
  (V_L : ℝ) (N : ℝ) (SA_diff : ℝ) 
  (h1 : V_L = 8)
  (h2 : N = 8)
  (h3 : SA_diff = 24) :
  (∀ V_S : ℝ, V_L = N * V_S → V_S = 1) :=
by
  sorry

end smaller_cube_volume_l51_51725


namespace arc_length_of_circle_l51_51380

theorem arc_length_of_circle (r : ℝ) (alpha : ℝ) (h_r : r = 10) (h_alpha : alpha = (2 * Real.pi) / 6) : 
  (alpha * r) = (10 * Real.pi) / 3 :=
by
  rw [h_r, h_alpha]
  sorry

end arc_length_of_circle_l51_51380


namespace find_a_find_extreme_points_l51_51883

def f (a b x : ℝ) : ℝ := a * x^2 + b * (Real.log x - x)

theorem find_a (a b : ℝ) 
  (h_tangent_perpendicular : ∀ (f' : ℝ → ℝ), f' = λ x, 2*a*x + b/x - b →
    f'(1) = -1) : a = -1/2 :=
by
  have : (λ x, 2*a*x + b/x - b) 1 = -1,
  { exact h_tangent_perpendicular (λ x, 2*a*x + b/x - b) rfl },
  have h : 2*a - b = -1 := this,
  linarith

theorem find_extreme_points (b : ℝ) :
  let f := λ x, -1/2 * x^2 + b * (Real.log x - x) in
  if -4 ≤ b ∧ b ≤ 0 then
    ∀ x, ¬ (0 < x ∧ x < +∞ ∧ (f' x = 0))
  else if b < -4 then
    ∃ x₁ x₂, 
      x₁ = -b/2 - (Real.sqrt (b^2 + 4*b))/2 ∧
      x₂ = -b/2 + (Real.sqrt (b^2 + 4*b))/2 ∧
      0 < x₁ ∧ x₁ < x₂ ∧
      ∀ x, (0 < x ∧ x < x₁ → f' x < 0) ∧
            (x₁ < x ∧ x < x₂ → f' x > 0) ∧
            (x₂ < x → f' x < 0)
  else
    ∃ x₂,
      x₂ = -b/2 + (Real.sqrt (b^2 + 4*b))/2 ∧
      ∀ x, (0 < x ∧ x < x₂ → f' x > 0) ∧
            (x₂ < x → f' x < 0)
  :=
by
  intros,
  sorry

end find_a_find_extreme_points_l51_51883


namespace find_A_minus_B_l51_51502

def A : ℤ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℤ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem find_A_minus_B : A - B = 128 := 
by
  -- Proof goes here
  sorry

end find_A_minus_B_l51_51502


namespace gold_distribution_possible_l51_51276

theorem gold_distribution_possible
  (n : ℕ) 
  (h : n = 100)
  (net_gain : Fin n → ℤ)
  (h_sum : ∑ i, net_gain i = 0)
  (h_non_negative : ∀ i, net_gain i ≥ -net_gain i) :
  ∃ (final_balances : Fin n → ℤ), 
  (∀ i, final_balances i = net_gain i) :=
sorry

end gold_distribution_possible_l51_51276


namespace max_area_triangle_l51_51450

/-- Given two fixed points A and B on the plane with distance 2 between them, 
and a point P moving such that the ratio of distances |PA| / |PB| = sqrt(2), 
prove that the maximum area of triangle PAB is 2 * sqrt(2). -/
theorem max_area_triangle 
  (A B P : EuclideanSpace ℝ (Fin 2)) 
  (hAB : dist A B = 2)
  (h_ratio : dist P A = Real.sqrt 2 * dist P B)
  (h_non_collinear : ¬ ∃ k : ℝ, ∃ l : ℝ, k ≠ l ∧ A = k • B ∧ P = l • B) 
  : ∃ S_max : ℝ, S_max = 2 * Real.sqrt 2 := 
sorry

end max_area_triangle_l51_51450


namespace inequality_proof_l51_51835

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51835


namespace cube_edge_length_l51_51402

noncomputable def length_of_cube_edge (v : ℝ) (pi := Real.pi) : ℝ :=
  let a := (sqrt 3) / 2 in a

theorem cube_edge_length (a v : ℝ) (pi := Real.pi) (h: v = (4 / 3) * pi * (sqrt 3 * a / 2) ^ 3) :
  a = (sqrt 3) / 2 :=
by
  sorry

end cube_edge_length_l51_51402


namespace solve_eq1_solve_eq2_l51_51154

variable (x : ℝ)

theorem solve_eq1 : (2 * x - 3 * (2 * x - 3) = x + 4) → (x = 1) :=
by
  intro h
  sorry

theorem solve_eq2 : ((3 / 4 * x - 1 / 4) - 1 = (5 / 6 * x - 7 / 6)) → (x = -1) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l51_51154


namespace spring_length_relation_l51_51745

open Real

theorem spring_length_relation (x : ℝ) : ∃ (k : ℝ), (k = 2) → (y : ℝ) = 10 + k * x :=
by
  use 2
  intro h
  rw h
  sorry

end spring_length_relation_l51_51745


namespace option_c_correct_l51_51255

theorem option_c_correct (α x1 x2 : ℝ) (hα1 : 0 < α) (hα2 : α < π) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  (x2 / x1) ^ Real.sin α > 1 :=
by
  sorry

end option_c_correct_l51_51255


namespace final_cost_in_gbp_l51_51078

/-- Original prices of trousers in USD -/
def price1 : ℝ := 100
def price2 : ℝ := 150
def price3 : ℝ := 200

/-- Discount rates -/
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.25
def global_discount : ℝ := 0.10

/-- Sales tax rate -/
def sales_tax : ℝ := 0.08

/-- Handling fee per trouser in USD -/
def handling_fee : ℝ := 5

/-- Conversion rate from USD to GBP -/
def usd_to_gbp : ℝ := 0.75

/-- Prove the final cost in GBP -/
theorem final_cost_in_gbp : 
  let discounted_price1 := price1 * (1 - discount1),
      discounted_price2 := price2 * (1 - discount2),
      discounted_price3 := price3 * (1 - discount3),
      total_discounted_price := discounted_price1 + discounted_price2 + discounted_price3,
      total_after_global_discount := total_discounted_price * (1 - global_discount),
      total_after_tax := total_after_global_discount * (1 + sales_tax),
      total_handling_fee := 3 * handling_fee,
      final_price_usd := total_after_tax + total_handling_fee,
      final_price_gbp := final_price_usd * usd_to_gbp
  in 
  final_price_gbp = 271.87 := by
    sorry

end final_cost_in_gbp_l51_51078


namespace product_ge_half_l51_51599

theorem product_ge_half (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3) (h_sum : x1 + x2 + x3 ≤ 1/2) :
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1/2 :=
by
  sorry

end product_ge_half_l51_51599


namespace range_of_x_in_obtuse_triangle_l51_51860

theorem range_of_x_in_obtuse_triangle (x : ℝ) (h1 : 1 < x) (h2 : x < 7)
  (h3 : x > 5 ∨ x < √7) :
  (3^2 + 4^2 < x^2 ∨ 3^2 + x^2 < 4^2) ↔ (1 < x ∧ x < √7 ∨ 5 < x ∧ x < 7) :=
sorry

end range_of_x_in_obtuse_triangle_l51_51860


namespace triangle_rel_length_l51_51932

variables {A B C K M : Type} [metric_space ℝ] [ordered_semiring ℝ]
noncomputable def triangle (A B C : Type) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 2 * (a*b + b*c + c*a)

-- Angle bisectors AK and CM
noncomputable def is_angle_bisector {A B C K : Type} : B ≠ K ∧ is_colinear A B K := sorry
noncomputable def is_angle_bisector {C M : Type} : C ≠ M ∧ is_colinear A C M := sorry

-- Given conditions
variable {ABC : Type} 
variable (AB BC : ℝ) (H : AB > BC)

-- Prove that AM > MK > KC
theorem triangle_rel_length (h : triangle ABC) (h_bisectAK : is_angle_bisector A B K) (h_bisectCM : is_angle_bisector C M) (H : AB > BC) : 
  AM > MK ∧ MK > KC :=
sorry

end triangle_rel_length_l51_51932


namespace _l51_51701

noncomputable def regular_tetrahedron_projection_theorem (a : ℝ) : Prop :=
  ∀ (v1 v2 v3 v4 : EuclideanSpace ℝ (Fin 3)),
  (is_regular_tetrahedron v1 v2 v3 v4 a) →
  (∃ center : EuclideanSpace ℝ (Fin 3),
  ∀ (plane : submodule ℝ (EuclideanSpace ℝ (Fin 3))),
  (is_center_of_tetrahedron center v1 v2 v3 v4) →
  (sum_of_projection_squares center [v1, v2, v3, v4] plane = a^2)) 

-- Helper definitions required for the theorem
def is_regular_tetrahedron (v1 v2 v3 v4 : EuclideanSpace ℝ (Fin 3)) (a : ℝ) : Prop := sorry
def is_center_of_tetrahedron (center : EuclideanSpace ℝ (Fin 3)) (v1 v2 v3 v4 : EuclideanSpace ℝ (Fin 3)) : Prop := sorry
def sum_of_projection_squares (center : EuclideanSpace ℝ (Fin 3)) (vertices : list (EuclideanSpace ℝ (Fin 3))) (plane : submodule ℝ (EuclideanSpace ℝ (Fin 3))) : ℝ := sorry

end _l51_51701


namespace problem1_part1_problem1_part2_problem2_problem3_l51_51564

noncomputable def floor (x : ℚ) : ℤ :=
  int.floor x

theorem problem1_part1 : floor 4.8 = 4 :=
by
  sorry

theorem problem1_part2 : floor (-6.5) = -7 :=
by
  sorry

theorem problem2 (x : ℚ) (h : floor x = 3) : 3 ≤ x ∧ x < 4 :=
by
  sorry

theorem problem3 (x : ℚ) (h : floor (5*x - 2) = 3*x + 1) : x = 5/3 :=
by
  sorry

end problem1_part1_problem1_part2_problem2_problem3_l51_51564


namespace min_triang_cover_l51_51238

theorem min_triang_cover (large_side : ℝ) (small_side : ℝ) 
  (h1 : large_side = 8) (h2 : small_side = 2) :
  let area_large := (sqrt 3 / 4) * large_side^2
  let area_small := (sqrt 3 / 4) * small_side^2
  (area_large / area_small) = 16 :=
by
  rw [h1, h2]
  let area_large := (sqrt 3 / 4) * (8^2)
  let area_small := (sqrt 3 / 4) * (2^2)
  have : (area_large / area_small) = (16 * sqrt 3) / sqrt 3 :=
    by rw [<-@mul_div_cancel _ _ (sqrt 3) (sqrt 3) (ne_of_gt (real.sqrt_pos.mpr zero_lt_three))]
  rw [<-mul_div_cancel' (sqrt 3 / 4) ((2^2) * (sqrt 3 / 4)) (sqrt 3)]
  simp [area_large, area_small, this]
  sorry

end min_triang_cover_l51_51238


namespace closest_perfect_square_to_350_l51_51671

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l51_51671


namespace find_multiply_l51_51955

def T := {x : ℝ // x ≠ 0}
def g (x : T) : T

axiom g_prop : ∀ (x y : T), x.val + y.val ≠ 0 →
  (g x).val + (g y).val = (g ⟨(x.val * y.val * (g ⟨x.val + y.val, sorry⟩).val) / 2, sorry⟩).val

theorem find_multiply : (1 * 1 = : ℝ) :=
by
  sorry

end find_multiply_l51_51955


namespace find_solution_l51_51808

theorem find_solution : ∃ z : ℝ, (sqrt (7 + 3 * z) = 13) ∧ (z = 54) :=
by
  use 54
  split
  · calc
      sqrt (7 + 3 * 54) = sqrt 169 := by sorry
      ... = 13 := by sorry
  · sorry

end find_solution_l51_51808


namespace analysis_hours_l51_51155

-- Define the conditions: number of bones and minutes per bone
def number_of_bones : Nat := 206
def minutes_per_bone : Nat := 45

-- Define the conversion factor: minutes per hour
def minutes_per_hour : Nat := 60

-- Define the total minutes spent analyzing all bones
def total_minutes (number_of_bones minutes_per_bone : Nat) : Nat :=
  number_of_bones * minutes_per_bone

-- Define the total hours required for analysis
def total_hours (total_minutes minutes_per_hour : Nat) : Float :=
  total_minutes.toFloat / minutes_per_hour.toFloat

-- Prove that total_hours equals 154.5 hours
theorem analysis_hours : total_hours (total_minutes number_of_bones minutes_per_bone) minutes_per_hour = 154.5 := by
  sorry

end analysis_hours_l51_51155


namespace neither_jia_nor_yi_has_winning_strategy_l51_51073

/-- 
  There are 99 points, each marked with a number from 1 to 99, placed 
  on 99 equally spaced points on a circle. Jia and Yi take turns 
  placing one piece at a time, with Jia going first. The player who 
  first makes the numbers on three consecutive points form an 
  arithmetic sequence wins. Prove that neither Jia nor Yi has a 
  guaranteed winning strategy, and both possess strategies to avoid 
  losing.
-/
theorem neither_jia_nor_yi_has_winning_strategy :
  ∀ (points : Fin 99 → ℕ), -- 99 points on the circle
  (∀ i, 1 ≤ points i ∧ points i ≤ 99) → -- Each point is numbered between 1 and 99
  ¬(∃ (player : Fin 99 → ℕ) (h : ∀ (i : Fin 99), player i ≠ 0 ∧ (player i = 1 ∨ player i = 2)),
    ∃ i : Fin 99, (points i + points (i + 1) + points (i + 2)) / 3 = points i)
:=
by
  sorry

end neither_jia_nor_yi_has_winning_strategy_l51_51073


namespace corey_candies_l51_51997

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end corey_candies_l51_51997


namespace problem_B_problem_C_l51_51377

variables (m n l : Type) [Line m] [Line n] [Line l]
variables (α β γ δ : Type) [Plane α] [Plane β] [Plane γ] [Plane δ]
variables (P : Type) [Point P]

-- Assumptions for problem B
axiom alpha_beta_intersection : α ∩ β = l
axiom beta_gamma_intersection : β ∩ γ = m
axiom gamma_alpha_intersection : γ ∩ α = n
axiom line_intersection P : l ∩ m = P

-- Assumptions for problem C
axiom perp_m_alpha : m ⊥ α
axiom perp_m_beta : m ⊥ β
axiom parallel_alpha_gamma : α ∥ γ

-- Firstly, B: Given the above conditions, we need to prove P ∈ n
theorem problem_B : P ∈ n := sorry

-- Second, C: Given the above conditions, we need to prove β ∥ γ
theorem problem_C : β ∥ γ := sorry

end problem_B_problem_C_l51_51377


namespace number_drawn_from_8th_group_l51_51737

-- Define the population, grouping, and conditions.
def population : List ℕ := List.range 100

def group_size : ℕ := 10

def groups (n : ℕ) : List ℕ :=
  (List.range (group_size * n)).filter (λ x, x / group_size = n - 1)

def systematic_sampling (m : ℕ) (k : ℕ) : ℕ :=
  let units_digit := (m + k) % 10
  group_size * (k - 1) + List.head! ((groups k).filter (λ x, x % 10 = units_digit))

-- Given conditions
def m := 8
def k := 8

-- Proof problem
theorem number_drawn_from_8th_group : systematic_sampling m k = 76 :=
by
  sorry

end number_drawn_from_8th_group_l51_51737


namespace final_net_earnings_l51_51569

-- Declare constants representing the problem conditions
def connor_hourly_rate : ℝ := 7.20
def connor_hours_worked : ℝ := 8.0
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate
def emily_hours_worked : ℝ := 10.0
def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

-- Combined final net earnings for the day
def combined_final_net_earnings (connor_hourly_rate emily_hourly_rate sarah_hourly_rate
                                  connor_hours_worked emily_hours_worked
                                  connor_deduction_rate emily_deduction_rate sarah_deduction_rate : ℝ) : ℝ :=
  let connor_gross := connor_hourly_rate * connor_hours_worked
  let emily_gross := emily_hourly_rate * emily_hours_worked
  let sarah_gross := sarah_hourly_rate * connor_hours_worked
  let connor_net := connor_gross * (1 - connor_deduction_rate)
  let emily_net := emily_gross * (1 - emily_deduction_rate)
  let sarah_net := sarah_gross * (1 - sarah_deduction_rate)
  connor_net + emily_net + sarah_net

-- The theorem statement proving their combined final net earnings
theorem final_net_earnings : 
  combined_final_net_earnings 7.20 14.40 36.00 8.0 10.0 0.05 0.08 0.10 = 498.24 :=
by sorry

end final_net_earnings_l51_51569


namespace flamingoes_needed_l51_51130

def feathers_per_flamingo : ℕ := 20
def safe_pluck_percentage : ℚ := 0.25
def boas_needed : ℕ := 12
def feathers_per_boa : ℕ := 200
def total_feathers_needed : ℕ := boas_needed * feathers_per_boa

theorem flamingoes_needed :
  480 = total_feathers_needed / (feathers_per_flamingo * safe_pluck_percentage).toNat :=
by sorry

end flamingoes_needed_l51_51130


namespace determine_length_BD_l51_51458

noncomputable def length_AC : ℝ := 10
noncomputable def length_BC : ℝ := 10
noncomputable def length_AB : ℝ := 3
noncomputable def length_CD : ℝ := 12

theorem determine_length_BD 
    (AC := length_AC) (BC := length_BC) (AB := length_AB) (CD := length_CD) : 
    BD (AC BC AB CD) = 5.3 :=
sorry

end determine_length_BD_l51_51458


namespace find_g_l51_51850

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2 * x else g x

axiom odd_f (x : ℝ) : f (-x) = -f x

theorem find_g (x : ℝ) (h : x < 0) : g x = -x^2 - 2*x :=
  sorry

end find_g_l51_51850


namespace cost_of_jam_l51_51343

theorem cost_of_jam (N B J : ℕ) (hN : N > 1) (h_total_cost : N * (5 * B + 6 * J) = 348) :
    6 * N * J = 348 := by
  sorry

end cost_of_jam_l51_51343


namespace projective_transform_exists_l51_51144

-- Define placeholder types for projective transformations and circles
axiom ProjectiveTransformation : Type
axiom Circle : Type
axiom Point : Type

-- Define projective transformation properties
axiom exists_transform_map_circle_to_circle : 
  ∀ (C : Circle), ∃ (T : ProjectiveTransformation), T.maps_circle_to_circle C

axiom map_point_to_center : 
  ∀ (T : ProjectiveTransformation) (C1 C2 : Circle) (M : Point), 
    T.maps_point_to_center C1 C2 M

axiom map_line_to_line : 
  ∀ (T : ProjectiveTransformation) (line : Set Point), T.maps_line_to_line line

noncomputable def find_transform 
  (C : Circle) (chord : Set Point) : ProjectiveTransformation :=
  sorry

theorem projective_transform_exists : 
  ∀ (C : Circle) (chord : Set Point), 
  ∃ (T : ProjectiveTransformation),
    (T.maps_circle_to_circle C) ∧ (T.maps_chord_to_diameter C chord) :=
begin
  intros C chord,
  unfold find_transform,
  use find_transform C chord,
  sorry
end

end projective_transform_exists_l51_51144


namespace flag_count_l51_51161

def colors := 3

def stripes := 3

noncomputable def number_of_flags (colors stripes : ℕ) : ℕ :=
  colors ^ stripes

theorem flag_count : number_of_flags colors stripes = 27 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end flag_count_l51_51161


namespace roots_of_polynomial_l51_51807

theorem roots_of_polynomial :
  roots (λ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6) = {1, 2, 3} := 
sorry

end roots_of_polynomial_l51_51807


namespace third_vertex_of_equilateral_triangle_l51_51761

/-- The third vertex of an equilateral triangle with vertices (0,3) and (6,3) located in the first quadrant.
Given that two vertices of the equilateral triangle are at (0,3) and (6,3), the third vertex must be at (6, 3 + 3 * Real.sqrt 3). -/
theorem third_vertex_of_equilateral_triangle :
  ∃ (x y : ℤ), 
    (0, 3) ≠ (x, y) ∧ 
    (6, 3) ≠ (x, y) ∧ 
    (0, 3) ≠ (6, 3) ∧ 
    ((x-0)^2 + (y-3)^2 = 36) ∧ 
    ((x-6)^2 + (y-3)^2 = 36) ∧ 
    (y > 3) ∧ 
    (x, y) = (6, 3 + 3 * Real.sqrt 3) :=
begin
  sorry
end

end third_vertex_of_equilateral_triangle_l51_51761


namespace minimum_value_of_z_minus_i_l51_51408

def complex (ℂ) (z : ℂ) (i : ℂ) : Prop :=
  let cond : Prop := abs (z + complex.I * sqrt 3) + abs (z - complex.I * sqrt 3) = 4
  in cond

theorem minimum_value_of_z_minus_i {z : ℂ} : 
  (complex ℂ z complex.I) → abs (z - complex.I) = (sqrt 6) / 3 :=
by
  sorry

end minimum_value_of_z_minus_i_l51_51408


namespace area_of_R_l51_51102

-- Definitions of S_- and S_+
def S_minus (x y : ℝ) : Prop := (x + 1)^2 + (y - 3/2)^2 = 1/4 ∧ x ≤ -1
def S_plus (x y : ℝ) : Prop := (x - 1)^2 + (y - 3/2)^2 = 1/4 ∧ x ≥ 1

-- Definition of R
def line (A B x y : ℝ) : Prop := A * x + B * y = 1
def R : set (ℝ × ℝ) := 
  {P | ∃ (A B C D : ℝ), S_minus A B ∧ S_plus C D ∧ 
         line A B (P.1) (P.2) ∧ 
         line C D (P.1) (P.2)}

-- Theorem to be proved
theorem area_of_R : 
  let area (R : set (ℝ × ℝ)) : ℝ := 1/6
  in area(R) = 1/6 := sorry

end area_of_R_l51_51102


namespace find_p_q_l51_51021

theorem find_p_q (p q : ℤ) (h : ∀ x : ℤ, (x - 5) * (x + 2) = x^2 + p * x + q) :
  p = -3 ∧ q = -10 :=
by {
  -- The proof would go here, but for now we'll use sorry to indicate it's incomplete.
  sorry
}

end find_p_q_l51_51021


namespace roots_of_polynomial_l51_51802

noncomputable def polynomial : Polynomial ℝ := Polynomial.mk [6, -11, 6, -1]

theorem roots_of_polynomial :
  (∃ r1 r2 r3 : ℝ, polynomial = (X - C r1) * (X - C r2) * (X - C r3) ∧ {r1, r2, r3} = {1, 2, 3}) :=
sorry

end roots_of_polynomial_l51_51802


namespace divisors_product_exceeds_16_digits_l51_51095

theorem divisors_product_exceeds_16_digits :
  let n := 1024
  let screen_digits := 16
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  (String.length (Natural.toString divisors_product) > screen_digits) := 
by {
  let n := 1024
  let screen_digits := 16
  let divisors := List.range (11)
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  show (String.length (Natural.toString divisors_product) > screen_digits),
  sorry
}

end divisors_product_exceeds_16_digits_l51_51095


namespace y_intercept_of_line_l51_51218

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end y_intercept_of_line_l51_51218


namespace no_real_roots_l51_51334

theorem no_real_roots 
    (h : ∀ x : ℝ, (3 * x^2 / (x - 2)) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) 
    : False := by
  sorry

end no_real_roots_l51_51334


namespace ponce_lighter_l51_51934

-- Define weights of Ishmael, Ponce, and Jalen as real numbers
variables (I P J : ℝ)

-- Given conditions
def ishmael_heavier (h : ℝ) : Prop := I = P + 20
def jalen_weight : Prop := J = 160
def average_weight : Prop := (I + P + J) / 3 = 160

-- Goal: Prove the difference in weight between Ponce and Jalen
theorem ponce_lighter (h : ishmael_heavier I P) (j : jalen_weight J) 
  (a : average_weight I P J) : P = 150 ∧ J - P = 10 :=
by {
  sorry
}

end ponce_lighter_l51_51934


namespace last_digit_fraction_inv_pow_3_15_l51_51632

theorem last_digit_fraction_inv_pow_3_15 : 
  (Nat.Mod (Nat.floor ((1 : ℚ) / (3^15))) 10) = 0 :=
by
  sorry

end last_digit_fraction_inv_pow_3_15_l51_51632


namespace share_of_a_l51_51705

variables (A B C : ℝ)

theorem share_of_a :
  A + B + C = 500 ∧
  A = (2/3) * (B + C) ∧
  B = (2/3) * (A + C)
  → A = 200 :=
by
  intro h
  cases h with h_sum h1
  cases h1 with h_a h_b
  sorry

end share_of_a_l51_51705


namespace last_digit_fraction_inv_pow_3_15_l51_51630

theorem last_digit_fraction_inv_pow_3_15 : 
  (Nat.Mod (Nat.floor ((1 : ℚ) / (3^15))) 10) = 0 :=
by
  sorry

end last_digit_fraction_inv_pow_3_15_l51_51630


namespace remainder_of_14_pow_53_mod_7_l51_51636

theorem remainder_of_14_pow_53_mod_7 : (14 ^ 53) % 7 = 0 := by
  sorry

end remainder_of_14_pow_53_mod_7_l51_51636


namespace ratio_of_width_to_length_is_correct_l51_51465

-- Define the given conditions
def length := 10
def perimeter := 36

-- Define the width and the expected ratio
def width (l P : Nat) : Nat := (P - 2 * l) / 2
def ratio (w l : Nat) := w / l

-- Statement to prove that given the conditions, the ratio of width to length is 4/5
theorem ratio_of_width_to_length_is_correct :
  ratio (width length perimeter) length = 4 / 5 :=
by
  sorry

end ratio_of_width_to_length_is_correct_l51_51465


namespace minimum_value_function_l51_51340

theorem minimum_value_function (x : ℝ) (h : x > 1) : 
  ∃ y, y = (16 - 2 * Real.sqrt 7) / 3 ∧ ∀ x > 1, (4*x^2 + 2*x + 5) / (x^2 + x + 1) ≥ y :=
sorry

end minimum_value_function_l51_51340


namespace coin_sum_varieties_l51_51293

theorem coin_sum_varieties : 
  let coins := {1, 1, 1, 5, 10, 25}
  let possible_sums := coins.to_finset.powerset.map (λ s => s.val.sum).erase 0 -- remove the empty subset sum
in possible_sums.card = 7 :=
by
  let coins := {1, 1, 1, 5, 10, 25}
  let pairs := coins.to_finset.powerset.filter (λ s => s.card = 2)  -- take only pairs
  let sums := pairs.map (λ s => s.sum)
  let distinct_sums := sums.erase_duplicates
  exact (distinct_sums.card = 7)

end coin_sum_varieties_l51_51293


namespace complex_purely_imaginary_l51_51406

theorem complex_purely_imaginary (a : ℝ) :
  (∀ z : ℂ, z = complex.mk (a * (a - 1)) (a) → z.im = z ∧ z.re = 0) → a = 1 :=
by
  intros h z hz
  have h1 : z.im = z := by
    exact h z hz
  have h2 : z.re = 0 := by
    exact h z hz
  sorry

end complex_purely_imaginary_l51_51406


namespace a4_eq_80_l51_51948

variable (a : ℕ → ℕ)

-- Condition: a₁ = 2
def a1 : Prop := a 1 = 2

-- Condition: The sequence {1 + a_n} is a geometric sequence with a common ratio of 3.
def geom_seq : Prop := ∀ n : ℕ, 1 + a (n + 1) = (1 + a 1) * 3^n

-- Question: Prove a₄ = 80.
theorem a4_eq_80 : a1 a ∧ geom_seq a → a 4 = 80 :=
by
  intro h,
  sorry

end a4_eq_80_l51_51948


namespace combination_square_octagon_tiles_l51_51684

-- Define the internal angles of the polygons
def internal_angle (shape : String) : Float :=
  match shape with
  | "Square"   => 90.0
  | "Pentagon" => 108.0
  | "Hexagon"  => 120.0
  | "Octagon"  => 135.0
  | _          => 0.0

-- Define the condition for the combination of two regular polygons to tile seamlessly
def can_tile (shape1 shape2 : String) : Bool :=
  let angle1 := internal_angle shape1
  let angle2 := internal_angle shape2
  angle1 + 2 * angle2 == 360.0

-- Define the tiling problem
theorem combination_square_octagon_tiles : can_tile "Square" "Octagon" = true :=
by {
  -- The proof of this theorem should show that Square and Octagon can indeed tile seamlessly
  sorry
}

end combination_square_octagon_tiles_l51_51684


namespace y_intercept_of_line_l51_51217

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end y_intercept_of_line_l51_51217


namespace find_q_l51_51799

theorem find_q (q : ℝ → ℝ) :
  (∀ x, q x = 0 → x = 2 ∨ x = -2) ∧
  (degree (q) < 3) ∧
  (q 3 = 15) →
  q = (λ x, 3 * x^2 - 12) :=
by
  sorry

end find_q_l51_51799


namespace distance_circle_line_l51_51061

noncomputable def rectangular_coordinates (ρ θ : ℝ) : (ℝ × ℝ) :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

def parametric_line (t : ℝ) : (ℝ × ℝ) :=
  (-1 + t, -1 - t)

def distance_point_line (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  let (x, y) := P
  (|A * x + B * y + C|) / (Real.sqrt (A^2 + B^2))

theorem distance_circle_line
  (ρ θ : ℝ)
  (hC1 : ρ = 2 * Real.cos θ + 2 * Real.sin θ)
  (hcenter : (1, 1) : ℝ × ℝ)
  (hindist : distance_point_line (1, 1) 1 1 2 = 2 * Real.sqrt 2)
  : (∃ r : ℝ, r = Real.sqrt 2 ∧ (distance_point_line (1, 1) 1 1 2 + r = 3 * Real.sqrt 2 ∧ distance_point_line (1, 1) 1 1 2 - r = Real.sqrt 2)) :=
by
  sorry

end distance_circle_line_l51_51061


namespace smallest_tree_height_l51_51192

theorem smallest_tree_height (tallest middle smallest : ℝ)
  (h1 : tallest = 108)
  (h2 : middle = (tallest / 2) - 6)
  (h3 : smallest = middle / 4) : smallest = 12 :=
by
  sorry

end smallest_tree_height_l51_51192


namespace number_of_animal_books_l51_51610

-- Step 1: Define the given conditions as constants
constant c : ℕ
constant t : ℕ
constant b_space : ℕ
constant b_trains : ℕ
constant b_animals : ℕ

-- Step 2: Specify the values for the given conditions
axiom cost_def : c = 16
axiom total_spent_def : t = 224
axiom books_space_def : b_space = 1
axiom books_trains_def : b_trains = 3

-- Step 3: Assert the mathematical statement to be proved
theorem number_of_animal_books : b_animals = 10 :=
by
  sorry

end number_of_animal_books_l51_51610


namespace smallest_rectangle_area_contains_L_shape_l51_51242

-- Condition: Side length of each square
def side_length : ℕ := 8

-- Condition: Number of squares
def num_squares : ℕ := 6

-- The correct answer (to be proven equivalent)
def expected_area : ℕ := 768

-- The main theorem stating the expected proof problem
theorem smallest_rectangle_area_contains_L_shape 
  (side_length : ℕ) (num_squares : ℕ) (h_shape : side_length = 8 ∧ num_squares = 6) : 
  ∃area, area = expected_area :=
by
  sorry

end smallest_rectangle_area_contains_L_shape_l51_51242


namespace exists_z0_abs_z0_le_1_and_abs_f_z0_ge_abs_c0_add_abs_cn_l51_51386

theorem exists_z0_abs_z0_le_1_and_abs_f_z0_ge_abs_c0_add_abs_cn
  (n : ℕ)
  (f : ℂ → ℂ)
  (c : Fin (n + 1) → ℂ)
  (h_f : ∀ z, f z = ∑ i in Finset.range (n + 1), c i * z ^ (n - i))
  : ∃ z0 : ℂ, |z0| ≤ 1 ∧ |f z0| ≥ |c 0| + |c n := 
sorry

end exists_z0_abs_z0_le_1_and_abs_f_z0_ge_abs_c0_add_abs_cn_l51_51386


namespace range_a_l51_51413

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then Real.log x / Real.log a else -2 * x + 8

theorem range_a (a : ℝ) (hf : ∀ x, f a x ≤ f a 2) :
  1 < a ∧ a ≤ Real.sqrt 3 := by
  sorry

end range_a_l51_51413


namespace closest_perfect_square_to_350_l51_51645

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l51_51645


namespace area_of_rectangle_l51_51303

variable (l w : ℕ)
variable [Nonempty ℕ]

theorem area_of_rectangle (h1 : l = 10) (h2 : w = 7) : l * w = 70 :=
by
  rw [h1, h2]
  exact Nat.mul_comm _ _
  -- sorry

end area_of_rectangle_l51_51303


namespace invertible_interval_l51_51332

noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem invertible_interval : 
  (∃ I : set ℝ, I ⊆ {-∞} ∪ {x : ℝ | x ≤ 3 / 2} ∧ 1 ∈ I ∧ ∀ x y ∈ I, g x = g y → x = y) → 
  I = set.Iic (3 / 2) :=
by
  sorry

end invertible_interval_l51_51332


namespace minimum_bailing_rate_l51_51260

/-
Problem statement:
While Amy and Ben are 1.5 miles away from the shore, their boat starts taking in water
at a rate of 12 gallons per minute due to a leak. The boat can hold a maximum of 40 gallons
of water before it sinks. Amy rows the boat towards the shore at a speed of 3 miles per hour
while Ben tries to bail out the water. Prove that the minimum rate at which Ben must bail water 
(in gallons per minute) for them to reach the shore without sinking is 11 gallons per minute.
-/

theorem minimum_bailing_rate :
  (let distance := 1.5 -- miles
       speed := 3 -- miles per hour
       leak_rate := 12 -- gallons per minute
       max_capacity := 40 -- gallons
       time_to_shore := (distance / speed) * 60 -- in minutes
       total_intake := leak_rate * time_to_shore -- total gallons taken in due to leak
       excess_water := total_intake - max_capacity -- gallons to be bailed out
       required_rate := excess_water / time_to_shore -- gallons per minute
  in required_rate.ceil = 11) := 
by
  sorry

end minimum_bailing_rate_l51_51260


namespace first_nonzero_digit_one_over_137_l51_51234

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end first_nonzero_digit_one_over_137_l51_51234


namespace sum_g_equals_1000_5_l51_51513

noncomputable def g (x : ℝ) : ℝ := 4 / (16^x + 4)

theorem sum_g_equals_1000_5 :
  (∑ i in finset.range 2001, g ((i + 1 : ℝ) / 2002)) = 1000.5 :=
sorry

end sum_g_equals_1000_5_l51_51513


namespace closest_perfect_square_to_350_l51_51652

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l51_51652


namespace highest_possible_characteristic_l51_51592

theorem highest_possible_characteristic (n : ℕ) (hn : n ≥ 2) (grid : Fin n → Fin n → ℕ) (hgrid : ∀ (i j : Fin n), 1 ≤ grid i j ∧ grid i j ≤ n^2 ∧ ∀ (i₁ i₂ j₁ j₂ : Fin n), (i₁ ≠ i₂ ∨ j₁ ≠ j₂) → grid i₁ j₁ ≠ grid i₂ j₂) :
  ∃ (char : ℚ), (∀ (i j : Fin n), ∀ (k₁ k₂ : Fin n), k₁ ≠ k₂ → (grid i k₁ / grid i k₂ < char) ∧ (grid k₁ j / grid k₂ j < char)) ∧ char = (n + 1) / n := 
sorry

end highest_possible_characteristic_l51_51592


namespace total_earnings_l51_51497

theorem total_earnings (initial_sign_up_bonus : ℕ) (referral_bonus : ℕ)
  (friends_day1 : ℕ) (friends_week : ℕ) : 
  initial_sign_up_bonus = 5 →
  referral_bonus = 5 →
  friends_day1 = 5 →
  friends_week = 7 →
  initial_sign_up_bonus + 
  (friends_day1 + friends_week) * referral_bonus + 
  (friends_day1 + friends_week) * referral_bonus = 125 :=
by
  intros h1 h2 h3 h4
  calc
    (5 : ℕ) + (12 : ℕ) * 5 + 12 * 5
        = 5 + 60 + 60   : by rw [h1, h2, h3, h4]
    ... = 125           : by norm_num

end total_earnings_l51_51497


namespace instantaneous_velocity_at_t3_l51_51186

open Real

noncomputable def displacement (t : ℝ) : ℝ := 4 - 2 * t + t ^ 2

theorem instantaneous_velocity_at_t3 : deriv displacement 3 = 4 := 
by
  sorry

end instantaneous_velocity_at_t3_l51_51186


namespace product_congruent_to_sign_l51_51103

open BigOperators

theorem product_congruent_to_sign (k : ℕ) (p : ℕ) (hp : p = 8 * k + 5) (prime_p : p.Prime)
  (r : Fin (2 * k + 2) → ℕ) (h_different_remainders : ∀ i j : Fin (2 * k + 2), i ≠ j → (r i ^ 4 % p) ≠ (r j ^ 4 % p)) : 
  (∏ i in Finset.range (2 * k + 1), ∏ j in Finset.range (2 * k + 1) \ {i}, (r i ^ 4 + r j ^ 4)) % p 
  = (if even k then 1 else p - 1) :=
sorry

end product_congruent_to_sign_l51_51103


namespace fraction_not_on_time_l51_51269

variable (n : ℕ)
variable (h40 : n = 40)
variable (males : ℕ)
variable (females : ℕ)
variable (males_on_time : ℕ)
variable (females_on_time : ℕ)

theorem fraction_not_on_time (h_males : males = (3 * n) / 5)
  (h_males_on_time : males_on_time = (7 * males) / 8)
  (h_females : females = n - males)
  (h_females_on_time : females_on_time = (9 * females) / 10)
  (h_males_off_time : n / 5) -- Assuming n / 5
  (h_females_off_time : n / 20) -- Assuming n / 20
  :
  (males - males_on_time + females - females_on_time) / n = 1 / 8 := by
  sorry

end fraction_not_on_time_l51_51269


namespace remaining_dresses_pockets_count_l51_51937

-- Definitions translating each condition in the problem.
def total_dresses : Nat := 24
def dresses_with_pockets : Nat := total_dresses / 2
def dresses_with_two_pockets : Nat := dresses_with_pockets / 3
def total_pockets : Nat := 32

-- Question translated into a proof problem using Lean's logic.
theorem remaining_dresses_pockets_count :
  (total_pockets - (dresses_with_two_pockets * 2)) / (dresses_with_pockets - dresses_with_two_pockets) = 3 := by
  sorry

end remaining_dresses_pockets_count_l51_51937


namespace sum_first_100_terms_eq_400_l51_51005

-- Define the sequence aₙ with the specified conditions
def seq : ℕ → ℚ
| 0       := 5
| (n + 1) := (2 * seq n - 1) / (seq n - 2)

-- Define the desired property to prove
theorem sum_first_100_terms_eq_400 : (Finset.range 100).sum (λ n, seq n) = 400 := 
by sorry

end sum_first_100_terms_eq_400_l51_51005


namespace find_a_value_l51_51394

open Set

theorem find_a_value (a : ℝ) (A B : Set ℝ)
  (hA : A = {3, 4, a^2 - 3 * a - 1})
  (hB : B = {2 * a, -3})
  (hInter : A ∩ B = {-3}) :
  a = 1 :=
begin
  sorry
end

end find_a_value_l51_51394


namespace baking_dish_to_recipe_book_ratio_is_2_l51_51126

-- Definitions of costs
def cost_recipe_book : ℕ := 6
def cost_ingredient : ℕ := 3
def num_ingredients : ℕ := 5
def cost_apron : ℕ := cost_recipe_book + 1
def total_spent : ℕ := 40

-- Definition to calculate the total cost excluding the baking dish
def cost_excluding_baking_dish : ℕ :=
  cost_recipe_book + cost_apron + cost_ingredient * num_ingredients

-- Definition of cost of baking dish
def cost_baking_dish : ℕ := total_spent - cost_excluding_baking_dish

-- Definition of the ratio
def ratio_baking_dish_to_recipe_book : ℕ := cost_baking_dish / cost_recipe_book

-- Theorem stating that the ratio is 2
theorem baking_dish_to_recipe_book_ratio_is_2 :
  ratio_baking_dish_to_recipe_book = 2 :=
sorry

end baking_dish_to_recipe_book_ratio_is_2_l51_51126


namespace solution_set_f_10x_gt_0_l51_51410

theorem solution_set_f_10x_gt_0 (f : ℝ → ℝ) :
  (∀ x : ℝ, f x < 0 ↔ x < -1 ∨ x > 1 / 2) →
  (∀ x : ℝ, f (10^x) > 0 ↔ x < -real.logb 10 2) :=
by {
  intro cond,
  sorry
}

end solution_set_f_10x_gt_0_l51_51410


namespace reflect_across_y_axis_l51_51580

theorem reflect_across_y_axis (x y : ℝ) :
  (x, y) = (1, 2) → (-x, y) = (-1, 2) :=
by
  intro h
  cases h
  sorry

end reflect_across_y_axis_l51_51580


namespace closest_square_to_350_l51_51662

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l51_51662


namespace ab_gt_bc_l51_51929

variable {A B C L K : Type}

-- Assume A, B, and C are points forming a triangle ABC
axiom Triangle (A B C : Type) : Prop

-- Assume there is a point L on the angle bisector BL
axiom AngleBisector (A B C : Type) (L : Type) : Prop

-- Assume there is a point K such that LK = AB on the extension of BL past L
axiom ExtensionPoint (A B : Type) (L : Type) (K : Type) : Prop

-- Assume AK is parallel to BC
axiom Parallel (A K B C : Type) : Prop

-- Theorem we need to prove: AB > BC
theorem ab_gt_bc
  (hTriangle : Triangle A B C)
  (hAngleBisector : AngleBisector A B C L)
  (hExtensionPoint : ExtensionPoint A B L K)
  (hParallel : Parallel A K B C)
  : A > B := 
sorry

end ab_gt_bc_l51_51929


namespace monotonicity_of_f_on_interval_range_of_k_l51_51881

section
variable (f g : ℝ → ℝ)

noncomputable def f : ℝ → ℝ := λ x, x / (x^2 + 4)

def g (k : ℝ) : ℝ → ℝ := λ x, k * x^2 + 2 * k * x + 1

theorem monotonicity_of_f_on_interval :
  ∀ x1 x2 : ℝ, -2 ≤ x1 → x1 < x2 → x2 ≤ 2 → f x1 ≤ f x2 := sorry

theorem range_of_k :
  ∀ k : ℝ, (∀ x1 : ℝ, -2 ≤ x1 → x1 ≤ 2 → ∃ x2 : ℝ, -1 ≤ x2 → x2 ≤ 2 → f x1 = g k x2) →
  k ∈ Set.Ioo (-∞) (-5/32) ∪ Set.Ioo (5/4) (∞) := sorry

end

end monotonicity_of_f_on_interval_range_of_k_l51_51881


namespace function_range_l51_51863

theorem function_range (x : ℝ) (h : x > 1) : 
  ∃ y, y = x + 1 / (x - 1) ∧ y ∈ set.Ici (3 : ℝ) :=
by
  sorry

end function_range_l51_51863


namespace tricycles_in_garage_l51_51602

theorem tricycles_in_garage 
    (T : ℕ) 
    (total_bicycles : ℕ := 3) 
    (total_unicycles : ℕ := 7) 
    (bicycle_wheels : ℕ := 2) 
    (tricycle_wheels : ℕ := 3) 
    (unicycle_wheels : ℕ := 1) 
    (total_wheels : ℕ := 25) 
    (eq_wheels : total_bicycles * bicycle_wheels + total_unicycles * unicycle_wheels + T * tricycle_wheels = total_wheels) :
    T = 4 :=
by {
  sorry
}

end tricycles_in_garage_l51_51602


namespace jelly_bean_matching_probability_l51_51311

theorem jelly_bean_matching_probability :
  let abe_jelly_beans := ["green", "red", "blue"]
  let bob_jelly_beans := ["green", "yellow", "red", "red", "blue"]
  let abe_prob (c : String) := (abe_jelly_beans.count c).toRat / abe_jelly_beans.length.toRat
  let bob_prob (c : String) := (bob_jelly_beans.count c).toRat / bob_jelly_beans.length.toRat
  let match_prob := ["green", "red", "blue"].sum (fun c => abe_prob c * bob_prob c)
  match_prob = 4 / 15 := 
sorry

end jelly_bean_matching_probability_l51_51311


namespace max_ab_at_extremum_l51_51444

noncomputable def f (a b x : ℝ) : ℝ := 4*x^3 - a*x^2 - 2*b*x + 2

theorem max_ab_at_extremum (a b : ℝ) (h0: a > 0) (h1 : b > 0) (h2 : ∃ x, f a b x = 4*x^3 - a*x^2 - 2*b*x + 2 ∧ x = 1 ∧ 12*x^2 - 2*a*x - 2*b = 0) :
  ab ≤ 9 := 
sorry  -- proof not required

end max_ab_at_extremum_l51_51444


namespace closest_perfect_square_to_350_l51_51677

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l51_51677


namespace rectangle_ratio_l51_51842

theorem rectangle_ratio 
  (s : ℝ) -- side length of the inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_area : s^2 = (inner_square_area : ℝ))
  (h_outer_area : 9 * inner_square_area = outer_square_area)
  (h_outer_side_eq : (s + 2 * y)^2 = outer_square_area)
  (h_longer_side_eq : x + y = 3 * s) :
  x / y = 2 :=
by sorry

end rectangle_ratio_l51_51842


namespace sum_of_numbers_in_table_l51_51040

noncomputable def sum_of_neighbors_is_one {α : Type*} [Add α] [One α] (n : α) :=
∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → 
  ((i > 1 → (n (i-1) j)) + (i < 4 → (n (i+1) j)) + (j > 1 → (n i (j-1))) + (j < 4 → (n i (j+1)))) = 1

noncomputable def table_sum {α : Type*} [Add α] (n : α) := 
  ∑ i in finset.range 4, ∑ j in finset.range 4, n i.succ j.succ

theorem sum_of_numbers_in_table : 
  ∀ (n : ℕ → ℕ → ℕ), sum_of_neighbors_is_one n → table_sum n = 6 := 
sorry

end sum_of_numbers_in_table_l51_51040


namespace min_value_l51_51522

open Complex

-- Define the conditions of the problem
def condition_one (z : ℂ) : Prop := abs (z - (3 - 2 * I)) = 3

-- Define the expression whose minimum value we want to find
def expression (z : ℂ) : ℝ :=
  abs (z + (1 - I))^2 + abs (z - (7 - 3 * I))^2

-- State the theorem
theorem min_value (z : ℂ) (h : condition_one z) : expression z = 94 :=
by sorry

end min_value_l51_51522


namespace roots_of_polynomial_l51_51804

noncomputable def polynomial : Polynomial ℝ := Polynomial.mk [6, -11, 6, -1]

theorem roots_of_polynomial :
  (∃ r1 r2 r3 : ℝ, polynomial = (X - C r1) * (X - C r2) * (X - C r3) ∧ {r1, r2, r3} = {1, 2, 3}) :=
sorry

end roots_of_polynomial_l51_51804


namespace count_odd_integers_divisible_by_3_l51_51894

theorem count_odd_integers_divisible_by_3 : 
  let count := (finset.filter (λ x => odd x ∧ x % 3 = 0) (finset.Ico 5 16)).card in
  count = 2 :=
by 
  sorry

end count_odd_integers_divisible_by_3_l51_51894


namespace simplest_quadratic_radical_l51_51256

theorem simplest_quadratic_radical (x y : ℝ) : 
  let A := √(1 / 25)
  let B := √16
  let C := √(x^2 * y^3)
  let D := √(2 * x + 1)
  (∃ expr, expr = D ∧ (
    (A = 1 / 5 ∧ ∀ sqrt_val, sqrt_val ≠ 1 / 5) ∧
    (B = 4 ∧ ∀ sqrt_val, sqrt_val ≠ 4) ∧
    (C = |x| * y * √y ∧ ∀ sqrt_val, sqrt_val ≠ |x| * y * √y) ∧ 
    (D = √(2 * x + 1) ∧ ∀ sqrt_val, sqrt_val = √(2 * x + 1))
  )) := sorry

end simplest_quadratic_radical_l51_51256


namespace other_candidate_votes_l51_51697

theorem other_candidate_votes (h1 : one_candidate_votes / valid_votes = 0.6)
    (h2 : 0.3 * total_votes = invalid_votes)
    (h3 : total_votes = 9000)
    (h4 : valid_votes + invalid_votes = total_votes) :
    valid_votes - one_candidate_votes = 2520 :=
by
  sorry

end other_candidate_votes_l51_51697


namespace first_nonzero_digit_right_decimal_l51_51232

/--
  To prove that the first nonzero digit to the right of the decimal point of the fraction 1/137 is 9
-/
theorem first_nonzero_digit_right_decimal (n : ℕ) (h1 : n = 137) :
  ∃ d, d = 9 ∧ (∀ k, 10 ^ k * 1 / 137 < 10^(k+1)) → the_first_nonzero_digit_right_of_decimal_is 9 := 
sorry

end first_nonzero_digit_right_decimal_l51_51232


namespace tetrahedron_least_value_g_final_sum_l51_51785

noncomputable def least_possible_value_of_g (EF GH EG EH FH FG : ℕ) : ℕ :=
  let median_PE := sqrt (2 * (EG ^ 2 + FG ^ 2) - GH ^ 2) / 2 in
  let EH' := 2 * sqrt (median_PE ^ 2 + (48 ^ 2 / 4)) in
  let g_R := 2 * EH' in
  g_R

theorem tetrahedron_least_value_g (EF GH EG EH FH FG : ℕ)
  (h1 : EG = 26) (h2 : FH = 26) (h3 : EH = 40) (h4 : FG = 40) (h5 : EF = 48) (h6 : GH = 48) :
  least_possible_value_of_g EF GH EG EH FH FG = 4 * sqrt 579 :=
by
  sorry  -- This is where the proof would go, no need to complete

theorem final_sum (EF GH EG EH FH FG : ℕ)
  (h1 : EG = 26) (h2 : FH = 26) (h3 : EH = 40) (h4 : FG = 40) (h5 : EF = 48) (h6 : GH = 48) :
  let least_value := 4 * sqrt 579 in
  let p := 4 in
  let q := 579 in
  p + q = 583 :=
by
  sorry  -- This is where the proof would go, no need to complete

end tetrahedron_least_value_g_final_sum_l51_51785


namespace minimum_of_f_l51_51681

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 / (2 * (Real.cos x) * (Real.sin x) - (Real.sin x)^2)

theorem minimum_of_f : ∀ x : ℝ, 0 < x ∧ x < Real.pi / 3 → f(x) ≥ 1 :=
by
  intro x
  intros h
  sorry

end minimum_of_f_l51_51681


namespace farmer_spent_on_corn_seeds_l51_51720

theorem farmer_spent_on_corn_seeds 
  (total_bags : ℕ) (price_per_bag : ℕ) (fertilizer_pesticide_cost : ℕ) (labor_cost : ℕ) (profit_percentage : ℕ) :
  total_bags = 10 ->
  price_per_bag = 11 ->
  fertilizer_pesticide_cost = 35 ->
  labor_cost = 15 ->
  profit_percentage = 10 ->
  let total_revenue := total_bags * price_per_bag in
  let profit := profit_percentage * total_revenue / 100 in
  let total_cost := total_revenue - profit in
  let corn_seed_cost := total_cost - (fertilizer_pesticide_cost + labor_cost) in
  corn_seed_cost = 49 :=
begin
  intros h_bags h_price h_fert h_labor h_profit,
  simp [h_bags, h_price, h_fert, h_labor, h_profit],
  dsimp only,
  calc
    (total_bags * price_per_bag) = 10 * 11 : by rw h_bags; rw h_price
    ... = 110 : by norm_num
    ... - (profit_percentage * 110 / 100) = 110 - 11 : by rw h_profit; norm_num
    ... = 99,
  let corn_seed_cost := 99 - (fertilizer_pesticide_cost + labor_cost),
  calc
    (fertilizer_pesticide_cost + labor_cost) = 35 + 15 : by rw h_fert; rw h_labor
    ... = 50 : by norm_num,
  calc
    99 - 50 = 49 : by norm_num,
  exact rfl,
end

end farmer_spent_on_corn_seeds_l51_51720


namespace closest_square_to_350_l51_51658

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l51_51658


namespace intersect_lines_at_single_point_l51_51854

open Triangle

theorem intersect_lines_at_single_point
  {A B C A1 B1 C1 K L M N P : Point}
  (h_reg_pentagon_KLNP : regular_pentagon MKLNP)
  (h_KL_on_BC : K ∈ line.segment B C)
  (hKL_mid_A1 : midpoint K L A1)
  (h_points_on_sides : M ∈ line.segment A B ∧ N ∈ line.segment A C)
  (h_C1_def : midpoint C1 P ∧ C1 ∈ line.segment A B)
  (h_B1_def : midpoint B1 P ∧ B1 ∈ line.segment A C)
  (h_ratios : (|B - A1| / |A1 - C|) * (|C - B1| / |B1 - A|) * (|A - C1| / |C1 - B|) = 1) :
  concurrent (line.mk A A1) (line.mk B B1) (line.mk C C1) :=
by
  sorry

end intersect_lines_at_single_point_l51_51854


namespace closest_perfect_square_to_350_l51_51669

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l51_51669


namespace total_children_on_playground_l51_51703

theorem total_children_on_playground
  (boys : ℕ) (girls : ℕ)
  (h_boys : boys = 44) (h_girls : girls = 53) :
  boys + girls = 97 :=
by 
  -- Proof omitted
  sorry

end total_children_on_playground_l51_51703


namespace soccer_team_starters_l51_51975

open Nat

-- Definitions representing the conditions
def total_players : ℕ := 18
def twins_included : ℕ := 2
def remaining_players : ℕ := total_players - twins_included
def starters_to_choose : ℕ := 7 - twins_included

-- Theorem statement to assert the solution
theorem soccer_team_starters :
  Nat.choose remaining_players starters_to_choose = 4368 :=
by
  -- Placeholder for proof
  sorry

end soccer_team_starters_l51_51975


namespace composite_sum_l51_51134

open Nat

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) : ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + c = x * y :=
by
  sorry

end composite_sum_l51_51134


namespace tan_of_acute_angle_l51_51865

theorem tan_of_acute_angle (α : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : Real.cos (π / 2 + α) = -3/5) : Real.tan α = 3 / 4 :=
by
  sorry

end tan_of_acute_angle_l51_51865


namespace problem_statement_l51_51558

-- Definition of proposition p and its condition
def proposition_p (x : ℝ) : Prop :=
  (∀x, cos(x) * sin(x) = (1/2) * sin(2 * x)) ∧
  (∀x, (1/2) * sin(2 * (x - (3 * π / 4))) = (1/2) * cos(2 * x))

-- Definition of proposition q and its condition
def proposition_q (m : ℝ) (hm : m > 0) : Prop :=
  (2 * (m^2) - m^2 = m^2) ∧
  (∀m, m > 0 → sqrt(1 + m^2 / (m^2 / 2)) = sqrt(3))

-- Conjunction of p and q
def conjunction (x : ℝ) (m : ℝ) (hm : m > 0) : Prop :=
  proposition_p x ∨ proposition_q m hm

-- Theorem to be proven
theorem problem_statement (x : ℝ) (m : ℝ) (hm : m > 0) : conjunction x m hm :=
by
  sorry

end problem_statement_l51_51558


namespace angle_DAE_right_l51_51608

/-- Two circles of equal radii intersect at points B and C. 
A point A is chosen on the first circle. 
The ray AB intersects the second circle at point D (D ≠ B). 
A point E is chosen on the ray DC such that DC = CE. 
Prove that the angle DAE is a right angle. -/
theorem angle_DAE_right (O₁ O₂ A B C D E : ℝ) (r : ℝ) (hO₁ : circle O₁ r)
(hO₂ : circle O₂ r) (hA : A ∈ circle_points O₁ r) (hB : B ∈ circle_points O₁ r ∧ B ∈ circle_points O₂ r) 
(hC : C ∈ circle_points O₁ r ∧ C ∈ circle_points O₂ r)
(hD : D ∈ circle_points O₂ r ∧ D ≠ B ∧ on_ray A B D) 
(hE : on_ray D C E ∧ dist D C = dist C E) : 
angle D A E = 90 := by
  sorry

end angle_DAE_right_l51_51608


namespace expected_value_is_one_dollar_l51_51735

def star_prob := 1 / 4
def moon_prob := 1 / 2
def sun_prob := 1 / 4

def star_prize := 2
def moon_prize := 4
def sun_penalty := -6

def expected_winnings := star_prob * star_prize + moon_prob * moon_prize + sun_prob * sun_penalty

theorem expected_value_is_one_dollar : expected_winnings = 1 := by
  sorry

end expected_value_is_one_dollar_l51_51735


namespace sugar_left_correct_l51_51137

-- Define the total amount of sugar bought by Pamela
def total_sugar : ℝ := 9.8

-- Define the amount of sugar spilled by Pamela
def spilled_sugar : ℝ := 5.2

-- Define the amount of sugar left after spilling
def sugar_left : ℝ := total_sugar - spilled_sugar

-- State that the amount of sugar left should be equivalent to the correct answer
theorem sugar_left_correct : sugar_left = 4.6 :=
by
  sorry

end sugar_left_correct_l51_51137


namespace circle_intersection_points_l51_51378

theorem circle_intersection_points (n : ℕ) (O : Point) (C : Fin n → Circle) 
  (H : ∀ i, (i : ℕ) < n → passes_through O (C i)) 
  (A : Fin n → Point) (H1 : ∀ i, (i : ℕ) < n → second_intersection (C i) (C ((i + 1) % n)) A[i])
  (B1 : Point) (H2 : on_circle B1 (C 0)) (H3 : B1 ≠ A 0)
  (B : Fin (n + 1) → Point)
  (H4 : ∀ i, (i : ℕ) < n → line_through (B i) (A i) (C ((i + 1) % n)) B[i + 1]) :
  B[n+1] = B1 :=
begin
  sorry
end

end circle_intersection_points_l51_51378


namespace tennis_balls_ordered_originally_l51_51306

-- Definitions according to the conditions in a)
def retailer_ordered_equal_white_yellow_balls (W Y : ℕ) : Prop :=
  W = Y

def dispatch_error (Y : ℕ) : ℕ :=
  Y + 90

def ratio_white_to_yellow (W Y : ℕ) : Prop :=
  W / dispatch_error Y = 8 / 13

-- Main statement
theorem tennis_balls_ordered_originally (W Y : ℕ) (h1 : retailer_ordered_equal_white_yellow_balls W Y)
  (h2 : ratio_white_to_yellow W Y) : W + Y = 288 :=
by
  sorry    -- Placeholder for the actual proof

end tennis_balls_ordered_originally_l51_51306


namespace joe_speed_l51_51938

theorem joe_speed (v : ℝ) (h1 : ∀ (v : ℝ), joe_speed = 2 * v)
    (h2 : (40 : ℝ) / 60) (h3 : 16 = 3 * v * (40 / 60)) : 
    joe_speed = 16 :=
sorry

end joe_speed_l51_51938


namespace rate_of_additional_investment_l51_51731

-- Defining the problem statement and conditions
def Principal1 : ℕ := 8000
def Rate1 : ℝ := 0.05
def Principal2 : ℕ := 4000
def DesiredRate : ℝ := 0.06

-- The statement to be proved
theorem rate_of_additional_investment 
  (Principal1 : ℕ) 
  (Principal2 : ℕ)
  (Rate1 : ℝ)
  (DesiredRate : ℝ) : 
  let Interest1 := Principal1 * Rate1
  let TotalPrincipal := Principal1 + Principal2
  let TotalDesiredInterest := TotalPrincipal * DesiredRate
  let AdditionalInterest := TotalDesiredInterest - Interest1
  let Rate2 := AdditionalInterest / Principal2 in
  Rate2 * 100 = 8 :=
sorry

end rate_of_additional_investment_l51_51731


namespace probability_of_correct_guess_l51_51146

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def valid_number (n : ℕ) : Prop :=
  55 < n ∧ n < 100 ∧ is_prime (n / 10) ∧ is_even (n % 10)

theorem probability_of_correct_guess : 
  let valid_numbers := Finset.filter valid_number (Finset.range 100) in
  1 / valid_numbers.card = (1 : ℚ) / 7 := 
by
  let valid_numbers := Finset.filter valid_number (Finset.range 100)
  sorry

end probability_of_correct_guess_l51_51146


namespace time_taken_to_pass_l51_51206

def speed_km_hr_to_m_s (speed_km_hr : ℝ) : ℝ := speed_km_hr * (5 / 18)

def length_train_a : ℝ := 1500
def speed_train_a_kmh : ℝ := 60
def speed_train_a_ms : ℝ := speed_km_hr_to_m_s speed_train_a_kmh

def length_train_b : ℝ := 1200
def speed_train_b_kmh : ℝ := 45
def speed_train_b_ms : ℝ := speed_km_hr_to_m_s speed_train_b_kmh

def length_train_c : ℝ := 900
def speed_train_c_kmh : ℝ := 30
def speed_train_c_ms : ℝ := speed_km_hr_to_m_s speed_train_c_kmh

def combined_length_train_a_b : ℝ := length_train_a + length_train_b

theorem time_taken_to_pass :
  let relative_speed := speed_train_c_ms in
  let time := combined_length_train_a_b / relative_speed in
  time = 324.07 :=
by
  sorry

end time_taken_to_pass_l51_51206


namespace last_digit_decimal_expansion_l51_51628

theorem last_digit_decimal_expansion :
  let x : ℚ := 1 / 3^15
  in (x.to_decimal.last_digit_unit = 5) :=
by
  sorry

end last_digit_decimal_expansion_l51_51628


namespace solve_abs_inequality_l51_51791

theorem solve_abs_inequality (x : ℝ) :
  3 ≤ abs ((x - 3)^2 - 4) ∧ abs ((x - 3)^2 - 4) ≤ 7 ↔ 3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11 :=
sorry

end solve_abs_inequality_l51_51791


namespace pyramid_volume_correct_l51_51780

noncomputable def pyramid_volume (AB BC PO : ℝ) (angle_APB : ℝ) : ℝ :=
  let base_area := AB * BC in
  let height := PO in
  (1 / 3) * base_area * height

theorem pyramid_volume_correct :
  let A := (2 : ℝ)
  let B := (1 : ℝ)
  let angle_APB := (real.pi / 2)  -- 90 degrees in radians
  let height_PO := (real.sqrt 5 / 2)
  pyramid_volume A B height_PO angle_APB = real.sqrt 5 / 3 :=
by
  sorry

end pyramid_volume_correct_l51_51780


namespace bill_bought_60_rats_l51_51326

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end bill_bought_60_rats_l51_51326


namespace Eva_count_by_first_completion_l51_51773

theorem Eva_count_by_first_completion :
  let radius_eva := 5
  let radius_liam := 7
  let radius_jay := 9
  let surface_area (r : ℕ) := 4 * real.pi * r^2
  let surface_eva := surface_area radius_eva
  let surface_liam := surface_area radius_liam
  let surface_jay := surface_area radius_jay
  let LCM := nat.lcm surface_eva.nat_abs (nat.lcm surface_liam.nat_abs surface_jay.nat_abs)
  let eva_covered := LCM / surface_eva.nat_abs
  eva_covered = 1029 :=
begin
  sorry
end

end Eva_count_by_first_completion_l51_51773


namespace roots_inequality_l51_51515

noncomputable def a : ℝ := Real.sqrt 2020

theorem roots_inequality (x1 x2 x3 : ℝ) (h_roots : ∀ x, (a * x^3 - 4040 * x^2 + 4 = 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3))
  (h_inequality: x1 < x2 ∧ x2 < x3) : x2 * (x1 + x3) = 2 :=
sorry

end roots_inequality_l51_51515


namespace origin_in_circle_m_gt_5_l51_51452

theorem origin_in_circle_m_gt_5 (m : ℝ) : ((0 - 1)^2 + (0 + 2)^2 < m) → (m > 5) :=
by
  intros h
  sorry

end origin_in_circle_m_gt_5_l51_51452


namespace area_enclosed_by_parabola_and_line_l51_51351

def parabola (x : ℝ) : ℝ := x^2

def line (x : ℝ) : ℝ := -2*x + 3

def intersection_area : ℝ := 
  ∫ x in -3..1, (line x - parabola x)

theorem area_enclosed_by_parabola_and_line :
  intersection_area = 32 / 3 :=
by
  -- Verification of the enclosed area integral calculation
  sorry

end area_enclosed_by_parabola_and_line_l51_51351


namespace circle_equation_slope_value_l51_51405

theorem circle_equation (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) :
  (x - 2)^2 + (y - 3)^2 = 1 :=
begin
  exact h,
end

theorem slope_value (k : ℝ) (h1 : ∃ x y : ℝ, y = k * x + 1 ∧ (x - 2)^2 + (y - 3)^2 = 1)
  (h2 : |dist (0, 1) (x1, y1) - dist (0, 1) (x2, y2)| = 2) :
  k = 1 :=
begin
  sorry
end

end circle_equation_slope_value_l51_51405


namespace find_explicit_formula_range_of_k_l51_51124

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 - b * x + 4

theorem find_explicit_formula (h_extremum_at_2 : f a b 2 = -4 / 3 ∧ (3 * a * 4 - b = 0)) :
  ∃ a b, f a b x = (1 / 3) * x ^ 3 - 4 * x + 4 :=
sorry

theorem range_of_k (h_extremum_at_2 : f (1 / 3) 4 2 = -4 / 3) :
  ∃ k, -4 / 3 < k ∧ k < 8 / 3 :=
sorry

end find_explicit_formula_range_of_k_l51_51124


namespace newton_birth_day_l51_51574

-- Definition of the conditions:
def anniversary_date := 1943 - 1643 = 300
def anniversary_day := "Monday"
def non_leap_year_condition (y : ℕ) : Prop := ¬(y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))

-- Main theorem statement: Newton was born on a Sunday.
theorem newton_birth_day
  (H1 : anniversary_date = 300)
  (H2 : anniversary_day = "Monday")
  (H3 : non_leap_year_condition 1643) :
  (birth_day : String) :=
  birth_day = "Sunday" := sorry

end newton_birth_day_l51_51574


namespace marble_problem_l51_51532

theorem marble_problem :
  let green_marbles := 8
  let purple_marbles := 7
  let total_marbles := green_marbles + purple_marbles
  let trials := 6
  let prob_green := (green_marbles : ℚ) / total_marbles
  let prob_purple := (purple_marbles : ℚ) / total_marbles
  let binom := Nat.choose trials 3
  let single_trial_prob := binom * (prob_green^3) * (prob_purple^3)
  let combined_prob := single_trial_prob * prob_purple in
  combined_prob = 4913248 / 34171875 := 
  sorry

end marble_problem_l51_51532


namespace egg_hunt_ratio_l51_51918

theorem egg_hunt_ratio :
  ∃ T : ℕ, (3 * T + 30 = 400 ∧ T = 123) ∧ (60 : ℚ) / (T - 20 : ℚ) = 60 / 103 :=
by
  sorry

end egg_hunt_ratio_l51_51918


namespace min_absolute_sum_value_l51_51244

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end min_absolute_sum_value_l51_51244


namespace Olivia_pays_4_dollars_l51_51548

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l51_51548


namespace exists_pairwise_coprime_l51_51707

theorem exists_pairwise_coprime (n : ℕ) (hn : n > 0) :
  ∃ (k : Fin (n + 1) → ℕ), (∀ i, 1 < k i) ∧ (Pairwise (λ i j, coprime (k i) (k j))) ∧ 
                           ∃ a, k 0 * k 1 * ⋯ * k n - 1 = a * (a - 1) :=
by sorry

end exists_pairwise_coprime_l51_51707


namespace simplify_expression_l51_51620

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l51_51620


namespace minimum_spend_on_boxes_l51_51682

def box_dimensions : ℕ × ℕ × ℕ := (20, 20, 12)
def cost_per_box : ℝ := 0.40
def total_volume : ℕ := 2400000
def volume_of_box (l w h : ℕ) : ℕ := l * w * h
def number_of_boxes (total_vol vol_per_box : ℕ) : ℕ := total_vol / vol_per_box
def total_cost (num_boxes : ℕ) (cost_box : ℝ) : ℝ := num_boxes * cost_box

theorem minimum_spend_on_boxes : total_cost (number_of_boxes total_volume (volume_of_box 20 20 12)) cost_per_box = 200 := by
  sorry

end minimum_spend_on_boxes_l51_51682


namespace length_of_train_l51_51750

-- Conditions
def speed_km_hr := 54 -- km/hr
def time_seconds := 7 -- seconds

-- Conversion functions
def km_per_hr_to_m_per_s (speed_km_hr : ℕ) : ℕ :=
  (speed_km_hr * 1000) / 3600

-- Main statement
theorem length_of_train :
  let speed_m_per_s := km_per_hr_to_m_per_s speed_km_hr in
  speed_m_per_s * time_seconds = 105 :=
by
  let speed_m_per_s := km_per_hr_to_m_per_s speed_km_hr
  show speed_m_per_s * time_seconds = 105
  sorry

end length_of_train_l51_51750


namespace min_value_fraction_l51_51176

theorem min_value_fraction (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∃ x₀, (2 * x₀ - 2) * (-2 * x₀ + a) = -1) : 
  ∃ a b, a + b = 5 / 2 → a > 0 → b > 0 → 
  (∀ a b, (1 / a + 4 / b) ≥ 18 / 5) :=
by
  sorry

end min_value_fraction_l51_51176


namespace math_club_total_members_l51_51055

theorem math_club_total_members (female_count : ℕ) (h_female : female_count = 6) (h_male_ratio : ∃ male_count : ℕ, male_count = 2 * female_count) :
  ∃ total_members : ℕ, total_members = female_count + classical.some h_male_ratio :=
by
  let male_count := classical.some h_male_ratio
  have h_male_count : male_count = 12 := by sorry
  existsi (female_count + male_count)
  rw [h_female, h_male_count]
  exact rfl

end math_club_total_members_l51_51055


namespace div_inside_parentheses_l51_51614

theorem div_inside_parentheses :
  100 / (6 / 2) = 100 / 3 :=
by
  sorry

end div_inside_parentheses_l51_51614


namespace trigonometric_identities_l51_51845

theorem trigonometric_identities 
  (α β : ℝ) 
  (h1 : α ∈ Ioo (π / 2) π) 
  (h2 : β ∈ Ioo (π / 2) π)
  (h3 : cos α = -4 / 5) 
  (h4 : sin β = 5 / 13) :
  sin (α + β) = -56 / 65 ∧ cos (α - β) = 63 / 65 ∧ tan (2 * α - β) = -253 / 204 := by
sorry

end trigonometric_identities_l51_51845


namespace count_males_not_in_orchestra_l51_51160

variable (females_band females_orchestra females_choir females_all
          males_band males_orchestra males_choir males_all total_students : ℕ)
variable (males_band_not_in_orchestra : ℕ)

theorem count_males_not_in_orchestra :
  females_band = 120 ∧ females_orchestra = 90 ∧ females_choir = 50 ∧ females_all = 30 ∧
  males_band = 90 ∧ males_orchestra = 120 ∧ males_choir = 40 ∧ males_all = 20 ∧
  total_students = 250 ∧ males_band_not_in_orchestra = (males_band - (males_band + males_orchestra + males_choir - males_all - total_students)) 
  → males_band_not_in_orchestra = 20 :=
by
  intros
  sorry

end count_males_not_in_orchestra_l51_51160


namespace roger_donated_coins_l51_51568

theorem roger_donated_coins : 
  ∀ (pennies nickels dimes remaining_coins : ℕ), 
  pennies = 42 → 
  nickels = 36 → 
  dimes = 15 → 
  remaining_coins = 27 → 
  (pennies + nickels + dimes - remaining_coins) = 66 :=
by
  intros pennies nickels dimes remaining_coins h_pennies h_nickels h_dimes h_remaining_coins
  rw [h_pennies, h_nickels, h_dimes, h_remaining_coins]
  sorry

end roger_donated_coins_l51_51568


namespace hyperbola_foci_coords_l51_51165

theorem hyperbola_foci_coords :
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, 4 * y^2 - 25 * x^2 = 100 →
  (x = 0 ∧ (y = c ∨ y = -c)) := by
  intros a b c x y h
  have h1 : 4 * y^2 = 100 + 25 * x^2 := by linarith
  have h2 : y^2 = 25 + 25/4 * x^2 := by linarith
  have h3 : x = 0 := by sorry
  have h4 : y = c ∨ y = -c := by sorry
  exact ⟨h3, h4⟩

end hyperbola_foci_coords_l51_51165


namespace num_valid_pairs_l51_51400

theorem num_valid_pairs : 
  (∃ (m n : ℤ), 
    1 ≤ m ∧ m ≤ 99 ∧ 
    1 ≤ n ∧ n ≤ 99 ∧ 
    ∃ k : ℤ, (m + n)^2 + 3*m + n = k^2) = 98 :=
sorry

end num_valid_pairs_l51_51400


namespace train_cross_time_l51_51749

def train_length := 100
def bridge_length := 275
def train_speed_kmph := 45

noncomputable def train_speed_mps : ℝ :=
  (train_speed_kmph * 1000.0) / 3600.0

theorem train_cross_time :
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  time = 30 :=
by 
  -- Introduce definitions to make sure they align with the initial conditions
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  -- Prove time = 30
  sorry

end train_cross_time_l51_51749


namespace first_nonzero_digit_right_decimal_l51_51229

/--
  To prove that the first nonzero digit to the right of the decimal point of the fraction 1/137 is 9
-/
theorem first_nonzero_digit_right_decimal (n : ℕ) (h1 : n = 137) :
  ∃ d, d = 9 ∧ (∀ k, 10 ^ k * 1 / 137 < 10^(k+1)) → the_first_nonzero_digit_right_of_decimal_is 9 := 
sorry

end first_nonzero_digit_right_decimal_l51_51229


namespace log_function_monotonic_increase_l51_51586

open Real

noncomputable def monotonic_increase_interval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≤ f y

theorem log_function_monotonic_increase :
  monotonic_increase_interval (λ x, log (x^2 - 6 * x + 8)) (Set.Ioi 4) :=
begin
  sorry
end

end log_function_monotonic_increase_l51_51586


namespace hundredth_number_is_100_l51_51076

/-- Define the sequence of numbers said by Jo, Blair, and Parker following the conditions described. --/
def next_number (turn : ℕ) : ℕ :=
  -- Each turn increments by one number starting from 1
  turn

-- Prove that the 100th number in the sequence is 100
theorem hundredth_number_is_100 :
  next_number 100 = 100 := 
by sorry

end hundredth_number_is_100_l51_51076


namespace xyz_length_correct_l51_51466

def length_of_XYZ (straight_segments : ℕ) (slanted_segments : ℕ) : ℝ :=
  straight_segments * 1 + slanted_segments * Real.sqrt 2

theorem xyz_length_correct : length_of_XYZ 14 5 = 14 + 5 * Real.sqrt 2 := 
by
  sorry

end xyz_length_correct_l51_51466


namespace problem_statement_l51_51913

noncomputable def bernoulli : ℝ → MeasureTheory.Measure ↥ℝ := sorry
noncomputable def binomial : ℕ → ℝ → MeasureTheory.Measure ↥ℝ := sorry

variables (X : MeasureTheory.Measure ↥ℝ) (Y : MeasureTheory.Measure ↥ℝ)
          (p_X : ℝ := 0.7) (n_Y : ℕ := 10) (p_Y : ℝ := 0.8)

-- X follows Bernoulli distribution with success probability of 0.7
hypothesis hX : X = bernoulli p_X

-- Y follows Binomial distribution with n = 10, p = 0.8
hypothesis hY : Y = binomial n_Y p_Y

-- Expected value and variance for X (Bernoulli)
def EX := p_X
def DX := p_X * (1 - p_X)

-- Expected value and variance for Y (Binomial)
def EY := n_Y * p_Y
def DY := n_Y * p_Y * (1 - p_Y)

theorem problem_statement : EX = 0.7 ∧ DX = 0.21 ∧ EY = 8 ∧ DY = 1.6 :=
by {
  sorry
}

end problem_statement_l51_51913


namespace choose_3_positions_out_of_8_correct_sequence_with_8_shots_correct_l51_51279

-- Definition for Part (a)
def choose_3_positions_out_of_8 : Nat :=
  Nat.choose 8 3

-- Theorem for Part (a)
theorem choose_3_positions_out_of_8_correct 
  : choose_3_positions_out_of_8 = 56 := 
by sorry

-- Definition for Part (b)
def sequence_with_8_shots : Nat :=
  Nat.fact 8 / (Nat.fact 3 * Nat.fact 2 * Nat.fact 3)

-- Theorem for Part (b)
theorem sequence_with_8_shots_correct 
  : sequence_with_8_shots = 560 := 
by sorry

end choose_3_positions_out_of_8_correct_sequence_with_8_shots_correct_l51_51279


namespace valid_n_count_l51_51362

def A_n (n: ℕ) : ℚ := (n * (n - 1) * (2 * n - 1) + 9 * n * (n - 1) + 12 * n - 12)/12

theorem valid_n_count : (Finset.filter (λ n, (A_n n).denom = 1) (Finset.range 101)).card = 99 := 
by
  sorry

end valid_n_count_l51_51362


namespace shop_discount_percentage_l51_51743

-- Define the given conditions
def p_bought : ℝ := 560
def p_original : ℝ := 933.33

-- Define the calculation of percentage discount
def discount_pct : ℝ := ((p_original - p_bought) / p_original) * 100

-- The proof statement (without the proof itself, just the statement)
theorem shop_discount_percentage : discount_pct = 40 := by
  sorry

end shop_discount_percentage_l51_51743


namespace chord_length_l51_51887

-- Definitions for the given conditions
def parametricCircle (θ : ℝ) : ℝ × ℝ :=
  (Math.cos θ, Math.sin θ + 2)

def polarLine (ρ θ : ℝ) : ℝ :=
  ρ * Math.sin θ + ρ * Math.cos θ

-- The main statement to prove
theorem chord_length (θ ρ : ℝ) 
    (h1 : ∀ θ, parametricCircle θ = (Math.cos θ, Math.sin θ + 2)) 
    (h2 : ∀ θ ρ, polarLine ρ θ = 1) : 
    ∃ d r, (r = 1) ∧ (0 ≤ r) ∧ (d = ρ / Math.sqrt 2) ∧ (ρ = Math.sqrt 2) → 
    ∀ chord_length, chord_length = 2 * Math.sqrt (r^2 - d^2) := 
sorry

end chord_length_l51_51887


namespace kolya_product_divisors_1024_l51_51087

theorem kolya_product_divisors_1024 :
  let number := 1024
  let screen_limit := 10^16
  ∃ (divisors : List ℕ), 
    (∀ d ∈ divisors, d ∣ number) ∧ 
    (∏ d in divisors, d) = 2^55 ∧ 
    2^55 > screen_limit :=
by
  sorry

end kolya_product_divisors_1024_l51_51087


namespace cos_diff_identity_l51_51399

theorem cos_diff_identity
  (α : ℝ)
  (hα1 : 0 < α)
  (hα2 : α < real.pi / 2)
  (h_tanα : real.tan α = 2) :
  real.cos (α - real.pi / 4) = (3 * real.sqrt 10) / 10 := 
sorry

end cos_diff_identity_l51_51399


namespace triangle_obtuse_l51_51926

/-- In a triangle where one side is twice the length of another and one angle is 30 degrees, 
    the triangle must be obtuse. -/
theorem triangle_obtuse {A B C : Type} [tri : IsTriangle A B C] 
    (h₁ : length A = 2 * length B) (h₂ : ∃ C, angle A B C = 30) : ∃ D, angle B C D > 90 :=
  sorry

end triangle_obtuse_l51_51926


namespace rick_books_division_l51_51983

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l51_51983


namespace car_production_l51_51290

theorem car_production (mp : ℕ) (h1 : 1800 = (mp + 50) * 12) : mp = 100 :=
by
  sorry

end car_production_l51_51290


namespace sum_of_roots_polynomial_l51_51341

theorem sum_of_roots_polynomial : 
  ∀ (x : ℝ), (∃ a b c d e : ℝ, a = 3 ∧ b = -6 ∧ c = -95 ∧ d = -30 ∧ e = -15 ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) → 
  (∑ (i : ℕ) in Finset.range 4, (λ i, x) i) = 2 :=
by
  sorry

end sum_of_roots_polynomial_l51_51341


namespace number_of_sequences_alternating_parity_l51_51903

/-- 
The number of sequences of 8 digits x_1, x_2, ..., x_8 where no two adjacent x_i have the same parity is 781,250.
-/
theorem number_of_sequences_alternating_parity : 
  let num_sequences := 10 * 5^7 
  ∑ x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 (digits : Fin 8 → Fin 10), 
    (∀ i : Fin 7, digits i % 2 ≠ digits (i + 1) % 2) → 1 = 781250 :=
by sorry

end number_of_sequences_alternating_parity_l51_51903


namespace last_digit_fraction_inv_pow_3_15_l51_51631

theorem last_digit_fraction_inv_pow_3_15 : 
  (Nat.Mod (Nat.floor ((1 : ℚ) / (3^15))) 10) = 0 :=
by
  sorry

end last_digit_fraction_inv_pow_3_15_l51_51631


namespace afternoon_sales_l51_51266

variable (x y : ℕ)

theorem afternoon_sales (hx : y = 2 * x) (hy : x + y = 390) : y = 260 := by
  sorry

end afternoon_sales_l51_51266


namespace part2_l51_51882

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part2 (x y : ℝ) (h₁ : |x - y - 1| ≤ 1 / 3) (h₂ : |2 * y + 1| ≤ 1 / 6) :
  f x < 1 := 
by
  sorry

end part2_l51_51882


namespace parking_arrangements_count_l51_51204

-- Define necessary components and assumptions for the problem

inductive Vehicle
| Truck1
| Truck2
| Bus1
| Bus2

open Vehicle

-- Define the parking arrangement and constraints
def is_valid_parking : list (option Vehicle) → Prop
| [] := true
| (none :: rest) := is_valid_parking rest
| (some vh :: rest) :=
  match rest with
  | [] := true
  | (some next_vh :: _) := vh ≠ next_vh ∧ is_valid_parking rest
  | (none :: rest_rest) := is_valid_parking (some vh :: rest_rest)
  end

-- Define the number of valid parking arrangements
def count_valid_parking_arrangements : ℕ :=
  (list.permutations [some Truck1, some Truck2, some Bus1, some Bus2, none, none, none]).count is_valid_parking

theorem parking_arrangements_count :
  count_valid_parking_arrangements = 440 :=
sorry

end parking_arrangements_count_l51_51204


namespace principal_amount_invested_l51_51316

noncomputable def calculate_principal : ℕ := sorry

theorem principal_amount_invested (P : ℝ) (y : ℝ) 
    (h1 : 300 = P * y * 2 / 100) -- Condition for simple interest
    (h2 : 307.50 = P * ((1 + y/100)^2 - 1)) -- Condition for compound interest
    : P = 73.53 := 
sorry

end principal_amount_invested_l51_51316


namespace partitionLShapeGrid_l51_51286

-- Definition of the problem
def canPartitionLShape (n : ℕ) (grid : Fin (3*n + 1) × Fin (3*n + 1) → Bool) : Prop :=
  ∃ partition : List (List (Fin (3*n + 1) × Fin (3*n + 1))),
    (∀ part ∈ partition, isLShape part) ∧
    (⋃ part ∈ part, part).erase (n, n) = { pos | grid pos = true }

-- An auxiliary definition for L-shape
def isLShape (coords : List (Fin n × Fin n)) : Prop :=
  ∃ a b c d : Fin n × Fin n, coords = [a, b, c, d] ∧
  (are_connected a b ∧ are_connected b c ∧ are_connected c d)

-- Connection between points in a grid (e.g., adjacents)
def are_connected (p1 p2 : Fin n × Fin n) : Prop :=
  (p1.1 = p2.1 ∧ |p1.2 - p2.2| = 1) ∨ (p1.2 = p2.2 ∧ |p1.1 - p2.1| = 1)

-- Main theorem statement
theorem partitionLShapeGrid (n : ℕ) (pos : ℕ →ⁿ grid: FinGrid (3 * n + 1)):
  ∃ p : PartitionsIntoLShapes, 
  isPartitionIntoLShapes p grid  :=
 sorry

end partitionLShapeGrid_l51_51286


namespace olivia_total_payment_l51_51552

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l51_51552


namespace two_not_in_star_AB_l51_51839

def f_M (M : Set ℕ) (x : ℕ) : ℤ :=
  if x ∈ M then -1 else 1

def star (M N : Set ℕ) : Set ℕ := {x | f_M M x * f_M N x = -1}

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 2, 4}

theorem two_not_in_star_AB : 2 ∉ star A B := by
  sorry

end two_not_in_star_AB_l51_51839


namespace first_nonzero_digit_one_over_137_l51_51235

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end first_nonzero_digit_one_over_137_l51_51235


namespace arithmetic_sequence_term_count_l51_51046

theorem arithmetic_sequence_term_count (a d n an : ℕ) (h₀ : a = 5) (h₁ : d = 7) (h₂ : an = 126) (h₃ : an = a + (n - 1) * d) : n = 18 := by
  sorry

end arithmetic_sequence_term_count_l51_51046


namespace problem_statement_l51_51514

-- Definitions and conditions from the problem
def g : ℕ → ℤ := sorry
axiom g_defined (x : ℕ) : ∀ (x : ℕ), x ≥ 0 → (∃ y : ℤ, g(x) = y)
axiom g_at_1 : g 1 = 3
axiom g_functional (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b)

-- Proof goal in Lean 4 statement
theorem problem_statement : g 2023 = 9 := 
by
  sorry

end problem_statement_l51_51514


namespace closest_perfect_square_to_350_l51_51656

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l51_51656


namespace cos_negative_pi_minus_alpha_l51_51856

noncomputable def cos_of_alpha (α : ℝ) : ℝ :=
  let x := -3
  let y := 4
  let r := Real.sqrt (x^2 + y^2)
  x / r

noncomputable def cos_of_negative_pi_minus_alpha (α : ℝ) : ℝ :=
  Real.cos (-(π + α))

theorem cos_negative_pi_minus_alpha :
  ∀ (α : ℝ), (∃ P : ℝ × ℝ, P = (-3, 4) ∧ P.1 = -3 ∧ P.2 = 4) →
  cos_of_negative_pi_minus_alpha α = 3 / 5 :=
by
  intro α h
  sorry

end cos_negative_pi_minus_alpha_l51_51856


namespace jane_minimum_packages_l51_51936

theorem jane_minimum_packages
  (motorcycle_cost : ℕ)
  (earnings_per_package : ℕ)
  (fuel_cost_per_package : ℕ)
  (gear_cost : ℕ)
  (total_expenses : ℕ := motorcycle_cost + gear_cost)
  (n : ℕ)
  (h_motorcycle_cost : motorcycle_cost = 3600)
  (h_earnings_per_package : earnings_per_package = 15)
  (h_fuel_cost_per_package : fuel_cost_per_package = 5)
  (h_gear_cost : gear_cost = 200)
  (h_total_expenses : total_expenses = 3800)
  (h_inequality : earnings_per_package * n - fuel_cost_per_package * n ≥ total_expenses) :
  n ≥ 380 :=
begin
  -- This is where the proof would go.
  sorry
end

end jane_minimum_packages_l51_51936


namespace product_mod_five_l51_51637

theorem product_mod_five (a b c : ℕ) (h₁ : a = 1236) (h₂ : b = 7483) (h₃ : c = 53) :
  (a * b * c) % 5 = 4 :=
by
  sorry

end product_mod_five_l51_51637


namespace area_ratio_inequality_l51_51554

theorem area_ratio_inequality {A B C X Y : Type*} [ordered_semiring A] [ordered_semiring B] [ordered_semiring C] 
  (AB : B) (BC : C) (AX : A) (XY : B) (YC : C)
  (angle_1 : AXY = 2 * angle_C)
  (angle_2 : CYX = 2 * angle_A) :
  S_AXYC / S_ABC ≤ (AX^2 + XY^2 + YC^2) / AC^2 :=
sorry

end area_ratio_inequality_l51_51554


namespace number_of_solutions_sin_cos_eq_l51_51358

theorem number_of_solutions_sin_cos_eq (a b : ℝ) (f g : ℝ → ℝ) :
  (∀ x ∈ Icc 0 (2 * π), f (π * cos x) = g (π * sin x))
  → (∃!s ∈ (finset.range (nat.ceil (2 * (π : ℝ)))), f ((π : ℝ) * cos s) = g ((π : ℝ) * sin s)) :=
by
  sorry

end number_of_solutions_sin_cos_eq_l51_51358


namespace first_nonzero_digit_fraction_l51_51226

theorem first_nonzero_digit_fraction :
  (∃ n: ℕ, 0 < n ∧ n < 10 ∧ (n / 137 % 1) * 10 < 10 ∧ ((n / 137 % 1) * 10).floor = 2) :=
sorry

end first_nonzero_digit_fraction_l51_51226


namespace minimum_value_of_fractions_l51_51007

theorem minimum_value_of_fractions (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) : 
  ∃ a b, (0 < a) ∧ (0 < b) ∧ (1 / a + 1 / b = 1) ∧ (∃ t, ∀ x y, (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / y = 1) -> t = (1 / (x - 1) + 4 / (y - 1))) := 
sorry

end minimum_value_of_fractions_l51_51007


namespace mod_exp_remainder_problem_statement_l51_51241

theorem mod_exp_remainder (a b n : ℕ) (h : a ≡ b [MOD n]) : a ^ 35 ≡ b ^ 35 [MOD n] :=
by sorry

theorem problem_statement :
  101 ^ 35 % 100 = 1 :=
by
  have h : 101 ≡ 1 [MOD 100] := by norm_num
  exact mod_exp_remainder 101 1 100 h

end mod_exp_remainder_problem_statement_l51_51241


namespace translate_parabola_up_and_right_l51_51209

theorem translate_parabola_up_and_right (x y : ℝ) :
  (∃ y', y' = x^2 ∧ y = y' + 3) →
  (∃ y'', y = (x - 5)^2 + 3) :=
by
  intros h
  cases h with y' hy'
  use (x - 5)^2 + 3
  sorry

end translate_parabola_up_and_right_l51_51209


namespace solve_system_l51_51992

def system_of_equations : Prop :=
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ x + 2 * y = -2 ∧ x = 2 ∧ y = -2

theorem solve_system : system_of_equations := by
  sorry

end solve_system_l51_51992


namespace mexica_numbers_less_than_2019_l51_51763

def is_mexica (m : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (∃ (d : ℕ), d = (finset.range (n+1)).filter (λ x, n % x = 0)).card ∧ m = n^d

theorem mexica_numbers_less_than_2019 :
  {m : ℕ | m < 2019 ∧ is_mexica m} = {1, 4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 64, 1296} :=
by sorry

end mexica_numbers_less_than_2019_l51_51763


namespace roots_of_polynomial_l51_51806

theorem roots_of_polynomial :
  roots (λ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6) = {1, 2, 3} := 
sorry

end roots_of_polynomial_l51_51806


namespace cookies_fit_in_box_l51_51012

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end cookies_fit_in_box_l51_51012


namespace clever_question_l51_51211

-- Define the conditions as predicates
def inhabitants_truthful (city : String) : Prop := 
  city = "Mars-Polis"

def inhabitants_lying (city : String) : Prop := 
  city = "Mars-City"

def responses (question : String) (city : String) : String :=
  if question = "Are we in Mars-City?" then
    if city = "Mars-City" then "No" else "Yes"
  else if question = "Do you live here?" then
    if city = "Mars-City" then "No" else "Yes"
  else "Unknown"

-- Define the main theorem
theorem clever_question (city : String) (initial_response : String) :
  (inhabitants_truthful city ∨ inhabitants_lying city) →
  responses "Are we in Mars-City?" city = initial_response →
  responses "Do you live here?" city = "Yes" ∨ responses "Do you live here?" city = "No" :=
by
  sorry

end clever_question_l51_51211


namespace largest_fraction_l51_51254

theorem largest_fraction :
  (∀ (A B C D E : ℚ), 
    A = 5 / 12 ∧ B = 7 / 15 ∧ C = 29 / 58 ∧ D = 151 / 303 ∧ E = 199 / 400 → 
    (∃ (max : ℚ), max = C ∧
      (max ≥ A) ∧ (max ≥ B) ∧ (max ≥ D) ∧ (max ≥ E))) :=
by
  intros A B C D E h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h
  cases h with h4 h5
  rw [h1, h2, h3, h4, h5]
  use (29 / 58)
  split
  { refl }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num; sorry }
  { norm_num; sorry }

end largest_fraction_l51_51254


namespace total_games_l51_51298

variable (L : ℕ) -- Number of games the team lost

-- Define the number of wins
def Wins := 3 * L + 14

theorem total_games (h_wins : Wins = 101) : (Wins + L = 130) :=
by
  sorry

end total_games_l51_51298


namespace reconstruct_points_l51_51135

noncomputable def symmetric (x y : ℝ) := 2 * y - x

theorem reconstruct_points (A' B' C' D' B C D : ℝ) :
  (∃ (A B C D : ℝ),
     B = (A + A') / 2 ∧  -- B is the midpoint of line segment AA'
     C = (B + B') / 2 ∧  -- C is the midpoint of line segment BB'
     D = (C + C') / 2 ∧  -- D is the midpoint of line segment CC'
     A = (D + D') / 2)   -- A is the midpoint of line segment DD'
  ↔ (∃ (A : ℝ), A = symmetric D D') → True := sorry

end reconstruct_points_l51_51135


namespace nonnegative_solutions_count_l51_51438

-- Define the equation x^2 + 6x + 9 = 0
def equation (x : ℝ) : Prop := x^2 + 6 * x + 9 = 0

-- Prove that the number of nonnegative solutions to the equation is 0
theorem nonnegative_solutions_count : finset.card ({x : ℝ | equation x ∧ 0 ≤ x}.to_finset) = 0 :=
by
  -- Proof goes here; add 'sorry' for now
  sorry

end nonnegative_solutions_count_l51_51438


namespace second_term_of_infinite_geometric_series_l51_51762

theorem second_term_of_infinite_geometric_series 
    (S : ℝ) (r : ℝ) (a : ℝ) (second_term : ℝ) 
    (hS : S = 20) (hr : r = 1 / 4) 
    (hSum : S = a / (1 - r)) :
    second_term = a * r := by
  have ha : a = 15 := 
    calc
      a = (S * (1 - r)) / 1 := by
        rw [hSum]
        ring
      ... = 15 := by
        rw [hS, hr]
        norm_num
  show second_term = a * r by
    rw [ha, hr]
    norm_num
    sorry

end second_term_of_infinite_geometric_series_l51_51762


namespace problem_l51_51333

-- Define sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y <= -1 }

-- Define set C as a function of a
def C (a : ℝ) : Set ℝ := { x | x < -a / 2 }

-- The statement of the problem: if B ⊆ C, then a < 2
theorem problem (a : ℝ) : (B ⊆ C a) → a < 2 :=
by sorry

end problem_l51_51333


namespace closest_perfect_square_to_350_l51_51674

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l51_51674


namespace range_of_a_l51_51006

noncomputable def sequence (a t : ℝ) (n : ℕ) : ℝ :=
nat.rec_on n a (λ n an, -an^2 + t * an)

def is_monotonically_increasing (f : ℕ → ℝ) : Prop :=
∀ n : ℕ, f n < f (n + 1)

theorem range_of_a (a t : ℝ) (h1 : a > 0)
    (h2 : ∃ t : ℝ, is_monotonically_increasing (sequence a t)) : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l51_51006


namespace log_problem_exp_problem_l51_51281

-- Question 1 Lean Statement
theorem log_problem (a : ℝ) (h₁: log 2 2 = a) :
  log 8 20 - 2 * log 2 20 = (log 2 20) / (3 * a) - 2 * (a + (log 2 10)) :=
sorry

-- Question 2 Lean Statement
theorem exp_problem :
  (Real.log 4)^0 + (9/4)^(-0.5) + Real.sqrt ((1 - Real.sqrt 3)^2) - 2^(Real.logBase 4 3) = (9/2) - 2 * Real.sqrt 3 :=
sorry

end log_problem_exp_problem_l51_51281


namespace closest_perfect_square_to_350_l51_51679

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l51_51679


namespace three_digit_numbers_without_7_l51_51993

theorem three_digit_numbers_without_7 : 
  let total_three_digit_numbers := 900
  let numbers_with_7 := 252
  total_three_digit_numbers - numbers_with_7 = 648 := 
by
  unfold total_three_digit_numbers
  unfold numbers_with_7
  sorry

end three_digit_numbers_without_7_l51_51993


namespace thief_speed_l51_51747

-- Define the conditions
def distance_to_thief_initial : ℝ := 225 / 1000 -- convert meters to kilometers
def speed_policeman : ℝ := 10 -- speed in km/hr
def distance_thief : ℝ := 900 / 1000 -- convert meters to kilometers
def total_distance_policeman : ℝ := distance_to_thief_initial + distance_thief -- sum of distances in km

-- The principal theorem
theorem thief_speed : ∃ v : ℝ, (1.125 / 10 = 0.9 / v) ∧ v = 8 :=
by
  use 8
  sorry -- Proof goes here

end thief_speed_l51_51747


namespace find_k_l51_51398

open_locale classical

variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Define vectors
variables {E : Type*} [add_comm_group E] [vector_space ℝ E]
variables (e1 e2 : E) (k : ℝ)

-- Non-collinear vectors condition
variable (hne : e1 ≠ 0 ∧ e2 ≠ 0 ∧ e1 + -e2 ≠ 0)

-- Given vectors a and b
def a : E := e1 + 2 • e2
def b : E := 2 • e1 - k • e2

-- Collinearity condition
noncomputable def collinear (u v : E) := ∃ t : ℝ, u = t • v

-- The theorem to prove
theorem find_k (hcol : collinear a b) : k = -4 :=
sorry

end find_k_l51_51398


namespace lcm_23_46_827_l51_51354

theorem lcm_23_46_827 :
  (23 * 46 * 827) / gcd (23 * 2) 827 = 38042 := by
  sorry

end lcm_23_46_827_l51_51354


namespace find_b_l51_51816

theorem find_b (b : ℝ) (h : log b 216 = -3 / 2) : b = 1 / 36 :=
sorry

end find_b_l51_51816


namespace min_absolute_sum_value_l51_51243

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end min_absolute_sum_value_l51_51243


namespace extremum_point_of_f_l51_51583

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem extremum_point_of_f : ∃ x, x = 1 ∧ (∀ y ≠ 1, f y ≥ f x) := 
sorry

end extremum_point_of_f_l51_51583


namespace interval_length_proof_l51_51781

-- Define the conditions
def is_lattice_point (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 40 ∧ 1 ≤ y ∧ y ≤ 40

def points_below_line (n : ℚ) (x y : ℕ) : Prop :=
  y ≤ n * x

-- Define the set of lattice points T
def T : finset (ℕ × ℕ) :=
  (finset.range 41).product (finset.range 41).filter (λ p, is_lattice_point p.fst p.snd)

theorem interval_length_proof :
  (∃ (n : ℚ), 
  T.card = 1600 ∧
  ((T.filter (λ p, points_below_line n p.fst p.snd)).card = 400 ∧
  let interval_length := (7/8 : ℚ) - (1/2 : ℚ) in
  let p := interval_length.num in
  let q := interval_length.denom in
  (p + q = 11))) :=
sorry

end interval_length_proof_l51_51781


namespace closest_perfect_square_to_350_l51_51648

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l51_51648


namespace y_intercept_of_line_l51_51219

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end y_intercept_of_line_l51_51219


namespace find_trajectory_l51_51733

noncomputable def trajectory_of_center (r : ℝ) (r_pos : r > 0) : Prop :=
  let O1 := (-3, 0) in
  let O2 := (3, 0) in
  ∀ (M : ℝ × ℝ),
    dist M O1 = 1 + r →
    dist M O2 = 9 - r →
    (dist M O1 + dist M O2 = 10) →
    ((M.fst^2) / 25 + (M.snd^2) / 16 = 1)

theorem find_trajectory :
  trajectory_of_center r r_pos := sorry

end find_trajectory_l51_51733


namespace total_earnings_l51_51498

theorem total_earnings (initial_sign_up_bonus : ℕ) (referral_bonus : ℕ)
  (friends_day1 : ℕ) (friends_week : ℕ) : 
  initial_sign_up_bonus = 5 →
  referral_bonus = 5 →
  friends_day1 = 5 →
  friends_week = 7 →
  initial_sign_up_bonus + 
  (friends_day1 + friends_week) * referral_bonus + 
  (friends_day1 + friends_week) * referral_bonus = 125 :=
by
  intros h1 h2 h3 h4
  calc
    (5 : ℕ) + (12 : ℕ) * 5 + 12 * 5
        = 5 + 60 + 60   : by rw [h1, h2, h3, h4]
    ... = 125           : by norm_num

end total_earnings_l51_51498


namespace elsa_cookie_baking_time_l51_51365

theorem elsa_cookie_baking_time :
  ∀ (baking_time white_icing_time chocolate_icing_time total_time : ℕ), 
  baking_time = 15 →
  white_icing_time = 30 →
  chocolate_icing_time = 30 →
  total_time = 120 →
  (total_time - (baking_time + white_icing_time + chocolate_icing_time)) = 45 :=
by
  intros baking_time white_icing_time chocolate_icing_time total_time
  intros h_baking_time h_white_icing_time h_chocolate_icing_time h_total_time
  rw [h_baking_time, h_white_icing_time, h_chocolate_icing_time, h_total_time]
  have : 15 + 30 + 30 = 75 := rfl
  rw this
  show 120 - 75 = 45
  rfl

end elsa_cookie_baking_time_l51_51365


namespace diagonals_concurrent_of_hexagon_l51_51047

theorem diagonals_concurrent_of_hexagon (ABCDEF : Type*)
  [convex_hexagon ABCDEF]
  (equal_sides : ∀ s, is_side_of_hexagon s ABCDEF → s.length = (side_length_of_hexagon ABCDEF))
  (angle_sum : (∠ A + ∠ C + ∠ E : ℝ) = (∠ B + ∠ D + ∠ F : ℝ)) :
  diagonals_concurrent ABCDEF :=
sorry

end diagonals_concurrent_of_hexagon_l51_51047


namespace problem_statement_l51_51952

theorem problem_statement : 
  let x := (2 + Real.sqrt 2)^6
  let n := Int.floor x
  let f := x - n
  x * (1 - f) = 64 := by 
  sorry

end problem_statement_l51_51952


namespace f_of_1_l51_51585

theorem f_of_1 (f : ℕ+ → ℕ+) (h_mono : ∀ {a b : ℕ+}, a < b → f a < f b)
  (h_fn_prop : ∀ n : ℕ+, f (f n) = 3 * n) : f 1 = 2 :=
sorry

end f_of_1_l51_51585


namespace max_value_of_expression_l51_51355

theorem max_value_of_expression : ∃ x : ℝ, (∀ y : ℝ, 5^y - 25^y ≤ 5^x - 25^x) ∧ (5^x - 25^x = 1 / 4) :=
sorry

end max_value_of_expression_l51_51355


namespace length_of_shortest_side_30_60_90_l51_51587

theorem length_of_shortest_side_30_60_90 (x : ℝ) : 
  (∃ x : ℝ, (2 * x = 15)) → x = 15 / 2 :=
by
  sorry

end length_of_shortest_side_30_60_90_l51_51587


namespace beetle_average_speed_correct_l51_51317

def beetle_average_speed (ant_distance_per_12_minutes : ℝ) 
                         (ant_flat_fraction : ℝ)
                         (ant_sandy_fraction : ℝ) 
                         (ant_gravel_fraction : ℝ)
                         (beetle_flat_fraction : ℝ) 
                         (beetle_sandy_fraction : ℝ)
                         (beetle_gravel_fraction : ℝ)
                         (ant_flat_time : ℝ) 
                         (ant_sandy_time : ℝ)
                         (ant_gravel_time : ℝ)
                         (total_time_minutes : ℕ) : ℝ :=
let ant_flat_distance := (ant_distance_per_12_minutes / 12) * ant_flat_time in
let ant_sandy_distance := ant_flat_distance * (1 - ant_sandy_fraction) * (ant_sandy_time / ant_flat_time) in
let ant_gravel_distance := ant_flat_distance * (1 - ant_gravel_fraction) * (ant_gravel_time / ant_flat_time) in
let ant_total_distance := ant_flat_distance + ant_sandy_distance + ant_gravel_distance in
let beetle_flat_distance := ant_flat_distance * (1 - beetle_flat_fraction) in
let beetle_sandy_distance := beetle_flat_distance * (1 - beetle_sandy_fraction) * (ant_sandy_time / ant_flat_time) in
let beetle_gravel_distance := beetle_flat_distance * (1 - beetle_gravel_fraction) * (ant_gravel_time / ant_flat_time) in
let beetle_total_distance := beetle_flat_distance + beetle_sandy_distance + beetle_gravel_distance in
let beetle_total_distance_km := beetle_total_distance / 1000 in
let total_time_hours := total_time_minutes / 60 in
beetle_total_distance_km / total_time_hours

theorem beetle_average_speed_correct : beetle_average_speed 600 0.1 0.2 0.15 0.05 0.25 4 3 5 12 = 2.2525 :=
by sorry

end beetle_average_speed_correct_l51_51317


namespace number_of_ways_to_read_olympiada_l51_51924

noncomputable def rectangular_table : matrix (fin 4) (fin 9) char :=
  !["O", 'Л', 'И', 'M', 'П', 'И', 'A', 'Д', 'A'],
  !['Л', 'И', 'M', 'П', 'И', 'A', 'Д', 'A', 'O'],
  !['И', 'M', 'П', 'И', 'A', 'Д', 'A', 'O', 'Л'],
  !['M', 'П', 'И', 'A', 'Д', 'A', 'O', 'Л', 'И']

noncomputable def paths_to_read_olympiada (table : matrix (fin 4) (fin 9) char) : ℕ :=
  sorry -- Path calculation to be implemented

theorem number_of_ways_to_read_olympiada :
  paths_to_read_olympiada rectangular_table = 93 :=
sorry

end number_of_ways_to_read_olympiada_l51_51924


namespace problem_statement_l51_51004

noncomputable def parametric_line (t : ℝ) :=
  ( (real.sqrt 2 / 2) * t, (real.sqrt 2 / 2) * t )

noncomputable def parametric_curve (θ : ℝ) :=
  ( real.sqrt 2 + 2 * real.cos θ, 2 * real.sqrt 2 + 2 * real.sin θ )

def point_p : ℝ × ℝ := ( 0, 3 * real.sqrt 2 )

theorem problem_statement :
  let l_polar := (π / 4) in
  let c_polar := ∀ ρ α, ρ^2 - 2 * real.sqrt 2 * ρ * real.cos α - 4 * real.sqrt 2 * ρ * real.sin α + 6 = 0 in
  let intersection_distance := real.sqrt 60 in
  let p_to_l_distance := 3 in
  let area_pab := (1 / 2) * intersection_distance * p_to_l_distance in
  l_polar = (π / 4) ∧
  c_polar ∧
  area_pab = 3 * real.sqrt 15 := by
  sorry

end problem_statement_l51_51004


namespace Olivia_pays_4_dollars_l51_51550

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l51_51550


namespace sum_zero_or_infinite_zero_terms_l51_51117

noncomputable def geometric_sequence {a : ℕ → ℝ} (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem sum_zero_or_infinite_zero_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (hq : geometric_sequence a q)
  (h0 : ∀ n, S n = (Finset.range n).sum a) :
  (∀ n, S n = 0) ∨ (∃ N, ∀ n ≥ N, S n = 0) :=
sorry


end sum_zero_or_infinite_zero_terms_l51_51117


namespace parallel_lines_count_l51_51907

theorem parallel_lines_count (n : ℕ) (h : 7 * (n - 1) = 588) : n = 85 :=
sorry

end parallel_lines_count_l51_51907


namespace necessary_sufficient_condition_l51_51590

theorem necessary_sufficient_condition (a : ℝ) : 
  (∀ x ∈ Icc (-2 : ℝ) 1, a * x^2 + 2 * a * x < 1 - 3 * a) ↔ a < 1 / 6 :=
by
  sorry

end necessary_sufficient_condition_l51_51590


namespace bus_speed_excluding_stoppages_l51_51345

-- Conditions
def speed_including_stoppages : ℝ := 24
def stoppage_time_per_hour : ℝ := 1/2

-- Theorem to prove
theorem bus_speed_excluding_stoppages : ∃ V : ℝ, stoppage_time_per_hour = 1/2 ∧ speed_including_stoppages = 24 ∧ V = 48 :=
by
  sorry

end bus_speed_excluding_stoppages_l51_51345


namespace triangle_ratio_l51_51063

variables (A B C D E F G : Type)
noncomputable theory

-- Define distances
variables (AB AC AE AF BD DC : ℝ) 
variables (ratio : ℝ)

-- Define points as vectors
variables (a b c d e f g : ℝ)

-- Hypotheses
hypothesis h1 : AB = 15
hypothesis h2 : AC = 18
hypothesis h3 : AE = 3 * AF
hypothesis h4 : BD / DC = 2 / 1

-- Ratio to prove
theorem triangle_ratio (h : ℝ) : BD / DC = 2 / 1 → AE = 3 * AF → AB = 15 → AC = 18 → 
  ∃ (EG GF : ℝ), EG / GF = 1 / 2 :=
by sorry

end triangle_ratio_l51_51063


namespace first_nonzero_digit_fraction_l51_51227

theorem first_nonzero_digit_fraction :
  (∃ n: ℕ, 0 < n ∧ n < 10 ∧ (n / 137 % 1) * 10 < 10 ∧ ((n / 137 % 1) * 10).floor = 2) :=
sorry

end first_nonzero_digit_fraction_l51_51227


namespace another_valid_expression_for_24_l51_51216

theorem another_valid_expression_for_24 :
  (8 * 9 / 6) + 8 = 24 :=
by begin
  sorry
end

end another_valid_expression_for_24_l51_51216


namespace min_value_of_f_l51_51181

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log x / Real.log 2) * (Real.log (2 * x) / Real.log 2) else 0

theorem min_value_of_f : ∃ x > 0, f x = -1/4 :=
sorry

end min_value_of_f_l51_51181


namespace combined_mass_of_individuals_l51_51714

-- Define constants and assumptions
def boat_length : ℝ := 4 -- in meters
def boat_breadth : ℝ := 3 -- in meters
def sink_depth_first_person : ℝ := 0.01 -- in meters (1 cm)
def sink_depth_second_person : ℝ := 0.02 -- in meters (2 cm)
def density_water : ℝ := 1000 -- in kg/m³ (density of freshwater)

-- Define volumes displaced
def volume_displaced_first : ℝ := boat_length * boat_breadth * sink_depth_first_person
def volume_displaced_both : ℝ := boat_length * boat_breadth * (sink_depth_first_person + sink_depth_second_person)

-- Define weights (which are equal to the masses under the assumption of constant gravity)
def weight_first_person : ℝ := volume_displaced_first * density_water
def weight_both_persons : ℝ := volume_displaced_both * density_water

-- Statement to prove the combined weight
theorem combined_mass_of_individuals : weight_both_persons = 360 :=
by
  -- Skip the proof
  sorry

end combined_mass_of_individuals_l51_51714


namespace roots_opposite_k_eq_2_l51_51914

theorem roots_opposite_k_eq_2 (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 + x2 = 0 ∧ x1 * x2 = -1 ∧ x1 ≠ x2 ∧ x1*x1 + (k-2)*x1 - 1 = 0 ∧ x2*x2 + (k-2)*x2 - 1 = 0) → k = 2 :=
by
  sorry

end roots_opposite_k_eq_2_l51_51914


namespace triangle_angle_inequality_l51_51560

open Real

theorem triangle_angle_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : α₁ + β₁ + γ₁ = π) :
  (cos α₁ / sin α) + (cos β₁ / sin β) + (cos γ₁ / sin γ) 
  ≤ (cos α / sin α) + (cos β / sin β) + (cos γ / sin γ) :=
sorry

end triangle_angle_inequality_l51_51560


namespace perpendicular_lines_l51_51393

theorem perpendicular_lines (m : ℝ) :
  (∃ k l : ℝ, k * m + (1 - m) * l = 3 ∧ (m - 1) * k + (2 * m + 3) * l = 2) → m = -3 ∨ m = 1 :=
by sorry

end perpendicular_lines_l51_51393


namespace find_x_eq_129_l51_51818

theorem find_x_eq_129 (x : ℕ) (h : sqrt (x + 15) = 12) : x = 129 := by
  sorry

end find_x_eq_129_l51_51818


namespace solution_set_l51_51418

noncomputable def f (x : ℝ) : ℝ := Real.exp(x) + Real.exp(-x) + Real.cos(x)

theorem solution_set (m : ℝ) : 
  f(2 * m) > f(m - 2) ↔ m ∈ Iio (-2) ∪ Ioi (2/3) :=
sorry

end solution_set_l51_51418


namespace converse_of_statement_l51_51164

theorem converse_of_statement (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by {
  sorry
}

end converse_of_statement_l51_51164


namespace maximum_I_minus_J_l51_51361

open Set Filter

noncomputable def I (f : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, x^2 * f x

noncomputable def J (f : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, x * (f x)^2

theorem maximum_I_minus_J (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Icc 0 1)) : 
  I(f) - J(f) ≤ 1/12 :=
sorry

end maximum_I_minus_J_l51_51361


namespace imaginary_part_of_fraction_l51_51875

-- Conditions
variables {x y : ℝ}
def z : ℂ := x + y * Complex.I
def z_eq_condition : Prop := (x / (1 - Complex.I) = 1 + y * Complex.I)
def z_conjugate : ℂ := Complex.conj z

-- Statement to prove
theorem imaginary_part_of_fraction (hx : z_eq_condition) : Complex.im (Complex.norm z / z_conjugate) = Real.sqrt 5 / 5 :=
sorry

end imaginary_part_of_fraction_l51_51875


namespace units_digit_17_pow_2005_l51_51640

theorem units_digit_17_pow_2005 : 
  (17^2005) % 10 = 7 := 
by 
  -- Use the given condition that 17 ≡ 7 (mod 10)
  have h1 : 17 % 10 = 7 := by norm_num,
  -- Compute the modulo 10 of the relevant powers of 7
  have h7_1 : 7 % 10 = 7 := by norm_num,
  have h7_2 : (7^2) % 10 = 9 := by norm_num,
  have h7_3 : (7^3) % 10 = 3 := by norm_num,
  have h7_4 : (7^4) % 10 = 1 := by norm_num,
  -- Calculate 2005 mod 4 to determine the position in the cycle
  have h_position : 2005 % 4 = 1 := by norm_num,
  -- Conclude the cycle position
  rw [←h1, Nat.pow_mod, h_position],
  exact h7_1,
  -- 'sorry' to skip the actual proof construction
  sorry

end units_digit_17_pow_2005_l51_51640


namespace inequality_proof_l51_51836

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51836


namespace problem_solution_l51_51283

theorem problem_solution : (90 + 5) * (12 / (180 / (3^2))) = 57 :=
by
  sorry

end problem_solution_l51_51283


namespace min_value_of_squares_l51_51519

theorem min_value_of_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3 * a * b * c = 8) : 
  ∃ m, m ≥ 4 ∧ ∀ a b c, a^3 + b^3 + c^3 - 3 * a * b * c = 8 → a^2 + b^2 + c^2 ≥ m :=
sorry

end min_value_of_squares_l51_51519


namespace count_valid_binary_sequences_with_constraints_l51_51356

theorem count_valid_binary_sequences_with_constraints : 
  let count_sequences := 
    ∑ i in (Finset.range 4), 
      Nat.choose 7 i * Nat.choose (7 - i) (6 - 2 * i)
  in count_sequences = 357 :=
by
  sorry

end count_valid_binary_sequences_with_constraints_l51_51356


namespace tan_22_5_eq_sqrt_form_l51_51185

theorem tan_22_5_eq_sqrt_form :
  ∃ (a b c d : ℕ), 
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ 
  (tan (real.pi / 8) = real.sqrt a - b + real.sqrt c - real.sqrt d) ∧
  (a + b + c + d = 3) := 
sorry

end tan_22_5_eq_sqrt_form_l51_51185


namespace power_function_is_D_l51_51685

-- Definitions of the functions
def f_A (x : ℝ) : ℝ := 2 * x^(1/2)
def f_B (x : ℝ) : ℝ := x^3 + x
def f_C (x : ℝ) : ℝ := 2^x
def f_D (x : ℝ) : ℝ := x^(1/2)

-- Main Theorem
theorem power_function_is_D : (∀ x : ℝ, (x > 0) → is_power_function f_D) :=
sorry

-- A utility definition that determines if a function is a power function
def is_power_function (f : ℝ → ℝ) : Prop := 
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x^a

end power_function_is_D_l51_51685


namespace evaluate_f_diff_l51_51029

def f (x : ℝ) : ℝ := x^4 + 3 * x^3 + 2 * x^2 + 7 * x

theorem evaluate_f_diff:
  f 6 - f (-6) = 1380 := by
  sorry

end evaluate_f_diff_l51_51029


namespace seven_digit_number_l51_51305

theorem seven_digit_number (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
(h1 : a_1 + a_2 = 9)
(h2 : a_2 + a_3 = 7)
(h3 : a_3 + a_4 = 9)
(h4 : a_4 + a_5 = 2)
(h5 : a_5 + a_6 = 8)
(h6 : a_6 + a_7 = 11)
(h_digits : ∀ (i : ℕ), i ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] → i < 10) :
a_1 = 9 ∧ a_2 = 0 ∧ a_3 = 7 ∧ a_4 = 2 ∧ a_5 = 0 ∧ a_6 = 8 ∧ a_7 = 3 :=
by sorry

end seven_digit_number_l51_51305


namespace compute_3X4_l51_51451

def operation_X (a b : ℤ) : ℤ := b + 12 * a - a^2

theorem compute_3X4 : operation_X 3 4 = 31 := 
by
  sorry

end compute_3X4_l51_51451


namespace sports_club_tennis_members_l51_51925

theorem sports_club_tennis_members 
  (total_members : ℕ)
  (badminton_players : ℕ)
  (neither_players : ℕ)
  (both_players : ℕ)
  (H_total : total_members = 30)
  (H_badminton : badminton_players = 17)
  (H_neither : neither_players = 3)
  (H_both : both_players = 9) :
  let tennis_players := total_members - neither_players - (badminton_players - both_players) in
  tennis_players = 19 :=
by
  let B_union_T := total_members - neither_players
  have H_union : B_union_T = 27, from by simp [H_total, H_neither]
  have H_eq : 27 = badminton_players + (total_members - neither_players - (badminton_players - both_players)) - both_players, from by simp [H_union, H_badminton, H_both]
  let tennis_players := B_union_T - badminton_players + both_players
  have H_tennis : tennis_players = 19 := by simp [H_eq]
  exact H_tennis

end sports_club_tennis_members_l51_51925


namespace find_highest_average_speed_l51_51299

def average_speed (Δdistance : ℝ) (Δtime : ℝ) : ℝ :=
  Δdistance / Δtime

-- Define condition parameters
variables (distances : ℕ → ℝ) -- Distance as a function of time in hours (non-negative integers)
variables (intervals : list (ℕ × ℕ)) -- List of intervals, each defined by a start and end hour

-- Define the specific intervals for consideration
def intervals := [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

-- Define the property we want to prove, which compares average speeds across intervals
def highest_average_speed_interval (distances : ℕ → ℝ) (interval : ℕ × ℕ) : Prop :=
  interval = (4, 6) ∧
  ∀ (i ∈ intervals), 
    average_speed (distances (interval.2) - distances (interval.1)) (interval.2 - interval.1) ≥
    average_speed (distances (i.2) - distances (i.1)) (i.2 - i.1)

-- The theorem statement based on given conditions
theorem find_highest_average_speed (distances : ℕ → ℝ) :
  highest_average_speed_interval distances (4, 6) :=
sorry  -- Proof is omitted

end find_highest_average_speed_l51_51299


namespace slope_of_line_l51_51876

noncomputable def line_eq (x y : ℝ) : Prop := x + 2 * y - 6 = 0

theorem slope_of_line : ∀ x y : ℝ, line_eq x y → ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b :=
by
  assume x y h,
  use -1/2,
  use 3,
  sorry

end slope_of_line_l51_51876


namespace inequality_proof_l51_51834

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l51_51834


namespace retain_two_significant_figures_of_31083_58_is_3_point_1_times_10_pow_4_l51_51136

def round_to_significant_figures (n : ℝ) (figs : ℕ) : ℝ :=
  let scale := 10 ^ (figs - 1 - Int.floor (Real.log10 n))
  (Real.round (n * scale)) / scale

theorem retain_two_significant_figures_of_31083_58_is_3_point_1_times_10_pow_4:
  round_to_significant_figures (31083.58) 2 = 3.1 * 10^4 :=
by
  sorry

end retain_two_significant_figures_of_31083_58_is_3_point_1_times_10_pow_4_l51_51136


namespace john_remaining_income_l51_51536

/-- 
  Mr. John's monthly income is $2000, and he spends 5% of his income on public transport.
  Prove that after deducting his monthly transport fare, his remaining income is $1900.
-/
theorem john_remaining_income : 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  income - transport_fare = 1900 := 
by 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  have transport_fare_eq : transport_fare = 100 := by sorry
  have remaining_income_eq : income - transport_fare = 1900 := by sorry
  exact remaining_income_eq

end john_remaining_income_l51_51536


namespace paint_cost_decrease_l51_51760

variables (C P : ℝ)
variable (cost_decrease_canvas : ℝ := 0.40)
variable (total_cost_decrease : ℝ := 0.56)
variable (paint_to_canvas_ratio : ℝ := 4)

theorem paint_cost_decrease (x : ℝ) : 
  P = 4 * C ∧ 
  P * (1 - x) + C * (1 - cost_decrease_canvas) = (1 - total_cost_decrease) * (P + C) → 
  x = 0.60 :=
by
  intro h
  sorry

end paint_cost_decrease_l51_51760


namespace continuity_sum_l51_51099

noncomputable def piecewise_function (x : ℝ) (a b c : ℝ) : ℝ :=
if h : x > 1 then a * (2 * x + 1) + 2
else if h' : -1 <= x && x <= 1 then b * x + 3
else 3 * x - c

theorem continuity_sum (a b c : ℝ) (h_cont1 : 3 * a = b + 1) (h_cont2 : c = 3 * a + 1) :
  a + c = 4 * a + 1 :=
by
  sorry

end continuity_sum_l51_51099


namespace cookies_per_batch_l51_51499

def family_size := 4
def chips_per_person := 18
def chips_per_cookie := 2
def batches := 3

theorem cookies_per_batch : (family_size * chips_per_person) / chips_per_cookie / batches = 12 := 
by
  -- Proof will go here
  sorry

end cookies_per_batch_l51_51499


namespace cos_alpha_l51_51404

theorem cos_alpha (
    P : ℝ × ℝ,
    hP : P = (-4, 3)
) : let α := real.angle_of_point P in real.cos α = -4 / 5 :=
by {
    let α := real.angle_of_point P,
    have h1 : α = real.angle_of_point (-4, 3), from by rw hP,
    rw h1,
    have h2 : ∥(-4, 3)∥ = 5, -- √((-4)^2 + 3^2)
    {
        have : ∥(-4, 3)∥ = real.sqrt(((-4)^2) + ((3)^2)), from sorry,
        rw this,
        have : (√(16 + 9) = √25), from sorry,
        rw this,
        exact sqrt_eq rfl (by norm_num),
    },
    change ∥P∥ with ∥(-4, 3)∥ at h2,
    rw real.angle_of_point_cos_eq (by exact h2) h1,
    norm_num
}

end cos_alpha_l51_51404


namespace closest_perfect_square_to_350_l51_51672

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l51_51672


namespace john_remaining_income_l51_51535

/-- 
  Mr. John's monthly income is $2000, and he spends 5% of his income on public transport.
  Prove that after deducting his monthly transport fare, his remaining income is $1900.
-/
theorem john_remaining_income : 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  income - transport_fare = 1900 := 
by 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  have transport_fare_eq : transport_fare = 100 := by sorry
  have remaining_income_eq : income - transport_fare = 1900 := by sorry
  exact remaining_income_eq

end john_remaining_income_l51_51535


namespace isosceles_right_triangle_area_l51_51593

noncomputable def triangle_area (p : ℝ) : ℝ :=
  (1 / 8) * ((p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2)) ^ 2

theorem isosceles_right_triangle_area (p : ℝ) :
  let perimeter := p + p * Real.sqrt 2 + 2
  let x := (p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2) / 2
  let area := 1 / 2 * x ^ 2
  area = triangle_area p :=
by
  sorry

end isosceles_right_triangle_area_l51_51593


namespace kolya_screen_display_limit_l51_51081

theorem kolya_screen_display_limit : 
  let n := 1024
  let screen_limit := 10^16
  (∀ d ∈ list.divisors n, d = (2^0) ∨ d = (2^1) ∨ d = (2^2) ∨ d = (2^3) ∨ d = (2^4) ∨ d = (2^5) ∨ d = (2^6) ∨ d = (2^7) ∨ d = (2^8) ∨ d = (2^9) ∨ d = (2^10)) →
  ((list.prod (list.divisors n)) > screen_limit) :=
begin
  sorry
end

end kolya_screen_display_limit_l51_51081


namespace find_minimal_product_l51_51757

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end find_minimal_product_l51_51757


namespace min_value_of_objective_l51_51527

-- Definitions and Conditions
variables (x y z : ℝ)
def condition1 := x * y - 3 ≥ 0
def condition2 := x - y ≥ 1
def condition3 := 2 - y - 3 ≤ 0

-- Objective function
def objective := z = 3 * x + 4 * y

-- Theorem statement for the minimum value of the objective function
theorem min_value_of_objective : condition1 x y ∧ condition2 x y ∧ condition3 y → objective 11 :=
by
  sorry

end min_value_of_objective_l51_51527


namespace triangle_area_percentage_difference_l51_51210

theorem triangle_area_percentage_difference
  (b h : ℝ) : 
  let bA := 1.12 * b in
  let hA := 0.88 * h in
  let area_B := 0.5 * b * h in
  let area_A := 0.5 * bA * hA in
  (area_B - area_A) / area_B * 100 = 1.44 :=
by
  sorry

end triangle_area_percentage_difference_l51_51210


namespace roots_satisfy_equation_l51_51976

noncomputable def polynomial := λ (α β : ℝ), (λ x : ℝ, α * x^3 - α * x^2 + β * x + β)

theorem roots_satisfy_equation (α β x1 x2 x3 : ℝ)
  (hα : α ≠ 0) (hβ : β ≠ 0) 
  (hx1 : polynomial α β x1 = 0) (hx2 : polynomial α β x2 = 0) (hx3 : polynomial α β x3 = 0) :
  (x1 + x2 + x3) * (1 / x1 + 1 / x2 + 1 / x3) = -1 :=
begin
  sorry -- proof not required
end

end roots_satisfy_equation_l51_51976


namespace smallest_number_l51_51789

/-
  Let's declare each number in its base form as variables,
  convert them to their decimal equivalents, and assert that the decimal
  value of $(31)_4$ is the smallest among the given numbers.

  Note: We're not providing the proof steps, just the statement.
-/

noncomputable def A_base7_to_dec : ℕ := 2 * 7^1 + 0 * 7^0
noncomputable def B_base5_to_dec : ℕ := 3 * 5^1 + 0 * 5^0
noncomputable def C_base6_to_dec : ℕ := 2 * 6^1 + 3 * 6^0
noncomputable def D_base4_to_dec : ℕ := 3 * 4^1 + 1 * 4^0

theorem smallest_number : D_base4_to_dec < A_base7_to_dec ∧ D_base4_to_dec < B_base5_to_dec ∧ D_base4_to_dec < C_base6_to_dec := by
  sorry

end smallest_number_l51_51789


namespace first_nonzero_digit_fraction_l51_51225

theorem first_nonzero_digit_fraction :
  (∃ n: ℕ, 0 < n ∧ n < 10 ∧ (n / 137 % 1) * 10 < 10 ∧ ((n / 137 % 1) * 10).floor = 2) :=
sorry

end first_nonzero_digit_fraction_l51_51225


namespace func_passes_through_fixed_point_l51_51172

theorem func_passes_through_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  a^(2 * (1 / 2) - 1) = 1 :=
by
  sorry

end func_passes_through_fixed_point_l51_51172


namespace divisible_by_power_of_3_l51_51521

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∀ n ≥ 1, a (n + 1) = (3 * a n ^ 2 + 1) / 2 - a n

theorem divisible_by_power_of_3 (a : ℕ → ℤ) (k : ℕ) (n := 3^k)
  (h : sequence a) : n ∣ a n :=
by
  sorry

end divisible_by_power_of_3_l51_51521


namespace all_but_finitely_many_repr_as_sum_of_distinct_squares_l51_51149

theorem all_but_finitely_many_repr_as_sum_of_distinct_squares :
  ∃ (K : ℕ), ∀ (n : ℕ), n > K → (∃ (sq : ℕ → Prop), (∀ k, sq k → ∃ m, k = m^2) ∧ (∃ (lst : list ℕ), (∀ x ∈ lst, sq x) ∧ lst.sum = n ∧ lst.nodup)) :=
sorry

end all_but_finitely_many_repr_as_sum_of_distinct_squares_l51_51149


namespace tetrahedron_area_relation_l51_51024

theorem tetrahedron_area_relation
  (O A B C : Type)
  (mutually_perpendicular : OA ⟂ OB ∧ OB ⟂ OC ∧ OC ⟂ OA)
  (S S₁ S₂ S₃ : ℝ)
  (S_area : S = area_face_opposite_O O A B C)
  (S₁_area : S₁ = area_side_face_AOB A O B)
  (S₂_area : S₂ = area_side_face_BOC B O C)
  (S₃_area : S₃ = area_side_face_COA C O A) :
  S^2 = S₁^2 + S₂^2 + S₃^2 :=
by sorry

end tetrahedron_area_relation_l51_51024


namespace angle_B_range_of_arithmetic_sequence_sides_l51_51037

theorem angle_B_range_of_arithmetic_sequence_sides
  (a b c : ℝ)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_arith_seq : 2 * b = a + c) :
  0 < angle B ∧ angle B ≤ π / 3 :=
sorry

end angle_B_range_of_arithmetic_sequence_sides_l51_51037


namespace math_club_total_members_l51_51052

   theorem math_club_total_members:
     ∀ (num_females num_males total_members : ℕ),
     num_females = 6 →
     num_males = 2 * num_females →
     total_members = num_females + num_males →
     total_members = 18 :=
   by
     intros num_females num_males total_members
     intros h_females h_males h_total
     rw [h_females, h_males] at h_total
     exact h_total
   
end math_club_total_members_l51_51052


namespace MrJohnMonthlySavings_l51_51537

theorem MrJohnMonthlySavings
  (monthly_income : ℝ := 2000)
  (percent_spent_on_transport : ℝ := 5 / 100)
  (transport_fare : ℝ := percent_spent_on_transport * monthly_income) :
  (monthly_income - transport_fare) = 1900 :=
begin
  sorry
end

end MrJohnMonthlySavings_l51_51537


namespace minimize_shelves_books_l51_51726

theorem minimize_shelves_books : 
  ∀ (n : ℕ),
    (n > 0 ∧ 130 % n = 0 ∧ 195 % n = 0) → 
    (n ≤ 65) := sorry

end minimize_shelves_books_l51_51726


namespace Whitney_money_left_over_l51_51691

theorem Whitney_money_left_over :
  let posters := 2
  let notebooks := 3
  let bookmarks := 2
  let cost_poster := 5
  let cost_notebook := 4
  let cost_bookmark := 2
  let total_cost_posters := posters * cost_poster
  let total_cost_notebooks := notebooks * cost_notebook
  let total_cost_bookmarks := bookmarks * cost_bookmark
  let total_cost := total_cost_posters + total_cost_notebooks + total_cost_bookmarks
  let initial_money := 2 * 20
  let money_left_over := initial_money - total_cost
  in
  money_left_over = 14 := sorry

end Whitney_money_left_over_l51_51691


namespace clock_angle_in_radians_l51_51576

theorem clock_angle_in_radians :
  let degree_per_minute := 6
  let total_minutes := (2 * 60 + 20)
  let total_degrees := - (degree_per_minute * total_minutes)
  let radians_conversion_factor := (Mathlib.pi / 180)
  let expected_radians := - (14 * Mathlib.pi / 3)
  total_degrees * radians_conversion_factor = expected_radians :=
by
  sorry

end clock_angle_in_radians_l51_51576


namespace shift_graphs_l51_51174

theorem shift_graphs :
  ∀ (x : ℝ),
  let f := λ (x : ℝ), sin x + (sqrt 3) * cos x,
      g := λ (x : ℝ), sin x - (sqrt 3) * cos x in
  g (x + (2 * π / 3)) = f x :=
by
  sorry

end shift_graphs_l51_51174


namespace phi_range_l51_51412

-- Define the function f and the conditions on ω and φ
def f (x : ℝ) (ω φ : ℝ) := 2 * Real.sin (ω * x + φ)

-- Given conditions
axiom ω_pos : ∀ ω : ℝ, ω > 0
axiom φ_bound : ∀ φ : ℝ, |φ| ≤ π / 2

-- Problem statement
noncomputable def distance_between_symmetry_centers := π

theorem phi_range (ω : ℝ) (φ : ℝ) (h_ω : ω = 1) (h_f : ∀ x : ℝ, x ∈ Ioo (-π / 12) (π / 3) → f x ω φ > 1) : (π / 4 ≤ φ ∧ φ ≤ π / 2) :=
sorry

end phi_range_l51_51412


namespace MrJohnMonthlySavings_l51_51538

theorem MrJohnMonthlySavings
  (monthly_income : ℝ := 2000)
  (percent_spent_on_transport : ℝ := 5 / 100)
  (transport_fare : ℝ := percent_spent_on_transport * monthly_income) :
  (monthly_income - transport_fare) = 1900 :=
begin
  sorry
end

end MrJohnMonthlySavings_l51_51538


namespace janice_purchase_l51_51489

theorem janice_purchase : 
  ∃ (a b c : ℕ), a + b + c = 50 ∧ 50 * a + 400 * b + 500 * c = 10000 ∧ a = 23 :=
by
  sorry

end janice_purchase_l51_51489


namespace viewable_area_around_square_l51_51777

-- Define the square and its properties
structure Square where
  side_length : ℝ

-- Define a walk around the boundary of a square
def boundary_of_square (s: Square) : set (ℝ × ℝ) :=
  {(x, y) | (x = 0 ∧ 0 ≤ y ∧ y ≤ s.side_length) ∨ 
           (y = 0 ∧ 0 ≤ x ∧ x ≤ s.side_length) ∨ 
           (x = s.side_length ∧ 0 ≤ y ∧ y <= s.side_length) ∨ 
           (y = s.side_length ∧ 0 ≤ x ∧ x <= s.side_length)}

-- Function defining the viewable region from a point on the path
def viewable_region (p : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  {q : ℝ × ℝ | dist p q ≤ radius}

-- Main theorem statement
theorem viewable_area_around_square : 
  ∃ viewable_area : ℝ, 
    let s := Square.mk 10 in
    let P := boundary_of_square s in
    viewable_area = 157 :=
sorry

end viewable_area_around_square_l51_51777


namespace ellipse_eccentricity_l51_51391

theorem ellipse_eccentricity 
  (a₁ a₂ c : ℝ) (e : ℝ)
  (h₁ : ∃ (F1 F2 P : ℝ), P ∈ ellipse (foci := (F1, F2)) ∧ P ∈ hyperbola (foci := (F1, F2)) ∧ angle F1 P F2 = 60)
  (h₂ : eccentricity hyperbola = (sqrt 2) / 2) :
  e = sqrt 6 / 2 := by
  sorry

end ellipse_eccentricity_l51_51391


namespace find_x_eq_129_l51_51819

theorem find_x_eq_129 (x : ℕ) (h : sqrt (x + 15) = 12) : x = 129 := by
  sorry

end find_x_eq_129_l51_51819


namespace b_2023_value_l51_51109

noncomputable def seq (b : ℕ → ℝ) : Prop := 
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) (h1 : seq b) (h2 : b 1 = 2 + Real.sqrt 5) (h3 : b 1984 = 12 + Real.sqrt 5) : 
  b 2023 = -4/3 + 10 * Real.sqrt 5 / 3 :=
sorry

end b_2023_value_l51_51109


namespace problem_1_l51_51962

open Set

variable (R : Set ℝ)
variable (A : Set ℝ := { x | 2 * x^2 - 7 * x + 3 ≤ 0 })
variable (B : Set ℝ := { x | x^2 + a < 0 })

theorem problem_1 (a : ℝ) : (a = -4 → (A ∩ B = { x : ℝ | 1 / 2 ≤ x ∧ x < 2 } ∧ A ∪ B = { x : ℝ | -2 < x ∧ x ≤ 3 })) ∧
  ((compl A ∩ B = B) → a ≥ -2) := by
  sorry

end problem_1_l51_51962


namespace closest_perfect_square_to_350_l51_51653

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l51_51653


namespace find_a5_l51_51476

variable {a_n : ℕ → ℤ}
variable (d : ℤ)

def arithmetic_sequence (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  a_n 1 = a1 ∧ ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 (h_seq : arithmetic_sequence a_n 6 d) (h_a3 : a_n 3 = 2) : a_n 5 = -2 :=
by
  obtain ⟨h_a1, h_arith⟩ := h_seq
  sorry

end find_a5_l51_51476


namespace general_term_formula_smallest_n_y_n_gt_55_div_9_l51_51526

-- Definitions
def sequence (x : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, 4 * x n - S n - 3 = 0

def another_sequence (x : ℕ → ℚ) (y : ℕ → ℚ) : Prop :=
  y 1 = 2 ∧ ∀ n : ℕ, y (n + 1) - y n = x n

-- Translation of Part (1)
theorem general_term_formula {x : ℕ → ℚ} {S : ℕ → ℚ} (h : sequence x S) :
  ∀ n : ℕ, x n = (4 / 3) ^ (n - 1) :=
sorry

-- Translation of Part (2)
theorem smallest_n_y_n_gt_55_div_9 {x : ℕ → ℚ} {y : ℕ → ℚ} (h1 : sequence x (λ n, ∑ i in Finset.range n, x i))
  (h2 : another_sequence x y) :
  ∃ n : ℕ, y n > 55 / 9 ∧ ∀ m : ℕ, m < n → ¬(y m > 55 / 9) :=
sorry

end general_term_formula_smallest_n_y_n_gt_55_div_9_l51_51526


namespace common_divisors_count_l51_51896

def prime_exponents (n : Nat) : List (Nat × Nat) :=
  if n = 9240 then [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
  else if n = 10800 then [(2, 4), (3, 3), (5, 2)]
  else []

def gcd_prime_exponents (exps1 exps2 : List (Nat × Nat)) : List (Nat × Nat) :=
  exps1.filterMap (fun (p1, e1) =>
    match exps2.find? (fun (p2, _) => p1 = p2) with
    | some (p2, e2) => if e1 ≤ e2 then some (p1, e1) else some (p1, e2)
    | none => none
  )

def count_divisors (exps : List (Nat × Nat)) : Nat :=
  exps.foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem common_divisors_count :
  count_divisors (gcd_prime_exponents (prime_exponents 9240) (prime_exponents 10800)) = 16 :=
by
  sorry

end common_divisors_count_l51_51896


namespace total_lockers_l51_51188

theorem total_lockers (cost_per_digit : ℕ) (total_cost : ℝ) :
  (cost_per_digit = 3) → (total_cost = 273.39) → 
  ∃ n : ℕ, n = 2555 ∧ 
    9 * 1 * (cost_per_digit / 100) + 
    90 * 2 * (cost_per_digit / 100) + 
    900 * 3 * (cost_per_digit / 100) +
    1001 * 4 * (cost_per_digit / 100) +
    555 * 4 * (cost_per_digit / 100) = total_cost := 
by 
  intros h_cost_per_digit h_total_cost
  use 2555
  split
  { refl }
  { 
    rw [h_cost_per_digit],
    norm_num,
    sorry -- proof skipped
  }

end total_lockers_l51_51188


namespace Olivia_pays_4_dollars_l51_51549

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l51_51549


namespace second_player_wins_with_optimal_play_l51_51203

def equilateral_triangle_game_optimal_play_winner (n : ℕ ) : bool :=
  -- Prove that with optimal play, the second player wins
  -- True indicates that the second player wins
  true

theorem second_player_wins_with_optimal_play (n : ℕ) (h : n > 0) : equilateral_triangle_game_optimal_play_winner n =
  true := 
sorry

end second_player_wins_with_optimal_play_l51_51203


namespace total_ages_l51_51775

variable (Bill_age Caroline_age : ℕ)
variable (h1 : Bill_age = 2 * Caroline_age - 1) (h2 : Bill_age = 17)

theorem total_ages : Bill_age + Caroline_age = 26 :=
by
  sorry

end total_ages_l51_51775


namespace sum_distances_to_faces_lt_n_minus_2_l51_51977

theorem sum_distances_to_faces_lt_n_minus_2 {n : ℕ} (P : Point) (polyhedron : Polyhedron n) 
    (dist_to_vertices_at_most_1 : ∀ v ∈ polyhedron.vertices, distance P v ≤ 1) :
    (∑ face in polyhedron.faces, distance P face) < n - 2 :=
by 
  sorry

end sum_distances_to_faces_lt_n_minus_2_l51_51977


namespace last_digit_of_decimal_expansion_of_fraction_l51_51633

theorem last_digit_of_decimal_expansion_of_fraction : 
  let x := (1 : ℚ) / (3 ^ 15) 
  in ∃ digit : ℕ, (digit < 10) ∧ last_digit (decimal_expansion x) = 7 :=
sorry

end last_digit_of_decimal_expansion_of_fraction_l51_51633


namespace sequence_product_equals_two_thirds_l51_51125

theorem sequence_product_equals_two_thirds :
  let a : ℕ → ℝ := λ n, if n = 0 then 1/2 else 1 + (a (n - 1) - 1)^2 in
  ∏ i in (Finset.range ∞), a i = 2/3 :=
by {
  sorry
}

end sequence_product_equals_two_thirds_l51_51125


namespace determine_powers_of_primes_l51_51851

-- Definition of the condition
def satisfies_condition (n N : ℕ) : Prop :=
  ∀ (k : ℕ) (d : Fin k → ℕ), (∀ i, d i ∣ n) →
  (∑ i, 1 / (d i) > N → ∃ S : Finset (Fin k), ∑ i in S, 1 / (d i) = N)

-- Lean theorem statement for the problem
theorem determine_powers_of_primes (N : ℕ) (n : ℕ) :
  (∃ p l : ℕ, Nat.Prime p ∧ n = p^l) ↔ satisfies_condition n N :=
sorry

end determine_powers_of_primes_l51_51851


namespace math_club_total_members_l51_51054

theorem math_club_total_members (female_count : ℕ) (h_female : female_count = 6) (h_male_ratio : ∃ male_count : ℕ, male_count = 2 * female_count) :
  ∃ total_members : ℕ, total_members = female_count + classical.some h_male_ratio :=
by
  let male_count := classical.some h_male_ratio
  have h_male_count : male_count = 12 := by sorry
  existsi (female_count + male_count)
  rw [h_female, h_male_count]
  exact rfl

end math_club_total_members_l51_51054


namespace min_OA_OB_eqn_min_PA_PB_eqn_l51_51382

variable {Point : Type} [InnerProductSpace ℝ Point] (O P A B : Point) (l : AffineSubspace ℝ Point)
variable [AffineMapClass l ℝ Point Point] 

-- Coordinates of the points
def O_coord : Point := (0 : ℝ, 0 : ℝ)
def P_coord : Point := (1 : ℝ, 4 : ℝ)
def A_coord (k : ℝ) : Point := (-4/k + 1, 0)
def B_coord (k : ℝ) : Point := (0, -k + 4)

def line_through_P (k : ℝ) : AffineMap ℝ Point ℝ :=
  AffineMap.mk (λ x, (P_coord.1 + (x - P_coord.2) / k, x)) (by simp)

def intersection_x_axis (k : ℝ) : Point :=
  A_coord k

def intersection_y_axis (k : ℝ) : Point :=
  B_coord k

-- The conditions
def is_line_through_P (k : ℝ) (l : AffineSubspace ℝ Point) : Prop :=
  ∃ P : Point, P = P_coord ∧ line_through_P k = l

-- Problem (1)
theorem min_OA_OB_eqn 
  (h1 : is_line_through_P (-2) l)
  (h2 : O = O_coord) (h3 : P = P_coord): 
  ∃ k : ℝ, k = -2 ∧ l = {x | 2 * (x.1 : ℝ) + (x.2 : ℝ) - 6 = 0} := 
sorry

-- Problem (2)
theorem min_PA_PB_eqn
  (h1 : is_line_through_P (-1) l)
  (h2 : O = O_coord) (h3 : P = P_coord): 
  ∃ k : ℝ, k = -1 ∧ l = {x | (x.1 : ℝ) + (x.2 : ℝ) - 5 = 0} := 
sorry

end min_OA_OB_eqn_min_PA_PB_eqn_l51_51382


namespace total_interest_is_68_l51_51732

-- Definitions of the initial conditions
def amount_2_percent : ℝ := 600
def amount_4_percent : ℝ := amount_2_percent + 800
def interest_rate_2_percent : ℝ := 0.02
def interest_rate_4_percent : ℝ := 0.04
def invested_total_1 : ℝ := amount_2_percent
def invested_total_2 : ℝ := amount_4_percent

-- The total interest calculation
def interest_2_percent : ℝ := invested_total_1 * interest_rate_2_percent
def interest_4_percent : ℝ := invested_total_2 * interest_rate_4_percent

-- Claim: The total interest earned is $68
theorem total_interest_is_68 : interest_2_percent + interest_4_percent = 68 := by
  sorry

end total_interest_is_68_l51_51732


namespace sum_b_n_eq_1893_l51_51857

def arith_seq (a_n : ℕ → ℕ) := ∀ n, a_n n = 1 + (n - 1)

theorem sum_b_n_eq_1893
    (a_n : ℕ → ℕ) (b_n : ℕ → ℕ)
    (S : ℕ → ℕ)
    (h1 : a_n 1 = 1)
    (h2 : S 7 = 28)
    (h3 : ∀ n, a_n n = n)
    (h4 : ∀ n, b_n n = ⌊log 10 (a_n n)⌋) :
  ∑ n in (finset.range 1000).map (finset.embedding.fin_subtype 0 1000), b_n (n + 1) = 1893 := sorry

end sum_b_n_eq_1893_l51_51857


namespace transport_tax_correct_l51_51766

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end transport_tax_correct_l51_51766


namespace xuzhou_test_2014_l51_51474

variables (A B C D : ℝ) -- Assume A, B, C, D are real numbers.

theorem xuzhou_test_2014 :
  (C < D) → (A > B) :=
sorry

end xuzhou_test_2014_l51_51474


namespace annual_interest_rate_is_10_l51_51077

-- Definitions of principal and total amount paid as conditions
def principal : ℝ := 150
def total_amount_paid : ℝ := 165

-- Definition of interest based on the conditions
def interest : ℝ := total_amount_paid - principal

-- Annual interest rate calculation
def annual_interest_rate (P A I : ℝ) : ℝ := (I / P) * 100

-- The theorem to prove the annual interest rate is 10%
theorem annual_interest_rate_is_10 
  (P A I : ℝ) 
  (hP : P = principal) 
  (hA : A = total_amount_paid) 
  (hI : I = interest) : 
  annual_interest_rate P A I = 10 :=
by
  -- Example proof here would go into detail verifying the rate is 10%
  sorry

end annual_interest_rate_is_10_l51_51077


namespace VR_passes_through_fixed_point_l51_51183

-- Define the fixed points and variable points
noncomputable def P : Point := sorry
noncomputable def Q : Point := sorry
def l : Line := sorry
def K : Circle := sorry
def R (variable_on_circle : Point -> Prop) : Point := sorry

-- Define the condition of not collinear
def not_collinear (P Q R : Point) : Prop := ¬ collinear P Q R

-- Define V as the intersection point
def V (circle_through_PQR : Circle) (line_l : Line) : Point := sorry

-- The main theorem stating the problem
theorem VR_passes_through_fixed_point :
  ∃ (W : Point), ∀ (R : Point), (variable_on_circle R) ∧ not_collinear P Q R ∧ V (circle_through_PQR) l = V' →
  passes_through (line VR) W :=
sorry

end VR_passes_through_fixed_point_l51_51183


namespace angle_between_a_b_l51_51395

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Hypotheses
def h1 : a + b + c = 0 :=
sorry

def h2 : ∥a∥ = 1 :=
sorry

def h3 : ∥b∥ = 2 :=
sorry

def h4 : ∥c∥ = sqrt 7 :=
sorry

-- Goal
theorem angle_between_a_b : 
  let θ := real.arccos ((inner a b) / (∥a∥ * ∥b∥)) in
  θ = real.pi / 3 :=
sorry

end angle_between_a_b_l51_51395


namespace extreme_values_of_f_l51_51416

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + x

theorem extreme_values_of_f :
  isLocalMin f (1 / 3) ∧ f (1 / 3) = 4 / 27 ∧
  isLocalMax f 1 ∧ f 1 = 0 :=
by
  sorry

end extreme_values_of_f_l51_51416


namespace passed_percentage_l51_51470

theorem passed_percentage (A B C AB BC AC ABC: ℝ) 
  (hA : A = 0.25) 
  (hB : B = 0.50) 
  (hC : C = 0.30) 
  (hAB : AB = 0.25) 
  (hBC : BC = 0.15) 
  (hAC : AC = 0.10) 
  (hABC : ABC = 0.05) 
  : 100 - (A + B + C - AB - BC - AC + ABC) = 40 := 
by 
  rw [hA, hB, hC, hAB, hBC, hAC, hABC]
  norm_num
  sorry

end passed_percentage_l51_51470


namespace triangle_ABC_properties_l51_51457

noncomputable def tan_sum_equal (A B : ℝ) : Prop :=
  tan A + tan B = sqrt 3 * (tan A * tan B - 1)

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (h_c : c = 7 / 2)
  (h_area : (1 / 2) * a * b * sin C = 3 * sqrt 3 / 2)
  (h_tan_eq : tan_sum_equal A B)
  (h_angles : A + B + C = π) -- Sum of angles in a triangle
  : C = π / 3 ∧ a + b = 11 / 2 := 
sorry

end triangle_ABC_properties_l51_51457


namespace divide_correct_result_l51_51263

theorem divide_correct_result (x : ℕ) (h : 40 + x = 52) : 24 / x = 2 :=
by
  have hx : x = 52 - 40 := by linarith
  rw hx
  norm_num

end divide_correct_result_l51_51263


namespace arithmetic_and_geometric_sequences_correct_l51_51759

theorem arithmetic_and_geometric_sequences_correct : ∀ (a : ℕ → ℝ),
  (∀ m n s t : ℕ, m ≠ 0 → n ≠ 0 → s ≠ 0 → t ≠ 0 → 
    (a_m + a_n = a_s + a_t → m + n = s + t) → false)
  ∧ (∀ S : ℕ → ℝ, (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) →
    ∃ d : ℝ, ∀ n : ℕ, S (2 * n) - S n = S n + n^2 * d)
  ∧ (∀ S : ℕ → ℝ, (∀ n : ℕ, S n = a_1 * (1 - (r ^ n)) / (1 - r)) → 
      ∃ r : ℝ, r ≠ 1 ∧ S (2 * n) - S n = r * (S (3 * n) - S (2 * n)) → false)
  ∧ (∀ S : ℕ → ℝ, ∃ A B : ℝ, A ≠ 0 ∧ B ≠ 0 ∧ (∀ n : ℕ, S n = A *(real.exp n) + B) → 
      A + B = 0) :=
by sorry

end arithmetic_and_geometric_sequences_correct_l51_51759


namespace kolya_cannot_see_full_result_l51_51091

theorem kolya_cannot_see_full_result : let n := 1024 in 
                                      let num_decimals := 16 in 
                                      (num_divisors_product n > 10^num_decimals) := 
begin
    let n := 1024,
    let num_decimals := 16,
    have product := (∑ i in range (nat.log2 n), i),
    have sum_of_powers := (2^product),
    have required_digits := (10^num_decimals),
    exact sum_of_powers > required_digits,
end

end kolya_cannot_see_full_result_l51_51091


namespace common_element_in_good_subsets_l51_51105

noncomputable def finite_set (A : Type) := A
noncomputable def is_good_subset (A : Type) (good : set (set A)) : Prop :=
  ∀ (B : set A), (|B| = 1024154) → ∀ (S : set A), (S ∈ good) ∧ (S ⊆ B) → (∃ a, ∀ (S : set A), S ∈ good → a ∈ S)

theorem common_element_in_good_subsets
  {A : Type} (S : finite_set A) (good : set (set A))
  (Hgood : ∀ (B : set A), (|B| = 1024154) → ∀ (S : set A), is_good_subset A good → (∃ a, ∀ (S : set A), S ∈ good → a ∈ S)) :
  ∃ a, ∀ (S : set A), S ∈ good → a ∈ S :=
  sorry

end common_element_in_good_subsets_l51_51105


namespace first_nonzero_digit_one_over_137_l51_51233

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end first_nonzero_digit_one_over_137_l51_51233


namespace problem_l51_51846

variables {α : Type*} [linear_ordered_field α]

def M : set α := {x | x^2 - 5 * x + 6 = 0}
def N (a : α) : set α := {x | a * x = 12}
def A : set α := {0, 4, 6}

noncomputable def non_empty_proper_subsets (s : set α) : set (set α) :=
{ t | t ⊆ s ∧ t ≠ ∅ ∧ t ≠ s }

theorem problem {a : α} (h : N a ⊆ M) :
  A = {0, 4, 6} ∧ non_empty_proper_subsets A = {{0}, {4}, {6}, {0, 4}, {0, 6}, {4, 6}} :=
sorry

end problem_l51_51846


namespace probability_even_sum_5_balls_drawn_l51_51715

theorem probability_even_sum_5_balls_drawn :
  let total_ways := (Nat.choose 12 5)
  let favorable_ways := (Nat.choose 6 0) * (Nat.choose 6 5) + 
                        (Nat.choose 6 2) * (Nat.choose 6 3) + 
                        (Nat.choose 6 4) * (Nat.choose 6 1)
  favorable_ways / total_ways = 1 / 2 :=
by sorry

end probability_even_sum_5_balls_drawn_l51_51715


namespace find_possible_values_of_m_l51_51782

theorem find_possible_values_of_m (m : ℕ) (h1 : 1 ≤ m ∧ m ≤ 720) (h2 : |6 * m - 0.5 * m| = 1) :
  m = 262 ∨ m = 458 :=
sorry

end find_possible_values_of_m_l51_51782


namespace divisors_product_exceeds_16_digits_l51_51097

theorem divisors_product_exceeds_16_digits :
  let n := 1024
  let screen_digits := 16
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  (String.length (Natural.toString divisors_product) > screen_digits) := 
by {
  let n := 1024
  let screen_digits := 16
  let divisors := List.range (11)
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  show (String.length (Natural.toString divisors_product) > screen_digits),
  sorry
}

end divisors_product_exceeds_16_digits_l51_51097


namespace find_function_l51_51800

def f (n : ℤ) : ℤ := n + (-1) ^ n

theorem find_function (f : ℤ → ℤ)
  (H1 : ∀ n : ℤ, f n + f (n + 1) = 2 * n + 1)
  (H2 : ∑ i in finset.range 64, f (i + 1) = 2015) :
  ∀ n : ℤ, f n = n + (-1) ^ n :=
by
  sorry

end find_function_l51_51800


namespace compute_f_1986_l51_51025

noncomputable def f : ℕ → ℤ :=
sorry -- Placeholder for the function definition

axiom f_def (x : ℕ) : x ≥ 0 → f x -- f is defined for all integers x ≥ 0

axiom f_1 : f 1 = 1

axiom f_2 : f 2 = 3

axiom functional_eq (a b : ℕ) : f (a + b) = f a + f b - 2 * f (a * b)

theorem compute_f_1986 : f 1986 = 3 :=
sorry

end compute_f_1986_l51_51025


namespace min_abs_sum_value_l51_51247

def abs_sum (x : ℝ) := |x + 3| + |x + 6| + |x + 7|

theorem min_abs_sum_value : ∃ x : ℝ, abs_sum x = 4 ∧ ∀ y : ℝ, abs_sum y ≥ abs_sum x := 
by 
  use -6
  have abs_sum_eq : abs_sum (-6) = 4 := by
    simp [abs_sum]
  -- Other conditions ensuring it is the minimum
  sorry

end min_abs_sum_value_l51_51247


namespace vacuum_total_time_l51_51069

theorem vacuum_total_time (x : ℕ) (hx : 2 * x + 5 = 27) :
  27 + x = 38 :=
by
  sorry

end vacuum_total_time_l51_51069


namespace inequality_proof_l51_51401

open Real

theorem inequality_proof 
  {n : ℕ}
  (a : Fin n → ℝ)
  (b : Fin n → ℝ)
  (h : (b 0) ^ 2 - ∑ i in Finset.range n \ {0}, (b i) ^ 2 > 0)
  : (∑ i in Finset.range n, (a i) ^ 2 - 2 * ∑ i in Finset.range n, (a i)^2)
  * ((b 0)^2 - ∑ i in Finset.range n \ {0}, (b i)^2)
  ≤ (∑ i in Finset.range n, (a i * b i))^2 := 
sorry

end inequality_proof_l51_51401


namespace intersection_area_after_5_minutes_rotation_l51_51214

noncomputable def area_of_intersection_after_rotation : ℝ := 
  let θ := real.pi / 6 in
  (1 - real.cos θ) * (1 - real.sin θ)

theorem intersection_area_after_5_minutes_rotation : 
  area_of_intersection_after_rotation = (2 - real.sqrt 3) / 4 := 
sorry

end intersection_area_after_5_minutes_rotation_l51_51214


namespace Simon_has_72_legos_l51_51150

theorem Simon_has_72_legos 
  (Kent_legos : ℕ)
  (h1 : Kent_legos = 40) 
  (Bruce_legos : ℕ) 
  (h2 : Bruce_legos = Kent_legos + 20) 
  (Simon_legos : ℕ) 
  (h3 : Simon_legos = Bruce_legos + (Bruce_legos/5)) :
  Simon_legos = 72 := 
  by
    -- Begin proof (not required for the problem)
    -- Proof steps would follow here
    sorry

end Simon_has_72_legos_l51_51150


namespace units_digit_of_product_l51_51815

theorem units_digit_of_product :
  let units_digit (n : ℕ) := n % 10 in
  (units_digit (17^2) = 9) ∧ (units_digit 29 = 9) →
  units_digit (17^2 * 29) = 1 :=
by
  intros units_digit h
  have h1 : units_digit 17 = 7, by sorry
  have h2 : units_digit (17^2) = (units_digit (7^2)), by sorry
  have h3 : units_digit (7^2) = 9, by sorry
  have h4 : units_digit 29 = 9, by sorry
  have h5 : units_digit ((units_digit (7^2)) * (units_digit 29)) = units_digit (9 * 9), by sorry
  have h6 : units_digit (9 * 9) = 1, by sorry
  exact h6

end units_digit_of_product_l51_51815


namespace compute_expression_l51_51778

noncomputable def log3 (x : ℝ) : ℝ := real.log x / real.log 3
noncomputable def log5 (x : ℝ) : ℝ := real.log x / real.log 5

theorem compute_expression : 
  let a := 2 * log3 2 - log3 (32 / 9) + log3 8
  let b := 5^(log5 3)
  a - b = -1 :=
by
  sorry

end compute_expression_l51_51778


namespace length_of_field_l51_51273

variable (w l : ℝ)
variable (H1 : l = 2 * w)
variable (pond_area : ℝ := 64)
variable (field_area : ℝ := l * w)
variable (H2 : pond_area = (1 / 98) * field_area)

theorem length_of_field : l = 112 :=
by
  sorry

end length_of_field_l51_51273


namespace eccentricity_range_l51_51872

theorem eccentricity_range (a b k : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_k : k ≠ 0) :
  let c := Real.sqrt (a^2 + b^2) in
  let e := c / a in
  1 < e ∧ e ≤ (2 * Real.sqrt 3) / 3 :=
by
  sorry

end eccentricity_range_l51_51872


namespace find_smallest_angle_20gon_l51_51166

theorem find_smallest_angle_20gon :
  ∃ (a1 : ℤ) (d : ℤ), 
    (∀ i, 1 ≤ i ∧ i ≤ 20 → 
      let angle := a1 + (i - 1) * d in 
      a1 + (i - 1) * d < 176 ∧ a1 + (i - 1) * d = 143) ∧
    a1 + 19 * d < 176 ∧ 20 * (a1 + (10 - 1) * d + a1 + 10 * d) / 2 = 3240 :=
sorry

end find_smallest_angle_20gon_l51_51166


namespace max_value_of_S_n_divided_l51_51390

noncomputable def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def S_n (a₁ d n : ℕ) : ℕ :=
  n * (n + 4)

theorem max_value_of_S_n_divided (a₁ d : ℕ) (h₁ : ∀ n, a₁ + (2 * n - 1) * d = 2 * (a₁ + (n - 1) * d) - 3)
  (h₂ : (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d)) :
  ∃ n, 2 * S_n a₁ d n / 2^n = 6 := 
sorry

end max_value_of_S_n_divided_l51_51390


namespace length_of_train_l51_51310

-- Definitions based on the conditions
def speed_kmph := 36 -- speed in km/h
def conversion_factor := 5 / 18 -- conversion factor from km/h to m/s
def speed_mps := speed_kmph * conversion_factor -- convert speed to m/s
def time_seconds := 16 -- time in seconds

-- The main statement: the length of the train in meters
theorem length_of_train : (speed_mps * time_seconds) = 160 := by
  -- Calculations based on the conditions
  have converted_speed : speed_mps = 10 := by
    sorry -- detailed conversion proof skipped
  have length : (speed_mps * time_seconds) = (10 * 16) := by
    sorry -- multiplications skipped
  show (speed_mps * time_seconds) = 160 from length

end length_of_train_l51_51310


namespace distinct_roots_of_quadratic_l51_51520

variable {a b : ℝ}
-- condition: a and b are distinct
variable (h_distinct: a ≠ b)

theorem distinct_roots_of_quadratic (a b : ℝ) (h_distinct : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x + a)*(x + b) = 2*x + a + b :=
by
  sorry

end distinct_roots_of_quadratic_l51_51520


namespace incorrect_statement_l51_51258

/-- A line segment with direction is called a directed line segment. -/
def directed_line_segment (a b : ℝ) : Prop := ∃ d : ℝ, d > 0 ∧ b = a + d

/-- The direction of the zero vector is indeterminate. -/
def direction_of_zero_vector_is_indeterminate : Prop := ∀ (v : ℝ), v = 0 → ∀ (d : ℝ), ¬(d = 0 → direction_of v = some d)

/-- Directed line segments that are parallel and of equal length represent the same vector. -/
def same_vector (a b c d : ℝ) : Prop := directed_line_segment a b ∧ directed_line_segment c d ∧ (b - a = d - c)

theorem incorrect_statement (a b c d : ℝ) :
  (directed_line_segment a b) →
  direction_of_zero_vector_is_indeterminate →
  (same_vector a b c d) →
  ¬(directed_line_segment a b = vector a b) :=
by
  intros
  sorry

end incorrect_statement_l51_51258


namespace find_k_l51_51905

variable (m n k : ℝ)

-- Conditions from the problem
def quadratic_roots : Prop := (m + n = -2) ∧ (m * n = k) ∧ (1/m + 1/n = 6)

-- Theorem statement
theorem find_k (h : quadratic_roots m n k) : k = -1/3 :=
sorry

end find_k_l51_51905


namespace closest_perfect_square_to_350_l51_51676

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l51_51676


namespace eccentricity_of_hyperbola_l51_51959

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : ℝ :=
  let c := real.sqrt (a^2 + b^2) in
  let P : ℝ × ℝ := (a * real.sqrt 5, b * real.sqrt (1 - 1/5)) in -- a simplifying assumption of the intersection
  let PF1 : ℝ := 4 * a in -- Given from the problem condition that |PF1| = 4a
  let PF2 : ℝ := 2 * a in -- Given from the problem condition that |PF2| = 2a
  let e : ℝ := c / a in
  e  -- The eccentricity

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  hyperbola_eccentricity a b h_a h_b = real.sqrt 5 :=
sorry 

end eccentricity_of_hyperbola_l51_51959


namespace bases_with_final_digit_one_in_725_in_base_10_l51_51366

theorem bases_with_final_digit_one_in_725_in_base_10 : 
  (nat.count (λ b, 2 ≤ b ∧ b ≤ 9 ∧ 725 % b = 1) (list.range' 2 8)) = 2 := 
sorry

end bases_with_final_digit_one_in_725_in_base_10_l51_51366


namespace area_triangle_QPO_l51_51048

variables {A B C D P Q O N M : Type} [linear_ordered_field k]

-- Given conditions in the problem
variables (AB CD BC AD : A → B → C → D → Type)
variables (trisect : BC → N )
variables (extension_twice_AB : AB → P )
variables (trisect_AD : AD → M)
variables (extension_twice_B : B → Q)
variables (intersection_DP_CQ : P → Q → O)
variables (area_ABCD : k)

-- The statement asserting the area of triangle QPO
theorem area_triangle_QPO (htrisect1 : DP trisects BC at N nearest to B)
                          (hextension1 : DP meets AB extended twice at P)
                          (htrisect2 : CQ trisects AD at M nearest to A)
                          (hextension2 : CQ meets the line through B extended twice at Q)
                          (hintersection : DP and CQ meet at O)
                          (harea_parallelogram : area of parallelogram ABCD = k) :
                          (area of triangle QPO = 5/6 * k) :=
sorry

end area_triangle_QPO_l51_51048


namespace line_circle_intersection_l51_51421

theorem line_circle_intersection (k : ℝ) :
  ∃ x y : ℝ, y = k * (x + 1 / 2) ∧ x^2 + y^2 = 1 :=
sorry

end line_circle_intersection_l51_51421


namespace max_distance_point_Q_to_line_min_area_triangle_AOB_l51_51422

-- Part (1) condition
def line_l (m : ℚ) : (ℚ × ℚ) → Prop := λ p, (2 * m + 1) * p.1 - (3 + m) * p.2 + m - 7 = 0

-- Part (1) proof problem
theorem max_distance_point_Q_to_line :
  let m := -22 / 19 in ∃ (P : ℚ × ℚ), line_l m P ∧ 
               (∃ (Q : ℚ × ℚ), Q = (3, 4) ∧ 
                 ∀ m' : ℚ, line_l m' Q → m' = m ∧
                   (distance (3, 4) P = real.sqrt 74)) := 
sorry

-- Part (2) conditions
def line_eq (m k : ℚ) : (ℚ × ℚ) → ℚ := λ p, p.2 + 3 - k * (p.1 + 2)
def intersects_x_neg_half_axis (p : ℚ × ℚ) : Prop := p.2 = 0 ∧ p.1 < 0
def intersects_y_neg_half_axis (p : ℚ × ℚ) : Prop := p.1 = 0 ∧ p.2 < 0

-- Part (2) proof problem
theorem min_area_triangle_AOB :
  ∃ (m k : ℚ), k < 0 ∧ 
                let A := (3 / k - 2, 0),
                    B := (0, 2 * k - 3) in 
                intersects_x_neg_half_axis A ∧
                intersects_y_neg_half_axis B ∧
                m = -3 / k ∧
                (area (3 / k - 2, 0) (0, 2 * k - 3) (0,0) = 12) ∧
                let line := line_eq m k in
                line = (λ p, 3 * p.1 + 2 * p.2 + 12) :=
sorry

end max_distance_point_Q_to_line_min_area_triangle_AOB_l51_51422


namespace problem1_problem2_l51_51329

-- Problem 1
theorem problem1 : sqrt 32 - sqrt 18 + sqrt (1 / 2) = (3 * sqrt 2) / 2 := 
  sorry

-- Problem 2
theorem problem2 : (2 * sqrt 3 + 1) * (2 * sqrt 3 - 1) - (sqrt 3 - 1)^2 = 7 + 2 * sqrt 3 := 
  sorry

end problem1_problem2_l51_51329


namespace jill_water_jars_total_count_l51_51074

theorem jill_water_jars_total_count :
  ∃ (x : ℕ), (1/4 : ℝ) * x + (1/2 : ℝ) * x + x = 35 ∧ 3 * x = 60 :=
begin
  -- Jill has x quarts, x half-gallons, and x gallons
  -- The sum of the volumes is 35 gallons
  -- Prove that the total number of jars (3x) is 60
  
  sorry
end

end jill_water_jars_total_count_l51_51074


namespace transportation_connect_any_city_l51_51461

-- Define a type for cities.
constant City : Type

-- Predicate representing that two cities are connected by an airline.
constant connected_by_airline : City → City → Prop

-- Predicate representing that two cities are connected by a canal.
constant connected_by_canal : City → City → Prop

-- condition that for any two cities, they're either connected by an airline or by a canal.
axiom connection_exists : ∀ (c1 c2 : City), connected_by_airline c1 c2 ∨ connected_by_canal c1 c2

-- Prove that for any number of cities (n ≥ 2), there exists a means of transportation that allows traveling from any city to any other.
theorem transportation_connect_any_city (n : ℕ) (h : n ≥ 2)
  (cities : Fin n → City) :
  (∀ i j, ∃ f : Fin n → Fin n, connected_by_airline (cities (f i)) (cities (f j)))
  ∨ (∀ i j, ∃ f : Fin n → Fin n, connected_by_canal (cities (f i)) (cities (f j))) :=
sorry

end transportation_connect_any_city_l51_51461


namespace James_wait_weeks_l51_51070

def JamesExercising (daysPainSubside : ℕ) (healingMultiplier : ℕ) (delayAfterHealing : ℕ) (totalDaysUntilHeavyLift : ℕ) : ℕ :=
  let healingTime := daysPainSubside * healingMultiplier
  let startWorkingOut := healingTime + delayAfterHealing
  let waitingPeriodDays := totalDaysUntilHeavyLift - startWorkingOut
  waitingPeriodDays / 7

theorem James_wait_weeks : 
  JamesExercising 3 5 3 39 = 3 :=
by
  sorry

end James_wait_weeks_l51_51070


namespace inequality_proof_l51_51828

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51828


namespace abs_sub_eq_sub_abs_eq_self_abs_sub_eq_sub_calculate_expression_1_calculate_sum_2022_l51_51700

theorem abs_sub_eq_sub (a b : ℝ) (h : a < b) : |a - b| = b - a := sorry

theorem abs_eq_self (c : ℝ) (h : c > 0) : |c| = c := sorry

theorem abs_sub_eq_sub' (d e : ℝ) (h : d ≥ e) : |d - e| = d - e := sorry

theorem calculate_expression_1 :
  |(1/5) - (150/557)| + |(150/557) - (1/2)| - |- (1/2)| = -(1/5) := sorry

theorem calculate_sum_2022 :
  (∑ k in Finset.range 2022, |(1/(k+2)) - (1/(k+1))|) = 1/2 - 1/2022 := (505 : ℚ) / 1011 := sorry

end abs_sub_eq_sub_abs_eq_self_abs_sub_eq_sub_calculate_expression_1_calculate_sum_2022_l51_51700


namespace asymptotes_of_hyperbola_l51_51420

theorem asymptotes_of_hyperbola
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∃ k : ℝ, (∀ x y, y^2 = (k * x)^2 / a^2 - x^2 / b^2 → 
           ((x - 2)^2 + y^2 = 1 → d = ∑ 1)) → 
  ∃ m n : ℝ, y = ±(sqrt 3/3)x :=
begin 
  sorry
end

end asymptotes_of_hyperbola_l51_51420


namespace quadratic_unbounded_above_l51_51811

theorem quadratic_unbounded_above : ∀ (x y : ℝ), ∃ M : ℝ, ∀ z : ℝ, M < (2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z) :=
by
  intro x y
  use 1000 -- Example to denote that for any point greater than 1000
  intro z
  have h1 : 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z ≥ 2 * 0^2 + 4 * 0 * y + 5 * y^2 + 8 * 0 - 6 * y + z := by sorry
  sorry

end quadratic_unbounded_above_l51_51811


namespace rick_division_steps_l51_51987

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l51_51987


namespace angle_ADF_eq_angle_ADE_l51_51928

-- Define an acute-angled triangle
variable (A B C : Type) [bundle A] [bundle B] [bundle C]

-- Define point D as the foot of the altitude from A to BC
variable (D : Type) [Located D A B C]

-- Define point P on AD
variable (P : Type) [Located P A D]

-- Line BP intersects AC at E
-- Line CP intersects AB at F
variable (E F : Type) [Located E B P C] [Located F C P B]

-- Prove ∠ADF = ∠ADE
theorem angle_ADF_eq_angle_ADE :
  ∠ B A C ∧ Geometry.is_acute_angle ∠ B A C →
  Geometry.is_altitude A B C D →
  Geometry.on_line P A D →
  Geometry.intersect_line B P C E →
  Geometry.intersect_line C P B F →
  Geometry.angle A D F = Geometry.angle A D E :=
begin
  sorry
end

end angle_ADF_eq_angle_ADE_l51_51928


namespace complex_z_imaginary_quadratic_roots_l51_51407

/-- Lean 4 Statement for the proof problem --/

theorem complex_z_imaginary (z : ℂ) (h1 : z.im = 0) (h2 : (z + 2) ^ 2 + 8 * complex.I).im = 0 :
    z = 2 * complex.I :=
by
  sorry

theorem quadratic_roots (p q : ℝ) (h : 2 * (complex.I - 1) ∈ set.range (λ x, 2 * x^2 + p * x + q)) :
    p = 4 ∧ q = 10 :=
by
  sorry

end complex_z_imaginary_quadratic_roots_l51_51407


namespace closest_perfect_square_to_350_l51_51675

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l51_51675


namespace acute_triangle_inequality_l51_51113

theorem acute_triangle_inequality 
    {A B C I : Type} 
    [triangle ABC]
    (h1: is_acute ABC)
    (h2: is_incenter I ABC) :
    3 * (dist A I ^ 2 + dist B I ^ 2 + dist C I ^ 2) >= dist A B ^ 2 + dist B C ^ 2 + dist C A ^ 2 :=
by 
    sorry

end acute_triangle_inequality_l51_51113


namespace monica_total_students_l51_51534

theorem monica_total_students 
  (c1 : ∀ (i: ℕ), i = 1 → i.students = 20)
  (c2 : ∀ (i: ℕ), (i = 2 ∨ i = 3) → i.students = 25)
  (c3 : ∀ (i: ℕ), i = 4 → i.students = c1 1 / 2)
  (c4 : ∀ (i: ℕ), (i = 5 ∨ i = 6) → i.students = 28)
  : (Σ i, i.students) = 136 := 
by
  sorry

end monica_total_students_l51_51534


namespace milly_needs_flamingoes_l51_51128

theorem milly_needs_flamingoes
  (flamingo_feathers : ℕ)
  (pluck_percent : ℚ)
  (num_boas : ℕ)
  (feathers_per_boa : ℕ)
  (pluckable_feathers_per_flamingo : ℕ)
  (total_feathers_needed : ℕ)
  (num_flamingoes : ℕ)
  (h1 : flamingo_feathers = 20)
  (h2 : pluck_percent = 0.25)
  (h3 : num_boas = 12)
  (h4 : feathers_per_boa = 200)
  (h5 : pluckable_feathers_per_flamingo = flamingo_feathers * pluck_percent)
  (h6 : total_feathers_needed = num_boas * feathers_per_boa)
  (h7 : num_flamingoes = total_feathers_needed / pluckable_feathers_per_flamingo)
  : num_flamingoes = 480 := 
by
  sorry

end milly_needs_flamingoes_l51_51128


namespace slope_angle_of_line_x_plus_2_eq_0_l51_51187

noncomputable def slope_angle_of_vertical_line : ℝ := Real.pi / 2

theorem slope_angle_of_line_x_plus_2_eq_0 :
  let line_eq := λ (x : ℝ), x + 2
  line_eq = (λ x, 0) → slope_angle_of_vertical_line = Real.pi / 2 := by
  intro line_eq
  intro h
  have h_vertical : ∀ x, line_eq x = 0 := by
    intro x
    exact h
  sorry

end slope_angle_of_line_x_plus_2_eq_0_l51_51187


namespace hyperbola_eccentricity_l51_51178

theorem hyperbola_eccentricity
  (a b c : ℝ) (x y : ℝ)
  (a_pos : 0 < a) (b_pos : 0 < b)
  (h_hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F1 : (ℝ × ℝ)) (F2 : (ℝ × ℝ))
  (h_F1 : F1 = (-c, 0)) 
  (h_F2 : F2 = (c, 0))
  (P : (ℝ × ℝ)) (h_P1 : P.1 = x) (h_P2 : P.2 = y)
  (h_P_on_l1 : y = b / a * x)
  (h_PF1_perp_l2 : (y - 0) / (x - (-c)) = - b / a)
  (h_PF2_para_l2 : (y - 0) / (x - c) = - b / a) :
  let e := c / a in
  e = 2 := by
  sorry

end hyperbola_eccentricity_l51_51178


namespace tea_sales_revenue_l51_51708

theorem tea_sales_revenue (x : ℝ) (price_last_year price_this_year : ℝ) (yield_last_year yield_this_year : ℝ) (revenue_last_year revenue_this_year : ℝ) :
  price_this_year = 10 * price_last_year →
  yield_this_year = 198.6 →
  yield_last_year = 198.6 + 87.4 →
  revenue_this_year = 198.6 * price_this_year →
  revenue_last_year = yield_last_year * price_last_year →
  revenue_this_year = revenue_last_year + 8500 →
  revenue_this_year = 9930 := 
by
  sorry

end tea_sales_revenue_l51_51708


namespace common_divisors_count_l51_51899

-- Definitions of the numbers involved
def n1 : ℕ := 9240
def n2 : ℕ := 10800

-- Prime factorizations based on the conditions
def factor_n1 : Prop := n1 = 2^3 * 3^1 * 5^1 * 7 * 11
def factor_n2 : Prop := n2 = 2^3 * 3^3 * 5^2

-- GCD as defined in the conditions
def gcd_value : ℕ := Nat.gcd n1 n2

-- Proof problem: prove the number of positive divisors of the gcd of n1 and n2 is 16
theorem common_divisors_count (h1 : factor_n1) (h2 : factor_n2) : Nat.divisors (Nat.gcd n1 n2).card = 16 := by
  sorry

end common_divisors_count_l51_51899


namespace proof_problem_l51_51916

axiom Triangle {α : Type*} [inner_product_space ℝ α] {p1 p2 p3 : α}
  (XY : dist p1 p2 = 13)
  (XZ : dist p1 p3 = 14)
  (YZ : dist p2 p3 = 15) : Prop

noncomputable def trigonometric_expression {α : Type*} [inner_product_space ℝ α] {p1 p2 p3 : α}
  (h : Triangle (XY : dist p1 p2 = 13) (XZ : dist p1 p3 = 14) (YZ : dist p2 p3 = 15)) : ℝ :=
  (cos ((angle p1 p2 p3 - angle p1 p3 p2) / 2)) / (sin (angle p2 p3 p1 / 2)) -
  (sin ((angle p1 p2 p3 - angle p1 p3 p2) / 2)) / (cos (angle p2 p3 p1 / 2))

theorem proof_problem {α : Type*} [inner_product_space ℝ α] {p1 p2 p3 : α}
  (h : Triangle (XY : dist p1 p2 = 13) (XZ : dist p1 p3 = 14) (YZ : dist p2 p3 = 15)) :
  trigonometric_expression h = 28 / 13 := 
sorry

end proof_problem_l51_51916


namespace milly_needs_flamingoes_l51_51127

theorem milly_needs_flamingoes
  (flamingo_feathers : ℕ)
  (pluck_percent : ℚ)
  (num_boas : ℕ)
  (feathers_per_boa : ℕ)
  (pluckable_feathers_per_flamingo : ℕ)
  (total_feathers_needed : ℕ)
  (num_flamingoes : ℕ)
  (h1 : flamingo_feathers = 20)
  (h2 : pluck_percent = 0.25)
  (h3 : num_boas = 12)
  (h4 : feathers_per_boa = 200)
  (h5 : pluckable_feathers_per_flamingo = flamingo_feathers * pluck_percent)
  (h6 : total_feathers_needed = num_boas * feathers_per_boa)
  (h7 : num_flamingoes = total_feathers_needed / pluckable_feathers_per_flamingo)
  : num_flamingoes = 480 := 
by
  sorry

end milly_needs_flamingoes_l51_51127


namespace sum_of_arithmetic_sequence_15_terms_l51_51786

/-- An arithmetic sequence starts at 3 and has a common difference of 4.
    Prove that the sum of the first 15 terms of this sequence is 465. --/
theorem sum_of_arithmetic_sequence_15_terms :
  let a := 3
  let d := 4
  let n := 15
  let aₙ := a + (n - 1) * d
  (n / 2) * (a + aₙ) = 465 :=
by
  sorry

end sum_of_arithmetic_sequence_15_terms_l51_51786


namespace exists_pairs_of_stops_l51_51198

def problem := ∃ (A1 B1 A2 B2 : Fin 6) (h1 : A1 < B1) (h2 : A2 < B2),
  (A1 ≠ A2 ∧ A1 ≠ B2 ∧ B1 ≠ A2 ∧ B1 ≠ B2) ∧
  ¬(∃ (a b : Fin 6), A1 = a ∧ B1 = b ∧ A2 = a ∧ B2 = b) -- such that no passenger boards at A1 and alights at B1
                                                              -- and no passenger boards at A2 and alights at B2.

theorem exists_pairs_of_stops (n : ℕ) (stops : Fin n) (max_passengers : ℕ) 
  (h : n = 6 ∧ max_passengers = 5 ∧ 
  ∀ (a b : Fin n), a < b → a < stops ∧ b < stops) : problem :=
sorry

end exists_pairs_of_stops_l51_51198


namespace necessity_of_A_for_B_l51_51375

variables {a b h : ℝ}

def PropA (a b h : ℝ) : Prop := |a - b| < 2 * h
def PropB (a b h : ℝ) : Prop := |a - 1| < h ∧ |b - 1| < h

theorem necessity_of_A_for_B (h_pos : 0 < h) : 
  (∀ a b, PropB a b h → PropA a b h) ∧ ¬ (∀ a b, PropA a b h → PropB a b h) :=
by sorry

end necessity_of_A_for_B_l51_51375


namespace complex_magnitude_value_l51_51840

theorem complex_magnitude_value (m : ℝ) (hm : |Complex.mk (-1 : ℝ) (2 * m : ℝ)| = 5) : m = Real.sqrt 6 :=
sorry

end complex_magnitude_value_l51_51840


namespace find_m_l51_51891

theorem find_m (m : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (2, -4) ∧ b = (-3, m) ∧ (‖a‖ * ‖b‖ + (a.1 * b.1 + a.2 * b.2)) = 0) : m = 6 := 
by 
  sorry

end find_m_l51_51891


namespace ellipse_standard_equation_l51_51859

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x/a)^2 + (y/b)^2 = 1

noncomputable def passes_through (x y b : ℝ) : Prop :=
  (0, 1) = (x, b)

noncomputable def eccentricity (a b e : ℝ) : Prop :=
  e = Real.sqrt(1 - (b/a)^2)

noncomputable def line_intersection (m : ℝ) : Prop :=
  ∃ x : ℝ, (x, (1/2)*x + m)

noncomputable def intersection_xaxis (m : ℝ) : Prop :=
  (x, 0) = (-2*m, 0)

noncomputable def distance_constant (m : ℝ) : Prop :=
  ∃ b n : ℝ, b = sqrt(10) / 2

theorem ellipse_standard_equation and_const_distance :
  ∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = (sqrt 3) / 2 ∧ 
  ellipse_equation a b ∧ passes_through 0 1 b ∧ eccentricity a b c ∧
  ∀ (m : ℝ), line_intersection m → intersection_xaxis m → distance_constant m :=
sorry

end ellipse_standard_equation_l51_51859


namespace sequence_sum_l51_51642

-- Define the sequence by its nth term for odd and even n
def seq (n : ℕ) : ℤ := if n % 2 = 1 then 3 + 5 * (n / 2) else -(3 + 5 * ((n - 1) / 2))

theorem sequence_sum : (∑ n in Finset.range 15, seq n) = 36 := by
  sorry

end sequence_sum_l51_51642


namespace coffee_ounces_per_cup_l51_51071

theorem coffee_ounces_per_cup
  (persons : ℕ)
  (cups_per_person_per_day : ℕ)
  (cost_per_ounce : ℝ)
  (total_spent_per_week : ℝ)
  (total_cups_per_day : ℕ)
  (total_cups_per_week : ℕ)
  (total_ounces : ℝ)
  (ounces_per_cup : ℝ) :
  persons = 4 →
  cups_per_person_per_day = 2 →
  cost_per_ounce = 1.25 →
  total_spent_per_week = 35 →
  total_cups_per_day = persons * cups_per_person_per_day →
  total_cups_per_week = total_cups_per_day * 7 →
  total_ounces = total_spent_per_week / cost_per_ounce →
  ounces_per_cup = total_ounces / total_cups_per_week →
  ounces_per_cup = 0.5 :=
by
  sorry

end coffee_ounces_per_cup_l51_51071


namespace olivia_total_payment_l51_51551

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l51_51551


namespace decagon_perimeter_30_l51_51240

theorem decagon_perimeter_30 (s : ℝ) (n : ℕ) (hs : s = 3) (hn : n = 10) : 
  let P := n * s in P = 30 := 
by
  sorry

end decagon_perimeter_30_l51_51240


namespace white_marbles_count_l51_51200

section Marbles

variable (total_marbles black_marbles red_marbles green_marbles white_marbles : Nat)

theorem white_marbles_count
  (h_total: total_marbles = 60)
  (h_black: black_marbles = 32)
  (h_red: red_marbles = 10)
  (h_green: green_marbles = 5)
  (h_color: total_marbles = black_marbles + red_marbles + green_marbles + white_marbles) : 
  white_marbles = 13 := 
by
  sorry 

end Marbles

end white_marbles_count_l51_51200


namespace complementary_event_A_l51_51467

def EventA (n : ℕ) := n ≥ 2

def ComplementaryEventA (n : ℕ) := n ≤ 1

theorem complementary_event_A (n : ℕ) : ComplementaryEventA n ↔ ¬ EventA n := by
  sorry

end complementary_event_A_l51_51467


namespace sum_of_log_floors_l51_51376

def F (m : ℕ) : ℕ := ⌊log 2 m⌋

theorem sum_of_log_floors :
  (∑ k in finset.range (2^10 + 1), F (2^10 + k)) = 10 * 2^10 + 1 :=
sorry

end sum_of_log_floors_l51_51376


namespace example_problem_requirements_l51_51110

noncomputable def numberOfValuesSatisfyingConditions : ℂ :=
  let f (z : ℂ) := -complex.I * conj z
  let modulusCondition (z : ℂ) := abs z = 3
  let functionCondition (z : ℂ) := f z = z
  let eligibleValues := {z : ℂ | modulusCondition z ∧ functionCondition z}
  #nat eligibleValues

theorem example_problem_requirements :
  let f (z : ℂ) := -complex.I * conj z
  let modulusCondition (z : ℂ) := abs z = 3
  let functionCondition (z : ℂ) := f z = z
  let eligibleValues := {z : ℂ | modulusCondition z ∧ functionCondition z}
  #nat eligibleValues = 2 := sorry

end example_problem_requirements_l51_51110


namespace iterative_average_difference_l51_51779

noncomputable def iterative_average (lst : List ℝ) : ℝ :=
match lst with
| [] => 0
| [a] => a
| [a, b] => (a + b) / 2
| [a, b, c] => (a + b + c) / 3
| a :: b :: c :: rest => 
  iterative_average ((((a + b + c) / 3 + rest.head) / 2) :: rest.tail)

theorem iterative_average_difference : 
  iterative_average [4, 5, 6, 7, 8] - iterative_average [8, 7, 6, 5, 4] = 2 := 
by 
  sorry

end iterative_average_difference_l51_51779


namespace led_message_count_l51_51199

theorem led_message_count : 
  let n := 7
  let colors := 2
  let lit_leds := 3
  let non_adjacent_combinations := 10
  (non_adjacent_combinations * (colors ^ lit_leds)) = 80 :=
by
  sorry

end led_message_count_l51_51199


namespace value_of_X_when_S_reaches_15000_l51_51911

def X : Nat → Nat
| 0       => 5
| (n + 1) => X n + 3

def S : Nat → Nat
| 0       => 0
| (n + 1) => S n + X (n + 1)

theorem value_of_X_when_S_reaches_15000 :
  ∃ n, S n ≥ 15000 ∧ X n = 299 := by
  sorry

end value_of_X_when_S_reaches_15000_l51_51911


namespace find_points_on_line_l51_51581

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-2 - (Real.sqrt 2) * t, 3 + (Real.sqrt 2) * t)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem find_points_on_line (t : ℝ) :
  distance (parametric_line t) (-2, 3) = Real.sqrt 2 →
  ((parametric_line t = (-3, 4)) ∨ (parametric_line t = (-1, 2))) :=
by
  sorry

end find_points_on_line_l51_51581


namespace matrix_determinant_eq_16_l51_51792

theorem matrix_determinant_eq_16 (x : ℝ) :
  (3 * x) * (4 * x) - (2 * x) = 16 ↔ x = 4 / 3 ∨ x = -1 :=
by sorry

end matrix_determinant_eq_16_l51_51792


namespace isosceles_triangle_sides_l51_51034

theorem isosceles_triangle_sides (P : ℝ) (a b c : ℝ) (h₀ : P = 26) (h₁ : a = 11) (h₂ : a = b ∨ a = c)
  (h₃ : a + b + c = P) : 
  (b = 11 ∧ c = 4) ∨ (b = 7.5 ∧ c = 7.5) :=
by
  sorry

end isosceles_triangle_sides_l51_51034


namespace closest_perfect_square_to_350_l51_51649

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l51_51649


namespace closest_perfect_square_to_350_l51_51647

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l51_51647


namespace arc_length_greater_than_2_l51_51607

-- Define the mathematical objects and conditions
variables {C1 C2 : Circle} {A B : Point}
variable {L : Arc C2}

-- State the conditions as hypotheses
hypothesis h1 : C1.intersect(C2) = {A, B}
hypothesis h2 : C1.radius = 1
hypothesis h3 : L ⊆ C1 ∧ L ∪ (C1 \ L) = C1 ∧ (area L = π/2)

-- State the theorem
theorem arc_length_greater_than_2 : length L > 2 :=
sorry

end arc_length_greater_than_2_l51_51607


namespace fifth_individual_selected_is_01_l51_51309

def random_table_part : list ℕ := [65, 72, 08, 02, 63, 14, 07, 02, 43, 69, 97, 08, 01]

def valid_individuals (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 20

theorem fifth_individual_selected_is_01 :
  let unique_valid_individuals := (random_table_part.filter valid_individuals).nodupkeys in
  nth unique_valid_individuals 4 = some 01 := by
  sorry

end fifth_individual_selected_is_01_l51_51309


namespace minimize_abs_difference_and_product_l51_51755

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end minimize_abs_difference_and_product_l51_51755


namespace cost_of_fencing_is_8750_rsquare_l51_51274

variable (l w : ℝ)
variable (area : ℝ := 7500)
variable (cost_per_meter : ℝ := 0.25)
variable (ratio_lw : ℝ := 4/3)

theorem cost_of_fencing_is_8750_rsquare :
  (l / w = ratio_lw) → 
  (l * w = area) → 
  (2 * (l + w) * cost_per_meter = 87.50) :=
by 
  intros h1 h2
  sorry

end cost_of_fencing_is_8750_rsquare_l51_51274


namespace diplomats_spoke_french_l51_51772

theorem diplomats_spoke_french :
  let T := 120
  let N_E := 32
  let N_F_N_E := 0.20 * T
  let B := 0.10 * T
  let Only_French := N_E - N_F_N_E
  let F := Only_French + B
  F = 20 :=
by
  let T := 120
  let N_E := 32
  let N_F_N_E := 0.20 * T
  let B := 0.10 * T
  let Only_French := N_E - N_F_N_E
  let F := Only_French + B
  sorry

end diplomats_spoke_french_l51_51772


namespace triangle_PQR_area_l51_51931

theorem triangle_PQR_area
  (QR PS QT : ℝ)
  (hQR : QR = 10)
  (hPS : PS = 6)
  (hQT : QT = 8) :
  ½ * QR * PS = 30 := 
sorry

end triangle_PQR_area_l51_51931


namespace magician_strategy_works_l51_51295

def magician_trick (n : ℕ) (hn : n ≥ 3) : Prop :=
  ∃ strategy, ∀ (initial_placement : list (ℕ × ℕ))
    (assistant_placement : list (ℕ × ℕ)),
    ... -- more details needed to specify the strategies and placements

theorem magician_strategy_works : ∀ n, n ≥ 3 → magician_trick n := by
  sorry

end magician_strategy_works_l51_51295


namespace minimum_value_parallel_lines_l51_51432

theorem minimum_value_parallel_lines (a b c : ℝ) (ha : a ≠ 0)
  (hf : ∀ x, f(x) = a*x + b)
  (hg : ∀ x, g(x) = a*x + c)
  (h_min_f : ∀ x, (f(x))^2 + 5*g(x) ≥ -17) :
  ∃ x, (g(x))^2 + 5*f(x) = 9 / 2 :=
by {
  sorry
}

end minimum_value_parallel_lines_l51_51432


namespace analytical_expression_l51_51852

theorem analytical_expression (k : ℝ) (h : k ≠ 0) (x y : ℝ) (hx : x = 4) (hy : y = 6) 
  (eqn : y = k * x) : y = (3 / 2) * x :=
by {
  sorry
}

end analytical_expression_l51_51852


namespace count_possible_c_values_l51_51822

theorem count_possible_c_values :
  let c_values := nat.filter (λ c, 
    ∃ x : ℝ, (9 * (⌊ x ⌋ : ℝ)) + 3 * (⌈ x ⌉ : ℝ) = c) 
    (list.range 501),
    count c_values = 84 := 
begin
  sorry
end

end count_possible_c_values_l51_51822


namespace number_of_valid_sequences_l51_51900

-- Definitions for conditions
def digit := Fin 10 -- Digit can be any number from 0 to 9
def is_odd (n : digit) : Prop := n.val % 2 = 1
def is_even (n : digit) : Prop := n.val % 2 = 0

def valid_sequence (s : Fin 8 → digit) : Prop :=
  ∀ i : Fin 7, (is_odd (s i) ↔ is_even (s (i+1)))

-- Theorem statement
theorem number_of_valid_sequences : 
  ∃ n, n = 781250 ∧ 
    ∃ s : (Fin 8 → digit), valid_sequence s :=
sorry -- Proof is not required

end number_of_valid_sequences_l51_51900


namespace combinations_of_three_toppings_l51_51140

theorem combinations_of_three_toppings : 
  ∃ n m : ℕ, n = 7 ∧ m = 3 ∧ nat.choose n m = 35 :=
by
  sorry

end combinations_of_three_toppings_l51_51140


namespace sequence_property_l51_51030

theorem sequence_property (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 9) : 7 * n * 15873 = n * 111111 :=
by sorry

end sequence_property_l51_51030


namespace find_number_l51_51606

/-- 
  Given that 23% of a number x is equal to 150, prove that x equals 15000 / 23.
-/
theorem find_number (x : ℝ) (h : (23 / 100) * x = 150) : x = 15000 / 23 :=
by
  sorry

end find_number_l51_51606


namespace closest_square_to_350_l51_51659

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l51_51659


namespace h_difference_l51_51360

-- Define σ(n) as the sum of all positive divisors of n.
def σ (n : ℕ) : ℕ :=
  ∑ i in divisors n, i

-- Define h(n) as the quotient obtained when σ(n) is divided by n.
def h (n : ℕ) : ℝ :=
  σ n / n

-- The theorem to prove the difference of h(450) and h(225).
theorem h_difference : h 450 - h 225 = 403 / 450 := by
  sorry

end h_difference_l51_51360


namespace volume_of_solid_of_revolution_l51_51194

theorem volume_of_solid_of_revolution :
  let y := λ x : ℝ, real.sqrt (2 * x)
  (∫ x in 0..1, 2 * x) * real.pi = real.pi :=
by
  have : ∫ x in 0..1, 2 * x = 1 := sorry
  rw [this, mul_one]
  exact rfl

end volume_of_solid_of_revolution_l51_51194


namespace tom_total_expenditure_l51_51208

noncomputable def tom_spent_total : ℝ :=
  let skateboard_price := 9.46
  let skateboard_discount := 0.10 * skateboard_price
  let discounted_skateboard := skateboard_price - skateboard_discount

  let marbles_price := 9.56
  let marbles_discount := 0.10 * marbles_price
  let discounted_marbles := marbles_price - marbles_discount

  let shorts_price := 14.50

  let figures_price := 12.60
  let figures_discount := 0.20 * figures_price
  let discounted_figures := figures_price - figures_discount

  let puzzle_price := 6.35
  let puzzle_discount := 0.15 * puzzle_price
  let discounted_puzzle := puzzle_price - puzzle_discount

  let game_price_eur := 20.50
  let game_discount_eur := 0.05 * game_price_eur
  let discounted_game_eur := game_price_eur - game_discount_eur
  let exchange_rate := 1.12
  let discounted_game_usd := discounted_game_eur * exchange_rate

  discounted_skateboard + discounted_marbles + shorts_price + discounted_figures + discounted_puzzle + discounted_game_usd

theorem tom_total_expenditure : abs (tom_spent_total - 68.91) < 0.01 :=
by norm_num1; sorry

end tom_total_expenditure_l51_51208


namespace integral_evaluation_l51_51796

noncomputable def a_n (n : ℕ) (x : ℝ) : ℝ := ∑ i in Finset.range n, x^i

noncomputable def b_n (n : ℕ) (x : ℝ) : ℝ := ∑ i in Finset.range n, (2 * i + 1) * x^i

noncomputable def f (n : ℕ) : ℝ := ∫ x in 0..1, a_n n x * b_n n x

theorem integral_evaluation (n : ℕ) : f(n) = n^2 :=
by
  sorry

end integral_evaluation_l51_51796


namespace closest_perfect_square_to_350_l51_51680

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l51_51680


namespace classroom_student_count_l51_51604

theorem classroom_student_count (n : ℕ) (students_avg : ℕ) (teacher_age : ℕ) (combined_avg : ℕ) 
  (h1 : students_avg = 8) (h2 : teacher_age = 32) (h3 : combined_avg = 11) 
  (h4 : (8 * n + 32) / (n + 1) = 11) : n + 1 = 8 :=
by
  sorry

end classroom_student_count_l51_51604


namespace ratio_of_segments_l51_51555

theorem ratio_of_segments (A B C D E F G H : Point) (m n : ℕ) (mx nx : ℝ) (x : ℝ) 
  (h_square : is_square A B C D)
  (h_BE : BE = AF = mx) 
  (h_EA : EA = FD = nx)
  (h_ratio : BE / EA = AF / FD = 2022 / 2023)
  (h_EC : EC = sqrt ((mx)^2 + ((m + n) * x)^2))
  (h_FC : FC = sqrt ((nx)^2 + ((m + n) * x)^2)) 
  (h_G : EG = (m / (2 * m + n)) * EC)
  (h_H : FH = (n / (2 * n + m)) * FC)
  (h_BG : BG = sqrt(2 * (m^2) * ((m + n)^2) * x^2 / (2 * m + n)^2))
  (h_DH : DH = sqrt(2 * (n^2) * ((m + n)^2) * x^2 / (2 * n + m)^2)) :
  GH / BD = 12271519 / 36814556 
:= by
  sorry

end ratio_of_segments_l51_51555


namespace simplify_complex_fraction_l51_51151

theorem simplify_complex_fraction : 
  (6 - 3 * Complex.I) / (-2 + 5 * Complex.I) = (-27 / 29) - (24 / 29) * Complex.I := 
by 
  sorry

end simplify_complex_fraction_l51_51151


namespace sum_a2_a9_l51_51873

variable {a : ℕ → ℝ} -- Define the sequence a_n
variable {S : ℕ → ℝ} -- Define the sum sequence S_n

-- The conditions
def arithmetic_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = (n * (a 1 + a n)) / 2

axiom S_10 : arithmetic_sum S a 10
axiom S_10_value : S 10 = 100

-- The goal
theorem sum_a2_a9 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 10 = 100) (h2 : arithmetic_sum S a 10) :
  a 2 + a 9 = 20 := 
sorry

end sum_a2_a9_l51_51873


namespace total_goals_is_21_l51_51265

   theorem total_goals_is_21 (x : ℝ) (h1 : 5 * (x + 0.2) = 4 * x + 5) : 4 * x + 5 = 21 :=
   by
     have h2 : x = 4,
     { 
       calc 
         5 * (x + 0.2) = 4 * x + 5 : h1
         ... ↔ 5 * x + 1 = 4 * x + 5 : by norm_num [mul_add]
         ... ↔ x = 4 : by linarith 
     },
     calc
       4 * x + 5 = 4 * 4 + 5 : by rw [h2]
       ... = 21 : by norm_num
   
   
end total_goals_is_21_l51_51265


namespace Whitney_money_left_over_l51_51690

theorem Whitney_money_left_over :
  let posters := 2
  let notebooks := 3
  let bookmarks := 2
  let cost_poster := 5
  let cost_notebook := 4
  let cost_bookmark := 2
  let total_cost_posters := posters * cost_poster
  let total_cost_notebooks := notebooks * cost_notebook
  let total_cost_bookmarks := bookmarks * cost_bookmark
  let total_cost := total_cost_posters + total_cost_notebooks + total_cost_bookmarks
  let initial_money := 2 * 20
  let money_left_over := initial_money - total_cost
  in
  money_left_over = 14 := sorry

end Whitney_money_left_over_l51_51690


namespace circumsphere_surface_area_tetrahedron_l51_51320

noncomputable def tetrahedron_surface_area : ℝ := (3 / 2) * Real.pi

def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

def is_centroid (P : ℝ × ℝ × ℝ) (A B C : ℝ × ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3, (A.3 + B.3 + C.3) / 3)

def is_midpoint (P : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

theorem circumsphere_surface_area_tetrahedron (A B C D G M : ℝ × ℝ × ℝ)
  (hA : A = (0, 0, 0))
  (hB : B = (1, 0, 0))
  (hC : C = (0, 1, 0))
  (hD : D = (0, 0, 1))
  (hG : is_centroid G B C D)
  (hM : is_midpoint M A G) :
  distance M B = Real.sqrt (3 / 4) →
  tetrahedron_surface_area = (3 / 2) * Real.pi :=
by
  sorry

end circumsphere_surface_area_tetrahedron_l51_51320


namespace exists_x_satisfying_f_l51_51392

theorem exists_x_satisfying_f (b c : ℤ) (N : ℕ) (hN : 2 ≤ N) :
  ∃ (x : ℕ → ℤ), (f (x (N + 1)) = ∏ i in finrange N, f (x i))
  where f (x : ℤ) := x^2 + b * x + c :=
sorry

end exists_x_satisfying_f_l51_51392


namespace custom_op_diff_l51_51027

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_diff : custom_op 8 5 - custom_op 5 8 = -12 :=
by
  sorry

end custom_op_diff_l51_51027


namespace closest_perfect_square_to_350_l51_51665

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l51_51665


namespace flamingoes_needed_l51_51129

def feathers_per_flamingo : ℕ := 20
def safe_pluck_percentage : ℚ := 0.25
def boas_needed : ℕ := 12
def feathers_per_boa : ℕ := 200
def total_feathers_needed : ℕ := boas_needed * feathers_per_boa

theorem flamingoes_needed :
  480 = total_feathers_needed / (feathers_per_flamingo * safe_pluck_percentage).toNat :=
by sorry

end flamingoes_needed_l51_51129


namespace simplify_trig_expression_l51_51571

-- We are defining the trigonometric functions with specific angles first
def cos_15 : ℝ := Real.cos (15 * Real.pi / 180)
def cos_45 : ℝ := Real.cos (45 * Real.pi / 180)
def sin_165 : ℝ := Real.sin (165 * Real.pi / 180)
def sin_45 : ℝ := Real.sin (45 * Real.pi / 180)

-- Now we state the theorem we want to prove
theorem simplify_trig_expression : (cos_15 * cos_45 - sin_165 * sin_45) = 1 / 2 := 
by sorry

end simplify_trig_expression_l51_51571


namespace number_of_students_l51_51322

theorem number_of_students 
  (P S : ℝ)
  (total_cost : ℝ) 
  (percent_free : ℝ) 
  (lunch_cost : ℝ)
  (h1 : percent_free = 0.40)
  (h2 : total_cost = 210)
  (h3 : lunch_cost = 7)
  (h4 : P = 0.60 * S)
  (h5 : P * lunch_cost = total_cost) :
  S = 50 :=
by
  sorry

end number_of_students_l51_51322


namespace hexagon_area_l51_51944

theorem hexagon_area (ABCDEF : Type) [regular_hexagon ABCDEF] (G H I : Type) [midpoints G H I ABCDEF]
  (area_triangle_GHI : area_of_triangle G H I = 225) : area_of_hexagon ABCDEF = 600 := 
sorry

end hexagon_area_l51_51944


namespace find_equal_integer_pairs_l51_51801
-- Import the entire Mathlib library to ensure all necessary definitions are available

-- State the problem using Lean 4 statements
theorem find_equal_integer_pairs (m n s : ℕ) (hmn : m ≥ n) 
  (h1 : (s * m) ^ (nat.totient (s * m) / 2) = (s * n) ^ (nat.totient (s * n) / 2)) 
  (h2 : nat.totient (s * m) = nat.totient (s * n)) : 
  m = n := 
by
  -- The proof is omitted for this statement
  sorry

end find_equal_integer_pairs_l51_51801


namespace inequality_proof_l51_51830

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l51_51830


namespace whitney_money_leftover_l51_51688

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def posters : ℕ := 2
def notebooks : ℕ := 3
def bookmarks : ℕ := 2

def initial_money : ℕ := 2 * 20

def total_cost : ℕ := posters * poster_cost + notebooks * notebook_cost + bookmarks * bookmark_cost

def money_left_over : ℕ := initial_money - total_cost

theorem whitney_money_leftover : money_left_over = 14 := by
  sorry

end whitney_money_leftover_l51_51688


namespace nala_seashells_l51_51542

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end nala_seashells_l51_51542


namespace min_abs_sum_l51_51249

open Real

theorem min_abs_sum : ∃ (x : ℝ), (∀ y : ℝ, ∑ z in [| y + 3, y + 6, y + 7].toFinset, abs z ≥ -2) :=
by
  sorry

end min_abs_sum_l51_51249


namespace inequality_proof_l51_51115

variable (n : ℕ) (a : Fin n → ℝ)

theorem inequality_proof (n_pos : 3 ≤ n) (a_pos : ∀ i, 0 < a i) : 
  (∑ i, a i) * (∑ (i j : Fin n), if i < j then a i * a j / (a i + a j) else 0) ≤ 
  (n / 2) * (∑ (i j : Fin n), if i < j then a i * a j else 0) := 
sorry

end inequality_proof_l51_51115


namespace inequality_solution_l51_51348

theorem inequality_solution (x : ℝ) : (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) ∨ x = -3 :=
by
sorry

end inequality_solution_l51_51348


namespace only_set_D_forms_right_triangle_l51_51314

-- Define the side lengths for each set
def set_A : ℝ × ℝ × ℝ := (4, 5, 6)
def set_B : ℝ × ℝ × ℝ := (1, Real.sqrt 2, 2.5)
def set_C : ℝ × ℝ × ℝ := (2, 3, 4)
def set_D : ℝ × ℝ × ℝ := (1.5, 2, 2.5)

-- Define the Pythagorean theorem checker
def is_right_triangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Prove that only set_D forms a right triangle
theorem only_set_D_forms_right_triangle :
  ¬ is_right_triangle set_A.1 set_A.2 set_A.3 ∧
  ¬ is_right_triangle set_B.1 set_B.2 set_B.3 ∧
  ¬ is_right_triangle set_C.1 set_C.2 set_C.3 ∧
  is_right_triangle set_D.1 set_D.2 set_D.3 :=
by {
  sorry
}

end only_set_D_forms_right_triangle_l51_51314


namespace expand_product_l51_51797

variable (x : ℂ)
def i : ℂ := complex.I

theorem expand_product :
  (x + i) * (x - 7) = x^2 - 7 * x + i * x - 7 * i :=
by sorry

end expand_product_l51_51797


namespace quadratic_inequality_solution_l51_51817

theorem quadratic_inequality_solution (b c : ℝ) 
  (h1 : ∀ x, x^2 + b * x + c ≤ 0 → x ∈ set.Icc (-2 : ℝ) 5) :
  b * c = 30 :=
sorry

end quadratic_inequality_solution_l51_51817


namespace tsiolkovsky_velocity_l51_51570

theorem tsiolkovsky_velocity (a : Real) (v0 : Real)
  (h1 : ∀ m1 m2, v0 = (2.8 : Real) / (2 * 0.7) ∧ v0 = 2)
  (h2 : ∀ m1 m2, v = v0 * Real.log ((m1 + m2) / m1))
  (h3 : Real.log 2 ≈ 0.7)
  (h4 : Real.log 3 ≈ 1.1) :
  let m1 := a in
  let m2 := 5 * a in
  let v := v0 * Real.log ((m1 + m2) / m1) in
  v ≈ 3.6 :=
by sorry

end tsiolkovsky_velocity_l51_51570


namespace trapezoid_perimeter_l51_51480

theorem trapezoid_perimeter (EF GH : ℝ) (h : ℝ) (a b : ℝ) :
  EF = 10 ∧ GH = 12 ∧ h = 6 ∧ a = 10 ∧ b = 12 →
  let side_length := real.sqrt (h^2 + ((b - a) / 2)^2) in
  let perimeter := EF + GH + 2 * side_length in
  perimeter = 22 + 2 * real.sqrt(37) :=
by
  intros
  sorry

end trapezoid_perimeter_l51_51480


namespace range_a_for_max_of_f_l51_51884

theorem range_a_for_max_of_f (a : ℝ) (h : ∀ x ∈ set.Icc (-1 : ℝ) 3, |x^2 - 2 * x - a| + a ≤ 3) :
  a ∈ set.Iic (-1) :=
begin
  -- Proof is omitted
  sorry
end

end range_a_for_max_of_f_l51_51884


namespace find_abs_product_abc_l51_51510

theorem find_abs_product_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h : a + 1 / b = b + 1 / c ∧ b + 1 / c = c + 1 / a) : |a * b * c| = 1 :=
sorry

end find_abs_product_abc_l51_51510


namespace find_a_l51_51512

noncomputable def a : ℝ :=
  let f (x : ℝ) := a^x
  let condition := (a > 0) ∧ (a ≠ 1) ∧ (f 1 - f (-1)) / (f 2 - f (-2)) = 3 / 10
  have condition : Prop := by
    sorry
  have h1 : f (1) = a
  have h2 : f (-1) = 1 / a
  have h3 : f (2) = a^2
  have h4 : f (-2) = 1 / (a^2)
  solve_by_elim

theorem find_a (a : ℝ) (h : (a > 0) ∧ (a ≠ 1) ∧ ((a - 1 / a) / (a^2 - 1 / a^2) = 3 / 10)) : a = 3 :=
  sorry

end find_a_l51_51512


namespace first_digit_one_over_137_l51_51224

-- Define the main problem in terms of first nonzero digit.
def first_nonzero_digit_right_of_decimal (n : ℕ) : ℕ :=
  let frac := 1 / (Rat.of_int n)
  let shifted_frac := frac * 10 ^ 3
  let integer_part := shifted_frac.to_nat
  integer_part % 10

theorem first_digit_one_over_137 :
  first_nonzero_digit_right_of_decimal 137 = 7 :=
by
  sorry

end first_digit_one_over_137_l51_51224


namespace triangle_obtuse_if_asinA_bsinB_less_csinC_l51_51930

theorem triangle_obtuse_if_asinA_bsinB_less_csinC
  (A B C : ℝ) (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_angle_sum : A + B + C = π)
  (h_cond : a * Real.sin A + b * Real.sin B < c * Real.sin C) :
  (π / 2 < C) :=
begin
  sorry
end

end triangle_obtuse_if_asinA_bsinB_less_csinC_l51_51930


namespace simplify_expression_l51_51618

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l51_51618


namespace closest_perfect_square_to_350_l51_51664

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l51_51664


namespace lines_concurrent_l51_51953

open EuclideanGeometry

variables {A B C D X Y Z P M N : Point}

def on_line (points : List Point) : Prop :=
  ∃ (ℓ : Line), ∀ p ∈ points, p ∈ ℓ

-- Given four distinct points A, B, C, D on a line in that order
axiom h1 : on_line [A, B, C, D]

-- The circles with diameters [AC] and [BD] intersect at points X and Y
axiom h2 : ∃ (circle_AC circle_BD : Circle),
  circle_AC = Circle.mk A C ∧
  circle_BD = Circle.mk B D ∧
  X ∈ circle_AC ∧ X ∈ circle_BD ∧
  Y ∈ circle_AC ∧ Y ∈ circle_BD

-- The line XY intersects BC at Z
axiom h3 : ∃ (ℓₓ : Line), ℓₓ = Line.mk X Y ∧ Z ∈ ℓₓ ∧ on_line [B, Z, C]

-- P is on line XY other than Z
axiom h4 : P ∈ Line.mk X Y ∧ P ≠ Z

-- Line CP intersects the circle with diameter [AC] at M
axiom h5 : ∃ (ℓ_CP : Line), ℓ_CP = Line.mk C P ∧ M ∈ Circle.mk A C ∧ M ≠ C ∧ M ∈ ℓ_CP

-- Line BP intersects the circle with diameter [BD] at N
axiom h6 : ∃ (ℓ_BP : Line), ℓ_BP = Line.mk B P ∧ N ∈ Circle.mk B D ∧ N ≠ B ∧ N ∈ ℓ_BP

-- Prove that lines AM, DN, and XY are concurrent
theorem lines_concurrent : Concurrency [Line.mk A M, Line.mk D N, Line.mk X Y] :=
sorry

end lines_concurrent_l51_51953


namespace largest_common_divisor_414_345_l51_51626

theorem largest_common_divisor_414_345 : ∃ d, d ∣ 414 ∧ d ∣ 345 ∧ 
                                      (∀ e, e ∣ 414 ∧ e ∣ 345 → e ≤ d) ∧ d = 69 :=
by 
  sorry

end largest_common_divisor_414_345_l51_51626


namespace six_projections_coplanar_l51_51141

variables {P : Type*} [EuclideanSpace P] 
variables (A B C D : P)

def is_projection_on_angle_bisector_plane (v : P) (E1 E2 : P → Prop) : Prop :=
  ∃ x : P, E1 x ∧ E2 (v - (v - x))

def projections_on_bisector_planes (D : P) (E1 E2 E3 E4 E5 E6 : P → Prop) : set P :=
  {p | (is_projection_on_angle_bisector_plane D E1 E2 p) ∨ (is_projection_on_angle_bisector_plane D E2 E3 p) ∨
       (is_projection_on_angle_bisector_plane D E3 E4 p) ∨ (is_projection_on_angle_bisector_plane D E4 E5 p) ∨
       (is_projection_on_angle_bisector_plane D E5 E6 p) ∨ (is_projection_on_angle_bisector_plane D E6 E1 p)}

theorem six_projections_coplanar :
  ∃ plane : P → Prop, ∀ p ∈ projections_on_bisector_planes D (λ p, p ∈ line_through_points A B) 
                                           (λ p, p ∈ line_through_points B C)
                                           (λ p, p ∈ line_through_points C A)
                                           (λ p, p ∈ plane_through_points A B C D)
                                           (λ p, p ∈ plane_through_points A C D B)
                                           (λ p, p ∈ plane_through_points A B D C), 
    plane p :=
sorry

end six_projections_coplanar_l51_51141


namespace income_expenditure_ratio_l51_51177

theorem income_expenditure_ratio
  (I E : ℕ)
  (h1 : I = 18000)
  (S : ℕ)
  (h2 : S = 2000)
  (h3 : S = I - E) :
  I.gcd E = 2000 ∧ I / I.gcd E = 9 ∧ E / I.gcd E = 8 :=
by sorry

end income_expenditure_ratio_l51_51177


namespace last_digit_of_decimal_expansion_of_fraction_l51_51635

theorem last_digit_of_decimal_expansion_of_fraction : 
  let x := (1 : ℚ) / (3 ^ 15) 
  in ∃ digit : ℕ, (digit < 10) ∧ last_digit (decimal_expansion x) = 7 :=
sorry

end last_digit_of_decimal_expansion_of_fraction_l51_51635


namespace point_A_coordinates_l51_51403

-- Given conditions
def point_A (a : ℝ) : ℝ × ℝ := (a + 1, a^2 - 4)
def negative_half_x_axis (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 = 0

-- Theorem statement
theorem point_A_coordinates (a : ℝ) (h : negative_half_x_axis (point_A a)) :
  point_A a = (-1, 0) :=
sorry

end point_A_coordinates_l51_51403


namespace max_value_of_expression_l51_51861

-- Define the conditions
def on_unit_circle (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 1

def collinear (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * y2 = x2 * y1

-- State the theorem
theorem max_value_of_expression {x1 y1 x2 y2 : ℝ}
  (hA : on_unit_circle x1 y1)
  (hB : on_unit_circle x2 y2)
  (h_collinear : collinear x1 y1 x2 y2) :
  2 * x1 + x2 + 2 * y1 + y2 ≤ sqrt 2 :=
sorry

end max_value_of_expression_l51_51861


namespace evaluate_expression_l51_51622

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l51_51622


namespace monica_total_students_l51_51533

theorem monica_total_students 
  (c1 : ∀ (i: ℕ), i = 1 → i.students = 20)
  (c2 : ∀ (i: ℕ), (i = 2 ∨ i = 3) → i.students = 25)
  (c3 : ∀ (i: ℕ), i = 4 → i.students = c1 1 / 2)
  (c4 : ∀ (i: ℕ), (i = 5 ∨ i = 6) → i.students = 28)
  : (Σ i, i.students) = 136 := 
by
  sorry

end monica_total_students_l51_51533


namespace squirrels_cannot_divide_equally_l51_51359

theorem squirrels_cannot_divide_equally
    (n : ℕ) : ¬ (∃ k, 2022 + n * (n + 1) = 5 * k) :=
by
sorry

end squirrels_cannot_divide_equally_l51_51359


namespace statement_a_statement_b_statement_c_statement_d_l51_51687

-- Statement A: ∀ a, b, c in ℝ, c ≠ 0 → (a/c^2 > b/c^2 → a > b)
theorem statement_a (a b c : ℝ) (hc : c ≠ 0) : a / c^2 > b / c^2 → a > b :=
sorry

-- Statement B: ∀ x in (0, π), sin(x) + 4/sin(x) ≥ 4
theorem statement_b (x : ℝ) (hx : 0 < x ∧ x < π) : sin x + 4 / sin x ≥ 4 :=
sorry

-- Statement C: ¬(∃ x in ℝ, x^2 + 2x + 3 < 0) → (∀ x in ℝ, x^2 + 2x + 3 ≥ 0)
theorem statement_c : ¬(∃ x : ℝ, x^2 + 2 * x + 3 < 0) → (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0) :=
sorry

-- Statement D: If choosing 3 different numbers from {1, 2, 3, 4, 5},
-- the probability that they form a right-angled triangle is 1/10
theorem statement_d : let S : Finset ℕ := {1, 2, 3, 4, 5} in
  let combos := Finset.powersetLen 3 S in
  let right_angle_triangles := combos.filter (λ s, ∃ a b c,
    s = {a, b, c} ∧ 
    (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)) in
  (Finset.card right_angle_triangles : ℝ) / (Finset.card combos : ℝ) = 1 / 10 :=
sorry

end statement_a_statement_b_statement_c_statement_d_l51_51687


namespace minimum_reciprocal_presses_l51_51530

def reciprocal (x : ℕ) : ℚ := 1 / x

theorem minimum_reciprocal_presses : (reciprocal (reciprocal 32) = 32) :=
by {
  sorry
}

end minimum_reciprocal_presses_l51_51530


namespace area_of_triangle_formed_by_line_l51_51751

-- Definitions based on problem conditions
def point : Type := ℝ × ℝ

def slope : ℝ := -1/2

def line_through (m : ℝ) (p : point) : Prop :=
  ∃ b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ (x, y) = p

def x_intercept (m b : ℝ) : ℝ := -b / m

def y_intercept (b : ℝ) : ℝ := b

def triangle_area (base height : ℝ) : ℝ := 1/2 * base * height

-- Proof statement in conditions of Lean 4 statement
theorem area_of_triangle_formed_by_line :
  ∀ (l : ℝ → ℝ), 
  (∃ (p : point), p = (2, -3) ∧ l = λ x, slope * x + (p.2 - slope * p.1)) →
  triangle_area (|x_intercept slope (2 * +3)|) (|y_intercept (2 - slope * 2)|) = 4 := 
by
  intros l h
  sorry

end area_of_triangle_formed_by_line_l51_51751


namespace compare_values_l51_51848

def a := Real.log 2
def b := 2023 / 2022
def c := Real.log 2023 / Real.log 2022

theorem compare_values : a < c ∧ c < b :=
by
  have ha : a = Real.log 2 := rfl
  have hb : b = 2023 / 2022 := rfl
  have hc : c = Real.log 2023 / Real.log 2022 := rfl
  sorry

end compare_values_l51_51848


namespace first_digit_one_over_137_l51_51222

-- Define the main problem in terms of first nonzero digit.
def first_nonzero_digit_right_of_decimal (n : ℕ) : ℕ :=
  let frac := 1 / (Rat.of_int n)
  let shifted_frac := frac * 10 ^ 3
  let integer_part := shifted_frac.to_nat
  integer_part % 10

theorem first_digit_one_over_137 :
  first_nonzero_digit_right_of_decimal 137 = 7 :=
by
  sorry

end first_digit_one_over_137_l51_51222


namespace min_abs_sum_l51_51250

open Real

theorem min_abs_sum : ∃ (x : ℝ), (∀ y : ℝ, ∑ z in [| y + 3, y + 6, y + 7].toFinset, abs z ≥ -2) :=
by
  sorry

end min_abs_sum_l51_51250


namespace problem_statement_l51_51075

/-- The sum of all positive integers from 1 to 200. -/
def jo_sum : ℕ := ∑ i in Finset.range 201, i

/-- The sum of all positive integers from 1 to 200 rounded to the nearest multiple of 20. -/
def leah_sum : ℕ := 
    ∑ i in Finset.range 201, 
        if i % 20 == 0 then i 
        else if i % 20 < 10 then (i - (i % 20)) 
        else (i + (20 - i % 20))

/-- The positive difference between Jo's sum and Leah's sum. -/
theorem problem_statement : |jo_sum - leah_sum| = 18100 :=
by
  sorry

end problem_statement_l51_51075


namespace angle_ADF_eq_60_l51_51482

theorem angle_ADF_eq_60
  (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (h_equilateral : (Metric.dist A B = Metric.dist B C) ∧ (Metric.dist B C = Metric.dist C A))
  (h_midpoint_D : Metric.dist B D = Metric.dist D C)
  (h_equal_BEC : Metric.dist B E = Metric.dist E C)
  (h_equal_AFB : Metric.dist A F = Metric.dist F B) :
  ∠ A D F = 60 :=
by
  sorry

end angle_ADF_eq_60_l51_51482


namespace kolya_cannot_see_full_result_l51_51093

theorem kolya_cannot_see_full_result : let n := 1024 in 
                                      let num_decimals := 16 in 
                                      (num_divisors_product n > 10^num_decimals) := 
begin
    let n := 1024,
    let num_decimals := 16,
    have product := (∑ i in range (nat.log2 n), i),
    have sum_of_powers := (2^product),
    have required_digits := (10^num_decimals),
    exact sum_of_powers > required_digits,
end

end kolya_cannot_see_full_result_l51_51093


namespace system_solution_unique_l51_51116

noncomputable def solution_system (n : ℕ) (a : Fin (2 * n) → ℝ) : Prop :=
(n ≥ 4) ∧
(∀ i : Fin (2 * n),
    if i.val.even then a i.succ = a i.pred + a (i.succ.succ)
    else a i.succ = (1 / a (i - 1)) + (1 / a (i + 1)))

theorem system_solution_unique (n : ℕ) (a : Fin (2 * n) → ℝ) :
  (n ≥ 4) ∧ (∀ i, 1 ≤ a i) ∧ solution_system n a → 
  (a = λ i, if i.val.even then 2 else 1) :=
sorry

end system_solution_unique_l51_51116


namespace evaluate_F_2_f_3_l51_51904

def f (a : ℕ) : ℕ := a^2 - 2*a
def F (a b : ℕ) : ℕ := b^2 + a*b

theorem evaluate_F_2_f_3 : F 2 (f 3) = 15 := by
  sorry

end evaluate_F_2_f_3_l51_51904


namespace pencils_total_correct_l51_51196

constant num_pencils_initial : ℕ
constant num_pencils_added : ℕ
constant total_pencils : ℕ

axiom initial_pencils_def : num_pencils_initial = 27
axiom added_pencils_def : num_pencils_added = 45
axiom total_pencils_def : total_pencils = 72

theorem pencils_total_correct : num_pencils_initial + num_pencils_added = total_pencils :=
by 
  rw [initial_pencils_def, added_pencils_def, total_pencils_def]
  sorry

end pencils_total_correct_l51_51196


namespace add_decimals_l51_51753

theorem add_decimals :
  5.467 + 3.92 = 9.387 :=
by
  sorry

end add_decimals_l51_51753


namespace find_m_for_increasing_function_l51_51001

def is_increasing_on_interval (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ x y, x ∈ I ∧ y ∈ I ∧ x ≤ y → f x ≤ f y

theorem find_m_for_increasing_function :
  {m : ℝ | 0 ≤ m ∧ m ≤ (1 : ℝ) / 4} =
  {m : ℝ | is_increasing_on_interval (λ x, m * x^2 + x + 5) (Set.Ici (-2))} :=
by
  sorry

end find_m_for_increasing_function_l51_51001


namespace marta_books_l51_51531

theorem marta_books (M : ℕ) (hM : M = 38) : 
  let books_now := (M + 10 - 5) + (3 * M) in 
  books_now = 157 := 
by
  unfold books_now
  rw [hM]
  norm_num
  sorry

end marta_books_l51_51531


namespace estimated_red_balls_l51_51471

theorem estimated_red_balls (total_balls : ℕ) (frequency_red : ℝ)
  (h1 : total_balls = 200) (h2 : frequency_red = 0.30) :
  total_balls * frequency_red = 60 := 
  by
    sorry

end estimated_red_balls_l51_51471


namespace tan_neq_sqrt3_sufficient_but_not_necessary_l51_51163

-- Definition of the condition: tan(α) ≠ √3
def condition_tan_neq_sqrt3 (α : ℝ) : Prop := Real.tan α ≠ Real.sqrt 3

-- Definition of the statement: α ≠ π/3
def statement_alpha_neq_pi_div_3 (α : ℝ) : Prop := α ≠ Real.pi / 3

-- The theorem to be proven
theorem tan_neq_sqrt3_sufficient_but_not_necessary {α : ℝ} :
  condition_tan_neq_sqrt3 α → statement_alpha_neq_pi_div_3 α :=
sorry

end tan_neq_sqrt3_sufficient_but_not_necessary_l51_51163


namespace derivative_at_neg_pi_over_two_l51_51878

def f (x : ℝ) : ℝ := x * Real.sin x - Real.cos x

theorem derivative_at_neg_pi_over_two :
  (deriv f) (-π / 2) = -2 :=
sorry

end derivative_at_neg_pi_over_two_l51_51878


namespace find_expression_for_an_l51_51724

-- Definitions for the problem conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def problem_conditions (a : ℕ → ℝ) (q : ℝ) :=
  geometric_sequence a q ∧
  a 1 + a 3 = 10 ∧
  a 2 + a 4 = 5

-- Statement of the problem
theorem find_expression_for_an (a : ℕ → ℝ) (q : ℝ) :
  problem_conditions a q → ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end find_expression_for_an_l51_51724


namespace transportation_tax_correct_l51_51764

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end transportation_tax_correct_l51_51764


namespace find_point_P_l51_51111

theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, P = (5, -3, 4) ∧
    let A : ℝ × ℝ × ℝ := (10, 0, 0) in
    let B : ℝ × ℝ × ℝ := (0, -6, 0) in
    let C : ℝ × ℝ × ℝ := (0, 0, 8) in
    let D : ℝ × ℝ × ℝ := (0, 0, 0) in
    let AP := (P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2 + (P.3 - A.3) ^ 2 in
    let BP := (P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2 + (P.3 - B.3) ^ 2 in
    let CP := (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 + (P.3 - C.3) ^ 2 in
    let DP := (P.1 - D.1) ^ 2 + (P.2 - D.2) ^ 2 + (P.3 - D.3) ^ 2 in
    AP = DP ∧ BP = DP ∧ CP = DP :=
by
  sorry

end find_point_P_l51_51111


namespace factor_quadratic_expression_l51_51798

theorem factor_quadratic_expression (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) :=
by
  sorry

end factor_quadratic_expression_l51_51798


namespace hexadecagon_area_l51_51739

theorem hexadecagon_area (r : ℝ) : 
  (∃ A : ℝ, A = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)) :=
sorry

end hexadecagon_area_l51_51739


namespace scientific_notation_141178_million_l51_51575

theorem scientific_notation_141178_million :
  ∃ a n, (1 ≤ a) ∧ (a < 10) ∧ (141178 * 10^6 = a * 10^n) ∧ a = 1.41178 ∧ n = 9 := 
by
  use 1.41178
  use 9
  split
  exact le_of_lt zero_lt_one
  split
  exact show (1.41178 < 10), by norm_num
  split
  norm_num
  rfl
  rfl

end scientific_notation_141178_million_l51_51575


namespace height_of_building_l51_51693

-- Define the conditions
def height_flagpole : ℝ := 18
def shadow_flagpole : ℝ := 45
def shadow_building : ℝ := 55

-- State the theorem to prove the height of the building
theorem height_of_building (h : ℝ) : (height_flagpole / shadow_flagpole) = (h / shadow_building) → h = 22 :=
by
  sorry

end height_of_building_l51_51693


namespace domain_of_f_f_gt_f_l51_51525

noncomputable def f (x : ℝ) (k : ℝ) := 
  1 / real.sqrt ((x^2 + x + k)^2 + 2 * (x^2 + x + k) - 3)

theorem domain_of_f (k : ℝ) (h : k < -6) : 
  set_of (λ x, (x^2 + x + k)^2 + 2 * (x^2 + x + k) - 3 > 0) = 
    {x : ℝ | x ∈ Ioo (-∞) (-1 - real.sqrt (2 - k)) ∪ Ioo (-1 - real.sqrt (2 - k)) (√(2 - k)) ∪ Ioo (-1 - real.sqrt (2 - k)) (1 + real.sqrt (-2 * k - 4))} := 
sorry

theorem f_gt_f (k : ℝ) (h : k < -6) : 
  set_of (λ x, f x k > f 1 k) =
    {x : ℝ | x ∈ Ioo (-1 - real.sqrt (-4 - k)) (-1 - real.sqrt (2 - k)) ∪ Ioo (1 - real.sqrt (-2 - k)) (-∞) ∪ Ioo (-1) (-1 + real.sqrt (-2 * k)) ∪ Ioo (-1 + real.sqrt (2 - k)) (1 + real.sqrt (-2 * k - 4))} := 
sorry

end domain_of_f_f_gt_f_l51_51525


namespace units_digit_of_square_tens_l51_51454

theorem units_digit_of_square_tens {a : ℕ} :
  (∃ t ∈ {1, 3, 5, 7, 9}, (a^2 % 100) / 10 = t) → (∃ u ∈ {4, 6}, a % 10 = u) :=
by sorry

end units_digit_of_square_tens_l51_51454


namespace gcd_condition_l51_51947

theorem gcd_condition (a b c : ℕ) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  Nat.gcd b c = 15 :=
sorry

end gcd_condition_l51_51947


namespace max_color_count_in_rectangular_subgrid_l51_51776

theorem max_color_count_in_rectangular_subgrid :
  ∀ (k : ℕ), (∀ grid : ℕ × ℕ → ℕ, (∀ i j : ℕ, grid i j < k) → 
  (∀ i j : ℕ, (∀ x y : ℕ, x < 3 ∧ y < 4 → grid (i + x) (j + y) ∈ (set.range id).image grid) → 
  (∃ sub_grid : ℕ × ℕ → ℕ, (∀ x y : ℕ, x < 3 ∧ y < 4 → sub_grid x y < 10))) → k ≤ 10 :=
by
  sorry

end max_color_count_in_rectangular_subgrid_l51_51776


namespace train_speed_l51_51267

theorem train_speed (length : ℕ) (time : ℕ) (h₁ : length = 300) (h₂ : time = 30) : length / time = 10 :=
by 
  rw [h₁, h₂] 
  norm_num
  sorry

end train_speed_l51_51267


namespace pyramid_values_l51_51479

theorem pyramid_values :
  ∃ (A B C D : ℕ),
    (A = 3000) ∧
    (D = 623) ∧
    (B = 700) ∧
    (C = 253) ∧
    (A = 1100 + 1800) ∧
    (D + 451 ≥ 1065) ∧ (D + 451 ≤ 1075) ∧ -- rounding to nearest ten
    (B + 440 ≥ 1050) ∧ (B + 440 ≤ 1150) ∧
    (B + 1070 ≥ 1700) ∧ (B + 1070 ≤ 1900) ∧
    (C + 188 ≥ 430) ∧ (C + 188 ≤ 450) ∧    -- rounding to nearest ten
    (C + 451 ≥ 695) ∧ (C + 451 ≤ 705) :=  -- using B = 700 for rounding range
sorry

end pyramid_values_l51_51479


namespace parallelogram_internal_angle_E_l51_51563

noncomputable def angle_F_external : ℝ := 50
noncomputable def angle_GF_internal : ℝ := 180 - angle_F_external
noncomputable def angle_E : ℝ := angle_GF_internal

theorem parallelogram_internal_angle_E (a b c d : ℝ) (EFGH_parallelogram : Parallelogram a b c d) :
  (angle_F_external = 50) → (angle_E = 130) := by
sorry

end parallelogram_internal_angle_E_l51_51563


namespace num_dimes_is_correct_l51_51017

-- Definitions for the conditions
def num_pennies : ℕ := 9
def num_nickels : ℕ := 4
def total_money : ℝ := 0.59

-- Constants for the values of coins
def value_of_penny : ℝ := 0.01
def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10

-- Calculate the money from pennies and nickels
def money_from_pennies : ℝ := num_pennies * value_of_penny
def money_from_nickels : ℝ := num_nickels * value_of_nickel
def money_from_pennies_and_nickels : ℝ := money_from_pennies + money_from_nickels

-- Calculate the required number of dimes
def num_dimes : ℕ := 
  let money_from_dimes := total_money - money_from_pennies_and_nickels
  money_from_dimes / value_of_dime |> Rational.floor

-- Prove that num_dimes equals 3
theorem num_dimes_is_correct : num_dimes = 3 :=
by 
  sorry -- Proof to be filled in

end num_dimes_is_correct_l51_51017


namespace math_club_total_members_l51_51056

theorem math_club_total_members (female_count : ℕ) (h_female : female_count = 6) (h_male_ratio : ∃ male_count : ℕ, male_count = 2 * female_count) :
  ∃ total_members : ℕ, total_members = female_count + classical.some h_male_ratio :=
by
  let male_count := classical.some h_male_ratio
  have h_male_count : male_count = 12 := by sorry
  existsi (female_count + male_count)
  rw [h_female, h_male_count]
  exact rfl

end math_club_total_members_l51_51056


namespace triangle_abc_l51_51039

/-!
# Problem Statement
In triangle ABC with side lengths a, b, and c opposite to vertices A, B, and C respectively, we are given that ∠A = 2 * ∠B. We need to prove that a² = b * (b + c).
-/

variables (A B C : Type) -- Define vertices of the triangle
variables (α β γ : ℝ) -- Define angles at vertices A, B, and C respectively.

-- Define sides of the triangle
variables (a b c x y : ℝ) -- Define sides opposite to the corresponding angles

-- Main statement to prove in Lean 4
theorem triangle_abc (h1 : α = 2 * β) (h2 : a = b * (2 * β)) :
  a^2 = b * (b + c) :=
sorry

end triangle_abc_l51_51039


namespace length_of_base_of_isosceles_triangle_l51_51579

noncomputable def length_congruent_sides : ℝ := 8
noncomputable def perimeter_triangle : ℝ := 26

theorem length_of_base_of_isosceles_triangle : 
  ∀ (b : ℝ), 
  2 * length_congruent_sides + b = perimeter_triangle → 
  b = 10 :=
by
  intros b h
  -- The proof is omitted.
  sorry

end length_of_base_of_isosceles_triangle_l51_51579


namespace f_has_at_least_five_prime_divisors_l51_51500

noncomputable def polynomial_f (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, c ≠ 0 ∧
  ∀ x : ℤ,
    f(x) = c * (x - 1) * x * (x + 1) * (x + 2) * (x^2 - x + 1)

theorem f_has_at_least_five_prime_divisors (f : ℤ → ℤ)
  (h1 : polynomial_f f)
  (h2 : ∀ n : ℤ, n ≥ 8 → ∃ p1 p2 p3 p4 p5 : ℤ, prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ prime p5 ∧
    ¬(p1 = p2) ∧ ¬(p2 = p3) ∧ ¬(p3 = p4) ∧ ¬(p4 = p5) ∧ ¬(p1 = p3) ∧ ¬(p1 = p4) ∧ ¬(p1 = p5) ∧ ¬(p2 = p4) ∧ ¬(p2 = p5) ∧ ¬(p3 = p5) ∧
    p1 ∣ f n ∧ p2 ∣ f n ∧ p3 ∣ f n ∧ p4 ∣ f n ∧ p5 ∣ f n) : Prop :=
sorry

end f_has_at_least_five_prime_divisors_l51_51500


namespace problem_l51_51509

-- Definitions for conditions
def countMultiplesOf (n upperLimit : ℕ) : ℕ :=
  (upperLimit - 1) / n

def a : ℕ := countMultiplesOf 4 40
def b : ℕ := countMultiplesOf 4 40

-- Statement to prove
theorem problem : (a + b)^2 = 324 := by
  sorry

end problem_l51_51509


namespace roots_of_cubic_l51_51951

/-- Let p, q, and r be the roots of the polynomial x^3 - 15x^2 + 10x + 24 = 0. 
   The value of (1 + p)(1 + q)(1 + r) is equal to 2. -/
theorem roots_of_cubic (p q r : ℝ)
  (h1 : p + q + r = 15)
  (h2 : p * q + q * r + r * p = 10)
  (h3 : p * q * r = -24) :
  (1 + p) * (1 + q) * (1 + r) = 2 := 
by 
  sorry

end roots_of_cubic_l51_51951


namespace find_m_l51_51335

-- Define the sequence u
def sequence_u : ℕ → ℝ
| 1       := 1
| m@(n+1) := if h : m % 3 = 0 then 2 + sequence_u (m / 3) else 2 / sequence_u n

-- Define the proof goal
theorem find_m (m : ℕ) : sequence_u m = 31 / 127 → m = 40 := 
by
  sorry

end find_m_l51_51335


namespace probability_one_of_A_or_C_on_first_day_l51_51456

theorem probability_one_of_A_or_C_on_first_day
  {A B C : Person} (on_duty : Person → Fin 3 → Prop) :
  (∀ p : Person, ∃ d : Fin 3, on_duty p d) →
  (∀ d : Fin 3, ∃! p : Person, on_duty p d) →
  (∃ d : Fin 3, on_duty A 0 ∨ on_duty C 0) →
  (∃ d : Fin 3, ¬ (on_duty A 0 ∧ on_duty C 0)) →
  (probability (one_of_A_C_on_first_day on_duty A C 0) = 2/3) :=
begin
  sorry
end

end probability_one_of_A_or_C_on_first_day_l51_51456


namespace problem1_problem2_l51_51706

-- Problem 1
theorem problem1 (x y : ℝ) : (x + 3 * y) * (x - 3 * y) - x^2) / (9 * y) = -y :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) : (m - 4) * (m + 1) + 3 * m = (m + 2) * (m - 2) :=
by sorry

end problem1_problem2_l51_51706


namespace kolya_product_divisors_1024_l51_51088

theorem kolya_product_divisors_1024 :
  let number := 1024
  let screen_limit := 10^16
  ∃ (divisors : List ℕ), 
    (∀ d ∈ divisors, d ∣ number) ∧ 
    (∏ d in divisors, d) = 2^55 ∧ 
    2^55 > screen_limit :=
by
  sorry

end kolya_product_divisors_1024_l51_51088


namespace value_of_b_l51_51175

theorem value_of_b 
  (midpoint_is_on_line : ∀ a b, (a, b) = ((1 + 7) / 2, (4 + 10) / 2) → a + b = 11):
  ∃ b, b = 11 :=
by 
  use 11
  exact midpoint_is_on_line _ _ rfl

end value_of_b_l51_51175


namespace olivia_total_payment_l51_51553

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l51_51553


namespace total_seashells_l51_51540

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end total_seashells_l51_51540


namespace swap_checkers_possible_l51_51784

def black_white_swap_possible (n : ℕ) : Prop :=
  ∃ (move_sequence : list (ℕ × bool)), -- list of moves and color black (true) or white (false)
    let initial_pos := (list.replicate n true) ++ [false] ++ (list.replicate n false) in
    let final_pos := (list.replicate n false) ++ [false] ++ (list.replicate n true) in
    let move := λ state move_sequence, -- logic for applying move_sequence to the state
      sorry in
    move initial_pos move_sequence = final_pos

theorem swap_checkers_possible (n : ℕ) : black_white_swap_possible n :=
sorry

end swap_checkers_possible_l51_51784


namespace possible_point_counts_l51_51601

theorem possible_point_counts (r b g : ℕ) (d_RB d_RG d_BG : ℕ) :
    r + b + g = 15 →
    r * b * d_RB = 51 →
    r * g * d_RG = 39 →
    b * g * d_BG = 1 →
    (r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3) :=
by {
    sorry
}

end possible_point_counts_l51_51601


namespace fraction_of_fraction_of_fraction_l51_51624

theorem fraction_of_fraction_of_fraction (a b c d : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) (h₃ : c = 1/6) (h₄ : d = 90) :
  (a * b * c * d) = 1 :=
by
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry -- To indicate that the proof is missing

end fraction_of_fraction_of_fraction_l51_51624


namespace first_nonzero_digit_fraction_l51_51228

theorem first_nonzero_digit_fraction :
  (∃ n: ℕ, 0 < n ∧ n < 10 ∧ (n / 137 % 1) * 10 < 10 ∧ ((n / 137 % 1) * 10).floor = 2) :=
sorry

end first_nonzero_digit_fraction_l51_51228


namespace problem_0_lt_a10_minus_sqrt2_lt_10_pow_minus_370_l51_51108

noncomputable def a : ℕ → ℝ 
| 0       => 1
| (n+1)   => (a n) / 2 + 1 / (a n)

theorem problem_0_lt_a10_minus_sqrt2_lt_10_pow_minus_370 :
  0 < a 10 - Real.sqrt 2 ∧ a 10 - Real.sqrt 2 < 10 ^ -370 := 
sorry

end problem_0_lt_a10_minus_sqrt2_lt_10_pow_minus_370_l51_51108


namespace zoey_finishes_on_monday_l51_51264

def total_reading_days (books : ℕ) : ℕ :=
  (books * (books + 1)) / 2 + books

def day_of_week (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem zoey_finishes_on_monday : 
  day_of_week 2 (total_reading_days 20) = 1 :=
by
  -- Definitions
  let books := 20
  let start_day := 2 -- Corresponding to Tuesday
  let days := total_reading_days books
  
  -- Prove day_of_week 2 (total_reading_days 20) = 1
  sorry

end zoey_finishes_on_monday_l51_51264


namespace sufficient_but_not_necessary_condition_l51_51481

theorem sufficient_but_not_necessary_condition 
  (A B C : Type) [InnerProductSpace ℝ A] 
  (AB AC : A) 
  (h : ⟪AB, AC⟫ = 0) : 
  (right_triangle ABC) ∧ (¬necessary_condition ABC h) :=
begin
  sorry,
end

end sufficient_but_not_necessary_condition_l51_51481


namespace arith_seq_sum_correct_l51_51598

-- Define the arithmetic sequence given the first term and common difference
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arith_seq_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given Problem Conditions
def a₁ := -5
def d := 3
def n := 20

-- Theorem: Sum of the first 20 terms of the arithmetic sequence is 470
theorem arith_seq_sum_correct : arith_seq_sum a₁ d n = 470 :=
  sorry

end arith_seq_sum_correct_l51_51598


namespace range_of_a_l51_51446

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 :=
by 
  intro h
  sorry

end range_of_a_l51_51446


namespace square_side_length_l51_51567

-- Definitions based on problem conditions
def right_triangle : Prop := ∃ (A B C : Point) (AB AC BC : ℝ), AB = 9 ∧ AC = 12 ∧ right_angle A B C ∧ hypotenuse B C

def square_on_hypotenuse : Prop :=
  right_triangle ∧ ∃ (S : Point → Point → Point → Point → Prop)
  (s : ℝ), ∃ h : S(D E F G) (s_on_hypotenuse h s), vertex_on_legs S D E

theorem square_side_length (s : ℝ) : square_on_hypotenuse → s = 180 / 37 :=
by
  intro h
  sorry

end square_side_length_l51_51567


namespace first_nonzero_digit_one_over_137_l51_51236

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end first_nonzero_digit_one_over_137_l51_51236


namespace rook_return_modulo_l51_51957

theorem rook_return_modulo (
  n : ℕ,
  h_n_pos : n > 1,
  moves : list (ℤ × ℤ),
  non_repeating_moves : ∀ i j, i ≠ j → moves.nth i ≠ moves.nth j,
  start_pos : ℤ × ℤ,
  rook_path : list (ℤ × ℤ),
  path_property : ∀ (i : ℕ) (x : ℤ) (y : ℤ),
    moves.nth i = some (x, y)
    → rook_path.nth i = some (start_pos.1 + x * n, start_pos.2 + y * n),
  returns_to_start : list.head rook_path = list.last rook_path (start_pos),
  closed_loop: ∀ (i : ℕ), i < rook_path.length → rook_path.nth i ≠ none
) :
  ∃ k : ℤ, k ≡ 1 [ZMOD (n : ℤ)] :=
sorry

end rook_return_modulo_l51_51957


namespace divisors_product_exceeds_16_digits_l51_51094

theorem divisors_product_exceeds_16_digits :
  let n := 1024
  let screen_digits := 16
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  (String.length (Natural.toString divisors_product) > screen_digits) := 
by {
  let n := 1024
  let screen_digits := 16
  let divisors := List.range (11)
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  show (String.length (Natural.toString divisors_product) > screen_digits),
  sorry
}

end divisors_product_exceeds_16_digits_l51_51094


namespace max_non_overlapping_regions_l51_51158

open Lean
open Mathlib

theorem max_non_overlapping_regions : ∀ (l1 l2 : Line) (A B : Fin 10 → Point) (p1 : ∀ i j, A i ≠ A j) (p2 : ∀ i j, B i ≠ B j) (p3 : parallel l1 l2) (p4 : ∀i, A i ∈ l1) (p5 : ∀i, B i ∈ l2),
  ∃ segs : Fin 10 → Segment, ∀ k < 10, ∃ m ≤ k, non_overlapping_regions (segs 0) (segs 1) ... (segs k) = 56 :=
sorry

end max_non_overlapping_regions_l51_51158


namespace evaluate_expression_l51_51623

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l51_51623


namespace exists_z0_abs_z0_le_1_and_abs_f_z0_ge_abs_c0_add_abs_cn_l51_51387

theorem exists_z0_abs_z0_le_1_and_abs_f_z0_ge_abs_c0_add_abs_cn
  (n : ℕ)
  (f : ℂ → ℂ)
  (c : Fin (n + 1) → ℂ)
  (h_f : ∀ z, f z = ∑ i in Finset.range (n + 1), c i * z ^ (n - i))
  : ∃ z0 : ℂ, |z0| ≤ 1 ∧ |f z0| ≥ |c 0| + |c n := 
sorry

end exists_z0_abs_z0_le_1_and_abs_f_z0_ge_abs_c0_add_abs_cn_l51_51387


namespace range_of_k_l51_51915

noncomputable section

open Classical

variables {A B C k : ℝ}

def is_acute_triangle (A B C : ℝ) := A < 90 ∧ B < 90 ∧ C < 90

theorem range_of_k (hA : A = 60) (hBC : BC = 6) (h_acute : is_acute_triangle A B C) : 
  2 * Real.sqrt 3 < k ∧ k < 4 * Real.sqrt 3 :=
sorry

end range_of_k_l51_51915


namespace parity_periodicity_of_shifted_function_l51_51879

def f (x : ℝ) : ℝ := (1 / 2) * Real.sin (2 * x)

theorem parity_periodicity_of_shifted_function :
  (∀ x : ℝ, f (x + (3 * Real.pi / 4)) = - (1 / 2) * Real.cos (2 * x)) ∧
  (∀ x : ℝ, - (1 / 2) * Real.cos (2 * x) = -(1 / 2) * Real.cos (-2 * x)) ∧
  (∃ T > 0, ∀ x : ℝ, (1 / 2) * Real.cos (2 * (x + T)) = (1 / 2) * Real.cos (2 * x) ∧ T = Real.pi) :=
by sorry

end parity_periodicity_of_shifted_function_l51_51879


namespace proof_problem_l51_51516

def MA : ℝ := 1
def a : ℝ := MA
def MC : ℝ := 2
def b : ℝ := MC
def angleAMC : ℝ := 120

-- Law of Cosines application for AC, given angle AMC is 120 degrees
noncomputable def AC := Real.sqrt (a^2 + b^2 + a * b)

-- Similarity of triangles implies these equal ratios hold
def triangleANMSimilar : (bc am : ℝ) → Prop :=  λ BC AM, BC / a = NC / MN
def triangleCNMSimilar : (ab mc : ℝ) → Prop :=  λ AB MC, AB / b = AN / MN

-- Combine the proportions derived from similarity
theorem proof_problem :
  AC = Real.sqrt 7 ∧ MN = 2 / 3 := 
by
  sorry

end proof_problem_l51_51516


namespace solve_for_x_l51_51572

theorem solve_for_x
  (n m x : ℕ)
  (h1 : 7 / 8 = n / 96)
  (h2 : 7 / 8 = (m + n) / 112)
  (h3 : 7 / 8 = (x - m) / 144) :
  x = 140 :=
by
  sorry

end solve_for_x_l51_51572


namespace value_of_a_squared_plus_b_squared_l51_51508

theorem value_of_a_squared_plus_b_squared (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) :
  a^2 + b^2 = 68 :=
sorry

end value_of_a_squared_plus_b_squared_l51_51508


namespace problem_l51_51118

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end problem_l51_51118


namespace find_number_of_pencils_l51_51920

noncomputable def number_of_pencils (K B R P : ℕ) : Prop :=
  (K = B + 10) ∧ (B = 2 * P) ∧ (R = P - 2) ∧ (K + B + R = 48) ∧ (P = 8)

theorem find_number_of_pencils :
  ∃ P, ∀ K B R : ℕ, number_of_pencils K B R P :=
by
  existsi 8
  intros K B R
  unfold number_of_pencils
  split;
  sorry

end find_number_of_pencils_l51_51920


namespace buying_pens_l51_51922

theorem buying_pens (total_pens : ℕ) (defective_pens : ℕ) (prob_non_defective : ℚ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 3) (h3 : prob_non_defective = 6 / 11) : 
  ∃ n : ℕ, n = 2 ∧ (∏ i in finset.range n, (total_pens - defective_pens - i) / (total_pens - i)) = prob_non_defective :=
by
  sorry

end buying_pens_l51_51922


namespace square_side_length_l51_51566

-- Definitions based on problem conditions
def right_triangle : Prop := ∃ (A B C : Point) (AB AC BC : ℝ), AB = 9 ∧ AC = 12 ∧ right_angle A B C ∧ hypotenuse B C

def square_on_hypotenuse : Prop :=
  right_triangle ∧ ∃ (S : Point → Point → Point → Point → Prop)
  (s : ℝ), ∃ h : S(D E F G) (s_on_hypotenuse h s), vertex_on_legs S D E

theorem square_side_length (s : ℝ) : square_on_hypotenuse → s = 180 / 37 :=
by
  intro h
  sorry

end square_side_length_l51_51566


namespace sum_solution_l51_51028

theorem sum_solution (x1 x2 : ℝ) (h1 : x1 + 2^x1 = 5) (h2 : x2 + log (2 : ℝ) x2 = 5) : x1 + x2 = 5 :=
sorry

end sum_solution_l51_51028


namespace no_int_sol_x4_p_eq_3y4_l51_51104

theorem no_int_sol_x4_p_eq_3y4 (p : ℕ) (hp : p > 5) (prime_p : Prime p) 
    (digit_cond : ∀ d ∈ (nat.digits 10 p), d % 3 ≠ 0 ∧ d % 7 ≠ 0) : 
    ¬(∃ x y : ℤ, x^4 + p = 3 * y^4) := by
  sorry

end no_int_sol_x4_p_eq_3y4_l51_51104


namespace part1_part2_l51_51000

-- Definitions of functions
def f (x a : ℝ) := x * |x - a|
def g (x : ℝ) := x^2 - 1

-- Problem translation to Lean statement

theorem part1 (a : ℝ) (x : ℝ) (h : a = 1) : 
  f x a ≥ g x ↔ x ∈ set.Icc (-1/2 : ℝ) 1 :=
by
  rw h
  sorry

noncomputable def F (a : ℝ) : ℝ :=
if a ≤ 4 * Real.sqrt 2 - 4 then 4 - 2 * a
else if 4 * Real.sqrt 2 - 4 < a ∧ a < 4 then a^2 / 4
else 2 * a - 4

theorem part2 (a : ℝ) : 
  ∀ x ∈ set.Icc (0 : ℝ) 2, f x a ≤ F a :=
by
  sorry

end part1_part2_l51_51000


namespace num_baskets_l51_51202

axiom num_apples_each_basket : ℕ
axiom total_apples : ℕ

theorem num_baskets (h1 : num_apples_each_basket = 17) (h2 : total_apples = 629) : total_apples / num_apples_each_basket = 37 :=
  sorry

end num_baskets_l51_51202


namespace ball_selection_properties_l51_51369

theorem ball_selection_properties :
  let red_balls := 2
  let black_balls := 2
  let selected_balls := 2
  let events := { R:Type := Bool // Type }
  let event_exactly_one_black (events) :=
    ∃ (b1: events.1) (b2: events.1), b1 ≠ b2 ∧ b1 = black ∧ b2 = red

  let event_both_black (events) :=
    ∃ (b1: events.1) (b2: events.1), b1 = black ∧ b2 = black

  let event_at_least_one_black (events) :=
    ∃ (b1: events.1) (b2: events.1), b1 = black ∨ b2 = black

  let event_both_red (events) :=
    ∃ (b1: events.1) (b2: events.1), b1 = red ∧ b2 = red

  -- Proving that the events are mutually exclusive
  (event_exactly_one_black ∧ ∀ (event_both_black : Prop), false) ∧
  -- Proving that the events are complementary
  (event_at_least_one_black ∧ event_both_red = ¬ true) := sorry

end ball_selection_properties_l51_51369


namespace max_ab_at_extremum_l51_51443

noncomputable def f (a b x : ℝ) : ℝ := 4*x^3 - a*x^2 - 2*b*x + 2

theorem max_ab_at_extremum (a b : ℝ) (h0: a > 0) (h1 : b > 0) (h2 : ∃ x, f a b x = 4*x^3 - a*x^2 - 2*b*x + 2 ∧ x = 1 ∧ 12*x^2 - 2*a*x - 2*b = 0) :
  ab ≤ 9 := 
sorry  -- proof not required

end max_ab_at_extremum_l51_51443


namespace find_f_1_plus_f_4_l51_51866

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Ioo (-2 : ℝ) 0 then -2^x else 0  -- Placeholder; define cases properly

axiom odd_function : ∀ x : ℝ, f (-x) = - f x

axiom periodic : ∀ x : ℝ, f (x + 2) = f (x - 2)

axiom specific_interval : ∀ x : ℝ, x ∈ set.Ioo (-2 : ℝ) 0 → f x = -2^x

theorem find_f_1_plus_f_4 : f 1 + f 4 = 1 / 2 :=
by
  sorry  -- Proof needs to be provided here

end find_f_1_plus_f_4_l51_51866


namespace first_nonzero_digit_right_decimal_l51_51231

/--
  To prove that the first nonzero digit to the right of the decimal point of the fraction 1/137 is 9
-/
theorem first_nonzero_digit_right_decimal (n : ℕ) (h1 : n = 137) :
  ∃ d, d = 9 ∧ (∀ k, 10 ^ k * 1 / 137 < 10^(k+1)) → the_first_nonzero_digit_right_of_decimal_is 9 := 
sorry

end first_nonzero_digit_right_decimal_l51_51231


namespace principal_amount_correct_l51_51694

noncomputable def principal_amount (R : ℚ) (T : ℚ) (SI : ℚ) : ℚ :=
  let rate_per_time := R * T / 100
  SI / rate_per_time

theorem principal_amount_correct :
  principal_amount 13 3 5400 ≈ 13846.15 :=
by
  sorry

end principal_amount_correct_l51_51694


namespace female_officers_l51_51271

theorem female_officers (on_duty_females : ℕ) (total_officers_on_duty : ℕ) (percent_on_duty : ℕ) (half_on_duty : ∀ n, n / 2 = on_duty_females) (ten_percent : ∀ n, n / 10 = (100 : ℕ)) : (100 * 10 : ℕ) = 1000 := 
by
  -- Definitions from conditions
  let F := 100 * 10
  have h1: F = 1000, from sorry
  exact h1

end female_officers_l51_51271


namespace probability_palindrome_divisible_by_11_is_zero_l51_51301

def is_palindrome (n : ℕ) :=
  3000 ≤ n ∧ n < 8000 ∧ ∃ (a b : ℕ), 3 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ (n : ℕ), is_palindrome n ∧ n % 11 = 0) → false := by sorry

end probability_palindrome_divisible_by_11_is_zero_l51_51301


namespace number_of_sequences_alternating_parity_l51_51902

/-- 
The number of sequences of 8 digits x_1, x_2, ..., x_8 where no two adjacent x_i have the same parity is 781,250.
-/
theorem number_of_sequences_alternating_parity : 
  let num_sequences := 10 * 5^7 
  ∑ x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 (digits : Fin 8 → Fin 10), 
    (∀ i : Fin 7, digits i % 2 ≠ digits (i + 1) % 2) → 1 = 781250 :=
by sorry

end number_of_sequences_alternating_parity_l51_51902


namespace ellipse_proof_area_triangle_GOH_circle_tangent_GH_l51_51858

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def is_eccentricity (a e : ℝ) : Prop :=
  e = real.sqrt(6) / 3

noncomputable def latus_rectum (x : ℝ) : Prop :=
  x = 3 * real.sqrt(6) / 2

theorem ellipse_proof (a b x y : ℝ) (e : ℝ) :
  a > b →
  b > 0 →
  is_eccentricity a e →
  latus_rectum x →
  ellipse_equation a b x y →
  a = 3 ∧ b = real.sqrt 3 ∧ ellipse_equation 3 (real.sqrt 3) x y :=
sorry

theorem area_triangle_GOH (G H O : ℝ × ℝ) :
  ∃ xG yG xH yH : ℝ, (
    let OG := (G.1 - O.1, G.2 - O.2),
        OH := (H.1 - O.1, H.2 - O.2) in
    let angle_O_G := real.arctan2(OG.2, OG.1) in
    angle_O_G = real.pi / 3 ∧
    2 * OG.1 + OG.2 = 0 ∧
    2 * OH.1 + OH.2 = 0 ∧
    G.1^2 = 9 / 10 ∧
    G.2^2 = 27 / 10 ∧
    H.1^2 = 9 / 2 ∧
    H.2^2 = 3 / 2) →
  let area := (G.1 * H.2 - G.2 * H.1) / 2 in
  area = 3 * real.sqrt(15) / 5 :=
sorry

theorem circle_tangent_GH : 
  ∃ R : ℝ, R = 3 / 2 ∧ ∀ G H O : ℝ × ℝ,
  (
    let OG := 3 * real.sqrt(10) / 5,
        OH := real.sqrt 6 in
    let GH := (H.1 - G.1, H.2 - G.2) in
    OG * OH = R * (real.sqrt (GH.1^2 + GH.2^2)) ∧
    OG^2 + OH^2 = real.sqrt ((G.1 - O.1)^2 + (G.2 - O.2)^2) ∧
    1 / OG^2 + 1 / OH^2 = 4 / 9) →
  let circle := O.1^2 + O.2^2 = 9 / 4 in
  circle = 9 / 4 :=
sorry

end ellipse_proof_area_triangle_GOH_circle_tangent_GH_l51_51858


namespace sum_of_cubes_ages_l51_51190

theorem sum_of_cubes_ages (d t h : ℕ) 
  (h1 : 4 * d + t = 3 * h) 
  (h2 : 4 * h ^ 2 = 2 * d ^ 2 + t ^ 2) 
  (h3 : Nat.gcd d (Nat.gcd t h) = 1)
  : d ^ 3 + t ^ 3 + h ^ 3 = 155557 :=
sorry

end sum_of_cubes_ages_l51_51190


namespace min_abs_sum_l51_51251

open Real

theorem min_abs_sum : ∃ (x : ℝ), (∀ y : ℝ, ∑ z in [| y + 3, y + 6, y + 7].toFinset, abs z ≥ -2) :=
by
  sorry

end min_abs_sum_l51_51251


namespace line_equation_l51_51475

theorem line_equation (l : ℝ → ℝ) :
  (∀ x, l x = -2 * x + b) ∧ (l 0 = -3) → ∃ b, l x = -2 * x - 3 :=
by
  sorry

end line_equation_l51_51475


namespace inequality_proof_l51_51833

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l51_51833


namespace tangent_line_parallel_to_x_axis_l51_51411

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_line_parallel_to_x_axis {x₀ : ℝ} (h : Deriv f x₀ = 0) :
  f x₀ = 1 / Real.exp 1 := by
  sorry

end tangent_line_parallel_to_x_axis_l51_51411


namespace olivia_pays_in_dollars_l51_51546

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l51_51546


namespace dividend_calculation_l51_51462

def quotient : ℕ := 65
def divisor : ℕ := 24
def remainder : ℕ := 5
def expected_dividend : ℕ := 1565

theorem dividend_calculation (q d r : ℕ) (h_q : q = quotient) (h_d : d = divisor) (h_r : r = remainder) : 
  (d * q) + r = expected_dividend := by
  rw [h_q, h_d, h_r]
  show (divisor * quotient) + remainder = expected_dividend
  calc 
    (24 * 65) + 5 = 1560 + 5 : rfl
    ... = 1565 : rfl

end dividend_calculation_l51_51462


namespace find_circle_diameter_l51_51582

noncomputable def circle_diameter (AB CD : ℝ) (h_AB : AB = 16) (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) : ℝ :=
  2 * 10

theorem find_circle_diameter (AB CD : ℝ)
  (h_AB : AB = 16)
  (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) :
  circle_diameter AB CD h_AB h_CD h_perp = 20 := 
  by sorry

end find_circle_diameter_l51_51582


namespace problem_statement_l51_51344

/-- 
Mathematically equivalent proof problem: 
Prove that (8^5 ÷ 8^3) ⋅ 16^4 ÷ 2^3 = 524288 
--/
theorem problem_statement : ((8^5 / 8^3) * 16^4 / 2^3) = 524288 := by
  sorry

end problem_statement_l51_51344


namespace nonnegative_solutions_count_l51_51437

-- Define the equation x^2 + 6x + 9 = 0
def equation (x : ℝ) : Prop := x^2 + 6 * x + 9 = 0

-- Prove that the number of nonnegative solutions to the equation is 0
theorem nonnegative_solutions_count : finset.card ({x : ℝ | equation x ∧ 0 ≤ x}.to_finset) = 0 :=
by
  -- Proof goes here; add 'sorry' for now
  sorry

end nonnegative_solutions_count_l51_51437


namespace roots_polynomial_expression_l51_51114

theorem roots_polynomial_expression (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a * b + a * c + b * c = -1)
  (h3 : a * b * c = -2) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 0 :=
by
  sorry

end roots_polynomial_expression_l51_51114


namespace enclosed_area_by_equation_l51_51352

theorem enclosed_area_by_equation :
  ∀ x y : ℝ, x^2 + y^2 - 6 * x + 8 * y = -9 → (real.pi * 4^2 = 16 * real.pi) :=
by
  intro x y h
  sorry

end enclosed_area_by_equation_l51_51352


namespace transport_tax_correct_l51_51767

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end transport_tax_correct_l51_51767


namespace intersection_of_A_B_l51_51428

-- Set definitions
def A : set ℤ := {0, 2}
def B : set ℤ := {-2, -1, 0, 1, 2}

-- Proof statement
theorem intersection_of_A_B : A ∩ B = {0, 2} :=
by {
  sorry
}

end intersection_of_A_B_l51_51428


namespace short_bushes_total_l51_51197

theorem short_bushes_total (current_short_bushes new_short_bushes : ℕ) (h1 : current_short_bushes = 37) (h2 : new_short_bushes = 20) :
  current_short_bushes + new_short_bushes = 57 :=
by
  rw [h1, h2]
  rfl

end short_bushes_total_l51_51197


namespace intersection_points_l51_51424

-- Conditions
def circle_parametric (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, 1 + Real.sin α)

def line_polar (p θ : ℝ) : Prop :=
  p * Real.sin θ = 1

-- Proof statement
theorem intersection_points (α p θ : ℝ) (h_polar : line_polar p θ) :
  (circle_parametric α = (-1, 1) ∨ circle_parametric α = (1, 1)) :=
by 
  sorry

end intersection_points_l51_51424


namespace find_unit_vector_perpendicular_to_a_and_b_l51_51433

-- Define the vectors a and b
def vec_a : ℝ × ℝ × ℝ := (0, 1, 1)
def vec_b : ℝ × ℝ × ℝ := (1, 2, 0)

-- Define the unit vectors e1 and e2
def unit_vec1 : ℝ × ℝ × ℝ := (sqrt 6 / 3, -sqrt 6 / 6, sqrt 6 / 6)
def unit_vec2 : ℝ × ℝ × ℝ := (-sqrt 6 / 3, sqrt 6 / 6, -sqrt 6 / 6)

-- Statement of the problem
theorem find_unit_vector_perpendicular_to_a_and_b :
  (vector.cross vec_a vec_b = unit_vec1 \/ vector.cross vec_a vec_b = unit_vec2) ∧
  ∥vector.cross vec_a vec_b∥ = 1 :=
sorry

end find_unit_vector_perpendicular_to_a_and_b_l51_51433


namespace total_area_correct_l51_51463

-- Define the conditions
def corner_angles_right : Prop := true -- All corner angles are right angles

def left_vert_edge_length : ℕ := 7 -- Left vertical edge length in units
def top_horiz_edge_length : ℕ := 7 -- Top horizontal edge length in units
def vert_seg_length : ℕ := 3 -- Vertical segment between horizontal segments length in units
def bottom_horiz_seg_length : ℕ := 3 -- Bottom horizontal segment between vertical segments length in units
def right_vert_seg_length : ℕ := 2 -- Right vertical segment length in units
def top_horiz_segment_length : ℕ := 5 -- Top horizontal edge segment between vertical segments length in units

-- The areas of the divided rectangles
def rect1_area : ℕ := left_vert_edge_length * top_horiz_edge_length -- 7 * 7
def rect2_area : ℕ := vert_seg_length * bottom_horiz_seg_length -- 3 * 5
def rect3_area : ℕ := right_vert_seg_length * top_horiz_segment_length -- 5 * 6

-- Total area of the shape
def total_area : ℕ := rect1_area + rect2_area + rect3_area

-- Lean theorem statement for the proof
theorem total_area_correct : total_area = 94 := by
  have h1 : rect1_area = 49 := rfl
  have h2 : rect2_area = 15 := rfl
  have h3 : rect3_area = 30 := rfl
  calc
    total_area
        = rect1_area + rect2_area + rect3_area : rfl
    ... = 49 + 15 + 30 : by rw [h1, h2, h3]
    ... = 94 : rfl

-- sorry to skip the proof
#check total_area_correct

end total_area_correct_l51_51463


namespace max_area_of_triangle_l51_51484

theorem max_area_of_triangle (A B C : Point) (AB : Segment)
  (hAB : AB.length = 3) (m : ℝ) (h_m_ge_2 : m ≥ 2) 
  (h_sinB_eq_m_sinA : sin B.angle = m * sin A.angle) :
  ∃ (S_max : ℝ), S_max = 3 ∧ area_of_triangle A B C = S_max := 
sorry

end max_area_of_triangle_l51_51484


namespace problem_l51_51119

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end problem_l51_51119


namespace find_xy_l51_51035

-- Define the conditions as constants for clarity
def condition1 (x : ℝ) : Prop := 0.60 / x = 6 / 2
def condition2 (x y : ℝ) : Prop := x / y = 8 / 12

theorem find_xy (x y : ℝ) (hx : condition1 x) (hy : condition2 x y) : 
  x = 0.20 ∧ y = 0.30 :=
by
  sorry

end find_xy_l51_51035


namespace CoreyCandies_l51_51999

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end CoreyCandies_l51_51999


namespace suff_but_not_nec_l51_51704

-- Define the conditions in Lean
def x_gt_4 (x : ℝ) : Prop := x > 4
def x_ge_4 (x : ℝ) : Prop := x ≥ 4

-- State the theorem about sufficiency and necessity
theorem suff_but_not_nec (x : ℝ) : (x_gt_4 x → x_ge_4 x) ∧ ¬(x_ge_4 x → x_gt_4 x) :=
by
  -- For sufficiency: If x > 4, then x ≥ 4
  have suff : x_gt_4 x → x_ge_4 x := λ h₁, le_of_lt h₁
  -- For necessity: If x ≥ 4, it doesn't necessarily mean x > 4 (x could be equal to 4)
  have not_nec : ¬(x_ge_4 x → x_gt_4 x) := λ h₂, (h₂ 4 (le_refl 4)).not_lt (lt_self_iff_false 4).mp
  -- Combine both results
  exact ⟨suff, not_nec⟩

end suff_but_not_nec_l51_51704


namespace A_completes_in_10_days_l51_51289

def work_rate (x : ℕ) : ℚ := 1 / x.to_rat

def combined_work_rate (x : ℕ) (b_days : ℕ) : ℚ := work_rate x + work_rate b_days

def work_done_together (x : ℕ) (b_days : ℕ) (days_together : ℕ) : ℚ := days_together * combined_work_rate x b_days

def work_done_alone (b_days : ℕ) (days_alone : ℕ) : ℚ := days_alone * work_rate b_days

def total_work_done (x : ℕ) (b_days : ℕ) (days_together : ℕ) (days_alone : ℕ) : ℚ := work_done_together x b_days days_together + work_done_alone b_days days_alone

theorem A_completes_in_10_days (x : ℕ) (b_days : ℕ) (days_together : ℕ) (days_alone : ℕ) :
  b_days = 20 →
  days_together = 5 →
  days_alone = 5 →
  total_work_done x b_days days_together days_alone = 1 →
  x = 10 := 
by
  intros hb_days hdays_together hdays_alone htotal_work
  sorry

end A_completes_in_10_days_l51_51289


namespace jenny_change_l51_51068

-- Definitions for the conditions
def single_sided_cost_per_page : ℝ := 0.10
def double_sided_cost_per_page : ℝ := 0.17
def pages_per_essay : ℕ := 25
def single_sided_copies : ℕ := 5
def double_sided_copies : ℕ := 2
def pen_cost_before_tax : ℝ := 1.50
def number_of_pens : ℕ := 7
def sales_tax_rate : ℝ := 0.10
def payment_amount : ℝ := 2 * 20.00

-- Hypothesis for the total costs and calculations
noncomputable def total_single_sided_cost : ℝ := single_sided_copies * pages_per_essay * single_sided_cost_per_page
noncomputable def total_double_sided_cost : ℝ := double_sided_copies * pages_per_essay * double_sided_cost_per_page
noncomputable def total_pen_cost_before_tax : ℝ := number_of_pens * pen_cost_before_tax
noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_pen_cost_before_tax
noncomputable def total_pen_cost : ℝ := total_pen_cost_before_tax + total_sales_tax
noncomputable def total_printing_cost : ℝ := total_single_sided_cost + total_double_sided_cost
noncomputable def total_cost : ℝ := total_printing_cost + total_pen_cost
noncomputable def change : ℝ := payment_amount - total_cost

-- The proof statement
theorem jenny_change : change = 7.45 := by
  sorry

end jenny_change_l51_51068


namespace routes_P_to_T_eq_three_l51_51331

-- Define points P, Q, R, S, T.
inductive Point
| P | Q | R | S | T

open Point

-- Define the allowed paths using a structure.
structure Path where
  source : Point
  destination : Point

-- List of allowed paths.
def paths : List Path :=
[
  { source := P, destination := Q },
  { source := Q, destination := R },
  { source := Q, destination := S },
  { source := S, destination := T },
  { source := Q, destination := T },
  { source := R, destination := T }
]

-- Define a predicate that checks if a sequence of points forms a valid route.
def valid_route : List Point → Bool
| [] => false
| [_] => false
| (p::q::rest) => (paths.contains {source := p, destination := q}) && valid_route (q::rest)

-- Determine all routes from point P to point T.
def routes_from_to (start end : Point) : List (List Point) :=
  List.filter (fun route => match route.head?, route.reverse.head? with
                             | some P, some target => target = end
                             | _, _ => false)
  (List.permutations [P, Q, R, S, T]).filter valid_route

def num_routes_P_to_T := routes_from_to P T |>.length

-- Prove the number of different routes from P to T is 3
theorem routes_P_to_T_eq_three : num_routes_P_to_T = 3 := by
  -- Proof omitted, just another check to skip.
  sorry

end routes_P_to_T_eq_three_l51_51331


namespace bob_grade_l51_51774

theorem bob_grade (g : ℕ) (h : g ≠ 9) (prod_grades : g = 2007 → g) : g = 3 :=
by
  sorry

end bob_grade_l51_51774


namespace stationery_sales_calculation_l51_51291

-- Definitions
def total_sales : ℕ := 120
def fabric_percentage : ℝ := 0.30
def jewelry_percentage : ℝ := 0.20
def knitting_percentage : ℝ := 0.15
def home_decor_percentage : ℝ := 0.10
def stationery_percentage := 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage)
def stationery_sales := stationery_percentage * total_sales

-- Statement to prove
theorem stationery_sales_calculation : stationery_sales = 30 := by
  -- Providing the initial values and assumptions to the context
  have h1 : total_sales = 120 := rfl
  have h2 : fabric_percentage = 0.30 := rfl
  have h3 : jewelry_percentage = 0.20 := rfl
  have h4 : knitting_percentage = 0.15 := rfl
  have h5 : home_decor_percentage = 0.10 := rfl
  
  -- Calculating the stationery percentage and sales
  have h_stationery_percentage : stationery_percentage = 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage) := rfl
  have h_stationery_sales : stationery_sales = stationery_percentage * total_sales := rfl

  -- The calculated value should match the proof's requirement
  sorry

end stationery_sales_calculation_l51_51291


namespace regions_formula_l51_51485

-- Define the number of regions R(n) created by n lines
def regions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

-- Theorem statement: for n lines, no two parallel, no three concurrent, the regions are defined by the formula
theorem regions_formula (n : ℕ) : regions n = 1 + (n * (n + 1)) / 2 := 
by sorry

end regions_formula_l51_51485


namespace kolya_product_divisors_1024_l51_51086

theorem kolya_product_divisors_1024 :
  let number := 1024
  let screen_limit := 10^16
  ∃ (divisors : List ℕ), 
    (∀ d ∈ divisors, d ∣ number) ∧ 
    (∏ d in divisors, d) = 2^55 ∧ 
    2^55 > screen_limit :=
by
  sorry

end kolya_product_divisors_1024_l51_51086


namespace jared_text_messages_in_january_l51_51490

theorem jared_text_messages_in_january :
  ∃ n, n = 4 ∧
    (∀ m, (m = 1 ∧ m = 2 ∧ m = 8 ∧ m = 16 → (m = 2 * 2 * 2)) → n = 2 * 2) :=
begin
  sorry

end jared_text_messages_in_january_l51_51490


namespace amount_spent_on_petrol_l51_51754

theorem amount_spent_on_petrol
    (rent milk groceries education miscellaneous savings salary petrol : ℝ)
    (h1 : rent = 5000)
    (h2 : milk = 1500)
    (h3 : groceries = 4500)
    (h4 : education = 2500)
    (h5 : miscellaneous = 2500)
    (h6 : savings = 0.10 * salary)
    (h7 : savings = 2000)
    (total_salary : salary = 20000) : petrol = 2000 := by
  sorry

end amount_spent_on_petrol_l51_51754


namespace equal_striped_areas_l51_51919

theorem equal_striped_areas (A B C D : ℝ) (h_AD_DB : D = A + B) (h_CD2 : C^2 = A * B) :
  (π * C^2 / 4 = π * B^2 / 8 - π * A^2 / 8 - π * D^2 / 8) := 
sorry

end equal_striped_areas_l51_51919


namespace number_of_integers_in_absolute_value_solution_l51_51015

theorem number_of_integers_in_absolute_value_solution :
  {x : ℤ | |(x : ℝ) - 2| ≤ 4.5}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_in_absolute_value_solution_l51_51015


namespace range_of_func_l51_51596

def func (x : ℕ) : ℕ := 2 * x + 1

theorem range_of_func : {y : ℕ | ∃ x : ℕ, x ∈ {1, 2, 3} ∧ y = func x} = {3, 5, 7} := 
by
  sorry

end range_of_func_l51_51596


namespace shaded_area_percentage_l51_51644

theorem shaded_area_percentage {WXYZ : ℝ} (h_right_angles : ∀(A B C : ℝ), ∠ABC = 90) 
(side_WXYZ : WXYZ = 6) 
(side_shaded1 : 2) 
(side_shaded2_1 : 3) (side_shaded2_2 : 5) 
(height_shaded3 : 1) (length_shaded3 : 6) : 
  (4 + (5^2 - 3^2) + 6) / (6^2) * 100 = 72.22 := by
  -- Here we skip the proof
  sorry

end shaded_area_percentage_l51_51644


namespace divisors_of_n_squared_not_dividing_n_l51_51956

def n : ℕ := 2^20 * 5^15
def n_squared : ℕ := n * n

theorem divisors_of_n_squared_not_dividing_n :
  (finset.filter (λ d : ℕ, d < n ∧ ¬d ∣ n) (finset.divisors n_squared)).card = 299 :=
by sorry

end divisors_of_n_squared_not_dividing_n_l51_51956


namespace find_value_of_m_l51_51282

noncomputable def circle_diameter_circle_through_origin_has_m_value : Prop :=
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + y^2 + x - 6y + m = 0 ↔ x + 2y - 3 = 0) → m = 3

theorem find_value_of_m : circle_diameter_circle_through_origin_has_m_value :=
sorry

end find_value_of_m_l51_51282


namespace number_of_12_tuples_l51_51357

def is_half_sequence (xs : List ℝ) : Prop :=
  ∀ i, 0 < i → i < xs.length → xs.nth_le i sorry = (xs.nth_le (i - 1) sorry) / 2

theorem number_of_12_tuples : 
  (∃! xs : List ℝ, 
    xs.length = 12 ∧ 
    (sum (List.range 12).map (λ i, (xs.nth_le i sorry - xs.nth_le (i + 1 % 12) sorry)^2)) = 1/13 ∧ 
    is_half_sequence xs) :=
sorry

end number_of_12_tuples_l51_51357


namespace flagpole_height_l51_51625

theorem flagpole_height (h : ℕ)
  (shadow_flagpole : ℕ := 72)
  (height_pole : ℕ := 18)
  (shadow_pole : ℕ := 27)
  (ratio_shadow : shadow_flagpole / shadow_pole = 8 / 3) :
  h = 48 :=
by
  sorry

end flagpole_height_l51_51625


namespace least_faces_triangular_pyramid_l51_51201

def triangular_prism_faces : ℕ := 5
def quadrangular_prism_faces : ℕ := 6
def triangular_pyramid_faces : ℕ := 4
def quadrangular_pyramid_faces : ℕ := 5
def truncated_quadrangular_pyramid_faces : ℕ := 5 -- assuming the minimum possible value

theorem least_faces_triangular_pyramid :
  triangular_pyramid_faces < triangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_pyramid_faces ∧
  triangular_pyramid_faces ≤ truncated_quadrangular_pyramid_faces :=
by
  sorry

end least_faces_triangular_pyramid_l51_51201


namespace line_intersects_circle_at_right_angle_l51_51003

theorem line_intersects_circle_at_right_angle 
  (a : ℝ) :
  (∃ A B : ℝ × ℝ, x + 2 * y = a ∧ x^2 + y^2 = 4) ∧ 
  (∃ OA OB : ℝ × ℝ, |OA + OB| = |OA - OB| ∧ A = OA ∧ B = OB) 
  → a = sqrt 10 ∨ a = -sqrt 10 :=
sorry

end line_intersects_circle_at_right_angle_l51_51003


namespace olivia_pays_in_dollars_l51_51545

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l51_51545


namespace kolya_cannot_see_full_result_l51_51092

theorem kolya_cannot_see_full_result : let n := 1024 in 
                                      let num_decimals := 16 in 
                                      (num_divisors_product n > 10^num_decimals) := 
begin
    let n := 1024,
    let num_decimals := 16,
    have product := (∑ i in range (nat.log2 n), i),
    have sum_of_powers := (2^product),
    have required_digits := (10^num_decimals),
    exact sum_of_powers > required_digits,
end

end kolya_cannot_see_full_result_l51_51092


namespace count_irrationals_in_list_l51_51060

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

theorem count_irrationals_in_list :
  let L := [3.1415926, real.cbrt 64, 1.010010001, 2 - real.sqrt 5, real.pi / 2, 22 / 3, 2, 2.15] in
  (L.filter is_irrational).length = 3 := 
by
  sorry

end count_irrationals_in_list_l51_51060


namespace intersection_points_of_line_ellipse_l51_51912

open Real

theorem intersection_points_of_line_ellipse (m n : ℝ) :
  ((∀ x y : ℝ, (m * x + n * y ≠ 4) ∨ (x^2 + y^2 ≠ 4)) -> 
  (m^2 + n^2 < 4)) -> 
  (∃ P : ℝ × ℝ, 2 = (count (λ (Q : ℝ × ℝ), (Q.1^2 / 9 + Q.2^2 / 4 = 1) ∧ 
    (Q.2 - n) / (Q.1 - m) = - ∞))) :=
sorry

end intersection_points_of_line_ellipse_l51_51912


namespace simplify_expression_l51_51619

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l51_51619


namespace relationship_y1_y2_l51_51036

theorem relationship_y1_y2 :
  ∀ (y1 y2 : ℝ), 
  (∃ (y1 y2 : ℝ), 
    y1 = (-1 / 2) + 4 ∧ 
    y2 = 1 + 4 ∧
    y1 < y2) :=
by {
  assume y1 y2,
  use ((-1 / 2) + 4),
  use (1 + 4),
  have h1 : ((-1 / 2) + 4) = 3.5 := by norm_num,
  have h2 : (1 + 4) = 5 := by norm_num,
  rw [h1, h2],
  exact lt_of_lt_of_le (by norm_num) (by norm_num),
  sorry
}

end relationship_y1_y2_l51_51036


namespace minimum_swaps_l51_51501

def is_chameleon (n : ℕ) (s : List Char) : Prop :=
  s.length = 3 * n ∧ s.count 'a' = n ∧ s.count 'b' = n ∧ s.count 'c' = n

def swap (s : List Char) (i : ℕ) : List Char :=
  if i < s.length - 1 then s.take i ++ [s[i+1], s[i]] ++ s.drop (i+2) else s

def num_swaps_to_convert (s t : List Char) : ℕ := 
  sorry -- Placeholder for the function to calculate minimum swaps

theorem minimum_swaps (n : ℕ) (X Y : List Char) (hX : is_chameleon n X) (hY : is_chameleon n Y) :
  ∃ (Z : List Char), is_chameleon n Z ∧ num_swaps_to_convert X Z ≥ 3 * n^2 / 2 :=
sorry

end minimum_swaps_l51_51501


namespace transportation_tax_correct_l51_51765

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end transportation_tax_correct_l51_51765


namespace divisors_product_exceeds_16_digits_l51_51096

theorem divisors_product_exceeds_16_digits :
  let n := 1024
  let screen_digits := 16
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  (String.length (Natural.toString divisors_product) > screen_digits) := 
by {
  let n := 1024
  let screen_digits := 16
  let divisors := List.range (11)
  let divisors_product := (List.range (11)).prod (fun x => 2^x)
  show (String.length (Natural.toString divisors_product) > screen_digits),
  sorry
}

end divisors_product_exceeds_16_digits_l51_51096


namespace PQ_perpendicular_AD_l51_51958

-- Define the data structures representing points and lines in a cyclic quadrilateral
variables {α : Type*} [EuclideanGeometry α]

-- Let ABCD be a convex cyclic quadrilateral
variables (A B C D E P Q : α)

-- Assume E is the midpoint of (A, B)
def midpoint (A B E : α) : Prop :=
  segment A E = segment E B

-- Assumptions
variables [H1 : IsCyclicQuadrilateral A B C D]
variables [H2 : IsAcuteAngle A B C]
variables [H3 : Midpoint A B E]
variables [H4 : PerpendicularThroughMidpoint E (line A B) P]
variables [H5 : PerpendicularThroughMidpoint E (line C D) Q]

-- Define perpendicular lines, lines (PQ) and (AD)
def perpendicular (l1 l2 : Line α) : Prop :=
  ∃ p q : α, p ≠ q ∧ l1 = line p q ∧ ∠ (line p q) l2 = 90

-- The final statement to prove that (PQ) is perpendicular to (AD)
theorem PQ_perpendicular_AD :
  perpendicular (line P Q) (line A D) :=
sorry

end PQ_perpendicular_AD_l51_51958


namespace general_formula_an_l51_51862

noncomputable def a : ℕ → ℝ
| 0     := 3
| (n+1) := 3 * a n

theorem general_formula_an (a : ℕ → ℝ)
  (h1 : ∀ n, log (a (n + 1)) = log (a n) + log 3)
  (h2 : log (a 1) + log (a 2) + log (a 3) = 6 * log 3) :
  ∀ n, a n = 3 ^ n :=
by
  intro n
  induction n with d hd
  · simp [a]
  · simp [a, hd]
    sorry

end general_formula_an_l51_51862


namespace closest_square_to_350_l51_51660

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l51_51660


namespace intersection_A_B_l51_51427

open Set

def isInSetA (x : ℕ) : Prop := ∃ n : ℕ, x = 3 * n + 2
def A : Set ℕ := { x | isInSetA x }
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_A_B :
  A ∩ B = {8, 14} :=
sorry

end intersection_A_B_l51_51427


namespace min_abs_sum_value_l51_51248

def abs_sum (x : ℝ) := |x + 3| + |x + 6| + |x + 7|

theorem min_abs_sum_value : ∃ x : ℝ, abs_sum x = 4 ∧ ∀ y : ℝ, abs_sum y ≥ abs_sum x := 
by 
  use -6
  have abs_sum_eq : abs_sum (-6) = 4 := by
    simp [abs_sum]
  -- Other conditions ensuring it is the minimum
  sorry

end min_abs_sum_value_l51_51248


namespace closest_square_to_350_l51_51661

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l51_51661


namespace segment_MP_length_correct_l51_51139

noncomputable def segment_MP_length (r : ℝ) (MN : ℝ) (P : ℝ) : Prop :=
∃ (M N : ℝ × ℝ), 
  -- M and N are points on the circle of radius r.
  (M.1^2 + M.2^2 = r^2) ∧ (N.1^2 + N.2^2 = r^2) ∧
  -- MN is the distance between points M and N.
  (real.dist M N = MN) ∧
  -- P is the midpoint of the minor arc between M and N.
  let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2) in
  -- The required condition is the length of MP.
  (real.dist M P = 2 * real.sqrt 10)

theorem segment_MP_length_correct : segment_MP_length 10 12 (2 * real.sqrt 10) :=
sorry

end segment_MP_length_correct_l51_51139


namespace find_a_l51_51511

noncomputable def a : ℝ :=
  let f (x : ℝ) := a^x
  let condition := (a > 0) ∧ (a ≠ 1) ∧ (f 1 - f (-1)) / (f 2 - f (-2)) = 3 / 10
  have condition : Prop := by
    sorry
  have h1 : f (1) = a
  have h2 : f (-1) = 1 / a
  have h3 : f (2) = a^2
  have h4 : f (-2) = 1 / (a^2)
  solve_by_elim

theorem find_a (a : ℝ) (h : (a > 0) ∧ (a ≠ 1) ∧ ((a - 1 / a) / (a^2 - 1 / a^2) = 3 / 10)) : a = 3 :=
  sorry

end find_a_l51_51511


namespace vector_magnitude_eq_3_sqrt_10_l51_51379

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem vector_magnitude_eq_3_sqrt_10 :
  let a := (2, -3, 5) 
  let b := (-3, 1, 2) 
  magnitude (a.1 - 2 * b.1, a.2 - 2 * b.2, a.3 - 2 * b.3) = 3 * real.sqrt 10 :=
by
  sorry

end vector_magnitude_eq_3_sqrt_10_l51_51379


namespace equivalence_of_expectations_l51_51561

variables {Ω : Type*} {ξ ξ_n : Ω → ℝ}
variables [measure_theory.probability_measure (measure_theory.measure_space.volume)]

-- Conditions
def conv_p (seq : ℕ → Ω → ℝ) (limit : Ω → ℝ) : Prop :=
  ∀ ε > 0, ∀ δ > 0, ∃ N, ∀ n ≥ N, measure_theory.probability_measure (measure_theory.measure_space.volume) {ω | abs (seq n ω - limit ω) > ε} < δ

def finite_expectation (X : Ω → ℝ) : Prop :=
  measure_theory.integrable X

-- Problem statement
theorem equivalence_of_expectations (h_conv : conv_p ξ_n ξ)
  (h_finite_ξ : finite_expectation ξ)
  (h_finite_ξ_n : ∀ n, finite_expectation (ξ_n n)) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, measure_theory.expectation (λ ω, abs (ξ_n n ω - ξ ω)) < ε) ↔ 
  (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (measure_theory.expectation (ξ_n n) - measure_theory.expectation ξ) < ε) :=
sorry

end equivalence_of_expectations_l51_51561


namespace garden_watering_system_pumps_l51_51722

-- Define conditions
def rate := 500 -- gallons per hour
def time := 30 / 60 -- hours, i.e., converting 30 minutes to hours

-- Theorem statement
theorem garden_watering_system_pumps :
  rate * time = 250 := by
  sorry

end garden_watering_system_pumps_l51_51722


namespace closest_square_to_350_l51_51657

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l51_51657


namespace selling_price_correct_l51_51014

-- Define the conditions
def boxes := 3
def face_masks_per_box := 20
def cost_price := 15  -- in dollars
def profit := 15      -- in dollars

-- Define the total number of face masks
def total_face_masks := boxes * face_masks_per_box

-- Define the total amount he wants after selling all face masks
def total_amount := cost_price + profit

-- Prove that the selling price per face mask is $0.50
noncomputable def selling_price_per_face_mask : ℚ :=
  total_amount / total_face_masks

theorem selling_price_correct : selling_price_per_face_mask = 0.50 := by
  sorry

end selling_price_correct_l51_51014


namespace sin_ratio_and_angle_measure_l51_51038

noncomputable theory

variables (α β γ a b c : ℝ)
variables (A B C : ℝ)
variables (h1 : α = a) (h2 : β = b) (h3 : γ = c)
variables (h4 : (a - 3 * b) * cos C = c * (3 * cos B - cos A))
variables (h5 : c = sqrt 7 * a)

theorem sin_ratio_and_angle_measure 
  (h_triangle: α = a ∧ β = b ∧ γ = c ∧ (a - 3 * b) * cos C = c * (3 * cos B - cos A) ∧ c = sqrt 7 * a):
  (sin B / sin A = 3) ∧ (C = π / 3) :=
by 
  sorry

end sin_ratio_and_angle_measure_l51_51038


namespace arithmetic_series_sum_l51_51328

theorem arithmetic_series_sum : 
  ∀ (a d a_n : ℤ), 
  a = -48 → d = 2 → a_n = 0 → 
  ∃ n S : ℤ, 
  a + (n - 1) * d = a_n ∧ 
  S = n * (a + a_n) / 2 ∧ 
  S = -600 :=
by
  intros a d a_n ha hd han
  have h₁ : a = -48 := ha
  have h₂ : d = 2 := hd
  have h₃ : a_n = 0 := han
  sorry

end arithmetic_series_sum_l51_51328


namespace circle_radius_l51_51717

theorem circle_radius (M N : ℝ) (h1 : M / N = 15) (h2 : M = 60 * Real.pi) : 
  ∃ r : ℝ, r = 2 * Real.sqrt 15 ∧ M = π * r^2 ∧ N = 2 * π * r :=
by
  use 2 * Real.sqrt 15
  split
  sorry
  sorry
  sorry

end circle_radius_l51_51717


namespace problem_CD_CE_l51_51383

noncomputable def big_circle : Type := sorry
noncomputable def small_circle : Type := sorry

def radius_small (O1 : small_circle) : ℝ := 3

variables (O : big_circle) (O1 : small_circle) (P A B C D E : point)
variables (AB l : line)

-- Conditions of the problem
axiom tangent_external_O_to_O1 : tangent O O1 P
axiom tangent_AB_O_A : tangent O AB A
axiom tangent_AB_O1_B : tangent O1 AB B
axiom tangent_l_O1_C : tangent O1 l C
axiom intersect_l_O_D_E : intersect l O D E
axiom parallel_l_AB : parallel l AB

-- Question to Prove
theorem problem_CD_CE : CD * CE = 9 := sorry

end problem_CD_CE_l51_51383


namespace maddie_lipsticks_l51_51967

theorem maddie_lipsticks (
  makeup_palettes_cost : ℕ := 15,
  hair_color_cost : ℕ := 4,
  lipstick_cost : ℚ := 2.50,
  total_paid : ℕ := 67,
  makeup_palettes_count : ℕ := 3,
  hair_color_count : ℕ := 3
) : ∃ lipstick_count : ℕ, lipstick_count = 4 :=
  by
  let total_cost := makeup_palettes_count * makeup_palettes_cost + hair_color_count * hair_color_cost
  let remaining_amount := total_paid - total_cost
  let lipstick_count := remaining_amount / lipstick_cost
  have h1 : remaining_amount = 10 := by sorry
  have h2 : lipstick_count = 4 := by sorry
  use 4
  exact h2

end maddie_lipsticks_l51_51967


namespace quadrilateral_perimeter_ABCD_l51_51058

-- Define the necessary geometric entities and conditions.
def angle_60 := 60 * (real.pi / 180)

def right_angle := real.pi / 2

noncomputable def length (x : ℝ) := ℝ

-- Define properties of triangles and their side lengths.
def triangle_30_60_90 (a b c : length) : Prop :=
  c = 2 * a ∧ b = a * real.sqrt 3

-- Define the triangles based on given conditions.
def triangle_ABE (A B E : length) : Prop :=
  triangle_30_60_90 E B A ∧ B = 18 * (1 / 2) ∧ A = 18 * (real.sqrt 3 / 2)

def triangle_BCE (B C E : length) : Prop :=
  triangle_30_60_90 E C B ∧ C = B * (real.sqrt 3 / 2) ∧ E = B * (1 / 2)

def triangle_CDE (C D E : length) : Prop :=
  triangle_30_60_90 E D C ∧ D = E * (real.sqrt 3 / 2) ∧ E = 4.5 * (1 / 2)

-- Define the quadrilateral.
def quadrilateral_ABCD (A B C D : length) :=
  triangle_ABE A B 18 ∧
  triangle_BCE 9 C 4.5 ∧
  triangle_CDE 4.5 D 2.25 ∧
  D = 20.25 ∧ A = 20.25 ∧
  B = 9 * (real.sqrt 3)

-- The perimeter function.
def perimeter (A B C D : length) : length :=
  A + B + C + D

-- Statement of the proof problem.
theorem quadrilateral_perimeter_ABCD :
  ∀ (A B C D : length),
    quadrilateral_ABCD A B C D →
    perimeter A B C D = 20.25 + 15.75 * real.sqrt 3 :=
by 
  intros A B C D h,
  sorry

end quadrilateral_perimeter_ABCD_l51_51058


namespace symmetric_line_equation_l51_51169

theorem symmetric_line_equation : 
  ∀ (P : ℝ × ℝ) (L : ℝ × ℝ × ℝ), 
  P = (1, 1) → 
  L = (2, 3, -6) → 
  (∃ (a b c : ℝ), a * 1 + b * 1 + c = 0 → a * x + b * y + c = 0 ↔ 2 * x + 3 * y - 4 = 0) 
:= 
sorry

end symmetric_line_equation_l51_51169


namespace john_reaching_floor_pushups_l51_51940

-- Definitions based on conditions
def john_train_days_per_week : ℕ := 5
def reps_to_progress : ℕ := 20
def variations : ℕ := 3  -- wall, incline, knee

-- Mathematical statement
theorem john_reaching_floor_pushups : 
  (reps_to_progress * variations) / john_train_days_per_week = 12 := 
by
  sorry

end john_reaching_floor_pushups_l51_51940


namespace fraction_of_females_is_two_thirds_l51_51770

-- Problem Definitions
variable (total_students : ℕ) (non_foreign_males : ℕ) (male_fraction_foreign : ℚ)
variable (total_males total_females : ℕ)

-- Conditions
def school_conditions :=
  total_students = 300 ∧
  non_foreign_males = 90 ∧
  male_fraction_foreign = 1 / 10 ∧
  total_males = non_foreign_males / (1 - male_fraction_foreign)

-- The ultimate goal
def fraction_females : ℚ := total_females / total_students

theorem fraction_of_females_is_two_thirds : 
  school_conditions total_students non_foreign_males male_fraction_foreign total_males total_females →
  total_females = total_students - total_males →
  fraction_females total_students total_females = 2 / 3 := 
by
  intros h_conditions h_females
  sorry

end fraction_of_females_is_two_thirds_l51_51770


namespace compare_decimal_to_fraction_l51_51591

theorem compare_decimal_to_fraction : (0.650 - (1 / 8) = 0.525) :=
by
  /- We need to prove that 0.650 - 1/8 = 0.525 -/
  sorry

end compare_decimal_to_fraction_l51_51591


namespace more_likely_white_crows_same_l51_51065

theorem more_likely_white_crows_same (a b c d : ℕ) 
  (h_birch_crows : a + b = 50) 
  (h_oak_crows : c + d = 50) 
  (h_birch_black_geq_white : b ≥ a) 
  (h_oak_black_geq_white : d ≥ c - 1) 
  (h_move_birch_to_oak : ∀ x, x = a + b ∧ x = c + d → a.move → b.move)
  (h_move_oak_to_birch : ∀ x, x = a + b ∧ x = c + d → a.move ← b.move) : 
  (let pa := (b * (d + 1) + a * (c + 1)) / (50 * 51) in 
  let pnot_a := (b * c + a * d) / (50 * 51) in 
  pa > pnot_a) := sorry

end more_likely_white_crows_same_l51_51065


namespace glove_selection_correct_l51_51368

-- Define the total number of different pairs of gloves
def num_pairs : Nat := 6

-- Define the required number of gloves to select
def num_gloves_to_select : Nat := 4

-- Define the function to calculate the number of ways to select 4 gloves with exactly one matching pair
noncomputable def count_ways_to_select_gloves (num_pairs : Nat) : Nat :=
  let select_pair := Nat.choose num_pairs 1
  let remaining_gloves := 2 * (num_pairs - 1)
  let select_two_from_remaining := Nat.choose remaining_gloves 2
  let subtract_unwanted_pairs := num_pairs - 1
  select_pair * (select_two_from_remaining - subtract_unwanted_pairs)

-- The correct answer we need to prove
def expected_result : Nat := 240

-- The theorem to prove the number of ways to select the gloves
theorem glove_selection_correct : count_ways_to_select_gloves num_pairs = expected_result :=
  by
    sorry

end glove_selection_correct_l51_51368


namespace telephone_number_l51_51308

/-- A represents a unique digit in the telephone number ABC-DEF-GHIJ. Prove that A = 9,
given the following conditions:
1. A > B > C, D > E > F, G > H > I > J.
2. D, E, F are consecutive digits.
3. G, H, I, J are consecutive digits that include both odd and even numbers.
4. A + B + C = 17.
-/
theorem telephone_number (A B C D E F G H I J : ℕ)
  (h_unique : List.nodup [A, B, C, D, E, F, G, H, I, J])
  (h_A_gt_BC : A > B ∧ B > C)
  (h_D_gt_EF : D > E ∧ E > F)
  (h_G_gt_HIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_DEF : D = E + 1 ∨ E = D + 1 ∧ E = F + 1 ∨ F = E + 1)
  (h_consecutive_GHIJ : (G - 3 = H + 2 ∧ H - 2 = I + 1 ∧ I - 1 = J) ∨ (J + 3 = I - 2 ∧ I + 2 = H - 1 ∧ H + 1 = G)) -- Ensuring that G, H, I, J are consecutive
  (h_odd_even_GHIJ : (G % 2 ≠ H % 2) ∨ (H % 2 ≠ I % 2) ∨ (I % 2 ≠ J % 2))
  (h_sum_ABC : A + B + C = 17) : A = 9 := 
sorry

end telephone_number_l51_51308


namespace grid_sum_property_l51_51261

theorem grid_sum_property
    (a b c x y z : ℕ)  -- Represent the grid elements as natural numbers
    (h1 : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (h2 : b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (h3 : c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (h4 : x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (h5 : y ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (h6 : z ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (h7 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (h8 : x ≠ y ∧ x ≠ z ∧ y ≠ z)
    (h9 : a ≠ x ∧ a ≠ y ∧ a ≠ z ∧ b ≠ x ∧ b ≠ y ∧ b ≠ z ∧ c ≠ x ∧ c ≠ y ∧ c ≠ z)
    (h10 : a + b + c = 23)
    (h11 : a + x = 14)
    (h12 : b + y = 16)
    (h13 : c + z = 17) :
    x + 2 * y + 3 * z = 49 :=
by
  sorry

end grid_sum_property_l51_51261


namespace second_set_parallel_lines_l51_51923

theorem second_set_parallel_lines (n : ℕ) :
  (5 * (n - 1)) = 280 → n = 71 :=
by
  intros h
  sorry

end second_set_parallel_lines_l51_51923


namespace infinite_pairs_exist_l51_51388

def sequence (L : ℕ) (n : ℕ) : ℕ :=
  L + ∑ i in range (n + 1), i.factorial

theorem infinite_pairs_exist (L : ℕ) (p : ℕ) (hL1 : L > 0) (hp_big : p > 20210802) (hp_prime : prime p) :
  ∃ (m k : ℕ), ∀ (i : ℕ), i < 100 → (p ^ k ∣ sequence L (m + i)) ∧ ¬ (p ^ (k + 1) ∣ sequence L (m + i)) :=
sorry

end infinite_pairs_exist_l51_51388


namespace largest_possible_d_l51_51122

noncomputable def largest_d (a b c d : ℝ) :=
  ∃ (a b c d : ℝ), 
  (a + b + c + d = 10) ∧ 
  (ab + ac + ad + bc + bd + cd = 20) ∧ 
  d = (5 + Real.sqrt 105) / 2

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
begin
  sorry
end

end largest_possible_d_l51_51122


namespace no_valid_n_that_meet_conditions_l51_51820

/-- A function to get the greatest prime factor of a number -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- Checks if the last digit of n is equal to the first digit of m -/
def last_digit_equals_first (n m : ℕ) : Prop := 
  n % 10 = (m / (10 ^ (nat.log 10 m - nat.log 10 m % 10))) % 10

theorem no_valid_n_that_meet_conditions :
  ∀ n : ℕ, n > 1 →
  (greatest_prime_factor n = nat.sqrt n) →
  (greatest_prime_factor (n + 72) = nat.sqrt (n + 72)) →
  (last_digit_equals_first n (n + 72)) →
  false :=
by {
  intros n hn hgn hn72 hdigits,
  sorry 
}

end no_valid_n_that_meet_conditions_l51_51820


namespace find_v_l51_51841

theorem find_v (a b : ℝ) (v : ℝ) (root : ℝ) : 
  a = 6 → 
  b = 31 → 
  root = (-31 - sqrt 481) / 12 → 
  ∃ v, root = (-b - sqrt(b^2 - 4 * a * v)) / (2 * a) ∧ v = 20 :=
by
  intros ha hb hroot
  use 20
  split
  · rw [ha, hb, hroot]
    ring
    sorry -- Proof here
  · refl

end find_v_l51_51841


namespace ratio_john_amount_l51_51748

theorem ratio_john_amount (total_amount : ℕ) (john_received : ℕ) (h_total : total_amount = 4800) (h_john : john_received = 1600) :
  (john_received : ℚ) / total_amount = 1 / 3 :=
by
  rw [h_total, h_john]
  norm_num
  exact sorry

end ratio_john_amount_l51_51748


namespace notebook_cost_l51_51734

-- Define the cost of notebook (n) and cost of cover (c)
variables (n c : ℝ)

-- Given conditions as definitions
def condition1 := n + c = 3.50
def condition2 := n = c + 2

-- Prove that the cost of the notebook (n) is 2.75
theorem notebook_cost (h1 : condition1 n c) (h2 : condition2 n c) : n = 2.75 := 
by
  sorry

end notebook_cost_l51_51734


namespace cube_surface_area_equals_486_l51_51738

-- Define the dimensions of the rectangular prism
def length_prism : ℕ := 9
def width_prism : ℕ := 3
def height_prism : ℕ := 27

-- Define the volume of the rectangular prism
def volume_prism : ℕ := length_prism * width_prism * height_prism

-- Define the side length of the cube with the same volume as the rectangular prism
def side_cube : ℕ := Nat.cbrt volume_prism

-- Define the surface area of the cube
def surface_area_cube : ℕ := 6 * side_cube * side_cube

-- Lean 4 proof statement
theorem cube_surface_area_equals_486 : surface_area_cube = 486 :=
by
  sorry

end cube_surface_area_equals_486_l51_51738


namespace percentage_of_number_is_40_l51_51973

theorem percentage_of_number_is_40 (N : ℝ) (P : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 35) 
  (h2 : (P/100) * N = 420) : 
  P = 40 := 
by
  sorry

end percentage_of_number_is_40_l51_51973


namespace min_value_l51_51363

def factorial_divides (n m : ℕ) : Prop :=
  ∀ p ∈ (nat.prime_divisors n), ∑ k in (range_one_inf), (n^(p:k)) ≤ ∑ k in (range_one_inf), (m^(p:k))

def k (n : ℕ) : ℕ :=
  max_m (factorial_divides (n!) (2016!))

theorem min_value : (finset.Ico 2 ℕ).min
  (λ n, n + k n) = 89 := sorry

end min_value_l51_51363


namespace find_equation_of_line_l51_51353

noncomputable def line_through_point (A : ℝ × ℝ) (k : ℝ) : ℝ × ℝ → Prop :=
  λ P, P.2 - A.2 = k * (P.1 - A.1)

noncomputable def forms_triangle_with_area (line : (ℝ × ℝ) → Prop) : Prop :=
  let x_intercept := -(2 / k) - 2 in
  let y_intercept := 2 * k + 2 in
  let triangle_area := (1 / 2) * abs(x_intercept * y_intercept) in
  triangle_area = 1

theorem find_equation_of_line :
  ∃ (k : ℝ), line_through_point (-2, 2) k ∧ forms_triangle_with_area (line_through_point (-2, 2) k) :=
  sorry

end find_equation_of_line_l51_51353


namespace poodle_barks_count_l51_51300

-- Define the conditions as hypothesis
variables (poodle_barks terrier_barks terrier_hushes : ℕ)

-- Define the conditions
def condition1 : Prop :=
  poodle_barks = 2 * terrier_barks

def condition2 : Prop :=
  terrier_hushes = terrier_barks / 2

def condition3 : Prop :=
  terrier_hushes = 6

-- The theorem we need to prove
theorem poodle_barks_count (poodle_barks terrier_barks terrier_hushes : ℕ)
  (h1 : condition1 poodle_barks terrier_barks)
  (h2 : condition2 terrier_barks terrier_hushes)
  (h3 : condition3 terrier_hushes) :
  poodle_barks = 24 :=
by
  -- Proof is not required as per instructions
  sorry

end poodle_barks_count_l51_51300


namespace ounces_per_pound_l51_51974

theorem ounces_per_pound :
  (∀ (x : ℝ),
    (1 ton = 2100 pounds) →
    (1680 packets weigh (16 + 4/x) pounds each) →
    (gunny_bag capacity = 13 tons) →
    (1680 * (16 + 4/x) = 27300) → 
    (x = 16)) :=
by 
  intros x ton_pounds packet_weight gunny_bag weight_eq
  sorry

end ounces_per_pound_l51_51974


namespace complex_polynomial_bound_l51_51385

noncomputable def polynomial (n : ℕ) (c : Fin n → ℂ) (z : ℂ) : ℂ :=
∑ i in Finset.range n, c i * z ^ (n - i)

theorem complex_polynomial_bound
  (n : ℕ)
  (c : Fin (n+1) → ℂ) :
  ∃ (z_0 : ℂ), |z_0| ≤ 1 ∧ |polynomial n c z_0| ≥ |c 0| + |c n| :=
sorry

end complex_polynomial_bound_l51_51385


namespace dolls_difference_l51_51528

-- Definitions from the conditions
def blonde_dolls : ℕ := 7
def brown_dolls : ℕ := 3.5 * blonde_dolls
def black_dolls : ℕ := (2 / 3) * brown_dolls
def red_dolls : ℕ := 1.5 * black_dolls
def grey_dolls : ℕ := (1 / 4) * red_dolls

-- Prove the main statement
theorem dolls_difference :
  let total_non_blonde := black_dolls + brown_dolls + red_dolls + grey_dolls
  total_non_blonde - blonde_dolls = 68 :=
sorry

end dolls_difference_l51_51528


namespace simplify_expression_l51_51989

theorem simplify_expression : 
  (sin 7 * 180⁻¹.pi + cos 15 * 180⁻¹.pi * sin 8 * 180⁻¹.pi) / 
  (cos 7 * 180⁻¹.pi - sin 15 * 180⁻¹.pi * sin 8 * 180⁻¹.pi) = 2 - sqrt 3 := 
by 
  sorry

end simplify_expression_l51_51989


namespace cost_of_whitewashing_is_correct_l51_51168

noncomputable def room_height : ℝ := 15
noncomputable def room_length : ℝ := 40
noncomputable def room_width : ℝ := 20

noncomputable def door1_height : ℝ := 7
noncomputable def door1_width : ℝ := 4
noncomputable def door2_height : ℝ := 5
noncomputable def door2_width : ℝ := 3

noncomputable def window1_height : ℝ := 5
noncomputable def window1_width : ℝ := 4
noncomputable def window2_height : ℝ := 4
noncomputable def window2_width : ℝ := 3
noncomputable def window3_height : ℝ := 3
noncomputable def window3_width : ℝ := 3
noncomputable def window4_height : ℝ := 4
noncomputable def window4_width : ℝ := 2.5
noncomputable def window5_height : ℝ := 6
noncomputable def window5_width : ℝ := 4

noncomputable def cost_per_sqft : ℝ := 10

theorem cost_of_whitewashing_is_correct :
  let
    wall_area := 2 * (room_height * room_length) + 2 * (room_height * room_width),
    total_wall_area := wall_area,
    door1_area := door1_height * door1_width,
    door2_area := door2_height * door2_width,
    window1_area := window1_height * window1_width,
    window2_area := window2_height * window2_width,
    window3_area := window3_height * window3_width,
    window4_area := window4_height * window4_width,
    window5_area := window5_height * window5_width,
    total_non_whitewashed_area := door1_area + door2_area + window1_area + window2_area + window3_area + window4_area + window5_area,
    whitewashed_area := total_wall_area - total_non_whitewashed_area,
    total_cost := whitewashed_area * cost_per_sqft
  in total_cost = 16820 := by
  sorry

end cost_of_whitewashing_is_correct_l51_51168


namespace a_n_is_perfect_square_l51_51961

def seqs (a b : ℕ → ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 0 ∧ ∀ n, a (n + 1) = 7 * a n + 6 * b n - 3 ∧ b (n + 1) = 8 * a n + 7 * b n - 4

theorem a_n_is_perfect_square (a b : ℕ → ℤ) (h : seqs a b) :
  ∀ n, ∃ k : ℤ, a n = k^2 :=
by
  sorry

end a_n_is_perfect_square_l51_51961


namespace power_fraction_neg_half_log_expression_l51_51330

theorem power_fraction_neg_half : (9/4)^(-1/2) = 2/3 := 
by sorry

theorem log_expression : 2^(Real.log2 3) + Real.log 1/100 = 1 :=
by sorry

end power_fraction_neg_half_log_expression_l51_51330


namespace collinear_intersections_l51_51518

variables {A B C D E F G H : Type*}
variables [parallelogram A B C D]
variables {points_on_lines : E ∈ [A, B] ∧ F ∈ [B, C] ∧ G ∈ [C, D] ∧ H ∈ [D, A]}
variables {equal_distances : AE = DG ∧ EB = GC ∧ AH = BF ∧ HD = FC}

theorem collinear_intersections (h : parallelogram A B C D)
  (hl : E ∈ [A, B] ∧ F ∈ [B, C] ∧ G ∈ [C, D] ∧ H ∈ [D, A])
  (hd : AE = DG ∧ EB = GC ∧ AH = BF ∧ HD = FC) :
  collinear ((line (B, H)) ∩ (line (D, E))),
            ((line (E, G)) ∩ (line (F, H))),
            C :=
sorry

end collinear_intersections_l51_51518


namespace total_koalas_l51_51195

namespace KangarooKoalaProof

variables {P Q R S T U V p q r s t u v : ℕ}
variables (h₁ : P = q + r + s + t + u + v)
variables (h₂ : Q = p + r + s + t + u + v)
variables (h₃ : R = p + q + s + t + u + v)
variables (h₄ : S = p + q + r + t + u + v)
variables (h₅ : T = p + q + r + s + u + v)
variables (h₆ : U = p + q + r + s + t + v)
variables (h₇ : V = p + q + r + s + t + u)
variables (h_total : P + Q + R + S + T + U + V = 2022)

theorem total_koalas : p + q + r + s + t + u + v = 337 :=
by
  sorry

end KangarooKoalaProof

end total_koalas_l51_51195


namespace intersection_complement_l51_51524

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B := {x : ℝ | x < 1}
def complement_B := {x : ℝ | x ≥ 1}

theorem intersection_complement :
  (set_A ∩ complement_B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l51_51524


namespace prove_f_three_eq_neg_three_l51_51033

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem prove_f_three_eq_neg_three (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -3 := by
  sorry

end prove_f_three_eq_neg_three_l51_51033


namespace part1_a_eq_zero_part2_range_of_a_l51_51002

noncomputable def f (x : ℝ) := abs (x + 1)
noncomputable def g (x : ℝ) (a : ℝ) := 2 * abs x + a

theorem part1_a_eq_zero :
  ∀ x, 0 < x + 1 → 0 < 2 * abs x → a = 0 →
  f x ≥ g x a ↔ (-1 / 3 : ℝ) ≤ x ∧ x ≤ 1 :=
sorry

theorem part2_range_of_a :
  ∃ x, f x ≥ g x a ↔ a ≤ 1 :=
sorry

end part1_a_eq_zero_part2_range_of_a_l51_51002


namespace sports_club_students_l51_51294

theorem sports_club_students :
  ∃ n : ℕ, n % 6 = 1 ∧ n % 9 = 3 ∧ n % 8 = 5 ∧ n = 169 :=
by
  use 169
  split
  · exact Nat.mod_eq_of_lt (Nat.zero_le 169) dec_trivial
  · split
    · exact Nat.mod_eq_of_lt (Nat.zero_le 169) dec_trivial
    · split
      · exact Nat.mod_eq_of_lt (Nat.zero_le 169) dec_trivial
      · rfl

end sports_club_students_l51_51294


namespace closest_perfect_square_to_350_l51_51666

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l51_51666


namespace smallest_d_for_inverse_domain_l51_51949

noncomputable def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

theorem smallest_d_for_inverse_domain : ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = -1 :=
by
  use -1
  constructor
  · sorry
  · rfl

end smallest_d_for_inverse_domain_l51_51949


namespace min_abs_sum_value_l51_51246

def abs_sum (x : ℝ) := |x + 3| + |x + 6| + |x + 7|

theorem min_abs_sum_value : ∃ x : ℝ, abs_sum x = 4 ∧ ∀ y : ℝ, abs_sum y ≥ abs_sum x := 
by 
  use -6
  have abs_sum_eq : abs_sum (-6) = 4 := by
    simp [abs_sum]
  -- Other conditions ensuring it is the minimum
  sorry

end min_abs_sum_value_l51_51246


namespace sum_of_valid_3_digit_numbers_l51_51639

theorem sum_of_valid_3_digit_numbers : 
  let digits := [1, 3, 4] in
  let numbers := 
    list.permutations digits |>.map (λ l, l.head! * 100 + l.nth 1 * 10 + l.nth 2) in
  (numbers.filter (λ n, (n / 100 + (n / 10) % 10 + n % 10) % 5 = 0)).sum = 0 :=
by sorry

end sum_of_valid_3_digit_numbers_l51_51639


namespace molecular_weight_of_compound_l51_51239

-- Given atomic weights in g/mol
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O  : ℝ := 15.999
def atomic_weight_H  : ℝ := 1.008

-- Given number of atoms in the compound
def num_atoms_Ca : ℕ := 1
def num_atoms_O  : ℕ := 2
def num_atoms_H  : ℕ := 2

-- Definition of the molecular weight
def molecular_weight : ℝ :=
  (num_atoms_Ca * atomic_weight_Ca) +
  (num_atoms_O * atomic_weight_O) +
  (num_atoms_H * atomic_weight_H)

-- The theorem to prove
theorem molecular_weight_of_compound : molecular_weight = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l51_51239


namespace minimum_area_of_Archimedean_triangle_l51_51162

-- Define the problem statement with necessary conditions
theorem minimum_area_of_Archimedean_triangle (p : ℝ) (hp : p > 0) :
  ∃ (ABQ_area : ℝ), ABQ_area = p^2 ∧ 
    (∀ (A B Q : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * p * A.1) ∧
      (B.2 ^ 2 = 2 * p * B.1) ∧
      (0, 0) = (p / 2, p / 2) ∧
      (Q.2 = 0) → 
      ABQ_area = p^2) :=
sorry

end minimum_area_of_Archimedean_triangle_l51_51162


namespace roots_of_polynomial_l51_51805

theorem roots_of_polynomial :
  roots (λ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6) = {1, 2, 3} := 
sorry

end roots_of_polynomial_l51_51805


namespace y_intercept_of_line_l51_51220

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end y_intercept_of_line_l51_51220


namespace rowing_distance_l51_51730

theorem rowing_distance
  (rowing_speed_in_still_water : ℝ)
  (velocity_of_current : ℝ)
  (total_time : ℝ)
  (H1 : rowing_speed_in_still_water = 5)
  (H2 : velocity_of_current = 1)
  (H3 : total_time = 1) :
  ∃ (D : ℝ), D = 2.4 := 
sorry

end rowing_distance_l51_51730


namespace cookies_in_box_l51_51011

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end cookies_in_box_l51_51011


namespace arc_length_RJP_correct_l51_51042

-- Define the conditions
def circle_center_O := true
def angle_RJP := 45 
def OR := 12

-- Define the correct answer
def arc_length_RJP := 6 * Real.pi

-- Lean 4 statement to be proven
theorem arc_length_RJP_correct : circle_center_O → angle_RJP = 45 → OR = 12 → arc_length_RJP = 6 * Real.pi :=
by
  intro h1 h2 h3
  sorry

end arc_length_RJP_correct_l51_51042


namespace initial_worth_is_30_l51_51147

-- Definitions based on conditions
def numberOfCoinsLeft := 2
def amountLeft := 12

-- Definition of the value of each gold coin based on amount left and number of coins left
def valuePerCoin : ℕ := amountLeft / numberOfCoinsLeft

-- Define the total worth of sold coins
def soldCoinsWorth (coinsSold : ℕ) : ℕ := coinsSold * valuePerCoin

-- The total initial worth of Roman's gold coins
def totalInitialWorth : ℕ := amountLeft + soldCoinsWorth 3

-- The proof goal
theorem initial_worth_is_30 : totalInitialWorth = 30 :=
by
  sorry

end initial_worth_is_30_l51_51147


namespace first_digit_one_over_137_l51_51223

-- Define the main problem in terms of first nonzero digit.
def first_nonzero_digit_right_of_decimal (n : ℕ) : ℕ :=
  let frac := 1 / (Rat.of_int n)
  let shifted_frac := frac * 10 ^ 3
  let integer_part := shifted_frac.to_nat
  integer_part % 10

theorem first_digit_one_over_137 :
  first_nonzero_digit_right_of_decimal 137 = 7 :=
by
  sorry

end first_digit_one_over_137_l51_51223


namespace rick_division_steps_l51_51985

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l51_51985


namespace tangent_line_at_origin_is_y_eq_neg_x_l51_51885

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a - 1) * x^2 - a * Real.sin x

theorem tangent_line_at_origin_is_y_eq_neg_x (a : ℝ) (h : f a = (a-1) * x^2 - a * Real.sin x ∧ ∀ x, f (-x) = -f x ∧ a = 1) :
  ∀ x y, (x, y) = (0, 0) → y = -x :=
sorry

end tangent_line_at_origin_is_y_eq_neg_x_l51_51885


namespace probability_sum_even_l51_51252

theorem probability_sum_even :
  let q := (1 : ℝ) / 14 in
  let p_even := 2 * q in
  let p_odd := q in
  let p_sum_even := (p_even ^ 2) + (p_odd ^ 2) in
  p_sum_even = 5 / 196 :=
by
  sorry

end probability_sum_even_l51_51252


namespace periodic_if_rational_smallest_period_if_rational_l51_51419

noncomputable def is_rational (x : ℝ) : Prop :=
∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

noncomputable def lcm (a b : ℝ) : ℝ :=
a * b / gcd (a.numerator, b.numerator).natur % (a.denominator * b.denominator % (gcd (a.numerator, b.numerator).natur))

noncomputable def f (α β : ℝ) (n m : ℕ) (x : ℝ) : ℝ :=
real.sin (α * x) ^ n * real.cos (β * x) ^ m

theorem periodic_if_rational {α β : ℝ} (hα : α > 0) (hβ : β > 0) (hαβ : α > β) (n m : ℕ) (hn : 0 < n) (hm : 0 < m) :
  (∃ (T : ℝ), ∀ x, f α β n m (x + T) = f α β n m x) ↔ is_rational (β / α) := by
sorry

theorem smallest_period_if_rational {α β : ℝ} (hα : α > 0) (hβ : β > 0) (hαβ : α > β) (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h_rat : is_rational (β / α)) :
  ∃ (T : ℝ), (∀ x, f α β n m (x + T) = f α β n m x) ∧ ((T = lcm (π / α) (π / β)) ∨ (T = 2 * lcm (π / α) (π / β))) := by
sorry

end periodic_if_rational_smallest_period_if_rational_l51_51419


namespace closest_perfect_square_to_350_l51_51651

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l51_51651


namespace mutually_exclusive_not_contradictory_l51_51464

theorem mutually_exclusive_not_contradictory 
  (balls : Finset String)
  (condition : ∀ (b : String), b ∈ balls → b = "red" ∨ b = "black")
  (two_taken : Finset String) :
  two_taken.card = 2 ∧ 
  (Finset.filter (λ b, b = "black") two_taken).card = 1 →
  ¬ (Finset.filter (λ b, b = "black") two_taken).card = 2 :=
by
  sorry

end mutually_exclusive_not_contradictory_l51_51464


namespace rick_group_division_l51_51981

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l51_51981


namespace jesse_room_width_l51_51072

theorem jesse_room_width (L W : ℕ) (h1 : L = 20) (h2 : L = W + 1) : W = 19 :=
by {
  sorry,
}

end jesse_room_width_l51_51072


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51826

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l51_51826


namespace sandbox_fill_cost_l51_51941

theorem sandbox_fill_cost :
  let length := 4
  let width := 3
  let height := 1.5
  let price_per_cubic_foot := 3
  let volume := length * width * height
  let cost := volume * price_per_cubic_foot
  cost = 54 :=
by
  let length := 4
  let width := 3
  let height := 1.5
  let price_per_cubic_foot := 3
  let volume := length * width * height
  let cost := volume * price_per_cubic_foot
  have h_volume : volume = 4 * 3 * 1.5 := rfl
  have h_cost : cost = volume * price_per_cubic_foot := rfl
  have h_volume_eq : 4 * 3 * 1.5 = 18 := by norm_num
  have h_cost_eq : 18 * 3 = 54 := by norm_num
  rw [h_volume_eq, h_cost_eq]
  exact eq.refl 54

end sandbox_fill_cost_l51_51941


namespace sum_real_values_x_l51_51814

theorem sum_real_values_x :
  let f (x : ℝ) : ℝ := 1 - x + x^2 - x^3 + x^4 - x^5 + ...
  has_sum_of_geometric_series (f x) (1 / (1 + x)) ->
  x = 1 - x + x^2 - x^3 + x^4 - x^5 + ... -> (|x| < 1) ->
  x = ( -1 + Real.sqrt 5 ) / 2 := by
  sorry

end sum_real_values_x_l51_51814


namespace find_f_of_one_third_l51_51445

def g (x : ℝ) : ℝ := 1 - 2 * x^2

def f (y : ℝ) (x : ℝ) (hx : x ≠ 0) : ℝ := (1 - 2 * x^2) / x^2

theorem find_f_of_one_third :
  f (1 / 3) (1 / Real.sqrt 3) (by norm_num : 1 / Real.sqrt 3 ≠ 0) = 1 :=
sorry

end find_f_of_one_third_l51_51445


namespace Freddy_is_18_l51_51844

-- Definitions based on the conditions
def Job_age : Nat := 5
def Stephanie_age : Nat := 4 * Job_age
def Freddy_age : Nat := Stephanie_age - 2

-- Statement to prove
theorem Freddy_is_18 : Freddy_age = 18 := by
  sorry

end Freddy_is_18_l51_51844


namespace min_absolute_sum_value_l51_51245

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end min_absolute_sum_value_l51_51245


namespace length_of_diagonal_PR_l51_51473

theorem length_of_diagonal_PR (PQ QR RS SP : ℝ) (angle_RSP : ℝ) (h1 : PQ = 12) (h2 : QR = 12) (h3 : RS = 20) (h4 : SP = 20) (h5 : angle_RSP = 60) :
  ∃ PR : ℝ, PR = 20 := 
begin
  -- Provided conditions about the quadrilateral PQRS.
  have h6 : RS = 20 := h3,
  have h7 : SP = 20 := h4,
  have h8 : angle_RSP = 60 := h5,
  -- We need to show that the length of diagonal PR is 20.
  use 20,
  -- Proof will be written here (skipped).
  -- sorry
end

end length_of_diagonal_PR_l51_51473


namespace solve_for_x_l51_51573

theorem solve_for_x (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) : 
  (x + 5) / (x - 3) = (x - 4) / (x + 2) → x = 1 / 7 :=
by
  sorry

end solve_for_x_l51_51573


namespace max_cos_sum_l51_51917

variable (A B C : ℝ)

def triangle_condition (A B : ℝ) := sin(A)^2 + cos(B)^2 = 1

theorem max_cos_sum (A B C : ℝ) (h : triangle_condition A B ∧ A + B + C = π ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π):
  cos(A) + cos(B) + cos(C) ≤ 2 :=
sorry

end max_cos_sum_l51_51917


namespace closest_perfect_square_to_350_l51_51670

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l51_51670


namespace inequality_proof_l51_51831

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l51_51831
