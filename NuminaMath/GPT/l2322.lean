import Mathlib

namespace Freddy_journey_time_l2322_232270

/-- Eddy and Freddy start simultaneously from city A. Eddy travels to city B, Freddy travels to city C.
    Eddy takes 3 hours from city A to city B, which is 900 km. The distance between city A and city C is
    300 km. The ratio of average speed of Eddy to Freddy is 4:1. Prove that Freddy takes 4 hours to travel. -/
theorem Freddy_journey_time (t_E : ℕ) (d_AB : ℕ) (d_AC : ℕ) (r : ℕ) (V_E V_F t_F : ℕ)
    (h1 : t_E = 3)
    (h2 : d_AB = 900)
    (h3 : d_AC = 300)
    (h4 : r = 4)
    (h5 : V_E = d_AB / t_E)
    (h6 : V_E = r * V_F)
    (h7 : t_F = d_AC / V_F)
  : t_F = 4 := 
  sorry

end Freddy_journey_time_l2322_232270


namespace opposite_of_neg_2023_l2322_232293

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l2322_232293


namespace arithmetic_sequence_properties_l2322_232239

/-- In an arithmetic sequence {a_n}, let S_n represent the sum of the first n terms, 
and it is given that S_6 < S_7 and S_7 > S_8. 
Prove that the correct statements among the given options are: 
1. The common difference d < 0 
2. S_9 < S_6 
3. S_7 is definitively the maximum value among all sums S_n. -/
theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_S6_lt_S7 : S 6 < S 7)
  (h_S7_gt_S8 : S 7 > S 8) :
  (a 7 > 0 ∧ a 8 < 0 ∧ ∃ d, ∀ n, a (n + 1) = a n + d ∧ d < 0 ∧ S 9 < S 6 ∧ ∀ n, S n ≤ S 7) :=
by
  -- Proof omitted
  sorry

end arithmetic_sequence_properties_l2322_232239


namespace find_a_8_l2322_232260

noncomputable def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∃ b : ℕ → ℤ, (∀ n : ℕ, 0 < n → b n = a (n + 1) - a n) ∧
  b 3 = -2 ∧ b 10 = 12

theorem find_a_8 (a : ℕ → ℤ) (h : sequence_a a) : a 8 = 3 :=
sorry

end find_a_8_l2322_232260


namespace probability_two_female_one_male_l2322_232222

-- Define basic conditions
def total_contestants : Nat := 7
def female_contestants : Nat := 4
def male_contestants : Nat := 3
def choose_count : Nat := 3

-- Calculate combinations (binomial coefficients)
def comb (n k : Nat) : Nat := Nat.choose n k

-- Define the probability calculation steps in Lean
def total_ways := comb total_contestants choose_count
def favorable_ways_female := comb female_contestants 2
def favorable_ways_male := comb male_contestants 1
def favorable_ways := favorable_ways_female * favorable_ways_male

theorem probability_two_female_one_male :
  (favorable_ways : ℚ) / (total_ways : ℚ) = 18 / 35 := by
  sorry

end probability_two_female_one_male_l2322_232222


namespace perimeter_triangle_formed_by_parallel_lines_l2322_232225

-- Defining the side lengths of the triangle ABC
def AB := 150
def BC := 270
def AC := 210

-- Defining the lengths of the segments formed by intersections with lines parallel to the sides of ABC
def length_lA := 65
def length_lB := 60
def length_lC := 20

-- The perimeter of the triangle formed by the intersection of the lines
theorem perimeter_triangle_formed_by_parallel_lines :
  let perimeter : ℝ := 5.71 + 20 + 83.33 + 65 + 91 + 60 + 5.71
  perimeter = 330.75 := by
  sorry

end perimeter_triangle_formed_by_parallel_lines_l2322_232225


namespace value_of_a_l2322_232269

def quadratic_vertex (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

def vertex_form (a h k x : ℤ) : ℤ :=
  a * (x - h)^2 + k

theorem value_of_a (a b c : ℤ) (h k x1 y1 x2 y2 : ℤ) (H_vert : h = 2) (H_vert_val : k = 3)
  (H_point : x1 = 1) (H_point_val : y1 = 5) (H_graph : ∀ x, quadratic_vertex a b c x = vertex_form a h k x) :
  a = 2 :=
by
  sorry

end value_of_a_l2322_232269


namespace rate_of_interest_is_12_percent_l2322_232212

variables (P r : ℝ)
variables (A5 A8 : ℝ)

-- Given conditions: 
axiom A5_condition : A5 = 9800
axiom A8_condition : A8 = 12005
axiom simple_interest_5_year : A5 = P + 5 * P * r / 100
axiom simple_interest_8_year : A8 = P + 8 * P * r / 100

-- The statement we aim to prove
theorem rate_of_interest_is_12_percent : r = 12 := 
sorry

end rate_of_interest_is_12_percent_l2322_232212


namespace train_length_l2322_232248

theorem train_length (x : ℕ)
  (h1 : ∀ (x : ℕ), (790 + x) / 33 = (860 - x) / 22) : x = 200 := by
  sorry

end train_length_l2322_232248


namespace geometric_seq_a4_l2322_232296

theorem geometric_seq_a4 (a : ℕ → ℕ) (q : ℕ) (h_q : q = 2) 
  (h_a1a3 : a 0 * a 2 = 6 * a 1) : a 3 = 24 :=
by
  -- Skipped proof
  sorry

end geometric_seq_a4_l2322_232296


namespace find_a_l2322_232243

theorem find_a (x y z a : ℝ) (h1 : ∃ k : ℝ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) 
              (h2 : x + y + z = 70) 
              (h3 : y = 15 * a - 5) : 
  a = 5 / 3 := 
by sorry

end find_a_l2322_232243


namespace find_quadratic_eq_l2322_232238

theorem find_quadratic_eq (x y : ℝ) 
  (h₁ : x + y = 10)
  (h₂ : |x - y| = 6) :
  x^2 - 10 * x + 16 = 0 :=
sorry

end find_quadratic_eq_l2322_232238


namespace exponent_relation_l2322_232220

theorem exponent_relation (a : ℝ) (m n : ℕ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m - n) = 3 := 
sorry

end exponent_relation_l2322_232220


namespace polynomial_expansion_coefficient_a8_l2322_232283

theorem polynomial_expansion_coefficient_a8 :
  let a := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  a_8 = 45 :=
by {
  sorry
}

end polynomial_expansion_coefficient_a8_l2322_232283


namespace sally_has_18_nickels_and_total_value_98_cents_l2322_232284

-- Define the initial conditions
def pennies_initial := 8
def nickels_initial := 7
def nickels_from_dad := 9
def nickels_from_mom := 2

-- Define calculations based on the initial conditions
def total_nickels := nickels_initial + nickels_from_dad + nickels_from_mom
def value_pennies := pennies_initial
def value_nickels := total_nickels * 5
def total_value := value_pennies + value_nickels

-- State the theorem to prove the correct answers
theorem sally_has_18_nickels_and_total_value_98_cents :
  total_nickels = 18 ∧ total_value = 98 := 
by {
  -- Proof goes here
  sorry
}

end sally_has_18_nickels_and_total_value_98_cents_l2322_232284


namespace quadratic_properties_l2322_232241

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l2322_232241


namespace ap_minus_aq_eq_8_l2322_232251

theorem ap_minus_aq_eq_8 (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) (p q : ℕ) 
  (h1 : ∀ n, S_n n = n^2 - 5 * n) 
  (h2 : ∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) 
  (h3 : p - q = 4) :
  a_n p - a_n q = 8 := sorry

end ap_minus_aq_eq_8_l2322_232251


namespace distance_between_parabola_vertices_l2322_232227

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_parabola_vertices :
  distance (0, 3) (0, -1) = 4 := 
by {
  -- Proof omitted here
  sorry
}

end distance_between_parabola_vertices_l2322_232227


namespace all_points_lie_on_circle_l2322_232255

theorem all_points_lie_on_circle {s : ℝ} :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := (2 * s) / (s^2 + 1)
  x^2 + y^2 = 1 :=
by
  sorry

end all_points_lie_on_circle_l2322_232255


namespace remainder_3_pow_20_div_5_l2322_232277

theorem remainder_3_pow_20_div_5 : (3 ^ 20) % 5 = 1 := 
by {
  sorry
}

end remainder_3_pow_20_div_5_l2322_232277


namespace g_triple_apply_l2322_232209

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_triple_apply : g (g (g 20)) = 1 :=
by
  sorry

end g_triple_apply_l2322_232209


namespace part1_part2_l2322_232298

namespace ClothingFactory

variables {x y m : ℝ} -- defining variables

-- The conditions
def condition1 : Prop := x + 2 * y = 5
def condition2 : Prop := 3 * x + y = 7
def condition3 : Prop := 1.8 * (100 - m) + 1.6 * m ≤ 168

-- Theorems to Prove
theorem part1 (h1 : x + 2 * y = 5) (h2 : 3 * x + y = 7) : 
  x = 1.8 ∧ y = 1.6 := 
sorry

theorem part2 (h1 : x = 1.8) (h2 : y = 1.6) (h3 : 1.8 * (100 - m) + 1.6 * m ≤ 168) : 
  m ≥ 60 := 
sorry

end ClothingFactory

end part1_part2_l2322_232298


namespace acute_angles_relation_l2322_232246

theorem acute_angles_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.sin α = (1 / 2) * Real.sin (α + β)) : α < β :=
sorry

end acute_angles_relation_l2322_232246


namespace find_investment_amount_l2322_232203

noncomputable def brokerage_fee (market_value : ℚ) : ℚ := (1 / 4 / 100) * market_value

noncomputable def actual_cost (market_value : ℚ) : ℚ := market_value + brokerage_fee market_value

noncomputable def income_per_100_face_value (interest_rate : ℚ) : ℚ := (interest_rate / 100) * 100

noncomputable def investment_amount (income : ℚ) (actual_cost_per_100 : ℚ) (income_per_100 : ℚ) : ℚ :=
  (income * actual_cost_per_100) / income_per_100

theorem find_investment_amount :
  investment_amount 756 (actual_cost 124.75) (income_per_100_face_value 10.5) = 9483.65625 :=
sorry

end find_investment_amount_l2322_232203


namespace chord_bisected_line_eq_l2322_232271

theorem chord_bisected_line_eq (x y : ℝ) (hx1 : x^2 + 4 * y^2 = 36) (hx2 : (4, 2) = ((x1 + x2) / 2, (y1 + y2) / 2)) :
  x + 2 * y - 8 = 0 :=
sorry

end chord_bisected_line_eq_l2322_232271


namespace average_cost_is_2_l2322_232254

noncomputable def total_amount_spent (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℕ :=
  apples_quantity * apples_cost + bananas_quantity * bananas_cost + oranges_quantity * oranges_cost

noncomputable def total_number_of_fruits (apples_quantity bananas_quantity oranges_quantity : ℕ) : ℕ :=
  apples_quantity + bananas_quantity + oranges_quantity

noncomputable def average_cost (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℚ :=
  (total_amount_spent apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℚ) /
  (total_number_of_fruits apples_quantity bananas_quantity oranges_quantity : ℚ)

theorem average_cost_is_2 :
  average_cost 12 4 4 2 1 3 = 2 := 
by
  sorry

end average_cost_is_2_l2322_232254


namespace S_contains_finite_but_not_infinite_arith_progressions_l2322_232268

noncomputable def S : Set ℤ := {n | ∃ k : ℕ, n = Int.floor (k * Real.pi)}

theorem S_contains_finite_but_not_infinite_arith_progressions :
  (∀ (k : ℕ), ∃ (a d : ℤ), ∀ (i : ℕ) (h : i < k), (a + i * d) ∈ S) ∧
  ¬(∃ (a d : ℤ), ∀ (n : ℕ), (a + n * d) ∈ S) :=
by
  sorry

end S_contains_finite_but_not_infinite_arith_progressions_l2322_232268


namespace reciprocal_of_negative_one_sixth_l2322_232236

theorem reciprocal_of_negative_one_sixth : ∃ x : ℚ, - (1/6) * x = 1 ∧ x = -6 :=
by
  use -6
  constructor
  . sorry -- Need to prove - (1 / 6) * (-6) = 1
  . sorry -- Need to verify x = -6

end reciprocal_of_negative_one_sixth_l2322_232236


namespace inequality_proof_l2322_232282

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) : 
  (x*y^2/z + y*z^2/x + z*x^2/y) ≥ (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l2322_232282


namespace smallest_b_not_divisible_by_5_l2322_232253

theorem smallest_b_not_divisible_by_5 :
  ∃ b : ℕ, b > 2 ∧ ¬ (5 ∣ (2 * b^3 - 1)) ∧ ∀ b' > 2, ¬ (5 ∣ (2 * (b'^3) - 1)) → b = 6 :=
by
  sorry

end smallest_b_not_divisible_by_5_l2322_232253


namespace polynomial_simplification_l2322_232202

variable (x : ℝ)

theorem polynomial_simplification :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 :=
by
  sorry

end polynomial_simplification_l2322_232202


namespace inequality_holds_range_of_expression_l2322_232272

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem inequality_holds (x : ℝ) : f x < |x - 2| + 4 ↔ x ∈ Set.Ioo (-5 : ℝ) 3 := by
  sorry

theorem range_of_expression (m n : ℝ) (h : m + n = 2) (hm : m > 0) (hn : n > 0) :
  (m^2 + 2) / m + (n^2 + 1) / n ∈ Set.Ici ((7 + 2 * Real.sqrt 2) / 2) := by
  sorry

end inequality_holds_range_of_expression_l2322_232272


namespace fettuccine_to_tortellini_ratio_l2322_232244

-- Definitions based on the problem conditions
def total_students := 800
def preferred_spaghetti := 320
def preferred_fettuccine := 200
def preferred_tortellini := 160
def preferred_penne := 120

-- Theorem to prove that the ratio is 5/4
theorem fettuccine_to_tortellini_ratio :
  (preferred_fettuccine : ℚ) / (preferred_tortellini : ℚ) = 5 / 4 :=
sorry

end fettuccine_to_tortellini_ratio_l2322_232244


namespace contradiction_example_l2322_232240

theorem contradiction_example (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
by
  sorry

end contradiction_example_l2322_232240


namespace sin_minus_cos_eq_pm_sqrt_b_l2322_232275

open Real

/-- If θ is an acute angle such that cos(2θ) = b, then sin(θ) - cos(θ) = ±√b. -/
theorem sin_minus_cos_eq_pm_sqrt_b (θ b : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hcos2θ : cos (2 * θ) = b) :
  sin θ - cos θ = sqrt b ∨ sin θ - cos θ = -sqrt b :=
sorry

end sin_minus_cos_eq_pm_sqrt_b_l2322_232275


namespace perimeter_of_rectangle_l2322_232291

theorem perimeter_of_rectangle (s : ℝ) (h1 : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end perimeter_of_rectangle_l2322_232291


namespace rosie_pie_count_l2322_232217

-- Conditions and definitions
def apples_per_pie (total_apples pies : ℕ) : ℕ := total_apples / pies

-- Theorem statement (mathematical proof problem)
theorem rosie_pie_count :
  ∀ (a p : ℕ), a = 12 → p = 3 → (36 : ℕ) / (apples_per_pie a p) = 9 :=
by
  intros a p ha hp
  rw [ha, hp]
  -- Skipping the proof
  sorry

end rosie_pie_count_l2322_232217


namespace mark_total_spending_l2322_232273

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l2322_232273


namespace number_of_green_fish_and_carp_drawn_is_6_l2322_232221

-- Definitions/parameters from the problem
def total_fish := 80 + 20 + 40 + 40 + 20
def sample_size := 20
def number_of_green_fish := 20
def number_of_carp := 40
def probability_of_being_drawn := sample_size / total_fish

-- Theorem to prove the combined number of green fish and carp drawn is 6
theorem number_of_green_fish_and_carp_drawn_is_6 :
  (number_of_green_fish + number_of_carp) * probability_of_being_drawn = 6 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_green_fish_and_carp_drawn_is_6_l2322_232221


namespace negation_proposition_l2322_232280

theorem negation_proposition (x : ℝ) : ¬ (x ≥ 1 → x^2 - 4 * x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4 * x + 2 < -1) :=
by
  sorry

end negation_proposition_l2322_232280


namespace smallest_whole_number_larger_than_perimeter_l2322_232265

-- Define the sides of the triangle
def side1 : ℕ := 7
def side2 : ℕ := 23

-- State the conditions using the triangle inequality theorem
def triangle_inequality_satisfied (s : ℕ) : Prop :=
  (side1 + side2 > s) ∧ (side1 + s > side2) ∧ (side2 + s > side1)

-- The proof statement
theorem smallest_whole_number_larger_than_perimeter
  (s : ℕ) (h : triangle_inequality_satisfied s) : 
  ∃ n : ℕ, n = 60 ∧ ∀ p : ℕ, (p > side1 + side2 + s) → (p ≥ n) :=
sorry

end smallest_whole_number_larger_than_perimeter_l2322_232265


namespace best_player_total_hits_l2322_232261

theorem best_player_total_hits
  (team_avg_hits_per_game : ℕ)
  (games_played : ℕ)
  (total_players : ℕ)
  (other_players_avg_hits_next_6_games : ℕ)
  (correct_answer : ℕ)
  (h1 : team_avg_hits_per_game = 15)
  (h2 : games_played = 5)
  (h3 : total_players = 11)
  (h4 : other_players_avg_hits_next_6_games = 6)
  (h5 : correct_answer = 25) :
  ∃ total_hits_of_best_player : ℕ,
  total_hits_of_best_player = correct_answer := by
  sorry

end best_player_total_hits_l2322_232261


namespace tetrahedron_volume_l2322_232286

noncomputable def volume_tetrahedron (A₁ A₂ : ℝ) (θ : ℝ) (d : ℝ) : ℝ :=
  (A₁ * A₂ * Real.sin θ) / (3 * d)

theorem tetrahedron_volume:
  ∀ (PQ PQR PQS : ℝ) (θ : ℝ),
  PQ = 5 → PQR = 20 → PQS = 18 → θ = Real.pi / 4 → volume_tetrahedron PQR PQS θ PQ = 24 * Real.sqrt 2 :=
by
  intros
  unfold volume_tetrahedron
  sorry

end tetrahedron_volume_l2322_232286


namespace count_inverses_modulo_11_l2322_232232

theorem count_inverses_modulo_11 : (Finset.filter (λ a => Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
  by
  sorry

end count_inverses_modulo_11_l2322_232232


namespace common_ratio_geometric_sequence_l2322_232228

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

theorem common_ratio_geometric_sequence
  (a3_eq : a 3 = 2 * S 2 + 1)
  (a4_eq : a 4 = 2 * S 3 + 1)
  (geometric_seq : ∀ n, a (n+1) = a 1 * (q ^ n))
  (h₀ : a 1 ≠ 0)
  (h₁ : q ≠ 0) :
  q = 3 :=
sorry

end common_ratio_geometric_sequence_l2322_232228


namespace inequality_a4b_to_abcd_l2322_232262

theorem inequality_a4b_to_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end inequality_a4b_to_abcd_l2322_232262


namespace correct_answer_is_B_l2322_232231

-- Define what it means to be a quadratic equation in one variable
def is_quadratic_in_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x ^ 2 + b * x + c = 0)

-- Conditions:
def eqA (x : ℝ) : Prop := 2 * x + 1 = 0
def eqB (x : ℝ) : Prop := x ^ 2 + 1 = 0
def eqC (x y : ℝ) : Prop := y ^ 2 + x = 1
def eqD (x : ℝ) : Prop := 1 / x + x ^ 2 = 1

-- Theorem statement: Prove which equation is a quadratic equation in one variable
theorem correct_answer_is_B : is_quadratic_in_one_variable eqB :=
sorry  -- Proof is not required as per the instructions

end correct_answer_is_B_l2322_232231


namespace max_reflections_l2322_232290

theorem max_reflections (P Q R M : Type) (angle : ℝ) :
  0 < angle ∧ angle ≤ 30 ∧ (∃ n : ℕ, 10 * n = angle) →
  ∃ n : ℕ, n ≤ 3 :=
by
  sorry

end max_reflections_l2322_232290


namespace second_train_speed_l2322_232299

theorem second_train_speed (len1 len2 dist t : ℕ) (h1 : len1 = 100) (h2 : len2 = 150) (h3 : dist = 50) (h4 : t = 60) : 
  (len1 + len2 + dist) / t = 5 := 
  by
  -- Definitions from conditions
  have h_len1 : len1 = 100 := h1
  have h_len2 : len2 = 150 := h2
  have h_dist : dist = 50 := h3
  have h_time : t = 60 := h4
  
  -- Proof deferred
  sorry

end second_train_speed_l2322_232299


namespace negation_equiv_l2322_232266

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l2322_232266


namespace project_B_days_l2322_232292

theorem project_B_days (B : ℕ) : 
  (1 / 20 + 1 / B) * 10 + (1 / B) * 5 = 1 -> B = 30 :=
by
  sorry

end project_B_days_l2322_232292


namespace find_h_from_quadratic_l2322_232213

theorem find_h_from_quadratic (
  p q r : ℝ) (h₁ : ∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) :
  ∀ m k h, (∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - h)^2 + k) → h = 5 :=
by
  intros m k h h₂
  sorry

end find_h_from_quadratic_l2322_232213


namespace gcd_612_468_l2322_232294

theorem gcd_612_468 : gcd 612 468 = 36 :=
by
  sorry

end gcd_612_468_l2322_232294


namespace transformed_parabola_correct_l2322_232226

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end transformed_parabola_correct_l2322_232226


namespace ratio_sqrt5_over_5_l2322_232288

noncomputable def radius_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
a / b

theorem ratio_sqrt5_over_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) :
  radius_ratio a b h = 1 / Real.sqrt 5 := 
sorry

end ratio_sqrt5_over_5_l2322_232288


namespace marked_price_correct_l2322_232245

noncomputable def marked_price (original_price discount_percent purchase_price profit_percent final_price_percent : ℝ) := 
  (purchase_price * (1 + profit_percent)) / final_price_percent

theorem marked_price_correct
  (original_price : ℝ)
  (discount_percent : ℝ)
  (profit_percent : ℝ)
  (final_price_percent : ℝ)
  (purchase_price : ℝ := original_price * (1 - discount_percent))
  (expected_marked_price : ℝ) :
  original_price = 40 →
  discount_percent = 0.15 →
  profit_percent = 0.25 →
  final_price_percent = 0.90 →
  expected_marked_price = 47.20 →
  marked_price original_price discount_percent purchase_price profit_percent final_price_percent = expected_marked_price := 
by
  intros
  sorry

end marked_price_correct_l2322_232245


namespace used_more_brown_sugar_l2322_232207

-- Define the amounts of sugar used
def brown_sugar : ℝ := 0.62
def white_sugar : ℝ := 0.25

-- Define the statement to prove
theorem used_more_brown_sugar : brown_sugar - white_sugar = 0.37 :=
by
  sorry

end used_more_brown_sugar_l2322_232207


namespace shortest_distance_between_stations_l2322_232249

/-- 
Given two vehicles A and B shuttling between two locations,
with Vehicle A stopping every 0.5 kilometers and Vehicle B stopping every 0.8 kilometers,
prove that the shortest distance between two stations where Vehicles A and B do not stop at the same place is 0.1 kilometers.
-/
theorem shortest_distance_between_stations :
  ∀ (dA dB : ℝ), (dA = 0.5) → (dB = 0.8) → ∃ δ : ℝ, (δ = 0.1) ∧ (∀ n m : ℕ, dA * n ≠ dB * m → abs ((dA * n) - (dB * m)) = δ) :=
by
  intros dA dB hA hB
  use 0.1
  sorry

end shortest_distance_between_stations_l2322_232249


namespace equations_solutions_l2322_232233

-- Definition and statement for Equation 1
noncomputable def equation1_solution1 : ℝ :=
  (-3 + Real.sqrt 17) / 4

noncomputable def equation1_solution2 : ℝ :=
  (-3 - Real.sqrt 17) / 4

-- Definition and statement for Equation 2
def equation2_solution : ℝ :=
  -6

-- Theorem proving the solutions to the given equations
theorem equations_solutions :
  (∃ x : ℝ, 2 * x^2 + 3 * x = 1 ∧ (x = equation1_solution1 ∨ x = equation1_solution2)) ∧
  (∃ x : ℝ, 3 / (x - 2) = 5 / (2 - x) - 1 ∧ x = equation2_solution) :=
by
  sorry

end equations_solutions_l2322_232233


namespace more_girls_than_boys_l2322_232297

theorem more_girls_than_boys
  (b g : ℕ)
  (ratio : b / g = 3 / 4)
  (total : b + g = 42) :
  g - b = 6 :=
sorry

end more_girls_than_boys_l2322_232297


namespace max_mass_of_grain_l2322_232201

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end max_mass_of_grain_l2322_232201


namespace factoring_sum_of_coefficients_l2322_232250

theorem factoring_sum_of_coefficients 
  (a b c d e f g h j k : ℤ)
  (h1 : 64 * x^6 - 729 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) :
  a + b + c + d + e + f + g + h + j + k = 30 :=
sorry

end factoring_sum_of_coefficients_l2322_232250


namespace circumcircle_radius_of_triangle_l2322_232230

theorem circumcircle_radius_of_triangle
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (AB BC : ℝ)
  (angle_ABC : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 4)
  (h_angle_ABC : angle_ABC = 120) :
  ∃ (R : ℝ), R = 4 := by
  sorry

end circumcircle_radius_of_triangle_l2322_232230


namespace first_tap_fill_time_l2322_232205

theorem first_tap_fill_time (T : ℚ) :
  (∀ (second_tap_empty_time : ℚ), second_tap_empty_time = 8) →
  (∀ (combined_fill_time : ℚ), combined_fill_time = 40 / 3) →
  (1/T - 1/8 = 3/40) →
  T = 5 :=
by
  intros h1 h2 h3
  sorry

end first_tap_fill_time_l2322_232205


namespace find_u5_l2322_232278

theorem find_u5 
  (u : ℕ → ℝ)
  (h_rec : ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n)
  (h_u3 : u 3 = 9)
  (h_u6 : u 6 = 243) : 
  u 5 = 69 :=
sorry

end find_u5_l2322_232278


namespace range_of_k_if_f_monotonically_increasing_l2322_232235

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k_if_f_monotonically_increasing :
  (∀ (x : ℝ), 1 < x → 0 ≤ (k - 1 / x)) → k ∈ Set.Ici (1: ℝ) :=
by
  intro hyp
  have : ∀ (x : ℝ), 1 < x → 0 ≤ k - 1 / x := hyp
  sorry

end range_of_k_if_f_monotonically_increasing_l2322_232235


namespace initial_number_of_rabbits_is_50_l2322_232256

-- Initial number of weasels
def initial_weasels := 100

-- Each fox catches 4 weasels and 2 rabbits per week
def weasels_caught_per_fox_per_week := 4
def rabbits_caught_per_fox_per_week := 2

-- There are 3 foxes
def num_foxes := 3

-- After 3 weeks, 96 weasels and rabbits are left
def weasels_and_rabbits_left := 96
def weeks := 3

theorem initial_number_of_rabbits_is_50 :
  (initial_weasels + (initial_weasels + weasels_and_rabbits_left)) - initial_weasels = 50 :=
by
  sorry

end initial_number_of_rabbits_is_50_l2322_232256


namespace number_of_intersections_l2322_232267

noncomputable def y1 (x: ℝ) : ℝ := (x - 1) ^ 4
noncomputable def y2 (x: ℝ) : ℝ := 2 ^ (abs x) - 2

theorem number_of_intersections : (∃ x₁ x₂ x₃ x₄ : ℝ, y1 x₁ = y2 x₁ ∧ y1 x₂ = y2 x₂ ∧ y1 x₃ = y2 x₃ ∧ y1 x₄ = y2 x₄ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
sorry

end number_of_intersections_l2322_232267


namespace foxes_wolves_bears_num_l2322_232218

-- Definitions and theorem statement
def num_hunters := 45
def num_rabbits := 2008
def rabbits_per_fox := 59
def rabbits_per_wolf := 41
def rabbits_per_bear := 40

theorem foxes_wolves_bears_num (x y z : ℤ) : 
  x + y + z = num_hunters → 
  rabbits_per_wolf * x + rabbits_per_fox * y + rabbits_per_bear * z = num_rabbits → 
  x = 18 ∧ y = 10 ∧ z = 17 :=
by 
  intro h1 h2 
  sorry

end foxes_wolves_bears_num_l2322_232218


namespace algebra_books_cannot_be_determined_uniquely_l2322_232274

theorem algebra_books_cannot_be_determined_uniquely (A H S M E : ℕ) (pos_A : A > 0) (pos_H : H > 0) (pos_S : S > 0) 
  (pos_M : M > 0) (pos_E : E > 0) (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ S ≠ M ∧ S ≠ E ∧ M ≠ E) 
  (cond1: S < A) (cond2: M > H) (cond3: A + 2 * H = S + 2 * M) : 
  E = 0 :=
sorry

end algebra_books_cannot_be_determined_uniquely_l2322_232274


namespace largest_possible_value_l2322_232200

noncomputable def largest_log_expression (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) : ℝ := 
  Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b

theorem largest_possible_value (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) (h3 : a = b) : 
  largest_log_expression a b h1 h2 = 0 :=
by
  sorry

end largest_possible_value_l2322_232200


namespace selling_price_before_clearance_l2322_232214

-- Define the cost price (CP)
def CP : ℝ := 100

-- Define the gain percent before the clearance sale
def gain_percent_before : ℝ := 0.35

-- Define the discount percent during the clearance sale
def discount_percent : ℝ := 0.10

-- Define the gain percent during the clearance sale
def gain_percent_sale : ℝ := 0.215

-- Calculate the selling price before the clearance sale (SP_before)
def SP_before : ℝ := CP * (1 + gain_percent_before)

-- Calculate the selling price during the clearance sale (SP_sale)
def SP_sale : ℝ := SP_before * (1 - discount_percent)

-- Proof statement in Lean 4
theorem selling_price_before_clearance : SP_before = 135 :=
by
  -- Place to fill in the proof later
  sorry

end selling_price_before_clearance_l2322_232214


namespace quadratic_distinct_zeros_range_l2322_232219

theorem quadratic_distinct_zeros_range (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - (k+1)*x1 + k + 4 = 0 ∧ x2^2 - (k+1)*x2 + k + 4 = 0)
  ↔ k ∈ (Set.Iio (-3) ∪ Set.Ioi 5) :=
by
  sorry

end quadratic_distinct_zeros_range_l2322_232219


namespace sum_of_reciprocals_eq_two_l2322_232263

theorem sum_of_reciprocals_eq_two (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : 1 / x + 1 / y = 2 := by
  sorry

end sum_of_reciprocals_eq_two_l2322_232263


namespace revenue_for_recent_quarter_l2322_232279

noncomputable def previous_year_revenue : ℝ := 85.0
noncomputable def percentage_fall : ℝ := 43.529411764705884
noncomputable def recent_quarter_revenue : ℝ := previous_year_revenue - (previous_year_revenue * (percentage_fall / 100))

theorem revenue_for_recent_quarter : recent_quarter_revenue = 48.0 := 
by 
  sorry -- Proof is skipped

end revenue_for_recent_quarter_l2322_232279


namespace exists_rhombus_with_given_side_and_diag_sum_l2322_232224

-- Define the context of the problem
variables (a s : ℝ)

-- Necessary definitions for a rhombus
structure Rhombus (side diag_sum : ℝ) :=
  (side_length : ℝ)
  (diag_sum : ℝ)
  (d1 d2 : ℝ)
  (side_length_eq : side_length = side)
  (diag_sum_eq : d1 + d2 = diag_sum)
  (a_squared : 2 * (side_length)^2 = d1^2 + d2^2)

-- The proof problem
theorem exists_rhombus_with_given_side_and_diag_sum (a s : ℝ) : 
  ∃ (r : Rhombus a (2*s)), r.side_length = a ∧ r.diag_sum = 2 * s :=
by
  sorry

end exists_rhombus_with_given_side_and_diag_sum_l2322_232224


namespace negation_abs_val_statement_l2322_232208

theorem negation_abs_val_statement (x : ℝ) :
  ¬ (|x| ≤ 3 ∨ |x| > 5) ↔ (|x| > 3 ∧ |x| ≤ 5) :=
by sorry

end negation_abs_val_statement_l2322_232208


namespace Kenny_jumping_jacks_wednesday_l2322_232229

variable (Sunday Monday Tuesday Wednesday Thursday Friday Saturday : ℕ)
variable (LastWeekTotal : ℕ := 324)
variable (SundayJumpingJacks : ℕ := 34)
variable (MondayJumpingJacks : ℕ := 20)
variable (TuesdayJumpingJacks : ℕ := 0)
variable (SomeDayJumpingJacks : ℕ := 64)
variable (FridayJumpingJacks : ℕ := 23)
variable (SaturdayJumpingJacks : ℕ := 61)

def Kenny_jumping_jacks_this_week (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ) : ℕ :=
  SundayJumpingJacks + MondayJumpingJacks + TuesdayJumpingJacks + WednesdayJumpingJacks + ThursdayJumpingJacks + FridayJumpingJacks + SaturdayJumpingJacks

def Kenny_jumping_jacks_to_beat (weekTotal : ℕ) : ℕ :=
  LastWeekTotal + 1

theorem Kenny_jumping_jacks_wednesday : 
  ∃ (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ), 
  Kenny_jumping_jacks_this_week WednesdayJumpingJacks ThursdayJumpingJacks = LastWeekTotal + 1 ∧ 
  (WednesdayJumpingJacks = 59 ∧ ThursdayJumpingJacks = 64) ∨ (WednesdayJumpingJacks = 64 ∧ ThursdayJumpingJacks = 59) :=
by
  sorry

end Kenny_jumping_jacks_wednesday_l2322_232229


namespace volume_of_remaining_sphere_after_hole_l2322_232242

noncomputable def volume_of_remaining_sphere (R : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * R^3
  let volume_cylinder := (4 / 3) * Real.pi * (R / 2)^3
  volume_sphere - volume_cylinder

theorem volume_of_remaining_sphere_after_hole : 
  volume_of_remaining_sphere 5 = (500 * Real.pi) / 3 :=
by
  sorry

end volume_of_remaining_sphere_after_hole_l2322_232242


namespace find_base_of_triangle_l2322_232285

-- Given data
def perimeter : ℝ := 20 -- The perimeter of the triangle
def tangent_segment : ℝ := 2.4 -- The segment of the tangent to the inscribed circle contained between the sides

-- Define the problem and expected result
theorem find_base_of_triangle (a b c : ℝ) (P : a + b + c = perimeter)
  (tangent_parallel_base : ℝ := tangent_segment):
  a = 4 ∨ a = 6 :=
sorry

end find_base_of_triangle_l2322_232285


namespace sum_of_reflected_coordinates_l2322_232204

noncomputable def sum_of_coordinates (C D : ℝ × ℝ) : ℝ :=
  C.1 + C.2 + D.1 + D.2

theorem sum_of_reflected_coordinates (y : ℝ) :
  let C := (3, y)
  let D := (3, -y)
  sum_of_coordinates C D = 6 :=
by
  sorry

end sum_of_reflected_coordinates_l2322_232204


namespace rectangle_area_l2322_232287

theorem rectangle_area {H W : ℝ} (h_height : H = 24) (ratio : W / H = 0.875) :
  H * W = 504 :=
by 
  sorry

end rectangle_area_l2322_232287


namespace evaluate_expression_l2322_232216

-- Define the base value
def base := 3000

-- Define the exponential expression
def exp_value := base ^ base

-- Prove that base * exp_value equals base ^ (1 + base)
theorem evaluate_expression : base * exp_value = base ^ (1 + base) := by
  sorry

end evaluate_expression_l2322_232216


namespace annuity_payment_l2322_232264

variable (P : ℝ) (A : ℝ) (i : ℝ) (n1 n2 : ℕ)

-- Condition: Principal amount
axiom principal_amount : P = 24000

-- Condition: Annual installment for the first 5 years
axiom annual_installment : A = 1500 

-- Condition: Annual interest rate
axiom interest_rate : i = 0.045 

-- Condition: Years before equal annual installments
axiom years_before_installment : n1 = 5 

-- Condition: Years for repayment after the first 5 years
axiom repayment_years : n2 = 7 

-- Remaining debt after n1 years
noncomputable def remaining_debt_after_n1 : ℝ :=
  P * (1 + i) ^ n1 - A * ((1 + i) ^ n1 - 1) / i

-- Annual payment for n2 years to repay the remaining debt
noncomputable def annual_payment (D : ℝ) : ℝ :=
  D * (1 + i) ^ n2 / (((1 + i) ^ n2 - 1) / i)

axiom remaining_debt_amount : remaining_debt_after_n1 P A i n1 = 21698.685 

theorem annuity_payment : annual_payment (remaining_debt_after_n1 P A i n1) = 3582 := by
  sorry

end annuity_payment_l2322_232264


namespace jane_paints_correct_area_l2322_232215

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end jane_paints_correct_area_l2322_232215


namespace original_price_of_article_l2322_232237

theorem original_price_of_article (new_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) 
  (h_reduction : reduction_percentage = 56/100) (h_new_price : new_price = 4400) :
  original_price = 10000 :=
sorry

end original_price_of_article_l2322_232237


namespace arithmetic_sequence_a5_zero_l2322_232247

variable {a : ℕ → ℤ}
variable {d : ℤ}

theorem arithmetic_sequence_a5_zero 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : d ≠ 0)
  (h3 : a 3 + a 9 = a 10 - a 8) : 
  a 5 = 0 := sorry

end arithmetic_sequence_a5_zero_l2322_232247


namespace initial_milk_quantity_l2322_232258

theorem initial_milk_quantity (A B C D : ℝ) (hA : A > 0)
  (hB : B = 0.55 * A)
  (hC : C = 1.125 * A)
  (hD : D = 0.8 * A)
  (hTransferBC : B + 150 = C - 150 + 100)
  (hTransferDC : C - 50 = D - 100)
  (hEqual : B + 150 = D - 100) : 
  A = 1000 :=
by sorry

end initial_milk_quantity_l2322_232258


namespace remainder_div_101_l2322_232259

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l2322_232259


namespace speed_of_faster_train_approx_l2322_232276

noncomputable def speed_of_slower_train_kmph : ℝ := 40
noncomputable def speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * 1000 / 3600
noncomputable def distance_train1 : ℝ := 250
noncomputable def distance_train2 : ℝ := 500
noncomputable def total_distance : ℝ := distance_train1 + distance_train2
noncomputable def crossing_time : ℝ := 26.99784017278618
noncomputable def relative_speed_train_crossing : ℝ := total_distance / crossing_time
noncomputable def speed_of_faster_train_mps : ℝ := relative_speed_train_crossing - speed_of_slower_train_mps
noncomputable def speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * 3600 / 1000

theorem speed_of_faster_train_approx : abs (speed_of_faster_train_kmph - 60.0152) < 0.001 :=
by 
  sorry

end speed_of_faster_train_approx_l2322_232276


namespace inscribed_square_length_l2322_232281

-- Define the right triangle PQR with given sides
variables (PQ QR PR : ℕ)
variables (h s : ℚ)

-- Given conditions
def right_triangle_PQR : Prop := PQ = 5 ∧ QR = 12 ∧ PR = 13
def altitude_Q_to_PR : Prop := h = (PQ * QR) / PR
def side_length_of_square : Prop := s = h * (1 - h / PR)

theorem inscribed_square_length (PQ QR PR h s : ℚ) 
    (right_triangle_PQR : PQ = 5 ∧ QR = 12 ∧ PR = 13)
    (altitude_Q_to_PR : h = (PQ * QR) / PR) 
    (side_length_of_square : s = h * (1 - h / PR)) 
    : s = 6540 / 2207 := by
  -- we skip the proof here as requested
  sorry

end inscribed_square_length_l2322_232281


namespace rectangle_sides_l2322_232211

theorem rectangle_sides (x y : ℕ) :
  (2 * x + 2 * y = x * y) →
  x > 0 →
  y > 0 →
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) :=
by
  sorry

end rectangle_sides_l2322_232211


namespace ratio_QP_l2322_232234

noncomputable def P : ℚ := 11 / 6
noncomputable def Q : ℚ := 5 / 2

theorem ratio_QP : Q / P = 15 / 11 := by 
  sorry

end ratio_QP_l2322_232234


namespace find_width_l2322_232295

theorem find_width (A : ℕ) (hA : A ≥ 120) (w : ℕ) (l : ℕ) (hl : l = w + 20) (h_area : w * l = A) : w = 4 :=
by sorry

end find_width_l2322_232295


namespace simplify_root_product_l2322_232223

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end simplify_root_product_l2322_232223


namespace boat_speed_determination_l2322_232206

theorem boat_speed_determination :
  ∃ x : ℝ, 
    (∀ u d : ℝ, u = 170 / (x + 6) ∧ d = 170 / (x - 6))
    ∧ (u + d = 68)
    ∧ (x = 9) := 
by
  sorry

end boat_speed_determination_l2322_232206


namespace positive_sum_inequality_l2322_232252

theorem positive_sum_inequality 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) ≥ (ab + bc + ca)^3 := 
by 
  sorry

end positive_sum_inequality_l2322_232252


namespace jerry_can_escape_l2322_232289

theorem jerry_can_escape (d : ℝ) (V_J V_T : ℝ) (h1 : (1 / 5) < d) (h2 : d < (1 / 4)) (h3 : V_T = 4 * V_J) :
  (4 * d) / V_J < 1 / (2 * V_J) :=
by
  sorry

end jerry_can_escape_l2322_232289


namespace legacy_total_earnings_l2322_232257

def floors := 4
def rooms_per_floor := 10
def hours_per_room := 6
def hourly_rate := 15
def total_rooms := floors * rooms_per_floor
def total_hours := total_rooms * hours_per_room
def total_earnings := total_hours * hourly_rate

theorem legacy_total_earnings :
  total_earnings = 3600 :=
by
  -- Proof to be filled in
  sorry

end legacy_total_earnings_l2322_232257


namespace number_of_roosters_l2322_232210

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end number_of_roosters_l2322_232210
