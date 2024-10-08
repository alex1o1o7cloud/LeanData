import Mathlib

namespace volume_of_tetrahedron_l223_223358

-- Define the setup of tetrahedron D-ABC
def tetrahedron_volume (V : ℝ) : Prop :=
  ∃ (DA : ℝ) (A B C D : ℝ × ℝ × ℝ), 
  A = (0, 0, 0) ∧ 
  B = (2, 0, 0) ∧ 
  C = (1, Real.sqrt 3, 0) ∧
  D = (1, Real.sqrt 3/3, DA) ∧
  DA = 2 * Real.sqrt 3 ∧
  ∃ tan_dihedral : ℝ, tan_dihedral = 2 ∧
  V = 2

-- The statement to prove the volume is indeed 2 given the conditions.
theorem volume_of_tetrahedron : ∃ V, tetrahedron_volume V :=
by 
  sorry

end volume_of_tetrahedron_l223_223358


namespace number_of_crayons_given_to_friends_l223_223636

def totalCrayonsLostOrGivenAway := 229
def crayonsLost := 16
def crayonsGivenToFriends := totalCrayonsLostOrGivenAway - crayonsLost

theorem number_of_crayons_given_to_friends :
  crayonsGivenToFriends = 213 :=
by
  sorry

end number_of_crayons_given_to_friends_l223_223636


namespace intersection_point_l223_223714

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem intersection_point :
  ∃ a : ℝ, g a = a ∧ a = -3 :=
by
  sorry

end intersection_point_l223_223714


namespace impossible_pawn_placement_l223_223632

theorem impossible_pawn_placement :
  ¬(∃ a b c : ℕ, a + b + c = 50 ∧ 
  ∀ (x y z : ℕ), 2 * a ≤ x ∧ x ≤ 2 * b ∧ 2 * b ≤ y ∧ y ≤ 2 * c ∧ 2 * c ≤ z ∧ z ≤ 2 * a) := sorry

end impossible_pawn_placement_l223_223632


namespace silverware_probability_l223_223381

-- Define the contents of the drawer
def forks := 6
def spoons := 6
def knives := 6

-- Total number of pieces of silverware
def total_silverware := forks + spoons + knives

-- Combinations formula for choosing r items out of n
def choose (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Total number of ways to choose 3 pieces out of 18
def total_ways := choose total_silverware 3

-- Number of ways to choose 1 fork, 1 spoon, and 1 knife
def specific_ways := forks * spoons * knives

-- Calculated probability
def probability := specific_ways / total_ways

theorem silverware_probability : probability = 9 / 34 := 
  sorry
 
end silverware_probability_l223_223381


namespace shipping_cost_per_unit_l223_223124

noncomputable def fixed_monthly_costs : ℝ := 16500
noncomputable def production_cost_per_component : ℝ := 80
noncomputable def production_quantity : ℝ := 150
noncomputable def selling_price_per_component : ℝ := 193.33

theorem shipping_cost_per_unit :
  ∀ (S : ℝ), (production_quantity * production_cost_per_component + production_quantity * S + fixed_monthly_costs) ≤ (production_quantity * selling_price_per_component) → S ≤ 3.33 :=
by
  intro S
  sorry

end shipping_cost_per_unit_l223_223124


namespace sequence_contains_prime_l223_223809

-- Define the conditions for being square-free and relatively prime
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Statement of the problem
theorem sequence_contains_prime :
  ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ 14 → 2 ≤ a i ∧ a i ≤ 1995 ∧ is_square_free (a i)) →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 14 → are_relatively_prime (a i) (a j)) →
  ∃ i, 1 ≤ i ∧ i ≤ 14 ∧ is_prime (a i) :=
sorry

end sequence_contains_prime_l223_223809


namespace value_of_fraction_pow_l223_223003

theorem value_of_fraction_pow (a b : ℤ) 
  (h1 : ∀ x, (x^2 + (a + 1)*x + a*b) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 4) : 
  ((1 / 2 : ℚ) ^ (a + 2*b) = 4) :=
sorry

end value_of_fraction_pow_l223_223003


namespace distance_with_wind_l223_223280

-- Define constants
def distance_against_wind : ℝ := 320
def speed_wind : ℝ := 20
def speed_plane_still_air : ℝ := 180

-- Calculate effective speeds
def effective_speed_with_wind : ℝ := speed_plane_still_air + speed_wind
def effective_speed_against_wind : ℝ := speed_plane_still_air - speed_wind

-- Define the proof statement
theorem distance_with_wind :
  ∃ (D : ℝ), (D / effective_speed_with_wind) = (distance_against_wind / effective_speed_against_wind) ∧ D = 400 :=
by
  sorry

end distance_with_wind_l223_223280


namespace find_a_l223_223682

theorem find_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a * b - a - b = 4) : a = 6 :=
sorry

end find_a_l223_223682


namespace symmetric_circle_equation_l223_223820

noncomputable def equation_of_symmetric_circle (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2 * x - 6 * y + 9 = 0) ∧ (2 * x + y + 5 = 0)

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), 
    equation_of_symmetric_circle x y → 
    ∃ a b : ℝ, ((x - a)^2 + (y - b)^2 = 1) ∧ (a + 7 = 0) ∧ (b + 1 = 0) :=
sorry

end symmetric_circle_equation_l223_223820


namespace expand_polynomial_l223_223979

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l223_223979


namespace divide_5000_among_x_and_y_l223_223629

theorem divide_5000_among_x_and_y (total_amount : ℝ) (ratio_x : ℝ) (ratio_y : ℝ) (parts : ℝ) :
  total_amount = 5000 → ratio_x = 2 → ratio_y = 8 → parts = ratio_x + ratio_y → 
  (total_amount / parts) * ratio_x = 1000 := 
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end divide_5000_among_x_and_y_l223_223629


namespace find_f_neg1_l223_223203

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f) (h_f1 : f 1 = 2)

-- Theorem stating the necessary proof
theorem find_f_neg1 : f (-1) = -2 :=
by
  sorry

end find_f_neg1_l223_223203


namespace same_answer_l223_223063

structure Person :=
(name : String)
(tellsTruth : Bool)

def Fedya : Person :=
{ name := "Fedya",
  tellsTruth := true }

def Vadim : Person :=
{ name := "Vadim",
  tellsTruth := false }

def question (p : Person) (q : String) : Bool :=
if p.tellsTruth then q = p.name else q ≠ p.name

theorem same_answer (q : String) :
  (question Fedya q = question Vadim q) :=
sorry

end same_answer_l223_223063


namespace cartons_in_load_l223_223679

theorem cartons_in_load 
  (crate_weight : ℕ)
  (carton_weight : ℕ)
  (num_crates : ℕ)
  (total_load_weight : ℕ)
  (h1 : crate_weight = 4)
  (h2 : carton_weight = 3)
  (h3 : num_crates = 12)
  (h4 : total_load_weight = 96) :
  ∃ C : ℕ, num_crates * crate_weight + C * carton_weight = total_load_weight ∧ C = 16 := 
by 
  sorry

end cartons_in_load_l223_223679


namespace question_l223_223997

def N : ℕ := 100101102 -- N should be defined properly but is simplified here for illustration.

theorem question (k : ℕ) (h : N = 100101102502499500) : (3^3 ∣ N) ∧ ¬(3^4 ∣ N) :=
sorry

end question_l223_223997


namespace max_ratio_l223_223496

theorem max_ratio {a b c d : ℝ} 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0) 
  (h2 : a^2 + b^2 + c^2 + d^2 = (a + b + c + d)^2 / 3) : 
  ∃ x, x = (7 + 2 * Real.sqrt 6) / 5 ∧ x = (a + c) / (b + d) :=
by
  sorry

end max_ratio_l223_223496


namespace line_points_satisfy_equation_l223_223949

theorem line_points_satisfy_equation (x_2 y_3 : ℝ) 
  (h_slope : ∃ k : ℝ, k = 2) 
  (h_P1 : ∃ P1 : ℝ × ℝ, P1 = (3, 5)) 
  (h_P2 : ∃ P2 : ℝ × ℝ, P2 = (x_2, 7)) 
  (h_P3 : ∃ P3 : ℝ × ℝ, P3 = (-1, y_3)) 
  (h_line : ∀ (x y : ℝ), y - 5 = 2 * (x - 3) ↔ 2 * x - y - 1 = 0) :
  x_2 = 4 ∧ y_3 = -3 :=
sorry

end line_points_satisfy_equation_l223_223949


namespace function_does_not_have_property_P_l223_223170

-- Definition of property P
def hasPropertyP (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f ((x1 + x2) / 2) = (f x1 + f x2) / 2

-- Function in question
def f (x : ℝ) : ℝ :=
  x^2

-- Statement that function f does not have property P
theorem function_does_not_have_property_P : ¬hasPropertyP f :=
  sorry

end function_does_not_have_property_P_l223_223170


namespace central_angle_of_regular_hexagon_l223_223673

theorem central_angle_of_regular_hexagon :
  ∀ (total_angle : ℝ) (sides : ℝ), total_angle = 360 → sides = 6 → total_angle / sides = 60 :=
by
  intros total_angle sides h_total_angle h_sides
  rw [h_total_angle, h_sides]
  norm_num

end central_angle_of_regular_hexagon_l223_223673


namespace car_total_distance_l223_223827

theorem car_total_distance (h1 h2 h3 : ℕ) :
  h1 = 180 → h2 = 160 → h3 = 220 → h1 + h2 + h3 = 560 :=
by
  intros h1_eq h2_eq h3_eq
  sorry

end car_total_distance_l223_223827


namespace train_speed_problem_l223_223914

theorem train_speed_problem (l1 l2 : ℝ) (v2 : ℝ) (t : ℝ) (v1 : ℝ) :
  l1 = 120 → l2 = 280 → v2 = 30 → t = 19.99840012798976 →
  0.4 / (t / 3600) = v1 + v2 → v1 = 42 :=
by
  intros hl1 hl2 hv2 ht hrel
  rw [hl1, hl2, hv2, ht] at *
  sorry

end train_speed_problem_l223_223914


namespace find_third_root_l223_223573

theorem find_third_root (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3 * b) * 1^2 + (b - 4 * a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (6 - a) = 0)
  : ∃ c : ℚ, c = 7 / 13 :=
sorry

end find_third_root_l223_223573


namespace equal_roots_polynomial_l223_223647

open ComplexConjugate

theorem equal_roots_polynomial (k : ℚ) :
  (3 : ℚ) * x^2 - k * x + 2 * x + (12 : ℚ) = 0 → 
  (b : ℚ) ^ 2 - 4 * (3 : ℚ) * (12 : ℚ) = 0 ↔ k = -10 ∨ k = 14 :=
by
  sorry

end equal_roots_polynomial_l223_223647


namespace sequence_ratio_l223_223128

theorem sequence_ratio :
  ∀ {a : ℕ → ℝ} (h₁ : a 1 = 1/2) (h₂ : ∀ n, a n = (a (n + 1)) * (a (n + 1))),
  (a 200 / a 300) = (301 / 201) :=
by
  sorry

end sequence_ratio_l223_223128


namespace range_of_m_l223_223558

-- Define points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation as a predicate
def line_l (m : ℝ) (p : ℝ × ℝ) : Prop := p.1 + m * p.2 + m = 0

-- Define the condition for the line intersecting the segment AB
def intersects_segment_AB (m : ℝ) : Prop :=
  let P : ℝ × ℝ := (0, -1)
  let k_PA := (P.2 - A.2) / (P.1 - A.1) -- Slope of PA
  let k_PB := (P.2 - B.2) / (P.1 - B.1) -- Slope of PB
  (k_PA <= -1 / m) ∧ (-1 / m <= k_PB)

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), intersects_segment_AB m → (1/2 ≤ m ∧ m ≤ 2) :=
by sorry

end range_of_m_l223_223558


namespace find_a_minus_b_l223_223689

-- Given definitions for conditions
variables (a b : ℤ)

-- Given conditions as hypotheses
def condition1 := a + 2 * b = 5
def condition2 := a * b = -12

theorem find_a_minus_b (h1 : condition1 a b) (h2 : condition2 a b) : a - b = -7 :=
sorry

end find_a_minus_b_l223_223689


namespace systematic_sampling_result_l223_223301

-- Define the set of bags numbered from 1 to 30
def bags : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the systematic sampling function
def systematic_sampling (n k interval : ℕ) : List ℕ :=
  List.range k |> List.map (λ i => n + i * interval)

-- Specific parameters for the problem
def number_of_bags := 30
def bags_drawn := 6
def interval := 5
def expected_samples := [2, 7, 12, 17, 22, 27]

-- Statement of the theorem
theorem systematic_sampling_result : 
  systematic_sampling 2 bags_drawn interval = expected_samples :=
by
  sorry

end systematic_sampling_result_l223_223301


namespace sum_of_coefficients_l223_223298

-- Define the polynomial P(x)
def P (x : ℤ) : ℤ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

-- State the theorem we intend to prove
theorem sum_of_coefficients : P 1 = 2019 :=
by
  -- Proof omitted
  sorry

end sum_of_coefficients_l223_223298


namespace glen_pop_l223_223288

/-- In the village of Glen, the total population can be formulated as 21h + 6c
given the relationships between people, horses, sheep, cows, and ducks.
We need to prove that 96 cannot be expressed in the form 21h + 6c for
non-negative integers h and c. -/
theorem glen_pop (h c : ℕ) : 21 * h + 6 * c ≠ 96 :=
by
sorry

end glen_pop_l223_223288


namespace correct_option_B_l223_223985

-- Define decimal representation of the numbers
def dec_13 : ℕ := 13
def dec_25 : ℕ := 25
def dec_11 : ℕ := 11
def dec_10 : ℕ := 10

-- Define binary representation of the numbers
def bin_1101 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 1*4 + 0*2 + 1*1 = 13
def bin_10110 : ℕ := 2^(4) + 2^(2) + 2^(1)  -- 1*16 + 0*8 + 1*4 + 1*2 + 0*1 = 22
def bin_1011 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 0*4 + 1*2 + 1*1 = 11
def bin_10 : ℕ := 2^(1)  -- 1*2 + 0*1 = 2

theorem correct_option_B : (dec_13 = bin_1101) := by
  -- Proof is skipped
  sorry

end correct_option_B_l223_223985


namespace nonagon_angles_l223_223056

/-- Determine the angles of the nonagon given specified conditions -/
theorem nonagon_angles (a : ℝ) (x : ℝ) 
  (h_angle_eq : ∀ (AIH BCD HGF : ℝ), AIH = x → BCD = x → HGF = x)
  (h_internal_sum : 7 * 180 = 1260)
  (h_tessellation : x + x + x + (360 - x) + (360 - x) + (360 - x) = 1080) :
  True := sorry

end nonagon_angles_l223_223056


namespace swimming_lane_length_l223_223752

-- Conditions
def num_round_trips : ℕ := 3
def total_distance : ℕ := 600

-- Hypothesis that 1 round trip is equivalent to 2 lengths of the lane
def lengths_per_round_trip : ℕ := 2

-- Statement to prove
theorem swimming_lane_length :
  (total_distance / (num_round_trips * lengths_per_round_trip) = 100) := by
  sorry

end swimming_lane_length_l223_223752


namespace value_range_of_f_l223_223197

def f (x : ℝ) := 2 * x ^ 2 + 4 * x + 1

theorem value_range_of_f :
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 4 → (∃ y ∈ Set.Icc (-1 : ℝ) 49, f x = y) :=
by sorry

end value_range_of_f_l223_223197


namespace complement_intersection_l223_223311

open Set

theorem complement_intersection {x : ℝ} :
  (x ∉ {x | -2 ≤ x ∧ x ≤ 2}) ∧ (x < 1) ↔ (x < -2) := 
by
  sorry

end complement_intersection_l223_223311


namespace correct_answer_l223_223619

def M : Set ℤ := {x | |x| < 5}

theorem correct_answer : {0} ⊆ M := by
  sorry

end correct_answer_l223_223619


namespace paper_clips_in_2_cases_l223_223915

variable (c b : ℕ)

theorem paper_clips_in_2_cases : 2 * (c * b) * 600 = (2 * c * b * 600) := by
  sorry

end paper_clips_in_2_cases_l223_223915


namespace probability_same_color_l223_223036

-- Define the total combinations function
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- The given values from the problem
def whiteBalls := 2
def blackBalls := 3
def totalBalls := whiteBalls + blackBalls
def drawnBalls := 2

-- Calculate combinations
def comb_white_2 := comb whiteBalls drawnBalls
def comb_black_2 := comb blackBalls drawnBalls
def comb_total_2 := comb totalBalls drawnBalls

-- The correct answer given in the solution
def correct_probability := 2 / 5

-- Statement for the proof in Lean
theorem probability_same_color : (comb_white_2 + comb_black_2) / comb_total_2 = correct_probability := by
  sorry

end probability_same_color_l223_223036


namespace operation_addition_x_l223_223208

theorem operation_addition_x (x : ℕ) (h : 106 + 106 + x + x = 19872) : x = 9830 :=
sorry

end operation_addition_x_l223_223208


namespace Tim_has_52_photos_l223_223444

theorem Tim_has_52_photos (T : ℕ) (Paul : ℕ) (Total : ℕ) (Tom : ℕ) : 
  (Paul = T + 10) → (Total = Tom + T + Paul) → (Tom = 38) → (Total = 152) → T = 52 :=
by
  intros hPaul hTotal hTom hTotalVal
  -- The proof would go here
  sorry

end Tim_has_52_photos_l223_223444


namespace solve_eq_l223_223458

theorem solve_eq {x : ℝ} (h : x + 2 * Real.sqrt x - 8 = 0) : x = 4 :=
by
  sorry

end solve_eq_l223_223458


namespace smallest_sum_of_factors_of_8_l223_223841

theorem smallest_sum_of_factors_of_8! :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a * b * c * d = Nat.factorial 8 ∧ a + b + c + d = 102 :=
sorry

end smallest_sum_of_factors_of_8_l223_223841


namespace range_of_a_l223_223662

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (1 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
sorry

end range_of_a_l223_223662


namespace inequality_solution_l223_223044

theorem inequality_solution (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x > 1) ∧ 
  (0 < a ∧ a < 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x < 1) :=
by {
  sorry
}

end inequality_solution_l223_223044


namespace equidistant_points_l223_223840

theorem equidistant_points (r d1 d2 : ℝ) (d1_eq : d1 = r) (d2_eq : d2 = 6) : 
  ∃ p : ℝ, p = 2 := 
sorry

end equidistant_points_l223_223840


namespace simplify_expr_l223_223236

-- Define the condition
def y : ℕ := 77

-- Define the expression and the expected result
def expr := (7 * y + 77) / 77

-- The theorem statement
theorem simplify_expr : expr = 8 :=
by
  sorry

end simplify_expr_l223_223236


namespace trigonometric_expression_l223_223989

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end trigonometric_expression_l223_223989


namespace distance_of_each_race_l223_223497

theorem distance_of_each_race (d : ℝ) : 
  (∃ (d : ℝ), 
    let lake_speed := 3 
    let ocean_speed := 2.5 
    let num_races := 10 
    let total_time := 11
    let num_lake_races := num_races / 2
    let num_ocean_races := num_races / 2
    (num_lake_races * (d / lake_speed) + num_ocean_races * (d / ocean_speed) = total_time)) →
  d = 3 :=
sorry

end distance_of_each_race_l223_223497


namespace length_of_first_platform_l223_223902

theorem length_of_first_platform 
  (train_length : ℕ) (first_time : ℕ) (second_platform_length : ℕ) (second_time : ℕ)
  (speed_first : ℕ) (speed_second : ℕ) :
  train_length = 230 → 
  first_time = 15 → 
  second_platform_length = 250 → 
  second_time = 20 → 
  speed_first = (train_length + L) / first_time →
  speed_second = (train_length + second_platform_length) / second_time →
  speed_first = speed_second →
  (L : ℕ) = 130 :=
by
  sorry

end length_of_first_platform_l223_223902


namespace probability_two_faces_no_faces_l223_223235

theorem probability_two_faces_no_faces :
  let side_length := 5
  let total_cubes := side_length ^ 3
  let painted_faces := 2 * (side_length ^ 2)
  let two_painted_faces := 16
  let no_painted_faces := total_cubes - painted_faces + two_painted_faces
  (two_painted_faces = 16) →
  (no_painted_faces = 91) →
  -- Total ways to choose 2 cubes from 125
  let total_ways := (total_cubes * (total_cubes - 1)) / 2
  -- Ways to choose 1 cube with 2 painted faces and 1 with no painted faces
  let successful_ways := two_painted_faces * no_painted_faces
  (successful_ways = 1456) →
  (total_ways = 7750) →
  -- The desired probability
  let probability := successful_ways / (total_ways : ℝ)
  probability = 4 / 21 :=
by
  intros side_length total_cubes painted_faces two_painted_faces no_painted_faces h1 h2 total_ways successful_ways h3 h4 probability
  sorry

end probability_two_faces_no_faces_l223_223235


namespace evaluation_result_l223_223500

noncomputable def evaluate_expression : ℝ :=
  let a := 210
  let b := 206
  let numerator := 980 ^ 2
  let denominator := a^2 - b^2
  numerator / denominator

theorem evaluation_result : evaluate_expression = 577.5 := 
  sorry  -- Placeholder for the proof

end evaluation_result_l223_223500


namespace Z_3_5_value_l223_223807

def Z (a b : ℕ) : ℕ :=
  b + 12 * a - a ^ 2

theorem Z_3_5_value : Z 3 5 = 32 := by
  sorry

end Z_3_5_value_l223_223807


namespace repeating_decimal_to_fraction_l223_223256

theorem repeating_decimal_to_fraction 
  (h : ∀ {x : ℝ}, (0.01 : ℝ) = 1 / 99 → x = 1.06 → (0.06 : ℝ) = 6 * 1 / 99): 
  1.06 = 35 / 33 :=
by sorry

end repeating_decimal_to_fraction_l223_223256


namespace cos_of_double_angles_l223_223483

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l223_223483


namespace condition_equiv_l223_223711

theorem condition_equiv (p q : Prop) : (¬ (p ∧ q) ∧ (p ∨ q)) ↔ ((p ∨ q) ∧ (¬ p ↔ q)) :=
  sorry

end condition_equiv_l223_223711


namespace arithmetic_mean_12_24_36_48_l223_223642

theorem arithmetic_mean_12_24_36_48 : (12 + 24 + 36 + 48) / 4 = 30 :=
by
  sorry

end arithmetic_mean_12_24_36_48_l223_223642


namespace min_value_of_expression_l223_223919

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end min_value_of_expression_l223_223919


namespace race_runners_l223_223068

theorem race_runners (n : ℕ) (h1 : 5 * 8 + (n - 5) * 10 = 70) : n = 8 :=
sorry

end race_runners_l223_223068


namespace largest_int_less_100_remainder_5_l223_223104

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end largest_int_less_100_remainder_5_l223_223104


namespace bowl_weight_after_refill_l223_223456

-- Define the problem conditions
def empty_bowl_weight : ℕ := 420
def day1_consumption : ℕ := 53
def day2_consumption : ℕ := 76
def day3_consumption : ℕ := 65
def day4_consumption : ℕ := 14

-- Define the total consumption over 4 days
def total_consumption : ℕ :=
  day1_consumption + day2_consumption + day3_consumption + day4_consumption

-- Define the final weight of the bowl after refilling
def final_bowl_weight : ℕ :=
  empty_bowl_weight + total_consumption

-- Statement to prove
theorem bowl_weight_after_refill : final_bowl_weight = 628 := by
  sorry

end bowl_weight_after_refill_l223_223456


namespace factorization_correctness_l223_223677

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end factorization_correctness_l223_223677


namespace find_n_l223_223142

theorem find_n (n : ℕ) (d : ℕ) (h_pos : n > 0) (h_digit : d < 10) (h_equiv : n * 999 = 810 * (100 * d + 25)) : n = 750 :=
  sorry

end find_n_l223_223142


namespace find_m_l223_223601

theorem find_m (m : ℝ) (x : ℝ) (y : ℝ) (h_eq_parabola : y = m * x^2)
  (h_directrix : y = 1 / 8) : m = -2 :=
by
  sorry

end find_m_l223_223601


namespace min_value_4x2_plus_y2_l223_223020

theorem min_value_4x2_plus_y2 {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) : 
  4 * x^2 + y^2 ≥ 18 := by
  sorry

end min_value_4x2_plus_y2_l223_223020


namespace sum_of_turning_angles_l223_223119

variable (radius distance : ℝ) (C : ℝ)

theorem sum_of_turning_angles (H1 : radius = 10) (H2 : distance = 30000) (H3 : C = 2 * radius * Real.pi) :
  (distance / C) * 2 * Real.pi ≥ 2998 :=
by
  sorry

end sum_of_turning_angles_l223_223119


namespace scholarship_awards_l223_223391

theorem scholarship_awards (x : ℕ) (h : 10000 * x + 2000 * (28 - x) = 80000) : x = 3 ∧ (28 - x) = 25 :=
by {
  sorry
}

end scholarship_awards_l223_223391


namespace number_of_solutions_l223_223525

theorem number_of_solutions (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  2 - 4 * Real.sin (2 * θ) + 3 * Real.cos (4 * θ) = 0 → 
  ∃ s : Fin 9, s.val = 8 :=
by
  sorry

end number_of_solutions_l223_223525


namespace find_t_l223_223006

theorem find_t (s t : ℤ) (h1 : 9 * s + 5 * t = 108) (h2 : s = t - 2) : t = 9 :=
sorry

end find_t_l223_223006


namespace seventh_term_geometric_seq_l223_223993

theorem seventh_term_geometric_seq (a r : ℝ) (h_pos: 0 < r) (h_fifth: a * r^4 = 16) (h_ninth: a * r^8 = 4) : a * r^6 = 8 := by
  sorry

end seventh_term_geometric_seq_l223_223993


namespace simplify_expression_l223_223285

noncomputable def simplify_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3

theorem simplify_expression :
  (Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3) / (Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3) = simplify_expr :=
by
  sorry

end simplify_expression_l223_223285


namespace evaluate_expression_l223_223772

theorem evaluate_expression :
  (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 :=
by sorry

end evaluate_expression_l223_223772


namespace num_five_digit_palindromes_with_even_middle_l223_223224

theorem num_five_digit_palindromes_with_even_middle :
  (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ∃ c', c = 2 * c' ∧ 0 ≤ c' ∧ c' ≤ 4 ∧ 10000 * a + 1000 * b + 100 * c + 10 * b + a ≤ 99999) →
  9 * 10 * 5 = 450 :=
by
  sorry

end num_five_digit_palindromes_with_even_middle_l223_223224


namespace part1_part2_l223_223011

def y (x : ℝ) : ℝ := -x^2 + 8*x - 7

-- Part (1) Lean statement
theorem part1 : ∀ x : ℝ, x < 4 → y x < y (x + 1) := sorry

-- Part (2) Lean statement
theorem part2 : ∀ x : ℝ, (x < 1 ∨ x > 7) → y x < 0 := sorry

end part1_part2_l223_223011


namespace find_base_l223_223419

theorem find_base (b : ℝ) (h : 2.134 * b^3 < 21000) : b ≤ 21 :=
by
  have h1 : b < (21000 / 2.134) ^ (1 / 3) := sorry
  have h2 : (21000 / 2.134) ^ (1 / 3) < 21.5 := sorry
  have h3 : b ≤ 21 := sorry
  exact h3

end find_base_l223_223419


namespace line_through_point_with_equal_intercepts_l223_223637

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, -2)

-- Define the property of having equal absolute intercepts
def has_equal_absolute_intercepts (a b : ℝ) : Prop :=
  |a| = |b|

-- Define the general form of a line equation
def line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main theorem: Any line passing through (3, -2) with equal absolute intercepts satisfies the given equations
theorem line_through_point_with_equal_intercepts (a b : ℝ) :
  has_equal_absolute_intercepts a b
  → line_eq 2 3 0 3 (-2)
  ∨ line_eq 1 1 (-1) 3 (-2)
  ∨ line_eq 1 (-1) (-5) 3 (-2) :=
by {
  sorry
}

end line_through_point_with_equal_intercepts_l223_223637


namespace integral_even_odd_l223_223027

open Real

theorem integral_even_odd (a : ℝ) :
  (∫ x in -a..a, x^2 + sin x) = 18 → a = 3 :=
by
  intros h
  -- We'll skip the proof
  sorry

end integral_even_odd_l223_223027


namespace wheel_rpm_is_approximately_5000_23_l223_223329

noncomputable def bus_wheel_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_min := (speed * 1000 * 100) / 60
  speed_cm_per_min / circumference

-- Conditions
def radius := 35
def speed := 66

-- Question (to be proved)
theorem wheel_rpm_is_approximately_5000_23 : 
  abs (bus_wheel_rpm radius speed - 5000.23) < 0.01 :=
by
  sorry

end wheel_rpm_is_approximately_5000_23_l223_223329


namespace original_pencils_example_l223_223883

-- Statement of the problem conditions
def original_pencils (total_pencils : ℕ) (added_pencils : ℕ) : ℕ :=
  total_pencils - added_pencils

-- Theorem we need to prove
theorem original_pencils_example : original_pencils 5 3 = 2 := 
by
  -- Proof
  sorry

end original_pencils_example_l223_223883


namespace ball_distribution_l223_223937

theorem ball_distribution :
  ∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ x1 x2 x3, f x1 x2 x3 → x1 + x2 + x3 = 10 ∧ x1 ≥ 1 ∧ x2 ≥ 2 ∧ x3 ≥ 3) ∧
    (∃ (count : ℕ), (count = 15) ∧ (∀ x1 x2 x3, f x1 x2 x3 → count = 15)) :=
sorry

end ball_distribution_l223_223937


namespace probability_of_2_reds_before_3_greens_l223_223623

theorem probability_of_2_reds_before_3_greens :
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  (favorable_arrangements / total_arrangements : ℚ) = (2 / 7 : ℚ) :=
by
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  have fraction_computation :
    (favorable_arrangements : ℚ) / (total_arrangements : ℚ) = (2 / 7 : ℚ)
  {
    sorry
  }
  exact fraction_computation

end probability_of_2_reds_before_3_greens_l223_223623


namespace negation_of_universal_proposition_l223_223693

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 3 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2 * x + 3 < 0 := 
sorry

end negation_of_universal_proposition_l223_223693


namespace deposit_amount_l223_223779

theorem deposit_amount (P : ℝ) (deposit remaining : ℝ) (h1 : deposit = 0.1 * P) (h2 : remaining = P - deposit) (h3 : remaining = 1350) : 
  deposit = 150 := 
by
  sorry

end deposit_amount_l223_223779


namespace tan_alpha_sqrt3_l223_223463

theorem tan_alpha_sqrt3 (α : ℝ) (h : Real.sin (α + 20 * Real.pi / 180) = Real.cos (α + 10 * Real.pi / 180) + Real.cos (α - 10 * Real.pi / 180)) :
  Real.tan α = Real.sqrt 3 := 
  sorry

end tan_alpha_sqrt3_l223_223463


namespace marble_problem_l223_223988

theorem marble_problem (R B : ℝ) 
  (h1 : R + B = 6000) 
  (h2 : (R + B) - |R - B| = 4800) 
  (h3 : B > R) : B = 3600 :=
sorry

end marble_problem_l223_223988


namespace clock_hands_form_right_angle_at_180_over_11_l223_223152

-- Define the angular speeds as constants
def ω_hour : ℝ := 0.5  -- Degrees per minute
def ω_minute : ℝ := 6  -- Degrees per minute

-- Function to calculate the angle of the hour hand after t minutes
def angle_hour (t : ℝ) : ℝ := ω_hour * t

-- Function to calculate the angle of the minute hand after t minutes
def angle_minute (t : ℝ) : ℝ := ω_minute * t

-- Theorem: Prove the two hands form a right angle at the given time
theorem clock_hands_form_right_angle_at_180_over_11 : 
  ∃ t : ℝ, (6 * t - 0.5 * t = 90) ∧ t = 180 / 11 :=
by 
  -- This is where the proof would go, but we skip it with sorry
  sorry

end clock_hands_form_right_angle_at_180_over_11_l223_223152


namespace arithmetic_sequence_m_value_l223_223057

theorem arithmetic_sequence_m_value (m : ℝ) (h : 2 + 6 = 2 * m) : m = 4 :=
by sorry

end arithmetic_sequence_m_value_l223_223057


namespace bobby_jumps_per_second_as_adult_l223_223738

-- Define the conditions as variables
def child_jumps_per_minute : ℕ := 30
def additional_jumps_as_adult : ℕ := 30

-- Theorem statement
theorem bobby_jumps_per_second_as_adult :
  (child_jumps_per_minute + additional_jumps_as_adult) / 60 = 1 :=
by
  -- placeholder for the proof
  sorry

end bobby_jumps_per_second_as_adult_l223_223738


namespace study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l223_223798

theorem study_video_game_inversely_proportional_1 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : s = 6) : v = 2 :=
by
  sorry

theorem study_video_game_inversely_proportional_2 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : v = 6) : s = 2 :=
by
  sorry

end study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l223_223798


namespace find_z_l223_223133

def M (z : ℂ) : Set ℂ := {1, 2, z * Complex.I}
def N : Set ℂ := {3, 4}

theorem find_z (z : ℂ) (h : M z ∩ N = {4}) : z = -4 * Complex.I := by
  sorry

end find_z_l223_223133


namespace third_of_ten_l223_223721

theorem third_of_ten : (1/3 : ℝ) * 10 = 8 / 3 :=
by
  have h : (1/4 : ℝ) * 20 = 4 := by sorry
  sorry

end third_of_ten_l223_223721


namespace planes_parallel_from_plane_l223_223819

-- Define the relationship functions
def parallel (P Q : Plane) : Prop := sorry -- Define parallelism predicate
def perpendicular (l : Line) (P : Plane) : Prop := sorry -- Define perpendicularity predicate

-- Declare the planes α, β, and γ
variable (α β γ : Plane)

-- Main theorem statement
theorem planes_parallel_from_plane (h1 : parallel γ α) (h2 : parallel γ β) : parallel α β := 
sorry

end planes_parallel_from_plane_l223_223819


namespace quadratic_inequality_solution_l223_223476

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} :=
sorry

end quadratic_inequality_solution_l223_223476


namespace sample_size_l223_223846

theorem sample_size {n : ℕ} (h_ratio : 2+3+4 = 9)
  (h_units_A : ∃ a : ℕ, a = 16)
  (h_stratified_sampling : ∃ B C : ℕ, B = 24 ∧ C = 32)
  : n = 16 + 24 + 32 := by
  sorry

end sample_size_l223_223846


namespace smallest_possible_value_of_N_l223_223416

theorem smallest_possible_value_of_N :
  ∀ (a b c d e f : ℕ), a + b + c + d + e + f = 3015 → (0 < a) → (0 < b) → (0 < c) → (0 < d) → (0 < e) → (0 < f) →
  (∃ N : ℕ, N = max (max (max (max (a + b) (b + c)) (c + d)) (d + e)) (e + f) ∧ N = 604) := 
by
  sorry

end smallest_possible_value_of_N_l223_223416


namespace max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l223_223553

-- Define the pentagon and various other polygons
inductive PolygonType
| pentagon
| triangle
| quadrilateral

-- Define a function that calculates the maximum number of intersections
def max_intersections (K L : PolygonType) : ℕ :=
  match K, L with
  | PolygonType.pentagon, PolygonType.triangle => 10
  | PolygonType.pentagon, PolygonType.quadrilateral => 16
  | _, _ => 0  -- We only care about the cases specified in our problem

-- Theorem a): When L is a triangle, the intersections should be 10
theorem max_intersections_pentagon_triangle : max_intersections PolygonType.pentagon PolygonType.triangle = 10 :=
  by 
  -- provide proof here, but currently it is skipped with sorry
  sorry

-- Theorem b): When L is a quadrilateral, the intersections should be 16
theorem max_intersections_pentagon_quadrilateral : max_intersections PolygonType.pentagon PolygonType.quadrilateral = 16 :=
  by
  -- provide proof here, but currently it is skipped with sorry
  sorry

end max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l223_223553


namespace thomas_total_training_hours_l223_223593

-- Define the conditions from the problem statement.
def training_hours_first_15_days : ℕ := 15 * 5
def training_hours_next_15_days : ℕ := (15 - 3) * (4 + 3)
def training_hours_next_12_days : ℕ := (12 - 2) * (4 + 3)

-- Prove that the total training hours equals 229.
theorem thomas_total_training_hours : 
  training_hours_first_15_days + training_hours_next_15_days + training_hours_next_12_days = 229 :=
by
  -- conditions as defined
  let t1 := 15 * 5
  let t2 := (15 - 3) * (4 + 3)
  let t3 := (12 - 2) * (4 + 3)
  show t1 + t2 + t3 = 229
  sorry

end thomas_total_training_hours_l223_223593


namespace candy_cost_l223_223076

theorem candy_cost (x : ℝ) : 
  (15 * x + 30 * 5) / (15 + 30) = 6 -> x = 8 :=
by sorry

end candy_cost_l223_223076


namespace avg_age_boys_class_l223_223563

-- Definitions based on conditions
def avg_age_students : ℝ := 15.8
def avg_age_girls : ℝ := 15.4
def ratio_boys_girls : ℝ := 1.0000000000000044

-- Using the given conditions to define the average age of boys
theorem avg_age_boys_class (B G : ℕ) (A_b : ℝ) 
  (h1 : avg_age_students = (B * A_b + G * avg_age_girls) / (B + G)) 
  (h2 : B = ratio_boys_girls * G) : 
  A_b = 16.2 :=
  sorry

end avg_age_boys_class_l223_223563


namespace combined_weight_of_Leo_and_Kendra_l223_223557

theorem combined_weight_of_Leo_and_Kendra :
  ∃ (K : ℝ), (92 + K = 160) ∧ (102 = 1.5 * K) :=
by
  sorry

end combined_weight_of_Leo_and_Kendra_l223_223557


namespace min_contribution_l223_223967

theorem min_contribution (x : ℝ) (h1 : 0 < x) (h2 : 10 * x = 20) (h3 : ∀ p, p ≠ 1 → p ≠ 2 → p ≠ 3 → p ≠ 4 → p ≠ 5 → p ≠ 6 → p ≠ 7 → p ≠ 8 → p ≠ 9 → p ≠ 10 → p ≤ 11) : 
  x = 2 := sorry

end min_contribution_l223_223967


namespace percentage_for_x_plus_y_l223_223493

theorem percentage_for_x_plus_y (x y : Real) (P : Real) 
  (h1 : 0.60 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := 
by 
  sorry

end percentage_for_x_plus_y_l223_223493


namespace compute_a4_b4_c4_l223_223800

theorem compute_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 8) (h2 : ab + ac + bc = 13) (h3 : abc = -22) : a^4 + b^4 + c^4 = 1378 :=
by
  sorry

end compute_a4_b4_c4_l223_223800


namespace complex_ratio_of_cubes_l223_223085

theorem complex_ratio_of_cubes (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 10) (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 :=
by
  sorry

end complex_ratio_of_cubes_l223_223085


namespace max_squares_at_a1_bksq_l223_223868

noncomputable def maximizePerfectSquares (a b : ℕ) : Prop := 
a ≠ b ∧ 
(∃ k : ℕ, k ≠ 1 ∧ b = k^2) ∧ 
a = 1

theorem max_squares_at_a1_bksq (a b : ℕ) : maximizePerfectSquares a b := 
by 
  sorry

end max_squares_at_a1_bksq_l223_223868


namespace point_D_is_on_y_axis_l223_223290

def is_on_y_axis (p : ℝ × ℝ) : Prop := p.fst = 0

def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (2, 1)
def point_D : ℝ × ℝ := (0, -3)

theorem point_D_is_on_y_axis : is_on_y_axis point_D :=
by
  sorry

end point_D_is_on_y_axis_l223_223290


namespace inequality_proof_l223_223817

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ (3 / 2) :=
sorry

end inequality_proof_l223_223817


namespace find_first_spill_l223_223766

def bottle_capacity : ℕ := 20
def refill_count : ℕ := 3
def days : ℕ := 7
def total_water_drunk : ℕ := 407
def second_spill : ℕ := 8

theorem find_first_spill :
  let total_without_spill := bottle_capacity * refill_count * days
  let total_spilled := total_without_spill - total_water_drunk
  let first_spill := total_spilled - second_spill
  first_spill = 5 :=
by
  -- Proof goes here.
  sorry

end find_first_spill_l223_223766


namespace integer_divisibility_l223_223906

open Nat

theorem integer_divisibility (n : ℕ) (h1 : ∃ m : ℕ, 2^n - 2 = n * m) : ∃ k : ℕ, 2^((2^n) - 1) - 2 = (2^n - 1) * k := by
  sorry

end integer_divisibility_l223_223906


namespace software_package_cost_l223_223073

theorem software_package_cost 
  (devices : ℕ) 
  (cost_first : ℕ) 
  (devices_covered_first : ℕ) 
  (devices_covered_second : ℕ) 
  (savings : ℕ)
  (total_cost_first : ℕ := (devices / devices_covered_first) * cost_first)
  (total_cost_second : ℕ := total_cost_first - savings)
  (num_packages_second : ℕ := devices / devices_covered_second)
  (cost_second : ℕ := total_cost_second / num_packages_second) :
  devices = 50 ∧ cost_first = 40 ∧ devices_covered_first = 5 ∧ devices_covered_second = 10 ∧ savings = 100 →
  cost_second = 60 := 
by
  sorry

end software_package_cost_l223_223073


namespace determinant_not_sufficient_nor_necessary_l223_223244

-- Definitions of the initial conditions
variables {a1 b1 a2 b2 c1 c2 : ℝ}

-- Conditions given: neither line coefficients form the zero vector
axiom non_zero_1 : a1^2 + b1^2 ≠ 0
axiom non_zero_2 : a2^2 + b2^2 ≠ 0

-- The matrix determinant condition and line parallelism
def determinant_condition (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 ≠ 0

def lines_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 = 0 ∧ a1 * c2 ≠ a2 * c1

-- Proof problem statement: proving equivalence
theorem determinant_not_sufficient_nor_necessary :
  ¬ (∀ a1 b1 a2 b2 c1 c2, (determinant_condition a1 b1 a2 b2 → lines_parallel a1 b1 c1 a2 b2 c2) ∧
                          (lines_parallel a1 b1 c1 a2 b2 c2 → determinant_condition a1 b1 a2 b2)) :=
sorry

end determinant_not_sufficient_nor_necessary_l223_223244


namespace acute_triangle_on_perpendicular_lines_l223_223490

theorem acute_triangle_on_perpendicular_lines :
  ∀ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) →
  ∃ (x y z : ℝ), (x^2 = (b^2 + c^2 - a^2) / 2) ∧ (y^2 = (a^2 + c^2 - b^2) / 2) ∧ (z^2 = (a^2 + b^2 - c^2) / 2) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) :=
by
  sorry

end acute_triangle_on_perpendicular_lines_l223_223490


namespace max_principals_in_10_years_l223_223728

theorem max_principals_in_10_years (h : ∀ p : ℕ, 4 * p ≤ 10) :
  ∃ n : ℕ, n ≤ 3 ∧ n = 3 :=
sorry

end max_principals_in_10_years_l223_223728


namespace find_q_l223_223024

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 17) (h2 : 6 * p + 5 * q = 20) : q = 2 / 11 :=
by
  sorry

end find_q_l223_223024


namespace correct_fraction_statement_l223_223413

theorem correct_fraction_statement (x : ℝ) :
  (∀ a b : ℝ, (-a) / (-b) = a / b) ∧
  (¬ (∀ a : ℝ, a / 0 = 0)) ∧
  (∀ a b : ℝ, b ≠ 0 → (a * b) / (c * b) = a / c) → 
  ((∃ (a b : ℝ), a = 0 → a / b = 0) ∧ 
   (∀ (a b : ℝ), (a * k) / (b * k) = a / b) ∧ 
   (∀ (a b : ℝ), (-a) / (-b) = a / b) ∧ 
   (x < 1 → (|2 - x| + x) / 2 ≠ 0) 
  -> (∀ (a b : ℝ), (-a) / (-b) = a / b)) :=
by sorry

end correct_fraction_statement_l223_223413


namespace daniel_total_worth_l223_223149

theorem daniel_total_worth
    (sales_tax_paid : ℝ)
    (sales_tax_rate : ℝ)
    (cost_tax_free_items : ℝ)
    (tax_rate_pos : 0 < sales_tax_rate) :
    sales_tax_paid = 0.30 →
    sales_tax_rate = 0.05 →
    cost_tax_free_items = 18.7 →
    ∃ (x : ℝ), 0.05 * x = 0.30 ∧ (x + cost_tax_free_items = 24.7) := by
    sorry

end daniel_total_worth_l223_223149


namespace XF_XG_value_l223_223034

-- Define the given conditions
noncomputable def AB := 4
noncomputable def BC := 3
noncomputable def CD := 7
noncomputable def DA := 9

noncomputable def DX (BD : ℚ) := (1 / 3) * BD
noncomputable def BY (BD : ℚ) := (1 / 4) * BD

-- Variables and points in the problem
variables (BD p q : ℚ)
variables (A B C D X Y E F G : Point)

-- Proof statement
theorem XF_XG_value 
(AB_eq : AB = 4) (BC_eq : BC = 3) (CD_eq : CD = 7) (DA_eq : DA = 9)
(DX_eq : DX BD = (1 / 3) * BD) (BY_eq : BY BD = (1 / 4) * BD)
(AC_BD_prod : p * q = 55) :
  XF * XG = (110 / 9) := 
by
  sorry

end XF_XG_value_l223_223034


namespace Albert_has_more_rocks_than_Jose_l223_223138

noncomputable def Joshua_rocks : ℕ := 80
noncomputable def Jose_rocks : ℕ := Joshua_rocks - 14
noncomputable def Albert_rocks : ℕ := Joshua_rocks + 6

theorem Albert_has_more_rocks_than_Jose :
  Albert_rocks - Jose_rocks = 20 := by
  sorry

end Albert_has_more_rocks_than_Jose_l223_223138


namespace determine_specialty_l223_223668

variables 
  (Peter_is_mathematician Sergey_is_physicist Roman_is_physicist : Prop)
  (Peter_is_chemist Sergey_is_mathematician Roman_is_chemist : Prop)

-- Conditions
axiom cond1 : Peter_is_mathematician → ¬ Sergey_is_physicist
axiom cond2 : ¬ Roman_is_physicist → Peter_is_mathematician
axiom cond3 : ¬ Sergey_is_mathematician → Roman_is_chemist

theorem determine_specialty 
  (h1 : ¬ Roman_is_physicist)
: Peter_is_chemist ∧ Sergey_is_mathematician ∧ Roman_is_physicist := 
by sorry

end determine_specialty_l223_223668


namespace palmer_first_week_photos_l223_223083

theorem palmer_first_week_photos :
  ∀ (X : ℕ), 
    100 + X + 2 * X + 80 = 380 →
    X = 67 :=
by
  intros X h
  -- h represents the condition 100 + X + 2 * X + 80 = 380
  sorry

end palmer_first_week_photos_l223_223083


namespace container_could_be_emptied_l223_223628

theorem container_could_be_emptied (a b c : ℕ) (h : 0 ≤ a ∧ a ≤ b ∧ b ≤ c) :
  ∃ (a' b' c' : ℕ), (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
  (∀ x y z : ℕ, (a, b, c) = (x, y, z) → (a', b', c') = (y + y, z - y, x - y)) :=
sorry

end container_could_be_emptied_l223_223628


namespace range_of_x_l223_223239

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + (a - 4) * x + 4 - 2 * a

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  ∀ x : ℝ, (f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  intro x
  sorry

end range_of_x_l223_223239


namespace geometric_series_proof_l223_223739

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l223_223739


namespace total_number_of_eyes_l223_223690

theorem total_number_of_eyes (n_spiders n_ants eyes_per_spider eyes_per_ant : ℕ)
  (h1 : n_spiders = 3) (h2 : n_ants = 50) (h3 : eyes_per_spider = 8) (h4 : eyes_per_ant = 2) :
  (n_spiders * eyes_per_spider + n_ants * eyes_per_ant) = 124 :=
by
  sorry

end total_number_of_eyes_l223_223690


namespace number_of_croutons_l223_223082

def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def crouton_calories : ℕ := 20
def total_salad_calories : ℕ := 350

theorem number_of_croutons : 
  ∃ n : ℕ, n * crouton_calories = total_salad_calories - (lettuce_calories + cucumber_calories) ∧ n = 12 :=
by
  sorry

end number_of_croutons_l223_223082


namespace Mary_paid_on_Tuesday_l223_223605

theorem Mary_paid_on_Tuesday 
  (credit_limit total_spent paid_on_thursday remaining_payment paid_on_tuesday : ℝ)
  (h1 : credit_limit = 100)
  (h2 : total_spent = credit_limit)
  (h3 : paid_on_thursday = 23)
  (h4 : remaining_payment = 62)
  (h5 : total_spent = paid_on_thursday + remaining_payment + paid_on_tuesday) :
  paid_on_tuesday = 15 :=
sorry

end Mary_paid_on_Tuesday_l223_223605


namespace reflection_image_l223_223180

theorem reflection_image (m b : ℝ) 
  (h1 : ∀ x y : ℝ, (x, y) = (0, 1) → (4, 5) = (2 * ((x + (m * y - y + b))/ (1 + m^2)) - x, 2 * ((y + (m * x - x + b)) / (1 + m^2)) - y))
  : m + b = 4 :=
sorry

end reflection_image_l223_223180


namespace abcd_product_l223_223481

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

axiom a_eq : a = Real.sqrt (4 - Real.sqrt (5 - a))
axiom b_eq : b = Real.sqrt (4 + Real.sqrt (5 - b))
axiom c_eq : c = Real.sqrt (4 - Real.sqrt (5 + c))
axiom d_eq : d = Real.sqrt (4 + Real.sqrt (5 + d))

theorem abcd_product : a * b * c * d = 11 := sorry

end abcd_product_l223_223481


namespace more_non_product_eight_digit_numbers_l223_223741

def num_eight_digit_numbers := 10^8 - 10^7
def num_four_digit_numbers := 9999 - 1000 + 1
def num_unique_products := (num_four_digit_numbers.choose 2) + num_four_digit_numbers

theorem more_non_product_eight_digit_numbers :
  (num_eight_digit_numbers - num_unique_products) > num_unique_products := by sorry

end more_non_product_eight_digit_numbers_l223_223741


namespace four_digit_num_condition_l223_223923

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l223_223923


namespace smallest_n_l223_223454

theorem smallest_n
  (n : ℕ)
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : n % 7 = 1)
  (h7 : 8 ∣ n) :
  n = 1681 :=
  sorry

end smallest_n_l223_223454


namespace exists_overlapping_pairs_l223_223709

-- Definition of conditions:
def no_boy_danced_with_all_girls (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ b : B, ∃ g : G, ¬ danced b g

def each_girl_danced_with_at_least_one_boy (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ g : G, ∃ b : B, danced b g

-- The main theorem to prove:
theorem exists_overlapping_pairs
  (B : Type) (G : Type) (danced : B → G → Prop)
  (h1 : no_boy_danced_with_all_girls B G danced)
  (h2 : each_girl_danced_with_at_least_one_boy B G danced) :
  ∃ (b1 b2 : B) (g1 g2 : G), b1 ≠ b2 ∧ g1 ≠ g2 ∧ danced b1 g1 ∧ danced b2 g2 :=
sorry

end exists_overlapping_pairs_l223_223709


namespace factorization_identity_l223_223291

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a ^ 2 + 1 - (b ^ 2 + 1)) ^ 3 + ((b ^ 2 + 1) - (c ^ 2 + 1)) ^ 3 + ((c ^ 2 + 1) - (a ^ 2 + 1)) ^ 3) /
  ((a - b) ^ 3 + (b - c) ^ 3 + (c - a) ^ 3)

theorem factorization_identity (a b c : ℝ) : 
  factor_expression a b c = (a + b) * (b + c) * (c + a) := 
by 
  sorry

end factorization_identity_l223_223291


namespace cost_of_items_l223_223823

variable (e t d : ℝ)

noncomputable def ques :=
  5 * e + 5 * t + 2 * d

axiom cond1 : 3 * e + 4 * t = 3.40
axiom cond2 : 4 * e + 3 * t = 4.00
axiom cond3 : 5 * e + 4 * t + 3 * d = 7.50

theorem cost_of_items : ques e t d = 6.93 :=
by
  sorry

end cost_of_items_l223_223823


namespace smallest_radius_squared_of_sphere_l223_223002

theorem smallest_radius_squared_of_sphere :
  ∃ (x y z : ℤ), 
  (x - 2)^2 + y^2 + z^2 = (x^2 + (y - 4)^2 + z^2) ∧
  (x - 2)^2 + y^2 + z^2 = (x^2 + y^2 + (z - 6)^2) ∧
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
  (∃ r, r^2 = (x - 2)^2 + (0 - y)^2 + (0 - z)^2) ∧
  51 = r^2 :=
sorry

end smallest_radius_squared_of_sphere_l223_223002


namespace larger_number_of_two_l223_223123

theorem larger_number_of_two (x y : ℝ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
sorry

end larger_number_of_two_l223_223123


namespace reflection_points_line_l223_223815

theorem reflection_points_line (m b : ℝ)
  (h1 : (10 : ℝ) = 2 * (6 - m * (6 : ℝ) + b)) -- Reflecting the point (6, (m * 6 + b)) to (10, 7)
  (h2 : (6 : ℝ) * m + b = 5) -- Midpoint condition
  (h3 : (6 : ℝ) = (2 + 10) / 2) -- Calculating midpoint x-coordinate
  (h4 : (5 : ℝ) = (3 + 7) / 2) -- Calculating midpoint y-coordinate
  : m + b = 15 :=
sorry

end reflection_points_line_l223_223815


namespace range_of_m_range_of_x_l223_223654

variable {a b m : ℝ}

-- Given conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom sum_eq_one : a + b = 1

-- Problem (I): Prove range of m
theorem range_of_m (h : ab ≤ m) : m ≥ 1 / 4 := by
  sorry

variable {x : ℝ}

-- Problem (II): Prove range of x
theorem range_of_x (h : 4 / a + 1 / b ≥ |2 * x - 1| - |x + 2|) : -2 ≤ x ∧ x ≤ 6 := by
  sorry

end range_of_m_range_of_x_l223_223654


namespace unique_non_overtaken_city_l223_223694

structure City :=
(size_left : ℕ)
(size_right : ℕ)

def canOvertake (A B : City) : Prop :=
  A.size_right > B.size_left 

theorem unique_non_overtaken_city (n : ℕ) (H : n > 0) (cities : Fin n → City) : 
  ∃! i : Fin n, ∀ j : Fin n, ¬ canOvertake (cities j) (cities i) :=
by
  sorry

end unique_non_overtaken_city_l223_223694


namespace find_d_l223_223254

theorem find_d (a d : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d * x + 12) :
  d = 7 :=
sorry

end find_d_l223_223254


namespace ad_plus_bc_eq_pm_one_l223_223676

theorem ad_plus_bc_eq_pm_one
  (a b c d : ℤ)
  (h1 : ∃ n : ℤ, n = ad + bc ∧ n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d) :
  ad + bc = 1 ∨ ad + bc = -1 := 
sorry

end ad_plus_bc_eq_pm_one_l223_223676


namespace quadratic_ineq_solution_set_l223_223332

theorem quadratic_ineq_solution_set {m : ℝ} :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
sorry

end quadratic_ineq_solution_set_l223_223332


namespace boots_cost_more_l223_223018

theorem boots_cost_more (S B : ℝ) 
  (h1 : 22 * S + 16 * B = 460) 
  (h2 : 8 * S + 32 * B = 560) : B - S = 5 :=
by
  -- Here we provide the statement only, skipping the proof
  sorry

end boots_cost_more_l223_223018


namespace smallest_positive_m_l223_223890

theorem smallest_positive_m (m : ℕ) : 
  (∃ n : ℤ, (10 * n * (n + 1) = 600) ∧ (m = 10 * (n + (n + 1)))) → (m = 170) :=
by 
  sorry

end smallest_positive_m_l223_223890


namespace min_value_of_quartic_function_l223_223380

theorem min_value_of_quartic_function : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ 1) → x^4 + (1 - x)^4 ≤ y^4 + (1 - y)^4) ∧ (x^4 + (1 - x)^4 = 1 / 8) :=
by
  sorry

end min_value_of_quartic_function_l223_223380


namespace temperature_at_night_is_minus_two_l223_223357

theorem temperature_at_night_is_minus_two (temperature_noon temperature_afternoon temperature_drop_by_night temperature_night : ℤ) : 
  temperature_noon = 5 → temperature_afternoon = 7 → temperature_drop_by_night = 9 → 
  temperature_night = temperature_afternoon - temperature_drop_by_night → 
  temperature_night = -2 := 
by
  intros h1 h2 h3 h4
  rw [h2, h3] at h4
  exact h4


end temperature_at_night_is_minus_two_l223_223357


namespace number_of_books_bought_l223_223565

def initial_books : ℕ := 35
def books_given_away : ℕ := 12
def final_books : ℕ := 56

theorem number_of_books_bought : initial_books - books_given_away + (final_books - (initial_books - books_given_away)) = final_books :=
by
  sorry

end number_of_books_bought_l223_223565


namespace alloy_problem_l223_223102

theorem alloy_problem (x y : ℝ) 
  (h1 : x + y = 1000) 
  (h2 : 0.25 * x + 0.50 * y = 450) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) :
  x = 200 ∧ y = 800 := 
sorry

end alloy_problem_l223_223102


namespace ferrisWheelPeopleCount_l223_223401

/-!
# Problem Description

We are given the following conditions:
- The ferris wheel has 6.0 seats.
- It has to run 2.333333333 times for everyone to get a turn.

We need to prove that the total number of people who want to ride the ferris wheel is 14.
-/

def ferrisWheelSeats : ℕ := 6
def ferrisWheelRuns : ℚ := 2333333333 / 1000000000

theorem ferrisWheelPeopleCount :
  (ferrisWheelSeats : ℚ) * ferrisWheelRuns = 14 :=
by
  sorry

end ferrisWheelPeopleCount_l223_223401


namespace problem_D_l223_223939

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b c : V}

def is_parallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

theorem problem_D (h₁ : is_parallel a b) (h₂ : is_parallel b c) (h₃ : b ≠ 0) : is_parallel a c :=
sorry

end problem_D_l223_223939


namespace find_g_of_3_l223_223511

theorem find_g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 5 * g (1 / x) = 2 * x) : g 3 = -32 / 63 :=
by sorry

end find_g_of_3_l223_223511


namespace problem_l223_223139

def T := {n : ℤ | ∃ (k : ℤ), n = 4 * (2*k + 1)^2 + 13}

theorem problem :
  (∀ n ∈ T, ¬ 2 ∣ n) ∧ (∀ n ∈ T, ¬ 5 ∣ n) :=
by
  sorry

end problem_l223_223139


namespace correct_graph_for_race_l223_223190

-- Define the conditions for the race.
def tortoise_constant_speed (d t : ℝ) := 
  ∃ k : ℝ, k > 0 ∧ d = k * t

def hare_behavior (d t t_nap t_end d_nap : ℝ) :=
  ∃ k1 k2 : ℝ, k1 > 0 ∧ k2 > 0 ∧ t_nap > 0 ∧ t_end > t_nap ∧
  (d = k1 * t ∨ (t_nap < t ∧ t < t_end ∧ d = d_nap) ∨ (t_end ≥ t ∧ d = d_nap + k2 * (t - t_end)))

-- Define the competition outcome.
def tortoise_wins (d_tortoise d_hare : ℝ) :=
  d_tortoise > d_hare

-- Proof that the graph which describes the race is Option (B).
theorem correct_graph_for_race :
  ∃ d_t d_h t t_nap t_end d_nap, 
    tortoise_constant_speed d_t t ∧ hare_behavior d_h t t_nap t_end d_nap ∧ tortoise_wins d_t d_h → "Option B" = "correct" :=
sorry -- Proof omitted.

end correct_graph_for_race_l223_223190


namespace space_filled_with_rhombic_dodecahedra_l223_223234

/-
  Given: Space can be filled completely using cubic cells (cubic lattice).
  To Prove: Space can be filled completely using rhombic dodecahedron cells.
-/

theorem space_filled_with_rhombic_dodecahedra :
  (∀ (cubic_lattice : Type), (∃ fill_space_with_cubes : (cubic_lattice → Prop), 
    ∀ x : cubic_lattice, fill_space_with_cubes x)) →
  (∃ (rhombic_dodecahedra_lattice : Type), 
      (∀ fill_space_with_rhombic_dodecahedra : rhombic_dodecahedra_lattice → Prop, 
        ∀ y : rhombic_dodecahedra_lattice, fill_space_with_rhombic_dodecahedra y)) :=
by {
  sorry
}

end space_filled_with_rhombic_dodecahedra_l223_223234


namespace bruce_michael_total_goals_l223_223909

theorem bruce_michael_total_goals (bruce_goals : ℕ) (michael_goals : ℕ) 
  (h₁ : bruce_goals = 4) (h₂ : michael_goals = 3 * bruce_goals) : bruce_goals + michael_goals = 16 :=
by sorry

end bruce_michael_total_goals_l223_223909


namespace carlson_max_candies_l223_223272

theorem carlson_max_candies : 
  (∀ (erase_two_and_sum : ℕ → ℕ → ℕ) 
    (eat_candies : ℕ → ℕ → ℕ), 
  ∃ (maximum_candies : ℕ), 
  (erase_two_and_sum 1 1 = 2) ∧
  (eat_candies 1 1 = 1) ∧ 
  (maximum_candies = 496)) :=
by
  sorry

end carlson_max_candies_l223_223272


namespace scientific_notation_of_neg_0_000008691_l223_223223

theorem scientific_notation_of_neg_0_000008691:
  -0.000008691 = -8.691 * 10^(-6) :=
sorry

end scientific_notation_of_neg_0_000008691_l223_223223


namespace arithmetic_sequence_seventh_term_l223_223108

theorem arithmetic_sequence_seventh_term (a d : ℝ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 14) 
  (h2 : a + 4 * d = 9) : 
  a + 6 * d = 13.4 := 
sorry

end arithmetic_sequence_seventh_term_l223_223108


namespace arithmetic_sequence_term_l223_223406

theorem arithmetic_sequence_term (a : ℕ → ℤ) (d : ℤ) (n : ℕ) :
  a 5 = 33 ∧ a 45 = 153 ∧ (∀ n, a n = a 1 + (n - 1) * d) ∧ a n = 201 → n = 61 :=
by
  sorry

end arithmetic_sequence_term_l223_223406


namespace basketball_prob_l223_223776

theorem basketball_prob :
  let P_A := 0.7
  let P_B := 0.6
  P_A * P_B = 0.88 := 
by 
  sorry

end basketball_prob_l223_223776


namespace solve_for_x_l223_223959

theorem solve_for_x : ∀ x : ℝ, ( (x * x^(2:ℝ)) ^ (1/6) )^2 = 4 → x = 4 := by
  intro x
  sorry

end solve_for_x_l223_223959


namespace length_of_platform_l223_223025

theorem length_of_platform (L : ℕ) :
  (∀ (V : ℚ), V = 600 / 52 → V = (600 + L) / 78) → L = 300 :=
by
  sorry

end length_of_platform_l223_223025


namespace probability_red_ball_l223_223334

-- Let P_red be the probability of drawing a red ball.
-- Let P_white be the probability of drawing a white ball.
-- Let P_black be the probability of drawing a black ball.
-- Let P_red_or_white be the probability of drawing a red or white ball.
-- Let P_red_or_black be the probability of drawing a red or black ball.

variable (P_red P_white P_black : ℝ)
variable (P_red_or_white P_red_or_black : ℝ)

-- Given conditions
axiom P_red_or_white_condition : P_red_or_white = 0.58
axiom P_red_or_black_condition : P_red_or_black = 0.62

-- The total probability must sum to 1.
axiom total_probability_condition : P_red + P_white + P_black = 1

-- Prove that the probability of drawing a red ball is 0.2.
theorem probability_red_ball : P_red = 0.2 :=
by
  -- To be proven
  sorry

end probability_red_ball_l223_223334


namespace arccos_one_eq_zero_l223_223429

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l223_223429


namespace johnsonville_max_members_l223_223759

theorem johnsonville_max_members 
  (n : ℤ) 
  (h1 : 15 * n % 30 = 6) 
  (h2 : 15 * n < 900) 
  : 15 * n ≤ 810 :=
sorry

end johnsonville_max_members_l223_223759


namespace problem_one_problem_two_l223_223880

-- Define the given vectors
def vector_oa : ℝ × ℝ := (-1, 3)
def vector_ob : ℝ × ℝ := (3, -1)
def vector_oc (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the subtraction of two 2D vectors
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Define the parallel condition (u and v are parallel if u = k*v for some scalar k)
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1  -- equivalent to u = k*v

-- Define the dot product in 2D
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Problem 1
theorem problem_one (m : ℝ) :
  is_parallel (vector_sub vector_ob vector_oa) (vector_oc m) ↔ m = -1 :=
by
-- Proof omitted
sorry

-- Problem 2
theorem problem_two (m : ℝ) :
  dot_product (vector_sub (vector_oc m) vector_oa) (vector_sub (vector_oc m) vector_ob) = 0 ↔
  m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2 :=
by
-- Proof omitted
sorry

end problem_one_problem_two_l223_223880


namespace Niklaus_walked_distance_l223_223587

noncomputable def MilesToFeet (miles : ℕ) : ℕ := miles * 5280
noncomputable def YardsToFeet (yards : ℕ) : ℕ := yards * 3

theorem Niklaus_walked_distance (n_feet : ℕ) :
  MilesToFeet 4 + YardsToFeet 975 + n_feet = 25332 → n_feet = 1287 := by
  sorry

end Niklaus_walked_distance_l223_223587


namespace min_value_eq_144_l223_223578

noncomputable def min_value (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) : ℝ :=
  if x <= 0 ∨ y <= 0 ∨ z <= 0 ∨ w <= 0 then 0 else (x + y + z) / (x * y * z * w)

theorem min_value_eq_144 (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) :
  min_value x y z w h_pos_x h_pos_y h_pos_z h_pos_w h_sum = 144 :=
sorry

end min_value_eq_144_l223_223578


namespace find_other_sides_of_triangle_l223_223956

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end find_other_sides_of_triangle_l223_223956


namespace significant_figures_and_precision_l223_223524

-- Definition of the function to count significant figures
def significant_figures (n : Float) : Nat :=
  -- Implementation of a function that counts significant figures
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- Definition of the function to determine precision
def precision (n : Float) : String :=
  -- Implementation of a function that returns the precision
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- The target number
def num := 0.03020

-- The properties of the number 0.03020
theorem significant_figures_and_precision :
  significant_figures num = 4 ∧ precision num = "ten-thousandth" :=
by
  sorry

end significant_figures_and_precision_l223_223524


namespace trajectory_equation_l223_223078

theorem trajectory_equation 
  (P : ℝ × ℝ)
  (h : (P.2 / (P.1 + 4)) * (P.2 / (P.1 - 4)) = -4 / 9) :
  P.1 ≠ 4 ∧ P.1 ≠ -4 → P.1^2 / 64 + P.2^2 / (64 / 9) = 1 :=
by
  sorry

end trajectory_equation_l223_223078


namespace sum_of_squares_eq_23456_l223_223439

theorem sum_of_squares_eq_23456 (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end sum_of_squares_eq_23456_l223_223439


namespace intersection_point_l223_223857

def line_parametric (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, -1 + 3 * t, -3 + 2 * t)

def on_plane (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

theorem intersection_point : ∃ t, line_parametric t = (5, 2, -1) ∧ on_plane 5 2 (-1) :=
by
  use 1
  sorry

end intersection_point_l223_223857


namespace michael_passes_donovan_after_laps_l223_223780

/-- The length of the track in meters -/
def track_length : ℕ := 400

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℕ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℕ := 36

/-- The number of laps that Michael will have to complete in order to pass Donovan -/
theorem michael_passes_donovan_after_laps : 
  ∃ (laps : ℕ), laps = 5 ∧ (∃ t : ℕ, 400 * t / 36 = 5 ∧ 400 * t / 45 < 5) :=
sorry

end michael_passes_donovan_after_laps_l223_223780


namespace value_spent_more_than_l223_223692

theorem value_spent_more_than (x : ℕ) (h : 8 * 12 + (x + 8) = 117) : x = 13 :=
by
  sorry

end value_spent_more_than_l223_223692


namespace arithmetic_series_first_term_l223_223825

theorem arithmetic_series_first_term :
  ∃ (a d : ℝ), (25 * (2 * a + 49 * d) = 200) ∧ (25 * (2 * a + 149 * d) = 2700) ∧ (a = -20.5) :=
by
  sorry

end arithmetic_series_first_term_l223_223825


namespace pyramid_volume_pyramid_surface_area_l223_223453

noncomputable def volume_of_pyramid (l : ℝ) := (l^3 * Real.sqrt 2) / 12

noncomputable def surface_area_of_pyramid (l : ℝ) := (l^2 * (2 + Real.sqrt 2)) / 2

theorem pyramid_volume (l : ℝ) :
  volume_of_pyramid l = (l^3 * Real.sqrt 2) / 12 :=
sorry

theorem pyramid_surface_area (l : ℝ) :
  surface_area_of_pyramid l = (l^2 * (2 + Real.sqrt 2)) / 2 :=
sorry

end pyramid_volume_pyramid_surface_area_l223_223453


namespace expression_evaluation_l223_223400

theorem expression_evaluation (a : ℕ) (h : a = 1580) : 
  2 * a - ((2 * a - 3) / (a + 1) - (a + 1) / (2 - 2 * a) - (a^2 + 3) / 2) * ((a^3 + 1) / (a^2 - a)) + 2 / a = 2 := 
sorry

end expression_evaluation_l223_223400


namespace boat_speed_still_water_l223_223526

variable (V_b V_s : ℝ)

def upstream : Prop := V_b - V_s = 10
def downstream : Prop := V_b + V_s = 40

theorem boat_speed_still_water (h1 : upstream V_b V_s) (h2 : downstream V_b V_s) : V_b = 25 :=
by
  sorry

end boat_speed_still_water_l223_223526


namespace smallest_pos_int_ends_in_6_divisible_by_11_l223_223944

theorem smallest_pos_int_ends_in_6_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 6 ∧ 11 ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 6 ∧ 11 ∣ m → n ≤ m := by
  sorry

end smallest_pos_int_ends_in_6_divisible_by_11_l223_223944


namespace cost_of_tax_free_items_l223_223171

theorem cost_of_tax_free_items (total_cost : ℝ) (tax_40_percent : ℝ) 
  (tax_30_percent : ℝ) (discount : ℝ) : 
  (total_cost = 120) →
  (tax_40_percent = 0.4 * total_cost) →
  (tax_30_percent = 0.3 * total_cost) →
  (discount = 0.05 * tax_30_percent) →
  (tax-free_items = total_cost - (tax_40_percent + (tax_30_percent - discount))) → 
  tax_free_items = 36 :=
by sorry

end cost_of_tax_free_items_l223_223171


namespace min_m_plus_n_l223_223451

open Nat

theorem min_m_plus_n (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 45 * m = n^3) (h_mult_of_five : 5 ∣ n) :
  m + n = 90 :=
sorry

end min_m_plus_n_l223_223451


namespace part1_part2_part3_l223_223594

-- Define the necessary constants and functions as per conditions
variable (a : ℝ) (f : ℝ → ℝ)
variable (hpos : a > 0) (hfa : f a = 1)

-- Conditions based on the problem statement
variable (hodd : ∀ x, f (-x) = -f x)
variable (hfe : ∀ x1 x2, f (x1 - x2) = (f x1 * f x2 + 1) / (f x2 - f x1))

-- 1. Prove that f(2a) = 0
theorem part1  : f (2 * a) = 0 := sorry

-- 2. Prove that there exists a constant T > 0 such that f(x + T) = f(x)
theorem part2 : ∃ T > 0, ∀ x, f (x + 4 * a) = f x := sorry

-- 3. Prove f(x) is decreasing on (0, 4a) given x ∈ (0, 2a) implies f(x) > 0
theorem part3 (hx_correct : ∀ x, 0 < x ∧ x < 2 * a → 0 < f x) :
  ∀ x1 x2, 0 < x2 ∧ x2  < x1 ∧ x1 < 4 * a → f x2 > f x1 := sorry

end part1_part2_part3_l223_223594


namespace correct_operation_l223_223086

theorem correct_operation (a : ℝ) : 2 * a^3 / a^2 = 2 * a := 
sorry

end correct_operation_l223_223086


namespace toothpick_pattern_15th_stage_l223_223718

theorem toothpick_pattern_15th_stage :
  let a₁ := 5
  let d := 3
  let n := 15
  a₁ + (n - 1) * d = 47 :=
by
  sorry

end toothpick_pattern_15th_stage_l223_223718


namespace trail_mix_total_weight_l223_223994

def peanuts : ℝ := 0.17
def chocolate_chips : ℝ := 0.17
def raisins : ℝ := 0.08

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.42 :=
by
  -- The proof would go here
  sorry

end trail_mix_total_weight_l223_223994


namespace solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l223_223422

theorem solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c ≤ 0 ↔ x ≤ -1 ∨ x ≥ 3) →
  b = -2*a →
  c = -3*a →
  a < 0 →
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) := 
by 
  intro h_root_set h_b_eq h_c_eq h_a_lt_0 
  sorry

end solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l223_223422


namespace initial_tickets_l223_223870

theorem initial_tickets (tickets_sold_week1 : ℕ) (tickets_sold_week2 : ℕ) (tickets_left : ℕ) 
  (h1 : tickets_sold_week1 = 38) (h2 : tickets_sold_week2 = 17) (h3 : tickets_left = 35) : 
  tickets_sold_week1 + tickets_sold_week2 + tickets_left = 90 :=
by 
  sorry

end initial_tickets_l223_223870


namespace cos_13pi_over_4_eq_neg_one_div_sqrt_two_l223_223438

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end cos_13pi_over_4_eq_neg_one_div_sqrt_two_l223_223438


namespace h_comp_h_3_l223_223238

def h (x : ℕ) : ℕ := 3 * x * x + 5 * x - 3

theorem h_comp_h_3 : h (h 3) = 4755 := by
  sorry

end h_comp_h_3_l223_223238


namespace range_of_m_l223_223969

noncomputable def f (x m : ℝ) := Real.exp x * (Real.log x + (1 / 2) * x ^ 2 - m * x)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → ((Real.exp x * ((1 / x) + x - m)) > 0)) → m < 2 := by
  sorry

end range_of_m_l223_223969


namespace minimum_quadratic_expression_l223_223295

theorem minimum_quadratic_expression : ∃ (x : ℝ), (∀ y : ℝ, y^2 - 6*y + 5 ≥ -4) ∧ (x^2 - 6*x + 5 = -4) :=
by
  sorry

end minimum_quadratic_expression_l223_223295


namespace common_root_and_param_l223_223616

theorem common_root_and_param :
  ∀ (x : ℤ) (P p : ℚ),
    (P = -((x^2 - x - 2) / (x - 1)) ∧ x ≠ 1) →
    (p = -((x^2 + 2*x - 1) / (x + 2)) ∧ x ≠ -2) →
    (-x + (2 / (x - 1)) = -x + (1 / (x + 2))) →
    x = -5 ∧ p = 14 / 3 :=
by
  intros x P p hP hp hroot
  sorry

end common_root_and_param_l223_223616


namespace max_value_f_l223_223425

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.sin (2 * x) + (a - 2) * Real.cos (2 * x)

theorem max_value_f (a : ℝ) (h : a < 0)
  (symm : ∀ x, f a (x - π / 4) = f a (-x - π / 4)) :
  ∃ x, f a x = 4 * Real.sqrt 2 :=
sorry

end max_value_f_l223_223425


namespace no_real_solutions_quadratic_solve_quadratic_eq_l223_223586

-- For Equation (1)

theorem no_real_solutions_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = -4 ∧ c = 5 ∧ (b^2 - 4 * a * c < 0)) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := 
by
  sorry

-- For Equation (2)

theorem solve_quadratic_eq {x : ℝ} (h_eq : (x + 1) * (x + 2) = 2 * x + 4) :
  x = -2 ∨ x = 1 :=
by
  sorry

end no_real_solutions_quadratic_solve_quadratic_eq_l223_223586


namespace subset_of_inter_eq_self_l223_223279

variable {α : Type*}
variables (M N : Set α)

theorem subset_of_inter_eq_self (h : M ∩ N = M) : M ⊆ N :=
sorry

end subset_of_inter_eq_self_l223_223279


namespace sum_abcd_l223_223258

variable {a b c d : ℚ}

theorem sum_abcd 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 :=
by sorry

end sum_abcd_l223_223258


namespace initial_number_of_girls_l223_223472

theorem initial_number_of_girls (p : ℝ) (h : (0.4 * p - 2) / p = 0.3) : 0.4 * p = 8 := 
by
  sorry

end initial_number_of_girls_l223_223472


namespace number_of_integers_l223_223150

theorem number_of_integers (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2020) (h3 : ∃ k : ℕ, n^n = k^2) : n = 1032 :=
sorry

end number_of_integers_l223_223150


namespace sequence_term_sum_max_value_sum_equality_l223_223789

noncomputable def a (n : ℕ) : ℝ := -2 * n + 6

def S (n : ℕ) : ℝ := -n^2 + 5 * n

theorem sequence_term (n : ℕ) : ∀ n, a n = 4 + (n - 1) * (-2) :=
by sorry

theorem sum_max_value (n : ℕ) : ∃ n, S n = 6 :=
by sorry

theorem sum_equality : S 2 = 6 ∧ S 3 = 6 :=
by sorry

end sequence_term_sum_max_value_sum_equality_l223_223789


namespace nth_smallest_d0_perfect_square_l223_223964

theorem nth_smallest_d0_perfect_square (n : ℕ) : 
  ∃ (d_0 : ℕ), (∃ v : ℕ, ∀ t : ℝ, (2 * t * t + d_0 = v * t) ∧ (∃ k : ℕ, v = k ∧ k * k = v * v)) 
               ∧ d_0 = 4^(n - 1) := 
by sorry

end nth_smallest_d0_perfect_square_l223_223964


namespace can_split_3x3x3_into_9_corners_l223_223801

-- Define the conditions
def number_of_cubes_in_3x3x3 : ℕ := 27
def number_of_units_in_corner : ℕ := 3
def number_of_corners : ℕ := 9

-- Prove the proposition
theorem can_split_3x3x3_into_9_corners :
  (number_of_corners * number_of_units_in_corner = number_of_cubes_in_3x3x3) :=
by
  sorry

end can_split_3x3x3_into_9_corners_l223_223801


namespace orange_ratio_l223_223286

theorem orange_ratio (total_oranges : ℕ) (brother_fraction : ℚ) (friend_receives : ℕ)
  (H1 : total_oranges = 12)
  (H2 : friend_receives = 2)
  (H3 : 1 / 4 * ((1 - brother_fraction) * total_oranges) = friend_receives) :
  brother_fraction * total_oranges / total_oranges = 1 / 3 :=
by
  sorry

end orange_ratio_l223_223286


namespace ninth_number_l223_223091

theorem ninth_number (S1 S2 Total N : ℕ)
  (h1 : S1 = 9 * 56)
  (h2 : S2 = 9 * 63)
  (h3 : Total = 17 * 59)
  (h4 : Total = S1 + S2 - N) :
  N = 68 :=
by 
  -- The proof is omitted, only the statement is needed.
  sorry

end ninth_number_l223_223091


namespace problem_1_problem_2_l223_223359

-- Define the sets A, B, C
def SetA (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def SetB : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def SetC : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

-- Problem 1
theorem problem_1 (a : ℝ) : SetA a = SetB → a = 5 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (SetA a ∩ SetB).Nonempty ∧ (SetA a ∩ SetC = ∅) → a = -2 := by
  sorry

end problem_1_problem_2_l223_223359


namespace product_evaluation_l223_223231

theorem product_evaluation :
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 *
  (1 / 512) * 1024 * (1 / 2048) * 4096 = 64 :=
by
  sorry

end product_evaluation_l223_223231


namespace carries_jellybeans_l223_223268

/-- Bert's box holds 150 jellybeans. --/
def bert_jellybeans : ℕ := 150

/-- Carrie's box is three times as high, three times as wide, and three times as long as Bert's box. --/
def volume_ratio : ℕ := 27

/-- Given that Carrie's box dimensions are three times those of Bert's and Bert's box holds 150 jellybeans, 
    we need to prove that Carrie's box holds 4050 jellybeans. --/
theorem carries_jellybeans : bert_jellybeans * volume_ratio = 4050 := 
by sorry

end carries_jellybeans_l223_223268


namespace ball_distribution_l223_223099

theorem ball_distribution (balls boxes : ℕ) (hballs : balls = 7) (hboxes : boxes = 4) :
  (∃ (ways : ℕ), ways = (Nat.choose (balls - 1) (boxes - 1)) ∧ ways = 20) :=
by
  sorry

end ball_distribution_l223_223099


namespace find_covered_number_l223_223175

theorem find_covered_number (a x : ℤ) (h : (x - a) / 2 = x + 3) (hx : x = -7) : a = 1 := by
  sorry

end find_covered_number_l223_223175


namespace base_five_to_base_ten_modulo_seven_l223_223297

-- Define the base five number 21014_5 as the corresponding base ten conversion
def base_five_number : ℕ := 2 * 5^4 + 1 * 5^3 + 0 * 5^2 + 1 * 5^1 + 4 * 5^0

-- The equivalent base ten result
def base_ten_number : ℕ := 1384

-- Verify the base ten equivalent of 21014_5
theorem base_five_to_base_ten : base_five_number = base_ten_number :=
by
  -- The expected proof should compute the value of base_five_number
  -- and check that it equals 1384
  sorry

-- Find the modulo operation result of 1384 % 7
def modulo_seven_result : ℕ := 6

-- Verify 1384 % 7 gives 6
theorem modulo_seven : base_ten_number % 7 = modulo_seven_result :=
by
  -- The expected proof should compute 1384 % 7
  -- and check that it equals 6
  sorry

end base_five_to_base_ten_modulo_seven_l223_223297


namespace platform_length_1000_l223_223729

open Nat Real

noncomputable def length_of_platform (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) : ℝ :=
  let speed := train_length / time_pole
  let platform_length := (speed * time_platform) - train_length
  platform_length

theorem platform_length_1000 :
  length_of_platform 300 9 39 = 1000 := by
  sorry

end platform_length_1000_l223_223729


namespace least_tiles_needed_l223_223881

-- Define the conditions
def hallway_length_ft : ℕ := 18
def hallway_width_ft : ℕ := 6
def tile_side_in : ℕ := 6
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Translate conditions
def hallway_length_in := feet_to_inches hallway_length_ft
def hallway_width_in := feet_to_inches hallway_width_ft

-- Define the areas
def hallway_area : ℕ := hallway_length_in * hallway_width_in
def tile_area : ℕ := tile_side_in * tile_side_in

-- State the theorem to be proved
theorem least_tiles_needed :
  hallway_area / tile_area = 432 := 
sorry

end least_tiles_needed_l223_223881


namespace circle_symmetry_l223_223428

theorem circle_symmetry {a : ℝ} (h : a ≠ 0) :
  ∀ {x y : ℝ}, (x^2 + y^2 + 2*a*x - 2*a*y = 0) → (x + y = 0) :=
sorry

end circle_symmetry_l223_223428


namespace inverse_proportion_function_l223_223850

theorem inverse_proportion_function (f : ℝ → ℝ) (h : ∀ x, f x = 1/x) : f 1 = 1 := 
by
  sorry

end inverse_proportion_function_l223_223850


namespace q_can_complete_work_in_25_days_l223_223462

-- Define work rates for p, q, and r
variables (W_p W_q W_r : ℝ)

-- Define total work
variable (W : ℝ)

-- Prove that q can complete the work in 25 days under given conditions
theorem q_can_complete_work_in_25_days
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 50) :
  W_q = W / 25 :=
by
  -- Given: W_p = W_q + W_r
  -- Given: W_p + W_q = W / 10
  -- Given: W_r = W / 50
  -- We need to prove: W_q = W / 25
  sorry

end q_can_complete_work_in_25_days_l223_223462


namespace x_squared_plus_y_squared_l223_223158

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 := by
  sorry

end x_squared_plus_y_squared_l223_223158


namespace find_angle_l223_223130

def complementary (x : ℝ) := 90 - x
def supplementary (x : ℝ) := 180 - x

theorem find_angle (x : ℝ) (h : supplementary x = 3 * complementary x) : x = 45 :=
by 
  sorry

end find_angle_l223_223130


namespace original_cube_volume_l223_223855

theorem original_cube_volume (a : ℕ) (h : (a + 2) * (a + 1) * (a - 1) + 6 = a^3) : a = 2 :=
by sorry

example : 2^3 = 8 := by norm_num

end original_cube_volume_l223_223855


namespace train_crossing_time_l223_223384

theorem train_crossing_time
  (length_of_train : ℝ)
  (speed_in_kmh : ℝ)
  (speed_in_mps : ℝ)
  (conversion_factor : ℝ)
  (time : ℝ)
  (h1 : length_of_train = 160)
  (h2 : speed_in_kmh = 36)
  (h3 : conversion_factor = 1 / 3.6)
  (h4 : speed_in_mps = speed_in_kmh * conversion_factor)
  (h5 : time = length_of_train / speed_in_mps) : time = 16 :=
by
  sorry

end train_crossing_time_l223_223384


namespace smallest_positive_x_for_maximum_sine_sum_l223_223478

theorem smallest_positive_x_for_maximum_sine_sum :
  ∃ x : ℝ, (0 < x) ∧ (∃ k m : ℕ, x = 450 + 1800 * k ∧ x = 630 + 2520 * m ∧ x = 12690) := by
  sorry

end smallest_positive_x_for_maximum_sine_sum_l223_223478


namespace max_d_n_l223_223232

open Int

def a_n (n : ℕ) : ℤ := 80 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (a_n n) (a_n (n + 1))

theorem max_d_n : ∃ n : ℕ, d_n n = 5 ∧ ∀ m : ℕ, d_n m ≤ 5 := by
  sorry

end max_d_n_l223_223232


namespace detour_distance_l223_223848

-- Definitions based on conditions:
def D_black : ℕ := sorry -- The original distance along the black route
def D_black_C : ℕ := sorry -- The distance from C to B along the black route
def D_red : ℕ := sorry -- The distance from C to B along the red route

-- Extra distance due to detour calculation
def D_extra := D_red - D_black_C

-- Prove that the extra distance is 14 km
theorem detour_distance : D_extra = 14 := by
  sorry

end detour_distance_l223_223848


namespace difference_of_M_and_m_l223_223174

-- Define the variables and conditions
def total_students : ℕ := 2500
def min_G : ℕ := 1750
def max_G : ℕ := 1875
def min_R : ℕ := 1000
def max_R : ℕ := 1125

-- The statement to prove
theorem difference_of_M_and_m : 
  ∃ G R m M, 
  (G = total_students - R + m) ∧ 
  (min_G ≤ G ∧ G ≤ max_G) ∧
  (min_R ≤ R ∧ R ≤ max_R) ∧
  (m = min_G + min_R - total_students) ∧
  (M = max_G + max_R - total_students) ∧
  (M - m = 250) :=
sorry

end difference_of_M_and_m_l223_223174


namespace overall_profit_or_loss_l223_223087

def price_USD_to_INR(price_usd : ℝ) : ℝ := price_usd * 75
def price_EUR_to_INR(price_eur : ℝ) : ℝ := price_eur * 80
def price_GBP_to_INR(price_gbp : ℝ) : ℝ := price_gbp * 100
def price_JPY_to_INR(price_jpy : ℝ) : ℝ := price_jpy * 0.7

def CP_grinder : ℝ := price_USD_to_INR (150 + 0.1 * 150)
def SP_grinder : ℝ := price_USD_to_INR (165 - 0.04 * 165)

def CP_mobile_phone : ℝ := price_EUR_to_INR ((100 - 0.05 * 100) + 0.15 * (100 - 0.05 * 100))
def SP_mobile_phone : ℝ := price_EUR_to_INR ((109.25 : ℝ) + 0.1 * 109.25)

def CP_laptop : ℝ := price_GBP_to_INR (200 + 0.08 * 200)
def SP_laptop : ℝ := price_GBP_to_INR (216 - 0.08 * 216)

def CP_camera : ℝ := price_JPY_to_INR ((12000 - 0.12 * 12000) + 0.05 * (12000 - 0.12 * 12000))
def SP_camera : ℝ := price_JPY_to_INR (11088 + 0.15 * 11088)

def total_CP : ℝ := CP_grinder + CP_mobile_phone + CP_laptop + CP_camera
def total_SP : ℝ := SP_grinder + SP_mobile_phone + SP_laptop + SP_camera

theorem overall_profit_or_loss :
  (total_SP - total_CP) = -184.76 := 
sorry

end overall_profit_or_loss_l223_223087


namespace royalty_amount_l223_223837

theorem royalty_amount (x : ℝ) (h1 : x > 800) (h2 : x ≤ 4000) (h3 : (x - 800) * 0.14 = 420) :
  x = 3800 :=
by
  sorry

end royalty_amount_l223_223837


namespace elise_spent_on_comic_book_l223_223484

-- Define the initial amount of money Elise had
def initial_amount : ℤ := 8

-- Define the amount saved from allowance
def saved_amount : ℤ := 13

-- Define the amount spent on puzzle
def spent_on_puzzle : ℤ := 18

-- Define the amount left after all expenditures
def amount_left : ℤ := 1

-- Define the total amount of money Elise had after saving
def total_amount : ℤ := initial_amount + saved_amount

-- Define the total amount spent which equals
-- the sum of amount spent on the comic book and the puzzle
def total_spent : ℤ := total_amount - amount_left

-- Define the amount spent on the comic book as the proposition to be proved
def spent_on_comic_book : ℤ := total_spent - spent_on_puzzle

-- State the theorem to prove how much Elise spent on the comic book
theorem elise_spent_on_comic_book : spent_on_comic_book = 2 :=
by
  sorry

end elise_spent_on_comic_book_l223_223484


namespace sum_three_ways_l223_223509

theorem sum_three_ways (n : ℕ) (h : n > 0) : 
  ∃ k, k = (n^2) / 12 ∧ k = (n^2) / 12 :=
sorry

end sum_three_ways_l223_223509


namespace find_f_l223_223829

theorem find_f (f : ℝ → ℝ) (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → x ≤ y → f x ≤ f y)
  (h₂ : ∀ x : ℝ, 0 < x → f (x ^ 4) + f (x ^ 2) + f x + f 1 = x ^ 4 + x ^ 2 + x + 1) :
  ∀ x : ℝ, 0 < x → f x = x := 
sorry

end find_f_l223_223829


namespace quadratic_eq_has_equal_roots_l223_223200

theorem quadratic_eq_has_equal_roots (q : ℚ) :
  (∃ x : ℚ, x^2 - 3 * x + q = 0 ∧ (x^2 - 3 * x + q = 0)) → q = 9 / 4 :=
by
  sorry

end quadratic_eq_has_equal_roots_l223_223200


namespace base_5_minus_base_8_in_base_10_l223_223899

def base_5 := 52143
def base_8 := 4310

theorem base_5_minus_base_8_in_base_10 :
  (5 * 5^4 + 2 * 5^3 + 1 * 5^2 + 4 * 5^1 + 3 * 5^0) -
  (4 * 8^3 + 3 * 8^2 + 1 * 8^1 + 0 * 8^0)
  = 1175 := by
  sorry

end base_5_minus_base_8_in_base_10_l223_223899


namespace stuart_initial_marbles_l223_223403

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l223_223403


namespace unattainable_value_of_y_l223_223531

noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

theorem unattainable_value_of_y :
  ∃ y : ℝ, y = -(1 / 3) ∧ ∀ x : ℝ, 3 * x + 4 ≠ 0 → f x ≠ y :=
by
  sorry

end unattainable_value_of_y_l223_223531


namespace probability_of_divisor_of_6_is_two_thirds_l223_223144

noncomputable def probability_divisor_of_6 : ℚ :=
  have divisors_of_6 : Finset ℕ := {1, 2, 3, 6}
  have total_possible_outcomes : ℕ := 6
  have favorable_outcomes : ℕ := 4
  have probability_event : ℚ := favorable_outcomes / total_possible_outcomes
  2 / 3

theorem probability_of_divisor_of_6_is_two_thirds :
  probability_divisor_of_6 = 2 / 3 :=
sorry

end probability_of_divisor_of_6_is_two_thirds_l223_223144


namespace avg_height_correct_l223_223912

theorem avg_height_correct (h1 h2 h3 h4 : ℝ) (h_distinct: h1 ≠ h2 ∧ h2 ≠ h3 ∧ h3 ≠ h4 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h4)
  (h_tallest: h4 = 152) (h_shortest: h1 = 137) 
  (h4_largest: h4 > h3 ∧ h4 > h2 ∧ h4 > h1) (h1_smallest: h1 < h2 ∧ h1 < h3 ∧ h1 < h4) :
  ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg := 
sorry

end avg_height_correct_l223_223912


namespace alexa_fractions_l223_223514

theorem alexa_fractions (alexa_days ethans_days : ℕ) 
  (h1 : alexa_days = 9) (h2 : ethans_days = 12) : 
  alexa_days / ethans_days = 3 / 4 := 
by 
  sorry

end alexa_fractions_l223_223514


namespace eval_64_pow_5_over_6_l223_223443

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l223_223443


namespace extrema_range_l223_223233

noncomputable def hasExtrema (a : ℝ) : Prop :=
  (4 * a^2 + 12 * a > 0)

theorem extrema_range (a : ℝ) : hasExtrema a ↔ (a < -3 ∨ a > 0) := sorry

end extrema_range_l223_223233


namespace inequality_proof_l223_223852

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by
  sorry

end inequality_proof_l223_223852


namespace largest_number_l223_223549

theorem largest_number (n : ℕ) (digits : List ℕ) (h_digits : ∀ d ∈ digits, d = 5 ∨ d = 3 ∨ d = 1) (h_sum : digits.sum = 15) : n = 555 :=
by
  sorry

end largest_number_l223_223549


namespace length_is_56_l223_223778

noncomputable def length_of_plot (b : ℝ) : ℝ := b + 12

theorem length_is_56 (b : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) (h_cost : cost_per_meter = 26.50) (h_total_cost : total_cost = 5300) (h_fencing : 26.50 * (4 * b + 24) = 5300) : length_of_plot b = 56 := 
by 
  sorry

end length_is_56_l223_223778


namespace moles_of_NaNO3_formed_l223_223038

/- 
  Define the reaction and given conditions.
  The following assumptions and definitions will directly come from the problem's conditions.
-/

/-- 
  Represents a chemical reaction: 1 molecule of AgNO3,
  1 molecule of NaOH producing 1 molecule of NaNO3 and 1 molecule of AgOH.
-/
def balanced_reaction (agNO3 naOH naNO3 agOH : ℕ) := agNO3 = 1 ∧ naOH = 1 ∧ naNO3 = 1 ∧ agOH = 1

/-- 
  Proves that the number of moles of NaNO3 formed is 1,
  given 1 mole of AgNO3 and 1 mole of NaOH.
-/
theorem moles_of_NaNO3_formed (agNO3 naOH naNO3 agOH : ℕ)
  (h : balanced_reaction agNO3 naOH naNO3 agOH) :
  naNO3 = 1 := 
by
  sorry  -- Proof will be added here later

end moles_of_NaNO3_formed_l223_223038


namespace n_squared_divisible_by_144_l223_223302

-- Definitions based on the conditions
variables (n k : ℕ)
def is_positive (n : ℕ) : Prop := n > 0
def largest_divisor_of_n_is_twelve (n : ℕ) : Prop := ∃ k, n = 12 * k
def divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

theorem n_squared_divisible_by_144
  (h1 : is_positive n)
  (h2 : largest_divisor_of_n_is_twelve n) :
  divisible_by (n * n) 144 :=
sorry

end n_squared_divisible_by_144_l223_223302


namespace find_m_n_l223_223368

theorem find_m_n (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_sol : (m + Real.sqrt n)^2 - 10 * (m + Real.sqrt n) + 1 = Real.sqrt (m + Real.sqrt n) * (m + Real.sqrt n + 1)) : m + n = 55 :=
sorry

end find_m_n_l223_223368


namespace fraction_of_fliers_sent_out_l223_223916

-- Definitions based on the conditions
def total_fliers : ℕ := 2500
def fliers_next_day : ℕ := 1500

-- Defining the fraction sent in the morning as x
variable (x : ℚ)

-- The remaining fliers after morning
def remaining_fliers_morning := (1 - x) * total_fliers

-- The remaining fliers after afternoon
def remaining_fliers_afternoon := remaining_fliers_morning - (1/4) * remaining_fliers_morning

-- The theorem statement
theorem fraction_of_fliers_sent_out :
  remaining_fliers_afternoon = fliers_next_day → x = 1/5 :=
sorry

end fraction_of_fliers_sent_out_l223_223916


namespace six_times_eightx_plus_tenpi_eq_fourP_l223_223375

variable {x : ℝ} {π P : ℝ}

theorem six_times_eightx_plus_tenpi_eq_fourP (h : 3 * (4 * x + 5 * π) = P) : 
    6 * (8 * x + 10 * π) = 4 * P :=
sorry

end six_times_eightx_plus_tenpi_eq_fourP_l223_223375


namespace inheritance_amount_l223_223494

theorem inheritance_amount (x : ℝ) (hx1 : 0.25 * x + 0.1 * x = 15000) : x = 42857 := 
by
  -- Proof omitted
  sorry

end inheritance_amount_l223_223494


namespace max_min_f_m1_possible_ns_l223_223299

noncomputable def f (a b : ℝ) (x : ℝ) (m : ℝ) : ℝ :=
  let a := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), -Real.sqrt 3)
  let b := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), Real.cos (2 * m * x))
  a.1 * b.1 + a.2 * b.2

theorem max_min_f_m1 (x : ℝ) (h₁ : x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  2 ≤ f (Real.sqrt 2) 1 x 1 ∧ f (Real.sqrt 2) 1 x 1 ≤ 3 :=
by
  sorry

theorem possible_ns (n : ℤ) (h₂ : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2017) ∧ f (Real.sqrt 2) ((n * Real.pi) / 2) x ((n * Real.pi) / 2) = 0) :
  n = 1 ∨ n = -1 :=
by
  sorry

end max_min_f_m1_possible_ns_l223_223299


namespace quadratic_inequality_solution_l223_223186

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_inequality_solution_l223_223186


namespace simplify_expression_l223_223405

theorem simplify_expression :
  (Real.sqrt 2 * 2 ^ (1 / 2 : ℝ) + 18 / 3 * 3 - 8 ^ (3 / 2 : ℝ)) = (20 - 16 * Real.sqrt 2) :=
by sorry

end simplify_expression_l223_223405


namespace geometric_sequence_property_l223_223023

-- Define the sequence and the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the main property we are considering
def given_property (a: ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n - 1) = (a n) ^ 2

-- State the theorem
theorem geometric_sequence_property {a : ℕ → ℝ} (n : ℕ) (hn : n ≥ 2) :
  (is_geometric_sequence a → given_property a n ∧ ∀ a, given_property a n → ¬ is_geometric_sequence a) := sorry

end geometric_sequence_property_l223_223023


namespace range_of_a_l223_223293

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, (x^2 - 4 * x) ∈ Set.Icc (-4 : ℝ) 32) →
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l223_223293


namespace first_digit_base5_of_312_is_2_l223_223183

theorem first_digit_base5_of_312_is_2 :
  ∃ d : ℕ, d = 2 ∧ (∀ n : ℕ, d * 5 ^ n ≤ 312 ∧ 312 < (d + 1) * 5 ^ n) :=
by
  sorry

end first_digit_base5_of_312_is_2_l223_223183


namespace smallest_among_5_neg7_0_neg53_l223_223035

-- Define the rational numbers involved as constants
def a : ℚ := 5
def b : ℚ := -7
def c : ℚ := 0
def d : ℚ := -5 / 3

-- Define the conditions as separate lemmas
lemma positive_greater_than_zero (x : ℚ) (hx : x > 0) : x > c := by sorry
lemma zero_greater_than_negative (x : ℚ) (hx : x < 0) : c > x := by sorry
lemma compare_negative_by_absolute_value (x y : ℚ) (hx : x < 0) (hy : y < 0) (habs : |x| > |y|) : x < y := by sorry

-- Prove the main assertion
theorem smallest_among_5_neg7_0_neg53 : 
    b < a ∧ b < c ∧ b < d := by
    -- Here we apply the defined conditions to show b is the smallest
    sorry

end smallest_among_5_neg7_0_neg53_l223_223035


namespace gcd_of_36_between_70_and_85_is_81_l223_223243

theorem gcd_of_36_between_70_and_85_is_81 {n : ℕ} (h1 : n ≥ 70) (h2 : n ≤ 85) (h3 : Nat.gcd 36 n = 9) : n = 81 :=
by
  -- proof
  sorry

end gcd_of_36_between_70_and_85_is_81_l223_223243


namespace fruit_shop_apples_l223_223289

-- Given conditions
def morning_fraction : ℚ := 3 / 10
def afternoon_fraction : ℚ := 4 / 10
def total_sold : ℕ := 140

-- Define the total number of apples and the resulting condition
def total_fraction_sold : ℚ := morning_fraction + afternoon_fraction

theorem fruit_shop_apples (A : ℕ) (h : total_fraction_sold * A = total_sold) : A = 200 := 
by sorry

end fruit_shop_apples_l223_223289


namespace jen_scored_more_l223_223797

def bryan_score : ℕ := 20
def total_points : ℕ := 35
def sammy_mistakes : ℕ := 7
def sammy_score : ℕ := total_points - sammy_mistakes
def jen_score : ℕ := sammy_score + 2

theorem jen_scored_more :
  jen_score - bryan_score = 10 := by
  -- Proof to be filled in
  sorry

end jen_scored_more_l223_223797


namespace hats_per_yard_of_velvet_l223_223414

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

end hats_per_yard_of_velvet_l223_223414


namespace chinese_team_wins_gold_l223_223096

noncomputable def prob_player_a_wins : ℚ := 3 / 7
noncomputable def prob_player_b_wins : ℚ := 1 / 4

theorem chinese_team_wins_gold : prob_player_a_wins + prob_player_b_wins = 19 / 28 := by
  sorry

end chinese_team_wins_gold_l223_223096


namespace parallelogram_base_l223_223408

theorem parallelogram_base
  (Area Height Base : ℕ)
  (h_area : Area = 120)
  (h_height : Height = 10)
  (h_area_eq : Area = Base * Height) :
  Base = 12 :=
by
  /- 
    We assume the conditions:
    1. Area = 120
    2. Height = 10
    3. Area = Base * Height 
    Then, we need to prove that Base = 12.
  -/
  sorry

end parallelogram_base_l223_223408


namespace last_four_digits_5_2011_l223_223394

theorem last_four_digits_5_2011 :
  (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_5_2011_l223_223394


namespace find_ratios_sum_l223_223046
noncomputable def Ana_biking_rate : ℝ := 8.6
noncomputable def Bob_biking_rate : ℝ := 6.2
noncomputable def CAO_biking_rate : ℝ := 5

variable (a b c : ℝ)

-- Conditions  
def Ana_distance := 2 * a + b + c = Ana_biking_rate
def Bob_distance := b + c = Bob_biking_rate
def Cao_distance := Real.sqrt (b^2 + c^2) = CAO_biking_rate

-- Main statement
theorem find_ratios_sum : 
  Ana_distance a b c ∧ 
  Bob_distance b c ∧ 
  Cao_distance b c →
  ∃ (p q r : ℕ), p + q + r = 37 ∧ Nat.gcd p q = 1 ∧ ((a / c) = p / r) ∧ ((b / c) = q / r) ∧ ((a / b) = p / q) :=
sorry

end find_ratios_sum_l223_223046


namespace find_sin_minus_cos_l223_223409

variable {a : ℝ}
variable {α : ℝ}

def point_of_angle (a : ℝ) (h : a < 0) := (3 * a, -4 * a)

theorem find_sin_minus_cos (a : ℝ) (h : a < 0) (ha : point_of_angle a h = (3 * a, -4 * a)) (sinα : ℝ) (cosα : ℝ) :
  sinα = 4 / 5 → cosα = -3 / 5 → sinα - cosα = 7 / 5 :=
by sorry

end find_sin_minus_cos_l223_223409


namespace train_speed_l223_223569

theorem train_speed (length time_speed: ℝ) (h1 : length = 400) (h2 : time_speed = 16) : length / time_speed = 25 := 
by
    sorry

end train_speed_l223_223569


namespace parametric_equations_solution_l223_223160

theorem parametric_equations_solution (t₁ t₂ : ℝ) : 
  (1 = 1 + 2 * t₁ ∧ 2 = 2 - 3 * t₁) ∧
  (-1 = 1 + 2 * t₂ ∧ 5 = 2 - 3 * t₂) ↔
  (t₁ = 0 ∧ t₂ = -1) :=
by
  sorry

end parametric_equations_solution_l223_223160


namespace find_m_given_solution_l223_223489

theorem find_m_given_solution (m x y : ℚ) (h₁ : x = 4) (h₂ : y = 3) (h₃ : m * x - y = 4) : m = 7 / 4 :=
by
  sorry

end find_m_given_solution_l223_223489


namespace range_of_a_l223_223933

variable (a : ℝ)

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a - 1) ^ x < (2 * a - 1) ^ y
def q (a : ℝ) : Prop := ∀ x : ℝ, 2 * a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (0 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a) :=
by
  sorry

end range_of_a_l223_223933


namespace equation_holds_l223_223826

variable (a b : ℝ)

theorem equation_holds : a^2 - b^2 - (-2 * b^2) = a^2 + b^2 :=
by sorry

end equation_holds_l223_223826


namespace winner_votes_percentage_l223_223115

-- Define the total votes as V
def total_votes (winner_votes : ℕ) (winning_margin : ℕ) : ℕ :=
  winner_votes + (winner_votes - winning_margin)

-- Define the percentage function
def percentage_of_votes (part : ℕ) (total : ℕ) : ℕ :=
  (part * 100) / total

-- Lean statement to prove the result
theorem winner_votes_percentage
  (winner_votes : ℕ)
  (winning_margin : ℕ)
  (H_winner_votes : winner_votes = 550)
  (H_winning_margin : winning_margin = 100) :
  percentage_of_votes winner_votes (total_votes winner_votes winning_margin) = 55 := by
  sorry

end winner_votes_percentage_l223_223115


namespace sport_vs_std_ratio_comparison_l223_223101

/-- Define the ratios for the standard formulation. -/
def std_flavor_syrup_ratio := 1 / 12
def std_flavor_water_ratio := 1 / 30

/-- Define the conditions for the sport formulation. -/
def sport_water := 15 -- ounces of water in the sport formulation
def sport_syrup := 1 -- ounce of corn syrup in the sport formulation

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation. -/
def sport_flavor_water_ratio := std_flavor_water_ratio / 2

/-- Calculate the amount of flavoring in the sport formulation. -/
def sport_flavor := sport_water * sport_flavor_water_ratio

/-- The ratio of flavoring to corn syrup in the sport formulation. -/
def sport_flavor_syrup_ratio := sport_flavor / sport_syrup

/-- The proof problem statement. -/
theorem sport_vs_std_ratio_comparison : sport_flavor_syrup_ratio = 3 * std_flavor_syrup_ratio := 
by
  -- proof would go here
  sorry

end sport_vs_std_ratio_comparison_l223_223101


namespace Zoe_siblings_l223_223153

structure Child where
  eyeColor : String
  hairColor : String
  height : String

def Emma : Child := { eyeColor := "Green", hairColor := "Red", height := "Tall" }
def Zoe : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Short" }
def Liam : Child := { eyeColor := "Green", hairColor := "Brown", height := "Short" }
def Noah : Child := { eyeColor := "Gray", hairColor := "Red", height := "Tall" }
def Mia : Child := { eyeColor := "Green", hairColor := "Red", height := "Short" }
def Lucas : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Tall" }

def sibling (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.height = c2.height

theorem Zoe_siblings : sibling Zoe Noah ∧ sibling Zoe Lucas ∧ ∃ x, sibling Noah x ∧ sibling Lucas x :=
by
  sorry

end Zoe_siblings_l223_223153


namespace find_b_for_smallest_c_l223_223059

theorem find_b_for_smallest_c (c b : ℝ) (h_c_pos : 0 < c) (h_b_pos : 0 < b)
  (polynomial_condition : ∀ x : ℝ, (x^4 - c*x^3 + b*x^2 - c*x + 1 = 0) → real) :
  c = 4 → b = 6 :=
by
  intros h_c_eq_4
  sorry

end find_b_for_smallest_c_l223_223059


namespace common_root_value_l223_223687

theorem common_root_value (p : ℝ) (hp : p > 0) : 
  (∃ x : ℝ, 3 * x ^ 2 - 4 * p * x + 9 = 0 ∧ x ^ 2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by {
  sorry
}

end common_root_value_l223_223687


namespace simplest_quadratic_radical_l223_223929
  
theorem simplest_quadratic_radical (A B C D: ℝ) 
  (hA : A = Real.sqrt 0.1) 
  (hB : B = Real.sqrt (-2)) 
  (hC : C = 3 * Real.sqrt 2) 
  (hD : D = -Real.sqrt 20) : C = 3 * Real.sqrt 2 :=
by
  have h1 : ∀ (x : ℝ), Real.sqrt x = Real.sqrt x := sorry
  sorry

end simplest_quadratic_radical_l223_223929


namespace sum_of_possible_radii_l223_223598

theorem sum_of_possible_radii :
  ∃ r1 r2 : ℝ, 
    (∀ r, (r - 5)^2 + r^2 = (r + 2)^2 → r = r1 ∨ r = r2) ∧ 
    r1 + r2 = 14 :=
sorry

end sum_of_possible_radii_l223_223598


namespace tickets_difference_l223_223987

-- Definitions of conditions
def tickets_won : Nat := 19
def tickets_for_toys : Nat := 12
def tickets_for_clothes : Nat := 7

-- Theorem statement: Prove that the difference between tickets used for toys and tickets used for clothes is 5
theorem tickets_difference : (tickets_for_toys - tickets_for_clothes = 5) := by
  sorry

end tickets_difference_l223_223987


namespace average_snowfall_per_hour_l223_223342

theorem average_snowfall_per_hour (total_snowfall : ℕ) (hours_per_week : ℕ) (total_snowfall_eq : total_snowfall = 210) (hours_per_week_eq : hours_per_week = 7 * 24) : 
  total_snowfall / hours_per_week = 5 / 4 :=
by
  -- skip the proof
  sorry

end average_snowfall_per_hour_l223_223342


namespace emily_second_round_points_l223_223862

theorem emily_second_round_points (P : ℤ)
  (first_round_points : ℤ := 16)
  (last_round_points_lost : ℤ := 48)
  (end_points : ℤ := 1)
  (points_equation : first_round_points + P - last_round_points_lost = end_points) :
  P = 33 :=
  by {
    sorry
  }

end emily_second_round_points_l223_223862


namespace probability_of_not_adjacent_to_edge_is_16_over_25_l223_223971

def total_squares : ℕ := 100
def perimeter_squares : ℕ := 36
def non_perimeter_squares : ℕ := total_squares - perimeter_squares
def probability_not_adjacent_to_edge : ℚ := non_perimeter_squares / total_squares

theorem probability_of_not_adjacent_to_edge_is_16_over_25 :
  probability_not_adjacent_to_edge = 16 / 25 := by
  sorry

end probability_of_not_adjacent_to_edge_is_16_over_25_l223_223971


namespace find_GQ_in_triangle_XYZ_l223_223907

noncomputable def GQ_in_triangle_XYZ_centroid : ℝ :=
  let XY := 13
  let XZ := 15
  let YZ := 24
  let centroid_ratio := 1 / 3
  let semi_perimeter := (XY + XZ + YZ) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - XY) * (semi_perimeter - XZ) * (semi_perimeter - YZ))
  let heightXR := (2 * area) / YZ
  (heightXR * centroid_ratio)

theorem find_GQ_in_triangle_XYZ :
  GQ_in_triangle_XYZ_centroid = 2.4 :=
sorry

end find_GQ_in_triangle_XYZ_l223_223907


namespace strawberry_jelly_amount_l223_223787

def totalJelly : ℕ := 6310
def blueberryJelly : ℕ := 4518
def strawberryJelly : ℕ := totalJelly - blueberryJelly

theorem strawberry_jelly_amount : strawberryJelly = 1792 := by
  rfl

end strawberry_jelly_amount_l223_223787


namespace joe_total_paint_used_l223_223876

-- Conditions
def initial_paint : ℕ := 360
def paint_first_week : ℕ := initial_paint * 1 / 4
def remaining_paint_after_first_week : ℕ := initial_paint - paint_first_week
def paint_second_week : ℕ := remaining_paint_after_first_week * 1 / 6

-- Theorem statement
theorem joe_total_paint_used : paint_first_week + paint_second_week = 135 := by
  sorry

end joe_total_paint_used_l223_223876


namespace cone_from_sector_l223_223008

def cone_can_be_formed (θ : ℝ) (r_sector : ℝ) (r_cone_base : ℝ) (l_slant_height : ℝ) : Prop :=
  θ = 270 ∧ r_sector = 12 ∧ ∃ L, L = θ / 360 * (2 * Real.pi * r_sector) ∧ 2 * Real.pi * r_cone_base = L ∧ l_slant_height = r_sector

theorem cone_from_sector (base_radius slant_height : ℝ) :
  cone_can_be_formed 270 12 base_radius slant_height ↔ base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l223_223008


namespace only_linear_equation_with_two_variables_l223_223129

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

end only_linear_equation_with_two_variables_l223_223129


namespace real_solutions_l223_223674

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end real_solutions_l223_223674


namespace find_f_of_3_l223_223567

-- Define the function f and its properties
variable {f : ℝ → ℝ}

-- Define the properties given in the problem
axiom f_mono_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_of_f_minus_exp : ∀ x : ℝ, f (f x - 2^x) = 3

-- The main theorem to prove
theorem find_f_of_3 : f 3 = 9 := 
sorry

end find_f_of_3_l223_223567


namespace solve_equation_l223_223434

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l223_223434


namespace find_P_plus_Q_l223_223088

theorem find_P_plus_Q (P Q : ℝ) (h : ∃ b c : ℝ, (x^2 + 3 * x + 4) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) : 
P + Q = 15 :=
by
  sorry

end find_P_plus_Q_l223_223088


namespace total_erasers_is_35_l223_223751

def Celine : ℕ := 10

def Gabriel : ℕ := Celine / 2

def Julian : ℕ := Celine * 2

def total_erasers : ℕ := Celine + Gabriel + Julian

theorem total_erasers_is_35 : total_erasers = 35 :=
  by
  sorry

end total_erasers_is_35_l223_223751


namespace remainder_when_7n_divided_by_4_l223_223722

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l223_223722


namespace circle_area_increase_l223_223347

theorem circle_area_increase (r : ℝ) :
  let A_initial := Real.pi * r^2
  let A_new := Real.pi * (2*r)^2
  let delta_A := A_new - A_initial
  let percentage_increase := (delta_A / A_initial) * 100
  percentage_increase = 300 := by
  sorry

end circle_area_increase_l223_223347


namespace difference_between_greatest_and_smallest_S_l223_223543

-- Conditions
def num_students := 47
def rows := 6
def columns := 8

-- The definition of position value calculation
def position_value (i j m n : ℕ) := i - m + (j - n)

-- The definition of S
def S (initial_empty final_empty : (ℕ × ℕ)) : ℤ :=
  let (i_empty, j_empty) := initial_empty
  let (i'_empty, j'_empty) := final_empty
  (i'_empty + j'_empty) - (i_empty + j_empty)

-- Main statement
theorem difference_between_greatest_and_smallest_S :
  let max_S := S (1, 1) (6, 8)
  let min_S := S (6, 8) (1, 1)
  max_S - min_S = 24 :=
sorry

end difference_between_greatest_and_smallest_S_l223_223543


namespace minimum_value_2a_plus_b_l223_223858

theorem minimum_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / (a + 1)) + (2 / (b - 2)) = 1 / 2) : 2 * a + b ≥ 16 := 
sorry

end minimum_value_2a_plus_b_l223_223858


namespace compute_gf3_l223_223590

def f (x : ℝ) : ℝ := x^3 - 3
def g (x : ℝ) : ℝ := 2 * x^2 - x + 4

theorem compute_gf3 : g (f 3) = 1132 := 
by 
  sorry

end compute_gf3_l223_223590


namespace min_likes_both_l223_223371

-- Definitions corresponding to the conditions
def total_people : ℕ := 200
def likes_beethoven : ℕ := 160
def likes_chopin : ℕ := 150

-- Problem statement to prove
theorem min_likes_both : ∃ x : ℕ, x = 110 ∧ x = likes_beethoven - (total_people - likes_chopin) := by
  sorry

end min_likes_both_l223_223371


namespace packet_a_weight_l223_223436

theorem packet_a_weight (A B C D E : ℕ) :
  A + B + C = 252 →
  A + B + C + D = 320 →
  E = D + 3 →
  B + C + D + E = 316 →
  A = 75 := by
  sorry

end packet_a_weight_l223_223436


namespace fries_remaining_time_l223_223564

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end fries_remaining_time_l223_223564


namespace remainder_of_powers_l223_223974

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l223_223974


namespace find_number_l223_223767

theorem find_number (x : ℚ) : (x + (-5/12) - (-5/2) = 1/3) → x = -7/4 :=
by
  sorry

end find_number_l223_223767


namespace vector_subtraction_magnitude_l223_223504

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition1 : Real := 3 -- |a|
def condition2 : Real := 2 -- |b|
def condition3 : Real := 4 -- |a + b|

-- Proving the statement
theorem vector_subtraction_magnitude (h1 : ‖a‖ = condition1) (h2 : ‖b‖ = condition2) (h3 : ‖a + b‖ = condition3) :
  ‖a - b‖ = Real.sqrt 10 :=
by
  sorry

end vector_subtraction_magnitude_l223_223504


namespace sum4_l223_223028

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := sorry
noncomputable def gamma : ℂ := sorry

axiom sum1 : alpha + beta + gamma = 1
axiom sum2 : alpha^2 + beta^2 + gamma^2 = 5
axiom sum3 : alpha^3 + beta^3 + gamma^3 = 9

theorem sum4 : alpha^4 + beta^4 + gamma^4 = 56 := by
  sorry

end sum4_l223_223028


namespace factor_polynomial_l223_223004

theorem factor_polynomial : 
  (∀ x : ℝ, (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)) :=
by
  intro x
  -- Left-hand side
  let lhs := x^2 + 6 * x + 9 - 64 * x^4
  -- Right-hand side after factorization
  let rhs := (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)
  -- Prove the equality
  show lhs = rhs
  sorry

end factor_polynomial_l223_223004


namespace rational_number_div_eq_l223_223748

theorem rational_number_div_eq :
  ∃ x : ℚ, (-2 : ℚ) / x = 8 ∧ x = -1 / 4 :=
by
  existsi (-1 / 4 : ℚ)
  sorry

end rational_number_div_eq_l223_223748


namespace quadratic_has_two_distinct_real_roots_l223_223397

variable {R : Type} [LinearOrderedField R]

theorem quadratic_has_two_distinct_real_roots (c d : R) :
  ∀ x : R, (x + c) * (x + d) - (2 * x + c + d) = 0 → 
  (x + c)^2 + 4 > 0 :=
by
  intros x h
  -- Proof (skipped)
  sorry

end quadratic_has_two_distinct_real_roots_l223_223397


namespace arithmetic_sequence_fifth_term_l223_223053

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15)
  (h2 : a + 10 * d = 18) : 
  a + 4 * d = 0 := 
sorry

end arithmetic_sequence_fifth_term_l223_223053


namespace neg_p_equivalent_to_forall_x2_ge_1_l223_223532

open Classical

variable {x : ℝ}

-- Definition of the original proposition p
def p : Prop := ∃ (x : ℝ), x^2 < 1

-- The negation of the proposition p
def not_p : Prop := ∀ (x : ℝ), x^2 ≥ 1

-- The theorem stating the equivalence
theorem neg_p_equivalent_to_forall_x2_ge_1 : ¬ p ↔ not_p := by
  sorry

end neg_p_equivalent_to_forall_x2_ge_1_l223_223532


namespace marta_hours_worked_l223_223216

-- Definitions of the conditions in Lean 4
def total_collected : ℕ := 240
def hourly_rate : ℕ := 10
def tips_collected : ℕ := 50
def work_earned : ℕ := total_collected - tips_collected

-- Goal: To prove the number of hours worked by Marta
theorem marta_hours_worked : work_earned / hourly_rate = 19 := by
  sorry

end marta_hours_worked_l223_223216


namespace min_value_of_3x_2y_l223_223306

noncomputable def min_value (x y: ℝ) : ℝ := 3 * x + 2 * y

theorem min_value_of_3x_2y (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y - x * y = 0) :
  min_value x y = 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_3x_2y_l223_223306


namespace find_m_l223_223167

theorem find_m (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((1/3 : ℝ) * x1^3 - 3 * x1 + m = 0) ∧ ((1/3 : ℝ) * x2^3 - 3 * x2 + m = 0)) ↔ (m = -2 * Real.sqrt 3 ∨ m = 2 * Real.sqrt 3) :=
sorry

end find_m_l223_223167


namespace problem_solution_l223_223155

theorem problem_solution :
  ∀ p q : ℝ, (3 * p ^ 2 - 5 * p - 21 = 0) → (3 * q ^ 2 - 5 * q - 21 = 0) →
  (9 * p ^ 3 - 9 * q ^ 3) * (p - q)⁻¹ = 88 :=
by 
  sorry

end problem_solution_l223_223155


namespace f_2009_is_one_l223_223179

   -- Define the properties of the function f
   variables (f : ℤ → ℤ)
   variable (h_even : ∀ x : ℤ, f x = f (-x))
   variable (h1 : f 1 = 1)
   variable (h2008 : f 2008 ≠ 1)
   variable (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b))

   -- Prove that f(2009) = 1
   theorem f_2009_is_one : f 2009 = 1 :=
   sorry
   
end f_2009_is_one_l223_223179


namespace solve_for_x_l223_223146

theorem solve_for_x (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 3 ^ (5 / 2) ∨ x = -3 ^ (5 / 2) :=
by
  sorry

end solve_for_x_l223_223146


namespace caterpillars_left_on_tree_l223_223195

-- Definitions based on conditions
def initialCaterpillars : ℕ := 14
def hatchedCaterpillars : ℕ := 4
def caterpillarsLeftToCocoon : ℕ := 8

-- The proof problem statement in Lean
theorem caterpillars_left_on_tree : initialCaterpillars + hatchedCaterpillars - caterpillarsLeftToCocoon = 10 :=
by
  -- solution steps will go here eventually
  sorry

end caterpillars_left_on_tree_l223_223195


namespace ab_bc_ca_fraction_l223_223480

theorem ab_bc_ca_fraction (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 10) (h3 : a * b * c = 12) :
    (a * b / c) + (b * c / a) + (c * a / b) = -17 / 3 := 
    sorry

end ab_bc_ca_fraction_l223_223480


namespace values_of_quadratic_expression_l223_223318

variable {x : ℝ}

theorem values_of_quadratic_expression (h : x^2 - 4 * x + 3 < 0) : 
  (8 < x^2 + 4 * x + 3) ∧ (x^2 + 4 * x + 3 < 24) :=
sorry

end values_of_quadratic_expression_l223_223318


namespace prairie_total_area_l223_223322

theorem prairie_total_area :
  let dust_covered := 64535
  let untouched := 522
  (dust_covered + untouched) = 65057 :=
by {
  let dust_covered := 64535
  let untouched := 522
  trivial
}

end prairie_total_area_l223_223322


namespace max_value_2xy_sqrt6_8yz2_l223_223849

theorem max_value_2xy_sqrt6_8yz2 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
sorry

end max_value_2xy_sqrt6_8yz2_l223_223849


namespace total_cost_l223_223435

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l223_223435


namespace apple_difference_l223_223577

def carla_apples : ℕ := 7
def tim_apples : ℕ := 1

theorem apple_difference : carla_apples - tim_apples = 6 := by
  sorry

end apple_difference_l223_223577


namespace solution1_solution2_l223_223336

noncomputable def problem1 : Prop :=
  ∃ (a b : ℤ), 
  (∃ (n : ℤ), 3*a - 14 = n ∧ a - 2 = n) ∧ 
  (b - 15 = -27) ∧ 
  a = 4 ∧ 
  b = -12 ∧ 
  (4*a + b = 4)

noncomputable def problem2 : Prop :=
  ∀ (a b : ℤ), 
  (a = 4) ∧ 
  (b = -12) → 
  (4*a + b = 4) → 
  (∃ n, n^2 = 4 ∧ (n = 2 ∨ n = -2))

theorem solution1 : problem1 := by { sorry }
theorem solution2 : problem2 := by { sorry }

end solution1_solution2_l223_223336


namespace total_marbles_l223_223338

/--
Some marbles in a bag are red and the rest are blue.
If one red marble is removed, then one-seventh of the remaining marbles are red.
If two blue marbles are removed instead of one red, then one-fifth of the remaining marbles are red.
Prove that the total number of marbles in the bag originally is 22.
-/
theorem total_marbles (r b : ℕ) (h1 : (r - 1) / (r + b - 1) = 1 / 7) (h2 : r / (r + b - 2) = 1 / 5) :
  r + b = 22 := by
  sorry

end total_marbles_l223_223338


namespace number_of_people_per_taxi_l223_223379

def num_people_in_each_taxi (x : ℕ) (cars taxis vans total : ℕ) : Prop :=
  (cars = 3 * 4) ∧ (vans = 2 * 5) ∧ (total = 58) ∧ (taxis = 6 * x) ∧ (cars + vans + taxis = total)

theorem number_of_people_per_taxi
  (x cars taxis vans total : ℕ)
  (h1 : cars = 3 * 4)
  (h2 : vans = 2 * 5)
  (h3 : total = 58)
  (h4 : taxis = 6 * x)
  (h5 : cars + vans + taxis = total) :
  x = 6 :=
by
  sorry

end number_of_people_per_taxi_l223_223379


namespace uma_income_l223_223010

theorem uma_income
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 5000)
  (h2 : 3 * x - 2 * y = 5000) :
  4 * x = 20000 :=
by
  sorry

end uma_income_l223_223010


namespace gcd_72_120_168_l223_223424

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
by
  sorry

end gcd_72_120_168_l223_223424


namespace value_of_b_cannot_form_arithmetic_sequence_l223_223866

theorem value_of_b 
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b > 0) :
  b = 5 * Real.sqrt 10 := 
sorry

theorem cannot_form_arithmetic_sequence 
  (d : ℝ)
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b = 5 * Real.sqrt 10) :
  ¬(∃ d, a1 + d = a2 ∧ a2 + d = a3) := 
sorry

end value_of_b_cannot_form_arithmetic_sequence_l223_223866


namespace leo_class_girls_l223_223017

theorem leo_class_girls (g b : ℕ) 
  (h_ratio : 3 * b = 4 * g) 
  (h_total : g + b = 35) : g = 15 := 
by
  sorry

end leo_class_girls_l223_223017


namespace arithmetic_example_l223_223450

theorem arithmetic_example : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end arithmetic_example_l223_223450


namespace smallest_p_l223_223121

theorem smallest_p (n p : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) (h3 : (n + p) % 10 = 0) : p = 1 := 
sorry

end smallest_p_l223_223121


namespace marie_tasks_finish_time_l223_223284

noncomputable def total_time (times : List ℕ) : ℕ :=
  times.foldr (· + ·) 0

theorem marie_tasks_finish_time :
  let task_times := [30, 40, 50, 60]
  let start_time := 8 * 60 -- Start time in minutes (8:00 AM)
  let end_time := start_time + total_time task_times
  end_time = 11 * 60 := -- 11:00 AM in minutes
by
  -- Add a placeholder for the proof
  sorry

end marie_tasks_finish_time_l223_223284


namespace math_proof_problem_l223_223016

-- Define the function and its properties
variable (f : ℝ → ℝ)
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity : ∀ x : ℝ, f (x + 1) = -f x
axiom increasing_on_interval : ∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y

-- Theorem statement expressing the questions and answers
theorem math_proof_problem :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧
  (f 2 = f 0) :=
by
  sorry

end math_proof_problem_l223_223016


namespace money_distribution_l223_223551

-- Declare the variables and the conditions as hypotheses
theorem money_distribution (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 40) :
  B + C = 340 :=
by
  sorry

end money_distribution_l223_223551


namespace calculate_expr1_calculate_expr2_l223_223793

/-- Statement 1: -5 * 3 - 8 / -2 = -11 -/
theorem calculate_expr1 : (-5) * 3 - 8 / -2 = -11 :=
by sorry

/-- Statement 2: (-1)^3 + (5 - (-3)^2) / 6 = -5/3 -/
theorem calculate_expr2 : (-1)^3 + (5 - (-3)^2) / 6 = -(5 / 3) :=
by sorry

end calculate_expr1_calculate_expr2_l223_223793


namespace arithmetic_sequence_sum_first_three_terms_l223_223488

theorem arithmetic_sequence_sum_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 4) (h5 : a 5 = 7) (h6 : a 6 = 10) : a 1 + a 2 + a 3 = -6 :=
sorry

end arithmetic_sequence_sum_first_three_terms_l223_223488


namespace minimum_distance_l223_223440

noncomputable def distance (M Q : ℝ × ℝ) : ℝ :=
  ( (M.1 - Q.1) ^ 2 + (M.2 - Q.2) ^ 2 ) ^ (1 / 2)

theorem minimum_distance (M : ℝ × ℝ) :
  ∃ Q : ℝ × ℝ, ( (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1 ) ∧ distance M Q = 1 :=
sorry

end minimum_distance_l223_223440


namespace smallest_divisible_by_2022_l223_223442

theorem smallest_divisible_by_2022 (n : ℕ) (N : ℕ) :
  (N = 20230110) ∧ (∃ k : ℕ, N = 2023 * 10^n + k) ∧ N % 2022 = 0 → 
  ∀ M: ℕ, (∃ m : ℕ, M = 2023 * 10^n + m) ∧ M % 2022 = 0 → N ≤ M :=
sorry

end smallest_divisible_by_2022_l223_223442


namespace sum_mod_16_l223_223271

theorem sum_mod_16 :
  (70 + 71 + 72 + 73 + 74 + 75 + 76 + 77) % 16 = 0 := 
by
  sorry

end sum_mod_16_l223_223271


namespace find_number_l223_223641

theorem find_number (x : ℝ) (h : x / 0.07 = 700) : x = 49 :=
sorry

end find_number_l223_223641


namespace ratio_distance_l223_223844

theorem ratio_distance
  (x : ℝ)
  (P : ℝ × ℝ)
  (hP_coords : P = (x, -9))
  (h_distance_y_axis : abs x = 18) :
  abs (-9) / abs x = 1 / 2 :=
by sorry

end ratio_distance_l223_223844


namespace train_length_l223_223354

theorem train_length 
  (speed_jogger_kmph : ℕ)
  (initial_distance_m : ℕ)
  (speed_train_kmph : ℕ)
  (pass_time_s : ℕ)
  (h_speed_jogger : speed_jogger_kmph = 9)
  (h_initial_distance : initial_distance_m = 230)
  (h_speed_train : speed_train_kmph = 45)
  (h_pass_time : pass_time_s = 35) : 
  ∃ length_train_m : ℕ, length_train_m = 580 := sorry

end train_length_l223_223354


namespace find_making_lines_parallel_l223_223649

theorem find_making_lines_parallel (m : ℝ) : 
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2 
  (line1_slope = line2_slope) ↔ (m = 1) := 
by
  -- definitions
  intros
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2
  -- equation for slopes to be equal
  have slope_equation : line1_slope = line2_slope ↔ (m = 1)
  sorry

  exact slope_equation

end find_making_lines_parallel_l223_223649


namespace shortest_chord_length_l223_223669

theorem shortest_chord_length 
  (C : ℝ → ℝ → Prop) 
  (l : ℝ → ℝ → ℝ → Prop) 
  (radius : ℝ) 
  (center_x center_y : ℝ) 
  (cx cy : ℝ) 
  (m : ℝ) :
  (∀ x y, C x y ↔ (x - 1)^2 + (y - 2)^2 = 25) →
  (∀ x y m, l x y m ↔ (2*m+1)*x + (m+1)*y - 7*m - 4 = 0) →
  center_x = 1 →
  center_y = 2 →
  radius = 5 →
  cx = 3 →
  cy = 1 →
  ∃ shortest_chord_length : ℝ, shortest_chord_length = 4 * Real.sqrt 5 := sorry

end shortest_chord_length_l223_223669


namespace remainder_of_x_mod_11_l223_223327

theorem remainder_of_x_mod_11 {x : ℤ} (h : x % 66 = 14) : x % 11 = 3 :=
sorry

end remainder_of_x_mod_11_l223_223327


namespace factorize_x_squared_plus_2x_l223_223769

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l223_223769


namespace max_volumes_on_fedor_shelf_l223_223486

theorem max_volumes_on_fedor_shelf 
  (S s1 s2 n : ℕ) 
  (h1 : S + s1 ≥ (n - 2) / 2) 
  (h2 : S + s2 < (n - 2) / 3) 
  : n = 12 := 
sorry

end max_volumes_on_fedor_shelf_l223_223486


namespace segment_distance_sum_l223_223098

theorem segment_distance_sum
  (AB_len : ℝ) (A'B'_len : ℝ) (D_midpoint : AB_len / 2 = 4)
  (D'_midpoint : A'B'_len / 2 = 6) (x : ℝ) (y : ℝ)
  (x_val : x = 3) :
  x + y = 10 :=
by sorry

end segment_distance_sum_l223_223098


namespace problem_statement_l223_223411

-- Definitions of propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x = 1
def q : Prop := ∀ x : ℝ, x^2 > 0

-- The proof problem
theorem problem_statement : ¬ (¬ p ∧ ¬ q) :=
by 
  -- sorry here indicates that actual proof is omitted
  sorry

end problem_statement_l223_223411


namespace family_travel_time_l223_223352

theorem family_travel_time (D : ℕ) (v1 v2 : ℕ) (d1 d2 : ℕ) (t1 t2 : ℕ) :
  D = 560 → 
  v1 = 35 → 
  v2 = 40 → 
  d1 = D / 2 →
  d2 = D / 2 →
  t1 = d1 / v1 →
  t2 = d2 / v2 → 
  t1 + t2 = 15 :=
by
  sorry

end family_travel_time_l223_223352


namespace distinct_zeros_abs_minus_one_l223_223539

theorem distinct_zeros_abs_minus_one : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁| - 1 = 0) ∧ (|x₂| - 1 = 0) := 
by
  sorry

end distinct_zeros_abs_minus_one_l223_223539


namespace joao_claudia_scores_l223_223791

theorem joao_claudia_scores (joao_score claudia_score total_score : ℕ) 
  (h1 : claudia_score = joao_score + 13)
  (h2 : total_score = joao_score + claudia_score)
  (h3 : 100 ≤ total_score ∧ total_score < 200) :
  joao_score = 68 ∧ claudia_score = 81 := by
  sorry

end joao_claudia_scores_l223_223791


namespace sum_of_numbers_l223_223806

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 :=
by
  sorry

end sum_of_numbers_l223_223806


namespace red_candies_remain_percentage_l223_223033

noncomputable def percent_red_candies_remain (N : ℝ) : ℝ :=
let total_initial_candies : ℝ := 5 * N
let green_candies_eat : ℝ := N
let remaining_after_green : ℝ := total_initial_candies - green_candies_eat

let half_orange_candies_eat : ℝ := N / 2
let remaining_after_half_orange : ℝ := remaining_after_green - half_orange_candies_eat

let half_all_remaining_candies_eat : ℝ := (N / 2) + (N / 4) + (N / 2) + (N / 2)
let remaining_after_half_all : ℝ := remaining_after_half_orange - half_all_remaining_candies_eat

let final_remaining_candies : ℝ := 0.32 * total_initial_candies
let candies_to_eat_finally : ℝ := remaining_after_half_all - final_remaining_candies
let each_color_final_eat : ℝ := candies_to_eat_finally / 2

let remaining_red_candies : ℝ := (N / 2) - each_color_final_eat

(remaining_red_candies / N) * 100

theorem red_candies_remain_percentage (N : ℝ) : percent_red_candies_remain N = 42.5 := by
  -- Proof skipped
  sorry

end red_candies_remain_percentage_l223_223033


namespace probability_of_shaded_section_l223_223706

theorem probability_of_shaded_section 
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (H1 : total_sections = 8)
  (H2 : shaded_sections = 4)
  : (shaded_sections / total_sections : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_shaded_section_l223_223706


namespace value_is_sqrt_5_over_3_l223_223658

noncomputable def findValue (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) : ℝ :=
  (x + y) / (x - y)

theorem value_is_sqrt_5_over_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) :
  findValue x y h1 h2 h3 = Real.sqrt (5 / 3) :=
sorry

end value_is_sqrt_5_over_3_l223_223658


namespace common_difference_of_common_terms_l223_223109

def sequence_a (n : ℕ) : ℕ := 4 * n - 3
def sequence_b (k : ℕ) : ℕ := 3 * k - 1

theorem common_difference_of_common_terms :
  ∃ (d : ℕ), (∀ (m : ℕ), 12 * m + 5 ∈ { x | ∃ (n k : ℕ), sequence_a n = x ∧ sequence_b k = x }) ∧ d = 12 := 
sorry

end common_difference_of_common_terms_l223_223109


namespace certainEvent_l223_223247

def scoopingTheMoonOutOfTheWaterMeansCertain : Prop :=
  ∀ (e : String), e = "scooping the moon out of the water" → (∀ (b : Bool), b = true)

theorem certainEvent (e : String) (h : e = "scooping the moon out of the water") : ∀ (b : Bool), b = true :=
  by
  sorry

end certainEvent_l223_223247


namespace difference_of_squares_l223_223251

theorem difference_of_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x y : ℤ, a = x^2 - y^2) ∨ 
  (∃ x y : ℤ, b = x^2 - y^2) ∨ 
  (∃ x y : ℤ, a + b = x^2 - y^2) :=
by
  sorry

end difference_of_squares_l223_223251


namespace ordering_of_exponentials_l223_223106

theorem ordering_of_exponentials :
  let A := 3^20
  let B := 6^10
  let C := 2^30
  B < A ∧ A < C :=
by
  -- Definitions and conditions
  have h1 : 6^10 = 3^10 * 2^10 := by sorry
  have h2 : 3^10 = 59049 := by sorry
  have h3 : 2^10 = 1024 := by sorry
  have h4 : 2^30 = (2^10)^3 := by sorry
  
  -- We know 3^20, 6^10, 2^30 by definition and conditions
  -- Comparison
  have h5 : 3^20 = (3^10)^2 := by sorry
  have h6 : 2^30 = 1024^3 := by sorry
  
  -- Combine to get results
  have h7 : (3^10)^2 > 6^10 := by sorry
  have h8 : 1024^3 > 6^10 := by sorry
  have h9 : 1024^3 > (3^10)^2 := by sorry

  exact ⟨h7, h9⟩

end ordering_of_exponentials_l223_223106


namespace workshop_worker_count_l223_223191

theorem workshop_worker_count (W T N : ℕ) (h1 : T = 7) (h2 : 8000 * W = 7 * 14000 + 6000 * N) (h3 : W = T + N) : W = 28 :=
by
  sorry

end workshop_worker_count_l223_223191


namespace complex_magnitude_addition_l223_223517

theorem complex_magnitude_addition :
  (Complex.abs (3 / 4 - 3 * Complex.I) + 5 / 12) = (9 * Real.sqrt 17 + 5) / 12 := 
  sorry

end complex_magnitude_addition_l223_223517


namespace gcd_13642_19236_34176_l223_223716

theorem gcd_13642_19236_34176 : Int.gcd (Int.gcd 13642 19236) 34176 = 2 := 
sorry

end gcd_13642_19236_34176_l223_223716


namespace point_equidistant_l223_223548

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def P : ℝ × ℝ × ℝ := (5, -3, 4)

theorem point_equidistant : dist A P = dist B P ∧ dist B P = dist C P ∧ dist C P = dist D P :=
by
  sorry

end point_equidistant_l223_223548


namespace part1_part2_l223_223201

def A (x : ℝ) : Prop := x < -3 ∨ x > 7
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def complement_R_A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 7

theorem part1 (m : ℝ) :
  (∀ x, complement_R_A x ∨ B m x → complement_R_A x) →
  m ≤ 4 :=
by
  sorry

theorem part2 (m : ℝ) (a b : ℝ) :
  (∀ x, complement_R_A x ∧ B m x ↔ (a ≤ x ∧ x ≤ b)) ∧ (b - a ≥ 1) →
  3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end part1_part2_l223_223201


namespace inequality_conditions_l223_223349

theorem inequality_conditions (x y z : ℝ) (h1 : y - x < 1.5 * abs x) (h2 : z = 2 * (y + x)) : 
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) :=
by
  sorry

end inequality_conditions_l223_223349


namespace teds_age_l223_223882

theorem teds_age (s t : ℕ) (h1 : t = 3 * s - 20) (h2 : t + s = 76) : t = 52 :=
by
  sorry

end teds_age_l223_223882


namespace geometric_series_sum_l223_223173

theorem geometric_series_sum :
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  (3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = S) → S = 765 :=
by
  -- conditions
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  have h : 3 * (1 - 2^n) / (1 - 2) = 765 := sorry
  sorry

end geometric_series_sum_l223_223173


namespace sum_of_ages_l223_223701

theorem sum_of_ages {a b c : ℕ} (h1 : a * b * c = 72) (h2 : b < a) (h3 : a < c) : a + b + c = 13 :=
sorry

end sum_of_ages_l223_223701


namespace negative_integer_reciprocal_of_d_l223_223448

def a : ℚ := 3
def b : ℚ := |1 / 3|
def c : ℚ := -2
def d : ℚ := -1 / 2

theorem negative_integer_reciprocal_of_d (h : d ≠ 0) : ∃ k : ℤ, (d⁻¹ : ℚ) = ↑k ∧ k < 0 :=
by
  sorry

end negative_integer_reciprocal_of_d_l223_223448


namespace sherman_drives_nine_hours_a_week_l223_223753

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l223_223753


namespace shaded_area_is_correct_l223_223241

def area_of_rectangle (l w : ℕ) : ℕ := l * w

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

def area_of_shaded_region : ℕ :=
  let length := 8
  let width := 4
  let area_rectangle := area_of_rectangle length width
  let area_triangle := area_of_triangle length width
  area_rectangle - area_triangle

theorem shaded_area_is_correct : area_of_shaded_region = 16 :=
by
  sorry

end shaded_area_is_correct_l223_223241


namespace sin_2B_sin_A_sin_C_eq_neg_7_over_8_l223_223675

theorem sin_2B_sin_A_sin_C_eq_neg_7_over_8
    (A B C : ℝ)
    (a b c : ℝ)
    (h1 : (2 * a + c) * Real.cos B + b * Real.cos C = 0)
    (h2 : 1/2 * a * c * Real.sin B = 15 * Real.sqrt 3)
    (h3 : a + b + c = 30) :
    (2 * Real.sin B * Real.cos B) / (Real.sin A + Real.sin C) = -7/8 := 
sorry

end sin_2B_sin_A_sin_C_eq_neg_7_over_8_l223_223675


namespace find_general_formula_l223_223043

theorem find_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h₀ : n > 0)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, S (n + 1) = 2 * S n + n + 1)
  (h₃ : ∀ n, S (n + 1) - S n = a (n + 1)) :
  a n = 2^n - 1 :=
sorry

end find_general_formula_l223_223043


namespace form_of_reasoning_is_wrong_l223_223613

-- Let's define the conditions
def some_rat_nums_are_proper_fractions : Prop :=
  ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

def integers_are_rational_numbers : Prop :=
  ∀ n : ℤ, ∃ q : ℚ, q = n

-- The major premise of the syllogism
def major_premise := some_rat_nums_are_proper_fractions

-- The minor premise of the syllogism
def minor_premise := integers_are_rational_numbers

-- The conclusion of the syllogism
def conclusion := ∀ n : ℤ, ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

-- We need to prove that the form of reasoning is wrong
theorem form_of_reasoning_is_wrong (H1 : major_premise) (H2 : minor_premise) : ¬ conclusion :=
by
  sorry -- proof to be filled in

end form_of_reasoning_is_wrong_l223_223613


namespace more_than_four_numbers_make_polynomial_prime_l223_223626

def polynomial (n : ℕ) : ℤ := n^3 - 10 * n^2 + 31 * n - 17

def is_prime (k : ℤ) : Prop :=
  k > 1 ∧ ∀ m : ℤ, m > 1 ∧ m < k → ¬ (m ∣ k)

theorem more_than_four_numbers_make_polynomial_prime :
  (∃ n1 n2 n3 n4 n5 : ℕ, 
    n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧ n5 > 0 ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ 
    n3 ≠ n4 ∧ n3 ≠ n5 ∧ 
    n4 ≠ n5 ∧ 
    is_prime (polynomial n1) ∧
    is_prime (polynomial n2) ∧
    is_prime (polynomial n3) ∧
    is_prime (polynomial n4) ∧
    is_prime (polynomial n5)) :=
sorry

end more_than_four_numbers_make_polynomial_prime_l223_223626


namespace find_xyz_l223_223639

open Complex

theorem find_xyz (a b c x y z : ℂ)
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0)
  (ha : a = (b + c) / (x + 1))
  (hb : b = (a + c) / (y + 1))
  (hc : c = (a + b) / (z + 1))
  (hxy_z_1 : x * y + x * z + y * z = 9)
  (hxy_z_2 : x + y + z = 5) :
  x * y * z = 13 := 
sorry

end find_xyz_l223_223639


namespace additional_people_needed_l223_223561

-- Definitions corresponding to the given conditions
def person_hours (n : ℕ) (t : ℕ) : ℕ := n * t
def initial_people : ℕ := 8
def initial_time : ℕ := 10
def total_person_hours := person_hours initial_people initial_time

-- Lean statement of the problem
theorem additional_people_needed (new_time : ℕ) (new_people : ℕ) : 
  new_time = 5 → person_hours new_people new_time = total_person_hours → new_people - initial_people = 8 :=
by
  intro h1 h2
  sorry

end additional_people_needed_l223_223561


namespace bruce_total_payment_l223_223924

def cost_of_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_of_mangoes (quantity rate : ℕ) : ℕ := quantity * rate

theorem bruce_total_payment : 
  cost_of_grapes 8 70 + cost_of_mangoes 11 55 = 1165 :=
by 
  sorry

end bruce_total_payment_l223_223924


namespace smallest_b_value_l223_223240

def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

def not_triangle (x y z : ℝ) : Prop :=
  ¬triangle_inequality x y z

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
    (h3 : not_triangle 2 a b) (h4 : not_triangle (1 / b) (1 / a) 1) :
    b >= 2 :=
by
  sorry

end smallest_b_value_l223_223240


namespace diesel_fuel_usage_l223_223737

theorem diesel_fuel_usage (weekly_spending : ℝ) (cost_per_gallon : ℝ) (weeks : ℝ) (result : ℝ): 
  weekly_spending = 36 → cost_per_gallon = 3 → weeks = 2 → result = 24 → 
  (weekly_spending / cost_per_gallon) * weeks = result :=
by
  intros
  sorry

end diesel_fuel_usage_l223_223737


namespace Meadowood_problem_l223_223333

theorem Meadowood_problem (s h : ℕ) : ¬(26 * s + 3 * h = 58) :=
sorry

end Meadowood_problem_l223_223333


namespace quadratic_functions_count_correct_even_functions_count_correct_l223_223495

def num_coefficients := 4
def valid_coefficients := [-1, 0, 1, 2]

def count_quadratic_functions : ℕ :=
  num_coefficients * num_coefficients * (num_coefficients - 1)

def count_even_functions : ℕ :=
  (num_coefficients - 1) * (num_coefficients - 2)

def total_quad_functions_correct : Prop := count_quadratic_functions = 18
def total_even_functions_correct : Prop := count_even_functions = 6

theorem quadratic_functions_count_correct : total_quad_functions_correct :=
by sorry

theorem even_functions_count_correct : total_even_functions_correct :=
by sorry

end quadratic_functions_count_correct_even_functions_count_correct_l223_223495


namespace area_difference_l223_223611

theorem area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15 / 2) :
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end area_difference_l223_223611


namespace initial_average_mark_of_class_l223_223461

theorem initial_average_mark_of_class
  (avg_excluded : ℝ) (n_excluded : ℕ) (avg_remaining : ℝ)
  (n_total : ℕ) : 
  avg_excluded = 70 → 
  n_excluded = 5 → 
  avg_remaining = 90 → 
  n_total = 10 → 
  (10 * (10 / n_total + avg_excluded - avg_remaining) / 10) = 80 :=
by 
  intros 
  sorry

end initial_average_mark_of_class_l223_223461


namespace soy_sauce_bottle_size_l223_223536

theorem soy_sauce_bottle_size 
  (ounces_per_cup : ℕ)
  (cups_recipe1 : ℕ)
  (cups_recipe2 : ℕ)
  (cups_recipe3 : ℕ)
  (number_of_bottles : ℕ)
  (total_ounces_needed : ℕ)
  (ounces_per_bottle : ℕ) :
  ounces_per_cup = 8 →
  cups_recipe1 = 2 →
  cups_recipe2 = 1 →
  cups_recipe3 = 3 →
  number_of_bottles = 3 →
  total_ounces_needed = (cups_recipe1 + cups_recipe2 + cups_recipe3) * ounces_per_cup →
  ounces_per_bottle = total_ounces_needed / number_of_bottles →
  ounces_per_bottle = 16 :=
by
  sorry

end soy_sauce_bottle_size_l223_223536


namespace max_value_of_xyz_l223_223261

noncomputable def max_product (x y z : ℝ) : ℝ :=
  x * y * z

theorem max_value_of_xyz (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x = y) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) (h6 : x ≤ z) (h7 : z ≤ 2 * x) :
  max_product x y z ≤ (1 / 27) := 
by
  sorry

end max_value_of_xyz_l223_223261


namespace problem_statement_l223_223760

variable {x y : Real}

theorem problem_statement (hx : x * y < 0) (hxy : x > |y|) : x + y > 0 := by
  sorry

end problem_statement_l223_223760


namespace find_sin_value_l223_223891

variable (x : ℝ)

theorem find_sin_value (h : Real.sin (x + Real.pi / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * Real.pi / 3 - x) = Real.sqrt 3 / 3 :=
by 
  sorry

end find_sin_value_l223_223891


namespace floor_sqrt_equality_l223_223485

theorem floor_sqrt_equality (n : ℕ) : 
  (Int.floor (Real.sqrt (4 * n + 1))) = (Int.floor (Real.sqrt (4 * n + 3))) := 
by 
  sorry

end floor_sqrt_equality_l223_223485


namespace original_square_area_l223_223304

theorem original_square_area (s : ℕ) (h1 : s + 5 = s + 5) (h2 : (s + 5)^2 = s^2 + 225) : s^2 = 400 :=
by
  sorry

end original_square_area_l223_223304


namespace quadratic_one_solution_m_l223_223733

theorem quadratic_one_solution_m (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 7 * x + m = 0) → 
  (∀ (x y : ℝ), 3 * x^2 - 7 * x + m = 0 → 3 * y^2 - 7 * y + m = 0 → x = y) → 
  m = 49 / 12 :=
by
  sorry

end quadratic_one_solution_m_l223_223733


namespace maximum_value_of_function_y_l223_223938

noncomputable def function_y (x : ℝ) : ℝ :=
  x * (3 - 2 * x)

theorem maximum_value_of_function_y : ∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ function_y x = 9 / 8 :=
by
  sorry

end maximum_value_of_function_y_l223_223938


namespace A_independent_of_beta_l223_223961

noncomputable def A (alpha beta : ℝ) : ℝ :=
  (Real.sin (alpha + beta) ^ 2) + (Real.sin (beta - alpha) ^ 2) - 
  2 * (Real.sin (alpha + beta)) * (Real.sin (beta - alpha)) * (Real.cos (2 * alpha))

theorem A_independent_of_beta (alpha beta : ℝ) : 
  ∃ (c : ℝ), ∀ beta : ℝ, A alpha beta = c :=
by
  sorry

end A_independent_of_beta_l223_223961


namespace find_explicit_formula_range_of_k_l223_223945

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 - b * x + 4

theorem find_explicit_formula (h_extremum_at_2 : f a b 2 = -4 / 3 ∧ (3 * a * 4 - b = 0)) :
  ∃ a b, f a b x = (1 / 3) * x ^ 3 - 4 * x + 4 :=
sorry

theorem range_of_k (h_extremum_at_2 : f (1 / 3) 4 2 = -4 / 3) :
  ∃ k, -4 / 3 < k ∧ k < 8 / 3 :=
sorry

end find_explicit_formula_range_of_k_l223_223945


namespace bottom_right_corner_value_l223_223182

variable (a b c x : ℕ)

/--
Conditions:
- The sums of the numbers in each of the four 2x2 grids forming part of the 3x3 grid are equal.
- Known values for corners: a, b, and c.
Conclusion:
- The bottom right corner value x must be 0.
-/

theorem bottom_right_corner_value (S: ℕ) (A B C D E: ℕ) :
  S = a + A + B + C →
  S = A + b + C + D →
  S = B + C + c + E →
  S = C + D + E + x →
  x = 0 :=
by
  sorry

end bottom_right_corner_value_l223_223182


namespace find_x_l223_223980

theorem find_x (x : ℝ) : (x * 16) / 100 = 0.051871999999999995 → x = 0.3242 := by
  intro h
  sorry

end find_x_l223_223980


namespace simplify_expression_l223_223781

theorem simplify_expression (t : ℝ) : (t ^ 5 * t ^ 3) / t ^ 2 = t ^ 6 :=
by
  sorry

end simplify_expression_l223_223781


namespace probability_of_first_hearts_and_second_clubs_l223_223389

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l223_223389


namespace system_solution_l223_223917

theorem system_solution (x y : ℝ) (h1 : x + y = 1) (h2 : x - y = 3) : x = 2 ∧ y = -1 :=
by
  sorry

end system_solution_l223_223917


namespace arithmetic_sequence_a1_a5_product_l223_223599

theorem arithmetic_sequence_a1_a5_product 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = 3) 
  (h_cond : (1 / a 1) + (1 / a 5) = 6 / 5) : 
  a 1 * a 5 = 5 := 
by
  sorry

end arithmetic_sequence_a1_a5_product_l223_223599


namespace find_m_l223_223455

theorem find_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := 
by
  sorry

end find_m_l223_223455


namespace triangle_third_side_max_length_l223_223570

theorem triangle_third_side_max_length (a b : ℕ) (ha : a = 5) (hb : b = 11) : ∃ (c : ℕ), c = 15 ∧ (a + c > b ∧ b + c > a ∧ a + b > c) :=
by 
  sorry

end triangle_third_side_max_length_l223_223570


namespace least_value_of_d_l223_223370

theorem least_value_of_d (c d : ℕ) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (hc_factors : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a ≠ b ∧ c = a * b) ∨ (∃ p : ℕ, p > 1 ∧ c = p^3))
  (hd_factors : ∃ factors : ℕ, factors = c ∧ ∃ divisors : Finset ℕ, divisors.card = factors ∧ ∀ k ∈ divisors, d % k = 0)
  (div_cd : d % c = 0) : d = 18 :=
sorry

end least_value_of_d_l223_223370


namespace find_marks_in_physics_l223_223218

theorem find_marks_in_physics (P C M : ℕ) (h1 : P + C + M = 225) (h2 : P + M = 180) (h3 : P + C = 140) : 
    P = 95 :=
sorry

end find_marks_in_physics_l223_223218


namespace sara_quarters_eq_l223_223995

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 21
def dad_quarters : ℕ := 49
def spent_quarters : ℕ := 15
def mom_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4
def amy_quarters (x : ℕ) := x

-- Define the function to compute total quarters
noncomputable def total_quarters (x : ℕ) : ℕ :=
initial_quarters + dad_quarters - spent_quarters + mom_dollars * quarters_per_dollar + amy_quarters x

-- Prove that the total number of quarters matches the expected value
theorem sara_quarters_eq (x : ℕ) : total_quarters x = 63 + x :=
by
  sorry

end sara_quarters_eq_l223_223995


namespace probability_X_eq_Y_correct_l223_223597

noncomputable def probability_X_eq_Y : ℝ :=
  let lower_bound := -20 * Real.pi
  let upper_bound := 20 * Real.pi
  let total_pairs := (upper_bound - lower_bound) * (upper_bound - lower_bound)
  let matching_pairs := 81
  matching_pairs / total_pairs

theorem probability_X_eq_Y_correct :
  probability_X_eq_Y = 81 / 1681 :=
by
  unfold probability_X_eq_Y
  sorry

end probability_X_eq_Y_correct_l223_223597


namespace average_rate_second_drive_l223_223756

theorem average_rate_second_drive 
 (distance : ℕ) (total_time : ℕ) (d1 d2 d3 : ℕ)
 (t1 t2 t3 : ℕ) (r1 r2 r3 : ℕ)
 (h_distance : d1 = d2 ∧ d2 = d3 ∧ d1 + d2 + d3 = distance)
 (h_total_time : t1 + t2 + t3 = total_time)
 (h_drive_1 : r1 = 4 ∧ t1 = d1 / r1)
 (h_drive_2 : r3 = 6 ∧ t3 = d3 / r3)
 (h_distance_total : distance = 180)
 (h_total_time_val : total_time = 37)
  : r2 = 5 := 
by sorry

end average_rate_second_drive_l223_223756


namespace James_average_speed_l223_223164

theorem James_average_speed (TotalDistance : ℝ) (BreakTime : ℝ) (TotalTripTime : ℝ) (h1 : TotalDistance = 42) (h2 : BreakTime = 1) (h3 : TotalTripTime = 9) :
  (TotalDistance / (TotalTripTime - BreakTime)) = 5.25 :=
by
  sorry

end James_average_speed_l223_223164


namespace linear_regression_intercept_l223_223702

theorem linear_regression_intercept :
  let x_values := [1, 2, 3, 4, 5]
  let y_values := [0.5, 0.8, 1.0, 1.2, 1.5]
  let x_mean := (x_values.sum / x_values.length : ℝ)
  let y_mean := (y_values.sum / y_values.length : ℝ)
  let slope := 0.24
  (x_mean = 3) →
  (y_mean = 1) →
  y_mean = slope * x_mean + 0.28 :=
by
  sorry

end linear_regression_intercept_l223_223702


namespace g_at_1001_l223_223926

open Function

variable (g : ℝ → ℝ)

axiom g_property : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_at_1 : g 1 = 3

theorem g_at_1001 : g 1001 = -997 :=
by
  sorry

end g_at_1001_l223_223926


namespace min_value_PA_PF_l223_223957

noncomputable def minimum_value_of_PA_and_PF_minimum 
  (x y : ℝ)
  (A : ℝ × ℝ)
  (F : ℝ × ℝ) : ℝ :=
  if ((A = (-1, 8)) ∧ (F = (0, 1)) ∧ (x^2 = 4 * y)) then 9 else 0

theorem min_value_PA_PF 
  (A : ℝ × ℝ := (-1, 8))
  (F : ℝ × ℝ := (0, 1))
  (P : ℝ × ℝ)
  (hP : P.1^2 = 4 * P.2) :
  minimum_value_of_PA_and_PF_minimum P.1 P.2 A F = 9 :=
by
  sorry

end min_value_PA_PF_l223_223957


namespace find_b_15_l223_223835

variable {a : ℕ → ℤ} (b : ℕ → ℤ) (S : ℕ → ℤ)

/-- An arithmetic sequence where S_n is the sum of the first n terms, with S_9 = -18 and S_13 = -52
   and a geometric sequence where b_5 = a_5 and b_7 = a_7. -/
theorem find_b_15 
  (h1 : S 9 = -18) 
  (h2 : S 13 = -52) 
  (h3 : b 5 = a 5) 
  (h4 : b 7 = a 7) 
  : b 15 = -64 := 
sorry

end find_b_15_l223_223835


namespace sum_min_values_eq_zero_l223_223050

-- Definitions of the polynomials
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c*x + d

-- Main theorem statement
theorem sum_min_values_eq_zero (b d : ℝ) :
  let a := -16
  let c := -8
  (-64 + b = 0) ∧ (-16 + d = 0) → (-64 + b + (-16 + d) = 0) :=
by
  intros
  rw [add_assoc]
  sorry

end sum_min_values_eq_zero_l223_223050


namespace twice_total_credits_l223_223771

theorem twice_total_credits (Aria Emily Spencer : ℕ) 
(Emily_has_20_credits : Emily = 20) 
(Aria_twice_Emily : Aria = 2 * Emily) 
(Emily_twice_Spencer : Emily = 2 * Spencer) : 
2 * (Aria + Emily + Spencer) = 140 :=
by
  sorry

end twice_total_credits_l223_223771


namespace conical_tank_volume_l223_223925

theorem conical_tank_volume
  (diameter : ℝ) (height : ℝ) (depth_linear : ∀ x : ℝ, 0 ≤ x ∧ x ≤ diameter / 2 → height - (height / (diameter / 2)) * x = 0) :
  diameter = 20 → height = 6 → (1 / 3) * Real.pi * (10 ^ 2) * height = 200 * Real.pi :=
by
  sorry

end conical_tank_volume_l223_223925


namespace chess_team_boys_count_l223_223449

theorem chess_team_boys_count (J S B : ℕ) 
  (h1 : J + S + B = 32) 
  (h2 : (1 / 3 : ℚ) * J + (1 / 2 : ℚ) * S + B = 18) : 
  B = 4 :=
by
  sorry

end chess_team_boys_count_l223_223449


namespace pages_revised_only_once_l223_223727

variable (x : ℕ)

def rate_first_time_typing := 6
def rate_revision := 4
def total_pages := 100
def pages_revised_twice := 15
def total_cost := 860

theorem pages_revised_only_once : 
  rate_first_time_typing * total_pages 
  + rate_revision * x 
  + rate_revision * pages_revised_twice * 2 
  = total_cost 
  → x = 35 :=
by
  sorry

end pages_revised_only_once_l223_223727


namespace exists_unique_integer_pair_l223_223966

theorem exists_unique_integer_pair (a : ℕ) (ha : 0 < a) :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x + (x + y - 1) * (x + y - 2) / 2 = a :=
by
  sorry

end exists_unique_integer_pair_l223_223966


namespace arithmetic_seq_third_sum_l223_223810

-- Define the arithmetic sequence using its first term and common difference
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * n

theorem arithmetic_seq_third_sum
  (a₁ d : ℤ)
  (h1 : (a₁ + (a₁ + 3 * d) + (a₁ + 6 * d) = 39))
  (h2 : ((a₁ + d) + (a₁ + 4 * d) + (a₁ + 7 * d) = 33)) :
  ((a₁ + 2 * d) + (a₁ + 5 * d) + (a₁ + 8 * d) = 27) :=
by
  sorry

end arithmetic_seq_third_sum_l223_223810


namespace area_of_square_l223_223464

noncomputable def square_area (s : ℝ) : ℝ := s ^ 2

theorem area_of_square
  {E F G H : Type}
  (ABCD : Type)
  (on_segments : E → F → G → H → Prop)
  (EG FH : ℝ)
  (angle_intersection : ℝ)
  (hEG : EG = 7)
  (hFH : FH = 8)
  (hangle : angle_intersection = 30) :
  ∃ s : ℝ, square_area s = 147 / 4 :=
sorry

end area_of_square_l223_223464


namespace patty_heavier_before_losing_weight_l223_223178

theorem patty_heavier_before_losing_weight {w_R w_P w_P' x : ℝ}
  (h1 : w_R = 100)
  (h2 : w_P = 100 * x)
  (h3 : w_P' = w_P - 235)
  (h4 : w_P' = w_R + 115) :
  x = 4.5 :=
by
  sorry

end patty_heavier_before_losing_weight_l223_223178


namespace part1_monotonic_intervals_part2_max_a_l223_223783

noncomputable def f1 (x : ℝ) := Real.log x - 2 * x^2

theorem part1_monotonic_intervals :
  (∀ x, 0 < x ∧ x < 0.5 → f1 x > 0) ∧ (∀ x, x > 0.5 → f1 x < 0) :=
by
  sorry

noncomputable def f2 (x a : ℝ) := Real.log x + a * x^2

theorem part2_max_a (a : ℤ) :
  (∀ x, x > 1 → f2 x a < Real.exp x) → a ≤ 1 :=
by
  sorry

end part1_monotonic_intervals_part2_max_a_l223_223783


namespace finitely_many_negative_terms_l223_223113

theorem finitely_many_negative_terms (A : ℝ) :
  (∀ (x : ℕ → ℝ), (∀ n, x n ≠ 0) ∧ (∀ n, x (n+1) = A - 1 / x n) →
  (∃ N, ∀ n ≥ N, x n ≥ 0)) ↔ A ≥ 2 :=
sorry

end finitely_many_negative_terms_l223_223113


namespace find_deductive_reasoning_l223_223221

noncomputable def is_deductive_reasoning (reasoning : String) : Prop :=
  match reasoning with
  | "B" => true
  | _ => false

theorem find_deductive_reasoning : is_deductive_reasoning "B" = true :=
  sorry

end find_deductive_reasoning_l223_223221


namespace fewer_people_third_bus_l223_223589

noncomputable def people_first_bus : Nat := 12
noncomputable def people_second_bus : Nat := 2 * people_first_bus
noncomputable def people_fourth_bus : Nat := people_first_bus + 9
noncomputable def total_people : Nat := 75
noncomputable def people_other_buses : Nat := people_first_bus + people_second_bus + people_fourth_bus
noncomputable def people_third_bus : Nat := total_people - people_other_buses

theorem fewer_people_third_bus :
  people_second_bus - people_third_bus = 6 :=
by
  sorry

end fewer_people_third_bus_l223_223589


namespace ThreeStudentsGotA_l223_223717

-- Definitions of students receiving A grades
variable (Edward Fiona George Hannah Ian : Prop)

-- Conditions given in the problem
axiom H1 : Edward → Fiona
axiom H2 : Fiona → George
axiom H3 : George → Hannah
axiom H4 : Hannah → Ian
axiom H5 : (Edward → False) ∧ (Fiona → False)

-- Theorem stating the final result
theorem ThreeStudentsGotA : (George ∧ Hannah ∧ Ian) ∧ 
                            (¬Edward ∧ ¬Fiona) ∧ 
                            (Edward ∨ Fiona ∨ George ∨ Hannah ∨ Ian) :=
by
  sorry

end ThreeStudentsGotA_l223_223717


namespace bradley_travel_time_l223_223069

theorem bradley_travel_time (T : ℕ) (h1 : T / 4 = 20) (h2 : T / 3 = 45) : T - 20 = 280 :=
by
  -- Placeholder for proof
  sorry

end bradley_travel_time_l223_223069


namespace percentage_calculation_l223_223710

theorem percentage_calculation 
  (number : ℝ)
  (h1 : 0.035 * number = 700) :
  0.024 * (1.5 * number) = 720 := 
by
  sorry

end percentage_calculation_l223_223710


namespace ways_to_turn_off_lights_l223_223684

-- Define the problem conditions
def streetlights := 12
def can_turn_off := 3
def not_turn_off_at_ends := true
def not_adjacent := true

-- The theorem to be proved
theorem ways_to_turn_off_lights : 
  ∃ n, 
  streetlights = 12 ∧ 
  can_turn_off = 3 ∧ 
  not_turn_off_at_ends ∧ 
  not_adjacent ∧ 
  n = 56 :=
by 
  sorry

end ways_to_turn_off_lights_l223_223684


namespace candy_probability_difference_l223_223803

theorem candy_probability_difference :
  let total := 2004
  let total_ways := Nat.choose total 2
  let different_ways := 2002 * 1002 / 2
  let same_ways := 1002 * 1001 / 2 + 1002 * 1001 / 2
  let q := (different_ways : ℚ) / total_ways
  let p := (same_ways : ℚ) / total_ways
  q - p = 1 / 2003 :=
by sorry

end candy_probability_difference_l223_223803


namespace minimum_gb_for_cheaper_plan_l223_223975

theorem minimum_gb_for_cheaper_plan : ∃ g : ℕ, (g ≥ 778) ∧ 
  (∀ g' < 778, 3000 + (if g' ≤ 500 then 8 * g' else 8 * 500 + 6 * (g' - 500)) ≥ 15 * g') ∧ 
  3000 + (if g ≤ 500 then 8 * g else 8 * 500 + 6 * (g - 500)) < 15 * g :=
by
  sorry

end minimum_gb_for_cheaper_plan_l223_223975


namespace subset_strict_M_P_l223_223960

-- Define the set M
def M : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}

-- Define the set P
def P : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

-- Prove that M is strictly a subset of P
theorem subset_strict_M_P : M ⊆ P ∧ ∃ x ∈ P, x ∉ M :=
by
  sorry

end subset_strict_M_P_l223_223960


namespace rectangle_area_l223_223934

theorem rectangle_area (length_of_rectangle radius_of_circle side_of_square : ℝ)
  (h1 : length_of_rectangle = (2 / 5) * radius_of_circle)
  (h2 : radius_of_circle = side_of_square)
  (h3 : side_of_square * side_of_square = 1225)
  (breadth_of_rectangle : ℝ)
  (h4 : breadth_of_rectangle = 10) : 
  length_of_rectangle * breadth_of_rectangle = 140 := 
by 
  sorry

end rectangle_area_l223_223934


namespace scholars_number_l223_223048

theorem scholars_number (n : ℕ) : n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 → n = 509 :=
by
  intro h
  sorry

end scholars_number_l223_223048


namespace total_marble_weight_l223_223872

theorem total_marble_weight (w1 w2 w3 : ℝ) (h_w1 : w1 = 0.33) (h_w2 : w2 = 0.33) (h_w3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 :=
by {
  sorry
}

end total_marble_weight_l223_223872


namespace amount_of_silver_l223_223031

-- Definitions
def total_silver (x : ℕ) : Prop :=
  (x - 4) % 7 = 0 ∧ (x + 8) % 9 = 1

-- Theorem to be proven
theorem amount_of_silver (x : ℕ) (h : total_silver x) : (x - 4)/7 = (x + 8)/9 :=
by sorry

end amount_of_silver_l223_223031


namespace area_of_paper_l223_223922

-- Define the variables and conditions
variable (L W : ℝ)
variable (h1 : 2 * L + 4 * W = 34)
variable (h2 : 4 * L + 2 * W = 38)

-- Statement to prove
theorem area_of_paper : L * W = 35 := 
by
  sorry

end area_of_paper_l223_223922


namespace roots_of_polynomial_l223_223799

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l223_223799


namespace stockings_total_cost_l223_223542

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end stockings_total_cost_l223_223542


namespace pipe_fill_rate_l223_223317

theorem pipe_fill_rate 
  (C : ℝ) (t : ℝ) (capacity : C = 4000) (time_to_fill : t = 300) :
  (3/4 * C / t) = 10 := 
by 
  sorry

end pipe_fill_rate_l223_223317


namespace vacuum_total_time_l223_223996

theorem vacuum_total_time (x : ℕ) (hx : 2 * x + 5 = 27) :
  27 + x = 38 :=
by
  sorry

end vacuum_total_time_l223_223996


namespace log_inequality_l223_223896

theorem log_inequality {a x : ℝ} (h1 : 0 < x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1) : 
  abs (Real.logb a (1 - x)) > abs (Real.logb a (1 + x)) :=
sorry

end log_inequality_l223_223896


namespace no_int_sol_eq_l223_223075

theorem no_int_sol_eq (x y z : ℤ) (h₀ : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : ¬ (x^2 + y^2 = 3 * z^2) := 
sorry

end no_int_sol_eq_l223_223075


namespace max_value_of_exp_diff_l223_223719

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end max_value_of_exp_diff_l223_223719


namespace length_of_real_axis_of_hyperbola_l223_223296

theorem length_of_real_axis_of_hyperbola :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 -> ∃ a : ℝ, 2 * a = 4 :=
by
intro x y h
sorry

end length_of_real_axis_of_hyperbola_l223_223296


namespace exists_third_degree_poly_with_positive_and_negative_roots_l223_223331

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end exists_third_degree_poly_with_positive_and_negative_roots_l223_223331


namespace linear_relationship_correct_profit_160_max_profit_l223_223392

-- Define the conditions for the problem
def data_points : List (ℝ × ℝ) := [(3.5, 280), (5.5, 120)]

-- The linear function relationship between y and x
def linear_relationship (x : ℝ) : ℝ := -80 * x + 560

-- The equation for profit, given selling price and sales quantity
def profit (x : ℝ) : ℝ := (x - 3) * (linear_relationship x) - 80

-- Prove the relationship y = -80x + 560 from given data points
theorem linear_relationship_correct : 
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = linear_relationship x :=
sorry

-- Prove the selling price x = 4 results in a profit of $160 per day
theorem profit_160 (x : ℝ) (h : profit x = 160) : x = 4 :=
sorry

-- Prove the maximum profit and corresponding selling price
theorem max_profit : 
  ∃ x : ℝ, ∃ w : ℝ, 3.5 ≤ x ∧ x ≤ 5.5 ∧ profit x = w ∧ ∀ y, 3.5 ≤ y ∧ y ≤ 5.5 → profit y ≤ w ∧ w = 240 ∧ x = 5 :=
sorry

end linear_relationship_correct_profit_160_max_profit_l223_223392


namespace compute_expression_l223_223874

theorem compute_expression : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end compute_expression_l223_223874


namespace platform_length_proof_l223_223148

noncomputable def train_length : ℝ := 480

noncomputable def speed_kmph : ℝ := 55

noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

noncomputable def crossing_time : ℝ := 71.99424046076314

noncomputable def total_distance_covered : ℝ := speed_mps * crossing_time

noncomputable def platform_length : ℝ := total_distance_covered - train_length

theorem platform_length_proof : platform_length = 620 := by
  sorry

end platform_length_proof_l223_223148


namespace animath_interns_pigeonhole_l223_223648

theorem animath_interns_pigeonhole (n : ℕ) (knows : Fin n → Finset (Fin n)) :
  ∃ (i j : Fin n), i ≠ j ∧ (knows i).card = (knows j).card :=
by
  sorry

end animath_interns_pigeonhole_l223_223648


namespace geometric_sequence_s6_s4_l223_223198

section GeometricSequence

variables {a : ℕ → ℝ} {a1 : ℝ} {q : ℝ}
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = a1 * (1 - q^(n + 1)) / (1 - q))
variable (h_ratio : S 4 / S 2 = 3)

theorem geometric_sequence_s6_s4 :
  S 6 / S 4 = 7 / 3 :=
sorry

end GeometricSequence

end geometric_sequence_s6_s4_l223_223198


namespace Barons_theorem_correct_l223_223736

theorem Barons_theorem_correct (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ ∃ k1 k2 : ℕ, an = k1 ^ 2 ∧ bn = k2 ^ 3 := 
sorry

end Barons_theorem_correct_l223_223736


namespace children_got_off_bus_l223_223953

-- Conditions
def original_number_of_children : ℕ := 43
def children_left_on_bus : ℕ := 21

-- Definition of the number of children who got off the bus
def children_got_off : ℕ := original_number_of_children - children_left_on_bus

-- Theorem stating the number of children who got off the bus
theorem children_got_off_bus : children_got_off = 22 :=
by
  -- This is to indicate where the proof would go
  sorry

end children_got_off_bus_l223_223953


namespace find_b_of_quadratic_eq_l223_223492

theorem find_b_of_quadratic_eq (a b c y1 y2 : ℝ) 
    (h1 : y1 = a * (2:ℝ)^2 + b * (2:ℝ) + c) 
    (h2 : y2 = a * (-2:ℝ)^2 + b * (-2:ℝ) + c) 
    (h_diff : y1 - y2 = 4) : b = 1 :=
by
  sorry

end find_b_of_quadratic_eq_l223_223492


namespace remainder_when_divided_by_x_minus_2_l223_223643

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + x^2 + 4

theorem remainder_when_divided_by_x_minus_2 : f 2 = 56 :=
by
  -- Proof steps will go here.
  sorry

end remainder_when_divided_by_x_minus_2_l223_223643


namespace vector_coordinates_l223_223671

-- Define the given vectors.
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

-- Define the proof goal.
theorem vector_coordinates :
  -2 • a - b = (-3, -1) :=
by
  sorry -- Proof not required.

end vector_coordinates_l223_223671


namespace boxes_with_neither_l223_223520

-- Definitions based on the conditions given
def total_boxes : Nat := 12
def boxes_with_markers : Nat := 8
def boxes_with_erasers : Nat := 5
def boxes_with_both : Nat := 4

-- The statement we want to prove
theorem boxes_with_neither :
  total_boxes - (boxes_with_markers + boxes_with_erasers - boxes_with_both) = 3 :=
by
  sorry

end boxes_with_neither_l223_223520


namespace arithmetic_mean_eq_2_l223_223812

theorem arithmetic_mean_eq_2 (a x : ℝ) (hx: x ≠ 0) :
  (1/2) * (((2 * x + a) / x) + ((2 * x - a) / x)) = 2 :=
by
  sorry

end arithmetic_mean_eq_2_l223_223812


namespace identity_holds_for_all_a_b_l223_223591

theorem identity_holds_for_all_a_b (a b : ℝ) :
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  x^2 + y^2 = 2 * (z^2 + t^2) :=
by {
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  sorry
}

end identity_holds_for_all_a_b_l223_223591


namespace two_times_sum_of_fourth_power_is_perfect_square_l223_223116

theorem two_times_sum_of_fourth_power_is_perfect_square (a b c : ℤ) 
  (h : a + b + c = 0) : 2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := 
by sorry

end two_times_sum_of_fourth_power_is_perfect_square_l223_223116


namespace face_opposite_A_l223_223335
noncomputable def cube_faces : List String := ["A", "B", "C", "D", "E", "F"]

theorem face_opposite_A (cube_faces : List String) 
  (h1 : cube_faces.length = 6)
  (h2 : "A" ∈ cube_faces) 
  (h3 : "B" ∈ cube_faces)
  (h4 : "C" ∈ cube_faces) 
  (h5 : "D" ∈ cube_faces)
  (h6 : "E" ∈ cube_faces) 
  (h7 : "F" ∈ cube_faces)
  : ("D" ≠ "A") := 
by
  sorry

end face_opposite_A_l223_223335


namespace tens_digit_of_expression_l223_223125

theorem tens_digit_of_expression :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 1 :=
by sorry

end tens_digit_of_expression_l223_223125


namespace monotonic_f_deriv_nonneg_l223_223364

theorem monotonic_f_deriv_nonneg (k : ℝ) :
  (∀ x : ℝ, (1 / 2) < x → k - 1 / x ≥ 0) ↔ k ≥ 2 :=
by sorry

end monotonic_f_deriv_nonneg_l223_223364


namespace shoes_sold_first_week_eq_100k_l223_223999

-- Define variables for purchase price and total revenue
def purchase_price : ℝ := 180
def total_revenue : ℝ := 216

-- Define markups
def first_week_markup : ℝ := 1.25
def remaining_markup : ℝ := 1.16

-- Define the conditions
theorem shoes_sold_first_week_eq_100k (x y : ℝ) 
  (h1 : x + y = purchase_price) 
  (h2 : first_week_markup * x + remaining_markup * y = total_revenue) :
  first_week_markup * x = 100  := 
sorry

end shoes_sold_first_week_eq_100k_l223_223999


namespace min_sum_abc_l223_223402

theorem min_sum_abc (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hprod : a * b * c = 2550) : a + b + c ≥ 48 :=
by sorry

end min_sum_abc_l223_223402


namespace find_f_13_l223_223320

noncomputable def f : ℕ → ℕ :=
  sorry

axiom condition1 (x : ℕ) : f (x + f x) = 3 * f x
axiom condition2 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
  sorry

end find_f_13_l223_223320


namespace find_n_square_divides_exponential_plus_one_l223_223277

theorem find_n_square_divides_exponential_plus_one :
  ∀ n : ℕ, (n^2 ∣ 2^n + 1) → (n = 1) :=
by
  sorry

end find_n_square_divides_exponential_plus_one_l223_223277


namespace max_distance_from_point_to_line_l223_223851

theorem max_distance_from_point_to_line (θ m : ℝ) :
  let P := (Real.cos θ, Real.sin θ)
  let d := (P.1 - m * P.2 - 2) / Real.sqrt (1 + m^2)
  ∃ (θ m : ℝ), d ≤ 3 := sorry

end max_distance_from_point_to_line_l223_223851


namespace angles_of_terminal_side_on_line_y_equals_x_l223_223042

noncomputable def set_of_angles_on_y_equals_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 180 + 45

theorem angles_of_terminal_side_on_line_y_equals_x (α : ℝ) :
  (∃ k : ℤ, α = k * 360 + 45) ∨ (∃ k : ℤ, α = k * 360 + 225) ↔ set_of_angles_on_y_equals_x α :=
by
  sorry

end angles_of_terminal_side_on_line_y_equals_x_l223_223042


namespace not_divisible_by_3_or_4_l223_223847

theorem not_divisible_by_3_or_4 (n : ℤ) : 
  ¬ (n^2 + 1) % 3 = 0 ∧ ¬ (n^2 + 1) % 4 = 0 := 
by
  sorry

end not_divisible_by_3_or_4_l223_223847


namespace solve_for_x_l223_223351

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem solve_for_x :
  F x 3 2 = F x 2 5 → x = 21/19 :=
  by
  sorry

end solve_for_x_l223_223351


namespace remainder_of_98_mult_102_div_12_l223_223353

theorem remainder_of_98_mult_102_div_12 : (98 * 102) % 12 = 0 := by
    sorry

end remainder_of_98_mult_102_div_12_l223_223353


namespace lillian_candies_addition_l223_223487

noncomputable def lillian_initial_candies : ℕ := 88
noncomputable def lillian_father_candies : ℕ := 5
noncomputable def lillian_total_candies : ℕ := 93

theorem lillian_candies_addition : lillian_initial_candies + lillian_father_candies = lillian_total_candies := by
  sorry

end lillian_candies_addition_l223_223487


namespace jose_share_of_profit_l223_223700

def investment_months (amount : ℕ) (months : ℕ) : ℕ := amount * months

def profit_share (investment_months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment_months * total_profit) / total_investment_months

theorem jose_share_of_profit :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 36000
  let tom_months := 12
  let jose_months := 10
  let tom_investment_months := investment_months tom_investment tom_months
  let jose_investment_months := investment_months jose_investment jose_months
  let total_investment_months := tom_investment_months + jose_investment_months
  profit_share jose_investment_months total_investment_months total_profit = 20000 :=
by
  sorry

end jose_share_of_profit_l223_223700


namespace all_numbers_appear_on_diagonal_l223_223650

theorem all_numbers_appear_on_diagonal 
  (n : ℕ) 
  (h_odd : n % 2 = 1)
  (A : Matrix (Fin n) (Fin n) (Fin n.succ))
  (h_elements : ∀ i j, 1 ≤ A i j ∧ A i j ≤ n) 
  (h_unique_row : ∀ i k, ∃! j, A i j = k)
  (h_unique_col : ∀ j k, ∃! i, A i j = k)
  (h_symmetric : ∀ i j, A i j = A j i)
  : ∀ k, 1 ≤ k ∧ k ≤ n → ∃ i, A i i = k := 
by {
  sorry
}

end all_numbers_appear_on_diagonal_l223_223650


namespace sequence_value_2016_l223_223120

theorem sequence_value_2016 :
  ∀ (a : ℕ → ℤ),
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
    a 2016 = -3 :=
by
  sorry

end sequence_value_2016_l223_223120


namespace Problem_l223_223473

theorem Problem (N : ℕ) (hn : N = 16) :
  (Nat.choose N 5) = 2002 := 
by 
  rw [hn] 
  sorry

end Problem_l223_223473


namespace solution_exists_in_interval_l223_223762

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 3

theorem solution_exists_in_interval : ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by {
  -- placeholder for the skipped proof
  sorry
}

end solution_exists_in_interval_l223_223762


namespace lemon_heads_each_person_l223_223355

-- Define the constants used in the problem
def totalLemonHeads : Nat := 72
def numberOfFriends : Nat := 6

-- The theorem stating the problem and the correct answer
theorem lemon_heads_each_person :
  totalLemonHeads / numberOfFriends = 12 := 
by
  sorry

end lemon_heads_each_person_l223_223355


namespace julie_initial_savings_l223_223713

def calculate_earnings (lawns newspapers dogs : ℕ) (price_lawn price_newspaper price_dog : ℝ) : ℝ :=
  (lawns * price_lawn) + (newspapers * price_newspaper) + (dogs * price_dog)

def calculate_total_spent_bike (earnings remaining_money : ℝ) : ℝ :=
  earnings + remaining_money

def calculate_initial_savings (cost_bike total_spent : ℝ) : ℝ :=
  cost_bike - total_spent

theorem julie_initial_savings :
  let cost_bike := 2345
  let lawns := 20
  let newspapers := 600
  let dogs := 24
  let price_lawn := 20
  let price_newspaper := 0.40
  let price_dog := 15
  let remaining_money := 155
  let earnings := calculate_earnings lawns newspapers dogs price_lawn price_newspaper price_dog
  let total_spent := calculate_total_spent_bike earnings remaining_money
  calculate_initial_savings cost_bike total_spent = 1190 :=
by
  -- Although the proof is not required, this setup assumes correctness.
  sorry

end julie_initial_savings_l223_223713


namespace school_allocation_methods_l223_223039

-- Define the conditions
def doctors : ℕ := 3
def nurses : ℕ := 6
def schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- The combinatorial function for binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Verify the number of allocation methods
theorem school_allocation_methods : 
  C doctors doctors_per_school * C nurses nurses_per_school *
  C (doctors - 1) doctors_per_school * C (nurses - 2) nurses_per_school *
  C (doctors - 2) doctors_per_school * C (nurses - 4) nurses_per_school = 540 := 
sorry

end school_allocation_methods_l223_223039


namespace barn_painting_total_area_l223_223398

theorem barn_painting_total_area :
  let width := 12
  let length := 15
  let height := 5
  let divider_width := 12
  let divider_height := 5

  let external_wall_area := 2 * (width * height + length * height)
  let dividing_wall_area := 2 * (divider_width * divider_height)
  let ceiling_area := width * length
  let total_area := 2 * external_wall_area + dividing_wall_area + ceiling_area

  total_area = 840 := by
    sorry

end barn_painting_total_area_l223_223398


namespace percent_problem_l223_223312

theorem percent_problem (x : ℝ) (h : 0.35 * 400 = 0.20 * x) : x = 700 :=
by sorry

end percent_problem_l223_223312


namespace largest_prime_divisor_for_primality_check_l223_223310

theorem largest_prime_divisor_for_primality_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : 
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1050 ∧ ∀ q, Prime q → q ≤ Int.sqrt n → q ≤ p := sorry

end largest_prime_divisor_for_primality_check_l223_223310


namespace mrs_hilt_total_distance_l223_223824

-- Define the distances and number of trips
def distance_to_water_fountain := 30
def distance_to_staff_lounge := 45
def trips_to_water_fountain := 4
def trips_to_staff_lounge := 3

-- Calculate the total distance for Mrs. Hilt's trips
def total_distance := (distance_to_water_fountain * 2 * trips_to_water_fountain) + 
                      (distance_to_staff_lounge * 2 * trips_to_staff_lounge)
                      
theorem mrs_hilt_total_distance : total_distance = 510 := 
by
  sorry

end mrs_hilt_total_distance_l223_223824


namespace slope_perpendicular_l223_223169

theorem slope_perpendicular (x1 y1 x2 y2 m : ℚ) 
  (hx1 : x1 = 3) (hy1 : y1 = -4) (hx2 : x2 = -6) (hy2 : y2 = 2) 
  (hm : m = (y2 - y1) / (x2 - x1)) :
  ∀ m_perpendicular: ℚ, m_perpendicular = (-1 / m) → m_perpendicular = 3/2 := 
sorry

end slope_perpendicular_l223_223169


namespace linda_savings_l223_223814

theorem linda_savings (S : ℕ) (h1 : (3 / 4) * S = x) (h2 : (1 / 4) * S = 240) : S = 960 :=
by
  sorry

end linda_savings_l223_223814


namespace find_original_number_l223_223040

theorem find_original_number : ∃ (N : ℤ), (∃ (k : ℤ), N - 30 = 87 * k) ∧ N = 117 :=
by
  sorry

end find_original_number_l223_223040


namespace yoongi_age_l223_223012

theorem yoongi_age
  (H Y : ℕ)
  (h1 : Y = H - 2)
  (h2 : Y + H = 18) :
  Y = 8 :=
by
  sorry

end yoongi_age_l223_223012


namespace find_x_l223_223559

theorem find_x 
  (x : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (hP : P = (x, 6)) 
  (hcos : Real.cos θ = -4/5) 
  : x = -8 := 
sorry

end find_x_l223_223559


namespace hexagonal_pyramid_cross_section_distance_l223_223576

theorem hexagonal_pyramid_cross_section_distance
  (A1 A2 : ℝ) (distance_between_planes : ℝ)
  (A1_area : A1 = 125 * Real.sqrt 3)
  (A2_area : A2 = 500 * Real.sqrt 3)
  (distance_between_planes_eq : distance_between_planes = 10) :
  ∃ h : ℝ, h = 20 :=
by
  sorry

end hexagonal_pyramid_cross_section_distance_l223_223576


namespace initial_elephants_count_l223_223992

def exodus_rate : ℕ := 2880
def exodus_time : ℕ := 4
def entrance_rate : ℕ := 1500
def entrance_time : ℕ := 7
def final_elephants : ℕ := 28980

theorem initial_elephants_count :
  final_elephants - (exodus_rate * exodus_time) + (entrance_rate * entrance_time) = 27960 := by
  sorry

end initial_elephants_count_l223_223992


namespace intersection_A_B_l223_223305

def A : Set ℤ := {-2, -1, 0, 1, 2, 3}
def B : Set ℤ := {x | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l223_223305


namespace gcd_sum_and_lcm_eq_gcd_l223_223720

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end gcd_sum_and_lcm_eq_gcd_l223_223720


namespace club_additional_members_l223_223859

theorem club_additional_members (current_members additional_members future_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 15) 
  (h3 : future_members = current_members + additional_members) : 
  future_members - current_members = 15 :=
by
  sorry

end club_additional_members_l223_223859


namespace length_of_diagonal_EG_l223_223177

theorem length_of_diagonal_EG (EF FG GH HE : ℕ) (hEF : EF = 7) (hFG : FG = 15) 
  (hGH : GH = 7) (hHE : HE = 7) (primeEG : Prime EG) : EG = 11 ∨ EG = 13 :=
by
  -- Apply conditions and proof steps here
  sorry

end length_of_diagonal_EG_l223_223177


namespace possible_values_of_a₁_l223_223699

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end possible_values_of_a₁_l223_223699


namespace number_of_friends_l223_223889

theorem number_of_friends (n : ℕ) (h1 : 100 % n = 0) (h2 : 100 % (n + 5) = 0) (h3 : 100 / n - 1 = 100 / (n + 5)) : n = 20 :=
by
  sorry

end number_of_friends_l223_223889


namespace gift_exchange_equation_l223_223362

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l223_223362


namespace nat_square_not_div_factorial_l223_223459

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end nat_square_not_div_factorial_l223_223459


namespace find_c2013_l223_223534

theorem find_c2013 :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ),
    (a 1 = 3) →
    (b 1 = 3) →
    (∀ n : ℕ, 1 ≤ n → a (n+1) - a n = 3) →
    (∀ n : ℕ, 1 ≤ n → b (n+1) = 3 * b n) →
    (∀ n : ℕ, c n = b (a n)) →
    c 2013 = 27^2013 := by
  sorry

end find_c2013_l223_223534


namespace infinite_sequence_exists_l223_223005

noncomputable def has_k_distinct_positive_divisors (n k : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ d ∈ S, d ∣ n

theorem infinite_sequence_exists :
    ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, 0 < k → ∃ n : ℕ, (a n > 0) ∧ has_k_distinct_positive_divisors (a n ^ 2 + a n + 2023) k) :=
  sorry

end infinite_sequence_exists_l223_223005


namespace problem_inequality_l223_223743

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_inequality (a x : ℝ) (h : a ∈ Set.Iic (-1/Real.exp 2)) :
  f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) := 
sorry

end problem_inequality_l223_223743


namespace triangle_inequality_l223_223074

theorem triangle_inequality 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : A + B + C = π) 
  (h5 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h6 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h7 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
  3 / 2 ≤ a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ∧
  (a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ≤ 
     2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) :=
sorry

end triangle_inequality_l223_223074


namespace decimal_expansion_of_13_over_625_l223_223281

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end decimal_expansion_of_13_over_625_l223_223281


namespace k_n_sum_l223_223275

theorem k_n_sum (k n : ℕ) (x y : ℕ):
  2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n → k + n = 6 :=
by sorry

end k_n_sum_l223_223275


namespace first_driver_spends_less_time_l223_223838

noncomputable def round_trip_time (d : ℝ) (v₁ v₂ : ℝ) : ℝ := (d / v₁) + (d / v₂)

theorem first_driver_spends_less_time (d : ℝ) : 
  round_trip_time d 80 80 < round_trip_time d 90 70 :=
by
  --We skip the proof here
  sorry

end first_driver_spends_less_time_l223_223838


namespace total_distance_to_run_l223_223340

theorem total_distance_to_run
  (track_length : ℕ)
  (initial_laps : ℕ)
  (additional_laps : ℕ)
  (total_laps := initial_laps + additional_laps) :
  track_length = 150 →
  initial_laps = 6 →
  additional_laps = 4 →
  total_laps * track_length = 1500 := by
  sorry

end total_distance_to_run_l223_223340


namespace mn_sum_value_l223_223645

-- Definition of the problem conditions
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_consecutive (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨
  (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨
  (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨
  (a = 5 ∧ b = 6) ∨ (a = 6 ∧ b = 5) ∨
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) ∨
  (a = 7 ∧ b = 8) ∨ (a = 8 ∧ b = 7) ∨
  (a = 8 ∧ b = 9) ∨ (a = 9 ∧ b = 8) ∨
  (a = 9 ∧ b = 1) ∨ (a = 1 ∧ b = 9)

noncomputable def m_n_sum : ℕ :=
  let total_permutations := 5040
  let valid_permutations := 60
  let probability := valid_permutations / total_permutations
  let m := 1
  let n := total_permutations / valid_permutations
  m + n

theorem mn_sum_value : m_n_sum = 85 :=
  sorry

end mn_sum_value_l223_223645


namespace cube_of_99999_is_correct_l223_223163

theorem cube_of_99999_is_correct : (99999 : ℕ)^3 = 999970000299999 :=
by
  sorry

end cube_of_99999_is_correct_l223_223163


namespace max_volume_is_16_l223_223378

noncomputable def max_volume (width : ℝ) (material : ℝ) : ℝ :=
  let l := (material - 2 * width) / (2 + 2 * width)
  let h := (material - 2 * l) / (2 * width + 2 * l)
  l * width * h

theorem max_volume_is_16 :
  max_volume 2 32 = 16 :=
by
  sorry

end max_volume_is_16_l223_223378


namespace most_likely_units_digit_sum_is_zero_l223_223634

theorem most_likely_units_digit_sum_is_zero :
  ∃ (units_digit : ℕ), 
  (∀ m n : ℕ, (1 ≤ m ∧ m ≤ 9) ∧ (1 ≤ n ∧ n ≤ 9) → 
    units_digit = (m + n) % 10) ∧ 
  units_digit = 0 :=
sorry

end most_likely_units_digit_sum_is_zero_l223_223634


namespace perpendicular_lines_l223_223918

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end perpendicular_lines_l223_223918


namespace value_of_b_l223_223608

theorem value_of_b (b : ℝ) : 
  (∃ (x : ℝ), x^2 + b * x - 45 = 0 ∧ x = -4) →
  b = -29 / 4 :=
by
  -- Introduce the condition and rewrite it properly
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- Proceed with assumption that we have the condition and need to prove the statement
  sorry

end value_of_b_l223_223608


namespace number_of_possible_triples_l223_223225

-- Given conditions
variables (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)

-- Revenue equation
def revenue_equation : Prop := 10 * x + 5 * y + z = 120

-- Proving the solution
theorem number_of_possible_triples (h : revenue_equation x y z) : 
  ∃ (n : ℕ), n = 121 :=
by
  sorry

end number_of_possible_triples_l223_223225


namespace determine_suit_cost_l223_223393

def cost_of_suit (J B V : ℕ) : Prop :=
  (J + B + V = 150)

theorem determine_suit_cost
  (J B V : ℕ)
  (h1 : J = B + V)
  (h2 : J + 2 * B = 175)
  (h3 : B + 2 * V = 100) :
  cost_of_suit J B V :=
by
  sorry

end determine_suit_cost_l223_223393


namespace half_of_expression_correct_l223_223978

theorem half_of_expression_correct :
  (2^12 + 3 * 2^10) / 2 = 2^9 * 7 :=
by
  sorry

end half_of_expression_correct_l223_223978


namespace sum_arith_seq_elems_l223_223396

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem sum_arith_seq_elems (a d : ℝ) 
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 48) :
  arithmetic_seq a d 6 + arithmetic_seq a d 7 = 24 := 
by 
  sorry

end sum_arith_seq_elems_l223_223396


namespace find_constants_u_v_l223_223193

theorem find_constants_u_v : 
  ∃ u v : ℝ, (∀ x : ℝ, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 :=
sorry

end find_constants_u_v_l223_223193


namespace coin_toss_dice_roll_l223_223217

theorem coin_toss_dice_roll :
  let coin_toss := 2 -- two outcomes for same side coin toss
  let dice_roll := 2 -- two outcomes for multiple of 3 on dice roll
  coin_toss * dice_roll = 4 :=
by
  sorry

end coin_toss_dice_roll_l223_223217


namespace water_tank_capacity_l223_223571

theorem water_tank_capacity (C : ℝ) (h : 0.70 * C - 0.40 * C = 36) : C = 120 :=
sorry

end water_tank_capacity_l223_223571


namespace num_solutions_l223_223316

-- Let x be a real number
variable (x : ℝ)

-- Define the given equation
def equation := (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)

-- Theorem: The number of values of x that satisfy the equation is 3
theorem num_solutions : ∃ (S : Finset ℝ), (∀ x, x ∈ S ↔ equation x) ∧ S.card = 3 := 
by
  sorry

end num_solutions_l223_223316


namespace ohara_triple_example_l223_223341

noncomputable def is_ohara_triple (a b x : ℕ) : Prop := 
  (Real.sqrt a + Real.sqrt b = x)

theorem ohara_triple_example : 
  is_ohara_triple 49 16 11 ∧ 11 ≠ 100 / 5 := 
by
  sorry

end ohara_triple_example_l223_223341


namespace y_n_is_square_of_odd_integer_l223_223052

-- Define the sequences and the initial conditions
def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

-- Helper function to check if a number is odd
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The theorem to prove
theorem y_n_is_square_of_odd_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, y n = k ^ 2 ∧ is_odd k := by
  sorry

end y_n_is_square_of_odd_integer_l223_223052


namespace area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l223_223625

-- Define the side lengths of squares A, B, and C
def side_length_A (s : ℝ) : ℝ := s
def side_length_B (s : ℝ) : ℝ := 2 * s
def side_length_C (s : ℝ) : ℝ := 3.6 * s

-- Define the areas of squares A, B, and C
def area_A (s : ℝ) : ℝ := (side_length_A s) ^ 2
def area_B (s : ℝ) : ℝ := (side_length_B s) ^ 2
def area_C (s : ℝ) : ℝ := (side_length_C s) ^ 2

-- Define the sum of areas of squares A and B
def sum_area_A_B (s : ℝ) : ℝ := area_A s + area_B s

-- Prove that the area of square C is 159.2% greater than the sum of areas of squares A and B
theorem area_C_greater_than_sum_area_A_B_by_159_point_2_percent (s : ℝ) : 
  ((area_C s - sum_area_A_B s) / (sum_area_A_B s)) * 100 = 159.2 := 
sorry

end area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l223_223625


namespace total_quantity_before_adding_water_l223_223513

variable (x : ℚ)
variable (milk water : ℚ)
variable (added_water : ℚ)

-- Mixture contains milk and water in the ratio 3:2
def initial_ratio (milk water : ℚ) : Prop := milk / water = 3 / 2

-- Adding 10 liters of water
def added_amount : ℚ := 10

-- New ratio of milk to water becomes 2:3 after adding 10 liters of water
def new_ratio (milk water : ℚ) (added_water : ℚ) : Prop :=
  milk / (water + added_water) = 2 / 3

theorem total_quantity_before_adding_water
  (h_ratio : initial_ratio milk water)
  (h_added : added_water = 10)
  (h_new_ratio : new_ratio milk water added_water) :
  milk + water = 20 :=
by
  sorry

end total_quantity_before_adding_water_l223_223513


namespace proof_problem_l223_223981

variable {a b x : ℝ}

theorem proof_problem (h1 : x = b / a) (h2 : a ≠ b) (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) :=
sorry

end proof_problem_l223_223981


namespace unique_rectangles_l223_223904

theorem unique_rectangles (a b x y : ℝ) (h_dim : a < b) 
    (h_perimeter : 2 * (x + y) = a + b)
    (h_area : x * y = (a * b) / 2) : 
    (∃ x y : ℝ, (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 2) ∧ (x < a) ∧ (y < b)) → 
    (∃! z w : ℝ, (2 * (z + w) = a + b) ∧ (z * y = (a * b) / 2) ∧ (z < a) ∧ (w < b)) :=
sorry

end unique_rectangles_l223_223904


namespace domain_of_f_l223_223664

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem domain_of_f :
  {x : ℝ | x + 1 > 0} = {x : ℝ | x > -1} :=
by
  sorry

end domain_of_f_l223_223664


namespace parallel_lines_l223_223887

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end parallel_lines_l223_223887


namespace quadratic_real_roots_condition_l223_223568

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) :=
sorry

end quadratic_real_roots_condition_l223_223568


namespace remainder_calculation_l223_223140

theorem remainder_calculation :
  ((2367 * 1023) % 500) = 41 := by
  sorry

end remainder_calculation_l223_223140


namespace total_apples_after_transactions_l223_223942

def initial_apples : ℕ := 65
def percentage_used : ℕ := 20
def apples_bought : ℕ := 15

theorem total_apples_after_transactions :
  (initial_apples * (1 - percentage_used / 100)) + apples_bought = 67 := 
by
  sorry

end total_apples_after_transactions_l223_223942


namespace find_m_repeated_root_l223_223665

theorem find_m_repeated_root (m : ℝ) :
  (∃ x : ℝ, (x - 1) ≠ 0 ∧ (m - 1) - x = 0) → m = 2 :=
by
  sorry

end find_m_repeated_root_l223_223665


namespace expressions_equal_iff_l223_223373

variable (a b c : ℝ)

theorem expressions_equal_iff :
  a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 :=
by
  sorry

end expressions_equal_iff_l223_223373


namespace max_triangle_area_l223_223207

noncomputable def max_area_of_triangle (a b c S : ℝ) : ℝ := 
if h : 4 * S = a^2 - (b - c)^2 ∧ b + c = 4 then 
  2 
else
  sorry

-- The statement we want to prove
theorem max_triangle_area : ∀ (a b c S : ℝ),
  (4 * S = a^2 - (b - c)^2) →
  (b + c = 4) →
  S ≤ max_area_of_triangle a b c S ∧ max_area_of_triangle a b c S = 2 :=
by sorry

end max_triangle_area_l223_223207


namespace find_F_l223_223703

-- Define the condition and the equation
def C (F : ℤ) : ℤ := (5 * (F - 30)) / 9

-- Define the assumption that C = 25
def C_condition : ℤ := 25

-- The theorem to prove that F = 75 given the conditions
theorem find_F (F : ℤ) (h : C F = C_condition) : F = 75 :=
sorry

end find_F_l223_223703


namespace find_r_l223_223796

noncomputable def parabola_vertex : (ℝ × ℝ) := (0, -1)

noncomputable def intersection_points (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (r - Real.sqrt (r^2 + 4)) / 2
  let y1 := r * x1
  let x2 := (r + Real.sqrt (r^2 + 4)) / 2
  let y2 := r * x2
  ((x1, y1), (x2, y2))

noncomputable def triangle_area (r : ℝ) : ℝ :=
  let base := Real.sqrt (r^2 + 4)
  let height := 2
  1/2 * base * height

theorem find_r (r : ℝ) (h : r > 0) : triangle_area r = 32 → r = Real.sqrt 1020 := 
by
  sorry

end find_r_l223_223796


namespace minValue_l223_223126

noncomputable def minValueOfExpression (a b c : ℝ) : ℝ :=
  (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a))

theorem minValue (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 2 * a + 2 * b + 2 * c = 3) : 
  minValueOfExpression a b c = 2 :=
  sorry

end minValue_l223_223126


namespace probability_two_tails_after_two_heads_l223_223646

noncomputable def fair_coin_probability : ℚ :=
  -- Given conditions:
  let p_head := (1 : ℚ) / 2
  let p_tail := (1 : ℚ) / 2

  -- Define the probability Q as stated in the problem
  let Q := ((1 : ℚ) / 4) / (1 - (1 : ℚ) / 4)

  -- Calculate the probability of starting with sequence "HTH"
  let p_HTH := p_head * p_tail * p_head

  -- Calculate the final probability
  p_HTH * Q

theorem probability_two_tails_after_two_heads :
  fair_coin_probability = (1 : ℚ) / 24 :=
by
  sorry

end probability_two_tails_after_two_heads_l223_223646


namespace arithmetic_sequence_term_l223_223066

theorem arithmetic_sequence_term (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 4 = 6)
    (h2 : 2 * (a 3) - (a 2) = 6)
    (h_sum : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  a 1 = -3 := 
by sorry

end arithmetic_sequence_term_l223_223066


namespace triangle_ratio_perimeter_l223_223267

theorem triangle_ratio_perimeter (AC BC : ℝ) (CD : ℝ) (AB : ℝ) (m n : ℕ) :
  AC = 15 → BC = 20 → AB = 25 → CD = 10 * Real.sqrt 3 →
  gcd m n = 1 → (2 * Real.sqrt ((AC * BC) / AB) + AB) / AB = m / n → m + n = 7 :=
by
  intros hAC hBC hAB hCD hmn hratio
  sorry

end triangle_ratio_perimeter_l223_223267


namespace coin_problem_l223_223019

theorem coin_problem : ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5) ∧ n % 9 = 0 :=
by
  sorry

end coin_problem_l223_223019


namespace rhombus_area_and_perimeter_l223_223606

theorem rhombus_area_and_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 26) :
  let area := (d1 * d2) / 2
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let perimeter := 4 * s
  area = 234 ∧ perimeter = 20 * Real.sqrt 10 := by
  sorry

end rhombus_area_and_perimeter_l223_223606


namespace picnic_basket_cost_l223_223755

theorem picnic_basket_cost :
  let sandwich_cost := 5
  let fruit_salad_cost := 3
  let soda_cost := 2
  let snack_bag_cost := 4
  let num_people := 4
  let num_sodas_per_person := 2
  let num_snack_bags := 3
  (num_people * sandwich_cost) + (num_people * fruit_salad_cost) + (num_people * num_sodas_per_person * soda_cost) + (num_snack_bags * snack_bag_cost) = 60 :=
by
  sorry

end picnic_basket_cost_l223_223755


namespace complement_of_supplement_of_30_degrees_l223_223540

def supplementary_angle (x : ℕ) : ℕ := 180 - x
def complementary_angle (x : ℕ) : ℕ := if x > 90 then x - 90 else 90 - x

theorem complement_of_supplement_of_30_degrees : complementary_angle (supplementary_angle 30) = 60 := by
  sorry

end complement_of_supplement_of_30_degrees_l223_223540


namespace no_difference_of_squares_equals_222_l223_223051

theorem no_difference_of_squares_equals_222 (a b : ℤ) : a^2 - b^2 ≠ 222 := 
  sorry

end no_difference_of_squares_equals_222_l223_223051


namespace rationalize_denominator_l223_223653

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l223_223653


namespace range_of_a_l223_223893

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x^2

theorem range_of_a {a : ℝ} : 
  (∀ x, Real.exp x - 2 * a * x ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 / 2 :=
by
  sorry

end range_of_a_l223_223893


namespace abs_sum_eq_abs_add_iff_ab_gt_zero_l223_223655

theorem abs_sum_eq_abs_add_iff_ab_gt_zero (a b : ℝ) :
  (|a + b| = |a| + |b|) → (a = 0 ∧ b = 0 ∨ ab > 0) :=
sorry

end abs_sum_eq_abs_add_iff_ab_gt_zero_l223_223655


namespace square_roots_of_x_l223_223080

theorem square_roots_of_x (a x : ℝ) 
    (h1 : (2 * a - 1) ^ 2 = x) 
    (h2 : (-a + 2) ^ 2 = x)
    (hx : 0 < x) 
    : x = 9 ∨ x = 1 := 
by sorry

end square_roots_of_x_l223_223080


namespace length_of_each_piece_after_subdividing_l223_223620

theorem length_of_each_piece_after_subdividing (total_length : ℝ) (num_initial_cuts : ℝ) (num_pieces_given : ℝ) (num_subdivisions : ℝ) (final_length : ℝ) : 
  total_length = 200 → 
  num_initial_cuts = 4 → 
  num_pieces_given = 2 → 
  num_subdivisions = 2 → 
  final_length = (total_length / num_initial_cuts / num_subdivisions) → 
  final_length = 25 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end length_of_each_piece_after_subdividing_l223_223620


namespace product_divisible_by_3_l223_223292

noncomputable def dice_prob_divisible_by_3 (n : ℕ) (faces : List ℕ) : ℚ := 
  let probability_div_3 := (1 / 3 : ℚ)
  let probability_not_div_3 := (2 / 3 : ℚ)
  1 - probability_not_div_3 ^ n

theorem product_divisible_by_3 (faces : List ℕ) (h_faces : faces = [1, 2, 3, 4, 5, 6]) :
  dice_prob_divisible_by_3 6 faces = 665 / 729 := 
  by 
    sorry

end product_divisible_by_3_l223_223292


namespace sum_of_first_cards_l223_223215

variables (a b c d : ℕ)

theorem sum_of_first_cards (a b c d : ℕ) : 
  ∃ x, x = b * (c + 1) + d - a :=
by
  sorry

end sum_of_first_cards_l223_223215


namespace baker_bought_131_new_cakes_l223_223202

def number_of_new_cakes_bought (initial_cakes: ℕ) (cakes_sold: ℕ) (excess_sold: ℕ): ℕ :=
    cakes_sold - excess_sold - initial_cakes

theorem baker_bought_131_new_cakes :
    number_of_new_cakes_bought 8 145 6 = 131 :=
by
  -- This is where the proof would normally go
  sorry

end baker_bought_131_new_cakes_l223_223202


namespace total_profit_l223_223404

theorem total_profit (C_profit : ℝ) (x : ℝ) (h1 : 4 * x = 48000) : 12 * x = 144000 :=
by
  sorry

end total_profit_l223_223404


namespace investment_ratio_l223_223640

theorem investment_ratio (P Q : ℝ) (h1 : (P * 5) / (Q * 9) = 7 / 9) : P / Q = 7 / 5 :=
by sorry

end investment_ratio_l223_223640


namespace inscribed_square_product_l223_223363

theorem inscribed_square_product (a b : ℝ)
  (h1 : a + b = 2 * Real.sqrt 5)
  (h2 : Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2) :
  a * b = -6 := 
by
  sorry

end inscribed_square_product_l223_223363


namespace solve_alcohol_mixture_problem_l223_223604

theorem solve_alcohol_mixture_problem (x y : ℝ) 
(h1 : x + y = 18) 
(h2 : 0.75 * x + 0.15 * y = 9) 
: x = 10.5 ∧ y = 7.5 :=
by 
  sorry

end solve_alcohol_mixture_problem_l223_223604


namespace find_recip_sum_of_shifted_roots_l223_223663

noncomputable def reciprocal_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) : ℝ :=
  1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2)

theorem find_recip_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) :
  reciprocal_sum_of_shifted_roots α β γ hαβγ = -19 / 14 :=
  sorry

end find_recip_sum_of_shifted_roots_l223_223663


namespace polar_area_enclosed_l223_223892

theorem polar_area_enclosed :
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  area = 8 * Real.pi / 3 :=
by
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  show area = 8 * Real.pi / 3
  sorry

end polar_area_enclosed_l223_223892


namespace kanul_machinery_expense_l223_223878

theorem kanul_machinery_expense :
  let Total := 93750
  let RawMaterials := 35000
  let Cash := 0.20 * Total
  let Machinery := Total - (RawMaterials + Cash)
  Machinery = 40000 := by
sorry

end kanul_machinery_expense_l223_223878


namespace horner_rule_v3_is_36_l223_223552

def f (x : ℤ) : ℤ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_rule_v3_is_36 :
  let v0 := 1;
  let v1 := v0 * 3 + 0;
  let v2 := v1 * 3 + 2;
  let v3 := v2 * 3 + 3;
  v3 = 36 := 
by
  sorry

end horner_rule_v3_is_36_l223_223552


namespace travel_from_A_to_C_l223_223731

def num_ways_A_to_B : ℕ := 5 + 2  -- 5 buses and 2 trains
def num_ways_B_to_C : ℕ := 3 + 2  -- 3 buses and 2 ferries

theorem travel_from_A_to_C :
  num_ways_A_to_B * num_ways_B_to_C = 35 :=
by
  -- The proof environment will be added here. 
  -- We include 'sorry' here for now.
  sorry

end travel_from_A_to_C_l223_223731


namespace most_stable_performance_l223_223588

theorem most_stable_performance 
    (s_A s_B s_C s_D : ℝ)
    (hA : s_A = 1.5)
    (hB : s_B = 2.6)
    (hC : s_C = 1.7)
    (hD : s_D = 2.8)
    (mean_score : ∀ (x : ℝ), x = 88.5) :
    s_A < s_C ∧ s_C < s_B ∧ s_B < s_D := by
  sorry

end most_stable_performance_l223_223588


namespace expected_heads_64_coins_l223_223920

noncomputable def expected_heads (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem expected_heads_64_coins : expected_heads 64 (15/16) = 60 := by
  sorry

end expected_heads_64_coins_l223_223920


namespace quadratic_has_one_solution_positive_value_of_n_l223_223730

theorem quadratic_has_one_solution_positive_value_of_n :
  ∃ n : ℝ, (4 * x ^ 2 + n * x + 1 = 0 → n ^ 2 - 16 = 0) ∧ n > 0 ∧ n = 4 :=
sorry

end quadratic_has_one_solution_positive_value_of_n_l223_223730


namespace quadratic_properties_l223_223506

theorem quadratic_properties (d e f : ℝ)
  (h1 : d * 1^2 + e * 1 + f = 3)
  (h2 : d * 2^2 + e * 2 + f = 0)
  (h3 : d * 9 + e * 3 + f = -3) :
  d + e + 2 * f = 19.5 :=
sorry

end quadratic_properties_l223_223506


namespace krikor_speed_increase_l223_223118

/--
Krikor traveled to work on two consecutive days, Monday and Tuesday, at different speeds.
Both days, he covered the same distance. On Monday, he traveled for 0.5 hours, and on
Tuesday, he traveled for \( \frac{5}{12} \) hours. Prove that the percentage increase in his speed 
from Monday to Tuesday is 20%.
-/
theorem krikor_speed_increase :
  ∀ (v1 v2 : ℝ), (0.5 * v1 = (5 / 12) * v2) → (v2 = (6 / 5) * v1) → 
  ((v2 - v1) / v1 * 100 = 20) :=
by
  -- Proof goes here
  sorry

end krikor_speed_increase_l223_223118


namespace inequality_system_solution_l223_223376

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l223_223376


namespace fruit_costs_l223_223147

theorem fruit_costs (
    A O B : ℝ
) (h1 : O = A + 0.28)
  (h2 : B = A - 0.15)
  (h3 : 3 * A + 7 * O + 5 * B = 7.84) :
  A = 0.442 ∧ O = 0.722 ∧ B = 0.292 :=
by
  -- The proof is omitted here; replacing with sorry for now
  sorry

end fruit_costs_l223_223147


namespace sector_area_proof_l223_223954

-- Define variables for the central angle, arc length, and derived radius
variables (θ L : ℝ) (r A: ℝ)

-- Define the conditions given in the problem
def central_angle_condition : Prop := θ = 2
def arc_length_condition : Prop := L = 4
def radius_condition : Prop := r = L / θ

-- Define the formula for the area of the sector
def area_of_sector_condition : Prop := A = (1 / 2) * r^2 * θ

-- The theorem that needs to be proved
theorem sector_area_proof :
  central_angle_condition θ ∧ arc_length_condition L ∧ radius_condition θ L r ∧ area_of_sector_condition r θ A → A = 4 :=
by
  sorry

end sector_area_proof_l223_223954


namespace isosceles_triangle_l223_223977

theorem isosceles_triangle {a b R : ℝ} {α β : ℝ} 
  (h : a * Real.tan α + b * Real.tan β = (a + b) * Real.tan ((α + β) / 2))
  (ha : a = 2 * R * Real.sin α) (hb : b = 2 * R * Real.sin β) :
  α = β := 
sorry

end isosceles_triangle_l223_223977


namespace problem1_problem2_l223_223773

-- Problem 1: Prove that (2sin(α) - cos(α)) / (sin(α) + 2cos(α)) = 3/4 given tan(α) = 2
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

-- Problem 2: Prove that 2sin^2(x) - sin(x)cos(x) + cos^2(x) = 2 - sin(2x)/2
theorem problem2 (x : ℝ) : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 2 - Real.sin (2 * x) / 2 := 
sorry

end problem1_problem2_l223_223773


namespace value_of_g_at_neg2_l223_223794

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_of_g_at_neg2 : g (-2) = 15 :=
by
  -- This is where the proof steps would go, but we'll skip it
  sorry

end value_of_g_at_neg2_l223_223794


namespace melissa_work_hours_l223_223901

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end melissa_work_hours_l223_223901


namespace vans_needed_for_trip_l223_223545

theorem vans_needed_for_trip (total_people : ℕ) (van_capacity : ℕ) (h_total_people : total_people = 24) (h_van_capacity : van_capacity = 8) : ℕ :=
  let exact_vans := total_people / van_capacity
  let vans_needed := if total_people % van_capacity = 0 then exact_vans else exact_vans + 1
  have h_exact : exact_vans = 3 := by sorry
  have h_vans_needed : vans_needed = 4 := by sorry
  vans_needed

end vans_needed_for_trip_l223_223545


namespace quadratic_inequality_real_solutions_l223_223932

theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∃ x : ℝ, x^2 - 10 * x + c < 0) ↔ c < 25 :=
sorry

end quadratic_inequality_real_solutions_l223_223932


namespace number_of_juniors_l223_223965

variable (J S x y : ℕ)

-- Conditions given in the problem
axiom total_students : J + S = 40
axiom junior_debate_team : 3 * J / 10 = x
axiom senior_debate_team : S / 5 = y
axiom equal_debate_team : x = y

-- The theorem to prove 
theorem number_of_juniors : J = 16 :=
by
  sorry

end number_of_juniors_l223_223965


namespace tangent_slope_at_one_l223_223530

def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end tangent_slope_at_one_l223_223530


namespace num_ways_to_buy_three_items_l223_223947

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l223_223947


namespace statement_c_correct_l223_223319

theorem statement_c_correct (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
by sorry

end statement_c_correct_l223_223319


namespace infinitely_many_n_divide_2n_plus_1_l223_223323

theorem infinitely_many_n_divide_2n_plus_1 :
    ∃ (S : Set ℕ), (∀ n ∈ S, n > 0 ∧ n ∣ (2 * n + 1)) ∧ Set.Infinite S :=
by
  sorry

end infinitely_many_n_divide_2n_plus_1_l223_223323


namespace percentage_female_guests_from_jay_family_l223_223117

def total_guests : ℕ := 240
def female_guests_percentage : ℕ := 60
def female_guests_from_jay_family : ℕ := 72

theorem percentage_female_guests_from_jay_family :
  (female_guests_from_jay_family : ℚ) / (total_guests * (female_guests_percentage / 100) : ℚ) * 100 = 50 := by
  sorry

end percentage_female_guests_from_jay_family_l223_223117


namespace quadratic_equation_with_roots_sum_and_difference_l223_223309

theorem quadratic_equation_with_roots_sum_and_difference (p q : ℚ)
  (h1 : p + q = 10)
  (h2 : abs (p - q) = 2) :
  (Polynomial.eval₂ (RingHom.id ℚ) p (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) ∧
  (Polynomial.eval₂ (RingHom.id ℚ) q (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) :=
by sorry

end quadratic_equation_with_roots_sum_and_difference_l223_223309


namespace a_range_l223_223962

noncomputable def f (x a : ℝ) : ℝ := |2 * x - 1| + |x - 2 * a|

def valid_a_range (a : ℝ) : Prop :=
∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ 4

theorem a_range (a : ℝ) : valid_a_range a → (1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) := 
sorry

end a_range_l223_223962


namespace sequence_property_l223_223621

def sequence_conditions (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  ∀ n ≥ 3, S n + S (n - 2) = 2 * S (n - 1) + n

theorem sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ) (h : sequence_conditions a S) : 
  ∀ n ≥ 3, a n = a (n - 1) + n :=
  sorry

end sequence_property_l223_223621


namespace largest_multiple_of_7_less_than_neg_100_l223_223337

theorem largest_multiple_of_7_less_than_neg_100 : 
  ∃ (x : ℤ), (∃ n : ℤ, x = 7 * n) ∧ x < -100 ∧ ∀ y : ℤ, (∃ m : ℤ, y = 7 * m) ∧ y < -100 → y ≤ x :=
by
  sorry

end largest_multiple_of_7_less_than_neg_100_l223_223337


namespace value_of_a_l223_223172

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end value_of_a_l223_223172


namespace employees_use_public_transportation_l223_223452

theorem employees_use_public_transportation
    (total_employees : ℕ)
    (drive_percentage : ℝ)
    (public_transportation_fraction : ℝ)
    (h1 : total_employees = 100)
    (h2 : drive_percentage = 0.60)
    (h3 : public_transportation_fraction = 0.50) :
    ((total_employees * (1 - drive_percentage)) * public_transportation_fraction) = 20 :=
by
    sorry

end employees_use_public_transportation_l223_223452


namespace trigonometric_proof_l223_223879

noncomputable def proof_problem (α β : Real) : Prop :=
  (β = 90 - α) → (Real.sin β = Real.cos α) → 
  (Real.sqrt 3 * Real.sin α + Real.sin β) / Real.sqrt (2 - 2 * Real.cos 100) = 1

-- Statement that incorporates all conditions and concludes the proof problem.
theorem trigonometric_proof :
  proof_problem 20 70 :=
by
  intros h1 h2
  sorry

end trigonometric_proof_l223_223879


namespace ratio_larva_to_cocoon_l223_223695

theorem ratio_larva_to_cocoon (total_days : ℕ) (cocoon_days : ℕ)
  (h1 : total_days = 120) (h2 : cocoon_days = 30) :
  (total_days - cocoon_days) / cocoon_days = 3 := by
  sorry

end ratio_larva_to_cocoon_l223_223695


namespace smallest_a_exists_l223_223412

theorem smallest_a_exists : ∃ a b c : ℕ, 
                          (∀ α β : ℝ, 
                          (α > 0 ∧ α ≤ 1 / 1000) ∧ 
                          (β > 0 ∧ β ≤ 1 / 1000) ∧ 
                          (α + β = -b / a) ∧ 
                          (α * β = c / a) ∧ 
                          (b * b - 4 * a * c > 0)) ∧ 
                          (a = 1001000) := sorry

end smallest_a_exists_l223_223412


namespace common_roots_l223_223950

noncomputable def p (x a : ℝ) := x^3 + a * x^2 + 14 * x + 7
noncomputable def q (x b : ℝ) := x^3 + b * x^2 + 21 * x + 15

theorem common_roots (a b : ℝ) (r s : ℝ) (hr : r ≠ s)
  (hp : p r a = 0) (hp' : p s a = 0)
  (hq : q r b = 0) (hq' : q s b = 0) :
  a = 5 ∧ b = 4 :=
by sorry

end common_roots_l223_223950


namespace total_campers_l223_223871

def campers_morning : ℕ := 36
def campers_afternoon : ℕ := 13
def campers_evening : ℕ := 49

theorem total_campers : campers_morning + campers_afternoon + campers_evening = 98 := by
  sorry

end total_campers_l223_223871


namespace sum_invariant_under_permutation_l223_223574

theorem sum_invariant_under_permutation (b : List ℝ) (σ : List ℕ) (hσ : σ.Perm (List.range b.length)) :
  (List.sum b) = (List.sum (σ.map (b.get!))) := by
  sorry

end sum_invariant_under_permutation_l223_223574


namespace angle_measure_of_E_l223_223921

theorem angle_measure_of_E (E F G H : ℝ) 
  (h1 : E = 3 * F) 
  (h2 : E = 4 * G) 
  (h3 : E = 6 * H) 
  (h_sum : E + F + G + H = 360) : 
  E = 206 := 
by 
  sorry

end angle_measure_of_E_l223_223921


namespace seashells_count_l223_223875

theorem seashells_count (total_seashells broken_seashells : ℕ) (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) : total_seashells - broken_seashells = 3 := by
  sorry

end seashells_count_l223_223875


namespace percentage_difference_is_20_l223_223657

/-
Barry can reach apples that are 5 feet high.
Larry is 5 feet tall.
When Barry stands on Larry's shoulders, they can reach 9 feet high.
-/
def Barry_height : ℝ := 5
def Larry_height : ℝ := 5
def Combined_height : ℝ := 9

/-
Prove the percentage difference between Larry's full height and his shoulder height is 20%.
-/
theorem percentage_difference_is_20 :
  ((Larry_height - (Combined_height - Barry_height)) / Larry_height) * 100 = 20 :=
by
  sorry

end percentage_difference_is_20_l223_223657


namespace total_earning_correct_l223_223864

-- Definitions based on conditions
def daily_wage_c : ℕ := 105
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

-- Given the ratio of their daily wages
def ratio_a : ℕ := 3
def ratio_b : ℕ := 4
def ratio_c : ℕ := 5

-- Now we calculate the daily wages based on the ratio
def unit_wage : ℕ := daily_wage_c / ratio_c
def daily_wage_a : ℕ := ratio_a * unit_wage
def daily_wage_b : ℕ := ratio_b * unit_wage

-- Total earnings are calculated by multiplying daily wages and days worked
def total_earning_a : ℕ := days_worked_a * daily_wage_a
def total_earning_b : ℕ := days_worked_b * daily_wage_b
def total_earning_c : ℕ := days_worked_c * daily_wage_c

def total_earning : ℕ := total_earning_a + total_earning_b + total_earning_c

-- Theorem to prove
theorem total_earning_correct : total_earning = 1554 := by
  sorry

end total_earning_correct_l223_223864


namespace sufficient_not_necessary_perpendicular_l223_223566

theorem sufficient_not_necessary_perpendicular (a : ℝ) :
  (∀ x y : ℝ, (a + 2) * x + 3 * a * y + 1 = 0 ∧
              (a - 2) * x + (a + 2) * y - 3 = 0 → false) ↔ a = -2 :=
sorry

end sufficient_not_necessary_perpendicular_l223_223566


namespace remainder_8354_11_l223_223007

theorem remainder_8354_11 : 8354 % 11 = 6 := sorry

end remainder_8354_11_l223_223007


namespace questionnaire_visitors_l223_223366

noncomputable def total_visitors :=
  let V := 600
  let E := (3 / 4) * V
  V

theorem questionnaire_visitors:
  ∃ (V : ℕ), V = 600 ∧
  (∀ (E : ℕ), E = (3 / 4) * V ∧ E + 150 = V) :=
by
    use 600
    sorry

end questionnaire_visitors_l223_223366


namespace reflection_of_point_l223_223315

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l223_223315


namespace polynomial_solution_l223_223991

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2)

theorem polynomial_solution (P : ℝ → ℝ) : is_solution P → ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
  sorry

end polynomial_solution_l223_223991


namespace prove_odd_function_definition_l223_223749

theorem prove_odd_function_definition (f : ℝ → ℝ) 
  (odd : ∀ x : ℝ, f (-x) = -f x)
  (pos_def : ∀ x : ℝ, 0 < x → f x = 2 * x ^ 2 - x + 1) :
  ∀ x : ℝ, x < 0 → f x = -2 * x ^ 2 - x - 1 :=
by
  intro x hx
  sorry

end prove_odd_function_definition_l223_223749


namespace injectivity_of_composition_l223_223734

variable {R : Type*} [LinearOrderedField R]

def injective (f : R → R) := ∀ a b, f a = f b → a = b

theorem injectivity_of_composition {f g : R → R} (h : injective (g ∘ f)) : injective f :=
by
  sorry

end injectivity_of_composition_l223_223734


namespace largest_divisible_by_digits_sum_l223_223399

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_divisible_by_digits_sum : ∃ n, n < 900 ∧ n % digits_sum n = 0 ∧ ∀ m, m < 900 ∧ m % digits_sum m = 0 → m ≤ 888 :=
by
  sorry

end largest_divisible_by_digits_sum_l223_223399


namespace haley_marbles_l223_223547

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (total_marbles : ℕ) 
  (h1 : boys = 11) (h2 : marbles_per_boy = 9) : total_marbles = 99 :=
by
  sorry

end haley_marbles_l223_223547


namespace cosine_third_angle_of_triangle_l223_223941

theorem cosine_third_angle_of_triangle (X Y Z : ℝ)
  (sinX_eq : Real.sin X = 4/5)
  (cosY_eq : Real.cos Y = 12/13)
  (triangle_sum : X + Y + Z = Real.pi) :
  Real.cos Z = -16/65 :=
by
  -- proof will be filled in
  sorry

end cosine_third_angle_of_triangle_l223_223941


namespace new_phone_plan_cost_l223_223210

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.30
def new_plan_cost := old_plan_cost + (increase_percentage * old_plan_cost)

theorem new_phone_plan_cost : new_plan_cost = 195 := by
  -- From the condition that the old plan cost is $150 and the increase percentage is 30%
  -- We should prove that the new plan cost is $195
  sorry

end new_phone_plan_cost_l223_223210


namespace percent_of_male_literate_l223_223562

noncomputable def female_percentage : ℝ := 0.6
noncomputable def total_employees : ℕ := 1500
noncomputable def literate_percentage : ℝ := 0.62
noncomputable def literate_female_employees : ℕ := 630

theorem percent_of_male_literate :
  let total_females := (female_percentage * total_employees)
  let total_males := total_employees - total_females
  let total_literate := literate_percentage * total_employees
  let literate_male_employees := total_literate - literate_female_employees
  let male_literate_percentage := (literate_male_employees / total_males) * 100
  male_literate_percentage = 50 := by
  sorry

end percent_of_male_literate_l223_223562


namespace initial_toys_count_l223_223321

theorem initial_toys_count (T : ℕ) (h : 10 * T + 300 = 580) : T = 28 :=
by
  sorry

end initial_toys_count_l223_223321


namespace tim_total_score_l223_223501

-- Definitions from conditions
def single_line_points : ℕ := 1000
def tetris_points : ℕ := 8 * single_line_points
def doubled_tetris_points : ℕ := 2 * tetris_points
def num_singles : ℕ := 6
def num_tetrises : ℕ := 4
def consecutive_tetrises : ℕ := 2
def regular_tetrises : ℕ := num_tetrises - consecutive_tetrises

-- Total score calculation
def total_score : ℕ :=
  num_singles * single_line_points +
  regular_tetrises * tetris_points +
  consecutive_tetrises * doubled_tetris_points

-- Prove that Tim's total score is 54000
theorem tim_total_score : total_score = 54000 :=
by 
  sorry

end tim_total_score_l223_223501


namespace solve_for_a_l223_223984

theorem solve_for_a
  (a x : ℚ)
  (h1 : (2 * a * x + 3) / (a - x) = 3 / 4)
  (h2 : x = 1) : a = -3 :=
by
  sorry

end solve_for_a_l223_223984


namespace probability_not_pulling_prize_twice_l223_223579

theorem probability_not_pulling_prize_twice
  (favorable : ℕ)
  (unfavorable : ℕ)
  (total : ℕ := favorable + unfavorable)
  (P_prize : ℚ := favorable / total)
  (P_not_prize : ℚ := 1 - P_prize)
  (P_not_prize_twice : ℚ := P_not_prize * P_not_prize) :
  P_not_prize_twice = 36 / 121 :=
by
  have favorable : ℕ := 5
  have unfavorable : ℕ := 6
  have total : ℕ := favorable + unfavorable
  have P_prize : ℚ := favorable / total
  have P_not_prize : ℚ := 1 - P_prize
  have P_not_prize_twice : ℚ := P_not_prize * P_not_prize
  sorry

end probability_not_pulling_prize_twice_l223_223579


namespace vasya_cuts_larger_area_l223_223968

noncomputable def E_Vasya_square_area : ℝ :=
  (1/6) * (1^2) + (1/6) * (2^2) + (1/6) * (3^2) + (1/6) * (4^2) + (1/6) * (5^2) + (1/6) * (6^2)

noncomputable def E_Asya_rectangle_area : ℝ :=
  (3.5 * 3.5)

theorem vasya_cuts_larger_area :
  E_Vasya_square_area > E_Asya_rectangle_area :=
  by
    sorry

end vasya_cuts_larger_area_l223_223968


namespace principal_sum_l223_223911

noncomputable def diff_simple_compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
(P * ((1 + r / 100)^t) - P) - (P * r * t / 100)

theorem principal_sum (P : ℝ) (r : ℝ) (t : ℝ) (h : diff_simple_compound_interest P r t = 631) (hr : r = 10) (ht : t = 2) :
    P = 63100 := by
  sorry

end principal_sum_l223_223911


namespace animals_total_sleep_in_one_week_l223_223940

-- Define the conditions
def cougar_sleep_per_night := 4 -- Cougar sleeps 4 hours per night
def zebra_extra_sleep := 2 -- Zebra sleeps 2 hours more than cougar

-- Calculate the sleep duration for the zebra
def zebra_sleep_per_night := cougar_sleep_per_night + zebra_extra_sleep

-- Total sleep duration per week
def week_nights := 7

-- Total weekly sleep durations
def cougar_weekly_sleep := cougar_sleep_per_night * week_nights
def zebra_weekly_sleep := zebra_sleep_per_night * week_nights

-- Total sleep time for both animals in one week
def total_weekly_sleep := cougar_weekly_sleep + zebra_weekly_sleep

-- The target theorem
theorem animals_total_sleep_in_one_week : total_weekly_sleep = 70 := by
  sorry

end animals_total_sleep_in_one_week_l223_223940


namespace unique_solution_to_equation_l223_223784

theorem unique_solution_to_equation (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) 
  (h : 1 + 5^x = 2^y + 2^z * 5^t) : (x, y, z, t) = (2, 4, 1, 1) := 
sorry

end unique_solution_to_equation_l223_223784


namespace magnitude_of_2a_plus_b_l223_223245

open Real

variables (a b : ℝ × ℝ) (angle : ℝ)

-- Conditions
axiom angle_between_a_b (a b : ℝ × ℝ) : angle = π / 3 -- 60 degrees in radians
axiom norm_a_eq_1 (a : ℝ × ℝ) : ‖a‖ = 1
axiom b_eq (b : ℝ × ℝ) : b = (3, 0)

-- Theorem
theorem magnitude_of_2a_plus_b (h1 : angle = π / 3) (h2 : ‖a‖ = 1) (h3 : b = (3, 0)) :
  ‖2 • a + b‖ = sqrt 19 :=
sorry

end magnitude_of_2a_plus_b_l223_223245


namespace original_cost_price_l223_223423

theorem original_cost_price (selling_price_friend : ℝ) (gain_percent : ℝ) (loss_percent : ℝ) 
  (final_selling_price : ℝ) : 
  final_selling_price = 54000 → gain_percent = 0.2 → loss_percent = 0.1 → 
  selling_price_friend = (1 - loss_percent) * x → final_selling_price = (1 + gain_percent) * selling_price_friend → 
  x = 50000 :=
by 
  sorry

end original_cost_price_l223_223423


namespace cameron_list_count_l223_223431

theorem cameron_list_count :
  let lower := 100
  let upper := 1000
  let step := 20
  let n_min := lower / step
  let n_max := upper / step
  lower % step = 0 ∧ upper % step = 0 →
  upper ≥ lower →
  n_max - n_min + 1 = 46 :=
by
  sorry

end cameron_list_count_l223_223431


namespace contractor_fine_per_absent_day_l223_223145

theorem contractor_fine_per_absent_day :
  ∀ (total_days absent_days wage_per_day total_receipt fine_per_absent_day : ℝ),
    total_days = 30 →
    wage_per_day = 25 →
    absent_days = 4 →
    total_receipt = 620 →
    (total_days - absent_days) * wage_per_day - absent_days * fine_per_absent_day = total_receipt →
    fine_per_absent_day = 7.50 :=
by
  intros total_days absent_days wage_per_day total_receipt fine_per_absent_day
  intro h1 h2 h3 h4 h5
  sorry

end contractor_fine_per_absent_day_l223_223145


namespace sequence_a_l223_223863

theorem sequence_a (a : ℕ → ℝ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n ≥ 2, a n / a (n + 1) + a n / a (n - 1) = 2) :
  a 12 = 1 / 6 :=
sorry

end sequence_a_l223_223863


namespace sector_central_angle_l223_223913

noncomputable def sector_radius (r l : ℝ) : Prop :=
2 * r + l = 10

noncomputable def sector_area (r l : ℝ) : Prop :=
(1 / 2) * l * r = 4

noncomputable def central_angle (α r l : ℝ) : Prop :=
α = l / r

theorem sector_central_angle (r l α : ℝ) 
  (h1 : sector_radius r l) 
  (h2 : sector_area r l) 
  (h3 : central_angle α r l) : 
  α = 1 / 2 := 
by
  sorry

end sector_central_angle_l223_223913


namespace total_cost_of_apples_and_bananas_l223_223264

variable (a b : ℝ)

theorem total_cost_of_apples_and_bananas (a b : ℝ) : 2 * a + 3 * b = 2 * a + 3 * b :=
by
  sorry

end total_cost_of_apples_and_bananas_l223_223264


namespace bestCompletion_is_advantage_l223_223114

-- Defining the phrase and the list of options
def phrase : String := "British students have a language ____ for jobs in the USA and Australia"

def options : List (String × String) := 
  [("A", "chance"), ("B", "ability"), ("C", "possibility"), ("D", "advantage")]

-- Defining the best completion function (using a placeholder 'sorry' for the logic which is not the focus here)
noncomputable def bestCompletion (phrase : String) (options : List (String × String)) : String :=
  "advantage"  -- We assume given the problem that this function correctly identifies 'advantage'

-- Lean theorem stating the desired property
theorem bestCompletion_is_advantage : bestCompletion phrase options = "advantage" :=
by sorry

end bestCompletion_is_advantage_l223_223114


namespace apples_per_box_l223_223482

variable (A : ℕ) -- Number of apples packed in a box

-- Conditions
def normal_boxes_per_day := 50
def days_per_week := 7
def boxes_first_week := normal_boxes_per_day * days_per_week * A
def boxes_second_week := (normal_boxes_per_day * A - 500) * days_per_week
def total_apples := 24500

-- Theorem
theorem apples_per_box : boxes_first_week + boxes_second_week = total_apples → A = 40 :=
by
  sorry

end apples_per_box_l223_223482


namespace shelter_total_cats_l223_223627

theorem shelter_total_cats (total_adult_cats num_female_cats num_litters avg_kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 150) 
  (h2 : num_female_cats = 2 * total_adult_cats / 3)
  (h3 : num_litters = 2 * num_female_cats / 3)
  (h4 : avg_kittens_per_litter = 5):
  total_adult_cats + num_litters * avg_kittens_per_litter = 480 :=
by
  sorry

end shelter_total_cats_l223_223627


namespace cars_per_day_l223_223141

noncomputable def paul_rate : ℝ := 2
noncomputable def jack_rate : ℝ := 3
noncomputable def paul_jack_rate : ℝ := paul_rate + jack_rate
noncomputable def hours_per_day : ℝ := 8
noncomputable def total_cars : ℝ := paul_jack_rate * hours_per_day

theorem cars_per_day : total_cars = 40 := by
  sorry

end cars_per_day_l223_223141


namespace parallel_lines_a_eq_neg1_l223_223269

theorem parallel_lines_a_eq_neg1 (a : ℝ) :
  ∀ (x y : ℝ), 
    (x + a * y + 6 = 0) ∧ ((a - 2) * x + 3 * y + 2 * a = 0) →
    (-1 / a = - (a - 2) / 3) → 
    a = -1 :=
by
  sorry

end parallel_lines_a_eq_neg1_l223_223269


namespace michael_ratio_l223_223607

-- Definitions
def Michael_initial := 42
def Brother_initial := 17

-- Conditions
def Brother_after_candy_purchase := 35
def Candy_cost := 3
def Brother_before_candy := Brother_after_candy_purchase + Candy_cost
def x := Brother_before_candy - Brother_initial

-- Prove the ratio of the money Michael gave to his brother to his initial amount is 1:2
theorem michael_ratio :
  x * 2 = Michael_initial := by
  sorry

end michael_ratio_l223_223607


namespace encounter_count_l223_223060

theorem encounter_count (vA vB d : ℝ) (h₁ : 5 * d / vA = 9 * d / vB) :
  ∃ encounters : ℝ, encounters = 3023 :=
by
  sorry

end encounter_count_l223_223060


namespace simplify_expression_l223_223047

theorem simplify_expression (n : ℤ) :
  (2 : ℝ) ^ (-(3 * n + 1)) + (2 : ℝ) ^ (-(3 * n - 2)) - 3 * (2 : ℝ) ^ (-3 * n) = (3 / 2) * (2 : ℝ) ^ (-3 * n) :=
by
  sorry

end simplify_expression_l223_223047


namespace fraction_of_64_l223_223030

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l223_223030


namespace greatest_number_of_sets_l223_223998

-- We define the number of logic and visual puzzles.
def n_logic : ℕ := 18
def n_visual : ℕ := 9

-- The theorem states that the greatest number of identical sets Mrs. Wilson can create is the GCD of 18 and 9.
theorem greatest_number_of_sets : gcd n_logic n_visual = 9 := by
  sorry

end greatest_number_of_sets_l223_223998


namespace sum_of_roots_quadratic_eq_l223_223685

theorem sum_of_roots_quadratic_eq (x₁ x₂ : ℝ) (h : x₁^2 + 2 * x₁ - 4 = 0 ∧ x₂^2 + 2 * x₂ - 4 = 0) : 
  x₁ + x₂ = -2 :=
sorry

end sum_of_roots_quadratic_eq_l223_223685


namespace students_on_zoo_trip_l223_223795

theorem students_on_zoo_trip (buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) 
  (h1 : buses = 7) (h2 : students_per_bus = 56) (h3 : students_in_cars = 4) : 
  buses * students_per_bus + students_in_cars = 396 :=
by
  sorry

end students_on_zoo_trip_l223_223795


namespace value_of_t_l223_223930

noncomputable def f (x t k : ℝ) : ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem value_of_t (a b t k : ℝ) (h1 : 0 < t) (h2 : 0 < k) 
  (h3 : a + b = t) (h4 : a * b = k) (h5 : 2 * a = b - 2) (h6 : (-2)^2 = a * b) : 
  t = 5 := 
  sorry

end value_of_t_l223_223930


namespace least_possible_value_d_l223_223635

theorem least_possible_value_d 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (hxy : x < y)
  (hyz : y < z)
  (hyx_gt_five : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_d_l223_223635


namespace exists_positive_n_l223_223660

theorem exists_positive_n {k : ℕ} (h_k : 0 < k) {m : ℕ} (h_m : m % 2 = 1) :
  ∃ n : ℕ, 0 < n ∧ (n^n - m) % 2^k = 0 := 
sorry

end exists_positive_n_l223_223660


namespace monomial_coeff_degree_product_l223_223813

theorem monomial_coeff_degree_product (m n : ℚ) (h₁ : m = -3/4) (h₂ : n = 4) : m * n = -3 := 
by
  sorry

end monomial_coeff_degree_product_l223_223813


namespace nested_composition_l223_223670

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem nested_composition : g (g (g (g (g (g 2))))) = 2 := by
  sorry

end nested_composition_l223_223670


namespace max_value_of_m_l223_223260

-- Define the function f(x)
def f (x : ℝ) := x^2 + 2 * x

-- Define the property of t and m such that the condition holds for all x in [1, m]
def valid_t_m (t m : ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ 3 * x

-- The proof statement ensuring the maximum value of m is 8
theorem max_value_of_m 
  (t : ℝ) (m : ℝ) 
  (ht : ∃ x : ℝ, valid_t_m t x ∧ x = 8) : 
  ∀ m, valid_t_m t m → m ≤ 8 :=
  sorry

end max_value_of_m_l223_223260


namespace new_player_weight_l223_223603

theorem new_player_weight 
  (original_players : ℕ)
  (original_avg_weight : ℝ)
  (new_players : ℕ)
  (new_avg_weight : ℝ)
  (new_total_weight : ℝ) :
  original_players = 20 →
  original_avg_weight = 180 →
  new_players = 21 →
  new_avg_weight = 181.42857142857142 →
  new_total_weight = 3810 →
  (new_total_weight - original_players * original_avg_weight) = 210 :=
by
  intros
  sorry

end new_player_weight_l223_223603


namespace arithmetic_sum_l223_223325

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n * d)

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum :
  ∀ (a d : ℕ),
  arithmetic_sequence a d 2 + arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 12 →
  sum_first_n_terms a d 7 = 28 :=
by
  sorry

end arithmetic_sum_l223_223325


namespace gcd_of_8247_13619_29826_l223_223515

theorem gcd_of_8247_13619_29826 : Nat.gcd (Nat.gcd 8247 13619) 29826 = 3 := 
sorry

end gcd_of_8247_13619_29826_l223_223515


namespace determine_parabola_l223_223227

-- Define the parabola passing through point P(1,1)
def parabola_passing_through (a b c : ℝ) :=
  (1:ℝ)^2 * a + 1 * b + c = 1

-- Define the condition that the tangent line at Q(2, -1) has a slope parallel to y = x - 3, which means slope = 1
def tangent_slope_at_Q (a b : ℝ) :=
  4 * a + b = 1

-- Define the parabola passing through point Q(2, -1)
def parabola_passing_through_Q (a b c : ℝ) :=
  (2:ℝ)^2 * a + (2:ℝ) * b + c = -1

-- The proof statement
theorem determine_parabola (a b c : ℝ):
  parabola_passing_through a b c ∧ 
  tangent_slope_at_Q a b ∧ 
  parabola_passing_through_Q a b c → 
  a = 3 ∧ b = -11 ∧ c = 9 :=
by
  sorry

end determine_parabola_l223_223227


namespace sum_of_vars_l223_223278

theorem sum_of_vars (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : x + y + z = 7 * x := 
by 
  sorry

end sum_of_vars_l223_223278


namespace bounded_infinite_sequence_l223_223842

noncomputable def sequence_x (n : ℕ) : ℝ :=
  4 * (Real.sqrt 2 * n - ⌊Real.sqrt 2 * n⌋)

theorem bounded_infinite_sequence (a : ℝ) (h : a > 1) :
  ∀ i j : ℕ, i ≠ j → (|sequence_x i - sequence_x j| * |(i - j : ℝ)|^a) ≥ 1 := 
by
  intros i j h_ij
  sorry

end bounded_infinite_sequence_l223_223842


namespace next_special_year_after_2009_l223_223176

def is_special_year (n : ℕ) : Prop :=
  ∃ d1 d2 d3 d4 : ℕ,
    (2000 ≤ n) ∧ (n < 10000) ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n) ∧
    (d1 ≠ 0) ∧
    ∀ (p q r s : ℕ),
    (p * 1000 + q * 100 + r * 10 + s < n) →
    (p ≠ d1 ∨ q ≠ d2 ∨ r ≠ d3 ∨ s ≠ d4)

theorem next_special_year_after_2009 : ∃ y : ℕ, is_special_year y ∧ y > 2009 ∧ y = 2022 :=
  sorry

end next_special_year_after_2009_l223_223176


namespace big_al_bananas_l223_223886

/-- Big Al ate 140 bananas from May 1 through May 6. Each day he ate five more bananas than on the previous day. On May 4, Big Al did not eat any bananas due to fasting. Prove that Big Al ate 38 bananas on May 6. -/
theorem big_al_bananas : 
  ∃ a : ℕ, (a + (a + 5) + (a + 10) + 0 + (a + 15) + (a + 20) = 140) ∧ ((a + 20) = 38) :=
by sorry

end big_al_bananas_l223_223886


namespace problem1_l223_223356

theorem problem1 (f : ℝ → ℝ) (x : ℝ) : 
  (f (x + 1/x) = x^2 + 1/x^2) -> f x = x^2 - 2 := 
sorry

end problem1_l223_223356


namespace range_of_a_l223_223062

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 :=
by 
  sorry

end range_of_a_l223_223062


namespace workers_complete_job_together_in_time_l223_223365

theorem workers_complete_job_together_in_time :
  let work_rate_A := 1 / 10 
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  time = 60 / 13 :=
by
  let work_rate_A := 1 / 10
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  sorry

end workers_complete_job_together_in_time_l223_223365


namespace gcd_lcm_sum_l223_223477

theorem gcd_lcm_sum (a b : ℕ) (h : a = 1999 * b) : Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end gcd_lcm_sum_l223_223477


namespace bob_fencing_needed_l223_223228

-- Problem conditions
def length : ℕ := 225
def width : ℕ := 125
def small_gate : ℕ := 3
def large_gate : ℕ := 10

-- Definition of perimeter
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Total width of the gates
def total_gate_width (g1 g2 : ℕ) : ℕ := g1 + g2

-- Amount of fencing needed
def fencing_needed (p gw : ℕ) : ℕ := p - gw

-- Theorem statement
theorem bob_fencing_needed :
  fencing_needed (perimeter length width) (total_gate_width small_gate large_gate) = 687 :=
by 
  sorry

end bob_fencing_needed_l223_223228


namespace train_speed_l223_223535

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end train_speed_l223_223535


namespace underachievers_l223_223888

-- Define the variables for the number of students in each group
variables (a b c : ℕ)

-- Given conditions as hypotheses
axiom total_students : a + b + c = 30
axiom top_achievers : a = 19
axiom average_students : c = 12

-- Prove the number of underachievers
theorem underachievers : b = 9 :=
by sorry

end underachievers_l223_223888


namespace negation_of_existence_l223_223726

theorem negation_of_existence (T : Type) (triangle : T → Prop) (sum_interior_angles : T → ℝ) :
  (¬ ∃ t : T, sum_interior_angles t ≠ 180) ↔ (∀ t : T, sum_interior_angles t = 180) :=
by 
  sorry

end negation_of_existence_l223_223726


namespace quadratic_function_points_l223_223860

theorem quadratic_function_points (a c y1 y2 y3 y4 : ℝ) (h_a : a < 0)
    (h_A : y1 = a * (-2)^2 - 4 * a * (-2) + c)
    (h_B : y2 = a * 0^2 - 4 * a * 0 + c)
    (h_C : y3 = a * 3^2 - 4 * a * 3 + c)
    (h_D : y4 = a * 5^2 - 4 * a * 5 + c)
    (h_condition : y2 * y4 < 0) : y1 * y3 < 0 :=
by
  sorry

end quadratic_function_points_l223_223860


namespace sum_of_integers_l223_223768

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l223_223768


namespace mn_values_l223_223505

theorem mn_values (m n : ℤ) (h : m^2 * n^2 + m^2 + n^2 + 10 * m * n + 16 = 0) : 
  (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
  sorry

end mn_values_l223_223505


namespace possible_slopes_of_line_intersects_ellipse_l223_223032

/-- 
A line whose y-intercept is (0, 3) intersects the ellipse 4x^2 + 9y^2 = 36. 
Find all possible slopes of this line. 
-/
theorem possible_slopes_of_line_intersects_ellipse :
  (∀ m : ℝ, ∃ x : ℝ, 4 * x^2 + 9 * (m * x + 3)^2 = 36) ↔ 
  (m <= - (Real.sqrt 5) / 3 ∨ m >= (Real.sqrt 5) / 3) :=
sorry

end possible_slopes_of_line_intersects_ellipse_l223_223032


namespace total_games_played_l223_223021

noncomputable def win_ratio : ℝ := 5.5
noncomputable def lose_ratio : ℝ := 4.5
noncomputable def tie_ratio : ℝ := 2.5
noncomputable def rained_out_ratio : ℝ := 1
noncomputable def higher_league_ratio : ℝ := 3.5
noncomputable def lost_games : ℝ := 13.5

theorem total_games_played :
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  total_parts * games_per_part = 51 :=
by
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  have : total_parts * games_per_part = 51 := sorry
  exact this

end total_games_played_l223_223021


namespace triangle_subsegment_length_l223_223732

noncomputable def length_of_shorter_subsegment (PQ QR PR PS SR : ℝ) :=
  PQ < QR ∧ 
  PR = 15 ∧ 
  PQ / QR = 1 / 5 ∧ 
  PS + SR = PR ∧ 
  PS = PQ / QR * SR → 
  PS = 5 / 2

theorem triangle_subsegment_length (PQ QR PR PS SR : ℝ) 
  (h1 : PQ < QR) 
  (h2 : PR = 15) 
  (h3 : PQ / QR = 1 / 5) 
  (h4 : PS + SR = PR) 
  (h5 : PS = PQ / QR * SR) : 
  length_of_shorter_subsegment PQ QR PR PS SR := 
sorry

end triangle_subsegment_length_l223_223732


namespace midpoint_translation_l223_223935

theorem midpoint_translation (x1 y1 x2 y2 tx ty mx my : ℤ) 
  (hx1 : x1 = 1) (hy1 : y1 = 3) (hx2 : x2 = 5) (hy2 : y2 = -7)
  (htx : tx = 3) (hty : ty = -4)
  (hmx : mx = (x1 + x2) / 2 + tx) (hmy : my = (y1 + y2) / 2 + ty) : 
  mx = 6 ∧ my = -6 :=
by
  sorry

end midpoint_translation_l223_223935


namespace find_brown_mms_second_bag_l223_223644

variable (x : ℕ)

-- Definitions based on the conditions
def BrownMmsFirstBag := 9
def BrownMmsThirdBag := 8
def BrownMmsFourthBag := 8
def BrownMmsFifthBag := 3
def AveBrownMmsPerBag := 8
def NumBags := 5

-- Condition specifying the average brown M&Ms per bag
axiom average_condition : AveBrownMmsPerBag = (BrownMmsFirstBag + x + BrownMmsThirdBag + BrownMmsFourthBag + BrownMmsFifthBag) / NumBags

-- Prove the number of brown M&Ms in the second bag
theorem find_brown_mms_second_bag : x = 12 := by
  sorry

end find_brown_mms_second_bag_l223_223644


namespace square_area_when_a_eq_b_eq_c_l223_223808

theorem square_area_when_a_eq_b_eq_c {a b c : ℝ} (h : a = b ∧ b = c) :
  ∃ x : ℝ, (x = a * Real.sqrt 2) ∧ (x ^ 2 = 2 * a ^ 2) :=
by
  sorry

end square_area_when_a_eq_b_eq_c_l223_223808


namespace oxen_count_b_l223_223387

theorem oxen_count_b 
  (a_oxen : ℕ) (a_months : ℕ)
  (b_months : ℕ) (x : ℕ)
  (c_oxen : ℕ) (c_months : ℕ)
  (total_rent : ℝ) (c_rent : ℝ)
  (h1 : a_oxen * a_months = 70)
  (h2 : c_oxen * c_months = 45)
  (h3 : c_rent / total_rent = 27 / 105)
  (h4 : total_rent = 105) :
  x = 12 :=
by 
  sorry

end oxen_count_b_l223_223387


namespace boys_count_l223_223560

theorem boys_count (B G : ℕ) (h1 : B + G = 41) (h2 : 12 * B + 8 * G = 460) : B = 33 := 
by
  sorry

end boys_count_l223_223560


namespace general_formula_a_n_T_n_greater_than_S_n_l223_223572

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l223_223572


namespace average_minutes_run_per_day_l223_223955

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end average_minutes_run_per_day_l223_223955


namespace smaller_variance_stability_l223_223122

variable {α : Type*}
variable [Nonempty α]

def same_average (X Y : α → ℝ) (avg : ℝ) : Prop := 
  (∀ x, X x = avg) ∧ (∀ y, Y y = avg)

def smaller_variance_is_stable (X Y : α → ℝ) : Prop := 
  (X = Y)

theorem smaller_variance_stability {X Y : α → ℝ} (avg : ℝ) :
  same_average X Y avg → smaller_variance_is_stable X Y :=
by sorry

end smaller_variance_stability_l223_223122


namespace domain_correct_l223_223300

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end domain_correct_l223_223300


namespace total_weight_moved_l223_223294

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end total_weight_moved_l223_223294


namespace possible_final_state_l223_223782

-- Definitions of initial conditions and operations
def initial_urn : (ℕ × ℕ) := (100, 100)  -- (W, B)

-- Define operations that describe changes in (white, black) marbles
inductive Operation
| operation1 : Operation
| operation2 : Operation
| operation3 : Operation
| operation4 : Operation

def apply_operation (op : Operation) (state : ℕ × ℕ) : ℕ × ℕ :=
  match op with
  | Operation.operation1 => (state.1, state.2 - 2)
  | Operation.operation2 => (state.1, state.2 - 1)
  | Operation.operation3 => (state.1, state.2 - 1)
  | Operation.operation4 => (state.1 - 2, state.2 + 1)

-- The final state in the form of the specific condition to prove.
def final_state (state : ℕ × ℕ) : Prop :=
  state = (2, 0)  -- 2 white marbles are an expected outcome.

-- Statement of the problem in Lean
theorem possible_final_state : ∃ (sequence : List Operation), 
  (sequence.foldl (fun state op => apply_operation op state) initial_urn).1 = 2 :=
sorry

end possible_final_state_l223_223782


namespace length_of_train_l223_223346

theorem length_of_train (speed : ℝ) (time : ℝ) (h1: speed = 48 * (1000 / 3600) * (1 / 1)) (h2: time = 9) : 
  (speed * time) = 119.97 :=
by
  sorry

end length_of_train_l223_223346


namespace maximize_perimeter_l223_223103

theorem maximize_perimeter 
  (l : ℝ) (c_f : ℝ) (C : ℝ) (b : ℝ)
  (hl: l = 400) (hcf: c_f = 5) (hC: C = 1500) :
  ∃ (y : ℝ), y = 180 :=
by
  sorry

end maximize_perimeter_l223_223103


namespace increasing_condition_l223_223382

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) + a * (Real.exp (-x))

theorem increasing_condition (a : ℝ) : (∀ x : ℝ, 0 ≤ (Real.exp (2 * x) - a) / (Real.exp x)) ↔ a ≤ 0 :=
by
  sorry

end increasing_condition_l223_223382


namespace no_n_such_that_n_times_s_is_20222022_l223_223441

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem
theorem no_n_such_that_n_times_s_is_20222022 :
  ∀ n : ℕ, n * sum_of_digits n ≠ 20222022 :=
by
  sorry

end no_n_such_that_n_times_s_is_20222022_l223_223441


namespace eval_x2_sub_y2_l223_223584

theorem eval_x2_sub_y2 (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x + y = 13) : x^2 - y^2 = -40 := by
  sorry

end eval_x2_sub_y2_l223_223584


namespace race_min_distance_l223_223107

noncomputable def min_distance : ℝ :=
  let A : ℝ × ℝ := (0, 300)
  let B : ℝ × ℝ := (1200, 500)
  let wall_length : ℝ := 1200
  let B' : ℝ × ℝ := (1200, -500)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B'

theorem race_min_distance :
  min_distance = 1442 := sorry

end race_min_distance_l223_223107


namespace units_digit_31_2020_units_digit_37_2020_l223_223617

theorem units_digit_31_2020 : ((31 ^ 2020) % 10) = 1 := by
  sorry

theorem units_digit_37_2020 : ((37 ^ 2020) % 10) = 1 := by
  sorry

end units_digit_31_2020_units_digit_37_2020_l223_223617


namespace h_at_2_l223_223395

noncomputable def h (x : ℝ) : ℝ := 
(x + 2) * (x - 1) * (x + 4) * (x - 3) - x^2

theorem h_at_2 : 
  h (-2) = -4 ∧ h (1) = -1 ∧ h (-4) = -16 ∧ h (3) = -9 → h (2) = -28 := 
by
  intro H
  sorry

end h_at_2_l223_223395


namespace ratio_equivalence_l223_223735

theorem ratio_equivalence (a b : ℝ) (hb : b ≠ 0) (h : a / b = 5 / 4) : (4 * a + 3 * b) / (4 * a - 3 * b) = 4 :=
sorry

end ratio_equivalence_l223_223735


namespace reciprocal_sum_is_1_implies_at_least_one_is_2_l223_223544

-- Lean statement for the problem
theorem reciprocal_sum_is_1_implies_at_least_one_is_2 (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1) : 
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := 
sorry

end reciprocal_sum_is_1_implies_at_least_one_is_2_l223_223544


namespace negation_universal_proposition_l223_223219

theorem negation_universal_proposition :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by sorry

end negation_universal_proposition_l223_223219


namespace find_three_digit_number_l223_223832

theorem find_three_digit_number (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : P ≠ R) 
  (h3 : Q ≠ R) 
  (h4 : P < 7) 
  (h5 : Q < 7) 
  (h6 : R < 7)
  (h7 : P ≠ 0) 
  (h8 : Q ≠ 0) 
  (h9 : R ≠ 0) 
  (h10 : 7 * P + Q + R = 7 * R) 
  (h11 : (7 * P + Q) + (7 * Q + P) = 49 + 7 * R + R)
  : P * 100 + Q * 10 + R = 434 :=
sorry

end find_three_digit_number_l223_223832


namespace no_perfect_powers_in_sequence_l223_223474

noncomputable def nth_triplet (n : Nat) : Nat × Nat × Nat :=
  Nat.recOn n (2, 3, 5) (λ _ ⟨a, b, c⟩ => (a + c, a + b, b + c))

def is_perfect_power (x : Nat) : Prop :=
  ∃ (m : Nat) (k : Nat), k ≥ 2 ∧ m^k = x

theorem no_perfect_powers_in_sequence : ∀ (n : Nat), ∀ (a b c : Nat),
  nth_triplet n = (a, b, c) →
  ¬(is_perfect_power a ∨ is_perfect_power b ∨ is_perfect_power c) :=
by
  intros
  sorry

end no_perfect_powers_in_sequence_l223_223474


namespace hash_fn_triple_40_l223_223556

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_fn_triple_40 : hash_fn (hash_fn (hash_fn 40)) = 12.56 := by
  sorry

end hash_fn_triple_40_l223_223556


namespace trigonometric_unique_solution_l223_223058

theorem trigonometric_unique_solution :
  (∃ x : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ 0 ≤ y ∧ y < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 ∧
    Real.sin y = 0.6 ∧ Real.cos y = 0.8 → x = y) :=
by
  sorry

end trigonometric_unique_solution_l223_223058


namespace ratio_of_areas_l223_223143

theorem ratio_of_areas (r : ℝ) (s1 s2 : ℝ) 
  (h1 : s1^2 = 4 / 5 * r^2)
  (h2 : s2^2 = 2 * r^2) :
  (s1^2 / s2^2) = 2 / 5 := by
  sorry

end ratio_of_areas_l223_223143


namespace sum_SHE_equals_6_l223_223314

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

end sum_SHE_equals_6_l223_223314


namespace original_weight_of_apple_box_l223_223410

theorem original_weight_of_apple_box:
  ∀ (x : ℕ), (3 * x - 12 = x) → x = 6 :=
by
  intros x h
  sorry

end original_weight_of_apple_box_l223_223410


namespace find_m_l223_223479

variable {m : ℝ}

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vector_diff (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_m (hm: dot_product vector_a (vector_diff vector_a (vector_b m)) = 0) : m = 3 :=
  by
  sorry

end find_m_l223_223479


namespace rectangle_area_192_l223_223946

variable (b l : ℝ) (A : ℝ)

-- Conditions
def length_is_thrice_breadth : Prop :=
  l = 3 * b

def perimeter_is_64 : Prop :=
  2 * (l + b) = 64

-- Area calculation
def area_of_rectangle : ℝ :=
  l * b

theorem rectangle_area_192 (h1 : length_is_thrice_breadth b l) (h2 : perimeter_is_64 b l) :
  area_of_rectangle l b = 192 := by
  sorry

end rectangle_area_192_l223_223946


namespace initial_bottle_caps_l223_223426

variable (initial_caps added_caps total_caps : ℕ)

theorem initial_bottle_caps 
  (h1 : added_caps = 7) 
  (h2 : total_caps = 14) 
  (h3 : total_caps = initial_caps + added_caps) : 
  initial_caps = 7 := 
by 
  sorry

end initial_bottle_caps_l223_223426


namespace product_of_solutions_l223_223805

theorem product_of_solutions (t : ℝ) (h : t^2 = 64) : t * (-t) = -64 :=
sorry

end product_of_solutions_l223_223805


namespace first_snail_time_proof_l223_223065

-- Define the conditions
def first_snail_speed := 2 -- speed in feet per minute
def second_snail_speed := 2 * first_snail_speed
def third_snail_speed := 5 * second_snail_speed
def third_snail_time := 2 -- time in minutes
def distance := third_snail_speed * third_snail_time

-- Define the time it took the first snail
def first_snail_time := distance / first_snail_speed

-- Define the theorem to be proven
theorem first_snail_time_proof : first_snail_time = 20 := 
by
  -- Proof should be filled here
  sorry

end first_snail_time_proof_l223_223065


namespace find_annual_interest_rate_l223_223308

variable (r : ℝ) -- The annual interest rate we want to prove

-- Define the conditions based on the problem statement
variable (I : ℝ := 300) -- interest earned
variable (P : ℝ := 10000) -- principal amount
variable (t : ℝ := 9 / 12) -- time in years

-- Define the simple interest formula condition
def simple_interest_formula : Prop :=
  I = P * r * t

-- The statement to prove
theorem find_annual_interest_rate : simple_interest_formula r ↔ r = 0.04 :=
  by
    unfold simple_interest_formula
    simp
    sorry

end find_annual_interest_rate_l223_223308


namespace number_of_sides_of_polygon_l223_223469

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)
noncomputable def sum_known_angles : ℝ := 3780

theorem number_of_sides_of_polygon
  (n : ℕ)
  (h1 : sum_known_angles + missing_angle = sum_of_interior_angles n)
  (h2 : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a = 3 * c ∧ b = 3 * c ∧ a + b + c ≤ sum_known_angles) :
  n = 23 :=
sorry

end number_of_sides_of_polygon_l223_223469


namespace linear_func_passing_point_l223_223237

theorem linear_func_passing_point :
  ∃ k : ℝ, ∀ x y : ℝ, (y = k * x + 1) → (x = -1 ∧ y = 0) → k = 1 :=
by
  sorry

end linear_func_passing_point_l223_223237


namespace uranus_appearance_minutes_after_6AM_l223_223764

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end uranus_appearance_minutes_after_6AM_l223_223764


namespace usual_time_to_catch_bus_l223_223550

variable {S T T' D : ℝ}

theorem usual_time_to_catch_bus (h1 : D = S * T)
  (h2 : D = (4 / 5) * S * T')
  (h3 : T' = T + 4) : T = 16 := by
  sorry

end usual_time_to_catch_bus_l223_223550


namespace bacteria_growth_time_l223_223672
-- Import necessary library

-- Define the conditions
def initial_bacteria_count : ℕ := 100
def final_bacteria_count : ℕ := 102400
def multiplication_factor : ℕ := 4
def multiplication_period_hours : ℕ := 6

-- Define the proof problem
theorem bacteria_growth_time :
  ∃ t : ℕ, t * multiplication_period_hours = 30 ∧ initial_bacteria_count * multiplication_factor^t = final_bacteria_count :=
by
  sorry

end bacteria_growth_time_l223_223672


namespace surface_area_of_cylinder_l223_223282

noncomputable def cylinder_surface_area
    (r : ℝ) (V : ℝ) (S : ℝ) : Prop :=
    r = 1 ∧ V = 2 * Real.pi ∧ S = 6 * Real.pi

theorem surface_area_of_cylinder
    (r : ℝ) (V : ℝ) : ∃ S : ℝ, cylinder_surface_area r V S :=
by
  use 6 * Real.pi
  sorry

end surface_area_of_cylinder_l223_223282


namespace susan_annual_percentage_increase_l223_223609

theorem susan_annual_percentage_increase :
  let initial_jerry := 14400
  let initial_susan := 6250
  let jerry_first_year := initial_jerry * (6 / 5 : ℝ)
  let jerry_second_year := jerry_first_year * (9 / 10 : ℝ)
  let jerry_third_year := jerry_second_year * (6 / 5 : ℝ)
  jerry_third_year = 18662.40 →
  (initial_susan : ℝ) * (1 + r)^3 = 18662.40 →
  r = 0.44 :=
by {
  sorry
}

end susan_annual_percentage_increase_l223_223609


namespace probability_of_mathematics_letter_l223_223222

-- Definitions for the problem
def english_alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

-- Set the total number of letters in the English alphabet
def total_letters := english_alphabet.card

-- Set the number of unique letters in 'MATHEMATICS'
def mathematics_unique_letters := mathematics_letters.card

-- Statement of the Lean theorem
theorem probability_of_mathematics_letter : (mathematics_unique_letters : ℚ) / total_letters = 4 / 13 :=
by
  sorry

end probability_of_mathematics_letter_l223_223222


namespace line_slope_intercept_l223_223084

theorem line_slope_intercept (a b : ℝ) 
  (h1 : (7 : ℝ) = a * 3 + b) 
  (h2 : (13 : ℝ) = a * (9/2) + b) : 
  a - b = 9 := 
sorry

end line_slope_intercept_l223_223084


namespace even_function_has_zero_coefficient_l223_223374

theorem even_function_has_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (x^2 + a*x) = (x^2 + a*(-x))) → a = 0 :=
by
  intro h
  -- the proof part is omitted as requested
  sorry

end even_function_has_zero_coefficient_l223_223374


namespace opposite_face_is_D_l223_223856

-- Define the six faces
inductive Face
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def is_adjacent (x y : Face) : Prop :=
(y = B ∧ x = A) ∨ (y = F ∧ x = A) ∨ (y = C ∧ x = A) ∨ (y = E ∧ x = A)

-- Define the problem statement in Lean
theorem opposite_face_is_D : 
  (∀ (x : Face), is_adjacent A x ↔ x = B ∨ x = F ∨ x = C ∨ x = E) →
  (¬ (is_adjacent A D)) →
  True :=
by
  intro adj_relation non_adj_relation
  sorry

end opposite_face_is_D_l223_223856


namespace sugar_needed_for_40_cookies_l223_223246

def num_cookies_per_cup_flour (a : ℕ) (b : ℕ) : ℕ := a / b

def cups_of_flour_needed (num_cookies : ℕ) (cookies_per_cup : ℕ) : ℕ := num_cookies / cookies_per_cup

def cups_of_sugar_needed (cups_flour : ℕ) (flour_to_sugar_ratio_num : ℕ) (flour_to_sugar_ratio_denom : ℕ) : ℚ := 
  (flour_to_sugar_ratio_denom * cups_flour : ℚ) / flour_to_sugar_ratio_num

theorem sugar_needed_for_40_cookies :
  let num_flour_to_make_24_cookies := 3
  let cookies := 24
  let ratio_num := 3
  let ratio_denom := 2
  num_cookies_per_cup_flour cookies num_flour_to_make_24_cookies = 8 →
  cups_of_flour_needed 40 8 = 5 →
  cups_of_sugar_needed 5 ratio_num ratio_denom = 10 / 3 :=
by 
  sorry

end sugar_needed_for_40_cookies_l223_223246


namespace lateral_surface_area_of_cone_l223_223516

theorem lateral_surface_area_of_cone (r h : ℝ) (r_is_4 : r = 4) (h_is_3 : h = 3) :
  ∃ A : ℝ, A = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_cone_l223_223516


namespace sum_nat_numbers_from_1_to_5_l223_223618

theorem sum_nat_numbers_from_1_to_5 : (1 + 2 + 3 + 4 + 5 = 15) :=
by
  sorry

end sum_nat_numbers_from_1_to_5_l223_223618


namespace apples_in_each_crate_l223_223943

theorem apples_in_each_crate
  (num_crates : ℕ) 
  (num_rotten : ℕ) 
  (num_boxes : ℕ) 
  (apples_per_box : ℕ) 
  (total_good_apples : ℕ) 
  (total_apples : ℕ)
  (h1 : num_crates = 12) 
  (h2 : num_rotten = 160) 
  (h3 : num_boxes = 100) 
  (h4 : apples_per_box = 20) 
  (h5 : total_good_apples = num_boxes * apples_per_box) 
  (h6 : total_apples = total_good_apples + num_rotten) : 
  total_apples / num_crates = 180 := 
by 
  sorry

end apples_in_each_crate_l223_223943


namespace find_xyz_l223_223499

variable (x y z : ℝ)

theorem find_xyz (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 192) : x * y * z = 842 :=
sorry

end find_xyz_l223_223499


namespace find_angle_A_l223_223740

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) 
  (h3 : B = Real.pi / 4) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l223_223740


namespace fifteenth_term_is_44_l223_223816

-- Define the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 3
def term_number : ℕ := 15

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Prove that the 15th term is 44
theorem fifteenth_term_is_44 : nth_term first_term common_difference term_number = 44 :=
by
  unfold nth_term first_term common_difference term_number
  sorry

end fifteenth_term_is_44_l223_223816


namespace at_least_one_not_less_than_neg_two_l223_223696

theorem at_least_one_not_less_than_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≥ -2 ∨ b + 1/c ≥ -2 ∨ c + 1/a ≥ -2) :=
sorry

end at_least_one_not_less_than_neg_two_l223_223696


namespace spring_length_function_l223_223697

noncomputable def spring_length (x : ℝ) : ℝ :=
  12 + 3 * x

theorem spring_length_function :
  ∀ (x : ℝ), spring_length x = 12 + 3 * x :=
by
  intro x
  rfl

end spring_length_function_l223_223697


namespace pete_should_leave_by_0730_l223_223612

def walking_time : ℕ := 10
def train_time : ℕ := 80
def latest_arrival_time : String := "0900"
def departure_time : String := "0730"

theorem pete_should_leave_by_0730 :
  (latest_arrival_time = "0900" → walking_time = 10 ∧ train_time = 80 → departure_time = "0730") := by
  sorry

end pete_should_leave_by_0730_l223_223612


namespace general_formula_a_sum_sn_l223_223287

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 * n

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ :=
  a n + 2 ^ (a n)

-- Define the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem general_formula_a :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_sn :
  ∀ n, S n = n * (n + 1) + (4^(n + 1) - 4) / 3 :=
sorry

end general_formula_a_sum_sn_l223_223287


namespace value_of_A_l223_223111

def random_value (c : Char) : ℤ := sorry

-- Given conditions
axiom H_value : random_value 'H' = 12
axiom MATH_value : random_value 'M' + random_value 'A' + random_value 'T' + random_value 'H' = 40
axiom TEAM_value : random_value 'T' + random_value 'E' + random_value 'A' + random_value 'M' = 50
axiom MEET_value : random_value 'M' + random_value 'E' + random_value 'E' + random_value 'T' = 44

-- Prove that A = 28
theorem value_of_A : random_value 'A' = 28 := by
  sorry

end value_of_A_l223_223111


namespace lincoln_high_students_club_overlap_l223_223212

theorem lincoln_high_students_club_overlap (total_students : ℕ)
  (drama_club_students science_club_students both_or_either_club_students : ℕ)
  (h1 : total_students = 500)
  (h2 : drama_club_students = 150)
  (h3 : science_club_students = 200)
  (h4 : both_or_either_club_students = 300) :
  drama_club_students + science_club_students - both_or_either_club_students = 50 :=
by
  sorry

end lincoln_high_students_club_overlap_l223_223212


namespace intersection_is_23_l223_223585

open Set

def setA : Set ℤ := {1, 2, 3, 4}
def setB : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem intersection_is_23 : setA ∩ setB = {2, 3} := 
by 
  sorry

end intersection_is_23_l223_223585


namespace fraction_of_students_with_partner_l223_223009

theorem fraction_of_students_with_partner
  (a b : ℕ)
  (condition1 : ∀ seventh, seventh ≠ 0 → ∀ tenth, tenth ≠ 0 → a * b = 0)
  (condition2 : b / 4 = (3 * a) / 7) :
  (b / 4 + 3 * a / 7) / (b + a) = 6 / 19 :=
by
  sorry

end fraction_of_students_with_partner_l223_223009


namespace bricks_in_wall_l223_223828

-- Definitions of conditions based on the problem statement
def time_first_bricklayer : ℝ := 12 
def time_second_bricklayer : ℝ := 15 
def reduced_productivity : ℝ := 12 
def combined_time : ℝ := 6
def total_bricks : ℝ := 720

-- Lean 4 statement of the proof problem
theorem bricks_in_wall (x : ℝ) 
  (h1 : (x / time_first_bricklayer + x / time_second_bricklayer - reduced_productivity) * combined_time = x) 
  : x = total_bricks := 
by {
  sorry
}

end bricks_in_wall_l223_223828


namespace non_negative_real_inequality_l223_223377

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end non_negative_real_inequality_l223_223377


namespace mr_william_farm_tax_l223_223262

noncomputable def total_tax_collected : ℝ := 3840
noncomputable def mr_william_percentage : ℝ := 16.666666666666668 / 100  -- Convert percentage to decimal

theorem mr_william_farm_tax : (total_tax_collected * mr_william_percentage) = 640 := by
  sorry

end mr_william_farm_tax_l223_223262


namespace movie_ticket_final_price_l223_223523

noncomputable def final_ticket_price (initial_price : ℝ) : ℝ :=
  let price_year_1 := initial_price * 1.12
  let price_year_2 := price_year_1 * 0.95
  let price_year_3 := price_year_2 * 1.08
  let price_year_4 := price_year_3 * 0.96
  let price_year_5 := price_year_4 * 1.06
  let price_after_tax := price_year_5 * 1.07
  let final_price := price_after_tax * 0.90
  final_price

theorem movie_ticket_final_price :
  final_ticket_price 100 = 112.61 := by
  sorry

end movie_ticket_final_price_l223_223523


namespace monotonicity_and_range_of_m_l223_223656

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 - a) / 2 * x ^ 2 + a * x - Real.log x

theorem monotonicity_and_range_of_m (a m : ℝ) (h₀ : 2 < a) (h₁ : a < 3)
  (h₂ : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 -> ma + Real.log 2 > |f x1 a - f x2 a|):
  m ≥ 0 :=
sorry

end monotonicity_and_range_of_m_l223_223656


namespace compare_logarithms_l223_223811

noncomputable def a : ℝ := Real.log 3 / Real.log 4 -- log base 4 of 3
noncomputable def b : ℝ := Real.log 4 / Real.log 3 -- log base 3 of 4
noncomputable def c : ℝ := Real.log 3 / Real.log 5 -- log base 5 of 3

theorem compare_logarithms : b > a ∧ a > c := sorry

end compare_logarithms_l223_223811


namespace scientific_notation_l223_223388

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l223_223388


namespace granger_total_payment_proof_l223_223683

-- Conditions
def cost_per_can_spam := 3
def cost_per_jar_peanut_butter := 5
def cost_per_loaf_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Calculation
def total_cost_spam := quantity_spam * cost_per_can_spam
def total_cost_peanut_butter := quantity_peanut_butter * cost_per_jar_peanut_butter
def total_cost_bread := quantity_bread * cost_per_loaf_bread

-- Total amount paid
def total_amount_paid := total_cost_spam + total_cost_peanut_butter + total_cost_bread

-- Theorem to be proven
theorem granger_total_payment_proof : total_amount_paid = 59 :=
by
  sorry

end granger_total_payment_proof_l223_223683


namespace sqrt_difference_of_cubes_is_integer_l223_223326

theorem sqrt_difference_of_cubes_is_integer (a b : ℕ) (h1 : a = 105) (h2 : b = 104) :
  (Int.sqrt (a^3 - b^3) = 181) :=
by
  sorry

end sqrt_difference_of_cubes_is_integer_l223_223326


namespace fraction_equivalence_l223_223790

theorem fraction_equivalence (a b c : ℝ) (h : (c - a) / (c - b) = 1) : 
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by
  sorry

end fraction_equivalence_l223_223790


namespace fraction_numerator_exceeds_denominator_l223_223089

theorem fraction_numerator_exceeds_denominator (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 3) :
  4 * x + 5 > 10 - 3 * x ↔ (5 / 7) < x ∧ x ≤ 3 :=
by 
  sorry

end fraction_numerator_exceeds_denominator_l223_223089


namespace min_value_expr_l223_223652

variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 1)

theorem min_value_expr : (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 :=
by
  sorry

end min_value_expr_l223_223652


namespace lines_through_point_l223_223253

theorem lines_through_point (k : ℝ) : ∀ x y : ℝ, (y = k * (x - 1)) ↔ (x = 1 ∧ y = 0) ∨ (x ≠ 1 ∧ y / (x - 1) = k) :=
by
  sorry

end lines_through_point_l223_223253


namespace satisfies_equation_l223_223575

noncomputable def y (b x : ℝ) : ℝ := (b + x) / (1 + b * x)

theorem satisfies_equation (b x : ℝ) :
  let y_val := y b x
  let y_prime := (1 - b^2) / (1 + b * x)^2
  y_val - x * y_prime = b * (1 + x^2 * y_prime) :=
by
  sorry

end satisfies_equation_l223_223575


namespace pants_cost_correct_l223_223596

def shirt_cost : ℕ := 43
def tie_cost : ℕ := 15
def total_paid : ℕ := 200
def change_received : ℕ := 2

def total_spent : ℕ := total_paid - change_received
def combined_cost : ℕ := shirt_cost + tie_cost
def pants_cost : ℕ := total_spent - combined_cost

theorem pants_cost_correct : pants_cost = 140 :=
by
  -- We'll leave the proof as an exercise.
  sorry

end pants_cost_correct_l223_223596


namespace arithmetic_progression_probability_l223_223205

theorem arithmetic_progression_probability (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_outcomes = 6^4 ∧ favorable_outcomes = 3 →
  favorable_outcomes / total_outcomes = 1 / 432 :=
by
  sorry

end arithmetic_progression_probability_l223_223205


namespace round_trip_ticket_percentage_l223_223804

variable (P : ℝ) -- Denotes total number of passengers
variable (R : ℝ) -- Denotes number of round-trip ticket holders

-- Condition 1: 15% of passengers held round-trip tickets and took their cars aboard
def condition1 : Prop := 0.15 * P = 0.40 * R

-- Prove that 37.5% of the ship's passengers held round-trip tickets.
theorem round_trip_ticket_percentage (h1 : condition1 P R) : R / P = 0.375 :=
by
  sorry

end round_trip_ticket_percentage_l223_223804


namespace value_of_expression_when_x_is_3_l223_223744

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end value_of_expression_when_x_is_3_l223_223744


namespace fewest_tiles_needed_l223_223610

def tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let length_tiles := (region_length + tile_length - 1) / tile_length
  let width_tiles := (region_width + tile_width - 1) / tile_width
  length_tiles * width_tiles

theorem fewest_tiles_needed :
  let tile_length := 2
  let tile_width := 5
  let region_length := 36
  let region_width := 72
  tiles_needed tile_length tile_width region_length region_width = 270 :=
by
  sorry

end fewest_tiles_needed_l223_223610


namespace cannot_determine_a_l223_223839

theorem cannot_determine_a 
  (n : ℝ) 
  (p : ℝ) 
  (a : ℝ) 
  (line_eq : ∀ (x y : ℝ), x = 5 * y + 5) 
  (pt1 : a = 5 * n + 5) 
  (pt2 : a + 2 = 5 * (n + p) + 5) : p = 0.4 → ¬∀ a' : ℝ, a = a' :=
by
  sorry

end cannot_determine_a_l223_223839


namespace sequence_total_sum_is_correct_l223_223467

-- Define the sequence pattern
def sequence_sum : ℕ → ℤ
| 0       => 1
| 1       => -2
| 2       => -4
| 3       => 8
| (n + 4) => sequence_sum n + 4

-- Define the number of groups in the sequence
def num_groups : ℕ := 319

-- Define the sum of each individual group
def group_sum : ℤ := 3

-- Define the total sum of the sequence
def total_sum : ℤ := num_groups * group_sum

theorem sequence_total_sum_is_correct : total_sum = 957 := by
  sorry

end sequence_total_sum_is_correct_l223_223467


namespace can_capacity_is_14_l223_223894

noncomputable def capacity_of_can 
    (initial_milk: ℝ) (initial_water: ℝ) 
    (added_milk: ℝ) (ratio_initial: ℝ) (ratio_final: ℝ): ℝ :=
  initial_milk + initial_water + added_milk

theorem can_capacity_is_14
    (M W: ℝ) 
    (ratio_initial : M / W = 1 / 5) 
    (added_milk : ℝ := 2) 
    (ratio_final:  (M + 2) / W = 2.00001 / 5.00001): 
    capacity_of_can M W added_milk (1 / 5) (2.00001 / 5.00001) = 14 := 
  by
    sorry

end can_capacity_is_14_l223_223894


namespace cricket_player_average_l223_223156

theorem cricket_player_average
  (A : ℕ)
  (h1 : 8 * A + 96 = 9 * (A + 8)) :
  A = 24 :=
by
  sorry

end cricket_player_average_l223_223156


namespace find_original_strength_l223_223328

variable (original_strength : ℕ)
variable (total_students : ℕ := original_strength + 12)
variable (original_avg_age : ℕ := 40)
variable (new_students : ℕ := 12)
variable (new_students_avg_age : ℕ := 32)
variable (new_avg_age_reduction : ℕ := 4)
variable (new_avg_age : ℕ := original_avg_age - new_avg_age_reduction)

theorem find_original_strength (h : (original_avg_age * original_strength + new_students * new_students_avg_age) / total_students = new_avg_age) :
  original_strength = 12 := 
sorry

end find_original_strength_l223_223328


namespace more_plastic_pipe_l223_223833

variable (m_copper m_plastic : Nat)
variable (total_cost cost_per_meter : Nat)

-- Conditions
variable (h1 : m_copper = 10)
variable (h2 : cost_per_meter = 4)
variable (h3 : total_cost = 100)
variable (h4 : m_copper * cost_per_meter + m_plastic * cost_per_meter = total_cost)

-- Proof that the number of more meters of plastic pipe bought compared to the copper pipe is 5
theorem more_plastic_pipe :
  m_plastic - m_copper = 5 :=
by
  -- Since proof is not required, we place sorry here.
  sorry

end more_plastic_pipe_l223_223833


namespace cindy_correct_answer_l223_223678

theorem cindy_correct_answer (x : ℝ) (h : (x - 10) / 5 = 50) : (x - 5) / 10 = 25.5 :=
sorry

end cindy_correct_answer_l223_223678


namespace incorrect_equation_a_neq_b_l223_223522

theorem incorrect_equation_a_neq_b (a b : ℝ) (h : a ≠ b) : a - b ≠ b - a :=
  sorry

end incorrect_equation_a_neq_b_l223_223522


namespace range_of_a_l223_223054

theorem range_of_a (a : ℝ) : (∀ x : ℤ, x > 2 * a - 3 ∧ 2 * (x : ℝ) ≥ 3 * ((x : ℝ) - 2) + 5) ↔ (1 / 2 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l223_223054


namespace more_likely_to_return_to_initial_count_l223_223958

noncomputable def P_A (a b c d : ℕ) : ℚ :=
(b * (d + 1) + a * (c + 1)) / (50 * 51)

noncomputable def P_A_bar (a b c d : ℕ) : ℚ :=
(b * c + a * d) / (50 * 51)

theorem more_likely_to_return_to_initial_count (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (h3 : b ≥ a) (h4 : d ≥ c - 1) (h5 : a > 0) :
P_A a b c d > P_A_bar a b c d := by
  sorry

end more_likely_to_return_to_initial_count_l223_223958


namespace transformed_line_equation_l223_223343

theorem transformed_line_equation {A B C x₀ y₀ : ℝ} 
    (h₀ : ¬(A = 0 ∧ B = 0)) 
    (h₁ : A * x₀ + B * y₀ + C = 0) : 
    ∀ {x y : ℝ}, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
by
    sorry

end transformed_line_equation_l223_223343


namespace algebraic_expression_evaluation_l223_223465

theorem algebraic_expression_evaluation (x m : ℝ) (h1 : 5 * (2 - 1) + 3 * m * 2 = -7) (h2 : m = -2) :
  5 * (x - 1) + 3 * m * x = -1 ↔ x = -4 :=
by
  sorry

end algebraic_expression_evaluation_l223_223465


namespace vertical_angles_congruent_l223_223259

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end vertical_angles_congruent_l223_223259


namespace root_magnitude_conditions_l223_223213

theorem root_magnitude_conditions (p : ℝ) (h : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = -p) ∧ (r1 * r2 = -12)) :
  (∃ r1 r2 : ℝ, (r1 ≠ r2) ∧ |r1| > 2 ∨ |r2| > 2) ∧ (∀ r1 r2 : ℝ, (r1 + r2 = -p) ∧ (r1 * r2 = -12) → |r1| * |r2| ≤ 14) :=
by
  -- Proof of the theorem goes here
  sorry

end root_magnitude_conditions_l223_223213


namespace find_d_value_l223_223445

theorem find_d_value (a b : ℚ) (d : ℚ) (h1 : a = 2) (h2 : b = 11) 
  (h3 : ∀ x, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) : 
  d = 53 / 4 :=
sorry

end find_d_value_l223_223445


namespace box_problem_l223_223196

theorem box_problem 
    (x y : ℕ) 
    (h1 : 10 * x + 20 * y = 18 * (x + y)) 
    (h2 : 10 * x + 20 * (y - 10) = 16 * (x + y - 10)) :
    x + y = 20 :=
sorry

end box_problem_l223_223196


namespace first_year_after_2020_with_digit_sum_4_l223_223538

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end first_year_after_2020_with_digit_sum_4_l223_223538


namespace domain_of_function_l223_223229

theorem domain_of_function :
  (∀ x : ℝ, 2 + x ≥ 0 ∧ 3 - x ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry

end domain_of_function_l223_223229


namespace team_points_difference_l223_223659

   -- Definitions for points of each member
   def Max_points : ℝ := 7
   def Dulce_points : ℝ := 5
   def Val_points : ℝ := 4 * (Max_points + Dulce_points)
   def Sarah_points : ℝ := 2 * Dulce_points
   def Steve_points : ℝ := 2.5 * (Max_points + Val_points)

   -- Definition for total points of their team
   def their_team_points : ℝ := Max_points + Dulce_points + Val_points + Sarah_points + Steve_points

   -- Definition for total points of the opponents' team
   def opponents_team_points : ℝ := 200

   -- The main theorem to prove
   theorem team_points_difference : their_team_points - opponents_team_points = 7.5 := by
     sorry
   
end team_points_difference_l223_223659


namespace arithmetic_sequence_a6_value_l223_223037

theorem arithmetic_sequence_a6_value (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_roots : ∀ x, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) :
  a 6 = -6 :=
by
  -- Definitions and given conditions would go here in a fully elaborated proof.
  sorry

end arithmetic_sequence_a6_value_l223_223037


namespace shot_put_surface_area_l223_223072

noncomputable def radius (d : ℝ) : ℝ := d / 2

noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem shot_put_surface_area :
  surface_area (radius 5) = 25 * Real.pi :=
by
  sorry

end shot_put_surface_area_l223_223072


namespace santiago_more_roses_l223_223127

def red_roses_santiago := 58
def red_roses_garrett := 24
def red_roses_difference := red_roses_santiago - red_roses_garrett

theorem santiago_more_roses : red_roses_difference = 34 := by
  sorry

end santiago_more_roses_l223_223127


namespace complex_repair_cost_l223_223324

theorem complex_repair_cost
  (charge_tire : ℕ)
  (cost_part_tire : ℕ)
  (num_tires : ℕ)
  (charge_complex : ℕ)
  (num_complex : ℕ)
  (profit_retail : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ)
  (profit_tire : ℕ := charge_tire - cost_part_tire)
  (total_profit_tire : ℕ := num_tires * profit_tire)
  (total_revenue_complex : ℕ := num_complex * charge_complex)
  (initial_profit : ℕ :=
    total_profit_tire + profit_retail - fixed_expenses)
  (needed_profit_complex : ℕ := total_profit - initial_profit) :
  needed_profit_complex = 100 / num_complex :=
by
  sorry

end complex_repair_cost_l223_223324


namespace solution_set_l223_223491

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l223_223491


namespace bears_in_stock_initially_l223_223055

theorem bears_in_stock_initially 
  (shipment_bears : ℕ) (shelf_bears : ℕ) (shelves_used : ℕ)
  (total_bears_shelved : shipment_bears + shelf_bears * shelves_used = 24) : 
  (24 - shipment_bears = 6) :=
by
  exact sorry

end bears_in_stock_initially_l223_223055


namespace determine_clothes_l223_223510

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l223_223510


namespace inscribed_angle_sum_l223_223666

theorem inscribed_angle_sum : 
  let arcs := 24 
  let arc_to_angle (n : ℕ) := 360 / arcs * n / 2 
  (arc_to_angle 4 + arc_to_angle 6 = 75) :=
by
  sorry

end inscribed_angle_sum_l223_223666


namespace find_sum_a_b_l223_223437

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end find_sum_a_b_l223_223437


namespace problem_l223_223757

theorem problem (a b c : ℝ) (Ha : a > 0) (Hb : b > 0) (Hc : c > 0) : 
  (|a| / a + |b| / b + |c| / c - (abc / |abc|) = 2 ∨ |a| / a + |b| / b + |c| / c - (abc / |abc|) = -2) :=
by
  sorry

end problem_l223_223757


namespace find_decimal_number_l223_223986

noncomputable def decimal_number (x : ℝ) : Prop := 
x > 0 ∧ (100000 * x = 5 * (1 / x))

theorem find_decimal_number {x : ℝ} (h : decimal_number x) : x = 1 / (100 * Real.sqrt 2) :=
by
  sorry

end find_decimal_number_l223_223986


namespace problem1_problem2_l223_223725

theorem problem1 : (-1 / 2) * (-8) + (-6) = -2 := by
  sorry

theorem problem2 : -(1^4) - 2 / (-1 / 3) - abs (-9) = -4 := by
  sorry

end problem1_problem2_l223_223725


namespace minimum_sugar_quantity_l223_223775

theorem minimum_sugar_quantity :
  ∃ s f : ℝ, s = 4 ∧ f ≥ 4 + s / 3 ∧ f ≤ 3 * s ∧ 2 * s + 3 * f ≤ 36 :=
sorry

end minimum_sugar_quantity_l223_223775


namespace prism_faces_l223_223466

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end prism_faces_l223_223466


namespace third_wins_against_seventh_l223_223151

-- Define the participants and their distinct points 
variables (p : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → p i ≠ p j)
-- descending order condition
variables (h_order : ∀ i j, i < j → p i > p j)
-- second place points equals sum of last four places
variables (h_second : p 2 = p 5 + p 6 + p 7 + p 8)

-- Theorem stating the third place player won against the seventh place player
theorem third_wins_against_seventh :
  p 3 > p 7 :=
sorry

end third_wins_against_seventh_l223_223151


namespace dream_clock_time_condition_l223_223521

theorem dream_clock_time_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 1)
  (h3 : (120 + 0.5 * 60 * x) = (240 - 6 * 60 * x)) :
  (4 + x) = 4 + 36 + 12 / 13 := by sorry

end dream_clock_time_condition_l223_223521


namespace beka_flies_more_l223_223688

-- Definitions
def beka_flight_distance : ℕ := 873
def jackson_flight_distance : ℕ := 563

-- The theorem we need to prove
theorem beka_flies_more : beka_flight_distance - jackson_flight_distance = 310 :=
by
  sorry

end beka_flies_more_l223_223688


namespace transport_cost_is_correct_l223_223873

-- Define the transport cost per kilogram
def transport_cost_per_kg : ℝ := 18000

-- Define the weight of the scientific instrument in kilograms
def weight_kg : ℝ := 0.5

-- Define the discount rate
def discount_rate : ℝ := 0.10

-- Define the cost calculation without the discount
def cost_without_discount : ℝ := weight_kg * transport_cost_per_kg

-- Define the final cost with the discount applied
def discounted_cost : ℝ := cost_without_discount * (1 - discount_rate)

-- The theorem stating that the discounted cost is $8,100
theorem transport_cost_is_correct : discounted_cost = 8100 := by
  sorry

end transport_cost_is_correct_l223_223873


namespace value_of_a_minus_b_l223_223274

theorem value_of_a_minus_b (a b : ℝ) :
  (∀ x, - (1 / 2 : ℝ) < x ∧ x < (1 / 3 : ℝ) → ax^2 + bx + 2 > 0) → a - b = -10 := by
sorry

end value_of_a_minus_b_l223_223274


namespace one_sixth_of_x_l223_223162

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 :=
sorry

end one_sixth_of_x_l223_223162


namespace count_zeros_in_10000_power_50_l223_223747

theorem count_zeros_in_10000_power_50 :
  10000^50 = 10^200 :=
by
  have h1 : 10000 = 10^4 := by sorry
  have h2 : (10^4)^50 = 10^(4 * 50) := by sorry
  exact h2.trans (by norm_num)

end count_zeros_in_10000_power_50_l223_223747


namespace cougar_ratio_l223_223022

theorem cougar_ratio (lions tigers total_cats cougars : ℕ) 
  (h_lions : lions = 12) 
  (h_tigers : tigers = 14) 
  (h_total : total_cats = 39) 
  (h_cougars : cougars = total_cats - (lions + tigers)) 
  : cougars * 2 = lions + tigers := 
by 
  rw [h_lions, h_tigers] 
  norm_num at * 
  sorry

end cougar_ratio_l223_223022


namespace minimum_triangle_area_l223_223417

theorem minimum_triangle_area :
  ∀ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (1 / m + 2 / n = 1) → (1 / 2 * m * n) = 4 :=
by
  sorry

end minimum_triangle_area_l223_223417


namespace workers_l223_223420

theorem workers (N C : ℕ) (h1 : N * C = 300000) (h2 : N * (C + 50) = 315000) : N = 300 :=
by
  sorry

end workers_l223_223420


namespace percentage_increase_from_March_to_January_l223_223270

variable {F J M : ℝ}

def JanuaryCondition (F J : ℝ) : Prop :=
  J = 0.90 * F

def MarchCondition (F M : ℝ) : Prop :=
  M = 0.75 * F

theorem percentage_increase_from_March_to_January (F J M : ℝ) (h1 : JanuaryCondition F J) (h2 : MarchCondition F M) :
  (J / M) = 1.20 := by 
  sorry

end percentage_increase_from_March_to_January_l223_223270


namespace tan_diff_l223_223830

theorem tan_diff (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) : 
  Real.tan (x - y) = 1 / 7 := 
by 
  sorry

end tan_diff_l223_223830


namespace range_of_a_l223_223248

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (2^x - 2^(-x))
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (2^x + 2^(-x))

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * f x + g (2 * x) ≥ 0) ↔ a ≥ -17 / 6 :=
by
  sorry

end range_of_a_l223_223248


namespace joker_then_spade_probability_correct_l223_223214

-- Defining the conditions of the deck
def deck_size : ℕ := 60
def joker_count : ℕ := 4
def suit_count : ℕ := 4
def cards_per_suit : ℕ := 15

-- The probability of drawing a Joker first and then a spade
def prob_joker_then_spade : ℚ :=
  (joker_count * (cards_per_suit - 1) + (deck_size - joker_count) * cards_per_suit) /
  (deck_size * (deck_size - 1))

-- The expected probability according to the solution
def expected_prob : ℚ := 224 / 885

theorem joker_then_spade_probability_correct :
  prob_joker_then_spade = expected_prob :=
by
  -- Skipping the actual proof steps
  sorry

end joker_then_spade_probability_correct_l223_223214


namespace complex_right_triangle_l223_223131

open Complex

theorem complex_right_triangle {z1 z2 a b : ℂ}
  (h1 : z2 = I * z1)
  (h2 : z1 + z2 = -a)
  (h3 : z1 * z2 = b) :
  a^2 / b = 2 :=
by sorry

end complex_right_triangle_l223_223131


namespace david_completion_time_l223_223518

theorem david_completion_time :
  (∃ D : ℕ, ∀ t : ℕ, 6 * (1 / D) + 3 * ((1 / D) + (1 / t)) = 1 -> D = 12) :=
sorry

end david_completion_time_l223_223518


namespace price_reduction_2100_yuan_l223_223705

-- Definitions based on conditions
def initial_sales : ℕ := 30
def initial_profit_per_item : ℕ := 50
def additional_sales_per_yuan (x : ℕ) : ℕ := 2 * x
def new_profit_per_item (x : ℕ) : ℕ := 50 - x
def target_profit : ℕ := 2100

-- Final proof statement, showing the price reduction needed
theorem price_reduction_2100_yuan (x : ℕ) 
  (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 := 
by 
  sorry

end price_reduction_2100_yuan_l223_223705


namespace angle_B_in_arithmetic_sequence_l223_223512

theorem angle_B_in_arithmetic_sequence (A B C : ℝ) (h_triangle_sum : A + B + C = 180) (h_arithmetic_sequence : 2 * B = A + C) : B = 60 := 
by 
  -- proof omitted
  sorry

end angle_B_in_arithmetic_sequence_l223_223512


namespace product_of_numbers_is_178_5_l223_223134

variables (a b c d : ℚ)

def sum_eq_36 := a + b + c + d = 36
def first_num_cond := a = 3 * (b + c + d)
def second_num_cond := b = 5 * c
def fourth_num_cond := d = (1 / 2) * c

theorem product_of_numbers_is_178_5 (h1 : sum_eq_36 a b c d)
  (h2 : first_num_cond a b c d) (h3 : second_num_cond b c) (h4 : fourth_num_cond d c) :
  a * b * c * d = 178.5 :=
by
  sorry

end product_of_numbers_is_178_5_l223_223134


namespace yuna_grandfather_age_l223_223372

def age_yuna : ℕ := 8
def age_father : ℕ := age_yuna + 20
def age_grandfather : ℕ := age_father + 25

theorem yuna_grandfather_age : age_grandfather = 53 := by
  sorry

end yuna_grandfather_age_l223_223372


namespace sum_a4_a5_a6_l223_223951

section ArithmeticSequence

variable {a : ℕ → ℝ}

-- Condition 1: The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

-- Condition 2: Given information
axiom a2_a8_eq_6 : a 2 + a 8 = 6

-- Question: Prove that a 4 + a 5 + a 6 = 9
theorem sum_a4_a5_a6 : is_arithmetic_sequence a → a 4 + a 5 + a 6 = 9 :=
by
  intro h_arith
  sorry

end ArithmeticSequence

end sum_a4_a5_a6_l223_223951


namespace solution_l223_223698

noncomputable def problem (x : ℝ) : Prop :=
  (Real.sqrt (Real.sqrt (53 - 3 * x)) + Real.sqrt (Real.sqrt (39 + 3 * x))) = 5

theorem solution :
  ∀ x : ℝ, problem x → x = -23 / 3 :=
by
  intro x
  intro h
  sorry

end solution_l223_223698


namespace fill_half_cistern_time_l223_223774

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l223_223774


namespace lcm_gcd_product_12_15_l223_223853

theorem lcm_gcd_product_12_15 : 
  let a := 12
  let b := 15
  lcm a b * gcd a b = 180 :=
by
  sorry

end lcm_gcd_product_12_15_l223_223853


namespace tammy_speed_proof_l223_223507

noncomputable def tammy_average_speed_second_day (v t : ℝ) :=
  v + 0.5

theorem tammy_speed_proof :
  ∃ v t : ℝ, 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    tammy_average_speed_second_day v t = 4 :=
by
  sorry

end tammy_speed_proof_l223_223507


namespace christmas_gift_distribution_l223_223990

theorem christmas_gift_distribution :
  ∃ n : ℕ, n = 30 ∧ 
  ∃ (gifts : Finset α) (students : Finset β) 
    (distribute : α → β) (a b c d : α),
    a ∈ gifts ∧ b ∈ gifts ∧ c ∈ gifts ∧ d ∈ gifts ∧ gifts.card = 4 ∧
    students.card = 3 ∧ 
    (∀ s ∈ students, ∃ g ∈ gifts, distribute g = s) ∧ 
    distribute a ≠ distribute b :=
sorry

end christmas_gift_distribution_l223_223990


namespace bus_stop_time_l223_223283

theorem bus_stop_time (v_no_stop v_with_stop : ℝ) (t_per_hour_minutes : ℝ) (h1 : v_no_stop = 48) (h2 : v_with_stop = 24) : t_per_hour_minutes = 30 := 
sorry

end bus_stop_time_l223_223283


namespace set_intersection_example_l223_223615

theorem set_intersection_example (A : Set ℕ) (B : Set ℕ) (hA : A = {1, 3, 5}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l223_223615


namespace max_a_condition_l223_223498

theorem max_a_condition (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) → a = -1 :=
by
  sorry

end max_a_condition_l223_223498


namespace pages_left_to_read_l223_223313

-- Define the given conditions
def total_pages : ℕ := 563
def pages_read : ℕ := 147

-- Define the proof statement
theorem pages_left_to_read : total_pages - pages_read = 416 :=
by
  -- The proof will be given here
  sorry

end pages_left_to_read_l223_223313


namespace at_least_two_same_books_l223_223026

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def satisfied (n : Nat) : Prop :=
  n / sum_of_digits n = 13

theorem at_least_two_same_books (n1 n2 n3 n4 : Nat) (h1 : satisfied n1) (h2 : satisfied n2) (h3 : satisfied n3) (h4 : satisfied n4) :
  n1 = n2 ∨ n1 = n3 ∨ n1 = n4 ∨ n2 = n3 ∨ n2 = n4 ∨ n3 = n4 :=
sorry

end at_least_two_same_books_l223_223026


namespace pancake_cut_l223_223110

theorem pancake_cut (n : ℕ) (h : 3 ≤ n) :
  ∃ (cut_piece : ℝ), cut_piece > 0 :=
sorry

end pancake_cut_l223_223110


namespace updated_mean_l223_223168

-- Definitions
def initial_mean := 200
def number_of_observations := 50
def decrement_per_observation := 9

-- Theorem stating the updated mean after decrementing each observation
theorem updated_mean : 
  (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 191 :=
by
  -- Placeholder for the proof
  sorry

end updated_mean_l223_223168


namespace divisible_iff_l223_223071

theorem divisible_iff (m n k : ℕ) (h : m > n) : 
  (3^(k+1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
sorry

end divisible_iff_l223_223071


namespace CDs_per_rack_l223_223745

theorem CDs_per_rack (racks_on_shelf : ℕ) (CDs_on_shelf : ℕ) (h1 : racks_on_shelf = 4) (h2 : CDs_on_shelf = 32) : 
  CDs_on_shelf / racks_on_shelf = 8 :=
by
  sorry

end CDs_per_rack_l223_223745


namespace maximum_value_of_chords_l223_223527

noncomputable def max_sum_of_chords (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : ℝ := 
  6 * Real.sqrt 10

theorem maximum_value_of_chords (P : Point) (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : 
  PA + PB + PC ≤ 6 * Real.sqrt 10 :=
by
  sorry

end maximum_value_of_chords_l223_223527


namespace simplify_and_evaluate_l223_223854

theorem simplify_and_evaluate (m : ℝ) (h_root : m^2 + 3 * m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 6 :=
by
  sorry

end simplify_and_evaluate_l223_223854


namespace inradius_of_right_triangle_l223_223249

variable (a b c : ℕ) -- Define the sides
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

noncomputable def area (a b : ℕ) : ℝ :=
  0.5 * (a : ℝ) * (b : ℝ)

noncomputable def semiperimeter (a b c : ℕ) : ℝ :=
  ((a + b + c) : ℝ) / 2

noncomputable def inradius (a b c : ℕ) : ℝ :=
  let s := semiperimeter a b c
  let A := area a b
  A / s

theorem inradius_of_right_triangle (h : right_triangle 7 24 25) : inradius 7 24 25 = 3 := by
  sorry

end inradius_of_right_triangle_l223_223249


namespace no_two_obtuse_angles_in_triangle_l223_223092

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end no_two_obtuse_angles_in_triangle_l223_223092


namespace min_value_eq_216_l223_223600

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c)

theorem min_value_eq_216 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  min_value a b c = 216 :=
sorry

end min_value_eq_216_l223_223600


namespace inequality_solution_set_l223_223867

theorem inequality_solution_set {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_inc : ∀ {x y : ℝ}, 0 < x → x < y → f x ≤ f y)
  (h_value : f 1 = 0) :
  {x | (f x - f (-x)) / x ≤ 0} = {x | -1 ≤ x ∧ x < 0} ∪ {x | 0 < x ∧ x ≤ 1} :=
by
  sorry


end inequality_solution_set_l223_223867


namespace pi_bounds_l223_223910

theorem pi_bounds :
  3 < Real.pi ∧ Real.pi < 4 :=
by
  sorry

end pi_bounds_l223_223910


namespace solve_fraction_l223_223537

variables (w x y : ℝ)

-- Conditions
def condition1 := w / x = 2 / 3
def condition2 := w / y = 6 / 15

-- Statement
theorem solve_fraction (h1 : condition1 w x) (h2 : condition2 w y) : (x + y) / y = 8 / 5 :=
sorry

end solve_fraction_l223_223537


namespace find_range_of_x_l223_223754

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end find_range_of_x_l223_223754


namespace train_speed_l223_223090

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l223_223090


namespace find_m_if_parallel_l223_223602

theorem find_m_if_parallel 
  (m : ℚ) 
  (a : ℚ × ℚ := (-2, 3)) 
  (b : ℚ × ℚ := (1, m - 3/2)) 
  (h : ∃ k : ℚ, (a.1 = k * b.1) ∧ (a.2 = k * b.2)) : 
  m = 0 := 
  sorry

end find_m_if_parallel_l223_223602


namespace pq_true_l223_223252

open Real

def p : Prop := ∃ x0 : ℝ, tan x0 = sqrt 3

def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem pq_true : p ∧ q :=
by
  sorry

end pq_true_l223_223252


namespace distinct_real_numbers_inequality_l223_223276

theorem distinct_real_numbers_inequality
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ( (2 * a - b) / (a - b) )^2 + ( (2 * b - c) / (b - c) )^2 + ( (2 * c - a) / (c - a) )^2 ≥ 5 :=
by {
    sorry
}

end distinct_real_numbers_inequality_l223_223276


namespace total_number_of_values_l223_223475

theorem total_number_of_values (S n : ℕ) (h1 : (S - 165 + 135) / n = 150) (h2 : S / n = 151) : n = 30 :=
by {
  sorry
}

end total_number_of_values_l223_223475


namespace radius_of_sphere_l223_223265

theorem radius_of_sphere (R : ℝ) (shots_count : ℕ) (shot_radius : ℝ) :
  shots_count = 125 →
  shot_radius = 1 →
  (shots_count : ℝ) * (4 / 3 * Real.pi * shot_radius^3) = 4 / 3 * Real.pi * R^3 →
  R = 5 :=
by
  intros h1 h2 h3
  sorry

end radius_of_sphere_l223_223265


namespace orthogonal_vectors_l223_223257

open Real

variables (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (a + b)^2 = (a - b)^2)

theorem orthogonal_vectors (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (h : (a + b)^2 = (a - b)^2) : a * b = 0 :=
by 
  sorry

end orthogonal_vectors_l223_223257


namespace larger_rectangle_area_l223_223742

/-- Given a smaller rectangle made out of three squares each of area 25 cm²,
    where two vertices of the smaller rectangle lie on the midpoints of the
    shorter sides of the larger rectangle and the other two vertices lie on
    the longer sides, prove the area of the larger rectangle is 150 cm². -/
theorem larger_rectangle_area (s : ℝ) (l W S_Larger W_Larger : ℝ)
  (h_s : s^2 = 25) 
  (h_small_dim : l = 3 * s ∧ W = s ∧ l * W = 3 * s^2) 
  (h_vertices : 2 * W = W_Larger ∧ l = S_Larger) :
  (S_Larger * W_Larger = 150) := 
by
  sorry

end larger_rectangle_area_l223_223742


namespace solve_for_x_l223_223821

theorem solve_for_x (x : ℝ) : 
  5 * x + 9 * x = 420 - 12 * (x - 4) -> 
  x = 18 :=
by
  intro h
  -- derivation will follow here
  sorry

end solve_for_x_l223_223821


namespace find_green_hats_l223_223786

variable (B G : ℕ)

theorem find_green_hats (h1 : B + G = 85) (h2 : 6 * B + 7 * G = 540) :
  G = 30 :=
by
  sorry

end find_green_hats_l223_223786


namespace mix_ratios_l223_223707

theorem mix_ratios (milk1 water1 milk2 water2 : ℕ) 
  (h1 : milk1 = 7) (h2 : water1 = 2)
  (h3 : milk2 = 8) (h4 : water2 = 1) :
  (milk1 + milk2) / (water1 + water2) = 5 :=
by
  -- Proof required here
  sorry

end mix_ratios_l223_223707


namespace movie_marathon_duration_l223_223680

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l223_223680


namespace elaine_rent_percentage_l223_223592

theorem elaine_rent_percentage (E : ℝ) (P : ℝ) 
  (h1 : E > 0) 
  (h2 : P > 0) 
  (h3 : 0.25 * 1.15 * E = 1.4375 * (P / 100) * E) : 
  P = 20 := 
sorry

end elaine_rent_percentage_l223_223592


namespace even_fn_solution_set_l223_223836

theorem even_fn_solution_set (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_f_def : ∀ x ≥ 0, f x = x^3 - 8) :
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by sorry

end even_fn_solution_set_l223_223836


namespace ratio_of_ages_l223_223430

theorem ratio_of_ages (joe_age_now james_age_now : ℕ) (h1 : joe_age_now = james_age_now + 10)
  (h2 : 2 * (joe_age_now + 8) = 3 * (james_age_now + 8)) : 
  (james_age_now + 8) / (joe_age_now + 8) = 2 / 3 := 
by
  sorry

end ratio_of_ages_l223_223430


namespace magnitude_diff_l223_223132

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition_1 : ‖a‖ = 2 := sorry
def condition_2 : ‖b‖ = 2 := sorry
def condition_3 : ‖a + b‖ = Real.sqrt 7 := sorry

-- Proof statement
theorem magnitude_diff (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a + b‖ = Real.sqrt 7) : 
  ‖a - b‖ = 3 :=
sorry

end magnitude_diff_l223_223132


namespace reciprocals_of_roots_l223_223330

variable (a b c k : ℝ)

theorem reciprocals_of_roots (kr ks : ℝ) (h_eq : a * kr^2 + k * c * kr + b = 0) (h_eq2 : a * ks^2 + k * c * ks + b = 0) :
  (1 / (kr^2)) + (1 / (ks^2)) = (k^2 * c^2 - 2 * a * b) / (b^2) :=
by
  sorry

end reciprocals_of_roots_l223_223330


namespace age_sum_l223_223446

theorem age_sum (my_age : ℕ) (mother_age : ℕ) (h1 : mother_age = 3 * my_age) (h2 : my_age = 10) :
  my_age + mother_age = 40 :=
by 
  -- proof omitted
  sorry

end age_sum_l223_223446


namespace positive_m_for_one_solution_l223_223273

theorem positive_m_for_one_solution :
  ∀ (m : ℝ), (∃ x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ 
  (∀ x y : ℝ, 9 * x^2 + m * x + 36 = 0 → 9 * y^2 + m * y + 36 = 0 → x = y) → m = 36 := 
by {
  sorry
}

end positive_m_for_one_solution_l223_223273


namespace mike_eggs_basket_l223_223181

theorem mike_eggs_basket : ∃ k : ℕ, (30 % k = 0) ∧ (42 % k = 0) ∧ k ≥ 4 ∧ (30 / k) ≥ 3 ∧ (42 / k) ≥ 3 ∧ k = 6 := 
by
  -- skipping the proof
  sorry

end mike_eggs_basket_l223_223181


namespace gcd_of_polynomials_l223_223622

/-- Given that a is an odd multiple of 7877, the greatest common divisor of
       7a^2 + 54a + 117 and 3a + 10 is 1. -/
theorem gcd_of_polynomials (a : ℤ) (h1 : a % 2 = 1) (h2 : 7877 ∣ a) :
  Int.gcd (7 * a ^ 2 + 54 * a + 117) (3 * a + 10) = 1 :=
sorry

end gcd_of_polynomials_l223_223622


namespace hyperbola_ratio_l223_223350

theorem hyperbola_ratio (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_foci_distance : c^2 = a^2 + b^2)
  (h_midpoint_on_hyperbola : ∀ x y, 
    (x, y) = (-(c / 2), c / 2) → ∃ (k l : ℝ), (k^2 / a^2) - (l^2 / b^2) = 1) :
  c / a = (Real.sqrt 10 + Real.sqrt 2) / 2 := 
sorry

end hyperbola_ratio_l223_223350


namespace subscription_ways_three_households_l223_223770

def num_subscription_ways (n_households : ℕ) (n_newspapers : ℕ) : ℕ :=
  if h : n_households = 3 ∧ n_newspapers = 5 then
    180
  else
    0

theorem subscription_ways_three_households :
  num_subscription_ways 3 5 = 180 :=
by
  unfold num_subscription_ways
  split_ifs
  . rfl
  . contradiction


end subscription_ways_three_households_l223_223770


namespace ratio_total_length_to_perimeter_l223_223447

noncomputable def length_initial : ℝ := 25
noncomputable def width_initial : ℝ := 15
noncomputable def extension : ℝ := 10
noncomputable def length_total : ℝ := length_initial + extension
noncomputable def perimeter_new : ℝ := 2 * (length_total + width_initial)
noncomputable def ratio : ℝ := length_total / perimeter_new

theorem ratio_total_length_to_perimeter : ratio = 35 / 100 := by
  sorry

end ratio_total_length_to_perimeter_l223_223447


namespace row_time_to_100_yards_l223_223161

theorem row_time_to_100_yards :
  let init_width_yd := 50
  let final_width_yd := 100
  let increase_width_yd_per_10m := 2
  let rowing_speed_mps := 5
  let current_speed_mps := 1
  let yard_to_meter := 0.9144
  let init_width_m := init_width_yd * yard_to_meter
  let final_width_m := final_width_yd * yard_to_meter
  let width_increase_m_per_10m := increase_width_yd_per_10m * yard_to_meter
  let total_width_increase := (final_width_m - init_width_m)
  let num_segments := total_width_increase / width_increase_m_per_10m
  let total_distance := num_segments * 10
  let effective_speed := rowing_speed_mps + current_speed_mps
  let time := total_distance / effective_speed
  time = 41.67 := by
  sorry

end row_time_to_100_yards_l223_223161


namespace first_discount_calculation_l223_223199

-- Define the given conditions and final statement
theorem first_discount_calculation (P : ℝ) (D : ℝ) :
  (1.35 * (1 - D / 100) * 0.85 = 1.03275) → (D = 10.022) :=
by
  -- Proof is not provided, to be done.
  sorry

end first_discount_calculation_l223_223199


namespace primes_square_condition_l223_223845

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end primes_square_condition_l223_223845


namespace ice_cream_ordering_ways_l223_223242

def number_of_cone_choices : ℕ := 2
def number_of_flavor_choices : ℕ := 4

theorem ice_cream_ordering_ways : number_of_cone_choices * number_of_flavor_choices = 8 := by
  sorry

end ice_cream_ordering_ways_l223_223242


namespace correct_transformation_l223_223983

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end correct_transformation_l223_223983


namespace value_of_f_neg6_l223_223661

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = -f x

theorem value_of_f_neg6 : f (-6) = 0 :=
by
  sorry

end value_of_f_neg6_l223_223661


namespace no_correct_option_l223_223746

-- Define the given table as a list of pairs
def table :=
  [(1, -2), (2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Define the given functions as potential options
def optionA (x : ℕ) : ℤ := x^2 - 5 * x + 4
def optionB (x : ℕ) : ℤ := x^2 - 3 * x
def optionC (x : ℕ) : ℤ := x^3 - 3 * x^2 + 2 * x
def optionD (x : ℕ) : ℤ := 2 * x^2 - 4 * x - 2
def optionE (x : ℕ) : ℤ := x^2 - 4 * x + 2

-- Prove that there is no correct option among the given options that matches the table
theorem no_correct_option : 
  ¬(∀ p ∈ table, p.snd = optionA p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionB p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionC p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionD p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionE p.fst) :=
by sorry

end no_correct_option_l223_223746


namespace total_volume_of_four_boxes_l223_223667

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l223_223667


namespace toothpicks_needed_base_1001_l223_223581

-- Define the number of small triangles at the base of the larger triangle
def base_triangle_count := 1001

-- Define the total number of small triangles using the sum of the first 'n' natural numbers
def total_small_triangles (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Calculate the total number of sides for all triangles if there was no sharing
def total_sides (n : ℕ) : ℕ :=
  3 * total_small_triangles n

-- Calculate the number of shared toothpicks
def shared_toothpicks (n : ℕ) : ℕ :=
  total_sides n / 2

-- Calculate the number of unshared perimeter toothpicks
def unshared_perimeter_toothpicks (n : ℕ) : ℕ :=
  3 * n

-- Calculate the total number of toothpicks required
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + unshared_perimeter_toothpicks n

-- Prove that the total toothpicks required for the base of 1001 small triangles is 755255
theorem toothpicks_needed_base_1001 : total_toothpicks base_triangle_count = 755255 :=
by {
  sorry
}

end toothpicks_needed_base_1001_l223_223581


namespace proof_problem_l223_223345

-- Define the propositions and conditions
def p : Prop := ∀ x > 0, 3^x > 1
def neg_p : Prop := ∃ x > 0, 3^x ≤ 1
def q (a : ℝ) : Prop := a < -2
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- The condition that q is a sufficient condition for f(x) to have a zero in [-1,2]
def has_zero_in_interval (a : ℝ) : Prop := 
  (-a + 3) * (2 * a + 3) ≤ 0

-- The proof problem statement
theorem proof_problem (a : ℝ) (P : p) (Q : has_zero_in_interval a) : ¬ p ∧ q a :=
by
  sorry

end proof_problem_l223_223345


namespace green_ball_probability_l223_223427

def prob_green_ball : ℚ :=
  let prob_container := (1 : ℚ) / 3
  let prob_green_I := (4 : ℚ) / 12
  let prob_green_II := (5 : ℚ) / 8
  let prob_green_III := (4 : ℚ) / 8
  prob_container * prob_green_I + prob_container * prob_green_II + prob_container * prob_green_III

theorem green_ball_probability :
  prob_green_ball = 35 / 72 :=
by
  -- Proof steps are omitted as "sorry" is used to skip the proof.
  sorry

end green_ball_probability_l223_223427


namespace sarah_probability_l223_223529

noncomputable def probability_odd_product_less_than_20 : ℚ :=
  let total_possibilities := 36
  let favorable_pairs := [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3)]
  let favorable_count := favorable_pairs.length
  let probability := favorable_count / total_possibilities
  probability

theorem sarah_probability : probability_odd_product_less_than_20 = 2 / 9 :=
by
  sorry

end sarah_probability_l223_223529


namespace range_of_a_l223_223136

variable (x a : ℝ)

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem range_of_a (h : ∀ x, q x a → p x)
  (h_not : ∃ x, ¬ q x a ∧ p x) : 1 ≤ a :=
sorry

end range_of_a_l223_223136


namespace parabola_solution_unique_l223_223383

theorem parabola_solution_unique (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 4 * a + 2 * b + c = -1) (h3 : 4 * a + b = 1) :
  a = 3 ∧ b = -11 ∧ c = 9 := 
  by sorry

end parabola_solution_unique_l223_223383


namespace carly_butterfly_days_l223_223457

-- Define the conditions
variable (x : ℕ) -- number of days Carly practices her butterfly stroke
def butterfly_hours_per_day := 3  -- hours per day for butterfly stroke
def backstroke_hours_per_day := 2  -- hours per day for backstroke stroke
def backstroke_days_per_week := 6  -- days per week for backstroke stroke
def total_hours_per_month := 96  -- total hours practicing swimming in a month
def weeks_in_month := 4  -- number of weeks in a month

-- The proof problem
theorem carly_butterfly_days :
  (butterfly_hours_per_day * x + backstroke_hours_per_day * backstroke_days_per_week) * weeks_in_month = total_hours_per_month
  → x = 4 := 
by
  sorry

end carly_butterfly_days_l223_223457


namespace find_k_l223_223936

theorem find_k (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 ↔ |k * x - 4| ≤ 2) → k = 2 :=
by
  sorry

end find_k_l223_223936


namespace relationship_y1_y2_l223_223555

theorem relationship_y1_y2
  (x1 y1 x2 y2 : ℝ)
  (hA : y1 = 3 * x1 + 4)
  (hB : y2 = 3 * x2 + 4)
  (h : x1 < x2) :
  y1 < y2 :=
sorry

end relationship_y1_y2_l223_223555


namespace perfect_square_proof_l223_223105

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem perfect_square_proof :
  isPerfectSquare (factorial 22 * factorial 23 * factorial 24 / 12) :=
sorry

end perfect_square_proof_l223_223105


namespace one_hundred_fiftieth_digit_l223_223528

theorem one_hundred_fiftieth_digit : (0.135 : Real) * 10^150 = 1 :=
by
  sorry

end one_hundred_fiftieth_digit_l223_223528


namespace price_of_tea_mixture_l223_223963

noncomputable def price_of_mixture (price1 price2 price3 : ℝ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3)

theorem price_of_tea_mixture :
  price_of_mixture 126 135 175.5 1 1 2 = 153 := 
by
  sorry

end price_of_tea_mixture_l223_223963


namespace problem1_problem2_l223_223135

-- Problem 1: Remainder of 2011-digit number with each digit 2 when divided by 9 is 8

theorem problem1 : (4022 % 9 = 8) := by
  sorry

-- Problem 2: Remainder of n-digit number with each digit 7 when divided by 9 and n % 9 = 3 is 3

theorem problem2 (n : ℕ) (h : n % 9 = 3) : ((7 * n) % 9 = 3) := by
  sorry

end problem1_problem2_l223_223135


namespace johns_weekly_allowance_l223_223266

theorem johns_weekly_allowance (A : ℝ) 
  (arcade_spent : A * (3/5) = 3 * (A/5)) 
  (remainder_after_arcade : (2/5) * A = A - 3 * (A/5))
  (toy_store_spent : (1/3) * (2/5) * A = 2 * (A/15)) 
  (remainder_after_toy_store : (2/5) * A - (2/15) * A = 4 * (A/15))
  (last_spent : (4/15) * A = 0.4) :
  A = 1.5 :=
sorry

end johns_weekly_allowance_l223_223266


namespace lines_condition_l223_223187

-- Assume x and y are real numbers representing coordinates on the lines l1 and l2
variables (x y : ℝ)

-- Points on the lines l1 and l2 satisfy the condition |x| - |y| = 0.
theorem lines_condition (x y : ℝ) (h : abs x = abs y) : abs x - abs y = 0 :=
by
  sorry

end lines_condition_l223_223187


namespace bad_carrots_count_l223_223154

def total_carrots (vanessa_carrots : ℕ) (mother_carrots : ℕ) : ℕ := 
vanessa_carrots + mother_carrots

def bad_carrots (total_carrots : ℕ) (good_carrots : ℕ) : ℕ := 
total_carrots - good_carrots

theorem bad_carrots_count : 
  ∀ (vanessa_carrots mother_carrots good_carrots : ℕ), 
  vanessa_carrots = 17 → 
  mother_carrots = 14 → 
  good_carrots = 24 → 
  bad_carrots (total_carrots vanessa_carrots mother_carrots) good_carrots = 7 := 
by 
  intros; 
  sorry

end bad_carrots_count_l223_223154


namespace math_problem_l223_223093

noncomputable def problem : Real :=
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5)

theorem math_problem :
  problem = 5 - 4 * Real.sqrt 2 :=
by
  sorry

end math_problem_l223_223093


namespace speed_in_still_water_l223_223908

namespace SwimmingProblem

variable (V_m V_s : ℝ)

-- Downstream condition
def downstream_condition : Prop := V_m + V_s = 18

-- Upstream condition
def upstream_condition : Prop := V_m - V_s = 13

-- The main theorem stating the problem
theorem speed_in_still_water (h_downstream : downstream_condition V_m V_s) 
                             (h_upstream : upstream_condition V_m V_s) :
    V_m = 15.5 :=
by
  sorry

end SwimmingProblem

end speed_in_still_water_l223_223908


namespace power_equivalence_l223_223220

theorem power_equivalence (L : ℕ) : 32^4 * 4^5 = 2^L → L = 30 :=
by
  sorry

end power_equivalence_l223_223220


namespace physicist_imons_no_entanglement_l223_223137

theorem physicist_imons_no_entanglement (G : SimpleGraph V) :
  (∃ ops : ℕ, ∀ v₁ v₂ : V, ¬G.Adj v₁ v₂) :=
by
  sorry

end physicist_imons_no_entanglement_l223_223137


namespace system_of_equations_solution_system_of_inequalities_no_solution_l223_223471

-- Problem 1: Solving system of linear equations
theorem system_of_equations_solution :
  ∃ x y : ℝ, x - 3*y = -5 ∧ 2*x + 2*y = 6 ∧ x = 1 ∧ y = 2 := by
  sorry

-- Problem 2: Solving the system of inequalities
theorem system_of_inequalities_no_solution :
  ¬ (∃ x : ℝ, 2*x < -4 ∧ (1/2)*x - 5 > 1 - (3/2)*x) := by
  sorry

end system_of_equations_solution_system_of_inequalities_no_solution_l223_223471


namespace chord_length_l223_223900

theorem chord_length
  (l_eq : ∀ (rho theta : ℝ), rho * (Real.sin theta - Real.cos theta) = 1)
  (gamma_eq : ∀ (rho : ℝ) (theta : ℝ), rho = 1) :
  ∃ AB : ℝ, AB = Real.sqrt 2 :=
by
  sorry

end chord_length_l223_223900


namespace ratio_celeste_bianca_l223_223785

-- Definitions based on given conditions
def bianca_hours : ℝ := 12.5
def celest_hours (x : ℝ) : ℝ := 12.5 * x
def mcclain_hours (x : ℝ) : ℝ := 12.5 * x - 8.5

-- The total time worked in hours
def total_hours : ℝ := 54

-- The ratio to prove
def celeste_bianca_ratio : ℝ := 2

-- The proof statement
theorem ratio_celeste_bianca (x : ℝ) (hx :  12.5 + 12.5 * x + (12.5 * x - 8.5) = total_hours) :
  celest_hours 2 / bianca_hours = celeste_bianca_ratio :=
by
  sorry

end ratio_celeste_bianca_l223_223785


namespace minute_hand_only_rotates_l223_223100

-- Define what constitutes translation and rotation
def is_translation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p1 p2 : ℝ), motion p1 p2 → (∃ d : ℝ, ∀ t : ℝ, motion (p1 + t) (p2 + t) ∧ |p1 - p2| = d)

def is_rotation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p : ℝ), ∃ c : ℝ, ∃ r : ℝ, (∀ (t : ℝ), |p - c| = r)

-- Define the condition that the minute hand of a clock undergoes a specific motion
def minute_hand_motion (p : ℝ) (t : ℝ) : Prop :=
  -- The exact definition here would involve trigonometric representation
  sorry

-- The main proof statement
theorem minute_hand_only_rotates :
  is_rotation minute_hand_motion ∧ ¬ is_translation minute_hand_motion :=
sorry

end minute_hand_only_rotates_l223_223100


namespace triangle_ineq_l223_223948

theorem triangle_ineq (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) < 5/2 := 
by
  sorry

end triangle_ineq_l223_223948


namespace eric_less_than_ben_l223_223433

variables (E B J : ℕ)

theorem eric_less_than_ben
  (hJ : J = 26)
  (hB : B = J - 9)
  (total_money : E + B + J = 50) :
  B - E = 10 :=
sorry

end eric_less_than_ben_l223_223433


namespace quadratic_equation_real_roots_k_value_l223_223386

theorem quadratic_equation_real_roots_k_value :
  (∀ k : ℕ, (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) <-> k = 1) :=
by
  sorry
  
end quadratic_equation_real_roots_k_value_l223_223386


namespace average_of_rest_of_class_l223_223884

def class_average (n : ℕ) (avg : ℕ) := n * avg
def sub_class_average (n : ℕ) (sub_avg : ℕ) := (n / 4) * sub_avg

theorem average_of_rest_of_class (n : ℕ) (h1 : class_average n 80 = 80 * n) (h2 : sub_class_average n 92 = (n / 4) * 92) :
  let A := 76
  A * (3 * n / 4) + (n / 4) * 92 = 80 * n := by
  sorry

end average_of_rest_of_class_l223_223884


namespace find_first_number_l223_223067

theorem find_first_number (HCF LCM number2 number1 : ℕ) 
    (hcf_condition : HCF = 12) 
    (lcm_condition : LCM = 396) 
    (number2_condition : number2 = 198) 
    (number1_condition : number1 * number2 = HCF * LCM) : 
    number1 = 24 := 
by 
    sorry

end find_first_number_l223_223067


namespace exists_three_digit_numbers_with_property_l223_223185

open Nat

def is_three_digit_number (n : ℕ) : Prop := (100 ≤ n ∧ n < 1000)

def distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def inserts_zeros_and_is_square (n : ℕ) (k : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  let transformed_number := a * 10^(2*k + 2) + b * 10^(k + 1) + c
  ∃ x : ℕ, transformed_number = x * x

theorem exists_three_digit_numbers_with_property:
  ∃ n1 n2 : ℕ, 
    is_three_digit_number n1 ∧ 
    is_three_digit_number n2 ∧ 
    distinct_digits n1 ∧ 
    distinct_digits n2 ∧ 
    ( ∀ k, inserts_zeros_and_is_square n1 k ) ∧ 
    ( ∀ k, inserts_zeros_and_is_square n2 k ) ∧ 
    n1 ≠ n2 := 
sorry

end exists_three_digit_numbers_with_property_l223_223185


namespace find_value_of_a_l223_223211

theorem find_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≤ 24) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 24) ∧
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≥ 3) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 3) → 
  a = 2 ∨ a = -5 :=
by
  sorry

end find_value_of_a_l223_223211


namespace tangent_line_at_e_range_of_a_l223_223230

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x - 2 * a * x

theorem tangent_line_at_e (a : ℝ) :
  a = 0 →
  ∃ m b : ℝ, (∀ x, y = m * x + b) ∧ 
             y = (2 / Real.exp 1 - 2 * Real.exp 1) * x + (Real.exp 1)^2 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Set.Ioi 1 → g a x < 0) →
  a ∈ Set.Icc (-1) 1 :=
sorry

end tangent_line_at_e_range_of_a_l223_223230


namespace triangle_properties_l223_223765

-- Define the given sides of the triangle
def a := 6
def b := 8
def c := 10

-- Define necessary parameters and properties
def isRightTriangle (a b c : Nat) : Prop := a^2 + b^2 = c^2
def area (a b : Nat) : Nat := (a * b) / 2
def semiperimeter (a b c : Nat) : Nat := (a + b + c) / 2
def inradius (A s : Nat) : Nat := A / s
def circumradius (c : Nat) : Nat := c / 2

-- The theorem statement
theorem triangle_properties :
  isRightTriangle a b c ∧
  area a b = 24 ∧
  semiperimeter a b c = 12 ∧
  inradius (area a b) (semiperimeter a b c) = 2 ∧
  circumradius c = 5 :=
by
  sorry

end triangle_properties_l223_223765


namespace probability_neither_prime_nor_composite_lemma_l223_223166

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬ is_prime n ∧ ¬ is_composite n

def probability_of_neither_prime_nor_composite (n : ℕ) : ℚ :=
  if 1 ≤ n ∧ n ≤ 97 then 1 / 97 else 0

theorem probability_neither_prime_nor_composite_lemma :
  probability_of_neither_prime_nor_composite 1 = 1 / 97 := by
  sorry

end probability_neither_prime_nor_composite_lemma_l223_223166


namespace my_problem_l223_223097

-- Definitions and conditions from the problem statement
variables (p q r u v w : ℝ)

-- Conditions
axiom h1 : 17 * u + q * v + r * w = 0
axiom h2 : p * u + 29 * v + r * w = 0
axiom h3 : p * u + q * v + 56 * w = 0
axiom h4 : p ≠ 17
axiom h5 : u ≠ 0

-- Problem statement to prove
theorem my_problem : (p / (p - 17)) + (q / (q - 29)) + (r / (r - 56)) = 0 :=
sorry

end my_problem_l223_223097


namespace rental_plans_count_l223_223905

-- Define the number of large buses, medium buses, and the total number of people.
def num_large_buses := 42
def num_medium_buses := 25
def total_people := 1511

-- State the theorem to prove that there are exactly 2 valid rental plans.
theorem rental_plans_count (x y : ℕ) :
  (num_large_buses * x + num_medium_buses * y = total_people) →
  (∃! (x y : ℕ), num_large_buses * x + num_medium_buses * y = total_people) :=
by
  sorry

end rental_plans_count_l223_223905


namespace probability_of_diamond_king_ace_l223_223001

noncomputable def probability_three_cards : ℚ :=
  (11 / 52) * (4 / 51) * (4 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (4 / 51) * (3 / 50)

theorem probability_of_diamond_king_ace :
  probability_three_cards = 284 / 132600 := 
by
  sorry

end probability_of_diamond_king_ace_l223_223001


namespace average_value_of_x_l223_223307

theorem average_value_of_x
  (x : ℝ)
  (h : (5 + 5 + x + 6 + 8) / 5 = 6) :
  x = 6 :=
sorry

end average_value_of_x_l223_223307


namespace train_length_l223_223112

theorem train_length
  (train_speed_kmph : ℝ)
  (person_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (h_train_speed : train_speed_kmph = 80)
  (h_person_speed : person_speed_kmph = 16)
  (h_time : time_seconds = 15)
  : (train_speed_kmph - person_speed_kmph) * (5/18) * time_seconds = 266.67 := 
by
  rw [h_train_speed, h_person_speed, h_time]
  norm_num
  sorry

end train_length_l223_223112


namespace third_box_number_l223_223895

def N : ℕ := 301

theorem third_box_number (N : ℕ) (h1 : N % 3 = 1) (h2 : N % 4 = 1) (h3 : N % 7 = 0) :
  ∃ x : ℕ, x > 4 ∧ x ≠ 7 ∧ N % x = 1 ∧ (∀ y > 4, y ≠ 7 → y < x → N % y ≠ 1) ∧ x = 6 :=
by
  sorry

end third_box_number_l223_223895


namespace quoted_price_of_shares_l223_223763

theorem quoted_price_of_shares :
  ∀ (investment nominal_value dividend_rate annual_income quoted_price : ℝ),
  investment = 4940 →
  nominal_value = 10 →
  dividend_rate = 14 →
  annual_income = 728 →
  quoted_price = 9.5 :=
by
  intros investment nominal_value dividend_rate annual_income quoted_price
  intros h_investment h_nominal_value h_dividend_rate h_annual_income
  sorry

end quoted_price_of_shares_l223_223763


namespace false_log_exists_x_l223_223361

theorem false_log_exists_x {x : ℝ} : ¬ ∃ x : ℝ, Real.log x = 0 :=
by sorry

end false_log_exists_x_l223_223361


namespace Matias_sales_l223_223554

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end Matias_sales_l223_223554


namespace infinite_geometric_series_sum_l223_223508

theorem infinite_geometric_series_sum :
  ∑' (n : ℕ), (1 : ℚ) * (-1 / 4 : ℚ) ^ n = 4 / 5 :=
by
  sorry

end infinite_geometric_series_sum_l223_223508


namespace first_place_prize_is_200_l223_223045

-- Define the conditions from the problem
def total_prize_money : ℤ := 800
def num_winners : ℤ := 18
def second_place_prize : ℤ := 150
def third_place_prize : ℤ := 120
def fourth_to_eighteenth_prize : ℤ := 22
def fourth_to_eighteenth_winners : ℤ := num_winners - 3

-- Define the amount awarded to fourth to eighteenth place winners
def total_fourth_to_eighteenth_prize : ℤ := fourth_to_eighteenth_winners * fourth_to_eighteenth_prize

-- Define the total amount awarded to second and third place winners
def total_second_and_third_prize : ℤ := second_place_prize + third_place_prize

-- Define the total amount awarded to second to eighteenth place winners
def total_second_to_eighteenth_prize : ℤ := total_fourth_to_eighteenth_prize + total_second_and_third_prize

-- Define the amount awarded to first place
def first_place_prize : ℤ := total_prize_money - total_second_to_eighteenth_prize

-- Statement for proof required
theorem first_place_prize_is_200 : first_place_prize = 200 :=
by
  -- Assuming the conditions are correct
  sorry

end first_place_prize_is_200_l223_223045


namespace truck_capacity_rental_plan_l223_223188

-- Define the variables for the number of boxes each type of truck can carry
variables {x y : ℕ}

-- Define the conditions for the number of boxes carried by trucks
axiom cond1 : 15 * x + 25 * y = 750
axiom cond2 : 10 * x + 30 * y = 700

-- Problem 1: Prove x = 25 and y = 15
theorem truck_capacity : x = 25 ∧ y = 15 :=
by
  sorry

-- Define the variables for the number of each type of truck
variables {m : ℕ}

-- Define the conditions for the total number of trucks and boxes to be carried
axiom cond3 : 25 * m + 15 * (70 - m) ≤ 1245
axiom cond4 : 70 - m ≤ 3 * m

-- Problem 2: Prove there is one valid rental plan with m = 18 and 70-m = 52
theorem rental_plan : 17 ≤ m ∧ m ≤ 19 ∧ 70 - m ≤ 3 * m ∧ (70-m = 52 → m = 18) :=
by
  sorry

end truck_capacity_rental_plan_l223_223188


namespace factorize_expression_l223_223831

variable {R : Type} [CommRing R] (m a : R)

theorem factorize_expression : m * a^2 - m = m * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expression_l223_223831


namespace cube_difference_l223_223970

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 :=
sorry

end cube_difference_l223_223970


namespace total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l223_223502

def chocolate_sold : ℕ := 6 + 7 + 4 + 8 + 9 + 10 + 5
def vanilla_sold : ℕ := 4 + 5 + 3 + 7 + 6 + 8 + 4
def strawberry_sold : ℕ := 3 + 2 + 6 + 4 + 5 + 7 + 4

theorem total_chocolate_sold : chocolate_sold = 49 :=
by
  unfold chocolate_sold
  rfl

theorem total_vanilla_sold : vanilla_sold = 37 :=
by
  unfold vanilla_sold
  rfl

theorem total_strawberry_sold : strawberry_sold = 31 :=
by
  unfold strawberry_sold
  rfl

end total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l223_223502


namespace find_f2_l223_223624

theorem find_f2 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 2 * x ^ 2) :
  f 2 = -1 / 4 :=
by
  sorry

end find_f2_l223_223624


namespace abs_div_one_add_i_by_i_l223_223159

noncomputable def imaginary_unit : ℂ := Complex.I

/-- The absolute value of the complex number (1 + i)/i is √2. -/
theorem abs_div_one_add_i_by_i : Complex.abs ((1 + imaginary_unit) / imaginary_unit) = Real.sqrt 2 := by
  sorry

end abs_div_one_add_i_by_i_l223_223159


namespace fraction_product_l223_223972

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l223_223972


namespace parallel_vectors_l223_223681

def a : (ℝ × ℝ) := (1, -2)
def b (x : ℝ) : (ℝ × ℝ) := (-2, x)

theorem parallel_vectors (x : ℝ) (h : 1 / -2 = -2 / x) : x = 4 := by
  sorry

end parallel_vectors_l223_223681


namespace jason_total_spent_l223_223708

-- Conditions
def shorts_cost : ℝ := 14.28
def jacket_cost : ℝ := 4.74

-- Statement to prove
theorem jason_total_spent : shorts_cost + jacket_cost = 19.02 := by
  -- Proof to be filled in
  sorry

end jason_total_spent_l223_223708


namespace negation_of_at_least_three_is_at_most_two_l223_223630

theorem negation_of_at_least_three_is_at_most_two :
  (¬ (∀ n : ℕ, n ≥ 3)) ↔ (∃ n : ℕ, n ≤ 2) :=
sorry

end negation_of_at_least_three_is_at_most_two_l223_223630


namespace geometric_progression_condition_l223_223758

theorem geometric_progression_condition (a b c d : ℝ) :
  (∃ r : ℝ, (b = a * r ∨ b = a * -r) ∧
             (c = a * r^2 ∨ c = a * (-r)^2) ∧
             (d = a * r^3 ∨ d = a * (-r)^3) ∧
             (a = b / r ∨ a = b / -r) ∧
             (b = c / r ∨ b = c / -r) ∧
             (c = d / r ∨ c = d / -r) ∧
             (d = a / r ∨ d = a / -r)) ↔
  (a = b ∨ a = -b) ∧ (a = c ∨ a = -c) ∧ (a = d ∨ a = -d) := sorry

end geometric_progression_condition_l223_223758


namespace tan_sum_identity_l223_223418

theorem tan_sum_identity (α β : ℝ)
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l223_223418


namespace selling_price_is_correct_l223_223822

def profit_percent : ℝ := 0.6
def cost_price : ℝ := 375
def profit : ℝ := profit_percent * cost_price
def selling_price : ℝ := cost_price + profit

theorem selling_price_is_correct : selling_price = 600 :=
by
  -- proof steps would go here
  sorry

end selling_price_is_correct_l223_223822


namespace sym_axis_of_curve_eq_zero_b_plus_d_l223_223927

theorem sym_axis_of_curve_eq_zero_b_plus_d
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h_symm : ∀ x : ℝ, 2 * x = (a * ((a * x + b) / (c * x + d)) + b) / (c * ((a * x + b) / (c * x + d)) + d)) :
  b + d = 0 :=
sorry

end sym_axis_of_curve_eq_zero_b_plus_d_l223_223927


namespace student_number_choice_l223_223723

theorem student_number_choice (x : ℤ) (h : 3 * x - 220 = 110) : x = 110 :=
by sorry

end student_number_choice_l223_223723


namespace find_x_l223_223715

noncomputable def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

noncomputable def vec_dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1)^2 + (v.2)^2)

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (1, x)) 
  (h3 : magnitude (vec_sub a b) = vec_dot a b) : 
  x = 1 / 3 :=
by
  sorry

end find_x_l223_223715


namespace children_on_ferris_wheel_l223_223651

theorem children_on_ferris_wheel (x : ℕ) (h : 5 * x + 3 * 5 + 8 * 2 * 5 = 110) : x = 3 :=
sorry

end children_on_ferris_wheel_l223_223651


namespace cakes_difference_l223_223348

theorem cakes_difference (cakes_bought cakes_sold : ℕ) (h1 : cakes_bought = 139) (h2 : cakes_sold = 145) : cakes_sold - cakes_bought = 6 :=
by
  sorry

end cakes_difference_l223_223348


namespace desired_markup_percentage_l223_223595

theorem desired_markup_percentage
  (initial_price : ℝ) (markup_rate : ℝ) (wholesale_price : ℝ) (additional_increase : ℝ) 
  (h1 : initial_price = wholesale_price * (1 + markup_rate)) 
  (h2 : initial_price = 34) 
  (h3 : markup_rate = 0.70) 
  (h4 : additional_increase = 6) 
  : ( (initial_price + additional_increase - wholesale_price) / wholesale_price * 100 ) = 100 := 
by
  sorry

end desired_markup_percentage_l223_223595


namespace salary_increase_l223_223000

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 0.65 * S = 0.5 * S + (P / 100) * (0.5 * S)) : P = 30 := 
by
  -- proof goes here
  sorry

end salary_increase_l223_223000


namespace tray_contains_correct_number_of_pieces_l223_223777

-- Define the dimensions of the tray
def tray_width : ℕ := 24
def tray_length : ℕ := 20
def tray_area : ℕ := tray_width * tray_length

-- Define the dimensions of each brownie piece
def piece_width : ℕ := 3
def piece_length : ℕ := 4
def piece_area : ℕ := piece_width * piece_length

-- Define the goal: the number of pieces of brownies that the tray contains
def num_pieces : ℕ := tray_area / piece_area

-- The statement to prove
theorem tray_contains_correct_number_of_pieces :
  num_pieces = 40 :=
by
  sorry

end tray_contains_correct_number_of_pieces_l223_223777


namespace binomial_expansion_coefficients_equal_l223_223460

theorem binomial_expansion_coefficients_equal (n : ℕ) (h : n ≥ 6)
  (h_eq : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : n = 7 := by
  sorry

end binomial_expansion_coefficients_equal_l223_223460


namespace find_tan_beta_l223_223519

variable (α β : ℝ)

def condition1 : Prop := Real.tan α = 3
def condition2 : Prop := Real.tan (α + β) = 2

theorem find_tan_beta (h1 : condition1 α) (h2 : condition2 α β) : Real.tan β = -1 / 7 := 
by {
  sorry
}

end find_tan_beta_l223_223519


namespace simplification_evaluation_l223_223631

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  ( (2 * x - 6) / (x - 2) ) / ( (5 / (x - 2)) - (x + 2) ) = Real.sqrt 2 - 2 :=
sorry

end simplification_evaluation_l223_223631


namespace find_n_value_l223_223638

theorem find_n_value (m n k : ℝ) (h1 : n = k / m) (h2 : m = k / 2) (h3 : k ≠ 0): n = 2 :=
sorry

end find_n_value_l223_223638


namespace shadow_boundary_function_correct_l223_223761

noncomputable def sphereShadowFunction : ℝ → ℝ :=
  λ x => (x + 1) / 2

theorem shadow_boundary_function_correct :
  ∀ (x y : ℝ), 
    -- Conditions: 
    -- The sphere with center (0,0,2) and radius 2
    -- A light source at point P = (1, -2, 3)
    -- The shadow must lie on the xy-plane, so z-coordinate is 0
    (sphereShadowFunction x = y) ↔ (- x + 2 * y - 1 = 0) :=
by
  intros x y
  sorry

end shadow_boundary_function_correct_l223_223761


namespace geometric_sequence_a8_value_l223_223015

variable {a : ℕ → ℕ}

-- Assuming a is a geometric sequence, provide the condition a_3 * a_9 = 4 * a_4
def geometric_sequence_condition (a : ℕ → ℕ) :=
  (a 3) * (a 9) = 4 * (a 4)

-- Prove that a_8 = 4 under the given condition
theorem geometric_sequence_a8_value (a : ℕ → ℕ) (h : geometric_sequence_condition a) : a 8 = 4 :=
  sorry

end geometric_sequence_a8_value_l223_223015


namespace monthly_energy_consumption_l223_223077

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end monthly_energy_consumption_l223_223077


namespace k_range_l223_223750

noncomputable def valid_k (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → x / Real.exp x < 1 / (k + 2 * x - x^2)

theorem k_range : {k : ℝ | valid_k k} = {k : ℝ | 0 ≤ k ∧ k < Real.exp 1 - 1} :=
by sorry

end k_range_l223_223750


namespace limit_sum_infinite_geometric_series_l223_223407

noncomputable def infinite_geometric_series_limit (a_1 q : ℝ) :=
  if |q| < 1 then (a_1 / (1 - q)) else 0

theorem limit_sum_infinite_geometric_series :
  infinite_geometric_series_limit 1 (1 / 3) = 3 / 2 :=
by
  sorry

end limit_sum_infinite_geometric_series_l223_223407


namespace proportion_solution_l223_223421

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 4.5 / (7 / 3)) : x = 0.3888888889 :=
by
  sorry

end proportion_solution_l223_223421


namespace max_garden_area_l223_223583

-- Definitions of conditions
def shorter_side (s : ℕ) := s
def longer_side (s : ℕ) := 2 * s
def total_perimeter (s : ℕ) := 2 * shorter_side s + 2 * longer_side s 
def garden_area (s : ℕ) := shorter_side s * longer_side s

-- Theorem with given conditions and conclusion to be proven
theorem max_garden_area (s : ℕ) (h_perimeter : total_perimeter s = 480) : garden_area s = 12800 :=
by
  sorry

end max_garden_area_l223_223583


namespace monotonicity_of_f_solve_inequality_l223_223303

noncomputable def f (x : ℝ) : ℝ := sorry

def f_defined : ∀ x > 0, ∃ y, f y = f x := sorry

axiom functional_eq : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y 

axiom f_gt_zero : ∀ x, x > 1 → f x > 0

theorem monotonicity_of_f : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality (x : ℝ) (h1 : f 2 = 1) (h2 : 0 < x) : 
  f x + f (x - 3) ≤ 2 ↔ 3 < x ∧ x ≤ 4 :=
sorry

end monotonicity_of_f_solve_inequality_l223_223303


namespace problem_solution_l223_223367

theorem problem_solution (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ ab < b^2 :=
sorry

end problem_solution_l223_223367


namespace arithmetic_expression_equality_l223_223079

theorem arithmetic_expression_equality :
  ( ( (4 + 6 + 5) * 2 ) / 4 - ( (3 * 2) / 4 ) ) = 6 :=
by sorry

end arithmetic_expression_equality_l223_223079


namespace sin_double_angle_l223_223843

theorem sin_double_angle 
  (α β : ℝ)
  (h1 : 0 < β)
  (h2 : β < α)
  (h3 : α < π / 4)
  (h_cos_diff : Real.cos (α - β) = 12 / 13)
  (h_sin_sum : Real.sin (α + β) = 4 / 5) :
  Real.sin (2 * α) = 63 / 65 := 
sorry

end sin_double_angle_l223_223843


namespace expansion_coeff_sum_l223_223788

theorem expansion_coeff_sum :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ, 
    (2*x - 1)^10 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10)
  → (1 - 20 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 1 → a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 20) :=
by
  sorry

end expansion_coeff_sum_l223_223788


namespace ratio_value_l223_223869

theorem ratio_value (c d : ℝ) (h1 : c = 15 - 4 * d) (h2 : c / d = 4) : d = 15 / 8 :=
by sorry

end ratio_value_l223_223869


namespace find_a_of_ellipse_foci_l223_223013

theorem find_a_of_ellipse_foci (a : ℝ) :
  (∀ x y : ℝ, a^2 * x^2 - (a / 2) * y^2 = 1) →
  (a^2 - (2 / a) = 4) →
  a = (1 - Real.sqrt 5) / 4 :=
by 
  intros h1 h2
  sorry

end find_a_of_ellipse_foci_l223_223013


namespace inequality_proof_l223_223903

variables {x y : ℝ}

theorem inequality_proof (hx_pos : x > 0) (hy_pos : y > 0) (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := 
by 
  sorry

end inequality_proof_l223_223903


namespace x_n_squared_leq_2007_l223_223546

def recurrence (x y : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ y 0 = 2007 ∧
  ∀ n, x (n + 1) = x n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (y n + y (n + 1)) ∧
       y (n + 1) = y n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (x n + x (n + 1))

theorem x_n_squared_leq_2007 (x y : ℕ → ℝ) (h : recurrence x y) : ∀ n, x n ^ 2 ≤ 2007 :=
by sorry

end x_n_squared_leq_2007_l223_223546


namespace mike_spent_on_speakers_l223_223468

-- Definitions of the conditions:
def total_car_parts_cost : ℝ := 224.87
def new_tires_cost : ℝ := 106.33

-- Statement of the proof problem:
theorem mike_spent_on_speakers : total_car_parts_cost - new_tires_cost = 118.54 :=
by
  sorry

end mike_spent_on_speakers_l223_223468


namespace polynomial_divisible_by_cube_l223_223712

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := 
  n^2 * x^(n+2) - (2 * n^2 + 2 * n - 1) * x^(n+1) + (n + 1)^2 * x^n - x - 1

theorem polynomial_divisible_by_cube (n : ℕ) (h : n > 0) : 
  ∃ Q, P n x = (x - 1)^3 * Q :=
sorry

end polynomial_divisible_by_cube_l223_223712


namespace number_of_terriers_groomed_l223_223898

-- Define the initial constants and the conditions from the problem statement
def time_to_groom_poodle := 30
def time_to_groom_terrier := 15
def number_of_poodles := 3
def total_grooming_time := 210

-- Define the problem to prove that the number of terriers groomed is 8
theorem number_of_terriers_groomed (groom_time_poodle groom_time_terrier num_poodles total_time : ℕ) : 
  groom_time_poodle = time_to_groom_poodle → 
  groom_time_terrier = time_to_groom_terrier →
  num_poodles = number_of_poodles →
  total_time = total_grooming_time →
  ∃ n : ℕ, n * groom_time_terrier + num_poodles * groom_time_poodle = total_time ∧ n = 8 := 
by
  intros h1 h2 h3 h4
  sorry

end number_of_terriers_groomed_l223_223898


namespace total_sides_tom_tim_l223_223192

def sides_per_die : Nat := 6

def tom_dice_count : Nat := 4
def tim_dice_count : Nat := 4

theorem total_sides_tom_tim : tom_dice_count * sides_per_die + tim_dice_count * sides_per_die = 48 := by
  sorry

end total_sides_tom_tim_l223_223192


namespace Seokgi_candies_l223_223189

theorem Seokgi_candies (C : ℕ) 
  (h1 : C / 2 + (C - C / 2) / 3 + 12 = C)
  (h2 : ∃ x, x = 12) :
  C = 36 := 
by 
  sorry

end Seokgi_candies_l223_223189


namespace ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l223_223582

-- Part A:
-- Define intersections of a given ellipse and parabola under conditions on m and n
theorem ellipse_parabola_four_intersections (m n : ℝ) :
  (3 / n < m) ∧ (m < (4 * m^2 + 9) / (4 * m)) ∧ (m > 3 / 2) →
  ∃ x y : ℝ, (x^2 / n + y^2 / 9 = 1) ∧ (y = x^2 - m) :=
sorry

-- Part B:
-- Prove four intersection points of given ellipse and parabola lie on same circle for m = n = 4
theorem intersection_points_lie_on_circle (x y : ℝ) :
  (4 / 4 + y^2 / 9 = 1) ∧ (y = x^2 - 4) →
  ∃ k l r : ℝ, ∀ x' y', ((x' - k)^2 + (y' - l)^2 = r^2) :=
sorry

end ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l223_223582


namespace rectangle_area_l223_223385

variables (y : ℝ) (length : ℝ) (width : ℝ)

-- Definitions based on conditions
def is_diagonal_y (length width y : ℝ) : Prop :=
  y^2 = length^2 + width^2

def is_length_three_times_width (length width : ℝ) : Prop :=
  length = 3 * width

-- Statement to prove
theorem rectangle_area (y : ℝ) (length width : ℝ)
  (h1 : is_diagonal_y length width y)
  (h2 : is_length_three_times_width length width) :
  length * width = 3 * (y^2 / 10) :=
sorry

end rectangle_area_l223_223385


namespace eight_child_cotton_l223_223094

theorem eight_child_cotton {a_1 a_8 d S_8 : ℕ} 
  (h1 : d = 17)
  (h2 : S_8 = 996)
  (h3 : 8 * a_1 + 28 * d = S_8) :
  a_8 = a_1 + 7 * d → a_8 = 184 := by
  intro h4
  subst_vars
  sorry

end eight_child_cotton_l223_223094


namespace hexagon_ratio_l223_223952

theorem hexagon_ratio 
  (hex_area : ℝ)
  (rs_bisects_area : ∃ (a b : ℝ), a + b = hex_area / 2 ∧ ∃ (x r s : ℝ), x = 4 ∧ r * s = (hex_area / 2 - 1))
  : ∀ (XR RS : ℝ), XR = RS → XR / RS = 1 :=
by
  sorry

end hexagon_ratio_l223_223952


namespace mens_wages_l223_223184

variable (M : ℕ) (wages_of_men : ℕ)

-- Conditions based on the problem
axiom eq1 : 15 * M = 90
axiom def_wages_of_men : wages_of_men = 5 * M

-- Prove that the total wages of the men are Rs. 30
theorem mens_wages : wages_of_men = 30 :=
by
  -- The proof would go here
  sorry

end mens_wages_l223_223184


namespace maximum_initial_jars_l223_223834

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l223_223834


namespace obtuse_triangle_range_a_l223_223861

noncomputable def is_obtuse_triangle (a b c : ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 90 ∧ θ ≤ 120 ∧ c^2 > a^2 + b^2

theorem obtuse_triangle_range_a (a : ℝ) :
  (a + (a + 1) > a + 2) →
  is_obtuse_triangle a (a + 1) (a + 2) →
  (1.5 ≤ a ∧ a < 3) :=
by
  sorry

end obtuse_triangle_range_a_l223_223861


namespace peter_total_miles_l223_223541

-- Definitions based on the conditions
def minutes_per_mile : ℝ := 20
def miles_walked_already : ℝ := 1
def additional_minutes : ℝ := 30

-- The value we want to prove
def total_miles_to_walk : ℝ := 2.5

-- Theorem statement corresponding to the proof problem
theorem peter_total_miles :
  (additional_minutes / minutes_per_mile) + miles_walked_already = total_miles_to_walk :=
sorry

end peter_total_miles_l223_223541


namespace janet_additional_money_needed_is_1225_l223_223081

def savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def months_required : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

noncomputable def total_rent : ℕ := rent_per_month * months_required
noncomputable def total_upfront_cost : ℕ := total_rent + deposit + utility_deposit + moving_costs
noncomputable def additional_money_needed : ℕ := total_upfront_cost - savings

theorem janet_additional_money_needed_is_1225 : additional_money_needed = 1225 :=
by
  sorry

end janet_additional_money_needed_is_1225_l223_223081


namespace probability_of_being_closer_to_origin_l223_223415

noncomputable def probability_closer_to_origin 
  (rect : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2})
  (origin : ℝ × ℝ := (0, 0))
  (point : ℝ × ℝ := (4, 2))
  : ℚ :=
1/3

theorem probability_of_being_closer_to_origin :
  probability_closer_to_origin = 1/3 :=
by sorry

end probability_of_being_closer_to_origin_l223_223415


namespace total_gallons_l223_223633

-- Definitions from conditions
def num_vans : ℕ := 6
def standard_capacity : ℕ := 8000
def reduced_capacity : ℕ := standard_capacity - (30 * standard_capacity / 100)
def increased_capacity : ℕ := standard_capacity + (50 * standard_capacity / 100)

-- Total number of specific types of vans
def num_standard_vans : ℕ := 2
def num_reduced_vans : ℕ := 1
def num_increased_vans : ℕ := num_vans - num_standard_vans - num_reduced_vans

-- The proof goal
theorem total_gallons : 
  (num_standard_vans * standard_capacity) + 
  (num_reduced_vans * reduced_capacity) + 
  (num_increased_vans * increased_capacity) = 
  57600 := 
by
  -- The necessary proof can be filled here
  sorry

end total_gallons_l223_223633


namespace min_value_of_a2_plus_b2_l223_223792

theorem min_value_of_a2_plus_b2 
  (a b : ℝ) 
  (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4 := 
sorry

end min_value_of_a2_plus_b2_l223_223792


namespace distance_between_consecutive_trees_l223_223802

noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (obstacle_pos : ℝ) (obstacle_gap : ℝ) : ℝ :=
  let planting_distance := yard_length - obstacle_gap
  let num_gaps := num_trees - 1
  planting_distance / num_gaps

theorem distance_between_consecutive_trees :
  distance_between_trees 600 36 250 10 = 16.857 := by
  sorry

end distance_between_consecutive_trees_l223_223802


namespace fraction_sum_condition_l223_223704

theorem fraction_sum_condition 
  (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0)
  (h : x + y = x * y): 
  (1/x + 1/y = 1) :=
by
  sorry

end fraction_sum_condition_l223_223704


namespace nellie_final_legos_l223_223897

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end nellie_final_legos_l223_223897


namespace algebraic_expression_l223_223263

-- Define a variable x
variable (x : ℝ)

-- State the theorem
theorem algebraic_expression : (5 * x - 3) = 5 * x - 3 :=
by
  sorry

end algebraic_expression_l223_223263


namespace triangle_PZQ_area_is_50_l223_223014

noncomputable def area_triangle_PZQ (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) : ℝ :=
  let RS := PQ -- since PQRS is a rectangle, RS = PQ
  let XY := RS - RX - SY
  let height := 2 * QR -- height is doubled due to triangle similarity ratio
  let area := 0.5 * PQ * height
  area

theorem triangle_PZQ_area_is_50 (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) :
  area_triangle_PZQ PQ QR RX SY hPQ hQR hRX hSY = 50 :=
  sorry

end triangle_PZQ_area_is_50_l223_223014


namespace ratio_twice_width_to_length_l223_223095

theorem ratio_twice_width_to_length (L W : ℝ) (k : ℤ)
  (h1 : L = 24)
  (h2 : W = 13.5)
  (h3 : L = k * W - 3) :
  2 * W / L = 9 / 8 := by
  sorry

end ratio_twice_width_to_length_l223_223095


namespace simplest_common_denominator_of_fractions_l223_223580

noncomputable def simplestCommonDenominator (a b : ℕ) (x y : ℕ) : ℕ := 6 * (x ^ 2) * (y ^ 3)

theorem simplest_common_denominator_of_fractions :
  simplestCommonDenominator 2 6 x y = 6 * x^2 * y^3 :=
by
  sorry

end simplest_common_denominator_of_fractions_l223_223580


namespace red_stripe_area_l223_223369

theorem red_stripe_area (diameter height stripe_width : ℝ) (num_revolutions : ℕ) 
  (diam_pos : 0 < diameter) (height_pos : 0 < height) (width_pos : 0 < stripe_width) (height_eq_80 : height = 80)
  (width_eq_3 : stripe_width = 3) (revolutions_eq_2 : num_revolutions = 2) :
  240 = stripe_width * height := 
by
  sorry

end red_stripe_area_l223_223369


namespace actual_distance_between_towns_l223_223390

theorem actual_distance_between_towns
  (d_map : ℕ) (scale1 : ℕ) (scale2 : ℕ) (distance1 : ℕ) (distance2 : ℕ) (remaining_distance : ℕ) :
  d_map = 9 →
  scale1 = 10 →
  distance1 = 5 →
  scale2 = 8 →
  remaining_distance = d_map - distance1 →
  d_map = distance1 + remaining_distance →
  (distance1 * scale1 + remaining_distance * scale2 = 82) := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end actual_distance_between_towns_l223_223390


namespace find_value_of_p_l223_223432

variable (x y : ℝ)

/-- Given that the hyperbola has the equation x^2 / 4 - y^2 / 12 = 1
    and the eccentricity e = 2, and that the parabola x = 2 * p * y^2 has its focus at (e, 0), 
    prove that the value of the real number p is 1/8. -/
theorem find_value_of_p :
  (∃ (p : ℝ), 
    (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) ∧ 
    (∀ (x y : ℝ), x = 2 * p * y^2) ∧
    (2 = 2)) →
    ∃ (p : ℝ), p = 1/8 :=
by 
  sorry

end find_value_of_p_l223_223432


namespace total_flowers_tuesday_l223_223360

def ginger_flower_shop (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) := 
  let lilacs_tuesday := lilacs_monday + lilacs_monday * 5 / 100
  let roses_tuesday := roses_monday - roses_monday * 4 / 100
  let tulips_tuesday := tulips_monday - tulips_monday * 7 / 100
  let gardenias_tuesday := gardenias_monday
  let orchids_tuesday := orchids_monday
  lilacs_tuesday + roses_tuesday + tulips_tuesday + gardenias_tuesday + orchids_tuesday

theorem total_flowers_tuesday (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) 
  (h1: lilacs_monday = 15)
  (h2: roses_monday = 3 * lilacs_monday)
  (h3: gardenias_monday = lilacs_monday / 2)
  (h4: tulips_monday = 2 * (roses_monday + gardenias_monday))
  (h5: orchids_monday = (roses_monday + gardenias_monday + tulips_monday) / 3):
  ginger_flower_shop lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday = 214 :=
by
  sorry

end total_flowers_tuesday_l223_223360


namespace smaller_of_two_digit_product_l223_223049

theorem smaller_of_two_digit_product (a b : ℕ) (h1 : a * b = 4896) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 32 :=
sorry

end smaller_of_two_digit_product_l223_223049


namespace biff_break_even_night_hours_l223_223973

-- Define the constants and conditions
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def lunch_cost : ℝ := 8
def dinner_cost : ℝ := 10
def accommodation_cost : ℝ := 35

def total_expenses_without_wifi : ℝ := ticket_cost + snacks_cost + headphones_cost + lunch_cost + dinner_cost + accommodation_cost

def earnings_per_hour : ℝ := 12
def wifi_cost_day : ℝ := 2
def wifi_cost_night : ℝ := 1

-- Define the total expenses with wifi cost variable
def total_expenses (D N : ℝ) : ℝ := total_expenses_without_wifi + (wifi_cost_day * D) + (wifi_cost_night * N)

-- Define the total earnings
def total_earnings (D N : ℝ) : ℝ := earnings_per_hour * (D + N)

-- Prove that the minimum number of hours Biff needs to work at night to break even is 8 hours
theorem biff_break_even_night_hours :
  ∃ N : ℕ, N = 8 ∧ total_earnings 0 N ≥ total_expenses 0 N := 
by 
  sorry

end biff_break_even_night_hours_l223_223973


namespace complement_of_A_is_correct_l223_223877

open Set

variable (U : Set ℝ) (A : Set ℝ)

def complement_of_A (U : Set ℝ) (A : Set ℝ) :=
  {x : ℝ | x ∉ A}

theorem complement_of_A_is_correct :
  (U = univ) →
  (A = {x : ℝ | x^2 - 2 * x > 0}) →
  (complement_of_A U A = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) :=
by
  intros hU hA
  simp [hU, hA, complement_of_A]
  sorry

end complement_of_A_is_correct_l223_223877


namespace problem1_problem2_l223_223194

-- Definitions of vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, 3)
def vec_c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Problem Part 1: Prove m = -1 given a ⊥ (b + c)
theorem problem1 (m : ℝ) (h : vec_a.1 * (vec_b + vec_c m).1 + vec_a.2 * (vec_b + vec_c m).2 = 0) : m = -1 :=
sorry

-- Problem Part 2: Prove k = -2 given k*a + b is collinear with 2*a - b
theorem problem2 (k : ℝ) (h : (k * vec_a.1 + vec_b.1) / (2 * vec_a.1 - vec_b.1) = (k * vec_a.2 + vec_b.2) / (2 * vec_a.2 - vec_b.2)) : k = -2 :=
sorry

end problem1_problem2_l223_223194


namespace optionA_is_square_difference_l223_223339

theorem optionA_is_square_difference (x y : ℝ) : 
  (-x + y) * (x + y) = -(x + y) * (x - y) :=
by sorry

end optionA_is_square_difference_l223_223339


namespace correct_relationships_l223_223250

open Real

theorem correct_relationships (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1/a < 1/b) := by
    sorry

end correct_relationships_l223_223250


namespace Alyssa_cookie_count_l223_223064

/--
  Alyssa had some cookies.
  Aiyanna has 140 cookies.
  Aiyanna has 11 more cookies than Alyssa.
  How many cookies does Alyssa have? 
-/
theorem Alyssa_cookie_count 
  (aiyanna_cookies : ℕ) 
  (more_cookies : ℕ)
  (h1 : aiyanna_cookies = 140)
  (h2 : more_cookies = 11)
  (h3 : aiyanna_cookies = alyssa_cookies + more_cookies) :
  alyssa_cookies = 129 := 
sorry

end Alyssa_cookie_count_l223_223064


namespace calc_first_term_l223_223686

theorem calc_first_term (a d : ℚ)
    (h1 : 15 * (2 * a + 29 * d) = 300)
    (h2 : 20 * (2 * a + 99 * d) = 2200) :
    a = -121 / 14 :=
by
  -- We can add the sorry placeholder here as we are not providing the complete proof steps
  sorry

end calc_first_term_l223_223686


namespace directrix_of_parabola_l223_223976

noncomputable def parabola_directrix (y : ℝ) (x : ℝ) : Prop :=
  y = 4 * x^2

theorem directrix_of_parabola : ∃ d : ℝ, (parabola_directrix (y := 4) (x := x) → d = -1/16) :=
by
  sorry

end directrix_of_parabola_l223_223976


namespace area_under_cos_l223_223724

theorem area_under_cos :
  ∫ x in (0 : ℝ)..(3 * Real.pi / 2), |Real.cos x| = 3 :=
by
  sorry

end area_under_cos_l223_223724


namespace space_per_bush_l223_223614

theorem space_per_bush (side_length : ℝ) (num_sides : ℝ) (num_bushes : ℝ) (h1 : side_length = 16) (h2 : num_sides = 3) (h3 : num_bushes = 12) :
  (num_sides * side_length) / num_bushes = 4 :=
by
  sorry

end space_per_bush_l223_223614


namespace stuart_segments_to_start_point_l223_223865

-- Definitions of given conditions
def concentric_circles {C : Type} (large small : Set C) (center : C) : Prop :=
  ∀ (x y : C), x ∈ large → y ∈ large → x ≠ y → (x = center ∨ y = center)

def tangent_to_small_circle {C : Type} (chord : Set C) (small : Set C) : Prop :=
  ∀ (x y : C), x ∈ chord → y ∈ chord → x ≠ y → (∀ z ∈ small, x ≠ z ∧ y ≠ z)

def measure_angle (ABC : Type) (θ : ℝ) : Prop :=
  θ = 60

-- The theorem to solve the problem
theorem stuart_segments_to_start_point 
    (C : Type)
    {large small : Set C} 
    {center : C} 
    {chords : List (Set C)}
    (h_concentric : concentric_circles large small center)
    (h_tangent : ∀ chord ∈ chords, tangent_to_small_circle chord small)
    (h_angle : ∀ ABC ∈ chords, measure_angle ABC 60)
    : ∃ n : ℕ, n = 3 := 
  sorry

end stuart_segments_to_start_point_l223_223865


namespace length_of_AB_l223_223206

theorem length_of_AB 
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 8)
  (hB : B.1 ^ 2 + B.2 ^ 2 = 8)
  (lA : A.1 - 2 * A.2 + 5 = 0)
  (lB : B.1 - 2 * B.2 + 5 = 0) :
  dist A B = 2 * Real.sqrt 3 := by
  sorry

end length_of_AB_l223_223206


namespace intersection_A_B_l223_223470

def set_A : Set ℝ := { x | abs (x - 1) < 2 }
def set_B : Set ℝ := { x | Real.log x / Real.log 2 > Real.log x / Real.log 3 }

theorem intersection_A_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l223_223470


namespace largest_possible_integer_in_list_l223_223209

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
  (a = 6) ∧ 
  (b = 6) ∧ 
  (c = 7) ∧ 
  (∀ x, x ≠ a ∨ x ≠ b ∨ x ≠ c → x ≠ 6) ∧ 
  (d > 7) ∧ 
  (12 = (a + b + c + d + e) / 5) ∧ 
  (max a (max b (max c (max d e))) = 33) := by
  sorry

end largest_possible_integer_in_list_l223_223209


namespace quadratic_roots_condition_l223_223061

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k*x^2 + 2*x + 1 = 0 ∧ k*y^2 + 2*y + 1 = 0) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_roots_condition_l223_223061


namespace razorback_tshirt_revenue_l223_223204

theorem razorback_tshirt_revenue 
    (total_tshirts : ℕ) (total_money : ℕ) 
    (h1 : total_tshirts = 245) 
    (h2 : total_money = 2205) : 
    (total_money / total_tshirts = 9) := 
by 
    sorry

end razorback_tshirt_revenue_l223_223204


namespace arc_length_of_curve_l223_223070

noncomputable def arc_length : ℝ :=
∫ t in (0 : ℝ)..(Real.pi / 3),
  (Real.sqrt ((t^2 * Real.cos t)^2 + (t^2 * Real.sin t)^2))

theorem arc_length_of_curve :
  arc_length = (Real.pi^3 / 81) :=
by
  sorry

end arc_length_of_curve_l223_223070


namespace day_of_month_l223_223533

/--
The 25th day of a particular month is a Monday. 
We need to prove that the 1st day of that month is a Friday.
-/
theorem day_of_month (h : (25 % 7 = 1)) : (1 % 7 = 5) :=
sorry

end day_of_month_l223_223533


namespace consecutive_integer_sum_l223_223029

theorem consecutive_integer_sum (a b c : ℕ) 
  (h1 : b = a + 2) 
  (h2 : c = a + 4) 
  (h3 : a + c = 140) 
  (h4 : b - a = 2) : a + b + c = 210 := 
sorry

end consecutive_integer_sum_l223_223029


namespace fish_worth_apples_l223_223344

-- Defining the variables
variables (f l r a : ℝ)

-- Conditions based on the problem
def condition1 : Prop := 5 * f = 3 * l
def condition2 : Prop := l = 6 * r
def condition3 : Prop := 3 * r = 2 * a

-- The statement of the problem
theorem fish_worth_apples (h1 : condition1 f l) (h2 : condition2 l r) (h3 : condition3 r a) : f = 12 / 5 * a :=
by
  sorry

end fish_worth_apples_l223_223344


namespace mul_eight_neg_half_l223_223165

theorem mul_eight_neg_half : 8 * (- (1/2: ℚ)) = -4 := 
by 
  sorry

end mul_eight_neg_half_l223_223165


namespace percentage_of_water_in_mixture_l223_223691

-- Definitions based on conditions from a)
def original_price : ℝ := 1 -- assuming $1 per liter for pure dairy
def selling_price : ℝ := 1.25 -- 25% profit means selling at $1.25
def profit_percentage : ℝ := 0.25 -- 25% profit

-- Theorem statement based on the equivalent problem in c)
theorem percentage_of_water_in_mixture : 
  (selling_price - original_price) / selling_price * 100 = 20 :=
by
  sorry

end percentage_of_water_in_mixture_l223_223691


namespace equal_ratios_l223_223931

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l223_223931


namespace count_triangles_in_figure_l223_223157

/-- 
The figure is a rectangle divided into 8 columns and 2 rows with additional diagonal and vertical lines.
We need to prove that there are 76 triangles in total in the figure.
-/
theorem count_triangles_in_figure : 
  let columns := 8 
  let rows := 2 
  let num_triangles := 76 
  ∃ total_triangles, total_triangles = num_triangles :=
by
  sorry

end count_triangles_in_figure_l223_223157


namespace total_pigs_in_barn_l223_223818

-- Define the number of pigs initially in the barn
def initial_pigs : ℝ := 2465.25

-- Define the number of pigs that join
def joining_pigs : ℝ := 5683.75

-- Define the total number of pigs after they join
def total_pigs : ℝ := 8149

-- The theorem that states the total number of pigs is the sum of initial and joining pigs
theorem total_pigs_in_barn : initial_pigs + joining_pigs = total_pigs := 
by
  sorry

end total_pigs_in_barn_l223_223818


namespace child_B_share_l223_223928

theorem child_B_share (total_money : ℕ) (ratio_A ratio_B ratio_C ratio_D ratio_E total_parts : ℕ) 
  (h1 : total_money = 12000)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 4)
  (h5 : ratio_D = 5)
  (h6 : ratio_E = 6)
  (h_total_parts : total_parts = ratio_A + ratio_B + ratio_C + ratio_D + ratio_E) :
  (total_money / total_parts) * ratio_B = 1800 :=
by
  sorry

end child_B_share_l223_223928


namespace find_x_l223_223503

noncomputable section

open Real

theorem find_x (x : ℝ) (hx : 0 < x ∧ x < 180) : 
  tan (120 * π / 180 - x * π / 180) = (sin (120 * π / 180) - sin (x * π / 180)) / (cos (120 * π / 180) - cos (x * π / 180)) →
  x = 100 :=
by
  sorry

end find_x_l223_223503


namespace expression_equals_neg_one_l223_223255

theorem expression_equals_neg_one (b y : ℝ) (hb : b ≠ 0) (h₁ : y ≠ b) (h₂ : y ≠ -b) :
  ( (b / (b + y) + y / (b - y)) / (y / (b + y) - b / (b - y)) ) = -1 :=
sorry

end expression_equals_neg_one_l223_223255


namespace first_storm_duration_l223_223226

theorem first_storm_duration
  (x y : ℕ)
  (h1 : 30 * x + 15 * y = 975)
  (h2 : x + y = 45) :
  x = 20 :=
by sorry

end first_storm_duration_l223_223226


namespace young_fish_per_pregnant_fish_l223_223041

-- Definitions based on conditions
def tanks := 3
def fish_per_tank := 4
def total_young_fish := 240

-- Calculations based on conditions
def total_pregnant_fish := tanks * fish_per_tank

-- The proof statement
theorem young_fish_per_pregnant_fish : total_young_fish / total_pregnant_fish = 20 := by
  sorry

end young_fish_per_pregnant_fish_l223_223041


namespace interest_earned_after_4_years_l223_223885

noncomputable def calculate_total_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let A := P * (1 + r) ^ t
  A - P

theorem interest_earned_after_4_years :
  calculate_total_interest 2000 0.12 4 = 1147.04 :=
by
  sorry

end interest_earned_after_4_years_l223_223885


namespace polynomial_roots_l223_223982

theorem polynomial_roots:
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end polynomial_roots_l223_223982
