import Mathlib

namespace geometric_and_arithmetic_sequence_solution_l1329_132981

theorem geometric_and_arithmetic_sequence_solution:
  ∃ a b : ℝ, 
    (a > 0) ∧                  -- a is positive
    (∃ r : ℝ, 10 * r = a ∧ a * r = 1 / 2) ∧   -- geometric sequence condition
    (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) ∧        -- arithmetic sequence condition
    a = Real.sqrt 5 ∧
    b = 10 - Real.sqrt 5 := 
by 
  sorry

end geometric_and_arithmetic_sequence_solution_l1329_132981


namespace geometric_series_common_ratio_l1329_132978

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 512) (hS : S = 2048) (h_sum : S = a / (1 - r)) : r = 3 / 4 :=
by
  rw [ha, hS] at h_sum 
  sorry

end geometric_series_common_ratio_l1329_132978


namespace speed_of_river_l1329_132957

-- Definitions of the conditions
def rowing_speed_still_water := 9 -- kmph in still water
def total_time := 1 -- hour for a round trip
def total_distance := 8.84 -- km

-- Distance to the place the man rows to
def d := total_distance / 2

-- Problem statement in Lean 4
theorem speed_of_river (v : ℝ) : 
  rowing_speed_still_water = 9 ∧
  total_time = 1 ∧
  total_distance = 8.84 →
  (4.42 / (rowing_speed_still_water + v) + 4.42 / (rowing_speed_still_water - v) = 1) →
  v = 1.2 := 
by
  sorry

end speed_of_river_l1329_132957


namespace pq_sum_eight_l1329_132933

theorem pq_sum_eight
  (p q : ℤ)
  (hp1 : p > 1)
  (hq1 : q > 1)
  (hs1 : (2 * q - 1) % p = 0)
  (hs2 : (2 * p - 1) % q = 0) : p + q = 8 := 
sorry

end pq_sum_eight_l1329_132933


namespace solve_for_x_l1329_132944

theorem solve_for_x (x y : ℝ) (h₁ : y = (x^2 - 9) / (x - 3)) (h₂ : y = 3 * x - 4) : x = 7 / 2 :=
by sorry

end solve_for_x_l1329_132944


namespace sum_first_100_terms_is_l1329_132948

open Nat

noncomputable def seq (a_n : ℕ → ℤ) : Prop :=
  a_n 2 = 2 ∧ ∀ n : ℕ, n > 0 → a_n (n + 2) + (-1)^(n + 1) * a_n n = 1 + (-1)^n

def sum_seq (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum f

theorem sum_first_100_terms_is :
  ∃ (a_n : ℕ → ℤ), seq a_n ∧ sum_seq a_n 100 = 2550 :=
by
  sorry

end sum_first_100_terms_is_l1329_132948


namespace rockham_soccer_league_members_count_l1329_132925

def cost_per_pair_of_socks : Nat := 4
def additional_cost_per_tshirt : Nat := 5
def cost_per_tshirt : Nat := cost_per_pair_of_socks + additional_cost_per_tshirt

def pairs_of_socks_per_member : Nat := 2
def tshirts_per_member : Nat := 2

def total_cost_per_member : Nat :=
  pairs_of_socks_per_member * cost_per_pair_of_socks + tshirts_per_member * cost_per_tshirt

def total_cost_all_members : Nat := 2366
def total_members : Nat := total_cost_all_members / total_cost_per_member

theorem rockham_soccer_league_members_count : total_members = 91 :=
by
  -- Given steps in the solution, verify each condition and calculation.
  sorry

end rockham_soccer_league_members_count_l1329_132925


namespace form_of_reasoning_is_incorrect_l1329_132936

-- Definitions from the conditions
def some_rational_numbers_are_fractions : Prop := 
  ∃ q : ℚ, ∃ f : ℚ, q = f / 1

def integers_are_rational_numbers : Prop :=
  ∀ z : ℤ, ∃ q : ℚ, q = z

-- The proposition to be proved
theorem form_of_reasoning_is_incorrect (h1 : some_rational_numbers_are_fractions) (h2 : integers_are_rational_numbers) : 
  ¬ ∀ z : ℤ, ∃ f : ℚ, f = z  := sorry

end form_of_reasoning_is_incorrect_l1329_132936


namespace abc_sum_square_identity_l1329_132949

theorem abc_sum_square_identity (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 941) (h2 : a + b + c = 31) :
  ab + bc + ca = 10 :=
by
  sorry

end abc_sum_square_identity_l1329_132949


namespace simplify_expression_l1329_132910

theorem simplify_expression : (2^4 * 2^4 * 2^4) = 2^12 :=
by
  sorry

end simplify_expression_l1329_132910


namespace find_b_l1329_132942

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (527816429 - b) % 17 = 0 ∧ b = 8 := 
by 
  sorry

end find_b_l1329_132942


namespace find_double_pieces_l1329_132918

theorem find_double_pieces (x : ℕ) 
  (h1 : 100 + 2 * x + 150 + 660 = 1000) : x = 45 :=
by sorry

end find_double_pieces_l1329_132918


namespace butterfly_1023_distance_l1329_132904

noncomputable def omega : Complex := Complex.exp (Complex.I * Real.pi / 4)

noncomputable def Q (n : ℕ) : Complex :=
  match n with
  | 0     => 0
  | k + 1 => Q k + (k + 1) * omega ^ k

noncomputable def butterfly_distance (n : ℕ) : ℝ := Complex.abs (Q n)

theorem butterfly_1023_distance : butterfly_distance 1023 = 511 * Real.sqrt (2 + Real.sqrt 2) :=
  sorry

end butterfly_1023_distance_l1329_132904


namespace toms_total_score_l1329_132945

def points_per_enemy : ℕ := 10
def enemies_killed : ℕ := 175

def base_score (enemies : ℕ) : ℝ := enemies * points_per_enemy

def bonus_percentage (enemies : ℕ) : ℝ :=
  if 100 ≤ enemies ∧ enemies < 150 then 0.50
  else if 150 ≤ enemies ∧ enemies < 200 then 0.75
  else if enemies ≥ 200 then 1.00
  else 0.0

def total_score (enemies : ℕ) : ℝ :=
  let base := base_score enemies
  let bonus := base * bonus_percentage enemies
  base + bonus

theorem toms_total_score :
  total_score enemies_killed = 3063 :=
by
  -- The proof will show the computed total score
  -- matches the expected value
  sorry

end toms_total_score_l1329_132945


namespace chadsRopeLength_l1329_132979

-- Define the constants and conditions
def joeysRopeLength : ℕ := 56
def joeyChadRatioNumerator : ℕ := 8
def joeyChadRatioDenominator : ℕ := 3

-- Prove that Chad's rope length is 21 cm
theorem chadsRopeLength (C : ℕ) 
  (h_ratio : joeysRopeLength * joeyChadRatioDenominator = joeyChadRatioNumerator * C) : 
  C = 21 :=
sorry

end chadsRopeLength_l1329_132979


namespace find_total_games_l1329_132976

-- Define the initial conditions
def avg_points_per_game : ℕ := 26
def games_played : ℕ := 15
def goal_avg_points : ℕ := 30
def required_avg_remaining : ℕ := 42

-- Statement of the proof problem
theorem find_total_games (G : ℕ) :
  avg_points_per_game * games_played + required_avg_remaining * (G - games_played) = goal_avg_points * G →
  G = 20 :=
by sorry

end find_total_games_l1329_132976


namespace largest_4_digit_congruent_15_mod_22_l1329_132921

theorem largest_4_digit_congruent_15_mod_22 :
  ∃ (x : ℤ), x < 10000 ∧ x % 22 = 15 ∧ (∀ (y : ℤ), y < 10000 ∧ y % 22 = 15 → y ≤ x) → x = 9981 :=
sorry

end largest_4_digit_congruent_15_mod_22_l1329_132921


namespace total_material_ordered_l1329_132943

theorem total_material_ordered (c b s : ℝ) (hc : c = 0.17) (hb : b = 0.17) (hs : s = 0.5) :
  c + b + s = 0.84 :=
by sorry

end total_material_ordered_l1329_132943


namespace range_of_x_l1329_132987

theorem range_of_x (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 3) :=
by sorry

end range_of_x_l1329_132987


namespace billy_boxes_of_candy_l1329_132959

theorem billy_boxes_of_candy (pieces_per_box total_pieces : ℕ) (h1 : pieces_per_box = 3) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 7 := 
by
  sorry

end billy_boxes_of_candy_l1329_132959


namespace girl_walking_speed_l1329_132920

-- Definitions of the conditions
def distance := 30 -- in kilometers
def time := 6 -- in hours

-- Definition of the walking speed function
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- The theorem we want to prove
theorem girl_walking_speed : speed distance time = 5 := by
  sorry

end girl_walking_speed_l1329_132920


namespace student_tickets_sold_l1329_132972

theorem student_tickets_sold (A S : ℝ) (h1 : A + S = 59) (h2 : 4 * A + 2.5 * S = 222.50) : S = 9 :=
by
  sorry

end student_tickets_sold_l1329_132972


namespace solve_work_problem_l1329_132983

variables (A B C : ℚ)

-- Conditions
def condition1 := B + C = 1/3
def condition2 := C + A = 1/4
def condition3 := C = 1/24

-- Conclusion (Question translated to proof statement)
theorem solve_work_problem (h1 : condition1 B C) (h2 : condition2 C A) (h3 : condition3 C) : A + B = 1/2 :=
by sorry

end solve_work_problem_l1329_132983


namespace remaining_hair_length_is_1_l1329_132908

-- Variables to represent the inches of hair
variable (initial_length cut_length : ℕ)

-- Given initial length and cut length
def initial_length_is_14 (initial_length : ℕ) := initial_length = 14
def cut_length_is_13 (cut_length : ℕ) := cut_length = 13

-- Definition of the remaining hair length
def remaining_length (initial_length cut_length : ℕ) := initial_length - cut_length

-- Main theorem: Proving the remaining hair length is 1 inch
theorem remaining_hair_length_is_1 : initial_length_is_14 initial_length → cut_length_is_13 cut_length → remaining_length initial_length cut_length = 1 := by
  intros h1 h2
  rw [initial_length_is_14, cut_length_is_13] at *
  simp [remaining_length]
  sorry

end remaining_hair_length_is_1_l1329_132908


namespace second_smallest_packs_of_hot_dogs_l1329_132940

theorem second_smallest_packs_of_hot_dogs
    (n : ℤ) 
    (h1 : ∃ m : ℤ, 12 * n = 8 * m + 6) :
    ∃ k : ℤ, n = 4 * k + 7 :=
sorry

end second_smallest_packs_of_hot_dogs_l1329_132940


namespace max_integer_in_form_3_x_3_sub_x_l1329_132901

theorem max_integer_in_form_3_x_3_sub_x :
  ∃ x : ℝ, ∀ y : ℝ, y = 3^(x * (3 - x)) → ⌊y⌋ ≤ 11 := 
sorry

end max_integer_in_form_3_x_3_sub_x_l1329_132901


namespace wire_length_l1329_132961

theorem wire_length (S L W : ℝ) (h1 : S = 20) (h2 : S = (2 / 7) * L) (h3 : W = S + L) : W = 90 :=
by sorry

end wire_length_l1329_132961


namespace increase_in_votes_l1329_132951

noncomputable def initial_vote_for (y : ℝ) : ℝ := 500 - y
noncomputable def revote_for (y : ℝ) : ℝ := (10 / 9) * y

theorem increase_in_votes {x x' y m : ℝ}
  (H1 : x + y = 500)
  (H2 : y - x = m)
  (H3 : x' - y = 2 * m)
  (H4 : x' + y = 500)
  (H5 : x' = (10 / 9) * y)
  (H6 : y = 282) :
  revote_for y - initial_vote_for y = 95 :=
by sorry

end increase_in_votes_l1329_132951


namespace compare_exponents_and_logs_l1329_132973

theorem compare_exponents_and_logs :
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  a > b ∧ b > c :=
by
  -- Definitions from the conditions
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  -- Proof here (omitted)
  sorry

end compare_exponents_and_logs_l1329_132973


namespace savings_percentage_correct_l1329_132980

theorem savings_percentage_correct :
  let original_price_jacket := 120
  let original_price_shirt := 60
  let original_price_shoes := 90
  let discount_jacket := 0.30
  let discount_shirt := 0.50
  let discount_shoes := 0.25
  let total_original_price := original_price_jacket + original_price_shirt + original_price_shoes
  let savings_jacket := original_price_jacket * discount_jacket
  let savings_shirt := original_price_shirt * discount_shirt
  let savings_shoes := original_price_shoes * discount_shoes
  let total_savings := savings_jacket + savings_shirt + savings_shoes
  let percentage_savings := (total_savings / total_original_price) * 100
  percentage_savings = 32.8 := 
by 
  sorry

end savings_percentage_correct_l1329_132980


namespace total_animal_crackers_eaten_l1329_132986

-- Define the context and conditions
def number_of_students : ℕ := 20
def uneaten_students : ℕ := 2
def crackers_per_pack : ℕ := 10

-- Define the statement and prove the question equals the answer given the conditions
theorem total_animal_crackers_eaten : 
  (number_of_students - uneaten_students) * crackers_per_pack = 180 := by
  sorry

end total_animal_crackers_eaten_l1329_132986


namespace total_apartment_units_l1329_132912

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end total_apartment_units_l1329_132912


namespace sufficient_not_necessary_l1329_132962

theorem sufficient_not_necessary (a : ℝ) (h : a ≠ 0) : 
  (a > 1 → a > 1 / a) ∧ (¬ (a > 1) → a > 1 / a → -1 < a ∧ a < 0) :=
sorry

end sufficient_not_necessary_l1329_132962


namespace steve_speed_on_way_back_l1329_132960

-- Let's define the variables and constants used in the problem.
def distance_to_work : ℝ := 30 -- in km
def total_time_on_road : ℝ := 6 -- in hours
def back_speed_ratio : ℝ := 2 -- Steve drives twice as fast on the way back

theorem steve_speed_on_way_back :
  ∃ v : ℝ, v > 0 ∧ (30 / v + 15 / v = 6) ∧ (2 * v = 15) := by
  sorry

end steve_speed_on_way_back_l1329_132960


namespace determine_fraction_l1329_132905

noncomputable def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

noncomputable def p (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem determine_fraction (a b : ℝ) (h : a + b = 1 / 4) :
  (p a b (-1)) / (q (-1)) = (a - b) / 4 :=
by
  sorry

end determine_fraction_l1329_132905


namespace circle_radius_five_d_value_l1329_132922

theorem circle_radius_five_d_value :
  ∀ (d : ℝ), (∃ (x y : ℝ), (x - 4)^2 + (y + 5)^2 = 41 - d) → d = 16 :=
by
  intros d h
  sorry

end circle_radius_five_d_value_l1329_132922


namespace smallest_number_with_ten_divisors_l1329_132985

/-- 
  Theorem: The smallest natural number n that has exactly 10 positive divisors is 48.
--/
theorem smallest_number_with_ten_divisors : 
  ∃ (n : ℕ), (∀ (p1 p2 p3 p4 p5 : ℕ) (a1 a2 a3 a4 a5 : ℕ), 
    n = p1^a1 * p2^a2 * p3^a3 * p4^a4 * p5^a5 → 
    n.factors.count = 10) 
    ∧ n = 48 := sorry

end smallest_number_with_ten_divisors_l1329_132985


namespace value_of_f_neg_2009_l1329_132993

def f (a b x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem value_of_f_neg_2009 (a b : ℝ) (h : f a b 2009 = 10) :
  f a b (-2009) = -14 :=
by 
  sorry

end value_of_f_neg_2009_l1329_132993


namespace sqrt_expression_meaningful_l1329_132998

/--
When is the algebraic expression √(x + 2) meaningful?
To ensure the algebraic expression √(x + 2) is meaningful, 
the expression under the square root, x + 2, must be greater than or equal to 0.
Thus, we need to prove that this condition is equivalent to x ≥ -2.
-/
theorem sqrt_expression_meaningful (x : ℝ) : (x + 2 ≥ 0) ↔ (x ≥ -2) :=
by
  sorry

end sqrt_expression_meaningful_l1329_132998


namespace number_of_cars_l1329_132931

theorem number_of_cars 
  (num_bikes : ℕ) (num_wheels_total : ℕ) (wheels_per_bike : ℕ) (wheels_per_car : ℕ)
  (h1 : num_bikes = 10) (h2 : num_wheels_total = 76) (h3 : wheels_per_bike = 2) (h4 : wheels_per_car = 4) :
  ∃ (C : ℕ), C = 14 := 
by
  sorry

end number_of_cars_l1329_132931


namespace arithmetic_seq_sum_l1329_132977

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) 
(h_given : a 2 + a 8 = 10) : 
a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l1329_132977


namespace inequality_proof_l1329_132955

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c :=
sorry

end inequality_proof_l1329_132955


namespace sum_of_distances_eq_l1329_132982

noncomputable def sum_of_distances_from_vertex_to_midpoints (A B C M N O : ℝ × ℝ) : ℝ :=
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  AM + AN + AO

theorem sum_of_distances_eq (A B C M N O : ℝ × ℝ) (h1 : B = (3, 0)) (h2 : C = (3/2, (3 * Real.sqrt 3/2))) (h3 : M = (3/2, 0)) (h4 : N = (9/4, (3 * Real.sqrt 3/4))) (h5 : O = (3/4, (3 * Real.sqrt 3/4))) :
  sum_of_distances_from_vertex_to_midpoints A B C M N O = 3 + (9 / 2) * Real.sqrt 3 :=
by
  sorry

end sum_of_distances_eq_l1329_132982


namespace seating_arrangements_l1329_132941

theorem seating_arrangements {n k : ℕ} (h1 : n = 8) (h2 : k = 6) :
  ∃ c : ℕ, c = (n - 1) * Nat.factorial k ∧ c = 20160 :=
by
  sorry

end seating_arrangements_l1329_132941


namespace max_radius_of_circle_l1329_132911

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l1329_132911


namespace jenny_grade_l1329_132939

theorem jenny_grade (J A B : ℤ) 
  (hA : A = J - 25) 
  (hB : B = A / 2) 
  (hB_val : B = 35) : 
  J = 95 :=
by
  sorry

end jenny_grade_l1329_132939


namespace fill_tank_time_l1329_132913

/-- 
If pipe A fills a tank in 30 minutes, pipe B fills the same tank in 20 minutes, 
and pipe C empties it in 40 minutes, then the time it takes to fill the tank 
when all three pipes are working together is 120/7 minutes.
-/
theorem fill_tank_time 
  (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ) (combined_rate : ℝ) (T : ℝ) :
  rate_A = 1/30 ∧ rate_B = 1/20 ∧ rate_C = -1/40 ∧ combined_rate = rate_A + rate_B + rate_C
  → T = 1 / combined_rate
  → T = 120 / 7 :=
by
  intros
  sorry

end fill_tank_time_l1329_132913


namespace total_balloons_l1329_132968

theorem total_balloons (T : ℕ) 
    (h1 : T / 4 = 100)
    : T = 400 := 
by
  sorry

end total_balloons_l1329_132968


namespace find_m_values_l1329_132906

-- Given function
def f (m x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

-- Theorem statement
theorem find_m_values (m : ℝ) :
  (∃ x y, f m x = 0 ∧ f m y = 0 ∧ (x = 0 ∨ y = 0)) →
  (m = 1 ∨ m = -(5/4)) :=
by sorry

end find_m_values_l1329_132906


namespace c_divisible_by_a_l1329_132919

theorem c_divisible_by_a {a b c : ℤ} (h1 : a ∣ b * c) (h2 : Int.gcd a b = 1) : a ∣ c :=
by
  sorry

end c_divisible_by_a_l1329_132919


namespace rect_area_162_l1329_132932

def rectangle_field_area (w l : ℝ) (A : ℝ) : Prop :=
  w = (1/2) * l ∧ 2 * (w + l) = 54 ∧ A = w * l

theorem rect_area_162 {w l A : ℝ} :
  rectangle_field_area w l A → A = 162 :=
by
  intro h
  sorry

end rect_area_162_l1329_132932


namespace g_at_2_l1329_132990

def g (x : ℝ) : ℝ := x^2 - 4

theorem g_at_2 : g 2 = 0 := by
  sorry

end g_at_2_l1329_132990


namespace symmetric_circle_equation_l1329_132915

-- Define original circle equation
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Define symmetric circle equation
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

theorem symmetric_circle_equation (x y : ℝ) : 
  symmetric_circle x y ↔ original_circle (-x) y :=
by sorry

end symmetric_circle_equation_l1329_132915


namespace max_cubes_fit_in_box_l1329_132952

theorem max_cubes_fit_in_box :
  ∀ (h w l : ℕ) (cube_vol box_max_cubes : ℕ),
    h = 12 → w = 8 → l = 9 → cube_vol = 27 → 
    box_max_cubes = (h * w * l) / cube_vol → box_max_cubes = 32 :=
by
  intros h w l cube_vol box_max_cubes h_def w_def l_def cube_vol_def box_max_cubes_def
  sorry

end max_cubes_fit_in_box_l1329_132952


namespace range_of_a_zero_value_of_a_minimum_l1329_132958

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (7 * a) / x

-- Problem 1: Range of a where f(x) has exactly one zero in its domain
theorem range_of_a_zero (a : ℝ) : 
  (∃! x : ℝ, (0 < x) ∧ f x a = 0) ↔ (a ∈ Set.Iic 0 ∪ {1 / (7 * Real.exp 1)}) := sorry

-- Problem 2: Value of a such that the minimum value of f(x) on [e, e^2] is 3
theorem value_of_a_minimum (a : ℝ) : 
  (∃ x : ℝ, (Real.exp 1 ≤ x ∧ x ≤ Real.exp 2) ∧ f x a = 3) ↔ (a = (Real.exp 2)^2 / 7) := sorry

end range_of_a_zero_value_of_a_minimum_l1329_132958


namespace range_of_p_l1329_132924

def sequence_sum (n : ℕ) : ℚ := (-1) ^ (n + 1) * (1 / 2 ^ n)

def a_n (n : ℕ) : ℚ :=
  if h : n = 0 then sequence_sum 1 else
  sequence_sum n - sequence_sum (n - 1)

theorem range_of_p (p : ℚ) : 
  (∃ n : ℕ, 0 < n ∧ (p - a_n n) * (p - a_n (n + 1)) < 0) ↔ 
  - 3 / 4 < p ∧ p < 1 / 2 :=
sorry

end range_of_p_l1329_132924


namespace coefficient_a6_l1329_132950

def expand_equation (x a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℝ) : Prop :=
  x * (x - 2) ^ 8 =
    a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 +
    a5 * (x - 1) ^ 5 + a6 * (x - 1) ^ 6 + a7 * (x - 1) ^ 7 + a8 * (x - 1) ^ 8 + 
    a9 * (x - 1) ^ 9

theorem coefficient_a6 (x a0 a1 a2 a3 a4 a5 a7 a8 a9 : ℝ) (h : expand_equation x a0 a1 a2 a3 a4 a5 (-28) a7 a8 a9) :
  a6 = -28 :=
sorry

end coefficient_a6_l1329_132950


namespace geometric_sequence_S8_l1329_132963

theorem geometric_sequence_S8 (S : ℕ → ℝ) (hs2 : S 2 = 4) (hs4 : S 4 = 16) : 
  S 8 = 160 := by
  sorry

end geometric_sequence_S8_l1329_132963


namespace sqrt_of_16_l1329_132988

theorem sqrt_of_16 : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_of_16_l1329_132988


namespace moores_law_l1329_132935

theorem moores_law (initial_transistors : ℕ) (doubling_period : ℕ) (t1 t2 : ℕ) 
  (initial_year : t1 = 1985) (final_year : t2 = 2010) (transistors_in_1985 : initial_transistors = 300000) 
  (doubles_every_two_years : doubling_period = 2) : 
  (initial_transistors * 2 ^ ((t2 - t1) / doubling_period) = 1228800000) := 
by
  sorry

end moores_law_l1329_132935


namespace monthly_income_P_l1329_132930

theorem monthly_income_P (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : (P + R) / 2 = 5200) :
  P = 4000 := 
sorry

end monthly_income_P_l1329_132930


namespace least_number_of_stamps_l1329_132927

theorem least_number_of_stamps : ∃ c f : ℕ, 3 * c + 4 * f = 50 ∧ c + f = 13 :=
by
  sorry

end least_number_of_stamps_l1329_132927


namespace train_crosses_pole_in_12_seconds_l1329_132953

noncomputable def time_to_cross_pole (speed train_length : ℕ) : ℕ := 
  train_length / speed

theorem train_crosses_pole_in_12_seconds 
  (speed : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) (train_crossing_time : ℕ)
  (h_speed : speed = 10) 
  (h_platform_length : platform_length = 320) 
  (h_time_to_cross_platform : time_to_cross_platform = 44) 
  (h_train_crossing_time : train_crossing_time = 12) :
  time_to_cross_pole speed 120 = train_crossing_time := 
by 
  sorry

end train_crosses_pole_in_12_seconds_l1329_132953


namespace problem_l1329_132966

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - (2 * m + 1) * x - 1
noncomputable def h (m : ℝ) (x : ℝ) := f m x + g m x

noncomputable def h_deriv (m : ℝ) (x : ℝ) : ℝ := m * x - (2 * m + 1) + (2 / x)

theorem problem (m : ℝ) : h_deriv m 1 = h_deriv m 3 → m = 2 / 3 :=
by
  sorry

end problem_l1329_132966


namespace no_grammatical_errors_in_B_l1329_132991

-- Definitions for each option’s description (conditions)
def sentence_A := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams."
def sentence_B := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region."
def sentence_C := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high."
def sentence_D := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves."

-- The statement that option B has no grammatical errors
theorem no_grammatical_errors_in_B : sentence_B = "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." :=
by
  sorry

end no_grammatical_errors_in_B_l1329_132991


namespace change_received_l1329_132928

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end change_received_l1329_132928


namespace Bella_average_speed_l1329_132965

theorem Bella_average_speed :
  ∀ (distance time : ℝ), 
  distance = 790 → 
  time = 15.8 → 
  (distance / time) = 50 :=
by intros distance time h_dist h_time
   -- According to the provided distances and time,
   -- we need to prove that the calculated speed is 50.
   sorry

end Bella_average_speed_l1329_132965


namespace speed_of_current_l1329_132937

theorem speed_of_current (m c : ℝ) (h1 : m + c = 20) (h2 : m - c = 18) : c = 1 :=
by
  sorry

end speed_of_current_l1329_132937


namespace checker_arrangements_five_digit_palindromes_l1329_132923

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem checker_arrangements :
  comb 32 12 * comb 20 12 = Nat.choose 32 12 * Nat.choose 20 12 := by
  sorry

theorem five_digit_palindromes :
  9 * 10 * 10 = 900 := by
  sorry

end checker_arrangements_five_digit_palindromes_l1329_132923


namespace knights_in_company_l1329_132902

theorem knights_in_company :
  ∃ k : ℕ, (k = 0 ∨ k = 6) ∧ k ≤ 39 ∧
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 39) →
    (∃ i : ℕ, (1 ≤ i ∧ i ≤ 39) ∧ n * k = 1 + (i - 1) * k) →
    ∃ i : ℕ, ∃ nk : ℕ, (nk = i * k ∧ nk ≤ 39 ∧ (nk ∣ k → i = 1 + (i - 1))) :=
by
  sorry

end knights_in_company_l1329_132902


namespace question_1_question_2_question_3_l1329_132903

variable (a b : ℝ)

-- (a * b)^n = a^n * b^n for natural numbers n
theorem question_1 (n : ℕ) : (a * b)^n = a^n * b^n := sorry

-- Calculate 2^5 * (-1/2)^5
theorem question_2 : 2^5 * (-1/2)^5 = -1 := sorry

-- Calculate (-0.125)^2022 * 2^2021 * 4^2020
theorem question_3 : (-0.125)^2022 * 2^2021 * 4^2020 = 1 / 32 := sorry

end question_1_question_2_question_3_l1329_132903


namespace percentage_of_money_spent_l1329_132994

theorem percentage_of_money_spent (initial_amount remaining_amount : ℝ) (h_initial : initial_amount = 500) (h_remaining : remaining_amount = 350) :
  (((initial_amount - remaining_amount) / initial_amount) * 100) = 30 :=
by
  -- Start the proof
  sorry

end percentage_of_money_spent_l1329_132994


namespace probability_point_not_above_x_axis_l1329_132900

theorem probability_point_not_above_x_axis (A B C D : ℝ × ℝ) :
  A = (9, 4) →
  B = (3, -2) →
  C = (-3, -2) →
  D = (3, 4) →
  (1 / 2 : ℚ) = 1 / 2 := 
by 
  intros hA hB hC hD 
  sorry

end probability_point_not_above_x_axis_l1329_132900


namespace rationalize_denominator_ABC_l1329_132970

theorem rationalize_denominator_ABC :
  let expr := (2 + Real.sqrt 5) / (3 - 2 * Real.sqrt 5)
  ∃ A B C : ℤ, expr = A + B * Real.sqrt C ∧ A * B * (C:ℤ) = -560 :=
by
  sorry

end rationalize_denominator_ABC_l1329_132970


namespace total_fruits_picked_l1329_132934

theorem total_fruits_picked (g_oranges g_apples a_oranges a_apples o_oranges o_apples : ℕ) :
  g_oranges = 45 →
  g_apples = a_apples + 5 →
  a_oranges = g_oranges - 18 →
  a_apples = 15 →
  o_oranges = 6 * 3 →
  o_apples = 6 * 2 →
  g_oranges + g_apples + a_oranges + a_apples + o_oranges + o_apples = 137 :=
by
  intros
  sorry

end total_fruits_picked_l1329_132934


namespace form_five_squares_l1329_132969

-- The conditions of the problem as premises
variables (initial_configuration : Set (ℕ × ℕ))               -- Initial positions of 12 matchsticks
          (final_configuration : Set (ℕ × ℕ))                 -- Final positions of matchsticks to form 5 squares
          (fixed_matchsticks : Set (ℕ × ℕ))                    -- Positions of 6 fixed matchsticks
          (movable_matchsticks : Set (ℕ × ℕ))                 -- Positions of 6 movable matchsticks

-- Condition to avoid duplication or free ends
variables (no_duplication : Prop)
          (no_free_ends : Prop)

-- Proof statement
theorem form_five_squares : ∃ rearranged_configuration, 
  rearranged_configuration = final_configuration ∧
  initial_configuration = fixed_matchsticks ∪ movable_matchsticks ∧
  no_duplication ∧
  no_free_ends :=
sorry -- Proof omitted.

end form_five_squares_l1329_132969


namespace repeating_decimal_as_fraction_l1329_132947

-- Given conditions
def repeating_decimal : ℚ := 7 + 832 / 999

-- Goal: Prove that the repeating decimal 7.\overline{832} equals 70/9
theorem repeating_decimal_as_fraction : repeating_decimal = 70 / 9 := by
  unfold repeating_decimal
  sorry

end repeating_decimal_as_fraction_l1329_132947


namespace add_two_integers_l1329_132989

/-- If the difference of two positive integers is 5 and their product is 180,
then their sum is 25. -/
theorem add_two_integers {x y : ℕ} (h1: x > y) (h2: x - y = 5) (h3: x * y = 180) : x + y = 25 :=
sorry

end add_two_integers_l1329_132989


namespace classA_classC_ratio_l1329_132926

-- Defining the sizes of classes B and C as given in conditions
def classB_size : ℕ := 20
def classC_size : ℕ := 120

-- Defining the size of class A based on the condition that it is twice as big as class B
def classA_size : ℕ := 2 * classB_size

-- Theorem to prove that the ratio of the size of class A to class C is 1:3
theorem classA_classC_ratio : classA_size / classC_size = 1 / 3 := 
sorry

end classA_classC_ratio_l1329_132926


namespace pow_div_pow_eq_result_l1329_132946

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end pow_div_pow_eq_result_l1329_132946


namespace find_x_l1329_132916
-- The first priority is to ensure the generated Lean code can be built successfully.

theorem find_x (x : ℤ) (h : 9823 + x = 13200) : x = 3377 :=
by
  sorry

end find_x_l1329_132916


namespace simplify_expression_l1329_132995

theorem simplify_expression (x y : ℝ) (h1 : x = 10) (h2 : y = -1/25) :
  ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = 2 / 5 := 
by
  sorry

end simplify_expression_l1329_132995


namespace cocktail_cost_per_litre_l1329_132954

theorem cocktail_cost_per_litre :
  let mixed_fruit_cost := 262.85
  let acai_berry_cost := 3104.35
  let mixed_fruit_volume := 37
  let acai_berry_volume := 24.666666666666668
  let total_cost := mixed_fruit_volume * mixed_fruit_cost + acai_berry_volume * acai_berry_cost
  let total_volume := mixed_fruit_volume + acai_berry_volume
  total_cost / total_volume = 1400 :=
by
  sorry

end cocktail_cost_per_litre_l1329_132954


namespace cistern_width_l1329_132997

theorem cistern_width (w : ℝ) (h : 8 * w + 2 * (1.25 * 8) + 2 * (1.25 * w) = 83) : w = 6 :=
by
  sorry

end cistern_width_l1329_132997


namespace find_some_multiplier_l1329_132964

theorem find_some_multiplier (m : ℕ) :
  (422 + 404)^2 - (m * 422 * 404) = 324 ↔ m = 4 :=
by
  sorry

end find_some_multiplier_l1329_132964


namespace vertical_asymptote_sum_l1329_132974

theorem vertical_asymptote_sum : 
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  ∃ p q : ℝ, (6 * p ^ 2 + 7 * p + 3 = 0) ∧ (6 * q ^ 2 + 7 * q + 3 = 0) ∧ p + q = -11 / 6 :=
by
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  exact sorry

end vertical_asymptote_sum_l1329_132974


namespace geom_sequence_second_term_l1329_132999

noncomputable def geom_sequence_term (a r : ℕ) (n : ℕ) : ℕ := a * r^(n-1)

theorem geom_sequence_second_term 
  (a1 a5: ℕ) (r: ℕ) 
  (h1: a1 = 5)
  (h2: a5 = geom_sequence_term a1 r 5)
  (h3: a5 = 320)
  (h_r: r^4 = 64): 
  geom_sequence_term a1 r 2 = 10 :=
by
  sorry

end geom_sequence_second_term_l1329_132999


namespace distinct_remainders_count_l1329_132971

theorem distinct_remainders_count {N : ℕ} (hN : N = 420) :
  ∃ (count : ℕ), (∀ n : ℕ, n ≥ 1 ∧ n ≤ N → ((n % 5 ≠ n % 6) ∧ (n % 5 ≠ n % 7) ∧ (n % 6 ≠ n % 7))) →
  count = 386 :=
by {
  sorry
}

end distinct_remainders_count_l1329_132971


namespace max_value_of_expression_l1329_132984

theorem max_value_of_expression :
  ∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 → 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 :=
by sorry

end max_value_of_expression_l1329_132984


namespace probability_r25_to_r35_l1329_132929

theorem probability_r25_to_r35 (n : ℕ) (r : Fin n → ℕ) (h : n = 50) 
  (distinct : ∀ i j : Fin n, i ≠ j → r i ≠ r j) : 1 + 1260 = 1261 :=
by
  sorry

end probability_r25_to_r35_l1329_132929


namespace units_digit_base9_addition_l1329_132996

theorem units_digit_base9_addition : 
  (∃ (d₁ d₂ : ℕ), d₁ < 9 ∧ d₂ < 9 ∧ (85 % 9 = d₁) ∧ (37 % 9 = d₂)) → ((d₁ + d₂) % 9 = 3) :=
by
  sorry

end units_digit_base9_addition_l1329_132996


namespace bouquets_ratio_l1329_132938

theorem bouquets_ratio (monday tuesday wednesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 3 * monday) 
  (h3 : monday + tuesday + wednesday = 60) :
  wednesday / tuesday = 1 / 3 :=
by sorry

end bouquets_ratio_l1329_132938


namespace correct_calculation_l1329_132967

theorem correct_calculation (a : ℝ) : -2 * a + (2 * a - 1) = -1 := by
  sorry

end correct_calculation_l1329_132967


namespace two_le_three_l1329_132975

/-- Proof that the proposition "2 ≤ 3" is true given the logical connective. -/
theorem two_le_three : 2 ≤ 3 := 
by
  sorry

end two_le_three_l1329_132975


namespace decimal_to_binary_25_l1329_132992

theorem decimal_to_binary_25: (1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0) = 25 :=
by 
  sorry

end decimal_to_binary_25_l1329_132992


namespace max_and_next_max_values_l1329_132917

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log a) / b

theorem max_and_next_max_values :
  let values := [4.0^(1/4), 5.0^(1/5), 16.0^(1/16), 25.0^(1/25)]
  ∃ max2 max1, 
    max1 = 4.0^(1/4) ∧ max2 = 5.0^(1/5) ∧ 
    (∀ x ∈ values, x <= max1) ∧ 
    (∀ x ∈ values, x < max1 → x <= max2) :=
by
  sorry

end max_and_next_max_values_l1329_132917


namespace sugar_mixture_problem_l1329_132909

theorem sugar_mixture_problem :
  ∃ x : ℝ, (9 * x + 7 * (63 - x) = 0.9 * (9.24 * 63)) ∧ x = 41.724 :=
by
  sorry

end sugar_mixture_problem_l1329_132909


namespace ratio_of_sides_of_rectangles_l1329_132956

theorem ratio_of_sides_of_rectangles (s x y : ℝ) 
  (hsx : x + s = 2 * s) 
  (hsy : s + 2 * y = 2 * s)
  (houter_inner_area : (2 * s) ^ 2 = 4 * s ^ 2) : 
  x / y = 2 :=
by
  -- Assuming the conditions hold, we are interested in proving that the ratio x / y = 2
  -- The proof will be provided here
  sorry

end ratio_of_sides_of_rectangles_l1329_132956


namespace sin_three_pi_over_two_l1329_132907

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_over_two_l1329_132907


namespace quadrilateral_trapezoid_or_parallelogram_l1329_132914

theorem quadrilateral_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ)
  (hs : s1^2 = s2 * s4) :
  (exists (is_trapezoid : Prop), is_trapezoid) ∨ (exists (is_parallelogram : Prop), is_parallelogram) :=
by
  sorry

end quadrilateral_trapezoid_or_parallelogram_l1329_132914
