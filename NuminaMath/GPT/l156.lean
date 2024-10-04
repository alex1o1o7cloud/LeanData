import Mathlib

namespace ramesh_transport_cost_l156_156106

-- Definitions for conditions
def labelled_price (P : ℝ) : Prop := P = 13500 / 0.80
def selling_price (P : ℝ) : Prop := P * 1.10 = 18975
def transport_cost (T : ℝ) (extra_amount : ℝ) (installation_cost : ℝ) : Prop := T = extra_amount - installation_cost

-- The theorem statement to be proved
theorem ramesh_transport_cost (P T extra_amount installation_cost: ℝ) 
  (h1 : labelled_price P) 
  (h2 : selling_price P) 
  (h3 : extra_amount = 18975 - P)
  (h4 : installation_cost = 250) : 
  transport_cost T extra_amount installation_cost :=
by
  sorry

end ramesh_transport_cost_l156_156106


namespace equal_number_of_experienced_fishermen_and_children_l156_156020

theorem equal_number_of_experienced_fishermen_and_children 
  (n : ℕ)
  (total_fish : ℕ)
  (children_catch : ℕ)
  (fishermen_catch : ℕ)
  (h1 : total_fish = n^2 + 5 * n + 22)
  (h2 : fishermen_catch - 10 = children_catch)
  (h3 : total_fish = n * children_catch + 11 * fishermen_catch)
  (h4 : fishermen_catch > children_catch)
  : n = 11 := 
sorry

end equal_number_of_experienced_fishermen_and_children_l156_156020


namespace find_value_of_a_l156_156973

theorem find_value_of_a (a : ℝ) (h : 0.005 * a = 65) : a = 130 := 
by
  sorry

end find_value_of_a_l156_156973


namespace tank_filling_time_l156_156675

theorem tank_filling_time (p q r s : ℝ) (leakage : ℝ) :
  (p = 1 / 6) →
  (q = 1 / 12) →
  (r = 1 / 24) →
  (s = 1 / 18) →
  (leakage = -1 / 48) →
  (1 / (p + q + r + s + leakage) = 48 / 15.67) :=
by
  intros hp hq hr hs hleak
  rw [hp, hq, hr, hs, hleak]
  norm_num
  sorry

end tank_filling_time_l156_156675


namespace quadratic_inequality_range_of_k_l156_156374

theorem quadratic_inequality_range_of_k :
  ∀ k : ℝ , (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) ↔ (-1 < k ∧ k ≤ 0) :=
sorry

end quadratic_inequality_range_of_k_l156_156374


namespace simplify_fraction_l156_156115

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end simplify_fraction_l156_156115


namespace ratio_proof_l156_156733

noncomputable def side_length_triangle(a : ℝ) : ℝ := a / 3
noncomputable def side_length_square(b : ℝ) : ℝ := b / 4
noncomputable def area_triangle(a : ℝ) : ℝ := (side_length_triangle(a)^2 * Mathlib.sqrt(3)) / 4
noncomputable def area_square(b : ℝ) : ℝ := (side_length_square(b))^2

theorem ratio_proof (a b : ℝ) (h : area_triangle(a) = area_square(b)) : a / b = 2 * Mathlib.sqrt(3) / 9 :=
by {
  sorry
}

end ratio_proof_l156_156733


namespace sqrt_18_mul_sqrt_32_eq_24_l156_156426
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l156_156426


namespace find_original_list_size_l156_156965

theorem find_original_list_size
  (n m : ℤ)
  (h1 : (m + 3) * (n + 1) = m * n + 20)
  (h2 : (m + 1) * (n + 2) = m * n + 22):
  n = 7 :=
sorry

end find_original_list_size_l156_156965


namespace probability_correct_l156_156044

noncomputable def probability_parallel_not_coincident : ℚ :=
  let total_points := 6
  let lines := total_points.choose 2
  let total_ways := lines * lines
  let parallel_not_coincident_pairs := 12
  parallel_not_coincident_pairs / total_ways

theorem probability_correct :
  probability_parallel_not_coincident = 4 / 75 := by
  sorry

end probability_correct_l156_156044


namespace weight_of_rod_l156_156784

theorem weight_of_rod (length1 length2 weight1 weight2 weight_per_meter : ℝ)
  (h1 : length1 = 6) (h2 : weight1 = 22.8) (h3 : length2 = 11.25)
  (h4 : weight_per_meter = weight1 / length1) :
  weight2 = weight_per_meter * length2 :=
by
  -- The proof would go here
  sorry

end weight_of_rod_l156_156784


namespace solve_custom_eq_l156_156613

namespace CustomProof

def custom_mul (a b : ℕ) : ℕ := a * b + a + b

theorem solve_custom_eq (x : ℕ) (h : custom_mul 3 x = 31) : x = 7 := 
by
  sorry

end CustomProof

end solve_custom_eq_l156_156613


namespace part1_l156_156497

def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

def setB (x : ℝ) : Prop := x ≠ 0 ∧ x ≤ 5 ∧ 0 < x

def setC (a x : ℝ) : Prop := 3 * a ≤ x ∧ x ≤ 2 * a + 1

def setInter (x : ℝ) : Prop := setA x ∧ setB x

theorem part1 (a : ℝ) : (∀ x, setC a x → setInter x) ↔ (0 < a ∧ a ≤ 1 / 2 ∨ 1 < a) :=
sorry

end part1_l156_156497


namespace quadratic_residue_distribution_even_odd_quadratic_residue_distribution_sum_of_quadratic_residues_l156_156540

theorem quadratic_residue_distribution (p : ℕ) (hp_prime: Nat.Prime p) (hp_mod: p % 4 = 1) :
  let residues := (1 + p) / 4 in
  ∃ qres nres, qres = residues ∧ nres = residues ∧
  set.univ.filter (λ x, (∃ y, y^2 % p = x)).card = qres ∧ -- count of quadratic residues
  set.univ.filter (λ x, ¬(∃ y, y^2 % p = x)).card = nres  -- count of non-quadratic residues
:= sorry

theorem even_odd_quadratic_residue_distribution (p : ℕ) (hp_prime: Nat.Prime p) (hp_mod: p % 4 = 1) :
  let residues := (p - 1) / 4 in
  ∃ even_qres odd_nres,
  set.univ.filter (λ x, x % 2 = 0 ∧ (∃ y, y^2 % p = x)).card = residues ∧ -- even quadratic residues
  set.univ.filter (λ x, x % 2 = 1 ∧ ¬(∃ y, y^2 % p = x)).card = residues   -- odd non-quadratic residues
:= sorry

theorem sum_of_quadratic_residues (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) :
  let sum_qr := p * (p - 1) / 4 in
  ∑ i in (set.univ.filter (λ x, (∃ y, y^2 % p = x)).to_finset), i = sum_qr
:= sorry

end quadratic_residue_distribution_even_odd_quadratic_residue_distribution_sum_of_quadratic_residues_l156_156540


namespace corrected_mean_is_correct_l156_156454

-- Define the initial conditions
def initial_mean : ℝ := 36
def n_obs : ℝ := 50
def incorrect_obs : ℝ := 23
def correct_obs : ℝ := 45

-- Calculate the incorrect total sum
def incorrect_total_sum : ℝ := initial_mean * n_obs

-- Define the corrected total sum
def corrected_total_sum : ℝ := incorrect_total_sum - incorrect_obs + correct_obs

-- State the main theorem to be proved
theorem corrected_mean_is_correct : corrected_total_sum / n_obs = 36.44 := by
  sorry

end corrected_mean_is_correct_l156_156454


namespace smallest_number_of_lawyers_l156_156192

/-- Given that:
- n is the number of delegates, where 220 < n < 254
- m is the number of economists, so the number of lawyers is n - m
- Each participant played with each other participant exactly once.
- A match winner got one point, the loser got none, and in case of a draw, both participants received half a point each.
- By the end of the tournament, each participant gained half of all their points from matches against economists.

Prove that the smallest number of lawyers participating in the tournament is 105. -/
theorem smallest_number_of_lawyers (n m : ℕ) (h1 : 220 < n) (h2 : n < 254)
  (h3 : m * (m - 1) + (n - m) * (n - m - 1) = n * (n - 1))
  (h4 : m * (m - 1) = 2 * (n * (n - 1)) / 4) :
  n - m = 105 :=
sorry

end smallest_number_of_lawyers_l156_156192


namespace total_project_hours_l156_156092

def research_hours : ℕ := 10
def proposal_hours : ℕ := 2
def report_hours_left : ℕ := 8

theorem total_project_hours :
  research_hours + proposal_hours + report_hours_left = 20 := 
  sorry

end total_project_hours_l156_156092


namespace minimum_value_x_squared_plus_12x_plus_5_l156_156567

theorem minimum_value_x_squared_plus_12x_plus_5 : ∃ x : ℝ, x^2 + 12 * x + 5 = -31 :=
by sorry

end minimum_value_x_squared_plus_12x_plus_5_l156_156567


namespace inequality_solution_l156_156345

theorem inequality_solution (x : ℝ) : (5 < x ∧ x ≤ 6) ↔ (x-3)/(x-5) ≥ 3 :=
by
  sorry

end inequality_solution_l156_156345


namespace total_bricks_proof_l156_156093

-- Define the initial conditions
def initial_courses := 3
def bricks_per_course := 400
def additional_courses := 2

-- Compute the number of bricks removed from the last course
def bricks_removed_from_last_course (bricks_per_course: ℕ) : ℕ :=
  bricks_per_course / 2

-- Calculate the total number of bricks
def total_bricks (initial_courses : ℕ) (bricks_per_course : ℕ) (additional_courses : ℕ) (bricks_removed : ℕ) : ℕ :=
  (initial_courses + additional_courses) * bricks_per_course - bricks_removed

-- Given values and the proof problem
theorem total_bricks_proof :
  total_bricks initial_courses bricks_per_course additional_courses (bricks_removed_from_last_course bricks_per_course) = 1800 :=
by
  sorry

end total_bricks_proof_l156_156093


namespace find_girls_l156_156272

theorem find_girls (n : ℕ) (h : 1 - (1 / Nat.choose (3 + n) 3) = 34 / 35) : n = 4 :=
  sorry

end find_girls_l156_156272


namespace discount_correct_l156_156663

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end discount_correct_l156_156663


namespace ratio_of_A_to_B_l156_156547

theorem ratio_of_A_to_B (A B C : ℝ) (hB : B = 270) (hBC : B = (1 / 4) * C) (hSum : A + B + C = 1440) : A / B = 1 / 3 :=
by
  -- The proof is omitted for this example
  sorry

end ratio_of_A_to_B_l156_156547


namespace polynomial_transformable_l156_156842

theorem polynomial_transformable (a b c d : ℝ) :
  (∃ A B : ℝ, ∀ z : ℝ, z^4 + A * z^2 + B = (z + a/4)^4 + a * (z + a/4)^3 + b * (z + a/4)^2 + c * (z + a/4) + d) ↔ a^3 - 4 * a * b + 8 * c = 0 :=
by
  sorry

end polynomial_transformable_l156_156842


namespace minimum_blue_chips_l156_156721

theorem minimum_blue_chips (w r b : ℕ) : 
  (b ≥ w / 3) ∧ (b ≤ r / 4) ∧ (w + b ≥ 75) → b ≥ 19 :=
by sorry

end minimum_blue_chips_l156_156721


namespace probability_angle_AMB_acute_l156_156385

theorem probability_angle_AMB_acute :
  let side_length := 4
  let square_area := side_length * side_length
  let semicircle_area := (1 / 2) * Real.pi * (side_length / 2) ^ 2
  let probability := 1 - semicircle_area / square_area
  probability = 1 - (Real.pi / 8) :=
sorry

end probability_angle_AMB_acute_l156_156385


namespace mn_value_l156_156518

-- Definitions
def exponent_m := 2
def exponent_n := 2

-- Theorem statement
theorem mn_value : exponent_m * exponent_n = 4 :=
by
  sorry

end mn_value_l156_156518


namespace solve_money_conditions_l156_156493

theorem solve_money_conditions 
  (a b : ℝ)
  (h1 : b - 4 * a < 78)
  (h2 : 6 * a - b = 36) :
  a < 57 ∧ b > -36 :=
sorry

end solve_money_conditions_l156_156493


namespace Seokhyung_drank_the_most_l156_156103

-- Define the conditions
def Mina_Amount := 0.6
def Seokhyung_Amount := 1.5
def Songhwa_Amount := Seokhyung_Amount - 0.6

-- Statement to prove that Seokhyung drank the most cola
theorem Seokhyung_drank_the_most : Seokhyung_Amount > Mina_Amount ∧ Seokhyung_Amount > Songhwa_Amount :=
by
  -- Proof skipped
  sorry

end Seokhyung_drank_the_most_l156_156103


namespace number_of_valid_permutations_l156_156066

noncomputable def count_valid_permutations : Nat :=
  let multiples_of_77 := [154, 231, 308, 385, 462, 539, 616, 693, 770, 847, 924]
  let total_count := multiples_of_77.foldl (fun acc x =>
    if x == 770 then
      acc + 3
    else if x == 308 then
      acc + 6 - 2
    else
      acc + 6) 0
  total_count

theorem number_of_valid_permutations : count_valid_permutations = 61 :=
  sorry

end number_of_valid_permutations_l156_156066


namespace find_pairs_l156_156624

theorem find_pairs (x y : Nat) (h : 1 + x + x^2 + x^3 + x^4 = y^2) : (x, y) = (0, 1) ∨ (x, y) = (3, 11) := by
  sorry

end find_pairs_l156_156624


namespace more_roses_than_orchids_l156_156703

-- Definitions
def roses_now : Nat := 12
def orchids_now : Nat := 2

-- Theorem statement
theorem more_roses_than_orchids : (roses_now - orchids_now) = 10 := by
  sorry

end more_roses_than_orchids_l156_156703


namespace find_opposite_of_neg_half_l156_156958

-- Define the given number
def given_num : ℚ := -1/2

-- Define what it means to find the opposite of a number
def opposite (x : ℚ) : ℚ := -x

-- State the theorem
theorem find_opposite_of_neg_half : opposite given_num = 1/2 :=
by
  -- Proof is omitted for now
  sorry

end find_opposite_of_neg_half_l156_156958


namespace part_a_correct_part_b_correct_l156_156330

-- Define the alphabet and mapping
inductive Letter
| C | H | M | O
deriving DecidableEq, Inhabited

open Letter

def letter_to_base4 (ch : Letter) : ℕ :=
  match ch with
  | C => 0
  | H => 1
  | M => 2
  | O => 3

def word_to_base4 (word : List Letter) : ℕ :=
  word.foldl (fun acc ch => acc * 4 + letter_to_base4 ch) 0

def base4_to_letter (n : ℕ) : Letter :=
  match n with
  | 0 => C
  | 1 => H
  | 2 => M
  | 3 => O
  | _ => C -- This should not occur if input is in valid base-4 range

def base4_to_word (n : ℕ) (size : ℕ) : List Letter :=
  if size = 0 then []
  else
    let quotient := n / 4
    let remainder := n % 4
    base4_to_letter remainder :: base4_to_word quotient (size - 1)

-- The size of the words is fixed at 8
def word_size : ℕ := 8

noncomputable def part_a : List Letter :=
  base4_to_word 2017 word_size

theorem part_a_correct :
  part_a = [H, O, O, H, M, C] := by
  sorry

def given_word : List Letter :=
  [H, O, M, C, H, O, M, C]

noncomputable def part_b : ℕ :=
  word_to_base4 given_word + 1 -- Adjust for zero-based indexing

theorem part_b_correct :
  part_b = 29299 := by
  sorry

end part_a_correct_part_b_correct_l156_156330


namespace quadratic_equal_real_roots_l156_156786

theorem quadratic_equal_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) ↔ m = 1/4 :=
by sorry

end quadratic_equal_real_roots_l156_156786


namespace number_of_unit_distance_pairs_lt_bound_l156_156495

/-- Given n distinct points in the plane, the number of pairs of points with a unit distance between them is less than n / 4 + (1 / sqrt 2) * n^(3 / 2). -/
theorem number_of_unit_distance_pairs_lt_bound (n : ℕ) (hn : 0 < n) :
  ∃ E : ℕ, E < n / 4 + (1 / Real.sqrt 2) * n^(3 / 2) :=
by
  sorry

end number_of_unit_distance_pairs_lt_bound_l156_156495


namespace maximum_value_of_f_inequality_holds_for_all_x_l156_156062

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

theorem maximum_value_of_f (a : ℝ) (h : 0 ≤ a) : 
  (∀ x, f a x ≤ f a 1) → f a 1 = 3 / Real.exp 1 → a = 1 := 
by 
  sorry

theorem inequality_holds_for_all_x (b : ℝ) : 
  (∀ a ≤ 0, ∀ x, 0 ≤ x → f a x ≤ b * Real.log (x + 1)) → 1 ≤ b := 
by 
  sorry

end maximum_value_of_f_inequality_holds_for_all_x_l156_156062


namespace fraction_sum_eq_one_l156_156038

theorem fraction_sum_eq_one (m n : ℝ) (h : m ≠ n) : (m / (m - n) + n / (n - m) = 1) :=
by
  sorry

end fraction_sum_eq_one_l156_156038


namespace geometric_sequence_problem_l156_156227

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_problem (a : ℕ → ℝ) (ha : geometric_sequence a) (h : a 4 + a 8 = 1 / 2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 1 / 4 :=
sorry

end geometric_sequence_problem_l156_156227


namespace sin_120_eq_sqrt3_div_2_l156_156214

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l156_156214


namespace bones_in_beef_l156_156724

def price_of_beef_with_bones : ℝ := 78
def price_of_boneless_beef : ℝ := 90
def price_of_bones : ℝ := 15
def fraction_of_bones_in_kg : ℝ := 0.16
def grams_per_kg : ℝ := 1000

theorem bones_in_beef :
  (fraction_of_bones_in_kg * grams_per_kg = 160) :=
by
  sorry

end bones_in_beef_l156_156724


namespace exist_three_sum_eq_third_l156_156541

theorem exist_three_sum_eq_third
  (A : Finset ℕ)
  (h_card : A.card = 52)
  (h_cond : ∀ (a : ℕ), a ∈ A → a ≤ 100) :
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = z :=
sorry

end exist_three_sum_eq_third_l156_156541


namespace simplify_expression_l156_156275

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x * (x - 4) = 2 * x^2 + 4 := by
  sorry

end simplify_expression_l156_156275


namespace width_of_park_l156_156183

theorem width_of_park (L : ℕ) (A_lawn : ℕ) (w_road : ℕ) (W : ℚ) :
  L = 60 → A_lawn = 2109 → w_road = 3 →
  60 * W - 2 * 60 * 3 = 2109 →
  W = 41.15 :=
by
  intros hL hA_lawn hw_road hEq
  -- The proof will go here
  sorry

end width_of_park_l156_156183


namespace quadratic_passing_origin_l156_156697

theorem quadratic_passing_origin (a b c : ℝ) (h : a ≠ 0) :
  ((∀ x y : ℝ, x = 0 → y = 0 → y = a * x^2 + b * x + c) ↔ c = 0) := 
by
  sorry

end quadratic_passing_origin_l156_156697


namespace probability_neither_event_l156_156580

theorem probability_neither_event (P_A P_B P_A_and_B : ℝ)
  (h1 : P_A = 0.25)
  (h2 : P_B = 0.40)
  (h3 : P_A_and_B = 0.20) :
  1 - (P_A + P_B - P_A_and_B) = 0.55 :=
by
  sorry

end probability_neither_event_l156_156580


namespace probability_of_condition_l156_156028

def Q_within_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

def condition (x y : ℝ) : Prop :=
  y > (1/2) * x

theorem probability_of_condition : 
  ∀ x y, Q_within_square x y → (0.75 = 3 / 4) :=
by
  sorry

end probability_of_condition_l156_156028


namespace spent_on_board_game_l156_156619

theorem spent_on_board_game (b : ℕ)
  (h1 : 4 * 7 = 28)
  (h2 : b + 28 = 30) : 
  b = 2 := 
sorry

end spent_on_board_game_l156_156619


namespace find_symmetric_point_l156_156717

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def line_equation (t : ℝ) : Point :=
  { x := -t, y := 1.5, z := 2 + t }

def M : Point := { x := -1, y := 0, z := -1 }

def is_midpoint (M M' M0 : Point) : Prop :=
  M0.x = (M.x + M'.x) / 2 ∧
  M0.y = (M.y + M'.y) / 2 ∧
  M0.z = (M.z + M'.z) / 2

theorem find_symmetric_point (M0 : Point) (h_line : ∃ t, M0 = line_equation t) :
  ∃ M' : Point, is_midpoint M M' M0 ∧ M' = { x := 3, y := 3, z := 3 } :=
sorry

end find_symmetric_point_l156_156717


namespace solve_for_n_l156_156819

theorem solve_for_n (n : ℕ) : (9^n * 9^n * 9^n * 9^n = 729^4) -> n = 3 := 
by
  sorry

end solve_for_n_l156_156819


namespace find_k_l156_156898

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem find_k (k : ℝ) (h : deriv (f k) 0 = 27) : k = 3 :=
by
  sorry

end find_k_l156_156898


namespace car_service_month_l156_156749

-- Define the conditions
def first_service_month : ℕ := 3 -- Representing March as the 3rd month
def service_interval : ℕ := 7
def total_services : ℕ := 13

-- Define an auxiliary function to calculate months and reduce modulo 12
def nth_service_month (first_month : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  (first_month + (interval * (n - 1))) % 12

-- The theorem statement
theorem car_service_month : nth_service_month first_service_month service_interval total_services = 3 :=
by
  -- The proof steps will go here
  sorry

end car_service_month_l156_156749


namespace find_f_prime_at_one_l156_156571

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end find_f_prime_at_one_l156_156571


namespace simplify_expr_l156_156422

theorem simplify_expr : (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1 / 2 := by
  sorry

end simplify_expr_l156_156422


namespace penguin_permutations_correct_l156_156648

def num_permutations_of_multiset (total : ℕ) (freqs : List ℕ) : ℕ :=
  Nat.factorial total / (freqs.foldl (λ acc x => acc * Nat.factorial x) 1)

def penguin_permutations : ℕ := num_permutations_of_multiset 7 [2, 1, 1, 1, 1, 1]

theorem penguin_permutations_correct : penguin_permutations = 2520 := by
  sorry

end penguin_permutations_correct_l156_156648


namespace train_length_l156_156390

theorem train_length (L : ℝ) (h1 : (L + 120) / 60 = L / 20) : L = 60 := 
sorry

end train_length_l156_156390


namespace find_number_l156_156566

theorem find_number (n : ℝ) (h : 3 / 5 * ((2 / 3 + 3 / 8) / n) - 1 / 16 = 0.24999999999999994) : n = 48 :=
  sorry

end find_number_l156_156566


namespace vacation_cost_l156_156131

theorem vacation_cost (n : ℕ) (h : 480 / n + 40 = 120) : n = 6 :=
sorry

end vacation_cost_l156_156131


namespace num_divisors_of_64m4_l156_156351

-- A positive integer m such that 120 * m^3 has 120 divisors
def has_120_divisors (m : ℕ) : Prop := (m > 0) ∧ ((List.range (120 * m^3 + 1)).filter (λ d, (120 * m^3) % d = 0)).length = 120

-- Prove that if such an m exists, then 64 * m^4 has 675 divisors
theorem num_divisors_of_64m4 (m : ℕ) (h : has_120_divisors m) : ((List.range (64 * m^4 + 1)).filter (λ d, (64 * m^4) % d = 0)).length = 675 :=
by
  sorry

end num_divisors_of_64m4_l156_156351


namespace least_prime_factor_of_5pow6_minus_5pow4_l156_156844

def least_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then (Nat.minFac n) else 0

theorem least_prime_factor_of_5pow6_minus_5pow4 : least_prime_factor (5^6 - 5^4) = 2 := by
  sorry

end least_prime_factor_of_5pow6_minus_5pow4_l156_156844


namespace moles_NaHCO3_combined_l156_156626

-- Define conditions as given in the problem
def moles_HNO3_combined := 1
def moles_NaNO3_result := 1

-- The chemical equation as a definition
def balanced_reaction (moles_NaHCO3 moles_HNO3 moles_NaNO3 : ℕ) : Prop :=
  moles_HNO3 = moles_NaNO3 ∧ moles_NaHCO3 = moles_HNO3

-- The proof problem statement
theorem moles_NaHCO3_combined :
  balanced_reaction 1 moles_HNO3_combined moles_NaNO3_result → 1 = 1 :=
by 
  sorry

end moles_NaHCO3_combined_l156_156626


namespace interchanged_digit_multiple_of_sum_l156_156377

theorem interchanged_digit_multiple_of_sum (n a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : n = 3 * (a + b)) 
  (h3 : 1 ≤ a) (h4 : a ≤ 9) 
  (h5 : 0 ≤ b) (h6 : b ≤ 9) : 
  10 * b + a = 8 * (a + b) := 
by 
  sorry

end interchanged_digit_multiple_of_sum_l156_156377


namespace trig_identity_l156_156855

theorem trig_identity : 
  ( 4 * Real.sin (40 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) / Real.cos (20 * Real.pi / 180) 
   - Real.tan (20 * Real.pi / 180) ) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l156_156855


namespace point_on_line_has_correct_y_l156_156856

theorem point_on_line_has_correct_y (a : ℝ) : (2 * 3 + a - 7 = 0) → a = 1 :=
by 
  sorry

end point_on_line_has_correct_y_l156_156856


namespace six_letter_word_combinations_l156_156800

theorem six_letter_word_combinations : ∃ n : ℕ, n = 26 * 26 * 26 := 
sorry

end six_letter_word_combinations_l156_156800


namespace sandy_more_tokens_than_siblings_l156_156112

-- Define the initial conditions
def initial_tokens : ℕ := 3000000
def initial_transaction_fee_percent : ℚ := 0.10
def value_increase_percent : ℚ := 0.20
def additional_tokens : ℕ := 500000
def additional_transaction_fee_percent : ℚ := 0.07
def sandy_keep_percent : ℚ := 0.40
def siblings : ℕ := 7
def sibling_transaction_fee_percent : ℚ := 0.05

-- Define the main theorem to prove
theorem sandy_more_tokens_than_siblings :
  let received_initial_tokens := initial_tokens * (1 - initial_transaction_fee_percent)
  let increased_tokens := received_initial_tokens * (1 + value_increase_percent)
  let received_additional_tokens := additional_tokens * (1 - additional_transaction_fee_percent)
  let total_tokens := increased_tokens + received_additional_tokens
  let sandy_tokens := total_tokens * sandy_keep_percent
  let remaining_tokens := total_tokens * (1 - sandy_keep_percent)
  let each_sibling_tokens := remaining_tokens / siblings * (1 - sibling_transaction_fee_percent)
  sandy_tokens - each_sibling_tokens = 1180307.1428 := sorry

end sandy_more_tokens_than_siblings_l156_156112


namespace hyperbola_focus_l156_156379

-- Definition of the hyperbola equation and foci
def is_hyperbola (x y : ℝ) (k : ℝ) : Prop :=
  x^2 - k * y^2 = 1

-- Definition of the hyperbola having a focus at (3, 0) and the value of k
def has_focus_at (k : ℝ) : Prop :=
  ∃ x y : ℝ, is_hyperbola x y k ∧ (x, y) = (3, 0)

theorem hyperbola_focus (k : ℝ) (h : has_focus_at k) : k = 1 / 8 :=
  sorry

end hyperbola_focus_l156_156379


namespace original_average_speed_l156_156990

theorem original_average_speed :
  ∀ (D : ℝ),
  (V = D / (5 / 6)) ∧ (60 = D / (2 / 3)) → V = 48 :=
by
  sorry

end original_average_speed_l156_156990


namespace smallest_n_for_sum_or_difference_divisible_l156_156629

theorem smallest_n_for_sum_or_difference_divisible (n : ℕ) :
  (∃ n : ℕ, ∀ (S : Finset ℤ), S.card = n → (∃ (x y : ℤ) (h₁ : x ≠ y), ((x + y) % 1991 = 0) ∨ ((x - y) % 1991 = 0))) ↔ n = 997 :=
sorry

end smallest_n_for_sum_or_difference_divisible_l156_156629


namespace bella_age_l156_156472

theorem bella_age (B : ℕ) 
  (h1 : (B + 9) + B + B / 2 = 27) 
  : B = 6 :=
by sorry

end bella_age_l156_156472


namespace zach_cookies_left_l156_156304

/- Defining the initial conditions on cookies baked each day -/
def cookies_monday : ℕ := 32
def cookies_tuesday : ℕ := cookies_monday / 2
def cookies_wednesday : ℕ := 3 * cookies_tuesday - 4 - 3
def cookies_thursday : ℕ := 2 * cookies_monday - 10 + 5
def cookies_friday : ℕ := cookies_wednesday - 6 - 4
def cookies_saturday : ℕ := cookies_monday + cookies_friday - 10

/- Aggregating total cookies baked throughout the week -/
def total_baked : ℕ := cookies_monday + cookies_tuesday + cookies_wednesday +
                      cookies_thursday + cookies_friday + cookies_saturday

/- Defining cookies lost each day -/
def daily_parents_eat : ℕ := 2 * 6
def neighbor_friday_eat : ℕ := 8
def friends_thursday_eat : ℕ := 3 * 2

def total_lost : ℕ := 4 + 3 + 10 + 6 + 4 + 10 + daily_parents_eat + neighbor_friday_eat + friends_thursday_eat

/- Calculating cookies left at end of six days -/
def cookies_left : ℕ := total_baked - total_lost

/- Proof objective -/
theorem zach_cookies_left : cookies_left = 200 := by
  sorry

end zach_cookies_left_l156_156304


namespace range_of_f_find_cos2θ_l156_156644

open Real

-- Definitions for f(x)
def f (x : ℝ) : ℝ := cos x * cos (x + π / 3)

-- Proof problem 1: Prove the range of f(x) in [0, π/2] is [-1/4, 1/2]
theorem range_of_f : set.image f (set.Icc 0 (π / 2)) = set.Icc (-1 / 4) (1 / 2) := 
sorry

-- Proof problem 2: If f(θ) = 13/20 and -π/6 < θ < π/6, find cos 2θ
theorem find_cos2θ (θ : ℝ) (h1 : f θ = 13 / 20) (h2 : -π / 6 < θ) (h3 : θ < π / 6) : cos (2 * θ) = (4 - 3 * sqrt 3) / 10 := 
sorry

end range_of_f_find_cos2θ_l156_156644


namespace A_finishes_work_in_8_days_l156_156458

theorem A_finishes_work_in_8_days 
  (A_work B_work W : ℝ) 
  (h1 : 4 * A_work + 6 * B_work = W)
  (h2 : (A_work + B_work) * 4.8 = W) :
  A_work = W / 8 :=
by
  -- We should provide the proof here, but we will use "sorry" for now.
  sorry

end A_finishes_work_in_8_days_l156_156458


namespace discriminant_of_polynomial_l156_156488

noncomputable def polynomial_discriminant (a b c : ℚ) : ℚ :=
b^2 - 4 * a * c

theorem discriminant_of_polynomial : polynomial_discriminant 2 (4 - (1/2 : ℚ)) 1 = 17 / 4 :=
by
  sorry

end discriminant_of_polynomial_l156_156488


namespace number_of_valid_partitions_l156_156259

open Finset

def valid_partition_count (s : Finset ℕ) :=
  ∑ n in (range 12).filter (λ n, n ≠ 6),
    choose 10 (n - 1)

theorem number_of_valid_partitions :
  let s := range (12 + 1) \ {0, 12} in
  valid_partition_count s - choose 10 5 = 772 :=
by simp [valid_partition_count, choose, Nat.choose]

end number_of_valid_partitions_l156_156259


namespace slope_angle_of_vertical_line_l156_156832

theorem slope_angle_of_vertical_line :
  ∀ {θ : ℝ}, (∀ x, (x = 3) → x = 3) → θ = 90 := by
  sorry

end slope_angle_of_vertical_line_l156_156832


namespace solve_trig_eq_l156_156680

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l156_156680


namespace candy_cookies_l156_156740

def trays : Nat := 4
def cookies_per_tray : Nat := 24
def packs : Nat := 8
def total_cookies : Nat := trays * cookies_per_tray
def cookies_per_pack : Nat := total_cookies / packs

theorem candy_cookies : 
  cookies_per_pack = 12 := 
by
  -- Calculate total cookies
  have h1 : total_cookies = trays * cookies_per_tray := rfl
  have h2 : total_cookies = 96 := by rw [h1]; norm_num
  
  -- Calculate cookies per pack
  have h3 : cookies_per_pack = total_cookies / packs := rfl
  have h4 : cookies_per_pack = 12 := by rw [h3, h2]; norm_num
  
  exact h4

end candy_cookies_l156_156740


namespace num_k_vals_l156_156756

-- Definitions of the conditions
def div_by_7 (n k : ℕ) : Prop :=
  (2 * 3^(6*n) + k * 2^(3*n + 1) - 1) % 7 = 0

-- Main theorem statement
theorem num_k_vals : 
  ∃ (S : Finset ℕ), (∀ k ∈ S, k < 100 ∧ ∀ n, div_by_7 n k) ∧ S.card = 14 := 
by
  sorry

end num_k_vals_l156_156756


namespace largest_of_seven_consecutive_integers_l156_156699

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2821) : 
  n + 6 = 406 := 
by
  -- Proof steps can be added here
  sorry

end largest_of_seven_consecutive_integers_l156_156699


namespace solve_x_from_equation_l156_156845

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end solve_x_from_equation_l156_156845


namespace pureAcidInSolution_l156_156850

/-- Define the conditions for the problem -/
def totalVolume : ℝ := 12
def percentageAcid : ℝ := 0.40

/-- State the theorem equivalent to the question:
    calculate the amount of pure acid -/
theorem pureAcidInSolution :
  totalVolume * percentageAcid = 4.8 := by
  sorry

end pureAcidInSolution_l156_156850


namespace number_of_companion_relation_subsets_l156_156378

def isCompanionRelationSet (A : Set ℚ) : Prop :=
  ∀ x ∈ A, (x ≠ 0 → (1 / x) ∈ A)

def M : Set ℚ := {-1, 0, 1 / 3, 1 / 2, 1, 2, 3, 4}

theorem number_of_companion_relation_subsets :
  ∃ n, n = 15 ∧
  (∀ A ⊆ M, isCompanionRelationSet A) :=
sorry

end number_of_companion_relation_subsets_l156_156378


namespace majka_numbers_product_l156_156668

/-- Majka created a three-digit funny and a three-digit cheerful number.
    - The funny number starts with an odd digit and alternates between odd and even.
    - The cheerful number starts with an even digit and alternates between even and odd.
    - All digits are distinct and nonzero.
    - The sum of these two numbers is 1617.
    - The product of these two numbers ends in 40.
    Prove that the product of these numbers is 635040.
-/
theorem majka_numbers_product :
  ∃ (a b c : ℕ) (D E F : ℕ),
    -- Define 3-digit funny number as (100 * a + 10 * b + c)
    -- with a and c odd, b even, and all distinct and nonzero
    (a % 2 = 1) ∧ (c % 2 = 1) ∧ (b % 2 = 0) ∧
    -- Define 3-digit cheerful number as (100 * D + 10 * E + F)
    -- with D and F even, E odd, and all distinct and nonzero
    (D % 2 = 0) ∧ (F % 2 = 0) ∧ (E % 2 = 1) ∧
    -- All digits are distinct and nonzero
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ D ∧ a ≠ E ∧ a ≠ F ∧
     b ≠ c ∧ b ≠ D ∧ b ≠ E ∧ b ≠ F ∧
     c ≠ D ∧ c ≠ E ∧ c ≠ F ∧
     D ≠ E ∧ D ≠ F ∧ E ≠ F) ∧
    (100 * a + 10 * b + c + 100 * D + 10 * E + F = 1617) ∧
    ((100 * a + 10 * b + c) * (100 * D + 10 * E + F) = 635040) := sorry

end majka_numbers_product_l156_156668


namespace exists_infinite_solutions_l156_156543

theorem exists_infinite_solutions :
  ∃ (x y z : ℤ), (∀ k : ℤ, x = 2 * k ∧ y = 999 - 2 * k ^ 2 ∧ z = 998 - 2 * k ^ 2) ∧ (x ^ 2 + y ^ 2 - z ^ 2 = 1997) :=
by 
  -- The proof should go here
  sorry

end exists_infinite_solutions_l156_156543


namespace dan_speed_must_exceed_48_l156_156691

theorem dan_speed_must_exceed_48 (d : ℕ) (s_cara : ℕ) (time_delay : ℕ) : 
  d = 120 → s_cara = 30 → time_delay = 3 / 2 → ∃ v : ℕ, v > 48 :=
by
  intro h1 h2 h3
  use 49
  sorry

end dan_speed_must_exceed_48_l156_156691


namespace value_of_f_l156_156569

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end value_of_f_l156_156569


namespace find_smallest_number_of_lawyers_l156_156191

noncomputable def smallest_number_of_lawyers (n : ℕ) (m : ℕ) : ℕ :=
if 220 < n ∧ n < 254 ∧
     (∀ x, 0 < x ≤ (n-1) ↔ (∃ p, (p = 1 ∨ p = 0.5) ∧ 
                                   (x + x = p * (n-1) ∧ 
                                   ∃ e_points, e_points = m * (m-1) / 2) ∧ 
                                   ∃ l_points, l_points = (n-m) * (n-m-1) / 2 ∧ 
                                   (e_points + l_points = n * (n-1) / 2))) 
then n - m else 0

theorem find_smallest_number_of_lawyers : 
  ∃ n m, 220 < n ∧ n < 254 ∧
         (∀ x, 0 < x ≤ (n-1) ↔ (∃ p, (p = 1 ∨ p = 0.5) ∧ 
                                   (x + x = p * (n-1) ∧ 
                                   ∃ e_points, e_points = m * (m-1) / 2) ∧ 
                                   ∃ l_points, l_points = (n-m) * (n-m-1) / 2 ∧ 
                                   (e_points + l_points = n * (n-1) / 2))) ∧
         smallest_number_of_lawyers n m = 105 :=
sorry

end find_smallest_number_of_lawyers_l156_156191


namespace linear_function_intersects_x_axis_at_2_0_l156_156109

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end linear_function_intersects_x_axis_at_2_0_l156_156109


namespace solve_quadratic_solve_inequality_system_l156_156310

theorem solve_quadratic :
  ∀ x : ℝ, x^2 - 6 * x + 5 = 0 ↔ x = 1 ∨ x = 5 :=
sorry

theorem solve_inequality_system :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2 * (x + 1) < 4) ↔ (-3 < x ∧ x < 1) :=
sorry

end solve_quadratic_solve_inequality_system_l156_156310


namespace minimize_total_cost_l156_156316

open Real

noncomputable def total_cost (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100) : ℝ :=
  (130 / x) * 2 * (2 + (x^2 / 360)) + (14 * 130 / x)

theorem minimize_total_cost :
  ∀ (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100),
  total_cost x h = (2340 / x) + (13 * x / 18)
  ∧ (x = 18 * sqrt 10 → total_cost x h = 26 * sqrt 10) :=
by
  sorry

end minimize_total_cost_l156_156316


namespace h_value_l156_156534

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem h_value :
  ∃ (h : ℝ → ℝ), (h 0 = 7)
  ∧ (∃ (a b c : ℝ), (f a = 0) ∧ (f b = 0) ∧ (f c = 0) ∧ (h (-8) = (1/49) * (-8 - a^3) * (-8 - b^3) * (-8 - c^3))) 
  ∧ h (-8) = -1813 := by
  sorry

end h_value_l156_156534


namespace find_divisor_l156_156306

theorem find_divisor (d : ℕ) (q r : ℕ) (h₁ : 190 = q * d + r) (h₂ : q = 9) (h₃ : r = 1) : d = 21 :=
by
  sorry

end find_divisor_l156_156306


namespace equalize_foma_ierema_l156_156142

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l156_156142


namespace find_y_l156_156710

theorem find_y (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_rem : x % y = 3) (h_div : (x:ℝ) / y = 96.15) : y = 20 :=
by
  sorry

end find_y_l156_156710


namespace fomagive_55_l156_156149

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l156_156149


namespace average_salary_l156_156975

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 14000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

theorem average_salary : (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8200 := 
  by 
    sorry

end average_salary_l156_156975


namespace find_x1_l156_156498

theorem find_x1 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/4) 
  : x1 = 3/4 := 
sorry

end find_x1_l156_156498


namespace maximum_value_of_f_for_n_le_1991_l156_156399

def f : ℕ → ℕ 
| 0       := 0
| 1       := 1
| (n + 2) := f (n / 2) + n % 2 -- this is translated from f(n) = f(floor(n/2)) + n - 2*floor(n/2)

theorem maximum_value_of_f_for_n_le_1991 : ∀ (_ : 0 ≤ n ∧ n ≤ 1991), f n ≤ 10 ∧ (∃ n, 0 ≤ n ∧ n ≤ 1991 ∧ f n = 10) :=
by
  sorry

end maximum_value_of_f_for_n_le_1991_l156_156399


namespace base_b_arithmetic_l156_156881

theorem base_b_arithmetic (b : ℕ) (h1 : 4 + 3 = 7) (h2 : 6 + 2 = 8) (h3 : 4 + 6 = 10) (h4 : 3 + 4 + 1 = 8) : b = 9 :=
  sorry

end base_b_arithmetic_l156_156881


namespace inequality_correct_l156_156538

open BigOperators

theorem inequality_correct {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ (a + b)^2 / 4 ∧ (a + b)^2 / 4 ≥ a * b :=
by 
  sorry

end inequality_correct_l156_156538


namespace intersection_complement_l156_156506

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 5, 6})
variable (hB : B = {1, 3, 4, 6, 7})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 5} :=
sorry

end intersection_complement_l156_156506


namespace ratio_sandra_amy_ruth_l156_156816

/-- Given the amounts received by Sandra and Amy, and an unknown amount received by Ruth,
    the ratio of the money shared between Sandra, Amy, and Ruth is 2:1:R/50. -/
theorem ratio_sandra_amy_ruth (R : ℝ) (hAmy : 50 > 0) (hSandra : 100 > 0) :
  (100 : ℝ) / 50 = 2 ∧ (50 : ℝ) / 50 = 1 ∧ ∃ (R : ℝ), (100/50 : ℝ) = 2 ∧ (50/50 : ℝ) = 1 ∧ (R / 50 : ℝ) = (R / 50 : ℝ) :=
by
  sorry

end ratio_sandra_amy_ruth_l156_156816


namespace last_year_sales_l156_156730

-- Define the conditions as constants
def sales_this_year : ℝ := 480
def percent_increase : ℝ := 0.50

-- The main theorem statement
theorem last_year_sales : 
  ∃ sales_last_year : ℝ, sales_this_year = sales_last_year * (1 + percent_increase) ∧ sales_last_year = 320 := 
by 
  sorry

end last_year_sales_l156_156730


namespace problem1_problem2_l156_156814

-- Statement for problem 1
theorem problem1 : 
  (-2020 - 2 / 3) + (2019 + 3 / 4) + (-2018 - 5 / 6) + (2017 + 1 / 2) = -2 - 1 / 4 := 
sorry

-- Statement for problem 2
theorem problem2 : 
  (-1 - 1 / 2) + (-2000 - 5 / 6) + (4000 + 3 / 4) + (-1999 - 2 / 3) = -5 / 4 := 
sorry

end problem1_problem2_l156_156814


namespace max_gcd_of_13n_plus_3_and_7n_plus_1_l156_156327

theorem max_gcd_of_13n_plus_3_and_7n_plus_1 (n : ℕ) (hn : 0 < n) :
  ∃ d, d = Nat.gcd (13 * n + 3) (7 * n + 1) ∧ ∀ m, m = Nat.gcd (13 * n + 3) (7 * n + 1) → m ≤ 8 := 
sorry

end max_gcd_of_13n_plus_3_and_7n_plus_1_l156_156327


namespace this_year_sales_l156_156462

def last_year_sales : ℝ := 320 -- in millions
def percent_increase : ℝ := 0.5 -- 50%

theorem this_year_sales : (last_year_sales * (1 + percent_increase)) = 480 := by
  sorry

end this_year_sales_l156_156462


namespace pentagons_from_15_points_l156_156342

theorem pentagons_from_15_points (n : ℕ) (h : n = 15) : (nat.choose 15 5) = 3003 := by
  rw h
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end pentagons_from_15_points_l156_156342


namespace seq_b_is_geometric_l156_156888

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence {a_n} with first term a_1 and common ratio q
def a_n (a₁ q : α) (n : ℕ) : α := a₁ * q^(n-1)

-- Define the sequence {b_n}
def b_n (a₁ q : α) (n : ℕ) : α :=
  a_n a₁ q (3*n - 2) + a_n a₁ q (3*n - 1) + a_n a₁ q (3*n)

-- Theorem stating {b_n} is a geometric sequence with common ratio q^3
theorem seq_b_is_geometric (a₁ q : α) (h : q ≠ 1) :
  ∀ n : ℕ, b_n a₁ q (n + 1) = q^3 * b_n a₁ q n :=
by
  sorry

end seq_b_is_geometric_l156_156888


namespace distance_cycled_l156_156991

variable (v t d : ℝ)

theorem distance_cycled (h1 : d = v * t)
                        (h2 : d = (v + 1) * (3 * t / 4))
                        (h3 : d = (v - 1) * (t + 3)) :
                        d = 36 :=
by
  sorry

end distance_cycled_l156_156991


namespace ones_digit_of_73_pow_351_l156_156757

theorem ones_digit_of_73_pow_351 : 
  (73 ^ 351) % 10 = 7 := 
by 
  sorry

end ones_digit_of_73_pow_351_l156_156757


namespace expected_value_parabola_l156_156504

def X_distribution (a b : ℤ) : ℚ :=
  if a=b then 0
  else if |a-b|=1 then 1
  else 2

theorem expected_value_parabola (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∀ a b ∈ {-3, -2, -1, 0, 1, 2, 3}, 
                       a * b > 0 ∧ a ≠ 0 ∧ b ≠ 0) :
  let X := |a - b| in (X = 0 * 1/3 + 1 * 4/9 + 2 * 2/9) → 
  (∑ e in {0, 1, 2}, e * X_distribution a b) / 3 * 3 * 2 * 7 = 8/9 :=
begin
  sorry
end

end expected_value_parabola_l156_156504


namespace number_of_candidates_l156_156989

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 132) : n = 12 :=
by
  sorry

end number_of_candidates_l156_156989


namespace percent_motorists_no_ticket_l156_156411

theorem percent_motorists_no_ticket (M : ℝ) :
  (0.14285714285714285 * M - 0.10 * M) / (0.14285714285714285 * M) * 100 = 30 :=
by
  sorry

end percent_motorists_no_ticket_l156_156411


namespace paint_needed_for_snake_l156_156796

open Nat

def total_paint (paint_per_segment segments additional_paint : Nat) : Nat :=
  paint_per_segment * segments + additional_paint

theorem paint_needed_for_snake :
  total_paint 240 336 20 = 80660 :=
by
  sorry

end paint_needed_for_snake_l156_156796


namespace sum_lent_l156_156728

theorem sum_lent (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ)
  (hR: R = 4) 
  (hT: T = 8) 
  (hI1 : I = P - 306) 
  (hI2 : I = P * R * T / 100) :
  P = 450 :=
by
  sorry

end sum_lent_l156_156728


namespace nth_equation_l156_156410

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by
  sorry

end nth_equation_l156_156410


namespace profit_percentage_is_correct_l156_156318

-- Definitions for the given conditions
def SP : ℝ := 850
def Profit : ℝ := 255
def CP : ℝ := SP - Profit

-- The target proof statement
theorem profit_percentage_is_correct : 
  (Profit / CP) * 100 = 42.86 := by
  sorry

end profit_percentage_is_correct_l156_156318


namespace total_toys_l156_156599

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end total_toys_l156_156599


namespace diminished_radius_10_percent_l156_156960

theorem diminished_radius_10_percent
  (r r' : ℝ) 
  (h₁ : r > 0)
  (h₂ : r' > 0)
  (h₃ : (π * r'^2) = 0.8100000000000001 * (π * r^2)) :
  r' = 0.9 * r :=
by sorry

end diminished_radius_10_percent_l156_156960


namespace total_cost_of_items_l156_156596

variables (E P M : ℝ)

-- Conditions
def condition1 : Prop := E + 3 * P + 2 * M = 240
def condition2 : Prop := 2 * E + 5 * P + 4 * M = 440

-- Question to prove
def question (E P M : ℝ) : ℝ := 3 * E + 4 * P + 6 * M

theorem total_cost_of_items (E P M : ℝ) :
  condition1 E P M →
  condition2 E P M →
  question E P M = 520 := 
by 
  intros h1 h2
  sorry

end total_cost_of_items_l156_156596


namespace min_odd_integers_l156_156163

-- Definitions of the conditions
variable (a b c d e f : ℤ)

-- The mathematical theorem statement
theorem min_odd_integers 
  (h1 : a + b = 30)
  (h2 : a + b + c + d = 50) 
  (h3 : a + b + c + d + e + f = 70)
  (h4 : e + f % 2 = 1) : 
  ∃ n, n ≥ 1 ∧ n = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                    (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                    (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0) :=
sorry

end min_odd_integers_l156_156163


namespace integer_difference_divisible_by_n_l156_156273

theorem integer_difference_divisible_by_n (n : ℕ) (h : n > 0) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end integer_difference_divisible_by_n_l156_156273


namespace line_equation_min_intercepts_l156_156984

theorem line_equation_min_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : 1 / a + 4 / b = 1) : 2 * 1 + 4 - 6 = 0 ↔ (a = 3 ∧ b = 6) :=
by
  sorry

end line_equation_min_intercepts_l156_156984


namespace find_value_of_triangle_l156_156893

theorem find_value_of_triangle (p : ℕ) (triangle : ℕ) 
  (h1 : triangle + p = 47) 
  (h2 : 3 * (triangle + p) - p = 133) :
  triangle = 39 :=
by 
  sorry

end find_value_of_triangle_l156_156893


namespace amount_paid_after_discount_l156_156031

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end amount_paid_after_discount_l156_156031


namespace five_digit_numbers_last_two_different_l156_156065

def total_five_digit_numbers : ℕ := 90000

def five_digit_numbers_last_two_same : ℕ := 9000

theorem five_digit_numbers_last_two_different :
  (total_five_digit_numbers - five_digit_numbers_last_two_same) = 81000 := 
by 
  sorry

end five_digit_numbers_last_two_different_l156_156065


namespace convex_pentagons_from_15_points_l156_156344

theorem convex_pentagons_from_15_points : (Nat.choose 15 5) = 3003 := 
by
  sorry

end convex_pentagons_from_15_points_l156_156344


namespace markers_blue_l156_156612

theorem markers_blue {total_markers red_markers blue_markers : ℝ} 
  (h_total : total_markers = 64.0) 
  (h_red : red_markers = 41.0) 
  (h_blue : blue_markers = total_markers - red_markers) : 
  blue_markers = 23.0 := 
by 
  sorry

end markers_blue_l156_156612


namespace probability_same_spot_l156_156704

theorem probability_same_spot :
  let students := ["A", "B"]
  let spots := ["Spot 1", "Spot 2"]
  let total_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 1"), ("B", "Spot 2")),
                         (("A", "Spot 2"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]
  let favorable_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                             (("A", "Spot 2"), ("B", "Spot 2"))]
  ∀ (students : List String) (spots : List String)
    (total_outcomes favorable_outcomes : List ((String × String) × (String × String))),
  (students = ["A", "B"]) →
  (spots = ["Spot 1", "Spot 2"]) →
  (total_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                     (("A", "Spot 1"), ("B", "Spot 2")),
                     (("A", "Spot 2"), ("B", "Spot 1")),
                     (("A", "Spot 2"), ("B", "Spot 2"))]) →
  (favorable_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]) →
  favorable_outcomes.length / total_outcomes.length = 1 / 2 := 
by
  intros
  sorry

end probability_same_spot_l156_156704


namespace david_total_course_hours_l156_156999

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l156_156999


namespace compare_logs_l156_156225

noncomputable def a := Real.log 3
noncomputable def b := Real.log 3 / Real.log 2 / 2
noncomputable def c := Real.log 2 / Real.log 3 / 2

theorem compare_logs : a > b ∧ b > c := by
  sorry

end compare_logs_l156_156225


namespace sum_of_primes_1_to_20_l156_156002

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l156_156002


namespace fomagive_55_l156_156147

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l156_156147


namespace parakeet_eats_2_grams_per_day_l156_156415

-- Define the conditions
def parrot_daily : ℕ := 14
def finch_daily (parakeet_daily : ℕ) : ℕ := parakeet_daily / 2
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def num_finches : ℕ := 4
def total_weekly_consumption : ℕ := 266

-- Define the daily consumption equation for all birds
def daily_consumption (parakeet_daily : ℕ) : ℕ :=
  num_parakeets * parakeet_daily + num_parrots * parrot_daily + num_finches * finch_daily parakeet_daily

-- Define the weekly consumption equation
def weekly_consumption (parakeet_daily : ℕ) : ℕ :=
  7 * daily_consumption parakeet_daily

-- State the theorem to prove that each parakeet eats 2 grams per day
theorem parakeet_eats_2_grams_per_day :
  (weekly_consumption 2) = total_weekly_consumption ↔ 2 = 2 :=
by
  sorry

end parakeet_eats_2_grams_per_day_l156_156415


namespace bus_is_there_probability_l156_156584

noncomputable def probability_bus_present : ℚ :=
  let total_area := 90 * 90
  let triangle_area := (75 * 75) / 2
  let parallelogram_area := 75 * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem bus_is_there_probability :
  probability_bus_present = 7/16 :=
by
  sorry

end bus_is_there_probability_l156_156584


namespace foma_gives_ierema_55_l156_156151

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l156_156151


namespace intersection_of_sets_l156_156900

def setM : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }
def setN : Set ℝ := { x | Real.log x ≥ 0 }

theorem intersection_of_sets : (setM ∩ setN) = { x | 1 ≤ x ∧ x ≤ 4 } := 
by {
  sorry
}

end intersection_of_sets_l156_156900


namespace translate_vertex_to_increase_l156_156505

def quadratic_function (x : ℝ) : ℝ := -x^2 + 1

theorem translate_vertex_to_increase (x : ℝ) :
  ∃ v, v = (2, quadratic_function 2) ∧
    (∀ x < 2, quadratic_function (x + 2) = quadratic_function x + 1 ∧
    ∀ x < 2, quadratic_function x < quadratic_function (x + 1)) :=
sorry

end translate_vertex_to_increase_l156_156505


namespace ratio_of_areas_l156_156684

noncomputable def side_length_WXYZ : ℝ := 16

noncomputable def WJ : ℝ := (3/4) * side_length_WXYZ
noncomputable def JX : ℝ := (1/4) * side_length_WXYZ

noncomputable def side_length_JKLM := 4 * Real.sqrt 2

noncomputable def area_JKLM := (side_length_JKLM)^2
noncomputable def area_WXYZ := (side_length_WXYZ)^2

theorem ratio_of_areas : area_JKLM / area_WXYZ = 1 / 8 :=
by
  sorry

end ratio_of_areas_l156_156684


namespace characterize_set_A_l156_156809

open Int

noncomputable def A : Set ℤ := { x | x^2 - 3 * x - 4 < 0 }

theorem characterize_set_A : A = {0, 1, 2, 3} :=
by
  sorry

end characterize_set_A_l156_156809


namespace cos_double_angle_l156_156763

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4 / 5 := 
  sorry

end cos_double_angle_l156_156763


namespace cylinder_surface_area_correct_l156_156866

noncomputable def cylinder_surface_area :=
  let r := 8   -- radius in cm
  let h := 10  -- height in cm
  let arc_angle := 90 -- degrees
  let x := 40
  let y := -40
  let z := 2
  x + y + z

theorem cylinder_surface_area_correct : cylinder_surface_area = 2 := by
  sorry

end cylinder_surface_area_correct_l156_156866


namespace product_xyz_l156_156780

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end product_xyz_l156_156780


namespace foma_should_give_ierema_55_coins_l156_156137

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156137


namespace find_nm_2023_l156_156638

theorem find_nm_2023 (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : (n + m) ^ 2023 = -1 := by
  sorry

end find_nm_2023_l156_156638


namespace sqrt_18_mul_sqrt_32_eq_24_l156_156424
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l156_156424


namespace sin_120_eq_sqrt3_div_2_l156_156207

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l156_156207


namespace teacher_A_realizes_fish_l156_156840

variable (Teacher : Type) (has_fish : Teacher → Prop) (is_laughing : Teacher → Prop)
variables (A B C : Teacher)

-- Initial assumptions
axiom all_laughing : is_laughing A ∧ is_laughing B ∧ is_laughing C
axiom each_thinks_others_have_fish : (¬has_fish A ∧ has_fish B ∧ has_fish C) 
                                      ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
                                      ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C)

-- The logical conclusion
theorem teacher_A_realizes_fish : (∃ A B C : Teacher, 
  is_laughing A ∧ is_laughing B ∧ is_laughing C ∧
  ((¬has_fish A ∧ has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C))) →
  (has_fish A ∧ is_laughing B ∧ is_laughing C) :=
sorry -- proof not required.

end teacher_A_realizes_fish_l156_156840


namespace sum_of_primes_between_1_and_20_l156_156008

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l156_156008


namespace savings_percentage_correct_l156_156321

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

end savings_percentage_correct_l156_156321


namespace sphere_surface_area_increase_l156_156653

theorem sphere_surface_area_increase (V A : ℝ) (r : ℝ)
  (hV : V = (4/3) * π * r^3)
  (hA : A = 4 * π * r^2)
  : (∃ r', (V = 8 * ((4/3) * π * r'^3)) ∧ (∃ A', A' = 4 * A)) :=
by
  sorry

end sphere_surface_area_increase_l156_156653


namespace binomial_variance_eq_p_mul_one_sub_p_l156_156917

open ProbabilityTheory

variables {X : Type} [DiscreteRandomVariable X ℝ]
variables (p q : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : q = 1 - p)

noncomputable def binomialVariance : ℝ := p * (1 - p)

theorem binomial_variance_eq_p_mul_one_sub_p
  (hX : ∀ x : X, x = 0 ∨ x = 1)
  (hp : ∀ x : X, x = 1 → P(x) = p)
  (hq : ∀ x : X, x = 0 → P(x) = q):
  variance X = p * (1 - p) := sorry

end binomial_variance_eq_p_mul_one_sub_p_l156_156917


namespace root_sum_of_reciprocals_l156_156352

theorem root_sum_of_reciprocals {m : ℝ} :
  (∃ (a b : ℝ), a ≠ b ∧ (a + b) = 2 * (m + 1) ∧ (a * b) = m^2 + 2 ∧ (1/a + 1/b) = 1) →
  m = 2 :=
by sorry

end root_sum_of_reciprocals_l156_156352


namespace product_signs_l156_156059

theorem product_signs (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  ( 
    (((-a * b > 0) ∧ (a * c < 0) ∧ (b * d < 0) ∧ (c * d < 0)) ∨ 
    ((-a * b < 0) ∧ (a * c > 0) ∧ (b * d > 0) ∧ (c * d > 0))) ∨
    (((-a * b < 0) ∧ (a * c > 0) ∧ (b * d < 0) ∧ (c * d > 0)) ∨ 
    ((-a * b > 0) ∧ (a * c < 0) ∧ (b * d > 0) ∧ (c * d < 0))) 
  ) := 
sorry

end product_signs_l156_156059


namespace problem_l156_156578

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end problem_l156_156578


namespace integral_example_l156_156484

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem integral_example : ∫ x in -2..-1, f x = -2 * Real.log 2 :=
by
  sorry

end integral_example_l156_156484


namespace solve_trig_eq_l156_156679

noncomputable theory

open Real

theorem solve_trig_eq (x : ℝ) : (12 * sin x - 5 * cos x = 13) →
  ∃ (k : ℤ), x = (π / 2) + arctan (5 / 12) + 2 * k * π :=
by
s∞rry

end solve_trig_eq_l156_156679


namespace avg_production_last_5_days_l156_156715

theorem avg_production_last_5_days
  (avg_first_25_days : ℕ)
  (total_days : ℕ)
  (avg_entire_month : ℕ)
  (h1 : avg_first_25_days = 60)
  (h2 : total_days = 30)
  (h3 : avg_entire_month = 58) : 
  (total_days * avg_entire_month - 25 * avg_first_25_days) / 5 = 48 := 
by
  sorry

end avg_production_last_5_days_l156_156715


namespace value_of_f_at_3_l156_156511

noncomputable def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

theorem value_of_f_at_3 : f 3 = 155 := by
  sorry

end value_of_f_at_3_l156_156511


namespace number_of_books_per_continent_l156_156604

theorem number_of_books_per_continent (total_books : ℕ) (total_continents : ℕ) 
  (h1 : total_books = 488) (h2 : total_continents = 4) :
  (total_books / total_continents) = 122 :=
begin
  -- Lean part does not need the proof steps.
  sorry
end

end number_of_books_per_continent_l156_156604


namespace new_commission_rate_l156_156694

theorem new_commission_rate (C1 : ℝ) (slump : ℝ) : C2 : ℝ :=
  assume (h1 : C1 = 0.04)
  (h2 : slump = 0.20000000000000007),
  have h3: C2 = (C1 / (1 - slump)), from
  sorry,
  show C2 = 0.05, by sorry

end new_commission_rate_l156_156694


namespace range_of_m_l156_156064

noncomputable def problem_statement
  (x y m : ℝ) : Prop :=
  (x - 2 * y + 5 ≥ 0) ∧
  (3 - x ≥ 0) ∧
  (x + y ≥ 0) ∧
  (m > 0)

theorem range_of_m (x y m : ℝ) :
  problem_statement x y m →
  ((∀ x y, problem_statement x y m → x^2 + y^2 ≤ m^2) ↔ m ≥ 3 * Real.sqrt 2) :=
by 
  intro h
  sorry

end range_of_m_l156_156064


namespace number_of_girls_in_club_l156_156702

theorem number_of_girls_in_club (total : ℕ) (C1 : total = 36) 
    (C2 : ∀ (S : Finset ℕ), S.card = 33 → ∃ g b : ℕ, g + b = 33 ∧ g > b) 
    (C3 : ∃ (S : Finset ℕ), S.card = 31 ∧ ∃ g b : ℕ, g + b = 31 ∧ b > g) : 
    ∃ G : ℕ, G = 20 :=
by
  sorry

end number_of_girls_in_club_l156_156702


namespace function_identity_l156_156099

variable (f : ℕ+ → ℕ+)

theorem function_identity (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : ∀ n : ℕ+, f n = n := sorry

end function_identity_l156_156099


namespace fraction_shaded_is_one_tenth_l156_156460

theorem fraction_shaded_is_one_tenth :
  ∀ (A L S: ℕ), A = 300 → L = 5 → S = 2 → 
  ((15 * 20 = A) → (A / L = 60) → (60 / S = 30) → (30 / A = 1 / 10)) :=
by sorry

end fraction_shaded_is_one_tenth_l156_156460


namespace truck_covered_distance_l156_156978

theorem truck_covered_distance (t : ℝ) (d_bike : ℝ) (d_truck : ℝ) (v_bike : ℝ) (v_truck : ℝ) :
  t = 8 ∧ d_bike = 136 ∧ v_truck = v_bike + 3 ∧ d_bike = v_bike * t →
  d_truck = v_truck * t :=
by
  sorry

end truck_covered_distance_l156_156978


namespace sum_of_money_l156_156577

theorem sum_of_money (P R : ℝ) (h : (P * 2 * (R + 3) / 100) = (P * 2 * R / 100) + 300) : P = 5000 :=
by
    -- We are given that the sum of money put at 2 years SI rate is Rs. 300 more when rate is increased by 3%.
    sorry

end sum_of_money_l156_156577


namespace large_block_volume_l156_156713

theorem large_block_volume (W D L : ℝ) (h1 : W * D * L = 3) : 
  (2 * W) * (2 * D) * (3 * L) = 36 := 
by 
  sorry

end large_block_volume_l156_156713


namespace tan_double_angle_l156_156632

theorem tan_double_angle {x : ℝ} (h : Real.tan (π - x) = 3 / 4) : Real.tan (2 * x) = -24 / 7 :=
by 
  sorry

end tan_double_angle_l156_156632


namespace bananas_count_l156_156729

theorem bananas_count 
  (total_oranges : ℕ)
  (total_percentage_good : ℝ)
  (percentage_rotten_oranges : ℝ)
  (percentage_rotten_bananas : ℝ)
  (total_good_fruits_percentage : ℝ)
  (B : ℝ) :
  total_oranges = 600 →
  total_percentage_good = 0.85 →
  percentage_rotten_oranges = 0.15 →
  percentage_rotten_bananas = 0.03 →
  total_good_fruits_percentage = 0.898 →
  B = 400  :=
by
  intros h_oranges h_good_percentage h_rotten_oranges h_rotten_bananas h_good_fruits_percentage
  sorry

end bananas_count_l156_156729


namespace number_is_fraction_l156_156074

theorem number_is_fraction (x : ℝ) : (0.30 * x = 0.25 * 40) → (x = 100 / 3) :=
begin
  intro h,
  sorry
end

end number_is_fraction_l156_156074


namespace acute_angle_condition_l156_156902

theorem acute_angle_condition 
  (m : ℝ) 
  (a : ℝ × ℝ := (2,1))
  (b : ℝ × ℝ := (m,6)) 
  (dot_product := a.1 * b.1 + a.2 * b.2)
  (magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2))
  (magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2))
  (cos_angle := dot_product / (magnitude_a * magnitude_b))
  (acute_angle : cos_angle > 0) : -3 < m ∧ m ≠ 12 :=
sorry

end acute_angle_condition_l156_156902


namespace gardening_project_cost_l156_156873

def cost_rose_bushes (number_of_bushes: ℕ) (cost_per_bush: ℕ) : ℕ := number_of_bushes * cost_per_bush
def cost_gardener (hourly_rate: ℕ) (hours_per_day: ℕ) (days: ℕ) : ℕ := hourly_rate * hours_per_day * days
def cost_soil (cubic_feet: ℕ) (cost_per_cubic_foot: ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem gardening_project_cost :
  cost_rose_bushes 20 150 + cost_gardener 30 5 4 + cost_soil 100 5 = 4100 :=
by
  sorry

end gardening_project_cost_l156_156873


namespace selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l156_156836

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

end selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l156_156836


namespace first_mission_days_l156_156393

-- Definitions
variable (x : ℝ) (extended_first_mission : ℝ) (second_mission : ℝ) (total_mission_time : ℝ)

axiom h1 : extended_first_mission = 1.60 * x
axiom h2 : second_mission = 3
axiom h3 : total_mission_time = 11
axiom h4 : extended_first_mission + second_mission = total_mission_time

-- Theorem to prove
theorem first_mission_days : x = 5 :=
by
  sorry

end first_mission_days_l156_156393


namespace maximum_profit_l156_156587

noncomputable def L1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def L2 (x : ℝ) : ℝ := 2 * x

theorem maximum_profit :
  (∀ (x1 x2 : ℝ), x1 + x2 = 15 → L1 x1 + L2 x2 ≤ 45.6) := sorry

end maximum_profit_l156_156587


namespace pepperoni_crust_ratio_l156_156532

-- Define the conditions as Lean 4 statements
def L : ℕ := 50
def C : ℕ := 2 * L
def D : ℕ := 210
def S : ℕ := L + C + D
def S_E : ℕ := S / 4
def CR : ℕ := 600
def CH : ℕ := 400
def PizzaTotal (P : ℕ) : ℕ := CR + P + CH
def PizzaEats (P : ℕ) : ℕ := (PizzaTotal P) / 5
def JacksonEats : ℕ := 330

theorem pepperoni_crust_ratio (P : ℕ) (h1 : S_E + PizzaEats P = JacksonEats) : P / CR = 1 / 3 :=
by sorry

end pepperoni_crust_ratio_l156_156532


namespace total_money_left_l156_156876

theorem total_money_left (david_start john_start emily_start : ℝ) 
  (david_percent_left john_percent_spent emily_percent_spent : ℝ) : 
  (david_start = 3200) → 
  (david_percent_left = 0.65) → 
  (john_start = 2500) → 
  (john_percent_spent = 0.60) → 
  (emily_start = 4000) → 
  (emily_percent_spent = 0.45) → 
  let david_spent := david_start / (1 + david_percent_left)
  let david_remaining := david_start - david_spent
  let john_remaining := john_start * (1 - john_percent_spent)
  let emily_remaining := emily_start * (1 - emily_percent_spent)
  david_remaining + john_remaining + emily_remaining = 4460.61 :=
by
  sorry

end total_money_left_l156_156876


namespace first_reduction_percentage_l156_156698

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.70 = P * 0.525 ↔ x = 25 := by
  sorry

end first_reduction_percentage_l156_156698


namespace proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l156_156530

noncomputable def prob_boy_pass_all_rounds : ℚ :=
  (5/6) * (4/5) * (3/4) * (2/3)

noncomputable def prob_girl_pass_all_rounds : ℚ :=
  (4/5) * (3/4) * (2/3) * (1/2)

def prob_xi_distribution : (ℚ × ℚ × ℚ × ℚ × ℚ) :=
  (64/225, 96/225, 52/225, 12/225, 1/225)

def exp_xi : ℚ :=
  (0 * (64/225) + 1 * (96/225) + 2 * (52/225) + 3 * (12/225) + 4 * (1/225))

theorem proof_prob_boy_pass_all_rounds :
  prob_boy_pass_all_rounds = 1/3 :=
by
  sorry

theorem proof_prob_girl_pass_all_rounds :
  prob_girl_pass_all_rounds = 1/5 :=
by
  sorry

theorem proof_xi_distribution :
  prob_xi_distribution = (64/225, 96/225, 52/225, 12/225, 1/225) :=
by
  sorry

theorem proof_exp_xi :
  exp_xi = 16/15 :=
by
  sorry

end proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l156_156530


namespace diagonal_rectangle_l156_156294

theorem diagonal_rectangle (l w : ℝ) (hl : l = 20 * Real.sqrt 5) (hw : w = 10 * Real.sqrt 3) :
    Real.sqrt (l^2 + w^2) = 10 * Real.sqrt 23 :=
by
  sorry

end diagonal_rectangle_l156_156294


namespace max_boxes_fit_l156_156716

theorem max_boxes_fit 
  (L_large W_large H_large : ℕ) 
  (L_small W_small H_small : ℕ) 
  (h1 : L_large = 12) 
  (h2 : W_large = 14) 
  (h3 : H_large = 16) 
  (h4 : L_small = 3) 
  (h5 : W_small = 7) 
  (h6 : H_small = 2) 
  : ((L_large * W_large * H_large) / (L_small * W_small * H_small) = 64) :=
by
  sorry

end max_boxes_fit_l156_156716


namespace distance_between_street_lights_l156_156971

theorem distance_between_street_lights :
  ∀ (n : ℕ) (L : ℝ), n = 18 → L = 16.4 → 8 > 0 →
  (L / (8 : ℕ) = 2.05) :=
by
  intros n L h_n h_L h_nonzero
  sorry

end distance_between_street_lights_l156_156971


namespace units_digit_G1000_is_3_l156_156616

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 1

theorem units_digit_G1000_is_3 : (G 1000) % 10 = 3 := sorry

end units_digit_G1000_is_3_l156_156616


namespace student_finished_6_problems_in_class_l156_156463

theorem student_finished_6_problems_in_class (total_problems : ℕ) (x y : ℕ) (h1 : total_problems = 15) (h2 : 3 * y = 2 * x) (h3 : x + y = total_problems) : y = 6 :=
sorry

end student_finished_6_problems_in_class_l156_156463


namespace find_discount_l156_156328

noncomputable def children_ticket_cost : ℝ := 4.25
noncomputable def adult_ticket_cost : ℝ := children_ticket_cost + 3.25
noncomputable def total_cost_without_discount : ℝ := 2 * adult_ticket_cost + 4 * children_ticket_cost
noncomputable def total_spent : ℝ := 30
noncomputable def discount_received : ℝ := total_cost_without_discount - total_spent

theorem find_discount :
  discount_received = 2 := by
  sorry

end find_discount_l156_156328


namespace quadrant_classification_l156_156646

theorem quadrant_classification :
  ∀ (x y : ℝ), (4 * x - 3 * y = 24) → (|x| = |y|) → 
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  intros x y h_line h_eqdist
  sorry

end quadrant_classification_l156_156646


namespace min_value_expression_l156_156667

theorem min_value_expression (θ φ : ℝ) :
  ∃ (θ φ : ℝ), (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
sorry

end min_value_expression_l156_156667


namespace savings_after_20_days_l156_156367

-- Definitions based on conditions
def daily_earnings : ℕ := 80
def days_worked : ℕ := 20
def total_spent : ℕ := 1360

-- Prove the savings after 20 days
theorem savings_after_20_days : daily_earnings * days_worked - total_spent = 240 :=
by
  sorry

end savings_after_20_days_l156_156367


namespace triangle_is_equilateral_l156_156945

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)

-- Define a triangle's circumradius and inradius
structure TriangleProperties :=
  (circumradius : ℝ)
  (inradius : ℝ)
  (circumcenter_incenter_sq_distance : ℝ) -- OI^2 = circumradius^2 - 2*circumradius*inradius

noncomputable def circumcenter_incenter_coincide (T : Triangle) (P : TriangleProperties) : Prop :=
  P.circumcenter_incenter_sq_distance = 0

theorem triangle_is_equilateral
  (T : Triangle)
  (P : TriangleProperties)
  (hR : P.circumradius = 2 * P.inradius)
  (hOI : circumcenter_incenter_coincide T P) :
  ∃ (R r : ℝ), T = {A := 1 * r, B := 1 * r, C := 1 * r} :=
by sorry

end triangle_is_equilateral_l156_156945


namespace katya_attached_squares_perimeter_l156_156257

theorem katya_attached_squares_perimeter :
  let p1 := 100 -- Perimeter of the larger square
  let p2 := 40  -- Perimeter of the smaller square
  let s1 := p1 / 4 -- Side length of the larger square
  let s2 := p2 / 4 -- Side length of the smaller square
  let combined_perimeter_without_internal_sides := p1 + p2
  let actual_perimeter := combined_perimeter_without_internal_sides - 2 * s2
  actual_perimeter = 120 :=
by
  sorry

end katya_attached_squares_perimeter_l156_156257


namespace foma_should_give_ierema_l156_156158

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l156_156158


namespace find_expression_l156_156664

variables (x y z : ℝ) (ω : ℂ)

theorem find_expression
  (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : z ≠ -1)
  (h4 : ω^3 = 1) (h5 : ω ≠ 1)
  (h6 : (1 / (x + ω) + 1 / (y + ω) + 1 / (z + ω) = ω)) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) = -1 / 3 :=
sorry

end find_expression_l156_156664


namespace fraction_sum_l156_156738

theorem fraction_sum : (3 / 9 : ℚ) + (7 / 14 : ℚ) = 5 / 6 := by
  sorry

end fraction_sum_l156_156738


namespace weight_conversion_l156_156312

theorem weight_conversion (a b : ℝ) (conversion_rate : ℝ) : a = 3600 → b = 600 → conversion_rate = 1000 → (a - b) / conversion_rate = 3 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end weight_conversion_l156_156312


namespace sin_cos_sum_l156_156169

theorem sin_cos_sum : (Real.sin (π/6) + Real.cos (π/3) = 1) :=
by
  have h1 : Real.sin (π / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (π / 3) = 1 / 2 := by sorry
  calc 
    Real.sin (π / 6) + Real.cos (π / 3)
        = 1 / 2 + 1 / 2 : by rw [h1, h2]
    ... = 1 : by norm_num

end sin_cos_sum_l156_156169


namespace part_a_7_pieces_l156_156583

theorem part_a_7_pieces (grid : Fin 4 × Fin 4 → Prop) (h : ∀ i j, ∃ n, grid (i, j) → n < 7)
  (hnoTwoInSameCell : ∀ (i₁ i₂ : Fin 4) (j₁ j₂ : Fin 4), (i₁, j₁) ≠ (i₂, j₂) → grid (i₁, j₁) ≠ grid (i₂, j₂))
  : ∀ (rowsRemoved colsRemoved : Finset (Fin 4)), rowsRemoved.card = 2 → colsRemoved.card = 2
    → ∃ i j, ¬ grid (i, j) := by sorry

end part_a_7_pieces_l156_156583


namespace solve_equation_125_eq_5_25_exp_x_min_2_l156_156551

theorem solve_equation_125_eq_5_25_exp_x_min_2 :
    ∃ x : ℝ, 125 = 5 * (25 : ℝ)^(x - 2) ∧ x = 3 := 
by
  sorry

end solve_equation_125_eq_5_25_exp_x_min_2_l156_156551


namespace integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l156_156621

-- Proof problem 1
theorem integers_abs_no_greater_than_2 :
    {n : ℤ | |n| ≤ 2} = {-2, -1, 0, 1, 2} :=
by {
  sorry
}

-- Proof problem 2
theorem pos_div_by_3_less_than_10 :
    {n : ℕ | n > 0 ∧ n % 3 = 0 ∧ n < 10} = {3, 6, 9} :=
by {
  sorry
}

-- Proof problem 3
theorem non_neg_int_less_than_5 :
    {n : ℤ | n = |n| ∧ n < 5} = {0, 1, 2, 3, 4} :=
by {
  sorry
}

-- Proof problem 4
theorem sum_eq_6_in_nat :
    {p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0} = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)} :=
by {
  sorry
}

-- Proof problem 5
theorem expressing_sequence:
    {-3, -1, 1, 3, 5} = {x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3} :=
by {
  sorry
}

end integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l156_156621


namespace find_vector_c_l156_156055

-- Definitions of the given vectors
def vector_a : ℝ × ℝ := (3, -1)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2)

-- The goal is to prove that vector_c = (5, 0)
theorem find_vector_c : vector_c = (5, 0) :=
by
  -- proof steps would go here
  sorry

end find_vector_c_l156_156055


namespace divisibility_by_7_l156_156548

theorem divisibility_by_7 (n : ℕ) (h : 0 < n) : 7 ∣ (3 ^ (2 * n + 2) - 2 ^ (n + 1)) :=
sorry

end divisibility_by_7_l156_156548


namespace inequality_solution_l156_156480

theorem inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := sorry

end inequality_solution_l156_156480


namespace fomagive_55_l156_156146

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l156_156146


namespace projection_matrix_solution_l156_156744

theorem projection_matrix_solution 
  (a c : ℚ) 
  (P : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 18/45], ![c, 27/45]])
  (hP : P * P = P) :
  (a, c) = (9/25, 12/25) :=
by
  sorry

end projection_matrix_solution_l156_156744


namespace average_of_original_set_l156_156946

-- Average of 8 numbers is some value A and the average of the new set where each number is 
-- multiplied by 8 is 168. We need to show that the original average A is 21.

theorem average_of_original_set (A : ℝ) (h1 : (64 * A) / 8 = 168) : A = 21 :=
by {
  -- This is the theorem statement, we add the proof next
  sorry -- proof placeholder
}

end average_of_original_set_l156_156946


namespace other_candidate_valid_votes_l156_156247

noncomputable def validVotes (totalVotes invalidPct : ℝ) : ℝ :=
  totalVotes * (1 - invalidPct)

noncomputable def otherCandidateVotes (validVotes oneCandidatePct : ℝ) : ℝ :=
  validVotes * (1 - oneCandidatePct)

theorem other_candidate_valid_votes :
  let totalVotes := 7500
  let invalidPct := 0.20
  let oneCandidatePct := 0.55
  validVotes totalVotes invalidPct = 6000 ∧
  otherCandidateVotes (validVotes totalVotes invalidPct) oneCandidatePct = 2700 :=
by
  sorry

end other_candidate_valid_votes_l156_156247


namespace unpainted_cubes_count_l156_156313

noncomputable def num_unpainted_cubes : ℕ :=
  let total_cubes := 216
  let painted_on_faces := 16 * 6 / 1  -- Central 4x4 areas on each face
  let shared_edges := ((4 * 4) * 6) / 2  -- Shared edges among faces
  let shared_corners := (4 * 6) / 3  -- Shared corners among faces
  let total_painted := painted_on_faces - shared_edges - shared_corners
  total_cubes - total_painted

theorem unpainted_cubes_count : num_unpainted_cubes = 160 := sorry

end unpainted_cubes_count_l156_156313


namespace find_value_of_expression_l156_156233

theorem find_value_of_expression
  (a b c m : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = m)
  (h5 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 := 
sorry

end find_value_of_expression_l156_156233


namespace number_of_dogs_with_both_tags_and_collars_l156_156469

-- Defining the problem
def total_dogs : ℕ := 80
def dogs_with_tags : ℕ := 45
def dogs_with_collars : ℕ := 40
def dogs_with_neither : ℕ := 1

-- Statement: Prove the number of dogs with both tags and collars
theorem number_of_dogs_with_both_tags_and_collars : 
  (dogs_with_tags + dogs_with_collars - total_dogs + dogs_with_neither) = 6 :=
by
  sorry

end number_of_dogs_with_both_tags_and_collars_l156_156469


namespace max_cos_a_l156_156101

theorem max_cos_a (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 :=
by
  -- Proof goes here
  sorry

end max_cos_a_l156_156101


namespace percentage_of_x_l156_156298

variable (x : ℝ)

theorem percentage_of_x (x : ℝ) : ((40 / 100) * (50 / 100) * x) = (20 / 100) * x := by
  sorry

end percentage_of_x_l156_156298


namespace spadesuit_evaluation_l156_156617

def spadesuit (a b : ℤ) : ℤ := Int.natAbs (a - b)

theorem spadesuit_evaluation :
  spadesuit 5 (spadesuit 3 9) = 1 := 
by 
  sorry

end spadesuit_evaluation_l156_156617


namespace plane_intersect_probability_l156_156865

-- Define the vertices of the rectangular prism
def vertices : List (ℝ × ℝ × ℝ) := 
  [(0,0,0), (2,0,0), (2,2,0), (0,2,0), 
   (0,0,1), (2,0,1), (2,2,1), (0,2,1)]

-- Calculate total number of ways to choose 3 vertices out of 8
def total_ways : ℕ := Nat.choose 8 3

-- Calculate the number of planes that do not intersect the interior of the prism
def non_intersecting_planes : ℕ := 6 * Nat.choose 4 3

-- Calculate the probability as a fraction
def probability_of_intersecting (total non_intersecting : ℕ) : ℚ :=
  1 - (non_intersecting : ℚ) / (total : ℚ)

-- The main theorem to state the probability is 4/7
theorem plane_intersect_probability : 
  probability_of_intersecting total_ways non_intersecting_planes = 4 / 7 := 
  by
    -- Skipping the proof
    sorry

end plane_intersect_probability_l156_156865


namespace greatest_value_2q_sub_r_l156_156123

theorem greatest_value_2q_sub_r : 
  ∃ (q r : ℕ), 965 = 22 * q + r ∧ 2 * q - r = 67 := 
by 
  sorry

end greatest_value_2q_sub_r_l156_156123


namespace select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l156_156386

variable (n m : ℕ) -- n for males, m for females
variable (mc fc : ℕ) -- mc for male captain, fc for female captain

def num_ways_3_males_2_females : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 4 2)

def num_ways_at_least_1_captain : ℕ :=
  (2 * (Nat.choose 8 4)) + (Nat.choose 8 3)

def num_ways_at_least_1_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 6 5)

def num_ways_both_captain_and_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 8 5) - (Nat.choose 5 4)

theorem select_3_males_2_females : num_ways_3_males_2_females = 120 := by
  sorry
  
theorem select_at_least_1_captain : num_ways_at_least_1_captain = 196 := by
  sorry
  
theorem select_at_least_1_female : num_ways_at_least_1_female = 246 := by
  sorry
  
theorem select_both_captain_and_female : num_ways_both_captain_and_female = 191 := by
  sorry

end select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l156_156386


namespace total_turns_to_fill_drum_l156_156605

variable (Q : ℝ) -- Capacity of bucket Q
variable (turnsP : ℝ) (P_capacity : ℝ) (R_capacity : ℝ) (drum_capacity : ℝ)

-- Condition: It takes 60 turns for bucket P to fill the empty drum
def bucketP_fills_drum_in_60_turns : Prop := turnsP = 60 ∧ P_capacity = 3 * Q ∧ drum_capacity = 60 * P_capacity

-- Condition: Bucket P has thrice the capacity as bucket Q
def bucketP_capacity : Prop := P_capacity = 3 * Q

-- Condition: Bucket R has half the capacity as bucket Q
def bucketR_capacity : Prop := R_capacity = Q / 2

-- Computation: Using all three buckets together, find the combined capacity filled in one turn
def combined_capacity_per_turn : ℝ := P_capacity + Q + R_capacity

-- Main Theorem: It takes 40 turns to fill the drum using all three buckets together
theorem total_turns_to_fill_drum
  (h1 : bucketP_fills_drum_in_60_turns Q turnsP P_capacity drum_capacity)
  (h2 : bucketP_capacity Q P_capacity)
  (h3 : bucketR_capacity Q R_capacity) :
  drum_capacity / combined_capacity_per_turn Q P_capacity (Q / 2) = 40 :=
by
  sorry

end total_turns_to_fill_drum_l156_156605


namespace S_2012_value_l156_156791

-- Define the first term of the arithmetic sequence
def a1 : ℤ := -2012

-- Define the common difference
def d : ℤ := 2

-- Define the sequence a_n
def a (n : ℕ) : ℤ := a1 + d * (n - 1)

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℤ := n * (a1 + a n) / 2

-- Formalize the given problem as a Lean statement
theorem S_2012_value : S 2012 = -2012 :=
by 
{
  -- The proof is omitted as requested
  sorry
}

end S_2012_value_l156_156791


namespace range_of_T_l156_156890

open Real

theorem range_of_T (x y z : ℝ) (h : x^2 + 2 * y^2 + 3 * z^2 = 4) : 
    - (2 * sqrt 6) / 3 ≤ x * y + y * z ∧ x * y + y * z ≤ (2 * sqrt 6) / 3 := 
by 
    sorry

end range_of_T_l156_156890


namespace painted_cubes_on_two_faces_l156_156848

theorem painted_cubes_on_two_faces (n : ℕ) (painted_faces_all : Prop) (equal_smaller_cubes : n = 27) : ∃ k : ℕ, k = 12 :=
by
  -- We only need the statement, not the proof
  sorry

end painted_cubes_on_two_faces_l156_156848


namespace molecular_weight_one_mole_l156_156449

theorem molecular_weight_one_mole
  (molecular_weight_7_moles : ℝ)
  (mole_count : ℝ)
  (h : molecular_weight_7_moles = 126)
  (k : mole_count = 7)
  : molecular_weight_7_moles / mole_count = 18 := 
sorry

end molecular_weight_one_mole_l156_156449


namespace inequality_l156_156359

variable (a b m : ℝ)

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < m) (h4 : a < b) :
  a / b < (a + m) / (b + m) :=
by
  sorry

end inequality_l156_156359


namespace find_triangle_with_properties_l156_156625

-- Define the angles forming an arithmetic progression
def angles_arithmetic_progression (α β γ : ℝ) : Prop :=
  β - α = γ - β

-- Define the sides forming an arithmetic progression
def sides_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the sides forming a geometric progression
def sides_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Define the sum of angles in a triangle
def sum_of_angles (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- The problem statement:
theorem find_triangle_with_properties 
    (α β γ a b c : ℝ)
    (h1 : angles_arithmetic_progression α β γ)
    (h2 : sum_of_angles α β γ)
    (h3 : sides_arithmetic_progression a b c ∨ sides_geometric_progression a b c) :
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by 
  sorry

end find_triangle_with_properties_l156_156625


namespace bryan_travel_hours_per_year_l156_156923

-- Definitions based on the conditions
def minutes_walk_to_bus_station := 5
def minutes_ride_bus := 20
def minutes_walk_to_job := 5
def days_per_year := 365

-- Total time for one-way travel in minutes
def one_way_travel_minutes := minutes_walk_to_bus_station + minutes_ride_bus + minutes_walk_to_job

-- Total daily travel time in minutes
def daily_travel_minutes := one_way_travel_minutes * 2

-- Convert daily travel time from minutes to hours
def daily_travel_hours := daily_travel_minutes / 60

-- Total yearly travel time in hours
def yearly_travel_hours := daily_travel_hours * days_per_year

-- The theorem to prove
theorem bryan_travel_hours_per_year : yearly_travel_hours = 365 :=
by {
  -- The preliminary arithmetic is not the core of the theorem
  sorry
}

end bryan_travel_hours_per_year_l156_156923


namespace probability_two_balls_red_l156_156314

variables (total_balls red_balls blue_balls green_balls picked_balls : ℕ)

def probability_of_both_red
  (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2) : ℚ :=
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem probability_two_balls_red (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2)
  (h_prob : probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28) : 
  probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28 := 
sorry

end probability_two_balls_red_l156_156314


namespace book_price_l156_156630

theorem book_price (n p : ℕ) (h : n * p = 104) (hn : 10 < n ∧ n < 60) : p = 2 ∨ p = 4 ∨ p = 8 :=
sorry

end book_price_l156_156630


namespace sum_primes_upto_20_l156_156009

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l156_156009


namespace other_candidate_valid_votes_l156_156246

-- Define the conditions of the problem
theorem other_candidate_valid_votes (total_votes : ℕ) (invalid_percent : ℝ) (candidate_percent : ℝ) (other_percent : ℝ) :
  total_votes = 7500 → invalid_percent = 20 → candidate_percent = 55 → other_percent = 45 →
  let valid_votes := (1 - invalid_percent / 100) * total_votes in
  let other_candidate_votes := (other_percent / 100) * valid_votes in
  other_candidate_votes = 2700 :=
begin
  intros,
  let valid_votes := (1 - invalid_percent / 100) * total_votes,
  let other_candidate_votes := (other_percent / 100) * valid_votes,
  have h_valid := valid_votes = 0.8 * total_votes,
  have h_votes := other_candidate_votes = 0.45 * valid_votes,
  simp at *,
  sorry
end

end other_candidate_valid_votes_l156_156246


namespace fomagive_55_l156_156148

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l156_156148


namespace unique_divisor_of_2_pow_n_minus_1_l156_156347

theorem unique_divisor_of_2_pow_n_minus_1 : ∀ (n : ℕ), n ≥ 1 → n ∣ (2^n - 1) → n = 1 := 
by
  intro n h1 h2
  sorry

end unique_divisor_of_2_pow_n_minus_1_l156_156347


namespace evaluate_expression_l156_156558

theorem evaluate_expression :
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 :=
by 
  sorry

end evaluate_expression_l156_156558


namespace ratio_second_to_first_l156_156935

-- Condition 1: The first bell takes 50 pounds of bronze
def first_bell_weight : ℕ := 50

-- Condition 2: The second bell is a certain size compared to the first bell
variable (x : ℕ) -- the ratio of the size of the second bell to the first bell
def second_bell_weight := first_bell_weight * x

-- Condition 3: The third bell is four times the size of the second bell
def third_bell_weight := 4 * second_bell_weight x

-- Condition 4: The total weight of bronze required is 550 pounds
def total_weight : ℕ := 550

-- Define the proof problem
theorem ratio_second_to_first (x : ℕ) (h : 50 + 50 * x + 200 * x = 550) : x = 2 :=
by
  sorry

end ratio_second_to_first_l156_156935


namespace perfect_cube_prime_l156_156104

theorem perfect_cube_prime (p : ℕ) (h_prime : Nat.Prime p) (h_cube : ∃ x : ℕ, 2 * p + 1 = x^3) : 
  2 * p + 1 = 27 ∧ p = 13 :=
by
  sorry

end perfect_cube_prime_l156_156104


namespace bert_puzzle_days_l156_156994

noncomputable def words_per_pencil : ℕ := 1050
noncomputable def words_per_puzzle : ℕ := 75

theorem bert_puzzle_days : words_per_pencil / words_per_puzzle = 14 := by
  sorry

end bert_puzzle_days_l156_156994


namespace max_xyz_squared_l156_156231

theorem max_xyz_squared 
  (x y z : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h1 : x * y * z = (14 - x) * (14 - y) * (14 - z)) 
  (h2 : x + y + z < 28) : 
  x^2 + y^2 + z^2 ≤ 219 :=
sorry

end max_xyz_squared_l156_156231


namespace problem_statement_l156_156930

variable {x y z : ℝ}

theorem problem_statement
  (h : x^2 + y^2 + z^2 + 9 = 4 * (x + y + z)) :
  x^4 + y^4 + z^4 + 16 * (x^2 + y^2 + z^2) ≥ 8 * (x^3 + y^3 + z^3) + 27 :=
by
  sorry

end problem_statement_l156_156930


namespace unique_solution_l156_156751

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem unique_solution (n : ℕ) :
  (0 < n ∧ is_prime (n + 1) ∧ is_prime (n + 3) ∧
   is_prime (n + 7) ∧ is_prime (n + 9) ∧
   is_prime (n + 13) ∧ is_prime (n + 15)) ↔ n = 4 :=
by
  sorry

end unique_solution_l156_156751


namespace no_linear_factor_with_integer_coefficients_l156_156335

def expression (x y z : ℤ) : ℤ :=
  x^2 - y^2 - z^2 + 3 * y * z + x + 2 * y - z

theorem no_linear_factor_with_integer_coefficients:
  ¬ ∃ (a b c d : ℤ), a ≠ 0 ∧ 
                      ∀ (x y z : ℤ), 
                        expression x y z = a * x + b * y + c * z + d := by
  sorry

end no_linear_factor_with_integer_coefficients_l156_156335


namespace number_of_rods_in_one_mile_l156_156892

theorem number_of_rods_in_one_mile (miles_to_furlongs : 1 = 10 * 1)
  (furlongs_to_rods : 1 = 50 * 1) : 1 = 500 * 1 :=
by {
  sorry
}

end number_of_rods_in_one_mile_l156_156892


namespace initial_sand_amount_l156_156326

theorem initial_sand_amount (lost_sand : ℝ) (arrived_sand : ℝ)
  (h1 : lost_sand = 2.4) (h2 : arrived_sand = 1.7) :
  lost_sand + arrived_sand = 4.1 :=
by
  rw [h1, h2]
  norm_num

end initial_sand_amount_l156_156326


namespace find_number_l156_156072

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l156_156072


namespace zero_in_interval_l156_156743

noncomputable def f (x : ℝ) := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end zero_in_interval_l156_156743


namespace number_of_digimon_packs_bought_l156_156804

noncomputable def cost_per_digimon_pack : ℝ := 4.45
noncomputable def cost_of_baseball_deck : ℝ := 6.06
noncomputable def total_spent : ℝ := 23.86

theorem number_of_digimon_packs_bought : 
  ∃ (D : ℕ), cost_per_digimon_pack * D + cost_of_baseball_deck = total_spent ∧ D = 4 := 
by
  use 4
  split
  · norm_num; exact ((4.45 * 4) + 6.06 = 23.86)
  · exact rfl

end number_of_digimon_packs_bought_l156_156804


namespace problem_statement_l156_156790

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem problem_statement :
  let l := { p : ℝ × ℝ | p.1 - p.2 - 2 = 0 }
  let C := { p : ℝ × ℝ | ∃ θ : ℝ, p = (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ) }
  let A := (-4, -6)
  let B := (4, 2)
  let P := (-2 * Real.sqrt 3, 2)
  let d := (|2 * Real.sqrt 3 * Real.cos (5 * Real.pi / 6) - 2|) / Real.sqrt 2
  distance A B = 8 * Real.sqrt 2 ∧ d = 3 * Real.sqrt 2 ∧
  let max_area := 1 / 2 * 8 * Real.sqrt 2 * 3 * Real.sqrt 2
  P ∈ C ∧ max_area = 24 := by
sorry

end problem_statement_l156_156790


namespace one_circle_equiv_three_squares_l156_156919

-- Define the weights of circles and squares symbolically
variables {w_circle w_square : ℝ}

-- Equations based on the conditions in the problem
-- 3 circles balance 5 squares
axiom eq1 : 3 * w_circle = 5 * w_square

-- 2 circles balance 3 squares and 1 circle
axiom eq2 : 2 * w_circle = 3 * w_square + w_circle

-- We need to prove that 1 circle is equivalent to 3 squares
theorem one_circle_equiv_three_squares : w_circle = 3 * w_square := 
by sorry

end one_circle_equiv_three_squares_l156_156919


namespace trigonometric_identity_l156_156373

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (π / 2 + α) * Real.cos (π + α) = -1 / 5 :=
by
  -- The proof will be skipped but the statement should be correct.
  sorry

end trigonometric_identity_l156_156373


namespace value_of_a3_l156_156891

theorem value_of_a3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (a : ℝ) (h₀ : (1 + x) * (a - x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7) 
(h₁ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) : 
a₃ = -5 :=
sorry

end value_of_a3_l156_156891


namespace intersection_P_Q_l156_156768

def P := {x : ℝ | 1 < x ∧ x < 3}
def Q := {x : ℝ | 2 < x}

theorem intersection_P_Q :
  P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := sorry

end intersection_P_Q_l156_156768


namespace brick_piles_l156_156286

theorem brick_piles (x y z : ℤ) :
  2 * (x - 100) = y + 100 ∧
  x + z = 6 * (y - z) →
  x = 170 ∧ y = 40 :=
by
  sorry

end brick_piles_l156_156286


namespace evaluate_x3_minus_y3_l156_156764

theorem evaluate_x3_minus_y3 (x y : ℤ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x^3 - y^3 = -448 :=
by
  sorry

end evaluate_x3_minus_y3_l156_156764


namespace no_solution_inequality_l156_156063

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  sorry

end no_solution_inequality_l156_156063


namespace regular_pentagon_diagonal_square_l156_156096

variable (a d : ℝ)
def is_regular_pentagon (a d : ℝ) : Prop :=
d ^ 2 = a ^ 2 + a * d

theorem regular_pentagon_diagonal_square :
  is_regular_pentagon a d :=
sorry

end regular_pentagon_diagonal_square_l156_156096


namespace factorize_x_squared_minus_25_l156_156049

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l156_156049


namespace rightmost_three_digits_of_7_pow_1994_l156_156564

theorem rightmost_three_digits_of_7_pow_1994 :
  (7 ^ 1994) % 800 = 49 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1994_l156_156564


namespace sqrt_mul_simplify_l156_156429

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l156_156429


namespace birdseed_mixture_l156_156016

theorem birdseed_mixture (x : ℝ) (h1 : 0.40 * x + 0.65 * (100 - x) = 50) : x = 60 :=
by
  sorry

end birdseed_mixture_l156_156016


namespace combined_percentage_grade4_l156_156130

-- Definitions based on the given conditions
def Pinegrove_total_students : ℕ := 120
def Maplewood_total_students : ℕ := 180

def Pinegrove_grade4_percentage : ℕ := 10
def Maplewood_grade4_percentage : ℕ := 20

theorem combined_percentage_grade4 :
  let combined_total_students := Pinegrove_total_students + Maplewood_total_students
  let Pinegrove_grade4_students := Pinegrove_grade4_percentage * Pinegrove_total_students / 100
  let Maplewood_grade4_students := Maplewood_grade4_percentage * Maplewood_total_students / 100 
  let combined_grade4_students := Pinegrove_grade4_students + Maplewood_grade4_students
  (combined_grade4_students * 100 / combined_total_students) = 16 := by
  sorry

end combined_percentage_grade4_l156_156130


namespace g_7_eq_98_l156_156216

noncomputable def g : ℕ → ℝ := sorry

axiom g_0 : g 0 = 0
axiom g_1 : g 1 = 2
axiom functional_equation (m n : ℕ) (h : m ≥ n) : g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

theorem g_7_eq_98 : g 7 = 98 :=
sorry

end g_7_eq_98_l156_156216


namespace trigonometric_identity_l156_156582

theorem trigonometric_identity :
  (Real.sqrt 3 / Real.cos (10 * Real.pi / 180) - 1 / Real.sin (170 * Real.pi / 180) = -4) :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_l156_156582


namespace plant_arrangement_count_l156_156544

-- Define the count of identical plants
def basil_count := 3
def aloe_count := 2

-- Define the count of identical lamps in each color
def white_lamp_count := 3
def red_lamp_count := 3

-- Define the total ways to arrange the plants under the lamps.
def arrangement_ways := 128

-- Formalize the problem statement proving the arrangements count
theorem plant_arrangement_count :
  (∃ f : Fin (basil_count + aloe_count) → Fin (white_lamp_count + red_lamp_count), True) ↔
  arrangement_ways = 128 :=
sorry

end plant_arrangement_count_l156_156544


namespace smallest_N_l156_156097

theorem smallest_N (p q r s t u : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hu : 0 < u)
  (h_sum : p + q + r + s + t + u = 2023) :
  ∃ N : ℕ, N = max (max (max (max (p + q) (q + r)) (r + s)) (s + t)) (t + u) ∧ N = 810 :=
sorry

end smallest_N_l156_156097


namespace product_of_all_possible_values_l156_156650

theorem product_of_all_possible_values (x : ℝ) :
  (|16 / x + 4| = 3) → ((x = -16 ∨ x = -16 / 7) →
  (x_1 = -16 ∧ x_2 = -16 / 7) →
  (x_1 * x_2 = 256 / 7)) :=
sorry

end product_of_all_possible_values_l156_156650


namespace fraction_transformed_l156_156651

variables (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab_pos : a * b > 0)

noncomputable def frac_orig := (a + 2 * b) / (2 * a * b)
noncomputable def frac_new := (3 * a + 2 * 3 * b) / (2 * 3 * a * 3 * b)

theorem fraction_transformed :
  frac_new a b = (1 / 3) * frac_orig a b :=
sorry

end fraction_transformed_l156_156651


namespace diagonal_crosses_768_unit_cubes_l156_156592

-- Defining the dimensions of the rectangular prism
def a : ℕ := 150
def b : ℕ := 324
def c : ℕ := 375

-- Computing the gcd values
def gcd_ab : ℕ := Nat.gcd a b
def gcd_ac : ℕ := Nat.gcd a c
def gcd_bc : ℕ := Nat.gcd b c
def gcd_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- Using the formula to compute the number of unit cubes the diagonal intersects
def num_unit_cubes : ℕ := a + b + c - gcd_ab - gcd_ac - gcd_bc + gcd_abc

-- Stating the theorem to prove
theorem diagonal_crosses_768_unit_cubes : num_unit_cubes = 768 := by
  sorry

end diagonal_crosses_768_unit_cubes_l156_156592


namespace evaluate_expression_l156_156337

noncomputable def lg (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression :
  lg 5 * lg 50 - lg 2 * lg 20 - lg 625 = -2 :=
by
  sorry

end evaluate_expression_l156_156337


namespace one_number_greater_than_one_l156_156787

theorem one_number_greater_than_one 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > (1 / a) + (1 / b) + (1 / c)) 
  : (1 < a ∨ 1 < b ∨ 1 < c) ∧ ¬(1 < a ∧ 1 < b ∧ 1 < c) :=
by
  sorry

end one_number_greater_than_one_l156_156787


namespace find_AX_l156_156250

variable (A B X C : Point)
variable (AB AC BC AX XB : ℝ)
variable (angleACX angleXCB : Angle)
variable (eqAngle : angleACX = angleXCB)

axiom length_AB : AB = 80
axiom length_AC : AC = 36
axiom length_BC : BC = 72

theorem find_AX (AB AC BC AX XB : ℝ) (angleACX angleXCB : Angle)
  (eqAngle : angleACX = angleXCB)
  (h1 : AB = 80)
  (h2 : AC = 36)
  (h3 : BC = 72) : AX = 80 / 3 :=
by
  sorry

end find_AX_l156_156250


namespace speed_of_man_in_still_water_l156_156576

theorem speed_of_man_in_still_water
  (v_m v_s : ℝ)
  (h1 : v_m + v_s = 4)
  (h2 : v_m - v_s = 2) :
  v_m = 3 := 
by sorry

end speed_of_man_in_still_water_l156_156576


namespace Bills_age_proof_l156_156872

variable {b t : ℚ}

theorem Bills_age_proof (h1 : b = 4 * t / 3) (h2 : b + 30 = 9 * (t + 30) / 8) : b = 24 := by 
  sorry

end Bills_age_proof_l156_156872


namespace problem_l156_156356

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f (-x) = f x)  -- f is an even function
variable (h_mono : ∀ x y : ℝ, 0 < x → x < y → f y < f x)  -- f is monotonically decreasing on (0, +∞)

theorem problem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l156_156356


namespace sqrt_18_mul_sqrt_32_eq_24_l156_156425
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l156_156425


namespace minimum_time_to_finish_route_l156_156953

-- Step (a): Defining conditions and necessary terms
def points : Nat := 12
def segments_between_points : ℕ := 17
def time_per_segment : ℕ := 10 -- in minutes
def total_time_in_minutes : ℕ := segments_between_points * time_per_segment -- Total time in minutes

-- Step (c): Proving the question == answer given conditions
theorem minimum_time_to_finish_route (K : ℕ) : K = 4 :=
by
  have time_in_hours : ℕ := total_time_in_minutes / 60
  have minimum_time : ℕ := 4
  sorry -- proof needed

end minimum_time_to_finish_route_l156_156953


namespace eighteenth_entry_of_sequence_l156_156221

def r_7 (n : ℕ) : ℕ := n % 7

theorem eighteenth_entry_of_sequence : ∃ n : ℕ, (r_7 (4 * n) ≤ 3) ∧ (∀ m : ℕ, m < 18 → (r_7 (4 * m) ≤ 3) → m ≠ n) ∧ n = 30 := 
by 
  sorry

end eighteenth_entry_of_sequence_l156_156221


namespace g_x_minus_3_l156_156362

def g (x : ℝ) : ℝ := x^2

theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6 * x + 9 :=
by
  -- This is where the proof would go
  sorry

end g_x_minus_3_l156_156362


namespace tangent_line_through_point_l156_156896

theorem tangent_line_through_point (a : ℝ) : 
  ∃ l : ℝ → ℝ, 
    (∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → y = a) ∧ 
    (∀ x y : ℝ, y = l x → (x - 1)^2 + y^2 = 4) → 
    a = 0 :=
by
  sorry

end tangent_line_through_point_l156_156896


namespace ellipse_equation_from_hyperbola_l156_156645

theorem ellipse_equation_from_hyperbola :
  (∃ (a b : ℝ), ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) →
  (x^2 / 16 + y^2 / 12 = 1)) :=
by
  sorry

end ellipse_equation_from_hyperbola_l156_156645


namespace John_gave_the_store_20_dollars_l156_156802

def slurpee_cost : ℕ := 2
def change_received : ℕ := 8
def slurpees_bought : ℕ := 6
def total_money_given : ℕ := slurpee_cost * slurpees_bought + change_received

theorem John_gave_the_store_20_dollars : total_money_given = 20 := 
by 
  sorry

end John_gave_the_store_20_dollars_l156_156802


namespace matrix_B_power_103_l156_156396

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_B_power_103 :
  B ^ 103 = B :=
by
  sorry

end matrix_B_power_103_l156_156396


namespace perfect_square_iff_all_perfect_squares_l156_156113

theorem perfect_square_iff_all_perfect_squares
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0):
  (∃ k : ℕ, (xy + 1) * (yz + 1) * (zx + 1) = k^2) ↔
  (∃ a b c : ℕ, xy + 1 = a^2 ∧ yz + 1 = b^2 ∧ zx + 1 = c^2) := 
sorry

end perfect_square_iff_all_perfect_squares_l156_156113


namespace trig_identity_proof_l156_156608

theorem trig_identity_proof :
  Real.sin (30 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) + 
  Real.sin (60 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) =
  Real.sqrt 2 / 2 := 
by
  sorry

end trig_identity_proof_l156_156608


namespace foma_should_give_ierema_55_coins_l156_156138

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156138


namespace solution_set_a_eq_half_l156_156833

theorem solution_set_a_eq_half (a : ℝ) : (∀ x : ℝ, (ax / (x - 1) < 1 ↔ (x < 1 ∨ x > 2))) → a = 1 / 2 :=
by
sorry

end solution_set_a_eq_half_l156_156833


namespace age_difference_l156_156102

-- Defining the age variables as fractions
variables (x y : ℚ)

-- Given conditions
axiom ratio1 : 2 * x / y = 2 / y
axiom ratio2 : (5 * x + 20) / (y + 20) = 8 / 3

-- The main theorem to prove the difference between Mahesh's and Suresh's ages.
theorem age_difference : 5 * x - y = (125 / 8) := sorry

end age_difference_l156_156102


namespace foma_should_give_ierema_l156_156161

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l156_156161


namespace A_and_C_work_together_in_2_hours_l156_156021

theorem A_and_C_work_together_in_2_hours
  (A_rate : ℚ)
  (B_rate : ℚ)
  (C_rate : ℚ)
  (A_4_hours : A_rate = 1 / 4)
  (B_12_hours : B_rate = 1 / 12)
  (B_and_C_3_hours : B_rate + C_rate = 1 / 3) :
  (A_rate + C_rate = 1 / 2) :=
by
  sorry

end A_and_C_work_together_in_2_hours_l156_156021


namespace product_xyz_l156_156781

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end product_xyz_l156_156781


namespace solve_trig_eq_l156_156119

open Real -- Open real number structure

theorem solve_trig_eq (x : ℝ) :
  (sin x)^2 + (sin (2 * x))^2 + (sin (3 * x))^2 = 2 ↔ 
  (∃ n : ℤ, x = π / 4 + (π * n) / 2)
  ∨ (∃ n : ℤ, x = π / 2 + π * n)
  ∨ (∃ n : ℤ, x = π / 6 + π * n ∨ x = -π / 6 + π * n) := by sorry

end solve_trig_eq_l156_156119


namespace discount_correct_l156_156662

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end discount_correct_l156_156662


namespace three_digit_numbers_sorted_desc_l156_156067

theorem three_digit_numbers_sorted_desc :
  ∃ n, n = 84 ∧
    ∀ (h t u : ℕ), 100 <= 100 * h + 10 * t + u ∧ 100 * h + 10 * t + u <= 999 →
    1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u → 
    n = 84 := 
by
  sorry

end three_digit_numbers_sorted_desc_l156_156067


namespace middle_integer_is_five_l156_156442

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def are_consecutive_odd_integers (a b c : ℤ) : Prop :=
  a < b ∧ b < c ∧ (∃ n : ℤ, a = b - 2 ∧ c = b + 2 ∧ is_odd a ∧ is_odd b ∧ is_odd c)

def sum_is_one_eighth_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_five :
  ∃ (a b c : ℤ), are_consecutive_odd_integers a b c ∧ sum_is_one_eighth_product a b c ∧ b = 5 :=
by
  sorry

end middle_integer_is_five_l156_156442


namespace Karsyn_payment_l156_156033

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end Karsyn_payment_l156_156033


namespace total_toys_l156_156600

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end total_toys_l156_156600


namespace volume_pyramid_problem_l156_156794

noncomputable def volume_of_pyramid : ℝ :=
  1 / 3 * 10 * 1.5

theorem volume_pyramid_problem :
  ∀ (AB BC CG : ℝ)
  (M : ℝ × ℝ × ℝ),
  AB = 4 →
  BC = 2 →
  CG = 3 →
  M = (2, 5, 1.5) →
  volume_of_pyramid = 5 := 
by
  intros AB BC CG M hAB hBC hCG hM
  sorry

end volume_pyramid_problem_l156_156794


namespace angle_in_third_quadrant_l156_156370

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) (h : π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π) :
  ∃ m : ℤ, -π - 2 * m * π < π / 2 - α ∧ (π / 2 - α) < -π / 2 - 2 * m * π :=
by
  -- Lean users note: The proof isn't required here, just setting up the statement as instructed.
  sorry

end angle_in_third_quadrant_l156_156370


namespace quotient_of_division_l156_156268

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 52) 
  (h2 : divisor = 3) 
  (h3 : remainder = 4) 
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 16 :=
by
  sorry

end quotient_of_division_l156_156268


namespace solve_quadratics_l156_156277

theorem solve_quadratics :
  (∃ x : ℝ, x^2 + 5 * x - 24 = 0) ∧ (∃ y, y^2 + 5 * y - 24 = 0) ∧
  (∃ z : ℝ, 3 * z^2 + 2 * z - 4 = 0) ∧ (∃ w, 3 * w^2 + 2 * w - 4 = 0) :=
by {
  sorry
}

end solve_quadratics_l156_156277


namespace no_function_exists_l156_156937

-- Main theorem statement
theorem no_function_exists : ¬ ∃ f : ℝ → ℝ, 
  (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * f (2 * y * f x + f y) = x^3 * f (y * f x)) ∧ 
  (∀ z : ℝ, 0 < z → f z > 0) :=
sorry

end no_function_exists_l156_156937


namespace no_play_students_count_l156_156674

theorem no_play_students_count :
  let total_students := 420
  let football_players := 325
  let cricket_players := 175
  let both_players := 130
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end no_play_students_count_l156_156674


namespace cuboid_cutout_l156_156870

theorem cuboid_cutout (x y : ℕ) (h1 : x * y = 36) (h2 : 0 < x) (h3 : x < 4) (h4 : 0 < y) (h5 : y < 15) :
  x + y = 15 :=
sorry

end cuboid_cutout_l156_156870


namespace cube_side_length_increase_20_percent_l156_156129

variable {s : ℝ} (initial_side_length_increase : ℝ) (percentage_surface_area_increase : ℝ) (percentage_volume_increase : ℝ)
variable (new_surface_area : ℝ) (new_volume : ℝ)

theorem cube_side_length_increase_20_percent :
  ∀ (s : ℝ),
  (initial_side_length_increase = 1.2 * s) →
  (new_surface_area = 6 * (1.2 * s)^2) →
  (new_volume = (1.2 * s)^3) →
  (percentage_surface_area_increase = ((new_surface_area - (6 * s^2)) / (6 * s^2)) * 100) →
  (percentage_volume_increase = ((new_volume - s^3) / s^3) * 100) →
  5 * (percentage_volume_increase - percentage_surface_area_increase) = 144 := by
  sorry

end cube_side_length_increase_20_percent_l156_156129


namespace condition_holds_l156_156224

theorem condition_holds 
  (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) : 
  (a = c ∨ a = -c) ∨ (a^2 - c^2 + d^2 = b^2) :=
by
  sorry

end condition_holds_l156_156224


namespace probability_prime_sum_30_l156_156199

def prime_numbers_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def prime_pairs_summing_to_30 : List (ℕ × ℕ) := [(7, 23), (11, 19), (13, 17)]

def num_prime_pairs := (prime_numbers_up_to_30.length.choose 2)

theorem probability_prime_sum_30 :
  (prime_pairs_summing_to_30.length / num_prime_pairs : ℚ) = 1 / 15 :=
sorry

end probability_prime_sum_30_l156_156199


namespace part_one_part_two_l156_156365

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem part_one:
  f a b (-1) = 0 → f a b x = x^2 + 2 * x + 1 :=
by
  sorry

theorem part_two:
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f 1 2 x > x + k) ↔ k < 1 :=
by
  sorry

end part_one_part_two_l156_156365


namespace sin_120_eq_half_l156_156211

theorem sin_120_eq_half :
  let Q := (-(Real.sqrt 3) / 2, 1 / 2) in -- coordinates for Q
  sin (120 * (Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_120_eq_half_l156_156211


namespace even_four_digit_increasing_count_l156_156771

theorem even_four_digit_increasing_count :
  let digits := {x // 1 ≤ x ∧ x ≤ 9}
  let even_digits := {x // x ∈ digits ∧ x % 2 = 0}
  {n : ℕ //
    ∃ a b c d : ℕ,
      n = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ even_digits ∧
      a < b ∧ b < c ∧ c < d} =
  17 :=
by sorry

end even_four_digit_increasing_count_l156_156771


namespace inequality_C_l156_156779

theorem inequality_C (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
by
  sorry

end inequality_C_l156_156779


namespace xy_identity_l156_156546

theorem xy_identity (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 :=
  sorry

end xy_identity_l156_156546


namespace emails_in_morning_and_afternoon_l156_156253

-- Conditions
def morning_emails : Nat := 5
def afternoon_emails : Nat := 8

-- Theorem statement
theorem emails_in_morning_and_afternoon : morning_emails + afternoon_emails = 13 := by
  -- Proof goes here, but adding sorry for now
  sorry

end emails_in_morning_and_afternoon_l156_156253


namespace equivalent_proof_problem_l156_156942

def math_problem (x y : ℚ) : ℚ :=
((x + y) * (3 * x - y) + y^2) / (-x)

theorem equivalent_proof_problem (hx : x = 4) (hy : y = -(1/4)) :
  math_problem x y = -23 / 2 :=
by
  sorry

end equivalent_proof_problem_l156_156942


namespace angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l156_156519

-- Conditions for (1): In ΔABC, A = 60°, a = 4√3, b = 4√2, prove B = 45°.
theorem angle_B_in_triangle_ABC
  (A : Real)
  (a b : Real)
  (hA : A = 60)
  (ha : a = 4 * Real.sqrt 3)
  (hb : b = 4 * Real.sqrt 2) :
  ∃ B : Real, B = 45 := by
  sorry

-- Conditions for (2): In ΔABC, a = 3√3, c = 2, B = 150°, prove b = 7.
theorem side_b_in_triangle_ABC
  (a c B : Real)
  (ha : a = 3 * Real.sqrt 3)
  (hc : c = 2)
  (hB : B = 150) :
  ∃ b : Real, b = 7 := by
  sorry

end angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l156_156519


namespace foma_should_give_ierema_55_coins_l156_156141

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156141


namespace correct_option_l156_156450

theorem correct_option (x y a b : ℝ) :
  ((x + 2 * y) ^ 2 ≠ x ^ 2 + 4 * y ^ 2) ∧
  ((-2 * (a ^ 3)) ^ 2 = 4 * (a ^ 6)) ∧
  (-6 * (a ^ 2) * (b ^ 5) + a * b ^ 2 ≠ -6 * a * (b ^ 3)) ∧
  (2 * (a ^ 2) * 3 * (a ^ 3) ≠ 6 * (a ^ 6)) :=
by
  sorry

end correct_option_l156_156450


namespace width_of_second_square_is_seven_l156_156602

-- The conditions translated into Lean definitions
def first_square : ℕ × ℕ := (8, 5)
def third_square : ℕ × ℕ := (5, 5)
def flag_dimensions : ℕ × ℕ := (15, 9)

-- The area calculation functions
def area (dim : ℕ × ℕ) : ℕ := dim.fst * dim.snd

-- Given areas for the first and third square
def area_first_square : ℕ := area first_square
def area_third_square : ℕ := area third_square

-- Desired flag area
def flag_area : ℕ := area flag_dimensions

-- Total area of first and third squares
def total_area_first_and_third : ℕ := area_first_square + area_third_square

-- Required area for the second square
def area_needed_second_square : ℕ := flag_area - total_area_first_and_third

-- Given length of the second square
def second_square_length : ℕ := 10

-- Solve for the width of the second square
def second_square_width : ℕ := area_needed_second_square / second_square_length

-- The proof goal
theorem width_of_second_square_is_seven : second_square_width = 7 := by
  sorry

end width_of_second_square_is_seven_l156_156602


namespace glucose_amount_in_45cc_l156_156575

noncomputable def glucose_in_container (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) : ℝ :=
  (concentration * poured_volume) / total_volume

theorem glucose_amount_in_45cc (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) :
  concentration = 10 → total_volume = 100 → poured_volume = 45 →
  glucose_in_container concentration total_volume poured_volume = 4.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end glucose_amount_in_45cc_l156_156575


namespace combined_total_time_l156_156394

theorem combined_total_time
  (Katherine_time : Real := 20)
  (Naomi_time : Real := Katherine_time * (1 + 1 / 4))
  (Lucas_time : Real := Katherine_time * (1 + 1 / 3))
  (Isabella_time : Real := Katherine_time * (1 + 1 / 2))
  (Naomi_total : Real := Naomi_time * 10)
  (Lucas_total : Real := Lucas_time * 10)
  (Isabella_total : Real := Isabella_time * 10) :
  Naomi_total + Lucas_total + Isabella_total = 816.7 := sorry

end combined_total_time_l156_156394


namespace amount_paid_after_discount_l156_156032

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end amount_paid_after_discount_l156_156032


namespace solve_r_l156_156758

theorem solve_r (r : ℚ) :
  (r^2 - 5*r + 4) / (r^2 - 8*r + 7) = (r^2 - 2*r - 15) / (r^2 - r - 20) →
  r = -5/4 :=
by
  -- Proof would go here
  sorry

end solve_r_l156_156758


namespace fraction_addition_l156_156574

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := 
by 
  sorry

end fraction_addition_l156_156574


namespace only_p_eq_3_l156_156818

theorem only_p_eq_3 (p : ℕ) (h1 : Prime p) (h2 : Prime (8 * p ^ 2 + 1)) : p = 3 := 
by
  sorry

end only_p_eq_3_l156_156818


namespace lcm_of_ratio_and_hcf_l156_156308

theorem lcm_of_ratio_and_hcf (a b : ℕ) (x : ℕ) (h_ratio : a = 3 * x ∧ b = 4 * x) (h_hcf : Nat.gcd a b = 4) : Nat.lcm a b = 48 :=
by
  sorry

end lcm_of_ratio_and_hcf_l156_156308


namespace solution_set_l156_156627

theorem solution_set {x : ℝ} :
  abs ((7 - x) / 4) < 3 ∧ 0 ≤ x ↔ 0 ≤ x ∧ x < 19 :=
by
  sorry

end solution_set_l156_156627


namespace count_even_strictly_increasing_integers_correct_l156_156773

-- Definition of condition: even four-digit integers with strictly increasing digits
def is_strictly_increasing {a b c d : ℕ} : Prop :=
1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ∈ {2, 4, 6, 8}

def count_even_strictly_increasing_integers : ℕ :=
(finset.range 10).choose 4.filter (λ l, is_strictly_increasing l.head l.nth 1 l.nth 2 l.nth 3).card

theorem count_even_strictly_increasing_integers_correct :
  count_even_strictly_increasing_integers = 46 := by
  sorry

end count_even_strictly_increasing_integers_correct_l156_156773


namespace minimum_square_area_l156_156445

-- Definitions of the given conditions
structure Rectangle where
  width : ℕ
  height : ℕ

def rect1 : Rectangle := { width := 2, height := 4 }
def rect2 : Rectangle := { width := 3, height := 5 }
def circle_diameter : ℕ := 3

-- Statement of the theorem
theorem minimum_square_area :
  ∃ sq_side : ℕ, 
    (sq_side ≥ 5 ∧ sq_side ≥ 7) ∧ 
    sq_side * sq_side = 49 := 
by
  use 7
  have h1 : 7 ≥ 5 := by norm_num
  have h2 : 7 ≥ 7 := by norm_num
  have h3 : 7 * 7 = 49 := by norm_num
  exact ⟨⟨h1, h2⟩, h3⟩

end minimum_square_area_l156_156445


namespace round_155_628_l156_156705

theorem round_155_628 :
  round (155.628 : Real) = 156 := by
  sorry

end round_155_628_l156_156705


namespace foma_gives_ierema_55_l156_156153

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l156_156153


namespace sum_of_primes_1_to_20_l156_156004

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l156_156004


namespace minimum_f_l156_156489

def f (x y : ℤ) : ℤ := |5 * x^2 + 11 * x * y - 5 * y^2|

theorem minimum_f (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : ∃ (m : ℤ), m = 5 ∧ ∀ (x y : ℤ), (x ≠ 0 ∨ y ≠ 0) → f x y ≥ m :=
by sorry

end minimum_f_l156_156489


namespace sin_cos_product_l156_156760

open Real

theorem sin_cos_product (θ : ℝ) (h : sin θ + cos θ = 3 / 4) : sin θ * cos θ = -7 / 32 := 
  by 
    sorry

end sin_cos_product_l156_156760


namespace fraction_denominator_l156_156947

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l156_156947


namespace factorize_x_squared_minus_25_l156_156048

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l156_156048


namespace reciprocal_roots_condition_l156_156676

theorem reciprocal_roots_condition (a b c : ℝ) (h : a ≠ 0) (roots_reciprocal : ∃ r s : ℝ, r * s = 1 ∧ r + s = -b/a ∧ r * s = c/a) : c = a :=
by
  sorry

end reciprocal_roots_condition_l156_156676


namespace sin_120_eq_sqrt3_div_2_l156_156213

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l156_156213


namespace sin_120_eq_sqrt3_div_2_l156_156201

theorem sin_120_eq_sqrt3_div_2 (θ : ℝ) (h₁ : sin (180 - θ) = sin θ) (h₂ : sin (60 : ℝ) = real.sqrt 3 / 2) : 
  sin (120 : ℝ) = real.sqrt 3 / 2 := 
  sorry

end sin_120_eq_sqrt3_div_2_l156_156201


namespace graph_passes_through_point_l156_156353

theorem graph_passes_through_point : ∀ (a : ℝ), a > 0 ∧ a ≠ 1 → (∃ x y, (x, y) = (0, 2) ∧ y = a^x + 1) :=
by
  intros a ha
  use 0
  use 2
  obtain ⟨ha1, ha2⟩ := ha
  have h : a^0 = 1 := by simp
  simp [h]
  sorry

end graph_passes_through_point_l156_156353


namespace fill_time_of_three_pipes_l156_156839

def rate (hours : ℕ) : ℚ := 1 / hours

def combined_rate : ℚ :=
  rate 12 + rate 15 + rate 20

def time_to_fill (rate : ℚ) : ℚ :=
  1 / rate

theorem fill_time_of_three_pipes :
  time_to_fill combined_rate = 5 := by
  sorry

end fill_time_of_three_pipes_l156_156839


namespace regular_octagon_diagonal_l156_156510

variable {a b c : ℝ}

-- Define a function to check for a regular octagon where a, b, c are respective side, shortest diagonal, and longest diagonal
def is_regular_octagon (a b c : ℝ) : Prop :=
  -- Here, we assume the standard geometric properties of a regular octagon.
  -- In a real formalization, we might model the octagon directly.

  -- longest diagonal c of a regular octagon (spans 4 sides)
  c = 2 * a

theorem regular_octagon_diagonal (a b c : ℝ) (h : is_regular_octagon a b c) : c = 2 * a :=
by
  exact h

end regular_octagon_diagonal_l156_156510


namespace simplify_expr1_simplify_expr2_l156_156550

variable (a b x y : ℝ)

theorem simplify_expr1 : 6 * a + 7 * b^2 - 9 + 4 * a - b^2 + 6 = 10 * a + 6 * b^2 - 3 :=
by
  sorry

theorem simplify_expr2 : 5 * x - 2 * (4 * x + 5 * y) + 3 * (3 * x - 4 * y) = 6 * x - 22 * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l156_156550


namespace arithmetic_sequence_term_12_l156_156525

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_term_12 (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 :=
by
  -- The following line ensures the theorem compiles correctly.
  sorry

end arithmetic_sequence_term_12_l156_156525


namespace melanie_attended_games_l156_156133

/-- Melanie attended 5 football games if there were 12 total games and she missed 7. -/
theorem melanie_attended_games (totalGames : ℕ) (missedGames : ℕ) (h₁ : totalGames = 12) (h₂ : missedGames = 7) :
  totalGames - missedGames = 5 := 
sorry

end melanie_attended_games_l156_156133


namespace foma_should_give_ierema_l156_156159

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l156_156159


namespace trader_total_discount_correct_l156_156190

theorem trader_total_discount_correct :
  let CP_A := 200
  let CP_B := 150
  let CP_C := 100
  let MSP_A := CP_A + 0.50 * CP_A
  let MSP_B := CP_B + 0.50 * CP_B
  let MSP_C := CP_C + 0.50 * CP_C
  let SP_A := 0.99 * CP_A
  let SP_B := 0.97 * CP_B
  let SP_C := 0.98 * CP_C
  let discount_A := MSP_A - SP_A
  let discount_B := MSP_B - SP_B
  let discount_C := MSP_C - SP_C
  let total_discount := discount_A + discount_B + discount_C
  total_discount = 233.5 := by sorry

end trader_total_discount_correct_l156_156190


namespace rachel_total_problems_l156_156938

theorem rachel_total_problems
    (problems_per_minute : ℕ)
    (minutes_before_bed : ℕ)
    (problems_next_day : ℕ) 
    (h1 : problems_per_minute = 5) 
    (h2 : minutes_before_bed = 12) 
    (h3 : problems_next_day = 16) : 
    problems_per_minute * minutes_before_bed + problems_next_day = 76 :=
by
  sorry

end rachel_total_problems_l156_156938


namespace service_center_milepost_l156_156954

theorem service_center_milepost :
  ∀ (first_exit seventh_exit service_fraction : ℝ), 
    first_exit = 50 →
    seventh_exit = 230 →
    service_fraction = 3 / 4 →
    (first_exit + service_fraction * (seventh_exit - first_exit) = 185) :=
by
  intros first_exit seventh_exit service_fraction h_first h_seventh h_fraction
  sorry

end service_center_milepost_l156_156954


namespace solve_arithmetic_sequence_l156_156118

theorem solve_arithmetic_sequence :
  ∃ x > 0, (x * x = (4 + 25) / 2) :=
by
  sorry

end solve_arithmetic_sequence_l156_156118


namespace swimming_speed_in_still_water_l156_156181

/-- The speed (in km/h) of a man swimming in still water given the speed of the water current
    and the time taken to swim a certain distance against the current. -/
theorem swimming_speed_in_still_water (v : ℝ) (speed_water : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed_water = 12) (h2 : time = 5) (h3 : distance = 40)
  (h4 : time = distance / (v - speed_water)) : v = 20 :=
by
  sorry

end swimming_speed_in_still_water_l156_156181


namespace prime_divisor_congruent_one_mod_p_l156_156666

theorem prime_divisor_congruent_one_mod_p (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ q ∣ p^p - 1 ∧ q % p = 1 :=
sorry

end prime_divisor_congruent_one_mod_p_l156_156666


namespace james_parking_tickets_l156_156391

-- Define the conditions
def ticket_cost_1 := 150
def ticket_cost_2 := 150
def ticket_cost_3 := 1 / 3 * ticket_cost_1
def total_cost := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def roommate_pays := total_cost / 2
def james_remaining_money := 325
def james_original_money := james_remaining_money + roommate_pays

-- Define the theorem we want to prove
theorem james_parking_tickets (h1: ticket_cost_1 = 150)
                              (h2: ticket_cost_1 = ticket_cost_2)
                              (h3: ticket_cost_3 = 1 / 3 * ticket_cost_1)
                              (h4: total_cost = ticket_cost_1 + ticket_cost_2 + ticket_cost_3)
                              (h5: roommate_pays = total_cost / 2)
                              (h6: james_remaining_money = 325)
                              (h7: james_original_money = james_remaining_money + roommate_pays):
                              total_cost = 350 :=
by
  sorry

end james_parking_tickets_l156_156391


namespace sin_120_eq_sqrt3_div_2_l156_156206

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l156_156206


namespace boats_left_l156_156810

def initial_boats : ℕ := 30
def percentage_eaten_by_fish : ℕ := 20
def boats_shot_with_arrows : ℕ := 2
def boats_blown_by_wind : ℕ := 3
def boats_sank : ℕ := 4

def boats_eaten_by_fish : ℕ := (initial_boats * percentage_eaten_by_fish) / 100

theorem boats_left : initial_boats - boats_eaten_by_fish - boats_shot_with_arrows - boats_blown_by_wind - boats_sank = 15 := by
  sorry

end boats_left_l156_156810


namespace num_factors_1728_l156_156649

open Nat

noncomputable def num_factors (n : ℕ) : ℕ :=
  (6 + 1) * (3 + 1)

theorem num_factors_1728 : 
  num_factors 1728 = 28 := by
  sorry

end num_factors_1728_l156_156649


namespace perfect_cubes_between_100_and_900_l156_156507

theorem perfect_cubes_between_100_and_900:
  ∃ n, n = 5 ∧ (∀ k, (k ≥ 5 ∧ k ≤ 9) → (k^3 ≥ 100 ∧ k^3 ≤ 900)) :=
by
  sorry

end perfect_cubes_between_100_and_900_l156_156507


namespace initial_weight_of_fish_l156_156315

theorem initial_weight_of_fish (B F : ℝ) 
  (h1 : B + F = 54) 
  (h2 : B + F / 2 = 29) : 
  F = 50 := 
sorry

end initial_weight_of_fish_l156_156315


namespace solve_trig_eq_l156_156678

noncomputable theory

open Real

theorem solve_trig_eq (x : ℝ) : (12 * sin x - 5 * cos x = 13) →
  ∃ (k : ℤ), x = (π / 2) + arctan (5 / 12) + 2 * k * π :=
by
s∞rry

end solve_trig_eq_l156_156678


namespace select_students_from_boys_and_girls_l156_156271

def number_of_ways (n : ℕ) (k : ℕ) := (nat.choose n k)

theorem select_students_from_boys_and_girls :
  let boys := 5
      girls := 4
      total_students := 4 in
  (number_of_ways boys 3) * (number_of_ways girls 1) + 
  (number_of_ways boys 2) * (number_of_ways girls 2) = 100 :=
by
  let boys := 5
  let girls := 4
  let total_students := 4
  show (number_of_ways boys 3) * (number_of_ways girls 1) + 
       (number_of_ways boys 2) * (number_of_ways girls 2) = 100
  sorry

end select_students_from_boys_and_girls_l156_156271


namespace total_people_ride_l156_156470

theorem total_people_ride (people_per_carriage : ℕ) (num_carriages : ℕ) (h1 : people_per_carriage = 12) (h2 : num_carriages = 15) : 
    people_per_carriage * num_carriages = 180 := by
  sorry

end total_people_ride_l156_156470


namespace total_ingredients_cups_l156_156831

theorem total_ingredients_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℚ) 
  (h_ratio : butter_ratio / sugar_ratio = 1 / 4 ∧ flour_ratio / sugar_ratio = 6 / 4) 
  (h_sugar : sugar_cups = 10) : 
  butter_ratio * (sugar_cups / sugar_ratio) + flour_ratio * (sugar_cups / sugar_ratio) + sugar_cups = 27.5 :=
by
  sorry

end total_ingredients_cups_l156_156831


namespace min_value_of_reciprocal_sum_l156_156085

noncomputable def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ((2016 * (a 1 + a 2016)) / 2 = 1008)

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) (h : arithmetic_sequence_condition a) :
  ∃ x : ℝ, x = 4 ∧ (∀ y, y = (1 / a 1001 + 1 / a 1016) → x ≤ y) :=
sorry

end min_value_of_reciprocal_sum_l156_156085


namespace find_triples_l156_156334

theorem find_triples :
  { (a, b, c) : ℕ × ℕ × ℕ | (c-1) * (a * b - b - a) = a + b - 2 } =
  { (2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3) } :=
by
  sorry

end find_triples_l156_156334


namespace smallest_number_l156_156868

theorem smallest_number:
    let a := 3.25
    let b := 3.26   -- 326% in decimal
    let c := 3.2    -- 3 1/5 in decimal
    let d := 3.75   -- 15/4 in decimal
    c < a ∧ c < b ∧ c < d :=
by
    sorry

end smallest_number_l156_156868


namespace joggers_meet_l156_156610

theorem joggers_meet (t_cathy t_david t_elena : ℕ) (h_cathy : t_cathy = 5) (h_david : t_david = 9) (h_elena : t_elena = 8) :
  ∃ t : ℕ, t = Nat.lcm (Nat.lcm t_cathy t_david) t_elena ∧ t = 360 ∧ t / t_cathy = 72 :=
begin
  use 360,
  split,
  { rw [h_cathy, h_david, h_elena],
    exact Nat.lcm_assoc 5 9 8 },
  split,
  { rw [Nat.lcm_assoc, h_cathy, h_david, h_elena],
    calc
      Nat.lcm (Nat.lcm 5 9) 8 = Nat.lcm 45 8      : by rw Nat.lcm_comm
                          ... = 360               : by sorry  },
  { rw h_cathy,
    exact Nat.div_eq_of_eq_mul Nat.zero_le Nat.zero_le sorry }
end

end joggers_meet_l156_156610


namespace number_of_employees_l156_156806

def fixed_time_coffee : ℕ := 5
def time_per_status_update : ℕ := 2
def time_per_payroll_update : ℕ := 3
def total_morning_routine : ℕ := 50

def time_per_employee : ℕ := time_per_status_update + time_per_payroll_update
def time_spent_on_employees : ℕ := total_morning_routine - fixed_time_coffee

theorem number_of_employees : (time_spent_on_employees / time_per_employee) = 9 := by
  sorry

end number_of_employees_l156_156806


namespace ellen_lost_legos_l156_156217

theorem ellen_lost_legos (L_initial L_final : ℕ) (h1 : L_initial = 2080) (h2 : L_final = 2063) : L_initial - L_final = 17 := by
  sorry

end ellen_lost_legos_l156_156217


namespace range_of_m_for_inequality_l156_156381

theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, |x-1| + |x+m| > 3} = {m : ℝ | m < -4 ∨ m > 2} :=
sorry

end range_of_m_for_inequality_l156_156381


namespace determine_range_a_l156_156838

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, 1 ≤ x → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

theorem determine_range_a (a : ℝ) (h : ¬ prop_p a ∧ (prop_p a ∨ prop_q a)) : 
  a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8) :=
sorry

end determine_range_a_l156_156838


namespace parabola_latus_rectum_equation_l156_156827

theorem parabola_latus_rectum_equation :
  (∃ (y x : ℝ), y^2 = 4 * x) → (∀ x, x = -1) :=
by
  sorry

end parabola_latus_rectum_equation_l156_156827


namespace initial_ratio_is_four_five_l156_156921

variable (M W : ℕ)

axiom initial_conditions :
  (M + 2 = 14) ∧ (2 * (W - 3) = 24)

theorem initial_ratio_is_four_five 
  (h : M + 2 = 14) 
  (k : 2 * (W - 3) = 24) : M / W = 4 / 5 :=
by
  sorry

end initial_ratio_is_four_five_l156_156921


namespace cookies_in_second_type_l156_156995

theorem cookies_in_second_type (x : ℕ) (h1 : 50 * 12 + 80 * x + 70 * 16 = 3320) : x = 20 :=
by sorry

end cookies_in_second_type_l156_156995


namespace num_positive_integers_which_make_polynomial_prime_l156_156222

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem num_positive_integers_which_make_polynomial_prime :
  (∃! n : ℕ, n > 0 ∧ is_prime (n^3 - 7 * n^2 + 18 * n - 10)) :=
sorry

end num_positive_integers_which_make_polynomial_prime_l156_156222


namespace solution_set_for_log_inequality_l156_156961

noncomputable def log_base_0_1 (x: ℝ) : ℝ := Real.log x / Real.log 0.1

theorem solution_set_for_log_inequality :
  ∀ x : ℝ, (0 < x) → 
  log_base_0_1 (2^x - 1) < 0 ↔ x > 1 :=
by
  sorry

end solution_set_for_log_inequality_l156_156961


namespace dany_farm_bushels_l156_156996

theorem dany_farm_bushels :
  let cows := 5
  let cows_bushels_per_day := 3
  let sheep := 4
  let sheep_bushels_per_day := 2
  let chickens := 8
  let chickens_bushels_per_day := 1
  let pigs := 6
  let pigs_bushels_per_day := 4
  let horses := 2
  let horses_bushels_per_day := 5
  cows * cows_bushels_per_day +
  sheep * sheep_bushels_per_day +
  chickens * chickens_bushels_per_day +
  pigs * pigs_bushels_per_day +
  horses * horses_bushels_per_day = 65 := by
  sorry

end dany_farm_bushels_l156_156996


namespace students_per_row_first_scenario_l156_156516

theorem students_per_row_first_scenario 
  (S R x : ℕ)
  (h1 : S = x * R + 6)
  (h2 : S = 12 * (R - 3))
  (h3 : S = 6 * R) :
  x = 5 :=
by
  sorry

end students_per_row_first_scenario_l156_156516


namespace digit_divisibility_l156_156665

theorem digit_divisibility : 
  (∃ (A : ℕ), A < 10 ∧ 
   (4573198080 + A) % 2 = 0 ∧ 
   (4573198080 + A) % 5 = 0 ∧ 
   (4573198080 + A) % 8 = 0 ∧ 
   (4573198080 + A) % 10 = 0 ∧ 
   (4573198080 + A) % 16 = 0 ∧ A = 0) := 
by { use 0; sorry }

end digit_divisibility_l156_156665


namespace sum_positive_implies_at_least_one_positive_l156_156299

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l156_156299


namespace difference_between_sevens_l156_156455

-- Define the numeral
def numeral : ℕ := 54179759

-- Define a function to find the place value of a digit at a specific position in a number
def place_value (n : ℕ) (pos : ℕ) : ℕ :=
  let digit := (n / 10^pos) % 10
  digit * 10^pos

-- Define specific place values for the two sevens
def first_seven_place : ℕ := place_value numeral 4  -- Ten-thousands place
def second_seven_place : ℕ := place_value numeral 1 -- Tens place

-- Define their values
def first_seven_value : ℕ := 7 * 10^4  -- 70,000
def second_seven_value : ℕ := 7 * 10^1  -- 70

-- Prove the difference between these place values
theorem difference_between_sevens : first_seven_value - second_seven_value = 69930 := by
  sorry

end difference_between_sevens_l156_156455


namespace isosceles_trapezoid_diagonal_length_l156_156229

theorem isosceles_trapezoid_diagonal_length
  (AB CD : ℝ) (AD BC : ℝ) :
  AB = 15 →
  CD = 9 →
  AD = 12 →
  BC = 12 →
  (AC : ℝ) = Real.sqrt 279 :=
by
  intros hAB hCD hAD hBC
  sorry

end isosceles_trapezoid_diagonal_length_l156_156229


namespace largest_prime_factor_4851_l156_156843

theorem largest_prime_factor_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 4851 → q ≤ p) :=
by
  -- todo: provide actual proof
  sorry

end largest_prime_factor_4851_l156_156843


namespace max_n_l156_156934

noncomputable def seq_a (n : ℕ) : ℤ := 3 * n - 1

noncomputable def seq_b (n : ℕ) : ℤ := 2 * n - 3

noncomputable def sum_T (n : ℕ) : ℤ := n * (3 * n + 1) / 2

noncomputable def sum_S (n : ℕ) : ℤ := n^2 - 2 * n

theorem max_n (n : ℕ) :
  ∃ n_max : ℕ, T_n < 20 * seq_b n ∧ (∀ m : ℕ, m > n_max → T_n ≥ 20 * seq_b n) :=
  sorry

end max_n_l156_156934


namespace sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l156_156607

theorem sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022:
  ( (Real.sqrt 10 + 3) ^ 2023 * (Real.sqrt 10 - 3) ^ 2022 = Real.sqrt 10 + 3 ) :=
by {
  sorry
}

end sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l156_156607


namespace find_a_l156_156912

theorem find_a (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h1 : ∀ x, f (g x) = x)
  (h2 : f x = (Real.log (x + 1) / Real.log 2) + a)
  (h3 : g 4 = 1) :
  a = 3 :=
sorry

end find_a_l156_156912


namespace sin_120_eq_sqrt3_div_2_l156_156205

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l156_156205


namespace exists_digits_divisible_by_73_no_digits_divisible_by_79_l156_156456

open Int

def decimal_digits (n : ℕ) := n < 10

theorem exists_digits_divisible_by_73 :
  ∃ (a b : ℕ), decimal_digits a ∧ decimal_digits b ∧ 
  (∀ n : ℕ, a * 10^(n+2) + b * 10^(n+1) + 222 * (10^n / 9) + 31 ≡ 0 [MOD 73]) :=
by {
  sorry
}

theorem no_digits_divisible_by_79 :
  ¬ ∃ (c d : ℕ), decimal_digits c ∧ decimal_digits d ∧ 
  (∀ n : ℕ, c * 10^(n+2) + d * 10^(n+1) + 222 * (10^n / 9) + 31 ≡ 0 [MOD 79]) :=
by {
  sorry
}

end exists_digits_divisible_by_73_no_digits_divisible_by_79_l156_156456


namespace distance_difference_l156_156043

theorem distance_difference (t : ℕ) (speed_alice speed_bob : ℕ) :
  speed_alice = 15 → speed_bob = 10 → t = 6 → (speed_alice * t) - (speed_bob * t) = 30 :=
by
  intros h1 h2 h3
  sorry

end distance_difference_l156_156043


namespace find_triangle_areas_l156_156526

variables (A B C D : Point)
variables (S_ABC S_ACD S_ABD S_BCD : ℝ)

def quadrilateral_area (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  S_ABC + S_ACD + S_ABD + S_BCD = 25

def conditions (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  (S_ABC = 2 * S_BCD) ∧ (S_ABD = 3 * S_ACD)

theorem find_triangle_areas
  (S_ABC S_ACD S_ABD S_BCD : ℝ) :
  quadrilateral_area S_ABC S_ACD S_ABD S_BCD →
  conditions S_ABC S_ACD S_ABD S_BCD →
  S_ABC = 10 ∧ S_ACD = 5 ∧ S_ABD = 15 ∧ S_BCD = 10 :=
by
  sorry

end find_triangle_areas_l156_156526


namespace length_ratio_proof_l156_156732

noncomputable def length_ratio (a b : ℝ) : ℝ :=
  let area_triangle := (sqrt 3 / 4) * (a / 3) ^ 2
  let area_square := (b / 4) ^ 2
  if area_triangle = area_square then
    a / b
  else
    0

theorem length_ratio_proof (a b : ℝ) (h : (sqrt 3 / 4) * (a / 3) ^ 2 = (b / 4) ^ 2) : length_ratio a b = sqrt ((3 * sqrt 3) / 4) :=
  sorry

end length_ratio_proof_l156_156732


namespace recurring_fraction_division_l156_156338

noncomputable def recurring_833 := 5 / 6
noncomputable def recurring_1666 := 5 / 3

theorem recurring_fraction_division : 
  (recurring_833 / recurring_1666) = 1 / 2 := 
by 
  sorry

end recurring_fraction_division_l156_156338


namespace anna_pizza_fraction_l156_156023

theorem anna_pizza_fraction :
  let total_slices := 16
  let anna_eats := 2
  let shared_slices := 1
  let anna_share := shared_slices / 3
  let fraction_alone := anna_eats / total_slices
  let fraction_shared := anna_share / total_slices
  fraction_alone + fraction_shared = 7 / 48 :=
by
  sorry

end anna_pizza_fraction_l156_156023


namespace find_prices_maximize_profit_l156_156979

-- Definition of conditions
def sales_eq1 (m n : ℝ) : Prop := 150 * m + 100 * n = 1450
def sales_eq2 (m n : ℝ) : Prop := 200 * m + 50 * n = 1100

def profit_function (x : ℕ) : ℝ := -2 * x + 1500
def range_x (x : ℕ) : Prop := 375 ≤ x ∧ x ≤ 500

-- Theorem to prove the unit prices
theorem find_prices : ∃ m n : ℝ, sales_eq1 m n ∧ sales_eq2 m n ∧ m = 3 ∧ n = 10 := 
sorry

-- Theorem to prove the profit function and maximum profit
theorem maximize_profit : ∃ (x : ℕ) (W : ℝ), range_x x ∧ W = profit_function x ∧ W = 750 :=
sorry

end find_prices_maximize_profit_l156_156979


namespace circle_equation_l156_156552

theorem circle_equation :
  ∃ (h k r : ℝ), 
    (∀ (x y : ℝ), (x, y) = (-6, 2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ (∀ (x y : ℝ), (x, y) = (2, -2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ r = 5
    ∧ h - k = -1
    ∧ (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end circle_equation_l156_156552


namespace min_value_abs_plus_2023_proof_l156_156509

noncomputable def min_value_abs_plus_2023 (a : ℚ) : Prop :=
  |a| + 2023 ≥ 2023

theorem min_value_abs_plus_2023_proof (a : ℚ) : min_value_abs_plus_2023 a :=
  by
  sorry

end min_value_abs_plus_2023_proof_l156_156509


namespace measure_angle_A_l156_156383

open Real

def triangle_area (a b c S : ℝ) (A B C : ℝ) : Prop :=
  S = (1 / 2) * b * c * sin A

def sides_and_angles (a b c A B C : ℝ) : Prop :=
  A = 2 * B

theorem measure_angle_A (a b c S A B C : ℝ)
  (h1 : triangle_area a b c S A B C)
  (h2 : sides_and_angles a b c A B C)
  (h3 : S = (a ^ 2) / 4) :
  A = π / 2 ∨ A = π / 4 :=
  sorry

end measure_angle_A_l156_156383


namespace perfect_square_n_l156_156879

open Nat

theorem perfect_square_n (n : ℕ) : 
  (∃ k : ℕ, 2 ^ (n + 1) * n = k ^ 2) ↔ 
  (∃ m : ℕ, n = 2 * m ^ 2) ∨ (∃ odd_k : ℕ, n = odd_k ^ 2 ∧ odd_k % 2 = 1) := 
sorry

end perfect_square_n_l156_156879


namespace magnification_proof_l156_156035

-- Define the conditions: actual diameter of the tissue and diameter of the magnified image
def actual_diameter := 0.0002
def magnified_diameter := 0.2

-- Define the magnification factor
def magnification_factor := magnified_diameter / actual_diameter

-- Prove that the magnification factor is 1000
theorem magnification_proof : magnification_factor = 1000 := by
  unfold magnification_factor
  unfold magnified_diameter
  unfold actual_diameter
  norm_num
  sorry

end magnification_proof_l156_156035


namespace annual_interest_rate_l156_156409

noncomputable def compound_interest 
  (P : ℝ) (A : ℝ) (n : ℕ) (t : ℝ) (r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem annual_interest_rate 
  (P := 140) (A := 169.40) (n := 2) (t := 1) :
  ∃ r : ℝ, compound_interest P A n t r ∧ r = 0.2 :=
sorry

end annual_interest_rate_l156_156409


namespace range_of_a_l156_156887

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) → ((x + a) * (x + 1) > 0)) ∧ 
  (∃ x : ℝ, ¬(x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) ∧ ((x + a) * (x + 1) > 0)) → 
  a ∈ Set.Iio (-3) := 
  sorry

end range_of_a_l156_156887


namespace angle_A_range_l156_156245

-- Definitions from the conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
axiom triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
axiom longest_side_a : a > b ∧ a > c
axiom inequality_a : a^2 < b^2 + c^2

-- Target proof statement
theorem angle_A_range (triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (longest_side_a : a > b ∧ a > c)
  (inequality_a : a^2 < b^2 + c^2) : 60 < A ∧ A < 90 := 
sorry

end angle_A_range_l156_156245


namespace slope_of_line_in_terms_of_angle_l156_156557

variable {x y : ℝ}

theorem slope_of_line_in_terms_of_angle (h : 2 * Real.sqrt 3 * x - 2 * y - 1 = 0) :
    ∃ α : ℝ, 0 ≤ α ∧ α < Real.pi ∧ Real.tan α = Real.sqrt 3 ∧ α = Real.pi / 3 :=
by
  sorry

end slope_of_line_in_terms_of_angle_l156_156557


namespace polynomial_function_value_l156_156236

theorem polynomial_function_value 
  (p q r s : ℝ) 
  (h : p - q + r - s = 4) : 
  2 * p + q - 3 * r + 2 * s = -8 := 
by 
  sorry

end polynomial_function_value_l156_156236


namespace vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_l156_156263

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

theorem vertex_coordinates (a : ℝ) (H : a = 1) : 
    (∃ v_x v_y : ℝ, quadratic_function a v_x = v_y ∧ v_x = -5 / 2 ∧ v_y = -9 / 4) := 
by {
    sorry
}

theorem quadratic_through_point : 
    (∃ a : ℝ, (quadratic_function a 0 = -2) ∧ (∀ x, quadratic_function a x = -2 * (x + 1)^2)) := 
by {
    sorry
}

theorem a_less_than_neg_2_fifth 
  (x1 x2 y1 y2 a : ℝ) (H1 : x1 + x2 = 2) (H2 : x1 < x2) (H3 : y1 > y2) 
  (Hfunc : ∀ x, quadratic_function (a * x + 2 * a + 2) (x + 1) = quadratic_function x y) :
    a < -2 / 5 := 
by {
    sorry
}

end vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_l156_156263


namespace ryan_more_hours_english_than_spanish_l156_156485

-- Define the time spent on various languages as constants
def hoursEnglish : ℕ := 7
def hoursSpanish : ℕ := 4

-- State the problem as a theorem
theorem ryan_more_hours_english_than_spanish : hoursEnglish - hoursSpanish = 3 :=
by sorry

end ryan_more_hours_english_than_spanish_l156_156485


namespace num_adult_tickets_l156_156027

variables (A C : ℕ)

def num_tickets (A C : ℕ) : Prop := A + C = 900
def total_revenue (A C : ℕ) : Prop := 7 * A + 4 * C = 5100

theorem num_adult_tickets : ∃ A, ∃ C, num_tickets A C ∧ total_revenue A C ∧ A = 500 := 
by
  sorry

end num_adult_tickets_l156_156027


namespace find_real_numbers_l156_156486

theorem find_real_numbers (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) :
  (x = 1 ∧ y = 2 ∧ z = -1) ∨ 
  (x = 1 ∧ y = -1 ∧ z = 2) ∨
  (x = 2 ∧ y = 1 ∧ z = -1) ∨ 
  (x = 2 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 2) ∨
  (x = -1 ∧ y = 2 ∧ z = 1) := 
sorry

end find_real_numbers_l156_156486


namespace base_case_of_interior_angle_sum_l156_156249

-- Definitions consistent with conditions: A convex polygon with at least n sides where n >= 3.
def convex_polygon (n : ℕ) : Prop := n ≥ 3

-- Proposition: If w the sum of angles for convex polygons, we start checking from n = 3.
theorem base_case_of_interior_angle_sum (n : ℕ) (h : convex_polygon n) :
  n = 3 := 
by
  sorry

end base_case_of_interior_angle_sum_l156_156249


namespace sum_pos_implies_one_pos_l156_156301

theorem sum_pos_implies_one_pos (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 := 
sorry

end sum_pos_implies_one_pos_l156_156301


namespace sum_of_numbers_in_ratio_with_lcm_l156_156689

theorem sum_of_numbers_in_ratio_with_lcm
  (x : ℕ)
  (h1 : Nat.lcm (2 * x) (Nat.lcm (3 * x) (5 * x)) = 120) :
  (2 * x) + (3 * x) + (5 * x) = 40 := 
sorry

end sum_of_numbers_in_ratio_with_lcm_l156_156689


namespace directrix_of_parabola_l156_156284

-- Define the parabola x^2 = 16y
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

-- Define the directrix equation
def directrix (y : ℝ) : Prop := y = -4

-- Theorem stating that the directrix of the given parabola is y = -4
theorem directrix_of_parabola : ∀ x y: ℝ, parabola x y → ∃ y, directrix y :=
by
  sorry

end directrix_of_parabola_l156_156284


namespace increase_in_area_is_44_percent_l156_156305

-- Let's define the conditions first
variables {r : ℝ} -- radius of the medium pizza
noncomputable def radius_large (r : ℝ) := 1.2 * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Now we state the Lean theorem that expresses the problem
theorem increase_in_area_is_44_percent (r : ℝ) : 
  (area (radius_large r) - area r) / area r * 100 = 44 :=
by
  sorry

end increase_in_area_is_44_percent_l156_156305


namespace smallest_n_l156_156880

theorem smallest_n (a b c n : ℕ) (h1 : n = 100 * a + 10 * b + c)
  (h2 : n = a + b + c + a * b + b * c + a * c + a * b * c)
  (h3 : n >= 100 ∧ n < 1000)
  (h4 : a ≥ 1 ∧ a ≤ 9)
  (h5 : b ≥ 0 ∧ b ≤ 9)
  (h6 : c ≥ 0 ∧ c ≤ 9) :
  n = 199 :=
sorry

end smallest_n_l156_156880


namespace inequality_solution_range_of_a_l156_156503

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

def range_y := Set.Icc (-2 : ℝ) 2

def subset_property (a : ℝ) : Prop := 
  Set.Icc a (2 * a - 1) ⊆ range_y

theorem inequality_solution (x : ℝ) :
  f x ≤ x^2 - 3 * x + 1 ↔ x ≤ 1 ∨ x ≥ 3 := sorry

theorem range_of_a (a : ℝ) :
  subset_property a ↔ 1 ≤ a ∧ a ≤ 3 / 2 := sorry

end inequality_solution_range_of_a_l156_156503


namespace solve_trig_eq_l156_156682

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l156_156682


namespace discriminant_divisible_l156_156122

theorem discriminant_divisible (a b: ℝ) (n: ℤ) (h: (∃ x1 x2: ℝ, 2018*x1^2 + a*x1 + b = 0 ∧ 2018*x2^2 + a*x2 + b = 0 ∧ x1 - x2 = n)): 
  ∃ k: ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := 
by 
  sorry

end discriminant_divisible_l156_156122


namespace servings_in_container_l156_156317

def convert_to_improper_fraction (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

def servings (container : ℚ) (serving_size : ℚ) : ℚ :=
  container / serving_size

def mixed_number (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

theorem servings_in_container : 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  servings container serving_size = expected_servings :=
by 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  sorry

end servings_in_container_l156_156317


namespace michael_and_sarah_games_l156_156291

namespace FourSquareLeague

-- Number of players in total
def totalPlayers : ℕ := 12

-- Set of all players, assuming Michael and Sarah are elements of this set
def players : Finset ℕ := Finset.range totalPlayers

-- Number of players per game
def playersPerGame : ℕ := 6

-- Subset of players excluding Michael and Sarah
def remainingPlayers : Finset ℕ := (players \ {0, 1})

-- Number of possible games Michael and Sarah can play together
def gamesWithMichaelAndSarah : ℕ := (Finset.card (remainingPlayers.powersetLen (playersPerGame - 2)))

-- Prove that the number of games with Michael and Sarah is 210
theorem michael_and_sarah_games : gamesWithMichaelAndSarah = 210 := by
  sorry

end FourSquareLeague

end michael_and_sarah_games_l156_156291


namespace find_a7_coefficient_l156_156895

theorem find_a7_coefficient (a_7 : ℤ) : 
    (∀ x : ℤ, (x+1)^5 * (2*x-1)^3 = a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) → a_7 = 28 :=
by
  sorry

end find_a7_coefficient_l156_156895


namespace find_littering_citations_l156_156725

def is_total_citations (total : ℕ) (L : ℕ) : Prop :=
  let off_leash_dogs := L
  let parking_fines := 2 * (L + off_leash_dogs)
  total = L + off_leash_dogs + parking_fines

theorem find_littering_citations :
  ∀ L : ℕ, is_total_citations 24 L → L = 4 :=
by
  intros L h
  let off_leash_dogs := L
  let parking_fines := 2 * (L + off_leash_dogs)
  have h1 : 24 = L + off_leash_dogs + parking_fines := h
  have h2 : off_leash_dogs = L := sorry -- Directly applies from problem statement
  have h3 : parking_fines = 2 * (L + L) := sorry -- Applies the double citation fine
  rw h2 at h3
  rw h3 at h1
  simp at h1
  exact sorry -- Final solve step, equating and solving for L

end find_littering_citations_l156_156725


namespace marge_funds_l156_156407

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end marge_funds_l156_156407


namespace value_of_f_l156_156570

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end value_of_f_l156_156570


namespace tangency_condition_l156_156643

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 3)^2 = 4

theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m → x^2 = 9 - 9 * y^2 ∧ x^2 = 4 + m * (y + 3)^2 → ((m - 9) * y^2 + 6 * m * y + (9 * m - 5) = 0 → 36 * m^2 - 4 * (m - 9) * (9 * m - 5) = 0 ) ) → 
  m = 5 / 54 :=
by
  sorry

end tangency_condition_l156_156643


namespace sum_series_a_sum_series_b_sum_series_c_l156_156685

-- Part (a)
theorem sum_series_a : (∑' n : ℕ, (1 / 2) ^ (n + 1)) = 1 := by
  --skip proof
  sorry

-- Part (b)
theorem sum_series_b : (∑' n : ℕ, (1 / 3) ^ (n + 1)) = 1/2 := by
  --skip proof
  sorry

-- Part (c)
theorem sum_series_c : (∑' n : ℕ, (1 / 4) ^ (n + 1)) = 1/3 := by
  --skip proof
  sorry

end sum_series_a_sum_series_b_sum_series_c_l156_156685


namespace marbles_per_friend_l156_156742

variable (initial_marbles remaining_marbles given_marbles_per_friend : ℕ)

-- conditions in a)
def condition_initial_marbles := initial_marbles = 500
def condition_remaining_marbles := 4 * remaining_marbles = 720
def condition_total_given_marbles := initial_marbles - remaining_marbles = 320
def condition_given_marbles_per_friend := given_marbles_per_friend * 4 = 320

-- question proof goal
theorem marbles_per_friend (initial_marbles: ℕ) (remaining_marbles: ℕ) (given_marbles_per_friend: ℕ) :
  (condition_initial_marbles initial_marbles) →
  (condition_remaining_marbles remaining_marbles) →
  (condition_total_given_marbles initial_marbles remaining_marbles) →
  (condition_given_marbles_per_friend given_marbles_per_friend) →
  given_marbles_per_friend = 80 :=
by
  intros hinitial hremaining htotal_given hgiven_per_friend
  sorry

end marbles_per_friend_l156_156742


namespace number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l156_156368

theorem number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100 :
  ∃! (n : ℕ), n = 3 ∧ ∀ (x y : ℕ), x > 0 → y > 0 → x^2 - y^2 = 100 ↔ (x, y) = (26, 24) ∨ (x, y) = (15, 10) ∨ (x, y) = (15, 5) :=
by
  sorry

end number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l156_156368


namespace equation_of_line_is_correct_l156_156957

/-! Given the circle x^2 + y^2 + 2x - 4y + a = 0 with a < 3 and the midpoint of the chord AB as C(-2, 3), prove that the equation of the line l that intersects the circle at points A and B is x - y + 5 = 0. -/

theorem equation_of_line_is_correct (a : ℝ) (h : a < 3) :
  ∃ l : ℝ × ℝ × ℝ, (l = (1, -1, 5)) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0 → 
    (x - y + 5 = 0)) :=
sorry

end equation_of_line_is_correct_l156_156957


namespace permutations_count_divisible_by_4_l156_156262

open Finset

theorem permutations_count_divisible_by_4 (n k : ℕ) (h : n = 4 * k) :
  ∃ σ : Equiv.Perm (Fin n), 
  (∀ j : Fin n, σ j + σ.symm j = n + 1) ∧ 
  (univ.image σ = univ : Finset (Fin n)) ∧ 
  univ.card = ∑_{σ : Equiv.Perm (Fin n)} 1 
  = (Nat.factorial (2 * k)) / (Nat.factorial k) :=
by
  sorry

end permutations_count_divisible_by_4_l156_156262


namespace given_roots_find_coefficients_l156_156054

theorem given_roots_find_coefficients {a b c : ℝ} :
  (1:ℝ)^5 + 2*(1)^4 + a * (1:ℝ)^2 + b * (1:ℝ) = c →
  (-1:ℝ)^5 + 2*(-1:ℝ)^4 + a * (-1:ℝ)^2 + b * (-1:ℝ) = c →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by
  intros h1 h2
  sorry

end given_roots_find_coefficients_l156_156054


namespace fare_ratio_l156_156737

theorem fare_ratio (F1 F2 : ℕ) (h1 : F1 = 96000) (h2 : F1 + F2 = 224000) : F1 / (Nat.gcd F1 F2) = 3 ∧ F2 / (Nat.gcd F1 F2) = 4 :=
by
  sorry

end fare_ratio_l156_156737


namespace problem_solution_l156_156018

-- Define the necessary conditions
def f (x : ℤ) : ℤ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Define the main theorem
theorem problem_solution :
  (Nat.gcd 840 1785 = 105) ∧ (f 2 = 62) :=
by {
  -- We include sorry here to indicate that the proof is omitted.
  sorry
}

end problem_solution_l156_156018


namespace horizontal_asymptote_value_l156_156514

theorem horizontal_asymptote_value :
  (∃ y : ℝ, ∀ x : ℝ, (y = (18 * x^5 + 6 * x^3 + 3 * x^2 + 5 * x + 4) / (6 * x^5 + 4 * x^3 + 5 * x^2 + 2 * x + 1)) → y = 3) :=
by
  sorry

end horizontal_asymptote_value_l156_156514


namespace combinatorial_difference_zero_combinatorial_sum_466_l156_156019

-- Statement for the first problem
theorem combinatorial_difference_zero : 
  Nat.choose 10 4 - Nat.choose 7 3 * 3.factorial = 0 :=
by
  sorry

-- Statement for the second problem
theorem combinatorial_sum_466 (n : ℕ) (h1 : 9.5 ≤ n) (h2 : n ≤ 10.5) (h3 : n = 10) :
  Nat.choose (3 * n) (38 - n) + Nat.choose (21 + n) (3 * n) = 466 :=
by
  sorry

end combinatorial_difference_zero_combinatorial_sum_466_l156_156019


namespace slope_of_line_l156_156164

theorem slope_of_line : ∀ (x y : ℝ), (x / 4 - y / 3 = 1) → ((3 * x / 4) - 3) = 0 → (y = (3 / 4) * x - 3) :=
by 
  intros x y h_eq h_slope 
  sorry

end slope_of_line_l156_156164


namespace student_average_marks_l156_156189

theorem student_average_marks 
(P C M : ℕ) 
(h1 : (P + M) / 2 = 90) 
(h2 : (P + C) / 2 = 70) 
(h3 : P = 65) : 
  (P + C + M) / 3 = 85 :=
  sorry

end student_average_marks_l156_156189


namespace petyas_number_l156_156416

theorem petyas_number :
  ∃ (N : ℕ), 
  (N % 2 = 1 ∧ ∃ (M : ℕ), N = 149 * M ∧ (M = Nat.mod (N : ℕ) (100))) →
  (N = 745 ∨ N = 3725) :=
by
  sorry

end petyas_number_l156_156416


namespace cos_2theta_l156_156240

theorem cos_2theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 5) : Real.cos (2 * θ) = -2 / 3 :=
by
  sorry

end cos_2theta_l156_156240


namespace time_to_eat_potatoes_l156_156077

theorem time_to_eat_potatoes (rate : ℕ → ℕ → ℝ) (potatoes : ℕ → ℕ → ℝ) 
    (minutes : ℕ) (hours : ℝ) (total_potatoes : ℕ) : 
    rate 3 20 = 9 / 1 -> potatoes 27 9 = 3 := 
by
  intro h1
  -- You can add intermediate steps here as optional comments for clarity during proof construction
  /- 
  Given: 
  rate 3 20 = 9 -> Jason's rate of eating potatoes is 9 potatoes per hour
  time = potatoes / rate -> 27 potatoes / 9 potatoes/hour = 3 hours
  -/
  sorry

end time_to_eat_potatoes_l156_156077


namespace total_sours_is_123_l156_156940

noncomputable def cherry_sours := 32
noncomputable def lemon_sours := 40 -- Derived from the ratio 4/5 = 32/x
noncomputable def orange_sours := 24 -- 25% of the total sours in the bag after adding them
noncomputable def grape_sours := 27 -- Derived from the ratio 3/2 = 40/y

theorem total_sours_is_123 :
  cherry_sours + lemon_sours + orange_sours + grape_sours = 123 :=
by
  sorry

end total_sours_is_123_l156_156940


namespace pump_no_leak_fill_time_l156_156987

noncomputable def pump_fill_time (P t l : ℝ) :=
  1 / P - 1 / l = 1 / t

theorem pump_no_leak_fill_time :
  ∃ P : ℝ, pump_fill_time P (13 / 6) 26 ∧ P = 2 :=
by
  sorry

end pump_no_leak_fill_time_l156_156987


namespace complex_solution_l156_156056

open Complex

theorem complex_solution (z : ℂ) (h : z + Complex.abs z = 1 + Complex.I) : z = Complex.I := 
by
  sorry

end complex_solution_l156_156056


namespace sum_of_primes_1_to_20_l156_156003

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l156_156003


namespace evaluate_expression_121point5_l156_156068

theorem evaluate_expression_121point5 :
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  (1 / 3) * x^4 * y^5 = 121.5 :=
by
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  sorry

end evaluate_expression_121point5_l156_156068


namespace parabola_maximum_value_l156_156512

noncomputable def maximum_parabola (a b c : ℝ) (h := -b / (2*a)) (k := a * h^2 + b * h + c) : Prop :=
  ∀ (x : ℝ), a ≠ 0 → b = 12 → c = 4 → a = -3 → k = 16

theorem parabola_maximum_value : maximum_parabola (-3) 12 4 :=
by
  sorry

end parabola_maximum_value_l156_156512


namespace mass_percentage_O_correct_l156_156755

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_B : ℝ := 10.81
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_H3BO3 : ℝ := (3 * molar_mass_H) + (1 * molar_mass_B) + (3 * molar_mass_O)

noncomputable def mass_percentage_O_in_H3BO3 : ℝ := ((3 * molar_mass_O) / molar_mass_H3BO3) * 100

theorem mass_percentage_O_correct : abs (mass_percentage_O_in_H3BO3 - 77.59) < 0.01 := 
sorry

end mass_percentage_O_correct_l156_156755


namespace number_of_clients_l156_156593

theorem number_of_clients (cars_clients_selects : ℕ)
                          (cars_selected_per_client : ℕ)
                          (each_car_selected_times : ℕ)
                          (total_cars : ℕ)
                          (h1 : total_cars = 18)
                          (h2 : cars_clients_selects = total_cars * each_car_selected_times)
                          (h3 : each_car_selected_times = 3)
                          (h4 : cars_selected_per_client = 3)
                          : total_cars * each_car_selected_times / cars_selected_per_client = 18 :=
by {
  sorry
}

end number_of_clients_l156_156593


namespace sin_30_plus_cos_60_l156_156170

-- Define the trigonometric evaluations as conditions
def sin_30_degree := 1 / 2
def cos_60_degree := 1 / 2

-- Lean statement for proving the sum of these values
theorem sin_30_plus_cos_60 : sin_30_degree + cos_60_degree = 1 := by
  sorry

end sin_30_plus_cos_60_l156_156170


namespace sin_half_angle_l156_156778

theorem sin_half_angle 
  (θ : ℝ) 
  (h_cos : |Real.cos θ| = 1 / 5) 
  (h_theta : 5 * Real.pi / 2 < θ ∧ θ < 3 * Real.pi)
  : Real.sin (θ / 2) = - (Real.sqrt 15) / 5 := 
by
  sorry

end sin_half_angle_l156_156778


namespace park_width_l156_156185

/-- The rectangular park theorem -/
theorem park_width 
  (length : ℕ)
  (lawn_area : ℤ)
  (road_width : ℕ)
  (crossroads : ℕ)
  (W : ℝ) :
  length = 60 →
  lawn_area = 2109 →
  road_width = 3 →
  crossroads = 2 →
  W = (2109 + (2 * 3 * 60) : ℝ) / 60 :=
sorry

end park_width_l156_156185


namespace least_x_value_l156_156707

theorem least_x_value : ∀ x : ℝ, (4 * x^2 + 7 * x + 3 = 5) → x = -2 ∨ x >= -2 := by 
    intro x
    intro h
    sorry

end least_x_value_l156_156707


namespace min_prime_factors_of_expression_l156_156226

theorem min_prime_factors_of_expression (m n : ℕ) : 
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ p1 ≠ p2 ∧ p1 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧ p2 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) := 
sorry

end min_prime_factors_of_expression_l156_156226


namespace linear_function_intersects_x_axis_at_2_0_l156_156108

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end linear_function_intersects_x_axis_at_2_0_l156_156108


namespace num_convex_pentagons_l156_156339

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l156_156339


namespace problem_l156_156252

theorem problem (a b c : ℝ) (h1 : ∀ (x : ℝ), x^2 + 3 * x - 1 = 0 → x^4 + a * x^2 + b * x + c = 0) :
  a + b + 4 * c + 100 = 93 := 
sorry

end problem_l156_156252


namespace books_per_continent_l156_156603

-- Definition of the given conditions
def total_books := 488
def continents_visited := 4

-- The theorem we need to prove
theorem books_per_continent : total_books / continents_visited = 122 :=
sorry

end books_per_continent_l156_156603


namespace math_solution_l156_156100

noncomputable def math_problem (x y z : ℝ) : Prop :=
  (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) ∧ (x + y + z = 1) → 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1 / 16)

theorem math_solution (x y z : ℝ) :
  math_problem x y z := 
by
  sorry

end math_solution_l156_156100


namespace consecutive_sum_15_number_of_valid_sets_l156_156905

theorem consecutive_sum_15 : 
  ∃ n (a : ℕ), n ≥ 2 ∧ a > 0 ∧ (n * (2 * a + n - 1)) = 30 :=
begin
  sorry
end

theorem number_of_valid_sets : 
  finset.card ((finset.filter (λ n a, n ≥ 2 ∧ a > 0 ∧ (n * (2 * a + n - 1)) = 30) (finset.range 15).product (finset.range 15))) = 2 :=
begin
  sorry
end

end consecutive_sum_15_number_of_valid_sets_l156_156905


namespace Karsyn_payment_l156_156034

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end Karsyn_payment_l156_156034


namespace percent_difference_calculation_l156_156777

theorem percent_difference_calculation :
  (0.80 * 45) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_calculation_l156_156777


namespace travel_time_without_walking_l156_156797

-- Definitions based on the problem's conditions
def walking_time_without_escalator (x y : ℝ) : Prop := 75 * x = y
def walking_time_with_escalator (x k y : ℝ) : Prop := 30 * (x + k) = y

-- Main theorem: Time taken to travel the distance with the escalator alone
theorem travel_time_without_walking (x k y : ℝ) (h1 : walking_time_without_escalator x y) (h2 : walking_time_with_escalator x k y) : y / k = 50 :=
by
  sorry

end travel_time_without_walking_l156_156797


namespace hiking_trip_rate_ratio_l156_156178

theorem hiking_trip_rate_ratio 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down : ℝ)
  (h1 : rate_up = 7) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 21) 
  (h4 : time_down = 2) : 
  (distance_down / time_down) / rate_up = 1.5 :=
by
  -- skip the proof as per instructions
  sorry

end hiking_trip_rate_ratio_l156_156178


namespace coefficient_of_x3y0_l156_156527

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def f (m n : ℕ) : ℕ :=
  binomial_coeff 6 m * binomial_coeff 4 n

theorem coefficient_of_x3y0 :
  f 3 0 = 20 :=
by
  sorry

end coefficient_of_x3y0_l156_156527


namespace math_problem_l156_156397

noncomputable def compute_value (c d : ℝ) : ℝ := 100 * c + d

-- Problem statement as a theorem
theorem math_problem
  (c d : ℝ)
  (H1 : ∀ x : ℝ, (x + c) * (x + d) * (x + 10) = 0 → x = -c ∨ x = -d ∨ x = -10)
  (H2 : ∀ x : ℝ, (x + 3 * c) * (x + 5) * (x + 8) = 0 → (x = -4 ∧ ∀ y : ℝ, y ≠ -4 → (y + d) * (y + 10) ≠ 0))
  (H3 : c ≠ 4 / 3 → 3 * c = d ∨ 3 * c = 10) :
  compute_value c d = 141.33 :=
by sorry

end math_problem_l156_156397


namespace gcd_1821_2993_l156_156346

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := 
by 
  sorry

end gcd_1821_2993_l156_156346


namespace smallest_a_not_invertible_mod_77_and_88_l156_156296

theorem smallest_a_not_invertible_mod_77_and_88 :
  ∃ (a : ℕ), (∀ (b : ℕ), b > 0 → gcd(a, 77) > 1 ∧ gcd(a, 88) > 1 ∧ (gcd(b, 77) > 1 ∧ gcd(b, 88) > 1) → a ≤ b) ∧ a = 14 :=
begin
  sorry
end

end smallest_a_not_invertible_mod_77_and_88_l156_156296


namespace trigonometric_expression_simplification_l156_156061

theorem trigonometric_expression_simplification (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := 
sorry

end trigonometric_expression_simplification_l156_156061


namespace calculate_expression_l156_156474

theorem calculate_expression : 
  (3.242 * (14 + 6) - 7.234 * 7) / 20 = 0.7101 :=
by
  sorry

end calculate_expression_l156_156474


namespace bernardo_probability_is_correct_l156_156871

noncomputable def bernardo_larger_probability : ℚ :=
  let total_bernardo_combinations := (Nat.choose 10 3 : ℚ)
  let total_silvia_combinations := (Nat.choose 8 3 : ℚ)
  let bernardo_has_10 := (Nat.choose 8 2 : ℚ) / total_bernardo_combinations
  let bernardo_not_has_10 := ((total_silvia_combinations - 1) / total_silvia_combinations) / 2
  bernardo_has_10 * 1 + (1 - bernardo_has_10) * bernardo_not_has_10

theorem bernardo_probability_is_correct :
  bernardo_larger_probability = 19 / 28 := by
  sorry

end bernardo_probability_is_correct_l156_156871


namespace simplify_fraction_l156_156117

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end simplify_fraction_l156_156117


namespace one_eq_a_l156_156885

theorem one_eq_a (x y z a : ℝ) (h₁: x + y + z = a) (h₂: 1/x + 1/y + 1/z = 1/a) :
  x = a ∨ y = a ∨ z = a :=
  sorry

end one_eq_a_l156_156885


namespace min_x_squared_plus_y_squared_l156_156974

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end min_x_squared_plus_y_squared_l156_156974


namespace park_width_l156_156186

/-- The rectangular park theorem -/
theorem park_width 
  (length : ℕ)
  (lawn_area : ℤ)
  (road_width : ℕ)
  (crossroads : ℕ)
  (W : ℝ) :
  length = 60 →
  lawn_area = 2109 →
  road_width = 3 →
  crossroads = 2 →
  W = (2109 + (2 * 3 * 60) : ℝ) / 60 :=
sorry

end park_width_l156_156186


namespace profit_margin_increase_l156_156864

theorem profit_margin_increase (CP : ℝ) (SP : ℝ) (NSP : ℝ) (initial_margin : ℝ) (desired_margin : ℝ) :
  initial_margin = 0.25 → desired_margin = 0.40 → SP = (1 + initial_margin) * CP → NSP = (1 + desired_margin) * CP →
  ((NSP - SP) / SP) * 100 = 12 := 
by 
  intros h1 h2 h3 h4
  sorry

end profit_margin_increase_l156_156864


namespace new_bathroom_area_l156_156556

variable (area : ℕ) (width : ℕ) (extension : ℕ)

theorem new_bathroom_area (h1 : area = 96) (h2 : width = 8) (h3 : extension = 2) :
  (let orig_length := area / width;
       new_length := orig_length + extension;
       new_width := width + extension;
       new_area := new_length * new_width
   in new_area) = 140 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end new_bathroom_area_l156_156556


namespace min_units_for_profitability_profitability_during_epidemic_l156_156722

-- Conditions
def assembly_line_cost : ℝ := 1.8
def selling_price_per_product : ℝ := 0.1
def max_annual_output : ℕ := 100

noncomputable def production_cost (x : ℕ) : ℝ := 5 + 135 / (x + 1)

-- Part 1: Prove Minimum x for profitability
theorem min_units_for_profitability (x : ℕ) :
  (10 - (production_cost x)) * x - assembly_line_cost > 0 ↔ x ≥ 63 := sorry

-- Part 2: Profitability and max profit output during epidemic
theorem profitability_during_epidemic (x : ℕ) :
  (60 < x ∧ x ≤ max_annual_output) → 
  ((10 - (production_cost x)) * 60 - (x - 60) - assembly_line_cost > 0) ↔ x = 89 := sorry

end min_units_for_profitability_profitability_during_epidemic_l156_156722


namespace typing_speed_in_6_minutes_l156_156448

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end typing_speed_in_6_minutes_l156_156448


namespace bottles_of_regular_soda_l156_156177

theorem bottles_of_regular_soda
  (diet_soda : ℕ)
  (apples : ℕ)
  (more_bottles_than_apples : ℕ)
  (R : ℕ)
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : more_bottles_than_apples = 26)
  (h4 : R + diet_soda = apples + more_bottles_than_apples) :
  R = 72 := 
by sorry

end bottles_of_regular_soda_l156_156177


namespace polynomial_remainder_l156_156015

theorem polynomial_remainder (p : Polynomial ℝ) :
  (p.eval 2 = 3) → (p.eval 3 = 9) → ∃ q : Polynomial ℝ, p = (Polynomial.X - 2) * (Polynomial.X - 3) * q + (6 * Polynomial.X - 9) :=
by
  sorry

end polynomial_remainder_l156_156015


namespace repeating_decimal_simplest_denominator_l156_156949

theorem repeating_decimal_simplest_denominator : 
  ∃ (a b : ℕ), (a / b = 2 / 3) ∧ nat.gcd a b = 1 ∧ b = 3 :=
by
  sorry

end repeating_decimal_simplest_denominator_l156_156949


namespace hours_between_dates_not_thirteen_l156_156884

def total_hours (start_date: ℕ × ℕ × ℕ × ℕ) (end_date: ℕ × ℕ × ℕ × ℕ) (days_in_dec: ℕ) : ℕ :=
  let (start_year, start_month, start_day, start_hour) := start_date
  let (end_year, end_month, end_day, end_hour) := end_date
  (days_in_dec - start_day) * 24 - start_hour + end_day * 24 + end_hour

theorem hours_between_dates_not_thirteen :
  let start_date := (2015, 12, 30, 23)
  let end_date := (2016, 1, 1, 12)
  let days_in_dec := 31
  total_hours start_date end_date days_in_dec ≠ 13 :=
by
  sorry

end hours_between_dates_not_thirteen_l156_156884


namespace line_equation_intercept_twice_x_intercept_l156_156754

theorem line_equation_intercept_twice_x_intercept 
  {x y : ℝ}
  (intersection_point : ∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0) 
  (y_intercept_is_twice_x_intercept : ∃ (a : ℝ), ∀ (x y : ℝ), y = 2 * a ∧ x = a) :
  (∃ (x y : ℝ), 2 * x - 3 * y = 0) ∨ (∃ (x y : ℝ), 2 * x + y - 8 = 0) :=
sorry

end line_equation_intercept_twice_x_intercept_l156_156754


namespace brian_total_distance_l156_156874

noncomputable def miles_per_gallon : ℝ := 20
noncomputable def tank_capacity : ℝ := 15
noncomputable def tank_fraction_remaining : ℝ := 3 / 7

noncomputable def total_miles_traveled (miles_per_gallon tank_capacity tank_fraction_remaining : ℝ) : ℝ :=
  let total_miles := miles_per_gallon * tank_capacity
  let fuel_used := tank_capacity * (1 - tank_fraction_remaining)
  let miles_traveled := fuel_used * miles_per_gallon
  miles_traveled

theorem brian_total_distance : 
  total_miles_traveled miles_per_gallon tank_capacity tank_fraction_remaining = 171.4 := 
by
  sorry

end brian_total_distance_l156_156874


namespace simplify_fraction_l156_156116

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end simplify_fraction_l156_156116


namespace problem1_solution_set_problem2_a_range_l156_156331

-- Define the function f
def f (x a : ℝ) := |x - a| + 5 * x

-- Problem Part 1: Prove for a = -1, the solution set for f(x) ≤ 5x + 3 is [-4, 2]
theorem problem1_solution_set :
  ∀ (x : ℝ), f x (-1) ≤ 5 * x + 3 ↔ -4 ≤ x ∧ x ≤ 2 := by
  sorry

-- Problem Part 2: Prove that if f(x) ≥ 0 for all x ≥ -1, then a ≥ 4 or a ≤ -6
theorem problem2_a_range (a : ℝ) :
  (∀ (x : ℝ), x ≥ -1 → f x a ≥ 0) ↔ a ≥ 4 ∨ a ≤ -6 := by
  sorry

end problem1_solution_set_problem2_a_range_l156_156331


namespace number_of_triangles_from_8_points_on_circle_l156_156620

-- Definitions based on the problem conditions
def points_on_circle : ℕ := 8

-- Problem statement without the proof
theorem number_of_triangles_from_8_points_on_circle :
  ∃ n : ℕ, n = (points_on_circle.choose 3) ∧ n = 56 := 
by
  sorry

end number_of_triangles_from_8_points_on_circle_l156_156620


namespace correct_proposition_D_l156_156967

theorem correct_proposition_D (a b c : ℝ) (h : a > b) : a - c > b - c :=
by
  sorry

end correct_proposition_D_l156_156967


namespace total_books_l156_156513

-- Lean 4 Statement
theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) (albert_books : ℝ) (total_books : ℝ) 
  (h1 : stu_books = 9) 
  (h2 : albert_ratio = 4.5) 
  (h3 : albert_books = stu_books * albert_ratio) 
  (h4 : total_books = stu_books + albert_books) : 
  total_books = 49.5 := 
sorry

end total_books_l156_156513


namespace solve_trig_eq_l156_156681

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l156_156681


namespace sheila_weekly_earnings_l156_156817

-- Defining the conditions
def hourly_wage : ℕ := 12
def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def hours_tt : ℕ := 6
def days_tt : ℕ := 2

-- Defining Sheila's total weekly earnings
noncomputable def weekly_earnings := (hours_mwf * hourly_wage * days_mwf) + (hours_tt * hourly_wage * days_tt)

-- The statement of the proof
theorem sheila_weekly_earnings : weekly_earnings = 432 :=
by
  sorry

end sheila_weekly_earnings_l156_156817


namespace remainder_of_3_pow_100_plus_5_mod_8_l156_156708

theorem remainder_of_3_pow_100_plus_5_mod_8 :
  (3^100 + 5) % 8 = 6 := by
sorry

end remainder_of_3_pow_100_plus_5_mod_8_l156_156708


namespace orthocenter_of_ABC_l156_156637

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨-1, 3, 2⟩
def B : Point3D := ⟨4, -2, 2⟩
def C : Point3D := ⟨2, -1, 6⟩

def orthocenter (A B C : Point3D) : Point3D :=
  -- formula to calculate the orthocenter
  sorry

theorem orthocenter_of_ABC :
  orthocenter A B C = ⟨101 / 150, 192 / 150, 232 / 150⟩ :=
by 
  -- proof steps
  sorry

end orthocenter_of_ABC_l156_156637


namespace only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l156_156218

theorem only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c
  (n a b c : ℕ) (hn : n > 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hca : c > a) (hcb : c > b) (hab : a ≤ b) :
  n * a + n * b = n * c ↔ (n = 2 ∧ b = a ∧ c = a + 1) := by
  sorry

end only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l156_156218


namespace sqrt_18_mul_sqrt_32_eq_24_l156_156423
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l156_156423


namespace cost_of_shoes_l156_156042

-- Define the conditions
def saved : Nat := 30
def earn_per_lawn : Nat := 5
def lawns_per_weekend : Nat := 3
def weekends_needed : Nat := 6

-- Prove the total amount saved is the cost of the shoes
theorem cost_of_shoes : saved + (earn_per_lawn * lawns_per_weekend * weekends_needed) = 120 := by
  sorry

end cost_of_shoes_l156_156042


namespace arithmetic_and_geometric_mean_l156_156815

theorem arithmetic_and_geometric_mean (x y : ℝ) (h1: (x + y) / 2 = 20) (h2: Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 :=
sorry

end arithmetic_and_geometric_mean_l156_156815


namespace marge_funds_for_fun_l156_156404

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end marge_funds_for_fun_l156_156404


namespace triangle_area_from_altitudes_l156_156464

noncomputable def triangleArea (altitude1 altitude2 altitude3 : ℝ) : ℝ :=
  sorry

theorem triangle_area_from_altitudes
  (h1 : altitude1 = 15)
  (h2 : altitude2 = 21)
  (h3 : altitude3 = 35) :
  triangleArea 15 21 35 = 245 * Real.sqrt 3 :=
sorry

end triangle_area_from_altitudes_l156_156464


namespace car_speed_conversion_l156_156172

noncomputable def miles_to_yards : ℕ :=
  1760

theorem car_speed_conversion (speed_mph : ℕ) (time_sec : ℝ) (distance_yards : ℕ) :
  speed_mph = 90 →
  time_sec = 0.5 →
  distance_yards = 22 →
  (1 : ℕ) * miles_to_yards = 1760 := by
  intros h1 h2 h3
  sorry

end car_speed_conversion_l156_156172


namespace paul_tickets_left_l156_156193

theorem paul_tickets_left (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) :
  initial_tickets = 11 → spent_tickets = 3 → remaining_tickets = initial_tickets - spent_tickets → remaining_tickets = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end paul_tickets_left_l156_156193


namespace sum_positive_implies_at_least_one_positive_l156_156300

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l156_156300


namespace car_catches_truck_in_7_hours_l156_156017

-- Definitions based on the conditions
def initial_distance := 175 -- initial distance in kilometers
def truck_speed := 40 -- speed of the truck in km/h
def car_initial_speed := 50 -- initial speed of the car in km/h
def car_speed_increase := 5 -- speed increase per hour for the car in km/h

-- The main statement to prove
theorem car_catches_truck_in_7_hours :
  ∃ n : ℕ, (n ≥ 0) ∧ 
  (car_initial_speed - truck_speed) * n + (car_speed_increase * n * (n - 1) / 2) = initial_distance :=
by
  existsi 7
  -- Check the equation for n = 7
  -- Simplify: car initial extra speed + sum of increase terms should equal initial distance
  -- (50 - 40) * 7 + 5 * 7 * 6 / 2 = 175
  -- (10) * 7 + 35 * 3 / 2 = 175
  -- 70 + 105 = 175
  sorry

end car_catches_truck_in_7_hours_l156_156017


namespace a6_value_l156_156792

variable (a_n : ℕ → ℤ)

/-- Given conditions in the arithmetic sequence -/
def arithmetic_sequence_property (a_n : ℕ → ℤ) :=
  ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)

/-- Given sum condition a_4 + a_5 + a_6 + a_7 + a_8 = 150 -/
def sum_condition :=
  a_n 4 + a_n 5 + a_n 6 + a_n 7 + a_n 8 = 150

theorem a6_value (h : arithmetic_sequence_property a_n) (hsum : sum_condition a_n) :
  a_n 6 = 30 := 
by
  sorry

end a6_value_l156_156792


namespace mass_percentage_O_in_mixture_l156_156479

/-- Mass percentage of oxygen in a mixture of Acetone and Methanol -/
theorem mass_percentage_O_in_mixture 
  (mass_acetone: ℝ)
  (mass_methanol: ℝ)
  (mass_O_acetone: ℝ)
  (mass_O_methanol: ℝ) 
  (total_mass: ℝ) : 
  mass_acetone = 30 → 
  mass_methanol = 20 → 
  mass_O_acetone = (16 / 58.08) * 30 →
  mass_O_methanol = (16 / 32.04) * 20 →
  total_mass = mass_acetone + mass_methanol →
  ((mass_O_acetone + mass_O_methanol) / total_mass) * 100 = 36.52 :=
by
  sorry

end mass_percentage_O_in_mixture_l156_156479


namespace ratio_of_coefficients_l156_156652

theorem ratio_of_coefficients (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (H1 : 8 * x - 6 * y = c) (H2 : 12 * y - 18 * x = d) :
  c / d = -4 / 9 := 
by {
  sorry
}

end ratio_of_coefficients_l156_156652


namespace smallest_interior_angle_l156_156388

open Real

theorem smallest_interior_angle (A B C : ℝ) (hA : 0 < A ∧ A < π)
    (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
    (h_sum_angles : A + B + C = π)
    (h_ratio : sin A / sin B = 2 / sqrt 6 ∧ sin A / sin C = 2 / (sqrt 3 + 1)) :
    min A (min B C) = π / 4 := 
  by sorry

end smallest_interior_angle_l156_156388


namespace find_t_correct_l156_156487

theorem find_t_correct : 
  ∃ t : ℝ, (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 15) = 15 * x^4 - 47 * x^3 + 115 * x^2 - 110 * x + 75) ∧ t = -10 :=
sorry

end find_t_correct_l156_156487


namespace expression_value_l156_156457

theorem expression_value : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by {
  sorry
}

end expression_value_l156_156457


namespace trapezoid_base_length_sets_l156_156350

open Nat

theorem trapezoid_base_length_sets :
  ∃ (sets : Finset (ℕ × ℕ)), sets.card = 5 ∧ 
    (∀ p ∈ sets, ∃ (b1 b2 : ℕ), b1 = 10 * p.1 ∧ b2 = 10 * p.2 ∧ b1 + b2 = 90) :=
by
  sorry

end trapezoid_base_length_sets_l156_156350


namespace square_area_l156_156988

theorem square_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  ∃ (s : ℝ), (s * Real.sqrt 2 = d) ∧ (s^2 = 144) := by
  sorry

end square_area_l156_156988


namespace find_clubs_l156_156452

theorem find_clubs (S D H C : ℕ) (h1 : S + D + H + C = 13)
  (h2 : S + C = 7) 
  (h3 : D + H = 6) 
  (h4 : D = 2 * S) 
  (h5 : H = 2 * D) 
  : C = 6 :=
by
  sorry

end find_clubs_l156_156452


namespace total_toys_l156_156598

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end total_toys_l156_156598


namespace foma_should_give_ierema_55_coins_l156_156135

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156135


namespace solution_set_of_inequality_l156_156834

theorem solution_set_of_inequality {x : ℝ} :
  {x : ℝ | |2 - 3 * x| ≥ 4} = {x : ℝ | x ≤ -2 / 3 ∨ 2 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l156_156834


namespace fraction_of_total_cost_l156_156741

theorem fraction_of_total_cost
  (p_r : ℕ) (c_r : ℕ) 
  (p_a : ℕ) (c_a : ℕ)
  (p_c : ℕ) (c_c : ℕ)
  (p_w : ℕ) (c_w : ℕ)
  (p_dap : ℕ) (c_dap : ℕ)
  (p_dc : ℕ) (c_dc : ℕ)
  (total_cost : ℕ)
  (combined_cost_rc : ℕ)
  (fraction : ℚ)
  (h1 : p_r = 5) (h2 : c_r = 2)
  (h3 : p_a = 4) (h4 : c_a = 6)
  (h5 : p_c = 3) (h6 : c_c = 8)
  (h7 : p_w = 2) (h8 : c_w = 10)
  (h9 : p_dap = 4) (h10 : c_dap = 5)
  (h11 : p_dc = 3) (h12 : c_dc = 3)
  (htotal_cost : total_cost = 107)
  (hcombined_cost_rc : combined_cost_rc = 19)
  (hfraction : fraction = 19 / 107) :
  fraction = combined_cost_rc / total_cost := sorry

end fraction_of_total_cost_l156_156741


namespace simplify_fraction_l156_156196

theorem simplify_fraction (m : ℝ) (h : m ≠ 1) : (m / (m - 1) + 1 / (1 - m) = 1) :=
by {
  sorry
}

end simplify_fraction_l156_156196


namespace product_xyz_eq_one_l156_156782

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end product_xyz_eq_one_l156_156782


namespace width_of_park_l156_156184

theorem width_of_park (L : ℕ) (A_lawn : ℕ) (w_road : ℕ) (W : ℚ) :
  L = 60 → A_lawn = 2109 → w_road = 3 →
  60 * W - 2 * 60 * 3 = 2109 →
  W = 41.15 :=
by
  intros hL hA_lawn hw_road hEq
  -- The proof will go here
  sorry

end width_of_park_l156_156184


namespace intersection_eq_l156_156901

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | Real.log (1 - x) > 0 }

theorem intersection_eq : A ∩ B = Set.Icc (-1) 0 :=
by
  sorry

end intersection_eq_l156_156901


namespace arithmetic_mean_is_b_l156_156195

variable (x a b : ℝ)
variable (hx : x ≠ 0)
variable (hb : b ≠ 0)

theorem arithmetic_mean_is_b : (1 / 2 : ℝ) * ((x * b + a) / x + (x * b - a) / x) = b :=
by
  sorry

end arithmetic_mean_is_b_l156_156195


namespace euler_children_mean_age_l156_156121

-- Define the ages of each child
def ages : List ℕ := [8, 8, 8, 13, 13, 16]

-- Define the total number of children
def total_children := 6

-- Define the correct sum of ages
def total_sum_ages := 66

-- Define the correct answer (mean age)
def mean_age := 11

-- Prove that the mean (average) age of these children is 11
theorem euler_children_mean_age : (List.sum ages) / total_children = mean_age :=
by
  sorry

end euler_children_mean_age_l156_156121


namespace smallest_k_l156_156051

theorem smallest_k (n k : ℕ) (h1: 2000 < n) (h2: n < 3000)
  (h3: ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1) :
  k = 9 :=
by
  sorry

end smallest_k_l156_156051


namespace greatest_five_digit_number_sum_of_digits_l156_156260

def is_five_digit_number (n : ℕ) : Prop :=
  10000 <= n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * (n / 10000)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + (n / 10000)

theorem greatest_five_digit_number_sum_of_digits (M : ℕ) 
  (h1 : is_five_digit_number M) 
  (h2 : digits_product M = 210) :
  digits_sum M = 20 := 
sorry

end greatest_five_digit_number_sum_of_digits_l156_156260


namespace sin_120_eq_sqrt3_div_2_l156_156208

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l156_156208


namespace jason_fires_weapon_every_15_seconds_l156_156014

theorem jason_fires_weapon_every_15_seconds
    (flame_duration_per_fire : ℕ)
    (total_flame_duration_per_minute : ℕ)
    (seconds_per_minute : ℕ)
    (h1 : flame_duration_per_fire = 5)
    (h2 : total_flame_duration_per_minute = 20)
    (h3 : seconds_per_minute = 60) :
    seconds_per_minute / (total_flame_duration_per_minute / flame_duration_per_fire) = 15 := 
by
  sorry

end jason_fires_weapon_every_15_seconds_l156_156014


namespace rhombus_diagonal_length_l156_156283

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 25) (h2 : A = 250) (h3 : A = (d1 * d2) / 2) : d2 = 20 := 
by
  rw [h1, h2] at h3
  sorry

end rhombus_diagonal_length_l156_156283


namespace arcsin_one_half_eq_pi_six_l156_156611

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry -- Proof omitted

end arcsin_one_half_eq_pi_six_l156_156611


namespace bathroom_new_area_l156_156555

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end bathroom_new_area_l156_156555


namespace students_in_two_courses_l156_156289

def total_students := 400
def num_math_modelling := 169
def num_chinese_literacy := 158
def num_international_perspective := 145
def num_all_three := 30
def num_none := 20

theorem students_in_two_courses : 
  ∃ x y z, 
    (num_math_modelling + num_chinese_literacy + num_international_perspective - (x + y + z) + num_all_three + num_none = total_students) ∧
    (x + y + z = 32) := 
  by
  sorry

end students_in_two_courses_l156_156289


namespace number_of_team_members_l156_156037

theorem number_of_team_members (x x1 x2 : ℕ) (h₀ : x = x1 + x2) (h₁ : 3 * x1 + 4 * x2 = 33) : x = 6 :=
sorry

end number_of_team_members_l156_156037


namespace cookie_distribution_l156_156739

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end cookie_distribution_l156_156739


namespace sin_120_eq_half_l156_156209

theorem sin_120_eq_half :
  let Q := (-(Real.sqrt 3) / 2, 1 / 2) in -- coordinates for Q
  sin (120 * (Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_120_eq_half_l156_156209


namespace john_gets_30_cans_l156_156801

def normal_price : ℝ := 0.60
def total_paid : ℝ := 9.00

theorem john_gets_30_cans :
  (total_paid / normal_price) * 2 = 30 :=
by
  sorry

end john_gets_30_cans_l156_156801


namespace gcd_204_85_l156_156956

theorem gcd_204_85: Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l156_156956


namespace remainder_when_divided_by_x_add_1_l156_156568

def q (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_when_divided_by_x_add_1 :
  q 2 = 6 → q (-1) = 20 :=
by
  intro hq2
  sorry

end remainder_when_divided_by_x_add_1_l156_156568


namespace max_good_triplets_l156_156933

-- Define the problem's conditions
variables (k : ℕ) (h_pos : 0 < k)

-- The statement to be proven
theorem max_good_triplets : ∃ T, T = 12 * k ^ 4 := 
sorry

end max_good_triplets_l156_156933


namespace fractions_equivalent_under_scaling_l156_156477

theorem fractions_equivalent_under_scaling (a b d k x : ℝ) (h₀ : d ≠ 0) (h₁ : k ≠ 0) :
  (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x)) ↔ b = d :=
by sorry

end fractions_equivalent_under_scaling_l156_156477


namespace A_finishes_race_in_36_seconds_l156_156918

-- Definitions of conditions
def distance_A := 130 -- A covers a distance of 130 meters
def distance_B := 130 -- B covers a distance of 130 meters
def time_B := 45 -- B covers the distance in 45 seconds
def distance_B_lag := 26 -- A beats B by 26 meters

-- Statement to prove
theorem A_finishes_race_in_36_seconds : 
  ∃ t : ℝ, distance_A / t + distance_B_lag = distance_B / time_B := sorry

end A_finishes_race_in_36_seconds_l156_156918


namespace number_is_correct_l156_156076

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l156_156076


namespace hallway_length_l156_156176

theorem hallway_length (s t d : ℝ) (h1 : 3 * s * t = 12) (h2 : s * t = d - 12) : d = 16 :=
sorry

end hallway_length_l156_156176


namespace f_decreasing_f_odd_l156_156635

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b

axiom negativity (x : ℝ) (h_pos : 0 < x) : f x < 0

theorem f_decreasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intros x
  sorry

end f_decreasing_f_odd_l156_156635


namespace line_eq_l156_156753

theorem line_eq (x y : ℝ) (point eq_direction_vector) (h₀ : point = (3, -2))
    (h₁ : eq_direction_vector = (-5, 3)) :
    3 * x + 5 * y + 1 = 0 := by sorry

end line_eq_l156_156753


namespace gift_wrapping_combinations_l156_156982

theorem gift_wrapping_combinations :
    (10 * 3 * 4 * 5 = 600) :=
by
    sorry

end gift_wrapping_combinations_l156_156982


namespace hayden_earnings_l156_156903

theorem hayden_earnings 
  (wage_per_hour : ℕ) 
  (pay_per_ride : ℕ)
  (bonus_per_review : ℕ)
  (number_of_rides : ℕ)
  (hours_worked : ℕ)
  (gas_cost_per_gallon : ℕ)
  (gallons_of_gas : ℕ)
  (positive_reviews : ℕ)
  : wage_per_hour = 15 → 
    pay_per_ride = 5 → 
    bonus_per_review = 20 → 
    number_of_rides = 3 → 
    hours_worked = 8 → 
    gas_cost_per_gallon = 3 → 
    gallons_of_gas = 17 → 
    positive_reviews = 2 → 
    (hours_worked * wage_per_hour + number_of_rides * pay_per_ride + positive_reviews * bonus_per_review + gallons_of_gas * gas_cost_per_gallon) = 226 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Further proof processing with these assumptions
  sorry

end hayden_earnings_l156_156903


namespace range_of_a_l156_156914

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l156_156914


namespace linear_expressions_constant_multiple_l156_156052

theorem linear_expressions_constant_multiple 
    (a b c p q r : ℝ)
    (h : (a*x + p)^2 + (b*x + q)^2 = (c*x + r)^2) : 
    a*b ≠ 0 → p*q ≠ 0 → (a / b = p / q) :=
by
  -- Given: (ax + p)^2 + (bx + q)^2 = (cx + r)^2
  -- Prove: a / b = p / q, implying that A(x) and B(x) can be expressed as the constant times C(x)
  sorry

end linear_expressions_constant_multiple_l156_156052


namespace rent_for_additional_hour_l156_156468

theorem rent_for_additional_hour (x : ℝ) :
  (25 + 10 * x = 125) → (x = 10) :=
by 
  sorry

end rent_for_additional_hour_l156_156468


namespace green_marbles_l156_156179

theorem green_marbles :
  ∀ (total: ℕ) (blue: ℕ) (red: ℕ) (yellow: ℕ), 
  total = 164 →
  blue = total / 2 →
  red = total / 4 →
  yellow = 14 →
  (total - (blue + red + yellow)) = 27 :=
by
  intros total blue red yellow h_total h_blue h_red h_yellow
  sorry

end green_marbles_l156_156179


namespace sum_pos_implies_one_pos_l156_156302

theorem sum_pos_implies_one_pos (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 := 
sorry

end sum_pos_implies_one_pos_l156_156302


namespace smallest_points_2016_l156_156628

theorem smallest_points_2016 (n : ℕ) :
  n = 28225 →
  ∀ (points : Fin n → (ℤ × ℤ)),
  ∃ i j : Fin n, i ≠ j ∧
    let dist_sq := (points i).fst - (points j).fst ^ 2 + (points i).snd - (points j).snd ^ 2 
    ∃ k : ℤ, dist_sq = 2016 * k :=
by
  intro h points
  sorry

end smallest_points_2016_l156_156628


namespace six_digit_number_all_equal_l156_156258

open Nat

theorem six_digit_number_all_equal (n : ℕ) (h : n = 21) : 12 * n^2 + 12 * n + 11 = 5555 :=
by
  rw [h]  -- Substitute n = 21
  sorry  -- Omit the actual proof steps

end six_digit_number_all_equal_l156_156258


namespace numberOfBaseballBoxes_l156_156993

-- Given conditions as Lean definitions and assumptions
def numberOfBasketballBoxes : ℕ := 4
def basketballCardsPerBox : ℕ := 10
def baseballCardsPerBox : ℕ := 8
def cardsGivenToClassmates : ℕ := 58
def cardsLeftAfterGiving : ℕ := 22

def totalBasketballCards : ℕ := numberOfBasketballBoxes * basketballCardsPerBox
def totalCardsBeforeGiving : ℕ := cardsLeftAfterGiving + cardsGivenToClassmates

-- Target number of baseball cards
def totalBaseballCards : ℕ := totalCardsBeforeGiving - totalBasketballCards

-- Prove that the number of baseball boxes is 5
theorem numberOfBaseballBoxes :
  totalBaseballCards / baseballCardsPerBox = 5 :=
sorry

end numberOfBaseballBoxes_l156_156993


namespace expected_value_geometric_seq_p10_lt_q10_l156_156281

/- Probability for penalties saved -/
def P_save := 1 / 9
def prob_distrib (k : ℕ) : ℚ :=
  match k with
  | 0 => (8 / 9) ^ 3
  | 1 => 3 * (1 / 9) * (8 / 9) ^ 2
  | 2 => 3 * (1 / 9) ^ 2 * (8 / 9)
  | 3 => (1 / 9) ^ 3
  | _ => 0 -- Since k ranges from 0 to 3

theorem expected_value : E (λ (X : ℕ), prob_distrib X) = 1 / 3 := sorry

/- Geometric sequence formation -/
def p (n : ℕ) : ℚ
| 0 => 1
| 1 => 0
| n + 2 => -1 / 2 * (p (n + 1)) + 1 / 2

theorem geometric_seq : ∀ n, (p n - 1 / 3) = (p 0 - 1 / 3) * (-1 / 2) ^ n := sorry

/- Comparison between p₁₀ and q₁₀ -/
def q (n : ℕ) : ℚ := 1 / 2 * (1 - p n)

theorem p10_lt_q10 : p 10 < q 10 := sorry

end expected_value_geometric_seq_p10_lt_q10_l156_156281


namespace interest_rate_calculation_l156_156867

theorem interest_rate_calculation (P1 P2 I1 I2 : ℝ) (r1 : ℝ) :
  P2 = 1648 ∧ P1 = 2678 - P2 ∧ I2 = P2 * 0.05 * 3 ∧ I1 = P1 * r1 * 8 ∧ I1 = I2 →
  r1 = 0.03 :=
by sorry

end interest_rate_calculation_l156_156867


namespace find_original_number_l156_156297

theorem find_original_number (k : ℤ) (h : 25 * k = N + 4) : ∃ N, N = 21 :=
by
  sorry

end find_original_number_l156_156297


namespace problem_statement_l156_156496

-- We begin by stating the variables x and y with the given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : x - 2 * y = 3
axiom h2 : (x - 2) * (y + 1) = 2

-- The theorem to prove
theorem problem_statement : (x^2 - 2) * (2 * y^2 - 1) = -9 :=
by
  sorry

end problem_statement_l156_156496


namespace perfect_square_m_value_l156_156290

theorem perfect_square_m_value (M X : ℤ) (hM : M > 1) (hX_lt_max : X < 8000) (hX_gt_min : 1000 < X) (hX_eq : X = M^3) : 
  (∃ M : ℤ, M > 1 ∧ 1000 < M^3 ∧ M^3 < 8000 ∧ (∃ k : ℤ, X = k * k) ∧ M = 16) :=
by
  use 16
  -- Here, we would normally provide the proof steps to show that 1000 < 16^3 < 8000 and 16^3 is a perfect square
  sorry

end perfect_square_m_value_l156_156290


namespace sofia_total_time_l156_156943

def distance1 : ℕ := 150
def speed1 : ℕ := 5
def distance2 : ℕ := 150
def speed2 : ℕ := 6
def laps : ℕ := 8
def time_per_lap := (distance1 / speed1) + (distance2 / speed2)
def total_time := 440  -- 7 minutes and 20 seconds in seconds

theorem sofia_total_time :
  laps * time_per_lap = total_time :=
by
  -- Proof steps are omitted and represented by sorry.
  sorry

end sofia_total_time_l156_156943


namespace sum_arithmetic_series_eq_499500_l156_156970

theorem sum_arithmetic_series_eq_499500 :
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  (n * (a1 + an) / 2) = 499500 := by {
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  show (n * (a1 + an) / 2) = 499500
  sorry
}

end sum_arithmetic_series_eq_499500_l156_156970


namespace volume_region_between_spheres_l156_156132

theorem volume_region_between_spheres 
    (r1 r2 : ℝ) 
    (h1 : r1 = 4) 
    (h2 : r2 = 7) 
    : 
    ( (4/3) * π * r2^3 - (4/3) * π * r1^3 ) = 372 * π := 
    sorry

end volume_region_between_spheres_l156_156132


namespace expected_value_of_coins_is_95_5_l156_156727

-- Define the individual coin values in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def fifty_cent_value : ℕ := 50
def dollar_value : ℕ := 100

-- Expected value function with 1/2 probability 
def expected_value (coin_value : ℕ) : ℚ := (coin_value : ℚ) / 2

-- Calculate the total expected value of all coins flipped
noncomputable def total_expected_value : ℚ :=
  expected_value penny_value +
  expected_value nickel_value +
  expected_value dime_value +
  expected_value quarter_value +
  expected_value fifty_cent_value +
  expected_value dollar_value

-- Prove that the expected total value is 95.5
theorem expected_value_of_coins_is_95_5 :
  total_expected_value = 95.5 := by
  sorry

end expected_value_of_coins_is_95_5_l156_156727


namespace brown_gumdrops_count_l156_156983

def gumdrops_conditions (total : ℕ) (blue : ℕ) (brown : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) : Prop :=
  total = blue + brown + red + yellow + green ∧
  blue = total * 25 / 100 ∧
  brown = total * 25 / 100 ∧
  red = total * 20 / 100 ∧
  yellow = total * 15 / 100 ∧
  green = 40 ∧
  green = total * 15 / 100

theorem brown_gumdrops_count: ∃ total blue brown red yellow green new_brown,
  gumdrops_conditions total blue brown red yellow green →
  new_brown = brown + blue / 3 →
  new_brown = 89 :=
by
  sorry

end brown_gumdrops_count_l156_156983


namespace problem_pf_qf_geq_f_pq_l156_156234

variable {R : Type*} [LinearOrderedField R]

theorem problem_pf_qf_geq_f_pq (f : R → R) (a b p q x y : R) (hpq : p + q = 1) :
  (∀ x y, p * f x + q * f y ≥ f (p * x + q * y)) ↔ (0 ≤ p ∧ p ≤ 1) := 
by
  sorry

end problem_pf_qf_geq_f_pq_l156_156234


namespace yao_ming_shots_l156_156522

-- Defining the conditions
def total_shots_made : ℕ := 14
def total_points_scored : ℕ := 28
def three_point_shots_made : ℕ := 3
def two_point_shots (x : ℕ) : ℕ := x
def free_throws_made (x : ℕ) : ℕ := total_shots_made - three_point_shots_made - x

-- The theorem we want to prove
theorem yao_ming_shots :
  ∃ (x y : ℕ),
    (total_shots_made = three_point_shots_made + x + y) ∧ 
    (total_points_scored = 3 * three_point_shots_made + 2 * x + y) ∧
    (x = 8) ∧
    (y = 3) :=
sorry

end yao_ming_shots_l156_156522


namespace fraction_ratios_l156_156239

theorem fraction_ratios (m n p q : ℕ) (h1 : (m : ℚ) / n = 18) (h2 : (p : ℚ) / n = 6) (h3 : (p : ℚ) / q = 1 / 15) :
  (m : ℚ) / q = 1 / 5 :=
sorry

end fraction_ratios_l156_156239


namespace count_even_strictly_increasing_integers_correct_l156_156774

-- Definition of condition: even four-digit integers with strictly increasing digits
def is_strictly_increasing {a b c d : ℕ} : Prop :=
1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ∈ {2, 4, 6, 8}

def count_even_strictly_increasing_integers : ℕ :=
(finset.range 10).choose 4.filter (λ l, is_strictly_increasing l.head l.nth 1 l.nth 2 l.nth 3).card

theorem count_even_strictly_increasing_integers_correct :
  count_even_strictly_increasing_integers = 46 := by
  sorry

end count_even_strictly_increasing_integers_correct_l156_156774


namespace Andre_final_price_l156_156412

theorem Andre_final_price :
  let treadmill_price := 1350
  let treadmill_discount_rate := 0.30
  let plate_price := 60
  let num_of_plates := 2
  let plate_discount_rate := 0.15
  let sales_tax_rate := 0.07
  let treadmill_discount := treadmill_price * treadmill_discount_rate
  let treadmill_discounted_price := treadmill_price - treadmill_discount
  let total_plate_price := plate_price * num_of_plates
  let plate_discount := total_plate_price * plate_discount_rate
  let plate_discounted_price := total_plate_price - plate_discount
  let total_price_before_tax := treadmill_discounted_price + plate_discounted_price
  let sales_tax := total_price_before_tax * sales_tax_rate
  let final_price := total_price_before_tax + sales_tax
  final_price = 1120.29 := 
by
  repeat { 
    sorry 
  }

end Andre_final_price_l156_156412


namespace sahil_machine_purchase_price_l156_156939

theorem sahil_machine_purchase_price
  (repair_cost : ℕ)
  (transportation_cost : ℕ)
  (selling_price : ℕ)
  (profit_percent : ℤ)
  (purchase_price : ℕ)
  (total_cost : ℕ)
  (profit_ratio : ℚ)
  (h1 : repair_cost = 5000)
  (h2 : transportation_cost = 1000)
  (h3 : selling_price = 30000)
  (h4 : profit_percent = 50)
  (h5 : total_cost = purchase_price + repair_cost + transportation_cost)
  (h6 : profit_ratio = profit_percent / 100)
  (h7 : selling_price = (1 + profit_ratio) * total_cost) :
  purchase_price = 14000 :=
by
  sorry

end sahil_machine_purchase_price_l156_156939


namespace largest_number_l156_156700

theorem largest_number (a b c : ℤ) 
  (h_sum : a + b + c = 67)
  (h_diff1 : c - b = 7)
  (h_diff2 : b - a = 3)
  : c = 28 :=
sorry

end largest_number_l156_156700


namespace infinite_integer_solutions_l156_156766

theorem infinite_integer_solutions (a b c k : ℤ) (D : ℤ) 
  (hD : D = b^2 - 4 * a * c) (hD_pos : D > 0) (hD_non_square : ¬ ∃ (n : ℤ), n^2 = D) 
  (hk_non_zero : k ≠ 0) :
  (∃ (x₀ y₀ : ℤ), a * x₀^2 + b * x₀ * y₀ + c * y₀^2 = k) →
  ∃ (f : ℤ → ℤ × ℤ), ∀ n : ℤ, a * (f n).1^2 + b * (f n).1 * (f n).2 + c * (f n).2^2 = k :=
by
  sorry

end infinite_integer_solutions_l156_156766


namespace probability_to_buy_ticket_l156_156857

def p : ℝ := 0.1
def q : ℝ := 0.9
def initial_money : ℝ := 20
def target_money : ℝ := 45
def ticket_cost : ℝ := 10
def prize : ℝ := 30

noncomputable def equation_lhs : ℝ := p^2 * (1 + 2 * q)
noncomputable def equation_rhs : ℝ := 1 - 2 * p * q^2

noncomputable def x2 : ℝ := equation_lhs / equation_rhs

theorem probability_to_buy_ticket : x2 = 0.033 := sorry

end probability_to_buy_ticket_l156_156857


namespace max_area_of_garden_l156_156658

theorem max_area_of_garden (l w : ℝ) (h : l + 2*w = 270) : l * w ≤ 9112.5 :=
sorry

end max_area_of_garden_l156_156658


namespace bird_cages_count_l156_156863

-- Definitions based on the conditions provided
def num_parrots_per_cage : ℕ := 2
def num_parakeets_per_cage : ℕ := 7
def total_birds_per_cage : ℕ := num_parrots_per_cage + num_parakeets_per_cage
def total_birds_in_store : ℕ := 54
def num_bird_cages : ℕ := total_birds_in_store / total_birds_per_cage

-- The proof we need to derive
theorem bird_cages_count : num_bird_cages = 6 := by
  sorry

end bird_cages_count_l156_156863


namespace exists_cube_number_divisible_by_six_in_range_l156_156623

theorem exists_cube_number_divisible_by_six_in_range :
  ∃ (y : ℕ), y > 50 ∧ y < 350 ∧ (∃ (n : ℕ), y = n^3) ∧ y % 6 = 0 :=
by 
  use 216
  sorry

end exists_cube_number_divisible_by_six_in_range_l156_156623


namespace ellipse_foci_distance_l156_156752

theorem ellipse_foci_distance :
  (∀ x y : ℝ, x^2 / 56 + y^2 / 14 = 8) →
  ∃ d : ℝ, d = 8 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_l156_156752


namespace students_play_basketball_l156_156081

theorem students_play_basketball 
  (total_students : ℕ)
  (cricket_players : ℕ)
  (both_players : ℕ)
  (total_students_eq : total_students = 880)
  (cricket_players_eq : cricket_players = 500)
  (both_players_eq : both_players = 220) 
  : ∃ B : ℕ, B = 600 :=
by
  sorry

end students_play_basketball_l156_156081


namespace area_of_triangle_LRK_l156_156086

noncomputable def area_triang_lrk (JL LM JP QM : ℝ) : ℝ :=
  let PQ := JL - (JP + QM) -- Given JL and length of JP and QM
  let ratio := PQ / JL -- Ratio of sides PQ to JL
  let height_TRK := LM * (JL / PQ) -- Compute scaled height of triangle RLK
  1/2 * JL * height_TRK -- Area of triangle RLK

theorem area_of_triangle_LRK (JL LM JP QM : ℝ) (h1 : JL = 8) (h2 : LM = 4) (h3 : JP = 2) (h4 : QM = 1) :
  area_triang_lrk JL LM JP QM = 25.6 :=
by
  rw [h1, h2, h3, h4]
  let PQ := JL - (JP + QM)
  have hPQ : PQ = 5 := by linarith
  let ratio := PQ / JL
  have hRatio : ratio = 5/8 := by norm_num [ratio, JL, PQ, hPQ]
  let height_TRK := LM * (JL / PQ)
  have hHeight_TRK : height_TRK = 6.4 := by norm_num [height_TRK, LM, JL, PQ, hPQ]
  suffices : 1 / 2 * JL * height_TRK = 25.6 by
    exact this
  norm_num [JL, height_TRK, hHeight_TRK]
  sorry

end area_of_triangle_LRK_l156_156086


namespace probability_even_sum_of_spins_l156_156447

theorem probability_even_sum_of_spins :
  let prob_even_first := 3 / 6
  let prob_odd_first := 3 / 6
  let prob_even_second := 2 / 5
  let prob_odd_second := 3 / 5
  let prob_both_even := prob_even_first * prob_even_second
  let prob_both_odd := prob_odd_first * prob_odd_second
  prob_both_even + prob_both_odd = 1 / 2 := 
by 
  sorry

end probability_even_sum_of_spins_l156_156447


namespace temperature_at_midnight_l156_156079

-- Define temperature in the morning
def T_morning := -2 -- in degrees Celsius

-- Temperature change at noon
def delta_noon := 12 -- in degrees Celsius

-- Temperature change at midnight
def delta_midnight := -8 -- in degrees Celsius

-- Function to compute temperature
def compute_temperature (T : ℤ) (delta1 : ℤ) (delta2 : ℤ) : ℤ :=
  T + delta1 + delta2

-- The proposition to prove
theorem temperature_at_midnight :
  compute_temperature T_morning delta_noon delta_midnight = 2 :=
by
  sorry

end temperature_at_midnight_l156_156079


namespace inverse_B_squared_l156_156371

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -3, 2],
    ![  1, -1 ]]

theorem inverse_B_squared :
  B⁻¹ = B_inv →
  (B^2)⁻¹ = B_inv * B_inv :=
by sorry

end inverse_B_squared_l156_156371


namespace percent_of_ducks_among_non_swans_l156_156094

theorem percent_of_ducks_among_non_swans
  (total_birds : ℕ) 
  (percent_ducks percent_swans percent_eagles percent_sparrows : ℕ)
  (h1 : percent_ducks = 40) 
  (h2 : percent_swans = 20) 
  (h3 : percent_eagles = 15) 
  (h4 : percent_sparrows = 25)
  (h_sum : percent_ducks + percent_swans + percent_eagles + percent_sparrows = 100) :
  (percent_ducks * 100) / (100 - percent_swans) = 50 :=
by
  sorry

end percent_of_ducks_among_non_swans_l156_156094


namespace f_increasing_l156_156437

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end f_increasing_l156_156437


namespace sqrt_mul_simplify_l156_156427

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l156_156427


namespace point_in_third_quadrant_l156_156500

section quadrant_problem

variables (a b : ℝ)

-- Given: Point (a, b) is in the fourth quadrant
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- To prove: Point (a / b, 2 * b - a) is in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- The theorem stating that if (a, b) is in the fourth quadrant,
-- then (a / b, 2 * b - a) is in the third quadrant
theorem point_in_third_quadrant (a b : ℝ) (h : in_fourth_quadrant a b) :
  in_third_quadrant (a / b) (2 * b - a) :=
  sorry

end quadrant_problem

end point_in_third_quadrant_l156_156500


namespace determine_parameters_l156_156053

theorem determine_parameters
(eq_poly : ∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c) :
  ({ -1, 1 } : set ℝ) = { x : ℝ | x^5 + 2*x^4 + a*x^2 + b*x = c } →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by 
  -- Proof can go here
  sorry

end determine_parameters_l156_156053


namespace quadrilateral_is_parallelogram_l156_156389

theorem quadrilateral_is_parallelogram
  (A B C D : Type)
  (angle_DAB angle_ABC angle_BAD angle_DCB : ℝ)
  (h1 : angle_DAB = 135)
  (h2 : angle_ABC = 45)
  (h3 : angle_BAD = 45)
  (h4 : angle_DCB = 45) :
  (A B C D : Type) → Prop :=
by
  -- Definitions and conditions are given.
  sorry

end quadrilateral_is_parallelogram_l156_156389


namespace values_satisfying_ggx_eq_gx_l156_156808

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem values_satisfying_ggx_eq_gx (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 1 ∨ x = 3 ∨ x = 4 :=
by
  -- The proof is omitted
  sorry

end values_satisfying_ggx_eq_gx_l156_156808


namespace abs_val_of_minus_two_and_half_l156_156282

-- Definition of the absolute value function for real numbers
def abs_val (x : ℚ) : ℚ := if x < 0 then -x else x

-- Prove that the absolute value of -2.5 (which is -5/2) is equal to 2.5 (which is 5/2)
theorem abs_val_of_minus_two_and_half : abs_val (-5/2) = 5/2 := by
  sorry

end abs_val_of_minus_two_and_half_l156_156282


namespace integer_for_all_n_l156_156078

theorem integer_for_all_n
  (x y : ℝ)
  (f : ℕ → ℤ)
  (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → f n = ((x^n - y^n) / (x - y))) :
  ∀ n : ℕ, 0 < n → f n = ((x^n - y^n) / (x - y)) :=
by sorry

end integer_for_all_n_l156_156078


namespace yellow_block_heavier_than_green_l156_156659

theorem yellow_block_heavier_than_green :
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  yellow_block_weight - green_block_weight = 0.2 := by
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  show yellow_block_weight - green_block_weight = 0.2
  sorry

end yellow_block_heavier_than_green_l156_156659


namespace garden_area_proof_l156_156859

def length_rect : ℕ := 20
def width_rect : ℕ := 18
def area_rect : ℕ := length_rect * width_rect

def side_square1 : ℕ := 4
def area_square1 : ℕ := side_square1 * side_square1

def side_square2 : ℕ := 5
def area_square2 : ℕ := side_square2 * side_square2

def area_remaining : ℕ := area_rect - area_square1 - area_square2

theorem garden_area_proof : area_remaining = 319 := by
  sorry

end garden_area_proof_l156_156859


namespace probability_all_selected_l156_156162

variables (P_Ram P_Ravi P_Rina : ℚ)

theorem probability_all_selected (hRam : P_Ram = 4/7) (hRavi : P_Ravi = 1/5) (hRina : P_Rina = 3/8) :
  P_Ram * P_Ravi * P_Rina = 3/70 :=
by
  -- Given conditions are already stated.
  -- Proof will be provided to complete the theorem.
  sorry

end probability_all_selected_l156_156162


namespace johns_weekly_earnings_after_raise_l156_156929

theorem johns_weekly_earnings_after_raise 
  (original_weekly_earnings : ℕ) 
  (percentage_increase : ℝ) 
  (new_weekly_earnings : ℕ)
  (h1 : original_weekly_earnings = 60)
  (h2 : percentage_increase = 0.16666666666666664) :
  new_weekly_earnings = 70 :=
sorry

end johns_weekly_earnings_after_raise_l156_156929


namespace sum_of_primes_1_to_20_l156_156000

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l156_156000


namespace four_digit_increasing_even_integers_l156_156776

theorem four_digit_increasing_even_integers : 
  let even_four_digit_strictly_increasing (n : ℕ) := 
    n >= 1000 ∧ n < 10000 ∧ (n % 2 = 0) ∧ (let d1 := n / 1000 % 10, 
                                                d2 := n / 100 % 10,
                                                d3 := n / 10 % 10,
                                                d4 := n % 10 in
                                            d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  in (finset.filter even_four_digit_strictly_increasing (finset.range 10000)).card = 46 :=
begin
  sorry
end

end four_digit_increasing_even_integers_l156_156776


namespace semicircle_radius_l156_156187

-- Definition of the problem conditions
variables (a h : ℝ) -- base and height of the triangle
variable (R : ℝ)    -- radius of the semicircle

-- Statement of the proof problem
theorem semicircle_radius (h_pos : 0 < h) (a_pos : 0 < a) 
(semicircle_condition : ∀ R > 0, a * (h - R) = 2 * R * h) : R = a * h / (a + 2 * h) :=
sorry

end semicircle_radius_l156_156187


namespace grain_output_scientific_notation_l156_156220

theorem grain_output_scientific_notation :
    682.85 * 10^6 = 6.8285 * 10^8 := 
by sorry

end grain_output_scientific_notation_l156_156220


namespace chord_intersects_inner_circle_probability_l156_156444

noncomputable def probability_of_chord_intersecting_inner_circle
  (radius_inner : ℝ) (radius_outer : ℝ)
  (chord_probability : ℝ) : Prop :=
  radius_inner = 3 ∧ radius_outer = 5 ∧ chord_probability = 0.205

theorem chord_intersects_inner_circle_probability :
  probability_of_chord_intersecting_inner_circle 3 5 0.205 :=
by {
  sorry
}

end chord_intersects_inner_circle_probability_l156_156444


namespace sum_primes_upto_20_l156_156010

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l156_156010


namespace range_of_a_l156_156400

open Set Real

def set_M (a : ℝ) : Set ℝ := { x | x * (x - a - 1) < 0 }
def set_N : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) : set_M a ⊆ set_N ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l156_156400


namespace recurring_six_denominator_l156_156948

theorem recurring_six_denominator :
  ∃ (d : ℕ), ∀ (S : ℚ), S = 0.6̅ → (∃ (n m : ℤ), S = n / m ∧ n.gcd m = 1 ∧ m = d) :=
by
  sorry

end recurring_six_denominator_l156_156948


namespace part1_part2_l156_156573

-- Define properties for the first part of the problem
def condition1 (weightA weightB : ℕ) : Prop :=
  weightA + weightB = 7500 ∧ weightA = 3 * weightB / 2

def question1_answer : Prop :=
  ∃ weightA weightB : ℕ, condition1 weightA weightB ∧ weightA = 4500 ∧ weightB = 3000

-- Combined condition for the second part of the problem scenarios
def condition2a (y : ℕ) : Prop := y ≤ 1800 ∧ 18 * y - 10 * y = 17400
def condition2b (y : ℕ) : Prop := 1800 < y ∧ y ≤ 3000 ∧ 18 * y - (15 * y - 9000) = 17400
def condition2c (y : ℕ) : Prop := y > 3000 ∧ 18 * y - (20 * y - 24000) = 17400

def question2_answer : Prop :=
  (∃ y : ℕ, condition2b y ∧ y = 2800) ∨ (∃ y : ℕ, condition2c y ∧ y = 3300)

-- The Lean statements for both parts of the problem
theorem part1 : question1_answer := sorry

theorem part2 : question2_answer := sorry

end part1_part2_l156_156573


namespace eventually_periodic_l156_156127

variable (u : ℕ → ℤ)

def bounded (u : ℕ → ℤ) : Prop :=
  ∃ (m M : ℤ), ∀ (n : ℕ), m ≤ u n ∧ u n ≤ M

def recurrence (u : ℕ → ℤ) (n : ℕ) : Prop := 
  u (n) = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

theorem eventually_periodic (hu_bounded : bounded u) (hu_recurrence : ∀ n ≥ 4, recurrence u n) :
  ∃ N M, ∀ k ≥ 0, u (N + k) = u (N + M + k) :=
sorry

end eventually_periodic_l156_156127


namespace quadratic_root_3_m_value_l156_156380

theorem quadratic_root_3_m_value (m : ℝ) : (∃ x : ℝ, 2*x*x - m*x + 3 = 0 ∧ x = 3) → m = 7 :=
by
  sorry

end quadratic_root_3_m_value_l156_156380


namespace sum_of_primes_between_1_and_20_l156_156007

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l156_156007


namespace same_solution_for_equations_l156_156436

theorem same_solution_for_equations (b x : ℝ) :
  (2 * x + 7 = 3) → 
  (b * x - 10 = -2) → 
  b = -4 :=
by
  sorry

end same_solution_for_equations_l156_156436


namespace church_full_capacity_l156_156560

theorem church_full_capacity
  (chairs_per_row : ℕ)
  (rows : ℕ)
  (people_per_chair : ℕ)
  (h1 : chairs_per_row = 6)
  (h2 : rows = 20)
  (h3 : people_per_chair = 5) :
  (chairs_per_row * rows * people_per_chair) = 600 := by
  sorry

end church_full_capacity_l156_156560


namespace remainder_of_division_l156_156413

theorem remainder_of_division (d : ℝ) (q : ℝ) (r : ℝ) : 
  d = 187.46067415730337 → q = 89 → 16698 = (d * q) + r → r = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  sorry

end remainder_of_division_l156_156413


namespace denominator_of_repeating_six_l156_156951

theorem denominator_of_repeating_six : ∃ d : ℕ, (0.6 : ℚ) = ((2 : ℚ) / 3) → d = 3 :=
begin
  sorry
end

end denominator_of_repeating_six_l156_156951


namespace sin_120_eq_sqrt3_div_2_l156_156200

theorem sin_120_eq_sqrt3_div_2 (θ : ℝ) (h₁ : sin (180 - θ) = sin θ) (h₂ : sin (60 : ℝ) = real.sqrt 3 / 2) : 
  sin (120 : ℝ) = real.sqrt 3 / 2 := 
  sorry

end sin_120_eq_sqrt3_div_2_l156_156200


namespace no_int_sol_eq_l156_156813

theorem no_int_sol_eq (x y z : ℤ) (h₀ : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : ¬ (x^2 + y^2 = 3 * z^2) := 
sorry

end no_int_sol_eq_l156_156813


namespace sin_120_eq_sqrt3_div_2_l156_156204

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l156_156204


namespace find_speed_of_first_train_l156_156563

variable (L1 L2 : ℝ) (V1 V2 : ℝ) (t : ℝ)

theorem find_speed_of_first_train (hL1 : L1 = 100) (hL2 : L2 = 200) (hV2 : V2 = 30) (ht: t = 14.998800095992321) :
  V1 = 42.005334224 := by
  -- Proof to be completed
  sorry

end find_speed_of_first_train_l156_156563


namespace distance_travelled_l156_156911

theorem distance_travelled (t : ℝ) (h : 15 * t = 10 * t + 20) : 10 * t = 40 :=
by
  have ht : t = 4 := by linarith
  rw [ht]
  norm_num

end distance_travelled_l156_156911


namespace value_is_200_l156_156070

variable (x value : ℝ)
variable (h1 : 0.20 * x = value)
variable (h2 : 1.20 * x = 1200)

theorem value_is_200 : value = 200 :=
by
  sorry

end value_is_200_l156_156070


namespace distinct_ordered_pairs_l156_156906

/-- There are 9 distinct ordered pairs of positive integers (m, n) such that the sum of the 
    reciprocals of m and n equals 1/6. -/
theorem distinct_ordered_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), s.card = 9 ∧ 
  ∀ (p : ℕ × ℕ), p ∈ s → 
    (0 < p.1 ∧ 0 < p.2) ∧ 
    (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6) :=
sorry

end distinct_ordered_pairs_l156_156906


namespace smallest_integer_x_l156_156709

theorem smallest_integer_x (x : ℤ) (h : x < 3 * x - 12) : x ≥ 7 :=
sorry

end smallest_integer_x_l156_156709


namespace necessary_and_sufficient_condition_l156_156060

theorem necessary_and_sufficient_condition (t : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
    (∀ n, S n = n^2 + 5*n + t) →
    (t = 0 ↔ (∀ n, a n = 2*n + 4 ∧ (n > 0 → a n = S n - S (n - 1)))) :=
by
  sorry

end necessary_and_sufficient_condition_l156_156060


namespace number_is_correct_l156_156075

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l156_156075


namespace geometric_seq_xyz_eq_neg_two_l156_156360

open Real

noncomputable def geometric_seq (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_seq_xyz_eq_neg_two (x y z : ℝ) :
  geometric_seq (-1) x y z (-2) → x * y * z = -2 :=
by
  intro h
  obtain ⟨r, hx, hy, hz, he⟩ := h
  rw [hx, hy, hz, he] at *
  sorry

end geometric_seq_xyz_eq_neg_two_l156_156360


namespace determinant_transformation_l156_156238

theorem determinant_transformation 
  (a b c d : ℝ)
  (h : a * d - b * c = 6) :
  (a * (5 * c + 2 * d) - c * (5 * a + 2 * b)) = 12 := by
  sorry

end determinant_transformation_l156_156238


namespace adjacent_girl_pairs_l156_156559

variable (boyCount girlCount : ℕ) 
variable (adjacentBoyPairs adjacentGirlPairs: ℕ)

theorem adjacent_girl_pairs
  (h1 : boyCount = 10)
  (h2 : girlCount = 15)
  (h3 : adjacentBoyPairs = 5) :
  adjacentGirlPairs = 10 :=
sorry

end adjacent_girl_pairs_l156_156559


namespace steve_num_nickels_l156_156944

-- Definitions for the conditions
def num_nickels (N : ℕ) : Prop :=
  ∃ D Q : ℕ, D = N + 4 ∧ Q = D + 3 ∧ 5 * N + 10 * D + 25 * Q + 5 = 380

-- Statement of the problem
theorem steve_num_nickels : num_nickels 4 :=
sorry

end steve_num_nickels_l156_156944


namespace neg_or_false_implies_or_true_l156_156517

theorem neg_or_false_implies_or_true (p q : Prop) (h : ¬(p ∨ q) = False) : p ∨ q :=
by {
  sorry
}

end neg_or_false_implies_or_true_l156_156517


namespace sin_120_eq_sqrt3_div_2_l156_156203

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l156_156203


namespace overall_percentage_of_favor_l156_156025

theorem overall_percentage_of_favor
    (n_starting : ℕ)
    (n_experienced : ℕ)
    (perc_starting_favor : ℝ)
    (perc_experienced_favor : ℝ)
    (in_favor_from_starting : ℕ)
    (in_favor_from_experienced : ℕ)
    (total_surveyed : ℕ)
    (total_in_favor : ℕ)
    (overall_percentage : ℝ) :
    n_starting = 300 →
    n_experienced = 500 →
    perc_starting_favor = 0.40 →
    perc_experienced_favor = 0.70 →
    in_favor_from_starting = 120 →
    in_favor_from_experienced = 350 →
    total_surveyed = 800 →
    total_in_favor = 470 →
    overall_percentage = (470 / 800) * 100 →
    overall_percentage = 58.75 :=
by
  sorry

end overall_percentage_of_favor_l156_156025


namespace necessary_sufficient_condition_l156_156439

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
by
  sorry

end necessary_sufficient_condition_l156_156439


namespace percentage_difference_l156_156237

theorem percentage_difference : (0.4 * 60 - (4/5 * 25)) = 4 := by
  sorry

end percentage_difference_l156_156237


namespace rented_apartment_years_l156_156799

-- Given conditions
def months_in_year := 12
def payment_first_3_years_per_month := 300
def payment_remaining_years_per_month := 350
def total_paid := 19200
def first_period_years := 3

-- Define the total payment calculation
def total_payment (additional_years: ℕ): ℕ :=
  (first_period_years * months_in_year * payment_first_3_years_per_month) + 
  (additional_years * months_in_year * payment_remaining_years_per_month)

-- Main theorem statement
theorem rented_apartment_years (additional_years: ℕ) :
  total_payment additional_years = total_paid → (first_period_years + additional_years) = 5 :=
by
  intros h
  -- This skips the proof
  sorry

end rented_apartment_years_l156_156799


namespace value_of_A_cos_alpha_plus_beta_l156_156899

noncomputable def f (A x : ℝ) : ℝ := A * Real.cos (x / 4 + Real.pi / 6)

theorem value_of_A {A : ℝ}
  (h1 : f A (Real.pi / 3) = Real.sqrt 2) :
  A = 2 := 
by
  sorry

theorem cos_alpha_plus_beta {α β : ℝ}
  (hαβ1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (hαβ2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h2 : f 2 (4*α + 4*Real.pi/3) = -30 / 17)
  (h3 : f 2 (4*β - 2*Real.pi/3) = 8 / 5) :
  Real.cos (α + β) = -13 / 85 :=
by
  sorry

end value_of_A_cos_alpha_plus_beta_l156_156899


namespace smallest_norm_value_l156_156537

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l156_156537


namespace sum_of_primes_between_1_and_20_l156_156006

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l156_156006


namespace mushroom_ratio_l156_156264

theorem mushroom_ratio (total_mushrooms safe_mushrooms uncertain_mushrooms : ℕ)
  (h_total : total_mushrooms = 32)
  (h_safe : safe_mushrooms = 9)
  (h_uncertain : uncertain_mushrooms = 5) :
  (total_mushrooms - safe_mushrooms - uncertain_mushrooms) / safe_mushrooms = 2 :=
by sorry

end mushroom_ratio_l156_156264


namespace equalize_foma_ierema_l156_156144

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l156_156144


namespace hannah_age_double_july_age_20_years_ago_l156_156013

/-- Define the current ages of July (J) and her husband (H) -/
def current_age_july : ℕ := 23
def current_age_husband : ℕ := 25

/-- Assertion that July's husband is 2 years older than her -/
axiom husband_older : current_age_husband = current_age_july + 2

/-- We denote the ages 20 years ago -/
def age_july_20_years_ago := current_age_july - 20
def age_hannah_20_years_ago := current_age_husband - 20 - 2 * (current_age_july - 20)

theorem hannah_age_double_july_age_20_years_ago :
  age_hannah_20_years_ago = 6 :=
by sorry

end hannah_age_double_july_age_20_years_ago_l156_156013


namespace solve_trig_eq_l156_156683

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l156_156683


namespace correct_option_B_l156_156110

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end correct_option_B_l156_156110


namespace sin_120_eq_sqrt3_div_2_l156_156202

theorem sin_120_eq_sqrt3_div_2 (θ : ℝ) (h₁ : sin (180 - θ) = sin θ) (h₂ : sin (60 : ℝ) = real.sqrt 3 / 2) : 
  sin (120 : ℝ) = real.sqrt 3 / 2 := 
  sorry

end sin_120_eq_sqrt3_div_2_l156_156202


namespace boys_and_girls_equal_l156_156473

theorem boys_and_girls_equal (m d M D : ℕ) (hm : m > 0) (hd : d > 0) (h1 : (M / m) ≠ (D / d)) (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d := 
sorry

end boys_and_girls_equal_l156_156473


namespace range_of_m_l156_156502

theorem range_of_m (m : ℝ) (h : 9 > m^2 ∧ m ≠ 0) : m ∈ Set.Ioo (-3) 0 ∨ m ∈ Set.Ioo 0 3 := 
sorry

end range_of_m_l156_156502


namespace expand_polynomial_l156_156878

noncomputable def polynomial_expansion : Prop :=
  ∀ (x : ℤ), (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18

theorem expand_polynomial : polynomial_expansion :=
by
  sorry

end expand_polynomial_l156_156878


namespace intersection_of_perpendicular_lines_l156_156292

theorem intersection_of_perpendicular_lines (x y : ℝ) : 
  (y = 3 * x + 4) ∧ (y = -1/3 * x + 4) → (x = 0 ∧ y = 4) :=
by
  sorry

end intersection_of_perpendicular_lines_l156_156292


namespace problem_solution_l156_156761

-- Definitions of the arithmetic sequence a_n and its common difference and first term
variables (a d : ℝ)

-- Definitions of arithmetic sequence conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

-- Required conditions for the proof
variables (h1 : d ≠ 0) (h2 : a ≠ 0)
variables (h3 : arithmetic_sequence a d 2 * arithmetic_sequence a d 8 = (arithmetic_sequence a d 4) ^ 2)

-- The target theorem to prove
theorem problem_solution : 
  (a + (a + 4 * d) + (a + 8 * d)) / ((a + d) + (a + 2 * d)) = 3 :=
sorry

end problem_solution_l156_156761


namespace trigonometric_identity_l156_156494

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end trigonometric_identity_l156_156494


namespace triangle_is_isosceles_l156_156920

theorem triangle_is_isosceles 
  (A B C : ℝ) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) 
  (h₀ : A + B + C = π) :
  (A = B) := 
sorry

end triangle_is_isosceles_l156_156920


namespace fraction_zero_implies_x_is_neg_2_l156_156382

theorem fraction_zero_implies_x_is_neg_2 {x : ℝ} 
  (h₁ : x^2 - 4 = 0)
  (h₂ : x^2 - 4 * x + 4 ≠ 0) 
  : x = -2 := 
by
  sorry

end fraction_zero_implies_x_is_neg_2_l156_156382


namespace cos_identity_example_l156_156372

theorem cos_identity_example (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 3 / 5) : Real.cos (Real.pi / 3 - α) = 3 / 5 := by
  sorry

end cos_identity_example_l156_156372


namespace workers_days_not_worked_l156_156166

theorem workers_days_not_worked (W N : ℕ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 :=
sorry

end workers_days_not_worked_l156_156166


namespace value_of_business_l156_156589

variable (business_value : ℝ) -- We are looking for the value of the business
variable (man_ownership_fraction : ℝ := 2/3) -- The fraction of the business the man owns
variable (sale_fraction : ℝ := 3/4) -- The fraction of the man's shares that were sold
variable (sale_amount : ℝ := 6500) -- The amount for which the fraction of the shares were sold

-- The main theorem we are trying to prove
theorem value_of_business (h1 : man_ownership_fraction = 2/3) (h2 : sale_fraction = 3/4) (h3 : sale_amount = 6500) :
    business_value = 39000 := 
sorry

end value_of_business_l156_156589


namespace part1_cos_A_part2_c_l156_156520

-- We define a triangle with sides a, b, c opposite to angles A, B, C respectively.
variables (a b c : ℝ) (A B C : ℝ)
-- Given conditions for the problem:
variable (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
variable (h_cos_sum : Real.cos B + Real.cos C = (2 * Real.sqrt 3) / 3)
variable (ha : a = 2 * Real.sqrt 3)

-- The first part of the problem statement proving cos A = 1/3 given the conditions.
theorem part1_cos_A : Real.cos A = 1 / 3 :=
by
  sorry

-- The second part of the problem statement proving c = 3 given the conditions.
theorem part2_c : c = 3 :=
by
  sorry

end part1_cos_A_part2_c_l156_156520


namespace gcd_180_270_450_l156_156963

theorem gcd_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 := by 
  sorry

end gcd_180_270_450_l156_156963


namespace david_course_hours_l156_156997

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l156_156997


namespace consecutive_integers_satisfy_inequality_l156_156418

theorem consecutive_integers_satisfy_inequality :
  ∀ (n m : ℝ), n + 1 = m ∧ n < Real.sqrt 26 ∧ Real.sqrt 26 < m → m + n = 11 :=
by
  sorry

end consecutive_integers_satisfy_inequality_l156_156418


namespace library_visits_l156_156533

theorem library_visits
  (william_visits_per_week : ℕ := 2)
  (jason_visits_per_week : ℕ := 4 * william_visits_per_week)
  (emma_visits_per_week : ℕ := 3 * jason_visits_per_week)
  (zoe_visits_per_week : ℕ := william_visits_per_week / 2)
  (chloe_visits_per_week : ℕ := emma_visits_per_week / 3)
  (jason_total_visits : ℕ := jason_visits_per_week * 8)
  (emma_total_visits : ℕ := emma_visits_per_week * 8)
  (zoe_total_visits : ℕ := zoe_visits_per_week * 8)
  (chloe_total_visits : ℕ := chloe_visits_per_week * 8)
  (total_visits : ℕ := jason_total_visits + emma_total_visits + zoe_total_visits + chloe_total_visits) :
  total_visits = 328 := by
  sorry

end library_visits_l156_156533


namespace cans_purchased_l156_156433

variable (N P T : ℕ)

theorem cans_purchased (N P T : ℕ) : N * (5 * (T - 1)) / P = 5 * N * (T - 1) / P :=
by
  sorry

end cans_purchased_l156_156433


namespace carnival_wait_time_l156_156198

theorem carnival_wait_time :
  ∀ (T : ℕ), 4 * 60 = 4 * 30 + T + 4 * 15 → T = 60 :=
by
  intro T
  intro h
  sorry

end carnival_wait_time_l156_156198


namespace constant_term_in_binomial_expansion_l156_156478

open Finset

-- Define the binomial coefficient to be used
def binomial (n k : ℕ) : ℕ := (finset.range k).prod (λ i, (n - i) / (i + 1))

-- Define the expansion term
def term (r : ℕ) : ℤ := binomial 8 r * (-2) ^ r

theorem constant_term_in_binomial_expansion :
  (x - 2 / x) ^ 8 = 1120 :=
sorry

end constant_term_in_binomial_expansion_l156_156478


namespace marge_funds_l156_156406

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end marge_funds_l156_156406


namespace geometric_sequence_sum_l156_156228

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n = (a 0) * q^n)
  (h2 : ∀ n, a n > a (n + 1))
  (h3 : a 2 + a 3 + a 4 = 28)
  (h4 : a 3 + 2 = (a 2 + a 4) / 2) :
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 63 :=
by {
  sorry
}

end geometric_sequence_sum_l156_156228


namespace problem1_problem2_l156_156849

theorem problem1 (n : ℕ) : 2 ≤ (1 + 1 / n) ^ n ∧ (1 + 1 / n) ^ n < 3 :=
sorry

theorem problem2 (n : ℕ) : (n / 3) ^ n < n! :=
sorry

end problem1_problem2_l156_156849


namespace cone_volume_is_3_6_l156_156319

-- Define the given conditions
def is_maximum_volume_cone_with_cutoff (cone_volume cutoff_volume : ℝ) : Prop :=
  cutoff_volume = 2 * cone_volume

def volume_difference (cutoff_volume cone_volume difference : ℝ) : Prop :=
  cutoff_volume - cone_volume = difference

-- The theorem to prove the volume of the cone
theorem cone_volume_is_3_6 
  (cone_volume cutoff_volume difference: ℝ)  
  (h1: is_maximum_volume_cone_with_cutoff cone_volume cutoff_volume)
  (h2: volume_difference cutoff_volume cone_volume 3.6) 
  : cone_volume = 3.6 :=
sorry

end cone_volume_is_3_6_l156_156319


namespace equalize_foma_ierema_l156_156156

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l156_156156


namespace lisa_investment_in_stocks_l156_156402

-- Definitions for the conditions
def total_investment (r : ℝ) : Prop := r + 7 * r = 200000
def stock_investment (r : ℝ) : ℝ := 7 * r

-- Given the conditions, we need to prove the amount invested in stocks
theorem lisa_investment_in_stocks (r : ℝ) (h : total_investment r) : stock_investment r = 175000 :=
by
  -- proof goes here
  sorry

end lisa_investment_in_stocks_l156_156402


namespace problem_min_value_l156_156499

theorem problem_min_value {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  (2 * a + b + c) ≥ 4 := 
  sorry

end problem_min_value_l156_156499


namespace license_plate_combinations_l156_156180

def number_of_license_plates : ℕ :=
  10^5 * 26^3 * 20

theorem license_plate_combinations :
  number_of_license_plates = 35152000000 := by
  -- Here's where the proof would go
  sorry

end license_plate_combinations_l156_156180


namespace sum_as_fraction_l156_156622

theorem sum_as_fraction :
  (0.1 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) = (13467 / 100000 : ℝ) :=
by
  sorry

end sum_as_fraction_l156_156622


namespace polynomial_divisible_by_cube_l156_156936

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := 
  n^2 * x^(n+2) - (2 * n^2 + 2 * n - 1) * x^(n+1) + (n + 1)^2 * x^n - x - 1

theorem polynomial_divisible_by_cube (n : ℕ) (h : n > 0) : 
  ∃ Q, P n x = (x - 1)^3 * Q :=
sorry

end polynomial_divisible_by_cube_l156_156936


namespace solve_x_from_equation_l156_156846

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end solve_x_from_equation_l156_156846


namespace range_of_m_in_inverse_proportion_function_l156_156913

theorem range_of_m_in_inverse_proportion_function (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (1 - m) / x > 0) ∧ (x < 0 → (1 - m) / x < 0))) ↔ m < 1 :=
by
  sorry

end range_of_m_in_inverse_proportion_function_l156_156913


namespace rectangular_garden_side_length_l156_156322

theorem rectangular_garden_side_length (a b : ℝ) (h1 : 2 * a + 2 * b = 60) (h2 : a * b = 200) (h3 : b = 10) : a = 20 :=
by
  sorry

end rectangular_garden_side_length_l156_156322


namespace total_cost_is_17_l156_156812

def taco_shells_cost : ℝ := 5
def bell_pepper_cost_per_unit : ℝ := 1.5
def bell_pepper_quantity : ℕ := 4
def meat_cost_per_pound : ℝ := 3
def meat_quantity : ℕ := 2

def total_spent : ℝ :=
  taco_shells_cost + (bell_pepper_cost_per_unit * bell_pepper_quantity) + (meat_cost_per_pound * meat_quantity)

theorem total_cost_is_17 : total_spent = 17 := 
  by sorry

end total_cost_is_17_l156_156812


namespace evaluate_expression_l156_156336

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 :=
by
  sorry

end evaluate_expression_l156_156336


namespace scientific_notation_of_1650000_l156_156969

theorem scientific_notation_of_1650000 : (1650000 : ℝ) = 1.65 * 10^6 := 
by {
  -- Proof goes here
  sorry
}

end scientific_notation_of_1650000_l156_156969


namespace correct_option_is_D_l156_156364

def p : Prop := 3 ≥ 3
def q : Prop := 3 > 4

theorem correct_option_is_D (hp : p) (hq : ¬ q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬ ¬ p :=
by
  sorry

end correct_option_is_D_l156_156364


namespace sqrt_37_range_l156_156047

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 :=
by
  sorry

end sqrt_37_range_l156_156047


namespace basketball_weight_l156_156673

variable (b c : ℝ)

theorem basketball_weight (h1 : 9 * b = 5 * c) (h2 : 3 * c = 75) : b = 125 / 9 :=
by
  sorry

end basketball_weight_l156_156673


namespace exists_2013_distinct_numbers_l156_156736

theorem exists_2013_distinct_numbers : 
  ∃ (a : ℕ → ℕ), 
    (∀ m n, m ≠ n → m < 2013 ∧ n < 2013 → (a m + a n) % (a m - a n) = 0) ∧
    (∀ k l, k < 2013 ∧ l < 2013 → (a k) ≠ (a l)) :=
sorry

end exists_2013_distinct_numbers_l156_156736


namespace triangle_areas_l156_156243

-- Define points based on the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Triangle DEF vertices
def D : Point := { x := 0, y := 4 }
def E : Point := { x := 6, y := 0 }
def F : Point := { x := 6, y := 5 }

-- Triangle GHI vertices
def G : Point := { x := 0, y := 8 }
def H : Point := { x := 0, y := 6 }
def I : Point := F  -- I and F are the same point

-- Auxiliary function to calculate area of a triangle given its vertices
def area (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- Prove that the areas are correct
theorem triangle_areas :
  area D E F = 15 ∧ area G H I = 6 :=
by
  sorry

end triangle_areas_l156_156243


namespace arithmetic_seq_problem_l156_156248

open Nat

def arithmetic_sequence (a : ℕ → ℚ) (a1 d : ℚ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_seq_problem :
  ∃ (a : ℕ → ℚ) (a1 d : ℚ),
    (arithmetic_sequence a a1 d) ∧
    (a 2 + a 3 + a 4 = 3) ∧
    (a 7 = 8) ∧
    (a 11 = 15) :=
  sorry

end arithmetic_seq_problem_l156_156248


namespace balls_in_boxes_ways_l156_156907

theorem balls_in_boxes_ways : ∃ (ways : ℕ), ways = 56 :=
by
  let n := 5
  let m := 4
  let ways := 56
  sorry

end balls_in_boxes_ways_l156_156907


namespace smallest_a_gcd_77_88_l156_156295

theorem smallest_a_gcd_77_88 :
  ∃ (a : ℕ), a > 0 ∧ (∀ b, b > 0 → b < a → (gcd b 77 > 1 ∧ gcd b 88 > 1) → false) ∧ gcd a 77 > 1 ∧ gcd a 88 > 1 ∧ a = 11 :=
by
  sorry

end smallest_a_gcd_77_88_l156_156295


namespace domain_of_function_l156_156952

theorem domain_of_function :
  {x : ℝ | x > 4 ∧ x ≠ 5} = (Set.Ioo 4 5 ∪ Set.Ioi 5) :=
by
  sorry

end domain_of_function_l156_156952


namespace complaints_over_3_days_l156_156824

theorem complaints_over_3_days
  (n : ℕ) (n_ss : ℕ) (n_both : ℕ) (total : ℕ)
  (h1 : n = 120)
  (h2 : n_ss = n + 1/3 * n)
  (h3 : n_both = n_ss + 0.20 * n_ss)
  (h4 : total = n_both * 3) :
  total = 576 :=
by
  sorry

end complaints_over_3_days_l156_156824


namespace discount_amount_l156_156661

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end discount_amount_l156_156661


namespace sqrt_23_parts_xy_diff_l156_156107

-- Problem 1: Integer and decimal parts of sqrt(23)
theorem sqrt_23_parts : ∃ (integer_part : ℕ) (decimal_part : ℝ), 
  integer_part = 4 ∧ decimal_part = Real.sqrt 23 - 4 ∧
  (integer_part : ℝ) + decimal_part = Real.sqrt 23 :=
by
  sorry

-- Problem 2: x - y for 9 + sqrt(3) = x + y with given conditions
theorem xy_diff : 
  ∀ (x y : ℝ), x = 10 → y = Real.sqrt 3 - 1 → x - y = 11 - Real.sqrt 3 :=
by
  sorry

end sqrt_23_parts_xy_diff_l156_156107


namespace foma_should_give_ierema_55_coins_l156_156140

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156140


namespace am_gm_inequality_l156_156261

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) ≥ a + b + c :=
  sorry

end am_gm_inequality_l156_156261


namespace simplify_and_evaluate_expression_l156_156274

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -2) :
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l156_156274


namespace part1_part2_l156_156633

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) :=
sorry

theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = a * b ∧ (|2 * a - 1| + |3 * b - 1| = 2 * Real.sqrt 6 + 3)) :=
sorry

end part1_part2_l156_156633


namespace factorize_x_squared_minus_25_l156_156050

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l156_156050


namespace janet_dresses_total_pockets_l156_156392

theorem janet_dresses_total_pockets :
  ∃ dresses pockets pocket_2 pocket_3,
  dresses = 24 ∧ 
  pockets = dresses / 2 ∧ 
  pocket_2 = pockets / 3 ∧ 
  pocket_3 = pockets - pocket_2 ∧ 
  (pocket_2 * 2 + pocket_3 * 3) = 32 := by
    sorry

end janet_dresses_total_pockets_l156_156392


namespace sales_growth_correct_equation_l156_156968

theorem sales_growth_correct_equation (x : ℝ) 
(sales_24th : ℝ) (total_sales_25th_26th : ℝ) 
(h_initial : sales_24th = 5000) (h_total : total_sales_25th_26th = 30000) :
  (5000 * (1 + x)) + (5000 * (1 + x)^2) = 30000 :=
sorry

end sales_growth_correct_equation_l156_156968


namespace train_length_is_correct_l156_156851

-- Define the given conditions and the expected result.
def train_speed_kmph : ℝ := 270
def time_seconds : ℝ := 5
def expected_length_meters : ℝ := 375

-- State the theorem to be proven.
theorem train_length_is_correct :
  (train_speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters := by
  sorry -- Proof is not required, so we use 'sorry'

end train_length_is_correct_l156_156851


namespace cylindrical_surface_area_increase_l156_156980

theorem cylindrical_surface_area_increase (x : ℝ) :
  (2 * Real.pi * (10 + x)^2 + 2 * Real.pi * (10 + x) * (5 + x) = 
   2 * Real.pi * 10^2 + 2 * Real.pi * 10 * (5 + x)) →
   (x = -10 + 5 * Real.sqrt 6 ∨ x = -10 - 5 * Real.sqrt 6) :=
by
  intro h
  sorry

end cylindrical_surface_area_increase_l156_156980


namespace triangle_properties_l156_156528

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (angle_A : A = 30) (angle_B : B = 45) (side_a : a = Real.sqrt 2) :
  b = 2 ∧ (1 / 2) * a * b * Real.sin (105 * Real.pi / 180) = (Real.sqrt 3 + 1) / 2 := by
sorry

end triangle_properties_l156_156528


namespace trapezoid_area_l156_156962

theorem trapezoid_area:
  let vert1 := (10, 10)
  let vert2 := (15, 15)
  let vert3 := (0, 15)
  let vert4 := (0, 10)
  let base1 := 10
  let base2 := 15
  let height := 5
  ∃ (area : ℝ), area = 62.5 := by
  sorry

end trapezoid_area_l156_156962


namespace convex_pentagons_from_15_points_l156_156343

theorem convex_pentagons_from_15_points : (Nat.choose 15 5) = 3003 := 
by
  sorry

end convex_pentagons_from_15_points_l156_156343


namespace solve_for_y_l156_156677

theorem solve_for_y (x y : ℝ) (h : 3 * x - 5 * y = 7) : y = (3 * x - 7) / 5 :=
sorry

end solve_for_y_l156_156677


namespace inradius_circumradius_l156_156417

variables {T : Type} [MetricSpace T]

theorem inradius_circumradius (K k : ℝ) (d r rho : ℝ) (triangle : T)
  (h1 : (k / K) = (rho / r))
  (h2 : k ≤ K / 2)
  (h3 : 2 * r * rho = r^2 - d^2)
  (h4 : d ≥ 0) :
  r ≥ 2 * rho :=
sorry

end inradius_circumradius_l156_156417


namespace minimum_toothpicks_removal_l156_156883

theorem minimum_toothpicks_removal
    (num_toothpicks : ℕ) 
    (num_triangles : ℕ) 
    (h1 : num_toothpicks = 40) 
    (h2 : num_triangles > 35) :
    ∃ (min_removal : ℕ), min_removal = 15 
    := 
    sorry

end minimum_toothpicks_removal_l156_156883


namespace smallest_number_l156_156690

-- Definitions of conditions for H, P, and S
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5
def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def satisfies_conditions_H (H : ℕ) : Prop :=
  is_cube (H / 2) ∧ is_fifth_power (H / 3) ∧ is_square (H / 5)

def satisfies_conditions_P (P A B C : ℕ) : Prop :=
  P / 2 = A^2 ∧ P / 3 = B^3 ∧ P / 5 = C^5

def satisfies_conditions_S (S D E F : ℕ) : Prop :=
  S / 2 = D^5 ∧ S / 3 = E^2 ∧ S / 5 = F^3

-- Main statement: Prove that P is the smallest number satisfying the conditions
theorem smallest_number (H P S A B C D E F : ℕ)
  (hH : satisfies_conditions_H H)
  (hP : satisfies_conditions_P P A B C)
  (hS : satisfies_conditions_S S D E F) :
  P ≤ H ∧ P ≤ S :=
  sorry

end smallest_number_l156_156690


namespace remainder_3_pow_9_div_5_l156_156285

theorem remainder_3_pow_9_div_5 : (3^9) % 5 = 3 := by
  sorry

end remainder_3_pow_9_div_5_l156_156285


namespace batsman_average_after_12th_innings_l156_156165

noncomputable def batsman_average (runs_in_12th_innings : ℕ) (average_increase : ℕ) (initial_average_after_11_innings : ℕ) : ℕ :=
initial_average_after_11_innings + average_increase

theorem batsman_average_after_12th_innings
(score_in_12th_innings : ℕ)
(average_increase : ℕ)
(initial_average_after_11_innings : ℕ)
(total_runs_after_11_innings := 11 * initial_average_after_11_innings)
(total_runs_after_12_innings := total_runs_after_11_innings + score_in_12th_innings)
(new_average_after_12_innings := total_runs_after_12_innings / 12)
:
score_in_12th_innings = 80 ∧ average_increase = 3 ∧ initial_average_after_11_innings = 44 → 
batsman_average score_in_12th_innings average_increase initial_average_after_11_innings = 47 := 
by
  -- skipping the actual proof for now
  sorry

end batsman_average_after_12th_innings_l156_156165


namespace average_score_in_all_matches_l156_156579

theorem average_score_in_all_matches (runs_match1_match2 : ℤ) (runs_other_matches : ℤ) (total_matches : ℤ) 
  (average1 : ℤ) (average2 : ℤ)
  (h1 : average1 = 40) (h2 : average2 = 10) (h3 : runs_match1_match2 = 2 * average1)
  (h4 : runs_other_matches = 3 * average2) (h5 : total_matches = 5) :
  ((runs_match1_match2 + runs_other_matches) / total_matches) = 22 := 
by
  sorry

end average_score_in_all_matches_l156_156579


namespace find_a8_l156_156366

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) = a n / n) (h2 : a 5 = 15) : a 8 = 24 :=
sorry

end find_a8_l156_156366


namespace find_constant_e_l156_156490

theorem find_constant_e {x y e : ℝ} : (x / (2 * y) = 3 / e) → ((7 * x + 4 * y) / (x - 2 * y) = 25) → (e = 2) :=
by
  intro h1 h2
  sorry

end find_constant_e_l156_156490


namespace mr_lee_harvested_apples_l156_156672

theorem mr_lee_harvested_apples :
  let number_of_baskets := 19
  let apples_per_basket := 25
  (number_of_baskets * apples_per_basket = 475) :=
by
  sorry

end mr_lee_harvested_apples_l156_156672


namespace log_equation_solution_l156_156877

theorem log_equation_solution (x : ℝ) (h : Real.log x + Real.log (x + 4) = Real.log (2 * x + 8)) : x = 2 :=
sorry

end log_equation_solution_l156_156877


namespace hyperbola_center_coordinates_l156_156219

-- Defining the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 16 - (2 * x - 1)^2 / 9 = 1

-- Stating the theorem to verify the center of the hyperbola
theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), (h = 1/2) ∧ (k = -2) ∧ 
    ∀ x y, hyperbola_eq x y ↔ ((y + 2)^2 / (4 / 3)^2 - (x - 1/2)^2 / (3 / 2)^2 = 1) :=
by sorry

end hyperbola_center_coordinates_l156_156219


namespace Marcy_sips_interval_l156_156403

theorem Marcy_sips_interval:
  ∀ (total_volume_ml sip_volume_ml total_time min_per_sip: ℕ),
  total_volume_ml = 2000 →
  sip_volume_ml = 40 →
  total_time = 250 →
  min_per_sip = total_time / (total_volume_ml / sip_volume_ml) →
  min_per_sip = 5 :=
by
  intros total_volume_ml sip_volume_ml total_time min_per_sip hv hs ht hm
  rw [hv, hs, ht] at hm
  simp at hm
  exact hm

end Marcy_sips_interval_l156_156403


namespace highest_place_joker_can_achieve_is_6_l156_156084

-- Define the total number of teams
def total_teams : ℕ := 16

-- Define conditions for points in football
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0

-- Condition definitions for Joker's performance in the tournament
def won_against_strong_teams (j k : ℕ) : Prop := j < k
def lost_against_weak_teams (j k : ℕ) : Prop := j > k

-- Define the performance of all teams
def teams (t : ℕ) := {n // n < total_teams}

-- Function to calculate Joker's points based on position k
def joker_points (k : ℕ) : ℕ := (total_teams - k) * points_win

theorem highest_place_joker_can_achieve_is_6 : ∃ k, k = 6 ∧ 
  (∀ j, 
    (j < k → won_against_strong_teams j k) ∧ 
    (j > k → lost_against_weak_teams j k) ∧
    (∃! p, p = joker_points k)) :=
by
  sorry

end highest_place_joker_can_achieve_is_6_l156_156084


namespace two_point_question_count_l156_156712

/-- Define the number of questions and points on the test,
    and prove that the number of 2-point questions is 30. -/
theorem two_point_question_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 := by
  sorry

end two_point_question_count_l156_156712


namespace prob1_prob2_prob3_prob4_l156_156609

theorem prob1 : (3^3)^2 = 3^6 := by
  sorry

theorem prob2 : (-4 * x * y^3) * (-2 * x^2) = 8 * x^3 * y^3 := by
  sorry

theorem prob3 : 2 * x * (3 * y - x^2) + 2 * x * x^2 = 6 * x * y := by
  sorry

theorem prob4 : (20 * x^3 * y^5 - 10 * x^4 * y^4 - 20 * x^3 * y^2) / (-5 * x^3 * y^2) = -4 * y^3 + 2 * x * y^2 + 4 := by
  sorry

end prob1_prob2_prob3_prob4_l156_156609


namespace prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l156_156355

-- Definitions of the entities involved
variables {L : Type} -- All lines
variables {P : Type} -- All planes

-- Relations
variables (perpendicular : L → P → Prop)
variables (parallel : P → P → Prop)

-- Conditions
variables (a b : L)
variables (α β : P)

-- Statements we want to prove
theorem prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha
  (H1 : parallel α β) 
  (H2 : perpendicular a β) : 
  perpendicular a α :=
  sorry

end prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l156_156355


namespace quotient_of_m_by_13_l156_156278

open BigOperators

def S : Finset (Fin 13) := {1, 4, 9, 3, 12, 10}

def m : ℕ := S.sum (λ x, (x : ℕ))

theorem quotient_of_m_by_13 :
  m / 13 = 3 :=
by
  sorry

end quotient_of_m_by_13_l156_156278


namespace average_messages_correct_l156_156925

-- Definitions for the conditions
def messages_monday := 220
def messages_tuesday := 1 / 2 * messages_monday
def messages_wednesday := 50
def messages_thursday := 50
def messages_friday := 50

-- Definition for the total and average messages
def total_messages := messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday
def average_messages := total_messages / 5

-- Statement to prove
theorem average_messages_correct : average_messages = 96 := 
by sorry

end average_messages_correct_l156_156925


namespace sample_size_correct_l156_156174

variable (total_employees young_employees middle_aged_employees elderly_employees young_in_sample sample_size : ℕ)

-- Conditions
def total_number_of_employees := 75
def number_of_young_employees := 35
def number_of_middle_aged_employees := 25
def number_of_elderly_employees := 15
def number_of_young_in_sample := 7
def stratified_sampling := true

-- The proof problem statement
theorem sample_size_correct :
  total_employees = total_number_of_employees ∧ 
  young_employees = number_of_young_employees ∧ 
  middle_aged_employees = number_of_middle_aged_employees ∧ 
  elderly_employees = number_of_elderly_employees ∧ 
  young_in_sample = number_of_young_in_sample ∧ 
  stratified_sampling → 
  sample_size = 15 := by sorry

end sample_size_correct_l156_156174


namespace calculate_expression_l156_156606

theorem calculate_expression (x : ℝ) (h₁ : x ≠ 5) (h₂ : x = 4) : (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  sorry

end calculate_expression_l156_156606


namespace max_students_distributing_items_l156_156853

-- Define the given conditions
def pens : Nat := 1001
def pencils : Nat := 910

-- Define the statement
theorem max_students_distributing_items :
  Nat.gcd pens pencils = 91 :=
by
  sorry

end max_students_distributing_items_l156_156853


namespace calc1_calc2_calc3_calc4_l156_156039

-- Proof problem definitions
theorem calc1 : 15 + (-22) = -7 := sorry

theorem calc2 : (-13) + (-8) = -21 := sorry

theorem calc3 : (-0.9) + 1.5 = 0.6 := sorry

theorem calc4 : (1 / 2) + (-2 / 3) = -1 / 6 := sorry

end calc1_calc2_calc3_calc4_l156_156039


namespace discount_amount_l156_156660

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end discount_amount_l156_156660


namespace train_length_l156_156588

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_head_start_m : ℝ := 240
noncomputable def train_passing_time_s : ℝ := 35.99712023038157

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def distance_covered_by_train : ℝ := relative_speed_mps * train_passing_time_s

theorem train_length :
  distance_covered_by_train - jogger_head_start_m = 119.9712023038157 :=
by
  sorry

end train_length_l156_156588


namespace divisor_of_2n_when_remainder_is_two_l156_156915

theorem divisor_of_2n_when_remainder_is_two (n : ℤ) (k : ℤ) : 
  (n = 22 * k + 12) → ∃ d : ℤ, d = 22 ∧ (2 * n) % d = 2 :=
by
  sorry

end divisor_of_2n_when_remainder_is_two_l156_156915


namespace max_p_l156_156656

theorem max_p (p q r s t u v w : ℕ)
  (h1 : p + q + r + s = 35)
  (h2 : q + r + s + t = 35)
  (h3 : r + s + t + u = 35)
  (h4 : s + t + u + v = 35)
  (h5 : t + u + v + w = 35)
  (h6 : q + v = 14) :
  p ≤ 20 :=
sorry

end max_p_l156_156656


namespace max_candies_ben_eats_l156_156287

theorem max_candies_ben_eats (total_candies : ℕ) (k : ℕ) (h_pos_k : k > 0) (b : ℕ) 
  (h_total : b + 2 * b + k * b = total_candies) (h_total_candies : total_candies = 30) : b = 6 :=
by
  -- placeholder for proof steps
  sorry

end max_candies_ben_eats_l156_156287


namespace radian_measure_of_200_degrees_l156_156959

theorem radian_measure_of_200_degrees :
  (200 : ℝ) * (Real.pi / 180) = (10 / 9) * Real.pi :=
sorry

end radian_measure_of_200_degrees_l156_156959


namespace walking_time_l156_156024

theorem walking_time 
  (speed_km_hr : ℝ := 10) 
  (distance_km : ℝ := 6) 
  : (distance_km / (speed_km_hr / 60)) = 36 :=
by
  sorry

end walking_time_l156_156024


namespace remaining_speed_20_kmph_l156_156194

theorem remaining_speed_20_kmph
  (D T : ℝ)
  (H1 : (2/3 * D) / (1/3 * T) = 80)
  (H2 : T = D / 40) :
  (D / 3) / (2/3 * T) = 20 :=
by 
  sorry

end remaining_speed_20_kmph_l156_156194


namespace sqrt_mul_simplify_l156_156428

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l156_156428


namespace unique_m_value_l156_156759

theorem unique_m_value : ∀ m : ℝ,
  (m ^ 2 - 5 * m + 6 = 0 ∧ m ^ 2 - 3 * m + 2 = 0) →
  (m ^ 2 - 3 * m + 2 = 2 * (m ^ 2 - 5 * m + 6)) →
  ((m ^ 2 - 5 * m + 6) * (m ^ 2 - 3 * m + 2) > 0) →
  m = 2 :=
by
  sorry

end unique_m_value_l156_156759


namespace successful_multiplications_in_one_hour_l156_156459

variable (multiplications_per_second : ℕ)
variable (error_rate_percentage : ℕ)

theorem successful_multiplications_in_one_hour
  (h1 : multiplications_per_second = 15000)
  (h2 : error_rate_percentage = 5)
  : (multiplications_per_second * 3600 * (100 - error_rate_percentage) / 100) 
    + (multiplications_per_second * 3600 * error_rate_percentage / 100) = 54000000 := by
  sorry

end successful_multiplications_in_one_hour_l156_156459


namespace exist_odd_distinct_integers_l156_156357

theorem exist_odd_distinct_integers (n : ℕ) (h1 : n % 2 = 1) (h2 : n > 3) (h3 : n % 3 ≠ 0) : 
  ∃ a b c : ℕ, a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  3 / (n : ℚ) = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) :=
sorry

end exist_odd_distinct_integers_l156_156357


namespace proof_volume_l156_156045

noncomputable def volume_set (a b c h r : ℝ) : ℝ := 
  let v_box := a * b * c
  let v_extensions := 2 * (a * b * h) + 2 * (a * c * h) + 2 * (b * c * h)
  let v_cylinder := Real.pi * r^2 * h
  let v_spheres := 8 * (1/6) * (Real.pi * r^3)
  v_box + v_extensions + v_cylinder + v_spheres

theorem proof_volume : 
  let a := 2; let b := 3; let c := 6
  let r := 2; let h := 3
  volume_set a b c h r = (540 + 48 * Real.pi) / 3 ∧ (540 + 48 + 3) = 591 :=
by 
  sorry

end proof_volume_l156_156045


namespace surface_area_of_circumscribing_sphere_l156_156639

theorem surface_area_of_circumscribing_sphere :
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  A = 17 * Real.pi :=
by
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  show A = 17 * Real.pi
  sorry

end surface_area_of_circumscribing_sphere_l156_156639


namespace initial_principal_amount_l156_156030

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_principal_amount :
  let P := 4410 / (compound_interest 1 0.07 4 2 * compound_interest 1 0.09 2 2)
  abs (P - 3238.78) < 0.01 :=
by
  sorry

end initial_principal_amount_l156_156030


namespace foma_should_give_ierema_l156_156160

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l156_156160


namespace solve_inequality_l156_156349

noncomputable def solutionSet := { x : ℝ | 0 < x ∧ x < 1 }

theorem solve_inequality (x : ℝ) : x^2 < x ↔ x ∈ solutionSet := 
sorry

end solve_inequality_l156_156349


namespace triangle_right_angle_l156_156088

theorem triangle_right_angle {a b c : ℝ} {A B C : ℝ} (h : a * Real.cos A + b * Real.cos B = c * Real.cos C) :
  (A = Real.pi / 2) ∨ (B = Real.pi / 2) ∨ (C = Real.pi / 2) :=
sorry

end triangle_right_angle_l156_156088


namespace compare_abc_l156_156358

noncomputable def a : ℝ := (1 / 4) * Real.logb 2 3
noncomputable def b : ℝ := 1 / 2
noncomputable def c : ℝ := (1 / 2) * Real.logb 5 3

theorem compare_abc : c < a ∧ a < b := sorry

end compare_abc_l156_156358


namespace simplify_fraction_l156_156114

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end simplify_fraction_l156_156114


namespace evaluate_expression_l156_156748

def cube_root (x : ℝ) := x^(1/3)

theorem evaluate_expression : (cube_root (9 / 32))^2 = (3/8) := 
by
  sorry

end evaluate_expression_l156_156748


namespace imo1983_q24_l156_156529

theorem imo1983_q24 :
  ∃ (S : Finset ℕ), S.card = 1983 ∧ 
    (∀ x ∈ S, x > 0 ∧ x ≤ 10^5) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z → (x + z ≠ 2 * y)) :=
sorry

end imo1983_q24_l156_156529


namespace maximum_value_l156_156828

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.sin x + 4 * Real.cos x

theorem maximum_value :
  (∃ x : ℝ, f(x) = 5) → 
  (∀ x : ℝ, f(x) ≤ 5) :=
sorry

end maximum_value_l156_156828


namespace fraction_equivalence_l156_156847

theorem fraction_equivalence : 
  (∀ (a b : ℕ), (a ≠ 0 ∧ b ≠ 0) → (15 * b = 25 * a ↔ a = 3 ∧ b = 5)) ∧
  (15 * 4 ≠ 25 * 3) ∧
  (15 * 3 ≠ 25 * 2) ∧
  (15 * 2 ≠ 25 * 1) ∧
  (15 * 7 ≠ 25 * 5) :=
by
  sorry

end fraction_equivalence_l156_156847


namespace total_toys_l156_156601

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end total_toys_l156_156601


namespace molecular_weight_constant_l156_156964

-- Define the molecular weight of bleach
def molecular_weight_bleach (num_moles : Nat) : Nat := 222

-- Theorem stating the molecular weight of any amount of bleach is 222 g/mol
theorem molecular_weight_constant (n : Nat) : molecular_weight_bleach n = 222 :=
by
  sorry

end molecular_weight_constant_l156_156964


namespace initial_ratio_l156_156922

def initial_men (M : ℕ) (W : ℕ) : Prop :=
  let men_after := M + 2
  let women_after := W - 3
  (2 * women_after = 24) ∧ (men_after = 14)

theorem initial_ratio (M W : ℕ) (h : initial_men M W) :
  (M = 12) ∧ (W = 15) → M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  intro hm hw
  have h12 : M = 12 := hm
  have h15 : W = 15 := hw
  sorry

end initial_ratio_l156_156922


namespace ella_probability_last_roll_l156_156910

theorem ella_probability_last_roll (n k : ℕ) (h₁ : n = 12) (h₂ : k = 2) :
  (∑ (i : ℕ) in finset.range 11, (5/6)^(i-1) * (1/6) * (5/6)^(10-i) * (1/6)) = 19531250 / 362797056 :=
by sorry

end ella_probability_last_roll_l156_156910


namespace positive_m_for_one_solution_l156_156223

theorem positive_m_for_one_solution :
  ∀ (m : ℝ), (∃ x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ 
  (∀ x y : ℝ, 9 * x^2 + m * x + 36 = 0 → 9 * y^2 + m * y + 36 = 0 → x = y) → m = 36 := 
by {
  sorry
}

end positive_m_for_one_solution_l156_156223


namespace comic_cost_is_4_l156_156545

-- Define initial amount of money Raul had.
def initial_money : ℕ := 87

-- Define number of comics bought by Raul.
def num_comics : ℕ := 8

-- Define the amount of money left after buying comics.
def money_left : ℕ := 55

-- Define the hypothesis condition about the money spent.
def total_spent : ℕ := initial_money - money_left

-- Define the main assertion that each comic cost $4.
def cost_per_comic (total_spent : ℕ) (num_comics : ℕ) : Prop :=
  total_spent / num_comics = 4

-- Main theorem statement
theorem comic_cost_is_4 : cost_per_comic total_spent num_comics :=
by
  -- Here we're skipping the proof for this exercise.
  sorry

end comic_cost_is_4_l156_156545


namespace diamond_problem_l156_156215

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_problem : (diamond (diamond 1 2) 3) - (diamond 1 (diamond 2 3)) = -7 / 30 := by
  sorry

end diamond_problem_l156_156215


namespace sphere_surface_area_increase_l156_156854

theorem sphere_surface_area_increase (r : ℝ) (h_r_pos : 0 < r):
  let A := 4 * π * r ^ 2
  let r' := 1.10 * r
  let A' := 4 * π * (r') ^ 2
  let ΔA := A' - A
  (ΔA / A) * 100 = 21 := by
  sorry

end sphere_surface_area_increase_l156_156854


namespace marys_score_l156_156408

def score (c w : ℕ) : ℕ := 30 + 4 * c - w
def valid_score_range (s : ℕ) : Prop := s > 90 ∧ s ≤ 170

theorem marys_score : ∃ c w : ℕ, c + w ≤ 35 ∧ score c w = 170 ∧ 
  ∀ (s : ℕ), (valid_score_range s ∧ ∃ c' w', score c' w' = s ∧ c' + w' ≤ 35) → 
  (s = 170) :=
by
  sorry

end marys_score_l156_156408


namespace measure_of_angle_C_l156_156869

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 7 * D) : C = 157.5 := 
by 
  sorry

end measure_of_angle_C_l156_156869


namespace cannot_be_value_of_omega_l156_156361

theorem cannot_be_value_of_omega (ω : ℤ) (φ : ℝ) (k n : ℤ) 
  (h1 : 0 < ω) 
  (h2 : |φ| < π / 2)
  (h3 : ω * (π / 12) + φ = k * π + π / 2)
  (h4 : -ω * (π / 6) + φ = n * π) : 
  ∀ m : ℤ, ω ≠ 4 * m := 
sorry

end cannot_be_value_of_omega_l156_156361


namespace foma_gives_ierema_55_l156_156150

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l156_156150


namespace four_digit_increasing_even_integers_l156_156775

theorem four_digit_increasing_even_integers : 
  let even_four_digit_strictly_increasing (n : ℕ) := 
    n >= 1000 ∧ n < 10000 ∧ (n % 2 = 0) ∧ (let d1 := n / 1000 % 10, 
                                                d2 := n / 100 % 10,
                                                d3 := n / 10 % 10,
                                                d4 := n % 10 in
                                            d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  in (finset.filter even_four_digit_strictly_increasing (finset.range 10000)).card = 46 :=
begin
  sorry
end

end four_digit_increasing_even_integers_l156_156775


namespace green_faction_lies_more_l156_156251

theorem green_faction_lies_more (r1 r2 r3 l1 l2 l3 : ℕ) 
  (h1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016) 
  (h2 : r1 + l2 + l3 = 1208) 
  (h3 : r2 + l1 + l3 = 908) 
  (h4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end green_faction_lies_more_l156_156251


namespace gumballs_ensure_four_same_color_l156_156320

-- Define the total number of gumballs in each color
def red_gumballs : ℕ := 10
def white_gumballs : ℕ := 9
def blue_gumballs : ℕ := 8
def green_gumballs : ℕ := 7

-- Define the minimum number of gumballs to ensure four of the same color
def min_gumballs_to_ensure_four_same_color : ℕ := 13

-- Prove that the minimum number of gumballs to ensure four of the same color is 13
theorem gumballs_ensure_four_same_color (n : ℕ) 
  (h₁ : red_gumballs ≥ 3)
  (h₂ : white_gumballs ≥ 3)
  (h₃ : blue_gumballs ≥ 3)
  (h₄ : green_gumballs ≥ 3)
  : n ≥ min_gumballs_to_ensure_four_same_color := 
sorry

end gumballs_ensure_four_same_color_l156_156320


namespace caroline_citrus_drinks_l156_156040

-- Definitions based on problem conditions
def citrus_drinks (oranges : ℕ) : ℕ := (oranges * 8) / 3

-- Define problem statement
theorem caroline_citrus_drinks : citrus_drinks 21 = 56 :=
by
  sorry

end caroline_citrus_drinks_l156_156040


namespace sum_of_abc_l156_156126

theorem sum_of_abc (a b c : ℕ) (h : a + b + c = 12) 
  (area_ratio : ℝ) (side_length_ratio : ℝ) 
  (ha : area_ratio = 50 / 98) 
  (hb : side_length_ratio = (Real.sqrt 50) / (Real.sqrt 98))
  (hc : side_length_ratio = (a * (Real.sqrt b)) / c) :
  a + b + c = 12 :=
by
  sorry

end sum_of_abc_l156_156126


namespace find_fg_minus_gf_l156_156279

def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 6 * x^2 + 12 * x + 11 := 
by 
  sorry

end find_fg_minus_gf_l156_156279


namespace total_lotus_flowers_l156_156631

theorem total_lotus_flowers (x : ℕ) (h1 : x > 0) 
  (c1 : 3 ∣ x)
  (c2 : 5 ∣ x)
  (c3 : 6 ∣ x)
  (c4 : 4 ∣ x)
  (h_total : x = x / 3 + x / 5 + x / 6 + x / 4 + 6) : 
  x = 120 :=
by
  sorry

end total_lotus_flowers_l156_156631


namespace part1_solution_part2_solution_l156_156687

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end part1_solution_part2_solution_l156_156687


namespace domain_of_inverse_function_l156_156765

noncomputable def log_inverse_domain : Set ℝ :=
  {y | y ≥ 5}

theorem domain_of_inverse_function :
  ∀ y, y ∈ log_inverse_domain ↔ ∃ x, x ≥ 3 ∧ y = 4 + Real.logb 2 (x - 1) :=
by
  sorry

end domain_of_inverse_function_l156_156765


namespace even_four_digit_increasing_count_l156_156772

theorem even_four_digit_increasing_count :
  let digits := {x // 1 ≤ x ∧ x ≤ 9}
  let even_digits := {x // x ∈ digits ∧ x % 2 = 0}
  {n : ℕ //
    ∃ a b c d : ℕ,
      n = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ even_digits ∧
      a < b ∧ b < c ∧ c < d} =
  17 :=
by sorry

end even_four_digit_increasing_count_l156_156772


namespace equalize_foma_ierema_l156_156157

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l156_156157


namespace parallelogram_proof_l156_156788

noncomputable def parallelogram_ratio (AP AB AQ AD AC AT : ℝ) (hP : AP / AB = 61 / 2022) (hQ : AQ / AD = 61 / 2065) (h_intersect : true) : ℕ :=
if h : AC / AT = 4087 / 61 then 67 else 0

theorem parallelogram_proof :
  ∀ (ABCD : Type) (P : Type) (Q : Type) (T : Type) 
     (AP AB AQ AD AC AT : ℝ) 
     (hP : AP / AB = 61 / 2022) 
     (hQ : AQ / AD = 61 / 2065)
     (h_intersect : true),
  parallelogram_ratio AP AB AQ AD AC AT hP hQ h_intersect = 67 :=
by
  sorry

end parallelogram_proof_l156_156788


namespace difference_between_sums_l156_156941

-- Define the arithmetic sequence sums
def sum_seq (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Define sets A and B
def sumA : ℕ := sum_seq 10 75
def sumB : ℕ := sum_seq 76 125

-- State the problem
theorem difference_between_sums : sumB - sumA = 2220 :=
by
  -- The proof is omitted
  sorry

end difference_between_sums_l156_156941


namespace arithmetic_sequence_properties_l156_156614

theorem arithmetic_sequence_properties
    (a_1 : ℕ)
    (d : ℕ)
    (sequence : Fin 240 → ℕ)
    (h1 : ∀ n, sequence n = a_1 + n * d)
    (h2 : sequence 0 = a_1)
    (h3 : 1 ≤ a_1 ∧ a_1 ≤ 9)
    (h4 : ∃ n₁, sequence n₁ = 100)
    (h5 : ∃ n₂, sequence n₂ = 3103) :
  (a_1 = 9 ∧ d = 13) ∨ (a_1 = 1 ∧ d = 33) ∨ (a_1 = 9 ∧ d = 91) :=
sorry

end arithmetic_sequence_properties_l156_156614


namespace constant_sequence_from_conditions_l156_156640

variable (k b : ℝ) [Nontrivial ℝ]
variable (a_n : ℕ → ℝ)

-- Define the conditions function
def cond1 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond2 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (d : ℝ), ∀ n, a_n (n + 1) = a_n n + d) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond3 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b - (k * a_n n + b) = m)

-- Lean statement to prove the problem
theorem constant_sequence_from_conditions (k b : ℝ) [Nontrivial ℝ] (a_n : ℕ → ℝ) :
  (cond1 k b a_n ∨ cond2 k b a_n ∨ cond3 k b a_n) → 
  ∃ c : ℝ, ∀ n, a_n n = c :=
by
  -- To be proven
  intros
  sorry

end constant_sequence_from_conditions_l156_156640


namespace average_salary_rest_l156_156826

noncomputable def average_salary_of_the_rest : ℕ := 6000

theorem average_salary_rest 
  (N : ℕ) 
  (A : ℕ)
  (T : ℕ)
  (A_T : ℕ)
  (Nr : ℕ)
  (Ar : ℕ)
  (H1 : N = 42)
  (H2 : A = 8000)
  (H3 : T = 7)
  (H4 : A_T = 18000)
  (H5 : Nr = N - T)
  (H6 : Nr = 42 - 7)
  (H7 : Ar = 6000)
  (H8 : 42 * 8000 = (Nr * Ar) + (T * 18000))
  : Ar = average_salary_of_the_rest :=
by
  sorry

end average_salary_rest_l156_156826


namespace correct_observation_value_l156_156554

theorem correct_observation_value (mean : ℕ) (n : ℕ) (incorrect_obs : ℕ) (corrected_mean : ℚ) (original_sum : ℚ) (remaining_sum : ℚ) (corrected_sum : ℚ) :
  mean = 30 →
  n = 50 →
  incorrect_obs = 23 →
  corrected_mean = 30.5 →
  original_sum = (n * mean) →
  remaining_sum = (original_sum - incorrect_obs) →
  corrected_sum = (n * corrected_mean) →
  ∃ x : ℕ, remaining_sum + x = corrected_sum → x = 48 :=
by
  intros h_mean h_n h_incorrect_obs h_corrected_mean h_original_sum h_remaining_sum h_corrected_sum
  have original_mean := h_mean
  have observations := h_n
  have incorrect_observation := h_incorrect_obs
  have new_mean := h_corrected_mean
  have original_sum_calc := h_original_sum
  have remaining_sum_calc := h_remaining_sum
  have corrected_sum_calc := h_corrected_sum
  use 48
  sorry

end correct_observation_value_l156_156554


namespace product_xyz_eq_one_l156_156783

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end product_xyz_eq_one_l156_156783


namespace find_smaller_part_l156_156311

noncomputable def smaller_part (x y : ℕ) : ℕ :=
  if x ≤ y then x else y

theorem find_smaller_part (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : smaller_part x y = 11 :=
  sorry

end find_smaller_part_l156_156311


namespace find_f_zero_l156_156642

variable (f : ℝ → ℝ)
variable (hf : ∀ x y : ℝ, f (x + y) = f x + f y + 1 / 2)

theorem find_f_zero : f 0 = -1 / 2 :=
by
  sorry

end find_f_zero_l156_156642


namespace projections_proportional_to_squares_l156_156542

theorem projections_proportional_to_squares
  (a b c a1 b1 : ℝ)
  (h₀ : c ≠ 0)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a1 = (a^2) / c)
  (h₃ : b1 = (b^2) / c) :
  (a1 / b1) = (a^2 / b^2) :=
by sorry

end projections_proportional_to_squares_l156_156542


namespace distinct_dragons_count_l156_156089

theorem distinct_dragons_count : 
  {n : ℕ // n = 7} :=
sorry

end distinct_dragons_count_l156_156089


namespace factorization_correctness_l156_156711

theorem factorization_correctness :
  (∀ x, x^2 + 2 * x + 1 = (x + 1)^2) ∧
  ¬ (∀ x, x * (x + 1) = x^2 + x) ∧
  ¬ (∀ x y, x^2 + x * y - 3 = x * (x + y) - 3) ∧
  ¬ (∀ x, x^2 + 6 * x + 4 = (x + 3)^2 - 5) :=
by
  sorry

end factorization_correctness_l156_156711


namespace sum_of_primes_between_1_and_20_l156_156005

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l156_156005


namespace find_w_l156_156332

noncomputable def line_p(t : ℝ) : (ℝ × ℝ) := (2 + 3 * t, 5 + 2 * t)
noncomputable def line_q(u : ℝ) : (ℝ × ℝ) := (-3 + 3 * u, 7 + 2 * u)

def vector_DC(t u : ℝ) : ℝ × ℝ := ((2 + 3 * t) - (-3 + 3 * u), (5 + 2 * t) - (7 + 2 * u))

def w_condition (w1 w2 : ℝ) : Prop := w1 + w2 = 3

theorem find_w (t u : ℝ) :
  ∃ w1 w2 : ℝ, 
    w_condition w1 w2 ∧ 
    (∃ k : ℝ, 
      sorry -- This is a placeholder for the projection calculation
    )
    :=
  sorry -- This is a placeholder for the final proof

end find_w_l156_156332


namespace algebraic_expression_value_l156_156966

theorem algebraic_expression_value (p q : ℝ)
  (h : p * 3^3 + q * 3 + 3 = 2005) :
  p * (-3)^3 + q * (-3) + 3 = -1999 :=
by
   sorry

end algebraic_expression_value_l156_156966


namespace m_range_decrease_y_l156_156793

theorem m_range_decrease_y {m : ℝ} : (∀ x1 x2 : ℝ, x1 < x2 → (2 * m + 2) * x1 + 5 > (2 * m + 2) * x2 + 5) ↔ m < -1 :=
by
  sorry

end m_range_decrease_y_l156_156793


namespace solve_fractional_equation_l156_156821

theorem solve_fractional_equation {x : ℝ} (h1 : x ≠ -1) (h2 : x ≠ 0) :
  6 / (x + 1) = (x + 5) / (x * (x + 1)) ↔ x = 1 :=
by
  -- This proof is left as an exercise.
  sorry

end solve_fractional_equation_l156_156821


namespace sum_of_primes_1_to_20_l156_156001

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l156_156001


namespace foma_should_give_ierema_55_coins_l156_156139

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156139


namespace marge_funds_for_fun_l156_156405

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end marge_funds_for_fun_l156_156405


namespace ancient_chinese_silver_problem_l156_156087

theorem ancient_chinese_silver_problem :
  ∃ (x y : ℤ), 7 * x = y - 4 ∧ 9 * x = y + 8 :=
by
  sorry

end ancient_chinese_silver_problem_l156_156087


namespace number_of_real_roots_l156_156829

theorem number_of_real_roots :
  ∃ (roots_count : ℕ), roots_count = 2 ∧
  (∀ x : ℝ, x^2 - |2 * x - 1| - 4 = 0 → (x = -1 - Real.sqrt 6 ∨ x = 3)) :=
sorry

end number_of_real_roots_l156_156829


namespace number_of_littering_citations_l156_156726

variable (L D P : ℕ)
variable (h1 : L = D)
variable (h2 : P = 2 * (L + D))
variable (h3 : L + D + P = 24)

theorem number_of_littering_citations : L = 4 :=
by
  sorry

end number_of_littering_citations_l156_156726


namespace total_cans_needed_l156_156723

-- Definitions
def cans_per_box : ℕ := 4
def number_of_boxes : ℕ := 203

-- Statement of the problem
theorem total_cans_needed : cans_per_box * number_of_boxes = 812 := 
by
  -- skipping the proof
  sorry

end total_cans_needed_l156_156723


namespace hypotenuse_length_l156_156323

theorem hypotenuse_length (a b c : ℝ) (h₁ : a + b + c = 40) (h₂ : 0.5 * a * b = 24) (h₃ : a^2 + b^2 = c^2) : c = 18.8 := sorry

end hypotenuse_length_l156_156323


namespace five_n_plus_three_composite_l156_156095

theorem five_n_plus_three_composite (n x y : ℕ) 
  (h_pos : 0 < n)
  (h1 : 2 * n + 1 = x ^ 2)
  (h2 : 3 * n + 1 = y ^ 2) : 
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = 5 * n + 3 := 
sorry

end five_n_plus_three_composite_l156_156095


namespace other_discount_l156_156453

theorem other_discount (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (other_discount : ℝ) :
  list_price = 70 → final_price = 61.74 → first_discount = 10 → (list_price * (1 - first_discount / 100) * (1 - other_discount / 100) = final_price) → other_discount = 2 := 
by
  intros h1 h2 h3 h4
  sorry

end other_discount_l156_156453


namespace num_convex_pentagons_l156_156340

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l156_156340


namespace committee_of_4_from_10_eq_210_l156_156175

theorem committee_of_4_from_10_eq_210 :
  (Nat.choose 10 4) = 210 :=
by
  sorry

end committee_of_4_from_10_eq_210_l156_156175


namespace new_sequence_69th_term_l156_156387

-- Definitions and conditions
def original_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ := a n

def new_sequence (a : ℕ → ℕ) (k : ℕ) : ℕ :=
if k % 4 = 1 then a (k / 4 + 1) else 0  -- simplified modeling, the inserted numbers are denoted arbitrarily as 0

-- The statement to be proven
theorem new_sequence_69th_term (a : ℕ → ℕ) : new_sequence a 69 = a 18 :=
by
  sorry

end new_sequence_69th_term_l156_156387


namespace increase_is_50_percent_l156_156803

theorem increase_is_50_percent (original new : ℕ) (h1 : original = 60) (h2 : new = 90) :
  ((new - original) * 100 / original) = 50 :=
by
  -- Proof can be filled here.
  sorry

end increase_is_50_percent_l156_156803


namespace total_pages_in_book_l156_156875

variable (p1 p2 p_total : ℕ)
variable (read_first_four_days : p1 = 4 * 45)
variable (read_next_three_days : p2 = 3 * 52)
variable (total_until_last_day : p_total = p1 + p2 + 15)

theorem total_pages_in_book : p_total = 351 :=
by
  -- Introduce the conditions
  rw [read_first_four_days, read_next_three_days] at total_until_last_day
  sorry

end total_pages_in_book_l156_156875


namespace equation_of_line_AB_l156_156586

theorem equation_of_line_AB 
  (x y : ℝ)
  (passes_through_P : (4 - 1)^2 + (1 - 0)^2 = 1)     
  (circle_eq : (x - 1)^2 + y^2 = 1) :
  3 * x + y - 4 = 0 :=
sorry

end equation_of_line_AB_l156_156586


namespace necessary_but_not_sufficient_condition_l156_156171

variable (x y : ℤ)

def p : Prop := x ≠ 2 ∨ y ≠ 4
def q : Prop := x + y ≠ 6

theorem necessary_but_not_sufficient_condition :
  (p x y → q x y) ∧ (¬q x y → ¬p x y) :=
sorry

end necessary_but_not_sufficient_condition_l156_156171


namespace amoeba_growth_after_5_days_l156_156992

theorem amoeba_growth_after_5_days : (3 : ℕ)^5 = 243 := by
  sorry

end amoeba_growth_after_5_days_l156_156992


namespace keith_stored_bales_l156_156561

theorem keith_stored_bales (initial_bales added_bales final_bales : ℕ) :
  initial_bales = 22 → final_bales = 89 → final_bales = initial_bales + added_bales → added_bales = 67 :=
by
  intros h_initial h_final h_eq
  sorry

end keith_stored_bales_l156_156561


namespace sum_of_fractions_and_decimal_l156_156701

theorem sum_of_fractions_and_decimal : 
    (3 / 25 : ℝ) + (1 / 5) + 55.21 = 55.53 :=
by 
  sorry

end sum_of_fractions_and_decimal_l156_156701


namespace number_exceeds_35_percent_by_245_l156_156167

theorem number_exceeds_35_percent_by_245 : 
  ∃ (x : ℝ), (0.35 * x + 245 = x) ∧ x = 376.92 := 
by
  sorry

end number_exceeds_35_percent_by_245_l156_156167


namespace no_solution_for_equation_l156_156276

theorem no_solution_for_equation :
  ¬ (∃ x : ℝ, 
    4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 5 * (12 - (4 * (x + 1) - 3 * x)) = 
    18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11)))) :=
by
  sorry

end no_solution_for_equation_l156_156276


namespace president_and_committee_combination_l156_156524

theorem president_and_committee_combination (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 ∧ k = 3 ∧ total = (10 * Nat.choose 9 3) → total = 840 :=
by
  intros
  sorry

end president_and_committee_combination_l156_156524


namespace smallest_norm_value_l156_156536

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l156_156536


namespace integer_roots_of_polynomial_l156_156986

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | (x^3 + a₂ * x^2 + a₁ * x - 18 = 0)} ⊆ {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
by sorry

end integer_roots_of_polynomial_l156_156986


namespace segments_form_quadrilateral_l156_156029

theorem segments_form_quadrilateral (a d : ℝ) (h_pos : a > 0 ∧ d > 0) (h_sum : 4 * a + 6 * d = 3) : 
  (∃ s1 s2 s3 s4 : ℝ, s1 + s2 + s3 > s4 ∧ s1 + s2 + s4 > s3 ∧ s1 + s3 + s4 > s2 ∧ s2 + s3 + s4 > s1) :=
sorry

end segments_form_quadrilateral_l156_156029


namespace minimum_distance_AB_l156_156894

-- Definitions of the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C2 (x y : ℝ) : Prop := y^2 - x + 1 = 0

theorem minimum_distance_AB :
  ∃ (A B : ℝ × ℝ), C1 A.1 A.2 ∧ C2 B.1 B.2 ∧ dist A B = 3*Real.sqrt 2 / 4 := sorry

end minimum_distance_AB_l156_156894


namespace carter_drum_sticks_l156_156041

def sets_per_show (used : ℕ) (tossed : ℕ) : ℕ := used + tossed

def total_sets (sets_per_show : ℕ) (num_shows : ℕ) : ℕ := sets_per_show * num_shows

theorem carter_drum_sticks :
  sets_per_show 8 10 * 45 = 810 :=
by
  sorry

end carter_drum_sticks_l156_156041


namespace simplify_expression_l156_156483

theorem simplify_expression (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 :=
by
  sorry

end simplify_expression_l156_156483


namespace initial_marbles_count_l156_156395

-- Leo's initial conditions and quantities
def initial_packs := 40
def marbles_per_pack := 10
def given_Manny (P: ℕ) := P / 4
def given_Neil (P: ℕ) := P / 8
def kept_by_Leo := 25

-- The equivalent proof problem stated in Lean
theorem initial_marbles_count (P: ℕ) (Manny_packs: ℕ) (Neil_packs: ℕ) (kept_packs: ℕ) :
  Manny_packs = given_Manny P → Neil_packs = given_Neil P → kept_packs = kept_by_Leo → P = initial_packs → P * marbles_per_pack = 400 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_marbles_count_l156_156395


namespace jellybean_ratio_l156_156531

theorem jellybean_ratio (jellybeans_large: ℕ) (large_glasses: ℕ) (small_glasses: ℕ) (total_jellybeans: ℕ) (jellybeans_per_large: ℕ) (jellybeans_per_small: ℕ)
  (h1 : jellybeans_large = 50)
  (h2 : large_glasses = 5)
  (h3 : small_glasses = 3)
  (h4 : total_jellybeans = 325)
  (h5 : jellybeans_per_large = jellybeans_large * large_glasses)
  (h6 : jellybeans_per_small * small_glasses = total_jellybeans - jellybeans_per_large)
  : jellybeans_per_small = 25 ∧ jellybeans_per_small / jellybeans_large = 1 / 2 :=
by
  sorry

end jellybean_ratio_l156_156531


namespace range_of_a_l156_156363

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - a ≥ 0) ↔ (a ≤ 0) :=
by
  sorry

end range_of_a_l156_156363


namespace range_of_a_l156_156058

noncomputable def p (a : ℝ) := ∀ x : ℝ, x^2 + a ≥ 0
noncomputable def q (a : ℝ) := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≥ 0) := by
  sorry

end range_of_a_l156_156058


namespace max_books_borrowed_l156_156244

theorem max_books_borrowed (students_total : ℕ) (students_no_books : ℕ) 
  (students_1_book : ℕ) (students_2_books : ℕ) (students_at_least_3_books : ℕ) 
  (average_books_per_student : ℝ) (H1 : students_total = 60) 
  (H2 : students_no_books = 4) 
  (H3 : students_1_book = 18) 
  (H4 : students_2_books = 20) 
  (H5 : students_at_least_3_books = students_total - (students_no_books + students_1_book + students_2_books)) 
  (H6 : average_books_per_student = 2.5) : 
  ∃ max_books : ℕ, max_books = 41 :=
by
  sorry

end max_books_borrowed_l156_156244


namespace mark_cans_correct_l156_156461

variable (R : ℕ) -- Rachel's cans
variable (J : ℕ) -- Jaydon's cans
variable (M : ℕ) -- Mark's cans
variable (T : ℕ) -- Total cans 

-- Conditions
def jaydon_cans (R : ℕ) : ℕ := 2 * R + 5
def mark_cans (J : ℕ) : ℕ := 4 * J
def total_cans (R : ℕ) (J : ℕ) (M : ℕ) : ℕ := R + J + M

theorem mark_cans_correct (R : ℕ) (J : ℕ) 
  (h1 : J = jaydon_cans R) 
  (h2 : M = mark_cans J) 
  (h3 : total_cans R J M = 135) : 
  M = 100 := 
sorry

end mark_cans_correct_l156_156461


namespace length_of_AE_l156_156976

/-- Given the conditions on the pentagon ABCDE:
1. AB = 2, BC = 2, CD = 5, DE = 7
2. AC is the largest side in triangle ABC
3. CE is the smallest side in triangle ECD
4. In triangle ACE all sides are integers and have distinct lengths,
prove that the length of side AE is 5. -/
theorem length_of_AE
  (AB BC CD DE : ℕ)
  (hAB : AB = 2)
  (hBC : BC = 2)
  (hCD : CD = 5)
  (hDE : DE = 7)
  (AC : ℕ) 
  (hAC_large : AB < AC ∧ BC < AC)
  (CE : ℕ)
  (hCE_small : CE < CD ∧ CE < DE)
  (AE : ℕ)
  (distinct_sides : ∀ x y z : ℕ, x ≠ y → x ≠ z → y ≠ z → (AC = x ∨ CE = x ∨ AE = x) → (AC = y ∨ CE = y ∨ AE = y) → (AC = z ∨ CE = z ∨ AE = z)) :
  AE = 5 :=
sorry

end length_of_AE_l156_156976


namespace PetesOriginalNumber_l156_156414

-- Define the context and problem
theorem PetesOriginalNumber (x : ℤ) (h : 3 * (2 * x + 12) = 90) : x = 9 :=
by
  -- proof goes here
  sorry

end PetesOriginalNumber_l156_156414


namespace equalize_foma_ierema_l156_156154

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l156_156154


namespace painted_cube_l156_156861

theorem painted_cube (n : ℕ) (h : 3 / 4 * (6 * n ^ 3) = 4 * n ^ 2) : n = 2 := sorry

end painted_cube_l156_156861


namespace lines_in_n_by_n_grid_l156_156288

def num_horizontal_lines (n : ℕ) : ℕ := n + 1
def num_vertical_lines (n : ℕ) : ℕ := n + 1
def total_lines (n : ℕ) : ℕ := num_horizontal_lines n + num_vertical_lines n

theorem lines_in_n_by_n_grid (n : ℕ) :
  total_lines n = 2 * (n + 1) := by
  sorry

end lines_in_n_by_n_grid_l156_156288


namespace fraction_simplification_l156_156069

variable {x y z : ℝ}
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z - z / x ≠ 0)

theorem fraction_simplification :
  (x^2 - 1 / y^2) / (z - z / x) = x / z :=
by
  sorry

end fraction_simplification_l156_156069


namespace age_of_20th_student_l156_156825

theorem age_of_20th_student (avg_age_20 : ℕ) (avg_age_9 : ℕ) (avg_age_10 : ℕ) :
  (avg_age_20 = 20) →
  (avg_age_9 = 11) →
  (avg_age_10 = 24) →
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  (age_20th = 61) :=
by
  intros h1 h2 h3
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  sorry

end age_of_20th_student_l156_156825


namespace equalize_foma_ierema_l156_156143

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l156_156143


namespace find_right_triangle_conditions_l156_156466

def is_right_triangle (A B C : ℝ) : Prop := 
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem find_right_triangle_conditions (A B C : ℝ):
  (A + B = C ∧ is_right_triangle A B C) ∨ 
  (A = B ∧ B = 2 * C ∧ is_right_triangle A B C) ∨ 
  (A / 30 = 1 ∧ B / 30 = 2 ∧ C / 30 = 3 ∧ is_right_triangle A B C) :=
sorry

end find_right_triangle_conditions_l156_156466


namespace solve_for_y_l156_156369

theorem solve_for_y 
  (x y : ℝ) 
  (h1 : 2 * x - 3 * y = 9) 
  (h2 : x + y = 8) : 
  y = 1.4 := 
sorry

end solve_for_y_l156_156369


namespace mn_values_l156_156539

theorem mn_values (m n : ℤ) (h : m^2 * n^2 + m^2 + n^2 + 10 * m * n + 16 = 0) : 
  (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
  sorry

end mn_values_l156_156539


namespace triangular_weight_is_60_l156_156128

variable (w_round w_triangular w_rectangular : ℝ)

axiom rectangular_weight : w_rectangular = 90
axiom balance1 : w_round + w_triangular = 3 * w_round
axiom balance2 : 4 * w_round + w_triangular = w_triangular + w_round + w_rectangular

theorem triangular_weight_is_60 :
  w_triangular = 60 :=
by
  sorry

end triangular_weight_is_60_l156_156128


namespace find_range_of_t_l156_156230

variable {f : ℝ → ℝ}

-- Definitions for the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ ⦃x y⦄, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x > f y

-- Given the conditions, we need to prove the statement
theorem find_range_of_t (h_odd : is_odd_function f)
    (h_decreasing : is_decreasing_on f (-1) 1)
    (h_inequality : ∀ t : ℝ, -1 < t ∧ t < 1 → f (1 - t) + f (1 - t^2) < 0) :
  ∀ t, -1 < t ∧ t < 1 → 0 < t ∧ t < 1 :=
  by
  sorry

end find_range_of_t_l156_156230


namespace identify_counterfeit_bag_l156_156443

theorem identify_counterfeit_bag (n : ℕ) (w W : ℕ) (H : ∃ k : ℕ, k ≤ n ∧ W = w * (n * (n + 1) / 2) - k) : 
  ∃ bag_num, bag_num = w * (n * (n + 1) / 2) - W := by
  sorry

end identify_counterfeit_bag_l156_156443


namespace library_books_l156_156916

theorem library_books (A : Prop) (B : Prop) (C : Prop) (D : Prop) :
  (¬A) → (B ∧ D) :=
by
  -- Assume the statement "All books in this library are available for lending." is represented by A.
  -- A is false.
  intro h_notA
  -- Show that statement II ("There is some book in this library not available for lending.")
  -- and statement IV ("Not all books in this library are available for lending.") are both true.
  -- These are represented as B and D, respectively.
  sorry

end library_books_l156_156916


namespace number_of_8_digit_increasing_integers_mod_1000_l156_156931

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_8_digit_increasing_integers_mod_1000 :
  let M := choose 9 8
  M % 1000 = 9 :=
by
  let M := choose 9 8
  show M % 1000 = 9
  sorry

end number_of_8_digit_increasing_integers_mod_1000_l156_156931


namespace line_point_coordinates_l156_156647

theorem line_point_coordinates (t : ℝ) (x y z : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) →
  t = 1/2 →
  (x, y, z) = (5, 3/2, 3) :=
by
  intros h1 h2
  sorry

end line_point_coordinates_l156_156647


namespace jason_average_messages_l156_156926

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end jason_average_messages_l156_156926


namespace count_irreducible_fractions_l156_156083

theorem count_irreducible_fractions : 
  let nums := [226, 227, 229, 232, 233, 236, 238, 239]
  in (∀ n ∈ nums, by apply (1 / 16 < n / 15 ∧ n / 15 < 1 / 15 ∧ Nat.gcd n 15 = 1)) 
  ∧ nums.length = 8 := 
sorry

end count_irreducible_fractions_l156_156083


namespace solution_set_f_pos_l156_156398

open Set Function

variables (f : ℝ → ℝ)
variables (h_even : ∀ x : ℝ, f (-x) = f x)
variables (h_diff : ∀ x ≠ 0, DifferentiableAt ℝ f x)
variables (h_pos : ∀ x : ℝ, x > 0 → f x + x * (f' x) > 0)
variables (h_at_2 : f 2 = 0)

theorem solution_set_f_pos :
  {x : ℝ | f x > 0} = (Iio (-2)) ∪ (Ioi 2) :=
by 
  sorry

end solution_set_f_pos_l156_156398


namespace Eric_test_score_l156_156267

theorem Eric_test_score (n : ℕ) (old_avg new_avg : ℚ) (Eric_score : ℚ) :
  n = 22 →
  old_avg = 84 →
  new_avg = 85 →
  Eric_score = (n * new_avg) - ((n - 1) * old_avg) →
  Eric_score = 106 :=
by
  intros h1 h2 h3 h4
  sorry

end Eric_test_score_l156_156267


namespace find_f_prime_at_one_l156_156572

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end find_f_prime_at_one_l156_156572


namespace find_principal_sum_l156_156168

theorem find_principal_sum (P : ℝ) (r : ℝ) (A2 : ℝ) (A3 : ℝ) : 
  (A2 = 7000) → (A3 = 9261) → 
  (A2 = P * (1 + r)^2) → (A3 = P * (1 + r)^3) → 
  P = 4000 :=
by
  intro hA2 hA3 hA2_eq hA3_eq
  -- here, we assume the proof steps leading to P = 4000
  sorry

end find_principal_sum_l156_156168


namespace five_goats_choir_l156_156688

theorem five_goats_choir 
  (total_members : ℕ)
  (num_rows : ℕ)
  (total_members_eq : total_members = 51)
  (num_rows_eq : num_rows = 4) :
  ∃ row_people : ℕ, row_people ≥ 13 :=
by 
  sorry

end five_goats_choir_l156_156688


namespace units_digit_35_pow_7_plus_93_pow_45_l156_156581

-- Definitions of units digit calculations for the specific values
def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_35_pow_7 : ℕ := units_digit (35 ^ 7)
def units_digit_93_pow_45 : ℕ := units_digit (93 ^ 45)

-- Statement to prove that the sum of the units digits is 8
theorem units_digit_35_pow_7_plus_93_pow_45 : 
  units_digit (35 ^ 7) + units_digit (93 ^ 45) = 8 :=
by 
  sorry -- proof omitted

end units_digit_35_pow_7_plus_93_pow_45_l156_156581


namespace circle_tangent_to_yaxis_and_line_l156_156641

theorem circle_tangent_to_yaxis_and_line :
  (∃ C : ℝ → ℝ → Prop, 
    (∀ x y r : ℝ, C x y ↔ (x - 3) ^ 2 + (y - 2) ^ 2 = 9 ∨ (x + 1 / 3) ^ 2 + (y - 2) ^ 2 = 1 / 9) ∧ 
    (∀ y : ℝ, C 0 y → y = 2) ∧ 
    (∀ x y: ℝ, C x y → (∃ x1 : ℝ, 4 * x - 3 * y + 9 = 0 → 4 * x1 + 3 = 0))) :=
sorry

end circle_tangent_to_yaxis_and_line_l156_156641


namespace find_range_of_k_l156_156354

noncomputable def range_of_k (a : ℝ) : Set ℝ :=
  {k | ∃ x, log a (x - a * k) = log a (x^2 - a^2)}

theorem find_range_of_k (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) : 
  range_of_k a = {k | k ∈ set.Ioo 0 1 ∪ set.Iio (-1)} :=
sorry

end find_range_of_k_l156_156354


namespace increase_in_disposable_income_l156_156255

-- John's initial weekly income and tax details
def initial_weekly_income : ℝ := 60
def initial_tax_rate : ℝ := 0.15

-- John's new weekly income and tax details
def new_weekly_income : ℝ := 70
def new_tax_rate : ℝ := 0.18

-- John's monthly expense
def monthly_expense : ℝ := 100

-- Weekly disposable income calculations
def initial_weekly_net : ℝ := initial_weekly_income * (1 - initial_tax_rate)
def new_weekly_net : ℝ := new_weekly_income * (1 - new_tax_rate)

-- Monthly disposable income calculations
def initial_monthly_income : ℝ := initial_weekly_net * 4
def new_monthly_income : ℝ := new_weekly_net * 4

def initial_disposable_income : ℝ := initial_monthly_income - monthly_expense
def new_disposable_income : ℝ := new_monthly_income - monthly_expense

-- Calculate the percentage increase
def percentage_increase : ℝ := ((new_disposable_income - initial_disposable_income) / initial_disposable_income) * 100

-- Claim: The percentage increase in John's disposable income is approximately 24.62%
theorem increase_in_disposable_income : abs(percentage_increase - 24.62) < 1e-2 := by
  sorry

end increase_in_disposable_income_l156_156255


namespace find_ratio_l156_156434

variable {x y k x1 x2 y1 y2 : ℝ}

-- Inverse proportionality
def inverse_proportional (x y k : ℝ) : Prop := x * y = k

-- Given conditions
axiom h1 : inverse_proportional x1 y1 k
axiom h2 : inverse_proportional x2 y2 k
axiom h3 : x1 ≠ 0
axiom h4 : x2 ≠ 0
axiom h5 : y1 ≠ 0
axiom h6 : y2 ≠ 0
axiom h7 : x1 / x2 = 3 / 4

theorem find_ratio : y1 / y2 = 4 / 3 :=
by
  sorry

end find_ratio_l156_156434


namespace find_f_x_l156_156235

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 1 = 3 * x + 2) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end find_f_x_l156_156235


namespace roots_numerically_equal_opposite_signs_l156_156508

theorem roots_numerically_equal_opposite_signs
  (a b d: ℝ) 
  (h: ∃ x : ℝ, (x^2 - (a + 1) * x) / ((b + 1) * x - d) = (n - 2) / (n + 2) ∧ x = -x)
  : n = 2 * (b - a) / (a + b + 2) := by
  sorry

end roots_numerically_equal_opposite_signs_l156_156508


namespace range_of_m_l156_156767

noncomputable def common_points (k : ℝ) (m : ℝ) := 
  ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)

theorem range_of_m (k : ℝ) (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)) ↔ 
  (m ∈ (Set.Ioo 1 5 ∪ Set.Ioi 5)) :=
by
  sorry

end range_of_m_l156_156767


namespace coral_remaining_pages_l156_156046

def pages_after_week1 (total_pages : ℕ) : ℕ :=
  total_pages / 2

def pages_after_week2 (remaining_pages_week1 : ℕ) : ℕ :=
  remaining_pages_week1 - (3 * remaining_pages_week1 / 10)

def pages_after_week3 (remaining_pages_week2 : ℕ) (reading_hours : ℕ) (reading_speed : ℕ) : ℕ :=
  remaining_pages_week2 - (reading_hours * reading_speed)

theorem coral_remaining_pages (total_pages remaining_pages_week1 remaining_pages_week2 remaining_pages_week3 : ℕ) 
  (reading_hours reading_speed unread_pages : ℕ)
  (h1 : total_pages = 600)
  (h2 : remaining_pages_week1 = pages_after_week1 total_pages)
  (h3 : remaining_pages_week2 = pages_after_week2 remaining_pages_week1)
  (h4 : reading_hours = 10)
  (h5 : reading_speed = 15)
  (h6 : remaining_pages_week3 = pages_after_week3 remaining_pages_week2 reading_hours reading_speed)
  (h7 : unread_pages = remaining_pages_week3) :
  unread_pages = 60 :=
by
  sorry

end coral_remaining_pages_l156_156046


namespace investment_duration_p_l156_156125

-- Given the investments ratio, profits ratio, and time period for q,
-- proving the time period of p's investment is 7 months.
theorem investment_duration_p (T_p T_q : ℕ) 
  (investment_ratio : 7 * T_p = 5 * T_q) 
  (profit_ratio : 7 * T_p / T_q = 7 / 10)
  (T_q_eq : T_q = 14) : T_p = 7 :=
by
  sorry

end investment_duration_p_l156_156125


namespace seokgi_initial_money_l156_156421

theorem seokgi_initial_money (X : ℝ) (h1 : X / 2 - X / 4 = 1250) : X = 5000 := by
  sorry

end seokgi_initial_money_l156_156421


namespace balls_in_boxes_l156_156908

def waysToPutBallsInBoxes (balls : ℕ) (boxes : ℕ) [Finite boxes] : ℕ :=
  Finset.card { f : Fin boxes → ℕ | (Finset.sum Finset.univ (fun i => f i)) = balls }

theorem balls_in_boxes : waysToPutBallsInBoxes 7 3 = 36 := by
  sorry

end balls_in_boxes_l156_156908


namespace product_of_a_and_b_is_zero_l156_156269

theorem product_of_a_and_b_is_zero
  (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)
  (h2 : b < 10)
  (h3 : a * (b + 10) = 190) :
  a * b = 0 :=
sorry

end product_of_a_and_b_is_zero_l156_156269


namespace find_number_l156_156071

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l156_156071


namespace find_student_ticket_price_l156_156731

variable (S : ℝ)
variable (student_tickets non_student_tickets total_tickets : ℕ)
variable (non_student_ticket_price total_revenue : ℝ)

theorem find_student_ticket_price 
  (h1 : student_tickets = 90)
  (h2 : non_student_tickets = 60)
  (h3 : total_tickets = student_tickets + non_student_tickets)
  (h4 : non_student_ticket_price = 8)
  (h5 : total_revenue = 930)
  (h6 : 90 * S + 60 * non_student_ticket_price = total_revenue) : 
  S = 5 := 
sorry

end find_student_ticket_price_l156_156731


namespace music_collections_l156_156597

open Finset

variables {A J M : Finset ℕ}

theorem music_collections :
  (|A ∩ J ∩ M| = 12) ∧
  (|A| = 25) ∧
  (|J \ (A ∪ M)| = 8) ∧
  (|M \ (A ∪ J)| = 5) →
  (|A \ (J ∪ M)| + |J \ (A ∪ M)| + |M \ (A ∪ J)| = 26) :=
begin
  sorry,
end

end music_collections_l156_156597


namespace polynomial_sum_zero_l156_156309

open BigOperators

theorem polynomial_sum_zero {f : ℕ → ℤ} {n : ℕ} (hf : ∀ k, Polynomial.degree (Polynomial.ofFinsupp (f k)) ≤ n - 1) :
  ∑ k in Finset.range (n + 1), (Nat.choose n k) * (-1)^k * f k = 0 :=
sorry

end polynomial_sum_zero_l156_156309


namespace range_of_a_l156_156098

theorem range_of_a (a : ℝ) (h₁ : 1/2 ≤ 1) (h₂ : a ≤ a + 1)
    (h_condition : ∀ x:ℝ, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) :
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l156_156098


namespace equivalent_problem_l156_156886

variable {x y : Real}

theorem equivalent_problem 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 15) :
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 :=
by
  sorry

end equivalent_problem_l156_156886


namespace Manny_lasagna_pieces_l156_156669

-- Define variables and conditions
variable (M : ℕ) -- Manny's desired number of pieces
variable (A : ℕ := 0) -- Aaron's pieces
variable (K : ℕ := 2 * M) -- Kai's pieces
variable (R : ℕ := M / 2) -- Raphael's pieces
variable (L : ℕ := 2 + R) -- Lisa's pieces

-- Prove that Manny wants 1 piece of lasagna
theorem Manny_lasagna_pieces (M : ℕ) (A : ℕ := 0) (K : ℕ := 2 * M) (R : ℕ := M / 2) (L : ℕ := 2 + R) 
  (h : M + A + K + R + L = 6) : M = 1 :=
by
  sorry

end Manny_lasagna_pieces_l156_156669


namespace average_blinks_in_normal_conditions_l156_156419

theorem average_blinks_in_normal_conditions (blink_gaming : ℕ) (k : ℚ) (blink_normal : ℚ) 
  (h_blink_gaming : blink_gaming = 10)
  (h_k : k = (3 / 5))
  (h_condition : blink_gaming = blink_normal - k * blink_normal) : 
  blink_normal = 25 := 
by 
  sorry

end average_blinks_in_normal_conditions_l156_156419


namespace base_equivalence_l156_156375

theorem base_equivalence : 
  ∀ (b : ℕ), (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 ↔ b = 10 := 
by
  sorry

end base_equivalence_l156_156375


namespace slips_with_3_l156_156686

theorem slips_with_3 (x : ℤ) 
    (h1 : 15 > 0) 
    (h2 : 3 > 0 ∧ 9 > 0) 
    (h3 : (3 * x + 9 * (15 - x)) / 15 = 5) : 
    x = 10 := 
sorry

end slips_with_3_l156_156686


namespace wire_cut_ratio_l156_156734

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end wire_cut_ratio_l156_156734


namespace solve_y_l156_156820

theorem solve_y (y : ℝ) (h : (4 * y - 2) / (5 * y - 5) = 3 / 4) : y = -7 :=
by
  sorry

end solve_y_l156_156820


namespace train_length_l156_156325

-- Definitions and conditions
variable (L : ℕ)
def condition1 (L : ℕ) : Prop := L + 100 = 15 * (L + 100) / 15
def condition2 (L : ℕ) : Prop := L + 250 = 20 * (L + 250) / 20

-- Theorem statement
theorem train_length (h1 : condition1 L) (h2 : condition2 L) : L = 350 := 
by 
  sorry

end train_length_l156_156325


namespace car_speed_l156_156585

theorem car_speed (distance time speed : ℝ)
  (h_const_speed : ∀ t : ℝ, t = time → speed = distance / t)
  (h_distance : distance = 48)
  (h_time : time = 8) :
  speed = 6 :=
by
  sorry

end car_speed_l156_156585


namespace sqrt_mul_simplify_l156_156430

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l156_156430


namespace max_sum_of_radii_in_prism_l156_156384

noncomputable def sum_of_radii (AB AD AA1 : ℝ) : ℝ :=
  let r (t : ℝ) := 2 - 2 * t
  let R (t : ℝ) := 3 * t / (1 + t)
  let f (t : ℝ) := R t + r t
  let t_max := 1 / 2
  f t_max

theorem max_sum_of_radii_in_prism :
  let AB := 5
  let AD := 3
  let AA1 := 4
  sum_of_radii AB AD AA1 = 21 / 10 := by
sorry

end max_sum_of_radii_in_prism_l156_156384


namespace pirate_treasure_division_l156_156860

theorem pirate_treasure_division (initial_treasure : ℕ) (p1_share p2_share p3_share p4_share p5_share remaining : ℕ)
  (h_initial : initial_treasure = 3000)
  (h_p1_share : p1_share = initial_treasure / 10)
  (h_p1_rem : remaining = initial_treasure - p1_share)
  (h_p2_share : p2_share = 2 * remaining / 10)
  (h_p2_rem : remaining = remaining - p2_share)
  (h_p3_share : p3_share = 3 * remaining / 10)
  (h_p3_rem : remaining = remaining - p3_share)
  (h_p4_share : p4_share = 4 * remaining / 10)
  (h_p4_rem : remaining = remaining - p4_share)
  (h_p5_share : p5_share = 5 * remaining / 10)
  (h_p5_rem : remaining = remaining - p5_share)
  (p6_p9_total : ℕ)
  (h_p6_p9_total : p6_p9_total = 20 * 4)
  (final_remaining : ℕ)
  (h_final_remaining : final_remaining = remaining - p6_p9_total) :
  final_remaining = 376 :=
by sorry

end pirate_treasure_division_l156_156860


namespace sin_120_eq_sqrt3_div_2_l156_156212

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l156_156212


namespace calculation_of_cube_exponent_l156_156329

theorem calculation_of_cube_exponent (a : ℤ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end calculation_of_cube_exponent_l156_156329


namespace tic_tac_toe_board_configurations_l156_156082

theorem tic_tac_toe_board_configurations :
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  total_configurations = 592 :=
by 
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  sorry

end tic_tac_toe_board_configurations_l156_156082


namespace total_number_of_components_l156_156837

-- Definitions based on the conditions in the problem
def number_of_B_components := 300
def number_of_C_components := 200
def sample_size := 45
def number_of_A_components_drawn := 20
def number_of_C_components_drawn := 10

-- The statement to be proved
theorem total_number_of_components :
  (number_of_A_components_drawn * (number_of_B_components + number_of_C_components) / sample_size) 
  + number_of_B_components 
  + number_of_C_components 
  = 900 := 
by 
  sorry

end total_number_of_components_l156_156837


namespace solve_Cheolsu_weight_l156_156435

def Cheolsu_weight (C M F : ℝ) :=
  (C + M + F) / 3 = M ∧
  C = (2 / 3) * M ∧
  F = 72

theorem solve_Cheolsu_weight {C M F : ℝ} (h : Cheolsu_weight C M F) : C = 36 :=
by
  sorry

end solve_Cheolsu_weight_l156_156435


namespace correct_word_to_complete_sentence_l156_156798

theorem correct_word_to_complete_sentence
  (parents_spoke_language : Bool)
  (learning_difficulty : String) :
  learning_difficulty = "It was hard for him to learn English in a family, in which neither of the parents spoke the language." :=
by
  sorry

end correct_word_to_complete_sentence_l156_156798


namespace cost_of_one_shirt_l156_156376

theorem cost_of_one_shirt
  (cost_J : ℕ)  -- The cost of one pair of jeans
  (cost_S : ℕ)  -- The cost of one shirt
  (h1 : 3 * cost_J + 2 * cost_S = 69)
  (h2 : 2 * cost_J + 3 * cost_S = 81) :
  cost_S = 21 :=
by
  sorry

end cost_of_one_shirt_l156_156376


namespace jame_annual_earnings_difference_l156_156091

-- Define conditions
def new_hourly_wage := 20
def new_hours_per_week := 40
def old_hourly_wage := 16
def old_hours_per_week := 25
def weeks_per_year := 52

-- Define annual earnings calculations
def annual_earnings_old (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

def annual_earnings_new (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

-- Problem statement to prove
theorem jame_annual_earnings_difference :
  annual_earnings_new new_hourly_wage new_hours_per_week weeks_per_year -
  annual_earnings_old old_hourly_wage old_hours_per_week weeks_per_year = 20800 := by
  sorry

end jame_annual_earnings_difference_l156_156091


namespace brick_wall_l156_156858

theorem brick_wall (y : ℕ) (h1 : ∀ y, 6 * ((y / 8) + (y / 12) - 12) = y) : y = 288 :=
sorry

end brick_wall_l156_156858


namespace temperature_on_Tuesday_l156_156307

variable (T W Th F : ℝ)

theorem temperature_on_Tuesday :
  (T + W + Th) / 3 = 52 →
  (W + Th + F) / 3 = 54 →
  F = 53 →
  T = 47 := by
  intros h₁ h₂ h₃
  sorry

end temperature_on_Tuesday_l156_156307


namespace min_value_of_quadratic_l156_156692

theorem min_value_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ x : ℝ, (x = -p / 2) ∧ ∀ y : ℝ, (y^2 + p * y + q) ≥ ((-p/2)^2 + p * (-p/2) + q) :=
sorry

end min_value_of_quadratic_l156_156692


namespace translated_parabola_eq_new_equation_l156_156438

-- Definitions following directly from the condition
def original_parabola (x : ℝ) : ℝ := 2 * x^2
def new_vertex : (ℝ × ℝ) := (-2, -2)
def new_parabola (x : ℝ) : ℝ := 2 * (x + 2)^2 - 2

-- Statement to prove the equivalency of the translated parabola equation
theorem translated_parabola_eq_new_equation :
  (∀ (x : ℝ), (original_parabola x = new_parabola (x - 2))) :=
by
  sorry

end translated_parabola_eq_new_equation_l156_156438


namespace sum_primes_upto_20_l156_156011

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l156_156011


namespace average_messages_correct_l156_156924

-- Definitions for the conditions
def messages_monday := 220
def messages_tuesday := 1 / 2 * messages_monday
def messages_wednesday := 50
def messages_thursday := 50
def messages_friday := 50

-- Definition for the total and average messages
def total_messages := messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday
def average_messages := total_messages / 5

-- Statement to prove
theorem average_messages_correct : average_messages = 96 := 
by sorry

end average_messages_correct_l156_156924


namespace pine_taller_than_maple_l156_156090

def height_maple : ℚ := 13 + 1 / 4
def height_pine : ℚ := 19 + 3 / 8

theorem pine_taller_than_maple :
  (height_pine - height_maple = 6 + 1 / 8) :=
sorry

end pine_taller_than_maple_l156_156090


namespace measure_of_angle_D_l156_156523

theorem measure_of_angle_D 
  (A B C D E F : ℝ)
  (h1 : A = B) (h2 : B = C) (h3 : C = F)
  (h4 : D = E) (h5 : A = D - 30) 
  (sum_angles : A + B + C + D + E + F = 720) : 
  D = 140 :=
by
  sorry

end measure_of_angle_D_l156_156523


namespace find_number_of_packs_l156_156805

-- Define the cost of a pack of Digimon cards
def cost_pack_digimon : ℝ := 4.45

-- Define the cost of the deck of baseball cards
def cost_deck_baseball : ℝ := 6.06

-- Define the total amount spent
def total_spent : ℝ := 23.86

-- Define the number of packs of Digimon cards Keith bought
def number_of_packs (D : ℝ) : Prop :=
  cost_pack_digimon * D + cost_deck_baseball = total_spent

-- Prove the number of packs is 4
theorem find_number_of_packs : ∃ D, number_of_packs D ∧ D = 4 :=
by
  -- the proof will be inserted here
  sorry

end find_number_of_packs_l156_156805


namespace pairwise_coprime_circle_l156_156811

theorem pairwise_coprime_circle :
  ∃ (circle : Fin 100 → ℕ),
    (∀ i, Nat.gcd (circle i) (Nat.gcd (circle ((i + 1) % 100)) (circle ((i - 1) % 100))) = 1) → 
    ∀ i j, i ≠ j → Nat.gcd (circle i) (circle j) = 1 :=
by
  sorry

end pairwise_coprime_circle_l156_156811


namespace compute_expression_l156_156197

theorem compute_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 :=
by
  have h : a = 3 := h1
  have k : b = 2 := h2
  rw [h, k]
  sorry

end compute_expression_l156_156197


namespace smallest_possible_value_expression_l156_156348

open Real

noncomputable def min_expression_value (a b c : ℝ) : ℝ :=
  (a + b)^2 + (b - c)^2 + (c - a)^2 / a^2

theorem smallest_possible_value_expression :
  ∀ (a b c : ℝ), a > b → b > c → a + c = 2 * b → a ≠ 0 → min_expression_value a b c = 7 / 2 := by
  sorry

end smallest_possible_value_expression_l156_156348


namespace jason_average_messages_l156_156927

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end jason_average_messages_l156_156927


namespace equivalent_single_percentage_change_l156_156324

theorem equivalent_single_percentage_change :
  let original_price : ℝ := 250
  let num_items : ℕ := 400
  let first_increase : ℝ := 0.15
  let second_increase : ℝ := 0.20
  let discount : ℝ := -0.10
  let third_increase : ℝ := 0.25

  -- Calculations
  let price_after_first_increase := original_price * (1 + first_increase)
  let price_after_second_increase := price_after_first_increase * (1 + second_increase)
  let price_after_discount := price_after_second_increase * (1 + discount)
  let final_price := price_after_discount * (1 + third_increase)

  -- Calculate percentage change
  let percentage_change := ((final_price - original_price) / original_price) * 100

  percentage_change = 55.25 :=
by
  sorry

end equivalent_single_percentage_change_l156_156324


namespace range_of_g_l156_156807

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x)^4 + (arcsin x)^4

theorem range_of_g :
  set.range g = set.Icc (-(π^4 / 8)) (π^4 / 4) :=
sorry

end range_of_g_l156_156807


namespace letters_posting_ways_l156_156655

theorem letters_posting_ways :
  let mailboxes := 4
  let letters := 3
  (mailboxes ^ letters) = 64 :=
by
  let mailboxes := 4
  let letters := 3
  show (mailboxes ^ letters) = 64
  sorry

end letters_posting_ways_l156_156655


namespace johns_disposable_income_increase_l156_156256

noncomputable def percentage_increase_of_johns_disposable_income
  (weekly_income_before : ℝ) (weekly_income_after : ℝ)
  (tax_rate_before : ℝ) (tax_rate_after : ℝ)
  (monthly_expense : ℝ) : ℝ :=
  let disposable_income_before := (weekly_income_before * (1 - tax_rate_before) * 4 - monthly_expense)
  let disposable_income_after := (weekly_income_after * (1 - tax_rate_after) * 4 - monthly_expense)
  (disposable_income_after - disposable_income_before) / disposable_income_before * 100

theorem johns_disposable_income_increase :
  percentage_increase_of_johns_disposable_income 60 70 0.15 0.18 100 = 24.62 :=
  by
  sorry

end johns_disposable_income_increase_l156_156256


namespace edward_spring_earnings_l156_156746

-- Define the relevant constants and the condition
def springEarnings := 2
def summerEarnings := 27
def expenses := 5
def totalEarnings := 24

-- The condition
def edwardCondition := summerEarnings - expenses = 22

-- The statement to prove
theorem edward_spring_earnings (h : edwardCondition) : springEarnings + 22 = totalEarnings :=
by
  -- Provide the proof here, but we'll use sorry to skip it
  sorry

end edward_spring_earnings_l156_156746


namespace nested_fraction_evaluation_l156_156747

theorem nested_fraction_evaluation : 
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))))) = (21 / 55) :=
by
  sorry

end nested_fraction_evaluation_l156_156747


namespace solve_inequality_l156_156615

theorem solve_inequality (x : ℝ) : 
  (3 * x - 6 > 12 - 2 * x + x^2) ↔ (-1 < x ∧ x < 6) :=
sorry

end solve_inequality_l156_156615


namespace find_constants_l156_156535

open Matrix 

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem find_constants :
  ∃ c d : ℝ, c = 1/12 ∧ d = 1/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end find_constants_l156_156535


namespace quadratic_sum_of_b_and_c_l156_156441

theorem quadratic_sum_of_b_and_c :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 - 20 * x + 36 = (x + b)^2 + c) ∧ b + c = -74 :=
by
  sorry

end quadratic_sum_of_b_and_c_l156_156441


namespace solve_for_a_minus_c_l156_156972

theorem solve_for_a_minus_c 
  (a b c d : ℝ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
  sorry

end solve_for_a_minus_c_l156_156972


namespace intersection_M_N_l156_156977

theorem intersection_M_N :
  let M := { x : ℝ | abs x ≤ 2 }
  let N := {-1, 0, 2, 3}
  M ∩ N = {-1, 0, 2} :=
by
  sorry

end intersection_M_N_l156_156977


namespace correct_option_B_l156_156111

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end correct_option_B_l156_156111


namespace g_at_zero_l156_156553

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_at_zero : g 0 = -Real.sqrt 2 :=
by
  -- proof to be completed
  sorry

end g_at_zero_l156_156553


namespace division_of_decimals_l156_156706

theorem division_of_decimals : (0.05 / 0.002) = 25 :=
by
  -- Proof will be filled here
  sorry

end division_of_decimals_l156_156706


namespace find_number_l156_156022

theorem find_number (number : ℚ) 
  (H1 : 8 * 60 = 480)
  (H2 : number / 6 = 16 / 480) :
  number = 1 / 5 := 
by
  sorry

end find_number_l156_156022


namespace num_adults_l156_156173

-- Definitions of the conditions
def num_children : Nat := 11
def child_ticket_cost : Nat := 4
def adult_ticket_cost : Nat := 8
def total_cost : Nat := 124

-- The proof problem statement
theorem num_adults (A : Nat) 
  (h1 : total_cost = num_children * child_ticket_cost + A * adult_ticket_cost) : 
  A = 10 := 
by
  sorry

end num_adults_l156_156173


namespace exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l156_156182

-- Define the notion of a balanced integer.
def isBalanced (N : ℕ) : Prop :=
  N = 1 ∨ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ N = p ^ (2 * k)

-- Define the polynomial P(x) = (x + a)(x + b)
def P (a b x : ℕ) : ℕ := (x + a) * (x + b)

theorem exists_distinct_a_b_all_P_balanced :
  ∃ (a b : ℕ), a ≠ b ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → isBalanced (P a b n) :=
sorry

theorem P_balanced_implies_a_eq_b (a b : ℕ) :
  (∀ n : ℕ, isBalanced (P a b n)) → a = b :=
sorry

end exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l156_156182


namespace arc_length_of_sector_l156_156080

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 3) :
  l = r * θ := by
  sorry

end arc_length_of_sector_l156_156080


namespace shelves_of_mystery_books_l156_156476

theorem shelves_of_mystery_books (total_books : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ) (M : ℕ) 
  (h_total_books : total_books = 54) 
  (h_picture_shelves : picture_shelves = 4) 
  (h_books_per_shelf : books_per_shelf = 6)
  (h_mystery_books : total_books - picture_shelves * books_per_shelf = M * books_per_shelf) :
  M = 5 :=
by
  sorry

end shelves_of_mystery_books_l156_156476


namespace find_a_l156_156897

variable {f : ℝ → ℝ}

-- Conditions
variables (a : ℝ) (domain : Set ℝ := Set.Ioo (3 - 2 * a) (a + 1))
variable (even_f : ∀ x, f (x + 1) = f (- (x + 1)))

-- The theorem stating the problem
theorem find_a (h : ∀ x, x ∈ domain ↔ x ∈ Set.Ioo (3 - 2 * a) (a + 1)) : a = 2 := by
  sorry

end find_a_l156_156897


namespace range_of_a_l156_156242

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end range_of_a_l156_156242


namespace count_consecutive_sets_sum_15_l156_156904

theorem count_consecutive_sets_sum_15 : 
  ∃ n : ℕ, 
    (n > 0 ∧
    ∃ a : ℕ, 
      (n ≥ 2 ∧ 
      ∃ s : (Finset ℕ), 
        (∀ x ∈ s, x ≥ 1) ∧ 
        (s.sum id = 15))
  ) → 
  n = 2 :=
  sorry

end count_consecutive_sets_sum_15_l156_156904


namespace range_of_k_l156_156762

-- Definitions for the conditions of p and q
def is_ellipse (k : ℝ) : Prop := (0 < k) ∧ (k < 4)
def is_hyperbola (k : ℝ) : Prop := 1 < k ∧ k < 3

-- The main proposition
theorem range_of_k (k : ℝ) : (is_ellipse k ∨ is_hyperbola k) → (1 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l156_156762


namespace sin_120_eq_half_l156_156210

theorem sin_120_eq_half :
  let Q := (-(Real.sqrt 3) / 2, 1 / 2) in -- coordinates for Q
  sin (120 * (Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_120_eq_half_l156_156210


namespace option_b_results_in_2x_cubed_l156_156451

variable (x : ℝ)

theorem option_b_results_in_2x_cubed : |x^3| + x^3 = 2 * x^3 := 
sorry

end option_b_results_in_2x_cubed_l156_156451


namespace isosceles_triangle_perimeter_l156_156830

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4)
  (h3 : a = b ∨ 2 * a > b) :
  (a ≠ b ∨ b = 2 * a) → 
  ∃ p : ℝ, p = a + b + b ∧ p = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l156_156830


namespace total_stairs_climbed_l156_156270

theorem total_stairs_climbed (samir_stairs veronica_stairs ravi_stairs total_stairs_climbed : ℕ) 
  (h_samir : samir_stairs = 318)
  (h_veronica : veronica_stairs = (318 / 2) + 18)
  (h_ravi : ravi_stairs = (3 * veronica_stairs) / 2) :
  samir_stairs + veronica_stairs + ravi_stairs = total_stairs_climbed ->
  total_stairs_climbed = 761 :=
by
  sorry

end total_stairs_climbed_l156_156270


namespace recurring_six_denominator_l156_156950

theorem recurring_six_denominator : 
  let T := (0.6666...) in
  (T = 2 / 3) → (denominator (2 / 3) = 3) :=
by
  sorry

end recurring_six_denominator_l156_156950


namespace diana_higher_than_apollo_probability_l156_156618

def diana_die : ℕ := 8
def apollo_die : ℕ := 6

def total_outcomes : ℕ := diana_die * apollo_die
def successful_outcomes : ℕ := 27

theorem diana_higher_than_apollo_probability :
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = 9 / 16 := by
  sorry

end diana_higher_than_apollo_probability_l156_156618


namespace full_seasons_already_aired_l156_156928

variable (days_until_premiere : ℕ)
variable (episodes_per_day : ℕ)
variable (episodes_per_season : ℕ)

theorem full_seasons_already_aired (h_days : days_until_premiere = 10)
                                  (h_episodes_day : episodes_per_day = 6)
                                  (h_episodes_season : episodes_per_season = 15) :
  (days_until_premiere * episodes_per_day) / episodes_per_season = 4 := by
  sorry

end full_seasons_already_aired_l156_156928


namespace obtuse_angles_at_intersection_l156_156446

theorem obtuse_angles_at_intersection (lines_intersect_x_at_diff_points : Prop) (lines_not_perpendicular : Prop) 
(lines_form_obtuse_angle_at_intersection : Prop) : 
(lines_intersect_x_at_diff_points ∧ lines_not_perpendicular ∧ lines_form_obtuse_angle_at_intersection) → 
  ∃ (n : ℕ), n = 2 :=
by 
  sorry

end obtuse_angles_at_intersection_l156_156446


namespace martian_year_length_ratio_l156_156841

theorem martian_year_length_ratio :
  let EarthDay := 24 -- hours
  let MarsDay := EarthDay + 2 / 3 -- hours (since 40 minutes is 2/3 of an hour)
  let MartianYearDays := 668
  let EarthYearDays := 365.25
  (MartianYearDays * MarsDay) / EarthYearDays = 1.88 := by
{
  sorry
}

end martian_year_length_ratio_l156_156841


namespace max_value_sum_faces_edges_vertices_l156_156590

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

def pyramid_faces_added : ℕ := 4
def pyramid_base_faces_covered : ℕ := 1
def pyramid_edges_added : ℕ := 4
def pyramid_vertices_added : ℕ := 1

def resulting_faces : ℕ := rectangular_prism_faces - pyramid_base_faces_covered + pyramid_faces_added
def resulting_edges : ℕ := rectangular_prism_edges + pyramid_edges_added
def resulting_vertices : ℕ := rectangular_prism_vertices + pyramid_vertices_added

def sum_resulting_faces_edges_vertices : ℕ := resulting_faces + resulting_edges + resulting_vertices

theorem max_value_sum_faces_edges_vertices : sum_resulting_faces_edges_vertices = 34 :=
by
  sorry

end max_value_sum_faces_edges_vertices_l156_156590


namespace distance_from_focus_to_asymptote_l156_156889

theorem distance_from_focus_to_asymptote
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a = b)
  (h2 : |a| / Real.sqrt 2 = 2) :
  Real.sqrt 2 * 2 = 2 * Real.sqrt 2 :=
by
  sorry

end distance_from_focus_to_asymptote_l156_156889


namespace diamond_not_commutative_diamond_not_associative_l156_156491

noncomputable def diamond (x y : ℝ) : ℝ :=
  x^2 * y / (x + y + 1)

theorem diamond_not_commutative (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x ≠ y → diamond x y ≠ diamond y x :=
by
  intro hxy
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : x^2 * y * (y + x + 1) = y^2 * x * (x + y + 1) := by
    sorry
  -- Simplify the equation to show the contradiction
  sorry

theorem diamond_not_associative (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (diamond x y) ≠ (diamond y x) → (diamond (diamond x y) z) ≠ (diamond x (diamond y z)) :=
by
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : (diamond x y)^2 * z / (diamond x y + z + 1) ≠ (x^2 * (diamond y z) / (x + diamond y z + 1)) :=
    by sorry
  -- Simplify the equation to show the contradiction
  sorry

end diamond_not_commutative_diamond_not_associative_l156_156491


namespace jane_original_number_l156_156254

theorem jane_original_number (x : ℝ) (h : 5 * (3 * x + 16) = 250) : x = 34 / 3 := 
sorry

end jane_original_number_l156_156254


namespace balls_in_boxes_l156_156909

theorem balls_in_boxes :
  let n := 7
  let k := 3
  (Nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  let n := 7
  let k := 3
  sorry

end balls_in_boxes_l156_156909


namespace sum_primes_upto_20_l156_156012

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l156_156012


namespace solve_equation1_solve_equation2_l156_156120

theorem solve_equation1 (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by
  sorry

theorem solve_equation2 (x : ℝ) : 2 * (x - 3)^2 = x - 3 ↔ (x = 3/2 ∨ x = 7/2) :=
by
  sorry

end solve_equation1_solve_equation2_l156_156120


namespace water_left_l156_156671

-- Conditions
def initial_water : ℚ := 3
def water_used : ℚ := 11 / 8

-- Proposition to be proven
theorem water_left :
  initial_water - water_used = 13 / 8 := by
  sorry

end water_left_l156_156671


namespace points_lie_on_ellipse_l156_156492

open Real

noncomputable def curve_points_all_lie_on_ellipse (s: ℝ) : Prop :=
  let x := 2 * cos s + 2 * sin s
  let y := 4 * (cos s - sin s)
  (x^2 / 8 + y^2 / 32 = 1)

-- Below statement defines the theorem we aim to prove:
theorem points_lie_on_ellipse (s: ℝ) : curve_points_all_lie_on_ellipse s :=
sorry -- This "sorry" is to indicate that the proof is omitted.

end points_lie_on_ellipse_l156_156492


namespace roots_greater_than_two_l156_156280

variable {x m : ℝ}

theorem roots_greater_than_two (h : ∀ x, x^2 - 2 * m * x + 4 = 0 → (∃ a b : ℝ, a > 2 ∧ b < 2 ∧ x = a ∨ x = b)) : 
  m > 2 :=
by
  sorry

end roots_greater_than_two_l156_156280


namespace cherry_ratio_l156_156475

theorem cherry_ratio (total_lollipops cherry_lollipops watermelon_lollipops sour_apple_lollipops grape_lollipops : ℕ) 
  (h_total : total_lollipops = 42) 
  (h_rest_equally_distributed : watermelon_lollipops = sour_apple_lollipops ∧ sour_apple_lollipops = grape_lollipops) 
  (h_grape : grape_lollipops = 7) 
  (h_total_sum : cherry_lollipops + watermelon_lollipops + sour_apple_lollipops + grape_lollipops = total_lollipops) : 
  cherry_lollipops = 21 ∧ (cherry_lollipops : ℚ) / total_lollipops = 1 / 2 :=
by
  sorry

end cherry_ratio_l156_156475


namespace max_value_in_range_l156_156241

noncomputable def x_range : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}

noncomputable def expression (x : ℝ) : ℝ :=
  Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

theorem max_value_in_range :
  ∀ x ∈ x_range, expression x ≤ (11 / 6) * Real.sqrt 3 :=
sorry

end max_value_in_range_l156_156241


namespace length_of_the_bridge_l156_156695

-- Conditions
def train_length : ℝ := 80
def train_speed_kmh : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Conversion factor
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Calculation
noncomputable def train_speed_ms : ℝ := train_speed_kmh * km_to_m / hr_to_s
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

-- Proof statement
theorem length_of_the_bridge : bridge_length = 295 :=
by
  sorry

end length_of_the_bridge_l156_156695


namespace absent_present_probability_l156_156654

theorem absent_present_probability : 
  ∀ (p_absent_normal p_absent_workshop p_present_workshop : ℚ), 
    p_absent_normal = 1 / 20 →
    p_absent_workshop = 2 * p_absent_normal →
    p_present_workshop = 1 - p_absent_workshop →
    p_absent_workshop = 1 / 10 →
    (p_present_workshop * p_absent_workshop + p_absent_workshop * p_present_workshop) * 100 = 18 :=
by
  intros
  sorry

end absent_present_probability_l156_156654


namespace david_course_hours_l156_156998

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l156_156998


namespace number_of_children_l156_156835

variables (n : ℕ) (y : ℕ) (d : ℕ)

def sum_of_ages (n : ℕ) (y : ℕ) (d : ℕ) : ℕ :=
  n * y + d * (n * (n - 1) / 2)

theorem number_of_children (H1 : sum_of_ages n 6 3 = 60) : n = 6 :=
by {
  sorry
}

end number_of_children_l156_156835


namespace equalize_foma_ierema_l156_156145

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l156_156145


namespace steve_initial_amount_l156_156432

theorem steve_initial_amount
  (P : ℝ) 
  (h : (1.1^2) * P = 121) : 
  P = 100 := 
by 
  sorry

end steve_initial_amount_l156_156432


namespace largest_possible_perimeter_l156_156696

theorem largest_possible_perimeter :
  ∃ (l w : ℕ), 8 * l + 8 * w = l * w - 1 ∧ 2 * l + 2 * w = 164 :=
sorry

end largest_possible_perimeter_l156_156696


namespace golf_balls_dozen_count_l156_156420

theorem golf_balls_dozen_count (n d : Nat) (h1 : n = 108) (h2 : d = 12) : n / d = 9 :=
by
  sorry

end golf_balls_dozen_count_l156_156420


namespace total_cards_proof_l156_156188

-- Define the standard size of a deck of playing cards
def standard_deck_size : Nat := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : Nat := 6

-- Define the number of additional cards the shopkeeper has
def additional_cards : Nat := 7

-- Define the total number of cards from the complete decks
def total_deck_cards : Nat := complete_decks * standard_deck_size

-- Define the total number of all cards the shopkeeper has
def total_cards : Nat := total_deck_cards + additional_cards

-- The theorem statement that we need to prove
theorem total_cards_proof : total_cards = 319 := by
  sorry

end total_cards_proof_l156_156188


namespace least_positive_value_l156_156293

theorem least_positive_value (x y z : ℤ) : ∃ x y z : ℤ, 0 < 72 * x + 54 * y + 36 * z ∧ ∀ (a b c : ℤ), 0 < 72 * a + 54 * b + 36 * c → 72 * x + 54 * y + 36 * z ≤ 72 * a + 54 * b + 36 * c :=
sorry

end least_positive_value_l156_156293


namespace equalize_foma_ierema_l156_156155

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l156_156155


namespace distance_proof_l156_156714

-- Definitions from the conditions
def avg_speed_to_retreat := 50
def avg_speed_back_home := 75
def total_round_trip_time := 10
def distance_between_home_and_retreat := 300

-- Theorem stating the problem
theorem distance_proof 
  (D : ℝ)
  (h1 : D / avg_speed_to_retreat + D / avg_speed_back_home = total_round_trip_time) :
  D = distance_between_home_and_retreat :=
sorry

end distance_proof_l156_156714


namespace unique_peg_placement_l156_156303

noncomputable def peg_placement := 
  ∃! f : (Fin 6 → Fin 6 → Option (Fin 5)), 
    (∀ i j, f i j = some 0 → (∀ k, k ≠ i → f k j ≠ some 0) ∧ (∀ l, l ≠ j → f i l ≠ some 0)) ∧  -- Yellow pegs
    (∀ i j, f i j = some 1 → (∀ k, k ≠ i → f k j ≠ some 1) ∧ (∀ l, l ≠ j → f i l ≠ some 1)) ∧  -- Red pegs
    (∀ i j, f i j = some 2 → (∀ k, k ≠ i → f k j ≠ some 2) ∧ (∀ l, l ≠ j → f i l ≠ some 2)) ∧  -- Green pegs
    (∀ i j, f i j = some 3 → (∀ k, k ≠ i → f k j ≠ some 3) ∧ (∀ l, l ≠ j → f i l ≠ some 3)) ∧  -- Blue pegs
    (∀ i j, f i j = some 4 → (∀ k, k ≠ i → f k j ≠ some 4) ∧ (∀ l, l ≠ j → f i l ≠ some 4)) ∧  -- Orange pegs
    (∃! i j, f i j = some 0) ∧
    (∃! i j, f i j = some 1) ∧
    (∃! i j, f i j = some 2) ∧
    (∃! i j, f i j = some 3) ∧
    (∃! i j, f i j = some 4)
    
theorem unique_peg_placement : peg_placement :=
sorry

end unique_peg_placement_l156_156303


namespace find_m_of_equation_has_positive_root_l156_156785

theorem find_m_of_equation_has_positive_root :
  (∃ x : ℝ, 0 < x ∧ (x - 1) / (x - 5) = (m * x) / (10 - 2 * x)) → m = -8 / 5 :=
by
  sorry

end find_m_of_equation_has_positive_root_l156_156785


namespace foma_should_give_ierema_55_coins_l156_156134

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156134


namespace sum_of_squares_eq_23456_l156_156515

theorem sum_of_squares_eq_23456 (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end sum_of_squares_eq_23456_l156_156515


namespace total_sticks_needed_l156_156549

theorem total_sticks_needed (simon_sticks gerry_sticks micky_sticks darryl_sticks : ℕ):
  simon_sticks = 36 →
  gerry_sticks = (2 * simon_sticks) / 3 →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  darryl_sticks = simon_sticks + gerry_sticks + micky_sticks + 1 →
  simon_sticks + gerry_sticks + micky_sticks + darryl_sticks = 259 :=
by
  intros h_simon h_gerry h_micky h_darryl
  rw [h_simon, h_gerry, h_micky, h_darryl]
  norm_num
  sorry

end total_sticks_needed_l156_156549


namespace fibonacci_p_arithmetic_periodic_l156_156105

-- Define p-arithmetic system and its properties
def p_arithmetic (p : ℕ) : Prop :=
  ∀ (a : ℤ), a ≠ 0 → a^(p-1) = 1

-- Define extraction of sqrt(5)
def sqrt5_extractable (p : ℕ) : Prop :=
  ∃ (r : ℝ), r^2 = 5

-- Define Fibonacci sequence in p-arithmetic
def fibonacci_p_arithmetic (p : ℕ) (v : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, v (n+2) = v (n+1) + v n

-- Main Theorem
theorem fibonacci_p_arithmetic_periodic (p : ℕ) (v : ℕ → ℤ) :
  p_arithmetic p →
  sqrt5_extractable p →
  fibonacci_p_arithmetic p v →
  (∀ k : ℕ, v (k + p) = v k) :=
by
  intros _ _ _
  sorry

end fibonacci_p_arithmetic_periodic_l156_156105


namespace positive_integer_sixk_l156_156333

theorem positive_integer_sixk (n : ℕ) :
  (∃ d1 d2 d3 : ℕ, d1 < d2 ∧ d2 < d3 ∧ d1 + d2 + d3 = n ∧ d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n) ↔ (∃ k : ℕ, n = 6 * k) :=
by
  sorry

end positive_integer_sixk_l156_156333


namespace part_a_part_b_l156_156852

open Finset

variables {X : Type} (ℱ : Finset (Finset X)) (n k : ℕ)

def condition_1 : Prop :=
  ℱ ⊆ powersetLen k (univ : Finset X)

def condition_2 : Prop :=
  ∀ {A B C : Finset X}, A ∈ ℱ → B ∈ ℱ → C ∈ ℱ →
  A ≠ B → B ≠ C → C ≠ A →
  at_most_one_empty [A ∩ B, B ∩ C, C ∩ A]

def at_most_one_empty (sets : List (Finset X)) : Prop :=
  length (filter (== ∅) sets) ≤ 1

theorem part_a (hX : card (univ : Finset X) = n)
  (h_subset : condition_1 ℱ)
  (h_inter : condition_2 ℱ)
  (h_kn : k ≤ n / 2) :
  card ℱ ≤ max 1 (4 - n / k) * nat.choose (n - 1) (k - 1) :=
sorry

theorem part_b (hX : card (univ : Finset X) = n)
  (h_subset : condition_1 ℱ)
  (h_inter : condition_2 ℱ)
  (h_kn : k ≤ n / 3)
  (hab_eq : card ℱ = max 1 (4 - n / k) * nat.choose (n - 1) (k - 1)) :
  true :=  -- Specify the exact condition for equality
sorry

end part_a_part_b_l156_156852


namespace exists_distinct_ij_l156_156932

theorem exists_distinct_ij (n : ℕ) (a : Fin n → ℤ) (h_distinct : Function.Injective a) (h_n_ge_3 : 3 ≤ n) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k, (a i + a j) ∣ 3 * a k → False) :=
by
  sorry

end exists_distinct_ij_l156_156932


namespace smooth_transition_l156_156985

theorem smooth_transition (R : ℝ) (x₀ y₀ : ℝ) :
  ∃ m : ℝ, ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = R^2 → y - y₀ = m * (x - x₀) :=
sorry

end smooth_transition_l156_156985


namespace new_rate_of_commission_l156_156693

theorem new_rate_of_commission 
  (R1 : ℝ) (R1_eq : R1 = 0.04) 
  (slump_percentage : ℝ) (slump_percentage_eq : slump_percentage = 0.20000000000000007)
  (income_unchanged : ∀ (B B_new : ℝ) (R2 : ℝ),
    B_new = B * (1 - slump_percentage) →
    B * R1 = B_new * R2 → 
    R2 = 0.05) : 
  true := 
by 
  sorry

end new_rate_of_commission_l156_156693


namespace complement_intersection_l156_156501

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x > 0}

def B : Set ℝ := {x | -3 < x ∧ x < 1}

def compA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_intersection :
  (compA ∩ B) = {x | 0 ≤ x ∧ x < 1} := by
  -- The proof goes here
  sorry

end complement_intersection_l156_156501


namespace M_eq_N_l156_156769

def M (u : ℤ) : Prop := ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l
def N (u : ℤ) : Prop := ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r

theorem M_eq_N : ∀ u : ℤ, M u ↔ N u := by
  sorry

end M_eq_N_l156_156769


namespace percent_increase_equilateral_triangles_l156_156595

theorem percent_increase_equilateral_triangles :
  let s₁ := 3
  let s₂ := 2 * s₁
  let s₃ := 2 * s₂
  let s₄ := 2 * s₃
  let P₁ := 3 * s₁
  let P₄ := 3 * s₄
  (P₄ - P₁) / P₁ * 100 = 700 :=
by
  sorry

end percent_increase_equilateral_triangles_l156_156595


namespace triangle_area_and_fraction_of_square_l156_156636

theorem triangle_area_and_fraction_of_square 
  (a b c s : ℕ) 
  (h_triangle : a = 9 ∧ b = 40 ∧ c = 41)
  (h_square : s = 41)
  (h_right_angle : a^2 + b^2 = c^2) :
  let area_triangle := (a * b) / 2
  let area_square := s^2
  let fraction := (a * b) / (2 * s^2)
  area_triangle = 180 ∧ fraction = 180 / 1681 := 
by
  sorry

end triangle_area_and_fraction_of_square_l156_156636


namespace foma_gives_ierema_55_l156_156152

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l156_156152


namespace only_positive_integer_cube_less_than_triple_l156_156565

theorem only_positive_integer_cube_less_than_triple (n : ℕ) (h : 0 < n ∧ n^3 < 3 * n) : n = 1 :=
sorry

end only_positive_integer_cube_less_than_triple_l156_156565


namespace data_transmission_time_l156_156481

def chunks_per_block : ℕ := 1024
def blocks : ℕ := 30
def transmission_rate : ℕ := 256
def seconds_in_minute : ℕ := 60

theorem data_transmission_time :
  (blocks * chunks_per_block) / transmission_rate / seconds_in_minute = 2 :=
by
  sorry

end data_transmission_time_l156_156481


namespace cartesian_to_polar_coords_l156_156232

theorem cartesian_to_polar_coords :
  ∃ ρ θ : ℝ, 
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) ∧ 
  (-1, Real.sqrt 3) = (ρ * Real.cos θ, ρ * Real.sin θ) :=
sorry

end cartesian_to_polar_coords_l156_156232


namespace problem_solution_l156_156745

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 :=
by
  sorry

end problem_solution_l156_156745


namespace foma_should_give_ierema_55_coins_l156_156136

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l156_156136


namespace ball_radius_l156_156719

theorem ball_radius (x r : ℝ) 
  (h1 : (15 : ℝ) ^ 2 + x ^ 2 = r ^ 2) 
  (h2 : x + 12 = r) : 
  r = 15.375 := 
sorry

end ball_radius_l156_156719


namespace heesu_has_greatest_sum_l156_156431

def sum_cards (cards : List Int) : Int :=
  cards.foldl (· + ·) 0

theorem heesu_has_greatest_sum :
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sum_cards heesu_cards > sum_cards sora_cards ∧ sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sorry

end heesu_has_greatest_sum_l156_156431


namespace pentagons_from_15_points_l156_156341

theorem pentagons_from_15_points (n : ℕ) (h : n = 15) : (nat.choose 15 5) = 3003 := by
  rw h
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end pentagons_from_15_points_l156_156341


namespace egg_roll_ratio_l156_156670

-- Define the conditions as hypotheses 
variables (Matthew_eats Patrick_eats Alvin_eats : ℕ)

-- Define the specific conditions
def conditions : Prop :=
  (Matthew_eats = 6) ∧
  (Patrick_eats = Alvin_eats / 2) ∧
  (Alvin_eats = 4)

-- Define the ratio of Matthew's egg rolls to Patrick's egg rolls
def ratio (a b : ℕ) := a / b

-- State the theorem with the corresponding proof problem
theorem egg_roll_ratio : conditions Matthew_eats Patrick_eats Alvin_eats → ratio Matthew_eats Patrick_eats = 3 :=
by
  -- Proof is not required as mentioned. Adding sorry to skip the proof.
  sorry

end egg_roll_ratio_l156_156670


namespace exists_ε_for_prob_l156_156440

noncomputable def ε : ℝ := 0.02

theorem exists_ε_for_prob : 
  ∃ ε > 0, 
  (∀ (X : ℕ) (n : ℕ) (p : ℝ), 
    n = 900 ∧ p = 0.5 ∧ 
    (1 / (real.sqrt (n * p * (1 - p))) * (X - n * p)) ∈ 
    (set.Icc (-ε) ε)) → 
  (prob (λ (X : ℕ) (n : ℕ) (p : ℝ), 
    (1 / (real.sqrt (n * p * (1 - p))) * (X - n * p)) ∈ 
    (set.Icc (-ε) ε) ∧ n = 900 ∧ p = 0.5) 
  = 0.77) :=
by
  use 0.02
  sorry

end exists_ε_for_prob_l156_156440


namespace geometric_sequence_sum_a_l156_156057

theorem geometric_sequence_sum_a (a : ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = 4^n + a) :
  a = -1 :=
sorry

end geometric_sequence_sum_a_l156_156057


namespace frog_vertical_boundary_prob_l156_156981

-- Define the type of points on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define the type of the rectangle
structure Rectangle where
  left_bottom : Point
  right_top : Point

-- Conditions
def start_point : Point := ⟨2, 3⟩
def boundary : Rectangle := ⟨⟨0, 0⟩, ⟨5, 5⟩⟩

-- Define the probability function
noncomputable def P (p : Point) : ℚ := sorry

-- Symmetry relations and recursive relations
axiom symmetry_P23 : P ⟨2, 3⟩ = P ⟨3, 3⟩
axiom symmetry_P22 : P ⟨2, 2⟩ = P ⟨3, 2⟩
axiom recursive_P23 : P ⟨2, 3⟩ = 1 / 4 + 1 / 4 * P ⟨2, 2⟩ + 1 / 4 * P ⟨1, 3⟩ + 1 / 4 * P ⟨3, 3⟩

-- Main Theorem
theorem frog_vertical_boundary_prob :
  P start_point = 2 / 3 := sorry

end frog_vertical_boundary_prob_l156_156981


namespace yard_length_l156_156789

-- Define the given conditions
def num_trees : ℕ := 26
def dist_between_trees : ℕ := 13

-- Calculate the length of the yard
def num_gaps : ℕ := num_trees - 1
def length_of_yard : ℕ := num_gaps * dist_between_trees

-- Theorem statement: the length of the yard is 325 meters
theorem yard_length : length_of_yard = 325 := by
  sorry

end yard_length_l156_156789


namespace probability_of_D_l156_156720

theorem probability_of_D (P_A P_B P_C P_D : ℚ) (hA : P_A = 1/4) (hB : P_B = 1/3) (hC : P_C = 1/6) 
  (hSum : P_A + P_B + P_C + P_D = 1) : P_D = 1/4 := 
by
  sorry

end probability_of_D_l156_156720


namespace number_is_fraction_l156_156073

theorem number_is_fraction (x : ℝ) : (0.30 * x = 0.25 * 40) → (x = 100 / 3) :=
begin
  intro h,
  sorry
end

end number_is_fraction_l156_156073


namespace solutionToEquations_solutionToInequalities_l156_156718

-- Part 1: Solve the system of equations
def solveEquations (x y : ℝ) : Prop :=
2 * x - y = 3 ∧ x + y = 6

theorem solutionToEquations (x y : ℝ) (h : solveEquations x y) : 
x = 3 ∧ y = 3 :=
sorry

-- Part 2: Solve the system of inequalities
def solveInequalities (x : ℝ) : Prop :=
3 * x > x - 4 ∧ (4 + x) / 3 > x + 2

theorem solutionToInequalities (x : ℝ) (h : solveInequalities x) : 
-2 < x ∧ x < -1 :=
sorry

end solutionToEquations_solutionToInequalities_l156_156718


namespace evaluate_expression_l156_156482

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) : 2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end evaluate_expression_l156_156482


namespace probability_valid_assignment_l156_156124

-- Define a type for Faces and Numbers
structure Dodecahedron :=
(faces : Fin 12 → ℕ) -- 12 faces with numbers 1 to 12 assigned uniquely to each face

-- Define adjacency relation
def is_adjacent (d : Dodecahedron) (i j : Fin 12) : Prop := sorry  -- needs dodecahedron's adjacency relation

-- Define directly opposite relation
def is_opposite (d : Dodecahedron) (i j : Fin 12) : Prop := sorry -- needs dodecahedron's opposite relation

-- Define a valid assignment
def valid_assignment (d : Dodecahedron) : Prop :=
∀ i j, (is_adjacent d i j ∨ is_opposite d i j) → ¬ (nat.succ d.faces i = d.faces j ∨ d.faces i = nat.succ d.faces j ∨ (d.faces i = 12 ∧ d.faces j = 1) ∨ (d.faces i = 1 ∧ d.faces j = 12))

-- Define the problem statement
theorem probability_valid_assignment : 
  ∃ m n : ℕ, (∀ d, valid_assignment d) → (m + n) = sorry := 
sorry

end probability_valid_assignment_l156_156124


namespace polygon_number_of_sides_l156_156521

theorem polygon_number_of_sides (h : ∀ (n : ℕ), (360 : ℝ) / (n : ℝ) = 1) : 
  360 = (1:ℝ) :=
  sorry

end polygon_number_of_sides_l156_156521


namespace harriet_smallest_stickers_l156_156770

theorem harriet_smallest_stickers 
  (S : ℕ) (a b c : ℕ)
  (h1 : S = 5 * a + 3)
  (h2 : S = 11 * b + 3)
  (h3 : S = 13 * c + 3)
  (h4 : S > 3) :
  S = 718 :=
by
  sorry

end harriet_smallest_stickers_l156_156770


namespace angle_B_l156_156795

open Real

-- Defining angles in degrees for clarity
noncomputable def sin_degree (d : ℝ) : ℝ :=
  sin (d * π / 180)

theorem angle_B (a b : ℝ) (angle_A : ℝ) (h₁ : a = 1) (h₂ : b = sqrt 3) (h₃ : angle_A = 30) :
  (sin_degree angle_A) * b = (sin_degree 60) * a :=
by
  sorry

end angle_B_l156_156795


namespace jacksonville_walmart_complaints_l156_156822

theorem jacksonville_walmart_complaints :
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  total_complaints = 576 :=
by
  -- Use the 'let' definitions from above to describe the proof problem
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  
  -- Here would be the place to write the proof steps, but it is skipped as per instructions
  sorry

end jacksonville_walmart_complaints_l156_156822


namespace sasha_age_l156_156471

theorem sasha_age :
  ∃ a : ℕ, 
    (M = 2 * a - 3) ∧
    (M = a + (a - 3)) ∧
    (a = 3) :=
by
  sorry

end sasha_age_l156_156471


namespace quadratic_inequality_l156_156955

-- Define the quadratic function and conditions
variables {a b c x0 y1 y2 y3 : ℝ}
variables (A : (a * x0^2 + b * x0 + c = 0))
variables (B : (a * (-2)^2 + b * (-2) + c = 0))
variables (C : (a + b + c) * (4 * a + 2 * b + c) < 0)
variables (D : a > 0)
variables (E1 : y1 = a * (-1)^2 + b * (-1) + c)
variables (E2 : y2 = a * (- (sqrt 2) / 2)^2 + b * (- (sqrt 2) / 2) + c)
variables (E3 : y3 = a * 1^2 + b * 1 + c)

-- Prove that y3 > y1 > y2
theorem quadratic_inequality : y3 > y1 ∧ y1 > y2 := by 
  sorry

end quadratic_inequality_l156_156955


namespace parities_of_E_10_11_12_l156_156862

noncomputable def E : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| (n + 3) => 2 * (E (n + 2)) + (E n)

theorem parities_of_E_10_11_12 :
  (E 10 % 2 = 0) ∧ (E 11 % 2 = 1) ∧ (E 12 % 2 = 1) := 
  by
  sorry

end parities_of_E_10_11_12_l156_156862


namespace angle_C_is_80_l156_156562

-- Define the angles A, B, and C
def isoscelesTriangle (A B C : ℕ) : Prop :=
  -- Triangle ABC is isosceles with A = B, and C is 30 degrees more than A
  A = B ∧ C = A + 30 ∧ A + B + C = 180

-- Problem: Prove that angle C is 80 degrees given the conditions
theorem angle_C_is_80 (A B C : ℕ) (h : isoscelesTriangle A B C) : C = 80 :=
by sorry

end angle_C_is_80_l156_156562


namespace train_length_is_300_l156_156594

-- Definitions based on the conditions
def trainCrossesPlatform (L V : ℝ) : Prop :=
  L + 400 = V * 42

def trainCrossesSignalPole (L V : ℝ) : Prop :=
  L = V * 18

-- The main theorem statement
theorem train_length_is_300 (L V : ℝ)
  (h1 : trainCrossesPlatform L V)
  (h2 : trainCrossesSignalPole L V) :
  L = 300 :=
by
  sorry

end train_length_is_300_l156_156594


namespace largest_apartment_size_l156_156036

theorem largest_apartment_size (cost_per_sqft : ℝ) (budget : ℝ) (s : ℝ) 
    (h₁ : cost_per_sqft = 1.20) 
    (h₂ : budget = 600) 
    (h₃ : 1.20 * s = 600) : 
    s = 500 := 
  sorry

end largest_apartment_size_l156_156036


namespace num_ways_to_choose_starting_lineup_l156_156026

-- Define conditions as Lean definitions
def team_size : ℕ := 12
def outfield_players : ℕ := 4

-- Define the function to compute the number of ways to choose the starting lineup
def choose_starting_lineup (team_size : ℕ) (outfield_players : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) outfield_players

-- The theorem to prove that the number of ways to choose the lineup is 3960
theorem num_ways_to_choose_starting_lineup : choose_starting_lineup team_size outfield_players = 3960 :=
  sorry

end num_ways_to_choose_starting_lineup_l156_156026


namespace plan1_has_higher_expected_loss_l156_156465

noncomputable def prob_minor_flooding : ℝ := 0.2
noncomputable def prob_major_flooding : ℝ := 0.05
noncomputable def cost_plan1 : ℝ := 4000
noncomputable def loss_major_plan1 : ℝ := 30000
noncomputable def loss_minor_plan2 : ℝ := 15000
noncomputable def loss_major_plan2 : ℝ := 30000

noncomputable def expected_loss_plan1 : ℝ :=
  (loss_major_plan1 * prob_major_flooding) + (cost_plan1 * prob_minor_flooding) + cost_plan1

noncomputable def expected_loss_plan2 : ℝ :=
  (loss_major_plan2 * prob_major_flooding) + (loss_minor_plan2 * prob_minor_flooding)

theorem plan1_has_higher_expected_loss : expected_loss_plan1 > expected_loss_plan2 :=
by
  sorry

end plan1_has_higher_expected_loss_l156_156465


namespace complaints_over_3_days_l156_156823

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end complaints_over_3_days_l156_156823


namespace rectangular_prism_diagonals_l156_156591

theorem rectangular_prism_diagonals :
  let l := 3
  let w := 4
  let h := 5
  let face_diagonals := 6 * 2
  let space_diagonals := 4
  face_diagonals + space_diagonals = 16 := 
by
  sorry

end rectangular_prism_diagonals_l156_156591


namespace pump_fill_time_without_leak_l156_156735

variable (T : ℕ)

def rate_pump (T : ℕ) : ℚ := 1 / T
def rate_leak : ℚ := 1 / 20

theorem pump_fill_time_without_leak : rate_pump T - rate_leak = rate_leak → T = 10 := by 
  intro h
  sorry

end pump_fill_time_without_leak_l156_156735


namespace solve_opposite_numbers_product_l156_156750

theorem solve_opposite_numbers_product :
  ∃ (x : ℤ), 3 * x - 2 * (-x) = 30 ∧ x * (-x) = -36 :=
by
  sorry

end solve_opposite_numbers_product_l156_156750


namespace solve_inequality_l156_156634

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (m n : ℝ) : f (m + n) = f m * f n
axiom f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1

theorem solve_inequality (x : ℝ) : f (x^2) * f (2 * x - 3) > 1 ↔ -3 < x ∧ x < 1 := sorry

end solve_inequality_l156_156634


namespace linda_total_miles_l156_156401

def calculate_total_miles (x : ℕ) : ℕ :=
  (60 / x) + (60 / (x + 4)) + (60 / (x + 8)) + (60 / (x + 12)) + (60 / (x + 16))

theorem linda_total_miles (x : ℕ) (hx1 : x > 0)
(hdx2 : 60 % x = 0)
(hdx3 : 60 % (x + 4) = 0) 
(hdx4 : 60 % (x + 8) = 0) 
(hdx5 : 60 % (x + 12) = 0) 
(hdx6 : 60 % (x + 16) = 0) :
  calculate_total_miles x = 33 := by
  sorry

end linda_total_miles_l156_156401


namespace option_B_shares_asymptotes_l156_156467

-- Define the given hyperbola equation
def given_hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

-- The asymptotes for the given hyperbola
def asymptotes_of_given_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Define the hyperbola for option B
def option_B_hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 16) = 1

-- The asymptotes for option B hyperbola
def asymptotes_of_option_B_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Theorem stating that the hyperbola in option B shares the same asymptotes as the given hyperbola
theorem option_B_shares_asymptotes :
  (∀ x y : ℝ, given_hyperbola x y → asymptotes_of_given_hyperbola x y) →
  (∀ x y : ℝ, option_B_hyperbola x y → asymptotes_of_option_B_hyperbola x y) :=
by
  intros h₁ h₂
  -- Here should be the proof to show they have the same asymptotes
  sorry

end option_B_shares_asymptotes_l156_156467


namespace real_value_of_b_l156_156882

open Real

theorem real_value_of_b : ∃ x : ℝ, (x^2 - 2 * x + 1 = 0) ∧ (x^2 + x - 2 = 0) :=
by
  sorry

end real_value_of_b_l156_156882


namespace Matias_longest_bike_ride_l156_156266

-- Define conditions in Lean
def blocks : ℕ := 4
def block_side_length : ℕ := 100
def streets : ℕ := 12

def Matias_route : Prop :=
  ∀ (intersections_used : ℕ), 
    intersections_used ≤ 4 → (streets - intersections_used/2 * 2) = 10

def correct_maximum_path_length : ℕ := 1000

-- Objective: Prove that given the conditions the longest route is 1000 meters
theorem Matias_longest_bike_ride :
  (100 * (streets - 2)) = correct_maximum_path_length :=
by
  sorry

end Matias_longest_bike_ride_l156_156266


namespace lorenzo_cans_l156_156265

theorem lorenzo_cans (c : ℕ) (tacks_per_can : ℕ) (total_tacks : ℕ) (boards_tested : ℕ) (remaining_tacks : ℕ) :
  boards_tested = 120 →
  remaining_tacks = 30 →
  total_tacks = 450 →
  tacks_per_can = (boards_tested + remaining_tacks) →
  c * tacks_per_can = total_tacks →
  c = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lorenzo_cans_l156_156265


namespace james_carrot_sticks_left_l156_156657

variable (original_carrot_sticks : ℕ)
variable (eaten_before_dinner : ℕ)
variable (eaten_after_dinner : ℕ)
variable (given_away_during_dinner : ℕ)

theorem james_carrot_sticks_left 
  (h1 : original_carrot_sticks = 50)
  (h2 : eaten_before_dinner = 22)
  (h3 : eaten_after_dinner = 15)
  (h4 : given_away_during_dinner = 8) :
  original_carrot_sticks - eaten_before_dinner - eaten_after_dinner - given_away_during_dinner = 5 := 
sorry

end james_carrot_sticks_left_l156_156657
