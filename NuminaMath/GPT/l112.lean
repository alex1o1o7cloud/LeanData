import Mathlib

namespace area_ratio_eq_l112_112408

-- Define the parameters used in the problem
variables (t t1 r ρ : ℝ)

-- Define the conditions given in the problem
def area_triangle_ABC : ℝ := t
def area_triangle_A1B1C1 : ℝ := t1
def circumradius_ABC : ℝ := r
def inradius_A1B1C1 : ℝ := ρ

-- Problem statement: Prove the given equation
theorem area_ratio_eq : t / t1 = 2 * ρ / r :=
sorry

end area_ratio_eq_l112_112408


namespace simplified_value_of_f_l112_112654

variable (x : ℝ)

noncomputable def f : ℝ := 3 * x + 5 - 4 * x^2 + 2 * x - 7 + x^2 - 3 * x + 8

theorem simplified_value_of_f : f x = -3 * x^2 + 2 * x + 6 := by
  unfold f
  sorry

end simplified_value_of_f_l112_112654


namespace athlete_heartbeats_l112_112155

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l112_112155


namespace cubic_identity_l112_112725

theorem cubic_identity (x y z : ℝ) (h1 : x + y + z = 13) (h2 : xy + xz + yz = 32) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 949 :=
by
  sorry

end cubic_identity_l112_112725


namespace trip_total_hours_l112_112792

theorem trip_total_hours
    (x : ℕ) -- additional hours of travel
    (dist_1 : ℕ := 30 * 6) -- distance for first 6 hours
    (dist_2 : ℕ := 46 * x) -- distance for additional hours
    (total_dist : ℕ := dist_1 + dist_2) -- total distance
    (total_time : ℕ := 6 + x) -- total time
    (avg_speed : ℕ := total_dist / total_time) -- average speed
    (h : avg_speed = 34) : total_time = 8 :=
by
  sorry

end trip_total_hours_l112_112792


namespace digit_for_divisibility_by_9_l112_112409

theorem digit_for_divisibility_by_9 (A : ℕ) (hA : A < 10) : 
  (∃ k : ℕ, 83 * 1000 + A * 10 + 5 = 9 * k) ↔ A = 2 :=
by
  sorry

end digit_for_divisibility_by_9_l112_112409


namespace number_of_carbon_atoms_l112_112799

/-- A proof to determine the number of carbon atoms in a compound given specific conditions
-/
theorem number_of_carbon_atoms
  (H_atoms : ℕ) (O_atoms : ℕ) (C_weight : ℕ) (H_weight : ℕ) (O_weight : ℕ) (Molecular_weight : ℕ) :
  H_atoms = 6 →
  O_atoms = 1 →
  C_weight = 12 →
  H_weight = 1 →
  O_weight = 16 →
  Molecular_weight = 58 →
  (Molecular_weight - (H_atoms * H_weight + O_atoms * O_weight)) / C_weight = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_carbon_atoms_l112_112799


namespace options_equal_results_l112_112814

theorem options_equal_results :
  (4^3 ≠ 3^4) ∧
  ((-5)^3 = (-5^3)) ∧
  ((-6)^2 ≠ -6^2) ∧
  ((- (5/2))^2 ≠ (- (2/5))^2) :=
by {
  sorry
}

end options_equal_results_l112_112814


namespace intersection_M_N_l112_112860

def M := { y : ℝ | ∃ x : ℝ, y = 2^x }
def N := { y : ℝ | ∃ x : ℝ, y = 2 * Real.sin x }

theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 2 } :=
by
  sorry

end intersection_M_N_l112_112860


namespace area_of_inner_square_l112_112632

theorem area_of_inner_square (s₁ s₂ : ℝ) (side_length_WXYZ : ℝ) (WI : ℝ) (area_IJKL : ℝ) 
  (h1 : s₁ = 10) 
  (h2 : s₂ = 10 - 2 * Real.sqrt 2)
  (h3 : side_length_WXYZ = 10)
  (h4 : WI = 2)
  (h5 : area_IJKL = (s₂)^2): 
  area_IJKL = 102 - 20 * Real.sqrt 2 :=
by
  sorry

end area_of_inner_square_l112_112632


namespace find_k_parallel_vectors_l112_112985

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_k_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (-2, 6)
  vector_parallel a b → k = -3 :=
by
  sorry

end find_k_parallel_vectors_l112_112985


namespace rectangle_length_l112_112929

-- Define a structure for the rectangle.
structure Rectangle where
  breadth : ℝ
  length : ℝ
  area : ℝ

-- Define the given conditions.
def givenConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.area = 6075

-- State the theorem.
theorem rectangle_length (r : Rectangle) (h : givenConditions r) : r.length = 135 :=
by
  sorry

end rectangle_length_l112_112929


namespace probability_of_diagonals_in_regular_hexagon_l112_112797

/-- Definitions based on conditions:
    - A regular hexagon has 6 vertices.
    - A diagonal is a line between any two non-adjacent vertices.
    - Compute the number of diagonals and apply combinatorial mathematics to compute the probability.--/
noncomputable def probability_of_diagonals_intersecting_inside_hexagon : ℚ :=
let vertices := 6 in
let sides := vertices in
let total_pairs := (vertices * (vertices - 1)) / 2 in
let diagonals := total_pairs - sides in
let diagonal_pairs := (diagonals * (diagonals - 1)) / 2 in
let intersecting_diagonals := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / (4 * 3 * 2 * 1)) in
(intersecting_diagonals : ℚ) / diagonal_pairs

theorem probability_of_diagonals_in_regular_hexagon :
  probability_of_diagonals_intersecting_inside_hexagon = 5 / 12 := by 
  sorry

end probability_of_diagonals_in_regular_hexagon_l112_112797


namespace basic_astrophysics_degrees_l112_112935

def percentages : List ℚ := [12, 22, 14, 27, 7, 5, 3, 4]

def total_budget_percentage : ℚ := 100

def degrees_in_circle : ℚ := 360

def remaining_percentage (lst : List ℚ) (total : ℚ) : ℚ :=
  total - lst.sum / 100  -- convert sum to percentage

def degrees_of_percentage (percent : ℚ) (circle_degrees : ℚ) : ℚ :=
  percent * (circle_degrees / total_budget_percentage) -- conversion rate per percentage point

theorem basic_astrophysics_degrees :
  degrees_of_percentage (remaining_percentage percentages total_budget_percentage) degrees_in_circle = 21.6 :=
by
  sorry

end basic_astrophysics_degrees_l112_112935


namespace remainder_abc_mod_5_l112_112722

theorem remainder_abc_mod_5
  (a b c : ℕ)
  (h₀ : a < 5)
  (h₁ : b < 5)
  (h₂ : c < 5)
  (h₃ : (a + 2 * b + 3 * c) % 5 = 0)
  (h₄ : (2 * a + 3 * b + c) % 5 = 2)
  (h₅ : (3 * a + b + 2 * c) % 5 = 3) :
  (a * b * c) % 5 = 3 :=
by
  sorry

end remainder_abc_mod_5_l112_112722


namespace athlete_heartbeats_during_race_l112_112149

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l112_112149


namespace sphere_weight_dependence_l112_112776

theorem sphere_weight_dependence 
  (r1 r2 SA1 SA2 weight1 weight2 : ℝ) 
  (h1 : r1 = 0.15) 
  (h2 : r2 = 2 * r1) 
  (h3 : SA1 = 4 * Real.pi * r1^2) 
  (h4 : SA2 = 4 * Real.pi * r2^2) 
  (h5 : weight1 = 8) 
  (h6 : weight1 / SA1 = weight2 / SA2) : 
  weight2 = 32 :=
by
  sorry

end sphere_weight_dependence_l112_112776


namespace elizabeth_stickers_l112_112178

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l112_112178


namespace part1_part2_part3_part3_expectation_l112_112087

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end part1_part2_part3_part3_expectation_l112_112087


namespace remainder_sum_products_l112_112770

theorem remainder_sum_products (a b c d : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) 
  (hd : d % 7 = 6) : 
  ((a * b + c * d) % 7) = 1 :=
by sorry

end remainder_sum_products_l112_112770


namespace box_volume_l112_112095

theorem box_volume (L W H : ℝ) (h1 : L * W = 120) (h2 : W * H = 72) (h3 : L * H = 60) : L * W * H = 720 := 
by sorry

end box_volume_l112_112095


namespace division_remainder_l112_112988

theorem division_remainder :
  ∃ (R D Q : ℕ), D = 3 * Q ∧ D = 3 * R + 3 ∧ 251 = D * Q + R ∧ R = 8 := by
  sorry

end division_remainder_l112_112988


namespace largest_divisible_number_l112_112296

theorem largest_divisible_number : ∃ n, n = 9950 ∧ n ≤ 9999 ∧ (∀ m, m ≤ 9999 ∧ m % 50 = 0 → m ≤ n) :=
by {
  sorry
}

end largest_divisible_number_l112_112296


namespace adams_father_total_amount_l112_112557

noncomputable def annual_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

noncomputable def total_interest (annual_interest : ℝ) (years : ℝ) : ℝ :=
  annual_interest * years

noncomputable def total_amount (principal : ℝ) (total_interest : ℝ) : ℝ :=
  principal + total_interest

theorem adams_father_total_amount :
  let principal := 2000
  let rate := 0.08
  let years := 2.5
  let annualInterest := annual_interest principal rate
  let interest := total_interest annualInterest years
  let amount := total_amount principal interest
  amount = 2400 :=
by sorry

end adams_father_total_amount_l112_112557


namespace max_value_even_function_1_2_l112_112476

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Given conditions
variables (f : ℝ → ℝ)
variable (h1 : even_function f)
variable (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → f x ≤ -2)

-- Prove the maximum value on [1, 2] is -2
theorem max_value_even_function_1_2 : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ -2) :=
sorry

end max_value_even_function_1_2_l112_112476


namespace triangle_rational_segments_l112_112796

theorem triangle_rational_segments (a b c : ℚ) (h : a + b > c ∧ a + c > b ∧ b + c > a):
  ∃ (ab1 cb1 : ℚ), (ab1 + cb1 = b) := sorry

end triangle_rational_segments_l112_112796


namespace x_value_unique_l112_112012

theorem x_value_unique (x : ℝ) (h : ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) :
  x = 3 / 2 :=
sorry

end x_value_unique_l112_112012


namespace problem_statement_l112_112497

theorem problem_statement
  (f : ℝ → ℝ)
  (h0 : ∀ x, 0 <= x → x <= 1 → 0 <= f x)
  (h1 : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
        (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) :
  ∀ (u v w : ℝ), 
    0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1 → 
    (w - v) / (w - u) * f u + (v - u) / (w - u) * f w ≤ f v + 2 :=
by
  intros u v w h
  sorry

end problem_statement_l112_112497


namespace books_choice_l112_112555

theorem books_choice (P S : Type) [Fintype P] [Fintype S] [DecidableEq P] [DecidableEq S] 
  (hP : Fintype.card P = 4) (hS : Fintype.card S = 2) 
  : ∃ (books : Finset (P ⊕ S)), books.card = 4 ∧ ∃ (s : S), s ∈ books := 
by
  have h1 : ∃ (books : Finset (P ⊕ S)), books.card = 4 := sorry
  have h2 : ∀ (books : Finset (P ⊕ S)), books.card = 4 → ∃ (s : S), s ∈ books := sorry
  exact ⟨_, h1, h2⟩

end books_choice_l112_112555


namespace roots_ellipse_condition_l112_112773

theorem roots_ellipse_condition (m n : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1^2 - m*x1 + n = 0 ∧ x2^2 - m*x2 + n = 0) 
  ↔ (m > 0 ∧ n > 0 ∧ m ≠ n) :=
sorry

end roots_ellipse_condition_l112_112773


namespace sqrt_four_eq_two_or_neg_two_l112_112274

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l112_112274


namespace chess_tournament_ratio_l112_112365

theorem chess_tournament_ratio:
  ∃ n : ℕ, (n * (n - 1)) / 2 = 231 ∧ (n - 1) = 21 := 
sorry

end chess_tournament_ratio_l112_112365


namespace math_problem_proof_l112_112656

theorem math_problem_proof (n : ℕ) 
  (h1 : n / 37 = 2) 
  (h2 : n % 37 = 26) :
  48 - n / 4 = 23 := by
  sorry

end math_problem_proof_l112_112656


namespace find_a6_l112_112042

-- Define an arithmetic progression.
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the necessary conditions given in the problem.
def conditions (a d : ℕ) : Prop :=
  (arithmetic_progression a d 1 + arithmetic_progression a d 2 + arithmetic_progression a d 3 = 168) ∧
  (arithmetic_progression a d 2 - arithmetic_progression a d 5 = 42)

-- State the theorem with the final value assertion.
theorem find_a6 (a d : ℕ) (h : conditions a (-14)) : 
  arithmetic_progression a (-14) 6 = 3 := 
sorry

end find_a6_l112_112042


namespace greatest_value_of_a_plus_b_l112_112472

-- Definition of the problem conditions
def is_pos_int (n : ℕ) := n > 0

-- Lean statement to prove the greatest possible value of a + b
theorem greatest_value_of_a_plus_b :
  ∃ a b : ℕ, is_pos_int a ∧ is_pos_int b ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 9) ∧ a + b = 100 :=
sorry  -- Proof omitted

end greatest_value_of_a_plus_b_l112_112472


namespace ratio_problem_l112_112598

-- Define the conditions and the required proof
theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 :=
by
  sorry

end ratio_problem_l112_112598


namespace counterexample_proof_l112_112680

theorem counterexample_proof :
  ∃ a : ℝ, |a - 1| > 1 ∧ ¬ (a > 2) :=
  sorry

end counterexample_proof_l112_112680


namespace distributor_income_proof_l112_112227

noncomputable def income_2017 (a k x : ℝ) : ℝ :=
  (a + k / (x - 7)) * (x - 5)

theorem distributor_income_proof (a : ℝ) (x : ℝ) (h_range : 10 ≤ x ∧ x ≤ 14) (h_k : k = 3 * a):
  income_2017 a (3 * a) x = 12 * a ↔ x = 13 := by
  sorry

end distributor_income_proof_l112_112227


namespace div_ad_bc_l112_112886

theorem div_ad_bc (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) : (a - c) ∣ (a * d + b * c) :=
sorry

end div_ad_bc_l112_112886


namespace percentage_of_only_cat_owners_l112_112868

theorem percentage_of_only_cat_owners (total_students total_dog_owners total_cat_owners both_cat_dog_owners : ℕ) 
(h_total_students : total_students = 500)
(h_total_dog_owners : total_dog_owners = 120)
(h_total_cat_owners : total_cat_owners = 80)
(h_both_cat_dog_owners : both_cat_dog_owners = 40) :
( (total_cat_owners - both_cat_dog_owners : ℕ) * 100 / total_students ) = 8 := 
by
  sorry

end percentage_of_only_cat_owners_l112_112868


namespace campers_afternoon_l112_112931

def morning_campers : ℕ := 52
def additional_campers : ℕ := 9
def total_campers_afternoon : ℕ := morning_campers + additional_campers

theorem campers_afternoon : total_campers_afternoon = 61 :=
by
  sorry

end campers_afternoon_l112_112931


namespace arc_length_of_sector_l112_112587

theorem arc_length_of_sector (r α : ℝ) (hα : α = Real.pi / 5) (hr : r = 20) : r * α = 4 * Real.pi :=
by
  sorry

end arc_length_of_sector_l112_112587


namespace slip_2_5_in_A_or_C_l112_112057

-- Define the slips and their values
def slips : List ℚ := [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 6]

-- Define the cups
inductive Cup
| A | B | C | D | E | F

open Cup

-- Define the given cups constraints
def sum_constraints : Cup → ℚ
| A => 6
| B => 7
| C => 8
| D => 9
| E => 10
| F => 10

-- Initial conditions for slips placement
def slips_in_cups (c : Cup) : List ℚ :=
match c with
| F => [1.5]
| B => [4]
| _ => []

-- We'd like to prove that:
def slip_2_5_can_go_into : Prop :=
  (slips_in_cups A = [2.5] ∧ slips_in_cups C = [2.5])

theorem slip_2_5_in_A_or_C : slip_2_5_can_go_into :=
sorry

end slip_2_5_in_A_or_C_l112_112057


namespace buratino_correct_l112_112879

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_nine_digit_number (n : ℕ) : Prop :=
  n >= 10^8 ∧ n < 10^9 ∧ (∀ i j : ℕ, i < 9 ∧ j < 9 ∧ i ≠ j → ((n / 10^i) % 10 ≠ (n / 10^j) % 10)) ∧
  (∀ i : ℕ, i < 9 → (n / 10^i) % 10 ≠ 7)

def can_form_prime (n : ℕ) : Prop :=
  ∃ m : ℕ, valid_nine_digit_number n ∧ (m < 1000 ∧ is_prime m ∧
   (∃ erase_indices : List ℕ, erase_indices.length = 6 ∧ 
    ∀ i : ℕ, i ∈ erase_indices → i < 9 ∧ 
    (n % 10^(9 - i)) / 10^(3 - i) = m))

theorem buratino_correct : 
  ∀ n : ℕ, valid_nine_digit_number n → ¬ can_form_prime n :=
by
  sorry

end buratino_correct_l112_112879


namespace total_cost_is_correct_l112_112215

def gravel_cost_per_cubic_foot : ℝ := 8
def discount_rate : ℝ := 0.10
def volume_in_cubic_yards : ℝ := 8
def conversion_factor : ℝ := 27

-- The initial cost for the given volume of gravel in cubic feet
noncomputable def initial_cost : ℝ := gravel_cost_per_cubic_foot * (volume_in_cubic_yards * conversion_factor)

-- The discount amount
noncomputable def discount_amount : ℝ := initial_cost * discount_rate

-- Total cost after applying discount
noncomputable def total_cost_after_discount : ℝ := initial_cost - discount_amount

theorem total_cost_is_correct : total_cost_after_discount = 1555.20 :=
sorry

end total_cost_is_correct_l112_112215


namespace factor_expression_l112_112457

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) :=
by
  sorry

end factor_expression_l112_112457


namespace initial_pencils_l112_112779

theorem initial_pencils (pencils_added initial_pencils total_pencils : ℕ) 
  (h1 : pencils_added = 3) 
  (h2 : total_pencils = 5) :
  initial_pencils = total_pencils - pencils_added := 
by 
  sorry

end initial_pencils_l112_112779


namespace hannah_dogs_food_total_l112_112471

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end hannah_dogs_food_total_l112_112471


namespace simplify_expression_l112_112543

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
  ( ((x+1)^2 * (x^2 - x + 1)^2 / (x^3 + 1)^2)^2 *
    ((x-1)^2 * (x^2 + x + 1)^2 / (x^3 - 1)^2)^2
  ) = 1 :=
by
  sorry

end simplify_expression_l112_112543


namespace cos_half_angle_inequality_1_cos_half_angle_inequality_2_l112_112740

open Real

variable {A B C : ℝ} (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA_sum : A + B + C = π)

theorem cos_half_angle_inequality_1 :
  cos (A / 2) < cos (B / 2) + cos (C / 2) :=
by sorry

theorem cos_half_angle_inequality_2 :
  cos (A / 2) < sin (B / 2) + sin (C / 2) :=
by sorry

end cos_half_angle_inequality_1_cos_half_angle_inequality_2_l112_112740


namespace combined_sleep_time_l112_112937

variables (cougar_night_sleep zebra_night_sleep total_sleep_cougar total_sleep_zebra total_weekly_sleep : ℕ)

theorem combined_sleep_time :
  (cougar_night_sleep = 4) →
  (zebra_night_sleep = cougar_night_sleep + 2) →
  (total_sleep_cougar = cougar_night_sleep * 7) →
  (total_sleep_zebra = zebra_night_sleep * 7) →
  (total_weekly_sleep = total_sleep_cougar + total_sleep_zebra) →
  total_weekly_sleep = 70 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end combined_sleep_time_l112_112937


namespace train_speed_54_kmh_l112_112308

theorem train_speed_54_kmh
  (train_length : ℕ)
  (tunnel_length : ℕ)
  (time_seconds : ℕ)
  (total_distance : ℕ := train_length + tunnel_length)
  (speed_mps : ℚ := total_distance / time_seconds)
  (conversion_factor : ℚ := 3.6) :
  train_length = 300 →
  tunnel_length = 1200 →
  time_seconds = 100 →
  speed_mps * conversion_factor = 54 := 
by
  intros h_train_length h_tunnel_length h_time_seconds
  sorry

end train_speed_54_kmh_l112_112308


namespace students_in_first_bus_l112_112524

theorem students_in_first_bus (total_buses : ℕ) (avg_students_per_bus : ℕ) 
(avg_remaining_students : ℕ) (num_remaining_buses : ℕ) 
(h1 : total_buses = 6) 
(h2 : avg_students_per_bus = 28) 
(h3 : avg_remaining_students = 26) 
(h4 : num_remaining_buses = 5) :
  (total_buses * avg_students_per_bus - num_remaining_buses * avg_remaining_students = 38) :=
by
  sorry

end students_in_first_bus_l112_112524


namespace total_heartbeats_correct_l112_112157

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l112_112157


namespace greatest_fraction_l112_112198

theorem greatest_fraction 
  (w x y z : ℕ)
  (hw : w > 0)
  (h_ordering : w < x ∧ x < y ∧ y < z) :
  (x + y + z) / (w + x + y) > (w + x + y) / (x + y + z) ∧
  (x + y + z) / (w + x + y) > (w + y + z) / (x + w + z) ∧
  (x + y + z) / (w + x + y) > (x + w + z) / (w + y + z) ∧
  (x + y + z) / (w + x + y) > (y + z + w) / (x + y + z) :=
sorry

end greatest_fraction_l112_112198


namespace convex_quad_no_triangle_l112_112620

/-- Given four angles of a convex quadrilateral, it is not always possible to choose any 
three of these angles so that they represent the lengths of the sides of some triangle. -/
theorem convex_quad_no_triangle (α β γ δ : ℝ) 
  (h_sum : α + β + γ + δ = 360) :
  ¬(∀ a b c : ℝ, a + b + c = 360 → (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end convex_quad_no_triangle_l112_112620


namespace bike_cost_l112_112113

theorem bike_cost (days_in_two_weeks : ℕ) 
  (bracelets_per_day : ℕ)
  (price_per_bracelet : ℕ)
  (total_bracelets : ℕ)
  (total_money : ℕ) 
  (h1 : days_in_two_weeks = 2 * 7)
  (h2 : bracelets_per_day = 8)
  (h3 : price_per_bracelet = 1)
  (h4 : total_bracelets = days_in_two_weeks * bracelets_per_day)
  (h5 : total_money = total_bracelets * price_per_bracelet) :
  total_money = 112 :=
sorry

end bike_cost_l112_112113


namespace middle_schoolers_count_l112_112735

theorem middle_schoolers_count (total_students girls_ratio primary_girls_ratio primary_boys_ratio : ℚ)
    (total_students_eq : total_students = 800)
    (girls_ratio_eq : girls_ratio = 5 / 8)
    (primary_girls_ratio_eq: primary_girls_ratio = 7 / 10)
    (primary_boys_ratio_eq: primary_boys_ratio = 2 / 5) :
    let girls := total_students * girls_ratio
        boys := total_students - girls
        primary_girls := girls * primary_girls_ratio
        middle_school_girls := girls - primary_girls
        primary_boys := boys * primary_boys_ratio
        middle_school_boys := boys - primary_boys
     in middle_school_girls + middle_school_boys = 330 :=
by 
  intros
  sorry

end middle_schoolers_count_l112_112735


namespace neg_P_l112_112377

def P := ∃ x : ℝ, (0 < x) ∧ (3^x < x^3)

theorem neg_P : ¬P ↔ ∀ x : ℝ, (0 < x) → (3^x ≥ x^3) :=
by
  sorry

end neg_P_l112_112377


namespace camille_total_birds_l112_112006

theorem camille_total_birds :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 :=
by
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  show cardinals + robins + blue_jays + sparrows + pigeons + finches = 55
  sorry

end camille_total_birds_l112_112006


namespace rhombus_perimeter_l112_112638

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 52 :=
by
  sorry

end rhombus_perimeter_l112_112638


namespace find_initial_books_l112_112626

/-- The number of books the class initially obtained from the library --/
def initial_books : ℕ := sorry

/-- The number of books added later --/
def books_added_later : ℕ := 23

/-- The total number of books the class has --/
def total_books : ℕ := 77

theorem find_initial_books : initial_books + books_added_later = total_books → initial_books = 54 :=
by
  intros h
  sorry

end find_initial_books_l112_112626


namespace range_of_a_l112_112355

noncomputable def f (a x : ℝ) := a / x - 1 + Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ f a x ≤ 0) → a ≤ 1 := 
sorry

end range_of_a_l112_112355


namespace total_heartbeats_during_race_l112_112161

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l112_112161


namespace equal_piece_length_l112_112133

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l112_112133


namespace tokens_per_pitch_l112_112077

theorem tokens_per_pitch 
  (tokens_macy : ℕ) (tokens_piper : ℕ)
  (hits_macy : ℕ) (hits_piper : ℕ)
  (misses_total : ℕ) (p : ℕ)
  (h1 : tokens_macy = 11)
  (h2 : tokens_piper = 17)
  (h3 : hits_macy = 50)
  (h4 : hits_piper = 55)
  (h5 : misses_total = 315)
  (h6 : 28 * p = hits_macy + hits_piper + misses_total) :
  p = 15 := 
by 
  sorry

end tokens_per_pitch_l112_112077


namespace equation_zero_solution_l112_112728

-- Define the conditions and the answer
def equation_zero (x : ℝ) : Prop := (x^2 + x - 2) / (x - 1) = 0
def non_zero_denominator (x : ℝ) : Prop := x - 1 ≠ 0
def solution_x (x : ℝ) : Prop := x = -2

-- The main theorem
theorem equation_zero_solution (x : ℝ) (h1 : equation_zero x) (h2 : non_zero_denominator x) : solution_x x := 
sorry

end equation_zero_solution_l112_112728


namespace part1_range_a_part2_range_a_l112_112346

-- Definitions of the propositions
def p (a : ℝ) := ∃ x : ℝ, x^2 + a * x + 2 = 0

def q (a : ℝ) := ∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - a < 0

-- Part 1: If p is true, find the range of values for a
theorem part1_range_a (a : ℝ) :
  p a → (a ≤ -2*Real.sqrt 2 ∨ a ≥ 2*Real.sqrt 2) := sorry

-- Part 2: If one of p or q is true and the other is false, find the range of values for a
theorem part2_range_a (a : ℝ) :
  (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a) →
  (a ≤ -2*Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2*Real.sqrt 2)) := sorry

end part1_range_a_part2_range_a_l112_112346


namespace man_l112_112549

theorem man's_speed_with_stream
  (V_m V_s : ℝ)
  (h1 : V_m = 6)
  (h2 : V_m - V_s = 4) :
  V_m + V_s = 8 :=
sorry

end man_l112_112549


namespace garden_area_enlargement_l112_112552

theorem garden_area_enlargement :
  let length := 60
  let width := 20
  (2 * (length + width)) = 160 →
  (160 / 4) = 40 →
  ((40 * 40) - (length * width) = 400) :=
begin
  intros,
  sorry,
end

end garden_area_enlargement_l112_112552


namespace which_is_system_lin_eq_l112_112922

def option_A : Prop := ∀ (x : ℝ), x - 1 = 2 * x
def option_B : Prop := ∀ (x y : ℝ), x - 1/y = 1
def option_C : Prop := ∀ (x z : ℝ), x + z = 3
def option_D : Prop := ∀ (x y z : ℝ), x - y + z = 1

theorem which_is_system_lin_eq (hA : option_A) (hB : option_B) (hC : option_C) (hD : option_D) :
    (∀ (x z : ℝ), x + z = 3) :=
by
  sorry

end which_is_system_lin_eq_l112_112922


namespace length_more_than_breadth_by_200_percent_l112_112393

noncomputable def length: ℝ := 19.595917942265423
noncomputable def total_cost: ℝ := 640
noncomputable def rate_per_sq_meter: ℝ := 5

theorem length_more_than_breadth_by_200_percent
  (area : ℝ := total_cost / rate_per_sq_meter)
  (breadth : ℝ := area / length) :
  ((length - breadth) / breadth) * 100 = 200 := by
  have h1 : area = 128 := by sorry
  have h2 : breadth = 128 / 19.595917942265423 := by sorry
  rw [h1, h2]
  sorry

end length_more_than_breadth_by_200_percent_l112_112393


namespace mary_income_percent_of_juan_l112_112623

variable (J : ℝ)
variable (T : ℝ)
variable (M : ℝ)

-- Conditions
def tim_income := T = 0.60 * J
def mary_income := M = 1.40 * T

-- Theorem to prove that Mary's income is 84 percent of Juan's income
theorem mary_income_percent_of_juan : tim_income J T → mary_income T M → M = 0.84 * J :=
by
  sorry

end mary_income_percent_of_juan_l112_112623


namespace orchestra_musicians_l112_112098

theorem orchestra_musicians : ∃ (m n : ℕ), (m = n^2 + 11) ∧ (m = n * (n + 5)) ∧ m = 36 :=
by {
  sorry
}

end orchestra_musicians_l112_112098


namespace continuous_stripe_probability_l112_112571

-- Define a structure representing the configuration of each face.
structure FaceConfiguration where
  is_diagonal : Bool
  edge_pair_or_vertex_pair : Bool

-- Define the cube configuration.
structure CubeConfiguration where
  face1 : FaceConfiguration
  face2 : FaceConfiguration
  face3 : FaceConfiguration
  face4 : FaceConfiguration
  face5 : FaceConfiguration
  face6 : FaceConfiguration

noncomputable def total_configurations : ℕ := 4^6

-- Define the function that checks if a configuration results in a continuous stripe.
def results_in_continuous_stripe (c : CubeConfiguration) : Bool := sorry

-- Define the number of configurations resulting in a continuous stripe.
noncomputable def configurations_with_continuous_stripe : ℕ :=
  Nat.card {c : CubeConfiguration // results_in_continuous_stripe c}

-- Define the probability calculation.
noncomputable def probability_continuous_stripe : ℚ :=
  configurations_with_continuous_stripe / total_configurations

-- The statement of the problem: Prove the probability of continuous stripe is 3/256.
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 256 :=
sorry

end continuous_stripe_probability_l112_112571


namespace minimum_value_l112_112487

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end minimum_value_l112_112487


namespace max_digit_d_l112_112019

theorem max_digit_d (d f : ℕ) (h₁ : d ≤ 9) (h₂ : f ≤ 9) (h₃ : (18 + d + f) % 3 = 0) (h₄ : (12 - (d + f)) % 11 = 0) : d = 1 :=
sorry

end max_digit_d_l112_112019


namespace recipe_flour_amount_l112_112504

theorem recipe_flour_amount
  (cups_of_sugar : ℕ) (cups_of_salt : ℕ) (cups_of_flour_added : ℕ)
  (additional_cups_of_flour : ℕ)
  (h1 : cups_of_sugar = 2)
  (h2 : cups_of_salt = 80)
  (h3 : cups_of_flour_added = 7)
  (h4 : additional_cups_of_flour = cups_of_sugar + 1) :
  cups_of_flour_added + additional_cups_of_flour = 10 :=
by {
  sorry
}

end recipe_flour_amount_l112_112504


namespace problem_part1_problem_part2_problem_part3_l112_112384

variable (a b x : ℝ) (p q : ℝ) (n x1 x2 : ℝ)
variable (h1 : x1 = -2) (h2 : x2 = 3)
variable (h3 : x1 < x2)

def equation1 := x + p / x = q
def solution1_p := p = -6
def solution1_q := q = 1

def equation2 := x + 7 / x = 8
def solution2 := x1 = 7

def equation3 := 2 * x + (n^2 - n) / (2 * x - 1) = 2 * n
def solution3 := (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)

theorem problem_part1 : ∀ (x : ℝ), (x + -6 / x = 1) → (p = -6 ∧ q = 1) := by
  sorry

theorem problem_part2 : (max 7 1 = 7) := by
  sorry

theorem problem_part3 : ∀ (n : ℝ), (∃ x1 x2, x1 < x2 ∧ (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)) := by
  sorry

end problem_part1_problem_part2_problem_part3_l112_112384


namespace value_of_x_l112_112984

variable {x y z : ℤ}

theorem value_of_x
  (h1 : x + y = 31)
  (h2 : y + z = 47)
  (h3 : x + z = 52)
  (h4 : y + z = x + 16) :
  x = 31 := by
  sorry

end value_of_x_l112_112984


namespace fraction_of_speedsters_l112_112301

/-- Let S denote the total number of Speedsters and T denote the total inventory. 
    Given the following conditions:
    1. 54 Speedster convertibles constitute 3/5 of all Speedsters (S).
    2. There are 30 vehicles that are not Speedsters.

    Prove that the fraction of the current inventory that is Speedsters is 3/4.
-/
theorem fraction_of_speedsters (S T : ℕ)
  (h1 : 3 / 5 * S = 54)
  (h2 : T = S + 30) :
  (S : ℚ) / T = 3 / 4 :=
by
  sorry

end fraction_of_speedsters_l112_112301


namespace water_bottles_needed_l112_112957

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l112_112957


namespace second_term_is_neg_12_l112_112644

-- Define the problem conditions
variables {a d : ℤ}
axiom tenth_term : a + 9 * d = 20
axiom eleventh_term : a + 10 * d = 24

-- Define the second term calculation
def second_term (a d : ℤ) := a + d

-- The problem statement: Prove that the second term is -12 given the conditions
theorem second_term_is_neg_12 : second_term a d = -12 :=
by sorry

end second_term_is_neg_12_l112_112644


namespace total_water_bottles_needed_l112_112955

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l112_112955


namespace curve_representation_l112_112765

def curve_set (x y : Real) : Prop := 
  ((x + y - 1) * Real.sqrt (x^2 + y^2 - 4) = 0)

def line_set (x y : Real) : Prop :=
  (x + y - 1 = 0) ∧ (x^2 + y^2 ≥ 4)

def circle_set (x y : Real) : Prop :=
  (x^2 + y^2 = 4)

theorem curve_representation (x y : Real) :
  curve_set x y ↔ (line_set x y ∨ circle_set x y) :=
sorry

end curve_representation_l112_112765


namespace total_end_of_year_students_l112_112559

theorem total_end_of_year_students :
  let start_fourth := 33
  let start_fifth := 45
  let start_sixth := 28
  let left_fourth := 18
  let joined_fourth := 14
  let left_fifth := 12
  let joined_fifth := 20
  let left_sixth := 10
  let joined_sixth := 16

  let end_fourth := start_fourth - left_fourth + joined_fourth
  let end_fifth := start_fifth - left_fifth + joined_fifth
  let end_sixth := start_sixth - left_sixth + joined_sixth
  
  end_fourth + end_fifth + end_sixth = 116 := by
    sorry

end total_end_of_year_students_l112_112559


namespace simplify_expression_l112_112950

theorem simplify_expression :
  (-2) ^ 2006 + (-1) ^ 3007 + 1 ^ 3010 - (-2) ^ 2007 = -2 ^ 2006 := 
sorry

end simplify_expression_l112_112950


namespace find_k_for_parallel_vectors_l112_112469

theorem find_k_for_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (9, k - 6)
  (1 * (k - 6) - 9 * k = 0) → k = -3 / 4 :=
by
  intros a b parallel_cond
  sorry

end find_k_for_parallel_vectors_l112_112469


namespace mechanical_pencils_and_pens_price_l112_112925

theorem mechanical_pencils_and_pens_price
    (x y : ℝ)
    (h₁ : 7 * x + 6 * y = 46.8)
    (h₂ : 3 * x + 5 * y = 32.2) :
  x = 2.4 ∧ y = 5 :=
sorry

end mechanical_pencils_and_pens_price_l112_112925


namespace square_chord_length_eq_l112_112736

def radius1 := 10
def radius2 := 7
def centers_distance := 15
def chord_length (x : ℝ) := 2 * x

theorem square_chord_length_eq :
    ∀ (x : ℝ), chord_length x = 15 →
    (10 + x)^2 - 200 * (Real.sqrt ((1 + 19.0 / 35.0) / 2)) = 200 - 200 * Real.sqrt (27.0 / 35.0) :=
sorry

end square_chord_length_eq_l112_112736


namespace zero_in_set_zero_l112_112923

-- Define that 0 is an element
def zero_element : Prop := true

-- Define that {0} is a set containing only the element 0
def set_zero : Set ℕ := {0}

-- The main theorem that proves 0 ∈ {0}
theorem zero_in_set_zero (h : zero_element) : 0 ∈ set_zero := 
by sorry

end zero_in_set_zero_l112_112923


namespace circle_sine_intersection_l112_112820

theorem circle_sine_intersection (h k r : ℝ) (hr : r > 0) :
  ∃ (n : ℕ), n > 16 ∧
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, (x - h)^2 + (2 * Real.sin x - k)^2 = r^2) ∧ xs.card = n :=
by
  sorry

end circle_sine_intersection_l112_112820


namespace minimum_value_of_f_l112_112701

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.cos x)^2) + (1 / (Real.sin x)^2)

theorem minimum_value_of_f : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ y = 4 :=
by
  sorry

end minimum_value_of_f_l112_112701


namespace population_reaches_210_l112_112586

noncomputable def population_function (x : ℕ) : ℝ :=
  200 * (1 + 0.01)^x

theorem population_reaches_210 :
  ∃ x : ℕ, population_function x >= 210 :=
by
  existsi 5
  apply le_of_lt
  sorry

end population_reaches_210_l112_112586


namespace num_of_nickels_is_two_l112_112142

theorem num_of_nickels_is_two (d n : ℕ) 
    (h1 : 10 * d + 5 * n = 70) 
    (h2 : d + n = 8) : 
    n = 2 := 
by 
    sorry

end num_of_nickels_is_two_l112_112142


namespace log_ratio_l112_112003

theorem log_ratio : (Real.logb 2 16) / (Real.logb 2 4) = 2 := sorry

end log_ratio_l112_112003


namespace earthquake_relief_team_selection_l112_112258

theorem earthquake_relief_team_selection : 
    ∃ (ways : ℕ), ways = 590 ∧ 
      ∃ (orthopedic neurosurgeon internist : ℕ), 
      orthopedic + neurosurgeon + internist = 5 ∧ 
      1 ≤ orthopedic ∧ 1 ≤ neurosurgeon ∧ 1 ≤ internist ∧
      orthopedic ≤ 3 ∧ neurosurgeon ≤ 4 ∧ internist ≤ 5 := 
  sorry

end earthquake_relief_team_selection_l112_112258


namespace part1_part2_l112_112206

theorem part1 (a : ℝ) (h : 48 * a^2 = 75) (ha : a > 0) : a = 5 / 4 :=
sorry

theorem part2 (θ : ℝ) 
  (h₁ : 10 * (Real.sin θ) ^ 2 = 5) 
  (h₀ : 0 < θ ∧ θ < Real.pi / 2) 
  : θ = Real.pi / 4 :=
sorry

end part1_part2_l112_112206


namespace number_of_chocolate_bars_l112_112681

theorem number_of_chocolate_bars (C : ℕ) (h1 : 50 * C = 250) : C = 5 := by
  sorry

end number_of_chocolate_bars_l112_112681


namespace equal_pieces_length_l112_112136

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l112_112136


namespace sum_of_roots_l112_112532

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots :
  (quadratic_eq 1 (-6) 9) x → (quadratic_eq 1 (-6) 9) y → x ≠ y → x + y = 6 :=
by
  sorry

end sum_of_roots_l112_112532


namespace fg_at_2_l112_112474

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := 2*x + 5

theorem fg_at_2 : f (g 2) = 729 := by
  sorry

end fg_at_2_l112_112474


namespace find_number_l112_112120

theorem find_number (n : ℕ) (h : n + 19 = 47) : n = 28 :=
by {
    sorry
}

end find_number_l112_112120


namespace village_current_population_l112_112484

theorem village_current_population (initial_population : ℕ) (ten_percent_die : ℕ)
  (twenty_percent_leave : ℕ) : 
  initial_population = 4399 →
  ten_percent_die = initial_population / 10 →
  twenty_percent_leave = (initial_population - ten_percent_die) / 5 →
  (initial_population - ten_percent_die) - twenty_percent_leave = 3167 :=
sorry

end village_current_population_l112_112484


namespace linear_function_positive_in_interval_abc_sum_greater_negative_one_l112_112297

-- Problem 1
theorem linear_function_positive_in_interval (f : ℝ → ℝ) (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n)
  (hf_m : f m > 0) (hf_n : f n > 0) : (∀ x : ℝ, m < x ∧ x < n → f x > 0) :=
sorry

-- Problem 2
theorem abc_sum_greater_negative_one (a b c : ℝ)
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : a * b + b * c + c * a > -1 :=
sorry

end linear_function_positive_in_interval_abc_sum_greater_negative_one_l112_112297


namespace find_number_l112_112934

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l112_112934


namespace smallest_n_exists_l112_112871

def connected (a b : ℕ) : Prop := -- define connection based on a picture not specified here, placeholder
sorry

def not_connected (a b : ℕ) : Prop := ¬ connected a b

def coprime (a n : ℕ) : Prop := ∀ k : ℕ, k > 1 → k ∣ a → ¬ k ∣ n

def common_divisor_greater_than_one (a n : ℕ) : Prop := ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ n

theorem smallest_n_exists :
  ∃ n : ℕ,
  (n = 35) ∧
  ∀ (numbers : Fin 7 → ℕ),
  (∀ i j, not_connected (numbers i) (numbers j) → coprime (numbers i + numbers j) n) ∧
  (∀ i j, connected (numbers i) (numbers j) → common_divisor_greater_than_one (numbers i + numbers j) n) := 
sorry

end smallest_n_exists_l112_112871


namespace convert_to_scientific_notation_l112_112604

theorem convert_to_scientific_notation (H : 1 = 10^9) : 
  3600 * (10 : ℝ)^9 = 3.6 * (10 : ℝ)^12 :=
by
  sorry

end convert_to_scientific_notation_l112_112604


namespace televisions_bought_l112_112312

theorem televisions_bought (T : ℕ)
  (television_cost : ℕ := 50)
  (figurine_cost : ℕ := 1)
  (num_figurines : ℕ := 10)
  (total_spent : ℕ := 260) :
  television_cost * T + figurine_cost * num_figurines = total_spent → T = 5 :=
by
  intros h
  sorry

end televisions_bought_l112_112312


namespace tax_liability_difference_l112_112678

theorem tax_liability_difference : 
  let annual_income := 150000
  let old_tax_rate := 0.45
  let new_tax_rate_1 := 0.30
  let new_tax_rate_2 := 0.35
  let new_tax_rate_3 := 0.40
  let mortgage_interest := 10000
  let old_tax_liability := annual_income * old_tax_rate
  let taxable_income_new := annual_income - mortgage_interest
  let new_tax_liability := 
    if taxable_income_new <= 50000 then 
      taxable_income_new * new_tax_rate_1
    else if taxable_income_new <= 100000 then 
      50000 * new_tax_rate_1 + (taxable_income_new - 50000) * new_tax_rate_2
    else 
      50000 * new_tax_rate_1 + 50000 * new_tax_rate_2 + (taxable_income_new - 100000) * new_tax_rate_3
  let tax_liability_difference := old_tax_liability - new_tax_liability
  tax_liability_difference = 19000 := 
by
  sorry

end tax_liability_difference_l112_112678


namespace max_dot_product_value_l112_112485

noncomputable def max_dot_product_BQ_CP (λ : ℝ) : ℝ :=
  - (3/5) * (λ - 2/3)^2 - 86/15

theorem max_dot_product_value :
  ∃ (λ : ℝ), 
    0 ≤ λ ∧ λ ≤ 1 ∧ max_dot_product_BQ_CP λ = -86/15 :=
by
  sorry

end max_dot_product_value_l112_112485


namespace betty_cookies_brownies_l112_112316

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end betty_cookies_brownies_l112_112316


namespace blake_change_l112_112560

theorem blake_change :
  let lollipop_count := 4
  let chocolate_count := 6
  let lollipop_cost := 2
  let chocolate_cost := 4 * lollipop_cost
  let total_received := 6 * 10
  let total_cost := (lollipop_count * lollipop_cost) + (chocolate_count * chocolate_cost)
  let change := total_received - total_cost
  change = 4 :=
by
  sorry

end blake_change_l112_112560


namespace find_m_l112_112167

theorem find_m (C D m : ℤ) (h1 : C = D + m) (h2 : C - 1 = 6 * (D - 1)) (h3 : C = D^3) : m = 0 :=
by sorry

end find_m_l112_112167


namespace water_bottles_needed_l112_112959

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l112_112959


namespace upper_limit_of_raise_l112_112256

theorem upper_limit_of_raise (lower upper : ℝ) (h_lower : lower = 0.05)
  (h_upper : upper > 0.08) (h_inequality : ∀ r, lower < r → r < upper)
  : upper < 0.09 :=
sorry

end upper_limit_of_raise_l112_112256


namespace volume_frustum_correct_l112_112809

noncomputable def volume_of_frustum : ℚ :=
  let V_original := (1 / 3 : ℚ) * (16^2) * 10
  let V_smaller := (1 / 3 : ℚ) * (8^2) * 5
  V_original - V_smaller

theorem volume_frustum_correct :
  volume_of_frustum = 2240 / 3 :=
by
  sorry

end volume_frustum_correct_l112_112809


namespace infinite_equal_pairs_l112_112044

theorem infinite_equal_pairs
  (a : ℤ → ℝ)
  (h : ∀ k : ℤ, a k = 1/4 * (a (k - 1) + a (k + 1)))
  (k p : ℤ) (hne : k ≠ p) (heq : a k = a p) :
  ∃ infinite_pairs : ℕ → (ℤ × ℤ), 
  (∀ n : ℕ, (infinite_pairs n).1 ≠ (infinite_pairs n).2) ∧
  (∀ n : ℕ, a (infinite_pairs n).1 = a (infinite_pairs n).2) :=
sorry

end infinite_equal_pairs_l112_112044


namespace sin_75_mul_sin_15_eq_one_fourth_l112_112904

theorem sin_75_mul_sin_15_eq_one_fourth : 
  Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_75_mul_sin_15_eq_one_fourth_l112_112904


namespace addition_neg3_plus_2_multiplication_neg3_times_2_l112_112439

theorem addition_neg3_plus_2 : -3 + 2 = -1 :=
  by
    sorry

theorem multiplication_neg3_times_2 : (-3) * 2 = -6 :=
  by
    sorry

end addition_neg3_plus_2_multiplication_neg3_times_2_l112_112439


namespace absolute_value_of_x_l112_112080

variable (x : ℝ)

theorem absolute_value_of_x (h: (| (3 + x) - (3 - x) |) = 8) : |x| = 4 :=
by sorry

end absolute_value_of_x_l112_112080


namespace dishonest_shopkeeper_weight_l112_112303

noncomputable def weight_used (gain_percent : ℝ) (correct_weight : ℝ) : ℝ :=
  correct_weight / (1 + gain_percent / 100)

theorem dishonest_shopkeeper_weight :
  weight_used 5.263157894736836 1000 = 950 := 
by
  sorry

end dishonest_shopkeeper_weight_l112_112303


namespace krishan_nandan_investment_ratio_l112_112371

theorem krishan_nandan_investment_ratio
    (X t : ℝ) (k : ℝ)
    (h1 : X * t = 6000)
    (h2 : X * t + k * X * 2 * t = 78000) :
    k = 6 := by
  sorry

end krishan_nandan_investment_ratio_l112_112371


namespace log_expression_value_l112_112967

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log_expression_value :
  (log2 8 * (log2 2 / log2 8)) + log2 4 = 3 :=
by
  sorry

end log_expression_value_l112_112967


namespace milk_leftover_after_milkshakes_l112_112437

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end milk_leftover_after_milkshakes_l112_112437


namespace gcd_polynomial_l112_112203

theorem gcd_polynomial {b : ℕ} (h : 570 ∣ b) : Nat.gcd (4*b^3 + 2*b^2 + 5*b + 95) b = 95 := 
sorry

end gcd_polynomial_l112_112203


namespace percentage_sum_l112_112511

theorem percentage_sum (A B C : ℕ) (x y : ℕ)
  (hA : A = 120) (hB : B = 110) (hC : C = 100)
  (hAx : A = C * (1 + x / 100))
  (hBy : B = C * (1 + y / 100)) : x + y = 30 := 
by
  sorry

end percentage_sum_l112_112511


namespace sum_equality_l112_112857

-- Define the conditions and hypothesis
variables (x y z : ℝ)
axiom condition : (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0

-- State the theorem
theorem sum_equality : x + y + z = 21 :=
by sorry

end sum_equality_l112_112857


namespace inner_prod_sum_real_inner_prod_modulus_l112_112343

open Complex

-- Define the given mathematical expressions
noncomputable def pair (α β : ℂ) : ℝ := (1 / 4) * (norm (α + β) ^ 2 - norm (α - β) ^ 2)

noncomputable def inner_prod (α β : ℂ) : ℂ := pair α β + Complex.I * pair α (Complex.I * β)

-- Prove the given mathematical statements

-- 1. Prove that ⟨α, β⟩ + ⟨β, α⟩ is a real number
theorem inner_prod_sum_real (α β : ℂ) : (inner_prod α β + inner_prod β α).im = 0 := sorry

-- 2. Prove that |⟨α, β⟩| = |α| * |β|
theorem inner_prod_modulus (α β : ℂ) : Complex.abs (inner_prod α β) = Complex.abs α * Complex.abs β := sorry

end inner_prod_sum_real_inner_prod_modulus_l112_112343


namespace inequality_problem_l112_112028

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := Real.logb (1 / 2) (1 / 3)

theorem inequality_problem :
  c > a ∧ a > b := by
  sorry

end inequality_problem_l112_112028


namespace triangle_area_ratio_l112_112993

/-
In triangle XYZ, XY=12, YZ=16, and XZ=20. Point D is on XY,
E is on YZ, and F is on XZ. Let XD=p*XY, YE=q*YZ, and ZF=r*XZ,
where p, q, r are positive and satisfy p+q+r=0.9 and p^2+q^2+r^2=0.29.
Prove that the ratio of the area of triangle DEF to the area of triangle XYZ 
can be written in the form m/n where m, n are relatively prime positive 
integers and m+n=137.
-/

theorem triangle_area_ratio :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m + n = 137 ∧ 
  ∃ (p q r : ℝ), p + q + r = 0.9 ∧ p^2 + q^2 + r^2 = 0.29 ∧ 
                  ∀ (XY YZ XZ : ℝ), XY = 12 ∧ YZ = 16 ∧ XZ = 20 → 
                  (1 - (p * (1 - r) + q * (1 - p) + r * (1 - q))) = (37 / 100) :=
by
   sorry

end triangle_area_ratio_l112_112993


namespace day_53_days_from_thursday_is_monday_l112_112114

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end day_53_days_from_thursday_is_monday_l112_112114


namespace cars_on_river_road_l112_112769

-- Define the given conditions
variables (B C : ℕ)
axiom ratio_condition : B = C / 13
axiom difference_condition : B = C - 60 

-- State the theorem to be proved
theorem cars_on_river_road : C = 65 :=
by
  -- proof would go here 
  sorry

end cars_on_river_road_l112_112769


namespace average_age_of_new_students_l112_112422

theorem average_age_of_new_students :
  ∀ (initial_group_avg_age new_group_avg_age : ℝ) (initial_students new_students total_students : ℕ),
  initial_group_avg_age = 14 →
  initial_students = 10 →
  new_group_avg_age = 15 →
  new_students = 5 →
  total_students = initial_students + new_students →
  (new_group_avg_age * total_students - initial_group_avg_age * initial_students) / new_students = 17 :=
by
  intros initial_group_avg_age new_group_avg_age initial_students new_students total_students
  sorry

end average_age_of_new_students_l112_112422


namespace cricket_bat_selling_price_l112_112139

theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) (C : ℝ) (selling_price : ℝ) 
  (h1 : profit = 150) 
  (h2 : profit_percentage = 20) 
  (h3 : profit = (profit_percentage / 100) * C) 
  (h4 : selling_price = C + profit) : 
  selling_price = 900 := 
sorry

end cricket_bat_selling_price_l112_112139


namespace even_function_f_l112_112861

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2^x - 1 else sorry

theorem even_function_f (h_even : ∀ x : ℝ, f x = f (-x)) : f 1 = -1 / 2 := by
  -- proof development skipped
  sorry

end even_function_f_l112_112861


namespace find_c_for_square_of_binomial_l112_112220

theorem find_c_for_square_of_binomial (c : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 50 * x + c = (x + b)^2) → c = 625 :=
by
  intro h
  obtain ⟨b, h⟩ := h
  sorry

end find_c_for_square_of_binomial_l112_112220


namespace find_pairs_l112_112698

theorem find_pairs (n k : ℕ) (h1 : (10^(k-1) ≤ n^n) ∧ (n^n < 10^k)) (h2 : (10^(n-1) ≤ k^k) ∧ (k^k < 10^n)) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) := by
  sorry

end find_pairs_l112_112698


namespace store_profit_loss_l112_112307

theorem store_profit_loss :
  ∃ (x y : ℝ), (1 + 0.25) * x = 135 ∧ (1 - 0.25) * y = 135 ∧ (135 - x) + (135 - y) = -18 :=
by
  sorry

end store_profit_loss_l112_112307


namespace calculate_k_l112_112166

theorem calculate_k (β : ℝ) (hβ : (Real.tan β + 1 / Real.tan β) ^ 2 = k + 1) : k = 1 := by
  sorry

end calculate_k_l112_112166


namespace brian_has_78_white_stones_l112_112317

-- Given conditions
variables (W B : ℕ) (R Bl : ℕ)
variables (x : ℕ)
variables (total_stones : ℕ := 330)
variables (total_collection1 : ℕ := 100)
variables (total_collection3 : ℕ := 130)

-- Condition: First collection stones sum to 100
#check W + B = 100

-- Condition: Brian has more white stones than black ones
#check W > B

-- Condition: Ratio of red to blue stones is 3:2 in the third collection
#check R + Bl = 130
#check R = 3 * x
#check Bl = 2 * x

-- Condition: Total number of stones in all three collections is 330
#check total_stones = total_collection1 + total_collection1 + total_collection3

-- New collection's magnetic stones ratio condition
#check 2 * W / 78 = 2

-- Prove that Brian has 78 white stones
theorem brian_has_78_white_stones
  (h1 : W + B = 100)
  (h2 : W > B)
  (h3 : R + Bl = 130)
  (h4 : R = 3 * x)
  (h5 : Bl = 2 * x)
  (h6 : 2 * W / 78 = 2) :
  W = 78 :=
sorry

end brian_has_78_white_stones_l112_112317


namespace opposite_numbers_add_l112_112978

theorem opposite_numbers_add : ∀ {a b : ℤ}, a + b = 0 → a + b + 3 = 3 :=
by
  intros
  sorry

end opposite_numbers_add_l112_112978


namespace sqrt_eq_two_or_neg_two_l112_112276

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l112_112276


namespace term_2012_of_T_is_2057_l112_112897

-- Define a function that checks if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the sequence T as all natural numbers which are not perfect squares
def T (n : ℕ) : ℕ :=
  (n + Nat.sqrt (4 * n)) 

-- The theorem to state the mathematical proof problem
theorem term_2012_of_T_is_2057 :
  T 2012 = 2057 :=
sorry

end term_2012_of_T_is_2057_l112_112897


namespace bottles_in_cups_l112_112611

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end bottles_in_cups_l112_112611


namespace parallel_lines_m_value_l112_112213

/-- Given two lines l_1: (3 + m) * x + 4 * y = 5 - 3 * m, and l_2: 2 * x + (5 + m) * y = 8,
the value of m for which l_1 is parallel to l_2 is -7. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
sorry

end parallel_lines_m_value_l112_112213


namespace positive_difference_of_solutions_is_zero_l112_112631

theorem positive_difference_of_solutions_is_zero : ∀ (x : ℂ), (x ^ 2 + 3 * x + 4 = 0) → 
  ∀ (y : ℂ), (y ^ 2 + 3 * y + 4 = 0) → |y.re - x.re| = 0 :=
by
  intro x hx y hy
  sorry

end positive_difference_of_solutions_is_zero_l112_112631


namespace solve_for_A_plus_B_l112_112649

-- Definition of the problem conditions
def T := 7 -- The common total sum for rows and columns

-- Summing the rows and columns in the partially filled table
variable (A B : ℕ)
def table_condition :=
  4 + 1 + 2 = T ∧
  2 + A + B = T ∧
  4 + 2 + B = T ∧
  1 + A + B = T

-- Statement to prove
theorem solve_for_A_plus_B (A B : ℕ) (h : table_condition A B) : A + B = 5 :=
by
  sorry

end solve_for_A_plus_B_l112_112649


namespace inequality_proof_l112_112032

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d)
  (h5 : a + b + c + d = 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := by
  sorry

end inequality_proof_l112_112032


namespace min_value_d1_d2_l112_112465

noncomputable def min_distance_sum : ℝ :=
  let d1 (u : ℝ) : ℝ := (1 / 5) * abs (3 * Real.cos u - 4 * Real.sin u - 10)
  let d2 (u : ℝ) : ℝ := 3 - Real.cos u
  let d_sum (u : ℝ) : ℝ := d1 u + d2 u
  ((5 - (4 * Real.sqrt 5 / 5)))

theorem min_value_d1_d2 :
  ∀ (P : ℝ × ℝ) (u : ℝ),
    P = (Real.cos u, Real.sin u) →
    (P.1 ^ 2 + P.2 ^ 2 = 1) →
    let d1 := (1 / 5) * abs (3 * P.1 - 4 * P.2 - 10)
    let d2 := 3 - P.1
    d1 + d2 ≥ (5 - (4 * Real.sqrt 5 / 5)) :=
by
  sorry

end min_value_d1_d2_l112_112465


namespace total_money_is_correct_l112_112427

-- Define the values of different types of coins and the amount of each.
def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4
def cash : ℕ := 45

-- Define the total amount of money.
def total_money : ℕ :=
  (gold_count * gold_value) +
  (silver_count * silver_value) +
  (bronze_count * bronze_value) +
  (titanium_count * titanium_value) + cash

-- The proof statement
theorem total_money_is_correct : total_money = 1055 := by
  sorry

end total_money_is_correct_l112_112427


namespace minimum_travel_time_l112_112286

structure TravelSetup where
  distance_ab : ℝ
  number_of_people : ℕ
  number_of_bicycles : ℕ
  speed_cyclist : ℝ
  speed_pedestrian : ℝ
  unattended_rule : Prop

theorem minimum_travel_time (setup : TravelSetup) : setup.distance_ab = 45 → 
                                                    setup.number_of_people = 3 → 
                                                    setup.number_of_bicycles = 2 → 
                                                    setup.speed_cyclist = 15 → 
                                                    setup.speed_pedestrian = 5 → 
                                                    setup.unattended_rule → 
                                                    ∃ t : ℝ, t = 3 := 
by
  intros
  sorry

end minimum_travel_time_l112_112286


namespace find_number_l112_112642

def is_three_digit_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
  n = 100 * x + 10 * y + z ∧ (100 * x + 10 * y + z) / 11 = x^2 + y^2 + z^2

theorem find_number : ∃ n : ℕ, is_three_digit_number n ∧ n = 550 :=
sorry

end find_number_l112_112642


namespace radius_of_inscribed_circle_is_three_fourths_l112_112672

noncomputable def circle_diameter : ℝ := Real.sqrt 12

noncomputable def radius_of_new_inscribed_circle : ℝ :=
  let R := circle_diameter / 2
  let s := R * Real.sqrt 3
  let h := s * Real.sqrt 3 / 2
  let a := Real.sqrt (h^2 - (h/2)^2)
  a * Real.sqrt 3 / 6

theorem radius_of_inscribed_circle_is_three_fourths :
  radius_of_new_inscribed_circle = 3 / 4 := sorry

end radius_of_inscribed_circle_is_three_fourths_l112_112672


namespace girls_more_than_boys_l112_112107

-- Defining the conditions
def ratio_boys_girls : Nat := 3 / 4
def total_students : Nat := 42

-- Defining the hypothesis based on conditions
theorem girls_more_than_boys : (total_students * ratio_boys_girls) / (3 + 4) * (4 - 3) = 6 := by
  sorry

end girls_more_than_boys_l112_112107


namespace num_red_balls_l112_112991

theorem num_red_balls (x : ℕ) (h1 : 60 = 60) (h2 : (x : ℝ) / (x + 60) = 0.25) : x = 20 :=
sorry

end num_red_balls_l112_112991


namespace maximum_sum_of_consecutive_numbers_on_checkerboard_l112_112875

theorem maximum_sum_of_consecutive_numbers_on_checkerboard : 
  ∃ (grid : Array (Array ℕ)), -- assume existence of a specific 5x5 grid placement of numbers
    (all_used_exactly_once : ∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ i j, grid[i][j] = n) ∧ 
    (consecutive_adjacent : ∀ n, 1 ≤ n ∧ n < 25 → 
       ∃ i j, ∃ (dir : (ℕ × ℕ)), (dir ∈ [(0, 1), (1, 0), (0, -1), (-1, 0)]) ∧ (grid[i][j] = n) ∧ (grid[i + dir.1][j + dir.2] = n + 1)) →
  ∃ color_center_grid : (ℕ × ℕ) → Bool, -- coloring function for the checkerboard
    let black_positions := 
      {pos : (ℕ × ℕ) | color_center_grid pos = true ∧ pos ≠ (2, 2)}, -- center square (indexed from 0) is black
      sum_black := black_positions.sum.1 grid := 
      (∑ n in (1 : set ℕ), n ∧ ¬ ∃ i j, grid[i][j] = n) : ℕ in
    sum_black = 169
    sorry

end maximum_sum_of_consecutive_numbers_on_checkerboard_l112_112875


namespace cube_root_1728_simplified_l112_112292

theorem cube_root_1728_simplified :
  let a := 12
  let b := 1
  a + b = 13 :=
by
  sorry

end cube_root_1728_simplified_l112_112292


namespace height_difference_l112_112236

variable {J L R : ℕ}

theorem height_difference
  (h1 : J = L + 15)
  (h2 : J = 152)
  (h3 : L + R = 295) :
  R - J = 6 :=
sorry

end height_difference_l112_112236


namespace fraction_of_red_marbles_after_tripling_blue_l112_112229

theorem fraction_of_red_marbles_after_tripling_blue (x : ℕ) (h₁ : ∃ y, y = (4 * x) / 7) (h₂ : ∃ z, z = (3 * x) / 7) :
  (3 * x / 7) / (((12 * x) / 7) + ((3 * x) / 7)) = 1 / 5 :=
by
  sorry

end fraction_of_red_marbles_after_tripling_blue_l112_112229


namespace determine_g_l112_112376

variable (g : ℕ → ℕ)

theorem determine_g (h : ∀ x, g (x + 1) = 2 * x + 3) : ∀ x, g x = 2 * x + 1 :=
by
  sorry

end determine_g_l112_112376


namespace range_of_m_l112_112195

-- Define the polynomial p(x)
def p (x : ℝ) (m : ℝ) := x^2 + 2*x - m

-- Given conditions: p(1) is false and p(2) is true
theorem range_of_m (m : ℝ) : 
  (p 1 m ≤ 0) ∧ (p 2 m > 0) → (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l112_112195


namespace length_of_equal_pieces_l112_112129

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l112_112129


namespace magic_triangle_max_sum_l112_112482

/-- In a magic triangle, each of the six consecutive whole numbers 11 to 16 is placed in one of the circles. 
    The sum, S, of the three numbers on each side of the triangle is the same. One of the sides must contain 
    three consecutive numbers. Prove that the largest possible value for S is 41. -/
theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ), 
  (a = 11 ∨ a = 12 ∨ a = 13 ∨ a = 14 ∨ a = 15 ∨ a = 16) ∧
  (b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16) ∧
  (c = 11 ∨ c = 12 ∨ c = 13 ∨ c = 14 ∨ c = 15 ∨ c = 16) ∧
  (d = 11 ∨ d = 12 ∨ d = 13 ∨ d = 14 ∨ d = 15 ∨ d = 16) ∧
  (e = 11 ∨ e = 12 ∨ e = 13 ∨ e = 14 ∨ e = 15 ∨ e = 16) ∧
  (f = 11 ∨ f = 12 ∨ f = 13 ∨ f = 14 ∨ f = 15 ∨ f = 16) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
  (a + b + c = S) ∧ (c + d + e = S) ∧ (e + f + a = S) ∧
  (∃ k, a = k ∧ b = k+1 ∧ c = k+2 ∨ b = k ∧ c = k+1 ∧ d = k+2 ∨ c = k ∧ d = k+1 ∧ e = k+2 ∨ d = k ∧ e = k+1 ∧ f = k+2) →
  S = 41 :=
by
  sorry

end magic_triangle_max_sum_l112_112482


namespace angle_of_inclination_l112_112450

theorem angle_of_inclination (θ : ℝ) (h_range : 0 ≤ θ ∧ θ < 180)
  (h_line : ∀ x y : ℝ, x + y - 1 = 0 → x = -y + 1) :
  θ = 135 :=
by 
  sorry

end angle_of_inclination_l112_112450


namespace ninety_eight_squared_l112_112324

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l112_112324


namespace cd_total_l112_112494

theorem cd_total :
  ∀ (Kristine Dawn Mark Alice : ℕ),
  Dawn = 10 →
  Kristine = Dawn + 7 →
  Mark = 2 * Kristine →
  Alice = (Kristine + Mark) - 5 →
  (Dawn + Kristine + Mark + Alice) = 107 :=
by
  intros Kristine Dawn Mark Alice hDawn hKristine hMark hAlice
  rw [hDawn, hKristine, hMark, hAlice]
  sorry

end cd_total_l112_112494


namespace percent_of_x_l112_112475

variable {x y z : ℝ}

-- Define the given conditions
def cond1 (z y : ℝ) : Prop := 0.45 * z = 0.9 * y
def cond2 (z x : ℝ) : Prop := z = 1.5 * x

-- State the theorem to prove
theorem percent_of_x (h1 : cond1 z y) (h2 : cond2 z x) : y = 0.75 * x :=
sorry

end percent_of_x_l112_112475


namespace total_heartbeats_correct_l112_112156

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l112_112156


namespace pizza_party_l112_112428

theorem pizza_party (boys girls : ℕ) :
  (7 * boys + 3 * girls ≤ 59) ∧ (6 * boys + 2 * girls ≥ 49) ∧ (boys + girls ≤ 10) → 
  boys = 8 ∧ girls = 1 := 
by sorry

end pizza_party_l112_112428


namespace unit_digit_7_pow_2023_l112_112250

theorem unit_digit_7_pow_2023 : (7^2023) % 10 = 3 :=
by
  -- Provide proof here
  sorry

end unit_digit_7_pow_2023_l112_112250


namespace building_total_floors_l112_112448

def earl_final_floor (start : ℕ) : ℕ :=
  start + 5 - 2 + 7

theorem building_total_floors (start : ℕ) (current : ℕ) (remaining : ℕ) (total : ℕ) :
  earl_final_floor start = current →
  remaining = 9 →
  total = current + remaining →
  start = 1 →
  total = 20 := by
sorry

end building_total_floors_l112_112448


namespace factorize_expression_l112_112572

theorem factorize_expression (x : ℝ) :
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) :=
  sorry

end factorize_expression_l112_112572


namespace speed_of_stream_l112_112670

-- Define the conditions as premises
def boat_speed_in_still_water : ℝ := 24
def travel_time_downstream : ℝ := 3
def distance_downstream : ℝ := 84

-- The effective speed downstream is the sum of the boat's speed and the speed of the stream
def effective_speed_downstream (stream_speed : ℝ) : ℝ :=
  boat_speed_in_still_water + stream_speed

-- The speed of the stream
theorem speed_of_stream (stream_speed : ℝ) :
  84 = effective_speed_downstream stream_speed * travel_time_downstream →
  stream_speed = 4 :=
by
  sorry

end speed_of_stream_l112_112670


namespace remainder_2001_to_2005_mod_19_l112_112415

theorem remainder_2001_to_2005_mod_19 :
  (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 :=
by
  -- Use modular arithmetic properties to convert each factor
  have h2001 : 2001 % 19 = 6 := by sorry
  have h2002 : 2002 % 19 = 7 := by sorry
  have h2003 : 2003 % 19 = 8 := by sorry
  have h2004 : 2004 % 19 = 9 := by sorry
  have h2005 : 2005 % 19 = 10 := by sorry

  -- Compute the product modulo 19
  have h_prod : (6 * 7 * 8 * 9 * 10) % 19 = 11 := by sorry

  -- Combining these results
  have h_final : ((2001 * 2002 * 2003 * 2004 * 2005) % 19) = (6 * 7 * 8 * 9 * 10) % 19 := by sorry
  exact Eq.trans h_final h_prod

end remainder_2001_to_2005_mod_19_l112_112415


namespace min_value_x_plus_y_l112_112034

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end min_value_x_plus_y_l112_112034


namespace parabola_vertex_l112_112637

theorem parabola_vertex:
  ∀ x: ℝ, ∀ y: ℝ, (y = (1 / 2) * x ^ 2 - 4 * x + 3) → (x = 4 ∧ y = -5) :=
sorry

end parabola_vertex_l112_112637


namespace distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l112_112210

variable (m : ℝ)

-- Part 1: Prove that if the quadratic equation has two distinct real roots, then m < 13/4.
theorem distinct_real_roots_iff_m_lt_13_over_4 (h : (3 * 3 - 4 * (m - 1)) > 0) : m < 13 / 4 := 
by
  sorry

-- Part 2: Prove that if the quadratic equation has two equal real roots, then the root is 3/2.
theorem equal_real_roots_root_eq_3_over_2 (h : (3 * 3 - 4 * (m - 1)) = 0) : m = 13 / 4 ∧ ∀ x, (x^2 + 3 * x + (13/4 - 1) = 0) → x = 3 / 2 :=
by
  sorry

end distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l112_112210


namespace geometric_series_common_ratio_l112_112311

theorem geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h_a : a = 500) (h_S : S = 3000) :
  ∃ r : ℝ, r = 5 / 6 :=
by
  sorry

end geometric_series_common_ratio_l112_112311


namespace athlete_heartbeats_during_race_l112_112150

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l112_112150


namespace sulfuric_acid_moles_l112_112451

-- Definitions based on the conditions
def iron_moles := 2
def hydrogen_moles := 2

-- The reaction equation in the problem
def reaction (Fe H₂SO₄ : ℕ) : Prop :=
  Fe + H₂SO₄ = hydrogen_moles

-- Goal: prove the number of moles of sulfuric acid used is 2
theorem sulfuric_acid_moles (Fe : ℕ) (H₂SO₄ : ℕ) (h : reaction Fe H₂SO₄) :
  H₂SO₄ = 2 :=
sorry

end sulfuric_acid_moles_l112_112451


namespace point_on_x_axis_l112_112859

theorem point_on_x_axis (m : ℤ) (P : ℤ × ℤ) (hP : P = (m + 3, m + 1)) (h : P.2 = 0) : P = (2, 0) :=
by 
  sorry

end point_on_x_axis_l112_112859


namespace athlete_heartbeats_calculation_l112_112152

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l112_112152


namespace range_of_a_l112_112480

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ x^2 + (1 - a) * x + 3 - a > 0) ↔ a < 3 := 
sorry

end range_of_a_l112_112480


namespace largest_A_l112_112496

namespace EquivalentProofProblem

def F (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f (3 * x) ≥ f (f (2 * x)) + x

theorem largest_A (f : ℝ → ℝ) (hf : F f) (x : ℝ) (hx : x > 0) : 
  ∃ A, (∀ (f : ℝ → ℝ), F f → ∀ x, x > 0 → f x ≥ A * x) ∧ A = 1 / 2 :=
sorry

end EquivalentProofProblem

end largest_A_l112_112496


namespace find_n_l112_112064

theorem find_n (n : ℕ) (h : 2^n = 2 * 16^2 * 4^3) : n = 15 :=
by
  sorry

end find_n_l112_112064


namespace frank_initial_candy_l112_112025

theorem frank_initial_candy (n : ℕ) (h1 : n = 21) (h2 : 2 > 0) :
  2 * n = 42 :=
by
  --* Use the hypotheses to establish the required proof
  sorry

end frank_initial_candy_l112_112025


namespace dot_product_is_4_l112_112214

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, -1)

-- Define the condition that a is parallel to (a + b)
def is_parallel (u v : ℝ × ℝ) : Prop := 
  (u.1 * v.2 - u.2 * v.1) = 0

theorem dot_product_is_4 (x : ℝ) (h_parallel : is_parallel (a x) (a x + b)) : 
  (a x).1 * b.1 + (a x).2 * b.2 = 4 :=
sorry

end dot_product_is_4_l112_112214


namespace find_length_of_AB_l112_112603

variable (A B C : ℝ)
variable (cos_C_div2 BC AC AB : ℝ)
variable (C_gt_0 : 0 < C / 2) (C_lt_pi : C / 2 < Real.pi)

axiom h1 : cos_C_div2 = Real.sqrt 5 / 5
axiom h2 : BC = 1
axiom h3 : AC = 5
axiom h4 : AB = Real.sqrt (BC ^ 2 + AC ^ 2 - 2 * BC * AC * (2 * cos_C_div2 ^ 2 - 1))

theorem find_length_of_AB : AB = 4 * Real.sqrt 2 :=
by
  sorry

end find_length_of_AB_l112_112603


namespace gumball_difference_l112_112319

theorem gumball_difference :
  let c := 17
  let l := 12
  let a := 24
  let t := 8
  let n := c + l + a + t
  let low := 14
  let high := 32
  ∃ x : ℕ, (low ≤ (n + x) / 7 ∧ (n + x) / 7 ≤ high) →
  (∃ x_min x_max, x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126) :=
by
  sorry

end gumball_difference_l112_112319


namespace selection_methods_correct_l112_112402

-- Define the number of students in each year
def first_year_students : ℕ := 3
def second_year_students : ℕ := 5
def third_year_students : ℕ := 4

-- Define the total number of different selection methods
def total_selection_methods : ℕ := first_year_students + second_year_students + third_year_students

-- Lean statement to prove the question is equivalent to the answer
theorem selection_methods_correct :
  total_selection_methods = 12 := by
  sorry

end selection_methods_correct_l112_112402


namespace second_group_students_l112_112679

theorem second_group_students 
  (total_students : ℕ) 
  (first_group_students : ℕ) 
  (h1 : total_students = 71) 
  (h2 : first_group_students = 34) : 
  total_students - first_group_students = 37 :=
by 
  sorry

end second_group_students_l112_112679


namespace solution_set_of_inequality_l112_112771

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l112_112771


namespace alpha_beta_property_l112_112617

theorem alpha_beta_property
  (α β : ℝ)
  (hαβ_roots : ∀ x : ℝ, (x = α ∨ x = β) → x^2 + x - 2023 = 0) :
  α^2 + 2 * α + β = 2022 :=
by
  sorry

end alpha_beta_property_l112_112617


namespace hulk_strength_l112_112514

theorem hulk_strength:
    ∃ n: ℕ, (2^(n-1) > 1000) ∧ (∀ m: ℕ, (2^(m-1) > 1000 → n ≤ m)) := sorry

end hulk_strength_l112_112514


namespace perimeter_of_stadium_l112_112767

-- Define the length and breadth as given conditions.
def length : ℕ := 100
def breadth : ℕ := 300

-- Define the perimeter function for a rectangle.
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Prove that the perimeter of the stadium is 800 meters given the length and breadth.
theorem perimeter_of_stadium : perimeter length breadth = 800 := 
by
  -- Placeholder for the formal proof.
  sorry

end perimeter_of_stadium_l112_112767


namespace no_such_function_exists_l112_112540

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (f x) = x^2 - 1996 :=
by
  sorry

end no_such_function_exists_l112_112540


namespace combination_n_2_l112_112411

theorem combination_n_2 (n : ℕ) (h : n > 0) : 
  nat.choose n 2 = n * (n - 1) / 2 :=
sorry

end combination_n_2_l112_112411


namespace debby_total_photos_l112_112251

theorem debby_total_photos (friends_photos family_photos : ℕ) (h1 : friends_photos = 63) (h2 : family_photos = 23) : friends_photos + family_photos = 86 :=
by sorry

end debby_total_photos_l112_112251


namespace balloons_given_by_mom_l112_112288

-- Definitions of the initial and total number of balloons
def initial_balloons := 26
def total_balloons := 60

-- Theorem: Proving the number of balloons Tommy's mom gave him
theorem balloons_given_by_mom : total_balloons - initial_balloons = 34 :=
by
  -- This proof is obvious from the setup, so we write sorry to skip the proof.
  sorry

end balloons_given_by_mom_l112_112288


namespace combined_value_of_silver_and_gold_l112_112818

noncomputable def silver_cube_side : ℝ := 3
def silver_weight_per_cubic_inch : ℝ := 6
def silver_price_per_ounce : ℝ := 25
def gold_layer_fraction : ℝ := 0.5
def gold_weight_per_square_inch : ℝ := 0.1
def gold_price_per_ounce : ℝ := 1800
def markup_percentage : ℝ := 1.10

def calculate_combined_value (side weight_per_cubic_inch silver_price layer_fraction weight_per_square_inch gold_price markup : ℝ) : ℝ :=
  let volume := side^3
  let weight_silver := volume * weight_per_cubic_inch
  let value_silver := weight_silver * silver_price
  let surface_area := 6 * side^2
  let area_gold := surface_area * layer_fraction
  let weight_gold := area_gold * weight_per_square_inch
  let value_gold := weight_gold * gold_price
  let total_value_before_markup := value_silver + value_gold
  let selling_price := total_value_before_markup * (1 + markup)
  selling_price

theorem combined_value_of_silver_and_gold :
  calculate_combined_value silver_cube_side silver_weight_per_cubic_inch silver_price_per_ounce gold_layer_fraction gold_weight_per_square_inch gold_price_per_ounce markup_percentage = 18711 :=
by
  sorry

end combined_value_of_silver_and_gold_l112_112818


namespace train_length_is_sixteenth_mile_l112_112943

theorem train_length_is_sixteenth_mile
  (train_speed : ℕ)
  (bridge_length : ℕ)
  (man_speed : ℕ)
  (cross_time : ℚ)
  (man_distance : ℚ)
  (length_of_train : ℚ)
  (h1 : train_speed = 80)
  (h2 : bridge_length = 1)
  (h3 : man_speed = 5)
  (h4 : cross_time = bridge_length / train_speed)
  (h5 : man_distance = man_speed * cross_time)
  (h6 : length_of_train = man_distance) :
  length_of_train = 1 / 16 :=
by sorry

end train_length_is_sixteenth_mile_l112_112943


namespace necessary_but_not_sufficient_for_inequality_l112_112127

theorem necessary_but_not_sufficient_for_inequality : 
  ∀ x : ℝ, (-2 < x ∧ x < 4) → (x < 5) ∧ (¬(x < 5) → (-2 < x ∧ x < 4) ) :=
by 
  sorry

end necessary_but_not_sufficient_for_inequality_l112_112127


namespace area_of_fourth_rectangle_l112_112141

theorem area_of_fourth_rectangle
    (x y z w : ℝ)
    (h1 : x * y = 24)
    (h2 : z * y = 15)
    (h3 : z * w = 9) :
    y * w = 15 := 
sorry

end area_of_fourth_rectangle_l112_112141


namespace elizabeth_stickers_count_l112_112176

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l112_112176


namespace sqrt_of_4_l112_112280

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l112_112280


namespace jake_last_10_shots_l112_112309

-- conditions
variable (total_shots_initially : ℕ) (shots_made_initially : ℕ) (percentage_initial : ℝ)
variable (total_shots_finally : ℕ) (shots_made_finally : ℕ) (percentage_final : ℝ)

axiom initial_conditions : shots_made_initially = percentage_initial * total_shots_initially
axiom final_conditions : shots_made_finally = percentage_final * total_shots_finally
axiom shots_difference : total_shots_finally - total_shots_initially = 10

-- prove that Jake made 7 out of the last 10 shots
theorem jake_last_10_shots : total_shots_initially = 30 → 
                             percentage_initial = 0.60 →
                             total_shots_finally = 40 → 
                             percentage_final = 0.62 →
                             shots_made_finally - shots_made_initially = 7 :=
by
  -- proofs to be filled in
  sorry

end jake_last_10_shots_l112_112309


namespace prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l112_112085

def X := {-1, 0, 1}
def A_accuracy := 0.5
def B_accuracy := 0.6

theorem prob_X_distribution :
  ∀ (x : X),
  (x = -1) → (P(X = -1) = 0.3) ∧
  (x = 0) → (P(X = 0) = 0.5) ∧
  (x = 1) → (P(X = 1) = 0.2) := by sorry

theorem prob_tie :
  P(tie) = 0.2569 := by sorry

def Y := {2, 3, 4}

theorem prob_Y_distribution :
  ∀ (y : Y),
  (y = 2) → (P(Y = 2) = 0.13) ∧
  (y = 3) → (P(Y = 3) = 0.13) ∧
  (y = 4) → (P(Y = 4) = 0.74) := by sorry

theorem expected_Y :
  E(Y) = 3.61 := by sorry

end prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l112_112085


namespace fraction_walk_home_l112_112817

theorem fraction_walk_home : 
  (1 - ((1 / 2) + (1 / 4) + (1 / 10) + (1 / 8))) = (1 / 40) :=
by 
  sorry

end fraction_walk_home_l112_112817


namespace max_intersections_three_circles_one_line_l112_112116

theorem max_intersections_three_circles_one_line (c1 c2 c3 : Circle) (L : Line) :
  greatest_number_points_of_intersection c1 c2 c3 L = 12 :=
sorry

end max_intersections_three_circles_one_line_l112_112116


namespace negation_of_existence_l112_112521

theorem negation_of_existence (h: ∃ x : ℝ, 0 < x ∧ (Real.log x + x - 1 ≤ 0)) :
  ¬ (∀ x : ℝ, 0 < x → ¬ (Real.log x + x - 1 ≤ 0)) :=
sorry

end negation_of_existence_l112_112521


namespace correct_sum_104th_parenthesis_l112_112330

noncomputable def sum_104th_parenthesis : ℕ := sorry

theorem correct_sum_104th_parenthesis :
  sum_104th_parenthesis = 2072 := 
by 
  sorry

end correct_sum_104th_parenthesis_l112_112330


namespace ratio_used_to_total_apples_l112_112561

noncomputable def total_apples_bonnie : ℕ := 8
noncomputable def total_apples_samuel : ℕ := total_apples_bonnie + 20
noncomputable def eaten_apples_samuel : ℕ := total_apples_samuel / 2
noncomputable def used_for_pie_samuel : ℕ := total_apples_samuel - eaten_apples_samuel - 10

theorem ratio_used_to_total_apples : used_for_pie_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 1 ∧
                                     total_apples_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 7 := by
  sorry

end ratio_used_to_total_apples_l112_112561


namespace analyze_a_b_m_n_l112_112237

theorem analyze_a_b_m_n (a b m n : ℕ) (ha : 1 < a) (hb : 1 < b) (hm : 1 < m) (hn : 1 < n)
  (h1 : Prime (a^n - 1))
  (h2 : Prime (b^m + 1)) :
  n = 2 ∧ ∃ k : ℕ, m = 2^k :=
by
  sorry

end analyze_a_b_m_n_l112_112237


namespace find_first_month_sales_l112_112803

noncomputable def avg_sales (sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ) : ℕ :=
(sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6

theorem find_first_month_sales :
  let sales_2 := 6927
  let sales_3 := 6855
  let sales_4 := 7230
  let sales_5 := 6562
  let sales_6 := 5091
  let avg_sales_needed := 6500
  ∃ sales_1, avg_sales sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 = avg_sales_needed := 
by
  sorry

end find_first_month_sales_l112_112803


namespace hyperbola_eq_l112_112853

theorem hyperbola_eq (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : -b / a = -1/2) (h4 : a^2 + b^2 = 5^2) :
  ∃ (a b : ℝ), (a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
  (∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2 - y^2 / b^2 = 1)})) := sorry

end hyperbola_eq_l112_112853


namespace cost_of_each_burger_l112_112835

theorem cost_of_each_burger (purchases_per_day : ℕ) (total_days : ℕ) (total_amount_spent : ℕ)
  (h1 : purchases_per_day = 4) (h2 : total_days = 30) (h3 : total_amount_spent = 1560) : 
  total_amount_spent / (purchases_per_day * total_days) = 13 :=
by
  subst h1
  subst h2
  subst h3
  sorry

end cost_of_each_burger_l112_112835


namespace absents_probability_is_correct_l112_112232

-- Conditions
def probability_absent := 1 / 10
def probability_present := 9 / 10

-- Calculation of combined probability
def combined_probability : ℚ :=
  3 * (probability_absent * probability_absent * probability_present)

-- Conversion to percentage
def percentage_probability : ℚ :=
  combined_probability * 100

-- Theorem statement
theorem absents_probability_is_correct :
  percentage_probability = 2.7 := 
sorry

end absents_probability_is_correct_l112_112232


namespace circles_intersect_l112_112358

-- Define the parameters and conditions given in the problem.
def r1 : ℝ := 5  -- Radius of circle O1
def r2 : ℝ := 8  -- Radius of circle O2
def d : ℝ := 8   -- Distance between the centers of O1 and O2

-- The main theorem that needs to be proven.
theorem circles_intersect (r1 r2 d : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 8) (h_d : d = 8) :
  r2 - r1 < d ∧ d < r1 + r2 :=
by
  sorry

end circles_intersect_l112_112358


namespace factorize_expr_l112_112695

def my_expr (a b : ℤ) : ℤ := 4 * a^2 * b - b

theorem factorize_expr (a b : ℤ) : my_expr a b = b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end factorize_expr_l112_112695


namespace min_value_a_is_1_or_100_l112_112747

noncomputable def f (x : ℝ) : ℝ := x + 100 / x

theorem min_value_a_is_1_or_100 (a : ℝ) (m1 m2 : ℝ) 
  (h1 : a > 0) 
  (h_m1 : ∀ x, 0 < x ∧ x ≤ a → f x ≥ m1)
  (h_m1_min : ∃ x, 0 < x ∧ x ≤ a ∧ f x = m1)
  (h_m2 : ∀ x, a ≤ x → f x ≥ m2)
  (h_m2_min : ∃ x, a ≤ x ∧ f x = m2)
  (h_prod : m1 * m2 = 2020) : 
  a = 1 ∨ a = 100 :=
sorry

end min_value_a_is_1_or_100_l112_112747


namespace sum_of_factors_180_l112_112655

open BigOperators

def sum_of_factors (n : ℕ) : ℕ :=
  ∑ d in (finset.range (n+1)).filter (λ d, n % d = 0), d

noncomputable def factor_180 := 180

theorem sum_of_factors_180 : sum_of_factors factor_180 = 546 := by
  sorry

end sum_of_factors_180_l112_112655


namespace f_1993_of_3_l112_112723

def f (x : ℚ) := (1 + x) / (1 - 3 * x)

def f_n (x : ℚ) : ℕ → ℚ
| 0 => x
| (n + 1) => f (f_n x n)

theorem f_1993_of_3 :
  f_n 3 1993 = 1 / 5 :=
sorry

end f_1993_of_3_l112_112723


namespace remaining_amount_after_purchase_l112_112333

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end remaining_amount_after_purchase_l112_112333


namespace ratio_fifth_term_l112_112839

-- Definitions of arithmetic sequences and sums
def arithmetic_seq_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * a 1 + (n - 1) * d 1) / 2

-- Conditions
variables (S_n S'_n : ℕ → ℕ) (n : ℕ)

-- Given conditions
axiom ratio_sum : ∀ (n : ℕ), S_n n / S'_n n = (5 * n + 3) / (2 * n + 7)
axiom sums_at_9 : S_n 9 = 9 * (S_n 1 + S_n 9) / 2
axiom sums'_at_9 : S'_n 9 = 9 * (S'_n 1 + S'_n 9) / 2

-- Theorem to prove
theorem ratio_fifth_term : (9 * (S_n 1 + S_n 9) / 2) / (9 * (S'_n 1 + S'_n 9) / 2) = 48 / 25 := sorry

end ratio_fifth_term_l112_112839


namespace inequality_always_true_l112_112535

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end inequality_always_true_l112_112535


namespace sequence_8th_term_is_sqrt23_l112_112643

noncomputable def sequence_term (n : ℕ) : ℝ := Real.sqrt (2 + 3 * (n - 1))

theorem sequence_8th_term_is_sqrt23 : sequence_term 8 = Real.sqrt 23 :=
by
  sorry

end sequence_8th_term_is_sqrt23_l112_112643


namespace min_children_l112_112173

theorem min_children (x : ℕ) : 
  (4 * x + 28 - 5 * (x - 1) < 5) ∧ (4 * x + 28 - 5 * (x - 1) ≥ 2) → (x = 29) :=
by
  sorry

end min_children_l112_112173


namespace parallel_lines_same_slope_l112_112445

theorem parallel_lines_same_slope (k : ℝ) : 
  (2*x + y + 1 = 0) ∧ (y = k*x + 3) → (k = -2) := 
by
  sorry

end parallel_lines_same_slope_l112_112445


namespace number_symmetry_equation_l112_112249

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) :=
by
  sorry

end number_symmetry_equation_l112_112249


namespace base7_to_base10_of_645_l112_112802

theorem base7_to_base10_of_645 :
  (6 * 7^2 + 4 * 7^1 + 5 * 7^0) = 327 := 
by 
  sorry

end base7_to_base10_of_645_l112_112802


namespace cube_root_of_neg_27_over_8_l112_112261

theorem cube_root_of_neg_27_over_8 :
  (- (3 : ℝ) / 2) ^ 3 = - (27 / 8 : ℝ) := 
by
  sorry

end cube_root_of_neg_27_over_8_l112_112261


namespace stick_length_l112_112996

theorem stick_length (x : ℕ) (h1 : 2 * x + (2 * x - 1) = 14) : x = 3 := sorry

end stick_length_l112_112996


namespace proposition_D_l112_112122

/-- Lean statement for proving the correct proposition D -/
theorem proposition_D {a b : ℝ} (h : |a| < b) : a^2 < b^2 :=
sorry

end proposition_D_l112_112122


namespace parabola_directrix_l112_112202

theorem parabola_directrix
  (p : ℝ) (hp : p > 0)
  (O : ℝ × ℝ := (0,0))
  (Focus_F : ℝ × ℝ := (p / 2, 0))
  (Point_P : ℝ × ℝ)
  (Point_Q : ℝ × ℝ)
  (H1 : Point_P.1 = p / 2 ∧ Point_P.2^2 = 2 * p * Point_P.1)
  (H2 : Point_P.1 = Point_P.1) -- This comes out of the perpendicularity of PF to x-axis
  (H3 : Point_Q.2 = 0)
  (H4 : ∃ k_OP slope_OP, slope_OP = 2 ∧ ∃ k_PQ slope_PQ, slope_PQ = -1 / 2 ∧ k_OP * k_PQ = -1)
  (H5 : abs (Point_Q.1 - Focus_F.1) = 6) :
  x = -3 / 2 := 
sorry

end parabola_directrix_l112_112202


namespace eric_green_marbles_l112_112181

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end eric_green_marbles_l112_112181


namespace unit_fraction_decomposition_l112_112018

theorem unit_fraction_decomposition (n : ℕ) (hn : 0 < n): 
  (1 : ℚ) / n = (1 : ℚ) / (2 * n) + (1 : ℚ) / (3 * n) + (1 : ℚ) / (6 * n) :=
by
  sorry

end unit_fraction_decomposition_l112_112018


namespace students_playing_long_tennis_l112_112606

theorem students_playing_long_tennis (n F B N L : ℕ)
  (h1 : n = 35)
  (h2 : F = 26)
  (h3 : B = 17)
  (h4 : N = 6)
  (h5 : L = (n - N) - (F - B)) :
  L = 20 :=
by
  sorry

end students_playing_long_tennis_l112_112606


namespace factorize_difference_of_squares_l112_112697

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) :=
by 
  sorry

end factorize_difference_of_squares_l112_112697


namespace min_value_frac_l112_112489

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end min_value_frac_l112_112489


namespace roots_quadratic_expr_l112_112847

theorem roots_quadratic_expr (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0)
    (h2 : Polynomial.eval n (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0) :
  m^2 + m * n + 2 * m = 0 :=
sorry

end roots_quadratic_expr_l112_112847


namespace range_of_m_l112_112109

noncomputable def has_two_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m 

theorem range_of_m (m : ℝ) : has_two_solutions m ↔ m > -(1/4) :=
sorry

end range_of_m_l112_112109


namespace sum_of_coefficients_l112_112902

-- Defining the given conditions
def vertex : ℝ × ℝ := (5, -4)
def point : ℝ × ℝ := (3, -2)

-- Defining the problem to prove the sum of the coefficients
theorem sum_of_coefficients (a b c : ℝ)
  (h_eq : ∀ y, 5 = a * ((-4) + y)^2 + c)
  (h_pt : 3 = a * ((-4) + (-2))^2 + b * (-2) + c) :
  a + b + c = -15 / 2 :=
sorry

end sum_of_coefficients_l112_112902


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l112_112913

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l112_112913


namespace probability_two_points_square_l112_112074

def gcd (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c  

theorem probability_two_points_square {a b c : ℕ} (hx : gcd a b c = 1)
  (h : (26 - Real.pi) / 32 = (a - b * Real.pi) / c) : a + b + c = 59 :=
by
  sorry

end probability_two_points_square_l112_112074


namespace pq_sum_of_harmonic_and_geometric_sequences_l112_112619

theorem pq_sum_of_harmonic_and_geometric_sequences
  (x y z : ℝ)
  (h1 : (1 / x - 1 / y) / (1 / y - 1 / z) = 1)
  (h2 : 3 * x * y = 7 * z) :
  ∃ p q : ℕ, (Nat.gcd p q = 1) ∧ p + q = 79 :=
by
  sorry

end pq_sum_of_harmonic_and_geometric_sequences_l112_112619


namespace find_simple_interest_rate_l112_112473

variable (P : ℝ) (n : ℕ) (r_c : ℝ) (t : ℝ) (I_c : ℝ) (I_s : ℝ) (r_s : ℝ)

noncomputable def compound_interest_amount (P r_c : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r_c / n) ^ (n * t)

noncomputable def simple_interest_amount (P r_s : ℝ) (t : ℝ) : ℝ :=
  P * r_s * t

theorem find_simple_interest_rate
  (hP : P = 5000)
  (hr_c : r_c = 0.16)
  (hn : n = 2)
  (ht : t = 1)
  (hI_c : I_c = compound_interest_amount P r_c n t - P)
  (hI_s : I_s = I_c - 16)
  (hI_s_def : I_s = simple_interest_amount P r_s t) :
  r_s = 0.1632 := sorry

end find_simple_interest_rate_l112_112473


namespace sum_is_24000_l112_112125

theorem sum_is_24000 (P : ℝ) (R : ℝ) (T : ℝ) : 
  (R = 5) → (T = 2) →
  ((P * (1 + R / 100)^T - P) - (P * R * T / 100) = 60) →
  P = 24000 :=
by
  sorry

end sum_is_24000_l112_112125


namespace longest_side_l112_112744

theorem longest_side (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 240)
  (h2 : l * w = 2880) :
  l = 86.835 ∨ w = 86.835 :=
sorry

end longest_side_l112_112744


namespace correct_survey_method_l112_112924

-- Definitions for the conditions
def visionStatusOfMiddleSchoolStudentsNationwide := "Comprehensive survey is impractical for this large population."
def batchFoodContainsPreservatives := "Comprehensive survey is unnecessary, sampling survey would suffice."
def airQualityOfCity := "Comprehensive survey is impractical due to vast area, sampling survey is appropriate."
def passengersCarryProhibitedItems := "Comprehensive survey is necessary for security reasons."

-- Theorem stating that option C is the correct and reasonable choice
theorem correct_survey_method : airQualityOfCity = "Comprehensive survey is impractical due to vast area, sampling survey is appropriate." := by
  sorry

end correct_survey_method_l112_112924


namespace length_of_equal_pieces_l112_112130

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l112_112130


namespace count_numbers_seven_times_sum_of_digits_l112_112217

open Nat

-- Function to calculate sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  ((n.digits 10).sum)

theorem count_numbers_seven_times_sum_of_digits :
  { n : ℕ // n > 0 ∧ n < 1000 ∧ (n = 7 * sum_of_digits n) }.card = 4 :=
by
  -- Proof would go here
  sorry

end count_numbers_seven_times_sum_of_digits_l112_112217


namespace largest_fraction_l112_112973

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end largest_fraction_l112_112973


namespace intersection_of_A_and_B_l112_112045

-- Define sets A and B
def setA : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x < 2}

-- Prove that A ∩ B = (-3, 2)
theorem intersection_of_A_and_B : {x : ℝ | x ∈ setA ∧ x ∈ setB} = {x : ℝ | -3 < x ∧ x < 2} := 
by 
  sorry

end intersection_of_A_and_B_l112_112045


namespace unique_g_zero_l112_112896

theorem unique_g_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) = g (x) + g (y) - 1) : g 0 = 1 :=
by
  sorry

end unique_g_zero_l112_112896


namespace exists_n_gt_1958_l112_112668

noncomputable def polyline_path (n : ℕ) : ℝ := sorry
noncomputable def distance_to_origin (n : ℕ) : ℝ := sorry 
noncomputable def sum_lengths (n : ℕ) : ℝ := sorry

theorem exists_n_gt_1958 :
  ∃ (n : ℕ), n > 1958 ∧ (sum_lengths n) / (distance_to_origin n) > 1958 := 
sorry

end exists_n_gt_1958_l112_112668


namespace max_band_members_l112_112111

theorem max_band_members (n : ℤ) (h1 : 30 * n % 21 = 9) (h2 : 30 * n < 1500) : 30 * n ≤ 1470 :=
by
  -- Proof to be filled in later
  sorry

end max_band_members_l112_112111


namespace solve_for_y_l112_112065

theorem solve_for_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 8) : y = 1 / 3 := 
by
  sorry

end solve_for_y_l112_112065


namespace balloons_division_correct_l112_112778

def number_of_balloons_per_school (yellow blue more_black num_schools: ℕ) : ℕ :=
  let black := yellow + more_black
  let total := yellow + blue + black
  total / num_schools

theorem balloons_division_correct :
  number_of_balloons_per_school 3414 5238 1762 15 = 921 := 
by
  sorry

end balloons_division_correct_l112_112778


namespace polynomial_factorization_l112_112391

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2
  = (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) :=
sorry

end polynomial_factorization_l112_112391


namespace intersection_of_A_and_B_l112_112462

def A : Set ℝ := { x | x^2 - 5 * x - 6 ≤ 0 }

def B : Set ℝ := { x | x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -1 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l112_112462


namespace original_average_age_l112_112635

variable (A : ℕ)
variable (N : ℕ := 2)
variable (new_avg_age : ℕ := 32)
variable (age_decrease : ℕ := 4)

theorem original_average_age :
  (A * N + new_avg_age * 2) / (N + 2) = A - age_decrease → A = 40 := 
by
  sorry

end original_average_age_l112_112635


namespace initial_number_of_persons_l112_112893

noncomputable def avg_weight_change : ℝ := 5.5
noncomputable def old_person_weight : ℝ := 68
noncomputable def new_person_weight : ℝ := 95.5
noncomputable def weight_diff : ℝ := new_person_weight - old_person_weight

theorem initial_number_of_persons (N : ℝ) 
  (h1 : avg_weight_change * N = weight_diff) : N = 5 :=
  by
  sorry

end initial_number_of_persons_l112_112893


namespace solution_set_inequality_range_of_t_l112_112207

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem solution_set_inequality :
  {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 2} :=
sorry

theorem range_of_t (t : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f (x - t) ≤ x - 2) ↔ 3 ≤ t ∧ t ≤ 3 + Real.sqrt 2 :=
sorry

end solution_set_inequality_range_of_t_l112_112207


namespace find_m_independent_quadratic_term_l112_112345

def quadratic_poly (m : ℝ) (x : ℝ) : ℝ :=
  -3 * x^2 + m * x^2 - x + 3

theorem find_m_independent_quadratic_term (m : ℝ) :
  (∀ x, quadratic_poly m x = -x + 3) → m = 3 :=
by 
  sorry

end find_m_independent_quadratic_term_l112_112345


namespace remainder_of_2n_divided_by_11_l112_112539

theorem remainder_of_2n_divided_by_11
  (n k : ℤ)
  (h : n = 22 * k + 12) :
  (2 * n) % 11 = 2 :=
by
  -- This is where the proof would go
  sorry

end remainder_of_2n_divided_by_11_l112_112539


namespace product_of_ab_l112_112060

theorem product_of_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 13) : a * b = -6 :=
by
  sorry

end product_of_ab_l112_112060


namespace equal_piece_length_l112_112132

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l112_112132


namespace HephaestusCharges_l112_112815

variable (x : ℕ)

theorem HephaestusCharges :
  3 * x + 6 * (12 - x) = 54 -> x = 6 :=
by
  intros h
  sorry

end HephaestusCharges_l112_112815


namespace value_of_expression_l112_112856

theorem value_of_expression (n m : ℤ) (h : m = 2 * n^2 + n + 1) : 8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end value_of_expression_l112_112856


namespace solution_unique_s_l112_112829

theorem solution_unique_s (s : ℝ) (hs : ⌊s⌋ + s = 22.7) : s = 11.7 :=
sorry

end solution_unique_s_l112_112829


namespace vector_expression_eval_l112_112017

open Real

noncomputable def v1 : ℝ × ℝ := (3, -8)
noncomputable def v2 : ℝ × ℝ := (2, -4)
noncomputable def k : ℝ := 5

theorem vector_expression_eval : (v1.1 - k * v2.1, v1.2 - k * v2.2) = (-7, 12) :=
  by sorry

end vector_expression_eval_l112_112017


namespace total_mass_grain_l112_112268

-- Given: the mass of the grain is 0.5 tons, and this constitutes 0.2 of the total mass
theorem total_mass_grain (m : ℝ) (h : 0.2 * m = 0.5) : m = 2.5 :=
by {
    -- Proof steps would go here
    sorry
}

end total_mass_grain_l112_112268


namespace number_of_moles_of_OC_NH2_2_formed_l112_112022

-- Definition: Chemical reaction condition
def reaction_eqn (x y : ℕ) : Prop := 
  x ≥ 1 ∧ y ≥ 2 ∧ x * 2 = y

-- Theorem: Prove that combining 3 moles of CO2 and 6 moles of NH3 results in 3 moles of OC(NH2)2
theorem number_of_moles_of_OC_NH2_2_formed (x y : ℕ) 
(h₁ : reaction_eqn x y)
(h₂ : x = 3)
(h₃ : y = 6) : 
x =  y / 2 :=
by {
    -- Proof is not provided
    sorry 
}

end number_of_moles_of_OC_NH2_2_formed_l112_112022


namespace Julio_fish_catch_rate_l112_112873

theorem Julio_fish_catch_rate (F : ℕ) : 
  (9 * F) - 15 = 48 → F = 7 :=
by
  intro h1
  --- proof
  sorry

end Julio_fish_catch_rate_l112_112873


namespace lcm_problem_l112_112836

theorem lcm_problem :
  ∃ k_values : Finset ℕ, (∀ k ∈ k_values, (60^10 : ℕ) = Nat.lcm (Nat.lcm (10^10) (12^12)) k) ∧ k_values.card = 121 :=
by
  sorry

end lcm_problem_l112_112836


namespace suitable_survey_l112_112419

def survey_suitable_for_census (A B C D : Prop) : Prop :=
  A ∧ ¬B ∧ ¬C ∧ ¬D

theorem suitable_survey {A B C D : Prop} (h_A : A) (h_B : ¬B) (h_C : ¬C) (h_D : ¬D) : survey_suitable_for_census A B C D :=
by
  unfold survey_suitable_for_census
  exact ⟨h_A, h_B, h_C, h_D⟩

end suitable_survey_l112_112419


namespace smallest_row_sum_greater_than_50_l112_112190

noncomputable def sum_interior_pascal (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem smallest_row_sum_greater_than_50 : ∃ n, sum_interior_pascal n > 50 ∧ (∀ m, m < n → sum_interior_pascal m ≤ 50) ∧ sum_interior_pascal 7 = 62 ∧ (sum_interior_pascal 7) % 2 = 0 :=
by
  sorry

end smallest_row_sum_greater_than_50_l112_112190


namespace unique_solution_l112_112755

theorem unique_solution (x y z : ℕ) (h_x : x > 1) (h_y : y > 1) (h_z : z > 1) :
  (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end unique_solution_l112_112755


namespace min_value_of_z_ineq_l112_112585

noncomputable def z (x y : ℝ) : ℝ := 2 * x + 4 * y

theorem min_value_of_z_ineq (k : ℝ) :
  (∃ x y : ℝ, (3 * x + y ≥ 0) ∧ (4 * x + 3 * y ≥ k) ∧ (z x y = -6)) ↔ k = 0 :=
by
  sorry

end min_value_of_z_ineq_l112_112585


namespace find_a_value_l112_112031

-- Problem statement
theorem find_a_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 + a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^(10)) → a = 3 :=
by sorry

end find_a_value_l112_112031


namespace combined_cost_of_apples_and_strawberries_l112_112283

theorem combined_cost_of_apples_and_strawberries :
  let cost_of_apples := 15
  let cost_of_strawberries := 26
  cost_of_apples + cost_of_strawberries = 41 :=
by
  sorry

end combined_cost_of_apples_and_strawberries_l112_112283


namespace shortest_path_length_l112_112909

theorem shortest_path_length (x y z : ℕ) (h1 : x + y = z + 1) (h2 : x + z = y + 5) (h3 : y + z = x + 7) : 
  min (min x y) z = 3 :=
by sorry

end shortest_path_length_l112_112909


namespace problem1_problem2_l112_112047

-- The first problem
theorem problem1 (x : ℝ) (h : Real.tan x = 3) :
  (2 * Real.sin (Real.pi - x) + 3 * Real.cos (-x)) /
  (Real.sin (x + Real.pi / 2) - Real.sin (x + Real.pi)) = 9 / 4 :=
by
  sorry

-- The second problem
theorem problem2 (x : ℝ) (h : Real.tan x = 3) :
  2 * Real.sin x ^ 2 - Real.sin (2 * x) + Real.cos x ^ 2 = 13 / 10 :=
by
  sorry

end problem1_problem2_l112_112047


namespace probe_distance_before_refuel_l112_112805

def total_distance : ℕ := 5555555555555
def distance_from_refuel : ℕ := 3333333333333
def distance_before_refuel : ℕ := 2222222222222

theorem probe_distance_before_refuel :
  total_distance - distance_from_refuel = distance_before_refuel := by
  sorry

end probe_distance_before_refuel_l112_112805


namespace hexagon_angle_R_l112_112367

theorem hexagon_angle_R (F I G U R E : ℝ) 
  (h1 : F = I ∧ I = R ∧ R = E)
  (h2 : G + U = 180) 
  (sum_angles_hexagon : F + I + G + U + R + E = 720) : 
  R = 135 :=
by sorry

end hexagon_angle_R_l112_112367


namespace largest_mersenne_prime_lt_500_l112_112575

open Nat

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n, (prime n) ∧ (p = 2^n - 1)

theorem largest_mersenne_prime_lt_500 : ∀ p, is_mersenne_prime p ∧ p < 500 → p ≤ 127 :=
by
  sorry

end largest_mersenne_prime_lt_500_l112_112575


namespace kernel_is_subgroup_l112_112373

variables {G H : Type*} [Group G] [Group H]
variable (ϕ : G →* H)

theorem kernel_is_subgroup : (ϕ.ker).subgroup G :=
by sorry

end kernel_is_subgroup_l112_112373


namespace count_k_values_l112_112837

-- Definitions based on the conditions
def k_satisfies_conditions (k : ℕ) : Prop :=
  let ⟨a, b, c⟩ := (nat.factorization k)
  a ≤ 20 ∧ b ≤ 10 ∧ c = 10

def count_satisfying_k : ℕ :=
  (21 * 11)

-- Main theorem statement
theorem count_k_values : count_satisfying_k = 231 :=
by
  -- Placeholder for the proof
  sorry

end count_k_values_l112_112837


namespace index_commutator_subgroup_is_even_l112_112992

open Finite FiniteGroup Subgroup

variable (G : Type*) [Group G] [Fintype G] [DecidableEq G]

/-- Let G' be the commutator subgroup of G. Assume |G'| = 2. 
    Prove that the index |G : G'| is even. -/
theorem index_commutator_subgroup_is_even (G' : Subgroup G) [IsNormalSubgroup G'] (hG' : Fintype.card G' = 2) :
  Even (Fintype.card G / Fintype.card G') :=
sorry

end index_commutator_subgroup_is_even_l112_112992


namespace integer_solutions_of_polynomial_l112_112700

theorem integer_solutions_of_polynomial :
  ∀ n : ℤ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = -1 ∨ n = 3 := 
by 
  sorry

end integer_solutions_of_polynomial_l112_112700


namespace proof_problem_l112_112294

noncomputable def problem_statement (x : ℝ) : Prop :=
  9.280 * real.log (x) / real.log (7) - real.log (7) / real.log (3) * real.log (x) / real.log (3) > real.log (0.25) / real.log (2)

theorem proof_problem (x : ℝ) : problem_statement x → 0 < x ∧ x < real.exp (real.log 3 * (2 / (real.log 3 / real.log 7 - real.log 7 / real.log 3))) :=
by
  sorry

end proof_problem_l112_112294


namespace direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l112_112717

-- Direct Proportional Function
theorem direct_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 1) → m = 1 :=
by 
  sorry

-- Inverse Proportional Function
theorem inverse_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = -1) → m = -1 :=
by 
  sorry

-- Quadratic Function
theorem quadratic_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 2) → (m = (-1 + Real.sqrt 13) / 2 ∨ m = (-1 - Real.sqrt 13) / 2) :=
by 
  sorry

-- Power Function
theorem power_function (m : ℝ) :
  (m^2 + 2 * m = 1) → (m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2) :=
by 
  sorry

end direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l112_112717


namespace jeanne_additional_tickets_l112_112999

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end jeanne_additional_tickets_l112_112999


namespace water_bottles_needed_l112_112962

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l112_112962


namespace sum_tens_ones_digits_3_plus_4_power_17_l112_112917

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l112_112917


namespace heartbeats_during_race_l112_112159

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l112_112159


namespace area_of_shaded_region_l112_112306

def side_length_of_square : ℝ := 12
def radius_of_quarter_circle : ℝ := 6

theorem area_of_shaded_region :
  let area_square := side_length_of_square ^ 2
  let area_full_circle := π * radius_of_quarter_circle ^ 2
  (area_square - area_full_circle) = 144 - 36 * π :=
by
  sorry

end area_of_shaded_region_l112_112306


namespace yogurt_packs_ordered_l112_112819

theorem yogurt_packs_ordered (P : ℕ) (price_per_pack refund_amount : ℕ) (expired_percentage : ℚ)
  (h1 : price_per_pack = 12)
  (h2 : refund_amount = 384)
  (h3 : expired_percentage = 0.40)
  (h4 : refund_amount / price_per_pack = 32)
  (h5 : 32 / expired_percentage = P) :
  P = 80 :=
sorry

end yogurt_packs_ordered_l112_112819


namespace draws_to_exceeding_sum_probability_l112_112933

open ProbabilityTheory

noncomputable def chips := {1, 2, 3, 4, 5}

theorem draws_to_exceeding_sum_probability : 
  ∀ draws : list ℕ, (draws.nodup ∧ ∀ x ∈ draws, x ∈ chips ∧ draws.length = 3) →
  (draws.sum > 4) →
  (∃ n : ℕ, n = 3 ∧ (Probability (draws.length = 3 ∧ draws.sum > 4 | sum_of_values <= 4) = 1 / 5)) :=
sorry

end draws_to_exceeding_sum_probability_l112_112933


namespace rabbit_fraction_l112_112981

theorem rabbit_fraction
  (initial_rabbits : ℕ) (added_rabbits : ℕ) (total_rabbits_seen : ℕ)
  (h_initial : initial_rabbits = 13)
  (h_added : added_rabbits = 7)
  (h_seen : total_rabbits_seen = 60) :
  (initial_rabbits + added_rabbits) / total_rabbits_seen = 1 / 3 :=
by
  -- we will prove this
  sorry

end rabbit_fraction_l112_112981


namespace sqrt_four_eq_two_or_neg_two_l112_112273

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l112_112273


namespace range_b_values_l112_112051

theorem range_b_values (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : ∀ x, f x = Real.exp x - 1) 
  (hg : ∀ x, g x = -x^2 + 4*x - 3) 
  (h : f a = g b) : 
  b ∈ Set.univ :=
by sorry

end range_b_values_l112_112051


namespace imaginary_unit_sum_l112_112969

theorem imaginary_unit_sum (i : ℂ) (H : i^4 = 1) : i^1234 + i^1235 + i^1236 + i^1237 = 0 :=
by
  sorry

end imaginary_unit_sum_l112_112969


namespace fraction_first_to_second_l112_112403

def digit_fraction_proof_problem (a b c d : ℕ) (number : ℕ) :=
  number = 1349 ∧
  a = b / 3 ∧
  c = a + b ∧
  d = 3 * b

theorem fraction_first_to_second (a b c d : ℕ) (number : ℕ) :
  digit_fraction_proof_problem a b c d number → a / b = 1 / 3 :=
by
  intro problem
  sorry

end fraction_first_to_second_l112_112403


namespace contrapositive_proposition_l112_112099

theorem contrapositive_proposition :
  (∀ x : ℝ, (x^2 < 4 → -2 < x ∧ x < 2)) ↔ (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4)) :=
by
  sorry

end contrapositive_proposition_l112_112099


namespace border_area_correct_l112_112940

-- Define the dimensions of the photograph
def photograph_height : ℕ := 12
def photograph_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the photograph
def area_photograph : ℕ := photograph_height * photograph_width

-- Define the total dimensions including the frame
def total_height : ℕ := photograph_height + 2 * border_width
def total_width : ℕ := photograph_width + 2 * border_width

-- Define the area of the framed area
def area_framed : ℕ := total_height * total_width

-- Define the area of the border
def area_border : ℕ := area_framed - area_photograph

theorem border_area_correct : area_border = 198 := by
  sorry

end border_area_correct_l112_112940


namespace find_A_l112_112811

theorem find_A (A B : ℝ) 
  (h1 : A - 3 * B = 303.1)
  (h2 : 10 * B = A) : 
  A = 433 :=
by
  sorry

end find_A_l112_112811


namespace athlete_heartbeats_l112_112154

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l112_112154


namespace solve_abs_inequality_l112_112272

theorem solve_abs_inequality (x : ℝ) :
  (|x-2| ≥ |x|) → x ≤ 1 :=
by
  sorry

end solve_abs_inequality_l112_112272


namespace Randy_bats_l112_112882

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end Randy_bats_l112_112882


namespace frac_add_eq_seven_halves_l112_112584

theorem frac_add_eq_seven_halves {x y : ℝ} (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 :=
by
  sorry

end frac_add_eq_seven_halves_l112_112584


namespace integers_within_range_l112_112766

def is_within_range (n : ℤ) : Prop :=
  (-1.3 : ℝ) < (n : ℝ) ∧ (n : ℝ) < 2.8

theorem integers_within_range :
  { n : ℤ | is_within_range n } = {-1, 0, 1, 2} :=
by
  sorry

end integers_within_range_l112_112766


namespace roots_expression_value_l112_112048

theorem roots_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : x1 * x2 = 2) :
  2 * x1 - x1 * x2 + 2 * x2 = 8 :=
by
  sorry

end roots_expression_value_l112_112048


namespace students_not_enrolled_l112_112794

-- Declare the conditions
def total_students : Nat := 79
def students_french : Nat := 41
def students_german : Nat := 22
def students_both : Nat := 9

-- Define the problem statement
theorem students_not_enrolled : total_students - (students_french + students_german - students_both) = 25 := by
  sorry

end students_not_enrolled_l112_112794


namespace sqrt_eq_two_or_neg_two_l112_112277

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l112_112277


namespace total_arrangements_is_42_l112_112556

theorem total_arrangements_is_42 :  -- Define the theorem
  let departments := 3,
      people_per_department := 2,
      returning_people := 2
  in
  (∃ (same_department_cases arrangements diff_department_cases arrangements_total: ℕ),
      (same_department_cases = Nat.choose departments 1 * Nat.perm people_per_department people_per_department) ∧
      (diff_department_cases = Nat.choose departments 2 * people_per_department * people_per_department * 3) ∧
      (arrangements_total = same_department_cases + diff_department_cases))
  → arrangements_total = 42 :=  -- Prove the total number of different arrangements is 42
by
  sorry

end total_arrangements_is_42_l112_112556


namespace water_bottles_needed_l112_112960

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l112_112960


namespace three_digit_numbers_l112_112020

theorem three_digit_numbers (a b c n : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
    (h5 : 0 ≤ c) (h6 : c ≤ 9) (h7 : n = 100 * a + 10 * b + c) (h8 : 10 * b + c = (100 * a + 10 * b + c) / 5) :
    n = 125 ∨ n = 250 ∨ n = 375 := 
by 
  sorry

end three_digit_numbers_l112_112020


namespace curve_to_standard_form_chord_line_equation_l112_112888

theorem curve_to_standard_form (k : ℝ) : 
  let x := 8 * k / (1 + k^2),
      y := 2 * (1 - k^2) / (1 + k^2)
  in (x^2 / 16 + y^2 / 4 = 1) :=
sorry

theorem chord_line_equation (t θ : ℝ) (A B P : ℝ × ℝ) :
  let x := 2 + t * Real.cos θ,
      y := 1 + t * Real.sin θ,
      mid := (2, 1)
  in P = mid → 
     A = (2 + t * Real.cos θ, 1 + t * Real.sin θ) →
     B = (2 - t * Real.cos θ, 1 - t * Real.sin θ) →
     P = ((A.1 + B.1)/2, (A.2 + B.2)/2) →
     (∀ t θ, x + 2 * y - 4 = 0) :=
sorry

end curve_to_standard_form_chord_line_equation_l112_112888


namespace consecutive_integers_sum_l112_112106

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end consecutive_integers_sum_l112_112106


namespace arithmetic_sequence_sum_l112_112067

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l112_112067


namespace find_sixth_term_l112_112039

open Nat

-- Given conditions
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_three_terms (a : ℕ → ℤ) : Prop :=
  a 1 + a 2 + a 3 = 168

def second_minus_fifth (a : ℕ → ℤ) : Prop :=
  a 2 - a 5 = 42

-- Prove question == answer given conditions
theorem find_sixth_term :
  ∀ (a : ℕ → ℤ), arithmetic_progression a → sum_of_first_three_terms a → second_minus_fifth a → a 6 = 0 :=
by
  sorry

end find_sixth_term_l112_112039


namespace aria_spent_on_cookies_l112_112191

def aria_spent : ℕ := 2356

theorem aria_spent_on_cookies :
  (let cookies_per_day := 4
  let cost_per_cookie := 19
  let days_in_march := 31
  let total_cookies := days_in_march * cookies_per_day
  let total_cost := total_cookies * cost_per_cookie
  total_cost = aria_spent) :=
  sorry

end aria_spent_on_cookies_l112_112191


namespace gcd_of_three_numbers_l112_112519

theorem gcd_of_three_numbers (a b c : ℕ) (h1: a = 4557) (h2: b = 1953) (h3: c = 5115) : 
    Nat.gcd a (Nat.gcd b c) = 93 :=
by
  rw [h1, h2, h3]
  -- Proof goes here
  sorry

end gcd_of_three_numbers_l112_112519


namespace at_most_one_cube_l112_112108

theorem at_most_one_cube (a : ℕ → ℕ) (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2018) :
  ∃! n, ∃ m : ℕ, a n = m ^ 3 := sorry

end at_most_one_cube_l112_112108


namespace inequality_conditions_l112_112590

theorem inequality_conditions (x y z : ℝ) (h1 : y - x < 1.5 * abs x) (h2 : z = 2 * (y + x)) : 
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) :=
by
  sorry

end inequality_conditions_l112_112590


namespace a_equals_bc_l112_112454

theorem a_equals_bc (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x y : ℝ, f x * g y = a * x * y + b * x + c * y + 1) → a = b * c :=
sorry

end a_equals_bc_l112_112454


namespace roots_quadratic_l112_112849

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end roots_quadratic_l112_112849


namespace suitcase_weight_on_return_l112_112491

def initial_weight : ℝ := 5
def perfume_count : ℝ := 5
def perfume_weight_oz : ℝ := 1.2
def chocolate_weight_lb : ℝ := 4
def soap_count : ℝ := 2
def soap_weight_oz : ℝ := 5
def jam_count : ℝ := 2
def jam_weight_oz : ℝ := 8
def oz_per_lb : ℝ := 16

theorem suitcase_weight_on_return :
  initial_weight + (perfume_count * perfume_weight_oz / oz_per_lb) + chocolate_weight_lb +
  (soap_count * soap_weight_oz / oz_per_lb) + (jam_count * jam_weight_oz / oz_per_lb) = 11 := 
  by
  sorry

end suitcase_weight_on_return_l112_112491


namespace sqrt_eq_two_or_neg_two_l112_112278

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l112_112278


namespace sin_sum_alpha_pi_over_3_l112_112845

theorem sin_sum_alpha_pi_over_3 (alpha : ℝ) (h1 : Real.cos (alpha + 2/3 * Real.pi) = 4/5) (h2 : -Real.pi/2 < alpha ∧ alpha < 0) :
  Real.sin (alpha + Real.pi/3) + Real.sin alpha = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_sum_alpha_pi_over_3_l112_112845


namespace youseff_time_difference_l112_112420

noncomputable def walking_time (blocks : ℕ) (time_per_block : ℕ) : ℕ := blocks * time_per_block
noncomputable def biking_time (blocks : ℕ) (time_per_block_seconds : ℕ) : ℕ := (blocks * time_per_block_seconds) / 60

theorem youseff_time_difference : walking_time 6 1 - biking_time 6 20 = 4 := by
  sorry

end youseff_time_difference_l112_112420


namespace shortest_side_length_l112_112738

theorem shortest_side_length (A B C : ℝ) (a b c : ℝ)
  (h_sinA : Real.sin A = 5 / 13)
  (h_cosB : Real.cos B = 3 / 5)
  (h_longest : c = 63)
  (h_angles : A < B ∧ C = π - (A + B)) :
  a = 25 := by
sorry

end shortest_side_length_l112_112738


namespace wrongly_noted_mark_l112_112763

theorem wrongly_noted_mark (n : ℕ) (avg_wrong avg_correct correct_mark : ℝ) (x : ℝ)
  (h1 : n = 30)
  (h2 : avg_wrong = 60)
  (h3 : avg_correct = 57.5)
  (h4 : correct_mark = 15)
  (h5 : n * avg_wrong - n * avg_correct = x - correct_mark)
  : x = 90 :=
sorry

end wrongly_noted_mark_l112_112763


namespace age_ratio_in_two_years_l112_112675

-- Definitions of conditions
def son_present_age : ℕ := 26
def age_difference : ℕ := 28
def man_present_age : ℕ := son_present_age + age_difference

-- Future ages after 2 years
def son_future_age : ℕ := son_present_age + 2
def man_future_age : ℕ := man_present_age + 2

-- The theorem to prove
theorem age_ratio_in_two_years : (man_future_age / son_future_age) = 2 := 
by
  -- Step-by-Step proof would go here
  sorry

end age_ratio_in_two_years_l112_112675


namespace continuous_stripe_probability_l112_112952

noncomputable def probability_continuous_stripe : ℚ :=
let total_combinations := 2^6 in
-- 3 pairs of parallel faces; for each pair, 4 favorable configurations.
let favorable_outcomes := 3 * 4 in
favorable_outcomes / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l112_112952


namespace reciprocal_of_complex_power_l112_112579

noncomputable def complex_num_reciprocal : ℂ :=
  (Complex.I) ^ 2023

theorem reciprocal_of_complex_power :
  ∀ z : ℂ, z = (Complex.I) ^ 2023 -> (1 / z) = Complex.I :=
by
  intro z
  intro hz
  have h_power : z = Complex.I ^ 2023 := by assumption
  sorry

end reciprocal_of_complex_power_l112_112579


namespace range_of_a_l112_112052

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a < x ∧ x < a + 1) → (-2 ≤ x ∧ x ≤ 2)) ↔ -2 ≤ a ∧ a ≤ 1 :=
by 
  sorry

end range_of_a_l112_112052


namespace problem1_problem2_problem3_l112_112005

-- Problem 1
theorem problem1 :
  1 - 1^2022 + ((-1/2)^2) * (-2)^3 * (-2)^2 - |Real.pi - 3.14|^0 = -10 :=
by sorry

-- Problem 2
variables (a b : ℝ)

theorem problem2 :
  a^3 * (-b^3)^2 + (-2 * a * b)^3 = a^3 * b^6 - 8 * a^3 * b^3 :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) :
  (2 * a^3 * b^2 - 3 * a^2 * b - 4 * a) * 2 * b = 4 * a^3 * b^3 - 6 * a^2 * b^2 - 8 * a * b :=
by sorry

end problem1_problem2_problem3_l112_112005


namespace roots_quadratic_l112_112848

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end roots_quadratic_l112_112848


namespace find_value_of_a_b_ab_l112_112705

variable (a b : ℝ)

theorem find_value_of_a_b_ab
  (h1 : 2 * a + 2 * b + a * b = 1)
  (h2 : a + b + 3 * a * b = -2) :
  a + b + a * b = 0 := 
sorry

end find_value_of_a_b_ab_l112_112705


namespace problem_solution_l112_112058

theorem problem_solution :
  2 ^ 2000 - 3 * 2 ^ 1999 + 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 = -5 * 2 ^ 1996 :=
by  -- initiate the proof script
  sorry  -- means "proof is omitted"

end problem_solution_l112_112058


namespace grandfather_age_l112_112780

variables (M G y z : ℕ)

-- Conditions
def condition1 : Prop := G = 6 * M
def condition2 : Prop := G + y = 5 * (M + y)
def condition3 : Prop := G + y + z = 4 * (M + y + z)

-- Theorem to prove Grandfather's current age is 72
theorem grandfather_age : 
  condition1 M G → 
  condition2 M G y → 
  condition3 M G y z → 
  G = 72 :=
by
  intros h1 h2 h3
  unfold condition1 at h1
  unfold condition2 at h2
  unfold condition3 at h3
  sorry

end grandfather_age_l112_112780


namespace trigonometric_identity_l112_112753

theorem trigonometric_identity (α : ℝ) :
    (1 / Real.sin (-α) - Real.sin (Real.pi + α)) /
    (1 / Real.cos (3 * Real.pi - α) + Real.cos (2 * Real.pi - α)) =
    1 / Real.tan α ^ 3 :=
    sorry

end trigonometric_identity_l112_112753


namespace days_from_thursday_l112_112115

theorem days_from_thursday (n : ℕ) (h : n = 53) : 
  (n % 7 = 4) ∧ (n % 7 = 4 → "Thursday" + 4 days = "Monday") :=
by 
  have h1 : n % 7 = 4 := by sorry
  have h2 : "Thursday" + 4 days = "Monday" := by sorry
  exact ⟨h1, h2 h1⟩

end days_from_thursday_l112_112115


namespace find_B_and_distance_l112_112498

noncomputable def pointA : ℝ × ℝ := (2, 4)

noncomputable def pointB : ℝ × ℝ := (-(1 + Real.sqrt 385) / 8, (-(1 + Real.sqrt 385) / 8) ^ 2)

noncomputable def distanceToOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem find_B_and_distance :
  (pointA.snd = pointA.fst ^ 2) ∧
  (pointB.snd = (-(1 + Real.sqrt 385) / 8) ^ 2) ∧
  (distanceToOrigin pointB = Real.sqrt ((-(1 + Real.sqrt 385) / 8) ^ 2 + (-(1 + Real.sqrt 385) / 8) ^ 4)) :=
  sorry

end find_B_and_distance_l112_112498


namespace range_of_x_l112_112565

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l112_112565


namespace rider_distance_traveled_l112_112798

noncomputable def caravan_speed := 1  -- km/h
noncomputable def rider_speed := 1 + Real.sqrt 2  -- km/h

theorem rider_distance_traveled : 
  (1 / (rider_speed - 1) + 1 / (rider_speed + 1)) = 1 :=
by
  sorry

end rider_distance_traveled_l112_112798


namespace mike_remaining_cards_l112_112380

def initial_cards (mike_cards : ℕ) : ℕ := 87
def sam_cards (sam_bought : ℕ) : ℕ := 13
def alex_cards (alex_bought : ℕ) : ℕ := 15

theorem mike_remaining_cards (mike_cards sam_bought alex_bought : ℕ) :
  mike_cards - (sam_bought + alex_bought) = 59 :=
by
  let mike_cards := initial_cards 87
  let sam_cards := sam_bought
  let alex_cards := alex_bought
  sorry

end mike_remaining_cards_l112_112380


namespace vector_subtraction_l112_112949

-- Lean definitions for the problem conditions
def v₁ : ℝ × ℝ := (3, -5)
def v₂ : ℝ × ℝ := (-2, 6)
def s₁ : ℝ := 4
def s₂ : ℝ := 3

-- The theorem statement
theorem vector_subtraction :
  s₁ • v₁ - s₂ • v₂ = (18, -38) :=
by
  sorry

end vector_subtraction_l112_112949


namespace area_of_triangle_DEF_l112_112591

-- Define point D
def pointD : ℝ × ℝ := (2, 5)

-- Reflect D over the y-axis to get E
def reflectY (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, P.2)
def pointE : ℝ × ℝ := reflectY pointD

-- Reflect E over the line y = -x to get F
def reflectYX (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, -P.1)
def pointF : ℝ × ℝ := reflectYX pointE

-- Define function to calculate the area of the triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Define the Lean 4 statement
theorem area_of_triangle_DEF : triangle_area pointD pointE pointF = 6 := by
  sorry

end area_of_triangle_DEF_l112_112591


namespace work_completion_days_l112_112791

theorem work_completion_days (a b : Type) (T : ℕ) (ha : T = 12) (hb : T = 6) : 
  (T = 4) :=
sorry

end work_completion_days_l112_112791


namespace sqrt_of_4_l112_112281

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l112_112281


namespace point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l112_112466

theorem point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb
  (x1 x2 : ℝ) : 
  (x1 * x2 / 4 = -1) ↔ ((x1 / 2) * (x2 / 2) = -1) :=
by sorry

end point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l112_112466


namespace func_eq_condition_l112_112339

variable (a : ℝ)

theorem func_eq_condition (f : ℝ → ℝ) :
  (∀ x : ℝ, f (Real.sin x) + a * f (Real.cos x) = Real.cos (2 * x)) ↔ a ∈ (Set.univ \ {1} : Set ℝ) :=
by
  sorry

end func_eq_condition_l112_112339


namespace factorize_expr_l112_112696

def my_expr (a b : ℤ) : ℤ := 4 * a^2 * b - b

theorem factorize_expr (a b : ℤ) : my_expr a b = b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end factorize_expr_l112_112696


namespace parabola_focus_distance_l112_112197

open Real

noncomputable def parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = 4 * P.1
def line_eq (P : ℝ × ℝ) : Prop := abs (P.1 + 2) = 6

theorem parabola_focus_distance (P : ℝ × ℝ) 
  (hp : parabola P) 
  (hl : line_eq P) : 
  dist P (1 / 4, 0) = 5 :=
sorry

end parabola_focus_distance_l112_112197


namespace sufficient_but_not_necessary_l112_112542

theorem sufficient_but_not_necessary (x : ℝ) : 
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l112_112542


namespace inequality_inequality_must_be_true_l112_112710

variables {a b c d : ℝ}

theorem inequality_inequality_must_be_true
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (a / d) < (b / c) :=
sorry

end inequality_inequality_must_be_true_l112_112710


namespace find_parallel_line_l112_112341

-- Definition of the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Definition of the original line equation
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Definition of the desired line equation
def desired_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement: defining the desired line based on the point and parallelism condition
theorem find_parallel_line (x y : ℝ) (hx : point.fst = 0) (hy : point.snd = 1) :
  ∃ m : ℝ, (2 * x + y + m = 0) ∧ (2 * 0 + 1 + m = 0) → desired_line x y :=
sorry

end find_parallel_line_l112_112341


namespace sum_of_numbers_l112_112119

-- Definitions for the numbers involved
def n1 : Nat := 1235
def n2 : Nat := 2351
def n3 : Nat := 3512
def n4 : Nat := 5123

-- Proof statement
theorem sum_of_numbers :
  n1 + n2 + n3 + n4 = 12221 := by
  sorry

end sum_of_numbers_l112_112119


namespace simplify_expr1_simplify_expr2_simplify_expr3_l112_112407

theorem simplify_expr1 : -2.48 + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem simplify_expr2 : (7/13) * (-9) + (7/13) * (-18) + (7/13) = -14 := by
  sorry

theorem simplify_expr3 : -((20 + 1/19) * 38) = -762 := by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l112_112407


namespace initial_strawberries_l112_112506

-- Define the conditions
def strawberries_eaten : ℝ := 42.0
def strawberries_left : ℝ := 36.0

-- State the theorem
theorem initial_strawberries :
  strawberries_eaten + strawberries_left = 78 :=
by
  sorry

end initial_strawberries_l112_112506


namespace determine_constants_l112_112192

theorem determine_constants :
  ∃ (a b c p : ℝ), (a = -1) ∧ (b = -1) ∧ (c = -1) ∧ (p = 3) ∧
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
  c - b = b - a ∧ c - b > 0 :=
by
  sorry

end determine_constants_l112_112192


namespace find_ordered_pair_l112_112326

-- Definitions based on the conditions
variable (a c : ℝ)
def has_exactly_one_solution :=
  (-6)^2 - 4 * a * c = 0

def sum_is_twelve :=
  a + c = 12

def a_less_than_c :=
  a < c

-- The proof statement
theorem find_ordered_pair
  (h₁ : has_exactly_one_solution a c)
  (h₂ : sum_is_twelve a c)
  (h₃ : a_less_than_c a c) :
  a = 3 ∧ c = 9 := 
sorry

end find_ordered_pair_l112_112326


namespace elizabeth_stickers_l112_112180

def initial_bottles := 10
def lost_at_school := 2
def lost_at_practice := 1
def stickers_per_bottle := 3

def total_remaining_bottles := initial_bottles - lost_at_school - lost_at_practice
def total_stickers := total_remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers : total_stickers = 21 :=
  by
  unfold total_stickers total_remaining_bottles initial_bottles lost_at_school lost_at_practice stickers_per_bottle
  simp
  sorry

end elizabeth_stickers_l112_112180


namespace compound_interest_example_l112_112558

theorem compound_interest_example :
  let P := 5000
  let r := 0.08
  let n := 4
  let t := 0.5
  let A := P * (1 + r / n) ^ (n * t)
  A = 5202 :=
by
  sorry

end compound_interest_example_l112_112558


namespace intersection_of_M_and_N_l112_112054

-- Define sets M and N
def M := {x : ℝ | (x + 2) * (x - 1) < 0}
def N := {x : ℝ | x + 1 < 0}

-- State the theorem for the intersection M ∩ N
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < -1} :=
sorry

end intersection_of_M_and_N_l112_112054


namespace Randy_bats_l112_112881

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end Randy_bats_l112_112881


namespace find_m_l112_112247

-- Define the function with given conditions
def f (m : ℕ) (n : ℕ) : ℕ := 
if n > m^2 then n - m + 14 else sorry

-- Define the main problem
theorem find_m (m : ℕ) (hyp : m ≥ 14) : f m 1995 = 1995 ↔ m = 14 ∨ m = 45 :=
by
  sorry

end find_m_l112_112247


namespace leap_year_1996_l112_112945

def divisible_by (n m : ℕ) : Prop := m % n = 0

def is_leap_year (y : ℕ) : Prop :=
  (divisible_by 4 y ∧ ¬divisible_by 100 y) ∨ divisible_by 400 y

theorem leap_year_1996 : is_leap_year 1996 :=
by
  sorry

end leap_year_1996_l112_112945


namespace quotient_of_larger_divided_by_smaller_l112_112394

theorem quotient_of_larger_divided_by_smaller
  (x y : ℕ)
  (h1 : x * y = 9375)
  (h2 : x + y = 400)
  (h3 : x > y) :
  x / y = 15 :=
sorry

end quotient_of_larger_divided_by_smaller_l112_112394


namespace assign_questions_to_students_l112_112401

theorem assign_questions_to_students:
  ∃ (assignment : Fin 20 → Fin 20), 
  (∀ s : Fin 20, ∃ q1 q2 : Fin 20, (assignment s = q1 ∨ assignment s = q2) ∧ q1 ≠ q2 ∧ ∀ q : Fin 20, ∃ s1 s2 : Fin 20, (assignment s1 = q ∧ assignment s2 = q) ∧ s1 ≠ s2) :=
by
  sorry

end assign_questions_to_students_l112_112401


namespace eta_expectation_and_variance_l112_112211

open Probability

def xi : PMF ℕ := PMF.binomial 5 0.5

def η : ℕ → ℝ := λ x => 5 * x

theorem eta_expectation_and_variance :
  (E (η <$> xi) = 25 / 2) ∧ (variance (η <$> xi) = 125 / 4) :=
by
  sorry

end eta_expectation_and_variance_l112_112211


namespace jacks_speed_l112_112872

-- Define the initial distance between Jack and Christina.
def initial_distance : ℝ := 360

-- Define Christina's speed.
def christina_speed : ℝ := 7

-- Define Lindy's speed.
def lindy_speed : ℝ := 12

-- Define the total distance Lindy travels.
def lindy_total_distance : ℝ := 360

-- Prove Jack's speed given the conditions.
theorem jacks_speed : ∃ v : ℝ, (initial_distance - christina_speed * (lindy_total_distance / lindy_speed)) / (lindy_total_distance / lindy_speed) = v ∧ v = 5 :=
by {
  sorry
}

end jacks_speed_l112_112872


namespace Diana_total_earnings_l112_112329

def July : ℝ := 150
def August : ℝ := 3 * July
def September : ℝ := 2 * August
def October : ℝ := September + 0.1 * September
def November : ℝ := 0.95 * October
def Total_earnings : ℝ := July + August + September + October + November

theorem Diana_total_earnings : Total_earnings = 3430.50 := by
  sorry

end Diana_total_earnings_l112_112329


namespace greater_than_neg4_1_l112_112790

theorem greater_than_neg4_1 (k : ℤ) (h1 : k = -4) : k > (-4.1 : ℝ) :=
by sorry

end greater_than_neg4_1_l112_112790


namespace part1_l112_112423

theorem part1 (a b : ℝ) : 3*(a - b)^2 - 6*(a - b)^2 + 2*(a - b)^2 = - (a - b)^2 :=
by
  sorry

end part1_l112_112423


namespace polynomial_remainder_l112_112703

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def rem (x : ℝ) : ℝ := 14 * x - 14

theorem polynomial_remainder :
  ∀ x : ℝ, p x % d x = rem x := 
by
  sorry

end polynomial_remainder_l112_112703


namespace centroid_coordinates_satisfy_l112_112650

noncomputable def P : ℝ × ℝ := (2, 5)
noncomputable def Q : ℝ × ℝ := (-1, 3)
noncomputable def R : ℝ × ℝ := (4, -2)

noncomputable def S : ℝ × ℝ := (
  (P.1 + Q.1 + R.1) / 3,
  (P.2 + Q.2 + R.2) / 3
)

theorem centroid_coordinates_satisfy :
  4 * S.1 + 3 * S.2 = 38 / 3 :=
by
  -- Proof will be added here
  sorry

end centroid_coordinates_satisfy_l112_112650


namespace tan_product_eq_four_l112_112903

-- Define the angles in degrees
def ang1 : ℝ := 17
def ang2 : ℝ := 18
def ang3 : ℝ := 27
def ang4 : ℝ := 28

-- Function to convert degrees to radians
def deg_to_rad (d : ℝ) : ℝ := d * (Real.pi / 180)

-- Define the tangents of the angles
def tan_ang1 : ℝ := Real.tan (deg_to_rad ang1)
def tan_ang2 : ℝ := Real.tan (deg_to_rad ang2)
def tan_ang3 : ℝ := Real.tan (deg_to_rad ang3)
def tan_ang4 : ℝ := Real.tan (deg_to_rad ang4)

-- Statement of the proof problem
theorem tan_product_eq_four : (1 + tan_ang1) * (1 + tan_ang2) * (1 + tan_ang3) * (1 + tan_ang4) = 4 :=
by sorry

end tan_product_eq_four_l112_112903


namespace big_SUV_wash_ratio_l112_112536

-- Defining constants for time taken for various parts of the car
def time_windows : ℕ := 4
def time_body : ℕ := 7
def time_tires : ℕ := 4
def time_waxing : ℕ := 9

-- Time taken to wash one normal car
def time_normal_car : ℕ := time_windows + time_body + time_tires + time_waxing

-- Given total time William spent washing all vehicles
def total_time : ℕ := 96

-- Time taken for two normal cars
def time_two_normal_cars : ℕ := 2 * time_normal_car

-- Time taken for the big SUV
def time_big_SUV : ℕ := total_time - time_two_normal_cars

-- Ratio of time taken to wash the big SUV to the time taken to wash a normal car
def time_ratio : ℕ := time_big_SUV / time_normal_car

theorem big_SUV_wash_ratio : time_ratio = 2 := by
  sorry

end big_SUV_wash_ratio_l112_112536


namespace vector_combination_l112_112719

-- Definitions of the given vectors and condition of parallelism
def vec_a : (ℝ × ℝ) := (1, -2)
def vec_b (m : ℝ) : (ℝ × ℝ) := (2, m)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0

-- Goal to prove
theorem vector_combination :
  ∀ m : ℝ, are_parallel vec_a (vec_b m) → 3 * vec_a.1 + 2 * (vec_b m).1 = 7 ∧ 3 * vec_a.2 + 2 * (vec_b m).2 = -14 :=
by
  intros m h_par
  sorry

end vector_combination_l112_112719


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l112_112919

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l112_112919


namespace bob_weight_l112_112400

variable (j b : ℕ)

theorem bob_weight :
  j + b = 210 →
  b - j = b / 3 →
  b = 126 :=
by
  intros h1 h2
  sorry

end bob_weight_l112_112400


namespace max_knights_among_10_l112_112825

def is_knight (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (p m ↔ (m ≥ n))

def is_liar (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (¬ p m ↔ (m ≥ n))

def greater_than (k : ℕ) (n : ℕ) := n > k

def less_than (k : ℕ) (n : ℕ) := n < k

def person_statement_1 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => greater_than 1 n
  | 2 => greater_than 2 n
  | 3 => greater_than 3 n
  | 4 => greater_than 4 n
  | 5 => greater_than 5 n
  | 6 => greater_than 6 n
  | 7 => greater_than 7 n
  | 8 => greater_than 8 n
  | 9 => greater_than 9 n
  | 10 => greater_than 10 n
  | _ => false

def person_statement_2 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => less_than 1 n
  | 2 => less_than 2 n
  | 3 => less_than 3 n
  | 4 => less_than 4 n
  | 5 => less_than 5 n
  | 6 => less_than 6 n
  | 7 => less_than 7 n
  | 8 => less_than 8 n
  | 9 => less_than 9 n
  | 10 => less_than 10 n
  | _ => false

theorem max_knights_among_10 (knights : ℕ) : 
  (∀ i < 10, (is_knight (person_statement_1 (i + 1)) (i + 1) ∨ is_liar (person_statement_1 (i + 1)) (i + 1))) ∧
  (∀ i < 10, (is_knight (person_statement_2 (i + 1)) (i + 1) ∨ is_liar (person_statement_2 (i + 1)) (i + 1))) →
  knights ≤ 8 := sorry

end max_knights_among_10_l112_112825


namespace fraction_meaningful_condition_l112_112783

-- Define a variable x
variable (x : ℝ)

-- State the condition that makes the fraction meaningful
def fraction_meaningful (x : ℝ) : Prop := (x - 2) ≠ 0

-- State the theorem we want to prove
theorem fraction_meaningful_condition : fraction_meaningful x ↔ x ≠ 2 := sorry

end fraction_meaningful_condition_l112_112783


namespace simplify_T_l112_112238

variable (x : ℝ)

theorem simplify_T :
  9 * (x + 2)^2 - 12 * (x + 2) + 4 = 4 * (1.5 * x + 2)^2 :=
by
  sorry

end simplify_T_l112_112238


namespace total_water_bottles_needed_l112_112954

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l112_112954


namespace find_sixth_term_l112_112038

open Nat

-- Given conditions
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_three_terms (a : ℕ → ℤ) : Prop :=
  a 1 + a 2 + a 3 = 168

def second_minus_fifth (a : ℕ → ℤ) : Prop :=
  a 2 - a 5 = 42

-- Prove question == answer given conditions
theorem find_sixth_term :
  ∀ (a : ℕ → ℤ), arithmetic_progression a → sum_of_first_three_terms a → second_minus_fifth a → a 6 = 0 :=
by
  sorry

end find_sixth_term_l112_112038


namespace breadth_is_13_l112_112515

variable (b l : ℕ) (breadth : ℕ)

/-
We have the following conditions:
1. The area of the rectangular plot is 23 times its breadth.
2. The difference between the length and the breadth is 10 metres.
We need to prove that the breadth of the plot is 13 metres.
-/

theorem breadth_is_13
  (h1 : l * b = 23 * b)
  (h2 : l - b = 10) :
  b = 13 := 
sorry

end breadth_is_13_l112_112515


namespace find_parallel_line_l112_112340

variables {x y : ℝ}

def line1 (x y : ℝ) : Prop := 2 * x + y - 3 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def point (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem find_parallel_line (hx : point 0 1) : line2 0 1 :=
by
  dsimp [line2, point] at *,
  sorry

end find_parallel_line_l112_112340


namespace necessary_condition_range_l112_112714

variables {x m : ℝ}

def p (x : ℝ) : Prop := x^2 - x - 2 < 0
def q (x m : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

theorem necessary_condition_range (H : ∀ x, q x m → p x) : -1 < m ∧ m < 1 :=
by {
  sorry
}

end necessary_condition_range_l112_112714


namespace inequality_proof_l112_112014

theorem inequality_proof (x y : ℝ) (h : |x - 2 * y| = 5) : x^2 + y^2 ≥ 5 := 
  sorry

end inequality_proof_l112_112014


namespace minimum_value_expression_l112_112374

theorem minimum_value_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 :=
by
  sorry

end minimum_value_expression_l112_112374


namespace sqrt_of_4_l112_112279

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l112_112279


namespace savings_after_increase_l112_112143

-- Conditions
def salary : ℕ := 5000
def initial_savings_ratio : ℚ := 0.20
def expense_increase_ratio : ℚ := 1.20

-- Derived initial values
def initial_savings : ℚ := initial_savings_ratio * salary
def initial_expenses : ℚ := ((1 : ℚ) - initial_savings_ratio) * salary

-- New expenses after increase
def new_expenses : ℚ := expense_increase_ratio * initial_expenses

-- Savings after expense increase
def final_savings : ℚ := salary - new_expenses

theorem savings_after_increase : final_savings = 200 := by
  sorry

end savings_after_increase_l112_112143


namespace probability_no_aces_opposite_l112_112455

open Nat

-- Define the conditions
def players : ℕ := 4
def total_cards : ℕ := 32
def cards_per_player : ℕ := 32 / 4 -- 8 cards per player

-- Define the events
def event_A := choose 24 8 -- one player receives 8 of the 24 non-ace cards
def event_B := choose 20 8 -- another specified player receives 8 of the 20 remaining non-ace cards

-- Define the probability calculation
def conditional_probability := event_B.toRational / event_A.toRational

-- Final theorem stating that the conditional probability is equal to 130 / 759
theorem probability_no_aces_opposite : conditional_probability = 130 / 759 := sorry

end probability_no_aces_opposite_l112_112455


namespace garden_area_enlargement_l112_112551

theorem garden_area_enlargement :
  let length := 60
  let width := 20
  (2 * (length + width)) = 160 →
  (160 / 4) = 40 →
  ((40 * 40) - (length * width) = 400) :=
begin
  intros,
  sorry,
end

end garden_area_enlargement_l112_112551


namespace taxes_paid_l112_112729

theorem taxes_paid (gross_pay net_pay : ℤ) (h1 : gross_pay = 450) (h2 : net_pay = 315) :
  gross_pay - net_pay = 135 := 
by 
  rw [h1, h2] 
  norm_num

end taxes_paid_l112_112729


namespace simplify_and_multiply_l112_112509

theorem simplify_and_multiply :
  let a := 3
  let b := 17
  let d1 := 504
  let d2 := 72
  let m := 5
  let n := 7
  let fraction1 := a / d1
  let fraction2 := b / d2
  ((fraction1 - (b * n / (d2 * n))) * (m / n)) = (-145 / 882) :=
by
  sorry

end simplify_and_multiply_l112_112509


namespace algebraic_expr_value_l112_112707

theorem algebraic_expr_value {a b : ℝ} (h: a + b = 1) : a^2 - b^2 + 2 * b + 9 = 10 := 
by
  sorry

end algebraic_expr_value_l112_112707


namespace average_weight_of_a_b_c_l112_112764

theorem average_weight_of_a_b_c (A B C : ℕ) 
  (h1 : (A + B) / 2 = 25) 
  (h2 : (B + C) / 2 = 28) 
  (hB : B = 16) : 
  (A + B + C) / 3 = 30 := 
by 
  sorry

end average_weight_of_a_b_c_l112_112764


namespace animals_total_sleep_in_one_week_l112_112936

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

end animals_total_sleep_in_one_week_l112_112936


namespace total_tickets_l112_112112

theorem total_tickets (A C : ℕ) (cost_adult cost_child total_cost : ℝ) 
  (h1 : cost_adult = 5.50) 
  (h2 : cost_child = 3.50) 
  (h3 : C = 16) 
  (h4 : total_cost = 83.50) 
  (h5 : cost_adult * A + cost_child * C = total_cost) : 
  A + C = 21 := 
by 
  sorry

end total_tickets_l112_112112


namespace B_contribution_to_capital_l112_112145

theorem B_contribution_to_capital (A_capital : ℝ) (A_months : ℝ) (B_months : ℝ) (profit_ratio_A : ℝ) (profit_ratio_B : ℝ) (B_contribution : ℝ) :
  A_capital = 4500 →
  A_months = 12 →
  B_months = 5 →
  profit_ratio_A = 2 →
  profit_ratio_B = 3 →
  B_contribution = (4500 * 12 * 3) / (5 * 2) → 
  B_contribution = 16200 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end B_contribution_to_capital_l112_112145


namespace area_of_square_field_l112_112260

theorem area_of_square_field (x : ℝ) 
  (h₁ : 1.10 * (4 * x - 2) = 732.6) : 
  x = 167 → x ^ 2 = 27889 := by
  sorry

end area_of_square_field_l112_112260


namespace initial_sheep_count_l112_112228

theorem initial_sheep_count 
    (S : ℕ)
    (initial_horses : ℕ := 100)
    (initial_chickens : ℕ := 9)
    (gifted_goats : ℕ := 37)
    (male_animals : ℕ := 53)
    (total_animals_half : ℕ := 106) :
    ((initial_horses + S + initial_chickens) / 2 + gifted_goats = total_animals_half) → 
    S = 29 :=
by
  intro h
  sorry

end initial_sheep_count_l112_112228


namespace new_tv_width_l112_112749

-- Define the conditions
def first_tv_width := 24
def first_tv_height := 16
def first_tv_cost := 672
def new_tv_height := 32
def new_tv_cost := 1152
def cost_difference := 1

-- Define the question as a theorem
theorem new_tv_width : 
  let first_tv_area := first_tv_width * first_tv_height
  let first_tv_cost_per_sq_inch := first_tv_cost / first_tv_area
  let new_tv_cost_per_sq_inch := first_tv_cost_per_sq_inch - cost_difference
  let new_tv_area := new_tv_cost / new_tv_cost_per_sq_inch
  let new_tv_width := new_tv_area / new_tv_height
  new_tv_width = 48 :=
by
  -- Here, we would normally provide the proof steps, but we insert sorry as required.
  sorry

end new_tv_width_l112_112749


namespace total_rainfall_2004_l112_112730

def average_rainfall_2003 := 50 -- in mm
def extra_rainfall_2004 := 3 -- in mm
def average_rainfall_2004 := average_rainfall_2003 + extra_rainfall_2004 -- in mm
def days_february_2004 := 29
def days_other_months := 30
def months := 12
def months_without_february := months - 1

theorem total_rainfall_2004 : 
  (average_rainfall_2004 * days_february_2004) + (months_without_february * average_rainfall_2004 * days_other_months) = 19027 := 
by sorry

end total_rainfall_2004_l112_112730


namespace binom_n_2_l112_112412

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l112_112412


namespace unbroken_seashells_l112_112785

theorem unbroken_seashells (total broken : ℕ) (h1 : total = 7) (h2 : broken = 4) : total - broken = 3 :=
by
  -- Proof goes here…
  sorry

end unbroken_seashells_l112_112785


namespace expansion_a0_value_l112_112582

theorem expansion_a0_value :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), (∀ x : ℝ, (x+1)^5 = a_0 + a_1*(x-1) + a_2*(x-1)^2 + a_3*(x-1)^3 + a_4*(x-1)^4 + a_5*(x-1)^5) ∧ a_0 = 32 :=
  sorry

end expansion_a0_value_l112_112582


namespace fractional_eq_a_range_l112_112600

theorem fractional_eq_a_range (a : ℝ) :
  (∃ x : ℝ, (a / (x + 2) = 1 - 3 / (x + 2)) ∧ (x < 0)) ↔ (a < -1 ∧ a ≠ -3) := by
  sorry

end fractional_eq_a_range_l112_112600


namespace marble_distribution_l112_112612

-- Define the problem statement using conditions extracted above
theorem marble_distribution :
  ∃ (A B C D : ℕ), A + B + C + D = 28 ∧
  (A = 7 ∨ B = 7 ∨ C = 7 ∨ D = 7) ∧
  ((A = 7 → B + C + D = 21) ∧
   (B = 7 → A + C + D = 21) ∧
   (C = 7 → A + B + D = 21) ∧
   (D = 7 → A + B + C = 21)) :=
sorry

end marble_distribution_l112_112612


namespace find_ratio_of_geometric_sequence_l112_112713

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  2 * a2 = a1 + a3

theorem find_ratio_of_geometric_sequence 
  {a : ℕ → ℝ} {q : ℝ}
  (h_pos : ∀ n, 0 < a n)
  (h_geo : geometric_sequence a q)
  (h_arith : arithmetic_sequence (a 1) ((1/2) * a 3) (2 * a 2)) :
  (a 10) / (a 8) = 3 + 2 * sqrt 2 :=
sorry

end find_ratio_of_geometric_sequence_l112_112713


namespace triangle_inequality_for_min_segments_l112_112349

theorem triangle_inequality_for_min_segments
  (a b c d : ℝ)
  (a1 b1 c1 : ℝ)
  (h1 : a1 = min a d)
  (h2 : b1 = min b d)
  (h3 : c1 = min c d)
  (h_triangle : c < a + b) :
  a1 + b1 > c1 ∧ a1 + c1 > b1 ∧ b1 + c1 > a1 := sorry

end triangle_inequality_for_min_segments_l112_112349


namespace shrimp_per_pound_l112_112653

theorem shrimp_per_pound (shrimp_per_guest guests : ℕ) (cost_per_pound : ℝ) (total_spent : ℝ)
  (hshrimp_per_guest : shrimp_per_guest = 5) (hguests : guests = 40) (hcost_per_pound : cost_per_pound = 17.0) (htotal_spent : total_spent = 170.0) :
  let total_shrimp := shrimp_per_guest * guests
  let total_pounds := total_spent / cost_per_pound
  total_shrimp / total_pounds = 20 :=
by
  sorry

end shrimp_per_pound_l112_112653


namespace circle_represents_valid_a_l112_112599

theorem circle_represents_valid_a (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2 * a * x - 4 * y + 5 * a = 0) → (a > 4 ∨ a < 1) :=
by
  sorry

end circle_represents_valid_a_l112_112599


namespace ninety_eight_squared_l112_112320

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l112_112320


namespace necessary_not_sufficient_condition_t_for_b_l112_112460

variable (x y : ℝ)

def condition_t : Prop := x ≤ 12 ∨ y ≤ 16
def condition_b : Prop := x + y ≤ 28 ∨ x * y ≤ 192

theorem necessary_not_sufficient_condition_t_for_b (h : condition_b x y) : condition_t x y ∧ ¬ (condition_t x y → condition_b x y) := by
  sorry

end necessary_not_sufficient_condition_t_for_b_l112_112460


namespace series_sum_correct_l112_112951

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem series_sum_correct :
  geometric_series_sum (1 / 2) (-1 / 3) 6 = 91 / 243 :=
by
  -- Proof goes here
  sorry

end series_sum_correct_l112_112951


namespace find_y_l112_112971

theorem find_y (x : ℝ) (h : x^2 + (1 / x)^2 = 7) : x + 1 / x = 3 :=
by
  sorry

end find_y_l112_112971


namespace average_a_b_l112_112518

theorem average_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27) : (A + B) / 2 = 40 := 
by
  sorry

end average_a_b_l112_112518


namespace number_of_tests_initially_l112_112069

theorem number_of_tests_initially (n : ℕ) (h1 : (90 * n) / n = 90)
  (h2 : ((90 * n) - 75) / (n - 1) = 95) : n = 4 :=
sorry

end number_of_tests_initially_l112_112069


namespace subset1_squares_equals_product_subset2_squares_equals_product_l112_112822

theorem subset1_squares_equals_product :
  (1^2 + 3^2 + 4^2 + 9^2 + 107^2 = 1 * 3 * 4 * 9 * 107) :=
sorry

theorem subset2_squares_equals_product :
  (3^2 + 4^2 + 9^2 + 107^2 + 11555^2 = 3 * 4 * 9 * 107 * 11555) :=
sorry

end subset1_squares_equals_product_subset2_squares_equals_product_l112_112822


namespace angle_E_in_quadrilateral_EFGH_l112_112368

theorem angle_E_in_quadrilateral_EFGH 
  (angle_E angle_F angle_G angle_H : ℝ) 
  (h1 : angle_E = 2 * angle_F)
  (h2 : angle_E = 3 * angle_G)
  (h3 : angle_E = 6 * angle_H)
  (sum_angles : angle_E + angle_F + angle_G + angle_H = 360) : 
  angle_E = 180 :=
by
  sorry

end angle_E_in_quadrilateral_EFGH_l112_112368


namespace solve_for_k_l112_112596

theorem solve_for_k (a k : ℝ) (h : a ^ 10 / (a ^ k) ^ 4 = a ^ 2) : k = 2 :=
by
  sorry

end solve_for_k_l112_112596


namespace necessarily_positive_expressions_l112_112883

theorem necessarily_positive_expressions
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  (b + b^2 > 0) ∧ (b + 3 * b^2 > 0) :=
sorry

end necessarily_positive_expressions_l112_112883


namespace one_more_square_possible_l112_112027

def grid_size : ℕ := 29
def total_cells : ℕ := grid_size * grid_size
def number_of_squares_removed : ℕ := 99
def cells_per_square : ℕ := 4
def total_removed_cells : ℕ := number_of_squares_removed * cells_per_square
def remaining_cells : ℕ := total_cells - total_removed_cells

theorem one_more_square_possible :
  remaining_cells ≥ cells_per_square :=
sorry

end one_more_square_possible_l112_112027


namespace find_bases_l112_112066

theorem find_bases {F1 F2 : ℝ} (R1 R2 : ℕ) 
                   (hR1 : R1 = 9)
                   (hR2 : R2 = 6)
                   (hF1_R1 : F1 = 0.484848 * 9^2 / (9^2 - 1))
                   (hF2_R1 : F2 = 0.848484 * 9^2 / (9^2 - 1))
                   (hF1_R2 : F1 = 0.353535 * 6^2 / (6^2 - 1))
                   (hF2_R2 : F2 = 0.535353 * 6^2 / (6^2 - 1))
                   : R1 + R2 = 15 :=
by
  sorry

end find_bases_l112_112066


namespace heartbeats_during_race_l112_112158

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l112_112158


namespace part1_part2_part3_part3_expectation_l112_112088

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end part1_part2_part3_part3_expectation_l112_112088


namespace shorter_leg_of_right_triangle_l112_112990

theorem shorter_leg_of_right_triangle (a b : ℕ) (h1 : a < b)
    (h2 : a^2 + b^2 = 65^2) : a = 16 :=
sorry

end shorter_leg_of_right_triangle_l112_112990


namespace symmetry_center_of_g_l112_112502

open Real

noncomputable def g (x : ℝ) : ℝ := cos ((1 / 2) * x - π / 6)

def center_of_symmetry : Set (ℝ × ℝ) := { p | ∃ k : ℤ, p = (2 * k * π + 4 * π / 3, 0) }

theorem symmetry_center_of_g :
  (∃ p : ℝ × ℝ, p ∈ center_of_symmetry) :=
sorry

end symmetry_center_of_g_l112_112502


namespace susie_earnings_l112_112094

-- Define the constants and conditions
def price_per_slice : ℕ := 3
def price_per_whole_pizza : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

-- Calculate earnings from slices and whole pizzas
def earnings_from_slices : ℕ := slices_sold * price_per_slice
def earnings_from_whole_pizzas : ℕ := whole_pizzas_sold * price_per_whole_pizza
def total_earnings : ℕ := earnings_from_slices + earnings_from_whole_pizzas

-- Prove that the total earnings are $117
theorem susie_earnings : total_earnings = 117 := by
  sorry

end susie_earnings_l112_112094


namespace ellipse_polar_inverse_sum_l112_112870

noncomputable def ellipse_equation (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

theorem ellipse_polar_inverse_sum (A B : ℝ × ℝ)
  (hA : ∃ α₁, ellipse_equation α₁ = A)
  (hB : ∃ α₂, ellipse_equation α₂ = B)
  (hPerp : A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / (A.1 ^ 2 + A.2 ^ 2) + 1 / (B.1 ^ 2 + B.2 ^ 2)) = 7 / 12 :=
by
  sorry

end ellipse_polar_inverse_sum_l112_112870


namespace solve_system_of_equations_l112_112541

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + 2 * x * y + x^2 - 6 * y - 6 * x + 5 = 0)
  (h2 : y - x + 1 = x^2 - 3 * x) : 
  ((x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) ∨ (x = -2 ∧ y = 7)) ∧ x ≠ 0 ∧ x ≠ 3 :=
by 
  sorry

end solve_system_of_equations_l112_112541


namespace circle_area_pi_div_2_l112_112443

open Real EuclideanGeometry

variable (x y : ℝ)

def circleEquation : Prop := 3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

theorem circle_area_pi_div_2
  (h : circleEquation x y) : 
  ∃ (r : ℝ), r = sqrt 0.5 ∧ π * r * r = π / 2 :=
by
  sorry

end circle_area_pi_div_2_l112_112443


namespace functional_equation_solution_l112_112338

theorem functional_equation_solution (f : ℚ → ℕ) :
  (∀ (x y : ℚ) (hx : 0 < x) (hy : 0 < y),
    f (x * y) * Nat.gcd (f x * f y) (f (x⁻¹) * f (y⁻¹)) = (x * y) * f (x⁻¹) * f (y⁻¹))
  → (∀ (x : ℚ) (hx : 0 < x), f x = x.num) :=
sorry

end functional_equation_solution_l112_112338


namespace two_distinct_solutions_exist_l112_112252

theorem two_distinct_solutions_exist :
  ∃ (a1 b1 c1 d1 e1 a2 b2 c2 d2 e2 : ℕ), 
    1 ≤ a1 ∧ a1 ≤ 9 ∧ 1 ≤ b1 ∧ b1 ≤ 9 ∧ 1 ≤ c1 ∧ c1 ≤ 9 ∧ 1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ e1 ∧ e1 ≤ 9 ∧
    1 ≤ a2 ∧ a2 ≤ 9 ∧ 1 ≤ b2 ∧ b2 ≤ 9 ∧ 1 ≤ c2 ∧ c2 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ e2 ∧ e2 ≤ 9 ∧
    (b1 - d1 = 2) ∧ (d1 - a1 = 3) ∧ (a1 - c1 = 1) ∧
    (b2 - d2 = 2) ∧ (d2 - a2 = 3) ∧ (a2 - c2 = 1) ∧
    ¬ (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) :=
by
  sorry

end two_distinct_solutions_exist_l112_112252


namespace problem_solution_l112_112354

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_solution (a m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) →
  a = 1 ∧ (∃ n : ℝ, f n 1 ≤ m - f (-n) 1) → 4 ≤ m := 
by
  sorry

end problem_solution_l112_112354


namespace magnitude_of_a_l112_112854

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)
variable (hθ : theta = π / 3)
variable (hb : ‖b‖ = 1)
variable (hab : ‖a + 2 • b‖ = 2 * sqrt 3)

theorem magnitude_of_a :
  ‖a‖ = 2 :=
by
  sorry

end magnitude_of_a_l112_112854


namespace find_abs_x_l112_112084

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l112_112084


namespace no_stew_left_l112_112625

theorem no_stew_left (company : Type) (stew : ℝ)
    (one_third_stayed : ℝ)
    (two_thirds_went : ℝ)
    (camp_consumption : ℝ)
    (range_consumption_per_portion : ℝ)
    (range_portion_multiplier : ℝ)
    (total_stew : ℝ) : 
    one_third_stayed = 1 / 3 →
    two_thirds_went = 2 / 3 →
    camp_consumption = 1 / 4 →
    range_portion_multiplier = 1.5 →
    total_stew = camp_consumption + (range_portion_multiplier * (two_thirds_went * (camp_consumption / one_third_stayed))) →
    total_stew = 1 →
    stew = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- here would be the proof steps
  sorry

end no_stew_left_l112_112625


namespace cannot_determine_position_l112_112121

-- Define the conditions
def east_longitude_122_north_latitude_43_6 : Prop := true
def row_6_seat_3_in_cinema : Prop := true
def group_1_in_classroom : Prop := false
def island_50_nautical_miles_north_northeast_another : Prop := true

-- Define the theorem
theorem cannot_determine_position :
  ¬ ((east_longitude_122_north_latitude_43_6 = false) ∧
     (row_6_seat_3_in_cinema = false) ∧
     (island_50_nautical_miles_north_northeast_another = false) ∧
     (group_1_in_classroom = true)) :=
by
  sorry

end cannot_determine_position_l112_112121


namespace expected_lifetime_at_least_four_l112_112241

universe u

variables (α : Type u) [MeasurableSpace α] {𝒫 : ProbabilitySpace α}
variables {ξ η : α → ℝ} [IsFiniteExpectation ξ] [IsFiniteExpectation η]

noncomputable def max_lifetime : α → ℝ := λ ω, max (ξ ω) (η ω)

theorem expected_lifetime_at_least_four 
  (h : ∀ ω, max (ξ ω) (η ω) ≥ η ω)
  (h_eta : @Expectation α _ _ η  = 4) : 
  @Expectation α _ _ max_lifetime ≥ 4 :=
by
  sorry

end expected_lifetime_at_least_four_l112_112241


namespace rosa_initial_flowers_l112_112508

-- Definitions derived from conditions
def initial_flowers (total_flowers : ℕ) (given_flowers : ℕ) : ℕ :=
  total_flowers - given_flowers

-- The theorem stating the proof problem
theorem rosa_initial_flowers : initial_flowers 90 23 = 67 :=
by
  -- The proof goes here
  sorry

end rosa_initial_flowers_l112_112508


namespace female_democrats_count_l112_112646

theorem female_democrats_count 
  (F M D : ℕ)
  (total_participants : F + M = 660)
  (total_democrats : F / 2 + M / 4 = 660 / 3)
  (female_democrats : D = F / 2) : 
  D = 110 := 
by
  sorry

end female_democrats_count_l112_112646


namespace megan_bottles_l112_112379

theorem megan_bottles (initial_bottles drank gave_away remaining_bottles : ℕ) 
  (h1 : initial_bottles = 45)
  (h2 : drank = 8)
  (h3 : gave_away = 12) :
  remaining_bottles = initial_bottles - (drank + gave_away) :=
by 
  sorry

end megan_bottles_l112_112379


namespace customer_paid_amount_l112_112673

theorem customer_paid_amount (O : ℕ) (D : ℕ) (P : ℕ) (hO : O = 90) (hD : D = 20) (hP : P = O - D) : P = 70 :=
sorry

end customer_paid_amount_l112_112673


namespace donny_remaining_money_l112_112331

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end donny_remaining_money_l112_112331


namespace diana_total_earnings_l112_112446

-- Define the earnings in each month
def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def september_earnings : ℕ := 2 * august_earnings

-- State the theorem that the total earnings over the three months is $1500
theorem diana_total_earnings : july_earnings + august_earnings + september_earnings = 1500 :=
by
  have h1 : august_earnings = 3 * july_earnings := rfl
  have h2 : september_earnings = 2 * august_earnings := rfl
  sorry

end diana_total_earnings_l112_112446


namespace complex_quadrant_l112_112271

-- Declare the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Declare the complex number z as per the condition
noncomputable def z : ℂ := (2 * i) / (i - 1)

-- State and prove that the complex number z lies in the fourth quadrant
theorem complex_quadrant : (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_quadrant_l112_112271


namespace car_value_correct_l112_112704

-- Define the initial value and the annual decrease percentages
def initial_value : ℝ := 10000
def annual_decreases : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

-- Function to compute the value of the car after n years
def value_after_years (initial_value : ℝ) (annual_decreases : List ℝ) : ℝ :=
  annual_decreases.foldl (λ acc decrease => acc * (1 - decrease)) initial_value

-- The target value after 5 years
def target_value : ℝ := 5348.88

-- Theorem stating that the computed value matches the target value
theorem car_value_correct :
  value_after_years initial_value annual_decreases = target_value := 
sorry

end car_value_correct_l112_112704


namespace sum_of_bases_is_16_l112_112441

/-
  Given the fractions G_1 and G_2 in two different bases S_1 and S_2, we need to show 
  that the sum of these bases S_1 and S_2 in base ten is 16.
-/
theorem sum_of_bases_is_16 (S_1 S_2 G_1 G_2 : ℕ) :
  (G_1 = (4 * S_1 + 5) / (S_1^2 - 1)) →
  (G_2 = (5 * S_1 + 4) / (S_1^2 - 1)) →
  (G_1 = (S_2 + 4) / (S_2^2 - 1)) →
  (G_2 = (4 * S_2 + 1) / (S_2^2 - 1)) →
  S_1 + S_2 = 16 :=
by
  intros hG1_S1 hG2_S1 hG1_S2 hG2_S2
  sorry

end sum_of_bases_is_16_l112_112441


namespace garden_enlargement_l112_112554

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l112_112554


namespace donut_selection_count_l112_112752

def num_donut_selections : ℕ :=
  Nat.choose 9 3

theorem donut_selection_count : num_donut_selections = 84 := 
by
  sorry

end donut_selection_count_l112_112752


namespace total_cubes_proof_l112_112855

def Grady_initial_red_cubes := 20
def Grady_initial_blue_cubes := 15
def Gage_initial_red_cubes := 10
def Gage_initial_blue_cubes := 12
def Harper_initial_red_cubes := 8
def Harper_initial_blue_cubes := 10

def Gage_red_received := (2 / 5) * Grady_initial_red_cubes
def Gage_blue_received := (1 / 3) * Grady_initial_blue_cubes

def Grady_red_after_Gage := Grady_initial_red_cubes - Gage_red_received
def Grady_blue_after_Gage := Grady_initial_blue_cubes - Gage_blue_received

def Harper_red_received := (1 / 4) * Grady_red_after_Gage
def Harper_blue_received := (1 / 2) * Grady_blue_after_Gage

def Gage_total_red := Gage_initial_red_cubes + Gage_red_received
def Gage_total_blue := Gage_initial_blue_cubes + Gage_blue_received

def Harper_total_red := Harper_initial_red_cubes + Harper_red_received
def Harper_total_blue := Harper_initial_blue_cubes + Harper_blue_received

def Gage_total_cubes := Gage_total_red + Gage_total_blue
def Harper_total_cubes := Harper_total_red + Harper_total_blue

def Gage_Harper_total_cubes := Gage_total_cubes + Harper_total_cubes

theorem total_cubes_proof : Gage_Harper_total_cubes = 61 := by
  sorry

end total_cubes_proof_l112_112855


namespace sum_a_b_eq_neg2_l112_112594

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : (a - 2)^2 + |b + 4| = 0) : a + b = -2 := 
by 
  sorry

end sum_a_b_eq_neg2_l112_112594


namespace rationalize_denominator_l112_112629

theorem rationalize_denominator :
  ∃ (A B C : ℤ), 
  (A + B * Real.sqrt C) = (2 + Real.sqrt 5) / (3 - Real.sqrt 5) 
  ∧ A = 11 ∧ B = 5 ∧ C = 5 ∧ A * B * C = 275 := by
  sorry

end rationalize_denominator_l112_112629


namespace peaches_per_basket_l112_112284

-- Given conditions as definitions in Lean 4
def red_peaches : Nat := 7
def green_peaches : Nat := 3

-- The proof statement showing each basket contains 10 peaches in total.
theorem peaches_per_basket : red_peaches + green_peaches = 10 := by
  sorry

end peaches_per_basket_l112_112284


namespace arithmetic_progression_a6_l112_112037

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l112_112037


namespace find_a6_l112_112043

-- Define an arithmetic progression.
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the necessary conditions given in the problem.
def conditions (a d : ℕ) : Prop :=
  (arithmetic_progression a d 1 + arithmetic_progression a d 2 + arithmetic_progression a d 3 = 168) ∧
  (arithmetic_progression a d 2 - arithmetic_progression a d 5 = 42)

-- State the theorem with the final value assertion.
theorem find_a6 (a d : ℕ) (h : conditions a (-14)) : 
  arithmetic_progression a (-14) 6 = 3 := 
sorry

end find_a6_l112_112043


namespace surface_area_ratio_l112_112531

-- Definitions based on conditions
def side_length (s : ℝ) := s > 0
def A_cube (s : ℝ) := 6 * s ^ 2
def A_rect (s : ℝ) := 2 * (2 * s) * (3 * s) + 2 * (2 * s) * (4 * s) + 2 * (3 * s) * (4 * s)

-- Theorem statement proving the ratio
theorem surface_area_ratio (s : ℝ) (h : side_length s) : A_cube s / A_rect s = 3 / 26 :=
by
  sorry

end surface_area_ratio_l112_112531


namespace product_slope_intercept_lt_neg1_l112_112265

theorem product_slope_intercept_lt_neg1 :
  let m := -3 / 4
  let b := 3 / 2
  m * b < -1 := 
by
  let m := -3 / 4
  let b := 3 / 2
  sorry

end product_slope_intercept_lt_neg1_l112_112265


namespace xyz_squared_sum_l112_112479

theorem xyz_squared_sum (x y z : ℝ) 
  (h1 : x^2 + 4 * y^2 + 16 * z^2 = 48)
  (h2 : x * y + 4 * y * z + 2 * z * x = 24) :
  x^2 + y^2 + z^2 = 21 :=
sorry

end xyz_squared_sum_l112_112479


namespace open_box_volume_l112_112676

theorem open_box_volume (l w s : ℕ) (h1 : l = 50)
  (h2 : w = 36) (h3 : s = 8) : (l - 2 * s) * (w - 2 * s) * s = 5440 :=
by {
  sorry
}

end open_box_volume_l112_112676


namespace brownie_pieces_count_l112_112621

def area_of_pan (length width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

def number_of_pieces (pan_area piece_area : ℕ) : ℕ := pan_area / piece_area

theorem brownie_pieces_count :
  let pan_length := 24
  let pan_width := 15
  let piece_side := 3
  let pan_area := area_of_pan pan_length pan_width
  let piece_area := area_of_piece piece_side
  number_of_pieces pan_area piece_area = 40 :=
by
  sorry

end brownie_pieces_count_l112_112621


namespace gcd_45_75_eq_15_l112_112188

theorem gcd_45_75_eq_15 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_eq_15_l112_112188


namespace train_cross_pole_time_l112_112424

noncomputable def L_train : ℝ := 300 -- Length of the train in meters
noncomputable def L_platform : ℝ := 870 -- Length of the platform in meters
noncomputable def t_platform : ℝ := 39 -- Time to cross the platform in seconds

theorem train_cross_pole_time
  (L_train : ℝ)
  (L_platform : ℝ)
  (t_platform : ℝ)
  (D : ℝ := L_train + L_platform)
  (v : ℝ := D / t_platform)
  (t_pole : ℝ := L_train / v) :
  t_pole = 10 :=
by sorry

end train_cross_pole_time_l112_112424


namespace comparison_M_N_l112_112970

def M (x : ℝ) : ℝ := x^2 - 3*x + 7
def N (x : ℝ) : ℝ := -x^2 + x + 1

theorem comparison_M_N (x : ℝ) : M x > N x :=
  by sorry

end comparison_M_N_l112_112970


namespace smallest_n_repeating_251_l112_112033

theorem smallest_n_repeating_251 (m n : ℕ) (hmn : m < n) (coprime : Nat.gcd m n = 1) :
  (∃ m : ℕ, ∃ n : ℕ, Nat.gcd m n = 1 ∧ m < n ∧ let r := real.to_rat_repr (↥m / ↥n) in (r.2 ≥ 1000 * m mod n = 251)) → n = 127 :=
sorry

end smallest_n_repeating_251_l112_112033


namespace sum_of_consecutive_integers_l112_112104

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end sum_of_consecutive_integers_l112_112104


namespace factorize_expression_l112_112185

theorem factorize_expression (a : ℝ) : 
  (a + 1) * (a + 2) + 1 / 4 = (a + 3 / 2)^2 := 
by 
  sorry

end factorize_expression_l112_112185


namespace exam_results_l112_112989

variable (E F G H : Prop)

def emma_statement : Prop := E → F
def frank_statement : Prop := F → ¬G
def george_statement : Prop := G → H
def exactly_two_asing : Prop :=
  (E ∧ F ∧ ¬G ∧ ¬H) ∨ (¬E ∧ F ∧ G ∧ ¬H) ∨
  (¬E ∧ ¬F ∧ G ∧ H) ∨ (¬E ∧ F ∧ ¬G ∧ H) ∨
  (E ∧ ¬F ∧ ¬G ∧ H)

theorem exam_results :
  (E ∧ F) ∨ (G ∧ H) :=
by {
  sorry
}

end exam_results_l112_112989


namespace triangle_cosines_identity_l112_112739

theorem triangle_cosines_identity 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) / a) + 
  (c^2 * Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) / b) + 
  (a^2 * Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / c) = 
  (a^4 + b^4 + c^4) / (2 * a * b * c) :=
by
  sorry

end triangle_cosines_identity_l112_112739


namespace log_identity_l112_112464

theorem log_identity
  (x : ℝ)
  (h1 : x < 1)
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^4) / Real.log 10 = 100) :
  (Real.log x / Real.log 10)^3 - Real.log (x^5) / Real.log 10 = -114 + Real.sqrt 104 := 
by
  sorry

end log_identity_l112_112464


namespace cannot_sum_85_with_five_coins_l112_112578

def coin_value (c : Nat) : Prop :=
  c = 1 ∨ c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50

theorem cannot_sum_85_with_five_coins : 
  ¬ ∃ (a b c d e : Nat), 
    coin_value a ∧ 
    coin_value b ∧ 
    coin_value c ∧ 
    coin_value d ∧ 
    coin_value e ∧ 
    a + b + c + d + e = 85 :=
by
  sorry

end cannot_sum_85_with_five_coins_l112_112578


namespace grade12_students_selected_l112_112527

theorem grade12_students_selected 
    (N : ℕ) (n10 : ℕ) (n12 : ℕ) (k : ℕ) 
    (h1 : N = 1200)
    (h2 : n10 = 240)
    (h3 : 3 * N / (k + 5 + 3) = n12)
    (h4 : k * N / (k + 5 + 3) = n10) :
    n12 = 360 := 
by sorry

end grade12_students_selected_l112_112527


namespace initial_concentration_is_40_l112_112547

noncomputable def initial_concentration_fraction : ℝ := 1 / 3
noncomputable def replaced_solution_concentration : ℝ := 25
noncomputable def resulting_concentration : ℝ := 35
noncomputable def initial_concentration := 40

theorem initial_concentration_is_40 (C : ℝ) (h1 : C = (3 / 2) * (resulting_concentration - (initial_concentration_fraction * replaced_solution_concentration))) :
  C = initial_concentration :=
by sorry

end initial_concentration_is_40_l112_112547


namespace eric_green_marbles_l112_112182

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end eric_green_marbles_l112_112182


namespace regular_18gon_symmetries_l112_112941

theorem regular_18gon_symmetries :
  let L := 18
  let R := 20
  L + R = 38 := by
sorry

end regular_18gon_symmetries_l112_112941


namespace weight_difference_l112_112097

noncomputable def W_A : ℝ := 78

variable (W_B W_C W_D W_E : ℝ)

axiom cond1 : (W_A + W_B + W_C) / 3 = 84
axiom cond2 : (W_A + W_B + W_C + W_D) / 4 = 80
axiom cond3 : (W_B + W_C + W_D + W_E) / 4 = 79

theorem weight_difference : W_E - W_D = 6 :=
by
  have h1 : W_A = 78 := rfl
  sorry

end weight_difference_l112_112097


namespace order_of_f_l112_112245

-- Define the function f
variables {f : ℝ → ℝ}

-- Definition of even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of monotonic increasing function on [0, +∞)
def monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, (0 ≤ x ∧ 0 ≤ y ∧ x ≤ y) → f x ≤ f y

-- The main problem statement
theorem order_of_f (h_even : even_function f) (h_mono : monotonically_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
  sorry

end order_of_f_l112_112245


namespace highest_red_ball_probability_l112_112605

theorem highest_red_ball_probability :
  ∀ (total balls red yellow black : ℕ),
    total = 10 →
    red = 7 →
    yellow = 2 →
    black = 1 →
    (red / total) > (yellow / total) ∧ (red / total) > (black / total) :=
by
  intro total balls red yellow black
  intro h_total h_red h_yellow h_black
  sorry

end highest_red_ball_probability_l112_112605


namespace color_plane_no_unit_equilateral_same_color_l112_112610

theorem color_plane_no_unit_equilateral_same_color :
  ∃ (coloring : ℝ × ℝ → ℕ), (∀ (A B C : ℝ × ℝ),
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    (coloring A ≠ coloring B ∨ coloring B ≠ coloring C ∨ coloring C ≠ coloring A)) :=
sorry

end color_plane_no_unit_equilateral_same_color_l112_112610


namespace maxSUVMileage_l112_112682

noncomputable def maxSUVDistance : ℝ := 217.12

theorem maxSUVMileage 
    (tripGal : ℝ) (mpgHighway : ℝ) (mpgCity : ℝ)
    (regularHighwayRatio : ℝ) (regularCityRatio : ℝ)
    (peakHighwayRatio : ℝ) (peakCityRatio : ℝ) :
    tripGal = 23 →
    mpgHighway = 12.2 →
    mpgCity = 7.6 →
    regularHighwayRatio = 0.4 →
    regularCityRatio = 0.6 →
    peakHighwayRatio = 0.25 →
    peakCityRatio = 0.75 →
    max ((tripGal * regularHighwayRatio * mpgHighway) + (tripGal * regularCityRatio * mpgCity))
        ((tripGal * peakHighwayRatio * mpgHighway) + (tripGal * peakCityRatio * mpgCity)) = maxSUVDistance :=
by
  intros
  -- Proof would go here
  sorry

end maxSUVMileage_l112_112682


namespace value_of_m_solve_system_relationship_x_y_l112_112128

-- Part 1: Prove the value of m is 1
theorem value_of_m (x : ℝ) (m : ℝ) (h1 : 2 - x = x + 4) (h2 : m * (1 - x) = x + 3) : m = 1 := sorry

-- Part 2: Solve the system of equations given m = 1
theorem solve_system (x y : ℝ) (h1 : 3 * x + 2 * 1 = - y) (h2 : 2 * x + 2 * y = 1 - 1) : x = -1 ∧ y = 1 := sorry

-- Part 3: Relationship between x and y regardless of m
theorem relationship_x_y (x y m : ℝ) (h1 : 3 * x + y = -2 * m) (h2 : 2 * x + 2 * y = m - 1) : 7 * x + 5 * y = -2 := sorry

end value_of_m_solve_system_relationship_x_y_l112_112128


namespace probability_of_selection_of_Ram_l112_112405

noncomputable def P_Ravi : ℚ := 1 / 5
noncomputable def P_Ram_and_Ravi : ℚ := 57 / 1000  -- This is the exact form of 0.05714285714285714

axiom independent_selection : ∀ (P_Ram P_Ravi : ℚ), P_Ram_and_Ravi = P_Ram * P_Ravi

theorem probability_of_selection_of_Ram (P_Ram : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ram = 2 / 7 := by
  intro h
  have h1 : P_Ram = P_Ram_and_Ravi / P_Ravi := sorry
  rw [h1, P_Ram_and_Ravi, P_Ravi]
  norm_num
  exact sorry

end probability_of_selection_of_Ram_l112_112405


namespace parallelogram_properties_l112_112550

variable {b h : ℕ}

theorem parallelogram_properties
  (hb : b = 20)
  (hh : h = 4) :
  (b * h = 80) ∧ ((b^2 + h^2) = 416) :=
by
  sorry

end parallelogram_properties_l112_112550


namespace workshop_workers_l112_112096

theorem workshop_workers (W N: ℕ) 
  (h1: 8000 * W = 70000 + 6000 * N) 
  (h2: W = 7 + N) : 
  W = 14 := 
  by 
    sorry

end workshop_workers_l112_112096


namespace homework_total_l112_112383

theorem homework_total :
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  math_pages + reading_pages + science_pages = 62 :=
by
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  show math_pages + reading_pages + science_pages = 62
  sorry

end homework_total_l112_112383


namespace locus_of_midpoint_of_square_l112_112807

theorem locus_of_midpoint_of_square (a : ℝ) (x y : ℝ) (h1 : x^2 + y^2 = 4 * a^2) :
  (∃ X Y : ℝ, 2 * X = x ∧ 2 * Y = y ∧ X^2 + Y^2 = a^2) :=
by {
  -- No proof is required, so we use 'sorry' here
  sorry
}

end locus_of_midpoint_of_square_l112_112807


namespace volume_frustum_2240_over_3_l112_112810

def volume_of_pyramid (base_edge: ℝ) (height: ℝ) : ℝ :=
    (1 / 3) * (base_edge ^ 2) * height

def volume_of_frustum (original_base_edge: ℝ) (original_height: ℝ)
  (smaller_base_edge: ℝ) (smaller_height: ℝ) : ℝ :=
  volume_of_pyramid original_base_edge original_height - volume_of_pyramid smaller_base_edge smaller_height

theorem volume_frustum_2240_over_3 :
  volume_of_frustum 16 10 8 5 = 2240 / 3 :=
by sorry

end volume_frustum_2240_over_3_l112_112810


namespace four_digit_integer_transformation_l112_112618

theorem four_digit_integer_transformation (a b c d n : ℕ) (A : ℕ)
  (hA : A = 1000 * a + 100 * b + 10 * c + d)
  (ha : a + 2 < 10)
  (hc : c + 2 < 10)
  (hb : b ≥ 2)
  (hd : d ≥ 2)
  (hA4 : 1000 ≤ A ∧ A < 10000) :
  (1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n)) = n * A → n = 2 → A = 1818 :=
by sorry

end four_digit_integer_transformation_l112_112618


namespace inequality_solution_set_l112_112569

theorem inequality_solution_set (x : ℝ) : (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := sorry

end inequality_solution_set_l112_112569


namespace gcd_of_72_and_90_l112_112290

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end gcd_of_72_and_90_l112_112290


namespace base_of_number_l112_112364

theorem base_of_number (b : ℕ) : 
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 :=
by
  sorry

end base_of_number_l112_112364


namespace jeanne_should_buy_more_tickets_l112_112997

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end jeanne_should_buy_more_tickets_l112_112997


namespace percentage_four_petals_l112_112370

def total_clovers : ℝ := 200
def percentage_three_petals : ℝ := 0.75
def percentage_two_petals : ℝ := 0.24
def earnings : ℝ := 554 -- cents

theorem percentage_four_petals :
  (total_clovers - (percentage_three_petals * total_clovers + percentage_two_petals * total_clovers)) / total_clovers * 100 = 1 := 
by sorry

end percentage_four_petals_l112_112370


namespace prism_cubes_paint_condition_l112_112285

theorem prism_cubes_paint_condition
  (m n r : ℕ)
  (h1 : m ≤ n)
  (h2 : n ≤ r)
  (h3 : (m - 2) * (n - 2) * (r - 2)
        - 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2)) 
        + 4 * (m - 2 + n - 2 + r - 2)
        = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := 
sorry

end prism_cubes_paint_condition_l112_112285


namespace intersection_on_circle_l112_112090

def parabola1 (X : ℝ) : ℝ := X^2 + X - 41
def parabola2 (Y : ℝ) : ℝ := Y^2 + Y - 40

theorem intersection_on_circle (X Y : ℝ) :
  parabola1 X = Y ∧ parabola2 Y = X → X^2 + Y^2 = 81 :=
by {
  sorry
}

end intersection_on_circle_l112_112090


namespace colton_stickers_left_l112_112168

theorem colton_stickers_left :
  let C := 72
  let F := 4 * 3 -- stickers given to three friends
  let M := F + 2 -- stickers given to Mandy
  let J := M - 10 -- stickers given to Justin
  let T := F + M + J -- total stickers given away
  C - T = 42 := by
  sorry

end colton_stickers_left_l112_112168


namespace total_marbles_l112_112731

theorem total_marbles (p y u : ℕ) :
  y + u = 10 →
  p + u = 12 →
  p + y = 6 →
  p + y + u = 14 :=
by
  intros h1 h2 h3
  sorry

end total_marbles_l112_112731


namespace obtuse_is_second_quadrant_l112_112660

-- Define the boundaries for an obtuse angle.
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define the second quadrant condition.
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The proof problem: Prove that an obtuse angle is a second quadrant angle.
theorem obtuse_is_second_quadrant (θ : ℝ) : is_obtuse θ → is_second_quadrant θ :=
by
  intro h
  sorry

end obtuse_is_second_quadrant_l112_112660


namespace number_of_solutions_in_positive_integers_l112_112522

theorem number_of_solutions_in_positive_integers (x y : ℕ) (h1 : 3 * x + 4 * y = 806) : 
  ∃ n : ℕ, n = 67 := 
sorry

end number_of_solutions_in_positive_integers_l112_112522


namespace base7_sum_correct_l112_112745

theorem base7_sum_correct : 
  ∃ (A B C : ℕ), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A = 2 ∨ A = 3 ∨ A = 5) ∧
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧
  A + B + C = 16 :=
by
  sorry

end base7_sum_correct_l112_112745


namespace percent_of_a_is_4b_l112_112721

variable (a b : ℝ)
variable (h : a = 1.2 * b)

theorem percent_of_a_is_4b :
  (4 * b) = (10 / 3 * 100 * a) / 100 :=
by sorry

end percent_of_a_is_4b_l112_112721


namespace solution_set_of_inequality_l112_112523

theorem solution_set_of_inequality (x : ℝ) :  (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
by
  sorry

end solution_set_of_inequality_l112_112523


namespace radical_axis_of_non_concentric_circles_l112_112750

theorem radical_axis_of_non_concentric_circles 
  {a R1 R2 : ℝ} (a_pos : a ≠ 0) (R1_pos : R1 > 0) (R2_pos : R2 > 0) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ x = (R2^2 - R1^2) / (4 * a) :=
by sorry

end radical_axis_of_non_concentric_circles_l112_112750


namespace elizabeth_stickers_l112_112179

def initial_bottles := 10
def lost_at_school := 2
def lost_at_practice := 1
def stickers_per_bottle := 3

def total_remaining_bottles := initial_bottles - lost_at_school - lost_at_practice
def total_stickers := total_remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers : total_stickers = 21 :=
  by
  unfold total_stickers total_remaining_bottles initial_bottles lost_at_school lost_at_practice stickers_per_bottle
  simp
  sorry

end elizabeth_stickers_l112_112179


namespace number_of_proper_subsets_of_set_l112_112899

open Finset

theorem number_of_proper_subsets_of_set {α : Type} (s : Finset α) (h : s = {0, 2, 3}) :
  (∑ k in range s.card, (s.card.choose k)) = 7 :=
by
  rw h
  simp
  sorry

end number_of_proper_subsets_of_set_l112_112899


namespace range_of_a_l112_112601

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 1) * x + a ≤ 0 → -4 ≤ x ∧ x ≤ 3) ↔ (-4 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l112_112601


namespace union_sets_l112_112073

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l112_112073


namespace water_bottles_needed_l112_112961

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l112_112961


namespace cooking_competition_probability_l112_112987

theorem cooking_competition_probability :
  let n := 8
  let f := 4
  let k := 2
  let total_pairs := Nat.choose n k
  let female_pairs := Nat.choose f k
  (female_pairs / total_pairs : ℚ) = 3 / 14 := by
  -- just the statement
  sorry

end cooking_competition_probability_l112_112987


namespace positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l112_112640

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l112_112640


namespace solution_of_inequality_l112_112983

theorem solution_of_inequality (a b : ℝ) (h : ∀ x : ℝ, (1 < x ∧ x < 3) ↔ (x^2 < a * x + b)) :
  b^a = 81 := 
sorry

end solution_of_inequality_l112_112983


namespace remainder_of_55_power_55_plus_55_l112_112979

-- Define the problem statement using Lean

theorem remainder_of_55_power_55_plus_55 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  sorry

end remainder_of_55_power_55_plus_55_l112_112979


namespace race_distance_l112_112608

theorem race_distance (a b c d : ℝ) 
  (h₁ : d / a = (d - 25) / b)
  (h₂ : d / b = (d - 15) / c)
  (h₃ : d / a = (d - 37) / c) : 
  d = 125 :=
by
  sorry

end race_distance_l112_112608


namespace line_through_M_intersects_lines_l112_112287

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def line1 (t : ℝ) : Point3D :=
  {x := 2 - t, y := 3, z := -2 + t}

def plane1 (p : Point3D) : Prop :=
  2 * p.x - 2 * p.y - p.z - 4 = 0

def plane2 (p : Point3D) : Prop :=
  p.x + 3 * p.y + 2 * p.z + 1 = 0

def param_eq (t : ℝ) : Point3D :=
  {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t}

theorem line_through_M_intersects_lines : 
  ∀ (t : ℝ), plane1 (param_eq t) ∧ plane2 (param_eq t) -> 
  ∃ t, param_eq t = {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t} :=
by
  intros t h
  sorry

end line_through_M_intersects_lines_l112_112287


namespace area_of_FDBG_l112_112737

noncomputable def area_quadrilateral (AB AC : ℝ) (area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let sin_A := (2 * area_ABC) / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sin_A
  let BC := (2 * area_ABC) / (AC * sin_A)
  let GC := BC / 3
  let area_AGC := (1 / 2) * AC * GC * sin_A
  area_ABC - (area_ADE + area_AGC)

theorem area_of_FDBG (AB AC : ℝ) (area_ABC : ℝ)
  (h1 : AB = 30)
  (h2 : AC = 15) 
  (h3 : area_ABC = 90) :
  area_quadrilateral AB AC area_ABC = 37.5 :=
by
  intros
  sorry

end area_of_FDBG_l112_112737


namespace unique_solution_set_l112_112212

theorem unique_solution_set :
  {a : ℝ | ∃! x : ℝ, (x^2 - 4) / (x + a) = 1} = { -17 / 4, -2, 2 } :=
by sorry

end unique_solution_set_l112_112212


namespace naomi_drives_to_parlor_l112_112624

theorem naomi_drives_to_parlor (d v t t_back : ℝ)
  (ht : t = d / v)
  (ht_back : t_back = 2 * d / v)
  (h_total : 2 * (t + t_back) = 6) : 
  t = 1 :=
by sorry

end naomi_drives_to_parlor_l112_112624


namespace radius_of_inscribed_circle_l112_112866

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (triangle : Triangle A B C)

-- Given conditions
def AC : ℝ := 24
def BC : ℝ := 10
def AB : ℝ := 26

-- Statement to be proved
theorem radius_of_inscribed_circle (hAC : triangle.side_length A C = AC)
                                   (hBC : triangle.side_length B C = BC)
                                   (hAB : triangle.side_length A B = AB) :
  triangle.incircle_radius = 4 :=
by sorry

end radius_of_inscribed_circle_l112_112866


namespace eval_complex_div_l112_112336

theorem eval_complex_div : 
  (i / (Real.sqrt 7 + 3 * I) = (3 / 16) + (Real.sqrt 7 / 16) * I) := 
by 
  sorry

end eval_complex_div_l112_112336


namespace middle_schoolers_count_l112_112734

theorem middle_schoolers_count (total_students : ℕ) (fraction_girls : ℚ) 
  (primary_girls_fraction : ℚ) (primary_boys_fraction : ℚ) 
  (num_girls : ℕ) (num_boys: ℕ) (primary_grade_girls : ℕ) 
  (primary_grade_boys : ℕ) :
  total_students = 800 →
  fraction_girls = 5 / 8 →
  primary_girls_fraction = 7 / 10 →
  primary_boys_fraction = 2 / 5 →
  num_girls = fraction_girls * total_students →
  num_boys = total_students - num_girls →
  primary_grade_girls = primary_girls_fraction * num_girls →
  primary_grade_boys = primary_boys_fraction * num_boys →
  total_students - (primary_grade_girls + primary_grade_boys) = 330 :=
by
  intros
  sorry

end middle_schoolers_count_l112_112734


namespace perimeter_of_first_square_l112_112397

theorem perimeter_of_first_square (p1 p2 p3 : ℕ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24) :
  p1 = 40 := 
  sorry

end perimeter_of_first_square_l112_112397


namespace combined_selling_price_l112_112505

theorem combined_selling_price :
  let cost_price_A := 180
  let profit_percent_A := 0.15
  let cost_price_B := 220
  let profit_percent_B := 0.20
  let cost_price_C := 130
  let profit_percent_C := 0.25
  let selling_price_A := cost_price_A * (1 + profit_percent_A)
  let selling_price_B := cost_price_B * (1 + profit_percent_B)
  let selling_price_C := cost_price_C * (1 + profit_percent_C)
  selling_price_A + selling_price_B + selling_price_C = 633.50 := by
  sorry

end combined_selling_price_l112_112505


namespace roll_two_dice_prime_sum_l112_112890

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end roll_two_dice_prime_sum_l112_112890


namespace find_a6_l112_112040

variable (a_n : ℕ → ℤ) (d : ℤ)

-- Conditions
axiom sum_first_three_terms (S3 : a_n 1 + a_n 2 + a_n 3 = 168)
axiom diff_terms (diff_a2_a5 : a_n 2 - a_n 5 = 42)

-- Definition of arithmetic progression 
def arith_prog (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

-- Proving that a6 = 3
theorem find_a6 (a1 : ℤ) (proof_S3 : a1 + (a1 + d) + (a1 + 2*d) = 168)
  (proof_diff : (a1 + d) - (a1 + 4*d) = 42) : a1 + 5*d = 3 :=
by
  sorry

end find_a6_l112_112040


namespace kevin_prizes_l112_112616

theorem kevin_prizes (total_prizes stuffed_animals yo_yos frisbees : ℕ)
  (h1 : total_prizes = 50) (h2 : stuffed_animals = 14) (h3 : yo_yos = 18) :
  frisbees = total_prizes - (stuffed_animals + yo_yos) → frisbees = 18 :=
by
  intro h4
  sorry

end kevin_prizes_l112_112616


namespace limit_of_derivative_squared_plus_cube_eq_zero_l112_112662

open Filter

variable (f : ℝ → ℝ)

theorem limit_of_derivative_squared_plus_cube_eq_zero 
  (h_deriv : Continuous f') 
  (h_limit : Tendsto (λ x, (f' x)^2 + (f x)^3) atTop (𝓝 0)) :
  Tendsto f atTop (𝓝 0) ∧ Tendsto (λ x, f' x) atTop (𝓝 0) := 
sorry

end limit_of_derivative_squared_plus_cube_eq_zero_l112_112662


namespace right_triangle_hypotenuse_equals_area_l112_112186

/-- Given a right triangle where the hypotenuse is equal to the area, 
    show that the scaling factor x satisfies the equation. -/
theorem right_triangle_hypotenuse_equals_area 
  (m n x : ℝ) (h_hyp: (m^2 + n^2) * x = mn * (m^2 - n^2) * x^2) :
  x = (m^2 + n^2) / (mn * (m^2 - n^2)) := 
by
  sorry

end right_triangle_hypotenuse_equals_area_l112_112186


namespace find_x_l112_112545

theorem find_x
  (x : ℝ)
  (h : 0.20 * x = 0.40 * 140 + 80) :
  x = 680 := 
sorry

end find_x_l112_112545


namespace ed_total_pets_l112_112826

theorem ed_total_pets (num_dogs num_cats : ℕ) (h_dogs : num_dogs = 2) (h_cats : num_cats = 3) :
  ∃ num_fish : ℕ, (num_fish = 2 * (num_dogs + num_cats)) ∧ (num_dogs + num_cats + num_fish) = 15 :=
by
  sorry

end ed_total_pets_l112_112826


namespace sum_of_squares_of_roots_l112_112398

theorem sum_of_squares_of_roots :
  ∃ x1 x2 : ℝ, (10 * x1 ^ 2 + 15 * x1 - 20 = 0) ∧ (10 * x2 ^ 2 + 15 * x2 - 20 = 0) ∧ (x1 ≠ x2) ∧ x1^2 + x2^2 = 25/4 :=
sorry

end sum_of_squares_of_roots_l112_112398


namespace min_value_of_expression_l112_112576

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 4 * x + 1 / x ^ 6 ≥ 5 :=
sorry

end min_value_of_expression_l112_112576


namespace jar_size_is_half_gallon_l112_112741

theorem jar_size_is_half_gallon : 
  ∃ (x : ℝ), (48 = 3 * 16) ∧ (16 + 16 * x + 16 * 0.25 = 28) ∧ x = 0.5 :=
by
  -- Implementation goes here
  sorry

end jar_size_is_half_gallon_l112_112741


namespace lambda_property_l112_112877
open Int

noncomputable def lambda : ℝ := 1 + Real.sqrt 2

theorem lambda_property (n : ℕ) (hn : n > 0) :
  2 * ⌊lambda * n⌋ = 1 - n + ⌊lambda * ⌊lambda * n⌋⌋ :=
sorry

end lambda_property_l112_112877


namespace product_not_power_of_two_l112_112759

theorem product_not_power_of_two (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, (36 * a + b) * (a + 36 * b) ≠ 2^k :=
by
  sorry

end product_not_power_of_two_l112_112759


namespace quadratic_complete_square_l112_112399

theorem quadratic_complete_square:
  ∃ (a b c : ℝ), (∀ (x : ℝ), 3 * x^2 + 9 * x - 81 = a * (x + b) * (x + b) + c) ∧ a + b + c = -83.25 :=
by {
  sorry
}

end quadratic_complete_square_l112_112399


namespace appliance_costs_l112_112671

theorem appliance_costs (a b : ℕ) 
  (h1 : a + 2 * b = 2300) 
  (h2 : 2 * a + b = 2050) : 
  a = 600 ∧ b = 850 := 
by 
  sorry

end appliance_costs_l112_112671


namespace arithmetic_question_l112_112947

theorem arithmetic_question :
  ((3.25 - 1.57) * 2) = 3.36 :=
by 
  sorry

end arithmetic_question_l112_112947


namespace unique_x_floor_eq_20_7_l112_112011

theorem unique_x_floor_eq_20_7 : ∀ x : ℝ, (⌊x⌋ + x + 1/2 = 20.7) → x = 10.2 :=
by
  sorry

end unique_x_floor_eq_20_7_l112_112011


namespace arun_borrowed_amount_l112_112928

theorem arun_borrowed_amount :
  ∃ P : ℝ, 
    (P * 0.08 * 4 + P * 0.10 * 6 + P * 0.12 * 5 = 12160) → P = 8000 :=
sorry

end arun_borrowed_amount_l112_112928


namespace minimum_m_value_l112_112356

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_m_value :
  (∃ m, ∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m) ∧
  ∀ m', (∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m') → 3 + Real.sqrt 3 / 2 ≤ m' :=
by
  sorry

end minimum_m_value_l112_112356


namespace b_geometric_l112_112348

def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

axiom a1 : a 1 = 1
axiom a_n_recurrence (n : ℕ) : a n + a (n + 1) = 1 / (3^n)
axiom b_def (n : ℕ) : b n = 3^(n - 1) * a n - 1/4

theorem b_geometric (n : ℕ) : b (n + 1) = -3 * b n := sorry

end b_geometric_l112_112348


namespace min_value_of_sum_of_powers_l112_112030

theorem min_value_of_sum_of_powers (x y : ℝ) (h : x + 3 * y = 1) : 
  2^x + 8^y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_sum_of_powers_l112_112030


namespace correct_options_l112_112789

variables (X Y ξ η : ℝ)

def follows_two_point_distribution (X : ℝ) : Prop :=
  P (X = 1) = 1/2

def variance_Y : Prop :=
  variance Y = 3

def follows_binomial_distribution (ξ : ℝ) : Prop :=
  ∃ n p, n = 4 ∧ p = 1/3 ∧ P(ξ = 3) = 32/81

def follows_normal_distribution (η : ℝ) (σ : ℝ) : Prop :=
  ∃ σ, P (η < 2) = 0.82 ∧ σ^2 ≥ 0

theorem correct_options :
  (follows_two_point_distribution X → E[X] = 1/2) ∧
  (variance_Y → variance (2 * Y + 1) = 12) ∧
  (follows_binomial_distribution ξ → P(ξ = 3) = 8/81) ∧
  (follows_normal_distribution η σ → P(0 < η ∧ η < 2) = 0.64) :=
by sorry

end correct_options_l112_112789


namespace product_of_four_consecutive_naturals_is_square_l112_112382

theorem product_of_four_consecutive_naturals_is_square (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2) := 
by
  sorry

end product_of_four_consecutive_naturals_is_square_l112_112382


namespace consecutive_integers_sum_l112_112105

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end consecutive_integers_sum_l112_112105


namespace ninety_eight_squared_l112_112325

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l112_112325


namespace permutation_and_combination_results_l112_112008

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem permutation_and_combination_results :
  A 5 2 = 20 ∧ C 6 3 + C 6 4 = 35 := by
  sorry

end permutation_and_combination_results_l112_112008


namespace abs_x_equals_4_l112_112081

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l112_112081


namespace triangle_angle_A_l112_112226

theorem triangle_angle_A (A B a b : ℝ) (h1 : b = 2 * a) (h2 : B = A + 60) : A = 30 :=
by sorry

end triangle_angle_A_l112_112226


namespace hannah_dogs_food_total_l112_112470

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end hannah_dogs_food_total_l112_112470


namespace min_value_frac_l112_112490

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end min_value_frac_l112_112490


namespace total_growth_of_trees_l112_112648

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end total_growth_of_trees_l112_112648


namespace baskets_weight_l112_112174

theorem baskets_weight 
  (weight_per_basket : ℕ)
  (num_baskets : ℕ)
  (total_weight : ℕ) 
  (h1 : weight_per_basket = 30)
  (h2 : num_baskets = 8)
  (h3 : total_weight = weight_per_basket * num_baskets) :
  total_weight = 240 := 
by
  sorry

end baskets_weight_l112_112174


namespace eric_has_correct_green_marbles_l112_112184

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end eric_has_correct_green_marbles_l112_112184


namespace right_triangle_properties_l112_112267

theorem right_triangle_properties (a b c : ℝ) (h1 : c = 13) (h2 : a = 5)
  (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 ∧ a + b + c = 30 := by
  sorry

end right_triangle_properties_l112_112267


namespace squares_difference_l112_112563

theorem squares_difference :
  1010^2 - 994^2 - 1008^2 + 996^2 = 8016 :=
by
  sorry

end squares_difference_l112_112563


namespace find_other_x_intercept_l112_112344

theorem find_other_x_intercept (a b c : ℝ) (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9)
  (h2 : a * 0^2 + b * 0 + c = 0) : ∃ x, x ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=
by
  sorry

end find_other_x_intercept_l112_112344


namespace scientific_notation_of_1653_billion_l112_112147

theorem scientific_notation_of_1653_billion :
  (1653 * (10 ^ 9) = 1.6553 * (10 ^ 12)) :=
sorry

end scientific_notation_of_1653_billion_l112_112147


namespace max_expression_value_l112_112049

noncomputable def A : ℝ := 15682 + (1 / 3579)
noncomputable def B : ℝ := 15682 - (1 / 3579)
noncomputable def C : ℝ := 15682 * (1 / 3579)
noncomputable def D : ℝ := 15682 / (1 / 3579)
noncomputable def E : ℝ := 15682.3579

theorem max_expression_value :
  D = 56109138 ∧ D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end max_expression_value_l112_112049


namespace HCF_of_two_numbers_l112_112652

theorem HCF_of_two_numbers (H L : ℕ) (product : ℕ) (h1 : product = 2560) (h2 : L = 128)
  (h3 : H * L = product) : H = 20 := by {
  -- The proof goes here.
  sorry
}

end HCF_of_two_numbers_l112_112652


namespace janet_initial_stickers_l112_112995

variable (x : ℕ)

theorem janet_initial_stickers (h : x + 53 = 56) : x = 3 := by
  sorry

end janet_initial_stickers_l112_112995


namespace expected_lifetime_flashlight_l112_112243

noncomputable theory

variables (ξ η : ℝ) -- ξ and η are continuous random variables representing the lifetimes
variables (T : ℝ) -- T is the lifetime of the flashlight

-- Define the maximum lifetime of the flashlight
def max_lifetime (ξ η : ℝ) : ℝ := max ξ η

-- Given condition: the expectation of η is 4
axiom expectation_eta : E η = 4

-- Theorem statement: expected lifetime of the flashlight is at least 4
theorem expected_lifetime_flashlight (ξ η : ℝ) (h : T = max_lifetime ξ η) : 
  E (max_lifetime ξ η) ≥ 4 :=
by 
  sorry

end expected_lifetime_flashlight_l112_112243


namespace lambda_range_l112_112748

noncomputable def lambda (S1 S2 S3 S4: ℝ) (S: ℝ) : ℝ :=
  4 * (S1 + S2 + S3 + S4) / S

theorem lambda_range (S1 S2 S3 S4: ℝ) (S: ℝ) (h_max: S = max (max S1 S2) (max S3 S4)) :
  2 < lambda S1 S2 S3 S4 S ∧ lambda S1 S2 S3 S4 S ≤ 4 :=
by
  sorry

end lambda_range_l112_112748


namespace expected_lifetime_flashlight_l112_112244

noncomputable section

variables (ξ η : ℝ) -- lifetimes of the blue and red lightbulbs
variables [probability_space ℙ] -- assuming a probability space ℙ

-- condition: expected lifetime of the red lightbulb is 4 years
axiom expected_eta : ℙ.𝔼(η) = 4

-- the main proof problem
theorem expected_lifetime_flashlight : ℙ.𝔼(max ξ η) ≥ 4 :=
sorry

end expected_lifetime_flashlight_l112_112244


namespace percentage_decrease_l112_112641

theorem percentage_decrease (P : ℝ) (new_price : ℝ) (x : ℝ) (h1 : new_price = 320) (h2 : P = 421.05263157894734) : x = 24 :=
by
  sorry

end percentage_decrease_l112_112641


namespace polynomial_identity_l112_112360

theorem polynomial_identity (x : ℝ) (hx : x^2 + x - 1 = 0) : x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 :=
sorry

end polynomial_identity_l112_112360


namespace milk_leftover_after_milkshakes_l112_112438

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end milk_leftover_after_milkshakes_l112_112438


namespace op_dot_of_10_5_l112_112900

-- Define the operation \odot
def op_dot (a b : ℕ) : ℕ := a + (2 * a) / b

-- Theorem stating that 10 \odot 5 = 14
theorem op_dot_of_10_5 : op_dot 10 5 = 14 :=
by
  sorry

end op_dot_of_10_5_l112_112900


namespace cube_sum_identity_l112_112633

theorem cube_sum_identity (p q r : ℝ)
  (h₁ : p + q + r = 4)
  (h₂ : pq + qr + rp = 6)
  (h₃ : pqr = -8) :
  p^3 + q^3 + r^3 = 64 := 
by
  sorry

end cube_sum_identity_l112_112633


namespace value_of_question_l112_112930

noncomputable def value_of_approx : ℝ := 0.2127541038062284

theorem value_of_question :
  ((0.76^3 - 0.1^3) / (0.76^2) + value_of_approx + 0.1^2) = 0.66 :=
by
  sorry

end value_of_question_l112_112930


namespace cost_of_one_dozen_pens_l112_112389

-- Define the initial conditions
def cost_pen : ℕ := 65
def cost_pencil := cost_pen / 5
def total_cost (pencils : ℕ) := 3 * cost_pen + pencils * cost_pencil

-- State the theorem
theorem cost_of_one_dozen_pens (pencils : ℕ) (h : total_cost pencils = 260) :
  12 * cost_pen = 780 :=
by
  -- Preamble to show/conclude that the proofs are given
  sorry

end cost_of_one_dozen_pens_l112_112389


namespace minimize_sum_of_cubes_l112_112172

theorem minimize_sum_of_cubes (x y : ℝ) (h : x + y = 8) : 
  (3 * x^2 - 3 * (8 - x)^2 = 0) → (x = 4) ∧ (y = 4) :=
by
  sorry

end minimize_sum_of_cubes_l112_112172


namespace athlete_heartbeats_during_race_l112_112148

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l112_112148


namespace number_of_valid_pairs_is_34_l112_112499

noncomputable def countValidPairs : Nat :=
  let primes : List Nat := [2, 3, 5, 7, 11, 13]
  let nonprimes : List Nat := [1, 4, 6, 8, 9, 10, 12, 14, 15]
  let countForN (n : Nat) : Nat :=
    match n with
    | 2 => Nat.choose 8 1
    | 3 => Nat.choose 7 2
    | 5 => Nat.choose 5 4
    | _ => 0
  primes.map countForN |>.sum

theorem number_of_valid_pairs_is_34 : countValidPairs = 34 :=
  sorry

end number_of_valid_pairs_is_34_l112_112499


namespace max_area_triangle_PAB_l112_112010

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 16) + (y^2 / 9) = 1

def point_A : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)

theorem max_area_triangle_PAB (P : ℝ × ℝ) (hP : ellipse_eq P.1 P.2) : 
  ∃ S, S = 6 * (sqrt 2 + 1) := 
sorry

end max_area_triangle_PAB_l112_112010


namespace james_writing_hours_per_week_l112_112235

variables (pages_per_hour : ℕ) (pages_per_day_per_person : ℕ) (people : ℕ) (days_per_week : ℕ)

theorem james_writing_hours_per_week
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_day_per_person = 5)
  (h3 : people = 2)
  (h4 : days_per_week = 7) :
  (pages_per_day_per_person * people * days_per_week) / pages_per_hour = 7 :=
by
  sorry

end james_writing_hours_per_week_l112_112235


namespace ellie_total_distance_after_six_steps_l112_112015

-- Define the initial conditions and parameters
def initial_position : ℚ := 0
def target_distance : ℚ := 5
def step_fraction : ℚ := 1 / 4
def steps : ℕ := 6

-- Define the function that calculates the sum of the distances walked
def distance_walked (n : ℕ) : ℚ :=
  let first_term := target_distance * step_fraction
  let common_ratio := 3 / 4
  first_term * (1 - common_ratio^n) / (1 - common_ratio)

-- Define the theorem we want to prove
theorem ellie_total_distance_after_six_steps :
  distance_walked steps = 16835 / 4096 :=
by 
  sorry

end ellie_total_distance_after_six_steps_l112_112015


namespace Intersect_A_B_l112_112972

-- Defining the sets A and B according to the problem's conditions
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x ∈ Set.univ | x^2 - 5*x + 4 < 0}

-- Prove that the intersection of A and B is {2}
theorem Intersect_A_B : A ∩ B = {2} := by
  sorry

end Intersect_A_B_l112_112972


namespace shaded_region_area_computed_correctly_l112_112806

noncomputable def side_length : ℝ := 15
noncomputable def quarter_circle_radius : ℝ := side_length / 3
noncomputable def square_area : ℝ := side_length ^ 2
noncomputable def circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2)
noncomputable def shaded_region_area : ℝ := square_area - circle_area

theorem shaded_region_area_computed_correctly : 
  shaded_region_area = 225 - 25 * Real.pi := 
by 
  -- This statement only defines the proof problem.
  sorry

end shaded_region_area_computed_correctly_l112_112806


namespace inequality_always_true_l112_112658

theorem inequality_always_true (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by sorry

end inequality_always_true_l112_112658


namespace base8_subtraction_l112_112337

theorem base8_subtraction : (325 : Nat) - (237 : Nat) = 66 :=
by 
  sorry

end base8_subtraction_l112_112337


namespace right_triangular_prism_volume_l112_112843

theorem right_triangular_prism_volume (R a h V : ℝ)
  (h1 : 4 * Real.pi * R^2 = 12 * Real.pi)
  (h2 : h = 2 * R)
  (h3 : (1 / 3) * (Real.sqrt 3 / 2) * a = R)
  (h4 : V = (1 / 2) * a * a * (Real.sin (Real.pi / 3)) * h) :
  V = 54 :=
by sorry

end right_triangular_prism_volume_l112_112843


namespace cone_volume_l112_112975

theorem cone_volume (r l: ℝ) (r_eq : r = 2) (l_eq : l = 4) (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) :
  (1 / 3) * π * r^2 * h = (8 * Real.sqrt 3 * π) / 3 :=
by
  -- Sorry to skip the proof
  sorry

end cone_volume_l112_112975


namespace garden_length_to_width_ratio_l112_112520

theorem garden_length_to_width_ratio (area : ℕ) (width : ℕ) (h_area : area = 432) (h_width : width = 12) :
  ∃ length : ℕ, length = area / width ∧ (length / width = 3) := 
by
  sorry

end garden_length_to_width_ratio_l112_112520


namespace range_of_m_l112_112844

open Set

theorem range_of_m (m : ℝ) : 
  (∀ x, (m + 1 ≤ x ∧ x ≤ 2 * m - 1) → (-2 < x ∧ x ≤ 5)) → 
  m ∈ Iic (3 : ℝ) :=
by
  intros h
  sorry

end range_of_m_l112_112844


namespace distinct_roots_of_cubic_l112_112270

noncomputable def roots_of_cubic : (ℚ × ℚ × ℚ) :=
  let a := 1
  let b := -2
  let c := 0
  (a, b, c)

theorem distinct_roots_of_cubic :
  ∃ a b c : ℚ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ roots_of_cubic = (a, b, c) := 
by
  use 1, -2, 0
  split; linarith
  split; linarith
  split; linarith
  dsimp only [roots_of_cubic]
  refl

end distinct_roots_of_cubic_l112_112270


namespace xy_value_l112_112724

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := 
by 
  sorry

end xy_value_l112_112724


namespace prob_three_friends_same_group_l112_112164

theorem prob_three_friends_same_group :
  let students := 800
  let groups := 4
  let group_size := students / groups
  let p_same_group := 1 / groups
  p_same_group * p_same_group = 1 / 16 := 
by
  sorry

end prob_three_friends_same_group_l112_112164


namespace find_value_l112_112059

variables {p q s u : ℚ}

theorem find_value
  (h1 : p / q = 5 / 6)
  (h2 : s / u = 7 / 15) :
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 :=
sorry

end find_value_l112_112059


namespace find_range_of_m_l112_112459

noncomputable def range_of_m (m : ℝ) : Prop :=
  ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m))

theorem find_range_of_m (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∨
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3)) ↔
  ¬((∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∧
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3))) →
  range_of_m m :=
sorry

end find_range_of_m_l112_112459


namespace roll_two_dice_prime_sum_l112_112889

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end roll_two_dice_prime_sum_l112_112889


namespace si_perpendicular_zt_l112_112076

open EuclideanGeometry

theorem si_perpendicular_zt
  {A B C O I E F T Z S : Point}
  (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A)
  (hCircumcenter : Circumcenter A B C O)
  (hIncenter : Incenter A B C I)
  (hE : OrthogonalProjection I (Segment A B) E)
  (hF : OrthogonalProjection I (Segment A C) F)
  (hT : LineThrough E I ∩ LineThrough O C T)
  (hZ : LineThrough F I ∩ LineThrough O B Z)
  (hS : IntersectionOfTangentsAtBc A B C S) :
  Perpendicular (LineThrough S I) (LineThrough Z T) := by
  sorry

end si_perpendicular_zt_l112_112076


namespace work_days_difference_l112_112295

theorem work_days_difference (d_a d_b : ℕ) (H1 : d_b = 15) (H2 : d_a = d_b / 3) : 15 - d_a = 10 := by
  sorry

end work_days_difference_l112_112295


namespace no_real_roots_of_f_l112_112269

def f (x : ℝ) : ℝ := (x + 1) * |x + 1| - x * |x| + 1

theorem no_real_roots_of_f :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end no_real_roots_of_f_l112_112269


namespace solve_quadratic_1_solve_quadratic_2_l112_112386

theorem solve_quadratic_1 (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by sorry

theorem solve_quadratic_2 (x : ℝ) : x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l112_112386


namespace total_promotional_items_l112_112939

def num_calendars : ℕ := 300
def num_date_books : ℕ := 200

theorem total_promotional_items : num_calendars + num_date_books = 500 := by
  sorry

end total_promotional_items_l112_112939


namespace expected_lifetime_flashlight_l112_112242

noncomputable def xi : ℝ := sorry
noncomputable def eta : ℝ := sorry

def T : ℝ := max xi eta

axiom E_eta_eq_4 : E eta = 4

theorem expected_lifetime_flashlight : E T ≥ 4 :=
by
  -- The solution will go here
  sorry

end expected_lifetime_flashlight_l112_112242


namespace minimum_time_for_tomato_egg_soup_l112_112100

noncomputable def cracking_egg_time : ℕ := 1
noncomputable def washing_chopping_tomatoes_time : ℕ := 2
noncomputable def boiling_tomatoes_time : ℕ := 3
noncomputable def adding_eggs_heating_time : ℕ := 1
noncomputable def stirring_egg_time : ℕ := 1

theorem minimum_time_for_tomato_egg_soup :
  washing_chopping_tomatoes_time + boiling_tomatoes_time + adding_eggs_heating_time = 6 :=
by
  -- proof to be filled
  sorry

end minimum_time_for_tomato_egg_soup_l112_112100


namespace min_value_of_fractions_l112_112746

theorem min_value_of_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    (a+b)/(c+d) + (a+c)/(b+d) + (a+d)/(b+c) + (b+c)/(a+d) + (b+d)/(a+c) + (c+d)/(a+b) ≥ 6 :=
by
  sorry

end min_value_of_fractions_l112_112746


namespace exists_seq_length_at_most_c_log_l112_112628

open Group

theorem exists_seq_length_at_most_c_log (G : Group) [Finite G] (hG : 1 < |G|) :
  ∃ c > 0, ∀ x ∈ G, ∃ L : List G, (L.length ≤ c * Real.log (|G|)) ∧ (∃ L' ⊆ L, List.prod L' = x) :=
by
  sorry

end exists_seq_length_at_most_c_log_l112_112628


namespace joshua_final_bottle_caps_l112_112493

def initial_bottle_caps : ℕ := 150
def bought_bottle_caps : ℕ := 23
def given_away_bottle_caps : ℕ := 37

theorem joshua_final_bottle_caps : (initial_bottle_caps + bought_bottle_caps - given_away_bottle_caps) = 136 := by
  sorry

end joshua_final_bottle_caps_l112_112493


namespace initial_matches_l112_112636

theorem initial_matches (x : ℕ) (h1 : (34 * x + 89) / (x + 1) = 39) : x = 10 := by
  sorry

end initial_matches_l112_112636


namespace solve_inequality_l112_112259

theorem solve_inequality (x : ℝ) :
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5 / 3 :=
by
  sorry

end solve_inequality_l112_112259


namespace equal_piece_length_l112_112134

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l112_112134


namespace number_of_ways_to_place_rooks_l112_112869

theorem number_of_ways_to_place_rooks :
  let columns := 6
  let rows := 2006
  let rooks := 3
  ((Nat.choose columns rooks) * (rows * (rows - 1) * (rows - 2))) = 20 * 2006 * 2005 * 2004 :=
by {
  sorry
}

end number_of_ways_to_place_rooks_l112_112869


namespace value_of_3x_plus_5y_l112_112602

variable (x y : ℚ)

theorem value_of_3x_plus_5y
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 3 * x + 5 * y = 6 := 
sorry

end value_of_3x_plus_5y_l112_112602


namespace largest_mersenne_prime_less_than_500_l112_112574

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_than_500 : ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p :=
sorry

end largest_mersenne_prime_less_than_500_l112_112574


namespace probability_prime_sum_two_dice_l112_112891

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def num_ways_sum_is (target_sum : ℕ) : ℕ :=
  finset.card { (a, b) : finset.univ × finset.univ | a + b = target_sum }

def total_outcomes : ℕ := 36

theorem probability_prime_sum_two_dice :
  (∑ n in (finset.range 13).filter is_prime, num_ways_sum_is n) / total_outcomes = 5 / 12 :=
sorry

end probability_prime_sum_two_dice_l112_112891


namespace johns_allowance_is_3_45_l112_112537

noncomputable def johns_weekly_allowance (A : ℝ) : Prop :=
  -- Condition 1: John spent 3/5 of his allowance at the arcade
  let spent_at_arcade := (3/5) * A
  -- Remaining allowance
  let remaining_after_arcade := A - spent_at_arcade
  -- Condition 2: He spent 1/3 of the remaining allowance at the toy store
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  -- Condition 3: He spent his last $0.92 at the candy store
  let spent_at_candy_store := 0.92
  -- Remaining amount after the candy store expenditure should be 0
  remaining_after_toy_store = spent_at_candy_store

theorem johns_allowance_is_3_45 : johns_weekly_allowance 3.45 :=
sorry

end johns_allowance_is_3_45_l112_112537


namespace elizabeth_stickers_l112_112177

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l112_112177


namespace diagonal_length_count_l112_112442

theorem diagonal_length_count :
  ∃ (x : ℕ) (h : (3 < x ∧ x < 22)), x = 18 := by
    sorry

end diagonal_length_count_l112_112442


namespace find_dallas_age_l112_112874

variable (Dallas_last_year Darcy_last_year Dexter_age Darcy_this_year Derek this_year_age : ℕ)

-- Conditions
axiom cond1 : Dallas_last_year = 3 * Darcy_last_year
axiom cond2 : Darcy_this_year = 2 * Dexter_age
axiom cond3 : Dexter_age = 8
axiom cond4 : Derek = this_year_age + 4

-- Theorem: Proving Dallas's current age
theorem find_dallas_age (Dallas_last_year : ℕ)
  (H1 : Dallas_last_year = 3 * (Darcy_this_year - 1))
  (H2 : Darcy_this_year = 2 * Dexter_age)
  (H3 : Dexter_age = 8)
  (H4 : Derek = (Dallas_last_year + 1) + 4) :
  Dallas_last_year + 1 = 46 :=
by
  sorry

end find_dallas_age_l112_112874


namespace solve_equation_l112_112687

theorem solve_equation :
  ∀ x : ℝ, (3 * x^2 / (x - 2) - (3 * x + 4) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) →
    (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6) :=
by
  intro x h
  -- the proof would go here
  sorry

end solve_equation_l112_112687


namespace triangulate_colored_polygon_l112_112786

theorem triangulate_colored_polygon (n : ℕ) (polygon : polygon (2 * n + 1)) 
  (coloring : ∀ v : polygon.vertices, v.color ≠ v.next.color) :
  ∃ (triangulation : set (polygon.diagonals)), 
    (∀ d ∈ triangulation, (d.src.color ≠ d.dst.color)) ∧ non_intersecting triangulation :=
sorry

end triangulate_colored_polygon_l112_112786


namespace trapezoid_ad_length_mn_l112_112529

open EuclideanGeometry

variables {A B C D O P : Point}
variables {m n : ℕ}

-- Given conditions
def is_trapezoid (A B C D : Point) : Prop := 
  A.y = B.y ∧ C.y = D.y ∧ B.x - A.x ≠ D.x - C.x

def length_eq (x y : ℕ) : Prop := 
  x = 43 ∧ y = 43

def perpendicular (A D B : Point) : Prop := 
  (A.x - D.x) * (D.x - B.x) + (A.y - D.y) * (D.y - B.y) = 0

def midpoint (P B D : Point) : Prop := 
  2 * P.x = B.x + D.x ∧ 2 * P.y = B.y + D.y

def inter_diag (A C B D O : Point) : Prop := 
  ∃ λ : ℝ, O = λ • A + (1 - λ) • C ∧  ∃ μ : ℝ, O = μ • B + (1 - μ) • D

def OP_length (O P : Point) (l : ℝ) : Prop := 
  dist O P = l

-- Prove the final tuple
theorem trapezoid_ad_length_mn (hT : is_trapezoid A B C D) (hL : length_eq (dist B C) (dist C D))
  (hP : perpendicular A D B) (hM : midpoint P B D) (hI : inter_diag A C B D O)
  (hO : OP_length O P 11) : 
  ∃ (m n : ℕ), dist A D = m * Real.sqrt n ∧ m + n = 194 := 
sorry

end trapezoid_ad_length_mn_l112_112529


namespace arithmetic_evaluation_l112_112948

theorem arithmetic_evaluation :
  -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 :=
by
  sorry

end arithmetic_evaluation_l112_112948


namespace number_of_cubes_with_icing_on_two_sides_l112_112548

def cake_cube : ℕ := 3
def smaller_cubes : ℕ := 27
def covered_faces : ℕ := 3
def layers_with_icing : ℕ := 2
def edge_cubes_per_layer_per_face : ℕ := 2

theorem number_of_cubes_with_icing_on_two_sides :
  (covered_faces * edge_cubes_per_layer_per_face * layers_with_icing) = 12 := by
  sorry

end number_of_cubes_with_icing_on_two_sides_l112_112548


namespace k_league_teams_l112_112510

theorem k_league_teams (n : ℕ) (h : n*(n-1)/2 = 91) : n = 14 := sorry

end k_league_teams_l112_112510


namespace sqrt_four_eq_two_or_neg_two_l112_112275

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l112_112275


namespace metallic_sheet_first_dimension_l112_112304

-- Given Conditions
variable (x : ℝ) (height width : ℝ)
def metallic_sheet :=
  (x > 0) ∧ (height = 8) ∧ (width = 36 - 2 * height)

-- Volume of the resulting box should be 5760 m³
def volume_box :=
  (width - 2 * height) * (x - 2 * height) * height = 5760

-- Prove the first dimension of the metallic sheet
theorem metallic_sheet_first_dimension (h1 : metallic_sheet x height width) (h2 : volume_box x height width) : 
  x = 52 :=
  sorry

end metallic_sheet_first_dimension_l112_112304


namespace peter_speed_l112_112615

theorem peter_speed (p : ℝ) (v_juan : ℝ) (d : ℝ) (t : ℝ) 
  (h1 : v_juan = p + 3) 
  (h2 : d = t * p + t * v_juan) 
  (h3 : t = 1.5) 
  (h4 : d = 19.5) : 
  p = 5 :=
by
  sorry

end peter_speed_l112_112615


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l112_112914

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l112_112914


namespace airplane_seats_l112_112310

theorem airplane_seats (F : ℕ) (h : F + 4 * F + 2 = 387) : F = 77 := by
  -- Proof goes here
  sorry

end airplane_seats_l112_112310


namespace polygon_sides_in_arithmetic_progression_l112_112388

theorem polygon_sides_in_arithmetic_progression 
  (a : ℕ → ℝ) (n : ℕ) (h1: ∀ i, 1 ≤ i ∧ i ≤ n → a i = a 1 + (i - 1) * 10) 
  (h2 : a n = 150) : n = 12 :=
sorry

end polygon_sides_in_arithmetic_progression_l112_112388


namespace class_ratio_and_percentage_l112_112732

theorem class_ratio_and_percentage:
  ∀ (female male : ℕ), female = 15 → male = 25 →
  (∃ ratio_n ratio_d : ℕ, gcd ratio_n ratio_d = 1 ∧ ratio_n = 5 ∧ ratio_d = 8 ∧
  ratio_n / ratio_d = male / (female + male))
  ∧
  (∃ percentage : ℕ, percentage = 40 ∧ percentage = 100 * (male - female) / male) :=
by
  intros female male hf hm
  have h1 : female = 15 := hf
  have h2 : male = 25 := hm
  sorry

end class_ratio_and_percentage_l112_112732


namespace tom_average_speed_l112_112910

theorem tom_average_speed
  (total_distance : ℕ)
  (distance1 : ℕ)
  (speed1 : ℕ)
  (distance2 : ℕ)
  (speed2 : ℕ)
  (H : total_distance = distance1 + distance2)
  (H1 : distance1 = 12)
  (H2 : speed1 = 24)
  (H3 : distance2 = 48)
  (H4 : speed2 = 48) :
  (total_distance : ℚ) / ((distance1 : ℚ) / speed1 + (distance2 : ℚ) / speed2) = 40 :=
by
  sorry

end tom_average_speed_l112_112910


namespace worm_length_difference_l112_112634

def worm_1_length : ℝ := 0.8
def worm_2_length : ℝ := 0.1
def difference := worm_1_length - worm_2_length

theorem worm_length_difference : difference = 0.7 := by
  sorry

end worm_length_difference_l112_112634


namespace range_of_a_l112_112053

noncomputable def set_A (a : ℝ) : Set ℝ := {x | x < a}
noncomputable def set_B : Set ℝ := {x | 1 < x ∧ x < 2}
noncomputable def complement_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2 }

theorem range_of_a (a : ℝ) : (set_A a ∪ complement_B) = Set.univ ↔ 2 ≤ a := 
by 
  sorry

end range_of_a_l112_112053


namespace find_range_of_m_l112_112089

open Real

-- Definition for proposition p (the discriminant condition)
def real_roots (m : ℝ) : Prop := (3 * 3) - 4 * m ≥ 0

-- Definition for proposition q (ellipse with foci on x-axis conditions)
def is_ellipse (m : ℝ) : Prop := 
  9 - m > 0 ∧ 
  m - 2 > 0 ∧ 
  9 - m > m - 2

-- Lean statement for the mathematically equivalent proof problem
theorem find_range_of_m (m : ℝ) : (real_roots m ∧ is_ellipse m) → (2 < m ∧ m ≤ 9 / 4) := 
by
  sorry

end find_range_of_m_l112_112089


namespace radius_of_circle_l112_112702

noncomputable def circle_radius (x y : ℝ) : ℝ := 
  let lhs := x^2 - 8 * x + y^2 - 4 * y + 16
  if lhs = 0 then 2 else 0

theorem radius_of_circle : circle_radius 0 0 = 2 :=
sorry

end radius_of_circle_l112_112702


namespace sum_of_k_binom_eq_l112_112833

theorem sum_of_k_binom_eq :
  (∑ k in {k : ℕ | binom 23 4 + binom 23 5 = binom 24 k}, k) = 24 := 
by
  sorry

end sum_of_k_binom_eq_l112_112833


namespace find_a6_l112_112041

variable (a_n : ℕ → ℤ) (d : ℤ)

-- Conditions
axiom sum_first_three_terms (S3 : a_n 1 + a_n 2 + a_n 3 = 168)
axiom diff_terms (diff_a2_a5 : a_n 2 - a_n 5 = 42)

-- Definition of arithmetic progression 
def arith_prog (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

-- Proving that a6 = 3
theorem find_a6 (a1 : ℤ) (proof_S3 : a1 + (a1 + d) + (a1 + 2*d) = 168)
  (proof_diff : (a1 + d) - (a1 + 4*d) = 42) : a1 + 5*d = 3 :=
by
  sorry

end find_a6_l112_112041


namespace sqrt_of_225_eq_15_l112_112544

theorem sqrt_of_225_eq_15 : Real.sqrt 225 = 15 :=
by
  sorry

end sqrt_of_225_eq_15_l112_112544


namespace surface_area_of_given_cube_l112_112140

-- Define the cube with its volume
def volume_of_cube : ℝ := 4913

-- Define the side length of the cube
def side_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the surface area of the cube
def surface_area_of_cube (side : ℝ) : ℝ := 6 * (side^2)

-- Statement of the theorem
theorem surface_area_of_given_cube : 
  surface_area_of_cube side_of_cube = 1734 := 
by
  -- Proof goes here
  sorry

end surface_area_of_given_cube_l112_112140


namespace red_marbles_count_l112_112932

variable (n : ℕ)

-- Conditions
def ratio_green_yellow_red := (3 * n, 4 * n, 2 * n)
def not_red_marbles := 3 * n + 4 * n = 63

-- Goal
theorem red_marbles_count (hn : not_red_marbles n) : 2 * n = 18 :=
by
  sorry

end red_marbles_count_l112_112932


namespace tickets_spent_correct_l112_112002

/-- Tom won 32 tickets playing 'whack a mole'. -/
def tickets_whack_mole : ℕ := 32

/-- Tom won 25 tickets playing 'skee ball'. -/
def tickets_skee_ball : ℕ := 25

/-- Tom is left with 50 tickets after spending some on a hat. -/
def tickets_left : ℕ := 50

/-- The total number of tickets Tom won from both games. -/
def tickets_total : ℕ := tickets_whack_mole + tickets_skee_ball

/-- The number of tickets Tom spent on the hat. -/
def tickets_spent : ℕ := tickets_total - tickets_left

-- Prove that the number of tickets Tom spent on the hat is 7.
theorem tickets_spent_correct : tickets_spent = 7 := by
  -- Proof goes here
  sorry

end tickets_spent_correct_l112_112002


namespace parabola_focus_l112_112580

theorem parabola_focus (x y : ℝ) : (y = x^2 / 8) → (y = x^2 / 8) ∧ (∃ p, p = (0, 2)) :=
by
  sorry

end parabola_focus_l112_112580


namespace assignment_methods_l112_112777

theorem assignment_methods : 
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  (doctors * (nurses.choose nurses_per_school)) = 12 := by
  sorry

end assignment_methods_l112_112777


namespace unique_combined_friends_count_l112_112234

theorem unique_combined_friends_count 
  (james_friends : ℕ)
  (susan_friends : ℕ)
  (john_multiplier : ℕ)
  (shared_friends : ℕ)
  (maria_shared_friends : ℕ)
  (maria_friends : ℕ)
  (h_james : james_friends = 90)
  (h_susan : susan_friends = 50)
  (h_john : ∃ (john_friends : ℕ), john_friends = john_multiplier * susan_friends ∧ john_multiplier = 4)
  (h_shared : shared_friends = 35)
  (h_maria_shared : maria_shared_friends = 10)
  (h_maria : maria_friends = 80) :
  ∃ (total_unique_friends : ℕ), total_unique_friends = 325 :=
by
  -- Proof is omitted
  sorry

end unique_combined_friends_count_l112_112234


namespace new_average_weight_l112_112517

theorem new_average_weight 
  (average_weight_19 : ℕ → ℝ)
  (weight_new_student : ℕ → ℝ)
  (new_student_count : ℕ)
  (old_student_count : ℕ)
  (h1 : average_weight_19 old_student_count = 15.0)
  (h2 : weight_new_student new_student_count = 11.0)
  : (average_weight_19 (old_student_count + new_student_count) = 14.8) :=
by
  sorry

end new_average_weight_l112_112517


namespace integer_solution_unique_l112_112500

theorem integer_solution_unique (x y : ℝ) (h : -1 < (y - x) / (x + y) ∧ (y - x) / (x + y) < 2) (hyx : ∃ n : ℤ, y = n * x) : y = x :=
by
  sorry

end integer_solution_unique_l112_112500


namespace exists_univariate_polynomial_l112_112431

def polynomial_in_three_vars (P : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ,
  P x y z = P x y (x * y - z) ∧
  P x y z = P x (z * x - y) z ∧
  P x y z = P (y * z - x) y z

theorem exists_univariate_polynomial (P : ℝ → ℝ → ℝ → ℝ) (h : polynomial_in_three_vars P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x * y * z) :=
sorry

end exists_univariate_polynomial_l112_112431


namespace betty_cookies_brownies_l112_112315

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end betty_cookies_brownies_l112_112315


namespace initial_men_count_l112_112231

theorem initial_men_count
  (M : ℕ)
  (h1 : ∀ T : ℕ, (M * 8 * 10 = T) → (5 * 16 * 12 = T)) :
  M = 12 :=
by
  sorry

end initial_men_count_l112_112231


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l112_112920

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l112_112920


namespace no_pairs_satisfy_equation_l112_112720

theorem no_pairs_satisfy_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2)) → False :=
by
  sorry

end no_pairs_satisfy_equation_l112_112720


namespace ceil_mul_eq_225_l112_112966

theorem ceil_mul_eq_225 {x : ℝ} (h₁ : ⌈x⌉ * x = 225) (h₂ : x > 0) : x = 15 :=
sorry

end ceil_mul_eq_225_l112_112966


namespace roots_quadratic_expr_l112_112846

theorem roots_quadratic_expr (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0)
    (h2 : Polynomial.eval n (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0) :
  m^2 + m * n + 2 * m = 0 :=
sorry

end roots_quadratic_expr_l112_112846


namespace dot_not_line_l112_112793

variable (D S DS T : Nat)
variable (h1 : DS = 20) (h2 : S = 36) (h3 : T = 60)
variable (h4 : T = D + S - DS)

theorem dot_not_line : (D - DS) = 24 :=
by
  sorry

end dot_not_line_l112_112793


namespace acute_triangle_probability_l112_112205

open Finset

noncomputable def isAcuteTriangleProb (n : ℕ) : Prop :=
  ∃ k : ℕ, (n = 2 * k ∧ (3 * (k - 2)) / (2 * (2 * k - 1)) = 93 / 125) ∨ (n = 2 * k + 1 ∧ (3 * (k - 1)) / (2 * (2 * k - 1)) = 93 / 125)

theorem acute_triangle_probability (n : ℕ) : isAcuteTriangleProb n → n = 376 ∨ n = 127 :=
by
  sorry

end acute_triangle_probability_l112_112205


namespace computer_production_per_month_l112_112800

def days : ℕ := 28
def hours_per_day : ℕ := 24
def intervals_per_hour : ℕ := 2
def computers_per_interval : ℕ := 3

theorem computer_production_per_month : 
  (days * hours_per_day * intervals_per_hour * computers_per_interval = 4032) :=
by sorry

end computer_production_per_month_l112_112800


namespace probability_same_color_boxes_l112_112230

def num_neckties := 6
def num_shirts := 5
def num_hats := 4
def num_socks := 3

def num_common_colors := 3

def total_combinations : ℕ := num_neckties * num_shirts * num_hats * num_socks

def same_color_combinations : ℕ := num_common_colors

def same_color_probability : ℚ :=
  same_color_combinations / total_combinations

theorem probability_same_color_boxes :
  same_color_probability = 1 / 120 :=
  by
    -- Proof would go here
    sorry

end probability_same_color_boxes_l112_112230


namespace remainder_division_of_product_l112_112414

theorem remainder_division_of_product
  (h1 : 1225 % 12 = 1)
  (h2 : 1227 % 12 = 3) :
  ((1225 * 1227 * 1) % 12) = 3 :=
by
  sorry

end remainder_division_of_product_l112_112414


namespace athlete_heartbeats_during_race_l112_112151

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l112_112151


namespace prop_A_prop_B_prop_C_prop_D_l112_112659

-- Proposition A: For all x ∈ ℝ, x² - x + 1 > 0
theorem prop_A (x : ℝ) : x^2 - x + 1 > 0 :=
sorry

-- Proposition B: a² + a = 0 is not a sufficient and necessary condition for a = 0
theorem prop_B : ¬(∀ a : ℝ, (a^2 + a = 0 ↔ a = 0)) :=
sorry

-- Proposition C: a > 1 and b > 1 is a sufficient and necessary condition for a + b > 2 and ab > 1
theorem prop_C (a b : ℝ) : (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b > 1) :=
sorry

-- Proposition D: a > 4 is a necessary and sufficient condition for the roots of the equation x² - ax + a = 0 to be all positive
theorem prop_D (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, x ≠ 0 → (x^2 - a*x + a = 0 → x > 0)) :=
sorry

end prop_A_prop_B_prop_C_prop_D_l112_112659


namespace smallest_k_for_factorial_divisibility_l112_112832

theorem smallest_k_for_factorial_divisibility : 
  ∃ (k : ℕ), (∀ n : ℕ, n < k → ¬(2040 ∣ n!)) ∧ (2040 ∣ k!) ∧ k = 17 :=
by
  -- We skip the actual proof steps and provide a placeholder for the proof
  sorry

end smallest_k_for_factorial_divisibility_l112_112832


namespace hall_length_width_difference_l112_112666

theorem hall_length_width_difference :
  ∃ (L W : ℝ), 
  (W = (1 / 2) * L) ∧
  (L * W = 288) ∧
  (L - W = 12) :=
by
  -- The mathematical proof follows from the conditions given
  sorry

end hall_length_width_difference_l112_112666


namespace consecutive_composites_l112_112194

theorem consecutive_composites 
  (a t d r : ℕ) (h_a_comp : ∃ p q, p > 1 ∧ q > 1 ∧ a = p * q)
  (h_t_comp : ∃ p q, p > 1 ∧ q > 1 ∧ t = p * q)
  (h_d_comp : ∃ p q, p > 1 ∧ q > 1 ∧ d = p * q)
  (h_r_pos : r > 0) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k < r → ∃ m : ℕ, m > 1 ∧ m ∣ (a * t^(n + k) + d) :=
  sorry

end consecutive_composites_l112_112194


namespace simplify_expression_l112_112092

theorem simplify_expression : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
    ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := 
  sorry

end simplify_expression_l112_112092


namespace total_collection_l112_112674

theorem total_collection (n : ℕ) (c : ℕ) (h_n : n = 88) (h_c : c = 88) : 
  (n * c / 100 : ℚ) = 77.44 :=
by
  sorry

end total_collection_l112_112674


namespace find_width_of_bobs_tv_l112_112946

def area (w h : ℕ) : ℕ := w * h

def weight_in_oz (area : ℕ) : ℕ := area * 4

def weight_in_lb (weight_in_oz : ℕ) : ℕ := weight_in_oz / 16

def width_of_bobs_tv (x : ℕ) : Prop :=
  area 48 100 = 4800 ∧
  weight_in_lb (weight_in_oz (area 48 100)) = 1200 ∧
  weight_in_lb (weight_in_oz (area x 60)) = 15 * x ∧
  15 * x = 1350

theorem find_width_of_bobs_tv : ∃ x : ℕ, width_of_bobs_tv x := sorry

end find_width_of_bobs_tv_l112_112946


namespace fraction_value_l112_112583

theorem fraction_value (a b : ℚ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 :=
by
  -- The proof goes here.
  sorry

end fraction_value_l112_112583


namespace quadratic_solution_symmetry_l112_112035

variable (a b c n : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : a * (-5)^2 + b * (-5) + c = -2.79)
variable (h₂ : a * 1^2 + b * 1 + c = -2.79)
variable (h₃ : a * 2^2 + b * 2 + c = 0)
variable (h₄ : a * 3^2 + b * 3 + c = n)

theorem quadratic_solution_symmetry :
  (x = 3 ∨ x = -7) ↔ (a * x^2 + b * x + c = n) :=
sorry

end quadratic_solution_symmetry_l112_112035


namespace length_of_faster_train_is_370_l112_112912

noncomputable def length_of_faster_train (vf vs : ℕ) (t : ℕ) : ℕ :=
  let rel_speed := vf - vs
  let rel_speed_m_per_s := rel_speed * 1000 / 3600
  rel_speed_m_per_s * t

theorem length_of_faster_train_is_370 :
  length_of_faster_train 72 36 37 = 370 := 
  sorry

end length_of_faster_train_is_370_l112_112912


namespace negate_exists_l112_112395

theorem negate_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔ (∀ x : ℝ, x ≥ Real.sin x ∨ x ≤ Real.tan x) :=
by
  sorry

end negate_exists_l112_112395


namespace charley_initial_pencils_l112_112007

theorem charley_initial_pencils (P : ℕ) (lost_initially : P - 6 = (P - 1/3 * (P - 6) - 6)) (current_pencils : P - 1/3 * (P - 6) - 6 = 16) : P = 30 := 
sorry

end charley_initial_pencils_l112_112007


namespace find_u5_l112_112512

theorem find_u5 
  (u : ℕ → ℝ)
  (h_rec : ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n)
  (h_u3 : u 3 = 9)
  (h_u6 : u 6 = 243) : 
  u 5 = 69 :=
sorry

end find_u5_l112_112512


namespace sum_tens_ones_digits_3_plus_4_power_17_l112_112918

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l112_112918


namespace intersection_of_M_and_N_l112_112468

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}
def intersection_M_N : Set ℕ := {0, 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := by
  sorry

end intersection_of_M_and_N_l112_112468


namespace eccentricity_of_ellipse_l112_112353

noncomputable def ellipse (a b c : ℝ) :=
  (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + c^2) ∧ (b = 2 * c)

theorem eccentricity_of_ellipse (a b c : ℝ) (h : ellipse a b c) :
  (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l112_112353


namespace ninety_eight_squared_l112_112323

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l112_112323


namespace greater_segment_difference_l112_112263

theorem greater_segment_difference :
  ∀ (L1 L2 : ℝ), L1 = 7 ∧ L1^2 - L2^2 = 32 → L1 - L2 = 7 - Real.sqrt 17 :=
by
  intros L1 L2 h
  sorry

end greater_segment_difference_l112_112263


namespace num_eq_7_times_sum_of_digits_l112_112216

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_eq_7_times_sum_of_digits : ∃! n < 1000, n = 7 * sum_of_digits n :=
sorry

end num_eq_7_times_sum_of_digits_l112_112216


namespace anne_speed_l112_112944

-- Definition of distance and time
def distance : ℝ := 6
def time : ℝ := 3

-- Statement to prove
theorem anne_speed : distance / time = 2 := by
  sorry

end anne_speed_l112_112944


namespace find_abs_x_l112_112083

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l112_112083


namespace regular_price_one_pound_is_20_l112_112144

variable (y : ℝ)
variable (discounted_price_quarter_pound : ℝ)

-- Conditions
axiom h1 : 0.6 * (y / 4) + 2 = discounted_price_quarter_pound
axiom h2 : discounted_price_quarter_pound = 2
axiom h3 : 0.1 * y = 2

-- Question: What is the regular price for one pound of cake?
theorem regular_price_one_pound_is_20 : y = 20 := 
  sorry

end regular_price_one_pound_is_20_l112_112144


namespace sound_pressure_level_l112_112898

theorem sound_pressure_level (p_0 p_1 p_2 p_3 : ℝ) (h_p0 : 0 < p_0)
  (L_p : ℝ → ℝ)
  (h_gasoline : 60 ≤ L_p p_1 ∧ L_p p_1 ≤ 90)
  (h_hybrid : 50 ≤ L_p p_2 ∧ L_p p_2 ≤ 60)
  (h_electric : L_p p_3 = 40)
  (h_L_p : ∀ p, L_p p = 20 * Real.log (p / p_0))
  : p_2 ≤ p_1 ∧ p_1 ≤ 100 * p_2 :=
by
  sorry

end sound_pressure_level_l112_112898


namespace triangle_angles_inequality_l112_112597

theorem triangle_angles_inequality (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) 
(h5 : A < Real.pi) (h6 : B < Real.pi) (h7 : C < Real.pi) : 
  A * Real.cos B + Real.sin A * Real.sin C > 0 := 
by 
  sorry

end triangle_angles_inequality_l112_112597


namespace find_v_value_l112_112225

theorem find_v_value (x : ℝ) (v : ℝ) (h1 : x = 3.0) (h2 : 5 * x + v = 19) : v = 4 := by
  sorry

end find_v_value_l112_112225


namespace at_least_one_travels_l112_112570

open ProbabilityTheory

/-
  Let A, B, and C be events representing persons A, B, and C traveling to Beijing respectively.
  The events are mutually independent and have the following probabilities:
  - P(A) = 1/3
  - P(B) = 1/4
  - P(C) = 1/5

  The problem requires us to prove the probability that at least one person travels to Beijing.
-/

noncomputable def prob_A : ℝ := 1 / 3
noncomputable def prob_B : ℝ := 1 / 4
noncomputable def prob_C : ℝ := 1 / 5

theorem at_least_one_travels :
  let prob_not_A := 1 - prob_A,
      prob_not_B := 1 - prob_B,
      prob_not_C := 1 - prob_C,
      prob_none_travel := prob_not_A * prob_not_B * prob_not_C,
      prob_at_least_one_travels := 1 - prob_none_travel 
  in prob_at_least_one_travels = 3 / 5 :=
by
  sorry

end at_least_one_travels_l112_112570


namespace find_piglets_l112_112526

theorem find_piglets (chickens piglets goats sick_animals : ℕ) 
  (h1 : chickens = 26) 
  (h2 : goats = 34) 
  (h3 : sick_animals = 50) 
  (h4 : (chickens + piglets + goats) / 2 = sick_animals) : piglets = 40 := 
by
  sorry

end find_piglets_l112_112526


namespace seeds_planted_on_wednesday_l112_112426

theorem seeds_planted_on_wednesday
  (total_seeds : ℕ) (seeds_thursday : ℕ) (seeds_wednesday : ℕ)
  (h_total : total_seeds = 22) (h_thursday : seeds_thursday = 2) :
  seeds_wednesday = 20 ↔ total_seeds - seeds_thursday = seeds_wednesday :=
by
  -- the proof would go here
  sorry

end seeds_planted_on_wednesday_l112_112426


namespace sum_of_two_digit_and_reverse_l112_112894

theorem sum_of_two_digit_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9)
  (h5 : (10 * a + b) - (10 * b + a) = 9 * (a + b)) : (10 * a + b) + (10 * b + a) = 11 :=
by
  sorry

end sum_of_two_digit_and_reverse_l112_112894


namespace find_min_a_l112_112980

theorem find_min_a (a : ℕ) (h1 : (3150 * a) = x^2) (h2 : a > 0) :
  a = 14 := by
  sorry

end find_min_a_l112_112980


namespace dice_probability_correct_l112_112289

noncomputable def probability_at_least_one_two_or_three : ℚ :=
  let total_outcomes := 64
  let favorable_outcomes := 64 - 36
  favorable_outcomes / total_outcomes

theorem dice_probability_correct :
  probability_at_least_one_two_or_three = 7 / 16 :=
by
  -- Proof will be provided here
  sorry

end dice_probability_correct_l112_112289


namespace greatest_integer_b_not_in_range_of_quadratic_l112_112530

theorem greatest_integer_b_not_in_range_of_quadratic :
  ∀ b : ℤ, (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ 5) ↔ (b^2 < 60) ∧ (b ≤ 7) := by
  sorry

end greatest_integer_b_not_in_range_of_quadratic_l112_112530


namespace sara_marbles_l112_112257

theorem sara_marbles : 10 - 7 = 3 :=
by
  sorry

end sara_marbles_l112_112257


namespace no_preimage_implies_p_gt_1_l112_112378

   noncomputable def f (x : ℝ) : ℝ :=
     -x^2 + 2 * x

   theorem no_preimage_implies_p_gt_1 (p : ℝ) (hp : ∀ x : ℝ, f x ≠ p) : p > 1 :=
   sorry
   
end no_preimage_implies_p_gt_1_l112_112378


namespace equal_pieces_length_l112_112137

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l112_112137


namespace students_enrolled_for_german_l112_112607

-- Defining the total number of students
def class_size : Nat := 40

-- Defining the number of students enrolled for both English and German
def enrolled_both : Nat := 12

-- Defining the number of students enrolled for only English and not German
def enrolled_only_english : Nat := 18

-- Using the conditions to define the number of students who enrolled for German
theorem students_enrolled_for_german (G G_only : Nat) 
  (h_class_size : G_only + enrolled_only_english + enrolled_both = class_size) 
  (h_G : G = G_only + enrolled_both) : 
  G = 22 := 
by
  -- placeholder for proof
  sorry

end students_enrolled_for_german_l112_112607


namespace min_x_plus_3y_l112_112463

noncomputable def minimum_x_plus_3y (x y : ℝ) : ℝ :=
  if h : (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) then x + 3*y else 0

theorem min_x_plus_3y : ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) → x + 3*y = 6 :=
by
  intros x y h
  sorry

end min_x_plus_3y_l112_112463


namespace quadratic_intersects_x_axis_at_two_points_l112_112716

theorem quadratic_intersects_x_axis_at_two_points (k : ℝ) :
  (k < 1 ∧ k ≠ 0) ↔ ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (kx1^2 + 2 * x1 + 1 = 0) ∧ (kx2^2 + 2 * x2 + 1 = 0) := 
by
  sorry

end quadratic_intersects_x_axis_at_two_points_l112_112716


namespace gain_percent_is_100_l112_112538

variable {C S : ℝ}

-- Given conditions
axiom h1 : 50 * C = 25 * S
axiom h2 : S = 2 * C

-- Prove the gain percent is 100%
theorem gain_percent_is_100 (h1 : 50 * C = 25 * S) (h2 : S = 2 * C) : (S - C) / C * 100 = 100 :=
by
  sorry

end gain_percent_is_100_l112_112538


namespace garden_enlargement_l112_112553

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l112_112553


namespace january_salary_l112_112126

variable (J F M A My : ℕ)

axiom average_salary_1 : (J + F + M + A) / 4 = 8000
axiom average_salary_2 : (F + M + A + My) / 4 = 8400
axiom may_salary : My = 6500

theorem january_salary : J = 4900 :=
by
  /- To be filled with the proof steps applying the given conditions -/
  sorry

end january_salary_l112_112126


namespace determinant_inequality_solution_l112_112864

theorem determinant_inequality_solution (a : ℝ) :
  (∀ x : ℝ, (x > -1 → x < (4 / a))) ↔ a = -4 := by
sorry

end determinant_inequality_solution_l112_112864


namespace find_second_number_l112_112110

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end find_second_number_l112_112110


namespace sum_of_missing_angles_l112_112801

theorem sum_of_missing_angles (angle_sum_known : ℕ) (divisor : ℕ) (total_sides : ℕ) (missing_angles_sum : ℕ)
  (h1 : angle_sum_known = 1620)
  (h2 : divisor = 180)
  (h3 : total_sides = 12)
  (h4 : angle_sum_known + missing_angles_sum = divisor * (total_sides - 2)) :
  missing_angles_sum = 180 :=
by
  -- Skipping the proof for this theorem
  sorry

end sum_of_missing_angles_l112_112801


namespace inequality_proof_l112_112501

open scoped BigOperators

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 1 / 2) :
  (∑ i, (a i)^2 / (∑ i, a i)^2) ≥ (∑ i, (1 - a i)^2 / (∑ i, (1 - a i))^2) := 
by 
  sorry

end inequality_proof_l112_112501


namespace no_real_solutions_l112_112385

theorem no_real_solutions : ¬ ∃ (r s : ℝ),
  (r - 50) / 3 = (s - 2 * r) / 4 ∧
  r^2 + 3 * s = 50 :=
by {
  -- sorry, proof steps would go here
  sorry
}

end no_real_solutions_l112_112385


namespace sarah_reads_100_words_per_page_l112_112757

noncomputable def words_per_page (W_pages : ℕ) (books : ℕ) (hours : ℕ) (pages_per_book : ℕ) (words_per_minute : ℕ) : ℕ :=
  (words_per_minute * 60 * hours) / books / pages_per_book

theorem sarah_reads_100_words_per_page :
  words_per_page 80 6 20 80 40 = 100 := 
sorry

end sarah_reads_100_words_per_page_l112_112757


namespace probability_losing_game_l112_112863

-- Define the odds of winning in terms of number of wins and losses
def odds_winning := (wins: ℕ, losses: ℕ) := (5, 3)

-- Given the odds of winning, calculate the total outcomes
def total_outcomes : ℕ := (odds_winning.1 + odds_winning.2)

-- Define the probability of losing the game
def probability_of_losing (wins losses: ℕ) (total: ℕ) : ℚ := (losses : ℚ) / (total : ℚ)

-- Given odds of 5:3, prove the probability of losing is 3/8
theorem probability_losing_game : probability_of_losing odds_winning.1 odds_winning.2 total_outcomes = 3 / 8 :=
by
  sorry

end probability_losing_game_l112_112863


namespace capacity_ratio_proof_l112_112691

noncomputable def capacity_ratio :=
  ∀ (C_X C_Y : ℝ), 
    (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y →
    (C_Y / C_X) = (1 / 2)

-- includes a statement without proof
theorem capacity_ratio_proof (C_X C_Y : ℝ) (h : (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y) : 
  (C_Y / C_X) = (1 / 2) :=
  by
    sorry

end capacity_ratio_proof_l112_112691


namespace graph_symmetry_l112_112050

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem graph_symmetry (ω φ : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h_sym_distance : ∀ x y, f ω φ x = f ω φ y → |x - y| = Real.pi / 4)
  (h_shifted_symmetry : ∀ x, f ω φ (x + 3 * Real.pi / 16) = f ω φ (-x - 3 * Real.pi / 16)) :
  (∀ x, f ω φ x = f ω φ (-x) → x = π / 16 ∨ x = -π / 4) :=
sorry

end graph_symmetry_l112_112050


namespace tan_inequality_l112_112208

open Real

theorem tan_inequality {x1 x2 : ℝ} 
  (h1 : 0 < x1 ∧ x1 < π / 2) 
  (h2 : 0 < x2 ∧ x2 < π / 2) 
  (h3 : x1 ≠ x2) : 
  (1 / 2 * (tan x1 + tan x2) > tan ((x1 + x2) / 2)) :=
sorry

end tan_inequality_l112_112208


namespace meal_combinations_correct_l112_112692

-- Let E denote the total number of dishes on the menu
def E : ℕ := 12

-- Let V denote the number of vegetarian dishes on the menu
def V : ℕ := 5

-- Define the function that computes the number of different combinations of meals Elena and Nasir can order
def meal_combinations (e : ℕ) (v : ℕ) : ℕ :=
  e * v

-- The theorem to prove that the number of different combinations of meals Elena and Nasir can order is 60
theorem meal_combinations_correct : meal_combinations E V = 60 := by
  sorry

end meal_combinations_correct_l112_112692


namespace find_number_l112_112788

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 9) : x = 4.5 :=
by
  sorry

end find_number_l112_112788


namespace trisect_chord_exists_l112_112246

noncomputable def distance (O P : Point) : ℝ := sorry
def trisect (P : Point) (A B : Point) : Prop := 2 * (distance A P) = distance P B

-- Main theorem based on the given conditions and conclusions
theorem trisect_chord_exists (O P : Point) (r : ℝ) (hP_in_circle : distance O P < r) :
  (∃ A B : Point, trisect P A B) ↔ 
  (distance O P > r / 3 ∨ distance O P = r / 3) :=
by
  sorry

end trisect_chord_exists_l112_112246


namespace same_terminal_side_l112_112813

open Real

/-- Given two angles θ₁ = -7π/9 and θ₂ = 11π/9, prove that they have the same terminal side
    which means proving θ₂ - θ₁ is an integer multiple of 2π. -/
theorem same_terminal_side (θ₁ θ₂ : ℝ) (hθ₁ : θ₁ = - (7 * π / 9)) (hθ₂ : θ₂ = 11 * π / 9) :
    ∃ (k : ℤ), θ₂ - θ₁ = 2 * π * k := by
  sorry

end same_terminal_side_l112_112813


namespace probability_of_losing_l112_112862

noncomputable def odds_of_winning : ℕ := 5
noncomputable def odds_of_losing : ℕ := 3
noncomputable def total_outcomes : ℕ := odds_of_winning + odds_of_losing

theorem probability_of_losing : 
  (odds_of_losing : ℚ) / (total_outcomes : ℚ) = 3 / 8 := 
by
  sorry

end probability_of_losing_l112_112862


namespace flour_amount_l112_112622

theorem flour_amount (a b : ℕ) (h₁ : a = 8) (h₂ : b = 2) : a + b = 10 := by
  sorry

end flour_amount_l112_112622


namespace result_of_dividing_295_by_5_and_adding_6_is_65_l112_112404

theorem result_of_dividing_295_by_5_and_adding_6_is_65 : (295 / 5) + 6 = 65 := by
  sorry

end result_of_dividing_295_by_5_and_adding_6_is_65_l112_112404


namespace four_thirds_of_nine_halves_l112_112187

theorem four_thirds_of_nine_halves :
  (4 / 3) * (9 / 2) = 6 := 
sorry

end four_thirds_of_nine_halves_l112_112187


namespace fraction_meaningful_l112_112477

theorem fraction_meaningful (x : ℝ) : (x ≠ 5) ↔ (x-5 ≠ 0) :=
by simp [sub_eq_zero]

end fraction_meaningful_l112_112477


namespace betty_cookies_brownies_l112_112313

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_cookies_brownies_l112_112313


namespace relay_race_l112_112667

theorem relay_race (n : ℕ) (H1 : 2004 % n = 0) (H2 : n ≤ 168) (H3 : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6 ∧ n ≠ 12): n = 167 :=
by
  sorry

end relay_race_l112_112667


namespace calories_per_candy_bar_l112_112906

theorem calories_per_candy_bar (total_calories : ℕ) (number_of_bars : ℕ) 
  (h : total_calories = 341) (n : number_of_bars = 11) : (total_calories / number_of_bars = 31) :=
by
  sorry

end calories_per_candy_bar_l112_112906


namespace range_of_a_l112_112223

theorem range_of_a (a : ℝ) : 
  4 * a^2 - 12 * (a + 6) > 0 ↔ a < -3 ∨ a > 6 := 
by sorry

end range_of_a_l112_112223


namespace solve_for_x_l112_112781

theorem solve_for_x :
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 2) + 1 / (2 * x) = 1 / x) ∧ (1 / (x + 5) + 1 / (x + 2) = 1 / (x + 3)) ∧ x = 2 :=
by
  sorry

end solve_for_x_l112_112781


namespace algebraic_expr_pos_int_vals_l112_112023

noncomputable def algebraic_expr_ineq (x : ℕ) : Prop :=
  x > 0 ∧ ((x + 1)/3 - (2*x - 1)/4 ≥ (x - 3)/6)

theorem algebraic_expr_pos_int_vals : {x : ℕ | algebraic_expr_ineq x} = {1, 2, 3} :=
sorry

end algebraic_expr_pos_int_vals_l112_112023


namespace trajectory_of_center_of_moving_circle_l112_112429

noncomputable def circle_tangency_condition_1 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 1
noncomputable def circle_tangency_condition_2 (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 9

def ellipse_equation (x y : ℝ) : Prop := x ^ 2 / 4 + y ^ 2 / 3 = 1

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  circle_tangency_condition_1 x y ∧ circle_tangency_condition_2 x y →
  ellipse_equation x y := sorry

end trajectory_of_center_of_moving_circle_l112_112429


namespace ratio_of_sequences_is_5_over_4_l112_112444

-- Definitions of arithmetic sequences
def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Hypotheses
def sequence_1_sum : ℕ :=
  arithmetic_sum 5 5 16

def sequence_2_sum : ℕ :=
  arithmetic_sum 4 4 16

-- Main statement to be proven
theorem ratio_of_sequences_is_5_over_4 : sequence_1_sum / sequence_2_sum = 5 / 4 := sorry

end ratio_of_sequences_is_5_over_4_l112_112444


namespace max_plus_shapes_l112_112432

def cover_square (x y : ℕ) : Prop :=
  3 * x + 5 * y = 49

theorem max_plus_shapes (x y : ℕ) (h1 : cover_square x y) (h2 : x ≥ 4) : y ≤ 5 :=
sorry

end max_plus_shapes_l112_112432


namespace range_of_a_l112_112357

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 3 then a - x else a * log x / log 2

theorem range_of_a (a : ℝ) (h : f a 2 < f a 4) : a > -2 := by
  sorry

end range_of_a_l112_112357


namespace square_inscribed_in_right_triangle_side_length_l112_112885

theorem square_inscribed_in_right_triangle_side_length
  (A B C X Y Z W : ℝ × ℝ)
  (AB BC AC : ℝ)
  (square_side : ℝ)
  (h : 0 < square_side) :
  -- Define the lengths of sides of the triangle.
  AB = 3 ∧ BC = 4 ∧ AC = 5 ∧

  -- Define the square inscribed in the triangle
  (W.1 - A.1)^2 + (W.2 - A.2)^2 = square_side^2 ∧
  (X.1 - W.1)^2 + (X.2 - W.2)^2 = square_side^2 ∧
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = square_side^2 ∧
  (Z.1 - W.1)^2 + (Z.2 - W.2)^2 = square_side^2 ∧
  (Z.1 - C.1)^2 + (Z.2 - C.2)^2 = square_side^2 ∧

  -- Points where square meets triangle sides
  X.1 = A.1 ∧ Z.1 = C.1 ∧ Y.1 = X.1 ∧ W.1 = Z.1 ∧ Z.2 = Y.2 ∧

  -- Right triangle condition
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 ∧
  
  -- Right angle at vertex B
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  →
  -- Prove the side length of the inscribed square
  square_side = 60 / 37 :=
sorry

end square_inscribed_in_right_triangle_side_length_l112_112885


namespace stock_return_to_original_l112_112146

theorem stock_return_to_original (x : ℝ) : 
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  1.56 * (1 - p/100) = 1 :=
by
  intro x
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  show 1.56 * (1 - p / 100) = 1
  sorry

end stock_return_to_original_l112_112146


namespace range_of_m_l112_112029

def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 9 ≤ m :=
by
  sorry

end range_of_m_l112_112029


namespace locus_of_C_l112_112711

variable (a : ℝ) (h : a > 0)

theorem locus_of_C : 
  ∃ (x y : ℝ), 
  (1 - a) * x^2 - 2 * a * x + (1 + a) * y^2 = 0 :=
sorry

end locus_of_C_l112_112711


namespace luke_can_see_silvia_for_22_point_5_minutes_l112_112503

/--
Luke is initially 0.75 miles behind Silvia. Luke rollerblades at 10 mph and Silvia cycles 
at 6 mph. Luke can see Silvia until she is 0.75 miles behind him. Prove that Luke can see 
Silvia for a total of 22.5 minutes.
-/
theorem luke_can_see_silvia_for_22_point_5_minutes :
    let distance := (3 / 4 : ℝ)
    let luke_speed := (10 : ℝ)
    let silvia_speed := (6 : ℝ)
    let relative_speed := luke_speed - silvia_speed
    let time_to_reach := distance / relative_speed
    let total_time := 2 * time_to_reach * 60 
    total_time = 22.5 :=
by
    sorry

end luke_can_see_silvia_for_22_point_5_minutes_l112_112503


namespace committees_including_past_officer_l112_112684

theorem committees_including_past_officer (total_candidates past_officers: ℕ) (positions: ℕ) 
  (h1: total_candidates = 20) 
  (h2: past_officers = 9) 
  (h3: positions = 6) : 
  choose total_candidates positions - choose (total_candidates - past_officers) positions = 38298 :=
by sorry

end committees_including_past_officer_l112_112684


namespace find_congruence_l112_112359

theorem find_congruence (x : ℤ) (h : 4 * x + 9 ≡ 3 [ZMOD 17]) : 3 * x + 12 ≡ 16 [ZMOD 17] :=
sorry

end find_congruence_l112_112359


namespace pyramid_volume_is_1_12_l112_112808

def base_rectangle_length_1 := 1
def base_rectangle_width_1_4 := 1 / 4
def pyramid_height_1 := 1

noncomputable def pyramid_volume : ℝ :=
  (1 / 3) * (base_rectangle_length_1 * base_rectangle_width_1_4) * pyramid_height_1

theorem pyramid_volume_is_1_12 : pyramid_volume = 1 / 12 :=
sorry

end pyramid_volume_is_1_12_l112_112808


namespace boat_current_ratio_l112_112663

noncomputable def boat_speed_ratio (b c : ℝ) (d : ℝ) : Prop :=
  let time_upstream := 6
  let time_downstream := 10
  d = time_upstream * (b - c) ∧ 
  d = time_downstream * (b + c) → 
  b / c = 4

theorem boat_current_ratio (b c d : ℝ) (h1 : d = 6 * (b - c)) (h2 : d = 10 * (b + c)) : b / c = 4 :=
by sorry

end boat_current_ratio_l112_112663


namespace distance_between_first_and_last_tree_l112_112647

theorem distance_between_first_and_last_tree
  (n : ℕ) (d_1_5 : ℝ) (h1 : n = 8) (h2 : d_1_5 = 100) :
  let interval_distance := d_1_5 / 4
  let total_intervals := n - 1
  let total_distance := interval_distance * total_intervals
  total_distance = 175 :=
by
  sorry

end distance_between_first_and_last_tree_l112_112647


namespace truffles_more_than_caramels_l112_112440

-- Define the conditions
def chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def peanut_clusters := (64 * chocolates) / 100
def truffles := chocolates - (caramels + nougats + peanut_clusters)

-- Define the claim
theorem truffles_more_than_caramels : (truffles - caramels) = 6 := by
  sorry

end truffles_more_than_caramels_l112_112440


namespace fundraiser_full_price_revenue_l112_112305

theorem fundraiser_full_price_revenue :
  ∃ (f h p : ℕ), f + h = 200 ∧ 
                f * p + h * (p / 2) = 2700 ∧ 
                f * p = 600 :=
by 
  sorry

end fundraiser_full_price_revenue_l112_112305


namespace symmetric_points_l112_112199

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_points_l112_112199


namespace circle_tangent_to_directrix_and_yaxis_on_parabola_l112_112342

noncomputable def circle1_eq (x y : ℝ) := (x - 1)^2 + (y - 1 / 2)^2 = 1
noncomputable def circle2_eq (x y : ℝ) := (x + 1)^2 + (y - 1 / 2)^2 = 1

theorem circle_tangent_to_directrix_and_yaxis_on_parabola :
  ∀ (x y : ℝ), (x^2 = 2 * y) → 
  ((y = -1 / 2 → circle1_eq x y) ∨ (y = -1 / 2 → circle2_eq x y)) :=
by
  intro x y h_parabola
  sorry

end circle_tangent_to_directrix_and_yaxis_on_parabola_l112_112342


namespace fraction_to_decimal_l112_112964

theorem fraction_to_decimal (numer: ℚ) (denom: ℕ) (h_denom: denom = 2^5 * 5^1) :
  numer.den = 160 → numer.num = 59 → numer == 0.36875 :=
by
  intros
  sorry  

end fraction_to_decimal_l112_112964


namespace calculate_mean_score_l112_112453

theorem calculate_mean_score (M SD : ℝ) 
  (h1 : M - 2 * SD = 60)
  (h2 : M + 3 * SD = 100) : 
  M = 76 :=
by
  sorry

end calculate_mean_score_l112_112453


namespace jeremy_gifted_37_goats_l112_112986

def initial_horses := 100
def initial_sheep := 29
def initial_chickens := 9

def total_initial_animals := initial_horses + initial_sheep + initial_chickens
def animals_bought_by_brian := total_initial_animals / 2
def animals_left_after_brian := total_initial_animals - animals_bought_by_brian

def total_male_animals := 53
def total_female_animals := 53
def total_remaining_animals := total_male_animals + total_female_animals

def goats_gifted_by_jeremy := total_remaining_animals - animals_left_after_brian

theorem jeremy_gifted_37_goats :
  goats_gifted_by_jeremy = 37 := 
by 
  sorry

end jeremy_gifted_37_goats_l112_112986


namespace custom_op_eval_l112_112458

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 5 * a + 2 * b - 1

-- State the required proof problem
theorem custom_op_eval : custom_op (-4) 6 = -9 := 
by
  -- use sorry to skip the proof
  sorry

end custom_op_eval_l112_112458


namespace simplified_expression_evaluates_to_2_l112_112630

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end simplified_expression_evaluates_to_2_l112_112630


namespace rhombus_diagonal_l112_112262

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) : d2 = 15 → area = 127.5 → d1 = 17 :=
by
  intros h1 h2
  sorry

end rhombus_diagonal_l112_112262


namespace absolute_value_of_x_l112_112079

variable (x : ℝ)

theorem absolute_value_of_x (h: (| (3 + x) - (3 - x) |) = 8) : |x| = 4 :=
by sorry

end absolute_value_of_x_l112_112079


namespace distinct_arrangements_l112_112361

-- Definitions based on the conditions
def boys : ℕ := 4
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def arrangements : ℕ := Nat.factorial boys * Nat.factorial (total_people - 2) * Nat.factorial 6

-- Main statement: Verify the number of distinct arrangements
theorem distinct_arrangements : arrangements = 8640 := by
  -- We will replace this proof with our Lean steps (which is currently omitted)
  sorry

end distinct_arrangements_l112_112361


namespace marbles_in_larger_container_l112_112138

-- Defining the conditions
def volume1 := 24 -- in cm³
def marbles1 := 30 -- number of marbles in the first container
def volume2 := 72 -- in cm³

-- Statement of the theorem
theorem marbles_in_larger_container : (marbles1 / volume1 : ℚ) * volume2 = 90 := by
  sorry

end marbles_in_larger_container_l112_112138


namespace max_correct_answers_l112_112481

theorem max_correct_answers (a b c : ℕ) (n : ℕ := 60) (p_correct : ℤ := 5) (p_blank : ℤ := 0) (p_incorrect : ℤ := -2) (S : ℤ := 150) :
        a + b + c = n ∧ p_correct * a + p_blank * b + p_incorrect * c = S → a ≤ 38 :=
by
  sorry

end max_correct_answers_l112_112481


namespace number_in_central_region_l112_112884

theorem number_in_central_region (a b c d : ℤ) :
  a + b + c + d = -4 →
  ∃ x : ℤ, x = -4 + 2 :=
by
  intros h
  use -2
  sorry

end number_in_central_region_l112_112884


namespace equal_pieces_length_l112_112135

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l112_112135


namespace probability_of_draw_l112_112926

-- Define the probabilities as constants
def prob_not_lose_xiao_ming : ℚ := 3 / 4
def prob_lose_xiao_dong : ℚ := 1 / 2

-- State the theorem we want to prove
theorem probability_of_draw :
  prob_not_lose_xiao_ming - prob_lose_xiao_dong = 1 / 4 :=
by
  sorry

end probability_of_draw_l112_112926


namespace perfect_number_mod_9_l112_112758

theorem perfect_number_mod_9 (N : ℕ) (hN : ∃ p, N = 2^(p-1) * (2^p - 1) ∧ Nat.Prime (2^p - 1)) (hN_ne_6 : N ≠ 6) : ∃ n : ℕ, N = 9 * n + 1 :=
by
  sorry

end perfect_number_mod_9_l112_112758


namespace Joshua_share_correct_l112_112070

noncomputable def Joshua_share (J : ℝ) : ℝ :=
  3 * J

noncomputable def Jasmine_share (J : ℝ) : ℝ :=
  J / 2

theorem Joshua_share_correct (J : ℝ) (h : J + 3 * J + J / 2 = 120) :
  Joshua_share J = 80.01 := by
  sorry

end Joshua_share_correct_l112_112070


namespace cannot_tile_surface_square_hexagon_l112_112907

-- Definitions of internal angles of the tile shapes
def internal_angle_triangle := 60
def internal_angle_square := 90
def internal_angle_hexagon := 120
def internal_angle_octagon := 135

-- The theorem to prove that square and hexagon cannot tile a surface without gaps or overlaps
theorem cannot_tile_surface_square_hexagon : ∀ (m n : ℕ), internal_angle_square * m + internal_angle_hexagon * n ≠ 360 := 
by sorry

end cannot_tile_surface_square_hexagon_l112_112907


namespace units_digit_of_45_pow_125_plus_7_pow_87_l112_112665

theorem units_digit_of_45_pow_125_plus_7_pow_87 :
  (45 ^ 125 + 7 ^ 87) % 10 = 8 :=
by
  -- sorry to skip the proof
  sorry

end units_digit_of_45_pow_125_plus_7_pow_87_l112_112665


namespace perpendicular_lines_implies_perpendicular_plane_l112_112834

theorem perpendicular_lines_implies_perpendicular_plane
  (triangle_sides : Line → Prop)
  (circle_diameters : Line → Prop)
  (perpendicular : Line → Line → Prop)
  (is_perpendicular_to_plane : Line → Prop) :
  (∀ l₁ l₂, triangle_sides l₁ → triangle_sides l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) ∧
  (∀ l₁ l₂, circle_diameters l₁ → circle_diameters l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) :=
  sorry

end perpendicular_lines_implies_perpendicular_plane_l112_112834


namespace probability_face_diamonds_then_spades_l112_112651

-- Define the standard deck of 52 cards.
def standard_deck := 52

-- Define the face cards of diamonds (Jack, Queen, King).
def face_cards_diamonds := 3

-- Define the face cards of spades (Jack, Queen, King).
def face_cards_spades := 3

-- Calculate the probability of drawing a face card of diamonds first.
def prob_first_face_diamonds := (face_cards_diamonds : ℚ) / standard_deck

-- Calculate the probability of drawing a face card of spades second.
def prob_second_face_spades := (face_cards_spades : ℚ) / (standard_deck - 1)

-- Calculate the combined probability.
def combined_probability : ℚ := prob_first_face_diamonds * prob_second_face_spades

-- Prove that the combined probability is 1/294.
theorem probability_face_diamonds_then_spades :
  combined_probability = 1 / 294 := by
  simp [combined_probability, prob_first_face_diamonds, prob_second_face_spades]
  sorry

end probability_face_diamonds_then_spades_l112_112651


namespace remaining_amount_after_purchase_l112_112334

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end remaining_amount_after_purchase_l112_112334


namespace pure_imaginary_solution_l112_112222

theorem pure_imaginary_solution (a : ℝ) (i : ℂ) (h : i*i = -1) : (∀ z : ℂ, z = 1 + a * i → (z ^ 2).re = 0) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_solution_l112_112222


namespace no_solution_if_n_eq_neg_one_l112_112824

theorem no_solution_if_n_eq_neg_one (n x y z : ℝ) :
  (n * x + y + z = 2) ∧ (x + n * y + z = 2) ∧ (x + y + n * z = 2) ↔ n = -1 → false :=
by
  sorry

end no_solution_if_n_eq_neg_one_l112_112824


namespace expression_divisible_by_25_l112_112507

theorem expression_divisible_by_25 (n : ℕ) : 
    (2^(n+2) * 3^n + 5 * n - 4) % 25 = 0 :=
by {
  sorry
}

end expression_divisible_by_25_l112_112507


namespace triangle_sum_is_16_l112_112761

-- Definition of the triangle operation
def triangle (a b c : ℕ) : ℕ := a * b - c

-- Lean theorem statement
theorem triangle_sum_is_16 : 
  triangle 2 4 3 + triangle 3 6 7 = 16 := 
by 
  sorry

end triangle_sum_is_16_l112_112761


namespace range_of_x_l112_112566

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l112_112566


namespace find_n_for_sine_equality_l112_112189

theorem find_n_for_sine_equality : 
  ∃ (n: ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (670 * Real.pi / 180) ∧ n = -50 := by
  sorry

end find_n_for_sine_equality_l112_112189


namespace machines_complete_job_in_12_days_l112_112911

-- Given the conditions
variable (D : ℕ) -- The number of days for 12 machines to complete the job
variable (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8))

-- Prove the number of days for 12 machines to complete the job
theorem machines_complete_job_in_12_days (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8)) : D = 12 :=
by
  sorry

end machines_complete_job_in_12_days_l112_112911


namespace reduced_price_l112_112421

theorem reduced_price (
  P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 9 = 900 / R - 900 / P)
  (h3 : P = 42.8571) :
  R = 30 :=
by {
  sorry
}

end reduced_price_l112_112421


namespace fraction_inequality_l112_112709

theorem fraction_inequality (a b c : ℝ) : 
  (a / (a + 2 * b + c)) + (b / (a + b + 2 * c)) + (c / (2 * a + b + c)) ≥ 3 / 4 := 
by
  sorry

end fraction_inequality_l112_112709


namespace polynomial_sum_is_integer_l112_112255

-- Define the integer polynomial and the integers a and b
variables (f : ℤ[X]) (a b : ℤ)

-- The theorem statement
theorem polynomial_sum_is_integer :
  ∃ c : ℤ, f.eval (a - real.sqrt b) + f.eval (a + real.sqrt b) = c :=
sorry

end polynomial_sum_is_integer_l112_112255


namespace helen_baked_more_raisin_cookies_l112_112592

-- Definitions based on conditions
def raisin_cookies_yesterday : ℕ := 300
def raisin_cookies_day_before : ℕ := 280

-- Theorem to prove the answer
theorem helen_baked_more_raisin_cookies : raisin_cookies_yesterday - raisin_cookies_day_before = 20 :=
by
  sorry

end helen_baked_more_raisin_cookies_l112_112592


namespace primes_p_plus_10_plus_14_l112_112478

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_p_plus_10_plus_14 (p : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (p + 10)) 
  (h3 : is_prime (p + 14)) 
  : p = 3 := sorry

end primes_p_plus_10_plus_14_l112_112478


namespace total_water_bottles_needed_l112_112956

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l112_112956


namespace B_max_at_125_l112_112821

noncomputable def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 : ℝ) ^ k

theorem B_max_at_125 :
  ∃ k, 0 ≤ k ∧ k ≤ 500 ∧ (∀ n, 0 ≤ n ∧ n ≤ 500 → B k ≥ B n) ∧ k = 125 :=
by
  sorry

end B_max_at_125_l112_112821


namespace arithmetic_progression_a6_l112_112036

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l112_112036


namespace whitewash_all_planks_not_whitewash_all_planks_l112_112784

open Finset

variable {N : ℕ} (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1))

def f (n : ℤ) : ℤ := n^2 + 3*n - 2

def f_equiv (x y : ℤ) : Prop := 2^(Nat.log2 (2 * N)) ∣ (f x - f y)

theorem whitewash_all_planks (N : ℕ) (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1)) : 
  ∀ n ∈ range N, ∃ m ∈ range N, f m = n :=
by {
  sorry
}

theorem not_whitewash_all_planks (N : ℕ) (not_power_of_two : ¬(∃ (k : ℕ), N = 2^(k + 1))) : 
  ∃ n ∈ range N, ∀ m ∈ range N, f m ≠ n :=
by {
  sorry
}

end whitewash_all_planks_not_whitewash_all_planks_l112_112784


namespace factorization_example_l112_112264

theorem factorization_example (C D : ℤ) (h : 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) : C * D + C = 25 := by
  sorry

end factorization_example_l112_112264


namespace discount_rate_on_pony_jeans_l112_112456

-- Define the conditions as Lean definitions
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def total_savings : ℝ := 8.91
def total_discount_rate : ℝ := 22
def number_of_fox_pairs : ℕ := 3
def number_of_pony_pairs : ℕ := 2

-- Given definitions of the discount rates on Fox and Pony jeans
variable (F P : ℝ)

-- The system of equations based on the conditions
axiom sum_of_discount_rates : F + P = total_discount_rate
axiom savings_equation : 
  number_of_fox_pairs * (fox_price * F / 100) + number_of_pony_pairs * (pony_price * P / 100) = total_savings

-- The theorem to prove
theorem discount_rate_on_pony_jeans : P = 11 := by
  sorry

end discount_rate_on_pony_jeans_l112_112456


namespace sum_of_five_integers_l112_112396

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end sum_of_five_integers_l112_112396


namespace remainder_of_f_div_r_minus_2_l112_112831

def f (r : ℝ) : ℝ := r^15 - 3

theorem remainder_of_f_div_r_minus_2 : f 2 = 32765 := by
  sorry

end remainder_of_f_div_r_minus_2_l112_112831


namespace gcd_of_72_and_90_l112_112291

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end gcd_of_72_and_90_l112_112291


namespace probability_of_getting_exactly_5_heads_l112_112063

noncomputable def num_ways_to_get_heads (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_getting_exactly_5_heads :
  let total_outcomes := 2 ^ 10
  let num_heads_5 := num_ways_to_get_heads 10 5
  let probability := num_heads_5 / total_outcomes
  probability = (63 : ℚ) / 256 :=
by
  sorry

end probability_of_getting_exactly_5_heads_l112_112063


namespace milk_leftover_l112_112436

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end milk_leftover_l112_112436


namespace tan_ratio_triangle_area_l112_112486

theorem tan_ratio (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A) :
  Real.tan A / Real.tan B = -4 := by
  sorry

theorem triangle_area (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A)
  (h2 : c = 2) (h3 : Real.tan C = 3 / 4) :
  ∃ S : ℝ, S = 1 / 2 * b * c * Real.sin A ∧ S = 4 / 3 := by
  sorry

end tan_ratio_triangle_area_l112_112486


namespace bottles_difference_l112_112447

noncomputable def Donald_drinks_bottles (P: ℕ): ℕ := 2 * P + 3
noncomputable def Paul_drinks_bottles: ℕ := 3
noncomputable def actual_Donald_bottles: ℕ := 9

theorem bottles_difference:
  actual_Donald_bottles - 2 * Paul_drinks_bottles = 3 :=
by 
  sorry

end bottles_difference_l112_112447


namespace robot_distance_proof_l112_112774

noncomputable def distance (south1 south2 south3 east1 east2 : ℝ) : ℝ :=
  Real.sqrt ((south1 + south2 + south3)^2 + (east1 + east2)^2)

theorem robot_distance_proof :
  distance 1.2 1.8 1.0 1.0 2.0 = 5.0 :=
by
  sorry

end robot_distance_proof_l112_112774


namespace value_of_2p_plus_q_l112_112362

theorem value_of_2p_plus_q (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p :=
by
  sorry

end value_of_2p_plus_q_l112_112362


namespace Prudence_sleep_weeks_l112_112026

def Prudence_sleep_per_week : Nat := 
  let nights_sleep_weekday := 6
  let nights_sleep_weekend := 9
  let weekday_nights := 5
  let weekend_nights := 2
  let naps := 1
  let naps_days := 2
  weekday_nights * nights_sleep_weekday + weekend_nights * nights_sleep_weekend + naps_days * naps

theorem Prudence_sleep_weeks (w : Nat) (h : w * Prudence_sleep_per_week = 200) : w = 4 :=
by
  sorry

end Prudence_sleep_weeks_l112_112026


namespace profit_percentage_for_unspecified_weight_l112_112938

-- Definitions to align with the conditions
def total_sugar : ℝ := 1000
def profit_400_kg : ℝ := 0.08
def unspecified_weight : ℝ := 600
def overall_profit : ℝ := 0.14
def total_400_kg := total_sugar - unspecified_weight
def total_overall_profit := total_sugar * overall_profit
def total_400_kg_profit := total_400_kg * profit_400_kg
def total_unspecified_weight_profit (profit_percentage : ℝ) := unspecified_weight * profit_percentage

-- The theorem statement
theorem profit_percentage_for_unspecified_weight : 
  ∃ (profit_percentage : ℝ), total_400_kg_profit + total_unspecified_weight_profit profit_percentage = total_overall_profit ∧ profit_percentage = 0.18 := by
  sorry

end profit_percentage_for_unspecified_weight_l112_112938


namespace base_s_computation_l112_112068

theorem base_s_computation (s : ℕ) (h : 550 * s + 420 * s = 1100 * s) : s = 7 := by
  sorry

end base_s_computation_l112_112068


namespace swimming_speed_in_still_water_l112_112430

-- Given conditions
def water_speed : ℝ := 4
def swim_time_against_current : ℝ := 2
def swim_distance_against_current : ℝ := 8

-- What we are trying to prove
theorem swimming_speed_in_still_water (v : ℝ) 
    (h1 : swim_distance_against_current = 8) 
    (h2 : swim_time_against_current = 2)
    (h3 : water_speed = 4) :
    v - water_speed = swim_distance_against_current / swim_time_against_current → v = 8 :=
by
  sorry

end swimming_speed_in_still_water_l112_112430


namespace mask_digits_l112_112895

theorem mask_digits : 
  ∃ (elephant mouse pig panda : ℕ), 
  (elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧ 
   mouse ≠ pig ∧ mouse ≠ panda ∧ pig ≠ panda) ∧
  (4 * 4 = 16) ∧ (7 * 7 = 49) ∧ (8 * 8 = 64) ∧ (9 * 9 = 81) ∧
  (elephant = 6) ∧ (mouse = 4) ∧ (pig = 8) ∧ (panda = 1) :=
by
  sorry

end mask_digits_l112_112895


namespace jenna_filter_change_15th_is_March_l112_112694

def month_of_nth_change (startMonth interval n : ℕ) : ℕ :=
  ((interval * (n - 1)) % 12 + startMonth) % 12

theorem jenna_filter_change_15th_is_March :
  month_of_nth_change 1 7 15 = 3 := 
  sorry

end jenna_filter_change_15th_is_March_l112_112694


namespace alice_walks_miles_each_morning_l112_112828

theorem alice_walks_miles_each_morning (x : ℕ) :
  (5 * x + 5 * 12 = 110) → x = 10 :=
by
  intro h
  -- Proof omitted
  sorry

end alice_walks_miles_each_morning_l112_112828


namespace second_butcher_packages_l112_112102

theorem second_butcher_packages (a b c: ℕ) (weight_per_package total_weight: ℕ)
    (first_butcher_packages: ℕ) (third_butcher_packages: ℕ)
    (cond1: a = 10) (cond2: b = 8) (cond3: weight_per_package = 4)
    (cond4: total_weight = 100):
    c = (total_weight - (first_butcher_packages * weight_per_package + third_butcher_packages * weight_per_package)) / weight_per_package →
    c = 7 := 
by 
  have first_butcher_packages := 10
  have third_butcher_packages := 8
  have weight_per_package := 4
  have total_weight := 100
  sorry

end second_butcher_packages_l112_112102


namespace problem_statement_l112_112204

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (-2^x + b) / (2^(x+1) + a)

theorem problem_statement :
  (∀ (x : ℝ), f (x) 2 1 = -f (-x) 2 1) ∧
  (∀ (t : ℝ), f (t^2 - 2*t) 2 1 + f (2*t^2 - k) 2 1 < 0 → k < -1/3) :=
by
  sorry

end problem_statement_l112_112204


namespace tank_weight_when_full_l112_112942

theorem tank_weight_when_full (p q : ℝ) (x y : ℝ)
  (h1 : x + (3/4) * y = p)
  (h2 : x + (1/3) * y = q) :
  x + y = (8/5) * p - (8/5) * q :=
by
  sorry

end tank_weight_when_full_l112_112942


namespace jessica_milk_problem_l112_112492

theorem jessica_milk_problem (gallons_owned : ℝ) (gallons_given : ℝ) : gallons_owned = 5 → gallons_given = 16 / 3 → gallons_owned - gallons_given = -(1 / 3) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- sorry

end jessica_milk_problem_l112_112492


namespace comparison_neg_fractions_l112_112921

theorem comparison_neg_fractions (a b : ℚ) (ha : a = -5/6) (hb : b = -4/5) :
  a < b ↔ -5/6 < -4/5 := 
by 
  have h : 5/6 > 4/5 := sorry
  exact h


end comparison_neg_fractions_l112_112921


namespace solve_fraction_eq_zero_l112_112772

theorem solve_fraction_eq_zero (x : ℝ) (h₁ : 3 - x = 0) (h₂ : 4 + 2 * x ≠ 0) : x = 3 :=
by sorry

end solve_fraction_eq_zero_l112_112772


namespace find_circles_tangent_to_axes_l112_112123

def tangent_to_axes_and_passes_through (R : ℝ) (P : ℝ × ℝ) :=
  let center := (R, R)
  (P.1 - R) ^ 2 + (P.2 - R) ^ 2 = R ^ 2

theorem find_circles_tangent_to_axes (x y : ℝ) :
  (tangent_to_axes_and_passes_through 1 (2, 1) ∧ tangent_to_axes_and_passes_through 1 (x, y)) ∨
  (tangent_to_axes_and_passes_through 5 (2, 1) ∧ tangent_to_axes_and_passes_through 5 (x, y)) :=
by {
  sorry
}

end find_circles_tangent_to_axes_l112_112123


namespace min_candidates_for_same_score_l112_112645

theorem min_candidates_for_same_score :
  (∃ S : ℕ, S ≥ 25 ∧ (∀ elect : Fin S → Fin 12, ∃ s : Fin 12, ∃ a b c : Fin S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ elect a = s ∧ elect b = s ∧ elect c = s)) := 
sorry

end min_candidates_for_same_score_l112_112645


namespace correct_calculation_l112_112417

theorem correct_calculation :
  (-7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2) ∧
  ¬ (2 * x + 3 * y = 5 * x * y) ∧
  ¬ (6 * x^2 - (-x^2) = 5 * x^2) ∧
  ¬ (4 * m * n - 3 * m * n = 1) :=
by
  sorry

end correct_calculation_l112_112417


namespace smallest_k_for_polygon_l112_112055

-- Definitions and conditions
def equiangular_decagon_interior_angle : ℝ := 144

-- Question transformation into a proof problem
theorem smallest_k_for_polygon (k : ℕ) (hk : k > 1) :
  (∀ (n2 : ℕ), n2 = 10 * k → ∃ (interior_angle : ℝ), interior_angle = k * equiangular_decagon_interior_angle ∧
  n2 ≥ 3) → k = 2 :=
by
  sorry

end smallest_k_for_polygon_l112_112055


namespace intersection_volume_is_zero_l112_112533

-- Definitions of the regions
def region1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 2
def region2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1

-- Main theorem stating the volume of their intersection
theorem intersection_volume_is_zero : 
  ∀ (x y z : ℝ), region1 x y z ∧ region2 x y z → (x = 0 ∧ y = 0 ∧ z = 2) := 
sorry

end intersection_volume_is_zero_l112_112533


namespace binom_n_2_l112_112410

theorem binom_n_2 (n : ℕ) (h : 1 < n) : (nat.choose n 2) = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l112_112410


namespace minimum_value_l112_112488

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end minimum_value_l112_112488


namespace minimum_value_of_x_plus_2y_l112_112196

-- Definitions for the problem conditions
def isPositive (z : ℝ) : Prop := z > 0

def condition (x y : ℝ) : Prop := 
  isPositive x ∧ isPositive y ∧ (x + 2*y + 2*x*y = 8) 

-- Statement of the problem
theorem minimum_value_of_x_plus_2y (x y : ℝ) (h : condition x y) : x + 2 * y ≥ 4 :=
sorry

end minimum_value_of_x_plus_2y_l112_112196


namespace students_correct_answers_l112_112688

theorem students_correct_answers
  (total_questions : ℕ)
  (correct_score per_question : ℕ)
  (incorrect_penalty : ℤ)
  (xiao_ming_score xiao_hong_score xiao_hua_score : ℤ)
  (xm_correct_answers xh_correct_answers xh_correct_answers : ℕ)
  (total : ℕ)
  (h_1 : total_questions = 10)
  (h_2 : correct_score = 10)
  (h_3 : incorrect_penalty = -3)
  (h_4 : xiao_ming_score = 87)
  (h_5 : xiao_hong_score = 74)
  (h_6 : xiao_hua_score = 9)
  (h_xm : xm_correct_answers = total_questions - (xiao_ming_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hong_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hua_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (expected : total = 20) :
  xm_correct_answers + xh_correct_answers + xh_correct_answers = total := 
sorry

end students_correct_answers_l112_112688


namespace water_bottles_needed_l112_112958

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l112_112958


namespace period_f_2pi_max_value_f_exists_max_f_l112_112718

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem period_f_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem max_value_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 := by
  sorry

-- Optional: Existence of the maximum value.
theorem exists_max_f : ∃ x : ℝ, f x = Real.sin 1 + 1 := by
  sorry

end period_f_2pi_max_value_f_exists_max_f_l112_112718


namespace gcd_polynomial_eval_l112_112715

theorem gcd_polynomial_eval (b : ℤ) (h : ∃ (k : ℤ), b = 570 * k) :
  Int.gcd (4 * b ^ 3 + b ^ 2 + 5 * b + 95) b = 95 := by
  sorry

end gcd_polynomial_eval_l112_112715


namespace shaded_region_area_l112_112366

noncomputable def radius_large : ℝ := 10
noncomputable def radius_small : ℝ := 4

theorem shaded_region_area :
  let area_large := Real.pi * radius_large^2 
  let area_small := Real.pi * radius_small^2 
  (area_large - 2 * area_small) = 68 * Real.pi :=
by
  sorry

end shaded_region_area_l112_112366


namespace problem_l112_112525

/-
A problem involving natural numbers a and b
where:
1. Their sum is 20000
2. One of them (b) is divisible by 5
3. Erasing the units digit of b gives the other number a

We want to prove their difference is 16358
-/

def nat_sum_and_difference (a b : ℕ) : Prop :=
  a + b = 20000 ∧
  b % 5 = 0 ∧
  (b % 10 = 0 ∧ b / 10 = a ∨ b % 10 = 5 ∧ (b - 5) / 10 = a)

theorem problem (a b : ℕ) (h : nat_sum_and_difference a b) : b - a = 16358 := 
  sorry

end problem_l112_112525


namespace box_dimensions_l112_112318

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  -- We assume the proof is correct based on given conditions
  sorry

end box_dimensions_l112_112318


namespace necessary_but_not_sufficient_l112_112253

theorem necessary_but_not_sufficient
  (x y : ℝ) :
  (x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧ ¬ (x^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 2*x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l112_112253


namespace xiao_yu_reading_days_l112_112661

-- Definition of Xiao Yu's reading problem
def number_of_pages_per_day := 15
def total_number_of_days := 24
def additional_pages_per_day := 3
def new_number_of_pages_per_day := number_of_pages_per_day + additional_pages_per_day
def total_pages := number_of_pages_per_day * total_number_of_days
def new_total_number_of_days := total_pages / new_number_of_pages_per_day

-- Theorem statement in Lean 4
theorem xiao_yu_reading_days : new_total_number_of_days = 20 :=
  sorry

end xiao_yu_reading_days_l112_112661


namespace gear_revolutions_l112_112686

theorem gear_revolutions (t : ℝ) (r_p r_q : ℝ) (h1 : r_q = 40) (h2 : t = 20)
 (h3 : (r_q / 60) * t = ((r_p / 60) * t) + 10) :
 r_p = 10 :=
 sorry

end gear_revolutions_l112_112686


namespace expected_lifetime_of_flashlight_at_least_4_l112_112240

-- Definitions for the lifetimes of the lightbulbs
variable (ξ η : ℝ)

-- Condition: The expected lifetime of the red lightbulb is 4 years.
axiom E_η_eq_4 : 𝔼[η] = 4

-- Definition stating the lifetime of the flashlight
def T := max ξ η

theorem expected_lifetime_of_flashlight_at_least_4 
  (h : 𝔼η = 4) :
  𝔼[max ξ η] ≥ 4 :=
by {
  sorry
}

end expected_lifetime_of_flashlight_at_least_4_l112_112240


namespace project_completion_time_l112_112564

theorem project_completion_time 
    (w₁ w₂ : ℕ) 
    (d₁ d₂ : ℕ) 
    (fraction₁ fraction₂ : ℝ)
    (h_work_fraction : fraction₁ = 1/2)
    (h_work_time : d₁ = 6)
    (h_first_workforce : w₁ = 90)
    (h_second_workforce : w₂ = 60)
    (h_fraction_done_by_first_team : w₁ * d₁ * (1 / 1080) = fraction₁)
    (h_fraction_done_by_second_team : w₂ * d₂ * (1 / 1080) = fraction₂)
    (h_total_fraction : fraction₂ = 1 - fraction₁) :
    d₂ = 9 :=
by 
  sorry

end project_completion_time_l112_112564


namespace linear_equation_in_two_variables_l112_112534

/--
Prove that Equation C (3x - 1 = 2 - 5y) is a linear equation in two variables 
given the equations in conditions.
-/
theorem linear_equation_in_two_variables :
  ∀ (x y : ℝ),
  (2 * x + 3 = x - 5) →
  (x * y + y = 2) →
  (3 * x - 1 = 2 - 5 * y) →
  (2 * x + (3 / y) = 7) →
  ∃ (A B C : ℝ), A * x + B * y = C :=
by 
  sorry

end linear_equation_in_two_variables_l112_112534


namespace calc_derivative_at_pi_over_2_l112_112708

noncomputable def f (x: ℝ) : ℝ := Real.exp x * Real.cos x

theorem calc_derivative_at_pi_over_2 : (deriv f) (Real.pi / 2) = -Real.exp (Real.pi / 2) :=
by
  sorry

end calc_derivative_at_pi_over_2_l112_112708


namespace propositions_imply_implication_l112_112009

theorem propositions_imply_implication (p q r : Prop) :
  ( ((p ∧ q ∧ ¬r) → ((p ∧ q) → r) = False) ∧ 
    ((¬p ∧ q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((p ∧ ¬q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((¬p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r) = True) ) → 
  ( (∀ (x : ℕ), x = 3) ) :=
by
  sorry

end propositions_imply_implication_l112_112009


namespace find_constants_l112_112239

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![0, 4]
]

def I : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![1, 0],
  ![0, 1]
]

theorem find_constants (c d : ℚ) : 
  (N⁻¹ = c • N + d • I) ↔ (c = -1/12 ∧ d = 7/12) :=
by
  sorry

end find_constants_l112_112239


namespace white_sox_wins_l112_112013

theorem white_sox_wins 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (games_lost : ℕ)
  (win_loss_difference : ℤ) 
  (total_games_condition : total_games = 162) 
  (lost_games_condition : games_lost = 63) 
  (win_loss_diff_condition : (games_won : ℤ) - games_lost = win_loss_difference) 
  (win_loss_difference_value : win_loss_difference = 36) 
  : games_won = 99 :=
by
  sorry

end white_sox_wins_l112_112013


namespace evaluate_expression_l112_112016

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end evaluate_expression_l112_112016


namespace symmetric_points_l112_112200

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_points_l112_112200


namespace unfilted_roses_remaining_l112_112742

/-- Initial number of roses received by Danielle --/
def initial_roses : ℕ := 2 * 12

/-- Number of roses received after trade --/
def roses_after_trade : ℕ := initial_roses + 12

/-- Number of roses after first night when half wilted --/
def after_first_night : ℕ := roses_after_trade / 2

/-- Total roses after removing wilted ones from the first night --/
def remaining_after_first_night : ℕ := roses_after_trade - after_first_night

/-- Number of roses after second night when half wilted --/
def after_second_night : ℕ := remaining_after_first_night / 2

/-- Total roses after removing wilted ones from the second night --/
def remaining_after_second_night : ℕ := remaining_after_first_night - after_second_night

/-- Prove that the number of unwilted roses remaining at the end is 9 --/
theorem unfilted_roses_remaining : remaining_after_second_night = 9 := by
  dsimp [initial_roses, roses_after_trade, after_first_night, remaining_after_first_night, after_second_night, remaining_after_second_night]
  sorry

end unfilted_roses_remaining_l112_112742


namespace problem_lean_statement_l112_112387

theorem problem_lean_statement (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 * x ^ 2 + 5 * x + 3)
  (h2 : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 0 :=
by sorry

end problem_lean_statement_l112_112387


namespace collinear_points_l112_112328

axiom collinear (A B C : ℝ × ℝ × ℝ) : Prop

theorem collinear_points (c d : ℝ) (h : collinear (2, c, d) (c, 3, d) (c, d, 4)) : c + d = 6 :=
sorry

end collinear_points_l112_112328


namespace athlete_heartbeats_calculation_l112_112153

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l112_112153


namespace rational_solves_abs_eq_l112_112193

theorem rational_solves_abs_eq (x : ℚ) : |6 + x| = |6| + |x| → 0 ≤ x := 
sorry

end rational_solves_abs_eq_l112_112193


namespace fresh_grape_weight_l112_112840

variable (D : ℝ) (F : ℝ)

axiom dry_grape_weight : D = 66.67
axiom fresh_grape_water_content : F * 0.25 = D * 0.75

theorem fresh_grape_weight : F = 200.01 :=
by sorry

end fresh_grape_weight_l112_112840


namespace angle_A_is_pi_div_3_length_b_l112_112974

open Real

theorem angle_A_is_pi_div_3
  (A B C : ℝ) (a b c : ℝ)
  (hABC : A + B + C = π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (hm : m = (sqrt 3, cos (π - A) - 1))
  (hn : n = (cos (π / 2 - A), 1))
  (horthogonal : m.1 * n.1 + m.2 * n.2 = 0) :
  A = π / 3 := 
sorry

theorem length_b 
  (A B : ℝ) (a b : ℝ)
  (hA : A = π / 3)
  (ha : a = 2)
  (hcosB : cos B = sqrt 3 / 3) :
  b = 4 * sqrt 2 / 3 :=
sorry

end angle_A_is_pi_div_3_length_b_l112_112974


namespace value_expression_possible_values_l112_112248

open Real

noncomputable def value_expression (a b : ℝ) : ℝ :=
  a^2 + 2 * a * b + b^2 + 2 * a^2 * b + 2 * a * b^2 + a^2 * b^2

theorem value_expression_possible_values (a b : ℝ)
  (h1 : (a / b) + (b / a) = 5 / 2)
  (h2 : a - b = 3 / 2) :
  value_expression a b = 0 ∨ value_expression a b = 81 :=
sorry

end value_expression_possible_values_l112_112248


namespace moon_iron_percentage_l112_112639

variables (x : ℝ) -- percentage of iron in the moon

-- Given conditions
def carbon_percentage_of_moon : ℝ := 0.20
def mass_of_moon : ℝ := 250
def mass_of_mars : ℝ := 2 * mass_of_moon
def mass_of_other_elements_on_mars : ℝ := 150
def composition_same (m : ℝ) (x : ℝ) := 
  (x / 100 * m + carbon_percentage_of_moon * m + (100 - x - 20) / 100 * m) = m

-- Theorem statement
theorem moon_iron_percentage : x = 50 :=
by
  sorry

end moon_iron_percentage_l112_112639


namespace simplify_fraction_l112_112091

theorem simplify_fraction :
  10 * (15 / 8) * (-40 / 45) = -(50 / 3) :=
sorry

end simplify_fraction_l112_112091


namespace mul_99_105_l112_112827

theorem mul_99_105 : 99 * 105 = 10395 := 
by
  -- Annotations and imports are handled; only the final Lean statement provided as requested.
  sorry

end mul_99_105_l112_112827


namespace angles_proof_l112_112595

-- Definitions (directly from the conditions)
variable {θ₁ θ₂ θ₃ θ₄ : ℝ}

def complementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 90
def supplementary (θ₃ θ₄ : ℝ) : Prop := θ₃ + θ₄ = 180

-- Theorem statement
theorem angles_proof (h1 : complementary θ₁ θ₂) (h2 : supplementary θ₃ θ₄) (h3 : θ₁ = θ₃) :
  θ₂ + 90 = θ₄ :=
by
  sorry

end angles_proof_l112_112595


namespace cos_identity_l112_112754

theorem cos_identity (x : ℝ) : 
  4 * Real.cos x * Real.cos (x + π / 3) * Real.cos (x - π / 3) = Real.cos (3 * x) :=
by
  sorry

end cos_identity_l112_112754


namespace curves_intersect_at_three_points_l112_112657

theorem curves_intersect_at_three_points :
  (∀ x y a : ℝ, (x^2 + y^2 = 4 * a^2) ∧ (y = x^2 - 2 * a) → a = 1) := sorry

end curves_intersect_at_three_points_l112_112657


namespace sin_symmetry_value_l112_112224

theorem sin_symmetry_value (ϕ : ℝ) (hϕ₀ : 0 < ϕ) (hϕ₁ : ϕ < π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end sin_symmetry_value_l112_112224


namespace cubic_polynomial_solution_l112_112024

theorem cubic_polynomial_solution (x : ℝ) :
  x^3 + 6*x^2 + 11*x + 6 = 12 ↔ x = -1 ∨ x = -2 ∨ x = -3 := by
  sorry

end cubic_polynomial_solution_l112_112024


namespace power_sum_inequality_l112_112350

theorem power_sum_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
by sorry

end power_sum_inequality_l112_112350


namespace move_point_right_l112_112751

theorem move_point_right (A B : ℤ) (hA : A = -3) (hAB : B = A + 4) : B = 1 :=
by {
  sorry
}

end move_point_right_l112_112751


namespace expected_red_hair_americans_l112_112880

theorem expected_red_hair_americans (prob_red_hair : ℝ) (sample_size : ℕ) :
  prob_red_hair = 1 / 6 → sample_size = 300 → (prob_red_hair * sample_size = 50) := by
  intros
  sorry

end expected_red_hair_americans_l112_112880


namespace original_number_is_142857_l112_112804

-- Definitions based on conditions
def six_digit_number (x : ℕ) : ℕ := 100000 + x
def moved_digit_number (x : ℕ) : ℕ := 10 * x + 1

-- Lean statement of the equivalent problem
theorem original_number_is_142857 : ∃ x, six_digit_number x = 142857 ∧ moved_digit_number x = 3 * six_digit_number x :=
  sorry

end original_number_is_142857_l112_112804


namespace inv_composition_l112_112816

theorem inv_composition (f g : ℝ → ℝ) (hf : Function.Bijective f) (hg : Function.Bijective g) (h : ∀ x, f⁻¹ (g x) = 2 * x - 4) : 
  g⁻¹ (f (-3)) = 1 / 2 :=
by
  sorry

end inv_composition_l112_112816


namespace sequence_satisfies_n_squared_l112_112467

theorem sequence_satisfies_n_squared (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) :
  ∀ n, a n = n^2 :=
by
  -- sorry
  sorry

end sequence_satisfies_n_squared_l112_112467


namespace reduced_price_l112_112927

theorem reduced_price (P Q : ℝ) (h : P ≠ 0) (h₁ : 900 = Q * P) (h₂ : 900 = (Q + 6) * (0.90 * P)) : 0.90 * P = 15 :=
by 
  sorry

end reduced_price_l112_112927


namespace absolute_sum_l112_112838

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem absolute_sum : 
    (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end absolute_sum_l112_112838


namespace sum_of_digits_7_pow_17_mod_100_l112_112916

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l112_112916


namespace class_6_1_students_l112_112513

noncomputable def number_of_students : ℕ :=
  let n := 30
  n

theorem class_6_1_students (n : ℕ) (t : ℕ) (h1 : (n + 1) * t = 527) (h2 : n % 5 = 0) : n = 30 :=
  by
  sorry

end class_6_1_students_l112_112513


namespace max_intersections_three_circles_one_line_l112_112117

theorem max_intersections_three_circles_one_line : 
  ∀ (C1 C2 C3 : Circle) (L : Line), 
  same_paper C1 C2 C3 L → 
  max_intersections C1 C2 C3 L = 12 := 
sorry

end max_intersections_three_circles_one_line_l112_112117


namespace positive_m_of_quadratic_has_one_real_root_l112_112727

theorem positive_m_of_quadratic_has_one_real_root : 
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, x^2 + 6 * m * x + m = 0 → x = -3 * m) :=
by
  sorry

end positive_m_of_quadratic_has_one_real_root_l112_112727


namespace inequality_greater_sqrt_two_l112_112254

theorem inequality_greater_sqrt_two (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
by 
  sorry

end inequality_greater_sqrt_two_l112_112254


namespace all_pets_combined_l112_112762

def Teddy_initial_dogs : Nat := 7
def Teddy_initial_cats : Nat := 8
def Teddy_initial_rabbits : Nat := 6

def Teddy_adopted_dogs : Nat := 2
def Teddy_adopted_rabbits : Nat := 4

def Ben_dogs : Nat := 3 * Teddy_initial_dogs
def Ben_cats : Nat := 2 * Teddy_initial_cats

def Dave_dogs : Nat := (Teddy_initial_dogs + Teddy_adopted_dogs) - 4
def Dave_cats : Nat := Teddy_initial_cats + 13
def Dave_rabbits : Nat := 3 * Teddy_initial_rabbits

def Teddy_current_dogs : Nat := Teddy_initial_dogs + Teddy_adopted_dogs
def Teddy_current_cats : Nat := Teddy_initial_cats
def Teddy_current_rabbits : Nat := Teddy_initial_rabbits + Teddy_adopted_rabbits

def Teddy_total : Nat := Teddy_current_dogs + Teddy_current_cats + Teddy_current_rabbits
def Ben_total : Nat := Ben_dogs + Ben_cats
def Dave_total : Nat := Dave_dogs + Dave_cats + Dave_rabbits

def total_pets_combined : Nat := Teddy_total + Ben_total + Dave_total

theorem all_pets_combined : total_pets_combined = 108 :=
by
  sorry

end all_pets_combined_l112_112762


namespace Q_transform_l112_112901

def rotate_180_clockwise (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  (2 * px - qx, 2 * py - qy)

def reflect_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (py, px)

def Q := (8, -11) -- from the reverse transformations

theorem Q_transform (c d : ℝ) :
  (reflect_y_equals_x (rotate_180_clockwise (2, -3) (c, d)) = (5, -4)) → (d - c = -19) :=
by sorry

end Q_transform_l112_112901


namespace matchstick_triangles_l112_112406

/-- Using 12 equal-length matchsticks, it is possible to form an isosceles triangle, an equilateral triangle, and a right-angled triangle without breaking or overlapping the matchsticks. --/
theorem matchstick_triangles :
  ∃ a b c : ℕ, a + b + c = 12 ∧ (a = b ∨ b = c ∨ a = c) ∧ (a * a + b * b = c * c ∨ a = b ∧ b = c) :=
by
  sorry

end matchstick_triangles_l112_112406


namespace correct_operation_l112_112418

theorem correct_operation (a : ℝ) : 
    (a ^ 2 + a ^ 4 ≠ a ^ 6) ∧ 
    (a ^ 2 * a ^ 3 ≠ a ^ 6) ∧ 
    (a ^ 3 / a ^ 2 = a) ∧ 
    ((a ^ 2) ^ 3 ≠ a ^ 5) :=
by
  sorry

end correct_operation_l112_112418


namespace cube_root_simplification_l112_112293

theorem cube_root_simplification (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : b = 1) : 3
  := sorry

end cube_root_simplification_l112_112293


namespace product_of_fractions_l112_112004

theorem product_of_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = 1 / 7 :=
  sorry

end product_of_fractions_l112_112004


namespace min_xy_sum_is_7_l112_112461

noncomputable def min_xy_sum (x y : ℝ) : ℝ := 
x + y

theorem min_xy_sum_is_7 (x y : ℝ) (h1 : x > 1) (h2 : y > 2) (h3 : (x - 1) * (y - 2) = 4) : 
  min_xy_sum x y = 7 := by 
  sorry

end min_xy_sum_is_7_l112_112461


namespace log_40_cannot_be_directly_calculated_l112_112706

theorem log_40_cannot_be_directly_calculated (log_3 log_5 : ℝ) (h1 : log_3 = 0.4771) (h2 : log_5 = 0.6990) : 
  ¬ (exists (log_40 : ℝ), (log_40 = (log_3 + log_5) + log_40)) :=
by {
  sorry
}

end log_40_cannot_be_directly_calculated_l112_112706


namespace trapezoid_AD_length_l112_112528

-- Definitions for the problem setup
variables {A B C D O P : Type}
variables (f : A → B → C → D → Prop)
variables (g : A → D → C → D → Prop)
variables (h : A → C → D → B → Prop)

-- The main theorem we want to prove
theorem trapezoid_AD_length
  (ABCD_trapezoid : f A B C D)
  (BC_CD_same : ∀ {x y}, (g B C x y → y = 43) ∧ (g B C x y → x = 43))
  (AD_perpendicular_BD : ∀ {x y}, h A D x y → ∃ (p : P), p = O)
  (O_intersection_AC_BD : g A C O B)
  (P_midpoint_BD : ∃ (p : P), p = P ∧ ∀ (x y : B ∗ D), y = x / 2)
  (OP_length : ∃ (len : ℝ), len = 11) :
  let m := 4 in let n := 190 in m + n = 194 := sorry

end trapezoid_AD_length_l112_112528


namespace car_distance_l112_112425

-- Define the conditions
def speed := 162  -- speed of the car in km/h
def time := 5     -- time taken in hours

-- Define the distance calculation
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

-- State the theorem
theorem car_distance : distance speed time = 810 := by
  -- Proof goes here
  sorry

end car_distance_l112_112425


namespace cattle_train_speed_is_correct_l112_112300

-- Given conditions as definitions
def cattle_train_speed (x : ℝ) : ℝ := x
def diesel_train_speed (x : ℝ) : ℝ := x - 33
def cattle_train_distance (x : ℝ) : ℝ := 6 * x
def diesel_train_distance (x : ℝ) : ℝ := 12 * (x - 33)

-- Statement to prove
theorem cattle_train_speed_is_correct (x : ℝ) :
  cattle_train_distance x + diesel_train_distance x = 1284 → 
  x = 93.33 :=
by
  intros h
  sorry

end cattle_train_speed_is_correct_l112_112300


namespace jeanne_should_buy_more_tickets_l112_112998

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end jeanne_should_buy_more_tickets_l112_112998


namespace find_a_plus_b_l112_112968

theorem find_a_plus_b (a b : ℚ) (y : ℚ) (x : ℚ) :
  (y = a + b / x) →
  (2 = a + b / (-2 : ℚ)) →
  (3 = a + b / (-6 : ℚ)) →
  a + b = 13 / 2 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_plus_b_l112_112968


namespace triangle_circle_property_l112_112266

-- Let a, b, and c be the lengths of the sides of a right triangle, where c is the hypotenuse.
variables {a b c : ℝ}

-- Let varrho_b be the radius of the circle inscribed around the leg b of the triangle.
variable {varrho_b : ℝ}

-- Assume the relationship a^2 + b^2 = c^2 (Pythagorean theorem).
axiom right_triangle : a^2 + b^2 = c^2

-- Prove that b + c = a + 2 * varrho_b
theorem triangle_circle_property (h : a^2 + b^2 = c^2) (radius_condition : varrho_b = (a*b)/(a+c-b)) : 
  b + c = a + 2 * varrho_b :=
sorry

end triangle_circle_property_l112_112266


namespace LukaLemonadeSolution_l112_112878

def LukaLemonadeProblem : Prop :=
  ∃ (L S W : ℕ), 
    (S = 3 * L) ∧
    (W = 3 * S) ∧
    (L = 4) ∧
    (W = 36)

theorem LukaLemonadeSolution : LukaLemonadeProblem :=
  by sorry

end LukaLemonadeSolution_l112_112878


namespace problem1_problem2_problem3_problem4_l112_112685

theorem problem1 : -16 - (-12) - 24 + 18 = -10 := 
by
  sorry

theorem problem2 : 0.125 + (1 / 4) + (-9 / 4) + (-0.25) = -2 := 
by
  sorry

theorem problem3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := 
by
  sorry

theorem problem4 : (-2 + 3) * 3 - (-2)^3 / 4 = 5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l112_112685


namespace donny_remaining_money_l112_112332

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end donny_remaining_money_l112_112332


namespace digging_project_depth_l112_112302

theorem digging_project_depth : 
  ∀ (P : ℕ) (D : ℝ), 
  (12 * P) * (25 * 30 * D) / 12 = (12 * P) * (75 * 20 * 50) / 12 → 
  D = 100 :=
by
  intros P D h
  sorry

end digging_project_depth_l112_112302


namespace max_marks_l112_112677

theorem max_marks (M : ℝ) (h1 : 80 + 10 = 90) (h2 : 0.30 * M = 90) : M = 300 :=
by
  sorry

end max_marks_l112_112677


namespace largest_divisor_of_n_l112_112726

-- Definitions and conditions from the problem
def is_positive_integer (n : ℕ) := n > 0
def is_divisible_by (a b : ℕ) := ∃ k : ℕ, a = k * b

-- Lean 4 statement encapsulating the problem
theorem largest_divisor_of_n (n : ℕ) (h1 : is_positive_integer n) (h2 : is_divisible_by (n * n) 72) : 
  ∃ v : ℕ, v = 12 ∧ is_divisible_by n v := 
sorry

end largest_divisor_of_n_l112_112726


namespace sequence_sixth_term_l112_112712

theorem sequence_sixth_term (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) 
  (h2 : ∀ n :ℕ, n > 0 → a (n + 1) = 2 * a n) 
  (h3 : a 1 = 3) : 
  a 6 = 96 := 
by
  sorry

end sequence_sixth_term_l112_112712


namespace days_to_complete_job_l112_112062

theorem days_to_complete_job (m₁ m₂ d₁ d₂ total_man_days : ℝ)
    (h₁ : m₁ = 30)
    (h₂ : d₁ = 8)
    (h₃ : total_man_days = 240)
    (h₄ : total_man_days = m₁ * d₁)
    (h₅ : m₂ = 40) :
    d₂ = total_man_days / m₂ := by
  sorry

end days_to_complete_job_l112_112062


namespace convert_base_8_to_7_l112_112170

def convert_base_8_to_10 (n : Nat) : Nat :=
  let d2 := n / 100 % 10
  let d1 := n / 10 % 10
  let d0 := n % 10
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def convert_base_10_to_7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else 
    let rec helper (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else helper (n / 7) ((n % 7) :: acc)
    helper n []

def represent_in_base_7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem convert_base_8_to_7 :
  represent_in_base_7 (convert_base_10_to_7 (convert_base_8_to_10 653)) = 1150 :=
by
  sorry

end convert_base_8_to_7_l112_112170


namespace Ryan_hours_learning_Spanish_is_4_l112_112963

-- Definitions based on conditions
def hoursLearningChinese : ℕ := 5
def hoursLearningSpanish := ∃ x : ℕ, hoursLearningChinese = x + 1

-- Proof Statement
theorem Ryan_hours_learning_Spanish_is_4 : ∃ x : ℕ, hoursLearningSpanish ∧ x = 4 :=
by
  sorry

end Ryan_hours_learning_Spanish_is_4_l112_112963


namespace lines_intersect_at_3_6_l112_112101

theorem lines_intersect_at_3_6 (c d : ℝ) 
  (h1 : 3 = 2 * 6 + c) 
  (h2 : 6 = 2 * 3 + d) : 
  c + d = -9 := by 
  sorry

end lines_intersect_at_3_6_l112_112101


namespace unwilted_roses_proof_l112_112743

-- Conditions
def initial_roses : Nat := 2 * 12
def traded_roses : Nat := 12
def first_day_roses (r: Nat) : Nat := r / 2
def second_day_roses (r: Nat) : Nat := r / 2

-- Initial number of roses
def total_roses : Nat := initial_roses + traded_roses

-- Number of unwilted roses after two days
def unwilted_roses : Nat := second_day_roses (first_day_roses total_roses)

-- Formal statement to prove
theorem unwilted_roses_proof : unwilted_roses = 9 := by
  sorry

end unwilted_roses_proof_l112_112743


namespace triple_solutions_l112_112690

theorem triple_solutions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) ↔ a! + b! = 2 ^ c! :=
by
  sorry

end triple_solutions_l112_112690


namespace working_light_bulbs_count_l112_112756

def lamps := 60
def bulbs_per_lamp := 7

def fraction_with_2_burnt := 1 / 3
def fraction_with_1_burnt := 1 / 4
def fraction_with_3_burnt := 1 / 5

def lamps_with_2_burnt := fraction_with_2_burnt * lamps
def lamps_with_1_burnt := fraction_with_1_burnt * lamps
def lamps_with_3_burnt := fraction_with_3_burnt * lamps
def lamps_with_all_working := lamps - (lamps_with_2_burnt + lamps_with_1_burnt + lamps_with_3_burnt)

def working_bulbs_from_2_burnt := lamps_with_2_burnt * (bulbs_per_lamp - 2)
def working_bulbs_from_1_burnt := lamps_with_1_burnt * (bulbs_per_lamp - 1)
def working_bulbs_from_3_burnt := lamps_with_3_burnt * (bulbs_per_lamp - 3)
def working_bulbs_from_all_working := lamps_with_all_working * bulbs_per_lamp

def total_working_bulbs := working_bulbs_from_2_burnt + working_bulbs_from_1_burnt + working_bulbs_from_3_burnt + working_bulbs_from_all_working

theorem working_light_bulbs_count : total_working_bulbs = 329 := by
  sorry

end working_light_bulbs_count_l112_112756


namespace pool_volume_l112_112581

variable {rate1 rate2 : ℕ}
variables {hose1 hose2 hose3 hose4 : ℕ}
variables {time : ℕ}

def hose1_rate := 2
def hose2_rate := 2
def hose3_rate := 3
def hose4_rate := 3
def fill_time := 25

def total_rate := hose1_rate + hose2_rate + hose3_rate + hose4_rate

theorem pool_volume (h : hose1 = hose1_rate ∧ hose2 = hose2_rate ∧ hose3 = hose3_rate ∧ hose4 = hose4_rate ∧ time = fill_time):
  total_rate * 60 * time = 15000 := 
by 
  sorry

end pool_volume_l112_112581


namespace length_of_equal_pieces_l112_112131

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l112_112131


namespace find_range_of_x_l112_112568

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end find_range_of_x_l112_112568


namespace new_socks_bought_l112_112841

theorem new_socks_bought :
  ∀ (original_socks throw_away new_socks total_socks : ℕ),
    original_socks = 28 →
    throw_away = 4 →
    total_socks = 60 →
    total_socks = original_socks - throw_away + new_socks →
    new_socks = 36 :=
by
  intros original_socks throw_away new_socks total_socks h_original h_throw h_total h_eq
  sorry

end new_socks_bought_l112_112841


namespace problem_solution_l112_112609

noncomputable def corrected_angles 
  (x1_star x2_star x3_star : ℝ) 
  (σ : ℝ) 
  (h_sum : x1_star + x2_star + x3_star - 180.0 = 0)  
  (h_var : σ^2 = (0.1)^2) : ℝ × ℝ × ℝ :=
  let Δ := 2.0 / 3.0 * 0.667
  let Δx1 := Δ * (σ^2 / 2)
  let Δx2 := Δ * (σ^2 / 2)
  let Δx3 := Δ * (σ^2 / 2)
  let corrected_x1 := x1_star - Δx1
  let corrected_x2 := x2_star - Δx2
  let corrected_x3 := x3_star - Δx3
  (corrected_x1, corrected_x2, corrected_x3)

theorem problem_solution :
  corrected_angles 31 62 89 (0.1) sorry sorry = (30.0 + 40 / 60, 61.0 + 40 / 60, 88 + 20 / 60) := 
  sorry

end problem_solution_l112_112609


namespace gcd_of_sum_and_product_l112_112282

theorem gcd_of_sum_and_product (x y : ℕ) (h1 : x + y = 1130) (h2 : x * y = 100000) : Int.gcd x y = 2 := 
sorry

end gcd_of_sum_and_product_l112_112282


namespace common_ratio_geom_series_l112_112699

theorem common_ratio_geom_series :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -16/21
  let a₃ : ℚ := -64/63
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -4/3 := 
by
  sorry

end common_ratio_geom_series_l112_112699


namespace isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l112_112982

def isosceles_right_triangle_initial_leg_length (x : ℝ) (h : ℝ) : Prop :=
  x + 4 * ((x + 4) / 2) ^ 2 = x * x / 2 + 112 

def isosceles_right_triangle_legs_correct (a b : ℝ) (h : ℝ) : Prop :=
  a = 26 ∧ b = 26 * Real.sqrt 2

theorem isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm :
  ∃ (x : ℝ) (h : ℝ), isosceles_right_triangle_initial_leg_length x h ∧ 
                       isosceles_right_triangle_legs_correct x (x * Real.sqrt 2) h := 
by
  sorry

end isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l112_112982


namespace waiter_customer_count_l112_112000

def initial_customers := 33
def customers_left := 31
def new_customers := 26

theorem waiter_customer_count :
  (initial_customers - customers_left) + new_customers = 28 :=
by
  -- This is a placeholder for the proof that can be filled later.
  sorry

end waiter_customer_count_l112_112000


namespace avg_weight_section_b_l112_112905

/-- Definition of the average weight of section B based on given conditions --/
theorem avg_weight_section_b :
  let W_A := 50
  let W_class := 54.285714285714285
  let num_A := 40
  let num_B := 30
  let total_class_weight := (num_A + num_B) * W_class
  let total_A_weight := num_A * W_A
  let total_B_weight := total_class_weight - total_A_weight
  let W_B := total_B_weight / num_B
  W_B = 60 :=
by
  sorry

end avg_weight_section_b_l112_112905


namespace marcia_savings_l112_112001

def hat_price := 60
def regular_price (n : ℕ) := n * hat_price
def discount_price (discount_percentage: ℕ) (price: ℕ) := price - (price * discount_percentage) / 100
def promotional_price := hat_price + discount_price 25 hat_price + discount_price 35 hat_price

theorem marcia_savings : (regular_price 3 - promotional_price) * 100 / regular_price 3 = 20 :=
by
  -- The proof steps would follow here.
  sorry

end marcia_savings_l112_112001


namespace square_field_area_l112_112516

theorem square_field_area (speed time perimeter : ℕ) (h1 : speed = 20) (h2 : time = 4) (h3 : perimeter = speed * time) :
  ∃ s : ℕ, perimeter = 4 * s ∧ s * s = 400 :=
by
  -- All conditions and definitions are stated, proof is skipped using sorry
  sorry

end square_field_area_l112_112516


namespace number_of_dozen_eggs_to_mall_l112_112233

-- Define the conditions as assumptions
def number_of_dozen_eggs_collected (x : Nat) : Prop :=
  x = 2 * 8

def number_of_dozen_eggs_to_market (x : Nat) : Prop :=
  x = 3

def number_of_dozen_eggs_for_pie (x : Nat) : Prop :=
  x = 4

def number_of_dozen_eggs_to_charity (x : Nat) : Prop :=
  x = 4

-- The theorem stating the answer to the problem
theorem number_of_dozen_eggs_to_mall 
  (h1 : ∃ x, number_of_dozen_eggs_collected x)
  (h2 : ∃ x, number_of_dozen_eggs_to_market x)
  (h3 : ∃ x, number_of_dozen_eggs_for_pie x)
  (h4 : ∃ x, number_of_dozen_eggs_to_charity x)
  : ∃ z, z = 5 := 
sorry

end number_of_dozen_eggs_to_mall_l112_112233


namespace necessary_but_not_sufficient_condition_l112_112221

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x : ℝ, x^2 - 3 * x + 2 = 0 ∧ x ≠ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l112_112221


namespace find_constants_l112_112830

theorem find_constants :
  ∃ P Q : ℚ, (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 7) / (x^2 - 3 * x - 18) = P / (x - 6) + Q / (x + 3)) ∧
    P = 31 / 9 ∧ Q = 5 / 9 :=
by
  sorry

end find_constants_l112_112830


namespace polynomial_solution_l112_112589

variable (P : ℚ) -- Assuming P is a constant polynomial

theorem polynomial_solution (P : ℚ) 
  (condition : P + (2 : ℚ) * X^2 + (5 : ℚ) * X - (2 : ℚ) = (2 : ℚ) * X^2 + (5 : ℚ) * X + (4 : ℚ)): 
  P = 6 := 
  sorry

end polynomial_solution_l112_112589


namespace sum_of_digits_of_greatest_prime_divisor_l112_112416

-- Define the number 32767
def number : ℕ := 32767

-- Assert that 32767 is 2^15 - 1
lemma number_def : number = 2^15 - 1 := by
  sorry

-- State that 151 is the greatest prime divisor of 32767
lemma greatest_prime_divisor : Nat.Prime 151 ∧ ∀ p : ℕ, Nat.Prime p → p ∣ number → p ≤ 151 := by
  sorry

-- Calculate the sum of the digits of 151
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Conclude the sum of the digits of the greatest prime divisor is 7
theorem sum_of_digits_of_greatest_prime_divisor : sum_of_digits 151 = 7 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_l112_112416


namespace length_of_room_l112_112392

theorem length_of_room (b : ℕ) (t : ℕ) (L : ℕ) (blue_tiles : ℕ) (tile_area : ℕ) (total_area : ℕ) (effective_area : ℕ) (blue_area : ℕ) :
  b = 10 →
  t = 2 →
  blue_tiles = 16 →
  tile_area = t * t →
  total_area = (L - 4) * (b - 4) →
  blue_area = blue_tiles * tile_area →
  2 * blue_area = 3 * total_area →
  L = 20 :=
by
  intros h_b h_t h_blue_tiles h_tile_area h_total_area h_blue_area h_proportion
  sorry

end length_of_room_l112_112392


namespace cubic_roots_arithmetic_progression_l112_112327

theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x : ℝ, x^3 + a * x^2 + b * x + c = 0) ∧ 
  (∀ x : ℝ, x^3 + a * x^2 + b * x + c = 0 → 
    (x = p - t ∨ x = p ∨ x = p + t) ∧ 
    (a ≠ 0)) ↔ 
  ((a * b / 3) - 2 * (a^3) / 27 - c = 0 ∧ (a^3 / 3) - b ≥ 0) := 
by sorry

end cubic_roots_arithmetic_progression_l112_112327


namespace range_of_values_l112_112209

theorem range_of_values (a b : ℝ) : (∀ x : ℝ, x < 1 → ax + b > 2 * (x + 1)) → b > 4 := 
by
  sorry

end range_of_values_l112_112209


namespace total_number_of_subjects_l112_112163

-- Definitions from conditions
def average_marks_5_subjects (total_marks : ℕ) : Prop :=
  74 * 5 = total_marks

def marks_in_last_subject (marks : ℕ) : Prop :=
  marks = 74

def total_average_marks (n : ℕ) (total_marks : ℕ) : Prop :=
  74 * n = total_marks

-- Lean 4 statement
theorem total_number_of_subjects (n total_marks total_marks_5 last_subject_marks : ℕ)
  (h1 : total_average_marks n total_marks)
  (h2 : average_marks_5_subjects total_marks_5)
  (h3 : marks_in_last_subject last_subject_marks)
  (h4 : total_marks = total_marks_5 + last_subject_marks) :
  n = 6 :=
sorry

end total_number_of_subjects_l112_112163


namespace find_c_plus_d_l112_112858

theorem find_c_plus_d (a b c d : ℝ) (h1 : a + b = 12) (h2 : b + c = 9) (h3 : a + d = 6) : 
  c + d = 3 := 
sorry

end find_c_plus_d_l112_112858


namespace wire_ratio_theorem_l112_112165

theorem wire_ratio_theorem
  {pieces_bonnie : ℕ} {length_piece_bonnie : ℕ} {volume_bonnie : ℕ}
  {pieces_roark : ℕ} {length_piece_roark : ℕ} {volume_roark : ℕ}
  (h_length_bonnie : pieces_bonnie = 12)
  (h_piece_length_bonnie : length_piece_bonnie = 6)
  (h_volume_bonnie : volume_bonnie = 6^3)
  (h_pieces_roark : pieces_roark = volume_bonnie)
  (h_piece_length_roark : length_piece_roark = 12)
  (h_volume_roark : volume_roark = 1) :
  (pieces_bonnie * length_piece_bonnie : ℚ) / (pieces_roark * length_piece_roark : ℚ) = 1 / 36 := 
sorry

end wire_ratio_theorem_l112_112165


namespace betty_cookies_brownies_l112_112314

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_cookies_brownies_l112_112314


namespace abs_x_equals_4_l112_112082

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l112_112082


namespace probability_value_at_least_75_cents_l112_112298

-- Given conditions
def box_contains (pennies nickels quarters : ℕ) : Prop :=
  pennies = 4 ∧ nickels = 3 ∧ quarters = 5

def draw_without_replacement (total_coins : ℕ) (drawn_coins : ℕ) : Prop :=
  total_coins = 12 ∧ drawn_coins = 5

def equal_probability (chosen_probability : ℚ) (total_coins : ℕ) : Prop :=
  chosen_probability = 1/total_coins

-- Probability that the value of coins drawn is at least 75 cents
theorem probability_value_at_least_75_cents
  (pennies nickels quarters total_coins drawn_coins : ℕ)
  (chosen_probability : ℚ) :
  box_contains pennies nickels quarters →
  draw_without_replacement total_coins drawn_coins →
  equal_probability chosen_probability total_coins →
  chosen_probability = 1/792 :=
by
  intros
  sorry

end probability_value_at_least_75_cents_l112_112298


namespace complement_of_A_in_U_l112_112977

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}
def complement : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l112_112977


namespace range_of_m_l112_112218

theorem range_of_m (m : ℝ) (H : ∀ x, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) : m < -1 / 2 :=
sorry

end range_of_m_l112_112218


namespace probability_brick_in_box_l112_112908

noncomputable def find_y (n : ℕ) (a b c : ℕ) : ℕ :=
if a < b ∧ b < c ∧ c < n then n - 3 else n

theorem probability_brick_in_box (c1 c2 c3 d1 d2 d3 : ℕ) 
  (hc1 : 1 ≤ c1) (hc2 : 1 ≤ c2) (hc3 : 1 ≤ c3)
  (hd1 : 1 ≤ d1) (hd2 : 1 ≤ d2) (hd3 : 1 ≤ d3)
  (hlow : c1 ≤ 50) (hrest : d1 ≠ c1 ∧ d1 ≠ c2 ∧ d1 ≠ c3 
  ∧ d2 ≠ c1 ∧ d2 ≠ c2 ∧ d2 ≠ c3 ∧ d3 ≠ c1 
  ∧ d3 ≠ c2 ∧ d3 ≠ c3 ∧ d1 ≠ d2) : 
  (c1 + c2 + c3 + d1 + d2 + d3) % 47 = 5 :=
by sorry

end probability_brick_in_box_l112_112908


namespace bryce_raisins_l112_112056

theorem bryce_raisins (x : ℕ) (h1 : x = 2 * (x - 8)) : x = 16 :=
by
  sorry

end bryce_raisins_l112_112056


namespace least_number_subtracted_378461_l112_112124

def least_number_subtracted (n : ℕ) : ℕ :=
  n % 13

theorem least_number_subtracted_378461 : least_number_subtracted 378461 = 5 :=
by
  -- actual proof would go here
  sorry

end least_number_subtracted_378461_l112_112124


namespace more_stable_scores_l112_112760

-- Define the variances for Student A and Student B
def variance_A : ℝ := 38
def variance_B : ℝ := 15

-- Formulate the theorem
theorem more_stable_scores : variance_A > variance_B → "B" = "B" :=
by
  intro h
  sorry

end more_stable_scores_l112_112760


namespace min_value_M_l112_112375

noncomputable def a (x y z : ℝ) : ℝ := log z + log (x / (y * z) + 1)
noncomputable def b (x y z : ℝ) : ℝ := log (1 / x) + log (x * y * z + 1)
noncomputable def c (x y z : ℝ) : ℝ := log y + log (1 / (x * y * z) + 1)
noncomputable def M (x y z : ℝ) : ℝ := max (a x y z) (max (b x y z) (c x y z))

theorem min_value_M : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ M x y z = log 2 := by
  sorry

end min_value_M_l112_112375


namespace express_in_scientific_notation_l112_112335

theorem express_in_scientific_notation :
  102200 = 1.022 * 10^5 :=
sorry

end express_in_scientific_notation_l112_112335


namespace total_heartbeats_during_race_l112_112160

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l112_112160


namespace max_two_digit_number_divisible_by_23_l112_112433

theorem max_two_digit_number_divisible_by_23 :
  ∃ n : ℕ, 
    (n < 100) ∧ 
    (1000 ≤ n * 109) ∧ 
    (n * 109 < 10000) ∧ 
    (n % 23 = 0) ∧ 
    (n / 23 < 10) ∧ 
    (n = 69) :=
by {
  sorry
}

end max_two_digit_number_divisible_by_23_l112_112433


namespace sum_of_consecutive_integers_l112_112103

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end sum_of_consecutive_integers_l112_112103


namespace single_solution_inequality_l112_112823

theorem single_solution_inequality (a : ℝ) :
  (∃! (x : ℝ), abs (x^2 + 2 * a * x + 3 * a) ≤ 2) ↔ a = 1 ∨ a = 2 := 
sorry

end single_solution_inequality_l112_112823


namespace max_points_of_intersection_l112_112118

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end max_points_of_intersection_l112_112118


namespace find_percentage_second_alloy_l112_112162

open Real

def percentage_copper_second_alloy (percentage_alloy1: ℝ) (ounces_alloy1: ℝ) (percentage_desired_alloy: ℝ) (total_ounces: ℝ) (percentage_second_alloy: ℝ) : Prop :=
  let copper_ounces_alloy1 := percentage_alloy1 * ounces_alloy1 / 100
  let desired_copper_ounces := percentage_desired_alloy * total_ounces / 100
  let needed_copper_ounces := desired_copper_ounces - copper_ounces_alloy1
  let ounces_alloy2 := total_ounces - ounces_alloy1
  (needed_copper_ounces / ounces_alloy2) * 100 = percentage_second_alloy

theorem find_percentage_second_alloy :
  percentage_copper_second_alloy 18 45 19.75 108 21 :=
by
  sorry

end find_percentage_second_alloy_l112_112162


namespace parabola_directrix_tangent_circle_l112_112352

theorem parabola_directrix_tangent_circle (p : ℝ) (h_pos : 0 < p) (h_tangent: ∃ x : ℝ, (x = p/2) ∧ (x-5)^2 + (0:ℝ)^2 = 25) : p = 20 :=
sorry

end parabola_directrix_tangent_circle_l112_112352


namespace sound_heard_in_4_seconds_l112_112546

/-- Given the distance between a boy and his friend is 1200 meters,
    the speed of the car is 108 km/hr, and the speed of sound is 330 m/s,
    the duration after which the friend hears the whistle is 4 seconds. -/
theorem sound_heard_in_4_seconds :
  let distance := 1200  -- distance in meters
  let speed_of_car_kmh := 108  -- speed of car in km/hr
  let speed_of_sound := 330  -- speed of sound in m/s
  let speed_of_car := speed_of_car_kmh * 1000 / 3600  -- convert km/hr to m/s
  let effective_speed_of_sound := speed_of_sound - speed_of_car
  let time := distance / effective_speed_of_sound
  time = 4 := 
by
  sorry

end sound_heard_in_4_seconds_l112_112546


namespace quadrilateral_circumscribed_l112_112072

structure ConvexQuad (A B C D : Type) := 
  (is_convex : True)
  (P : Type)
  (interior : True)
  (angle_APB_angle_CPD_eq_angle_BPC_angle_DPA : True)
  (angle_PAD_angle_PCD_eq_angle_PAB_angle_PCB : True)
  (angle_PDC_angle_PBC_eq_angle_PDA_angle_PBA : True)

theorem quadrilateral_circumscribed (A B C D : Type) (quad : ConvexQuad A B C D) : True := 
sorry

end quadrilateral_circumscribed_l112_112072


namespace solve_system_of_equations_l112_112887

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x + 2 * y = 5 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 := 
by
  sorry

end solve_system_of_equations_l112_112887


namespace value_of_expression_l112_112075

theorem value_of_expression (r s : ℝ) (h₁ : 3 * r^2 - 5 * r - 7 = 0) (h₂ : 3 * s^2 - 5 * s - 7 = 0) : 
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
sorry

end value_of_expression_l112_112075


namespace final_values_comparison_l112_112627

theorem final_values_comparison :
  let AA_initial : ℝ := 100
  let BB_initial : ℝ := 100
  let CC_initial : ℝ := 100
  let AA_year1 := AA_initial * 1.20
  let BB_year1 := BB_initial * 0.75
  let CC_year1 := CC_initial
  let AA_year2 := AA_year1 * 0.80
  let BB_year2 := BB_year1 * 1.25
  let CC_year2 := CC_year1
  AA_year2 = 96 ∧ BB_year2 = 93.75 ∧ CC_year2 = 100 ∧ BB_year2 < AA_year2 ∧ AA_year2 < CC_year2 :=
by {
  -- Definitions from conditions
  let AA_initial : ℝ := 100;
  let BB_initial : ℝ := 100;
  let CC_initial : ℝ := 100;
  let AA_year1 := AA_initial * 1.20;
  let BB_year1 := BB_initial * 0.75;
  let CC_year1 := CC_initial;
  let AA_year2 := AA_year1 * 0.80;
  let BB_year2 := BB_year1 * 1.25;
  let CC_year2 := CC_year1;

  -- Use sorry to skip the actual proof
  sorry
}

end final_values_comparison_l112_112627


namespace positive_difference_median_mode_l112_112787

-- Definition of the data set
def data : List ℕ := [12, 13, 14, 15, 15, 22, 22, 22, 31, 31, 38, 39, 40, 41, 42, 43, 52, 58, 59]

-- Definition of the mode
def mode (l : List ℕ) : ℕ := 22  -- Specific to the data set provided

-- Definition of the median
def median (l : List ℕ) : ℕ := 31  -- Specific to the data set provided

-- Proof statement
theorem positive_difference_median_mode : 
  (median data - mode data) = 9 := by 
  sorry

end positive_difference_median_mode_l112_112787


namespace ninety_eight_squared_l112_112322

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l112_112322


namespace team_E_speed_l112_112795

noncomputable def average_speed_team_E (d t_E t_A v_A v_E : ℝ) : Prop :=
  d = 300 ∧
  t_A = t_E - 3 ∧
  v_A = v_E + 5 ∧
  d = v_E * t_E ∧
  d = v_A * t_A →
  v_E = 20

theorem team_E_speed : ∃ (v_E : ℝ), average_speed_team_E 300 t_E (t_E - 3) (v_E + 5) v_E :=
by
  sorry

end team_E_speed_l112_112795


namespace find_U_l112_112390

-- Declare the variables and conditions
def digits : Set ℤ := {1, 2, 3, 4, 5, 6}

theorem find_U (P Q R S T U : ℤ) :
  -- Condition: Digits are distinct and each is in {1, 2, 3, 4, 5, 6}
  (P ∈ digits) ∧ (Q ∈ digits) ∧ (R ∈ digits) ∧ (S ∈ digits) ∧ (T ∈ digits) ∧ (U ∈ digits) ∧
  (P ≠ Q) ∧ (P ≠ R) ∧ (P ≠ S) ∧ (P ≠ T) ∧ (P ≠ U) ∧
  (Q ≠ R) ∧ (Q ≠ S) ∧ (Q ≠ T) ∧ (Q ≠ U) ∧
  (R ≠ S) ∧ (R ≠ T) ∧ (R ≠ U) ∧ (S ≠ T) ∧ (S ≠ U) ∧ (T ≠ U) ∧
  -- Condition: The three-digit number PQR is divisible by 9
  (100 * P + 10 * Q + R) % 9 = 0 ∧
  -- Condition: The three-digit number QRS is divisible by 4
  (10 * Q + R) % 4 = 0 ∧
  -- Condition: The three-digit number RST is divisible by 3
  (10 * R + S) % 3 = 0 ∧
  -- Condition: The sum of the digits is divisible by 5
  (P + Q + R + S + T + U) % 5 = 0
  -- Conclusion: U = 4
  → U = 4 :=
by sorry

end find_U_l112_112390


namespace negation_of_proposition_l112_112976

theorem negation_of_proposition :
  (¬ ∃ m : ℝ, 1 / (m^2 + m - 6) > 0) ↔ (∀ m : ℝ, (1 / (m^2 + m - 6) < 0) ∨ (m^2 + m - 6 = 0)) :=
by
  sorry

end negation_of_proposition_l112_112976


namespace elizabeth_stickers_count_l112_112175

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l112_112175


namespace range_of_m_l112_112865

-- Define the first circle
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y + 1 = 0

-- Define the second circle
def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - m = 0

-- Lean statement for the proof problem
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by sorry

end range_of_m_l112_112865


namespace analogical_reasoning_ineq_l112_112351

-- Formalization of the conditions and the theorem to be proved

def positive (a : ℕ → ℝ) (n : ℕ) := ∀ i, 1 ≤ i → i ≤ n → a i > 0

theorem analogical_reasoning_ineq {a : ℕ → ℝ} (hpos : positive a 4) (hsum : a 1 + a 2 + a 3 + a 4 = 1) : 
  (1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4) ≥ 16 := 
sorry

end analogical_reasoning_ineq_l112_112351


namespace projection_plane_right_angle_l112_112593

-- Given conditions and definitions
def is_right_angle (α β : ℝ) : Prop := α = 90 ∧ β = 90
def is_parallel_to_side (plane : ℝ → ℝ → Prop) (side : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, plane x y ↔ a * x + b * y = c ∧ ∃ d e : ℝ, ∀ x y : ℝ, side x y ↔ d * x + e * y = 90

theorem projection_plane_right_angle (plane : ℝ → ℝ → Prop) (side1 side2 : ℝ → ℝ → Prop) :
  is_right_angle (90 : ℝ) (90 : ℝ) →
  (is_parallel_to_side plane side1 ∨ is_parallel_to_side plane side2) →
  ∃ α β : ℝ, is_right_angle α β :=
by 
  sorry

end projection_plane_right_angle_l112_112593


namespace power_function_point_l112_112852

theorem power_function_point (n : ℕ) (hn : 2^n = 8) : n = 3 := 
by
  sorry

end power_function_point_l112_112852


namespace measure_angle_ABC_approx_l112_112876

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

noncomputable def angle_ABC : ℝ :=
  let A : ℝ × ℝ × ℝ := (-3, 1, 5)
  let B : ℝ × ℝ × ℝ := (-4, -2, 4)
  let C : ℝ × ℝ × ℝ := (-5, -2, 6)
  let AB := distance A B
  let AC := distance A C
  let BC := distance B C
  real.acos ((AB ^ 2 + BC ^ 2 - AC ^ 2) / (2 * AB * BC))

theorem measure_angle_ABC_approx : angle_ABC ≈ 87.13 / 180 * real.pi :=
  sorry

end measure_angle_ABC_approx_l112_112876


namespace lana_needs_to_sell_more_muffins_l112_112495

/--
Lana aims to sell 20 muffins at the bake sale.
She sells 12 muffins in the morning.
She sells another 4 in the afternoon.
How many more muffins does Lana need to sell to hit her goal?
-/
theorem lana_needs_to_sell_more_muffins (goal morningSales afternoonSales : ℕ)
  (h_goal : goal = 20) (h_morning : morningSales = 12) (h_afternoon : afternoonSales = 4) :
  goal - (morningSales + afternoonSales) = 4 :=
by
  sorry

end lana_needs_to_sell_more_muffins_l112_112495


namespace total_animals_is_200_l112_112733

-- Definitions for the conditions
def num_cows : Nat := 40
def num_sheep : Nat := 56
def num_goats : Nat := 104

-- The theorem to prove the total number of animals is 200
theorem total_animals_is_200 : num_cows + num_sheep + num_goats = 200 := by
  sorry

end total_animals_is_200_l112_112733


namespace mh_range_l112_112851

theorem mh_range (x m : ℝ) (h : 1 / 3 < x ∧ x < 1 / 2) (hx : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 := 
sorry

end mh_range_l112_112851


namespace Susie_earnings_l112_112093

theorem Susie_earnings :
  let price_per_slice := 3 in
  let slices_sold := 24 in
  let price_per_pizza := 15 in
  let pizzas_sold := 3 in
  let earnings_from_slices := price_per_slice * slices_sold in
  let earnings_from_pizzas := price_per_pizza * pizzas_sold in
  let total_earnings := earnings_from_slices + earnings_from_pizzas in
  total_earnings = 117 :=
by
  sorry

end Susie_earnings_l112_112093


namespace text_messages_in_march_l112_112613

/-
Jared sent text messages each month according to the formula:
  T_n = n^3 - n^2 + n
We need to prove that the number of text messages Jared will send in March
(which is the 5th month) is given by T_5 = 105.
-/

def T (n : ℕ) : ℕ := n^3 - n^2 + n

theorem text_messages_in_march : T 5 = 105 :=
by
  -- proof goes here
  sorry

end text_messages_in_march_l112_112613


namespace ninety_eight_squared_l112_112321

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l112_112321


namespace books_left_over_after_repacking_l112_112483

def initial_boxes : ℕ := 1430
def books_per_initial_box : ℕ := 42
def weight_per_book : ℕ := 200 -- in grams
def books_per_new_box : ℕ := 45
def max_weight_per_new_box : ℕ := 9000 -- in grams (9 kg)

def total_books : ℕ := initial_boxes * books_per_initial_box

theorem books_left_over_after_repacking :
  total_books % books_per_new_box = 30 :=
by
  -- Proof goes here
  sorry

end books_left_over_after_repacking_l112_112483


namespace prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l112_112086

def X := {-1, 0, 1}
def A_accuracy := 0.5
def B_accuracy := 0.6

theorem prob_X_distribution :
  ∀ (x : X),
  (x = -1) → (P(X = -1) = 0.3) ∧
  (x = 0) → (P(X = 0) = 0.5) ∧
  (x = 1) → (P(X = 1) = 0.2) := by sorry

theorem prob_tie :
  P(tie) = 0.2569 := by sorry

def Y := {2, 3, 4}

theorem prob_Y_distribution :
  ∀ (y : Y),
  (y = 2) → (P(Y = 2) = 0.13) ∧
  (y = 3) → (P(Y = 3) = 0.13) ∧
  (y = 4) → (P(Y = 4) = 0.74) := by sorry

theorem expected_Y :
  E(Y) = 3.61 := by sorry

end prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l112_112086


namespace find_function_l112_112994

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1) :
  ∀ x : ℝ, f x = if x ≠ 0.5 then 1 / (0.5 - x) else 0.5 :=
by
  sorry

end find_function_l112_112994


namespace meals_neither_vegan_kosher_nor_gluten_free_l112_112078

def total_clients : ℕ := 50
def n_vegan : ℕ := 10
def n_kosher : ℕ := 12
def n_gluten_free : ℕ := 6
def n_both_vegan_kosher : ℕ := 3
def n_both_vegan_gluten_free : ℕ := 4
def n_both_kosher_gluten_free : ℕ := 2
def n_all_three : ℕ := 1

/-- The number of clients who need a meal that is neither vegan, kosher, nor gluten-free. --/
theorem meals_neither_vegan_kosher_nor_gluten_free :
  total_clients - (n_vegan + n_kosher + n_gluten_free - n_both_vegan_kosher - n_both_vegan_gluten_free - n_both_kosher_gluten_free + n_all_three) = 30 :=
by
  sorry

end meals_neither_vegan_kosher_nor_gluten_free_l112_112078


namespace continuous_stripe_probability_l112_112953

noncomputable def probability_continuous_stripe_encircle_cube : ℚ :=
  let total_combinations : ℕ := 2^6
  let favor_combinations : ℕ := 3 * 4 -- 3 pairs of parallel faces, with 4 valid combinations each
  favor_combinations / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe_encircle_cube = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l112_112953


namespace find_range_of_x_l112_112567

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end find_range_of_x_l112_112567


namespace weather_condition_l112_112434

theorem weather_condition (T : ℝ) (windy : Prop) (kites_will_fly : Prop) 
  (h1 : (T > 25 ∧ windy) → kites_will_fly) 
  (h2 : ¬ kites_will_fly) : T ≤ 25 ∨ ¬ windy :=
by 
  sorry

end weather_condition_l112_112434


namespace value_of_e_l112_112061

variable (e : ℝ)
noncomputable def eq1 : Prop :=
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - e) / 18 = (2 * 0.3 + 4) / 3)

theorem value_of_e : eq1 e → e = 6 := by
  intro h
  sorry

end value_of_e_l112_112061


namespace average_headcount_is_correct_l112_112413

/-- The student headcount data for the specified semesters -/
def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

noncomputable def average_headcount : ℕ :=
  (student_headcount.sum) / student_headcount.length

theorem average_headcount_is_correct : average_headcount = 11029 := by
  sorry

end average_headcount_is_correct_l112_112413


namespace sum_terms_a1_a17_l112_112842

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 := by
  sorry

end sum_terms_a1_a17_l112_112842


namespace house_number_units_digit_is_five_l112_112381

/-- Define the house number as a two-digit number -/
def is_two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- Define the properties for the statements -/
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_power_of_prime (n : ℕ) : Prop := ∃ p : ℕ, Nat.Prime p ∧ p ^ Nat.log p n = n
def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def has_digit_seven (n : ℕ) : Prop := (n / 10 = 7 ∨ n % 10 = 7)

/-- The theorem stating that the units digit of the house number is 5 -/
theorem house_number_units_digit_is_five (n : ℕ) 
  (h1 : is_two_digit_number n)
  (h2 : (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n) ∨ 
        (¬is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n))
  : n % 10 = 5 := 
sorry

end house_number_units_digit_is_five_l112_112381


namespace sum_of_digits_7_pow_17_mod_100_l112_112915

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l112_112915


namespace geometric_sequence_eleventh_term_l112_112169

theorem geometric_sequence_eleventh_term (a₁ : ℚ) (r : ℚ) (n : ℕ) (hₐ : a₁ = 5) (hᵣ : r = 2 / 3) (hₙ : n = 11) :
  (a₁ * r^(n - 1) = 5120 / 59049) :=
by
  -- conditions of the problem
  rw [hₐ, hᵣ, hₙ]
  sorry

end geometric_sequence_eleventh_term_l112_112169


namespace tan_theta_neq_2sqrt2_l112_112219

theorem tan_theta_neq_2sqrt2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < Real.pi) (h₁ : Real.sin θ + Real.cos θ = (2 * Real.sqrt 2 - 1) / 3) : Real.tan θ = -2 * Real.sqrt 2 := by
  sorry

end tan_theta_neq_2sqrt2_l112_112219


namespace length_AB_l112_112046

theorem length_AB 
  (P : ℝ × ℝ) 
  (hP : 3 * P.1 + 4 * P.2 + 8 = 0)
  (C : ℝ × ℝ := (1, 1))
  (A B : ℝ × ℝ)
  (hA : (A.1 - 1)^2 + (A.2 - 1)^2 = 1 ∧ (3 * A.1 + 4 * A.2 + 8 ≠ 0))
  (hB : (B.1 - 1)^2 + (B.2 - 1)^2 = 1 ∧ (3 * B.1 + 4 * B.2 + 8 ≠ 0)) :
  dist A B = 4 * Real.sqrt 2 / 3 := sorry

end length_AB_l112_112046


namespace existence_of_epsilon_and_u_l112_112347

theorem existence_of_epsilon_and_u (n : ℕ) (h : 0 < n) :
  ∀ k ≥ 1, ∃ ε : ℝ, (0 < ε ∧ ε < 1 / k) ∧
  (∀ (a : Fin n → ℝ), (∀ i, 0 < a i) → ∃ u > 0, ∀ i, ε < (u * a i - ⌊u * a i⌋) ∧ (u * a i - ⌊u * a i⌋) < 1 / k) :=
by {
  sorry
}

end existence_of_epsilon_and_u_l112_112347


namespace vertex_below_x_axis_l112_112363

theorem vertex_below_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + a < 0) → a < 1 :=
by 
  sorry

end vertex_below_x_axis_l112_112363


namespace P_at_10_l112_112768

-- Define the main properties of the polynomial
variable (P : ℤ → ℤ)
axiom quadratic (a b c : ℤ) : (∀ n : ℤ, P n = a * n^2 + b * n + c) 

-- Conditions for the polynomial
axiom int_coefficients : ∃ (a b c : ℤ), ∀ n : ℤ, P n = a * n^2 + b * n + c
axiom relatively_prime (n : ℤ) (hn : 0 < n) : Int.gcd (P n) n = 1 ∧ Int.gcd (P (P n)) n = 1
axiom P_at_3 : P 3 = 89

-- The main theorem to prove
theorem P_at_10 : P 10 = 859 := by sorry

end P_at_10_l112_112768


namespace watched_commercials_eq_100_l112_112071

variable (x : ℕ) -- number of people who watched commercials
variable (s : ℕ := 27) -- number of subscribers
variable (rev_comm : ℝ := 0.50) -- revenue per commercial
variable (rev_sub : ℝ := 1.00) -- revenue per subscriber
variable (total_rev : ℝ := 77.00) -- total revenue

theorem watched_commercials_eq_100 (h : rev_comm * (x : ℝ) + rev_sub * (s : ℝ) = total_rev) : x = 100 := by
  sorry

end watched_commercials_eq_100_l112_112071


namespace base7_multiplication_l112_112452

theorem base7_multiplication (a b : ℕ) (h₁ : a = 3 * 7^2 + 2 * 7^1 + 5) (h₂ : b = 3) : 
  let ab := (a * b) in
  nat_repr_ab7 3111 := nat_repr'_base ab 7 :=
begin
  sorry
end

end base7_multiplication_l112_112452


namespace digit_2023_in_fractional_expansion_l112_112021

theorem digit_2023_in_fractional_expansion :
  ∃ d : ℕ, (d = 4) ∧ (∃ n_block : ℕ, n_block = 6 ∧ (∃ p : Nat, p = 2023 ∧ ∃ r : ℕ, r = p % n_block ∧ r = 1)) :=
sorry

end digit_2023_in_fractional_expansion_l112_112021


namespace min_value_frac_function_l112_112577

theorem min_value_frac_function (x : ℝ) (h : x > -1) : (x^2 / (x + 1)) ≥ 0 :=
sorry

end min_value_frac_function_l112_112577


namespace factor_expression_l112_112965

variable (y : ℝ)

theorem factor_expression : 
  6*y*(y + 2) + 15*(y + 2) + 12 = 3*(2*y + 5)*(y + 2) :=
sorry

end factor_expression_l112_112965


namespace friends_bought_boxes_l112_112693

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56
def pencils_per_box : ℕ := rainbow_colors

theorem friends_bought_boxes (emily_box : ℕ := 1) :
  (total_pencils / pencils_per_box) - emily_box = 7 := by
  sorry

end friends_bought_boxes_l112_112693


namespace find_pairs_of_nonneg_ints_l112_112573

theorem find_pairs_of_nonneg_ints (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (m, n) = (9, 3) ∨ (m, n) = (6, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end find_pairs_of_nonneg_ints_l112_112573


namespace no_solution_for_k_eq_2_l112_112449

theorem no_solution_for_k_eq_2 :
  ∀ m n : ℕ, m ≠ n → ¬ (lcm m n - gcd m n = 2 * (m - n)) :=
by
  sorry

end no_solution_for_k_eq_2_l112_112449


namespace flower_garden_width_l112_112614

-- Define the conditions
def gardenArea : ℝ := 143.2
def gardenLength : ℝ := 4
def gardenWidth : ℝ := 35.8

-- The proof statement (question to answer)
theorem flower_garden_width :
    gardenWidth = gardenArea / gardenLength :=
by 
  sorry

end flower_garden_width_l112_112614


namespace ratio_equality_l112_112689

theorem ratio_equality (x y u v p q : ℝ) (h : (x / y) * (u / v) * (p / q) = 1) :
  (x / y) * (u / v) * (p / q) = 1 := 
by sorry

end ratio_equality_l112_112689


namespace bus_trip_distance_l112_112299

variable (D : ℝ) (S : ℝ := 50)

theorem bus_trip_distance :
  (D / (S + 5) = D / S - 1) → D = 550 := by
  sorry

end bus_trip_distance_l112_112299


namespace not_possible_2020_parts_possible_2023_parts_l112_112812

-- Define the initial number of parts and the operation that adds two parts
def initial_parts : Nat := 1
def operation (n : Nat) : Nat := n + 2

theorem not_possible_2020_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2020) : False :=
sorry

theorem possible_2023_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2023) : True :=
sorry

end not_possible_2020_parts_possible_2023_parts_l112_112812


namespace eric_has_correct_green_marbles_l112_112183

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end eric_has_correct_green_marbles_l112_112183


namespace n_equals_23_l112_112372

open Finset

theorem n_equals_23 {k n : ℕ} (hk : k ≥ 6) (hn : n = 2 * k - 1)
  (T : Finset (Vector ℕ n)) (x y : Vector ℕ n)
  (d : Vector ℕ n → Vector ℕ n → ℕ)
  (h_dist : ∀ x y : Vector ℕ n, d x y = x.toList.zipWith (λ a b, if a ≠ b then 1 else 0) y.toList |>.sum)
  (S : Finset (Vector ℕ n))
  (hS : S.card = 2 ^ k)
  (h_unique : ∀ x ∈ T, ∃! y ∈ S, d x y ≤ 3) :
  n = 23 :=
sorry

end n_equals_23_l112_112372


namespace simplify_expr_l112_112669

-- Define variables and conditions
variables (x y a b c : ℝ)

-- State the theorem
theorem simplify_expr : 
  (2 - y) * 24 * (x - y + 2 * (a - 2 - 3 * c) * a - 2 * b + c) = 
  2 + 4 * b^2 - a * b - c^2 :=
sorry

end simplify_expr_l112_112669


namespace relationship_among_values_l112_112588

-- Assume there exists a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition 1: f is strictly increasing on (0, 3)
def increasing_on_0_to_3 : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f x < f y

-- Condition 2: f(x + 3) is an even function
def even_function_shifted : Prop :=
  ∀ x : ℝ, f (x + 3) = f (-(x + 3))

-- The theorem we need to prove
theorem relationship_among_values 
  (h1 : increasing_on_0_to_3 f)
  (h2 : even_function_shifted f) :
  f (9/2) < f 2 ∧ f 2 < f (7/2) :=
sorry

end relationship_among_values_l112_112588


namespace fraction_less_than_mode_l112_112867

def mode {α : Type*} [decidable_eq α] (l : list α) : α :=
(l.nth_le (l.indexes l.max).head (by simp)).get_or_else l.head

theorem fraction_less_than_mode (l : list ℕ) (h_mode : ∃ m, mode l = m ∧ ∀ x ∈ l, x ≤ m) 
  (h_fraction : (l.count(< mode l) : ℚ) / l.length = 2 / 9) :
  ∃ l, (l.count(< mode l) : ℚ) / l.length = 2 / 9 :=
by {
  sorry
}

end fraction_less_than_mode_l112_112867


namespace oranges_to_apples_equiv_apples_for_36_oranges_l112_112369

-- Conditions
def weight_equiv (oranges apples : ℕ) : Prop :=
  9 * oranges = 6 * apples

-- Question (Theorem to Prove)
theorem oranges_to_apples_equiv_apples_for_36_oranges:
  ∃ (apples : ℕ), apples = 24 ∧ weight_equiv 36 apples :=
by
  use 24
  sorry

end oranges_to_apples_equiv_apples_for_36_oranges_l112_112369


namespace minimum_value_l112_112201

theorem minimum_value (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (2 / (x + 3 * y) + 1 / (x - y)) = (3 + 2 * Real.sqrt 2) / 2 := sorry

end minimum_value_l112_112201


namespace alpha_minus_beta_l112_112850

-- Providing the conditions
variable (α β : ℝ)
variable (hα1 : 0 < α ∧ α < Real.pi / 2)
variable (hβ1 : 0 < β ∧ β < Real.pi / 2)
variable (hα2 : Real.tan α = 4 / 3)
variable (hβ2 : Real.tan β = 1 / 7)

-- The goal is to show that α - β = π / 4 given the conditions
theorem alpha_minus_beta :
  α - β = Real.pi / 4 := by
  sorry

end alpha_minus_beta_l112_112850


namespace Malcom_has_more_cards_l112_112562

-- Define the number of cards Brandon has
def Brandon_cards : ℕ := 20

-- Define the number of cards Malcom has initially, to be found
def Malcom_initial_cards (n : ℕ) := n

-- Define the given condition: Malcom has 14 cards left after giving away half of his cards
def Malcom_half_condition (n : ℕ) := n / 2 = 14

-- Prove that Malcom had 8 more cards than Brandon initially
theorem Malcom_has_more_cards (n : ℕ) (h : Malcom_half_condition n) :
  Malcom_initial_cards n - Brandon_cards = 8 :=
by
  sorry

end Malcom_has_more_cards_l112_112562


namespace mowers_mow_l112_112782

theorem mowers_mow (mowers hectares days mowers_new days_new : ℕ)
  (h1 : 3 * 3 * days = 3 * hectares)
  (h2 : 5 * days_new = 5 * (days_new * hectares / days)) :
  5 * days_new * (hectares / (3 * days)) = 25 / 3 :=
sorry

end mowers_mow_l112_112782


namespace distribution_methods_l112_112171

open Finset

def volunteers : Finset (Fin 5) := univ

def groupings : Finset (Finset (Finset (Fin 5))) :=
  (Finset.powerset volunteers).filter (λ s, s.card = 2)

def count_groupings : ℕ :=
  groupings.card * (groupings.erase univ).card * 1 / (2 * 6)

def venues : Finset (Fin 3) := univ

def permutations : Finset (Finset (Fin 3)) :=
  powersetLen 3 venues

def count_permutations : ℕ := permutations.card

theorem distribution_methods :
  count_groupings * count_permutations = 90 :=
  by
    sorry

end distribution_methods_l112_112171


namespace pine_tree_next_one_in_between_l112_112683

theorem pine_tree_next_one_in_between (n : ℕ) (p s : ℕ) (trees : n = 2019) (pines : p = 1009) (spruces : s = 1010)
    (equal_intervals : true) : 
    ∃ (i : ℕ), (i < n) ∧ ((i + 1) % n ∈ {j | j < p}) ∧ ((i + 3) % n ∈ {j | j < p}) :=
  sorry

end pine_tree_next_one_in_between_l112_112683


namespace dan_present_age_l112_112664

theorem dan_present_age : ∃ x : ℕ, (x + 18 = 8 * (x - 3)) ∧ x = 6 :=
by
  -- We skip the proof steps
  sorry

end dan_present_age_l112_112664


namespace milk_leftover_l112_112435

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end milk_leftover_l112_112435


namespace probability_prime_sum_two_dice_l112_112892

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def num_ways_sum_is (target_sum : ℕ) : ℕ :=
  finset.card { (a, b) : finset.univ × finset.univ | a + b = target_sum }

def total_outcomes : ℕ := 36

theorem probability_prime_sum_two_dice :
  (∑ n in (finset.range 13).filter is_prime, num_ways_sum_is n) / total_outcomes = 5 / 12 :=
sorry

end probability_prime_sum_two_dice_l112_112892


namespace sphere_weight_dependence_l112_112775

theorem sphere_weight_dependence 
  (r1 r2 SA1 SA2 weight1 weight2 : ℝ) 
  (h1 : r1 = 0.15) 
  (h2 : r2 = 2 * r1) 
  (h3 : SA1 = 4 * Real.pi * r1^2) 
  (h4 : SA2 = 4 * Real.pi * r2^2) 
  (h5 : weight1 = 8) 
  (h6 : weight1 / SA1 = weight2 / SA2) : 
  weight2 = 32 :=
by
  sorry

end sphere_weight_dependence_l112_112775
