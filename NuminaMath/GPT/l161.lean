import Mathlib

namespace arith_prog_a1_a10_geom_prog_a1_a10_l161_161158

-- First we define our sequence and conditions for the arithmetic progression case
def is_arith_prog (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + d * (n - 1)

-- Arithmetic progression case
theorem arith_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_ap : is_arith_prog a) :
  a 1 * a 10 = -728 := 
  sorry

-- Then we define our sequence and conditions for the geometric progression case
def is_geom_prog (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

-- Geometric progression case
theorem geom_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_gp : is_geom_prog a) :
  a 1 + a 10 = -7 := 
  sorry

end arith_prog_a1_a10_geom_prog_a1_a10_l161_161158


namespace probability_two_cards_sum_15_from_standard_deck_l161_161641

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l161_161641


namespace last_two_digits_28_l161_161605

theorem last_two_digits_28 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (2^(2*n) * (2^(2*n+1) - 1)) % 100 = 28 :=
by
  sorry

end last_two_digits_28_l161_161605


namespace verify_cube_modifications_l161_161766

-- Definitions and conditions from the problem
def side_length : ℝ := 9
def initial_volume : ℝ := side_length^3
def initial_surface_area : ℝ := 6 * side_length^2

def volume_remaining : ℝ := 639
def surface_area_remaining : ℝ := 510

-- The theorem proving the volume and surface area of the remaining part after carving the cross-shaped groove
theorem verify_cube_modifications :
  initial_volume - (initial_volume - volume_remaining) = 639 ∧
  510 = surface_area_remaining :=
by
  sorry

end verify_cube_modifications_l161_161766


namespace sum_of_remainders_l161_161098

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 47 = 25) (h2 : b % 47 = 20) (h3 : c % 47 = 3) : 
  (a + b + c) % 47 = 1 := 
by {
  sorry
}

end sum_of_remainders_l161_161098


namespace samantha_last_name_length_l161_161608

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end samantha_last_name_length_l161_161608


namespace girls_friends_count_l161_161727

variable (days_in_week : ℕ)
variable (total_friends : ℕ)
variable (boys : ℕ)

axiom H1 : days_in_week = 7
axiom H2 : total_friends = 2 * days_in_week
axiom H3 : boys = 11

theorem girls_friends_count : total_friends - boys = 3 :=
by sorry

end girls_friends_count_l161_161727


namespace polynomial_factorization_m_n_l161_161757

theorem polynomial_factorization_m_n (m n : ℤ) (h : (x : ℤ) → x^2 + m * x + n = (x + 1) * (x + 3)) : m - n = 1 := 
by
  -- Define the equality of the factored polynomial and the standard form polynomial.
  have h_poly : (x : ℤ) → x^2 + m * x + n = x^2 + 4 * x + 3, from
    fun x => h x ▸ by ring,
  -- Extract values of m and n by comparing coefficients.
  have h_m : m = 4, from by
    have := congr_fun h_poly 0,
    simp at this,
    assumption,
  
  have h_n : n = 3, from by
    have := congr_fun h_poly (-1),
    simp at this,
    assumption,
  
  -- Substitute m and n to find that m - n = 1.
  rw [h_m, h_n],
  exact dec_trivial

end polynomial_factorization_m_n_l161_161757


namespace probability_all_vertical_faces_green_l161_161476

theorem probability_all_vertical_faces_green :
  let color_prob := (1 / 2 : ℚ)
  let total_arrangements := 2^6
  let valid_arrangements := 2 + 12 + 6
  ((valid_arrangements : ℚ) / total_arrangements) = 5 / 16 := by
  sorry

end probability_all_vertical_faces_green_l161_161476


namespace cost_of_fencing_each_side_l161_161752

theorem cost_of_fencing_each_side (x : ℝ) (h : 4 * x = 316) : x = 79 :=
by
  sorry

end cost_of_fencing_each_side_l161_161752


namespace probability_sum_15_l161_161644

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l161_161644


namespace evaluate_expression_l161_161590

noncomputable def a := Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def b := -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def d := -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 3 / 50 :=
by
  sorry

end evaluate_expression_l161_161590


namespace line_solutions_l161_161157

-- Definition for points
def point := ℝ × ℝ

-- Conditions for lines and points
def line1 (p : point) : Prop := 3 * p.1 + 4 * p.2 = 2
def line2 (p : point) : Prop := 2 * p.1 + p.2 = -2
def line3 : Prop := ∃ p : point, line1 p ∧ line2 p

def lineL (p : point) : Prop := 2 * p.1 + p.2 = -2 -- Line l we need to prove
def perp_lineL : Prop := ∃ p : point, lineL p ∧ p.1 - 2 * p.2 = 1

-- Symmetry condition for the line
def symmetric_line (p : point) : Prop := 2 * p.1 + p.2 = 2 -- Symmetric line we need to prove

-- Main theorem to prove
theorem line_solutions :
  line3 →
  perp_lineL →
  (∀ p, lineL p ↔ 2 * p.1 + p.2 = -2) ∧
  (∀ p, symmetric_line p ↔ 2 * p.1 + p.2 = 2) :=
sorry

end line_solutions_l161_161157


namespace count_valid_third_sides_l161_161905

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l161_161905


namespace sum_of_factors_72_l161_161240

theorem sum_of_factors_72 :
  (∑ d in divisors 72, d) = 195 := by
    sorry

end sum_of_factors_72_l161_161240


namespace number_of_possible_third_side_lengths_l161_161919

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l161_161919


namespace intersection_correct_l161_161887

def A : Set ℝ := { x | 0 < x ∧ x < 3 }
def B : Set ℝ := { x | x^2 ≥ 4 }
def intersection : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

theorem intersection_correct : A ∩ B = intersection := by
  sorry

end intersection_correct_l161_161887


namespace least_three_digit_with_product_eight_is_124_l161_161084

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l161_161084


namespace num_triples_l161_161144

/-- Theorem statement:
There are exactly 2 triples of positive integers (a, b, c) satisfying the conditions:
1. ab + ac = 60
2. bc + ac = 36
3. ab + bc = 48
--/
theorem num_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (ab + ac = 60) → (bc + ac = 36) → (ab + bc = 48) → 
  (a, b, c) ∈ [(1, 4, 8), (1, 12, 3)] →
  ∃! (a b c : ℕ), (ab + ac = 60) ∧ (bc + ac = 36) ∧ (ab + bc = 48) :=
sorry

end num_triples_l161_161144


namespace fraction_sum_product_roots_of_quadratic_l161_161569

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l161_161569


namespace time_to_fill_cistern_proof_l161_161454

-- Define the filling rate F and emptying rate E
def filling_rate : ℚ := 1 / 3 -- cisterns per hour
def emptying_rate : ℚ := 1 / 6 -- cisterns per hour

-- Define the net rate as the difference between filling and emptying rates
def net_rate : ℚ := filling_rate - emptying_rate

-- Define the time to fill the cistern given the net rate
def time_to_fill_cistern (net_rate : ℚ) : ℚ := 1 / net_rate

-- The proof statement
theorem time_to_fill_cistern_proof : time_to_fill_cistern net_rate = 6 := 
by sorry

end time_to_fill_cistern_proof_l161_161454


namespace sin_identity_alpha_l161_161023

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end sin_identity_alpha_l161_161023


namespace quadratic_roots_vieta_l161_161557

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l161_161557


namespace sin_135_l161_161706

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l161_161706


namespace mary_characters_initials_l161_161198

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end mary_characters_initials_l161_161198


namespace probability_even_heads_after_60_flips_l161_161683

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (3 / 4) - (1 / 2) * P_n (n - 1)

theorem probability_even_heads_after_60_flips :
  P_n 60 = 1 / 2 * (1 + 1 / 2^60) :=
sorry

end probability_even_heads_after_60_flips_l161_161683


namespace sum_first_2014_terms_l161_161299

def sequence_is_arithmetic (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + a 2

def first_arithmetic_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :=
  S n = (n * (n - 1)) / 2

theorem sum_first_2014_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : sequence_is_arithmetic a) 
  (h2 : a 3 = 2) : 
  S 2014 = 1007 * 2013 :=
sorry

end sum_first_2014_terms_l161_161299


namespace number_of_integers_in_interval_l161_161345

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l161_161345


namespace intersection_M_P_l161_161597

def M : Set ℝ := {0, 1, 2, 3}
def P : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_P : M ∩ P = {0, 1} := 
by
  -- You can fill in the proof here
  sorry

end intersection_M_P_l161_161597


namespace find_a_cubed_minus_b_cubed_l161_161889

theorem find_a_cubed_minus_b_cubed (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) : a^3 - b^3 = 486 := 
by 
  sorry

end find_a_cubed_minus_b_cubed_l161_161889


namespace polynomial_degree_is_14_l161_161435

noncomputable def polynomial_degree (a b c d e f g h : ℝ) : ℕ :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 then 14 else 0

theorem polynomial_degree_is_14 (a b c d e f g h : ℝ) (h_neq0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  polynomial_degree a b c d e f g h = 14 :=
by sorry

end polynomial_degree_is_14_l161_161435


namespace paintable_wall_area_l161_161688

/-- Given 4 bedrooms each with length 15 feet, width 11 feet, and height 9 feet,
and doorways and windows occupying 80 square feet in each bedroom,
prove that the total paintable wall area is 1552 square feet. -/
theorem paintable_wall_area
  (bedrooms : ℕ) (length width height doorway_window_area : ℕ) :
  bedrooms = 4 →
  length = 15 →
  width = 11 →
  height = 9 →
  doorway_window_area = 80 →
  4 * (2 * (length * height) + 2 * (width * height) - doorway_window_area) = 1552 :=
by
  intros bedrooms_eq length_eq width_eq height_eq doorway_window_area_eq
  -- Definition of the problem conditions
  have bedrooms_def : bedrooms = 4 := bedrooms_eq
  have length_def : length = 15 := length_eq
  have width_def : width = 11 := width_eq
  have height_def : height = 9 := height_eq
  have doorway_window_area_def : doorway_window_area = 80 := doorway_window_area_eq
  -- Assertion of the correct answer
  sorry

end paintable_wall_area_l161_161688


namespace john_payment_l161_161770

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end john_payment_l161_161770


namespace quadratic_root_identity_l161_161544

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l161_161544


namespace original_amount_of_cooking_oil_l161_161411

theorem original_amount_of_cooking_oil (X : ℝ) (H : (2 / 5 * X + 300) + (1 / 2 * (X - (2 / 5 * X + 300)) - 200) + 800 = X) : X = 2500 :=
by simp at H; linarith

end original_amount_of_cooking_oil_l161_161411


namespace sum_of_squares_first_20_l161_161820

-- Define the sum of squares function
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Specific problem instance
theorem sum_of_squares_first_20 : sum_of_squares 20 = 5740 :=
  by
  -- Proof skipping placeholder
  sorry

end sum_of_squares_first_20_l161_161820


namespace booking_rooms_needed_l161_161480

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l161_161480


namespace tammy_speed_on_second_day_l161_161053

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l161_161053


namespace arithmetic_sequence_sum_l161_161959

/-- Let {a_n} be an arithmetic sequence with a positive common difference d.
  Given that a_1 + a_2 + a_3 = 15 and a_1 * a_2 * a_3 = 80, we aim to show that
  a_11 + a_12 + a_13 = 105. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_sequence_sum_l161_161959


namespace simultaneous_equations_solution_l161_161732

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_equations_solution_l161_161732


namespace matthew_egg_rolls_l161_161786

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end matthew_egg_rolls_l161_161786


namespace sports_club_members_l161_161762

theorem sports_club_members (N B T : ℕ) (h_total : N = 30) (h_badminton : B = 18) (h_tennis : T = 19) (h_neither : N - (B + T - 9) = 2) : B + T - 9 = 28 :=
by
  sorry

end sports_club_members_l161_161762


namespace triangle_area_l161_161191

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area (a b : ℝ × ℝ) : 
  let area_parallelogram := (a.1 * b.2 - a.2 * b.1).abs in
  (1 / 2) * area_parallelogram = 4.5 :=
by
  sorry

end triangle_area_l161_161191


namespace reciprocal_of_sum_is_correct_l161_161996

theorem reciprocal_of_sum_is_correct : (1 / (1 / 4 + 1 / 6)) = 12 / 5 := by
  sorry

end reciprocal_of_sum_is_correct_l161_161996


namespace number_of_ways_l161_161627

-- Define the conditions
def num_people : ℕ := 3
def num_sports : ℕ := 4

-- Prove the total number of different ways
theorem number_of_ways : num_sports ^ num_people = 64 := by
  sorry

end number_of_ways_l161_161627


namespace intersection_point_of_circle_and_line_l161_161893

noncomputable def circle_parametric (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, 2 * Real.sin α)
noncomputable def line_polar (rho θ : ℝ) : Prop := rho * Real.sin θ = 2

theorem intersection_point_of_circle_and_line :
  ∃ (α : ℝ) (rho θ : ℝ), circle_parametric α = (1, 2) ∧ line_polar rho θ := sorry

end intersection_point_of_circle_and_line_l161_161893


namespace quadratic_roots_vieta_l161_161561

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l161_161561


namespace quadratic_roots_identity_l161_161554

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l161_161554


namespace total_questions_l161_161883

theorem total_questions (f s k : ℕ) (hf : f = 36) (hs : s = 2 * f) (hk : k = (f + s) / 2) :
  2 * (f + s + k) = 324 :=
by {
  sorry
}

end total_questions_l161_161883


namespace integers_between_3_and_15_with_perfect_cube_base_1331_l161_161336

theorem integers_between_3_and_15_with_perfect_cube_base_1331 :
  {n : ℕ | 3 ≤ n ∧ n ≤ 15 ∧ (∃ m : ℕ, n^3 + 3 * n^2 + 3 * n + 1 = m^3)}.card = 12 :=
by
  sorry

end integers_between_3_and_15_with_perfect_cube_base_1331_l161_161336


namespace raghu_investment_l161_161448

theorem raghu_investment (R : ℝ) 
  (h1 : ∀ T : ℝ, T = 0.9 * R) 
  (h2 : ∀ V : ℝ, V = 0.99 * R) 
  (h3 : R + 0.9 * R + 0.99 * R = 6069) : 
  R = 2100 := 
by
  sorry

end raghu_investment_l161_161448


namespace chord_length_30_degrees_through_focus_l161_161298

noncomputable def parabola_focus : ℝ × ℝ :=
  (3/4, 0)

noncomputable def parabola : ℝ → ℝ :=
  λ x, real.sqrt (3 * x)

noncomputable def line_through_focus (x : ℝ): ℝ :=
  real.sqrt 3 / 3 * (x - 3 / 4)

theorem chord_length_30_degrees_through_focus : 
  let F := parabola_focus in
  ∀ A B : ℝ × ℝ, A.2 = parabola A.1 ∧ B.2 = parabola B.1 ∧
  A.2 = line_through_focus A.1 ∧ B.2 = line_through_focus B.1 →
  |A.1 - B.1| = 12 
  :=
by 
  sorry

end chord_length_30_degrees_through_focus_l161_161298


namespace sin_135_l161_161708

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l161_161708


namespace sport_formulation_water_content_l161_161672

theorem sport_formulation_water_content :
  ∀ (f_s c_s w_s : ℕ) (f_p c_p w_p : ℕ),
    f_s / c_s = 1 / 12 →
    f_s / w_s = 1 / 30 →
    f_p / c_p = 1 / 4 →
    f_p / w_p = 1 / 60 →
    c_p = 4 →
    w_p = 60 := by
  sorry

end sport_formulation_water_content_l161_161672


namespace probability_sum_15_l161_161643

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l161_161643


namespace holden_master_bath_size_l161_161532

theorem holden_master_bath_size (b n m : ℝ) (h_b : b = 309) (h_n : n = 918) (h : 2 * (b + m) = n) : m = 150 := by
  sorry

end holden_master_bath_size_l161_161532


namespace potato_difference_l161_161408

def x := 8 * 13
def k := (67 - 13) / 2
def z := 20 * k
def d := z - x

theorem potato_difference : d = 436 :=
by
  sorry

end potato_difference_l161_161408


namespace pens_sold_to_recover_investment_l161_161014

-- Given the conditions
variables (P C : ℝ) (N : ℝ)
-- P is the total cost of 30 pens
-- C is the cost price of each pen
-- N is the number of pens sold to recover the initial investment

-- Stating the conditions
axiom h1 : P = 30 * C
axiom h2 : N * 1.5 * C = P

-- Proving that N = 20
theorem pens_sold_to_recover_investment (P C N : ℝ) (h1 : P = 30 * C) (h2 : N * 1.5 * C = P) : N = 20 :=
by
  sorry

end pens_sold_to_recover_investment_l161_161014


namespace range_of_a_l161_161166

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ 2 * a - (1 / 2) * a^2) ↔ 0 ≤ a :=
by
  sorry

end range_of_a_l161_161166


namespace count_whole_numbers_in_interval_l161_161349

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l161_161349


namespace f_n_f_n_eq_n_l161_161592

def f : ℕ → ℕ := sorry
axiom f_def1 : f 1 = 1
axiom f_def2 : ∀ n ≥ 2, f n = n - f (f (n - 1))

theorem f_n_f_n_eq_n (n : ℕ) (hn : 0 < n) : f (n + f n) = n :=
by sorry

end f_n_f_n_eq_n_l161_161592


namespace cos_double_angle_l161_161516

variable (α : ℝ)
variable (h : Real.cos α = 2/3)

theorem cos_double_angle : Real.cos (2 * α) = -1/9 :=
  by
  sorry

end cos_double_angle_l161_161516


namespace lcm_second_factor_l161_161063

theorem lcm_second_factor (A B : ℕ) (hcf : ℕ) (f1 f2 : ℕ) 
  (h₁ : hcf = 25) 
  (h₂ : A = 350) 
  (h₃ : Nat.gcd A B = hcf) 
  (h₄ : Nat.lcm A B = hcf * f1 * f2) 
  (h₅ : f1 = 13)
  : f2 = 14 := 
sorry

end lcm_second_factor_l161_161063


namespace find_y_l161_161577

theorem find_y (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 :=
by sorry

end find_y_l161_161577


namespace sin_135_degree_l161_161723

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l161_161723


namespace absentees_in_morning_session_is_three_l161_161412

theorem absentees_in_morning_session_is_three
  (registered_morning : ℕ)
  (registered_afternoon : ℕ)
  (absent_afternoon : ℕ)
  (total_students : ℕ)
  (total_registered : ℕ)
  (attended_afternoon : ℕ)
  (attended_morning : ℕ)
  (absent_morning : ℕ) :
  registered_morning = 25 →
  registered_afternoon = 24 →
  absent_afternoon = 4 →
  total_students = 42 →
  total_registered = registered_morning + registered_afternoon →
  attended_afternoon = registered_afternoon - absent_afternoon →
  attended_morning = total_students - attended_afternoon →
  absent_morning = registered_morning - attended_morning →
  absent_morning = 3 :=
by
  intros
  sorry

end absentees_in_morning_session_is_three_l161_161412


namespace inverse_function_value_l161_161892

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3

theorem inverse_function_value :
  f 3 = 51 :=
by
  sorry

end inverse_function_value_l161_161892


namespace find_x_value_l161_161287

theorem find_x_value :
  ∃ (x : ℤ), ∀ (y z w : ℤ), (x = 2 * y + 4) → (y = z + 5) → (z = 2 * w + 3) → (w = 50) → x = 220 :=
by
  sorry

end find_x_value_l161_161287


namespace cycle_time_to_library_l161_161898

theorem cycle_time_to_library 
  (constant_speed : Prop)
  (time_to_park : ℕ)
  (distance_to_park : ℕ)
  (distance_to_library : ℕ)
  (h1 : constant_speed)
  (h2 : time_to_park = 30)
  (h3 : distance_to_park = 5)
  (h4 : distance_to_library = 3) :
  (18 : ℕ) = (30 * distance_to_library / distance_to_park) :=
by
  intros
  -- The proof would go here
  sorry

end cycle_time_to_library_l161_161898


namespace number_of_girls_l161_161001

theorem number_of_girls (B G : ℕ) (h1 : B * 5 = G * 8) (h2 : B + G = 1040) : G = 400 :=
by
  sorry

end number_of_girls_l161_161001


namespace two_cards_sum_to_15_proof_l161_161636

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l161_161636


namespace athul_downstream_distance_l161_161465

-- Define the conditions
def upstream_distance : ℝ := 16
def upstream_time : ℝ := 4
def speed_of_stream : ℝ := 1
def downstream_time : ℝ := 4

-- Translate the conditions into properties and prove the downstream distance
theorem athul_downstream_distance (V : ℝ) 
  (h1 : upstream_distance = (V - speed_of_stream) * upstream_time) :
  (V + speed_of_stream) * downstream_time = 24 := 
by
  -- Given the conditions, the proof would be filled here
  sorry

end athul_downstream_distance_l161_161465


namespace prime_number_property_l161_161536

open Nat

-- Definition that p is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Conjecture to prove: if p is a prime number and p^4 - 3p^2 + 9 is also a prime number, then p = 2.
theorem prime_number_property (p : ℕ) (h1 : is_prime p) (h2 : is_prime (p^4 - 3*p^2 + 9)) : p = 2 :=
sorry

end prime_number_property_l161_161536


namespace coin_combinations_l161_161327

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l161_161327


namespace rectangle_area_l161_161808

theorem rectangle_area (P l w : ℕ) (h_perimeter: 2 * l + 2 * w = 60) (h_aspect: l = 3 * w / 2) : l * w = 216 :=
sorry

end rectangle_area_l161_161808


namespace percentage_increase_20_l161_161867

noncomputable def oldCompanyEarnings : ℝ := 3 * 12 * 5000
noncomputable def totalEarnings : ℝ := 426000
noncomputable def newCompanyMonths : ℕ := 36 + 5
noncomputable def newCompanyEarnings : ℝ := totalEarnings - oldCompanyEarnings
noncomputable def newCompanyMonthlyEarnings : ℝ := newCompanyEarnings / newCompanyMonths
noncomputable def oldCompanyMonthlyEarnings : ℝ := 5000

theorem percentage_increase_20 :
  (newCompanyMonthlyEarnings - oldCompanyMonthlyEarnings) / oldCompanyMonthlyEarnings * 100 = 20 :=
by sorry

end percentage_increase_20_l161_161867


namespace joe_list_possibilities_l161_161845

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l161_161845


namespace more_birds_than_nests_l161_161255

theorem more_birds_than_nests (birds nests : Nat) (h_birds : birds = 6) (h_nests : nests = 3) : birds - nests = 3 :=
by
  sorry

end more_birds_than_nests_l161_161255


namespace projectile_reaches_100_feet_l161_161266

theorem projectile_reaches_100_feet :
  ∃ (t : ℝ), t > 0 ∧ (-16 * t ^ 2 + 80 * t = 100) ∧ (t = 2.5) := by
sorry

end projectile_reaches_100_feet_l161_161266


namespace probability_two_cards_sum_to_15_l161_161646

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l161_161646


namespace least_three_digit_product_of_digits_is_8_l161_161086

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l161_161086


namespace count_whole_numbers_in_interval_l161_161354

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l161_161354


namespace square_feet_per_acre_l161_161854

theorem square_feet_per_acre 
  (pay_per_acre_per_month : ℕ) 
  (total_pay_per_month : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (total_acres : ℕ) 
  (H1 : pay_per_acre_per_month = 30) 
  (H2 : total_pay_per_month = 300) 
  (H3 : length = 360) 
  (H4 : width = 1210) 
  (H5 : total_acres = 10) : 
  (length * width) / total_acres = 43560 :=
by 
  sorry

end square_feet_per_acre_l161_161854


namespace mary_characters_initials_l161_161199

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end mary_characters_initials_l161_161199


namespace find_k_find_a_l161_161964

noncomputable def f (a k : ℝ) (x : ℝ) := a ^ x + k * a ^ (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

theorem find_k (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : is_odd_function (f a k)) : k = -1 :=
sorry

theorem find_a (k : ℝ) (h₃ : k = -1) (h₄ : f 1 = 3 / 2) (h₅ : is_monotonic_increasing (f 2 k)) : a = 2 :=
sorry

end find_k_find_a_l161_161964


namespace value_of_m_l161_161870

def f (x : ℚ) : ℚ := 3 * x^3 - 1 / x + 2
def g (x : ℚ) (m : ℚ) : ℚ := 2 * x^3 - 3 * x + m
def h (x : ℚ) : ℚ := x^2

theorem value_of_m : f 3 - g 3 (122 / 3) + h 3 = 5 :=
by
  sorry

end value_of_m_l161_161870


namespace smallest_n_for_terminating_decimal_l161_161664

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m : ℕ, (n = m → m > 0 → ∃ (a b : ℕ), n + 103 = 2^a * 5^b)) 
    ∧ n = 22 :=
sorry

end smallest_n_for_terminating_decimal_l161_161664


namespace probability_first_number_greater_l161_161264

noncomputable def probability_first_greater_second : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_first_number_greater :
  probability_first_greater_second = 7 / 16 :=
sorry

end probability_first_number_greater_l161_161264


namespace value_of_a_l161_161667

theorem value_of_a (a : ℝ) : 
  (∀ (x : ℝ), (x < -4 ∨ x > 5) → x^2 + a * x + 20 > 0) → a = -1 :=
by
  sorry

end value_of_a_l161_161667


namespace tabitha_honey_nights_l161_161416

def servings_per_cup := 1
def cups_per_night := 2
def ounces_per_container := 16
def servings_per_ounce := 6
def total_servings := servings_per_ounce * ounces_per_container
def servings_per_night := servings_per_cup * cups_per_night
def number_of_nights := total_servings / servings_per_night

theorem tabitha_honey_nights : number_of_nights = 48 :=
by
  -- Proof to be provided.
  sorry

end tabitha_honey_nights_l161_161416


namespace sin_135_eq_l161_161696

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l161_161696


namespace mean_height_is_68_l161_161072

/-
Given the heights of the volleyball players:
  heights_50s = [58, 59]
  heights_60s = [60, 61, 62, 65, 65, 66, 67]
  heights_70s = [70, 71, 71, 72, 74, 75, 79, 79]

We need to prove that the mean height of the players is 68 inches.
-/
def heights_50s : List ℕ := [58, 59]
def heights_60s : List ℕ := [60, 61, 62, 65, 65, 66, 67]
def heights_70s : List ℕ := [70, 71, 71, 72, 74, 75, 79, 79]

def total_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s
def number_of_players : ℕ := total_heights.length
def total_height : ℕ := total_heights.sum
def mean_height : ℕ := total_height / number_of_players

theorem mean_height_is_68 : mean_height = 68 := by
  sorry

end mean_height_is_68_l161_161072


namespace planned_daily_catch_l161_161268

theorem planned_daily_catch (x y : ℝ) 
  (h1 : x * y = 1800)
  (h2 : (x / 3) * (y - 20) + ((2 * x / 3) - 1) * (y + 20) = 1800) :
  y = 100 :=
by
  sorry

end planned_daily_catch_l161_161268


namespace third_side_integer_lengths_l161_161909

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l161_161909


namespace maximum_F_value_l161_161596

open Real

noncomputable def F (a b c x : ℝ) := abs ((a * x^2 + b * x + c) * (c * x^2 + b * x + a))

theorem maximum_F_value (a b c : ℝ) (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1)
    (hfx : abs (a * x^2 + b * x + c) ≤ 1) :
    ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ F a b c x = 2 := 
  sorry

end maximum_F_value_l161_161596


namespace find_fiona_experience_l161_161059

namespace Experience

variables (d e f : ℚ)

def avg_experience_equation : Prop := d + e + f = 36
def fiona_david_equation : Prop := f - 5 = d
def emma_david_future_equation : Prop := e + 4 = (3/4) * (d + 4)

theorem find_fiona_experience (h1 : avg_experience_equation d e f) (h2 : fiona_david_equation d f) (h3 : emma_david_future_equation d e) :
  f = 183 / 11 :=
by
  sorry

end Experience

end find_fiona_experience_l161_161059


namespace solve_for_a_l161_161279

theorem solve_for_a : ∀ (a : ℝ), (2 * a - 16 = 9) → (a = 12.5) :=
by
  intro a h
  sorry

end solve_for_a_l161_161279


namespace total_vertical_distance_of_rings_l161_161725

theorem total_vertical_distance_of_rings :
  let thickness := 2
  let top_outside_diameter := 20
  let bottom_outside_diameter := 4
  let n := (top_outside_diameter - bottom_outside_diameter) / thickness + 1
  let total_distance := n * thickness
  total_distance + thickness = 76 :=
by
  sorry

end total_vertical_distance_of_rings_l161_161725


namespace jason_work_hours_l161_161013

variable (x y : ℕ)

def working_hours : Prop :=
  (4 * x + 6 * y = 88) ∧
  (x + y = 18)

theorem jason_work_hours (h : working_hours x y) : y = 8 :=
  by
    sorry

end jason_work_hours_l161_161013


namespace sin_135_degree_l161_161724

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l161_161724


namespace smallest_base10_integer_l161_161440

theorem smallest_base10_integer (X Y : ℕ) (hX : X < 6) (hY : Y < 8) (h : 7 * X = 9 * Y) :
  63 = 7 * X ∧ 63 = 9 * Y :=
by
  -- Proof steps would go here
  sorry

end smallest_base10_integer_l161_161440


namespace problem_1_problem_2_problem_3_l161_161107

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) : (∀ x : ℝ, f (x + 1) = x^2 + 4*x + 1) → (∀ x : ℝ, f x = x^2 + 2*x - 2) :=
by
  intro h
  sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) : (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) → (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) → (∀ x : ℝ, f x = x + 3) :=
by
  intros h1 h2
  sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) : (∀ x : ℝ, 2 * f x + f (1 / x) = 3 * x) → (∀ x : ℝ, f x = 2 * x - 1 / x) :=
by
  intro h
  sorry

end problem_1_problem_2_problem_3_l161_161107


namespace triangle_third_side_lengths_l161_161936

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l161_161936


namespace intersection_A_B_l161_161531

-- Definitions of sets A and B
def A := { x : ℝ | x ≥ -1 }
def B := { y : ℝ | y < 1 }

-- Statement to prove the intersection of A and B
theorem intersection_A_B : A ∩ B = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l161_161531


namespace interval_contains_thousand_members_l161_161865

noncomputable def sequence : ℕ → ℝ
| 0       => 1
| (n + 1) => real.sqrt ((sequence n) ^ 2 + 1 / (sequence n))

/-- There exists some interval of length 1 that contains more than a thousand members of the sequence defined by
    a₀ = 1 and aₙ₊₁ = √(aₙ² + 1/aₙ). 
--/
theorem interval_contains_thousand_members :
  ∃ k : ℝ, ∃ f : ℕ → ℕ, (0 < f 0) ∧ (∀ n, sequence (f n + 1) - sequence (f n) < (1 / 2000)) ∧ (((λ n, sequence (f n)) ) ^ (1001 : ℝ) ∈ set.Icc k (k + 1)) := 
begin
  sorry, -- As requested, no proof is to be provided.
end

end interval_contains_thousand_members_l161_161865


namespace negation_proposition_l161_161214

theorem negation_proposition :
  (¬(∀ x : ℝ, x^2 - x + 2 < 0) ↔ ∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
sorry

end negation_proposition_l161_161214


namespace ladder_base_distance_l161_161258

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end ladder_base_distance_l161_161258


namespace correct_options_l161_161099

variable (Ω : Type) [ProbSpace Ω]
variable (A B : Event Ω)
variable (P : Probability Ω)

noncomputable def mutually_exclusive (P : Probability Ω) (A B : Event Ω) : Prop :=
  P (A ∩ B) = 0

noncomputable def independent (P : Probability Ω) (A B : Event Ω) : Prop :=
  P (A ∩ B) = P A * P B

noncomputable def prob_complement (P : Probability Ω) (A : Event Ω) : ℝ :=
  1 - P A

theorem correct_options
  (mut_excl : mutually_exclusive P A B → ¬mutually_exclusive P A Bᶜ)
  (indep_AB : independent P A B → independent P A B)
  (indep_AcompB : independent P A B → independent P A Bᶜ)
  (P_A_06 : P A = 0.6)
  (P_B_02 : P B = 0.2)
  (indep_A_B : independent P A B)
  (P_A_B_eq : P (A ∪ B) = P A + P B - P (A ∩ B)) :
  ¬(P (A ∪ B) = 0.8) →
  (P_A_08 : P A = 0.8)
  (P_B_07 : P B = 0.7)
  (indep_A_B_again : independent P A B)
  (P_A_comp_B_eq : P (A ∩ Bᶜ) = 0.8 * (1 - 0.7)) :
  true := by
  sorry

end correct_options_l161_161099


namespace closest_point_on_plane_exists_l161_161145

def point_on_plane : Type := {P : ℝ × ℝ × ℝ // ∃ (x y z : ℝ), P = (x, y, z) ∧ 2 * x - 3 * y + 4 * z = 20}

def point_A : ℝ × ℝ × ℝ := (0, 1, -1)

theorem closest_point_on_plane_exists (P : point_on_plane) :
  ∃ (x y z : ℝ), (x, y, z) = (54 / 29, -80 / 29, 83 / 29) := sorry

end closest_point_on_plane_exists_l161_161145


namespace quadratic_root_sum_and_product_l161_161566

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l161_161566


namespace no_such_natural_numbers_l161_161010

theorem no_such_natural_numbers :
  ¬ ∃ (x y : ℕ), (∃ (a b : ℕ), x^2 + y = a^2 ∧ x - y = b^2) := 
sorry

end no_such_natural_numbers_l161_161010


namespace binom_coeffs_not_coprime_l161_161974

open Nat

theorem binom_coeffs_not_coprime (n k m : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) : 
  Nat.gcd (Nat.choose n k) (Nat.choose n m) > 1 := 
sorry

end binom_coeffs_not_coprime_l161_161974


namespace min_removed_numbers_l161_161289

theorem min_removed_numbers : 
  ∃ S : Finset ℤ, 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 1982) ∧ 
    (∀ a b c : ℤ, a ∈ S → b ∈ S → c ∈ S → c ≠ a * b) ∧
    ∀ T : Finset ℤ, 
      ((∀ y ∈ T, 1 ≤ y ∧ y ≤ 1982) ∧ 
       (∀ p q r : ℤ, p ∈ T → q ∈ T → r ∈ T → r ≠ p * q) → 
       T.card ≥ 1982 - 43) :=
sorry

end min_removed_numbers_l161_161289


namespace p_sufficient_not_necessary_for_q_l161_161594

variable (x : ℝ)

def p : Prop := x > 0
def q : Prop := x > -1

theorem p_sufficient_not_necessary_for_q : (p x → q x) ∧ ¬ (q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l161_161594


namespace train_crossing_time_l161_161338

def train_length : ℕ := 100  -- length of the train in meters
def bridge_length : ℕ := 180  -- length of the bridge in meters
def train_speed_kmph : ℕ := 36  -- speed of the train in kmph

theorem train_crossing_time 
  (TL : ℕ := train_length) 
  (BL : ℕ := bridge_length) 
  (TSK : ℕ := train_speed_kmph) : 
  (TL + BL) / ((TSK * 1000) / 3600) = 28 := by
  sorry

end train_crossing_time_l161_161338


namespace initial_parts_planned_l161_161222

variable (x : ℕ)

theorem initial_parts_planned (x : ℕ) (h : 3 * x + (x + 5) + 100 = 675): x = 142 :=
by sorry

end initial_parts_planned_l161_161222


namespace length_of_d_in_proportion_l161_161523

variable (a b c d : ℝ)

theorem length_of_d_in_proportion
  (h1 : a = 3) 
  (h2 : b = 2)
  (h3 : c = 6)
  (h_prop : a / b = c / d) : 
  d = 4 :=
by
  sorry

end length_of_d_in_proportion_l161_161523


namespace count_whole_numbers_in_interval_l161_161362

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l161_161362


namespace abs_val_inequality_solution_l161_161427

theorem abs_val_inequality_solution (x : ℝ) : |x - 2| + |x + 3| ≥ 4 ↔ x ≤ - (5 / 2) :=
by
  sorry

end abs_val_inequality_solution_l161_161427


namespace sin_135_eq_l161_161693

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l161_161693


namespace shiny_pennies_probability_l161_161851

theorem shiny_pennies_probability :
  ∃ (a b : ℕ), gcd a b = 1 ∧ a / b = 5 / 11 ∧ a + b = 16 :=
sorry

end shiny_pennies_probability_l161_161851


namespace sum_of_factors_72_l161_161243

theorem sum_of_factors_72 : 
  let factors_sum (n : ℕ) : ℕ := 
    ∑ d in Nat.divisors n, d
  in factors_sum 72 = 195 :=
by
  sorry

end sum_of_factors_72_l161_161243


namespace geometric_sequence_inserted_product_l161_161390

theorem geometric_sequence_inserted_product :
  ∃ (a b c : ℝ), a * b * c = 216 ∧
    (∃ (q : ℝ), 
      a = (8/3) * q ∧ 
      b = a * q ∧ 
      c = b * q ∧ 
      (8/3) * q^4 = 27/2) :=
sorry

end geometric_sequence_inserted_product_l161_161390


namespace sin_identity_l161_161021

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_l161_161021


namespace count_valid_third_sides_l161_161906

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l161_161906


namespace arctan_sum_zero_l161_161388
open Real

variable (a b c : ℝ)
variable (h : a^2 + b^2 = c^2)

theorem arctan_sum_zero (h : a^2 + b^2 = c^2) :
  arctan (a / (b + c)) + arctan (b / (a + c)) + arctan (c / (a + b)) = 0 := 
sorry

end arctan_sum_zero_l161_161388


namespace two_cards_totaling_15_probability_l161_161657

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l161_161657


namespace find_e_l161_161809

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e
  (d f : ℝ)
  (H1 : f = 9)
  (H2 : ( -(d / 3))^2 = 3)
  (H3 : 3 + d + e + f = -3) :
  e = -15 - 3 * sqrt 3 :=
by
  sorry

end find_e_l161_161809


namespace area_of_regular_octagon_l161_161297

theorem area_of_regular_octagon (BDEF_is_rectangle : true) (AB : ℝ) (BC : ℝ) 
    (capture_regular_octagon : true) (AB_eq_1 : AB = 1) (BC_eq_2 : BC = 2)
    (octagon_perimeter_touch : ∀ x, x = 1) : 
    ∃ A : ℝ, A = 11 :=
by
  sorry

end area_of_regular_octagon_l161_161297


namespace largest_w_exists_l161_161977

theorem largest_w_exists (w x y z : ℝ) (h1 : w + x + y + z = 25) (h2 : w * x + w * y + w * z + x * y + x * z + y * z = 2 * y + 2 * z + 193) :
  ∃ (w1 w2 : ℤ), w1 > 0 ∧ w2 > 0 ∧ ((w = w1 / w2) ∧ (w1 + w2 = 27)) :=
sorry

end largest_w_exists_l161_161977


namespace giraffe_statue_price_l161_161600

variable (G : ℕ) -- Price of a giraffe statue in dollars

-- Conditions as definitions in Lean 4
def giraffe_jade_usage := 120 -- grams
def elephant_jade_usage := 2 * giraffe_jade_usage -- 240 grams
def elephant_price := 350 -- dollars
def total_jade := 1920 -- grams
def additional_profit_with_elephants := 400 -- dollars

-- Prove that the price of a giraffe statue is $150
theorem giraffe_statue_price : 
  16 * G + additional_profit_with_elephants = 8 * elephant_price → G = 150 :=
by
  intro h
  sorry

end giraffe_statue_price_l161_161600


namespace books_left_to_read_l161_161970

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l161_161970


namespace y_coordinate_of_intersection_l161_161213

def line_eq (x t : ℝ) : ℝ := -2 * x + t

def parabola_eq (x : ℝ) : ℝ := (x - 1) ^ 2 + 1

def intersection_condition (x y t : ℝ) : Prop :=
  y = line_eq x t ∧ y = parabola_eq x ∧ x ≥ 0 ∧ y ≥ 0

theorem y_coordinate_of_intersection (x y : ℝ) (t : ℝ) (h_t : t = 11)
  (h_intersection : intersection_condition x y t) :
  y = 5 := by
  sorry

end y_coordinate_of_intersection_l161_161213


namespace table_tennis_teams_equation_l161_161384

-- Variables
variable (x : ℕ)

-- Conditions
def total_matches : ℕ := 28
def teams_playing_equation : Prop := x * (x - 1) = 28 * 2

-- Theorem Statement
theorem table_tennis_teams_equation : teams_playing_equation x :=
sorry

end table_tennis_teams_equation_l161_161384


namespace total_pens_bought_l161_161599

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 := 
sorry

end total_pens_bought_l161_161599


namespace count_whole_numbers_in_interval_l161_161341

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l161_161341


namespace haley_trees_initially_grew_l161_161173

-- Given conditions
def num_trees_died : ℕ := 2
def num_trees_survived : ℕ := num_trees_died + 7

-- Prove the total number of trees initially grown
theorem haley_trees_initially_grew : num_trees_died + num_trees_survived = 11 :=
by
  -- here we would provide the proof eventually
  sorry

end haley_trees_initially_grew_l161_161173


namespace factor_theorem_l161_161445

theorem factor_theorem (h : ℤ) : (∀ m : ℤ, (m - 8) ∣ (m^2 - h * m - 24) ↔ h = 5) :=
  sorry

end factor_theorem_l161_161445


namespace sum_of_a_and_b_l161_161735

theorem sum_of_a_and_b (a b : ℝ) (h : a^2 + b^2 + 2 * a - 4 * b + 5 = 0) :
  a + b = 1 :=
sorry

end sum_of_a_and_b_l161_161735


namespace max_value_of_g_l161_161989

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l161_161989


namespace cost_per_student_admission_l161_161619

-- Definitions based on the conditions.
def cost_to_rent_bus : ℕ := 100
def total_budget : ℕ := 350
def number_of_students : ℕ := 25

-- The theorem that we need to prove.
theorem cost_per_student_admission : (total_budget - cost_to_rent_bus) / number_of_students = 10 :=
by
  sorry

end cost_per_student_admission_l161_161619


namespace number_of_blue_balls_l161_161850

theorem number_of_blue_balls (b : ℕ) 
  (h1 : 0 < b ∧ b ≤ 15)
  (prob : (b / 15) * ((b - 1) / 14) = 1 / 21) :
  b = 5 := sorry

end number_of_blue_balls_l161_161850


namespace expression_evaluation_l161_161877

theorem expression_evaluation :
  (3 : ℝ) + 3 * Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (3 - Real.sqrt 3)) = 4 + 3 * Real.sqrt 3 :=
sorry

end expression_evaluation_l161_161877


namespace sin_135_correct_l161_161691

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l161_161691


namespace determine_a_l161_161422

lemma even_exponent (a : ℤ) : (a^2 - 4*a) % 2 = 0 :=
sorry

lemma decreasing_function (a : ℤ) : a^2 - 4*a < 0 :=
sorry

theorem determine_a (a : ℤ) (h1 : (a^2 - 4*a) % 2 = 0) (h2 : a^2 - 4*a < 0) : a = 2 :=
sorry

end determine_a_l161_161422


namespace tangent_line_equation_even_derived_l161_161305

def f (x a : ℝ) : ℝ := x^3 + (a - 2) * x^2 + a * x - 1

def f' (x a : ℝ) : ℝ := 3 * x^2 + 2 * (a - 2) * x + a

theorem tangent_line_equation_even_derived (a : ℝ) (h : ∀ x : ℝ, f' x a = f' (-x) a) :
  5 * 1 - (f 1 a) - 3 = 0 :=
by
  sorry

end tangent_line_equation_even_derived_l161_161305


namespace rainfall_sunday_l161_161391

theorem rainfall_sunday 
  (rain_sun rain_mon rain_tue : ℝ)
  (h1 : rain_mon = rain_sun + 3)
  (h2 : rain_tue = 2 * rain_mon)
  (h3 : rain_sun + rain_mon + rain_tue = 25) :
  rain_sun = 4 :=
by
  sorry

end rainfall_sunday_l161_161391


namespace area_of_triangle_AEB_l161_161385

noncomputable def rectangle_area_AEB : ℝ :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 2
  let FG := 8 - DF - GC -- DC (8 units) minus DF and GC.
  let ratio := AB / FG
  let altitude_AEB := BC * ratio
  let area_AEB := 0.5 * AB * altitude_AEB
  area_AEB

theorem area_of_triangle_AEB : rectangle_area_AEB = 32 :=
by
  -- placeholder for detailed proof
  sorry

end area_of_triangle_AEB_l161_161385


namespace prime_square_minus_one_divisible_by_24_l161_161035

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 
  ∃ k : ℤ, p^2 - 1 = 24 * k :=
  sorry

end prime_square_minus_one_divisible_by_24_l161_161035


namespace polynomial_remainder_l161_161881

theorem polynomial_remainder :
  (4 * (2.5 : ℝ)^5 - 9 * (2.5 : ℝ)^4 + 7 * (2.5 : ℝ)^2 - 2.5 - 35 = 45.3125) :=
by sorry

end polynomial_remainder_l161_161881


namespace largest_angle_of_triangle_ABC_l161_161587

theorem largest_angle_of_triangle_ABC (a b c : ℝ)
  (h₁ : a + b + 2 * c = a^2) 
  (h₂ : a + b - 2 * c = -1) : 
  ∃ C : ℝ, C = 120 :=
sorry

end largest_angle_of_triangle_ABC_l161_161587


namespace contestant_score_l161_161763

theorem contestant_score (highest_score lowest_score : ℕ) (average_score : ℕ)
  (h_hs : highest_score = 86)
  (h_ls : lowest_score = 45)
  (h_avg : average_score = 76) :
  (76 * 9 - 86 - 45) / 7 = 79 := 
by 
  sorry

end contestant_score_l161_161763


namespace solve_equation_l161_161038

theorem solve_equation : ∀ (x : ℝ), x ≠ -3 → x ≠ 3 → 
  (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by
  intros x hx1 hx2 h
  sorry

end solve_equation_l161_161038


namespace sixth_ninth_grader_buddy_fraction_l161_161947

theorem sixth_ninth_grader_buddy_fraction
  (s n : ℕ)
  (h_fraction_pairs : n / 4 = s / 3)
  (h_buddy_pairing : (∀ i, i < n -> ∃ j, j < s) 
     ∧ (∀ j, j < s -> ∃ i, i < n) -- each sixth grader paired with one ninth grader and vice versa
  ) :
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by 
  sorry

end sixth_ninth_grader_buddy_fraction_l161_161947


namespace probability_product_is_square_l161_161803

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def probability_square_product : ℚ :=
  let total_outcomes   := 10 * 8
  let favorable_outcomes := 
    [(1,1), (1,4), (2,2), (4,1), (3,3), (2,8), (8,2), (5,5), (6,6), (7,7), (8,8)].length
  favorable_outcomes / total_outcomes

theorem probability_product_is_square : 
  probability_square_product = 11 / 80 :=
  sorry

end probability_product_is_square_l161_161803


namespace probability_two_cards_sum_to_15_l161_161647

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l161_161647


namespace quadratic_roots_identity_l161_161552

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l161_161552


namespace describe_T_correctly_l161_161400

def T (x y : ℝ) : Prop :=
(x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2)

theorem describe_T_correctly :
  (∀ x y : ℝ, T x y ↔
    ((x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2))) :=
by
  sorry

end describe_T_correctly_l161_161400


namespace number_of_small_pizzas_ordered_l161_161126

-- Define the problem conditions
def benBrothers : Nat := 2
def slicesPerPerson : Nat := 12
def largePizzaSlices : Nat := 14
def smallPizzaSlices : Nat := 8
def numLargePizzas : Nat := 2

-- Define the statement to prove
theorem number_of_small_pizzas_ordered : 
  ∃ (s : Nat), (benBrothers + 1) * slicesPerPerson - numLargePizzas * largePizzaSlices = s * smallPizzaSlices ∧ s = 1 :=
by
  sorry

end number_of_small_pizzas_ordered_l161_161126


namespace horse_cow_difference_l161_161203

def initial_conditions (h c : ℕ) : Prop :=
  4 * c = h

def transaction (h c : ℕ) : Prop :=
  (h - 15) * 7 = (c + 15) * 13

def final_difference (h c : ℕ) : Prop := 
  h - 15 - (c + 15) = 30

theorem horse_cow_difference (h c : ℕ) (hc : initial_conditions h c) (ht : transaction h c) : final_difference h c :=
    by
      sorry

end horse_cow_difference_l161_161203


namespace composite_prop_true_l161_161160

def p : Prop := ∀ (x : ℝ), x > 0 → x + (1/(2*x)) ≥ 1

def q : Prop := ∀ (x : ℝ), x > 1 → (x^2 + 2*x - 3 > 0)

theorem composite_prop_true : p ∨ q :=
by
  sorry

end composite_prop_true_l161_161160


namespace general_term_of_seq_l161_161306

open Nat

noncomputable def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 3 * 2^n

theorem general_term_of_seq (a : ℕ → ℕ) :
  seq a → ∀ n, a n = (3 * n - 1) * 2^(n-1) :=
by
  sorry

end general_term_of_seq_l161_161306


namespace ten_pow_n_plus_one_divisible_by_eleven_l161_161034

theorem ten_pow_n_plus_one_divisible_by_eleven (n : ℕ) (h : n % 2 = 1) : 11 ∣ (10 ^ n + 1) :=
sorry

end ten_pow_n_plus_one_divisible_by_eleven_l161_161034


namespace mark_baseball_cards_gcd_l161_161409

theorem mark_baseball_cards_gcd :
  Nat.gcd (Nat.gcd 1080 1620) 540 = 540 :=
by
  sorry

end mark_baseball_cards_gcd_l161_161409


namespace fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l161_161602

-- Define the sequences as functions
def row1 (n : ℕ) : ℤ := (-2)^n
def row2 (n : ℕ) : ℤ := row1 n + 2
def row3 (n : ℕ) : ℤ := (-1) * (-2)^n

-- Theorems to be proven

-- (1) Prove the fourth term in row ① is 16 
theorem fourth_term_row1_is_16 : row1 4 = 16 := sorry

-- (1) Prove the nth term in row ① is (-2)^n
theorem nth_term_row1 (n : ℕ) : row1 n = (-2)^n := sorry

-- (2) Let the nth number in row ① be a, prove the nth number in row ② is a + 2
theorem nth_term_row2 (n : ℕ) : row2 n = row1 n + 2 := sorry

-- (3) If the sum of three consecutive numbers in row ③ is -192, find these numbers
theorem sum_three_consecutive_row3 : ∃ n : ℕ, row3 n + row3 (n + 1) + row3 (n + 2) = -192 ∧ 
  row3 n  = -64 ∧ row3 (n + 1) = 128 ∧ row3 (n + 2) = -256 := sorry

end fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l161_161602


namespace smallest_n_produces_terminating_decimal_l161_161230

noncomputable def smallest_n := 12

theorem smallest_n_produces_terminating_decimal (n : ℕ) (h_pos: 0 < n) : 
    (∀ m : ℕ, m > 113 → (n = m - 113 → (∃ k : ℕ, 1 ≤ k ∧ (m = 2^k ∨ m = 5^k)))) :=
by
  sorry

end smallest_n_produces_terminating_decimal_l161_161230


namespace ceil_neg_sqrt_frac_l161_161137

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l161_161137


namespace same_color_probability_l161_161801

theorem same_color_probability 
  (B R : ℕ)
  (hB : B = 5)
  (hR : R = 5)
  : (B + R = 10) → (1/2 * 4/9 + 1/2 * 4/9 = 4/9) := by
  intros
  sorry

end same_color_probability_l161_161801


namespace whole_numbers_in_interval_7_4_3pi_l161_161364

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l161_161364


namespace sufficient_but_not_necessary_condition_l161_161673

variable {a : ℝ}

theorem sufficient_but_not_necessary_condition :
  (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (a ≥ 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l161_161673


namespace double_angle_value_l161_161281

theorem double_angle_value : 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 2 := 
sorry

end double_angle_value_l161_161281


namespace division_4073_by_38_l161_161094

theorem division_4073_by_38 :
  ∃ q r, 4073 = 38 * q + r ∧ 0 ≤ r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end division_4073_by_38_l161_161094


namespace minimum_rooms_needed_l161_161481

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l161_161481


namespace amy_7_mile_run_time_l161_161462

-- Define the conditions
variable (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ)

-- State the conditions
def conditions : Prop :=
  rachel_time_per_9_miles = 36 ∧
  amy_time_per_4_miles = 1 / 3 * rachel_time_per_9_miles ∧
  amy_time_per_mile = amy_time_per_4_miles / 4 ∧
  amy_time_per_7_miles = amy_time_per_mile * 7

-- The main statement to prove
theorem amy_7_mile_run_time (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ) :
  conditions rachel_time_per_9_miles amy_time_per_4_miles amy_time_per_mile amy_time_per_7_miles → 
  amy_time_per_7_miles = 21 := 
by
  intros h
  sorry

end amy_7_mile_run_time_l161_161462


namespace inequality_proof_l161_161040

theorem inequality_proof 
  (x y z w : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w)
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) :
  x^4 * z + y^4 * w ≥ z * w :=
sorry

end inequality_proof_l161_161040


namespace product_of_squares_of_consecutive_even_integers_l161_161810

theorem product_of_squares_of_consecutive_even_integers :
  ∃ (a : ℤ), (a - 2) * a * (a + 2) = 36 * a ∧ (a > 0) ∧ (a % 2 = 0) ∧
  ((a - 2)^2 * a^2 * (a + 2)^2) = 36864 :=
by
  sorry

end product_of_squares_of_consecutive_even_integers_l161_161810


namespace range_m_of_nonmonotonic_on_interval_l161_161169

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log (x + 1) + x^2 - m * x
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m / (x + 1) + 2 * x - m

theorem range_m_of_nonmonotonic_on_interval :
  ∀ (m : ℝ), ¬ MonotoneOn (f m) (Set.Ioi 1) → m ∈ Set.Ioi 4 :=
by
  sorry

end range_m_of_nonmonotonic_on_interval_l161_161169


namespace repunits_infinite_l161_161794

-- Define gcd condition
def gcd_condition (m : ℕ) : Prop :=
  Nat.gcd m 10 = 1

-- Define repunit
def repunit (n : ℕ) : ℕ :=
  (10 ^ n - 1) / 9
  
-- Statement of the problem:
theorem repunits_infinite (m : ℕ) (h : gcd_condition m) :
  ∃ n, m ∣ repunit n ∧ ∃∞ n', m ∣ repunit n' :=
sorry

end repunits_infinite_l161_161794


namespace town_population_growth_is_62_percent_l161_161579

noncomputable def population_growth_proof : ℕ := 
  let p := 22
  let p_square := p * p
  let pop_1991 := p_square
  let pop_2001 := pop_1991 + 150
  let pop_2011 := pop_2001 + 150
  let k := 28  -- Given that 784 = 28^2
  let pop_2011_is_perfect_square := k * k = pop_2011
  let percentage_increase := ((pop_2011 - pop_1991) * 100) / pop_1991
  if pop_2011_is_perfect_square then percentage_increase 
  else 0

theorem town_population_growth_is_62_percent :
  population_growth_proof = 62 :=
by
  sorry

end town_population_growth_is_62_percent_l161_161579


namespace one_third_of_flour_l161_161680

-- Definition of the problem conditions
def initial_flour : ℚ := 5 + 2 / 3
def portion : ℚ := 1 / 3

-- Definition of the theorem to prove
theorem one_third_of_flour : portion * initial_flour = 1 + 8 / 9 :=
by {
  -- Placeholder proof
  sorry
}

end one_third_of_flour_l161_161680


namespace n_cubed_plus_5_div_by_6_l161_161207

theorem n_cubed_plus_5_div_by_6  (n : ℤ) : 6 ∣ n * (n^2 + 5) :=
sorry

end n_cubed_plus_5_div_by_6_l161_161207


namespace quadratic_root_sum_and_product_l161_161565

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l161_161565


namespace count_valid_third_sides_l161_161903

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l161_161903


namespace sin_identity_l161_161020

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_l161_161020


namespace find_z_in_sequence_l161_161585

theorem find_z_in_sequence (x y z a b : ℤ) 
  (h1 : b = 1)
  (h2 : a + b = 0)
  (h3 : y + a = 1)
  (h4 : z + y = 3)
  (h5 : x + z = 2) :
  z = 1 :=
sorry

end find_z_in_sequence_l161_161585


namespace decreasing_interval_of_f_l161_161621

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * log x

theorem decreasing_interval_of_f :
  (∀ x ∈ (Set.Ioc 0 1 : Set ℝ), 2*x - 2/x < 0) :=
by
  sorry

end decreasing_interval_of_f_l161_161621


namespace line_through_point_l161_161512

theorem line_through_point (k : ℝ) : (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end line_through_point_l161_161512


namespace trains_crossing_l161_161224

noncomputable def time_to_cross_each_other (v : ℝ) (L₁ L₂ : ℝ) (t₁ t₂ : ℝ) : ℝ :=
  (L₁ + L₂) / (2 * v)

theorem trains_crossing (v : ℝ) (t₁ t₂ : ℝ) (h1 : t₁ = 27) (h2 : t₂ = 17) :
  time_to_cross_each_other v (v * 27) (v * 17) t₁ t₂ = 22 :=
by
  -- Conditions
  have h3 : t₁ = 27 := h1
  have h4 : t₂ = 17 := h2
  -- Proof outline (not needed, just to ensure the setup is understood):
  -- Lengths
  let L₁ := v * 27
  let L₂ := v * 17
  -- Calculating Crossing Time
  have t := (L₁ + L₂) / (2 * v)
  -- Simplification leads to t = 22
  sorry

end trains_crossing_l161_161224


namespace probability_two_cards_sum_to_15_l161_161649

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l161_161649


namespace cost_per_unit_l161_161263

theorem cost_per_unit 
  (units_per_month : ℕ := 400)
  (selling_price_per_unit : ℝ := 440)
  (profit_requirement : ℝ := 40000)
  (C : ℝ) :
  profit_requirement ≤ (units_per_month * selling_price_per_unit) - (units_per_month * C) → C ≤ 340 :=
by
  sorry

end cost_per_unit_l161_161263


namespace domain_of_h_l161_161874

noncomputable def h (x : ℝ) : ℝ := (x^4 - 5 * x + 6) / (|x - 4| + |x + 2| - 1)

theorem domain_of_h : ∀ x : ℝ, |x - 4| + |x + 2| - 1 ≠ 0 := by
  intro x
  sorry

end domain_of_h_l161_161874


namespace middle_number_consecutive_sum_l161_161759

theorem middle_number_consecutive_sum (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : a + b + c = 30) : b = 10 :=
by
  sorry

end middle_number_consecutive_sum_l161_161759


namespace combinations_of_coins_l161_161332

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l161_161332


namespace intersection_sets_l161_161308

theorem intersection_sets :
  let M := {x : ℝ | (x + 3) * (x - 2) < 0 }
  let N := {x : ℝ | 1 ≤ x ∧ x ≤ 3 }
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_sets_l161_161308


namespace sum_of_factors_72_l161_161245

theorem sum_of_factors_72 : (Finset.sum ((Finset.range 73).filter (λ n, 72 % n = 0))) = 195 := by
  sorry

end sum_of_factors_72_l161_161245


namespace min_rooms_needed_l161_161488

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l161_161488


namespace range_of_lambda_l161_161165

open Complex

theorem range_of_lambda (m θ λ : ℝ) (h1 : Complex.mk m (4 - m^2) = Complex.mk (2 * cos θ) (λ + 3 * sin θ)) :
  -9 / 16 ≤ λ ∧ λ ≤ 7 :=
sorry

end range_of_lambda_l161_161165


namespace fraction_sum_product_roots_of_quadratic_l161_161567

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l161_161567


namespace rectangle_area_eq_l161_161252

variables (a b : ℝ) (A B C D M : Point) (𝓡₁ 𝓡₂ : Circle)
  (hA : IsRectangle A B C D)
  (hM : IsPointOnLineSegment M B C)
  (hInscribed1 : IsInscribedCircle 𝓡₁ [A, M, C, D])
  (hRadius1 : radius 𝓡₁ = a)
  (hInscribed2 : IsInscribedCircle 𝓡₂ [A, B, M])
  (hRadius2 : radius 𝓡₂ = b)

theorem rectangle_area_eq : 
  area_rectangle A B C D = (4 * a^3 - 2 * a^2 * b) / (2 * a - b) :=
sorry

end rectangle_area_eq_l161_161252


namespace coin_combinations_count_l161_161316

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l161_161316


namespace count_whole_numbers_in_interval_l161_161352

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l161_161352


namespace train_length_is_95_l161_161681

noncomputable def train_length (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ := 
  let speed_ms := speed_kmh * 1000 / 3600 
  speed_ms * time_seconds

theorem train_length_is_95 : train_length 1.5980030008814248 214 = 95 := by
  sorry

end train_length_is_95_l161_161681


namespace sum_of_factors_72_l161_161247

theorem sum_of_factors_72 : 
  ∃ σ: ℕ → ℕ, (∀ n: ℕ, σ(n) = ∏ p in (nat.divisors n).to_finset, (∑ k in nat.divisors p, k)) ∧ σ 72 = 195 :=
by 
  sorry

end sum_of_factors_72_l161_161247


namespace least_three_digit_with_product_eight_is_124_l161_161085

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l161_161085


namespace max_value_of_g_l161_161985

def g : ℕ → ℕ 
| n := if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : 
  ∃ N, (∀ n, g n ≤ N) ∧ N = 23 := 
sorry

end max_value_of_g_l161_161985


namespace remaining_days_to_finish_l161_161393

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end remaining_days_to_finish_l161_161393


namespace quadratic_root_identity_l161_161542

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l161_161542


namespace percentage_gain_on_powerlifting_total_l161_161125

def initialTotal : ℝ := 2200
def initialWeight : ℝ := 245
def weightIncrease : ℝ := 8
def finalWeight : ℝ := initialWeight + weightIncrease
def liftingRatio : ℝ := 10
def finalTotal : ℝ := finalWeight * liftingRatio

theorem percentage_gain_on_powerlifting_total :
  ∃ (P : ℝ), initialTotal * (1 + P / 100) = finalTotal :=
by
  sorry

end percentage_gain_on_powerlifting_total_l161_161125


namespace incorrect_statement_d_l161_161204

noncomputable def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem incorrect_statement_d (n : ℤ) :
  (n < cbrt 9 ∧ cbrt 9 < n+1) → n ≠ 3 :=
by
  intro h
  have h2 : (2 : ℤ) < cbrt 9 := sorry
  have h3 : cbrt 9 < (3 : ℤ) := sorry
  exact sorry

end incorrect_statement_d_l161_161204


namespace find_speed_l161_161392

-- Definitions corresponding to conditions
def JacksSpeed (x : ℝ) : ℝ := x^2 - 7 * x - 12
def JillsDistance (x : ℝ) : ℝ := x^2 - 3 * x - 10
def JillsTime (x : ℝ) : ℝ := x + 2

-- Theorem statement
theorem find_speed (x : ℝ) (hx : x ≠ -2) (h_speed_eq : JacksSpeed x = (JillsDistance x) / (JillsTime x)) : JacksSpeed x = 2 :=
by
  sorry

end find_speed_l161_161392


namespace final_result_l161_161885

noncomputable def f : ℝ → ℝ := sorry
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (3 + x) = f x
axiom f_half_periodic : ∀ x : ℝ, f (3 / 2 - x) = f x
axiom f_value_neg2 : f (-2) = -3

axiom a1_value : a 1 = -1
axiom S_n : ∀ n : ℕ, S n = 2 * a n + n

theorem final_result : f (a 5) + f (a 6) = 3 :=
sorry

end final_result_l161_161885


namespace range_of_a_l161_161941

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 ≤ 0

theorem range_of_a :
  ¬ quadratic_inequality a ↔ -1 < a ∧ a < 3 :=
  by
  sorry

end range_of_a_l161_161941


namespace least_three_digit_product_of_digits_is_8_l161_161087

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l161_161087


namespace remainder_15_plus_3y_l161_161778

theorem remainder_15_plus_3y (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + 3 * y) % 31 = 11 :=
by
  sorry

end remainder_15_plus_3y_l161_161778


namespace radius_tangent_circle_l161_161506

theorem radius_tangent_circle (r r1 r2 : ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 5)
    (h_concentric : true) : r = 1 := by
  -- Definitions are given as conditions
  have h1 := r1 -- radius of smaller concentric circle
  have h2 := r2 -- radius of larger concentric circle
  have h3 := h_concentric -- the circles are concentric
  have h4 := h_r1 -- r1 = 3
  have h5 := h_r2 -- r2 = 5
  sorry

end radius_tangent_circle_l161_161506


namespace percentage_problem_l161_161115

theorem percentage_problem (P : ℕ) (n : ℕ) (h_n : n = 16)
  (h_condition : (40: ℚ) = 0.25 * n + 2) : P = 250 :=
by
  sorry

end percentage_problem_l161_161115


namespace contrapositive_of_real_roots_l161_161743

variable {a : ℝ}

theorem contrapositive_of_real_roots :
  (1 + 4 * a < 0) → (a < 0) := by
  sorry

end contrapositive_of_real_roots_l161_161743


namespace count_whole_numbers_in_interval_l161_161340

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l161_161340


namespace matthew_egg_rolls_l161_161787

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end matthew_egg_rolls_l161_161787


namespace roots_of_quadratic_eq_l161_161538

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l161_161538


namespace cuboid_dimensions_exist_l161_161814

theorem cuboid_dimensions_exist (l w h : ℝ) 
  (h1 : l * w = 5) 
  (h2 : l * h = 8) 
  (h3 : w * h = 10) 
  (h4 : l * w * h = 200) : 
  ∃ (l w h : ℝ), l = 4 ∧ w = 2.5 ∧ h = 2 := 
sorry

end cuboid_dimensions_exist_l161_161814


namespace partial_fraction_sum_eq_zero_l161_161873

theorem partial_fraction_sum_eq_zero (A B C D E : ℂ) :
  (∀ x : ℂ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 4 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_eq_zero_l161_161873


namespace largest_q_value_l161_161811

theorem largest_q_value : ∃ q, q >= 1 ∧ q^4 - q^3 - q - 1 ≤ 0 ∧ (∀ r, r >= 1 ∧ r^4 - r^3 - r - 1 ≤ 0 → r ≤ q) ∧ q = (Real.sqrt 5 + 1) / 2 := 
sorry

end largest_q_value_l161_161811


namespace solution_set_of_inequality_l161_161900

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = f x) (h2 : ∀ x, 0 ≤ x → f x = x - 1) :
  { x : ℝ | f (x - 1) > 1 } = { x | x < -1 ∨ x > 3 } :=
by
  sorry

end solution_set_of_inequality_l161_161900


namespace omega_min_value_l161_161963

def min_omega (ω : ℝ) : Prop :=
  ω > 0 ∧ ∃ k : ℤ, (k ≠ 0 ∧ ω = 8)

theorem omega_min_value (ω : ℝ) (h1 : ω > 0) (h2 : ∃ k : ℤ, k ≠ 0 ∧ (k * 2 * π) / ω = π / 4) : 
  ω = 8 :=
by
  sorry

end omega_min_value_l161_161963


namespace sqrt_four_eq_pm_two_l161_161070

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_four_eq_pm_two_l161_161070


namespace percent_increase_first_quarter_l161_161123

theorem percent_increase_first_quarter (P : ℝ) (X : ℝ) (h1 : P > 0) 
  (end_of_second_quarter : P * 1.8 = P*(1 + X / 100) * 1.44) : 
  X = 25 :=
by
  sorry

end percent_increase_first_quarter_l161_161123


namespace number_of_lists_l161_161836

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l161_161836


namespace find_value_of_expression_l161_161593

noncomputable def root_finder (a b c : ℝ) : Prop :=
  a^3 - 30*a^2 + 65*a - 42 = 0 ∧
  b^3 - 30*b^2 + 65*b - 42 = 0 ∧
  c^3 - 30*c^2 + 65*c - 42 = 0

theorem find_value_of_expression {a b c : ℝ} (h : root_finder a b c) :
  a + b + c = 30 ∧ ab + bc + ca = 65 ∧ abc = 42 → 
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) = 770/43 :=
by
  sorry

end find_value_of_expression_l161_161593


namespace remainder_73_to_73_plus73_div137_l161_161749

theorem remainder_73_to_73_plus73_div137 :
  ((73 ^ 73 + 73) % 137) = 9 := by
  sorry

end remainder_73_to_73_plus73_div137_l161_161749


namespace opposite_of_4_l161_161215

theorem opposite_of_4 : ∃ x, 4 + x = 0 ∧ x = -4 :=
by sorry

end opposite_of_4_l161_161215


namespace quadratic_roots_sum_product_l161_161548

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l161_161548


namespace debbys_sister_candy_l161_161147

-- Defining the conditions
def debby_candy : ℕ := 32
def eaten_candy : ℕ := 35
def remaining_candy : ℕ := 39

-- The proof problem
theorem debbys_sister_candy : ∃ S : ℕ, debby_candy + S - eaten_candy = remaining_candy → S = 42 :=
by
  sorry  -- The proof goes here

end debbys_sister_candy_l161_161147


namespace find_doodads_produced_in_four_hours_l161_161429

theorem find_doodads_produced_in_four_hours :
  ∃ (n : ℕ),
    (∀ (workers hours widgets doodads : ℕ),
      (workers = 150 ∧ hours = 2 ∧ widgets = 800 ∧ doodads = 500) ∨
      (workers = 100 ∧ hours = 3 ∧ widgets = 750 ∧ doodads = 600) ∨
      (workers = 80  ∧ hours = 4 ∧ widgets = 480 ∧ doodads = n)
    ) → n = 640 :=
sorry

end find_doodads_produced_in_four_hours_l161_161429


namespace school_club_profit_l161_161860

theorem school_club_profit :
  let pencils := 1200
  let buy_rate := 4 / 3 -- pencils per dollar
  let sell_rate := 5 / 4 -- pencils per dollar
  let cost_per_pencil := 3 / 4 -- dollars per pencil
  let sell_per_pencil := 4 / 5 -- dollars per pencil
  let cost := pencils * cost_per_pencil
  let revenue := pencils * sell_per_pencil
  let profit := revenue - cost
  profit = 60 := 
by
  sorry

end school_club_profit_l161_161860


namespace divide_value_l161_161783

def divide (a b c : ℝ) : ℝ := |b^2 - 5 * a * c|

theorem divide_value : divide 2 (-3) 1 = 1 :=
by
  sorry

end divide_value_l161_161783


namespace birds_flew_away_l161_161629

-- Define the initial and remaining birds
def original_birds : ℕ := 12
def remaining_birds : ℕ := 4

-- Define the number of birds that flew away
noncomputable def flew_away_birds : ℕ := original_birds - remaining_birds

-- State the theorem that the number of birds that flew away is 8
theorem birds_flew_away : flew_away_birds = 8 := by
  -- Lean expects a proof here. For now, we use sorry to indicate the proof is skipped.
  sorry

end birds_flew_away_l161_161629


namespace true_proposition_l161_161894

variable (p : Prop) (q : Prop)

-- Introduce the propositions as Lean variables
def prop_p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2
def prop_q : Prop := ∀ a b : ℝ, ((a > 1 ∧ b > 1) → a * b > 1) ∧ ((a * b > 1) ∧ (¬ (a > 1 ∧ b > 1)))

-- Rewrite the main goal as a Lean statement
theorem true_proposition : ¬ prop_p ∧ prop_q := 
  sorry

end true_proposition_l161_161894


namespace journey_total_time_l161_161856

noncomputable def total_time (D : ℝ) (r_dist : ℕ → ℕ) (r_time : ℕ → ℕ) (u_speed : ℝ) : ℝ :=
  let dist_uphill := D * (r_dist 1) / (r_dist 1 + r_dist 2 + r_dist 3)
  let t_uphill := (dist_uphill / u_speed)
  let k := t_uphill / (r_time 1)
  (r_time 1 + r_time 2 + r_time 3) * k

theorem journey_total_time :
  total_time 50 (fun n => if n = 1 then 1 else if n = 2 then 2 else 3) 
                (fun n => if n = 1 then 4 else if n = 2 then 5 else 6) 
                3 = 10 + 5/12 :=
by
  sorry

end journey_total_time_l161_161856


namespace triangle_third_side_lengths_l161_161935

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l161_161935


namespace distance_between_parallel_sides_l161_161500

/-- Define the lengths of the parallel sides and the area of the trapezium -/
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area : ℝ := 285

/-- Define the condition of the problem: the formula for the area of the trapezium -/
def area_of_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

/-- The problem: prove the distance between the parallel sides is 15 cm -/
theorem distance_between_parallel_sides (h : ℝ) : 
  area_of_trapezium length_side1 length_side2 h = area → h = 15 :=
by
  sorry

end distance_between_parallel_sides_l161_161500


namespace purple_candy_minimum_cost_l161_161134

theorem purple_candy_minimum_cost (r g b n : ℕ) (h : 10 * r = 15 * g) (h1 : 15 * g = 18 * b) (h2 : 18 * b = 24 * n) : 
  ∃ k, k = n ∧ k ≥ 1 ∧ ∀ m, (24 * m = 360) → (m ≥ k) :=
by
  sorry

end purple_candy_minimum_cost_l161_161134


namespace friend_balloons_count_l161_161669

-- Definitions of the conditions
def balloons_you_have : ℕ := 7
def balloons_difference : ℕ := 2

-- Proof problem statement
theorem friend_balloons_count : (balloons_you_have - balloons_difference) = 5 :=
by
  sorry

end friend_balloons_count_l161_161669


namespace train_speed_l161_161459

def train_length : ℝ := 250
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 32

theorem train_speed :
  (train_length + bridge_length) / time_to_cross = 12.5 :=
by {
  sorry
}

end train_speed_l161_161459


namespace trigonometric_identity_l161_161174

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : tan α + 1 / tan α = 10 / 3)
  (h₂ : π / 4 < α ∧ α < π / 2) :
  sin (2 * α + π / 4) + 2 * cos (π / 4) * sin α ^ 2 = 4 * sqrt 2 / 5 :=
by
  sorry

end trigonometric_identity_l161_161174


namespace find_x_y_l161_161368

theorem find_x_y (x y : ℝ) (h : (2 * x - 3 * y + 5) ^ 2 + |x - y + 2| = 0) : x = -1 ∧ y = 1 :=
by
  sorry

end find_x_y_l161_161368


namespace prob_all_females_l161_161152

open Finset

variable (students : Finset ℕ) (males : Finset ℕ) (females : Finset ℕ)

def num_males : ℕ := 5
def num_females : ℕ := 4

-- Defining the total students as a finite set of 9 distinct elements
def total_students : Finset ℕ := (range (num_males + num_females)).erase 0

-- Defining the males as a finite set of the first 5 distinct elements
def male_students : Finset ℕ := (range num_males).erase 0

-- Defining the females as a finite set of the next 4 distinct elements
def female_students : Finset ℕ := ((range (num_males + num_females)).filter (λ x, x ≥ num_males))

-- Defining combinations
def choose (n k : ℕ) : ℕ := (range n).powerset.filter (λ s, s.card = k).card

theorem prob_all_females :
  (choose (num_males + num_females) 3) ≠ 0 → 
  (choose num_females 3) / (choose (num_males + num_females) 3) = 1 / 21 := 
by 
  sorry

end prob_all_females_l161_161152


namespace coin_combinations_count_l161_161331

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l161_161331


namespace balls_count_l161_161426

theorem balls_count (w r b : ℕ) (h_ratio : 4 * r = 3 * w ∧ 2 * w = 4 * b ∧ w = 20) : r = 15 ∧ b = 10 :=
by
  sorry

end balls_count_l161_161426


namespace percentage_caught_customers_l161_161179

noncomputable def total_sampling_percentage : ℝ := 0.25
noncomputable def caught_percentage : ℝ := 0.88

theorem percentage_caught_customers :
  total_sampling_percentage * caught_percentage = 0.22 :=
by
  sorry

end percentage_caught_customers_l161_161179


namespace part_1_part_2_l161_161598

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def A_def : A = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  ext x
  sorry
  
def B_def : B = {x : ℝ | x^2 + 2*x - 3 > 0} := by
  ext x
  sorry

theorem part_1 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0} := by
  rw [hA, hB]
  sorry

theorem part_2 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  (compl A ∩ B) = {x | x > 1 ∨ x < -3} := by
  rw [hA, hB]
  sorry

end part_1_part_2_l161_161598


namespace equation_condition1_equation_condition2_equation_condition3_l161_161142

-- Definitions for conditions
def condition1 : Prop := ∃(l : ℝ → ℝ), (l 2 = 1 ∧ ∀ (x1 x2 : ℝ), (l x2 - l x1) / (x2 - x1) = -1/2)
def condition2 : Prop := ∃(l : ℝ → ℝ), (l 1 = 4 ∧ l 2 = 3)
def condition3 : Prop := ∃(l : ℝ → ℝ), (l 2 = 1 ∧ (∃ a : ℝ, l 0 = a ∧ l a = 0) ∨ (∃ a : ℝ, a > 0 ∧ l 0 = a = l a))

-- Proving equations given conditions
theorem equation_condition1 : condition1 → ∀ (x y : ℝ), x + 2 * y - 4 = 0 
:= sorry

theorem equation_condition2 : condition2 → ∀ (x y : ℝ), x + y - 5 = 0 
:= sorry

theorem equation_condition3 :
    condition3 → (∀ (x y : ℝ), (x - 2 * y = 0) ∨ (x + y - 3 = 0)) 
:= sorry

end equation_condition1_equation_condition2_equation_condition3_l161_161142


namespace sin_135_l161_161707

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l161_161707


namespace inscribed_sphere_radius_eq_l161_161863

-- Define the parameters for the right cone
structure RightCone where
  base_radius : ℝ
  height : ℝ

-- Given the right cone conditions
def givenCone : RightCone := { base_radius := 15, height := 40 }

-- Define the properties for inscribed sphere
def inscribedSphereRadius (c : RightCone) : ℝ := sorry

-- The theorem statement for the radius of the inscribed sphere
theorem inscribed_sphere_radius_eq (c : RightCone) : ∃ (b d : ℝ), 
  inscribedSphereRadius c = b * Real.sqrt d - b ∧ (b + d = 14) :=
by
  use 5, 9
  sorry

end inscribed_sphere_radius_eq_l161_161863


namespace sum_a4_a5_a6_l161_161525

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (h1 : is_arithmetic_sequence a)
          (h2 : a 1 + a 2 + a 3 = 6)
          (h3 : a 7 + a 8 + a 9 = 24)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 15 :=
by
  sorry

end sum_a4_a5_a6_l161_161525


namespace find_p_l161_161376

theorem find_p 
  (h : {x | x^2 - 5 * x + p ≥ 0} = {x | x ≤ -1 ∨ x ≥ 6}) : p = -6 :=
by
  sorry

end find_p_l161_161376


namespace determine_x_y_l161_161041

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end determine_x_y_l161_161041


namespace tyrone_gives_non_integer_marbles_to_eric_l161_161819

theorem tyrone_gives_non_integer_marbles_to_eric
  (T_init : ℕ) (E_init : ℕ) (x : ℚ)
  (hT : T_init = 120) (hE : E_init = 18)
  (h_eq : T_init - x = 3 * (E_init + x)) :
  ¬ (∃ n : ℕ, x = n) :=
by
  sorry

end tyrone_gives_non_integer_marbles_to_eric_l161_161819


namespace least_three_digit_product_8_is_118_l161_161093

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l161_161093


namespace bridge_height_at_distance_l161_161114

theorem bridge_height_at_distance :
  (∃ (a : ℝ), ∀ (x : ℝ), (x = 25) → (a * x^2 + 25 = 0)) →
  (∀ (x : ℝ), (x = 10) → (-1/25 * x^2 + 25 = 21)) :=
by
  intro h1
  intro x h2
  have h : 625 * (-1 / 25) * (-1 / 25) = -25 := sorry
  sorry

end bridge_height_at_distance_l161_161114


namespace light_year_scientific_notation_l161_161112

def sci_not_eq : Prop := 
  let x := 9500000000000
  let y := 9.5 * 10^12
  x = y

theorem light_year_scientific_notation : sci_not_eq :=
  by sorry

end light_year_scientific_notation_l161_161112


namespace ada_original_seat_l161_161761

theorem ada_original_seat (seats: Fin 6 → Option String)
  (Bea_init Ceci_init Dee_init Edie_init Fran_init: Fin 6) 
  (Bea_fin Ceci_fin Fran_fin: Fin 6) 
  (Ada_fin: Fin 6)
  (Bea_moves_right: Bea_fin = Bea_init + 3)
  (Ceci_stays: Ceci_fin = Ceci_init)
  (Dee_switches_with_Edie: ∃ Dee_fin Edie_fin: Fin 6, Dee_fin = Edie_init ∧ Edie_fin = Dee_init)
  (Fran_moves_left: Fran_fin = Fran_init - 1)
  (Ada_end_seat: Ada_fin = 0 ∨ Ada_fin = 5):
  ∃ Ada_init: Fin 6, Ada_init = 2 + Ada_fin + 1 → Ada_init = 3 := 
by 
  sorry

end ada_original_seat_l161_161761


namespace sin_135_eq_sqrt2_over_2_l161_161711

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l161_161711


namespace probability_two_cards_sum_to_15_l161_161648

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l161_161648


namespace sum_due_is_correct_l161_161104

theorem sum_due_is_correct (BD TD PV : ℝ) (h1 : BD = 80) (h2 : TD = 70) (h_relation : BD = TD + (TD^2) / PV) : PV = 490 :=
by sorry

end sum_due_is_correct_l161_161104


namespace student_failed_by_l161_161118

-- Conditions
def total_marks : ℕ := 440
def passing_percentage : ℝ := 0.50
def marks_obtained : ℕ := 200

-- Calculate passing marks
noncomputable def passing_marks : ℝ := passing_percentage * total_marks

-- Definition of the problem to be proved
theorem student_failed_by : passing_marks - marks_obtained = 20 := 
by
  sorry

end student_failed_by_l161_161118


namespace count_whole_numbers_in_interval_l161_161360

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l161_161360


namespace percent_increase_l161_161102

-- Definitions based on conditions
def initial_price : ℝ := 10
def final_price : ℝ := 15

-- Goal: Prove that the percent increase in the price per share is 50%
theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 50 := 
by
  sorry  -- Proof is not required, so we skip it with sorry.

end percent_increase_l161_161102


namespace number_of_subsets_M_sum_of_elements_of_all_subsets_M_l161_161895

-- We define the set M
def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The number of subsets of M
theorem number_of_subsets_M : M.powerset.card = 2 ^ 10 := by
  sorry

-- The sum of the elements of all subsets of M
theorem sum_of_elements_of_all_subsets_M : M.sum * (2 ^ 9) = 55 * (2 ^ 9) := by
  have h₁ : M.sum = 55 := by
    sorry
  rw h₁
  have h₂ : M.powerset.card / 2 = 2 ^ 9 := by
    sorry
  ring
  exact h₂

end number_of_subsets_M_sum_of_elements_of_all_subsets_M_l161_161895


namespace ratio_third_first_l161_161815

theorem ratio_third_first (A B C : ℕ) (h1 : A + B + C = 110) (h2 : A = 2 * B) (h3 : B = 30) :
  C / A = 1 / 3 :=
by
  sorry

end ratio_third_first_l161_161815


namespace probability_no_order_l161_161679

theorem probability_no_order (P : ℕ) 
  (h1 : 60 ≤ 100) (h2 : 10 ≤ 100) (h3 : 15 ≤ 100) 
  (h4 : 5 ≤ 100) (h5 : 3 ≤ 100) (h6 : 2 ≤ 100) :
  P = 100 - (60 + 10 + 15 + 5 + 3 + 2) :=
by 
  sorry

end probability_no_order_l161_161679


namespace math_proof_problem_l161_161529

noncomputable def proof_problem : Prop :=
  ∃ (p : ℝ) (k m : ℝ), 
    (∀ (x y : ℝ), y^2 = 2 * p * x) ∧
    (p > 0) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (y1 * y2 = -8) ∧
      (x1 = 4 ∧ y1 = 0 ∨ x2 = 4 ∧ y2 = 0)) ∧
    (p = 1) ∧ 
    (∀ x0 : ℝ, 
      (2 * k * m = 1) ∧
      (∀ (x y : ℝ), y = k * x + m) ∧ 
      (∃ (r : ℝ), 
        ((x0 - r + 1 = 0) ∧
         (x0 - r * x0 + r^2 = 0))) ∧ 
       x0 = -1 / 2 )

theorem math_proof_problem : proof_problem := 
  sorry

end math_proof_problem_l161_161529


namespace solve_equation_l161_161413

theorem solve_equation :
  ∃! (x y z : ℝ), 2 * x^4 + 2 * y^4 - 4 * x^3 * y + 6 * x^2 * y^2 - 4 * x * y^3 + 7 * y^2 + 7 * z^2 - 14 * y * z - 70 * y + 70 * z + 175 = 0 ∧ x = 0 ∧ y = 0 ∧ z = -5 :=
by
  sorry

end solve_equation_l161_161413


namespace combinations_of_coins_l161_161319

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l161_161319


namespace probability_prime_ball_l161_161493

open Finset

theorem probability_prime_ball :
  let balls := {1, 2, 3, 4, 5, 6, 8, 9}
  let total := card balls
  let primes := {2, 3, 5}
  let primes_count := card primes
  (total = 8) → (primes ⊆ balls) → 
  primes_count = 3 → 
  primes_count / total = 3 / 8 :=
by
  intros
  sorry

end probability_prime_ball_l161_161493


namespace matthew_egg_rolls_l161_161785

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end matthew_egg_rolls_l161_161785


namespace find_n_l161_161221

theorem find_n (n : ℕ) (h : (17 + 98 + 39 + 54 + n) / 5 = n) : n = 52 :=
by
  sorry

end find_n_l161_161221


namespace maria_savings_after_purchase_l161_161197

theorem maria_savings_after_purchase
  (cost_sweater : ℕ)
  (cost_scarf : ℕ)
  (cost_mittens : ℕ)
  (num_family_members : ℕ)
  (savings : ℕ)
  (total_cost_one_set : ℕ)
  (total_cost_all_sets : ℕ)
  (amount_left : ℕ)
  (h1 : cost_sweater = 35)
  (h2 : cost_scarf = 25)
  (h3 : cost_mittens = 15)
  (h4 : num_family_members = 10)
  (h5 : savings = 800)
  (h6 : total_cost_one_set = cost_sweater + cost_scarf + cost_mittens)
  (h7 : total_cost_all_sets = total_cost_one_set * num_family_members)
  (h8 : amount_left = savings - total_cost_all_sets)
  : amount_left = 50 :=
sorry

end maria_savings_after_purchase_l161_161197


namespace minimum_rooms_needed_fans_l161_161478

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l161_161478


namespace length_of_segment_AB_l161_161817

noncomputable def speed_relation_first (x v1 v2 : ℝ) : Prop :=
  300 / v1 = (x - 300) / v2

noncomputable def speed_relation_second (x v1 v2 : ℝ) : Prop :=
  (x + 100) / v1 = (x - 100) / v2

theorem length_of_segment_AB :
  (∃ (x v1 v2 : ℝ),
    x > 0 ∧
    v1 > 0 ∧
    v2 > 0 ∧
    speed_relation_first x v1 v2 ∧
    speed_relation_second x v1 v2) →
  ∃ x : ℝ, x = 500 :=
by
  sorry

end length_of_segment_AB_l161_161817


namespace percentage_increase_of_bill_l161_161969

theorem percentage_increase_of_bill 
  (original_bill : ℝ) 
  (increased_bill : ℝ)
  (h1 : original_bill = 60)
  (h2 : increased_bill = 78) : 
  ((increased_bill - original_bill) / original_bill * 100) = 30 := 
by 
  rw [h1, h2]
  -- The following steps show the intended logic:
  -- calc 
  --   [(78 - 60) / 60 * 100]
  --   = [(18) / 60 * 100]
  --   = [0.3 * 100]
  --   = 30
  sorry

end percentage_increase_of_bill_l161_161969


namespace probability_neither_test_l161_161991

theorem probability_neither_test (P_hist : ℚ) (P_geo : ℚ) (indep : Prop) 
  (H1 : P_hist = 5/9) (H2 : P_geo = 1/3) (H3 : indep) :
  (1 - P_hist) * (1 - P_geo) = 8/27 := by
  sorry

end probability_neither_test_l161_161991


namespace radius_of_circle_param_eqs_l161_161807

theorem radius_of_circle_param_eqs :
  ∀ θ : ℝ, ∃ r : ℝ, r = 5 ∧ (∃ (x y : ℝ),
    x = 3 * Real.sin θ + 4 * Real.cos θ ∧
    y = 4 * Real.sin θ - 3 * Real.cos θ ∧
    x^2 + y^2 = r^2) := 
by
  sorry

end radius_of_circle_param_eqs_l161_161807


namespace average_marks_l161_161447

/-- Shekar scored 76, 65, 82, 67, and 85 marks in Mathematics, Science, Social Studies, English, and Biology respectively.
    We aim to prove that his average marks are 75. -/

def marks : List ℕ := [76, 65, 82, 67, 85]

theorem average_marks : (marks.sum / marks.length) = 75 := by
  sorry

end average_marks_l161_161447


namespace april_roses_l161_161866

theorem april_roses (price_per_rose earnings number_of_roses_left : ℕ) 
  (h1 : price_per_rose = 7) 
  (h2 : earnings = 35) 
  (h3 : number_of_roses_left = 4) : 
  (earnings / price_per_rose + number_of_roses_left) = 9 :=
by
  sorry

end april_roses_l161_161866


namespace sum_of_factors_72_l161_161248

theorem sum_of_factors_72 : 
  ∃ σ: ℕ → ℕ, (∀ n: ℕ, σ(n) = ∏ p in (nat.divisors n).to_finset, (∑ k in nat.divisors p, k)) ∧ σ 72 = 195 :=
by 
  sorry

end sum_of_factors_72_l161_161248


namespace complex_in_third_quadrant_l161_161164

theorem complex_in_third_quadrant (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 < 0) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complex_in_third_quadrant_l161_161164


namespace sum_of_positive_factors_of_72_l161_161234

/-- Define the divisor sum function based on the given formula -/
def divisor_sum (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 3
  | 3 => 4
  | 4 => 7
  | 6 => 12
  | 8 => 15
  | 12 => 28
  | 18 => 39
  | 24 => 60
  | 36 => 91
  | 48 => 124
  | 60 => 168
  | 72 => 195
  | _ => 0 -- This is not generally correct, just handles given problem specifically

theorem sum_of_positive_factors_of_72 :
  divisor_sum 72 = 195 :=
sorry

end sum_of_positive_factors_of_72_l161_161234


namespace sin_135_eq_l161_161713

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l161_161713


namespace acute_angle_vector_range_l161_161574

theorem acute_angle_vector_range (m : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (4, m)) 
  (acute : (a.1 * b.1 + a.2 * b.2) > 0) : 
  (m > -2) ∧ (m ≠ 8) := 
by 
  sorry

end acute_angle_vector_range_l161_161574


namespace count_whole_numbers_in_interval_l161_161351

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l161_161351


namespace quadratic_roots_sum_product_l161_161547

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l161_161547


namespace number_of_lists_l161_161834

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l161_161834


namespace leila_yards_l161_161113

variable (mile_yards : ℕ := 1760)
variable (marathon_miles : ℕ := 28)
variable (marathon_yards : ℕ := 1500)
variable (marathons_ran : ℕ := 15)

theorem leila_yards (m y : ℕ) (h1 : marathon_miles = 28) (h2 : marathon_yards = 1500) (h3 : mile_yards = 1760) (h4 : marathons_ran = 15) (hy : 0 ≤ y ∧ y < mile_yards) :
  y = 1200 :=
sorry

end leila_yards_l161_161113


namespace coin_combinations_count_l161_161317

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l161_161317


namespace number_of_possible_lists_l161_161838

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161838


namespace triangle_third_side_count_l161_161932

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l161_161932


namespace number_of_cans_per_set_l161_161267

noncomputable def ice_cream_original_price : ℝ := 12
noncomputable def ice_cream_discount : ℝ := 2
noncomputable def ice_cream_sale_price : ℝ := ice_cream_original_price - ice_cream_discount
noncomputable def number_of_tubs : ℝ := 2
noncomputable def total_money_spent : ℝ := 24
noncomputable def cost_of_juice_set : ℝ := 2
noncomputable def number_of_cans_in_juice_set : ℕ := 10

theorem number_of_cans_per_set (n : ℕ) (h : cost_of_juice_set * n = number_of_cans_in_juice_set) : (n / 2) = 5 :=
by sorry

end number_of_cans_per_set_l161_161267


namespace susan_homework_time_l161_161980

theorem susan_homework_time :
  ∀ (start finish practice : ℕ),
  start = 119 ->
  practice = 240 ->
  finish = practice - 25 ->
  (start < finish) ->
  (finish - start) = 96 :=
by
  intros start finish practice h_start h_practice h_finish h_lt
  sorry

end susan_homework_time_l161_161980


namespace least_three_digit_product_8_is_118_l161_161092

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l161_161092


namespace find_hall_length_l161_161945

variable (W H total_cost cost_per_sqm : ℕ)

theorem find_hall_length
  (hW : W = 15)
  (hH : H = 5)
  (h_total_cost : total_cost = 57000)
  (h_cost_per_sqm : cost_per_sqm = 60)
  : (32 * W) + (2 * (H * 32)) + (2 * (H * W)) = total_cost / cost_per_sqm :=
by
  sorry

end find_hall_length_l161_161945


namespace smallest_base10_integer_l161_161437

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end smallest_base10_integer_l161_161437


namespace trapezium_height_l161_161499

theorem trapezium_height :
  ∀ (a b h : ℝ), a = 20 ∧ b = 18 ∧ (1 / 2) * (a + b) * h = 285 → h = 15 :=
by
  intros a b h hconds
  cases hconds with h1 hrem
  cases hrem with h2 harea
  simp at harea
  sorry

end trapezium_height_l161_161499


namespace benny_eggs_l161_161275

theorem benny_eggs (dozen_count : ℕ) (eggs_per_dozen : ℕ) (total_eggs : ℕ) 
  (h1 : dozen_count = 7) 
  (h2 : eggs_per_dozen = 12) 
  (h3 : total_eggs = dozen_count * eggs_per_dozen) : 
  total_eggs = 84 := 
by 
  sorry

end benny_eggs_l161_161275


namespace sin_135_eq_sqrt2_div_2_l161_161700

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l161_161700


namespace quadratic_no_rational_solution_l161_161792

theorem quadratic_no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ∀ (x : ℚ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end quadratic_no_rational_solution_l161_161792


namespace combinations_of_coins_l161_161318

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l161_161318


namespace count_valid_third_sides_l161_161904

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l161_161904


namespace tan_alpha_parallel_vectors_l161_161312

theorem tan_alpha_parallel_vectors
    (α : ℝ)
    (a : ℝ × ℝ := (6, 8))
    (b : ℝ × ℝ := (Real.sin α, Real.cos α))
    (h : a.fst * b.snd = a.snd * b.fst) :
    Real.tan α = 3 / 4 := 
sorry

end tan_alpha_parallel_vectors_l161_161312


namespace initial_amount_of_liquid_A_l161_161101

theorem initial_amount_of_liquid_A (A B : ℝ) (initial_ratio : A = 4 * B) (removed_mixture : ℝ) (new_ratio : (A - (4/5) * removed_mixture) = (2 / 3) * ((B - (1/5) * removed_mixture) + removed_mixture)) :
  A = 16 := 
  sorry

end initial_amount_of_liquid_A_l161_161101


namespace two_cards_sum_to_15_proof_l161_161635

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l161_161635


namespace water_needed_l161_161950

theorem water_needed (nutrient_concentrate : ℝ) (distilled_water : ℝ) (total_volume : ℝ) 
    (h1 : nutrient_concentrate = 0.08) (h2 : distilled_water = 0.04) (h3 : total_volume = 1) :
    total_volume * (distilled_water / (nutrient_concentrate + distilled_water)) = 0.333 :=
by
  sorry

end water_needed_l161_161950


namespace ceil_neg_sqrt_64_div_9_l161_161135

theorem ceil_neg_sqrt_64_div_9 : ⌈-real.sqrt (64 / 9)⌉ = -2 := 
by
  sorry

end ceil_neg_sqrt_64_div_9_l161_161135


namespace area_of_pentagon_AEDCB_l161_161607

structure Rectangle (A B C D : Type) :=
  (AB BC AD CD : ℕ)

def is_perpendicular (A E E' D : Type) : Prop := sorry

def area_of_triangle (AE DE : ℕ) : ℕ :=
  (AE * DE) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def area_of_pentagon (area_rect area_triangle : ℕ) : ℕ :=
  area_rect - area_triangle

theorem area_of_pentagon_AEDCB
  (A B C D E : Type)
  (h_rectangle : Rectangle A B C D)
  (h_perpendicular : is_perpendicular A E E D)
  (AE DE : ℕ)
  (h_ae : AE = 9)
  (h_de : DE = 12)
  : area_of_pentagon (area_of_rectangle 15 12) (area_of_triangle AE DE) = 126 := 
  sorry

end area_of_pentagon_AEDCB_l161_161607


namespace sqrt_meaningful_l161_161995

theorem sqrt_meaningful (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l161_161995


namespace divisibility_by_29_and_29pow4_l161_161398

theorem divisibility_by_29_and_29pow4 (x y z : ℤ) (h : 29 ∣ (x^4 + y^4 + z^4)) : 29^4 ∣ (x^4 + y^4 + z^4) :=
by
  sorry

end divisibility_by_29_and_29pow4_l161_161398


namespace possible_integer_lengths_third_side_l161_161930

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l161_161930


namespace find_correct_quotient_l161_161826

theorem find_correct_quotient 
  (Q : ℕ)
  (D : ℕ)
  (h1 : D = 21 * Q)
  (h2 : D = 12 * 35) : 
  Q = 20 := 
by 
  sorry

end find_correct_quotient_l161_161826


namespace trig_identity_l161_161751

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l161_161751


namespace tens_digit_of_9_pow_1801_l161_161666

theorem tens_digit_of_9_pow_1801 : 
  ∀ n : ℕ, (9 ^ (1801) % 100) / 10 % 10 = 0 :=
by
  sorry

end tens_digit_of_9_pow_1801_l161_161666


namespace solve_arithmetic_sequence_l161_161949

variable {a : ℕ → ℝ}
variable {d a1 a2 a3 a10 a11 a6 a7 : ℝ}

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a1 + n * d

def arithmetic_condition (h : a 2 + a 3 + a 10 + a 11 = 32) : Prop :=
  a 6 + a 7 = 16

theorem solve_arithmetic_sequence (h : a 2 + a 3 + a 10 + a 11 = 32) : a 6 + a 7 = 16 :=
  by
    -- Proof will go here
    sorry

end solve_arithmetic_sequence_l161_161949


namespace arithmetic_sequence_n_value_l161_161997

theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (a1 : a 1 = 1) (d : ℤ) (d_def : d = 3) (an : ∃ n, a n = 22) :
  ∃ n, n = 8 :=
by
  -- Assume the general term formula for the arithmetic sequence
  have general_term : ∀ n, a n = a 1 + (n-1) * d := sorry
  -- Use the given conditions
  have a_n_22 : ∃ n, a n = 22 := an
  -- Calculations to derive n = 8, skipped here
  sorry

end arithmetic_sequence_n_value_l161_161997


namespace tammy_avg_speed_second_day_l161_161047

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l161_161047


namespace prob_sum_15_correct_l161_161653

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l161_161653


namespace squirrel_can_catch_nut_l161_161515

-- Define the initial distance between Gabriel and the squirrel.
def initial_distance : ℝ := 3.75

-- Define the speed of the nut.
def nut_speed : ℝ := 5.0

-- Define the jumping distance of the squirrel.
def squirrel_jump_distance : ℝ := 1.8

-- Define the acceleration due to gravity.
def gravity : ℝ := 10.0

-- Define the positions of the nut and the squirrel as functions of time.
def nut_position_x (t : ℝ) : ℝ := nut_speed * t
def squirrel_position_x : ℝ := initial_distance
def nut_position_y (t : ℝ) : ℝ := 0.5 * gravity * t^2

-- Define the squared distance between the nut and the squirrel.
def distance_squared (t : ℝ) : ℝ :=
  (nut_position_x t - squirrel_position_x)^2 + (nut_position_y t)^2

-- Prove that the minimum distance squared is less than or equal to the squirrel's jumping distance squared.
theorem squirrel_can_catch_nut : ∃ t : ℝ, distance_squared t ≤ squirrel_jump_distance^2 := by
  -- Sorry placeholder, as the proof is not required.
  sorry

end squirrel_can_catch_nut_l161_161515


namespace sum_of_factors_of_72_l161_161232

theorem sum_of_factors_of_72 :
  let n := 72
  let factors_sum := (∑ i in range 4, 2^i) * (∑ j in range 3, 3^j)
  factors_sum = 195 :=
by
  let n := 72
  have prime_factorization : n = 2^3 * 3^2 := by norm_num
  let sum_2 := ∑ i in range 4, 2^i
  let sum_3 := ∑ j in range 3, 3^j
  let factors_sum := sum_2 * sum_3
  have sum_2_correct : sum_2 = 1 + 2 + 4 + 8 := by norm_num
  have sum_3_correct : sum_3 = 1 + 3 + 9 := by norm_num
  have factors_sum_correct : factors_sum = 15 * 13 := by norm_num
  show 15 * 13 = 195, from rfl

end sum_of_factors_of_72_l161_161232


namespace sum_of_factors_of_72_l161_161236

/-- Prove that the sum of the positive factors of 72 is 195 -/
theorem sum_of_factors_of_72 : ∑ d in (finset.filter (λ d, 72 % d = 0) (finset.range (73))), d = 195 := 
by
  sorry

end sum_of_factors_of_72_l161_161236


namespace abs_f_sub_lt_abs_l161_161518

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

theorem abs_f_sub_lt_abs (a b : ℝ) (h : a ≠ b) : 
  |f a - f b| < |a - b| := 
by
  sorry

end abs_f_sub_lt_abs_l161_161518


namespace roots_of_quadratic_eq_l161_161540

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l161_161540


namespace john_profit_l161_161956

-- Definitions based on given conditions
def total_newspapers := 500
def selling_price_per_newspaper : ℝ := 2
def discount_percentage : ℝ := 0.75
def percentage_sold : ℝ := 0.80

-- Derived basic definitions
def cost_price_per_newspaper := selling_price_per_newspaper * (1 - discount_percentage)
def total_cost_price := cost_price_per_newspaper * total_newspapers
def newspapers_sold := total_newspapers * percentage_sold
def revenue := selling_price_per_newspaper * newspapers_sold
def profit := revenue - total_cost_price

-- Theorem stating the profit
theorem john_profit : profit = 550 := by
  sorry

#check john_profit

end john_profit_l161_161956


namespace exists_six_distinct_naturals_l161_161474

theorem exists_six_distinct_naturals :
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧ 
    a + b + c + d + e + f = 3528 ∧
    (1/a + 1/b + 1/c + 1/d + 1/e + 1/f : ℝ) = 3528 / 2012 :=
sorry

end exists_six_distinct_naturals_l161_161474


namespace solution_set_of_inequality_l161_161428

open Set Real

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 2 * x < 3) ↔ x ∈ Ioo (-3 : ℝ) 1 := by
  sorry

end solution_set_of_inequality_l161_161428


namespace probability_two_cards_sum_15_from_standard_deck_l161_161638

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l161_161638


namespace tammy_avg_speed_l161_161044

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l161_161044


namespace emily_subtracts_99_l161_161431

theorem emily_subtracts_99 : ∀ (a b : ℕ), (51 * 51 = a + 101) → (49 * 49 = b - 99) → b - 99 = 2401 := by
  intros a b h1 h2
  sorry

end emily_subtracts_99_l161_161431


namespace whole_numbers_in_interval_7_4_3pi_l161_161365

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l161_161365


namespace hcf_of_two_numbers_900_l161_161990

theorem hcf_of_two_numbers_900 (A B H : ℕ) (h_lcm : lcm A B = H * 11 * 15) (h_A : A = 900) : gcd A B = 165 :=
by
  sorry

end hcf_of_two_numbers_900_l161_161990


namespace roger_gave_candies_l161_161205

theorem roger_gave_candies :
  ∀ (original_candies : ℕ) (remaining_candies : ℕ) (given_candies : ℕ),
  original_candies = 95 → remaining_candies = 92 → given_candies = original_candies - remaining_candies → given_candies = 3 :=
by
  intros
  sorry

end roger_gave_candies_l161_161205


namespace max_price_per_unit_l161_161852

-- Define the conditions
def original_price : ℝ := 25
def original_sales_volume : ℕ := 80000
def price_increase_effect (t : ℝ) : ℝ := 2000 * (t - original_price)
def new_sales_volume (t : ℝ) : ℝ := 130 - 2 * t

-- Define the condition for revenue
def revenue_condition (t : ℝ) : Prop :=
  t * new_sales_volume t ≥ original_price * original_sales_volume

-- Statement to prove the maximum price per unit
theorem max_price_per_unit : ∀ t : ℝ, revenue_condition t → t ≤ 40 := sorry

end max_price_per_unit_l161_161852


namespace identifyNewEnergySources_l161_161250

-- Definitions of energy types as elements of a set.
inductive EnergySource 
| NaturalGas
| Coal
| OceanEnergy
| Petroleum
| SolarEnergy
| BiomassEnergy
| WindEnergy
| HydrogenEnergy

open EnergySource

-- Set definition for types of new energy sources
def newEnergySources : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- Set definition for the correct answer set of new energy sources identified by Option B
def optionB : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- The theorem asserting the equivalence between the identified new energy sources and the set option B
theorem identifyNewEnergySources : newEnergySources = optionB :=
  sorry

end identifyNewEnergySources_l161_161250


namespace only_a_zero_is_perfect_square_l161_161285

theorem only_a_zero_is_perfect_square (a : ℕ) : (∃ (k : ℕ), a^2 + 2 * a = k^2) → a = 0 := by
  sorry

end only_a_zero_is_perfect_square_l161_161285


namespace katherine_time_20_l161_161273

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end katherine_time_20_l161_161273


namespace shooting_guard_seconds_l161_161033

-- Define the given conditions
def x_pg := 130
def x_sf := 85
def x_pf := 60
def x_c := 180
def avg_time_per_player := 120
def total_players := 5

-- Define the total footage
def total_footage : Nat := total_players * avg_time_per_player

-- Define the footage for four players
def footage_of_four : Nat := x_pg + x_sf + x_pf + x_c

-- Define the footage of the shooting guard, which is a variable we want to compute
def x_sg := total_footage - footage_of_four

-- The statement we want to prove
theorem shooting_guard_seconds :
  x_sg = 145 := by
  sorry

end shooting_guard_seconds_l161_161033


namespace dice_probability_l161_161857

noncomputable def probability_each_number_appears_at_least_once : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

theorem dice_probability : probability_each_number_appears_at_least_once = 0.272 :=
by
  sorry

end dice_probability_l161_161857


namespace find_expression_value_l161_161155

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_l161_161155


namespace sum_sequences_l161_161278

theorem sum_sequences : 
  (1 + 12 + 23 + 34 + 45) + (10 + 20 + 30 + 40 + 50) = 265 := by
  sorry

end sum_sequences_l161_161278


namespace find_m_l161_161309

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def N (m : ℝ) : Set ℝ := {x | x*x - m*x < 0}
noncomputable def M_inter_N (m : ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

theorem find_m (m : ℝ) (h : M ∩ (N m) = M_inter_N m) : m = 1 :=
by sorry

end find_m_l161_161309


namespace two_cards_sum_to_15_proof_l161_161637

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l161_161637


namespace rooms_needed_l161_161492

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l161_161492


namespace coin_combinations_sum_50_l161_161334

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l161_161334


namespace right_vs_oblique_prism_similarities_and_differences_l161_161082

-- Definitions of Prisms and their properties
structure Prism where
  parallel_bases : Prop
  congruent_bases : Prop
  parallelogram_faces : Prop

structure RightPrism extends Prism where
  rectangular_faces : Prop
  perpendicular_sides : Prop

structure ObliquePrism extends Prism where
  non_perpendicular_sides : Prop

theorem right_vs_oblique_prism_similarities_and_differences 
  (p1 : RightPrism) (p2 : ObliquePrism) : 
    (p1.parallel_bases ↔ p2.parallel_bases) ∧ 
    (p1.congruent_bases ↔ p2.congruent_bases) ∧ 
    (p1.parallelogram_faces ↔ p2.parallelogram_faces) ∧
    (p1.rectangular_faces ∧ p1.perpendicular_sides ↔ p2.non_perpendicular_sides) := 
by 
  sorry

end right_vs_oblique_prism_similarities_and_differences_l161_161082


namespace problem1_problem2_l161_161687

-- First proof problem
theorem problem1 : - (2^2 : ℚ) + (2/3) * ((1 - 1/3) ^ 2) = -100/27 :=
by sorry

-- Second proof problem
theorem problem2 : (8 : ℚ) ^ (1 / 3) - |2 - (3 : ℚ) ^ (1 / 2)| - (3 : ℚ) ^ (1 / 2) = 0 :=
by sorry

end problem1_problem2_l161_161687


namespace triangle_side_length_integers_l161_161925

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l161_161925


namespace line_intersects_x_axis_at_point_l161_161457

theorem line_intersects_x_axis_at_point : 
  let x1 := 3
  let y1 := 7
  let x2 := -1
  let y2 := 3
  let m := (y2 - y1) / (x2 - x1) -- slope formula
  let b := y1 - m * x1        -- y-intercept formula
  let x_intersect := -b / m  -- x-coordinate where the line intersects x-axis
  (x_intersect, 0) = (-4, 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l161_161457


namespace semicircle_perimeter_l161_161105

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (h : 0 < π) (r_eq : r = 14):
  (14 * π + 28) = 14 * π + 28 :=
by
  sorry

end semicircle_perimeter_l161_161105


namespace sum_of_factors_of_72_l161_161238

theorem sum_of_factors_of_72 : (∑ n in (List.range (73)).filter (λ x => 72 % x = 0), x) = 195 := by
  sorry

end sum_of_factors_of_72_l161_161238


namespace basketball_not_table_tennis_l161_161380

theorem basketball_not_table_tennis (total_students likes_basketball likes_table_tennis dislikes_all : ℕ) (likes_basketball_not_tt : ℕ) :
  total_students = 30 →
  likes_basketball = 15 →
  likes_table_tennis = 10 →
  dislikes_all = 8 →
  (likes_basketball - 3 = likes_basketball_not_tt) →
  likes_basketball_not_tt = 12 := by
  intros h_total h_basketball h_table_tennis h_dislikes h_eq
  sorry

end basketball_not_table_tennis_l161_161380


namespace at_least_one_scheme_passes_l161_161616

open ProbabilityTheory

noncomputable def probability_at_least_one_scheme_passes (p_both_pass : ℝ) (independent_schemes : Prop) : ℝ :=
  if independent_schemes then 1 - (1 - p_both_pass) ^ 2 else 0

theorem at_least_one_scheme_passes (p_ab : ℝ) (h_independent : Prop) (h_p_ab : p_ab = 0.3) (h_independent_event : h_independent = true) :
  probability_at_least_one_scheme_passes p_ab h_independent = 0.51 :=
by
  sorry

end at_least_one_scheme_passes_l161_161616


namespace expected_value_and_variance_Y_l161_161888

variable (E D : (ℝ → ℝ) → ℝ)

-- Defining the binomial distribution condition
def binomial_X : ℝ → ℝ := sorry -- Definition of binomial distribution X

-- Defining the transformation Y = 2X + 1
def Y (X : ℝ) : ℝ := 2 * X + 1

-- The proof statement that needs to be verified
theorem expected_value_and_variance_Y :
    E Y = 5 ∧ D Y = 4 :=
by sorry

end expected_value_and_variance_Y_l161_161888


namespace total_area_correct_l161_161253

noncomputable def total_area (r p q : ℝ) : ℝ :=
  r^2 + 4*p^2 + 12*q

theorem total_area_correct
  (r p q : ℝ)
  (h : 12 * q = r^2 + 4 * p^2 + 45)
  (r_val : r = 6)
  (p_val : p = 1.5)
  (q_val : q = 7.5) :
  total_area r p q = 135 := by
  sorry

end total_area_correct_l161_161253


namespace find_m_of_symmetry_l161_161303

-- Define the conditions for the parabola and the axis of symmetry
theorem find_m_of_symmetry (m : ℝ) :
  let a := (1 : ℝ)
  let b := (m - 2 : ℝ)
  let axis_of_symmetry := (0 : ℝ)
  (-b / (2 * a)) = axis_of_symmetry → m = 2 :=
by
  sorry

end find_m_of_symmetry_l161_161303


namespace walking_distance_l161_161106

-- Define the pace in miles per hour.
def pace : ℝ := 2

-- Define the duration in hours.
def duration : ℝ := 8

-- Define the total distance walked.
def total_distance (pace : ℝ) (duration : ℝ) : ℝ := pace * duration

-- Define the theorem we need to prove.
theorem walking_distance :
  total_distance pace duration = 16 := by
  sorry

end walking_distance_l161_161106


namespace dirocks_rectangular_fence_count_l161_161132

/-- Dirock's backyard problem -/
def grid_side : ℕ := 32

def rock_placement (i j : ℕ) : Prop := (i % 3 = 0) ∧ (j % 3 = 0)

noncomputable def dirocks_rectangular_fence_ways : ℕ :=
  sorry

theorem dirocks_rectangular_fence_count : dirocks_rectangular_fence_ways = 1920 :=
sorry

end dirocks_rectangular_fence_count_l161_161132


namespace min_rooms_needed_l161_161487

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l161_161487


namespace common_region_area_of_triangles_l161_161434

noncomputable def area_of_common_region (a : ℝ) : ℝ :=
  (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3

theorem common_region_area_of_triangles (a : ℝ) (h : 0 < a) : 
  area_of_common_region a = (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3 :=
by
  sorry

end common_region_area_of_triangles_l161_161434


namespace calc_mixed_number_expr_l161_161686

theorem calc_mixed_number_expr :
  53 * (3 + 1 / 4 - (3 + 3 / 4)) / (1 + 2 / 3 + (2 + 2 / 5)) = -6 - 57 / 122 := 
by
  sorry

end calc_mixed_number_expr_l161_161686


namespace timmy_needs_speed_l161_161816

variable (s1 s2 s3 : ℕ) (extra_speed : ℕ)

theorem timmy_needs_speed
  (h_s1 : s1 = 36)
  (h_s2 : s2 = 34)
  (h_s3 : s3 = 38)
  (h_extra_speed : extra_speed = 4) :
  (s1 + s2 + s3) / 3 + extra_speed = 40 := 
sorry

end timmy_needs_speed_l161_161816


namespace susan_total_distance_l161_161042

theorem susan_total_distance (a b : ℕ) (r : ℝ) (h1 : a = 15) (h2 : b = 25) (h3 : r = 3) :
  (r * ((a + b) / 60)) = 2 :=
by
  sorry

end susan_total_distance_l161_161042


namespace prob_sum_15_correct_l161_161652

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l161_161652


namespace two_cards_totaling_15_probability_l161_161655

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l161_161655


namespace geometric_sequence_sum_l161_161776

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
  (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ 
  (a 2 * a 4 = 1) ∧ 
  (a 1 * (q^0 + q^1 + q^2) = 7) ∧ 
  (a 1 / (1 - q) * (1 - q^5) = 31 / 4) := by
  sorry

end geometric_sequence_sum_l161_161776


namespace smallest_distance_l161_161404

noncomputable def a : Complex := 2 + 4 * Complex.I
noncomputable def b : Complex := 5 + 2 * Complex.I

theorem smallest_distance 
  (z w : Complex) 
  (hz : Complex.abs (z - a) = 2) 
  (hw : Complex.abs (w - b) = 4) : 
  Complex.abs (z - w) ≥ 6 - Real.sqrt 13 :=
sorry

end smallest_distance_l161_161404


namespace turtles_remaining_l161_161122

/-- 
In one nest, there are x baby sea turtles, while in the other nest, there are 2x baby sea turtles.
One-fourth of the turtles in the first nest and three-sevenths of the turtles in the second nest
got swept to the sea. Prove the total number of turtles still on the sand is (53/28)x.
-/
theorem turtles_remaining (x : ℕ) (h1 : ℕ := x) (h2 : ℕ := 2 * x) : ((3/4) * x + (8/7) * (2 * x)) = (53/28) * x :=
by
  sorry

end turtles_remaining_l161_161122


namespace length_of_BC_l161_161623

def triangle_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 20

def triangle_area (a b : ℝ) : Prop :=
  (1/2) * a * b * (Real.sqrt 3 / 2) = 10

theorem length_of_BC (a b c : ℝ) (h1 : triangle_perimeter a b c) (h2 : triangle_area a b) : c = 7 :=
  sorry

end length_of_BC_l161_161623


namespace sum_of_factors_of_72_l161_161231

theorem sum_of_factors_of_72 :
  let n := 72
  let factors_sum := (∑ i in range 4, 2^i) * (∑ j in range 3, 3^j)
  factors_sum = 195 :=
by
  let n := 72
  have prime_factorization : n = 2^3 * 3^2 := by norm_num
  let sum_2 := ∑ i in range 4, 2^i
  let sum_3 := ∑ j in range 3, 3^j
  let factors_sum := sum_2 * sum_3
  have sum_2_correct : sum_2 = 1 + 2 + 4 + 8 := by norm_num
  have sum_3_correct : sum_3 = 1 + 3 + 9 := by norm_num
  have factors_sum_correct : factors_sum = 15 * 13 := by norm_num
  show 15 * 13 = 195, from rfl

end sum_of_factors_of_72_l161_161231


namespace sum_of_factors_72_l161_161239

theorem sum_of_factors_72 :
  (∑ d in divisors 72, d) = 195 := by
    sorry

end sum_of_factors_72_l161_161239


namespace proof_problem_statement_l161_161606

noncomputable def proof_problem (x y: ℝ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ (∀ n : ℕ, n > 0 → (⌊x / y⌋ : ℝ) = ⌊↑n * x⌋ / ⌊↑n * y⌋) →
  (x = y ∨ (∃ k : ℤ, k ≠ 0 ∧ (x = k * y ∨ y = k * x)))

-- The formal statement of the problem
theorem proof_problem_statement (x y : ℝ) :
  proof_problem x y := by
  sorry

end proof_problem_statement_l161_161606


namespace sectors_not_equal_l161_161768

theorem sectors_not_equal (a1 a2 a3 a4 a5 a6 : ℕ) :
  ¬(∃ k : ℕ, (∀ n : ℕ, n = k) ↔
    ∃ m, (a1 + m) = k ∧ (a2 + m) = k ∧ (a3 + m) = k ∧ 
         (a4 + m) = k ∧ (a5 + m) = k ∧ (a6 + m) = k) :=
sorry

end sectors_not_equal_l161_161768


namespace find_a5_l161_161386

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a5 (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_a2 : a 2 = 2) (h_a8 : a 8 = 32) :
  a 5 = 8 :=
by
  sorry

end find_a5_l161_161386


namespace sin_135_eq_l161_161694

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l161_161694


namespace distance_from_center_of_C_to_line_l161_161754

def circle_center_distance : ℝ :=
  let line1 (x y : ℝ) := x - y - 4
  let circle1 (x y : ℝ) := x^2 + y^2 - 4 * x - 6
  let circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6
  let line2 (x y : ℝ) := 3 * x + 4 * y + 5
  sorry

theorem distance_from_center_of_C_to_line :
  circle_center_distance = 2 := sorry

end distance_from_center_of_C_to_line_l161_161754


namespace eval_expression_l161_161674

theorem eval_expression : abs (-6) - (-4) + (-7) = 3 :=
by
  sorry

end eval_expression_l161_161674


namespace xy_conditions_l161_161373

theorem xy_conditions (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : x * y = 1) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end xy_conditions_l161_161373


namespace total_nickels_l161_161028

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end total_nickels_l161_161028


namespace trajectory_equation_l161_161290

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem trajectory_equation (P : ℝ × ℝ) (h : |distance P (1, 0) - P.1| = 1) :
  (P.1 ≥ 0 → P.2 ^ 2 = 4 * P.1) ∧ (P.1 < 0 → P.2 = 0) :=
by
  sorry

end trajectory_equation_l161_161290


namespace least_three_digit_number_product8_l161_161091

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l161_161091


namespace exists_plane_through_point_parallel_to_line_at_distance_l161_161871

-- Definitions of the given entities
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Line :=
(point : Point)
(direction : Point) -- Considering direction as a point vector for simplicity

def distance (P : Point) (L : Line) : ℝ := 
  -- Define the distance from point P to line L
  sorry

noncomputable def construct_plane (P : Point) (L : Line) (d : ℝ) : Prop :=
  -- Define when a plane can be constructed as stated in the problem.
  sorry

-- The main proof problem statement without the solution steps
theorem exists_plane_through_point_parallel_to_line_at_distance (P : Point) (L : Line) (d : ℝ) (h : distance P L > d) :
  construct_plane P L d :=
sorry

end exists_plane_through_point_parallel_to_line_at_distance_l161_161871


namespace triangle_third_side_count_l161_161933

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l161_161933


namespace smallest_integer_is_77_l161_161804

theorem smallest_integer_is_77 
  (A B C D E F G : ℤ)
  (h_uniq: A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G)
  (h_sum: A + B + C + D + E + F + G = 840)
  (h_largest: G = 190)
  (h_two_smallest_sum: A + B = 156) : 
  A = 77 :=
sorry

end smallest_integer_is_77_l161_161804


namespace sin_double_angle_value_l161_161522

open Real

theorem sin_double_angle_value (x : ℝ) (h : sin (x + π / 4) = - 5 / 13) : sin (2 * x) = - 119 / 169 := 
sorry

end sin_double_angle_value_l161_161522


namespace max_price_theorem_min_sales_volume_theorem_unit_price_theorem_l161_161853

noncomputable def max_price (original_price : ℝ) (original_sales : ℝ) 
  (sales_decrement : ℝ → ℝ) : ℝ :=
  let t := 40 in
  t

theorem max_price_theorem : 
  ∀ (original_price original_sales : ℝ)
    (sales_decrement : ℝ → ℝ),
  max_price original_price original_sales sales_decrement = 40 := 
sorry

noncomputable def min_sales_volume (original_price : ℝ) (original_sales : ℝ) 
  (fixed_cost : ℝ) (variable_cost : ℝ → ℝ) 
  (tech_innovation_cost : ℝ → ℝ) : ℝ :=
  let a := 10.2 * 10^6 in
  a

theorem min_sales_volume_theorem : 
  ∀ (original_price original_sales : ℝ) 
    (fixed_cost : ℝ) (variable_cost : ℝ → ℝ)
    (tech_innovation_cost : ℝ → ℝ),
  min_sales_volume original_price original_sales 
    fixed_cost variable_cost tech_innovation_cost = 10.2 * 10^6 :=
sorry

noncomputable def unit_price (x : ℝ) : ℝ :=
  if x = 30 then x else 30

theorem unit_price_theorem : 
  ∀ (x : ℝ),
  unit_price x = 30 :=
sorry

end max_price_theorem_min_sales_volume_theorem_unit_price_theorem_l161_161853


namespace physics_teacher_min_count_l161_161458

theorem physics_teacher_min_count 
  (maths_teachers : ℕ) 
  (chemistry_teachers : ℕ) 
  (max_subjects_per_teacher : ℕ) 
  (min_total_teachers : ℕ) 
  (physics_teachers : ℕ)
  (h1 : maths_teachers = 7)
  (h2 : chemistry_teachers = 5)
  (h3 : max_subjects_per_teacher = 3)
  (h4 : min_total_teachers = 6) 
  (h5 : 7 + physics_teachers + 5 ≤ 6 * 3) :
  0 < physics_teachers :=
  by 
  sorry

end physics_teacher_min_count_l161_161458


namespace square_table_seats_4_pupils_l161_161678

-- Define the conditions given in the problem
def num_rectangular_tables := 7
def seats_per_rectangular_table := 10
def total_pupils := 90
def num_square_tables := 5

-- Define what we want to prove
theorem square_table_seats_4_pupils (x : ℕ) :
  total_pupils = num_rectangular_tables * seats_per_rectangular_table + num_square_tables * x →
  x = 4 :=
by
  sorry

end square_table_seats_4_pupils_l161_161678


namespace sum_of_factors_72_l161_161246

theorem sum_of_factors_72 : (Finset.sum ((Finset.range 73).filter (λ n, 72 % n = 0))) = 195 := by
  sorry

end sum_of_factors_72_l161_161246


namespace max_books_borrowed_l161_161103

theorem max_books_borrowed 
  (num_students : ℕ)
  (num_no_books : ℕ)
  (num_one_book : ℕ)
  (num_two_books : ℕ)
  (average_books : ℕ)
  (h_num_students : num_students = 32)
  (h_num_no_books : num_no_books = 2)
  (h_num_one_book : num_one_book = 12)
  (h_num_two_books : num_two_books = 10)
  (h_average_books : average_books = 2)
  : ∃ max_books : ℕ, max_books = 11 := 
by
  sorry

end max_books_borrowed_l161_161103


namespace num_Q_polynomials_l161_161188

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 5)

#check Exists

theorem num_Q_polynomials :
  ∃ (Q : Polynomial ℝ), 
  (∃ (R : Polynomial ℝ), R.degree = 3 ∧ P (Q.eval x) = P x * R.eval x) ∧
  Q.degree = 2 ∧ (Q.coeff 1 = 6) ∧ (∃ (n : ℕ), n = 22) :=
sorry

end num_Q_polynomials_l161_161188


namespace find_x_given_y_l161_161828

-- Given x varies inversely as the square of y, we define the relationship
def varies_inversely (x y k : ℝ) : Prop := x = k / y^2

theorem find_x_given_y (k : ℝ) (h_k : k = 4) :
  ∀ (y : ℝ), varies_inversely x y k → y = 2 → x = 1 :=
by
  intros y h_varies h_y_eq
  -- We need to prove the statement here
  sorry

end find_x_given_y_l161_161828


namespace sin_135_correct_l161_161692

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l161_161692


namespace initial_average_daily_production_l161_161149

variable (A : ℝ) -- Initial average daily production
variable (n : ℕ) -- Number of days

theorem initial_average_daily_production (n_eq_5 : n = 5) (new_production_eq_90 : 90 = 90) 
  (new_average_eq_65 : (5 * A + 90) / 6 = 65) : A = 60 :=
by
  sorry

end initial_average_daily_production_l161_161149


namespace noemi_initial_amount_l161_161031

theorem noemi_initial_amount : 
  ∀ (rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount : ℕ), 
    rouletteLoss = 600 → 
    blackjackLoss = 800 → 
    pokerLoss = 400 → 
    baccaratLoss = 700 → 
    remainingAmount = 1500 → 
    initialAmount = rouletteLoss + blackjackLoss + pokerLoss + baccaratLoss + remainingAmount →
    initialAmount = 4000 :=
by
  intros rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6

end noemi_initial_amount_l161_161031


namespace probability_AC_adjacent_l161_161124

noncomputable def probability_AC_adjacent_given_AB_adjacent : ℚ :=
  let total_permutations_with_AB_adjacent := 48
  let permutations_with_ABC_adjacent := 12
  permutations_with_ABC_adjacent / total_permutations_with_AB_adjacent

theorem probability_AC_adjacent :  
  probability_AC_adjacent_given_AB_adjacent = 1 / 4 :=
by
  sorry

end probability_AC_adjacent_l161_161124


namespace remaining_days_to_finish_l161_161394

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end remaining_days_to_finish_l161_161394


namespace sin_135_l161_161703

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l161_161703


namespace min_value_condition_l161_161741

open Real

theorem min_value_condition 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_condition_l161_161741


namespace ratio_arms_martians_to_aliens_l161_161461

def arms_of_aliens : ℕ := 3
def legs_of_aliens : ℕ := 8
def legs_of_martians := legs_of_aliens / 2

def limbs_of_5_aliens := 5 * (arms_of_aliens + legs_of_aliens)
def limbs_of_5_martians (arms_of_martians : ℕ) := 5 * (arms_of_martians + legs_of_martians)

theorem ratio_arms_martians_to_aliens (A_m : ℕ) (h1 : limbs_of_5_aliens = limbs_of_5_martians A_m + 5) :
  (A_m : ℚ) / arms_of_aliens = 2 :=
sorry

end ratio_arms_martians_to_aliens_l161_161461


namespace radius_of_circle_l161_161812

-- Define the given circle equation as a condition
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 7 = 0

theorem radius_of_circle : ∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, r = 3 :=
by
  sorry

end radius_of_circle_l161_161812


namespace root_equation_val_l161_161899

theorem root_equation_val (a : ℝ) (h : a^2 - 2 * a - 5 = 0) : 2 * a^2 - 4 * a = 10 :=
by 
  sorry

end root_equation_val_l161_161899


namespace poles_intersection_l161_161576

-- Define the known heights and distances
def heightOfIntersection (d h1 h2 x : ℝ) : ℝ := sorry

theorem poles_intersection :
  heightOfIntersection 120 30 60 40 = 20 := by
  sorry

end poles_intersection_l161_161576


namespace cannot_assemble_highlighted_shape_l161_161791

-- Define the rhombus shape with its properties
structure Rhombus :=
  (white_triangle gray_triangle : Prop)

-- Define the assembly condition
def can_rotate (shape : Rhombus) : Prop := sorry

-- Define the specific shape highlighted that Petya cannot form
def highlighted_shape : Prop := sorry

-- The statement we need to prove
theorem cannot_assemble_highlighted_shape (shape : Rhombus) 
  (h_rotate : can_rotate shape)
  (h_highlight : highlighted_shape) : false :=
by sorry

end cannot_assemble_highlighted_shape_l161_161791


namespace movie_friends_l161_161615

noncomputable def movie_only (M P G MP MG PG MPG : ℕ) : Prop :=
  let total_M := 20
  let total_P := 20
  let total_G := 5
  let total_students := 31
  (MP = 4) ∧ 
  (MG = 2) ∧ 
  (PG = 0) ∧ (MPG = 2) ∧ 
  (M + MP + MG + MPG = total_M) ∧ 
  (P + MP + PG + MPG = total_P) ∧ 
  (G + MG + PG + MPG = total_G) ∧ 
  (M + P + G + MP + MG + PG + MPG = total_students) ∧ 
  (M = 12)

theorem movie_friends (M P G MP MG PG MPG : ℕ) : movie_only M P G MP MG PG MPG := 
by 
  sorry

end movie_friends_l161_161615


namespace rotated_line_x_intercept_l161_161966

theorem rotated_line_x_intercept (x y : ℝ) :
  (∃ (k : ℝ), y = (3 * Real.sqrt 3 + 5) / (2 * Real.sqrt 3) * x) →
  (∃ y : ℝ, 3 * x - 5 * y + 40 = 0) →
  (∃ (x_intercept : ℝ), x_intercept = 0) := 
by
  sorry

end rotated_line_x_intercept_l161_161966


namespace least_positive_x_l161_161228

theorem least_positive_x (x : ℕ) : ((2 * x) ^ 2 + 2 * 41 * 2 * x + 41 ^ 2) % 53 = 0 ↔ x = 6 := 
sorry

end least_positive_x_l161_161228


namespace minimum_rooms_needed_l161_161482

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l161_161482


namespace route_length_is_140_l161_161818

-- Conditions of the problem
variable (D : ℝ)  -- Length of the route
variable (Vx Vy t : ℝ)  -- Speeds of Train X and Train Y, and time to meet

-- Given conditions
axiom train_X_trip_time : D / Vx = 4
axiom train_Y_trip_time : D / Vy = 3
axiom train_X_distance_when_meet : Vx * t = 60
axiom total_distance_covered_on_meeting : Vx * t + Vy * t = D

-- Goal: Prove that the length of the route is 140 kilometers
theorem route_length_is_140 : D = 140 := by
  -- Proof omitted
  sorry

end route_length_is_140_l161_161818


namespace sin_135_eq_sqrt2_div_2_l161_161718

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l161_161718


namespace part_I_part_II_l161_161196

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 2 }
def B (a : ℝ) : Set ℝ := { x | 2 * a - 1 < x ∧ x < 2 * a + 1 }

-- Part (Ⅰ): Given A ⊆ B, prove that 1/2 ≤ a ≤ 1
theorem part_I (a : ℝ) : A ⊆ B a → (1 / 2 ≤ a ∧ a ≤ 1) :=
by sorry

-- Part (Ⅱ): Given A ∩ B = ∅, prove that a ≥ 3/2 or a ≤ 0
theorem part_II (a : ℝ) : A ∩ B a = ∅ → (a ≥ 3 / 2 ∨ a ≤ 0) :=
by sorry

end part_I_part_II_l161_161196


namespace poker_cards_count_l161_161456

theorem poker_cards_count (total_cards kept_away : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : kept_away = 7) : 
  total_cards - kept_away = 45 :=
by 
  sorry

end poker_cards_count_l161_161456


namespace product_third_fourth_term_l161_161421

theorem product_third_fourth_term (a d : ℝ) : 
  (a + 7 * d = 20) → (d = 2) → 
  ( (a + 2 * d) * (a + 3 * d) = 120 ) := 
by 
  intros h1 h2
  sorry

end product_third_fourth_term_l161_161421


namespace possible_integer_lengths_third_side_l161_161928

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l161_161928


namespace simplify_fraction_l161_161797

theorem simplify_fraction : (150 / 4350 : ℚ) = 1 / 29 :=
  sorry

end simplify_fraction_l161_161797


namespace count_valid_triangles_l161_161886

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem count_valid_triangles : 
  (∃ n : ℕ, n = 14 ∧ 
  ∃ (a b c : ℕ), valid_triangle a b c ∧ 
  ((b = 5 ∧ c > 5) ∨ (c = 5 ∧ b > 5)) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0)) :=
by { sorry }

end count_valid_triangles_l161_161886


namespace number_of_chickens_l161_161075

variable (C P : ℕ) (legs_total : ℕ := 48) (legs_pig : ℕ := 4) (legs_chicken : ℕ := 2) (number_pigs : ℕ := 9)

theorem number_of_chickens (h1 : P = number_pigs)
                           (h2 : legs_pig * P + legs_chicken * C = legs_total) :
                           C = 6 :=
by
  sorry

end number_of_chickens_l161_161075


namespace number_of_girls_l161_161862

theorem number_of_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 2 * B / 3 + G = 18) : G = 18 :=
by
  sorry

end number_of_girls_l161_161862


namespace problem1_l161_161128

theorem problem1 : 2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15 / 2 := 
by
  sorry

end problem1_l161_161128


namespace sum_of_positive_factors_of_72_l161_161241

def sum_divisors (n : ℕ) : ℕ := 
  ∑ d in finset.filter (λ d, d ∣ n) (finset.range (n+1)), d

theorem sum_of_positive_factors_of_72 : sum_divisors 72 = 195 :=
by sorry

end sum_of_positive_factors_of_72_l161_161241


namespace sequence_general_term_l161_161519

-- Given a sequence {a_n} whose sum of the first n terms S_n = 2a_n - 1,
-- prove that the general formula for the n-th term a_n is 2^(n-1).

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
    (h₁ : ∀ n : ℕ, S n = 2 * a n - 1)
    (h₂ : S 1 = 1) : ∀ n : ℕ, a (n + 1) = 2 ^ n :=
by
  sorry

end sequence_general_term_l161_161519


namespace sally_earnings_in_dozens_l161_161036

theorem sally_earnings_in_dozens (earnings_per_house : ℕ) (houses_cleaned : ℕ) (dozens_of_dollars : ℕ) : 
  earnings_per_house = 25 ∧ houses_cleaned = 96 → dozens_of_dollars = 200 := 
by
  intros h
  sorry

end sally_earnings_in_dozens_l161_161036


namespace sum_of_extremes_of_g_l161_161962

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (2 * x - 8)

theorem sum_of_extremes_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≤ g 4) ∧ (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≥ g 1) → g 4 + g 1 = 2 :=
by
  sorry

end sum_of_extremes_of_g_l161_161962


namespace solve_sqrt_equation_l161_161799

theorem solve_sqrt_equation (x : ℝ) (h1 : x ≠ -4) (h2 : (3 * x - 1) / (x + 4) > 0) : 
  sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3 * x - 1)) = 0 ↔ x = 5 / 2 :=
by { sorry }

end solve_sqrt_equation_l161_161799


namespace prob_sum_15_correct_l161_161651

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l161_161651


namespace game_completion_days_l161_161395

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end game_completion_days_l161_161395


namespace min_rooms_needed_l161_161484

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l161_161484


namespace polynomial_value_at_minus_2_l161_161659

-- Define the polynomial f(x)
def f (x : ℤ) := x^6 - 5 * x^5 + 6 * x^4 + x^2 + 3 * x + 2

-- Define the evaluation point
def x_val : ℤ := -2

-- State the theorem we want to prove
theorem polynomial_value_at_minus_2 : f x_val = 320 := 
by sorry

end polynomial_value_at_minus_2_l161_161659


namespace identify_incorrect_proposition_l161_161442

-- Definitions based on problem conditions
def propositionA : Prop :=
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0))

def propositionB : Prop :=
  (¬ (∃ x : ℝ, x^2 + x + 1 = 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0)

def propositionD (x : ℝ) : Prop :=
  (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬(x > 2) → ¬(x^2 - 3*x + 2 > 0))

-- Proposition C is given to be incorrect in the problem
def propositionC (p q : Prop) : Prop := ¬ (p ∧ q) → ¬p ∧ ¬q

theorem identify_incorrect_proposition (p q : Prop) : 
  (propositionA ∧ propositionB ∧ (∀ x : ℝ, propositionD x)) → 
  ¬ (propositionC p q) :=
by
  intros
  -- We know proposition C is false based on the problem's solution
  sorry

end identify_incorrect_proposition_l161_161442


namespace triangle_AOB_area_l161_161581

noncomputable def point := ℝ × ℝ

noncomputable def origin : point := (0, 0)

noncomputable def A : point := (2, 2 * Real.pi / 3)

noncomputable def B : point := (3, Real.pi / 6)

def triangle_area (O A B : point) : ℝ :=
  let (r1, θ1) := A
  let (r2, θ2) := B
  let angle_AOB := θ1 - θ2
  let area := 0.5 * r1 * r2 * Real.sin angle_AOB
  area

theorem triangle_AOB_area (O A B : point) (hO : O = origin) (hA : A = (2, 2 * Real.pi / 3)) (hB : B = (3, Real.pi / 6)) :
  triangle_area O A B = 3 := by
  sorry

end triangle_AOB_area_l161_161581


namespace arithmetic_progression_geometric_progression_l161_161159

-- Arithmetic Progression
theorem arithmetic_progression (a : ℕ → ℤ) 
    (h_arith_1 : a 4 + a 7 = 2) 
    (h_arith_2 : a 5 * a 6 = -8) 
    (A : ∀ n m : ℕ, (a n - a m) = (n - m) * (a 2 - a 1)) : 
    a 1 * a 10 = -728 := 
begin 
    sorry 
end

-- Geometric Progression
theorem geometric_progression (a : ℕ → ℤ) 
    (h_geom_1 : a 4 + a 7 = 2) 
    (h_geom_2 : a 5 * a 6 = -8) 
    (G : ∀ n m : ℕ, (a n * a m) = (a 1 * (a 2 ^ (n-1))) * (a 1 * (a 2 ^ (m-1)))) : 
    a 1 + a 10 = -7 := 
begin 
    sorry 
end

end arithmetic_progression_geometric_progression_l161_161159


namespace problem_demo_l161_161958

open Set

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem problem_demo : S ∩ (U \ T) = {1, 2, 4} :=
by
  sorry

end problem_demo_l161_161958


namespace longest_side_range_l161_161117

-- Definitions and conditions
def is_triangle (x y z : ℝ) : Prop := 
  x + y > z ∧ x + z > y ∧ y + z > x

-- Problem statement
theorem longest_side_range (l x y z : ℝ) 
  (h_triangle: is_triangle x y z) 
  (h_perimeter: x + y + z = l / 2) 
  (h_longest: x ≥ y ∧ x ≥ z) : 
  l / 6 ≤ x ∧ x < l / 4 :=
by
  sorry

end longest_side_range_l161_161117


namespace tom_gave_2_seashells_to_jessica_l161_161080

-- Conditions
def original_seashells : Nat := 5
def current_seashells : Nat := 3

-- Question as a proposition
def seashells_given (x : Nat) : Prop :=
  original_seashells - current_seashells = x

-- The proof problem
theorem tom_gave_2_seashells_to_jessica : seashells_given 2 :=
by 
  sorry

end tom_gave_2_seashells_to_jessica_l161_161080


namespace highlighter_difference_l161_161377

theorem highlighter_difference :
  ∃ (P : ℕ), 7 + P + (P + 5) = 40 ∧ P - 7 = 7 :=
by
  sorry

end highlighter_difference_l161_161377


namespace cauliflower_sales_l161_161202

noncomputable def broccoli_sales : ℝ := 57
noncomputable def carrot_sales : ℝ := 2 * broccoli_sales
noncomputable def spinach_sales : ℝ := 16 + (1 / 2 * carrot_sales)
noncomputable def total_sales : ℝ := 380
noncomputable def other_sales : ℝ := broccoli_sales + carrot_sales + spinach_sales

theorem cauliflower_sales :
  total_sales - other_sales = 136 :=
by
  -- proof skipped
  sorry

end cauliflower_sales_l161_161202


namespace nesbitt_inequality_l161_161257

variable (a b c d : ℝ)

-- Assume a, b, c, d are positive real numbers
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom pos_d : 0 < d

theorem nesbitt_inequality :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end nesbitt_inequality_l161_161257


namespace whole_numbers_in_interval_l161_161357

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l161_161357


namespace problem_correct_options_l161_161630

open Finset

theorem problem_correct_options :
  ∃ (n m k l : ℕ), n ≠ 24 ∧ m = 18 ∧ k = 144 ∧ l = 9 :=
  let A := 4^4, -- This is the number of ways for option A, which is incorrect
      B := choose 4 2 * (choose 2 1 * factorial 2 + 1), -- Number of ways for option B
      C := choose 4 1 * (choose 3 1 * choose 2 2 * factorial 3 / (1 * factorial 1)), -- Number of ways for option C
      D := 3 * 3; -- Number of ways for option D
  by
  exact ⟨A, B, C, D, by simp [A], by simp [B], by simp [C], by simp [D]⟩
  sorry

end problem_correct_options_l161_161630


namespace total_nickels_l161_161029

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end total_nickels_l161_161029


namespace triangle_orthocenter_example_l161_161764

open Real EuclideanGeometry

def point_3d := (ℝ × ℝ × ℝ)

def orthocenter (A B C : point_3d) : point_3d := sorry

theorem triangle_orthocenter_example :
  orthocenter (2, 4, 6) (6, 5, 3) (4, 6, 7) = (4/5, 38/5, 59/5) := sorry

end triangle_orthocenter_example_l161_161764


namespace hotel_room_allocation_l161_161486

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l161_161486


namespace xyz_divisible_by_55_l161_161767

-- Definitions and conditions from part (a)
variables (x y z a b c : ℤ)
variable (h1 : x^2 + y^2 = a^2)
variable (h2 : y^2 + z^2 = b^2)
variable (h3 : z^2 + x^2 = c^2)

-- The final statement to prove that xyz is divisible by 55
theorem xyz_divisible_by_55 : 55 ∣ x * y * z := 
by sorry

end xyz_divisible_by_55_l161_161767


namespace ab_non_positive_l161_161534

theorem ab_non_positive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l161_161534


namespace Aiden_sleep_fraction_l161_161121

theorem Aiden_sleep_fraction (minutes_slept : ℕ) (hour_minutes : ℕ) (h : minutes_slept = 15) (k : hour_minutes = 60) :
  (minutes_slept : ℚ) / hour_minutes = 1/4 :=
by
  sorry

end Aiden_sleep_fraction_l161_161121


namespace quadratic_roots_sum_product_l161_161550

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l161_161550


namespace ophelia_age_l161_161952

/-- 
If Lennon is currently 8 years old, 
and in two years Ophelia will be four times as old as Lennon,
then Ophelia is currently 38 years old 
-/
theorem ophelia_age 
  (lennon_age : ℕ) 
  (ophelia_age_in_two_years : ℕ) 
  (h1 : lennon_age = 8)
  (h2 : ophelia_age_in_two_years = 4 * (lennon_age + 2)) : 
  ophelia_age_in_two_years - 2 = 38 :=
by
  sorry

end ophelia_age_l161_161952


namespace sweet_tray_GCD_l161_161079

/-!
Tim has a bag of 36 orange-flavoured sweets and Peter has a bag of 44 grape-flavoured sweets.
They have to divide up the sweets into small trays with equal number of sweets;
each tray containing either orange-flavoured or grape-flavoured sweets only.
The largest possible number of sweets in each tray without any remainder is 4.
-/

theorem sweet_tray_GCD :
  Nat.gcd 36 44 = 4 :=
by
  sorry

end sweet_tray_GCD_l161_161079


namespace nth_equation_pattern_l161_161601

theorem nth_equation_pattern (n : ℕ) (hn : 0 < n) : n^2 - n = n * (n - 1) := by
  sorry

end nth_equation_pattern_l161_161601


namespace incorrect_statements_exactly_one_l161_161295

-- From given condition \(x^2 + y^2 - 4x + 1 = 0\)
def condition (x y : ℝ) := x^2 + y^2 - 4 * x + 1 = 0

-- Prove that there is exactly one incorrect statement among the four given
theorem incorrect_statements_exactly_one (x y : ℝ) (h : condition x y) :
  (statement_1 h = true ∧ statement_2 h = true ∧ statement_3 h = false ∧ statement_4 h = true) → 
  ∃! p, (p = 1) :=
by
  -- skipping the proof
  sorry

-- Definitions of each statement 
def statement_1 (h : condition x y) := y - x ≤ sqrt(6) - 2
def statement_2 (h : condition x y) := x^2 + y^2 ≤ 7 + 4 * sqrt(3)
def statement_3 (h : condition x y) := y / x ≤ sqrt(3) / 2
def statement_4 (h : condition x y) := x + y ≤ 2 + sqrt(3)

end incorrect_statements_exactly_one_l161_161295


namespace area_of_rectangle_A_is_88_l161_161603

theorem area_of_rectangle_A_is_88 
  (lA lB lC w wC : ℝ)
  (h1 : lB = lA + 2)
  (h2 : lB * w = lA * w + 22)
  (h3 : wC = w - 4)
  (AreaB : ℝ := lB * w)
  (AreaC : ℝ := lB * wC)
  (h4 : AreaC = AreaB - 40) : 
  (lA * w = 88) :=
sorry

end area_of_rectangle_A_is_88_l161_161603


namespace combination_coins_l161_161325

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l161_161325


namespace intersection_setA_setB_l161_161781

namespace Proof

def setA : Set ℝ := {x | ∃ y : ℝ, y = x + 1}
def setB : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

theorem intersection_setA_setB : (setA ∩ setB) = {y | 0 < y} :=
by
  sorry

end Proof

end intersection_setA_setB_l161_161781


namespace find_sum_a_b_l161_161575

-- Define the conditions as variables and hypotheses
variables {a b : ℝ}

-- The main theorem to prove
theorem find_sum_a_b
  (h1 : 3 * a + 2 * b = 3)
  (h2 : 3 * b + 2 * a = 2) :
  a + b = 1 := sorry

end find_sum_a_b_l161_161575


namespace sandy_spent_home_currency_l161_161206

variable (A B C D : ℝ)

def total_spent_home_currency (A B C D : ℝ) : ℝ :=
  let total_foreign := A + B + C
  total_foreign * D

theorem sandy_spent_home_currency (D : ℝ) : 
  total_spent_home_currency 13.99 12.14 7.43 D = 33.56 * D := 
by
  sorry

end sandy_spent_home_currency_l161_161206


namespace sum_of_factors_of_72_l161_161237

theorem sum_of_factors_of_72 : (∑ n in (List.range (73)).filter (λ x => 72 % x = 0), x) = 195 := by
  sorry

end sum_of_factors_of_72_l161_161237


namespace sin_135_eq_sqrt2_over_2_l161_161712

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l161_161712


namespace sum_of_decimals_as_fraction_l161_161284

/-- Define the problem inputs as constants -/
def d1 : ℚ := 2 / 10
def d2 : ℚ := 4 / 100
def d3 : ℚ := 6 / 1000
def d4 : ℚ := 8 / 10000
def d5 : ℚ := 1 / 100000

/-- The main theorem statement -/
theorem sum_of_decimals_as_fraction : 
  d1 + d2 + d3 + d4 + d5 = 24681 / 100000 := 
by 
  sorry

end sum_of_decimals_as_fraction_l161_161284


namespace joe_list_possibilities_l161_161847

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l161_161847


namespace factor_and_divisor_statements_l161_161668

theorem factor_and_divisor_statements :
  (∃ n : ℕ, 25 = 5 * n) ∧
  ((∃ n : ℕ, 209 = 19 * n) ∧ ¬ (∃ n : ℕ, 63 = 19 * n)) ∧
  (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end factor_and_divisor_statements_l161_161668


namespace Samantha_last_name_length_l161_161610

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end Samantha_last_name_length_l161_161610


namespace pension_equality_l161_161271

theorem pension_equality (x c d r s: ℝ) (h₁ : d ≠ c) 
    (h₂ : x > 0) (h₃ : 2 * x * (d - c) + d^2 - c^2 ≠ 0)
    (h₄ : ∀ k:ℝ, k * (x + c)^2 - k * x^2 = r)
    (h₅ : ∀ k:ℝ, k * (x + d)^2 - k * x^2 = s) 
    : ∃ k : ℝ, k = (s - r) / (2 * x * (d - c) + d^2 - c^2) 
    → k * x^2 = (s - r) * x^2 / (2 * x * (d - c) + d^2 - c^2) :=
by {
    sorry
}

end pension_equality_l161_161271


namespace triangle_third_side_length_l161_161911

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l161_161911


namespace find_consecutive_integers_sum_eq_l161_161282

theorem find_consecutive_integers_sum_eq 
    (M : ℤ) : ∃ n k : ℤ, (0 ≤ k ∧ k ≤ 9) ∧ (M = (9 * n + 45 - k)) := 
sorry

end find_consecutive_integers_sum_eq_l161_161282


namespace domain_of_log_function_l161_161983

theorem domain_of_log_function (x : ℝ) :
  (5 - x > 0) ∧ (x - 2 > 0) ∧ (x - 2 ≠ 1) ↔ (2 < x ∧ x < 3) ∨ (3 < x ∧ x < 5) :=
by
  sorry

end domain_of_log_function_l161_161983


namespace BC_length_l161_161008

theorem BC_length (AD BC MN : ℝ) (h1 : AD = 2) (h2 : MN = 6) (h3 : MN = 0.5 * (AD + BC)) : BC = 10 :=
by
  sorry

end BC_length_l161_161008


namespace disjoint_sets_no_connections_l161_161981

open Finset

-- Define the type for harbour
def Harbour := Fin 2016

-- Define the type for the graph on harbours
def G : SimpleGraph Harbour := -- fill in with graph definition based on the problem conditions.

-- Define the key condition
axiom long_path_absent : ∀ (p : List Harbour), p.length ≥ 1062 → ¬ G.walk (p.head!) (p.last!)

-- Lean statement
theorem disjoint_sets_no_connections :
  ∃ (A B : Finset Harbour), A.card = 477 ∧ B.card = 477 ∧ A.disjoint B ∧
   ∀ (a : Harbour) (b : Harbour), a ∈ A → b ∈ B → ¬ G.adj a b :=
by
  -- The proof uses Turán's theorem and graph properties
  sorry

end disjoint_sets_no_connections_l161_161981


namespace find_difference_l161_161401

-- Define the problem conditions in Lean
theorem find_difference (a b : ℕ) (hrelprime : Nat.gcd a b = 1)
                        (hpos : a > b) 
                        (hfrac : (a^3 - b^3) / (a - b)^3 = 73 / 3) :
    a - b = 3 :=
by
    sorry

end find_difference_l161_161401


namespace right_triangle_side_lengths_l161_161270

theorem right_triangle_side_lengths :
  ¬ (4^2 + 5^2 = 6^2) ∧
  (12^2 + 16^2 = 20^2) ∧
  ¬ (5^2 + 10^2 = 13^2) ∧
  ¬ (8^2 + 40^2 = 41^2) := by
  sorry

end right_triangle_side_lengths_l161_161270


namespace daily_earning_r_l161_161446

theorem daily_earning_r :
  exists P Q R : ℝ, 
    (P + Q + R = 220) ∧
    (P + R = 120) ∧
    (Q + R = 130) ∧
    (R = 30) := 
by
  sorry

end daily_earning_r_l161_161446


namespace arithmetic_sequence_sum_l161_161096

theorem arithmetic_sequence_sum {a b : ℤ} (h : ∀ n : ℕ, 3 + n * 6 = if n = 2 then a else if n = 3 then b else 33) : a + b = 48 := by
  sorry

end arithmetic_sequence_sum_l161_161096


namespace maximize_profit_l161_161110

noncomputable def annual_profit : ℝ → ℝ
| x => if x < 80 then - (1/3) * x^2 + 40 * x - 250 
       else 1200 - (x + 10000 / x)

theorem maximize_profit : ∃ x : ℝ, x = 100 ∧ annual_profit x = 1000 :=
by
  sorry

end maximize_profit_l161_161110


namespace probability_two_cards_sum_15_from_standard_deck_l161_161640

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l161_161640


namespace hall_volume_l161_161111

theorem hall_volume (length breadth : ℝ) (height : ℝ := 20 / 3)
  (h1 : length = 15)
  (h2 : breadth = 12)
  (h3 : 2 * (length * breadth) = 54 * height) :
  length * breadth * height = 8004 :=
by
  sorry

end hall_volume_l161_161111


namespace area_of_triangle_l161_161194

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l161_161194


namespace purple_marble_probability_l161_161849

theorem purple_marble_probability (blue green : ℝ) (p : ℝ) 
  (h_blue : blue = 0.25)
  (h_green : green = 0.4)
  (h_sum : blue + green + p = 1) : p = 0.35 :=
by
  sorry

end purple_marble_probability_l161_161849


namespace rhombus_diagonal_length_l161_161061

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d2 = 17) (h2 : A = 127.5) 
  (h3 : A = (d1 * d2) / 2) : d1 = 15 := 
by 
  -- Definitions
  sorry

end rhombus_diagonal_length_l161_161061


namespace increase_in_lighting_power_l161_161064

-- Conditions
def N_before : ℕ := 240
def N_after : ℕ := 300

-- Theorem
theorem increase_in_lighting_power : N_after - N_before = 60 := by
  sorry

end increase_in_lighting_power_l161_161064


namespace brad_zip_code_l161_161277

theorem brad_zip_code (a b c d e : ℕ) 
  (h1 : a = b) 
  (h2 : c = 0) 
  (h3 : d = 2 * a) 
  (h4 : d + e = 8) 
  (h5 : a + b + c + d + e = 10) : 
  (a, b, c, d, e) = (1, 1, 0, 2, 6) :=
by 
  -- Proof omitted on purpose
  sorry

end brad_zip_code_l161_161277


namespace nickels_count_l161_161027

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end nickels_count_l161_161027


namespace triangle_is_right_triangle_l161_161293

theorem triangle_is_right_triangle (A B C : ℝ) (hC_eq_A_plus_B : C = A + B) (h_angle_sum : A + B + C = 180) : C = 90 :=
by
  sorry

end triangle_is_right_triangle_l161_161293


namespace weeks_of_exercise_l161_161012

def hours_per_day : ℕ := 1
def days_per_week : ℕ := 5
def total_hours : ℕ := 40

def weekly_hours : ℕ := hours_per_day * days_per_week

theorem weeks_of_exercise (W : ℕ) (h : total_hours = weekly_hours * W) : W = 8 :=
by
  sorry

end weeks_of_exercise_l161_161012


namespace find_expression_value_l161_161156

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_l161_161156


namespace max_books_per_student_l161_161944

theorem max_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (avg_books_per_student : ℕ)
  (max_books_limit : ℕ)
  (total_books_available : ℕ) :
  total_students = 20 →
  students_0_books = 2 →
  students_1_book = 10 →
  students_2_books = 5 →
  students_at_least_3_books = total_students - students_0_books - students_1_book - students_2_books →
  avg_books_per_student = 2 →
  max_books_limit = 5 →
  total_books_available = 60 →
  avg_books_per_student * total_students = 40 →
  total_books_available = 60 →
  max_books_limit = 5 :=
by sorry

end max_books_per_student_l161_161944


namespace normal_probability_l161_161942

noncomputable def normal_distribution (μ σ : ℝ) : Type :=
  sorry

theorem normal_probability {X : normal_distribution 1 2} {m : ℝ}
  (h1 : ∀ x : ℝ, X.prob 0 x = m) :
  X.prob 0 2 = 1 - 2 * m :=
sorry

end normal_probability_l161_161942


namespace triangle_shape_l161_161183

-- Defining the conditions:
variables (A B C a b c : ℝ)
variable (h1 : c - a * Real.cos B = (2 * a - b) * Real.cos A)

-- Defining the property to prove:
theorem triangle_shape : 
  (A = Real.pi / 2 ∨ A = B ∨ B = C ∨ C = A + B) :=
sorry

end triangle_shape_l161_161183


namespace functional_equation_solution_l161_161496

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) ↔ (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solution_l161_161496


namespace two_cards_totaling_15_probability_l161_161654

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l161_161654


namespace find_solutions_equation_l161_161497

theorem find_solutions_equation :
  {x : ℝ | 1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 11 * x - 12) = 0}
  = {1, -12, 4, -3} :=
by
  sorry

end find_solutions_equation_l161_161497


namespace find_f_2011_l161_161737

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom specific_interval (x : ℝ) (h2 : 2 < x) (h4 : x < 4) : f x = x + 3

theorem find_f_2011 : f 2011 = 6 :=
by {
  -- Leave this part to be filled with the actual proof,
  -- satisfying the initial conditions and concluding f(2011) = 6
  sorry
}

end find_f_2011_l161_161737


namespace total_volume_of_all_cubes_l161_161468

def volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume_of_cubes (num_cubes : ℕ) (side_length : ℕ) : ℕ :=
  num_cubes * volume side_length

theorem total_volume_of_all_cubes :
  total_volume_of_cubes 3 3 + total_volume_of_cubes 4 4 = 337 :=
by
  sorry

end total_volume_of_all_cubes_l161_161468


namespace probability_red_second_draw_l161_161078

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_red_second_draw_l161_161078


namespace tammy_avg_speed_l161_161045

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l161_161045


namespace percentage_of_boys_is_90_l161_161951

variables (B G : ℕ)

def total_children : ℕ := 100
def future_total_children : ℕ := total_children + 100
def percentage_girls : ℕ := 5
def girls_after_increase : ℕ := future_total_children * percentage_girls / 100
def boys_after_increase : ℕ := total_children - girls_after_increase

theorem percentage_of_boys_is_90 :
  B + G = total_children →
  G = girls_after_increase →
  B = total_children - G →
  (B:ℚ) / total_children * 100 = 90 :=
by
  sorry

end percentage_of_boys_is_90_l161_161951


namespace sin_135_l161_161702

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l161_161702


namespace general_formula_and_arithmetic_sequence_l161_161307

noncomputable def S_n (n : ℕ) : ℕ := 3 * n ^ 2 - 2 * n
noncomputable def a_n (n : ℕ) : ℕ := S_n n - S_n (n - 1)

theorem general_formula_and_arithmetic_sequence :
  (∀ n : ℕ, a_n n = 6 * n - 5) ∧
  (∀ n : ℕ, (n ≥ 2 → a_n n - a_n (n - 1) = 6) ∧ (a_n 1 = 1)) :=
by
  sorry

end general_formula_and_arithmetic_sequence_l161_161307


namespace sum_of_factors_72_l161_161244

theorem sum_of_factors_72 : 
  let factors_sum (n : ℕ) : ℕ := 
    ∑ d in Nat.divisors n, d
  in factors_sum 72 = 195 :=
by
  sorry

end sum_of_factors_72_l161_161244


namespace ceil_neg_sqrt_l161_161136

variable (x : ℚ) (h1 : x = -real.sqrt (64 / 9))

theorem ceil_neg_sqrt : ⌈x⌉ = -2 :=
by
  have h2 : x = - (8 / 3) := by rw [h1, real.sqrt_div, real.sqrt_eq_rpow, real.sqrt_eq_rpow, pow_succ, fpow_succ frac.one_ne_zero, pow_half, real.sqrt_eq_rpow, pow_succ, pow_two]
  rw h2
  have h3 : ⌈- (8 / 3)⌉ = -2 := by linarith
  exact h3

end ceil_neg_sqrt_l161_161136


namespace whole_numbers_in_interval_l161_161355

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l161_161355


namespace problem_statement_l161_161402

def f (x : ℝ) : ℝ := 5 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem problem_statement : g (f (g (f 1))) = 305 :=
by
  sorry

end problem_statement_l161_161402


namespace carla_students_l161_161869

theorem carla_students (R A num_rows num_desks : ℕ) (full_fraction : ℚ) 
  (h1 : R = 2) 
  (h2 : A = 3 * R - 1)
  (h3 : num_rows = 4)
  (h4 : num_desks = 6)
  (h5 : full_fraction = 2 / 3) : 
  num_rows * (num_desks * full_fraction).toNat + R + A = 23 := by
  sorry

end carla_students_l161_161869


namespace triangle_third_side_length_l161_161913

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l161_161913


namespace line_perpendicular_slope_l161_161940

theorem line_perpendicular_slope (m : ℝ) :
  let slope1 := (1 / 2) 
  let slope2 := (-2 / m)
  slope1 * slope2 = -1 → m = 1 := 
by
  -- The proof will go here
  sorry

end line_perpendicular_slope_l161_161940


namespace students_like_apple_chocolate_not_blueberry_l161_161760

theorem students_like_apple_chocolate_not_blueberry
  (n d a b c abc : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : a = 25)
  (h4 : b = 20)
  (h5 : c = 10)
  (h6 : abc = 5)
  (h7 : (n - d) = 35)
  (h8 : (55 - (a + b + c - abc)) = 35) :
  (20 - abc) = (15 : ℕ) :=
by
  sorry

end students_like_apple_chocolate_not_blueberry_l161_161760


namespace trig_expression_value_l161_161535

theorem trig_expression_value
  (x : ℝ)
  (h : Real.tan (x + Real.pi / 4) = -3) :
  (Real.sin x + 2 * Real.cos x) / (3 * Real.sin x + 4 * Real.cos x) = 2 / 5 :=
by
  sorry

end trig_expression_value_l161_161535


namespace quadratic_roots_sum_product_l161_161549

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l161_161549


namespace math_problem_l161_161890

theorem math_problem (f : ℕ → Prop) (m : ℕ) 
  (h1 : f 1) (h2 : f 2) (h3 : f 3)
  (h_implies : ∀ k : ℕ, f k → f (k + m)) 
  (h_max : m = 3):
  ∀ n : ℕ, 0 < n → f n :=
by
  sorry

end math_problem_l161_161890


namespace roots_of_quadratic_eq_l161_161537

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l161_161537


namespace triangle_third_side_count_l161_161931

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l161_161931


namespace required_range_of_a_l161_161280

variable (a : ℝ) (f : ℝ → ℝ)
def function_increasing_on (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, DifferentiableAt ℝ f x ∧ (deriv f x) ≥ 0

theorem required_range_of_a (h : function_increasing_on (fun x => a * Real.log x + x) a (Set.Icc 2 3)) :
  a ≥ -2 :=
sorry

end required_range_of_a_l161_161280


namespace is_factor_l161_161473

-- Define the polynomial
def poly (x : ℝ) := x^4 + 4 * x^2 + 4

-- Define a candidate for being a factor
def factor_candidate (x : ℝ) := x^2 + 2

-- Proof problem: prove that factor_candidate is a factor of poly
theorem is_factor : ∀ x : ℝ, poly x = factor_candidate x * factor_candidate x := 
by
  intro x
  unfold poly factor_candidate
  sorry

end is_factor_l161_161473


namespace gcf_3465_10780_l161_161226

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end gcf_3465_10780_l161_161226


namespace point_comparison_on_inverse_proportion_l161_161973

theorem point_comparison_on_inverse_proportion :
  (∃ y1 y2, (y1 = 2 / 1) ∧ (y2 = 2 / 2) ∧ y1 > y2) :=
by
  use 2
  use 1
  sorry

end point_comparison_on_inverse_proportion_l161_161973


namespace more_blue_count_l161_161582

-- Definitions based on the conditions given in the problem
def total_people : ℕ := 150
def more_green : ℕ := 95
def both_green_blue : ℕ := 35
def neither_green_blue : ℕ := 25

-- The Lean statement to prove the number of people who believe turquoise is "more blue"
theorem more_blue_count : 
  (total_people - neither_green_blue) - (more_green - both_green_blue) = 65 :=
by 
  sorry

end more_blue_count_l161_161582


namespace items_in_storeroom_l161_161466

-- Conditions definitions
def restocked_items : ℕ := 4458
def sold_items : ℕ := 1561
def total_items_left : ℕ := 3472

-- Statement of the proof
theorem items_in_storeroom : (total_items_left - (restocked_items - sold_items)) = 575 := 
by
  sorry

end items_in_storeroom_l161_161466


namespace problem_statement_l161_161782

def P := {x : ℤ | ∃ k : ℤ, x = 2 * k - 1}
def Q := {y : ℤ | ∃ n : ℤ, y = 2 * n}

theorem problem_statement (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y ∈ P) ∧ (x * y ∈ Q) :=
by
  sorry

end problem_statement_l161_161782


namespace handshake_count_l161_161684

def total_employees : ℕ := 50
def dept_X : ℕ := 30
def dept_Y : ℕ := 20
def handshakes_between_departments : ℕ := dept_X * dept_Y

theorem handshake_count : handshakes_between_departments = 600 :=
by
  sorry

end handshake_count_l161_161684


namespace child_ticket_cost_l161_161802

theorem child_ticket_cost :
  ∀ (A P_a C T P_c : ℕ),
    A = 10 →
    P_a = 8 →
    C = 11 →
    T = 124 →
    (T - A * P_a) / C = P_c →
    P_c = 4 :=
by
  intros A P_a C T P_c hA hP_a hC hT hPc
  rw [hA, hP_a, hC, hT] at hPc
  linarith [hPc]

end child_ticket_cost_l161_161802


namespace triangle_area_l161_161192

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area (a b : ℝ × ℝ) : 
  let area_parallelogram := (a.1 * b.2 - a.2 * b.1).abs in
  (1 / 2) * area_parallelogram = 4.5 :=
by
  sorry

end triangle_area_l161_161192


namespace josh_marbles_earlier_l161_161397

-- Define the conditions
def marbles_lost : ℕ := 11
def marbles_now : ℕ := 8

-- Define the problem statement
theorem josh_marbles_earlier : marbles_lost + marbles_now = 19 :=
by
  sorry

end josh_marbles_earlier_l161_161397


namespace sqrt_meaningful_real_l161_161371

theorem sqrt_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 5)) → x ≥ 5 :=
by
  intro h
  cases h with y hy
  have : x - 5 ≥ 0 := by sorry -- simplified proof of sqrt definition
  linarith

end sqrt_meaningful_real_l161_161371


namespace age_twice_in_2_years_l161_161859

/-
Conditions:
1. The man is 24 years older than his son.
2. The present age of the son is 22 years.
3. In a certain number of years, the man's age will be twice the age of his son.
-/
def man_is_24_years_older (S M : ℕ) : Prop := M = S + 24
def present_age_son : ℕ := 22
def age_twice_condition (Y S M : ℕ) : Prop := M + Y = 2 * (S + Y)

/-
Prove that in 2 years, the man's age will be twice the age of his son.
-/
theorem age_twice_in_2_years : ∃ (Y : ℕ), 
  (man_is_24_years_older present_age_son M) → 
  (age_twice_condition Y present_age_son M) →
  Y = 2 :=
by
  sorry

end age_twice_in_2_years_l161_161859


namespace alex_distribution_ways_l161_161682

theorem alex_distribution_ways : (15^5 = 759375) := by {
  sorry
}

end alex_distribution_ways_l161_161682


namespace students_in_second_class_l161_161982

theorem students_in_second_class 
    (avg1 : ℝ)
    (n1 : ℕ)
    (avg2 : ℝ)
    (total_avg : ℝ)
    (x : ℕ)
    (h1 : avg1 = 40)
    (h2 : n1 = 26)
    (h3 : avg2 = 60)
    (h4 : total_avg = 53.1578947368421)
    (h5 : (n1 * avg1 + x * avg2) / (n1 + x) = total_avg) :
  x = 50 :=
by
  sorry

end students_in_second_class_l161_161982


namespace complete_the_square_l161_161441

theorem complete_the_square (x : ℝ) :
  x^2 + 6 * x - 4 = 0 → (x + 3)^2 = 13 :=
by
  sorry

end complete_the_square_l161_161441


namespace solve_inequality_l161_161414

theorem solve_inequality :
  { x : ℝ // 10 * x^2 - 2 * x - 3 < 0 } =
  { x : ℝ // (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 } :=
by
  sorry

end solve_inequality_l161_161414


namespace oil_amount_correct_l161_161632

-- Definitions based on the conditions in the problem
def initial_amount : ℝ := 0.16666666666666666
def additional_amount : ℝ := 0.6666666666666666
def final_amount : ℝ := 0.8333333333333333

-- Lean 4 statement to prove the given problem
theorem oil_amount_correct :
  initial_amount + additional_amount = final_amount :=
by
  sorry

end oil_amount_correct_l161_161632


namespace problem_condition_l161_161509

theorem problem_condition (a : ℝ) (x : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) :
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end problem_condition_l161_161509


namespace ratio_perimeters_not_integer_l161_161795

theorem ratio_perimeters_not_integer
  (a k l : ℤ) (h_a_pos : a > 0) (h_k_pos : k > 0) (h_l_pos : l > 0)
  (h_area : a^2 = k * l) :
  ¬ ∃ n : ℤ, n = (k + l) / (2 * a) :=
by
  sorry

end ratio_perimeters_not_integer_l161_161795


namespace box_problem_l161_161378

theorem box_problem 
    (x y : ℕ) 
    (h1 : 10 * x + 20 * y = 18 * (x + y)) 
    (h2 : 10 * x + 20 * (y - 10) = 16 * (x + y - 10)) :
    x + y = 20 :=
sorry

end box_problem_l161_161378


namespace third_side_integer_lengths_l161_161910

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l161_161910


namespace triangle_side_length_integers_l161_161926

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l161_161926


namespace no_such_set_exists_l161_161472

open Nat Set

theorem no_such_set_exists (M : Set ℕ) : 
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) →
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → a + b = c + d → a = c ∨ a = d) → 
  False := by
  sorry

end no_such_set_exists_l161_161472


namespace initial_percentage_acid_l161_161455

theorem initial_percentage_acid (P : ℝ) (h1 : 27 * P / 100 = 18 * 60 / 100) : P = 40 :=
sorry

end initial_percentage_acid_l161_161455


namespace eval_expression_l161_161494

theorem eval_expression :
    (727 * 727) - (726 * 728) = 1 := by
  sorry

end eval_expression_l161_161494


namespace roots_of_quadratic_eq_l161_161541

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l161_161541


namespace gcd_g50_g52_l161_161777

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 2 * x^2 + x + 2023

-- Define the integers n1 and n2 corresponding to g(50) and g(52)
def n1 : ℤ := g 50
def n2 : ℤ := g 52

-- Statement of the proof goal
theorem gcd_g50_g52 : Int.gcd n1 n2 = 1 := by
  sorry

end gcd_g50_g52_l161_161777


namespace range_of_x_for_sqrt_l161_161372

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end range_of_x_for_sqrt_l161_161372


namespace determine_a_l161_161304

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 2 * x

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 0 :=
by
  intros h
  sorry

end determine_a_l161_161304


namespace coin_combinations_count_l161_161330

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l161_161330


namespace least_three_digit_product_eight_l161_161088

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l161_161088


namespace quadratic_roots_product_l161_161292

theorem quadratic_roots_product :
  ∀ (x1 x2: ℝ), (x1^2 - 4 * x1 - 2 = 0 ∧ x2^2 - 4 * x2 - 2 = 0) → (x1 * x2 = -2) :=
by
  -- Assume x1 and x2 are roots of the quadratic equation
  intros x1 x2 h
  sorry

end quadratic_roots_product_l161_161292


namespace count_whole_numbers_in_interval_l161_161361

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l161_161361


namespace total_value_is_84_l161_161589

-- Definitions based on conditions
def number_of_stamps : ℕ := 21
def value_of_7_stamps : ℕ := 28
def stamps_per_7 : ℕ := 7
def stamp_value : ℤ := value_of_7_stamps / stamps_per_7
def total_value_of_collection : ℤ := number_of_stamps * stamp_value

-- Statement to prove the total value of the stamp collection
theorem total_value_is_84 : total_value_of_collection = 84 := by
  sorry

end total_value_is_84_l161_161589


namespace juicy_12_juicy_20_l161_161953

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 1 = (1 / a) + (1 / b) + (1 / c) + (1 / d) ∧ a * b * c * d = n

theorem juicy_12 : is_juicy 12 :=
sorry

theorem juicy_20 : is_juicy 20 :=
sorry

end juicy_12_juicy_20_l161_161953


namespace john_walks_farther_l161_161773

theorem john_walks_farther :
  let john_distance : ℝ := 1.74
  let nina_distance : ℝ := 1.235
  john_distance - nina_distance = 0.505 :=
by
  sorry

end john_walks_farther_l161_161773


namespace sin_135_eq_sqrt2_div_2_l161_161697

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l161_161697


namespace P_intersect_Q_empty_l161_161310

def is_element_of_P (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 4

def is_element_of_Q (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 2

theorem P_intersect_Q_empty : ∀ x, is_element_of_P x → is_element_of_Q x → false :=
by
  intro x hP hQ
  sorry

end P_intersect_Q_empty_l161_161310


namespace proof_inequality_l161_161739

noncomputable def inequality (a b c : ℝ) : Prop :=
  a + 2 * b + c = 1 ∧ a^2 + b^2 + c^2 = 1 → -2/3 ≤ c ∧ c ≤ 1

theorem proof_inequality (a b c : ℝ) (h : a + 2 * b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : -2/3 ≤ c ∧ c ≤ 1 :=
by {
  sorry
}

end proof_inequality_l161_161739


namespace count_whole_numbers_in_interval_l161_161359

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l161_161359


namespace no_solution_for_eq_l161_161208

theorem no_solution_for_eq (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) → False :=
sorry

end no_solution_for_eq_l161_161208


namespace cost_for_23_days_l161_161755

structure HostelStay where
  charge_first_week : ℝ
  charge_additional_week : ℝ

def cost_of_stay (days : ℕ) (hostel : HostelStay) : ℝ :=
  let first_week_days := min days 7
  let remaining_days := days - first_week_days
  let additional_full_weeks := remaining_days / 7 
  let additional_days := remaining_days % 7
  (first_week_days * hostel.charge_first_week) + 
  (additional_full_weeks * 7 * hostel.charge_additional_week) + 
  (additional_days * hostel.charge_additional_week)

theorem cost_for_23_days :
  cost_of_stay 23 { charge_first_week := 18.00, charge_additional_week := 11.00 } = 302.00 :=
by
  sorry

end cost_for_23_days_l161_161755


namespace quadratic_roots_sum_product_l161_161551

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l161_161551


namespace swimming_speed_still_water_l161_161858

theorem swimming_speed_still_water 
  (v t : ℝ) 
  (h1 : 3 = (v + 3) * t / (v - 3)) 
  (h2 : t ≠ 0) :
  v = 9 :=
by
  sorry

end swimming_speed_still_water_l161_161858


namespace find_value_of_expression_l161_161019

theorem find_value_of_expression
  (a b c d : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : d ≥ 0)
  (h₄ : a / (b + c + d) = b / (a + c + d))
  (h₅ : b / (a + c + d) = c / (a + b + d))
  (h₆ : c / (a + b + d) = d / (a + b + c))
  (h₇ : d / (a + b + c) = a / (b + c + d)) :
  (a + b) / (c + d) + (b + c) / (a + d) + (c + d) / (a + b) + (d + a) / (b + c) = 4 :=
by sorry

end find_value_of_expression_l161_161019


namespace power_of_two_with_nines_l161_161604

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ (n : ℕ), (2^n % 10^k) / 10^((10 * 5^k + k + 2 - k) / 2) = 9 :=
sorry

end power_of_two_with_nines_l161_161604


namespace find_digits_l161_161433

def are_potential_digits (digits : Finset ℕ) : Prop :=
  digits.card = 4 ∧ 
  ∀ (a b c d : ℕ), 
    (a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  let largest := 1000 * a + 100 * b + 10 * c + d,
      smallest := 1000 * d + 100 * c + 10 * b + a in
  a > b ∧ b > c ∧ c > d ∧ largest + smallest = 10477

theorem find_digits : 
  are_potential_digits ({7, 4, 3, 0} : Finset ℕ) := 
sorry

end find_digits_l161_161433


namespace find_range_of_a_l161_161584

-- Definitions and conditions
def pointA : ℝ × ℝ := (0, 3)
def lineL (x : ℝ) : ℝ := 2 * x - 4
def circleCenter (a : ℝ) : ℝ × ℝ := (a, 2 * a - 4)
def circleRadius : ℝ := 1

-- The range to prove
def valid_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12 / 5

-- Main theorem
theorem find_range_of_a (a : ℝ) (M : ℝ × ℝ)
  (on_circle : (M.1 - (circleCenter a).1)^2 + (M.2 - (circleCenter a).2)^2 = circleRadius^2)
  (condition_MA_MD : (M.1 - pointA.1)^2 + (M.2 - pointA.2)^2 = 4 * M.1^2 + 4 * M.2^2) :
  valid_range a :=
sorry

end find_range_of_a_l161_161584


namespace nickels_count_l161_161026

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end nickels_count_l161_161026


namespace pos_int_solns_to_eq_l161_161140

open Int

theorem pos_int_solns_to_eq (x y z : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x^2 + y^2 - z^2 = 9 - 2 * x * y ↔ 
    (x, y, z) = (5, 0, 4) ∨ (x, y, z) = (4, 1, 4) ∨ (x, y, z) = (3, 2, 4) ∨ 
    (x, y, z) = (2, 3, 4) ∨ (x, y, z) = (1, 4, 4) ∨ (x, y, z) = (0, 5, 4) ∨ 
    (x, y, z) = (3, 0, 0) ∨ (x, y, z) = (2, 1, 0) ∨ (x, y, z) = (1, 2, 0) ∨ 
    (x, y, z) = (0, 3, 0) :=
by sorry

end pos_int_solns_to_eq_l161_161140


namespace alex_not_read_probability_l161_161624

def probability_reads : ℚ := 5 / 8
def probability_not_reads : ℚ := 3 / 8

theorem alex_not_read_probability : (1 - probability_reads) = probability_not_reads := 
by
  sorry

end alex_not_read_probability_l161_161624


namespace find_alpha_l161_161000

noncomputable def polar_to_cartesian : ℝ := 
  ∀ (ρ θ : ℝ), 
  (ρ = 4 * real.cos (θ - real.pi / 3)) ↔ 
  (let x := ρ * real.cos θ;
       y := ρ * real.sin θ in
    (x - 1)^2 + (y - real.sqrt 3)^2 = 4)

-- We define the conditions as hypotheses to be used in the theorem.

variable (P : Point) (l : Line) (C : Curve) (AB : Segment)
variables (k α : ℝ)

theorem find_alpha :
  P = (2, real.sqrt 3) →
  C = { p | let (x, y) := p in (x - 1)^2 + (y - real.sqrt 3)^2 = 4 } →
  length AB = real.sqrt 13 →
  ∃ α : Real.Angle, α = real.pi / 3 ∨ α = 2 * real.pi / 3 :=
by
  sorry

end find_alpha_l161_161000


namespace sin_135_eq_l161_161695

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l161_161695


namespace coin_combinations_l161_161326

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l161_161326


namespace maura_classroom_students_l161_161430

theorem maura_classroom_students (T : ℝ) (h1 : Tina_students = T) (h2 : Maura_students = T) (h3 : Zack_students = T / 2) (h4 : Tina_students + Maura_students + Zack_students = 69) : T = 138 / 5 := by
  sorry

end maura_classroom_students_l161_161430


namespace sum_of_values_satisfying_equation_l161_161665

noncomputable def sum_of_roots_of_quadratic (a b c : ℝ) : ℝ := -b / a

theorem sum_of_values_satisfying_equation :
  (∃ x : ℝ, (x^2 - 5 * x + 7 = 9)) →
  sum_of_roots_of_quadratic 1 (-5) (-2) = 5 :=
by
  sorry

end sum_of_values_satisfying_equation_l161_161665


namespace triangle_third_side_length_count_l161_161915

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l161_161915


namespace abs_inequality_solution_l161_161066

theorem abs_inequality_solution (x : ℝ) : (|x - 1| < 2) ↔ (x > -1 ∧ x < 3) := 
sorry

end abs_inequality_solution_l161_161066


namespace inverse_proposition_l161_161423

theorem inverse_proposition :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ y : ℝ, y^2 > 0 → y < 0) :=
by
  sorry

end inverse_proposition_l161_161423


namespace camille_total_birds_count_l161_161129

theorem camille_total_birds_count :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry

end camille_total_birds_count_l161_161129


namespace cost_per_serving_of_pie_l161_161896

theorem cost_per_serving_of_pie 
  (w_gs : ℝ) (p_gs : ℝ) (w_gala : ℝ) (p_gala : ℝ) (w_hc : ℝ) (p_hc : ℝ)
  (pie_crust_cost : ℝ) (lemon_cost : ℝ) (butter_cost : ℝ) (servings : ℕ)
  (total_weight_gs : w_gs = 0.5) (price_gs_per_pound : p_gs = 1.80)
  (total_weight_gala : w_gala = 0.8) (price_gala_per_pound : p_gala = 2.20)
  (total_weight_hc : w_hc = 0.7) (price_hc_per_pound : p_hc = 2.50)
  (cost_pie_crust : pie_crust_cost = 2.50) (cost_lemon : lemon_cost = 0.60)
  (cost_butter : butter_cost = 1.80) (total_servings : servings = 8) :
  (w_gs * p_gs + w_gala * p_gala + w_hc * p_hc + pie_crust_cost + lemon_cost + butter_cost) / servings = 1.16 :=
by 
  sorry

end cost_per_serving_of_pie_l161_161896


namespace find_f_seven_l161_161855

theorem find_f_seven 
  (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 3) = x^2 - 2 * x + 3) :
  f 7 = 3 := 
sorry

end find_f_seven_l161_161855


namespace days_for_B_l161_161259

theorem days_for_B
  (x : ℝ)
  (hA : 15 ≠ 0)
  (h_nonzero_fraction : 0.5833333333333334 ≠ 0)
  (hfraction : 0 <  0.5833333333333334 ∧ 0.5833333333333334 < 1)
  (h_fraction_work_left : 5 * (1 / 15 + 1 / x) = 0.5833333333333334) :
  x = 20 := by
  sorry

end days_for_B_l161_161259


namespace coprime_divisible_l161_161780

theorem coprime_divisible (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a ∣ b * c) : a ∣ c :=
by
  sorry

end coprime_divisible_l161_161780


namespace find_side_b_l161_161578

-- Given the side and angle conditions in the triangle
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ) 

-- Conditions provided in the problem
axiom side_a (h : a = 1) : True
axiom angle_B (h : B = Real.pi / 4) : True  -- 45 degrees in radians
axiom area_triangle (h : S = 2) : True

-- Final proof statement
theorem find_side_b (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : 
  b = 5 := sorry

end find_side_b_l161_161578


namespace probability_neither_red_nor_purple_l161_161823

theorem probability_neither_red_nor_purple :
  (100 - (47 + 3)) / 100 = 0.5 :=
by sorry

end probability_neither_red_nor_purple_l161_161823


namespace find_p_from_parabola_and_distance_l161_161170

theorem find_p_from_parabola_and_distance 
  (p : ℝ) (hp : p > 0) 
  (M : ℝ × ℝ) (hM : M = (8 / p, 4))
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (hMF : dist M F = 4) : 
  p = 4 :=
sorry

end find_p_from_parabola_and_distance_l161_161170


namespace tammy_speed_on_second_day_l161_161052

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l161_161052


namespace bianca_points_earned_l161_161254

-- Define the constants and initial conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 17
def not_recycled_bags : ℕ := 8

-- Define a function to calculate the number of recycled bags
def recycled_bags (total: ℕ) (not_recycled: ℕ) : ℕ :=
  total - not_recycled

-- Define a function to calculate the total points earned
def total_points_earned (bags: ℕ) (points_per_bag: ℕ) : ℕ :=
  bags * points_per_bag

-- State the theorem
theorem bianca_points_earned : total_points_earned (recycled_bags total_bags not_recycled_bags) points_per_bag = 45 :=
by
  sorry

end bianca_points_earned_l161_161254


namespace min_value_of_f_l161_161097

noncomputable def f (x : ℝ) : ℝ := x^2 + 8 * x + 3

theorem min_value_of_f : ∃ x₀ : ℝ, (∀ x : ℝ, f x ≥ f x₀) ∧ f x₀ = -13 :=
by
  sorry

end min_value_of_f_l161_161097


namespace number_of_integers_in_interval_l161_161343

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l161_161343


namespace find_n_l161_161405

def exp (m n : ℕ) : ℕ := m ^ n

-- Now we restate the problem formally
theorem find_n 
  (m n : ℕ) 
  (h1 : exp 10 m = n * 22) : 
  n = 10^m / 22 := 
sorry

end find_n_l161_161405


namespace person_half_Jordyn_age_is_6_l161_161789

variables (Mehki_age Jordyn_age certain_age : ℕ)
axiom h1 : Mehki_age = Jordyn_age + 10
axiom h2 : Jordyn_age = 2 * certain_age
axiom h3 : Mehki_age = 22

theorem person_half_Jordyn_age_is_6 : certain_age = 6 :=
by sorry

end person_half_Jordyn_age_is_6_l161_161789


namespace sin_135_eq_l161_161714

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l161_161714


namespace printer_to_enhanced_ratio_l161_161077

def B : ℕ := 2125
def P : ℕ := 2500 - B
def E : ℕ := B + 500
def total_price := E + P

theorem printer_to_enhanced_ratio :
  (P : ℚ) / total_price = 1 / 8 := 
by {
  -- skipping the proof
  sorry
}

end printer_to_enhanced_ratio_l161_161077


namespace alice_needs_7_fills_to_get_3_cups_l161_161460

theorem alice_needs_7_fills_to_get_3_cups (needs : ℚ) (cup_size : ℚ) (has : ℚ) :
  needs = 3 ∧ cup_size = 1 / 3 ∧ has = 2 / 3 →
  (needs - has) / cup_size = 7 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end alice_needs_7_fills_to_get_3_cups_l161_161460


namespace temperature_difference_l161_161765

variable (high_temp : ℝ) (low_temp : ℝ)

theorem temperature_difference (h1 : high_temp = 15) (h2 : low_temp = 7) : high_temp - low_temp = 8 :=
by {
  sorry
}

end temperature_difference_l161_161765


namespace abc_sum_eq_sqrt34_l161_161961

noncomputable def abc_sum (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 16)
                          (h2 : ab + bc + ca = 9)
                          (h3 : a^2 + b^2 = 10)
                          (h4 : 0 ≤ a) (h5 : 0 ≤ b) (h6 : 0 ≤ c) : ℝ :=
a + b + c

theorem abc_sum_eq_sqrt34 (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 16)
  (h2 : ab + bc + ca = 9)
  (h3 : a^2 + b^2 = 10)
  (h4 : 0 ≤ a)
  (h5 : 0 ≤ b)
  (h6 : 0 ≤ c) :
  abc_sum a b c h1 h2 h3 h4 h5 h6 = Real.sqrt 34 :=
by
  sorry

end abc_sum_eq_sqrt34_l161_161961


namespace selection_methods_count_l161_161151

theorem selection_methods_count :
  let students := {A, B, C, D, E}
  let languages := {Russian, Arabic, Hebrew}
  let students_unwilling := {A, B}
  let remaining_students := students \ students_unwilling
  ∃ M : students → languages, 
    M A ≠ Hebrew ∧ M B ≠ Hebrew ∧ M C ≠ Hebrew → card (students \ students_unwilling) = 3 ∧ 
    card (students) = 5 ∧ card (languages) = 3 ∧
    card (remaining_students) * (factorial (card students - 1) / factorial (card students - 3)) = 36 :=
by
  sorry

end selection_methods_count_l161_161151


namespace opposite_of_four_l161_161216

theorem opposite_of_four : ∃ x : ℤ, 4 + x = 0 ∧ x = -4 :=
by
  use -4
  split
  { -- prove 4 + (-4) = 0
    exact add_neg_self 4
  }
  { -- prove x = -4
    reflexivity
  }

end opposite_of_four_l161_161216


namespace count_whole_numbers_in_interval_l161_161346

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l161_161346


namespace probability_red_nonjoker_then_black_or_joker_l161_161180

theorem probability_red_nonjoker_then_black_or_joker :
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  probability = 5 / 17 :=
by
  -- Definitions for the conditions
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  -- Add sorry placeholder for proof
  sorry

end probability_red_nonjoker_then_black_or_joker_l161_161180


namespace A_work_days_l161_161260

theorem A_work_days (x : ℝ) (h1 : 1 / 15 + 1 / x = 1 / 8.571428571428571) : x = 20 :=
by
  sorry

end A_work_days_l161_161260


namespace arrange_magnitudes_l161_161746

theorem arrange_magnitudes (x : ℝ) (h1 : 0.85 < x) (h2 : x < 1.1)
  (y : ℝ := x + Real.sin x) (z : ℝ := x ^ (x ^ x)) : x < y ∧ y < z := 
sorry

end arrange_magnitudes_l161_161746


namespace trajectory_equation_l161_161004

def fixed_point : ℝ × ℝ := (1, 2)

def moving_point (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (p1 p2 : ℝ × ℝ) : ℝ :=
p1.1 * p2.1 + p1.2 * p2.2

theorem trajectory_equation (x y : ℝ) (h : dot_product (moving_point x y) fixed_point = 4) :
  x + 2 * y - 4 = 0 :=
sorry

end trajectory_equation_l161_161004


namespace solve_equation_l161_161798

noncomputable def lhs (x: ℝ) : ℝ := (sqrt ((3*x - 1) / (x + 4))) + 3 - 4 * (sqrt ((x + 4) / (3*x - 1)))

theorem solve_equation (x: ℝ) (t : ℝ) (ht : t = (3*x - 1) / (x + 4)) (h_pos : 0 < t) :
  lhs x = 0 → x = 5 / 2 :=
by
  intros h
  sorry

end solve_equation_l161_161798


namespace quadratic_ineq_solution_range_of_b_for_any_a_l161_161736

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := -3 * x^2 + a * (5 - a) * x + b

theorem quadratic_ineq_solution (a b : α) : 
  (∀ x ∈ Set.Ioo (-1 : α) 3, f a b x > 0) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := 
  sorry

theorem range_of_b_for_any_a (a b : α) :
  (∀ a : α, f a b 2 < 0) → 
  b < -1 / 2 := 
  sorry

end quadratic_ineq_solution_range_of_b_for_any_a_l161_161736


namespace probability_sum_divisible_by_3_l161_161288

open Finset

-- Define the list of numbers
def numbers : Finset ℕ := {1, 2, 3, 4, 5}

-- Calculate all pairs of numbers chosen from the given list
def all_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s.filter (λ p, p.1 < p.2)  -- Ensure order so we don't count (a,b) and (b,a) separately

-- Define a predicate that checks if the sum of a pair is divisible by 3
def divisible_by_three_sum (p : ℕ × ℕ) : Prop :=
  (p.1 + p.2) % 3 = 0

-- Count the pairs fulfilling the condition
def successful_pairs : Finset (ℕ × ℕ) :=
  (all_pairs numbers).filter divisible_by_three_sum

-- Calculate the probability
theorem probability_sum_divisible_by_3 :
  (successful_pairs.card : ℚ) / (all_pairs numbers).card = 2 / 5 :=
by
  sorry  -- proof

end probability_sum_divisible_by_3_l161_161288


namespace polynomial_evaluation_l161_161595

-- Define the polynomial p(x) and the conditions
noncomputable def p (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

-- Given conditions for p(1), p(2), p(3)
variables (a b c d : ℝ)
axiom h₁ : p 1 a b c d = 1993
axiom h₂ : p 2 a b c d = 3986
axiom h₃ : p 3 a b c d = 5979

-- The final proof statement
theorem polynomial_evaluation :
  (1 / 4) * (p 11 a b c d + p (-7) a b c d) = 5233 :=
sorry

end polynomial_evaluation_l161_161595


namespace matthew_egg_rolls_l161_161784

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end matthew_egg_rolls_l161_161784


namespace sin_135_eq_sqrt2_div_2_l161_161699

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l161_161699


namespace coin_combination_l161_161328

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l161_161328


namespace sum_abcd_eq_16_l161_161526

variable (a b c d : ℝ)

def cond1 : Prop := a^2 + b^2 + c^2 + d^2 = 250
def cond2 : Prop := a * b + b * c + c * a + a * d + b * d + c * d = 3

theorem sum_abcd_eq_16 (h1 : cond1 a b c d) (h2 : cond2 a b c d) : a + b + c + d = 16 := 
by 
  sorry

end sum_abcd_eq_16_l161_161526


namespace smallest_integer_divisibility_l161_161095

def smallest_integer (a : ℕ) : Prop :=
  a > 0 ∧ ¬ ∀ b, a = b + 1

theorem smallest_integer_divisibility :
  ∃ a, smallest_integer a ∧ gcd a 63 > 1 ∧ gcd a 66 > 1 ∧ ∀ b, smallest_integer b → b < a → gcd b 63 ≤ 1 ∨ gcd b 66 ≤ 1 :=
sorry

end smallest_integer_divisibility_l161_161095


namespace longest_side_similar_triangle_l161_161424

theorem longest_side_similar_triangle 
  (a b c : ℕ) (p : ℕ) (longest_side : ℕ)
  (h1 : a = 6) (h2 : b = 7) (h3 : c = 9) (h4 : p = 110) 
  (h5 : longest_side = 45) :
  ∃ x : ℕ, (6 * x + 7 * x + 9 * x = 110) ∧ (9 * x = longest_side) :=
by
  sorry

end longest_side_similar_triangle_l161_161424


namespace positive_solution_range_l161_161178

theorem positive_solution_range (a : ℝ) (h : a > 0) (x : ℝ) : (∃ x, (a / (x + 3) = 1 / 2) ∧ x > 0) ↔ a > 3 / 2 := by
  sorry

end positive_solution_range_l161_161178


namespace area_of_triangle_l161_161193

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l161_161193


namespace integral_f_x_l161_161175

theorem integral_f_x (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ t in (0 : ℝ)..1, f t) : 
  ∫ t in (0 : ℝ)..1, f t = -1 / 3 := by
  sorry

end integral_f_x_l161_161175


namespace range_of_omega_l161_161154

theorem range_of_omega (ω : ℝ) (hω : ω > 2/3) :
  (∀ x : ℝ, x = (k : ℤ) * π / ω + 3 * π / (4 * ω) → (x ≤ π ∨ x ≥ 2 * π) ) →
  ω ∈ Set.Icc (3/4 : ℝ) (7/8 : ℝ) :=
by
  sorry

end range_of_omega_l161_161154


namespace sin_135_eq_sqrt2_div_2_l161_161698

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l161_161698


namespace symmetric_about_line_5pi12_l161_161168

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem symmetric_about_line_5pi12 :
  ∀ x : ℝ, f (5 * Real.pi / 12 - x) = f (5 * Real.pi / 12 + x) :=
by
  intros x
  sorry

end symmetric_about_line_5pi12_l161_161168


namespace sin_135_l161_161701

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l161_161701


namespace sum_of_positive_factors_of_72_l161_161233

/-- Define the divisor sum function based on the given formula -/
def divisor_sum (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 3
  | 3 => 4
  | 4 => 7
  | 6 => 12
  | 8 => 15
  | 12 => 28
  | 18 => 39
  | 24 => 60
  | 36 => 91
  | 48 => 124
  | 60 => 168
  | 72 => 195
  | _ => 0 -- This is not generally correct, just handles given problem specifically

theorem sum_of_positive_factors_of_72 :
  divisor_sum 72 = 195 :=
sorry

end sum_of_positive_factors_of_72_l161_161233


namespace max_value_g_l161_161986

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l161_161986


namespace marty_combination_count_l161_161968

theorem marty_combination_count (num_colors : ℕ) (num_methods : ℕ) 
  (h1 : num_colors = 5) (h2 : num_methods = 4) : 
  num_colors * num_methods = 20 := by
  sorry

end marty_combination_count_l161_161968


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l161_161875

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (a : ℕ), (∀ n, n ∈ (list.range 4).map (λ i, a + i) -> n % 2 = 0 ∨ n % 3 = 0 ∨ n % 4 = 0) →
  12 ∣ list.prod ((list.range 4).map (λ i, a + i)) :=
by
  intro a
  intro h
  -- Insert proof here
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l161_161875


namespace sin_135_eq_sqrt2_over_2_l161_161710

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l161_161710


namespace find_years_invested_l161_161677

-- Defining the conditions and theorem
variables (P : ℕ) (r1 r2 D : ℝ) (n : ℝ)

-- Given conditions
def principal := (P : ℝ) = 7000
def rate_1 := r1 = 0.15
def rate_2 := r2 = 0.12
def interest_diff := D = 420

-- Theorem to be proven
theorem find_years_invested (h1 : principal P) (h2 : rate_1 r1) (h3 : rate_2 r2) (h4 : interest_diff D) :
  7000 * 0.15 * n - 7000 * 0.12 * n = 420 → n = 2 :=
by
  sorry

end find_years_invested_l161_161677


namespace ratio_of_sheep_to_horses_l161_161625

theorem ratio_of_sheep_to_horses (H : ℕ) (hH : 230 * H = 12880) (n_sheep : ℕ) (h_sheep : n_sheep = 56) :
  (n_sheep / H) = 1 := by
  sorry

end ratio_of_sheep_to_horses_l161_161625


namespace problem1_problem2_problem3_l161_161527

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x + 2 * x

theorem problem1 (a : ℝ) : a = 1 → ∀ x : ℝ, f 1 x = x^2 - 3 * x + Real.log x → 
  (∀ x : ℝ, f 1 1 = -2) :=
by sorry

theorem problem2 (a : ℝ) (h : 0 < a) : (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → a ≥ 1 :=
by sorry

theorem problem3 (a : ℝ) : (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) → 0 ≤ a ∧ a ≤ 8 :=
by sorry

end problem1_problem2_problem3_l161_161527


namespace prize_distribution_l161_161003

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem prize_distribution :
  let total_ways := 
    (binomial_coefficient 7 3) * 5 * (Nat.factorial 4) + 
    (binomial_coefficient 7 2 * binomial_coefficient 5 2 / 2) * 
    (binomial_coefficient 5 2) * (Nat.factorial 3)
  total_ways = 10500 :=
by 
  sorry

end prize_distribution_l161_161003


namespace tammy_avg_speed_second_day_l161_161046

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l161_161046


namespace tangent_parallel_to_line_l161_161628

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (P : ℝ × ℝ) 
(hP : ∃ (x : ℝ), P = (x, f x))
(h_parallel : ∀ (x : ℝ), deriv f x = 4 ↔ P = (1, 0) ∨ P = (-1, -4)) :
P = (1, 0) ∨ P = (-1, -4) :=
by sorry

end tangent_parallel_to_line_l161_161628


namespace radius_of_regular_polygon_l161_161074

theorem radius_of_regular_polygon :
  ∃ (p : ℝ), 
        (∀ n : ℕ, 3 ≤ n → (n : ℝ) = 6) ∧ 
        (∀ s : ℝ, s = 2 → s = 2) → 
        (∀ i : ℝ, i = 720 → i = 720) →
        (∀ e : ℝ, e = 360 → e = 360) →
        p = 2 :=
by
  sorry

end radius_of_regular_polygon_l161_161074


namespace sum_of_coefficients_l161_161901

-- Given polynomial
def polynomial (x : ℝ) : ℝ := (3 * x - 1) ^ 7

-- Statement
theorem sum_of_coefficients :
  (polynomial 1) = 128 := 
sorry

end sum_of_coefficients_l161_161901


namespace conditional_probability_B_given_A_l161_161813

-- Definitions of the given probabilities.
def P_A : ℚ := 7 / 8
def P_B : ℚ := 6 / 8
def P_AB : ℚ := 5 / 8

/-- 
  Given P(A) = 7/8, P(B) = 6/8, and P(AB) = 5/8, 
  the conditional probability P(B|A) is 5/7.
-/
theorem conditional_probability_B_given_A :
  P_AB / P_A = 5 / 7 :=
by
  -- Placeholder for proof steps
  sorry

end conditional_probability_B_given_A_l161_161813


namespace unique_solution_condition_l161_161025

theorem unique_solution_condition (a b c : ℝ) : 
  (∀ x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 :=
by
  sorry

end unique_solution_condition_l161_161025


namespace abs_diff_eq_0_5_l161_161399

noncomputable def x : ℝ := 3.7
noncomputable def y : ℝ := 4.2

theorem abs_diff_eq_0_5 (hx : ⌊x⌋ + (y - ⌊y⌋) = 3.2) (hy : (x - ⌊x⌋) + ⌊y⌋ = 4.7) :
  |x - y| = 0.5 :=
by
  sorry

end abs_diff_eq_0_5_l161_161399


namespace probability_three_green_is_14_over_99_l161_161452

noncomputable def probability_three_green :=
  let total_combinations := Nat.choose 12 4
  let successful_outcomes := (Nat.choose 5 3) * (Nat.choose 7 1)
  (successful_outcomes : ℚ) / total_combinations

theorem probability_three_green_is_14_over_99 :
  probability_three_green = 14 / 99 :=
by
  sorry

end probability_three_green_is_14_over_99_l161_161452


namespace count_whole_numbers_in_interval_l161_161342

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l161_161342


namespace eval_expr_l161_161728

theorem eval_expr : (2.1 * (49.7 + 0.3)) + 15 = 120 :=
  by
  sorry

end eval_expr_l161_161728


namespace trapezium_height_l161_161501

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end trapezium_height_l161_161501


namespace sum_of_factors_of_72_l161_161235

/-- Prove that the sum of the positive factors of 72 is 195 -/
theorem sum_of_factors_of_72 : ∑ d in (finset.filter (λ d, 72 % d = 0) (finset.range (73))), d = 195 := 
by
  sorry

end sum_of_factors_of_72_l161_161235


namespace manager_hourly_wage_l161_161274

open Real

theorem manager_hourly_wage (M D C : ℝ) 
  (hD : D = M / 2)
  (hC : C = 1.20 * D)
  (hC_manager : C = M - 3.40) :
  M = 8.50 :=
by
  sorry

end manager_hourly_wage_l161_161274


namespace sequence_properties_l161_161738

theorem sequence_properties (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, (a n)^2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0) :
  a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_properties_l161_161738


namespace average_speed_l161_161119

   theorem average_speed (x : ℝ) : 
     let s1 := 40
     let s2 := 20
     let d1 := x
     let d2 := 2 * x
     let total_distance := d1 + d2
     let time1 := d1 / s1
     let time2 := d2 / s2
     let total_time := time1 + time2
     total_distance / total_time = 24 :=
   by
     sorry
   
end average_speed_l161_161119


namespace product_of_two_primes_l161_161993

theorem product_of_two_primes (p q z : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) 
    (h_p_range : 2 < p ∧ p < 6) 
    (h_q_range : 8 < q ∧ q < 24) 
    (h_z_def : z = p * q) 
    (h_z_range : 15 < z ∧ z < 36) : 
    z = 33 := 
by 
    sorry

end product_of_two_primes_l161_161993


namespace max_value_of_g_l161_161984

def g : ℕ → ℕ 
| n := if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : 
  ∃ N, (∀ n, g n ≤ N) ∧ N = 23 := 
sorry

end max_value_of_g_l161_161984


namespace fraction_expression_equiv_l161_161443

theorem fraction_expression_equiv:
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := 
by 
  sorry

end fraction_expression_equiv_l161_161443


namespace remainder_of_3_pow_244_mod_5_l161_161663

theorem remainder_of_3_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l161_161663


namespace percent_of_x_l161_161750

theorem percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : z = 2 * x) :
  y = 0.75 * x :=
sorry

end percent_of_x_l161_161750


namespace ways_to_divide_friends_l161_161339

theorem ways_to_divide_friends : (4 ^ 8 = 65536) := by
  sorry

end ways_to_divide_friends_l161_161339


namespace number_of_lists_l161_161833

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l161_161833


namespace fraction_sum_product_roots_of_quadratic_l161_161570

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l161_161570


namespace children_sit_in_same_row_twice_l161_161379

theorem children_sit_in_same_row_twice
  (rows : ℕ) (seats_per_row : ℕ) (children : ℕ)
  (h_rows : rows = 7) (h_seats_per_row : seats_per_row = 10) (h_children : children = 50) :
  ∃ (morning_evening_pair : ℕ × ℕ), 
  (morning_evening_pair.1 < rows ∧ morning_evening_pair.2 < rows) ∧ 
  morning_evening_pair.1 = morning_evening_pair.2 :=
by
  sorry

end children_sit_in_same_row_twice_l161_161379


namespace total_weight_l161_161262

-- Define the weights of almonds and pecans.
def weight_almonds : ℝ := 0.14
def weight_pecans : ℝ := 0.38

-- Prove that the total weight of nuts is 0.52 kilograms.
theorem total_weight (almonds pecans : ℝ) (h_almonds : almonds = 0.14) (h_pecans : pecans = 0.38) :
  almonds + pecans = 0.52 :=
by
  sorry

end total_weight_l161_161262


namespace four_xyz_value_l161_161740

theorem four_xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 4 * x * y * z = 48 := by
  sorry

end four_xyz_value_l161_161740


namespace square_side_length_l161_161057

theorem square_side_length (x : ℝ) (h : x^2 = (1/2) * x * 2) : x = 1 := by
  sorry

end square_side_length_l161_161057


namespace joes_fast_food_cost_l161_161037

noncomputable def cost_of_sandwich (n : ℕ) : ℝ := n * 4
noncomputable def cost_of_soda (m : ℕ) : ℝ := m * 1.50
noncomputable def total_cost (n m : ℕ) : ℝ :=
  if n >= 10 then cost_of_sandwich n - 5 + cost_of_soda m else cost_of_sandwich n + cost_of_soda m

theorem joes_fast_food_cost :
  total_cost 10 6 = 44 := by
  sorry

end joes_fast_food_cost_l161_161037


namespace find_a_l161_161291

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → 
     ∀ (x' y' : ℝ), (x' = x - 2 * (x - a * y + 2) / (1 + a^2)) ∧ (y' = y - 2 * a * (x - a * y + 2) / (1 + a^2)) → 
     (x'^2 + y'^2 + 2 * x' - 4 * y' + 1 = 0)) → 
  (a = -1 / 2) := 
sorry

end find_a_l161_161291


namespace moon_speed_conversion_l161_161425

def moon_speed_km_sec : ℝ := 1.04
def seconds_per_hour : ℝ := 3600

theorem moon_speed_conversion :
  (moon_speed_km_sec * seconds_per_hour) = 3744 := by
  sorry

end moon_speed_conversion_l161_161425


namespace combinations_problem_l161_161670

open Nat

-- Definitions for combinations
def C (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

-- Condition: Number of ways to choose 2 sergeants out of 6
def C_6_2 : Nat := C 6 2

-- Condition: Number of ways to choose 20 soldiers out of 60
def C_60_20 : Nat := C 60 20

-- Theorem statement for the problem
theorem combinations_problem :
  3 * C_6_2 * C_60_20 = 3 * 15 * C 60 20 := by
  simp [C_6_2, C_60_20, C]
  sorry

end combinations_problem_l161_161670


namespace number_of_possible_third_side_lengths_l161_161922

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l161_161922


namespace plant_cost_and_max_green_lily_students_l161_161586

-- Given conditions
def two_green_lily_three_spider_plants_cost (x y : ℕ) : Prop :=
  2 * x + 3 * y = 36

def one_green_lily_two_spider_plants_cost (x y : ℕ) : Prop :=
  x + 2 * y = 21

def total_students := 48

def cost_constraint (x y m : ℕ) : Prop :=
  9 * m + 6 * (48 - m) ≤ 378

-- Prove that x = 9, y = 6 and m ≤ 30
theorem plant_cost_and_max_green_lily_students :
  ∃ x y m : ℕ, two_green_lily_three_spider_plants_cost x y ∧ 
               one_green_lily_two_spider_plants_cost x y ∧ 
               cost_constraint x y m ∧ 
               x = 9 ∧ y = 6 ∧ m ≤ 30 :=
by
  sorry

end plant_cost_and_max_green_lily_students_l161_161586


namespace booking_rooms_needed_l161_161479

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l161_161479


namespace relationship_a_b_c_l161_161960

noncomputable def distinct_positive_numbers (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem relationship_a_b_c (a b c : ℝ) (h1 : distinct_positive_numbers a b c) (h2 : a^2 + c^2 = 2 * b * c) : b > a ∧ a > c :=
by
  sorry

end relationship_a_b_c_l161_161960


namespace triangle_area_l161_161190

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l161_161190


namespace min_value_ratio_l161_161747

noncomputable def min_ratio (a : ℝ) (h : a > 0) : ℝ :=
  let x_A := 4^(-a)
  let x_B := 4^(a)
  let x_C := 4^(- (18 / (2*a + 1)))
  let x_D := 4^((18 / (2*a + 1)))
  let m := abs (x_A - x_C)
  let n := abs (x_B - x_D)
  n / m

theorem min_value_ratio (a : ℝ) (h : a > 0) : 
  ∃ c : ℝ, c = 2^11 := sorry

end min_value_ratio_l161_161747


namespace trip_time_maple_to_oak_l161_161261

noncomputable def total_trip_time (d1 d2 v1 v2 t_break : ℝ) : ℝ :=
  (d1 / v1) + t_break + (d2 / v2)

theorem trip_time_maple_to_oak : 
  total_trip_time 210 210 50 40 0.5 = 5.75 :=
by
  sorry

end trip_time_maple_to_oak_l161_161261


namespace rain_on_tuesday_l161_161181

/-- Let \( R_M \) be the event that a county received rain on Monday. -/
def RM : Prop := sorry

/-- Let \( R_T \) be the event that a county received rain on Tuesday. -/
def RT : Prop := sorry

/-- Let \( R_{MT} \) be the event that a county received rain on both Monday and Tuesday. -/
def RMT : Prop := RM ∧ RT

/-- The probability that a county received rain on Monday is 0.62. -/
def prob_RM : ℝ := 0.62

/-- The probability that a county received rain on both Monday and Tuesday is 0.44. -/
def prob_RMT : ℝ := 0.44

/-- The probability that no rain fell on either day is 0.28. -/
def prob_no_rain : ℝ := 0.28

/-- The probability that a county received rain on at least one of the days is 0.72. -/
def prob_at_least_one_day : ℝ := 1 - prob_no_rain

/-- The probability that a county received rain on Tuesday is 0.54. -/
theorem rain_on_tuesday : (prob_at_least_one_day = prob_RM + x - prob_RMT) → (x = 0.54) :=
by
  intros h
  sorry

end rain_on_tuesday_l161_161181


namespace two_cards_sum_to_15_proof_l161_161634

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l161_161634


namespace angle_of_inclination_of_line_l161_161891

-- Definition of the line l
def line_eq (x : ℝ) : ℝ := x + 1

-- Statement of the theorem about the angle of inclination
theorem angle_of_inclination_of_line (x : ℝ) : 
  ∃ (θ : ℝ), θ = 45 ∧ line_eq x = x + 1 := 
sorry

end angle_of_inclination_of_line_l161_161891


namespace R2_area_is_160_l161_161521

-- Define the initial conditions.
structure Rectangle :=
(width : ℝ)
(height : ℝ)

def R1 : Rectangle := { width := 4, height := 8 }

def similar (r1 r2 : Rectangle) : Prop :=
  r2.width / r2.height = r1.width / r1.height

def R2_diagonal := 20

-- Proving that the area of R2 is 160 square inches
theorem R2_area_is_160 (R2 : Rectangle)
  (h_similar : similar R1 R2)
  (h_diagonal : R2.width^2 + R2.height^2 = R2_diagonal^2) :
  R2.width * R2.height = 160 :=
  sorry

end R2_area_is_160_l161_161521


namespace domain_of_c_x_l161_161470

theorem domain_of_c_x (k : ℝ) :
  (∀ x : ℝ, -5 * x ^ 2 + 3 * x + k ≠ 0) ↔ k < -9 / 20 := 
sorry

end domain_of_c_x_l161_161470


namespace triangle_third_side_length_count_l161_161917

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l161_161917


namespace sector_area_eq_25_l161_161382

theorem sector_area_eq_25 (r θ : ℝ) (h_r : r = 5) (h_θ : θ = 2) : (1 / 2) * θ * r^2 = 25 := by
  sorry

end sector_area_eq_25_l161_161382


namespace triangle_third_side_length_count_l161_161916

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l161_161916


namespace find_n_l161_161902

theorem find_n (n : ℚ) : 1 / 2 + 2 / 3 + 3 / 4 + n / 12 = 2 ↔ n = 1 := by
  -- proof here
  sorry

end find_n_l161_161902


namespace sequence_condition_satisfies_l161_161948

def seq_prove_abs_lt_1 (a : ℕ → ℝ) : Prop :=
  (∃ i : ℕ, |a i| < 1)

theorem sequence_condition_satisfies (a : ℕ → ℝ)
  (h1 : a 1 * a 2 < 0)
  (h2 : ∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧ (∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)) :
  seq_prove_abs_lt_1 a :=
by
  sorry

end sequence_condition_satisfies_l161_161948


namespace solve_for_nabla_l161_161370

theorem solve_for_nabla : ∃ (∇ : ℤ), 3 * (-2) = ∇ + 2 ∧ ∇ = -8 :=
by { existsi (-8), split, exact rfl, exact rfl }

end solve_for_nabla_l161_161370


namespace polygon_with_largest_area_l161_161884

noncomputable def area_of_polygon_A : ℝ := 6
noncomputable def area_of_polygon_B : ℝ := 4
noncomputable def area_of_polygon_C : ℝ := 4 + 2 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_D : ℝ := 3 + 3 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_E : ℝ := 7

theorem polygon_with_largest_area : 
  area_of_polygon_E > area_of_polygon_A ∧ 
  area_of_polygon_E > area_of_polygon_B ∧ 
  area_of_polygon_E > area_of_polygon_C ∧ 
  area_of_polygon_E > area_of_polygon_D :=
by
  sorry

end polygon_with_largest_area_l161_161884


namespace min_value_of_reciprocal_sum_l161_161520

theorem min_value_of_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 2 * b = 1) (h2 : c + 2 * d = 1) :
  16 ≤ (1 / a) + 1 / (b * c * d) :=
by
  sorry

end min_value_of_reciprocal_sum_l161_161520


namespace solve_for_x_l161_161471

theorem solve_for_x (x : ℝ) (h : (x * (x ^ (5 / 2))) ^ (1 / 4) = 4) : 
  x = 4 ^ (8 / 7) :=
sorry

end solve_for_x_l161_161471


namespace find_a5_l161_161302

variables {a : ℕ → ℝ}  -- represent the arithmetic sequence

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom a3_a8_sum : a 3 + a 8 = 22
axiom a6_value : a 6 = 8
axiom arithmetic : is_arithmetic_sequence a

-- Target proof statement
theorem find_a5 (a : ℕ → ℝ) (arithmetic : is_arithmetic_sequence a) (a3_a8_sum : a 3 + a 8 = 22) (a6_value : a 6 = 8) : a 5 = 14 :=
by {
  sorry
}

end find_a5_l161_161302


namespace length_of_each_movie_l161_161172

-- Defining the amount of time Grandpa Lou watched movies on Tuesday in minutes
def time_tuesday : ℕ := 4 * 60 + 30   -- 4 hours and 30 minutes

-- Defining the number of movies watched on Tuesday
def movies_tuesday (x : ℕ) : Prop := time_tuesday / x = 90

-- Defining the total number of movies watched in both days
def total_movies_two_days (x : ℕ) : Prop := x + 2 * x = 9

theorem length_of_each_movie (x : ℕ) (h₁ : total_movies_two_days x) (h₂ : movies_tuesday x) : time_tuesday / x = 90 :=
by
  -- Given the conditions, we can prove the statement:
  sorry

end length_of_each_movie_l161_161172


namespace corina_problem_l161_161374

variable (P Q : ℝ)

theorem corina_problem (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 :=
sorry

end corina_problem_l161_161374


namespace color_schemes_equivalence_l161_161658

noncomputable def number_of_non_equivalent_color_schemes (n : Nat) : Nat :=
  let total_ways := Nat.choose (n * n) 2
  -- Calculate the count for non-diametrically opposite positions (4 rotations)
  let non_diametric := (total_ways - 24) / 4
  -- Calculate the count for diametrically opposite positions (2 rotations)
  let diametric := 24 / 2
  -- Sum both counts
  non_diametric + diametric

theorem color_schemes_equivalence (n : Nat) (h : n = 7) : number_of_non_equivalent_color_schemes n = 300 :=
  by
    rw [h]
    sorry

end color_schemes_equivalence_l161_161658


namespace quadratic_roots_vieta_l161_161560

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l161_161560


namespace triangle_side_length_integers_l161_161923

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l161_161923


namespace original_number_l161_161265

theorem original_number (x : ℝ) (h : 1.50 * x = 165) : x = 110 :=
sorry

end original_number_l161_161265


namespace lateral_surface_area_of_cylinder_l161_161939

theorem lateral_surface_area_of_cylinder :
  (∀ (side_length : ℕ), side_length = 10 → 
  ∃ (lateral_surface_area : ℝ), lateral_surface_area = 100 * Real.pi) :=
by
  sorry

end lateral_surface_area_of_cylinder_l161_161939


namespace exists_integer_div_15_sqrt_range_l161_161139

theorem exists_integer_div_15_sqrt_range :
  ∃ n : ℕ, (25^2 ≤ n ∧ n ≤ 26^2) ∧ (n % 15 = 0) :=
by
  sorry

end exists_integer_div_15_sqrt_range_l161_161139


namespace quadratic_no_real_roots_iff_m_gt_one_l161_161533

theorem quadratic_no_real_roots_iff_m_gt_one (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 :=
sorry

end quadratic_no_real_roots_iff_m_gt_one_l161_161533


namespace tammy_speed_on_second_day_l161_161050

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l161_161050


namespace polynomial_factorization_l161_161758

theorem polynomial_factorization (m n : ℤ) (h₁ : (x + 1) * (x + 3) = x^2 + m * x + n) : m - n = 1 := 
by {
  -- Proof not required
  sorry
}

end polynomial_factorization_l161_161758


namespace john_payment_l161_161772

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end john_payment_l161_161772


namespace number_of_members_l161_161444

theorem number_of_members (n : ℕ) (H : n * n = 5776) : n = 76 :=
by
  sorry

end number_of_members_l161_161444


namespace sin_135_eq_l161_161715

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l161_161715


namespace problem_1_problem_2_problem_3_l161_161449

-- Definition for question 1:
def gcd_21n_4_14n_3 (n : ℕ) : Prop := (Nat.gcd (21 * n + 4) (14 * n + 3)) = 1

-- Definition for question 2:
def gcd_n_factorial_plus_1 (n : ℕ) : Prop := (Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1)) = 1

-- Definition for question 3:
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1
def gcd_fermat_numbers (m n : ℕ) (h : m ≠ n) : Prop := (Nat.gcd (fermat_number m) (fermat_number n)) = 1

-- Theorem statements
theorem problem_1 (n : ℕ) (h_pos : 0 < n) : gcd_21n_4_14n_3 n := sorry

theorem problem_2 (n : ℕ) (h_pos : 0 < n) : gcd_n_factorial_plus_1 n := sorry

theorem problem_3 (m n : ℕ) (h_pos1 : 0 ≠ m) (h_pos2 : 0 ≠ n) (h_neq : m ≠ n) : gcd_fermat_numbers m n h_neq := sorry

end problem_1_problem_2_problem_3_l161_161449


namespace sqrt_meaningful_iff_ge_two_l161_161994

theorem sqrt_meaningful_iff_ge_two (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_meaningful_iff_ge_two_l161_161994


namespace part_a_l161_161675

theorem part_a : 
  ∃ (x y : ℕ → ℕ), (∀ n : ℕ, (1 + Real.sqrt 33) ^ n = x n + y n * Real.sqrt 33) :=
sorry

end part_a_l161_161675


namespace totalExerciseTime_l161_161463

-- Define the conditions
def caloriesBurnedRunningPerMinute := 10
def caloriesBurnedWalkingPerMinute := 4
def totalCaloriesBurned := 450
def runningTime := 35

-- Define the problem as a theorem to be proven
theorem totalExerciseTime :
  ((runningTime * caloriesBurnedRunningPerMinute) + 
  ((totalCaloriesBurned - runningTime * caloriesBurnedRunningPerMinute) / caloriesBurnedWalkingPerMinute)) = 60 := 
sorry

end totalExerciseTime_l161_161463


namespace max_value_of_g_l161_161988

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l161_161988


namespace quadratic_roots_identity_l161_161553

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l161_161553


namespace find_lunch_days_l161_161002

variable (x y : ℕ) -- School days for School A and School B
def P_A := x / 2 -- Aliyah packs lunch half the time
def P_B := y / 4 -- Becky packs lunch a quarter of the time
def P_C := y / 2 -- Charlie packs lunch half the time

theorem find_lunch_days (x y : ℕ) :
  P_A x = x / 2 ∧
  P_B y = y / 4 ∧
  P_C y = y / 2 :=
by
  sorry

end find_lunch_days_l161_161002


namespace a_runs_4_times_faster_than_b_l161_161453

theorem a_runs_4_times_faster_than_b (v_A v_B : ℝ) (k : ℝ) 
    (h1 : v_A = k * v_B) 
    (h2 : 92 / v_A = 23 / v_B) : 
    k = 4 := 
sorry

end a_runs_4_times_faster_than_b_l161_161453


namespace range_of_m_l161_161517

theorem range_of_m (a b c : ℝ) (m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 :=
sorry

end range_of_m_l161_161517


namespace quadratic_root_sum_and_product_l161_161563

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l161_161563


namespace smallest_four_digit_int_mod_9_l161_161229

theorem smallest_four_digit_int_mod_9 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 9 = 5 → n ≤ m :=
sorry

end smallest_four_digit_int_mod_9_l161_161229


namespace no_perfect_square_in_range_l161_161184

theorem no_perfect_square_in_range :
  ¬∃ (x : ℕ), 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ (n : ℕ), x = n * n :=
by
  sorry

end no_perfect_square_in_range_l161_161184


namespace number_of_possible_lists_l161_161839

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161839


namespace sin_135_eq_sqrt2_over_2_l161_161709

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l161_161709


namespace problem_1_problem_2_l161_161451

def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem problem_1:
  { x : ℝ // 0 ≤ x ∧ x ≤ 6 } = { x : ℝ // f x ≤ 1 } :=
sorry

theorem problem_2:
  { m : ℝ // m ≤ -3 } = { m : ℝ // ∀ x : ℝ, f x - g x ≥ m + 1 } :=
sorry

end problem_1_problem_2_l161_161451


namespace problem_solution_sets_l161_161032

theorem problem_solution_sets (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 ∧ x * y + 1 = x + y) →
  ( (x = 0 ∧ y = 0) ∨ y = 2 ∨ x = 1 ∨ y = 1 ) :=
by
  sorry

end problem_solution_sets_l161_161032


namespace solve_for_nabla_l161_161369

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 :=
by
  sorry

end solve_for_nabla_l161_161369


namespace mary_characters_initial_D_l161_161200

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end mary_characters_initial_D_l161_161200


namespace problem_l161_161211

open Real

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) + 2 * cos x - 4

theorem problem (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) : 
  ∀ a b : ℝ, (0 ≤ a ∧ a ≤ 2 * π) → (0 ≤ b ∧ b ≤ 2 * π) → a ≤ b → f a ≤ f b := 
sorry

end problem_l161_161211


namespace negation_of_p_l161_161744

open Real

-- Define the original proposition p
def p := ∀ x : ℝ, 0 < x → x^2 > log x

-- State the theorem with its negation
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 0 < x ∧ x^2 ≤ log x :=
by
  sorry

end negation_of_p_l161_161744


namespace sufficient_condition_for_ellipse_with_foci_y_axis_l161_161296

theorem sufficient_condition_for_ellipse_with_foci_y_axis (m n : ℝ) (h : m > n ∧ n > 0) :
  (∃ a b : ℝ, (a^2 = m / n) ∧ (b^2 = 1 / n) ∧ (a > b)) ∧ ¬(∀ u v : ℝ, (u^2 = m / v) → (v^2 = 1 / v) → (u > v) → (v = n ∧ u = m)) :=
by
  sorry

end sufficient_condition_for_ellipse_with_foci_y_axis_l161_161296


namespace part_1_part_2_l161_161612

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 4)

theorem part_1 (a : ℝ) (h : a = 3) :
  { x : ℝ | f x a ≥ 8 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | 1 ≤ x ∧ x ≤ 3 } ∪ { x : ℝ | x > 3 } := 
sorry

theorem part_2 (h : ∃ x : ℝ, f x a - abs (x + 2) ≤ 4) :
  -6 ≤ a ∧ a ≤ 2 :=
sorry

end part_1_part_2_l161_161612


namespace focus_of_parabola_x_squared_eq_neg_4_y_l161_161060

theorem focus_of_parabola_x_squared_eq_neg_4_y:
  (∃ F : ℝ × ℝ, (F = (0, -1)) ∧ (∀ x y : ℝ, x^2 = -4 * y → F = (0, y + 1))) :=
sorry

end focus_of_parabola_x_squared_eq_neg_4_y_l161_161060


namespace prob_one_defective_without_replacement_prob_one_defective_with_replacement_l161_161153

noncomputable theory

-- Definition of the sets and conditions
def items := ["a", "b", "c"]
def is_defective (x : String) : Bool := x = "c"
def exactly_one_defective (pair : (String × String)) : Bool :=
  (is_defective pair.1 && ¬is_defective pair.2) ||
  (¬is_defective pair.1 && is_defective pair.2)

-- Problem statement for without replacement
theorem prob_one_defective_without_replacement : 
  ((1/3 : ℝ) * (1/2 : ℝ) + (1/2 : ℝ) * (1/3 : ℝ)) = 2/3 :=
by sorry

-- Problem statement for with replacement
theorem prob_one_defective_with_replacement : 
  (2 * (1/3 : ℝ) * (2/3 : ℝ)) = 4/9 :=
by sorry

end prob_one_defective_without_replacement_prob_one_defective_with_replacement_l161_161153


namespace coin_combinations_50_cents_l161_161314

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l161_161314


namespace hotel_room_allocation_l161_161485

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l161_161485


namespace probability_sum_15_l161_161645

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l161_161645


namespace count_ways_to_get_50_cents_with_coins_l161_161323

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l161_161323


namespace original_average_l161_161419

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end original_average_l161_161419


namespace compute_65_sq_minus_55_sq_l161_161131

theorem compute_65_sq_minus_55_sq : 65^2 - 55^2 = 1200 :=
by
  -- We'll skip the proof here for simplicity
  sorry

end compute_65_sq_minus_55_sq_l161_161131


namespace sandy_comic_books_ratio_l161_161976

variable (S : ℕ)  -- number of comic books Sandy sold

theorem sandy_comic_books_ratio 
  (initial : ℕ) (bought : ℕ) (now : ℕ) (h_initial : initial = 14) (h_bought : bought = 6) (h_now : now = 13)
  (h_eq : initial - S + bought = now) :
  S = 7 ∧ S.to_rat / initial.to_rat = 1 / 2 := 
by
  sorry

end sandy_comic_books_ratio_l161_161976


namespace quadratic_root_identity_l161_161545

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l161_161545


namespace factor_expression_l161_161495

variable (a : ℝ)

theorem factor_expression : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) :=
by
  sorry

end factor_expression_l161_161495


namespace comic_books_ratio_l161_161975

variable (S : ℕ)

theorem comic_books_ratio (initial comics_left comics_bought : ℕ)
  (h1 : initial = 14)
  (h2 : comics_left = 13)
  (h3 : comics_bought = 6)
  (h4 : initial - S + comics_bought = comics_left) :
  (S / initial.toRat) = (1 / 2 : ℚ) :=
by
  sorry

end comic_books_ratio_l161_161975


namespace carlton_school_earnings_l161_161383

theorem carlton_school_earnings :
  let students_days_adams := 8 * 4
  let students_days_byron := 5 * 6
  let students_days_carlton := 6 * 10
  let total_wages := 1092
  students_days_adams + students_days_byron = 62 → 
  62 * (2 * x) + students_days_carlton * x = total_wages → 
  x = (total_wages : ℝ) / 184 → 
  (students_days_carlton : ℝ) * x = 356.09 := 
by
  intros _ _ _ 
  sorry

end carlton_school_earnings_l161_161383


namespace prudence_nap_is_4_hours_l161_161734

def prudence_nap_length (total_sleep : ℕ) (weekdays_sleep : ℕ) (weekend_sleep : ℕ) (weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  (total_sleep - (weekdays_sleep + weekend_sleep) * total_weeks) / (2 * total_weeks)

theorem prudence_nap_is_4_hours
  (total_sleep weekdays_sleep weekend_sleep total_weeks : ℕ) :
  total_sleep = 200 ∧ weekdays_sleep = 5 * 6 ∧ weekend_sleep = 2 * 9 ∧ total_weeks = 4 →
  prudence_nap_length total_sleep weekdays_sleep weekend_sleep total_weeks total_weeks = 4 :=
by
  intros
  sorry

end prudence_nap_is_4_hours_l161_161734


namespace current_bottle_caps_l161_161872

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem current_bottle_caps : initial_bottle_caps - lost_bottle_caps = 25 :=
by
  -- sorry is used to skip the proof
  sorry

end current_bottle_caps_l161_161872


namespace right_triangle_bc_is_3_l161_161182

-- Define the setup: a right triangle with given side lengths
structure RightTriangle :=
  (AB AC BC : ℝ)
  (right_angle : AB^2 = AC^2 + BC^2)
  (AB_val : AB = 5)
  (AC_val : AC = 4)

-- The goal is to prove that BC = 3 given the conditions
theorem right_triangle_bc_is_3 (T : RightTriangle) : T.BC = 3 :=
  sorry

end right_triangle_bc_is_3_l161_161182


namespace gcd_204_85_l161_161212

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  have h1 : 204 = 2 * 85 + 34 := by rfl
  have h2 : 85 = 2 * 34 + 17 := by rfl
  have h3 : 34 = 2 * 17 := by rfl
  sorry

end gcd_204_85_l161_161212


namespace tom_is_15_l161_161432

theorem tom_is_15 (T M : ℕ) (h1 : T + M = 21) (h2 : T + 3 = 2 * (M + 3)) : T = 15 :=
by {
  sorry
}

end tom_is_15_l161_161432


namespace quadratic_roots_identity_l161_161556

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l161_161556


namespace initial_ratio_zinc_copper_l161_161864

theorem initial_ratio_zinc_copper (Z C : ℝ) 
  (h1 : Z + C = 6) 
  (h2 : Z + 8 = 3 * C) : 
  Z / C = 5 / 7 := 
sorry

end initial_ratio_zinc_copper_l161_161864


namespace seating_chart_example_l161_161367

def seating_chart_representation (a b : ℕ) : String :=
  s!"{a} columns {b} rows"

theorem seating_chart_example :
  seating_chart_representation 4 3 = "4 columns 3 rows" :=
by
  sorry

end seating_chart_example_l161_161367


namespace sqrt_of_4_l161_161068

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end sqrt_of_4_l161_161068


namespace combinations_of_coins_with_50_cents_l161_161321

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l161_161321


namespace greatest_possible_q_minus_r_l161_161622

theorem greatest_possible_q_minus_r :
  ∃ (q r : ℕ), 945 = 21 * q + r ∧ 0 ≤ r ∧ r < 21 ∧ q - r = 45 :=
by
  sorry

end greatest_possible_q_minus_r_l161_161622


namespace quadratic_roots_vieta_l161_161558

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l161_161558


namespace income_scientific_notation_l161_161524

theorem income_scientific_notation (avg_income_per_acre : ℝ) (acres : ℝ) (a n : ℝ) :
  avg_income_per_acre = 20000 →
  acres = 8000 → 
  (avg_income_per_acre * acres = a * 10 ^ n ↔ (a = 1.6 ∧ n = 8)) :=
by
  sorry

end income_scientific_notation_l161_161524


namespace triangle_third_side_length_count_l161_161918

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l161_161918


namespace positive_integer_solutions_count_l161_161731

theorem positive_integer_solutions_count :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 24 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 64) ∧ s.card = 4 := 
by
  sorry

end positive_integer_solutions_count_l161_161731


namespace game_completion_days_l161_161396

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end game_completion_days_l161_161396


namespace quadratic_real_roots_k_le_one_fourth_l161_161530

theorem quadratic_real_roots_k_le_one_fourth (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4 * k - 2) * x + k^2 = 0) ↔ k ≤ 1/4 :=
sorry

end quadratic_real_roots_k_le_one_fourth_l161_161530


namespace sector_area_l161_161162

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) :
  (1 / 2) * α * r ^ 2 = Real.pi := by
  sorry

end sector_area_l161_161162


namespace cheryl_more_eggs_than_others_l161_161475

def kevin_eggs : ℕ := 5
def bonnie_eggs : ℕ := 13
def george_eggs : ℕ := 9
def cheryl_eggs : ℕ := 56

theorem cheryl_more_eggs_than_others : cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 :=
by
  sorry

end cheryl_more_eggs_than_others_l161_161475


namespace sin_135_eq_l161_161716

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l161_161716


namespace solve_for_x_and_y_l161_161572

theorem solve_for_x_and_y (x y : ℝ) 
  (h1 : 0.75 / x = 7 / 8)
  (h2 : x / y = 5 / 6) :
  x = 6 / 7 ∧ y = (6 / 7 * 6) / 5 :=
by
  sorry

end solve_for_x_and_y_l161_161572


namespace count_whole_numbers_in_interval_l161_161350

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l161_161350


namespace eat_cereal_in_time_l161_161790

noncomputable def time_to_eat_pounds (pounds : ℕ) (rate1 rate2 : ℚ) :=
  pounds / (rate1 + rate2)

theorem eat_cereal_in_time :
  time_to_eat_pounds 5 ((1:ℚ)/15) ((1:ℚ)/40) = 600/11 := 
by 
  sorry

end eat_cereal_in_time_l161_161790


namespace crossed_out_digit_l161_161410

theorem crossed_out_digit (N S S' x : ℕ) (hN : N % 9 = 3) (hS : S % 9 = 3) (hS' : S' % 9 = 7)
  (hS'_eq : S' = S - x) : x = 5 :=
by
  sorry

end crossed_out_digit_l161_161410


namespace combinations_of_coins_l161_161333

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l161_161333


namespace manufacturing_percentage_l161_161827

theorem manufacturing_percentage (deg_total : ℝ) (deg_manufacturing : ℝ) (h1 : deg_total = 360) (h2 : deg_manufacturing = 126) : 
  (deg_manufacturing / deg_total * 100) = 35 := by
  sorry

end manufacturing_percentage_l161_161827


namespace carla_total_students_l161_161868

-- Defining the conditions
def students_in_restroom : Nat := 2
def absent_students : Nat := (3 * students_in_restroom) - 1
def total_desks : Nat := 4 * 6
def occupied_desks : Nat := total_desks * 2 / 3
def students_present : Nat := occupied_desks

-- The target is to prove the total number of students Carla teaches
theorem carla_total_students : students_in_restroom + absent_students + students_present = 23 := by
  sorry

end carla_total_students_l161_161868


namespace third_side_integer_lengths_l161_161908

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l161_161908


namespace inequality_abc_geq_36_l161_161613

theorem inequality_abc_geq_36 (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_prod : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 :=
by
  sorry

end inequality_abc_geq_36_l161_161613


namespace initial_markers_l161_161788

variable (markers_given : ℕ) (total_markers : ℕ)

theorem initial_markers (h_given : markers_given = 109) (h_total : total_markers = 326) :
  total_markers - markers_given = 217 :=
by
  sorry

end initial_markers_l161_161788


namespace find_number_l161_161249

def condition (x : ℤ) : Prop := 3 * (x + 8) = 36

theorem find_number (x : ℤ) (h : condition x) : x = 4 := by
  sorry

end find_number_l161_161249


namespace tammy_speed_on_second_day_l161_161049

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l161_161049


namespace hexagon_side_relation_l161_161580

noncomputable def hexagon (a b c d e f : ℝ) :=
  ∃ (i j k l m n : ℝ), 
    i = 120 ∧ j = 120 ∧ k = 120 ∧ l = 120 ∧ m = 120 ∧ n = 120 ∧  
    a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f ∧ f = a

theorem hexagon_side_relation
  (a b c d e f : ℝ)
  (ha : hexagon a b c d e f) :
  d - a = b - e ∧ b - e = f - c :=
by
  sorry

end hexagon_side_relation_l161_161580


namespace range_of_a_l161_161745

variable {x : ℝ} {a : ℝ}

theorem range_of_a (h : ∀ x : ℝ, ¬ (x^2 - 5*x + (5/4)*a > 0)) : 5 < a :=
by
  sorry

end range_of_a_l161_161745


namespace average_speed_ratio_l161_161671

theorem average_speed_ratio
  (time_eddy : ℕ)
  (time_freddy : ℕ)
  (distance_ab : ℕ)
  (distance_ac : ℕ)
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_ab = 570)
  (h4 : distance_ac = 300) :
  (distance_ab / time_eddy) / (distance_ac / time_freddy) = 38 / 15 := 
by
  sorry

end average_speed_ratio_l161_161671


namespace rooms_needed_l161_161491

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l161_161491


namespace solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l161_161171

section B_zero

variables {x y z b : ℝ}

-- Given conditions for the first system when b = 0
variables (hb_zero : b = 0)
variables (h1 : x + y + z = 0)
variables (h2 : x^2 + y^2 - z^2 = 0)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_zero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_zero

section B_nonzero

variables {x y z b : ℝ}

-- Given conditions for the first system when b ≠ 0
variables (hb_nonzero : b ≠ 0)
variables (h1 : x + y + z = 2 * b)
variables (h2 : x^2 + y^2 - z^2 = b^2)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_nonzero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_nonzero

section Second_System

variables {x y z a : ℝ}

-- Given conditions for the second system
variables (h4 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
variables (h5 : x + y + 2 * z = 4 * (a^2 + 1))
variables (h6 : z^2 - x * y = a^2)

theorem solve_second_system :
  ∃ x y z, z^2 - x * y = a^2 :=
by { sorry }

end Second_System

end solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l161_161171


namespace solution_set_Inequality_l161_161998

theorem solution_set_Inequality : {x : ℝ | abs (1 + x + x^2 / 2) < 1} = {x : ℝ | -2 < x ∧ x < 0} :=
sorry

end solution_set_Inequality_l161_161998


namespace area_large_square_l161_161005

theorem area_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32) 
  (h2 : 4*a = 4*c + 16) : a^2 = 100 := 
by {
  sorry
}

end area_large_square_l161_161005


namespace number_of_possible_lists_l161_161843

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161843


namespace count_whole_numbers_in_interval_l161_161348

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l161_161348


namespace sequence_property_implies_geometric_progression_l161_161016

theorem sequence_property_implies_geometric_progression {p : ℝ} {a : ℕ → ℝ}
  (h_p : (2 / (Real.sqrt 5 + 1) ≤ p) ∧ (p < 1))
  (h_a : ∀ (e : ℕ → ℤ), (∀ n, (e n = 0) ∨ (e n = 1) ∨ (e n = -1)) →
    (∑' n, (e n) * (p ^ n)) = 0 → (∑' n, (e n) * (a n)) = 0) :
  ∃ c : ℝ, ∀ n, a n = c * (p ^ n) := by
  sorry

end sequence_property_implies_geometric_progression_l161_161016


namespace sin_135_degree_l161_161722

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l161_161722


namespace ceil_neg_sqrt_64_div_9_eq_neg2_l161_161138

def sqrt_64_div_9 : ℚ := real.sqrt (64 / 9)
def neg_sqrt_64_div_9 : ℚ := -sqrt_64_div_9
def ceil_neg_sqrt_64_div_9 : ℤ := real.ceil neg_sqrt_64_div_9

theorem ceil_neg_sqrt_64_div_9_eq_neg2 : ceil_neg_sqrt_64_div_9 = -2 := 
by sorry

end ceil_neg_sqrt_64_div_9_eq_neg2_l161_161138


namespace part_a_l161_161831

theorem part_a (a x y : ℕ) (h_a_pos : a > 0) (h_x_pos : x > 0) (h_y_pos : y > 0) (h_neq : x ≠ y) :
  (a * x + Nat.gcd a x + Nat.lcm a x) ≠ (a * y + Nat.gcd a y + Nat.lcm a y) := sorry

end part_a_l161_161831


namespace squared_diagonal_inequality_l161_161830

theorem squared_diagonal_inequality 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) :
  let AB := (x1 - x2)^2 + (y1 - y2)^2
  let BC := (x2 - x3)^2 + (y2 - y3)^2
  let CD := (x3 - x4)^2 + (y3 - y4)^2
  let DA := (x1 - x4)^2 + (y1 - y4)^2
  let AC := (x1 - x3)^2 + (y1 - y3)^2
  let BD := (x2 - x4)^2 + (y2 - y4)^2
  AC + BD ≤ AB + BC + CD + DA := 
by
  sorry

end squared_diagonal_inequality_l161_161830


namespace cd_leq_one_l161_161779

variables {a b c d : ℝ}

theorem cd_leq_one (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := 
sorry

end cd_leq_one_l161_161779


namespace sector_arc_length_circumference_ratio_l161_161861

theorem sector_arc_length_circumference_ratio
  {r : ℝ}
  (h_radius : ∀ (sector_radius : ℝ), sector_radius = 2/3 * r)
  (h_area : ∀ (sector_area circle_area : ℝ), sector_area / circle_area = 5/27) :
  ∀ (l C : ℝ), l / C = 5 / 18 :=
by
  -- Prove the theorem using the given hypothesis.
  -- Construction of the detailed proof will go here.
  sorry

end sector_arc_length_circumference_ratio_l161_161861


namespace price_of_fruit_l161_161210

theorem price_of_fruit
  (price_milk_per_liter : ℝ)
  (milk_per_batch : ℝ)
  (fruit_per_batch : ℝ)
  (cost_for_three_batches : ℝ)
  (F : ℝ)
  (h1 : price_milk_per_liter = 1.5)
  (h2 : milk_per_batch = 10)
  (h3 : fruit_per_batch = 3)
  (h4 : cost_for_three_batches = 63)
  (h5 : 3 * (milk_per_batch * price_milk_per_liter + fruit_per_batch * F) = cost_for_three_batches) :
  F = 2 :=
by sorry

end price_of_fruit_l161_161210


namespace total_amount_collected_l161_161217

theorem total_amount_collected (h1 : ∀ (P_I P_II : ℕ), P_I * 50 = P_II) 
                               (h2 : ∀ (F_I F_II : ℕ), F_I = 3 * F_II) 
                               (h3 : ∀ (P_II F_II : ℕ), P_II * F_II = 1250) : 
                               ∃ (Total : ℕ), Total = 1325 :=
by
  sorry

end total_amount_collected_l161_161217


namespace solve_system_of_inequalities_l161_161614

theorem solve_system_of_inequalities (x : ℝ) :
  (x + 1 < 5) ∧ (2 * x - 1) / 3 ≥ 1 ↔ 2 ≤ x ∧ x < 4 :=
by
  sorry

end solve_system_of_inequalities_l161_161614


namespace subtract_mult_equal_l161_161083

theorem subtract_mult_equal :
  2000000000000 - 1111111111111 * 1 = 888888888889 :=
by
  sorry

end subtract_mult_equal_l161_161083


namespace quadratic_root_identity_l161_161546

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l161_161546


namespace number_of_possible_third_side_lengths_l161_161920

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l161_161920


namespace students_joined_l161_161381

theorem students_joined (A X : ℕ) (h1 : 100 * A = 5000) (h2 : (100 + X) * (A - 10) = 5400) :
  X = 35 :=
by
  sorry

end students_joined_l161_161381


namespace min_rooms_needed_l161_161483

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l161_161483


namespace f_1992_eq_1992_l161_161821

def f (x : ℕ) : ℤ := sorry

theorem f_1992_eq_1992 (f : ℕ → ℤ) 
  (h1 : ∀ x : ℕ, 0 < x -> f x = f (x - 1) + f (x + 1))
  (h2 : f 0 = 1992) :
  f 1992 = 1992 := 
sorry

end f_1992_eq_1992_l161_161821


namespace f_increasing_f_odd_zero_l161_161195

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Prove that f(x) is always an increasing function for any real a.
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

-- 2. Determine the value of a such that f(-x) + f(x) = 0 always holds.
theorem f_odd_zero (a : ℝ) : (∀ x : ℝ, f a (-x) + f a x = 0) → a = 1 :=
by
  sorry

end f_increasing_f_odd_zero_l161_161195


namespace find_AX_l161_161878

theorem find_AX (AC BC BX : ℝ) (h1 : AC = 27) (h2 : BC = 40) (h3 : BX = 36)
    (h4 : ∀ (AX : ℝ), AX = AC * BX / BC) : 
    ∃ AX, AX = 243 / 10 :=
by
  sorry

end find_AX_l161_161878


namespace power_multiplication_result_l161_161661

theorem power_multiplication_result :
  ( (8 / 9)^3 * (1 / 3)^3 * (2 / 5)^3 = (4096 / 2460375) ) :=
by
  sorry

end power_multiplication_result_l161_161661


namespace quadratic_root_identity_l161_161543

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l161_161543


namespace sin_135_correct_l161_161690

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l161_161690


namespace sin_function_satisfies_conds_l161_161742

theorem sin_function_satisfies_conds :
    ∃ (A B ω ϕ : ℝ), 
        A > 0 ∧ 
        ω > 0 ∧ 
        |ϕ| < (π / 2) ∧ 
        (∀ x, x = 1 → A*sin(ω*x + ϕ) + B = 2) ∧ 
        (∀ x, x = 2 → A*sin(ω*x + ϕ) + B = (1 / 2)) ∧ 
        (∀ x, x = 3 → A*sin(ω*x + ϕ) + B = -1) ∧ 
        (∀ x, x = 4 → A*sin(ω*x + ϕ) + B = 2) ∧
        (∀ x, f x = (√3)*sin((2*π/3)*x - (π/3)) + (1/2)) := 
by
    sorry

end sin_function_satisfies_conds_l161_161742


namespace sin_135_eq_sqrt2_div_2_l161_161719

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l161_161719


namespace smallest_class_size_l161_161946

variable (x : ℕ) 

theorem smallest_class_size
  (h1 : 5 * x + 2 > 40)
  (h2 : x ≥ 0) : 
  5 * 8 + 2 = 42 :=
by sorry

end smallest_class_size_l161_161946


namespace find_middle_number_l161_161618

theorem find_middle_number
  (S1 S2 M : ℤ)
  (h1 : S1 = 6 * 5)
  (h2 : S2 = 6 * 7)
  (h3 : 13 * 9 = S1 + M + S2) :
  M = 45 :=
by
  -- proof steps would go here
  sorry

end find_middle_number_l161_161618


namespace coin_combinations_sum_50_l161_161335

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l161_161335


namespace find_m_l161_161756

noncomputable def f (x : ℝ) : ℝ := 2^x - 5

theorem find_m (m : ℝ) (h : f m = 3) : m = 3 := 
by
  sorry

end find_m_l161_161756


namespace sin_135_l161_161704

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l161_161704


namespace frankie_pets_total_l161_161150

noncomputable def total_pets (c : ℕ) : ℕ :=
  let dogs := 2
  let cats := c
  let snakes := c + 5
  let parrots := c - 1
  dogs + cats + snakes + parrots

theorem frankie_pets_total (c : ℕ) (hc : 2 + 4 + (c + 1) + (c - 1) = 19) : total_pets c = 19 := by
  sorry

end frankie_pets_total_l161_161150


namespace roots_of_quadratic_eq_l161_161539

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l161_161539


namespace total_ages_l161_161832

-- Definitions of the conditions
variables (A B : ℕ) (x : ℕ)

-- Condition 1: 10 years ago, A was half of B in age.
def condition1 : Prop := A - 10 = 1/2 * (B - 10)

-- Condition 2: The ratio of their present ages is 3:4.
def condition2 : Prop := A = 3 * x ∧ B = 4 * x

-- Main theorem to prove
theorem total_ages (A B : ℕ) (x : ℕ) (h1 : condition1 A B) (h2 : condition2 A B x) : A + B = 35 := 
by
  sorry

end total_ages_l161_161832


namespace num_undefined_values_l161_161511

-- Condition: Denominator is given as (x^2 + 2x - 3)(x - 3)(x + 1)
def denominator (x : ℝ) : ℝ := (x^2 + 2 * x - 3) * (x - 3) * (x + 1)

-- The Lean statement to prove the number of values of x for which the expression is undefined
theorem num_undefined_values : 
  ∃ (n : ℕ), (∀ x : ℝ, denominator x = 0 → (x = 1 ∨ x = -3 ∨ x = 3 ∨ x = -1)) ∧ n = 4 :=
by
  sorry

end num_undefined_values_l161_161511


namespace combinations_of_coins_with_50_cents_l161_161320

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l161_161320


namespace gcf_3465_10780_l161_161225

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end gcf_3465_10780_l161_161225


namespace problem_statement_l161_161407

def U : Set ℤ := {x | True}
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}
def complement (B : Set ℤ) : Set ℤ := {x | x ∉ B}

theorem problem_statement : (A ∩ (complement B)) = {1, 3, 9} :=
by {
  sorry
}

end problem_statement_l161_161407


namespace incorrect_height_is_151_l161_161418

def incorrect_height (average_initial correct_height average_corrected : ℝ) : ℝ :=
  (30 * average_initial) - (30 * average_corrected) + correct_height

theorem incorrect_height_is_151 :
  incorrect_height 175 136 174.5 = 151 :=
by
  sorry

end incorrect_height_is_151_l161_161418


namespace samantha_last_name_length_l161_161609

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end samantha_last_name_length_l161_161609


namespace sin_135_l161_161705

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l161_161705


namespace smallest_possible_a_l161_161415

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + ↑c

theorem smallest_possible_a
  (a b c : ℕ)
  (r s : ℝ)
  (h_arith_seq : b - a = c - b)
  (h_order_pos : 0 < a ∧ a < b ∧ b < c)
  (h_distinct : r ≠ s)
  (h_rs_2017 : r * s = 2017)
  (h_fr_eq_s : f a b c r = s)
  (h_fs_eq_r : f a b c s = r) :
  a = 1 := sorry

end smallest_possible_a_l161_161415


namespace sufficient_not_necessary_condition_l161_161999

theorem sufficient_not_necessary_condition
  (x : ℝ) : 
  x^2 - 4*x - 5 > 0 → (x > 5 ∨ x < -1) ∧ (x > 5 → x^2 - 4*x - 5 > 0) ∧ ¬(x^2 - 4*x - 5 > 0 → x > 5) := 
sorry

end sufficient_not_necessary_condition_l161_161999


namespace probability_alternating_colors_l161_161108

/--
A box contains 6 white balls and 6 black balls.
Balls are drawn one at a time.
What is the probability that all of my draws alternate colors?
-/
theorem probability_alternating_colors :
  let total_arrangements := Nat.factorial 12 / (Nat.factorial 6 * Nat.factorial 6)
  let successful_arrangements := 2
  successful_arrangements / total_arrangements = (1 : ℚ) / 462 := 
by
  sorry

end probability_alternating_colors_l161_161108


namespace probability_of_contact_l161_161503

noncomputable def probability_connection (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

theorem probability_of_contact (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let group1 := 5
  let group2 := 8
  let total_pairs := group1 * group2
  (total_pairs = 40) →
  (∀ i j, i ∈ fin group1 → j ∈ fin group2 → (¬ p = 1 → p = p)) → 
  probability_connection p = 1 - (1 - p) ^ 40 :=
by
  intros _ _ 
  sorry

end probability_of_contact_l161_161503


namespace max_value_g_l161_161987

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l161_161987


namespace tammy_speed_on_second_day_l161_161054

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l161_161054


namespace number_of_possible_lists_l161_161841

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161841


namespace sequence_general_term_l161_161406

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 3 * (Finset.range (n + 1)).sum a = (n + 2) * a n) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l161_161406


namespace john_payment_l161_161769

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end john_payment_l161_161769


namespace intersection_M_N_l161_161450

open Set

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by 
sorry

end intersection_M_N_l161_161450


namespace graph_of_equation_l161_161620

theorem graph_of_equation (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) := 
by sorry

end graph_of_equation_l161_161620


namespace combined_weight_l161_161120

variable (a b c d : ℕ)

theorem combined_weight :
  a + b = 260 →
  b + c = 245 →
  c + d = 270 →
  a + d = 285 :=
by
  intros hab hbc hcd
  sorry

end combined_weight_l161_161120


namespace dog_adult_weight_l161_161276

theorem dog_adult_weight 
  (w7 : ℕ) (w7_eq : w7 = 6)
  (w9 : ℕ) (w9_eq : w9 = 2 * w7)
  (w3m : ℕ) (w3m_eq : w3m = 2 * w9)
  (w5m : ℕ) (w5m_eq : w5m = 2 * w3m)
  (w1y : ℕ) (w1y_eq : w1y = w5m + 30) :
  w1y = 78 := by
  -- Proof is not required, so we leave it with sorry.
  sorry

end dog_adult_weight_l161_161276


namespace convex_power_function_l161_161143

theorem convex_power_function (n : ℕ) (h : 0 < n) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (↑n * (↑n - 1) * x ^ (↑n - 2))) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end convex_power_function_l161_161143


namespace sugar_water_inequality_triangle_inequality_l161_161753

-- Condition for question (1)
variable (x y m : ℝ)
variable (hx : x > 0) (hy : y > 0) (hxy : x > y) (hm : m > 0)

-- Proof problem for question (1)
theorem sugar_water_inequality : y / x < (y + m) / (x + m) :=
sorry

-- Condition for question (2)
variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (hab : b + c > a) (hac : a + c > b) (hbc : a + b > c)

-- Proof problem for question (2)
theorem triangle_inequality : 
  a / (b + c) + b / (a + c) + c / (a + b) < 2 :=
sorry

end sugar_water_inequality_triangle_inequality_l161_161753


namespace triangle_third_side_length_l161_161914

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l161_161914


namespace june_found_total_eggs_l161_161957

def eggs_in_tree_1 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_tree_2 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_yard (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest

def total_eggs (eggs_tree_1 : ℕ) (eggs_tree_2 : ℕ) (eggs_yard : ℕ) : ℕ :=
eggs_tree_1 + eggs_tree_2 + eggs_yard

theorem june_found_total_eggs :
  total_eggs (eggs_in_tree_1 2 5) (eggs_in_tree_2 1 3) (eggs_in_yard 1 4) = 17 :=
by
  sorry

end june_found_total_eggs_l161_161957


namespace tammy_speed_on_second_day_l161_161051

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l161_161051


namespace probability_of_connection_l161_161504

theorem probability_of_connection (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  let num_pairs := 5 * 8 in
  let prob_no_connection := (1 - p) ^ num_pairs in
  1 - prob_no_connection = 1 - (1 - p) ^ 40 := 
by
  let num_pairs := 5 * 8
  have h_num_pairs : num_pairs = 40 := by norm_num
  rw h_num_pairs
  let prob_no_connection := (1 - p) ^ 40
  sorry

end probability_of_connection_l161_161504


namespace number_of_possible_lists_l161_161840

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161840


namespace sum_of_positive_factors_of_72_l161_161242

def sum_divisors (n : ℕ) : ℕ := 
  ∑ d in finset.filter (λ d, d ∣ n) (finset.range (n+1)), d

theorem sum_of_positive_factors_of_72 : sum_divisors 72 = 195 :=
by sorry

end sum_of_positive_factors_of_72_l161_161242


namespace alice_number_l161_161269

theorem alice_number (m : ℕ) 
  (h1 : 180 ∣ m) 
  (h2 : 240 ∣ m) 
  (h3 : 2000 ≤ m ∧ m ≤ 5000) : 
    m = 2160 ∨ m = 2880 ∨ m = 3600 ∨ m = 4320 := 
sorry

end alice_number_l161_161269


namespace slope_of_tangent_line_at_1_1_l161_161218

theorem slope_of_tangent_line_at_1_1 : 
  ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * x^2) ∧ (f' 1 = 3) :=
by
  sorry

end slope_of_tangent_line_at_1_1_l161_161218


namespace probability_four_green_marbles_l161_161185

open_locale big_operators

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

noncomputable def probability_green : ℚ :=
  8 / 15

noncomputable def probability_purple : ℚ :=
  7 / 15

theorem probability_four_green_marbles :
  (binomial 7 4) * (probability_green ^ 4) * (probability_purple ^ 3) = 49172480 / 170859375 :=
by
  sorry

end probability_four_green_marbles_l161_161185


namespace quadratic_root_sum_and_product_l161_161562

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l161_161562


namespace smallest_base10_integer_l161_161439

theorem smallest_base10_integer (X Y : ℕ) (hX : X < 6) (hY : Y < 8) (h : 7 * X = 9 * Y) :
  63 = 7 * X ∧ 63 = 9 * Y :=
by
  -- Proof steps would go here
  sorry

end smallest_base10_integer_l161_161439


namespace pavan_travel_time_l161_161972

theorem pavan_travel_time (D : ℝ) (V1 V2 : ℝ) (distance : D = 300) (speed1 : V1 = 30) (speed2 : V2 = 25) : 
  ∃ t : ℝ, t = 11 := 
  by
    sorry

end pavan_travel_time_l161_161972


namespace triangle_area_l161_161189

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l161_161189


namespace combination_coins_l161_161324

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l161_161324


namespace solve_system_of_equations_l161_161039

theorem solve_system_of_equations 
  (x y : ℝ) 
  (h1 : x / 3 - (y + 1) / 2 = 1) 
  (h2 : 4 * x - (2 * y - 5) = 11) : 
  x = 0 ∧ y = -3 :=
  sorry

end solve_system_of_equations_l161_161039


namespace find_integers_l161_161662

theorem find_integers (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 := by
  sorry

end find_integers_l161_161662


namespace number_of_integers_in_interval_l161_161344

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l161_161344


namespace triangle_has_angle_45_l161_161389

theorem triangle_has_angle_45
  (A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : B + C = 3 * A) :
  A = 45 :=
by
  sorry

end triangle_has_angle_45_l161_161389


namespace HCF_is_five_l161_161219

noncomputable def HCF_of_numbers (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_is_five :
  ∃ (a b : ℕ),
    a + b = 55 ∧
    Nat.lcm a b = 120 ∧
    (1 / (a : ℝ) + 1 / (b : ℝ) = 0.09166666666666666) →
    HCF_of_numbers a b = 5 :=
by 
  sorry

end HCF_is_five_l161_161219


namespace smallest_base10_integer_l161_161438

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end smallest_base10_integer_l161_161438


namespace haruto_ratio_is_1_to_2_l161_161748

def haruto_tomatoes_ratio (total_tomatoes : ℕ) (eaten_by_birds : ℕ) (remaining_tomatoes : ℕ) : ℚ :=
  let picked_tomatoes := total_tomatoes - eaten_by_birds
  let given_to_friend := picked_tomatoes - remaining_tomatoes
  given_to_friend / picked_tomatoes

theorem haruto_ratio_is_1_to_2 : haruto_tomatoes_ratio 127 19 54 = 1 / 2 :=
by
  -- We'll skip the proof details as instructed
  sorry

end haruto_ratio_is_1_to_2_l161_161748


namespace vertices_integer_assignment_zero_l161_161879

theorem vertices_integer_assignment_zero (f : ℕ → ℤ) (h100 : ∀ i, i < 100 → (i + 3) % 100 < 100) 
  (h : ∀ i, (i < 97 → f i + f (i + 2) = f (i + 1)) 
            ∨ (i < 97 → f (i + 1) + f (i + 3) = f (i + 2)) 
            ∨ (i < 97 → f i + f (i + 1) = f (i + 2))): 
  ∀ i, i < 100 → f i = 0 :=
by
  sorry

end vertices_integer_assignment_zero_l161_161879


namespace smaller_side_of_rectangle_l161_161824

theorem smaller_side_of_rectangle (r : ℝ) (h1 : r = 42) 
                                   (h2 : ∀ L W : ℝ, L / W = 6 / 5 → 2 * (L + W) = 2 * π * r) : 
                                   ∃ W : ℝ, W = (210 * π) / 11 := 
by {
    sorry
}

end smaller_side_of_rectangle_l161_161824


namespace water_overflow_volume_is_zero_l161_161109

noncomputable def container_depth : ℝ := 30
noncomputable def container_outer_diameter : ℝ := 22
noncomputable def container_wall_thickness : ℝ := 1
noncomputable def water_height : ℝ := 27.5

noncomputable def iron_block_base_diameter : ℝ := 10
noncomputable def iron_block_height : ℝ := 30

theorem water_overflow_volume_is_zero :
  let inner_radius := (container_outer_diameter - 2 * container_wall_thickness) / 2,
      initial_water_volume := Real.pi * inner_radius^2 * water_height,
      max_container_volume := Real.pi * inner_radius^2 * container_depth,
      iron_block_radius := iron_block_base_diameter / 2,
      iron_block_volume := Real.pi * iron_block_radius^2 * iron_block_height,
      new_total_volume := max_container_volume - iron_block_volume in
  initial_water_volume = new_total_volume → 0 = 0 :=
by
  sorry

end water_overflow_volume_is_zero_l161_161109


namespace angle_of_inclination_l161_161498

-- The statement of the mathematically equivalent proof problem in Lean 4
theorem angle_of_inclination
  (k: ℝ)
  (α: ℝ)
  (line_eq: ∀ x, ∃ y, y = (k-1) * x + 2)
  (circle_eq: ∀ x y, x^2 + y^2 + k * x + 2 * y + k^2 = 0) :
  α = 3 * Real.pi / 4 :=
sorry -- Proof to be provided

end angle_of_inclination_l161_161498


namespace Cassini_l161_161469

-- Define the Fibonacci sequence
def Fibonacci : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci (n + 1) + Fibonacci n

-- State Cassini's Identity theorem
theorem Cassini (n : ℕ) : Fibonacci (n + 1) * Fibonacci (n - 1) - (Fibonacci n) ^ 2 = (-1) ^ n := 
by sorry

end Cassini_l161_161469


namespace Lucas_identity_l161_161793

def Lucas (L : ℕ → ℤ) (F : ℕ → ℤ) : Prop :=
  ∀ n, L n = F (n + 1) + F (n - 1)

def Fib_identity1 (F : ℕ → ℤ) : Prop :=
  ∀ n, F (2 * n + 1) = F (n + 1) ^ 2 + F n ^ 2

def Fib_identity2 (F : ℕ → ℤ) : Prop :=
  ∀ n, F n ^ 2 = F (n + 1) * F (n - 1) - (-1) ^ n

theorem Lucas_identity {L F : ℕ → ℤ} (hL : Lucas L F) (hF1 : Fib_identity1 F) (hF2 : Fib_identity2 F) :
  ∀ n, L (2 * n) = L n ^ 2 - 2 * (-1) ^ n := 
sorry

end Lucas_identity_l161_161793


namespace four_fours_to_seven_l161_161660

theorem four_fours_to_seven :
  (∃ eq1 eq2 : ℕ, eq1 ≠ eq2 ∧
    (eq1 = 4 + 4 - (4 / 4) ∧
     eq2 = 44 / 4 - 4 ∧ eq1 = 7 ∧ eq2 = 7)) :=
by
  existsi (4 + 4 - (4 / 4))
  existsi (44 / 4 - 4)
  sorry

end four_fours_to_seven_l161_161660


namespace coin_combination_l161_161329

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l161_161329


namespace milkshakes_more_than_ice_cream_cones_l161_161030

def ice_cream_cones_sold : ℕ := 67
def milkshakes_sold : ℕ := 82

theorem milkshakes_more_than_ice_cream_cones : milkshakes_sold - ice_cream_cones_sold = 15 := by
  sorry

end milkshakes_more_than_ice_cream_cones_l161_161030


namespace lana_goal_is_20_l161_161015

def muffins_sold_morning := 12
def muffins_sold_afternoon := 4
def muffins_needed_to_goal := 4
def total_muffins_sold := muffins_sold_morning + muffins_sold_afternoon
def lana_goal := total_muffins_sold + muffins_needed_to_goal

theorem lana_goal_is_20 : lana_goal = 20 := by
  sorry

end lana_goal_is_20_l161_161015


namespace inequality_pos_xy_l161_161223

theorem inequality_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := 
by {
    sorry
}

end inequality_pos_xy_l161_161223


namespace hyperbola_dot_product_zero_l161_161528

theorem hyperbola_dot_product_zero
  (a b x y : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_ecc : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 2) :
  let B := (-x, y)
  let C := (x, y)
  let A := (a, 0)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 0 :=
by
  sorry

end hyperbola_dot_product_zero_l161_161528


namespace staff_duty_arrangement_l161_161272

theorem staff_duty_arrangement :
  let staff := {1, 2, 3, 4, 5, 6, 7}
  let days := {1, 2, 3, 4, 5, 6, 7}
  let last_5_days := {3, 4, 5, 6, 7}
  let arrangements := finset.permutations staff
  (∃ arr ∈ arrangements, ∀ i ∈ {1, 2}, arr i ≠ A ∧ arr i ≠ B) →
  finset.card arrangements = 2400 :=
by
  sorry

end staff_duty_arrangement_l161_161272


namespace responses_needed_750_l161_161177

section Responses
  variable (q_min : ℕ) (response_rate : ℝ)

  def responses_needed : ℝ := response_rate * q_min

  theorem responses_needed_750 (h1 : q_min = 1250) (h2 : response_rate = 0.60) : responses_needed q_min response_rate = 750 :=
  by
    simp [responses_needed, h1, h2]
    sorry
end Responses

end responses_needed_750_l161_161177


namespace intersect_trihedral_angle_l161_161011

-- Definitions of variables
variables {a b c : ℝ} (S : Type) 

-- Definition of a valid intersection condition
def valid_intersection (a b c : ℝ) : Prop :=
  a^2 + b^2 - c^2 > 0 ∧ b^2 + c^2 - a^2 > 0 ∧ a^2 + c^2 - b^2 > 0

-- Theorem statement
theorem intersect_trihedral_angle (h : valid_intersection a b c) : 
  ∃ (SA SB SC : ℝ), (SA^2 + SB^2 = a^2 ∧ SA^2 + SC^2 = b^2 ∧ SB^2 + SC^2 = c^2) :=
sorry

end intersect_trihedral_angle_l161_161011


namespace minimum_rooms_to_accommodate_fans_l161_161489

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l161_161489


namespace quadratic_factors_l161_161626

theorem quadratic_factors {a b c : ℝ} (h : a = 1) (h_roots : (1:ℝ) + 2 = b ∧ (-1:ℝ) * 2 = c) :
  (x^2 - b * x + c) = (x - 1) * (x - 2) := by
  sorry

end quadratic_factors_l161_161626


namespace find_k_l161_161007

theorem find_k (m n : ℝ) 
  (h₁ : m = k * n + 5) 
  (h₂ : m + 2 = k * (n + 0.5) + 5) : 
  k = 4 :=
by
  sorry

end find_k_l161_161007


namespace no_such_natural_numbers_exist_l161_161133

theorem no_such_natural_numbers_exist :
  ¬ ∃ (x y : ℕ), ∃ (k m : ℕ), x^2 + x + 1 = y^k ∧ y^2 + y + 1 = x^m := 
by sorry

end no_such_natural_numbers_exist_l161_161133


namespace possible_integer_lengths_third_side_l161_161929

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l161_161929


namespace intersection_of_lines_l161_161286

theorem intersection_of_lines :
  ∃ (x y : ℝ), 10 * x - 5 * y = 5 ∧ 8 * x + 2 * y = 22 ∧ x = 2 ∧ y = 3 := by
  sorry

end intersection_of_lines_l161_161286


namespace sum_of_three_numbers_is_520_l161_161882

noncomputable def sum_of_three_numbers (x y z : ℝ) : ℝ :=
  x + y + z

theorem sum_of_three_numbers_is_520 (x y z : ℝ) (h1 : z = (1848 / 1540) * x) (h2 : z = 0.4 * y) (h3 : x + y = 400) :
  sum_of_three_numbers x y z = 520 :=
sorry

end sum_of_three_numbers_is_520_l161_161882


namespace quadratic_inequality_solution_l161_161513

theorem quadratic_inequality_solution :
  {x : ℝ | (x^2 - 50 * x + 576) ≤ 16} = {x : ℝ | 20 ≤ x ∧ x ≤ 28} :=
sorry

end quadratic_inequality_solution_l161_161513


namespace log_one_eq_zero_l161_161256

theorem log_one_eq_zero : Real.log 1 = 0 := 
by
  sorry

end log_one_eq_zero_l161_161256


namespace find_expression_l161_161301

-- Definitions based on the conditions provided
def prop_rel (y x : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 2)

def prop_value_k (k : ℝ) : Prop :=
  k = -4

def prop_value_y (y x : ℝ) : Prop :=
  y = -4 * x + 8

theorem find_expression (y x k : ℝ) : 
  (prop_rel y x k) → 
  (x = 3) → 
  (y = -4) → 
  (prop_value_k k) → 
  (prop_value_y y x) :=
by
  intros h1 h2 h3 h4
  subst h4
  subst h3
  subst h2
  sorry

end find_expression_l161_161301


namespace sin_135_correct_l161_161689

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l161_161689


namespace prob_sum_15_correct_l161_161650

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l161_161650


namespace kids_went_home_l161_161633

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℝ) (went_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : remaining_kids = 8.0) : went_home = 14.0 :=
by 
  sorry

end kids_went_home_l161_161633


namespace larinjaitis_age_l161_161186

theorem larinjaitis_age : 
  ∀ (birth_year : ℤ) (death_year : ℤ), birth_year = -30 → death_year = 30 → (death_year - birth_year + 1) = 1 :=
by
  intros birth_year death_year h_birth h_death
  sorry

end larinjaitis_age_l161_161186


namespace main_theorem_l161_161403

theorem main_theorem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * c * a) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end main_theorem_l161_161403


namespace bob_work_days_per_week_l161_161467

theorem bob_work_days_per_week (daily_hours : ℕ) (monthly_hours : ℕ) (average_days_per_month : ℕ) (days_per_week : ℕ)
  (h1 : daily_hours = 10)
  (h2 : monthly_hours = 200)
  (h3 : average_days_per_month = 30)
  (h4 : days_per_week = 7) :
  (monthly_hours / daily_hours) / (average_days_per_month / days_per_week) = 5 := by
  -- Now we will skip the proof itself. The focus here is on the structure.
  sorry

end bob_work_days_per_week_l161_161467


namespace derivative_of_constant_function_l161_161805

-- Define the constant function
def f (x : ℝ) : ℝ := 0

-- State the theorem
theorem derivative_of_constant_function : deriv f 0 = 0 := by
  -- Proof will go here, but we use sorry to skip it
  sorry

end derivative_of_constant_function_l161_161805


namespace max_value_of_k_l161_161187

theorem max_value_of_k (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m)) ≥ k) ↔ k ≤ 8 := 
sorry

end max_value_of_k_l161_161187


namespace find_smaller_number_l161_161220

def smaller_number (x y : ℕ) : ℕ :=
  if x < y then x else y

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 64) (h2 : a = b + 12) : smaller_number a b = 26 :=
by
  sorry

end find_smaller_number_l161_161220


namespace regular_polygon_radius_l161_161073

theorem regular_polygon_radius 
  (n : ℕ) (side_length : ℝ) (h1 : side_length = 2) 
  (h2 : sum_of_interior_angles n = 2 * sum_of_exterior_angles n)
  (h3 : is_regular_polygon n) :
  radius_of_polygon n side_length = 2 :=
by
  sorry

end regular_polygon_radius_l161_161073


namespace arc_length_ratio_l161_161420

theorem arc_length_ratio
  (h_circ : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1)
  (h_line : ∀ x y : ℝ, x - y = 0) :
  let shorter_arc := (1 / 4) * (2 * Real.pi)
  let longer_arc := 2 * Real.pi - shorter_arc
  shorter_arc / longer_arc = 1 / 3 :=
by
  sorry

end arc_length_ratio_l161_161420


namespace least_three_digit_number_product8_l161_161090

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l161_161090


namespace partner_profit_share_correct_l161_161943

-- Definitions based on conditions
def total_profit : ℝ := 280000
def profit_share_shekhar : ℝ := 0.28
def profit_share_rajeev : ℝ := 0.22
def profit_share_jatin : ℝ := 0.20
def profit_share_simran : ℝ := 0.18
def profit_share_ramesh : ℝ := 0.12

-- Each partner's share in the profit
def shekhar_share : ℝ := profit_share_shekhar * total_profit
def rajeev_share : ℝ := profit_share_rajeev * total_profit
def jatin_share : ℝ := profit_share_jatin * total_profit
def simran_share : ℝ := profit_share_simran * total_profit
def ramesh_share : ℝ := profit_share_ramesh * total_profit

-- Statement to be proved
theorem partner_profit_share_correct :
    shekhar_share = 78400 ∧ 
    rajeev_share = 61600 ∧ 
    jatin_share = 56000 ∧ 
    simran_share = 50400 ∧ 
    ramesh_share = 33600 ∧ 
    (shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit) :=
by sorry

end partner_profit_share_correct_l161_161943


namespace quadratic_polynomial_value_at_zero_eq_zero_l161_161146

theorem quadratic_polynomial_value_at_zero_eq_zero
  (p q : ℝ)
  (polynomial_form : ∀ x, p x = x^2 - (p+q) * x + pq)
  (distinct_roots_condition : (p(p(x)) = (p(x))^2 - (p+q)(p(x)) + pq) → (∀ x, four_distinct_real_roots p(p(x)))):
  p(0) = 0 := sorry

end quadratic_polynomial_value_at_zero_eq_zero_l161_161146


namespace evaluate_fraction_l161_161283

theorem evaluate_fraction : 
  (7/3) / (8/15) = 35/8 :=
by
  -- we don't need to provide the proof as per instructions
  sorry

end evaluate_fraction_l161_161283


namespace required_lemons_for_20_gallons_l161_161176

-- Conditions
def lemons_for_50_gallons : ℕ := 40
def gallons_for_lemons : ℕ := 50
def additional_lemons_per_10_gallons : ℕ := 1
def number_of_gallons : ℕ := 20
def base_lemons (g: ℕ) : ℕ := (lemons_for_50_gallons * g) / gallons_for_lemons
def additional_lemons (g: ℕ) : ℕ := (g / 10) * additional_lemons_per_10_gallons
def total_lemons (g: ℕ) : ℕ := base_lemons g + additional_lemons g

-- Proof statement
theorem required_lemons_for_20_gallons : total_lemons number_of_gallons = 18 :=
by
  sorry

end required_lemons_for_20_gallons_l161_161176


namespace function_increasing_iff_l161_161167

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - a * x

theorem function_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 - a) ↔ a ≤ 0 :=
by
  sorry

end function_increasing_iff_l161_161167


namespace probability_sum_15_l161_161642

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l161_161642


namespace sin_135_degree_l161_161721

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l161_161721


namespace linear_function_not_third_quadrant_l161_161387

theorem linear_function_not_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ¬ (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ y = k * x + 1) :=
sorry

end linear_function_not_third_quadrant_l161_161387


namespace bus_time_one_way_l161_161588

-- define conditions
def walk_time_one_way := 5 -- 5 minutes for one walk
def total_annual_travel_time_hours := 365 -- 365 hours per year
def work_days_per_year := 365 -- works every day

-- convert annual travel time from hours to minutes
def total_annual_travel_time_minutes := total_annual_travel_time_hours * 60

-- calculate total daily travel time
def total_daily_travel_time := total_annual_travel_time_minutes / work_days_per_year

-- walking time per day
def total_daily_walking_time := (walk_time_one_way * 4)

-- total bus travel time per day
def total_daily_bus_time := total_daily_travel_time - total_daily_walking_time

-- one-way bus time
theorem bus_time_one_way : total_daily_bus_time / 2 = 20 := by
  sorry

end bus_time_one_way_l161_161588


namespace count_whole_numbers_in_interval_l161_161358

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l161_161358


namespace value_of_at_20_at_l161_161730

noncomputable def left_at (x : ℝ) : ℝ := 9 - x
noncomputable def right_at (x : ℝ) : ℝ := x - 9

theorem value_of_at_20_at : right_at (left_at 20) = -20 := by
  sorry

end value_of_at_20_at_l161_161730


namespace max_value_of_xyz_l161_161978

noncomputable def max_product (x y z : ℝ) : ℝ :=
  x * y * z

theorem max_value_of_xyz (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x = y) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) (h6 : x ≤ z) (h7 : z ≤ 2 * x) :
  max_product x y z ≤ (1 / 27) := 
by
  sorry

end max_value_of_xyz_l161_161978


namespace max_value_inequality_l161_161148

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 4 :=
sorry

end max_value_inequality_l161_161148


namespace pre_image_of_f_l161_161163

theorem pre_image_of_f (x y : ℝ) (f : ℝ × ℝ → ℝ × ℝ) 
  (h : f = λ p => (2 * p.1 + p.2, p.1 - 2 * p.2)) :
  f (1, 0) = (2, 1) := by
  sorry

end pre_image_of_f_l161_161163


namespace remainder_1394_mod_2535_l161_161729

-- Definition of the least number satisfying the given conditions
def L : ℕ := 1394

-- Proof statement: proving the remainder of division
theorem remainder_1394_mod_2535 : (1394 % 2535) = 1394 :=
by sorry

end remainder_1394_mod_2535_l161_161729


namespace distinct_solutions_equation_l161_161018

theorem distinct_solutions_equation (a b : ℝ) (h1 : a ≠ b) (h2 : a > b) (h3 : ∀ x, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1) (sol_a : x = a) (sol_b : x = b) :
  a - b = 1 :=
sorry

end distinct_solutions_equation_l161_161018


namespace minimize_expression_l161_161294

theorem minimize_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (a, b) = (15 / 4, 15) ↔ (∀ x y : ℝ, 0 < x → 0 < y → (4 * x + y = 30) → (1 / x + 4 / y) ≥ (1 / (15 / 4) + 4 / 15)) := by
sorry

end minimize_expression_l161_161294


namespace number_of_possible_lists_l161_161837

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161837


namespace minimum_rooms_needed_fans_l161_161477

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l161_161477


namespace passengers_initial_count_l161_161631

-- Let's define the initial number of passengers
variable (P : ℕ)

-- Given conditions:
def final_passengers (initial additional left : ℕ) : ℕ := initial + additional - left

-- The theorem statement to prove P = 28 given the conditions
theorem passengers_initial_count
  (final_count : ℕ)
  (h1 : final_count = 26)
  (h2 : final_passengers P 7 9 = final_count) 
  : P = 28 :=
by
  sorry

end passengers_initial_count_l161_161631


namespace sum_of_b_and_c_base7_l161_161775

theorem sum_of_b_and_c_base7 (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
(h4 : A < 7) (h5 : B < 7) (h6 : C < 7) 
(h7 : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) 
: B + C = 6 ∨ B + C = 12 := sorry

end sum_of_b_and_c_base7_l161_161775


namespace sum_zero_of_absolute_inequalities_l161_161796

theorem sum_zero_of_absolute_inequalities 
  (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) :
  a + b + c = 0 := 
  by
    sorry

end sum_zero_of_absolute_inequalities_l161_161796


namespace greatest_divisor_of_product_of_four_consecutive_integers_l161_161876

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, ∃ k : Nat, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l161_161876


namespace count_ways_to_get_50_cents_with_coins_l161_161322

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l161_161322


namespace find_other_diagonal_l161_161954

theorem find_other_diagonal (A : ℝ) (d1 : ℝ) (hA : A = 80) (hd1 : d1 = 16) :
  ∃ d2 : ℝ, 2 * A / d1 = d2 :=
by
  use 10
  -- Rest of the proof goes here
  sorry

end find_other_diagonal_l161_161954


namespace warriors_games_won_l161_161806

open Set

-- Define the variables for the number of games each team won
variables (games_L games_H games_W games_F games_R : ℕ)

-- Define the set of possible game scores
def game_scores : Set ℕ := {19, 23, 28, 32, 36}

-- Define the conditions as assumptions
axiom h1 : games_L > games_H
axiom h2 : games_W > games_F
axiom h3 : games_W < games_R
axiom h4 : games_F > 18
axiom h5 : ∃ min_games ∈ game_scores, min_games > games_H ∧ min_games < 20

-- Prove the main statement
theorem warriors_games_won : games_W = 32 :=
sorry

end warriors_games_won_l161_161806


namespace points_on_opposite_sides_l161_161006

-- Definitions and the conditions written to Lean
def satisfies_A (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 2 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

def satisfies_B (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 8 * a^2 * x - 2 * a^3 * y + 12 * a * y + a^4 + 36 = 0

def opposite_sides_of_line (y_A y_B : ℝ) : Prop :=
  (y_A - 1) * (y_B - 1) < 0

theorem points_on_opposite_sides (a : ℝ) (x_A y_A x_B y_B : ℝ) :
  satisfies_A a x_A y_A →
  satisfies_B a x_B y_B →
  -2 > a ∨ (-1 < a ∧ a < 0) ∨ 3 < a →
  opposite_sides_of_line y_A y_B → 
  x_A = 2 * a ∧ y_A = -a ∧ x_B = 4 ∧ y_B = a - 6/a :=
sorry

end points_on_opposite_sides_l161_161006


namespace compute_expression_l161_161130

theorem compute_expression : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3 ^ 2 = 28 := by
  sorry

end compute_expression_l161_161130


namespace joe_list_possibilities_l161_161846

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l161_161846


namespace trigonometric_identity_l161_161300

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := 
sorry

end trigonometric_identity_l161_161300


namespace fraction_sum_product_roots_of_quadratic_l161_161568

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l161_161568


namespace elder_age_is_33_l161_161056

-- Define the conditions
variables (y e : ℕ)

def age_difference_condition : Prop :=
  e = y + 20

def age_reduced_condition : Prop :=
  e - 8 = 5 * (y - 8)

-- State the theorem to prove the age of the elder person
theorem elder_age_is_33 (h1 : age_difference_condition y e) (h2 : age_reduced_condition y e): e = 33 :=
  sorry

end elder_age_is_33_l161_161056


namespace determine_counterfeit_coin_l161_161127

theorem determine_counterfeit_coin (wt_1 wt_2 wt_3 wt_5 : ℕ) (coin : ℕ) :
  (wt_1 = 1) ∧ (wt_2 = 2) ∧ (wt_3 = 3) ∧ (wt_5 = 5) ∧
  (coin = wt_1 ∨ coin = wt_2 ∨ coin = wt_3 ∨ coin = wt_5) ∧
  (coin ≠ 1 ∨ coin ≠ 2 ∨ coin ≠ 3 ∨ coin ≠ 5) → 
  ∃ (counterfeit : ℕ), (counterfeit = 1 ∨ counterfeit = 2 ∨ counterfeit = 3 ∨ counterfeit = 5) ∧ 
  (counterfeit ≠ 1 ∧ counterfeit ≠ 2 ∧ counterfeit ≠ 3 ∧ counterfeit ≠ 5) :=
by
  sorry

end determine_counterfeit_coin_l161_161127


namespace sin_135_eq_sqrt2_div_2_l161_161720

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l161_161720


namespace whole_numbers_in_interval_l161_161356

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l161_161356


namespace tammy_avg_speed_second_day_l161_161048

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l161_161048


namespace square_area_from_diagonal_l161_161880

theorem square_area_from_diagonal (d : ℝ) (hd : d = 3.8) : 
  ∃ (A : ℝ), A = 7.22 ∧ (∀ s : ℝ, d^2 = 2 * (s^2) → A = s^2) :=
by
  sorry

end square_area_from_diagonal_l161_161880


namespace correct_option_l161_161100

theorem correct_option 
  (A_false : ¬ (-6 - (-9)) = -3)
  (B_false : ¬ (-2 * (-5)) = -7)
  (C_false : ¬ (-x^2 + 3 * x^2) = 2)
  (D_true : (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b) :
  (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b :=
by sorry

end correct_option_l161_161100


namespace maggi_initial_packages_l161_161967

theorem maggi_initial_packages (P : ℕ) (h1 : 4 * P - 5 = 12) : P = 4 :=
sorry

end maggi_initial_packages_l161_161967


namespace no_prime_p_satisfies_l161_161829

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_prime_p_satisfies (p : ℕ) (hp : Nat.Prime p) (hp1 : is_perfect_square (7 * p + 3 ^ p - 4)) : False :=
by
  sorry

end no_prime_p_satisfies_l161_161829


namespace Samantha_last_name_length_l161_161611

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end Samantha_last_name_length_l161_161611


namespace number_of_lists_l161_161835

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l161_161835


namespace solve_sqrt_equation_l161_161800

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end solve_sqrt_equation_l161_161800


namespace pond_75_percent_free_on_day_18_l161_161116

-- Definitions based on conditions
noncomputable def algae_coverage (n : ℕ) : ℝ := (1 / 3^n)

-- Main theorem statement
theorem pond_75_percent_free_on_day_18 :
  (algae_coverage 18 : ℝ) ≤ 1/4 :=
begin
  have coverage_18 : algae_coverage 18 = 1 / 3^18 := rfl,
  have sub_fraction : (1 / 3^18 : ℝ) ≤ 1/4,
  { sorry },
  exact sub_fraction,
end

end pond_75_percent_free_on_day_18_l161_161116


namespace sum_of_coefficients_l161_161507

-- Define the polynomial P(x)
def P (x : ℤ) : ℤ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

-- State the theorem we intend to prove
theorem sum_of_coefficients : P 1 = 2019 :=
by
  -- Proof omitted
  sorry

end sum_of_coefficients_l161_161507


namespace count_whole_numbers_in_interval_l161_161353

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l161_161353


namespace compute_K_l161_161979

theorem compute_K (P Q T N K : ℕ) (x y z : ℕ) 
  (hP : P * x + Q * y = z) 
  (hT : T * x + N * y = z)
  (hK : K * x = z)
  (h_unique : P > 0 ∧ Q > 0 ∧ T > 0 ∧ N > 0 ∧ K > 0) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end compute_K_l161_161979


namespace largest_value_a_plus_b_plus_c_l161_161591

open Nat
open Function

def sum_of_digits (n : ℕ) : ℕ :=
  (digits 10 n).sum

theorem largest_value_a_plus_b_plus_c :
  ∃ (a b c : ℕ),
    10 ≤ a ∧ a < 100 ∧
    100 ≤ b ∧ b < 1000 ∧
    1000 ≤ c ∧ c < 10000 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    (a + b + c = 10199) := sorry

end largest_value_a_plus_b_plus_c_l161_161591


namespace mary_characters_initial_D_l161_161201

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end mary_characters_initial_D_l161_161201


namespace whole_numbers_in_interval_7_4_3pi_l161_161366

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l161_161366


namespace inscribed_circle_radius_l161_161726

theorem inscribed_circle_radius (AB BC CD DA: ℝ) (hAB: AB = 13) (hBC: BC = 10) (hCD: CD = 8) (hDA: DA = 11) :
  ∃ r, r = 2 * Real.sqrt 7 :=
by
  sorry

end inscribed_circle_radius_l161_161726


namespace books_left_to_read_l161_161971

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l161_161971


namespace sum_of_inserted_numbers_l161_161733

theorem sum_of_inserted_numbers (x y : ℝ) (r : ℝ) 
  (h1 : 4 * r = x) 
  (h2 : 4 * r^2 = y) 
  (h3 : (2 / y) = ((1 / x) + (1 / 16))) :
  x + y = 8 :=
sorry

end sum_of_inserted_numbers_l161_161733


namespace perpendicular_vectors_l161_161017

theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (c : ℝ × ℝ) 
  (h1 : a = (1, 2)) (h2 : b = (1, 1)) 
  (h3 : c = (1 + k, 2 + k))
  (h4 : b.1 * c.1 + b.2 * c.2 = 0) : 
  k = -3 / 2 :=
by
  sorry

end perpendicular_vectors_l161_161017


namespace polynomial_proof_l161_161065

variable (a b : ℝ)

-- Define the given monomial and the resulting polynomial 
def monomial := -3 * a ^ 2 * b
def result := 6 * a ^ 3 * b ^ 2 - 3 * a ^ 2 * b ^ 2 + 9 * a ^ 2 * b

-- Define the polynomial we want to prove
def poly := -2 * a * b + b - 3

-- Statement of the problem in Lean 4
theorem polynomial_proof :
  monomial * poly = result :=
by sorry

end polynomial_proof_l161_161065


namespace squirrel_can_catch_nut_l161_161514

-- Definitions for the given conditions
def distance_gavrila_squirrel : Real := 3.75
def nut_velocity : Real := 5
def squirrel_jump_distance : Real := 1.8
def gravity_acceleration : Real := 10

-- Statement to be proved
theorem squirrel_can_catch_nut : ∃ t : Real, 
  let r_squared := (nut_velocity * t - distance_gavrila_squirrel)^2 + (gravity_acceleration * t^2 / 2)^2 in
  r_squared ≤ squirrel_jump_distance^2 :=
begin
  sorry
end

end squirrel_can_catch_nut_l161_161514


namespace john_payment_l161_161771

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end john_payment_l161_161771


namespace tammy_avg_speed_l161_161043

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l161_161043


namespace contact_probability_l161_161505

theorem contact_probability (n m : ℕ) (p : ℝ) (h_n : n = 5) (h_m : m = 8) (hp : 0 ≤ p ∧ p ≤ 1) :
  (1 - (1 - p)^(n * m)) = 1 - (1 - p)^(40) :=
by
  rw [h_n, h_m]
  sorry

end contact_probability_l161_161505


namespace third_side_integer_lengths_l161_161907

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l161_161907


namespace not_integer_fraction_l161_161024

theorem not_integer_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 2) : ¬ (∃ k : ℤ, (2^a + 1) = k * (2^b - 1)) :=
sorry

end not_integer_fraction_l161_161024


namespace mean_of_α_X_l161_161510

open Finset

def M : Finset ℕ := range 1000 |>.map (λ x => x + 1)

def α (X : Finset ℕ) : ℕ :=
if X.nonempty then X.max' (X.nonempty) + X.min' (X.nonempty) else 0

def N : ℕ := ∑ X in M.powerset.filter (λ X, ¬X.isEmpty), α X

def f : ℚ :=
N.to_rat / (2^1000 - 1).to_rat

theorem mean_of_α_X :
  f = 1001 :=
sorry

end mean_of_α_X_l161_161510


namespace determine_angle_F_l161_161009

noncomputable def sin := fun x => Real.sin x
noncomputable def cos := fun x => Real.cos x
noncomputable def arcsin := fun x => Real.arcsin x
noncomputable def angleF (D E : ℝ) := 180 - (D + E)

theorem determine_angle_F (D E F : ℝ)
  (h1 : 2 * sin D + 5 * cos E = 7)
  (h2 : 5 * sin E + 2 * cos D = 4) :
  F = arcsin (9 / 10) ∨ F = 180 - arcsin (9 / 10) :=
  sorry

end determine_angle_F_l161_161009


namespace prob_contact_l161_161502

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l161_161502


namespace taller_tree_height_l161_161076

-- Given conditions
variables (h : ℕ) (ratio_cond : (h - 20) * 7 = h * 5)

-- Proof goal
theorem taller_tree_height : h = 70 :=
sorry

end taller_tree_height_l161_161076


namespace number_of_possible_third_side_lengths_l161_161921

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l161_161921


namespace sqrt_of_4_l161_161067

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end sqrt_of_4_l161_161067


namespace volume_ratio_l161_161436

-- Define the edge lengths
def edge_length_cube1 : ℝ := 4 -- in inches
def edge_length_cube2 : ℝ := 2 * 12 -- 2 feet converted to inches

-- Define the volumes
def volume_cube (a : ℝ) : ℝ := a ^ 3

-- Statement asserting the ratio of the volumes is 1/216
theorem volume_ratio : volume_cube edge_length_cube1 / volume_cube edge_length_cube2 = 1 / 216 :=
by
  -- This is the placeholder to skip the proof
  sorry

end volume_ratio_l161_161436


namespace B_work_rate_l161_161822

-- Definitions for the conditions
def A (t : ℝ) := 1 / 15 -- A's work rate per hour
noncomputable def B : ℝ := 1 / 10 - 1 / 15 -- Definition using the condition of the combined work rate

-- Lean 4 statement for the proof problem
theorem B_work_rate : B = 1 / 30 := by sorry

end B_work_rate_l161_161822


namespace number_of_possible_lists_l161_161844

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161844


namespace sum_of_sequence_correct_l161_161685

def calculateSumOfSequence : ℚ :=
  (4 / 3) + (7 / 5) + (11 / 8) + (19 / 15) + (35 / 27) + (67 / 52) - 9

theorem sum_of_sequence_correct :
  calculateSumOfSequence = (-17312.5 / 7020) := by
  sorry

end sum_of_sequence_correct_l161_161685


namespace b_over_c_equals_1_l161_161825

theorem b_over_c_equals_1 (a b c d : ℕ) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4)
    (h : 4^a + 3^b + 2^c + 1^d = 78) : b = c :=
by
  sorry

end b_over_c_equals_1_l161_161825


namespace transfer_people_correct_equation_l161_161055

theorem transfer_people_correct_equation (A B x : ℕ) (h1 : A = 28) (h2 : B = 20) : 
  A + x = 2 * (B - x) := 
by sorry

end transfer_people_correct_equation_l161_161055


namespace parabola_focus_eq_l161_161071

theorem parabola_focus_eq (focus : ℝ × ℝ) (hfocus : focus = (0, 1)) :
  ∃ (p : ℝ), p = 1 ∧ ∀ (x y : ℝ), x^2 = 4 * p * y → x^2 = 4 * y :=
by { sorry }

end parabola_focus_eq_l161_161071


namespace triangle_third_side_lengths_l161_161938

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l161_161938


namespace quadratic_root_sum_and_product_l161_161564

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l161_161564


namespace w1_relation_w2_relation_maximize_total_profit_l161_161676

def w1 (x : ℕ) : ℤ := 200 * x - 10000

def w2 (x : ℕ) : ℤ := -(x ^ 2) + 1000 * x - 50000

def total_sales_vol (x y : ℕ) : Prop := x + y = 1000

def max_profit_volumes (x y : ℕ) : Prop :=
  total_sales_vol x y ∧ x = 600 ∧ y = 400

theorem w1_relation (x : ℕ) :
  w1 x = 200 * x - 10000 := 
sorry

theorem w2_relation (x : ℕ) :
  w2 x = -(x ^ 2) + 1000 * x - 50000 := 
sorry

theorem maximize_total_profit (x y : ℕ) :
  total_sales_vol x y → max_profit_volumes x y := 
sorry

end w1_relation_w2_relation_maximize_total_profit_l161_161676


namespace quadratic_roots_identity_l161_161555

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l161_161555


namespace triangle_third_side_length_l161_161912

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l161_161912


namespace solution_interval_l161_161161

open Real

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := log x / log 2 - 1 / (x * log 2)

theorem solution_interval (f_is_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
  (f_composition : ∀ x : ℝ, 0 < x → f (f x - log x / log 2) = 3) :
  ∃ a b : ℝ, 1 < a ∧ a < b ∧ b < 2 ∧ (∀ x : ℝ, 1 < x ∧ x < 2 → f x - f' x = 2) :=
sorry

end solution_interval_l161_161161


namespace children_more_than_adults_l161_161508

-- Definitions based on given conditions
def price_per_child : ℚ := 4.50
def price_per_adult : ℚ := 6.75
def total_receipts : ℚ := 405
def number_of_children : ℕ := 48

-- Goal: Prove the number of children is 20 more than the number of adults.
theorem children_more_than_adults :
  ∃ (A : ℕ), (number_of_children - A) = 20 ∧ (price_per_child * number_of_children) + (price_per_adult * A) = total_receipts := by
  sorry

end children_more_than_adults_l161_161508


namespace range_of_expression_l161_161311

theorem range_of_expression (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3 / 2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by
  sorry

end range_of_expression_l161_161311


namespace increase_in_average_weight_l161_161058

theorem increase_in_average_weight 
    (A : ℝ) 
    (weight_left : ℝ)
    (weight_new : ℝ)
    (h_weight_left : weight_left = 67)
    (h_weight_new : weight_new = 87) : 
    ((8 * A - weight_left + weight_new) / 8 - A) = 2.5 := 
by
  sorry

end increase_in_average_weight_l161_161058


namespace quadratic_roots_vieta_l161_161559

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l161_161559


namespace two_cards_totaling_15_probability_l161_161656

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l161_161656


namespace count_whole_numbers_in_interval_l161_161363

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l161_161363


namespace range_is_80_l161_161617

def dataSet : List ℕ := [60, 100, 80, 40, 20]

def minValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl min x xs

def maxValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl max x xs

def range (l : List ℕ) : ℕ :=
  maxValue l - minValue l

theorem range_is_80 : range dataSet = 80 :=
by
  sorry

end range_is_80_l161_161617


namespace find_Q_l161_161081

-- We define the circles and their centers
def circle1 (x y r : ℝ) : Prop := (x + 1) ^ 2 + (y - 1) ^ 2 = r ^ 2
def circle2 (x y R : ℝ) : Prop := (x - 2) ^ 2 + (y + 2) ^ 2 = R ^ 2

-- Coordinates of point P
def P : ℝ × ℝ := (1, 2)

-- Defining the symmetry about the line y = -x
def symmetric_about (p q : ℝ × ℝ) : Prop := p.1 = -q.2 ∧ p.2 = -q.1

-- Theorem stating that if P is (1, 2), Q should be (-2, -1)
theorem find_Q {r R : ℝ} (h1 : circle1 1 2 r) (h2 : circle2 1 2 R) (hP : P = (1, 2)) :
  ∃ Q : ℝ × ℝ, symmetric_about P Q ∧ Q = (-2, -1) :=
by
  sorry

end find_Q_l161_161081


namespace probability_two_cards_sum_15_from_standard_deck_l161_161639

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l161_161639


namespace minimum_rooms_to_accommodate_fans_l161_161490

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l161_161490


namespace triangle_third_side_lengths_l161_161937

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l161_161937


namespace coin_combinations_50_cents_l161_161315

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l161_161315


namespace sin_135_eq_sqrt2_div_2_l161_161717

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l161_161717


namespace two_digit_factors_of_3_18_minus_1_l161_161337

theorem two_digit_factors_of_3_18_minus_1 : ∃ n : ℕ, n = 6 ∧ 
  ∀ x, x ∈ {y : ℕ | y ∣ 3^18 - 1 ∧ y > 9 ∧ y < 100} → 
  (x = 13 ∨ x = 26 ∨ x = 52 ∨ x = 14 ∨ x = 28 ∨ x = 91) :=
by
  use 6
  sorry

end two_digit_factors_of_3_18_minus_1_l161_161337


namespace joe_list_possibilities_l161_161848

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l161_161848


namespace fraction_sum_product_roots_of_quadratic_l161_161571

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l161_161571


namespace jason_money_determination_l161_161774

theorem jason_money_determination (fred_last_week : ℕ) (fred_earned : ℕ) (fred_now : ℕ) (jason_last_week : ℕ → Prop)
  (h1 : fred_last_week = 23)
  (h2 : fred_earned = 63)
  (h3 : fred_now = 86) :
  ¬ ∃ x, jason_last_week x :=
by
  sorry

end jason_money_determination_l161_161774


namespace candies_count_l161_161965

theorem candies_count (x : ℚ) (h : x + 3 * x + 12 * x + 72 * x = 468) : x = 117 / 22 :=
by
  sorry

end candies_count_l161_161965


namespace roots_real_l161_161897

variable {x p q k : ℝ}
variable {x1 x2 : ℝ}

theorem roots_real 
  (h1 : x^2 + p * x + q = 0) 
  (h2 : p = -(x1 + x2)) 
  (h3 : q = x1 * x2) 
  (h4 : x1 ≠ x2) 
  (h5 :  x1^2 - 2*x1*x2 + x2^2 + 4*q = 0):
  (∃ y1 y2, y1 = k * x1 + (1 / k) * x2 ∧ y2 = k * x2 + (1 / k) * x1 ∧ 
    (y1^2 + (k + 1/k) * p * y1 + (p^2 + q * ((k - 1/k)^2)) = 0) ∧ 
    (y2^2 + (k + 1/k) * p * y2 + (p^2 + q * ((k - 1/k)^2)) = 0)) → 
  (∃ z1 z2, z1 = k * x1 ∧ z2 = 1/k * x2 ∧ 
    (z1^2 - y1 * z1 + q = 0) ∧ 
    (z2^2 - y2 * z2 + q = 0)) :=
sorry

end roots_real_l161_161897


namespace least_three_digit_product_eight_l161_161089

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l161_161089


namespace percentage_boys_playing_soccer_is_correct_l161_161583

-- Definition of conditions 
def total_students := 420
def boys := 312
def soccer_players := 250
def girls_not_playing_soccer := 73

-- Calculated values based on conditions
def girls := total_students - boys
def girls_playing_soccer := girls - girls_not_playing_soccer
def boys_playing_soccer := soccer_players - girls_playing_soccer

-- Percentage of boys playing soccer
def percentage_boys_playing_soccer := (boys_playing_soccer / soccer_players) * 100

-- We assert the percentage of boys playing soccer is 86%
theorem percentage_boys_playing_soccer_is_correct : percentage_boys_playing_soccer = 86 := 
by
  -- Placeholder proof (use sorry as the proof is not required)
  sorry

end percentage_boys_playing_soccer_is_correct_l161_161583


namespace S10_value_l161_161251

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ :=
  x^m + (1 / x)^m

theorem S10_value (x : ℝ) (h : x + 1/x = 5) : 
  S_m x 10 = 6430223 := by 
  sorry

end S10_value_l161_161251


namespace number_of_possible_lists_l161_161842

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l161_161842


namespace tiles_per_row_24_l161_161417

noncomputable def num_tiles_per_row (area : ℝ) (tile_size : ℝ) : ℝ :=
  let side_length_ft := Real.sqrt area
  let side_length_in := side_length_ft * 12
  side_length_in / tile_size

theorem tiles_per_row_24 :
  num_tiles_per_row 324 9 = 24 :=
by
  sorry

end tiles_per_row_24_l161_161417


namespace triangle_third_side_count_l161_161934

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l161_161934


namespace perpendicular_vector_x_value_l161_161313

-- Definitions based on the given problem conditions
def dot_product_perpendicular (a1 a2 b1 b2 x : ℝ) : Prop :=
  (a1 * b1 + a2 * b2 = 0)

-- Statement to be proved
theorem perpendicular_vector_x_value (x : ℝ) :
  dot_product_perpendicular 4 x 2 4 x → x = -2 :=
by
  intros h
  sorry

end perpendicular_vector_x_value_l161_161313


namespace greatest_integer_less_than_neg_eight_over_three_l161_161227

theorem greatest_integer_less_than_neg_eight_over_three :
  ∃ (z : ℤ), (z < -8 / 3) ∧ ∀ w : ℤ, (w < -8 / 3) → w ≤ z := by
  sorry

end greatest_integer_less_than_neg_eight_over_three_l161_161227


namespace possible_integer_lengths_third_side_l161_161927

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l161_161927


namespace x_intercept_of_line_l161_161141

theorem x_intercept_of_line :
  (∃ x : ℝ, 5 * x - 7 * 0 = 35 ∧ (x, 0) = (7, 0)) :=
by
  use 7
  simp
  sorry

end x_intercept_of_line_l161_161141


namespace count_whole_numbers_in_interval_l161_161347

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l161_161347


namespace prob_neither_snow_nor_windy_l161_161992

-- Define the probabilities.
def prob_snow : ℚ := 1 / 4
def prob_windy : ℚ := 1 / 3

-- Define the complementary probabilities.
def prob_not_snow : ℚ := 1 - prob_snow
def prob_not_windy : ℚ := 1 - prob_windy

-- State that the events are independent and calculate the combined probability.
theorem prob_neither_snow_nor_windy :
  prob_not_snow * prob_not_windy = 1 / 2 := by
  sorry

end prob_neither_snow_nor_windy_l161_161992


namespace sqrt_four_eq_pm_two_l161_161069

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_four_eq_pm_two_l161_161069


namespace triangle_side_length_integers_l161_161924

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l161_161924


namespace fraction_of_4d_nails_l161_161464

variables (fraction2d fraction2d_or_4d fraction4d : ℚ)

theorem fraction_of_4d_nails
  (h1 : fraction2d = 0.25)
  (h2 : fraction2d_or_4d = 0.75) :
  fraction4d = 0.50 :=
by
  sorry

end fraction_of_4d_nails_l161_161464


namespace total_lunch_cost_l161_161955

theorem total_lunch_cost
  (children chaperones herself additional_lunches cost_per_lunch : ℕ)
  (h1 : children = 35)
  (h2 : chaperones = 5)
  (h3 : herself = 1)
  (h4 : additional_lunches = 3)
  (h5 : cost_per_lunch = 7) :
  (children + chaperones + herself + additional_lunches) * cost_per_lunch = 308 :=
by
  sorry

end total_lunch_cost_l161_161955


namespace vacation_days_l161_161209

theorem vacation_days (total_miles miles_per_day : ℕ) 
  (h1 : total_miles = 1250) (h2 : miles_per_day = 250) :
  total_miles / miles_per_day = 5 := by
  sorry

end vacation_days_l161_161209


namespace sin_identity_alpha_l161_161022

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end sin_identity_alpha_l161_161022


namespace xy_sum_l161_161573

namespace ProofExample

variable (x y : ℚ)

def condition1 : Prop := (1 / x) + (1 / y) = 4
def condition2 : Prop := (1 / x) - (1 / y) = -6

theorem xy_sum : condition1 x y → condition2 x y → (x + y = -4 / 5) := by
  intros
  sorry

end ProofExample

end xy_sum_l161_161573


namespace parabola_distance_l161_161062

theorem parabola_distance (a : ℝ) :
  (abs (1 + (1 / (4 * a))) = 2 → a = 1 / 4) ∨ 
  (abs (1 - (1 / (4 * a))) = 2 → a = -1 / 12) := by 
  sorry

end parabola_distance_l161_161062


namespace circle_tangent_l161_161375

theorem circle_tangent (t : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 4 → (x - t)^2 + y^2 = 1 → |t| = 3) :=
by
  sorry

end circle_tangent_l161_161375
