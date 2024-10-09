import Mathlib

namespace katie_ds_games_l1704_170453

theorem katie_ds_games (new_friends_games old_friends_games total_friends_games katie_games : ℕ) 
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_friends_games = 141)
  (h4 : total_friends_games = new_friends_games + old_friends_games + katie_games) :
  katie_games = 0 :=
by
  sorry

end katie_ds_games_l1704_170453


namespace part1_part2_l1704_170485

-- Part (1)
theorem part1 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (arithmetic_seq : ∀ n, a_n (n+1) = a_n n + d)
  (S1_eq : S_n 1 = 5)
  (S2_eq : S_n 2 = 18) :
  ∀ n, a_n n = 3 * n + 2 := by
  sorry

-- Part (2)
theorem part2 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (geometric_seq : ∃ q, ∀ n, a_n (n+1) = q * a_n n)
  (S1_eq : S_n 1 = 3)
  (S2_eq : S_n 2 = 15) :
  ∀ n, S_n n = (3^(n+2) - 6 * n - 9) / 4 := by
  sorry

end part1_part2_l1704_170485


namespace problem_1_problem_2_problem_3_problem_4_l1704_170428

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by
  sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by
  sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by
  sorry

-- Problem 4
theorem problem_4 : (-19 + 15 / 16) * 8 = -159 + 1 / 2 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l1704_170428


namespace johnny_ran_4_times_l1704_170460

-- Block length is 200 meters
def block_length : ℕ := 200

-- Distance run by Johnny is Johnny's running times times the block length
def johnny_distance (J : ℕ) : ℕ := J * block_length

-- Distance run by Mickey is half of Johnny's running times times the block length
def mickey_distance (J : ℕ) : ℕ := (J / 2) * block_length

-- Average distance run by Johnny and Mickey is 600 meters
def average_distance_condition (J : ℕ) : Prop :=
  ((johnny_distance J + mickey_distance J) / 2) = 600

-- We are to prove that Johnny ran 4 times based on the condition
theorem johnny_ran_4_times (J : ℕ) (h : average_distance_condition J) : J = 4 :=
sorry

end johnny_ran_4_times_l1704_170460


namespace minimize_y_at_x_l1704_170495

noncomputable def minimize_y (a b x : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + 2 * (a - b) * x

theorem minimize_y_at_x (a b : ℝ) :
  ∃ x : ℝ, minimize_y a b x = minimize_y a b (b / 2) := by
  sorry

end minimize_y_at_x_l1704_170495


namespace sum_of_final_two_numbers_l1704_170479

theorem sum_of_final_two_numbers (x y T : ℕ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by
  sorry

end sum_of_final_two_numbers_l1704_170479


namespace roots_eq_two_iff_a_gt_neg1_l1704_170424

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l1704_170424


namespace min_value_geometric_sequence_l1704_170475

-- Definition for conditions and problem setup
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

-- Given data
variable (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_sum : a 2015 + a 2017 = Real.pi)

-- Goal statement
theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = (Real.pi^2) / 2 ∧ (
    ∀ a : ℕ → ℝ, 
    is_geometric_sequence a → 
    a 2015 + a 2017 = Real.pi → 
    a 2016 * (a 2014 + a 2018) ≥ (Real.pi^2) / 2
  ) :=
sorry

end min_value_geometric_sequence_l1704_170475


namespace proof_equiv_l1704_170438

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 6 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem proof_equiv (x : ℝ) : f (g x) - g (f x) = 6 * x ^ 2 - 12 * x + 9 := by
  sorry

end proof_equiv_l1704_170438


namespace solve_quadratic_l1704_170452

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, 
  (-6) * x1^2 + 11 * x1 - 3 = 0 ∧ (-6) * x2^2 + 11 * x2 - 3 = 0 ∧ x1 = 1.5 ∧ x2 = 1 / 3 :=
by
  sorry

end solve_quadratic_l1704_170452


namespace usual_time_to_reach_school_l1704_170413

variable (R T : ℝ)
variable (h : T * R = (T - 4) * (7/6 * R))

theorem usual_time_to_reach_school (h : T * R = (T - 4) * (7/6 * R)) : T = 28 := by
  sorry

end usual_time_to_reach_school_l1704_170413


namespace perimeter_square_l1704_170482

-- Definition of the side length
def side_length : ℝ := 9

-- Definition of the perimeter calculation
def perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem stating that the perimeter of a square with side length 9 cm is 36 cm
theorem perimeter_square : perimeter side_length = 36 := 
by sorry

end perimeter_square_l1704_170482


namespace grey_pairs_coincide_l1704_170491

theorem grey_pairs_coincide (h₁ : 4 = orange_count / 2) 
                                (h₂ : 6 = green_count / 2)
                                (h₃ : 9 = grey_count / 2)
                                (h₄ : 3 = orange_pairs)
                                (h₅ : 4 = green_pairs)
                                (h₆ : 1 = orange_grey_pairs) :
    grey_pairs = 6 := by
  sorry

noncomputable def half_triangle_counts : (ℕ × ℕ × ℕ) := (4, 6, 9)

noncomputable def triangle_pairs : (ℕ × ℕ × ℕ) := (3, 4, 1)

noncomputable def prove_grey_pairs (orange_count green_count grey_count : ℕ)
                                   (orange_pairs green_pairs orange_grey_pairs : ℕ) : ℕ :=
  sorry

end grey_pairs_coincide_l1704_170491


namespace sum_of_w_l1704_170477

def g (y : ℝ) : ℝ := (2 * y)^3 - 2 * (2 * y) + 5

theorem sum_of_w (w1 w2 w3 : ℝ)
  (hw1 : g (2 * w1) = 13)
  (hw2 : g (2 * w2) = 13)
  (hw3 : g (2 * w3) = 13) :
  w1 + w2 + w3 = -1 / 4 :=
sorry

end sum_of_w_l1704_170477


namespace arithmetic_sequence_sum_l1704_170493

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l1704_170493


namespace smartphone_customers_l1704_170467

theorem smartphone_customers (k : ℝ) (p1 p2 c1 c2 : ℝ)
  (h₁ : p1 * c1 = k)
  (h₂ : 20 = p1)
  (h₃ : 200 = c1)
  (h₄ : 400 = c2) :
  p2 * c2 = k  → p2 = 10 :=
by
  sorry

end smartphone_customers_l1704_170467


namespace lcm_18_35_is_630_l1704_170481

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l1704_170481


namespace alina_sent_fewer_messages_l1704_170445

-- Definitions based on conditions
def messages_lucia_day1 : Nat := 120
def messages_lucia_day2 : Nat := 1 / 3 * messages_lucia_day1
def messages_lucia_day3 : Nat := messages_lucia_day1
def messages_total : Nat := 680

-- Def statement for Alina's messages on the first day, which we need to find as 100
def messages_alina_day1 : Nat := 100

-- Condition checks
def condition_alina_day2 : Prop := 2 * messages_alina_day1 = 2 * 100
def condition_alina_day3 : Prop := messages_alina_day1 = 100
def condition_total_messages : Prop := 
  messages_alina_day1 + messages_lucia_day1 +
  2 * messages_alina_day1 + messages_lucia_day2 +
  messages_alina_day1 + messages_lucia_day1 = messages_total

-- Theorem statement
theorem alina_sent_fewer_messages :
  messages_lucia_day1 - messages_alina_day1 = 20 :=
by
  -- Ensure the conditions hold
  have h1 : messages_alina_day1 = 100 := by sorry
  have h2 : condition_alina_day2 := by sorry
  have h3 : condition_alina_day3 := by sorry
  have h4 : condition_total_messages := by sorry
  -- Prove the theorem
  sorry

end alina_sent_fewer_messages_l1704_170445


namespace probability_problem_l1704_170448

def ang_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def ben_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def jasmin_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]

def boxes : Fin 6 := sorry  -- represents 6 empty boxes
def white_restriction (box : Fin 6) : Prop := box ≠ 0  -- white block can't be in the first box

def probability_at_least_one_box_three_same_color : ℚ := 1 / 72  -- The given probability

theorem probability_problem (p q : ℕ) 
  (hpq_coprime : Nat.gcd p q = 1) 
  (hprob_eq : probability_at_least_one_box_three_same_color = p / q) :
  p + q = 73 :=
sorry

end probability_problem_l1704_170448


namespace perfect_square_digits_l1704_170410

theorem perfect_square_digits (x y : ℕ) (h_ne_zero : x ≠ 0) (h_perfect_square : ∀ n: ℕ, n ≥ 1 → ∃ k: ℕ, (10^(n + 2) * x + 10^(n + 1) * 6 + 10 * y + 4) = k^2) :
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0) :=
sorry

end perfect_square_digits_l1704_170410


namespace hypotenuse_length_l1704_170469

-- Define the properties of the right-angled triangle
variables (α β γ : ℝ) (a b c : ℝ)
-- Right-angled triangle condition
axiom right_angled_triangle : α = 30 ∧ β = 60 ∧ γ = 90 → c = 2 * a

-- Given side opposite 30° angle is 6 cm
axiom side_opposite_30_is_6cm : a = 6

-- Proof that hypotenuse is 12 cm
theorem hypotenuse_length : c = 12 :=
by 
  sorry

end hypotenuse_length_l1704_170469


namespace cost_price_percentage_l1704_170454

/-- The cost price (CP) as a percentage of the marked price (MP) given 
that the discount is 18% and the gain percent is 28.125%. -/
theorem cost_price_percentage (MP CP : ℝ) (h1 : CP / MP = 0.64) : 
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l1704_170454


namespace unique_f_l1704_170418

def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 10^10 }

noncomputable def f : ℕ → ℕ := sorry

axiom f_cond (x : ℕ) (hx : x ∈ S) :
  f (x + 1) % (10^10) = (f (f x) + 1) % (10^10)

axiom f_boundary :
  f (10^10 + 1) % (10^10) = f 1

theorem unique_f (x : ℕ) (hx : x ∈ S) :
  f x % (10^10) = x % (10^10) :=
sorry

end unique_f_l1704_170418


namespace problem_proof_l1704_170455

variable (A B C a b c : ℝ)
variable (ABC_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (sides_opposite : a = (b * sin A / sin B) ∧ b = (a * sin B / sin A))
variable (cos_eq : b + b * cos A = a * cos B)

theorem problem_proof :
  (A = 2 * B ∧ (π / 6 < B ∧ B < π / 4) ∧ a^2 = b^2 + b * c) :=
  sorry

end problem_proof_l1704_170455


namespace smallest_sector_angle_l1704_170457

-- Definitions and conditions identified in step a.

def a1 (d : ℕ) : ℕ := (48 - 14 * d) / 2

-- Proof statement
theorem smallest_sector_angle : ∀ d : ℕ, d ≥ 0 → d ≤ 3 → 15 * (a1 d + (a1 d + 14 * d)) = 720 → (a1 d = 3) :=
by
  sorry

end smallest_sector_angle_l1704_170457


namespace value_of_fraction_l1704_170471

theorem value_of_fraction : (121^2 - 112^2) / 9 = 233 := by
  -- use the difference of squares property
  sorry

end value_of_fraction_l1704_170471


namespace a_2_correct_l1704_170439

noncomputable def a_2_value (a a1 a2 a3 : ℝ) : Prop :=
∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3

theorem a_2_correct (a a1 a2 a3 : ℝ) (h : a_2_value a a1 a2 a3) : a2 = 6 :=
sorry

end a_2_correct_l1704_170439


namespace f_x_plus_1_l1704_170474

-- Given function definition
def f (x : ℝ) := x^2

-- Statement to prove
theorem f_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x + 1 := 
by
  rw [f]
  -- This simplifies to:
  -- (x + 1)^2 = x^2 + 2 * x + 1
  sorry

end f_x_plus_1_l1704_170474


namespace range_of_a_l1704_170480

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x + 5| > a) → a < 8 := by
  sorry

end range_of_a_l1704_170480


namespace fraction_remaining_distance_l1704_170446

theorem fraction_remaining_distance
  (total_distance : ℕ)
  (first_stop_fraction : ℚ)
  (remaining_distance_after_second_stop : ℕ)
  (fraction_between_stops : ℚ) :
  total_distance = 280 →
  first_stop_fraction = 1/2 →
  remaining_distance_after_second_stop = 105 →
  (fraction_between_stops * (total_distance - (first_stop_fraction * total_distance)) + remaining_distance_after_second_stop = (total_distance - (first_stop_fraction * total_distance))) →
  fraction_between_stops = 1/4 :=
by
  sorry

end fraction_remaining_distance_l1704_170446


namespace tan_addition_example_l1704_170402

theorem tan_addition_example (x : ℝ) (h : Real.tan x = 1/3) : 
  Real.tan (x + π/3) = 2 + 5 * Real.sqrt 3 / 3 := 
by 
  sorry

end tan_addition_example_l1704_170402


namespace scout_weekend_earnings_l1704_170401

-- Definitions for conditions
def base_pay_per_hour : ℝ := 10.00
def tip_saturday : ℝ := 5.00
def tip_sunday_low : ℝ := 3.00
def tip_sunday_high : ℝ := 7.00
def transportation_cost_per_delivery : ℝ := 1.00
def hours_worked_saturday : ℝ := 6
def deliveries_saturday : ℝ := 5
def hours_worked_sunday : ℝ := 8
def deliveries_sunday : ℝ := 10
def deliveries_sunday_low_tip : ℝ := 5
def deliveries_sunday_high_tip : ℝ := 5
def holiday_multiplier : ℝ := 2

-- Calculation of total earnings for the weekend after transportation costs
theorem scout_weekend_earnings : 
  let base_pay_saturday := hours_worked_saturday * base_pay_per_hour
  let tips_saturday := deliveries_saturday * tip_saturday
  let transportation_costs_saturday := deliveries_saturday * transportation_cost_per_delivery
  let total_earnings_saturday := base_pay_saturday + tips_saturday - transportation_costs_saturday

  let base_pay_sunday := hours_worked_sunday * base_pay_per_hour * holiday_multiplier
  let tips_sunday := deliveries_sunday_low_tip * tip_sunday_low + deliveries_sunday_high_tip * tip_sunday_high
  let transportation_costs_sunday := deliveries_sunday * transportation_cost_per_delivery
  let total_earnings_sunday := base_pay_sunday + tips_sunday - transportation_costs_sunday

  let total_earnings_weekend := total_earnings_saturday + total_earnings_sunday

  total_earnings_weekend = 280.00 :=
by
  -- Add detailed proof here
  sorry

end scout_weekend_earnings_l1704_170401


namespace calc_x6_plus_inv_x6_l1704_170483

theorem calc_x6_plus_inv_x6 (x : ℝ) (hx : x + (1 / x) = 7) : x^6 + (1 / x^6) = 103682 := by
  sorry

end calc_x6_plus_inv_x6_l1704_170483


namespace quadratic_distinct_zeros_l1704_170490

theorem quadratic_distinct_zeros (m : ℝ) : 
  (x^2 + m * x + (m + 3)) = 0 → 
  (0 < m^2 - 4 * (m + 3)) ↔ (m < -2) ∨ (m > 6) :=
sorry

end quadratic_distinct_zeros_l1704_170490


namespace triangle_largest_angle_and_type_l1704_170472

theorem triangle_largest_angle_and_type
  (a b c : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 4 * k) 
  (h3 : b = 3 * k) 
  (h4 : c = 2 * k) 
  (h5 : a ≥ b) 
  (h6 : a ≥ c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := 
by
  -- Replace 'by' with 'sorry' to denote that the proof should go here
  sorry

end triangle_largest_angle_and_type_l1704_170472


namespace prove_P_plus_V_eq_zero_l1704_170478

variable (P Q R S T U V : ℤ)

-- Conditions in Lean
def sequence_conditions (P Q R S T U V : ℤ) :=
  S = 7 ∧
  P + Q + R = 27 ∧
  Q + R + S = 27 ∧
  R + S + T = 27 ∧
  S + T + U = 27 ∧
  T + U + V = 27 ∧
  U + V + P = 27

-- Assertion that needs to be proved
theorem prove_P_plus_V_eq_zero (P Q R S T U V : ℤ) (h : sequence_conditions P Q R S T U V) : 
  P + V = 0 := by
  sorry

end prove_P_plus_V_eq_zero_l1704_170478


namespace parallel_lines_coplanar_l1704_170430

axiom Plane : Type
axiom Point : Type
axiom Line : Type

axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

axiom α : Plane
axiom β : Plane

axiom in_plane (p : Point) (π : Plane) : Prop
axiom parallel_plane (π1 π2 : Plane) : Prop
axiom parallel_line (l1 l2 : Line) : Prop
axiom line_through (P Q : Point) : Line
axiom coplanar (P Q R S : Point) : Prop

-- Conditions
axiom A_in_α : in_plane A α
axiom C_in_α : in_plane C α
axiom B_in_β : in_plane B β
axiom D_in_β : in_plane D β
axiom α_parallel_β : parallel_plane α β

-- Statement
theorem parallel_lines_coplanar :
  parallel_line (line_through A C) (line_through B D) ↔ coplanar A B C D :=
sorry

end parallel_lines_coplanar_l1704_170430


namespace courtyard_width_l1704_170466

theorem courtyard_width 
  (length_of_courtyard : ℝ) 
  (num_paving_stones : ℕ) 
  (length_of_stone width_of_stone : ℝ) 
  (total_area_stone : ℝ) 
  (W : ℝ) : 
  length_of_courtyard = 40 →
  num_paving_stones = 132 →
  length_of_stone = 2.5 →
  width_of_stone = 2 →
  total_area_stone = 660 →
  40 * W = 660 →
  W = 16.5 :=
by
  intros
  sorry

end courtyard_width_l1704_170466


namespace problem_solution_l1704_170462

theorem problem_solution (a b c : ℝ) (h : (a / (36 - a)) + (b / (45 - b)) + (c / (54 - c)) = 8) :
    (4 / (36 - a)) + (5 / (45 - b)) + (6 / (54 - c)) = 11 / 9 := 
by
  sorry

end problem_solution_l1704_170462


namespace polynomial_expansion_correct_l1704_170432

def polynomial1 (x : ℝ) := 3 * x^2 - 4 * x + 3
def polynomial2 (x : ℝ) := -2 * x^2 + 3 * x - 4

theorem polynomial_expansion_correct {x : ℝ} :
  (polynomial1 x) * (polynomial2 x) = -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 :=
by
  sorry

end polynomial_expansion_correct_l1704_170432


namespace missing_bricks_is_26_l1704_170436

-- Define the number of bricks per row and the number of rows
def bricks_per_row : Nat := 10
def number_of_rows : Nat := 6

-- Calculate the total number of bricks for a fully completed wall
def total_bricks_full_wall : Nat := bricks_per_row * number_of_rows

-- Assume the number of bricks currently present
def bricks_currently_present : Nat := total_bricks_full_wall - 26

-- Define a function that calculates the number of missing bricks
def number_of_missing_bricks (total_bricks : Nat) (bricks_present : Nat) : Nat :=
  total_bricks - bricks_present

-- Prove that the number of missing bricks is 26
theorem missing_bricks_is_26 : 
  number_of_missing_bricks total_bricks_full_wall bricks_currently_present = 26 :=
by
  sorry

end missing_bricks_is_26_l1704_170436


namespace solve_inequality_l1704_170464

open Set

theorem solve_inequality (x : ℝ) :
  { x | (x^2 - 9) / (x^2 - 16) > 0 } = (Iio (-4)) ∪ (Ioi 4) :=
by
  sorry

end solve_inequality_l1704_170464


namespace average_minutes_correct_l1704_170494

variable (s : ℕ)
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2

def minutes_sixth_graders := 18 * sixth_graders s
def minutes_seventh_graders := 20 * seventh_graders s
def minutes_eighth_graders := 22 * eighth_graders s

def total_minutes := minutes_sixth_graders s + minutes_seventh_graders s + minutes_eighth_graders s
def total_students := sixth_graders s + seventh_graders s + eighth_graders s

def average_minutes := total_minutes s / total_students s

theorem average_minutes_correct : average_minutes s = 170 / 9 := sorry

end average_minutes_correct_l1704_170494


namespace train_speed_l1704_170484

theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 125) (h_bridge : length_bridge = 250) (h_time : time = 30) :
    (length_train + length_bridge) / time * 3.6 = 45 := by
  sorry

end train_speed_l1704_170484


namespace option_D_correct_l1704_170461

variables (Line : Type) (Plane : Type)
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (perpendicular_planes : Plane → Plane → Prop)

theorem option_D_correct (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicular_planes α β :=
sorry

end option_D_correct_l1704_170461


namespace largest_n_l1704_170451

-- Define the condition that n, x, y, z are positive integers
def conditions (n x y z : ℕ) := (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < n) 

-- Formulate the main theorem
theorem largest_n (x y z : ℕ) : 
  conditions 8 x y z →
  8^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 10 :=
by 
  sorry

end largest_n_l1704_170451


namespace no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l1704_170433

variables {a b : ℝ} (spells : list (ℝ × ℝ)) (infinite_spells : ℕ → ℝ × ℝ)

-- Condition: 0 < a < b
def valid_spell (spell : ℝ × ℝ) : Prop := 0 < spell.1 ∧ spell.1 < spell.2

-- Question a: Finite set of spells, prove that no spell set exists such that the second wizard can guarantee a win.
theorem no_finite_spells_guarantee_second_wizard_win :
  (∀ spell ∈ spells, valid_spell spell) →
  ¬(∃ (strategy : ℕ → ℝ × ℝ), ∀ n, valid_spell (strategy n) ∧ ∃ k, n < k ∧ valid_spell (strategy k)) :=
sorry

-- Question b: Infinite set of spells, prove that there exists a spell set such that the second wizard can guarantee a win.
theorem exists_infinite_spells_guarantee_second_wizard_win :
  (∀ n, valid_spell (infinite_spells n)) →
  ∃ (strategy : ℕ → ℝ × ℝ), ∀ n, ∃ k, n < k ∧ valid_spell (strategy k) :=
sorry

end no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l1704_170433


namespace question_l1704_170499

variable (a : ℝ)

def condition_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3

def condition_q (a : ℝ) : Prop := ∀ (x y : ℝ) , x > y → (5 - 2 * a)^x < (5 - 2 * a)^y

theorem question (h1 : condition_p a ∨ condition_q a)
                (h2 : ¬ (condition_p a ∧ condition_q a)) : a = 2 ∨ a ≥ 5 / 2 :=
sorry

end question_l1704_170499


namespace units_digit_of_square_l1704_170444

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := 
by 
  sorry

end units_digit_of_square_l1704_170444


namespace cos_angle_equiv_370_l1704_170473

open Real

noncomputable def find_correct_n : ℕ :=
  sorry

theorem cos_angle_equiv_370 (n : ℕ) (h : 0 ≤ n ∧ n ≤ 180) : cos (n * π / 180) = cos (370 * π / 180) → n = 10 :=
by
  sorry

end cos_angle_equiv_370_l1704_170473


namespace velocity_zero_at_t_eq_2_l1704_170487

noncomputable def motion_equation (t : ℝ) : ℝ := -4 * t^3 + 48 * t

theorem velocity_zero_at_t_eq_2 :
  (exists t : ℝ, t > 0 ∧ deriv (motion_equation) t = 0) :=
by
  sorry

end velocity_zero_at_t_eq_2_l1704_170487


namespace min_value_of_quadratic_l1704_170450

theorem min_value_of_quadratic (x : ℝ) : ∃ y, y = x^2 + 14*x + 20 ∧ ∀ z, z = x^2 + 14*x + 20 → z ≥ -29 :=
by
  sorry

end min_value_of_quadratic_l1704_170450


namespace find_ab_l1704_170417

-- Define the conditions and the goal
theorem find_ab (a b : ℝ) (h1 : a^2 + b^2 = 26) (h2 : a + b = 7) : ab = 23 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end find_ab_l1704_170417


namespace sqrt_difference_l1704_170404

theorem sqrt_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 :=
sorry

end sqrt_difference_l1704_170404


namespace solve_fractional_eq_l1704_170435

theorem solve_fractional_eq {x : ℚ} : (3 / (x - 1)) = (1 / x) ↔ x = -1/2 :=
by sorry

end solve_fractional_eq_l1704_170435


namespace sqrt_15_minus_1_range_l1704_170419

theorem sqrt_15_minus_1_range (h : 9 < 15 ∧ 15 < 16) : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := 
  sorry

end sqrt_15_minus_1_range_l1704_170419


namespace find_initial_books_l1704_170431

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

end find_initial_books_l1704_170431


namespace fran_speed_calculation_l1704_170416

theorem fran_speed_calculation:
  let Joann_speed := 15
  let Joann_time := 5
  let Fran_time := 4
  let Fran_speed := (Joann_speed * Joann_time) / Fran_time
  Fran_speed = 18.75 := by
  sorry

end fran_speed_calculation_l1704_170416


namespace sum_of_g_35_l1704_170470

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 - 3
noncomputable def g (y : ℝ) : ℝ := y^2 + y + 1

theorem sum_of_g_35 : g 35 = 21 := 
by
  sorry

end sum_of_g_35_l1704_170470


namespace find_y_l1704_170422

theorem find_y 
  (y : ℝ) 
  (h1 : (y^2 - 11 * y + 24) / (y - 3) + (2 * y^2 + 7 * y - 18) / (2 * y - 3) = -10)
  (h2 : y ≠ 3)
  (h3 : y ≠ 3 / 2) : 
  y = -4 := 
sorry

end find_y_l1704_170422


namespace minimum_n_divisible_20_l1704_170465

theorem minimum_n_divisible_20 :
  ∃ (n : ℕ), (∀ (l : List ℕ), l.length = n → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0) ∧ 
  (∀ m, m < n → ¬(∀ (l : List ℕ), l.length = m → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0)) := 
⟨9, 
  by sorry, 
  by sorry⟩

end minimum_n_divisible_20_l1704_170465


namespace problem_statement_l1704_170442

theorem problem_statement (A B : ℤ) (h1 : A * B = 15) (h2 : -7 * B - 8 * A = -94) : AB + A = 20 := by
  sorry

end problem_statement_l1704_170442


namespace seeder_path_length_l1704_170488

theorem seeder_path_length (initial_grain : ℤ) (decrease_percent : ℝ) (seeding_rate : ℝ) (width : ℝ) 
  (H_initial_grain : initial_grain = 250) 
  (H_decrease_percent : decrease_percent = 14 / 100) 
  (H_seeding_rate : seeding_rate = 175) 
  (H_width : width = 4) :
  (initial_grain * decrease_percent / seeding_rate) * 10000 / width = 500 := 
by 
  sorry

end seeder_path_length_l1704_170488


namespace find_m_plus_n_l1704_170498

variable (U : Set ℝ) (A : Set ℝ) (CUA : Set ℝ) (m n : ℝ)
  -- Condition 1: The universal set U is the set of all real numbers
  (hU : U = Set.univ)
  -- Condition 2: A is defined as the set of all x such that (x - 1)(x - m) > 0
  (hA : A = { x : ℝ | (x - 1) * (x - m) > 0 })
  -- Condition 3: The complement of A in U is [-1, -n]
  (hCUA : CUA = { x : ℝ | x ∈ U ∧ x ∉ A } ∧ CUA = Icc (-1) (-n))

theorem find_m_plus_n : m + n = -2 :=
  sorry 

end find_m_plus_n_l1704_170498


namespace tim_total_spent_l1704_170440

variable (lunch_cost : ℝ)
variable (tip_percentage : ℝ)
variable (total_spent : ℝ)

theorem tim_total_spent (h_lunch_cost : lunch_cost = 60.80)
                        (h_tip_percentage : tip_percentage = 0.20)
                        (h_total_spent : total_spent = lunch_cost + (tip_percentage * lunch_cost)) :
                        total_spent = 72.96 :=
sorry

end tim_total_spent_l1704_170440


namespace sum_of_first_and_third_l1704_170412

theorem sum_of_first_and_third :
  ∀ (A B C : ℕ),
  A + B + C = 330 →
  A = 2 * B →
  C = A / 3 →
  B = 90 →
  A + C = 240 :=
by
  intros A B C h1 h2 h3 h4
  sorry

end sum_of_first_and_third_l1704_170412


namespace fractional_pizza_eaten_after_six_trips_l1704_170476

def pizza_eaten : ℚ := (1/3) * (1 - (2/3)^6) / (1 - 2/3)

theorem fractional_pizza_eaten_after_six_trips : pizza_eaten = 665 / 729 :=
by
  -- proof will go here
  sorry

end fractional_pizza_eaten_after_six_trips_l1704_170476


namespace complement_intersection_l1704_170406

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {1, 2, 3})
variable (hB : B = {2, 3, 4})

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 4} :=
by
  sorry

end complement_intersection_l1704_170406


namespace find_cost_price_l1704_170437

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l1704_170437


namespace initial_sugar_weight_l1704_170447

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end initial_sugar_weight_l1704_170447


namespace color_5x5_grid_excluding_two_corners_l1704_170409

-- Define the total number of ways to color a 5x5 grid with each row and column having exactly one colored cell
def total_ways : Nat := 120

-- Define the number of ways to color a 5x5 grid excluding one specific corner cell such that each row and each column has exactly one colored cell
def ways_excluding_one_corner : Nat := 96

-- Prove the number of ways to color the grid excluding two specific corner cells is 78
theorem color_5x5_grid_excluding_two_corners : total_ways - (ways_excluding_one_corner + ways_excluding_one_corner - 6) = 78 := by
  -- We state our given conditions directly as definitions
  -- Now we state our theorem explicitly and use the correct answer we derived
  sorry

end color_5x5_grid_excluding_two_corners_l1704_170409


namespace problem_D_l1704_170434

variable (f : ℕ → ℝ)

-- Function condition: If f(k) ≥ k^2, then f(k+1) ≥ (k+1)^2
axiom f_property (k : ℕ) (hk : f k ≥ k^2) : f (k + 1) ≥ (k + 1)^2

theorem problem_D (hf4 : f 4 ≥ 25) : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_D_l1704_170434


namespace lilly_fish_l1704_170411

-- Define the conditions
def total_fish : ℕ := 18
def rosy_fish : ℕ := 8

-- Statement: Prove that Lilly has 10 fish
theorem lilly_fish (h1 : total_fish = 18) (h2 : rosy_fish = 8) :
  total_fish - rosy_fish = 10 :=
by sorry

end lilly_fish_l1704_170411


namespace Emily_total_cost_l1704_170489

theorem Emily_total_cost :
  let cost_curtains := 2 * 30
  let cost_prints := 9 * 15
  let installation_cost := 50
  let total_cost := cost_curtains + cost_prints + installation_cost
  total_cost = 245 := by
{
 sorry
}

end Emily_total_cost_l1704_170489


namespace max_rabbits_with_traits_l1704_170443

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l1704_170443


namespace average_infected_per_round_is_nine_l1704_170426

theorem average_infected_per_round_is_nine (x : ℝ) :
  1 + x + x * (1 + x) = 100 → x = 9 :=
by {
  sorry
}

end average_infected_per_round_is_nine_l1704_170426


namespace additional_discount_percentage_l1704_170405

theorem additional_discount_percentage
  (MSRP : ℝ)
  (p : ℝ)
  (d : ℝ)
  (sale_price : ℝ)
  (H1 : MSRP = 45.0)
  (H2 : p = 0.30)
  (H3 : d = MSRP - (p * MSRP))
  (H4 : d = 31.50)
  (H5 : sale_price = 25.20) :
  sale_price = d - (0.20 * d) :=
by
  sorry

end additional_discount_percentage_l1704_170405


namespace quadratic_solution_l1704_170496

theorem quadratic_solution :
  ∀ x : ℝ, (3 * x - 1) * (2 * x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 :=
by
  sorry

end quadratic_solution_l1704_170496


namespace grains_in_gray_parts_l1704_170403

theorem grains_in_gray_parts (total1 total2 shared : ℕ) (h1 : total1 = 87) (h2 : total2 = 110) (h_shared : shared = 68) :
  (total1 - shared) + (total2 - shared) = 61 :=
by sorry

end grains_in_gray_parts_l1704_170403


namespace trajectory_of_centroid_l1704_170441

def foci (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (0, 1) ∧ F2 = (0, -1)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 3) + (P.2^2 / 4) = 1

def centroid_eq (G : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, on_ellipse P ∧ 
  foci (0, 1) (0, -1) ∧ 
  G = (P.1 / 3, (1 + -1 + P.2) / 3)

theorem trajectory_of_centroid :
  ∀ G : ℝ × ℝ, (centroid_eq G → 3 * G.1^2 + (9 * G.2^2) / 4 = 1 ∧ G.1 ≠ 0) :=
by 
  intros G h
  sorry

end trajectory_of_centroid_l1704_170441


namespace sufficient_not_necessary_l1704_170414

def M : Set ℤ := {1, 2}
def N (a : ℤ) : Set ℤ := {a^2}

theorem sufficient_not_necessary (a : ℤ) :
  (a = 1 → N a ⊆ M) ∧ (N a ⊆ M → a = 1) = false :=
by 
  sorry

end sufficient_not_necessary_l1704_170414


namespace smallest_positive_period_1_smallest_positive_period_2_l1704_170423

-- To prove the smallest positive period T for f(x) = |sin x| + |cos x| is π/2
theorem smallest_positive_period_1 : ∃ T > 0, T = Real.pi / 2 ∧ ∀ x : ℝ, (abs (Real.sin (x + T)) + abs (Real.cos (x + T)) = abs (Real.sin x) + abs (Real.cos x))  := sorry

-- To prove the smallest positive period T for f(x) = tan (2x/3) is 3π/2
theorem smallest_positive_period_2 : ∃ T > 0, T = 3 * Real.pi / 2 ∧ ∀ x : ℝ, (Real.tan ((2 * x) / 3 + T) = Real.tan ((2 * x) / 3)) := sorry

end smallest_positive_period_1_smallest_positive_period_2_l1704_170423


namespace pentagon_perimeter_l1704_170486

noncomputable def perimeter_pentagon (FG GH HI IJ : ℝ) (FH FI FJ : ℝ) : ℝ :=
  FG + GH + HI + IJ + FJ

theorem pentagon_perimeter : 
  ∀ (FG GH HI IJ : ℝ), 
  ∀ (FH FI FJ : ℝ),
  FG = 1 → GH = 1 → HI = 1 → IJ = 1 →
  FH^2 = FG^2 + GH^2 → FI^2 = FH^2 + HI^2 → FJ^2 = FI^2 + IJ^2 →
  perimeter_pentagon FG GH HI IJ FJ = 6 :=
by
  intros FG GH HI IJ FH FI FJ
  intros H_FG H_GH H_HI H_IJ
  intros H1 H2 H3
  sorry

end pentagon_perimeter_l1704_170486


namespace xyz_line_segments_total_length_l1704_170458

noncomputable def total_length_XYZ : ℝ :=
  let length_X := 2 * Real.sqrt 2
  let length_Y := 2 + 2 * Real.sqrt 2
  let length_Z := 2 + Real.sqrt 2
  length_X + length_Y + length_Z

theorem xyz_line_segments_total_length : total_length_XYZ = 4 + 5 * Real.sqrt 2 := 
  sorry

end xyz_line_segments_total_length_l1704_170458


namespace net_investment_change_l1704_170427

variable (I : ℝ)

def first_year_increase (I : ℝ) : ℝ := I * 1.75
def second_year_decrease (W : ℝ) : ℝ := W * 0.70

theorem net_investment_change : 
  let I' := first_year_increase 100 
  let I'' := second_year_decrease I' 
  I'' - 100 = 22.50 :=
by
  sorry

end net_investment_change_l1704_170427


namespace fewer_columns_after_rearrangement_l1704_170459

theorem fewer_columns_after_rearrangement : 
  ∀ (T R R' C C' fewer_columns : ℕ),
    T = 30 → 
    R = 5 → 
    R' = R + 4 →
    C * R = T →
    C' * R' = T →
    fewer_columns = C - C' →
    fewer_columns = 3 :=
by
  intros T R R' C C' fewer_columns hT hR hR' hCR hC'R' hfewer_columns
  -- sorry to skip the proof part
  sorry

end fewer_columns_after_rearrangement_l1704_170459


namespace negation_of_proposition_l1704_170420

theorem negation_of_proposition :
  (∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (¬ ∃ x : ℝ, x^2 + 1 < 0) :=
by
  sorry

end negation_of_proposition_l1704_170420


namespace number_picked_by_person_announcing_average_5_l1704_170429

-- Definition of given propositions and assumptions
def numbers_picked (b : Fin 6 → ℕ) (average : Fin 6 → ℕ) :=
  (b 4 = 15) ∧
  (average 4 = 8) ∧
  (average 1 = 5) ∧
  (b 2 + b 4 = 16) ∧
  (b 0 + b 2 = 10) ∧
  (b 4 + b 0 = 12)

-- Prove that given the conditions, the number picked by the person announcing an average of 5 is 7
theorem number_picked_by_person_announcing_average_5 (b : Fin 6 → ℕ) (average : Fin 6 → ℕ)
  (h : numbers_picked b average) : b 2 = 7 :=
  sorry

end number_picked_by_person_announcing_average_5_l1704_170429


namespace find_f_neg5_l1704_170415

-- Define the function f and the constants a, b, and c
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

-- State the main theorem we want to prove
theorem find_f_neg5 (a b c : ℝ) (h : f 5 a b c = 9) : f (-5) a b c = 1 :=
by
  sorry

end find_f_neg5_l1704_170415


namespace factor_expression_l1704_170463

theorem factor_expression (y : ℝ) : 
  (16 * y ^ 6 + 36 * y ^ 4 - 9) - (4 * y ^ 6 - 6 * y ^ 4 - 9) = 6 * y ^ 4 * (2 * y ^ 2 + 7) := 
by sorry

end factor_expression_l1704_170463


namespace sum_infinite_geometric_series_l1704_170407

theorem sum_infinite_geometric_series :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  (a / (1 - r) = (3 : ℚ) / 8) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  sorry

end sum_infinite_geometric_series_l1704_170407


namespace M_inter_P_eq_l1704_170449

-- Define the sets M and P
def M : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 4 * x + y = 6 }
def P : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 3 * x + 2 * y = 7 }

-- Prove that the intersection of M and P is {(1, 2)}
theorem M_inter_P_eq : M ∩ P = { (1, 2) } := 
by 
sorry

end M_inter_P_eq_l1704_170449


namespace infinite_sum_fraction_equals_quarter_l1704_170421

theorem infinite_sum_fraction_equals_quarter :
  (∑' n : ℕ, (3 ^ n) / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1))) = 1 / 4 :=
by
  -- With the given conditions, we need to prove the above statement
  -- The conditions have been used to express the problem in Lean
  sorry

end infinite_sum_fraction_equals_quarter_l1704_170421


namespace fraction_identity_l1704_170497

theorem fraction_identity
  (x w y z : ℝ)
  (hxw_pos : x * w > 0)
  (hyz_pos : y * z > 0)
  (hxw_inv_sum : 1 / x + 1 / w = 20)
  (hyz_inv_sum : 1 / y + 1 / z = 25)
  (hxw_inv : 1 / (x * w) = 6)
  (hyz_inv : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 :=
by
  -- proof omitted
  sorry

end fraction_identity_l1704_170497


namespace remaining_mushroom_pieces_l1704_170425

theorem remaining_mushroom_pieces 
  (mushrooms : ℕ) 
  (pieces_per_mushroom : ℕ) 
  (pieces_used_by_kenny : ℕ) 
  (pieces_used_by_karla : ℕ) 
  (mushrooms_cut : mushrooms = 22) 
  (pieces_per_mushroom_def : pieces_per_mushroom = 4) 
  (kenny_pieces_def : pieces_used_by_kenny = 38) 
  (karla_pieces_def : pieces_used_by_karla = 42) : 
  (mushrooms * pieces_per_mushroom - (pieces_used_by_kenny + pieces_used_by_karla)) = 8 := 
by 
  sorry

end remaining_mushroom_pieces_l1704_170425


namespace part1_part2_l1704_170400

-- Problem statement (1)
theorem part1 (a : ℝ) (h : a = -3) :
  (∀ x : ℝ, (x^2 + a * x + 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  { x : ℝ // (x^2 + a * x + 2) ≥ 1 - x^2 } = { x : ℝ // x ≤ 1 / 2 ∨ x ≥ 1 } :=
sorry

-- Problem statement (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x + 2) + x^2 + 1 = 2 * x^2 + a * x + 3) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ (2 * x^2 + a * x + 3) = 0) →
  -5 < a ∧ a < -2 * Real.sqrt 6 :=
sorry

end part1_part2_l1704_170400


namespace square_difference_l1704_170492

theorem square_difference (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, c^2 = a^2 - b^2 :=
by
  sorry

end square_difference_l1704_170492


namespace numerology_eq_l1704_170408

theorem numerology_eq : 2222 - 222 + 22 - 2 = 2020 :=
by
  sorry

end numerology_eq_l1704_170408


namespace Jorge_age_in_2005_l1704_170468

theorem Jorge_age_in_2005
  (age_Simon_2010 : ℕ)
  (age_difference : ℕ)
  (age_of_Simon_2010 : age_Simon_2010 = 45)
  (age_difference_Simon_Jorge : age_difference = 24)
  (age_Simon_2005 : ℕ := age_Simon_2010 - 5)
  (age_Jorge_2005 : ℕ := age_Simon_2005 - age_difference) :
  age_Jorge_2005 = 16 := by
  sorry

end Jorge_age_in_2005_l1704_170468


namespace cubic_product_of_roots_l1704_170456

theorem cubic_product_of_roots (k : ℝ) :
  (∃ a b c : ℝ, a + b + c = 2 ∧ ab + bc + ca = 1 ∧ abc = -k ∧ -k = (max (max a b) c - min (min a b) c)^2) ↔ k = -2 :=
by
  sorry

end cubic_product_of_roots_l1704_170456
