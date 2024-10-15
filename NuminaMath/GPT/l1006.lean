import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1006_100659

noncomputable def has_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * a * 9^(Real.sin x) + 4 * a * 3^(Real.sin x) + a - 8 = 0

theorem range_of_a : ∀ a : ℝ,
  (has_solutions a ↔ (8 / 31 <= a ∧ a <= 72 / 23)) := sorry

end NUMINAMATH_GPT_range_of_a_l1006_100659


namespace NUMINAMATH_GPT_problem_statement_l1006_100667

theorem problem_statement (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1006_100667


namespace NUMINAMATH_GPT_divisible_sum_l1006_100671

theorem divisible_sum (k : ℕ) (n : ℕ) (h : n = 2^(k-1)) : 
  ∀ (S : Finset ℕ), S.card = 2*n - 1 → ∃ T ⊆ S, T.card = n ∧ T.sum id % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisible_sum_l1006_100671


namespace NUMINAMATH_GPT_hot_dogs_total_l1006_100644

theorem hot_dogs_total (D : ℕ)
  (h1 : 9 = 2 * D + D + 3) :
  (2 * D + 9 + D = 15) :=
by sorry

end NUMINAMATH_GPT_hot_dogs_total_l1006_100644


namespace NUMINAMATH_GPT_age_proof_l1006_100630

   variable (x : ℝ)
   
   theorem age_proof (h : 3 * (x + 5) - 3 * (x - 5) = x) : x = 30 :=
   by
     sorry
   
end NUMINAMATH_GPT_age_proof_l1006_100630


namespace NUMINAMATH_GPT_correct_analytical_method_l1006_100637

-- Definitions of the different reasoning methods
def reasoning_from_cause_to_effect : Prop := ∀ (cause effect : Prop), cause → effect
def reasoning_from_effect_to_cause : Prop := ∀ (cause effect : Prop), effect → cause
def distinguishing_and_mutually_inferring : Prop := ∀ (cause effect : Prop), (cause ↔ effect)
def proving_converse_statement : Prop := ∀ (P Q : Prop), (P → Q) → (Q → P)

-- Definition of the analytical method
def analytical_method : Prop := reasoning_from_effect_to_cause

-- Theorem stating that the analytical method is the method of reasoning from effect to cause
theorem correct_analytical_method : analytical_method = reasoning_from_effect_to_cause := 
by 
  -- Complete this proof with refined arguments
  sorry

end NUMINAMATH_GPT_correct_analytical_method_l1006_100637


namespace NUMINAMATH_GPT_count_multiples_of_15_l1006_100656

theorem count_multiples_of_15 (a b n : ℕ) (h_gte : 25 ≤ a) (h_lte : b ≤ 205) (h15 : n = 15) : 
  (∃ (k : ℕ), a ≤ k * n ∧ k * n ≤ b ∧ 1 ≤ k - 1 ∧ k - 1 ≤ 12) :=
sorry

end NUMINAMATH_GPT_count_multiples_of_15_l1006_100656


namespace NUMINAMATH_GPT_opposite_of_neg_two_l1006_100638

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_l1006_100638


namespace NUMINAMATH_GPT_find_s2_side_length_l1006_100628

-- Define the variables involved
variables (r s : ℕ)

-- Conditions based on problem statement
def height_eq : Prop := 2 * r + s = 2160
def width_eq : Prop := 2 * r + 3 * s + 110 = 4020

-- The theorem stating that s = 875 given the conditions
theorem find_s2_side_length (h1 : height_eq r s) (h2 : width_eq r s) : s = 875 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_s2_side_length_l1006_100628


namespace NUMINAMATH_GPT_invalid_prob_distribution_D_l1006_100660

noncomputable def sum_of_probs_A : ℚ :=
  0 + 1/2 + 0 + 0 + 1/2

noncomputable def sum_of_probs_B : ℚ :=
  0.1 + 0.2 + 0.3 + 0.4

noncomputable def sum_of_probs_C (p : ℚ) (hp : 0 ≤ p ∧ p ≤ 1) : ℚ :=
  p + (1 - p)

noncomputable def sum_of_probs_D : ℚ :=
  (1/1*2) + (1/2*3) + (1/3*4) + (1/4*5) + (1/5*6) + (1/6*7) + (1/7*8)

theorem invalid_prob_distribution_D :
  sum_of_probs_D ≠ 1 := sorry

end NUMINAMATH_GPT_invalid_prob_distribution_D_l1006_100660


namespace NUMINAMATH_GPT_max_total_weight_l1006_100645

-- Definitions
def A_max_weight := 5
def E_max_weight := 2 * A_max_weight
def total_swallows := 90
def A_to_E_ratio := 2

-- Main theorem statement
theorem max_total_weight :
  ∃ A E, (A = A_to_E_ratio * E) ∧ (A + E = total_swallows) ∧ ((A * A_max_weight + E * E_max_weight) = 600) :=
  sorry

end NUMINAMATH_GPT_max_total_weight_l1006_100645


namespace NUMINAMATH_GPT_xyz_stock_final_price_l1006_100652

theorem xyz_stock_final_price :
  let s0 := 120
  let s1 := s0 + s0 * 1.5
  let s2 := s1 - s1 * 0.3
  let s3 := s2 + s2 * 0.2
  s3 = 252 := by
  sorry

end NUMINAMATH_GPT_xyz_stock_final_price_l1006_100652


namespace NUMINAMATH_GPT_thirtieth_term_of_arithmetic_seq_l1006_100686

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end NUMINAMATH_GPT_thirtieth_term_of_arithmetic_seq_l1006_100686


namespace NUMINAMATH_GPT_combined_area_correct_l1006_100601

def popsicle_stick_length_gino : ℚ := 9 / 2
def popsicle_stick_width_gino : ℚ := 2 / 5
def popsicle_stick_length_me : ℚ := 6
def popsicle_stick_width_me : ℚ := 3 / 5

def number_of_sticks_gino : ℕ := 63
def number_of_sticks_me : ℕ := 50

def side_length_square : ℚ := number_of_sticks_gino / 4 * popsicle_stick_length_gino
def area_square : ℚ := side_length_square ^ 2

def length_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_length_me
def width_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_width_me
def area_rectangle : ℚ := length_rectangle * width_rectangle

def combined_area : ℚ := area_square + area_rectangle

theorem combined_area_correct : combined_area = 6806.25 := by
  sorry

end NUMINAMATH_GPT_combined_area_correct_l1006_100601


namespace NUMINAMATH_GPT_assignment_schemes_with_at_least_one_girl_l1006_100651

theorem assignment_schemes_with_at_least_one_girl
  (boys girls : ℕ)
  (tasks : ℕ)
  (hb : boys = 4)
  (hg : girls = 3)
  (ht : tasks = 3)
  (total_choices : ℕ := (boys + girls).choose tasks * tasks.factorial)
  (all_boys : ℕ := boys.choose tasks * tasks.factorial) :
  total_choices - all_boys = 186 :=
by
  sorry

end NUMINAMATH_GPT_assignment_schemes_with_at_least_one_girl_l1006_100651


namespace NUMINAMATH_GPT_probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l1006_100608

-- Conditions
def red_ball_probability := 1 / 2
def yellow_ball_probability := 1 / 2
def num_draws := 3

-- Define the events and their probabilities
def prob_three_red : ℚ := red_ball_probability ^ num_draws
def prob_three_same : ℚ := 2 * (red_ball_probability ^ num_draws)
def prob_not_all_same : ℚ := 1 - prob_three_same / 2

-- Lean statements
theorem probability_three_red_balls : prob_three_red = 1 / 8 :=
by
  sorry

theorem probability_three_same_color_balls : prob_three_same = 1 / 4 :=
by
  sorry

theorem probability_not_all_same_color_balls : prob_not_all_same = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l1006_100608


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l1006_100681

theorem min_value_of_x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l1006_100681


namespace NUMINAMATH_GPT_homes_distance_is_65_l1006_100647

noncomputable def distance_between_homes
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance : ℕ)
  (time : ℕ) : ℕ :=
  maxwell_distance + brad_speed * time

theorem homes_distance_is_65
  (maxwell_speed : ℕ := 2)
  (brad_speed : ℕ := 3)
  (maxwell_distance : ℕ := 26)
  (time : ℕ := maxwell_distance / maxwell_speed) :
  distance_between_homes maxwell_speed brad_speed maxwell_distance time = 65 :=
by 
  sorry

end NUMINAMATH_GPT_homes_distance_is_65_l1006_100647


namespace NUMINAMATH_GPT_apples_per_person_l1006_100673

-- Define conditions
def total_apples : ℝ := 45
def number_of_people : ℝ := 3.0

-- Theorem statement: Calculate how many apples each person received.
theorem apples_per_person : 
  (total_apples / number_of_people) = 15 := 
by
  sorry

end NUMINAMATH_GPT_apples_per_person_l1006_100673


namespace NUMINAMATH_GPT_skiing_ratio_l1006_100689

theorem skiing_ratio (S : ℕ) (H1 : 4000 ≤ 12000) (H2 : S + 4000 = 12000) : S / 4000 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_skiing_ratio_l1006_100689


namespace NUMINAMATH_GPT_geometric_sequence_sufficient_and_necessary_l1006_100648

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_sufficient_and_necessary (a : ℕ → ℝ) (h1 : a 0 > 0) :
  (a 0 < a 1) ↔ (is_geometric_sequence a ∧ is_increasing_sequence a) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sufficient_and_necessary_l1006_100648


namespace NUMINAMATH_GPT_balance_problem_l1006_100662

variable {G B Y W : ℝ}

theorem balance_problem
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 7.5 * B)
  (h3 : 5 * B = 3.5 * W) :
  5 * G + 4 * Y + 3 * W = (170 / 7) * B := by
  sorry

end NUMINAMATH_GPT_balance_problem_l1006_100662


namespace NUMINAMATH_GPT_problem_statement_l1006_100663

noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def β : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 50
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1006_100663


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_m_l1006_100629

def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

theorem problem1_solution_set :
  {x : ℝ | f x ≤ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ 10} := 
sorry

theorem problem2_range_of_m (m : ℝ) (h : ∃ x : ℝ, f x - g x ≥ m - 3) :
  m ≤ 5 :=
sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_m_l1006_100629


namespace NUMINAMATH_GPT_probability_of_specific_combination_l1006_100635

def total_shirts : ℕ := 3
def total_shorts : ℕ := 7
def total_socks : ℕ := 4
def total_clothes : ℕ := total_shirts + total_shorts + total_socks
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def favorable_outcomes : ℕ := (choose total_shirts 2) * (choose total_shorts 1) * (choose total_socks 1)
def total_outcomes : ℕ := choose total_clothes 4

theorem probability_of_specific_combination :
  favorable_outcomes / total_outcomes = 84 / 1001 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_of_specific_combination_l1006_100635


namespace NUMINAMATH_GPT_smallest_rel_prime_to_180_l1006_100642

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ (∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → y ≥ x) ∧ x = 7 :=
  sorry

end NUMINAMATH_GPT_smallest_rel_prime_to_180_l1006_100642


namespace NUMINAMATH_GPT_cos_of_three_pi_div_two_l1006_100680

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_three_pi_div_two_l1006_100680


namespace NUMINAMATH_GPT_allocate_25_rubles_in_4_weighings_l1006_100657

theorem allocate_25_rubles_in_4_weighings :
  ∃ (coins : ℕ) (coins5 : ℕ → ℕ), 
    (coins = 1600) ∧ 
    (coins5 0 = 800 ∧ coins5 1 = 800) ∧
    (coins5 2 = 400 ∧ coins5 3 = 400) ∧
    (coins5 4 = 200 ∧ coins5 5 = 200) ∧
    (coins5 6 = 100 ∧ coins5 7 = 100) ∧
    (
      25 = 20 + 5 ∧ 
      (∃ i j k l m n, coins5 i = 400 ∧ coins5 j = 400 ∧ coins5 k = 200 ∧
        coins5 l = 200 ∧ coins5 m = 100 ∧ coins5 n = 100)
    )
  := 
sorry

end NUMINAMATH_GPT_allocate_25_rubles_in_4_weighings_l1006_100657


namespace NUMINAMATH_GPT_symmetric_points_coords_l1006_100627

theorem symmetric_points_coords (a b : ℝ) :
    let N := (a, -b)
    let P := (-a, -b)
    let Q := (b, a)
    N = (a, -b) ∧ P = (-a, -b) ∧ Q = (b, a) →
    Q = (b, a) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_symmetric_points_coords_l1006_100627


namespace NUMINAMATH_GPT_total_capsules_sold_in_2_weeks_l1006_100692

-- Define the conditions as constants
def Earnings100mgPerWeek := 80
def CostPer100mgCapsule := 5
def Earnings500mgPerWeek := 60
def CostPer500mgCapsule := 2

-- Theorem to prove the total number of capsules sold in 2 weeks
theorem total_capsules_sold_in_2_weeks : 
  (Earnings100mgPerWeek / CostPer100mgCapsule) * 2 + (Earnings500mgPerWeek / CostPer500mgCapsule) * 2 = 92 :=
by
  sorry

end NUMINAMATH_GPT_total_capsules_sold_in_2_weeks_l1006_100692


namespace NUMINAMATH_GPT_exponential_decreasing_range_l1006_100688

theorem exponential_decreasing_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x y : ℝ, x < y → a^y < a^x) : 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_GPT_exponential_decreasing_range_l1006_100688


namespace NUMINAMATH_GPT_volume_P3_correct_m_plus_n_l1006_100679

noncomputable def P_0_volume : ℚ := 1

noncomputable def tet_volume (v : ℚ) : ℚ := (1/27) * v

noncomputable def volume_P3 : ℚ := 
  let ΔP1 := 4 * tet_volume P_0_volume
  let ΔP2 := (2/9) * ΔP1
  let ΔP3 := (2/9) * ΔP2
  P_0_volume + ΔP1 + ΔP2 + ΔP3

theorem volume_P3_correct : volume_P3 = 22615 / 6561 := 
by {
  sorry
}

theorem m_plus_n : 22615 + 6561 = 29176 := 
by {
  sorry
}

end NUMINAMATH_GPT_volume_P3_correct_m_plus_n_l1006_100679


namespace NUMINAMATH_GPT_accounting_majors_l1006_100685

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end NUMINAMATH_GPT_accounting_majors_l1006_100685


namespace NUMINAMATH_GPT_exercise_l1006_100602

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end NUMINAMATH_GPT_exercise_l1006_100602


namespace NUMINAMATH_GPT_probability_edge_within_five_hops_l1006_100691

def is_edge_square (n : ℕ) (coord : ℕ × ℕ) : Prop := 
  coord.1 = 1 ∨ coord.1 = n ∨ coord.2 = 1 ∨ coord.2 = n

def is_central_square (coord : ℕ × ℕ) : Prop :=
  (coord = (2, 2)) ∨ (coord = (2, 3)) ∨ (coord = (3, 2)) ∨ (coord = (3, 3))

noncomputable def probability_of_edge_in_n_hops (n : ℕ) : ℚ := sorry

theorem probability_edge_within_five_hops : probability_of_edge_in_n_hops 4 = 7 / 8 :=
sorry

end NUMINAMATH_GPT_probability_edge_within_five_hops_l1006_100691


namespace NUMINAMATH_GPT_solve_for_x_l1006_100620

-- Define the given condition
def condition (x : ℝ) : Prop := (x - 5) ^ 3 = -((1 / 27)⁻¹)

-- State the problem as a Lean theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1006_100620


namespace NUMINAMATH_GPT_part_a_part_b_l1006_100619

-- Definitions of the basic tiles, colorings, and the proposition

inductive Color
| black : Color
| white : Color

structure Tile :=
(c00 c01 c10 c11 : Color)

-- Ali's forbidden tiles (6 types for part (a))
def forbiddenTiles_6 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white
]

-- Ali's forbidden tiles (7 types for part (b))
def forbiddenTiles_7 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white
]

-- Propositions to be proved

-- Part (a): Mohammad can color the infinite table with no forbidden tiles present
theorem part_a :
  ∃f : ℕ × ℕ → Color, ∀ t ∈ forbiddenTiles_6, ∃ x y : ℕ, ¬(f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

-- Part (b): Ali can present 7 forbidden tiles such that Mohammad cannot achieve his goal
theorem part_b :
  ∀ f : ℕ × ℕ → Color, ∃ t ∈ forbiddenTiles_7, ∃ x y : ℕ, (f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l1006_100619


namespace NUMINAMATH_GPT_rachel_age_is_24_5_l1006_100625

/-- Rachel is 4 years older than Leah -/
def rachel_age_eq_leah_plus_4 (R L : ℝ) : Prop := R = L + 4

/-- Together, Rachel and Leah are twice as old as Sam -/
def rachel_and_leah_eq_twice_sam (R L S : ℝ) : Prop := R + L = 2 * S

/-- Alex is twice as old as Rachel -/
def alex_eq_twice_rachel (A R : ℝ) : Prop := A = 2 * R

/-- The sum of all four friends' ages is 92 -/
def sum_ages_eq_92 (R L S A : ℝ) : Prop := R + L + S + A = 92

theorem rachel_age_is_24_5 (R L S A : ℝ) :
  rachel_age_eq_leah_plus_4 R L →
  rachel_and_leah_eq_twice_sam R L S →
  alex_eq_twice_rachel A R →
  sum_ages_eq_92 R L S A →
  R = 24.5 := 
by 
  sorry

end NUMINAMATH_GPT_rachel_age_is_24_5_l1006_100625


namespace NUMINAMATH_GPT_children_count_l1006_100639

variable (M W C : ℕ)

theorem children_count (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : M + W + C = 300) : C = 30 := by
  sorry

end NUMINAMATH_GPT_children_count_l1006_100639


namespace NUMINAMATH_GPT_cars_overtake_distance_l1006_100621

def speed_red_car : ℝ := 30
def speed_black_car : ℝ := 50
def time_to_overtake : ℝ := 1
def distance_between_cars : ℝ := 20

theorem cars_overtake_distance :
  (speed_black_car - speed_red_car) * time_to_overtake = distance_between_cars :=
by sorry

end NUMINAMATH_GPT_cars_overtake_distance_l1006_100621


namespace NUMINAMATH_GPT_average_tree_height_l1006_100632

theorem average_tree_height : 
  ∀ (T₁ T₂ T₃ T₄ T₅ T₆ : ℕ),
  T₂ = 27 ->
  ((T₁ = 3 * T₂) ∨ (T₁ = T₂ / 3)) ->
  ((T₃ = 3 * T₂) ∨ (T₃ = T₂ / 3)) ->
  ((T₄ = 3 * T₃) ∨ (T₄ = T₃ / 3)) ->
  ((T₅ = 3 * T₄) ∨ (T₅ = T₄ / 3)) ->
  ((T₆ = 3 * T₅) ∨ (T₆ = T₅ / 3)) ->
  (T₁ + T₂ + T₃ + T₄ + T₅ + T₆) / 6 = 22 := 
by 
  intros T₁ T₂ T₃ T₄ T₅ T₆ hT2 hT1 hT3 hT4 hT5 hT6
  sorry

end NUMINAMATH_GPT_average_tree_height_l1006_100632


namespace NUMINAMATH_GPT_inequality_proof_l1006_100699

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1006_100699


namespace NUMINAMATH_GPT_min_red_chips_l1006_100696

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ w / 3) 
  (h2 : b ≤ r / 4) 
  (h3 : w + b ≥ 72) :
  72 ≤ r :=
by
  sorry

end NUMINAMATH_GPT_min_red_chips_l1006_100696


namespace NUMINAMATH_GPT_sum_every_third_odd_integer_l1006_100653

theorem sum_every_third_odd_integer (a₁ d n : ℕ) (S : ℕ) 
  (h₁ : a₁ = 201) 
  (h₂ : d = 6) 
  (h₃ : n = 50) 
  (h₄ : S = (n * (2 * a₁ + (n - 1) * d)) / 2) 
  (h₅ : a₁ + (n - 1) * d = 495) 
  : S = 17400 := 
  by sorry

end NUMINAMATH_GPT_sum_every_third_odd_integer_l1006_100653


namespace NUMINAMATH_GPT_exists_m_inequality_l1006_100646

theorem exists_m_inequality (a b : ℝ) (h : a > b) : ∃ m : ℝ, m < 0 ∧ a * m < b * m :=
by
  sorry

end NUMINAMATH_GPT_exists_m_inequality_l1006_100646


namespace NUMINAMATH_GPT_roots_are_integers_l1006_100654

theorem roots_are_integers (a b : ℤ) (h_discriminant : ∃ (q r : ℚ), r ≠ 0 ∧ a^2 - 4 * b = (q/r)^2) : 
  ∃ x y : ℤ, x^2 - a * x + b = 0 ∧ y^2 - a * y + b = 0 := 
sorry

end NUMINAMATH_GPT_roots_are_integers_l1006_100654


namespace NUMINAMATH_GPT_total_crayons_l1006_100611

-- Definitions for the conditions
def crayons_per_child : Nat := 12
def number_of_children : Nat := 18

-- The statement to be proved
theorem total_crayons :
  (crayons_per_child * number_of_children = 216) := 
by
  sorry

end NUMINAMATH_GPT_total_crayons_l1006_100611


namespace NUMINAMATH_GPT_inequality_proof_l1006_100609

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a * b * c := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l1006_100609


namespace NUMINAMATH_GPT_trader_goal_l1006_100675

theorem trader_goal 
  (profit : ℕ)
  (half_profit : ℕ)
  (donation : ℕ)
  (total_funds : ℕ)
  (made_above_goal : ℕ)
  (goal : ℕ)
  (h1 : profit = 960)
  (h2 : half_profit = profit / 2)
  (h3 : donation = 310)
  (h4 : total_funds = half_profit + donation)
  (h5 : made_above_goal = 180)
  (h6 : goal = total_funds - made_above_goal) :
  goal = 610 :=
by 
  sorry

end NUMINAMATH_GPT_trader_goal_l1006_100675


namespace NUMINAMATH_GPT_remainder_of_N_mod_37_l1006_100682

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_N_mod_37_l1006_100682


namespace NUMINAMATH_GPT_alice_gadgets_sales_l1006_100612

variable (S : ℝ) -- Variable to denote the worth of gadgets Alice sold
variable (E : ℝ) -- Variable to denote Alice's total earnings

theorem alice_gadgets_sales :
  let basic_salary := 240
  let commission_percentage := 0.02
  let save_amount := 29
  let save_percentage := 0.10
  
  -- Total earnings equation
  let earnings_eq := E = basic_salary + commission_percentage * S
  
  -- Savings equation
  let savings_eq := save_percentage * E = save_amount
  
  -- Solve the system of equations to show S = 2500
  S = 2500 :=
by
  sorry

end NUMINAMATH_GPT_alice_gadgets_sales_l1006_100612


namespace NUMINAMATH_GPT_solve_for_x_l1006_100640

theorem solve_for_x : ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1006_100640


namespace NUMINAMATH_GPT_ratio_of_bases_l1006_100643

theorem ratio_of_bases 
(AB CD : ℝ) 
(h_trapezoid : AB < CD) 
(h_AC : ∃ k : ℝ, k = 2 * CD ∧ k = AC) 
(h_altitude : AB = (D - foot)) : 
AB / CD = 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_bases_l1006_100643


namespace NUMINAMATH_GPT_domain_ln_l1006_100697

theorem domain_ln (x : ℝ) : (1 - 2 * x > 0) ↔ x < (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_domain_ln_l1006_100697


namespace NUMINAMATH_GPT_negation_of_exists_real_solution_equiv_l1006_100605

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_real_solution_equiv_l1006_100605


namespace NUMINAMATH_GPT_jared_current_age_condition_l1006_100695

variable (t j: ℕ)

-- Conditions
def tom_current_age := 25
def tom_future_age_condition := t + 5 = 30
def jared_past_age_condition := j - 2 = 2 * (t - 2)

-- Question
theorem jared_current_age_condition : 
  (t + 5 = 30) ∧ (j - 2 = 2 * (t - 2)) → j = 48 :=
by
  sorry

end NUMINAMATH_GPT_jared_current_age_condition_l1006_100695


namespace NUMINAMATH_GPT_two_pipes_fill_time_l1006_100684

theorem two_pipes_fill_time (R : ℝ) (h : 3 * R = 1 / 8) : 2 * R = 1 / 12 := 
by sorry

end NUMINAMATH_GPT_two_pipes_fill_time_l1006_100684


namespace NUMINAMATH_GPT_janet_final_lives_l1006_100600

-- Given conditions
def initial_lives : ℕ := 47
def lives_lost_in_game : ℕ := 23
def points_collected : ℕ := 1840
def lives_per_100_points : ℕ := 2
def penalty_per_200_points : ℕ := 1

-- Definitions based on conditions
def remaining_lives_after_game : ℕ := initial_lives - lives_lost_in_game
def lives_earned_from_points : ℕ := (points_collected / 100) * lives_per_100_points
def lives_lost_due_to_penalties : ℕ := points_collected / 200

-- Theorem statement
theorem janet_final_lives : remaining_lives_after_game + lives_earned_from_points - lives_lost_due_to_penalties = 51 :=
by
  sorry

end NUMINAMATH_GPT_janet_final_lives_l1006_100600


namespace NUMINAMATH_GPT_exponentiation_comparison_l1006_100670

theorem exponentiation_comparison :
  1.7 ^ 0.3 > 0.9 ^ 0.3 :=
by sorry

end NUMINAMATH_GPT_exponentiation_comparison_l1006_100670


namespace NUMINAMATH_GPT_S13_equals_26_l1006_100615

open Nat

variable (a : Nat → ℕ)

-- Define the arithmetic sequence property
def arithmetic_sequence (d a₁ : Nat → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a₁ + n * d

-- Define the summation property
def sum_of_first_n_terms (S : Nat → ℕ) (a₁ : ℕ) (d : ℕ) : Prop :=
   ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2

-- The given condition
def condition (a₁ d : ℕ) : Prop :=
  2 * (a₁ + 4 * d) + 3 * (a₁ + 6 * d) + 2 * (a₁ + 8 * d) = 14

-- The Lean statement for the proof problem
theorem S13_equals_26 (a₁ d : ℕ) (S : Nat → ℕ) 
  (h_seq : arithmetic_sequence a d a₁) 
  (h_sum : sum_of_first_n_terms S a₁ d)
  (h_cond : condition a₁ d) : 
  S 13 = 26 := 
sorry

end NUMINAMATH_GPT_S13_equals_26_l1006_100615


namespace NUMINAMATH_GPT_cost_price_per_meter_l1006_100616

-- We define the given conditions
def meters_sold : ℕ := 60
def selling_price : ℕ := 8400
def profit_per_meter : ℕ := 12

-- We need to prove that the cost price per meter is Rs. 128
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 128 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l1006_100616


namespace NUMINAMATH_GPT_find_counterfeit_coin_l1006_100641

def is_counterfeit (coins : Fin 9 → ℝ) (i : Fin 9) : Prop :=
  ∀ j : Fin 9, j ≠ i → coins j = coins 0 ∧ coins i < coins 0

def algorithm_exists (coins : Fin 9 → ℝ) : Prop :=
  ∃ f : (Fin 9 → ℝ) → Fin 9, is_counterfeit coins (f coins)

theorem find_counterfeit_coin (coins : Fin 9 → ℝ) (h : ∃ i : Fin 9, is_counterfeit coins i) : algorithm_exists coins :=
by sorry

end NUMINAMATH_GPT_find_counterfeit_coin_l1006_100641


namespace NUMINAMATH_GPT_cookies_in_second_type_l1006_100634

theorem cookies_in_second_type (x : ℕ) (h1 : 50 * 12 + 80 * x + 70 * 16 = 3320) : x = 20 :=
by sorry

end NUMINAMATH_GPT_cookies_in_second_type_l1006_100634


namespace NUMINAMATH_GPT_capture_probability_correct_l1006_100626

structure ProblemConditions where
  rachel_speed : ℕ -- seconds per lap
  robert_speed : ℕ -- seconds per lap
  rachel_direction : Bool -- true if counterclockwise, false if clockwise
  robert_direction : Bool -- true if counterclockwise, false if clockwise
  start_time : ℕ -- 0 seconds
  end_time_start : ℕ -- 900 seconds
  end_time_end : ℕ -- 1200 seconds
  photo_coverage_fraction : ℚ -- fraction of the track covered by the photo

noncomputable def probability_capture_in_photo (pc : ProblemConditions) : ℚ :=
  sorry -- define and prove the exact probability

-- Given the conditions in the problem
def problem_instance : ProblemConditions :=
{
  rachel_speed := 120,
  robert_speed := 100,
  rachel_direction := true,
  robert_direction := false,
  start_time := 0,
  end_time_start := 900,
  end_time_end := 1200,
  photo_coverage_fraction := 1/3
}

-- The theorem statement we are asked to prove
theorem capture_probability_correct :
  probability_capture_in_photo problem_instance = 1/9 :=
sorry

end NUMINAMATH_GPT_capture_probability_correct_l1006_100626


namespace NUMINAMATH_GPT_solve_quadratic_simplify_expression_l1006_100613

-- 1. Solve the equation 2x^2 - 3x + 1 = 0
theorem solve_quadratic (x : ℝ) :
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
sorry

-- 2. Simplify the given expression
theorem simplify_expression (a b : ℝ) :
  ( (a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a) ) / (b^2 / (a^2 - a*b)) = a / b :=
sorry

end NUMINAMATH_GPT_solve_quadratic_simplify_expression_l1006_100613


namespace NUMINAMATH_GPT_haley_trees_grown_after_typhoon_l1006_100693

def original_trees := 9
def trees_died := 4
def current_trees := 10

theorem haley_trees_grown_after_typhoon (newly_grown_trees : ℕ) :
  (original_trees - trees_died) + newly_grown_trees = current_trees → newly_grown_trees = 5 :=
by
  sorry

end NUMINAMATH_GPT_haley_trees_grown_after_typhoon_l1006_100693


namespace NUMINAMATH_GPT_smallest_positive_n_l1006_100690

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end NUMINAMATH_GPT_smallest_positive_n_l1006_100690


namespace NUMINAMATH_GPT_red_bowl_values_possible_l1006_100694

theorem red_bowl_values_possible (r b y : ℕ) 
(h1 : r + b + y = 27)
(h2 : 15 * r + 3 * b + 18 * y = 378) : 
  r = 11 ∨ r = 16 ∨ r = 21 := 
  sorry

end NUMINAMATH_GPT_red_bowl_values_possible_l1006_100694


namespace NUMINAMATH_GPT_marvin_solved_yesterday_l1006_100683

variables (M : ℕ)

def Marvin_yesterday := M
def Marvin_today := 3 * M
def Arvin_yesterday := 2 * M
def Arvin_today := 6 * M
def total_problems := Marvin_yesterday + Marvin_today + Arvin_yesterday + Arvin_today

theorem marvin_solved_yesterday :
  total_problems M = 480 → M = 40 :=
sorry

end NUMINAMATH_GPT_marvin_solved_yesterday_l1006_100683


namespace NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l1006_100674

theorem value_of_x_squared_minus_y_squared 
  (x y : ℚ)
  (h1 : x + y = 5 / 8) 
  (h2 : x - y = 3 / 8) :
  x^2 - y^2 = 15 / 64 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l1006_100674


namespace NUMINAMATH_GPT_prism_surface_area_is_8pi_l1006_100650

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

end NUMINAMATH_GPT_prism_surface_area_is_8pi_l1006_100650


namespace NUMINAMATH_GPT_magic_square_sum_l1006_100610

theorem magic_square_sum (x y z w v: ℕ) (h1: 27 + w + 22 = 49 + w)
  (h2: 27 + 18 + x = 45 + x) (h3: 22 + 24 + y = 46 + y)
  (h4: 49 + w = 46 + y) (hw: w = y - 3) (hx: x = y + 1)
  (hz: z = x + 3) : x + z = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_magic_square_sum_l1006_100610


namespace NUMINAMATH_GPT_range_of_a_l1006_100678

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → x^2 - 2*x + a < 0) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1006_100678


namespace NUMINAMATH_GPT_limit_example_l1006_100603

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - 11| ∧ |x - 11| < δ) → |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε :=
by
  sorry

end NUMINAMATH_GPT_limit_example_l1006_100603


namespace NUMINAMATH_GPT_find_values_l1006_100604

-- Define the conditions as Lean hypotheses
variables (A B : ℝ)

-- State the problem conditions
def condition1 := 30 - (4 * A + 5) = 3 * B
def condition2 := B = 2 * A

-- State the main theorem to be proved
theorem find_values (h1 : condition1 A B) (h2 : condition2 A B) : A = 2.5 ∧ B = 5 :=
by { sorry }

end NUMINAMATH_GPT_find_values_l1006_100604


namespace NUMINAMATH_GPT_fraction_addition_l1006_100661

theorem fraction_addition : (3 / 8) + (9 / 12) = 9 / 8 := sorry

end NUMINAMATH_GPT_fraction_addition_l1006_100661


namespace NUMINAMATH_GPT_car_selection_proportion_l1006_100676

def production_volume_emgrand : ℕ := 1600
def production_volume_king_kong : ℕ := 6000
def production_volume_freedom_ship : ℕ := 2000
def total_selected_cars : ℕ := 48

theorem car_selection_proportion :
  (8, 30, 10) = (
    total_selected_cars * production_volume_emgrand /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_king_kong /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_freedom_ship /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship)
  ) :=
by sorry

end NUMINAMATH_GPT_car_selection_proportion_l1006_100676


namespace NUMINAMATH_GPT_min_number_knights_l1006_100633

theorem min_number_knights (h1 : ∃ n : ℕ, n = 7) (h2 : ∃ s : ℕ, s = 42) (h3 : ∃ l : ℕ, l = 24) :
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 7 ∧ k * (7 - k) = 12 ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_number_knights_l1006_100633


namespace NUMINAMATH_GPT_polly_to_sandy_ratio_l1006_100668

variable {W P S : ℝ}
variable (h1 : S = (5/2) * W) (h2 : P = 2 * W)

theorem polly_to_sandy_ratio : P = (4/5) * S := by
  sorry

end NUMINAMATH_GPT_polly_to_sandy_ratio_l1006_100668


namespace NUMINAMATH_GPT_katie_has_more_games_l1006_100658

   -- Conditions
   def katie_games : Nat := 81
   def friends_games : Nat := 59

   -- Problem statement
   theorem katie_has_more_games : (katie_games - friends_games) = 22 :=
   by
     -- Proof to be provided
     sorry
   
end NUMINAMATH_GPT_katie_has_more_games_l1006_100658


namespace NUMINAMATH_GPT_smith_gave_randy_l1006_100636

theorem smith_gave_randy :
  ∀ (s amount_given amount_left : ℕ), amount_given = 1200 → amount_left = 2000 → s = amount_given + amount_left → s = 3200 :=
by
  intros s amount_given amount_left h_given h_left h_total
  rw [h_given, h_left] at h_total
  exact h_total

end NUMINAMATH_GPT_smith_gave_randy_l1006_100636


namespace NUMINAMATH_GPT_negation_of_p_l1006_100618

theorem negation_of_p (p : Prop) : (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ (∀ x : ℝ, x > 0 → ¬ ((x + 1) * Real.exp x > 1)) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1006_100618


namespace NUMINAMATH_GPT_expansion_contains_x4_l1006_100698

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def expansion_term (x : ℂ) (i : ℂ) : ℂ :=
  binomial_coeff 6 2 * x^4 * i^2

theorem expansion_contains_x4 (x i : ℂ) (hi : i = Complex.I) : 
  expansion_term x i = -15 * x^4 := by
  sorry

end NUMINAMATH_GPT_expansion_contains_x4_l1006_100698


namespace NUMINAMATH_GPT_cone_and_sphere_volume_l1006_100672

theorem cone_and_sphere_volume (π : ℝ) (r h : ℝ) (V_cylinder : ℝ) (V_cone V_sphere V_total : ℝ) 
  (h_cylinder : V_cylinder = 54 * π) 
  (h_radius : h = 3 * r)
  (h_cone : V_cone = (1 / 3) * π * r^2 * h) 
  (h_sphere : V_sphere = (4 / 3) * π * r^3) :
  V_total = 42 * π := 
by
  sorry

end NUMINAMATH_GPT_cone_and_sphere_volume_l1006_100672


namespace NUMINAMATH_GPT_rate_of_discount_l1006_100669

theorem rate_of_discount (marked_price : ℝ) (selling_price : ℝ) (rate : ℝ)
  (h_marked : marked_price = 125) (h_selling : selling_price = 120)
  (h_rate : rate = ((marked_price - selling_price) / marked_price) * 100) :
  rate = 4 :=
by
  subst h_marked
  subst h_selling
  subst h_rate
  sorry

end NUMINAMATH_GPT_rate_of_discount_l1006_100669


namespace NUMINAMATH_GPT_percentage_of_boys_and_additional_boys_l1006_100666

theorem percentage_of_boys_and_additional_boys (total_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ)
  (total_students_eq : total_students = 42) (ratio_condition : boys_ratio = 3 ∧ girls_ratio = 4) :
  let total_groups := total_students / (boys_ratio + girls_ratio)
  let total_boys := boys_ratio * total_groups
  (total_boys * 100 / total_students = 300 / 7) ∧ (21 - total_boys = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_boys_and_additional_boys_l1006_100666


namespace NUMINAMATH_GPT_distinct_pairs_count_l1006_100617

theorem distinct_pairs_count : 
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ x = x^2 + y^2 ∧ y = 3 * x * y) ∧ 
    S.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_distinct_pairs_count_l1006_100617


namespace NUMINAMATH_GPT_eyes_given_to_dog_l1006_100607

-- Definitions of the conditions
def fish_per_person : ℕ := 4
def number_of_people : ℕ := 3
def eyes_per_fish : ℕ := 2
def eyes_eaten_by_Oomyapeck : ℕ := 22

-- The proof statement
theorem eyes_given_to_dog : ∃ (eyes_given_to_dog : ℕ), eyes_given_to_dog = 4 * 3 * 2 - 22 := by
  sorry

end NUMINAMATH_GPT_eyes_given_to_dog_l1006_100607


namespace NUMINAMATH_GPT_cylinder_height_to_diameter_ratio_l1006_100655

theorem cylinder_height_to_diameter_ratio
  (r h : ℝ)
  (inscribed_sphere : h = 2 * r)
  (cylinder_volume : π * r^2 * h = 3 * (4/3) * π * r^3) :
  (h / (2 * r)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_to_diameter_ratio_l1006_100655


namespace NUMINAMATH_GPT_balloons_total_l1006_100649

theorem balloons_total (a b : ℕ) (h1 : a = 47) (h2 : b = 13) : a + b = 60 := 
by
  -- Since h1 and h2 provide values for a and b respectively,
  -- the result can be proved using these values.
  sorry

end NUMINAMATH_GPT_balloons_total_l1006_100649


namespace NUMINAMATH_GPT_mixed_sum_proof_l1006_100614

def mixed_sum : ℚ :=
  3 + 1/3 + 4 + 1/2 + 5 + 1/5 + 6 + 1/6

def smallest_whole_number_greater_than_mixed_sum : ℤ :=
  Int.ceil (mixed_sum)

theorem mixed_sum_proof :
  smallest_whole_number_greater_than_mixed_sum = 20 := by
  sorry

end NUMINAMATH_GPT_mixed_sum_proof_l1006_100614


namespace NUMINAMATH_GPT_root_of_quadratic_expression_l1006_100631

theorem root_of_quadratic_expression (n : ℝ) (h : n^2 - 5 * n + 4 = 0) : n^2 - 5 * n = -4 :=
by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_expression_l1006_100631


namespace NUMINAMATH_GPT_exist_rectangle_same_color_l1006_100664

-- Define the colors.
inductive Color
| red
| green
| blue

open Color

-- Define the point and the plane.
structure Point :=
(x : ℝ) (y : ℝ)

-- Assume a coloring function that assigns colors to points on the plane.
def coloring : Point → Color := sorry

-- The theorem stating the existence of a rectangle with vertices of the same color.
theorem exist_rectangle_same_color :
  ∃ (A B C D : Point), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  coloring A = coloring B ∧ coloring B = coloring C ∧ coloring C = coloring D :=
sorry

end NUMINAMATH_GPT_exist_rectangle_same_color_l1006_100664


namespace NUMINAMATH_GPT_xy_value_l1006_100606

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end NUMINAMATH_GPT_xy_value_l1006_100606


namespace NUMINAMATH_GPT_ArianaBoughtTulips_l1006_100622

theorem ArianaBoughtTulips (total_flowers : ℕ) (fraction_roses : ℚ) (carnations : ℕ) 
    (h_total : total_flowers = 40) (h_fraction : fraction_roses = 2/5) (h_carnations : carnations = 14) : 
    total_flowers - (total_flowers * fraction_roses + carnations) = 10 := by
  sorry

end NUMINAMATH_GPT_ArianaBoughtTulips_l1006_100622


namespace NUMINAMATH_GPT_parabola_vertex_l1006_100665

theorem parabola_vertex (y x : ℝ) (h : y = x^2 - 6 * x + 1) : 
  ∃ v_x v_y, (v_x, v_y) = (3, -8) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_vertex_l1006_100665


namespace NUMINAMATH_GPT_solve_system_l1006_100677

-- Define the system of equations
def eq1 (x y : ℝ) : Prop := 2 * x - y = 8
def eq2 (x y : ℝ) : Prop := 3 * x + 2 * y = 5

-- State the theorem to be proved
theorem solve_system : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 3 ∧ y = -2 := 
by 
  exists 3
  exists -2
  -- Proof steps would go here, but we're using sorry to indicate it's incomplete
  sorry

end NUMINAMATH_GPT_solve_system_l1006_100677


namespace NUMINAMATH_GPT_factorize_expression_l1006_100623

variable {R : Type*} [CommRing R] (a b : R)

theorem factorize_expression : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1006_100623


namespace NUMINAMATH_GPT_sin_half_alpha_l1006_100624

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end NUMINAMATH_GPT_sin_half_alpha_l1006_100624


namespace NUMINAMATH_GPT_sandwich_total_calories_l1006_100687

-- Given conditions
def bacon_calories := 2 * 125
def bacon_percentage := 20 / 100

-- Statement to prove
theorem sandwich_total_calories :
  bacon_calories / bacon_percentage = 1250 := 
sorry

end NUMINAMATH_GPT_sandwich_total_calories_l1006_100687
