import Mathlib

namespace probability_edge_within_five_hops_l1635_163598

def is_edge_square (n : ℕ) (coord : ℕ × ℕ) : Prop := 
  coord.1 = 1 ∨ coord.1 = n ∨ coord.2 = 1 ∨ coord.2 = n

def is_central_square (coord : ℕ × ℕ) : Prop :=
  (coord = (2, 2)) ∨ (coord = (2, 3)) ∨ (coord = (3, 2)) ∨ (coord = (3, 3))

noncomputable def probability_of_edge_in_n_hops (n : ℕ) : ℚ := sorry

theorem probability_edge_within_five_hops : probability_of_edge_in_n_hops 4 = 7 / 8 :=
sorry

end probability_edge_within_five_hops_l1635_163598


namespace inequality_proof_l1635_163529

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a * b * c := 
by {
  sorry
}

end inequality_proof_l1635_163529


namespace problem_statement_l1635_163510

theorem problem_statement : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end problem_statement_l1635_163510


namespace intersection_unique_point_l1635_163597

theorem intersection_unique_point
    (h1 : ∀ (x y : ℝ), 2 * x + 3 * y = 6)
    (h2 : ∀ (x y : ℝ), 4 * x - 3 * y = 6)
    (h3 : ∀ y : ℝ, 2 = 2)
    (h4 : ∀ x : ℝ, y = 2 / 3)
    : ∃! (x y : ℝ), (2 * x + 3 * y = 6) ∧ (4 * x - 3 * y = 6) ∧ (x = 2) ∧ (y = 2 / 3) := 
by
    sorry

end intersection_unique_point_l1635_163597


namespace smallest_x_for_M_cube_l1635_163586

theorem smallest_x_for_M_cube (x M : ℤ) (h1 : 1890 * x = M^3) : x = 4900 :=
sorry

end smallest_x_for_M_cube_l1635_163586


namespace total_crayons_l1635_163581

-- Definitions for the conditions
def crayons_per_child : Nat := 12
def number_of_children : Nat := 18

-- The statement to be proved
theorem total_crayons :
  (crayons_per_child * number_of_children = 216) := 
by
  sorry

end total_crayons_l1635_163581


namespace green_beans_weight_l1635_163563

/-- 
    Mary uses plastic grocery bags that can hold a maximum of twenty pounds. 
    She buys some green beans, 6 pounds milk, and twice the amount of carrots as green beans. 
    She can fit 2 more pounds of groceries in that bag. 
    Prove that the weight of green beans she bought is equal to 4 pounds.
-/
theorem green_beans_weight (G : ℕ) (H1 : ∀ g : ℕ, g + 6 + 2 * g ≤ 20 - 2) : G = 4 :=
by 
  have H := H1 4
  sorry

end green_beans_weight_l1635_163563


namespace arithmetic_sequence_a5_l1635_163553

variable (a : ℕ → ℝ)

-- Conditions translated to Lean definitions
def cond1 : Prop := a 3 = 7
def cond2 : Prop := a 9 = 19

-- Theorem statement that needs to be proved
theorem arithmetic_sequence_a5 (h1 : cond1 a) (h2 : cond2 a) : a 5 = 11 :=
sorry

end arithmetic_sequence_a5_l1635_163553


namespace limit_example_l1635_163536

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - 11| ∧ |x - 11| < δ) → |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε :=
by
  sorry

end limit_example_l1635_163536


namespace mixed_sum_proof_l1635_163564

def mixed_sum : ℚ :=
  3 + 1/3 + 4 + 1/2 + 5 + 1/5 + 6 + 1/6

def smallest_whole_number_greater_than_mixed_sum : ℤ :=
  Int.ceil (mixed_sum)

theorem mixed_sum_proof :
  smallest_whole_number_greater_than_mixed_sum = 20 := by
  sorry

end mixed_sum_proof_l1635_163564


namespace length_of_wooden_block_l1635_163575

theorem length_of_wooden_block (cm_to_m : ℝ := 30 / 100) (base_length : ℝ := 31) :
  base_length + cm_to_m = 31.3 :=
by
  sorry

end length_of_wooden_block_l1635_163575


namespace set_union_intersection_example_l1635_163505

open Set

theorem set_union_intersection_example :
  let A := {1, 3, 4, 5}
  let B := {2, 4, 6}
  let C := {0, 1, 2, 3, 4}
  (A ∪ B) ∩ C = ({1, 2, 3, 4} : Set ℕ) :=
by
  sorry

end set_union_intersection_example_l1635_163505


namespace polynomial_non_negative_l1635_163506

theorem polynomial_non_negative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  -- we would include the proof steps here
  sorry

end polynomial_non_negative_l1635_163506


namespace max_minute_hands_l1635_163512

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end max_minute_hands_l1635_163512


namespace find_whole_number_l1635_163566

theorem find_whole_number (N : ℕ) : 9.25 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 9.75 → N = 38 := by
  intros h
  have hN : 37 < (N : ℝ) ∧ (N : ℝ) < 39 := by
    -- This part follows directly from multiplying the inequality by 4.
    sorry

  -- Convert to integer comparison
  have h1 : 38 ≤ N := by
    -- Since 37 < N, N must be at least 38 as N is an integer.
    sorry
    
  have h2 : N < 39 := by
    sorry

  -- Conclude that N = 38 as it is the single whole number within the range.
  sorry

end find_whole_number_l1635_163566


namespace number_of_rings_l1635_163576

def is_number_ring (A : Set ℝ) : Prop :=
  ∀ (a b : ℝ), a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A ∧ (a * b) ∈ A

def Z := { n : ℝ | ∃ k : ℤ, n = k }
def N := { n : ℝ | ∃ k : ℕ, n = k }
def Q := { n : ℝ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b }
def R := { n : ℝ | True }
def M := { x : ℝ | ∃ (n m : ℤ), x = n + m * Real.sqrt 2 }
def P := { x : ℝ | ∃ (m n : ℕ), n ≠ 0 ∧ x = m / (2 * n) }

theorem number_of_rings :
  (is_number_ring Z) ∧ ¬(is_number_ring N) ∧ (is_number_ring Q) ∧ 
  (is_number_ring R) ∧ (is_number_ring M) ∧ ¬(is_number_ring P) :=
by sorry

end number_of_rings_l1635_163576


namespace two_integers_difference_l1635_163542

theorem two_integers_difference
  (x y : ℕ)
  (h_sum : x + y = 5)
  (h_cube_diff : x^3 - y^3 = 63)
  (h_gt : x > y) :
  x - y = 3 := 
sorry

end two_integers_difference_l1635_163542


namespace num_valid_a_values_l1635_163501

theorem num_valid_a_values : 
  ∃ S : Finset ℕ, (∀ a ∈ S, a < 100 ∧ (a^3 + 23) % 24 = 0) ∧ S.card = 5 :=
sorry

end num_valid_a_values_l1635_163501


namespace man_speed_proof_l1635_163554

noncomputable def man_speed_to_post_office (v : ℝ) : Prop :=
  let distance := 19.999999999999996
  let time_back := distance / 4
  let total_time := 5 + 48 / 60
  v > 0 ∧ distance / v + time_back = total_time

theorem man_speed_proof : ∃ v : ℝ, man_speed_to_post_office v ∧ v = 25 := by
  sorry

end man_speed_proof_l1635_163554


namespace primes_p_plus_10_plus_14_l1635_163557

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_p_plus_10_plus_14 (p : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (p + 10)) 
  (h3 : is_prime (p + 14)) 
  : p = 3 := sorry

end primes_p_plus_10_plus_14_l1635_163557


namespace solution_set_l1635_163543

-- Given conditions
variable (x : ℝ)

def inequality1 := 2 * x + 1 > 0
def inequality2 := (x + 1) / 3 > x - 1

-- The proof statement
theorem solution_set (h1 : inequality1 x) (h2 : inequality2 x) :
  -1 / 2 < x ∧ x < 2 :=
sorry

end solution_set_l1635_163543


namespace logical_inconsistency_in_dihedral_angle_def_l1635_163578

-- Define the given incorrect definition
def incorrect_dihedral_angle_def : String :=
  "A dihedral angle is an angle formed by two half-planes originating from one straight line."

-- Define the correct definition
def correct_dihedral_angle_def : String :=
  "A dihedral angle is a spatial figure consisting of two half-planes that share a common edge."

-- Define the logical inconsistency
theorem logical_inconsistency_in_dihedral_angle_def :
  incorrect_dihedral_angle_def ≠ correct_dihedral_angle_def := by
  sorry

end logical_inconsistency_in_dihedral_angle_def_l1635_163578


namespace find_x_l1635_163524

variable (a b c d e f g h x : ℤ)

def cell_relationships (a b c d e f g h x : ℤ) : Prop :=
  (a = 10) ∧
  (h = 3) ∧
  (a = 10 + b) ∧
  (b = c + a) ∧
  (c = b + d) ∧
  (d = c + h) ∧
  (e = 10 + f) ∧
  (f = e + g) ∧
  (g = d + h) ∧
  (h = g + x)

theorem find_x : cell_relationships a b c d e f g h x → x = 7 :=
sorry

end find_x_l1635_163524


namespace probability_heads_and_multiple_of_five_l1635_163590

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def coin_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

def die_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

theorem probability_heads_and_multiple_of_five :
  coin_is_fair ∧ die_is_fair →
  (1 / 2) * (1 / 6) = (1 / 12) :=
by
  intro h
  sorry

end probability_heads_and_multiple_of_five_l1635_163590


namespace S13_equals_26_l1635_163584

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

end S13_equals_26_l1635_163584


namespace no_x4_term_implies_a_zero_l1635_163513

theorem no_x4_term_implies_a_zero (a : ℝ) :
  ¬ (∃ (x : ℝ), -5 * x^3 * (x^2 + a * x + 5) = -5 * x^5 - 5 * a * x^4 - 25 * x^3 + 5 * a * x^4) →
  a = 0 :=
by
  -- Step through the proof process to derive this conclusion
  sorry

end no_x4_term_implies_a_zero_l1635_163513


namespace min_value_of_f_l1635_163555

noncomputable def f (x : ℝ) : ℝ :=
  x^2 / (x - 3)

theorem min_value_of_f : ∀ x > 3, f x ≥ 12 :=
by
  sorry

end min_value_of_f_l1635_163555


namespace largest_beverage_amount_l1635_163514

theorem largest_beverage_amount :
  let Milk := (3 / 8 : ℚ)
  let Cider := (7 / 10 : ℚ)
  let OrangeJuice := (11 / 15 : ℚ)
  OrangeJuice > Milk ∧ OrangeJuice > Cider :=
by
  have Milk := (3 / 8 : ℚ)
  have Cider := (7 / 10 : ℚ)
  have OrangeJuice := (11 / 15 : ℚ)
  sorry

end largest_beverage_amount_l1635_163514


namespace negation_of_exists_real_solution_equiv_l1635_163559

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l1635_163559


namespace original_time_between_maintenance_checks_l1635_163577

theorem original_time_between_maintenance_checks (x : ℝ) 
  (h1 : 2 * x = 60) : x = 30 := sorry

end original_time_between_maintenance_checks_l1635_163577


namespace fox_jeans_price_l1635_163551

theorem fox_jeans_price (F : ℝ) (P : ℝ) 
  (pony_price : P = 18) 
  (total_savings : 3 * F * 0.08 + 2 * P * 0.14 = 8.64)
  (total_discount_rate : 0.08 + 0.14 = 0.22)
  (pony_discount_rate : 0.14 = 13.999999999999993 / 100) 
  : F = 15 :=
by
  sorry

end fox_jeans_price_l1635_163551


namespace x_squared_minus_y_squared_l1635_163509

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end x_squared_minus_y_squared_l1635_163509


namespace distinct_pairs_count_l1635_163550

theorem distinct_pairs_count : 
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ x = x^2 + y^2 ∧ y = 3 * x * y) ∧ 
    S.card = 4 :=
by
  sorry

end distinct_pairs_count_l1635_163550


namespace triangle_area_ratio_l1635_163573

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l1635_163573


namespace function_three_distinct_zeros_l1635_163569

theorem function_three_distinct_zeros (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - 3 * a * x + a) ∧ (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  a > 1/4 :=
by
  sorry

end function_three_distinct_zeros_l1635_163569


namespace range_of_a_for_monotonic_f_l1635_163548

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a^2 * x^2 + a * x

theorem range_of_a_for_monotonic_f (a : ℝ) : 
  (∀ x, 1 < x → f a x ≤ f a (1 : ℝ)) ↔ (a ≤ -1 / 2 ∨ 1 ≤ a) := 
by
  sorry

end range_of_a_for_monotonic_f_l1635_163548


namespace functional_eq_f800_l1635_163519

theorem functional_eq_f800
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y)
  (h2 : f 1000 = 6)
  : f 800 = 7.5 := by
  sorry

end functional_eq_f800_l1635_163519


namespace combined_area_correct_l1635_163545

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

end combined_area_correct_l1635_163545


namespace granddaughter_age_is_12_l1635_163533

/-
Conditions:
- Betty is 60 years old.
- Her daughter is 40 percent younger than Betty.
- Her granddaughter is one-third her mother's age.

Question:
- Prove that the granddaughter is 12 years old.
-/

def age_of_Betty := 60

def age_of_daughter (age_of_Betty : ℕ) : ℕ :=
  age_of_Betty - age_of_Betty * 40 / 100

def age_of_granddaughter (age_of_daughter : ℕ) : ℕ :=
  age_of_daughter / 3

theorem granddaughter_age_is_12 (h1 : age_of_Betty = 60) : age_of_granddaughter (age_of_daughter age_of_Betty) = 12 := by
  sorry

end granddaughter_age_is_12_l1635_163533


namespace magic_square_sum_l1635_163580

theorem magic_square_sum (x y z w v: ℕ) (h1: 27 + w + 22 = 49 + w)
  (h2: 27 + 18 + x = 45 + x) (h3: 22 + 24 + y = 46 + y)
  (h4: 49 + w = 46 + y) (hw: w = y - 3) (hx: x = y + 1)
  (hz: z = x + 3) : x + z = 45 :=
by {
  sorry
}

end magic_square_sum_l1635_163580


namespace solve_quadratic_simplify_expression_l1635_163583

-- 1. Solve the equation 2x^2 - 3x + 1 = 0
theorem solve_quadratic (x : ℝ) :
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
sorry

-- 2. Simplify the given expression
theorem simplify_expression (a b : ℝ) :
  ( (a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a) ) / (b^2 / (a^2 - a*b)) = a / b :=
sorry

end solve_quadratic_simplify_expression_l1635_163583


namespace work_schedules_lcm_l1635_163503

theorem work_schedules_lcm : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := 
by 
  sorry

end work_schedules_lcm_l1635_163503


namespace range_of_f_l1635_163517

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2 * t) / (t^2 + 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 2 :=
sorry

end range_of_f_l1635_163517


namespace average_licks_to_center_l1635_163526

theorem average_licks_to_center (Dan_lcks Michael_lcks Sam_lcks David_lcks Lance_lcks : ℕ)
  (h1 : Dan_lcks = 58) 
  (h2 : Michael_lcks = 63) 
  (h3 : Sam_lcks = 70) 
  (h4 : David_lcks = 70) 
  (h5 : Lance_lcks = 39) :
  (Dan_lcks + Michael_lcks + Sam_lcks + David_lcks + Lance_lcks) / 5 = 60 :=
by {
  sorry
}

end average_licks_to_center_l1635_163526


namespace area_gray_region_correct_l1635_163516

def center_C : ℝ × ℝ := (3, 5)
def radius_C : ℝ := 3
def center_D : ℝ × ℝ := (9, 5)
def radius_D : ℝ := 3

noncomputable def area_gray_region : ℝ :=
  let rectangle_area := (center_D.1 - center_C.1) * (center_C.2 - (center_C.2 - radius_C))
  let sector_area := (1 / 4) * radius_C ^ 2 * Real.pi
  rectangle_area - 2 * sector_area

theorem area_gray_region_correct :
  area_gray_region = 18 - 9 / 2 * Real.pi :=
by
  sorry

end area_gray_region_correct_l1635_163516


namespace janet_final_lives_l1635_163544

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

end janet_final_lives_l1635_163544


namespace probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l1635_163530

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

end probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l1635_163530


namespace lasagna_ground_mince_l1635_163572

theorem lasagna_ground_mince (total_ground_mince : ℕ) (num_cottage_pies : ℕ) (ground_mince_per_cottage_pie : ℕ) 
  (num_lasagnas : ℕ) (L : ℕ) : 
  total_ground_mince = 500 ∧ num_cottage_pies = 100 ∧ ground_mince_per_cottage_pie = 3 
  ∧ num_lasagnas = 100 ∧ total_ground_mince - num_cottage_pies * ground_mince_per_cottage_pie = num_lasagnas * L 
  → L = 2 := 
by sorry

end lasagna_ground_mince_l1635_163572


namespace quadratic_roots_sum_product_l1635_163541

noncomputable def quadratic_sum (a b c : ℝ) : ℝ := -b / a
noncomputable def quadratic_product (a b c : ℝ) : ℝ := c / a

theorem quadratic_roots_sum_product :
  let a := 9
  let b := -45
  let c := 50
  quadratic_sum a b c = 5 ∧ quadratic_product a b c = 50 / 9 :=
by
  sorry

end quadratic_roots_sum_product_l1635_163541


namespace xy_value_l1635_163565

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end xy_value_l1635_163565


namespace angela_insects_l1635_163511

theorem angela_insects (A J D : ℕ) (h1 : A = J / 2) (h2 : J = 5 * D) (h3 : D = 30) : A = 75 :=
by
  sorry

end angela_insects_l1635_163511


namespace problem1_problem2_l1635_163528

open Real

-- Proof problem 1: Given condition and the required result.
theorem problem1 (x y : ℝ) (h : (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7) :
  x^2 + y^2 = 5 :=
sorry

-- Proof problem 2: Solve the polynomial equation.
theorem problem2 (x : ℝ) :
  (x = sqrt 2 ∨ x = -sqrt 2 ∨ x = 2 ∨ x = -2) ↔ (x^4 - 6 * x^2 + 8 = 0) :=
sorry

end problem1_problem2_l1635_163528


namespace inequality_abc_l1635_163515

theorem inequality_abc {a b c : ℝ} {n : ℕ} 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) (hn : 0 < n) :
  (1 / (1 + a)^(1 / n : ℝ)) + (1 / (1 + b)^(1 / n : ℝ)) + (1 / (1 + c)^(1 / n : ℝ)) 
  ≤ 3 / (1 + (a * b * c)^(1 / 3 : ℝ))^(1 / n : ℝ) := sorry

end inequality_abc_l1635_163515


namespace Joey_downhill_speed_l1635_163527

theorem Joey_downhill_speed
  (Route_length : ℝ) (Time_uphill : ℝ) (Speed_uphill : ℝ) (Overall_average_speed : ℝ) (Extra_time_due_to_rain : ℝ) :
  Route_length = 5 →
  Time_uphill = 1.25 →
  Speed_uphill = 4 →
  Overall_average_speed = 6 →
  Extra_time_due_to_rain = 0.25 →
  ((2 * Route_length) / Overall_average_speed - Time_uphill - Extra_time_due_to_rain) * (Route_length / (2 * Route_length / Overall_average_speed - Time_uphill - Extra_time_due_to_rain)) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Joey_downhill_speed_l1635_163527


namespace find_x_y_l1635_163547

theorem find_x_y (x y : ℝ) : 
    (3 * x + 2 * y + 5 * x + 7 * x = 360) →
    (x = y) →
    (x = 360 / 17) ∧ (y = 360 / 17) := by
  intros h₁ h₂
  sorry

end find_x_y_l1635_163547


namespace bell_rings_count_l1635_163539

def classes : List String := ["Maths", "English", "History", "Geography", "Chemistry", "Physics", "Literature", "Music"]

def total_classes : Nat := classes.length

def rings_per_class : Nat := 2

def classes_before_music : Nat := total_classes - 1

def rings_before_music : Nat := classes_before_music * rings_per_class

def current_class_rings : Nat := 1

def total_rings_by_now : Nat := rings_before_music + current_class_rings

theorem bell_rings_count :
  total_rings_by_now = 15 := by
  sorry

end bell_rings_count_l1635_163539


namespace sum_of_sides_le_twice_third_side_l1635_163537

theorem sum_of_sides_le_twice_third_side 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180)
  (h3 : a / (Real.sin A) = b / (Real.sin B))
  (h4 : a / (Real.sin A) = c / (Real.sin C))
  (h5 : b / (Real.sin B) = c / (Real.sin C)) : 
  a + c ≤ 2 * b := 
by 
  sorry

end sum_of_sides_le_twice_third_side_l1635_163537


namespace find_first_number_l1635_163521

theorem find_first_number (x : ℝ) (h1 : 2994 / x = 175) (h2 : 29.94 / 1.45 = 17.5) : x = 17.1 :=
by
  sorry

end find_first_number_l1635_163521


namespace profit_percentage_with_discount_l1635_163500

theorem profit_percentage_with_discount
    (P M : ℝ)
    (h1 : M = 1.27 * P)
    (h2 : 0 < P) :
    ((0.95 * M - P) / P) * 100 = 20.65 :=
by
  sorry

end profit_percentage_with_discount_l1635_163500


namespace beggars_society_votes_l1635_163587

def total_voting_members (votes_for votes_against additional_against : ℕ) :=
  let majority := additional_against / 4
  let initial_difference := votes_for - votes_against
  let updated_against := votes_against + additional_against
  let updated_for := votes_for - additional_against
  updated_for + updated_against

theorem beggars_society_votes :
  total_voting_members 115 92 12 = 207 :=
by
  -- Proof goes here
  sorry

end beggars_society_votes_l1635_163587


namespace volume_of_solid_l1635_163525

noncomputable def s : ℝ := 2 * Real.sqrt 2

noncomputable def h : ℝ := 3 * s

noncomputable def base_area (a b : ℝ) : ℝ := 1 / 2 * a * b

noncomputable def volume (base_area height : ℝ) : ℝ := base_area * height

theorem volume_of_solid : volume (base_area s s) h = 24 * Real.sqrt 2 :=
by
  -- The proof will go here
  sorry

end volume_of_solid_l1635_163525


namespace cost_price_per_meter_l1635_163549

-- We define the given conditions
def meters_sold : ℕ := 60
def selling_price : ℕ := 8400
def profit_per_meter : ℕ := 12

-- We need to prove that the cost price per meter is Rs. 128
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 128 :=
by
  sorry

end cost_price_per_meter_l1635_163549


namespace usual_time_to_catch_bus_l1635_163568

variables (S T T' : ℝ)

theorem usual_time_to_catch_bus
  (h1 : T' = (5 / 4) * T)
  (h2 : T' - T = 6) : T = 24 :=
sorry

end usual_time_to_catch_bus_l1635_163568


namespace solution_set_l1635_163570

def within_bounds (x : ℝ) : Prop := |2 * x + 1| < 1

theorem solution_set : {x : ℝ | within_bounds x} = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end solution_set_l1635_163570


namespace exercise_l1635_163592

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end exercise_l1635_163592


namespace smallest_positive_n_l1635_163599

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l1635_163599


namespace area_is_12_5_l1635_163522

-- Define the triangle XYZ
structure Triangle := 
  (X Y Z : Type) 
  (XZ YZ : ℝ) 
  (angleX angleY angleZ : ℝ)

-- Provided conditions in the problem
def triangleXYZ : Triangle := {
  X := ℝ, 
  Y := ℝ, 
  Z := ℝ, 
  XZ := 5,
  YZ := 5,
  angleX := 45,
  angleY := 45,
  angleZ := 90
}

-- Lean statement to prove the area of triangle XYZ
theorem area_is_12_5 (t : Triangle) 
  (h1 : t.angleZ = 90)
  (h2 : t.angleX = 45)
  (h3 : t.angleY = 45)
  (h4 : t.XZ = 5)
  (h5 : t.YZ = 5) : 
  (1/2 * t.XZ * t.YZ) = 12.5 :=
sorry

end area_is_12_5_l1635_163522


namespace janet_dresses_total_pockets_l1635_163593

theorem janet_dresses_total_pockets :
  ∃ dresses pockets pocket_2 pocket_3,
  dresses = 24 ∧ 
  pockets = dresses / 2 ∧ 
  pocket_2 = pockets / 3 ∧ 
  pocket_3 = pockets - pocket_2 ∧ 
  (pocket_2 * 2 + pocket_3 * 3) = 32 := by
    sorry

end janet_dresses_total_pockets_l1635_163593


namespace find_abs_xyz_l1635_163594

variables {x y z : ℝ}

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem find_abs_xyz
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : distinct x y z)
  (h3 : x + 1 / y = 2)
  (h4 : y + 1 / z = 2)
  (h5 : z + 1 / x = 2) :
  |x * y * z| = 1 :=
by sorry

end find_abs_xyz_l1635_163594


namespace eyes_given_to_dog_l1635_163531

-- Definitions of the conditions
def fish_per_person : ℕ := 4
def number_of_people : ℕ := 3
def eyes_per_fish : ℕ := 2
def eyes_eaten_by_Oomyapeck : ℕ := 22

-- The proof statement
theorem eyes_given_to_dog : ∃ (eyes_given_to_dog : ℕ), eyes_given_to_dog = 4 * 3 * 2 - 22 := by
  sorry

end eyes_given_to_dog_l1635_163531


namespace find_values_l1635_163546

-- Define the conditions as Lean hypotheses
variables (A B : ℝ)

-- State the problem conditions
def condition1 := 30 - (4 * A + 5) = 3 * B
def condition2 := B = 2 * A

-- State the main theorem to be proved
theorem find_values (h1 : condition1 A B) (h2 : condition2 A B) : A = 2.5 ∧ B = 5 :=
by { sorry }

end find_values_l1635_163546


namespace quadratic_roots_ratio_l1635_163558

theorem quadratic_roots_ratio {m n p : ℤ} (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : p ≠ 0)
  (h₃ : ∃ r1 r2 : ℤ, r1 * r2 = m ∧ n = 9 * r1 * r2 ∧ p = -(r1 + r2) ∧ m = -3 * (r1 + r2)) :
  n / p = -27 := by
  sorry

end quadratic_roots_ratio_l1635_163558


namespace cost_of_coat_eq_l1635_163502

-- Define the given conditions
def total_cost : ℕ := 110
def cost_of_shoes : ℕ := 30
def cost_per_jeans : ℕ := 20
def num_of_jeans : ℕ := 2

-- Define the cost calculation for the jeans
def cost_of_jeans : ℕ := num_of_jeans * cost_per_jeans

-- Define the known total cost (shoes and jeans)
def known_total_cost : ℕ := cost_of_shoes + cost_of_jeans

-- Prove James' coat cost
theorem cost_of_coat_eq :
  (total_cost - known_total_cost) = 40 :=
by
  sorry

end cost_of_coat_eq_l1635_163502


namespace chessboard_number_determination_l1635_163534

theorem chessboard_number_determination (d_n : ℤ) (a_n b_n a_1 b_1 c_0 d_0 : ℤ) :
  (∀ i j : ℤ, d_n + a_n = b_n + a_1 + b_1 - (c_0 + d_0) → 
   a_n + b_n = c_0 + d_0 + d_n) →
  ∃ x : ℤ, x = a_1 + b_1 - d_n ∧ 
  x = d_n + (a_1 - c_0) + (b_1 - d_0) :=
by
  sorry

end chessboard_number_determination_l1635_163534


namespace solution_set_f_derivative_l1635_163535

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 1

theorem solution_set_f_derivative :
  { x : ℝ | (deriv f x) < 0 } = { x : ℝ | -1 < x ∧ x < 3 } :=
by
  sorry

end solution_set_f_derivative_l1635_163535


namespace river_bend_students_more_than_pets_l1635_163540

theorem river_bend_students_more_than_pets 
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (hamsters_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ := students_per_classroom * number_of_classrooms)
  (total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms)
  (total_hamsters : ℕ := hamsters_per_classroom * number_of_classrooms)
  (total_pets : ℕ := total_rabbits + total_hamsters) :
  students_per_classroom = 24 ∧ rabbits_per_classroom = 2 ∧ hamsters_per_classroom = 3 ∧ number_of_classrooms = 5 →
  total_students - total_pets = 95 :=
by
  sorry

end river_bend_students_more_than_pets_l1635_163540


namespace contribution_amount_l1635_163518

-- Definitions based on conditions
variable (x : ℝ)

-- Total amount needed
def total_needed := 200

-- Contributions from different families
def contribution_two_families := 2 * x
def contribution_eight_families := 8 * 10 -- 80
def contribution_ten_families := 10 * 5 -- 50
def total_contribution := contribution_two_families + contribution_eight_families + contribution_ten_families

-- Amount raised so far given they need 30 more to reach the target
def raised_so_far := total_needed - 30 -- 170

-- Statement to prove
theorem contribution_amount :
  total_contribution x = raised_so_far →
  x = 20 := by 
  sorry

end contribution_amount_l1635_163518


namespace average_eq_35_implies_y_eq_50_l1635_163560

theorem average_eq_35_implies_y_eq_50 (y : ℤ) (h : (15 + 30 + 45 + y) / 4 = 35) : y = 50 :=
by
  sorry

end average_eq_35_implies_y_eq_50_l1635_163560


namespace g_26_equals_125_l1635_163585

noncomputable def g : ℕ → ℕ := sorry

axiom g_property : ∀ x, g (x + g x) = 5 * g x
axiom g_initial : g 1 = 5

theorem g_26_equals_125 : g 26 = 125 :=
by
  sorry

end g_26_equals_125_l1635_163585


namespace garrison_provisions_last_initially_l1635_163596

noncomputable def garrison_initial_provisions (x : ℕ) : Prop :=
  ∃ x : ℕ, 2000 * (x - 21) = 3300 * 20 ∧ x = 54

theorem garrison_provisions_last_initially :
  garrison_initial_provisions 54 :=
by
  sorry

end garrison_provisions_last_initially_l1635_163596


namespace alice_gadgets_sales_l1635_163582

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

end alice_gadgets_sales_l1635_163582


namespace find_x_l1635_163562

theorem find_x (x : ℕ) (hx : x > 0 ∧ x <= 100) 
    (mean_twice_mode : (40 + 57 + 76 + 90 + x + x) / 6 = 2 * x) : 
    x = 26 :=
sorry

end find_x_l1635_163562


namespace min_value_of_reciprocal_sum_l1635_163538

theorem min_value_of_reciprocal_sum {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (hgeom : 3 = Real.sqrt (3^a * 3^b)) : (1 / a + 1 / b) = 2 :=
sorry  -- Proof not required, only the statement is needed.

end min_value_of_reciprocal_sum_l1635_163538


namespace rightmost_three_digits_of_7_pow_1993_l1635_163574

theorem rightmost_three_digits_of_7_pow_1993 :
  7^1993 % 1000 = 407 := 
sorry

end rightmost_three_digits_of_7_pow_1993_l1635_163574


namespace projection_sum_of_squares_l1635_163552

theorem projection_sum_of_squares (a : ℝ) (α β γ : ℝ) 
    (h1 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) 
    (h2 : (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = 2) :
    4 * a^2 * ((Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2) = 8 * a^2 := 
by
  sorry

end projection_sum_of_squares_l1635_163552


namespace jellybean_probability_l1635_163532

theorem jellybean_probability :
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 2
  let yellow_jellybeans := 5
  let total_picks := 4
  let successful_outcomes := 10 * 7 
  let total_outcomes := Nat.choose 12 4 
  let required_probability := 14 / 99 
  successful_outcomes = 70 ∧ total_outcomes = 495 → 
  successful_outcomes / total_outcomes = required_probability := 
by 
  intros
  sorry

end jellybean_probability_l1635_163532


namespace correct_factorization_A_l1635_163520

-- Define the polynomial expressions
def expression_A : Prop :=
  (x : ℝ) → x^2 - x - 6 = (x + 2) * (x - 3)

def expression_B : Prop :=
  (x : ℝ) → x^2 - 1 = x * (x - 1 / x)

def expression_C : Prop :=
  (x y : ℝ) → 7 * x^2 * y^5 = x * y * 7 * x * y^4

def expression_D : Prop :=
  (x : ℝ) → x^2 + 4 * x + 4 = x * (x + 4) + 4

-- The correct factorization from left to right
theorem correct_factorization_A : expression_A := 
by 
  -- Proof omitted
  sorry

end correct_factorization_A_l1635_163520


namespace find_value_of_a_l1635_163508

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end find_value_of_a_l1635_163508


namespace dvd_cd_ratio_l1635_163591

theorem dvd_cd_ratio (total_sales : ℕ) (dvd_sales : ℕ) (cd_sales : ℕ) (h1 : total_sales = 273) (h2 : dvd_sales = 168) (h3 : cd_sales = total_sales - dvd_sales) : (dvd_sales / Nat.gcd dvd_sales cd_sales) = 8 ∧ (cd_sales / Nat.gcd dvd_sales cd_sales) = 5 :=
by
  sorry

end dvd_cd_ratio_l1635_163591


namespace quadratic_sum_l1635_163567

theorem quadratic_sum (b c : ℤ) : 
  (∃ b c : ℤ, (x^2 - 10*x + 15 = 0) ↔ ((x + b)^2 = c)) → b + c = 5 :=
by
  intros h
  sorry

end quadratic_sum_l1635_163567


namespace find_f_neg_a_l1635_163589

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end find_f_neg_a_l1635_163589


namespace dartboard_central_angle_l1635_163579

-- Define the conditions
variables {A : ℝ} {x : ℝ}

-- State the theorem
theorem dartboard_central_angle (h₁ : A > 0) (h₂ : (1/4 : ℝ) = ((x / 360) * A) / A) : x = 90 := 
by sorry

end dartboard_central_angle_l1635_163579


namespace average_minutes_per_player_l1635_163571

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l1635_163571


namespace not_necessarily_true_l1635_163523

theorem not_necessarily_true (x y : ℝ) (h : x > y) : ¬ (x^2 > y^2) :=
sorry

end not_necessarily_true_l1635_163523


namespace range_of_m_common_tangents_with_opposite_abscissas_l1635_163595

section part1
variable {x : ℝ}

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def h (m : ℝ) (x : ℝ) := m * f x / Real.sin x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 0 Real.pi, h m x ≥ Real.sqrt 2) ↔ m ∈ Set.Ici (Real.sqrt 2 / Real.exp (Real.pi / 4)) := 
by
  sorry
end part1

section part2
variable {x : ℝ}

noncomputable def g (x : ℝ) := Real.log x
noncomputable def f_tangent_line_at (x₁ : ℝ) (x : ℝ) := Real.exp x₁ * x + (1 - x₁) * Real.exp x₁
noncomputable def g_tangent_line_at (x₂ : ℝ) (x : ℝ) := x / x₂ + Real.log x₂ - 1

theorem common_tangents_with_opposite_abscissas :
  ∃ x₁ x₂ : ℝ, (f_tangent_line_at x₁ = g_tangent_line_at (Real.exp (-x₁))) ∧ (x₁ = -x₂) :=
by
  sorry
end part2

end range_of_m_common_tangents_with_opposite_abscissas_l1635_163595


namespace gino_initial_sticks_l1635_163588

-- Definitions based on the conditions
def given_sticks : ℕ := 50
def remaining_sticks : ℕ := 13
def initial_sticks (x y : ℕ) : ℕ := x + y

-- Theorem statement based on the mathematically equivalent proof problem
theorem gino_initial_sticks :
  initial_sticks given_sticks remaining_sticks = 63 :=
by
  sorry

end gino_initial_sticks_l1635_163588


namespace prob_twins_street_l1635_163507

variable (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

theorem prob_twins_street : p ≠ 1 → real := sorry

end prob_twins_street_l1635_163507


namespace find_k_series_sum_l1635_163561

theorem find_k_series_sum (k : ℝ) :
  (2 + ∑' n : ℕ, (2 + (n + 1) * k) / 2 ^ (n + 1)) = 6 -> k = 1 :=
by 
  sorry

end find_k_series_sum_l1635_163561


namespace pizza_slices_correct_l1635_163556

-- Definitions based on conditions
def john_slices : Nat := 3
def sam_slices : Nat := 2 * john_slices
def eaten_slices : Nat := john_slices + sam_slices
def remaining_slices : Nat := 3
def total_slices : Nat := eaten_slices + remaining_slices

-- The statement to be proven.
theorem pizza_slices_correct : total_slices = 12 := by
  sorry

end pizza_slices_correct_l1635_163556


namespace speed_of_j_l1635_163504

theorem speed_of_j (j p : ℝ) 
  (h_faster : j > p)
  (h_distance_j : 24 / j = 24 / j)
  (h_distance_p : 24 / p = 24 / p)
  (h_sum_speeds : j + p = 7)
  (h_sum_times : 24 / j + 24 / p = 14) : j = 4 := 
sorry

end speed_of_j_l1635_163504
