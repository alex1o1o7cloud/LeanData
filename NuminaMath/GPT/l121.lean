import Mathlib

namespace point_D_coordinates_l121_121690

theorem point_D_coordinates 
  (F : (ℕ × ℕ)) 
  (coords_F : F = (5,5)) 
  (D : (ℕ × ℕ)) 
  (coords_D : D = (2,4)) :
  (D = (2,4)) :=
by 
  sorry

end point_D_coordinates_l121_121690


namespace betty_berries_july_five_l121_121546
open Nat

def betty_bear_berries : Prop :=
  ∃ (b : ℕ), (5 * b + 100 = 150) ∧ (b + 40 = 50)

theorem betty_berries_july_five : betty_bear_berries :=
  sorry

end betty_berries_july_five_l121_121546


namespace theater_seats_l121_121267

theorem theater_seats
  (A : ℕ) -- Number of adult tickets
  (C : ℕ) -- Number of child tickets
  (hC : C = 63) -- 63 child tickets sold
  (total_revenue : ℕ) -- Total Revenue
  (hRev : total_revenue = 519) -- Total revenue is 519
  (adult_ticket_price : ℕ := 12) -- Price per adult ticket
  (child_ticket_price : ℕ := 5) -- Price per child ticket
  (hRevEq : adult_ticket_price * A + child_ticket_price * C = total_revenue) -- Revenue equation
  : A + C = 80 := sorry

end theater_seats_l121_121267


namespace find_smallest_integer_l121_121666

/-- There exists an integer n such that:
   n ≡ 1 [MOD 3],
   n ≡ 2 [MOD 4],
   n ≡ 3 [MOD 5],
   and the smallest such n is 58. -/
theorem find_smallest_integer :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 3 ∧ n = 58 :=
by
  -- Proof goes here (not provided as per the instructions)
  sorry

end find_smallest_integer_l121_121666


namespace meaningful_fraction_iff_l121_121408

theorem meaningful_fraction_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (2 - x)) ↔ x ≠ 2 := by
  sorry

end meaningful_fraction_iff_l121_121408


namespace initial_sand_amount_l121_121854

theorem initial_sand_amount (lost_sand : ℝ) (arrived_sand : ℝ)
  (h1 : lost_sand = 2.4) (h2 : arrived_sand = 1.7) :
  lost_sand + arrived_sand = 4.1 :=
by
  rw [h1, h2]
  norm_num

end initial_sand_amount_l121_121854


namespace maximize_rectangle_area_l121_121822

theorem maximize_rectangle_area (l w : ℝ) (h : l + w ≥ 40) : l * w ≤ 400 :=
by sorry

end maximize_rectangle_area_l121_121822


namespace opposite_of_negative_rational_l121_121626

theorem opposite_of_negative_rational : - (-(4/3)) = (4/3) :=
by
  sorry

end opposite_of_negative_rational_l121_121626


namespace rope_length_l121_121891

theorem rope_length (h1 : ∃ x : ℝ, 4 * x = 20) : 
  ∃ l : ℝ, l = 35 := by
sorry

end rope_length_l121_121891


namespace tangency_condition_l121_121192

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6
def hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

-- The theorem statement based on the question and correct answer:
theorem tangency_condition (n : ℝ) (x y : ℝ) : 
  ellipse x y → hyperbola x y n → n = -6 :=
sorry

end tangency_condition_l121_121192


namespace b_alone_days_l121_121615

theorem b_alone_days {a b : ℝ} (h1 : a + b = 1/6) (h2 : a = 1/11) : b = 1/(66/5) :=
by sorry

end b_alone_days_l121_121615


namespace negation_proposition_l121_121713

variable (a : ℝ)

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
by
  sorry

end negation_proposition_l121_121713


namespace part1_part2_l121_121452

open Set

def f (x : ℝ) : ℝ := abs (x + 2) - abs (2 * x - 1)

def M : Set ℝ := { x | f x > 0 }

theorem part1 :
  M = { x | - (1 / 3 : ℝ) < x ∧ x < 3 } :=
sorry

theorem part2 :
  ∀ (x y : ℝ), x ∈ M → y ∈ M → abs (x + y + x * y) < 15 :=
sorry

end part1_part2_l121_121452


namespace fraction_finding_l121_121669

theorem fraction_finding (x : ℝ) (h : (3 / 4) * x * (2 / 3) = 0.4) : x = 0.8 :=
sorry

end fraction_finding_l121_121669


namespace fraction_spent_on_furniture_l121_121488

variable (original_savings : ℕ)
variable (tv_cost : ℕ)
variable (f : ℚ)

-- Defining the conditions
def conditions := original_savings = 500 ∧ tv_cost = 100 ∧
  f = (original_savings - tv_cost) / original_savings

-- The theorem we want to prove
theorem fraction_spent_on_furniture : conditions original_savings tv_cost f → f = 4 / 5 := by
  sorry

end fraction_spent_on_furniture_l121_121488


namespace solution_set_inequality_l121_121465

theorem solution_set_inequality (a b : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → x^2 + a * x + b ≤ 0) :
  a * b = 6 :=
by {
  sorry
}

end solution_set_inequality_l121_121465


namespace graph_of_equation_is_shifted_hyperbola_l121_121467

-- Definitions
def given_equation (x y : ℝ) : Prop := x^2 - 4*y^2 - 2*x = 0

-- Theorem statement
theorem graph_of_equation_is_shifted_hyperbola :
  ∀ x y : ℝ, given_equation x y = ((x - 1)^2 = 1 + 4*y^2) :=
by
  sorry

end graph_of_equation_is_shifted_hyperbola_l121_121467


namespace base_7_to_base_10_conversion_l121_121086

theorem base_7_to_base_10_conversion :
  (6 * 7^2 + 5 * 7^1 + 3 * 7^0) = 332 :=
by sorry

end base_7_to_base_10_conversion_l121_121086


namespace tetrahedron_sum_eq_14_l121_121660

theorem tetrahedron_sum_eq_14 :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  let edges := 6
  let corners := 4
  let faces := 4
  show edges + corners + faces = 14
  sorry

end tetrahedron_sum_eq_14_l121_121660


namespace cost_of_antibiotics_for_a_week_l121_121178

noncomputable def antibiotic_cost : ℕ := 3
def doses_per_day : ℕ := 3
def days_in_week : ℕ := 7

theorem cost_of_antibiotics_for_a_week : doses_per_day * days_in_week * antibiotic_cost = 63 :=
by
  sorry

end cost_of_antibiotics_for_a_week_l121_121178


namespace quadratic_real_roots_l121_121293

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l121_121293


namespace minimum_questionnaires_l121_121951

theorem minimum_questionnaires (p : ℝ) (r : ℝ) (n_min : ℕ) (h1 : p = 0.65) (h2 : r = 300) :
  n_min = ⌈r / p⌉ ∧ n_min = 462 := 
by
  sorry

end minimum_questionnaires_l121_121951


namespace normal_vector_proof_l121_121170

-- Define the 3D vector type
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a specific normal vector n
def n : Vector3D := ⟨1, -2, 2⟩

-- Define the vector v we need to prove is a normal vector of the same plane
def v : Vector3D := ⟨2, -4, 4⟩

-- Define the statement (without the proof)
theorem normal_vector_proof : v = ⟨2 * n.x, 2 * n.y, 2 * n.z⟩ :=
by
  sorry

end normal_vector_proof_l121_121170


namespace pyramid_volume_and_base_edge_l121_121804

theorem pyramid_volume_and_base_edge:
  ∀ (r: ℝ) (h: ℝ) (_: r = 5) (_: h = 10), 
  ∃ s V: ℝ,
    s = (10 * Real.sqrt 6) / 3 ∧ 
    V = (2000 / 9) :=
by
    sorry

end pyramid_volume_and_base_edge_l121_121804


namespace custom_op_value_l121_121549

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_value_l121_121549


namespace range_of_a_l121_121852

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₀ d : ℝ), ∀ n, a n = a₀ + n * d

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a_seq) 
  (h2 : a_seq 0 = a)
  (h3 : ∀ n, b n = (1 + a_seq n) / a_seq n)
  (h4 : ∀ n : ℕ, 0 < n → b n ≥ b 8) :
  -8 < a ∧ a < -7 :=
sorry

end range_of_a_l121_121852


namespace coordinates_of_A_l121_121006

theorem coordinates_of_A 
  (a : ℝ)
  (h1 : (a - 1) = 3 + (3 * a - 2)) :
  (a - 1, 3 * a - 2) = (-2, -5) :=
by
  sorry

end coordinates_of_A_l121_121006


namespace unique_valid_number_l121_121582

-- Define the form of the three-digit number.
def is_form_sixb5 (n : ℕ) : Prop :=
  ∃ b : ℕ, b < 10 ∧ n = 600 + 10 * b + 5

-- Define the condition for divisibility by 11.
def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

-- Define the alternating sum property for our specific number format.
def alternating_sum_cond (b : ℕ) : Prop :=
  (11 - b) % 11 = 0

-- The final proposition to be proved.
theorem unique_valid_number : ∃ n, is_form_sixb5 n ∧ is_divisible_by_11 n ∧ n = 605 :=
by {
  sorry
}

end unique_valid_number_l121_121582


namespace x_coordinate_point_P_l121_121161

theorem x_coordinate_point_P (x y : ℝ) (h_on_parabola : y^2 = 4 * x) 
  (h_distance : dist (x, y) (1, 0) = 3) : x = 2 :=
sorry

end x_coordinate_point_P_l121_121161


namespace find_pure_imaginary_solutions_l121_121351

noncomputable def poly_eq_zero (x : ℂ) : Prop :=
  x^4 - 6 * x^3 + 13 * x^2 - 42 * x - 72 = 0

noncomputable def is_imaginary (x : ℂ) : Prop :=
  x.im ≠ 0 ∧ x.re = 0

theorem find_pure_imaginary_solutions :
  ∀ x : ℂ, poly_eq_zero x ∧ is_imaginary x ↔ (x = Complex.I * Real.sqrt 7 ∨ x = -Complex.I * Real.sqrt 7) :=
by sorry

end find_pure_imaginary_solutions_l121_121351


namespace anusha_receives_84_l121_121466

-- Define the conditions as given in the problem
def anusha_amount (A : ℕ) (B : ℕ) (E : ℕ) : Prop :=
  12 * A = 8 * B ∧ 12 * A = 6 * E ∧ A + B + E = 378

-- Lean statement to prove the amount Anusha gets is 84
theorem anusha_receives_84 (A B E : ℕ) (h : anusha_amount A B E) : A = 84 :=
sorry

end anusha_receives_84_l121_121466


namespace length_of_larger_sheet_l121_121640

theorem length_of_larger_sheet : 
  ∃ L : ℝ, 2 * (L * 11) = 2 * (5.5 * 11) + 100 ∧ L = 10 :=
by
  sorry

end length_of_larger_sheet_l121_121640


namespace total_students_are_45_l121_121708

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end total_students_are_45_l121_121708


namespace number_of_possible_digits_to_make_divisible_by_4_l121_121906

def four_digit_number_divisible_by_4 (N : ℕ) : Prop :=
  let number := N * 1000 + 264
  number % 4 = 0

theorem number_of_possible_digits_to_make_divisible_by_4 :
  ∃ (count : ℕ), count = 10 ∧ (∀ (N : ℕ), N < 10 → four_digit_number_divisible_by_4 N) :=
by {
  sorry
}

end number_of_possible_digits_to_make_divisible_by_4_l121_121906


namespace max_value_of_fraction_l121_121496

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l121_121496


namespace canoe_upstream_speed_l121_121115

namespace canoe_speed

def V_c : ℝ := 12.5            -- speed of the canoe in still water in km/hr
def V_downstream : ℝ := 16     -- speed of the canoe downstream in km/hr

theorem canoe_upstream_speed :
  ∃ (V_upstream : ℝ), V_upstream = V_c - (V_downstream - V_c) ∧ V_upstream = 9 := by
  sorry

end canoe_speed

end canoe_upstream_speed_l121_121115


namespace solve_weights_problem_l121_121907

variable (a b c d : ℕ) 

def weights_problem := 
  a + b = 280 ∧ 
  a + d = 300 ∧ 
  c + d = 290 → 
  b + c = 270

theorem solve_weights_problem (a b c d : ℕ) : weights_problem a b c d :=
 by
  sorry

end solve_weights_problem_l121_121907


namespace xyz_eq_neg10_l121_121230

noncomputable def complex_numbers := {z : ℂ // z ≠ 0}

variables (a b c x y z : complex_numbers)

def condition1 := a.val = (b.val + c.val) / (x.val - 3)
def condition2 := b.val = (a.val + c.val) / (y.val - 3)
def condition3 := c.val = (a.val + b.val) / (z.val - 3)
def condition4 := x.val * y.val + x.val * z.val + y.val * z.val = 9
def condition5 := x.val + y.val + z.val = 6

theorem xyz_eq_neg10 (a b c x y z : complex_numbers) :
  condition1 a b c x ∧ condition2 a b c y ∧ condition3 a b c z ∧
  condition4 x y z ∧ condition5 x y z → x.val * y.val * z.val = -10 :=
by sorry

end xyz_eq_neg10_l121_121230


namespace maximum_value_of_a_l121_121740

theorem maximum_value_of_a {x y a : ℝ} (hx : x > 1 / 3) (hy : y > 1) :
  (∀ x y, x > 1 / 3 → y > 1 → 9 * x^2 / (a^2 * (y - 1)) + y^2 / (a^2 * (3 * x - 1)) ≥ 1)
  ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end maximum_value_of_a_l121_121740


namespace prove_box_problem_l121_121093

noncomputable def boxProblem : Prop :=
  let height1 := 2
  let width1 := 4
  let length1 := 6
  let clay1 := 48
  let height2 := 3 * height1
  let width2 := 2 * width1
  let length2 := 1.5 * length1
  let volume1 := height1 * width1 * length1
  let volume2 := height2 * width2 * length2
  let n := (volume2 / volume1) * clay1
  n = 432

theorem prove_box_problem : boxProblem := by
  sorry

end prove_box_problem_l121_121093


namespace linda_original_amount_l121_121372

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end linda_original_amount_l121_121372


namespace tournament_total_games_l121_121970

theorem tournament_total_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) : 
  (n * (n - 1) / 2) * k = 1740 := by
  -- Given conditions
  have h1 : n = 30 := h_n
  have h2 : k = 4 := h_k

  -- Calculation using provided values
  sorry

end tournament_total_games_l121_121970


namespace other_root_zero_l121_121999

theorem other_root_zero (b : ℝ) (x : ℝ) (hx_root : x^2 + b * x = 0) (h_x_eq_minus_two : x = -2) : 
  (0 : ℝ) = 0 :=
by
  sorry

end other_root_zero_l121_121999


namespace solve_equation_l121_121325

-- Definitions for the variables and the main equation
def equation (x y z : ℤ) : Prop :=
  5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30

-- The statement that needs to be proved
theorem solve_equation (x y z : ℤ) :
  equation x y z ↔ (x, y, z) = (1, 5, 0) ∨ (x, y, z) = (1, -5, 0) ∨ (x, y, z) = (-1, 5, 0) ∨ (x, y, z) = (-1, -5, 0) :=
by
  sorry

end solve_equation_l121_121325


namespace total_possible_match_sequences_l121_121762

theorem total_possible_match_sequences :
  let num_teams := 2
  let team_size := 7
  let possible_sequences := 2 * (Nat.choose (2 * team_size - 1) (team_size - 1))
  possible_sequences = 3432 :=
by
  sorry

end total_possible_match_sequences_l121_121762


namespace intersection_distance_l121_121576

noncomputable def distance_between_intersections (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, 
    l A.1 A.2 ∧ C A.1 A.2 ∧ l B.1 B.2 ∧ C B.1 B.2 ∧ 
    dist A B = Real.sqrt 6

def line_l (x y : ℝ) : Prop :=
  x - y + 1 = 0

def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sqrt 2 * Real.sin θ

theorem intersection_distance :
  distance_between_intersections line_l curve_C :=
sorry

end intersection_distance_l121_121576


namespace student_history_score_l121_121195

theorem student_history_score 
  (math : ℕ) 
  (third : ℕ) 
  (average : ℕ) 
  (H : ℕ) 
  (h_math : math = 74)
  (h_third : third = 67)
  (h_avg : average = 75)
  (h_overall_avg : (math + third + H) / 3 = average) : 
  H = 84 :=
by
  sorry

end student_history_score_l121_121195


namespace sum_sqrt_inequality_l121_121340

theorem sum_sqrt_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (3 / 2) * (a + b + c) ≥ (Real.sqrt (a^2 + b * c) + Real.sqrt (b^2 + c * a) + Real.sqrt (c^2 + a * b)) :=
by
  sorry

end sum_sqrt_inequality_l121_121340


namespace a_100_value_l121_121863

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     => 0    -- using 0-index for convenience
| (n+1) => a n + 4

-- Prove the value of the 100th term in the sequence
theorem a_100_value : a 100 = 397 := 
by {
  -- proof would go here
  sorry
}

end a_100_value_l121_121863


namespace circle_range_k_l121_121930

theorem circle_range_k (k : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 4 * x + 4 * y + 10 - k = 0) → k > 2 :=
by
  sorry

end circle_range_k_l121_121930


namespace pyramid_top_block_l121_121064

theorem pyramid_top_block (a b c d e : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
                         (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
                         (h : a * b ^ 4 * c ^ 6 * d ^ 4 * e = 140026320) : 
                         (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨ 
                         (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨ 
                         (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨ 
                         (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1) := 
sorry

end pyramid_top_block_l121_121064


namespace min_number_of_lucky_weights_l121_121030

-- Definitions and conditions
def weight (n: ℕ) := n -- A weight is represented as a natural number.

def is_lucky (weights: Finset ℕ) (w: ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a ≠ b ∧ w = a + b
-- w is "lucky" if it's the sum of two other distinct weights in the set.

def min_lucky_guarantee (weights: Finset ℕ) (k: ℕ) : Prop :=
  ∀ (w1 w2 : ℕ), w1 ∈ weights ∧ w2 ∈ weights →
    ∃ (lucky_weights : Finset ℕ), lucky_weights.card = k ∧
    (is_lucky weights w1 ∧ is_lucky weights w2 ∧ (w1 ≥ 3 * w2 ∨ w2 ≥ 3 * w1))
-- The minimum number k of "lucky" weights ensures there exist two weights 
-- such that their masses differ by at least a factor of three.

-- The theorem to be proven
theorem min_number_of_lucky_weights (weights: Finset ℕ) (h_distinct: weights.card = 100) :
  ∃ k, min_lucky_guarantee weights k ∧ k = 87 := 
sorry

end min_number_of_lucky_weights_l121_121030


namespace james_meditation_sessions_l121_121173

theorem james_meditation_sessions (minutes_per_session : ℕ) (hours_per_week : ℕ) (days_per_week : ℕ) (h1 : minutes_per_session = 30) (h2 : hours_per_week = 7) (h3 : days_per_week = 7) : 
  (hours_per_week * 60 / days_per_week / minutes_per_session) = 2 := 
by 
  sorry

end james_meditation_sessions_l121_121173


namespace number_of_games_X_l121_121224

variable (x : ℕ) -- Total number of games played by team X
variable (y : ℕ) -- Wins by team Y
variable (ly : ℕ) -- Losses by team Y
variable (dy : ℕ) -- Draws by team Y
variable (wx : ℕ) -- Wins by team X
variable (lx : ℕ) -- Losses by team X
variable (dx : ℕ) -- Draws by team X

axiom wins_ratio_X : wx = 3 * x / 4
axiom wins_ratio_Y : y = 2 * (x + 12) / 3
axiom wins_difference : y = wx + 4
axiom losses_difference : ly = lx + 5
axiom draws_difference : dy = dx + 3
axiom eq_losses_draws : lx + dx = (x - wx)

theorem number_of_games_X : x = 48 :=
by
  sorry

end number_of_games_X_l121_121224


namespace xy_value_l121_121337

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by sorry

end xy_value_l121_121337


namespace percentage_profit_l121_121063

theorem percentage_profit 
  (C S : ℝ) 
  (h : 29 * C = 24 * S) : 
  ((S - C) / C) * 100 = 20.83 := 
by
  sorry

end percentage_profit_l121_121063


namespace correct_operation_l121_121120

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l121_121120


namespace albert_number_l121_121889

theorem albert_number :
  ∃ (n : ℕ), (1 / (n : ℝ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) ∧ 
             ∃ m : ℕ, (1 / (m : ℝ) + 1 / 2 = 1 / 3 + 2 / (m + 1)) ∧ m ≠ n :=
sorry

end albert_number_l121_121889


namespace inequality_solution_l121_121521

open Set

def f (x : ℝ) : ℝ := |x| + x^2 + 2

def solution_set : Set ℝ := { x | x < -2 ∨ x > 4 / 3 }

theorem inequality_solution :
  { x : ℝ | f (2 * x - 1) > f (3 - x) } = solution_set := by
  sorry

end inequality_solution_l121_121521


namespace trajectory_eqn_l121_121761

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Conditions given in the problem
def PA_squared (P : ℝ × ℝ) : ℝ := (P.1 + 1)^2 + P.2^2
def PB_squared (P : ℝ × ℝ) : ℝ := (P.1 - 1)^2 + P.2^2

-- The main statement to prove
theorem trajectory_eqn (P : ℝ × ℝ) (h : PA_squared P = 3 * PB_squared P) : 
  P.1^2 + P.2^2 - 4 * P.1 + 1 = 0 :=
by 
  sorry

end trajectory_eqn_l121_121761


namespace trigonometric_identity_proof_l121_121008

theorem trigonometric_identity_proof (θ : ℝ) 
  (h : Real.tan (θ + Real.pi / 4) = -3) : 
  2 * Real.sin θ ^ 2 - Real.cos θ ^ 2 = 7 / 5 :=
sorry

end trigonometric_identity_proof_l121_121008


namespace range_of_sum_coords_on_ellipse_l121_121785

theorem range_of_sum_coords_on_ellipse (x y : ℝ) 
  (h : x^2 / 144 + y^2 / 25 = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := 
sorry

end range_of_sum_coords_on_ellipse_l121_121785


namespace find_median_of_first_twelve_positive_integers_l121_121943

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end find_median_of_first_twelve_positive_integers_l121_121943


namespace statement_B_statement_D_l121_121748

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.cos x + Real.sqrt 3 * Real.sin x) - Real.sqrt 3 + 1

theorem statement_B (x₁ x₂ : ℝ) (h1 : -π / 12 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 5 * π / 12) :
  f x₁ < f x₂ := sorry

theorem statement_D (x₁ x₂ x₃ : ℝ) (h1 : π / 3 ≤ x₁) (h2 : x₁ ≤ π / 2) (h3 : π / 3 ≤ x₂) (h4 : x₂ ≤ π / 2) (h5 : π / 3 ≤ x₃) (h6 : x₃ ≤ π / 2) :
  f x₁ + f x₂ - f x₃ > 2 := sorry

end statement_B_statement_D_l121_121748


namespace find_r_l121_121346

variable (m r : ℝ)

theorem find_r (h1 : 5 = m * 3^r) (h2 : 45 = m * 9^(2 * r)) : r = 2 / 3 := by
  sorry

end find_r_l121_121346


namespace kareem_largest_l121_121696

def jose_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let triple := minus_two * 3
  triple + 5

def thuy_final : ℕ :=
  let start := 15
  let triple := start * 3
  let minus_two := triple - 2
  minus_two + 5

def kareem_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let add_five := minus_two + 5
  add_five * 3

theorem kareem_largest : kareem_final > jose_final ∧ kareem_final > thuy_final := by
  sorry

end kareem_largest_l121_121696


namespace max_groups_l121_121810

theorem max_groups (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) : Nat.gcd boys girls = 20 := 
  by
  rw [h1, h2]
  -- Proof steps would be here
  sorry

end max_groups_l121_121810


namespace problem200_squared_minus_399_composite_l121_121557

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ¬ is_prime n

theorem problem200_squared_minus_399_composite : is_composite (200^2 - 399) :=
sorry

end problem200_squared_minus_399_composite_l121_121557


namespace overlap_per_connection_is_4_cm_l121_121305

-- Condition 1: There are 24 tape measures.
def number_of_tape_measures : Nat := 24

-- Condition 2: Each tape measure is 28 cm long.
def length_of_one_tape_measure : Nat := 28

-- Condition 3: The total length of all connected tape measures is 580 cm.
def total_length_with_overlaps : Nat := 580

-- The question to prove: The overlap per connection is 4 cm.
theorem overlap_per_connection_is_4_cm 
  (n : Nat) (length_one : Nat) (total_length : Nat) 
  (h_n : n = number_of_tape_measures)
  (h_length_one : length_one = length_of_one_tape_measure)
  (h_total_length : total_length = total_length_with_overlaps) :
  ((n * length_one - total_length) / (n - 1)) = 4 := 
by 
  sorry

end overlap_per_connection_is_4_cm_l121_121305


namespace abs_quadratic_inequality_solution_l121_121094

theorem abs_quadratic_inequality_solution (x : ℝ) :
  |x^2 - 4 * x + 3| ≤ 3 ↔ 0 ≤ x ∧ x ≤ 4 :=
by sorry

end abs_quadratic_inequality_solution_l121_121094


namespace find_a_l121_121114

theorem find_a (a x : ℝ) (h : x = -1) (heq : -2 * (x - a) = 4) : a = 1 :=
by
  sorry

end find_a_l121_121114


namespace max_men_with_all_amenities_marrried_l121_121554

theorem max_men_with_all_amenities_marrried :
  let total_men := 100
  let married_men := 85
  let men_with_TV := 75
  let men_with_radio := 85
  let men_with_AC := 70
  (∀ s : Finset ℕ, s.card ≤ total_men) →
  (∀ s : Finset ℕ, s.card ≤ married_men) →
  (∀ s : Finset ℕ, s.card ≤ men_with_TV) →
  (∀ s : Finset ℕ, s.card ≤ men_with_radio) →
  (∀ s : Finset ℕ, s.card ≤ men_with_AC) →
  (∀ s : Finset ℕ, s.card ≤ min married_men (min men_with_TV (min men_with_radio men_with_AC))) :=
by
  intros
  sorry

end max_men_with_all_amenities_marrried_l121_121554


namespace intersection_with_complement_l121_121703

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {0, 2, 4}

theorem intersection_with_complement (hU : U = {0, 1, 2, 3, 4})
                                     (hA : A = {0, 1, 2, 3})
                                     (hB : B = {0, 2, 4}) :
  A ∩ (U \ B) = {1, 3} :=
by sorry

end intersection_with_complement_l121_121703


namespace base_k_number_eq_binary_l121_121772

theorem base_k_number_eq_binary (k : ℕ) (h : k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_number_eq_binary_l121_121772


namespace acute_triangle_l121_121433

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ 0 < A ∧ 0 < B ∧ 0 < C

def each_angle_less_than_sum_of_others (A B C : ℝ) : Prop :=
  A < B + C ∧ B < A + C ∧ C < A + B

theorem acute_triangle (A B C : ℝ) 
  (h1 : is_triangle A B C) 
  (h2 : each_angle_less_than_sum_of_others A B C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := 
sorry

end acute_triangle_l121_121433


namespace reciprocal_of_neg_2023_l121_121973

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l121_121973


namespace y_coordinate_of_intersection_l121_121995

def line_eq (x t : ℝ) : ℝ := -2 * x + t

def parabola_eq (x : ℝ) : ℝ := (x - 1) ^ 2 + 1

def intersection_condition (x y t : ℝ) : Prop :=
  y = line_eq x t ∧ y = parabola_eq x ∧ x ≥ 0 ∧ y ≥ 0

theorem y_coordinate_of_intersection (x y : ℝ) (t : ℝ) (h_t : t = 11)
  (h_intersection : intersection_condition x y t) :
  y = 5 := by
  sorry

end y_coordinate_of_intersection_l121_121995


namespace trains_meet_80_km_from_A_l121_121638

-- Define the speeds of the trains
def speed_train_A : ℝ := 60 
def speed_train_B : ℝ := 90 

-- Define the distance between locations A and B
def distance_AB : ℝ := 200 

-- Define the time when the trains meet
noncomputable def meeting_time : ℝ := distance_AB / (speed_train_A + speed_train_B)

-- Define the distance from location A to where the trains meet
noncomputable def distance_from_A (speed_A : ℝ) (meeting_time : ℝ) : ℝ :=
  speed_A * meeting_time

-- Prove the statement
theorem trains_meet_80_km_from_A :
  distance_from_A speed_train_A meeting_time = 80 :=
by
  -- leaving the proof out, it's just an assumption due to 'sorry'
  sorry

end trains_meet_80_km_from_A_l121_121638


namespace ball_hits_ground_at_t_l121_121693

theorem ball_hits_ground_at_t (t : ℝ) : 
  (∃ t, -8 * t^2 - 12 * t + 64 = 0 ∧ 0 ≤ t) → t = 2 :=
by
  sorry

end ball_hits_ground_at_t_l121_121693


namespace acute_triangle_l121_121050

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ area, 
    (area = (1 / 2) * a * b * Real.sin C) ∧
    (a / Real.sin A = 2 * c / Real.sqrt 3) ∧
    (c = Real.sqrt 7) ∧
    (area = (3 * Real.sqrt 3) / 2)

theorem acute_triangle (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  C = 60 ∧ a^2 + b^2 = 13 :=
by
  obtain ⟨_, h_area, h_sine, h_c, h_area_eq⟩ := h
  sorry

end acute_triangle_l121_121050


namespace max_value_of_expression_l121_121875

noncomputable def max_expression_value (x y : ℝ) : ℝ :=
  let expr := x^2 + 6 * y + 2
  14

theorem max_value_of_expression 
  (x y : ℝ) (h : x^2 + y^2 = 4) : ∃ (M : ℝ), M = 14 ∧ ∀ x y, x^2 + y^2 = 4 → x^2 + 6 * y + 2 ≤ M :=
  by
    use 14
    sorry

end max_value_of_expression_l121_121875


namespace solve_quadratic_abs_l121_121054

theorem solve_quadratic_abs (x : ℝ) :
  x^2 - |x| - 1 = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 ∨ 
                   x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

end solve_quadratic_abs_l121_121054


namespace sum_of_consecutive_integers_l121_121656

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l121_121656


namespace students_left_correct_l121_121002

-- Define the initial number of students
def initial_students : ℕ := 8

-- Define the number of new students
def new_students : ℕ := 8

-- Define the final number of students
def final_students : ℕ := 11

-- Define the number of students who left during the year
def students_who_left : ℕ :=
  (initial_students + new_students) - final_students

theorem students_left_correct : students_who_left = 5 :=
by
  -- Instantiating the definitions
  let initial := initial_students
  let new := new_students
  let final := final_students

  -- Calculation of students who left
  let L := (initial + new) - final

  -- Asserting the result
  show L = 5
  sorry

end students_left_correct_l121_121002


namespace actual_time_before_storm_is_18_18_l121_121434

theorem actual_time_before_storm_is_18_18 :
  ∃ h m : ℕ, (h = 18) ∧ (m = 18) ∧ 
            ((09 = (if h == 0 then 1 else h - 1) ∨ 09 = (if h == 23 then 0 else h + 1)) ∧ 
             (09 = (if m == 0 then 1 else m - 1) ∨ 09 = (if m == 59 then 0 else m + 1))) := 
  sorry

end actual_time_before_storm_is_18_18_l121_121434


namespace tan_alpha_eq_neg_sqrt_15_l121_121776

/-- Given α in the interval (0, π) and the equation tan(2α) = sin(α) / (2 + cos(α)), prove that tan(α) = -√15. -/
theorem tan_alpha_eq_neg_sqrt_15 (α : ℝ) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end tan_alpha_eq_neg_sqrt_15_l121_121776


namespace sin_120_eq_sqrt3_div_2_l121_121831

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l121_121831


namespace current_population_l121_121819

def initial_population : ℕ := 4200
def percentage_died : ℕ := 10
def percentage_left : ℕ := 15

theorem current_population (pop : ℕ) (died left : ℕ) 
  (h1 : pop = initial_population) 
  (h2 : died = pop * percentage_died / 100) 
  (h3 : left = (pop - died) * percentage_left / 100) 
  (h4 : ∀ remaining, remaining = pop - died - left) 
  : (pop - died - left) = 3213 := 
by sorry

end current_population_l121_121819


namespace function_increasing_probability_l121_121979

noncomputable def is_increasing_on_interval (a b : ℤ) : Prop :=
∀ x : ℝ, x > 1 → 2 * a * x - 2 * b > 0

noncomputable def valid_pairs : List (ℤ × ℤ) :=
[(0, -1), (1, -1), (1, 1), (2, -1), (2, 1)]

noncomputable def total_pairs : ℕ :=
3 * 4

noncomputable def probability_of_increasing_function : ℚ :=
(valid_pairs.length : ℚ) / total_pairs

theorem function_increasing_probability :
  probability_of_increasing_function = 5 / 12 :=
by
  sorry

end function_increasing_probability_l121_121979


namespace max_alpha_beta_square_l121_121058

theorem max_alpha_beta_square (k : ℝ) (α β : ℝ)
  (h1 : α^2 - (k - 2) * α + (k^2 + 3 * k + 5) = 0)
  (h2 : β^2 - (k - 2) * β + (k^2 + 3 * k + 5) = 0)
  (h3 : α ≠ β) :
  (α^2 + β^2) ≤ 18 :=
sorry

end max_alpha_beta_square_l121_121058


namespace point_cannot_exist_on_line_l121_121118

theorem point_cannot_exist_on_line (m k : ℝ) (h : m * k > 0) : ¬ (2000 * m + k = 0) :=
sorry

end point_cannot_exist_on_line_l121_121118


namespace polynomial_multiplication_equiv_l121_121699

theorem polynomial_multiplication_equiv (x : ℝ) : 
  (x^4 + 50*x^2 + 625)*(x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := 
by 
  sorry

end polynomial_multiplication_equiv_l121_121699


namespace vertex_of_parabola_l121_121992

theorem vertex_of_parabola : 
  (exists (a b: ℝ), ∀ x: ℝ, (a * (x - 1)^2 + b = (x - 1)^2 - 2)) → (1, -2) = (1, -2) :=
by
  intro h
  sorry

end vertex_of_parabola_l121_121992


namespace cakes_remain_l121_121801

def initial_cakes := 110
def sold_cakes := 75
def new_cakes := 76

theorem cakes_remain : (initial_cakes - sold_cakes) + new_cakes = 111 :=
by
  sorry

end cakes_remain_l121_121801


namespace range_of_m_l121_121151

def positive_numbers (a b : ℝ) : Prop := a > 0 ∧ b > 0

def equation_condition (a b : ℝ) : Prop := 9 * a + b = a * b

def inequality_for_any_x (a b m : ℝ) : Prop := ∀ x : ℝ, a + b ≥ -x^2 + 2 * x + 18 - m

theorem range_of_m :
  ∀ (a b m : ℝ),
    positive_numbers a b →
    equation_condition a b →
    inequality_for_any_x a b m →
    m ≥ 3 :=
by
  sorry

end range_of_m_l121_121151


namespace S10_value_l121_121252

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ :=
  x^m + (1 / x)^m

theorem S10_value (x : ℝ) (h : x + 1/x = 5) : 
  S_m x 10 = 6430223 := by 
  sorry

end S10_value_l121_121252


namespace sufficient_condition_l121_121849

theorem sufficient_condition 
  (x y z : ℤ)
  (H : x = y ∧ y = z)
  : x * (x - y) + y * (y - z) + z * (z - x) = 0 :=
by 
  sorry

end sufficient_condition_l121_121849


namespace total_winter_clothing_l121_121745

def num_scarves (boxes : ℕ) (scarves_per_box : ℕ) : ℕ := boxes * scarves_per_box
def num_mittens (boxes : ℕ) (mittens_per_box : ℕ) : ℕ := boxes * mittens_per_box
def num_hats (boxes : ℕ) (hats_per_box : ℕ) : ℕ := boxes * hats_per_box
def num_jackets (boxes : ℕ) (jackets_per_box : ℕ) : ℕ := boxes * jackets_per_box

theorem total_winter_clothing :
    num_scarves 4 8 + num_mittens 3 6 + num_hats 2 5 + num_jackets 1 3 = 63 :=
by
  -- The proof will use the given definitions and calculate the total
  sorry

end total_winter_clothing_l121_121745


namespace michael_earnings_l121_121471

-- Define variables for pay rates and hours.
def regular_pay_rate : ℝ := 7.00
def overtime_multiplier : ℝ := 2
def regular_hours : ℝ := 40
def overtime_hours (total_hours : ℝ) : ℝ := total_hours - regular_hours

-- Define the earnings functions.
def regular_earnings (hourly_rate : ℝ) (hours : ℝ) : ℝ := hourly_rate * hours
def overtime_earnings (hourly_rate : ℝ) (multiplier : ℝ) (hours : ℝ) : ℝ := hourly_rate * multiplier * hours

-- Total earnings calculation.
def total_earnings (total_hours : ℝ) : ℝ := 
regular_earnings regular_pay_rate regular_hours + 
overtime_earnings regular_pay_rate overtime_multiplier (overtime_hours total_hours)

-- The theorem to prove the correct earnings for 42.857142857142854 hours worked.
theorem michael_earnings : total_earnings 42.857142857142854 = 320 := by
  sorry

end michael_earnings_l121_121471


namespace children_tickets_sold_l121_121203

-- Given conditions
variables (A C : ℕ) -- A represents the number of adult tickets, C the number of children tickets.
variables (total_money total_tickets price_adult price_children : ℕ)
variables (total_money_eq : total_money = 104)
variables (total_tickets_eq : total_tickets = 21)
variables (price_adult_eq : price_adult = 6)
variables (price_children_eq : price_children = 4)
variables (money_eq : price_adult * A + price_children * C = total_money)
variables (tickets_eq : A + C = total_tickets)

-- Problem statement: prove that C = 11
theorem children_tickets_sold : C = 11 :=
by
  -- Necessary Lean code to handle proof here (omitting proof details as instructed)
  sorry

end children_tickets_sold_l121_121203


namespace jamal_green_marbles_l121_121728

theorem jamal_green_marbles
  (Y B K T : ℕ)
  (hY : Y = 12)
  (hB : B = 10)
  (hK : K = 1)
  (h_total : 1 / T = 1 / 28) :
  T - (Y + B + K) = 5 :=
by
  -- sorry, proof goes here
  sorry

end jamal_green_marbles_l121_121728


namespace triangle_angles_correct_l121_121689

open Real

noncomputable def angle_triple (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = 2 * b * cos C ∧ 
    sin A * sin (B / 2 + C) = sin C * (sin (B / 2) + sin A)

theorem triangle_angles_correct (A B C : ℝ) (h : angle_triple A B C) :
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := 
sorry

end triangle_angles_correct_l121_121689


namespace head_start_fraction_of_length_l121_121200

-- Define the necessary variables and assumptions.
variables (Va Vb L H : ℝ)

-- Given conditions
def condition_speed_relation : Prop := Va = (22 / 19) * Vb
def condition_dead_heat : Prop := (L / Va) = ((L - H) / Vb)

-- The statement to be proven
theorem head_start_fraction_of_length (h_speed_relation: condition_speed_relation Va Vb) (h_dead_heat: condition_dead_heat L Va H Vb) : 
  H = (3 / 22) * L :=
sorry

end head_start_fraction_of_length_l121_121200


namespace find_k_l121_121442

theorem find_k (d : ℤ) (h : d ≠ 0) (a : ℤ → ℤ) 
  (a_def : ∀ n, a n = 4 * d + (n - 1) * d) 
  (geom_mean_condition : ∃ k, a k * a k = a 1 * a 6) : 
  ∃ k, k = 3 := 
by
  sorry

end find_k_l121_121442


namespace triangle_ABC_is_acute_l121_121694

theorem triangle_ABC_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h1: a^2 + b^2 >= c^2) (h2: b^2 + c^2 >= a^2) (h3: c^2 + a^2 >= b^2)
  (h4: (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11)
  (h5: (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) : 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 :=
sorry

end triangle_ABC_is_acute_l121_121694


namespace max_marks_l121_121534

variable (M : ℝ)

-- Conditions
def needed_to_pass (M : ℝ) := 0.20 * M
def pradeep_marks := 390
def marks_short := 25
def total_marks_needed := pradeep_marks + marks_short

-- Theorem statement
theorem max_marks : needed_to_pass M = total_marks_needed → M = 2075 := by
  sorry

end max_marks_l121_121534


namespace required_run_rate_is_correct_l121_121199

open Nat

noncomputable def requiredRunRate (initialRunRate : ℝ) (initialOvers : ℕ) (targetRuns : ℕ) (totalOvers : ℕ) : ℝ :=
  let runsScored := initialRunRate * initialOvers
  let runsNeeded := targetRuns - runsScored
  let remainingOvers := totalOvers - initialOvers
  runsNeeded / (remainingOvers : ℝ)

theorem required_run_rate_is_correct :
  (requiredRunRate 3.6 10 282 50 = 6.15) :=
by
  sorry

end required_run_rate_is_correct_l121_121199


namespace trig_identity_simplify_l121_121191

-- Define the problem in Lean 4
theorem trig_identity_simplify (α : Real) : (Real.sin (α - Real.pi / 2) * Real.tan (Real.pi - α)) = Real.sin α :=
by
  sorry

end trig_identity_simplify_l121_121191


namespace geometric_series_first_term_l121_121190

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 24)
  (h_sum : S = a / (1 - r)) : 
  a = 18 :=
by {
  -- valid proof body goes here
  sorry
}

end geometric_series_first_term_l121_121190


namespace solution_set_of_inequality_l121_121779

theorem solution_set_of_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ (x > 2 ∨ x < -1) :=
by
  sorry

end solution_set_of_inequality_l121_121779


namespace work_completion_days_l121_121505

theorem work_completion_days (a b : Type) (T : ℕ) (ha : T = 12) (hb : T = 6) : 
  (T = 4) :=
sorry

end work_completion_days_l121_121505


namespace least_integer_x_l121_121510

theorem least_integer_x (x : ℤ) (h : 240 ∣ x^2) : x = 60 :=
sorry

end least_integer_x_l121_121510


namespace minimum_resistors_required_l121_121987

-- Define the grid configuration and the connectivity condition
def isReliableGrid (m : ℕ) (n : ℕ) (failures : Finset (ℕ × ℕ)) : Prop :=
m * n > 9 ∧ (∀ (a b : ℕ), a ≠ b → (a, b) ∉ failures)

-- Minimum number of resistors ensuring connectivity with up to 9 failures
theorem minimum_resistors_required :
  ∃ (m n : ℕ), 5 * 5 = 25 ∧ isReliableGrid 5 5 ∅ :=
by
  let m : ℕ := 5
  let n : ℕ := 5
  have h₁ : m * n = 25 := by rfl
  have h₂ : isReliableGrid 5 5 ∅ := by
    unfold isReliableGrid
    exact ⟨by norm_num, sorry⟩ -- formal proof omitted for brevity
  exact ⟨m, n, h₁, h₂⟩

end minimum_resistors_required_l121_121987


namespace calculate_new_shipment_bears_l121_121347

theorem calculate_new_shipment_bears 
  (initial_bears : ℕ)
  (shelves : ℕ)
  (bears_per_shelf : ℕ)
  (total_bears_on_shelves : ℕ) 
  (h_total_bears_on_shelves : total_bears_on_shelves = shelves * bears_per_shelf)
  : initial_bears = 6 → shelves = 4 → bears_per_shelf = 6 → total_bears_on_shelves - initial_bears = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end calculate_new_shipment_bears_l121_121347


namespace required_weekly_hours_approx_27_l121_121978

noncomputable def planned_hours_per_week : ℝ := 25
noncomputable def planned_weeks : ℝ := 15
noncomputable def total_amount : ℝ := 4500
noncomputable def sick_weeks : ℝ := 3
noncomputable def increased_wage_weeks : ℝ := 5
noncomputable def wage_increase_factor : ℝ := 1.5 -- 50%

-- Normal hourly wage
noncomputable def normal_hourly_wage : ℝ := total_amount / (planned_hours_per_week * planned_weeks)

-- Increased hourly wage
noncomputable def increased_hourly_wage : ℝ := normal_hourly_wage * wage_increase_factor

-- Earnings in the last 5 weeks at increased wage
noncomputable def earnings_in_last_5_weeks : ℝ := increased_hourly_wage * planned_hours_per_week * increased_wage_weeks

-- Amount needed before the wage increase
noncomputable def amount_needed_before_wage_increase : ℝ := total_amount - earnings_in_last_5_weeks

-- We have 7 weeks before the wage increase
noncomputable def weeks_before_increase : ℝ := planned_weeks - sick_weeks - increased_wage_weeks

-- New required weekly hours before wage increase
noncomputable def required_weekly_hours : ℝ := amount_needed_before_wage_increase / (normal_hourly_wage * weeks_before_increase)

theorem required_weekly_hours_approx_27 :
  abs (required_weekly_hours - 27) < 1 :=
sorry

end required_weekly_hours_approx_27_l121_121978


namespace greatest_possible_value_q_minus_r_l121_121029

theorem greatest_possible_value_q_minus_r : ∃ q r : ℕ, 1025 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
by {
  sorry
}

end greatest_possible_value_q_minus_r_l121_121029


namespace subset_condition_for_A_B_l121_121020

open Set

theorem subset_condition_for_A_B {a : ℝ} (A B : Set ℝ) 
  (hA : A = {x | abs (x - 2) < a}) 
  (hB : B = {x | x^2 - 2 * x - 3 < 0}) :
  B ⊆ A ↔ 3 ≤ a :=
  sorry

end subset_condition_for_A_B_l121_121020


namespace pairs_of_socks_calculation_l121_121732

variable (num_pairs_socks : ℤ)
variable (cost_per_pair : ℤ := 950) -- in cents
variable (cost_shoes : ℤ := 9200) -- in cents
variable (money_jack_has : ℤ := 4000) -- in cents
variable (money_needed : ℤ := 7100) -- in cents
variable (total_money_needed : ℤ := money_jack_has + money_needed)

theorem pairs_of_socks_calculation (x : ℤ) (h : cost_per_pair * x + cost_shoes = total_money_needed) : x = 2 :=
by
  sorry

end pairs_of_socks_calculation_l121_121732


namespace peter_speed_l121_121390

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end peter_speed_l121_121390


namespace unique_peg_placement_l121_121579

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

end unique_peg_placement_l121_121579


namespace total_crayons_l121_121721

-- Define the number of crayons Billy has
def billy_crayons : ℝ := 62.0

-- Define the number of crayons Jane has
def jane_crayons : ℝ := 52.0

-- Formulate the theorem to prove the total number of crayons
theorem total_crayons : billy_crayons + jane_crayons = 114.0 := by
  sorry

end total_crayons_l121_121721


namespace intersection_M_N_complement_N_U_l121_121106

-- Definitions for the sets and the universal set
def U := Set ℝ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def N : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) } -- Simplified domain interpretation for N

-- Intersection and complement calculations
theorem intersection_M_N (x : ℝ) : x ∈ M ∧ x ∈ N ↔ x ∈ { x | -2 ≤ x ∧ x ≤ 1 } := by sorry

theorem complement_N_U (x : ℝ) : x ∉ N ↔ x ∈ { x | x > 1 } := by sorry

end intersection_M_N_complement_N_U_l121_121106


namespace frog_probability_0_4_l121_121523

-- Definitions and conditions
def vertices : List (ℤ × ℤ) := [(1,1), (1,6), (5,6), (5,1)]
def start_position : ℤ × ℤ := (2,3)

-- Probabilities for transition, boundary definitions, this mimics the recursive nature described
def P : ℤ × ℤ → ℝ
| (x, 1) => 1   -- Boundary condition for horizontal sides
| (x, 6) => 1   -- Boundary condition for horizontal sides
| (1, y) => 0   -- Boundary condition for vertical sides
| (5, y) => 0   -- Boundary condition for vertical sides
| (x, y) => sorry  -- General case for other positions

-- The theorem to prove
theorem frog_probability_0_4 : P (2, 3) = 0.4 :=
by
  sorry

end frog_probability_0_4_l121_121523


namespace find_phi_l121_121010

theorem find_phi 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + Real.pi / 6))
  (h2 : 0 < phi ∧ phi < Real.pi / 2)
  (h3 : ∀ x, y x = f (x - phi) ∧ y x = y (-x)) :
  phi = Real.pi / 3 :=
by
  sorry

end find_phi_l121_121010


namespace sets_satisfy_union_l121_121799

theorem sets_satisfy_union (A : Set Int) : (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (X : Finset (Set Int)), X.card = 4 ∧ ∀ B ∈ X, A = B) :=
  sorry

end sets_satisfy_union_l121_121799


namespace warehouse_painted_area_l121_121331

theorem warehouse_painted_area :
  let length := 8
  let width := 6
  let height := 3.5
  let door_width := 1
  let door_height := 2
  let front_back_area := 2 * (length * height)
  let left_right_area := 2 * (width * height)
  let total_wall_area := front_back_area + left_right_area
  let door_area := door_width * door_height
  let painted_area := total_wall_area - door_area
  painted_area = 96 :=
by
  -- Sorry to skip the actual proof steps
  sorry

end warehouse_painted_area_l121_121331


namespace diagonal_BD_l121_121336

variables {A B C D : Point}
variables {AB BC BE : ℝ}
variables {parallelogram : ABCD A B C D}

-- Conditions
def side_AB : AB = 3 := sorry
def side_BC : BC = 5 := sorry
def intersection_BE : BE = 9 := sorry

-- Goal 
theorem diagonal_BD : ∀ (BD : ℝ), BD = 34 / 9 :=
by sorry

end diagonal_BD_l121_121336


namespace rowing_upstream_speed_l121_121448

-- Definitions based on conditions
def V_m : ℝ := 45 -- speed of the man in still water
def V_downstream : ℝ := 53 -- speed of the man rowing downstream
def V_s : ℝ := V_downstream - V_m -- speed of the stream
def V_upstream : ℝ := V_m - V_s -- speed of the man rowing upstream

-- The goal is to prove that the speed of the man rowing upstream is 37 kmph
theorem rowing_upstream_speed :
  V_upstream = 37 := by
  sorry

end rowing_upstream_speed_l121_121448


namespace value_of_x_l121_121865

theorem value_of_x (x : ℝ) (h : x = 12 + (20 / 100) * 12) : x = 14.4 :=
by sorry

end value_of_x_l121_121865


namespace fraction_given_to_jerry_l121_121845

-- Define the problem conditions
def initial_apples := 2
def slices_per_apple := 8
def total_slices := initial_apples * slices_per_apple -- 2 * 8 = 16

def remaining_slices_after_eating := 5
def slices_before_eating := remaining_slices_after_eating * 2 -- 5 * 2 = 10
def slices_given_to_jerry := total_slices - slices_before_eating -- 16 - 10 = 6

-- Define the proof statement to verify that the fraction of slices given to Jerry is 3/8
theorem fraction_given_to_jerry : (slices_given_to_jerry : ℚ) / total_slices = 3 / 8 :=
by
  -- skip the actual proof, just outline the goal
  sorry

end fraction_given_to_jerry_l121_121845


namespace ghost_enter_exit_ways_l121_121265

theorem ghost_enter_exit_ways : 
  (∃ (enter_win : ℕ) (exit_win : ℕ), enter_win ≠ exit_win ∧ 1 ≤ enter_win ∧ enter_win ≤ 8 ∧ 1 ≤ exit_win ∧ exit_win ≤ 8) →
  ∃ (ways : ℕ), ways = 8 * 7 :=
by
  sorry

end ghost_enter_exit_ways_l121_121265


namespace total_books_l121_121897

-- Define the number of books Victor originally had and the number he bought
def original_books : ℕ := 9
def bought_books : ℕ := 3

-- The proof problem statement: Prove Victor has a total of original_books + bought_books books
theorem total_books : original_books + bought_books = 12 := by
  -- proof will go here, using sorry to indicate it's omitted
  sorry

end total_books_l121_121897


namespace calendar_reuse_initial_year_l121_121954

theorem calendar_reuse_initial_year (y k : ℕ)
    (h2064 : 2052 % 4 = 0)
    (h_y: y + 28 * k = 2052) :
    y = 1912 := by
  sorry

end calendar_reuse_initial_year_l121_121954


namespace sqrt_condition_l121_121512

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_condition_l121_121512


namespace probability_different_colors_is_correct_l121_121532

-- Definitions of chip counts
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def green_chips := 3
def total_chips := blue_chips + red_chips + yellow_chips + green_chips

-- Definition of the probability calculation
def probability_different_colors := 
  ((blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)) +
  ((red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)) +
  ((yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)) +
  ((green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips))

-- Given the problem conditions, we assert the correct answer
theorem probability_different_colors_is_correct :
  probability_different_colors = (119 / 162) := 
sorry

end probability_different_colors_is_correct_l121_121532


namespace scale_model_height_l121_121765

/-- 
Given a scale model ratio and the actual height of the skyscraper in feet,
we can deduce the height of the model in inches.
-/
theorem scale_model_height
  (scale_ratio : ℕ := 25)
  (actual_height_feet : ℕ := 1250) :
  (actual_height_feet / scale_ratio) * 12 = 600 :=
by 
  sorry

end scale_model_height_l121_121765


namespace length_of_each_song_l121_121553

-- Conditions
def first_side_songs : Nat := 6
def second_side_songs : Nat := 4
def total_length_of_tape : Nat := 40

-- Definition of length of each song
def total_songs := first_side_songs + second_side_songs

-- Question: Prove that each song is 4 minutes long
theorem length_of_each_song (h1 : first_side_songs = 6) 
                            (h2 : second_side_songs = 4) 
                            (h3 : total_length_of_tape = 40) 
                            (h4 : total_songs = first_side_songs + second_side_songs) : 
  total_length_of_tape / total_songs = 4 :=
by
  sorry

end length_of_each_song_l121_121553


namespace PQ_PR_QR_div_l121_121828

theorem PQ_PR_QR_div (p q r : ℝ)
    (midQR : p = 0) (midPR : q = 0) (midPQ : r = 0) :
    (4 * (q ^ 2 + r ^ 2) + 4 * (p ^ 2 + r ^ 2) + 4 * (p ^ 2 + q ^ 2)) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 :=
by {
    sorry
}

end PQ_PR_QR_div_l121_121828


namespace fraction_to_decimal_l121_121333

theorem fraction_to_decimal : (5 / 8 : ℝ) = 0.625 := 
  by sorry

end fraction_to_decimal_l121_121333


namespace people_sitting_between_same_l121_121285

theorem people_sitting_between_same 
  (n : ℕ) (h_even : n % 2 = 0) 
  (f : Fin (2 * n) → Fin (2 * n)) :
  ∃ (a b : Fin (2 * n)), 
  ∃ (k k' : ℕ), k < 2 * n ∧ k' < 2 * n ∧ (a : ℕ) < (b : ℕ) ∧ 
  ((b - a = k) ∧ (f b - f a = k)) ∨ ((a - b + 2*n = k') ∧ ((f a - f b + 2 * n) % (2 * n) = k')) :=
by
  sorry

end people_sitting_between_same_l121_121285


namespace Rover_has_46_spots_l121_121238

theorem Rover_has_46_spots (G C R : ℕ) 
  (h1 : G = 5 * C)
  (h2 : C = (1/2 : ℝ) * R - 5)
  (h3 : G + C = 108) : 
  R = 46 :=
by
  sorry

end Rover_has_46_spots_l121_121238


namespace num_whole_numbers_between_sqrt_50_and_sqrt_200_l121_121077

theorem num_whole_numbers_between_sqrt_50_and_sqrt_200 :
  let lower := Nat.ceil (Real.sqrt 50)
  let upper := Nat.floor (Real.sqrt 200)
  lower <= upper ∧ (upper - lower + 1) = 7 :=
by
  sorry

end num_whole_numbers_between_sqrt_50_and_sqrt_200_l121_121077


namespace minimum_value_problem_l121_121932

theorem minimum_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 + 4 * x^2 + 2 * x + 1) * (y^3 + 4 * y^2 + 2 * y + 1) * (z^3 + 4 * z^2 + 2 * z + 1) / (x * y * z) ≥ 1331 :=
sorry

end minimum_value_problem_l121_121932


namespace prob_first_three_heads_all_heads_l121_121250

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l121_121250


namespace household_peak_consumption_l121_121574

theorem household_peak_consumption
  (p_orig p_peak p_offpeak : ℝ)
  (consumption : ℝ)
  (monthly_savings : ℝ)
  (x : ℝ)
  (h_orig : p_orig = 0.52)
  (h_peak : p_peak = 0.55)
  (h_offpeak : p_offpeak = 0.35)
  (h_consumption : consumption = 200)
  (h_savings : monthly_savings = 0.10) :
  (p_orig - p_peak) * x + (p_orig - p_offpeak) * (consumption - x) ≥ p_orig * consumption * monthly_savings → x ≤ 118 :=
sorry

end household_peak_consumption_l121_121574


namespace parabola_latus_rectum_l121_121062

theorem parabola_latus_rectum (p : ℝ) (H : ∀ y : ℝ, y^2 = 2 * p * -2) : p = 4 :=
sorry

end parabola_latus_rectum_l121_121062


namespace intersection_point_of_y_eq_4x_minus_2_with_x_axis_l121_121007

theorem intersection_point_of_y_eq_4x_minus_2_with_x_axis :
  ∃ x, (4 * x - 2 = 0 ∧ (x, 0) = (1 / 2, 0)) :=
by
  sorry

end intersection_point_of_y_eq_4x_minus_2_with_x_axis_l121_121007


namespace find_x_l121_121668

theorem find_x (x : ℝ) :
  let P1 := (2, 10)
  let P2 := (6, 2)
  
  -- Slope of the line joining (2, 10) and (6, 2)
  let slope12 := (P2.2 - P1.2) / (P2.1 - P1.1)
  
  -- Slope of the line joining (2, 10) and (x, -3)
  let P3 := (x, -3)
  let slope13 := (P3.2 - P1.2) / (P3.1 - P1.1)
  
  -- Condition that both slopes are equal
  slope12 = slope13
  
  -- To Prove: x must be 8.5
  → x = 8.5 :=
sorry

end find_x_l121_121668


namespace geometric_sequence_common_ratio_l121_121594

theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a (n + 1) = a n * q) →
  (∀ n m, n < m → a n < a m) →
  a 2 = 2 →
  a 4 - a 3 = 4 →
  q = 2 :=
by
  intros a q h_geo h_inc h_a2 h_a4_a3
  sorry

end geometric_sequence_common_ratio_l121_121594


namespace product_modulo_7_l121_121673

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l121_121673


namespace polar_to_rectangular_l121_121842

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 7) (h_θ : θ = π / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) :=
by 
  -- proof goes here
  sorry

end polar_to_rectangular_l121_121842


namespace determine_y_l121_121237

-- Define the main problem in a Lean theorem
theorem determine_y (y : ℕ) : 9^10 + 9^10 + 9^10 = 3^y → y = 21 :=
by
  -- proof not required, so we add sorry
  sorry

end determine_y_l121_121237


namespace sandy_total_puppies_l121_121643

-- Definitions based on conditions:
def original_puppies : ℝ := 8.0
def additional_puppies : ℝ := 4.0

-- Theorem statement: total_puppies should be 12.0
theorem sandy_total_puppies : original_puppies + additional_puppies = 12.0 := 
by
  sorry

end sandy_total_puppies_l121_121643


namespace goldbach_134_l121_121963

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0
noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem goldbach_134 (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum : p + q = 134) (h_diff : p ≠ q) : 
  ∃ (d : ℕ), d = 134 - (2 * p) ∧ d ≤ 128 := 
sorry

end goldbach_134_l121_121963


namespace range_of_f_l121_121256

def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f :
  Set.range f = {0, -1} := 
sorry

end range_of_f_l121_121256


namespace largest_root_in_interval_l121_121780

theorem largest_root_in_interval :
  ∃ (r : ℝ), (2 < r ∧ r < 3) ∧ (∃ (a_2 a_1 a_0 : ℝ), 
    |a_2| ≤ 3 ∧ |a_1| ≤ 3 ∧ |a_0| ≤ 3 ∧ a_2 + a_1 + a_0 = -6 ∧ r^3 + a_2 * r^2 + a_1 * r + a_0 = 0) :=
sorry

end largest_root_in_interval_l121_121780


namespace standard_equation_of_circle_tangent_to_x_axis_l121_121750

theorem standard_equation_of_circle_tangent_to_x_axis :
  ∀ (x y : ℝ), ((x + 3) ^ 2 + (y - 4) ^ 2 = 16) :=
by
  -- Definitions based on the conditions
  let center_x := -3
  let center_y := 4
  let radius := 4

  sorry

end standard_equation_of_circle_tangent_to_x_axis_l121_121750


namespace arithmetic_sqrt_of_4_eq_2_l121_121599

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end arithmetic_sqrt_of_4_eq_2_l121_121599


namespace factorize_expr_l121_121290

theorem factorize_expr (a : ℝ) : a^2 - 8 * a = a * (a - 8) :=
sorry

end factorize_expr_l121_121290


namespace annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l121_121399

-- Define principal amounts for Paul, Emma and Harry
def principalPaul : ℚ := 5000
def principalEmma : ℚ := 3000
def principalHarry : ℚ := 7000

-- Define time periods for Paul, Emma and Harry
def timePaul : ℚ := 2
def timeEmma : ℚ := 4
def timeHarry : ℚ := 3

-- Define interests received from Paul, Emma and Harry
def interestPaul : ℚ := 2200
def interestEmma : ℚ := 3400
def interestHarry : ℚ := 3900

-- Define the simple interest formula 
def simpleInterest (P : ℚ) (R : ℚ) (T : ℚ) : ℚ := P * R * T

-- Prove the annual interest rates for each loan 
theorem annual_interest_rate_Paul : 
  ∃ (R : ℚ), simpleInterest principalPaul R timePaul = interestPaul ∧ R = 0.22 := 
by
  sorry

theorem annual_interest_rate_Emma : 
  ∃ (R : ℚ), simpleInterest principalEmma R timeEmma = interestEmma ∧ R = 0.2833 := 
by
  sorry

theorem annual_interest_rate_Harry : 
  ∃ (R : ℚ), simpleInterest principalHarry R timeHarry = interestHarry ∧ R = 0.1857 := 
by
  sorry

end annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l121_121399


namespace grandmaster_plays_21_games_l121_121958

theorem grandmaster_plays_21_games (a : ℕ → ℕ) (n : ℕ) :
  (∀ i, 1 ≤ a (i + 1) - a i) ∧ (∀ i, a (i + 7) - a i ≤ 10) →
  ∃ (i j : ℕ), i < j ∧ (a j - a i = 21) :=
sorry

end grandmaster_plays_21_games_l121_121958


namespace meaningful_expr_iff_x_ne_neg_5_l121_121411

theorem meaningful_expr_iff_x_ne_neg_5 (x : ℝ) : (x + 5 ≠ 0) ↔ (x ≠ -5) :=
by
  sorry

end meaningful_expr_iff_x_ne_neg_5_l121_121411


namespace new_rate_ratio_l121_121714

/--
Hephaestus charged 3 golden apples for the first six months and raised his rate halfway through the year.
Apollo paid 54 golden apples in total for the entire year.
The ratio of the new rate to the old rate is 2.
-/
theorem new_rate_ratio
  (old_rate new_rate : ℕ)
  (total_payment : ℕ)
  (H1 : old_rate = 3)
  (H2 : total_payment = 54)
  (H3 : ∀ R : ℕ, new_rate = R * old_rate ∧ total_payment = 18 + 18 * R) :
  ∃ (R : ℕ), R = 2 :=
by {
  sorry
}

end new_rate_ratio_l121_121714


namespace intersection_A_B_l121_121758

def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | -1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_A_B_l121_121758


namespace smallest_non_factor_product_of_48_l121_121387

theorem smallest_non_factor_product_of_48 :
  ∃ (x y : ℕ), x ≠ y ∧ x * y ≤ 48 ∧ (x ∣ 48) ∧ (y ∣ 48) ∧ ¬ (x * y ∣ 48) ∧ x * y = 18 :=
by
  sorry

end smallest_non_factor_product_of_48_l121_121387


namespace determine_m_value_l121_121071

theorem determine_m_value
  (m : ℝ)
  (h : ∀ x : ℝ, -7 < x ∧ x < -1 ↔ mx^2 + 8 * m * x + 28 < 0) :
  m = 4 := by
  sorry

end determine_m_value_l121_121071


namespace time_difference_l121_121551

-- Define the capacity of the tanks
def capacity : ℕ := 20

-- Define the inflow rates of tanks A and B in litres per hour
def inflow_rate_A : ℕ := 2
def inflow_rate_B : ℕ := 4

-- Define the times to fill tanks A and B
def time_A : ℕ := capacity / inflow_rate_A
def time_B : ℕ := capacity / inflow_rate_B

-- Proving the time difference between filling tanks A and B
theorem time_difference : (time_A - time_B) = 5 := by
  sorry

end time_difference_l121_121551


namespace tangent_line_and_point_l121_121144

theorem tangent_line_and_point (x0 y0 k: ℝ) (hx0 : x0 ≠ 0) 
  (hC : y0 = x0^3 - 3 * x0^2 + 2 * x0) (hl : y0 = k * x0) 
  (hk_tangent : k = 3 * x0^2 - 6 * x0 + 2) : 
  (k = -1/4) ∧ (x0 = 3/2) ∧ (y0 = -3/8) :=
by
  sorry

end tangent_line_and_point_l121_121144


namespace concentration_of_first_solution_l121_121948

theorem concentration_of_first_solution
  (C : ℝ)
  (h : 4 * (C / 100) + 0.2 = 0.36) :
  C = 4 :=
by
  sorry

end concentration_of_first_solution_l121_121948


namespace evaluate_expression_l121_121196

theorem evaluate_expression (a : ℝ) (h : a = 4 / 3) : 
  (4 * a^2 - 12 * a + 9) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l121_121196


namespace circle_circumference_l121_121572

theorem circle_circumference (a b : ℝ) (h1 : a = 9) (h2 : b = 12) :
  ∃ c : ℝ, c = 15 * Real.pi :=
by
  sorry

end circle_circumference_l121_121572


namespace Sandy_tokens_more_than_siblings_l121_121705

theorem Sandy_tokens_more_than_siblings :
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  -- Definitions as per conditions
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  -- Conclusion
  show Sandy_tokens - sibling_tokens = 375000
  sorry

end Sandy_tokens_more_than_siblings_l121_121705


namespace eastville_to_westpath_travel_time_l121_121701

theorem eastville_to_westpath_travel_time :
  ∀ (d t₁ t₂ : ℝ) (s₁ s₂ : ℝ), 
  t₁ = 6 → s₁ = 80 → s₂ = 50 → d = s₁ * t₁ → t₂ = d / s₂ → t₂ = 9.6 := 
by
  intros d t₁ t₂ s₁ s₂ ht₁ hs₁ hs₂ hd ht₂
  sorry

end eastville_to_westpath_travel_time_l121_121701


namespace rachel_picked_total_apples_l121_121413

-- Define the conditions
def num_trees : ℕ := 4
def apples_per_tree_picked : ℕ := 7
def apples_remaining : ℕ := 29

-- Define the total apples picked
def total_apples_picked : ℕ := num_trees * apples_per_tree_picked

-- Formal statement of the goal
theorem rachel_picked_total_apples : total_apples_picked = 28 := 
by
  sorry

end rachel_picked_total_apples_l121_121413


namespace explicit_form_of_function_l121_121723

theorem explicit_form_of_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f x + f x * f y + y - 1) = f (x * f x + x * y) + y - 1) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end explicit_form_of_function_l121_121723


namespace part1_part2_l121_121130

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end part1_part2_l121_121130


namespace tank_C_capacity_is_80_percent_of_tank_B_l121_121069

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := 
  Real.pi * r^2 * h

theorem tank_C_capacity_is_80_percent_of_tank_B :
  ∀ (h_C c_C h_B c_B : ℝ), 
    h_C = 10 ∧ c_C = 8 ∧ h_B = 8 ∧ c_B = 10 → 
    (volume_of_cylinder (c_C / (2 * Real.pi)) h_C) / 
    (volume_of_cylinder (c_B / (2 * Real.pi)) h_B) * 100 = 80 := 
by 
  intros h_C c_C h_B c_B h_conditions
  obtain ⟨h_C_10, c_C_8, h_B_8, c_B_10⟩ := h_conditions
  sorry

end tank_C_capacity_is_80_percent_of_tank_B_l121_121069


namespace maci_school_supplies_cost_l121_121184

theorem maci_school_supplies_cost :
  let blue_pen_cost := 0.10
  let red_pen_cost := 2 * blue_pen_cost
  let pencil_cost := red_pen_cost / 2
  let notebook_cost := 10 * blue_pen_cost
  let blue_pen_count := 10
  let red_pen_count := 15
  let pencil_count := 5
  let notebook_count := 3
  let total_pen_count := blue_pen_count + red_pen_count
  let total_cost_before_discount := 
      blue_pen_count * blue_pen_cost + 
      red_pen_count * red_pen_cost + 
      pencil_count * pencil_cost + 
      notebook_count * notebook_cost
  let pen_discount_rate := if total_pen_count > 12 then 0.10 else 0
  let notebook_discount_rate := if notebook_count > 4 then 0.20 else 0
  let pen_discount := pen_discount_rate * (blue_pen_count * blue_pen_cost + red_pen_count * red_pen_cost)
  let total_cost_after_discount := 
      total_cost_before_discount - pen_discount
  total_cost_after_discount = 7.10 :=
by
  sorry

end maci_school_supplies_cost_l121_121184


namespace expand_expression_l121_121105

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end expand_expression_l121_121105


namespace timothy_total_cost_l121_121141

-- Define the costs of the individual items
def costOfLand (acres : Nat) (cost_per_acre : Nat) : Nat :=
  acres * cost_per_acre

def costOfHouse : Nat :=
  120000

def costOfCows (number_of_cows : Nat) (cost_per_cow : Nat) : Nat :=
  number_of_cows * cost_per_cow

def costOfChickens (number_of_chickens : Nat) (cost_per_chicken : Nat) : Nat :=
  number_of_chickens * cost_per_chicken

def installationCost (hours : Nat) (cost_per_hour : Nat) (equipment_fee : Nat) : Nat :=
  (hours * cost_per_hour) + equipment_fee

-- Define the total cost function
def totalCost : Nat :=
  costOfLand 30 20 +
  costOfHouse +
  costOfCows 20 1000 +
  costOfChickens 100 5 +
  installationCost 6 100 6000

-- Theorem to state the total cost
theorem timothy_total_cost : totalCost = 147700 :=
by
  -- Placeholder for the proof, for now leave it as sorry
  sorry

end timothy_total_cost_l121_121141


namespace fraction_value_is_one_fourth_l121_121892

theorem fraction_value_is_one_fourth (k : Nat) (hk : k ≥ 1) :
  (10^k + 6 * (10^k - 1) / 9) / (60 * (10^k - 1) / 9 + 4) = 1 / 4 :=
by
  sorry

end fraction_value_is_one_fourth_l121_121892


namespace number_of_terminating_decimals_l121_121678

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end number_of_terminating_decimals_l121_121678


namespace analogical_reasoning_correctness_l121_121641

theorem analogical_reasoning_correctness 
  (a b c : ℝ)
  (va vb vc : ℝ) :
  (a + b) * c = (a * c + b * c) ↔ 
  (va + vb) * vc = (va * vc + vb * vc) := 
sorry

end analogical_reasoning_correctness_l121_121641


namespace Rachel_total_score_l121_121946

theorem Rachel_total_score
    (points_per_treasure : ℕ)
    (treasures_first_level : ℕ)
    (treasures_second_level : ℕ)
    (h1 : points_per_treasure = 9)
    (h2 : treasures_first_level = 5)
    (h3 : treasures_second_level = 2) : 
    (points_per_treasure * treasures_first_level + points_per_treasure * treasures_second_level = 63) :=
by
    sorry

end Rachel_total_score_l121_121946


namespace max_value_of_f_l121_121609

noncomputable def f (x : ℝ) : ℝ :=
  2022 * x ^ 2 * Real.log (x + 2022) / ((Real.log (x + 2022)) ^ 3 + 2 * x ^ 3)

theorem max_value_of_f : ∃ x : ℝ, 0 < x ∧ f x ≤ 674 :=
by
  sorry

end max_value_of_f_l121_121609


namespace isosceles_triangle_l121_121697

theorem isosceles_triangle (a b c : ℝ) (h : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) : 
  a = b ∨ b = c ∨ c = a :=
sorry

end isosceles_triangle_l121_121697


namespace correct_operation_l121_121712

-- Definitions based on conditions
def exprA (a b : ℤ) : ℤ := 3 * a * b - a * b
def exprB (a : ℤ) : ℤ := -3 * a^2 - 5 * a^2
def exprC (x : ℤ) : ℤ := -3 * x - 2 * x

-- Statement to prove that exprB is correct
theorem correct_operation (a : ℤ) : exprB a = -8 * a^2 := by
  sorry

end correct_operation_l121_121712


namespace increasing_sequence_a_range_l121_121380

theorem increasing_sequence_a_range (f : ℕ → ℝ) (a : ℝ)
  (h1 : ∀ n, f n = if n ≤ 7 then (3 - a) * n - 3 else a ^ (n - 6))
  (h2 : ∀ n : ℕ, f n < f (n + 1)) :
  2 < a ∧ a < 3 :=
sorry

end increasing_sequence_a_range_l121_121380


namespace minimum_a_l121_121311

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 1)
noncomputable def g (x a : ℝ) : ℝ := f x - a

theorem minimum_a (a : ℝ) : (∃ x : ℝ, g x a = 0) ↔ (a ≥ 1) :=
by sorry

end minimum_a_l121_121311


namespace rate_of_descent_correct_l121_121682

def depth := 3500 -- in feet
def time := 100 -- in minutes

def rate_of_descent : ℕ := depth / time

theorem rate_of_descent_correct : rate_of_descent = 35 := by
  -- We intentionally skip the proof part as per the requirement
  sorry

end rate_of_descent_correct_l121_121682


namespace units_digit_of_23_mul_51_squared_l121_121881

theorem units_digit_of_23_mul_51_squared : 
  ∀ n m : ℕ, (n % 10 = 3) ∧ ((m^2 % 10) = 1) → (n * m^2 % 10) = 3 :=
by
  intros n m h
  sorry

end units_digit_of_23_mul_51_squared_l121_121881


namespace shark_sightings_relationship_l121_121847

theorem shark_sightings_relationship (C D R : ℕ) (h₁ : C + D = 40) (h₂ : C = R - 8) (h₃ : C = 24) :
  R = 32 :=
by
  sorry

end shark_sightings_relationship_l121_121847


namespace sphere_hemisphere_radius_relationship_l121_121556

theorem sphere_hemisphere_radius_relationship (r : ℝ) (R : ℝ) (π : ℝ) (h : 0 < π):
  (4 / 3) * π * R^3 = (2 / 3) * π * r^3 →
  r = 3 * (2^(1/3 : ℝ)) →
  R = 3 :=
by
  sorry

end sphere_hemisphere_radius_relationship_l121_121556


namespace max_profit_price_l121_121183

-- Define the conditions
def hotel_rooms : ℕ := 50
def base_price : ℕ := 180
def price_increase : ℕ := 10
def expense_per_room : ℕ := 20

-- Define the price as a function of x
def room_price (x : ℕ) : ℕ := base_price + price_increase * x

-- Define the number of occupied rooms as a function of x
def occupied_rooms (x : ℕ) : ℕ := hotel_rooms - x

-- Define the profit function
def profit (x : ℕ) : ℕ := (room_price x - expense_per_room) * occupied_rooms x

-- The statement to be proven:
theorem max_profit_price : ∃ (x : ℕ), room_price x = 350 ∧ ∀ y : ℕ, profit y ≤ profit x :=
by
  sorry

end max_profit_price_l121_121183


namespace third_side_length_l121_121823

theorem third_side_length (x : ℝ) (h1 : 2 + 4 > x) (h2 : 4 + x > 2) (h3 : x + 2 > 4) : x = 4 :=
by {
  sorry
}

end third_side_length_l121_121823


namespace speed_of_train_A_l121_121339

noncomputable def train_speed_A (V_B : ℝ) (T_A T_B : ℝ) : ℝ :=
  (T_B / T_A) * V_B

theorem speed_of_train_A : train_speed_A 165 9 4 = 73.33 :=
by
  sorry

end speed_of_train_A_l121_121339


namespace reb_min_biking_speed_l121_121974

theorem reb_min_biking_speed (driving_time_minutes driving_speed driving_distance biking_distance_minutes biking_reduction_percentage biking_distance_hours : ℕ) 
  (driving_time_eqn: driving_time_minutes = 45) 
  (driving_speed_eqn: driving_speed = 40) 
  (driving_distance_eqn: driving_distance = driving_speed * driving_time_minutes / 60)
  (biking_reduction_percentage_eqn: biking_reduction_percentage = 20)
  (biking_distance_eqn: biking_distance = driving_distance * (100 - biking_reduction_percentage) / 100)
  (biking_distance_hours_eqn: biking_distance_minutes = 120)
  (biking_hours_eqn: biking_distance_hours = biking_distance_minutes / 60)
  : (biking_distance / biking_distance_hours) ≥ 12 := 
by
  sorry

end reb_min_biking_speed_l121_121974


namespace geometric_sequence_sum_l121_121559

theorem geometric_sequence_sum (a : Nat → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (hq : q > 1) (h2011_root : 4 * a 2011 ^ 2 - 8 * a 2011 + 3 = 0)
  (h2012_root : 4 * a 2012 ^ 2 - 8 * a 2012 + 3 = 0) :
  a 2013 + a 2014 = 18 :=
sorry

end geometric_sequence_sum_l121_121559


namespace polar_to_rectangular_l121_121996

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 5) (h₂ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (5 / 2, -5 * Real.sqrt 3 / 2) :=
by sorry

end polar_to_rectangular_l121_121996


namespace find_x_l121_121388

theorem find_x (x : ℝ) (h: 0.8 * 90 = 70 / 100 * x + 30) : x = 60 :=
by
  sorry

end find_x_l121_121388


namespace find_range_of_m_l121_121538

noncomputable def range_of_m (m : ℝ) : Prop :=
  ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m))

theorem find_range_of_m (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∨
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3)) ↔
  ¬((∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∧
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3))) →
  range_of_m m :=
sorry

end find_range_of_m_l121_121538


namespace probability_of_X_l121_121672

variable (P : Prop → ℝ)
variable (event_X event_Y : Prop)

-- Defining the conditions
variable (hYP : P event_Y = 2 / 3)
variable (hXYP : P (event_X ∧ event_Y) = 0.13333333333333333)

-- Proving that the probability of selection of X is 0.2
theorem probability_of_X : P event_X = 0.2 := by
  sorry

end probability_of_X_l121_121672


namespace marble_ratio_is_two_to_one_l121_121747

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end marble_ratio_is_two_to_one_l121_121747


namespace equation_of_parallel_line_l121_121792

theorem equation_of_parallel_line : 
  ∃ l : ℝ, (∀ x y : ℝ, 2 * x - 3 * y + 8 = 0 ↔ l = 2 * x - 3 * y + 8) :=
sorry

end equation_of_parallel_line_l121_121792


namespace gcd_polynomials_l121_121953

theorem gcd_polynomials (b : ℕ) (hb : ∃ k : ℕ, b = 2 * 7771 * k) :
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 19) = 8 :=
by sorry

end gcd_polynomials_l121_121953


namespace bowling_ball_weight_l121_121013

theorem bowling_ball_weight (b k : ℝ) (h1 : 5 * b = 3 * k) (h2 : 4 * k = 120) : b = 18 :=
by
  sorry

end bowling_ball_weight_l121_121013


namespace percentage_markup_l121_121709

theorem percentage_markup 
  (selling_price : ℝ) 
  (cost_price : ℝ) 
  (h1 : selling_price = 8215)
  (h2 : cost_price = 6625)
  : ((selling_price - cost_price) / cost_price) * 100 = 24 := 
  by
    sorry

end percentage_markup_l121_121709


namespace find_single_digit_number_l121_121756

-- Define the given conditions:
def single_digit (A : ℕ) := A < 10
def rounded_down_tens (x : ℕ) (result: ℕ) := (x / 10) * 10 = result

-- Lean statement of the problem:
theorem find_single_digit_number (A : ℕ) (H1 : single_digit A) (H2 : rounded_down_tens (A * 1000 + 567) 2560) : A = 2 :=
sorry

end find_single_digit_number_l121_121756


namespace largest_measureable_quantity_is_1_l121_121027

theorem largest_measureable_quantity_is_1 : 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd 496 403) 713) 824) 1171 = 1 :=
  sorry

end largest_measureable_quantity_is_1_l121_121027


namespace min_expr_value_l121_121087

theorem min_expr_value (a b c : ℝ) (h₀ : b > c) (h₁ : c > a) (h₂ : a > 0) (h₃ : b ≠ 0) :
  (∀ (a b c : ℝ), b > c → c > a → a > 0 → b ≠ 0 → 
   (2 + 6 * a^2 = (a+b)^3 / b^2 + (b-c)^2 / b^2 + (c-a)^3 / b^2) →
   2 <= (a + b)^3 / b^2 + (b - c)^2 / b^2 + (c - a)^3 / b^2) :=
by 
  sorry

end min_expr_value_l121_121087


namespace portion_to_joe_and_darcy_eq_half_l121_121047

open Int

noncomputable def portion_given_to_joe_and_darcy : ℚ := 
let total_slices := 8
let portion_to_carl := 1 / 4
let slices_to_carl := portion_to_carl * total_slices
let slices_left := 2
let slices_given_to_joe_and_darcy := total_slices - slices_to_carl - slices_left
let portion_to_joe_and_darcy := slices_given_to_joe_and_darcy / total_slices
portion_to_joe_and_darcy

theorem portion_to_joe_and_darcy_eq_half :
  portion_given_to_joe_and_darcy = 1 / 2 :=
sorry

end portion_to_joe_and_darcy_eq_half_l121_121047


namespace crushing_load_l121_121736

theorem crushing_load (T H C : ℝ) (L : ℝ) 
  (h1 : T = 5) (h2 : H = 10) (h3 : C = 3)
  (h4 : L = C * 25 * T^4 / H^2) : 
  L = 468.75 :=
by
  sorry

end crushing_load_l121_121736


namespace chosen_number_l121_121462

theorem chosen_number (x: ℤ) (h: 2 * x - 152 = 102) : x = 127 :=
by
  sorry

end chosen_number_l121_121462


namespace sides_of_polygon_l121_121424

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l121_121424


namespace spaceship_not_moving_time_l121_121137

-- Definitions based on the conditions given
def total_journey_time : ℕ := 3 * 24  -- 3 days in hours

def first_travel_time : ℕ := 10
def first_break_time : ℕ := 3
def second_travel_time : ℕ := 10
def second_break_time : ℕ := 1

def subsequent_travel_period : ℕ := 11  -- 11 hours traveling, then 1 hour break

-- Function to compute total break time
def total_break_time (total_travel_time : ℕ) : ℕ :=
  let remaining_time := total_journey_time - (first_travel_time + first_break_time + second_travel_time + second_break_time)
  let subsequent_breaks := remaining_time / subsequent_travel_period
  first_break_time + second_break_time + subsequent_breaks

theorem spaceship_not_moving_time : total_break_time total_journey_time = 8 := by
  sorry

end spaceship_not_moving_time_l121_121137


namespace find_a_l121_121654

theorem find_a (a x : ℝ) (h1 : 2 * (x - 1) - 6 = 0) (h2 : 1 - (3 * a - x) / 3 = 0) (h3 : x = 4) : a = -1 / 3 :=
by
  sorry

end find_a_l121_121654


namespace correct_answer_l121_121320

def total_contestants : Nat := 56
def selected_contestants : Nat := 14

theorem correct_answer :
  (total_contestants = 56) →
  (selected_contestants = 14) →
  (selected_contestants = 14) :=
by
  intro h_total h_selected
  exact h_selected

end correct_answer_l121_121320


namespace two_class_students_l121_121569

-- Define the types of students and total sum variables
variables (H M E HM HE ME HME : ℕ)
variable (Total_Students : ℕ)

-- Given conditions
axiom condition1 : Total_Students = 68
axiom condition2 : H = 19
axiom condition3 : M = 14
axiom condition4 : E = 26
axiom condition5 : HME = 3

-- Inclusion-Exclusion principle formula application
def exactly_two_classes : Prop := 
  Total_Students = H + M + E - (HM + HE + ME) + HME

-- Theorem to prove the number of students registered for exactly two classes is 6
theorem two_class_students : H + M + E - 2 * HME + HME - (HM + HE + ME) = 6 := by
  sorry

end two_class_students_l121_121569


namespace paco_initial_cookies_l121_121341

theorem paco_initial_cookies (x : ℕ) (h : x - 2 + 36 = 2 + 34) : x = 2 :=
by
-- proof steps will be filled in here
sorry

end paco_initial_cookies_l121_121341


namespace calculation_of_expression_l121_121409

theorem calculation_of_expression :
  (1.99 ^ 2 - 1.98 * 1.99 + 0.99 ^ 2) = 1 := 
by sorry

end calculation_of_expression_l121_121409


namespace arithmetic_sequence_sum_l121_121535

theorem arithmetic_sequence_sum 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 = 12) : 
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28) :=
sorry

end arithmetic_sequence_sum_l121_121535


namespace percent_not_filler_l121_121415

theorem percent_not_filler (total_weight filler_weight : ℕ) (h1 : total_weight = 180) (h2 : filler_weight = 45) : 
  ((total_weight - filler_weight) * 100 / total_weight = 75) :=
by 
  sorry

end percent_not_filler_l121_121415


namespace green_sweets_count_l121_121851

def total_sweets := 285
def red_sweets := 49
def neither_red_nor_green_sweets := 177

theorem green_sweets_count : 
  (total_sweets - red_sweets - neither_red_nor_green_sweets) = 59 :=
by
  -- The proof will go here
  sorry

end green_sweets_count_l121_121851


namespace parallelogram_sides_l121_121870

theorem parallelogram_sides (x y : ℝ) (h₁ : 4 * x + 1 = 11) (h₂ : 10 * y - 3 = 5) : x + y = 3.3 :=
sorry

end parallelogram_sides_l121_121870


namespace not_divisor_60_l121_121741

variable (k : ℤ)
def n : ℤ := k * (k + 1) * (k + 2)

theorem not_divisor_60 
  (h₁ : ∃ k, n = k * (k + 1) * (k + 2) ∧ 5 ∣ n) : ¬(60 ∣ n) := 
sorry

end not_divisor_60_l121_121741


namespace tins_of_beans_left_l121_121065

theorem tins_of_beans_left (cases : ℕ) (tins_per_case : ℕ) (damage_percentage : ℝ) (h_cases : cases = 15)
  (h_tins_per_case : tins_per_case = 24) (h_damage_percentage : damage_percentage = 0.05) :
  let total_tins := cases * tins_per_case
  let damaged_tins := total_tins * damage_percentage
  let tins_left := total_tins - damaged_tins
  tins_left = 342 :=
by
  sorry

end tins_of_beans_left_l121_121065


namespace arithmetic_sequence_a2a3_l121_121300

noncomputable def arithmetic_sequence_sum (a : Nat → ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a (n + 1) = a n + d

theorem arithmetic_sequence_a2a3 
  (a : Nat → ℝ) (d : ℝ) 
  (arith_seq : arithmetic_sequence_sum a d)
  (H : a 1 + a 2 + a 3 + a 4 = 30) : 
  a 2 + a 3 = 15 :=
by 
sorry

end arithmetic_sequence_a2a3_l121_121300


namespace fish_total_count_l121_121298

theorem fish_total_count :
  let num_fishermen : ℕ := 20
  let fish_caught_per_fisherman : ℕ := 400
  let fish_caught_by_twentieth_fisherman : ℕ := 2400
  (19 * fish_caught_per_fisherman + fish_caught_by_twentieth_fisherman) = 10000 :=
by
  sorry

end fish_total_count_l121_121298


namespace seq_periodic_l121_121392

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1/4
  else ite (n > 1) (1 - (1 / (seq (n-1)))) 0 -- handle invalid cases with a default zero

theorem seq_periodic {n : ℕ} (h : seq 1 = 1/4) (h2 : ∀ k ≥ 2, seq k = 1 - (1 / (seq (k-1)))) :
  seq 2014 = 1/4 :=
sorry

end seq_periodic_l121_121392


namespace west_for_200_is_neg_200_l121_121119

-- Given a definition for driving east
def driving_east (d : Int) : Int := d

-- Driving east for 80 km is +80 km
def driving_east_80 : Int := driving_east 80

-- Driving west should be the negative of driving east
def driving_west (d : Int) : Int := -d

-- Driving west for 200 km is -200 km
def driving_west_200 : Int := driving_west 200

-- Theorem to prove the given condition and expected result
theorem west_for_200_is_neg_200 : driving_west_200 = -200 :=
by
  -- Proof step is skipped
  sorry

end west_for_200_is_neg_200_l121_121119


namespace range_of_a_l121_121757

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ -1 → f a x ≥ a) : -3 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l121_121757


namespace average_trees_planted_l121_121321

def A := 225
def B := A + 48
def C := A - 24
def total_trees := A + B + C
def average := total_trees / 3

theorem average_trees_planted :
  average = 233 := by
  sorry

end average_trees_planted_l121_121321


namespace l_shape_area_l121_121494

theorem l_shape_area (P : ℝ) (L : ℝ) (x : ℝ)
  (hP : P = 52) 
  (hL : L = 16) 
  (h_x : L + (L - x) + 2 * (16 - x) = P)
  (h_split : 2 * (16 - x) * x = 120) :
  2 * ((16 - x) * x) = 120 :=
by
  -- This is the proof problem statement
  sorry

end l_shape_area_l121_121494


namespace find_g4_l121_121363

variables (g : ℝ → ℝ)

-- Given conditions
axiom condition1 : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1
axiom condition2 : g 4 + 3 * g (-2) = 35
axiom condition3 : g (-2) + 3 * g 4 = 5

theorem find_g4 : g 4 = -5 / 2 :=
by
  sorry

end find_g4_l121_121363


namespace compute_zeta_seventh_power_sum_l121_121929

noncomputable def complex_seventh_power_sum : Prop :=
  ∀ (ζ₁ ζ₂ ζ₃ : ℂ), 
    (ζ₁ + ζ₂ + ζ₃ = 1) ∧ 
    (ζ₁^2 + ζ₂^2 + ζ₃^2 = 3) ∧
    (ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) →
    (ζ₁^7 + ζ₂^7 + ζ₃^7 = 71)

theorem compute_zeta_seventh_power_sum : complex_seventh_power_sum :=
by
  sorry

end compute_zeta_seventh_power_sum_l121_121929


namespace find_product_x_plus_1_x_minus_1_l121_121619

theorem find_product_x_plus_1_x_minus_1 (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x = 128) : (x + 1) * (x - 1) = 24 := sorry

end find_product_x_plus_1_x_minus_1_l121_121619


namespace intersection_eq_l121_121539

def set_M : Set ℝ := { x : ℝ | (x + 3) * (x - 2) < 0 }
def set_N : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq : set_M ∩ set_N = { x : ℝ | 1 ≤ x ∧ x < 2 } := by
  sorry

end intersection_eq_l121_121539


namespace max_cookies_without_ingredients_l121_121129

-- Defining the number of cookies and their composition
def total_cookies : ℕ := 36
def peanuts : ℕ := (2 * total_cookies) / 3
def chocolate_chips : ℕ := total_cookies / 3
def raisins : ℕ := total_cookies / 4
def oats : ℕ := total_cookies / 8

-- Proving the largest number of cookies without any ingredients
theorem max_cookies_without_ingredients : (total_cookies - (max (max peanuts chocolate_chips) raisins)) = 12 := by
    sorry

end max_cookies_without_ingredients_l121_121129


namespace initial_average_mark_l121_121884

theorem initial_average_mark (A : ℝ) (n_total n_excluded remaining_students_avg : ℝ) 
  (h1 : n_total = 25) 
  (h2 : n_excluded = 5) 
  (h3 : remaining_students_avg = 90)
  (excluded_students_avg : ℝ)
  (h_excluded_avg : excluded_students_avg = 40)
  (A_def : (n_total * A) = (n_excluded * excluded_students_avg + (n_total - n_excluded) * remaining_students_avg)) :
  A = 80 := 
by
  sorry

end initial_average_mark_l121_121884


namespace money_left_after_purchase_l121_121911

noncomputable def initial_money : ℝ := 200
noncomputable def candy_bars : ℝ := 25
noncomputable def bags_of_chips : ℝ := 10
noncomputable def soft_drinks : ℝ := 15

noncomputable def cost_per_candy_bar : ℝ := 3
noncomputable def cost_per_bag_of_chips : ℝ := 2.5
noncomputable def cost_per_soft_drink : ℝ := 1.75

noncomputable def discount_candy_bars : ℝ := 0.10
noncomputable def discount_bags_of_chips : ℝ := 0.05
noncomputable def sales_tax : ℝ := 0.06

theorem money_left_after_purchase : initial_money - 
  ( ((candy_bars * cost_per_candy_bar * (1 - discount_candy_bars)) + 
    (bags_of_chips * cost_per_bag_of_chips * (1 - discount_bags_of_chips)) + 
    (soft_drinks * cost_per_soft_drink)) * 
    (1 + sales_tax)) = 75.45 := by
  sorry

end money_left_after_purchase_l121_121911


namespace a11_a12_a13_eq_105_l121_121984

variable (a : ℕ → ℝ) -- Define the arithmetic sequence
variable (d : ℝ) -- Define the common difference

-- Assume the conditions given in step a)
axiom arith_seq (n : ℕ) : a n = a 0 + n * d
axiom sum_3_eq_15 : a 0 + a 1 + a 2 = 15
axiom prod_3_eq_80 : a 0 * a 1 * a 2 = 80
axiom pos_diff : d > 0

theorem a11_a12_a13_eq_105 : a 10 + a 11 + a 12 = 105 :=
sorry

end a11_a12_a13_eq_105_l121_121984


namespace kids_french_fries_cost_l121_121824

noncomputable def cost_burger : ℝ := 5
noncomputable def cost_fries : ℝ := 3
noncomputable def cost_soft_drink : ℝ := 3
noncomputable def cost_special_burger_meal : ℝ := 9.50
noncomputable def cost_kids_burger : ℝ := 3
noncomputable def cost_kids_juice_box : ℝ := 2
noncomputable def cost_kids_meal : ℝ := 5
noncomputable def savings : ℝ := 10

noncomputable def total_adult_meal_individual : ℝ := 2 * cost_burger + 2 * cost_fries + 2 * cost_soft_drink
noncomputable def total_adult_meal_deal : ℝ := 2 * cost_special_burger_meal

noncomputable def total_kids_meal_individual (F : ℝ) : ℝ := 2 * cost_kids_burger + 2 * F + 2 * cost_kids_juice_box
noncomputable def total_kids_meal_deal : ℝ := 2 * cost_kids_meal

noncomputable def total_cost_individual (F : ℝ) : ℝ := total_adult_meal_individual + total_kids_meal_individual F
noncomputable def total_cost_deal : ℝ := total_adult_meal_deal + total_kids_meal_deal

theorem kids_french_fries_cost : ∃ F : ℝ, total_cost_individual F - total_cost_deal = savings ∧ F = 3.50 := 
by
  use 3.50
  sorry

end kids_french_fries_cost_l121_121824


namespace average_weight_whole_class_l121_121524

def sectionA_students : Nat := 36
def sectionB_students : Nat := 44
def avg_weight_sectionA : Float := 40.0 
def avg_weight_sectionB : Float := 35.0
def total_weight_sectionA := avg_weight_sectionA * Float.ofNat sectionA_students
def total_weight_sectionB := avg_weight_sectionB * Float.ofNat sectionB_students
def total_students := sectionA_students + sectionB_students
def total_weight := total_weight_sectionA + total_weight_sectionB
def avg_weight_class := total_weight / Float.ofNat total_students

theorem average_weight_whole_class :
  avg_weight_class = 37.25 := by
  sorry

end average_weight_whole_class_l121_121524


namespace rectangle_area_l121_121111

theorem rectangle_area (x : ℝ) (w : ℝ) (h1 : (3 * w)^2 + w^2 = x^2) : (3 * w) * w = 3 * x^2 / 10 :=
by
  sorry

end rectangle_area_l121_121111


namespace max_value_expr_l121_121893

def point_on_line (m n : ℝ) : Prop :=
  3 * m + n = -1

def mn_positive (m n : ℝ) : Prop :=
  m * n > 0

theorem max_value_expr (m n : ℝ) (h1 : point_on_line m n) (h2 : mn_positive m n) :
  (3 / m + 1 / n) = -16 :=
sorry

end max_value_expr_l121_121893


namespace quilt_shaded_fraction_l121_121095

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_full_square := 4
  let shaded_half_triangles_as_square := 2
  let total_area := total_squares
  let shaded_area := shaded_full_square + shaded_half_triangles_as_square
  shaded_area / total_area = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l121_121095


namespace sum_of_palindromes_l121_121022

-- Define a three-digit palindrome predicate
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ b < 10 ∧ n = 100*a + 10*b + a

-- Define the product of the two palindromes equaling 436,995
theorem sum_of_palindromes (a b : ℕ) (h_a : is_palindrome a) (h_b : is_palindrome b) (h_prod : a * b = 436995) : 
  a + b = 1332 :=
sorry

end sum_of_palindromes_l121_121022


namespace cyclic_sum_inequality_l121_121967

variable (a b c : ℝ)
variable (pos_a : a > 0)
variable (pos_b : b > 0)
variable (pos_c : c > 0)

theorem cyclic_sum_inequality :
  ( (a^3 + b^3) / (a^2 + a * b + b^2) + 
    (b^3 + c^3) / (b^2 + b * c + c^2) + 
    (c^3 + a^3) / (c^2 + c * a + a^2) ) ≥ 
  (2 / 3) * (a + b + c) := 
  sorry

end cyclic_sum_inequality_l121_121967


namespace find_a_l121_121606

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1 ∧ x ≥ 2

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, point_on_hyperbola x y ∧ (min ((x - a)^2 + y^2) = 3)) → 
  (a = -1 ∨ a = 2 * Real.sqrt 5) :=
by
  sorry

end find_a_l121_121606


namespace lisa_eggs_l121_121332

theorem lisa_eggs :
  ∃ x : ℕ, (5 * 52) * (4 * x + 3 + 2) = 3380 ∧ x = 2 :=
by
  sorry

end lisa_eggs_l121_121332


namespace expression_divisible_by_264_l121_121969

theorem expression_divisible_by_264 (n : ℕ) (h : n > 1) : ∃ k : ℤ, 7^(2*n) - 4^(2*n) - 297 = 264 * k :=
by 
  sorry

end expression_divisible_by_264_l121_121969


namespace simplify_rationalize_expr_l121_121793

theorem simplify_rationalize_expr : 
  (1 / (2 + 1 / (Real.sqrt 5 - 2))) = (4 - Real.sqrt 5) / 11 := 
by 
  sorry

end simplify_rationalize_expr_l121_121793


namespace solution_l121_121816

-- Conditions
def x : ℚ := 3/5
def y : ℚ := 5/3

-- Proof problem
theorem solution : (1/3) * x^8 * y^9 = 5/9 := sorry

end solution_l121_121816


namespace pencils_remaining_in_drawer_l121_121498

-- Definitions of the conditions
def total_pencils_initially : ℕ := 34
def pencils_taken : ℕ := 22

-- The theorem statement with the correct answer
theorem pencils_remaining_in_drawer : total_pencils_initially - pencils_taken = 12 :=
by
  sorry

end pencils_remaining_in_drawer_l121_121498


namespace determine_OP_l121_121270

variables (a b c d q : ℝ)
variables (P : ℝ)
variables (h_ratio : (|a - P| / |P - d| = |b - P| / |P - c|))
variables (h_twice : P = 2 * q)

theorem determine_OP : P = 2 * q :=
sorry

end determine_OP_l121_121270


namespace vector_parallel_cos_sin_l121_121327

theorem vector_parallel_cos_sin (θ : ℝ) (a b : ℝ × ℝ) (ha : a = (Real.cos θ, Real.sin θ)) (hb : b = (1, -2)) :
  ∀ (h : ∃ k : ℝ, a = (k * 1, k * (-2))), 
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 3 := 
by
  sorry

end vector_parallel_cos_sin_l121_121327


namespace triangles_with_two_colors_l121_121439

theorem triangles_with_two_colors {n : ℕ} 
  (h1 : ∀ (p : Finset ℝ) (hn : p.card = n) 
      (e : p → p → Prop), 
      (∀ (x y : p), e x y → e x y = red ∨ e x y = yellow ∨ e x y = green) /\
      (∀ (a b c : p), 
        (e a b = red ∨ e a b = yellow ∨ e a b = green) ∧ 
        (e b c = red ∨ e b c = yellow ∨ e b c = green) ∧ 
        (e a c = red ∨ e a c = yellow ∨ e a c = green) → 
        (e a b ≠ e b c ∨ e b c ≠ e a c ∨ e a b ≠ e a c))) :
  n < 13 := 
sorry

end triangles_with_two_colors_l121_121439


namespace angles_sum_l121_121324

def points_on_circle (A B C R S O : Type) : Prop := sorry

def arc_measure (B R S : Type) (m1 m2 : ℝ) : Prop := sorry

def angle_T (A C B S : Type) (T : ℝ) : Prop := sorry

def angle_U (O C B S : Type) (U : ℝ) : Prop := sorry

theorem angles_sum
  (A B C R S O : Type)
  (h1 : points_on_circle A B C R S O)
  (h2 : arc_measure B R S 48 54)
  (h3 : angle_T A C B S 78)
  (h4 : angle_U O C B S 27) :
  78 + 27 = 105 :=
by sorry

end angles_sum_l121_121324


namespace find_n_in_permutation_combination_equation_l121_121113

-- Lean statement for the proof problem

theorem find_n_in_permutation_combination_equation :
  ∃ (n : ℕ), (n > 0) ∧ (Nat.factorial 8 / Nat.factorial (8 - n) = 2 * (Nat.factorial 8 / (Nat.factorial 2 * Nat.factorial 6)))
  := sorry

end find_n_in_permutation_combination_equation_l121_121113


namespace candy_cost_proof_l121_121617

theorem candy_cost_proof (x : ℝ) (h1 : 10 ≤ 30) (h2 : 0 ≤ 5) (h3 : 0 ≤ 6) 
(h4 : 10 * x + 20 * 5 = 6 * 30) : x = 8 := by
  sorry

end candy_cost_proof_l121_121617


namespace min_packs_for_soda_l121_121271

theorem min_packs_for_soda (max_packs : ℕ) (packs : List ℕ) : 
  let num_cans := 95
  let max_each_pack := 4
  let pack_8 := packs.count 8 
  let pack_15 := packs.count 15
  let pack_18 := packs.count 18
  pack_8 ≤ max_each_pack ∧ pack_15 ≤ max_each_pack ∧ pack_18 ≤ max_each_pack ∧ 
  pack_8 * 8 + pack_15 * 15 + pack_18 * 18 = num_cans ∧ 
  pack_8 + pack_15 + pack_18 = max_packs → max_packs = 6 :=
sorry

end min_packs_for_soda_l121_121271


namespace market_value_of_stock_l121_121720

def face_value : ℝ := 100
def dividend_percentage : ℝ := 0.13
def yield : ℝ := 0.08

theorem market_value_of_stock : 
  (dividend_percentage * face_value / yield) * 100 = 162.50 :=
by
  sorry

end market_value_of_stock_l121_121720


namespace gcd_problem_l121_121517

theorem gcd_problem 
  (b : ℤ) 
  (hb_odd : b % 2 = 1) 
  (hb_multiples_of_8723 : ∃ (k : ℤ), b = 8723 * k) : 
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 15) = 3 := 
by 
  sorry

end gcd_problem_l121_121517


namespace first_term_of_geometric_series_l121_121251

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l121_121251


namespace verify_statements_l121_121003

def line1 (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def line2 (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

theorem verify_statements (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ x = -1 ∧ y = -1) ∧
  (∀ x y : ℝ, (line1 a x y ∧ line2 a x y) → (a = 0 ∨ a = -4)) :=
by sorry

end verify_statements_l121_121003


namespace solve_equation_l121_121410

theorem solve_equation (x : ℝ) (h : x ≠ 1) : -x^2 = (2 * x + 4) / (x - 1) → (x = -2 ∨ x = 1) :=
by
  sorry

end solve_equation_l121_121410


namespace total_marbles_l121_121460

theorem total_marbles (r b g : ℕ) (total : ℕ) 
  (h_ratio : 2 * g = 4 * b) 
  (h_blue_marbles : b = 36) 
  (h_total_formula : total = r + b + g) 
  : total = 108 :=
by
  sorry

end total_marbles_l121_121460


namespace gcd_98_63_l121_121525

-- Definition of gcd
def gcd_euclidean := ∀ (a b : ℕ), ∃ (g : ℕ), gcd a b = g

-- Statement of the problem using Lean
theorem gcd_98_63 : gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l121_121525


namespace cost_of_each_ring_l121_121089

theorem cost_of_each_ring (R : ℝ) 
  (h1 : 4 * 12 + 8 * R = 80) : R = 4 :=
by 
  sorry

end cost_of_each_ring_l121_121089


namespace arithmetic_square_root_problem_l121_121177

open Real

theorem arithmetic_square_root_problem 
  (a b c : ℝ)
  (ha : 5 * a - 2 = -27)
  (hb : b = ⌊sqrt 22⌋)
  (hc : c = -sqrt (4 / 25)) :
  sqrt (4 * a * c + 7 * b) = 6 := by
  sorry

end arithmetic_square_root_problem_l121_121177


namespace probability_x_gt_9y_in_rectangle_l121_121297

theorem probability_x_gt_9y_in_rectangle :
  let a := 1007
  let b := 1008
  let area_triangle := (a * a / 18 : ℚ)
  let area_rectangle := (a * b : ℚ)
  area_triangle / area_rectangle = (1 : ℚ) / 18 :=
by
  sorry

end probability_x_gt_9y_in_rectangle_l121_121297


namespace alberto_spent_more_l121_121075

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end alberto_spent_more_l121_121075


namespace percentage_difference_liliane_alice_l121_121127

theorem percentage_difference_liliane_alice :
  let J := 200
  let L := 1.30 * J
  let A := 1.15 * J
  (L - A) / A * 100 = 13.04 :=
by
  sorry

end percentage_difference_liliane_alice_l121_121127


namespace unique_n_value_l121_121362

theorem unique_n_value (n : ℕ) (d : ℕ → ℕ) (h1 : 1 = d 1) (h2 : ∀ i, d i ≤ n) (h3 : ∀ i j, i < j → d i < d j) 
                       (h4 : d (n - 1) = n) (h5 : ∃ k, k ≥ 4 ∧ ∀ i ≤ k, d i ∣ n)
                       (h6 : ∃ d1 d2 d3 d4, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ n = d1^2 + d2^2 + d3^2 + d4^2) : 
                       n = 130 := sorry

end unique_n_value_l121_121362


namespace buffet_dishes_l121_121379

-- To facilitate the whole proof context, but skipping proof parts with 'sorry'

-- Oliver will eat if there is no mango in the dishes

variables (D : ℕ) -- Total number of dishes

-- Conditions:
variables (h1 : 3 <= D) -- there are at least 3 dishes with mango salsa
variables (h2 : 1 ≤ D / 6) -- one-sixth of dishes have fresh mango
variables (h3 : 1 ≤ D) -- there's at least one dish with mango jelly
variables (h4 : D / 6 ≥ 2) -- Oliver can pick out the mangoes from 2 of dishes with fresh mango
variables (h5 : D - (3 + (D / 6 - 2) + 1) = 28) -- there are 28 dishes Oliver can eat

theorem buffet_dishes : D = 36 :=
by
  sorry -- Skip the actual proof

end buffet_dishes_l121_121379


namespace domain_f_l121_121777

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.sqrt (x + 5)) + Real.log (2^x + 1)

theorem domain_f :
  {x : ℝ | (-5 ≤ x)} = {x : ℝ | f x ∈ Set.univ} := sorry

end domain_f_l121_121777


namespace jo_age_l121_121873

theorem jo_age (j d g : ℕ) (even_j : 2 * j = j * 2) (even_d : 2 * d = d * 2) (even_g : 2 * g = g * 2)
    (h : 8 * j * d * g = 2024) : 2 * j = 46 :=
sorry

end jo_age_l121_121873


namespace larger_number_is_84_l121_121971

theorem larger_number_is_84 (x y : ℕ) (HCF LCM : ℕ)
  (h_hcf : HCF = 84)
  (h_lcm : LCM = 21)
  (h_ratio : x * 4 = y)
  (h_product : x * y = HCF * LCM) :
  y = 84 :=
by
  sorry

end larger_number_is_84_l121_121971


namespace speed_with_stream_l121_121961

noncomputable def man_speed_still_water : ℝ := 5
noncomputable def speed_against_stream : ℝ := 4

theorem speed_with_stream :
  ∃ V_s, man_speed_still_water + V_s = 6 :=
by
  use man_speed_still_water - speed_against_stream
  sorry

end speed_with_stream_l121_121961


namespace cricketer_average_after_19_innings_l121_121350

theorem cricketer_average_after_19_innings
  (A : ℝ) 
  (total_runs_after_18 : ℝ := 18 * A) 
  (runs_in_19th : ℝ := 99) 
  (new_avg : ℝ := A + 4) 
  (total_runs_after_19 : ℝ := total_runs_after_18 + runs_in_19th) 
  (equation : 19 * new_avg = total_runs_after_19) : 
  new_avg = 27 :=
by
  sorry

end cricketer_average_after_19_innings_l121_121350


namespace cannot_be_2009_l121_121544

theorem cannot_be_2009 (a b c : ℕ) (h : b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) : (b * 1^2 + c * 1 + a ≠ 2009) :=
by
  sorry

end cannot_be_2009_l121_121544


namespace two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l121_121247

open Nat

theorem two_pow_m_minus_one_not_divide_three_pow_n_minus_one 
  (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hmo : Odd m) (hno : Odd n) : ¬ (∃ k : ℕ, 2^m - 1 = k * (3^n - 1)) := by
  sorry

end two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l121_121247


namespace prop_neg_or_not_l121_121508

theorem prop_neg_or_not (p q : Prop) (h : ¬(p ∨ ¬ q)) : ¬ p ∧ q :=
by
  sorry

end prop_neg_or_not_l121_121508


namespace inequality_solution_l121_121206

noncomputable def f (a b x : ℝ) : ℝ := 1 / Real.sqrt x + 1 / Real.sqrt (a + b - x)

theorem inequality_solution 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (x : ℝ) 
  (hx : x ∈ Set.Ioo (min a b) (max a b)) : 
  f a b x < f a b a ∧ f a b x < f a b b := 
sorry

end inequality_solution_l121_121206


namespace kristin_runs_n_times_faster_l121_121555

theorem kristin_runs_n_times_faster (D K S : ℝ) (n : ℝ) 
  (h1 : K = n * S) 
  (h2 : 12 * D / K = 4 * D / S) : 
  n = 3 :=
by
  sorry

end kristin_runs_n_times_faster_l121_121555


namespace count_true_propositions_l121_121081

theorem count_true_propositions :
  let prop1 := false  -- Proposition ① is false
  let prop2 := true   -- Proposition ② is true
  let prop3 := true   -- Proposition ③ is true
  let prop4 := false  -- Proposition ④ is false
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) +
  (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 :=
by
  -- The theorem is expected to be proven here
  sorry

end count_true_propositions_l121_121081


namespace square_AP_square_equals_2000_l121_121357

noncomputable def square_side : ℝ := 100
noncomputable def midpoint_AB : ℝ := square_side / 2
noncomputable def distance_MP : ℝ := 50
noncomputable def distance_PC : ℝ := square_side

/-- Given a square ABCD with side length 100, midpoint M of AB, MP = 50, and PC = 100, prove AP^2 = 2000 -/
theorem square_AP_square_equals_2000 :
  ∃ (P : ℝ × ℝ), (dist (P.1, P.2) (midpoint_AB, 0) = distance_MP) ∧ (dist (P.1, P.2) (square_side, square_side) = distance_PC) ∧ ((P.1) ^ 2 + (P.2) ^ 2 = 2000) := 
sorry


end square_AP_square_equals_2000_l121_121357


namespace shaded_region_perimeter_l121_121073

theorem shaded_region_perimeter (r : ℝ) (h : r = 12 / Real.pi) :
  3 * (24 / 6) = 12 := 
by
  sorry

end shaded_region_perimeter_l121_121073


namespace emails_difference_l121_121472

theorem emails_difference
  (emails_morning : ℕ)
  (emails_afternoon : ℕ)
  (h_morning : emails_morning = 10)
  (h_afternoon : emails_afternoon = 3)
  : emails_morning - emails_afternoon = 7 := by
  sorry

end emails_difference_l121_121472


namespace exist_non_special_symmetric_concat_l121_121482

-- Define the notion of a binary series being symmetric
def is_symmetric (xs : List Bool) : Prop :=
  ∀ i, i < xs.length → xs.get? i = xs.get? (xs.length - 1 - i)

-- Define the notion of a binary series being special
def is_special (xs : List Bool) : Prop :=
  (∀ x ∈ xs, x) ∨ (∀ x ∈ xs, ¬x)

-- The main theorem statement
theorem exist_non_special_symmetric_concat (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (A B : List Bool), A.length = m ∧ B.length = n ∧ ¬is_special A ∧ ¬is_special B ∧ is_symmetric (A ++ B) :=
sorry

end exist_non_special_symmetric_concat_l121_121482


namespace find_k_l121_121918

theorem find_k (k : ℝ) : 
  (1 / 2) * |k| * |k / 2| = 4 → (k = 4 ∨ k = -4) := 
sorry

end find_k_l121_121918


namespace solve_ab_eq_l121_121968

theorem solve_ab_eq:
  ∃ a b : ℝ, (1 + (2 : ℂ) * (Complex.I)) * (a : ℂ) + (b : ℂ) = (2 : ℂ) * (Complex.I) ∧ a = 1 ∧ b = -1 := by
  sorry

end solve_ab_eq_l121_121968


namespace final_price_is_correct_l121_121687

-- Define the original price
def original_price : ℝ := 10

-- Define the first reduction percentage
def first_reduction_percentage : ℝ := 0.30

-- Define the second reduction percentage
def second_reduction_percentage : ℝ := 0.50

-- Define the price after the first reduction
def price_after_first_reduction : ℝ := original_price * (1 - first_reduction_percentage)

-- Define the final price after the second reduction
def final_price : ℝ := price_after_first_reduction * (1 - second_reduction_percentage)

-- Theorem to prove the final price is $3.50
theorem final_price_is_correct : final_price = 3.50 := by
  sorry

end final_price_is_correct_l121_121687


namespace minneapolis_st_louis_temperature_l121_121558

theorem minneapolis_st_louis_temperature (N M L : ℝ) (h1 : M = L + N)
                                         (h2 : M - 7 = L + N - 7)
                                         (h3 : L + 5 = L + 5)
                                         (h4 : (M - 7) - (L + 5) = |(L + N - 7) - (L + 5)|) :
  ∃ (N1 N2 : ℝ), (|N - 12| = 4) ∧ N1 = 16 ∧ N2 = 8 ∧ N1 * N2 = 128 :=
by {
  sorry
}

end minneapolis_st_louis_temperature_l121_121558


namespace tan_435_eq_2_add_sqrt_3_l121_121783

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end tan_435_eq_2_add_sqrt_3_l121_121783


namespace corrected_mean_is_45_55_l121_121577

-- Define the initial conditions
def mean_of_100_observations (mean : ℝ) : Prop :=
  mean = 45

def incorrect_observation : ℝ := 32
def correct_observation : ℝ := 87

-- Define the calculation of the corrected mean
noncomputable def corrected_mean (incorrect_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (n : ℕ) : ℝ :=
  let sum_original := incorrect_mean * n
  let difference := correct_obs - incorrect_obs
  (sum_original + difference) / n

-- Theorem: The corrected new mean is 45.55
theorem corrected_mean_is_45_55 : corrected_mean 45 32 87 100 = 45.55 :=
by
  sorry

end corrected_mean_is_45_55_l121_121577


namespace complement_of_A_in_U_l121_121833

-- Define the universal set U
def U : Set ℕ := {2, 3, 4}

-- Define set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Prove the complement of A in U is {4}
theorem complement_of_A_in_U : C_U_A = {4} := 
  by 
  sorry

end complement_of_A_in_U_l121_121833


namespace polygon_sides_l121_121782

theorem polygon_sides (h : ∀ (θ : ℕ), θ = 108) : ∃ n : ℕ, n = 5 :=
by
  sorry

end polygon_sides_l121_121782


namespace ab_cd_divisible_eq_one_l121_121665

theorem ab_cd_divisible_eq_one (a b c d : ℕ) (h1 : ∃ e : ℕ, e = ab - cd ∧ (e ∣ a) ∧ (e ∣ b) ∧ (e ∣ c) ∧ (e ∣ d)) : ab - cd = 1 :=
sorry

end ab_cd_divisible_eq_one_l121_121665


namespace total_pieces_of_art_l121_121910

variable (A : ℕ) (displayed : ℕ) (sculptures_on_display : ℕ) (not_on_display : ℕ) (paintings_not_on_display : ℕ) (sculptures_not_on_display : ℕ)

-- Constants and conditions from the problem
axiom H1 : displayed = 1 / 3 * A
axiom H2 : sculptures_on_display = 1 / 6 * displayed
axiom H3 : not_on_display = 2 / 3 * A
axiom H4 : paintings_not_on_display = 1 / 3 * not_on_display
axiom H5 : sculptures_not_on_display = 800
axiom H6 : sculptures_not_on_display = 2 / 3 * not_on_display

-- Prove that the total number of pieces of art is 1800
theorem total_pieces_of_art : A = 1800 :=
by
  sorry

end total_pieces_of_art_l121_121910


namespace remainder_of_11_pow_2023_mod_33_l121_121014

theorem remainder_of_11_pow_2023_mod_33 : (11 ^ 2023) % 33 = 11 := 
by
  sorry

end remainder_of_11_pow_2023_mod_33_l121_121014


namespace complex_z_pow_l121_121621

open Complex

theorem complex_z_pow {z : ℂ} (h : (1 + z) / (1 - z) = (⟨0, 1⟩ : ℂ)) : z ^ 2019 = -⟨0, 1⟩ := by
  sorry

end complex_z_pow_l121_121621


namespace exists_large_absolute_value_solutions_l121_121652

theorem exists_large_absolute_value_solutions : 
  ∃ (x1 x2 y1 y2 y3 y4 : ℤ), 
    x1 + x2 = y1 + y2 + y3 + y4 ∧ 
    x1^2 + x2^2 = y1^2 + y2^2 + y3^2 + y4^2 ∧ 
    x1^3 + x2^3 = y1^3 + y2^3 + y3^3 + y4^3 ∧ 
    abs x1 > 2020 ∧ abs x2 > 2020 ∧ abs y1 > 2020 ∧ abs y2 > 2020 ∧ abs y3 > 2020 ∧ abs y4 > 2020 :=
  by
  sorry

end exists_large_absolute_value_solutions_l121_121652


namespace additional_songs_added_l121_121737

theorem additional_songs_added (original_songs : ℕ) (song_duration : ℕ) (total_duration : ℕ) :
  original_songs = 25 → song_duration = 3 → total_duration = 105 → 
  (total_duration - original_songs * song_duration) / song_duration = 10 :=
by
  intros h1 h2 h3
  sorry

end additional_songs_added_l121_121737


namespace smallest_a_l121_121618

theorem smallest_a (x a : ℝ) (hx : x > 0) (ha : a > 0) (hineq : x + a / x ≥ 4) : a ≥ 4 :=
sorry

end smallest_a_l121_121618


namespace parabola_vertex_l121_121401

theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, t^2 + 2 * t - 2 ≥ y) ∧ (x^2 + 2 * x - 2 = y) ∧ (x = -1) ∧ (y = -3) :=
by sorry

end parabola_vertex_l121_121401


namespace max_M_correct_l121_121717

variable (A : ℝ) (x y : ℝ)

axiom A_pos : A > 0

noncomputable def max_M : ℝ :=
if A ≤ 4 then 2 + A / 2 else 2 * Real.sqrt A

theorem max_M_correct : 
  (∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y + A/(x + y) ≥ max_M A / Real.sqrt (x * y)) ∧ 
  (A ≤ 4 → max_M A = 2 + A / 2) ∧ 
  (A > 4 → max_M A = 2 * Real.sqrt A) :=
sorry

end max_M_correct_l121_121717


namespace phase_shift_cosine_l121_121286

theorem phase_shift_cosine (x : ℝ) : 2 * x + (Real.pi / 2) = 0 → x = - (Real.pi / 4) :=
by
  intro h
  sorry

end phase_shift_cosine_l121_121286


namespace exists_tetrahedra_volume_and_face_area_conditions_l121_121966

noncomputable def volume (T : Tetrahedron) : ℝ := sorry
noncomputable def face_area (T : Tetrahedron) : List ℝ := sorry

-- The existence of two tetrahedra such that the volume of T1 > T2 
-- and the area of each face of T1 does not exceed any face of T2.
theorem exists_tetrahedra_volume_and_face_area_conditions :
  ∃ (T1 T2 : Tetrahedron), 
    (volume T1 > volume T2) ∧ 
    (∀ (a1 : ℝ), a1 ∈ face_area T1 → 
      ∃ (a2 : ℝ), a2 ∈ face_area T2 ∧ a2 ≥ a1) :=
sorry

end exists_tetrahedra_volume_and_face_area_conditions_l121_121966


namespace quadratic_expression_value_l121_121395

theorem quadratic_expression_value (x1 x2 : ℝ)
    (h1: x1^2 + 5 * x1 + 1 = 0)
    (h2: x2^2 + 5 * x2 + 1 = 0) :
    ( (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 ) = 220 := 
sorry

end quadratic_expression_value_l121_121395


namespace complex_expression_is_none_of_the_above_l121_121760

-- We define the problem in Lean, stating that the given complex expression is not equal to any of the simplified forms
theorem complex_expression_is_none_of_the_above (x : ℝ) :
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x^3+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x-1)^4 ) :=
sorry

end complex_expression_is_none_of_the_above_l121_121760


namespace total_seeds_eaten_correct_l121_121540

-- Define the number of seeds each player ate
def seeds_first_player : ℕ := 78
def seeds_second_player : ℕ := 53
def seeds_third_player (seeds_second_player : ℕ) : ℕ := seeds_second_player + 30

-- Define the total seeds eaten
def total_seeds_eaten (seeds_first_player seeds_second_player seeds_third_player : ℕ) : ℕ :=
  seeds_first_player + seeds_second_player + seeds_third_player

-- Statement of the theorem
theorem total_seeds_eaten_correct : total_seeds_eaten seeds_first_player seeds_second_player (seeds_third_player seeds_second_player) = 214 :=
by
  sorry

end total_seeds_eaten_correct_l121_121540


namespace smallest_term_of_sequence_l121_121591

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

-- The statement that the 5th term is the smallest in the sequence
theorem smallest_term_of_sequence : ∀ n : ℕ, a 5 ≤ a n := by
  sorry

end smallest_term_of_sequence_l121_121591


namespace half_angle_in_second_and_fourth_quadrants_l121_121389

theorem half_angle_in_second_and_fourth_quadrants
  (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + 3 * π / 2) :
  (∃ m : ℤ, m * π + π / 2 < α / 2 ∧ α / 2 < m * π + 3 * π / 4) :=
by sorry

end half_angle_in_second_and_fourth_quadrants_l121_121389


namespace selling_price_l121_121598

theorem selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
    cost_price = 1600 → loss_percentage = 0.15 → 
    (cost_price - (loss_percentage * cost_price)) = 1360 :=
by
  intros h_cp h_lp
  rw [h_cp, h_lp]
  norm_num

end selling_price_l121_121598


namespace amber_total_cost_l121_121234

/-
Conditions:
1. Base cost of the plan: $25.
2. Cost for text messages with different rates for the first 120 messages and additional messages.
3. Cost for additional talk time.
4. Given specific usage data for Amber in January.

Objective:
Prove that the total monthly cost for Amber is $47.
-/
noncomputable def base_cost : ℕ := 25
noncomputable def text_message_cost (total_messages : ℕ) : ℕ :=
  if total_messages <= 120 then
    3 * total_messages
  else
    3 * 120 + 2 * (total_messages - 120)

noncomputable def talk_time_cost (talk_hours : ℕ) : ℕ :=
  if talk_hours <= 25 then
    0
  else
    15 * 60 * (talk_hours - 25)

noncomputable def total_monthly_cost (total_messages : ℕ) (talk_hours : ℕ) : ℕ :=
  base_cost + ((text_message_cost total_messages) / 100) + ((talk_time_cost talk_hours) / 100)

theorem amber_total_cost : total_monthly_cost 140 27 = 47 := by
  sorry

end amber_total_cost_l121_121234


namespace degree_of_p_x2_q_x4_l121_121608

-- Definitions to capture the given problem conditions
def is_degree_3 (p : Polynomial ℝ) : Prop := p.degree = 3
def is_degree_6 (q : Polynomial ℝ) : Prop := q.degree = 6

-- Statement of the proof problem
theorem degree_of_p_x2_q_x4 (p q : Polynomial ℝ) (hp : is_degree_3 p) (hq : is_degree_6 q) :
  (p.comp (Polynomial.X ^ 2) * q.comp (Polynomial.X ^ 4)).degree = 30 :=
sorry

end degree_of_p_x2_q_x4_l121_121608


namespace convert_octal_127_to_binary_l121_121688

def octal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 1 => 3  -- 001 in binary
  | 2 => 2  -- 010 in binary
  | 7 => 7  -- 111 in binary
  | _ => 0  -- No other digits are used in this example

theorem convert_octal_127_to_binary :
  octal_to_binary 1 * 1000000 + octal_to_binary 2 * 1000 + octal_to_binary 7 = 1010111 :=
by
  -- Proof would go here
  sorry

end convert_octal_127_to_binary_l121_121688


namespace positive_number_property_l121_121928

theorem positive_number_property (y : ℝ) (hy : 0 < y) : 
  (y^2 / 100) + 6 = 10 → y = 20 := by
  sorry

end positive_number_property_l121_121928


namespace more_ducks_than_four_times_chickens_l121_121364

def number_of_chickens (C : ℕ) : Prop :=
  185 = 150 + C

def number_of_ducks (C : ℕ) (MoreDucks : ℕ) : Prop :=
  150 = 4 * C + MoreDucks

theorem more_ducks_than_four_times_chickens (C MoreDucks : ℕ) (h1 : number_of_chickens C) (h2 : number_of_ducks C MoreDucks) : MoreDucks = 10 := by
  sorry

end more_ducks_than_four_times_chickens_l121_121364


namespace bus_profit_problem_l121_121143

def independent_variable := "number of passengers per month"
def dependent_variable := "monthly profit"

-- Given monthly profit equation
def monthly_profit (x : ℕ) : ℤ := 2 * x - 4000

-- 1. Independent and Dependent variables
def independent_variable_defined_correctly : Prop :=
  independent_variable = "number of passengers per month"

def dependent_variable_defined_correctly : Prop :=
  dependent_variable = "monthly profit"

-- 2. Minimum passenger volume to avoid losses
def minimum_passenger_volume_no_loss : Prop :=
  ∀ x : ℕ, (monthly_profit x >= 0) → (x >= 2000)

-- 3. Monthly profit prediction for 4230 passengers
def monthly_profit_prediction_4230 (x : ℕ) : Prop :=
  x = 4230 → monthly_profit x = 4460

theorem bus_profit_problem :
  independent_variable_defined_correctly ∧
  dependent_variable_defined_correctly ∧
  minimum_passenger_volume_no_loss ∧
  monthly_profit_prediction_4230 4230 :=
by
  sorry

end bus_profit_problem_l121_121143


namespace find_angle_A_range_of_bc_l121_121933

-- Define the necessary conditions and prove the size of angle A
theorem find_angle_A 
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : b * (Real.sin B + Real.sin C) = (a - c) * (Real.sin A + Real.sin C))
  (h₂ : B > Real.pi / 2)
  (h₃ : A + B + C = Real.pi)
  (h₄ : a > 0) (h₅ : b > 0) (h₆ : c > 0): 
  A = 2 * Real.pi / 3 :=
sorry

-- Define the necessary conditions and prove the range for b+c when a = sqrt(3)/2
theorem range_of_bc 
  (a b c : ℝ)
  (A : ℝ)
  (h₁ : A = 2 * Real.pi / 3)
  (h₂ : a = Real.sqrt 3 / 2)
  (h₃ : a > 0) (h₄ : b > 0) (h₅ : c > 0)
  (h₆ : A + B + C = Real.pi)
  (h₇ : B + C = Real.pi / 3) : 
  Real.sqrt 3 / 2 < b + c ∧ b + c ≤ 1 :=
sorry

end find_angle_A_range_of_bc_l121_121933


namespace number_of_students_earning_B_l121_121527

variables (a b c : ℕ) -- since we assume we only deal with whole numbers

-- Given conditions:
-- 1. The probability of earning an A is twice the probability of earning a B.
axiom h1 : a = 2 * b
-- 2. The probability of earning a C is equal to the probability of earning a B.
axiom h2 : c = b
-- 3. The only grades are A, B, or C and there are 45 students in the class.
axiom h3 : a + b + c = 45

-- Prove that the number of students earning a B is 11.
theorem number_of_students_earning_B : b = 11 :=
by
    sorry

end number_of_students_earning_B_l121_121527


namespace john_total_spent_l121_121502

/-- John's expenditure calculations -/
theorem john_total_spent :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 5
  let original_video_card_cost := 300
  let upgraded_video_card_cost := original_video_card_cost * 2
  let additional_upgrade_cost := upgraded_video_card_cost - original_video_card_cost
  let total_spent := computer_cost + peripherals_cost + additional_upgrade_cost
  total_spent = 2100 :=
by
  sorry

end john_total_spent_l121_121502


namespace necessary_condition_range_l121_121937

variables {x m : ℝ}

def p (x : ℝ) : Prop := x^2 - x - 2 < 0
def q (x m : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

theorem necessary_condition_range (H : ∀ x, q x m → p x) : -1 < m ∧ m < 1 :=
by {
  sorry
}

end necessary_condition_range_l121_121937


namespace solution_set_of_inequality_l121_121016

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3*x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l121_121016


namespace greatest_product_two_integers_sum_2004_l121_121998

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l121_121998


namespace johnson_and_martinez_tied_at_may_l121_121431

def home_runs_johnson (m : String) : ℕ :=
  if m = "January" then 2 else
  if m = "February" then 12 else
  if m = "March" then 20 else
  if m = "April" then 15 else
  if m = "May" then 9 else 0

def home_runs_martinez (m : String) : ℕ :=
  if m = "January" then 5 else
  if m = "February" then 9 else
  if m = "March" then 15 else
  if m = "April" then 20 else
  if m = "May" then 9 else 0

def cumulative_home_runs (player_home_runs : String → ℕ) (months : List String) : ℕ :=
  months.foldl (λ acc m => acc + player_home_runs m) 0

def months_up_to_may : List String :=
  ["January", "February", "March", "April", "May"]

theorem johnson_and_martinez_tied_at_may :
  cumulative_home_runs home_runs_johnson months_up_to_may
  = cumulative_home_runs home_runs_martinez months_up_to_may :=
by
    sorry

end johnson_and_martinez_tied_at_may_l121_121431


namespace chord_length_l121_121788

theorem chord_length (x y : ℝ) :
  (x^2 + y^2 - 2 * x - 4 * y = 0) →
  (x + 2 * y - 5 + Real.sqrt 5 = 0) →
  ∃ l, l = 4 :=
by
  intros h_circle h_line
  sorry

end chord_length_l121_121788


namespace keith_initial_cards_l121_121754

theorem keith_initial_cards (new_cards : ℕ) (cards_after_incident : ℕ) (total_cards : ℕ) :
  new_cards = 8 →
  cards_after_incident = 46 →
  total_cards = 2 * cards_after_incident →
  (total_cards - new_cards) = 84 :=
by
  intros
  sorry

end keith_initial_cards_l121_121754


namespace range_of_a_l121_121836

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 1) / (x - a) < 0}

theorem range_of_a (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : (1 / 3 : ℝ) ≤ a ∧ a < 1 / 2 ∨ 2 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l121_121836


namespace teacher_works_days_in_month_l121_121124

theorem teacher_works_days_in_month (P : ℕ) (W : ℕ) (M : ℕ) (T : ℕ) (H1 : P = 5) (H2 : W = 5) (H3 : M = 6) (H4 : T = 3600) : 
  (T / M) / (P * W) = 24 :=
by
  sorry

end teacher_works_days_in_month_l121_121124


namespace minimum_disks_needed_l121_121563

-- Define the conditions
def total_files : ℕ := 25
def disk_capacity : ℝ := 2.0
def files_06MB : ℕ := 5
def size_06MB_file : ℝ := 0.6
def files_10MB : ℕ := 10
def size_10MB_file : ℝ := 1.0
def files_03MB : ℕ := total_files - files_06MB - files_10MB
def size_03MB_file : ℝ := 0.3

-- Define the theorem that needs to be proved
theorem minimum_disks_needed : 
    ∃ (disks: ℕ), disks = 10 ∧ 
    (5 * size_06MB_file + 10 * size_10MB_file + 10 * size_03MB_file) ≤ disks * disk_capacity := 
by
  sorry

end minimum_disks_needed_l121_121563


namespace fence_length_l121_121634

theorem fence_length {w l : ℕ} (h1 : l = 2 * w) (h2 : 30 = 2 * l + 2 * w) : l = 10 := by
  sorry

end fence_length_l121_121634


namespace Doug_age_l121_121542

theorem Doug_age (Q J D : ℕ) (h1 : Q = J + 6) (h2 : J = D - 3) (h3 : Q = 19) : D = 16 := by
  sorry

end Doug_age_l121_121542


namespace cookie_store_expense_l121_121236

theorem cookie_store_expense (B D: ℝ) 
  (h₁: D = (1 / 2) * B)
  (h₂: B = D + 20):
  B + D = 60 := by
  sorry

end cookie_store_expense_l121_121236


namespace sum_of_angles_subtended_by_arcs_l121_121588

theorem sum_of_angles_subtended_by_arcs
  (A B X Y C : Type)
  (arc_AX arc_XC : ℝ)
  (h1 : arc_AX = 58)
  (h2 : arc_XC = 62)
  (R S : ℝ)
  (hR : R = arc_AX / 2)
  (hS : S = arc_XC / 2) :
  R + S = 60 :=
by
  rw [hR, hS, h1, h2]
  norm_num

end sum_of_angles_subtended_by_arcs_l121_121588


namespace percent_decrease_is_30_l121_121904

def original_price : ℝ := 100
def sale_price : ℝ := 70
def decrease_in_price : ℝ := original_price - sale_price

theorem percent_decrease_is_30 : (decrease_in_price / original_price) * 100 = 30 :=
by
  sorry

end percent_decrease_is_30_l121_121904


namespace find_g_1_l121_121091

theorem find_g_1 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (2*x - 3) = 2*x^2 - x + 4) : 
  g 1 = 11.5 :=
sorry

end find_g_1_l121_121091


namespace length_of_goods_train_l121_121478

/-- The length of the goods train given the conditions of the problem --/
theorem length_of_goods_train
  (speed_passenger_train : ℝ) (speed_goods_train : ℝ) 
  (time_taken_to_pass : ℝ) (length_goods_train : ℝ) :
  speed_passenger_train = 80 / 3.6 →  -- Convert 80 km/h to m/s
  speed_goods_train    = 32 / 3.6 →  -- Convert 32 km/h to m/s
  time_taken_to_pass   = 9 →
  length_goods_train   = 280 → 
  length_goods_train = (speed_passenger_train + speed_goods_train) * time_taken_to_pass := by
    sorry

end length_of_goods_train_l121_121478


namespace winning_majority_vote_l121_121334

def total_votes : ℕ := 600

def winning_percentage : ℝ := 0.70

def losing_percentage : ℝ := 0.30

theorem winning_majority_vote : (0.70 * (total_votes : ℝ) - 0.30 * (total_votes : ℝ)) = 240 := 
by
  sorry

end winning_majority_vote_l121_121334


namespace find_principal_l121_121680

noncomputable def principal_amount (P : ℝ) (r : ℝ) : Prop :=
  (800 = (P * r * 2) / 100) ∧ (820 = P * (1 + r / 100)^2 - P)

theorem find_principal (P : ℝ) (r : ℝ) (h : principal_amount P r) : P = 8000 :=
by
  sorry

end find_principal_l121_121680


namespace complement_and_intersection_l121_121450

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {-2, -1, 0}
def B : Set ℤ := {0, 1, 2}

theorem complement_and_intersection :
  ((U \ A) ∩ B) = {1, 2} := 
by
  sorry

end complement_and_intersection_l121_121450


namespace star_equiv_zero_l121_121519

-- Define the new operation for real numbers a and b
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Prove that (x^2 - y^2) star (y^2 - x^2) equals 0
theorem star_equiv_zero (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := 
by sorry

end star_equiv_zero_l121_121519


namespace intersection_M_S_l121_121315

def M := {x : ℕ | 0 < x ∧ x < 4 }

def S : Set ℕ := {2, 3, 5}

theorem intersection_M_S : (M ∩ S) = {2, 3} := by
  sorry

end intersection_M_S_l121_121315


namespace least_multiple_of_25_gt_450_correct_l121_121844

def least_multiple_of_25_gt_450 : ℕ :=
  475

theorem least_multiple_of_25_gt_450_correct (n : ℕ) (h1 : 25 ∣ n) (h2 : n > 450) : n ≥ least_multiple_of_25_gt_450 :=
by
  sorry

end least_multiple_of_25_gt_450_correct_l121_121844


namespace days_worked_per_week_l121_121603

theorem days_worked_per_week
  (hourly_wage : ℕ) (hours_per_day : ℕ) (total_earnings : ℕ) (weeks : ℕ)
  (H_wage : hourly_wage = 12) (H_hours : hours_per_day = 9) (H_earnings : total_earnings = 3780) (H_weeks : weeks = 7) :
  (total_earnings / weeks) / (hourly_wage * hours_per_day) = 5 :=
by 
  sorry

end days_worked_per_week_l121_121603


namespace exists_gcd_one_l121_121796

theorem exists_gcd_one (p q r : ℤ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : Int.gcd p (Int.gcd q r) = 1) : ∃ a : ℤ, Int.gcd p (q + a * r) = 1 :=
sorry

end exists_gcd_one_l121_121796


namespace a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l121_121055

theorem a_m_power_m_divides_a_n_power_n:
  ∀ (a : ℕ → ℕ) (m : ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) ∧ m > 1 → ∃ n > m, (a m) ^ m ∣ (a n) ^ n := by 
  sorry

theorem a1_does_not_divide_any_an_power_n:
  ∀ (a : ℕ → ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) → ¬ ∃ n > 1, (a 1) ∣ (a n) ^ n := by
  sorry

end a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l121_121055


namespace javier_first_throw_distance_l121_121302

noncomputable def javelin_first_throw_initial_distance (x : Real) : Real :=
  let throw1_adjusted := 2 * x * 0.95 - 2
  let throw2_adjusted := x * 0.92 - 4
  let throw3_adjusted := 4 * x - 1
  if (throw1_adjusted + throw2_adjusted + throw3_adjusted = 1050) then
    2 * x
  else
    0

theorem javier_first_throw_distance : ∃ x : Real, javelin_first_throw_initial_distance x = 310 :=
by
  sorry

end javier_first_throw_distance_l121_121302


namespace integral_value_l121_121491

noncomputable def integral_sin_pi_over_2_to_pi : ℝ := ∫ x in (Real.pi / 2)..Real.pi, Real.sin x

theorem integral_value : integral_sin_pi_over_2_to_pi = 1 := by
  sorry

end integral_value_l121_121491


namespace julia_tuesday_kids_l121_121878

-- Definitions based on the given conditions in the problem.
def monday_kids : ℕ := 15
def monday_tuesday_kids : ℕ := 33

-- The problem statement to prove the number of kids played with on Tuesday.
theorem julia_tuesday_kids :
  (∃ tuesday_kids : ℕ, tuesday_kids = monday_tuesday_kids - monday_kids) →
  18 = monday_tuesday_kids - monday_kids :=
by
  intro h
  sorry

end julia_tuesday_kids_l121_121878


namespace exists_xy_nat_divisible_l121_121420

theorem exists_xy_nat_divisible (n : ℕ) : ∃ x y : ℤ, (x^2 + y^2 - 2018) % n = 0 :=
by
  use 43, 13
  sorry

end exists_xy_nat_divisible_l121_121420


namespace area_of_playground_l121_121903

variable (l w : ℝ)

-- Conditions:
def perimeter_eq : Prop := 2 * l + 2 * w = 90
def length_three_times_width : Prop := l = 3 * w

-- Theorem:
theorem area_of_playground (h1 : perimeter_eq l w) (h2 : length_three_times_width l w) : l * w = 379.6875 :=
  sorry

end area_of_playground_l121_121903


namespace solve_special_sine_system_l121_121962

noncomputable def special_sine_conditions1 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  let z := -(Real.pi / 2) + 2 * Real.pi * k
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = (-1)^n * Real.pi / 6 + Real.pi * n ∧
  z = -Real.pi / 2 + 2 * Real.pi * k

noncomputable def special_sine_conditions2 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := -Real.pi / 2 + 2 * Real.pi * k
  let z := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = -Real.pi / 2 + 2 * Real.pi * k ∧
  z = (-1)^n * Real.pi / 6 + Real.pi * n

theorem solve_special_sine_system (m n k : ℤ) :
  special_sine_conditions1 m n k ∨ special_sine_conditions2 m n k :=
sorry

end solve_special_sine_system_l121_121962


namespace equation_solution_unique_l121_121924

theorem equation_solution_unique (m a b : ℕ) (hm : 1 < m) (ha : 1 < a) (hb : 1 < b) :
  ((m + 1) * a = m * b + 1) ↔ m = 2 :=
sorry

end equation_solution_unique_l121_121924


namespace least_integer_solution_l121_121085

theorem least_integer_solution (x : ℤ) (h : x^2 = 2 * x + 98) : x = -7 :=
by {
  sorry
}

end least_integer_solution_l121_121085


namespace brody_battery_fraction_l121_121483

theorem brody_battery_fraction (full_battery : ℕ) (battery_left_after_exam : ℕ) (exam_duration : ℕ) 
  (battery_before_exam : ℕ) (battery_used : ℕ) (fraction_used : ℚ) 
  (h1 : full_battery = 60)
  (h2 : battery_left_after_exam = 13)
  (h3 : exam_duration = 2)
  (h4 : battery_before_exam = battery_left_after_exam + exam_duration)
  (h5 : battery_used = full_battery - battery_before_exam)
  (h6 : fraction_used = battery_used / full_battery) :
  fraction_used = 3 / 4 := 
sorry

end brody_battery_fraction_l121_121483


namespace solve_eq1_solve_eq2_l121_121403

theorem solve_eq1 (x : ℝ) : (x^2 - 2 * x - 8 = 0) ↔ (x = 4 ∨ x = -2) :=
sorry

theorem solve_eq2 (x : ℝ) : (2 * x^2 - 4 * x + 1 = 0) ↔ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
sorry

end solve_eq1_solve_eq2_l121_121403


namespace joe_eggs_town_hall_l121_121915

-- Define the conditions.
def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_total : ℕ := 20

-- Define the desired result.
def eggs_town_hall : ℕ := eggs_total - eggs_club_house - eggs_park

-- The statement that needs to be proved.
theorem joe_eggs_town_hall : eggs_town_hall = 3 :=
by
  sorry

end joe_eggs_town_hall_l121_121915


namespace beads_left_in_container_l121_121180

theorem beads_left_in_container 
  (initial_beads green brown red total_beads taken_beads remaining_beads : Nat) 
  (h1 : green = 1) (h2 : brown = 2) (h3 : red = 3) 
  (h4 : total_beads = green + brown + red)
  (h5 : taken_beads = 2) 
  (h6 : remaining_beads = total_beads - taken_beads) : 
  remaining_beads = 4 := 
by
  sorry

end beads_left_in_container_l121_121180


namespace werewolf_eats_per_week_l121_121755
-- First, we import the necessary libraries

-- We define the conditions using Lean definitions

-- The vampire drains 3 people a week
def vampire_drains_per_week : Nat := 3

-- The total population of the village
def village_population : Nat := 72

-- The number of weeks both can live off the population
def weeks : Nat := 9

-- Prove the number of people the werewolf eats per week (W) given the conditions
theorem werewolf_eats_per_week :
  ∃ W : Nat, vampire_drains_per_week * weeks + weeks * W = village_population ∧ W = 5 :=
by
  sorry

end werewolf_eats_per_week_l121_121755


namespace distance_light_travels_500_years_l121_121371

-- Define the given conditions
def distance_in_one_year_miles : ℝ := 5.87e12
def years_traveling : ℝ := 500
def miles_to_kilometers : ℝ := 1.60934

-- Define the expected distance in kilometers after 500 years
def expected_distance_in_kilometers : ℝ  := 4.723e15

-- State the theorem: the distance light travels in 500 years in kilometers
theorem distance_light_travels_500_years :
  (distance_in_one_year_miles * years_traveling * miles_to_kilometers) 
    = expected_distance_in_kilometers := 
by
  sorry

end distance_light_travels_500_years_l121_121371


namespace problem_fraction_of_complex_numbers_l121_121241

/--
Given \(i\) is the imaginary unit, prove that \(\frac {1-i}{1+i} = -i\).
-/
theorem problem_fraction_of_complex_numbers (i : ℂ) (h_i : i^2 = -1) : 
  ((1 - i) / (1 + i)) = -i := 
sorry

end problem_fraction_of_complex_numbers_l121_121241


namespace field_area_l121_121385

theorem field_area
  (L : ℕ) (W : ℕ) (A : ℕ)
  (h₁ : L = 20)
  (h₂ : 2 * W + L = 100)
  (h₃ : A = L * W) :
  A = 800 := by
  sorry

end field_area_l121_121385


namespace polynomial_evaluation_l121_121658

def polynomial_at (x : ℝ) : ℝ :=
  let f := (7 : ℝ) * x^5 + 12 * x^4 - 5 * x^3 - 6 * x^2 + 3 * x - 5
  f

theorem polynomial_evaluation : polynomial_at 3 = 2488 :=
by
  sorry

end polynomial_evaluation_l121_121658


namespace interval_of_segmentation_l121_121770

-- Define the population size and sample size as constants.
def population_size : ℕ := 2000
def sample_size : ℕ := 40

-- State the theorem for the interval of segmentation.
theorem interval_of_segmentation :
  population_size / sample_size = 50 :=
sorry

end interval_of_segmentation_l121_121770


namespace rectangle_area_function_relationship_l121_121436

theorem rectangle_area_function_relationship (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end rectangle_area_function_relationship_l121_121436


namespace marked_price_l121_121566

theorem marked_price (x : ℝ) (purchase_price : ℝ) (selling_price : ℝ) (profit_margin : ℝ) 
  (h_purchase_price : purchase_price = 100)
  (h_profit_margin : profit_margin = 0.2)
  (h_selling_price : selling_price = purchase_price * (1 + profit_margin))
  (h_price_relation : 0.8 * x = selling_price) : 
  x = 150 :=
by sorry

end marked_price_l121_121566


namespace percentage_of_total_money_raised_from_donations_l121_121869

-- Define the conditions
def max_donation := 1200
def num_donors_max := 500
def half_donation := max_donation / 2
def num_donors_half := 3 * num_donors_max
def total_money_raised := 3750000

-- Define the amounts collected from each group
def amount_from_max_donors := num_donors_max * max_donation
def amount_from_half_donors := num_donors_half * half_donation
def total_amount_from_donations := amount_from_max_donors + amount_from_half_donors

-- Define the percentage calculation
def percentage_of_total := (total_amount_from_donations / total_money_raised) * 100

-- State the theorem (but not the proof)
theorem percentage_of_total_money_raised_from_donations : 
  percentage_of_total = 40 := by
  sorry

end percentage_of_total_money_raised_from_donations_l121_121869


namespace natural_number_1981_l121_121511

theorem natural_number_1981 (x : ℕ) 
  (h1 : ∃ a : ℕ, x - 45 = a^2)
  (h2 : ∃ b : ℕ, x + 44 = b^2) :
  x = 1981 :=
sorry

end natural_number_1981_l121_121511


namespace solution_set_of_inequality_l121_121308

theorem solution_set_of_inequality (x : ℝ) : (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by sorry

end solution_set_of_inequality_l121_121308


namespace event_probability_l121_121088

noncomputable def probability_event : ℝ :=
  let a : ℝ := (1 : ℝ) / 2
  let b : ℝ := (3 : ℝ) / 2
  let interval_length : ℝ := 2
  (b - a) / interval_length

theorem event_probability :
  probability_event = (3 : ℝ) / 4 :=
by
  -- Proof step will be supplied here
  sorry

end event_probability_l121_121088


namespace book_loss_percentage_l121_121009

theorem book_loss_percentage 
  (C S : ℝ) 
  (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := 
by 
  sorry

end book_loss_percentage_l121_121009


namespace find_abc_l121_121921

theorem find_abc (a b c : ℝ) 
  (h1 : a = 0.8 * b) 
  (h2 : c = 1.4 * b) 
  (h3 : c - a = 72) : 
  a = 96 ∧ b = 120 ∧ c = 168 := 
by
  sorry

end find_abc_l121_121921


namespace cannot_tile_10x10_board_l121_121220

-- Define the tiling board problem
def typeA_piece (i j : ℕ) : Prop := 
  ((i ≤ 98) ∧ (j ≤ 98) ∧ (i % 2 = 0) ∧ (j % 2 = 0))

def typeB_piece (i j : ℕ) : Prop := 
  ((i + 2 < 10) ∧ (j + 2 < 10))

def typeC_piece (i j : ℕ) : Prop := 
  ((i % 4 = 0 ∨ i % 4 = 2) ∧ (j % 4 = 0 ∨ j % 4 = 2))

-- Main theorem statement
theorem cannot_tile_10x10_board : 
  ¬ (∃ f : Fin 25 → Fin 10 × Fin 10, 
    (∀ k : Fin 25, typeA_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeB_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeC_piece (f k).1 (f k).2)) :=
sorry

end cannot_tile_10x10_board_l121_121220


namespace average_age_of_team_l121_121710

theorem average_age_of_team 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (remaining_avg : ℕ → ℕ) 
  (h1 : n = 11)
  (h2 : captain_age = 27)
  (h3 : wicket_keeper_age = 28)
  (h4 : ∀ A, remaining_avg A = A - 1)
  (h5 : ∀ A, 11 * A = 9 * (remaining_avg A) + captain_age + wicket_keeper_age) : 
  ∃ A, A = 32 :=
by
  sorry

end average_age_of_team_l121_121710


namespace graph_passes_through_point_l121_121239

theorem graph_passes_through_point (a : ℝ) (h : a < 0) : (0, 0) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, (1 - a)^x - 1)} :=
by
  sorry

end graph_passes_through_point_l121_121239


namespace number_of_correct_conclusions_l121_121625

-- Define the conditions given in the problem
def conclusion1 (x : ℝ) : Prop := x > 0 → x > Real.sin x
def conclusion2 (x : ℝ) : Prop := (x - Real.sin x = 0 → x = 0) → (x ≠ 0 → x - Real.sin x ≠ 0)
def conclusion3 (p q : Prop) : Prop := (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def conclusion4 : Prop := ¬(∀ x : ℝ, x - Real.log x > 0) = ∃ x : ℝ, x - Real.log x ≤ 0

-- Prove the number of correct conclusions is 3
theorem number_of_correct_conclusions : 
  (∃ x1 : ℝ, conclusion1 x1) ∧
  (∃ x1 : ℝ, conclusion2 x1) ∧
  (∃ p q : Prop, conclusion3 p q) ∧
  ¬conclusion4 →
  3 = 3 :=
by
  intros
  sorry

end number_of_correct_conclusions_l121_121625


namespace copy_pages_count_l121_121268

-- Definitions and conditions
def cost_per_page : ℕ := 5  -- Cost per page in cents
def total_money : ℕ := 50 * 100  -- Total money in cents

-- Proof goal
theorem copy_pages_count : total_money / cost_per_page = 1000 := 
by sorry

end copy_pages_count_l121_121268


namespace cost_price_of_radio_l121_121867

-- Define the conditions
def selling_price : ℝ := 1335
def loss_percentage : ℝ := 0.11

-- Define what we need to prove
theorem cost_price_of_radio (C : ℝ) (h1 : selling_price = 0.89 * C) : C = 1500 :=
by
  -- This is where we would put the proof, but we can leave it as a sorry for now.
  sorry

end cost_price_of_radio_l121_121867


namespace find_a_sq_plus_b_sq_l121_121480

theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) :
  a^2 + b^2 = 29 := by
  sorry

end find_a_sq_plus_b_sq_l121_121480


namespace find_total_amount_l121_121913

noncomputable def total_amount (A T yearly_income : ℝ) : Prop :=
  0.05 * A + 0.06 * (T - A) = yearly_income

theorem find_total_amount :
  ∃ T : ℝ, total_amount 1600 T 140 ∧ T = 2600 :=
sorry

end find_total_amount_l121_121913


namespace total_earnings_from_selling_working_games_l121_121386

-- Conditions definition
def total_games : ℕ := 16
def broken_games : ℕ := 8
def working_games : ℕ := total_games - broken_games
def game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

-- Proof problem statement
theorem total_earnings_from_selling_working_games : List.sum game_prices = 68 := by
  sorry

end total_earnings_from_selling_working_games_l121_121386


namespace deg_to_rad_neg_630_l121_121622

theorem deg_to_rad_neg_630 :
  (-630 : ℝ) * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end deg_to_rad_neg_630_l121_121622


namespace pascal_triangle_ratio_l121_121175

theorem pascal_triangle_ratio (n r : ℕ) (hn1 : 5 * r = 2 * n - 3) (hn2 : 7 * r = 3 * n - 11) : n = 34 :=
by
  -- The proof steps will fill here eventually
  sorry

end pascal_triangle_ratio_l121_121175


namespace additional_books_l121_121284

theorem additional_books (initial_books total_books additional_books : ℕ)
  (h_initial : initial_books = 54)
  (h_total : total_books = 77) :
  additional_books = total_books - initial_books :=
by
  sorry

end additional_books_l121_121284


namespace eden_stuffed_bears_l121_121681

theorem eden_stuffed_bears 
  (initial_bears : ℕ) 
  (percentage_kept : ℝ) 
  (sisters : ℕ) 
  (eden_initial_bears : ℕ)
  (h1 : initial_bears = 65) 
  (h2 : percentage_kept = 0.40) 
  (h3 : sisters = 4) 
  (h4 : eden_initial_bears = 20) :
  ∃ eden_bears : ℕ, eden_bears = 29 :=
by
  sorry

end eden_stuffed_bears_l121_121681


namespace solve_xyz_l121_121826

variable {x y z : ℝ}

theorem solve_xyz (h1 : (x + y + z) * (xy + xz + yz) = 35) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : x * y * z = 8 := 
by
  sorry

end solve_xyz_l121_121826


namespace sum_of_triangle_angles_is_540_l121_121861

theorem sum_of_triangle_angles_is_540
  (A1 A3 A5 B2 B4 B6 C7 C8 C9 : ℝ)
  (H1 : A1 + A3 + A5 = 180)
  (H2 : B2 + B4 + B6 = 180)
  (H3 : C7 + C8 + C9 = 180) :
  A1 + A3 + A5 + B2 + B4 + B6 + C7 + C8 + C9 = 540 :=
by
  sorry

end sum_of_triangle_angles_is_540_l121_121861


namespace seq_general_form_l121_121533

theorem seq_general_form (p r : ℝ) (a : ℕ → ℝ)
  (hp : p > r)
  (hr : r > 0)
  (h_init : a 1 = r)
  (h_recurrence : ∀ n : ℕ, a (n+1) = p * a n + r^(n+1)) :
  ∀ n : ℕ, a n = r * (p^n - r^n) / (p - r) :=
by
  sorry

end seq_general_form_l121_121533


namespace minimum_value_of_expression_l121_121098

theorem minimum_value_of_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) :
  ∃ (x : ℝ), x = (1 / (a - 1) + 9 / (b - 1)) ∧ x = 6 :=
by
  sorry

end minimum_value_of_expression_l121_121098


namespace melanie_dimes_l121_121208

variable (initial_dimes : ℕ) -- initial dimes Melanie had
variable (dimes_from_dad : ℕ) -- dimes given by dad
variable (dimes_to_mother : ℕ) -- dimes given to mother

def final_dimes (initial_dimes dimes_from_dad dimes_to_mother : ℕ) : ℕ :=
  initial_dimes + dimes_from_dad - dimes_to_mother

theorem melanie_dimes :
  initial_dimes = 7 →
  dimes_from_dad = 8 →
  dimes_to_mother = 4 →
  final_dimes initial_dimes dimes_from_dad dimes_to_mother = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end melanie_dimes_l121_121208


namespace difference_in_amount_paid_l121_121207

variable (P Q : ℝ)

theorem difference_in_amount_paid (hP : P > 0) (hQ : Q > 0) :
  (1.10 * P * 0.80 * Q - P * Q) = -0.12 * (P * Q) := 
by 
  sorry

end difference_in_amount_paid_l121_121207


namespace arithmetic_sequence_a3_l121_121131

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end arithmetic_sequence_a3_l121_121131


namespace problem1_problem2_l121_121935

open Real

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 3| - |2 * x - a|

-- Problem (1)
theorem problem1 {a : ℝ} (h : ∃ x, f x a ≤ -5) : a ≤ -8 ∨ a ≥ 2 :=
sorry

-- Problem (2)
theorem problem2 {a : ℝ} (h : ∀ x, f (x - 1/2) a + f (-x - 1/2) a = 0) : a = 1 :=
sorry

end problem1_problem2_l121_121935


namespace max_of_x_l121_121455

theorem max_of_x (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 10) : x ≤ 3 := by
  sorry

end max_of_x_l121_121455


namespace decrease_percent_revenue_l121_121632

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.05 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 16 := 
by
  sorry

end decrease_percent_revenue_l121_121632


namespace remainder_of_towers_l121_121775

open Nat

def count_towers (m : ℕ) : ℕ :=
  match m with
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 18
  | 5 => 54
  | 6 => 162
  | _ => 0

theorem remainder_of_towers : (count_towers 6) % 100 = 62 :=
  by
  sorry

end remainder_of_towers_l121_121775


namespace smallest_possible_a_l121_121292

theorem smallest_possible_a (a b c : ℝ) 
  (h1 : (∀ x, y = a * x ^ 2 + b * x + c ↔ y = a * (x + 1/3) ^ 2 + 5/9))
  (h2 : a > 0)
  (h3 : ∃ n : ℤ, a + b + c = n) : 
  a = 1/4 :=
sorry

end smallest_possible_a_l121_121292


namespace solution_set_of_inequality_l121_121416

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 0) : 
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio (0 : ℝ) ∪ Set.Ici (1 / 2) :=
by sorry

end solution_set_of_inequality_l121_121416


namespace range_of_b_over_a_l121_121501

-- Define the problem conditions and conclusion
theorem range_of_b_over_a 
  (a b c : ℝ) (A B C : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
  (h_sum_angles : A + B + C = π) 
  (h_sides_relation : ∀ x, (x^2 + c^2 - a^2 - ab = 0 ↔ x = 0)) : 
  1 < b / a ∧ b / a < 2 := 
sorry

end range_of_b_over_a_l121_121501


namespace solve_equation_l121_121005

theorem solve_equation (x : ℝ) : x*(x-3)^2*(5+x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := 
by 
  sorry

end solve_equation_l121_121005


namespace problem_statement_l121_121402

noncomputable def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x / Real.log 2 else sorry

theorem problem_statement : f (1 / 2) < f (1 / 3) ∧ f (1 / 3) < f 2 :=
by
  -- Definitions based on given conditions
  have h1 : ∀ x : ℝ, f (2 - x) = f x := sorry
  have h2 : ∀ x : ℝ, 1 ≤ x → f x = Real.log x / Real.log 2 := sorry
  -- Proof of the statement based on h1 and h2
  sorry

end problem_statement_l121_121402


namespace cole_drive_to_work_time_l121_121126

variables (D : ℝ) (T_work T_home : ℝ)

def speed_work : ℝ := 75
def speed_home : ℝ := 105
def total_time : ℝ := 2

theorem cole_drive_to_work_time :
  (T_work = D / speed_work) ∧
  (T_home = D / speed_home) ∧
  (T_work + T_home = total_time) →
  T_work * 60 = 70 :=
by
  sorry

end cole_drive_to_work_time_l121_121126


namespace student_total_marks_l121_121426

theorem student_total_marks (total_questions correct_answers incorrect_answer_score correct_answer_score : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_answers = 38)
    (h3 : correct_answer_score = 4)
    (h4 : incorrect_answer_score = 1)
    (incorrect_answers := total_questions - correct_answers) 
    : (correct_answers * correct_answer_score - incorrect_answers * incorrect_answer_score) = 130 :=
by
  -- proof to be provided here
  sorry

end student_total_marks_l121_121426


namespace S_5_equals_31_l121_121704

-- Define the sequence sum function S
def S (n : Nat) : Nat := 2^n - 1

-- The theorem to prove that S(5) = 31
theorem S_5_equals_31 : S 5 = 31 :=
by
  rw [S]
  sorry

end S_5_equals_31_l121_121704


namespace inequality_l121_121150

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem inequality : c < a ∧ a < b := 
by 
  sorry

end inequality_l121_121150


namespace no_30_cents_l121_121934

/-- Given six coins selected from nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total value of the six coins cannot be 30 cents or less. -/
theorem no_30_cents {n d q : ℕ} (h : n + d + q = 6) (hn : n * 5 + d * 10 + q * 25 <= 30) : false :=
by
  sorry

end no_30_cents_l121_121934


namespace suitable_sampling_method_l121_121729

noncomputable def is_stratified_sampling_suitable (mountainous hilly flat low_lying sample_size : ℕ) (yield_dependent_on_land_type : Bool) : Bool :=
  if yield_dependent_on_land_type && mountainous + hilly + flat + low_lying > 0 then true else false

theorem suitable_sampling_method :
  is_stratified_sampling_suitable 8000 12000 24000 4000 480 true = true :=
by
  sorry

end suitable_sampling_method_l121_121729


namespace boat_speed_in_still_water_l121_121988

theorem boat_speed_in_still_water (V_b : ℝ) (D : ℝ) (V_s : ℝ) 
  (h1 : V_s = 3) 
  (h2 : D = (V_b + V_s) * 1) 
  (h3 : D = (V_b - V_s) * 1.5) : 
  V_b = 15 := 
by 
  sorry

end boat_speed_in_still_water_l121_121988


namespace quadratic_intersection_y_axis_l121_121381

theorem quadratic_intersection_y_axis :
  (∃ y, y = 3 * (0: ℝ)^2 - 4 * (0: ℝ) + 5 ∧ (0, y) = (0, 5)) :=
by
  sorry

end quadratic_intersection_y_axis_l121_121381


namespace maximal_points_coloring_l121_121530

/-- Given finitely many points in the plane where no three points are collinear,
which are colored either red or green, such that any monochromatic triangle
contains at least one point of the other color in its interior, the maximal number
of such points is 8. -/
theorem maximal_points_coloring (points : Finset (ℝ × ℝ))
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬ ∃ k b, ∀ p ∈ [p1, p2, p3], p.2 = k * p.1 + b)
  (colored : (ℝ × ℝ) → Prop)
  (h_coloring : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    colored p1 = colored p2 → colored p2 = colored p3 →
    ∃ p, p ∈ points ∧ colored p ≠ colored p1) :
  points.card ≤ 8 :=
sorry

end maximal_points_coloring_l121_121530


namespace power_of_b_l121_121227

theorem power_of_b (b n : ℕ) (hb : b > 1) (hn : n > 1) (h : ∀ k > 1, ∃ a_k : ℤ, k ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, b = A ^ n :=
by
  sorry

end power_of_b_l121_121227


namespace answer_to_rarely_infrequently_word_l121_121914

-- Declare variables and definitions based on given conditions
-- In this context, we'll introduce a basic definition for the word "seldom".

noncomputable def is_word_meaning_rarely (w : String) : Prop :=
  w = "seldom"

-- Now state the problem in the form of a Lean theorem
theorem answer_to_rarely_infrequently_word : ∃ w, is_word_meaning_rarely w :=
by
  use "seldom"
  unfold is_word_meaning_rarely
  rfl

end answer_to_rarely_infrequently_word_l121_121914


namespace sum_of_cubes_zero_l121_121345

variables {a b c : ℝ}

theorem sum_of_cubes_zero (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) : a^3 + b^3 + c^3 = 0 :=
sorry

end sum_of_cubes_zero_l121_121345


namespace find_X_l121_121820

def tax_problem (X I T : ℝ) (income : ℝ) (total_tax : ℝ) :=
  (income = 56000) ∧ (total_tax = 8000) ∧ (T = 0.12 * X + 0.20 * (I - X))

theorem find_X :
  ∃ X : ℝ, ∀ I T : ℝ, tax_problem X I T 56000 8000 → X = 40000 := 
  by
    sorry

end find_X_l121_121820


namespace num_20_paise_coins_l121_121752

theorem num_20_paise_coins (x y : ℕ) (h1 : x + y = 344) (h2 : 20 * x + 25 * y = 7100) : x = 300 :=
by
  sorry

end num_20_paise_coins_l121_121752


namespace common_ratio_of_geometric_sequence_l121_121975

-- Define the problem conditions and goal
theorem common_ratio_of_geometric_sequence 
  (a1 : ℝ)  -- nonzero first term
  (h₁ : a1 ≠ 0) -- first term is nonzero
  (r : ℝ)  -- common ratio
  (h₂ : r > 0) -- ratio is positive
  (h₃ : ∀ n m : ℕ, n ≠ m → a1 * r^n ≠ a1 * r^m) -- distinct terms in sequence
  (h₄ : a1 * r * r * r = (a1 * r) * (a1 * r^3) ∧ a1 * r ≠ (a1 * r^4)) -- arithmetic sequence condition
  : r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l121_121975


namespace pitchers_of_lemonade_l121_121985

theorem pitchers_of_lemonade (glasses_per_pitcher : ℕ) (total_glasses_served : ℕ)
  (h1 : glasses_per_pitcher = 5) (h2 : total_glasses_served = 30) :
  total_glasses_served / glasses_per_pitcher = 6 := by
  sorry

end pitchers_of_lemonade_l121_121985


namespace probability_A_wins_championship_distribution_and_expectation_B_l121_121108

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l121_121108


namespace subtraction_correct_l121_121083

def x : ℝ := 5.75
def y : ℝ := 1.46
def result : ℝ := 4.29

theorem subtraction_correct : x - y = result := 
by
  sorry

end subtraction_correct_l121_121083


namespace trajectory_of_Q_l121_121034

-- Define Circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define Line l
def lineL (x y : ℝ) : Prop := x + y = 2

-- Define Conditions based on polar definitions
def polarCircle (ρ θ : ℝ) : Prop := ρ = 2

def polarLine (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 2

-- Define points on ray OP
def pointP (ρ₁ θ : ℝ) : Prop := ρ₁ = 2 / (Real.cos θ + Real.sin θ)
def pointR (ρ₂ θ : ℝ) : Prop := ρ₂ = 2

-- Prove the trajectory of Q
theorem trajectory_of_Q (O P R Q : ℝ × ℝ)
  (ρ₁ θ ρ ρ₂ : ℝ)
  (h1: circleC O.1 O.2)
  (h2: lineL P.1 P.2)
  (h3: polarCircle ρ₂ θ)
  (h4: polarLine ρ₁ θ)
  (h5: ρ * ρ₁ = ρ₂^2) :
  ρ = 2 * (Real.cos θ + Real.sin θ) :=
by
  sorry

end trajectory_of_Q_l121_121034


namespace number_of_people_l121_121242

theorem number_of_people (total_eggs : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : 
  total_eggs = 36 → eggs_per_omelet = 4 → omelets_per_person = 3 → 
  (total_eggs / eggs_per_omelet) / omelets_per_person = 3 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_l121_121242


namespace radius_of_inscribed_circle_l121_121790

theorem radius_of_inscribed_circle (a b c : ℝ) (r : ℝ) 
  (ha : a = 5) (hb : b = 10) (hc : c = 20)
  (h : 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))) :
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by
  -- Statements and conditions are setup, but the proof is omitted.
  sorry

end radius_of_inscribed_circle_l121_121790


namespace sequence_bounded_l121_121078

theorem sequence_bounded (a : ℕ → ℕ) (a1 : ℕ) (h1 : a 0 = a1)
  (heven : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n) = a (2 * n - 1) - d)
  (hodd : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n + 1) = a (2 * n) + d) :
  ∀ n : ℕ, a n ≤ 10 * a1 := 
by
  sorry

end sequence_bounded_l121_121078


namespace inequality_inequality_l121_121349

theorem inequality_inequality (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) :
  ac + bd ≤ 8 :=
sorry

end inequality_inequality_l121_121349


namespace corn_purchase_l121_121607

theorem corn_purchase : ∃ c b : ℝ, c + b = 30 ∧ 89 * c + 55 * b = 2170 ∧ c = 15.3 := 
by
  sorry

end corn_purchase_l121_121607


namespace evaluate_polynomial_at_5_l121_121514

def polynomial (x : ℕ) : ℕ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem evaluate_polynomial_at_5 : polynomial 5 = 7548 := by
  sorry

end evaluate_polynomial_at_5_l121_121514


namespace geric_initial_bills_l121_121287

theorem geric_initial_bills (G K J : ℕ) 
  (h1: G = 2 * K)
  (h2: K = J - 2)
  (h3: J - 3 = 7) : G = 16 := 
  by 
  sorry

end geric_initial_bills_l121_121287


namespace vector_subtraction_l121_121642

open Real

def vector_a : (ℝ × ℝ) := (3, 2)
def vector_b : (ℝ × ℝ) := (0, -1)

theorem vector_subtraction : 
  3 • vector_b - vector_a = (-3, -5) :=
by 
  -- Proof needs to be written here.
  sorry

end vector_subtraction_l121_121642


namespace speedster_convertibles_l121_121611

noncomputable def total_inventory (not_speedsters : Nat) (fraction_not_speedsters : ℝ) : ℝ :=
  (not_speedsters : ℝ) / fraction_not_speedsters

noncomputable def number_speedsters (total_inventory : ℝ) (fraction_speedsters : ℝ) : ℝ :=
  total_inventory * fraction_speedsters

noncomputable def number_convertibles (number_speedsters : ℝ) (fraction_convertibles : ℝ) : ℝ :=
  number_speedsters * fraction_convertibles

theorem speedster_convertibles : (not_speedsters = 30) ∧ (fraction_not_speedsters = 2 / 3) ∧ (fraction_speedsters = 1 / 3) ∧ (fraction_convertibles = 4 / 5) →
  number_convertibles (number_speedsters (total_inventory not_speedsters fraction_not_speedsters) fraction_speedsters) fraction_convertibles = 12 :=
by
  intros h
  sorry

end speedster_convertibles_l121_121611


namespace find_f_three_l121_121941

noncomputable def f : ℝ → ℝ := sorry -- f(x) is a linear function

axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

axiom equation : ∀ x, f x = 3 * (f⁻¹ x) + 9

axiom f_zero : f 0 = 3

axiom f_inv_three : f⁻¹ 3 = 0

theorem find_f_three : f 3 = 6 * Real.sqrt 3 := 
by sorry

end find_f_three_l121_121941


namespace count_even_divisors_8_l121_121862

theorem count_even_divisors_8! :
  ∃ (even_divisors total : ℕ),
    even_divisors = 84 ∧
    total = 56 :=
by
  /-
    To formulate the problem in Lean:
    We need to establish two main facts:
    1. The count of even divisors of 8! is 84.
    2. The count of those even divisors that are multiples of both 2 and 3 is 56.
  -/
  sorry

end count_even_divisors_8_l121_121862


namespace Uncle_Fyodor_age_l121_121470

variable (age : ℕ)

-- Conditions from the problem
def Sharik_statement : Prop := age > 11
def Matroskin_statement : Prop := age > 10

-- The theorem stating the problem to be proved
theorem Uncle_Fyodor_age
  (H : (Sharik_statement age ∧ ¬Matroskin_statement age) ∨ (¬Sharik_statement age ∧ Matroskin_statement age)) :
  age = 11 :=
by
  sorry

end Uncle_Fyodor_age_l121_121470


namespace ornamental_rings_remaining_l121_121391

-- Definitions based on conditions
variable (initial_stock : ℕ) (final_stock : ℕ)

-- Condition 1
def condition1 := initial_stock + 200 = 3 * initial_stock

-- Condition 2
def condition2 := final_stock = (200 + initial_stock) * 1 / 4 - (200 + initial_stock) / 4 + 300 - 150

-- Theorem statement to prove the final stock is 225
theorem ornamental_rings_remaining
  (h1 : condition1 initial_stock)
  (h2 : condition2 initial_stock final_stock) :
  final_stock = 225 :=
sorry

end ornamental_rings_remaining_l121_121391


namespace sum_of_coefficients_no_y_l121_121249

-- Defining the problem conditions
def expansion (a b c : ℤ) (n : ℕ) : ℤ := (a - b + c)^n

-- Summing the coefficients of the terms that do not contain y
noncomputable def coefficients_sum (a b : ℤ) (n : ℕ) : ℤ :=
  (a - b)^n

theorem sum_of_coefficients_no_y (n : ℕ) (h : 0 < n) : 
  coefficients_sum 4 3 n = 1 :=
by
  sorry

end sum_of_coefficients_no_y_l121_121249


namespace problem_l121_121475

theorem problem (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + (1/r^4) = 7 := 
by
  sorry

end problem_l121_121475


namespace average_weight_increase_l121_121487

noncomputable def average_increase (A : ℝ) : ℝ :=
  let initial_total := 10 * A
  let new_total := initial_total + 25
  let new_average := new_total / 10
  new_average - A

theorem average_weight_increase (A : ℝ) : average_increase A = 2.5 := by
  sorry

end average_weight_increase_l121_121487


namespace pen_and_pencil_total_cost_l121_121722

theorem pen_and_pencil_total_cost :
  ∀ (pen pencil : ℕ), pen = 4 → pen = 2 * pencil → pen + pencil = 6 :=
by
  intros pen pencil
  intro h1
  intro h2
  sorry

end pen_and_pencil_total_cost_l121_121722


namespace inequality_for_positive_reals_l121_121602

theorem inequality_for_positive_reals 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a^3 + b^3 + a * b * c)) + (1 / (b^3 + c^3 + a * b * c)) + 
  (1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) := 
sorry

end inequality_for_positive_reals_l121_121602


namespace carbon_copies_after_folding_l121_121667

def initial_sheets : ℕ := 6
def initial_carbons (sheets : ℕ) : ℕ := sheets - 1
def final_copies (sheets : ℕ) : ℕ := sheets - 1

theorem carbon_copies_after_folding :
  (final_copies initial_sheets) =
  initial_carbons initial_sheets :=
by {
    -- sorry is a placeholder for the proof
    sorry
}

end carbon_copies_after_folding_l121_121667


namespace total_money_made_l121_121504

structure Building :=
(floors : Nat)
(rooms_per_floor : Nat)

def cleaning_time_per_room : Nat := 8

structure CleaningRates :=
(first_4_hours_rate : Int)
(next_4_hours_rate : Int)
(unpaid_break_hours : Nat)

def supply_cost : Int := 1200

def total_earnings (b : Building) (c : CleaningRates) : Int :=
  let rooms := b.floors * b.rooms_per_floor
  let earnings_per_room := (4 * c.first_4_hours_rate + 4 * c.next_4_hours_rate)
  rooms * earnings_per_room - supply_cost

theorem total_money_made (b : Building) (c : CleaningRates) : 
  b.floors = 12 →
  b.rooms_per_floor = 25 →
  cleaning_time_per_room = 8 →
  c.first_4_hours_rate = 20 →
  c.next_4_hours_rate = 25 →
  c.unpaid_break_hours = 1 →
  total_earnings b c = 52800 := 
by
  intros
  sorry

end total_money_made_l121_121504


namespace g_at_neg10_l121_121908

def g (x : ℤ) : ℤ := 
  if x < -3 then 3 * x + 7 else 4 - x

theorem g_at_neg10 : g (-10) = -23 := by
  -- The proof goes here
  sorry

end g_at_neg10_l121_121908


namespace impossible_odd_sum_l121_121580

theorem impossible_odd_sum (n m : ℤ) (h1 : (n^3 + m^3) % 2 = 0) (h2 : (n^3 + m^3) % 4 = 0) : (n + m) % 2 = 0 :=
sorry

end impossible_odd_sum_l121_121580


namespace Jermaine_more_than_Terrence_l121_121880

theorem Jermaine_more_than_Terrence :
  ∀ (total_earnings Terrence_earnings Emilee_earnings : ℕ),
    total_earnings = 90 →
    Terrence_earnings = 30 →
    Emilee_earnings = 25 →
    (total_earnings - Terrence_earnings - Emilee_earnings) - Terrence_earnings = 5 := by
  sorry

end Jermaine_more_than_Terrence_l121_121880


namespace first_car_departure_time_l121_121917

variable (leave_time : Nat) -- in minutes past 8:00 am

def speed : Nat := 60 -- km/h
def firstCarTimeAt32 : Nat := 32 -- minutes since 8:00 am
def secondCarFactorAt32 : Nat := 3
def firstCarTimeAt39 : Nat := 39 -- minutes since 8:00 am
def secondCarFactorAt39 : Nat := 2

theorem first_car_departure_time :
  let firstCarSpeed := (60 / 60 : Nat) -- km/min
  let d1_32 := firstCarSpeed * firstCarTimeAt32
  let d2_32 := firstCarSpeed * (firstCarTimeAt32 - leave_time)
  let d1_39 := firstCarSpeed * firstCarTimeAt39
  let d2_39 := firstCarSpeed * (firstCarTimeAt39 - leave_time)
  d1_32 = secondCarFactorAt32 * d2_32 →
  d1_39 = secondCarFactorAt39 * d2_39 →
  leave_time = 11 :=
by
  intros h1 h2
  sorry

end first_car_departure_time_l121_121917


namespace area_of_circle_above_below_lines_l121_121526

noncomputable def circle_area : ℝ :=
  40 * Real.pi

theorem area_of_circle_above_below_lines :
  ∃ (x y : ℝ), (x^2 + y^2 - 16*x - 8*y = 0) ∧ (y > x - 4) ∧ (y < -x + 4) ∧
  (circle_area = 40 * Real.pi) :=
  sorry

end area_of_circle_above_below_lines_l121_121526


namespace min_value_of_expression_l121_121895

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y = 5 := 
sorry

end min_value_of_expression_l121_121895


namespace slope_of_monotonically_decreasing_function_l121_121464

theorem slope_of_monotonically_decreasing_function
  (k b : ℝ)
  (H : ∀ x₁ x₂, x₁ ≤ x₂ → k * x₁ + b ≥ k * x₂ + b) : k < 0 := sorry

end slope_of_monotonically_decreasing_function_l121_121464


namespace fraction_identity_l121_121427

theorem fraction_identity (x y z v : ℝ) (hy : y ≠ 0) (hv : v ≠ 0)
    (h : x / y + z / v = 1) : x / y - z / v = (x / y) ^ 2 - (z / v) ^ 2 := by
  sorry

end fraction_identity_l121_121427


namespace fred_grew_38_cantaloupes_l121_121045

/-
  Fred grew some cantaloupes. Tim grew 44 cantaloupes.
  Together, they grew a total of 82 cantaloupes.
  Prove that Fred grew 38 cantaloupes.
-/

theorem fred_grew_38_cantaloupes (T F : ℕ) (h1 : T = 44) (h2 : T + F = 82) : F = 38 :=
by
  rw [h1] at h2
  linarith

end fred_grew_38_cantaloupes_l121_121045


namespace minimum_value_of_a_l121_121964

theorem minimum_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y ≥ 9) : ∃ a > 0, a ≥ 4 :=
sorry

end minimum_value_of_a_l121_121964


namespace remainder_when_P_divided_by_DD_l121_121894

noncomputable def remainder (a b : ℕ) : ℕ := a % b

theorem remainder_when_P_divided_by_DD' (P D Q R D' Q'' R'' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  remainder P (D * D') = R :=
by {
  sorry
}

end remainder_when_P_divided_by_DD_l121_121894


namespace find_m_of_odd_number_sequence_l121_121584

theorem find_m_of_odd_number_sequence : 
  ∃ m : ℕ, m > 1 ∧ (∃ a : ℕ, a = m * (m - 1) + 1 ∧ a = 2023) ↔ m = 45 :=
by
    sorry

end find_m_of_odd_number_sequence_l121_121584


namespace div_problem_l121_121176

theorem div_problem (a b c : ℝ) (h1 : a / (b * c) = 4) (h2 : (a / b) / c = 12) : a / b = 4 * Real.sqrt 3 := 
by
  sorry

end div_problem_l121_121176


namespace solve_inequality1_solve_inequality2_l121_121806

-- Proof problem 1
theorem solve_inequality1 (x : ℝ) : 
  2 < |2 * x - 5| → |2 * x - 5| ≤ 7 → -1 ≤ x ∧ x < (3 / 2) ∨ (7 / 2) < x ∧ x ≤ 6 :=
sorry

-- Proof problem 2
theorem solve_inequality2 (x : ℝ) : 
  (1 / (x - 1)) > (x + 1) → x < - Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2) :=
sorry

end solve_inequality1_solve_inequality2_l121_121806


namespace otimes_eq_abs_m_leq_m_l121_121545

noncomputable def otimes (x y : ℝ) : ℝ :=
if x ≤ y then x else y

theorem otimes_eq_abs_m_leq_m' :
  ∀ (m : ℝ), otimes (abs (m - 1)) m = abs (m - 1) → m ∈ Set.Ici (1 / 2) := 
by
  sorry

end otimes_eq_abs_m_leq_m_l121_121545


namespace intersection_complement_l121_121216

open Set

def U : Set ℝ := univ
def A : Set ℤ := {x : ℤ | -3 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement (U A B : Set ℝ) : A ∩ (U \ B) = {0, 1} := sorry

end intersection_complement_l121_121216


namespace value_of_b_l121_121193

theorem value_of_b (a b c : ℤ) : 
  (∃ d : ℤ, a = 17 + d ∧ b = 17 + 2 * d ∧ c = 17 + 3 * d ∧ 41 = 17 + 4 * d) → b = 29 :=
by
  intros h
  sorry


end value_of_b_l121_121193


namespace john_saves_money_l121_121040

theorem john_saves_money :
  let original_spending := 4 * 2
  let new_price_per_coffee := 2 + (2 * 0.5)
  let new_coffees := 4 / 2
  let new_spending := new_coffees * new_price_per_coffee
  original_spending - new_spending = 2 :=
by
  -- calculations omitted
  sorry

end john_saves_money_l121_121040


namespace Erik_money_left_l121_121817

theorem Erik_money_left 
  (init_money : ℝ)
  (loaf_of_bread : ℝ) (n_loaves_of_bread : ℝ)
  (carton_of_orange_juice : ℝ) (n_cartons_of_orange_juice : ℝ)
  (dozen_eggs : ℝ) (n_dozens_of_eggs : ℝ)
  (chocolate_bar : ℝ) (n_chocolate_bars : ℝ)
  (pound_apples : ℝ) (n_pounds_apples : ℝ)
  (pound_grapes : ℝ) (n_pounds_grapes : ℝ)
  (discount_bread_and_eggs : ℝ) (discount_other_items : ℝ)
  (sales_tax : ℝ) :
  n_loaves_of_bread = 3 →
  loaf_of_bread = 3 →
  n_cartons_of_orange_juice = 3 →
  carton_of_orange_juice = 6 →
  n_dozens_of_eggs = 2 →
  dozen_eggs = 4 →
  n_chocolate_bars = 5 →
  chocolate_bar = 2 →
  n_pounds_apples = 4 →
  pound_apples = 1.25 →
  n_pounds_grapes = 1.5 →
  pound_grapes = 2.5 →
  discount_bread_and_eggs = 0.1 →
  discount_other_items = 0.05 →
  sales_tax = 0.06 →
  init_money = 86 →
  (init_money - 
     (n_loaves_of_bread * loaf_of_bread * (1 - discount_bread_and_eggs) + 
      n_cartons_of_orange_juice * carton_of_orange_juice * (1 - discount_other_items) + 
      n_dozens_of_eggs * dozen_eggs * (1 - discount_bread_and_eggs) + 
      n_chocolate_bars * chocolate_bar * (1 - discount_other_items) + 
      n_pounds_apples * pound_apples * (1 - discount_other_items) + 
      n_pounds_grapes * pound_grapes * (1 - discount_other_items)) * (1 + sales_tax)) = 32.78 :=
by
  sorry

end Erik_money_left_l121_121817


namespace average_six_consecutive_integers_starting_with_d_l121_121218

theorem average_six_consecutive_integers_starting_with_d (c : ℝ) (d : ℝ)
  (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5 :=
by
  sorry -- Proof to be completed

end average_six_consecutive_integers_starting_with_d_l121_121218


namespace circle_areas_sum_l121_121354

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l121_121354


namespace inequality_solution_l121_121186

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2 ↔ (0 < x ∧ x ≤ 0.5) ∨ (6 ≤ x) :=
by { sorry }

end inequality_solution_l121_121186


namespace card_at_position_52_l121_121198

def cards_order : List String := ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

theorem card_at_position_52 : cards_order[(52 % 13)] = "A" :=
by
  -- proof will be added here
  sorry

end card_at_position_52_l121_121198


namespace pipe_q_fill_time_l121_121261

theorem pipe_q_fill_time :
  ∀ (T : ℝ), (2 * (1 / 10 + 1 / T) + 10 * (1 / T) = 1) → T = 15 :=
by
  intro T
  intro h
  sorry

end pipe_q_fill_time_l121_121261


namespace max_sum_arith_seq_l121_121509

theorem max_sum_arith_seq :
  let a1 := 29
  let d := 2
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  S_n 10 = S_n 20 → S_n 20 = 960 := by
sorry

end max_sum_arith_seq_l121_121509


namespace watch_loss_percentage_l121_121211

theorem watch_loss_percentage 
  (cost_price : ℕ) (gain_percent : ℕ) (extra_amount : ℕ) (selling_price_loss : ℕ)
  (h_cost_price : cost_price = 2500)
  (h_gain_percent : gain_percent = 10)
  (h_extra_amount : extra_amount = 500)
  (h_gain_condition : cost_price + gain_percent * cost_price / 100 = selling_price_loss + extra_amount) :
  (cost_price - selling_price_loss) * 100 / cost_price = 10 := 
by 
  sorry

end watch_loss_percentage_l121_121211


namespace dhoni_remaining_earnings_l121_121674

theorem dhoni_remaining_earnings :
  let rent := 0.20
  let dishwasher := 0.15
  let bills := 0.10
  let car := 0.08
  let grocery := 0.12
  let tax := 0.05
  let expenses := rent + dishwasher + bills + car + grocery + tax
  let remaining_after_expenses := 1.0 - expenses
  let savings := 0.40 * remaining_after_expenses
  let remaining_after_savings := remaining_after_expenses - savings
  remaining_after_savings = 0.18 := by
sorry

end dhoni_remaining_earnings_l121_121674


namespace k_domain_all_reals_l121_121716

noncomputable def domain_condition (k : ℝ) : Prop :=
  9 + 28 * k < 0

noncomputable def k_values : Set ℝ :=
  {k : ℝ | domain_condition k}

theorem k_domain_all_reals :
  k_values = {k : ℝ | k < -9 / 28} :=
by
  sorry

end k_domain_all_reals_l121_121716


namespace sum_of_distances_from_circumcenter_to_sides_l121_121367

theorem sum_of_distances_from_circumcenter_to_sides :
  let r1 := 3
  let r2 := 5
  let r3 := 7
  let a := r1 + r2
  let b := r1 + r3
  let c := r2 + r3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r_incircle := area / s
  r_incircle = Real.sqrt 7 →
  let sum_distances := (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
  sum_distances = (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
:= sorry

end sum_of_distances_from_circumcenter_to_sides_l121_121367


namespace exist_two_divisible_by_n_l121_121751

theorem exist_two_divisible_by_n (n : ℤ) (a : Fin (n.toNat + 1) → ℤ) :
  ∃ (i j : Fin (n.toNat + 1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end exist_two_divisible_by_n_l121_121751


namespace berries_difference_l121_121373

theorem berries_difference (total_berries : ℕ) (dima_rate : ℕ) (sergey_rate : ℕ)
  (sergey_berries_picked : ℕ) (dima_berries_picked : ℕ)
  (dima_basket : ℕ) (sergey_basket : ℕ) :
  total_berries = 900 →
  sergey_rate = 2 * dima_rate →
  sergey_berries_picked = 2 * (total_berries / 3) →
  dima_berries_picked = total_berries / 3 →
  sergey_basket = sergey_berries_picked / 2 →
  dima_basket = (2 * dima_berries_picked) / 3 →
  sergey_basket > dima_basket ∧ sergey_basket - dima_basket = 100 :=
by
  intro h_total h_rate h_sergey_picked h_dima_picked h_sergey_basket h_dima_basket
  sorry

end berries_difference_l121_121373


namespace ribbon_per_gift_l121_121453

-- Definitions for the conditions in the problem
def total_ribbon_used : ℚ := 4/15
def num_gifts: ℕ := 5

-- Statement to prove
theorem ribbon_per_gift : total_ribbon_used / num_gifts = 4 / 75 :=
by
  sorry

end ribbon_per_gift_l121_121453


namespace scientific_notation_of_distance_l121_121359

theorem scientific_notation_of_distance :
  ∃ a n, (1 ≤ a ∧ a < 10) ∧ 384000 = a * 10^n ∧ a = 3.84 ∧ n = 5 :=
by
  sorry

end scientific_notation_of_distance_l121_121359


namespace original_avg_expenditure_correct_l121_121430

variables (A B C a b c X Y Z : ℝ)
variables (hA : A > 0) (hB : B > 0) (hC : C > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem original_avg_expenditure_correct
    (h_orig_exp : (A * X + B * Y + C * Z) / (A + B + C) - 1 
    = ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42):
    True := 
sorry

end original_avg_expenditure_correct_l121_121430


namespace ratio_of_probabilities_l121_121797

noncomputable def balls_toss (balls bins : ℕ) : Nat := by
  sorry

def prob_A : ℚ := by
  sorry
  
def prob_B : ℚ := by
  sorry

theorem ratio_of_probabilities (balls : ℕ) (bins : ℕ) 
  (h_balls : balls = 20) (h_bins : bins = 5) (p q : ℚ) 
  (h_p : p = prob_A) (h_q : q = prob_B) :
  (p / q) = 4 := by
  sorry

end ratio_of_probabilities_l121_121797


namespace average_infection_per_round_l121_121677

theorem average_infection_per_round (x : ℝ) (h1 : 1 + x + x * (1 + x) = 100) : x = 9 :=
sorry

end average_infection_per_round_l121_121677


namespace find_two_digit_number_l121_121882

theorem find_two_digit_number (x : ℕ) (h1 : (x + 3) % 3 = 0) (h2 : (x + 7) % 7 = 0) (h3 : (x - 4) % 4 = 0) : x = 84 := 
by
  -- Place holder for the proof
  sorry

end find_two_digit_number_l121_121882


namespace parabola_satisfies_given_condition_l121_121070

variable {p : ℝ}
variable {x1 x2 : ℝ}

-- Condition 1: The equation of the parabola is y^2 = 2px where p > 0.
def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Condition 2: The parabola has a focus F.
-- Condition 3: A line passes through the focus F with an inclination angle of π/3.
def line_through_focus (p : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p / 2)

-- Condition 4 & 5: The line intersects the parabola at points A and B with distance |AB| = 8.
def intersection_points (p : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 ≠ x2 ∧ parabola_equation p x1 (Real.sqrt 3 * (x1 - p / 2)) ∧ parabola_equation p x2 (Real.sqrt 3 * (x2 - p / 2)) ∧
  abs (x1 - x2) * Real.sqrt (1 + 3) = 8

-- The proof statement
theorem parabola_satisfies_given_condition (hp : 0 < p) (hintersect : intersection_points p x1 x2) : 
  parabola_equation 3 x1 (Real.sqrt 3 * (x1 - 3 / 2)) ∧ parabola_equation 3 x2 (Real.sqrt 3 * (x2 - 3 / 2)) := sorry

end parabola_satisfies_given_condition_l121_121070


namespace conditional_probability_A_given_B_l121_121132

noncomputable def P (A B : Prop) : ℝ := sorry -- Placeholder for the probability function

variables (A B : Prop)

axiom P_A_def : P A = 4/15
axiom P_B_def : P B = 2/15
axiom P_AB_def : P (A ∧ B) = 1/10

theorem conditional_probability_A_given_B : P (A ∧ B) / P B = 3/4 :=
by
  rw [P_AB_def, P_B_def]
  norm_num
  sorry

end conditional_probability_A_given_B_l121_121132


namespace eval_expression_l121_121476

theorem eval_expression : 3 * (3 + 3) / 3 = 6 := by
  sorry

end eval_expression_l121_121476


namespace tenth_term_arithmetic_sequence_l121_121024

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ : ℚ) (d : ℚ), 
  (a₁ = 3/4) → (d = 1/2) →
  (a₁ + 9 * d) = 21/4 :=
by
  intro a₁ d ha₁ hd
  rw [ha₁, hd]
  sorry

end tenth_term_arithmetic_sequence_l121_121024


namespace negative_integer_example_l121_121266

def is_negative_integer (n : ℤ) := n < 0

theorem negative_integer_example : is_negative_integer (-2) :=
by
  -- Proof will go here
  sorry

end negative_integer_example_l121_121266


namespace find_smallest_m_l121_121925

def is_in_S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), ((1 / 2 : ℝ) ≤ x) ∧ (x ≤ Real.sqrt 2 / 2) ∧ (z = (x : ℂ) + (y : ℂ) * Complex.I)

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def smallest_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, is_in_S z ∧ is_nth_root_of_unity z n

theorem find_smallest_m : smallest_m 24 :=
  sorry

end find_smallest_m_l121_121925


namespace number_of_factors_l121_121813

theorem number_of_factors : 
  ∃ (count : ℕ), count = 45 ∧
    (∀ n : ℕ, (1 ≤ n ∧ n ≤ 500) → 
      ∃ a b : ℤ, (x - a) * (x - b) = x^2 + 2 * x - n) :=
by
  sorry

end number_of_factors_l121_121813


namespace exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l121_121205

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared :
  ∃ (n : ℕ), sum_of_digits n = 1000 ∧ sum_of_digits (n ^ 2) = 1000000 := sorry

end exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l121_121205


namespace tan_pi_over_12_plus_tan_7pi_over_12_l121_121067

theorem tan_pi_over_12_plus_tan_7pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (7 * Real.pi / 12)) = -4 * (3 - Real.sqrt 3) / 5 :=
by
  sorry

end tan_pi_over_12_plus_tan_7pi_over_12_l121_121067


namespace highest_temperature_l121_121461

theorem highest_temperature
  (initial_temp : ℝ := 60)
  (final_temp : ℝ := 170)
  (heating_rate : ℝ := 5)
  (cooling_rate : ℝ := 7)
  (total_time : ℝ := 46) :
  ∃ T : ℝ, (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time ∧ T = 240 :=
by
  sorry

end highest_temperature_l121_121461


namespace star_problem_l121_121991

def star_problem_proof (p q r s u : ℤ) (S : ℤ): Prop :=
  (S = 64) →
  ({n : ℤ | n = 19 ∨ n = 21 ∨ n = 23 ∨ n = 25 ∨ n = 27} = {p, q, r, s, u}) →
  (p + q + r + s + u = 115) →
  (9 + p + q + 7 = S) →
  (3 + p + u + 15 = S) →
  (3 + q + r + 11 = S) →
  (9 + u + s + 11 = S) →
  (15 + s + r + 7 = S) →
  (q = 27)

theorem star_problem : ∃ p q r s u S, star_problem_proof p q r s u S := by
  -- Proof goes here
  sorry

end star_problem_l121_121991


namespace avg_and_var_of_scaled_shifted_data_l121_121099

-- Definitions of average and variance
noncomputable def avg (l: List ℝ) : ℝ := (l.sum) / l.length
noncomputable def var (l: List ℝ) : ℝ := (l.map (λ x => (x - avg l) ^ 2)).sum / l.length

theorem avg_and_var_of_scaled_shifted_data
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_avg : avg (List.ofFn x) = 2)
  (h_var : var (List.ofFn x) = 3) :
  avg (List.ofFn (λ i => 2 * x i + 3)) = 7 ∧ var (List.ofFn (λ i => 2 * x i + 3)) = 12 := by
  sorry

end avg_and_var_of_scaled_shifted_data_l121_121099


namespace marys_remaining_money_l121_121182

def drinks_cost (p : ℝ) := 4 * p
def medium_pizzas_cost (p : ℝ) := 3 * (3 * p)
def large_pizzas_cost (p : ℝ) := 2 * (5 * p)
def total_initial_money := 50

theorem marys_remaining_money (p : ℝ) : 
  total_initial_money - (drinks_cost p + medium_pizzas_cost p + large_pizzas_cost p) = 50 - 23 * p :=
by
  sorry

end marys_remaining_money_l121_121182


namespace regular_price_of_tire_l121_121025

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 3 = 240) : x = 79 :=
by
  sorry

end regular_price_of_tire_l121_121025


namespace subtraction_888_55_555_55_l121_121278

theorem subtraction_888_55_555_55 : 888.88 - 555.55 = 333.33 :=
by
  sorry

end subtraction_888_55_555_55_l121_121278


namespace midpoint_of_segment_l121_121474

theorem midpoint_of_segment (a b : ℝ) : (a + b) / 2 = (a + b) / 2 :=
sorry

end midpoint_of_segment_l121_121474


namespace factor_expression_l121_121743

theorem factor_expression (x : ℝ) :
  x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := 
  sorry

end factor_expression_l121_121743


namespace infinite_series_sum_eq_33_div_8_l121_121344

noncomputable def infinite_series_sum: ℝ :=
  ∑' n: ℕ, n^3 / (3^n : ℝ)

theorem infinite_series_sum_eq_33_div_8:
  infinite_series_sum = 33 / 8 :=
sorry

end infinite_series_sum_eq_33_div_8_l121_121344


namespace initial_members_count_l121_121513

theorem initial_members_count (n : ℕ) (W : ℕ)
  (h1 : W = n * 48)
  (h2 : W + 171 = (n + 2) * 51) : 
  n = 23 :=
by sorry

end initial_members_count_l121_121513


namespace right_triangle_area_l121_121481

theorem right_triangle_area (a_square_area b_square_area hypotenuse_square_area : ℝ)
  (ha : a_square_area = 36) (hb : b_square_area = 64) (hc : hypotenuse_square_area = 100)
  (leg1 leg2 hypotenuse : ℝ)
  (hleg1 : leg1 * leg1 = a_square_area)
  (hleg2 : leg2 * leg2 = b_square_area)
  (hhyp : hypotenuse * hypotenuse = hypotenuse_square_area) :
  (1/2) * leg1 * leg2 = 24 :=
by
  sorry

end right_triangle_area_l121_121481


namespace cupric_cyanide_formation_l121_121437

/--
Given:
1 mole of CuSO₄ 
2 moles of HCN

Prove:
The number of moles of Cu(CN)₂ formed is 0.
-/
theorem cupric_cyanide_formation (CuSO₄ HCN : ℕ) (h₁ : CuSO₄ = 1) (h₂ : HCN = 2) : 0 = 0 :=
by
  -- Proof goes here
  sorry

end cupric_cyanide_formation_l121_121437


namespace perfect_score_l121_121592

theorem perfect_score (P : ℕ) (h : 3 * P = 63) : P = 21 :=
by
  -- Proof to be provided
  sorry

end perfect_score_l121_121592


namespace no_opposite_identical_numbers_l121_121590

open Finset

theorem no_opposite_identical_numbers : 
  ∀ (f g : Fin 20 → Fin 20), 
  (∀ i : Fin 20, ∃ j : Fin 20, f j = i ∧ g j = (i + j) % 20) → 
  ∃ k : ℤ, ∀ i : Fin 20, f (i + k) % 20 ≠ g i 
  := by
    sorry

end no_opposite_identical_numbers_l121_121590


namespace number_of_students_in_third_group_l121_121323

-- Definitions based on given conditions
def students_group1 : ℕ := 9
def students_group2 : ℕ := 10
def tissues_per_box : ℕ := 40
def total_tissues : ℕ := 1200

-- Define the number of students in the third group as a variable
variable {x : ℕ}

-- Prove that the number of students in the third group is 11
theorem number_of_students_in_third_group (h : 360 + 400 + 40 * x = 1200) : x = 11 :=
by sorry

end number_of_students_in_third_group_l121_121323


namespace triangle_acute_angles_integer_solution_l121_121887

theorem triangle_acute_angles_integer_solution :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), (20 < x ∧ x < 27) ∧ (12 < x ∧ x < 36) ↔ (x = 21 ∨ x = 22 ∨ x = 23 ∨ x = 24 ∨ x = 25 ∨ x = 26) :=
by
  sorry

end triangle_acute_angles_integer_solution_l121_121887


namespace bianca_birthday_money_l121_121226

-- Define the conditions
def num_friends : ℕ := 5
def money_per_friend : ℕ := 6

-- State the proof problem
theorem bianca_birthday_money : num_friends * money_per_friend = 30 :=
by
  sorry

end bianca_birthday_money_l121_121226


namespace minimal_height_exists_l121_121135

noncomputable def height_min_material (x : ℝ) : ℝ := 4 / (x^2)

theorem minimal_height_exists
  (x h : ℝ)
  (volume_cond : x^2 * h = 4)
  (surface_area_cond : h = height_min_material x) :
  h = 1 := by
  sorry

end minimal_height_exists_l121_121135


namespace powerThreeExpression_l121_121613

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l121_121613


namespace original_selling_price_l121_121103

theorem original_selling_price (P : ℝ) (h1 : ∀ P, 1.17 * P = 1.10 * P + 42) :
    1.10 * P = 660 := by
  sorry

end original_selling_price_l121_121103


namespace binders_can_bind_books_l121_121104

theorem binders_can_bind_books :
  (∀ (binders books days : ℕ), binders * days * books = 18 * 10 * 900 → 
    11 * binders * 12 = 660) :=
sorry

end binders_can_bind_books_l121_121104


namespace initial_capacity_of_bottle_l121_121920

theorem initial_capacity_of_bottle 
  (C : ℝ)
  (h1 : 1/3 * 3/4 * C = 1) : 
  C = 4 :=
by
  sorry

end initial_capacity_of_bottle_l121_121920


namespace cannot_achieve_61_cents_with_six_coins_l121_121675

theorem cannot_achieve_61_cents_with_six_coins :
  ¬ ∃ (p n d q : ℕ), 
      p + n + d + q = 6 ∧ 
      p + 5 * n + 10 * d + 25 * q = 61 :=
by
  sorry

end cannot_achieve_61_cents_with_six_coins_l121_121675


namespace problem1_problem2_l121_121202

theorem problem1 : (82 - 15) * (32 + 18) = 3350 :=
by
  sorry

theorem problem2 : (25 + 4) * 75 = 2175 :=
by
  sorry

end problem1_problem2_l121_121202


namespace increasing_log_condition_range_of_a_l121_121194

noncomputable def t (x a : ℝ) := x^2 - a*x + 3*a

theorem increasing_log_condition :
  (∀ x ≥ 2, 2 * x - a ≥ 0) ∧ a > -4 ∧ a ≤ 4 →
  ∀ x ≥ 2, x^2 - a*x + 3*a > 0 :=
by
  sorry

theorem range_of_a
  (h1 : ∀ x ≥ 2, 2 * x - a ≥ 0)
  (h2 : 4 - 2 * a + 3 * a > 0)
  (h3 : ∀ x ≥ 2, t x a > 0)
  : a > -4 ∧ a ≤ 4 :=
by
  sorry

end increasing_log_condition_range_of_a_l121_121194


namespace cut_square_into_rectangles_l121_121136

theorem cut_square_into_rectangles :
  ∃ x y : ℕ, 3 * x + 4 * y = 25 :=
by
  -- Given that the total area is 25 and we are using rectangles of areas 3 and 4
  -- we need to verify the existence of integers x and y such that 3x + 4y = 25
  existsi 7
  existsi 1
  sorry

end cut_square_into_rectangles_l121_121136


namespace allocation_methods_count_l121_121794

theorem allocation_methods_count :
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  ∃ (allocation_methods : ℕ), allocation_methods = 12 := 
by
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  use doctors * Nat.choose nurses 2
  sorry

end allocation_methods_count_l121_121794


namespace percent_decrease_apr_to_may_l121_121684

theorem percent_decrease_apr_to_may (P : ℝ) 
  (h1 : ∀ P : ℝ, P > 0 → (1.35 * P = P + 0.35 * P))
  (h2 : ∀ x : ℝ, P * (1.35 * (1 - x / 100) * 1.5) = 1.62000000000000014 * P)
  (h3 : 0 < x ∧ x < 100)
  : x = 20 :=
  sorry

end percent_decrease_apr_to_may_l121_121684


namespace choir_members_correct_l121_121404

noncomputable def choir_membership : ℕ :=
  let n := 226
  n

theorem choir_members_correct (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
by
  sorry

end choir_members_correct_l121_121404


namespace hulk_jump_exceeds_2000_l121_121981

theorem hulk_jump_exceeds_2000 {n : ℕ} (h : n ≥ 1) :
  2^(n - 1) > 2000 → n = 12 :=
by
  sorry

end hulk_jump_exceeds_2000_l121_121981


namespace find_s_l121_121299

theorem find_s : ∃ s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 :=
by
  sorry

end find_s_l121_121299


namespace function_has_zero_in_interval_l121_121174

   theorem function_has_zero_in_interval (fA fB fC fD : ℝ → ℝ) (hA : ∀ x, fA x = x - 3)
       (hB : ∀ x, fB x = 2^x) (hC : ∀ x, fC x = x^2) (hD : ∀ x, fD x = Real.log x) :
       ∃ x, 0 < x ∧ x < 2 ∧ fD x = 0 :=
   by
       sorry
   
end function_has_zero_in_interval_l121_121174


namespace x_coordinate_at_2005th_stop_l121_121263

theorem x_coordinate_at_2005th_stop :
 (∃ (f : ℕ → ℤ × ℤ),
    f 0 = (0, 0) ∧
    f 1 = (1, 0) ∧
    f 2 = (1, 1) ∧
    f 3 = (0, 1) ∧
    f 4 = (-1, 1) ∧
    f 5 = (-1, 0) ∧
    f 9 = (2, -1))
  → (∃ (f : ℕ → ℤ × ℤ), f 2005 = (3, -n)) := sorry

end x_coordinate_at_2005th_stop_l121_121263


namespace parabola_hyperbola_focus_l121_121980

theorem parabola_hyperbola_focus (p : ℝ) (hp : 0 < p) :
  (∃ k : ℝ, y^2 = 2 * k * x ∧ k > 0) ∧ (x^2 - y^2 / 3 = 1) → (p = 4) :=
by
  sorry

end parabola_hyperbola_focus_l121_121980


namespace value_of_fraction_l121_121769

open Real

theorem value_of_fraction (a : ℝ) (h : a^2 + a - 1 = 0) : (1 - a) / a + a / (1 + a) = 1 := 
by { sorry }

end value_of_fraction_l121_121769


namespace radius_of_smaller_molds_l121_121217

noncomputable def hemisphereVolume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  ∀ (R r : ℝ), R = 2 ∧ (64 * hemisphereVolume r) = hemisphereVolume R → r = 1 / 2 :=
by
  intros R r h
  sorry

end radius_of_smaller_molds_l121_121217


namespace polynomial_simplification_l121_121090

def A (x : ℝ) := 5 * x^2 + 4 * x - 1
def B (x : ℝ) := -x^2 - 3 * x + 3
def C (x : ℝ) := 8 - 7 * x - 6 * x^2

theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 :=
by
  simp [A, B, C]
  sorry

end polynomial_simplification_l121_121090


namespace length_of_notebook_is_24_l121_121857

-- Definitions
def span_of_hand : ℕ := 12
def length_of_notebook (span : ℕ) : ℕ := 2 * span

-- Theorem statement that proves the question == answer given conditions
theorem length_of_notebook_is_24 :
  length_of_notebook span_of_hand = 24 :=
sorry

end length_of_notebook_is_24_l121_121857


namespace math_problem_l121_121768

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l121_121768


namespace solve_quadratic_equation_l121_121877

theorem solve_quadratic_equation (m : ℝ) : 9 * m^2 - (2 * m + 1)^2 = 0 → m = 1 ∨ m = -1/5 :=
by
  intro h
  sorry

end solve_quadratic_equation_l121_121877


namespace find_x_l121_121012

theorem find_x : ∃ (x : ℝ), x > 0 ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_x_l121_121012


namespace line_positional_relationship_l121_121269

variables {Point Line Plane : Type}

-- Definitions of the conditions
def is_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
def is_within_plane (b : Line) (α : Plane) : Prop := sorry
def no_common_point (a b : Line) : Prop := sorry
def parallel_or_skew (a b : Line) : Prop := sorry

-- Proof statement in Lean
theorem line_positional_relationship
  (a b : Line) (α : Plane)
  (h₁ : is_parallel_to_plane a α)
  (h₂ : is_within_plane b α)
  (h₃ : no_common_point a b) :
  parallel_or_skew a b :=
sorry

end line_positional_relationship_l121_121269


namespace pentagon_edges_same_color_l121_121036

theorem pentagon_edges_same_color
  (A B : Fin 5 → Fin 5)
  (C : (Fin 5 → Fin 5) × (Fin 5 → Fin 5) → Bool)
  (condition : ∀ (i j : Fin 5), ∀ (k l m : Fin 5), (C (i, j) = C (k, l) → C (i, j) ≠ C (k, m))) :
  (∀ (x : Fin 5), C (A x, A ((x + 1) % 5)) = C (B x, B ((x + 1) % 5))) :=
by
sorry

end pentagon_edges_same_color_l121_121036


namespace maximum_unique_walks_l121_121407

-- Define the conditions
def starts_at_A : Prop := true
def crosses_bridge_1_first : Prop := true
def finishes_at_B : Prop := true
def six_bridges_linking_two_islands_and_banks : Prop := true

-- Define the theorem to prove the maximum number of unique walks is 6
theorem maximum_unique_walks : starts_at_A ∧ crosses_bridge_1_first ∧ finishes_at_B ∧ six_bridges_linking_two_islands_and_banks → ∃ n, n = 6 :=
by
  intros
  existsi 6
  sorry

end maximum_unique_walks_l121_121407


namespace a_10_value_l121_121123

-- Definitions for the initial conditions and recurrence relation.
def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧
  ∀ n, a (n + 1) = (8 / 5) * a n + (6 / 5) * (Real.sqrt (4 ^ n - a n ^ 2))

-- Statement that proves a_10 = 24576 / 25 given the conditions.
theorem a_10_value (a : ℕ → ℝ) (h : seq a) : a 10 = 24576 / 25 :=
by
  sorry

end a_10_value_l121_121123


namespace exists_k_tastrophic_function_l121_121469

noncomputable def k_tastrophic (f : ℕ+ → ℕ+) (k : ℕ) (n : ℕ+) : Prop :=
(f^[k] n) = n^k

theorem exists_k_tastrophic_function (k : ℕ) (h : k > 1) : ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, k_tastrophic f k n :=
by sorry

end exists_k_tastrophic_function_l121_121469


namespace plane_split_into_four_regions_l121_121841

theorem plane_split_into_four_regions {x y : ℝ} :
  (y = 3 * x) ∨ (y = (1 / 3) * x - (2 / 3)) →
  ∃ r : ℕ, r = 4 :=
by
  intro h
  -- We must show that these lines split the plane into 4 regions
  sorry

end plane_split_into_four_regions_l121_121841


namespace tensor_calculation_jiaqi_statement_l121_121931

def my_tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem tensor_calculation :
  my_tensor (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := 
by
  sorry

theorem jiaqi_statement (a b : ℝ) (h : a + b = 0) :
  my_tensor a a + my_tensor b b = 2 * a * b := 
by
  sorry

end tensor_calculation_jiaqi_statement_l121_121931


namespace total_children_l121_121445

-- Definitions for the conditions in the problem
def boys : ℕ := 19
def girls : ℕ := 41

-- Theorem stating the total number of children is 60
theorem total_children : boys + girls = 60 :=
by
  -- calculation done to show steps, but not necessary for the final statement
  sorry

end total_children_l121_121445


namespace solve_for_a_b_and_extrema_l121_121221

noncomputable def f (a b x : ℝ) := -2 * a * Real.sin (2 * x + (Real.pi / 6)) + 2 * a + b

theorem solve_for_a_b_and_extrema:
  ∃ (a b : ℝ), a > 0 ∧ 
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) ∧ 
  a = 2 ∧ b = -5 ∧
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 4),
    (f a b (Real.pi / 6) = -5 ∨ f a b 0 = -3)) :=
by
  sorry

end solve_for_a_b_and_extrema_l121_121221


namespace minimum_value_of_x_plus_y_l121_121936

-- Define the conditions as a hypothesis and the goal theorem statement.
theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 9 / y = 1) :
  x + y = 16 :=
by
  sorry

end minimum_value_of_x_plus_y_l121_121936


namespace max_cigarettes_with_staggered_packing_l121_121163

theorem max_cigarettes_with_staggered_packing :
  ∃ n : ℕ, n > 160 ∧ n = 176 :=
by
  let diameter := 2
  let rows_initial := 8
  let cols_initial := 20
  let total_initial := rows_initial * cols_initial
  have h1 : total_initial = 160 := by norm_num
  let alternative_packing_capacity := 176
  have h2 : alternative_packing_capacity > total_initial := by norm_num
  use alternative_packing_capacity
  exact ⟨h2, rfl⟩

end max_cigarettes_with_staggered_packing_l121_121163


namespace line_through_two_quadrants_l121_121384

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l121_121384


namespace eval_expression_l121_121738

theorem eval_expression : -30 + 12 * (8 / 4)^2 = 18 :=
by
  sorry

end eval_expression_l121_121738


namespace geom_arith_seq_l121_121068

theorem geom_arith_seq (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_arith : 2 * a 3 - (a 5 / 2) = (a 5 / 2) - 3 * a 1) (hq : q > 0) :
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 :=
by
  sorry

end geom_arith_seq_l121_121068


namespace geese_percentage_l121_121786

noncomputable def percentage_of_geese_among_non_swans (geese swans herons ducks : ℝ) : ℝ :=
  (geese / (100 - swans)) * 100

theorem geese_percentage (geese swans herons ducks : ℝ)
  (h1 : geese = 40)
  (h2 : swans = 20)
  (h3 : herons = 15)
  (h4 : ducks = 25) :
  percentage_of_geese_among_non_swans geese swans herons ducks = 50 :=
by
  simp [percentage_of_geese_among_non_swans, h1, h2, h3, h4]
  sorry

end geese_percentage_l121_121786


namespace tank_capacity_l121_121076

-- Definitions from conditions
def initial_fraction := (1 : ℚ) / 4  -- The tank is 1/4 full initially
def added_amount := 5  -- Adding 5 liters

-- The proof problem to show that the tank's total capacity c equals 60 liters
theorem tank_capacity
  (c : ℚ)  -- The total capacity of the tank in liters
  (h1 : c / 4 + added_amount = c / 3)  -- Adding 5 liters makes the tank 1/3 full
  : c = 60 := 
sorry

end tank_capacity_l121_121076


namespace exists_a_max_value_of_four_l121_121773

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * a * Real.sin x + 3 * a - 1

theorem exists_a_max_value_of_four :
  ∃ a : ℝ, (a = 1) ∧ ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f a x ≤ 4 := 
sorry

end exists_a_max_value_of_four_l121_121773


namespace arithmetic_sequence_check_l121_121101

theorem arithmetic_sequence_check 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h : ∀ n : ℕ, a (n+1) = a n + d) 
  : (∀ n : ℕ, (a n + 1) - (a (n - 1) + 1) = d) 
    ∧ (∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 2 * d)
    ∧ (∀ n : ℕ, a (n + 1) - (a n + n) = d + 1) := 
by
  sorry

end arithmetic_sequence_check_l121_121101


namespace greatest_b_value_l121_121051

theorem greatest_b_value (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 15 ≠ -9) ↔ b = 9 :=
sorry

end greatest_b_value_l121_121051


namespace intersection_line_eq_l121_121537

-- Definitions of the circles
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*y - 6 = 0

-- The theorem stating that the equation of the line passing through their intersection points is x = y
theorem intersection_line_eq (x y : ℝ) :
  (circle1 x y → circle2 x y → x = y) := 
by
  intro h1 h2
  sorry

end intersection_line_eq_l121_121537


namespace overall_gain_loss_percent_zero_l121_121447

theorem overall_gain_loss_percent_zero (CP_A CP_B CP_C SP_A SP_B SP_C : ℝ)
  (h1 : CP_A = 600) (h2 : CP_B = 700) (h3 : CP_C = 800)
  (h4 : SP_A = 450) (h5 : SP_B = 750) (h6 : SP_C = 900) :
  ((SP_A + SP_B + SP_C) - (CP_A + CP_B + CP_C)) / (CP_A + CP_B + CP_C) * 100 = 0 :=
by
  sorry

end overall_gain_loss_percent_zero_l121_121447


namespace mark_leftover_amount_l121_121171

-- Definitions
def raise_percentage : ℝ := 0.05
def old_hourly_wage : ℝ := 40
def hours_per_week : ℝ := 8 * 5
def old_weekly_expenses : ℝ := 600
def new_expense : ℝ := 100

-- Calculate new hourly wage
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)

-- Calculate weekly earnings at the new wage
def weekly_earnings : ℝ := new_hourly_wage * hours_per_week

-- Calculate new total weekly expenses
def total_weekly_expenses : ℝ := old_weekly_expenses + new_expense

-- Calculate leftover amount
def leftover_per_week : ℝ := weekly_earnings - total_weekly_expenses

theorem mark_leftover_amount : leftover_per_week = 980 := by
  sorry

end mark_leftover_amount_l121_121171


namespace negation_statement_l121_121821

variables {a b c : ℝ}

theorem negation_statement (h : a * b * c = 0) : ¬(a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end negation_statement_l121_121821


namespace slope_angle_line_l121_121883
open Real

theorem slope_angle_line (x y : ℝ) :
  x + sqrt 3 * y - 1 = 0 → ∃ θ : ℝ, θ = 150 ∧
  ∃ (m : ℝ), m = -sqrt 3 / 3 ∧ θ = arctan m :=
by
  sorry

end slope_angle_line_l121_121883


namespace smallest_n_square_partition_l121_121149

theorem smallest_n_square_partition (n : ℕ) (h : ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧ n = 40 * a + 49 * b) : n ≥ 2000 :=
by sorry

end smallest_n_square_partition_l121_121149


namespace frustum_volume_correct_l121_121604

-- Define the base edge of the original pyramid
def base_edge_pyramid := 16

-- Define the height (altitude) of the original pyramid
def height_pyramid := 10

-- Define the base edge of the smaller pyramid after the cut
def base_edge_smaller_pyramid := 8

-- Define the function to calculate the volume of a square pyramid
def volume_square_pyramid (base_edge : ℕ) (height : ℕ) : ℚ :=
  (1 / 3) * (base_edge ^ 2) * height

-- Calculate the volume of the original pyramid
def V := volume_square_pyramid base_edge_pyramid height_pyramid

-- Calculate the volume of the smaller pyramid
def V_small := volume_square_pyramid base_edge_smaller_pyramid (height_pyramid / 2)

-- Calculate the volume of the frustum
def V_frustum := V - V_small

-- Prove that the volume of the frustum is 213.33 cubic centimeters
theorem frustum_volume_correct : V_frustum = 213.33 := by
  sorry

end frustum_volume_correct_l121_121604


namespace problem_A_problem_B_problem_C_problem_D_l121_121784

theorem problem_A : 2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5 := by
  sorry

theorem problem_B : 3 * Real.sqrt 3 * (3 * Real.sqrt 2) ≠ 3 * Real.sqrt 6 := by
  sorry

theorem problem_C : (Real.sqrt 27 / Real.sqrt 3) = 3 := by
  sorry

theorem problem_D : 2 * Real.sqrt 2 - Real.sqrt 2 ≠ 2 := by
  sorry

end problem_A_problem_B_problem_C_problem_D_l121_121784


namespace flower_shop_percentage_l121_121017

theorem flower_shop_percentage (C : ℕ) : 
  let V := (1/3 : ℝ) * C
  let T := (1/12 : ℝ) * C
  let R := T
  let total := C + V + T + R
  (C / total) * 100 = 66.67 := 
by
  sorry

end flower_shop_percentage_l121_121017


namespace rachel_age_when_emily_half_age_l121_121109

-- Conditions
def Emily_current_age : ℕ := 20
def Rachel_current_age : ℕ := 24

-- Proof statement
theorem rachel_age_when_emily_half_age :
  ∃ x : ℕ, (Emily_current_age - x = (Rachel_current_age - x) / 2) ∧ (Rachel_current_age - x = 8) := 
sorry

end rachel_age_when_emily_half_age_l121_121109


namespace children_got_off_bus_l121_121671

theorem children_got_off_bus :
  ∀ (initial_children final_children new_children off_children : ℕ),
    initial_children = 21 → final_children = 16 → new_children = 5 →
    initial_children - off_children + new_children = final_children →
    off_children = 10 :=
by
  intro initial_children final_children new_children off_children
  intros h_init h_final h_new h_eq
  sorry

end children_got_off_bus_l121_121671


namespace twenty_four_point_solution_l121_121601

theorem twenty_four_point_solution : (5 - (1 / 5)) * 5 = 24 := 
by 
  sorry

end twenty_four_point_solution_l121_121601


namespace sine_angle_greater_implies_angle_greater_l121_121802

noncomputable def triangle := {ABC : Type* // Π A B C : ℕ, 
  A + B + C = 180 ∧ 0 < A ∧ A < 180 ∧ 0 < B ∧ B < 180 ∧ 0 < C ∧ C < 180}

variables {A B C : ℕ} (T : triangle)

theorem sine_angle_greater_implies_angle_greater (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180)
  (h3 : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) (h_sine : Real.sin A > Real.sin B) :
  A > B := 
sorry

end sine_angle_greater_implies_angle_greater_l121_121802


namespace max_chords_l121_121209

noncomputable def max_closed_chords (n : ℕ) (h : n ≥ 3) : ℕ :=
  n

/-- Given an integer number n ≥ 3 and n distinct points on a circle, labeled 1 through n,
prove that the maximum number of closed chords [ij], i ≠ j, having pairwise non-empty intersections is n. -/
theorem max_chords {n : ℕ} (h : n ≥ 3) :
  max_closed_chords n h = n := 
sorry

end max_chords_l121_121209


namespace find_real_number_x_l121_121343

theorem find_real_number_x 
    (x : ℝ) 
    (i : ℂ) 
    (h_imaginary_unit : i*i = -1) 
    (h_equation : (1 - 2*i)*(x + i) = 4 - 3*i) : 
    x = 2 := 
by
  sorry

end find_real_number_x_l121_121343


namespace emma_reaches_jack_after_33_minutes_l121_121702

-- Definitions from conditions
def distance_initial : ℝ := 30  -- 30 km apart initially
def combined_speed : ℝ := 2     -- combined speed is 2 km/min
def time_before_breakdown : ℝ := 6 -- Jack biked for 6 minutes before breaking down

-- Assume speeds
def v_J (v_E : ℝ) : ℝ := 2 * v_E  -- Jack's speed is twice Emma's speed

-- Assertion to prove
theorem emma_reaches_jack_after_33_minutes :
  ∀ v_E : ℝ, ((v_J v_E + v_E = combined_speed) → 
              (distance_initial - combined_speed * time_before_breakdown = 18) → 
              (v_E > 0) → 
              (time_before_breakdown + 18 / v_E = 33)) :=
by 
  intro v_E 
  intros h1 h2 h3 
  have h4 : v_J v_E = 2 * v_E := rfl
  sorry

end emma_reaches_jack_after_33_minutes_l121_121702


namespace evaluate_expression_l121_121215

theorem evaluate_expression : (164^2 - 148^2) / 16 = 312 := 
by 
  sorry

end evaluate_expression_l121_121215


namespace find_a_l121_121164

theorem find_a (a : ℝ) :
  {x : ℝ | (x + a) / ((x + 1) * (x + 3)) > 0} = {x : ℝ | x > -3 ∧ x ≠ -1} →
  a = 1 := 
by sorry

end find_a_l121_121164


namespace sheets_in_total_l121_121231

theorem sheets_in_total (boxes_needed : ℕ) (sheets_per_box : ℕ) (total_sheets : ℕ) 
  (h1 : boxes_needed = 7) (h2 : sheets_per_box = 100) : total_sheets = boxes_needed * sheets_per_box := by
  sorry

end sheets_in_total_l121_121231


namespace max_value_of_x_minus_y_l121_121489

theorem max_value_of_x_minus_y
  (x y : ℝ)
  (h : 2 * (x ^ 2 + y ^ 2 - x * y) = x + y) :
  x - y ≤ 1 / 2 := 
sorry

end max_value_of_x_minus_y_l121_121489


namespace radius_of_circumscribed_circle_l121_121596

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end radius_of_circumscribed_circle_l121_121596


namespace expression_value_l121_121059

theorem expression_value : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := 
by 
  sorry

end expression_value_l121_121059


namespace lucas_seq_units_digit_M47_l121_121096

def lucas_seq : ℕ → ℕ := 
  sorry -- skipped sequence generation for brevity

def M (n : ℕ) : ℕ :=
  if n = 0 then 3 else
  if n = 1 then 1 else
  lucas_seq n -- will call the lucas sequence generator

-- Helper function to get the units digit of a number
def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem lucas_seq_units_digit_M47 : units_digit (M (M 6)) = 3 := 
sorry

end lucas_seq_units_digit_M47_l121_121096


namespace total_games_attended_l121_121633

def games_in_months (this_month previous_month next_month following_month fifth_month : ℕ) : ℕ :=
  this_month + previous_month + next_month + following_month + fifth_month

theorem total_games_attended :
  games_in_months 24 32 29 19 34 = 138 :=
by
  -- Proof will be provided, but ignored for this problem
  sorry

end total_games_attended_l121_121633


namespace smallest_b_for_fraction_eq_l121_121497

theorem smallest_b_for_fraction_eq (a b : ℕ) (h1 : 1000 ≤ a ∧ a < 10000) (h2 : 100000 ≤ b ∧ b < 1000000)
(h3 : 1/2006 = 1/a + 1/b) : b = 120360 := sorry

end smallest_b_for_fraction_eq_l121_121497


namespace Amy_initial_cupcakes_l121_121288

def initialCupcakes (packages : ℕ) (cupcakesPerPackage : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakesPerPackage + eaten

theorem Amy_initial_cupcakes :
  let packages := 9
  let cupcakesPerPackage := 5
  let eaten := 5
  initialCupcakes packages cupcakesPerPackage eaten = 50 :=
by
  sorry

end Amy_initial_cupcakes_l121_121288


namespace penny_purchase_exceeded_minimum_spend_l121_121578

theorem penny_purchase_exceeded_minimum_spend :
  let bulk_price_per_pound := 5
  let minimum_spend := 40
  let tax_per_pound := 1
  let total_paid := 240
  let total_cost_per_pound := bulk_price_per_pound + tax_per_pound
  let pounds_purchased := total_paid / total_cost_per_pound
  let minimum_pounds_to_spend := minimum_spend / bulk_price_per_pound
  pounds_purchased - minimum_pounds_to_spend = 32 :=
by
  -- The proof is omitted here as per the instructions.
  sorry

end penny_purchase_exceeded_minimum_spend_l121_121578


namespace geometric_sequence_a7_l121_121939

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

-- Given condition
axiom geom_seq_condition : a 4 * a 10 = 9

-- proving the required result
theorem geometric_sequence_a7 (h : is_geometric_sequence a r) : a 7 = 3 ∨ a 7 = -3 :=
by
  sorry

end geometric_sequence_a7_l121_121939


namespace polygon_sides_eight_l121_121890

theorem polygon_sides_eight (n : ℕ) (h : 180 * (n - 2) = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l121_121890


namespace sum_of_remainders_l121_121142

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9) = 3 :=
by
  sorry

end sum_of_remainders_l121_121142


namespace max_gold_coins_l121_121219

theorem max_gold_coins (n : ℕ) (k : ℕ) (H1 : n = 13 * k + 3) (H2 : n < 150) : n ≤ 146 := 
by
  sorry

end max_gold_coins_l121_121219


namespace find_y_value_l121_121185

theorem find_y_value (k : ℝ) (x y : ℝ) (h1 : y = k * x^(1/5)) (h2 : y = 4) (h3 : x = 32) :
  y = 6 := by
  sorry

end find_y_value_l121_121185


namespace road_length_10_trees_10_intervals_l121_121260

theorem road_length_10_trees_10_intervals 
  (n_trees : ℕ) (n_intervals : ℕ) (tree_interval : ℕ) 
  (h_trees : n_trees = 10) (h_intervals : n_intervals = 9) (h_interval_length : tree_interval = 10) : 
  n_intervals * tree_interval = 90 := 
by 
  sorry

end road_length_10_trees_10_intervals_l121_121260


namespace grade_above_B_l121_121767

theorem grade_above_B (total_students : ℕ) (percentage_below_B : ℕ) (students_above_B : ℕ) :
  total_students = 60 ∧ percentage_below_B = 40 ∧ students_above_B = total_students * (100 - percentage_below_B) / 100 →
  students_above_B = 36 :=
by
  sorry

end grade_above_B_l121_121767


namespace aunt_money_calculation_l121_121808

variable (total_money_received aunt_money : ℕ)
variable (bank_amount grandfather_money : ℕ := 150)

theorem aunt_money_calculation (h1 : bank_amount = 45) (h2 : bank_amount = total_money_received / 5) (h3 : total_money_received = aunt_money + grandfather_money) :
  aunt_money = 75 :=
by
  -- The proof is captured in these statements:
  sorry

end aunt_money_calculation_l121_121808


namespace increase_80_by_150_percent_l121_121731

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l121_121731


namespace multiple_of_remainder_l121_121663

theorem multiple_of_remainder (R V D Q k : ℤ) (h1 : R = 6) (h2 : V = 86) (h3 : D = 5 * Q) 
  (h4 : D = k * R + 2) (h5 : V = D * Q + R) : k = 3 := by
  sorry

end multiple_of_remainder_l121_121663


namespace B_A_equals_expectedBA_l121_121154

noncomputable def MatrixA : Matrix (Fin 2) (Fin 2) ℝ := sorry
noncomputable def MatrixB : Matrix (Fin 2) (Fin 2) ℝ := sorry
def MatrixAB : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 1], ![-2, 4]]
def expectedBA : Matrix (Fin 2) (Fin 2) ℝ := ![![10, 2], ![-4, 8]]

theorem B_A_equals_expectedBA (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = MatrixAB) : 
  B * A = expectedBA := by
  sorry

end B_A_equals_expectedBA_l121_121154


namespace log_geometric_sequence_l121_121187

theorem log_geometric_sequence :
  ∀ (a : ℕ → ℝ), (∀ n, 0 < a n) → (∃ r : ℝ, ∀ n, a (n + 1) = a n * r) →
  a 2 * a 18 = 16 → Real.logb 2 (a 10) = 2 :=
by
  intros a h_positive h_geometric h_condition
  sorry

end log_geometric_sequence_l121_121187


namespace temperature_difference_l121_121121

def h : ℤ := 10
def l : ℤ := -5
def d : ℤ := 15

theorem temperature_difference : h - l = d :=
by
  rw [h, l, d]
  sorry

end temperature_difference_l121_121121


namespace cars_travel_same_distance_l121_121916

-- Define all the variables and conditions
def TimeR : ℝ := sorry -- the time taken by car R
def TimeP : ℝ := TimeR - 2
def SpeedR : ℝ := 58.4428877022476
def SpeedP : ℝ := SpeedR + 10

-- state the distance travelled by both cars
def DistanceR : ℝ := SpeedR * TimeR
def DistanceP : ℝ := SpeedP * TimeP

-- Prove that both distances are the same and equal to 800
theorem cars_travel_same_distance : DistanceR = 800 := by
  sorry

end cars_travel_same_distance_l121_121916


namespace digits_difference_l121_121814

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l121_121814


namespace find_divisor_l121_121568

theorem find_divisor (n x y z a b c : ℕ) (h1 : 63 = n * x + a) (h2 : 91 = n * y + b) (h3 : 130 = n * z + c) (h4 : a + b + c = 26) : n = 43 :=
sorry

end find_divisor_l121_121568


namespace mistaken_divisor_is_12_l121_121074

theorem mistaken_divisor_is_12 (dividend : ℕ) (mistaken_divisor : ℕ) (correct_divisor : ℕ) 
  (mistaken_quotient : ℕ) (correct_quotient : ℕ) (remainder : ℕ) :
  remainder = 0 ∧ correct_divisor = 21 ∧ mistaken_quotient = 42 ∧ correct_quotient = 24 ∧ 
  dividend = mistaken_quotient * mistaken_divisor ∧ dividend = correct_quotient * correct_divisor →
  mistaken_divisor = 12 :=
by 
  sorry

end mistaken_divisor_is_12_l121_121074


namespace max_distance_from_circle_to_line_l121_121649

theorem max_distance_from_circle_to_line :
  ∀ (P : ℝ × ℝ), (P.1 - 1)^2 + P.2^2 = 9 →
  ∀ (x y : ℝ), 5 * x + 12 * y + 8 = 0 →
  ∃ (d : ℝ), d = 4 :=
by
  -- Proof is omitted as instructed.
  sorry

end max_distance_from_circle_to_line_l121_121649


namespace pencil_distribution_l121_121691

theorem pencil_distribution (n : ℕ) (friends : ℕ): 
  (friends = 4) → (n = 8) → 
  (∃ A B C D : ℕ, A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 1 ∧ D ≥ 1 ∧ A + B + C + D = n) →
  (∃! k : ℕ, k = 20) :=
by
  intros friends_eq n_eq h
  use 20
  sorry

end pencil_distribution_l121_121691


namespace heather_lighter_than_combined_weights_l121_121830

noncomputable def heather_weight : ℝ := 87.5
noncomputable def emily_weight : ℝ := 45.3
noncomputable def elizabeth_weight : ℝ := 38.7
noncomputable def george_weight : ℝ := 56.9

theorem heather_lighter_than_combined_weights :
  heather_weight - (emily_weight + elizabeth_weight + george_weight) = -53.4 :=
by 
  sorry

end heather_lighter_than_combined_weights_l121_121830


namespace sufficient_but_not_necessary_condition_l121_121383

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 2*x < 0) → (|x - 2| < 2) ∧ ¬(|x - 2| < 2) → (x^2 - 2*x < 0 ↔ |x-2| < 2) :=
sorry

end sufficient_but_not_necessary_condition_l121_121383


namespace max_k_value_l121_121749

theorem max_k_value :
  ∃ A B C k : ℕ, 
  (A ≠ 0) ∧ 
  (A < 10) ∧ 
  (B < 10) ∧ 
  (C < 10) ∧
  (10 * A + B) * k = 100 * A + 10 * C + B ∧
  (∀ k' : ℕ, 
     ((A ≠ 0) ∧ (A < 10) ∧ (B < 10) ∧ (C < 10) ∧
     (10 * A + B) * k' = 100 * A + 10 * C + B) 
     → k' ≤ 19) ∧
  k = 19 :=
sorry

end max_k_value_l121_121749


namespace sprint_time_l121_121214

def speed (Mark : Type) : ℝ := 6.0
def distance (Mark : Type) : ℝ := 144.0

theorem sprint_time (Mark : Type) : (distance Mark) / (speed Mark) = 24 := by
  sorry

end sprint_time_l121_121214


namespace angle_measure_l121_121223

theorem angle_measure (x : ℝ) (h1 : (180 - x) = 3*x - 2) : x = 45.5 :=
by
  sorry

end angle_measure_l121_121223


namespace erin_trolls_count_l121_121092

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l121_121092


namespace sticker_price_l121_121160

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end sticker_price_l121_121160


namespace jessica_attended_games_l121_121692

/-- 
Let total_games be the total number of soccer games.
Let initially_planned be the number of games Jessica initially planned to attend.
Let commitments_skipped be the number of games skipped due to other commitments.
Let rescheduled_games be the rescheduled games during the season.
Let additional_missed be the additional games missed due to rescheduling.
-/
theorem jessica_attended_games
    (total_games initially_planned commitments_skipped rescheduled_games additional_missed : ℕ)
    (h1 : total_games = 12)
    (h2 : initially_planned = 8)
    (h3 : commitments_skipped = 3)
    (h4 : rescheduled_games = 2)
    (h5 : additional_missed = 4) :
    (initially_planned - commitments_skipped) - additional_missed = 1 := by
  sorry

end jessica_attended_games_l121_121692


namespace joan_exam_time_difference_l121_121597

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l121_121597


namespace profit_per_box_type_A_and_B_maximize_profit_l121_121655

-- Condition definitions
def total_boxes : ℕ := 600
def profit_type_A : ℕ := 40000
def profit_type_B : ℕ := 160000
def profit_difference : ℕ := 200

-- Question 1: Proving the profit per box for type A and B
theorem profit_per_box_type_A_and_B (x : ℝ) :
  (profit_type_A / x + profit_type_B / (x + profit_difference) = total_boxes)
  → (x = 200) ∧ (x + profit_difference = 400) :=
sorry

-- Condition definitions for question 2
def price_reduction_per_box_A (a : ℕ) : ℕ := 5 * a
def price_increase_per_box_B (a : ℕ) : ℕ := 5 * a

-- Initial number of boxes sold for type A and B
def initial_boxes_sold_A : ℕ := 200
def initial_boxes_sold_B : ℕ := 400

-- General profit function
def profit (a : ℕ) : ℝ :=
  (initial_boxes_sold_A + 2 * a) * (200 - price_reduction_per_box_A a) +
  (initial_boxes_sold_B - 2 * a) * (400 + price_increase_per_box_B a)

-- Question 2: Proving the price reduction and maximum profit
theorem maximize_profit (a : ℕ) :
  ((price_reduction_per_box_A a = 75) ∧ (profit a = 204500)) :=
sorry

end profit_per_box_type_A_and_B_maximize_profit_l121_121655


namespace cat_food_inequality_l121_121356

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l121_121356


namespace curve_touches_all_Ca_l121_121922

theorem curve_touches_all_Ca (a : ℝ) (h : a > 0) : ∃ C : ℝ → ℝ, ∀ x y, (y - a^2)^2 = x^2 * (a^2 - x^2) → y = C x ∧ C x = 3 * x^2 / 4 :=
sorry

end curve_touches_all_Ca_l121_121922


namespace find_first_term_l121_121781

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l121_121781


namespace erica_riding_time_is_65_l121_121766

-- Definition of Dave's riding time
def dave_time : ℕ := 10

-- Definition of Chuck's riding time based on Dave's time
def chuck_time (dave_time : ℕ) : ℕ := 5 * dave_time

-- Definition of Erica's additional riding time calculated as 30% of Chuck's time
def erica_additional_time (chuck_time : ℕ) : ℕ := (30 * chuck_time) / 100

-- Definition of Erica's total riding time as Chuck's time plus her additional time
def erica_total_time (chuck_time : ℕ) (erica_additional_time : ℕ) : ℕ := chuck_time + erica_additional_time

-- The proof problem: Erica's total riding time should be 65 minutes.
theorem erica_riding_time_is_65 : erica_total_time (chuck_time dave_time) (erica_additional_time (chuck_time dave_time)) = 65 :=
by
  -- The proof is skipped here
  sorry

end erica_riding_time_is_65_l121_121766


namespace original_price_of_book_l121_121245

-- Define the conditions as Lean 4 statements
variable (P : ℝ)  -- Original price of the book
variable (P_new : ℝ := 480)  -- New price of the book
variable (increase_percentage : ℝ := 0.60)  -- Percentage increase in the price

-- Prove the question: original price equals to $300
theorem original_price_of_book :
  P + increase_percentage * P = P_new → P = 300 :=
by
  sorry

end original_price_of_book_l121_121245


namespace greatest_n_l121_121983

def S := { xy : ℕ × ℕ | ∃ x y : ℕ, xy = (x * y, x + y) }

def in_S (a : ℕ) : Prop := ∃ x y : ℕ, a = x * y * (x + y)

def pow_mod (a b m : ℕ) : ℕ := (a ^ b) % m

def satisfies_condition (a : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → in_S (a + pow_mod 2 k 9)

theorem greatest_n (a : ℕ) (n : ℕ) : 
  satisfies_condition a n → n ≤ 3 :=
sorry

end greatest_n_l121_121983


namespace find_a5_l121_121243

-- Definitions related to the conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a5 (a : ℕ → ℕ) (h_arith : arithmetic_sequence a) (h_a3 : a 3 = 3)
  (h_geo : geometric_sequence (a 1) (a 2) (a 4)) :
  a 5 = 5 ∨ a 5 = 3 :=
  sorry

end find_a5_l121_121243


namespace number_minus_29_l121_121571

theorem number_minus_29 (x : ℕ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end number_minus_29_l121_121571


namespace trigonometric_expression_identity_l121_121940

open Real

theorem trigonometric_expression_identity :
  (1 - 1 / cos (35 * (pi / 180))) * 
  (1 + 1 / sin (55 * (pi / 180))) * 
  (1 - 1 / sin (35 * (pi / 180))) * 
  (1 + 1 / cos (55 * (pi / 180))) = 1 := by
  sorry

end trigonometric_expression_identity_l121_121940


namespace total_people_bought_tickets_l121_121787

-- Definitions based on the conditions from step a)
def num_adults := 375
def num_children := 3 * num_adults
def total_revenue := 7 * num_adults + 3 * num_children

-- Statement of the theorem based on the question in step a)
theorem total_people_bought_tickets : (num_adults + num_children) = 1500 :=
by
  -- The proof is omitted, but we're ensuring the correctness of the theorem statement.
  sorry

end total_people_bought_tickets_l121_121787


namespace length_AB_is_4_l121_121876

section HyperbolaProof

/-- Define the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 8) = 1

/-- Define the line l given by x = 2√6 -/
def line_l (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 6

/-- Define the condition for intersection points -/
def intersect_points (x y : ℝ) : Prop :=
  hyperbola x y ∧ line_l x

/-- Prove the length of the line segment AB is 4 -/
theorem length_AB_is_4 :
  ∀ y : ℝ, intersect_points (2 * Real.sqrt 6) y → |y| = 2 → length_AB = 4 :=
sorry

end HyperbolaProof

end length_AB_is_4_l121_121876


namespace find_number_l121_121856

theorem find_number (x : ℕ) (h1 : x - 13 = 31) : x + 11 = 55 :=
  sorry

end find_number_l121_121856


namespace graph_comparison_l121_121398

theorem graph_comparison :
  (∀ x : ℝ, (x^2 - x + 3) < (x^2 - x + 5)) :=
by
  sorry

end graph_comparison_l121_121398


namespace smallest_sum_arith_geo_sequence_l121_121585

theorem smallest_sum_arith_geo_sequence 
  (A B C D: ℕ) 
  (h1: A > 0) 
  (h2: B > 0) 
  (h3: C > 0) 
  (h4: D > 0)
  (h5: 2 * B = A + C)
  (h6: B * D = C * C)
  (h7: 3 * C = 4 * B) : 
  A + B + C + D = 43 := 
sorry

end smallest_sum_arith_geo_sequence_l121_121585


namespace Jennifer_future_age_Jordana_future_age_Jordana_current_age_l121_121066

variable (Jennifer_age_now Jordana_age_now : ℕ)

-- Conditions
def age_in_ten_years (current_age : ℕ) : ℕ := current_age + 10
theorem Jennifer_future_age : age_in_ten_years Jennifer_age_now = 30 := sorry
theorem Jordana_future_age : age_in_ten_years Jordana_age_now = 3 * age_in_ten_years Jennifer_age_now := sorry

-- Question to prove
theorem Jordana_current_age : Jordana_age_now = 80 := sorry

end Jennifer_future_age_Jordana_future_age_Jordana_current_age_l121_121066


namespace sin2theta_cos2theta_sum_l121_121495

theorem sin2theta_cos2theta_sum (θ : ℝ) (h1 : Real.sin θ = 2 * Real.cos θ) (h2 : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_cos2theta_sum_l121_121495


namespace apple_cost_is_2_l121_121116

def total_spent (hummus_cost chicken_cost bacon_cost vegetable_cost : ℕ) : ℕ :=
  2 * hummus_cost + chicken_cost + bacon_cost + vegetable_cost

theorem apple_cost_is_2 :
  ∀ (hummus_cost chicken_cost bacon_cost vegetable_cost total_money apples_cost : ℕ),
    hummus_cost = 5 →
    chicken_cost = 20 →
    bacon_cost = 10 →
    vegetable_cost = 10 →
    total_money = 60 →
    apples_cost = 5 →
    (total_money - total_spent hummus_cost chicken_cost bacon_cost vegetable_cost) / apples_cost = 2 :=
by
  intros
  sorry

end apple_cost_is_2_l121_121116


namespace photo_arrangements_l121_121244

-- The description of the problem conditions translated into definitions
def num_positions := 6  -- Total positions (1 teacher + 5 students)

def teacher_positions := 4  -- Positions where teacher can stand (not at either end)

def student_permutations : ℕ := Nat.factorial 5  -- Number of ways to arrange 5 students

-- The total number of valid arrangements where the teacher does not stand at either end
def total_valid_arrangements : ℕ := teacher_positions * student_permutations

-- Statement to be proven
theorem photo_arrangements:
  total_valid_arrangements = 480 :=
by
  sorry

end photo_arrangements_l121_121244


namespace Rickey_took_30_minutes_l121_121262

variables (R P : ℝ)

-- Define the conditions
def Prejean_speed_is_three_quarters_of_Rickey := P = 4 / 3 * R
def total_time_is_70 := R + P = 70

-- Define the statement to prove
theorem Rickey_took_30_minutes 
  (h1 : Prejean_speed_is_three_quarters_of_Rickey R P) 
  (h2 : total_time_is_70 R P) : R = 30 :=
by
  sorry

end Rickey_took_30_minutes_l121_121262


namespace min_value_expression_l121_121627

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 10 + 6 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (a + 2 * y = 1) → ( (y^2 + a + 1) / (a * y)  ≥  c )) :=
sorry

end min_value_expression_l121_121627


namespace correct_negation_of_p_l121_121459

open Real

def proposition_p (x : ℝ) := x > 0 → sin x ≥ -1

theorem correct_negation_of_p :
  ¬ (∀ x, proposition_p x) ↔ (∃ x, x > 0 ∧ sin x < -1) :=
by
  sorry

end correct_negation_of_p_l121_121459


namespace diagonal_of_rectangular_prism_l121_121616

theorem diagonal_of_rectangular_prism
  (width height depth : ℕ)
  (h1 : width = 15)
  (h2 : height = 20)
  (h3 : depth = 25) : 
  (width ^ 2 + height ^ 2 + depth ^ 2).sqrt = 25 * (2 : ℕ).sqrt :=
by {
  sorry
}

end diagonal_of_rectangular_prism_l121_121616


namespace career_preference_representation_l121_121562

noncomputable def male_to_female_ratio : ℕ × ℕ := (2, 3)
noncomputable def total_students := male_to_female_ratio.1 + male_to_female_ratio.2
noncomputable def students_prefer_career := 2
noncomputable def full_circle_degrees := 360

theorem career_preference_representation :
  (students_prefer_career / total_students : ℚ) * full_circle_degrees = 144 := by
  sorry

end career_preference_representation_l121_121562


namespace sum_of_opposite_numbers_is_zero_l121_121759

theorem sum_of_opposite_numbers_is_zero {a b : ℝ} (h : a + b = 0) : a + b = 0 := 
h

end sum_of_opposite_numbers_is_zero_l121_121759


namespace simplify_expression_l121_121210

noncomputable def simplify_fraction (x : ℝ) (h : x ≠ 2) : ℝ :=
  (1 + (1 / (x - 2))) / ((x - x^2) / (x - 2))

theorem simplify_expression (x : ℝ) (h : x ≠ 2) : simplify_fraction x h = -(x - 1) / x :=
  sorry

end simplify_expression_l121_121210


namespace find_scalars_l121_121484

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2;
    3, 1]

noncomputable def B4 : Matrix (Fin 2) (Fin 2) ℝ :=
  B * B * B * B

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_scalars (r s : ℝ) (hB : B^4 = r • B + s • I) :
  (r, s) = (51, 52) :=
  sorry

end find_scalars_l121_121484


namespace lcm_24_36_40_l121_121394

-- Define the natural numbers 24, 36, and 40
def n1 : ℕ := 24
def n2 : ℕ := 36
def n3 : ℕ := 40

-- Define the prime factorization of each number
def factors_n1 := [2^3, 3^1] -- 24 = 2^3 * 3^1
def factors_n2 := [2^2, 3^2] -- 36 = 2^2 * 3^2
def factors_n3 := [2^3, 5^1] -- 40 = 2^3 * 5^1

-- Prove that the LCM of n1, n2, n3 is 360
theorem lcm_24_36_40 : Nat.lcm (Nat.lcm n1 n2) n3 = 360 := sorry

end lcm_24_36_40_l121_121394


namespace molecular_weight_of_one_mole_l121_121360

theorem molecular_weight_of_one_mole 
  (molicular_weight_9_moles : ℕ) 
  (weight_9_moles : ℕ)
  (h : molicular_weight_9_moles = 972 ∧ weight_9_moles = 9) : 
  molicular_weight_9_moles / weight_9_moles = 108 := 
  by
    sorry

end molecular_weight_of_one_mole_l121_121360


namespace part1_proof_l121_121477

variable (a r : ℝ) (f : ℝ → ℝ)

axiom a_gt_1 : a > 1
axiom r_gt_1 : r > 1

axiom f_condition : ∀ x > 0, f x * f x ≤ a * x * f (x / a)
axiom f_bound : ∀ x, 0 < x ∧ x < 1 / 2^2005 → f x < 2^2005

theorem part1_proof : ∀ x > 0, f x ≤ a^(1 - r) * x := 
by 
  sorry

end part1_proof_l121_121477


namespace Ingrid_cookie_percentage_l121_121657

theorem Ingrid_cookie_percentage : 
  let irin_ratio := 9.18
  let ingrid_ratio := 5.17
  let nell_ratio := 2.05
  let kim_ratio := 3.45
  let linda_ratio := 4.56
  let total_cookies := 800
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio + kim_ratio + linda_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let ingrid_cookies := ingrid_share * total_cookies
  let ingrid_percentage := (ingrid_cookies / total_cookies) * 100
  ingrid_percentage = 21.25 :=
by
  sorry

end Ingrid_cookie_percentage_l121_121657


namespace find_multiple_of_numerator_l121_121583

theorem find_multiple_of_numerator
  (n d k : ℕ)
  (h1 : d = k * n - 1)
  (h2 : (n + 1) / (d + 1) = 3 / 5)
  (h3 : (n : ℚ) / d = 5 / 9) : k = 2 :=
sorry

end find_multiple_of_numerator_l121_121583


namespace complex_subtraction_l121_121361

theorem complex_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 + 3 * I) (h2 : z2 = 3 + I) :
  z1 - z2 = -1 + 2 * I := 
by
  sorry

end complex_subtraction_l121_121361


namespace alpha_beta_square_eq_eight_l121_121982

open Real

theorem alpha_beta_square_eq_eight :
  ∃ α β : ℝ, 
  (∀ x : ℝ, x^2 - 2 * x - 1 = 0 ↔ x = α ∨ x = β) → 
  (α ≠ β) → 
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_eq_eight_l121_121982


namespace average_weight_of_whole_class_l121_121378

theorem average_weight_of_whole_class :
  let students_A := 50
  let students_B := 50
  let avg_weight_A := 60
  let avg_weight_B := 80
  let total_students := students_A + students_B
  let total_weight_A := students_A * avg_weight_A
  let total_weight_B := students_B * avg_weight_B
  let total_weight := total_weight_A + total_weight_B
  let avg_weight := total_weight / total_students
  avg_weight = 70 := 
by 
  sorry

end average_weight_of_whole_class_l121_121378


namespace third_term_arithmetic_sequence_l121_121506

variable (a d : ℤ)
variable (h1 : a + 20 * d = 12)
variable (h2 : a + 21 * d = 15)

theorem third_term_arithmetic_sequence : a + 2 * d = -42 := by
  sorry

end third_term_arithmetic_sequence_l121_121506


namespace oranges_after_selling_l121_121560

-- Definitions derived from the conditions
def oranges_picked := 37
def oranges_sold := 10
def oranges_left := 27

-- The theorem to prove that Joan is left with 27 oranges
theorem oranges_after_selling (h : oranges_picked - oranges_sold = oranges_left) : oranges_left = 27 :=
by
  -- Proof omitted
  sorry

end oranges_after_selling_l121_121560


namespace factor_M_l121_121279

theorem factor_M (a b c d : ℝ) : 
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 =
  (a * c + b * d - a^2 - b^2)^2 :=
by
  sorry

end factor_M_l121_121279


namespace find_AD_l121_121479

noncomputable def A := 0
noncomputable def C := 3
noncomputable def B (x : ℝ) := C - x
noncomputable def D (x : ℝ) := A + 3 + x

-- conditions
def AC := 3
def BD := 4
def ratio_condition (x : ℝ) := (A + C - x - (A + 3)) / x = (A + 3 + x) / x

-- theorem statement
theorem find_AD (x : ℝ) (h1 : AC = 3) (h2 : BD = 4) (h3 : ratio_condition x) :
  D x = 6 :=
sorry

end find_AD_l121_121479


namespace cost_equality_store_comparison_for_10_l121_121291

-- price definitions
def teapot_price := 30
def teacup_price := 5
def teapot_count := 5

-- store A and B promotional conditions
def storeA_cost (x : Nat) : Real := 5 * x + 125
def storeB_cost (x : Nat) : Real := 4.5 * x + 135

theorem cost_equality (x : Nat) (h : x > 5) :
  storeA_cost x = storeB_cost x → x = 20 := by
  sorry

theorem store_comparison_for_10 (x : Nat) (h : x = 10) :
  storeA_cost x < storeB_cost x := by
  sorry

end cost_equality_store_comparison_for_10_l121_121291


namespace gym_membership_count_l121_121707

theorem gym_membership_count :
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  number_of_members = 300 :=
by
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  sorry

end gym_membership_count_l121_121707


namespace find_x_l121_121993

theorem find_x (x : ℝ) : 17 + x + 2 * x + 13 = 60 → x = 10 :=
by
  sorry

end find_x_l121_121993


namespace unique_solution_inequality_l121_121650

theorem unique_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, -3 ≤ x^2 - 2 * a * x + a ∧ x^2 - 2 * a * x + a ≤ -2 → ∃! x : ℝ, x^2 - 2 * a * x + a = -2) ↔ (a = 2 ∨ a = -1) :=
sorry

end unique_solution_inequality_l121_121650


namespace min_value_l121_121972

theorem min_value (x : ℝ) (h : x > 2) : ∃ y, y = 22 ∧ 
  ∀ z, (z > 2) → (y ≤ (z^2 + 8) / (Real.sqrt (z - 2))) := 
sorry

end min_value_l121_121972


namespace algebraic_expression_evaluation_l121_121128

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + 3 * x - 5 = 2) : 2 * x^2 + 6 * x - 3 = 11 :=
sorry

end algebraic_expression_evaluation_l121_121128


namespace complex_solution_l121_121529

theorem complex_solution (z : ℂ) (h : z * (0 + 1 * I) = (0 + 1 * I) - 1) : z = 1 + I :=
by
  sorry

end complex_solution_l121_121529


namespace smallest_y_l121_121117

theorem smallest_y (y : ℕ) : (27^y > 3^24) ↔ (y ≥ 9) :=
sorry

end smallest_y_l121_121117


namespace sandy_more_tokens_than_siblings_l121_121742

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

end sandy_more_tokens_than_siblings_l121_121742


namespace Bobby_ate_5_pancakes_l121_121843

theorem Bobby_ate_5_pancakes
  (total_pancakes : ℕ := 21)
  (dog_eaten : ℕ := 7)
  (leftover : ℕ := 9) :
  (total_pancakes - dog_eaten - leftover = 5) := by
  sorry

end Bobby_ate_5_pancakes_l121_121843


namespace brownies_pieces_count_l121_121023

theorem brownies_pieces_count:
  let pan_width := 24
  let pan_length := 15
  let piece_width := 3
  let piece_length := 2
  pan_width * pan_length / (piece_width * piece_length) = 60 := 
by
  sorry

end brownies_pieces_count_l121_121023


namespace rihanna_money_left_l121_121679

-- Definitions of the item costs
def cost_of_mangoes : ℝ := 6 * 3
def cost_of_apple_juice : ℝ := 4 * 3.50
def cost_of_potato_chips : ℝ := 2 * 2.25
def cost_of_chocolate_bars : ℝ := 3 * 1.75

-- Total cost computation
def total_cost : ℝ := cost_of_mangoes + cost_of_apple_juice + cost_of_potato_chips + cost_of_chocolate_bars

-- Initial amount of money Rihanna has
def initial_money : ℝ := 50

-- Remaining money after the purchases
def remaining_money : ℝ := initial_money - total_cost

-- The theorem stating that the remaining money is $8.25
theorem rihanna_money_left : remaining_money = 8.25 := by
  -- Lean will require the proof here.
  sorry

end rihanna_money_left_l121_121679


namespace abcdefg_defghij_value_l121_121888

variable (a b c d e f g h i : ℚ)

theorem abcdefg_defghij_value :
  (a / b = -7 / 3) →
  (b / c = -5 / 2) →
  (c / d = 2) →
  (d / e = -3 / 2) →
  (e / f = 4 / 3) →
  (f / g = -1 / 4) →
  (g / h = 3 / -5) →
  (abcdefg / defghij = (-21 / 16) * (c / i)) :=
by
  sorry

end abcdefg_defghij_value_l121_121888


namespace a_10_equals_1024_l121_121295

-- Define the sequence a_n and its properties
variable {a : ℕ → ℕ}
variable (h_prop : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)
variable (h_a2 : a 2 = 4)

-- Prove the statement that a_10 = 1024 given the above conditions.
theorem a_10_equals_1024 : a 10 = 1024 :=
sorry

end a_10_equals_1024_l121_121295


namespace find_b_c_find_a_range_l121_121901

noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c
noncomputable def g (a b c x : ℝ) : ℝ := f a b c x + 2 * x
noncomputable def f_prime (a b x : ℝ) : ℝ := x^2 - a * x + b
noncomputable def g_prime (a b x : ℝ) : ℝ := f_prime a b x + 2

theorem find_b_c (a c : ℝ) (h_f0 : f a 0 c 0 = c) (h_tangent_y_eq_1 : 1 = c) : 
  b = 0 ∧ c = 1 :=
by
  sorry

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, g_prime a 0 x ≥ 0) ↔ a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end find_b_c_find_a_range_l121_121901


namespace area_to_paint_l121_121396

def height_of_wall : ℝ := 10
def length_of_wall : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 3
def door_height : ℝ := 1
def door_length : ℝ := 7

theorem area_to_paint : 
  let total_wall_area := height_of_wall * length_of_wall
  let window_area := window_height * window_length
  let door_area := door_height * door_length
  let area_to_paint := total_wall_area - window_area - door_area
  area_to_paint = 134 := 
by 
  sorry

end area_to_paint_l121_121396


namespace zyka_expense_increase_l121_121595

theorem zyka_expense_increase (C_k C_c : ℝ) (h1 : 0.5 * C_k = 0.2 * C_c) : 
  (((1.2 * C_c) - C_c) / C_c) * 100 = 20 := by
  sorry

end zyka_expense_increase_l121_121595


namespace second_alloy_amount_l121_121955

theorem second_alloy_amount (x : ℝ) :
  let chromium_first_alloy := 0.12 * 15
  let chromium_second_alloy := 0.08 * x
  let total_weight := 15 + x
  let chromium_percentage_new_alloy := (0.12 * 15 + 0.08 * x) / (15 + x)
  chromium_percentage_new_alloy = (28 / 300) →
  x = 30 := sorry

end second_alloy_amount_l121_121955


namespace rectangle_hall_length_l121_121567

variable (L B : ℝ)

theorem rectangle_hall_length (h1 : B = (2 / 3) * L) (h2 : L * B = 2400) : L = 60 :=
by sorry

end rectangle_hall_length_l121_121567


namespace inequality_abcde_l121_121846

theorem inequality_abcde
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d) : 
  1 / a + 1 / b + 4 / c + 16 / d ≥ 64 / (a + b + c + d) := 
  sorry

end inequality_abcde_l121_121846


namespace hexagon_perimeter_l121_121994

theorem hexagon_perimeter (s : ℕ) (P : ℕ) (h1 : s = 8) (h2 : 6 > 0) 
                          (h3 : P = 6 * s) : P = 48 := by
  sorry

end hexagon_perimeter_l121_121994


namespace smallest_n_for_geometric_sequence_divisibility_l121_121318

theorem smallest_n_for_geometric_sequence_divisibility :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬ (2 * 10 ^ 6 ∣ (30 ^ (m - 1) * (5 / 6)))) ∧ (2 * 10 ^ 6 ∣ (30 ^ (n - 1) * (5 / 6))) ∧ n = 8 :=
by
  sorry

end smallest_n_for_geometric_sequence_divisibility_l121_121318


namespace triangle_square_ratio_l121_121369

theorem triangle_square_ratio (s_t s_s : ℕ) (h : 3 * s_t = 4 * s_s) : (s_t : ℚ) / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l121_121369


namespace gcd_m_n_is_one_l121_121352

def m : ℕ := 122^2 + 234^2 + 344^2

def n : ℕ := 123^2 + 235^2 + 343^2

theorem gcd_m_n_is_one : Nat.gcd m n = 1 :=
by
  sorry

end gcd_m_n_is_one_l121_121352


namespace Henry_trays_per_trip_l121_121976

theorem Henry_trays_per_trip (trays1 trays2 trips : ℕ) (h1 : trays1 = 29) (h2 : trays2 = 52) (h3 : trips = 9) :
  (trays1 + trays2) / trips = 9 :=
by
  sorry

end Henry_trays_per_trip_l121_121976


namespace sum_of_legs_of_right_triangle_l121_121800

theorem sum_of_legs_of_right_triangle
  (a b : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b = a + 2)
  (h3 : a^2 + b^2 = 50^2) :
  a + b = 70 := by
  sorry

end sum_of_legs_of_right_triangle_l121_121800


namespace repeating_decimals_expr_as_fraction_l121_121225

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end repeating_decimals_expr_as_fraction_l121_121225


namespace greatest_integer_with_gcd_6_l121_121019

theorem greatest_integer_with_gcd_6 (x : ℕ) :
  x < 150 ∧ gcd x 12 = 6 → x = 138 :=
by
  sorry

end greatest_integer_with_gcd_6_l121_121019


namespace set_A_membership_l121_121037

theorem set_A_membership (U : Finset ℕ) (A : Finset ℕ) (B : Finset ℕ)
  (hU : U.card = 193)
  (hB : B.card = 49)
  (hneither : (U \ (A ∪ B)).card = 59)
  (hAandB : (A ∩ B).card = 25) :
  A.card = 110 := sorry

end set_A_membership_l121_121037


namespace algebra_1_algebra_2_l121_121700

variable (x1 x2 : ℝ)
variable (h_root1 : x1^2 - 2*x1 - 1 = 0)
variable (h_root2 : x2^2 - 2*x2 - 1 = 0)
variable (h_sum : x1 + x2 = 2)
variable (h_prod : x1 * x2 = -1)

theorem algebra_1 : (x1 + x2) * (x1 * x2) = -2 := by
  -- Proof here
  sorry

theorem algebra_2 : (x1 - x2)^2 = 8 := by
  -- Proof here
  sorry

end algebra_1_algebra_2_l121_121700


namespace combined_original_price_l121_121028

theorem combined_original_price (S P : ℝ) 
  (hS : 0.25 * S = 6) 
  (hP : 0.60 * P = 12) :
  S + P = 44 :=
by
  sorry

end combined_original_price_l121_121028


namespace original_pencils_l121_121156

-- Define the conditions given in the problem
variable (total_pencils_now : ℕ) [DecidableEq ℕ] (pencils_by_Mike : ℕ)

-- State the problem to prove
theorem original_pencils (h1 : total_pencils_now = 71) (h2 : pencils_by_Mike = 30) : total_pencils_now - pencils_by_Mike = 41 := by
  sorry

end original_pencils_l121_121156


namespace a_pow_b_iff_a_minus_1_b_positive_l121_121791

theorem a_pow_b_iff_a_minus_1_b_positive (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : 
  (a^b > 1) ↔ ((a - 1) * b > 0) := 
sorry

end a_pow_b_iff_a_minus_1_b_positive_l121_121791


namespace find_n_l121_121172

noncomputable def n : ℕ := sorry -- Explicitly define n as a variable, but the value is not yet provided.

theorem find_n (h₁ : n > 0)
    (h₂ : Real.sqrt 3 > (n + 4) / (n + 1))
    (h₃ : Real.sqrt 3 < (n + 3) / n) : 
    n = 4 :=
sorry

end find_n_l121_121172


namespace nearest_integer_is_11304_l121_121683

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l121_121683


namespace abs_inequality_solution_l121_121032

theorem abs_inequality_solution (x : ℝ) : |x + 2| + |x - 1| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 :=
sorry

end abs_inequality_solution_l121_121032


namespace A_is_false_l121_121370

variables {a b : ℝ}

-- Condition: Proposition B - The sum of the roots of the equation is 2
axiom sum_of_roots : ∀ (x1 x2 : ℝ), x1 + x2 = -a

-- Condition: Proposition C - x = 3 is a root of the equation
axiom root3 : ∃ (x1 x2 : ℝ), (x1 = 3 ∨ x2 = 3)

-- Condition: Proposition D - The two roots have opposite signs
axiom opposite_sign_roots : ∀ (x1 x2 : ℝ), x1 * x2 < 0

-- Prove: Proposition A is false
theorem A_is_false : ¬ (∃ x1 x2 : ℝ, x1 = 1 ∨ x2 = 1) :=
by
  sorry

end A_is_false_l121_121370


namespace fixed_monthly_charge_for_100_GB_l121_121041

theorem fixed_monthly_charge_for_100_GB
  (fixed_charge M : ℝ)
  (extra_charge_per_GB : ℝ := 0.25)
  (total_bill : ℝ := 65)
  (GB_over : ℝ := 80)
  (extra_charge : ℝ := GB_over * extra_charge_per_GB) :
  total_bill = M + extra_charge → M = 45 :=
by sorry

end fixed_monthly_charge_for_100_GB_l121_121041


namespace isosceles_triangle_perimeter_l121_121456

noncomputable def perimeter_of_isosceles_triangle : ℝ :=
  let BC := 10
  let height := 6
  let half_base := BC / 2
  let side := Real.sqrt (height^2 + half_base^2)
  let perimeter := 2 * side + BC
  perimeter

theorem isosceles_triangle_perimeter :
  let BC := 10
  let height := 6
  perimeter_of_isosceles_triangle = 2 * Real.sqrt (height^2 + (BC / 2)^2) + BC := by
  sorry

end isosceles_triangle_perimeter_l121_121456


namespace midpoint_one_sixth_one_twelfth_l121_121167

theorem midpoint_one_sixth_one_twelfth : (1 : ℚ) / 8 = (1 / 6 + 1 / 12) / 2 := by
  sorry

end midpoint_one_sixth_one_twelfth_l121_121167


namespace find_a_for_parallel_lines_l121_121593

theorem find_a_for_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 ↔ 2 * x + (a + 1) * y + 1 = 0) → a = -3 :=
by
  sorry

end find_a_for_parallel_lines_l121_121593


namespace central_angle_of_sector_l121_121715

theorem central_angle_of_sector
  (r : ℝ) (S_sector : ℝ) (alpha : ℝ) (h₁ : r = 2) (h₂ : S_sector = (2 / 5) * Real.pi)
  (h₃ : S_sector = (1 / 2) * alpha * r^2) : alpha = Real.pi / 5 :=
by
  sorry

end central_angle_of_sector_l121_121715


namespace percentage_increase_twice_l121_121644

theorem percentage_increase_twice (P : ℝ) (x : ℝ) :
  P * (1 + x)^2 = P * 1.3225 → x = 0.15 :=
by
  intro h
  have h1 : (1 + x)^2 = 1.3225 := by sorry
  have h2 : x^2 + 2 * x = 0.3225 := by sorry
  have h3 : x = (-2 + Real.sqrt 5.29) / 2 := by sorry
  have h4 : x = -2 / 2 + Real.sqrt 5.29 / 2 := by sorry
  have h5 : x = 0.15 := by sorry
  exact h5

end percentage_increase_twice_l121_121644


namespace rashmi_late_time_is_10_l121_121612

open Real

noncomputable def rashmi_late_time : ℝ :=
  let d : ℝ := 9.999999999999993
  let v1 : ℝ := 5 / 60 -- km per minute
  let v2 : ℝ := 6 / 60 -- km per minute
  let time1 := d / v1 -- time taken at 5 kmph
  let time2 := d / v2 -- time taken at 6 kmph
  let difference := time1 - time2
  let T := difference / 2 -- The time she was late or early
  T

theorem rashmi_late_time_is_10 : rashmi_late_time = 10 := by
  simp [rashmi_late_time]
  sorry

end rashmi_late_time_is_10_l121_121612


namespace sum_of_two_integers_l121_121421

theorem sum_of_two_integers (x y : ℝ) (h₁ : x^2 + y^2 = 130) (h₂ : x * y = 45) : x + y = 2 * Real.sqrt 55 :=
sorry

end sum_of_two_integers_l121_121421


namespace max_elements_set_M_l121_121589

theorem max_elements_set_M (n : ℕ) (hn : n ≥ 2) (M : Finset (ℕ × ℕ))
  (hM : ∀ {i k}, (i, k) ∈ M → i < k → ∀ {m}, k < m → (k, m) ∉ M) :
  M.card ≤ n^2 / 4 :=
sorry

end max_elements_set_M_l121_121589


namespace fraction_multiplication_l121_121646

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l121_121646


namespace triangle_angle_contradiction_l121_121735

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α < 60) (h3 : β < 60) (h4 : γ < 60) : false := 
sorry

end triangle_angle_contradiction_l121_121735


namespace necessary_not_sufficient_l121_121725

-- Definitions and conditions based on the problem statement
def x_ne_1 (x : ℝ) : Prop := x ≠ 1
def polynomial_ne_zero (x : ℝ) : Prop := (x^2 - 3 * x + 2) ≠ 0

-- The theorem statement
theorem necessary_not_sufficient (x : ℝ) : 
  (∀ x, polynomial_ne_zero x → x_ne_1 x) ∧ ¬ (∀ x, x_ne_1 x → polynomial_ne_zero x) :=
by 
  intros
  sorry

end necessary_not_sufficient_l121_121725


namespace problem_statement_l121_121102

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end problem_statement_l121_121102


namespace calc_tan_fraction_l121_121727

theorem calc_tan_fraction :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h_tan_30 : Real.tan (30 * Real.pi / 180) = Real.sqrt 3 / 3 := by sorry
  sorry

end calc_tan_fraction_l121_121727


namespace smallest_x_l121_121834

theorem smallest_x (x : ℚ) (h : 7 * (4 * x^2 + 4 * x + 5) = x * (4 * x - 35)) : 
  x = -5/3 ∨ x = -7/8 := by
  sorry

end smallest_x_l121_121834


namespace sales_in_fourth_month_l121_121255

theorem sales_in_fourth_month
  (sale1 : ℕ)
  (sale2 : ℕ)
  (sale3 : ℕ)
  (sale5 : ℕ)
  (sale6 : ℕ)
  (average : ℕ)
  (h_sale1 : sale1 = 2500)
  (h_sale2 : sale2 = 6500)
  (h_sale3 : sale3 = 9855)
  (h_sale5 : sale5 = 7000)
  (h_sale6 : sale6 = 11915)
  (h_average : average = 7500) :
  ∃ sale4 : ℕ, sale4 = 14230 := by
  sorry

end sales_in_fourth_month_l121_121255


namespace find_b_skew_lines_l121_121637

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3*t, 3 + 4*t, b + 5*t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6*u, 6 + 3*u, 1 + 2*u)

noncomputable def lines_are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem find_b_skew_lines (b : ℝ) : b ≠ -12 / 5 → lines_are_skew b :=
by
  sorry

end find_b_skew_lines_l121_121637


namespace polar_eq_of_circle_product_of_distances_MA_MB_l121_121872

noncomputable def circle_center := (2, Real.pi / 3)
noncomputable def circle_radius := 2

-- Polar equation of the circle
theorem polar_eq_of_circle :
  ∀ (ρ θ : ℝ),
    (circle_center.snd = Real.pi / 3) →
    ρ = 2 * 2 * Real.cos (θ - circle_center.snd) → 
    ρ = 4 * Real.cos (θ - (Real.pi / 3)) :=
by 
  sorry

noncomputable def point_M := (1, -2)

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ := 
  (1 + 1/2 * t, -2 + Real.sqrt 3 / 2 * t)

noncomputable def cartesian_center := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
noncomputable def cartesian_radius := 2

-- Cartesian form of the circle equation from the polar coordinates
noncomputable def cartesian_eq (x y : ℝ) : Prop :=
  (x - cartesian_center.fst)^2 + (y - cartesian_center.snd)^2 = circle_radius^2

-- Product of distances |MA| * |MB|
theorem product_of_distances_MA_MB :
  ∃ (t1 t2 : ℝ),
  (∀ t, parametric_line t ∈ {p : ℝ × ℝ | cartesian_eq p.fst p.snd}) → 
  (point_M.fst, point_M.snd) = (1, -2) →
  t1 * t2 = 3 + 4 * Real.sqrt 3 :=
by
  sorry

end polar_eq_of_circle_product_of_distances_MA_MB_l121_121872


namespace quadratic_with_real_roots_l121_121015

theorem quadratic_with_real_roots: 
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4 * x₁ + k = 0 ∧ x₂^2 + 4 * x₂ + k = 0) ↔ (k ≤ 4) := 
by 
  sorry

end quadratic_with_real_roots_l121_121015


namespace circle_people_count_l121_121046

def num_people (n : ℕ) (a b : ℕ) : Prop :=
  a = 7 ∧ b = 18 ∧ (b = a + (n / 2))

theorem circle_people_count (n : ℕ) (a b : ℕ) (h : num_people n a b) : n = 24 :=
by
  sorry

end circle_people_count_l121_121046


namespace cubic_yard_to_cubic_meter_l121_121947

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end cubic_yard_to_cubic_meter_l121_121947


namespace jake_fewer_peaches_undetermined_l121_121647

theorem jake_fewer_peaches_undetermined 
    (steven_peaches : ℕ) 
    (steven_apples : ℕ) 
    (jake_fewer_peaches : steven_peaches > jake_peaches) 
    (jake_more_apples : jake_apples = steven_apples + 3) 
    (steven_peaches_val : steven_peaches = 9) 
    (steven_apples_val : steven_apples = 8) : 
    ∃ n : ℕ, jake_peaches = n ∧ ¬(∃ m : ℕ, steven_peaches - jake_peaches = m) := 
sorry

end jake_fewer_peaches_undetermined_l121_121647


namespace find_f_1_div_2007_l121_121733

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_1_div_2007 :
  f 0 = 0 ∧
  (∀ x, f x + f (1 - x) = 1) ∧
  (∀ x, f (x / 5) = f x / 2) ∧
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f x1 ≤ f x2) →
  f (1 / 2007) = 1 / 32 :=
sorry

end find_f_1_div_2007_l121_121733


namespace new_average_weight_l121_121147

def average_weight (A B C D E : ℝ) : Prop :=
  (A + B + C) / 3 = 70 ∧
  (A + B + C + D) / 4 = 70 ∧
  E = D + 3 ∧
  A = 81

theorem new_average_weight (A B C D E : ℝ) (h: average_weight A B C D E) : 
  (B + C + D + E) / 4 = 68 :=
by
  sorry

end new_average_weight_l121_121147


namespace triangle_altitude_from_equal_area_l121_121457

variable (x : ℝ)

theorem triangle_altitude_from_equal_area (h : x^2 = (1 / 2) * x * altitude) :
  altitude = 2 * x := by
  sorry

end triangle_altitude_from_equal_area_l121_121457


namespace interval_of_monotonic_increase_l121_121328

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (6 + x - x^2)

theorem interval_of_monotonic_increase :
  {x : ℝ | -2 < x ∧ x < 3} → x ∈ Set.Ioc (1/2) 3 :=
by
  sorry

end interval_of_monotonic_increase_l121_121328


namespace range_of_g_l121_121228

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f (x))))

theorem range_of_g : ∀ x, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by
  intro x h
  sorry

end range_of_g_l121_121228


namespace sickness_temperature_increase_l121_121273

theorem sickness_temperature_increase :
  ∀ (normal_temp fever_threshold current_temp : ℕ), normal_temp = 95 → fever_threshold = 100 →
  current_temp = fever_threshold + 5 → (current_temp - normal_temp = 10) :=
by
  intros normal_temp fever_threshold current_temp h1 h2 h3
  sorry

end sickness_temperature_increase_l121_121273


namespace quadratic_root_exists_l121_121400

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l121_121400


namespace find_f7_l121_121957

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x^7 + b * x^3 + c * x - 5

theorem find_f7 (a b c : ℝ) (h : f (-7) a b c = 7) : f 7 a b c = -17 :=
by
  sorry

end find_f7_l121_121957


namespace eval_expression_l121_121204

theorem eval_expression : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 :=
by
  sorry

end eval_expression_l121_121204


namespace largest_digit_B_divisible_by_4_l121_121306

theorem largest_digit_B_divisible_by_4 :
  ∃ B : ℕ, B = 9 ∧ ∀ k : ℕ, (k ≤ 9 → (∃ n : ℕ, 4 * n = 10 * B + 792 % 100)) :=
by
  sorry

end largest_digit_B_divisible_by_4_l121_121306


namespace area_diff_l121_121326

-- Defining the side lengths of squares
def side_length_small_square : ℕ := 4
def side_length_large_square : ℕ := 10

-- Calculating the areas
def area_small_square : ℕ := side_length_small_square ^ 2
def area_large_square : ℕ := side_length_large_square ^ 2

-- Theorem statement
theorem area_diff (a_small a_large : ℕ) (h1 : a_small = side_length_small_square ^ 2) (h2 : a_large = side_length_large_square ^ 2) : 
  a_large - a_small = 84 :=
by
  sorry

end area_diff_l121_121326


namespace geometric_sequence_x_l121_121778

theorem geometric_sequence_x (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l121_121778


namespace part_a_part_b_l121_121366

variable {A : Type} [Ring A] (h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6)

-- Part (a)
theorem part_a (x : A) (n : Nat) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 :=
sorry

-- Part (b)
theorem part_b (x : A) : x^4 = x :=
by
  have h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6 := h
  sorry

end part_a_part_b_l121_121366


namespace determine_BD_l121_121197

def quadrilateral (AB BC CD DA BD : ℕ) : Prop :=
AB = 6 ∧ BC = 15 ∧ CD = 8 ∧ DA = 12 ∧ (7 < BD ∧ BD < 18)

theorem determine_BD : ∃ BD : ℕ, quadrilateral 6 15 8 12 BD ∧ 8 ≤ BD ∧ BD ≤ 17 :=
by
  sorry

end determine_BD_l121_121197


namespace like_terms_correct_l121_121233

theorem like_terms_correct : 
  (¬(∀ x y z w : ℝ, (x * y^2 = z ∧ x^2 * y = w)) ∧ 
   ¬(∀ x y : ℝ, (x * y = -2 * y)) ∧ 
    (2^3 = 8 ∧ 3^2 = 9) ∧ 
   ¬(∀ x y z w : ℝ, (5 * x * y = z ∧ 6 * x * y^2 = w))) :=
by
  sorry

end like_terms_correct_l121_121233


namespace distance_reflection_x_axis_l121_121500

/--
Given points C and its reflection over the x-axis C',
prove that the distance between C and C' is 6.
-/
theorem distance_reflection_x_axis :
  let C := (-2, 3)
  let C' := (-2, -3)
  dist C C' = 6 := by
  sorry

end distance_reflection_x_axis_l121_121500


namespace jelly_cost_l121_121548

theorem jelly_cost (B J : ℕ) 
  (h1 : 15 * (6 * B + 7 * J) = 315) 
  (h2 : 0 ≤ B) 
  (h3 : 0 ≤ J) : 
  15 * J * 7 = 315 := 
sorry

end jelly_cost_l121_121548


namespace xiaoming_total_money_l121_121277

def xiaoming_money (x : ℕ) := 9 * x

def fresh_milk_cost (y : ℕ) := 6 * y

def yogurt_cost_equation (x y : ℕ) := y = x + 6

theorem xiaoming_total_money (x : ℕ) (y : ℕ)
  (h1: fresh_milk_cost y = xiaoming_money x)
  (h2: yogurt_cost_equation x y) : xiaoming_money x = 108 := 
  sorry

end xiaoming_total_money_l121_121277


namespace solve_inequality_l121_121428

theorem solve_inequality (x : ℝ) : (1 + x) / 3 < x / 2 → x > 2 := 
by {
  sorry
}

end solve_inequality_l121_121428


namespace find_g_values_l121_121375

variables (f g : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ x y, g (x - y) = g x * g y + f x * f y
axiom cond2 : f (-1) = -1
axiom cond3 : f 0 = 0
axiom cond4 : f 1 = 1

-- Goal
theorem find_g_values : g 0 = 1 ∧ g 1 = 0 ∧ g 2 = -1 :=
by
  sorry

end find_g_values_l121_121375


namespace largest_sum_of_three_faces_l121_121522

theorem largest_sum_of_three_faces (faces : Fin 6 → ℕ)
  (h_unique : ∀ i j, i ≠ j → faces i ≠ faces j)
  (h_range : ∀ i, 1 ≤ faces i ∧ faces i ≤ 6)
  (h_opposite_sum : ∀ i, faces i + faces (5 - i) = 10) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ faces i + faces j + faces k = 12 :=
by sorry

end largest_sum_of_three_faces_l121_121522


namespace find_m_l121_121866

theorem find_m 
  (m : ℝ) 
  (h1 : |m + 1| ≠ 0)
  (h2 : m^2 = 1) : 
  m = 1 := sorry

end find_m_l121_121866


namespace find_a_l121_121965

noncomputable def set_A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem find_a (a : ℝ) (h : 3 ∈ set_A a) : a = -3 / 2 :=
by
  sorry

end find_a_l121_121965


namespace percentage_invalid_votes_l121_121827

theorem percentage_invalid_votes
  (total_votes : ℕ)
  (votes_for_A : ℕ)
  (candidate_A_percentage : ℝ)
  (total_votes_count : total_votes = 560000)
  (votes_for_A_count : votes_for_A = 404600)
  (candidate_A_percentage_count : candidate_A_percentage = 0.85) :
  ∃ (x : ℝ), (x / 100) * total_votes = total_votes - votes_for_A / candidate_A_percentage ∧ x = 15 :=
by
  sorry

end percentage_invalid_votes_l121_121827


namespace chocolate_bars_per_box_l121_121531

theorem chocolate_bars_per_box (total_chocolate_bars boxes : ℕ) (h1 : total_chocolate_bars = 710) (h2 : boxes = 142) : total_chocolate_bars / boxes = 5 := by
  sorry

end chocolate_bars_per_box_l121_121531


namespace bookshop_inventory_l121_121319

theorem bookshop_inventory
  (initial_inventory : ℕ := 743)
  (saturday_sales_instore : ℕ := 37)
  (saturday_sales_online : ℕ := 128)
  (sunday_sales_instore : ℕ := 2 * saturday_sales_instore)
  (sunday_sales_online : ℕ := saturday_sales_online + 34)
  (new_shipment : ℕ := 160) :
  (initial_inventory - (saturday_sales_instore + saturday_sales_online + sunday_sales_instore + sunday_sales_online) + new_shipment = 502) :=
by
  sorry

end bookshop_inventory_l121_121319


namespace minimize_S_n_at_7_l121_121169

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 2 - a 1

def conditions (a : ℕ → ℤ) : Prop :=
arithmetic_sequence a ∧ a 2 = -11 ∧ (a 5 + a 9 = -2)

-- Define the sum of first n terms of the sequence
def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Define the minimum S_n and that it occurs at n = 7
theorem minimize_S_n_at_7 (a : ℕ → ℤ) (n : ℕ) (h : conditions a) :
  ∀ m, S a m ≥ S a 7 := sorry

end minimize_S_n_at_7_l121_121169


namespace subset_neg1_of_leq3_l121_121044

theorem subset_neg1_of_leq3 :
  {x | x = -1} ⊆ {x | x ≤ 3} :=
sorry

end subset_neg1_of_leq3_l121_121044


namespace roots_product_eq_three_l121_121133

theorem roots_product_eq_three
  (p q r : ℝ)
  (h : (3:ℝ) * p ^ 3 - 8 * p ^ 2 + p - 9 = 0 ∧
       (3:ℝ) * q ^ 3 - 8 * q ^ 2 + q - 9 = 0 ∧
       (3:ℝ) * r ^ 3 - 8 * r ^ 2 + r - 9 = 0) :
  p * q * r = 3 :=
sorry

end roots_product_eq_three_l121_121133


namespace total_payment_l121_121832

def cement_bags := 500
def cost_per_bag := 10
def lorries := 20
def tons_per_lorry := 10
def cost_per_ton := 40

theorem total_payment : cement_bags * cost_per_bag + lorries * tons_per_lorry * cost_per_ton = 13000 := by
  sorry

end total_payment_l121_121832


namespace complement_intersection_l121_121018

-- Definitions for the sets
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to a universal set
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Theorem to prove
theorem complement_intersection :
  complement U (A ∩ B) = {1, 4, 6} :=
by
  sorry

end complement_intersection_l121_121018


namespace find_number_l121_121959

theorem find_number (x : ℝ) (h: x - (3 / 5) * x = 58) : x = 145 :=
by {
  sorry
}

end find_number_l121_121959


namespace lollipops_given_l121_121848

theorem lollipops_given (initial_people later_people : ℕ) (total_people groups_of_five : ℕ) :
  initial_people = 45 →
  later_people = 15 →
  total_people = initial_people + later_people →
  groups_of_five = total_people / 5 →
  total_people = 60 →
  groups_of_five = 12 :=
by intros; sorry

end lollipops_given_l121_121848


namespace sum_of_integers_eq_l121_121049

-- We define the conditions
variables (x y : ℕ)
-- The conditions specified in the problem
def diff_condition : Prop := x - y = 16
def prod_condition : Prop := x * y = 63

-- The theorem stating that given the conditions, the sum is 2*sqrt(127)
theorem sum_of_integers_eq : diff_condition x y → prod_condition x y → x + y = 2 * Real.sqrt 127 :=
by
  sorry

end sum_of_integers_eq_l121_121049


namespace trajectory_center_of_C_number_of_lines_l_l121_121443

noncomputable def trajectory_equation : Prop :=
  ∃ (a b : ℝ), a = 4 ∧ b^2 = 12 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_count : Prop :=
  ∀ (k m : ℤ), 
  ∃ (num_lines : ℕ), 
  (∀ (x : ℝ), (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 48 = 0 → num_lines = 9 ∨ num_lines = 0) ∧
  (∀ (x : ℝ), (3 - k^2) * x^2 - 2 * k * m * x - m^2 - 12 = 0 → num_lines = 9 ∨ num_lines = 0)

theorem trajectory_center_of_C :
  trajectory_equation :=
sorry

theorem number_of_lines_l :
  line_count :=
sorry

end trajectory_center_of_C_number_of_lines_l_l121_121443


namespace range_of_real_number_a_l121_121938

theorem range_of_real_number_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 1 = 0 → x = a) ↔ (a = 0 ∨ a ≥ 9/4) :=
sorry

end range_of_real_number_a_l121_121938


namespace polynomial_sum_l121_121148

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l121_121148


namespace martha_total_cost_l121_121322

-- Definitions for the conditions
def amount_cheese_needed : ℝ := 1.5 -- in kg
def amount_meat_needed : ℝ := 0.5 -- in kg
def cost_cheese_per_kg : ℝ := 6.0 -- in dollars per kg
def cost_meat_per_kg : ℝ := 8.0 -- in dollars per kg

-- Total cost that needs to be calculated
def total_cost : ℝ :=
  (amount_cheese_needed * cost_cheese_per_kg) +
  (amount_meat_needed * cost_meat_per_kg)

-- Statement of the theorem
theorem martha_total_cost : total_cost = 13 := by
  sorry

end martha_total_cost_l121_121322


namespace dropped_score_l121_121301

variable (A B C D : ℕ)

theorem dropped_score (h1 : A + B + C + D = 180) (h2 : A + B + C = 150) : D = 30 := by
  sorry

end dropped_score_l121_121301


namespace sale_price_60_l121_121753

theorem sale_price_60 (original_price : ℕ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) 
  (h2 : discount_percentage = 0.40) :
  sale_price = (original_price : ℝ) * (1 - discount_percentage) :=
by
  sorry

end sale_price_60_l121_121753


namespace measure_of_angle_B_l121_121746

theorem measure_of_angle_B (a b c R : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C)
  (h4 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = Real.pi / 4 :=
by
  sorry

end measure_of_angle_B_l121_121746


namespace largest_inscribed_triangle_area_l121_121874

theorem largest_inscribed_triangle_area (r : ℝ) (h_r : r = 12) : ∃ A : ℝ, A = 144 :=
by
  sorry

end largest_inscribed_triangle_area_l121_121874


namespace sum_and_product_of_radical_l121_121711

theorem sum_and_product_of_radical (a b : ℝ) (h1 : 2 * a = -4) (h2 : a^2 - b = 1) :
  a + b = 1 :=
sorry

end sum_and_product_of_radical_l121_121711


namespace area_of_large_square_l121_121528

theorem area_of_large_square (s : ℝ) (h : 2 * s^2 = 14) : 9 * s^2 = 63 := by
  sorry

end area_of_large_square_l121_121528


namespace not_taking_ship_probability_l121_121629

-- Real non-negative numbers as probabilities
variables (P_train P_ship P_car P_airplane : ℝ)

-- Conditions
axiom h_train : 0 ≤ P_train ∧ P_train ≤ 1 ∧ P_train = 0.3
axiom h_ship : 0 ≤ P_ship ∧ P_ship ≤ 1 ∧ P_ship = 0.1
axiom h_car : 0 ≤ P_car ∧ P_car ≤ 1 ∧ P_car = 0.4
axiom h_airplane : 0 ≤ P_airplane ∧ P_airplane ≤ 1 ∧ P_airplane = 0.2

-- Prove that the probability of not taking a ship is 0.9
theorem not_taking_ship_probability : 1 - P_ship = 0.9 :=
by
  sorry

end not_taking_ship_probability_l121_121629


namespace mabel_petals_remaining_l121_121112

/-- Mabel has 5 daisies, each with 8 petals. If she gives 2 daisies to her teacher,
how many petals does she have on the remaining daisies in her garden? -/
theorem mabel_petals_remaining :
  (5 - 2) * 8 = 24 :=
by
  sorry

end mabel_petals_remaining_l121_121112


namespace bees_population_reduction_l121_121355

theorem bees_population_reduction :
  ∀ (initial_population loss_per_day : ℕ),
  initial_population = 80000 → 
  loss_per_day = 1200 → 
  ∃ days : ℕ, initial_population - days * loss_per_day = initial_population / 4 ∧ days = 50 :=
by
  intros initial_population loss_per_day h_initial h_loss
  use 50
  sorry

end bees_population_reduction_l121_121355


namespace solve_for_n_l121_121446

theorem solve_for_n (n : ℤ) : (3 : ℝ)^(2 * n + 2) = 1 / 9 ↔ n = -2 := by
  sorry

end solve_for_n_l121_121446


namespace black_squares_covered_by_trominoes_l121_121376

theorem black_squares_covered_by_trominoes (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (k : ℕ), k * k = (n + 1) / 2 ∧ n ≥ 7) ↔ n ≥ 7 :=
by
  sorry

end black_squares_covered_by_trominoes_l121_121376


namespace car_tank_capacity_is_12_gallons_l121_121839

noncomputable def truck_tank_capacity : ℕ := 20
noncomputable def truck_tank_half_full : ℕ := truck_tank_capacity / 2
noncomputable def car_tank_third_full (car_tank_capacity : ℕ) : ℕ := car_tank_capacity / 3
noncomputable def total_gallons_added : ℕ := 18

theorem car_tank_capacity_is_12_gallons (car_tank_capacity : ℕ) 
    (h1 : truck_tank_half_full + (car_tank_third_full car_tank_capacity) + 18 = truck_tank_capacity + car_tank_capacity) 
    (h2 : total_gallons_added = 18) : car_tank_capacity = 12 := 
by
  sorry

end car_tank_capacity_is_12_gallons_l121_121839


namespace joe_paint_fraction_l121_121000

theorem joe_paint_fraction :
  let total_paint := 360
  let fraction_first_week := 1 / 9
  let used_first_week := (fraction_first_week * total_paint)
  let remaining_after_first_week := total_paint - used_first_week
  let total_used := 104
  let used_second_week := total_used - used_first_week
  let fraction_second_week := used_second_week / remaining_after_first_week
  fraction_second_week = 1 / 5 :=
by
  sorry

end joe_paint_fraction_l121_121000


namespace vectors_perpendicular_vector_combination_l121_121565

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_c : ℝ × ℝ := (1, 1)

-- Auxiliary definition of vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Auxiliary definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

-- Auxiliary definition of scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Proof that (vector_a + vector_b) is perpendicular to vector_c
theorem vectors_perpendicular : dot_product (vector_add vector_a vector_b) vector_c = 0 :=
by sorry

-- Proof that vector_c = 5 * vector_a + 3 * vector_b
theorem vector_combination : vector_c = vector_add (scalar_mul 5 vector_a) (scalar_mul 3 vector_b) :=
by sorry

end vectors_perpendicular_vector_combination_l121_121565


namespace find_a_purely_imaginary_l121_121052

noncomputable def purely_imaginary_condition (a : ℝ) : Prop :=
    (2 * a - 1) / (a^2 + 1) = 0 ∧ (a + 2) / (a^2 + 1) ≠ 0

theorem find_a_purely_imaginary :
    ∀ (a : ℝ), purely_imaginary_condition a ↔ a = 1/2 := 
by
  sorry

end find_a_purely_imaginary_l121_121052


namespace minimum_value_of_function_l121_121623

theorem minimum_value_of_function : ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_of_function_l121_121623


namespace number_of_apples_l121_121564

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end number_of_apples_l121_121564


namespace sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l121_121283

-- Problem 1: Define the sum of the first n odd numbers and prove it equals n^2 when n = 5.
theorem sum_first_five_odds_equals_25 : (1 + 3 + 5 + 7 + 9 = 5^2) := 
sorry

-- Problem 2: Prove that if the smallest number in the decomposition of m^3 is 21, then m = 5.
theorem smallest_in_cube_decomposition_eq_21 : 
  (∃ m : ℕ, m > 0 ∧ 21 = 2 * m - 1 ∧ m = 5) := 
sorry

end sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l121_121283


namespace andrew_paid_1428_l121_121868

-- Define the constants for the problem
def rate_per_kg_grapes : ℕ := 98
def kg_grapes : ℕ := 11

def rate_per_kg_mangoes : ℕ := 50
def kg_mangoes : ℕ := 7

-- Calculate the cost of grapes and mangoes
def cost_grapes := rate_per_kg_grapes * kg_grapes
def cost_mangoes := rate_per_kg_mangoes * kg_mangoes

-- Calculate the total amount paid
def total_amount_paid := cost_grapes + cost_mangoes

-- State the proof problem
theorem andrew_paid_1428 :
  total_amount_paid = 1428 :=
by
  -- Add the proof to verify the calculations
  sorry

end andrew_paid_1428_l121_121868


namespace number_of_ways_to_label_decagon_equal_sums_l121_121840

open Nat

-- Formal definition of the problem
def sum_of_digits : Nat := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

-- The problem statement: Prove there are 3840 ways to label digits ensuring the given condition
theorem number_of_ways_to_label_decagon_equal_sums :
  ∃ (n : Nat), n = 3840 ∧ ∀ (A B C D E F G H I J K L : Nat), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧ (A ≠ H) ∧ (A ≠ I) ∧ (A ≠ J) ∧ (A ≠ K) ∧ (A ≠ L) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧ (B ≠ H) ∧ (B ≠ I) ∧ (B ≠ J) ∧ (B ≠ K) ∧ (B ≠ L) ∧
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧ (C ≠ H) ∧ (C ≠ I) ∧ (C ≠ J) ∧ (C ≠ K) ∧ (C ≠ L) ∧
    (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧ (D ≠ H) ∧ (D ≠ I) ∧ (D ≠ J) ∧ (D ≠ K) ∧ (D ≠ L) ∧
    (E ≠ F) ∧ (E ≠ G) ∧ (E ≠ H) ∧ (E ≠ I) ∧ (E ≠ J) ∧ (E ≠ K) ∧ (E ≠ L) ∧
    (F ≠ G) ∧ (F ≠ H) ∧ (F ≠ I) ∧ (F ≠ J) ∧ (F ≠ K) ∧ (F ≠ L) ∧
    (G ≠ H) ∧ (G ≠ I) ∧ (G ≠ J) ∧ (G ≠ K) ∧ (G ≠ L) ∧
    (H ≠ I) ∧ (H ≠ J) ∧ (H ≠ K) ∧ (H ≠ L) ∧
    (I ≠ J) ∧ (I ≠ K) ∧ (I ≠ L) ∧
    (J ≠ K) ∧ (J ≠ L) ∧
    (K ≠ L) ∧
    (A + L + F = B + L + G) ∧ (B + L + G = C + L + H) ∧ 
    (C + L + H = D + L + I) ∧ (D + L + I = E + L + J) ∧ 
    (E + L + J = F + L + K) ∧ (F + L + K = A + L + F) :=
sorry

end number_of_ways_to_label_decagon_equal_sums_l121_121840


namespace isosceles_triangle_angles_l121_121850

theorem isosceles_triangle_angles (α β γ : ℝ) (h_iso : α = β ∨ α = γ ∨ β = γ) (h_angle : α + β + γ = 180) (h_40 : α = 40 ∨ β = 40 ∨ γ = 40) :
  (α = 70 ∧ β = 70 ∧ γ = 40) ∨ (α = 40 ∧ β = 100 ∧ γ = 40) ∨ (α = 40 ∧ β = 40 ∧ γ = 100) :=
by
  sorry

end isosceles_triangle_angles_l121_121850


namespace conversion_correct_l121_121486

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.enum.foldl (λ acc ⟨i, digit⟩ => acc + digit * 2^i) 0

def n : List ℕ := [1, 0, 1, 1, 1, 1, 0, 1, 1]

theorem conversion_correct :
  binary_to_decimal n = 379 :=
by 
  sorry

end conversion_correct_l121_121486


namespace ellipse_standard_equation_l121_121795

theorem ellipse_standard_equation :
  ∀ (a b c : ℝ), a = 9 → c = 6 → b = Real.sqrt (a^2 - c^2) →
  (b ≠ 0 ∧ a ≠ 0 → (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)) :=
by
  sorry

end ellipse_standard_equation_l121_121795


namespace simplify_expression_l121_121254

theorem simplify_expression :
  (1 / (Real.sqrt 8 + Real.sqrt 11) +
   1 / (Real.sqrt 11 + Real.sqrt 14) +
   1 / (Real.sqrt 14 + Real.sqrt 17) +
   1 / (Real.sqrt 17 + Real.sqrt 20) +
   1 / (Real.sqrt 20 + Real.sqrt 23) +
   1 / (Real.sqrt 23 + Real.sqrt 26) +
   1 / (Real.sqrt 26 + Real.sqrt 29) +
   1 / (Real.sqrt 29 + Real.sqrt 32)) = 
  (2 * Real.sqrt 2 / 3) :=
by sorry

end simplify_expression_l121_121254


namespace moles_NaOH_to_form_H2O_2_moles_l121_121885

-- Define the reaction and moles involved
def reaction : String := "NH4NO3 + NaOH -> NaNO3 + NH3 + H2O"
def moles_H2O_produced : Nat := 2
def moles_NaOH_required (moles_H2O : Nat) : Nat := moles_H2O

-- Theorem stating the required moles of NaOH to produce 2 moles of H2O
theorem moles_NaOH_to_form_H2O_2_moles : moles_NaOH_required moles_H2O_produced = 2 := 
by
  sorry

end moles_NaOH_to_form_H2O_2_moles_l121_121885


namespace perpendicular_lines_l121_121435

theorem perpendicular_lines (a : ℝ) : (x + 2*y + 1 = 0) ∧ (ax + y - 2 = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l121_121435


namespace simplify_expression_l121_121157

theorem simplify_expression (x : ℤ) : 120 * x - 55 * x = 65 * x := by
  sorry

end simplify_expression_l121_121157


namespace zander_construction_cost_l121_121246

noncomputable def cost_of_cement (num_bags : ℕ) (price_per_bag : ℕ) : ℕ :=
  num_bags * price_per_bag

noncomputable def amount_of_sand (num_lorries : ℕ) (tons_per_lorry : ℕ) : ℕ :=
  num_lorries * tons_per_lorry

noncomputable def cost_of_sand (total_tons : ℕ) (price_per_ton : ℕ) : ℕ :=
  total_tons * price_per_ton

noncomputable def total_cost (cost_cement : ℕ) (cost_sand : ℕ) : ℕ :=
  cost_cement + cost_sand

theorem zander_construction_cost :
  total_cost (cost_of_cement 500 10) (cost_of_sand (amount_of_sand 20 10) 40) = 13000 :=
by
  sorry

end zander_construction_cost_l121_121246


namespace decoded_word_is_correct_l121_121181

-- Assume that we have a way to represent figures and encoded words
structure Figure1
structure Figure2

-- Assume the existence of a key that maps arrow patterns to letters
def decode (f1 : Figure1) (f2 : Figure2) : String := sorry

theorem decoded_word_is_correct (f1 : Figure1) (f2 : Figure2) :
  decode f1 f2 = "КОМПЬЮТЕР" :=
by
  sorry

end decoded_word_is_correct_l121_121181


namespace arrange_polynomial_l121_121031

theorem arrange_polynomial :
  ∀ (x y : ℝ), 2 * x^3 * y - 4 * y^2 + 5 * x^2 = 5 * x^2 + 2 * x^3 * y - 4 * y^2 :=
by
  sorry

end arrange_polynomial_l121_121031


namespace complement_of_A_with_respect_to_U_l121_121405

open Set

-- Definitions
def U : Set ℤ := {-1, 1, 3}
def A : Set ℤ := {-1}

-- Theorem statement
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {1, 3} :=
by
  sorry

end complement_of_A_with_respect_to_U_l121_121405


namespace locus_of_tangency_centers_l121_121468

def locus_of_centers (a b : ℝ) : Prop := 8 * a ^ 2 + 9 * b ^ 2 - 16 * a - 64 = 0

theorem locus_of_tangency_centers (a b : ℝ)
  (hx1 : ∃ x y : ℝ, x ^ 2 + y ^ 2 = 1) 
  (hx2 : ∃ x y : ℝ, (x - 2) ^ 2 + y ^ 2 = 25) 
  (hcent : ∃ r : ℝ, a^2 + b^2 = (r + 1)^2 ∧ (a - 2)^2 + b^2 = (5 - r)^2) : 
  locus_of_centers a b :=
sorry

end locus_of_tangency_centers_l121_121468


namespace parabola_tangent_line_l121_121312

theorem parabola_tangent_line (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b * x + 2 = 2 * x + 3 → a = -1 ∧ b = 4) :=
sorry

end parabola_tangent_line_l121_121312


namespace factor_polynomial_l121_121610

theorem factor_polynomial (a b c : ℝ) : 
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) = (a - b) * (b - c) * (c - a) * (a * b + a * c + b * c) :=
by 
  sorry

end factor_polynomial_l121_121610


namespace online_textbooks_cost_l121_121056

theorem online_textbooks_cost (x : ℕ) :
  (5 * 10) + x + 3 * x = 210 → x = 40 :=
by
  sorry

end online_textbooks_cost_l121_121056


namespace distinct_digit_sum_l121_121949

theorem distinct_digit_sum (a b c d : ℕ) (h1 : a + c = 10) (h2 : b + c = 9) (h3 : a + d = 1)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : a ≠ d) (h7 : b ≠ c) (h8 : b ≠ d) (h9 : c ≠ d)
  (h10 : a < 10) (h11 : b < 10) (h12 : c < 10) (h13 : d < 10)
  (h14 : 0 ≤ a) (h15 : 0 ≤ b) (h16 : 0 ≤ c) (h17 : 0 ≤ d) :
  a + b + c + d = 18 :=
sorry

end distinct_digit_sum_l121_121949


namespace days_worked_together_l121_121661

theorem days_worked_together (W : ℝ) (h1 : ∀ (a b : ℝ), (a + b) * 40 = W) 
                             (h2 : ∀ a, a * 16 = W) 
                             (x : ℝ) 
                             (h3 : (x * (W / 40) + 12 * (W / 16)) = W) : 
                             x = 10 := 
by
  sorry

end days_worked_together_l121_121661


namespace units_digit_base_6_l121_121229

theorem units_digit_base_6 (n m : ℕ) (h₁ : n = 312) (h₂ : m = 67) : (312 * 67) % 6 = 0 :=
by {
  sorry
}

end units_digit_base_6_l121_121229


namespace problem_l121_121631

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem (a : ℝ) (x : ℝ) (hx : x ∈ Set.Ici (-5)) (ha : a = 1) : 
  f x a + x + 5 ≥ -6 / Real.exp 5 := 
sorry

end problem_l121_121631


namespace damaged_potatoes_l121_121035

theorem damaged_potatoes (initial_potatoes : ℕ) (weight_per_bag : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) :
  initial_potatoes = 6500 →
  weight_per_bag = 50 →
  price_per_bag = 72 →
  total_sales = 9144 →
  ∃ damaged_potatoes : ℕ, damaged_potatoes = initial_potatoes - (total_sales / price_per_bag) * weight_per_bag ∧
                               damaged_potatoes = 150 :=
by
  intros _ _ _ _ 
  exact sorry

end damaged_potatoes_l121_121035


namespace remaining_amount_to_be_paid_is_1080_l121_121294

noncomputable def deposit : ℕ := 120
noncomputable def total_price : ℕ := 10 * deposit
noncomputable def remaining_amount : ℕ := total_price - deposit

theorem remaining_amount_to_be_paid_is_1080 :
  remaining_amount = 1080 :=
by
  sorry

end remaining_amount_to_be_paid_is_1080_l121_121294


namespace find_other_root_l121_121258

theorem find_other_root (b : ℝ) (h : ∀ x : ℝ, x^2 - b * x + 3 = 0 → x = 3 ∨ ∃ y, y = 1) :
  ∃ y, y = 1 :=
by
  sorry

end find_other_root_l121_121258


namespace quadratic_less_than_zero_for_x_in_0_1_l121_121989

theorem quadratic_less_than_zero_for_x_in_0_1 (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, 0 < x ∧ x < 1 → (a * x^2 + b * x + c) < 0 :=
by
  sorry

end quadratic_less_than_zero_for_x_in_0_1_l121_121989


namespace solution_system_solution_rational_l121_121886

-- Definitions for the system of equations
def sys_eq_1 (x y : ℤ) : Prop := 2 * x - y = 3
def sys_eq_2 (x y : ℤ) : Prop := x + y = -12

-- Theorem to prove the solution of the system of equations
theorem solution_system (x y : ℤ) (h1 : sys_eq_1 x y) (h2 : sys_eq_2 x y) : x = -3 ∧ y = -9 :=
by {
  sorry
}

-- Definition for the rational equation
def rational_eq (x : ℤ) : Prop := (2 / (1 - x) : ℚ) + 1 = (x / (1 + x) : ℚ)

-- Theorem to prove the solution of the rational equation
theorem solution_rational (x : ℤ) (h : rational_eq x) : x = -3 :=
by {
  sorry
}

end solution_system_solution_rational_l121_121886


namespace sequence_of_arrows_512_to_517_is_B_C_D_E_A_l121_121859

noncomputable def sequence_from_512_to_517 : List Char :=
  let pattern := ['A', 'B', 'C', 'D', 'E']
  pattern.drop 2 ++ pattern.take 2

theorem sequence_of_arrows_512_to_517_is_B_C_D_E_A : sequence_from_512_to_517 = ['B', 'C', 'D', 'E', 'A'] :=
  sorry

end sequence_of_arrows_512_to_517_is_B_C_D_E_A_l121_121859


namespace solve_equation1_solve_equation2_l121_121636

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 2 * x - 4 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 6 = x * (3 - x)

-- State the first proof problem
theorem solve_equation1 (x : ℝ) :
  equation1 x ↔ (x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) := by
  sorry

-- State the second proof problem
theorem solve_equation2 (x : ℝ) :
  equation2 x ↔ (x = 3 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l121_121636


namespace potatoes_cost_l121_121811

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end potatoes_cost_l121_121811


namespace fred_earned_correctly_l121_121730

-- Assuming Fred's earnings from different sources
def fred_earned_newspapers := 16 -- dollars
def fred_earned_cars := 74 -- dollars

-- Total earnings over the weekend
def fred_earnings := fred_earned_newspapers + fred_earned_cars

-- Given condition that Fred earned 90 dollars over the weekend
def fred_earnings_given := 90 -- dollars

-- The theorem stating that Fred's total earnings match the given earnings
theorem fred_earned_correctly : fred_earnings = fred_earnings_given := by
  sorry

end fred_earned_correctly_l121_121730


namespace solve_cubic_diophantine_l121_121289

theorem solve_cubic_diophantine :
  (∃ x y z : ℤ, x^3 + y^3 + z^3 - 3 * x * y * z = 2003) ↔ 
  (x = 667 ∧ y = 668 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 668 ∧ z = 667) :=
sorry

end solve_cubic_diophantine_l121_121289


namespace average_and_variance_of_original_data_l121_121473

theorem average_and_variance_of_original_data (μ σ_sq : ℝ)
  (h1 : 2 * μ - 80 = 1.2)
  (h2 : 4 * σ_sq = 4.4) :
  μ = 40.6 ∧ σ_sq = 1.1 :=
by
  sorry

end average_and_variance_of_original_data_l121_121473


namespace ratio_of_heights_eq_three_twentieths_l121_121812

noncomputable def base_circumference : ℝ := 32 * Real.pi
noncomputable def original_height : ℝ := 60
noncomputable def shorter_volume : ℝ := 768 * Real.pi

theorem ratio_of_heights_eq_three_twentieths
  (base_circumference : ℝ)
  (original_height : ℝ)
  (shorter_volume : ℝ)
  (h' : ℝ)
  (ratio : ℝ) :
  base_circumference = 32 * Real.pi →
  original_height = 60 →
  shorter_volume = 768 * Real.pi →
  (1 / 3 * Real.pi * (base_circumference / (2 * Real.pi))^2 * h') = shorter_volume →
  ratio = h' / original_height →
  ratio = 3 / 20 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end ratio_of_heights_eq_three_twentieths_l121_121812


namespace milk_cost_is_3_l121_121422

def Banana_cost : ℝ := 2
def Sales_tax_rate : ℝ := 0.20
def Total_spent : ℝ := 6

theorem milk_cost_is_3 (Milk_cost : ℝ) :
  Total_spent = (Milk_cost + Banana_cost) + Sales_tax_rate * (Milk_cost + Banana_cost) → 
  Milk_cost = 3 :=
by
  simp [Banana_cost, Sales_tax_rate, Total_spent]
  sorry

end milk_cost_is_3_l121_121422


namespace rose_part_payment_l121_121280

-- Defining the conditions
def total_cost (T : ℝ) := 0.95 * T = 5700
def part_payment (x : ℝ) (T : ℝ) := x = 0.05 * T

-- The proof problem: Prove that the part payment Rose made is $300
theorem rose_part_payment : ∃ T x, total_cost T ∧ part_payment x T ∧ x = 300 :=
by
  sorry

end rose_part_payment_l121_121280


namespace age_of_b_l121_121838

-- Define the conditions as per the problem statement
variables (A B C D E : ℚ)

axiom cond1 : A = B + 2
axiom cond2 : B = 2 * C
axiom cond3 : D = A - 3
axiom cond4 : E = D / 2 + 3
axiom cond5 : A + B + C + D + E = 70

theorem age_of_b : B = 16.625 :=
by {
  -- Placeholder for the proof
  sorry
}

end age_of_b_l121_121838


namespace max_k_value_l121_121950

noncomputable def max_k : ℝ := sorry 

theorem max_k_value :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 1 ∧ y = k * x - 2 ∧  (x - 4)^2 + y^2 ≤ 4) ↔ 
  k ≤ 4 / 3 := sorry

end max_k_value_l121_121950


namespace tea_garden_problem_pruned_to_wild_conversion_l121_121038

-- Definitions and conditions as per the problem statement
def total_area : ℕ := 16
def total_yield : ℕ := 660
def wild_yield_per_mu : ℕ := 30
def pruned_yield_per_mu : ℕ := 50

-- Lean 4 statement as per the proof problem
theorem tea_garden_problem :
  ∃ (x y : ℕ), (x + y = total_area) ∧ (wild_yield_per_mu * x + pruned_yield_per_mu * y = total_yield) ∧
  x = 7 ∧ y = 9 :=
sorry

-- Additional theorem for the conversion condition
theorem pruned_to_wild_conversion :
  ∀ (a : ℕ), (wild_yield_per_mu * (7 + a) ≥ pruned_yield_per_mu * (9 - a)) → a ≥ 3 :=
sorry

end tea_garden_problem_pruned_to_wild_conversion_l121_121038


namespace option_D_correct_l121_121639

noncomputable def y1 (x : ℝ) : ℝ := 1 / x
noncomputable def y2 (x : ℝ) : ℝ := x^2
noncomputable def y3 (x : ℝ) : ℝ := (1 / 2)^x
noncomputable def y4 (x : ℝ) : ℝ := 1 / x^2

theorem option_D_correct :
  (∀ x : ℝ, y4 x = y4 (-x)) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → y4 x₁ > y4 x₂) :=
by
  sorry

end option_D_correct_l121_121639


namespace selling_price_is_correct_l121_121309

def wholesale_cost : ℝ := 24.35
def gross_profit_percentage : ℝ := 0.15

def gross_profit : ℝ := gross_profit_percentage * wholesale_cost
def selling_price : ℝ := wholesale_cost + gross_profit

theorem selling_price_is_correct :
  selling_price = 28.00 :=
by
  sorry

end selling_price_is_correct_l121_121309


namespace infinite_series_sum_l121_121162

noncomputable def sum_geometric_series (a b : ℝ) (h : ∑' n : ℕ, a / b ^ (n + 1) = 3) : ℝ :=
  ∑' n : ℕ, a / b ^ (n + 1)

theorem infinite_series_sum (a b c : ℝ) (h : sum_geometric_series a b (by sorry) = 3) :
  ∑' n : ℕ, (c * a) / (a + b) ^ (n + 1) = 3 * c / 4 :=
sorry

end infinite_series_sum_l121_121162


namespace boat_speed_in_still_water_l121_121097

-- Define the conditions
def speed_of_stream : ℝ := 3 -- (speed in km/h)
def time_downstream : ℝ := 1 -- (time in hours)
def time_upstream : ℝ := 1.5 -- (time in hours)

-- Define the goal by proving the speed of the boat in still water
theorem boat_speed_in_still_water : 
  ∃ V_b : ℝ, (V_b + speed_of_stream) * time_downstream = (V_b - speed_of_stream) * time_upstream ∧ V_b = 15 :=
by
  sorry -- (Proof will be provided here)

end boat_speed_in_still_water_l121_121097


namespace volume_of_cuboid_is_250_cm3_l121_121004

-- Define the edge length of the cube
def edge_length (a : ℕ) : ℕ := 5

-- Define the volume of a single cube
def cube_volume := (edge_length 5) ^ 3

-- Define the total volume of the cuboid formed by placing two such cubes in a line
def cuboid_volume := 2 * cube_volume

-- Theorem stating the volume of the cuboid formed
theorem volume_of_cuboid_is_250_cm3 : cuboid_volume = 250 := by
  sorry

end volume_of_cuboid_is_250_cm3_l121_121004


namespace carlos_cycles_more_than_diana_l121_121659

theorem carlos_cycles_more_than_diana :
  let slope_carlos := 1
  let slope_diana := 0.75
  let rate_carlos := slope_carlos * 20
  let rate_diana := slope_diana * 20
  let distance_carlos_after_3_hours := 3 * rate_carlos
  let distance_diana_after_3_hours := 3 * rate_diana
  distance_carlos_after_3_hours - distance_diana_after_3_hours = 15 :=
sorry

end carlos_cycles_more_than_diana_l121_121659


namespace saree_stripes_l121_121084

theorem saree_stripes
  (G : ℕ) (B : ℕ) (Br : ℕ) (total_stripes : ℕ) (total_patterns : ℕ)
  (h1 : G = 3 * Br)
  (h2 : B = 5 * G)
  (h3 : Br = 4)
  (h4 : B + G + Br = 100)
  (h5 : total_stripes = 100)
  (h6 : total_patterns = total_stripes / 3) :
  B = 84 ∧ total_patterns = 33 := 
  by {
    sorry
  }

end saree_stripes_l121_121084


namespace problem_l121_121902

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.log x + (a + 1) * (1 / x - 2)

theorem problem (a x : ℝ) (ha_pos : a > 0) :
  f a x > - (a^2 / (a + 1)) - 2 :=
sorry

end problem_l121_121902


namespace diane_trip_length_l121_121520

-- Define constants and conditions
def first_segment_fraction : ℚ := 1 / 4
def middle_segment_length : ℚ := 24
def last_segment_fraction : ℚ := 1 / 3

def total_trip_length (x : ℚ) : Prop :=
  (1 - first_segment_fraction - last_segment_fraction) * x = middle_segment_length

theorem diane_trip_length : ∃ x : ℚ, total_trip_length x ∧ x = 57.6 := by
  sorry

end diane_trip_length_l121_121520


namespace yoki_cans_l121_121492

-- Definitions of the conditions
def total_cans_collected : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_cans := avi_initial_cans / 2

-- Statement that needs to be proved
theorem yoki_cans : ∀ (total_cans_collected ladonna_cans : ℕ) 
  (prikya_cans : ℕ := 2 * ladonna_cans) 
  (avi_initial_cans : ℕ := 8) 
  (avi_cans : ℕ := avi_initial_cans / 2), 
  (total_cans_collected = 85) → 
  (ladonna_cans = 25) → 
  (prikya_cans = 2 * ladonna_cans) →
  (avi_initial_cans = 8) → 
  (avi_cans = avi_initial_cans / 2) → 
  total_cans_collected - (ladonna_cans + prikya_cans + avi_cans) = 6 :=
by
  intros total_cans_collected ladonna_cans prikya_cans avi_initial_cans avi_cans H1 H2 H3 H4 H5
  sorry

end yoki_cans_l121_121492


namespace sale_in_2nd_month_l121_121425

-- Defining the variables for the sales in the months
def sale_in_1st_month : ℝ := 6435
def sale_in_3rd_month : ℝ := 7230
def sale_in_4th_month : ℝ := 6562
def sale_in_5th_month : ℝ := 6855
def required_sale_in_6th_month : ℝ := 5591
def required_average_sale : ℝ := 6600
def number_of_months : ℝ := 6
def total_sales_needed : ℝ := required_average_sale * number_of_months

-- Proof statement
theorem sale_in_2nd_month : sale_in_1st_month + x + sale_in_3rd_month + sale_in_4th_month + sale_in_5th_month + required_sale_in_6th_month = total_sales_needed → x = 6927 :=
by
  sorry

end sale_in_2nd_month_l121_121425


namespace value_of_b_l121_121152

theorem value_of_b (a b : ℕ) (q : ℝ)
  (h1 : q = 0.5)
  (h2 : a = 2020)
  (h3 : q = a / b) : b = 4040 := by
  sorry

end value_of_b_l121_121152


namespace tan_add_sin_l121_121423

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end tan_add_sin_l121_121423


namespace sqrt_expr_eq_l121_121977

theorem sqrt_expr_eq : (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 :=
by sorry

end sqrt_expr_eq_l121_121977


namespace find_a8_l121_121001

/-!
Let {a_n} be an arithmetic sequence, with S_n representing the sum of the first n terms.
Given:
1. S_6 = 8 * S_3
2. a_3 - a_5 = 8
Prove: a_8 = -26
-/

noncomputable def arithmetic_seq (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_seq (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem find_a8 (a_1 d : ℤ)
  (h1 : sum_arithmetic_seq a_1 d 6 = 8 * sum_arithmetic_seq a_1 d 3)
  (h2 : arithmetic_seq a_1 d 3 - arithmetic_seq a_1 d 5 = 8) :
  arithmetic_seq a_1 d 8 = -26 :=
  sorry

end find_a8_l121_121001


namespace S_21_equals_4641_l121_121275

-- Define the first element of the nth set
def first_element_of_set (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

-- Define the last element of the nth set
def last_element_of_set (n : ℕ) : ℕ :=
  (first_element_of_set n) + n - 1

-- Define the sum of the nth set
def S (n : ℕ) : ℕ :=
  n * ((first_element_of_set n) + (last_element_of_set n)) / 2

-- The goal statement we want to prove
theorem S_21_equals_4641 : S 21 = 4641 := by
  sorry

end S_21_equals_4641_l121_121275


namespace correct_expression_l121_121942

theorem correct_expression (a b c : ℝ) : 3 * a - (2 * b - c) = 3 * a - 2 * b + c :=
sorry

end correct_expression_l121_121942


namespace curve_symmetry_l121_121550

-- Define the curve equation
def curve_eq (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define the symmetry condition about the line y = -x
def symmetry_about_y_equals_neg_x (x y : ℝ) : Prop :=
  curve_eq (-y) (-x)

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop := curve_eq x y

-- Proof statement: The curve xy^2 - x^2y = -2 is symmetric about the line y = -x.
theorem curve_symmetry : ∀ (x y : ℝ), original_curve x y ↔ symmetry_about_y_equals_neg_x x y :=
by
  sorry

end curve_symmetry_l121_121550


namespace triangle_sine_cosine_l121_121805

theorem triangle_sine_cosine (a b A : ℝ) (B C : ℝ) (c : ℝ) 
  (ha : a = Real.sqrt 7) 
  (hb : b = 2) 
  (hA : A = 60 * Real.pi / 180) 
  (hsinB : Real.sin B = Real.sin B := by sorry)
  (hc : c = 3 := by sorry) :
  (Real.sin B = Real.sqrt 21 / 7) ∧ (c = 3) := 
sorry

end triangle_sine_cosine_l121_121805


namespace polynomial_remainder_l121_121441

theorem polynomial_remainder (a b : ℝ) (h : ∀ x : ℝ, (x^3 - 2*x^2 + a*x + b) % ((x - 1)*(x - 2)) = 2*x + 1) : 
  a = 1 ∧ b = 3 := 
sorry

end polynomial_remainder_l121_121441


namespace cubic_roots_natural_numbers_l121_121397

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end cubic_roots_natural_numbers_l121_121397


namespace plane_equation_l121_121253

def point := ℝ × ℝ × ℝ
def vector := ℝ × ℝ × ℝ

def point_on_plane (P : point) (a b c d : ℝ) : Prop :=
  match P with
  | (x, y, z) => a * x + b * y + c * z + d = 0

def normal_to_plane (n : vector) (a b c : ℝ) : Prop :=
  match n with
  | (nx, ny, nz) => (a, b, c) = (nx, ny, nz)

theorem plane_equation
  (P₀ : point) (u : vector)
  (x₀ y₀ z₀ : ℝ) (a b c d : ℝ)
  (h1 : P₀ = (1, 2, 1))
  (h2 : u = (-2, 1, 3))
  (h3 : point_on_plane (1, 2, 1) a b c d)
  (h4 : normal_to_plane (-2, 1, 3) a b c)
  : (2 : ℝ) * (x₀ : ℝ) - (y₀ : ℝ) - (3 : ℝ) * (z₀ : ℝ) + (3 : ℝ) = 0 :=
sorry

end plane_equation_l121_121253


namespace max_value_of_quadratic_at_2_l121_121451

def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

theorem max_value_of_quadratic_at_2 : ∃ (x : ℝ), x = 2 ∧ ∀ y : ℝ, f y ≤ f x :=
by
  use 2
  sorry

end max_value_of_quadratic_at_2_l121_121451


namespace circumscribed_center_on_Ox_axis_l121_121952

-- Define the quadratic equation
noncomputable def quadratic_eq (p x : ℝ) : ℝ := 2^p * x^2 + 5 * p * x - 2^(p^2)

-- Define the conditions for the problem
def intersects_Ox (p : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq p x1 = 0 ∧ quadratic_eq p x2 = 0 ∧ x1 ≠ x2

def intersects_Oy (p : ℝ) : Prop := quadratic_eq p 0 = -2^(p^2)

-- Define the problem statement
theorem circumscribed_center_on_Ox_axis :
  (∀ p : ℝ, intersects_Ox p ∧ intersects_Oy p → (p = 0 ∨ p = -1)) →
  (0 + (-1) = -1) :=
sorry

end circumscribed_center_on_Ox_axis_l121_121952


namespace least_subtracted_number_l121_121235

def is_sum_of_digits_at_odd_places (n : ℕ) : ℕ :=
  (n / 100000) % 10 + (n / 1000) % 10 + (n / 10) % 10

def is_sum_of_digits_at_even_places (n : ℕ) : ℕ :=
  (n / 10000) % 10 + (n / 100) % 10 + (n % 10)

def diff_digits_odd_even (n : ℕ) : ℕ :=
  is_sum_of_digits_at_odd_places n - is_sum_of_digits_at_even_places n

theorem least_subtracted_number :
  ∃ x : ℕ, (427398 - x) % 11 = 0 ∧ x = 7 :=
by
  sorry

end least_subtracted_number_l121_121235


namespace area_of_circle_l121_121011

theorem area_of_circle (C : ℝ) (hC : C = 36 * Real.pi) : 
  ∃ k : ℝ, (∃ r : ℝ, r = 18 ∧ k = r^2 ∧ (pi * r^2 = k * pi)) ∧ k = 324 :=
by
  sorry

end area_of_circle_l121_121011


namespace acute_angle_30_l121_121763

theorem acute_angle_30 (α : ℝ) (h : Real.cos (π / 6) * Real.sin α = Real.sqrt 3 / 4) : α = π / 6 := 
by 
  sorry

end acute_angle_30_l121_121763


namespace problem_statement_l121_121986

noncomputable def nonreal_omega_root (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω^2 + ω + 1 = 0

theorem problem_statement (ω : ℂ) (h : nonreal_omega_root ω) :
  (1 - 2 * ω + ω^2)^6 + (1 + 2 * ω - ω^2)^6 = 1458 :=
sorry

end problem_statement_l121_121986


namespace dryer_cost_l121_121080

theorem dryer_cost (washer_dryer_total_cost washer_cost dryer_cost : ℝ) (h1 : washer_dryer_total_cost = 1200) (h2 : washer_cost = dryer_cost + 220) :
  dryer_cost = 490 :=
by
  sorry

end dryer_cost_l121_121080


namespace volume_of_extended_parallelepiped_l121_121021

theorem volume_of_extended_parallelepiped :
  let main_box_volume := 3 * 3 * 6
  let external_boxes_volume := 2 * (3 * 3 * 1 + 3 * 6 * 1 + 3 * 6 * 1)
  let spheres_volume := 8 * (1 / 8) * (4 / 3) * Real.pi * (1 ^ 3)
  let cylinders_volume := 12 * (1 / 4) * Real.pi * 1^2 * 3 + 12 * (1 / 4) * Real.pi * 1^2 * 6
  main_box_volume + external_boxes_volume + spheres_volume + cylinders_volume = (432 + 52 * Real.pi) / 3 :=
by
  sorry

end volume_of_extended_parallelepiped_l121_121021


namespace smallest_positive_integer_solution_l121_121365

theorem smallest_positive_integer_solution : ∃ n : ℕ, 23 * n % 9 = 310 % 9 ∧ n = 8 :=
by
  sorry

end smallest_positive_integer_solution_l121_121365


namespace total_shaded_area_approx_l121_121718

noncomputable def area_of_shaded_regions (r1 r2 : ℝ) :=
  let area_smaller_circle := 3 * 6 - (1 / 2) * Real.pi * r1^2
  let area_larger_circle := 6 * 12 - (1 / 2) * Real.pi * r2^2
  area_smaller_circle + area_larger_circle

theorem total_shaded_area_approx :
  abs (area_of_shaded_regions 3 6 - 19.4) < 0.05 :=
by
  sorry

end total_shaded_area_approx_l121_121718


namespace sum_of_coordinates_B_l121_121734

theorem sum_of_coordinates_B
  (x y : ℤ)
  (Mx My : ℤ)
  (Ax Ay : ℤ)
  (M : Mx = 2 ∧ My = -3)
  (A : Ax = -4 ∧ Ay = -5)
  (midpoint_x : (x + Ax) / 2 = Mx)
  (midpoint_y : (y + Ay) / 2 = My) :
  x + y = 7 :=
by
  sorry

end sum_of_coordinates_B_l121_121734


namespace apricot_tea_calories_l121_121201

theorem apricot_tea_calories :
  let apricot_juice_weight := 150
  let apricot_juice_calories_per_100g := 30
  let honey_weight := 50
  let honey_calories_per_100g := 304
  let water_weight := 300
  let apricot_tea_weight := apricot_juice_weight + honey_weight + water_weight
  let apricot_juice_calories := apricot_juice_weight * apricot_juice_calories_per_100g / 100
  let honey_calories := honey_weight * honey_calories_per_100g / 100
  let total_calories := apricot_juice_calories + honey_calories
  let caloric_density := total_calories / apricot_tea_weight
  let tea_weight := 250
  let calories_in_250g_tea := tea_weight * caloric_density
  calories_in_250g_tea = 98.5 := by
  sorry

end apricot_tea_calories_l121_121201


namespace max_parrots_l121_121919

-- Define the parameters and conditions for the problem
def N : ℕ := 2018
def Y : ℕ := 1009
def number_of_islanders (R L P : ℕ) := R + L + P = N

-- Define the main theorem
theorem max_parrots (R L P : ℕ) (h : number_of_islanders R L P) (hY : Y = 1009) :
  P = 1009 :=
sorry

end max_parrots_l121_121919


namespace evaluate_fractions_l121_121645

theorem evaluate_fractions (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 35 :=
by
  sorry

end evaluate_fractions_l121_121645


namespace intersection_point_l121_121158

theorem intersection_point : 
  ∃ (x y : ℚ), y = - (5/3 : ℚ) * x ∧ y + 3 = 15 * x - 6 ∧ x = 27 / 50 ∧ y = - 9 / 10 := 
by
  sorry

end intersection_point_l121_121158


namespace base_representing_350_as_four_digit_number_with_even_final_digit_l121_121724

theorem base_representing_350_as_four_digit_number_with_even_final_digit {b : ℕ} :
  b ^ 3 ≤ 350 ∧ 350 < b ^ 4 ∧ (∃ d1 d2 d3 d4, 350 = d1 * b^3 + d2 * b^2 + d3 * b + d4 ∧ d4 % 2 = 0) ↔ b = 6 :=
by sorry

end base_representing_350_as_four_digit_number_with_even_final_digit_l121_121724


namespace swimming_pool_length_correct_l121_121825

noncomputable def swimming_pool_length (V_removed: ℝ) (W: ℝ) (H: ℝ) (gal_to_cuft: ℝ): ℝ :=
  V_removed / (W * H / gal_to_cuft)

theorem swimming_pool_length_correct:
  swimming_pool_length 3750 25 0.5 7.48052 = 40.11 :=
by
  sorry

end swimming_pool_length_correct_l121_121825


namespace curve_is_line_segment_l121_121858

noncomputable def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = Real.cos θ ^ 2 ∧ p.2 = Real.sin θ ^ 2}

theorem curve_is_line_segment :
  (∀ p ∈ parametric_curve, p.1 + p.2 = 1 ∧ p.1 ∈ Set.Icc 0 1) :=
by
  sorry

end curve_is_line_segment_l121_121858


namespace max_value_of_expression_l121_121358

noncomputable def max_expression_value (x y : ℝ) :=
  x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  max_expression_value x y ≤ 961 / 8 :=
sorry

end max_value_of_expression_l121_121358


namespace problem1_problem2_problem3_problem4_l121_121586

-- Problem 1
theorem problem1 : (-3 : ℝ) ^ 2 + (1 / 2) ^ (-1 : ℝ) + (Real.pi - 3) ^ 0 = 12 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (8 * x ^ 4 + 4 * x ^ 3 - x ^ 2) / (-2 * x) ^ 2 = 2 * x ^ 2 + x - 1 / 4 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (2 * x + 1) ^ 2 - (4 * x + 1) * (x + 1) = -x :=
by
  sorry

-- Problem 4
theorem problem4 (x y : ℝ) : (x + 2 * y - 3) * (x - 2 * y + 3) = x ^ 2 - 4 * y ^ 2 + 12 * y - 9 :=
by
  sorry

end problem1_problem2_problem3_problem4_l121_121586


namespace arrangement_exists_l121_121803

-- Definitions of pairwise coprimeness and gcd
def pairwise_coprime (a b c d : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1

def common_divisor (x y : ℕ) : Prop := ∃ d > 1, d ∣ x ∧ d ∣ y

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

-- Main theorem statement
theorem arrangement_exists :
  ∃ a b c d ab cd ad bc abcd : ℕ,
    pairwise_coprime a b c d ∧
    ab = a * b ∧ cd = c * d ∧ ad = a * d ∧ bc = b * c ∧ abcd = a * b * c * d ∧
    (common_divisor ab abcd ∧ common_divisor cd abcd ∧ common_divisor ad abcd ∧ common_divisor bc abcd) ∧
    (common_divisor ab ad ∧ common_divisor ab bc ∧ common_divisor cd ad ∧ common_divisor cd bc) ∧
    (relatively_prime ab cd ∧ relatively_prime ad bc) :=
by
  -- The proof will be filled here
  sorry

end arrangement_exists_l121_121803


namespace final_number_is_correct_l121_121503

def initial_number := 9
def doubled_number (x : ℕ) := x * 2
def added_number (x : ℕ) := x + 13
def trebled_number (x : ℕ) := x * 3

theorem final_number_is_correct : trebled_number (added_number (doubled_number initial_number)) = 93 := by
  sorry

end final_number_is_correct_l121_121503


namespace calculate_expression_l121_121662

def smallest_positive_two_digit_multiple_of_7 : ℕ := 14
def smallest_positive_three_digit_multiple_of_5 : ℕ := 100

theorem calculate_expression : 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  (c * d) - 100 = 1300 :=
by 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  sorry

end calculate_expression_l121_121662


namespace product_of_p_r_s_l121_121444

theorem product_of_p_r_s :
  ∃ p r s : ℕ, 3^p + 3^5 = 252 ∧ 2^r + 58 = 122 ∧ 5^3 * 6^s = 117000 ∧ p * r * s = 36 :=
by
  sorry

end product_of_p_r_s_l121_121444


namespace geom_series_common_ratio_l121_121259

theorem geom_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hNewS : (ar^3) / (1 - r) = S / 27) : r = 1 / 3 :=
by
  sorry

end geom_series_common_ratio_l121_121259


namespace average_speed_of_car_l121_121744

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end average_speed_of_car_l121_121744


namespace combined_resistance_l121_121651

theorem combined_resistance (x y r : ℝ) (hx : x = 5) (hy : y = 7) (h_parallel : 1 / r = 1 / x + 1 / y) : 
  r = 35 / 12 := 
by 
  sorry

end combined_resistance_l121_121651


namespace grocer_display_proof_l121_121454

-- Define the arithmetic sequence conditions
def num_cans_in_display (n : ℕ) : Prop :=
  let a := 1
  let d := 2
  (n * n = 225) 

-- Prove the total weight is 1125 kg
def total_weight_supported (weight_per_can : ℕ) (total_cans : ℕ) : Prop :=
  (total_cans * weight_per_can = 1125)

-- State the main theorem combining the two proofs.
theorem grocer_display_proof (n weight_per_can total_cans : ℕ) :
  num_cans_in_display n → total_weight_supported weight_per_can total_cans → 
  n = 15 ∧ total_cans * weight_per_can = 1125 :=
by {
  sorry
}

end grocer_display_proof_l121_121454


namespace range_neg2a_plus_3_l121_121536

theorem range_neg2a_plus_3 (a : ℝ) (h : a < 1) : -2 * a + 3 > 1 :=
sorry

end range_neg2a_plus_3_l121_121536


namespace remainder_div_P_by_D_plus_D_l121_121485

theorem remainder_div_P_by_D_plus_D' 
  (P Q D R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D + D') = R :=
by
  -- Proof is not required.
  sorry

end remainder_div_P_by_D_plus_D_l121_121485


namespace range_of_linear_function_l121_121899

theorem range_of_linear_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  3 < -2 * x + 5 ∧ -2 * x + 5 < 7 :=
by {
  sorry
}

end range_of_linear_function_l121_121899


namespace a_can_be_any_real_l121_121829

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : e ≠ 0) :
  ∃ a : ℝ, true :=
by sorry

end a_can_be_any_real_l121_121829


namespace average_messages_correct_l121_121335

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

end average_messages_correct_l121_121335


namespace pool_capacity_l121_121860

noncomputable def total_capacity : ℝ := 1000

theorem pool_capacity
    (C : ℝ)
    (H1 : 0.75 * C = 0.45 * C + 300)
    (H2 : 300 / 0.3 = 1000)
    : C = total_capacity :=
by
  -- Solution steps are omitted, proof goes here.
  sorry

end pool_capacity_l121_121860


namespace percentage_of_bottle_danny_drank_l121_121670

theorem percentage_of_bottle_danny_drank
    (x : ℝ)  -- percentage of the first bottle Danny drinks, represented as a real number
    (b1 b2 b3 : ℝ)  -- volumes of the three bottles, represented as real numbers
    (h_b1 : b1 = 1)  -- first bottle is full (1 bottle)
    (h_b2 : b2 = 1)  -- second bottle is full (1 bottle)
    (h_b3 : b3 = 1)  -- third bottle is full (1 bottle)
    (h_given_away1 : b2 * 0.7 = 0.7)  -- gave away 70% of the second bottle
    (h_given_away2 : b3 * 0.7 = 0.7)  -- gave away 70% of the third bottle
    (h_soda_left : b1 * (1 - x) + b2 * 0.3 + b3 * 0.3 = 0.7)  -- 70% of bottle left
    : x = 0.9 :=
by
  sorry

end percentage_of_bottle_danny_drank_l121_121670


namespace degrees_to_minutes_l121_121264

theorem degrees_to_minutes (d : ℚ) (fractional_part : ℚ) (whole_part : ℤ) :
  1 ≤ d ∧ d = fractional_part + whole_part ∧ fractional_part = 0.45 ∧ whole_part = 1 →
  (whole_part + fractional_part) * 60 = 1 * 60 + 27 :=
by { sorry }

end degrees_to_minutes_l121_121264


namespace common_divisor_l121_121896

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end common_divisor_l121_121896


namespace water_left_l121_121412

theorem water_left (initial_water: ℚ) (science_experiment_use: ℚ) (plant_watering_use: ℚ)
  (h1: initial_water = 3)
  (h2: science_experiment_use = 5 / 4)
  (h3: plant_watering_use = 1 / 2) :
  (initial_water - science_experiment_use - plant_watering_use = 5 / 4) :=
by
  rw [h1, h2, h3]
  norm_num

end water_left_l121_121412


namespace total_amount_paid_l121_121145

def jacket_price : ℝ := 150
def sale_discount : ℝ := 0.25
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_amount_paid : 
  (jacket_price * (1 - sale_discount) - coupon_discount) * (1 + sales_tax) = 112.75 := 
by
  sorry

end total_amount_paid_l121_121145


namespace lower_denomination_cost_l121_121818

-- Conditions
def total_stamps : ℕ := 20
def total_cost_cents : ℕ := 706
def high_denomination_stamps : ℕ := 18
def high_denomination_cost : ℕ := 37
def low_denomination_stamps : ℕ := total_stamps - high_denomination_stamps

-- Theorem proving the cost of the lower denomination stamp.
theorem lower_denomination_cost :
  ∃ (x : ℕ), (high_denomination_stamps * high_denomination_cost) + (low_denomination_stamps * x) = total_cost_cents
  ∧ x = 20 :=
by
  use 20
  sorry

end lower_denomination_cost_l121_121818


namespace cupcake_ratio_l121_121923

theorem cupcake_ratio (C B : ℕ) (hC : C = 4) (hTotal : C + B = 12) : B / C = 2 :=
by
  sorry

end cupcake_ratio_l121_121923


namespace inequality_satisfaction_l121_121107

theorem inequality_satisfaction (x y : ℝ) : 
  y - x < Real.sqrt (x^2) ↔ (y < 0 ∨ y < 2 * x) := by 
sorry

end inequality_satisfaction_l121_121107


namespace mary_age_proof_l121_121281

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof_l121_121281


namespace number_of_goats_l121_121146

-- Mathematical definitions based on the conditions
def number_of_hens : ℕ := 10
def total_cost : ℤ := 2500
def price_per_hen : ℤ := 50
def price_per_goat : ℤ := 400

-- Prove the number of goats
theorem number_of_goats (G : ℕ) : 
  number_of_hens * price_per_hen + G * price_per_goat = total_cost ↔ G = 5 := 
by
  sorry

end number_of_goats_l121_121146


namespace Hallie_earnings_l121_121418

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end Hallie_earnings_l121_121418


namespace answer_one_answer_two_answer_three_l121_121042

def point_condition (A B : ℝ) (P : ℝ) (k : ℝ) : Prop := |A - P| = k * |B - P|

def question_one : Prop :=
  let A := -3
  let B := 6
  let k := 2
  let P := 3
  point_condition A B P k

def question_two : Prop :=
  ∀ x k : ℝ, |x + 2| + |x - 1| = 3 → point_condition (-3) 6 x k → (1 / 8 ≤ k ∧ k ≤ 4 / 5)

def question_three : Prop :=
  let A := -3
  let B := 6
  ∃ t : ℝ, t = 3 / 2 ∧ point_condition A (-3 + t) (6 - 2 * t) 3

theorem answer_one : question_one := by sorry

theorem answer_two : question_two := by sorry

theorem answer_three : question_three := by sorry

end answer_one_answer_two_answer_three_l121_121042


namespace original_number_exists_l121_121168

theorem original_number_exists : 
  ∃ (t o : ℕ), (10 * t + o = 74) ∧ (t = o * o - 9) ∧ (10 * o + t = 10 * t + o - 27) :=
by
  sorry

end original_number_exists_l121_121168


namespace bucket_full_weight_l121_121222

variables (x y p q : Real)

theorem bucket_full_weight (h1 : x + (1 / 4) * y = p)
                           (h2 : x + (3 / 4) * y = q) :
    x + y = 3 * q - p :=
by
  sorry

end bucket_full_weight_l121_121222


namespace greatest_common_divisor_of_120_and_m_l121_121463

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end greatest_common_divisor_of_120_and_m_l121_121463


namespace four_digit_flippies_div_by_4_l121_121541

def is_flippy (n : ℕ) : Prop := 
  let digits := [4, 6]
  n / 1000 ∈ digits ∧
  (n / 100 % 10) ∈ digits ∧
  ((n / 10 % 10) = if (n / 100 % 10) = 4 then 6 else 4) ∧
  (n % 10) = if (n / 1000) = 4 then 6 else 4

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

theorem four_digit_flippies_div_by_4 : 
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_flippy n ∧ is_divisible_by_4 n :=
by
  sorry

end four_digit_flippies_div_by_4_l121_121541


namespace exposed_sides_correct_l121_121134

-- Define the number of sides of each polygon
def sides_triangle := 3
def sides_square := 4
def sides_pentagon := 5
def sides_hexagon := 6
def sides_heptagon := 7

-- Total sides from all polygons
def total_sides := sides_triangle + sides_square + sides_pentagon + sides_hexagon + sides_heptagon

-- Number of shared sides
def shared_sides := 4

-- Final number of exposed sides
def exposed_sides := total_sides - shared_sides

-- Statement to prove
theorem exposed_sides_correct : exposed_sides = 21 :=
by {
  -- This part will contain the proof which we do not need. Replace with 'sorry' for now.
  sorry
}

end exposed_sides_correct_l121_121134


namespace factorial_ratio_integer_l121_121944

theorem factorial_ratio_integer (m n : ℕ) : 
    (m ≥ 0) → (n ≥ 0) → ∃ k : ℤ, k = (2 * m).factorial * (2 * n).factorial / ((m.factorial * n.factorial * (m + n).factorial) : ℝ) :=
by
  sorry

end factorial_ratio_integer_l121_121944


namespace sum_of_solutions_l121_121313

theorem sum_of_solutions (y x : ℝ) (h1 : y = 7) (h2 : x^2 + y^2 = 100) : 
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_l121_121313


namespace part_one_part_two_l121_121079

noncomputable def a (n : ℕ) : ℚ := if n = 1 then 1 / 2 else 2 ^ (n - 1) / (1 + 2 ^ (n - 1))

noncomputable def b (n : ℕ) : ℚ := n / a n

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => b (i + 1))

/-Theorem:
1. Prove that for all n > 0, a(n) = 2^(n-1) / (1 + 2^(n-1)).
2. Prove that for all n ≥ 3, S(n) > n^2 / 2 + 4.
-/
theorem part_one (n : ℕ) (h : n > 0) : a n = 2 ^ (n - 1) / (1 + 2 ^ (n - 1)) := sorry

theorem part_two (n : ℕ) (h : n ≥ 3) : S n > n ^ 2 / 2 + 4 := sorry

end part_one_part_two_l121_121079


namespace jiaqi_grade_is_95_3_l121_121898

def extracurricular_score : ℝ := 96
def mid_term_score : ℝ := 92
def final_exam_score : ℝ := 97

def extracurricular_weight : ℝ := 0.2
def mid_term_weight : ℝ := 0.3
def final_exam_weight : ℝ := 0.5

def total_grade : ℝ :=
  extracurricular_score * extracurricular_weight +
  mid_term_score * mid_term_weight +
  final_exam_score * final_exam_weight

theorem jiaqi_grade_is_95_3 : total_grade = 95.3 :=
by
  simp [total_grade, extracurricular_score, mid_term_score, final_exam_score,
    extracurricular_weight, mid_term_weight, final_exam_weight]
  sorry

end jiaqi_grade_is_95_3_l121_121898


namespace restaurant_total_cost_l121_121620

def total_cost
  (adults kids : ℕ)
  (adult_meal_cost adult_drink_cost adult_dessert_cost kid_drink_cost kid_dessert_cost : ℝ) : ℝ :=
  let num_adults := adults
  let num_kids := kids
  let adult_total := num_adults * (adult_meal_cost + adult_drink_cost + adult_dessert_cost)
  let kid_total := num_kids * (kid_drink_cost + kid_dessert_cost)
  adult_total + kid_total

theorem restaurant_total_cost :
  total_cost 4 9 7 4 3 2 1.5 = 87.5 :=
by
  sorry

end restaurant_total_cost_l121_121620


namespace proof_problem_l121_121138

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l121_121138


namespace compound_interest_years_l121_121053

-- Define the parameters
def principal : ℝ := 7500
def future_value : ℝ := 8112
def annual_rate : ℝ := 0.04
def compounding_periods : ℕ := 1

-- Define the proof statement
theorem compound_interest_years :
  ∃ t : ℕ, future_value = principal * (1 + annual_rate / compounding_periods) ^ t ∧ t = 2 :=
by
  sorry

end compound_interest_years_l121_121053


namespace households_using_all_three_brands_correct_l121_121581

noncomputable def total_households : ℕ := 5000
noncomputable def non_users : ℕ := 1200
noncomputable def only_X : ℕ := 800
noncomputable def only_Y : ℕ := 600
noncomputable def only_Z : ℕ := 300

-- Let A be the number of households that used all three brands of soap
variable (A : ℕ)

-- For every household that used all three brands, 5 used only two brands and 10 used just one brand.
-- Number of households that used only two brands = 5 * A
-- Number of households that used only one brand = 10 * A

-- The equation for households that used just one brand:
def households_using_all_three_brands :=
10 * A = only_X + only_Y + only_Z

theorem households_using_all_three_brands_correct :
  (total_households - non_users = only_X + only_Y + only_Z + 5 * A + 10 * A) →
  (A = 170) := by
sorry

end households_using_all_three_brands_correct_l121_121581


namespace phone_cost_l121_121212

theorem phone_cost (C : ℝ) (h1 : 0.40 * C + 780 = C) : C = 1300 := by
  sorry

end phone_cost_l121_121212


namespace solve_f_435_l121_121726

variable (f : ℝ → ℝ)

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (3 - x) = f x

-- To Prove
theorem solve_f_435 : f 435 = 0 :=
by
  sorry

end solve_f_435_l121_121726


namespace find_horizontal_length_l121_121122

variable (v h : ℝ)

-- Conditions
def is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 (v h : ℝ) : Prop :=
  2 * h + 2 * v = 54 ∧ h = v + 3

-- The proof we aim to show
theorem find_horizontal_length (v h : ℝ) :
  is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 v h → h = 15 :=
by
  sorry

end find_horizontal_length_l121_121122


namespace digit_swap_division_l121_121429

theorem digit_swap_division (ab ba : ℕ) (k1 k2 : ℤ) (a b : ℕ) :
  (ab = 10 * a + b) ∧ (ba = 10 * b + a) →
  (ab % 7 = 1) ∧ (ba % 7 = 1) →
  ∃ n, n = 4 :=
by
  sorry

end digit_swap_division_l121_121429


namespace abs_inequality_solution_set_l121_121314

theorem abs_inequality_solution_set {x : ℝ} : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end abs_inequality_solution_set_l121_121314


namespace difference_of_scores_l121_121630

variable {x y : ℝ}

theorem difference_of_scores (h : x / y = 4) : x - y = 3 * y := by
  sorry

end difference_of_scores_l121_121630


namespace number_of_solutions_l121_121248

theorem number_of_solutions :
  ∃ (sols : Finset ℝ), 
    (∀ x, x ∈ sols → 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 2 * (Real.sin x)^3 - 5 * (Real.sin x)^2 + 2 * Real.sin x = 0) 
    ∧ Finset.card sols = 5 := 
by
  sorry

end number_of_solutions_l121_121248


namespace tangent_identity_problem_l121_121871

theorem tangent_identity_problem 
    (α β : ℝ) 
    (h1 : Real.tan (α + β) = 1) 
    (h2 : Real.tan (α - π / 3) = 1 / 3) 
    : Real.tan (β + π / 3) = 1 / 2 := 
sorry

end tangent_identity_problem_l121_121871


namespace part_I_part_II_l121_121552

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∃! x : ℤ, x = -3 ∧ g x 3 > -1) → m = 3 := 
sorry

theorem part_II (m : ℝ) : 
  (∀ x : ℝ, f x a > g x m) → a < 4 := 
sorry

end part_I_part_II_l121_121552


namespace zilla_savings_l121_121624

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l121_121624


namespace parabola_c_value_l121_121879

theorem parabola_c_value (b c : ℝ) 
  (h1 : 5 = 2 * 1^2 + b * 1 + c)
  (h2 : 17 = 2 * 3^2 + b * 3 + c) : 
  c = 5 := 
by
  sorry

end parabola_c_value_l121_121879


namespace find_x3_y3_l121_121026

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l121_121026


namespace collinear_A₁_F_B_iff_q_eq_4_l121_121853

open Real

theorem collinear_A₁_F_B_iff_q_eq_4
  (m q : ℝ) (h_m : m ≠ 0)
  (A B : ℝ × ℝ)
  (h_A : 3 * (m * A.snd + q)^2 + 4 * A.snd^2 = 12)
  (h_B : 3 * (m * B.snd + q)^2 + 4 * B.snd^2 = 12)
  (A₁ : ℝ × ℝ := (A.fst, -A.snd))
  (F : ℝ × ℝ := (1, 0)) :
  ((q = 4) ↔ (∃ k : ℝ, k * (F.fst - A₁.fst) = F.snd - A₁.snd ∧ k * (B.fst - F.fst) = B.snd - F.snd)) :=
sorry

end collinear_A₁_F_B_iff_q_eq_4_l121_121853


namespace total_apples_collected_l121_121368

-- Definitions based on conditions
def number_of_green_apples : ℕ := 124
def number_of_red_apples : ℕ := 3 * number_of_green_apples

-- Proof statement
theorem total_apples_collected : number_of_red_apples + number_of_green_apples = 496 := by
  sorry

end total_apples_collected_l121_121368


namespace ratio_of_triangle_side_to_rectangle_width_l121_121274

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l121_121274


namespace product_is_solution_quotient_is_solution_l121_121213

-- Definitions and conditions from the problem statement
variable (a b c d : ℤ)

-- The conditions
axiom h1 : a^2 - 5 * b^2 = 1
axiom h2 : c^2 - 5 * d^2 = 1

-- Lean 4 statement for the first part: the product
theorem product_is_solution :
  ∃ (m n : ℤ), ((m + n * (5:ℚ)) = (a + b * (5:ℚ)) * (c + d * (5:ℚ))) ∧ (m^2 - 5 * n^2 = 1) :=
sorry

-- Lean 4 statement for the second part: the quotient
theorem quotient_is_solution :
  ∃ (p q : ℤ), ((p + q * (5:ℚ)) = (a + b * (5:ℚ)) / (c + d * (5:ℚ))) ∧ (p^2 - 5 * q^2 = 1) :=
sorry

end product_is_solution_quotient_is_solution_l121_121213


namespace remaining_homes_proof_l121_121499

-- Define the total number of homes
def total_homes : ℕ := 200

-- Distributed homes after the first hour
def homes_distributed_first_hour : ℕ := (2 * total_homes) / 5

-- Remaining homes after the first hour
def remaining_homes_first_hour : ℕ := total_homes - homes_distributed_first_hour

-- Distributed homes in the next 2 hours
def homes_distributed_next_two_hours : ℕ := (60 * remaining_homes_first_hour) / 100

-- Remaining homes after the next 2 hours
def homes_remaining : ℕ := remaining_homes_first_hour - homes_distributed_next_two_hours

theorem remaining_homes_proof : homes_remaining = 48 := by
  sorry

end remaining_homes_proof_l121_121499


namespace tire_mileage_l121_121493

theorem tire_mileage (total_miles_driven : ℕ) (x : ℕ) (spare_tire_miles : ℕ):
  total_miles_driven = 40000 →
  spare_tire_miles = 2 * x →
  4 * x + spare_tire_miles = total_miles_driven →
  x = 6667 := 
by
  intros h_total h_spare h_eq
  sorry

end tire_mileage_l121_121493


namespace is_inverse_g1_is_inverse_g2_l121_121575

noncomputable def f (x : ℝ) := 3 + 2*x - x^2

noncomputable def g1 (x : ℝ) := -1 + Real.sqrt (4 - x)
noncomputable def g2 (x : ℝ) := -1 - Real.sqrt (4 - x)

theorem is_inverse_g1 : ∀ x, f (g1 x) = x :=
by
  intro x
  sorry

theorem is_inverse_g2 : ∀ x, f (g2 x) = x :=
by
  intro x
  sorry

end is_inverse_g1_is_inverse_g2_l121_121575


namespace find_number_l121_121165

theorem find_number (n : ℕ) (h1 : 45 = 11 * n + 1) : n = 4 :=
  sorry

end find_number_l121_121165


namespace tan_x_eq_sqrt3_intervals_of_monotonic_increase_l121_121140

noncomputable def m (x : ℝ) : ℝ × ℝ :=
  (Real.sin (x - Real.pi / 6), 1)

noncomputable def n (x : ℝ) : ℝ × ℝ :=
  (Real.cos x, 1)

noncomputable def f (x : ℝ) : ℝ :=
  (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Proof for part 1
theorem tan_x_eq_sqrt3 (x : ℝ) (h₀ : m x = n x) : Real.tan x = Real.sqrt 3 :=
sorry

-- Proof for part 2
theorem intervals_of_monotonic_increase (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) ↔ 
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) :=
sorry

end tan_x_eq_sqrt3_intervals_of_monotonic_increase_l121_121140


namespace find_interest_rate_l121_121043

theorem find_interest_rate
  (P : ℝ)  -- Principal amount
  (A : ℝ)  -- Final amount
  (T : ℝ)  -- Time period in years
  (H1 : P = 1000)
  (H2 : A = 1120)
  (H3 : T = 2.4)
  : ∃ R : ℝ, (A - P) = (P * R * T) / 100 ∧ R = 5 :=
by
  -- Proof with calculations to be provided here
  sorry

end find_interest_rate_l121_121043


namespace log_expression_as_product_l121_121573

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_as_product (A m n p : ℝ) (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hA : 0 < A) :
  log m A * log n A + log n A * log p A + log p A * log m A =
  log A (m * n * p) * log p A * log n A * log m A :=
by
  sorry

end log_expression_as_product_l121_121573


namespace five_fourths_of_fifteen_fourths_l121_121614

theorem five_fourths_of_fifteen_fourths :
  (5 / 4) * (15 / 4) = 75 / 16 := by
  sorry

end five_fourths_of_fifteen_fourths_l121_121614


namespace equation_one_solution_equation_two_solution_l121_121695

variables (x : ℝ)

theorem equation_one_solution (h : 2 * (x + 3) = 5 * x) : x = 2 :=
sorry

theorem equation_two_solution (h : (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6) : x = -9.2 :=
sorry

end equation_one_solution_equation_two_solution_l121_121695


namespace find_roots_combination_l121_121458

theorem find_roots_combination 
  (α β : ℝ)
  (hα : α^2 - 3 * α + 1 = 0)
  (hβ : β^2 - 3 * β + 1 = 0) :
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end find_roots_combination_l121_121458


namespace at_least_two_participants_solved_exactly_five_l121_121771

open Nat Real

variable {n : ℕ}  -- Number of participants
variable {pij : ℕ → ℕ → ℕ} -- Number of contestants who correctly answered both the i-th and j-th problems

-- Conditions as definitions in Lean 4
def conditions (n : ℕ) (pij : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 6 → pij i j > (2 * n) / 5) ∧
  (∀ k, ¬ (∀ i, 1 ≤ i ∧ i ≤ 6 → pij k i = 1))

-- Main theorem statement
theorem at_least_two_participants_solved_exactly_five (n : ℕ) (pij : ℕ → ℕ → ℕ) (h : conditions n pij) : ∃ k₁ k₂, k₁ ≠ k₂ ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₁ i = 1) ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₂ i = 1) := sorry

end at_least_two_participants_solved_exactly_five_l121_121771


namespace problem_b_is_proposition_l121_121419

def is_proposition (s : String) : Prop :=
  s = "sin 45° = 1" ∨ s = "x^2 + 2x - 1 > 0"

theorem problem_b_is_proposition : is_proposition "sin 45° = 1" :=
by
  -- insert proof steps to establish that "sin 45° = 1" is a proposition
  sorry

end problem_b_is_proposition_l121_121419


namespace zoes_apartment_number_units_digit_is_1_l121_121438

-- Defining the conditions as the initial problem does
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_digit_two (n : ℕ) : Prop :=
  n / 10 = 2 ∨ n % 10 = 2

def three_out_of_four (n : ℕ) : Prop :=
  (is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬ has_digit_two n) ∨
  (is_square n ∧ is_odd n ∧ ¬ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (is_square n ∧ ¬ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (¬ is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n)

theorem zoes_apartment_number_units_digit_is_1 : ∃ n : ℕ, is_two_digit_number n ∧ three_out_of_four n ∧ n % 10 = 1 :=
by
  sorry

end zoes_apartment_number_units_digit_is_1_l121_121438


namespace Aimee_escalator_time_l121_121276

theorem Aimee_escalator_time (d : ℝ) (v_esc : ℝ) (v_walk : ℝ) :
  v_esc = d / 60 → v_walk = d / 90 → (d / (v_esc + v_walk)) = 36 :=
by
  intros h1 h2
  sorry

end Aimee_escalator_time_l121_121276


namespace p_sufficient_for_not_q_l121_121272

variable (x : ℝ)
def p : Prop := 0 < x ∧ x ≤ 1
def q : Prop := 1 / x < 1

theorem p_sufficient_for_not_q : p x → ¬q x :=
by
  sorry

end p_sufficient_for_not_q_l121_121272


namespace seeds_in_fourth_pot_l121_121685

-- Define the conditions as variables
def total_seeds : ℕ := 10
def number_of_pots : ℕ := 4
def seeds_per_pot : ℕ := 3

-- Define the theorem to prove the quantity of seeds planted in the fourth pot
theorem seeds_in_fourth_pot :
  (total_seeds - (seeds_per_pot * (number_of_pots - 1))) = 1 := by
  sorry

end seeds_in_fourth_pot_l121_121685


namespace integer_pairs_summing_to_six_l121_121837

theorem integer_pairs_summing_to_six :
  ∃ m n : ℤ, m + n + m * n = 6 ∧ ((m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0)) :=
by
  sorry

end integer_pairs_summing_to_six_l121_121837


namespace find_point_on_y_axis_l121_121809

/-- 
Given points A (1, 2, 3) and B (2, -1, 4), and a point P on the y-axis 
such that the distances |PA| and |PB| are equal, 
prove that the coordinates of point P are (0, -7/6, 0).
 -/
theorem find_point_on_y_axis
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, 3))
  (hB : B = (2, -1, 4))
  (P : ℝ × ℝ × ℝ)
  (hP : ∃ y : ℝ, P = (0, y, 0)) :
  dist A P = dist B P → P = (0, -7/6, 0) :=
by
  sorry

end find_point_on_y_axis_l121_121809


namespace percentage_difference_l121_121377

theorem percentage_difference (X : ℝ) (h1 : first_num = 0.70 * X) (h2 : second_num = 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 10 := by
  sorry

end percentage_difference_l121_121377


namespace acute_angle_ACD_l121_121100

theorem acute_angle_ACD (α : ℝ) (h : α ≤ 120) :
  ∃ (ACD : ℝ), ACD = Real.arcsin ((Real.tan (α / 2)) / Real.sqrt 3) :=
sorry

end acute_angle_ACD_l121_121100


namespace find_a1_a10_value_l121_121490

variable {α : Type} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a1_a10_value (a : ℕ → α) (h1 : is_geometric_sequence a)
    (h2 : a 4 + a 7 = 2) (h3 : a 5 * a 6 = -8) : a 1 + a 10 = -7 := by
  sorry

end find_a1_a10_value_l121_121490


namespace problem_solution_l121_121348

noncomputable def g (x : ℝ) (P : ℝ) (Q : ℝ) (R : ℝ) : ℝ := x^2 / (P * x^2 + Q * x + R)

theorem problem_solution (P Q R : ℤ) 
  (h1 : ∀ x > 5, g x P Q R > 0.5)
  (h2 : P * (-3)^2 + Q * (-3) + R = 0)
  (h3 : P * 4^2 + Q * 4 + R = 0)
  (h4 : ∃ y : ℝ, y = 1 / P ∧ ∀ x : ℝ, abs (g x P Q R - y) < ε):
  P + Q + R = -24 :=
by
  sorry

end problem_solution_l121_121348


namespace find_y_l121_121945

theorem find_y (y : ℝ) (a b : ℝ × ℝ) (h_a : a = (4, 2)) (h_b : b = (6, y)) (h_parallel : 4 * y - 2 * 6 = 0) :
  y = 3 :=
sorry

end find_y_l121_121945


namespace lemonade_total_difference_is_1860_l121_121905

-- Define the conditions
def stanley_rate : Nat := 4
def stanley_price : Real := 1.50

def carl_rate : Nat := 7
def carl_price : Real := 1.30

def lucy_rate : Nat := 5
def lucy_price : Real := 1.80

def hours : Nat := 3

-- Compute the total amounts for each sibling
def stanley_total : Real := stanley_rate * hours * stanley_price
def carl_total : Real := carl_rate * hours * carl_price
def lucy_total : Real := lucy_rate * hours * lucy_price

-- Compute the individual differences
def diff_stanley_carl : Real := carl_total - stanley_total
def diff_stanley_lucy : Real := lucy_total - stanley_total
def diff_carl_lucy : Real := carl_total - lucy_total

-- Sum the differences
def total_difference : Real := diff_stanley_carl + diff_stanley_lucy + diff_carl_lucy

-- The proof statement
theorem lemonade_total_difference_is_1860 :
  total_difference = 18.60 :=
by
  sorry

end lemonade_total_difference_is_1860_l121_121905


namespace car_speed_first_hour_l121_121719

theorem car_speed_first_hour (x : ℝ) (h_second_hour_speed : x + 80 / 2 = 85) : x = 90 :=
sorry

end car_speed_first_hour_l121_121719


namespace winning_percentage_l121_121789

theorem winning_percentage (total_votes majority : ℕ) (h1 : total_votes = 455) (h2 : majority = 182) :
  ∃ P : ℕ, P = 70 ∧ (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority := 
sorry

end winning_percentage_l121_121789


namespace circle_eq_center_tangent_l121_121240

theorem circle_eq_center_tangent (x y : ℝ) : 
  let center := (5, 4)
  let radius := 4
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 :=
by
  sorry

end circle_eq_center_tangent_l121_121240


namespace initial_food_days_l121_121232

theorem initial_food_days (x : ℕ) (h : 760 * (x - 2) = 3040 * 5) : x = 22 := by
  sorry

end initial_food_days_l121_121232


namespace log_product_eq_one_l121_121956

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_product_eq_one :
  log_base 5 2 * log_base 4 25 = 1 := 
by
  sorry

end log_product_eq_one_l121_121956


namespace unattainable_value_l121_121815

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) : 
  ¬ ∃ y : ℝ, y = (1 - x) / (3 * x + 4) ∧ y = -1/3 :=
by
  sorry

end unattainable_value_l121_121815


namespace problem_statement_l121_121307

-- Definitions of sets S and P
def S : Set ℝ := {x | x^2 - 3 * x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2 * a + 15}

-- Proof statement
theorem problem_statement (a : ℝ) : 
  (S = {x | -2 < x ∧ x < 5}) ∧ (S ⊆ P a → a ∈ Set.Icc (-5 : ℝ) (-3 : ℝ)) :=
by
  sorry

end problem_statement_l121_121307


namespace exists_solution_in_interval_l121_121864

theorem exists_solution_in_interval : ∃ x ∈ (Set.Ioo (3: ℝ) (4: ℝ)), Real.log x / Real.log 2 + x - 5 = 0 :=
by
  sorry

end exists_solution_in_interval_l121_121864


namespace digit_making_527B_divisible_by_9_l121_121033

theorem digit_making_527B_divisible_by_9 (B : ℕ) : 14 + B ≡ 0 [MOD 9] → B = 4 :=
by
  intro h
  -- sorry is used in place of the actual proof.
  sorry

end digit_making_527B_divisible_by_9_l121_121033


namespace triangle_side_range_a_l121_121909

theorem triangle_side_range_a {a : ℝ} : 2 < a ∧ a < 5 ↔
  3 + (2 * a + 1) > 8 ∧ 
  8 - 3 < 2 * a + 1 ∧ 
  8 - (2 * a + 1) < 3 :=
by
  sorry

end triangle_side_range_a_l121_121909


namespace solve_for_x_l121_121739

theorem solve_for_x (x : ℝ) (h : 3 * (x - 5) = 3 * (18 - 5)) : x = 18 :=
by
  sorry

end solve_for_x_l121_121739


namespace solve_equation_l121_121855

theorem solve_equation :
  ∃ (a b c d : ℚ), 
  (a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + 2 / 5 = 0) ∧ 
  (a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5) := sorry

end solve_equation_l121_121855


namespace initial_average_mark_l121_121600

-- Define the conditions
def total_students := 13
def average_mark := 72
def excluded_students := 5
def excluded_students_average := 40
def remaining_students := total_students - excluded_students
def remaining_students_average := 92

-- Define the total marks calculations
def initial_total_marks (A : ℕ) : ℕ := total_students * A
def excluded_total_marks : ℕ := excluded_students * excluded_students_average
def remaining_total_marks : ℕ := remaining_students * remaining_students_average

-- Prove the initial average mark
theorem initial_average_mark : 
  initial_total_marks average_mark = excluded_total_marks + remaining_total_marks →
  average_mark = 72 :=
by
  sorry

end initial_average_mark_l121_121600


namespace probability_of_red_light_l121_121449

-- Definitions based on the conditions
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Statement of the problem to prove the probability of seeing red light
theorem probability_of_red_light : (red_duration : ℚ) / total_cycle_time = 2 / 5 := 
by sorry

end probability_of_red_light_l121_121449


namespace fill_pipe_half_time_l121_121330

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l121_121330


namespace function_is_monotonically_decreasing_l121_121048

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 3)

theorem function_is_monotonically_decreasing :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → deriv f x ≤ 0 :=
by
  sorry

end function_is_monotonically_decreasing_l121_121048


namespace quadratic_polynomial_correct_l121_121296

noncomputable def q (x : ℝ) : ℝ := (11/10) * x^2 - (21/10) * x + 5

theorem quadratic_polynomial_correct :
  (q (-1) = 4) ∧ (q 2 = 1) ∧ (q 4 = 10) :=
by
  -- Proof goes here
  sorry

end quadratic_polynomial_correct_l121_121296


namespace area_increase_by_nine_l121_121440

theorem area_increase_by_nine (a : ℝ) :
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := extended_side_length^2;
  extended_area / original_area = 9 :=
by
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := (extended_side_length)^2;
  sorry

end area_increase_by_nine_l121_121440


namespace find_enclosed_area_l121_121282

def area_square (side_length : ℕ) : ℕ :=
  side_length * side_length

def area_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem find_enclosed_area :
  let side1 := 3
  let side2 := 6
  let area1 := area_square side1
  let area2 := area_square side2
  let area_tri := 2 * area_triangle side1 side2
  area1 + area2 + area_tri = 63 :=
by
  sorry

end find_enclosed_area_l121_121282


namespace count_total_wheels_l121_121039

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end count_total_wheels_l121_121039


namespace mod_congruent_integers_l121_121990

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end mod_congruent_integers_l121_121990


namespace career_preference_degrees_l121_121338

theorem career_preference_degrees (boys girls : ℕ) (ratio_boys_to_girls : boys / gcd boys girls = 2 ∧ girls / gcd boys girls = 3) 
  (boys_preference : ℕ) (girls_preference : ℕ) 
  (h1 : boys_preference = boys / 3)
  (h2 : girls_preference = 2 * girls / 3) : 
  (boys_preference + girls_preference) / (boys + girls) * 360 = 192 :=
by
  sorry

end career_preference_degrees_l121_121338


namespace triangle_equilateral_of_equal_angle_ratios_l121_121303

theorem triangle_equilateral_of_equal_angle_ratios
  (a b c : ℝ)
  (h₁ : a + b + c = 180)
  (h₂ : a = b)
  (h₃ : b = c) :
  a = 60 ∧ b = 60 ∧ c = 60 :=
by
  sorry

end triangle_equilateral_of_equal_angle_ratios_l121_121303


namespace score_difference_l121_121316

-- Definitions of the given conditions
def Layla_points : ℕ := 70
def Total_points : ℕ := 112

-- The statement to be proven
theorem score_difference : (Layla_points - (Total_points - Layla_points)) = 28 :=
by sorry

end score_difference_l121_121316


namespace ducks_and_geese_meeting_l121_121676

theorem ducks_and_geese_meeting:
  ∀ x : ℕ, ( ∀ ducks_speed : ℚ, ducks_speed = (1/7) ) → 
         ( ∀ geese_speed : ℚ, geese_speed = (1/9) ) → 
         (ducks_speed * x + geese_speed * x = 1) :=
by
  sorry

end ducks_and_geese_meeting_l121_121676


namespace range_of_a_l121_121605

-- Define the inequality problem
def inequality_always_true (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 3 * a * x + a - 2 < 0

-- Define the range condition for "a"
def range_condition (a : ℝ) : Prop :=
  (a = 0 ∧ (-2 < 0)) ∨
  (a ≠ 0 ∧ a < 0 ∧ a * (5 * a + 8) < 0)

-- The main theorem stating the equivalence
theorem range_of_a (a : ℝ) : inequality_always_true a ↔ a ∈ Set.Icc (- (8 / 5)) 0 := by
  sorry

end range_of_a_l121_121605


namespace min_value_l121_121927

theorem min_value (x : ℝ) (h : x > 1) : ∃ m : ℝ, m = 2 * Real.sqrt 5 ∧ ∀ y : ℝ, y = Real.sqrt (x - 1) → (x = y^2 + 1) → (x + 4) / y = m :=
by
  sorry

end min_value_l121_121927


namespace double_meat_sandwich_bread_count_l121_121653

theorem double_meat_sandwich_bread_count (x : ℕ) :
  14 * 2 + 12 * x = 64 → x = 3 := by
  intro h
  sorry

end double_meat_sandwich_bread_count_l121_121653


namespace opposite_of_2023_l121_121926

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l121_121926


namespace students_taking_neither_l121_121648

theorem students_taking_neither (total students_cs students_electronics students_both : ℕ)
  (h1 : total = 60) (h2 : students_cs = 42) (h3 : students_electronics = 35) (h4 : students_both = 25) :
  total - (students_cs - students_both + students_electronics - students_both + students_both) = 8 :=
by {
  sorry
}

end students_taking_neither_l121_121648


namespace problem_provable_l121_121516

noncomputable def given_expression (a : ℝ) : ℝ :=
  (1 / (a + 2)) / ((a^2 - 4 * a + 4) / (a^2 - 4)) - (2 / (a - 2))

theorem problem_provable : given_expression (Real.sqrt 5 + 2) = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_provable_l121_121516


namespace division_and_multiplication_l121_121635

theorem division_and_multiplication (x : ℝ) (h : x = 9) : (x / 6 * 12) = 18 := by
  sorry

end division_and_multiplication_l121_121635


namespace predicted_whales_l121_121188

theorem predicted_whales (num_last_year num_this_year num_next_year : ℕ)
  (h1 : num_this_year = 2 * num_last_year)
  (h2 : num_last_year = 4000)
  (h3 : num_next_year = 8800) :
  num_next_year - num_this_year = 800 :=
by
  sorry

end predicted_whales_l121_121188


namespace cinema_chairs_l121_121310

theorem cinema_chairs (chairs_between : ℕ) (h : chairs_between = 30) :
  chairs_between + 2 = 32 := by
  sorry

end cinema_chairs_l121_121310


namespace gardener_tree_arrangement_l121_121960

theorem gardener_tree_arrangement :
  let maple_trees := 4
  let oak_trees := 5
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)
  let valid_slots := 9  -- as per slots identified in the solution
  let valid_arrangements := 1 * Nat.choose valid_slots oak_trees
  let probability := valid_arrangements / total_arrangements
  probability = 1 / 75075 →
  (1 + 75075) = 75076 := by {
    sorry
  }

end gardener_tree_arrangement_l121_121960


namespace discount_equivalence_l121_121587

variable (Original_Price : ℝ)

theorem discount_equivalence (h1 : Real) (h2 : Real) :
  (h1 = 0.5 * Original_Price) →
  (h2 = 0.7 * h1) →
  (Original_Price - h2) / Original_Price = 0.65 :=
by
  intros
  sorry

end discount_equivalence_l121_121587


namespace find_y_of_rectangle_area_l121_121329

theorem find_y_of_rectangle_area (y : ℝ) (h1 : y > 0) 
(h2 : (0, 0) = (0, 0)) (h3 : (0, 6) = (0, 6)) 
(h4 : (y, 6) = (y, 6)) (h5 : (y, 0) = (y, 0)) 
(h6 : 6 * y = 42) : y = 7 :=
by {
  sorry
}

end find_y_of_rectangle_area_l121_121329


namespace fraction_auto_installment_credit_extended_by_finance_companies_l121_121900

def total_consumer_installment_credit : ℝ := 291.6666666666667
def auto_instalment_percentage : ℝ := 0.36
def auto_finance_companies_credit_extended : ℝ := 35

theorem fraction_auto_installment_credit_extended_by_finance_companies :
  auto_finance_companies_credit_extended / (auto_instalment_percentage * total_consumer_installment_credit) = 1 / 3 :=
by
  sorry

end fraction_auto_installment_credit_extended_by_finance_companies_l121_121900


namespace each_friend_pays_6413_l121_121166

noncomputable def amount_each_friend_pays (total_bill : ℝ) (friends : ℕ) (first_discount : ℝ) (second_discount : ℝ) : ℝ :=
  let bill_after_first_coupon := total_bill * (1 - first_discount)
  let bill_after_second_coupon := bill_after_first_coupon * (1 - second_discount)
  bill_after_second_coupon / friends

theorem each_friend_pays_6413 :
  amount_each_friend_pays 600 8 0.10 0.05 = 64.13 :=
by
  sorry

end each_friend_pays_6413_l121_121166


namespace number_of_days_at_Tom_house_l121_121764

-- Define the constants and conditions
def total_people := 6
def plates_per_person_per_day := 6
def total_plates := 144

-- Prove that the number of days they were at Tom's house is 4
theorem number_of_days_at_Tom_house : total_plates / (total_people * plates_per_person_per_day) = 4 :=
  sorry

end number_of_days_at_Tom_house_l121_121764


namespace angle_C_eq_pi_div_3_find_ab_values_l121_121393

noncomputable def find_angle_C (A B C : ℝ) (a b c : ℝ) : ℝ :=
  if c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C then C else 0

noncomputable def find_sides_ab (A B C : ℝ) (c S : ℝ) : Set (ℝ × ℝ) :=
  if C = Real.pi / 3 ∧ c = 2 * Real.sqrt 3 ∧ S = 2 * Real.sqrt 3 then
    { (a, b) | a^4 - 20 * a^2 + 64 = 0 ∧ b = 8 / a } else
    ∅

theorem angle_C_eq_pi_div_3 (A B C : ℝ) (a b c : ℝ) :
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C)
  ↔ (C = Real.pi / 3) :=
sorry

theorem find_ab_values (A B C : ℝ) (c S a b : ℝ) :
  (C = Real.pi / 3) ∧ (c = 2 * Real.sqrt 3) ∧ (S = 2 * Real.sqrt 3) ∧ (a^4 - 20 * a^2 + 64 = 0) ∧ (b = 8 / a)
  ↔ ((a, b) = (2, 4) ∨ (a, b) = (4, 2)) :=
sorry

end angle_C_eq_pi_div_3_find_ab_values_l121_121393


namespace find_x_in_triangle_l121_121317

theorem find_x_in_triangle (y z : ℝ) (cos_Y_minus_Z : ℝ) (h1 : y = 7) (h2 : z = 6) (h3 : cos_Y_minus_Z = 1 / 2) : 
    ∃ x : ℝ, x = Real.sqrt 73 :=
by
  existsi Real.sqrt 73
  sorry

end find_x_in_triangle_l121_121317


namespace solve_system_of_equations_l121_121547

theorem solve_system_of_equations (x y z : ℝ) :
  (2 * x^2 / (1 + x^2) = y) →
  (2 * y^2 / (1 + y^2) = z) →
  (2 * z^2 / (1 + z^2) = x) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l121_121547


namespace min_value_expr_l121_121798

theorem min_value_expr (x y : ℝ) : 
  ∃ x y : ℝ, (x, y) = (4, 0) ∧ (∀ x y : ℝ, x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ -22) :=
by
  sorry

end min_value_expr_l121_121798


namespace ones_digit_seven_consecutive_integers_l121_121507

theorem ones_digit_seven_consecutive_integers (k : ℕ) (hk : k % 5 = 1) :
  (k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10 = 0 :=
by
  sorry

end ones_digit_seven_consecutive_integers_l121_121507


namespace average_temperature_l121_121125

def highTemps : List ℚ := [51, 60, 56, 55, 48, 63, 59]
def lowTemps : List ℚ := [42, 50, 44, 43, 41, 46, 45]

def dailyAverage (high low : ℚ) : ℚ :=
  (high + low) / 2

def averageOfAverages (tempsHigh tempsLow : List ℚ) : ℚ :=
  (List.sum (List.zipWith dailyAverage tempsHigh tempsLow)) / tempsHigh.length

theorem average_temperature :
  averageOfAverages highTemps lowTemps = 50.2 :=
  sorry

end average_temperature_l121_121125


namespace face_value_of_share_l121_121382

theorem face_value_of_share (FV : ℝ) (dividend_percent : ℝ) (interest_percent : ℝ) (market_value : ℝ) :
  dividend_percent = 0.09 → 
  interest_percent = 0.12 →
  market_value = 33 →
  (0.09 * FV = 0.12 * 33) → FV = 44 :=
by
  intros
  sorry

end face_value_of_share_l121_121382


namespace sequence_properties_l121_121432

/-- Theorem setup:
Assume a sequence {a_n} with a_1 = 1 and a_{n+1} = 2a_n / (a_n + 2)
Also, define b_n = 1 / a_n
-/
theorem sequence_properties 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  -- Prove that {b_n} (b n = 1 / a n) is arithmetic with common difference 1/2
  (∃ b : ℕ → ℝ, (∀ n : ℕ, b n = 1 / a n) ∧ (∀ n : ℕ, b (n + 1) = b n + 1 / 2)) ∧ 
  -- Prove the general formula for a_n
  (∀ n : ℕ, a (n + 1) = 2 / (n + 1)) := 
sorry


end sequence_properties_l121_121432


namespace log_101600_l121_121139

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_101600 (h : log_base_10 102 = 0.3010) : log_base_10 101600 = 2.3010 :=
by
  sorry

end log_101600_l121_121139


namespace summer_camp_students_l121_121257

theorem summer_camp_students (x : ℕ)
  (h1 : (1 / 6) * x = n_Shanghai)
  (h2 : n_Tianjin = 24)
  (h3 : (1 / 4) * x = n_Chongqing)
  (h4 : n_Beijing = (3 / 2) * (n_Shanghai + n_Tianjin)) :
  x = 180 :=
by
  sorry

end summer_camp_students_l121_121257


namespace lauren_earnings_tuesday_l121_121072

def money_from_commercials (num_commercials : ℕ) (rate_per_commercial : ℝ) : ℝ :=
  num_commercials * rate_per_commercial

def money_from_subscriptions (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  num_subscriptions * rate_per_subscription

def total_money (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  money_from_commercials num_commercials rate_per_commercial + money_from_subscriptions num_subscriptions rate_per_subscription

theorem lauren_earnings_tuesday (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) :
  num_commercials = 100 → rate_per_commercial = 0.50 → num_subscriptions = 27 → rate_per_subscription = 1.00 → 
  total_money num_commercials rate_per_commercial num_subscriptions rate_per_subscription = 77 :=
by
  intros h1 h2 h3 h4
  simp [money_from_commercials, money_from_subscriptions, total_money, h1, h2, h3, h4]
  sorry

end lauren_earnings_tuesday_l121_121072


namespace proof_problem_l121_121057

theorem proof_problem (a b A B : ℝ) (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (h_f_def : ∀ θ : ℝ, f θ = 1 + a * Real.cos θ + b * Real.sin θ + A * Real.sin (2 * θ) + B * Real.cos (2 * θ)) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end proof_problem_l121_121057


namespace parabola_min_value_l121_121912

variable {x0 y0 : ℝ}

def isOnParabola (x0 y0 : ℝ) : Prop := x0^2 = y0

noncomputable def expression (y0 x0 : ℝ) : ℝ :=
  Real.sqrt 2 * y0 + |x0 - y0 - 2|

theorem parabola_min_value :
  isOnParabola x0 y0 → ∃ (m : ℝ), m = (9 / 4 : ℝ) - (Real.sqrt 2 / 4) ∧ 
  ∀ y0 x0, expression y0 x0 ≥ (9 / 4 : ℝ) - (Real.sqrt 2 / 4) := 
by
  sorry

end parabola_min_value_l121_121912


namespace no_solution_for_eq_l121_121061

theorem no_solution_for_eq (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) → False :=
sorry

end no_solution_for_eq_l121_121061


namespace total_number_of_animals_l121_121406

-- Define the data and conditions
def total_legs : ℕ := 38
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the proof problem
theorem total_number_of_animals (h1 : total_legs = 38) 
                                (h2 : chickens = 5) 
                                (h3 : chicken_legs = 2) 
                                (h4 : sheep_legs = 4) : 
  (∃ sheep : ℕ, chickens + sheep = 12) :=
by 
  sorry

end total_number_of_animals_l121_121406


namespace compute_expression_l121_121060

theorem compute_expression : (-1) ^ 2014 + (π - 3.14) ^ 0 - (1 / 2) ^ (-2) = -2 := by
  sorry

end compute_expression_l121_121060


namespace sum_pqrs_eq_3150_l121_121082

theorem sum_pqrs_eq_3150
  (p q r s : ℝ)
  (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) (h5 : q ≠ s) (h6 : r ≠ s)
  (hroots1 : ∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 → (x = r ∨ x = s))
  (hroots2 : ∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 → (x = p ∨ x = q)) :
  p + q + r + s = 3150 :=
by
  sorry

end sum_pqrs_eq_3150_l121_121082


namespace five_op_two_l121_121515

-- Definition of the operation
def op (a b : ℝ) := 3 * a + 4 * b

-- The theorem statement
theorem five_op_two : op 5 2 = 23 := by
  sorry

end five_op_two_l121_121515


namespace sequence_problem_l121_121189

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n - a (n - 1) = a 1 - a 0

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b n * b (n - 1) = b 1 * b 0

theorem sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : a 0 = -9) (ha1 : a 3 = -1) (ha_seq : arithmetic_sequence a)
  (hb : b 0 = -9) (hb4 : b 4 = -1) (hb_seq : geometric_sequence b) :
  b 2 * (a 2 - a 1) = -8 :=
sorry

end sequence_problem_l121_121189


namespace overlapping_area_of_congruent_isosceles_triangles_l121_121835

noncomputable def isosceles_right_triangle (hypotenuse : ℝ) := 
  {l : ℝ // l = hypotenuse / Real.sqrt 2}

theorem overlapping_area_of_congruent_isosceles_triangles (hypotenuse : ℝ) 
  (A₁ A₂ : isosceles_right_triangle hypotenuse) (h_congruent : A₁ = A₂) :
  hypotenuse = 10 → 
  let leg := hypotenuse / Real.sqrt 2 
  let area := (leg * leg) / 2 
  let shared_area := area / 2 
  shared_area = 12.5 :=
by
  sorry

end overlapping_area_of_congruent_isosceles_triangles_l121_121835


namespace cat_total_birds_caught_l121_121417

theorem cat_total_birds_caught (day_birds night_birds : ℕ) 
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) :
  day_birds + night_birds = 24 :=
sorry

end cat_total_birds_caught_l121_121417


namespace greenfield_academy_math_count_l121_121561

theorem greenfield_academy_math_count (total_players taking_physics both_subjects : ℕ) 
(h_total: total_players = 30) 
(h_physics: taking_physics = 15) 
(h_both: both_subjects = 3) : 
∃ taking_math : ℕ, taking_math = 21 :=
by
  sorry

end greenfield_academy_math_count_l121_121561


namespace tim_final_soda_cans_l121_121774

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l121_121774


namespace overall_average_correct_l121_121518

noncomputable def overall_average : ℝ :=
  let students1 := 60
  let students2 := 35
  let students3 := 45
  let students4 := 42
  let avgMarks1 := 50
  let avgMarks2 := 60
  let avgMarks3 := 55
  let avgMarks4 := 45
  let total_students := students1 + students2 + students3 + students4
  let total_marks := (students1 * avgMarks1) + (students2 * avgMarks2) + (students3 * avgMarks3) + (students4 * avgMarks4)
  total_marks / total_students

theorem overall_average_correct : overall_average = 52.00 := by
  sorry

end overall_average_correct_l121_121518


namespace simplify_T_l121_121628

variable (x : ℝ)

theorem simplify_T :
  9 * (x + 2)^2 - 12 * (x + 2) + 4 = 4 * (1.5 * x + 2)^2 :=
by
  sorry

end simplify_T_l121_121628


namespace correlation_index_l121_121414

variable (height_variation_weight_explained : ℝ)
variable (random_errors_contribution : ℝ)

def R_squared : ℝ := height_variation_weight_explained

theorem correlation_index (h1 : height_variation_weight_explained = 0.64) (h2 : random_errors_contribution = 0.36) : R_squared height_variation_weight_explained = 0.64 :=
by
  exact h1  -- Placeholder for actual proof, since only statement is required

end correlation_index_l121_121414


namespace proof_problem_l121_121807

theorem proof_problem (a b : ℝ) (h1 : (5 * a + 2)^(1/3) = 3) (h2 : (3 * a + b - 1)^(1/2) = 4) :
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3)^(1/2) = 4 :=
by
  sorry

end proof_problem_l121_121807


namespace plane_difference_correct_l121_121664

noncomputable def max_planes : ℕ := 27
noncomputable def min_planes : ℕ := 7
noncomputable def diff_planes : ℕ := max_planes - min_planes

theorem plane_difference_correct : diff_planes = 20 := by
  sorry

end plane_difference_correct_l121_121664


namespace divides_a_square_minus_a_and_a_cube_minus_a_l121_121543

theorem divides_a_square_minus_a_and_a_cube_minus_a (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) :=
by
  sorry

end divides_a_square_minus_a_and_a_cube_minus_a_l121_121543


namespace min_of_quadratic_l121_121997

theorem min_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, x^2 + 7 * x + 3 ≤ y^2 + 7 * y + 3) ∧ x = -7 / 2 :=
by
  sorry

end min_of_quadratic_l121_121997


namespace solve_equation_l121_121353

-- Defining the original equation as a Lean function
def equation (x : ℝ) : Prop :=
  (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2))

theorem solve_equation :
  ∃ x : ℝ, equation x ∧ x = -13 / 2 :=
by
  -- Equation specification and transformations
  sorry

end solve_equation_l121_121353


namespace intersection_eq_l121_121155

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_eq : A ∩ B = {2, 4, 8} := by
  sorry

end intersection_eq_l121_121155


namespace common_ratio_is_two_l121_121304

-- Geometric sequence definition
noncomputable def common_ratio (n : ℕ) (a : ℕ → ℝ) : ℝ :=
a 2 / a 1

-- The sequence has 10 terms
def ten_terms (a : ℕ → ℝ) : Prop :=
∀ n, 1 ≤ n ∧ n ≤ 10

-- The product of the odd terms is 2
def product_of_odd_terms (a : ℕ → ℝ) : Prop :=
(a 1) * (a 3) * (a 5) * (a 7) * (a 9) = 2

-- The product of the even terms is 64
def product_of_even_terms (a : ℕ → ℝ) : Prop :=
(a 2) * (a 4) * (a 6) * (a 8) * (a 10) = 64

-- The problem statement to prove that the common ratio q is 2
theorem common_ratio_is_two (a : ℕ → ℝ) (q : ℝ) (h1 : ten_terms a) 
(h2 : product_of_odd_terms a) (h3 : product_of_even_terms a) : q = 2 :=
by {
  sorry
}

end common_ratio_is_two_l121_121304


namespace area_of_farm_l121_121110

theorem area_of_farm (W L : ℝ) (hW : W = 30) 
  (hL_fence_cost : 14 * (L + W + Real.sqrt (L^2 + W^2)) = 1680) : 
  W * L = 1200 :=
by
  sorry -- Proof not required

end area_of_farm_l121_121110


namespace remainder_is_five_l121_121698

theorem remainder_is_five (A : ℕ) (h : 17 = 6 * 2 + A) : A = 5 :=
sorry

end remainder_is_five_l121_121698


namespace knicks_eq_knocks_l121_121153

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end knicks_eq_knocks_l121_121153


namespace ratio_calculation_l121_121374

theorem ratio_calculation (A B C : ℚ)
  (h_ratio : (A / B = 3 / 2) ∧ (B / C = 2 / 5)) :
  (4 * A + 3 * B) / (5 * C - 2 * B) = 15 / 23 := by
  sorry

end ratio_calculation_l121_121374


namespace tim_total_money_raised_l121_121179

-- Definitions based on conditions
def maxDonation : ℤ := 1200
def numMaxDonors : ℤ := 500
def numHalfDonors : ℤ := 3 * numMaxDonors
def halfDonation : ℤ := maxDonation / 2
def totalPercent : ℚ := 0.4

def totalDonationFromMaxDonors : ℤ := numMaxDonors * maxDonation
def totalDonationFromHalfDonors : ℤ := numHalfDonors * halfDonation
def totalDonation : ℤ := totalDonationFromMaxDonors + totalDonationFromHalfDonors

-- Proposition that Tim's total money raised is $3,750,000
theorem tim_total_money_raised : (totalDonation : ℚ) / totalPercent = 3750000 := by
  -- Verified in the proof steps
  sorry

end tim_total_money_raised_l121_121179


namespace factor_b_value_l121_121706

theorem factor_b_value (a b : ℤ) (h : ∀ x : ℂ, (x^2 - x - 1) ∣ (a*x^3 + b*x^2 + 1)) : b = -2 := 
sorry

end factor_b_value_l121_121706


namespace regular_polygon_sides_l121_121342

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end regular_polygon_sides_l121_121342


namespace workers_number_l121_121159

theorem workers_number (W A : ℕ) (h1 : W * 25 = A) (h2 : (W + 10) * 15 = A) : W = 15 :=
by
  sorry

end workers_number_l121_121159


namespace meaning_of_probability_l121_121570

-- Definitions

def probability_of_winning (p : ℚ) : Prop :=
  p = 1 / 4

-- Theorem statement
theorem meaning_of_probability :
  probability_of_winning (1 / 4) →
  ∀ n : ℕ, (n ≠ 0) → (n / 4 * 4) = n :=
by
  -- Placeholder proof
  sorry

end meaning_of_probability_l121_121570


namespace disease_cases_linear_decrease_l121_121686

theorem disease_cases_linear_decrease (cases_1970 cases_2010 cases_1995 cases_2005 : ℕ)
  (year_1970 year_2010 year_1995 year_2005 : ℕ)
  (h_cases_1970 : cases_1970 = 800000)
  (h_cases_2010 : cases_2010 = 200)
  (h_year_1970 : year_1970 = 1970)
  (h_year_2010 : year_2010 = 2010)
  (h_year_1995 : year_1995 = 1995)
  (h_year_2005 : year_2005 = 2005)
  (linear_decrease : ∀ t, cases_1970 - (cases_1970 - cases_2010) * (t - year_1970) / (year_2010 - year_1970) = cases_1970 - t * (cases_1970 - cases_2010) / (year_2010 - year_1970))
  : cases_1995 = 300125 ∧ cases_2005 = 100175 := sorry

end disease_cases_linear_decrease_l121_121686
