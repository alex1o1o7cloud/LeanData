import Mathlib

namespace rose_part_payment_l1824_182425

-- Defining the conditions
def total_cost (T : ℝ) := 0.95 * T = 5700
def part_payment (x : ℝ) (T : ℝ) := x = 0.05 * T

-- The proof problem: Prove that the part payment Rose made is $300
theorem rose_part_payment : ∃ T x, total_cost T ∧ part_payment x T ∧ x = 300 :=
by
  sorry

end rose_part_payment_l1824_182425


namespace phase_shift_cosine_l1824_182467

theorem phase_shift_cosine (x : ℝ) : 2 * x + (Real.pi / 2) = 0 → x = - (Real.pi / 4) :=
by
  intro h
  sorry

end phase_shift_cosine_l1824_182467


namespace solution_set_of_inequality_l1824_182486

theorem solution_set_of_inequality (x : ℝ) : (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by sorry

end solution_set_of_inequality_l1824_182486


namespace abs_inequality_solution_set_l1824_182460

theorem abs_inequality_solution_set {x : ℝ} : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end abs_inequality_solution_set_l1824_182460


namespace original_price_of_book_l1824_182422

-- Define the conditions as Lean 4 statements
variable (P : ℝ)  -- Original price of the book
variable (P_new : ℝ := 480)  -- New price of the book
variable (increase_percentage : ℝ := 0.60)  -- Percentage increase in the price

-- Prove the question: original price equals to $300
theorem original_price_of_book :
  P + increase_percentage * P = P_new → P = 300 :=
by
  sorry

end original_price_of_book_l1824_182422


namespace first_term_of_geometric_series_l1824_182412

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l1824_182412


namespace factor_M_l1824_182440

theorem factor_M (a b c d : ℝ) : 
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 =
  (a * c + b * d - a^2 - b^2)^2 :=
by
  sorry

end factor_M_l1824_182440


namespace pipe_q_fill_time_l1824_182426

theorem pipe_q_fill_time :
  ∀ (T : ℝ), (2 * (1 / 10 + 1 / T) + 10 * (1 / T) = 1) → T = 15 :=
by
  intro T
  intro h
  sorry

end pipe_q_fill_time_l1824_182426


namespace p_sufficient_for_not_q_l1824_182406

variable (x : ℝ)
def p : Prop := 0 < x ∧ x ≤ 1
def q : Prop := 1 / x < 1

theorem p_sufficient_for_not_q : p x → ¬q x :=
by
  sorry

end p_sufficient_for_not_q_l1824_182406


namespace prob_first_three_heads_all_heads_l1824_182456

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l1824_182456


namespace cost_equality_store_comparison_for_10_l1824_182480

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

end cost_equality_store_comparison_for_10_l1824_182480


namespace xyz_eq_neg10_l1824_182439

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

end xyz_eq_neg10_l1824_182439


namespace find_enclosed_area_l1824_182448

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

end find_enclosed_area_l1824_182448


namespace repeating_decimals_expr_as_fraction_l1824_182418

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end repeating_decimals_expr_as_fraction_l1824_182418


namespace theater_seats_l1824_182458

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

end theater_seats_l1824_182458


namespace vector_parallel_cos_sin_l1824_182471

theorem vector_parallel_cos_sin (θ : ℝ) (a b : ℝ × ℝ) (ha : a = (Real.cos θ, Real.sin θ)) (hb : b = (1, -2)) :
  ∀ (h : ∃ k : ℝ, a = (k * 1, k * (-2))), 
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 3 := 
by
  sorry

end vector_parallel_cos_sin_l1824_182471


namespace people_sitting_between_same_l1824_182417

theorem people_sitting_between_same 
  (n : ℕ) (h_even : n % 2 = 0) 
  (f : Fin (2 * n) → Fin (2 * n)) :
  ∃ (a b : Fin (2 * n)), 
  ∃ (k k' : ℕ), k < 2 * n ∧ k' < 2 * n ∧ (a : ℕ) < (b : ℕ) ∧ 
  ((b - a = k) ∧ (f b - f a = k)) ∨ ((a - b + 2*n = k') ∧ ((f a - f b + 2 * n) % (2 * n) = k')) :=
by
  sorry

end people_sitting_between_same_l1824_182417


namespace x_coordinate_at_2005th_stop_l1824_182443

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

end x_coordinate_at_2005th_stop_l1824_182443


namespace regular_polygon_sides_l1824_182493

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end regular_polygon_sides_l1824_182493


namespace find_other_root_l1824_182445

theorem find_other_root (b : ℝ) (h : ∀ x : ℝ, x^2 - b * x + 3 = 0 → x = 3 ∨ ∃ y, y = 1) :
  ∃ y, y = 1 :=
by
  sorry

end find_other_root_l1824_182445


namespace like_terms_correct_l1824_182416

theorem like_terms_correct : 
  (¬(∀ x y z w : ℝ, (x * y^2 = z ∧ x^2 * y = w)) ∧ 
   ¬(∀ x y : ℝ, (x * y = -2 * y)) ∧ 
    (2^3 = 8 ∧ 3^2 = 9) ∧ 
   ¬(∀ x y z w : ℝ, (5 * x * y = z ∧ 6 * x * y^2 = w))) :=
by
  sorry

end like_terms_correct_l1824_182416


namespace quadratic_polynomial_correct_l1824_182491

noncomputable def q (x : ℝ) : ℝ := (11/10) * x^2 - (21/10) * x + 5

theorem quadratic_polynomial_correct :
  (q (-1) = 4) ∧ (q 2 = 1) ∧ (q 4 = 10) :=
by
  -- Proof goes here
  sorry

end quadratic_polynomial_correct_l1824_182491


namespace diagonal_BD_l1824_182484

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

end diagonal_BD_l1824_182484


namespace correct_answer_l1824_182469

def total_contestants : Nat := 56
def selected_contestants : Nat := 14

theorem correct_answer :
  (total_contestants = 56) →
  (selected_contestants = 14) →
  (selected_contestants = 14) :=
by
  intro h_total h_selected
  exact h_selected

end correct_answer_l1824_182469


namespace angles_sum_l1824_182481

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

end angles_sum_l1824_182481


namespace xy_value_l1824_182485

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by sorry

end xy_value_l1824_182485


namespace negative_integer_example_l1824_182452

def is_negative_integer (n : ℤ) := n < 0

theorem negative_integer_example : is_negative_integer (-2) :=
by
  -- Proof will go here
  sorry

end negative_integer_example_l1824_182452


namespace power_of_b_l1824_182441

theorem power_of_b (b n : ℕ) (hb : b > 1) (hn : n > 1) (h : ∀ k > 1, ∃ a_k : ℤ, k ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, b = A ^ n :=
by
  sorry

end power_of_b_l1824_182441


namespace largest_digit_B_divisible_by_4_l1824_182464

theorem largest_digit_B_divisible_by_4 :
  ∃ B : ℕ, B = 9 ∧ ∀ k : ℕ, (k ≤ 9 → (∃ n : ℕ, 4 * n = 10 * B + 792 % 100)) :=
by
  sorry

end largest_digit_B_divisible_by_4_l1824_182464


namespace initial_food_days_l1824_182411

theorem initial_food_days (x : ℕ) (h : 760 * (x - 2) = 3040 * 5) : x = 22 := by
  sorry

end initial_food_days_l1824_182411


namespace mary_age_proof_l1824_182447

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof_l1824_182447


namespace smallest_possible_a_l1824_182499

theorem smallest_possible_a (a b c : ℝ) 
  (h1 : (∀ x, y = a * x ^ 2 + b * x + c ↔ y = a * (x + 1/3) ^ 2 + 5/9))
  (h2 : a > 0)
  (h3 : ∃ n : ℤ, a + b + c = n) : 
  a = 1/4 :=
sorry

end smallest_possible_a_l1824_182499


namespace degrees_to_minutes_l1824_182450

theorem degrees_to_minutes (d : ℚ) (fractional_part : ℚ) (whole_part : ℤ) :
  1 ≤ d ∧ d = fractional_part + whole_part ∧ fractional_part = 0.45 ∧ whole_part = 1 →
  (whole_part + fractional_part) * 60 = 1 * 60 + 27 :=
by { sorry }

end degrees_to_minutes_l1824_182450


namespace score_difference_l1824_182478

-- Definitions of the given conditions
def Layla_points : ℕ := 70
def Total_points : ℕ := 112

-- The statement to be proven
theorem score_difference : (Layla_points - (Total_points - Layla_points)) = 28 :=
by sorry

end score_difference_l1824_182478


namespace find_a5_l1824_182404

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

end find_a5_l1824_182404


namespace find_y_of_rectangle_area_l1824_182496

theorem find_y_of_rectangle_area (y : ℝ) (h1 : y > 0) 
(h2 : (0, 0) = (0, 0)) (h3 : (0, 6) = (0, 6)) 
(h4 : (y, 6) = (y, 6)) (h5 : (y, 0) = (y, 0)) 
(h6 : 6 * y = 42) : y = 7 :=
by {
  sorry
}

end find_y_of_rectangle_area_l1824_182496


namespace intersection_M_S_l1824_182477

def M := {x : ℕ | 0 < x ∧ x < 4 }

def S : Set ℕ := {2, 3, 5}

theorem intersection_M_S : (M ∩ S) = {2, 3} := by
  sorry

end intersection_M_S_l1824_182477


namespace subtraction_888_55_555_55_l1824_182420

theorem subtraction_888_55_555_55 : 888.88 - 555.55 = 333.33 :=
by
  sorry

end subtraction_888_55_555_55_l1824_182420


namespace plane_equation_l1824_182407

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

end plane_equation_l1824_182407


namespace simplify_expression_l1824_182453

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

end simplify_expression_l1824_182453


namespace determine_OP_l1824_182442

variables (a b c d q : ℝ)
variables (P : ℝ)
variables (h_ratio : (|a - P| / |P - d| = |b - P| / |P - c|))
variables (h_twice : P = 2 * q)

theorem determine_OP : P = 2 * q :=
sorry

end determine_OP_l1824_182442


namespace a_10_equals_1024_l1824_182482

-- Define the sequence a_n and its properties
variable {a : ℕ → ℕ}
variable (h_prop : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)
variable (h_a2 : a 2 = 4)

-- Prove the statement that a_10 = 1024 given the above conditions.
theorem a_10_equals_1024 : a 10 = 1024 :=
sorry

end a_10_equals_1024_l1824_182482


namespace sum_of_cubes_zero_l1824_182489

variables {a b c : ℝ}

theorem sum_of_cubes_zero (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) : a^3 + b^3 + c^3 = 0 :=
sorry

end sum_of_cubes_zero_l1824_182489


namespace calculate_new_shipment_bears_l1824_182487

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

end calculate_new_shipment_bears_l1824_182487


namespace cookie_store_expense_l1824_182430

theorem cookie_store_expense (B D: ℝ) 
  (h₁: D = (1 / 2) * B)
  (h₂: B = D + 20):
  B + D = 60 := by
  sorry

end cookie_store_expense_l1824_182430


namespace copy_pages_count_l1824_182408

-- Definitions and conditions
def cost_per_page : ℕ := 5  -- Cost per page in cents
def total_money : ℕ := 50 * 100  -- Total money in cents

-- Proof goal
theorem copy_pages_count : total_money / cost_per_page = 1000 := 
by sorry

end copy_pages_count_l1824_182408


namespace javier_first_throw_distance_l1824_182472

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

end javier_first_throw_distance_l1824_182472


namespace Rover_has_46_spots_l1824_182428

theorem Rover_has_46_spots (G C R : ℕ) 
  (h1 : G = 5 * C)
  (h2 : C = (1/2 : ℝ) * R - 5)
  (h3 : G + C = 108) : 
  R = 46 :=
by
  sorry

end Rover_has_46_spots_l1824_182428


namespace road_length_10_trees_10_intervals_l1824_182431

theorem road_length_10_trees_10_intervals 
  (n_trees : ℕ) (n_intervals : ℕ) (tree_interval : ℕ) 
  (h_trees : n_trees = 10) (h_intervals : n_intervals = 9) (h_interval_length : tree_interval = 10) : 
  n_intervals * tree_interval = 90 := 
by 
  sorry

end road_length_10_trees_10_intervals_l1824_182431


namespace cinema_chairs_l1824_182463

theorem cinema_chairs (chairs_between : ℕ) (h : chairs_between = 30) :
  chairs_between + 2 = 32 := by
  sorry

end cinema_chairs_l1824_182463


namespace range_of_f_l1824_182403

def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f :
  Set.range f = {0, -1} := 
sorry

end range_of_f_l1824_182403


namespace parabola_tangent_line_l1824_182461

theorem parabola_tangent_line (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b * x + 2 = 2 * x + 3 → a = -1 ∧ b = 4) :=
sorry

end parabola_tangent_line_l1824_182461


namespace problem_fraction_of_complex_numbers_l1824_182427

/--
Given \(i\) is the imaginary unit, prove that \(\frac {1-i}{1+i} = -i\).
-/
theorem problem_fraction_of_complex_numbers (i : ℂ) (h_i : i^2 = -1) : 
  ((1 - i) / (1 + i)) = -i := 
sorry

end problem_fraction_of_complex_numbers_l1824_182427


namespace photo_arrangements_l1824_182410

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

end photo_arrangements_l1824_182410


namespace graph_passes_through_point_l1824_182433

theorem graph_passes_through_point (a : ℝ) (h : a < 0) : (0, 0) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, (1 - a)^x - 1)} :=
by
  sorry

end graph_passes_through_point_l1824_182433


namespace circle_eq_center_tangent_l1824_182434

theorem circle_eq_center_tangent (x y : ℝ) : 
  let center := (5, 4)
  let radius := 4
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 :=
by
  sorry

end circle_eq_center_tangent_l1824_182434


namespace ghost_enter_exit_ways_l1824_182451

theorem ghost_enter_exit_ways : 
  (∃ (enter_win : ℕ) (exit_win : ℕ), enter_win ≠ exit_win ∧ 1 ≤ enter_win ∧ enter_win ≤ 8 ∧ 1 ≤ exit_win ∧ exit_win ≤ 8) →
  ∃ (ways : ℕ), ways = 8 * 7 :=
by
  sorry

end ghost_enter_exit_ways_l1824_182451


namespace bookshop_inventory_l1824_182494

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

end bookshop_inventory_l1824_182494


namespace sheets_in_total_l1824_182457

theorem sheets_in_total (boxes_needed : ℕ) (sheets_per_box : ℕ) (total_sheets : ℕ) 
  (h1 : boxes_needed = 7) (h2 : sheets_per_box = 100) : total_sheets = boxes_needed * sheets_per_box := by
  sorry

end sheets_in_total_l1824_182457


namespace geom_series_common_ratio_l1824_182446

theorem geom_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hNewS : (ar^3) / (1 - r) = S / 27) : r = 1 / 3 :=
by
  sorry

end geom_series_common_ratio_l1824_182446


namespace line_positional_relationship_l1824_182424

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

end line_positional_relationship_l1824_182424


namespace arithmetic_sequence_a2a3_l1824_182483

noncomputable def arithmetic_sequence_sum (a : Nat → ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a (n + 1) = a n + d

theorem arithmetic_sequence_a2a3 
  (a : Nat → ℝ) (d : ℝ) 
  (arith_seq : arithmetic_sequence_sum a d)
  (H : a 1 + a 2 + a 3 + a 4 = 30) : 
  a 2 + a 3 = 15 :=
by 
sorry

end arithmetic_sequence_a2a3_l1824_182483


namespace average_messages_correct_l1824_182476

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

end average_messages_correct_l1824_182476


namespace dropped_score_l1824_182474

variable (A B C D : ℕ)

theorem dropped_score (h1 : A + B + C + D = 180) (h2 : A + B + C = 150) : D = 30 := by
  sorry

end dropped_score_l1824_182474


namespace sickness_temperature_increase_l1824_182435

theorem sickness_temperature_increase :
  ∀ (normal_temp fever_threshold current_temp : ℕ), normal_temp = 95 → fever_threshold = 100 →
  current_temp = fever_threshold + 5 → (current_temp - normal_temp = 10) :=
by
  intros normal_temp fever_threshold current_temp h1 h2 h3
  sorry

end sickness_temperature_increase_l1824_182435


namespace solve_cubic_diophantine_l1824_182468

theorem solve_cubic_diophantine :
  (∃ x y z : ℤ, x^3 + y^3 + z^3 - 3 * x * y * z = 2003) ↔ 
  (x = 667 ∧ y = 668 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 668 ∧ z = 667) :=
sorry

end solve_cubic_diophantine_l1824_182468


namespace sales_in_fourth_month_l1824_182414

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

end sales_in_fourth_month_l1824_182414


namespace fill_pipe_half_time_l1824_182492

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l1824_182492


namespace min_packs_for_soda_l1824_182419

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

end min_packs_for_soda_l1824_182419


namespace range_of_g_l1824_182401

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f (x))))

theorem range_of_g : ∀ x, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by
  intro x h
  sorry

end range_of_g_l1824_182401


namespace S_21_equals_4641_l1824_182432

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

end S_21_equals_4641_l1824_182432


namespace ratio_of_triangle_side_to_rectangle_width_l1824_182436

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

end ratio_of_triangle_side_to_rectangle_width_l1824_182436


namespace sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l1824_182400

-- Problem 1: Define the sum of the first n odd numbers and prove it equals n^2 when n = 5.
theorem sum_first_five_odds_equals_25 : (1 + 3 + 5 + 7 + 9 = 5^2) := 
sorry

-- Problem 2: Prove that if the smallest number in the decomposition of m^3 is 21, then m = 5.
theorem smallest_in_cube_decomposition_eq_21 : 
  (∃ m : ℕ, m > 0 ∧ 21 = 2 * m - 1 ∧ m = 5) := 
sorry

end sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l1824_182400


namespace S10_value_l1824_182413

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ :=
  x^m + (1 / x)^m

theorem S10_value (x : ℝ) (h : x + 1/x = 5) : 
  S_m x 10 = 6430223 := by 
  sorry

end S10_value_l1824_182413


namespace two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l1824_182402

open Nat

theorem two_pow_m_minus_one_not_divide_three_pow_n_minus_one 
  (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hmo : Odd m) (hno : Odd n) : ¬ (∃ k : ℕ, 2^m - 1 = k * (3^n - 1)) := by
  sorry

end two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l1824_182402


namespace number_of_solutions_l1824_182415

theorem number_of_solutions :
  ∃ (sols : Finset ℝ), 
    (∀ x, x ∈ sols → 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 2 * (Real.sin x)^3 - 5 * (Real.sin x)^2 + 2 * Real.sin x = 0) 
    ∧ Finset.card sols = 5 := 
by
  sorry

end number_of_solutions_l1824_182415


namespace additional_books_l1824_182454

theorem additional_books (initial_books total_books additional_books : ℕ)
  (h_initial : initial_books = 54)
  (h_total : total_books = 77) :
  additional_books = total_books - initial_books :=
by
  sorry

end additional_books_l1824_182454


namespace xiaoming_total_money_l1824_182438

def xiaoming_money (x : ℕ) := 9 * x

def fresh_milk_cost (y : ℕ) := 6 * y

def yogurt_cost_equation (x y : ℕ) := y = x + 6

theorem xiaoming_total_money (x : ℕ) (y : ℕ)
  (h1: fresh_milk_cost y = xiaoming_money x)
  (h2: yogurt_cost_equation x y) : xiaoming_money x = 108 := 
  sorry

end xiaoming_total_money_l1824_182438


namespace factorize_expr_l1824_182479

theorem factorize_expr (a : ℝ) : a^2 - 8 * a = a * (a - 8) :=
sorry

end factorize_expr_l1824_182479


namespace winning_majority_vote_l1824_182475

def total_votes : ℕ := 600

def winning_percentage : ℝ := 0.70

def losing_percentage : ℝ := 0.30

theorem winning_majority_vote : (0.70 * (total_votes : ℝ) - 0.30 * (total_votes : ℝ)) = 240 := 
by
  sorry

end winning_majority_vote_l1824_182475


namespace interval_of_monotonic_increase_l1824_182495

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (6 + x - x^2)

theorem interval_of_monotonic_increase :
  {x : ℝ | -2 < x ∧ x < 3} → x ∈ Set.Ioc (1/2) 3 :=
by
  sorry

end interval_of_monotonic_increase_l1824_182495


namespace least_subtracted_number_l1824_182429

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

end least_subtracted_number_l1824_182429


namespace sum_of_coefficients_no_y_l1824_182455

-- Defining the problem conditions
def expansion (a b c : ℤ) (n : ℕ) : ℤ := (a - b + c)^n

-- Summing the coefficients of the terms that do not contain y
noncomputable def coefficients_sum (a b : ℤ) (n : ℕ) : ℤ :=
  (a - b)^n

theorem sum_of_coefficients_no_y (n : ℕ) (h : 0 < n) : 
  coefficients_sum 4 3 n = 1 :=
by
  sorry

end sum_of_coefficients_no_y_l1824_182455


namespace bianca_birthday_money_l1824_182409

-- Define the conditions
def num_friends : ℕ := 5
def money_per_friend : ℕ := 6

-- State the proof problem
theorem bianca_birthday_money : num_friends * money_per_friend = 30 :=
by
  sorry

end bianca_birthday_money_l1824_182409


namespace average_trees_planted_l1824_182470

def A := 225
def B := A + 48
def C := A - 24
def total_trees := A + B + C
def average := total_trees / 3

theorem average_trees_planted :
  average = 233 := by
  sorry

end average_trees_planted_l1824_182470


namespace Rickey_took_30_minutes_l1824_182405

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

end Rickey_took_30_minutes_l1824_182405


namespace zander_construction_cost_l1824_182423

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

end zander_construction_cost_l1824_182423


namespace amber_total_cost_l1824_182449

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

end amber_total_cost_l1824_182449


namespace units_digit_base_6_l1824_182421

theorem units_digit_base_6 (n m : ℕ) (h₁ : n = 312) (h₂ : m = 67) : (312 * 67) % 6 = 0 :=
by {
  sorry
}

end units_digit_base_6_l1824_182421


namespace selling_price_is_correct_l1824_182462

def wholesale_cost : ℝ := 24.35
def gross_profit_percentage : ℝ := 0.15

def gross_profit : ℝ := gross_profit_percentage * wholesale_cost
def selling_price : ℝ := wholesale_cost + gross_profit

theorem selling_price_is_correct :
  selling_price = 28.00 :=
by
  sorry

end selling_price_is_correct_l1824_182462


namespace triangle_equilateral_of_equal_angle_ratios_l1824_182473

theorem triangle_equilateral_of_equal_angle_ratios
  (a b c : ℝ)
  (h₁ : a + b + c = 180)
  (h₂ : a = b)
  (h₃ : b = c) :
  a = 60 ∧ b = 60 ∧ c = 60 :=
by
  sorry

end triangle_equilateral_of_equal_angle_ratios_l1824_182473


namespace career_preference_degrees_l1824_182497

theorem career_preference_degrees (boys girls : ℕ) (ratio_boys_to_girls : boys / gcd boys girls = 2 ∧ girls / gcd boys girls = 3) 
  (boys_preference : ℕ) (girls_preference : ℕ) 
  (h1 : boys_preference = boys / 3)
  (h2 : girls_preference = 2 * girls / 3) : 
  (boys_preference + girls_preference) / (boys + girls) * 360 = 192 :=
by
  sorry

end career_preference_degrees_l1824_182497


namespace determine_y_l1824_182459

-- Define the main problem in a Lean theorem
theorem determine_y (y : ℕ) : 9^10 + 9^10 + 9^10 = 3^y → y = 21 :=
by
  -- proof not required, so we add sorry
  sorry

end determine_y_l1824_182459


namespace overlap_per_connection_is_4_cm_l1824_182466

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

end overlap_per_connection_is_4_cm_l1824_182466


namespace problem_solution_l1824_182465

noncomputable def g (x : ℝ) (P : ℝ) (Q : ℝ) (R : ℝ) : ℝ := x^2 / (P * x^2 + Q * x + R)

theorem problem_solution (P Q R : ℤ) 
  (h1 : ∀ x > 5, g x P Q R > 0.5)
  (h2 : P * (-3)^2 + Q * (-3) + R = 0)
  (h3 : P * 4^2 + Q * 4 + R = 0)
  (h4 : ∃ y : ℝ, y = 1 / P ∧ ∀ x : ℝ, abs (g x P Q R - y) < ε):
  P + Q + R = -24 :=
by
  sorry

end problem_solution_l1824_182465


namespace find_r_l1824_182490

variable (m r : ℝ)

theorem find_r (h1 : 5 = m * 3^r) (h2 : 45 = m * 9^(2 * r)) : r = 2 / 3 := by
  sorry

end find_r_l1824_182490


namespace find_s_l1824_182498

theorem find_s : ∃ s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 :=
by
  sorry

end find_s_l1824_182498


namespace fraction_to_decimal_l1824_182488

theorem fraction_to_decimal : (5 / 8 : ℝ) = 0.625 := 
  by sorry

end fraction_to_decimal_l1824_182488


namespace Aimee_escalator_time_l1824_182437

theorem Aimee_escalator_time (d : ℝ) (v_esc : ℝ) (v_walk : ℝ) :
  v_esc = d / 60 → v_walk = d / 90 → (d / (v_esc + v_walk)) = 36 :=
by
  intros h1 h2
  sorry

end Aimee_escalator_time_l1824_182437


namespace summer_camp_students_l1824_182444

theorem summer_camp_students (x : ℕ)
  (h1 : (1 / 6) * x = n_Shanghai)
  (h2 : n_Tianjin = 24)
  (h3 : (1 / 4) * x = n_Chongqing)
  (h4 : n_Beijing = (3 / 2) * (n_Shanghai + n_Tianjin)) :
  x = 180 :=
by
  sorry

end summer_camp_students_l1824_182444
