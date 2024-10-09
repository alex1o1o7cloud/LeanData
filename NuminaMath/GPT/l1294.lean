import Mathlib

namespace sin_half_angle_l1294_129415

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l1294_129415


namespace tiling_scheme_3_3_3_3_6_l1294_129493

-- Definitions based on the conditions.
def angle_equilateral_triangle := 60
def angle_regular_hexagon := 120

-- The theorem states that using four equilateral triangles and one hexagon around a point forms a valid tiling.
theorem tiling_scheme_3_3_3_3_6 : 
  4 * angle_equilateral_triangle + angle_regular_hexagon = 360 := 
by
  -- Skip the proof with sorry
  sorry

end tiling_scheme_3_3_3_3_6_l1294_129493


namespace Jason_more_blue_marbles_l1294_129421

theorem Jason_more_blue_marbles (Jason_blue_marbles Tom_blue_marbles : ℕ) 
  (hJ : Jason_blue_marbles = 44) (hT : Tom_blue_marbles = 24) :
  Jason_blue_marbles - Tom_blue_marbles = 20 :=
by
  sorry

end Jason_more_blue_marbles_l1294_129421


namespace compute_nested_operations_l1294_129474

def operation (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem compute_nested_operations :
  operation 5 (operation 6 (operation 7 (operation 8 9))) = 3588 / 587 :=
  sorry

end compute_nested_operations_l1294_129474


namespace gcd_of_polynomials_l1294_129461

theorem gcd_of_polynomials (b : ℤ) (h : b % 2 = 1 ∧ 8531 ∣ b) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 :=
by
  sorry

end gcd_of_polynomials_l1294_129461


namespace math_proof_problem_l1294_129483

noncomputable def a : ℝ := Real.sqrt 18
noncomputable def b : ℝ := (-1 / 3) ^ (-2 : ℤ)
noncomputable def c : ℝ := abs (-3 * Real.sqrt 2)
noncomputable def d : ℝ := (1 - Real.sqrt 2) ^ 0

theorem math_proof_problem : a - b - c - d = -10 := by
  -- Sorry is used to skip the proof, as the proof steps are not required for this problem.
  sorry

end math_proof_problem_l1294_129483


namespace remainder_of_product_l1294_129475

theorem remainder_of_product (a b c : ℕ) (h₁ : a % 7 = 3) (h₂ : b % 7 = 4) (h₃ : c % 7 = 5) :
  (a * b * c) % 7 = 4 :=
by
  sorry

end remainder_of_product_l1294_129475


namespace final_price_on_monday_l1294_129429

-- Definitions based on the conditions
def saturday_price : ℝ := 50
def sunday_increase : ℝ := 1.2
def monday_discount : ℝ := 0.2

-- The statement to prove
theorem final_price_on_monday : 
  let sunday_price := saturday_price * sunday_increase
  let monday_price := sunday_price * (1 - monday_discount)
  monday_price = 48 :=
by
  sorry

end final_price_on_monday_l1294_129429


namespace sin_half_angle_l1294_129410

theorem sin_half_angle 
  (θ : ℝ) 
  (h_cos : |Real.cos θ| = 1 / 5) 
  (h_theta : 5 * Real.pi / 2 < θ ∧ θ < 3 * Real.pi)
  : Real.sin (θ / 2) = - (Real.sqrt 15) / 5 := 
by
  sorry

end sin_half_angle_l1294_129410


namespace find_x_plus_y_l1294_129473

theorem find_x_plus_y (x y : ℤ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) : x + y = -1 ∨ x + y = -5 :=
sorry

end find_x_plus_y_l1294_129473


namespace dot_product_necessity_l1294_129463

variables (a b : ℝ → ℝ → ℝ)

def dot_product (a b : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
  a x y * b x y

def angle_is_acute (a b : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  0 < a x y

theorem dot_product_necessity (a b : ℝ → ℝ → ℝ) (x y : ℝ) :
  dot_product a b x y > 0 ↔ angle_is_acute a b x y :=
sorry

end dot_product_necessity_l1294_129463


namespace sufficient_but_not_necessary_condition_l1294_129447

noncomputable def f (x a : ℝ) : ℝ := (x + 1) / x + Real.sin x - a^2

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a = 1) : 
  (∀ x, f x a + f (-x) a = 0) ↔ (a = 1) ∨ (a = -1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1294_129447


namespace solve_xyz_l1294_129497

theorem solve_xyz (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) :
  (x / 21) * (y / 189) + z = 1 ↔ x = 21 ∧ y = 567 ∧ z = 0 :=
sorry

end solve_xyz_l1294_129497


namespace alice_age_30_l1294_129492

variable (A T : ℕ)

def tom_younger_alice (A T : ℕ) := T = A - 15
def ten_years_ago (A T : ℕ) := A - 10 = 4 * (T - 10)

theorem alice_age_30 (A T : ℕ) (h1 : tom_younger_alice A T) (h2 : ten_years_ago A T) : A = 30 := 
by sorry

end alice_age_30_l1294_129492


namespace minimum_magnitude_l1294_129434

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem minimum_magnitude (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3 * Complex.I) = 15) :
  smallest_magnitude_z z = (768 / 265 : ℝ) :=
by
  sorry

end minimum_magnitude_l1294_129434


namespace vector_parallel_eq_l1294_129477

theorem vector_parallel_eq (m : ℝ) : 
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  a.1 * b.2 = a.2 * b.1 -> m = -6 := 
by 
  sorry

end vector_parallel_eq_l1294_129477


namespace gcd_2183_1947_l1294_129494

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := 
by 
  sorry

end gcd_2183_1947_l1294_129494


namespace common_point_of_function_and_inverse_l1294_129498

-- Define the points P, Q, M, and N
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (1, 2)
def M : ℝ × ℝ := (2, 3)
def N : ℝ × ℝ := (0.5, 0.25)

-- Define a predicate to check if a point lies on the line y = x
def lies_on_y_eq_x (point : ℝ × ℝ) : Prop := point.1 = point.2

-- The main theorem statement
theorem common_point_of_function_and_inverse (a : ℝ) : 
  lies_on_y_eq_x P ∧ ¬ lies_on_y_eq_x Q ∧ ¬ lies_on_y_eq_x M ∧ ¬ lies_on_y_eq_x N :=
by
  -- We write 'sorry' here to skip the proof
  sorry

end common_point_of_function_and_inverse_l1294_129498


namespace critical_force_rod_truncated_cone_l1294_129489

-- Define the given conditions
variable (r0 : ℝ) (q : ℝ) (E : ℝ) (l : ℝ) (π : ℝ)

-- Assumptions
axiom q_positive : q > 0

-- Definition for the new radius based on q
def r1 : ℝ := r0 * (1 + q)

-- Proof problem statement
theorem critical_force_rod_truncated_cone (h : q > 0) : 
  ∃ Pkp : ℝ, Pkp = (E * π * r0^4 * 4.743 / l^2) * (1 + 2 * q) :=
sorry

end critical_force_rod_truncated_cone_l1294_129489


namespace least_possible_value_l1294_129418

theorem least_possible_value (x y : ℝ) : 
  ∃ (x y : ℝ), (xy + 1)^2 + (x + y + 1)^2 = 0 := 
sorry

end least_possible_value_l1294_129418


namespace debby_pancakes_l1294_129452

def total_pancakes (B A P : ℕ) : ℕ := B + A + P

theorem debby_pancakes : 
  total_pancakes 20 24 23 = 67 := by 
  sorry

end debby_pancakes_l1294_129452


namespace evaluate_expression_l1294_129470

theorem evaluate_expression (a b c : ℤ) 
  (h1 : c = b - 12) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2) * (b + 1) / (b - 3) * (c + 10) / (c + 7) = 10 / 3) :=
by
  sorry

end evaluate_expression_l1294_129470


namespace cell_count_at_end_of_days_l1294_129499

-- Defining the conditions
def initial_cells : ℕ := 2
def split_ratio : ℕ := 3
def days : ℕ := 9
def cycle_days : ℕ := 3

-- The main statement to be proved
theorem cell_count_at_end_of_days :
  (initial_cells * split_ratio^((days / cycle_days) - 1)) = 18 :=
by
  sorry

end cell_count_at_end_of_days_l1294_129499


namespace f_strictly_decreasing_l1294_129481

-- Define the function g(x) = x^2 - 2x - 3
def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the function f(x) = log_{1/2}(g(x))
noncomputable def f (x : ℝ) : ℝ := Real.log (g x) / Real.log (1 / 2)

-- The problem statement to prove: f(x) is strictly decreasing on the interval (3, ∞)
theorem f_strictly_decreasing : ∀ x y : ℝ, 3 < x → x < y → f y < f x := by
  sorry

end f_strictly_decreasing_l1294_129481


namespace inequality_proof_l1294_129405

theorem inequality_proof 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a ≤ 2 * b) 
  (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a ^ 2 + b ^ 2) ∧ 2 * (a ^ 2 + b ^ 2) ≤ 5 * a * b := 
by
  sorry

end inequality_proof_l1294_129405


namespace square_floor_tile_count_l1294_129406

theorem square_floor_tile_count (n : ℕ) (h : 2 * n - 1 = 49) : n^2 = 625 := by
  sorry

end square_floor_tile_count_l1294_129406


namespace rectangle_area_l1294_129401

theorem rectangle_area (c h x : ℝ) (h_pos : 0 < h) (c_pos : 0 < c) : 
  (A : ℝ) = (x * (c * x / h)) :=
by
  sorry

end rectangle_area_l1294_129401


namespace diagonals_in_decagon_l1294_129471

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l1294_129471


namespace g_values_l1294_129451

variable (g : ℝ → ℝ)

-- Condition: ∀ x y z ∈ ℝ, g(x^2 + y * g(z)) = x * g(x) + 2 * z * g(y)
axiom g_axiom : ∀ x y z : ℝ, g (x^2 + y * g z) = x * g x + 2 * z * g y

-- Proposition: The possible values of g(4) are 0 and 8.
theorem g_values : g 4 = 0 ∨ g 4 = 8 :=
by
  sorry

end g_values_l1294_129451


namespace ratio_of_pieces_l1294_129446

-- Definitions from the conditions
def total_length : ℝ := 28
def shorter_piece_length : ℝ := 8.000028571387755

-- Derived definition
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- Statement to prove the ratio
theorem ratio_of_pieces : 
  (shorter_piece_length / longer_piece_length) = 0.400000571428571 :=
by
  -- Use sorry to skip the proof
  sorry

end ratio_of_pieces_l1294_129446


namespace negation_of_proposition_l1294_129453

theorem negation_of_proposition (a b : ℝ) :
  ¬(a > b → 2 * a > 2 * b) ↔ (a ≤ b → 2 * a ≤ 2 * b) :=
by
  sorry

end negation_of_proposition_l1294_129453


namespace function_domain_l1294_129450

theorem function_domain (x : ℝ) :
  (x - 3 > 0) ∧ (5 - x ≥ 0) ↔ (3 < x ∧ x ≤ 5) :=
by
  sorry

end function_domain_l1294_129450


namespace quadratic_to_standard_form_div_l1294_129491

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end quadratic_to_standard_form_div_l1294_129491


namespace ellipse_condition_l1294_129432

theorem ellipse_condition (k : ℝ) :
  (4 < k ∧ k < 9) ↔ (9 - k > 0 ∧ k - 4 > 0 ∧ 9 - k ≠ k - 4) :=
by sorry

end ellipse_condition_l1294_129432


namespace min_value_l1294_129423

open Real

-- Definitions
variables (a b : ℝ)
axiom a_gt_zero : a > 0
axiom b_gt_one : b > 1
axiom sum_eq : a + b = 3 / 2

-- The theorem to be proved.
theorem min_value (a : ℝ) (b : ℝ) (a_gt_zero : a > 0) (b_gt_one : b > 1) (sum_eq : a + b = 3 / 2) :
  ∃ (m : ℝ), m = 6 + 4 * sqrt 2 ∧ ∀ (x y : ℝ), (x > 0) → (y > 1) → (x + y = 3 / 2) → (∃ (z : ℝ), z = 2 / x + 1 / (y - 1) ∧ z ≥ m) :=
sorry

end min_value_l1294_129423


namespace shopkeeper_percentage_profit_l1294_129416

variable {x : ℝ} -- cost price per kg of apples

theorem shopkeeper_percentage_profit 
  (total_weight : ℝ)
  (first_half_sold_at : ℝ)
  (second_half_sold_at : ℝ)
  (first_half_profit : ℝ)
  (second_half_profit : ℝ)
  (total_cost_price : ℝ)
  (total_selling_price : ℝ)
  (total_profit : ℝ)
  (percentage_profit : ℝ) :
  total_weight = 100 →
  first_half_sold_at = 0.5 * total_weight →
  second_half_sold_at = 0.5 * total_weight →
  first_half_profit = 25 →
  second_half_profit = 30 →
  total_cost_price = x * total_weight →
  total_selling_price = (first_half_sold_at * (1 + first_half_profit / 100) * x) + (second_half_sold_at * (1 + second_half_profit / 100) * x) →
  total_profit = total_selling_price - total_cost_price →
  percentage_profit = (total_profit / total_cost_price) * 100 →
  percentage_profit = 27.5 := by
  sorry

end shopkeeper_percentage_profit_l1294_129416


namespace g_is_even_and_symmetric_l1294_129495

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem g_is_even_and_symmetric :
  (∀ x : ℝ, g x = g (-x)) ∧ (∀ k : ℤ, g ((2 * k - 1) * π / 8) = 0) :=
by
  sorry

end g_is_even_and_symmetric_l1294_129495


namespace derivative_at_pi_l1294_129419

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_at_pi :
  deriv f π = -1 / (π^2) :=
sorry

end derivative_at_pi_l1294_129419


namespace range_of_a_l1294_129403

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^3 - 3 * a^2 * x + 1 ≠ 3)) 
  → (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l1294_129403


namespace rectangle_area_l1294_129407

theorem rectangle_area
  (width : ℕ) (length : ℕ)
  (h1 : width = 7)
  (h2 : length = 4 * width) :
  length * width = 196 := by
  sorry

end rectangle_area_l1294_129407


namespace area_of_border_l1294_129414

theorem area_of_border
  (h_photo : Nat := 9)
  (w_photo : Nat := 12)
  (border_width : Nat := 3) :
  (let area_photo := h_photo * w_photo
    let h_frame := h_photo + 2 * border_width
    let w_frame := w_photo + 2 * border_width
    let area_frame := h_frame * w_frame
    let area_border := area_frame - area_photo
    area_border = 162) := 
  sorry

end area_of_border_l1294_129414


namespace sum_lent_out_l1294_129455

theorem sum_lent_out (P R : ℝ) (h1 : 780 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 684 := 
  sorry

end sum_lent_out_l1294_129455


namespace bill_toilet_paper_duration_l1294_129458

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end bill_toilet_paper_duration_l1294_129458


namespace correct_standardized_statement_l1294_129462

-- Define and state the conditions as Lean 4 definitions and propositions
structure GeometricStatement :=
  (description : String)
  (is_standardized : Prop)

def optionA : GeometricStatement := {
  description := "Line a and b intersect at point m",
  is_standardized := False -- due to use of lowercase 'm'
}

def optionB : GeometricStatement := {
  description := "Extend line AB",
  is_standardized := False -- since a line cannot be further extended
}

def optionC : GeometricStatement := {
  description := "Extend ray AO (where O is the endpoint) in the opposite direction",
  is_standardized := False -- incorrect definition of ray extension
}

def optionD : GeometricStatement := {
  description := "Extend line segment AB to C such that BC=AB",
  is_standardized := True -- correct by geometric principles
}

-- The theorem stating that option D is the correct and standardized statement
theorem correct_standardized_statement : optionD.is_standardized = True ∧
                                         optionA.is_standardized = False ∧
                                         optionB.is_standardized = False ∧
                                         optionC.is_standardized = False :=
  by sorry

end correct_standardized_statement_l1294_129462


namespace regular_octagon_exterior_angle_l1294_129428

theorem regular_octagon_exterior_angle : 
  ∀ (n : ℕ), n = 8 → (180 * (n - 2) / n) + (180 - (180 * (n - 2) / n)) = 180 := by
  sorry

end regular_octagon_exterior_angle_l1294_129428


namespace probability_of_same_color_correct_l1294_129460

/-- Define events and their probabilities based on the given conditions --/
def probability_of_two_black_stones : ℚ := 1 / 7
def probability_of_two_white_stones : ℚ := 12 / 35

/-- Define the probability of drawing two stones of the same color --/
def probability_of_two_same_color_stones : ℚ :=
  probability_of_two_black_stones + probability_of_two_white_stones

theorem probability_of_same_color_correct :
  probability_of_two_same_color_stones = 17 / 35 :=
by
  -- We only set up the theorem, the proof is not considered here
  sorry

end probability_of_same_color_correct_l1294_129460


namespace rental_cost_equal_mileage_l1294_129431

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end rental_cost_equal_mileage_l1294_129431


namespace percentage_apples_sold_l1294_129480

noncomputable def original_apples : ℝ := 750
noncomputable def remaining_apples : ℝ := 300

theorem percentage_apples_sold (A P : ℝ) (h1 : A = 750) (h2 : A * (1 - P / 100) = 300) : 
  P = 60 :=
by
  sorry

end percentage_apples_sold_l1294_129480


namespace hall_width_l1294_129487

theorem hall_width 
  (L H cost total_expenditure : ℕ)
  (W : ℕ)
  (h1 : L = 20)
  (h2 : H = 5)
  (h3 : cost = 20)
  (h4 : total_expenditure = 19000)
  (h5 : total_expenditure = (L * W + 2 * (H * L) + 2 * (H * W)) * cost) :
  W = 25 := 
sorry

end hall_width_l1294_129487


namespace five_letter_word_with_at_least_one_consonant_l1294_129433

def letter_set : Set Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Set Char := {'B', 'C', 'D', 'F'}
def vowels : Set Char := {'A', 'E'}

-- Calculate the total number of 5-letter words using the letter set
def total_words : ℕ := 6^5

-- Calculate the number of 5-letter words using only vowels
def vowel_only_words : ℕ := 2^5

-- Number of 5-letter words with at least one consonant
def words_with_consonant : ℕ := total_words - vowel_only_words

theorem five_letter_word_with_at_least_one_consonant :
  words_with_consonant = 7744 :=
by
  sorry

end five_letter_word_with_at_least_one_consonant_l1294_129433


namespace irreducible_fraction_l1294_129420

-- Statement of the theorem
theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry -- Proof would be placed here

end irreducible_fraction_l1294_129420


namespace expression_eq_one_if_and_only_if_k_eq_one_l1294_129424

noncomputable def expression (a b c k : ℝ) :=
  (k * a^2 * b^2 + a^2 * c^2 + b^2 * c^2) /
  ((a^2 - b * c) * (b^2 - a * c) + (a^2 - b * c) * (c^2 - a * b) + (b^2 - a * c) * (c^2 - a * b))

theorem expression_eq_one_if_and_only_if_k_eq_one
  (a b c k : ℝ) (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  expression a b c k = 1 ↔ k = 1 :=
by
  sorry

end expression_eq_one_if_and_only_if_k_eq_one_l1294_129424


namespace major_axis_length_l1294_129485

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by sorry

end major_axis_length_l1294_129485


namespace repeating_decimal_to_fraction_l1294_129412

noncomputable def x : ℚ := 0.6 + 41 / 990  

theorem repeating_decimal_to_fraction (h : x = 0.6 + 41 / 990) : x = 127 / 198 :=
by sorry

end repeating_decimal_to_fraction_l1294_129412


namespace determine_a1_a2_a3_l1294_129457

theorem determine_a1_a2_a3 (a a1 a2 a3 : ℝ)
  (h : ∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3) :
  a1 + a2 + a3 = 19 :=
by
  sorry

end determine_a1_a2_a3_l1294_129457


namespace customers_who_bought_four_paintings_each_l1294_129439

/-- Tracy's art fair conditions:
- 20 people came to look at the art
- Four customers bought two paintings each
- Twelve customers bought one painting each
- Tracy sold a total of 36 paintings

We need to prove the number of customers who bought four paintings each. -/
theorem customers_who_bought_four_paintings_each:
  let total_customers := 20
  let customers_bought_two_paintings := 4
  let customers_bought_one_painting := 12
  let total_paintings_sold := 36
  let paintings_per_customer_buying_two := 2
  let paintings_per_customer_buying_one := 1
  let paintings_per_customer_buying_four := 4
  (customers_bought_two_paintings * paintings_per_customer_buying_two +
   customers_bought_one_painting * paintings_per_customer_buying_one +
   x * paintings_per_customer_buying_four = total_paintings_sold) →
  (customers_bought_two_paintings + customers_bought_one_painting + x = total_customers) →
  x = 4 :=
by
  intro h1 h2
  sorry

end customers_who_bought_four_paintings_each_l1294_129439


namespace jake_peaches_is_seven_l1294_129440

-- Definitions based on conditions
def steven_peaches : ℕ := 13
def jake_peaches (steven : ℕ) : ℕ := steven - 6

-- The theorem we want to prove
theorem jake_peaches_is_seven : jake_peaches steven_peaches = 7 := sorry

end jake_peaches_is_seven_l1294_129440


namespace zach_needs_more_tickets_l1294_129484

theorem zach_needs_more_tickets {ferris_wheel_tickets roller_coaster_tickets log_ride_tickets zach_tickets : ℕ} :
  ferris_wheel_tickets = 2 ∧
  roller_coaster_tickets = 7 ∧
  log_ride_tickets = 1 ∧
  zach_tickets = 1 →
  (ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - zach_tickets = 9) :=
by
  intro h
  sorry

end zach_needs_more_tickets_l1294_129484


namespace largest_fraction_among_given_l1294_129482

theorem largest_fraction_among_given (f1 f2 f3 f4 f5 : ℚ)
  (h1 : f1 = 2/5) 
  (h2 : f2 = 4/9) 
  (h3 : f3 = 7/15) 
  (h4 : f4 = 11/18) 
  (h5 : f5 = 16/35) 
  : f1 < f4 ∧ f2 < f4 ∧ f3 < f4 ∧ f5 < f4 :=
by
  sorry

end largest_fraction_among_given_l1294_129482


namespace even_function_value_l1294_129427

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_value (h_even : ∀ x, f a b x = f a b (-x))
    (h_domain : a - 1 = -2 * a) :
    f a (0 : ℝ) (1 / 2) = 13 / 12 :=
by
  sorry

end even_function_value_l1294_129427


namespace competition_problem_l1294_129437

theorem competition_problem (n : ℕ) (s : ℕ) (correct_first_12 : s = (12 * 13) / 2)
    (gain_708_if_last_12_correct : s + 708 = (n - 11) * (n + 12) / 2):
    n = 71 :=
by
  sorry

end competition_problem_l1294_129437


namespace half_sum_squares_ge_product_l1294_129422

theorem half_sum_squares_ge_product (x y : ℝ) : 
  1 / 2 * (x^2 + y^2) ≥ x * y := 
by 
  sorry

end half_sum_squares_ge_product_l1294_129422


namespace mark_sideline_time_l1294_129488

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end mark_sideline_time_l1294_129488


namespace rectangle_area_problem_l1294_129413

/--
Given a rectangle with dimensions \(3x - 4\) and \(4x + 6\),
show that the area of the rectangle equals \(12x^2 + 2x - 24\) if and only if \(x \in \left(\frac{4}{3}, \infty\right)\).
-/
theorem rectangle_area_problem 
  (x : ℝ) 
  (h1 : 3 * x - 4 > 0)
  (h2 : 4 * x + 6 > 0) :
  (3 * x - 4) * (4 * x + 6) = 12 * x^2 + 2 * x - 24 ↔ x > 4 / 3 :=
sorry

end rectangle_area_problem_l1294_129413


namespace sequence_positions_l1294_129490

noncomputable def position_of_a4k1 (x : ℕ) : ℕ := 4 * x + 1
noncomputable def position_of_a4k2 (x : ℕ) : ℕ := 4 * x + 2
noncomputable def position_of_a4k3 (x : ℕ) : ℕ := 4 * x + 3
noncomputable def position_of_a4k (x : ℕ) : ℕ := 4 * x

theorem sequence_positions (k : ℕ) :
  (6 + 1964 = 1970 ∧ position_of_a4k1 1964 = 7857) ∧
  (6 + 1965 = 1971 ∧ position_of_a4k1 1965 = 7861) ∧
  (8 + 1962 = 1970 ∧ position_of_a4k2 1962 = 7850) ∧
  (8 + 1963 = 1971 ∧ position_of_a4k2 1963 = 7854) ∧
  (16 + 2 * 977 = 1970 ∧ position_of_a4k3 977 = 3911) ∧
  (14 + 2 * (979 - 1) = 1970 ∧ position_of_a4k 979 = 3916) :=
by sorry

end sequence_positions_l1294_129490


namespace bankers_discount_problem_l1294_129476

theorem bankers_discount_problem
  (BD : ℚ) (TD : ℚ) (SD : ℚ)
  (h1 : BD = 36)
  (h2 : TD = 30)
  (h3 : BD = TD + TD^2 / SD) :
  SD = 150 := 
sorry

end bankers_discount_problem_l1294_129476


namespace overlapping_segments_length_l1294_129411

theorem overlapping_segments_length 
    (total_length : ℝ) 
    (actual_distance : ℝ) 
    (num_overlaps : ℕ) 
    (h1 : total_length = 98) 
    (h2 : actual_distance = 83)
    (h3 : num_overlaps = 6) :
    (total_length - actual_distance) / num_overlaps = 2.5 :=
by
  sorry

end overlapping_segments_length_l1294_129411


namespace inequality_one_solution_inequality_two_solution_l1294_129472

theorem inequality_one_solution (x : ℝ) :
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3 / 2) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  (x + 1) / (x - 2) ≤ 2 ↔ (x < 2 ∨ x ≥ 5) :=
sorry

end inequality_one_solution_inequality_two_solution_l1294_129472


namespace part_I_part_II_l1294_129464

variable {a b c : ℝ}
variable (habc : a ∈ Set.Ioi 0)
variable (hbbc : b ∈ Set.Ioi 0)
variable (hcbc : c ∈ Set.Ioi 0)
variable (h_sum : a + b + c = 1)

theorem part_I : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 :=
by
  sorry

theorem part_II : (a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2 :=
by
  sorry

end part_I_part_II_l1294_129464


namespace sqrt_9_minus_1_eq_2_l1294_129425

theorem sqrt_9_minus_1_eq_2 : Real.sqrt 9 - 1 = 2 := by
  sorry

end sqrt_9_minus_1_eq_2_l1294_129425


namespace angle_slope_condition_l1294_129465

theorem angle_slope_condition (α k : Real) (h₀ : k = Real.tan α) (h₁ : 0 ≤ α ∧ α < Real.pi) : 
  (α < Real.pi / 3) → (k < Real.sqrt 3) ∧ ¬((k < Real.sqrt 3) → (α < Real.pi / 3)) := 
sorry

end angle_slope_condition_l1294_129465


namespace positive_root_of_quadratic_eqn_l1294_129402

theorem positive_root_of_quadratic_eqn 
  (b : ℝ)
  (h1 : ∃ x0 : ℝ, x0^2 - 4 * x0 + b = 0 ∧ (-x0)^2 + 4 * (-x0) - b = 0) 
  : ∃ x : ℝ, (x^2 + b * x - 4 = 0) ∧ x = 2 := 
by
  sorry

end positive_root_of_quadratic_eqn_l1294_129402


namespace perpendicular_transfer_l1294_129404

variables {Line Plane : Type} 
variables (a b : Line) (α β : Plane)

def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

theorem perpendicular_transfer
  (h1 : perpendicular a α)
  (h2 : parallel_planes α β) :
  perpendicular a β := 
sorry

end perpendicular_transfer_l1294_129404


namespace reunion_handshakes_l1294_129409

/-- 
Given 15 boys at a reunion:
- 5 are left-handed and will only shake hands with other left-handed boys.
- Each boy shakes hands exactly once with each of the others unless they forget.
- Three boys each forget to shake hands with two others.

Prove that the total number of handshakes is 49. 
-/
theorem reunion_handshakes : 
  let total_boys := 15
  let left_handed := 5
  let forgetful_boys := 3
  let forgotten_handshakes_per_boy := 2

  let total_handshakes := total_boys * (total_boys - 1) / 2
  let left_left_handshakes := left_handed * (left_handed - 1) / 2
  let left_right_handshakes := left_handed * (total_boys - left_handed)
  let distinct_forgotten_handshakes := forgetful_boys * forgotten_handshakes_per_boy / 2

  total_handshakes 
    - left_right_handshakes 
    - distinct_forgotten_handshakes
    - left_left_handshakes
  = 49 := 
sorry

end reunion_handshakes_l1294_129409


namespace width_minimizes_fencing_l1294_129449

-- Define the conditions for the problem
def garden_area_cond (w : ℝ) : Prop :=
  w * (w + 10) ≥ 150

-- Define the main statement to prove
theorem width_minimizes_fencing (w : ℝ) (h : w ≥ 0) : garden_area_cond w → w = 10 :=
  by
  sorry

end width_minimizes_fencing_l1294_129449


namespace negation_of_implication_l1294_129466

theorem negation_of_implication (x : ℝ) :
  (¬ (x = 0 ∨ x = 1) → x^2 - x ≠ 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) :=
by sorry

end negation_of_implication_l1294_129466


namespace smallest_root_of_equation_l1294_129456

theorem smallest_root_of_equation :
  let a := (x : ℝ) - 4 / 5
  let b := (x : ℝ) - 2 / 5
  let c := (x : ℝ) - 1 / 2
  (a^2 + a * b + c^2 = 0) → (x = 4 / 5 ∨ x = 14 / 15) ∧ (min (4 / 5) (14 / 15) = 14 / 15) :=
by
  sorry

end smallest_root_of_equation_l1294_129456


namespace gemstone_necklaces_count_l1294_129478

-- Conditions
def num_bead_necklaces : ℕ := 3
def price_per_necklace : ℕ := 7
def total_earnings : ℕ := 70

-- Proof Problem
theorem gemstone_necklaces_count : (total_earnings - num_bead_necklaces * price_per_necklace) / price_per_necklace = 7 := by
  sorry

end gemstone_necklaces_count_l1294_129478


namespace benny_days_worked_l1294_129469

/-- Benny works 3 hours a day and in total he worked for 18 hours. 
We need to prove that he worked for 6 days. -/
theorem benny_days_worked (hours_per_day : ℕ) (total_hours : ℕ)
  (h1 : hours_per_day = 3)
  (h2 : total_hours = 18) :
  total_hours / hours_per_day = 6 := 
by sorry

end benny_days_worked_l1294_129469


namespace original_cost_price_l1294_129445

theorem original_cost_price (C : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) 
    (h3 : S_new = 1.25 * C - 14.70) (h4 : S_new = 1.04 * C) : C = 70 := 
by {
  sorry
}

end original_cost_price_l1294_129445


namespace cubic_expression_value_l1294_129438

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end cubic_expression_value_l1294_129438


namespace gcd_7429_13356_l1294_129417

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end gcd_7429_13356_l1294_129417


namespace remainder_equivalence_l1294_129436

theorem remainder_equivalence (x y q r : ℕ) (hxy : x = q * y + r) (hy_pos : 0 < y) (h_r : 0 ≤ r ∧ r < y) : 
  ((x - 3 * q * y) % y) = r := 
by 
  sorry

end remainder_equivalence_l1294_129436


namespace union_A_B_inter_complement_A_B_range_a_l1294_129496

-- Define the sets A, B, and C
def A : Set ℝ := { x | 2 < x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | 5 - a < x ∧ x < a }

-- Part (I)
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x < 10 } := sorry

theorem inter_complement_A_B :
  (Set.univ \ A) ∩ B = { x | 7 ≤ x ∧ x < 10 } := sorry

-- Part (II)
theorem range_a (a : ℝ) (h : C a ⊆ B) : a ≤ 3 := sorry

end union_A_B_inter_complement_A_B_range_a_l1294_129496


namespace planes_parallel_l1294_129444

variables {a b c : Type} {α β γ : Type}
variables (h_lines : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Conditions based on the propositions
variables (h1 : parallel α γ)
variables (h2 : parallel β γ)

-- Theorem to prove
theorem planes_parallel (h1: parallel α γ) (h2 : parallel β γ) : parallel α β := 
sorry

end planes_parallel_l1294_129444


namespace no_half_probability_socks_l1294_129430

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l1294_129430


namespace compare_f_values_l1294_129459

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * Real.cos x

theorem compare_f_values :
  f 0 < f (-1 / 3) ∧ f (-1 / 3) < f (2 / 5) :=
by
  sorry

end compare_f_values_l1294_129459


namespace rectangle_area_l1294_129479

theorem rectangle_area (r length width : ℝ) (h_ratio : length = 3 * width) (h_incircle : width = 2 * r) (h_r : r = 7) : length * width = 588 :=
by
  sorry

end rectangle_area_l1294_129479


namespace min_days_equal_duties_l1294_129441

/--
Uncle Chernomor appoints 9 or 10 of the 33 warriors to duty each evening. 
Prove that the minimum number of days such that each warrior has been on duty the same number of times is 7.
-/
theorem min_days_equal_duties (k l m : ℕ) (k_nonneg : 0 ≤ k) (l_nonneg : 0 ≤ l)
  (h : 9 * k + 10 * l = 33 * m) (h_min : k + l = 7) : m = 2 :=
by 
  -- The necessary proof will go here
  sorry

end min_days_equal_duties_l1294_129441


namespace peyton_manning_total_yards_l1294_129435

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end peyton_manning_total_yards_l1294_129435


namespace infinite_solutions_iff_m_eq_2_l1294_129467

theorem infinite_solutions_iff_m_eq_2 (m x y : ℝ) :
  (m*x + 4*y = m + 2 ∧ x + m*y = m) ↔ (m = 2) ∧ (m > 1) :=
by
  sorry

end infinite_solutions_iff_m_eq_2_l1294_129467


namespace simplest_common_denominator_l1294_129468

variable (m n a : ℕ)

theorem simplest_common_denominator (h₁ : m > 0) (h₂ : n > 0) (h₃ : a > 0) :
  ∃ l : ℕ, l = 2 * a^2 := 
sorry

end simplest_common_denominator_l1294_129468


namespace difference_is_cube_sum_1996_impossible_l1294_129426

theorem difference_is_cube (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  M - m = (n - 1)^3 := 
by {
  sorry
}

theorem sum_1996_impossible (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  ¬(1996 ∈ {x | m ≤ x ∧ x ≤ M}) := 
by {
  sorry
}

end difference_is_cube_sum_1996_impossible_l1294_129426


namespace solve_for_a_l1294_129443

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem solve_for_a (a : ℝ) : E a 3 2 = E a 5 3 ↔ a = -1/16 :=
by
  sorry

end solve_for_a_l1294_129443


namespace problem_solution_l1294_129408

theorem problem_solution : ∃ n : ℕ, (n > 0) ∧ (21 - 3 * n > 15) ∧ (∀ m : ℕ, (m > 0) ∧ (21 - 3 * m > 15) → m = n) :=
by
  sorry

end problem_solution_l1294_129408


namespace value_of_other_bills_is_40_l1294_129400

-- Define the conditions using Lean definitions
def class_fund_contains_only_10_and_other_bills (total_amount : ℕ) (num_other_bills num_10_bills : ℕ) : Prop :=
  total_amount = 120 ∧ num_other_bills = 3 ∧ num_10_bills = 2 * num_other_bills

def value_of_each_other_bill (total_amount num_other_bills : ℕ) : ℕ :=
  total_amount / num_other_bills

-- The theorem we want to prove
theorem value_of_other_bills_is_40 (total_amount num_other_bills : ℕ) 
  (h : class_fund_contains_only_10_and_other_bills total_amount num_other_bills (2 * num_other_bills)) :
  value_of_each_other_bill total_amount num_other_bills = 40 := 
by 
  -- We use the conditions here to ensure they are part of the proof even if we skip the actual proof with sorry
  have h1 : total_amount = 120 := by sorry
  have h2 : num_other_bills = 3 := by sorry
  -- Skipping the proof
  sorry

end value_of_other_bills_is_40_l1294_129400


namespace people_on_williams_bus_l1294_129486

theorem people_on_williams_bus
  (P : ℕ)
  (dutch_people : ℕ)
  (dutch_americans : ℕ)
  (window_seats : ℕ)
  (h1 : dutch_people = (3 * P) / 5)
  (h2 : dutch_americans = dutch_people / 2)
  (h3 : window_seats = dutch_americans / 3)
  (h4 : window_seats = 9) : 
  P = 90 :=
sorry

end people_on_williams_bus_l1294_129486


namespace daily_average_books_l1294_129454

theorem daily_average_books (x : ℝ) (h1 : 4 * x + 1.4 * x = 216) : x = 40 :=
by 
  sorry

end daily_average_books_l1294_129454


namespace grazing_months_of_B_l1294_129448

variable (A_cows A_months C_cows C_months D_cows D_months A_rent total_rent : ℕ)
variable (B_cows x : ℕ)

theorem grazing_months_of_B
  (hA_cows : A_cows = 24)
  (hA_months : A_months = 3)
  (hC_cows : C_cows = 35)
  (hC_months : C_months = 4)
  (hD_cows : D_cows = 21)
  (hD_months : D_months = 3)
  (hA_rent : A_rent = 1440)
  (htotal_rent : total_rent = 6500)
  (hB_cows : B_cows = 10) :
  x = 5 := 
sorry

end grazing_months_of_B_l1294_129448


namespace factor_expression_correct_l1294_129442

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_expression_correct (a b c : ℝ) :
  factor_expression a b c = (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_correct_l1294_129442
