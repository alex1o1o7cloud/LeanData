import Mathlib

namespace sum_of_digits_decrease_by_10_percent_l1669_166940

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum -- Assuming this method computes the sum of the digits

theorem sum_of_digits_decrease_by_10_percent :
  ∃ (n m : ℕ), m = 11 * n / 10 ∧ sum_of_digits m = 9 * sum_of_digits n / 10 :=
by
  sorry

end sum_of_digits_decrease_by_10_percent_l1669_166940


namespace fraction_increase_by_two_times_l1669_166959

theorem fraction_increase_by_two_times (x y : ℝ) : 
  let new_val := ((2 * x) * (2 * y)) / (2 * x + 2 * y)
  let original_val := (x * y) / (x + y)
  new_val = 2 * original_val := 
by
  sorry

end fraction_increase_by_two_times_l1669_166959


namespace red_button_probability_l1669_166957

theorem red_button_probability :
  let jarA_red := 6
  let jarA_blue := 9
  let jarA_total := jarA_red + jarA_blue
  let jarA_half := jarA_total / 2
  let removed_total := jarA_total - jarA_half
  let removed_red := removed_total / 2
  let removed_blue := removed_total / 2
  let jarA_red_remaining := jarA_red - removed_red
  let jarA_blue_remaining := jarA_blue - removed_blue
  let jarB_red := removed_red
  let jarB_blue := removed_blue
  let jarA_total_remaining := jarA_red_remaining + jarA_blue_remaining
  let jarB_total := jarB_red + jarB_blue
  (jarA_total = 15) →
  (jarA_red_remaining = 6 - removed_red) →
  (jarA_blue_remaining = 9 - removed_blue) →
  (jarB_red = removed_red) →
  (jarB_blue = removed_blue) →
  (jarA_red_remaining + jarA_blue_remaining = 9) →
  (jarB_red + jarB_blue = 6) →
  let prob_red_JarA := jarA_red_remaining / jarA_total_remaining
  let prob_red_JarB := jarB_red / jarB_total
  prob_red_JarA * prob_red_JarB = 1 / 6 := by sorry

end red_button_probability_l1669_166957


namespace thabo_books_220_l1669_166933

def thabo_books_total (H PNF PF Total : ℕ) : Prop :=
  (H = 40) ∧
  (PNF = H + 20) ∧
  (PF = 2 * PNF) ∧
  (Total = H + PNF + PF)

theorem thabo_books_220 : ∃ H PNF PF Total : ℕ, thabo_books_total H PNF PF 220 :=
by {
  sorry
}

end thabo_books_220_l1669_166933


namespace cost_per_component_l1669_166963

theorem cost_per_component (C : ℝ) : 
  (150 * C + 150 * 4 + 16500 = 150 * 193.33) → 
  C = 79.33 := 
by
  intro h
  sorry

end cost_per_component_l1669_166963


namespace solve_weight_of_bowling_ball_l1669_166932

-- Conditions: Eight bowling balls equal the weight of five canoes
-- and four canoes weigh 120 pounds.

def weight_of_bowling_ball : Prop :=
  ∃ (b c : ℝ), (8 * b = 5 * c) ∧ (4 * c = 120) ∧ (b = 18.75)

theorem solve_weight_of_bowling_ball : weight_of_bowling_ball :=
  sorry

end solve_weight_of_bowling_ball_l1669_166932


namespace multiplication_trick_l1669_166910

theorem multiplication_trick (a b c : ℕ) (h : b + c = 10) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by
  sorry

end multiplication_trick_l1669_166910


namespace pens_cost_l1669_166977

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end pens_cost_l1669_166977


namespace chi_squared_confidence_level_l1669_166923

theorem chi_squared_confidence_level 
  (chi_squared_value : ℝ)
  (p_value_3841 : ℝ)
  (p_value_5024 : ℝ)
  (h1 : chi_squared_value = 4.073)
  (h2 : p_value_3841 = 0.05)
  (h3 : p_value_5024 = 0.025)
  (h4 : 3.841 ≤ chi_squared_value ∧ chi_squared_value < 5.024) :
  ∃ confidence_level : ℝ, confidence_level = 0.95 :=
by 
  sorry

end chi_squared_confidence_level_l1669_166923


namespace total_cost_of_topsoil_l1669_166993

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end total_cost_of_topsoil_l1669_166993


namespace Joan_orange_balloons_l1669_166966

theorem Joan_orange_balloons (originally_has : ℕ) (received : ℕ) (final_count : ℕ) 
  (h1 : originally_has = 8) (h2 : received = 2) : 
  final_count = 10 := by
  sorry

end Joan_orange_balloons_l1669_166966


namespace benny_lost_books_l1669_166982

-- Define the initial conditions
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def total_books : ℕ := sandy_books + tim_books
def remaining_books : ℕ := 19

-- Define the proof problem to find out the number of books Benny lost
theorem benny_lost_books : total_books - remaining_books = 24 :=
by
  sorry -- Insert proof here

end benny_lost_books_l1669_166982


namespace one_bag_covers_250_sqfeet_l1669_166946

noncomputable def lawn_length : ℝ := 22
noncomputable def lawn_width : ℝ := 36
noncomputable def bags_count : ℝ := 4
noncomputable def extra_area : ℝ := 208

noncomputable def lawn_area : ℝ := lawn_length * lawn_width
noncomputable def total_covered_area : ℝ := lawn_area + extra_area
noncomputable def one_bag_area : ℝ := total_covered_area / bags_count

theorem one_bag_covers_250_sqfeet :
  one_bag_area = 250 := 
by
  sorry

end one_bag_covers_250_sqfeet_l1669_166946


namespace hyperbola_equation_foci_shared_l1669_166921

theorem hyperbola_equation_foci_shared :
  ∃ m : ℝ, (∃ c : ℝ, c = 2 * Real.sqrt 2 ∧ 
              ∃ a b : ℝ, a^2 = 12 ∧ b^2 = 4 ∧ c^2 = a^2 - b^2) ∧ 
    (c = 2 * Real.sqrt 2 → (∃ a b : ℝ, a^2 = m ∧ b^2 = m - 8 ∧ c^2 = a^2 + b^2)) → 
  (∃ m : ℝ, m = 7) := 
sorry

end hyperbola_equation_foci_shared_l1669_166921


namespace fruits_eaten_total_l1669_166953

variable (apples blueberries bonnies : ℕ)

noncomputable def total_fruits_eaten : ℕ :=
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 / 4 * third_dog_bonnies
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies

theorem fruits_eaten_total:
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 * third_dog_bonnies / 4
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies = 240 := by
  sorry

end fruits_eaten_total_l1669_166953


namespace handshaking_remainder_div_1000_l1669_166929

/-- Given eleven people where each person shakes hands with exactly three others, 
  let handshaking_count be the number of distinct handshaking arrangements.
  Find the remainder when handshaking_count is divided by 1000. -/
def handshaking_count (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) : Nat :=
  sorry

theorem handshaking_remainder_div_1000 (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) :
  (handshaking_count P hP handshakes H) % 1000 = 800 :=
sorry

end handshaking_remainder_div_1000_l1669_166929


namespace profit_after_discount_l1669_166926

noncomputable def purchase_price : ℝ := 100
noncomputable def increase_rate : ℝ := 0.25
noncomputable def discount_rate : ℝ := 0.10

theorem profit_after_discount :
  let selling_price := purchase_price * (1 + increase_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit := discounted_price - purchase_price
  profit = 12.5 :=
by
  sorry 

end profit_after_discount_l1669_166926


namespace binomial_sum_of_coefficients_l1669_166985

-- Given condition: for the third term in the expansion, the binomial coefficient is 15
def binomial_coefficient_condition (n : ℕ) := Nat.choose n 2 = 15

-- The goal: the sum of the coefficients of all terms in the expansion is 1/64
theorem binomial_sum_of_coefficients (n : ℕ) (h : binomial_coefficient_condition n) :
  (1:ℚ) / (2 : ℚ)^6 = 1 / 64 :=
by 
  have h₁ : n = 6 := by sorry -- Solve for n using the given condition.
  sorry -- Prove the sum of coefficients when x is 1.

end binomial_sum_of_coefficients_l1669_166985


namespace jill_llamas_count_l1669_166987

theorem jill_llamas_count : 
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  herd_after_sell = 18 := 
by
  -- Definitions for the conditions
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  -- Proof will be filled in here.
  sorry

end jill_llamas_count_l1669_166987


namespace unique_cell_50_distance_l1669_166904

-- Define the distance between two cells
def kingDistance (p1 p2 : ℤ × ℤ) : ℤ :=
  max (abs (p1.1 - p2.1)) (abs (p1.2 - p2.2))

-- A condition stating three cells with specific distances
variables (A B C : ℤ × ℤ) (hAB : kingDistance A B = 100) (hBC : kingDistance B C = 100) (hCA : kingDistance C A = 100)

-- A proposition to prove there is exactly one cell at a distance of 50 from all three given cells
theorem unique_cell_50_distance : ∃! D : ℤ × ℤ, kingDistance D A = 50 ∧ kingDistance D B = 50 ∧ kingDistance D C = 50 :=
sorry

end unique_cell_50_distance_l1669_166904


namespace k_value_function_range_l1669_166905

noncomputable def f : ℝ → ℝ := λ x => Real.log x + x

def is_k_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ (∀ x, a ≤ x ∧ x ≤ b → (f x = k * x)) ∧ (k > 0)

theorem k_value_function_range :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.log x + x) →
  (∃ (k : ℝ), is_k_value_function f k) →
  1 < k ∧ k < 1 + (1 / Real.exp 1) :=
by
  sorry

end k_value_function_range_l1669_166905


namespace part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l1669_166989

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l1669_166989


namespace greatest_y_l1669_166902

theorem greatest_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : y ≤ -2 :=
by {
  sorry
}

end greatest_y_l1669_166902


namespace find_W_l1669_166919

def digit_sum_eq (X Y Z W : ℕ) : Prop := X * 10 + Y + Z * 10 + X = W * 10 + X
def digit_diff_eq (X Y Z : ℕ) : Prop := X * 10 + Y - (Z * 10 + X) = X
def is_digit (n : ℕ) : Prop := n < 10

theorem find_W (X Y Z W : ℕ) (h1 : digit_sum_eq X Y Z W) (h2 : digit_diff_eq X Y Z) 
  (hX : is_digit X) (hY : is_digit Y) (hZ : is_digit Z) (hW : is_digit W) : W = 0 := 
sorry

end find_W_l1669_166919


namespace log_division_simplification_l1669_166939

theorem log_division_simplification (log_base_half : ℝ → ℝ) (log_base_half_pow5 :  log_base_half (2 ^ 5) = 5 * log_base_half 2)
  (log_base_half_pow1 : log_base_half (2 ^ 1) = 1 * log_base_half 2) :
  (log_base_half 32) / (log_base_half 2) = 5 :=
sorry

end log_division_simplification_l1669_166939


namespace books_loaned_out_during_month_l1669_166978

-- Define the initial conditions
def initial_books : ℕ := 75
def remaining_books : ℕ := 65
def loaned_out_percentage : ℝ := 0.80
def returned_books_ratio : ℝ := loaned_out_percentage
def not_returned_ratio : ℝ := 1 - returned_books_ratio
def difference : ℕ := initial_books - remaining_books

-- Define the main theorem
theorem books_loaned_out_during_month : ∃ (x : ℕ), not_returned_ratio * (x : ℝ) = (difference : ℝ) ∧ x = 50 :=
by
  existsi 50
  simp [not_returned_ratio, difference]
  sorry

end books_loaned_out_during_month_l1669_166978


namespace largest_number_among_options_l1669_166975

theorem largest_number_among_options :
  let A := 0.983
  let B := 0.9829
  let C := 0.9831
  let D := 0.972
  let E := 0.9819
  C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_among_options_l1669_166975


namespace sum_of_first_five_terms_l1669_166972

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem sum_of_first_five_terms :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 5 / 6 := 
by 
  unfold a
  -- sorry is used as a placeholder for the actual proof
  sorry

end sum_of_first_five_terms_l1669_166972


namespace correct_equation_D_l1669_166947

theorem correct_equation_D : (|5 - 3| = - (3 - 5)) :=
by
  sorry

end correct_equation_D_l1669_166947


namespace pens_each_student_gets_now_l1669_166907

-- Define conditions
def red_pens_per_student := 62
def black_pens_per_student := 43
def num_students := 3
def pens_taken_first_month := 37
def pens_taken_second_month := 41

-- Define total pens bought and remaining pens after each month
def total_pens := num_students * (red_pens_per_student + black_pens_per_student)
def remaining_pens_after_first_month := total_pens - pens_taken_first_month
def remaining_pens_after_second_month := remaining_pens_after_first_month - pens_taken_second_month

-- Theorem statement
theorem pens_each_student_gets_now :
  (remaining_pens_after_second_month / num_students) = 79 :=
by
  sorry

end pens_each_student_gets_now_l1669_166907


namespace simplify_expr_to_polynomial_l1669_166994

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end simplify_expr_to_polynomial_l1669_166994


namespace ordering_of_four_numbers_l1669_166999

variable (m n α β : ℝ)
variable (h1 : m < n)
variable (h2 : α < β)
variable (h3 : 2 * (α - m) * (α - n) - 7 = 0)
variable (h4 : 2 * (β - m) * (β - n) - 7 = 0)

theorem ordering_of_four_numbers : α < m ∧ m < n ∧ n < β :=
by
  sorry

end ordering_of_four_numbers_l1669_166999


namespace ax5_by5_eq_6200_div_29_l1669_166997

variables (a b x y : ℝ)

-- Given conditions
axiom h1 : a * x + b * y = 5
axiom h2 : a * x^2 + b * y^2 = 11
axiom h3 : a * x^3 + b * y^3 = 30
axiom h4 : a * x^4 + b * y^4 = 80

-- Statement to prove
theorem ax5_by5_eq_6200_div_29 : a * x^5 + b * y^5 = 6200 / 29 :=
by
  sorry

end ax5_by5_eq_6200_div_29_l1669_166997


namespace spherical_to_rectangular_coordinates_l1669_166954

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l1669_166954


namespace time_for_b_l1669_166915

theorem time_for_b (A B C : ℚ) (H1 : A + B + C = 1/4) (H2 : A = 1/12) (H3 : C = 1/18) : B = 1/9 :=
by {
  sorry
}

end time_for_b_l1669_166915


namespace calculate_f_g_of_1_l1669_166967

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem calculate_f_g_of_1 : f (g 1) = 39 :=
by
  -- Enable quick skippable proof with 'sorry'
  sorry

end calculate_f_g_of_1_l1669_166967


namespace student_correct_answers_l1669_166913

theorem student_correct_answers (c w : ℕ) 
  (h1 : c + w = 60)
  (h2 : 4 * c - w = 120) : 
  c = 36 :=
sorry

end student_correct_answers_l1669_166913


namespace trigonometric_identity_l1669_166998

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A * Real.cos B * Real.cos C + Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = Real.sin A * Real.sin B * Real.sin C :=
by 
  sorry

end trigonometric_identity_l1669_166998


namespace height_previous_year_l1669_166912

theorem height_previous_year (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) 
  (h1 : current_height = 126)
  (h2 : growth_rate = 0.05) 
  (h3 : current_height = 1.05 * previous_height) : 
  previous_height = 120 :=
sorry

end height_previous_year_l1669_166912


namespace can_determine_counterfeit_l1669_166945

-- Define the conditions of the problem
structure ProblemConditions where
  totalCoins : ℕ := 100
  exaggeration : ℕ

-- Define the problem statement
theorem can_determine_counterfeit (P : ProblemConditions) : 
  ∃ strategy : ℕ → Prop, 
    ∀ (k : ℕ), strategy P.exaggeration -> 
    (∀ i, i < 100 → (P.totalCoins = 100 ∧ ∃ n, n > 0 ∧ 
     ∀ j, j < P.totalCoins → (P.totalCoins = j + 1 ∨ P.totalCoins = 99 + j))) := 
sorry

end can_determine_counterfeit_l1669_166945


namespace brick_height_l1669_166943

/-- A certain number of bricks, each measuring 25 cm x 11.25 cm x some height, 
are needed to build a wall of 8 m x 6 m x 22.5 cm. 
If 6400 bricks are needed, prove that the height of each brick is 6 cm. -/
theorem brick_height (h : ℝ) : 
  6400 * (25 * 11.25 * h) = (800 * 600 * 22.5) → h = 6 :=
by
  sorry

end brick_height_l1669_166943


namespace solve_equation_l1669_166948

theorem solve_equation (x : ℝ) : 2 * x + 17 = 32 - 3 * x → x = 3 := 
by 
  sorry

end solve_equation_l1669_166948


namespace find_cost_price_l1669_166955

theorem find_cost_price (C S : ℝ) (h1 : S = 1.35 * C) (h2 : S - 25 = 0.98 * C) : C = 25 / 0.37 :=
by
  sorry

end find_cost_price_l1669_166955


namespace smallest_distance_l1669_166934

open Complex

noncomputable def a := 2 + 4 * Complex.I
noncomputable def b := 8 + 6 * Complex.I

theorem smallest_distance (z w : ℂ)
    (hz : abs (z - a) = 2)
    (hw : abs (w - b) = 4) :
    abs (z - w) ≥ 2 * Real.sqrt 10 - 6 := by
  sorry

end smallest_distance_l1669_166934


namespace sum_of_digits_of_multiple_of_990_l1669_166988

theorem sum_of_digits_of_multiple_of_990 (a b c : ℕ) (h₀ : a < 10 ∧ b < 10 ∧ c < 10)
  (h₁ : ∃ (d e f g : ℕ), 123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c = 123000 + 9000 + 900 + 90 + 9 + 0)
  (h2 : (123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c) % 990 = 0) :
  a + b + c = 12 :=
by {
  sorry
}

end sum_of_digits_of_multiple_of_990_l1669_166988


namespace total_dots_not_visible_l1669_166922

def total_dots_on_dice (n : ℕ): ℕ := n * 21
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 4 + 5 + 6
def total_dice : ℕ := 4

theorem total_dots_not_visible :
  total_dots_on_dice total_dice - visible_dots = 58 := by
  sorry

end total_dots_not_visible_l1669_166922


namespace coby_travel_time_l1669_166979

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l1669_166979


namespace solve_for_x_l1669_166970

theorem solve_for_x (x : ℝ) (h : (10 - 6 * x)^ (1 / 3) = -2) : x = 3 := 
by
  sorry

end solve_for_x_l1669_166970


namespace sin_600_eq_neg_sqrt_3_div_2_l1669_166930

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l1669_166930


namespace evaluate_expression_l1669_166973

noncomputable def complex_numbers_condition (a b : ℂ) := a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + a * b + b^2 = 0)

theorem evaluate_expression (a b : ℂ) (h : complex_numbers_condition a b) : 
  (a^5 + b^5) / (a + b)^5 = -2 := 
by
  sorry

end evaluate_expression_l1669_166973


namespace circle_equation_l1669_166937

open Real

theorem circle_equation (x y : ℝ) :
  let center := (2, -1)
  let line := (x + y = 7)
  (center.1 - 2)^2 + (center.2 + 1)^2 = 18 :=
by
  sorry

end circle_equation_l1669_166937


namespace evaluate_expression_l1669_166935

theorem evaluate_expression : 101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end evaluate_expression_l1669_166935


namespace expression_equals_one_l1669_166958

theorem expression_equals_one (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_sum : a + b + c = 1) :
  (a^3 * b^3 / ((a^3 - b * c) * (b^3 - a * c)) + a^3 * c^3 / ((a^3 - b * c) * (c^3 - a * b)) +
    b^3 * c^3 / ((b^3 - a * c) * (c^3 - a * b))) = 1 :=
by
  sorry

end expression_equals_one_l1669_166958


namespace carrots_picked_first_day_l1669_166927

theorem carrots_picked_first_day (X : ℕ) 
  (H1 : X - 10 + 47 = 60) : X = 23 :=
by 
  -- We state the proof steps here, completing the proof with sorry
  sorry

end carrots_picked_first_day_l1669_166927


namespace Michael_needs_more_money_l1669_166906

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end Michael_needs_more_money_l1669_166906


namespace number_of_female_students_in_sample_l1669_166960

theorem number_of_female_students_in_sample (male_students female_students sample_size : ℕ)
  (h1 : male_students = 560)
  (h2 : female_students = 420)
  (h3 : sample_size = 280) :
  (female_students * sample_size) / (male_students + female_students) = 120 := 
sorry

end number_of_female_students_in_sample_l1669_166960


namespace symmetric_point_of_P_l1669_166941

-- Define a point in the Cartesian coordinate system
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define central symmetry with respect to the origin
def symmetric (p : Point) : Point :=
  { x := -p.x, y := -p.y }

-- Given point P with coordinates (1, -2)
def P : Point := { x := 1, y := -2 }

-- The theorem to be proved: the symmetric point of P is (-1, 2)
theorem symmetric_point_of_P :
  symmetric P = { x := -1, y := 2 } :=
by
  -- Proof is omitted.
  sorry

end symmetric_point_of_P_l1669_166941


namespace probability_at_least_one_expired_l1669_166992

theorem probability_at_least_one_expired (total_bottles expired_bottles selected_bottles : ℕ) : 
  total_bottles = 10 → expired_bottles = 3 → selected_bottles = 3 → 
  (∃ probability, probability = 17 / 24) :=
by
  sorry

end probability_at_least_one_expired_l1669_166992


namespace find_C_l1669_166981

noncomputable def A := {x : ℝ | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 6 = 0}
def C := {a : ℝ | (A ∪ (B a)) = A}

theorem find_C : C = {0, 2, 3} := by
  sorry

end find_C_l1669_166981


namespace grandmother_times_older_l1669_166980

variables (M G Gr : ℕ)

-- Conditions
def MilenasAge : Prop := M = 7
def GrandfatherAgeRelation : Prop := Gr = G + 2
def AgeDifferenceRelation : Prop := Gr - M = 58

-- Theorem to prove
theorem grandmother_times_older (h1 : MilenasAge M) (h2 : GrandfatherAgeRelation G Gr) (h3 : AgeDifferenceRelation M Gr) :
  G / M = 9 :=
sorry

end grandmother_times_older_l1669_166980


namespace starting_number_is_100_l1669_166916

theorem starting_number_is_100 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, k = 10 ∧ n = 1000 - (k - 1) * 100) :
  n = 100 := by
  sorry

end starting_number_is_100_l1669_166916


namespace line_equation_l1669_166986

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end line_equation_l1669_166986


namespace average_age_without_teacher_l1669_166949

theorem average_age_without_teacher 
  (A : ℕ) 
  (h : 15 * A + 26 = 16 * (A + 1)) : 
  A = 10 :=
sorry

end average_age_without_teacher_l1669_166949


namespace polynomial_inequality_l1669_166938

theorem polynomial_inequality (x : ℝ) : -6 * x^2 + 2 * x - 8 < 0 :=
sorry

end polynomial_inequality_l1669_166938


namespace consecutive_page_sum_l1669_166990

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) = 2156) : n + (n + 1) = 93 :=
sorry

end consecutive_page_sum_l1669_166990


namespace man_l1669_166976

-- Defining the conditions as variables in Lean
variables (S : ℕ) (M : ℕ)
-- Given conditions
def son_present_age := S = 25
def man_present_age := M = S + 27

-- Goal: the ratio of the man's age to the son's age in two years is 2:1
theorem man's_age_ratio_in_two_years (h1 : son_present_age S) (h2 : man_present_age S M) :
  (M + 2) / (S + 2) = 2 := sorry

end man_l1669_166976


namespace escalator_times_comparison_l1669_166924

variable (v v1 v2 l : ℝ)
variable (h_v_lt_v1 : v < v1)
variable (h_v1_lt_v2 : v1 < v2)

theorem escalator_times_comparison
  (h_cond : v < v1 ∧ v1 < v2) :
  (l / (v1 + v) + l / (v2 - v)) < (l / (v1 - v) + l / (v2 + v)) :=
  sorry

end escalator_times_comparison_l1669_166924


namespace max_complete_dresses_l1669_166962

namespace DressMaking

-- Define the initial quantities of fabric
def initial_silk : ℕ := 600
def initial_satin : ℕ := 400
def initial_chiffon : ℕ := 350

-- Define the quantities given to each of 8 friends
def silk_per_friend : ℕ := 15
def satin_per_friend : ℕ := 10
def chiffon_per_friend : ℕ := 5

-- Define the quantities required to make one dress
def silk_per_dress : ℕ := 5
def satin_per_dress : ℕ := 3
def chiffon_per_dress : ℕ := 2

-- Calculate the remaining quantities
def remaining_silk : ℕ := initial_silk - 8 * silk_per_friend
def remaining_satin : ℕ := initial_satin - 8 * satin_per_friend
def remaining_chiffon : ℕ := initial_chiffon - 8 * chiffon_per_friend

-- Calculate the maximum number of dresses that can be made
def max_dresses_silk : ℕ := remaining_silk / silk_per_dress
def max_dresses_satin : ℕ := remaining_satin / satin_per_dress
def max_dresses_chiffon : ℕ := remaining_chiffon / chiffon_per_dress

-- The main theorem indicating the number of complete dresses
theorem max_complete_dresses : max_dresses_silk = 96 ∧ max_dresses_silk ≤ max_dresses_satin ∧ max_dresses_silk ≤ max_dresses_chiffon := by
  sorry

end DressMaking

end max_complete_dresses_l1669_166962


namespace find_common_students_l1669_166952

theorem find_common_students
  (total_english : ℕ)
  (total_math : ℕ)
  (difference_only_english_math : ℕ)
  (both_english_math : ℕ) :
  total_english = both_english_math + (both_english_math + 10) →
  total_math = both_english_math + both_english_math →
  difference_only_english_math = 10 →
  total_english = 30 →
  total_math = 20 →
  both_english_math = 10 :=
by
  intros
  sorry

end find_common_students_l1669_166952


namespace sara_initial_quarters_l1669_166996

theorem sara_initial_quarters (total_quarters: ℕ) (dad_gave: ℕ) (initial_quarters: ℕ) 
  (h1: dad_gave = 49) (h2: total_quarters = 70) (h3: total_quarters = initial_quarters + dad_gave) :
  initial_quarters = 21 := 
by {
  sorry
}

end sara_initial_quarters_l1669_166996


namespace sum_of_digits_l1669_166944

noncomputable def digits_divisibility (C F : ℕ) : Prop :=
  (C >= 0 ∧ C <= 9 ∧ F >= 0 ∧ F <= 9) ∧
  (C + 8 + 5 + 4 + F + 7 + 2) % 9 = 0 ∧
  (100 * 4 + 10 * F + 72) % 8 = 0

theorem sum_of_digits (C F : ℕ) (h : digits_divisibility C F) : C + F = 10 :=
sorry

end sum_of_digits_l1669_166944


namespace abc_sum_is_32_l1669_166956

theorem abc_sum_is_32 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b + c = 31) (h5 : b * c + a = 31) (h6 : a * c + b = 31) : 
  a + b + c = 32 := 
by
  -- Proof goes here
  sorry

end abc_sum_is_32_l1669_166956


namespace ordering_y1_y2_y3_l1669_166984

-- Conditions
def A (y₁ : ℝ) : Prop := ∃ b : ℝ, y₁ = -4^2 + 2*4 + b
def B (y₂ : ℝ) : Prop := ∃ b : ℝ, y₂ = -(-1)^2 + 2*(-1) + b
def C (y₃ : ℝ) : Prop := ∃ b : ℝ, y₃ = -(1)^2 + 2*1 + b

-- Question translated to a proof problem
theorem ordering_y1_y2_y3 (y₁ y₂ y₃ : ℝ) :
  A y₁ → B y₂ → C y₃ → y₁ < y₂ ∧ y₂ < y₃ :=
sorry

end ordering_y1_y2_y3_l1669_166984


namespace people_visited_neither_l1669_166936

theorem people_visited_neither (total_people iceland_visitors norway_visitors both_visitors : ℕ)
  (h1 : total_people = 100)
  (h2 : iceland_visitors = 55)
  (h3 : norway_visitors = 43)
  (h4 : both_visitors = 61) :
  total_people - (iceland_visitors + norway_visitors - both_visitors) = 63 :=
by
  sorry

end people_visited_neither_l1669_166936


namespace math_problem_l1669_166950

variable (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c < d)

theorem math_problem : a - c > b - d :=
by {
  sorry
}

end math_problem_l1669_166950


namespace find_unknown_gift_l1669_166969

def money_from_aunt : ℝ := 9
def money_from_uncle : ℝ := 9
def money_from_bestfriend1 : ℝ := 22
def money_from_bestfriend2 : ℝ := 22
def money_from_bestfriend3 : ℝ := 22
def money_from_sister : ℝ := 7
def mean_money : ℝ := 16.3
def number_of_gifts : ℕ := 7

theorem find_unknown_gift (X : ℝ)
  (h1: money_from_aunt = 9)
  (h2: money_from_uncle = 9)
  (h3: money_from_bestfriend1 = 22)
  (h4: money_from_bestfriend2 = 22)
  (h5: money_from_bestfriend3 = 22)
  (h6: money_from_sister = 7)
  (h7: mean_money = 16.3)
  (h8: number_of_gifts = 7)
  : X = 23.1 := sorry

end find_unknown_gift_l1669_166969


namespace average_of_remaining_primes_l1669_166971

theorem average_of_remaining_primes (avg30: ℕ) (avg15: ℕ) (h1 : avg30 = 110) (h2 : avg15 = 95) : 
  ((30 * avg30 - 15 * avg15) / 15) = 125 := 
by
  -- Proof
  sorry

end average_of_remaining_primes_l1669_166971


namespace intersection_of_P_and_Q_l1669_166965

noncomputable def P : Set ℝ := {x | 0 < Real.log x / Real.log 8 ∧ Real.log x / Real.log 8 < 2 * (Real.log 3 / Real.log 8)}
noncomputable def Q : Set ℝ := {x | 2 / (2 - x) > 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l1669_166965


namespace inequality_proof_l1669_166928

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_proof_l1669_166928


namespace working_capacity_ratio_l1669_166914

theorem working_capacity_ratio (team_p_engineers : ℕ) (team_q_engineers : ℕ) (team_p_days : ℕ) (team_q_days : ℕ) :
  team_p_engineers = 20 → team_q_engineers = 16 → team_p_days = 32 → team_q_days = 30 →
  (team_p_days / team_q_days) = (16:ℤ) / (15:ℤ) :=
by
  intros h1 h2 h3 h4
  sorry

end working_capacity_ratio_l1669_166914


namespace weight_feel_when_lowered_l1669_166931

-- Conditions from the problem
def num_plates : ℕ := 10
def weight_per_plate : ℝ := 30
def technology_increase : ℝ := 0.20
def incline_increase : ℝ := 0.15

-- Calculate the contributions
def total_weight_without_factors : ℝ := num_plates * weight_per_plate
def weight_with_technology : ℝ := total_weight_without_factors * (1 + technology_increase)
def weight_with_incline : ℝ := weight_with_technology * (1 + incline_increase)

-- Theorem statement we want to prove
theorem weight_feel_when_lowered : weight_with_incline = 414 := by
  sorry

end weight_feel_when_lowered_l1669_166931


namespace complex_div_l1669_166968

open Complex

theorem complex_div (i : ℂ) (hi : i = Complex.I) : 
  (6 + 7 * i) / (1 + 2 * i) = 4 - i := 
by 
  sorry

end complex_div_l1669_166968


namespace maximum_value_expression_l1669_166901

noncomputable def expression (s t : ℝ) := -2 * s^2 + 24 * s + 3 * t - 38

theorem maximum_value_expression : ∀ (s : ℝ), expression s 4 ≤ 46 :=
by sorry

end maximum_value_expression_l1669_166901


namespace B_equals_1_2_3_l1669_166908

def A : Set ℝ := { x | x^2 ≤ 4 }
def B : Set ℕ := { x | x > 0 ∧ (x - 1:ℝ) ∈ A }

theorem B_equals_1_2_3 : B = {1, 2, 3} :=
by
  sorry

end B_equals_1_2_3_l1669_166908


namespace craig_total_distance_l1669_166918

-- Define the distances Craig walked
def dist_school_to_david : ℝ := 0.27
def dist_david_to_home : ℝ := 0.73

-- Prove the total distance walked
theorem craig_total_distance : dist_school_to_david + dist_david_to_home = 1.00 :=
by
  -- Proof goes here
  sorry

end craig_total_distance_l1669_166918


namespace cost_of_article_l1669_166991

variable (C : ℝ)
variable (SP1 SP2 : ℝ)
variable (G : ℝ)

theorem cost_of_article (h1 : SP1 = 380) 
                        (h2 : SP2 = 420)
                        (h3 : SP1 = C + G)
                        (h4 : SP2 = C + G + 0.08 * G) :
  C = 120 :=
by
  sorry

end cost_of_article_l1669_166991


namespace matrix_B6_eq_sB_plus_tI_l1669_166911

noncomputable section

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![1, -1;
     4, 2]

theorem matrix_B6_eq_sB_plus_tI :
  ∃ s t : ℤ, B^6 = s • B + t • (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  have B2_eq : B^2 = -3 • B :=
    -- Matrix multiplication and scalar multiplication
    sorry
  use 81, 0
  have B4_eq : B^4 = 9 • B^2 := by
    rw [B2_eq]
    -- Calculation steps for B^4 equation
    sorry
  have B6_eq : B^6 = B^4 * B^2 := by
    rw [B4_eq, B2_eq]
    -- Calculation steps for B^6 final equation
    sorry
  rw [B6_eq]
  -- Final steps to show (81 • B + 0 • I = 81 • B)
  sorry

end matrix_B6_eq_sB_plus_tI_l1669_166911


namespace Jo_has_least_l1669_166917

variable (Money : Type) 
variable (Bo Coe Flo Jo Moe Zoe : Money)
variable [LT Money] [LE Money] -- Money type is an ordered type with less than and less than or equal relations.

-- Conditions
axiom h1 : Jo < Flo 
axiom h2 : Flo < Bo
axiom h3 : Jo < Moe
axiom h4 : Moe < Bo
axiom h5 : Bo < Coe
axiom h6 : Coe < Zoe

-- The main statement to prove that Jo has the least money.
theorem Jo_has_least (h1 : Jo < Flo) (h2 : Flo < Bo) (h3 : Jo < Moe) (h4 : Moe < Bo) (h5 : Bo < Coe) (h6 : Coe < Zoe) : 
  ∀ x, x = Jo ∨ x = Bo ∨ x = Flo ∨ x = Moe ∨ x = Coe ∨ x = Zoe → Jo ≤ x :=
by
  -- Proof is skipped using sorry
  sorry

end Jo_has_least_l1669_166917


namespace central_angle_of_sector_l1669_166974

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def arc_length (r α : ℝ) : ℝ := r * α

theorem central_angle_of_sector :
  ∀ (r α : ℝ),
    circumference r = 2 * Real.pi + 2 →
    arc_length r α = 2 * Real.pi - 2 →
    α = Real.pi - 1 :=
by
  intros r α hcirc harc
  sorry

end central_angle_of_sector_l1669_166974


namespace find_x_plus_y_l1669_166995

theorem find_x_plus_y
  (x y : ℤ)
  (hx : |x| = 2)
  (hy : |y| = 3)
  (hxy : x > y) : x + y = -1 := 
sorry

end find_x_plus_y_l1669_166995


namespace hilary_big_toenails_count_l1669_166942

def fit_toenails (total_capacity : ℕ) (big_toenail_space_ratio : ℕ) (current_regular : ℕ) (additional_regular : ℕ) : ℕ :=
  (total_capacity - (current_regular + additional_regular)) / big_toenail_space_ratio

theorem hilary_big_toenails_count :
  fit_toenails 100 2 40 20 = 10 :=
  by
    sorry

end hilary_big_toenails_count_l1669_166942


namespace find_arithmetic_sequence_l1669_166964

theorem find_arithmetic_sequence (a d : ℝ) (h1 : (a - d) + a + (a + d) = 12) (h2 : (a - d) * a * (a + d) = 48) :
  (a = 4 ∧ d = 2) ∨ (a = 4 ∧ d = -2) :=
sorry

end find_arithmetic_sequence_l1669_166964


namespace exists_b_for_a_ge_condition_l1669_166961

theorem exists_b_for_a_ge_condition (a : ℝ) (h : a ≥ -Real.sqrt 2 - 1 / 4) :
  ∃ b : ℝ, ∃ x y : ℝ, 
    y = x^2 - a ∧
    x^2 + y^2 + 8 * b^2 = 4 * b * (y - x) + 1 :=
sorry

end exists_b_for_a_ge_condition_l1669_166961


namespace average_speed_round_trip_l1669_166920

noncomputable def average_speed (d : ℝ) (v_to v_from : ℝ) : ℝ :=
  let time_to := d / v_to
  let time_from := d / v_from
  let total_time := time_to + time_from
  let total_distance := 2 * d
  total_distance / total_time

theorem average_speed_round_trip (d : ℝ) :
  average_speed d 60 40 = 48 :=
by
  sorry

end average_speed_round_trip_l1669_166920


namespace cost_per_candy_bar_l1669_166900

-- Define the conditions as hypotheses
variables (candy_bars_total : ℕ) (candy_bars_paid_by_dave : ℕ) (amount_paid_by_john : ℝ)
-- Assume the given values
axiom total_candy_bars : candy_bars_total = 20
axiom candy_bars_by_dave : candy_bars_paid_by_dave = 6
axiom paid_by_john : amount_paid_by_john = 21

-- Define the proof problem
theorem cost_per_candy_bar :
  (amount_paid_by_john / (candy_bars_total - candy_bars_paid_by_dave) = 1.50) :=
by
  sorry

end cost_per_candy_bar_l1669_166900


namespace correct_ordering_of_fractions_l1669_166983

theorem correct_ordering_of_fractions :
  let a := (6 : ℚ) / 17
  let b := (8 : ℚ) / 25
  let c := (10 : ℚ) / 31
  let d := (1 : ℚ) / 3
  b < d ∧ d < c ∧ c < a :=
by
  sorry

end correct_ordering_of_fractions_l1669_166983


namespace minimum_value_fraction_l1669_166925

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b + c ≥ a)

theorem minimum_value_fraction : (b / c + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end minimum_value_fraction_l1669_166925


namespace volume_ratio_proof_l1669_166951

-- Definitions based on conditions
def edge_ratio (a b : ℝ) : Prop := a = 3 * b
def volume_ratio (V_large V_small : ℝ) : Prop := V_large = 27 * V_small

-- Problem statement
theorem volume_ratio_proof (e V_small V_large : ℝ) 
  (h1 : edge_ratio (3 * e) e)
  (h2 : volume_ratio V_large V_small) : 
  V_large / V_small = 27 := 
by sorry

end volume_ratio_proof_l1669_166951


namespace inverse_function_domain_l1669_166903

noncomputable def f (x : ℝ) : ℝ := -3 + Real.log (x - 1) / Real.log 2

theorem inverse_function_domain :
  ∀ x : ℝ, x ≥ 5 → ∃ y : ℝ, f x = y ∧ y ≥ -1 :=
by
  intro x hx
  use f x
  sorry

end inverse_function_domain_l1669_166903


namespace john_text_messages_per_day_l1669_166909

theorem john_text_messages_per_day (m n : ℕ) (h1 : m = 20) (h2 : n = 245) : 
  m + n / 7 = 55 :=
by
  sorry

end john_text_messages_per_day_l1669_166909
