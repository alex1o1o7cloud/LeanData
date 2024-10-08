import Mathlib

namespace find_c_l64_64122

-- Define the function f(x)
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Define the first derivative of f(x)
def f_prime (x c : ℝ) : ℝ := 3 * x ^ 2 - 4 * c * x + c ^ 2

-- Define the condition that f(x) has a local maximum at x = 2
def is_local_max (f' : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f' x0 = 0 ∧ (∀ x, x < x0 → f' x > 0) ∧ (∀ x, x > x0 → f' x < 0)

-- The main theorem stating the equivalent proof problem
theorem find_c (c : ℝ) : is_local_max (f_prime 2) 2 → c = 6 := 
  sorry

end find_c_l64_64122


namespace completing_the_square_equation_l64_64165

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l64_64165


namespace complement_union_in_set_l64_64402

open Set

theorem complement_union_in_set {U A B : Set ℕ} 
  (hU : U = {1, 3, 5, 9}) 
  (hA : A = {1, 3, 9}) 
  (hB : B = {1, 9}) : 
  (U \ (A ∪ B)) = {5} := 
  by sorry

end complement_union_in_set_l64_64402


namespace count_valid_sequences_returning_rectangle_l64_64321

/-- The transformations that can be applied to the rectangle -/
inductive Transformation
| rot90   : Transformation
| rot180  : Transformation
| rot270  : Transformation
| reflYeqX  : Transformation
| reflYeqNegX : Transformation

/-- Apply a transformation to a point (x, y) -/
def apply_transformation (t : Transformation) (p : ℝ × ℝ) : ℝ × ℝ :=
match t with
| Transformation.rot90   => (-p.2,  p.1)
| Transformation.rot180  => (-p.1, -p.2)
| Transformation.rot270  => ( p.2, -p.1)
| Transformation.reflYeqX  => ( p.2,  p.1)
| Transformation.reflYeqNegX => (-p.2, -p.1)

/-- Apply a sequence of transformations to a list of points -/
def apply_sequence (seq : List Transformation) (points : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  seq.foldl (λ acc t => acc.map (apply_transformation t)) points

/-- Prove that there are exactly 12 valid sequences of three transformations that return the rectangle to its original position -/
theorem count_valid_sequences_returning_rectangle :
  let rectangle := [(0,0), (6,0), (6,2), (0,2)];
  let transformations := [Transformation.rot90, Transformation.rot180, Transformation.rot270, Transformation.reflYeqX, Transformation.reflYeqNegX];
  let seq_transformations := List.replicate 3 transformations;
  (seq_transformations.filter (λ seq => apply_sequence seq rectangle = rectangle)).length = 12 :=
sorry

end count_valid_sequences_returning_rectangle_l64_64321


namespace john_bought_packs_l64_64705

def students_in_classes : List ℕ := [20, 25, 18, 22, 15]
def packs_per_student : ℕ := 3

theorem john_bought_packs :
  (students_in_classes.sum) * packs_per_student = 300 := by
  sorry

end john_bought_packs_l64_64705


namespace final_value_A_eq_B_pow_N_l64_64360

-- Definitions of conditions
def compute_A (A B : ℕ) (N : ℕ) : ℕ :=
    if N ≤ 0 then 
        1 
    else 
        let rec compute_loop (A' B' N' : ℕ) : ℕ :=
            if N' = 0 then A' 
            else 
                let B'' := B' * B'
                let N'' := N' / 2
                let A'' := if N' % 2 = 1 then A' * B' else A'
                compute_loop A'' B'' N'' 
        compute_loop A B N

-- Theorem statement
theorem final_value_A_eq_B_pow_N (A B N : ℕ) : compute_A A B N = B ^ N :=
    sorry

end final_value_A_eq_B_pow_N_l64_64360


namespace problem_statement_l64_64709

variables {R : Type*} [LinearOrderedField R]

-- Definitions of f and its derivatives
variable (f : R → R)
variable (f' : R → R) 
variable (f'' : R → R)

-- Conditions given in the math problem
axiom decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2
axiom derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x

-- Lean 4 statement for the proof problem
theorem problem_statement (decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2)
    (derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x) :
    ∀ x : R, f x > 0 :=
by
  sorry

end problem_statement_l64_64709


namespace short_trees_after_planting_l64_64524

-- Defining the conditions as Lean definitions
def current_short_trees : Nat := 3
def newly_planted_short_trees : Nat := 9

-- Defining the question (assertion to prove) with the expected answer
theorem short_trees_after_planting : current_short_trees + newly_planted_short_trees = 12 := by
  sorry

end short_trees_after_planting_l64_64524


namespace overall_percentage_change_in_membership_l64_64813

theorem overall_percentage_change_in_membership :
  let M := 1
  let fall_inc := 1.08
  let winter_inc := 1.15
  let spring_dec := 0.81
  (M * fall_inc * winter_inc * spring_dec - M) / M * 100 = 24.2 := by
  sorry

end overall_percentage_change_in_membership_l64_64813


namespace at_least_one_two_prob_l64_64681

-- Definitions and conditions corresponding to the problem
def total_outcomes (n : ℕ) : ℕ := n * n
def no_twos_outcomes (n : ℕ) : ℕ := (n - 1) * (n - 1)

-- The probability calculation
def probability_at_least_one_two (n : ℕ) : ℚ := 
  let tot_outs := total_outcomes n
  let no_twos := no_twos_outcomes n
  (tot_outs - no_twos : ℚ) / tot_outs

-- Our main theorem to be proved
theorem at_least_one_two_prob : 
  probability_at_least_one_two 6 = 11 / 36 := 
by
  sorry

end at_least_one_two_prob_l64_64681


namespace fraction_of_green_marbles_half_l64_64178

-- Definitions based on given conditions
def initial_fraction (x : ℕ) : ℚ := 1 / 3

-- Number of blue, red, and green marbles initially
def blue_marbles (x : ℕ) : ℚ := initial_fraction x * x
def red_marbles (x : ℕ) : ℚ := initial_fraction x * x
def green_marbles (x : ℕ) : ℚ := initial_fraction x * x

-- Number of green marbles after doubling
def doubled_green_marbles (x : ℕ) : ℚ := 2 * green_marbles x

-- New total number of marbles
def new_total_marbles (x : ℕ) : ℚ := blue_marbles x + red_marbles x + doubled_green_marbles x

-- New fraction of green marbles after doubling
def new_fraction_of_green_marbles (x : ℕ) : ℚ := doubled_green_marbles x / new_total_marbles x

theorem fraction_of_green_marbles_half (x : ℕ) (hx : x > 0) :
  new_fraction_of_green_marbles x = 1 / 2 :=
by
  sorry

end fraction_of_green_marbles_half_l64_64178


namespace max_value_of_M_l64_64071

noncomputable def M (x y z : ℝ) := min (min x y) z

theorem max_value_of_M
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_zero : b^2 - 4 * a * c ≥ 0) :
  M ((b + c) / a) ((c + a) / b) ((a + b) / c) ≤ 5 / 4 :=
sorry

end max_value_of_M_l64_64071


namespace overall_average_score_l64_64151

theorem overall_average_score
  (mean_morning mean_evening : ℕ)
  (ratio_morning_evening : ℚ) 
  (h1 : mean_morning = 90)
  (h2 : mean_evening = 80)
  (h3 : ratio_morning_evening = 4 / 5) : 
  ∃ overall_mean : ℚ, overall_mean = 84 :=
by
  sorry

end overall_average_score_l64_64151


namespace proof_f_of_2_add_g_of_3_l64_64715

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x^2 + 2 * x - 1

theorem proof_f_of_2_add_g_of_3 : f (2 + g 3) = 44 :=
by
  sorry

end proof_f_of_2_add_g_of_3_l64_64715


namespace cannot_be_external_diagonals_l64_64613

theorem cannot_be_external_diagonals (a b c : ℕ) : 
  ¬(3^2 + 4^2 = 6^2) :=
by
  sorry

end cannot_be_external_diagonals_l64_64613


namespace speed_conversion_l64_64431

theorem speed_conversion (speed_kmph : ℕ) (conversion_rate : ℚ) : (speed_kmph = 600) ∧ (conversion_rate = 0.6) → (speed_kmph * conversion_rate / 60 = 6) :=
by
  sorry

end speed_conversion_l64_64431


namespace fraction_of_blueberry_tart_l64_64135

/-- Let total leftover tarts be 0.91.
    Let the tart filled with cherries be 0.08.
    Let the tart filled with peaches be 0.08.
    Prove that the fraction of the tart filled with blueberries is 0.75. --/
theorem fraction_of_blueberry_tart (H_total : Real) (H_cherry : Real) (H_peach : Real)
  (H1 : H_total = 0.91) (H2 : H_cherry = 0.08) (H3 : H_peach = 0.08) :
  (H_total - (H_cherry + H_peach)) = 0.75 :=
sorry

end fraction_of_blueberry_tart_l64_64135


namespace find_money_of_Kent_l64_64848

variable (Alison Brittany Brooke Kent : ℝ)

def money_relations (h1 : Alison = 4000)
    (h2 : Alison = Brittany / 2)
    (h3 : Brittany = 4 * Brooke)
    (h4 : Brooke = 2 * Kent) : Prop :=
  Kent = 1000

theorem find_money_of_Kent
  {Alison Brittany Brooke Kent : ℝ}
  (h1 : Alison = 4000)
  (h2 : Alison = Brittany / 2)
  (h3 : Brittany = 4 * Brooke)
  (h4 : Brooke = 2 * Kent) :
  money_relations Alison Brittany Brooke Kent h1 h2 h3 h4 :=
by 
  sorry

end find_money_of_Kent_l64_64848


namespace quadratic_graphs_intersect_at_one_point_l64_64745

theorem quadratic_graphs_intersect_at_one_point
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3)
  (h_intersect_fg : ∃ x₀ : ℝ, (a1 - a2) * x₀^2 + (b1 - b2) * x₀ + (c1 - c2) = 0 ∧ (b1 - b2)^2 - 4 * (a1 - a2) * (c1 - c2) = 0)
  (h_intersect_gh : ∃ x₁ : ℝ, (a2 - a3) * x₁^2 + (b2 - b3) * x₁ + (c2 - c3) = 0 ∧ (b2 - b3)^2 - 4 * (a2 - a3) * (c2 - c3) = 0)
  (h_intersect_fh : ∃ x₂ : ℝ, (a1 - a3) * x₂^2 + (b1 - b3) * x₂ + (c1 - c3) = 0 ∧ (b1 - b3)^2 - 4 * (a1 - a3) * (c1 - c3) = 0) :
  ∃ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0) ∧ (a2 * x^2 + b2 * x + c2 = 0) ∧ (a3 * x^2 + b3 * x + c3 = 0) :=
by
  sorry

end quadratic_graphs_intersect_at_one_point_l64_64745


namespace smallest_value_of_n_l64_64625

theorem smallest_value_of_n (a b c m n : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 2010) (h4 : (a! * b! * c!) = m * 10 ^ n) : ∃ n, n = 500 := 
sorry

end smallest_value_of_n_l64_64625


namespace nancy_indian_food_freq_l64_64948

-- Definitions based on the problem
def antacids_per_indian_day := 3
def antacids_per_mexican_day := 2
def antacids_per_other_day := 1
def mexican_per_week := 2
def total_antacids_per_month := 60
def weeks_per_month := 4
def days_per_week := 7

-- The proof statement
theorem nancy_indian_food_freq :
  ∃ (I : ℕ), (total_antacids_per_month = 
    weeks_per_month * (antacids_per_indian_day * I + 
    antacids_per_mexican_day * mexican_per_week + 
    antacids_per_other_day * (days_per_week - I - mexican_per_week))) ∧ I = 3 :=
by
  sorry

end nancy_indian_food_freq_l64_64948


namespace new_average_page_count_l64_64720

theorem new_average_page_count
  (n : ℕ) (a : ℕ) (p1 p2 : ℕ)
  (h_n : n = 80) (h_a : a = 120)
  (h_p1 : p1 = 150) (h_p2 : p2 = 170) :
  (n - 2) ≠ 0 → 
  ((n * a - (p1 + p2)) / (n - 2) = 119) := 
by sorry

end new_average_page_count_l64_64720


namespace calculation_proof_l64_64352

theorem calculation_proof : (96 / 6) * 3 / 2 = 24 := by
  sorry

end calculation_proof_l64_64352


namespace other_root_of_quadratic_l64_64059

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, x^2 + a * x - 2 = 0 → x = -1) → ∃ m, x = m ∧ m = 2 :=
by
  sorry

end other_root_of_quadratic_l64_64059


namespace sum_absolute_values_of_first_ten_terms_l64_64857

noncomputable def S (n : ℕ) : ℤ := n^2 - 4 * n + 2

noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

noncomputable def absolute_sum_10 : ℤ :=
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|

theorem sum_absolute_values_of_first_ten_terms : absolute_sum_10 = 68 := by
  sorry

end sum_absolute_values_of_first_ten_terms_l64_64857


namespace area_of_fifteen_sided_figure_l64_64534

def point : Type := ℕ × ℕ

def vertices : List point :=
  [(1,1), (1,3), (3,5), (4,5), (5,4), (5,3), (6,3), (6,2), (5,1), (4,1), (3,2), (2,2), (1,1)]

def graph_paper_area (vs : List point) : ℚ :=
  -- Placeholder for actual area calculation logic
  -- The area for the provided vertices is found to be 11 cm^2.
  11

theorem area_of_fifteen_sided_figure : graph_paper_area vertices = 11 :=
by
  -- The actual proof would involve detailed steps to show that the area is indeed 11 cm^2
  -- Placeholder proof
  sorry

end area_of_fifteen_sided_figure_l64_64534


namespace prove_expression_value_l64_64432

theorem prove_expression_value (x y : ℝ) (h1 : 4 * x + y = 18) (h2 : x + 4 * y = 20) :
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
sorry

end prove_expression_value_l64_64432


namespace measure_diagonal_of_brick_l64_64656

def RectangularParallelepiped (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def DiagonalMeasurementPossible (a b c : ℝ) : Prop :=
  ∃ d : ℝ, d = (a^2 + b^2 + c^2)^(1/2)

theorem measure_diagonal_of_brick (a b c : ℝ) 
  (h : RectangularParallelepiped a b c) : DiagonalMeasurementPossible a b c :=
by
  sorry

end measure_diagonal_of_brick_l64_64656


namespace natural_number_pairs_l64_64167

theorem natural_number_pairs (a b : ℕ) (p q : ℕ) :
  a ≠ b →
  (∃ p, a + b = 2^p) →
  (∃ q, ab + 1 = 2^q) →
  (a = 1 ∧ b = 2^p - 1 ∨ a = 2^q - 1 ∧ b = 2^q + 1) :=
by intro hne hp hq; sorry

end natural_number_pairs_l64_64167


namespace oysters_eaten_l64_64975

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end oysters_eaten_l64_64975


namespace distance_triangle_four_points_l64_64566

variable {X : Type*} [MetricSpace X]

theorem distance_triangle_four_points (A B C D : X) :
  dist A D ≤ dist A B + dist B C + dist C D :=
by
  sorry

end distance_triangle_four_points_l64_64566


namespace minimum_possible_value_of_Box_l64_64020

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end minimum_possible_value_of_Box_l64_64020


namespace range_of_a_l64_64678

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a_l64_64678


namespace calculate_value_l64_64911

theorem calculate_value (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 :=
by
  sorry

end calculate_value_l64_64911


namespace simon_paid_amount_l64_64674

theorem simon_paid_amount:
  let pansy_price := 2.50
  let hydrangea_price := 12.50
  let petunia_price := 1.00
  let pansies_count := 5
  let hydrangeas_count := 1
  let petunias_count := 5
  let discount_rate := 0.10
  let change_received := 23.00

  let total_cost_before_discount := (pansies_count * pansy_price) + (hydrangeas_count * hydrangea_price) + (petunias_count * petunia_price)
  let discount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  let amount_paid_with := total_cost_after_discount + change_received

  amount_paid_with = 50.00 :=
by
  sorry

end simon_paid_amount_l64_64674


namespace incorrect_statement_l64_64735

-- Definitions based on the given conditions
def tripling_triangle_altitude_triples_area (b h : ℝ) : Prop :=
  3 * (1/2 * b * h) = 1/2 * b * (3 * h)

def halving_rectangle_base_halves_area (b h : ℝ) : Prop :=
  1/2 * b * h = 1/2 * (b * h)

def tripling_circle_radius_triples_area (r : ℝ) : Prop :=
  3 * (Real.pi * r^2) = Real.pi * (3 * r)^2

def tripling_divisor_and_numerator_leaves_quotient_unchanged (a b : ℝ) (hb : b ≠ 0) : Prop :=
  a / b = 3 * a / (3 * b)

def halving_negative_quantity_makes_it_greater (x : ℝ) : Prop :=
  x < 0 → (x / 2) > x

-- The incorrect statement is that tripling the radius of a circle triples the area
theorem incorrect_statement : ∃ r : ℝ, tripling_circle_radius_triples_area r → False :=
by
  use 1
  simp [tripling_circle_radius_triples_area]
  sorry

end incorrect_statement_l64_64735


namespace zero_in_M_l64_64189

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
by
  sorry

end zero_in_M_l64_64189


namespace exists_infinitely_many_n_l64_64963

def digit_sum (m : ℕ) : ℕ := sorry  -- Define the digit sum function

theorem exists_infinitely_many_n (S : ℕ → ℕ)
  (hS : ∀ m : ℕ, S m = digit_sum m) :
  ∃ᶠ n in at_top, S (3^n) ≥ S (3^(n + 1)) := 
sorry

end exists_infinitely_many_n_l64_64963


namespace first_term_geometric_sequence_b_n_bounded_l64_64496

-- Definition: S_n = 3a_n - 5n for any n in ℕ*
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 3 * a n - 5 * n

-- The sequence a_n is given such that
-- Proving the first term a_1
theorem first_term (a : ℕ → ℝ) (h : ∀ n, S (n + 1) a = S n a + a n + 1 - 5) : 
  a 1 = 5 / 2 :=
sorry

-- Prove that {a_n + 5} is a geometric sequence with common ratio 3/2
theorem geometric_sequence (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 5 * n) : 
  ∃ r, (∀ n, a (n + 1) + 5 = r * (a n + 5)) ∧ r = 3 / 2 :=
sorry

-- Prove that there exists m such that b_n < m always holds for b_n = (9n + 4) / (a_n + 5)
theorem b_n_bounded (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, b n = (9 * ↑n + 4) / (a n + 5)) 
  (h2 : ∀ n, a n = (15 / 2) * (3 / 2)^(n-1) - 5) :
  ∃ m, ∀ n, b n < m ∧ m = 88 / 45 :=
sorry

end first_term_geometric_sequence_b_n_bounded_l64_64496


namespace eval_floor_abs_neg_45_7_l64_64796

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l64_64796


namespace equation_is_linear_in_one_variable_l64_64344

theorem equation_is_linear_in_one_variable (n : ℤ) :
  (∀ x : ℝ, (n - 2) * x ^ |n - 1| + 5 = 0 → False) → n = 0 := by
  sorry

end equation_is_linear_in_one_variable_l64_64344


namespace longest_chord_of_circle_l64_64760

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end longest_chord_of_circle_l64_64760


namespace find_pos_integers_A_B_l64_64468

noncomputable def concat (A B : ℕ) : ℕ :=
  let b := Nat.log 10 B + 1
  A * 10 ^ b + B

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def satisfiesConditions (A B : ℕ) : Prop :=
  isPerfectSquare (concat A B) ∧ concat A B = 2 * A * B

theorem find_pos_integers_A_B :
  ∃ (A B : ℕ), A = (5 ^ b + 1) / 2 ∧ B = 2 ^ b * A * 100 ^ m ∧ b % 2 = 1 ∧ ∀ m : ℕ, satisfiesConditions A B :=
sorry

end find_pos_integers_A_B_l64_64468


namespace fraction_of_students_between_11_and_13_is_two_fifths_l64_64852

def totalStudents : ℕ := 45
def under11 : ℕ :=  totalStudents / 3
def over13 : ℕ := 12
def between11and13 : ℕ := totalStudents - (under11 + over13)
def fractionBetween11and13 : ℚ := between11and13 / totalStudents

theorem fraction_of_students_between_11_and_13_is_two_fifths :
  fractionBetween11and13 = 2 / 5 := 
by 
  sorry

end fraction_of_students_between_11_and_13_is_two_fifths_l64_64852


namespace find_y_intercept_l64_64510

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end find_y_intercept_l64_64510


namespace weight_difference_l64_64499

def weight_chemistry : ℝ := 7.12
def weight_geometry : ℝ := 0.62

theorem weight_difference : weight_chemistry - weight_geometry = 6.50 :=
by
  sorry

end weight_difference_l64_64499


namespace g_of_f_of_3_is_1852_l64_64336

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 - x + 2

theorem g_of_f_of_3_is_1852 : g (f 3) = 1852 := by
  sorry

end g_of_f_of_3_is_1852_l64_64336


namespace line_perpendicular_to_plane_implies_parallel_l64_64406

-- Definitions for lines and planes in space
axiom Line : Type
axiom Plane : Type

-- Relation of perpendicularity between a line and a plane
axiom perp : Line → Plane → Prop

-- Relation of parallelism between two lines
axiom parallel : Line → Line → Prop

-- The theorem to be proved
theorem line_perpendicular_to_plane_implies_parallel (x y : Line) (z : Plane) :
  perp x z → perp y z → parallel x y :=
by sorry

end line_perpendicular_to_plane_implies_parallel_l64_64406


namespace arcsin_neg_half_eq_neg_pi_six_l64_64065

theorem arcsin_neg_half_eq_neg_pi_six : 
  Real.arcsin (-1 / 2) = -Real.pi / 6 := 
sorry

end arcsin_neg_half_eq_neg_pi_six_l64_64065


namespace midpoint_product_l64_64776

theorem midpoint_product (x' y' : ℤ) 
  (h1 : (0 + x') / 2 = 2) 
  (h2 : (9 + y') / 2 = 4) : 
  (x' * y') = -4 :=
by
  sorry

end midpoint_product_l64_64776


namespace largest_prime_factor_of_1729_is_19_l64_64439

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) := is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_1729_is_19 : largest_prime_factor 1729 19 :=
by
  sorry

end largest_prime_factor_of_1729_is_19_l64_64439


namespace fixed_point_of_function_l64_64175

theorem fixed_point_of_function (a : ℝ) : 
  (a - 1) * 2^1 - 2 * a = -2 := by
  sorry

end fixed_point_of_function_l64_64175


namespace garden_area_l64_64220

theorem garden_area (w l : ℕ) (h1 : l = 3 * w + 30) (h2 : 2 * (w + l) = 780) : 
  w * l = 27000 := 
by 
  sorry

end garden_area_l64_64220


namespace find_angle_A_l64_64412

-- Variables representing angles A and B
variables (A B : ℝ)

-- The conditions of the problem translated into Lean
def angle_relationship := A = 2 * B - 15
def angle_supplementary := A + B = 180

-- The theorem statement we need to prove
theorem find_angle_A (h1 : angle_relationship A B) (h2 : angle_supplementary A B) : A = 115 :=
by { sorry }

end find_angle_A_l64_64412


namespace men_in_first_group_l64_64277

theorem men_in_first_group (M : ℕ) (h1 : (M * 7 * 18) = (12 * 7 * 12)) : M = 8 :=
by sorry

end men_in_first_group_l64_64277


namespace solution_set_of_inequality_l64_64180

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_mono : ∀ {x1 x2}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) (h_f1 : f 1 = 0) :
  {x | (x - 1) * f x > 0} = {x | -1 < x ∧ x < 1} ∪ {x | 1 < x} :=
by
  sorry

end solution_set_of_inequality_l64_64180


namespace system1_solution_system2_solution_l64_64522

-- Problem 1
theorem system1_solution (x y : ℝ) (h1 : 3 * x - 2 * y = 6) (h2 : 2 * x + 3 * y = 17) : 
  x = 4 ∧ y = 3 :=
by {
  sorry
}

-- Problem 2
theorem system2_solution (x y : ℝ) (h1 : x + 4 * y = 14) 
  (h2 : (x - 3) / 4 - (y - 3) / 3 = 1 / 12) : 
  x = 3 ∧ y = 11 / 4 :=
by {
  sorry
}

end system1_solution_system2_solution_l64_64522


namespace repaved_inches_before_today_l64_64979

theorem repaved_inches_before_today :
  let A := 4000
  let B := 3500
  let C := 2500
  let repaved_A := 0.70 * A
  let repaved_B := 0.60 * B
  let repaved_C := 0.80 * C
  let total_repaved_before := repaved_A + repaved_B + repaved_C
  let repaved_today := 950
  let new_total_repaved := total_repaved_before + repaved_today
  new_total_repaved - repaved_today = 6900 :=
by
  sorry

end repaved_inches_before_today_l64_64979


namespace boxes_containing_neither_l64_64726

theorem boxes_containing_neither (total_boxes markers erasers both : ℕ) 
  (h_total : total_boxes = 15) (h_markers : markers = 8) (h_erasers : erasers = 5) (h_both : both = 4) :
  total_boxes - (markers + erasers - both) = 6 :=
by
  sorry

end boxes_containing_neither_l64_64726


namespace min_three_beverages_overlap_l64_64270

variable (a b c d : ℝ)
variable (ha : a = 0.9)
variable (hb : b = 0.8)
variable (hc : c = 0.7)

theorem min_three_beverages_overlap : d = 0.7 :=
by
  sorry

end min_three_beverages_overlap_l64_64270


namespace probability_prime_sum_is_correct_l64_64882

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def cube_rolls_prob_prime_sum : ℚ :=
  let possible_outcomes := 36
  let prime_sums_count := 15
  prime_sums_count / possible_outcomes

theorem probability_prime_sum_is_correct :
  cube_rolls_prob_prime_sum = 5 / 12 :=
by
  -- The problem statement verifies that we have to show the calculation is correct
  sorry

end probability_prime_sum_is_correct_l64_64882


namespace sufficient_not_necessary_ellipse_l64_64032

theorem sufficient_not_necessary_ellipse (m n : ℝ) (h : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m ≠ n) ∧
  ¬(∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m > n ∧ n > 0) :=
sorry

end sufficient_not_necessary_ellipse_l64_64032


namespace focus_of_parabola_l64_64629

theorem focus_of_parabola (x y : ℝ) (h : y = 2 * x^2) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / 8) :=
by
  sorry

end focus_of_parabola_l64_64629


namespace sin_double_angle_l64_64153

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l64_64153


namespace max_side_of_triangle_with_perimeter_30_l64_64317

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end max_side_of_triangle_with_perimeter_30_l64_64317


namespace parallelogram_perimeter_l64_64880

def perimeter_of_parallelogram (a b : ℝ) : ℝ :=
  2 * (a + b)

theorem parallelogram_perimeter
  (side1 side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_side2 : side2 = 12) :
  perimeter_of_parallelogram side1 side2 = 60 := 
by
  sorry

end parallelogram_perimeter_l64_64880


namespace avg_price_of_towels_l64_64196

def towlesScenario (t1 t2 t3 : ℕ) (price1 price2 price3 : ℕ) : ℕ :=
  (t1 * price1 + t2 * price2 + t3 * price3) / (t1 + t2 + t3)

theorem avg_price_of_towels :
  towlesScenario 3 5 2 100 150 500 = 205 := by
  sorry

end avg_price_of_towels_l64_64196


namespace distinct_roots_and_ratios_l64_64161

open Real

theorem distinct_roots_and_ratios (a b : ℝ) (h1 : a^2 - 3*a - 1 = 0) (h2 : b^2 - 3*b - 1 = 0) (h3 : a ≠ b) :
  b/a + a/b = -11 :=
sorry

end distinct_roots_and_ratios_l64_64161


namespace mary_baseball_cards_l64_64288

theorem mary_baseball_cards :
  let initial_cards := 18
  let torn_cards := 8
  let fred_gifted_cards := 26
  let bought_cards := 40
  let exchanged_cards := 10
  let lost_cards := 5
  
  let remaining_cards := initial_cards - torn_cards
  let after_gift := remaining_cards + fred_gifted_cards
  let after_buy := after_gift + bought_cards
  let after_exchange := after_buy - exchanged_cards + exchanged_cards
  let final_count := after_exchange - lost_cards
  
  final_count = 71 :=
by
  sorry

end mary_baseball_cards_l64_64288


namespace minimum_value_l64_64497

theorem minimum_value (n : ℝ) (h : n > 0) : n + 32 / n^2 ≥ 6 := 
sorry

end minimum_value_l64_64497


namespace binary_equals_octal_l64_64843

-- Define the binary number 1001101 in decimal
def binary_1001101_decimal : ℕ := 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the octal number 115 in decimal
def octal_115_decimal : ℕ := 1 * 8^2 + 1 * 8^1 + 5 * 8^0

-- Theorem statement
theorem binary_equals_octal :
  binary_1001101_decimal = octal_115_decimal :=
sorry

end binary_equals_octal_l64_64843


namespace cost_price_of_apple_is_18_l64_64967

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end cost_price_of_apple_is_18_l64_64967


namespace area_of_rectangle_l64_64036

theorem area_of_rectangle (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 90) : w * l = 379.6875 :=
by
  sorry

end area_of_rectangle_l64_64036


namespace find_principal_amount_l64_64669

-- Define the given conditions
def interest_rate1 : ℝ := 0.08
def interest_rate2 : ℝ := 0.10
def interest_rate3 : ℝ := 0.12
def period1 : ℝ := 4
def period2 : ℝ := 6
def period3 : ℝ := 5
def total_interest_paid : ℝ := 12160

-- Goal is to find the principal amount P
theorem find_principal_amount (P : ℝ) :
  total_interest_paid = P * (interest_rate1 * period1 + interest_rate2 * period2 + interest_rate3 * period3) →
  P = 8000 :=
by
  sorry

end find_principal_amount_l64_64669


namespace solution_to_equation_l64_64365

noncomputable def equation (x : ℝ) : ℝ := 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2

theorem solution_to_equation :
  equation 3.294 = 0 ∧ equation (-0.405) = 0 :=
by
  sorry

end solution_to_equation_l64_64365


namespace cheaper_to_buy_more_l64_64238

def cost (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 60 then 13 * n
  else if 61 ≤ n ∧ n ≤ 90 then 12 * n
  else if 91 ≤ n then 11 * n
  else 0

theorem cheaper_to_buy_more (n : ℕ) : 
  (∃ m, m < n ∧ cost (m + 1) < cost m) ↔ n = 9 := sorry

end cheaper_to_buy_more_l64_64238


namespace train_speed_and_length_l64_64936

theorem train_speed_and_length (V l : ℝ) 
  (h1 : 7 * V = l) 
  (h2 : 25 * V = 378 + l) : 
  V = 21 ∧ l = 147 :=
by
  sorry

end train_speed_and_length_l64_64936


namespace arithmetic_sum_calculation_l64_64889

theorem arithmetic_sum_calculation :
  3 * (71 + 75 + 79 + 83 + 87 + 91) = 1458 :=
by
  sorry

end arithmetic_sum_calculation_l64_64889


namespace largest_non_formable_amount_l64_64073

-- Definitions and conditions from the problem
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def cannot_be_formed (n a b : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ a * x + b * y

-- The statement to prove
theorem largest_non_formable_amount :
  is_coprime 8 15 ∧ cannot_be_formed 97 8 15 :=
by
  sorry

end largest_non_formable_amount_l64_64073


namespace highest_page_number_l64_64836

/-- Given conditions: Pat has 19 instances of the digit '7' and an unlimited supply of all 
other digits. Prove that the highest page number Pat can number without exceeding 19 instances 
of the digit '7' is 99. -/
theorem highest_page_number (num_of_sevens : ℕ) (highest_page : ℕ) 
  (h1 : num_of_sevens = 19) : highest_page = 99 :=
sorry

end highest_page_number_l64_64836


namespace change_is_41_l64_64179

-- Define the cost of shirts and sandals as given in the problem conditions
def cost_of_shirts : ℕ := 10 * 5
def cost_of_sandals : ℕ := 3 * 3
def total_cost : ℕ := cost_of_shirts + cost_of_sandals

-- Define the amount given
def amount_given : ℕ := 100

-- Calculate the change
def change := amount_given - total_cost

-- State the theorem
theorem change_is_41 : change = 41 := 
by 
  -- Filling this with justification steps would be the actual proof
  -- but it's not required, so we use 'sorry' to indicate the theorem
  sorry

end change_is_41_l64_64179


namespace box_volume_l64_64920

theorem box_volume
  (L W H : ℝ)
  (h1 : L * W = 120)
  (h2 : W * H = 72)
  (h3 : L * H = 60) :
  L * W * H = 720 :=
by
  -- The proof goes here
  sorry

end box_volume_l64_64920


namespace second_term_arithmetic_sequence_l64_64758

theorem second_term_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 15 * d = 8)
  (h2 : a + 16 * d = 10) : 
  a + d = -20 := 
by sorry

end second_term_arithmetic_sequence_l64_64758


namespace algebraic_expression_value_l64_64772

theorem algebraic_expression_value {m n : ℝ} 
  (h1 : n = m - 2022) 
  (h2 : m * n = -2022) : 
  (2022 / m) + ((m^2 - 2022 * m) / n) = 2022 := 
by sorry

end algebraic_expression_value_l64_64772


namespace fraction_of_6_l64_64309

theorem fraction_of_6 (x y : ℕ) (h : (x / y : ℚ) * 6 + 6 = 10) : (x / y : ℚ) = 2 / 3 :=
by
  sorry

end fraction_of_6_l64_64309


namespace ratio_of_areas_of_circles_l64_64830

theorem ratio_of_areas_of_circles 
  (R_A R_B : ℝ) 
  (h : (π / 2 * R_A) = (π / 3 * R_B)) : 
  (π * R_A ^ 2) / (π * R_B ^ 2) = (4 : ℚ) / 9 := 
sorry

end ratio_of_areas_of_circles_l64_64830


namespace value_of_A_l64_64427

theorem value_of_A (A : ℕ) : (A * 1000 + 567) % 100 < 50 → (A * 1000 + 567) / 10 * 10 = 2560 → A = 2 :=
by
  intro h1 h2
  sorry

end value_of_A_l64_64427


namespace factorial_expression_simplification_l64_64218

theorem factorial_expression_simplification : (3 * (Nat.factorial 5) + 15 * (Nat.factorial 4)) / (Nat.factorial 6) = 1 := by
  sorry

end factorial_expression_simplification_l64_64218


namespace roots_quadratic_l64_64949

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end roots_quadratic_l64_64949


namespace infinite_n_exist_l64_64583

def S (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem infinite_n_exist (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ᶠ n in at_top, S n ≡ n [MOD p] :=
sorry

end infinite_n_exist_l64_64583


namespace original_laborers_l64_64743

theorem original_laborers (x : ℕ) : (x * 8 = (x - 3) * 14) → x = 7 :=
by
  intro h
  sorry

end original_laborers_l64_64743


namespace max_value_l64_64764

-- Definition of the ellipse and the goal function
def ellipse (x y : ℝ) := 2 * x^2 + 3 * y^2 = 12

-- Definition of the function we want to maximize
def func (x y : ℝ) := x + 2 * y

-- The theorem to prove that the maximum value of x + 2y on the ellipse is √22
theorem max_value (x y : ℝ) (h : ellipse x y) : ∃ θ : ℝ, func x y ≤ Real.sqrt 22 :=
by
  sorry

end max_value_l64_64764


namespace cylinder_height_to_radius_ratio_l64_64421

theorem cylinder_height_to_radius_ratio (V r h : ℝ) (hV : V = π * r^2 * h) (hS : sorry) :
  h / r = 2 :=
sorry

end cylinder_height_to_radius_ratio_l64_64421


namespace f_nonnegative_when_a_ge_one_l64_64297

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

noncomputable def h (a : ℝ) : ℝ := Real.log a + 1 - (1 / a)

theorem f_nonnegative_when_a_ge_one (a : ℝ) (x : ℝ) (h_a : a ≥ 1) : f a x ≥ 0 := by
  sorry  -- Placeholder for the proof.

end f_nonnegative_when_a_ge_one_l64_64297


namespace books_sold_correct_l64_64096

-- Define the initial number of books, number of books added, and the final number of books.
def initial_books : ℕ := 41
def added_books : ℕ := 2
def final_books : ℕ := 10

-- Define the number of books sold.
def sold_books : ℕ := initial_books + added_books - final_books

-- The theorem we need to prove: the number of books sold is 33.
theorem books_sold_correct : sold_books = 33 := by
  sorry

end books_sold_correct_l64_64096


namespace radius_increase_l64_64909

theorem radius_increase (ΔC : ℝ) (ΔC_eq : ΔC = 0.628) : Δr = 0.1 :=
by
  sorry

end radius_increase_l64_64909


namespace pictures_per_day_calc_l64_64974

def years : ℕ := 3
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

def number_of_cards : ℕ := total_spent / cost_per_card
def total_images : ℕ := number_of_cards * images_per_card
def days_in_year : ℕ := 365
def total_days : ℕ := years * days_in_year

theorem pictures_per_day_calc : 
  (total_images / total_days) = 10 := 
by
  sorry

end pictures_per_day_calc_l64_64974


namespace max_value_of_k_l64_64043

theorem max_value_of_k:
  ∃ (k : ℕ), 
  (∀ (a b : ℕ → ℕ) (h : ∀ i, a i < b i) (no_share : ∀ i j, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)) (distinct_sums : ∀ i j, i ≠ j → a i + b i ≠ a j + b j) (sum_limit : ∀ i, a i + b i ≤ 3011), 
    k ≤ 3011 ∧ k = 1204) := sorry

end max_value_of_k_l64_64043


namespace anthony_path_shortest_l64_64619

noncomputable def shortest_distance (A B C D M : ℝ) : ℝ :=
  4 + 2 * Real.sqrt 3

theorem anthony_path_shortest {A B C D : ℝ} (M : ℝ) (side_length : ℝ) (h : side_length = 4) : 
  shortest_distance A B C D M = 4 + 2 * Real.sqrt 3 :=
by 
  sorry

end anthony_path_shortest_l64_64619


namespace x_must_be_even_l64_64900

theorem x_must_be_even (x : ℤ) (h : ∃ (n : ℤ), (2 * x / 3 - x / 6) = n) : ∃ (k : ℤ), x = 2 * k :=
by
  sorry

end x_must_be_even_l64_64900


namespace price_of_water_margin_comics_l64_64875

-- Define the conditions
variables (x : ℕ) (y : ℕ)

-- Condition 1: Price relationship
def price_relationship : Prop := y = x + 60

-- Condition 2: Total expenditure on Romance of the Three Kingdoms comic books
def total_expenditure_romance_three_kingdoms : Prop := 60 * (y / 60) = 3600

-- Condition 3: Total expenditure on Water Margin comic books
def total_expenditure_water_margin : Prop := 120 * (x / 120) = 4800

-- Condition 4: Number of sets relationship
def number_of_sets_relationship : Prop := y = (4800 / x) / 2

-- The main statement to prove
theorem price_of_water_margin_comics (x : ℕ) (h1: price_relationship x (x + 60))
  (h2: total_expenditure_romance_three_kingdoms x)
  (h3: total_expenditure_water_margin x)
  (h4: number_of_sets_relationship x (x + 60)) : x = 120 :=
sorry

end price_of_water_margin_comics_l64_64875


namespace smallest_possible_z_l64_64346

theorem smallest_possible_z (w x y z : ℕ) (k : ℕ) (h1 : w = x - 1) (h2 : y = x + 1) (h3 : z = x + 2)
  (h4 : w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z) (h5 : k = 2) (h6 : w^3 + x^3 + y^3 = k * z^3) : z = 6 :=
by
  sorry

end smallest_possible_z_l64_64346


namespace slope_product_constant_l64_64263

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 2 * y ↔ x ^ 2 = 2 * p * y)

theorem slope_product_constant :
  ∀ (x1 y1 x2 y2 k1 k2 : ℝ) (P A B : ℝ × ℝ),
  P = (2, 2) →
  A = (x1, y1) →
  B = (x2, y2) →
  (∀ k: ℝ, y1 = k * (x1 + 2) + 4 ∧ y2 = k * (x2 + 2) + 4) →
  k1 = (y1 - 2) / (x1 - 2) →
  k2 = (y2 - 2) / (x2 - 2) →
  (x1 + x2 = 2 * k) →
  (x1 * x2 = -4 * k - 8) →
  k1 * k2 = -1 := 
  sorry

end slope_product_constant_l64_64263


namespace wrapping_paper_fraction_used_l64_64029

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l64_64029


namespace min_value_ineq_l64_64584

noncomputable def minimum_value (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) : ℝ :=
  1 / a + 4 / b

theorem min_value_ineq (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) :
  minimum_value a b ha hb h ≥ 16 :=
sorry

end min_value_ineq_l64_64584


namespace range_of_k_for_obtuse_triangle_l64_64177

theorem range_of_k_for_obtuse_triangle (k : ℝ) (a b c : ℝ) (h₁ : a = k) (h₂ : b = k + 2) (h₃ : c = k + 4) : 
  2 < k ∧ k < 6 :=
by
  sorry

end range_of_k_for_obtuse_triangle_l64_64177


namespace rolling_cube_dot_path_l64_64308

theorem rolling_cube_dot_path (a b c : ℝ) (h_edge : a = 1) (h_dot_top : True):
  c = (1 + Real.sqrt 5) / 2 := by
  sorry

end rolling_cube_dot_path_l64_64308


namespace quadrilateral_impossible_l64_64828

theorem quadrilateral_impossible (a b c d : ℕ) (h1 : 2 * a ^ 2 - 18 * a + 36 = 0)
    (h2 : b ^ 2 - 20 * b + 75 = 0) (h3 : c ^ 2 - 20 * c + 75 = 0) (h4 : 2 * d ^ 2 - 18 * d + 36 = 0) :
    ¬(a + b > d ∧ a + c > d ∧ b + c > d ∧ a + d > c ∧ b + d > c ∧ c + d > b ∧
      a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by
  sorry

end quadrilateral_impossible_l64_64828


namespace right_angled_triangles_count_l64_64868

theorem right_angled_triangles_count :
    ∃ n : ℕ, n = 31 ∧ ∀ (a b : ℕ), (b < 2011) ∧ (a * a = (b + 1) * (b + 1) - b * b) → n = 31 :=
by
  sorry

end right_angled_triangles_count_l64_64868


namespace euler_disproven_conjecture_solution_l64_64489

theorem euler_disproven_conjecture_solution : 
  ∃ (n : ℕ), n^5 = 133^5 + 110^5 + 84^5 + 27^5 ∧ n = 144 :=
by
  use 144
  have h : 144^5 = 133^5 + 110^5 + 84^5 + 27^5 := sorry
  exact ⟨h, rfl⟩

end euler_disproven_conjecture_solution_l64_64489


namespace total_distance_traveled_l64_64174

-- Define the parameters and conditions
def hoursPerDay : ℕ := 2
def daysPerWeek : ℕ := 5
def daysPeriod1 : ℕ := 3
def daysPeriod2 : ℕ := 2
def speedPeriod1 : ℕ := 12 -- speed in km/h from Monday to Wednesday
def speedPeriod2 : ℕ := 9 -- speed in km/h from Thursday to Friday

-- This is the theorem we want to prove
theorem total_distance_traveled : (daysPeriod1 * hoursPerDay * speedPeriod1) + (daysPeriod2 * hoursPerDay * speedPeriod2) = 108 :=
by
  sorry

end total_distance_traveled_l64_64174


namespace quadrilateral_area_inequality_l64_64221

theorem quadrilateral_area_inequality 
  (T : ℝ) (a b c d e f : ℝ) (φ : ℝ) 
  (hT : T = (1/2) * e * f * Real.sin φ) 
  (hptolemy : e * f ≤ a * c + b * d) : 
  2 * T ≤ a * c + b * d := 
sorry

end quadrilateral_area_inequality_l64_64221


namespace cube_volume_doubled_l64_64922

theorem cube_volume_doubled (a : ℝ) (h : a > 0) : 
  ((2 * a)^3 - a^3) / a^3 = 7 :=
by
  sorry

end cube_volume_doubled_l64_64922


namespace initial_number_of_men_l64_64146

theorem initial_number_of_men (P : ℝ) (M : ℝ) (h1 : P = 15 * M * (P / (15 * M))) (h2 : P = 12.5 * (M + 200) * (P / (12.5 * (M + 200)))) : M = 1000 :=
by
  sorry

end initial_number_of_men_l64_64146


namespace find_a_n_find_b_n_find_T_n_l64_64960

-- definitions of sequences and common ratios
variable (a_n b_n : ℕ → ℕ)
variable (S_n T_n : ℕ → ℕ)
variable (q : ℝ)
variable (n : ℕ)

-- conditions
axiom a1 : a_n 1 = 1
axiom S3 : S_n 3 = 9
axiom b1 : b_n 1 = 1
axiom b3 : b_n 3 = 20
axiom q_pos : q > 0
axiom geo_seq : (∀ n, b_n n / a_n n = q ^ (n - 1))

-- goals to prove
theorem find_a_n : ∀ n, a_n n = 2 * n - 1 := 
by sorry

theorem find_b_n : ∀ n, b_n n = (2 * n - 1) * 2 ^ (n - 1) := 
by sorry

theorem find_T_n : ∀ n, T_n n = (2 * n - 3) * 2 ^ n + 3 :=
by sorry

end find_a_n_find_b_n_find_T_n_l64_64960


namespace short_side_is_7_l64_64971

variable (L S : ℕ)

-- Given conditions
def perimeter : ℕ := 38
def long_side : ℕ := 12

-- In Lean, prove that the short side is 7 given L and P
theorem short_side_is_7 (h1 : 2 * L + 2 * S = perimeter) (h2 : L = long_side) : S = 7 := by
  sorry

end short_side_is_7_l64_64971


namespace kaleb_initial_cherries_l64_64542

/-- Kaleb's initial number of cherries -/
def initial_cherries : ℕ := 67

/-- Cherries that Kaleb ate -/
def eaten_cherries : ℕ := 25

/-- Cherries left after eating -/
def left_cherries : ℕ := 42

/-- Prove that the initial number of cherries is 67 given the conditions. -/
theorem kaleb_initial_cherries :
  eaten_cherries + left_cherries = initial_cherries :=
by
  sorry

end kaleb_initial_cherries_l64_64542


namespace p_eval_at_neg_one_l64_64984

noncomputable def p (x : ℝ) : ℝ :=
  x^2 - 2*x + 9

theorem p_eval_at_neg_one : p (-1) = 12 := by
  sorry

end p_eval_at_neg_one_l64_64984


namespace solve_problem_l64_64369

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 2 x else 3^x

theorem solve_problem : f (f (1 / 2)) = 1 / 3 := by
  sorry

end solve_problem_l64_64369


namespace mean_is_not_51_l64_64375

def frequencies : List Nat := [5, 8, 7, 13, 7]
def pH_values : List Float := [4.8, 4.9, 5.0, 5.2, 5.3]

def total_observations : Nat := List.sum frequencies

def mean (freqs : List Nat) (values : List Float) : Float :=
  let weighted_sum := List.sum (List.zipWith (· * ·) values (List.map (Float.ofNat) freqs))
  weighted_sum / (Float.ofNat total_observations)

theorem mean_is_not_51 : mean frequencies pH_values ≠ 5.1 := by
  -- Proof skipped
  sorry

end mean_is_not_51_l64_64375


namespace C_plus_D_l64_64825

theorem C_plus_D (D C : ℚ) (h : ∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (D * x - 17) / ((x - 3) * (x - 5)) = C / (x - 3) + 2 / (x - 5)) :
  C + D = 32 / 5 :=
by
  sorry

end C_plus_D_l64_64825


namespace number_of_dogs_in_shelter_l64_64752

variables (D C R P : ℕ)

-- Conditions
axiom h1 : 15 * C = 7 * D
axiom h2 : 9 * P = 5 * R
axiom h3 : 15 * (C + 8) = 11 * D
axiom h4 : 7 * P = 5 * (R + 6)

theorem number_of_dogs_in_shelter : D = 30 :=
by sorry

end number_of_dogs_in_shelter_l64_64752


namespace find_z_l64_64084

open Complex

theorem find_z (z : ℂ) (h : ((1 - I) ^ 2) / z = 1 + I) : z = -1 - I :=
sorry

end find_z_l64_64084


namespace ball_bounce_height_l64_64423

theorem ball_bounce_height (initial_height : ℝ) (r : ℝ) (k : ℕ) : 
  initial_height = 1000 → r = 1/2 → (r ^ k * initial_height < 1) → k = 10 := by
sorry

end ball_bounce_height_l64_64423


namespace salmon_at_rest_oxygen_units_l64_64019

noncomputable def salmonSwimSpeed (x : ℝ) : ℝ := (1/2) * Real.log (x / 100 * Real.pi) / Real.log 3

theorem salmon_at_rest_oxygen_units :
  ∃ x : ℝ, salmonSwimSpeed x = 0 ∧ x = 100 / Real.pi :=
by
  sorry

end salmon_at_rest_oxygen_units_l64_64019


namespace numberOfPairsPaddlesSold_l64_64557

def totalSalesPaddles : ℝ := 735
def avgPricePerPairPaddles : ℝ := 9.8

theorem numberOfPairsPaddlesSold :
  totalSalesPaddles / avgPricePerPairPaddles = 75 := 
by
  sorry

end numberOfPairsPaddlesSold_l64_64557


namespace train_distance_difference_l64_64532

theorem train_distance_difference 
  (speed1 speed2 : ℕ) (distance : ℕ) (meet_time : ℕ)
  (h_speed1 : speed1 = 16)
  (h_speed2 : speed2 = 21)
  (h_distance : distance = 444)
  (h_meet_time : meet_time = distance / (speed1 + speed2)) :
  (speed2 * meet_time) - (speed1 * meet_time) = 60 :=
by
  sorry

end train_distance_difference_l64_64532


namespace min_ab_correct_l64_64964

noncomputable def min_ab (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) : ℝ :=
  (6 - 2 * Real.sqrt 3) / 3

theorem min_ab_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) :
  a + b ≥ min_ab a b c h1 h2 :=
sorry

end min_ab_correct_l64_64964


namespace exponential_monotonicity_l64_64789

theorem exponential_monotonicity {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b :=
by 
  sorry 

end exponential_monotonicity_l64_64789


namespace other_team_scored_l64_64507

open Nat

def points_liz_scored (free_throws three_pointers jump_shots : Nat) : Nat :=
  free_throws * 1 + three_pointers * 3 + jump_shots * 2

def points_deficit := 20
def points_liz_deficit := points_liz_scored 5 3 4 - points_deficit
def final_loss_margin := 8
def other_team_score := points_liz_scored 5 3 4 + final_loss_margin

theorem other_team_scored
  (points_liz : Nat := points_liz_scored 5 3 4)
  (final_deficit : Nat := points_deficit)
  (final_margin : Nat := final_loss_margin)
  (other_team_points : Nat := other_team_score) :
  other_team_points = 30 := 
sorry

end other_team_scored_l64_64507


namespace probability_of_black_yellow_green_probability_of_not_red_or_green_l64_64962

namespace ProbabilityProof

/- Definitions of events A, B, C, D representing probabilities as real numbers -/
variables (P_A P_B P_C P_D : ℝ)

/- Conditions stated in the problem -/
def conditions (h1 : P_A = 1 / 3)
               (h2 : P_B + P_C = 5 / 12)
               (h3 : P_C + P_D = 5 / 12)
               (h4 : P_A + P_B + P_C + P_D = 1) :=
  true

/- Proof that P(B) = 1/4, P(C) = 1/6, and P(D) = 1/4 given the conditions -/
theorem probability_of_black_yellow_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1) :
  P_B = 1 / 4 ∧ P_C = 1 / 6 ∧ P_D = 1 / 4 :=
by
  sorry

/- Proof that the probability of not drawing a red or green ball is 5/12 -/
theorem probability_of_not_red_or_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1)
  (h5 : P_B = 1 / 4)
  (h6 : P_C = 1 / 6)
  (h7 : P_D = 1 / 4) :
  1 - (P_A + P_D) = 5 / 12 :=
by
  sorry

end ProbabilityProof

end probability_of_black_yellow_green_probability_of_not_red_or_green_l64_64962


namespace positive_difference_two_numbers_l64_64056

theorem positive_difference_two_numbers (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 2 * y - 3 * x = 5) : abs (y - x) = 8 := 
sorry

end positive_difference_two_numbers_l64_64056


namespace axis_of_symmetry_parabola_l64_64384

theorem axis_of_symmetry_parabola (x y : ℝ) :
  x^2 + 2*x*y + y^2 + 3*x + y = 0 → x + y + 1 = 0 :=
by {
  sorry
}

end axis_of_symmetry_parabola_l64_64384


namespace wendy_points_earned_l64_64388

-- Define the conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def bags_not_recycled : ℕ := 2

-- Define the statement to be proved
theorem wendy_points_earned : (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by
  sorry

end wendy_points_earned_l64_64388


namespace toms_dog_is_12_l64_64207

def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := toms_rabbit_age * 3

theorem toms_dog_is_12 : toms_dog_age = 12 :=
by
  sorry

end toms_dog_is_12_l64_64207


namespace john_makes_200_profit_l64_64700

noncomputable def john_profit (num_woodburnings : ℕ) (price_per_woodburning : ℕ) (cost_of_wood : ℕ) : ℕ :=
  (num_woodburnings * price_per_woodburning) - cost_of_wood

theorem john_makes_200_profit :
  john_profit 20 15 100 = 200 :=
by
  sorry

end john_makes_200_profit_l64_64700


namespace solve_for_buttons_l64_64037

def number_of_buttons_on_second_shirt (x : ℕ) : Prop :=
  200 * 3 + 200 * x = 1600

theorem solve_for_buttons : ∃ x : ℕ, number_of_buttons_on_second_shirt x ∧ x = 5 := by
  sorry

end solve_for_buttons_l64_64037


namespace nina_money_l64_64195

theorem nina_money :
  ∃ (m C : ℝ), 
    m = 6 * C ∧ 
    m = 8 * (C - 1) ∧ 
    m = 24 :=
by
  sorry

end nina_money_l64_64195


namespace simplify_expression_correct_l64_64622

noncomputable def simplify_expression : ℝ :=
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt (48))))

theorem simplify_expression_correct : simplify_expression = (Real.sqrt 6) + (Real.sqrt 2) :=
  sorry

end simplify_expression_correct_l64_64622


namespace kishore_miscellaneous_expenses_l64_64873

theorem kishore_miscellaneous_expenses :
  ∀ (rent milk groceries education petrol savings total_salary total_specified_expenses : ℝ),
  rent = 5000 →
  milk = 1500 →
  groceries = 4500 →
  education = 2500 →
  petrol = 2000 →
  savings = 2300 →
  (savings / 0.10) = total_salary →
  (rent + milk + groceries + education + petrol) = total_specified_expenses →
  (total_salary - (total_specified_expenses + savings)) = 5200 :=
by
  intros rent milk groceries education petrol savings total_salary total_specified_expenses
  sorry

end kishore_miscellaneous_expenses_l64_64873


namespace symmetric_points_sum_l64_64253

theorem symmetric_points_sum {c e : ℤ} 
  (P : ℤ × ℤ × ℤ) 
  (sym_xoy : ℤ × ℤ × ℤ) 
  (sym_y : ℤ × ℤ × ℤ) 
  (hP : P = (-4, -2, 3)) 
  (h_sym_xoy : sym_xoy = (-4, -2, -3)) 
  (h_sym_y : sym_y = (4, -2, 3)) 
  (hc : c = -3) 
  (he : e = 4) : 
  c + e = 1 :=
by
  -- Proof goes here
  sorry

end symmetric_points_sum_l64_64253


namespace isosceles_triangle_sides_l64_64741

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end isosceles_triangle_sides_l64_64741


namespace certain_number_divisible_l64_64147

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end certain_number_divisible_l64_64147


namespace part_I_part_II_l64_64450

-- Definition of the sequence a_n with given conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else (n^2 + n) / 2

-- Define the sum of the first n terms S_n
def S_n (n : ℕ) : ℕ :=
  (n + 2) / 3 * a_n n

-- Define the sequence b_n in terms of a_n
def b_n (n : ℕ) : ℚ := 1 / a_n n

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℚ :=
  2 * (1 - 1 / (n + 1))

-- Theorem statement for part (I)
theorem part_I (n : ℕ) : 
  a_n 2 = 3 ∧ a_n 3 = 6 ∧ (∀ (n : ℕ), n ≥ 2 → a_n n = (n^2 + n) / 2) := sorry

-- Theorem statement for part (II)
theorem part_II (n : ℕ) : 
  T_n n = 2 * (1 - 1 / (n + 1)) := sorry

end part_I_part_II_l64_64450


namespace speed_of_point_C_l64_64783

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l64_64783


namespace general_formula_sum_and_min_value_l64_64329

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def a1 := (a 1 = -5)
def a_condition := (3 * a 3 + a 5 = 0)

-- Prove the general formula for an arithmetic sequence
theorem general_formula (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0) : 
  ∀ n, a n = 2 * n - 7 := 
by
  sorry

-- Using the general formula to find the sum Sn and its minimum value
theorem sum_and_min_value (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0)
  (h : ∀ n, a n = 2 * n - 7) : 
  ∀ n, S n = n^2 - 6 * n ∧ ∃ n, S n = -9 :=
by
  sorry

end general_formula_sum_and_min_value_l64_64329


namespace expression_value_l64_64513

theorem expression_value :
  (35 + 12) ^ 2 - (12 ^ 2 + 35 ^ 2 - 2 * 12 * 35) = 1680 :=
by
  sorry

end expression_value_l64_64513


namespace square_of_leg_l64_64125

theorem square_of_leg (a c b : ℝ) (h1 : c = 2 * a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = 3 * a^2 + 4 * a + 1 :=
by
  sorry

end square_of_leg_l64_64125


namespace find_unknown_blankets_rate_l64_64605

noncomputable def unknown_blankets_rate : ℝ :=
  let total_cost_3_blankets := 3 * 100
  let discount := 0.10 * total_cost_3_blankets
  let cost_3_blankets_after_discount := total_cost_3_blankets - discount
  let cost_1_blanket := 150
  let tax := 0.15 * cost_1_blanket
  let cost_1_blanket_after_tax := cost_1_blanket + tax
  let total_avg_price_per_blanket := 150
  let total_blankets := 6
  let total_cost := total_avg_price_per_blanket * total_blankets
  (total_cost - cost_3_blankets_after_discount - cost_1_blanket_after_tax) / 2

theorem find_unknown_blankets_rate : unknown_blankets_rate = 228.75 :=
  by
    sorry

end find_unknown_blankets_rate_l64_64605


namespace probability_single_trial_l64_64458

theorem probability_single_trial (p : ℚ) (h₁ : (1 - p)^4 = 16 / 81) : p = 1 / 3 :=
sorry

end probability_single_trial_l64_64458


namespace period_f_axis_of_symmetry_f_max_value_f_l64_64788

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 5)

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem axis_of_symmetry_f (k : ℤ) :
  ∀ x, 2 * x - Real.pi / 5 = Real.pi / 4 + k * Real.pi → x = 9 * Real.pi / 40 + k * Real.pi / 2 := sorry

theorem max_value_f :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 ∧ x = 7 * Real.pi / 20 := sorry

end period_f_axis_of_symmetry_f_max_value_f_l64_64788


namespace monotone_decreasing_intervals_l64_64586

theorem monotone_decreasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, deriv f x = (x - 2) * (x^2 - 1)) :
  ((∀ x : ℝ, x < -1 → deriv f x < 0) ∧ (∀ x : ℝ, 1 < x → x < 2 → deriv f x < 0)) :=
by
  sorry

end monotone_decreasing_intervals_l64_64586


namespace task1_task2_l64_64479

-- Define the conditions and the probabilities to be proven

def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1

def total_combinations := Nat.choose total_pens 2

def combinations_with_exactly_one_first_class : Nat :=
  (first_class_pens * (total_pens - first_class_pens))

def probability_one_first_class_pen : ℚ :=
  combinations_with_exactly_one_first_class / total_combinations

def combinations_without_any_third_class : Nat :=
  Nat.choose (first_class_pens + second_class_pens) 2

def probability_no_third_class_pen : ℚ :=
  combinations_without_any_third_class / total_combinations

theorem task1 : probability_one_first_class_pen = 3 / 5 := 
  sorry

theorem task2 : probability_no_third_class_pen = 2 / 3 := 
  sorry

end task1_task2_l64_64479


namespace triangle_area_upper_bound_l64_64894

variable {α : Type u}
variable [LinearOrderedField α]
variable {A B C : α} -- Points A, B, C as elements of some field.

-- Definitions for the lengths of the sides, interpreted as scalar distances.
variable (AB AC : α)

-- Assume that AB and AC are lengths of sides of the triangle
-- Assume the area of the triangle is non-negative and does not exceed the specified bound.
theorem triangle_area_upper_bound (S : α) (habc : S = (1 / 2) * AB * AC) :
  S ≤ (1 / 2) * AB * AC := 
sorry

end triangle_area_upper_bound_l64_64894


namespace water_usage_l64_64978

theorem water_usage (payment : ℝ) (usage : ℝ) : 
  payment = 7.2 → (usage ≤ 6 → payment = usage * 0.8) → (usage > 6 → payment = 4.8 + (usage - 6) * 1.2) → usage = 8 :=
by
  sorry

end water_usage_l64_64978


namespace trapezoid_area_l64_64914

theorem trapezoid_area (u l h : ℕ) (hu : u = 12) (hl : l = u + 4) (hh : h = 10) : 
  (1 / 2 : ℚ) * (u + l) * h = 140 := by
  sorry

end trapezoid_area_l64_64914


namespace sum_due_in_years_l64_64094

theorem sum_due_in_years 
  (D : ℕ)
  (S : ℕ)
  (r : ℚ)
  (H₁ : D = 168)
  (H₂ : S = 768)
  (H₃ : r = 14 / 100) :
  ∃ t : ℕ, t = 2 := 
by
  sorry

end sum_due_in_years_l64_64094


namespace number_of_perfect_square_factors_l64_64892

theorem number_of_perfect_square_factors (a b c d : ℕ) :
  (∀ a b c d, 
    (0 ≤ a ∧ a ≤ 4) ∧ 
    (0 ≤ b ∧ b ≤ 2) ∧ 
    (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ 
    (a % 2 = 0) ∧ 
    (b % 2 = 0) ∧ 
    (c = 0) ∧ 
    (d = 0)
  → 3 * 2 * 1 * 1 = 6) := by
  sorry

end number_of_perfect_square_factors_l64_64892


namespace pradeep_failure_marks_l64_64950

theorem pradeep_failure_marks :
  let total_marks := 925
  let pradeep_score := 160
  let passing_percentage := 20
  let passing_marks := (passing_percentage / 100) * total_marks
  let failed_by := passing_marks - pradeep_score
  failed_by = 25 :=
by
  sorry

end pradeep_failure_marks_l64_64950


namespace geometric_series_sum_l64_64169

theorem geometric_series_sum :
  let a := 2 / 3
  let r := 1 / 3
  a / (1 - r) = 1 :=
by
  sorry

end geometric_series_sum_l64_64169


namespace rational_functional_equation_l64_64503

theorem rational_functional_equation (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (f = λ x => x) ∨ (f = λ x => -x) :=
by
  sorry

end rational_functional_equation_l64_64503


namespace no_solution_bills_l64_64946

theorem no_solution_bills (x y z : ℕ) (h1 : x + y + z = 10) (h2 : x + 3 * y + 5 * z = 25) : false :=
by
  sorry

end no_solution_bills_l64_64946


namespace least_subtr_from_12702_to_div_by_99_l64_64048

theorem least_subtr_from_12702_to_div_by_99 : ∃ k : ℕ, 12702 - k = 99 * (12702 / 99) ∧ 0 ≤ k ∧ k < 99 :=
by
  sorry

end least_subtr_from_12702_to_div_by_99_l64_64048


namespace absolute_value_inequality_l64_64476

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by
  sorry

end absolute_value_inequality_l64_64476


namespace find_some_number_l64_64814

def op (x w : ℕ) := (2^x) / (2^w)

theorem find_some_number (n : ℕ) (hn : 0 < n) : (op (op 4 n) n) = 4 → n = 2 :=
by
  sorry

end find_some_number_l64_64814


namespace profit_per_meter_l64_64835

theorem profit_per_meter (number_of_meters : ℕ) (total_selling_price cost_price_per_meter : ℝ) 
  (h1 : number_of_meters = 85) 
  (h2 : total_selling_price = 8925) 
  (h3 : cost_price_per_meter = 90) :
  (total_selling_price - cost_price_per_meter * number_of_meters) / number_of_meters = 15 :=
  sorry

end profit_per_meter_l64_64835


namespace neg_of_exists_l64_64808

theorem neg_of_exists (P : ℝ → Prop) : 
  (¬ ∃ x: ℝ, x ≥ 3 ∧ x^2 - 2 * x + 3 < 0) ↔ (∀ x: ℝ, x ≥ 3 → x^2 - 2 * x + 3 ≥ 0) :=
by
  sorry

end neg_of_exists_l64_64808


namespace find_f_of_five_thirds_l64_64941

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_f_of_five_thirds (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_fun : ∀ x : ℝ, f (1 + x) = f (-x))
  (h_val : f (-1 / 3) = 1 / 3) : 
  f (5 / 3) = 1 / 3 :=
  sorry

end find_f_of_five_thirds_l64_64941


namespace sufficient_not_necessary_l64_64358

-- Define set A and set B
def setA (x : ℝ) := x > 5
def setB (x : ℝ) := x > 3

-- Statement:
theorem sufficient_not_necessary (x : ℝ) : setA x → setB x :=
by
  intro h
  exact sorry

end sufficient_not_necessary_l64_64358


namespace cylinder_volume_ratio_l64_64673

theorem cylinder_volume_ratio
  (S1 S2 : ℝ) (v1 v2 : ℝ)
  (lateral_area_equal : 2 * Real.pi * S1.sqrt = 2 * Real.pi * S2.sqrt)
  (base_area_ratio : S1 / S2 = 16 / 9) :
  v1 / v2 = 4 / 3 :=
by
  sorry

end cylinder_volume_ratio_l64_64673


namespace sum_of_11378_and_121_is_odd_l64_64985

theorem sum_of_11378_and_121_is_odd (h1 : Even 11378) (h2 : Odd 121) : Odd (11378 + 121) :=
by
  sorry

end sum_of_11378_and_121_is_odd_l64_64985


namespace base_h_equation_l64_64227

theorem base_h_equation (h : ℕ) : 
  (5 * h^3 + 7 * h^2 + 3 * h + 4) + (6 * h^3 + 4 * h^2 + 2 * h + 1) = 
  1 * h^4 + 4 * h^3 + 1 * h^2 + 5 * h + 5 → 
  h = 10 := 
sorry

end base_h_equation_l64_64227


namespace price_reduction_l64_64685

theorem price_reduction (x : ℝ) 
  (initial_price : ℝ := 60) 
  (final_price : ℝ := 48.6) :
  initial_price * (1 - x) * (1 - x) = final_price :=
by
  sorry

end price_reduction_l64_64685


namespace solve_for_N_l64_64846

theorem solve_for_N :
    (481 + 483 + 485 + 487 + 489 + 491 = 3000 - N) → (N = 84) :=
by
    -- Proof is omitted
    sorry

end solve_for_N_l64_64846


namespace possible_values_of_a_l64_64608

noncomputable def f (x a : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 2 * a * x + 2 else x + 9 / x - 3 * a

theorem possible_values_of_a (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ 1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end possible_values_of_a_l64_64608


namespace factorization_result_l64_64442

theorem factorization_result :
  ∃ (c d : ℕ), (c > d) ∧ ((x^2 - 20 * x + 91) = (x - c) * (x - d)) ∧ (2 * d - c = 1) :=
by
  -- Using the conditions and proving the given equation
  sorry

end factorization_result_l64_64442


namespace width_of_room_l64_64703

-- Definitions from conditions
def length : ℝ := 8
def total_cost : ℝ := 34200
def cost_per_sqm : ℝ := 900

-- Theorem stating the width of the room
theorem width_of_room : (total_cost / cost_per_sqm) / length = 4.75 := by 
  sorry

end width_of_room_l64_64703


namespace negation_of_existence_statement_l64_64520

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 3 * x + 2 = 0) = ∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existence_statement_l64_64520


namespace problem1_union_problem2_intersection_problem3_subset_l64_64749

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 4 ≤ 0}

theorem problem1_union (m : ℝ) (hm : m = 2) : A ∪ B m = {x | -1 ≤ x ∧ x ≤ 4} :=
sorry

theorem problem2_intersection (m : ℝ) (h : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3}) : m = 3 :=
sorry

theorem problem3_subset (m : ℝ) (h : A ⊆ {x | ¬ (x ∈ B m)}) : m > 5 ∨ m < -3 :=
sorry

end problem1_union_problem2_intersection_problem3_subset_l64_64749


namespace p_necessary_not_sufficient_for_q_l64_64121

open Classical

variable (p q : Prop)

theorem p_necessary_not_sufficient_for_q (h1 : ¬(p → q)) (h2 : ¬q → ¬p) : (¬(p → q) ∧ (¬q → ¬p) ∧ (¬p → ¬q ∧ ¬(¬q → p))) := by
  sorry

end p_necessary_not_sufficient_for_q_l64_64121


namespace ordered_pairs_divide_square_sum_l64_64561

theorem ordered_pairs_divide_square_sum :
  { (m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ (mn - 1) ∣ (m^2 + n^2) } = { (1, 2), (1, 3), (2, 1), (3, 1) } := 
sorry

end ordered_pairs_divide_square_sum_l64_64561


namespace swimming_pool_cost_l64_64794

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l64_64794


namespace smallest_integer_is_17_l64_64217

theorem smallest_integer_is_17
  (a b c d : ℕ)
  (h1 : b = 33)
  (h2 : d = b + 3)
  (h3 : (a + b + c + d) = 120)
  (h4 : a ≤ b)
  (h5 : c > b)
  : a = 17 :=
sorry

end smallest_integer_is_17_l64_64217


namespace double_root_equation_correct_statements_l64_64451

theorem double_root_equation_correct_statements
  (a b c : ℝ) (r₁ r₂ : ℝ)
  (h1 : a ≠ 0)
  (h2 : r₁ = 2 * r₂)
  (h3 : r₁ ≠ r₂)
  (h4 : a * r₁ ^ 2 + b * r₁ + c = 0)
  (h5 : a * r₂ ^ 2 + b * r₂ + c = 0) :
  (∀ (m n : ℝ), (∀ (r : ℝ), r = 2 → (x - r) * (m * x + n) = 0 → 4 * m ^ 2 + 5 * m * n + n ^ 2 = 0)) ∧
  (∀ (p q : ℝ), p * q = 2 → ∃ x, p * x ^ 2 + 3 * x + q = 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ = -1 / p ∧ x₂ = -q ∧ x₁ = 2 * x₂)) ∧
  (2 * b ^ 2 = 9 * a * c) :=
by
  sorry

end double_root_equation_correct_statements_l64_64451


namespace mountaineering_team_problem_l64_64332

structure Climber :=
  (total_students : ℕ)
  (advanced_climbers : ℕ)
  (intermediate_climbers : ℕ)
  (beginners : ℕ)

structure Experience :=
  (advanced_points : ℕ)
  (intermediate_points : ℕ)
  (beginner_points : ℕ)

structure TeamComposition :=
  (advanced_needed : ℕ)
  (intermediate_needed : ℕ)
  (beginners_needed : ℕ)
  (max_experience : ℕ)

def team_count (students : Climber) (xp : Experience) (comp : TeamComposition) : ℕ :=
  let total_experience := comp.advanced_needed * xp.advanced_points +
                          comp.intermediate_needed * xp.intermediate_points +
                          comp.beginners_needed * xp.beginner_points
  let max_teams_from_advanced := students.advanced_climbers / comp.advanced_needed
  let max_teams_from_intermediate := students.intermediate_climbers / comp.intermediate_needed
  let max_teams_from_beginners := students.beginners / comp.beginners_needed
  if total_experience ≤ comp.max_experience then
    min (max_teams_from_advanced) $ min (max_teams_from_intermediate) (max_teams_from_beginners)
  else 0

def problem : Prop :=
  team_count
    ⟨172, 45, 70, 57⟩
    ⟨80, 50, 30⟩
    ⟨5, 8, 5, 1000⟩ = 8

-- Let's declare the theorem now:
theorem mountaineering_team_problem : problem := sorry

end mountaineering_team_problem_l64_64332


namespace hyperbola_imaginary_axis_twice_real_axis_l64_64952

theorem hyperbola_imaginary_axis_twice_real_axis (m : ℝ) : 
  (exists (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0), mx^2 + b^2 * y^2 = b^2) ∧
  (b = 2 * a) ∧ (m < 0) → 
  m = -1 / 4 := 
sorry

end hyperbola_imaginary_axis_twice_real_axis_l64_64952


namespace sheela_overall_total_income_l64_64795

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end sheela_overall_total_income_l64_64795


namespace other_carton_racket_count_l64_64691

def num_total_cartons : Nat := 38
def num_total_rackets : Nat := 100
def num_specific_cartons : Nat := 24
def num_rackets_per_specific_carton : Nat := 3

def num_remaining_cartons := num_total_cartons - num_specific_cartons
def num_remaining_rackets := num_total_rackets - (num_specific_cartons * num_rackets_per_specific_carton)

theorem other_carton_racket_count :
  (num_remaining_rackets / num_remaining_cartons) = 2 :=
by
  sorry

end other_carton_racket_count_l64_64691


namespace perpendicular_lines_parallel_l64_64102

noncomputable def line := Type
noncomputable def plane := Type

variables (m n : line) (α : plane)

def parallel (l1 l2 : line) : Prop := sorry -- Definition of parallel lines
def perpendicular (l : line) (α : plane) : Prop := sorry -- Definition of perpendicular line to a plane

theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end perpendicular_lines_parallel_l64_64102


namespace find_sum_of_a_and_b_l64_64142

theorem find_sum_of_a_and_b (a b : ℝ) (h1 : 0.005 * a = 0.65) (h2 : 0.0125 * b = 1.04) : a + b = 213.2 :=
  sorry

end find_sum_of_a_and_b_l64_64142


namespace bella_bracelets_l64_64729

theorem bella_bracelets (h_beads_per_bracelet : Nat)
  (h_initial_beads : Nat) 
  (h_additional_beads : Nat) 
  (h_friends : Nat):
  h_beads_per_bracelet = 8 →
  h_initial_beads = 36 →
  h_additional_beads = 12 →
  h_friends = (h_initial_beads + h_additional_beads) / h_beads_per_bracelet →
  h_friends = 6 :=
by
  intros h_beads_per_bracelet_eq h_initial_beads_eq h_additional_beads_eq h_friends_eq
  subst_vars
  sorry

end bella_bracelets_l64_64729


namespace curve_properties_l64_64933

noncomputable def curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

theorem curve_properties :
  curve 1 0 ∧ curve 0 1 ∧ curve (1/4) (1/4) ∧ 
  (∀ p : ℝ × ℝ, curve p.1 p.2 → curve p.2 p.1) :=
by
  sorry

end curve_properties_l64_64933


namespace find_months_contributed_l64_64187

theorem find_months_contributed (x : ℕ) (profit_A profit_total : ℝ)
  (contrib_A : ℝ) (contrib_B : ℝ) (months_B : ℕ) :
  profit_A / profit_total = (contrib_A * x) / (contrib_A * x + contrib_B * months_B) →
  profit_A = 4800 →
  profit_total = 8400 →
  contrib_A = 5000 →
  contrib_B = 6000 →
  months_B = 5 →
  x = 8 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end find_months_contributed_l64_64187


namespace regression_estimate_l64_64045

theorem regression_estimate:
  ∀ (x : ℝ), (1.43 * x + 257 = 400) → x = 100 :=
by
  intro x
  intro h
  sorry

end regression_estimate_l64_64045


namespace circle_symmetry_l64_64294

def initial_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 1 = 0

def standard_form_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

def symmetric_circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

theorem circle_symmetry :
  (∀ x y : ℝ, initial_circle_eq x y ↔ standard_form_eq x y) →
  (∀ x y : ℝ, standard_form_eq x y → symmetric_circle_eq (-x) (-y)) →
  ∀ x y : ℝ, initial_circle_eq x y → symmetric_circle_eq x y :=
by
  intros h1 h2 x y hxy
  sorry

end circle_symmetry_l64_64294


namespace root_division_7_pow_l64_64730

theorem root_division_7_pow : 
  ( (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = 7 ^ (3 / 28) ) :=
sorry

end root_division_7_pow_l64_64730


namespace gcd_is_3_l64_64862

noncomputable def a : ℕ := 130^2 + 240^2 + 350^2
noncomputable def b : ℕ := 131^2 + 241^2 + 351^2

theorem gcd_is_3 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_is_3_l64_64862


namespace initial_quarters_l64_64563

variable (q : ℕ)

theorem initial_quarters (h : q + 3 = 11) : q = 8 :=
by
  sorry

end initial_quarters_l64_64563


namespace eval_polynomial_correct_l64_64486

theorem eval_polynomial_correct (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) (hy_pos : 0 < y) :
  y^3 - 3 * y^2 - 9 * y + 3 = 3 :=
sorry

end eval_polynomial_correct_l64_64486


namespace sum_of_200_terms_l64_64799

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a1 a200 : ℝ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

def collinearity_condition (a1 a200 : ℝ) : Prop :=
a1 + a200 = 1

-- Proof statement
theorem sum_of_200_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 a200 : ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_collinear : collinearity_condition a1 a200) : 
  S 200 = 100 := 
sorry

end sum_of_200_terms_l64_64799


namespace regular_tetrahedron_ratio_l64_64156

/-- In plane geometry, the ratio of the radius of the circumscribed circle to the 
inscribed circle of an equilateral triangle is 2:1, --/
def ratio_radii_equilateral_triangle : ℚ := 2 / 1

/-- In space geometry, we study the relationship between the radii of the circumscribed
sphere and the inscribed sphere of a regular tetrahedron. --/
def ratio_radii_regular_tetrahedron : ℚ := 3 / 1

/-- Prove the ratio of the radius of the circumscribed sphere to the inscribed sphere
of a regular tetrahedron is 3 : 1, given the ratio is 2 : 1 for the equilateral triangle. --/
theorem regular_tetrahedron_ratio : 
  ratio_radii_equilateral_triangle = 2 / 1 → 
  ratio_radii_regular_tetrahedron = 3 / 1 :=
by
  sorry

end regular_tetrahedron_ratio_l64_64156


namespace multiply_658217_99999_l64_64506

theorem multiply_658217_99999 : 658217 * 99999 = 65821034183 := 
by
  sorry

end multiply_658217_99999_l64_64506


namespace twice_midpoint_l64_64954

open Complex

def z1 : ℂ := -7 + 5 * I
def z2 : ℂ := 9 - 11 * I

theorem twice_midpoint : 2 * ((z1 + z2) / 2) = 2 - 6 * I := 
by
  -- Sorry is used to skip the proof
  sorry

end twice_midpoint_l64_64954


namespace perpendicular_planes_normal_vector_l64_64014

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem perpendicular_planes_normal_vector {m : ℝ} 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (h₁ : a = (1, 2, -2)) 
  (h₂ : b = (-2, 1, m)) 
  (h₃ : dot_product a b = 0) : 
  m = 0 := 
sorry

end perpendicular_planes_normal_vector_l64_64014


namespace flavors_remaining_to_try_l64_64464

def total_flavors : ℕ := 100
def flavors_tried_two_years_ago (total_flavors : ℕ) : ℕ := total_flavors / 4
def flavors_tried_last_year (flavors_tried_two_years_ago : ℕ) : ℕ := 2 * flavors_tried_two_years_ago

theorem flavors_remaining_to_try
  (total_flavors : ℕ)
  (flavors_tried_two_years_ago : ℕ)
  (flavors_tried_last_year : ℕ) :
  flavors_tried_two_years_ago = total_flavors / 4 →
  flavors_tried_last_year = 2 * flavors_tried_two_years_ago →
  total_flavors - (flavors_tried_two_years_ago + flavors_tried_last_year) = 25 :=
by
  sorry

end flavors_remaining_to_try_l64_64464


namespace income_increase_is_17_percent_l64_64570

def sales_percent_increase (original_items : ℕ) 
                           (original_price : ℝ) 
                           (discount_percent : ℝ) 
                           (sales_increase_percent : ℝ) 
                           (new_items_sold : ℕ) 
                           (new_income : ℝ)
                           (percent_increase : ℝ) : Prop :=
  let original_income := original_items * original_price
  let discounted_price := original_price * (1 - discount_percent / 100)
  let increased_sales := original_items + (original_items * sales_increase_percent / 100)
  original_income = original_items * original_price ∧
  new_income = discounted_price * increased_sales ∧
  new_items_sold = original_items * (1 + sales_increase_percent / 100) ∧
  percent_increase = ((new_income - original_income) / original_income) * 100 ∧
  original_items = 100 ∧ original_price = 1 ∧ discount_percent = 10 ∧ sales_increase_percent = 30 ∧ 
  new_items_sold = 130 ∧ new_income = 117 ∧ percent_increase = 17

theorem income_increase_is_17_percent :
  sales_percent_increase 100 1 10 30 130 117 17 :=
sorry

end income_increase_is_17_percent_l64_64570


namespace solve_fractional_equation_l64_64015

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end solve_fractional_equation_l64_64015


namespace largest_number_is_B_l64_64194
open Real

noncomputable def A := 0.989
noncomputable def B := 0.998
noncomputable def C := 0.899
noncomputable def D := 0.9899
noncomputable def E := 0.8999

theorem largest_number_is_B :
  B = max (max (max (max A B) C) D) E :=
by
  sorry

end largest_number_is_B_l64_64194


namespace base_length_of_parallelogram_l64_64610

-- Definitions and conditions
def parallelogram_area (base altitude : ℝ) : ℝ := base * altitude
def altitude (base : ℝ) : ℝ := 2 * base

-- Main theorem to prove
theorem base_length_of_parallelogram (A : ℝ) (base : ℝ)
  (hA : A = 200) 
  (h_altitude : altitude base = 2 * base) 
  (h_area : parallelogram_area base (altitude base) = A) : 
  base = 10 := 
sorry

end base_length_of_parallelogram_l64_64610


namespace divisor_is_20_l64_64737

theorem divisor_is_20 (D q1 q2 q3 : ℕ) :
  (242 = D * q1 + 11) ∧
  (698 = D * q2 + 18) ∧
  (940 = D * q3 + 9) →
  D = 20 :=
by
  sorry

end divisor_is_20_l64_64737


namespace collinear_c1_c2_l64_64403

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 7, 0)
def b : ℝ × ℝ × ℝ := (1, -3, 4)

-- Define the vectors c1 and c2 based on a and b
def c1 : ℝ × ℝ × ℝ := (4 * 3, 4 * 7, 4 * 0) - (2 * 1, 2 * -3, 2 * 4)
def c2 : ℝ × ℝ × ℝ := (1, -3, 4) - (2 * 3, 2 * 7, 2 * 0)

-- The theorem to prove that c1 and c2 are collinear
theorem collinear_c1_c2 : c1 = (-2 : ℝ) • c2 := by sorry

end collinear_c1_c2_l64_64403


namespace E_plays_2_games_l64_64446

-- Definitions for the students and the number of games they played
def students := ["A", "B", "C", "D", "E"]
def games_played_by (S : String) : Nat :=
  if S = "A" then 4 else
  if S = "B" then 3 else
  if S = "C" then 2 else 
  if S = "D" then 1 else
  2  -- this is the number of games we need to prove for student E 

-- Theorem stating the number of games played by E
theorem E_plays_2_games : games_played_by "E" = 2 :=
  sorry

end E_plays_2_games_l64_64446


namespace power_mean_inequality_l64_64343

theorem power_mean_inequality
  (n : ℕ) (hn : 0 < n) (x1 x2 : ℝ) :
  (x1^n + x2^n)^(n+1) / (x1^(n-1) + x2^(n-1))^n ≤ (x1^(n+1) + x2^(n+1))^n / (x1^n + x2^n)^(n-1) :=
by
  sorry

end power_mean_inequality_l64_64343


namespace nth_monomial_correct_l64_64541

-- Definitions of the sequence of monomials

def coeff (n : ℕ) : ℕ := 3 * n + 2
def exponent (n : ℕ) : ℕ := n

def nth_monomial (n : ℕ) (a : ℕ) : ℕ := (coeff n) * (a ^ (exponent n))

-- Theorem statement
theorem nth_monomial_correct (n : ℕ) (a : ℕ) : nth_monomial n a = (3 * n + 2) * (a ^ n) :=
by
  sorry

end nth_monomial_correct_l64_64541


namespace cost_price_of_watch_l64_64097

theorem cost_price_of_watch
  (C : ℝ)
  (h1 : 0.9 * C + 225 = 1.05 * C) :
  C = 1500 :=
by sorry

end cost_price_of_watch_l64_64097


namespace balloon_minimum_volume_l64_64163

theorem balloon_minimum_volume 
  (p V : ℝ)
  (h1 : p * V = 24000)
  (h2 : p ≤ 40000) : 
  V ≥ 0.6 :=
  sorry

end balloon_minimum_volume_l64_64163


namespace remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l64_64993

theorem remainder_of_9_6_plus_8_7_plus_7_8_mod_7 : (9^6 + 8^7 + 7^8) % 7 = 2 := 
by sorry

end remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l64_64993


namespace jen_total_birds_l64_64171

-- Define the number of chickens and ducks
variables (C D : ℕ)

-- Define the conditions
def ducks_condition (C D : ℕ) : Prop := D = 4 * C + 10
def num_ducks (D : ℕ) : Prop := D = 150

-- Define the total number of birds
def total_birds (C D : ℕ) : ℕ := C + D

-- Prove that the total number of birds is 185 given the conditions
theorem jen_total_birds (C D : ℕ) (h1 : ducks_condition C D) (h2 : num_ducks D) : total_birds C D = 185 :=
by
  sorry

end jen_total_birds_l64_64171


namespace students_can_do_both_l64_64773

variable (total_students swimmers gymnasts neither : ℕ)

theorem students_can_do_both (h1 : total_students = 60)
                             (h2 : swimmers = 27)
                             (h3 : gymnasts = 28)
                             (h4 : neither = 15) : 
                             total_students - (total_students - swimmers + total_students - gymnasts - neither) = 10 := 
by 
  sorry

end students_can_do_both_l64_64773


namespace unique_n_divides_2_pow_n_minus_1_l64_64483

theorem unique_n_divides_2_pow_n_minus_1 (n : ℕ) (h : n ∣ 2^n - 1) : n = 1 :=
sorry

end unique_n_divides_2_pow_n_minus_1_l64_64483


namespace find_simple_interest_rate_l64_64807

theorem find_simple_interest_rate (P A T SI R : ℝ)
  (hP : P = 750)
  (hA : A = 1125)
  (hT : T = 5)
  (hSI : SI = A - P)
  (hSI_def : SI = (P * R * T) / 100) : R = 10 :=
by
  -- Proof would go here
  sorry

end find_simple_interest_rate_l64_64807


namespace remainder_of_polynomial_division_l64_64474

-- Definitions based on conditions in the problem
def polynomial (x : ℝ) : ℝ := 8 * x^4 - 22 * x^3 + 9 * x^2 + 10 * x - 45

def divisor (x : ℝ) : ℝ := 4 * x - 8

-- Proof statement as per the problem equivalence
theorem remainder_of_polynomial_division : polynomial 2 = -37 := by
  sorry

end remainder_of_polynomial_division_l64_64474


namespace find_n_l64_64803

theorem find_n (n : ℕ) (h : ∀ x : ℝ, (n : ℝ) < x ∧ x < (n + 1 : ℝ) → 3 * x - 5 = 0) :
  n = 1 :=
sorry

end find_n_l64_64803


namespace find_base_17_digit_l64_64938

theorem find_base_17_digit (a : ℕ) (h1 : 0 ≤ a ∧ a < 17) 
  (h2 : (25 + a) % 16 = 0) : a = 7 :=
sorry

end find_base_17_digit_l64_64938


namespace number_of_valid_numbers_l64_64663

def is_valid_number (N : ℕ) : Prop :=
  N ≥ 1000 ∧ N < 10000 ∧ ∃ a x : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ x < 1000 ∧ 
  N = 1000 * a + x ∧ x = N / 9

theorem number_of_valid_numbers : ∃ (n : ℕ), n = 7 ∧ ∀ N, is_valid_number N → N < 1000 * (n + 2) := 
sorry

end number_of_valid_numbers_l64_64663


namespace find_value_a2_b2_c2_l64_64438

variable (a b c p q r : ℝ)
variable (h1 : a * b = p)
variable (h2 : b * c = q)
variable (h3 : c * a = r)
variable (h4 : p ≠ 0)
variable (h5 : q ≠ 0)
variable (h6 : r ≠ 0)

theorem find_value_a2_b2_c2 : a^2 + b^2 + c^2 = 1 :=
by sorry

end find_value_a2_b2_c2_l64_64438


namespace cost_proof_l64_64652

-- Given conditions
def total_cost : Int := 190
def working_days : Int := 19
def trips_per_day : Int := 2
def total_trips : Int := working_days * trips_per_day

-- Define the problem to prove
def cost_per_trip : Int := 5

theorem cost_proof : (total_cost / total_trips = cost_per_trip) := 
by 
  -- This is a placeholder to indicate that we're skipping the proof
  sorry

end cost_proof_l64_64652


namespace ratio_of_red_to_black_l64_64816

theorem ratio_of_red_to_black (r b : ℕ) (h_r : r = 26) (h_b : b = 70) :
  r / Nat.gcd r b = 13 ∧ b / Nat.gcd r b = 35 :=
by
  sorry

end ratio_of_red_to_black_l64_64816


namespace balls_in_drawers_l64_64133

theorem balls_in_drawers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : (k ^ n) = 32 :=
by
  rw [h_n, h_k]
  sorry

end balls_in_drawers_l64_64133


namespace coats_leftover_l64_64556

theorem coats_leftover :
  ∀ (total_coats : ℝ) (num_boxes : ℝ),
  total_coats = 385.5 →
  num_boxes = 7.5 →
  ∃ extra_coats : ℕ, extra_coats = 3 :=
by
  intros total_coats num_boxes h1 h2
  sorry

end coats_leftover_l64_64556


namespace find_x_value_l64_64116

theorem find_x_value (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := 
by 
  sorry

end find_x_value_l64_64116


namespace initial_red_marbles_l64_64243

theorem initial_red_marbles (r g : ℕ) (h1 : r * 3 = 7 * g) (h2 : 4 * (r - 14) = g + 30) : r = 24 := 
sorry

end initial_red_marbles_l64_64243


namespace probability_exactly_one_red_ball_l64_64505

-- Define the given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 3
def children : ℕ := 10

-- Define the question and calculate the probability
theorem probability_exactly_one_red_ball : 
  (3 * (3 / 10) * ((7 / 10) * (7 / 10))) = 0.441 := 
by 
  sorry

end probability_exactly_one_red_ball_l64_64505


namespace muffins_division_l64_64132

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end muffins_division_l64_64132


namespace gray_region_area_l64_64301

theorem gray_region_area (d_small r_large r_small π : ℝ) (h1 : d_small = 6)
    (h2 : r_large = 3 * r_small) (h3 : r_small = d_small / 2) :
    (π * r_large ^ 2 - π * r_small ^ 2) = 72 * π := 
by
  -- The proof will be filled here
  sorry

end gray_region_area_l64_64301


namespace max_value_of_k_l64_64595

theorem max_value_of_k (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m)) ≥ k) ↔ k ≤ 8 := 
sorry

end max_value_of_k_l64_64595


namespace a_capital_used_l64_64199

theorem a_capital_used (C P x : ℕ) (h_b_contributes : 3 * C / 4 - C ≥ 0) 
(h_b_receives : 2 * P / 3 - P ≥ 0) 
(h_b_money_used : 10 > 0) 
(h_ratio : 1 / 2 = x / 30) 
: x = 15 :=
sorry

end a_capital_used_l64_64199


namespace chocolates_divisible_l64_64456

theorem chocolates_divisible (n m : ℕ) (h1 : n > 0) (h2 : m > 0) : 
  (n ≤ m) ∨ (m % (n - m) = 0) :=
sorry

end chocolates_divisible_l64_64456


namespace diagonals_of_seven_sided_polygon_l64_64904

-- Define the number of sides of the polygon
def n : ℕ := 7

-- Calculate the number of diagonals in a polygon with n sides
def numberOfDiagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- The statement to prove
theorem diagonals_of_seven_sided_polygon : numberOfDiagonals n = 14 := by
  -- Here we will write the proof steps, but they're not needed now.
  sorry

end diagonals_of_seven_sided_polygon_l64_64904


namespace buffy_breath_time_l64_64665

theorem buffy_breath_time (k : ℕ) (b : ℕ) (f : ℕ) 
  (h1 : k = 3 * 60) 
  (h2 : b = k - 20) 
  (h3 : f = b - 40) :
  f = 120 :=
by {
  sorry
}

end buffy_breath_time_l64_64665


namespace total_bill_l64_64763

theorem total_bill (total_people : ℕ) (children : ℕ) (adult_cost : ℕ) (child_cost : ℕ)
  (h : total_people = 201) (hc : children = 161) (ha : adult_cost = 8) (hc_cost : child_cost = 4) :
  (201 - 161) * 8 + 161 * 4 = 964 :=
by
  rw [←h, ←hc, ←ha, ←hc_cost]
  sorry

end total_bill_l64_64763


namespace lemonade_glasses_l64_64005

theorem lemonade_glasses (total_lemons : ℝ) (lemons_per_glass : ℝ) (glasses : ℝ) :
  total_lemons = 18.0 → lemons_per_glass = 2.0 → glasses = total_lemons / lemons_per_glass → glasses = 9 :=
by
  intro h_total_lemons h_lemons_per_glass h_glasses
  sorry

end lemonade_glasses_l64_64005


namespace grasshopper_jump_l64_64331

theorem grasshopper_jump (frog_jump grasshopper_jump : ℕ)
  (h1 : frog_jump = grasshopper_jump + 17)
  (h2 : frog_jump = 53) :
  grasshopper_jump = 36 :=
by
  sorry

end grasshopper_jump_l64_64331


namespace water_wasted_per_hour_l64_64113

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end water_wasted_per_hour_l64_64113


namespace intersection_chord_line_eq_l64_64636

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
noncomputable def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

theorem intersection_chord_line_eq (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : 
  2 * x + y = 0 :=
sorry

end intersection_chord_line_eq_l64_64636


namespace find_f_65_l64_64219

theorem find_f_65 (f : ℝ → ℝ) (h_eq : ∀ x y : ℝ, f (x * y) = x * f y) (h_f1 : f 1 = 40) : f 65 = 2600 :=
by
  sorry

end find_f_65_l64_64219


namespace circles_coincide_l64_64983

-- Definitions for circle being inscribed in an angle and touching each other
structure Circle :=
  (radius : ℝ)
  (center: ℝ × ℝ)

def inscribed_in_angle (c : Circle) (θ: ℝ) : Prop :=
  -- Placeholder definition for circle inscribed in an angle
  sorry

def touches (c₁ c₂ : Circle) : Prop :=
  -- Placeholder definition for circles touching each other
  sorry

-- The angles of the triangle ABC are A, B, and C.
-- We are given the following conditions:
variables (A B C : ℝ) -- angles
variables (S1 S2 S3 S4 S5 S6 S7: Circle) -- circles

-- Circle S1 is inscribed in angle A
axiom S1_condition : inscribed_in_angle S1 A

-- Circle S2 is inscribed in angle B and touches S1 externally
axiom S2_condition : inscribed_in_angle S2 B ∧ touches S2 S1

-- Circle S3 is inscribed in angle C and touches S2
axiom S3_condition : inscribed_in_angle S3 C ∧ touches S3 S2

-- Circle S4 is inscribed in angle A and touches S3
axiom S4_condition : inscribed_in_angle S4 A ∧ touches S4 S3

-- We repeat this pattern up to circle S7
axiom S5_condition : inscribed_in_angle S5 B ∧ touches S5 S4
axiom S6_condition : inscribed_in_angle S6 C ∧ touches S6 S5
axiom S7_condition : inscribed_in_angle S7 A ∧ touches S7 S6

-- We need to prove the circle S7 coincides with S1
theorem circles_coincide : S7 = S1 :=
by
  -- Proof is skipped using sorry
  sorry

end circles_coincide_l64_64983


namespace gcd_n_cube_plus_m_square_l64_64780

theorem gcd_n_cube_plus_m_square (n m : ℤ) (h : n > 2^3) : Int.gcd (n^3 + m^2) (n + 2) = 1 :=
by
  sorry

end gcd_n_cube_plus_m_square_l64_64780


namespace value_is_100_l64_64953

theorem value_is_100 (number : ℕ) (h : number = 20) : 5 * number = 100 :=
by
  sorry

end value_is_100_l64_64953


namespace total_paint_is_correct_l64_64248

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end total_paint_is_correct_l64_64248


namespace value_of_a_b_l64_64878

theorem value_of_a_b:
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ (a + 6 * 10^3 + 7 * 10^2 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end value_of_a_b_l64_64878


namespace solve_equation1_solve_equation2_l64_64849

theorem solve_equation1 (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by
  sorry

theorem solve_equation2 (x : ℝ) : 2 * (x - 3)^2 = x - 3 ↔ (x = 3/2 ∨ x = 7/2) :=
by
  sorry

end solve_equation1_solve_equation2_l64_64849


namespace onion_harvest_scientific_notation_l64_64383

theorem onion_harvest_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 325000000 = a * 10^n ∧ a = 3.25 ∧ n = 8 := 
by
  sorry

end onion_harvest_scientific_notation_l64_64383


namespace similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l64_64992

variable {a b c m_c a' b' c' m_c' : ℝ}

/- The first proof problem -/
theorem similar_right_triangles_hypotenuse_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c')) :
  a * a' + b * b' = c * c' := by
  sorry

/- The second proof problem -/
theorem similar_right_triangles_reciprocal_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c') ∧ (m_c = k * m_c')) :
  (1 / (a * a') + 1 / (b * b')) = 1 / (m_c * m_c') := by
  sorry

end similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l64_64992


namespace roy_cat_finishes_food_on_wednesday_l64_64240

-- Define the conditions
def morning_consumption := (1 : ℚ) / 5
def evening_consumption := (1 : ℚ) / 6
def total_cans := 10

-- Define the daily consumption calculation
def daily_consumption := morning_consumption + evening_consumption

-- Define the day calculation function
def day_cat_finishes_food : String :=
  let total_days := total_cans / daily_consumption
  if total_days ≤ 7 then "certain day within a week"
  else if total_days ≤ 14 then "Wednesday next week"
  else "later"

-- The main theorem to prove
theorem roy_cat_finishes_food_on_wednesday : day_cat_finishes_food = "Wednesday next week" := sorry

end roy_cat_finishes_food_on_wednesday_l64_64240


namespace order_of_abcd_l64_64414

-- Define the rational numbers a, b, c, d
variables {a b c d : ℚ}

-- State the conditions as assumptions
axiom h1 : a + b = c + d
axiom h2 : a + d < b + c
axiom h3 : c < d

-- The goal is to prove the correct order of a, b, c, d
theorem order_of_abcd (a b c d : ℚ) (h1 : a + b = c + d) (h2 : a + d < b + c) (h3 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end order_of_abcd_l64_64414


namespace average_of_4_8_N_l64_64092

-- Define the condition for N
variable (N : ℝ) (cond : 7 < N ∧ N < 15)

-- State the theorem to prove
theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) :
  (frac12 + N) / 3 = 7 ∨ (12 + N) / 3 = 9 :=
sorry

end average_of_4_8_N_l64_64092


namespace arithmetic_sequence_a6_l64_64661

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_root1 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 2 = x)
  (h_root2 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 10 = x) : 
  a 6 = -6 := 
by
  sorry

end arithmetic_sequence_a6_l64_64661


namespace target_runs_correct_l64_64295

noncomputable def target_runs (run_rate1 : ℝ) (ovs1 : ℕ) (run_rate2 : ℝ) (ovs2 : ℕ) : ℝ :=
  (run_rate1 * ovs1) + (run_rate2 * ovs2)

theorem target_runs_correct : target_runs 4.5 12 8.052631578947368 38 = 360 :=
by
  sorry

end target_runs_correct_l64_64295


namespace find_x_squared_plus_y_squared_l64_64558

variable (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : y + 7 = (x - 3)^2) (h2 : x + 7 = (y - 3)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 17 :=
by
  sorry  -- Proof to be provided

end find_x_squared_plus_y_squared_l64_64558


namespace evaluate_exp_power_l64_64713

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l64_64713


namespace find_f_of_2_l64_64209

-- Given definitions:
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def defined_on_neg_inf_to_0 (f : ℝ → ℝ) : Prop := ∀ x, x < 0 → f x = 2 * x^3 + x^2

-- The main theorem to prove:
theorem find_f_of_2 (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_def : defined_on_neg_inf_to_0 f) :
  f 2 = 12 :=
sorry

end find_f_of_2_l64_64209


namespace cindy_total_time_to_travel_one_mile_l64_64440

-- Definitions for the conditions
def run_speed : ℝ := 3 -- Cindy's running speed in miles per hour.
def walk_speed : ℝ := 1 -- Cindy's walking speed in miles per hour.
def run_distance : ℝ := 0.5 -- Distance run by Cindy in miles.
def walk_distance : ℝ := 0.5 -- Distance walked by Cindy in miles.

-- Theorem statement
theorem cindy_total_time_to_travel_one_mile : 
  ((run_distance / run_speed) + (walk_distance / walk_speed)) * 60 = 40 := 
by
  sorry

end cindy_total_time_to_travel_one_mile_l64_64440


namespace visitors_not_enjoyed_not_understood_l64_64089

theorem visitors_not_enjoyed_not_understood (V E U : ℕ) (hv_v : V = 520)
  (hu_e : E = U) (he : E = 3 * V / 4) : (V / 4) = 130 :=
by
  rw [hv_v] at he
  sorry

end visitors_not_enjoyed_not_understood_l64_64089


namespace evaluate_expression_l64_64540

theorem evaluate_expression (k : ℤ): 
  2^(-(3*k+1)) - 2^(-(3*k-2)) + 2^(-(3*k)) - 2^(-(3*k+3)) = -((21:ℚ)/(8:ℚ)) * 2^(-(3*k)) := 
by 
  sorry

end evaluate_expression_l64_64540


namespace find_k_l64_64885

-- Define the lines as given in the problem
def line1 (k : ℝ) (x y : ℝ) : Prop := k * x + (1 - k) * y - 3 = 0
def line2 (k : ℝ) (x y : ℝ) : Prop := (k - 1) * x + (2 * k + 3) * y - 2 = 0

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop :=
  let slope1 := -k / (1 - k)
  let slope2 := -(k - 1) / (2 * k + 3)
  slope1 * slope2 = -1

-- Problem statement: Prove that the lines are perpendicular implies k == 1 or k == -3
theorem find_k (k : ℝ) : perpendicular k → (k = 1 ∨ k = -3) :=
sorry

end find_k_l64_64885


namespace Tom_allowance_leftover_l64_64910

theorem Tom_allowance_leftover :
  let initial_allowance := 12
  let first_week_spending := (1/3) * initial_allowance
  let remaining_after_first_week := initial_allowance - first_week_spending
  let second_week_spending := (1/4) * remaining_after_first_week
  let final_amount := remaining_after_first_week - second_week_spending
  final_amount = 6 :=
by
  sorry

end Tom_allowance_leftover_l64_64910


namespace find_numbers_l64_64626

theorem find_numbers (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 24) : x = 21 ∧ y = -3 :=
by
  sorry

end find_numbers_l64_64626


namespace smallest_number_of_coins_l64_64261

theorem smallest_number_of_coins (d q : ℕ) (h₁ : 10 * d + 25 * q = 265) (h₂ : d > q) :
  d + q = 16 :=
sorry

end smallest_number_of_coins_l64_64261


namespace chocolate_bar_min_breaks_l64_64279

theorem chocolate_bar_min_breaks (n : ℕ) (h : n = 40) : ∃ k : ℕ, k = n - 1 := 
by 
  sorry

end chocolate_bar_min_breaks_l64_64279


namespace tetrahedron_edge_length_l64_64429

theorem tetrahedron_edge_length (a : ℝ) (V : ℝ) 
  (h₀ : V = 0.11785113019775793) 
  (h₁ : V = (Real.sqrt 2 / 12) * a^3) : a = 1 := by
  sorry

end tetrahedron_edge_length_l64_64429


namespace main_inequality_equality_condition_l64_64809

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem main_inequality 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) :=
  sorry

theorem equality_condition 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end main_inequality_equality_condition_l64_64809


namespace beetle_number_of_routes_128_l64_64970

noncomputable def beetle_routes (A B : Type) : Nat :=
  let choices_at_first_step := 4
  let choices_at_second_step := 4
  let choices_at_third_step := 4
  let choices_at_final_step := 2
  choices_at_first_step * choices_at_second_step * choices_at_third_step * choices_at_final_step

theorem beetle_number_of_routes_128 (A B : Type) :
  beetle_routes A B = 128 :=
  by sorry

end beetle_number_of_routes_128_l64_64970


namespace cells_after_9_days_l64_64555

noncomputable def remaining_cells (initial : ℕ) (days : ℕ) : ℕ :=
  let rec divide_and_decay (cells: ℕ) (remaining_days: ℕ) : ℕ :=
    if remaining_days = 0 then cells
    else
      let divided := cells * 2
      let decayed := (divided * 9) / 10
      divide_and_decay decayed (remaining_days - 3)
  divide_and_decay initial days

theorem cells_after_9_days :
  remaining_cells 5 9 = 28 := by
  sorry

end cells_after_9_days_l64_64555


namespace area_of_region_a_area_of_region_b_area_of_region_c_l64_64305

-- Definition of regions and their areas
def area_of_square : Real := sorry
def area_of_diamond : Real := sorry
def area_of_hexagon : Real := sorry

-- Define the conditions for the regions
def region_a (x y : ℝ) := abs x ≤ 1 ∧ abs y ≤ 1
def region_b (x y : ℝ) := abs x + abs y ≤ 10
def region_c (x y : ℝ) := abs x + abs y + abs (x + y) ≤ 2020

-- Prove that the areas match the calculated solutions
theorem area_of_region_a : area_of_square = 4 := 
by sorry

theorem area_of_region_b : area_of_diamond = 200 := 
by sorry

theorem area_of_region_c : area_of_hexagon = 3060300 := 
by sorry

end area_of_region_a_area_of_region_b_area_of_region_c_l64_64305


namespace promotional_codes_one_tenth_l64_64504

open Nat

def promotional_chars : List Char := ['C', 'A', 'T', '3', '1', '1', '9']

def count_promotional_codes (chars : List Char) (len : Nat) : Nat := sorry

theorem promotional_codes_one_tenth : count_promotional_codes promotional_chars 5 / 10 = 60 :=
by 
  sorry

end promotional_codes_one_tenth_l64_64504


namespace sequence_last_number_is_one_l64_64692

theorem sequence_last_number_is_one :
  ∃ (a : ℕ → ℤ), (a 1 = 1) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → a (n + 1) = a n + a (n + 2)) ∧ (a 1999 = 1) := sorry

end sequence_last_number_is_one_l64_64692


namespace condition_for_M_eq_N_l64_64682

theorem condition_for_M_eq_N (a1 b1 c1 a2 b2 c2 : ℝ) 
    (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0) 
    (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) → 
    (M = {x : ℝ | a1 * x ^ 2 + b1 * x + c1 > 0} ∧
     N = {x : ℝ | a2 * x ^ 2 + b2 * x + c2 > 0} →
    (¬ (M = N))) ∨ (¬ (N = {} ↔ (M = N))) :=
sorry

end condition_for_M_eq_N_l64_64682


namespace expression_value_l64_64604

theorem expression_value : 
  29^2 - 27^2 + 25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 389 :=
by
  sorry

end expression_value_l64_64604


namespace g_at_8_equals_minus_30_l64_64148

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_8_equals_minus_30 :
  (∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) →
  g 8 = -30 :=
by
  intro h
  sorry

end g_at_8_equals_minus_30_l64_64148


namespace fourth_term_eq_156_l64_64598

-- Definition of the sequence term
def seq_term (n : ℕ) : ℕ :=
  (List.range n).map (λ k => 5^k) |>.sum

-- Theorem to prove the fourth term equals 156
theorem fourth_term_eq_156 : seq_term 4 = 156 :=
sorry

end fourth_term_eq_156_l64_64598


namespace max_sum_of_xj4_minus_xj5_l64_64500

theorem max_sum_of_xj4_minus_xj5 (n : ℕ) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i) 
  (h_sum : (Finset.univ.sum x) = 1) : 
  (Finset.univ.sum (λ j => (x j)^4 - (x j)^5)) ≤ 1 / 12 :=
sorry

end max_sum_of_xj4_minus_xj5_l64_64500


namespace semicircle_problem_l64_64303

theorem semicircle_problem (N : ℕ) (r : ℝ) (π : ℝ) (hπ : 0 < π) 
  (h1 : ∀ (r : ℝ), ∃ (A B : ℝ), A = N * (π * r^2 / 2) ∧ B = (π * (N^2 * r^2 / 2) - N * (π * r^2 / 2)) ∧ A / B = 1 / 3) :
  N = 4 :=
by
  sorry

end semicircle_problem_l64_64303


namespace max_abs_eq_one_vertices_l64_64237

theorem max_abs_eq_one_vertices (x y : ℝ) :
  (max (|x + y|) (|x - y|) = 1) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
sorry

end max_abs_eq_one_vertices_l64_64237


namespace total_amount_l64_64046

noncomputable def A : ℝ := 360.00000000000006
noncomputable def B : ℝ := (3/2) * A
noncomputable def C : ℝ := 4 * B

theorem total_amount (A B C : ℝ)
  (hA : A = 360.00000000000006)
  (hA_B : A = (2/3) * B)
  (hB_C : B = (1/4) * C) :
  A + B + C = 3060.0000000000007 := by
  sorry

end total_amount_l64_64046


namespace school_club_profit_l64_64787

-- Definition of the problem conditions
def candy_bars_bought : ℕ := 800
def cost_per_four_bars : ℚ := 3
def bars_per_four_bars : ℕ := 4
def sell_price_per_three_bars : ℚ := 2
def bars_per_three_bars : ℕ := 3
def sales_fee_per_bar : ℚ := 0.05

-- Definition for cost calculations
def cost_per_bar : ℚ := cost_per_four_bars / bars_per_four_bars
def total_cost : ℚ := candy_bars_bought * cost_per_bar

-- Definition for revenue calculations
def sell_price_per_bar : ℚ := sell_price_per_three_bars / bars_per_three_bars
def total_revenue : ℚ := candy_bars_bought * sell_price_per_bar

-- Definition for total sales fee
def total_sales_fee : ℚ := candy_bars_bought * sales_fee_per_bar

-- Definition of profit
def profit : ℚ := total_revenue - total_cost - total_sales_fee

-- The statement to be proved
theorem school_club_profit : profit = -106.64 := by sorry

end school_club_profit_l64_64787


namespace polynomial_value_l64_64172

theorem polynomial_value (x y : ℝ) (h : x + 2 * y = 6) : 2 * x + 4 * y - 5 = 7 :=
by
  sorry

end polynomial_value_l64_64172


namespace motorist_travel_time_l64_64260

noncomputable def total_time (dist1 dist2 speed1 speed2 : ℝ) : ℝ :=
  (dist1 / speed1) + (dist2 / speed2)

theorem motorist_travel_time (speed1 speed2 : ℝ) (total_dist : ℝ) (half_dist : ℝ) :
  speed1 = 60 → speed2 = 48 → total_dist = 324 → half_dist = total_dist / 2 →
  total_time half_dist half_dist speed1 speed2 = 6.075 :=
by
  intros h1 h2 h3 h4
  simp [total_time, h1, h2, h3, h4]
  sorry

end motorist_travel_time_l64_64260


namespace probability_three_digit_multiple_5_remainder_3_div_7_l64_64465

theorem probability_three_digit_multiple_5_remainder_3_div_7 :
  (∃ (P : ℝ), P = (26 / 900)) := 
by sorry

end probability_three_digit_multiple_5_remainder_3_div_7_l64_64465


namespace employee_n_salary_l64_64117

theorem employee_n_salary (x : ℝ) (h : x + 1.2 * x = 583) : x = 265 := sorry

end employee_n_salary_l64_64117


namespace find_value_of_reciprocal_sin_double_angle_l64_64931

open Real

noncomputable def point := ℝ × ℝ

def term_side_angle_passes_through (α : ℝ) (P : point) :=
  ∃ (r : ℝ), P = (r * cos α, r * sin α)

theorem find_value_of_reciprocal_sin_double_angle (α : ℝ) (P : point) (h : term_side_angle_passes_through α P) :
  P = (-2, 1) → (1 / sin (2 * α)) = -5 / 4 :=
by
  intro hP
  sorry

end find_value_of_reciprocal_sin_double_angle_l64_64931


namespace solve_equation_l64_64276

theorem solve_equation (x : ℝ) : 
  (x - 4)^6 + (x - 6)^6 = 64 → x = 4 ∨ x = 6 :=
by
  sorry

end solve_equation_l64_64276


namespace scientific_notation_of_600000_l64_64995

theorem scientific_notation_of_600000 :
  600000 = 6 * 10^5 :=
sorry

end scientific_notation_of_600000_l64_64995


namespace amount_paid_is_correct_l64_64052

-- Conditions given in the problem
def jimmy_shorts_count : ℕ := 3
def jimmy_short_price : ℝ := 15.0
def irene_shirts_count : ℕ := 5
def irene_shirt_price : ℝ := 17.0
def discount_rate : ℝ := 0.10

-- Define the total cost for jimmy
def jimmy_total_cost : ℝ := jimmy_shorts_count * jimmy_short_price

-- Define the total cost for irene
def irene_total_cost : ℝ := irene_shirts_count * irene_shirt_price

-- Define the total cost before discount
def total_cost_before_discount : ℝ := jimmy_total_cost + irene_total_cost

-- Define the discount amount
def discount_amount : ℝ := total_cost_before_discount * discount_rate

-- Define the total amount to pay
def total_amount_to_pay : ℝ := total_cost_before_discount - discount_amount

-- The proposition we need to prove
theorem amount_paid_is_correct : total_amount_to_pay = 117 := by
  sorry

end amount_paid_is_correct_l64_64052


namespace volume_of_cube_surface_area_times_l64_64452

theorem volume_of_cube_surface_area_times (V1 : ℝ) (hV1 : V1 = 8) : 
  ∃ V2, V2 = 24 * Real.sqrt 3 :=
sorry

end volume_of_cube_surface_area_times_l64_64452


namespace subset_of_positive_reals_l64_64647

def M := { x : ℝ | x > -1 }

theorem subset_of_positive_reals : {0} ⊆ M :=
by
  sorry

end subset_of_positive_reals_l64_64647


namespace sum_of_different_roots_eq_six_l64_64740

theorem sum_of_different_roots_eq_six (a b : ℝ) (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_different_roots_eq_six_l64_64740


namespace find_z_to_8_l64_64251

noncomputable def complex_number_z (z : ℂ) : Prop :=
  z + z⁻¹ = 2 * Complex.cos (Real.pi / 4)

theorem find_z_to_8 (z : ℂ) (h : complex_number_z z) : (z ^ 8 + (z ^ 8)⁻¹ = 2) :=
by
  sorry

end find_z_to_8_l64_64251


namespace p_sufficient_not_necessary_q_l64_64011

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l64_64011


namespace solve_for_alpha_l64_64473

variables (α β γ δ : ℝ)

theorem solve_for_alpha (h : α + β + γ + δ = 360) : α = 360 - β - γ - δ :=
by sorry

end solve_for_alpha_l64_64473


namespace baseball_game_earnings_l64_64686

theorem baseball_game_earnings (W S : ℝ) 
  (h1 : W + S = 4994.50) 
  (h2 : W = S - 1330.50) : 
  S = 3162.50 := 
by 
  sorry

end baseball_game_earnings_l64_64686


namespace neg_p_is_true_neg_q_is_true_l64_64112

theorem neg_p_is_true : ∃ m : ℝ, ∀ x : ℝ, (x^2 + x - m = 0 → False) :=
sorry

theorem neg_q_is_true : ∀ x : ℝ, (x^2 + x + 1 > 0) :=
sorry

end neg_p_is_true_neg_q_is_true_l64_64112


namespace rectangular_floor_length_l64_64641

theorem rectangular_floor_length
    (cost_per_square : ℝ)
    (total_cost : ℝ)
    (carpet_length : ℝ)
    (carpet_width : ℝ)
    (floor_width : ℝ)
    (floor_area : ℝ) 
    (H1 : cost_per_square = 15)
    (H2 : total_cost = 225)
    (H3 : carpet_length = 2)
    (H4 : carpet_width = 2)
    (H5 : floor_width = 6)
    (H6 : floor_area = floor_width * carpet_length * carpet_width * 15): 
    floor_area / floor_width = 10 :=
by
  sorry

end rectangular_floor_length_l64_64641


namespace expression_D_divisible_by_9_l64_64191

theorem expression_D_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
by
  sorry

end expression_D_divisible_by_9_l64_64191


namespace sin_cos_15_degree_l64_64765

theorem sin_cos_15_degree :
  (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_15_degree_l64_64765


namespace sum_11_terms_l64_64915

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n)

def condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 7 = 14

-- Proof Problem
theorem sum_11_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum_formula : sum_first_n_terms S a)
  (h_condition : condition a) :
  S 11 = 77 := 
sorry

end sum_11_terms_l64_64915


namespace river_width_l64_64592

theorem river_width (boat_max_speed : ℝ) (river_current_speed : ℝ) (time_to_cross : ℝ) (width : ℝ) :
  boat_max_speed = 4 ∧ river_current_speed = 3 ∧ time_to_cross = 2 ∧ width = 8 → 
  width = boat_max_speed * time_to_cross := by
  intros h
  cases h
  sorry

end river_width_l64_64592


namespace die_face_never_lays_on_board_l64_64597

structure Chessboard :=
(rows : ℕ)
(cols : ℕ)
(h_size : rows = 8 ∧ cols = 8)

structure Die :=
(faces : Fin 6 → Nat)  -- a die has 6 faces

structure Position :=
(x : ℕ)
(y : ℕ)

structure State :=
(position : Position)
(bottom_face : Fin 6)
(visited : Fin 64 → Bool)

def initial_position : Position := ⟨0, 0⟩  -- top-left corner (a1)

def initial_state (d : Die) : State :=
  { position := initial_position,
    bottom_face := 0,
    visited := λ _ => false }

noncomputable def can_roll_over_entire_board_without_one_face_touching (board : Chessboard) (d : Die) : Prop :=
  ∃ f : Fin 6, ∀ s : State, -- for some face f of the die
    ((s.position.x < board.rows ∧ s.position.y < board.cols) → 
      s.visited (⟨s.position.x + board.rows * s.position.y, by sorry⟩) = true) → -- every cell visited
      ¬(s.bottom_face = f) -- face f is never the bottom face

theorem die_face_never_lays_on_board (board : Chessboard) (d : Die) :
  can_roll_over_entire_board_without_one_face_touching board d :=
  sorry

end die_face_never_lays_on_board_l64_64597


namespace inequality_proof_l64_64424

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
    (((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2) ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l64_64424


namespace quadratic_real_roots_exists_l64_64211

theorem quadratic_real_roots_exists :
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 * x1 - 6 * x1 + 8 = 0) ∧ (x2 * x2 - 6 * x2 + 8 = 0) :=
by
  sorry

end quadratic_real_roots_exists_l64_64211


namespace evaluate_expression_l64_64779

theorem evaluate_expression : ((3^4)^3 + 5) - ((4^3)^4 + 5) = -16245775 := by
  sorry

end evaluate_expression_l64_64779


namespace money_given_by_school_correct_l64_64491

-- Definitions from the problem conditions
def cost_per_book : ℕ := 12
def number_of_students : ℕ := 30
def out_of_pocket : ℕ := 40

-- Derived definition from these conditions
def total_cost : ℕ := cost_per_book * number_of_students
def money_given_by_school : ℕ := total_cost - out_of_pocket

-- The theorem stating that the amount given by the school is $320
theorem money_given_by_school_correct : money_given_by_school = 320 :=
by
  sorry -- Proof placeholder

end money_given_by_school_correct_l64_64491


namespace k_value_l64_64281

theorem k_value (m n k : ℤ) (h₁ : m + 2 * n + 5 = 0) (h₂ : (m + 2) + 2 * (n + k) + 5 = 0) : k = -1 :=
by sorry

end k_value_l64_64281


namespace scientific_notation_of_8200000_l64_64575

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l64_64575


namespace no_7_edges_edges_greater_than_5_l64_64030

-- Define the concept of a convex polyhedron in terms of its edges and faces.
structure ConvexPolyhedron where
  V : ℕ    -- Number of vertices
  E : ℕ    -- Number of edges
  F : ℕ    -- Number of faces
  Euler : V - E + F = 2   -- Euler's characteristic

-- Define properties of convex polyhedron

-- Part (a) statement: A convex polyhedron cannot have exactly 7 edges.
theorem no_7_edges (P : ConvexPolyhedron) : P.E ≠ 7 :=
sorry

-- Part (b) statement: A convex polyhedron can have any number of edges greater than 5 and different from 7.
theorem edges_greater_than_5 (n : ℕ) (h : n > 5) (h2 : n ≠ 7) : ∃ P : ConvexPolyhedron, P.E = n :=
sorry

end no_7_edges_edges_greater_than_5_l64_64030


namespace WillyLucyHaveMoreCrayons_l64_64433

-- Definitions from the conditions
def WillyCrayons : ℕ := 1400
def LucyCrayons : ℕ := 290
def MaxCrayons : ℕ := 650

-- Theorem statement
theorem WillyLucyHaveMoreCrayons : WillyCrayons + LucyCrayons - MaxCrayons = 1040 := 
by 
  sorry

end WillyLucyHaveMoreCrayons_l64_64433


namespace bridge_must_hold_weight_l64_64408

def weight_of_full_can (soda_weight empty_can_weight : ℕ) : ℕ :=
  soda_weight + empty_can_weight

def total_weight_of_full_cans (num_full_cans weight_per_full_can : ℕ) : ℕ :=
  num_full_cans * weight_per_full_can

def total_weight_of_empty_cans (num_empty_cans empty_can_weight : ℕ) : ℕ :=
  num_empty_cans * empty_can_weight

theorem bridge_must_hold_weight :
  let num_full_cans := 6
  let soda_weight := 12
  let empty_can_weight := 2
  let num_empty_cans := 2
  let weight_per_full_can := weight_of_full_can soda_weight empty_can_weight
  let total_full_cans_weight := total_weight_of_full_cans num_full_cans weight_per_full_can
  let total_empty_cans_weight := total_weight_of_empty_cans num_empty_cans empty_can_weight
  total_full_cans_weight + total_empty_cans_weight = 88 := by
  sorry

end bridge_must_hold_weight_l64_64408


namespace sequence_general_term_l64_64847

-- Define the sequence
def a : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * a n + 1

-- State the theorem
theorem sequence_general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

end sequence_general_term_l64_64847


namespace perfect_square_tens_place_l64_64002

/-- A whole number ending in 5 can only be a perfect square if the tens place is 2. -/
theorem perfect_square_tens_place (n : ℕ) (h₁ : n % 10 = 5) : ∃ k : ℕ, n = k * k → (n / 10) % 10 = 2 :=
sorry

end perfect_square_tens_place_l64_64002


namespace initial_kola_volume_l64_64393

theorem initial_kola_volume (V : ℝ) (S : ℝ) :
  S = 0.14 * V →
  (S + 3.2) / (V + 20) = 0.14111111111111112 →
  V = 340 :=
by
  intro h_S h_equation
  sorry

end initial_kola_volume_l64_64393


namespace calculate_expression_l64_64866

theorem calculate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 3*(x^4 + 2*y^2)/9 = 37 + 2/3 :=
by
  intros x y hx hy
  sorry

end calculate_expression_l64_64866


namespace smallest_positive_integer_ends_6996_l64_64969

theorem smallest_positive_integer_ends_6996 :
  ∃ m : ℕ, (m % 4 = 0 ∧ m % 9 = 0 ∧ ∀ d ∈ m.digits 10, d = 6 ∨ d = 9 ∧ m.digits 10 ∩ {6, 9} ≠ ∅ ∧ m % 10000 = 6996) :=
sorry

end smallest_positive_integer_ends_6996_l64_64969


namespace exists_point_P_equal_distance_squares_l64_64099

-- Define the points in the plane representing the vertices of the triangles
variables {A1 A2 A3 B1 B2 B3 C1 C2 C3 : ℝ × ℝ}
-- Define the function that calculates the square distance between two points
def sq_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Define the proof statement
theorem exists_point_P_equal_distance_squares :
  ∃ P : ℝ × ℝ,
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P B1 + sq_distance P B2 + sq_distance P B3 ∧
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P C1 + sq_distance P C2 + sq_distance P C3 := sorry

end exists_point_P_equal_distance_squares_l64_64099


namespace find_a2_l64_64798

variable (a : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℤ) : Prop :=
  y * y = x * z

-- Specific condition for the problem
axiom h_arithmetic : is_arithmetic_sequence a 2
axiom h_geometric : is_geometric_sequence (a 1 + 2) (a 3 + 6) (a 4 + 8)

-- Theorem to prove
theorem find_a2 : a 1 + 2 = -8 := 
sorry

-- We assert that the value of a_2 must satisfy the given conditions

end find_a2_l64_64798


namespace sum_of_number_and_radical_conjugate_l64_64213

theorem sum_of_number_and_radical_conjugate : 
  (10 - Real.sqrt 2018) + (10 + Real.sqrt 2018) = 20 := 
by 
  sorry

end sum_of_number_and_radical_conjugate_l64_64213


namespace certain_percentage_of_1600_l64_64040

theorem certain_percentage_of_1600 (P : ℝ) 
  (h : 0.05 * (P / 100 * 1600) = 20) : 
  P = 25 :=
by 
  sorry

end certain_percentage_of_1600_l64_64040


namespace field_ratio_l64_64315

theorem field_ratio (w l: ℕ) (h: l = 36)
  (h_area_ratio: 81 = (1/8) * l * w)
  (h_multiple: ∃ k : ℕ, l = k * w) :
  l / w = 2 :=
by 
  sorry

end field_ratio_l64_64315


namespace probability_of_two_mathematicians_living_contemporarily_l64_64100

noncomputable def probability_of_contemporary_lifespan : ℚ :=
  let total_area := 500 * 500
  let triangle_area := 0.5 * 380 * 380
  let non_overlap_area := 2 * triangle_area
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem probability_of_two_mathematicians_living_contemporarily :
  probability_of_contemporary_lifespan = 2232 / 5000 :=
by
  -- The actual proof would go here
  sorry

end probability_of_two_mathematicians_living_contemporarily_l64_64100


namespace tiling_2x12_l64_64637

def d : Nat → Nat
| 0     => 0  -- Unused but for safety in function definition
| 1     => 1
| 2     => 2
| (n+1) => d n + d (n-1)

theorem tiling_2x12 : d 12 = 233 := by
  sorry

end tiling_2x12_l64_64637


namespace area_covered_by_both_strips_is_correct_l64_64805

-- Definitions of lengths of the strips and areas
def length_total : ℝ := 16
def length_left : ℝ := 9
def length_right : ℝ := 7
def area_left_only : ℝ := 27
def area_right_only : ℝ := 18

noncomputable def width_strip : ℝ := sorry -- The width can be inferred from solution but is not the focus of the proof.

-- Definition of the area covered by both strips
def S : ℝ := 13.5

-- Proof statement
theorem area_covered_by_both_strips_is_correct :
  ∀ w : ℝ,
    length_left * w - S = area_left_only ∧ length_right * w - S = area_right_only →
    S = 13.5 := 
by
  sorry

end area_covered_by_both_strips_is_correct_l64_64805


namespace not_all_x_ne_1_imp_x2_ne_0_l64_64722

theorem not_all_x_ne_1_imp_x2_ne_0 : ¬ (∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) :=
sorry

end not_all_x_ne_1_imp_x2_ne_0_l64_64722


namespace max_k_l64_64517

def seq (a : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = k * (a n) ^ 2 + 1

def bounded (a : ℕ → ℝ) (c : ℝ) : Prop :=
∀ n : ℕ, a n < c

theorem max_k (k : ℝ) (c : ℝ) (a : ℕ → ℝ) :
  a 1 = 1 →
  seq a k →
  bounded a c →
  0 < k ∧ k ≤ 1 / 4 :=
by
  sorry

end max_k_l64_64517


namespace admission_cutoff_score_l64_64347

theorem admission_cutoff_score (n : ℕ) (x : ℚ) (admitted_average non_admitted_average total_average : ℚ)
    (h1 : admitted_average = x + 15)
    (h2 : non_admitted_average = x - 20)
    (h3 : total_average = 90)
    (h4 : (admitted_average * (2 / 5) + non_admitted_average * (3 / 5)) = total_average) : x = 96 := 
by
  sorry

end admission_cutoff_score_l64_64347


namespace concentric_circle_ratio_l64_64811

theorem concentric_circle_ratio (r R : ℝ) (hRr : R > r)
  (new_circles_tangent : ∀ (C1 C2 C3 : ℝ), C1 = C2 ∧ C2 = C3 ∧ C1 < R ∧ r < C1): 
  R = 3 * r := by sorry

end concentric_circle_ratio_l64_64811


namespace evaluate_expression_l64_64680

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end evaluate_expression_l64_64680


namespace ticket_cost_is_nine_l64_64241

theorem ticket_cost_is_nine (bought_tickets : ℕ) (left_tickets : ℕ) (spent_dollars : ℕ) 
  (h1 : bought_tickets = 6) 
  (h2 : left_tickets = 3) 
  (h3 : spent_dollars = 27) : 
  spent_dollars / (bought_tickets - left_tickets) = 9 :=
by
  -- Using the imported library and the given conditions
  sorry

end ticket_cost_is_nine_l64_64241


namespace odd_not_div_by_3_l64_64512

theorem odd_not_div_by_3 (n : ℤ) (h1 : Odd n) (h2 : ¬ ∃ k : ℤ, n = 3 * k) : 6 ∣ (n^2 + 5) :=
  sorry

end odd_not_div_by_3_l64_64512


namespace cube_root_of_nine_irrational_l64_64917

theorem cube_root_of_nine_irrational : ¬ ∃ (r : ℚ), r^3 = 9 :=
by sorry

end cube_root_of_nine_irrational_l64_64917


namespace sequence_general_term_l64_64907

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

theorem sequence_general_term (h : ∀ n : ℕ, S n = 2 * n - a n) :
  ∀ n : ℕ, a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_general_term_l64_64907


namespace quadratic_inequality_l64_64529

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ 0 ≤ k ∧ k ≤ 16 :=
by sorry

end quadratic_inequality_l64_64529


namespace average_of_quantities_l64_64701

theorem average_of_quantities (a1 a2 a3 a4 a5 : ℝ) :
  ((a1 + a2 + a3) / 3 = 4) →
  ((a4 + a5) / 2 = 21.5) →
  ((a1 + a2 + a3 + a4 + a5) / 5 = 11) :=
by
  intros h3 h2
  sorry

end average_of_quantities_l64_64701


namespace child_is_late_l64_64409

theorem child_is_late 
  (distance : ℕ)
  (rate1 rate2 : ℕ) 
  (early_arrival : ℕ)
  (time_late_at_rate1 : ℕ)
  (time_required_by_rate1 : ℕ)
  (time_required_by_rate2 : ℕ)
  (actual_time : ℕ)
  (T : ℕ) :
  distance = 630 ∧ 
  rate1 = 5 ∧ 
  rate2 = 7 ∧ 
  early_arrival = 30 ∧
  (time_required_by_rate1 = distance / rate1) ∧
  (time_required_by_rate2 = distance / rate2) ∧
  (actual_time + T = time_required_by_rate1) ∧
  (actual_time - early_arrival = time_required_by_rate2) →
  T = 6 := 
by
  intros
  sorry

end child_is_late_l64_64409


namespace bottles_last_days_l64_64518

theorem bottles_last_days :
  let total_bottles := 8066
  let bottles_per_day := 109
  total_bottles / bottles_per_day = 74 :=
by
  sorry

end bottles_last_days_l64_64518


namespace concave_sequence_count_l64_64108

   theorem concave_sequence_count (m : ℕ) (h : 2 ≤ m) :
     ∀ b_0, (b_0 = 1 ∨ b_0 = 2) → 
     (∃ b : ℕ → ℕ, (∀ k, 2 ≤ k ∧ k ≤ m → b k + b (k - 2) ≤ 2 * b (k - 1)) → 
     (∃ S : ℕ, S ≤ 2^m)) :=
   by 
     sorry
   
end concave_sequence_count_l64_64108


namespace range_of_a_l64_64689

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l64_64689


namespace no_such_function_exists_l64_64372

theorem no_such_function_exists (f : ℝ → ℝ) (Hf : ∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) : False :=
by
  sorry

end no_such_function_exists_l64_64372


namespace percent_of_ac_is_db_l64_64001

variable (a b c d : ℝ)

-- Given conditions
variable (h1 : c = 0.25 * a)
variable (h2 : c = 0.10 * b)
variable (h3 : d = 0.50 * b)

-- Theorem statement: Prove the final percentage
theorem percent_of_ac_is_db : (d * b) / (a * c) * 100 = 1250 :=
by
  sorry

end percent_of_ac_is_db_l64_64001


namespace middle_integer_is_zero_l64_64819

-- Mathematical equivalent proof problem in Lean 4

theorem middle_integer_is_zero
  (n : ℤ)
  (h : (n - 2) + n + (n + 2) = (1 / 5) * ((n - 2) * n * (n + 2))) :
  n = 0 :=
by
  sorry

end middle_integer_is_zero_l64_64819


namespace polar_to_cartesian_l64_64495

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the goal as a theorem
theorem polar_to_cartesian : ∀ (x y : ℝ), 
  (∃ θ : ℝ, polar_eq (Real.sqrt (x^2 + y^2)) θ ∧ x = (Real.sqrt (x^2 + y^2)) * Real.cos θ 
  ∧ y = (Real.sqrt (x^2 + y^2)) * Real.sin θ) → (x-1)^2 + y^2 = 1 :=
by
  intro x y
  intro h
  sorry

end polar_to_cartesian_l64_64495


namespace total_books_l64_64871

theorem total_books (Zig_books : ℕ) (Flo_books : ℕ) (Tim_books : ℕ) 
  (hz : Zig_books = 60) (hf : Zig_books = 4 * Flo_books) (ht : Tim_books = Flo_books / 2) :
  Zig_books + Flo_books + Tim_books = 82 := by
  sorry

end total_books_l64_64871


namespace minimum_value_l64_64898

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem minimum_value :
  ∃ x₀ : ℝ, (∀ x : ℝ, f x₀ ≤ f x) ∧ f x₀ = -2 := by
  sorry

end minimum_value_l64_64898


namespace min_value_abs_2a_minus_b_l64_64010

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) : ∃ c : ℝ, c = |2 * a - b| ∧ c = 1 := 
sorry

end min_value_abs_2a_minus_b_l64_64010


namespace area_of_equilateral_triangle_example_l64_64576

noncomputable def area_of_equilateral_triangle_with_internal_point (a b c : ℝ) (d_pa : ℝ) (d_pb : ℝ) (d_pc : ℝ) : ℝ :=
  if h : ((d_pa = 3) ∧ (d_pb = 4) ∧ (d_pc = 5)) then
    (9 + (25 * Real.sqrt 3)/4)
  else
    0

theorem area_of_equilateral_triangle_example :
  area_of_equilateral_triangle_with_internal_point 3 4 5 3 4 5 = 9 + (25 * Real.sqrt 3)/4 :=
  by sorry

end area_of_equilateral_triangle_example_l64_64576


namespace part1_part2_l64_64697

noncomputable def inverse_function_constant (k : ℝ) : Prop :=
  (∀ x : ℝ, 0 < x → (x, 3) ∈ {p : ℝ × ℝ | p.snd = k / p.fst})

noncomputable def range_m (m : ℝ) : Prop :=
  0 < m → m < 3

theorem part1 (k : ℝ) (hk : k ≠ 0) (h : (1, 3).snd = k / (1, 3).fst) :
  k = 3 := by
  sorry

theorem part2 (m : ℝ) (hm : m ≠ 0) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) := by
  sorry

end part1_part2_l64_64697


namespace Faraway_not_possible_sum_l64_64630

theorem Faraway_not_possible_sum (h g : ℕ) : (74 ≠ 21 * h + 6 * g) ∧ (89 ≠ 21 * h + 6 * g) :=
by
  sorry

end Faraway_not_possible_sum_l64_64630


namespace average_of_remaining_numbers_l64_64860

theorem average_of_remaining_numbers (sum : ℕ) (average : ℕ) (remaining_sum : ℕ) (remaining_average : ℚ) :
  (average = 90) →
  (sum = 1080) →
  (remaining_sum = sum - 72 - 84) →
  (remaining_average = remaining_sum / 10) →
  remaining_average = 92.4 :=
by
  sorry

end average_of_remaining_numbers_l64_64860


namespace fill_time_with_leak_l64_64012

theorem fill_time_with_leak (A L : ℝ) (hA : A = 1 / 5) (hL : L = 1 / 10) :
  1 / (A - L) = 10 :=
by 
  sorry

end fill_time_with_leak_l64_64012


namespace coeff_x3_in_expansion_l64_64899

theorem coeff_x3_in_expansion : 
  ∃ c : ℕ, (c = 80) ∧ (∃ r : ℕ, r = 1 ∧ (2 * x + 1 / x) ^ 5 = (2 * x) ^ (5 - r) * (1 / x) ^ r)
:= sorry

end coeff_x3_in_expansion_l64_64899


namespace order_numbers_l64_64152

theorem order_numbers : (5 / 2 : ℝ) < (3 : ℝ) ∧ (3 : ℝ) < Real.sqrt (10) := 
by
  sorry

end order_numbers_l64_64152


namespace sum_of_legs_is_104_l64_64977

theorem sum_of_legs_is_104 (x : ℕ) (h₁ : x^2 + (x + 2)^2 = 53^2) : x + (x + 2) = 104 := sorry

end sum_of_legs_is_104_l64_64977


namespace trigonometric_identity_l64_64834

theorem trigonometric_identity
  (θ : ℝ)
  (h : Real.tan θ = 1 / 3) :
  Real.sin (3 / 2 * Real.pi + 2 * θ) = -4 / 5 :=
by sorry

end trigonometric_identity_l64_64834


namespace find_added_value_l64_64714

theorem find_added_value (avg_15_numbers : ℤ) (new_avg : ℤ) (x : ℤ)
    (H1 : avg_15_numbers = 40) 
    (H2 : new_avg = 50) 
    (H3 : (600 + 15 * x) / 15 = new_avg) : 
    x = 10 := 
sorry

end find_added_value_l64_64714


namespace number_of_throwers_l64_64335

theorem number_of_throwers (total_players throwers right_handed : ℕ) 
  (h1 : total_players = 64)
  (h2 : right_handed = 55) 
  (h3 : ∀ T N, T + N = total_players → 
  T + (2/3 : ℚ) * N = right_handed) : 
  throwers = 37 := 
sorry

end number_of_throwers_l64_64335


namespace gain_percent_l64_64618

theorem gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1440) : 
  ((selling_price - cost_price) / cost_price) * 100 = 60 :=
by
  sorry

end gain_percent_l64_64618


namespace total_boxes_sold_is_189_l64_64987

-- Define the conditions
def boxes_sold_friday : ℕ := 40
def boxes_sold_saturday := 2 * boxes_sold_friday - 10
def boxes_sold_sunday := boxes_sold_saturday / 2
def boxes_sold_monday := boxes_sold_sunday + (boxes_sold_sunday / 4)

-- Define the total boxes sold over the four days
def total_boxes_sold := boxes_sold_friday + boxes_sold_saturday + boxes_sold_sunday + boxes_sold_monday

-- Theorem to prove the total number of boxes sold is 189
theorem total_boxes_sold_is_189 : total_boxes_sold = 189 := by
  sorry

end total_boxes_sold_is_189_l64_64987


namespace cost_two_cones_l64_64736

-- Definition for the cost of a single ice cream cone
def cost_one_cone : ℕ := 99

-- The theorem to prove the cost of two ice cream cones
theorem cost_two_cones : 2 * cost_one_cone = 198 := 
by 
  sorry

end cost_two_cones_l64_64736


namespace area_of_inscribed_rectangle_l64_64003

theorem area_of_inscribed_rectangle (h_triangle_altitude : 12 > 0)
  (h_segment_XZ : 15 > 0)
  (h_PQ_eq_one_third_PS : ∀ PQ PS : ℚ, PS = 3 * PQ) :
  ∃ PQ PS : ℚ, 
    (YM = 12) ∧
    (XZ = 15) ∧
    (PQ = (15 / 8 : ℚ)) ∧
    (PS = 3 * PQ) ∧ 
    ((PQ * PS) = (675 / 64 : ℚ)) :=
by
  -- Proof would go here.
  sorry

end area_of_inscribed_rectangle_l64_64003


namespace rectangular_prism_diagonals_l64_64998

theorem rectangular_prism_diagonals (length width height : ℕ) (length_eq : length = 4) (width_eq : width = 3) (height_eq : height = 2) : 
  ∃ (total_diagonals : ℕ), total_diagonals = 16 :=
by
  let face_diagonals := 12
  let space_diagonals := 4
  let total_diagonals := face_diagonals + space_diagonals
  use total_diagonals
  sorry

end rectangular_prism_diagonals_l64_64998


namespace rectangle_area_ratio_l64_64119

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := 
  sorry

end rectangle_area_ratio_l64_64119


namespace shape_described_by_constant_phi_is_cone_l64_64488

-- Definition of spherical coordinates
-- (ρ, θ, φ) where ρ is the radial distance,
-- θ is the azimuthal angle, and φ is the polar angle.
structure SphericalCoordinates :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- The condition that φ is equal to a constant d
def satisfies_condition (p : SphericalCoordinates) (d : ℝ) : Prop :=
  p.φ = d

-- The main theorem to prove
theorem shape_described_by_constant_phi_is_cone (d : ℝ) :
  ∃ (S : Set SphericalCoordinates), (∀ p ∈ S, satisfies_condition p d) ∧
  (∀ p, satisfies_condition p d → ∃ ρ θ, p = ⟨ρ, θ, d⟩) ∧
  (∀ ρ θ, ρ > 0 → θ ∈ [0, 2 * Real.pi] → SphericalCoordinates.mk ρ θ d ∈ S) :=
sorry

end shape_described_by_constant_phi_is_cone_l64_64488


namespace min_n_1014_dominoes_l64_64667

theorem min_n_1014_dominoes (n : ℕ) :
  (n + 1) ^ 2 ≥ 6084 → n ≥ 77 :=
sorry

end min_n_1014_dominoes_l64_64667


namespace add_num_denom_fraction_l64_64027

theorem add_num_denom_fraction (n : ℚ) : (2 + n) / (7 + n) = 3 / 5 ↔ n = 11 / 2 := 
by
  sorry

end add_num_denom_fraction_l64_64027


namespace net_pay_rate_is_26_dollars_per_hour_l64_64333

-- Defining the conditions
noncomputable def total_distance (time_hours : ℝ) (speed_mph : ℝ) : ℝ :=
  time_hours * speed_mph

noncomputable def adjusted_fuel_efficiency (original_efficiency : ℝ) (decrease_percentage : ℝ) : ℝ :=
  original_efficiency * (1 - decrease_percentage)

noncomputable def gasoline_used (distance : ℝ) (efficiency : ℝ) : ℝ :=
  distance / efficiency

noncomputable def earnings (rate_per_mile : ℝ) (distance : ℝ) : ℝ :=
  rate_per_mile * distance

noncomputable def updated_gasoline_price (original_price : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_price * (1 + increase_percentage)

noncomputable def total_cost_gasoline (gasoline_price : ℝ) (gasoline_used : ℝ) : ℝ :=
  gasoline_price * gasoline_used

noncomputable def net_earnings (earnings : ℝ) (cost : ℝ) : ℝ :=
  earnings - cost

noncomputable def net_rate_of_pay (net_earnings : ℝ) (time_hours : ℝ) : ℝ :=
  net_earnings / time_hours

-- Given constants
def time_hours : ℝ := 3
def speed_mph : ℝ := 50
def original_efficiency : ℝ := 30
def decrease_percentage : ℝ := 0.10
def rate_per_mile : ℝ := 0.60
def original_gasoline_price : ℝ := 2.00
def increase_percentage : ℝ := 0.20

-- Proof problem statement
theorem net_pay_rate_is_26_dollars_per_hour :
  net_rate_of_pay 
    (net_earnings
       (earnings rate_per_mile (total_distance time_hours speed_mph))
       (total_cost_gasoline
          (updated_gasoline_price original_gasoline_price increase_percentage)
          (gasoline_used
            (total_distance time_hours speed_mph)
            (adjusted_fuel_efficiency original_efficiency decrease_percentage))))
    time_hours = 26 := 
  sorry

end net_pay_rate_is_26_dollars_per_hour_l64_64333


namespace pencils_ratio_l64_64230

theorem pencils_ratio
  (Sarah_pencils : ℕ)
  (Tyrah_pencils : ℕ)
  (Tim_pencils : ℕ)
  (h1 : Tyrah_pencils = 12)
  (h2 : Tim_pencils = 16)
  (h3 : Tim_pencils = 8 * Sarah_pencils) :
  Tyrah_pencils / Sarah_pencils = 6 :=
by
  sorry

end pencils_ratio_l64_64230


namespace largest_stamps_per_page_l64_64348

theorem largest_stamps_per_page (h1 : Nat := 1050) (h2 : Nat := 1260) (h3 : Nat := 1470) :
  Nat.gcd h1 (Nat.gcd h2 h3) = 210 :=
by
  sorry

end largest_stamps_per_page_l64_64348


namespace simplify_and_evaluate_l64_64081

-- Given conditions: x = 1/3 and y = -1/2
def x : ℚ := 1 / 3
def y : ℚ := -1 / 2

-- Problem statement: 
-- Prove that (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2
theorem simplify_and_evaluate :
  (2 * x + 3 * y)^2 - (2 * x + y) * (2 * x - y) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_l64_64081


namespace B_correct_A_inter_B_correct_l64_64160

def A := {x : ℝ | 1 < x ∧ x < 8}
def B := {x : ℝ | x^2 - 5 * x - 14 ≥ 0}

theorem B_correct : B = {x : ℝ | x ≤ -2 ∨ x ≥ 7} := 
sorry

theorem A_inter_B_correct : A ∩ B = {x : ℝ | 7 ≤ x ∧ x < 8} :=
sorry

end B_correct_A_inter_B_correct_l64_64160


namespace number_of_methods_l64_64042

def doctors : ℕ := 6
def days : ℕ := 3

theorem number_of_methods : (days^doctors) = 729 := 
by sorry

end number_of_methods_l64_64042


namespace correct_solutions_l64_64607

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end correct_solutions_l64_64607


namespace polar_to_cartesian_equiv_l64_64632

noncomputable def polar_to_cartesian (rho theta : ℝ) : Prop :=
  let x := rho * Real.cos theta
  let y := rho * Real.sin theta
  (Real.sqrt 3 * x + y = 2) ↔ (rho * Real.cos (theta - Real.pi / 6) = 1)

theorem polar_to_cartesian_equiv (rho theta : ℝ) : polar_to_cartesian rho theta :=
by
  sorry

end polar_to_cartesian_equiv_l64_64632


namespace valid_numbers_are_135_and_144_l64_64415

noncomputable def find_valid_numbers : List ℕ :=
  let numbers := [135, 144]
  numbers.filter (λ n =>
    let a := n / 100
    let b := (n / 10) % 10
    let c := n % 10
    n = (100 * a + 10 * b + c) ∧ n = a * b * c * (a + b + c)
  )

theorem valid_numbers_are_135_and_144 :
  find_valid_numbers = [135, 144] :=
by
  sorry

end valid_numbers_are_135_and_144_l64_64415


namespace repeating_decimal_to_fraction_denominator_l64_64362

theorem repeating_decimal_to_fraction_denominator :
  ∀ (S : ℚ), (S = 0.27) → (∃ a b : ℤ, b ≠ 0 ∧ S = a / b ∧ Int.gcd a b = 1 ∧ b = 3) :=
by
  sorry

end repeating_decimal_to_fraction_denominator_l64_64362


namespace reciprocal_neg_half_l64_64973

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l64_64973


namespace school_points_l64_64330

theorem school_points (a b c : ℕ) (h1 : a + b + c = 285)
  (h2 : ∃ x : ℕ, a - 8 = x ∧ b - 12 = x ∧ c - 7 = x) : a + c = 187 :=
sorry

end school_points_l64_64330


namespace divisible_by_900_l64_64785

theorem divisible_by_900 (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) := 
by 
  sorry

end divisible_by_900_l64_64785


namespace product_of_four_consecutive_integers_is_perfect_square_l64_64628

theorem product_of_four_consecutive_integers_is_perfect_square :
  ∃ k : ℤ, ∃ n : ℤ, k = (n-1) * n * (n+1) * (n+2) ∧
    k = 0 ∧
    ((n = 0) ∨ (n = -1) ∨ (n = 1) ∨ (n = -2)) :=
by
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l64_64628


namespace men_in_second_group_l64_64932

theorem men_in_second_group (M : ℕ) : 
    (18 * 20 = M * 24) → M = 15 :=
by
  intro h
  sorry

end men_in_second_group_l64_64932


namespace smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l64_64278

theorem smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits : 
  ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1 ∧ d % 2 = 0) ∧ 
    (n % 11 = 0)) ∧ n = 1056 :=
by
  sorry

end smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l64_64278


namespace largest_number_with_four_digits_divisible_by_72_is_9936_l64_64552

theorem largest_number_with_four_digits_divisible_by_72_is_9936 :
  ∃ n : ℕ, (n < 10000 ∧ n ≥ 1000) ∧ (72 ∣ n) ∧ (∀ m : ℕ, (m < 10000 ∧ m ≥ 1000) ∧ (72 ∣ m) → m ≤ n) :=
sorry

end largest_number_with_four_digits_divisible_by_72_is_9936_l64_64552


namespace two_digit_numbers_non_repeating_l64_64790

-- The set of available digits is given as 0, 1, 2, 3, 4
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Ensure the tens place digits are subset of 1, 2, 3, 4 (exclude 0)
def valid_tens : List ℕ := [1, 2, 3, 4]

theorem two_digit_numbers_non_repeating :
  let num_tens := valid_tens.length
  let num_units := (digits.length - 1)
  num_tens * num_units = 16 :=
by
  -- Observe num_tens = 4, since valid_tens = [1, 2, 3, 4]
  -- Observe num_units = 4, since digits.length = 5 and we exclude the tens place digit
  sorry

end two_digit_numbers_non_repeating_l64_64790


namespace no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l64_64994

-- Definition of a natural number being divisible by another
def divisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

-- Definition of the sum of the digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

-- Statement of the problem
theorem no_nat_number_divisible_by_1998_has_digit_sum_lt_27 :
  ¬ ∃ n : ℕ, divisible n 1998 ∧ sum_of_digits n < 27 :=
by 
  sorry

end no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l64_64994


namespace complex_magnitude_l64_64205

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z with the given condition
variable (z : ℂ) (h : z * (1 + i) = 2 * i)

-- Statement of the problem: Prove that |z + 2 * i| = √10
theorem complex_magnitude (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs (z + 2 * i) = Real.sqrt 10 := 
sorry

end complex_magnitude_l64_64205


namespace identical_solutions_k_value_l64_64677

theorem identical_solutions_k_value (k : ℝ) :
  (∀ (x y : ℝ), y = x^2 ∧ y = 4 * x + k → (x - 2)^2 = 0) → k = -4 :=
by
  sorry

end identical_solutions_k_value_l64_64677


namespace union_sets_l64_64158

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_sets : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_sets_l64_64158


namespace right_triangle_sides_l64_64818

theorem right_triangle_sides :
  (4^2 + 5^2 ≠ 6^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (6^2 + 8^2 ≠ 11^2) ∧
  (5^2 + 12^2 ≠ 23^2) :=
by
  repeat { sorry }

end right_triangle_sides_l64_64818


namespace linear_function_not_passing_through_third_quadrant_l64_64044

theorem linear_function_not_passing_through_third_quadrant
  (m : ℝ)
  (h : 4 + 4 * m < 0) : 
  ∀ x y : ℝ, (y = m * x - m) → ¬ (x < 0 ∧ y < 0) :=
by
  sorry

end linear_function_not_passing_through_third_quadrant_l64_64044


namespace pebbles_collected_by_tenth_day_l64_64593

-- Define the initial conditions
def a : ℕ := 2
def r : ℕ := 2
def n : ℕ := 10

-- Total pebbles collected by the end of the 10th day
def total_pebbles (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Proof statement
theorem pebbles_collected_by_tenth_day : total_pebbles a r n = 2046 :=
  by sorry

end pebbles_collected_by_tenth_day_l64_64593


namespace closest_integer_to_cuberoot_of_200_l64_64696

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l64_64696


namespace problem1_problem2_l64_64746

noncomputable def f (x a c : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

-- Problem 1: Prove that for c = 19, the inequality f(1, a, 19) > 0 holds for -2 < a < 8
theorem problem1 (a : ℝ) : f 1 a 19 > 0 ↔ -2 < a ∧ a < 8 :=
by sorry

-- Problem 2: Given that f(x) > 0 has solution set (-1, 3), find a and c
theorem problem2 (a c : ℝ) (hx : ∀ x, -1 < x ∧ x < 3 → f x a c > 0) : 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9 :=
by sorry

end problem1_problem2_l64_64746


namespace toothbrush_count_l64_64130

theorem toothbrush_count (T A : ℕ) (h1 : 53 + 67 + 46 = 166)
  (h2 : 67 - 36 = 31) (h3 : A = 31) (h4 : T = 166 + 2 * A) :
  T = 228 :=
  by 
  -- Using Lean's sorry keyword to skip the proof
  sorry

end toothbrush_count_l64_64130


namespace maximum_area_of_sector_l64_64385

theorem maximum_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 10) : 
  (1 / 2 * l * r) ≤ 25 / 4 := 
sorry

end maximum_area_of_sector_l64_64385


namespace xiao_ming_proposition_false_l64_64675

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m * m ≤ n → m = 1 ∨ m = n → m ∣ n

def check_xiao_ming_proposition : Prop :=
  ∃ n : ℕ, ∃ (k : ℕ), k < n → ∃ (p q : ℕ), p = q → n^2 - n + 11 = p * q ∧ p > 1 ∧ q > 1

theorem xiao_ming_proposition_false : ¬ (∀ n: ℕ, is_prime (n^2 - n + 11)) :=
by
  sorry

end xiao_ming_proposition_false_l64_64675


namespace min_value_of_expression_l64_64523

theorem min_value_of_expression (x y z : ℝ) : ∃ a : ℝ, (∀ x y z : ℝ, x^2 + x * y + y^2 + y * z + z^2 ≥ a) ∧ (a = 0) :=
sorry

end min_value_of_expression_l64_64523


namespace factor_expression_l64_64633

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l64_64633


namespace moles_of_HNO3_l64_64339

theorem moles_of_HNO3 (HNO3 NaHCO3 NaNO3 : ℝ)
  (h1 : NaHCO3 = 1) (h2 : NaNO3 = 1) :
  HNO3 = 1 :=
by sorry

end moles_of_HNO3_l64_64339


namespace value_a8_l64_64514

def sequence_sum (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem value_a8 : a 8 = 15 :=
by
  sorry

end value_a8_l64_64514


namespace initial_cupcakes_l64_64475

   theorem initial_cupcakes (X : ℕ) (condition : X - 20 + 20 = 26) : X = 26 :=
   by
     sorry
   
end initial_cupcakes_l64_64475


namespace min_value_of_x2_plus_y2_l64_64981

open Real

theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) :
  x^2 + y^2 ≥ 7 - 4 * sqrt 3 := sorry

end min_value_of_x2_plus_y2_l64_64981


namespace range_of_a_l64_64149

theorem range_of_a (a : ℝ)
  (A : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0})
  (B : Set ℝ := {x : ℝ | x ≥ a - 1})
  (H : A ∪ B = Set.univ) :
  a ≤ 2 :=
by
  sorry

end range_of_a_l64_64149


namespace find_y_l64_64832

theorem find_y (y : ℝ) (h_cond : y = (1 / y) * (-y) - 3) : y = -4 := 
sorry

end find_y_l64_64832


namespace triangle_area_on_ellipse_l64_64791

def onEllipse (p : ℝ × ℝ) : Prop := (p.1)^2 + 4 * (p.2)^2 = 4

def isCentroid (C : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  C = ((A.1 + B.1) / 3, (A.2 + B.2) / 3)

theorem triangle_area_on_ellipse
  (A B C : ℝ × ℝ)
  (h₁ : A ≠ B)
  (h₂ : B ≠ C)
  (h₃ : C ≠ A)
  (h₄ : onEllipse A)
  (h₅ : onEllipse B)
  (h₆ : onEllipse C)
  (h₇ : isCentroid C A B)
  (h₈ : C = (0, 0))  : 
  1 / 2 * (A.1 - B.1) * (B.2 - A.2) = 1 :=
by
  sorry

end triangle_area_on_ellipse_l64_64791


namespace problem_intersection_l64_64254

theorem problem_intersection (a b : ℝ) 
    (h1 : b = - 2 / a) 
    (h2 : b = a + 3) 
    : 1 / a - 1 / b = -3 / 2 :=
by
  sorry

end problem_intersection_l64_64254


namespace perpendicular_lines_l64_64662

theorem perpendicular_lines (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0) ∧ (∀ x y : ℝ, 2 * x + m * y - 6 = 0) → m = -1 :=
by
  sorry

end perpendicular_lines_l64_64662


namespace joseph_total_payment_l64_64310
-- Importing necessary libraries

-- Defining the variables and conditions
variables (W : ℝ) -- The cost for the water heater

-- Conditions
def condition1 := 3 * W -- The cost for the refrigerator
def condition2 := 2 * W = 500 -- The electric oven
def condition3 := 300 -- The cost for the air conditioner
def condition4 := 100 -- The cost for the washing machine

-- Calculate total cost
def total_cost := (3 * W) + W + 500 + 300 + 100

-- The theorem stating the total amount Joseph pays
theorem joseph_total_payment : total_cost = 1900 :=
by 
  have hW := condition2;
  sorry

end joseph_total_payment_l64_64310


namespace largest_divisor_of_m_l64_64643

theorem largest_divisor_of_m (m : ℕ) (h1 : 0 < m) (h2 : 39 ∣ m^2) : 39 ∣ m := sorry

end largest_divisor_of_m_l64_64643


namespace find_center_of_circle_l64_64314

theorem find_center_of_circle (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 16 = 0 →
  (x + 1)^2 + (y - 3)^2 = 6 :=
by
  intro h
  sorry

end find_center_of_circle_l64_64314


namespace shaded_fraction_l64_64596

theorem shaded_fraction (side_length : ℝ) (base : ℝ) (height : ℝ) (H1: side_length = 4) (H2: base = 3) (H3: height = 2):
  ((side_length ^ 2) - 2 * (1 / 2 * base * height)) / (side_length ^ 2) = 5 / 8 := by
  sorry

end shaded_fraction_l64_64596


namespace greatest_M_inequality_l64_64498

theorem greatest_M_inequality :
  ∀ x y z : ℝ, x^4 + y^4 + z^4 + x * y * z * (x + y + z) ≥ (2/3) * (x * y + y * z + z * x)^2 :=
by
  sorry

end greatest_M_inequality_l64_64498


namespace gcd_of_X_and_Y_l64_64895

theorem gcd_of_X_and_Y (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : 5 * X = 4 * Y) :
  Nat.gcd X Y = 9 := 
sorry

end gcd_of_X_and_Y_l64_64895


namespace complement_angle_l64_64398

theorem complement_angle (A : Real) (h : A = 55) : 90 - A = 35 := by
  sorry

end complement_angle_l64_64398


namespace ducks_and_dogs_total_l64_64710

theorem ducks_and_dogs_total (d g : ℕ) (h1 : d = g + 2) (h2 : 4 * g - 2 * d = 10) : d + g = 16 :=
  sorry

end ducks_and_dogs_total_l64_64710


namespace breaks_difference_l64_64134

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end breaks_difference_l64_64134


namespace total_tin_in_new_alloy_l64_64033

-- Define the weights of alloy A and alloy B
def weightAlloyA : Float := 135
def weightAlloyB : Float := 145

-- Define the ratio of lead to tin in alloy A
def ratioLeadToTinA : Float := 3 / 5

-- Define the ratio of tin to copper in alloy B
def ratioTinToCopperB : Float := 2 / 3

-- Define the total parts for alloy A and alloy B
def totalPartsA : Float := 3 + 5
def totalPartsB : Float := 2 + 3

-- Define the fraction of tin in alloy A and alloy B
def fractionTinA : Float := 5 / totalPartsA
def fractionTinB : Float := 2 / totalPartsB

-- Calculate the amount of tin in alloy A and alloy B
def tinInAlloyA : Float := fractionTinA * weightAlloyA
def tinInAlloyB : Float := fractionTinB * weightAlloyB

-- Calculate the total amount of tin in the new alloy
def totalTinInNewAlloy : Float := tinInAlloyA + tinInAlloyB

-- The theorem to be proven
theorem total_tin_in_new_alloy : totalTinInNewAlloy = 142.375 := by
  sorry

end total_tin_in_new_alloy_l64_64033


namespace cups_added_l64_64359

/--
A bowl was half full of water. Some cups of water were then added to the bowl, filling the bowl to 70% of its capacity. There are now 14 cups of water in the bowl.
Prove that the number of cups of water added to the bowl is 4.
-/
theorem cups_added (C : ℚ) (h1 : C / 2 + 0.2 * C = 14) : 
  14 - C / 2 = 4 :=
by
  sorry

end cups_added_l64_64359


namespace height_of_pyramid_equal_to_cube_volume_l64_64426

theorem height_of_pyramid_equal_to_cube_volume :
  (∃ h : ℝ, (5:ℝ)^3 = (1/3:ℝ) * (10:ℝ)^2 * h) ↔ h = 3.75 :=
by
  sorry

end height_of_pyramid_equal_to_cube_volume_l64_64426


namespace q_transformation_l64_64649

theorem q_transformation (w m z : ℝ) (q : ℝ) (h_q : q = 5 * w / (4 * m * z^2)) :
  let w' := 4 * w
  let m' := 2 * m
  let z' := 3 * z
  q = 5 * w / (4 * m * z^2) → (5 * w') / (4 * m' * (z'^2)) = (5 / 18) * q := by
  sorry

end q_transformation_l64_64649


namespace f_inv_f_inv_14_l64_64077

noncomputable def f (x : ℝ) : ℝ := 3 * x + 7

noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 3

theorem f_inv_f_inv_14 : f_inv (f_inv 14) = -14 / 9 :=
by {
  sorry
}

end f_inv_f_inv_14_l64_64077


namespace min_y_value_l64_64956

noncomputable def y (x : ℝ) : ℝ :=
  (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

theorem min_y_value : 
  ∃ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x' : ℝ, abs (x' - 5.92) < δ → abs (y x' - y 5.92) < ε) :=
sorry

end min_y_value_l64_64956


namespace sufficient_condition_parallel_planes_l64_64164

-- Definitions for lines and planes
variable {Line Plane : Type}
variable {m n : Line}
variable {α β : Plane}

-- Relations between lines and planes
variable (parallel_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Condition for sufficient condition for α parallel β
theorem sufficient_condition_parallel_planes
  (h1 : parallel_line m n)
  (h2 : perpendicular_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  parallel_plane α β :=
sorry

end sufficient_condition_parallel_planes_l64_64164


namespace total_shoes_tried_on_l64_64891

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l64_64891


namespace no_real_roots_of_quadratic_l64_64863

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem no_real_roots_of_quadratic (h : quadratic_discriminant 1 (-1) 1 < 0) :
  ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l64_64863


namespace confidence_level_for_relationship_l64_64600

-- Define the problem conditions and the target question.
def chi_squared_value : ℝ := 8.654
def critical_value : ℝ := 6.635
def confidence_level : ℝ := 99

theorem confidence_level_for_relationship (h : chi_squared_value > critical_value) : confidence_level = 99 :=
sorry

end confidence_level_for_relationship_l64_64600


namespace cubic_expression_l64_64624

theorem cubic_expression (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1008 :=
sorry

end cubic_expression_l64_64624


namespace total_tickets_sold_l64_64569

theorem total_tickets_sold
    (n₄₅ : ℕ) (n₆₀ : ℕ) (total_sales : ℝ) 
    (price₄₅ price₆₀ : ℝ)
    (h₁ : n₄₅ = 205)
    (h₂ : price₄₅ = 4.5)
    (h₃ : total_sales = 1972.5)
    (h₄ : price₆₀ = 6.0)
    (h₅ : total_sales = n₄₅ * price₄₅ + n₆₀ * price₆₀) :
    n₄₅ + n₆₀ = 380 := 
by
  sorry

end total_tickets_sold_l64_64569


namespace minimum_number_of_guests_l64_64413

theorem minimum_number_of_guests (total_food : ℝ) (max_food_per_guest : ℝ) (H₁ : total_food = 406) (H₂ : max_food_per_guest = 2.5) : 
  ∃ n : ℕ, (n : ℝ) ≥ 163 ∧ total_food / max_food_per_guest ≤ (n : ℝ) := 
by
  sorry

end minimum_number_of_guests_l64_64413


namespace part_I_part_II_l64_64017

theorem part_I (a b : ℝ) (h1 : 0 < a) (h2 : b * a = 2)
  (h3 : (1 + b) * a = 3) :
  (a = 1) ∧ (b = 2) :=
by {
  sorry
}

theorem part_II (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1 : ℝ) / x + 2 / y = 1)
  (k : ℝ) : 2 * x + y ≥ k^2 + k + 2 → (-3 ≤ k) ∧ (k ≤ 2) :=
by {
  sorry
}

end part_I_part_II_l64_64017


namespace product_largest_smallest_using_digits_l64_64057

theorem product_largest_smallest_using_digits (a b : ℕ) (h1 : 100 * 6 + 10 * 2 + 0 = a) (h2 : 100 * 2 + 10 * 0 + 6 = b) : a * b = 127720 := by
  -- The proof will go here
  sorry

end product_largest_smallest_using_digits_l64_64057


namespace correctness_of_statements_l64_64274

theorem correctness_of_statements 
  (A B C D : Prop)
  (h1 : A → B) (h2 : ¬(B → A))
  (h3 : C → B) (h4 : B → C)
  (h5 : D → C) (h6 : ¬(C → D)) : 
  (A → (C ∧ ¬(C → A))) ∧ (¬(A → D) ∧ ¬(D → A)) := 
by
  -- Proof will go here.
  sorry

end correctness_of_statements_l64_64274


namespace expected_digits_die_l64_64980

noncomputable def expected_number_of_digits (numbers : List ℕ) : ℚ :=
  let one_digit_numbers := numbers.filter (λ n => n < 10)
  let two_digit_numbers := numbers.filter (λ n => n >= 10)
  let p_one_digit := (one_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  let p_two_digit := (two_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  p_one_digit * 1 + p_two_digit * 2

theorem expected_digits_die :
  expected_number_of_digits [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = 1.5833 := 
by
  sorry

end expected_digits_die_l64_64980


namespace mean_of_remaining_two_l64_64896

def seven_numbers := [1865, 1990, 2015, 2023, 2105, 2120, 2135]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_of_remaining_two
  (h : mean (seven_numbers.take 5) = 2043) :
  mean (seven_numbers.drop 5) = 969 :=
by
  sorry

end mean_of_remaining_two_l64_64896


namespace largest_number_systematic_sampling_l64_64492

theorem largest_number_systematic_sampling (n k a1 a2: ℕ) (h1: n = 60) (h2: a1 = 3) (h3: a2 = 9) (h4: k = a2 - a1):
  ∃ largest, largest = a1 + k * (n / k - 1) := by
  sorry

end largest_number_systematic_sampling_l64_64492


namespace probability_at_least_one_deciphers_l64_64038

theorem probability_at_least_one_deciphers (P_A P_B : ℚ) (hA : P_A = 1/2) (hB : P_B = 1/3) :
    P_A + P_B - P_A * P_B = 2/3 := by
  sorry

end probability_at_least_one_deciphers_l64_64038


namespace largest_triangle_angle_l64_64018

-- Define the angles
def angle_sum := (105 : ℝ) -- Degrees
def delta_angle := (36 : ℝ) -- Degrees
def total_sum := (180 : ℝ) -- Degrees

-- Theorem statement
theorem largest_triangle_angle (a b c : ℝ) (h1 : a + b = angle_sum)
  (h2 : b = a + delta_angle) (h3 : a + b + c = total_sum) : c = 75 :=
sorry

end largest_triangle_angle_l64_64018


namespace system_of_equations_solution_l64_64298

theorem system_of_equations_solution (x y : ℤ) (h1 : 2 * x + 5 * y = 26) (h2 : 4 * x - 2 * y = 4) : 
    x = 3 ∧ y = 4 :=
by
  sorry

end system_of_equations_solution_l64_64298


namespace range_of_m_l64_64368

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), x^2 - 4 * x - 2 * m + 1 ≤ 0) ↔ m ∈ Set.Ici (3 : ℝ) := 
sorry

end range_of_m_l64_64368


namespace convex_over_real_l64_64399

def f (x : ℝ) : ℝ := x^4 - 2 * x^3 + 36 * x^2 - x + 7

theorem convex_over_real : ∀ x : ℝ, 0 ≤ (12 * x^2 - 12 * x + 72) :=
by sorry

end convex_over_real_l64_64399


namespace blueberries_in_each_blue_box_l64_64471

theorem blueberries_in_each_blue_box (S B : ℕ) (h1 : S - B = 12) (h2 : 2 * S = 76) : B = 26 := by
  sorry

end blueberries_in_each_blue_box_l64_64471


namespace find_constant_d_l64_64050

noncomputable def polynomial_g (d : ℝ) (x : ℝ) := d * x^4 + 17 * x^3 - 5 * d * x^2 + 45

theorem find_constant_d (d : ℝ) : polynomial_g d 5 = 0 → d = -4.34 :=
by
  sorry

end find_constant_d_l64_64050


namespace seashells_needed_to_reach_target_l64_64943

-- Definitions based on the conditions
def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

-- Statement to prove
theorem seashells_needed_to_reach_target : target_seashells - current_seashells = 6 :=
by
  sorry

end seashells_needed_to_reach_target_l64_64943


namespace winning_votes_cast_l64_64704

theorem winning_votes_cast (V : ℝ) (h1 : 0.40 * V = 280) : 0.70 * V = 490 :=
by
  sorry

end winning_votes_cast_l64_64704


namespace distance_between_A_and_B_l64_64635

theorem distance_between_A_and_B 
  (v1 v2: ℝ) (s: ℝ)
  (h1 : (s - 8) / v1 = s / v2)
  (h2 : s / (2 * v1) = (s - 15) / v2)
  (h3: s = 40) : 
  s = 40 := 
sorry

end distance_between_A_and_B_l64_64635


namespace van_distance_l64_64781

noncomputable def distance_covered (initial_time new_time speed : ℝ) : ℝ :=
  speed * new_time

theorem van_distance :
  distance_covered 5 (5 * (3 / 2)) 60 = 450 := 
by
  sorry

end van_distance_l64_64781


namespace max_value_expr_l64_64327

-- Define the expression
def expr (a b c d : ℝ) : ℝ :=
  a + b + c + d - a * b - b * c - c * d - d * a

-- The main theorem
theorem max_value_expr :
  (∀ (a b c d : ℝ), 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → expr a b c d ≤ 2) ∧
  (∃ (a b c d : ℝ), 0 ≤ a ∧ a = 1 ∧ 0 ≤ b ∧ b = 0 ∧ 0 ≤ c ∧ c = 1 ∧ 0 ≤ d ∧ d = 0 ∧ expr a b c d = 2) :=
  by
  sorry

end max_value_expr_l64_64327


namespace evaluate_expression_l64_64190

theorem evaluate_expression :
  (3 / 2) * ((8 / 3) * ((15 / 8) - (5 / 6))) / (((7 / 8) + (11 / 6)) / (13 / 4)) = 5 :=
by
  sorry

end evaluate_expression_l64_64190


namespace triangle_problem_l64_64778

noncomputable def triangle_sin_B (a b : ℝ) (A : ℝ) : ℝ :=
  b * Real.sin A / a

noncomputable def triangle_side_c (a b A : ℝ) : ℝ :=
  let discr := b^2 + a^2 - 2 * b * a * Real.cos A
  Real.sqrt discr

noncomputable def sin_diff_angle (sinB cosB sinC cosC : ℝ) : ℝ :=
  sinB * cosC - cosB * sinC

theorem triangle_problem
  (a b : ℝ)
  (A : ℝ)
  (ha : a = Real.sqrt 39)
  (hb : b = 2)
  (hA : A = Real.pi * (2 / 3)) :
  (triangle_sin_B a b A = Real.sqrt 13 / 13) ∧
  (triangle_side_c a b A = 5) ∧
  (sin_diff_angle (Real.sqrt 13 / 13) (2 * Real.sqrt 39 / 13) (5 * Real.sqrt 13 / 26) (3 * Real.sqrt 39 / 26) = -7 * Real.sqrt 3 / 26) :=
by sorry

end triangle_problem_l64_64778


namespace problem_3_div_27_l64_64063

theorem problem_3_div_27 (a b : ℕ) (h : 2^a = 8^(b + 1)) : 3^a / 27^b = 27 := by
  -- proof goes here
  sorry

end problem_3_div_27_l64_64063


namespace zeoland_speeding_fine_l64_64318

-- Define the conditions
def fine_per_mph (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ) : ℕ :=
  total_fine / (actual_speed - speed_limit)

-- Variables for the given problem
variables (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ)
variable (fine_per_mph_over_limit : ℕ)

-- Theorem statement
theorem zeoland_speeding_fine :
  total_fine = 256 ∧ speed_limit = 50 ∧ actual_speed = 66 →
  fine_per_mph total_fine actual_speed speed_limit = 16 :=
by
  sorry

end zeoland_speeding_fine_l64_64318


namespace frac_wx_l64_64087

theorem frac_wx (x y z w : ℚ) (h1 : x / y = 5) (h2 : y / z = 1 / 2) (h3 : z / w = 7) : w / x = 2 / 35 :=
by
  sorry

end frac_wx_l64_64087


namespace brown_beads_initial_l64_64381

theorem brown_beads_initial (B : ℕ) 
  (h1 : 1 = 1) -- There is 1 green bead in the container.
  (h2 : 3 = 3) -- There are 3 red beads in the container.
  (h3 : 4 = 4) -- Tom left 4 beads in the container.
  (h4 : 2 = 2) -- Tom took out 2 beads.
  (h5 : 6 = 2 + 4) -- Total initial beads before Tom took any out.
  : B = 2 := sorry

end brown_beads_initial_l64_64381


namespace pond_depth_l64_64986

theorem pond_depth (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) :
    V = L * W * D ↔ D = 5 := 
by
  rw [hL, hW, hV]
  constructor
  · intro h1
    linarith
  · intro h2
    rw [h2]
    linarith

#check pond_depth

end pond_depth_l64_64986


namespace adjusted_volume_bowling_ball_l64_64287

noncomputable def bowling_ball_diameter : ℝ := 40
noncomputable def hole1_diameter : ℝ := 5
noncomputable def hole1_depth : ℝ := 10
noncomputable def hole2_diameter : ℝ := 4
noncomputable def hole2_depth : ℝ := 12
noncomputable def expected_adjusted_volume : ℝ := 10556.17 * Real.pi

theorem adjusted_volume_bowling_ball :
  let radius := bowling_ball_diameter / 2
  let volume_ball := (4 / 3) * Real.pi * radius^3
  let hole1_radius := hole1_diameter / 2
  let hole1_volume := Real.pi * hole1_radius^2 * hole1_depth
  let hole2_radius := hole2_diameter / 2
  let hole2_volume := Real.pi * hole2_radius^2 * hole2_depth
  let adjusted_volume := volume_ball - hole1_volume - hole2_volume
  adjusted_volume = expected_adjusted_volume :=
by
  sorry

end adjusted_volume_bowling_ball_l64_64287


namespace max_value_of_f_l64_64747

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.cos x - 3

theorem max_value_of_f : ∀ x : ℝ, f x ≤ -1/2 :=
by
  sorry

end max_value_of_f_l64_64747


namespace Winnie_the_Pooh_guarantee_kilogram_l64_64188

noncomputable def guarantee_minimum_honey : Prop :=
  ∃ (a1 a2 a3 a4 a5 : ℝ), 
    a1 + a2 + a3 + a4 + a5 = 3 ∧
    min (min (a1 + a2) (a2 + a3)) (min (a3 + a4) (a4 + a5)) ≥ 1

theorem Winnie_the_Pooh_guarantee_kilogram :
  guarantee_minimum_honey :=
sorry

end Winnie_the_Pooh_guarantee_kilogram_l64_64188


namespace minimal_team_members_l64_64800

theorem minimal_team_members (n : ℕ) : 
  (n ≡ 1 [MOD 6]) ∧ (n ≡ 2 [MOD 8]) ∧ (n ≡ 3 [MOD 9]) → n = 343 := 
by
  sorry

end minimal_team_members_l64_64800


namespace total_pens_bought_l64_64280

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l64_64280


namespace wholesale_price_l64_64060

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end wholesale_price_l64_64060


namespace cheryl_material_need_l64_64623

-- Cheryl's conditions
def cheryl_material_used (x : ℚ) : Prop :=
  x + 2/3 - 4/9 = 2/3

-- The proof problem statement
theorem cheryl_material_need : ∃ x : ℚ, cheryl_material_used x ∧ x = 4/9 :=
  sorry

end cheryl_material_need_l64_64623


namespace rectangle_area_l64_64231

theorem rectangle_area (a b : ℝ) (h : 2 * a^2 - 11 * a + 5 = 0) (hb : 2 * b^2 - 11 * b + 5 = 0) : a * b = 5 / 2 :=
sorry

end rectangle_area_l64_64231


namespace coin_outcomes_equivalent_l64_64648

theorem coin_outcomes_equivalent :
  let outcomes_per_coin := 2
  let total_coins := 3
  (outcomes_per_coin ^ total_coins) = 8 :=
by
  sorry

end coin_outcomes_equivalent_l64_64648


namespace factorization_problem_1_factorization_problem_2_l64_64621

-- Problem 1: Factorize 2(m-n)^2 - m(n-m) and show it equals (n-m)(2n - 3m)
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) :=
by
  sorry

-- Problem 2: Factorize -4xy^2 + 4x^2y + y^3 and show it equals y(2x - y)^2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 :=
by
  sorry

end factorization_problem_1_factorization_problem_2_l64_64621


namespace solution_set_of_inequality_l64_64568

theorem solution_set_of_inequality (x : ℝ) : (x^2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := 
by 
  sorry

end solution_set_of_inequality_l64_64568


namespace crumbs_triangle_area_l64_64123

theorem crumbs_triangle_area :
  ∀ (table_length table_width : ℝ) (crumbs : ℕ),
    table_length = 2 ∧ table_width = 1 ∧ crumbs = 500 →
    ∃ (triangle_area : ℝ), (triangle_area < 0.005 ∧ ∃ (a b c : Type), a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by
  sorry

end crumbs_triangle_area_l64_64123


namespace problem_min_x_plus_2y_l64_64215

theorem problem_min_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) : 
  x + 2 * y ≥ -2 * Real.sqrt 2 - 1 :=
sorry

end problem_min_x_plus_2y_l64_64215


namespace factory_workers_total_payroll_l64_64022

theorem factory_workers_total_payroll (total_office_payroll : ℝ) (number_factory_workers : ℝ) 
(number_office_workers : ℝ) (salary_difference : ℝ) 
(average_office_salary : ℝ) (average_factory_salary : ℝ) 
(h1 : total_office_payroll = 75000) (h2 : number_factory_workers = 15)
(h3 : number_office_workers = 30) (h4 : salary_difference = 500)
(h5 : average_office_salary = total_office_payroll / number_office_workers)
(h6 : average_office_salary = average_factory_salary + salary_difference) :
  number_factory_workers * average_factory_salary = 30000 :=
by
  sorry

end factory_workers_total_payroll_l64_64022


namespace comparison_of_negatives_l64_64416

theorem comparison_of_negatives : -2 < - (3 / 2) :=
by
  sorry

end comparison_of_negatives_l64_64416


namespace min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l64_64000

-- Definitions for the problem conditions
def initial_points : ℕ := 52
def record_points : ℕ := 89
def max_shots : ℕ := 10
def points_range : Finset ℕ := Finset.range 11 \ {0}

-- Lean statement for the first question
theorem min_score_seventh_shot_to_break_record (x₇ : ℕ) (h₁: x₇ ∈ points_range) :
  initial_points + x₇ + 30 > record_points ↔ x₇ ≥ 8 :=
by sorry

-- Lean statement for the second question
theorem shots_hitting_10_to_break_record_when_7th_shot_is_8 (x₈ x₉ x₁₀ : ℕ)
  (h₂ : 8 ∈ points_range) 
  (h₃ : x₈ ∈ points_range) (h₄ : x₉ ∈ points_range) (h₅ : x₁₀ ∈ points_range) :
  initial_points + 8 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∧ x₉ = 10 ∧ x₁₀ = 10) :=
by sorry

-- Lean statement for the third question
theorem necessary_shot_of_10_when_7th_shot_is_10 (x₈ x₉ x₁₀ : ℕ)
  (h₆ : 10 ∈ points_range)
  (h₇ : x₈ ∈ points_range) (h₈ : x₉ ∈ points_range) (h₉ : x₁₀ ∈ points_range) :
  initial_points + 10 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∨ x₉ = 10 ∨ x₁₀ = 10) :=
by sorry

end min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l64_64000


namespace bob_mean_score_l64_64698

-- Conditions
def scores : List ℝ := [68, 72, 76, 80, 85, 90]
def alice_scores (a1 a2 a3 : ℝ) : Prop := a1 < a2 ∧ a2 < a3 ∧ a1 + a2 + a3 = 225
def bob_scores (b1 b2 b3 : ℝ) : Prop := b1 + b2 + b3 = 246

-- Theorem statement proving Bob's mean score
theorem bob_mean_score (a1 a2 a3 b1 b2 b3 : ℝ) (h1 : a1 ∈ scores) (h2 : a2 ∈ scores) (h3 : a3 ∈ scores)
  (h4 : b1 ∈ scores) (h5 : b2 ∈ scores) (h6 : b3 ∈ scores)
  (h7 : alice_scores a1 a2 a3)
  (h8 : bob_scores b1 b2 b3)
  (h9 : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3)
  : (b1 + b2 + b3) / 3 = 82 :=
sorry

end bob_mean_score_l64_64698


namespace complete_square_form_l64_64640

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 10 * x + 15

theorem complete_square_form (b c : ℤ) (h : ∀ x : ℝ, quadratic_expr x = 0 ↔ (x + b)^2 = c) :
  b + c = 5 :=
sorry

end complete_square_form_l64_64640


namespace evaluate_neg2012_l64_64877

def func (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_neg2012 (a b c : ℝ) (h : func a b c 2012 = 3) : func a b c (-2012) = -1 :=
by
  sorry

end evaluate_neg2012_l64_64877


namespace jade_handled_80_transactions_l64_64902

variable (mabel anthony cal jade : ℕ)

-- Conditions
def mabel_transactions : mabel = 90 :=
by sorry

def anthony_transactions : anthony = mabel + (10 * mabel / 100) :=
by sorry

def cal_transactions : cal = 2 * anthony / 3 :=
by sorry

def jade_transactions : jade = cal + 14 :=
by sorry

-- Proof problem
theorem jade_handled_80_transactions :
  mabel = 90 →
  anthony = mabel + (10 * mabel / 100) →
  cal = 2 * anthony / 3 →
  jade = cal + 14 →
  jade = 80 :=
by
  intros
  subst_vars
  -- The proof steps would normally go here, but we leave it with sorry.
  sorry

end jade_handled_80_transactions_l64_64902


namespace stans_average_speed_l64_64908

noncomputable def average_speed (distance1 distance2 distance3 : ℝ) (time1_hrs time1_mins time2 time3_hrs time3_mins : ℝ) : ℝ :=
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1_hrs + time1_mins / 60 + time2 + time3_hrs + time3_mins / 60
  total_distance / total_time

theorem stans_average_speed  :
  average_speed 350 420 330 5 40 7 5 30 = 60.54 :=
by
  -- sorry block indicates missing proof
  sorry

end stans_average_speed_l64_64908


namespace sixth_graders_more_than_seventh_l64_64982

theorem sixth_graders_more_than_seventh (c_pencil : ℕ) (h_cents : c_pencil > 0)
    (h_cond : ∀ n : ℕ, n * c_pencil = 221 ∨ n * c_pencil = 286)
    (h_sixth_graders : 35 > 0) :
    ∃ n6 n7 : ℕ, n6 > n7 ∧ n6 - n7 = 5 :=
by
  sorry

end sixth_graders_more_than_seventh_l64_64982


namespace necessary_and_sufficient_condition_l64_64273

theorem necessary_and_sufficient_condition (a b : ℝ) : a^2 * b > a * b^2 ↔ 1/a < 1/b := 
sorry

end necessary_and_sufficient_condition_l64_64273


namespace base8_subtraction_correct_l64_64547

def base8_sub (a b : Nat) : Nat := sorry  -- function to perform base 8 subtraction

theorem base8_subtraction_correct :
  base8_sub 0o126 0o45 = 0o41 := sorry

end base8_subtraction_correct_l64_64547


namespace evaluate_expression_l64_64462

theorem evaluate_expression :
  abs ((4^2 - 8 * (3^2 - 12))^2) - abs (Real.sin (5 * Real.pi / 6) - Real.cos (11 * Real.pi / 3)) = 1600 :=
by
  sorry

end evaluate_expression_l64_64462


namespace sally_nickels_count_l64_64999

theorem sally_nickels_count (original_nickels dad_nickels mom_nickels : ℕ) 
    (h1: original_nickels = 7) 
    (h2: dad_nickels = 9) 
    (h3: mom_nickels = 2) 
    : original_nickels + dad_nickels + mom_nickels = 18 :=
by
  sorry

end sally_nickels_count_l64_64999


namespace new_person_weight_l64_64074

theorem new_person_weight (W x : ℝ) (h1 : (W - 55 + x) / 8 = (W / 8) + 2.5) : x = 75 := by
  -- Proof omitted
  sorry

end new_person_weight_l64_64074


namespace set_contains_one_implies_values_l64_64204

theorem set_contains_one_implies_values (x : ℝ) (A : Set ℝ) (hA : A = {x, x^2}) (h1 : 1 ∈ A) : x = 1 ∨ x = -1 := by
  sorry

end set_contains_one_implies_values_l64_64204


namespace numeral_eq_7000_l64_64591

theorem numeral_eq_7000 
  (local_value face_value numeral : ℕ)
  (h1 : face_value = 7)
  (h2 : local_value - face_value = 6993) : 
  numeral = 7000 :=
by
  sorry

end numeral_eq_7000_l64_64591


namespace cubic_identity_l64_64718

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := 
by
  sorry

end cubic_identity_l64_64718


namespace fraction_calculation_l64_64732

theorem fraction_calculation :
  ( (12^4 + 324) * (26^4 + 324) * (38^4 + 324) * (50^4 + 324) * (62^4 + 324)) /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) =
  73.481 :=
by
  sorry

end fraction_calculation_l64_64732


namespace optimal_strategy_l64_64376

-- Define the conditions
def valid_N (N : ℤ) : Prop :=
  0 ≤ N ∧ N ≤ 20

def score (N : ℤ) (other_teams_count : ℤ) : ℤ :=
  if other_teams_count > N then N else 0

-- The mathematical problem statement
theorem optimal_strategy : ∃ N : ℤ, valid_N N ∧ (∀ other_teams_count : ℤ, score 1 other_teams_count ≥ score N other_teams_count ∧ score 1 other_teams_count ≠ 0) :=
sorry

end optimal_strategy_l64_64376


namespace max_students_per_class_l64_64072

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l64_64072


namespace a2018_is_4035_l64_64996

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℝ := sorry

axiom domain : ∀ x : ℝ, true 
axiom condition_2 : ∀ x : ℝ, x < 0 → f x > 1
axiom condition_3 : ∀ x y : ℝ, f x * f y = f (x + y)
axiom sequence_def : ∀ n : ℕ, n > 0 → a 1 = f 0 ∧ f (a (n + 1)) = 1 / f (-2 - a n)

theorem a2018_is_4035 : a 2018 = 4035 :=
sorry

end a2018_is_4035_l64_64996


namespace congruence_equiv_l64_64806

theorem congruence_equiv (x : ℤ) (h : 5 * x + 9 ≡ 3 [ZMOD 18]) : 3 * x + 14 ≡ 14 [ZMOD 18] :=
sorry

end congruence_equiv_l64_64806


namespace square_side_length_l64_64275

theorem square_side_length (x y : ℕ) (h_gcd : Nat.gcd x y = 5) (h_area : ∃ a : ℝ, a^2 = (169 / 6) * ↑(Nat.lcm x y)) : ∃ a : ℝ, a = 65 * Real.sqrt 2 :=
by
  sorry

end square_side_length_l64_64275


namespace friends_division_l64_64721

def num_ways_to_divide (total_friends teams : ℕ) : ℕ :=
  4^8 - (Nat.choose 4 1) * 3^8 + (Nat.choose 4 2) * 2^8 - (Nat.choose 4 3) * 1^8

theorem friends_division (total_friends teams : ℕ) (h_friends : total_friends = 8) (h_teams : teams = 4) :
  num_ways_to_divide total_friends teams = 39824 := by
  sorry

end friends_division_l64_64721


namespace triangle_angles_l64_64821

theorem triangle_angles
  (h_a a h_b b : ℝ)
  (h_a_ge_a : h_a ≥ a)
  (h_b_ge_b : h_b ≥ b)
  (a_ge_h_b : a ≥ h_b)
  (b_ge_h_a : b ≥ h_a) : 
  a = b ∧ 
  (a = h_a ∧ b = h_b) → 
  ∃ A B C : ℝ, Set.toFinset ({A, B, C} : Set ℝ) = {90, 45, 45} := 
by 
  sorry

end triangle_angles_l64_64821


namespace largest_square_side_length_largest_rectangle_dimensions_l64_64200

variable (a b : ℝ)

-- Part a
theorem largest_square_side_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ s : ℝ, s = (a * b) / (a + b) :=
sorry

-- Part b
theorem largest_rectangle_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, (x = a / 2 ∧ y = b / 2) :=
sorry

end largest_square_side_length_largest_rectangle_dimensions_l64_64200


namespace original_denominator_is_15_l64_64313

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end original_denominator_is_15_l64_64313


namespace arithmetic_sequence_a2_a9_l64_64378

theorem arithmetic_sequence_a2_a9 (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 5 + a 6 = 12) :
  a 2 + a 9 = 12 :=
sorry

end arithmetic_sequence_a2_a9_l64_64378


namespace find_p_l64_64869

theorem find_p (p : ℚ) : (∀ x : ℚ, (3 * x + 4) = 0 → (4 * x ^ 3 + p * x ^ 2 + 17 * x + 24) = 0) → p = 13 / 4 :=
by
  sorry

end find_p_l64_64869


namespace line_passes_fixed_point_l64_64759

theorem line_passes_fixed_point (k : ℝ) :
    ((k + 1) * -1) - ((2 * k - 1) * 1) + 3 * k = 0 :=
by
    -- The proof is omitted as the primary aim is to ensure the correct Lean statement.
    sorry

end line_passes_fixed_point_l64_64759


namespace y_x_cubed_monotonic_increasing_l64_64923

theorem y_x_cubed_monotonic_increasing : 
  ∀ x1 x2 : ℝ, (x1 ≤ x2) → (x1^3 ≤ x2^3) :=
by
  intros x1 x2 h
  sorry

end y_x_cubed_monotonic_increasing_l64_64923


namespace sad_children_count_l64_64302

-- Definitions of conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 18
def girls : ℕ := 42
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- Calculate the number of children who are either happy or sad
def happy_or_sad_children : ℕ := total_children - neither_happy_nor_sad_children

-- Prove that the number of sad children is 10
theorem sad_children_count : happy_or_sad_children - happy_children = 10 := by
  sorry

end sad_children_count_l64_64302


namespace repeatingDecimal_exceeds_l64_64845

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l64_64845


namespace largest_eight_digit_number_contains_even_digits_l64_64955

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l64_64955


namespace max_value_of_a_b_c_l64_64851

theorem max_value_of_a_b_c (a b c : ℤ) (h1 : a + b = 2006) (h2 : c - a = 2005) (h3 : a < b) : 
  a + b + c = 5013 :=
sorry

end max_value_of_a_b_c_l64_64851


namespace conference_games_scheduled_l64_64249

theorem conference_games_scheduled
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_games_per_pair : ℕ)
  (inter_games_per_pair : ℕ)
  (h_div : divisions = 3)
  (h_teams : teams_per_division = 4)
  (h_intra : intra_games_per_pair = 3)
  (h_inter : inter_games_per_pair = 2) :
  let intra_division_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_games_per_pair
  let intra_division_total := intra_division_games * divisions
  let inter_division_games := teams_per_division * (teams_per_division * (divisions - 1)) * inter_games_per_pair
  let inter_division_total := inter_division_games * divisions / 2
  let total_games := intra_division_total + inter_division_total
  total_games = 150 :=
by
  sorry

end conference_games_scheduled_l64_64249


namespace find_sum_of_abcd_l64_64411

theorem find_sum_of_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) :
  a + b + c + d = -26 / 3 :=
sorry

end find_sum_of_abcd_l64_64411


namespace initial_roses_count_l64_64023

theorem initial_roses_count 
  (roses_to_mother : ℕ)
  (roses_to_grandmother : ℕ)
  (roses_to_sister : ℕ)
  (roses_kept : ℕ)
  (initial_roses : ℕ)
  (h_mother : roses_to_mother = 6)
  (h_grandmother : roses_to_grandmother = 9)
  (h_sister : roses_to_sister = 4)
  (h_kept : roses_kept = 1)
  (h_initial : initial_roses = roses_to_mother + roses_to_grandmother + roses_to_sister + roses_kept) :
  initial_roses = 20 :=
by
  rw [h_mother, h_grandmother, h_sister, h_kept] at h_initial
  exact h_initial

end initial_roses_count_l64_64023


namespace min_x_plus_9y_l64_64944

variable {x y : ℝ}

theorem min_x_plus_9y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / y = 1) : x + 9 * y ≥ 16 :=
  sorry

end min_x_plus_9y_l64_64944


namespace find_expression_value_l64_64672

theorem find_expression_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5 * x^6 + x^2 = 8436*x - 338 := 
by {
  sorry
}

end find_expression_value_l64_64672


namespace min_value_of_x_prime_factors_l64_64560

theorem min_value_of_x_prime_factors (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
    (h : 5 * x^7 = 13 * y^11)
    (hx_factorization : x = a^c * b^d) : a + b + c + d = 32 := sorry

end min_value_of_x_prime_factors_l64_64560


namespace equivalent_expression_l64_64750

theorem equivalent_expression (x : ℝ) (hx : x > 0) : (x^2 * x^(1/4))^(1/3) = x^(3/4) := 
  sorry

end equivalent_expression_l64_64750


namespace max_ab_value_l64_64095

variable (a b c : ℝ)

-- Conditions
axiom h1 : 0 < a ∧ a < 1
axiom h2 : 0 < b ∧ b < 1
axiom h3 : 0 < c ∧ c < 1
axiom h4 : 3 * a + 2 * b = 1

-- Goal
theorem max_ab_value : ab = 1 / 24 :=
by
  sorry

end max_ab_value_l64_64095


namespace car_miles_per_tankful_in_city_l64_64767

-- Define constants for the given values
def miles_per_tank_on_highway : ℝ := 462
def fewer_miles_per_gallon : ℝ := 15
def miles_per_gallon_in_city : ℝ := 40

-- Prove the car traveled 336 miles per tankful in the city
theorem car_miles_per_tankful_in_city :
  (miles_per_tank_on_highway / (miles_per_gallon_in_city + fewer_miles_per_gallon)) * miles_per_gallon_in_city = 336 := 
by
  sorry

end car_miles_per_tankful_in_city_l64_64767


namespace proof_rewritten_eq_and_sum_l64_64104

-- Define the given equation
def given_eq (x : ℝ) : Prop := 64 * x^2 + 80 * x - 72 = 0

-- Define the rewritten form of the equation
def rewritten_eq (x : ℝ) : Prop := (8 * x + 5)^2 = 97

-- Define the correctness of rewriting the equation
def correct_rewrite (x : ℝ) : Prop :=
  given_eq x → rewritten_eq x

-- Define the correct value of a + b + c
def correct_sum : Prop :=
  8 + 5 + 97 = 110

-- The final theorem statement
theorem proof_rewritten_eq_and_sum (x : ℝ) : correct_rewrite x ∧ correct_sum :=
by
  sorry

end proof_rewritten_eq_and_sum_l64_64104


namespace parabola_vertex_y_axis_opens_upwards_l64_64966

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end parabola_vertex_y_axis_opens_upwards_l64_64966


namespace part_I_part_II_l64_64323

-- Define the function f
def f (x: ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- The conditions and questions transformed into Lean statements
theorem part_I : ∃ m, (∀ x: ℝ, f x ≤ m) ∧ (m = f (-1)) ∧ (m = 2) := by
  sorry

theorem part_II (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : a^2 + 3 * b^2 + 2 * c^2 = 2) : 
  ∃ n, (∀ a b c : ℝ, (0 < a ∧ 0 < b ∧ 0 < c) ∧ (a^2 + 3 * b^2 + 2 * c^2 = 2) → ab + 2 * bc ≤ n) ∧ (n = 1) := by
  sorry

end part_I_part_II_l64_64323


namespace orange_ring_weight_correct_l64_64824

-- Define the weights as constants
def purple_ring_weight := 0.3333333333333333
def white_ring_weight := 0.4166666666666667
def total_weight := 0.8333333333
def orange_ring_weight := 0.0833333333

-- Theorem statement
theorem orange_ring_weight_correct :
  total_weight - purple_ring_weight - white_ring_weight = orange_ring_weight :=
by
  -- Sorry is added to skip the proof part as per the instruction
  sorry

end orange_ring_weight_correct_l64_64824


namespace arithmetic_prog_includes_1999_l64_64407

-- Definitions based on problem conditions
def is_in_arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_prog_includes_1999
  (d : ℕ) (h_pos : d > 0) 
  (h_includes7 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 7)
  (h_includes15 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 15)
  (h_includes27 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 27) :
  ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 1999 := 
sorry

end arithmetic_prog_includes_1999_l64_64407


namespace cost_of_450_chocolates_l64_64047

theorem cost_of_450_chocolates :
  ∀ (cost_per_box : ℝ) (candies_per_box total_candies : ℕ),
  cost_per_box = 7.50 →
  candies_per_box = 30 →
  total_candies = 450 →
  (total_candies / candies_per_box : ℝ) * cost_per_box = 112.50 :=
by
  intros cost_per_box candies_per_box total_candies h1 h2 h3
  sorry

end cost_of_450_chocolates_l64_64047


namespace area_of_triangles_l64_64137

theorem area_of_triangles
  (ABC_area : ℝ)
  (AD : ℝ)
  (DB : ℝ)
  (h_AD_DB : AD + DB = 7)
  (h_equal_areas : ABC_area = 12) :
  (∃ ABE_area : ℝ, ABE_area = 36 / 7) ∧ (∃ DBF_area : ℝ, DBF_area = 36 / 7) :=
by
  sorry

end area_of_triangles_l64_64137


namespace fraction_of_total_calls_l64_64853

-- Definitions based on conditions
variable (B : ℚ) -- Calls processed by each member of Team B
variable (N : ℚ) -- Number of members in Team B

-- The fraction of calls processed by each member of Team A
def team_A_call_fraction : ℚ := 1 / 5

-- The fraction of calls processed by each member of Team C
def team_C_call_fraction : ℚ := 7 / 8

-- The fraction of agents in Team A relative to Team B
def team_A_agents_fraction : ℚ := 5 / 8

-- The fraction of agents in Team C relative to Team B
def team_C_agents_fraction : ℚ := 3 / 4

-- Total calls processed by Team A, Team B, and Team C
def total_calls_team_A : ℚ := (B * team_A_call_fraction) * (N * team_A_agents_fraction)
def total_calls_team_B : ℚ := B * N
def total_calls_team_C : ℚ := (B * team_C_call_fraction) * (N * team_C_agents_fraction)

-- Sum of total calls processed by all teams
def total_calls_all_teams : ℚ := total_calls_team_A B N + total_calls_team_B B N + total_calls_team_C B N

-- Potential total calls if all teams were as efficient as Team B
def potential_total_calls : ℚ := 3 * (B * N)

-- Fraction of total calls processed by all teams combined
def processed_fraction : ℚ := total_calls_all_teams B N / potential_total_calls B N

theorem fraction_of_total_calls : processed_fraction B N = 19 / 32 :=
by
  sorry -- Proof omitted

end fraction_of_total_calls_l64_64853


namespace num_possible_lists_l64_64115

theorem num_possible_lists :
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  total_lists = 40 := by
{
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  show total_lists = 40
  exact rfl
}

end num_possible_lists_l64_64115


namespace solve_for_n_l64_64410

theorem solve_for_n (n : ℕ) (h : 2 * n - 5 = 1) : n = 3 :=
by
  sorry

end solve_for_n_l64_64410


namespace largest_digit_divisible_by_4_l64_64290

theorem largest_digit_divisible_by_4 :
  ∃ (A : ℕ), A ≤ 9 ∧ (∃ n : ℕ, 100000 * 4 + 10000 * A + 67994 = n * 4) ∧ 
  (∀ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (∃ m : ℕ, 100000 * 4 + 10000 * B + 67994 = m * 4) → B ≤ A) :=
sorry

end largest_digit_divisible_by_4_l64_64290


namespace sophomores_selected_correct_l64_64443

-- Define the number of students in each grade and the total spots for the event
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300
def totalSpots : ℕ := 40

-- Calculate the total number of students
def totalStudents : ℕ := freshmen + sophomores + juniors

-- The correct answer we want to prove
def numberOfSophomoresSelected : ℕ := (sophomores * totalSpots) / totalStudents

-- Statement to be proved
theorem sophomores_selected_correct : numberOfSophomoresSelected = 26 := by
  -- Proof is omitted
  sorry

end sophomores_selected_correct_l64_64443


namespace problem_sol_52_l64_64797

theorem problem_sol_52 
  (x y: ℝ)
  (h1: x + y = 7)
  (h2: 4 * x * y = 7)
  (a b c d : ℕ)
  (hx_form : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hd_pos : 0 < d)
  : a + b + c + d = 52 := sorry

end problem_sol_52_l64_64797


namespace find_special_n_l64_64734

open Nat

def is_divisor (d n : ℕ) : Prop := n % d = 0

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def special_primes_condition (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem find_special_n (n : ℕ) (h : n > 1) :
  special_primes_condition n → n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end find_special_n_l64_64734


namespace common_ratio_geometric_series_l64_64594

-- Define the first three terms of the series
def first_term := (-3: ℚ) / 5
def second_term := (-5: ℚ) / 3
def third_term := (-125: ℚ) / 27

-- Prove that the common ratio = 25/9
theorem common_ratio_geometric_series :
  (second_term / first_term) = (25 : ℚ) / 9 :=
by
  sorry

end common_ratio_geometric_series_l64_64594


namespace find_b_compare_f_l64_64387

-- Definition from conditions
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Prove that b = 4
theorem find_b (b c : ℝ) (h : ∀ x : ℝ, f (2 + x) b c = f (2 - x) b c) : b = 4 :=
sorry

-- Part 2: Prove the comparison of f(\frac{5}{4}) and f(-a^2 - a + 1)
theorem compare_f (c : ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (2 + x) 4 c = f (2 - x) 4 c) (h₂ : f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c) :
f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c := 
sorry

end find_b_compare_f_l64_64387


namespace remaining_leaves_l64_64441

def initial_leaves := 1000
def first_week_shed := (2 / 5 : ℚ) * initial_leaves
def leaves_after_first_week := initial_leaves - first_week_shed
def second_week_shed := (40 / 100 : ℚ) * leaves_after_first_week
def leaves_after_second_week := leaves_after_first_week - second_week_shed
def third_week_shed := (3 / 4 : ℚ) * second_week_shed
def leaves_after_third_week := leaves_after_second_week - third_week_shed

theorem remaining_leaves (initial_leaves first_week_shed leaves_after_first_week second_week_shed leaves_after_second_week third_week_shed leaves_after_third_week: ℚ) : 
  leaves_after_third_week = 180 := by
  sorry

end remaining_leaves_l64_64441


namespace find_Natisfy_condition_l64_64460

-- Define the original number
def N : Nat := 2173913043478260869565

-- Define the function to move the first digit of a number to the end
def move_first_digit_to_end (n : Nat) : Nat := sorry

-- The proof statement
theorem find_Natisfy_condition : 
  let new_num1 := N * 4
  let new_num2 := new_num1 / 5
  move_first_digit_to_end N = new_num2 
:=
  sorry

end find_Natisfy_condition_l64_64460


namespace digit_b_divisible_by_5_l64_64118

theorem digit_b_divisible_by_5 (B : ℕ) (h : B = 0 ∨ B = 5) : 
  (∃ n : ℕ, (947 * 10 + B) = 5 * n) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_b_divisible_by_5_l64_64118


namespace union_of_sets_l64_64435

def A : Set ℤ := {0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets :
  A ∪ B = {0, 1, 2} :=
by
  sorry

end union_of_sets_l64_64435


namespace problem_A_eq_7_problem_A_eq_2012_l64_64224

open Nat

-- Problem statement for A = 7
theorem problem_A_eq_7 (n k : ℕ) :
  (n! + 7 * n = n^k) ↔ ((n, k) = (2, 4) ∨ (n, k) = (3, 3)) :=
sorry

-- Problem statement for A = 2012
theorem problem_A_eq_2012 (n k : ℕ) :
  ¬ (n! + 2012 * n = n^k) :=
sorry

end problem_A_eq_7_problem_A_eq_2012_l64_64224


namespace find_f_neg_half_l64_64233

def is_odd_function {α β : Type*} [AddGroup α] [Neg β] (f : α → β) : Prop :=
  ∀ x : α, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 0

theorem find_f_neg_half (f_odd : is_odd_function f) (f_pos : ∀ x > 0, f x = Real.log x / Real.log 2) :
  f (-1/2) = 1 := by
  sorry

end find_f_neg_half_l64_64233


namespace no_x_axis_intersection_iff_l64_64371

theorem no_x_axis_intersection_iff (m : ℝ) :
    (∀ x : ℝ, x^2 - x + m ≠ 0) ↔ m > 1 / 4 :=
by
  sorry

end no_x_axis_intersection_iff_l64_64371


namespace frac_plus_a_ge_seven_l64_64235

theorem frac_plus_a_ge_seven (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := 
by
  sorry

end frac_plus_a_ge_seven_l64_64235


namespace find_k_l64_64184

theorem find_k (n m : ℕ) (hn : n > 0) (hm : m > 0) (h : (1 : ℚ) / n^2 + 1 / m^2 = k / (n^2 + m^2)) : k = 4 :=
sorry

end find_k_l64_64184


namespace power_mod_lemma_l64_64585

theorem power_mod_lemma : (7^137 % 13) = 11 := by
  sorry

end power_mod_lemma_l64_64585


namespace intersection_A_B_l64_64858

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_A_B : A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l64_64858


namespace final_reduced_price_l64_64472

noncomputable def original_price (P : ℝ) (Q : ℝ) : ℝ := 800 / Q

noncomputable def price_after_first_week (P : ℝ) : ℝ := 0.90 * P
noncomputable def price_after_second_week (price1 : ℝ) : ℝ := 0.85 * price1
noncomputable def price_after_third_week (price2 : ℝ) : ℝ := 0.80 * price2

noncomputable def reduced_price (P : ℝ) : ℝ :=
  let price1 := price_after_first_week P
  let price2 := price_after_second_week price1
  price_after_third_week price2

theorem final_reduced_price :
  ∃ P Q : ℝ, 
    800 = Q * P ∧
    800 = (Q + 5) * reduced_price P ∧
    abs (reduced_price P - 62.06) < 0.01 :=
by
  sorry

end final_reduced_price_l64_64472


namespace binary_to_decimal_11011_is_27_l64_64822

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l64_64822


namespace highest_a_value_l64_64098

theorem highest_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a = 23 :=
sorry

end highest_a_value_l64_64098


namespace possible_box_dimensions_l64_64528

-- Define the initial conditions
def edge_length_original_box := 4
def edge_length_dice := 1
def total_cubes := (edge_length_original_box * edge_length_original_box * edge_length_original_box)

-- Prove that these are the possible dimensions of boxes with square bases that fit all the dice
theorem possible_box_dimensions :
  ∃ (len1 len2 len3 : ℕ), 
  total_cubes = (len1 * len2 * len3) ∧ 
  (len1 = len2) ∧ 
  ((len1, len2, len3) = (1, 1, 64) ∨ (len1, len2, len3) = (2, 2, 16) ∨ (len1, len2, len3) = (4, 4, 4) ∨ (len1, len2, len3) = (8, 8, 1)) :=
by {
  sorry -- The proof would be placed here
}

end possible_box_dimensions_l64_64528


namespace prod_of_consecutive_nums_divisible_by_504_l64_64111

theorem prod_of_consecutive_nums_divisible_by_504
  (a : ℕ)
  (h : ∃ b : ℕ, a = b ^ 3) :
  (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := 
sorry

end prod_of_consecutive_nums_divisible_by_504_l64_64111


namespace average_age_of_guardians_and_fourth_graders_l64_64028

theorem average_age_of_guardians_and_fourth_graders (num_fourth_graders num_guardians : ℕ)
  (avg_age_fourth_graders avg_age_guardians : ℕ)
  (h1 : num_fourth_graders = 40)
  (h2 : avg_age_fourth_graders = 10)
  (h3 : num_guardians = 60)
  (h4 : avg_age_guardians = 35)
  : (num_fourth_graders * avg_age_fourth_graders + num_guardians * avg_age_guardians) / (num_fourth_graders + num_guardians) = 25 :=
by
  sorry

end average_age_of_guardians_and_fourth_graders_l64_64028


namespace profit_with_discount_l64_64761

theorem profit_with_discount (CP SP_with_discount SP_no_discount : ℝ) (discount profit_no_discount : ℝ) (H1 : discount = 0.1) (H2 : profit_no_discount = 0.3889) (H3 : SP_no_discount = CP * (1 + profit_no_discount)) (H4 : SP_with_discount = SP_no_discount * (1 - discount)) : (SP_with_discount - CP) / CP * 100 = 25 :=
by
  -- The proof will be filled here
  sorry

end profit_with_discount_l64_64761


namespace mul_mod_l64_64284

theorem mul_mod (n1 n2 n3 : ℤ) (h1 : n1 = 2011) (h2 : n2 = 1537) (h3 : n3 = 450) : 
  (2011 * 1537) % 450 = 307 := by
  sorry

end mul_mod_l64_64284


namespace probability_z_l64_64912

variable (p q x y z : ℝ)

-- Conditions
def condition1 : Prop := z = p * y + q * x
def condition2 : Prop := x = p + q * x^2
def condition3 : Prop := y = q + p * y^2
def condition4 : Prop := x ≠ y

-- Theorem Statement
theorem probability_z : condition1 p q x y z ∧ condition2 p q x ∧ condition3 p q y ∧ condition4 x y → z = 2 * q := by
  sorry

end probability_z_l64_64912


namespace coed_softball_team_total_players_l64_64769

theorem coed_softball_team_total_players (M W : ℕ) 
  (h1 : W = M + 4) 
  (h2 : (M : ℚ) / W = 0.6363636363636364) :
  M + W = 18 := 
by sorry

end coed_softball_team_total_players_l64_64769


namespace jellybeans_final_count_l64_64609

-- Defining the initial number of jellybeans and operations
def initial_jellybeans : ℕ := 37
def removed_first : ℕ := 15
def added_back : ℕ := 5
def removed_second : ℕ := 4

-- Defining the final number of jellybeans to prove it equals 23
def final_jellybeans : ℕ := (initial_jellybeans - removed_first) + added_back - removed_second

-- The theorem that states the final number of jellybeans is 23
theorem jellybeans_final_count : final_jellybeans = 23 :=
by
  -- The proof will be provided here if needed
  sorry

end jellybeans_final_count_l64_64609


namespace constants_unique_l64_64271

theorem constants_unique (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → (5 * x) / ((x - 4) * (x - 2) ^ 2) = A / (x - 4) + B / (x - 2) + C / (x - 2) ^ 2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by
  sorry

end constants_unique_l64_64271


namespace tens_digit_of_8_pow_1234_l64_64679

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_l64_64679


namespace ellipse_semi_focal_range_l64_64564

-- Definitions and conditions from the problem
variables (a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : a^2 = b^2 + c^2)

-- Statement of the theorem
theorem ellipse_semi_focal_range : 1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 :=
by 
  sorry

end ellipse_semi_focal_range_l64_64564


namespace missing_digit_l64_64515

theorem missing_digit (x : ℕ) (h1 : x ≥ 0) (h2 : x ≤ 9) : 
  (if x ≥ 2 then 9 * 1000 + x * 100 + 2 * 10 + 1 else 9 * 100 + 2 * 10 + x * 1) - (1 * 1000 + 2 * 100 + 9 * 10 + x) = 8262 → x = 5 :=
by 
  sorry

end missing_digit_l64_64515


namespace nat_solution_unique_l64_64186

theorem nat_solution_unique (n : ℕ) (h : 2 * n - 1 / n^5 = 3 - 2 / n) : 
  n = 1 :=
sorry

end nat_solution_unique_l64_64186


namespace regular_polygon_sides_l64_64708

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l64_64708


namespace Petya_tore_out_sheets_l64_64225

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l64_64225


namespace product_of_third_side_l64_64062

/-- Two sides of a right triangle have lengths 5 and 7. The product of the possible lengths of 
the third side is exactly √1776. -/
theorem product_of_third_side :
  let a := 5
  let b := 7
  (Real.sqrt (a^2 + b^2) * Real.sqrt (b^2 - a^2)) = Real.sqrt 1776 := 
by 
  let a := 5
  let b := 7
  sorry

end product_of_third_side_l64_64062


namespace part_I_solution_part_II_solution_l64_64477

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

theorem part_I_solution :
  ∀ x : ℝ, f x 3 ≥ 1 ↔ 0 ≤ x ∧ x ≤ (4 / 3) := by
  sorry

theorem part_II_solution :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a - |2*x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) := by
  sorry

end part_I_solution_part_II_solution_l64_64477


namespace geometric_and_arithmetic_sequences_l64_64702

theorem geometric_and_arithmetic_sequences (a b c x y : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : 2 * x = a + b)
  (h3 : 2 * y = b + c) :
  (a / x + c / y) = 2 := 
by 
  sorry

end geometric_and_arithmetic_sequences_l64_64702


namespace net_investment_change_l64_64437

def initial_investment : ℝ := 100
def first_year_increase (init : ℝ) : ℝ := init * 1.50
def second_year_decrease (value : ℝ) : ℝ := value * 0.70

theorem net_investment_change :
  second_year_decrease (first_year_increase initial_investment) - initial_investment = 5 :=
by
  -- This will be placeholder proof
  sorry

end net_investment_change_l64_64437


namespace major_arc_circumference_l64_64082

noncomputable def circumference_major_arc 
  (A B C : Point) (r : ℝ) (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) : ℝ :=
  let total_circumference := 2 * Real.pi * r
  let major_arc_angle := 360 - angle_ACB
  major_arc_angle / 360 * total_circumference

theorem major_arc_circumference (A B C : Point) (r : ℝ)
  (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) :
  circumference_major_arc A B C r angle_ACB h1 h2 = (500 / 3) * Real.pi :=
  sorry

end major_arc_circumference_l64_64082


namespace min_value_expression_l64_64091

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_eq : a * b * c = 64)

theorem min_value_expression :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 192 :=
by {
  sorry
}

end min_value_expression_l64_64091


namespace roots_of_f_non_roots_of_g_l64_64601

-- Part (a)

def f (x : ℚ) := x^20 - 123 * x^10 + 1

theorem roots_of_f (a : ℚ) (h : f a = 0) : 
  f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0 :=
by
  sorry

-- Part (b)

def g (x : ℚ) := x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1

theorem non_roots_of_g (β : ℚ) (h : g β = 0) : 
  g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0 :=
by
  sorry

end roots_of_f_non_roots_of_g_l64_64601


namespace correct_option_D_l64_64394

theorem correct_option_D (a : ℝ) : (-a^3)^2 = a^6 :=
sorry

end correct_option_D_l64_64394


namespace sum_roots_of_quadratic_l64_64717

theorem sum_roots_of_quadratic (a b : ℝ) (h₁ : a^2 - a - 6 = 0) (h₂ : b^2 - b - 6 = 0) (h₃ : a ≠ b) :
  a + b = 1 :=
sorry

end sum_roots_of_quadratic_l64_64717


namespace monotonicity_intervals_inequality_condition_l64_64380

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + 2 * x + 1)

theorem monotonicity_intervals :
  (∀ x ∈ Set.Iio (-3 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ), 0 > (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioi (-1 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) := sorry

theorem inequality_condition (a : ℝ) : 
  (∀ x > 0, Real.exp x * (x^2 + 2 * x + 1) > a * x^2 + a * x + 1) ↔ a ≤ 3 := sorry

end monotonicity_intervals_inequality_condition_l64_64380


namespace real_roots_exist_for_all_real_K_l64_64658

theorem real_roots_exist_for_all_real_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x-1) * (x-2) * (x-3) :=
by
  sorry

end real_roots_exist_for_all_real_K_l64_64658


namespace ay_bz_cx_lt_S_squared_l64_64269

theorem ay_bz_cx_lt_S_squared 
  (S : ℝ) (a b c x y z : ℝ) 
  (hS : 0 < S) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a + x = S) 
  (h2 : b + y = S) 
  (h3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := 
sorry

end ay_bz_cx_lt_S_squared_l64_64269


namespace longest_side_of_triangle_l64_64328

theorem longest_side_of_triangle (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) 
    (cond : a^2 + b^2 - 6 * a - 10 * b + 34 = 0) 
    (triangle_ineq1 : a + b > c)
    (triangle_ineq2 : a + c > b)
    (triangle_ineq3 : b + c > a)
    (hScalene: a ≠ b ∧ b ≠ c ∧ a ≠ c) : c = 6 ∨ c = 7 := 
by {
  sorry
}

end longest_side_of_triangle_l64_64328


namespace min_value_f_l64_64034

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end min_value_f_l64_64034


namespace tom_books_l64_64093

theorem tom_books (books_may books_june books_july : ℕ) (h_may : books_may = 2) (h_june : books_june = 6) (h_july : books_july = 10) : 
books_may + books_june + books_july = 18 := by
sorry

end tom_books_l64_64093


namespace max_omega_value_l64_64642

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + φ)

theorem max_omega_value 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : 0 < ω) 
  (hφ : |φ| ≤ Real.pi / 2)
  (h_zero : f ω φ (-Real.pi / 4) = 0)
  (h_sym : f ω φ (Real.pi / 4) = f ω φ (-Real.pi / 4))
  (h_monotonic : ∀ x₁ x₂, (Real.pi / 18) < x₁ → x₁ < x₂ → x₂ < (5 * Real.pi / 36) → f ω φ x₁ < f ω φ x₂) :
  ω = 9 :=
  sorry

end max_omega_value_l64_64642


namespace value_of_x_l64_64449

theorem value_of_x
  (x : ℝ)
  (h1 : x = 0)
  (h2 : x^2 - 1 ≠ 0) :
  (x = 0) ↔ (x ^ 2 - 1 ≠ 0) :=
by
  sorry

end value_of_x_l64_64449


namespace triangle_area_inscribed_in_circle_l64_64041

theorem triangle_area_inscribed_in_circle :
  ∀ (x : ℝ), (2 * x)^2 + (3 * x)^2 = (4 * x)^2 → (5 = (4 * x) / 2) → (1/2 * (2 * x) * (3 * x) = 18.75) :=
by
  -- Assume all necessary conditions
  intros x h_ratio h_radius
  -- Skip the proof part using sorry
  sorry

end triangle_area_inscribed_in_circle_l64_64041


namespace largest_value_l64_64531

-- Definition: Given the condition of a quadratic equation
def equation (a : ℚ) : Prop :=
  8 * a^2 + 6 * a + 2 = 0

-- Theorem: Prove the largest value of 3a + 2 is 5/4 given the condition
theorem largest_value (a : ℚ) (h : equation a) : 
  ∃ m, ∀ b, equation b → (3 * b + 2 ≤ m) ∧ (m = 5 / 4) :=
by
  sorry

end largest_value_l64_64531


namespace remainder_when_sum_divided_by_7_l64_64801

-- Define the sequence
def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the arithmetic sequence
def arithmetic_sequence_sum (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
def a : ℕ := 3
def d : ℕ := 7
def last_term : ℕ := 304

-- Calculate the number of terms in the sequence
noncomputable def n : ℕ := (last_term + 4) / 7

-- Calculate the sum
noncomputable def S : ℕ := arithmetic_sequence_sum a d n

-- Lean 4 statement to prove the remainder
theorem remainder_when_sum_divided_by_7 : S % 7 = 3 := by
  -- sorry placeholder for proof
  sorry

end remainder_when_sum_divided_by_7_l64_64801


namespace earnings_per_widget_l64_64719

/-
Theorem:
Given:
1. Hourly wage is $12.50.
2. Hours worked in a week is 40.
3. Total weekly earnings are $580.
4. Number of widgets produced in a week is 500.

We want to prove:
The earnings per widget are $0.16.
-/

theorem earnings_per_widget (hourly_wage : ℝ) (hours_worked : ℝ)
  (total_weekly_earnings : ℝ) (widgets_produced : ℝ) :
  (hourly_wage = 12.50) →
  (hours_worked = 40) →
  (total_weekly_earnings = 580) →
  (widgets_produced = 500) →
  ( (total_weekly_earnings - hourly_wage * hours_worked) / widgets_produced = 0.16) :=
by
  intros h_wage h_hours h_earnings h_widgets
  sorry

end earnings_per_widget_l64_64719


namespace directrix_of_parabola_l64_64876

-- Define the given condition:
def parabola_eq (x : ℝ) : ℝ := 8 * x^2 + 4 * x + 2

-- State the theorem:
theorem directrix_of_parabola :
  (∀ x : ℝ, parabola_eq x = 8 * (x + 1/4)^2 + 1) → (y = 31 / 32) :=
by
  -- We'll prove this later
  sorry

end directrix_of_parabola_l64_64876


namespace simplify_expression_l64_64617

variable (x y : ℝ)

theorem simplify_expression : (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y :=
by
  sorry

end simplify_expression_l64_64617


namespace tabitha_honey_days_l64_64212

noncomputable def days_of_honey (cups_per_day servings_per_cup total_servings : ℕ) : ℕ :=
  total_servings / (cups_per_day * servings_per_cup)

theorem tabitha_honey_days :
  let cups_per_day := 3
  let servings_per_cup := 1
  let ounces_container := 16
  let servings_per_ounce := 6
  let total_servings := ounces_container * servings_per_ounce
  days_of_honey cups_per_day servings_per_cup total_servings = 32 :=
by
  sorry

end tabitha_honey_days_l64_64212


namespace nate_matches_left_l64_64361

def initial_matches : ℕ := 70
def matches_dropped : ℕ := 10
def matches_eaten : ℕ := 2 * matches_dropped
def total_matches_lost : ℕ := matches_dropped + matches_eaten
def remaining_matches : ℕ := initial_matches - total_matches_lost

theorem nate_matches_left : remaining_matches = 40 := by
  sorry

end nate_matches_left_l64_64361


namespace selling_price_ratio_l64_64485

theorem selling_price_ratio (CP SP1 SP2 : ℝ) (h1 : SP1 = CP + 0.5 * CP) (h2 : SP2 = CP + 3 * CP) :
  SP2 / SP1 = 8 / 3 :=
by
  sorry

end selling_price_ratio_l64_64485


namespace domain_of_f_l64_64744

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f y = x}

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : domain f = {x | x > 1} := sorry

end domain_of_f_l64_64744


namespace fraction_of_tips_l64_64311

variable (S T : ℝ) -- assuming S is salary and T is tips
variable (h : T / (S + T) = 0.7142857142857143)

/-- 
If the fraction of the waiter's income from tips is 0.7142857142857143,
then the fraction of his salary that were his tips is 2.5.
-/
theorem fraction_of_tips (h : T / (S + T) = 0.7142857142857143) : T / S = 2.5 :=
sorry

end fraction_of_tips_l64_64311


namespace angle_greater_than_150_l64_64562

theorem angle_greater_than_150 (a b c R : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c < 2 * R) : 
  ∃ (A : ℝ), A > 150 ∧ ( ∃ (B C : ℝ), A + B + C = 180 ) :=
sorry

end angle_greater_than_150_l64_64562


namespace circle_intersection_l64_64009

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_intersection (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ (∃ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0)) ↔ 4 < m ∧ m < 144 :=
by
  have h1 : distance (0, 0) (3, -4) = 5 := by sorry
  have h2 : ∀ m, |7 - Real.sqrt m| < 5 ↔ 4 < m ∧ m < 144 := by sorry
  exact sorry

end circle_intersection_l64_64009


namespace differential_equation_approx_solution_l64_64883

open Real

noncomputable def approximate_solution (x : ℝ) : ℝ := 0.1 * exp (x ^ 2 / 2)

theorem differential_equation_approx_solution :
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 →
  ∀ (y : ℝ), -1/2 ≤ y ∧ y ≤ 1/2 →
  abs (approximate_solution x - y) < 1 / 650 :=
sorry

end differential_equation_approx_solution_l64_64883


namespace complement_M_in_U_l64_64565

-- Define the universal set U and set M
def U : Finset ℕ := {4, 5, 6, 8, 9}
def M : Finset ℕ := {5, 6, 8}

-- Define the complement of M in U
def complement (U M : Finset ℕ) : Finset ℕ := U \ M

-- Prove that the complement of M in U is {4, 9}
theorem complement_M_in_U : complement U M = {4, 9} := by
  sorry

end complement_M_in_U_l64_64565


namespace number_of_oarsmen_l64_64614

-- Define the conditions
variables (n : ℕ)
variables (W : ℕ)
variables (h_avg_increase : (W + 40) / n = W / n + 2)

-- Lean 4 statement without the proof
theorem number_of_oarsmen : n = 20 :=
by
  sorry

end number_of_oarsmen_l64_64614


namespace average_operating_time_l64_64304

-- Definition of problem conditions
def cond1 : Nat := 5 -- originally had 5 air conditioners
def cond2 : Nat := 6 -- after installing 1 more
def total_hours : Nat := 24 * 5 -- total operating hours allowable in 24 hours

-- Formalize the average operating time calculation
theorem average_operating_time : (total_hours / cond2) = 20 := by
  sorry

end average_operating_time_l64_64304


namespace not_jog_probability_eq_l64_64890

def P_jog : ℚ := 5 / 8

theorem not_jog_probability_eq :
  1 - P_jog = 3 / 8 :=
by
  sorry

end not_jog_probability_eq_l64_64890


namespace value_of_k_l64_64581

   noncomputable def k (a b : ℝ) : ℝ := 3 / 4

   theorem value_of_k (a b k : ℝ) 
     (h1: b = 4 * k + 1) 
     (h2: 5 = a * k + 1) 
     (h3: b + 1 = a * k + 1) : 
     k = 3 / 4 := 
   by 
     -- Proof goes here 
     sorry
   
end value_of_k_l64_64581


namespace largest_n_for_factoring_l64_64239

theorem largest_n_for_factoring :
  ∃ (n : ℤ), (∀ (A B : ℤ), (3 * A + B = n) → (3 * A * B = 90) → n = 271) :=
by sorry

end largest_n_for_factoring_l64_64239


namespace distance_A_beats_B_l64_64546

theorem distance_A_beats_B 
  (A_time : ℝ) (A_distance : ℝ) (B_time : ℝ) (B_distance : ℝ)
  (hA : A_distance = 128) (hA_time : A_time = 28)
  (hB : B_distance = 128) (hB_time : B_time = 32) :
  (A_distance - (B_distance * (A_time / B_time))) = 16 :=
by
  sorry

end distance_A_beats_B_l64_64546


namespace platform_length_l64_64829

noncomputable def train_length : ℕ := 1200
noncomputable def time_to_cross_tree : ℕ := 120
noncomputable def time_to_pass_platform : ℕ := 230

theorem platform_length
  (v : ℚ)
  (h1 : v = train_length / time_to_cross_tree)
  (total_distance : ℚ)
  (h2 : total_distance = v * time_to_pass_platform)
  (platform_length : ℚ)
  (h3 : total_distance = train_length + platform_length) :
  platform_length = 1100 := by 
  sorry

end platform_length_l64_64829


namespace find_f_2014_l64_64897

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f x = f (x + 1) - f (x + 2)
axiom interval_def : ∀ x, 0 < x ∧ x < 3 → f x = x^2

theorem find_f_2014 : f 2014 = -1 := sorry

end find_f_2014_l64_64897


namespace square_area_from_diagonal_l64_64356

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l64_64356


namespace shopkeeper_gain_l64_64690

noncomputable def overall_percentage_gain (P : ℝ) (increase_percentage : ℝ) (discount1_percentage : ℝ) (discount2_percentage : ℝ) : ℝ :=
  let increased_price := P * (1 + increase_percentage)
  let price_after_first_discount := increased_price * (1 - discount1_percentage)
  let final_price := price_after_first_discount * (1 - discount2_percentage)
  ((final_price - P) / P) * 100

theorem shopkeeper_gain : 
  overall_percentage_gain 100 0.32 0.10 0.15 = 0.98 :=
by
  sorry

end shopkeeper_gain_l64_64690


namespace maximize_profit_l64_64008

variables (a x : ℝ) (t : ℝ := 5 - 12 / (x + 3)) (cost : ℝ := 10 + 2 * t) 
  (price : ℝ := 5 + 20 / t) (profit : ℝ := 2 * (price * t - cost - x))

-- Assume non-negativity and upper bound on promotional cost
variable (h_a_nonneg : 0 ≤ a)
variable (h_a_pos : 0 < a)

noncomputable def profit_function (x : ℝ) : ℝ := 20 - 4 / x - x

-- Prove the maximum promotional cost that maximizes the profit
theorem maximize_profit : 
  (if a ≥ 2 then ∃ y, y = 2 ∧ profit_function y = profit_function 2 
   else ∃ y, y = a ∧ profit_function y = profit_function a) := 
sorry

end maximize_profit_l64_64008


namespace sum_g_h_k_l64_64817

def polynomial_product_constants (d g h k : ℤ) : Prop :=
  ((5 * d^2 + 4 * d + g) * (4 * d^2 + h * d - 5) = 20 * d^4 + 11 * d^3 - 9 * d^2 + k * d - 20)

theorem sum_g_h_k (d g h k : ℤ) (h1 : polynomial_product_constants d g h k) : g + h + k = -16 :=
by
  sorry

end sum_g_h_k_l64_64817


namespace correct_calculation_given_conditions_l64_64549

variable (number : ℤ)

theorem correct_calculation_given_conditions 
  (h : number + 16 = 64) : number - 16 = 32 := by
  sorry

end correct_calculation_given_conditions_l64_64549


namespace total_weight_is_1kg_total_weight_in_kg_eq_1_l64_64638

theorem total_weight_is_1kg 
  (weight_msg : ℕ := 80)
  (weight_salt : ℕ := 500)
  (weight_detergent : ℕ := 420) :
  (weight_msg + weight_salt + weight_detergent) = 1000 := by
sorry

theorem total_weight_in_kg_eq_1 
  (total_weight_g : ℕ := weight_msg + weight_salt + weight_detergent) :
  (total_weight_g = 1000) → (total_weight_g / 1000 = 1) := by
sorry

end total_weight_is_1kg_total_weight_in_kg_eq_1_l64_64638


namespace MrKishore_petrol_expense_l64_64183

theorem MrKishore_petrol_expense 
  (rent milk groceries education misc savings salary expenses petrol : ℝ)
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_education : education = 2500)
  (h_misc : misc = 700)
  (h_savings : savings = 1800)
  (h_salary : salary = 18000)
  (h_expenses_equation : expenses = rent + milk + groceries + education + petrol + misc)
  (h_savings_equation : savings = salary * 0.10)
  (h_total_equation : salary = expenses + savings) :
  petrol = 2000 :=
by
  sorry

end MrKishore_petrol_expense_l64_64183


namespace product_of_binaries_l64_64688

-- Step a) Define the binary numbers as Lean 4 terms.
def bin_11011 : ℕ := 0b11011
def bin_111 : ℕ := 0b111
def bin_101 : ℕ := 0b101

-- Step c) Define the goal to be proven.
theorem product_of_binaries :
  bin_11011 * bin_111 * bin_101 = 0b1110110001 :=
by
  -- proof goes here
  sorry

end product_of_binaries_l64_64688


namespace salesperson_commission_l64_64530

noncomputable def commission (sale_price : ℕ) (rate : ℚ) : ℚ :=
  rate * sale_price

noncomputable def total_commission (machines_sold : ℕ) (first_rate : ℚ) (second_rate : ℚ) (sale_price : ℕ) : ℚ :=
  let first_commission := commission sale_price first_rate * 100
  let second_commission := commission sale_price second_rate * (machines_sold - 100)
  first_commission + second_commission

theorem salesperson_commission :
  total_commission 130 0.03 0.04 10000 = 42000 := by
  sorry

end salesperson_commission_l64_64530


namespace hyperbola_range_l64_64937

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (|m| - 1) - y^2 / (m - 2) = 1)) → (-1 < m ∧ m < 1) ∨ (m > 2) := by
  sorry

end hyperbola_range_l64_64937


namespace exists_negative_root_of_P_l64_64079

def P(x : ℝ) : ℝ := x^7 - 2 * x^6 - 7 * x^4 - x^2 + 10

theorem exists_negative_root_of_P : ∃ x : ℝ, x < 0 ∧ P x = 0 :=
sorry

end exists_negative_root_of_P_l64_64079


namespace multiplication_more_than_subtraction_l64_64482

def x : ℕ := 22

def multiplication_result : ℕ := 3 * x
def subtraction_result : ℕ := 62 - x
def difference : ℕ := multiplication_result - subtraction_result

theorem multiplication_more_than_subtraction : difference = 26 :=
by
  sorry

end multiplication_more_than_subtraction_l64_64482


namespace christine_stickers_needed_l64_64695

-- Define the number of stickers Christine has
def stickers_has : ℕ := 11

-- Define the number of stickers required for the prize
def stickers_required : ℕ := 30

-- Define the formula to calculate the number of stickers Christine needs
def stickers_needed : ℕ := stickers_required - stickers_has

-- The theorem we need to prove
theorem christine_stickers_needed : stickers_needed = 19 :=
by
  sorry

end christine_stickers_needed_l64_64695


namespace mother_older_than_twice_petra_l64_64670

def petra_age : ℕ := 11
def mother_age : ℕ := 36

def twice_petra_age : ℕ := 2 * petra_age

theorem mother_older_than_twice_petra : mother_age - twice_petra_age = 14 := by
  sorry

end mother_older_than_twice_petra_l64_64670


namespace greater_number_l64_64080

theorem greater_number (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 2) (h3 : a > b) : a = 21 := by
  sorry

end greater_number_l64_64080


namespace fraction_income_spent_on_rent_l64_64723

theorem fraction_income_spent_on_rent
  (hourly_wage : ℕ)
  (work_hours_per_week : ℕ)
  (weeks_in_month : ℕ)
  (food_expense : ℕ)
  (tax_expense : ℕ)
  (remaining_income : ℕ) :
  hourly_wage = 30 →
  work_hours_per_week = 48 →
  weeks_in_month = 4 →
  food_expense = 500 →
  tax_expense = 1000 →
  remaining_income = 2340 →
  ((hourly_wage * work_hours_per_week * weeks_in_month - remaining_income - (food_expense + tax_expense)) / (hourly_wage * work_hours_per_week * weeks_in_month) = 1/3) :=
by
  intros h_wage h_hours h_weeks h_food h_taxes h_remaining
  sorry

end fraction_income_spent_on_rent_l64_64723


namespace perfect_square_trinomial_l64_64493

theorem perfect_square_trinomial (k : ℝ) : (∃ a b : ℝ, (a * x + b) ^ 2 = x^2 - k * x + 4) → (k = 4 ∨ k = -4) :=
by
  sorry

end perfect_square_trinomial_l64_64493


namespace no_real_solutions_l64_64467

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
    (a = 0) ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a ^ 2 > 0) :=
by
  sorry

end no_real_solutions_l64_64467


namespace pyramid_base_edge_length_l64_64103

noncomputable def edge_length_of_pyramid_base : ℝ :=
  let R := 4 -- radius of the hemisphere
  let h := 12 -- height of the pyramid
  let base_length := 6 -- edge-length of the base of the pyramid to be proved
  -- assume necessary geometric configurations of the pyramid and sphere
  base_length

theorem pyramid_base_edge_length :
  ∀ R h base_length, R = 4 → h = 12 → edge_length_of_pyramid_base = base_length → base_length = 6 :=
by
  intros R h base_length hR hH hBaseLength
  have R_spec : R = 4 := hR
  have h_spec : h = 12 := hH
  have base_length_spec : edge_length_of_pyramid_base = base_length := hBaseLength
  sorry

end pyramid_base_edge_length_l64_64103


namespace ribbon_cuts_l64_64021

theorem ribbon_cuts (rolls : ℕ) (length_per_roll : ℕ) (piece_length : ℕ) (total_rolls : rolls = 5) (roll_length : length_per_roll = 50) (piece_size : piece_length = 2) : 
  (rolls * ((length_per_roll / piece_length) - 1) = 120) :=
by
  sorry

end ribbon_cuts_l64_64021


namespace range_of_k_l64_64334

theorem range_of_k 
  (h : ∀ x : ℝ, x = 1 → k^2 * x^2 - 6 * k * x + 8 ≥ 0) :
  k ≥ 4 ∨ k ≤ 2 := by
sorry

end range_of_k_l64_64334


namespace cars_count_l64_64129

theorem cars_count
  (distance : ℕ)
  (time_between_cars : ℕ)
  (total_time_hours : ℕ)
  (cars_per_hour : ℕ)
  (expected_cars_count : ℕ) :
  distance = 3 →
  time_between_cars = 20 →
  total_time_hours = 10 →
  cars_per_hour = 3 →
  expected_cars_count = total_time_hours * cars_per_hour →
  expected_cars_count = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4] at h5
  exact h5


end cars_count_l64_64129


namespace mean_of_three_l64_64232

theorem mean_of_three (x y z a : ℝ)
  (h₁ : (x + y) / 2 = 5)
  (h₂ : (y + z) / 2 = 9)
  (h₃ : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 :=
by
  sorry

end mean_of_three_l64_64232


namespace man_work_alone_l64_64833

theorem man_work_alone (W: ℝ) (M S: ℝ)
  (hS: S = W / 6.67)
  (hMS: M + S = W / 4):
  W / M = 10 :=
by {
  -- This is a placeholder for the proof
  sorry
}

end man_work_alone_l64_64833


namespace mountaineers_arrangement_l64_64659
open BigOperators

-- Definition to state the number of combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The main statement translating our problem
theorem mountaineers_arrangement :
  (choose 4 2) * (choose 6 2) = 120 := by
  sorry

end mountaineers_arrangement_l64_64659


namespace emily_subtracts_99_l64_64404

theorem emily_subtracts_99 : ∀ (a b : ℕ), (51 * 51 = a + 101) → (49 * 49 = b - 99) → b - 99 = 2401 := by
  intros a b h1 h2
  sorry

end emily_subtracts_99_l64_64404


namespace parabola_tangent_hyperbola_l64_64428

theorem parabola_tangent_hyperbola (m : ℝ) :
  (∀ x : ℝ, (x^2 + 5)^2 - m * x^2 = 4 → y = x^2 + 5)
  ∧ (∀ y : ℝ, y ≥ 5 → y^2 - m * x^2 = 4) →
  (m = 10 + 2 * Real.sqrt 21 ∨ m = 10 - 2 * Real.sqrt 21) :=
  sorry

end parabola_tangent_hyperbola_l64_64428


namespace people_count_l64_64603

theorem people_count (wheels_per_person total_wheels : ℕ) (h1 : wheels_per_person = 4) (h2 : total_wheels = 320) :
  total_wheels / wheels_per_person = 80 :=
sorry

end people_count_l64_64603


namespace determine_8_genuine_coins_l64_64850

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end determine_8_genuine_coins_l64_64850


namespace keith_turnips_l64_64085

theorem keith_turnips (a t k : ℕ) (h1 : a = 9) (h2 : t = 15) : k = t - a := by
  sorry

end keith_turnips_l64_64085


namespace a_is_4_when_b_is_3_l64_64784

theorem a_is_4_when_b_is_3 
  (a : ℝ) (b : ℝ) (k : ℝ)
  (h1 : ∀ b, a * b^2 = k)
  (h2 : a = 9 ∧ b = 2) :
  a = 4 :=
by
  sorry

end a_is_4_when_b_is_3_l64_64784


namespace part1_a1_union_part2_A_subset_complement_B_l64_64197

open Set Real

-- Definitions for Part (1)
def A : Set ℝ := {x | (x - 1) / (x - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a^2 - 1 < 0}

-- Statement for Part (1)
theorem part1_a1_union (a : ℝ) (h : a = 1) : A ∪ B 1 = {x | 0 < x ∧ x < 5} :=
sorry

-- Definitions for Part (2)
def complement_B (a : ℝ) : Set ℝ := {x | x ≤ a - 1 ∨ x ≥ a + 1}

-- Statement for Part (2)
theorem part2_A_subset_complement_B : (∀ x, (1 < x ∧ x < 5) → (x ≤ a - 1 ∨ x ≥ a + 1)) → (a ≤ 0 ∨ a ≥ 6) :=
sorry

end part1_a1_union_part2_A_subset_complement_B_l64_64197


namespace fraction_division_l64_64706

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end fraction_division_l64_64706


namespace rhombus_locus_l64_64887

-- Define the coordinates of the vertices of the rhombus
structure Point :=
(x : ℝ)
(y : ℝ)

def A (e : ℝ) : Point := ⟨e, 0⟩
def B (f : ℝ) : Point := ⟨0, f⟩
def C (e : ℝ) : Point := ⟨-e, 0⟩
def D (f : ℝ) : Point := ⟨0, -f⟩

-- Define the distance squared from a point P to a point Q
def dist_sq (P Q : Point) : ℝ := (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the geometric locus problem
theorem rhombus_locus (P : Point) (e f : ℝ) :
  dist_sq P (A e) = dist_sq P (B f) + dist_sq P (C e) + dist_sq P (D f) ↔
  (if e > f then
    (dist_sq P (A e) = (e^2 - f^2) ∨ dist_sq P (C e) = (e^2 - f^2))
   else if e = f then
    (P = A e ∨ P = B f ∨ P = C e ∨ P = D f)
   else
    false) :=
sorry

end rhombus_locus_l64_64887


namespace inequality_proof_l64_64202

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 := 
sorry

end inequality_proof_l64_64202


namespace problem_statement_l64_64350

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end problem_statement_l64_64350


namespace x_intercept_is_7_0_l64_64668

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_is_7_0_l64_64668


namespace find_greatest_consecutive_integer_l64_64228

theorem find_greatest_consecutive_integer (n : ℤ) 
  (h : n^2 + (n + 1)^2 = 452) : n + 1 = 15 :=
sorry

end find_greatest_consecutive_integer_l64_64228


namespace amanda_car_round_trip_time_l64_64494

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end amanda_car_round_trip_time_l64_64494


namespace min_value_3x_4y_l64_64154

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 25 :=
sorry

end min_value_3x_4y_l64_64154


namespace rain_ratio_l64_64913

def monday_rain := 2 + 1 -- inches of rain on Monday
def wednesday_rain := 0 -- inches of rain on Wednesday
def thursday_rain := 1 -- inches of rain on Thursday
def average_rain_per_day := 4 -- daily average rain total
def days_in_week := 5 -- days in a week
def weekly_total_rain := average_rain_per_day * days_in_week

-- Theorem statement
theorem rain_ratio (tuesday_rain : ℝ) (friday_rain : ℝ) 
  (h1 : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h2 : monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = weekly_total_rain) :
  tuesday_rain / monday_rain = 2 := 
sorry

end rain_ratio_l64_64913


namespace suff_not_nec_for_abs_eq_one_l64_64712

variable (m : ℝ)

theorem suff_not_nec_for_abs_eq_one (hm : m = 1) : |m| = 1 ∧ (¬(|m| = 1 → m = 1)) := by
  sorry

end suff_not_nec_for_abs_eq_one_l64_64712


namespace hyperbola_condition_l64_64925

theorem hyperbola_condition (k : ℝ) : (3 - k) * (k - 2) < 0 ↔ k < 2 ∨ k > 3 := by
  sorry

end hyperbola_condition_l64_64925


namespace initial_students_count_l64_64367

variable (initial_students : ℕ)
variable (number_of_new_boys : ℕ := 5)
variable (initial_percentage_girls : ℝ := 0.40)
variable (new_percentage_girls : ℝ := 0.32)

theorem initial_students_count (h : initial_percentage_girls * initial_students = new_percentage_girls * (initial_students + number_of_new_boys)) : 
  initial_students = 20 := 
by 
  sorry

end initial_students_count_l64_64367


namespace valid_paths_from_P_to_Q_l64_64867

-- Define the grid dimensions and alternate coloring conditions
def grid_width := 10
def grid_height := 8
def is_white_square (r c : ℕ) : Bool :=
  (r + c) % 2 = 1

-- Define the starting and ending squares P and Q
def P : ℕ × ℕ := (0, grid_width / 2)
def Q : ℕ × ℕ := (grid_height - 1, grid_width / 2)

-- Define a function to count valid 9-step paths from P to Q
noncomputable def count_valid_paths : ℕ :=
  -- Here the function to compute valid paths would be defined
  -- This is broad outline due to lean's framework missing specific combinatorial functions
  245

-- The theorem to state the proof problem
theorem valid_paths_from_P_to_Q : count_valid_paths = 245 :=
sorry

end valid_paths_from_P_to_Q_l64_64867


namespace factor_expression_l64_64838

variable (x : ℝ)

theorem factor_expression :
  (18 * x ^ 6 + 50 * x ^ 4 - 8) - (2 * x ^ 6 - 6 * x ^ 4 - 8) = 8 * x ^ 4 * (2 * x ^ 2 + 7) :=
by
  sorry

end factor_expression_l64_64838


namespace translate_vertex_to_increase_l64_64509

def quadratic_function (x : ℝ) : ℝ := -x^2 + 1

theorem translate_vertex_to_increase (x : ℝ) :
  ∃ v, v = (2, quadratic_function 2) ∧
    (∀ x < 2, quadratic_function (x + 2) = quadratic_function x + 1 ∧
    ∀ x < 2, quadratic_function x < quadratic_function (x + 1)) :=
sorry

end translate_vertex_to_increase_l64_64509


namespace find_x_square_l64_64068

theorem find_x_square (x : ℝ) (h_pos : x > 0) (h_condition : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end find_x_square_l64_64068


namespace safe_travel_exists_l64_64527

def total_travel_time : ℕ := 16
def first_crater_cycle : ℕ := 18
def first_crater_duration : ℕ := 1
def second_crater_cycle : ℕ := 10
def second_crater_duration : ℕ := 1

theorem safe_travel_exists : 
  ∃ t : ℕ, t ∈ { t | (∀ k : ℕ, t % first_crater_cycle ≠ k ∨ t % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, t % second_crater_cycle ≠ k ∨ t % second_crater_cycle ≥ second_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % first_crater_cycle ≠ k ∨ (t + total_travel_time) % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % second_crater_cycle ≠ k ∨ (t + total_travel_time) % second_crater_cycle ≥ second_crater_duration) } :=
sorry

end safe_travel_exists_l64_64527


namespace find_a_l64_64299

def f(x : ℚ) : ℚ := x / 3 + 2
def g(x : ℚ) : ℚ := 5 - 2 * x

theorem find_a (a : ℚ) (h : f (g a) = 4) : a = -1 / 2 :=
by
  sorry

end find_a_l64_64299


namespace remainder_addition_l64_64138

theorem remainder_addition (m : ℕ) (k : ℤ) (h : m = 9 * k + 4) : (m + 2025) % 9 = 4 := by
  sorry

end remainder_addition_l64_64138


namespace train_crossing_time_l64_64855

theorem train_crossing_time
  (L1 L2 : ℝ) (v : ℝ) 
  (t1 t2 t : ℝ) 
  (h_t1 : t1 = 27)
  (h_t2 : t2 = 17)
  (hv_ratio : v = v)
  (h_L1 : L1 = v * t1)
  (h_L2 : L2 = v * t2)
  (h_t12 : t = (L1 + L2) / (v + v)) :
  t = 22 :=
by
  sorry

end train_crossing_time_l64_64855


namespace distinct_nat_numbers_l64_64247

theorem distinct_nat_numbers 
  (a b c : ℕ) (p q r : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_sum : a + b + c = 55) 
  (h_ab : a + b = p * p) 
  (h_bc : b + c = q * q) 
  (h_ca : c + a = r * r) : 
  a = 19 ∧ b = 6 ∧ c = 30 :=
sorry

end distinct_nat_numbers_l64_64247


namespace lcm_of_10_and_21_l64_64370

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 :=
by
  sorry

end lcm_of_10_and_21_l64_64370


namespace sphere_radius_and_volume_l64_64934

theorem sphere_radius_and_volume (A : ℝ) (d : ℝ) (π : ℝ) (r : ℝ) (R : ℝ) (V : ℝ) 
  (h_cross_section : A = π) (h_distance : d = 1) (h_radius : r = 1) :
  R = Real.sqrt (r^2 + d^2) ∧ V = (4 / 3) * π * R^3 := 
by
  sorry

end sphere_radius_and_volume_l64_64934


namespace overall_gain_is_2_89_l64_64611

noncomputable def overall_gain_percentage : ℝ :=
  let cost1 := 500000
  let gain1 := 0.10
  let sell1 := cost1 * (1 + gain1)

  let cost2 := 600000
  let loss2 := 0.05
  let sell2 := cost2 * (1 - loss2)

  let cost3 := 700000
  let gain3 := 0.15
  let sell3 := cost3 * (1 + gain3)

  let cost4 := 800000
  let loss4 := 0.12
  let sell4 := cost4 * (1 - loss4)

  let cost5 := 900000
  let gain5 := 0.08
  let sell5 := cost5 * (1 + gain5)

  let total_cost := cost1 + cost2 + cost3 + cost4 + cost5
  let total_sell := sell1 + sell2 + sell3 + sell4 + sell5
  let overall_gain := total_sell - total_cost
  (overall_gain / total_cost) * 100

theorem overall_gain_is_2_89 :
  overall_gain_percentage = 2.89 :=
by
  -- Proof goes here
  sorry

end overall_gain_is_2_89_l64_64611


namespace find_m_l64_64660

theorem find_m (m : ℝ) 
  (h : (1 : ℝ) * (-3 : ℝ) + (3 : ℝ) * ((3 : ℝ) + 2 * m) = 0) : 
  m = -1 :=
by sorry

end find_m_l64_64660


namespace exists_real_x_for_sequence_floor_l64_64319

open Real

theorem exists_real_x_for_sequence_floor (a : Fin 1998 → ℕ)
  (h1 : ∀ n : Fin 1998, 0 ≤ a n)
  (h2 : ∀ (i j : Fin 1998), (i.val + j.val ≤ 1997) → (a i + a j ≤ a ⟨i.val + j.val, sorry⟩ ∧ a ⟨i.val + j.val, sorry⟩ ≤ a i + a j + 1)) :
  ∃ x : ℝ, ∀ n : Fin 1998, a n = ⌊(n.val + 1) * x⌋ :=
sorry

end exists_real_x_for_sequence_floor_l64_64319


namespace woman_finishes_work_in_225_days_l64_64070

theorem woman_finishes_work_in_225_days
  (M W : ℝ)
  (h1 : (10 * M + 15 * W) * 6 = 1)
  (h2 : M * 100 = 1) :
  1 / W = 225 :=
by
  sorry

end woman_finishes_work_in_225_days_l64_64070


namespace pudding_cups_initial_l64_64069

theorem pudding_cups_initial (P : ℕ) (students : ℕ) (extra_cups : ℕ) 
  (h1 : students = 218) (h2 : extra_cups = 121) (h3 : P + extra_cups = students) : P = 97 := 
by
  sorry

end pudding_cups_initial_l64_64069


namespace test_total_questions_l64_64266

theorem test_total_questions (total_points : ℕ) (num_5_point_questions : ℕ) (points_per_5_point_question : ℕ) (points_per_10_point_question : ℕ) : 
  total_points = 200 → 
  num_5_point_questions = 20 → 
  points_per_5_point_question = 5 → 
  points_per_10_point_question = 10 → 
  (total_points = (num_5_point_questions * points_per_5_point_question) + 
    ((total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) * points_per_10_point_question) →
  (num_5_point_questions + (total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end test_total_questions_l64_64266


namespace propositions_alpha_and_beta_true_l64_64322

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

def strictly_increasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

def strictly_decreasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x > f y

def alpha (f : ℝ → ℝ) : Prop :=
∀ x, ∃ g h : ℝ → ℝ, even_function g ∧ odd_function h ∧ f x = g x + h x

def beta (f : ℝ → ℝ) : Prop :=
∀ x, strictly_increasing_function f → ∃ p q : ℝ → ℝ, 
  strictly_increasing_function p ∧ strictly_decreasing_function q ∧ f x = p x + q x

theorem propositions_alpha_and_beta_true (f : ℝ → ℝ) :
  alpha f ∧ beta f :=
by
  sorry

end propositions_alpha_and_beta_true_l64_64322


namespace total_students_l64_64539

theorem total_students (h1 : 15 * 70 = 1050) 
                       (h2 : 10 * 95 = 950) 
                       (h3 : 1050 + 950 = 2000)
                       (h4 : 80 * N = 2000) :
  N = 25 :=
by sorry

end total_students_l64_64539


namespace positive_integer_solutions_inequality_l64_64259

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_l64_64259


namespace problem_equivalent_l64_64434

theorem problem_equivalent (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) (hz_eq : z = 10 * y) :
  (x + 4 * y + z) / (4 * x - y - z) = 0 :=
by
  sorry

end problem_equivalent_l64_64434


namespace quadratic_has_two_real_roots_l64_64589

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x^2 - (m + 1) * x + (3 * m - 6) = 0 :=
by
  sorry

end quadratic_has_two_real_roots_l64_64589


namespace correlation_identification_l64_64013

noncomputable def relationship (a b : Type) : Prop := 
  ∃ (f : a → b), true

def correlation (a b : Type) : Prop :=
  relationship a b ∧ relationship b a

def deterministic (a b : Type) : Prop :=
  ∀ x y : a, ∃! z : b, true

def age_wealth : Prop := correlation ℕ ℝ
def point_curve_coordinates : Prop := deterministic (ℝ × ℝ) (ℝ × ℝ)
def apple_production_climate : Prop := correlation ℝ ℝ
def tree_diameter_height : Prop := correlation ℝ ℝ

theorem correlation_identification :
  age_wealth ∧ apple_production_climate ∧ tree_diameter_height ∧ ¬point_curve_coordinates := 
by
  -- proof of these properties
  sorry

end correlation_identification_l64_64013


namespace zongzi_unit_prices_max_type_A_zongzi_l64_64683

theorem zongzi_unit_prices (x : ℝ) : 
  (800 / x - 1200 / (2 * x) = 50) → 
  (x = 4 ∧ 2 * x = 8) :=
by
  intro h
  sorry

theorem max_type_A_zongzi (m : ℕ) : 
  (m ≤ 200) → 
  (8 * m + 4 * (200 - m) ≤ 1150) → 
  (m ≤ 87) :=
by
  intros h1 h2
  sorry

end zongzi_unit_prices_max_type_A_zongzi_l64_64683


namespace cost_price_of_bicycle_l64_64250

variables {CP_A SP_AB SP_BC : ℝ}

theorem cost_price_of_bicycle (h1 : SP_AB = CP_A * 1.2)
                             (h2 : SP_BC = SP_AB * 1.25)
                             (h3 : SP_BC = 225) :
                             CP_A = 150 :=
by sorry

end cost_price_of_bicycle_l64_64250


namespace max_remainder_when_divided_by_8_l64_64942

-- Define the problem: greatest possible remainder when apples divided by 8.
theorem max_remainder_when_divided_by_8 (n : ℕ) : ∃ r : ℕ, r < 8 ∧ r = 7 ∧ n % 8 = r := 
sorry

end max_remainder_when_divided_by_8_l64_64942


namespace overlap_length_l64_64341

-- Variables in the conditions
variables (tape_length overlap total_length : ℕ)

-- Conditions
def two_tapes_overlap := (tape_length + tape_length - overlap = total_length)

-- The proof statement we need to prove
theorem overlap_length (h : two_tapes_overlap 275 overlap 512) : overlap = 38 :=
by
  sorry

end overlap_length_l64_64341


namespace smallest_K_for_triangle_l64_64574

theorem smallest_K_for_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) 
  : ∃ K : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → a + c > b → (a^2 + c^2) / b^2 > K) ∧ K = 1 / 2 :=
by
  sorry

end smallest_K_for_triangle_l64_64574


namespace coefficient_x_squared_in_expansion_l64_64903

theorem coefficient_x_squared_in_expansion :
  (∃ c : ℤ, (1 + x)^6 * (1 - x) = c * x^2 + b * x + a) → c = 9 :=
by
  sorry

end coefficient_x_squared_in_expansion_l64_64903


namespace calculate_value_l64_64090

theorem calculate_value : (245^2 - 225^2) / 20 = 470 :=
by
  sorry

end calculate_value_l64_64090


namespace find_number_l64_64553

theorem find_number (num : ℝ) (x : ℝ) (h1 : x = 0.08999999999999998) (h2 : num / x = 0.1) : num = 0.008999999999999999 :=
by 
  sorry

end find_number_l64_64553


namespace acronym_XYZ_length_l64_64285

theorem acronym_XYZ_length :
  let X_length := 2 * Real.sqrt 2
  let Y_length := 1 + 2 * Real.sqrt 2
  let Z_length := 4 + Real.sqrt 5
  X_length + Y_length + Z_length = 5 + 4 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end acronym_XYZ_length_l64_64285


namespace three_digit_number_is_11_times_sum_of_digits_l64_64535

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end three_digit_number_is_11_times_sum_of_digits_l64_64535


namespace calculate_expression_evaluate_expression_l64_64884

theorem calculate_expression (a : ℕ) (h : a = 2020) :
  (a^4 - 3*a^3*(a+1) + 4*a*(a+1)^3 - (a+1)^4 + 1) / (a*(a+1)) = a^2 + 4*a + 6 :=
by sorry

theorem evaluate_expression :
  (2020^2 + 4 * 2020 + 6) = 4096046 :=
by sorry

end calculate_expression_evaluate_expression_l64_64884


namespace find_x_of_total_area_l64_64651

theorem find_x_of_total_area 
  (x : Real)
  (h_triangle : (1/2) * (4 * x) * (3 * x) = 6 * x^2)
  (h_square1 : (3 * x)^2 = 9 * x^2)
  (h_square2 : (6 * x)^2 = 36 * x^2)
  (h_total : 6 * x^2 + 9 * x^2 + 36 * x^2 = 700) :
  x = Real.sqrt (700 / 51) :=
by {
  sorry
}

end find_x_of_total_area_l64_64651


namespace Jame_tears_30_cards_at_a_time_l64_64445

theorem Jame_tears_30_cards_at_a_time
    (cards_per_deck : ℕ)
    (times_per_week : ℕ)
    (decks : ℕ)
    (weeks : ℕ)
    (total_cards : ℕ := decks * cards_per_deck)
    (total_times : ℕ := weeks * times_per_week)
    (cards_at_a_time : ℕ := total_cards / total_times)
    (h1 : cards_per_deck = 55)
    (h2 : times_per_week = 3)
    (h3 : decks = 18)
    (h4 : weeks = 11) :
    cards_at_a_time = 30 := by
  -- Proof can be added here
  sorry

end Jame_tears_30_cards_at_a_time_l64_64445


namespace general_term_arithmetic_sum_first_n_terms_geometric_l64_64150

-- Definitions and assumptions based on given conditions
def a (n : ℕ) : ℤ := 2 * n + 1

-- Given conditions
def initial_a1 : ℤ := 3
def common_difference : ℤ := 2

-- Validate the general formula for the arithmetic sequence
theorem general_term_arithmetic : ∀ n : ℕ, a n = 2 * n + 1 := 
by sorry

-- Definitions and assumptions for geometric sequence
def b (n : ℕ) : ℤ := 3^n

-- Sum of the first n terms of the geometric sequence
def Sn (n : ℕ) : ℤ := 3 / 2 * (3^n - 1)

-- Validate the sum formula for the geometric sequence
theorem sum_first_n_terms_geometric (n : ℕ) : Sn n = 3 / 2 * (3^n - 1) := 
by sorry

end general_term_arithmetic_sum_first_n_terms_geometric_l64_64150


namespace complete_square_solution_l64_64223

theorem complete_square_solution :
  ∀ (x : ℝ), (x^2 + 8*x + 9 = 0) → ((x + 4)^2 = 7) :=
by
  intro x h_eq
  sorry

end complete_square_solution_l64_64223


namespace coordinates_of_P_with_respect_to_origin_l64_64511

def point (x y : ℝ) : Prop := True

theorem coordinates_of_P_with_respect_to_origin :
  point 2 (-3) ↔ point 2 (-3) := by
  sorry

end coordinates_of_P_with_respect_to_origin_l64_64511


namespace smallest_class_number_l64_64242

theorem smallest_class_number (x : ℕ)
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = 57) :
  x = 2 :=
by sorry

end smallest_class_number_l64_64242


namespace min_marks_required_l64_64058

-- Definitions and conditions
def grid_size := 7
def strip_size := 4

-- Question and answer as a proof statement
theorem min_marks_required (n : ℕ) (h : grid_size = 2 * n - 1) : 
  (∃ marks : ℕ, 
    (∀ row col : ℕ, 
      row < grid_size → col < grid_size → 
      (∃ i j : ℕ, 
        i < strip_size → j < strip_size → 
        (marks ≥ 12)))) :=
sorry

end min_marks_required_l64_64058


namespace factor_determines_d_l64_64162

theorem factor_determines_d (d : ℚ) :
  (∀ x : ℚ, x - 4 ∣ d * x^3 - 8 * x^2 + 5 * d * x - 12) → d = 5 / 3 := by
  sorry

end factor_determines_d_l64_64162


namespace geometric_sequence_ratio_l64_64757

theorem geometric_sequence_ratio 
  (a_n b_n : ℕ → ℝ) 
  (S_n T_n : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S_n n = a_n n * (1 - (1/2)^n)) 
  (h2 : ∀ n : ℕ, T_n n = b_n n * (1 - (1/3)^n))
  (h3 : ∀ n, n > 0 → (S_n n) / (T_n n) = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 :=
by
  sorry

end geometric_sequence_ratio_l64_64757


namespace geometric_sequence_sum_l64_64357

noncomputable def sum_of_first_n_terms (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_a1 : a_1 = 1) (h_a5 : a_5 = 16) :
  sum_of_first_n_terms 1 q 7 = 127 :=
by
  sorry

end geometric_sequence_sum_l64_64357


namespace cone_sector_volume_ratio_l64_64543

theorem cone_sector_volume_ratio 
  (H R : ℝ) 
  (nonneg_H : 0 ≤ H) 
  (nonneg_R : 0 ≤ R) :
  let volume_original := (1/3) * π * R^2 * H
  let volume_sector   := (1/12) * π * R^2 * H
  volume_sector / volume_sector = 1 :=
  by
    sorry

end cone_sector_volume_ratio_l64_64543


namespace find_x_in_interval_l64_64351

theorem find_x_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (h_eq : (2 - Real.sin (2 * x)) * Real.sin (x + π / 4) = 1) : x = π / 4 := 
sorry

end find_x_in_interval_l64_64351


namespace minutes_until_8_00_am_l64_64837

-- Definitions based on conditions
def time_in_minutes (hours : Nat) (minutes : Nat) : Nat := hours * 60 + minutes

def current_time : Nat := time_in_minutes 7 30 + 16

def target_time : Nat := time_in_minutes 8 0

-- The theorem we need to prove
theorem minutes_until_8_00_am : target_time - current_time = 14 :=
by
  sorry

end minutes_until_8_00_am_l64_64837


namespace square_area_l64_64831

noncomputable def side_length1 (x : ℝ) : ℝ := 5 * x - 20
noncomputable def side_length2 (x : ℝ) : ℝ := 25 - 2 * x

theorem square_area (x : ℝ) (h : side_length1 x = side_length2 x) :
  (side_length1 x)^2 = 7225 / 49 :=
by
  sorry

end square_area_l64_64831


namespace arithmetic_sequence_general_term_geometric_sequence_sum_l64_64312

section ArithmeticSequence

variable {a_n : ℕ → ℤ} {d : ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a_n (n + 1) - a_n n = d

theorem arithmetic_sequence_general_term (h : is_arithmetic_sequence a_n 2) :
  ∃ a1 : ℤ, ∀ n, a_n n = 2 * n + a1 :=
sorry

end ArithmeticSequence

section GeometricSequence

variable {b_n : ℕ → ℤ} {a_n : ℕ → ℤ}

def is_geometric_sequence_with_reference (b_n : ℕ → ℤ) (a_n : ℕ → ℤ) :=
  b_n 1 = a_n 1 ∧ b_n 2 = a_n 4 ∧ b_n 3 = a_n 13

theorem geometric_sequence_sum (h : is_geometric_sequence_with_reference b_n a_n)
  (h_arith : is_arithmetic_sequence a_n 2) :
  ∃ b1 : ℤ, ∀ n, b_n n = b1 * 3^(n - 1) ∧
                (∃ Sn : ℕ → ℤ, Sn n = (3 * (3^n - 1)) / 2) :=
sorry

end GeometricSequence

end arithmetic_sequence_general_term_geometric_sequence_sum_l64_64312


namespace solution_is_correct_l64_64770

-- Define the options
inductive Options
| A_some_other
| B_someone_else
| C_other_person
| D_one_other

-- Define the condition as a function that returns the correct option
noncomputable def correct_option : Options :=
Options.B_someone_else

-- The theorem stating that the correct option must be the given choice
theorem solution_is_correct : correct_option = Options.B_someone_else :=
by
  sorry

end solution_is_correct_l64_64770


namespace proportion_correct_l64_64620

theorem proportion_correct (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x ≠ 0 ∧ y ≠ 0) : x / y = 2 / 5 := 
sorry

end proportion_correct_l64_64620


namespace sqrt_product_l64_64959

theorem sqrt_product (a b c : ℝ) (ha : a = 72) (hb : b = 18) (hc : c = 8) :
  (Real.sqrt a) * (Real.sqrt b) * (Real.sqrt c) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l64_64959


namespace range_of_x_l64_64487

theorem range_of_x (f : ℝ → ℝ) (h_increasing : ∀ x y, x ≤ y → f x ≤ f y) (h_defined : ∀ x, -1 ≤ x ∧ x ≤ 1)
  (h_condition : ∀ x, f (x-2) < f (1-x)) : ∀ x, 1 ≤ x ∧ x < 3/2 :=
by
  sorry

end range_of_x_l64_64487


namespace blue_eyes_blonde_hair_logic_l64_64968

theorem blue_eyes_blonde_hair_logic :
  ∀ (a b c d : ℝ), 
  (a / (a + b) > (a + c) / (a + b + c + d)) →
  (a / (a + c) > (a + b) / (a + b + c + d)) :=
by
  intro a b c d h
  sorry

end blue_eyes_blonde_hair_logic_l64_64968


namespace intersection_of_A_and_B_l64_64144

def A : Set ℝ := { x | x^2 - x - 2 ≥ 0 }
def B : Set ℝ := { x | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -2 ≤ x ∧ x ≤ -1 } := by
-- The proof would go here
sorry

end intersection_of_A_and_B_l64_64144


namespace sum_faces_of_cube_l64_64571

theorem sum_faces_of_cube (p u q v r w : ℕ) (hp : 0 < p) (hu : 0 < u) (hq : 0 < q) (hv : 0 < v)
    (hr : 0 < r) (hw : 0 < w)
    (h_sum_vertices : p * q * r + p * v * r + p * q * w + p * v * w 
        + u * q * r + u * v * r + u * q * w + u * v * w = 2310) : 
    p + u + q + v + r + w = 40 := 
sorry

end sum_faces_of_cube_l64_64571


namespace survey_support_percentage_l64_64684

theorem survey_support_percentage 
  (num_men : ℕ) (percent_men_support : ℝ)
  (num_women : ℕ) (percent_women_support : ℝ)
  (h_men : num_men = 200)
  (h_percent_men_support : percent_men_support = 0.7)
  (h_women : num_women = 500)
  (h_percent_women_support : percent_women_support = 0.75) :
  (num_men * percent_men_support + num_women * percent_women_support) / (num_men + num_women) * 100 = 74 := 
by
  sorry

end survey_support_percentage_l64_64684


namespace quadratic_inequality_solution_l64_64120

theorem quadratic_inequality_solution (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l64_64120


namespace blocks_given_by_father_l64_64382

theorem blocks_given_by_father :
  ∀ (blocks_original total_blocks blocks_given : ℕ), 
  blocks_original = 2 →
  total_blocks = 8 →
  blocks_given = total_blocks - blocks_original →
  blocks_given = 6 :=
by
  intros blocks_original total_blocks blocks_given h1 h2 h3
  sorry

end blocks_given_by_father_l64_64382


namespace train_length_is_549_95_l64_64265

noncomputable def length_of_train 
(speed_of_train : ℝ) -- 63 km/hr
(speed_of_man : ℝ) -- 3 km/hr
(time_to_cross : ℝ) -- 32.997 seconds
: ℝ := 
(speed_of_train - speed_of_man) * (5 / 18) * time_to_cross

theorem train_length_is_549_95 (speed_of_train : ℝ) (speed_of_man : ℝ) (time_to_cross : ℝ) :
    speed_of_train = 63 → speed_of_man = 3 → time_to_cross = 32.997 →
    length_of_train speed_of_train speed_of_man time_to_cross = 549.95 :=
by
  intros h_train h_man h_time
  rw [h_train, h_man, h_time]
  norm_num
  sorry

end train_length_is_549_95_l64_64265


namespace num_zeros_in_product_l64_64537

theorem num_zeros_in_product : ∀ (a b : ℕ), (a = 125) → (b = 960) → (∃ n, a * b = n * 10^4) :=
by
  sorry

end num_zeros_in_product_l64_64537


namespace abs_difference_of_numbers_l64_64810

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) : |x - y| = 4 * Real.sqrt 10 :=
by
  sorry

end abs_difference_of_numbers_l64_64810


namespace percent_of_number_l64_64478

theorem percent_of_number (N : ℝ) (h : (4 / 5) * (3 / 8) * N = 24) : 2.5 * N = 200 :=
by
  sorry

end percent_of_number_l64_64478


namespace intersection_at_one_point_l64_64107

-- Define the quadratic equation derived from the intersection condition
def quadratic (y k : ℝ) : ℝ :=
  3 * y^2 - 2 * y + (k - 4)

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2)^2 - 4 * 3 * (k - 4)

-- The statement of the problem in Lean
theorem intersection_at_one_point (k : ℝ) :
  (∃ y : ℝ, quadratic y k = 0 ∧ discriminant k = 0) ↔ k = 13 / 3 :=
by 
  sorry

end intersection_at_one_point_l64_64107


namespace Q_share_of_profit_l64_64054

def P_investment : ℕ := 54000
def Q_investment : ℕ := 36000
def total_profit : ℕ := 18000

theorem Q_share_of_profit : Q_investment * total_profit / (P_investment + Q_investment) = 7200 := by
  sorry

end Q_share_of_profit_l64_64054


namespace mutually_exclusive_any_two_l64_64965

variables (A B C : Prop)
axiom all_not_defective : A
axiom all_defective : B
axiom not_all_defective : C

theorem mutually_exclusive_any_two :
  (¬(A ∧ B)) ∧ (¬(A ∧ C)) ∧ (¬(B ∧ C)) :=
sorry

end mutually_exclusive_any_two_l64_64965


namespace rectangle_area_l64_64533

theorem rectangle_area (L W P : ℝ) (hL : L = 13) (hP : P = 50) (hP_eq : P = 2 * L + 2 * W) :
  L * W = 156 :=
by
  have hL_val : L = 13 := hL
  have hP_val : P = 50 := hP
  have h_perimeter : P = 2 * L + 2 * W := hP_eq
  sorry

end rectangle_area_l64_64533


namespace log_expression_value_l64_64444

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log_expression_value :
  (log2 8 * (log2 2 / log2 8)) + log2 4 = 3 :=
by
  sorry

end log_expression_value_l64_64444


namespace snack_bar_training_count_l64_64267

noncomputable def num_trained_in_snack_bar 
  (total_employees : ℕ) 
  (trained_in_buffet : ℕ) 
  (trained_in_dining_room : ℕ) 
  (trained_in_two_restaurants : ℕ) 
  (trained_in_three_restaurants : ℕ) : ℕ :=
  total_employees - trained_in_buffet - trained_in_dining_room + 
  trained_in_two_restaurants + trained_in_three_restaurants

theorem snack_bar_training_count : 
  num_trained_in_snack_bar 39 17 18 4 2 = 8 :=
sorry

end snack_bar_training_count_l64_64267


namespace find_n_l64_64864

theorem find_n (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n ≤ 11) (h₂ : 10389 % 12 = n) : n = 9 :=
by sorry

end find_n_l64_64864


namespace yunas_math_score_l64_64461

theorem yunas_math_score (K E M : ℕ) 
  (h1 : (K + E) / 2 = 92) 
  (h2 : (K + E + M) / 3 = 94) : 
  M = 98 :=
sorry

end yunas_math_score_l64_64461


namespace theater_revenue_l64_64076

theorem theater_revenue
  (total_seats : ℕ)
  (adult_price : ℕ)
  (child_price : ℕ)
  (child_tickets_sold : ℕ)
  (total_sold_out : total_seats = 80)
  (child_tickets_sold_cond : child_tickets_sold = 63)
  (adult_ticket_price_cond : adult_price = 12)
  (child_ticket_price_cond : child_price = 5)
  : total_seats * adult_price + child_tickets_sold * child_price = 519 :=
by
  -- proof omitted
  sorry

end theater_revenue_l64_64076


namespace cos_4_arccos_fraction_l64_64501

theorem cos_4_arccos_fraction :
  (Real.cos (4 * Real.arccos (2 / 5))) = (-47 / 625) :=
by
  sorry

end cos_4_arccos_fraction_l64_64501


namespace unsold_tomatoes_l64_64738

theorem unsold_tomatoes (total_harvest sold_maxwell sold_wilson : ℝ) 
(h_total_harvest : total_harvest = 245.5)
(h_sold_maxwell : sold_maxwell = 125.5)
(h_sold_wilson : sold_wilson = 78) :
(total_harvest - (sold_maxwell + sold_wilson) = 42) :=
by {
  sorry
}

end unsold_tomatoes_l64_64738


namespace find_58th_digit_in_fraction_l64_64131

def decimal_representation_of_fraction : ℕ := sorry

theorem find_58th_digit_in_fraction:
  (decimal_representation_of_fraction = 4) := sorry

end find_58th_digit_in_fraction_l64_64131


namespace probability_approx_l64_64919

noncomputable def circumscribed_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

noncomputable def single_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R / 3)^3

noncomputable def total_spheres_volume (R : ℝ) : ℝ :=
  6 * single_sphere_volume R

noncomputable def probability_inside_spheres (R : ℝ) : ℝ :=
  total_spheres_volume R / circumscribed_sphere_volume R

theorem probability_approx (R : ℝ) (hR : R > 0) : 
  abs (probability_inside_spheres R - 0.053) < 0.001 := sorry

end probability_approx_l64_64919


namespace max_homework_time_l64_64572

theorem max_homework_time :
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  biology + history + geography = 180 :=
by
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  show biology + history + geography = 180
  sorry

end max_homework_time_l64_64572


namespace distinct_roots_condition_l64_64457

noncomputable def f (x c : ℝ) : ℝ := x^2 + 6*x + c

theorem distinct_roots_condition (c : ℝ) :
  (∀x : ℝ, f (f x c) = 0 → ∃ a b : ℝ, (a ≠ b) ∧ f x c = a * (x - b) * (x - c) ) →
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end distinct_roots_condition_l64_64457


namespace find_b_l64_64481

theorem find_b (b : ℚ) (m : ℚ) 
  (h1 : x^2 + b*x + 1/6 = (x + m)^2 + 1/18) 
  (h2 : b < 0) : 
  b = -2/3 := 
sorry

end find_b_l64_64481


namespace no_solutions_a_l64_64639

theorem no_solutions_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := 
sorry

end no_solutions_a_l64_64639


namespace centipede_shoes_and_socks_l64_64039

-- Define number of legs
def num_legs : ℕ := 10

-- Define the total number of items
def total_items : ℕ := 2 * num_legs

-- Define the total permutations without constraints
def total_permutations : ℕ := Nat.factorial total_items

-- Define the probability constraint for each leg
def single_leg_probability : ℚ := 1 / 2

-- Define the combined probability constraint for all legs
def all_legs_probability : ℚ := single_leg_probability ^ num_legs

-- Define the number of valid permutations (the answer to prove)
def valid_permutations : ℚ := total_permutations / all_legs_probability

theorem centipede_shoes_and_socks : valid_permutations = (Nat.factorial 20 : ℚ) / 2^10 :=
by
  -- The proof is omitted
  sorry

end centipede_shoes_and_socks_l64_64039


namespace area_ratio_l64_64874

noncomputable def initial_areas (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def misallocated_areas (a b : ℝ) :=
  let b' := b + 0.1 * a - 0.5 * b
  b' = 0.4 * (a + b)

noncomputable def final_ratios (a b c : ℝ) :=
  let a' := 0.9 * a + 0.5 * b
  let b' := b + 0.1 * a - 0.5 * b
  let c' := 0.5 * c
  a' + b' + c' = a + b + c ∧ a' / b' = 2 ∧ b' / c' = 1 

theorem area_ratio (a b c m : ℝ) (h1 : initial_areas a b c) 
  (h2 : misallocated_areas a b)
  (h3 : final_ratios a b c) : 
  (m = 0.4 * a) → (m / (a + b + c) = 1 / 20) :=
sorry

end area_ratio_l64_64874


namespace average_person_funding_l64_64392

-- Define the conditions from the problem
def total_amount_needed : ℝ := 1000
def amount_already_have : ℝ := 200
def number_of_people : ℝ := 80

-- Define the correct answer
def average_funding_per_person : ℝ := 10

-- Formulate the proof statement
theorem average_person_funding :
  (total_amount_needed - amount_already_have) / number_of_people = average_funding_per_person :=
by
  sorry

end average_person_funding_l64_64392


namespace solve_linear_system_l64_64859

theorem solve_linear_system :
  ∃ (x y : ℚ), (4 * x - 3 * y = 2) ∧ (6 * x + 5 * y = 1) ∧ (x = 13 / 38) ∧ (y = -4 / 19) :=
by
  sorry

end solve_linear_system_l64_64859


namespace monthly_salary_l64_64226

variable (S : ℝ)
variable (Saves : ℝ)
variable (NewSaves : ℝ)

open Real

theorem monthly_salary (h1 : Saves = 0.30 * S)
                       (h2 : NewSaves = Saves - 0.25 * Saves)
                       (h3 : NewSaves = 400) :
    S = 1777.78 := by
    sorry

end monthly_salary_l64_64226


namespace least_positive_integer_n_l64_64338

theorem least_positive_integer_n (n : ℕ) (h : (n > 0)) :
  (∃ m : ℕ, m > 0 ∧ (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 8) ∧ (∀ k : ℕ, k > 0 ∧ (1 / (k : ℝ) - 1 / (k + 1 : ℝ) < 1 / 8) → m ≤ k)) →
  n = 3 := by
  sorry

end least_positive_integer_n_l64_64338


namespace number_students_first_class_l64_64762

theorem number_students_first_class
  (average_first_class : ℝ)
  (average_second_class : ℝ)
  (students_second_class : ℕ)
  (combined_average : ℝ)
  (total_students : ℕ)
  (total_marks_first_class : ℝ)
  (total_marks_second_class : ℝ)
  (total_combined_marks : ℝ)
  (x : ℕ)
  (h1 : average_first_class = 50)
  (h2 : average_second_class = 65)
  (h3 : students_second_class = 40)
  (h4 : combined_average = 59.23076923076923)
  (h5 : total_students = x + 40)
  (h6 : total_marks_first_class = 50 * x)
  (h7 : total_marks_second_class = 65 * 40)
  (h8 : total_combined_marks = 59.23076923076923 * (x + 40))
  (h9 : total_marks_first_class + total_marks_second_class = total_combined_marks) :
  x = 25 :=
sorry

end number_students_first_class_l64_64762


namespace original_proposition_contrapositive_converse_inverse_negation_false_l64_64214

variable {a b c : ℝ}

-- Original Proposition
theorem original_proposition (h : a < b) : a + c < b + c :=
sorry

-- Contrapositive
theorem contrapositive (h : a + c >= b + c) : a >= b :=
sorry

-- Converse
theorem converse (h : a + c < b + c) : a < b :=
sorry

-- Inverse
theorem inverse (h : a >= b) : a + c >= b + c :=
sorry

-- Negation is false
theorem negation_false (h : a < b) : ¬ (a + c >= b + c) :=
sorry

end original_proposition_contrapositive_converse_inverse_negation_false_l64_64214


namespace apple_tree_yield_l64_64606

theorem apple_tree_yield (A : ℝ) 
    (h1 : Magdalena_picks_day1 = A / 5)
    (h2 : Magdalena_picks_day2 = 2 * (A / 5))
    (h3 : Magdalena_picks_day3 = (A / 5) + 20)
    (h4 : remaining_apples = 20)
    (total_picked : Magdalena_picks_day1 + Magdalena_picks_day2 + Magdalena_picks_day3 + remaining_apples = A)
    : A = 200 :=
by
    sorry

end apple_tree_yield_l64_64606


namespace b_is_nth_power_l64_64469

theorem b_is_nth_power (b n : ℕ) (h1 : b > 1) (h2 : n > 1) 
    (h3 : ∀ k > 1, ∃ a_k : ℕ, k ∣ (b - a_k^n)) : 
    ∃ A : ℕ, b = A^n :=
sorry

end b_is_nth_power_l64_64469


namespace product_of_smallest_primes_l64_64155

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l64_64155


namespace problem_statement_l64_64742

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end problem_statement_l64_64742


namespace train_length_eq_1800_l64_64935

theorem train_length_eq_1800 (speed_kmh : ℕ) (time_sec : ℕ) (distance : ℕ) (L : ℕ)
  (h_speed : speed_kmh = 216)
  (h_time : time_sec = 60)
  (h_distance : distance = 60 * time_sec)
  (h_total_distance : distance = 2 * L) :
  L = 1800 := by
  sorry

end train_length_eq_1800_l64_64935


namespace correct_statements_in_triangle_l64_64844

theorem correct_statements_in_triangle (a b c : ℝ) (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  (c = a * Real.cos B + b * Real.cos A) ∧ 
  (a^3 + b^3 = c^3 → a^2 + b^2 > c^2) :=
by
  sorry

end correct_statements_in_triangle_l64_64844


namespace g_at_neg1_l64_64268

-- Defining even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- Given functions f and g
variables (f g : ℝ → ℝ)

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^(1 - x)

-- Proof statement
theorem g_at_neg1 : g (-1) = -3 / 2 :=
by
  sorry

end g_at_neg1_l64_64268


namespace domain_of_f_l64_64420

-- Define the function domain transformation
theorem domain_of_f (f : ℝ → ℝ) : 
  (∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → -7 ≤ 2*x - 3 ∧ 2*x - 3 ≤ 1) ↔ (∀ (y : ℝ), -7 ≤ y ∧ y ≤ 1) :=
sorry

end domain_of_f_l64_64420


namespace find_C_l64_64006

def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

theorem find_C (C : ℝ × ℝ) (B : ℝ × ℝ) :
  -- Median condition: M is the midpoint of A and B
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  -- Given coordinates for A, M, L
  A = (2, 8) → M = (4, 11) → L = (6, 6) →
  -- Correct answer
  C = (14, 2) :=
by
  intros hmedian hA hM hL
  sorry

end find_C_l64_64006


namespace inequality_am_gm_holds_l64_64782

theorem inequality_am_gm_holds 
    (a b c : ℝ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (hc : c > 0) 
    (h : a^3 + b^3 = c^3) : 
  a^2 + b^2 - c^2 > 6 * (c - a) * (c - b) := 
sorry

end inequality_am_gm_holds_l64_64782


namespace ab_equals_six_l64_64185

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l64_64185


namespace problem_statement_l64_64422

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := x^n

variable (a : ℝ)
variable (h : a ≠ 1)

theorem problem_statement :
  (f 11 (f 13 a)) ^ 14 = f 2002 a ∧
  f 11 (f 13 (f 14 a)) = f 2002 a :=
by
  sorry

end problem_statement_l64_64422


namespace sum_of_nine_consecutive_even_integers_mod_10_l64_64856

theorem sum_of_nine_consecutive_even_integers_mod_10 : 
  (10112 + 10114 + 10116 + 10118 + 10120 + 10122 + 10124 + 10126 + 10128) % 10 = 0 := by
  sorry

end sum_of_nine_consecutive_even_integers_mod_10_l64_64856


namespace square_of_binomial_l64_64430

theorem square_of_binomial (c : ℝ) : (∃ b : ℝ, ∀ x : ℝ, 9 * x^2 - 30 * x + c = (3 * x + b)^2) ↔ c = 25 :=
by
  sorry

end square_of_binomial_l64_64430


namespace equal_heights_of_cylinder_and_cone_l64_64666

theorem equal_heights_of_cylinder_and_cone
  (r h : ℝ)
  (hc : h > 0)
  (hr : r > 0)
  (V_cylinder V_cone : ℝ)
  (V_cylinder_eq : V_cylinder = π * r ^ 2 * h)
  (V_cone_eq : V_cone = 1/3 * π * r ^ 2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
h = h := -- Since we are given that the heights are initially the same
sorry

end equal_heights_of_cylinder_and_cone_l64_64666


namespace alchemy_value_l64_64771

def letter_values : List Int :=
  [3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1,
  0, 1, 2, 3]

def char_value (c : Char) : Int :=
  letter_values.getD ((c.toNat - 'A'.toNat) % 13) 0

def word_value (s : String) : Int :=
  s.toList.map char_value |>.sum

theorem alchemy_value :
  word_value "ALCHEMY" = 8 :=
by
  sorry

end alchemy_value_l64_64771


namespace storks_more_than_birds_l64_64774

theorem storks_more_than_birds 
  (initial_birds : ℕ) 
  (joined_storks : ℕ) 
  (joined_birds : ℕ) 
  (h_init_birds : initial_birds = 3) 
  (h_joined_storks : joined_storks = 6) 
  (h_joined_birds : joined_birds = 2) : 
  (joined_storks - (initial_birds + joined_birds)) = 1 := 
by 
  -- Proof goes here
  sorry

end storks_more_than_birds_l64_64774


namespace men_complete_units_per_day_l64_64203

noncomputable def UnitsCompletedByMen (total_units : ℕ) (units_by_women : ℕ) : ℕ :=
  total_units - units_by_women

theorem men_complete_units_per_day :
  UnitsCompletedByMen 12 3 = 9 := by
  -- Proof skipped
  sorry

end men_complete_units_per_day_l64_64203


namespace marked_price_correct_l64_64755

noncomputable def marked_price (cost_price : ℝ) (profit_margin : ℝ) (selling_percentage : ℝ) : ℝ :=
  (cost_price * (1 + profit_margin)) / selling_percentage

theorem marked_price_correct :
  marked_price 1360 0.15 0.8 = 1955 :=
by
  sorry

end marked_price_correct_l64_64755


namespace how_many_grapes_l64_64255

-- Define the conditions given in the problem
def apples_to_grapes :=
  (3 / 4) * 12 = 6

-- Define the result to prove
def grapes_value :=
  (1 / 3) * 9 = 2

-- The statement combining the conditions and the problem to be proven
theorem how_many_grapes : apples_to_grapes → grapes_value :=
by
  intro h
  sorry

end how_many_grapes_l64_64255


namespace pond_water_after_evaporation_l64_64879

theorem pond_water_after_evaporation 
  (I R D : ℕ) 
  (h_initial : I = 250)
  (h_evaporation_rate : R = 1)
  (h_days : D = 50) : 
  I - (R * D) = 200 := 
by 
  sorry

end pond_water_after_evaporation_l64_64879


namespace greatest_value_k_l64_64291

theorem greatest_value_k (k : ℝ) (h : ∀ x : ℝ, (x - 1) ∣ (x^2 + 2*k*x - 3*k^2)) : k ≤ 1 :=
  by
  sorry

end greatest_value_k_l64_64291


namespace solution_comparison_l64_64588

open Real

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-(d / c) > -(f / e)) ↔ ((f / e) > (d / c)) :=
by
  sorry

end solution_comparison_l64_64588


namespace incorrect_option_D_l64_64140

variable {p q : Prop}

theorem incorrect_option_D (hp : ¬p) (hq : q) : ¬(¬q) := 
by 
  sorry  

end incorrect_option_D_l64_64140


namespace wendy_washing_loads_l64_64055

theorem wendy_washing_loads (shirts sweaters machine_capacity : ℕ) (total_clothes := shirts + sweaters) 
  (loads := total_clothes / machine_capacity) 
  (remainder := total_clothes % machine_capacity) 
  (h_shirts : shirts = 39) 
  (h_sweaters : sweaters = 33) 
  (h_machine_capacity : machine_capacity = 8) : loads = 9 ∧ remainder = 0 := 
by 
  sorry

end wendy_washing_loads_l64_64055


namespace solve_quadratic_roots_l64_64728

theorem solve_quadratic_roots : ∀ x : ℝ, (x - 1)^2 = 1 → (x = 2 ∨ x = 0) :=
by
  sorry

end solve_quadratic_roots_l64_64728


namespace domain_of_rational_func_l64_64193

noncomputable def rational_func (x : ℝ) : ℝ := (2 * x ^ 3 - 3 * x ^ 2 + 5 * x - 1) / (x ^ 2 - 5 * x + 6)

theorem domain_of_rational_func : 
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ↔ (∃ y : ℝ, rational_func y = x) :=
by
  sorry

end domain_of_rational_func_l64_64193


namespace total_amount_proof_l64_64582

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem total_amount_proof (x_ratio y_ratio z_ratio : ℝ) (y_share : ℝ) 
  (h1 : y_ratio = 0.45) (h2 : z_ratio = 0.50) (h3 : y_share = 54) 
  : total_amount (y_share / y_ratio) y_share (z_ratio * (y_share / y_ratio)) = 234 :=
by
  sorry

end total_amount_proof_l64_64582


namespace total_pay_XY_l64_64293

-- Assuming X's pay is 120% of Y's pay and Y's pay is 268.1818181818182,
-- Prove that the total pay to X and Y is 590.00.
theorem total_pay_XY (Y_pay : ℝ) (X_pay : ℝ) (total_pay : ℝ) :
  Y_pay = 268.1818181818182 →
  X_pay = 1.2 * Y_pay →
  total_pay = X_pay + Y_pay →
  total_pay = 590.00 :=
by
  intros hY hX hT
  sorry

end total_pay_XY_l64_64293


namespace matchstick_game_winner_a_matchstick_game_winner_b_l64_64067

def is_winning_position (pile1 pile2 : Nat) : Bool :=
  (pile1 % 2 = 1) && (pile2 % 2 = 1)

theorem matchstick_game_winner_a : is_winning_position 101 201 = true := 
by
  -- Theorem statement for (101 matches, 201 matches)
  -- The second player wins
  sorry

theorem matchstick_game_winner_b : is_winning_position 100 201 = false := 
by
  -- Theorem statement for (100 matches, 201 matches)
  -- The first player wins
  sorry

end matchstick_game_winner_a_matchstick_game_winner_b_l64_64067


namespace not_prime_n_quad_plus_n_sq_plus_one_l64_64653

theorem not_prime_n_quad_plus_n_sq_plus_one (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + n^2 + 1) :=
by
  sorry

end not_prime_n_quad_plus_n_sq_plus_one_l64_64653


namespace find_b_value_l64_64141

theorem find_b_value
    (k1 k2 b : ℝ)
    (y1 y2 : ℝ → ℝ)
    (a n : ℝ)
    (h1 : ∀ x, y1 x = k1 / x)
    (h2 : ∀ x, y2 x = k2 * x + b)
    (intersection_A : y1 1 = 4)
    (intersection_B : y2 a = 1 ∧ y1 a = 1)
    (translated_C_y1 : y1 (-1) = n + 6)
    (translated_C_y2 : y2 1 = n)
    (k1k2_nonzero : k1 ≠ 0 ∧ k2 ≠ 0)
    (sum_k1_k2 : k1 + k2 = 0) :
  b = -6 :=
sorry

end find_b_value_l64_64141


namespace triangle_acd_area_l64_64519

noncomputable def area_of_triangle : ℝ := sorry

theorem triangle_acd_area (AB CD : ℝ) (h : CD = 3 * AB) (area_trapezoid: ℝ) (h1: area_trapezoid = 20) :
  area_of_triangle = 15 := 
sorry

end triangle_acd_area_l64_64519


namespace solve_system_l64_64168

variable (y : ℝ) (x1 x2 x3 x4 x5 : ℝ)

def system_of_equations :=
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x3

theorem solve_system :
  (y = 2 → x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∧
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) →
   x1 + x2 + x3 + x4 + x5 = 0 ∧ ∀ (x1 x5 : ℝ), system_of_equations y x1 x2 x3 x4 x5) :=
sorry

end solve_system_l64_64168


namespace stefan_more_vail_l64_64751

/-- Aiguo had 20 seashells --/
def a : ℕ := 20

/-- Vail had 5 less seashells than Aiguo --/
def v : ℕ := a - 5

/-- The total number of seashells of Stefan, Vail, and Aiguo is 66 --/
def total_seashells (s v a : ℕ) : Prop := s + v + a = 66

theorem stefan_more_vail (s v a : ℕ)
  (h_a : a = 20)
  (h_v : v = a - 5)
  (h_total : total_seashells s v a) :
  s - v = 16 :=
by {
  -- proofs would go here
  sorry
}

end stefan_more_vail_l64_64751


namespace big_bea_bananas_l64_64930

theorem big_bea_bananas :
  ∃ (b : ℕ), (b + (b + 8) + (b + 16) + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 196) ∧ (b + 48 = 52) := by
  sorry

end big_bea_bananas_l64_64930


namespace percentage_increase_240_to_288_l64_64395

theorem percentage_increase_240_to_288 :
  let initial := 240
  let final := 288
  ((final - initial) / initial) * 100 = 20 := by 
  sorry

end percentage_increase_240_to_288_l64_64395


namespace negation_proposition_true_l64_64676

theorem negation_proposition_true (x : ℝ) : (¬ (|x| > 1 → x > 1)) ↔ (|x| ≤ 1 → x ≤ 1) :=
by sorry

end negation_proposition_true_l64_64676


namespace cos_double_angle_l64_64988

variable (α : ℝ)
variable (h : Real.cos α = 2/3)

theorem cos_double_angle : Real.cos (2 * α) = -1/9 :=
  by
  sorry

end cos_double_angle_l64_64988


namespace B_pow_150_l64_64599

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_150 : B ^ 150 = 1 :=
by
  sorry

end B_pow_150_l64_64599


namespace line_intersects_y_axis_at_5_l64_64484

theorem line_intersects_y_axis_at_5 :
  ∃ (b : ℝ), ∀ (x y : ℝ), (x - 2 = 0 ∧ y - 9 = 0) ∨ (x - 4 = 0 ∧ y - 13 = 0) →
  (y = 2 * x + b) ∧ (b = 5) :=
by
  sorry

end line_intersects_y_axis_at_5_l64_64484


namespace exists_set_no_three_ap_l64_64579

theorem exists_set_no_three_ap (n : ℕ) (k : ℕ) :
  (n ≥ 1983) →
  (k ≤ 100000) →
  ∃ S : Finset ℕ,
    S.card = n ∧
    (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → b ≠ (a + c) / 2) :=
sorry

end exists_set_no_three_ap_l64_64579


namespace ice_cream_flavors_l64_64083

-- We have four basic flavors and want to combine four scoops from these flavors.
def ice_cream_combinations : ℕ :=
  Nat.choose 7 3

theorem ice_cream_flavors : ice_cream_combinations = 35 :=
by
  sorry

end ice_cream_flavors_l64_64083


namespace fraction_subtraction_equivalence_l64_64210

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end fraction_subtraction_equivalence_l64_64210


namespace minimum_red_pieces_l64_64353

theorem minimum_red_pieces (w b r : ℕ) 
  (h1 : b ≤ w / 2) 
  (h2 : r ≥ 3 * b) 
  (h3 : w + b ≥ 55) : r = 57 := 
sorry

end minimum_red_pieces_l64_64353


namespace cards_traded_between_Padma_and_Robert_l64_64173

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end cards_traded_between_Padma_and_Robert_l64_64173


namespace favorable_probability_l64_64921

noncomputable def probability_favorable_events (L : ℝ) : ℝ :=
  1 - (0.5 * (5 / 12 * L)^2 / (0.5 * L^2))

theorem favorable_probability (L : ℝ) (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ L)
  (h3 : 0 ≤ y) (h4 : y ≤ L)
  (h5 : 0 ≤ x + y) (h6 : x + y ≤ L)
  (h7 : x ≤ 5 / 12 * L) (h8 : y ≤ 5 / 12 * L)
  (h9 : x + y ≥ 7 / 12 * L) :
  probability_favorable_events L = 15 / 16 :=
by sorry

end favorable_probability_l64_64921


namespace spadesuit_evaluation_l64_64602

def spadesuit (a b : ℝ) : ℝ := abs (a - b)

theorem spadesuit_evaluation : spadesuit 1.5 (spadesuit 2.5 (spadesuit 4.5 6)) = 0.5 :=
by
  sorry

end spadesuit_evaluation_l64_64602


namespace expected_revenue_day_14_plan_1_more_reasonable_plan_l64_64615

-- Define the initial conditions
def initial_valuation : ℕ := 60000
def rain_probability : ℚ := 0.4
def no_rain_probability : ℚ := 0.6
def hiring_cost : ℕ := 32000

-- Calculate the expected revenue if Plan ① is adopted
def expected_revenue_plan_1_day_14 : ℚ :=
  (initial_valuation / 10000) * (1/2 * rain_probability + no_rain_probability)

-- Calculate the total revenue for Plan ①
def total_revenue_plan_1 : ℚ :=
  (initial_valuation / 10000) + 2 * expected_revenue_plan_1_day_14

-- Calculate the total revenue for Plan ②
def total_revenue_plan_2 : ℚ :=
  3 * (initial_valuation / 10000) - (hiring_cost / 10000)

-- Define the lemmas to prove
theorem expected_revenue_day_14_plan_1 :
  expected_revenue_plan_1_day_14 = 4.8 := 
  by sorry

theorem more_reasonable_plan :
  total_revenue_plan_1 > total_revenue_plan_2 :=
  by sorry

end expected_revenue_day_14_plan_1_more_reasonable_plan_l64_64615


namespace find_M_l64_64516

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end find_M_l64_64516


namespace smallest_integer_y_l64_64972

theorem smallest_integer_y : ∃ y : ℤ, (8:ℚ) / 11 < y / 17 ∧ ∀ z : ℤ, ((8:ℚ) / 11 < z / 17 → y ≤ z) :=
by
  sorry

end smallest_integer_y_l64_64972


namespace min_value_of_expr_l64_64109

-- Define the expression
def expr (x y : ℝ) : ℝ := (x * y + 1)^2 + (x - y)^2

-- Statement to prove that the minimum value of the expression is 1
theorem min_value_of_expr : ∃ x y : ℝ, expr x y = 1 ∧ ∀ a b : ℝ, expr a b ≥ 1 :=
by
  -- Here the proof would be provided, but we leave it as sorry as per instructions.
  sorry

end min_value_of_expr_l64_64109


namespace max_min_values_l64_64538

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem max_min_values : 
  ∃ (max_val min_val : ℝ), 
    max_val = 7 ∧ min_val = -20 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max_val) ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min_val ≤ f x) := 
by
  sorry

end max_min_values_l64_64538


namespace sqrt_product_simplified_l64_64508

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l64_64508


namespace negation_divisible_by_5_is_odd_l64_64840

theorem negation_divisible_by_5_is_odd : 
  ¬∀ n : ℤ, (n % 5 = 0) → (n % 2 ≠ 0) ↔ ∃ n : ℤ, (n % 5 = 0) ∧ (n % 2 = 0) := 
by 
  sorry

end negation_divisible_by_5_is_odd_l64_64840


namespace f_increasing_on_interval_l64_64390

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
noncomputable def vec_b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

noncomputable def f (x t : ℝ) : ℝ :=
  let (a1, a2) := vec_a x
  let (b1, b2) := vec_b x t
  a1 * b1 + a2 * b2

noncomputable def f_prime (x t : ℝ) : ℝ :=
  2 * x - 3 * x^2 + t

theorem f_increasing_on_interval :
  ∀ t x, -1 < x → x < 1 → (0 ≤ f_prime x t) → (t ≥ 5) :=
sorry

end f_increasing_on_interval_l64_64390


namespace find_DY_length_l64_64024

noncomputable def angle_bisector_theorem (DE DY EF FY : ℝ) : ℝ :=
  (DE * FY) / EF

theorem find_DY_length :
  ∀ (DE EF FY : ℝ), DE = 26 → EF = 34 → FY = 30 →
  angle_bisector_theorem DE DY EF FY = 22.94 := 
by
  intros
  sorry

end find_DY_length_l64_64024


namespace batteries_difference_is_correct_l64_64337

-- Define the number of batteries used in each item
def flashlights_batteries : ℝ := 3.5
def toys_batteries : ℝ := 15.75
def remote_controllers_batteries : ℝ := 7.25
def wall_clock_batteries : ℝ := 4.8
def wireless_mouse_batteries : ℝ := 3.4

-- Define the combined total of batteries used in the other items
def combined_total : ℝ := flashlights_batteries + remote_controllers_batteries + wall_clock_batteries + wireless_mouse_batteries

-- Define the difference between the total number of batteries used in toys and the combined total of other items
def batteries_difference : ℝ := toys_batteries - combined_total

theorem batteries_difference_is_correct : batteries_difference = -3.2 :=
by
  sorry

end batteries_difference_is_correct_l64_64337


namespace problem_statement_l64_64961

-- Definitions of the conditions
variables (x y z w : ℕ)

-- The proof problem
theorem problem_statement
  (hx : x^3 = y^2)
  (hz : z^4 = w^3)
  (hzx : z - x = 17)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hw_pos : w > 0) :
  w - y = 229 :=
sorry

end problem_statement_l64_64961


namespace city_mpg_l64_64051

-- Define the conditions
variables {T H C : ℝ}
axiom cond1 : H * T = 560
axiom cond2 : (H - 6) * T = 336

-- The formal proof goal
theorem city_mpg : C = 9 :=
by
  have h1 : H = 560 / T := by sorry
  have h2 : (560 / T - 6) * T = 336 := by sorry
  have h3 : C = H - 6 := by sorry
  have h4 :  C = 9 := by sorry
  exact h4

end city_mpg_l64_64051


namespace arithmetic_sequence_minimum_value_S_n_l64_64731

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l64_64731


namespace find_sample_size_l64_64448

-- Definitions based on conditions
def ratio_students : ℕ := 2 + 3 + 5
def grade12_ratio : ℚ := 5 / ratio_students
def sample_grade12_students : ℕ := 150

-- The goal is to find n such that the proportion is maintained
theorem find_sample_size (n : ℕ) (h : grade12_ratio = sample_grade12_students / ↑n) : n = 300 :=
by sorry


end find_sample_size_l64_64448


namespace lowest_score_l64_64316

theorem lowest_score (score1 score2 : ℕ) (max_score : ℕ) (desired_mean : ℕ) (lowest_possible_score : ℕ) 
  (h_score1 : score1 = 82) (h_score2 : score2 = 75) (h_max_score : max_score = 100) (h_desired_mean : desired_mean = 85)
  (h_lowest_possible_score : lowest_possible_score = 83) : 
  ∃ x1 x2 : ℕ, x1 = max_score ∧ x2 = lowest_possible_score ∧ (score1 + score2 + x1 + x2) / 4 = desired_mean := by
  sorry

end lowest_score_l64_64316


namespace range_of_a_l64_64289

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 3) - 2 * a ≥ abs (x + a)) ↔ ( -3/2 ≤ a ∧ a < -1/2) := 
by sorry

end range_of_a_l64_64289


namespace no_fractional_solution_l64_64939

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l64_64939


namespace find_m_l64_64664

theorem find_m (a b m : ℤ) (h1 : a - b = 6) (h2 : a + b = 0) : 2 * a + b = m → m = 3 :=
by
  sorry

end find_m_l64_64664


namespace line_equation_l64_64257

theorem line_equation (b : ℝ) :
  (∃ b, (∀ x y, y = (3/4) * x + b) ∧ 
  (1/2) * |b| * |- (4/3) * b| = 6 →
  (3 * x - 4 * y + 12 = 0 ∨ 3 * x - 4 * y - 12 = 0)) := 
sorry

end line_equation_l64_64257


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l64_64405

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l64_64405


namespace tables_needed_l64_64455

-- Conditions
def n_invited : ℕ := 18
def n_no_show : ℕ := 12
def capacity_per_table : ℕ := 3

-- Calculation of attendees
def n_attendees : ℕ := n_invited - n_no_show

-- Proof for the number of tables needed
theorem tables_needed : (n_attendees / capacity_per_table) = 2 := by
  -- Sorry will be here to show it's incomplete
  sorry

end tables_needed_l64_64455


namespace infinite_radical_solution_l64_64634

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l64_64634


namespace unpainted_unit_cubes_l64_64397

theorem unpainted_unit_cubes (total_units : ℕ) (painted_per_face : ℕ) (painted_edges_adjustment : ℕ) :
  total_units = 216 → painted_per_face = 12 → painted_edges_adjustment = 36 → 
  total_units - (painted_per_face * 6 - painted_edges_adjustment) = 108 :=
by
  intros h_tot_units h_painted_face h_edge_adj
  sorry

end unpainted_unit_cubes_l64_64397


namespace tree_height_at_2_years_l64_64016

-- Define the conditions
def triples_height (height : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, height (n + 1) = 3 * height n

def height_at_5_years (height : ℕ → ℕ) : Prop :=
  height 5 = 243

-- Set up the problem statement
theorem tree_height_at_2_years (height : ℕ → ℕ) 
  (H1 : triples_height height) 
  (H2 : height_at_5_years height) : 
  height 2 = 9 :=
sorry

end tree_height_at_2_years_l64_64016


namespace find_t_l64_64176

open Real

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem find_t (t : ℝ) 
  (area_eq_50 : area_of_triangle 3 15 15 0 0 t = 50) :
  t = 325 / 12 ∨ t = 125 / 12 := 
sorry

end find_t_l64_64176


namespace simplify_fraction_l64_64363

theorem simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by {
  -- Proof skipped
  sorry
}

end simplify_fraction_l64_64363


namespace moles_of_HCl_formed_l64_64373

-- Conditions: 1 mole of Methane (CH₄) and 2 moles of Chlorine (Cl₂)
def methane := 1 -- 1 mole of methane
def chlorine := 2 -- 2 moles of chlorine

-- Reaction: CH₄ + Cl₂ → CH₃Cl + HCl
-- We state that 1 mole of methane reacts with 1 mole of chlorine to form 1 mole of hydrochloric acid
def reaction (methane chlorine : ℕ) : ℕ := methane

-- Theorem: Prove 1 mole of hydrochloric acid (HCl) is formed
theorem moles_of_HCl_formed : reaction methane chlorine = 1 := by
  sorry

end moles_of_HCl_formed_l64_64373


namespace find_c_l64_64078

-- Defining the given condition
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 + c

theorem find_c : (∃ c : ℝ, ∀ x : ℝ, parabola x c = 2 * x^2 + 1) :=
by 
  sorry

end find_c_l64_64078


namespace transformation_identity_l64_64466

theorem transformation_identity (n : Nat) (h : 2 ≤ n) : 
  n * Real.sqrt (n / (n ^ 2 - 1)) = Real.sqrt (n + n / (n ^ 2 - 1)) := 
sorry

end transformation_identity_l64_64466


namespace work_completion_days_l64_64940

theorem work_completion_days (a b : ℕ) (h1 : a + b = 6) (h2 : a + b = 15 / 4) : a = 6 :=
by
  sorry

end work_completion_days_l64_64940


namespace optimal_tower_configuration_l64_64655

theorem optimal_tower_configuration (x y : ℕ) (h : x + 2 * y = 30) :
    x * y ≤ 112 := by
  sorry

end optimal_tower_configuration_l64_64655


namespace two_m_plus_three_b_l64_64208

noncomputable def m : ℚ := (-(3/2) - (1/2)) / (2 - (-1))

noncomputable def b : ℚ := (1/2) - m * (-1)

theorem two_m_plus_three_b :
  2 * m + 3 * b = -11 / 6 :=
by
  sorry

end two_m_plus_three_b_l64_64208


namespace jasmine_total_cost_l64_64389

noncomputable def total_cost_jasmine
  (coffee_beans_amount : ℕ)
  (milk_amount : ℕ)
  (coffee_beans_cost : ℝ)
  (milk_cost : ℝ)
  (discount_combined : ℝ)
  (additional_discount_milk : ℝ)
  (tax_rate : ℝ) : ℝ :=
  let total_before_discounts := coffee_beans_amount * coffee_beans_cost + milk_amount * milk_cost
  let total_after_combined_discount := total_before_discounts - discount_combined * total_before_discounts
  let milk_cost_after_additional_discount := milk_amount * milk_cost - additional_discount_milk * (milk_amount * milk_cost)
  let total_after_all_discounts := coffee_beans_amount * coffee_beans_cost + milk_cost_after_additional_discount
  let tax := tax_rate * total_after_all_discounts
  total_after_all_discounts + tax

theorem jasmine_total_cost :
  total_cost_jasmine 4 2 2.50 3.50 0.10 0.05 0.08 = 17.98 :=
by
  unfold total_cost_jasmine
  sorry

end jasmine_total_cost_l64_64389


namespace number_of_correct_statements_l64_64244

theorem number_of_correct_statements:
  (¬∀ (a : ℝ), -a < 0) ∧
  (∀ (x : ℝ), |x| = -x → x < 0) ∧
  (∀ (a : ℚ), (∀ (b : ℚ), |b| ≥ |a|) → a = 0) ∧
  (∀ (x y : ℝ), 5 * x^2 * y ≠ 0 → 2 + 1 = 3) →
  2 = 2 := sorry

end number_of_correct_statements_l64_64244


namespace consecutive_product_neq_consecutive_even_product_l64_64842

open Nat

theorem consecutive_product_neq_consecutive_even_product :
  ∀ m n : ℕ, m * (m + 1) ≠ 4 * n * (n + 1) :=
by
  intros m n
  -- Proof is omitted, as per instructions.
  sorry

end consecutive_product_neq_consecutive_even_product_l64_64842


namespace product_zero_when_b_is_3_l64_64590

theorem product_zero_when_b_is_3 (b : ℤ) (h : b = 3) :
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) *
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by {
  sorry
}

end product_zero_when_b_is_3_l64_64590


namespace only_integers_square_less_than_three_times_l64_64753

-- We want to prove that the only integers n that satisfy n^2 < 3n are 1 and 2.
theorem only_integers_square_less_than_three_times (n : ℕ) (h : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
sorry

end only_integers_square_less_than_three_times_l64_64753


namespace jan_discount_percentage_l64_64459

theorem jan_discount_percentage :
  ∃ percent_discount : ℝ,
    ∀ (roses_bought dozen : ℕ) (rose_cost amount_paid : ℝ),
      roses_bought = 5 * dozen → dozen = 12 →
      rose_cost = 6 →
      amount_paid = 288 →
      (roses_bought * rose_cost - amount_paid) / (roses_bought * rose_cost) * 100 = percent_discount →
      percent_discount = 20 :=
by
  sorry

end jan_discount_percentage_l64_64459


namespace napkins_total_l64_64396

theorem napkins_total (o a w : ℕ) (ho : o = 10) (ha : a = 2 * o) (hw : w = 15) :
  w + o + a = 45 :=
by
  sorry

end napkins_total_l64_64396


namespace divides_power_diff_l64_64724

theorem divides_power_diff (x : ℤ) (y z w : ℕ) (hy : y % 2 = 1) (hz : z % 2 = 1) (hw : w % 2 = 1) : 17 ∣ x^(y^(z^w)) - x^(y^z) := 
by
  sorry

end divides_power_diff_l64_64724


namespace latte_price_l64_64007

theorem latte_price
  (almond_croissant_price salami_croissant_price plain_croissant_price focaccia_price total_spent : ℝ)
  (lattes_count : ℕ)
  (H1 : almond_croissant_price = 4.50)
  (H2 : salami_croissant_price = 4.50)
  (H3 : plain_croissant_price = 3.00)
  (H4 : focaccia_price = 4.00)
  (H5 : total_spent = 21.00)
  (H6 : lattes_count = 2) :
  (total_spent - (almond_croissant_price + salami_croissant_price + plain_croissant_price + focaccia_price)) / lattes_count = 2.50 :=
by
  -- skip the proof
  sorry

end latte_price_l64_64007


namespace next_divisor_after_391_l64_64391

theorem next_divisor_after_391 (m : ℕ) (h1 : m % 2 = 0) (h2 : m ≥ 1000 ∧ m < 10000) (h3 : 391 ∣ m) : 
  ∃ n, n > 391 ∧ n ∣ m ∧ (∀ k, k > 391 ∧ k < n → ¬ k ∣ m) ∧ n = 782 :=
sorry

end next_divisor_after_391_l64_64391


namespace find_m_value_l64_64101

theorem find_m_value
  (x y : ℤ)
  (h1 : x = 2)
  (h2 : y = m)
  (h3 : 3 * x + 2 * y = 10) : 
  m = 2 :=
by
  sorry

end find_m_value_l64_64101


namespace radius_smaller_circle_l64_64454

theorem radius_smaller_circle (A₁ A₂ A₃ : ℝ) (s : ℝ)
  (h1 : A₁ + A₂ = 12 * Real.pi)
  (h2 : A₃ = (Real.sqrt 3 / 4) * s^2)
  (h3 : 2 * A₂ = A₁ + A₁ + A₂ + A₃) :
  ∃ r : ℝ, r = Real.sqrt (6 - (Real.sqrt 3 / 8) * s^2) := by
  sorry

end radius_smaller_circle_l64_64454


namespace determine_exponent_l64_64820

theorem determine_exponent (m : ℕ) (hm : m > 0) (h_symm : ∀ x : ℝ, x^m - 3 = (-(x))^m - 3)
  (h_decr : ∀ (x y : ℝ), 0 < x ∧ x < y → x^m - 3 > y^m - 3) : m = 1 := 
sorry

end determine_exponent_l64_64820


namespace negation_of_existence_l64_64525

theorem negation_of_existence :
  ¬(∃ x : ℝ, x^2 + 2 * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l64_64525


namespace probability_selecting_cooking_l64_64182

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l64_64182


namespace min_max_diff_val_l64_64631

def find_min_max_diff (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : ℝ :=
  let m := 0
  let M := 1
  M - m

theorem min_max_diff_val (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : find_min_max_diff x y hx hy = 1 :=
by sorry

end min_max_diff_val_l64_64631


namespace greatest_b_value_for_integer_solution_eq_l64_64283

theorem greatest_b_value_for_integer_solution_eq : ∀ (b : ℤ), (∃ (x : ℤ), x^2 + b * x = -20) → b > 0 → b ≤ 21 :=
by
  sorry

end greatest_b_value_for_integer_solution_eq_l64_64283


namespace athlete_total_heartbeats_l64_64124

/-
  An athlete's heart rate starts at 140 beats per minute at the beginning of a race
  and increases by 5 beats per minute for each subsequent mile. How many times does
  the athlete's heart beat during a 10-mile race if the athlete runs at a pace of
  6 minutes per mile?
-/

def athlete_heartbeats (initial_rate : ℕ) (increase_rate : ℕ) (miles : ℕ) (minutes_per_mile : ℕ) : ℕ :=
  let n := miles
  let a := initial_rate
  let l := initial_rate + (increase_rate * (miles - 1))
  let S := (n * (a + l)) / 2
  S * minutes_per_mile

theorem athlete_total_heartbeats :
  athlete_heartbeats 140 5 10 6 = 9750 :=
sorry

end athlete_total_heartbeats_l64_64124


namespace probability_of_union_l64_64644

def total_cards : ℕ := 52
def king_of_hearts : ℕ := 1
def spades : ℕ := 13

theorem probability_of_union :
  let P_A := king_of_hearts / total_cards
  let P_B := spades / total_cards
  (P_A + P_B) = (7 / 26) :=
by
  sorry

end probability_of_union_l64_64644


namespace leak_drain_time_l64_64379

/-- Statement: Given the rates at which a pump fills a tank and a leak drains the tank, 
prove that the leak can drain all the water in the tank in 14 hours. -/
theorem leak_drain_time :
  (∀ P L: ℝ, P = 1/2 → (P - L) = 3/7 → L = 1/14 → (1 / L) = 14) := 
by
  intros P L hP hPL hL
  -- Proof is omitted (to be provided)
  sorry

end leak_drain_time_l64_64379


namespace max_2x_plus_y_value_l64_64786

open Real

def on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 4 + P.2^2 = 1)

def max_value_2x_plus_y (P : ℝ × ℝ) (h : on_ellipse P) : ℝ := 
  2 * P.1 + P.2

theorem max_2x_plus_y_value (P : ℝ × ℝ) (h : on_ellipse P):
  ∃ (m : ℝ), max_value_2x_plus_y P h = m ∧ m = sqrt 17 :=
sorry

end max_2x_plus_y_value_l64_64786


namespace problem_solution_l64_64320

-- Definitions and Assumptions
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, f x - (deriv^[2]) f x > 0)

-- Statement to Prove
theorem problem_solution : e * f 2015 > f 2016 :=
by
  sorry

end problem_solution_l64_64320


namespace number_of_sides_of_polygon_l64_64945

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l64_64945


namespace Marty_combinations_l64_64340

theorem Marty_combinations:
  let colors := ({blue, green, yellow, black, white} : Finset String)
  let tools := ({brush, roller, sponge, spray_gun} : Finset String)
  colors.card * tools.card = 20 := 
by
  sorry

end Marty_combinations_l64_64340


namespace sufficient_but_not_necessary_l64_64401

-- Define the conditions
def abs_value_condition (x : ℝ) : Prop := |x| < 2
def quadratic_condition (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Theorem statement
theorem sufficient_but_not_necessary : (∀ x : ℝ, abs_value_condition x → quadratic_condition x) ∧ ¬ (∀ x : ℝ, quadratic_condition x → abs_value_condition x) :=
by
  sorry

end sufficient_but_not_necessary_l64_64401


namespace max_singular_words_l64_64386

theorem max_singular_words (alphabet_length : ℕ) (word_length : ℕ) (strip_length : ℕ) 
  (num_non_overlapping_pieces : ℕ) (h_alphabet : alphabet_length = 25)
  (h_word_length : word_length = 17) (h_strip_length : strip_length = 5^18)
  (h_non_overlapping : num_non_overlapping_pieces = 5^16) : 
  ∃ max_singular_words, max_singular_words = 2 * 5^17 :=
by {
  -- proof to be completed
  sorry
}

end max_singular_words_l64_64386


namespace hannah_dogs_food_total_l64_64206

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end hannah_dogs_food_total_l64_64206


namespace nested_radical_solution_l64_64170

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l64_64170


namespace gcd_437_323_eq_19_l64_64804

theorem gcd_437_323_eq_19 : Int.gcd 437 323 = 19 := 
by 
  sorry

end gcd_437_323_eq_19_l64_64804


namespace resistance_of_one_rod_l64_64733

section RodResistance

variables (R_0 R : ℝ)

-- Given: the resistance of the entire construction is 8 Ω
def entire_construction_resistance : Prop := R = 8

-- Given: formula for the equivalent resistance
def equivalent_resistance_formula : Prop := R = 4 / 10 * R_0

-- To prove: the resistance of one rod is 20 Ω
theorem resistance_of_one_rod 
  (h1 : entire_construction_resistance R)
  (h2 : equivalent_resistance_formula R_0 R) :
  R_0 = 20 :=
sorry

end RodResistance

end resistance_of_one_rod_l64_64733


namespace min_expression_value_l64_64766

theorem min_expression_value (a b c : ℝ) (h_sum : a + b + c = -1) (h_abc : a * b * c ≤ -3) :
  3 ≤ (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) :=
sorry

end min_expression_value_l64_64766


namespace acute_angle_l64_64650

variables (x : ℝ)

def a : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem acute_angle (x : ℝ) : 
  (-2 / 3 < x) → x ≠ -2 / 3 → dot_product (2, x) (1, 3) > 0 :=
by
  intros h1 h2
  sorry

end acute_angle_l64_64650


namespace percentage_of_second_discount_is_correct_l64_64216

def car_original_price : ℝ := 12000
def first_discount : ℝ := 0.20
def final_price_after_discounts : ℝ := 7752
def third_discount : ℝ := 0.05

def solve_percentage_second_discount : Prop := 
  ∃ (second_discount : ℝ), 
    (car_original_price * (1 - first_discount) * (1 - second_discount) * (1 - third_discount) = final_price_after_discounts) ∧ 
    (second_discount * 100 = 15)

theorem percentage_of_second_discount_is_correct : solve_percentage_second_discount :=
  sorry

end percentage_of_second_discount_is_correct_l64_64216


namespace sqrt_condition_iff_l64_64419

theorem sqrt_condition_iff (x : ℝ) : (∃ y : ℝ, y = (2 * x + 3) ∧ (0 ≤ y)) ↔ (x ≥ -3 / 2) :=
by sorry

end sqrt_condition_iff_l64_64419


namespace stella_toilet_paper_packs_l64_64693

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end stella_toilet_paper_packs_l64_64693


namespace farmer_total_profit_l64_64463

theorem farmer_total_profit :
  let group1_revenue := 3 * 375
  let group1_cost := (8 * 13 + 3 * 15) * 3
  let group1_profit := group1_revenue - group1_cost

  let group2_revenue := 4 * 425
  let group2_cost := (5 * 14 + 9 * 16) * 4
  let group2_profit := group2_revenue - group2_cost

  let group3_revenue := 2 * 475
  let group3_cost := (10 * 15 + 8 * 18) * 2
  let group3_profit := group3_revenue - group3_cost

  let group4_revenue := 1 * 550
  let group4_cost := 20 * 20 * 1
  let group4_profit := group4_revenue - group4_cost

  let total_profit := group1_profit + group2_profit + group3_profit + group4_profit
  total_profit = 2034 :=
by
  sorry

end farmer_total_profit_l64_64463


namespace expenditure_representation_correct_l64_64181

-- Define the representation of income
def income_representation (income : ℝ) : ℝ :=
  income

-- Define the representation of expenditure
def expenditure_representation (expenditure : ℝ) : ℝ :=
  -expenditure

-- Condition: an income of 10.5 yuan is represented as +10.5 yuan.
-- We need to prove: an expenditure of 6 yuan is represented as -6 yuan.
theorem expenditure_representation_correct (h : income_representation 10.5 = 10.5) : 
  expenditure_representation 6 = -6 :=
by
  sorry

end expenditure_representation_correct_l64_64181


namespace g_function_property_l64_64687

variable {g : ℝ → ℝ}
variable {a b : ℝ}

theorem g_function_property 
  (h1 : ∀ a c : ℝ, c^3 * g a = a^3 * g c)
  (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 :=
  sorry

end g_function_property_l64_64687


namespace negation_of_prop_l64_64754

theorem negation_of_prop :
  ¬ (∀ x : ℝ, x^2 - 1 > 0) ↔ ∃ x : ℝ, x^2 - 1 ≤ 0 :=
sorry

end negation_of_prop_l64_64754


namespace football_cost_is_correct_l64_64053

def total_spent_on_toys : ℝ := 12.30
def spent_on_marbles : ℝ := 6.59
def spent_on_football := total_spent_on_toys - spent_on_marbles

theorem football_cost_is_correct : spent_on_football = 5.71 :=
by
  sorry

end football_cost_is_correct_l64_64053


namespace minimum_trucks_needed_l64_64201

theorem minimum_trucks_needed {n : ℕ} (total_weight : ℕ) (box_weight : ℕ → ℕ) (truck_capacity : ℕ) :
  (total_weight = 10 ∧ truck_capacity = 3 ∧ (∀ b, box_weight b ≤ 1) ∧ (∃ n, 3 * n ≥ total_weight)) → n ≥ 5 :=
by
  -- We need to prove the statement based on the given conditions.
  sorry

end minimum_trucks_needed_l64_64201


namespace range_of_m_l64_64727

theorem range_of_m (m : ℝ)
  (h₁ : (m^2 - 4) ≥ 0)
  (h₂ : (4 * (m - 2)^2 - 16) < 0) :
  1 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l64_64727


namespace combined_weight_is_correct_l64_64905

-- Define the conditions
def elephant_weight_tons : ℕ := 3
def ton_in_pounds : ℕ := 2000
def donkey_weight_percentage : ℕ := 90

-- Convert elephant's weight to pounds
def elephant_weight_pounds : ℕ := elephant_weight_tons * ton_in_pounds

-- Calculate the donkeys's weight
def donkey_weight_pounds : ℕ := elephant_weight_pounds - (elephant_weight_pounds * donkey_weight_percentage / 100)

-- Define the combined weight
def combined_weight : ℕ := elephant_weight_pounds + donkey_weight_pounds

-- Prove the combined weight is 6600 pounds
theorem combined_weight_is_correct : combined_weight = 6600 :=
by
  sorry

end combined_weight_is_correct_l64_64905


namespace serpent_ridge_trail_length_l64_64292

/-- Phoenix hiked the Serpent Ridge Trail last week. It took her five days to complete the trip.
The first two days she hiked a total of 28 miles. The second and fourth days she averaged 15 miles per day.
The last three days she hiked a total of 42 miles. The total hike for the first and third days was 30 miles.
How many miles long was the trail? -/
theorem serpent_ridge_trail_length
  (a b c d e : ℕ)
  (h1 : a + b = 28)
  (h2 : b + d = 30)
  (h3 : c + d + e = 42)
  (h4 : a + c = 30) :
  a + b + c + d + e = 70 :=
sorry

end serpent_ridge_trail_length_l64_64292


namespace find_A_l64_64229

theorem find_A (J : ℤ := 15)
  (JAVA_pts : ℤ := 50)
  (AJAX_pts : ℤ := 53)
  (AXLE_pts : ℤ := 40)
  (L : ℤ := 12)
  (JAVA_eq : ∀ A V : ℤ, 2 * A + V + J = JAVA_pts)
  (AJAX_eq : ∀ A X : ℤ, 2 * A + X + J = AJAX_pts)
  (AXLE_eq : ∀ A X E : ℤ, A + X + L + E = AXLE_pts) : A = 21 :=
sorry

end find_A_l64_64229


namespace solve_for_x_l64_64573

theorem solve_for_x (x : ℝ) (h : (x - 15) / 3 = (3 * x + 10) / 8) : x = -150 := 
by
  sorry

end solve_for_x_l64_64573


namespace integer_triples_condition_l64_64893

theorem integer_triples_condition (p q r : ℤ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) 
  (h4 : ((p - 1) * (q - 1) * (r - 1)) ∣ (p * q * r - 1)) : (p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15) :=
sorry

end integer_triples_condition_l64_64893


namespace find_number_l64_64049

theorem find_number (n : ℕ) (h₁ : ∀ x : ℕ, 21 + 7 * x = n ↔ 3 + x = 47):
  n = 329 :=
by
  -- Proof will go here
  sorry

end find_number_l64_64049


namespace geo_seq_bn_plus_2_general_formula_an_l64_64480

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 2 = 4
axiom h3 : ∀ n, b n = a (n + 1) - a n
axiom h4 : ∀ n, b (n + 1) = 2 * b n + 2

-- Proof goals
theorem geo_seq_bn_plus_2 : (∀ n, ∃ r : ℕ, b n + 2 = 4 * 2^n) :=
  sorry

theorem general_formula_an : (∀ n, a n = 2^(n + 1) - 2 * n) :=
  sorry

end geo_seq_bn_plus_2_general_formula_an_l64_64480


namespace relationship_among_a_b_c_l64_64567

noncomputable def a := (1/2)^(2/3)
noncomputable def b := (1/5)^(2/3)
noncomputable def c := (1/2)^(1/3)

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l64_64567


namespace compound_interest_rate_l64_64306

theorem compound_interest_rate (
  P : ℝ) (r : ℝ)  (A : ℕ → ℝ) :
  A 2 = 2420 ∧ A 3 = 3025 ∧ 
  (∀ n : ℕ, A n = P * (1 + r / 100)^n) → r = 25 :=
by
  sorry

end compound_interest_rate_l64_64306


namespace f_5_5_l64_64145

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_even (x : ℝ) : f x = f (-x) := sorry

lemma f_recurrence (x : ℝ) : f (x + 2) = - (1 / f x) := sorry

lemma f_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) : f x = x := sorry

theorem f_5_5 : f 5.5 = 2.5 :=
by
  sorry

end f_5_5_l64_64145


namespace arithmetic_sequence_50th_term_l64_64377

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 7
  let n := 50
  (a_1 + (n - 1) * d) = 346 :=
by
  let a_1 := 3
  let d := 7
  let n := 50
  show (a_1 + (n - 1) * d) = 346
  sorry

end arithmetic_sequence_50th_term_l64_64377


namespace average_production_last_5_days_l64_64645

theorem average_production_last_5_days
  (avg_first_25_days : ℕ → ℕ → ℕ → ℕ → Prop)
  (avg_monthly : ℕ)
  (total_days : ℕ)
  (days_first_period : ℕ)
  (avg_production_first_period : ℕ)
  (avg_total_monthly : ℕ)
  (days_second_period : ℕ)
  (total_production_five_days : ℕ):
  (days_first_period = 25) →
  (avg_production_first_period = 50) →
  (avg_total_monthly = 48) →
  (total_production_five_days = 190) →
  (days_second_period = 5) →
  avg_first_25_days days_first_period avg_production_first_period 
  (days_first_period * avg_production_first_period) avg_total_monthly ∧
  avg_monthly = avg_total_monthly →
  ((days_first_period + days_second_period) * avg_monthly - 
  days_first_period * avg_production_first_period = total_production_five_days) →
  (total_production_five_days / days_second_period = 38) := sorry

end average_production_last_5_days_l64_64645


namespace area_of_rhombus_is_375_l64_64127

-- define the given diagonals
def diagonal1 := 25
def diagonal2 := 30

-- define the formula for the area of a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

-- state the theorem
theorem area_of_rhombus_is_375 : area_of_rhombus diagonal1 diagonal2 = 375 := 
by 
  -- The proof is omitted as per the requirement
  sorry

end area_of_rhombus_is_375_l64_64127


namespace equilateral_triangle_area_in_circle_l64_64580

theorem equilateral_triangle_area_in_circle (r : ℝ) (h : r = 9) :
  let s := 2 * r * Real.sin (π / 3)
  let A := (Real.sqrt 3 / 4) * s^2
  A = (243 * Real.sqrt 3) / 4 := by
  sorry

end equilateral_triangle_area_in_circle_l64_64580


namespace multiples_6_8_not_both_l64_64035

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l64_64035


namespace example_one_example_two_l64_64490

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l64_64490


namespace B_subset_A_A_inter_B_empty_l64_64544

-- Definitions for the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}

-- Proof statement for Part (1)
theorem B_subset_A (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (-1 / 2 < a ∧ a < 1) := sorry

-- Proof statement for Part (2)
theorem A_inter_B_empty (a : ℝ) : (∀ x, ¬(x ∈ A ∧ x ∈ B a)) ↔ (a ≤ -4 ∨ a ≥ 2) := sorry

end B_subset_A_A_inter_B_empty_l64_64544


namespace exists_integers_not_all_zero_l64_64326

-- Given conditions
variables (a b c : ℝ)
variables (ab bc ca : ℚ)
variables (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca)
variables (x y z : ℤ)

-- The theorem to prove
theorem exists_integers_not_all_zero (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca):
  ∃ (x y z : ℤ), (¬ (x = 0 ∧ y = 0 ∧ z = 0)) ∧ (a * x + b * y + c * z = 0) :=
sorry

end exists_integers_not_all_zero_l64_64326


namespace distance_between_points_l64_64106

def point1 : ℝ × ℝ := (3.5, -2)
def point2 : ℝ × ℝ := (7.5, 5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 65 := by
  sorry

end distance_between_points_l64_64106


namespace factor_expression_l64_64282

theorem factor_expression (b : ℚ) : 
  294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) :=
by 
  sorry

end factor_expression_l64_64282


namespace probability_two_tails_two_heads_l64_64366

theorem probability_two_tails_two_heads :
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  total_probability = 3 / 8 :=
by
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  sorry

end probability_two_tails_two_heads_l64_64366


namespace area_of_black_region_l64_64854

theorem area_of_black_region (side_small side_large : ℕ) 
  (h1 : side_small = 5) 
  (h2 : side_large = 9) : 
  (side_large * side_large) - (side_small * side_small) = 56 := 
by
  sorry

end area_of_black_region_l64_64854


namespace fraction_of_pelicans_moved_l64_64425

-- Conditions
variables (P : ℕ)
variables (n_Sharks : ℕ := 60) -- Number of sharks in Pelican Bay
variables (n_Pelicans_original : ℕ := 2 * P) -- Twice the original number of Pelicans in Shark Bite Cove
variables (n_Pelicans_remaining : ℕ := 20) -- Number of remaining Pelicans in Shark Bite Cove

-- Proof to show fraction that moved
theorem fraction_of_pelicans_moved (h : 2 * P = n_Sharks) : (P - n_Pelicans_remaining) / P = 1 / 3 :=
by {
  sorry
}

end fraction_of_pelicans_moved_l64_64425


namespace kayla_less_than_vika_l64_64453

variable (S K V : ℕ)
variable (h1 : S = 216)
variable (h2 : S = 4 * K)
variable (h3 : V = 84)

theorem kayla_less_than_vika (S K V : ℕ) (h1 : S = 216) (h2 : S = 4 * K) (h3 : V = 84) : V - K = 30 :=
by
  sorry

end kayla_less_than_vika_l64_64453


namespace result_when_j_divided_by_26_l64_64342

noncomputable def j := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 10 11) (Nat.lcm 12 13)) (Nat.lcm 14 15))

theorem result_when_j_divided_by_26 : j / 26 = 2310 := by 
  sorry

end result_when_j_divided_by_26_l64_64342


namespace sum_first_ten_terms_arithmetic_l64_64707

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l64_64707


namespace difference_of_solutions_l64_64792

theorem difference_of_solutions (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ∃ a b : ℝ, a ≠ b ∧ (x = a ∨ x = b) ∧ abs (a - b) = 22 :=
by
  sorry

end difference_of_solutions_l64_64792


namespace Craig_walk_distance_l64_64775

/-- Craig walked some distance from school to David's house and 0.7 miles from David's house to his own house. 
In total, Craig walked 0.9 miles. Prove that the distance Craig walked from school to David's house is 0.2 miles. 
--/
theorem Craig_walk_distance (d_school_David d_David_Craig d_total : ℝ) 
  (h1 : d_David_Craig = 0.7) 
  (h2 : d_total = 0.9) : 
  d_school_David = 0.2 :=
by 
  sorry

end Craig_walk_distance_l64_64775


namespace average_marks_of_a_b_c_d_l64_64578

theorem average_marks_of_a_b_c_d (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : A = 43)
  (h3 : (B + C + D + E) / 4 = 48)
  (h4 : E = D + 3) :
  (A + B + C + D) / 4 = 47 :=
by
  -- This theorem will be justified
  admit

end average_marks_of_a_b_c_d_l64_64578


namespace height_on_hypotenuse_correct_l64_64286

noncomputable def height_on_hypotenuse (a b : ℝ) (ha : a = 3) (hb : b = 4) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let area := (a * b) / 2
  (2 * area) / c

theorem height_on_hypotenuse_correct (h : ℝ) : 
  height_on_hypotenuse 3 4 rfl rfl = 12 / 5 :=
by
  sorry

end height_on_hypotenuse_correct_l64_64286


namespace y_increase_by_41_8_units_l64_64550

theorem y_increase_by_41_8_units :
  ∀ (x y : ℝ),
    (∀ k : ℝ, y = 2 + k * 11 / 5 → x = 1 + k * 5) →
    x = 20 → y = 41.8 :=
by
  sorry

end y_increase_by_41_8_units_l64_64550


namespace prime_cubed_plus_seven_composite_l64_64646

theorem prime_cubed_plus_seven_composite (P : ℕ) (hP_prime : Nat.Prime P) (hP3_plus_5_prime : Nat.Prime (P ^ 3 + 5)) : ¬ Nat.Prime (P ^ 3 + 7) :=
by
  sorry

end prime_cubed_plus_seven_composite_l64_64646


namespace PartA_l64_64417

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x, f (f x) = f x)

theorem PartA : ∀ x, (deriv f x = 0) ∨ (deriv f (f x) = 1) :=
by
  sorry

end PartA_l64_64417


namespace sequence_G_51_l64_64951

theorem sequence_G_51 :
  ∀ G : ℕ → ℚ, 
  (∀ n : ℕ, G (n + 1) = (3 * G n + 2) / 2) → 
  G 1 = 3 → 
  G 51 = (3^51 + 1) / 2 := by 
  sorry

end sequence_G_51_l64_64951


namespace train_speed_l64_64872

noncomputable def speed_of_train (length_of_train length_of_overbridge time: ℝ) : ℝ :=
  (length_of_train + length_of_overbridge) / time

theorem train_speed (length_of_train length_of_overbridge time speed: ℝ)
  (h1 : length_of_train = 600)
  (h2 : length_of_overbridge = 100)
  (h3 : time = 70)
  (h4 : speed = 10) :
  speed_of_train length_of_train length_of_overbridge time = speed :=
by
  simp [speed_of_train, h1, h2, h3, h4]
  sorry

end train_speed_l64_64872


namespace a_pow_a_b_pow_b_c_pow_c_ge_one_l64_64554

theorem a_pow_a_b_pow_b_c_pow_c_ge_one
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a + b + c = Real.rpow a (1/7) + Real.rpow b (1/7) + Real.rpow c (1/7)) :
    a^a * b^b * c^c ≥ 1 := 
by
  sorry

end a_pow_a_b_pow_b_c_pow_c_ge_one_l64_64554


namespace remainder_7623_div_11_l64_64075

theorem remainder_7623_div_11 : 7623 % 11 = 0 := 
by sorry

end remainder_7623_div_11_l64_64075


namespace measure_of_angle4_l64_64657

def angle1 := 62
def angle2 := 36
def angle3 := 24
def angle4 : ℕ := 122

theorem measure_of_angle4 (d e : ℕ) (h1 : angle1 + angle2 + angle3 + d + e = 180) (h2 : d + e = 58) :
  angle4 = 180 - (angle1 + angle2 + angle3 + d + e) :=
by
  sorry

end measure_of_angle4_l64_64657


namespace exists_airline_route_within_same_republic_l64_64861

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l64_64861


namespace abs_five_minus_sqrt_pi_l64_64143

theorem abs_five_minus_sqrt_pi : |5 - Real.sqrt Real.pi| = 3.22755 := by
  sorry

end abs_five_minus_sqrt_pi_l64_64143


namespace intersection_S_T_l64_64192

def S : Set ℝ := { y | y ≥ 0 }
def T : Set ℝ := { x | x > 1 }

theorem intersection_S_T :
  S ∩ T = { z | z > 1 } :=
sorry

end intersection_S_T_l64_64192


namespace max_value_a_l64_64345

noncomputable def setA (a : ℝ) : Set ℝ := { x | (x - 1) * (x - a) ≥ 0 }
noncomputable def setB (a : ℝ) : Set ℝ := { x | x ≥ a - 1 }

theorem max_value_a (a : ℝ) :
  (setA a ∪ setB a = Set.univ) → a ≤ 2 := by
  sorry

end max_value_a_l64_64345


namespace sufficient_but_not_necessary_not_necessary_l64_64502

theorem sufficient_but_not_necessary (x y : ℝ) (h : x < y ∧ y < 0) : x^2 > y^2 :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

theorem not_necessary (x y : ℝ) (h : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

end sufficient_but_not_necessary_not_necessary_l64_64502


namespace polynomial_simplification_l64_64256

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 9 * x - 8) + (-x^5 + x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 14) = 
  -x^5 + 3 * x^4 + x^3 - x^2 + 3 * x + 6 :=
by
  sorry

end polynomial_simplification_l64_64256


namespace turns_in_two_hours_l64_64548

theorem turns_in_two_hours (turns_per_30_sec : ℕ) (minutes_in_hour : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → 
  minutes_in_hour = 60 → 
  hours = 2 → 
  (12 * (minutes_in_hour * hours)) = 1440 := 
by
  sorry

end turns_in_two_hours_l64_64548


namespace trajectory_of_P_l64_64545

open Real

-- Definitions of points F1 and F2
def F1 : (ℝ × ℝ) := (-4, 0)
def F2 : (ℝ × ℝ) := (4, 0)

-- Definition of the condition on moving point P
def satisfies_condition (P : (ℝ × ℝ)) : Prop :=
  abs (dist P F2 - dist P F1) = 4

-- Definition of the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1 ∧ x ≤ -2

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, satisfies_condition P → ∃ x y : ℝ, P = (x, y) ∧ hyperbola_equation x y :=
by
  sorry

end trajectory_of_P_l64_64545


namespace total_earnings_proof_l64_64061

-- Definitions of the given conditions
def monthly_earning : ℕ := 4000
def monthly_saving : ℕ := 500
def total_savings_needed : ℕ := 45000

-- Lean statement for the proof problem
theorem total_earnings_proof : 
  (total_savings_needed / monthly_saving) * monthly_earning = 360000 :=
by
  sorry

end total_earnings_proof_l64_64061


namespace intersection_unique_point_x_coordinate_l64_64551

theorem intersection_unique_point_x_coordinate (a b : ℝ) (h : a ≠ b) : 
  (∃ x y : ℝ, y = x^2 + 2*a*x + 6*b ∧ y = x^2 + 2*b*x + 6*a) → ∃ x : ℝ, x = 3 :=
by
  sorry

end intersection_unique_point_x_coordinate_l64_64551


namespace radius_of_fourth_circle_is_12_l64_64159

theorem radius_of_fourth_circle_is_12 (r : ℝ) (radii : Fin 7 → ℝ) 
  (h_geometric : ∀ i, radii (Fin.succ i) = r * radii i) 
  (h_smallest : radii 0 = 6)
  (h_largest : radii 6 = 24) :
  radii 3 = 12 :=
by
  sorry

end radius_of_fourth_circle_is_12_l64_64159


namespace group_card_exchanges_l64_64989

theorem group_card_exchanges (x : ℕ) (hx : x * (x - 1) = 90) : x = 10 :=
by { sorry }

end group_card_exchanges_l64_64989


namespace solution_set_of_inequality_l64_64947

theorem solution_set_of_inequality (x : ℝ) (h : |x - 1| < 1) : 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l64_64947


namespace eval_expression_l64_64559

theorem eval_expression (x y z : ℝ) (h1 : y > z) (h2 : z > 0) (h3 : x = y + z) : 
  ( (y+z+y)^z + (y+z+z)^y ) / (y^z + z^y) = 2^y + 2^z :=
by
  sorry

end eval_expression_l64_64559


namespace binomial_coefficient_fourth_term_l64_64064

theorem binomial_coefficient_fourth_term (n k : ℕ) (hn : n = 5) (hk : k = 3) : Nat.choose n k = 10 := by
  sorry

end binomial_coefficient_fourth_term_l64_64064


namespace num_digits_of_prime_started_numerals_l64_64997

theorem num_digits_of_prime_started_numerals (n : ℕ) (h : 4 * 10^(n-1) = 400) : n = 3 := 
  sorry

end num_digits_of_prime_started_numerals_l64_64997


namespace woman_wait_time_to_be_caught_l64_64924

theorem woman_wait_time_to_be_caught 
  (man_speed_mph : ℝ) (woman_speed_mph : ℝ) (wait_time_minutes : ℝ) 
  (conversion_factor : ℝ) (distance_apart_miles : ℝ) :
  man_speed_mph = 6 →
  woman_speed_mph = 12 →
  wait_time_minutes = 10 →
  conversion_factor = 1 / 60 →
  distance_apart_miles = (woman_speed_mph * conversion_factor) * wait_time_minutes →
  ∃ minutes_to_catch_up : ℝ, minutes_to_catch_up = distance_apart_miles / (man_speed_mph * conversion_factor) ∧ minutes_to_catch_up = 20 := sorry

end woman_wait_time_to_be_caught_l64_64924


namespace problem1_problem2_l64_64756

variable {α : ℝ}

-- Given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 3

-- Proof statements to be shown
theorem problem1 (h : tan_alpha α) : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by sorry

theorem problem2 (h : tan_alpha α) : Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6 :=
by sorry

end problem1_problem2_l64_64756


namespace expressible_numbers_count_l64_64272

theorem expressible_numbers_count : ∃ k : ℕ, k = 2222 ∧ ∀ n : ℕ, n ≤ 2000 → ∃ x : ℝ, n = Int.floor x + Int.floor (3 * x) + Int.floor (5 * x) :=
by sorry

end expressible_numbers_count_l64_64272


namespace stratified_sampling_number_of_grade12_students_in_sample_l64_64536

theorem stratified_sampling_number_of_grade12_students_in_sample 
  (total_students : ℕ)
  (students_grade10 : ℕ)
  (students_grade11_minus_grade12 : ℕ)
  (sampled_students_grade10 : ℕ)
  (total_students_eq : total_students = 1290)
  (students_grade10_eq : students_grade10 = 480)
  (students_grade11_minus_grade12_eq : students_grade11_minus_grade12 = 30)
  (sampled_students_grade10_eq : sampled_students_grade10 = 96) :
  ∃ n : ℕ, n = 78 :=
by
  -- Proof would go here, but we are skipping with "sorry"
  sorry

end stratified_sampling_number_of_grade12_students_in_sample_l64_64536


namespace cost_to_paint_cube_l64_64136

def side_length := 30 -- in feet
def cost_per_kg := 40 -- Rs. per kg
def coverage_per_kg := 20 -- sq. ft. per kg

def area_of_one_face := side_length * side_length
def total_surface_area := 6 * area_of_one_face
def paint_required := total_surface_area / coverage_per_kg
def total_cost := paint_required * cost_per_kg

theorem cost_to_paint_cube : total_cost = 10800 := 
by
  -- proof here would follow the solution steps provided in the solution part, which are omitted
  sorry

end cost_to_paint_cube_l64_64136


namespace xyz_sum_is_22_l64_64906

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end xyz_sum_is_22_l64_64906


namespace scaled_triangle_height_l64_64901

theorem scaled_triangle_height (h b₁ h₁ b₂ h₂ : ℝ)
  (h₁_eq : h₁ = 6) (b₁_eq : b₁ = 12) (b₂_eq : b₂ = 8) :
  (b₁ / h₁ = b₂ / h₂) → h₂ = 4 :=
by
  -- Given conditions
  have h₁_eq : h₁ = 6 := h₁_eq
  have b₁_eq : b₁ = 12 := b₁_eq
  have b₂_eq : b₂ = 8 := b₂_eq
  -- Proof will go here
  sorry

end scaled_triangle_height_l64_64901


namespace cos_double_angle_unit_circle_l64_64114

theorem cos_double_angle_unit_circle (α y₀ : ℝ) (h : (1/2)^2 + y₀^2 = 1) : 
  Real.cos (2 * α) = -1/2 :=
by 
  -- The proof is omitted
  sorry

end cos_double_angle_unit_circle_l64_64114


namespace people_per_team_l64_64354

theorem people_per_team 
  (managers : ℕ) (employees : ℕ) (teams : ℕ) 
  (h1 : managers = 23) (h2 : employees = 7) (h3 : teams = 6) :
  (managers + employees) / teams = 5 :=
by
  sorry

end people_per_team_l64_64354


namespace lcm_of_18_and_30_l64_64307

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l64_64307


namespace fifth_term_arithmetic_sequence_l64_64716

-- Conditions provided
def first_term (x y : ℝ) := x + y^2
def second_term (x y : ℝ) := x - y^2
def third_term (x y : ℝ) := x - 3*y^2
def fourth_term (x y : ℝ) := x - 5*y^2

-- Proof to determine the fifth term
theorem fifth_term_arithmetic_sequence (x y : ℝ) :
  (fourth_term x y) - (third_term x y) = -2*y^2 →
  (x - 5 * y^2) - 2 * y^2 = x - 7 * y^2 :=
by sorry

end fifth_term_arithmetic_sequence_l64_64716


namespace find_k_l64_64802

variables {x k : ℝ}

theorem find_k (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (h2 : k ≠ 0) : k = 8 :=
sorry

end find_k_l64_64802


namespace constant_abs_difference_l64_64929

variable (a : ℕ → ℝ)

-- Define the condition for the recurrence relation
def recurrence_relation : Prop := ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n

-- State the theorem
theorem constant_abs_difference (h : recurrence_relation a) : ∃ C : ℝ, ∀ n ≥ 2, |(a n)^2 - (a (n-1)) * (a (n+1))| = C :=
    sorry

end constant_abs_difference_l64_64929


namespace find_k_l64_64234

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the condition for vectors to be parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Translate the problem condition
def problem_condition (k : ℝ) : Prop :=
  let lhs := (k * a.1 + b.1, k * a.2 + b.2)
  let rhs := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  is_parallel lhs rhs

-- The goal is to find k such that the condition holds
theorem find_k : problem_condition (-1/3) :=
by
  sorry

end find_k_l64_64234


namespace no_int_solutions_l64_64222

theorem no_int_solutions (c x y : ℤ) (h1 : 0 < c) (h2 : c % 2 = 1) : x ^ 2 - y ^ 3 ≠ (2 * c) ^ 3 - 1 :=
sorry

end no_int_solutions_l64_64222


namespace correct_ordering_of_powers_l64_64325

theorem correct_ordering_of_powers : 
  7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 :=
  by
    sorry

end correct_ordering_of_powers_l64_64325


namespace oxygen_atom_count_l64_64086

-- Definitions and conditions
def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def molecular_weight_O : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def total_molecular_weight : ℝ := 65.0

-- Theorem statement
theorem oxygen_atom_count : 
  ∃ (num_oxygen_atoms : ℕ), 
  num_oxygen_atoms * molecular_weight_O = total_molecular_weight - (num_carbon_atoms * molecular_weight_C + num_hydrogen_atoms * molecular_weight_H) 
  ∧ num_oxygen_atoms = 1 :=
by
  sorry

end oxygen_atom_count_l64_64086


namespace xena_head_start_l64_64926

theorem xena_head_start
  (xena_speed : ℝ) (dragon_speed : ℝ) (time : ℝ) (burn_distance : ℝ) 
  (xena_speed_eq : xena_speed = 15) 
  (dragon_speed_eq : dragon_speed = 30) 
  (time_eq : time = 32) 
  (burn_distance_eq : burn_distance = 120) :
  (dragon_speed * time - burn_distance) - (xena_speed * time) = 360 := 
  by 
  sorry

end xena_head_start_l64_64926


namespace order_y1_y2_y3_l64_64126

-- Defining the parabolic function and the points A, B, C
def parabola (a x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

-- Points A, B, C
def y1 (a : ℝ) : ℝ := parabola a (-1)
def y2 (a : ℝ) : ℝ := parabola a 2
def y3 (a : ℝ) : ℝ := parabola a 4

-- Assumption: a > 0
variables (a : ℝ) (h : a > 0)

-- The theorem to prove
theorem order_y1_y2_y3 : 
  y2 a < y1 a ∧ y1 a < y3 a :=
sorry

end order_y1_y2_y3_l64_64126


namespace angle_sum_l64_64865

theorem angle_sum (y : ℝ) (h : 3 * y + y = 120) : y = 30 :=
sorry

end angle_sum_l64_64865


namespace arithmetic_sequence_a5_l64_64812

theorem arithmetic_sequence_a5 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : S 4 = 16)
  (h_sum : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h_a : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  a 5 = 9 :=
by 
  sorry

end arithmetic_sequence_a5_l64_64812


namespace distance_from_dormitory_to_city_l64_64587

theorem distance_from_dormitory_to_city (D : ℝ) (h : (1/2) * D + (1/4) * D + 6 = D) : D = 24 :=
by
  sorry

end distance_from_dormitory_to_city_l64_64587


namespace marigold_ratio_l64_64245

theorem marigold_ratio :
  ∃ x, 14 + 25 + x = 89 ∧ x / 25 = 2 := by
  sorry

end marigold_ratio_l64_64245


namespace pages_left_in_pad_l64_64990

-- Definitions from conditions
def total_pages : ℕ := 120
def science_project_pages (total : ℕ) : ℕ := total * 25 / 100
def math_homework_pages : ℕ := 10

-- Proving the final number of pages left
theorem pages_left_in_pad :
  let remaining_pages_after_usage := total_pages - science_project_pages total_pages - math_homework_pages
  let pages_left_after_art_project := remaining_pages_after_usage / 2
  pages_left_after_art_project = 40 :=
by
  sorry

end pages_left_in_pad_l64_64990


namespace correct_operation_l64_64364

noncomputable def check_operations : Prop :=
    ∀ (a : ℝ), ( a^6 / a^3 = a^3 ) ∧ 
               ¬( 3 * a^5 + a^5 = 4 * a^10 ) ∧
               ¬( (2 * a)^3 = 2 * a^3 ) ∧
               ¬( (a^2)^4 = a^6 )

theorem correct_operation : check_operations :=
by
  intro a
  have h1 : a^6 / a^3 = a^3 := by
    sorry
  have h2 : ¬(3 * a^5 + a^5 = 4 * a^10) := by
    sorry
  have h3 : ¬((2 * a)^3 = 2 * a^3) := by
    sorry
  have h4 : ¬((a^2)^4 = a^6) := by
    sorry
  exact ⟨h1, h2, h3, h4⟩

end correct_operation_l64_64364


namespace naomi_wash_time_l64_64577

theorem naomi_wash_time (C T S : ℕ) (h₁ : T = 2 * C) (h₂ : S = 2 * C - 15) (h₃ : C + T + S = 135) : C = 30 :=
by
  sorry

end naomi_wash_time_l64_64577


namespace value_of_coins_l64_64957

theorem value_of_coins (n d : ℕ) (hn : n + d = 30)
    (hv : 10 * n + 5 * d = 5 * n + 10 * d + 90) :
    300 - 5 * n = 180 := by
  sorry

end value_of_coins_l64_64957


namespace num_carnations_l64_64916

-- Define the conditions
def num_roses : ℕ := 5
def total_flowers : ℕ := 10

-- Define the statement we want to prove
theorem num_carnations : total_flowers - num_roses = 5 :=
by {
  -- The proof itself is not required, so we use 'sorry' to indicate incomplete proof
  sorry
}

end num_carnations_l64_64916


namespace seeds_in_fourth_pot_l64_64616

theorem seeds_in_fourth_pot (total_seeds : ℕ) (total_pots : ℕ) (seeds_per_pot : ℕ) (first_three_pots : ℕ)
  (h1 : total_seeds = 10) (h2 : total_pots = 4) (h3 : seeds_per_pot = 3) (h4 : first_three_pots = 3) : 
  (total_seeds - (seeds_per_pot * first_three_pots)) = 1 :=
by
  sorry

end seeds_in_fourth_pot_l64_64616


namespace solve_for_y_l64_64004

theorem solve_for_y :
  ∃ y : ℚ, 2 * y + 3 * y = 200 - (4 * y + (10 * y / 2)) ∧ y = 100 / 7 :=
by {
  -- Assertion only, proof is not required as per instructions.
  sorry
}

end solve_for_y_l64_64004


namespace total_team_formation_plans_l64_64928

def numberOfWaysToChooseDoctors (m f : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose m (k - 1) * Nat.choose f 1) +
  (Nat.choose m 1 * Nat.choose f (k - 1))

theorem total_team_formation_plans :
  let m := 5
  let f := 4
  let total := 3
  numberOfWaysToChooseDoctors m f total = 70 :=
by
  let m := 5
  let f := 4
  let total := 3
  unfold numberOfWaysToChooseDoctors
  sorry

end total_team_formation_plans_l64_64928


namespace radius_of_spheres_in_cone_l64_64436

-- Given Definitions
def cone_base_radius : ℝ := 6
def cone_height : ℝ := 15
def tangent_spheres (r : ℝ) : Prop :=
  r = (12 * Real.sqrt 29) / 29

-- Problem Statement
theorem radius_of_spheres_in_cone :
  ∃ r : ℝ, tangent_spheres r :=
sorry

end radius_of_spheres_in_cone_l64_64436


namespace solve_quadratic_equation_solve_linear_factor_equation_l64_64349

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x^2 - 6 * x + 1 = 0 → (x = 3 - 2 * Real.sqrt 2 ∨ x = 3 + 2 * Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

theorem solve_linear_factor_equation :
  ∀ (x : ℝ), x * (2 * x - 1) = 2 * (2 * x - 1) → (x = 1 / 2 ∨ x = 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_equation_solve_linear_factor_equation_l64_64349


namespace P_shape_points_length_10_l64_64025

def P_shape_points (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem P_shape_points_length_10 :
  P_shape_points 10 = 31 := 
by 
  sorry

end P_shape_points_length_10_l64_64025


namespace rectangle_perimeter_eq_circle_circumference_l64_64725

theorem rectangle_perimeter_eq_circle_circumference (l : ℝ) :
  2 * (l + 3) = 10 * Real.pi -> l = 5 * Real.pi - 3 :=
by
  intro h
  sorry

end rectangle_perimeter_eq_circle_circumference_l64_64725


namespace min_polyline_distance_between_circle_and_line_l64_64627

def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

def on_line (Q : ℝ × ℝ) : Prop :=
  2 * Q.1 + Q.2 = 2 * Real.sqrt 5

theorem min_polyline_distance_between_circle_and_line :
  ∃ P Q, on_circle P ∧ on_line Q ∧ polyline_distance P Q = (Real.sqrt 5) / 2 :=
by
  sorry

end min_polyline_distance_between_circle_and_line_l64_64627


namespace common_elements_count_l64_64300

theorem common_elements_count (S T : Set ℕ) (hS : S = {n | ∃ k : ℕ, k < 3000 ∧ n = 5 * (k + 1)})
    (hT : T = {n | ∃ k : ℕ, k < 3000 ∧ n = 8 * (k + 1)}) :
    S ∩ T = {n | ∃ m : ℕ, m < 375 ∧ n = 40 * (m + 1)} :=
by {
  sorry
}

end common_elements_count_l64_64300


namespace largest_integer_value_of_x_l64_64991

theorem largest_integer_value_of_x (x : ℤ) (h : 8 - 5 * x > 22) : x ≤ -3 :=
sorry

end largest_integer_value_of_x_l64_64991


namespace brother_birth_year_1990_l64_64526

variable (current_year : ℕ) -- Assuming the current year is implicit for the problem, it should be 2010 if Karina is 40 years old.
variable (karina_birth_year : ℕ)
variable (karina_current_age : ℕ)
variable (brother_current_age : ℕ)
variable (karina_twice_of_brother : Prop)

def karinas_brother_birth_year (karina_birth_year karina_current_age brother_current_age : ℕ) : ℕ :=
  karina_birth_year + brother_current_age

theorem brother_birth_year_1990 
  (h1 : karina_birth_year = 1970) 
  (h2 : karina_current_age = 40) 
  (h3 : karina_twice_of_brother) : 
  karinas_brother_birth_year 1970 40 20 = 1990 := 
by
  sorry

end brother_birth_year_1990_l64_64526


namespace logarithmic_relationship_l64_64262

theorem logarithmic_relationship
  (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (h5 : m = Real.log c / Real.log a)
  (h6 : n = Real.log c / Real.log b)
  (h7 : r = a ^ c) :
  n < m ∧ m < r :=
sorry

end logarithmic_relationship_l64_64262


namespace profit_per_package_l64_64827

theorem profit_per_package
  (packages_first_center_per_day : ℕ)
  (packages_second_center_multiplier : ℕ)
  (weekly_profit : ℕ)
  (days_per_week : ℕ)
  (H1 : packages_first_center_per_day = 10000)
  (H2 : packages_second_center_multiplier = 3)
  (H3 : weekly_profit = 14000)
  (H4 : days_per_week = 7) :
  (weekly_profit / (packages_first_center_per_day * days_per_week + 
                    packages_second_center_multiplier * packages_first_center_per_day * days_per_week) : ℝ) = 0.05 :=
by
  sorry

end profit_per_package_l64_64827


namespace simply_connected_polyhedron_faces_l64_64793

def polyhedron_faces_condition (σ3 σ4 σ5 : Nat) (V E F : Nat) : Prop :=
  V - E + F = 2

theorem simply_connected_polyhedron_faces : 
  ∀ (σ3 σ4 σ5 : Nat) (V E F : Nat),
  polyhedron_faces_condition σ3 σ4 σ5 V E F →
  (σ4 = 0 ∧ σ5 = 0 → σ3 ≥ 4) ∧
  (σ3 = 0 ∧ σ5 = 0 → σ4 ≥ 6) ∧
  (σ3 = 0 ∧ σ4 = 0 → σ5 ≥ 12) := 
by
  intros
  sorry

end simply_connected_polyhedron_faces_l64_64793


namespace complement_union_eq_l64_64198

-- Define the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement of a set within another set
def complement (S T : Set ℕ) : Set ℕ := { x | x ∈ S ∧ x ∉ T }

-- Define the union of M and N
def union_M_N : Set ℕ := {x | x ∈ M ∨ x ∈ N}

-- State the theorem
theorem complement_union_eq :
  complement U union_M_N = {4} :=
sorry

end complement_union_eq_l64_64198


namespace find_t_l64_64815

noncomputable def a_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a (n + 1) = 3 * a n + 3 ^ n

noncomputable def b_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) : Prop :=
  ∀ n : ℕ, b n = (a (n + 1) + t) / 3^(n + 1)

theorem find_t (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) :
  a_sequence a →
  b_sequence a b t →
  (∀ n : ℕ, (b (n + 1) - b n) = (b 1 - b 0)) →
  t = -1 / 2 :=
by
  sorry

end find_t_l64_64815


namespace adam_change_l64_64374

-- Defining the given amount Adam has and the cost of the airplane.
def amountAdamHas : ℝ := 5.00
def costOfAirplane : ℝ := 4.28

-- Statement of the theorem to be proven.
theorem adam_change : amountAdamHas - costOfAirplane = 0.72 := by
  sorry

end adam_change_l64_64374


namespace compute_expression_l64_64881

theorem compute_expression : 10 * (3 / 27) * 36 = 40 := 
by 
  sorry

end compute_expression_l64_64881


namespace christmas_bonus_remainder_l64_64258

theorem christmas_bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l64_64258


namespace roots_quad_sum_abs_gt_four_sqrt_three_l64_64236

theorem roots_quad_sum_abs_gt_four_sqrt_three
  (p r1 r2 : ℝ)
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 12)
  (h3 : p^2 > 48) : 
  |r1 + r2| > 4 * Real.sqrt 3 := 
by 
  sorry

end roots_quad_sum_abs_gt_four_sqrt_three_l64_64236


namespace min_ab_l64_64823

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a * b = a + b + 3) : a * b ≥ 9 :=
sorry

end min_ab_l64_64823


namespace baker_sold_cakes_l64_64888

def initialCakes : Nat := 110
def additionalCakes : Nat := 76
def remainingCakes : Nat := 111
def cakesSold : Nat := 75

theorem baker_sold_cakes :
  initialCakes + additionalCakes - remainingCakes = cakesSold := by
  sorry

end baker_sold_cakes_l64_64888


namespace find_n_l64_64918

theorem find_n (n : ℤ) 
  (h : (3 + 16 + 33 + (n + 1)) / 4 = 20) : n = 27 := 
by
  sorry

end find_n_l64_64918


namespace max_k_value_l64_64976

theorem max_k_value (m : ℝ) (h : 0 < m ∧ m < 1/2) : 
  ∃ k : ℝ, (∀ m, 0 < m ∧ m < 1/2 → (1 / m + 2 / (1 - 2 * m)) ≥ k) ∧ k = 8 :=
by sorry

end max_k_value_l64_64976


namespace inequality_solution_subset_l64_64128

theorem inequality_solution_subset {x a : ℝ} : (∀ x, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 :=
by sorry

end inequality_solution_subset_l64_64128


namespace betty_eggs_used_l64_64157

-- Conditions as definitions
def ratio_sugar_cream_cheese (sugar cream_cheese : ℚ) : Prop :=
  sugar / cream_cheese = 1 / 4

def ratio_vanilla_cream_cheese (vanilla cream_cheese : ℚ) : Prop :=
  vanilla / cream_cheese = 1 / 2

def ratio_eggs_vanilla (eggs vanilla : ℚ) : Prop :=
  eggs / vanilla = 2

-- Given conditions
def sugar_used : ℚ := 2 -- cups of sugar

-- The statement to prove
theorem betty_eggs_used (cream_cheese vanilla eggs : ℚ) 
  (h1 : ratio_sugar_cream_cheese sugar_used cream_cheese)
  (h2 : ratio_vanilla_cream_cheese vanilla cream_cheese)
  (h3 : ratio_eggs_vanilla eggs vanilla) :
  eggs = 8 :=
sorry

end betty_eggs_used_l64_64157


namespace smartphone_charging_time_l64_64841

theorem smartphone_charging_time :
  ∀ (T S : ℕ), T = 53 → T + (1 / 2 : ℚ) * S = 66 → S = 26 :=
by
  intros T S hT equation
  sorry

end smartphone_charging_time_l64_64841


namespace sum_max_min_interval_l64_64886

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem sum_max_min_interval (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1) :
  let M := max (f a) (f b)
  let m := min (f a) (f b)
  M + m = 6 :=
by
  rw [h₁, h₂]
  let M := max (f (-1)) (f 1)
  let m := min (f (-1)) (f 1)
  sorry

end sum_max_min_interval_l64_64886


namespace intersecting_point_value_l64_64418

theorem intersecting_point_value (c d : ℤ) (h1 : d = 5 * (-5) + c) (h2 : -5 = 5 * d + c) : 
  d = -5 := 
sorry

end intersecting_point_value_l64_64418


namespace inverse_undefined_at_one_l64_64355

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

theorem inverse_undefined_at_one : ∀ (x : ℝ), (x = 1) → ¬∃ y : ℝ, f y = x :=
by
  sorry

end inverse_undefined_at_one_l64_64355


namespace marla_night_cost_is_correct_l64_64654

def lizard_value_bc := 8 -- 1 lizard is worth 8 bottle caps
def lizard_value_gw := 5 / 3 -- 3 lizards are worth 5 gallons of water
def horse_value_gw := 80 -- 1 horse is worth 80 gallons of water
def marla_daily_bc := 20 -- Marla can scavenge 20 bottle caps each day
def marla_days := 24 -- It takes Marla 24 days to collect the bottle caps

noncomputable def marla_night_cost_bc : ℕ :=
((marla_daily_bc * marla_days) - (horse_value_gw / lizard_value_gw * (3 * lizard_value_bc))) / marla_days

theorem marla_night_cost_is_correct :
  marla_night_cost_bc = 4 := by
  sorry

end marla_night_cost_is_correct_l64_64654


namespace edward_garage_sale_games_l64_64139

variables
  (G_total : ℕ) -- total number of games
  (G_good : ℕ) -- number of good games
  (G_bad : ℕ) -- number of bad games
  (G_friend : ℕ) -- number of games bought from a friend
  (G_garage : ℕ) -- number of games bought at the garage sale

-- The conditions
def total_games (G_total : ℕ) (G_good : ℕ) (G_bad : ℕ) : Prop :=
  G_total = G_good + G_bad

def garage_sale_games (G_total : ℕ) (G_friend : ℕ) (G_garage : ℕ) : Prop :=
  G_total = G_friend + G_garage

-- The theorem to be proved
theorem edward_garage_sale_games
  (G_total : ℕ) 
  (G_good : ℕ) 
  (G_bad : ℕ)
  (G_friend : ℕ) 
  (G_garage : ℕ) 
  (h1 : total_games G_total G_good G_bad)
  (h2 : G_good = 24)
  (h3 : G_bad = 31)
  (h4 : G_friend = 41) :
  G_garage = 14 :=
by
  sorry

end edward_garage_sale_games_l64_64139


namespace solution_set_of_inequality_l64_64296

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end solution_set_of_inequality_l64_64296


namespace total_spending_is_correct_l64_64826

def total_spending : ℝ :=
  let meal_expenses_10 := 10 * 18
  let meal_expenses_5 := 5 * 25
  let total_meal_expenses := meal_expenses_10 + meal_expenses_5
  let service_charge := 50
  let total_before_discount := total_meal_expenses + service_charge
  let discount := 0.05 * total_meal_expenses
  let total_after_discount := total_before_discount - discount
  let tip := 0.10 * total_before_discount
  total_after_discount + tip

theorem total_spending_is_correct : total_spending = 375.25 :=
by
  sorry

end total_spending_is_correct_l64_64826


namespace poster_height_proportion_l64_64739

-- Defining the given conditions
def original_width : ℕ := 3
def original_height : ℕ := 2
def new_width : ℕ := 12
def scale_factor := new_width / original_width

-- The statement to prove the new height
theorem poster_height_proportion :
  scale_factor = 4 → (original_height * scale_factor) = 8 :=
by
  sorry

end poster_height_proportion_l64_64739


namespace symmetric_point_A_is_B_l64_64246

/-
  Define the symmetric point function for reflecting a point across the origin.
  Define the coordinate of point A.
  Assert that the symmetric point of A has coordinates (-2, 6).
-/

structure Point where
  x : ℤ
  y : ℤ

def symmetric_point (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def A : Point := ⟨2, -6⟩

def B : Point := ⟨-2, 6⟩

theorem symmetric_point_A_is_B : symmetric_point A = B := by
  sorry

end symmetric_point_A_is_B_l64_64246


namespace polar_center_coordinates_l64_64264

-- Define polar coordinate system equation
def polar_circle (ρ θ : ℝ) := ρ = 2 * Real.sin θ

-- Define the theorem: Given the equation of a circle in polar coordinates, its center in polar coordinates.
theorem polar_center_coordinates :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi → ∃ ρ, polar_circle ρ θ) →
  (∀ ρ θ, polar_circle ρ θ → 0 ≤ θ ∧ θ < 2 * Real.pi → (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = -1 ∧ θ = 3 * Real.pi / 2)) :=
by {
  sorry 
}

end polar_center_coordinates_l64_64264


namespace gummies_remain_l64_64699

theorem gummies_remain
  (initial_candies : ℕ)
  (sibling_candies_per : ℕ)
  (num_siblings : ℕ)
  (best_friend_fraction : ℝ)
  (cousin_fraction : ℝ)
  (kept_candies : ℕ)
  (result : ℕ)
  (h_initial : initial_candies = 500)
  (h_sibling_candies_per : sibling_candies_per = 35)
  (h_num_siblings : num_siblings = 3)
  (h_best_friend_fraction : best_friend_fraction = 0.5)
  (h_cousin_fraction : cousin_fraction = 0.25)
  (h_kept_candies : kept_candies = 50)
  (h_result : result = 99) : 
  (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋ - 
  ⌊cousin_fraction * (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋)⌋ 
  - kept_candies) = result := 
by {
  sorry
}

end gummies_remain_l64_64699


namespace percentage_favoring_all_three_l64_64066

variable (A B C A_union_B_union_C Y X : ℝ)

-- Conditions
axiom hA : A = 0.50
axiom hB : B = 0.30
axiom hC : C = 0.20
axiom hA_union_B_union_C : A_union_B_union_C = 0.78
axiom hY : Y = 0.17

-- Question: Prove that the percentage of those asked favoring all three proposals is 5%
theorem percentage_favoring_all_three :
  A = 0.50 → B = 0.30 → C = 0.20 →
  A_union_B_union_C = 0.78 →
  Y = 0.17 →
  X = 0.05 :=
by
  intros
  sorry

end percentage_favoring_all_three_l64_64066


namespace quadratic_has_real_root_l64_64105

theorem quadratic_has_real_root (p : ℝ) : 
  ∃ x : ℝ, 3 * (p + 2) * x^2 - p * x - (4 * p + 7) = 0 :=
sorry

end quadratic_has_real_root_l64_64105


namespace find_principal_l64_64470

theorem find_principal
  (P R : ℝ)
  (h : (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140) :
  P = 1000 := by
sorry

end find_principal_l64_64470


namespace simplify_polynomial_expression_l64_64031

noncomputable def polynomial_expression (x : ℝ) := 
  (3 * x^3 + x^2 - 5 * x + 9) * (x + 2) - (x + 2) * (2 * x^3 - 4 * x + 8) + (x^2 - 6 * x + 13) * (x + 2) * (x - 3)

theorem simplify_polynomial_expression (x : ℝ) :
  polynomial_expression x = 2 * x^4 + x^3 + 9 * x^2 + 23 * x + 2 :=
sorry

end simplify_polynomial_expression_l64_64031


namespace system1_solution_system2_solution_l64_64447

-- Statement for the Part 1 Equivalent Problem.
theorem system1_solution :
  ∀ (x y : ℤ),
    (x - 3 * y = -10) ∧ (x + y = 6) → (x = 2 ∧ y = 4) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

-- Statement for the Part 2 Equivalent Problem.
theorem system2_solution :
  ∀ (x y : ℚ),
    (x / 2 - (y - 1) / 3 = 1) ∧ (4 * x - y = 8) → (x = 12 / 5 ∧ y = 8 / 5) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

end system1_solution_system2_solution_l64_64447


namespace max_k_condition_l64_64324

theorem max_k_condition (k : ℕ) (total_goods : ℕ) (num_platforms : ℕ) (platform_capacity : ℕ) :
  total_goods = 1500 ∧ num_platforms = 25 ∧ platform_capacity = 80 → 
  (∀ (c : ℕ), 1 ≤ c ∧ c ≤ k → c ∣ k) → 
  (∀ (total : ℕ), total ≤ num_platforms * platform_capacity → total ≥ total_goods) → 
  k ≤ 26 := 
sorry

end max_k_condition_l64_64324


namespace frequency_of_2_l64_64400

def num_set := "20231222"
def total_digits := 8
def count_of_2 := 5

theorem frequency_of_2 : (count_of_2 : ℚ) / total_digits = 5 / 8 := by
  sorry

end frequency_of_2_l64_64400


namespace min_value_part1_l64_64166

open Real

theorem min_value_part1 (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by {
  sorry
}

end min_value_part1_l64_64166


namespace combined_weight_l64_64870

theorem combined_weight (S R : ℕ) (h1 : S = 71) (h2 : S - 5 = 2 * R) : S + R = 104 := by
  sorry

end combined_weight_l64_64870


namespace mn_value_l64_64694

variables {x m n : ℝ} -- Define variables x, m, n as real numbers

theorem mn_value (h : x^2 + m * x - 15 = (x + 3) * (x + n)) : m * n = 10 :=
by {
  -- Sorry for skipping the proof steps
  sorry
}

end mn_value_l64_64694


namespace hexagon_side_lengths_l64_64026

theorem hexagon_side_lengths (n : ℕ) (h1 : n ≥ 0) (h2 : n ≤ 6) (h3 : 10 * n + 8 * (6 - n) = 56) : n = 4 :=
sorry

end hexagon_side_lengths_l64_64026


namespace farm_total_amount_90000_l64_64612

-- Defining the conditions
def apples_produce (mangoes: ℕ) : ℕ := 2 * mangoes
def oranges_produce (mangoes: ℕ) : ℕ := mangoes + 200

-- Defining the total produce of all fruits
def total_produce (mangoes: ℕ) : ℕ := apples_produce mangoes + mangoes + oranges_produce mangoes

-- Defining the price per kg
def price_per_kg : ℕ := 50

-- Defining the total amount from selling all fruits
noncomputable def total_amount (mangoes: ℕ) : ℕ := total_produce mangoes * price_per_kg

-- Proving that the total amount he got in that season is $90,000
theorem farm_total_amount_90000 : total_amount 400 = 90000 := by
  sorry

end farm_total_amount_90000_l64_64612


namespace polygon_sides_l64_64252

theorem polygon_sides (n : Nat) (h : (360 : ℝ) / (180 * (n - 2)) = 2 / 9) : n = 11 :=
by
  sorry

end polygon_sides_l64_64252


namespace length_of_DF_l64_64748

theorem length_of_DF
  (D E F P Q: Type)
  (DP: ℝ)
  (EQ: ℝ)
  (h1: DP = 27)
  (h2: EQ = 36)
  (perp: ∀ (u v: Type), u ≠ v):
  ∃ (DF: ℝ), DF = 4 * Real.sqrt 117 :=
by
  sorry

end length_of_DF_l64_64748


namespace median_of_36_consecutive_integers_l64_64768

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l64_64768


namespace smallest_positive_integer_in_form_l64_64958

theorem smallest_positive_integer_in_form :
  ∃ (m n p : ℤ), 1234 * m + 56789 * n + 345 * p = 1 := sorry

end smallest_positive_integer_in_form_l64_64958


namespace john_avg_speed_last_30_minutes_l64_64839

open Real

/-- John drove 160 miles in 120 minutes. His average speed during the first
30 minutes was 55 mph, during the second 30 minutes was 75 mph, and during
the third 30 minutes was 60 mph. Prove that his average speed during the
last 30 minutes was 130 mph. -/
theorem john_avg_speed_last_30_minutes (total_distance : ℝ) (total_time_minutes : ℝ)
  (speed_1 : ℝ) (speed_2 : ℝ) (speed_3 : ℝ) (speed_4 : ℝ) :
  total_distance = 160 →
  total_time_minutes = 120 →
  speed_1 = 55 →
  speed_2 = 75 →
  speed_3 = 60 →
  (speed_1 + speed_2 + speed_3 + speed_4) / 4 = total_distance / (total_time_minutes / 60) →
  speed_4 = 130 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end john_avg_speed_last_30_minutes_l64_64839


namespace total_number_of_elements_l64_64927

theorem total_number_of_elements (a b c : ℕ) : 
  (a = 2 ∧ b = 2 ∧ c = 2) ∧ 
  (3.95 = ((4.4 * 2 + 3.85 * 2 + 3.6000000000000014 * 2) / 6)) ->
  a + b + c = 6 := 
by
  sorry

end total_number_of_elements_l64_64927


namespace sum_of_solutions_l64_64777

theorem sum_of_solutions : 
  (∀ x : ℝ, (3 * x) / 15 = 4 / x) → (0 + 4 = 4) :=
by
  sorry

end sum_of_solutions_l64_64777


namespace carnival_ring_toss_l64_64711

theorem carnival_ring_toss (total_amount : ℕ) (days : ℕ) (amount_per_day : ℕ) 
  (h1 : total_amount = 420) 
  (h2 : days = 3) 
  (h3 : total_amount = days * amount_per_day) : amount_per_day = 140 :=
by
  sorry

end carnival_ring_toss_l64_64711


namespace solve_system_l64_64110

theorem solve_system : ∃ x y : ℝ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_l64_64110


namespace smallest_five_digit_divisible_by_2_5_11_l64_64671

theorem smallest_five_digit_divisible_by_2_5_11 : ∃ n, n >= 10000 ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n % 11 = 0 ∧ n = 10010 :=
by
  sorry

end smallest_five_digit_divisible_by_2_5_11_l64_64671


namespace max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l64_64521

noncomputable def f (a x : ℝ) := a * x + Real.log x
noncomputable def g (a x : ℝ) := x * f a x
noncomputable def e := Real.exp 1

-- Statement for part (1)
theorem max_value_fx_when_a_neg1 : 
  ∀ x : ℝ, 0 < x → (f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement for part (2)
theorem find_a_when_max_fx_is_neg3 : 
  (∀ x : ℝ, 0 < x ∧ x ≤ e → (f (-e^2) x ≤ -3)) →
  (∃ a : ℝ, a = -e^2) :=
sorry

-- Statement for part (3)
theorem inequality_gx_if_a_pos (a : ℝ) (hapos : 0 < a) 
  (x1 x2 : ℝ) (hxpos1 : 0 < x1) (hxpos2 : 0 < x2) (hx12 : x1 ≠ x2) :
  2 * g a ((x1 + x2) / 2) < g a x1 + g a x2 :=
sorry

end max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l64_64521


namespace intersection_of_sets_A_B_l64_64088

def set_A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 > 0 }
def set_B : Set ℝ := { x : ℝ | -2 < x ∧ x ≤ 2 }
def set_intersection : Set ℝ := { x : ℝ | -2 < x ∧ x < -1 }

theorem intersection_of_sets_A_B :
  (set_A ∩ set_B) = set_intersection :=
  sorry

end intersection_of_sets_A_B_l64_64088
