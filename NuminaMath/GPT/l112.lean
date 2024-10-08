import Mathlib

namespace circuit_length_is_365_l112_112735

-- Definitions based on given conditions
def runs_morning := 7
def runs_afternoon := 3
def total_distance_week := 25550
def total_runs_day := runs_morning + runs_afternoon
def total_runs_week := total_runs_day * 7

-- Statement of the problem to be proved
theorem circuit_length_is_365 :
  total_distance_week / total_runs_week = 365 :=
sorry

end circuit_length_is_365_l112_112735


namespace ratio_of_M_to_R_l112_112289

variable (M Q P N R : ℝ)

theorem ratio_of_M_to_R :
      M = 0.40 * Q →
      Q = 0.25 * P →
      N = 0.60 * P →
      R = 0.30 * N →
      M / R = 5 / 9 := by
  sorry

end ratio_of_M_to_R_l112_112289


namespace garden_width_l112_112919

variable (W : ℝ) (L : ℝ := 225) (small_gate : ℝ := 3) (large_gate: ℝ := 10) (total_fencing : ℝ := 687)

theorem garden_width :
  2 * L + 2 * W - (small_gate + large_gate) = total_fencing → W = 125 := 
by
  sorry

end garden_width_l112_112919


namespace integer_roots_abs_sum_l112_112595

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end integer_roots_abs_sum_l112_112595


namespace cyclic_quadrilateral_l112_112586

theorem cyclic_quadrilateral (T : ℕ) (S : ℕ) (AB BC CD DA : ℕ) (M N : ℝ × ℝ) (AC BD PQ MN : ℝ) (m n : ℕ) :
  T = 2378 → 
  S = 2 + 3 + 7 + 8 → 
  AB = S - 11 → 
  BC = 2 → 
  CD = 3 → 
  DA = 10 → 
  AC * BD = 47 → 
  PQ / MN = 1/2 → 
  m + n = 3 :=
by
  sorry

end cyclic_quadrilateral_l112_112586


namespace y_square_range_l112_112331

theorem y_square_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) : 
  230 ≤ y^2 ∧ y^2 < 240 :=
sorry

end y_square_range_l112_112331


namespace parallelogram_area_l112_112363

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) 
  (h_b : b = 7) (h_h : h = 2 * b) (h_A : A = b * h) : A = 98 :=
by {
  sorry
}

end parallelogram_area_l112_112363


namespace polygon_exterior_angle_l112_112114

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end polygon_exterior_angle_l112_112114


namespace cos_3theta_l112_112904

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_3theta_l112_112904


namespace sphere_surface_area_l112_112210

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (S : ℝ)
  (hV : V = 36 * π)
  (hvol : V = (4 / 3) * π * r^3) :
  S = 4 * π * r^2 :=
by
  sorry

end sphere_surface_area_l112_112210


namespace neon_signs_blink_together_l112_112389

theorem neon_signs_blink_together :
  Nat.lcm (Nat.lcm (Nat.lcm 7 11) 13) 17 = 17017 :=
by
  sorry

end neon_signs_blink_together_l112_112389


namespace find_angle_B_find_side_b_l112_112610

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {m n : ℝ × ℝ}
variable {dot_product_max : ℝ}

-- Conditions
def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin A + c * Real.sin C - b * Real.sin B = Real.sqrt 2 * a * Real.sin C

def vectors (m n : ℝ × ℝ) := 
  m = (Real.cos A, Real.cos (2 * A)) ∧ n = (12, -5)

def side_length_a (a : ℝ) := 
  a = 4

-- Questions and Proof Problems
theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1 : triangle_condition a b c A B C) : 
  B = π / 4 :=
sorry

theorem find_side_b (A B C : ℝ) (a b c : ℝ) 
  (m n : ℝ × ℝ) (max_dot_product_condition : Real.cos A = 3 / 5) 
  (ha : side_length_a a) (hb : b = a * Real.sin B / Real.sin A) : 
  b = 5 * Real.sqrt 2 / 2 :=
sorry

end find_angle_B_find_side_b_l112_112610


namespace cos_arith_prog_impossible_l112_112140

theorem cos_arith_prog_impossible
  (x y z : ℝ)
  (sin_arith_prog : 2 * Real.sin y = Real.sin x + Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by
  sorry

end cos_arith_prog_impossible_l112_112140


namespace cos_difference_simplify_l112_112165

theorem cos_difference_simplify 
  (x : ℝ) 
  (y : ℝ) 
  (z : ℝ) 
  (h1 : x = Real.cos 72)
  (h2 : y = Real.cos 144)
  (h3 : y = -Real.cos 36)
  (h4 : x = 2 * (Real.cos 36)^2 - 1)
  (hz : z = Real.cos 36)
  : x - y = 1 / 2 :=
by
  sorry

end cos_difference_simplify_l112_112165


namespace initial_population_first_village_equals_l112_112060

-- Definitions of the conditions
def initial_population_second_village : ℕ := 42000
def decrease_first_village_per_year : ℕ := 1200
def increase_second_village_per_year : ℕ := 800
def years : ℕ := 13

-- Proposition we want to prove
/-- The initial population of the first village such that both villages have the same population after 13 years. -/
theorem initial_population_first_village_equals :
  ∃ (P : ℕ), (P - decrease_first_village_per_year * years) = (initial_population_second_village + increase_second_village_per_year * years) 
  := sorry

end initial_population_first_village_equals_l112_112060


namespace power_eq_45_l112_112318

theorem power_eq_45 (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end power_eq_45_l112_112318


namespace find_x_l112_112843

theorem find_x : ∃ x : ℝ, (0.40 * x - 30 = 50) ∧ x = 200 :=
by
  sorry

end find_x_l112_112843


namespace find_a_l112_112467

theorem find_a (a : ℝ) (h : (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) + (1 / Real.log 7 / Real.log a) = 1) : 
  a = 105 := 
sorry

end find_a_l112_112467


namespace anna_reading_time_l112_112243

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l112_112243


namespace num_of_factorizable_poly_l112_112937

theorem num_of_factorizable_poly : 
  ∃ (n : ℕ), (1 ≤ n ∧ n ≤ 2023) ∧ 
              (∃ (a : ℤ), n = a * (a + 1)) :=
sorry

end num_of_factorizable_poly_l112_112937


namespace total_players_on_ground_l112_112472

def cricket_players : ℕ := 15
def hockey_players : ℕ := 12
def football_players : ℕ := 13
def softball_players : ℕ := 15

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 55 := 
by
  sorry

end total_players_on_ground_l112_112472


namespace inequality_proof_l112_112401

theorem inequality_proof
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (ha1 : 0 < a1) (ha2 : 0 < a2) (ha3 : 0 < a3)
  (hb1 : 0 < b1) (hb2 : 0 < b2) (hb3 : 0 < b3) :
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 ≥
    4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) :=
sorry

end inequality_proof_l112_112401


namespace next_equalities_from_conditions_l112_112976

-- Definitions of the equality conditions
def eq1 : Prop := 3^2 + 4^2 = 5^2
def eq2 : Prop := 10^2 + 11^2 + 12^2 = 13^2 + 14^2
def eq3 : Prop := 21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2
def eq4 : Prop := 36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2

-- The next equalities we want to prove
def eq5 : Prop := 55^2 + 56^2 + 57^2 + 58^2 + 59^2 + 60^2 = 61^2 + 62^2 + 63^2 + 64^2 + 65^2
def eq6 : Prop := 78^2 + 79^2 + 80^2 + 81^2 + 82^2 + 83^2 + 84^2 = 85^2 + 86^2 + 87^2 + 88^2 + 89^2 + 90^2

theorem next_equalities_from_conditions : eq1 → eq2 → eq3 → eq4 → (eq5 ∧ eq6) :=
by
  sorry

end next_equalities_from_conditions_l112_112976


namespace problem_statement_l112_112956

theorem problem_statement (a : ℝ) (h : a^2 - 2 * a + 1 = 0) : 4 * a - 2 * a^2 + 2 = 4 := 
sorry

end problem_statement_l112_112956


namespace find_a7_l112_112570

-- Definitions based on given conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k : ℕ, a (n + k) = a n + k * (a 1 - a 0)

-- Given condition in Lean statement
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 11 = 22

-- Proof problem
theorem find_a7 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : a 7 = 11 := 
  sorry

end find_a7_l112_112570


namespace pure_imaginary_condition_l112_112920

theorem pure_imaginary_condition (m : ℝ) (h : (m^2 - 3 * m) = 0) : (m = 0) :=
by
  sorry

end pure_imaginary_condition_l112_112920


namespace greatest_product_sum_300_l112_112912

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l112_112912


namespace counterexample_disproving_proposition_l112_112408

theorem counterexample_disproving_proposition (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = angle2) : False :=
by
  have h_contradiction : angle1 ≠ angle2 := sorry
  exact h_contradiction h2

end counterexample_disproving_proposition_l112_112408


namespace increasing_or_decreasing_subseq_l112_112893

theorem increasing_or_decreasing_subseq {m n : ℕ} (a : Fin (m * n + 1) → ℝ) :
  ∃ (idx_incr : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_incr i) < a (idx_incr j)) ∨ 
  ∃ (idx_decr : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_decr i) > a (idx_decr j)) :=
by
  sorry

end increasing_or_decreasing_subseq_l112_112893


namespace number_of_math_players_l112_112626

theorem number_of_math_players (total_players physics_players both_players : ℕ)
    (h1 : total_players = 25)
    (h2 : physics_players = 15)
    (h3 : both_players = 6)
    (h4 : total_players = physics_players + (total_players - physics_players - (total_players - physics_players - both_players)) + both_players ) :
  total_players - (physics_players - both_players) = 16 :=
sorry

end number_of_math_players_l112_112626


namespace function_monotonically_increasing_l112_112495

-- The function y = x^2 - 2x + 8
def f (x : ℝ) : ℝ := x^2 - 2 * x + 8

-- The theorem stating the function is monotonically increasing on (1, +∞)
theorem function_monotonically_increasing : ∀ x y : ℝ, (1 < x) → (x < y) → (f x < f y) :=
by
  -- Proof is omitted
  sorry

end function_monotonically_increasing_l112_112495


namespace find_x_collinear_l112_112895

def vec := ℝ × ℝ

def collinear (u v: vec): Prop :=
  ∃ k: ℝ, u = (k * v.1, k * v.2)

theorem find_x_collinear:
  ∀ (x: ℝ), (let a : vec := (1, 2)
              let b : vec := (x, 1)
              collinear a (a.1 - b.1, a.2 - b.2)) → x = 1 / 2 :=
by
  intros x h
  sorry

end find_x_collinear_l112_112895


namespace find_ratio_of_d1_and_d2_l112_112698

theorem find_ratio_of_d1_and_d2
  (x y d1 d2 : ℝ)
  (h1 : x + 4 * d1 = y)
  (h2 : x + 5 * d2 = y)
  (h3 : d1 ≠ 0)
  (h4 : d2 ≠ 0) :
  d1 / d2 = 5 / 4 := 
by 
  sorry

end find_ratio_of_d1_and_d2_l112_112698


namespace calculation_correct_l112_112456

theorem calculation_correct :
  (Int.ceil ((15 : ℚ) / 8 * ((-35 : ℚ) / 4)) - 
  Int.floor (((15 : ℚ) / 8) * Int.floor ((-35 : ℚ) / 4 + (1 : ℚ) / 4))) = 1 := by
  sorry

end calculation_correct_l112_112456


namespace exists_n_not_coprime_l112_112236

theorem exists_n_not_coprime (p q : ℕ) (h1 : Nat.gcd p q = 1) (h2 : q > p) (h3 : q - p > 1) :
  ∃ (n : ℕ), Nat.gcd (p + n) (q + n) ≠ 1 :=
by
  sorry

end exists_n_not_coprime_l112_112236


namespace length_of_leg_of_isosceles_right_triangle_l112_112875

def is_isosceles_right_triangle (a b h : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = h^2

def median_to_hypotenuse (m h : ℝ) : Prop :=
  m = h / 2

theorem length_of_leg_of_isosceles_right_triangle (m : ℝ) (h a : ℝ)
  (h1 : median_to_hypotenuse m h)
  (h2 : h = 2 * m)
  (h3 : is_isosceles_right_triangle a a h) :
  a = 15 * Real.sqrt 2 :=
by
  -- Skipping the proof
  sorry

end length_of_leg_of_isosceles_right_triangle_l112_112875


namespace length_of_train_l112_112557

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end length_of_train_l112_112557


namespace ice_cream_cone_cost_l112_112197

theorem ice_cream_cone_cost (total_sales : ℝ) (free_cones_given : ℕ) (cost_per_cone : ℝ) 
  (customers_per_group : ℕ) (cones_sold_per_group : ℕ) 
  (h1 : total_sales = 100)
  (h2: free_cones_given = 10)
  (h3: customers_per_group = 6)
  (h4: cones_sold_per_group = 5) :
  cost_per_cone = 2 := sorry

end ice_cream_cone_cost_l112_112197


namespace find_x_l112_112502

theorem find_x (x : ℝ) (h : 2 * x - 3 * x + 5 * x = 80) : x = 20 :=
by 
  -- placeholder for proof
  sorry 

end find_x_l112_112502


namespace algebraic_expression_evaluation_l112_112654

noncomputable def algebraic_expression (x : ℝ) : ℝ :=
  (x / (x^2 + 2*x + 1) - 1 / (2*x + 2)) / ((x - 1) / (4*x + 4))

noncomputable def substitution_value : ℝ :=
  2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_evaluation :
  algebraic_expression substitution_value = Real.sqrt 2 := by
  sorry

end algebraic_expression_evaluation_l112_112654


namespace probability_of_event_A_l112_112307

def probability_event_A : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes

-- Statement of the theorem
theorem probability_of_event_A :
  probability_event_A = 1 / 6 :=
by
  -- This is where the proof would go, replaced with sorry for now.
  sorry

end probability_of_event_A_l112_112307


namespace part1_part2_l112_112619

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + (1 + a) * Real.exp (-x)

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = 0 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≥ a + 1) → a ≤ 3 := by
  sorry

end part1_part2_l112_112619


namespace find_triangle_height_l112_112010

-- Define the problem conditions
def Rectangle.perimeter (l : ℕ) (w : ℕ) : ℕ := 2 * l + 2 * w
def Rectangle.area (l : ℕ) (w : ℕ) : ℕ := l * w
def Triangle.area (b : ℕ) (h : ℕ) : ℕ := (b * h) / 2

-- Conditions
namespace Conditions
  -- Perimeter of the rectangle is 60 cm
  def rect_perimeter (l w : ℕ) : Prop := Rectangle.perimeter l w = 60
  -- Base of the right triangle is 15 cm
  def tri_base : ℕ := 15
  -- Areas of the rectangle and the triangle are equal
  def equal_areas (l w h : ℕ) : Prop := Rectangle.area l w = Triangle.area tri_base h
end Conditions

-- Proof problem: Given these conditions, prove h = 30
theorem find_triangle_height (l w h : ℕ) 
  (h1 : Conditions.rect_perimeter l w)
  (h2 : Conditions.equal_areas l w h) : h = 30 :=
  sorry

end find_triangle_height_l112_112010


namespace bisection_contains_root_l112_112675

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem bisection_contains_root : (1 < 1.5) ∧ f 1 < 0 ∧ f 1.5 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end bisection_contains_root_l112_112675


namespace find_value_l112_112975

theorem find_value (x y : ℝ) (h : x - 2 * y = 1) : 3 - 4 * y + 2 * x = 5 := sorry

end find_value_l112_112975


namespace math_proof_l112_112740

noncomputable def math_problem (x : ℝ) : ℝ :=
  (3 / (2 * x) * (1 / 2) * (2 / 5) * 5020) - ((2 ^ 3) * (1 / (3 * x + 2)) * 250) + Real.sqrt (900 / x)

theorem math_proof :
  math_problem 4 = 60.393 :=
by
  sorry

end math_proof_l112_112740


namespace total_pages_proof_l112_112142

/-
Conditions:
1. Johnny's essay has 150 words.
2. Madeline's essay is double the length of Johnny's essay.
3. Timothy's essay has 30 more words than Madeline's essay.
4. One page contains 260 words.

Question:
Prove that the total number of pages do Johnny, Madeline, and Timothy's essays fill is 5.
-/

def johnny_words : ℕ := 150
def words_per_page : ℕ := 260

def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 30

def pages (words : ℕ) : ℕ := (words + words_per_page - 1) / words_per_page  -- division rounding up

def johnny_pages : ℕ := pages johnny_words
def madeline_pages : ℕ := pages madeline_words
def timothy_pages : ℕ := pages timothy_words

def total_pages : ℕ := johnny_pages + madeline_pages + timothy_pages

theorem total_pages_proof : total_pages = 5 :=
by sorry

end total_pages_proof_l112_112142


namespace simplify_expression_l112_112679

theorem simplify_expression (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 :=
sorry

end simplify_expression_l112_112679


namespace find_t_l112_112419

theorem find_t : ∃ t, ∀ (x y : ℝ), (x, y) = (0, 1) ∨ (x, y) = (-6, -3) → (t, 7) ∈ {p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ ((0, 1) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) ∧ ((-6, -3) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) } → t = 9 :=
by
  sorry

end find_t_l112_112419


namespace positive_integer_representation_l112_112930

theorem positive_integer_representation (a b c n : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) 
  (h₄ : n = (abc + a * b + a) / (abc + c * b + c)) : n = 1 ∨ n = 2 := 
by
  sorry

end positive_integer_representation_l112_112930


namespace find_a_l112_112357

theorem find_a (a b x : ℝ) (h1 : a ≠ b)
  (h2 : a^3 + b^3 = 35 * x^3)
  (h3 : a^2 - b^2 = 4 * x^2) : a = 2 * x ∨ a = -2 * x :=
by
  sorry

end find_a_l112_112357


namespace walk_to_Lake_Park_restaurant_time_l112_112659

-- Define the problem parameters
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_time_gone : ℕ := 32

-- Define the goal to prove
theorem walk_to_Lake_Park_restaurant_time :
  total_time_gone - (time_to_hidden_lake + time_from_hidden_lake) = 10 :=
by
  -- skipping the proof here
  sorry

end walk_to_Lake_Park_restaurant_time_l112_112659


namespace imaginary_part_z1z2_l112_112078

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end imaginary_part_z1z2_l112_112078


namespace units_digit_S7890_l112_112020

noncomputable def c : ℝ := 4 + 3 * Real.sqrt 2
noncomputable def d : ℝ := 4 - 3 * Real.sqrt 2
noncomputable def S (n : ℕ) : ℝ := (1/2:ℝ) * (c^n + d^n)

theorem units_digit_S7890 : (S 7890) % 10 = 8 :=
sorry

end units_digit_S7890_l112_112020


namespace part_a_part_b_part_c_part_d_l112_112999

open Nat

theorem part_a (y z : ℕ) (hy : 0 < y) (hz : 0 < z) : 
  (1 = 1 / y + 1 / z) ↔ (y = 2 ∧ z = 1) := 
by 
  sorry

theorem part_b (y z : ℕ) (hy : y ≥ 2) (hz : 0 < z) : 
  (1 / 2 + 1 / y = 1 / 2 + 1 / z) ↔ (y = z ∧ y ≥ 2) ∨ (y = 1 ∧ z = 1) := 
by 
  sorry 

theorem part_c (y z : ℕ) (hy : y ≥ 3) (hz : 0 < z) : 
  (1 / 3 + 1 / y = 1 / 2 + 1 / z) ↔ 
    (y = 3 ∧ z = 6) ∨ 
    (y = 4 ∧ z = 12) ∨ 
    (y = 5 ∧ z = 30) ∨ 
    (y = 2 ∧ z = 3) := 
by 
  sorry 

theorem part_d (x y : ℕ) (hx : x ≥ 4) (hy : y ≥ 4) : 
  ¬(1 / x + 1 / y = 1 / 2 + 1 / z) := 
by 
  sorry

end part_a_part_b_part_c_part_d_l112_112999


namespace track_meet_girls_short_hair_l112_112935

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l112_112935


namespace eval_expression_l112_112039

theorem eval_expression : 3 ^ 4 - 4 * 3 ^ 3 + 6 * 3 ^ 2 - 4 * 3 + 1 = 16 := 
by 
  sorry

end eval_expression_l112_112039


namespace option_B_can_be_factored_l112_112366

theorem option_B_can_be_factored (a b : ℝ) : 
  (-a^2 + b^2) = (b+a)*(b-a) := 
by
  sorry

end option_B_can_be_factored_l112_112366


namespace find_multiple_l112_112326

-- Definitions of the conditions
def is_positive (x : ℝ) : Prop := x > 0

-- Main statement
theorem find_multiple (x : ℝ) (h : is_positive x) (hx : x = 8) : ∃ k : ℝ, x + 8 = k * (1 / x) ∧ k = 128 :=
by
  use 128
  sorry

end find_multiple_l112_112326


namespace find_coefficients_l112_112949

theorem find_coefficients (A B C D : ℚ) :
  (∀ x : ℚ, x ≠ -1 → 
  (A / (x + 1)) + (B / (x + 1)^2) + ((C * x + D) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1))) →
  A = 1 ∧ B = 1 ∧ C = -1 ∧ D = -1 :=
sorry

end find_coefficients_l112_112949


namespace percent_of_75_of_125_l112_112459

theorem percent_of_75_of_125 : (75 / 125) * 100 = 60 := by
  sorry

end percent_of_75_of_125_l112_112459


namespace revenue_function_correct_strategy_not_profitable_l112_112398

-- Given conditions 
def purchase_price : ℝ := 1
def last_year_price : ℝ := 2
def last_year_sales_volume : ℕ := 10000
def last_year_revenue : ℝ := 20000
def proportionality_constant : ℝ := 4
def increased_sales_volume (x : ℝ) : ℝ := proportionality_constant * (2 - x) ^ 2

-- Questions translated to Lean statements
def revenue_this_year (x : ℝ) : ℝ := 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17

theorem revenue_function_correct (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x = 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17 :=
by
  sorry

theorem strategy_not_profitable (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x ≤ last_year_revenue :=
by
  sorry

end revenue_function_correct_strategy_not_profitable_l112_112398


namespace pies_from_apples_l112_112987

theorem pies_from_apples (total_apples : ℕ) (percent_handout : ℝ) (apples_per_pie : ℕ) 
  (h_total : total_apples = 800) (h_percent : percent_handout = 0.65) (h_per_pie : apples_per_pie = 15) : 
  (total_apples * (1 - percent_handout)) / apples_per_pie = 18 := 
by 
  sorry

end pies_from_apples_l112_112987


namespace interest_rate_eq_five_percent_l112_112242

def total_sum : ℝ := 2665
def P2 : ℝ := 1332.5
def P1 : ℝ := total_sum - P2

theorem interest_rate_eq_five_percent :
  (3 * 0.03 * P1 = r * 0.03 * P2) → r = 5 :=
by
  sorry

end interest_rate_eq_five_percent_l112_112242


namespace flour_more_than_salt_l112_112407

open Function

-- Definitions based on conditions
def flour_needed : ℕ := 12
def flour_added : ℕ := 2
def salt_needed : ℕ := 7
def salt_added : ℕ := 0

-- Given that these definitions hold, prove the following theorem
theorem flour_more_than_salt : (flour_needed - flour_added) - (salt_needed - salt_added) = 3 :=
by
  -- Here you would include the proof, but as instructed, we skip it with "sorry".
  sorry

end flour_more_than_salt_l112_112407


namespace remaining_subtasks_l112_112794

def total_problems : ℝ := 72.0
def finished_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks :
    (total_problems * subtasks_per_problem - finished_problems * subtasks_per_problem) = 200 := 
by
  sorry

end remaining_subtasks_l112_112794


namespace option_b_option_c_option_d_l112_112340

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end option_b_option_c_option_d_l112_112340


namespace find_x_l112_112323

theorem find_x (x : ℝ) (h : 2 * x - 1 = -( -x + 5 )) : x = -6 :=
by
  sorry

end find_x_l112_112323


namespace minimum_value_l112_112643

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + (1 / (a * b * c)) ≥ 10 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l112_112643


namespace chord_slope_of_ellipse_bisected_by_point_A_l112_112682

theorem chord_slope_of_ellipse_bisected_by_point_A :
  ∀ (P Q : ℝ × ℝ),
  (P.1^2 / 36 + P.2^2 / 9 = 1) ∧ (Q.1^2 / 36 + Q.2^2 / 9 = 1) ∧ 
  ((P.1 + Q.1) / 2 = 1) ∧ ((P.2 + Q.2) / 2 = 1) →
  (Q.2 - P.2) / (Q.1 - P.1) = -1 / 4 :=
by
  intros
  sorry

end chord_slope_of_ellipse_bisected_by_point_A_l112_112682


namespace find_a_l112_112608

theorem find_a (a : ℝ) (h1 : 1 < a) (h2 : 1 + a = 3) : a = 2 :=
sorry

end find_a_l112_112608


namespace molar_mass_of_compound_l112_112106

variable (total_weight : ℝ) (num_moles : ℝ)

theorem molar_mass_of_compound (h1 : total_weight = 2352) (h2 : num_moles = 8) :
    total_weight / num_moles = 294 :=
by
  rw [h1, h2]
  norm_num

end molar_mass_of_compound_l112_112106


namespace andrew_paid_in_dollars_l112_112926

def local_currency_to_dollars (units : ℝ) : ℝ := units * 0.25

def cost_of_fruits : ℝ :=
  let cost_grapes := 7 * 68
  let cost_mangoes := 9 * 48
  let cost_apples := 5 * 55
  let cost_oranges := 4 * 38
  let total_cost_grapes_mangoes := cost_grapes + cost_mangoes
  let total_cost_apples_oranges := cost_apples + cost_oranges
  let discount_grapes_mangoes := 0.10 * total_cost_grapes_mangoes
  let discounted_grapes_mangoes := total_cost_grapes_mangoes - discount_grapes_mangoes
  let discounted_apples_oranges := total_cost_apples_oranges - 25
  let total_discounted_cost := discounted_grapes_mangoes + discounted_apples_oranges
  let sales_tax := 0.05 * total_discounted_cost
  let total_tax := sales_tax + 15
  let total_amount_with_taxes := total_discounted_cost + total_tax
  total_amount_with_taxes

theorem andrew_paid_in_dollars : local_currency_to_dollars cost_of_fruits = 323.79 :=
  by
  sorry

end andrew_paid_in_dollars_l112_112926


namespace relationship_between_x_x2_and_x3_l112_112457

theorem relationship_between_x_x2_and_x3 (x : ℝ) (h : -1 < x ∧ x < 0) :
  x ^ 3 < x ∧ x < x ^ 2 :=
by
  sorry

end relationship_between_x_x2_and_x3_l112_112457


namespace find_other_number_l112_112126

theorem find_other_number (LCM : ℕ) (HCF : ℕ) (n1 : ℕ) (n2 : ℕ) 
  (h_lcm : LCM = 2310) (h_hcf : HCF = 26) (h_n1 : n1 = 210) :
  n2 = 286 :=
by
  sorry

end find_other_number_l112_112126


namespace vectors_perpendicular_l112_112045

def vec (a b : ℝ) := (a, b)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

@[simp]
def a := vec (-1) 2
@[simp]
def b := vec 1 3

theorem vectors_perpendicular :
  dot_product a (vector_sub a b) = 0 := by
  sorry

end vectors_perpendicular_l112_112045


namespace license_plate_increase_factor_l112_112246

def old_plate_count : ℕ := 26^2 * 10^3
def new_plate_count : ℕ := 26^4 * 10^4
def increase_factor : ℕ := new_plate_count / old_plate_count

theorem license_plate_increase_factor : increase_factor = 2600 :=
by
  unfold increase_factor
  rw [old_plate_count, new_plate_count]
  norm_num
  sorry

end license_plate_increase_factor_l112_112246


namespace solution_to_inequality_l112_112579

theorem solution_to_inequality (x : ℝ) :
  (∃ y : ℝ, y = x^(1/3) ∧ y + 3 / (y + 2) ≤ 0) ↔ x < -8 := 
sorry

end solution_to_inequality_l112_112579


namespace tan_theta_minus_pi_over_4_l112_112821

theorem tan_theta_minus_pi_over_4 (θ : Real) (h1 : θ ∈ Set.Ioc (-(π / 2)) 0)
  (h2 : Real.sin (θ + π / 4) = 3 / 5) : Real.tan (θ - π / 4) = - (4 / 3) :=
by
  /- Proof goes here -/
  sorry

end tan_theta_minus_pi_over_4_l112_112821


namespace part_a_part_b_l112_112501

-- Conditions
def ornament_to_crackers (n : ℕ) : ℕ := n * 2
def sparklers_to_garlands (n : ℕ) : ℕ := (n / 5) * 2
def garlands_to_ornaments (n : ℕ) : ℕ := n * 4

-- Part (a)
theorem part_a (sparklers : ℕ) (h : sparklers = 10) : ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) = 32 :=
by
  sorry

-- Part (b)
theorem part_b (ornaments : ℕ) (crackers : ℕ) (sparklers : ℕ) (h₁ : ornaments = 5) (h₂ : crackers = 1) (h₃ : sparklers = 2) :
  ornament_to_crackers ornaments + crackers > ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) :=
by
  sorry

end part_a_part_b_l112_112501


namespace minnie_more_than_week_l112_112814

-- Define the variables and conditions
variable (M : ℕ) -- number of horses Minnie mounts per day
variable (mickey_daily : ℕ) -- number of horses Mickey mounts per day

axiom mickey_daily_formula : mickey_daily = 2 * M - 6
axiom mickey_total_per_week : mickey_daily * 7 = 98
axiom days_in_week : 7 = 7

-- Theorem: Minnie mounts 3 more horses per day than there are days in a week
theorem minnie_more_than_week (M : ℕ) 
  (h1 : mickey_daily = 2 * M - 6)
  (h2 : mickey_daily * 7 = 98)
  (h3 : 7 = 7) :
  M - 7 = 3 := 
sorry

end minnie_more_than_week_l112_112814


namespace tan_alpha_plus_pi_l112_112445

-- Define the given conditions and prove the desired equality.
theorem tan_alpha_plus_pi 
  (α : ℝ) 
  (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (π - α) = 1 / 3) : 
  Real.tan (α + π) = -2 * Real.sqrt 2 :=
by
  sorry

end tan_alpha_plus_pi_l112_112445


namespace length_of_train_l112_112681

-- Define the conditions as variables
def speed : ℝ := 39.27272727272727
def time : ℝ := 55
def length_bridge : ℝ := 480

-- Calculate the total distance using the given conditions
def total_distance : ℝ := speed * time

-- Prove that the length of the train is 1680 meters
theorem length_of_train :
  (total_distance - length_bridge) = 1680 :=
by
  sorry

end length_of_train_l112_112681


namespace find_angle_OD_base_l112_112936

noncomputable def angle_between_edge_and_base (α β : ℝ): ℝ :=
  Real.arctan ((Real.sin α * Real.sin β) / Real.sqrt (Real.sin (α - β) * Real.sin (α + β)))

theorem find_angle_OD_base (α β : ℝ) :
  ∃ γ : ℝ, γ = angle_between_edge_and_base α β :=
sorry

end find_angle_OD_base_l112_112936


namespace opposite_of_negative_five_l112_112545

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end opposite_of_negative_five_l112_112545


namespace pace_ratio_l112_112888

variable (P P' D : ℝ)

-- Usual time to reach the office in minutes
def T_usual := 120

-- Time to reach the office on the late day in minutes
def T_late := 140

-- Distance to the office is the same
def office_distance_usual := P * T_usual
def office_distance_late := P' * T_late

theorem pace_ratio (h : office_distance_usual = office_distance_late) : P' / P = 6 / 7 :=
by
  sorry

end pace_ratio_l112_112888


namespace geometric_sequence_a5_l112_112118

-- Definitions based on the conditions:
variable {a : ℕ → ℝ} -- the sequence {a_n}
variable (q : ℝ) -- the common ratio of the geometric sequence

-- The sequence is geometric and terms are given:
axiom seq_geom (n m : ℕ) : a n = a 0 * q ^ n
axiom a_3_is_neg4 : a 3 = -4
axiom a_7_is_neg16 : a 7 = -16

-- The specific theorem we are proving:
theorem geometric_sequence_a5 :
  a 5 = -8 :=
by {
  sorry
}

end geometric_sequence_a5_l112_112118


namespace area_of_circumscribed_circle_eq_48pi_l112_112783

noncomputable def side_length := 12
noncomputable def radius := (2/3) * (side_length / 2) * (Real.sqrt 3)
noncomputable def area := Real.pi * radius^2

theorem area_of_circumscribed_circle_eq_48pi :
  area = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_eq_48pi_l112_112783


namespace candy_peanut_butter_is_192_l112_112862

/-
   Define the conditions and the statement to be proved.
   The definitions follow directly from the problem's conditions.
-/
def candy_problem : Prop :=
  ∃ (peanut_butter_jar grape_jar banana_jar coconut_jar : ℕ),
    banana_jar = 43 ∧
    grape_jar = banana_jar + 5 ∧
    peanut_butter_jar = 4 * grape_jar ∧
    coconut_jar = 2 * banana_jar - 10 ∧
    peanut_butter_jar = 192
  -- The tuple (question, conditions, correct answer) is translated into this lemma

theorem candy_peanut_butter_is_192 : candy_problem :=
  by
    -- Skipping the actual proof as requested
    sorry

end candy_peanut_butter_is_192_l112_112862


namespace students_not_making_the_cut_l112_112531

-- Define the total number of girls, boys, and the number of students called back
def number_of_girls : ℕ := 39
def number_of_boys : ℕ := 4
def students_called_back : ℕ := 26

-- Define the total number of students trying out
def total_students : ℕ := number_of_girls + number_of_boys

-- Formulate the problem statement as a theorem
theorem students_not_making_the_cut : total_students - students_called_back = 17 := 
by 
  -- Omitted proof, just the statement
  sorry

end students_not_making_the_cut_l112_112531


namespace dividend_calculation_l112_112438

theorem dividend_calculation (D : ℝ) (Q : ℝ) (R : ℝ) (Dividend : ℝ) (h1 : D = 47.5) (h2 : Q = 24.3) (h3 : R = 32.4)  :
  Dividend = D * Q + R := by
  rw [h1, h2, h3]
  sorry -- This skips the actual computation proof

end dividend_calculation_l112_112438


namespace count_ball_distributions_l112_112927

theorem count_ball_distributions : 
  ∃ (n : ℕ), n = 3 ∧
  (∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → (∀ (dist : ℕ → ℕ), (sorry: Prop))) := sorry

end count_ball_distributions_l112_112927


namespace g_correct_l112_112921

-- Define the polynomials involved
def p1 (x : ℝ) : ℝ := 2 * x^5 + 4 * x^3 - 3 * x
def p2 (x : ℝ) : ℝ := 7 * x^3 + 5 * x - 2

-- Define g(x) as the polynomial we need to find
def g (x : ℝ) : ℝ := -2 * x^5 + 3 * x^3 + 8 * x - 2

-- Now, state the condition
def condition (x : ℝ) : Prop := p1 x + g x = p2 x

-- Prove the condition holds with the defined polynomials
theorem g_correct (x : ℝ) : condition x :=
by
  change p1 x + g x = p2 x
  sorry

end g_correct_l112_112921


namespace amount_for_gifts_and_charitable_causes_l112_112494

namespace JillExpenses

def net_monthly_salary : ℝ := 3700
def discretionary_income : ℝ := 0.20 * net_monthly_salary -- 1/5 * 3700
def vacation_fund : ℝ := 0.30 * discretionary_income
def savings : ℝ := 0.20 * discretionary_income
def eating_out_and_socializing : ℝ := 0.35 * discretionary_income
def gifts_and_charitable_causes : ℝ := discretionary_income - (vacation_fund + savings + eating_out_and_socializing)

theorem amount_for_gifts_and_charitable_causes : gifts_and_charitable_causes = 111 := sorry

end JillExpenses

end amount_for_gifts_and_charitable_causes_l112_112494


namespace balance_expenses_l112_112885

-- Define the basic amounts paid by Alice, Bob, and Carol
def alicePaid : ℕ := 120
def bobPaid : ℕ := 150
def carolPaid : ℕ := 210

-- The total expenditure
def totalPaid : ℕ := alicePaid + bobPaid + carolPaid

-- Each person's share of the total expenses
def eachShare : ℕ := totalPaid / 3

-- Amount Alice should give to balance the expenses
def a : ℕ := eachShare - alicePaid

-- Amount Bob should give to balance the expenses
def b : ℕ := eachShare - bobPaid

-- The statement to be proven
theorem balance_expenses : a - b = 30 :=
by
  sorry

end balance_expenses_l112_112885


namespace range_of_a_l112_112040

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x - 3 > 0 → x > a) ↔ a ≥ 1 :=
by
  sorry

end range_of_a_l112_112040


namespace number_of_children_l112_112150

theorem number_of_children 
  (A C : ℕ) 
  (h1 : A + C = 201) 
  (h2 : 8 * A + 4 * C = 964) : 
  C = 161 := 
sorry

end number_of_children_l112_112150


namespace triangle_area_546_l112_112105

theorem triangle_area_546 :
  ∀ (a b c : ℕ), a = 13 ∧ b = 84 ∧ c = 85 ∧ a^2 + b^2 = c^2 →
  (1 / 2 : ℝ) * (a * b) = 546 :=
by
  intro a b c
  intro h
  sorry

end triangle_area_546_l112_112105


namespace Vasya_mushrooms_l112_112661

def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digitsSum (n : ℕ) : ℕ := (n / 100) + ((n % 100) / 10) + (n % 10)

theorem Vasya_mushrooms :
  ∃ n : ℕ, isThreeDigit n ∧ digitsSum n = 14 ∧ n = 950 := 
by
  sorry

end Vasya_mushrooms_l112_112661


namespace height_radius_ratio_l112_112963

variables (R H V : ℝ) (π : ℝ) (A : ℝ)

-- Given conditions
def volume_condition : Prop := π * R^2 * H = V / 2
def surface_area : ℝ := 2 * π * R^2 + 2 * π * R * H

-- Statement to prove
theorem height_radius_ratio (h_volume : volume_condition R H V π) :
  H / R = 2 := 
sorry

end height_radius_ratio_l112_112963


namespace cube_painted_faces_l112_112747

noncomputable def painted_faces_count (side_length painted_cubes_edge middle_cubes_edge : ℕ) : ℕ :=
  let total_corners := 8
  let total_edges := 12
  total_corners + total_edges * middle_cubes_edge

theorem cube_painted_faces :
  ∀ side_length : ℕ, side_length = 4 →
  ∀ painted_cubes_edge middle_cubes_edge total_cubes : ℕ,
  total_cubes = side_length * side_length * side_length →
  painted_cubes_edge = 3 →
  middle_cubes_edge = 2 →
  painted_faces_count side_length painted_cubes_edge middle_cubes_edge = 32 := sorry

end cube_painted_faces_l112_112747


namespace derivative_at_one_l112_112447

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_at_one : (deriv f 1) = 4 := by
  sorry

end derivative_at_one_l112_112447


namespace find_abc_squares_l112_112057

variable (a b c x : ℕ)

theorem find_abc_squares (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 99 * (c - a) = 65 * x) (h4 : 495 = 65 * x) : a^2 + b^2 + c^2 = 53 :=
  sorry

end find_abc_squares_l112_112057


namespace cost_of_individual_roll_l112_112424

theorem cost_of_individual_roll
  (p : ℕ) (c : ℝ) (s : ℝ) (x : ℝ)
  (hc : c = 9)
  (hp : p = 12)
  (hs : s = 0.25)
  (h : 12 * x = 9 * (1 + s)) :
  x = 0.9375 :=
by
  sorry

end cost_of_individual_roll_l112_112424


namespace geometric_sequence_ratio_l112_112285

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
(h_arith : 2 * a 1 * q = a 0 + a 0 * q * q) :
  q = 2 + Real.sqrt 3 ∨ q = 2 - Real.sqrt 3 := 
by
  sorry

end geometric_sequence_ratio_l112_112285


namespace smallest_four_digit_int_equiv_8_mod_9_l112_112391

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end smallest_four_digit_int_equiv_8_mod_9_l112_112391


namespace solution_of_equation_l112_112555

theorem solution_of_equation (a b c : ℕ) :
    a^(b + 20) * (c - 1) = c^(b + 21) - 1 ↔ 
    (∃ b' : ℕ, b = b' ∧ a = 1 ∧ c = 0) ∨ 
    (∃ a' b' : ℕ, a = a' ∧ b = b' ∧ c = 1) :=
by sorry

end solution_of_equation_l112_112555


namespace rectangular_block_height_l112_112736

theorem rectangular_block_height (l w h : ℕ) 
  (volume_eq : l * w * h = 42) 
  (perimeter_eq : 2 * l + 2 * w = 18) : 
  h = 3 :=
by
  sorry

end rectangular_block_height_l112_112736


namespace mean_noon_temperature_l112_112651

def temperatures : List ℕ := [82, 80, 83, 88, 90, 92, 90, 95]

def mean_temperature (temps : List ℕ) : ℚ :=
  (temps.foldr (λ a b => a + b) 0 : ℚ) / temps.length

theorem mean_noon_temperature :
  mean_temperature temperatures = 87.5 := by
  sorry

end mean_noon_temperature_l112_112651


namespace stratified_sampling_model_A_l112_112562

theorem stratified_sampling_model_A (r_A r_B r_C n x : ℕ) 
  (r_A_eq : r_A = 2) (r_B_eq : r_B = 3) (r_C_eq : r_C = 5) 
  (n_eq : n = 80) : 
  (r_A * n / (r_A + r_B + r_C) = x) -> x = 16 := 
by 
  intros h
  rw [r_A_eq, r_B_eq, r_C_eq, n_eq] at h
  norm_num at h
  exact h.symm

end stratified_sampling_model_A_l112_112562


namespace ads_not_blocked_not_interesting_l112_112409

theorem ads_not_blocked_not_interesting:
  (let A_blocks := 0.75
   let B_blocks := 0.85
   let C_blocks := 0.95
   let A_let_through := 1 - A_blocks
   let B_let_through := 1 - B_blocks
   let C_let_through := 1 - C_blocks
   let all_let_through := A_let_through * B_let_through * C_let_through
   let interesting := 0.15
   let not_interesting := 1 - interesting
   (all_let_through * not_interesting) = 0.00159375) :=
  sorry

end ads_not_blocked_not_interesting_l112_112409


namespace sum_terms_sequence_l112_112826

noncomputable def geometric_sequence := ℕ → ℝ

variables (a : geometric_sequence)
variables (r : ℝ) (h_pos : ∀ n, a n > 0)

-- Geometric sequence condition
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r

-- Given condition
axiom h_condition : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100

-- The goal is to prove that a_4 + a_6 = 10
theorem sum_terms_sequence : a 4 + a 6 = 10 :=
by
  sorry

end sum_terms_sequence_l112_112826


namespace stacy_days_to_complete_paper_l112_112017

-- Conditions as definitions
def total_pages : ℕ := 63
def pages_per_day : ℕ := 9

-- The problem statement
theorem stacy_days_to_complete_paper : total_pages / pages_per_day = 7 :=
by
  sorry

end stacy_days_to_complete_paper_l112_112017


namespace triangle_inequality_l112_112304

theorem triangle_inequality (A B C : ℝ) (k : ℝ) (hABC : A + B + C = π) (h1 : 1 ≤ k) (h2 : k ≤ 2) :
  (1 / (k - Real.cos A)) + (1 / (k - Real.cos B)) + (1 / (k - Real.cos C)) ≥ 6 / (2 * k - 1) := 
by
  sorry

end triangle_inequality_l112_112304


namespace gcd_determinant_l112_112479

theorem gcd_determinant (a b : ℤ) (h : Int.gcd a b = 1) :
  Int.gcd (a + b) (a^2 + b^2 - a * b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a * b) = 3 :=
sorry

end gcd_determinant_l112_112479


namespace cube_volume_surface_area_x_l112_112942

theorem cube_volume_surface_area_x (x s : ℝ) (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_x_l112_112942


namespace range_of_a_l112_112957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 > 1) (h4 : ∀ x, g a x ≤ 0 → ¬(x < 0 ∧ g a x > 0)) :
  2 < a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l112_112957


namespace geometric_seq_a3_equals_3_l112_112529

variable {a : ℕ → ℝ}
variable (h_geometric : ∀ m n p q, m + n = p + q → a m * a n = a p * a q)
variable (h_pos : ∀ n, n > 0 → a n > 0)
variable (h_cond : a 2 * a 4 = 9)

theorem geometric_seq_a3_equals_3 : a 3 = 3 := by
  sorry

end geometric_seq_a3_equals_3_l112_112529


namespace urn_gold_coins_percent_l112_112591

theorem urn_gold_coins_percent (perc_beads : ℝ) (perc_silver_coins : ℝ) (perc_gold_coins : ℝ) :
  perc_beads = 0.2 →
  perc_silver_coins = 0.4 →
  perc_gold_coins = 0.48 :=
by
  intros h1 h2
  sorry

end urn_gold_coins_percent_l112_112591


namespace sum_of_numbers_l112_112993

theorem sum_of_numbers : 3 + 33 + 333 + 33.3 = 402.3 :=
  by
    sorry

end sum_of_numbers_l112_112993


namespace inequality_solution_l112_112341

theorem inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := sorry

end inequality_solution_l112_112341


namespace prism_volume_l112_112640

/-- The volume of a rectangular prism given the areas of three of its faces. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 :=
by
  sorry

end prism_volume_l112_112640


namespace unknown_number_is_7_l112_112900

theorem unknown_number_is_7 (x : ℤ) (hx : x > 0)
  (h : (1 / 4 : ℚ) * (10 * x + 7 - x ^ 2) - x = 0) : x = 7 :=
  sorry

end unknown_number_is_7_l112_112900


namespace total_dots_is_78_l112_112646

-- Define the conditions as Lean definitions
def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

-- Define the total number of ladybugs
def total_ladybugs : ℕ := ladybugs_monday + ladybugs_tuesday

-- Define the total number of dots
def total_dots : ℕ := total_ladybugs * dots_per_ladybug

-- Theorem stating the problem to solve
theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end total_dots_is_78_l112_112646


namespace prime_numbers_solution_l112_112739

theorem prime_numbers_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h1 : Nat.Prime (p + q)) (h2 : Nat.Prime (p^2 + q^2 - q)) : p = 3 ∧ q = 2 :=
by
  sorry

end prime_numbers_solution_l112_112739


namespace sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l112_112266

noncomputable def P (x : ℝ) : Prop := (x - 1)^2 > 16
noncomputable def Q (x a : ℝ) : Prop := x^2 + (a - 8) * x - 8 * a ≤ 0

theorem sufficient_not_necessary (a : ℝ) (x : ℝ) :
  a = 3 →
  (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem necessary_and_sufficient (a : ℝ) :
  (-5 ≤ a ∧ a ≤ 3) ↔ ∀ x, (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem P_inter_Q (a : ℝ) (x : ℝ) :
  (a > 3 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a) ∨ (5 < x ∧ x ≤ 8)) ∧
  (-5 ≤ a ∧ a ≤ 3 → (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8)) ∧
  (-8 ≤ a ∧ a < -5 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) ∧
  (a < -8 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) :=
sorry

end sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l112_112266


namespace watch_correction_l112_112907

noncomputable def correction_time (loss_per_day : ℕ) (start_date : ℕ) (end_date : ℕ) (spring_forward_hour : ℕ) (correction_time_hour : ℕ) : ℝ :=
  let n_days := end_date - start_date
  let total_hours_watch := n_days * 24 + correction_time_hour - spring_forward_hour
  let loss_rate_per_hour := (loss_per_day : ℝ) / 24
  let total_loss := loss_rate_per_hour * total_hours_watch
  total_loss

theorem watch_correction :
  correction_time 3 1 5 1 6 = 6.625 :=
by
  sorry

end watch_correction_l112_112907


namespace quadratic_two_distinct_real_roots_l112_112724

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end quadratic_two_distinct_real_roots_l112_112724


namespace not_perfect_square_l112_112426

theorem not_perfect_square (n : ℕ) (h : 0 < n) : ¬ ∃ k : ℕ, k * k = 2551 * 543^n - 2008 * 7^n :=
by
  sorry

end not_perfect_square_l112_112426


namespace Frank_work_hours_l112_112867

def hoursWorked (h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday : Nat) : Nat :=
  h_monday + h_tuesday + h_wednesday + h_thursday + h_friday + h_saturday

theorem Frank_work_hours
  (h_monday : Nat := 8)
  (h_tuesday : Nat := 10)
  (h_wednesday : Nat := 7)
  (h_thursday : Nat := 9)
  (h_friday : Nat := 6)
  (h_saturday : Nat := 4) :
  hoursWorked h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday = 44 :=
by
  unfold hoursWorked
  sorry

end Frank_work_hours_l112_112867


namespace bus_departure_l112_112569

theorem bus_departure (current_people : ℕ) (min_people : ℕ) (required_people : ℕ) 
  (h1 : current_people = 9) (h2 : min_people = 16) : required_people = 7 :=
by 
  sorry

end bus_departure_l112_112569


namespace balloons_initial_count_l112_112692

theorem balloons_initial_count (x : ℕ) (h : x + 13 = 60) : x = 47 :=
by
  -- proof skipped
  sorry

end balloons_initial_count_l112_112692


namespace square_area_ratio_l112_112019

theorem square_area_ratio (s₁ s₂ d₂ : ℝ)
  (h1 : s₁ = 2 * d₂)
  (h2 : d₂ = s₂ * Real.sqrt 2) :
  (s₁^2) / (s₂^2) = 8 :=
by
  sorry

end square_area_ratio_l112_112019


namespace mean_of_four_integers_l112_112837

theorem mean_of_four_integers (x : ℝ) (h : (78 + 83 + 82 + x) / 4 = 80) : x = 77 ∧ x = 80 - 3 :=
by
  have h1 : 78 + 83 + 82 + x = 4 * 80 := by sorry
  have h2 : 78 + 83 + 82 = 243 := by sorry
  have h3 : 243 + x = 320 := by sorry
  have h4 : x = 320 - 243 := by sorry
  have h5 : x = 77 := by sorry
  have h6 : x = 80 - 3 := by sorry
  exact ⟨h5, h6⟩

end mean_of_four_integers_l112_112837


namespace find_oranges_to_put_back_l112_112721

theorem find_oranges_to_put_back (A O x : ℕ) (h₁ : A + O = 15) (h₂ : 40 * A + 60 * O = 720) (h₃ : (360 + 360 - 60 * x) / (15 - x) = 45) : x = 3 := by
  sorry

end find_oranges_to_put_back_l112_112721


namespace other_endpoint_of_diameter_l112_112076

-- Define the basic data
def center : ℝ × ℝ := (5, 2)
def endpoint1 : ℝ × ℝ := (0, -3)
def endpoint2 : ℝ × ℝ := (10, 7)

-- State the final properties to be proved
theorem other_endpoint_of_diameter :
  ∃ (e2 : ℝ × ℝ), e2 = endpoint2 ∧
    dist center endpoint2 = dist endpoint1 center :=
sorry

end other_endpoint_of_diameter_l112_112076


namespace unique_two_digit_number_l112_112568

theorem unique_two_digit_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 99) : 
  (13 * n) % 100 = 42 → n = 34 :=
by
  sorry

end unique_two_digit_number_l112_112568


namespace add_one_gt_add_one_l112_112117

theorem add_one_gt_add_one (a b c : ℝ) (h : a > b) : (a + c) > (b + c) :=
sorry

end add_one_gt_add_one_l112_112117


namespace beads_per_necklace_l112_112204

theorem beads_per_necklace (n : ℕ) (b : ℕ) (total_beads : ℕ) (total_necklaces : ℕ)
  (h1 : total_necklaces = 6) (h2 : total_beads = 18) (h3 : b * total_necklaces = total_beads) :
  b = 3 :=
by {
  sorry
}

end beads_per_necklace_l112_112204


namespace quadratic_roots_value_r_l112_112625

theorem quadratic_roots_value_r
  (a b m p r : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h_root1 : a^2 - m*a + 3 = 0)
  (h_root2 : b^2 - m*b + 3 = 0)
  (h_ab : a * b = 3)
  (h_root3 : (a + 1/b) * (b + 1/a) = r) :
  r = 16 / 3 :=
sorry

end quadratic_roots_value_r_l112_112625


namespace remainder_of_sum_l112_112327

theorem remainder_of_sum (c d : ℤ) (p q : ℤ) (h1 : c = 60 * p + 53) (h2 : d = 45 * q + 28) : 
  (c + d) % 15 = 6 := 
by
  sorry

end remainder_of_sum_l112_112327


namespace sarah_meets_vegetable_requirement_l112_112584

def daily_vegetable_requirement : ℝ := 2
def total_days : ℕ := 5
def weekly_requirement : ℝ := daily_vegetable_requirement * total_days

def sunday_consumption : ℝ := 3
def monday_consumption : ℝ := 1.5
def tuesday_consumption : ℝ := 1.5
def wednesday_consumption : ℝ := 1.5
def thursday_consumption : ℝ := 2.5

def total_consumption : ℝ := sunday_consumption + monday_consumption + tuesday_consumption + wednesday_consumption + thursday_consumption

theorem sarah_meets_vegetable_requirement : total_consumption = weekly_requirement :=
by
  sorry

end sarah_meets_vegetable_requirement_l112_112584


namespace find_length_of_shop_l112_112547

noncomputable def length_of_shop (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  (monthly_rent * 12) / annual_rent_per_sqft / width

theorem find_length_of_shop
  (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ)
  (h_monthly_rent : monthly_rent = 3600)
  (h_width : width = 20)
  (h_annual_rent_per_sqft : annual_rent_per_sqft = 120) 
  : length_of_shop monthly_rent width annual_rent_per_sqft = 18 := 
sorry

end find_length_of_shop_l112_112547


namespace product_of_solutions_abs_eq_l112_112192

theorem product_of_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, |6 * x1 + 2| + 5 = 47 ∧ |6 * x2 + 2| + 5 = 47 ∧ x ≠ x1 ∧ x ≠ x2 ∧ x1 * x2 = -440 / 9) :=
by
  sorry

end product_of_solutions_abs_eq_l112_112192


namespace reduced_price_per_kg_l112_112982

theorem reduced_price_per_kg {P R : ℝ} (H1 : R = 0.75 * P) (H2 : 1100 = 1100 / P * P) (H3 : 1100 = (1100 / P + 5) * R) : R = 55 :=
by sorry

end reduced_price_per_kg_l112_112982


namespace point_reflection_l112_112624

-- Define the original point and the reflection function
structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

-- Define the original point
def M : Point := ⟨-5, 2⟩

-- State the theorem to prove the reflection
theorem point_reflection : reflect_y_axis M = ⟨5, 2⟩ :=
  sorry

end point_reflection_l112_112624


namespace proof_problem_l112_112448

noncomputable def sqrt_repeated (x : ℕ) (y : ℕ) : ℕ :=
Nat.sqrt x ^ y

theorem proof_problem (x y z : ℕ) :
  (sqrt_repeated x y = z) ↔ 
  ((∃ t : ℕ, x = t^2 ∧ y = 1 ∧ z = t) ∨ (x = 0 ∧ z = 0 ∧ y ≠ 0)) :=
sorry

end proof_problem_l112_112448


namespace factorization_problem_l112_112276

theorem factorization_problem 
  (C D : ℤ)
  (h1 : 15 * y ^ 2 - 76 * y + 48 = (C * y - 16) * (D * y - 3))
  (h2 : C * D = 15)
  (h3 : C * (-3) + D * (-16) = -76)
  (h4 : (-16) * (-3) = 48) : 
  C * D + C = 20 :=
by { sorry }

end factorization_problem_l112_112276


namespace convex_pentagon_largest_angle_l112_112581

theorem convex_pentagon_largest_angle 
  (x : ℝ)
  (h1 : (x + 2) + (2 * x + 3) + (3 * x + 6) + (4 * x + 5) + (5 * x + 4) = 540) :
  5 * x + 4 = 532 / 3 :=
by
  sorry

end convex_pentagon_largest_angle_l112_112581


namespace average_monthly_growth_rate_correct_l112_112745

theorem average_monthly_growth_rate_correct:
  (∃ x : ℝ, 30000 * (1 + x)^2 = 36300) ↔ 3 * (1 + x)^2 = 3.63 := 
by {
  sorry -- proof placeholder
}

end average_monthly_growth_rate_correct_l112_112745


namespace sum_numbers_l112_112452

theorem sum_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end sum_numbers_l112_112452


namespace angles_geometric_sequence_count_l112_112202

def is_geometric_sequence (a b c : ℝ) : Prop :=
  (a = b * c) ∨ (b = a * c) ∨ (c = a * b)

theorem angles_geometric_sequence_count : 
  ∃! (angles : Finset ℝ), 
    (∀ θ ∈ angles, 0 < θ ∧ θ < 2 * Real.pi ∧ ¬∃ k : ℤ, θ = k * (Real.pi / 2)) ∧
    ∀ θ ∈ angles,
      is_geometric_sequence (Real.sin θ ^ 2) (Real.cos θ) (Real.tan θ) ∧
    angles.card = 2 := 
sorry

end angles_geometric_sequence_count_l112_112202


namespace algebra_inequality_l112_112763

theorem algebra_inequality (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end algebra_inequality_l112_112763


namespace average_greater_median_l112_112354

theorem average_greater_median :
  let h : ℝ := 120
  let s1 : ℝ := 4
  let s2 : ℝ := 4
  let s3 : ℝ := 5
  let s4 : ℝ := 7
  let s5 : ℝ := 9
  let median : ℝ := (s3 + s4) / 2
  let average : ℝ := (h + s1 + s2 + s3 + s4 + s5) / 6
  average - median = 18.8333 := by
    sorry

end average_greater_median_l112_112354


namespace f_bound_l112_112809

noncomputable def f : ℕ+ → ℝ := sorry

axiom f_1 : f 1 = 3 / 2
axiom f_ineq (x y : ℕ+) : f (x + y) ≥ (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_bound (x : ℕ+) : f x ≥ 1 / 4 * x * (x + 1) * (2 * x + 1) := sorry

end f_bound_l112_112809


namespace problem_demo_l112_112027

open Set

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem problem_demo : S ∩ (U \ T) = {1, 2, 4} :=
by
  sorry

end problem_demo_l112_112027


namespace larger_model_ratio_smaller_model_ratio_l112_112223

-- Definitions for conditions
def statue_height := 305 -- The height of the actual statue in feet
def larger_model_height := 10 -- The height of the larger model in inches
def smaller_model_height := 5 -- The height of the smaller model in inches

-- The ratio calculation for larger model
theorem larger_model_ratio : 
  (statue_height : ℝ) / (larger_model_height : ℝ) = 30.5 := by
  sorry

-- The ratio calculation for smaller model
theorem smaller_model_ratio : 
  (statue_height : ℝ) / (smaller_model_height : ℝ) = 61 := by
  sorry

end larger_model_ratio_smaller_model_ratio_l112_112223


namespace average_weight_of_girls_l112_112212

theorem average_weight_of_girls (avg_weight_boys : ℕ) (num_boys : ℕ) (avg_weight_class : ℕ) (num_students : ℕ) :
  num_boys = 15 →
  avg_weight_boys = 48 →
  num_students = 25 →
  avg_weight_class = 45 →
  ( (avg_weight_class * num_students - avg_weight_boys * num_boys) / (num_students - num_boys) ) = 27 :=
by
  intros h_num_boys h_avg_weight_boys h_num_students h_avg_weight_class
  sorry

end average_weight_of_girls_l112_112212


namespace express_y_in_terms_of_x_l112_112091

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := 
by
  sorry

end express_y_in_terms_of_x_l112_112091


namespace height_relationship_l112_112790

theorem height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
sorry

end height_relationship_l112_112790


namespace calories_per_slice_l112_112147

theorem calories_per_slice (n k t c : ℕ) (h1 : n = 8) (h2 : k = n / 2) (h3 : k * c = t) (h4 : t = 1200) : c = 300 :=
by sorry

end calories_per_slice_l112_112147


namespace fraction_to_decimal_l112_112676

theorem fraction_to_decimal : (3 / 24 : ℚ) = 0.125 := 
by
  -- proof will be filled here
  sorry

end fraction_to_decimal_l112_112676


namespace value_of_x_add_y_l112_112299

theorem value_of_x_add_y (x y : ℝ) 
  (h1 : x + Real.sin y = 2023)
  (h2 : x + 2023 * Real.cos y = 2021)
  (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (3 * Real.pi / 4)) : 
  x + y = 2023 - (Real.sqrt 2) / 2 + (3 * Real.pi) / 4 := 
sorry

end value_of_x_add_y_l112_112299


namespace sin_arithmetic_sequence_l112_112423

noncomputable def sin_value (a : ℝ) := Real.sin (a * (Real.pi / 180))

theorem sin_arithmetic_sequence (a : ℝ) : 
  (0 < a) ∧ (a < 360) ∧ (sin_value a + sin_value (3 * a) = 2 * sin_value (2 * a)) ↔ a = 90 ∨ a = 270 :=
by 
  sorry

end sin_arithmetic_sequence_l112_112423


namespace bugs_eat_total_flowers_l112_112044

theorem bugs_eat_total_flowers :
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  total = 17 :=
by
  -- Applying given values to compute the total flowers eaten
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  
  -- Verify the total is 17
  have h_total : total = 17 := 
    by
    sorry

  -- Proving the final result
  exact h_total

end bugs_eat_total_flowers_l112_112044


namespace no_solution_to_system_l112_112443

theorem no_solution_to_system : ∀ (x y : ℝ), ¬ (y^2 - (⌊x⌋ : ℝ)^2 = 2001 ∧ x^2 + (⌊y⌋ : ℝ)^2 = 2001) :=
by sorry

end no_solution_to_system_l112_112443


namespace circle_properties_l112_112290

theorem circle_properties :
  ∃ p q s : ℝ, 
  (∀ x y : ℝ, x^2 + 16 * y + 89 = -y^2 - 12 * x ↔ (x + p)^2 + (y + q)^2 = s^2) ∧ 
  p + q + s = -14 + Real.sqrt 11 :=
by
  use -6, -8, Real.sqrt 11
  sorry

end circle_properties_l112_112290


namespace triangle_inequality_l112_112260

variables {R : Type*} [LinearOrderedField R]

theorem triangle_inequality 
  (a b c u v w : R)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (a + b + c) * (1 / u + 1 / v + 1 / w) ≤ 3 * (a / u + b / v + c / w) :=
sorry

end triangle_inequality_l112_112260


namespace points_for_correct_answer_l112_112116

theorem points_for_correct_answer
  (x y a b : ℕ)
  (hx : x - y = 7)
  (hsum : a + b = 43)
  (hw_score : a * x - b * (20 - x) = 328)
  (hz_score : a * y - b * (20 - y) = 27) :
  a = 25 := 
sorry

end points_for_correct_answer_l112_112116


namespace savings_after_increase_l112_112271

theorem savings_after_increase (salary savings_rate increase_rate : ℝ) (old_savings old_expenses new_expenses new_savings : ℝ)
  (h_salary : salary = 6000)
  (h_savings_rate : savings_rate = 0.2)
  (h_increase_rate : increase_rate = 0.2)
  (h_old_savings : old_savings = savings_rate * salary)
  (h_old_expenses : old_expenses = salary - old_savings)
  (h_new_expenses : new_expenses = old_expenses * (1 + increase_rate))
  (h_new_savings : new_savings = salary - new_expenses) :
  new_savings = 240 :=
by sorry

end savings_after_increase_l112_112271


namespace benjamin_decade_expense_l112_112917

-- Define the constants
def yearly_expense : ℕ := 3000
def years : ℕ := 10

-- Formalize the statement
theorem benjamin_decade_expense : yearly_expense * years = 30000 := 
by
  sorry

end benjamin_decade_expense_l112_112917


namespace employees_paid_per_shirt_l112_112804

theorem employees_paid_per_shirt:
  let num_employees := 20
  let shirts_per_employee_per_day := 20
  let hours_per_shift := 8
  let wage_per_hour := 12
  let price_per_shirt := 35
  let nonemployee_expenses_per_day := 1000
  let profit_per_day := 9080
  let total_shirts_made_per_day := num_employees * shirts_per_employee_per_day
  let total_daily_wages := num_employees * hours_per_shift * wage_per_hour
  let total_revenue := total_shirts_made_per_day * price_per_shirt
  let per_shirt_payment := (total_revenue - (total_daily_wages + nonemployee_expenses_per_day)) / total_shirts_made_per_day
  per_shirt_payment = 27.70 :=
sorry

end employees_paid_per_shirt_l112_112804


namespace harry_fish_count_l112_112470

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l112_112470


namespace puppies_given_to_friends_l112_112413

def original_puppies : ℕ := 8
def current_puppies : ℕ := 4

theorem puppies_given_to_friends : original_puppies - current_puppies = 4 :=
by
  sorry

end puppies_given_to_friends_l112_112413


namespace monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l112_112997

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the problem stating that when a <= 0, f(x) is increasing on (0, +∞)
theorem monotonicity_increasing_when_a_nonpos (a : ℝ) (h : a ≤ 0) :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Define the problem stating that when a > 0, f(x) is increasing on (0, 1/a) and decreasing on (1/a, +∞)
theorem monotonicity_increasing_decreasing_when_a_pos (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x < (1 / a) → y < (1 / a) → f a x < f a y) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (1 / a) < x → (1 / a) < y → f a y < f a x) :=
sorry

-- Define the problem for the range of a such that f(x) < 0 for all x in (0, +∞)
theorem range_of_a_for_f_less_than_zero (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x < 0) ↔ a ∈ Set.Ioi (1 / Real.exp 1) :=
sorry

end monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l112_112997


namespace shorter_trisector_length_eq_l112_112186

theorem shorter_trisector_length_eq :
  ∀ (DE EF DF FG : ℝ), DE = 6 → EF = 8 → DF = Real.sqrt (DE^2 + EF^2) → 
  FG = 2 * (24 / (3 + 4 * Real.sqrt 3)) → 
  FG = (192 * Real.sqrt 3 - 144) / 39 :=
by
  intros
  sorry

end shorter_trisector_length_eq_l112_112186


namespace original_average_rent_is_800_l112_112641

def original_rent (A : ℝ) : Prop :=
  let friends : ℝ := 4
  let old_rent : ℝ := 800
  let increased_rent : ℝ := old_rent * 1.25
  let new_total_rent : ℝ := (850 * friends)
  old_rent * 4 - 800 + increased_rent = new_total_rent

theorem original_average_rent_is_800 (A : ℝ) : original_rent A → A = 800 :=
by 
  sorry

end original_average_rent_is_800_l112_112641


namespace range_of_a_l112_112738

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l112_112738


namespace simplify_expression_eq_l112_112567

theorem simplify_expression_eq (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by 
  sorry

end simplify_expression_eq_l112_112567


namespace weekly_allowance_is_8_l112_112611

variable (A : ℝ)

def condition_1 (A : ℝ) : Prop := ∃ A : ℝ, A / 2 + 8 = 12

theorem weekly_allowance_is_8 (A : ℝ) (h : condition_1 A) : A = 8 :=
sorry

end weekly_allowance_is_8_l112_112611


namespace cube_root_sum_lt_sqrt_sum_l112_112851

theorem cube_root_sum_lt_sqrt_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
    sorry

end cube_root_sum_lt_sqrt_sum_l112_112851


namespace value_calculation_l112_112742

-- Definition of constants used in the problem
def a : ℝ := 1.3333
def b : ℝ := 3.615
def expected_value : ℝ := 4.81998845

-- The proposition to be proven
theorem value_calculation : a * b = expected_value :=
by sorry

end value_calculation_l112_112742


namespace jellybean_total_l112_112333

theorem jellybean_total (large_jellybeans_per_glass : ℕ) 
  (small_jellybeans_per_glass : ℕ) 
  (num_large_glasses : ℕ) 
  (num_small_glasses : ℕ) 
  (h1 : large_jellybeans_per_glass = 50) 
  (h2 : small_jellybeans_per_glass = large_jellybeans_per_glass / 2) 
  (h3 : num_large_glasses = 5) 
  (h4 : num_small_glasses = 3) : 
  (num_large_glasses * large_jellybeans_per_glass + num_small_glasses * small_jellybeans_per_glass) = 325 :=
by
  sorry

end jellybean_total_l112_112333


namespace valid_triangle_count_l112_112191

def point := (ℤ × ℤ)

def isValidPoint (p : point) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4

def isCollinear (p1 p2 p3 : point) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def isValidTriangle (p1 p2 p3 : point) : Prop :=
  isValidPoint p1 ∧ isValidPoint p2 ∧ isValidPoint p3 ∧ ¬isCollinear p1 p2 p3

def numberOfValidTriangles : ℕ :=
  sorry -- This will contain the combinatorial calculations from the solution.

theorem valid_triangle_count : numberOfValidTriangles = 520 :=
  sorry -- Proof will show combinatorial result from counting non-collinear combinations.

end valid_triangle_count_l112_112191


namespace max_xy_l112_112845

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 2 * y = 12) : 
  xy ≤ 6 :=
sorry

end max_xy_l112_112845


namespace sum_of_consecutive_integers_l112_112461

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) = 358800) : 
  n + (n + 1) + (n + 2) + (n + 3) = 98 :=
sorry

end sum_of_consecutive_integers_l112_112461


namespace find_first_day_speed_l112_112995

theorem find_first_day_speed (t : ℝ) (d : ℝ) (v : ℝ) (h1 : d = 2.5) 
  (h2 : v * (t - 7/60) = d) (h3 : 10 * (t - 8/60) = d) : v = 9.375 :=
by {
  -- Proof omitted for brevity
  sorry
}

end find_first_day_speed_l112_112995


namespace solve_equation_l112_112365

theorem solve_equation {x : ℂ} : (x - 2)^4 + (x - 6)^4 = 272 →
  x = 6 ∨ x = 2 ∨ x = 4 + 2 * Complex.I ∨ x = 4 - 2 * Complex.I :=
by
  intro h
  sorry

end solve_equation_l112_112365


namespace smallest_n_with_314_in_decimal_l112_112964

theorem smallest_n_with_314_in_decimal {m n : ℕ} (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n) 
  (h_contains_314 : ∃ k : ℕ, (10^k * m) % n == 314) : n = 315 :=
sorry

end smallest_n_with_314_in_decimal_l112_112964


namespace inequality_holds_l112_112819

theorem inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
sorry

end inequality_holds_l112_112819


namespace palindrome_probability_divisible_by_7_l112_112345

-- Define the conditions
def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1001 * a + 110 * b

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Define the proof problem
theorem palindrome_probability_divisible_by_7 : 
  (∃ (n : ℕ), is_four_digit_palindrome n ∧ is_divisible_by_7 n) →
  ∃ p : ℚ, p = 1/5 :=
sorry

end palindrome_probability_divisible_by_7_l112_112345


namespace find_y_l112_112782

variable {L B y : ℝ}

theorem find_y (h1 : 2 * ((L + y) + (B + y)) - 2 * (L + B) = 16) : y = 4 :=
by
  sorry

end find_y_l112_112782


namespace arithmetic_expression_evaluation_l112_112714

theorem arithmetic_expression_evaluation : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = -3 := 
by
  sorry

end arithmetic_expression_evaluation_l112_112714


namespace odd_function_property_l112_112506

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) : ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by
  -- The proof is omitted as per the instruction
  sorry

end odd_function_property_l112_112506


namespace phone_extension_permutations_l112_112613

theorem phone_extension_permutations : 
  (∃ (l : List ℕ), l = [5, 7, 8, 9, 0] ∧ Nat.factorial l.length = 120) :=
sorry

end phone_extension_permutations_l112_112613


namespace flowers_given_to_mother_l112_112717

-- Definitions based on conditions:
def Alissa_flowers : Nat := 16
def Melissa_flowers : Nat := 16
def flowers_left : Nat := 14

-- The proof problem statement:
theorem flowers_given_to_mother :
  Alissa_flowers + Melissa_flowers - flowers_left = 18 := by
  sorry

end flowers_given_to_mother_l112_112717


namespace statement2_true_l112_112618

def digit : ℕ := sorry

def statement1 : Prop := digit = 2
def statement2 : Prop := digit ≠ 3
def statement3 : Prop := digit = 5
def statement4 : Prop := digit ≠ 6

def condition : Prop := (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (¬ statement1 ∨ ¬ statement2 ∨ ¬ statement3 ∨ ¬ statement4)

theorem statement2_true (h : condition) : statement2 :=
sorry

end statement2_true_l112_112618


namespace percentage_of_360_is_120_l112_112122

theorem percentage_of_360_is_120 (part whole : ℝ) (h1 : part = 120) (h2 : whole = 360) : 
  ((part / whole) * 100 = 33.33) :=
by
  sorry

end percentage_of_360_is_120_l112_112122


namespace needed_adjustment_l112_112377

def price_adjustment (P : ℝ) : ℝ :=
  let P_reduced := P - 0.20 * P
  let P_raised := P_reduced + 0.10 * P_reduced
  let P_target := P - 0.10 * P
  P_target - P_raised

theorem needed_adjustment (P : ℝ) : price_adjustment P = 2 * (P / 100) := sorry

end needed_adjustment_l112_112377


namespace sphere_surface_area_diameter_4_l112_112944

noncomputable def sphere_surface_area (d : ℝ) : ℝ :=
  4 * Real.pi * (d / 2) ^ 2

theorem sphere_surface_area_diameter_4 :
  sphere_surface_area 4 = 16 * Real.pi :=
by
  sorry

end sphere_surface_area_diameter_4_l112_112944


namespace find_integer_pairs_l112_112043

noncomputable def satisfies_equation (x y : ℤ) :=
  12 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 28 * (x + y)

theorem find_integer_pairs (m n : ℤ) :
  satisfies_equation (3 * m - 4 * n) (4 * n) :=
sorry

end find_integer_pairs_l112_112043


namespace ratio_spaghetti_to_manicotti_l112_112962

-- Definitions of the given conditions
def total_students : ℕ := 800
def spaghetti_preferred : ℕ := 320
def manicotti_preferred : ℕ := 160

-- The theorem statement
theorem ratio_spaghetti_to_manicotti : spaghetti_preferred / manicotti_preferred = 2 :=
by sorry

end ratio_spaghetti_to_manicotti_l112_112962


namespace total_people_correct_l112_112160

-- Define the daily changes as given conditions
def daily_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

-- Define the total number of people given 'a' and daily changes
def total_people (a : ℝ) : ℝ :=
  7 * a + daily_changes.sum

-- Lean statement for proving the total number of people
theorem total_people_correct (a : ℝ) : 
  total_people a = 7 * a + 13.2 :=
by
  -- This statement needs a proof, so we leave a placeholder 'sorry'
  sorry

end total_people_correct_l112_112160


namespace line_equation_l112_112435

theorem line_equation (P : ℝ × ℝ) (hP : P = (1, 5)) (h1 : ∃ a, a ≠ 0 ∧ (P.1 + P.2 = a)) (h2 : x_intercept = y_intercept) : 
  (∃ a, a ≠ 0 ∧ P = (a, 0) ∧ P = (0, a) → x + y - 6 = 0) ∨ (5*P.1 - P.2 = 0) :=
by
  sorry

end line_equation_l112_112435


namespace minimize_average_cost_l112_112744

noncomputable def average_comprehensive_cost (x : ℝ) : ℝ :=
  560 + 48 * x + 2160 * 10^6 / (2000 * x)

theorem minimize_average_cost : 
  ∃ x_min : ℝ, x_min ≥ 10 ∧ 
  ∀ x ≥ 10, average_comprehensive_cost x ≥ average_comprehensive_cost x_min :=
sorry

end minimize_average_cost_l112_112744


namespace point_not_on_graph_l112_112554

theorem point_not_on_graph : ¬ ∃ (x y : ℝ), (y = (x - 1) / (x + 2)) ∧ (x = -2) ∧ (y = 3) :=
by
  sorry

end point_not_on_graph_l112_112554


namespace cards_in_center_pile_l112_112096

/-- Represents the number of cards in each pile initially. -/
def initial_cards (x : ℕ) : Prop := x ≥ 2

/-- Represents the state of the piles after step 2. -/
def step2 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 2 ∧ right = x

/-- Represents the state of the piles after step 3. -/
def step3 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 3 ∧ right = x - 1

/-- Represents the state of the piles after step 4. -/
def step4 (x : ℕ) (left center : ℕ) : Prop :=
  left = 2 * x - 4 ∧ center = 5

/-- Prove that after performing all steps, the number of cards in the center pile is 5. -/
theorem cards_in_center_pile (x : ℕ) :
  initial_cards x →
  (∃ l₁ c₁ r₁, step2 x l₁ c₁ r₁) →
  (∃ l₂ c₂ r₂, step3 x l₂ c₂ r₂) →
  (∃ l₃ c₃, step4 x l₃ c₃) →
  ∃ (center_final : ℕ), center_final = 5 :=
by
  sorry

end cards_in_center_pile_l112_112096


namespace picnic_total_cost_is_correct_l112_112007

-- Define the conditions given in the problem
def number_of_people : Nat := 4
def cost_per_sandwich : Nat := 5
def cost_per_fruit_salad : Nat := 3
def sodas_per_person : Nat := 2
def cost_per_soda : Nat := 2
def number_of_snack_bags : Nat := 3
def cost_per_snack_bag : Nat := 4

-- Calculate the total cost based on the given conditions
def total_cost_sandwiches : Nat := number_of_people * cost_per_sandwich
def total_cost_fruit_salads : Nat := number_of_people * cost_per_fruit_salad
def total_cost_sodas : Nat := number_of_people * sodas_per_person * cost_per_soda
def total_cost_snack_bags : Nat := number_of_snack_bags * cost_per_snack_bag

def total_spent : Nat := total_cost_sandwiches + total_cost_fruit_salads + total_cost_sodas + total_cost_snack_bags

-- The statement we want to prove
theorem picnic_total_cost_is_correct : total_spent = 60 :=
by
  -- Proof would be written here
  sorry

end picnic_total_cost_is_correct_l112_112007


namespace slope_of_perpendicular_line_l112_112988

-- Define the line equation as a condition
def line_eqn (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of the given line from its equation
noncomputable def original_slope : ℝ := 2 / 3

-- Define the negative reciprocal of the original slope
noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

-- State the theorem
theorem slope_of_perpendicular_line : perp_slope original_slope = -3 / 2 :=
by 
  sorry

end slope_of_perpendicular_line_l112_112988


namespace tangent_line_through_point_l112_112884

theorem tangent_line_through_point (a : ℝ) : 
  ∃ l : ℝ → ℝ, 
    (∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → y = a) ∧ 
    (∀ x y : ℝ, y = l x → (x - 1)^2 + y^2 = 4) → 
    a = 0 :=
by
  sorry

end tangent_line_through_point_l112_112884


namespace even_iff_b_eq_zero_l112_112177

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Given that f' is an even function, prove that b = 0.
theorem even_iff_b_eq_zero (h : ∀ x : ℝ, f' x = f' (-x)) : b = 0 :=
  sorry

end even_iff_b_eq_zero_l112_112177


namespace find_x_l112_112211

variable (BrandA_millet : ℝ) (Mix_millet : ℝ) (Mix_ratio_A : ℝ) (Mix_ratio_B : ℝ)

axiom BrandA_contains_60_percent_millet : BrandA_millet = 0.60
axiom Mix_contains_50_percent_millet : Mix_millet = 0.50
axiom Mix_composition : Mix_ratio_A = 0.60 ∧ Mix_ratio_B = 0.40

theorem find_x (x : ℝ) :
  Mix_ratio_A * BrandA_millet + Mix_ratio_B * x = Mix_millet →
  x = 0.35 :=
by
  sorry

end find_x_l112_112211


namespace star_three_and_four_l112_112330

def star (a b : ℝ) : ℝ := 4 * a + 5 * b - 2 * a * b

theorem star_three_and_four : star 3 4 = 8 :=
by
  sorry

end star_three_and_four_l112_112330


namespace color_blocks_probability_at_least_one_box_match_l112_112422

/-- Given Ang, Ben, and Jasmin each having 6 blocks of different colors (red, blue, yellow, white, green, and orange) 
    and they independently place one of their blocks into each of 6 empty boxes, 
    the proof shows that the probability that at least one box receives 3 blocks all of the same color is 1/6. 
    Since 1/6 is equal to the fraction m/n where m=1 and n=6 are relatively prime, thus m+n=7. -/
theorem color_blocks_probability_at_least_one_box_match (p : ℕ × ℕ) (h : p = (1, 6)) : p.1 + p.2 = 7 :=
by {
  sorry
}

end color_blocks_probability_at_least_one_box_match_l112_112422


namespace simplify_fraction_l112_112923

-- Define the numbers involved and state their GCD
def num1 := 90
def num2 := 8100

-- State the GCD condition using a Lean 4 statement
def gcd_condition (a b : ℕ) := Nat.gcd a b = 90

-- Define the original fraction and the simplified fraction
def original_fraction := num1 / num2
def simplified_fraction := 1 / 90

-- State the proof problem that the original fraction simplifies to the simplified fraction
theorem simplify_fraction : gcd_condition num1 num2 → original_fraction = simplified_fraction := 
by
  sorry

end simplify_fraction_l112_112923


namespace verify_extrema_l112_112985

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end verify_extrema_l112_112985


namespace ratio_father_to_children_after_5_years_l112_112433

def father's_age := 15
def sum_children_ages := father's_age / 3

def father's_age_after_5_years := father's_age + 5
def sum_children_ages_after_5_years := sum_children_ages + 10

theorem ratio_father_to_children_after_5_years :
  father's_age_after_5_years / sum_children_ages_after_5_years = 4 / 3 := by
  sorry

end ratio_father_to_children_after_5_years_l112_112433


namespace jessica_has_100_dollars_l112_112194

-- Define the variables for Rodney, Ian, and Jessica
variables (R I J : ℝ)

-- Given conditions
axiom rodney_more_than_ian : R = I + 35
axiom ian_half_of_jessica : I = J / 2
axiom jessica_more_than_rodney : J = R + 15

-- The statement to prove
theorem jessica_has_100_dollars : J = 100 :=
by
  -- Proof will be completed here
  sorry

end jessica_has_100_dollars_l112_112194


namespace father_twice_marika_age_in_2036_l112_112910

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l112_112910


namespace length_of_other_diagonal_l112_112387

theorem length_of_other_diagonal (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 15) (h2 : A = 150) : d2 = 20 :=
by
  sorry

end length_of_other_diagonal_l112_112387


namespace percentage_of_fish_gone_bad_l112_112273

-- Definitions based on conditions
def fish_per_roll : ℕ := 40
def total_fish_bought : ℕ := 400
def sushi_rolls_made : ℕ := 8

-- Definition of fish calculations
def total_fish_used (rolls: ℕ) (per_roll: ℕ) : ℕ := rolls * per_roll
def fish_gone_bad (total : ℕ) (used : ℕ) : ℕ := total - used
def percentage (part : ℕ) (whole : ℕ) : ℚ := (part : ℚ) / (whole : ℚ) * 100

-- Theorem to prove the percentage of bad fish
theorem percentage_of_fish_gone_bad :
  percentage (fish_gone_bad total_fish_bought (total_fish_used sushi_rolls_made fish_per_roll)) total_fish_bought = 20 := by
  sorry

end percentage_of_fish_gone_bad_l112_112273


namespace complete_the_square_l112_112928

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end complete_the_square_l112_112928


namespace train_speed_in_kmh_l112_112183

theorem train_speed_in_kmh 
  (train_length : ℕ) 
  (crossing_time : ℕ) 
  (conversion_factor : ℕ) 
  (hl : train_length = 120) 
  (ht : crossing_time = 6) 
  (hc : conversion_factor = 36) :
  train_length / crossing_time * conversion_factor / 10 = 72 := by
  sorry

end train_speed_in_kmh_l112_112183


namespace tenth_number_drawn_eq_195_l112_112720

noncomputable def total_students : Nat := 1000
noncomputable def sample_size : Nat := 50
noncomputable def first_selected_number : Nat := 15  -- Note: 0015 is 15 in natural number

theorem tenth_number_drawn_eq_195 
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_selected_number = 15) :
  15 + (20 * 9) = 195 := 
by
  sorry

end tenth_number_drawn_eq_195_l112_112720


namespace alcohol_concentration_l112_112560

theorem alcohol_concentration (x : ℝ) (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.35 →
  target_concentration = 0.50 →
  (2.1 + x) / (6 + x) = target_concentration →
  x = 1.8 :=
by
  intros h1 h2 h3 h4
  sorry

end alcohol_concentration_l112_112560


namespace additional_fee_per_minute_for_second_plan_l112_112441

theorem additional_fee_per_minute_for_second_plan :
  (∃ x : ℝ, (22 + 0.13 * 280 = 8 + x * 280) ∧ x = 0.18) :=
sorry

end additional_fee_per_minute_for_second_plan_l112_112441


namespace subtracting_is_adding_opposite_l112_112042

theorem subtracting_is_adding_opposite (a b : ℚ) : a - b = a + (-b) :=
by sorry

end subtracting_is_adding_opposite_l112_112042


namespace necessary_and_sufficient_condition_l112_112897

-- Define the conditions and question in Lean 4
variable (a : ℝ) 

-- State the theorem based on the conditions and the correct answer
theorem necessary_and_sufficient_condition :
  (a > 0) ↔ (
    let z := (⟨-a, -5⟩ : ℂ)
    ∃ (x y : ℝ), (z = x + y * I) ∧ x < 0 ∧ y < 0
  ) := by
  sorry

end necessary_and_sufficient_condition_l112_112897


namespace sqrt8_same_type_as_sqrt2_l112_112746

theorem sqrt8_same_type_as_sqrt2 :
  (∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 8) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 4) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 6) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 10) :=
by
  sorry

end sqrt8_same_type_as_sqrt2_l112_112746


namespace simplify_expression_l112_112647

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l112_112647


namespace cubing_identity_l112_112540

theorem cubing_identity (x : ℂ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
  sorry

end cubing_identity_l112_112540


namespace bakery_baguettes_l112_112941

theorem bakery_baguettes : 
  ∃ B : ℕ, 
  (∃ B : ℕ, 3 * B - 138 = 6) ∧ 
  B = 48 :=
by
  sorry

end bakery_baguettes_l112_112941


namespace cost_price_of_pots_l112_112417

variable (C : ℝ)

-- Define the conditions
def selling_price (C : ℝ) := 1.25 * C
def total_revenue (selling_price : ℝ) := 150 * selling_price

-- State the main proof goal
theorem cost_price_of_pots (h : total_revenue (selling_price C) = 450) : C = 2.4 := by
  sorry

end cost_price_of_pots_l112_112417


namespace line_intersects_semicircle_at_two_points_l112_112325

theorem line_intersects_semicircle_at_two_points
  (m : ℝ) :
  (3 ≤ m ∧ m < 3 * Real.sqrt 2) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ (y₁ = -x₁ + m ∧ y₁ = Real.sqrt (9 - x₁^2)) ∧ (y₂ = -x₂ + m ∧ y₂ = Real.sqrt (9 - x₂^2))) :=
by
  -- The proof goes here
  sorry

end line_intersects_semicircle_at_two_points_l112_112325


namespace salary_calculation_l112_112852

variable {A B : ℝ}

theorem salary_calculation (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) : A = 4500 :=
by
  sorry

end salary_calculation_l112_112852


namespace smallest_yellow_candies_l112_112916
open Nat

theorem smallest_yellow_candies 
  (h_red : ∃ c : ℕ, 16 * c = 720)
  (h_green : ∃ c : ℕ, 18 * c = 720)
  (h_blue : ∃ c : ℕ, 20 * c = 720)
  : ∃ n : ℕ, 30 * n = 720 ∧ n = 24 := 
by
  -- Provide the proof here
  sorry

end smallest_yellow_candies_l112_112916


namespace smallest_possible_value_expression_l112_112253

open Real

noncomputable def min_expression_value (a b c : ℝ) : ℝ :=
  (a + b)^2 + (b - c)^2 + (c - a)^2 / a^2

theorem smallest_possible_value_expression :
  ∀ (a b c : ℝ), a > b → b > c → a + c = 2 * b → a ≠ 0 → min_expression_value a b c = 7 / 2 := by
  sorry

end smallest_possible_value_expression_l112_112253


namespace sequence_a31_value_l112_112163

theorem sequence_a31_value 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h₀ : a 1 = 0) 
  (h₁ : ∀ n, a (n + 1) = a n + b n) 
  (h₂ : b 15 + b 16 = 15)
  (h₃ : ∀ m n : ℕ, (b n - b m) = (n - m) * (b 2 - b 1)) :
  a 31 = 225 :=
by
  sorry

end sequence_a31_value_l112_112163


namespace percentage_of_360_equals_126_l112_112947

/-- 
  Prove that (126 / 360) * 100 equals 35.
-/
theorem percentage_of_360_equals_126 : (126 / 360 : ℝ) * 100 = 35 := by
  sorry

end percentage_of_360_equals_126_l112_112947


namespace total_fraction_inspected_l112_112483

-- Define the fractions of products inspected by John, Jane, and Roy.
variables (J N R : ℝ)
-- Define the rejection rates for John, Jane, and Roy.
variables (rJ rN rR : ℝ)
-- Define the total rejection rate.
variable (r_total : ℝ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  (rJ = 0.007) ∧ (rN = 0.008) ∧ (rR = 0.01) ∧ (r_total = 0.0085) ∧
  (0.007 * J + 0.008 * N + 0.01 * R = 0.0085)

-- The proof statement that the total fraction of products inspected is 1.
theorem total_fraction_inspected (h : conditions J N R rJ rN rR r_total) : J + N + R = 1 :=
sorry

end total_fraction_inspected_l112_112483


namespace no_partition_equal_product_l112_112306

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end no_partition_equal_product_l112_112306


namespace average_speed_round_trip_l112_112234

variable (D : ℝ) (u v : ℝ)
  
theorem average_speed_round_trip (h1 : u = 96) (h2 : v = 88) : 
  (2 * u * v) / (u + v) = 91.73913043 := 
by 
  sorry

end average_speed_round_trip_l112_112234


namespace James_balloons_l112_112376

theorem James_balloons (A J : ℕ) (h1 : A = 513) (h2 : J = A + 208) : J = 721 :=
by {
  sorry
}

end James_balloons_l112_112376


namespace range_m_l112_112300

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_m (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ (-1 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_m_l112_112300


namespace average_brown_mms_per_bag_l112_112402

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end average_brown_mms_per_bag_l112_112402


namespace range_of_a_and_m_l112_112622

open Set

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 1 = 0}

-- Conditions as hypotheses
def condition1 : A ∪ B a = A := sorry
def condition2 : A ∩ C m = C m := sorry

-- Theorem to prove the correct range of a and m
theorem range_of_a_and_m : (a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2) :=
by
  -- Proof goes here
  sorry

end range_of_a_and_m_l112_112622


namespace solve_system_of_equations_l112_112437

theorem solve_system_of_equations :
  ∃ x y z : ℚ, 
    (y * z = 3 * y + 2 * z - 8) ∧
    (z * x = 4 * z + 3 * x - 8) ∧
    (x * y = 2 * x + y - 1) ∧
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ 
     (x = 3 ∧ y = 5 / 2 ∧ z = -1)) := 
by
  sorry

end solve_system_of_equations_l112_112437


namespace equivalent_operation_l112_112449

theorem equivalent_operation : 
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  (initial_op * multiply_3_2) = (5 / 4 : ℝ) :=
by
  -- setup operations
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  -- state the goal
  have h : (initial_op * multiply_3_2) = (5 / 4 : ℝ) := sorry
  exact h

end equivalent_operation_l112_112449


namespace jenna_peeled_potatoes_l112_112498

-- Definitions of constants
def initial_potatoes : ℕ := 60
def homer_rate : ℕ := 4
def jenna_rate : ℕ := 6
def combined_rate : ℕ := homer_rate + jenna_rate
def homer_time : ℕ := 6
def remaining_potatoes : ℕ := initial_potatoes - (homer_rate * homer_time)
def combined_time : ℕ := 4 -- Rounded from 3.6

-- Statement to prove
theorem jenna_peeled_potatoes : remaining_potatoes / combined_rate * jenna_rate = 24 :=
by
  sorry

end jenna_peeled_potatoes_l112_112498


namespace maximum_garden_area_l112_112503

theorem maximum_garden_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 400) : 
  l * w ≤ 10000 :=
by {
  -- proving the theorem
  sorry
}

end maximum_garden_area_l112_112503


namespace james_vs_combined_l112_112277

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687
def olivia_balloons : ℕ := 395
def combined_balloons : ℕ := amy_balloons + felix_balloons + olivia_balloons

theorem james_vs_combined :
  1222 = 1222 ∧ 513 = 513 ∧ 687 = 687 ∧ 395 = 395 → combined_balloons - james_balloons = 373 := by
  sorry

end james_vs_combined_l112_112277


namespace quadratic_extreme_values_l112_112356

theorem quadratic_extreme_values (y1 y2 y3 y4 : ℝ) 
  (h1 : y2 < y3) 
  (h2 : y3 = y4) 
  (h3 : ∀ x, ∃ (a b c : ℝ), ∀ y, y = a * x * x + b * x + c) :
  (y1 < y2) ∧ (y2 < y3) :=
by
  sorry

end quadratic_extreme_values_l112_112356


namespace aquarium_pufferfish_problem_l112_112107

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end aquarium_pufferfish_problem_l112_112107


namespace red_blue_beads_ratio_l112_112537

-- Definitions based on the conditions
def has_red_beads (betty : Type) := betty → ℕ
def has_blue_beads (betty : Type) := betty → ℕ

def betty : Type := Unit

-- Given conditions
def num_red_beads : has_red_beads betty := λ _ => 30
def num_blue_beads : has_blue_beads betty := λ _ => 20
def red_to_blue_ratio := 3 / 2

-- Theorem to prove the ratio
theorem red_blue_beads_ratio (R B: ℕ) (h_red : R = 30) (h_blue : B = 20) :
  (R / gcd R B) / (B / gcd R B ) = red_to_blue_ratio :=
by sorry

end red_blue_beads_ratio_l112_112537


namespace cook_weave_l112_112594

theorem cook_weave (Y C W OC CY CYW : ℕ) (hY : Y = 25) (hC : C = 15) (hW : W = 8) (hOC : OC = 2)
  (hCY : CY = 7) (hCYW : CYW = 3) : 
  ∃ (CW : ℕ), CW = 9 :=
by 
  have CW : ℕ := C - OC - (CY - CYW) 
  use CW
  sorry

end cook_weave_l112_112594


namespace antifreeze_solution_l112_112153

theorem antifreeze_solution (x : ℝ) 
  (h1 : 26 * x + 13 * 0.54 = 39 * 0.58) : 
  x = 0.6 := 
by 
  sorry

end antifreeze_solution_l112_112153


namespace heather_blocks_remaining_l112_112170

-- Definitions of the initial amount of blocks and the amount shared
def initial_blocks : ℕ := 86
def shared_blocks : ℕ := 41

-- The statement to be proven
theorem heather_blocks_remaining : (initial_blocks - shared_blocks = 45) :=
by sorry

end heather_blocks_remaining_l112_112170


namespace fraction_zero_implies_x_eq_two_l112_112024

theorem fraction_zero_implies_x_eq_two (x : ℝ) (h : (x^2 - 4) / (x + 2) = 0) : x = 2 :=
sorry

end fraction_zero_implies_x_eq_two_l112_112024


namespace handshake_count_l112_112372

def total_handshakes (men women : ℕ) := 
  (men * (men - 1)) / 2 + men * (women - 1)

theorem handshake_count :
  let men := 13
  let women := 13
  total_handshakes men women = 234 :=
by
  sorry

end handshake_count_l112_112372


namespace interval_monotonic_increase_max_min_values_range_of_m_l112_112169

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

-- The interval of monotonic increase for f(x)
theorem interval_monotonic_increase :
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} = 
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} := 
by sorry

-- Maximum and minimum values of f(x) when x ∈ [π/4, π/2]
theorem max_min_values (x : ℝ) (h : x ∈ Set.Icc (π / 4) (π / 2)) :
  (f x ≤ 0 ∧ (f x = 0 ↔ x = π / 3)) ∧ (f x ≥ -1/2 ∧ (f x = -1/2 ↔ x = π / 2)) :=
by sorry

-- Range of m for the inequality |f(x) - m| < 1 when x ∈ [π/4, π/2]
theorem range_of_m (m : ℝ) (h : ∀ x ∈ Set.Icc (π / 4) (π / 2), |f x - m| < 1) :
  m ∈ Set.Ioo (-1) (1/2) :=
by sorry

end interval_monotonic_increase_max_min_values_range_of_m_l112_112169


namespace academic_integers_l112_112063

def is_academic (n : ℕ) (h : n ≥ 2) : Prop :=
  ∃ (S P : Finset ℕ), (S ∩ P = ∅) ∧ (S ∪ P = Finset.range (n + 1)) ∧ (S.sum id = P.prod id)

theorem academic_integers :
  { n | ∃ h : n ≥ 2, is_academic n h } = { n | n = 3 ∨ n ≥ 5 } :=
by
  sorry

end academic_integers_l112_112063


namespace jacob_younger_than_michael_l112_112254

variables (J M : ℕ)

theorem jacob_younger_than_michael (h1 : M + 9 = 2 * (J + 9)) (h2 : J = 5) : M - J = 14 :=
by
  -- Insert proof steps here
  sorry

end jacob_younger_than_michael_l112_112254


namespace total_pages_in_book_is_250_l112_112349

-- Definitions
def avg_pages_first_part := 36
def days_first_part := 3
def avg_pages_second_part := 44
def days_second_part := 3
def pages_last_day := 10

-- Calculate total pages
def total_pages := (days_first_part * avg_pages_first_part) + (days_second_part * avg_pages_second_part) + pages_last_day

-- Theorem statement
theorem total_pages_in_book_is_250 : total_pages = 250 := by
  sorry

end total_pages_in_book_is_250_l112_112349


namespace scientific_notation_of_8450_l112_112446

theorem scientific_notation_of_8450 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (8450 : ℝ) = a * 10^n ∧ (a = 8.45) ∧ (n = 3) :=
sorry

end scientific_notation_of_8450_l112_112446


namespace B_work_days_l112_112241

theorem B_work_days (A B C : ℕ) (hA : A = 15) (hC : C = 30) (H : (5 / 15) + ((10 * (1 / C + 1 / B)) / (1 / C + 1 / B)) = 1) : B = 30 := by
  sorry

end B_work_days_l112_112241


namespace cookies_per_box_correct_l112_112052

variable (cookies_per_box : ℕ)

-- Define the conditions
def morning_cookie : ℕ := 1 / 2
def bed_cookie : ℕ := 1 / 2
def day_cookies : ℕ := 2
def daily_cookies := morning_cookie + bed_cookie + day_cookies

def days : ℕ := 30
def total_cookies := days * daily_cookies

def boxes : ℕ := 2
def total_cookies_in_boxes : ℕ := cookies_per_box * boxes

-- Theorem we want to prove
theorem cookies_per_box_correct :
  total_cookies_in_boxes = 90 → cookies_per_box = 45 :=
by
  sorry

end cookies_per_box_correct_l112_112052


namespace linear_function_quadrants_l112_112981

theorem linear_function_quadrants (k b : ℝ) :
  (∀ x, (0 < x → 0 < k * x + b) ∧ (x < 0 → 0 < k * x + b) ∧ (x < 0 → k * x + b < 0)) →
  k > 0 ∧ b > 0 :=
by
  sorry

end linear_function_quadrants_l112_112981


namespace number_of_tables_l112_112295

-- Define conditions
def chairs_in_base5 : ℕ := 310  -- chairs in base-5
def chairs_base10 : ℕ := 3 * 5^2 + 1 * 5^1 + 0 * 5^0  -- conversion to base-10
def people_per_table : ℕ := 3

-- The theorem to prove
theorem number_of_tables : chairs_base10 / people_per_table = 26 := by
  -- include the automatic proof here
  sorry

end number_of_tables_l112_112295


namespace john_uber_profit_l112_112025

theorem john_uber_profit
  (P0 : ℝ) (T : ℝ) (P : ℝ)
  (hP0 : P0 = 18000)
  (hT : T = 6000)
  (hP : P = 18000) :
  P + (P0 - T) = 30000 :=
by
  sorry

end john_uber_profit_l112_112025


namespace linear_function_properties_l112_112347

def linear_function (x : ℝ) : ℝ := -2 * x + 1

theorem linear_function_properties :
  (∀ x, linear_function x = -2 * x + 1) ∧
  (∀ x₁ x₂, x₁ < x₂ → linear_function x₁ > linear_function x₂) ∧
  (linear_function 0 = 1) ∧
  ((∃ x, x > 0 ∧ linear_function x > 0) ∧ (∃ x, x < 0 ∧ linear_function x > 0) ∧ (∃ x, x > 0 ∧ linear_function x < 0))
  :=
by
  sorry

end linear_function_properties_l112_112347


namespace division_sequence_l112_112931

theorem division_sequence : (120 / 5) / 2 / 3 = 4 := by
  sorry

end division_sequence_l112_112931


namespace roots_sum_reciprocal_squares_l112_112392

theorem roots_sum_reciprocal_squares (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + bc + ca = 20) (h3 : abc = 3) :
  (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2) = 328 / 9 := 
by
  sorry

end roots_sum_reciprocal_squares_l112_112392


namespace chord_length_of_intersection_l112_112524

theorem chord_length_of_intersection 
  (A B C : ℝ) (x0 y0 r : ℝ)
  (line_eq : A * x0 + B * y0 + C = 0)
  (circle_eq : (x0 - 1)^2 + (y0 - 3)^2 = r^2) 
  (A_line : A = 4) (B_line : B = -3) (C_line : C = 0) 
  (x0_center : x0 = 1) (y0_center : y0 = 3) (r_circle : r^2 = 10) :
  2 * (Real.sqrt (r^2 - ((A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2)))^2)) = 6 :=
by
  sorry

end chord_length_of_intersection_l112_112524


namespace total_photos_newspaper_l112_112112

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l112_112112


namespace skirt_more_than_pants_l112_112032

def amount_cut_off_skirt : ℝ := 0.75
def amount_cut_off_pants : ℝ := 0.5

theorem skirt_more_than_pants : 
  amount_cut_off_skirt - amount_cut_off_pants = 0.25 := 
by
  sorry

end skirt_more_than_pants_l112_112032


namespace katarina_miles_l112_112633

theorem katarina_miles 
  (total_miles : ℕ) 
  (miles_harriet : ℕ) 
  (miles_tomas : ℕ)
  (miles_tyler : ℕ)
  (miles_katarina : ℕ) 
  (combined_miles : total_miles = 195) 
  (same_miles : miles_tomas = miles_harriet ∧ miles_tyler = miles_harriet)
  (harriet_miles : miles_harriet = 48) :
  miles_katarina = 51 :=
sorry

end katarina_miles_l112_112633


namespace inequality_solution_l112_112938

theorem inequality_solution (x : ℝ) (h : |(x + 4) / 2| < 3) : -10 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l112_112938


namespace sin_alpha_beta_l112_112135

theorem sin_alpha_beta (a b c α β : Real) (h₁ : a * Real.cos α + b * Real.sin α + c = 0)
  (h₂ : a * Real.cos β + b * Real.sin β + c = 0) (h₃ : 0 < α) (h₄ : α < β) (h₅ : β < π) :
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
by 
  sorry

end sin_alpha_beta_l112_112135


namespace selling_price_eq_100_l112_112899

variable (CP SP : ℝ)

-- Conditions
def gain : ℝ := 20
def gain_percentage : ℝ := 0.25

-- The proof of the selling price
theorem selling_price_eq_100
  (h1 : gain = 20)
  (h2 : gain_percentage = 0.25)
  (h3 : gain = gain_percentage * CP)
  (h4 : SP = CP + gain) :
  SP = 100 := sorry

end selling_price_eq_100_l112_112899


namespace Lois_books_total_l112_112621

-- Definitions based on the conditions
def initial_books : ℕ := 150
def books_given_to_nephew : ℕ := initial_books / 4
def remaining_books : ℕ := initial_books - books_given_to_nephew
def non_fiction_books : ℕ := remaining_books * 60 / 100
def kept_non_fiction_books : ℕ := non_fiction_books / 2
def fiction_books : ℕ := remaining_books - non_fiction_books
def lent_fiction_books : ℕ := fiction_books / 3
def remaining_fiction_books : ℕ := fiction_books - lent_fiction_books
def newly_purchased_books : ℕ := 12

-- The total number of books Lois has now
def total_books_now : ℕ := kept_non_fiction_books + remaining_fiction_books + newly_purchased_books

-- Theorem statement
theorem Lois_books_total : total_books_now = 76 := by
  sorry

end Lois_books_total_l112_112621


namespace birth_date_of_older_friend_l112_112239

/-- Lean 4 statement for the proof problem --/
theorem birth_date_of_older_friend
  (d m y : ℕ)
  (h1 : y ≥ 1900 ∧ y < 2000)
  (h2 : d + 7 < 32) -- Assuming the month has at most 31 days
  (h3 : ((d+7) * 10^4 + m * 10^2 + y % 100) = 6 * (d * 10^4 + m * 10^2 + y % 100))
  (h4 : m > 0 ∧ m < 13)  -- Months are between 1 and 12
  (h5 : (d * 10^4 + m * 10^2 + y % 100) < (d+7) * 10^4 + m * 10^2 + y % 100) -- d < d+7 so older means smaller number
  : d = 1 ∧ m = 4 ∧ y = 1900 :=
by
  sorry -- Proof omitted

end birth_date_of_older_friend_l112_112239


namespace total_weight_of_dumbbell_system_l112_112667

-- Definitions from the given conditions
def weight_pair1 : ℕ := 3
def weight_pair2 : ℕ := 5
def weight_pair3 : ℕ := 8

-- Goal: Prove that the total weight of the dumbbell system is 32 lbs
theorem total_weight_of_dumbbell_system :
  2 * weight_pair1 + 2 * weight_pair2 + 2 * weight_pair3 = 32 :=
by sorry

end total_weight_of_dumbbell_system_l112_112667


namespace find_third_vertex_l112_112450

open Real

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (9, 3)
def vertex2 : ℝ × ℝ := (0, 0)

-- Define the conditions
def on_negative_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.1 < 0

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Statement of the problem in Lean
theorem find_third_vertex :
  ∃ (vertex3 : ℝ × ℝ), 
    on_negative_x_axis vertex3 ∧ 
    area_of_triangle vertex1 vertex2 vertex3 = 45 ∧
    vertex3 = (-30, 0) :=
sorry

end find_third_vertex_l112_112450


namespace trivia_game_answer_l112_112167

theorem trivia_game_answer (correct_first_half : Nat)
    (points_per_question : Nat) (final_score : Nat) : 
    correct_first_half = 8 → 
    points_per_question = 8 →
    final_score = 80 →
    (final_score - correct_first_half * points_per_question) / points_per_question = 2 :=
by
    intros h1 h2 h3
    sorry

end trivia_game_answer_l112_112167


namespace expected_worth_is_1_33_l112_112018

noncomputable def expected_worth_of_coin_flip : ℝ :=
  let prob_heads := 2 / 3
  let profit_heads := 5
  let prob_tails := 1 / 3
  let loss_tails := -6
  (prob_heads * profit_heads + prob_tails * loss_tails)

theorem expected_worth_is_1_33 : expected_worth_of_coin_flip = 1.33 := by
  sorry

end expected_worth_is_1_33_l112_112018


namespace cistern_width_l112_112098

theorem cistern_width (l d A : ℝ) (h_l: l = 5) (h_d: d = 1.25) (h_A: A = 42.5) :
  ∃ w : ℝ, 5 * w + 2 * (1.25 * 5) + 2 * (1.25 * w) = 42.5 ∧ w = 4 :=
by
  use 4
  sorry

end cistern_width_l112_112098


namespace sculptures_not_on_display_count_l112_112355

noncomputable def total_art_pieces : ℕ := 1800
noncomputable def pieces_on_display : ℕ := total_art_pieces / 3
noncomputable def pieces_not_on_display : ℕ := total_art_pieces - pieces_on_display
noncomputable def sculptures_on_display : ℕ := pieces_on_display / 6
noncomputable def sculptures_not_on_display : ℕ := pieces_not_on_display * 2 / 3

theorem sculptures_not_on_display_count : sculptures_not_on_display = 800 :=
by {
  -- Since this is a statement only as requested, we use sorry to skip the proof
  sorry
}

end sculptures_not_on_display_count_l112_112355


namespace total_students_in_class_l112_112704

theorem total_students_in_class
  (S : ℕ)
  (H1 : 5/8 * S = S - 60)
  (H2 : 60 = 3/8 * S) :
  S = 160 :=
by
  sorry

end total_students_in_class_l112_112704


namespace hiker_speed_correct_l112_112129

variable (hikerSpeed : ℝ)
variable (cyclistSpeed : ℝ := 15)
variable (cyclistTravelTime : ℝ := 5 / 60)  -- Converted 5 minutes to hours
variable (hikerCatchUpTime : ℝ := 13.75 / 60)  -- Converted 13.75 minutes to hours
variable (cyclistDistance : ℝ := cyclistSpeed * cyclistTravelTime)

theorem hiker_speed_correct :
  (hikerSpeed * hikerCatchUpTime = cyclistDistance) →
  hikerSpeed = 60 / 11 :=
by
  intro hiker_eq_cyclist_distance
  sorry

end hiker_speed_correct_l112_112129


namespace find_c_of_parabola_l112_112427

theorem find_c_of_parabola (a b c : ℚ) (h_vertex : (5 : ℚ) = a * (3 : ℚ)^2 + b * (3 : ℚ) + c)
    (h_point : (7 : ℚ) = a * (1 : ℚ)^2 + b * (1 : ℚ) + c) :
  c = 19 / 2 :=
by
  sorry

end find_c_of_parabola_l112_112427


namespace crazy_silly_school_books_movies_correct_l112_112093

noncomputable def crazy_silly_school_books_movies (B M : ℕ) : Prop :=
  M = 61 ∧ M = B + 2 ∧ M = 10 ∧ B = 8

theorem crazy_silly_school_books_movies_correct {B M : ℕ} :
  crazy_silly_school_books_movies B M → B = 8 :=
by
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end crazy_silly_school_books_movies_correct_l112_112093


namespace number_of_blue_balloons_l112_112486

def total_balloons : ℕ := 37
def red_balloons : ℕ := 14
def green_balloons : ℕ := 10

theorem number_of_blue_balloons : (total_balloons - red_balloons - green_balloons) = 13 := 
by
  -- Placeholder for the proof
  sorry

end number_of_blue_balloons_l112_112486


namespace relationship_a_b_c_l112_112145

noncomputable def a : ℝ := Real.sin (Real.pi / 16)
noncomputable def b : ℝ := 0.25
noncomputable def c : ℝ := 2 * Real.log 2 - Real.log 3

theorem relationship_a_b_c : a < b ∧ b < c :=
by
  sorry

end relationship_a_b_c_l112_112145


namespace measure_angle_ADC_l112_112080

variable (A B C D : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Definitions for the angles
variable (angle_ABC angle_BCD angle_ADC : ℝ)

-- Conditions for the problem
axiom Angle_ABC_is_4_times_Angle_BCD : angle_ABC = 4 * angle_BCD
axiom Angle_BCD_ADC_sum_to_180 : angle_BCD + angle_ADC = 180

-- The theorem that we want to prove
theorem measure_angle_ADC (Angle_ABC_is_4_times_Angle_BCD: angle_ABC = 4 * angle_BCD)
    (Angle_BCD_ADC_sum_to_180: angle_BCD + angle_ADC = 180) : 
    angle_ADC = 144 :=
by
  sorry

end measure_angle_ADC_l112_112080


namespace Arthur_total_distance_l112_112421

/-- Arthur walks 8 blocks south and then 10 blocks west. Each block is one-fourth of a mile.
How many miles did Arthur walk in total? -/
theorem Arthur_total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length_miles : ℝ) :
  blocks_south = 8 ∧ blocks_west = 10 ∧ block_length_miles = 1/4 →
  (blocks_south + blocks_west) * block_length_miles = 4.5 :=
by
  intro h
  have h1 : blocks_south = 8 := h.1
  have h2 : blocks_west = 10 := h.2.1
  have h3 : block_length_miles = 1 / 4 := h.2.2
  sorry

end Arthur_total_distance_l112_112421


namespace nine_skiers_four_overtakes_impossible_l112_112909

theorem nine_skiers_four_overtakes_impossible :
  ∀ (skiers : Fin 9 → ℝ),  -- skiers are represented by their speeds
  (∀ i j, i < j → skiers i ≤ skiers j) →  -- skiers start sequentially and maintain constant speeds
  ¬(∀ i, (∃ a b : Fin 9, (a ≠ i ∧ b ≠ i ∧ (skiers a < skiers i ∧ skiers i < skiers b ∨ skiers b < skiers i ∧ skiers i < skiers a)))) →
    false := 
by
  sorry

end nine_skiers_four_overtakes_impossible_l112_112909


namespace proposition_3_proposition_4_l112_112136

variable {Line Plane : Type} -- Introduce the types for lines and planes
variable (m n : Line) (α β : Plane) -- Introduce specific lines and planes

-- Define parallel and perpendicular relations
variables {parallel : Line → Plane → Prop} {perpendicular : Line → Plane → Prop}
variables {parallel_line : Line → Line → Prop} {perpendicular_line : Line → Line → Prop}
variables {parallel_plane : Plane → Plane → Prop} {perpendicular_plane : Plane → Plane → Prop}

-- Define subset: a line n is in a plane α
variable {subset : Line → Plane → Prop}

-- Hypotheses for propositions 3 and 4
axiom prop3_hyp1 : perpendicular m α
axiom prop3_hyp2 : parallel_line m n
axiom prop3_hyp3 : parallel_plane α β

axiom prop4_hyp1 : perpendicular_line m n
axiom prop4_hyp2 : perpendicular m α
axiom prop4_hyp3 : perpendicular n β

theorem proposition_3 (h1 : perpendicular m α) (h2 : parallel_line m n) (h3 : parallel_plane α β) : perpendicular n β := sorry

theorem proposition_4 (h1 : perpendicular_line m n) (h2 : perpendicular m α) (h3 : perpendicular n β) : perpendicular_plane α β := sorry

end proposition_3_proposition_4_l112_112136


namespace umar_age_is_ten_l112_112757

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end umar_age_is_ten_l112_112757


namespace returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l112_112728

noncomputable def driving_distances : List ℤ := [-5, 3, 6, -4, 7, -2]

def fare (distance : ℕ) : ℕ :=
  if distance ≤ 3 then 8 else 8 + 2 * (distance - 3)

theorem returns_to_start_point_after_fourth_passenger :
  List.sum (driving_distances.take 4) = 0 :=
by
  sorry

theorem distance_after_last_passenger :
  List.sum driving_distances = 5 :=
by
  sorry

theorem total_earnings :
  (fare 5 + fare 3 + fare 6 + fare 4 + fare 7 + fare 2) = 68 :=
by
  sorry

end returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l112_112728


namespace soda_cost_l112_112282

variable (b s f : ℝ)

noncomputable def keegan_equation : Prop :=
  3 * b + 2 * s + f = 975

noncomputable def alex_equation : Prop :=
  2 * b + 3 * s + f = 900

theorem soda_cost (h1 : keegan_equation b s f) (h2 : alex_equation b s f) : s = 18.75 :=
by
  sorry

end soda_cost_l112_112282


namespace truck_total_distance_l112_112485

noncomputable def truck_distance (b t : ℝ) : ℝ :=
  let acceleration := b / 3
  let time_seconds := 300 + t
  let distance_feet := (1 / 2) * (acceleration / t) * time_seconds^2
  distance_feet / 5280

theorem truck_total_distance (b t : ℝ) : 
  truck_distance b t = b * (90000 + 600 * t + t ^ 2) / (31680 * t) :=
by
  sorry

end truck_total_distance_l112_112485


namespace corrected_sum_l112_112708

theorem corrected_sum : 37541 + 43839 ≠ 80280 → 37541 + 43839 = 81380 :=
by
  sorry

end corrected_sum_l112_112708


namespace linear_eq_implies_m_eq_1_l112_112287

theorem linear_eq_implies_m_eq_1 (x y m : ℝ) (h : 3 * (x ^ |m|) + (m + 1) * y = 6) (hm_abs : |m| = 1) (hm_ne_zero : m + 1 ≠ 0) : m = 1 :=
  sorry

end linear_eq_implies_m_eq_1_l112_112287


namespace smallest_possible_norm_l112_112279

-- Defining the vector \begin{pmatrix} -2 \\ 4 \end{pmatrix}
def vec_a : ℝ × ℝ := (-2, 4)

-- Condition: the norm of \mathbf{v} + \begin{pmatrix} -2 \\ 4 \end{pmatrix} = 10
def satisfies_condition (v : ℝ × ℝ) : Prop :=
  (Real.sqrt ((v.1 + vec_a.1) ^ 2 + (v.2 + vec_a.2) ^ 2)) = 10

noncomputable def smallest_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_possible_norm (v : ℝ × ℝ) (h : satisfies_condition v) : smallest_norm v = 10 - 2 * Real.sqrt 5 := by
  sorry

end smallest_possible_norm_l112_112279


namespace determine_set_B_l112_112111
open Set

/-- Given problem conditions and goal in Lean 4 -/
theorem determine_set_B (U A B : Set ℕ) (hU : U = { x | x < 10 } )
  (hA_inter_compl_B : A ∩ (U \ B) = {1, 3, 5, 7, 9} ) :
  B = {2, 4, 6, 8} :=
by
  sorry

end determine_set_B_l112_112111


namespace false_statements_l112_112793

variable (a b c : ℝ)

theorem false_statements (a b c : ℝ) :
  ¬(a > b → a^2 > b^2) ∧ ¬((a^2 > b^2) → a > b) ∧ ¬(a > b → a * c^2 > b * c^2) ∧ ¬(a > b ↔ |a| > |b|) :=
by
  sorry

end false_statements_l112_112793


namespace isabel_earnings_l112_112998

theorem isabel_earnings :
  ∀ (bead_necklaces gem_necklaces cost_per_necklace : ℕ),
    bead_necklaces = 3 →
    gem_necklaces = 3 →
    cost_per_necklace = 6 →
    (bead_necklaces + gem_necklaces) * cost_per_necklace = 36 := by
sorry

end isabel_earnings_l112_112998


namespace milk_cartons_total_l112_112533

theorem milk_cartons_total (regular_milk soy_milk : ℝ) (h1 : regular_milk = 0.5) (h2 : soy_milk = 0.1) :
  regular_milk + soy_milk = 0.6 :=
by
  rw [h1, h2]
  norm_num

end milk_cartons_total_l112_112533


namespace equalize_costs_l112_112335

theorem equalize_costs (X Y Z : ℝ) (h1 : Y > X) (h2 : Z > Y) : 
  (Y + (Z - (X + Z - 2 * Y) / 3) = Z) → 
   (Y - (Y + Z - (X + Z - 2 * Y)) / 3 = (X + Z - 2 * Y) / 3) := sorry

end equalize_costs_l112_112335


namespace inequality_transfers_l112_112966

variables (a b c d : ℝ)

theorem inequality_transfers (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_transfers_l112_112966


namespace profit_calculation_l112_112315

variable (x y : ℝ)

-- Conditions
def fabric_constraints_1 : Prop := (0.5 * x + 0.9 * (50 - x) ≤ 38)
def fabric_constraints_2 : Prop := (x + 0.2 * (50 - x) ≤ 26)
def x_range : Prop := (17.5 ≤ x ∧ x ≤ 20)

-- Goal
def profit_expression : ℝ := 15 * x + 1500

theorem profit_calculation (h1 : fabric_constraints_1 x) (h2 : fabric_constraints_2 x) (h3 : x_range x) : y = profit_expression x :=
by
  sorry

end profit_calculation_l112_112315


namespace ratio_of_triangle_areas_l112_112630

theorem ratio_of_triangle_areas (a k : ℝ) (h_pos_a : 0 < a) (h_pos_k : 0 < k)
    (h_triangle_division : true) (h_square_area : ∃ s, s = a^2) (h_area_one_triangle : ∃ t, t = k * a^2) :
    ∃ r, r = (1 / (4 * k)) :=
by
  sorry

end ratio_of_triangle_areas_l112_112630


namespace remainder_division_l112_112468

theorem remainder_division (x : ℝ) :
  (x ^ 2021 + 1) % (x ^ 12 - x ^ 9 + x ^ 6 - x ^ 3 + 1) = -x ^ 4 + 1 :=
sorry

end remainder_division_l112_112468


namespace scientific_notation_GDP_l112_112932

theorem scientific_notation_GDP (h : 1 = 10^9) : 32.07 * 10^9 = 3.207 * 10^10 := by
  sorry

end scientific_notation_GDP_l112_112932


namespace Ronald_eggs_initially_l112_112574

def total_eggs_shared (friends eggs_per_friend : Nat) : Nat :=
  friends * eggs_per_friend

theorem Ronald_eggs_initially (eggs : Nat) (candies : Nat) (friends : Nat) (eggs_per_friend : Nat)
  (h1 : friends = 8) (h2 : eggs_per_friend = 2) (h_share : total_eggs_shared friends eggs_per_friend = 16) :
  eggs = 16 := by
  sorry

end Ronald_eggs_initially_l112_112574


namespace f_at_2018_l112_112481

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x
axiom f_at_4 : f 4 = 5

theorem f_at_2018 : f 2018 = 5 :=
by
  -- Proof goes here
  sorry

end f_at_2018_l112_112481


namespace condition_not_right_triangle_l112_112162

theorem condition_not_right_triangle 
  (AB BC AC : ℕ) (angleA angleB angleC : ℕ)
  (h_A : AB = 3 ∧ BC = 4 ∧ AC = 5)
  (h_B : AB / BC = 3 / 4 ∧ BC / AC = 4 / 5 ∧ AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB)
  (h_C : angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 ∧ angleA + angleB + angleC = 180)
  (h_D : angleA = 40 ∧ angleB = 50 ∧ angleA + angleB + angleC = 180) :
  angleA = 45 ∧ angleB = 60 ∧ angleC = 75 ∧ (¬ (angleA = 90 ∨ angleB = 90 ∨ angleC = 90)) :=
sorry

end condition_not_right_triangle_l112_112162


namespace evaluate_fraction_sqrt_l112_112751

theorem evaluate_fraction_sqrt :
  (Real.sqrt ((1 / 8) + (1 / 18)) = (Real.sqrt 26) / 12) :=
by
  sorry

end evaluate_fraction_sqrt_l112_112751


namespace proof_problem_l112_112521

def number := 432

theorem proof_problem (y : ℕ) (n : ℕ) (h1 : y = 36) (h2 : 6^5 * 2 / n = y) : n = number :=
by 
  -- proof steps would go here
  sorry

end proof_problem_l112_112521


namespace smallest_class_size_l112_112520

theorem smallest_class_size
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 2)
  (total_students : 5 * x + 2 > 40) :
  ∃ (n : ℕ), n = 5 * x + 2 ∧ n = 42 :=
by
  sorry

end smallest_class_size_l112_112520


namespace find_larger_integer_l112_112684

variable (x : ℤ) (smaller larger : ℤ)
variable (ratio_1_to_4 : smaller = 1 * x ∧ larger = 4 * x)
variable (condition : smaller + 12 = larger)

theorem find_larger_integer : larger = 16 :=
by
  sorry

end find_larger_integer_l112_112684


namespace flag_covering_proof_l112_112808

def grid_covering_flag_ways (m n num_flags cells_per_flag : ℕ) :=
  if m * n / cells_per_flag = num_flags then 2^num_flags else 0

theorem flag_covering_proof :
  grid_covering_flag_ways 9 18 18 9 = 262144 := by
  sorry

end flag_covering_proof_l112_112808


namespace intersection_points_l112_112666

noncomputable def line1 (x y : ℝ) : Prop := 3 * x - 2 * y = 12
noncomputable def line2 (x y : ℝ) : Prop := 2 * x + 4 * y = 8
noncomputable def line3 (x y : ℝ) : Prop := -5 * x + 15 * y = 30
noncomputable def line4 (x : ℝ) : Prop := x = -3

theorem intersection_points : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ 
  (∃ (x y : ℝ), line1 x y ∧ x = -3 ∧ y = -10.5) ∧ 
  ¬(∃ (x y : ℝ), line2 x y ∧ line3 x y) ∧
  ∃ (x y : ℝ), line4 x ∧ y = -10.5 :=
  sorry

end intersection_points_l112_112666


namespace binomial_sum_l112_112951

theorem binomial_sum :
  (Nat.choose 10 3) + (Nat.choose 10 4) = 330 :=
by
  sorry

end binomial_sum_l112_112951


namespace Allyson_age_is_28_l112_112399

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end Allyson_age_is_28_l112_112399


namespace no_real_roots_in_interval_l112_112264

variable {a b c : ℝ}

theorem no_real_roots_in_interval (ha : 0 < a) (h : 12 * a + 5 * b + 2 * c > 0) :
  ¬ ∃ α β, (2 < α ∧ α < 3) ∧ (2 < β ∧ β < 3) ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 := by
  sorry

end no_real_roots_in_interval_l112_112264


namespace cos_double_angle_l112_112553

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l112_112553


namespace value_range_of_function_l112_112388

theorem value_range_of_function :
  ∀ (x : ℝ), -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -1 ≤ Real.sin x * Real.sin x - 2 * Real.sin x ∧ Real.sin x * Real.sin x - 2 * Real.sin x ≤ 3 :=
by
  sorry

end value_range_of_function_l112_112388


namespace part1_part2_l112_112308

-- Part 1: Positive integers with leading digit 6 that become 1/25 of the original number when the leading digit is removed.
theorem part1 (n : ℕ) (m : ℕ) (h1 : m = 6 * 10^n + m) (h2 : m = (6 * 10^n + m) / 25) :
  m = 625 * 10^(n - 2) ∨
  m = 625 * 10^(n - 2 + 1) ∨
  ∃ k : ℕ, m = 625 * 10^(n - 2 + k) :=
sorry

-- Part 2: No positive integer exists which becomes 1/35 of the original number when its leading digit is removed.
theorem part2 (n : ℕ) (m : ℕ) (h : m = 6 * 10^n + m) :
  m ≠ (6 * 10^n + m) / 35 :=
sorry

end part1_part2_l112_112308


namespace x_gt_1_sufficient_but_not_necessary_x_gt_0_l112_112663

theorem x_gt_1_sufficient_but_not_necessary_x_gt_0 (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬(x > 0 → x > 1) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_x_gt_0_l112_112663


namespace f_96_value_l112_112571

noncomputable def f : ℕ → ℕ :=
sorry

axiom condition_1 (a b : ℕ) : 
  f (a * b) = f a + f b

axiom condition_2 (n : ℕ) (hp : Nat.Prime n) (hlt : 10 < n) : 
  f n = 0

axiom condition_3 : 
  f 1 < f 243 ∧ f 243 < f 2 ∧ f 2 < 11

axiom condition_4 : 
  f 2106 < 11

theorem f_96_value :
  f 96 = 31 :=
sorry

end f_96_value_l112_112571


namespace narrow_black_stripes_count_l112_112729

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l112_112729


namespace M_inter_N_is_01_l112_112460

variable (x : ℝ)

def M := { x : ℝ | Real.log (1 - x) < 0 }
def N := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem M_inter_N_is_01 : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  -- Proof will go here
  sorry

end M_inter_N_is_01_l112_112460


namespace complement_M_in_U_l112_112789

open Set

theorem complement_M_in_U : 
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  U \ M = {3, 7} := 
by
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  sorry

end complement_M_in_U_l112_112789


namespace range_of_b_l112_112184

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x + 2
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (f x b) b

theorem range_of_b (b : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f x b = y) → (∀ z : ℝ, ∃ x : ℝ, g x b = z) → b ≥ 4 ∨ b ≤ -2 :=
sorry

end range_of_b_l112_112184


namespace probability_greater_than_n_l112_112416

theorem probability_greater_than_n (n : ℕ) : 
  (1 ≤ n ∧ n ≤ 5) → (∃ k, k = 6 - n - 1 ∧ k / 6 = 1 / 2) → n = 3 := 
by sorry

end probability_greater_than_n_l112_112416


namespace find_m_for_increasing_graph_l112_112310

theorem find_m_for_increasing_graph (m : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → (m + 1) * x ^ (3 - m^2) < (m + 1) * y ^ (3 - m^2) → x < y) ↔ m = -2 :=
by
  sorry

end find_m_for_increasing_graph_l112_112310


namespace quadratic_function_range_l112_112454

theorem quadratic_function_range (x : ℝ) (h : x ≥ 0) : 
  3 ≤ x^2 + 2 * x + 3 :=
by {
  sorry
}

end quadratic_function_range_l112_112454


namespace angelina_speed_from_library_to_gym_l112_112069

theorem angelina_speed_from_library_to_gym :
  ∃ (v : ℝ), 
    (840 / v - 510 / (1.5 * v) = 40) ∧
    (510 / (1.5 * v) - 480 / (2 * v) = 20) ∧
    (2 * v = 25) :=
by
  sorry

end angelina_speed_from_library_to_gym_l112_112069


namespace gcd_2024_2048_l112_112035

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end gcd_2024_2048_l112_112035


namespace average_number_of_glasses_per_box_l112_112268

-- Definitions and conditions
variables (S L : ℕ) -- S is the number of smaller boxes, L is the number of larger boxes

-- Condition 1: One box contains 12 glasses, and the other contains 16 glasses.
-- (This is implicitly understood in the equation for total glasses)

-- Condition 3: There are 16 more larger boxes than smaller smaller boxes
def condition_3 := L = S + 16

-- Condition 4: The total number of glasses is 480.
def condition_4 := 12 * S + 16 * L = 480

-- Proving the average number of glasses per box is 15
theorem average_number_of_glasses_per_box (h1 : condition_3 S L) (h2 : condition_4 S L) :
  (480 : ℝ) / (S + L) = 15 :=
by 
  -- Assuming S and L are natural numbers 
  sorry

end average_number_of_glasses_per_box_l112_112268


namespace maximal_k_value_l112_112397

noncomputable def max_edges (n : ℕ) : ℕ :=
  2 * n - 4
   
theorem maximal_k_value (k n : ℕ) (h1 : n = 2016) (h2 : k ≤ max_edges n) :
  k = 4028 :=
by sorry

end maximal_k_value_l112_112397


namespace sufficient_budget_for_kvass_l112_112528

variables (x y : ℝ)

theorem sufficient_budget_for_kvass (h1 : x + y = 1) (h2 : 0.6 * x + 1.2 * y = 1) : 
  3 * y ≥ 1.44 * y :=
by
  sorry

end sufficient_budget_for_kvass_l112_112528


namespace sum_terms_a1_a17_l112_112127

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 :=
sorry

end sum_terms_a1_a17_l112_112127


namespace sqrt_sum_inequality_l112_112558

theorem sqrt_sum_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h_sum : a + b + c = 3) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a) :=
by
  sorry

end sqrt_sum_inequality_l112_112558


namespace marcella_shoes_lost_l112_112412

theorem marcella_shoes_lost (pairs_initial : ℕ) (pairs_left_max : ℕ) (individuals_initial : ℕ) (individuals_left_max : ℕ) (pairs_initial_eq : pairs_initial = 25) (pairs_left_max_eq : pairs_left_max = 20) (individuals_initial_eq : individuals_initial = pairs_initial * 2) (individuals_left_max_eq : individuals_left_max = pairs_left_max * 2) : (individuals_initial - individuals_left_max) = 10 := 
by
  sorry

end marcella_shoes_lost_l112_112412


namespace car_rental_cost_l112_112137

variable (x : ℝ)

theorem car_rental_cost (h : 65 + 0.40 * 325 = x * 325) : x = 0.60 :=
by 
  sorry

end car_rental_cost_l112_112137


namespace triangle_reciprocal_sum_l112_112664

variables {A B C D L M N : Type} -- Points are types
variables {t_1 t_2 t_3 t_4 t_5 t_6 : ℝ} -- Areas are real numbers

-- Assume conditions as hypotheses
variable (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
variable (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
variable (h3 : ∀ (t1 t5 t3 t4 : ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5))

theorem triangle_reciprocal_sum 
  (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
  (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
  (h3 : ∀ (t1 t5 t3 t4: ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5)) :
  (1 / t_1 + 1 / t_3 + 1 / t_5) = (1 / t_2 + 1 / t_4 + 1 / t_6) :=
sorry

end triangle_reciprocal_sum_l112_112664


namespace value_of_x_l112_112607

theorem value_of_x (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 :=
by
  sorry

end value_of_x_l112_112607


namespace total_coffee_cost_l112_112030

def vacation_days : ℕ := 40
def daily_coffee : ℕ := 3
def pods_per_box : ℕ := 30
def box_cost : ℕ := 8

theorem total_coffee_cost : vacation_days * daily_coffee / pods_per_box * box_cost = 32 := by
  -- proof goes here
  sorry

end total_coffee_cost_l112_112030


namespace continuous_sum_m_l112_112101

noncomputable def g : ℝ → ℝ → ℝ
| x, m => if x < m then x^2 + 4 else 3 * x + 6

theorem continuous_sum_m :
  ∀ m1 m2 : ℝ, (∀ m : ℝ, (g m m1 = g m m2) → g m (m1 + m2) = g m m1 + g m m2) →
  m1 + m2 = 3 :=
sorry

end continuous_sum_m_l112_112101


namespace express_x_in_terms_of_y_l112_112425

variable {x y : ℝ}

theorem express_x_in_terms_of_y (h : 3 * x - 4 * y = 6) : x = (6 + 4 * y) / 3 := 
sorry

end express_x_in_terms_of_y_l112_112425


namespace sum_sublist_eq_100_l112_112890

theorem sum_sublist_eq_100 {l : List ℕ}
  (h_len : l.length = 2 * 31100)
  (h_max : ∀ x ∈ l, x ≤ 100)
  (h_sum : l.sum = 200) :
  ∃ (s : List ℕ), s ⊆ l ∧ s.sum = 100 := 
sorry

end sum_sublist_eq_100_l112_112890


namespace units_digit_product_l112_112097

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_product (a b c : ℕ) :
  units_digit a = 7 → units_digit b = 3 → units_digit c = 9 →
  units_digit ((a * b) * c) = 9 :=
by
  intros h1 h2 h3
  sorry

end units_digit_product_l112_112097


namespace percent_of_N_in_M_l112_112221

theorem percent_of_N_in_M (N M : ℝ) (hM : M ≠ 0) : (N / M) * 100 = 100 * N / M :=
by
  sorry

end percent_of_N_in_M_l112_112221


namespace polynomial_roots_l112_112970

noncomputable def f (x : ℝ) : ℝ := 8 * x^4 + 28 * x^3 - 74 * x^2 - 8 * x + 48

theorem polynomial_roots:
  ∃ (a b c d : ℝ), a = -3 ∧ b = -1 ∧ c = -1 ∧ d = 2 ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) :=
sorry

end polynomial_roots_l112_112970


namespace smallest_y_for_perfect_fourth_power_l112_112484

-- Define the conditions
def x : ℕ := 7 * 24 * 48
def y : ℕ := 6174

-- The theorem we need to prove
theorem smallest_y_for_perfect_fourth_power (x y : ℕ) 
  (hx : x = 7 * 24 * 48) 
  (hy : y = 6174) : ∃ k : ℕ, (∃ z : ℕ, z * z * z * z = x * y) :=
sorry

end smallest_y_for_perfect_fourth_power_l112_112484


namespace arithmetic_sequence_sum_ratio_l112_112013

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a S : ℕ → ℚ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def condition_1 (a : ℕ → ℚ) : Prop :=
  is_arithmetic_sequence a

def condition_2 (a : ℕ → ℚ) : Prop :=
  (a 5) / (a 3) = 5 / 9

-- Proof statement
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : condition_1 a) (h2 : condition_2 a) (h3 : sum_of_first_n_terms a S) : 
  (S 9) / (S 5) = 1 := 
sorry

end arithmetic_sequence_sum_ratio_l112_112013


namespace min_value_expression_l112_112051

theorem min_value_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_expression_l112_112051


namespace total_brownies_correct_l112_112250

def brownies_initial : Nat := 24
def father_ate : Nat := brownies_initial / 3
def remaining_after_father : Nat := brownies_initial - father_ate
def mooney_ate : Nat := remaining_after_father / 4
def remaining_after_mooney : Nat := remaining_after_father - mooney_ate
def benny_ate : Nat := (remaining_after_mooney * 2) / 5
def remaining_after_benny : Nat := remaining_after_mooney - benny_ate
def snoopy_ate : Nat := 3
def remaining_after_snoopy : Nat := remaining_after_benny - snoopy_ate
def new_batch : Nat := 24
def total_brownies : Nat := remaining_after_snoopy + new_batch

theorem total_brownies_correct : total_brownies = 29 :=
by
  sorry

end total_brownies_correct_l112_112250


namespace find_Q_over_P_l112_112677

theorem find_Q_over_P (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -7 → x ≠ 0 → x ≠ 5 →
    (P / (x + 7 : ℝ) + Q / (x^2 - 6 * x) = (x^2 - 6 * x + 14) / (x^3 + x^2 - 30 * x))) :
  Q / P = 12 :=
  sorry

end find_Q_over_P_l112_112677


namespace sin_330_eq_neg_half_l112_112375

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_half_l112_112375


namespace parity_of_exponentiated_sum_l112_112854

theorem parity_of_exponentiated_sum
  : (1 ^ 1994 + 9 ^ 1994 + 8 ^ 1994 + 6 ^ 1994) % 2 = 0 := 
by
  sorry

end parity_of_exponentiated_sum_l112_112854


namespace mass_percentage_O_is_26_2_l112_112178

noncomputable def mass_percentage_O_in_Benzoic_acid : ℝ :=
  let molar_mass_C := 12.01
  let molar_mass_H := 1.01
  let molar_mass_O := 16.00
  let molar_mass_Benzoic_acid := (7 * molar_mass_C) + (6 * molar_mass_H) + (2 * molar_mass_O)
  let mass_O_in_Benzoic_acid := 2 * molar_mass_O
  (mass_O_in_Benzoic_acid / molar_mass_Benzoic_acid) * 100

theorem mass_percentage_O_is_26_2 :
  mass_percentage_O_in_Benzoic_acid = 26.2 := by
  sorry

end mass_percentage_O_is_26_2_l112_112178


namespace lambda_inequality_l112_112072

-- Define the problem hypothesis and conclusion
theorem lambda_inequality (n : ℕ) (hn : n ≥ 4) (lambda_n : ℝ) :
  lambda_n ≥ 2 * Real.sin ((n-2) * Real.pi / (2 * n)) :=
by
  -- Placeholder for the proof
  sorry

end lambda_inequality_l112_112072


namespace factorization_implies_k_l112_112632

theorem factorization_implies_k (x y k : ℝ) (h : ∃ (a b c d e f : ℝ), 
                            x^3 + 3 * x^2 - 2 * x * y - k * x - 4 * y = (a * x + b * y + c) * (d * x^2 + e * xy + f)) :
  k = -2 :=
sorry

end factorization_implies_k_l112_112632


namespace roger_allowance_spend_l112_112589

variable (A m s : ℝ)

-- Conditions from the problem
def condition1 : Prop := m = 0.25 * (A - 2 * s)
def condition2 : Prop := s = 0.10 * (A - 0.5 * m)
def goal : Prop := m + s = 0.59 * A

theorem roger_allowance_spend (h1 : condition1 A m s) (h2 : condition2 A m s) : goal A m s :=
  sorry

end roger_allowance_spend_l112_112589


namespace minimum_handshakes_l112_112859

noncomputable def min_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

theorem minimum_handshakes (n k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  min_handshakes n k = 45 :=
by
  -- We provide the conditions directly
  -- n = 30, k = 3
  rw [h1, h2]
  -- then show that min_handshakes 30 3 = 45
  show min_handshakes 30 3 = 45
  sorry 

end minimum_handshakes_l112_112859


namespace gambler_largest_amount_received_l112_112014

def largest_amount_received_back (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : ℕ :=
  3000 - (30 * a + 100 * b)

theorem gambler_largest_amount_received (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : 
    largest_amount_received_back x y a b h1 h2 h3 = 2030 :=
by sorry

end gambler_largest_amount_received_l112_112014


namespace upper_limit_of_people_l112_112428

theorem upper_limit_of_people (P : ℕ) (h1 : 36 = (3 / 8) * P) (h2 : P > 50) (h3 : (5 / 12) * P = 40) : P = 96 :=
by
  sorry

end upper_limit_of_people_l112_112428


namespace vector_expression_l112_112722

variables (a b c : ℝ × ℝ)
variables (m n : ℝ)

noncomputable def vec_a : ℝ × ℝ := (1, 1)
noncomputable def vec_b : ℝ × ℝ := (1, -1)
noncomputable def vec_c : ℝ × ℝ := (-1, 2)

/-- Prove that vector c can be expressed in terms of vectors a and b --/
theorem vector_expression : 
  vec_c = m • vec_a + n • vec_b → (m = 1/2 ∧ n = -3/2) :=
sorry

end vector_expression_l112_112722


namespace rectangular_solid_dimension_change_l112_112252

theorem rectangular_solid_dimension_change (a b : ℝ) (h : 2 * a^2 + 4 * a * b = 0.6 * (6 * a^2)) : b = 0.4 * a :=
by sorry

end rectangular_solid_dimension_change_l112_112252


namespace find_other_root_l112_112634

theorem find_other_root (k r : ℝ) (h1 : ∀ x : ℝ, 3 * x^2 + k * x + 6 = 0) (h2 : ∃ x : ℝ, 3 * x^2 + k * x + 6 = 0 ∧ x = 3) :
  r = 2 / 3 :=
sorry

end find_other_root_l112_112634


namespace sum_zero_l112_112055

noncomputable def f : ℝ → ℝ := sorry

theorem sum_zero :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (x + 5) = f x) →
  f (1 / 3) = 1 →
  f (16 / 3) + f (29 / 3) + f 12 + f (-7) = 0 :=
by
  intros hodd hperiod hvalue
  sorry

end sum_zero_l112_112055


namespace find_k_of_inverse_proportion_l112_112699

theorem find_k_of_inverse_proportion (k x y : ℝ) (h : y = k / x) (hx : x = 2) (hy : y = 6) : k = 12 :=
by
  sorry

end find_k_of_inverse_proportion_l112_112699


namespace number_of_solutions_eq_4_l112_112638

noncomputable def num_solutions := 
  ∃ n : ℕ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → (3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * Real.cos x = 0) → n = 4)

-- To state the above more clearly, we can add an abbreviation function for the equation.
noncomputable def equation (x : ℝ) : ℝ := 3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * Real.cos x

theorem number_of_solutions_eq_4 :
  (∃ n, n = 4 ∧ ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → equation x = 0 → true) := sorry

end number_of_solutions_eq_4_l112_112638


namespace white_balls_in_bag_l112_112693

   theorem white_balls_in_bag (m : ℕ) (h : m ≤ 7) :
     (2 * (m * (m - 1) / 2) / (7 * 6 / 2)) + ((m * (7 - m)) / (7 * 6 / 2)) = 6 / 7 → m = 3 :=
   by
     intros h_eq
     sorry
   
end white_balls_in_bag_l112_112693


namespace arithmetic_geometric_sum_l112_112760

theorem arithmetic_geometric_sum (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h_arith : 2 * b = a + c) (h_geom : a^2 = b * c) 
  (h_sum : a + 3 * b + c = 10) : a = -4 :=
by
  sorry

end arithmetic_geometric_sum_l112_112760


namespace solve_inequality_l112_112665

theorem solve_inequality (a x : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) :
  ((0 ≤ a ∧ a < 1/2 → a < x ∧ x < 1 - a) ∧ 
   (a = 1/2 → false) ∧ 
   (1/2 < a ∧ a ≤ 1 → 1 - a < x ∧ x < a)) ↔ (x - a) * (x + a - 1) < 0 := 
by
  sorry

end solve_inequality_l112_112665


namespace domain_cannot_be_0_to_3_l112_112011

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Define the range of the function f
def range_f : Set ℝ := Set.Icc 1 2

-- Statement that the domain [0, 3] cannot be the domain of f given the range
theorem domain_cannot_be_0_to_3 :
  ∀ (f : ℝ → ℝ) (range_f : Set ℝ),
    (∀ x, 1 ≤ f x ∧ f x ≤ 2) →
    ¬ ∃ dom : Set ℝ, dom = Set.Icc 0 3 ∧ 
      (∀ x ∈ dom, f x ∈ range_f) :=
by
  sorry

end domain_cannot_be_0_to_3_l112_112011


namespace jason_manager_years_l112_112293

-- Definitions based on the conditions
def jason_bartender_years : ℕ := 9
def jason_total_months : ℕ := 150
def additional_months_excluded : ℕ := 6

-- Conversion from months to years
def total_years := jason_total_months / 12
def excluded_years := additional_months_excluded / 12

-- Lean statement for the proof problem
theorem jason_manager_years :
  total_years - jason_bartender_years - excluded_years = 3 := by
  sorry

end jason_manager_years_l112_112293


namespace line_equation_parallel_to_x_axis_through_point_l112_112653

-- Define the point (3, -2)
def point : ℝ × ℝ := (3, -2)

-- Define a predicate for a line being parallel to the X-axis
def is_parallel_to_x_axis (line : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, line x k

-- Define the equation of the line passing through the given point
def equation_of_line_through_point (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- State the theorem to be proved
theorem line_equation_parallel_to_x_axis_through_point :
  ∀ (line : ℝ → ℝ → Prop), 
    (equation_of_line_through_point point line) → (is_parallel_to_x_axis line) → (∀ x, line x (-2)) :=
by
  sorry

end line_equation_parallel_to_x_axis_through_point_l112_112653


namespace larger_number_is_34_l112_112695

theorem larger_number_is_34 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : y = 34 :=
by
  sorry

end larger_number_is_34_l112_112695


namespace original_children_count_l112_112616

theorem original_children_count (x : ℕ) (h1 : 46800 / x + 1950 = 46800 / (x - 2))
    : x = 8 :=
sorry

end original_children_count_l112_112616


namespace randy_money_left_l112_112689

theorem randy_money_left (initial_money lunch ice_cream_cone remaining : ℝ) 
  (h1 : initial_money = 30)
  (h2 : lunch = 10)
  (h3 : remaining = initial_money - lunch)
  (h4 : ice_cream_cone = remaining * (1/4)) :
  (remaining - ice_cream_cone) = 15 := by
  sorry

end randy_money_left_l112_112689


namespace ratio_of_perimeters_l112_112946

theorem ratio_of_perimeters (s S : ℝ) 
  (h1 : S = 3 * s) : 
  (4 * S) / (4 * s) = 3 :=
by
  sorry

end ratio_of_perimeters_l112_112946


namespace eccentricity_of_given_ellipse_l112_112857

noncomputable def eccentricity_of_ellipse : ℝ :=
  let a : ℝ := 1
  let b : ℝ := 1 / 2
  let c : ℝ := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_given_ellipse :
  eccentricity_of_ellipse = Real.sqrt (3) / 2 :=
by
  -- Proof is omitted.
  sorry

end eccentricity_of_given_ellipse_l112_112857


namespace abs_inequality_l112_112996

theorem abs_inequality (x y : ℝ) (h1 : |x| < 2) (h2 : |y| < 2) : |4 - x * y| > 2 * |x - y| :=
by
  sorry

end abs_inequality_l112_112996


namespace student_walking_time_l112_112128

-- Define the conditions
def total_time_walking_and_bus : ℕ := 90  -- Total time walking to school and taking the bus back home
def total_time_bus_both_ways : ℕ := 30 -- Total time taking the bus both ways

-- Calculate the time taken for walking both ways
def time_bus_one_way : ℕ := total_time_bus_both_ways / 2
def time_walking_one_way : ℕ := total_time_walking_and_bus - time_bus_one_way
def total_time_walking_both_ways : ℕ := 2 * time_walking_one_way

-- State the theorem to be proved
theorem student_walking_time :
  total_time_walking_both_ways = 150 := by
  sorry

end student_walking_time_l112_112128


namespace multiples_of_231_l112_112766

theorem multiples_of_231 (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ 99 → i % 2 = 1 → 231 ∣ 10^j - 10^i) :
  ∃ n, n = 416 :=
by sorry

end multiples_of_231_l112_112766


namespace sum_of_first_50_digits_is_216_l112_112301

noncomputable def sum_first_50_digits_of_fraction : Nat :=
  let repeating_block := [0, 0, 0, 9, 9, 9]
  let full_cycles := 8
  let remaining_digits := [0, 0]
  let sum_full_cycles := full_cycles * (repeating_block.sum)
  let sum_remaining_digits := remaining_digits.sum
  sum_full_cycles + sum_remaining_digits

theorem sum_of_first_50_digits_is_216 :
  sum_first_50_digits_of_fraction = 216 := by
  sorry

end sum_of_first_50_digits_is_216_l112_112301


namespace no_2018_zero_on_curve_l112_112980

theorem no_2018_zero_on_curve (a c d : ℝ) (hac : a * c > 0) : ¬∃(d : ℝ), (2018 : ℝ) ^ 2 * a + 2018 * c + d = 0 := 
by {
  sorry
}

end no_2018_zero_on_curve_l112_112980


namespace quadratic_real_roots_l112_112549

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end quadratic_real_roots_l112_112549


namespace num_pos_cubes_ending_in_5_lt_5000_l112_112453

theorem num_pos_cubes_ending_in_5_lt_5000 : 
  (∃ (n1 n2 : ℕ), (n1 ≤ 5000 ∧ n2 ≤ 5000) ∧ (n1^3 % 10 = 5 ∧ n2^3 % 10 = 5) ∧ (n1^3 < 5000 ∧ n2^3 < 5000) ∧ n1 ≠ n2 ∧ 
  ∀ n, (n^3 < 5000 ∧ n^3 % 10 = 5) → (n = n1 ∨ n = n2)) :=
sorry

end num_pos_cubes_ending_in_5_lt_5000_l112_112453


namespace find_missing_number_l112_112291

theorem find_missing_number 
  (x : ℕ) 
  (avg : (744 + 745 + 747 + 748 + 749 + some_num + 753 + 755 + x) / 9 = 750)
  (hx : x = 755) : 
  some_num = 804 := 
  sorry

end find_missing_number_l112_112291


namespace quadratic_roots_l112_112185

theorem quadratic_roots (a b k : ℝ) (h₁ : a + b = -2) (h₂ : a * b = k / 3)
    (h₃ : |a - b| = 1/2 * (a^2 + b^2)) : k = 0 ∨ k = 6 :=
sorry

end quadratic_roots_l112_112185


namespace unique_solution_exists_q_l112_112382

theorem unique_solution_exists_q :
  (∃ q : ℝ, q ≠ 0 ∧ (∀ x y : ℝ, (2 * q * x^2 - 20 * x + 5 = 0) ∧ (2 * q * y^2 - 20 * y + 5 = 0) → x = y)) ↔ q = 10 := 
sorry

end unique_solution_exists_q_l112_112382


namespace max_n_value_is_9_l112_112358

variable (a b c d n : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : c > d)
variable (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d)))

theorem max_n_value_is_9 (h1 : a > b) (h2 : b > c) (h3 : c > d)
    (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end max_n_value_is_9_l112_112358


namespace symmetric_line_equation_l112_112284

theorem symmetric_line_equation (x y : ℝ) :
  (∀ x y : ℝ, x - 3 * y + 5 = 0 ↔ 3 * x - y - 5 = 0) :=
by 
  sorry

end symmetric_line_equation_l112_112284


namespace problem_proof_l112_112781

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

noncomputable def probability_ratio_pq : ℕ :=
let p := binomial 10 2 * binomial 30 2 * binomial 28 2
let q := binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3
p / (q / (binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3))

theorem problem_proof :
  probability_ratio_pq = 7371 :=
sorry

end problem_proof_l112_112781


namespace probability_two_of_three_survive_l112_112050

-- Let's define the necessary components
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of exactly 2 out of 3 seedlings surviving
theorem probability_two_of_three_survive (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  binomial_coefficient 3 2 * p^2 * (1 - p) = 3 * p^2 * (1 - p) :=
by
  sorry

end probability_two_of_three_survive_l112_112050


namespace bill_difference_is_zero_l112_112200

theorem bill_difference_is_zero
    (a b : ℝ)
    (h1 : 0.25 * a = 5)
    (h2 : 0.15 * b = 3) :
    a - b = 0 := 
by 
  sorry

end bill_difference_is_zero_l112_112200


namespace ned_trays_per_trip_l112_112168

def trays_from_table1 : ℕ := 27
def trays_from_table2 : ℕ := 5
def total_trips : ℕ := 4
def total_trays : ℕ := trays_from_table1 + trays_from_table2
def trays_per_trip : ℕ := total_trays / total_trips

theorem ned_trays_per_trip :
  trays_per_trip = 8 :=
by
  -- proof is skipped
  sorry

end ned_trays_per_trip_l112_112168


namespace range_of_a_l112_112866

theorem range_of_a (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 < 2)) ∧ (a - b + 1 = 1) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l112_112866


namespace forgotten_angle_measure_l112_112094

theorem forgotten_angle_measure 
  (total_sum : ℕ) 
  (measured_sum : ℕ) 
  (sides : ℕ) 
  (n_minus_2 : ℕ)
  (polygon_has_18_sides : sides = 18)
  (interior_angle_sum : total_sum = n_minus_2 * 180)
  (n_minus : n_minus_2 = (sides - 2))
  (measured : measured_sum = 2754) :
  ∃ forgotten_angle, forgotten_angle = total_sum - measured_sum ∧ forgotten_angle = 126 :=
by
  sorry

end forgotten_angle_measure_l112_112094


namespace ezekiel_first_day_distance_l112_112436

noncomputable def distance_first_day (total_distance second_day_distance third_day_distance : ℕ) :=
  total_distance - (second_day_distance + third_day_distance)

theorem ezekiel_first_day_distance:
  ∀ (total_distance second_day_distance third_day_distance : ℕ),
  total_distance = 50 →
  second_day_distance = 25 →
  third_day_distance = 15 →
  distance_first_day total_distance second_day_distance third_day_distance = 10 :=
by
  intros total_distance second_day_distance third_day_distance h1 h2 h3
  sorry

end ezekiel_first_day_distance_l112_112436


namespace inequality_nonnegative_reals_l112_112344

theorem inequality_nonnegative_reals (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 :=
sorry

end inequality_nonnegative_reals_l112_112344


namespace nth_term_l112_112439

theorem nth_term (b : ℕ → ℝ) (h₀ : b 1 = 1)
  (h_rec : ∀ n ≥ 1, (b (n + 1))^2 = 36 * (b n)^2) : 
  b 50 = 6^49 :=
by
  sorry

end nth_term_l112_112439


namespace abs_eq_implies_y_eq_half_l112_112752

theorem abs_eq_implies_y_eq_half (y : ℝ) (h : |y - 3| = |y + 2|) : y = 1 / 2 :=
by 
  sorry

end abs_eq_implies_y_eq_half_l112_112752


namespace variance_of_scores_l112_112222

def scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

def mean (xs : List ℕ) : ℚ := xs.sum / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

theorem variance_of_scores : variance scores = 4 := by
  sorry

end variance_of_scores_l112_112222


namespace zero_unique_multiple_prime_l112_112796

-- Condition: let n be a number
def n : Int := sorry

-- Condition: let p be any prime number
def is_prime (p : Int) : Prop := sorry  -- Predicate definition for prime number

-- Proof problem statement
theorem zero_unique_multiple_prime (n : Int) :
  (∀ p : Int, is_prime p → (∃ k : Int, n * p = k * p)) ↔ (n = 0) := by
  sorry

end zero_unique_multiple_prime_l112_112796


namespace total_number_of_coins_l112_112139

-- Definitions and conditions
def num_coins_25c := 17
def num_coins_10c := 17

-- Statement to prove
theorem total_number_of_coins : num_coins_25c + num_coins_10c = 34 := by
  sorry

end total_number_of_coins_l112_112139


namespace miae_closer_than_hyori_l112_112868

def bowl_volume : ℝ := 1000
def miae_estimate : ℝ := 1100
def hyori_estimate : ℝ := 850

def miae_difference : ℝ := abs (miae_estimate - bowl_volume)
def hyori_difference : ℝ := abs (bowl_volume - hyori_estimate)

theorem miae_closer_than_hyori : miae_difference < hyori_difference :=
by
  sorry

end miae_closer_than_hyori_l112_112868


namespace find_d_l112_112288

theorem find_d
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd4 : 4 = a * Real.sin 0 + d)
  (hdm2 : -2 = a * Real.sin (π) + d) :
  d = 1 := by
  sorry

end find_d_l112_112288


namespace root_expression_l112_112090

theorem root_expression {p q x1 x2 : ℝ}
  (h1 : x1^2 + p * x1 + q = 0)
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1 / x2 + x2 / x1) = (p^2 - 2 * q) / q :=
by {
  sorry
}

end root_expression_l112_112090


namespace number_of_2_face_painted_cubes_l112_112190

-- Condition definitions based on the problem statement
def painted_faces (n : ℕ) (type : String) : ℕ :=
  if type = "corner" then 8
  else if type = "edge" then 12
  else if type = "face" then 24
  else if type = "inner" then 9
  else 0

-- The mathematical proof statement
theorem number_of_2_face_painted_cubes : painted_faces 27 "edge" = 12 :=
by
  sorry

end number_of_2_face_painted_cubes_l112_112190


namespace largest_val_is_E_l112_112337

noncomputable def A : ℚ := 4 / (2 - 1/4)
noncomputable def B : ℚ := 4 / (2 + 1/4)
noncomputable def C : ℚ := 4 / (2 - 1/3)
noncomputable def D : ℚ := 4 / (2 + 1/3)
noncomputable def E : ℚ := 4 / (2 - 1/2)

theorem largest_val_is_E : E > A ∧ E > B ∧ E > C ∧ E > D := 
by sorry

end largest_val_is_E_l112_112337


namespace cost_of_each_shirt_is_8_l112_112173

-- Define the conditions
variables (S : ℝ)
def shirts_cost := 4 * S
def pants_cost := 2 * 18
def jackets_cost := 2 * 60
def total_cost := shirts_cost S + pants_cost + jackets_cost
def carrie_pays := 94

-- The goal is to prove that S equals 8 given the conditions above
theorem cost_of_each_shirt_is_8
  (h1 : carrie_pays = total_cost S / 2) : S = 8 :=
sorry

end cost_of_each_shirt_is_8_l112_112173


namespace area_of_triangle_l112_112756

-- Define the lines as functions
def line1 : ℝ → ℝ := fun x => 3 * x - 4
def line2 : ℝ → ℝ := fun x => -2 * x + 16

-- Define the vertices of the triangle formed by lines and y-axis
def vertex1 : ℝ × ℝ := (0, -4)
def vertex2 : ℝ × ℝ := (0, 16)
def vertex3 : ℝ × ℝ := (4, 8)

-- Define the proof statement
theorem area_of_triangle : 
  let A := vertex1 
  let B := vertex2 
  let C := vertex3 
  -- Compute the area of the triangle using the determinant formula
  let area := (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 40 := 
by
  sorry

end area_of_triangle_l112_112756


namespace FindAngleB_FindIncircleRadius_l112_112411

-- Define the problem setting
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Condition 1: a + c = 2b * sin (C + π / 6)
def Condition1 (T : Triangle) : Prop :=
  T.a + T.c = 2 * T.b * Real.sin (T.C + Real.pi / 6)

-- Condition 2: (b + c) (sin B - sin C) = (a - c) sin A
def Condition2 (T : Triangle) : Prop :=
  (T.b + T.c) * (Real.sin T.B - Real.sin T.C) = (T.a - T.c) * Real.sin T.A

-- Condition 3: (2a - c) cos B = b cos C
def Condition3 (T : Triangle) : Prop :=
  (2 * T.a - T.c) * Real.cos T.B = T.b * Real.cos T.C

-- Given: radius of incircle and dot product of vectors condition
def Given (T : Triangle) (r : ℝ) : Prop :=
  (T.a + T.c = 4 * Real.sqrt 3) ∧
  (2 * T.b * (T.a * T.c * Real.cos T.B - 3 * Real.sqrt 3 / 2) = 6)

-- Proof of B = π / 3
theorem FindAngleB (T : Triangle) :
  (Condition1 T ∨ Condition2 T ∨ Condition3 T) → T.B = Real.pi / 3 := 
sorry

-- Proof for the radius of the incircle
theorem FindIncircleRadius (T : Triangle) (r : ℝ) :
  Given T r → T.B = Real.pi / 3 → r = 1 := 
sorry


end FindAngleB_FindIncircleRadius_l112_112411


namespace complex_magnitude_condition_l112_112455

noncomputable def magnitude_of_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem complex_magnitude_condition (z : ℂ) (i : ℂ) (h : i * i = -1) (h1 : z - 2 * i = 1 + z * i) :
  magnitude_of_z z = Real.sqrt (10) / 2 :=
by
  -- proof goes here
  sorry

end complex_magnitude_condition_l112_112455


namespace mixed_groups_count_l112_112371

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l112_112371


namespace residue_of_minus_963_plus_100_mod_35_l112_112770

-- Defining the problem in Lean 4
theorem residue_of_minus_963_plus_100_mod_35 : 
  ((-963 + 100) % 35) = 12 :=
by
  sorry

end residue_of_minus_963_plus_100_mod_35_l112_112770


namespace even_function_value_l112_112835

theorem even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_def : ∀ x : ℝ, 0 < x → f x = 2^x + 1) :
  f (-2) = 5 :=
  sorry

end even_function_value_l112_112835


namespace range_of_m_l112_112229

def quadratic_nonnegative (m : ℝ) : Prop :=
∀ x : ℝ, m * x^2 + m * x + 1 ≥ 0

theorem range_of_m (m : ℝ) :
  quadratic_nonnegative m ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l112_112229


namespace equivalent_problem_l112_112741

-- Definitions that correspond to conditions
def valid_n (n : ℕ) : Prop := n < 13 ∧ (4 * n) % 13 = 1

-- The equivalent proof problem
theorem equivalent_problem (n : ℕ) (h : valid_n n) : ((3 ^ n) ^ 4 - 3) % 13 = 6 := by
  sorry

end equivalent_problem_l112_112741


namespace geometric_seq_condition_l112_112881

-- Defining a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Defining an increasing sequence
def is_increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The condition to be proved
theorem geometric_seq_condition (a : ℕ → ℝ) (h_geo : is_geometric_seq a) :
  (a 0 < a 1 → is_increasing_seq a) ∧ (is_increasing_seq a → a 0 < a 1) :=
by 
  sorry

end geometric_seq_condition_l112_112881


namespace barker_high_school_team_count_l112_112669

theorem barker_high_school_team_count (students_total : ℕ) (baseball_team : ℕ) (hockey_team : ℕ) 
  (both_sports : ℕ) : 
  students_total = 36 → baseball_team = 25 → hockey_team = 19 → both_sports = (baseball_team + hockey_team - students_total) → both_sports = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end barker_high_school_team_count_l112_112669


namespace flagpole_height_l112_112218

theorem flagpole_height (h : ℕ) (shadow_flagpole : ℕ) (height_building : ℕ) (shadow_building : ℕ) (similar_conditions : Prop) 
  (H1 : shadow_flagpole = 45) 
  (H2 : height_building = 24) 
  (H3 : shadow_building = 60) 
  (H4 : similar_conditions) 
  (H5 : h / 45 = 24 / 60) : h = 18 := 
by 
sorry

end flagpole_height_l112_112218


namespace cos_alpha_add_pi_over_4_l112_112526

theorem cos_alpha_add_pi_over_4 (x y r : ℝ) (α : ℝ) (h1 : P = (3, -4)) (h2 : r = Real.sqrt (x^2 + y^2)) (h3 : x / r = Real.cos α) (h4 : y / r = Real.sin α) :
  Real.cos (α + Real.pi / 4) = (7 * Real.sqrt 2) / 10 := by
  sorry

end cos_alpha_add_pi_over_4_l112_112526


namespace e_exp_f_neg2_l112_112227

noncomputable def f : ℝ → ℝ := sorry

-- Conditions:
axiom h_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_ln_pos : ∀ x : ℝ, x > 0 → f x = Real.log x

-- Theorem to prove:
theorem e_exp_f_neg2 : Real.exp (f (-2)) = 1 / 2 := by
  sorry

end e_exp_f_neg2_l112_112227


namespace nth_inequality_l112_112511

theorem nth_inequality (x : ℝ) (n : ℕ) (h_x_pos : 0 < x) : x + (n^n / x^n) ≥ n + 1 := 
sorry

end nth_inequality_l112_112511


namespace equation_parallel_equation_perpendicular_l112_112380

variables {x y : ℝ}

def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x - 5 * y + 14 = 0
def l3 (x y : ℝ) := 2 * x - y + 7 = 0

theorem equation_parallel {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : 2 * x - y + 6 = 0 :=
sorry

theorem equation_perpendicular {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : x + 2 * y - 2 = 0 :=
sorry

end equation_parallel_equation_perpendicular_l112_112380


namespace negative_double_inequality_l112_112674

theorem negative_double_inequality (a : ℝ) (h : a < 0) : 2 * a < a :=
by { sorry }

end negative_double_inequality_l112_112674


namespace log_4_135_eq_half_log_2_45_l112_112596

noncomputable def a : ℝ := Real.log 135 / Real.log 4
noncomputable def b : ℝ := Real.log 45 / Real.log 2

theorem log_4_135_eq_half_log_2_45 : a = b / 2 :=
by
  sorry

end log_4_135_eq_half_log_2_45_l112_112596


namespace inequality_condition_l112_112759

theorem inequality_condition (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2 * x + a ≥ -y^2 - 2 * y) → a ≥ 2 :=
by
  sorry

end inequality_condition_l112_112759


namespace trigonometric_expression_value_l112_112906

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  2 * (Real.sin α)^2 + 4 * Real.sin α * Real.cos α - 9 * (Real.cos α)^2 = 21 / 10 :=
by
  sorry

end trigonometric_expression_value_l112_112906


namespace value_of_a_l112_112548

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := (2 : ℝ) * x - y - 1 = 0

def line2 (x y a : ℝ) : Prop := (2 : ℝ) * x + (a + 1) * y + 2 = 0

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 x y) → (line2 x y a)

-- The theorem to be proved
theorem value_of_a (a : ℝ) : parallel_lines a → a = -2 :=
sorry

end value_of_a_l112_112548


namespace arithmetic_sequence_sum_l112_112442

noncomputable def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℕ :=
  n * 2^n

def S_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (h1 : a_n 1 + a_n 2 + a_n 3 = 6)
  (h2 : a_n 5 = 5)
  (h3 : ∀ n, b_n n = a_n n * 2^(a_n n)) :
  (∀ n, a_n n = n) ∧ (∀ n, S_n n = (n - 1) * 2^(n + 1) + 2) :=
by
  sorry

end arithmetic_sequence_sum_l112_112442


namespace algebraic_expression_value_l112_112786

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end algebraic_expression_value_l112_112786


namespace ratio_of_numbers_l112_112550

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l112_112550


namespace find_y_l112_112911

theorem find_y
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hr : x % y = 8)
  (hq : x / y = 96) 
  (hr_decimal : (x:ℚ) / (y:ℚ) = 96.16) :
  y = 50 := 
sorry

end find_y_l112_112911


namespace sum_a2000_inv_a2000_l112_112395

theorem sum_a2000_inv_a2000 (a : ℂ) (h : a^2 - a + 1 = 0) : a^2000 + 1/(a^2000) = -1 :=
by
    sorry

end sum_a2000_inv_a2000_l112_112395


namespace percent_absent_students_l112_112828

def total_students : ℕ := 180
def num_boys : ℕ := 100
def num_girls : ℕ := 80
def fraction_boys_absent : ℚ := 1 / 5
def fraction_girls_absent : ℚ := 1 / 4

theorem percent_absent_students : 
  (fraction_boys_absent * num_boys + fraction_girls_absent * num_girls) / total_students = 22.22 / 100 := 
  sorry

end percent_absent_students_l112_112828


namespace cheolsu_initial_number_l112_112543

theorem cheolsu_initial_number (x : ℚ) (h : x + (-5/12) - (-5/2) = 1/3) : x = -7/4 :=
by 
  sorry

end cheolsu_initial_number_l112_112543


namespace ajay_distance_l112_112848

/- Definitions -/
def speed : ℝ := 50 -- Ajay's speed in km/hour
def time : ℝ := 30 -- Time taken in hours

/- Theorem statement -/
theorem ajay_distance : (speed * time = 1500) :=
by
  sorry

end ajay_distance_l112_112848


namespace lisa_interest_after_10_years_l112_112081

noncomputable def compounded_amount (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r) ^ n

theorem lisa_interest_after_10_years :
  let P := 2000
  let r := (2 : ℚ) / 100
  let n := 10
  let A := compounded_amount P r n
  A - P = 438 := by
    let P := 2000
    let r := (2 : ℚ) / 100
    let n := 10
    let A := compounded_amount P r n
    have : A - P = 438 := sorry
    exact this

end lisa_interest_after_10_years_l112_112081


namespace ratio_lateral_surface_area_to_surface_area_l112_112195

theorem ratio_lateral_surface_area_to_surface_area (r : ℝ) (h : ℝ) (V_sphere V_cone A_cone A_sphere : ℝ)
    (h_eq : h = r)
    (V_sphere_eq : V_sphere = (4 / 3) * Real.pi * r^3)
    (V_cone_eq : V_cone = (1 / 3) * Real.pi * (2 * r)^2 * h)
    (V_eq : V_sphere = V_cone)
    (A_cone_eq : A_cone = 2 * Real.sqrt 5 * Real.pi * r^2)
    (A_sphere_eq : A_sphere = 4 * Real.pi * r^2) :
    A_cone / A_sphere = Real.sqrt 5 / 2 := by
  sorry

end ratio_lateral_surface_area_to_surface_area_l112_112195


namespace solve_inequality_l112_112296

noncomputable def g (x : ℝ) := Real.arcsin x + x^3

theorem solve_inequality (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1)
    (h2 : Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3 > 0) :
    0 < x ∧ x ≤ 1 :=
by
  sorry

end solve_inequality_l112_112296


namespace proof_equivalence_l112_112489

noncomputable def compute_expression (N : ℕ) (M : ℕ) : ℚ :=
  ((N - 3)^3 + (N - 2)^3 + (N - 1)^3 + N^3 + (N + 1)^3 + (N + 2)^3 + (N + 3)^3) /
  ((M - 3) * (M - 2) + (M - 1) * M + M * (M + 1) + (M + 2) * (M + 3))

theorem proof_equivalence:
  let N := 65536
  let M := 32768
  compute_expression N M = 229376 := 
  by
    sorry

end proof_equivalence_l112_112489


namespace find_angle_l112_112849

theorem find_angle (A : ℝ) (h : 0 < A ∧ A < π) 
  (c : 4 * π * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = π / 6 ∨ A = 5 * π / 6 :=
sorry

end find_angle_l112_112849


namespace modulus_of_complex_division_l112_112012

noncomputable def complexDivisionModulus : ℂ := Complex.normSq (2 * Complex.I / (Complex.I - 1))

theorem modulus_of_complex_division : complexDivisionModulus = Real.sqrt 2 := by
  sorry

end modulus_of_complex_division_l112_112012


namespace profit_percentage_is_30_percent_l112_112240

theorem profit_percentage_is_30_percent (CP SP : ℕ) (h1 : CP = 280) (h2 : SP = 364) :
  ((SP - CP : ℤ) / (CP : ℤ) : ℚ) * 100 = 30 :=
by sorry

end profit_percentage_is_30_percent_l112_112240


namespace rotated_triangle_forms_two_cones_l112_112224

/-- Prove that the spatial geometric body formed when a right-angled triangle 
is rotated 360° around its hypotenuse is two cones. -/
theorem rotated_triangle_forms_two_cones (a b c : ℝ) (h1 : a^2 + b^2 = c^2) : 
  ∃ (cones : ℕ), cones = 2 :=
by
  sorry

end rotated_triangle_forms_two_cones_l112_112224


namespace simplify_fraction_l112_112085

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l112_112085


namespace children_on_bus_l112_112986

theorem children_on_bus (initial_children additional_children total_children : ℕ)
  (h1 : initial_children = 64)
  (h2 : additional_children = 14)
  (h3 : total_children = initial_children + additional_children) :
  total_children = 78 :=
by
  rw [h1, h2] at h3
  exact h3

end children_on_bus_l112_112986


namespace triangle_side_solution_l112_112617

/-- 
Given \( a \geq b \geq c > 0 \) and \( a < b + c \), a solution to the equation 
\( b \sqrt{x^{2} - c^{2}} + c \sqrt{x^{2} - b^{2}} = a x \) is provided by 
\( x = \frac{abc}{2 \sqrt{p(p-a)(p-b)(p-c)}} \) where \( p = \frac{1}{2}(a+b+c) \).
-/

theorem triangle_side_solution (a b c x : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  b * (Real.sqrt (x^2 - c^2)) + c * (Real.sqrt (x^2 - b^2)) = a * x → 
  x = (a * b * c) / (2 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
sorry

end triangle_side_solution_l112_112617


namespace evaluate_expression_l112_112791

theorem evaluate_expression : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 :=
by
  sorry

end evaluate_expression_l112_112791


namespace equation1_solution_equation2_solution_l112_112071

theorem equation1_solution (x : ℝ) (h : 5 / (x + 1) = 1 / (x - 3)) : x = 4 :=
sorry

theorem equation2_solution (x : ℝ) (h : (2 - x) / (x - 3) + 2 = 1 / (3 - x)) : x = 7 / 3 :=
sorry

end equation1_solution_equation2_solution_l112_112071


namespace cost_per_minute_l112_112231

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end cost_per_minute_l112_112231


namespace common_difference_l112_112525

variable {a : ℕ → ℤ} -- Define the arithmetic sequence

theorem common_difference (h : a 2015 = a 2013 + 6) : 
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 := 
by
  use 3
  sorry

end common_difference_l112_112525


namespace find_smallest_n_l112_112642

noncomputable def smallest_n (c : ℕ) (n : ℕ) : Prop :=
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ c → n + 2 - 2*k ≥ 0) ∧ c * (n - c + 1) = 2009

theorem find_smallest_n : ∃ n c : ℕ, smallest_n c n ∧ n = 89 :=
sorry

end find_smallest_n_l112_112642


namespace find_a_l112_112386

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end find_a_l112_112386


namespace melanie_mother_dimes_l112_112780

-- Definitions based on the conditions
variables (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ)

-- Conditions
def melanie_conditions := initial_dimes = 7 ∧ dimes_given_to_dad = 8 ∧ current_dimes = 3

-- Question to be proved is equivalent to proving the number of dimes given by her mother
theorem melanie_mother_dimes (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ) (dimes_given_by_mother : ℤ) 
  (h : melanie_conditions initial_dimes dimes_given_to_dad current_dimes) : 
  dimes_given_by_mother = 4 :=
by 
  sorry

end melanie_mother_dimes_l112_112780


namespace time_to_cross_is_correct_l112_112898

noncomputable def train_cross_bridge_time : ℝ :=
  let length_train := 130
  let speed_train_kmh := 45
  let length_bridge := 245.03
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_ms
  time

theorem time_to_cross_is_correct : train_cross_bridge_time = 30.0024 :=
by
  sorry

end time_to_cross_is_correct_l112_112898


namespace Triamoeba_Count_After_One_Week_l112_112275

def TriamoebaCount (n : ℕ) : ℕ :=
  3 ^ n

theorem Triamoeba_Count_After_One_Week : TriamoebaCount 7 = 2187 :=
by
  -- This is the statement to be proved
  sorry

end Triamoeba_Count_After_One_Week_l112_112275


namespace coeff_x6_in_expansion_l112_112087

theorem coeff_x6_in_expansion : 
  (Polynomial.coeff ((1 - 3 * Polynomial.X ^ 3) ^ 7 : Polynomial ℤ) 6) = 189 :=
by
  sorry

end coeff_x6_in_expansion_l112_112087


namespace mean_temperature_is_0_5_l112_112694

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature_is_0_5 :
  (temperatures.sum / temperatures.length) = 0.5 :=
by
  sorry

end mean_temperature_is_0_5_l112_112694


namespace total_tickets_sold_l112_112497

def SeniorPrice : Nat := 10
def RegularPrice : Nat := 15
def TotalSales : Nat := 855
def RegularTicketsSold : Nat := 41

theorem total_tickets_sold : ∃ (S R : Nat), R = RegularTicketsSold ∧ 10 * S + 15 * R = TotalSales ∧ S + R = 65 :=
by
  sorry

end total_tickets_sold_l112_112497


namespace not_true_B_l112_112593

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem not_true_B (x y : ℝ) : 2 * star x y ≠ star (2 * x) (2 * y) := by
  sorry

end not_true_B_l112_112593


namespace fraction_of_third_is_eighth_l112_112103

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l112_112103


namespace solution_set_of_abs_inequality_is_real_l112_112504

theorem solution_set_of_abs_inequality_is_real (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| + m - 7 > 0) ↔ m > 4 :=
by
  sorry

end solution_set_of_abs_inequality_is_real_l112_112504


namespace pendulum_faster_17_seconds_winter_l112_112480

noncomputable def pendulum_period (l g : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (l / g)

noncomputable def pendulum_seconds_faster_in_winter (T : ℝ) (l : ℝ) (g : ℝ) (shorten : ℝ) (hours : ℝ) : ℝ :=
  let summer_period := T
  let winter_length := l - shorten
  let winter_period := pendulum_period winter_length g
  let summer_cycles := (hours * 60 * 60) / summer_period
  let winter_cycles := (hours * 60 * 60) / winter_period
  winter_cycles - summer_cycles

theorem pendulum_faster_17_seconds_winter :
  let T := 1
  let l := 980 * (1 / (4 * Real.pi ^ 2))
  let g := 980
  let shorten := 0.01 / 100
  let hours := 24
  pendulum_seconds_faster_in_winter T l g shorten hours = 17 :=
by
  sorry

end pendulum_faster_17_seconds_winter_l112_112480


namespace person_speed_approx_l112_112614

noncomputable def convertDistance (meters : ℝ) : ℝ := meters * 0.000621371
noncomputable def convertTime (minutes : ℝ) (seconds : ℝ) : ℝ := (minutes + (seconds / 60)) / 60
noncomputable def calculateSpeed (distance_miles : ℝ) (time_hours : ℝ) : ℝ := distance_miles / time_hours

theorem person_speed_approx (street_length_meters : ℝ) (time_min : ℝ) (time_sec : ℝ) :
  street_length_meters = 900 →
  time_min = 3 →
  time_sec = 20 →
  abs ((calculateSpeed (convertDistance street_length_meters) (convertTime time_min time_sec)) - 10.07) < 0.01 :=
by
  sorry

end person_speed_approx_l112_112614


namespace equal_area_centroid_S_l112_112831

noncomputable def P : ℝ × ℝ := (-4, 3)
noncomputable def Q : ℝ × ℝ := (7, -5)
noncomputable def R : ℝ × ℝ := (0, 6)
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem equal_area_centroid_S (x y : ℝ) (h : (x, y) = centroid P Q R) :
  10 * x + y = 34 / 3 := by
  sorry

end equal_area_centroid_S_l112_112831


namespace mickys_sticks_more_l112_112683

theorem mickys_sticks_more 
  (simons_sticks : ℕ := 36)
  (gerrys_sticks : ℕ := (2 * simons_sticks) / 3)
  (total_sticks_needed : ℕ := 129)
  (total_simons_and_gerrys_sticks : ℕ := simons_sticks + gerrys_sticks)
  (mickys_sticks : ℕ := total_sticks_needed - total_simons_and_gerrys_sticks) :
  mickys_sticks - total_simons_and_gerrys_sticks = 9 :=
by
  sorry

end mickys_sticks_more_l112_112683


namespace cara_arrangements_l112_112575

theorem cara_arrangements (n : ℕ) (h : n = 7) : ∃ k : ℕ, k = 6 :=
by
  sorry

end cara_arrangements_l112_112575


namespace simplify_and_evaluate_l112_112033

theorem simplify_and_evaluate (a : ℤ) (h : a = -4) :
  (4 * a ^ 2 - 3 * a) - (2 * a ^ 2 + a - 1) + (2 - a ^ 2 + 4 * a) = 19 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l112_112033


namespace find_inverse_of_512_l112_112519

-- Define the function f with the given properties
def f : ℕ → ℕ := sorry

axiom f_initial : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

-- State the problem as a theorem
theorem find_inverse_of_512 : ∃ x, f x = 512 ∧ x = 1280 :=
by 
  -- Sorry to skip the proof
  sorry

end find_inverse_of_512_l112_112519


namespace time_to_pass_pole_l112_112725

def length_of_train : ℝ := 240
def length_of_platform : ℝ := 650
def time_to_pass_platform : ℝ := 89

theorem time_to_pass_pole (length_of_train length_of_platform time_to_pass_platform : ℝ) 
  (h_train : length_of_train = 240)
  (h_platform : length_of_platform = 650)
  (h_time : time_to_pass_platform = 89)
  : (length_of_train / ((length_of_train + length_of_platform) / time_to_pass_platform)) = 24 := by
  -- Let the speed of the train be v, hence
  -- v = (length_of_train + length_of_platform) / time_to_pass_platform
  -- What we need to prove is  
  -- length_of_train / v = 24
  sorry

end time_to_pass_pole_l112_112725


namespace equation_B_no_solution_l112_112321

theorem equation_B_no_solution : ¬ ∃ x : ℝ, |-2 * x| + 6 = 0 :=
by
  sorry

end equation_B_no_solution_l112_112321


namespace property_check_l112_112773

noncomputable def f (x : ℝ) : ℤ := ⌈x⌉ -- Define the ceiling function

theorem property_check :
  (¬ (∀ x : ℝ, f (2 * x) = 2 * f x)) ∧
  (∀ x1 x2 : ℝ, f x1 = f x2 → |x1 - x2| < 1) ∧
  (∀ x1 x2 : ℝ, f (x1 + x2) ≤ f x1 + f x2) ∧
  (¬ (∀ x : ℝ, f x + f (x + 0.5) = f (2 * x))) :=
by
  sorry

end property_check_l112_112773


namespace faces_painted_morning_l112_112491

def faces_of_cuboid : ℕ := 6
def faces_painted_evening : ℕ := 3

theorem faces_painted_morning : faces_of_cuboid - faces_painted_evening = 3 := 
by 
  sorry

end faces_painted_morning_l112_112491


namespace opponent_score_l112_112297

theorem opponent_score (s g c total opponent : ℕ)
  (h1 : s = 20)
  (h2 : g = 2 * s)
  (h3 : c = 2 * g)
  (h4 : total = s + g + c)
  (h5 : total - 55 = opponent) :
  opponent = 85 := by
  sorry

end opponent_score_l112_112297


namespace red_balls_number_l112_112148

namespace BallDrawing

variable (x : ℕ) -- define x as the number of red balls

noncomputable def total_balls : ℕ := x + 4
noncomputable def yellow_ball_probability : ℚ := 4 / total_balls x

theorem red_balls_number : yellow_ball_probability x = 0.2 → x = 16 :=
by
  unfold yellow_ball_probability
  sorry

end BallDrawing

end red_balls_number_l112_112148


namespace product_of_repeating_decimal_l112_112109

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l112_112109


namespace number_of_linear_eqs_l112_112588

def is_linear_eq_in_one_var (eq : String) : Bool :=
  match eq with
  | "0.3x = 1" => true
  | "x/2 = 5x + 1" => true
  | "x = 6" => true
  | _ => false

theorem number_of_linear_eqs :
  let eqs := ["x - 2 = 2 / x", "0.3x = 1", "x/2 = 5x + 1", "x^2 - 4x = 3", "x = 6", "x + 2y = 0"]
  (eqs.filter is_linear_eq_in_one_var).length = 3 :=
by
  sorry

end number_of_linear_eqs_l112_112588


namespace total_games_played_l112_112807

-- Definition of the conditions
def teams : Nat := 10
def games_per_pair : Nat := 4

-- Statement of the problem
theorem total_games_played (teams games_per_pair : Nat) : 
  teams = 10 → 
  games_per_pair = 4 → 
  ∃ total_games, total_games = 180 :=
by
  intro h1 h2
  sorry

end total_games_played_l112_112807


namespace sum_of_decimals_l112_112255

theorem sum_of_decimals : 1.000 + 0.101 + 0.011 + 0.001 = 1.113 :=
by
  sorry

end sum_of_decimals_l112_112255


namespace simplify_expression_l112_112784

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2*x - 2)/(x + 1)) / ((x^2 - x) / (2*x + 2)) = 2 - Real.sqrt 2 := 
by
  -- Here we should include the proof steps, but we skip it with "sorry"
  sorry

end simplify_expression_l112_112784


namespace option_d_not_equal_four_thirds_l112_112054

theorem option_d_not_equal_four_thirds :
  1 + (2 / 7) ≠ 4 / 3 :=
by
  sorry

end option_d_not_equal_four_thirds_l112_112054


namespace base_conversion_subtraction_l112_112465

theorem base_conversion_subtraction :
  (4 * 6^4 + 3 * 6^3 + 2 * 6^2 + 1 * 6^1 + 0 * 6^0) - (3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0) = 4776 :=
by {
  sorry
}

end base_conversion_subtraction_l112_112465


namespace remainder_of_prime_division_l112_112099

theorem remainder_of_prime_division
  (p : ℕ) (hp : Nat.Prime p)
  (r : ℕ) (hr : r = p % 210) 
  (hcomp : ¬ Nat.Prime r)
  (hsum : ∃ a b : ℕ, r = a^2 + b^2) : 
  r = 169 := 
sorry

end remainder_of_prime_division_l112_112099


namespace minimum_value_inequality_l112_112248

theorem minimum_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (Real.sqrt ((x^2 + 4 * y^2) * (2 * x^2 + 3 * y^2)) / (x * y)) ≥ 2 * Real.sqrt (2 * Real.sqrt 6) :=
sorry

end minimum_value_inequality_l112_112248


namespace tan_alpha_eq_neg_five_twelfths_l112_112187

-- Define the angle α and the given conditions
variables (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π)

-- The goal is to prove that tan α = -5 / 12
theorem tan_alpha_eq_neg_five_twelfths (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π) :
  Real.tan α = -5 / 12 :=
sorry

end tan_alpha_eq_neg_five_twelfths_l112_112187


namespace probability_of_red_card_l112_112631

theorem probability_of_red_card (successful_attempts not_successful_attempts : ℕ) (h : successful_attempts = 5) (h2 : not_successful_attempts = 8) : (successful_attempts / (successful_attempts + not_successful_attempts) : ℚ) = 5 / 13 := by
  sorry

end probability_of_red_card_l112_112631


namespace gross_profit_without_discount_l112_112984

variable (C P : ℝ) -- Defining the cost and the full price as real numbers

-- Condition 1: Merchant sells an item at 10% discount (0.9P)
-- Condition 2: Makes a gross profit of 20% of the cost (0.2C)
-- SP = C + GP implies 0.9 P = 1.2 C

theorem gross_profit_without_discount :
  (0.9 * P = 1.2 * C) → ((C / 3) / C * 100 = 33.33) :=
by
  intro h
  sorry

end gross_profit_without_discount_l112_112984


namespace range_of_b_l112_112871

noncomputable def f (b x : ℝ) : ℝ := -x^3 + b * x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → -3 * x^2 + b ≥ 0) ↔ b ≥ 3 := sorry

end range_of_b_l112_112871


namespace middle_card_is_four_l112_112351

theorem middle_card_is_four (a b c : ℕ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                            (h2 : a + b + c = 15)
                            (h3 : a < b ∧ b < c)
                            (h_casey : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_tracy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_stacy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            : b = 4 := 
sorry

end middle_card_is_four_l112_112351


namespace evaporation_amount_l112_112749

noncomputable def water_evaporated_per_day (total_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  (percentage_evaporated / 100) * total_water / days

theorem evaporation_amount :
  water_evaporated_per_day 10 7 50 = 0.014 :=
by
  sorry

end evaporation_amount_l112_112749


namespace initial_trees_count_l112_112673

variable (x : ℕ)

-- Conditions of the problem
def initial_rows := 24
def additional_rows := 12
def total_rows := initial_rows + additional_rows
def trees_per_row_initial := x
def trees_per_row_final := 28

-- Total number of trees should remain constant
theorem initial_trees_count :
  initial_rows * trees_per_row_initial = total_rows * trees_per_row_final → 
  trees_per_row_initial = 42 := 
by sorry

end initial_trees_count_l112_112673


namespace algebraic_expression_value_l112_112348

theorem algebraic_expression_value (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 :=
sorry

end algebraic_expression_value_l112_112348


namespace sufficient_but_not_necessary_l112_112719

def p (x : ℝ) : Prop := |x - 4| > 2
def q (x : ℝ) : Prop := x > 1

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 6 → x > 1) ∧ ¬(∀ x, x > 1 → 2 ≤ x ∧ x ≤ 6) :=
  sorry

end sufficient_but_not_necessary_l112_112719


namespace coordinates_P_l112_112505

theorem coordinates_P 
  (P1 P2 P : ℝ × ℝ)
  (hP1 : P1 = (2, -1))
  (hP2 : P2 = (0, 5))
  (h_ext_line : ∃ t : ℝ, P = (P1.1 + t * (P2.1 - P1.1), P1.2 + t * (P2.2 - P1.2)) ∧ t ≠ 1)
  (h_distance : dist P1 P = 2 * dist P P2) :
  P = (-2, 11) := 
by
  sorry

end coordinates_P_l112_112505


namespace bricks_in_chimney_proof_l112_112924

noncomputable def bricks_in_chimney (h : ℕ) : Prop :=
  let brenda_rate := h / 8
  let brandon_rate := h / 12
  let combined_rate_with_decrease := (brenda_rate + brandon_rate) - 12
  (6 * combined_rate_with_decrease = h) 

theorem bricks_in_chimney_proof : ∃ h : ℕ, bricks_in_chimney h ∧ h = 288 :=
sorry

end bricks_in_chimney_proof_l112_112924


namespace find_three_digit_number_l112_112036

theorem find_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
    (x - 6) % 7 = 0 ∧
    (x - 7) % 8 = 0 ∧
    (x - 8) % 9 = 0 ∧
    x = 503 :=
by
  sorry

end find_three_digit_number_l112_112036


namespace value_of_expression_l112_112065

theorem value_of_expression :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end value_of_expression_l112_112065


namespace smallest_integer_quad_ineq_l112_112206

-- Definition of the condition
def quad_ineq (n : ℤ) := n^2 - 14 * n + 45 > 0

-- Lean 4 statement of the math proof problem
theorem smallest_integer_quad_ineq : ∃ n : ℤ, quad_ineq n ∧ ∀ m : ℤ, quad_ineq m → n ≤ m :=
  by
    existsi 10
    sorry

end smallest_integer_quad_ineq_l112_112206


namespace sum_first_five_terms_eq_ninety_three_l112_112989

variable (a : ℕ → ℕ)

-- Definitions
def geometric_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

variables (a1 : ℕ) (a2 : ℕ) (a4 : ℕ)
variables (S : ℕ → ℕ)

-- Conditions
axiom a1_value : a1 = 3
axiom a2a4_value : a2 * a4 = 144

-- Question: Prove S_5 = 93
theorem sum_first_five_terms_eq_ninety_three
    (h1 : geometric_sequence a)
    (h2 : a 1 = a1)
    (h3 : a 2 = a2)
    (h4 : a 4 = a4)
    (Sn_def : S 5 = (a1 * (1 - (2:ℕ)^5)) / (1 - 2)) :
  S 5 = 93 :=
sorry

end sum_first_five_terms_eq_ninety_three_l112_112989


namespace convert_to_scientific_notation_l112_112131

theorem convert_to_scientific_notation (N : ℕ) (h : 2184300000 = 2184.3 * 10^6) : 
    (2184300000 : ℝ) = 2.1843 * 10^7 :=
by 
  sorry

end convert_to_scientific_notation_l112_112131


namespace point_not_in_second_quadrant_l112_112026

-- Define the point P and the condition
def point_is_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def point (m : ℝ) : ℝ × ℝ :=
  (m + 1, m)

-- The main theorem stating that P cannot be in the second quadrant
theorem point_not_in_second_quadrant (m : ℝ) : ¬ point_is_in_second_quadrant (point m) :=
by
  sorry

end point_not_in_second_quadrant_l112_112026


namespace value_of_a_l112_112702

theorem value_of_a (a : ℝ) (A B : ℝ × ℝ) (hA : A = (a - 2, 2 * a + 7)) (hB : B = (1, 5)) (h_parallel : (A.1 = B.1)) : a = 3 :=
by {
  sorry
}

end value_of_a_l112_112702


namespace mitzi_money_left_l112_112476

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end mitzi_money_left_l112_112476


namespace number_of_jump_sequences_l112_112292

def jump_sequences (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (a 3 = 3) ∧
  (∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 2))

theorem number_of_jump_sequences :
  ∃ a : ℕ → ℕ, jump_sequences a ∧ a 11 = 60 :=
by
  sorry

end number_of_jump_sequences_l112_112292


namespace Jason_age_l112_112462

theorem Jason_age : ∃ J K : ℕ, (J = 7 * K) ∧ (J + 4 = 3 * (2 * (K + 2))) ∧ (J = 56) :=
by
  sorry

end Jason_age_l112_112462


namespace solution1_solution2_l112_112959

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l112_112959


namespace part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l112_112539

-- Part 1: Proof that the solutions of 2x + y - 6 = 0 under positive integer constraints are (2, 2) and (1, 4)
theorem part1_positive_integer_solutions : 
  (∃ x y : ℤ, 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0) → 
  ({(x, y) | 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0} = {(2, 2), (1, 4)})
:= sorry

-- Part 2: Proof that if x = y, the value of m that satisfies the system of equations is -4
theorem part2_value_of_m (x y m : ℤ) : 
  x = y → (∃ m, (2 * x + y - 6 = 0 ∧ 2 * x - 2 * y + m * y + 8 = 0)) → m = -4
:= sorry

-- Part 3: Proof that regardless of m, there is a fixed solution (x, y) = (-4, 0) for the equation 2x - 2y + my + 8 = 0
theorem part3_fixed_solution (m : ℤ) : 
  2 * x - 2 * y + m * y + 8 = 0 → (x, y) = (-4, 0)
:= sorry

end part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l112_112539


namespace complete_square_eq_l112_112656

theorem complete_square_eq (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by
  sorry

end complete_square_eq_l112_112656


namespace fencing_cost_approx_122_52_l112_112088

noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

noncomputable def fencing_cost (d rate : ℝ) : ℝ := circumference d * rate

theorem fencing_cost_approx_122_52 :
  let d := 26
  let rate := 1.50
  abs (fencing_cost d rate - 122.52) < 1 :=
by
  let d : ℝ := 26
  let rate : ℝ := 1.50
  let cost := fencing_cost d rate
  sorry

end fencing_cost_approx_122_52_l112_112088


namespace general_term_of_seq_l112_112466

open Nat

noncomputable def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 3 * 2^n

theorem general_term_of_seq (a : ℕ → ℕ) :
  seq a → ∀ n, a n = (3 * n - 1) * 2^(n-1) :=
by
  sorry

end general_term_of_seq_l112_112466


namespace puppies_per_cage_l112_112199

theorem puppies_per_cage (initial_puppies : ℕ) (sold_puppies : ℕ) (remaining_puppies : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 9)
  (h5 : puppies_per_cage = remaining_puppies / cages) : 
  puppies_per_cage = 9 := 
by
  -- The proof should go here
  sorry

end puppies_per_cage_l112_112199


namespace scientific_notation_correct_l112_112095

noncomputable def scientific_notation (x : ℕ) : Prop :=
  x = 3010000000 → 3.01 * (10 ^ 9) = 3.01 * (10 ^ 9)

theorem scientific_notation_correct : 
  scientific_notation 3010000000 :=
by
  intros h
  sorry

end scientific_notation_correct_l112_112095


namespace ellipse_circle_inequality_l112_112850

theorem ellipse_circle_inequality
  (a b : ℝ) (x y : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : (x1^2) / (a^2) + (y1^2) / (b^2) = 1)
  (h_ellipse2 : (x2^2) / (a^2) + (y2^2) / (b^2) = 1)
  (h_ab : a > b ∧ b > 0)
  (h_circle : (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 :=
sorry

end ellipse_circle_inequality_l112_112850


namespace michael_can_cover_both_classes_l112_112405

open Nat

def total_students : ℕ := 30
def german_students : ℕ := 20
def japanese_students : ℕ := 24

-- Calculate the number of students taking both German and Japanese using inclusion-exclusion principle.
def both_students : ℕ := german_students + japanese_students - total_students

-- Calculate the number of students only taking German.
def only_german_students : ℕ := german_students - both_students

-- Calculate the number of students only taking Japanese.
def only_japanese_students : ℕ := japanese_students - both_students

-- Calculate the total number of ways to choose 2 students out of 30.
def total_ways_to_choose_2 : ℕ := (total_students * (total_students - 1)) / 2

-- Calculate the number of ways to choose 2 students only taking German or only taking Japanese.
def undesirable_outcomes : ℕ := (only_german_students * (only_german_students - 1)) / 2 + (only_japanese_students * (only_japanese_students - 1)) / 2

-- Calculate the probability of undesirable outcomes.
def undesirable_probability : ℚ := undesirable_outcomes / total_ways_to_choose_2

-- Calculate the probability Michael can cover both German and Japanese classes.
def desired_probability : ℚ := 1 - undesirable_probability

theorem michael_can_cover_both_classes : desired_probability = 25 / 29 := by sorry

end michael_can_cover_both_classes_l112_112405


namespace total_cable_cost_l112_112393

theorem total_cable_cost 
    (num_east_west_streets : ℕ)
    (length_east_west_street : ℕ)
    (num_north_south_streets : ℕ)
    (length_north_south_street : ℕ)
    (cable_multiplier : ℕ)
    (cable_cost_per_mile : ℕ)
    (h1 : num_east_west_streets = 18)
    (h2 : length_east_west_street = 2)
    (h3 : num_north_south_streets = 10)
    (h4 : length_north_south_street = 4)
    (h5 : cable_multiplier = 5)
    (h6 : cable_cost_per_mile = 2000) :
    (num_east_west_streets * length_east_west_street + num_north_south_streets * length_north_south_street) * cable_multiplier * cable_cost_per_mile = 760000 := 
by
    sorry

end total_cable_cost_l112_112393


namespace line_contains_point_iff_k_eq_neg1_l112_112256

theorem line_contains_point_iff_k_eq_neg1 (k : ℝ) :
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ (2 - k * x = -4 * y)) ↔ k = -1 :=
by
  sorry

end line_contains_point_iff_k_eq_neg1_l112_112256


namespace relative_value_ex1_max_value_of_m_plus_n_l112_112041

-- Definition of relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ := abs (a - n) + abs (b - n)

-- First problem statement
theorem relative_value_ex1 : relative_relationship_value 2 (-5) 2 = 7 := by
  sorry

-- Second problem statement: maximum value of m + n given the relative relationship value is 2
theorem max_value_of_m_plus_n (m n : ℚ) (h : relative_relationship_value m n 2 = 2) : m + n ≤ 6 := by
  sorry

end relative_value_ex1_max_value_of_m_plus_n_l112_112041


namespace product_of_two_numbers_l112_112205

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end product_of_two_numbers_l112_112205


namespace total_ticket_cost_l112_112394

theorem total_ticket_cost (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : V = G - 276) 
  (price_vip : ℕ := 45) 
  (price_regular : ℕ := 20) : 
  (price_vip * V + price_regular * G = 6950) :=
by sorry

end total_ticket_cost_l112_112394


namespace jake_peaches_is_7_l112_112940

variable (Steven_peaches Jake_peaches Jill_peaches : ℕ)

-- Conditions:
def Steven_has_19_peaches : Steven_peaches = 19 := by sorry

def Jake_has_12_fewer_peaches_than_Steven : Jake_peaches = Steven_peaches - 12 := by sorry

def Jake_has_72_more_peaches_than_Jill : Jake_peaches = Jill_peaches + 72 := by sorry

-- Proof problem:
theorem jake_peaches_is_7 
    (Steven_peaches Jake_peaches Jill_peaches : ℕ)
    (h1 : Steven_peaches = 19)
    (h2 : Jake_peaches = Steven_peaches - 12)
    (h3 : Jake_peaches = Jill_peaches + 72) :
    Jake_peaches = 7 := by sorry

end jake_peaches_is_7_l112_112940


namespace find_other_integer_l112_112971

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 140) (h2 : x = 20 ∨ y = 20) : x = 20 ∧ y = 20 :=
by
  sorry

end find_other_integer_l112_112971


namespace remainder_calculation_l112_112400

theorem remainder_calculation 
  (x : ℤ) (y : ℝ)
  (hx : 0 < x)
  (hy : y = 70.00000000000398)
  (hx_div_y : (x : ℝ) / y = 86.1) :
  x % y = 7 :=
by
  sorry

end remainder_calculation_l112_112400


namespace min_value_n_l112_112313

theorem min_value_n (n : ℕ) (h1 : 4 ∣ 60 * n) (h2 : 8 ∣ 60 * n) : n = 1 := 
  sorry

end min_value_n_l112_112313


namespace at_least_30_cents_probability_l112_112146

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l112_112146


namespace tank_capacity_percentage_l112_112360

noncomputable def radius (C : ℝ) := C / (2 * Real.pi)
noncomputable def volume (r h : ℝ) := Real.pi * r^2 * h

theorem tank_capacity_percentage :
  let r_M := radius 8
  let r_B := radius 10
  let V_M := volume r_M 10
  let V_B := volume r_B 8
  (V_M / V_B * 100) = 80 :=
by
  sorry

end tank_capacity_percentage_l112_112360


namespace intersection_points_l112_112918

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

noncomputable def x1 : ℝ := (3 + Real.sqrt 209) / 4
noncomputable def x2 : ℝ := (3 - Real.sqrt 209) / 4

noncomputable def y1 : ℝ := parabola1 x1
noncomputable def y2 : ℝ := parabola1 x2

theorem intersection_points :
  (parabola1 x1 = parabola2 x1) ∧ (parabola1 x2 = parabola2 x2) :=
by
  sorry

end intersection_points_l112_112918


namespace residue_class_equivalence_l112_112464

variable {a m : ℤ}
variable {b : ℤ}

def residue_class (a m b : ℤ) : Prop := ∃ t : ℤ, b = m * t + a

theorem residue_class_equivalence (m a b : ℤ) :
  (∃ t : ℤ, b = m * t + a) ↔ b % m = a % m :=
by sorry

end residue_class_equivalence_l112_112464


namespace not_perfect_square_l112_112113

theorem not_perfect_square (a : ℤ) : ¬ (∃ x : ℤ, a^2 + 4 = x^2) := 
sorry

end not_perfect_square_l112_112113


namespace green_peppers_weight_l112_112894

theorem green_peppers_weight (total_weight : ℝ) (w : ℝ) (h1 : total_weight = 5.666666667)
  (h2 : 2 * w = total_weight) : w = 2.8333333335 :=
by
  sorry

end green_peppers_weight_l112_112894


namespace elliot_book_pages_l112_112180

theorem elliot_book_pages : 
  ∀ (initial_pages read_per_day days_in_week remaining_pages total_pages: ℕ), 
    initial_pages = 149 → 
    read_per_day = 20 → 
    days_in_week = 7 → 
    remaining_pages = 92 → 
    total_pages = initial_pages + (read_per_day * days_in_week) + remaining_pages → 
    total_pages = 381 :=
by
  intros initial_pages read_per_day days_in_week remaining_pages total_pages
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  assumption

end elliot_book_pages_l112_112180


namespace tan_x_eq_2_solution_set_l112_112648

theorem tan_x_eq_2_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} = {x : ℝ | Real.tan x = 2} :=
sorry

end tan_x_eq_2_solution_set_l112_112648


namespace correct_calculation_l112_112846

theorem correct_calculation :
  (∀ (x : ℝ), (x^3 * 2 * x^4 = 2 * x^7) ∧
  (x^6 / x^3 = x^2) ∧
  ((x^3)^4 = x^7) ∧
  (x^2 + x = x^3)) → 
  (∀ (x : ℝ), x^3 * 2 * x^4 = 2 * x^7) :=
by
  intros h x
  have A := h x
  exact A.1

end correct_calculation_l112_112846


namespace solve_problem_l112_112141

-- Define the polynomial g(x) as given in the problem
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- Define the condition given in the problem
def condition (p q r s t : ℝ) : Prop := g p q r s t (-2) = -4

-- State the theorem to be proved
theorem solve_problem (p q r s t : ℝ) (h : condition p q r s t) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 4 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l112_112141


namespace total_packing_peanuts_used_l112_112605

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def large_orders_sent : ℕ := 3
def small_orders_sent : ℕ := 4

theorem total_packing_peanuts_used :
  (large_orders_sent * large_order_weight) + (small_orders_sent * small_order_weight) = 800 := 
by
  sorry

end total_packing_peanuts_used_l112_112605


namespace find_largest_square_area_l112_112120

def area_of_largest_square (XY YZ XZ : ℝ) (sum_of_areas : ℝ) (right_angle : Prop) : Prop :=
  sum_of_areas = XY^2 + YZ^2 + XZ^2 + 4 * YZ^2 ∧  -- sum of areas condition
  right_angle ∧                                    -- right angle condition
  XZ^2 = XY^2 + YZ^2 ∧                             -- Pythagorean theorem
  sum_of_areas = 650 ∧                             -- total area condition
  XY = YZ                                          -- assumption for simplified solving.

theorem find_largest_square_area (XY YZ XZ : ℝ) (sum_of_areas : ℝ):
  area_of_largest_square XY YZ XZ sum_of_areas (90 = 90) → 2 * XY^2 + 5 * YZ^2 = 650 → XZ^2 = 216.67 :=
sorry

end find_largest_square_area_l112_112120


namespace total_plate_combinations_l112_112230

open Nat

def valid_letters := 24
def letter_positions := (choose 4 2)
def valid_digits := 10
def total_combinations := letter_positions * (valid_letters * valid_letters) * (valid_digits ^ 3)

theorem total_plate_combinations : total_combinations = 3456000 :=
  by
    -- Replace this sorry with steps to prove the theorem
    sorry

end total_plate_combinations_l112_112230


namespace value_of_y_for_absolute_value_eq_zero_l112_112769

theorem value_of_y_for_absolute_value_eq_zero :
  ∃ (y : ℚ), |(2:ℚ) * y - 3| ≤ 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_for_absolute_value_eq_zero_l112_112769


namespace rhombus_properties_l112_112263

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2
noncomputable def side_length_of_rhombus (d1 d2 : ℝ) : ℝ := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)

theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 16) :
  area_of_rhombus d1 d2 = 144 ∧ side_length_of_rhombus d1 d2 = Real.sqrt 145 := by
  sorry

end rhombus_properties_l112_112263


namespace number_of_functions_with_given_range_l112_112302

theorem number_of_functions_with_given_range : 
  let S := {2, 5, 10}
  let R (x : ℤ) := x^2 + 1
  ∃ f : ℤ → ℤ, (∀ y ∈ S, ∃ x : ℤ, f x = y) ∧ (f '' {x | R x ∈ S} = S) :=
    sorry

end number_of_functions_with_given_range_l112_112302


namespace exist_indices_for_sequences_l112_112062

open Nat

theorem exist_indices_for_sequences 
  (a b c : ℕ → ℕ) : 
  ∃ p q, p ≠ q ∧ p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
  sorry

end exist_indices_for_sequences_l112_112062


namespace find_k_l112_112478

-- Define the conditions
variables (k : ℝ) -- the variable k
variables (x1 : ℝ) -- x1 coordinate of point A on the graph y = k/x
variable (AREA_ABCD : ℝ := 10) -- the area of the quadrilateral ABCD

-- The statement to be proven
theorem find_k (k : ℝ) (h1 : ∀ x1 : ℝ, (0 < x1 ∧ 2 * abs k = AREA_ABCD → x1 * abs k * 2 = AREA_ABCD)) : k = -5 :=
sorry

end find_k_l112_112478


namespace find_k_l112_112635

-- Definitions of the vectors and condition about perpendicularity
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (-2, k)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- The theorem that states if vector_a is perpendicular to (2 * vector_a - vector_b), then k = 14
theorem find_k (k : ℝ) (h : perpendicular vector_a (2 • vector_a - vector_b k)) : k = 14 := sorry

end find_k_l112_112635


namespace proof_problem_l112_112056

-- Let P, Q, R be points on a circle of radius s
-- Given: PQ = PR, PQ > s, and minor arc QR is 2s
-- Prove: PQ / QR = sin(1)

noncomputable def point_on_circle (s : ℝ) : ℝ → ℝ × ℝ := sorry
def radius {s : ℝ} (P Q : ℝ × ℝ ) : Prop := dist P Q = s

theorem proof_problem (s : ℝ) (P Q R : ℝ × ℝ)
  (hPQ : dist P Q = dist P R)
  (hPQ_gt_s : dist P Q > s)
  (hQR_arc_len : 1 = s) :
  dist P Q / (2 * s) = Real.sin 1 := 
sorry

end proof_problem_l112_112056


namespace marcia_savings_l112_112832

def hat_price := 60
def regular_price (n : ℕ) := n * hat_price
def discount_price (discount_percentage: ℕ) (price: ℕ) := price - (price * discount_percentage) / 100
def promotional_price := hat_price + discount_price 25 hat_price + discount_price 35 hat_price

theorem marcia_savings : (regular_price 3 - promotional_price) * 100 / regular_price 3 = 20 :=
by
  -- The proof steps would follow here.
  sorry

end marcia_savings_l112_112832


namespace union_complement_eq_l112_112737

/-- The universal set U and sets A and B as given in the problem. -/
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

/-- The lean statement of our proof problem. -/
theorem union_complement_eq : A ∪ (U \ B) = {0, 1, 2, 3} := by
  sorry

end union_complement_eq_l112_112737


namespace problem1_problem2_l112_112696

-- Define points A, B, C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := -2}
def B : Point := {x := 2, y := 1}
def C : Point := {x := 3, y := 2}

-- Function to compute vector difference
def vector_sub (p1 p2 : Point) : Point :=
  {x := p1.x - p2.x, y := p1.y - p2.y}

-- Function to compute vector scalar multiplication
def scalar_mul (k : ℝ) (p : Point) : Point :=
  {x := k * p.x, y := k * p.y}

-- Function to add two vectors
def vec_add (p1 p2 : Point) : Point :=
  {x := p1.x + p2.x, y := p1.y + p2.y}

-- Problem 1
def result_vector : Point :=
  let AB := vector_sub B A
  let AC := vector_sub C A
  let BC := vector_sub C B
  vec_add (scalar_mul 3 AB) (vec_add (scalar_mul (-2) AC) BC)

-- Prove the coordinates are (0, 2)
theorem problem1 : result_vector = {x := 0, y := 2} := by
  sorry

-- Problem 2
def D : Point :=
  let BC := vector_sub C B
  {x := 1 + BC.x, y := (-2) + BC.y}

-- Prove the coordinates are (2, -1)
theorem problem2 : D = {x := 2, y := -1} := by
  sorry

end problem1_problem2_l112_112696


namespace opposite_of_fraction_l112_112541

theorem opposite_of_fraction : - (11 / 2022 : ℚ) = -(11 / 2022) := 
by
  sorry

end opposite_of_fraction_l112_112541


namespace arithmetic_sequence_property_l112_112048

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence
variable {d : ℝ} -- Define the common difference
variable {a1 : ℝ} -- Define the first term

-- Suppose the sum of the first 17 terms equals 306
axiom h1 : S 17 = 306
-- Suppose the sum of the first n terms of an arithmetic sequence formula
axiom sum_formula : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
-- Suppose the relation between the first term, common difference and sum of the first 17 terms
axiom relation : a1 + 8 * d = 18 

theorem arithmetic_sequence_property : a 7 - (a 3) / 3 = 12 := 
by sorry

end arithmetic_sequence_property_l112_112048


namespace probability_of_drawing_2_red_1_white_l112_112004

def total_balls : ℕ := 7
def red_balls : ℕ := 4
def white_balls : ℕ := 3
def draws : ℕ := 3

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_drawing_2_red_1_white :
  (combinations red_balls 2) * (combinations white_balls 1) / (combinations total_balls draws) = 18 / 35 := by
  sorry

end probability_of_drawing_2_red_1_white_l112_112004


namespace find_triples_l112_112343

-- Defining the conditions
def divides (x y : ℕ) : Prop := ∃ k, y = k * x

-- The main Lean statement
theorem find_triples (a b c : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  divides a (b * c - 1) → divides b (a * c - 1) → divides c (a * b - 1) →
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 3) ∨
  (a = 3 ∧ b = 2 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 2) ∨
  (a = 5 ∧ b = 2 ∧ c = 3) ∨ (a = 5 ∧ b = 3 ∧ c = 2) :=
sorry

end find_triples_l112_112343


namespace Frank_initial_savings_l112_112778

theorem Frank_initial_savings 
  (cost_per_toy : Nat)
  (number_of_toys : Nat)
  (allowance : Nat)
  (total_cost : Nat)
  (initial_savings : Nat)
  (h1 : cost_per_toy = 8)
  (h2 : number_of_tys = 5)
  (h3 : allowance = 37)
  (h4 : total_cost = number_of_toys * cost_per_toy)
  (h5 : initial_savings + allowance = total_cost)
  : initial_savings = 3 := 
by
  sorry

end Frank_initial_savings_l112_112778


namespace number_of_dogs_l112_112031

theorem number_of_dogs (total_animals cats : ℕ) (probability : ℚ) (h1 : total_animals = 7) (h2 : cats = 2) (h3 : probability = 2 / 7) :
  total_animals - cats = 5 := 
by
  sorry

end number_of_dogs_l112_112031


namespace real_b_values_for_non_real_roots_l112_112384

theorem real_b_values_for_non_real_roots (b : ℝ) :
  let discriminant := b^2 - 4 * 1 * 16
  discriminant < 0 ↔ -8 < b ∧ b < 8 := 
sorry

end real_b_values_for_non_real_roots_l112_112384


namespace tourism_revenue_scientific_notation_l112_112203

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end tourism_revenue_scientific_notation_l112_112203


namespace remainder_when_divided_l112_112342

theorem remainder_when_divided (x : ℤ) (k : ℤ) (h: x = 82 * k + 5) : 
  ((x + 17) % 41) = 22 := by
  sorry

end remainder_when_divided_l112_112342


namespace line_equation_l112_112305

theorem line_equation
  (P : ℝ × ℝ) (hP : P = (1, -1))
  (h_perp : ∀ x y : ℝ, 3 * x - 2 * y = 0 → 2 * x + 3 * y = 0):
  ∃ m : ℝ, (2 * P.1 + 3 * P.2 + m = 0) ∧ m = 1 :=
by
  sorry

end line_equation_l112_112305


namespace ephraim_keiko_same_heads_probability_l112_112805

def coin_toss_probability_same_heads : ℚ :=
  let keiko_prob_0 := 1 / 4
  let keiko_prob_1 := 1 / 2
  let keiko_prob_2 := 1 / 4
  let ephraim_prob_0 := 1 / 8
  let ephraim_prob_1 := 3 / 8
  let ephraim_prob_2 := 3 / 8
  let ephraim_prob_3 := 1 / 8
  (keiko_prob_0 * ephraim_prob_0) 
  + (keiko_prob_1 * ephraim_prob_1) 
  + (keiko_prob_2 * ephraim_prob_2)

theorem ephraim_keiko_same_heads_probability : 
  coin_toss_probability_same_heads = 11 / 32 :=
by 
  unfold coin_toss_probability_same_heads
  norm_num
  sorry

end ephraim_keiko_same_heads_probability_l112_112805


namespace tan_gt_neg_one_solution_set_l112_112420

def tangent_periodic_solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 2

theorem tan_gt_neg_one_solution_set (x : ℝ) :
  tangent_periodic_solution_set x ↔ Real.tan x > -1 :=
by
  sorry

end tan_gt_neg_one_solution_set_l112_112420


namespace hyperbola_equation_is_correct_l112_112269

-- Given Conditions
def hyperbola_eq (x y : ℝ) (a : ℝ) : Prop := (x^2) / (a^2) - (y^2) / 4 = 1
def asymptote_eq (x y : ℝ) : Prop := y = (1 / 2) * x

-- Correct answer to be proven
def hyperbola_correct (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 4 = 1

theorem hyperbola_equation_is_correct (x y : ℝ) (a : ℝ) :
  (hyperbola_eq x y a) → (asymptote_eq x y) → (a = 4) → hyperbola_correct x y :=
by 
  intros h_hyperbola h_asymptote h_a
  sorry

end hyperbola_equation_is_correct_l112_112269


namespace fraction_zero_l112_112929

theorem fraction_zero (x : ℝ) (h₁ : x - 3 = 0) (h₂ : x ≠ 0) : (x - 3) / (4 * x) = 0 :=
by
  sorry

end fraction_zero_l112_112929


namespace pairs_satisfied_condition_l112_112181

def set_A : Set ℕ := {1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 20, 22, 30, 33, 44, 55, 60, 66, 110, 132, 165, 220, 330, 660}
def set_B : Set ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def is_valid_pair (a b : ℕ) := a ∈ set_A ∧ b ∈ set_B ∧ (a - b = 4)

def valid_pairs : Set (ℕ × ℕ) := 
  {(6, 2), (10, 6), (12, 8), (22, 18)}

theorem pairs_satisfied_condition :
  { (a, b) | is_valid_pair a b } = valid_pairs := 
sorry

end pairs_satisfied_condition_l112_112181


namespace primes_equal_l112_112863

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_equal (p q r n : ℕ) (h_prime_p : is_prime p) (h_prime_q : is_prime q)
(h_prime_r : is_prime r) (h_pos_n : 0 < n)
(h1 : (p + n) % (q * r) = 0)
(h2 : (q + n) % (r * p) = 0)
(h3 : (r + n) % (p * q) = 0) : p = q ∧ q = r := by
  sorry

end primes_equal_l112_112863


namespace fraction_numerator_l112_112509

theorem fraction_numerator (x : ℚ) : 
  (∃ y : ℚ, y = 4 * x + 4 ∧ x / y = 3 / 7) → x = -12 / 5 :=
by
  sorry

end fraction_numerator_l112_112509


namespace parallelogram_circumference_l112_112414

-- Define the lengths of the sides of the parallelogram.
def side1 : ℝ := 18
def side2 : ℝ := 12

-- Define the formula for the circumference (or perimeter) of the parallelogram.
def circumference (a b : ℝ) : ℝ :=
  2 * (a + b)

-- Statement of the proof problem:
theorem parallelogram_circumference : circumference side1 side2 = 60 := 
  by
    sorry

end parallelogram_circumference_l112_112414


namespace speed_of_train_b_l112_112086

-- Defining the known data
def train_a_speed := 60 -- km/h
def train_a_time_after_meeting := 9 -- hours
def train_b_time_after_meeting := 4 -- hours

-- Statement we want to prove
theorem speed_of_train_b : ∃ (V_b : ℝ), V_b = 135 :=
by
  -- Sorry placeholder, as the proof is not required
  sorry

end speed_of_train_b_l112_112086


namespace heesu_has_greatest_sum_l112_112049

def sum_cards (cards : List Int) : Int :=
  cards.foldl (· + ·) 0

theorem heesu_has_greatest_sum :
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sum_cards heesu_cards > sum_cards sora_cards ∧ sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sorry

end heesu_has_greatest_sum_l112_112049


namespace waiter_total_customers_l112_112879

theorem waiter_total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) (tables_eq : tables = 6) (women_eq : women_per_table = 3) (men_eq : men_per_table = 5) :
  tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end waiter_total_customers_l112_112879


namespace total_number_of_coins_is_15_l112_112968

theorem total_number_of_coins_is_15 (x : ℕ) (h : 1*x + 5*x + 10*x + 25*x + 50*x = 273) : 5 * x = 15 :=
by {
  -- Proof omitted
  sorry
}

end total_number_of_coins_is_15_l112_112968


namespace count_students_with_green_eyes_l112_112477

-- Definitions for the given conditions
def total_students := 50
def students_with_both := 10
def students_with_neither := 5

-- Let the number of students with green eyes be y
variable (y : ℕ) 

-- There are twice as many students with brown hair as with green eyes
def students_with_brown := 2 * y

-- There are y - 10 students with green eyes only
def students_with_green_only := y - students_with_both

-- There are 2y - 10 students with brown hair only
def students_with_brown_only := students_with_brown - students_with_both

-- Proof statement
theorem count_students_with_green_eyes (y : ℕ) 
  (h1 : (students_with_green_only) + (students_with_brown_only) + students_with_both + students_with_neither = total_students) : y = 15 := 
by
  -- sorry to skip the proof
  sorry

end count_students_with_green_eyes_l112_112477


namespace nth_equation_l112_112707

theorem nth_equation (n : ℕ) (hn : n > 0) : 9 * n + (n - 1) = 10 * n - 1 :=
sorry

end nth_equation_l112_112707


namespace simplify_and_evaluate_l112_112066

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  ((1 / (x - 1)) + (1 / (x + 1))) / (x^2 / (3 * x^2 - 3))

theorem simplify_and_evaluate : simplified_expression (Real.sqrt 2) = 3 * Real.sqrt 2 :=
by 
  sorry

end simplify_and_evaluate_l112_112066


namespace cost_of_eraser_l112_112156

theorem cost_of_eraser 
  (s n c : ℕ)
  (h1 : s > 18)
  (h2 : n > 2)
  (h3 : c > n)
  (h4 : s * c * n = 3978) : 
  c = 17 :=
sorry

end cost_of_eraser_l112_112156


namespace range_of_a_l112_112678

open Real

theorem range_of_a (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |2^x₁ - a| = 1 ∧ |2^x₂ - a| = 1) ↔ 1 < a :=
by 
    sorry

end range_of_a_l112_112678


namespace inradii_sum_l112_112609

theorem inradii_sum (ABCD : Type) (r_a r_b r_c r_d : ℝ) 
  (inscribed_quadrilateral : Prop) 
  (inradius_BCD : Prop) 
  (inradius_ACD : Prop) 
  (inradius_ABD : Prop) 
  (inradius_ABC : Prop) 
  (Tebo_theorem : Prop) :
  r_a + r_c = r_b + r_d := 
by
  sorry

end inradii_sum_l112_112609


namespace x_intercept_of_line_l112_112374

theorem x_intercept_of_line (x y : ℚ) (h_eq : 4 * x + 7 * y = 28) (h_y : y = 0) : (x, y) = (7, 0) := 
by 
  sorry

end x_intercept_of_line_l112_112374


namespace decode_CLUE_is_8671_l112_112953

def BEST_OF_LUCK_code : List (Char × Nat) :=
  [('B', 0), ('E', 1), ('S', 2), ('T', 3), ('O', 4), ('F', 5),
   ('L', 6), ('U', 7), ('C', 8), ('K', 9)]

def decode (code : List (Char × Nat)) (word : String) : Option Nat :=
  word.toList.mapM (λ c => List.lookup c code) >>= (λ digits => 
  Option.some (Nat.ofDigits 10 digits))

theorem decode_CLUE_is_8671 :
  decode BEST_OF_LUCK_code "CLUE" = some 8671 :=
by
  -- Proof omitted
  sorry

end decode_CLUE_is_8671_l112_112953


namespace find_height_of_larger_cuboid_l112_112015

-- Define the larger cuboid dimensions
def Length_large : ℝ := 18
def Width_large : ℝ := 15
def Volume_large (Height_large : ℝ) : ℝ := Length_large * Width_large * Height_large

-- Define the smaller cuboid dimensions
def Length_small : ℝ := 5
def Width_small : ℝ := 6
def Height_small : ℝ := 3
def Volume_small : ℝ := Length_small * Width_small * Height_small

-- Define the total volume of 6 smaller cuboids
def Total_volume_small : ℝ := 6 * Volume_small

-- State the problem and the proof goal
theorem find_height_of_larger_cuboid : 
  ∃ H : ℝ, Volume_large H = Total_volume_small :=
by
  use 2
  sorry

end find_height_of_larger_cuboid_l112_112015


namespace wall_length_l112_112157

theorem wall_length (mirror_side length width : ℝ) (h1 : mirror_side = 21) (h2 : width = 28) 
  (h3 : 2 * mirror_side^2 = width * length) : length = 31.5 := by
  sorry

end wall_length_l112_112157


namespace prob_CD_l112_112164

variable (P : String → ℚ)
variable (x : ℚ)

axiom probA : P "A" = 1 / 3
axiom probB : P "B" = 1 / 4
axiom probC : P "C" = 2 * x
axiom probD : P "D" = x
axiom sumProb : P "A" + P "B" + P "C" + P "D" = 1

theorem prob_CD :
  P "D" = 5 / 36 ∧ P "C" = 5 / 18 := by
  sorry

end prob_CD_l112_112164


namespace rooms_per_floor_l112_112198

-- Definitions for each of the conditions
def numberOfFloors : ℕ := 4
def hoursPerRoom : ℕ := 6
def hourlyRate : ℕ := 15
def totalEarnings : ℕ := 3600

-- Statement of the problem
theorem rooms_per_floor : 
  (totalEarnings / hourlyRate) / hoursPerRoom / numberOfFloors = 10 := 
  sorry

end rooms_per_floor_l112_112198


namespace find_length_AB_l112_112922

variables {A B C D E : Type} -- Define variables A, B, C, D, E as types, representing points

-- Define lengths of the segments AD and CD
def length_AD : ℝ := 2
def length_CD : ℝ := 2

-- Define the angles at vertices B, C, and D
def angle_B : ℝ := 30
def angle_C : ℝ := 90
def angle_D : ℝ := 120

-- The goal is to prove the length of segment AB
theorem find_length_AB : 
  (∃ (A B C D : Type) 
    (angle_B angle_C angle_D length_AD length_CD : ℝ), 
      angle_B = 30 ∧ 
      angle_C = 90 ∧ 
      angle_D = 120 ∧ 
      length_AD = 2 ∧ 
      length_CD = 2) → 
  (length_AB = 6) := by sorry

end find_length_AB_l112_112922


namespace cannot_finish_third_l112_112061

-- Define the racers
inductive Racer
| P | Q | R | S | T | U
open Racer

-- Define the conditions
def beats (a b : Racer) : Prop := sorry  -- placeholder for strict order
def ties (a b : Racer) : Prop := sorry   -- placeholder for tie condition
def position (r : Racer) (p : Fin (6)) : Prop := sorry  -- placeholder for position in the race

theorem cannot_finish_third :
  (beats P Q) ∧
  (ties P R) ∧
  (beats Q S) ∧
  ∃ p₁ p₂ p₃, position P p₁ ∧ position T p₂ ∧ position Q p₃ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧
  ∃ p₄ p₅, position U p₄ ∧ position S p₅ ∧ p₄ < p₅ →
  ¬ position P (3 : Fin (6)) ∧ ¬ position U (3 : Fin (6)) ∧ ¬ position S (3 : Fin (6)) :=
by sorry   -- Proof is omitted

end cannot_finish_third_l112_112061


namespace find_n_solution_l112_112636

theorem find_n_solution (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_solution_l112_112636


namespace increasing_on_interval_l112_112939

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x
noncomputable def f2 (x : ℝ) : ℝ := x * Real.exp 2
noncomputable def f3 (x : ℝ) : ℝ := x^3 - x
noncomputable def f4 (x : ℝ) : ℝ := Real.log x - x

theorem increasing_on_interval (x : ℝ) (h : 0 < x) : 
  f2 (x) = x * Real.exp 2 ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f1 x < f1 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f3 x < f3 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f4 x < f4 y) :=
by sorry

end increasing_on_interval_l112_112939


namespace recommended_daily_serving_l112_112767

theorem recommended_daily_serving (mg_per_pill : ℕ) (pills_per_week : ℕ) (total_mg_week : ℕ) (days_per_week : ℕ) 
  (h1 : mg_per_pill = 50) (h2 : pills_per_week = 28) (h3 : total_mg_week = pills_per_week * mg_per_pill) 
  (h4 : days_per_week = 7) : 
  total_mg_week / days_per_week = 200 :=
by
  sorry

end recommended_daily_serving_l112_112767


namespace sum_first_5_terms_l112_112108

variable {a : ℕ → ℝ}
variable (h : 2 * a 2 = a 1 + 3)

theorem sum_first_5_terms (a : ℕ → ℝ) (h : 2 * a 2 = a 1 + 3) : 
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
sorry

end sum_first_5_terms_l112_112108


namespace remaining_numbers_l112_112703

-- Define the problem statement in Lean 4
theorem remaining_numbers (S S5 S3 : ℝ) (A3 : ℝ) 
  (h1 : S / 8 = 20) 
  (h2 : S5 / 5 = 12) 
  (h3 : S3 = S - S5) 
  (h4 : A3 = 100 / 3) : 
  S3 / A3 = 3 :=
sorry

end remaining_numbers_l112_112703


namespace particle_max_height_and_time_l112_112711

theorem particle_max_height_and_time (t : ℝ) (s : ℝ) 
  (height_eq : s = 180 * t - 18 * t^2) :
  ∃ t₁ : ℝ, ∃ s₁ : ℝ, s₁ = 450 ∧ t₁ = 5 ∧ s = 180 * t₁ - 18 * t₁^2 :=
sorry

end particle_max_height_and_time_l112_112711


namespace reading_homework_is_4_l112_112969

-- Defining the conditions.
variables (R : ℕ)  -- Number of pages of reading homework
variables (M : ℕ)  -- Number of pages of math homework

-- Rachel has 7 pages of math homework.
def math_homework_equals_7 : Prop := M = 7

-- Rachel has 3 more pages of math homework than reading homework.
def math_minus_reads_is_3 : Prop := M = R + 3

-- Prove the number of pages of reading homework is 4.
theorem reading_homework_is_4 (M R : ℕ) 
  (h1 : math_homework_equals_7 M) -- M = 7
  (h2 : math_minus_reads_is_3 M R) -- M = R + 3
  : R = 4 :=
sorry

end reading_homework_is_4_l112_112969


namespace anthony_pencils_l112_112905

theorem anthony_pencils (P : Nat) (h : P + 56 = 65) : P = 9 :=
by
  sorry

end anthony_pencils_l112_112905


namespace probability_sum_7_is_1_over_3_l112_112130

def odd_die : Set ℕ := {1, 3, 5}
def even_die : Set ℕ := {2, 4, 6}

noncomputable def total_outcomes : ℕ := 6 * 6

noncomputable def favorable_outcomes : ℕ := 4 + 4 + 4

noncomputable def probability_sum_7 : ℚ := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_sum_7_is_1_over_3 :
  probability_sum_7 = 1 / 3 :=
by
  sorry

end probability_sum_7_is_1_over_3_l112_112130


namespace circumference_of_tank_B_l112_112339

noncomputable def radius_of_tank (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def volume_of_tank (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem circumference_of_tank_B 
  (h_A : ℝ) (C_A : ℝ) (h_B : ℝ) (volume_ratio : ℝ)
  (hA_pos : 0 < h_A) (CA_pos : 0 < C_A) (hB_pos : 0 < h_B) (vr_pos : 0 < volume_ratio) :
  2 * Real.pi * (radius_of_tank (volume_of_tank (radius_of_tank C_A) h_A / (volume_ratio * Real.pi * h_B))) = 17.7245 :=
by 
  sorry

end circumference_of_tank_B_l112_112339


namespace inequality_of_positive_reals_l112_112046

variable {a b c : ℝ}

theorem inequality_of_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end inequality_of_positive_reals_l112_112046


namespace vivian_mail_in_august_l112_112806

-- Conditions
def april_mail : ℕ := 5
def may_mail : ℕ := 2 * april_mail
def june_mail : ℕ := 2 * may_mail
def july_mail : ℕ := 2 * june_mail

-- Question: Prove that Vivian will send 80 pieces of mail in August.
theorem vivian_mail_in_august : 2 * july_mail = 80 :=
by
  -- Sorry to skip the proof
  sorry

end vivian_mail_in_august_l112_112806


namespace Dongdong_test_score_l112_112825

theorem Dongdong_test_score (a b c : ℕ) (h1 : a + b + c = 280) : a ≥ 94 ∨ b ≥ 94 ∨ c ≥ 94 :=
by
  sorry

end Dongdong_test_score_l112_112825


namespace problem1_problem2_problem3_problem4_l112_112559

open Set

def M : Set ℝ := { x | x > 3 / 2 }
def N : Set ℝ := { x | x < 1 ∨ x > 3 }
def R := {x : ℝ | 1 ≤ x ∧ x ≤ 3 / 2}

theorem problem1 : M = { x | 2 * x - 3 > 0 } := sorry
theorem problem2 : N = { x | (x - 3) * (x - 1) > 0 } := sorry
theorem problem3 : M ∩ N = { x | x > 3 } := sorry
theorem problem4 : (M ∪ N)ᶜ = R := sorry

end problem1_problem2_problem3_problem4_l112_112559


namespace quadratic_has_equal_roots_l112_112977

theorem quadratic_has_equal_roots (b : ℝ) (h : ∃ x : ℝ, b*x^2 + 2*b*x + 4 = 0 ∧ b*x^2 + 2*b*x + 4 = 0) :
  b = 4 :=
sorry

end quadratic_has_equal_roots_l112_112977


namespace find_value_l112_112324

open Classical

variables (a b c : ℝ)

-- Assume a, b, c are roots of the polynomial x^3 - 24x^2 + 50x - 42
def is_root (x : ℝ) : Prop := x^3 - 24*x^2 + 50*x - 42 = 0

-- Vieta's formulas for the given polynomial
axiom h1 : is_root a
axiom h2 : is_root b
axiom h3 : is_root c
axiom h4 : a + b + c = 24
axiom h5 : a * b + b * c + c * a = 50
axiom h6 : a * b * c = 42

-- We want to prove the given expression equals 476/43
theorem find_value : 
  (a/(1/a + b*c) + b/(1/b + c*a) + c/(1/c + a*b) = 476/43) :=
sorry

end find_value_l112_112324


namespace work_completion_in_16_days_l112_112249

theorem work_completion_in_16_days (A B : ℕ) :
  (1 / A + 1 / B = 1 / 40) → (10 * (1 / A + 1 / B) = 1 / 4) →
  (12 * 1 / A = 3 / 4) → A = 16 :=
by
  intros h1 h2 h3
  -- Proof is omitted by "sorry".
  sorry

end work_completion_in_16_days_l112_112249


namespace fraction_to_decimal_l112_112602

theorem fraction_to_decimal :
  (51 / 160 : ℝ) = 0.31875 := 
by
  sorry

end fraction_to_decimal_l112_112602


namespace sphere_volume_l112_112670

theorem sphere_volume (length width : ℝ) (angle_deg : ℝ) (h_length : length = 4) (h_width : width = 3) (h_angle : angle_deg = 60) :
  ∃ (volume : ℝ), volume = (125 / 6) * Real.pi :=
by
  sorry

end sphere_volume_l112_112670


namespace correct_system_of_equations_l112_112352

theorem correct_system_of_equations :
  ∃ (x y : ℕ), 
    x + y = 38 
    ∧ 26 * x + 20 * y = 952 := 
by
  sorry

end correct_system_of_equations_l112_112352


namespace quadratic_expression_value_l112_112800

theorem quadratic_expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : a + b - 1 = 1) : (1 - a - b) = -1 :=
sorry

end quadratic_expression_value_l112_112800


namespace length_fraction_of_radius_l112_112510

noncomputable def side_of_square_area (A : ℕ) : ℕ := Nat.sqrt A
noncomputable def radius_of_circle_from_square_area (A : ℕ) : ℕ := side_of_square_area A

noncomputable def length_of_rectangle_from_area_breadth (A b : ℕ) : ℕ := A / b
noncomputable def fraction_of_radius (len rad : ℕ) : ℚ := len / rad

theorem length_fraction_of_radius 
  (A_square A_rect breadth : ℕ) 
  (h_square_area : A_square = 1296)
  (h_rect_area : A_rect = 360)
  (h_breadth : breadth = 10) : 
  fraction_of_radius 
    (length_of_rectangle_from_area_breadth A_rect breadth)
    (radius_of_circle_from_square_area A_square) = 1 := 
by
  sorry

end length_fraction_of_radius_l112_112510


namespace find_price_per_package_l112_112403

theorem find_price_per_package (P : ℝ) :
  (10 * P + 50 * (4/5 * P) = 1340) → (P = 26.80) := by
  intros h
  sorry

end find_price_per_package_l112_112403


namespace stone_10th_image_l112_112655

-- Definition of the recursive sequence
def stones (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 1 => stones n + 3 * (n + 1) + 1

-- The statement we need to prove
theorem stone_10th_image : stones 9 = 145 := 
  sorry

end stone_10th_image_l112_112655


namespace magnitude_of_vector_l112_112182

open Complex

theorem magnitude_of_vector (z : ℂ) (h : z = 1 - I) : 
  ‖(2 / z + z^2)‖ = Real.sqrt 2 :=
by
  sorry

end magnitude_of_vector_l112_112182


namespace g_x_even_l112_112687

theorem g_x_even (a b c : ℝ) (g : ℝ → ℝ):
  (∀ x, g x = a * x^6 + b * x^4 - c * x^2 + 5)
  → g 32 = 3
  → g 32 + g (-32) = 6 :=
by
  sorry

end g_x_even_l112_112687


namespace triangles_intersection_area_is_zero_l112_112350

-- Define the vertices of the two triangles
def vertex_triangle_1 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (0, 2)
| ⟨1, _⟩ => (2, 1)
| ⟨2, _⟩ => (0, 0)

def vertex_triangle_2 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (2, 2)
| ⟨1, _⟩ => (0, 1)
| ⟨2, _⟩ => (2, 0)

-- The area of the intersection of the two triangles
def area_intersection (v1 v2 : Fin 3 → (ℝ × ℝ)) : ℝ :=
  0

-- The theorem to prove
theorem triangles_intersection_area_is_zero :
  area_intersection vertex_triangle_1 vertex_triangle_2 = 0 :=
by
  -- Proof is omitted here.
  sorry

end triangles_intersection_area_is_zero_l112_112350


namespace fran_ate_15_green_macaroons_l112_112733

variable (total_red total_green initial_remaining green_macaroons_eaten : ℕ)

-- Conditions as definitions
def initial_red_macaroons := 50
def initial_green_macaroons := 40
def total_macaroons := 90
def remaining_macaroons := 45

-- Total eaten macaroons
def total_eaten_macaroons (G : ℕ) := G + 2 * G

-- The proof statement
theorem fran_ate_15_green_macaroons
  (h1 : total_red = initial_red_macaroons)
  (h2 : total_green = initial_green_macaroons)
  (h3 : initial_remaining = remaining_macaroons)
  (h4 : total_macaroons = initial_red_macaroons + initial_green_macaroons)
  (h5 : initial_remaining = total_macaroons - total_eaten_macaroons green_macaroons_eaten):
  green_macaroons_eaten = 15 :=
  by
  sorry

end fran_ate_15_green_macaroons_l112_112733


namespace tan_sum_example_l112_112990

theorem tan_sum_example :
  let t1 := Real.tan (17 * Real.pi / 180)
  let t2 := Real.tan (43 * Real.pi / 180)
  t1 + t2 + Real.sqrt 3 * t1 * t2 = Real.sqrt 3 := sorry

end tan_sum_example_l112_112990


namespace people_at_first_concert_l112_112552

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end people_at_first_concert_l112_112552


namespace cos_sum_of_arctan_roots_l112_112761

theorem cos_sum_of_arctan_roots (α β : ℝ) (hα : -π/2 < α ∧ α < 0) (hβ : -π/2 < β ∧ β < 0) 
  (h1 : Real.tan α + Real.tan β = -3 * Real.sqrt 3) 
  (h2 : Real.tan α * Real.tan β = 4) : 
  Real.cos (α + β) = - 1 / 2 :=
sorry

end cos_sum_of_arctan_roots_l112_112761


namespace prime_ge_5_div_24_l112_112810

theorem prime_ge_5_div_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) : 24 ∣ p^2 - 1 := 
sorry

end prime_ge_5_div_24_l112_112810


namespace max_hours_wednesday_l112_112522

theorem max_hours_wednesday (x : ℕ) 
    (h1 : ∀ (d w : ℕ), w = x → d = x → d + w + (x + 3) = 3 * 3) 
    (h2 : ∀ (a b c : ℕ), a = b → b = c → (a + b + (c + 3))/3 = 3) :
  x = 2 := 
by
  sorry

end max_hours_wednesday_l112_112522


namespace point_coordinates_l112_112873

/-- Given the vector from point A to point B, if point A is the origin, then point B will have coordinates determined by the vector. -/
theorem point_coordinates (A B: ℝ × ℝ) (v: ℝ × ℝ) 
  (h: A = (0, 0)) (h_v: v = (-2, 4)) (h_ab: B = (A.1 + v.1, A.2 + v.2)): 
  B = (-2, 4) :=
by
  sorry

end point_coordinates_l112_112873


namespace four_integers_sum_product_odd_impossible_l112_112820

theorem four_integers_sum_product_odd_impossible (a b c d : ℤ) :
  ¬ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ 
     (a + b + c + d) % 2 = 1) :=
by
  sorry

end four_integers_sum_product_odd_impossible_l112_112820


namespace four_times_sum_of_cubes_gt_cube_sum_l112_112943

theorem four_times_sum_of_cubes_gt_cube_sum
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 :=
by
  sorry

end four_times_sum_of_cubes_gt_cube_sum_l112_112943


namespace point_below_line_range_l112_112882

theorem point_below_line_range (t : ℝ) : (2 * (-2) - 3 * t + 6 > 0) → t < (2 / 3) :=
by {
  sorry
}

end point_below_line_range_l112_112882


namespace final_weight_is_sixteen_l112_112958

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end final_weight_is_sixteen_l112_112958


namespace students_at_start_of_year_l112_112298

theorem students_at_start_of_year (S : ℝ) (h1 : S + 46.0 = 56) : S = 10 :=
sorry

end students_at_start_of_year_l112_112298


namespace sum_of_ages_53_l112_112006

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end sum_of_ages_53_l112_112006


namespace average_income_l112_112726

/-- The daily incomes of the cab driver over 5 days. --/
def incomes : List ℕ := [400, 250, 650, 400, 500]

/-- Prove that the average income of the cab driver over these 5 days is $440. --/
theorem average_income : (incomes.sum / incomes.length) = 440 := by
  sorry

end average_income_l112_112726


namespace kaleb_saved_initial_amount_l112_112023

theorem kaleb_saved_initial_amount (allowance toys toy_price : ℕ) (total_savings : ℕ)
  (h1 : allowance = 15)
  (h2 : toys = 6)
  (h3 : toy_price = 6)
  (h4 : total_savings = toys * toy_price - allowance) :
  total_savings = 21 :=
  sorry

end kaleb_saved_initial_amount_l112_112023


namespace sequence_length_arithmetic_sequence_l112_112538

theorem sequence_length_arithmetic_sequence :
  ∀ (a d l n : ℕ), a = 5 → d = 3 → l = 119 → l = a + (n - 1) * d → n = 39 :=
by
  intros a d l n ha hd hl hln
  sorry

end sequence_length_arithmetic_sequence_l112_112538


namespace compute_abs_a_plus_b_plus_c_l112_112201

variable (a b c : ℝ)

theorem compute_abs_a_plus_b_plus_c (h1 : a^2 - b * c = 14)
                                   (h2 : b^2 - c * a = 14)
                                   (h3 : c^2 - a * b = -3) :
                                   |a + b + c| = 5 :=
sorry

end compute_abs_a_plus_b_plus_c_l112_112201


namespace molecular_weight_3_moles_l112_112037

theorem molecular_weight_3_moles
  (C_weight : ℝ)
  (H_weight : ℝ)
  (N_weight : ℝ)
  (O_weight : ℝ)
  (Molecular_formula : ℕ → ℕ → ℕ → ℕ → Prop)
  (molecular_weight : ℝ)
  (moles : ℝ) :
  C_weight = 12.01 →
  H_weight = 1.008 →
  N_weight = 14.01 →
  O_weight = 16.00 →
  Molecular_formula 13 9 5 7 →
  molecular_weight = 156.13 + 9.072 + 70.05 + 112.00 →
  moles = 3 →
  3 * molecular_weight = 1041.756 :=
by
  sorry

end molecular_weight_3_moles_l112_112037


namespace intersection_of_A_and_B_l112_112514

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_of_A_and_B : (A ∩ B) = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l112_112514


namespace W_k_two_lower_bound_l112_112887

-- Define W(k, 2)
def W (k : ℕ) (c : ℕ) : ℕ := -- smallest number such that for every n >= W(k, 2), 
  -- any 2-coloring of the set {1, 2, ..., n} contains a monochromatic arithmetic progression of length k
  sorry 

-- Define the statement to prove
theorem W_k_two_lower_bound (k : ℕ) : ∃ C > 0, W k 2 ≥ C * 2^(k / 2) :=
by
  sorry

end W_k_two_lower_bound_l112_112887


namespace matthew_total_time_l112_112215

def assemble_time : ℝ := 1
def bake_time_normal : ℝ := 1.5
def decorate_time : ℝ := 1
def bake_time_double : ℝ := bake_time_normal * 2

theorem matthew_total_time :
  assemble_time + bake_time_double + decorate_time = 5 := 
by 
  -- The proof will be filled in here
  sorry

end matthew_total_time_l112_112215


namespace measure_ADC_l112_112159

-- Definitions
def angle_measures (x y ADC : ℝ) : Prop :=
  2 * x + 60 + 2 * y = 180 ∧ x + y = 60 ∧ x + y + ADC = 180

-- Goal
theorem measure_ADC (x y ADC : ℝ) (h : angle_measures x y ADC) : ADC = 120 :=
by {
  -- Solution could go here, skipped for brevity
  sorry
}

end measure_ADC_l112_112159


namespace deductive_reasoning_not_always_correct_l112_112123

theorem deductive_reasoning_not_always_correct (P: Prop) (Q: Prop) 
    (h1: (P → Q) → (P → Q)) :
    (¬ (∀ P Q : Prop, (P → Q) → Q → Q)) :=
sorry

end deductive_reasoning_not_always_correct_l112_112123


namespace real_number_representation_l112_112332

theorem real_number_representation (x : ℝ) 
  (h₀ : 0 < x) (h₁ : x ≤ 1) :
  ∃ (n : ℕ → ℕ), (∀ k, n k > 0) ∧ (∀ k, n (k + 1) = n k * 2 ∨ n (k + 1) = n k * 3 ∨ n (k + 1) = n k * 4) ∧ 
  (x = ∑' k, 1 / (n k)) :=
sorry

end real_number_representation_l112_112332


namespace restaurant_total_dishes_l112_112777

noncomputable def total_couscous_received : ℝ := 15.4 + 45
noncomputable def total_chickpeas_received : ℝ := 19.8 + 33

-- Week 1, ratio of 5:3 (couscous:chickpeas)
noncomputable def sets_of_ratio_week1_couscous : ℝ := total_couscous_received / 5
noncomputable def sets_of_ratio_week1_chickpeas : ℝ := total_chickpeas_received / 3
noncomputable def dishes_week1 : ℝ := min sets_of_ratio_week1_couscous sets_of_ratio_week1_chickpeas

-- Week 2, ratio of 3:2 (couscous:chickpeas)
noncomputable def sets_of_ratio_week2_couscous : ℝ := total_couscous_received / 3
noncomputable def sets_of_ratio_week2_chickpeas : ℝ := total_chickpeas_received / 2
noncomputable def dishes_week2 : ℝ := min sets_of_ratio_week2_couscous sets_of_ratio_week2_chickpeas

-- Total dishes rounded down
noncomputable def total_dishes : ℝ := dishes_week1 + dishes_week2

theorem restaurant_total_dishes :
  ⌊total_dishes⌋ = 32 :=
by {
  sorry
}

end restaurant_total_dishes_l112_112777


namespace quadratic_properties_l112_112507

open Real

noncomputable section

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Vertex form of the quadratic
def vertexForm (x : ℝ) : ℝ := (x - 2)^2 - 1

-- Axis of symmetry
def axisOfSymmetry : ℝ := 2

-- Vertex of the quadratic
def vertex : ℝ × ℝ := (2, -1)

-- Minimum value of the quadratic
def minimumValue : ℝ := -1

-- Interval where the function decreases
def decreasingInterval (x : ℝ) : Prop := -1 ≤ x ∧ x < 2

-- Range of y in the interval -1 <= x < 3
def rangeOfY (y : ℝ) : Prop := -1 ≤ y ∧ y ≤ 8

-- Main statement
theorem quadratic_properties :
  (∀ x, quadratic x = vertexForm x) ∧
  (∃ x, axisOfSymmetry = x) ∧
  (∃ v, vertex = v) ∧
  (minimumValue = -1) ∧
  (∀ x, -1 ≤ x ∧ x < 2 → quadratic x > quadratic (x + 1)) ∧
  (∀ y, (∃ x, -1 ≤ x ∧ x < 3 ∧ y = quadratic x) → rangeOfY y) :=
sorry

end quadratic_properties_l112_112507


namespace min_val_of_a2_plus_b2_l112_112220

variable (a b : ℝ)

def condition := 3 * a - 4 * b - 2 = 0

theorem min_val_of_a2_plus_b2 : condition a b → (∃ a b : ℝ, a^2 + b^2 = 4 / 25) := by 
  sorry

end min_val_of_a2_plus_b2_l112_112220


namespace max_value_expr_l112_112432

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end max_value_expr_l112_112432


namespace polygon_sides_from_diagonals_l112_112582

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l112_112582


namespace minimum_value_of_expression_l112_112188

theorem minimum_value_of_expression {x y : ℝ} (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ m : ℝ, m = 0.75 ∧ ∀ z : ℝ, (∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y = 1 ∧ z = 2 * x + 3 * y ^ 2) → z ≥ m :=
sorry

end minimum_value_of_expression_l112_112188


namespace age_sum_l112_112620

variable {S R K : ℝ}

theorem age_sum 
  (h1 : S = R + 10)
  (h2 : S + 12 = 3 * (R - 5))
  (h3 : K = R / 2) :
  S + R + K = 56.25 := 
by 
  sorry

end age_sum_l112_112620


namespace kite_area_is_192_l112_112270

-- Define the points with doubled dimensions
def A : (ℝ × ℝ) := (0, 16)
def B : (ℝ × ℝ) := (8, 24)
def C : (ℝ × ℝ) := (16, 16)
def D : (ℝ × ℝ) := (8, 0)

-- Calculate the area of the kite
noncomputable def kiteArea (A B C D : ℝ × ℝ) : ℝ :=
  let baseUpper := abs (C.1 - A.1)
  let heightUpper := abs (B.2 - A.2)
  let areaUpper := 1 / 2 * baseUpper * heightUpper
  let baseLower := baseUpper
  let heightLower := abs (B.2 - D.2)
  let areaLower := 1 / 2 * baseLower * heightLower
  areaUpper + areaLower

-- State the theorem to prove the kite area is 192 square inches
theorem kite_area_is_192 : kiteArea A B C D = 192 := 
  sorry

end kite_area_is_192_l112_112270


namespace sum_of_three_largest_l112_112658

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l112_112658


namespace hyperbola_vertex_distance_l112_112074

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l112_112074


namespace negation_of_exists_geq_prop_l112_112002

open Classical

variable (P : Prop) (Q : Prop)

-- Original proposition:
def exists_geq_prop : Prop := 
  ∃ x : ℝ, x^2 + x + 1 ≥ 0

-- Its negation:
def forall_lt_neg : Prop :=
  ∀ x : ℝ, x^2 + x + 1 < 0

-- The theorem to prove:
theorem negation_of_exists_geq_prop : ¬ exists_geq_prop ↔ forall_lt_neg := 
by 
  -- The proof steps will be filled in here
  sorry

end negation_of_exists_geq_prop_l112_112002


namespace probability_x_lt_2y_l112_112487

noncomputable def probability_x_lt_2y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1/2) * 4 * 2
  let area_rectangle : ℚ := 4 * 2
  (area_triangle / area_rectangle)

theorem probability_x_lt_2y (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 2) :
  probability_x_lt_2y_in_rectangle = 1/2 := by
  sorry

end probability_x_lt_2y_l112_112487


namespace range_of_a_l112_112776

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 - x ^ 2 + x - 5

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ x_max x_min : ℝ, x_max ≠ x_min ∧
  f a x_max = max (f a x_max) (f a x_min) ∧ f a x_min = min (f a x_max) (f a x_min)) → 
  a < 1 / 3 ∧ a ≠ 0 := sorry

end range_of_a_l112_112776


namespace probability_of_selecting_letter_a_l112_112294

def total_ways := Nat.choose 5 2
def ways_to_select_a := 4
def probability_of_selecting_a := (ways_to_select_a : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_letter_a :
  probability_of_selecting_a = 2 / 5 :=
by
  -- proof steps will be filled in here
  sorry

end probability_of_selecting_letter_a_l112_112294


namespace intersection_A_B_l112_112458

def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_B_l112_112458


namespace proof_problem_l112_112286

theorem proof_problem (α : ℝ) (h1 : 0 < α ∧ α < π)
    (h2 : Real.sin α + Real.cos α = 1 / 5) :
    (Real.tan α = -4 / 3) ∧ 
    ((Real.sin (3 * Real.pi / 2 + α) * Real.sin (Real.pi / 2 - α) * (Real.tan (Real.pi - α))^3) / 
    (Real.cos (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)) = -4 / 3) :=
by
  sorry

end proof_problem_l112_112286


namespace all_points_same_value_l112_112217

theorem all_points_same_value {f : ℤ × ℤ → ℕ}
  (h : ∀ x y : ℤ, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ k : ℕ, ∀ x y : ℤ, f (x, y) = k :=
sorry

end all_points_same_value_l112_112217


namespace hexagon_sum_balanced_assignment_exists_l112_112320

-- Definitions based on the conditions
def is_valid_assignment (a b c d e f g : ℕ) : Prop :=
a + b + g = a + c + g ∧ a + b + g = a + d + g ∧ a + b + g = a + e + g ∧
a + b + g = b + c + g ∧ a + b + g = b + d + g ∧ a + b + g = b + e + g ∧
a + b + g = c + d + g ∧ a + b + g = c + e + g ∧ a + b + g = d + e + g

-- The theorem we want to prove
theorem hexagon_sum_balanced_assignment_exists :
  ∃ (a b c d e f g : ℕ), 
  (a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 2 ∨ b = 3 ∨ b = 5) ∧ 
  (c = 2 ∨ c = 3 ∨ c = 5) ∧ 
  (d = 2 ∨ d = 3 ∨ d = 5) ∧ 
  (e = 2 ∨ e = 3 ∨ e = 5) ∧
  (f = 2 ∨ f = 3 ∨ f = 5) ∧
  (g = 2 ∨ g = 3 ∨ g = 5) ∧
  is_valid_assignment a b c d e f g :=
sorry

end hexagon_sum_balanced_assignment_exists_l112_112320


namespace c_work_rate_l112_112124

noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem c_work_rate (A B C: ℝ) 
  (h1 : A + B = work_rate 28) 
  (h2 : A + B + C = work_rate 21) : C = work_rate 84 := by
  -- Proof will go here
  sorry

end c_work_rate_l112_112124


namespace problem_translation_l112_112872

variables {a : ℕ → ℤ} (S : ℕ → ℤ)

-- Definition of the arithmetic sequence and its sum function
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ (d : ℤ), ∀ (n m : ℕ), a (n + 1) = a n + d

-- Sum of the first n terms defined recursively
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 0 then 0 else a n + sum_first_n_terms a (n - 1)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : S 5 > S 6

-- To be proved: Option D does not necessarily hold
theorem problem_translation : ¬(a 3 + a 6 + a 12 < 2 * a 7) := sorry

end problem_translation_l112_112872


namespace find_max_sum_pair_l112_112856

theorem find_max_sum_pair :
  ∃ a b : ℕ, 2 * a * b + 3 * b = b^2 + 6 * a + 6 ∧ (∀ a' b' : ℕ, 2 * a' * b' + 3 * b' = b'^2 + 6 * a' + 6 → a + b ≥ a' + b') ∧ a = 5 ∧ b = 9 :=
by {
  sorry
}

end find_max_sum_pair_l112_112856


namespace barium_atoms_in_compound_l112_112058

noncomputable def barium_atoms (total_molecular_weight : ℝ) (weight_ba_per_atom : ℝ) (weight_br_per_atom : ℝ) (num_br_atoms : ℕ) : ℝ :=
  (total_molecular_weight - (num_br_atoms * weight_br_per_atom)) / weight_ba_per_atom

theorem barium_atoms_in_compound :
  barium_atoms 297 137.33 79.90 2 = 1 :=
by
  unfold barium_atoms
  norm_num
  sorry

end barium_atoms_in_compound_l112_112058


namespace circle_equation_l112_112536

theorem circle_equation (x y : ℝ) :
  let C := (4, -6)
  let r := 4
  (x - C.1)^2 + (y - C.2)^2 = r^2 →
  (x - 4)^2 + (y + 6)^2 = 16 :=
by
  intros
  sorry

end circle_equation_l112_112536


namespace even_function_periodicity_l112_112005

noncomputable def f : ℝ → ℝ :=
sorry -- The actual function definition is not provided here but assumed to exist.

theorem even_function_periodicity (x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 2)
  (h2 : f (x + 2) = f x)
  (hf_even : ∀ x, f x = f (-x))
  (hf_segment : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x^2 + 2*x - 1) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2 - 6*x + 7 :=
sorry

end even_function_periodicity_l112_112005


namespace range_of_a_for_quadratic_inequality_l112_112396

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ (x : ℝ), 1 ≤ x ∧ x < 5 → x^2 - (a + 1)*x + a ≤ 0) ↔ (4 ≤ a ∧ a < 5) :=
sorry

end range_of_a_for_quadratic_inequality_l112_112396


namespace fish_tagging_problem_l112_112572

theorem fish_tagging_problem
  (N : ℕ) (T : ℕ)
  (h1 : N = 1250)
  (h2 : T = N / 25) :
  T = 50 :=
sorry

end fish_tagging_problem_l112_112572


namespace valid_assignments_count_l112_112143

noncomputable def validAssignments : Nat := sorry

theorem valid_assignments_count : validAssignments = 4 := 
by {
  sorry
}

end valid_assignments_count_l112_112143


namespace min_calls_correct_l112_112994

-- Define a function that calculates the minimum number of calls given n people
def min_calls (n : ℕ) : ℕ :=
  2 * n - 2

-- Theorem to prove that min_calls(n) given the conditions is equal to 2n - 2
theorem min_calls_correct (n : ℕ) (h : n ≥ 2) : min_calls n = 2 * n - 2 :=
by
  sorry

end min_calls_correct_l112_112994


namespace total_pencils_correct_l112_112316
  
def original_pencils : ℕ := 2
def added_pencils : ℕ := 3
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 5 := 
by
  -- proof state will be filled here 
  sorry

end total_pencils_correct_l112_112316


namespace manager_salary_l112_112880

theorem manager_salary (avg_salary_employees : ℝ) (num_employees : ℕ) (salary_increase : ℝ) (manager_salary : ℝ) :
  avg_salary_employees = 1500 →
  num_employees = 24 →
  salary_increase = 400 →
  (num_employees + 1) * (avg_salary_employees + salary_increase) - num_employees * avg_salary_employees = manager_salary →
  manager_salary = 11500 := 
by
  intros h_avg_salary_employees h_num_employees h_salary_increase h_computation
  sorry

end manager_salary_l112_112880


namespace flower_bed_profit_l112_112115

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end flower_bed_profit_l112_112115


namespace sandra_total_beignets_l112_112527

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l112_112527


namespace not_perfect_cube_of_N_l112_112592

-- Define a twelve-digit number
def N : ℕ := 100000000000

-- Define the condition that a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℤ, n = k ^ 3

-- Problem statement: Prove that 100000000000 is not a perfect cube
theorem not_perfect_cube_of_N : ¬ is_perfect_cube N :=
by sorry

end not_perfect_cube_of_N_l112_112592


namespace glass_ball_radius_l112_112787

theorem glass_ball_radius (x y r : ℝ) (h_parabola : x^2 = 2 * y) (h_touch : y = r) (h_range : 0 ≤ y ∧ y ≤ 20) : 0 < r ∧ r ≤ 1 :=
sorry

end glass_ball_radius_l112_112787


namespace more_candidates_selected_l112_112361

theorem more_candidates_selected (total_a total_b selected_a selected_b : ℕ)
  (h1 : total_a = 8000)
  (h2 : total_b = 8000)
  (h3 : selected_a = 6 * total_a / 100)
  (h4 : selected_b = 7 * total_b / 100) :
  selected_b - selected_a = 80 :=
  sorry

end more_candidates_selected_l112_112361


namespace probability_at_least_two_green_l112_112799

def total_apples := 10
def red_apples := 6
def green_apples := 4
def choose_apples := 3

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_two_green :
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) = 40 ∧ 
  binomial total_apples choose_apples = 120 ∧
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) / binomial total_apples choose_apples = 1 / 3 := by
sorry

end probability_at_least_two_green_l112_112799


namespace coat_price_reduction_l112_112965

theorem coat_price_reduction :
  let orig_price := 500
  let first_discount := 0.15 * orig_price
  let price_after_first := orig_price - first_discount
  let second_discount := 0.10 * price_after_first
  let price_after_second := price_after_first - second_discount
  let tax := 0.07 * price_after_second
  let price_with_tax := price_after_second + tax
  let final_price := price_with_tax - 200
  let reduction_amount := orig_price - final_price
  let percent_reduction := (reduction_amount / orig_price) * 100
  percent_reduction = 58.145 :=
by
  sorry

end coat_price_reduction_l112_112965


namespace salary_increase_l112_112475

theorem salary_increase (x : ℝ) 
  (h : ∀ s : ℕ, 1 ≤ s ∧ s ≤ 5 → ∃ p : ℝ, p = 7.50 + x * (s - 1))
  (h₁ : ∃ p₁ p₅ : ℝ, 1 ≤ 1 ∧ 5 ≤ 5 ∧ p₅ = p₁ + 1.25) :
  x = 0.3125 := sorry

end salary_increase_l112_112475


namespace num_new_students_l112_112517

theorem num_new_students 
  (original_avg_age : ℕ) 
  (original_num_students : ℕ) 
  (new_avg_age : ℕ) 
  (age_decrease : ℕ) 
  (total_age_orginal : ℕ := original_num_students * original_avg_age) 
  (total_new_students : ℕ := (original_avg_age - age_decrease) * (original_num_students + 12))
  (x : ℕ := total_new_students - total_age_orginal) :
  original_avg_age = 40 → 
  original_num_students = 12 →
  new_avg_age = 32 →
  age_decrease = 4 →
  x = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end num_new_students_l112_112517


namespace max_radius_of_inner_spheres_l112_112650

theorem max_radius_of_inner_spheres (R : ℝ) : 
  ∃ r : ℝ, (2 * r ≤ R) ∧ (r ≤ (4 * Real.sqrt 2 - 1) / 4 * R) :=
sorry

end max_radius_of_inner_spheres_l112_112650


namespace num_people_in_group_l112_112933

-- Given conditions as definitions
def cost_per_adult_meal : ℤ := 3
def num_kids : ℤ := 7
def total_cost : ℤ := 15

-- Statement to prove
theorem num_people_in_group : 
  ∃ (num_adults : ℤ), 
    total_cost = num_adults * cost_per_adult_meal ∧ 
    (num_adults + num_kids) = 12 :=
by
  sorry

end num_people_in_group_l112_112933


namespace men_wages_l112_112945

theorem men_wages (W : ℕ) (wage : ℕ) :
  (5 + W + 8) * wage = 75 ∧ 5 * wage = W * wage ∧ W * wage = 8 * wage → 
  wage = 5 := 
by
  sorry

end men_wages_l112_112945


namespace integer_values_b_l112_112865

theorem integer_values_b (b : ℤ) : 
  (∃ (x1 x2 : ℤ), x1 + x2 = -b ∧ x1 * x2 = 7 * b) ↔ b = 0 ∨ b = 36 ∨ b = -28 ∨ b = -64 :=
by
  sorry

end integer_values_b_l112_112865


namespace intersection_of_A_and_B_l112_112362

-- Define the sets A and B
def setA : Set ℝ := { x | -1 < x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

-- The intersection of sets A and B
def intersectAB : Set ℝ := { x | 2 < x ∧ x ≤ 4 }

-- The theorem statement to be proved
theorem intersection_of_A_and_B : ∀ x, x ∈ setA ∩ setB ↔ x ∈ intersectAB := by
  sorry

end intersection_of_A_and_B_l112_112362


namespace line_intersects_y_axis_at_point_l112_112196

def line_intersects_y_axis (x1 y1 x2 y2 : ℚ) : Prop :=
  ∃ c : ℚ, ∀ x : ℚ, y1 + (y2 - y1) / (x2 - x1) * (x - x1) = (y2 - y1) / (x2 - x1) * x + c

theorem line_intersects_y_axis_at_point :
  line_intersects_y_axis 3 21 (-9) (-6) :=
  sorry

end line_intersects_y_axis_at_point_l112_112196


namespace algebra_expression_bound_l112_112563

theorem algebra_expression_bound (x y m : ℝ) 
  (h1 : x + y + m = 6) 
  (h2 : 3 * x - y + m = 4) : 
  (-2 * x * y + 1) ≤ 3 / 2 := 
by 
  sorry

end algebra_expression_bound_l112_112563


namespace number_of_true_statements_l112_112079

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem number_of_true_statements (n : ℕ) :
  let s1 := reciprocal 4 + reciprocal 8 ≠ reciprocal 12
  let s2 := reciprocal 9 - reciprocal 3 ≠ reciprocal 6
  let s3 := reciprocal 5 * reciprocal 10 = reciprocal 50
  let s4 := reciprocal 16 / reciprocal 4 = reciprocal 4
  (cond s1 1 0) + (cond s2 1 0) + (cond s3 1 0) + (cond s4 1 0) = 2 := by
  sorry

end number_of_true_statements_l112_112079


namespace image_relative_velocity_l112_112328

-- Definitions of the constants
def f : ℝ := 0.2
def x : ℝ := 0.5
def vt : ℝ := 3

-- Lens equation
def lens_equation (f x y : ℝ) : Prop :=
  (1 / x) + (1 / y) = 1 / f

-- Image distance
noncomputable def y (f x : ℝ) : ℝ :=
  1 / (1 / f - 1 / x)

-- Derivative of y with respect to x
noncomputable def dy_dx (f x : ℝ) : ℝ :=
  (f^2) / (x - f)^2

-- Image velocity
noncomputable def vk (vt dy_dx : ℝ) : ℝ :=
  vt * dy_dx

-- Relative velocity
noncomputable def v_rel (vt vk : ℝ) : ℝ :=
  vk - vt

-- Theorem to prove the relative velocity
theorem image_relative_velocity : v_rel vt (vk vt (dy_dx f x)) = -5 / 3 := 
by
  sorry

end image_relative_velocity_l112_112328


namespace rectangle_area_l112_112974

def length : ℝ := 15
def width : ℝ := 0.9 * length
def area : ℝ := length * width

theorem rectangle_area : area = 202.5 := by
  sorry

end rectangle_area_l112_112974


namespace speed_of_stream_l112_112104

theorem speed_of_stream (v : ℝ) (h1 : 22 > 0) (h2 : 8 > 0) (h3 : 216 = (22 + v) * 8) : v = 5 := 
by 
  sorry

end speed_of_stream_l112_112104


namespace triangle_side_length_l112_112709

theorem triangle_side_length (a b c : ℝ) (A : ℝ) 
  (h_a : a = 2) (h_c : c = 2) (h_A : A = 30) :
  b = 2 * Real.sqrt 3 :=
by
  sorry

end triangle_side_length_l112_112709


namespace shorter_leg_length_l112_112627

theorem shorter_leg_length (a b c : ℝ) (h1 : b = 10) (h2 : a^2 + b^2 = c^2) (h3 : c = 2 * a) : 
  a = 10 * Real.sqrt 3 / 3 :=
by
  sorry

end shorter_leg_length_l112_112627


namespace river_trip_longer_than_lake_trip_l112_112329

theorem river_trip_longer_than_lake_trip (v w : ℝ) (h1 : v > w) : 
  (20 * v) / (v^2 - w^2) > 20 / v :=
by {
  sorry
}

end river_trip_longer_than_lake_trip_l112_112329


namespace simplify_fraction_l112_112144

theorem simplify_fraction :
  (175 / 1225) * 25 = 25 / 7 :=
by
  -- Code to indicate proof steps would go here.
  sorry

end simplify_fraction_l112_112144


namespace balloon_arrangements_l112_112878

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l112_112878


namespace satellite_modular_units_24_l112_112853

-- Define basic parameters
variables (U N S : ℕ)
def fraction_upgraded : ℝ := 0.2

-- Define the conditions as Lean premises
axiom non_upgraded_per_unit_eq_sixth_total_upgraded : N = S / 6
axiom fraction_sensors_upgraded : (S : ℝ) = fraction_upgraded * (S + U * N)

-- The main statement to be proved
theorem satellite_modular_units_24 (h1 : N = S / 6) (h2 : (S : ℝ) = fraction_upgraded * (S + U * N)) : U = 24 :=
by
  -- The actual proof steps will be written here.
  sorry

end satellite_modular_units_24_l112_112853


namespace trigonometric_identity_l112_112590

theorem trigonometric_identity
  (x : ℝ) 
  (h_tan : Real.tan x = -1/2) :
  (3 * Real.sin x ^ 2 - 2) / (Real.sin x * Real.cos x) = 7 / 2 := 
by
  sorry

end trigonometric_identity_l112_112590


namespace range_of_k_l112_112892

theorem range_of_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ k ≤ 1 :=
by sorry

end range_of_k_l112_112892


namespace maximize_exponential_sum_l112_112073

theorem maximize_exponential_sum (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) : 
  e^a + e^b + e^c + e^d ≤ 4 * Real.exp 1 := 
sorry

end maximize_exponential_sum_l112_112073


namespace intersection_M_N_l112_112482

-- Definition of the sets M and N
def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6 * x < 0}

-- Intersection of M and N
def intersection : Set ℝ := {x | 4 < x ∧ x < 6}

-- Theorem statement asserting the equality between the intersection and the desired set
theorem intersection_M_N : ∀ (x : ℝ), x ∈ M ∩ N ↔ x ∈ intersection := by
  sorry

end intersection_M_N_l112_112482


namespace library_books_new_releases_l112_112216

theorem library_books_new_releases (P Q R S : Prop) 
  (h : ¬P) 
  (P_iff_Q : P ↔ Q)
  (Q_implies_R : Q → R)
  (S_iff_notP : S ↔ ¬P) : 
  Q ∧ S := by 
  sorry

end library_books_new_releases_l112_112216


namespace alyssa_gave_away_puppies_l112_112022

def start_puppies : ℕ := 12
def remaining_puppies : ℕ := 5

theorem alyssa_gave_away_puppies : 
  start_puppies - remaining_puppies = 7 := 
by
  sorry

end alyssa_gave_away_puppies_l112_112022


namespace discounted_price_of_russian_doll_l112_112792

theorem discounted_price_of_russian_doll (original_price : ℕ) (number_of_dolls_original : ℕ) (number_of_dolls_discounted : ℕ) (discounted_price : ℕ) :
  original_price = 4 →
  number_of_dolls_original = 15 →
  number_of_dolls_discounted = 20 →
  (number_of_dolls_original * original_price) = 60 →
  (number_of_dolls_discounted * discounted_price) = 60 →
  discounted_price = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end discounted_price_of_russian_doll_l112_112792


namespace number_zero_points_eq_three_l112_112219

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - x^2

theorem number_zero_points_eq_three : ∃ x1 x2 x3 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (∀ y : ℝ, f y = 0 → (y = x1 ∨ y = x2 ∨ y = x3)) :=
sorry

end number_zero_points_eq_three_l112_112219


namespace David_is_8_years_older_than_Scott_l112_112834

noncomputable def DavidAge : ℕ := 14 -- Since David was 8 years old, 6 years ago
noncomputable def RichardAge : ℕ := DavidAge + 6
noncomputable def ScottAge : ℕ := (RichardAge + 8) / 2 - 8
noncomputable def AgeDifference : ℕ := DavidAge - ScottAge

theorem David_is_8_years_older_than_Scott :
  AgeDifference = 8 :=
by
  sorry

end David_is_8_years_older_than_Scott_l112_112834


namespace sin_30_is_half_l112_112713

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l112_112713


namespace ababab_divisible_by_13_l112_112235

theorem ababab_divisible_by_13 (a b : ℕ) (ha: a < 10) (hb: b < 10) : 
  13 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) := 
by
  sorry

end ababab_divisible_by_13_l112_112235


namespace monotonic_increasing_condition_l112_112237

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → y a x₁ ≤ y a x₂) ↔ 
  (a = 0 ∨ a > 0) :=
sorry

end monotonic_increasing_condition_l112_112237


namespace days_spent_on_Orbius5_l112_112228

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end days_spent_on_Orbius5_l112_112228


namespace line_through_P_with_opposite_sign_intercepts_l112_112474

theorem line_through_P_with_opposite_sign_intercepts 
  (P : ℝ × ℝ) (hP : P = (3, -2)) 
  (h : ∀ (A B : ℝ), A ≠ 0 → B ≠ 0 → A * B < 0) : 
  (∀ (x y : ℝ), (x = 5 ∧ y = -5) → (5 * x - 5 * y - 25 = 0)) ∨ (∀ (x y : ℝ), (3 * y = -2) → (y = - (2 / 3) * x)) :=
sorry

end line_through_P_with_opposite_sign_intercepts_l112_112474


namespace sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l112_112972

-- Definitions based on conditions
def standard_flavor_to_water_ratio := 1 / 30
def standard_flavor_to_corn_syrup_ratio := 1 / 12
def sport_water_amount := 60
def sport_corn_syrup_amount := 4
def sport_flavor_to_water_ratio := 1 / 60
def sport_flavor_amount := 1 -- derived from sport_water_amount * sport_flavor_to_water_ratio

-- The main theorem to prove
theorem sport_flavoring_to_corn_syrup_ratio_is_three_times_standard :
  1 / 4 = 3 * (1 / 12) :=
by
  sorry

end sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l112_112972


namespace toy_swords_count_l112_112802

variable (s : ℕ)

def cost_lego := 250
def cost_toy_sword := 120
def cost_play_dough := 35

def total_cost (s : ℕ) :=
  3 * cost_lego + s * cost_toy_sword + 10 * cost_play_dough

theorem toy_swords_count : total_cost s = 1940 → s = 7 := by
  sorry

end toy_swords_count_l112_112802


namespace distinct_four_digit_integers_with_digit_product_eight_l112_112444

theorem distinct_four_digit_integers_with_digit_product_eight : 
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ (a b c d : ℕ), 10 > a ∧ 10 > b ∧ 10 > c ∧ 10 > d ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 8) ∧ (∃ (count : ℕ), count = 20 ) :=
sorry

end distinct_four_digit_integers_with_digit_product_eight_l112_112444


namespace option_C_l112_112712

theorem option_C (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > 0) :
  (b + c) / (a + c) > b / a :=
sorry

end option_C_l112_112712


namespace age_of_new_person_l112_112381

theorem age_of_new_person (T : ℝ) (A : ℝ) (h : T / 20 - 4 = (T - 60 + A) / 20) : A = 40 :=
sorry

end age_of_new_person_l112_112381


namespace largest_circle_area_in_region_S_l112_112657

-- Define the region S
def region_S (x y : ℝ) : Prop :=
  |x + (1 / 2) * y| ≤ 10 ∧ |x| ≤ 10 ∧ |y| ≤ 10

-- The question is to determine the value of k such that the area of the largest circle 
-- centered at (0, 0) fitting inside region S is k * π.
theorem largest_circle_area_in_region_S :
  ∃ k : ℝ, k = 80 :=
sorry

end largest_circle_area_in_region_S_l112_112657


namespace proof_problem_l112_112404

-- Conditions
def in_fourth_quadrant (α : ℝ) : Prop := (α > 3 * Real.pi / 2) ∧ (α < 2 * Real.pi)
def x_coordinate_unit_circle (α : ℝ) : Prop := Real.cos α = 1/3

-- Proof statement
theorem proof_problem (α : ℝ) (h1 : in_fourth_quadrant α) (h2 : x_coordinate_unit_circle α) :
  Real.tan α = -2 * Real.sqrt 2 ∧
  ((Real.sin α)^2 - Real.sqrt 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.cos α)^2) = 6 / 5 :=
by
  sorry

end proof_problem_l112_112404


namespace find_m_l112_112901

open Set

def U : Set ℕ := {0, 1, 2, 3}
def A (m : ℤ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}
def complement_A (m : ℤ) : Set ℕ := {1, 2}

theorem find_m (m : ℤ) (hA : complement_A m = U \ A m) : m = -3 :=
by
  sorry

end find_m_l112_112901


namespace geom_sequence_common_ratio_l112_112649

variable {α : Type*} [LinearOrderedField α]

theorem geom_sequence_common_ratio (a1 q : α) (h : a1 > 0) (h_eq : a1 + a1 * q + a1 * q^2 + a1 * q = 9 * a1 * q^2) : q = 1 / 2 :=
by sorry

end geom_sequence_common_ratio_l112_112649


namespace exists_segment_with_points_l112_112430

theorem exists_segment_with_points (S : Finset ℕ) (n : ℕ) (hS : S.card = 6 * n)
  (hB : ∃ B : Finset ℕ, B ⊆ S ∧ B.card = 4 * n) (hG : ∃ G : Finset ℕ, G ⊆ S ∧ G.card = 2 * n) :
  ∃ t : Finset ℕ, t ⊆ S ∧ t.card = 3 * n ∧ (∃ B' : Finset ℕ, B' ⊆ t ∧ B'.card = 2 * n) ∧ (∃ G' : Finset ℕ, G' ⊆ t ∧ G'.card = n) :=
  sorry

end exists_segment_with_points_l112_112430


namespace smallest_positive_n_l112_112886

theorem smallest_positive_n : ∃ n : ℕ, 3 * n ≡ 8 [MOD 26] ∧ n = 20 :=
by 
  use 20
  simp
  sorry

end smallest_positive_n_l112_112886


namespace part1_part2_l112_112152

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m
def h (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem part1 (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 :=
by
  sorry

theorem part2 (m : ℝ) : (∃ x : ℝ, f x < g x m) ↔ m > 4 :=
by
  sorry

end part1_part2_l112_112152


namespace has_only_one_zero_point_l112_112110

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + (a / 2) * x^2

theorem has_only_one_zero_point (a : ℝ) (h : -Real.exp 1 ≤ a ∧ a ≤ 0) :
  ∃! x : ℝ, f x a = 0 :=
sorry

end has_only_one_zero_point_l112_112110


namespace tetrahedron_volume_l112_112265

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l112_112265


namespace evaluate_expression_l112_112383

theorem evaluate_expression :
  2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 :=
by
  sorry

end evaluate_expression_l112_112383


namespace algebra_expression_never_zero_l112_112961

theorem algebra_expression_never_zero (x : ℝ) : (1 : ℝ) / (x - 1) ≠ 0 :=
sorry

end algebra_expression_never_zero_l112_112961


namespace train_speed_l112_112334

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l112_112334


namespace sum_of_letters_l112_112064

def A : ℕ := 0
def B : ℕ := 1
def C : ℕ := 2
def M : ℕ := 12

theorem sum_of_letters :
  A + B + M + C = 15 :=
by
  sorry

end sum_of_letters_l112_112064


namespace sum_of_consecutive_integers_with_product_506_l112_112364

theorem sum_of_consecutive_integers_with_product_506 :
  ∃ x : ℕ, (x * (x + 1) = 506) → (x + (x + 1) = 45) :=
by
  sorry

end sum_of_consecutive_integers_with_product_506_l112_112364


namespace crayons_count_l112_112473

theorem crayons_count 
  (initial_crayons erasers : ℕ) 
  (erasers_count end_crayons : ℕ) 
  (initial_erasers : erasers = 38) 
  (end_crayons_more_erasers : end_crayons = erasers + 353) : 
  initial_crayons = end_crayons := 
by 
  sorry

end crayons_count_l112_112473


namespace vaccine_codes_l112_112660

theorem vaccine_codes (vaccines : List ℕ) :
  vaccines = [785, 567, 199, 507, 175] :=
  by
  sorry

end vaccine_codes_l112_112660


namespace value_of_v_over_u_l112_112089

variable (u v : ℝ) 

theorem value_of_v_over_u (h : u - v = (u + v) / 2) : v / u = 1 / 3 :=
by
  sorry

end value_of_v_over_u_l112_112089


namespace sum_of_first_21_terms_l112_112811

def is_constant_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem sum_of_first_21_terms (a : ℕ → ℕ) (h1 : is_constant_sum_sequence a 5) (h2 : a 1 = 2) : (Finset.range 21).sum a = 52 :=
by
  sorry

end sum_of_first_21_terms_l112_112811


namespace odd_square_diff_div_by_eight_l112_112876

theorem odd_square_diff_div_by_eight (n p : ℤ) : 
  (2 * n + 1)^2 - (2 * p + 1)^2 % 8 = 0 := 
by 
-- Here we declare the start of the proof.
  sorry

end odd_square_diff_div_by_eight_l112_112876


namespace intersection_with_xz_plane_l112_112314

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def parametric_eqn (p : Point3D) (d : Point3D) (t : ℝ) : Point3D :=
  Point3D.mk (p.x + t * d.x) (p.y + t * d.y) (p.z + t * d.z)

theorem intersection_with_xz_plane (p1 p2 : Point3D) :
  let d := direction_vector p1 p2
  let t := (p1.y / d.y)
  parametric_eqn p1 d t = Point3D.mk 4 0 9 :=
sorry

#check intersection_with_xz_plane

end intersection_with_xz_plane_l112_112314


namespace total_items_given_out_l112_112762

-- Miss Davis gave 15 popsicle sticks and 20 straws to each group.
def popsicle_sticks_per_group := 15
def straws_per_group := 20
def items_per_group := popsicle_sticks_per_group + straws_per_group

-- There are 10 groups in total.
def number_of_groups := 10

-- Prove the total number of items given out equals 350.
theorem total_items_given_out : items_per_group * number_of_groups = 350 :=
by
  sorry

end total_items_given_out_l112_112762


namespace area_of_square_field_l112_112718

def side_length : ℕ := 7
def expected_area : ℕ := 49

theorem area_of_square_field : (side_length * side_length) = expected_area := 
by
  -- The proof steps will be filled here
  sorry

end area_of_square_field_l112_112718


namespace solve_for_a_l112_112869

-- Definitions: Real number a, Imaginary unit i, complex number.
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem solve_for_a :
  ∀ (a : ℝ) (i : ℂ),
    i = Complex.I →
    is_purely_imaginary ( (3 * i / (1 + 2 * i)) * (1 - (a / 3) * i) ) →
    a = -6 :=
by
  sorry

end solve_for_a_l112_112869


namespace minimize_tangent_triangle_area_l112_112047

open Real

theorem minimize_tangent_triangle_area {a b x y : ℝ} 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  (∃ x y : ℝ, (x = a / sqrt 2 ∨ x = -a / sqrt 2) ∧ (y = b / sqrt 2 ∨ y = -b / sqrt 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_tangent_triangle_area_l112_112047


namespace quadratic_root_identity_l112_112672

theorem quadratic_root_identity (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 :=
by
  sorry

end quadratic_root_identity_l112_112672


namespace cubes_product_fraction_l112_112174

theorem cubes_product_fraction :
  (4^3 * 6^3 * 8^3 * 9^3 : ℚ) / (10^3 * 12^3 * 14^3 * 15^3) = 576 / 546875 := 
sorry

end cubes_product_fraction_l112_112174


namespace min_h4_for_ahai_avg_ge_along_avg_plus_4_l112_112764

-- Definitions from conditions
variables (a1 a2 a3 a4 : ℝ)
variables (h1 h2 h3 h4 : ℝ)

-- Conditions from the problem
axiom a1_gt_80 : a1 > 80
axiom a2_gt_80 : a2 > 80
axiom a3_gt_80 : a3 > 80
axiom a4_gt_80 : a4 > 80

axiom h1_eq_a1_plus_1 : h1 = a1 + 1
axiom h2_eq_a2_plus_2 : h2 = a2 + 2
axiom h3_eq_a3_plus_3 : h3 = a3 + 3

-- Lean 4 statement for the problem
theorem min_h4_for_ahai_avg_ge_along_avg_plus_4 : h4 ≥ 99 :=
by
  sorry

end min_h4_for_ahai_avg_ge_along_avg_plus_4_l112_112764


namespace inequality_transformation_l112_112434

variable {x y : ℝ}

theorem inequality_transformation (h : x > y) : x + 5 > y + 5 :=
by
  sorry

end inequality_transformation_l112_112434


namespace element_of_sequence_l112_112576

/-
Proving that 63 is an element of the sequence defined by aₙ = n² + 2n.
-/
theorem element_of_sequence (n : ℕ) (h : 63 = n^2 + 2 * n) : ∃ n : ℕ, 63 = n^2 + 2 * n :=
by
  sorry

end element_of_sequence_l112_112576


namespace number_of_fish_initially_tagged_l112_112908

theorem number_of_fish_initially_tagged {N T : ℕ}
  (hN : N = 1250)
  (h_ratio : 2 / 50 = T / N) :
  T = 50 :=
by
  sorry

end number_of_fish_initially_tagged_l112_112908


namespace evaluate_g_of_neg_one_l112_112059

def g (x : ℤ) : ℤ :=
  x^2 - 2*x + 1

theorem evaluate_g_of_neg_one :
  g (g (g (g (g (g (-1 : ℤ)))))) = 15738504 := by
  sorry

end evaluate_g_of_neg_one_l112_112059


namespace sufficient_but_not_necessary_l112_112973

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end sufficient_but_not_necessary_l112_112973


namespace li_to_zhang_l112_112171

theorem li_to_zhang :
  (∀ (meter chi : ℕ), 3 * meter = chi) →
  (∀ (zhang chi : ℕ), 10 * zhang = chi) →
  (∀ (kilometer li : ℕ), 2 * li = kilometer) →
  (1 * lin = 150 * zhang) :=
by
  intro h_meter h_zhang h_kilometer
  sorry

end li_to_zhang_l112_112171


namespace area_of_triangle_AEB_l112_112925

structure Rectangle :=
  (A B C D : Type)
  (AB : ℝ)
  (BC : ℝ)
  (F G E : Type)
  (DF : ℝ)
  (GC : ℝ)
  (AF_BG_intersect_at_E : Prop)

def rectangle_example : Rectangle := {
  A := Unit,
  B := Unit,
  C := Unit,
  D := Unit,
  AB := 8,
  BC := 4,
  F := Unit,
  G := Unit,
  E := Unit,
  DF := 2,
  GC := 3,
  AF_BG_intersect_at_E := true
}

theorem area_of_triangle_AEB (r : Rectangle) (h : r = rectangle_example) :
  ∃ area : ℝ, area = 128 / 3 :=
by
  sorry

end area_of_triangle_AEB_l112_112925


namespace inequality_solution_l112_112758

theorem inequality_solution (x : ℝ) : 2 * x - 1 ≤ 3 → x ≤ 2 :=
by
  intro h
  -- Here we would perform the solution steps, but we'll skip the proof with sorry.
  sorry

end inequality_solution_l112_112758


namespace probability_red_side_l112_112261

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end probability_red_side_l112_112261


namespace G_at_8_l112_112960

noncomputable def G (x : ℝ) : ℝ := sorry

theorem G_at_8 :
  (G 4 = 8) →
  (∀ x : ℝ, (x^2 + 3 * x + 2 ≠ 0) →
    G (2 * x) / G (x + 2) = 4 - (16 * x + 8) / (x^2 + 3 * x + 2)) →
  G 8 = 112 / 3 :=
by
  intros h1 h2
  sorry

end G_at_8_l112_112960


namespace min_even_integers_among_eight_l112_112600

theorem min_even_integers_among_eight :
  ∃ (x y z a b m n o : ℤ), 
    x + y + z = 30 ∧
    x + y + z + a + b = 49 ∧
    x + y + z + a + b + m + n + o = 78 ∧
    (∀ e : ℕ, (∀ x y z a b m n o : ℤ, x + y + z = 30 ∧ x + y + z + a + b = 49 ∧ x + y + z + a + b + m + n + o = 78 → 
    e = 2)) := sorry

end min_even_integers_among_eight_l112_112600


namespace general_formula_seq_arithmetic_l112_112523

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Conditions from the problem
axiom sum_condition (n : ℕ) : (1 - q) * S n + q^n = 1
axiom nonzero_q : q * (q - 1) ≠ 0
axiom arithmetic_S : S 3 + S 9 = 2 * S 6

-- Stating the proof goals
theorem general_formula (n : ℕ) : a n = q^(n-1) :=
sorry

theorem seq_arithmetic : a 2 + a 8 = 2 * a 5 :=
sorry

end general_formula_seq_arithmetic_l112_112523


namespace trapezium_area_correct_l112_112915

def a : ℚ := 20  -- Length of the first parallel side
def b : ℚ := 18  -- Length of the second parallel side
def h : ℚ := 20  -- Distance (height) between the parallel sides

def trapezium_area (a b h : ℚ) : ℚ :=
  (1/2) * (a + b) * h

theorem trapezium_area_correct : trapezium_area a b h = 380 := 
  by
    sorry  -- Proof goes here

end trapezium_area_correct_l112_112915


namespace imaginary_part_of_z_l112_112983

namespace ComplexNumberProof

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number
def z : ℂ := i^2 * (1 + i)

-- Prove the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
    -- Proof goes here
    sorry

end ComplexNumberProof

end imaginary_part_of_z_l112_112983


namespace at_least_one_heart_or_king_l112_112359

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l112_112359


namespace focus_of_parabola_l112_112016

theorem focus_of_parabola (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) ∈ {p : ℝ × ℝ | ∃ x y, y = 4 * x^2 ∧ p = (0, 1 / (4 * (1 / y)))} :=
by
  sorry

end focus_of_parabola_l112_112016


namespace min_time_calculation_l112_112952

noncomputable def min_time_to_receive_keys (diameter cyclist_speed_road cyclist_speed_alley pedestrian_speed : ℝ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let distance_pedestrian := pedestrian_speed * 1
  let min_time := (2 * Real.pi * radius - 2 * distance_pedestrian) / (cyclist_speed_road + cyclist_speed_alley)
  min_time

theorem min_time_calculation :
  min_time_to_receive_keys 4 15 20 6 = (2 * Real.pi - 2) / 21 :=
by
  sorry

end min_time_calculation_l112_112952


namespace find_initial_quarters_l112_112518

variables {Q : ℕ} -- Initial number of quarters

def quarters_to_dollars (q : ℕ) : ℝ := q * 0.25

noncomputable def initial_cash : ℝ := 40
noncomputable def cash_given_to_sister : ℝ := 5
noncomputable def quarters_given_to_sister : ℕ := 120
noncomputable def remaining_total : ℝ := 55

theorem find_initial_quarters (Q : ℕ) (h1 : quarters_to_dollars Q + 40 = 90) : Q = 200 :=
by { sorry }

end find_initial_quarters_l112_112518


namespace ordered_pair_solution_l112_112251

theorem ordered_pair_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end ordered_pair_solution_l112_112251


namespace divides_expression_l112_112274

theorem divides_expression (n : ℕ) : 7 ∣ (3^(12 * n^2 + 1) + 2^(6 * n + 2)) := sorry

end divides_expression_l112_112274


namespace find_slope_intercept_l112_112500

def line_eqn (x y : ℝ) : Prop :=
  -3 * (x - 5) + 2 * (y + 1) = 0

theorem find_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_eqn x y → y = m * x + b) ∧ (m = 3/2) ∧ (b = -17/2) := sorry

end find_slope_intercept_l112_112500


namespace integral_cos_neg_one_l112_112431

theorem integral_cos_neg_one: 
  ∫ x in (Set.Icc (Real.pi / 2) Real.pi), Real.cos x = -1 :=
by
  sorry

end integral_cos_neg_one_l112_112431


namespace bullying_instances_l112_112008

-- Let's denote the total number of suspension days due to bullying and serious incidents.
def total_suspension_days : ℕ := (3 * (10 + 10)) + 14

-- Each instance of bullying results in a 3-day suspension.
def days_per_instance : ℕ := 3

-- The number of instances of bullying given the total suspension days.
def instances_of_bullying := total_suspension_days / days_per_instance

-- We must prove that Kris is responsible for 24 instances of bullying.
theorem bullying_instances : instances_of_bullying = 24 := by
  sorry

end bullying_instances_l112_112008


namespace cutting_stick_ways_l112_112743

theorem cutting_stick_ways :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ a ∈ s, 2 * a.1 + 3 * a.2 = 14) ∧
  s.card = 2 := 
by
  sorry

end cutting_stick_ways_l112_112743


namespace number_of_chickens_l112_112748

variable (C P : ℕ) (legs_total : ℕ := 48) (legs_pig : ℕ := 4) (legs_chicken : ℕ := 2) (number_pigs : ℕ := 9)

theorem number_of_chickens (h1 : P = number_pigs)
                           (h2 : legs_pig * P + legs_chicken * C = legs_total) :
                           C = 6 :=
by
  sorry

end number_of_chickens_l112_112748


namespace construct_all_naturals_starting_from_4_l112_112978

-- Define the operations f, g, h
def f (n : ℕ) : ℕ := 10 * n
def g (n : ℕ) : ℕ := 10 * n + 4
def h (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n  -- h is only meaningful if n is even

-- Main theorem: prove that starting from 4, every natural number can be constructed
theorem construct_all_naturals_starting_from_4 :
  ∀ (n : ℕ), ∃ (k : ℕ), (f^[k] 4 = n ∨ g^[k] 4 = n ∨ h^[k] 4 = n) :=
by sorry


end construct_all_naturals_starting_from_4_l112_112978


namespace elegant_interval_solution_l112_112038

noncomputable def elegant_interval : ℝ → ℝ × ℝ := sorry

theorem elegant_interval_solution (m : ℝ) (a b : ℕ) (s : ℝ) (p : ℕ) :
  a < m ∧ m < b ∧ a + 1 = b ∧ 3 < s + b ∧ s + b ≤ 13 ∧ s = Real.sqrt a ∧ b * b + a * s = p → p = 33 ∨ p = 127 := 
by sorry

end elegant_interval_solution_l112_112038


namespace min_cans_needed_l112_112836

theorem min_cans_needed (C : ℕ → ℕ) (H : C 1 = 15) : ∃ n, C n * n >= 64 ∧ ∀ m, m < n → C 1 * m < 64 :=
by
  sorry

end min_cans_needed_l112_112836


namespace exists_group_of_four_l112_112322

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end exists_group_of_four_l112_112322


namespace focus_of_parabola_l112_112639

theorem focus_of_parabola (h : ∀ y x, y^2 = 8 * x ↔ ∃ p, y^2 = 4 * p * x ∧ p = 2): (2, 0) ∈ {f | ∃ x y, y^2 = 8 * x ∧ f = (p, 0)} :=
by
  sorry

end focus_of_parabola_l112_112639


namespace cost_per_bag_proof_minimize_total_cost_l112_112311

-- Definitions of given conditions
variable (x y : ℕ) -- cost per bag for brands A and B respectively
variable (m : ℕ) -- number of bags of brand B

def first_purchase_eq := 100 * x + 150 * y = 7000
def second_purchase_eq := 180 * x + 120 * y = 8100
def cost_per_bag_A : ℕ := 25
def cost_per_bag_B : ℕ := 30
def total_bags := 300
def constraint := (300 - m) ≤ 2 * m

-- Prove the costs per bag
theorem cost_per_bag_proof (h1 : first_purchase_eq x y)
                           (h2 : second_purchase_eq x y) :
  x = cost_per_bag_A ∧ y = cost_per_bag_B :=
sorry

-- Define the cost function and prove the purchase strategy
def total_cost (m : ℕ) : ℕ := 25 * (300 - m) + 30 * m

theorem minimize_total_cost (h : constraint m) :
  m = 100 ∧ total_cost 100 = 8000 :=
sorry

end cost_per_bag_proof_minimize_total_cost_l112_112311


namespace distribute_seedlings_l112_112257

noncomputable def box_contents : List ℕ := [28, 51, 135, 67, 123, 29, 56, 38, 79]

def total_seedlings (contents : List ℕ) : ℕ := contents.sum

def obtainable_by_sigmas (contents : List ℕ) (σs : List ℕ) : Prop :=
  ∃ groups : List (List ℕ),
    (groups.length = σs.length) ∧
    (∀ g ∈ groups, contents.contains g.sum) ∧
    (∀ g, g ∈ groups → g.sum ∈ σs)

theorem distribute_seedlings : 
  total_seedlings box_contents = 606 →
  obtainable_by_sigmas box_contents [202, 202, 202] ∧
  ∃ way1 way2 : List (List ℕ),
    (way1 ≠ way2) ∧
    (obtainable_by_sigmas box_contents [202, 202, 202]) :=
by
  sorry

end distribute_seedlings_l112_112257


namespace cube_and_reciprocal_l112_112132

theorem cube_and_reciprocal (m : ℝ) (hm : m + 1/m = 10) : m^3 + 1/m^3 = 970 := 
by
  sorry

end cube_and_reciprocal_l112_112132


namespace paving_stone_length_l112_112542

theorem paving_stone_length 
  (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (num_stones : ℕ) (stone_width : ℝ) 
  (courtyard_area : ℝ) 
  (total_stones_area : ℝ) 
  (L : ℝ) :
  courtyard_length = 50 →
  courtyard_width = 16.5 →
  num_stones = 165 →
  stone_width = 2 →
  courtyard_area = courtyard_length * courtyard_width →
  total_stones_area = num_stones * stone_width * L →
  courtyard_area = total_stones_area →
  L = 2.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end paving_stone_length_l112_112542


namespace geometric_sequence_sum_l112_112175

/-- Given a geometric sequence with common ratio r = 2, and the sum of the first four terms
    equals 1, the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a r : ℝ) (h : r = 2) (h_sum_four : a * (1 + r + r^2 + r^3) = 1) :
  a * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) = 17 :=
by
  sorry

end geometric_sequence_sum_l112_112175


namespace number_of_sides_l112_112367

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l112_112367


namespace triangle_angles_and_type_l112_112418

theorem triangle_angles_and_type
  (largest_angle : ℝ)
  (smallest_angle : ℝ)
  (middle_angle : ℝ)
  (h1 : largest_angle = 90)
  (h2 : largest_angle = 3 * smallest_angle)
  (h3 : largest_angle + smallest_angle + middle_angle = 180) :
  (largest_angle = 90 ∧ middle_angle = 60 ∧ smallest_angle = 30 ∧ largest_angle = 90) := by
  sorry

end triangle_angles_and_type_l112_112418


namespace inequality1_inequality2_l112_112795

theorem inequality1 (x : ℝ) : 
  x^2 - 2 * x - 1 > 0 -> x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1 := 
by sorry

theorem inequality2 (x : ℝ) : 
  (2 * x - 1) / (x - 3) ≥ 3 -> 3 < x ∧ x <= 8 := 
by sorry

end inequality1_inequality2_l112_112795


namespace problem_statement_l112_112346

-- Define the variables
variables (S T Tie : ℝ)

-- Define the given conditions
def condition1 : Prop := 6 * S + 4 * T + 2 * Tie = 80
def condition2 : Prop := 5 * S + 3 * T + 2 * Tie = 110

-- Define the question to be proved
def target : Prop := 4 * S + 2 * T + 2 * Tie = 50

-- Lean theorem statement
theorem problem_statement (h1 : condition1 S T Tie) (h2 : condition2 S T Tie) : target S T Tie :=
  sorry

end problem_statement_l112_112346


namespace number_of_trees_in_park_l112_112829

def number_of_trees (length width area_per_tree : ℕ) : ℕ :=
  (length * width) / area_per_tree

theorem number_of_trees_in_park :
  number_of_trees 1000 2000 20 = 100000 :=
by
  sorry

end number_of_trees_in_park_l112_112829


namespace angle_value_l112_112176

theorem angle_value (x y : ℝ) (h_parallel : True)
  (h_alt_int_ang : x = y)
  (h_triangle_sum : 2 * x + x + 60 = 180) : 
  y = 40 := 
by
  sorry

end angle_value_l112_112176


namespace profit_bicycle_l112_112134

theorem profit_bicycle (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 650) 
  (h2 : x + 2 * y = 350) : 
  x = 150 ∧ y = 100 :=
by 
  sorry

end profit_bicycle_l112_112134


namespace Jane_saves_five_dollars_l112_112685

noncomputable def first_pair_cost : ℝ := 50
noncomputable def second_pair_cost_A : ℝ := first_pair_cost * 0.6
noncomputable def second_pair_cost_B : ℝ := first_pair_cost - 15
noncomputable def promotion_A_total_cost : ℝ := first_pair_cost + second_pair_cost_A
noncomputable def promotion_B_total_cost : ℝ := first_pair_cost + second_pair_cost_B
noncomputable def Jane_savings : ℝ := promotion_B_total_cost - promotion_A_total_cost

theorem Jane_saves_five_dollars : Jane_savings = 5 := by
  sorry

end Jane_saves_five_dollars_l112_112685


namespace value_of_expression_l112_112283

theorem value_of_expression : 48^2 - 2 * 48 * 3 + 3^2 = 2025 :=
by
  sorry

end value_of_expression_l112_112283


namespace base7_to_base10_l112_112623

-- Define the base-7 number 521 in base-7
def base7_num : Nat := 5 * 7^2 + 2 * 7^1 + 1 * 7^0

-- State the theorem that needs to be proven
theorem base7_to_base10 : base7_num = 260 :=
by
  -- Proof steps will go here, but we'll skip and insert a sorry for now
  sorry

end base7_to_base10_l112_112623


namespace numerical_puzzle_l112_112565

noncomputable def THETA (T : ℕ) (A : ℕ) : ℕ := 1000 * T + 100 * T + 10 * T + A
noncomputable def BETA (B : ℕ) (T : ℕ) (A : ℕ) : ℕ := 1000 * B + 100 * T + 10 * T + A
noncomputable def GAMMA (Γ : ℕ) (E : ℕ) (M : ℕ) (A : ℕ) : ℕ := 10000 * Γ + 1000 * E + 100 * M + 10 * M + A

theorem numerical_puzzle
  (T : ℕ) (B : ℕ) (E : ℕ) (M : ℕ) (Γ : ℕ) (A : ℕ)
  (h1 : A = 0)
  (h2 : Γ = 1)
  (h3 : T + T = M)
  (h4 : 2 * E = M)
  (h5 : T ≠ B)
  (h6 : B ≠ E)
  (h7 : E ≠ M)
  (h8 : M ≠ Γ)
  (h9 : Γ ≠ T)
  (h10 : Γ ≠ B)
  (h11 : THETA T A + BETA B T A = GAMMA Γ E M A) :
  THETA 4 0 + BETA 5 4 0 = GAMMA 1 9 8 0 :=
by {
  sorry
}

end numerical_puzzle_l112_112565


namespace minimal_sum_of_squares_l112_112379

theorem minimal_sum_of_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ p q r : ℕ, a + b = p^2 ∧ b + c = q^2 ∧ a + c = r^2) ∧
  a + b + c = 55 := 
by sorry

end minimal_sum_of_squares_l112_112379


namespace distance_of_hyperbola_vertices_l112_112731

-- Define the hyperbola equation condition
def hyperbola : Prop := ∃ (y x : ℝ), (y^2 / 16) - (x^2 / 9) = 1

-- Define a variable for the distance between the vertices
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

-- The main statement to be proved
theorem distance_of_hyperbola_vertices :
  hyperbola → distance_between_vertices 4 = 8 :=
by
  intro h
  sorry

end distance_of_hyperbola_vertices_l112_112731


namespace number_symmetry_equation_l112_112564

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) :=
by
  sorry

end number_symmetry_equation_l112_112564


namespace volume_of_each_cube_is_correct_l112_112225

def box_length : ℕ := 12
def box_width : ℕ := 16
def box_height : ℕ := 6
def total_volume : ℕ := 1152
def number_of_cubes : ℕ := 384

theorem volume_of_each_cube_is_correct :
  (total_volume / number_of_cubes = 3) :=
by
  sorry

end volume_of_each_cube_is_correct_l112_112225


namespace largest_lcm_l112_112544

theorem largest_lcm :
  max (max (max (max (max (Nat.lcm 12 2) (Nat.lcm 12 4)) 
                    (Nat.lcm 12 6)) 
                 (Nat.lcm 12 8)) 
            (Nat.lcm 12 10)) 
      (Nat.lcm 12 12) = 60 :=
by sorry

end largest_lcm_l112_112544


namespace sequence_value_x_l112_112732

theorem sequence_value_x (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 5) 
  (h3 : a3 = 11) 
  (h4 : a4 = 20) 
  (h5 : a6 = 47)
  (h6 : a2 - a1 = 3) 
  (h7 : a3 - a2 = 6) 
  (h8 : a4 - a3 = 9) 
  (h9 : a6 - a5 = 15) : 
  a5 = 32 :=
sorry

end sequence_value_x_l112_112732


namespace tetrahedron_through_hole_tetrahedron_cannot_through_hole_l112_112606

/--
A regular tetrahedron with edge length 1 can pass through a circular hole if and only if the radius \( R \) is at least 0.4478, given that the thickness of the hole can be neglected.
-/

theorem tetrahedron_through_hole (R : ℝ) (h1 : R = 0.45) : true :=
by sorry

theorem tetrahedron_cannot_through_hole (R : ℝ) (h1 : R = 0.44) : false :=
by sorry

end tetrahedron_through_hole_tetrahedron_cannot_through_hole_l112_112606


namespace simplify_expression_l112_112207

theorem simplify_expression : 20 * (9 / 14) * (1 / 18) = 5 / 7 :=
by sorry

end simplify_expression_l112_112207


namespace darry_small_ladder_climbs_l112_112716

-- Define the constants based on the conditions
def full_ladder_steps := 11
def full_ladder_climbs := 10
def small_ladder_steps := 6
def total_steps := 152

-- Darry's total steps climbed via full ladder
def full_ladder_total_steps := full_ladder_steps * full_ladder_climbs

-- Define x as the number of times Darry climbed the smaller ladder
variable (x : ℕ)

-- Prove that x = 7 given the conditions
theorem darry_small_ladder_climbs (h : full_ladder_total_steps + small_ladder_steps * x = total_steps) : x = 7 :=
by 
  sorry

end darry_small_ladder_climbs_l112_112716


namespace find_interest_rate_of_first_investment_l112_112753

noncomputable def total_interest : ℚ := 73
noncomputable def interest_rate_7_percent : ℚ := 0.07
noncomputable def invested_400 : ℚ := 400
noncomputable def interest_7_percent := invested_400 * interest_rate_7_percent
noncomputable def interest_first_investment := total_interest - interest_7_percent
noncomputable def invested_first : ℚ := invested_400 - 100
noncomputable def interest_first : ℚ := 45  -- calculated as total_interest - interest_7_percent

theorem find_interest_rate_of_first_investment (r : ℚ) :
  interest_first = invested_first * r * 1 → 
  r = 0.15 :=
by
  sorry

end find_interest_rate_of_first_investment_l112_112753


namespace minimum_cost_peking_opera_l112_112406

theorem minimum_cost_peking_opera (T p₆ p₁₀ : ℕ) (xₛ yₛ : ℕ) :
  T = 140 ∧ p₆ = 6 ∧ p₁₀ = 10 ∧ xₛ + yₛ = T ∧ yₛ ≥ 2 * xₛ →
  6 * xₛ + 10 * yₛ = 1216 ∧ xₛ = 46 ∧ yₛ = 94 :=
by
   -- Proving this is skipped (left as a sorry)
  sorry

end minimum_cost_peking_opera_l112_112406


namespace values_of_x_minus_y_l112_112691

theorem values_of_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : y > x) : x - y = -2 ∨ x - y = -8 :=
  sorry

end values_of_x_minus_y_l112_112691


namespace find_number_and_n_l112_112312

def original_number (x y z n : ℕ) : Prop := 
  n = 2 ∧ 100 * x + 10 * y + z = 178

theorem find_number_and_n (x y z n : ℕ) :
  (∀ x y z n, original_number x y z n) ↔ (n = 2 ∧ 100 * x + 10 * y + z = 178) := 
sorry

end find_number_and_n_l112_112312


namespace car_average_speed_l112_112102

-- Define the given conditions
def total_time_hours : ℕ := 5
def total_distance_miles : ℕ := 200

-- Define the average speed calculation
def average_speed (distance time : ℕ) : ℕ :=
  distance / time

-- State the theorem to be proved
theorem car_average_speed :
  average_speed total_distance_miles total_time_hours = 40 :=
by
  sorry

end car_average_speed_l112_112102


namespace find_c_d_of_cubic_common_roots_l112_112858

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end find_c_d_of_cubic_common_roots_l112_112858


namespace correct_calculation_l112_112841

theorem correct_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 :=
by sorry

end correct_calculation_l112_112841


namespace q_true_or_false_l112_112723

variable (p q : Prop)

theorem q_true_or_false (h1 : ¬ (p ∧ q)) (h2 : ¬ p) : q ∨ ¬ q :=
by
  sorry

end q_true_or_false_l112_112723


namespace relationship_between_coefficients_l112_112247

theorem relationship_between_coefficients
  (b c : ℝ)
  (h_discriminant : b^2 - 4 * c ≥ 0)
  (h_root_condition : ∃ x1 x2 : ℝ, x1^2 = -x2 ∧ x1 + x2 = -b ∧ x1 * x2 = c):
  b^3 - 3 * b * c - c^2 - c = 0 :=
by
  sorry

end relationship_between_coefficients_l112_112247


namespace determine_x1_l112_112833

theorem determine_x1
  (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 :=
by
  sorry

end determine_x1_l112_112833


namespace calculate_expression_l112_112771

/-
We need to prove that the value of 18 * 36 + 54 * 18 + 18 * 9 is equal to 1782.
-/

theorem calculate_expression : (18 * 36 + 54 * 18 + 18 * 9 = 1782) :=
by
  have a1 : Int := 18 * 36
  have a2 : Int := 54 * 18
  have a3 : Int := 18 * 9
  sorry

end calculate_expression_l112_112771


namespace factor_in_range_l112_112053

-- Define the given constants
def a : ℕ := 201212200619
def lower_bound : ℕ := 6000000000
def upper_bound : ℕ := 6500000000
def m : ℕ := 6490716149

-- The Lean proof statement
theorem factor_in_range :
  m ∣ a ∧ lower_bound < m ∧ m < upper_bound :=
by
  exact ⟨sorry, sorry, sorry⟩

end factor_in_range_l112_112053


namespace walking_time_l112_112021

-- Define the conditions as Lean definitions
def minutes_in_hour : Nat := 60

def work_hours : Nat := 6
def work_minutes := work_hours * minutes_in_hour
def sitting_interval : Nat := 90
def walking_time_per_interval : Nat := 10

-- State the main theorem
theorem walking_time (h1 : 10 * 90 = 600) (h2 : 10 * (work_hours * 60) / 90 = 40) : 
  work_minutes / sitting_interval * walking_time_per_interval = 40 :=
  sorry

end walking_time_l112_112021


namespace mark_weekly_reading_l112_112258

-- Using the identified conditions
def daily_reading_hours : ℕ := 2
def additional_weekly_hours : ℕ := 4

-- Prove the total number of hours Mark wants to read per week is 18 hours
theorem mark_weekly_reading : (daily_reading_hours * 7 + additional_weekly_hours) = 18 := by
  -- Placeholder for proof
  sorry

end mark_weekly_reading_l112_112258


namespace total_distance_eq_l112_112914

def distance_traveled_by_bus : ℝ := 2.6
def distance_traveled_by_subway : ℝ := 5.98
def total_distance_traveled : ℝ := distance_traveled_by_bus + distance_traveled_by_subway

theorem total_distance_eq : total_distance_traveled = 8.58 := by
  sorry

end total_distance_eq_l112_112914


namespace exists_three_digit_number_cube_ends_in_777_l112_112133

theorem exists_three_digit_number_cube_ends_in_777 :
  ∃ x : ℤ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 := 
sorry

end exists_three_digit_number_cube_ends_in_777_l112_112133


namespace additional_distance_sam_runs_more_than_sarah_l112_112979

theorem additional_distance_sam_runs_more_than_sarah
  (street_width : ℝ) (block_side_length : ℝ)
  (h1 : street_width = 30) (h2 : block_side_length = 500) :
  let P_Sarah := 4 * block_side_length
  let P_Sam := 4 * (block_side_length + 2 * street_width)
  P_Sam - P_Sarah = 240 :=
by
  sorry

end additional_distance_sam_runs_more_than_sarah_l112_112979


namespace units_digit_17_pow_2024_l112_112028

theorem units_digit_17_pow_2024 : (17 ^ 2024) % 10 = 1 := 
by
  sorry

end units_digit_17_pow_2024_l112_112028


namespace find_complex_number_l112_112166

open Complex

theorem find_complex_number (a b : ℝ) (z : ℂ) 
  (h₁ : (∀ b: ℝ, (b^2 + 4 * b + 4 = 0) ∧ (b + a = 0))) :
  z = 2 - 2 * Complex.I :=
  sorry

end find_complex_number_l112_112166


namespace car_B_speed_90_l112_112499

def car_speed_problem (distance : ℝ) (ratio_A : ℕ) (ratio_B : ℕ) (time_minutes : ℝ) : Prop :=
  let x := distance / (ratio_A + ratio_B) * (60 / time_minutes)
  (ratio_B * x = 90)

theorem car_B_speed_90 
  (distance : ℝ := 88)
  (ratio_A : ℕ := 5)
  (ratio_B : ℕ := 6)
  (time_minutes : ℝ := 32)
  : car_speed_problem distance ratio_A ratio_B time_minutes :=
by
  sorry

end car_B_speed_90_l112_112499


namespace smallest_three_digit_solution_l112_112262

theorem smallest_three_digit_solution (n : ℕ) : 
  75 * n ≡ 225 [MOD 345] → 100 ≤ n ∧ n ≤ 999 → n = 118 :=
by
  intros h1 h2
  sorry

end smallest_three_digit_solution_l112_112262


namespace infinite_solutions_or_no_solutions_l112_112154

theorem infinite_solutions_or_no_solutions (a b : ℚ) :
  (∃ (x y : ℚ), a * x^2 + b * y^2 = 1) →
  (∀ (k : ℚ), a * k^2 + b ≠ 0 → ∃ (x_k y_k : ℚ), a * x_k^2 + b * y_k^2 = 1) :=
by
  intro h_sol h_k
  sorry

end infinite_solutions_or_no_solutions_l112_112154


namespace arith_seq_general_term_sum_b_n_l112_112280

-- Definitions and conditions
structure ArithSeq (f : ℕ → ℕ) :=
  (d : ℕ)
  (d_ne_zero : d ≠ 0)
  (Sn : ℕ → ℕ)
  (a3_plus_S5 : f 3 + Sn 5 = 42)
  (geom_seq : (f 4)^2 = (f 1) * (f 13))

-- Given the definitions and conditions, prove the general term formula of the sequence
theorem arith_seq_general_term (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℕ) 
  (d_ne_zero : d ≠ 0) (a3_plus_S5 : a_n 3 + S_n 5 = 42)
  (geom_seq : (a_n 4)^2 = (a_n 1) * (a_n 13)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of the sequence b_n
theorem sum_b_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℕ) (n : ℕ):
  b_n n = 1 / (a_n (n - 1) * a_n n) →
  T_n n = (1 / 2) * (1 - (1 / (2 * n - 1))) →
  T_n n = (n - 1) / (2 * n - 1) :=
sorry

end arith_seq_general_term_sum_b_n_l112_112280


namespace closest_number_l112_112534

theorem closest_number
  (a b c : ℝ)
  (h₀ : a = Real.sqrt 5)
  (h₁ : b = 3)
  (h₂ : b = (a + c) / 2) :
  abs (c - 3.5) ≤ abs (c - 2) ∧ abs (c - 3.5) ≤ abs (c - 2.5) ∧ abs (c - 3.5) ≤ abs (c - 3)  :=
by
  sorry

end closest_number_l112_112534


namespace beetles_consumed_per_day_l112_112390

-- Definitions
def bird_eats_beetles (n : Nat) : Nat := 12 * n
def snake_eats_birds (n : Nat) : Nat := 3 * n
def jaguar_eats_snakes (n : Nat) : Nat := 5 * n
def crocodile_eats_jaguars (n : Nat) : Nat := 2 * n

-- Initial values
def initial_jaguars : Nat := 6
def initial_crocodiles : Nat := 30
def net_increase_birds : Nat := 4
def net_increase_snakes : Nat := 2
def net_increase_jaguars : Nat := 1

-- Proof statement
theorem beetles_consumed_per_day : 
  bird_eats_beetles (snake_eats_birds (jaguar_eats_snakes initial_jaguars)) = 1080 := 
by 
  sorry

end beetles_consumed_per_day_l112_112390


namespace sin_double_angle_given_condition_l112_112469

open Real

variable (x : ℝ)

theorem sin_double_angle_given_condition :
  sin (π / 4 - x) = 3 / 5 → sin (2 * x) = 7 / 25 :=
by
  intro h
  sorry

end sin_double_angle_given_condition_l112_112469


namespace john_weekly_earnings_after_raise_l112_112138

theorem john_weekly_earnings_after_raise (original_earnings : ℝ) (raise_percentage : ℝ) (raise_amount new_earnings : ℝ) 
  (h1 : original_earnings = 50) (h2 : raise_percentage = 60) (h3 : raise_amount = (raise_percentage / 100) * original_earnings) 
  (h4 : new_earnings = original_earnings + raise_amount) : 
  new_earnings = 80 := 
by sorry

end john_weekly_earnings_after_raise_l112_112138


namespace boys_under_six_ratio_l112_112214

theorem boys_under_six_ratio (total_students : ℕ) (two_third_boys : (2/3 : ℚ) * total_students = 25) (boys_under_six : ℕ) (boys_under_six_eq : boys_under_six = 19) :
  boys_under_six / 25 = 19 / 25 :=
by
  sorry

end boys_under_six_ratio_l112_112214


namespace sum_of_17th_roots_of_unity_except_1_l112_112671

theorem sum_of_17th_roots_of_unity_except_1 :
  Complex.exp (2 * Real.pi * Complex.I / 17) +
  Complex.exp (4 * Real.pi * Complex.I / 17) +
  Complex.exp (6 * Real.pi * Complex.I / 17) +
  Complex.exp (8 * Real.pi * Complex.I / 17) +
  Complex.exp (10 * Real.pi * Complex.I / 17) +
  Complex.exp (12 * Real.pi * Complex.I / 17) +
  Complex.exp (14 * Real.pi * Complex.I / 17) +
  Complex.exp (16 * Real.pi * Complex.I / 17) +
  Complex.exp (18 * Real.pi * Complex.I / 17) +
  Complex.exp (20 * Real.pi * Complex.I / 17) +
  Complex.exp (22 * Real.pi * Complex.I / 17) +
  Complex.exp (24 * Real.pi * Complex.I / 17) +
  Complex.exp (26 * Real.pi * Complex.I / 17) +
  Complex.exp (28 * Real.pi * Complex.I / 17) +
  Complex.exp (30 * Real.pi * Complex.I / 17) +
  Complex.exp (32 * Real.pi * Complex.I / 17) = 0 := sorry

end sum_of_17th_roots_of_unity_except_1_l112_112671


namespace cos_90_eq_zero_l112_112734

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l112_112734


namespace impossible_event_abs_lt_zero_l112_112415

theorem impossible_event_abs_lt_zero (a : ℝ) : ¬ (|a| < 0) :=
sorry

end impossible_event_abs_lt_zero_l112_112415


namespace average_first_15_nat_l112_112077

-- Define the sequence and necessary conditions
def sum_first_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_first_15_nat : (sum_first_n_nat 15) / 15 = 8 := 
by 
  -- Here we shall place the proof to show the above statement holds true
  sorry

end average_first_15_nat_l112_112077


namespace cos_theta_eq_neg_2_div_sqrt_13_l112_112680

theorem cos_theta_eq_neg_2_div_sqrt_13 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < π) 
  (h3 : Real.tan θ = -3/2) : 
  Real.cos θ = -2 / Real.sqrt 13 :=
sorry

end cos_theta_eq_neg_2_div_sqrt_13_l112_112680


namespace hyperbola_center_coordinates_l112_112172

theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), 
  (∀ x y : ℝ, 
    ((4 * y - 6) ^ 2 / 36 - (5 * x - 3) ^ 2 / 49 = -1) ↔
    ((x - h) ^ 2 / ((7 / 5) ^ 2) - (y - k) ^ 2 / ((3 / 2) ^ 2) = 1)) ∧
  h = 3 / 5 ∧ k = 3 / 2 :=
by sorry

end hyperbola_center_coordinates_l112_112172


namespace total_molecular_weight_of_products_l112_112208

/-- Problem Statement: Determine the total molecular weight of the products formed when
    8 moles of Copper(II) carbonate (CuCO3) react with 6 moles of Diphosphorus pentoxide (P4O10)
    to form Copper(II) phosphate (Cu3(PO4)2) and Carbon dioxide (CO2). -/
theorem total_molecular_weight_of_products 
  (moles_CuCO3 : ℕ) 
  (moles_P4O10 : ℕ)
  (atomic_weight_Cu : ℝ := 63.55)
  (atomic_weight_P : ℝ := 30.97)
  (atomic_weight_O : ℝ := 16.00)
  (atomic_weight_C : ℝ := 12.01)
  (molecular_weight_CuCO3 : ℝ := atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O)
  (molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O)
  (molecular_weight_Cu3PO4_2 : ℝ := (3 * atomic_weight_Cu) + (2 * atomic_weight_P) + (8 * atomic_weight_O))
  (moles_Cu3PO4_2_formed : ℝ := (8 : ℝ) / 3)
  (moles_CO2_formed : ℝ := 8)
  (total_molecular_weight_Cu3PO4_2 : ℝ := moles_Cu3PO4_2_formed * molecular_weight_Cu3PO4_2)
  (total_molecular_weight_CO2 : ℝ := moles_CO2_formed * molecular_weight_CO2) : 
  (total_molecular_weight_Cu3PO4_2 + total_molecular_weight_CO2) = 1368.45 := by
  sorry

end total_molecular_weight_of_products_l112_112208


namespace ratio_of_sums_l112_112690

theorem ratio_of_sums (a b c u v w : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
    (h1 : a^2 + b^2 + c^2 = 9) (h2 : u^2 + v^2 + w^2 = 49) (h3 : a * u + b * v + c * w = 21) : 
    (a + b + c) / (u + v + w) = 3 / 7 := 
by
  sorry

end ratio_of_sums_l112_112690


namespace total_number_of_crayons_l112_112577

def number_of_blue_crayons := 3
def number_of_red_crayons := 4 * number_of_blue_crayons
def number_of_green_crayons := 2 * number_of_red_crayons
def number_of_yellow_crayons := number_of_green_crayons / 2

theorem total_number_of_crayons :
  number_of_blue_crayons + number_of_red_crayons + number_of_green_crayons + number_of_yellow_crayons = 51 :=
by 
  -- Proof is not required
  sorry

end total_number_of_crayons_l112_112577


namespace prime_p_is_2_l112_112700

theorem prime_p_is_2 (p q r : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h_sum : p + q = r) (h_lt : p < q) : 
  p = 2 :=
sorry

end prime_p_is_2_l112_112700


namespace functions_same_l112_112801

theorem functions_same (x : ℝ) : (∀ x, (y = x) → (∀ x, (y = (x^3 + x) / (x^2 + 1)))) :=
by sorry

end functions_same_l112_112801


namespace no_such_coins_l112_112705

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l112_112705


namespace height_of_sky_island_l112_112336

theorem height_of_sky_island (day_climb : ℕ) (night_slide : ℕ) (days : ℕ) (final_day_climb : ℕ) :
  day_climb = 25 →
  night_slide = 3 →
  days = 64 →
  final_day_climb = 25 →
  (days - 1) * (day_climb - night_slide) + final_day_climb = 1411 :=
by
  -- Add the formal proof here
  sorry

end height_of_sky_island_l112_112336


namespace angle_sum_at_point_l112_112566

theorem angle_sum_at_point (x : ℝ) (h : 170 + 3 * x = 360) : x = 190 / 3 :=
by
  sorry

end angle_sum_at_point_l112_112566


namespace sum_first_10_terms_l112_112934

variable (a : ℕ → ℕ)

def condition (p q : ℕ) : Prop :=
  p + q = 11 ∧ p < q

axiom condition_a_p_a_q : ∀ (p q : ℕ), (condition p q) → (a p + a q = 2^p)

theorem sum_first_10_terms (a : ℕ → ℕ) (h : ∀ (p q : ℕ), condition p q → a p + a q = 2^p) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 62) :=
by 
  sorry

end sum_first_10_terms_l112_112934


namespace largest_prime_y_in_triangle_l112_112902

-- Define that a number is prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_y_in_triangle : 
  ∃ (x y z : ℕ), is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 90 ∧ y < x ∧ y > z ∧ y = 47 :=
by
  sorry

end largest_prime_y_in_triangle_l112_112902


namespace value_of_expression_l112_112948

theorem value_of_expression (m n : ℝ) (h : m + 2 * n = 1) : 3 * m^2 + 6 * m * n + 6 * n = 3 :=
by
  sorry -- Placeholder for the proof

end value_of_expression_l112_112948


namespace intersection_complement_l112_112715

open Set

variable {α : Type*}
noncomputable def A : Set ℝ := {x | x^2 ≥ 1}
noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_complement :
  A ∩ (compl B) = (Iic (-1)) ∪ (Ioi 2) := by
sorry

end intersection_complement_l112_112715


namespace parallel_lines_necessary_and_sufficient_l112_112842

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l112_112842


namespace dozen_pencils_l112_112765

-- Define the given conditions
def pencils_total : ℕ := 144
def pencils_per_dozen : ℕ := 12

-- Theorem stating the desired proof
theorem dozen_pencils (h : pencils_total = 144) (hdozen : pencils_per_dozen = 12) : 
  pencils_total / pencils_per_dozen = 12 :=
by
  sorry

end dozen_pencils_l112_112765


namespace parallelepiped_analogy_l112_112149

-- Define the possible plane figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the concept of a parallelepiped
structure Parallelepiped : Type

-- The theorem asserting the parallelogram is the correct analogy
theorem parallelepiped_analogy : 
  ∀ (fig : PlaneFigure), 
    (fig = PlaneFigure.Parallelogram) ↔ 
    (fig = PlaneFigure.Parallelogram) :=
by sorry

end parallelepiped_analogy_l112_112149


namespace largest_divisor_of_even_diff_squares_l112_112797

theorem largest_divisor_of_even_diff_squares (m n : ℤ) (h_m_even : ∃ k : ℤ, m = 2 * k) (h_n_even : ∃ k : ℤ, n = 2 * k) (h_n_lt_m : n < m) : 
  ∃ d : ℤ, d = 16 ∧ ∀ p : ℤ, (p ∣ (m^2 - n^2)) → p ≤ d :=
sorry

end largest_divisor_of_even_diff_squares_l112_112797


namespace find_x_l112_112278

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

noncomputable def lcm_of_10_to_15 : ℕ :=
  leastCommonMultiple 10 (leastCommonMultiple 11 (leastCommonMultiple 12 (leastCommonMultiple 13 (leastCommonMultiple 14 15))))

theorem find_x :
  (lcm_of_10_to_15 / 2310 = 26) := by
  sorry

end find_x_l112_112278


namespace length_of_second_parallel_side_l112_112727

-- Define the given conditions
def parallel_side1 : ℝ := 20
def distance : ℝ := 14
def area : ℝ := 266

-- Define the theorem to prove the length of the second parallel side
theorem length_of_second_parallel_side (x : ℝ) 
  (h : area = (1 / 2) * (parallel_side1 + x) * distance) : 
  x = 18 :=
sorry

end length_of_second_parallel_side_l112_112727


namespace Mairead_triathlon_l112_112179

noncomputable def convert_km_to_miles (km: Float) : Float :=
  0.621371 * km

noncomputable def convert_yards_to_miles (yd: Float) : Float :=
  0.000568182 * yd

noncomputable def convert_feet_to_miles (ft: Float) : Float :=
  0.000189394 * ft

noncomputable def total_distance_in_miles := 
  let run_distance_km := 40.0
  let run_distance_miles := convert_km_to_miles run_distance_km
  let walk_distance_miles := 3.0/5.0 * run_distance_miles
  let jog_distance_yd := 5.0 * (walk_distance_miles * 1760.0)
  let jog_distance_miles := convert_yards_to_miles jog_distance_yd
  let bike_distance_ft := 3.0 * (jog_distance_miles * 5280.0)
  let bike_distance_miles := convert_feet_to_miles bike_distance_ft
  let swim_distance_miles := 2.5
  run_distance_miles + walk_distance_miles + jog_distance_miles + bike_distance_miles + swim_distance_miles

theorem Mairead_triathlon:
  total_distance_in_miles = 340.449562 ∧
  (convert_km_to_miles 40.0) / 10.0 = 2.485484 ∧
  (3.0/5.0 * (convert_km_to_miles 40.0)) / 10.0 = 1.4912904 ∧
  (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0))) / 10.0 = 7.45454544 ∧
  (convert_feet_to_miles (3.0 * (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0)) * 5280.0))) / 10.0 = 22.36363636 ∧
  2.5 / 10.0 = 0.25 := sorry

end Mairead_triathlon_l112_112179


namespace lottery_ticket_random_event_l112_112070

-- Define the type of possible outcomes of buying a lottery ticket
inductive LotteryOutcome
| Win
| Lose

-- Define the random event condition
def is_random_event (outcome: LotteryOutcome) : Prop :=
  match outcome with
  | LotteryOutcome.Win => True
  | LotteryOutcome.Lose => True

-- The theorem to prove that buying 1 lottery ticket and winning is a random event
theorem lottery_ticket_random_event : is_random_event LotteryOutcome.Win :=
by
  sorry

end lottery_ticket_random_event_l112_112070


namespace socks_choice_count_l112_112573

variable (white_socks : ℕ) (brown_socks : ℕ) (blue_socks : ℕ) (black_socks : ℕ)

theorem socks_choice_count :
  white_socks = 5 →
  brown_socks = 4 →
  blue_socks = 2 →
  black_socks = 2 →
  (white_socks.choose 2) + (brown_socks.choose 2) + (blue_socks.choose 2) + (black_socks.choose 2) = 18 :=
by
  -- Here the proof would be elaborated
  sorry

end socks_choice_count_l112_112573


namespace three_digit_numbers_proof_l112_112838

-- Definitions and conditions
def are_digits_distinct (A B C : ℕ) := (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C)

def is_arithmetic_mean (A B C : ℕ) := 2 * B = A + C

def geometric_mean_property (A B C : ℕ) := 
  (100 * A + 10 * B + C) * (100 * C + 10 * A + B) = (100 * B + 10 * C + A)^2

-- statement of the proof problem
theorem three_digit_numbers_proof :
  ∃ A B C : ℕ, (10 ≤ A) ∧ (A ≤ 99) ∧ (10 ≤ B) ∧ (B ≤ 99) ∧ (10 ≤ C) ∧ (C ≤ 99) ∧
  (A * 100 + B * 10 + C = 432 ∨ A * 100 + B * 10 + C = 864) ∧
  are_digits_distinct A B C ∧
  is_arithmetic_mean A B C ∧
  geometric_mean_property A B C :=
by {
  -- The Lean proof goes here
  sorry
}

end three_digit_numbers_proof_l112_112838


namespace brass_to_band_ratio_l112_112662

theorem brass_to_band_ratio
  (total_students : ℕ)
  (marching_band_fraction brass_saxophone_fraction saxophone_alto_fraction : ℚ)
  (alto_saxophone_students : ℕ)
  (h1 : total_students = 600)
  (h2 : marching_band_fraction = 1 / 5)
  (h3 : brass_saxophone_fraction = 1 / 5)
  (h4 : saxophone_alto_fraction = 1 / 3)
  (h5 : alto_saxophone_students = 4) :
  ((brass_saxophone_fraction * saxophone_alto_fraction) * total_students * marching_band_fraction = 4) →
  ((brass_saxophone_fraction * 3 * marching_band_fraction * total_students) / (marching_band_fraction * total_students) = 1 / 2) :=
by {
  -- Here we state the proof but leave it as a sorry placeholder.
  sorry
}

end brass_to_band_ratio_l112_112662


namespace least_distance_on_cone_l112_112213

noncomputable def least_distance_fly_could_crawl_cone (R C : ℝ) (slant_height : ℝ) (start_dist vertex_dist : ℝ) : ℝ :=
  if start_dist = 150 ∧ vertex_dist = 450 ∧ R = 500 ∧ C = 800 * Real.pi ∧ slant_height = R ∧ 
     (500 * (8 * Real.pi / 5) = 800 * Real.pi) then 600 else 0

theorem least_distance_on_cone : least_distance_fly_could_crawl_cone 500 (800 * Real.pi) 500 150 450 = 600 :=
by
  sorry

end least_distance_on_cone_l112_112213


namespace simplify_expression_l112_112706

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l112_112706


namespace cubic_sum_l112_112490

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := by
  -- Using the provided conditions x + y = 5 and x^2 + y^2 = 13
  sorry

end cubic_sum_l112_112490


namespace problem_statement_l112_112847

noncomputable def f_B (x : ℝ) : ℝ := -x^2
noncomputable def f_D (x : ℝ) : ℝ := Real.cos x

theorem problem_statement :
  (∀ x : ℝ, f_B (-x) = f_B x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_B x1 > f_B x2) ∧
  (∀ x : ℝ, f_D (-x) = f_D x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_D x1 > f_D x2) :=
  sorry

end problem_statement_l112_112847


namespace part1_part2_l112_112818

-- Problem 1: Given |x| = 9, |y| = 5, x < 0, y > 0, prove x + y = -4
theorem part1 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : x < 0) (h4 : y > 0) : x + y = -4 :=
sorry

-- Problem 2: Given |x| = 9, |y| = 5, |x + y| = x + y, prove x - y = 4 or x - y = 14
theorem part2 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : |x + y| = x + y) : x - y = 4 ∨ x - y = 14 :=
sorry

end part1_part2_l112_112818


namespace problem_statement_l112_112645

theorem problem_statement (x : ℝ) (h : x^2 + 4 * x - 2 = 0) : 3 * x^2 + 12 * x - 23 = -17 :=
sorry

end problem_statement_l112_112645


namespace probability_of_A_not_losing_l112_112151

/-- The probability of player A winning is 0.3,
    and the probability of a draw between player A and player B is 0.4.
    Hence, the probability of player A not losing is 0.7. -/
theorem probability_of_A_not_losing (pA_win p_draw : ℝ) (hA_win : pA_win = 0.3) (h_draw : p_draw = 0.4) : 
  (pA_win + p_draw = 0.7) :=
by
  sorry

end probability_of_A_not_losing_l112_112151


namespace canteen_needs_bananas_l112_112000

-- Define the given conditions
def total_bananas := 9828
def weeks := 9
def days_in_week := 7
def bananas_in_dozen := 12

-- Calculate the required value and prove the equivalence
theorem canteen_needs_bananas : 
  (total_bananas / (weeks * days_in_week)) / bananas_in_dozen = 13 :=
by
  -- This is where the proof would go
  sorry

end canteen_needs_bananas_l112_112000


namespace total_pens_bought_l112_112889

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l112_112889


namespace range_of_function_l112_112338

theorem range_of_function (y : ℝ) (t: ℝ) (x : ℝ) (h_t : t = x^2 - 1) (h_domain : t ∈ Set.Ici (-1)) :
  ∃ (y_set : Set ℝ), ∀ y ∈ y_set, y = (1/3)^t ∧ y_set = Set.Ioo 0 3 ∨ y_set = Set.Icc 0 3 := by
  sorry

end range_of_function_l112_112338


namespace perpendicular_line_eq_l112_112125

theorem perpendicular_line_eq :
  ∃ (A B C : ℝ), (A * 0 + B * 4 + C = 0) ∧ (A = 3) ∧ (B = 1) ∧ (C = -4) ∧ (3 * 1 + 1 * -3 = 0) :=
sorry

end perpendicular_line_eq_l112_112125


namespace cannot_determine_remaining_pictures_l112_112378

theorem cannot_determine_remaining_pictures (taken_pics : ℕ) (dolphin_show_pics : ℕ) (total_pics : ℕ) :
  taken_pics = 28 → dolphin_show_pics = 16 → total_pics = 44 → 
  (∀ capacity : ℕ, ¬ (total_pics + x = capacity)) → 
  ¬ ∃ remaining_pics : ℕ, remaining_pics = capacity - total_pics :=
by {
  sorry
}

end cannot_determine_remaining_pictures_l112_112378


namespace difference_before_flipping_l112_112754

-- Definitions based on the conditions:
variables (Y G : ℕ) -- Number of yellow and green papers

-- Condition: flipping 16 yellow papers changes counts as described
def papers_after_flipping (Y G : ℕ) : Prop :=
  Y - 16 = G + 16

-- Condition: after flipping, there are 64 more green papers than yellow papers.
def green_more_than_yellow_after_flipping (G Y : ℕ) : Prop :=
  G + 16 = (Y - 16) + 64

-- Statement: Prove the difference in the number of green and yellow papers before flipping was 32.
theorem difference_before_flipping (Y G : ℕ) (h1 : papers_after_flipping Y G) (h2 : green_more_than_yellow_after_flipping G Y) :
  (Y - G) = 32 :=
by
  sorry

end difference_before_flipping_l112_112754


namespace rhombus_perimeter_l112_112870

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 40 :=
by
  sorry

end rhombus_perimeter_l112_112870


namespace geometric_prog_105_l112_112991

theorem geometric_prog_105 {a q : ℝ} 
  (h_sum : a + a * q + a * q^2 = 105) 
  (h_arith : a * q - a = (a * q^2 - 15) - a * q) :
  (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 0.5) :=
by
  sorry

end geometric_prog_105_l112_112991


namespace distance_from_C_to_B_is_80_l112_112451

theorem distance_from_C_to_B_is_80
  (x : ℕ)
  (h1 : x = 60)
  (h2 : ∀ (ab cb : ℕ), ab = x → cb = x + 20  → (cb = 80))
  : x + 20 = 80 := by
  sorry

end distance_from_C_to_B_is_80_l112_112451


namespace smallest_s_plus_d_l112_112583

theorem smallest_s_plus_d (s d : ℕ) (h_pos_s : s > 0) (h_pos_d : d > 0)
  (h_eq : 1 / s + 1 / (2 * s) + 1 / (3 * s) = 1 / (d^2 - 2 * d)) :
  s + d = 50 :=
sorry

end smallest_s_plus_d_l112_112583


namespace Saheed_earnings_l112_112816

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end Saheed_earnings_l112_112816


namespace regular_tetrahedron_of_angle_l112_112535

-- Definition and condition from the problem
def angle_between_diagonals (shape : Type _) (adj_sides_diag_angle : ℝ) : Prop :=
  adj_sides_diag_angle = 60

-- Theorem stating the problem in Lean 4
theorem regular_tetrahedron_of_angle (shape : Type _) (adj_sides_diag_angle : ℝ) 
  (h : angle_between_diagonals shape adj_sides_diag_angle) : 
  shape = regular_tetrahedron :=
sorry

end regular_tetrahedron_of_angle_l112_112535


namespace martha_pins_l112_112546

theorem martha_pins (k : ℕ) :
  (2 + 9 * k > 45) ∧ (2 + 14 * k < 90) ↔ (k = 5 ∨ k = 6) :=
by
  sorry

end martha_pins_l112_112546


namespace magnitude_of_b_l112_112585

open Real

noncomputable def a : ℝ × ℝ := (-sqrt 3, 1)

theorem magnitude_of_b (b : ℝ × ℝ)
    (h1 : (a.1 + 2 * b.1, a.2 + 2 * b.2) = (a.1, a.2))
    (h2 : (a.1 + b.1, a.2 + b.2) = (b.1, b.2)) :
    sqrt (b.1 ^ 2 + b.2 ^ 2) = sqrt 2 :=
sorry

end magnitude_of_b_l112_112585


namespace avg_decrease_by_one_l112_112830

noncomputable def average_decrease (obs : Fin 7 → ℕ) : ℕ :=
  let sum6 := 90
  let seventh := 8
  let new_sum := sum6 + seventh
  let new_avg := new_sum / 7
  let old_avg := 15
  old_avg - new_avg

theorem avg_decrease_by_one :
  (average_decrease (fun _ => 0)) = 1 :=
by
  sorry

end avg_decrease_by_one_l112_112830


namespace quadratic_nonneg_range_l112_112861

theorem quadratic_nonneg_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end quadratic_nonneg_range_l112_112861


namespace three_digit_number_550_l112_112755

theorem three_digit_number_550 (N : ℕ) (a b c : ℕ) (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : 11 ∣ N)
  (h6 : N / 11 = a^2 + b^2 + c^2) : N = 550 :=
by
  sorry

end three_digit_number_550_l112_112755


namespace cards_received_while_in_hospital_l112_112644

theorem cards_received_while_in_hospital (T H C : ℕ) (hT : T = 690) (hC : C = 287) (hH : H = T - C) : H = 403 :=
by
  sorry

end cards_received_while_in_hospital_l112_112644


namespace frame_cover_100x100_l112_112580

theorem frame_cover_100x100 :
  ∃! (cover: (ℕ → ℕ → Prop)), (∀ (n : ℕ) (frame: ℕ → ℕ → Prop),
    (∃ (i j : ℕ), (cover (i + n) j ∧ frame (i + n) j ∧ cover (i - n) j ∧ frame (i - n) j) ∧
                   (∃ (k l : ℕ), (cover k (l + n) ∧ frame k (l + n) ∧ cover k (l - n) ∧ frame k (l - n)))) →
    (∃ (i' j' k' l' : ℕ), cover i' j' ∧ frame i' j' ∧ cover k' l' ∧ frame k' l')) :=
sorry

end frame_cover_100x100_l112_112580


namespace shaded_area_l112_112001

theorem shaded_area (x1 y1 x2 y2 x3 y3 : ℝ) 
  (vA vB vC vD vE vF : ℝ × ℝ)
  (h1 : vA = (0, 0))
  (h2 : vB = (0, 12))
  (h3 : vC = (12, 12))
  (h4 : vD = (12, 0))
  (h5 : vE = (24, 0))
  (h6 : vF = (18, 12))
  (h_base : 32 - 12 = 20)
  (h_height : 12 = 12) :
  (1 / 2 : ℝ) * 20 * 12 = 120 :=
by
  sorry

end shaded_area_l112_112001


namespace find_normal_price_l112_112385

theorem find_normal_price (P : ℝ) (S : ℝ) (d1 d2 d3 : ℝ) : 
  (P * (1 - d1) * (1 - d2) * (1 - d3) = S) → S = 144 → d1 = 0.12 → d2 = 0.22 → d3 = 0.15 → P = 246.81 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_normal_price_l112_112385


namespace smallest_number_of_locks_and_keys_l112_112688

open Finset Nat

-- Definitions based on conditions
def committee : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

def can_open_safe (members : Finset ℕ) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 6 → members ⊆ subset

def cannot_open_safe (members : Finset ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset.card = 5 ∧ members ⊆ subset

-- Proof statement
theorem smallest_number_of_locks_and_keys :
  ∃ (locks keys : ℕ), locks = 462 ∧ keys = 2772 ∧
  (∀ (subset : Finset ℕ), subset.card = 6 → can_open_safe subset) ∧
  (∀ (subset : Finset ℕ), subset.card = 5 → ¬can_open_safe subset) :=
sorry

end smallest_number_of_locks_and_keys_l112_112688


namespace c_linear_combination_of_a_b_l112_112822

-- Definitions of vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

-- Theorem stating the relationship between vectors a, b, and c
theorem c_linear_combination_of_a_b :
  c = (1 / 2 : ℝ) • a + (-3 / 2 : ℝ) • b :=
  sorry

end c_linear_combination_of_a_b_l112_112822


namespace power_of_integer_is_two_l112_112628

-- Definitions based on conditions
def is_power_of_integer (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), n = m^k

-- Given conditions translated to Lean definitions
def g : ℕ := 14
def n : ℕ := 3150 * g

-- The proof problem statement in Lean
theorem power_of_integer_is_two (h : g = 14) : is_power_of_integer n :=
sorry

end power_of_integer_is_two_l112_112628


namespace tangent_line_intersect_x_l112_112823

noncomputable def tangent_intercept_x : ℚ := 9/2

theorem tangent_line_intersect_x (x : ℚ)
  (h₁ : x > 0)
  (h₂ : ∃ r₁ r₂ d : ℚ, r₁ = 3 ∧ r₂ = 5 ∧ d = 12 ∧ x = (r₂ * d) / (r₁ + r₂)) :
  x = tangent_intercept_x :=
by
  sorry

end tangent_line_intersect_x_l112_112823


namespace last_student_score_is_61_l112_112244

noncomputable def average_score_19_students := 82
noncomputable def average_score_20_students := 84
noncomputable def total_students := 20
noncomputable def oliver_multiplier := 2

theorem last_student_score_is_61 
  (total_score_19_students : ℝ := total_students - 1 * average_score_19_students)
  (total_score_20_students : ℝ := total_students * average_score_20_students)
  (oliver_score : ℝ := total_score_20_students - total_score_19_students)
  (last_student_score : ℝ := oliver_score / oliver_multiplier) :
  last_student_score = 61 :=
sorry

end last_student_score_is_61_l112_112244


namespace negation_of_implication_l112_112913

theorem negation_of_implication (x : ℝ) :
  ¬ (x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negation_of_implication_l112_112913


namespace linear_function_through_origin_l112_112516

theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 1) * x + m ^ 2 - 1) → (x = 0 ∧ y = 0) → m = -1) :=
sorry

end linear_function_through_origin_l112_112516


namespace speed_of_second_train_40_kmph_l112_112303

noncomputable def length_train_1 : ℝ := 140
noncomputable def length_train_2 : ℝ := 160
noncomputable def crossing_time : ℝ := 10.799136069114471
noncomputable def speed_train_1 : ℝ := 60

theorem speed_of_second_train_40_kmph :
  let total_distance := length_train_1 + length_train_2
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  let speed_train_2 := relative_speed_kmph - speed_train_1
  speed_train_2 = 40 :=
by
  sorry

end speed_of_second_train_40_kmph_l112_112303


namespace arcsin_zero_l112_112238

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end arcsin_zero_l112_112238


namespace students_only_biology_students_biology_or_chemistry_but_not_both_l112_112083

def students_enrolled_in_both : ℕ := 15
def total_biology_students : ℕ := 35
def students_only_chemistry : ℕ := 18

theorem students_only_biology (h₀ : students_enrolled_in_both ≤ total_biology_students) :
  total_biology_students - students_enrolled_in_both = 20 := by
  sorry

theorem students_biology_or_chemistry_but_not_both :
  total_biology_students - students_enrolled_in_both + students_only_chemistry = 38 := by
  sorry

end students_only_biology_students_biology_or_chemistry_but_not_both_l112_112083


namespace range_of_m_l112_112226

theorem range_of_m (a m : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  m * (a + 1/a) / Real.sqrt 2 > 1 → m ≥ Real.sqrt 2 / 2 := by
  sorry

end range_of_m_l112_112226


namespace min_rings_to_connect_all_segments_l112_112410

-- Define the problem setup
structure ChainSegment where
  rings : Fin 3 → Type

-- Define the number of segments
def num_segments : ℕ := 5

-- Define the minimum number of rings to be opened and rejoined
def min_rings_to_connect (seg : Fin num_segments) : ℕ :=
  3

theorem min_rings_to_connect_all_segments :
  ∀ segs : Fin num_segments,
  (∃ n, n = min_rings_to_connect segs) :=
by
  -- Proof to be provided
  sorry

end min_rings_to_connect_all_segments_l112_112410


namespace sqrt_sq_eq_l112_112530

theorem sqrt_sq_eq (x : ℝ) : (Real.sqrt x) ^ 2 = x := by
  sorry

end sqrt_sq_eq_l112_112530


namespace probability_auntie_em_can_park_l112_112232

/-- A parking lot has 20 spaces in a row. -/
def total_spaces : ℕ := 20

/-- Fifteen cars arrive, each requiring one parking space, and their drivers choose spaces at random from among the available spaces. -/
def cars : ℕ := 15

/-- Auntie Em's SUV requires 3 adjacent empty spaces. -/
def required_adjacent_spaces : ℕ := 3

/-- Calculate the probability that there are 3 consecutive empty spaces among the 5 remaining spaces after 15 cars are parked in 20 spaces.
Expected answer is (12501 / 15504) -/
theorem probability_auntie_em_can_park : 
    (1 - (↑(Nat.choose 15 5) / ↑(Nat.choose 20 5))) = (12501 / 15504) := 
sorry

end probability_auntie_em_can_park_l112_112232


namespace cubic_polynomial_range_l112_112954

-- Define the conditions and the goal in Lean
theorem cubic_polynomial_range :
  ∀ x : ℝ, (x^2 - 5 * x + 6 < 0) → (41 < x^3 + 5 * x^2 + 6 * x + 1) ∧ (x^3 + 5 * x^2 + 6 * x + 1 < 91) :=
by
  intros x hx
  have h1 : 2 < x := sorry
  have h2 : x < 3 := sorry
  have h3 : (x^3 + 5 * x^2 + 6 * x + 1) > 41 := sorry
  have h4 : (x^3 + 5 * x^2 + 6 * x + 1) < 91 := sorry
  exact ⟨h3, h4⟩ 

end cubic_polynomial_range_l112_112954


namespace triangle_number_arrangement_l112_112827

noncomputable def numbers := [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

theorem triangle_number_arrangement : 
  ∃ (f : Fin 9 → Fin 9), 
    (numbers[f 0] + numbers[f 1] + numbers[f 2] = 
     numbers[f 3] + numbers[f 4] + numbers[f 5] ∧ 
     numbers[f 3] + numbers[f 4] + numbers[f 5] = 
     numbers[f 6] + numbers[f 7] + numbers[f 8]) :=
sorry

end triangle_number_arrangement_l112_112827


namespace union_of_intervals_l112_112601

open Set

variable {α : Type*}

theorem union_of_intervals : 
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  A ∪ B = Ioo (-1 : ℝ) 2 := 
by
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  sorry

end union_of_intervals_l112_112601


namespace sum_of_squares_nonnegative_l112_112785

theorem sum_of_squares_nonnegative (x y z : ℝ) : x^2 + y^2 + z^2 - x * y - x * z - y * z ≥ 0 :=
  sorry

end sum_of_squares_nonnegative_l112_112785


namespace find_a_l112_112774

theorem find_a (a : ℕ) (h_pos : 0 < a)
  (h_cube : ∀ n : ℕ, 0 < n → ∃ k : ℤ, 4 * ((a : ℤ) ^ n + 1) = k^3) :
  a = 1 :=
sorry

end find_a_l112_112774


namespace one_minus_repeating_decimal_three_equals_two_thirds_l112_112084

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end one_minus_repeating_decimal_three_equals_two_thirds_l112_112084


namespace ball_distribution_ways_l112_112493

theorem ball_distribution_ways :
  let R := 5
  let W := 3
  let G := 2
  let total_balls := 10
  let balls_in_first_box := 4
  ∃ (distributions : ℕ), distributions = (Nat.choose total_balls balls_in_first_box) ∧ distributions = 210 :=
by
  sorry

end ball_distribution_ways_l112_112493


namespace problem1_problem2_l112_112068

noncomputable def f (x : Real) : Real := 
  let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2

theorem problem1 (x : Real) : 
  ∃ k : Int, - Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi :=
  sorry

theorem problem2 (A B C a b c : Real)
  (h1 : a = Real.sqrt 7)
  (h2 : Real.sin B = 2 * Real.sin C)
  (h3 : f A = 2)
  : (∃ area : Real, area = (7 * Real.sqrt 3) / 6) :=
  sorry

end problem1_problem2_l112_112068


namespace largest_integer_solution_of_inequality_l112_112561

theorem largest_integer_solution_of_inequality :
  ∃ x : ℤ, x < 2 ∧ (∀ y : ℤ, y < 2 → y ≤ x) ∧ -x + 3 > 1 :=
sorry

end largest_integer_solution_of_inequality_l112_112561


namespace smallest_z_is_14_l112_112155

-- Define the consecutive even integers and the equation.
def w (k : ℕ) := 2 * k
def x (k : ℕ) := 2 * k + 2
def y (k : ℕ) := 2 * k + 4
def z (k : ℕ) := 2 * k + 6

theorem smallest_z_is_14 : ∃ k : ℕ, z k = 14 ∧ w k ^ 3 + x k ^ 3 + y k ^ 3 = z k ^ 3 :=
by sorry

end smallest_z_is_14_l112_112155


namespace inequality_neg_multiplication_l112_112599

theorem inequality_neg_multiplication (m n : ℝ) (h : m > n) : -2 * m < -2 * n :=
by {
  sorry
}

end inequality_neg_multiplication_l112_112599


namespace nearest_integer_to_3_plus_sqrt2_pow_four_l112_112903

open Real

theorem nearest_integer_to_3_plus_sqrt2_pow_four : 
  (∃ n : ℤ, abs (n - (3 + (sqrt 2))^4) < 0.5) ∧ 
  (abs (382 - (3 + (sqrt 2))^4) < 0.5) := 
by 
  sorry

end nearest_integer_to_3_plus_sqrt2_pow_four_l112_112903


namespace evening_temperature_is_correct_l112_112092

-- Define the temperatures at noon and in the evening
def T_noon : ℤ := 3
def T_evening : ℤ := -2

-- State the theorem to prove
theorem evening_temperature_is_correct : T_evening = -2 := by
  sorry

end evening_temperature_is_correct_l112_112092


namespace range_of_m_condition_l112_112532

theorem range_of_m_condition (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ * x₁ - 2 * m * x₁ + m - 3 = 0) 
  (h₂ : x₂ * x₂ - 2 * m * x₂ + m - 3 = 0)
  (hx₁ : x₁ > -1 ∧ x₁ < 0)
  (hx₂ : x₂ > 3) :
  m > 6 / 5 ∧ m < 3 :=
sorry

end range_of_m_condition_l112_112532


namespace opposite_number_of_sqrt_of_9_is_neg3_l112_112950

theorem opposite_number_of_sqrt_of_9_is_neg3 :
  - (Real.sqrt 9) = -3 :=
by
  -- The proof is omitted as required.
  sorry

end opposite_number_of_sqrt_of_9_is_neg3_l112_112950


namespace proper_polygons_m_lines_l112_112429

noncomputable def smallest_m := 2

theorem proper_polygons_m_lines (P : Finset (Set (ℝ × ℝ)))
  (properly_placed : ∀ (p1 p2 : Set (ℝ × ℝ)), p1 ∈ P → p2 ∈ P → ∃ l : Set (ℝ × ℝ), (0, 0) ∈ l ∧ ∀ (p : Set (ℝ × ℝ)), p ∈ P → ¬Disjoint l p) :
  ∃ (m : ℕ), m = smallest_m ∧ ∀ (lines : Finset (Set (ℝ × ℝ))), 
    (∀ l ∈ lines, (0, 0) ∈ l) → lines.card = m → ∀ p ∈ P, ∃ l ∈ lines, ¬Disjoint l p := sorry

end proper_polygons_m_lines_l112_112429


namespace jennie_speed_difference_l112_112233

noncomputable def average_speed_difference : ℝ :=
  let distance := 200
  let time_heavy_traffic := 5
  let construction_delay := 0.5
  let rest_stops_heavy := 0.5
  let time_no_traffic := 4
  let rest_stops_no_traffic := 1 / 3
  let actual_driving_time_heavy := time_heavy_traffic - construction_delay - rest_stops_heavy
  let actual_driving_time_no := time_no_traffic - rest_stops_no_traffic
  let average_speed_heavy := distance / actual_driving_time_heavy
  let average_speed_no := distance / actual_driving_time_no
  average_speed_no - average_speed_heavy

theorem jennie_speed_difference :
  average_speed_difference = 4.5 :=
sorry

end jennie_speed_difference_l112_112233


namespace least_f_e_l112_112319

theorem least_f_e (e : ℝ) (he : e > 0) : 
  ∃ f, (∀ (a b c d : ℝ), a^3 + b^3 + c^3 + d^3 ≤ e^2 * (a^2 + b^2 + c^2 + d^2) + f * (a^4 + b^4 + c^4 + d^4)) ∧ f = 1 / (4 * e^2) :=
sorry

end least_f_e_l112_112319


namespace function_passes_through_fixed_point_l112_112353

noncomputable def f (a x : ℝ) := a^(x+1) - 1

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  f a (-1) = 0 := by
  sorry

end function_passes_through_fixed_point_l112_112353


namespace totalCupsOfLiquid_l112_112551

def amountOfOil : ℝ := 0.17
def amountOfWater : ℝ := 1.17

theorem totalCupsOfLiquid : amountOfOil + amountOfWater = 1.34 := by
  sorry

end totalCupsOfLiquid_l112_112551


namespace perpendicular_lines_l112_112440

theorem perpendicular_lines (m : ℝ) :
  (m+2)*(m-1) + m*(m-4) = 0 ↔ m = 2 ∨ m = -1/2 :=
by 
  sorry

end perpendicular_lines_l112_112440


namespace fewest_four_dollar_frisbees_l112_112578

theorem fewest_four_dollar_frisbees (x y: ℕ): 
    x + y = 64 ∧ 3 * x + 4 * y = 200 → y = 8 := by sorry

end fewest_four_dollar_frisbees_l112_112578


namespace inequality_relationship_l112_112597

variable (a b : ℝ)

theorem inequality_relationship
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end inequality_relationship_l112_112597


namespace distance_between_two_girls_after_12_hours_l112_112775

theorem distance_between_two_girls_after_12_hours :
  let speed1 := 7 -- speed of the first girl (km/hr)
  let speed2 := 3 -- speed of the second girl (km/hr)
  let time := 12 -- time (hours)
  let distance1 := speed1 * time -- distance traveled by the first girl
  let distance2 := speed2 * time -- distance traveled by the second girl
  distance1 + distance2 = 120 := -- total distance
by
  -- Here, we would provide the proof, but we put sorry to skip it
  sorry

end distance_between_two_girls_after_12_hours_l112_112775


namespace mask_digit_correctness_l112_112768

noncomputable def elephant_mask_digit : ℕ :=
  6
  
noncomputable def mouse_mask_digit : ℕ :=
  4

noncomputable def guinea_pig_mask_digit : ℕ :=
  8

noncomputable def panda_mask_digit : ℕ :=
  1

theorem mask_digit_correctness :
  (∃ (d1 d2 d3 d4 : ℕ), d1 * d1 = 16 ∧ d2 * d2 = 64 ∧ d3 * d3 = 49 ∧ d4 * d4 = 81) →
  elephant_mask_digit = 6 ∧ mouse_mask_digit = 4 ∧ guinea_pig_mask_digit = 8 ∧ panda_mask_digit = 1 :=
by
  -- skip the proof
  sorry

end mask_digit_correctness_l112_112768


namespace paint_proof_l112_112512

/-- 
Suppose Jack's room has 27 square meters of wall and ceiling area. He has three choices for paint:
- Using 1 can of paint leaves 1 liter of paint left over,
- Using 5 gallons of paint leaves 1 liter of paint left over,
- Using 4 gallons and 2.8 liters of paint.

1. Prove: The ratio between the volume of a can and the volume of a gallon is 1:5.
2. Prove: The volume of a gallon is 3.8 liters.
3. Prove: The paint's coverage is 1.5 square meters per liter.
-/
theorem paint_proof (A : ℝ) (C G : ℝ) (R : ℝ):
  ∀ (H1: A = 27) (H2: C - 1 = 27) (H3: 5 * G - 1 = 27) (H4: 4 * G + 2.8 = 27), 
  (C / G = 1 / 5) ∧ (G = 3.8) ∧ ((A / (5 * G - 1)) = 1.5) :=
by
  sorry

end paint_proof_l112_112512


namespace hyperbola_vertex_distance_l112_112034

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l112_112034


namespace numberOfBoys_is_50_l112_112492

-- Define the number of boys and the conditions given.
def numberOfBoys (B G : ℕ) : Prop :=
  B / G = 5 / 13 ∧ G = B + 80

-- The theorem that we need to prove.
theorem numberOfBoys_is_50 (B G : ℕ) (h : numberOfBoys B G) : B = 50 :=
  sorry

end numberOfBoys_is_50_l112_112492


namespace range_of_a_l112_112955

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 :=
by
  sorry

end range_of_a_l112_112955


namespace solution_ne_zero_l112_112309

theorem solution_ne_zero (a x : ℝ) (h : x = a * x + 1) : x ≠ 0 := sorry

end solution_ne_zero_l112_112309


namespace slope_of_line_l112_112891

noncomputable def line_eq (x y : ℝ) := x / 4 + y / 5 = 1

theorem slope_of_line : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4) :=
sorry

end slope_of_line_l112_112891


namespace solve_for_a_l112_112513

theorem solve_for_a (a : ℝ) (h : ∃ x, x = 2 ∧ a * x - 4 * (x - a) = 1) : a = 3 / 2 :=
sorry

end solve_for_a_l112_112513


namespace distribution_ways_l112_112817

-- Define the conditions
def num_papers : ℕ := 7
def num_friends : ℕ := 10

-- Define the theorem to prove the number of ways to distribute the papers
theorem distribution_ways : (num_friends ^ num_papers) = 10000000 := by
  -- This is where the proof would go
  sorry

end distribution_ways_l112_112817


namespace cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l112_112896

-- Part (a)
theorem cupSaucersCombination :
  (5 : ℕ) * (3 : ℕ) = 15 :=
by
  -- Proof goes here
  sorry

-- Part (b)
theorem cupSaucerSpoonCombination :
  (5 : ℕ) * (3 : ℕ) * (4 : ℕ) = 60 :=
by
  -- Proof goes here
  sorry

-- Part (c)
theorem twoDifferentItemsCombination :
  (5 * 3 + 5 * 4 + 3 * 4 : ℕ) = 47 :=
by
  -- Proof goes here
  sorry

end cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l112_112896


namespace find_y_squared_l112_112189

theorem find_y_squared (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : 2 * x - y = 20) : y ^ 2 = 4 := 
sorry

end find_y_squared_l112_112189


namespace number_of_extreme_points_l112_112730

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - a

theorem number_of_extreme_points (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 6 * x + 4) > 0) →
  0 = 0 :=
by
  intro h
  sorry

end number_of_extreme_points_l112_112730


namespace original_price_of_sarees_l112_112193
open Real

theorem original_price_of_sarees (P : ℝ) (h : 0.70 * 0.80 * P = 224) : P = 400 :=
sorry

end original_price_of_sarees_l112_112193


namespace problem_solution_l112_112652

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * n - 1

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_c (n : ℕ) : ℕ :=
  sequence_a n * 2 ^ (sequence_a n)

def sum_T (n : ℕ) : ℕ :=
  (6 * n - 5) * 2 ^ (2 * n + 1) + 10

theorem problem_solution (n : ℕ) (hn : n ≥ 1) :
  ∀ n, (sum_S 1 = 1) ∧ (sequence_a 1 = 1) ∧ 
          (∀ n ≥ 2, sequence_a n = 2 * n - 1) ∧
          (sum_T n = (6 * n - 5) * 2 ^ (2 * n + 1) + 10 / 9) :=
by sorry

end problem_solution_l112_112652


namespace students_gold_award_freshmen_l112_112813

theorem students_gold_award_freshmen 
    (total_students total_award_winners : ℕ)
    (students_selected exchange_meeting : ℕ)
    (freshmen_selected gold_award_selected : ℕ)
    (prop1 : total_award_winners = 120)
    (prop2 : exchange_meeting = 24)
    (prop3 : freshmen_selected = 6)
    (prop4 : gold_award_selected = 4) :
    ∃ (gold_award_students : ℕ), gold_award_students = 4 ∧ gold_award_students ≤ freshmen_selected :=
by
  sorry

end students_gold_award_freshmen_l112_112813


namespace max_odd_numbers_in_pyramid_l112_112824

-- Define the properties of the pyramid
def is_sum_of_immediate_below (p : Nat → Nat → Nat) : Prop :=
  ∀ r c : Nat, r > 0 → p r c = p (r - 1) c + p (r - 1) (c + 1)

-- Define what it means for a number to be odd
def is_odd (n : Nat) : Prop := n % 2 = 1

-- Define the pyramid structure and number of rows
def pyramid (n : Nat) := { p : Nat → Nat → Nat // is_sum_of_immediate_below p ∧ n = 6 }

-- Theorem statement
theorem max_odd_numbers_in_pyramid (p : Nat → Nat → Nat) (h : is_sum_of_immediate_below p ∧ 6 = 6) : ∃ k : Nat, (∀ i j, is_odd (p i j) → k ≤ 14) := 
sorry

end max_odd_numbers_in_pyramid_l112_112824


namespace count_false_propositions_l112_112603

theorem count_false_propositions 
  (P : Prop) 
  (inverse_P : Prop) 
  (negation_P : Prop) 
  (converse_P : Prop) 
  (h1 : ¬P) 
  (h2 : inverse_P) 
  (h3 : negation_P ↔ ¬P) 
  (h4 : converse_P ↔ P) : 
  ∃ n : ℕ, n = 2 ∧ 
  ¬P ∧ ¬converse_P ∧ 
  inverse_P ∧ negation_P := 
sorry

end count_false_propositions_l112_112603


namespace fraction_identity_l112_112710

theorem fraction_identity (a b : ℚ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 :=
by
  sorry

end fraction_identity_l112_112710


namespace find_value_of_y_l112_112612

theorem find_value_of_y (x y : ℚ) 
  (h1 : x = 51) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 63000) : 
  y = 8 / 17 := 
by 
  sorry

end find_value_of_y_l112_112612


namespace ellipsoid_center_and_axes_sum_l112_112119

theorem ellipsoid_center_and_axes_sum :
  let x₀ := -2
  let y₀ := 3
  let z₀ := 1
  let A := 6
  let B := 4
  let C := 2
  x₀ + y₀ + z₀ + A + B + C = 14 := 
by
  sorry

end ellipsoid_center_and_axes_sum_l112_112119


namespace functional_equation_solution_l112_112158

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) :=
sorry

end functional_equation_solution_l112_112158


namespace min_value_four_x_plus_one_over_x_l112_112883

theorem min_value_four_x_plus_one_over_x (x : ℝ) (hx : x > 0) : 4*x + 1/x ≥ 4 := by
  sorry

end min_value_four_x_plus_one_over_x_l112_112883


namespace probability_no_order_l112_112267

theorem probability_no_order (P : ℕ) 
  (h1 : 60 ≤ 100) (h2 : 10 ≤ 100) (h3 : 15 ≤ 100) 
  (h4 : 5 ≤ 100) (h5 : 3 ≤ 100) (h6 : 2 ≤ 100) :
  P = 100 - (60 + 10 + 15 + 5 + 3 + 2) :=
by 
  sorry

end probability_no_order_l112_112267


namespace pages_to_read_tomorrow_l112_112992

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l112_112992


namespace total_legs_l112_112496

def total_heads : ℕ := 16
def num_cats : ℕ := 7
def cat_legs : ℕ := 4
def captain_legs : ℕ := 1
def human_legs : ℕ := 2

theorem total_legs : (num_cats * cat_legs + (total_heads - num_cats) * human_legs - human_legs + captain_legs) = 45 :=
by 
  -- Proof skipped
  sorry

end total_legs_l112_112496


namespace factorization_correct_l112_112369

theorem factorization_correct :
  ∀ (m a b x y : ℝ), 
    (m^2 - 4 = (m + 2) * (m - 2)) ∧
    ((a + 3) * (a - 3) = a^2 - 9) ∧
    (a^2 - b^2 + 1 = (a + b) * (a - b) + 1) ∧
    (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3) →
    (m^2 - 4 = (m + 2) * (m - 2)) :=
by
  intros m a b x y h
  have ⟨hA, hB, hC, hD⟩ := h
  exact hA

end factorization_correct_l112_112369


namespace pier_to_village_trip_l112_112697

theorem pier_to_village_trip :
  ∃ (x t : ℝ), 
  (x / 10 + x / 8 = t + 1 / 60) ∧
  (5 * t / 2 + 4 * t / 2 = x) ∧
  (x = 6) ∧
  (t = 4 / 3) :=
by
  sorry

end pier_to_village_trip_l112_112697


namespace intersection_points_l112_112373

variables {α β : Type*} [DecidableEq α] {f : α → β} {x m : α}

theorem intersection_points (dom : α → Prop) (h : dom x → ∃! y, f x = y) : 
  (∃ y, f m = y) ∨ ¬ ∃ y, f m = y :=
by
  sorry

end intersection_points_l112_112373


namespace only_number_smaller_than_zero_l112_112370

theorem only_number_smaller_than_zero : ∀ (x : ℝ), (x = 5 ∨ x = 2 ∨ x = 0 ∨ x = -Real.sqrt 2) → x < 0 → x = -Real.sqrt 2 :=
by
  intro x hx h
  sorry

end only_number_smaller_than_zero_l112_112370


namespace train_length_is_900_l112_112855

def train_length_crossing_pole (L V : ℕ) : Prop :=
  L = V * 18

def train_length_crossing_platform (L V : ℕ) : Prop :=
  L + 1050 = V * 39

theorem train_length_is_900 (L V : ℕ) (h1 : train_length_crossing_pole L V) (h2 : train_length_crossing_platform L V) : L = 900 := 
by
  sorry

end train_length_is_900_l112_112855


namespace george_earnings_after_deductions_l112_112864

noncomputable def george_total_earnings : ℕ := 35 + 12 + 20 + 21

noncomputable def tax_deduction (total_earnings : ℕ) : ℚ := total_earnings * 0.10

noncomputable def uniform_fee : ℚ := 15

noncomputable def final_earnings (total_earnings : ℕ) (tax_deduction : ℚ) (uniform_fee : ℚ) : ℚ :=
  total_earnings - tax_deduction - uniform_fee

theorem george_earnings_after_deductions : 
  final_earnings george_total_earnings (tax_deduction george_total_earnings) uniform_fee = 64.2 := 
  by
  sorry

end george_earnings_after_deductions_l112_112864


namespace kolacky_bounds_l112_112209

theorem kolacky_bounds (x y : ℕ) (h : 9 * x + 4 * y = 219) :
  294 ≤ 12 * x + 6 * y ∧ 12 * x + 6 * y ≤ 324 :=
sorry

end kolacky_bounds_l112_112209


namespace lcm_of_36_48_75_l112_112029

-- Definitions of the numbers and their factorizations
def num1 := 36
def num2 := 48
def num3 := 75

def factor_36 := (2^2, 3^2)
def factor_48 := (2^4, 3^1)
def factor_75 := (3^1, 5^2)

def highest_power_2 := 2^4
def highest_power_3 := 3^2
def highest_power_5 := 5^2

def lcm_36_48_75 := highest_power_2 * highest_power_3 * highest_power_5

-- The theorem statement
theorem lcm_of_36_48_75 : lcm_36_48_75 = 3600 := by
  sorry

end lcm_of_36_48_75_l112_112029


namespace not_associative_star_l112_112488

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

theorem not_associative_star : ¬ (∀ x y z : ℝ, star (star x y) z = star x (star y z)) :=
by
  sorry

end not_associative_star_l112_112488


namespace half_radius_of_circle_y_l112_112515

theorem half_radius_of_circle_y
  (r_x r_y : ℝ)
  (hx : π * r_x ^ 2 = π * r_y ^ 2)
  (hc : 2 * π * r_x = 10 * π) :
  r_y / 2 = 2.5 :=
by
  sorry

end half_radius_of_circle_y_l112_112515


namespace one_over_a5_eq_30_l112_112686

noncomputable def S : ℕ → ℝ
| n => n / (n + 1)

noncomputable def a (n : ℕ) := if n = 0 then S 0 else S n - S (n - 1)

theorem one_over_a5_eq_30 :
  (1 / a 5) = 30 :=
by
  sorry

end one_over_a5_eq_30_l112_112686


namespace total_students_l112_112668

-- Define the conditions
def students_in_front : Nat := 7
def position_from_back : Nat := 6

-- Define the proof problem
theorem total_students : (students_in_front + 1 + (position_from_back - 1)) = 13 := by
  -- Proof steps will go here (use sorry to skip for now)
  sorry

end total_students_l112_112668


namespace isosceles_triangle_angle_split_l112_112272

theorem isosceles_triangle_angle_split (A B C1 C2 : ℝ)
  (h_isosceles : A = B)
  (h_greater_than_third : A > C1)
  (h_split : C1 + C2 = C) :
  C1 = C2 :=
sorry

end isosceles_triangle_angle_split_l112_112272


namespace product_of_five_consecutive_integers_not_square_l112_112615

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : a > 0) :
  ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l112_112615


namespace emily_needs_375_nickels_for_book_l112_112003

theorem emily_needs_375_nickels_for_book
  (n : ℕ)
  (book_cost : ℝ)
  (five_dollars : ℝ)
  (one_dollars : ℝ)
  (quarters : ℝ)
  (nickel_value : ℝ)
  (total_money : ℝ)
  (h1 : book_cost = 46.25)
  (h2 : five_dollars = 4 * 5)
  (h3 : one_dollars = 5 * 1)
  (h4 : quarters = 10 * 0.25)
  (h5 : nickel_value = n * 0.05)
  (h6 : total_money = five_dollars + one_dollars + quarters + nickel_value) 
  (h7 : total_money ≥ book_cost) :
  n ≥ 375 :=
by 
  sorry

end emily_needs_375_nickels_for_book_l112_112003


namespace beef_weight_loss_percentage_l112_112317

noncomputable def weight_after_processing : ℝ := 570
noncomputable def weight_before_processing : ℝ := 876.9230769230769

theorem beef_weight_loss_percentage :
  (weight_before_processing - weight_after_processing) / weight_before_processing * 100 = 35 :=
by
  sorry

end beef_weight_loss_percentage_l112_112317


namespace sum_of_consecutive_even_integers_divisible_by_three_l112_112750

theorem sum_of_consecutive_even_integers_divisible_by_three (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p = 3 ∧ p ∣ (n + (n + 2) + (n + 4)) :=
by 
  sorry

end sum_of_consecutive_even_integers_divisible_by_three_l112_112750


namespace seating_arrangements_total_l112_112779

def num_round_tables := 3
def num_rect_tables := 4
def num_square_tables := 2
def num_couches := 2
def num_benches := 3
def num_extra_chairs := 5

def seats_per_round_table := 6
def seats_per_rect_table := 7
def seats_per_square_table := 4
def seats_per_couch := 3
def seats_per_bench := 5

def total_seats : Nat :=
  num_round_tables * seats_per_round_table +
  num_rect_tables * seats_per_rect_table +
  num_square_tables * seats_per_square_table +
  num_couches * seats_per_couch +
  num_benches * seats_per_bench +
  num_extra_chairs

theorem seating_arrangements_total :
  total_seats = 80 :=
by
  simp [total_seats, num_round_tables, seats_per_round_table,
        num_rect_tables, seats_per_rect_table, num_square_tables,
        seats_per_square_table, num_couches, seats_per_couch,
        num_benches, seats_per_bench, num_extra_chairs]
  done

end seating_arrangements_total_l112_112779


namespace mars_colony_cost_l112_112788

theorem mars_colony_cost :
  let total_cost := 45000000000
  let number_of_people := 300000000
  total_cost / number_of_people = 150 := 
by sorry

end mars_colony_cost_l112_112788


namespace bus_people_difference_l112_112508

theorem bus_people_difference 
  (initial : ℕ) (got_off : ℕ) (got_on : ℕ) (current : ℕ) 
  (h_initial : initial = 35)
  (h_got_off : got_off = 18)
  (h_got_on : got_on = 15)
  (h_current : current = initial - got_off + got_on) :
  initial - current = 3 := by
  sorry

end bus_people_difference_l112_112508


namespace maximum_marks_l112_112587

theorem maximum_marks (M : ℝ) (h : 0.5 * M = 50 + 10) : M = 120 :=
by
  sorry

end maximum_marks_l112_112587


namespace Xiaolong_dad_age_correct_l112_112637
noncomputable def Xiaolong_age (x : ℕ) : ℕ := x
noncomputable def mom_age (x : ℕ) : ℕ := 9 * x
noncomputable def dad_age (x : ℕ) : ℕ := 9 * x + 3
noncomputable def dad_age_next_year (x : ℕ) : ℕ := 9 * x + 4
noncomputable def Xiaolong_age_next_year (x : ℕ) : ℕ := x + 1
noncomputable def dad_age_predicated_next_year (x : ℕ) : ℕ := 8 * (x + 1)

theorem Xiaolong_dad_age_correct (x : ℕ) (h : 9 * x + 4 = 8 * (x + 1)) : dad_age x = 39 := by
  sorry

end Xiaolong_dad_age_correct_l112_112637


namespace band_members_minimum_n_l112_112860

theorem band_members_minimum_n 
  (n : ℕ) 
  (h1 : n % 6 = 3) 
  (h2 : n % 8 = 5) 
  (h3 : n % 9 = 7) : 
  n ≥ 165 := 
sorry

end band_members_minimum_n_l112_112860


namespace angle_measure_triple_complement_l112_112067

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l112_112067


namespace unique_diff_subset_l112_112874

noncomputable def exists_unique_diff_subset : Prop :=
  ∃ S : Set ℕ, 
    (∀ n : ℕ, n > 0 → ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ n = a - b)

theorem unique_diff_subset : exists_unique_diff_subset :=
  sorry

end unique_diff_subset_l112_112874


namespace smallest_integer_y_l112_112121

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end smallest_integer_y_l112_112121


namespace smallest_positive_integer_l112_112556

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 60606 * n = 273 :=
sorry

end smallest_positive_integer_l112_112556


namespace ratio_of_red_to_blue_marbles_l112_112967

theorem ratio_of_red_to_blue_marbles:
  ∀ (R B : ℕ), 
    R + B = 30 →
    2 * (20 - B) = 10 →
    B = 15 → 
    R = 15 →
    R / B = 1 :=
by intros R B h₁ h₂ h₃ h₄
   sorry

end ratio_of_red_to_blue_marbles_l112_112967


namespace max_men_with_all_items_l112_112604

theorem max_men_with_all_items (total_men married men_with_TV men_with_radio men_with_AC men_with_car men_with_smartphone : ℕ) 
  (H_married : married = 2300) 
  (H_TV : men_with_TV = 2100) 
  (H_radio : men_with_radio = 2600) 
  (H_AC : men_with_AC = 1800) 
  (H_car : men_with_car = 2500) 
  (H_smartphone : men_with_smartphone = 2200) : 
  ∃ m, m ≤ married ∧ m ≤ men_with_TV ∧ m ≤ men_with_radio ∧ m ≤ men_with_AC ∧ m ≤ men_with_car ∧ m ≤ men_with_smartphone ∧ m = 1800 := 
  sorry

end max_men_with_all_items_l112_112604


namespace complex_exponential_to_rectangular_form_l112_112368

theorem complex_exponential_to_rectangular_form :
  Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) = -1 - Complex.I := by
  -- Proof will go here
  sorry

end complex_exponential_to_rectangular_form_l112_112368


namespace solve_k_l112_112009

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end solve_k_l112_112009


namespace boys_girls_ratio_l112_112629

-- Definitions used as conditions
variable (B G : ℕ)

-- Conditions
def condition1 : Prop := B + G = 32
def condition2 : Prop := B = 2 * (G - 8)

-- Proof that the ratio of boys to girls initially is 1:1
theorem boys_girls_ratio (h1 : condition1 B G) (h2 : condition2 B G) : (B : ℚ) / G = 1 := by
  sorry

end boys_girls_ratio_l112_112629


namespace trains_meet_in_2067_seconds_l112_112844

def length_of_train1 : ℝ := 100  -- Length of Train 1 in meters
def length_of_train2 : ℝ := 200  -- Length of Train 2 in meters
def initial_distance : ℝ := 630  -- Initial distance between trains in meters
def speed_of_train1_kmh : ℝ := 90  -- Speed of Train 1 in km/h
def speed_of_train2_kmh : ℝ := 72  -- Speed of Train 2 in km/h

noncomputable def speed_of_train1_ms : ℝ := speed_of_train1_kmh * (1000 / 3600)
noncomputable def speed_of_train2_ms : ℝ := speed_of_train2_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := speed_of_train1_ms + speed_of_train2_ms
noncomputable def total_distance : ℝ := initial_distance + length_of_train1 + length_of_train2
noncomputable def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_2067_seconds : time_to_meet = 20.67 := 
by
  sorry

end trains_meet_in_2067_seconds_l112_112844


namespace distance_between_parallel_lines_l112_112281

class ParallelLines (A B c1 c2 : ℝ)

theorem distance_between_parallel_lines (A B c1 c2 : ℝ)
  [h : ParallelLines A B c1 c2] : 
  A = 4 → B = 3 → c1 = 1 → c2 = -9 → 
  (|c1 - c2| / Real.sqrt (A^2 + B^2)) = 2 :=
by
  intros hA hB hc1 hc2
  rw [hA, hB, hc1, hc2]
  norm_num
  sorry

end distance_between_parallel_lines_l112_112281


namespace max_blocks_fit_l112_112840

-- Defining the dimensions of the box and blocks
def box_length : ℝ := 4
def box_width : ℝ := 3
def box_height : ℝ := 2

def block_length : ℝ := 3
def block_width : ℝ := 1
def block_height : ℝ := 1

-- Theorem stating the maximum number of blocks that fit
theorem max_blocks_fit : (24 / 3 = 8) ∧ (1 * 3 * 2 = 6) → 6 = 6 := 
by
  sorry

end max_blocks_fit_l112_112840


namespace determine_g1_l112_112812

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x^2 * y - x^3 + 1)

theorem determine_g1 : g 1 = 2 := sorry

end determine_g1_l112_112812


namespace least_possible_square_area_l112_112471

theorem least_possible_square_area (s : ℝ) (h1 : 4.5 ≤ s) (h2 : s < 5.5) : s * s ≥ 20.25 := by
  sorry

end least_possible_square_area_l112_112471


namespace largest_composite_not_written_l112_112877

theorem largest_composite_not_written (n : ℕ) (hn : n = 2022) : ¬ ∃ d > 1, 2033 = n + d := 
by
  sorry

end largest_composite_not_written_l112_112877


namespace solve_equation_l112_112803

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2/3 → (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2)) →
  (∀ x : ℝ, x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3) :=
by
  sorry

end solve_equation_l112_112803


namespace range_of_x_l112_112598

-- Problem Statement
theorem range_of_x (x : ℝ) (h : 0 ≤ x - 8) : 8 ≤ x :=
by {
  sorry
}

end range_of_x_l112_112598


namespace smallest_number_of_weights_l112_112463

/-- The smallest number of weights in a set that can be divided into 4, 5, and 6 equal piles is 11. -/
theorem smallest_number_of_weights (n : ℕ) (M : ℕ) : (∀ k : ℕ, (k = 4 ∨ k = 5 ∨ k = 6) → M % k = 0) → n = 11 :=
sorry

end smallest_number_of_weights_l112_112463


namespace yellow_surface_area_fraction_minimal_l112_112161

theorem yellow_surface_area_fraction_minimal 
  (total_cubes : ℕ)
  (edge_length : ℕ)
  (yellow_cubes : ℕ)
  (blue_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (yellow_fraction : ℚ) :
  total_cubes = 64 ∧
  edge_length = 4 ∧
  yellow_cubes = 16 ∧
  blue_cubes = 48 ∧
  total_surface_area = 6 * edge_length * edge_length ∧
  yellow_surface_area = 15 →
  yellow_fraction = (yellow_surface_area : ℚ) / total_surface_area :=
sorry

end yellow_surface_area_fraction_minimal_l112_112161


namespace min_x_plus_y_l112_112259

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y = 9 :=
sorry

end min_x_plus_y_l112_112259


namespace percentage_increase_after_decrease_l112_112772

theorem percentage_increase_after_decrease (P : ℝ) :
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  (P_decreased * (1 + x / 100) = P_final) → x = 65.71 := 
by 
  intros
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  have h : (P_decreased * (1 + x / 100) = P_final) := by assumption
  sorry

end percentage_increase_after_decrease_l112_112772


namespace soda_cost_l112_112100

theorem soda_cost (total_cost sandwich_price : ℝ) (num_sandwiches num_sodas : ℕ) (total : total_cost = 8.38)
  (sandwich_cost : sandwich_price = 2.45) (total_sandwiches : num_sandwiches = 2) (total_sodas : num_sodas = 4) :
  ((total_cost - (num_sandwiches * sandwich_price)) / num_sodas) = 0.87 :=
by
  sorry

end soda_cost_l112_112100


namespace win_sector_area_l112_112701

noncomputable def radius : ℝ := 8
noncomputable def probability : ℝ := 1 / 4
noncomputable def total_area : ℝ := Real.pi * radius^2

theorem win_sector_area :
  ∃ (W : ℝ), W = probability * total_area ∧ W = 16 * Real.pi :=
by
  -- Proof skipped
  sorry

end win_sector_area_l112_112701


namespace central_angle_radian_measure_l112_112082

namespace SectorProof

variables (R l : ℝ)
variables (α : ℝ)

-- Given conditions
def condition1 : Prop := 2 * R + l = 20
def condition2 : Prop := 1 / 2 * l * R = 9
def α_definition : Prop := α = l / R

-- Central angle result
theorem central_angle_radian_measure (h1 : condition1 R l) (h2 : condition2 R l) :
  α_definition α l R → α = 2 / 9 :=
by
  intro h_α
  -- proof steps would be here, but we skip them with sorry
  sorry

end SectorProof

end central_angle_radian_measure_l112_112082


namespace sport_tournament_attendance_l112_112798

theorem sport_tournament_attendance :
  let total_attendance := 500
  let team_A_supporters := 0.35 * total_attendance
  let team_B_supporters := 0.25 * total_attendance
  let team_C_supporters := 0.20 * total_attendance
  let team_D_supporters := 0.15 * total_attendance
  let AB_overlap := 0.10 * team_A_supporters
  let BC_overlap := 0.05 * team_B_supporters
  let CD_overlap := 0.07 * team_C_supporters
  let atmosphere_attendees := 30
  let total_supporters := team_A_supporters + team_B_supporters + team_C_supporters + team_D_supporters
                         - (AB_overlap + BC_overlap + CD_overlap)
  let unsupported_people := total_attendance - total_supporters - atmosphere_attendees
  unsupported_people = 26 :=
by
  sorry

end sport_tournament_attendance_l112_112798


namespace parabola_distance_l112_112839

theorem parabola_distance (a : ℝ) :
  (abs (1 + (1 / (4 * a))) = 2 → a = 1 / 4) ∨ 
  (abs (1 - (1 / (4 * a))) = 2 → a = -1 / 12) := by 
  sorry

end parabola_distance_l112_112839


namespace innings_played_l112_112245

noncomputable def cricket_player_innings : Nat :=
  let average_runs := 32
  let increase_in_average := 6
  let next_innings_runs := 158
  let new_average := average_runs + increase_in_average
  let runs_before_next_innings (n : Nat) := average_runs * n
  let total_runs_after_next_innings (n : Nat) := runs_before_next_innings n + next_innings_runs
  let total_runs_with_new_average (n : Nat) := new_average * (n + 1)

  let n := (total_runs_after_next_innings 20) - (total_runs_with_new_average 20)
  
  n
     
theorem innings_played : cricket_player_innings = 20 := by
  sorry

end innings_played_l112_112245


namespace quadratic_example_correct_l112_112075

-- Define the quadratic function
def quad_func (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- Conditions defined
def condition1 := quad_func 1 = 0
def condition2 := quad_func 5 = 0
def condition3 := quad_func 3 = 8

-- Theorem statement combining the conditions
theorem quadratic_example_correct :
  condition1 ∧ condition2 ∧ condition3 :=
by
  -- Proof omitted as per instructions
  sorry

end quadratic_example_correct_l112_112075


namespace robin_hid_150_seeds_l112_112815

theorem robin_hid_150_seeds
    (x y : ℕ)
    (h1 : 5 * x = 6 * y)
    (h2 : y = x - 5) : 
    5 * x = 150 :=
by
    sorry

end robin_hid_150_seeds_l112_112815
