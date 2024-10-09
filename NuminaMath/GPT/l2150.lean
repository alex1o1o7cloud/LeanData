import Mathlib

namespace value_of_x2_minus_y2_l2150_215093

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x2_minus_y2_l2150_215093


namespace find_sum_of_abcd_l2150_215033

theorem find_sum_of_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) :
  a + b + c + d = -26 / 3 :=
sorry

end find_sum_of_abcd_l2150_215033


namespace combined_rate_last_year_l2150_215085

noncomputable def combine_effective_rate_last_year (r_increased: ℝ) (r_this_year: ℝ) : ℝ :=
  r_this_year / r_increased

theorem combined_rate_last_year
  (compounding_frequencies : List String)
  (r_increased : ℝ)
  (r_this_year : ℝ)
  (combined_interest_rate_this_year : r_this_year = 0.11)
  (interest_rate_increase : r_increased = 1.10) :
  combine_effective_rate_last_year r_increased r_this_year = 0.10 :=
by
  sorry

end combined_rate_last_year_l2150_215085


namespace determine_m_l2150_215081

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem determine_m (m : ℝ) : 3 * f 5 m = 2 * g 5 m → m = 10 / 7 := 
by sorry

end determine_m_l2150_215081


namespace arithmetic_sequence_a2_a9_l2150_215031

theorem arithmetic_sequence_a2_a9 (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 5 + a 6 = 12) :
  a 2 + a 9 = 12 :=
sorry

end arithmetic_sequence_a2_a9_l2150_215031


namespace functional_equation_solution_l2150_215099

variable (f : ℝ → ℝ)

-- Declare the conditions as hypotheses
axiom cond1 : ∀ x : ℝ, 0 < x → 0 < f x
axiom cond2 : f 1 = 1
axiom cond3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

-- State the theorem to be proved
theorem functional_equation_solution : ∀ x : ℝ, f x = x :=
sorry

end functional_equation_solution_l2150_215099


namespace identity_function_uniq_l2150_215079

theorem identity_function_uniq (f g h : ℝ → ℝ)
    (hg : ∀ x, g x = x + 1)
    (hh : ∀ x, h x = x^2)
    (H1 : ∀ x, f (g x) = g (f x))
    (H2 : ∀ x, f (h x) = h (f x)) :
  ∀ x, f x = x :=
by
  sorry

end identity_function_uniq_l2150_215079


namespace range_of_function_l2150_215094

open Real

noncomputable def f (x : ℝ) : ℝ := -cos x ^ 2 - 4 * sin x + 6

theorem range_of_function : 
  ∀ y, (∃ x, y = f x) ↔ 2 ≤ y ∧ y ≤ 10 :=
by
  sorry

end range_of_function_l2150_215094


namespace daily_reading_goal_l2150_215086

-- Define the constants for pages read each day
def pages_on_sunday : ℕ := 43
def pages_on_monday : ℕ := 65
def pages_on_tuesday : ℕ := 28
def pages_on_wednesday : ℕ := 0
def pages_on_thursday : ℕ := 70
def pages_on_friday : ℕ := 56
def pages_on_saturday : ℕ := 88

-- Define the total pages read in the week
def total_pages := pages_on_sunday + pages_on_monday + pages_on_tuesday + pages_on_wednesday 
                    + pages_on_thursday + pages_on_friday + pages_on_saturday

-- The theorem that expresses Berry's daily reading goal
theorem daily_reading_goal : total_pages / 7 = 50 :=
by
  sorry

end daily_reading_goal_l2150_215086


namespace solve_for_n_l2150_215032

theorem solve_for_n (n : ℕ) (h : 2 * n - 5 = 1) : n = 3 :=
by
  sorry

end solve_for_n_l2150_215032


namespace arithmetic_sequence_50th_term_l2150_215058

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

end arithmetic_sequence_50th_term_l2150_215058


namespace find_v₃_value_l2150_215087

def f (x : ℕ) : ℕ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def v₃_expr (x : ℕ) : ℕ := (((7 * x + 6) * x + 5) * x + 4)

theorem find_v₃_value : v₃_expr 3 = 262 := by
  sorry

end find_v₃_value_l2150_215087


namespace axis_of_symmetry_parabola_l2150_215066

theorem axis_of_symmetry_parabola (x y : ℝ) :
  x^2 + 2*x*y + y^2 + 3*x + y = 0 → x + y + 1 = 0 :=
by {
  sorry
}

end axis_of_symmetry_parabola_l2150_215066


namespace next_divisor_after_391_l2150_215050

theorem next_divisor_after_391 (m : ℕ) (h1 : m % 2 = 0) (h2 : m ≥ 1000 ∧ m < 10000) (h3 : 391 ∣ m) : 
  ∃ n, n > 391 ∧ n ∣ m ∧ (∀ k, k > 391 ∧ k < n → ¬ k ∣ m) ∧ n = 782 :=
sorry

end next_divisor_after_391_l2150_215050


namespace prime_product_solution_l2150_215098

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_product_solution (p_1 p_2 p_3 p_4 : ℕ) :
  is_prime p_1 ∧ is_prime p_2 ∧ is_prime p_3 ∧ is_prime p_4 ∧ 
  p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_1 ≠ p_4 ∧ p_2 ≠ p_3 ∧ p_2 ≠ p_4 ∧ p_3 ≠ p_4 ∧
  2 * p_1 + 3 * p_2 + 5 * p_3 + 7 * p_4 = 162 ∧
  11 * p_1 + 7 * p_2 + 5 * p_3 + 4 * p_4 = 162 
  → p_1 * p_2 * p_3 * p_4 = 570 := 
by
  sorry

end prime_product_solution_l2150_215098


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l2150_215012

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

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l2150_215012


namespace comparison_of_negatives_l2150_215014

theorem comparison_of_negatives : -2 < - (3 / 2) :=
by
  sorry

end comparison_of_negatives_l2150_215014


namespace ball_bounce_height_l2150_215019

theorem ball_bounce_height (initial_height : ℝ) (r : ℝ) (k : ℕ) : 
  initial_height = 1000 → r = 1/2 → (r ^ k * initial_height < 1) → k = 10 := by
sorry

end ball_bounce_height_l2150_215019


namespace smallest_M_convex_quadrilateral_l2150_215077

section ConvexQuadrilateral

-- Let a, b, c, d be the sides of a convex quadrilateral
variables {a b c d M : ℝ}

-- Condition to ensure that a, b, c, d are the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d < 360

-- The theorem statement
theorem smallest_M_convex_quadrilateral (hconvex : is_convex_quadrilateral a b c d) : ∃ M, (∀ a b c d, is_convex_quadrilateral a b c d → (a^2 + b^2) / (c^2 + d^2) > M) ∧ M = 1/2 :=
by sorry

end ConvexQuadrilateral

end smallest_M_convex_quadrilateral_l2150_215077


namespace jasmine_total_cost_l2150_215060

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

end jasmine_total_cost_l2150_215060


namespace average_person_funding_l2150_215051

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

end average_person_funding_l2150_215051


namespace radius_of_spheres_in_cone_l2150_215040

-- Given Definitions
def cone_base_radius : ℝ := 6
def cone_height : ℝ := 15
def tangent_spheres (r : ℝ) : Prop :=
  r = (12 * Real.sqrt 29) / 29

-- Problem Statement
theorem radius_of_spheres_in_cone :
  ∃ r : ℝ, tangent_spheres r :=
sorry

end radius_of_spheres_in_cone_l2150_215040


namespace bridge_must_hold_weight_l2150_215044

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

end bridge_must_hold_weight_l2150_215044


namespace net_investment_change_l2150_215041

def initial_investment : ℝ := 100
def first_year_increase (init : ℝ) : ℝ := init * 1.50
def second_year_decrease (value : ℝ) : ℝ := value * 0.70

theorem net_investment_change :
  second_year_decrease (first_year_increase initial_investment) - initial_investment = 5 :=
by
  -- This will be placeholder proof
  sorry

end net_investment_change_l2150_215041


namespace napkins_total_l2150_215016

theorem napkins_total (o a w : ℕ) (ho : o = 10) (ha : a = 2 * o) (hw : w = 15) :
  w + o + a = 45 :=
by
  sorry

end napkins_total_l2150_215016


namespace union_of_sets_l2150_215039

def A : Set ℤ := {0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets :
  A ∪ B = {0, 1, 2} :=
by
  sorry

end union_of_sets_l2150_215039


namespace convex_over_real_l2150_215055

def f (x : ℝ) : ℝ := x^4 - 2 * x^3 + 36 * x^2 - x + 7

theorem convex_over_real : ∀ x : ℝ, 0 ≤ (12 * x^2 - 12 * x + 72) :=
by sorry

end convex_over_real_l2150_215055


namespace three_x_plus_three_y_plus_three_z_l2150_215095

theorem three_x_plus_three_y_plus_three_z (x y z : ℝ) 
  (h1 : y + z = 20 - 5 * x)
  (h2 : x + z = -18 - 5 * y)
  (h3 : x + y = 10 - 5 * z) :
  3 * x + 3 * y + 3 * z = 36 / 7 := by
  sorry

end three_x_plus_three_y_plus_three_z_l2150_215095


namespace fraction_of_pelicans_moved_l2150_215064

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

end fraction_of_pelicans_moved_l2150_215064


namespace incorrect_statement_l2150_215082

def geom_seq (a r : ℝ) : ℕ → ℝ
| 0       => a
| (n + 1) => r * geom_seq a r n

theorem incorrect_statement
  (a : ℝ) (r : ℝ) (S6 : ℝ)
  (h1 : r = 1 / 2)
  (h2 : S6 = a * (1 - (1 / 2) ^ 6) / (1 - 1 / 2))
  (h3 : S6 = 378) :
  geom_seq a r 2 / S6 ≠ 1 / 8 :=
by 
  have h4 : a = 192 := by sorry
  have h5 : geom_seq 192 (1 / 2) 2 = 192 * (1 / 2) ^ 2 := by sorry
  exact sorry

end incorrect_statement_l2150_215082


namespace f_increasing_on_interval_l2150_215059

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

end f_increasing_on_interval_l2150_215059


namespace frequency_of_2_l2150_215009

def num_set := "20231222"
def total_digits := 8
def count_of_2 := 5

theorem frequency_of_2 : (count_of_2 : ℚ) / total_digits = 5 / 8 := by
  sorry

end frequency_of_2_l2150_215009


namespace leak_drain_time_l2150_215065

/-- Statement: Given the rates at which a pump fills a tank and a leak drains the tank, 
prove that the leak can drain all the water in the tank in 14 hours. -/
theorem leak_drain_time :
  (∀ P L: ℝ, P = 1/2 → (P - L) = 3/7 → L = 1/14 → (1 / L) = 14) := 
by
  intros P L hP hPL hL
  -- Proof is omitted (to be provided)
  sorry

end leak_drain_time_l2150_215065


namespace find_angle_A_l2150_215034

-- Variables representing angles A and B
variables (A B : ℝ)

-- The conditions of the problem translated into Lean
def angle_relationship := A = 2 * B - 15
def angle_supplementary := A + B = 180

-- The theorem statement we need to prove
theorem find_angle_A (h1 : angle_relationship A B) (h2 : angle_supplementary A B) : A = 115 :=
by { sorry }

end find_angle_A_l2150_215034


namespace sqrt_condition_iff_l2150_215010

theorem sqrt_condition_iff (x : ℝ) : (∃ y : ℝ, y = (2 * x + 3) ∧ (0 ≤ y)) ↔ (x ≥ -3 / 2) :=
by sorry

end sqrt_condition_iff_l2150_215010


namespace certain_number_is_2_l2150_215084

theorem certain_number_is_2 
    (X : ℕ) 
    (Y : ℕ) 
    (h1 : X = 15) 
    (h2 : 0.40 * (X : ℝ) = 0.80 * 5 + (Y : ℝ)) : 
    Y = 2 := 
  sorry

end certain_number_is_2_l2150_215084


namespace collinear_c1_c2_l2150_215029

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 7, 0)
def b : ℝ × ℝ × ℝ := (1, -3, 4)

-- Define the vectors c1 and c2 based on a and b
def c1 : ℝ × ℝ × ℝ := (4 * 3, 4 * 7, 4 * 0) - (2 * 1, 2 * -3, 2 * 4)
def c2 : ℝ × ℝ × ℝ := (1, -3, 4) - (2 * 3, 2 * 7, 2 * 0)

-- The theorem to prove that c1 and c2 are collinear
theorem collinear_c1_c2 : c1 = (-2 : ℝ) • c2 := by sorry

end collinear_c1_c2_l2150_215029


namespace brown_beads_initial_l2150_215045

theorem brown_beads_initial (B : ℕ) 
  (h1 : 1 = 1) -- There is 1 green bead in the container.
  (h2 : 3 = 3) -- There are 3 red beads in the container.
  (h3 : 4 = 4) -- Tom left 4 beads in the container.
  (h4 : 2 = 2) -- Tom took out 2 beads.
  (h5 : 6 = 2 + 4) -- Total initial beads before Tom took any out.
  : B = 2 := sorry

end brown_beads_initial_l2150_215045


namespace child_is_late_l2150_215018

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

end child_is_late_l2150_215018


namespace problem_statement_l2150_215038

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := x^n

variable (a : ℝ)
variable (h : a ≠ 1)

theorem problem_statement :
  (f 11 (f 13 a)) ^ 14 = f 2002 a ∧
  f 11 (f 13 (f 14 a)) = f 2002 a :=
by
  sorry

end problem_statement_l2150_215038


namespace onion_harvest_scientific_notation_l2150_215008

theorem onion_harvest_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 325000000 = a * 10^n ∧ a = 3.25 ∧ n = 8 := 
by
  sorry

end onion_harvest_scientific_notation_l2150_215008


namespace order_of_abcd_l2150_215035

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

end order_of_abcd_l2150_215035


namespace original_number_l2150_215092

theorem original_number (N : ℤ) : (∃ k : ℤ, N - 7 = 12 * k) → N = 19 :=
by
  intros h
  sorry

end original_number_l2150_215092


namespace line_perpendicular_to_plane_implies_parallel_l2150_215062

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

end line_perpendicular_to_plane_implies_parallel_l2150_215062


namespace cylinder_height_to_radius_ratio_l2150_215037

theorem cylinder_height_to_radius_ratio (V r h : ℝ) (hV : V = π * r^2 * h) (hS : sorry) :
  h / r = 2 :=
sorry

end cylinder_height_to_radius_ratio_l2150_215037


namespace quadratic_function_inequality_l2150_215080

theorem quadratic_function_inequality
  (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hx1_pos : 0 < x1)
  (hx2_pos : x1 < x2)
  (hy1 : y1 = x1^2 - 1)
  (hy2 : y2 = x2^2 - 1) :
  y1 < y2 := 
sorry

end quadratic_function_inequality_l2150_215080


namespace monotonicity_intervals_inequality_condition_l2150_215063

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + 2 * x + 1)

theorem monotonicity_intervals :
  (∀ x ∈ Set.Iio (-3 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ), 0 > (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioi (-1 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) := sorry

theorem inequality_condition (a : ℝ) : 
  (∀ x > 0, Real.exp x * (x^2 + 2 * x + 1) > a * x^2 + a * x + 1) ↔ a ≤ 3 := sorry

end monotonicity_intervals_inequality_condition_l2150_215063


namespace cosA_value_area_of_triangle_l2150_215073

noncomputable def cosA (a b c : ℝ) (cos_C : ℝ) : ℝ :=
  if (a ≠ 0 ∧ cos_C ≠ 0) then (2 * b - c) * cos_C / a else 1 / 2

noncomputable def area_triangle (a b c : ℝ) (cosA_val : ℝ) : ℝ :=
  let S := a * b * (Real.sqrt (1 - cosA_val ^ 2)) / 2
  S

theorem cosA_value (a b c : ℝ) (cos_C : ℝ) : a * cos_C = (2 * b - c) * (cosA a b c cos_C) → cosA a b c cos_C = 1 / 2 :=
by
  sorry

theorem area_of_triangle (a b c : ℝ) (cos_A : ℝ) (cos_A_proof : a * cos_C = (2 * b - c) * cos_A) (h₀ : a = 6) (h₁ : b + c = 8) : area_triangle a b c cos_A = 7 * Real.sqrt 3 / 3 :=
by
  sorry

end cosA_value_area_of_triangle_l2150_215073


namespace unpainted_unit_cubes_l2150_215067

theorem unpainted_unit_cubes (total_units : ℕ) (painted_per_face : ℕ) (painted_edges_adjustment : ℕ) :
  total_units = 216 → painted_per_face = 12 → painted_edges_adjustment = 36 → 
  total_units - (painted_per_face * 6 - painted_edges_adjustment) = 108 :=
by
  intros h_tot_units h_painted_face h_edge_adj
  sorry

end unpainted_unit_cubes_l2150_215067


namespace find_b_value_l2150_215096

theorem find_b_value (a b c A B C : ℝ) 
  (h1 : a = 1)
  (h2 : B = 120 * (π / 180))
  (h3 : c = b * Real.cos C + c * Real.cos B)
  (h4 : c = 1) : 
  b = Real.sqrt 3 :=
by
  sorry

end find_b_value_l2150_215096


namespace speed_conversion_l2150_215056

theorem speed_conversion (speed_kmph : ℕ) (conversion_rate : ℚ) : (speed_kmph = 600) ∧ (conversion_rate = 0.6) → (speed_kmph * conversion_rate / 60 = 6) :=
by
  sorry

end speed_conversion_l2150_215056


namespace find_solutions_l2150_215000

theorem find_solutions (x y : Real) :
    (x = 1 ∧ y = 2) ∨
    (x = 1 ∧ y = 0) ∨
    (x = -4 ∧ y = 6) ∨
    (x = -5 ∧ y = 2) ∨
    (x = -3 ∧ y = 0) ↔
    x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0 := by
  sorry

end find_solutions_l2150_215000


namespace tetrahedron_edge_length_l2150_215022

theorem tetrahedron_edge_length (a : ℝ) (V : ℝ) 
  (h₀ : V = 0.11785113019775793) 
  (h₁ : V = (Real.sqrt 2 / 12) * a^3) : a = 1 := by
  sorry

end tetrahedron_edge_length_l2150_215022


namespace adam_change_l2150_215028

-- Defining the given amount Adam has and the cost of the airplane.
def amountAdamHas : ℝ := 5.00
def costOfAirplane : ℝ := 4.28

-- Statement of the theorem to be proven.
theorem adam_change : amountAdamHas - costOfAirplane = 0.72 := by
  sorry

end adam_change_l2150_215028


namespace parabola_distance_to_focus_l2150_215005

theorem parabola_distance_to_focus (P : ℝ × ℝ) (y_axis_dist : ℝ) (hx : P.1 = 4) (hy : P.2 ^ 2 = 32) :
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 36 :=
by {
  sorry
}

end parabola_distance_to_focus_l2150_215005


namespace sarah_min_correct_l2150_215075

theorem sarah_min_correct (c : ℕ) (hc : c * 8 + 10 ≥ 110) : c ≥ 13 :=
sorry

end sarah_min_correct_l2150_215075


namespace blocks_given_by_father_l2150_215007

theorem blocks_given_by_father :
  ∀ (blocks_original total_blocks blocks_given : ℕ), 
  blocks_original = 2 →
  total_blocks = 8 →
  blocks_given = total_blocks - blocks_original →
  blocks_given = 6 :=
by
  intros blocks_original total_blocks blocks_given h1 h2 h3
  sorry

end blocks_given_by_father_l2150_215007


namespace domain_of_f_l2150_215011

-- Define the function domain transformation
theorem domain_of_f (f : ℝ → ℝ) : 
  (∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → -7 ≤ 2*x - 3 ∧ 2*x - 3 ≤ 1) ↔ (∀ (y : ℝ), -7 ≤ y ∧ y ≤ 1) :=
sorry

end domain_of_f_l2150_215011


namespace max_singular_words_l2150_215043

theorem max_singular_words (alphabet_length : ℕ) (word_length : ℕ) (strip_length : ℕ) 
  (num_non_overlapping_pieces : ℕ) (h_alphabet : alphabet_length = 25)
  (h_word_length : word_length = 17) (h_strip_length : strip_length = 5^18)
  (h_non_overlapping : num_non_overlapping_pieces = 5^16) : 
  ∃ max_singular_words, max_singular_words = 2 * 5^17 :=
by {
  -- proof to be completed
  sorry
}

end max_singular_words_l2150_215043


namespace percentage_increase_240_to_288_l2150_215061

theorem percentage_increase_240_to_288 :
  let initial := 240
  let final := 288
  ((final - initial) / initial) * 100 = 20 := by 
  sorry

end percentage_increase_240_to_288_l2150_215061


namespace complement_union_in_set_l2150_215026

open Set

theorem complement_union_in_set {U A B : Set ℕ} 
  (hU : U = {1, 3, 5, 9}) 
  (hA : A = {1, 3, 9}) 
  (hB : B = {1, 9}) : 
  (U \ (A ∪ B)) = {5} := 
  by sorry

end complement_union_in_set_l2150_215026


namespace range_of_z_l2150_215083

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 :=
by
  sorry

end range_of_z_l2150_215083


namespace mike_total_earning_l2150_215090

theorem mike_total_earning 
  (first_job : ℕ := 52)
  (hours : ℕ := 12)
  (wage_per_hour : ℕ := 9) :
  first_job + (hours * wage_per_hour) = 160 :=
by
  sorry

end mike_total_earning_l2150_215090


namespace no_prime_roots_l2150_215078

noncomputable def roots_are_prime (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q

theorem no_prime_roots : 
  ∀ k : ℕ, ¬ (∃ p q : ℕ, roots_are_prime p q ∧ p + q = 65 ∧ p * q = k) := 
sorry

end no_prime_roots_l2150_215078


namespace prove_expression_value_l2150_215047

theorem prove_expression_value (x y : ℝ) (h1 : 4 * x + y = 18) (h2 : x + 4 * y = 20) :
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
sorry

end prove_expression_value_l2150_215047


namespace find_real_number_a_l2150_215097

variable (U : Set ℕ) (M : Set ℕ) (a : ℕ)

theorem find_real_number_a :
  U = {1, 3, 5, 7} →
  M = {1, a} →
  (U \ M) = {5, 7} →
  a = 3 :=
by
  intros hU hM hCompU
  -- Proof part will be here
  sorry

end find_real_number_a_l2150_215097


namespace intersecting_point_value_l2150_215036

theorem intersecting_point_value (c d : ℤ) (h1 : d = 5 * (-5) + c) (h2 : -5 = 5 * d + c) : 
  d = -5 := 
sorry

end intersecting_point_value_l2150_215036


namespace WillyLucyHaveMoreCrayons_l2150_215046

-- Definitions from the conditions
def WillyCrayons : ℕ := 1400
def LucyCrayons : ℕ := 290
def MaxCrayons : ℕ := 650

-- Theorem statement
theorem WillyLucyHaveMoreCrayons : WillyCrayons + LucyCrayons - MaxCrayons = 1040 := 
by 
  sorry

end WillyLucyHaveMoreCrayons_l2150_215046


namespace evaluate_expression_when_c_eq_4_and_k_eq_2_l2150_215089

theorem evaluate_expression_when_c_eq_4_and_k_eq_2 :
  ( (4^4 - 4 * (4 - 1)^4 + 2) ^ 4 ) = 18974736 :=
by
  -- Definitions
  let c := 4
  let k := 2
  -- Evaluations
  let a := c^c
  let b := c * (c - 1)^c
  let expression := (a - b + k)^c
  -- Proof
  have result : expression = 18974736 := sorry
  exact result

end evaluate_expression_when_c_eq_4_and_k_eq_2_l2150_215089


namespace number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l2150_215003

section FiveFives

def five : ℕ := 5

-- Definitions for each number 1 to 17 using five fives.
def one : ℕ := (five / five) * (five / five)
def two : ℕ := (five / five) + (five / five)
def three : ℕ := (five * five - five) / five
def four : ℕ := (five - five / five) * (five / five)
def five_num : ℕ := five + (five - five) * (five / five)
def six : ℕ := five + (five + five) / (five + five)
def seven : ℕ := five + (five * five - five^2) / five
def eight : ℕ := (five + five + five) / five + five
def nine : ℕ := five + (five - five / five)
def ten : ℕ := five + five
def eleven : ℕ := (55 - 55 / five) / five
def twelve : ℕ := five * (five - five / five) / five
def thirteen : ℕ := (five * five - five - five) / five + five
def fourteen : ℕ := five + five + five - (five / five)
def fifteen : ℕ := five + five + five
def sixteen : ℕ := five + five + five + (five / five)
def seventeen : ℕ := five + five + five + ((five / five) + (five / five))

-- Proof statements to be provided
theorem number_one : one = 1 := sorry
theorem number_two : two = 2 := sorry
theorem number_three : three = 3 := sorry
theorem number_four : four = 4 := sorry
theorem number_five : five_num = 5 := sorry
theorem number_six : six = 6 := sorry
theorem number_seven : seven = 7 := sorry
theorem number_eight : eight = 8 := sorry
theorem number_nine : nine = 9 := sorry
theorem number_ten : ten = 10 := sorry
theorem number_eleven : eleven = 11 := sorry
theorem number_twelve : twelve = 12 := sorry
theorem number_thirteen : thirteen = 13 := sorry
theorem number_fourteen : fourteen = 14 := sorry
theorem number_fifteen : fifteen = 15 := sorry
theorem number_sixteen : sixteen = 16 := sorry
theorem number_seventeen : seventeen = 17 := sorry

end FiveFives

end number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l2150_215003


namespace correct_option_D_l2150_215049

theorem correct_option_D (a : ℝ) : (-a^3)^2 = a^6 :=
sorry

end correct_option_D_l2150_215049


namespace lily_cups_in_order_l2150_215072

theorem lily_cups_in_order :
  ∀ (rose_rate lily_rate : ℕ) (order_rose_cups total_payment hourly_wage : ℕ),
    rose_rate = 6 →
    lily_rate = 7 →
    order_rose_cups = 6 →
    total_payment = 90 →
    hourly_wage = 30 →
    ∃ lily_cups: ℕ, lily_cups = 14 :=
by
  intros
  sorry

end lily_cups_in_order_l2150_215072


namespace wendy_points_earned_l2150_215053

-- Define the conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def bags_not_recycled : ℕ := 2

-- Define the statement to be proved
theorem wendy_points_earned : (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by
  sorry

end wendy_points_earned_l2150_215053


namespace arithmetic_prog_includes_1999_l2150_215017

-- Definitions based on problem conditions
def is_in_arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_prog_includes_1999
  (d : ℕ) (h_pos : d > 0) 
  (h_includes7 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 7)
  (h_includes15 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 15)
  (h_includes27 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 27) :
  ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 1999 := 
sorry

end arithmetic_prog_includes_1999_l2150_215017


namespace problem_equivalent_l2150_215048

theorem problem_equivalent (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) (hz_eq : z = 10 * y) :
  (x + 4 * y + z) / (4 * x - y - z) = 0 :=
by
  sorry

end problem_equivalent_l2150_215048


namespace solve_system_of_equations_l2150_215074

def system_of_equations(x y z: ℝ): Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

theorem solve_system_of_equations :
  ∀ (x y z: ℝ), system_of_equations x y z ↔
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2 ∨ z = -1) ∨
  (x = -3 ∧ y = 2 ∨ z = 1) :=
by
  sorry

end solve_system_of_equations_l2150_215074


namespace problem1_problem2_l2150_215088

-- Definitions
def vec_a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)
def sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Problem 1: Prove the value of m such that vec_a ⊥ (vec_a - b(m))
theorem problem1 (m : ℝ) (h_perp: dot vec_a (sub vec_a (b m)) = 0) : m = -4 := sorry

-- Problem 2: Prove the value of k such that k * vec_a + b(-4) is parallel to vec_a - b(-4)
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def parallel (u v : ℝ × ℝ) := ∃ (k : ℝ), scale k u = v

theorem problem2 (k : ℝ) (h_parallel: parallel (add (scale k vec_a) (b (-4))) (sub vec_a (b (-4)))) : k = -1 := sorry

end problem1_problem2_l2150_215088


namespace emily_subtracts_99_l2150_215030

theorem emily_subtracts_99 : ∀ (a b : ℕ), (51 * 51 = a + 101) → (49 * 49 = b - 99) → b - 99 = 2401 := by
  intros a b h1 h2
  sorry

end emily_subtracts_99_l2150_215030


namespace fgh_supermarkets_in_us_more_than_canada_l2150_215004

theorem fgh_supermarkets_in_us_more_than_canada
  (total_supermarkets : ℕ)
  (us_supermarkets : ℕ)
  (canada_supermarkets : ℕ)
  (h1 : total_supermarkets = 70)
  (h2 : us_supermarkets = 42)
  (h3 : us_supermarkets + canada_supermarkets = total_supermarkets):
  us_supermarkets - canada_supermarkets = 14 :=
by
  sorry

end fgh_supermarkets_in_us_more_than_canada_l2150_215004


namespace find_b_compare_f_l2150_215052

-- Definition from conditions
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Prove that b = 4
theorem find_b (b c : ℝ) (h : ∀ x : ℝ, f (2 + x) b c = f (2 - x) b c) : b = 4 :=
sorry

-- Part 2: Prove the comparison of f(\frac{5}{4}) and f(-a^2 - a + 1)
theorem compare_f (c : ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (2 + x) 4 c = f (2 - x) 4 c) (h₂ : f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c) :
f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c := 
sorry

end find_b_compare_f_l2150_215052


namespace expression_result_l2150_215006

-- We define the mixed number fractions as conditions
def mixed_num_1 := 2 + 1 / 2         -- 2 1/2
def mixed_num_2 := 3 + 1 / 3         -- 3 1/3
def mixed_num_3 := 4 + 1 / 4         -- 4 1/4
def mixed_num_4 := 1 + 1 / 6         -- 1 1/6

-- Here are their improper fractions
def improper_fraction_1 := 5 / 2     -- (2 + 1/2) converted to improper fraction
def improper_fraction_2 := 10 / 3    -- (3 + 1/3) converted to improper fraction
def improper_fraction_3 := 17 / 4    -- (4 + 1/4) converted to improper fraction
def improper_fraction_4 := 7 / 6     -- (1 + 1/6) converted to improper fraction

-- Define the problematic expression
def expression := (improper_fraction_1 - improper_fraction_2)^2 / (improper_fraction_3 + improper_fraction_4)

-- Statement of the simplified result
theorem expression_result : expression = 5 / 39 :=
by
  sorry

end expression_result_l2150_215006


namespace hunter_ants_l2150_215071

variable (spiders : ℕ) (ladybugs_before : ℕ) (ladybugs_flew : ℕ) (total_insects : ℕ)

theorem hunter_ants (h1 : spiders = 3)
                    (h2 : ladybugs_before = 8)
                    (h3 : ladybugs_flew = 2)
                    (h4 : total_insects = 21) :
  ∃ ants : ℕ, ants = total_insects - (spiders + (ladybugs_before - ladybugs_flew)) ∧ ants = 12 :=
by
  sorry

end hunter_ants_l2150_215071


namespace parabola_tangent_hyperbola_l2150_215021

theorem parabola_tangent_hyperbola (m : ℝ) :
  (∀ x : ℝ, (x^2 + 5)^2 - m * x^2 = 4 → y = x^2 + 5)
  ∧ (∀ y : ℝ, y ≥ 5 → y^2 - m * x^2 = 4) →
  (m = 10 + 2 * Real.sqrt 21 ∨ m = 10 - 2 * Real.sqrt 21) :=
  sorry

end parabola_tangent_hyperbola_l2150_215021


namespace initial_kola_volume_l2150_215042

theorem initial_kola_volume (V : ℝ) (S : ℝ) :
  S = 0.14 * V →
  (S + 3.2) / (V + 20) = 0.14111111111111112 →
  V = 340 :=
by
  intro h_S h_equation
  sorry

end initial_kola_volume_l2150_215042


namespace inequality_proof_l2150_215020

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
    (((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2) ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l2150_215020


namespace kim_spends_time_on_coffee_l2150_215076

noncomputable def time_per_employee_status_update : ℕ := 2
noncomputable def time_per_employee_payroll_update : ℕ := 3
noncomputable def number_of_employees : ℕ := 9
noncomputable def total_morning_routine_time : ℕ := 50

theorem kim_spends_time_on_coffee :
  ∃ C : ℕ, C + (time_per_employee_status_update * number_of_employees) + 
  (time_per_employee_payroll_update * number_of_employees) = total_morning_routine_time ∧
  C = 5 :=
by
  sorry

end kim_spends_time_on_coffee_l2150_215076


namespace range_of_a_l2150_215002

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + (a-1) * x + 1 ≤ 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l2150_215002


namespace mean_is_not_51_l2150_215054

def frequencies : List Nat := [5, 8, 7, 13, 7]
def pH_values : List Float := [4.8, 4.9, 5.0, 5.2, 5.3]

def total_observations : Nat := List.sum frequencies

def mean (freqs : List Nat) (values : List Float) : Float :=
  let weighted_sum := List.sum (List.zipWith (· * ·) values (List.map (Float.ofNat) freqs))
  weighted_sum / (Float.ofNat total_observations)

theorem mean_is_not_51 : mean frequencies pH_values ≠ 5.1 := by
  -- Proof skipped
  sorry

end mean_is_not_51_l2150_215054


namespace chicago_denver_temperature_l2150_215091

def temperature_problem (C D : ℝ) (N : ℝ) : Prop :=
  (C = D - N) ∧ (abs ((D - N + 4) - (D - 2)) = 1)

theorem chicago_denver_temperature (C D N : ℝ) (h : temperature_problem C D N) :
  N = 5 ∨ N = 7 → (5 * 7 = 35) :=
by sorry

end chicago_denver_temperature_l2150_215091


namespace PartA_l2150_215015

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x, f (f x) = f x)

theorem PartA : ∀ x, (deriv f x = 0) ∨ (deriv f (f x) = 1) :=
by
  sorry

end PartA_l2150_215015


namespace height_of_pyramid_equal_to_cube_volume_l2150_215025

theorem height_of_pyramid_equal_to_cube_volume :
  (∃ h : ℝ, (5:ℝ)^3 = (1/3:ℝ) * (10:ℝ)^2 * h) ↔ h = 3.75 :=
by
  sorry

end height_of_pyramid_equal_to_cube_volume_l2150_215025


namespace square_of_binomial_l2150_215023

theorem square_of_binomial (c : ℝ) : (∃ b : ℝ, ∀ x : ℝ, 9 * x^2 - 30 * x + c = (3 * x + b)^2) ↔ c = 25 :=
by
  sorry

end square_of_binomial_l2150_215023


namespace maximum_area_of_sector_l2150_215027

theorem maximum_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 10) : 
  (1 / 2 * l * r) ≤ 25 / 4 := 
sorry

end maximum_area_of_sector_l2150_215027


namespace sufficient_but_not_necessary_l2150_215024

-- Define the conditions
def abs_value_condition (x : ℝ) : Prop := |x| < 2
def quadratic_condition (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Theorem statement
theorem sufficient_but_not_necessary : (∀ x : ℝ, abs_value_condition x → quadratic_condition x) ∧ ¬ (∀ x : ℝ, quadratic_condition x → abs_value_condition x) :=
by
  sorry

end sufficient_but_not_necessary_l2150_215024


namespace minimum_number_of_guests_l2150_215070

theorem minimum_number_of_guests (total_food : ℝ) (max_food_per_guest : ℝ) (H₁ : total_food = 406) (H₂ : max_food_per_guest = 2.5) : 
  ∃ n : ℕ, (n : ℝ) ≥ 163 ∧ total_food / max_food_per_guest ≤ (n : ℝ) := 
by
  sorry

end minimum_number_of_guests_l2150_215070


namespace optimal_strategy_l2150_215057

-- Define the conditions
def valid_N (N : ℤ) : Prop :=
  0 ≤ N ∧ N ≤ 20

def score (N : ℤ) (other_teams_count : ℤ) : ℤ :=
  if other_teams_count > N then N else 0

-- The mathematical problem statement
theorem optimal_strategy : ∃ N : ℤ, valid_N N ∧ (∀ other_teams_count : ℤ, score 1 other_teams_count ≥ score N other_teams_count ∧ score 1 other_teams_count ≠ 0) :=
sorry

end optimal_strategy_l2150_215057


namespace true_discount_is_36_l2150_215001

noncomputable def calc_true_discount (BD SD : ℝ) : ℝ := BD / (1 + BD / SD)

theorem true_discount_is_36 :
  let BD := 42
  let SD := 252
  calc_true_discount BD SD = 36 := 
by
  -- proof here
  sorry

end true_discount_is_36_l2150_215001


namespace value_of_A_l2150_215069

theorem value_of_A (A : ℕ) : (A * 1000 + 567) % 100 < 50 → (A * 1000 + 567) / 10 * 10 = 2560 → A = 2 :=
by
  intro h1 h2
  sorry

end value_of_A_l2150_215069


namespace valid_numbers_are_135_and_144_l2150_215013

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

end valid_numbers_are_135_and_144_l2150_215013


namespace complement_angle_l2150_215068

theorem complement_angle (A : Real) (h : A = 55) : 90 - A = 35 := by
  sorry

end complement_angle_l2150_215068
