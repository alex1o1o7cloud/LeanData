import Mathlib

namespace triangle_formation_conditions_l2300_230081

theorem triangle_formation_conditions (a b c : ℝ) :
  (a + b > c ∧ |a - b| < c) ↔ (a + b > c ∧ b + c > a ∧ c + a > b ∧ |a - b| < c ∧ |b - c| < a ∧ |c - a| < b) :=
sorry

end triangle_formation_conditions_l2300_230081


namespace sufficient_but_not_necessary_l2300_230073

theorem sufficient_but_not_necessary {a b : ℝ} (h₁ : a < b) (h₂ : b < 0) : 
  (a^2 > b^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by
  sorry

end sufficient_but_not_necessary_l2300_230073


namespace absolute_difference_AB_l2300_230028

noncomputable def A : Real := 12 / 7
noncomputable def B : Real := 20 / 7

theorem absolute_difference_AB : |A - B| = 8 / 7 := by
  sorry

end absolute_difference_AB_l2300_230028


namespace inverse_proportion_l2300_230026

theorem inverse_proportion (x : ℝ) (y : ℝ) (f₁ f₂ f₃ f₄ : ℝ → ℝ) (h₁ : f₁ x = 2 * x) (h₂ : f₂ x = x / 2) (h₃ : f₃ x = 2 / x) (h₄ : f₄ x = 2 / (x - 1)) :
  f₃ x * x = 2 := sorry

end inverse_proportion_l2300_230026


namespace brown_stripes_l2300_230075

theorem brown_stripes (B G Bl : ℕ) (h1 : G = 3 * B) (h2 : Bl = 5 * G) (h3 : Bl = 60) : B = 4 :=
by {
  sorry
}

end brown_stripes_l2300_230075


namespace simplify_expression_l2300_230098

variable (a b : ℕ)

theorem simplify_expression (a b : ℕ) : 5 * a * b - 7 * a * b + 3 * a * b = a * b := by
  sorry

end simplify_expression_l2300_230098


namespace find_special_integer_l2300_230096

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧ (21 ∣ n) ∧ 30 ≤ Real.sqrt n ∧ Real.sqrt n ≤ 30.5 ∧ n = 903 := 
sorry

end find_special_integer_l2300_230096


namespace contrapositive_statement_l2300_230045

theorem contrapositive_statement :
  (∀ n : ℕ, (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0) →
  (∀ n : ℕ, n % 10 ≠ 0 → ¬(n % 2 = 0 ∧ n % 5 = 0)) :=
by
  sorry

end contrapositive_statement_l2300_230045


namespace Chris_buys_48_golf_balls_l2300_230043

theorem Chris_buys_48_golf_balls (total_golf_balls : ℕ) (dozen_to_balls : ℕ → ℕ)
  (dan_buys : ℕ) (gus_buys : ℕ) (chris_buys : ℕ) :
  dozen_to_balls 1 = 12 →
  dan_buys = 5 →
  gus_buys = 2 →
  total_golf_balls = 132 →
  (chris_buys * 12) + (dan_buys * 12) + (gus_buys * 12) = total_golf_balls →
  chris_buys * 12 = 48 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Chris_buys_48_golf_balls_l2300_230043


namespace constant_ratio_of_arithmetic_sequence_l2300_230068

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n-1) * d

-- The main theorem stating the result
theorem constant_ratio_of_arithmetic_sequence 
  (a : ℕ → ℝ) (c : ℝ) (h_seq : arithmetic_sequence a)
  (h_const : ∀ n : ℕ, a n ≠ 0 ∧ a (2 * n) ≠ 0 ∧ a n / a (2 * n) = c) :
  c = 1 ∨ c = 1 / 2 :=
sorry

end constant_ratio_of_arithmetic_sequence_l2300_230068


namespace tan_diff_identity_l2300_230020

theorem tan_diff_identity 
  (α : ℝ)
  (h : Real.tan α = -4/3) : Real.tan (α - Real.pi / 4) = 7 := 
sorry

end tan_diff_identity_l2300_230020


namespace r_pow_four_solution_l2300_230093

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l2300_230093


namespace locate_z_in_fourth_quadrant_l2300_230088

def z_in_quadrant_fourth (z : ℂ) : Prop :=
  (z.re > 0) ∧ (z.im < 0)

theorem locate_z_in_fourth_quadrant (z : ℂ) (i : ℂ) (h : i * i = -1) 
(hz : z * (1 + i) = 1) : z_in_quadrant_fourth z :=
sorry

end locate_z_in_fourth_quadrant_l2300_230088


namespace function_monotonically_increasing_iff_range_of_a_l2300_230040

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem function_monotonically_increasing_iff_range_of_a (a : ℝ) :
  (∀ x, (deriv (f a) x) ≥ 0) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
by
  sorry

end function_monotonically_increasing_iff_range_of_a_l2300_230040


namespace number_of_people_with_cards_greater_than_0p3_l2300_230039

theorem number_of_people_with_cards_greater_than_0p3 :
  (∃ (number_of_people : ℕ),
     number_of_people = (if 0.3 < 0.8 then 1 else 0) +
                        (if 0.3 < (1 / 2) then 1 else 0) +
                        (if 0.3 < 0.9 then 1 else 0) +
                        (if 0.3 < (1 / 3) then 1 else 0)) →
  number_of_people = 4 :=
by
  sorry

end number_of_people_with_cards_greater_than_0p3_l2300_230039


namespace largest_value_is_B_l2300_230065

def exprA := 1 + 2 * 3 + 4
def exprB := 1 + 2 + 3 * 4
def exprC := 1 + 2 + 3 + 4
def exprD := 1 * 2 + 3 + 4
def exprE := 1 * 2 + 3 * 4

theorem largest_value_is_B : exprB = 15 ∧ exprB > exprA ∧ exprB > exprC ∧ exprB > exprD ∧ exprB > exprE := 
by
  sorry

end largest_value_is_B_l2300_230065


namespace heath_average_carrots_per_hour_l2300_230071

theorem heath_average_carrots_per_hour 
  (rows1 rows2 : ℕ)
  (plants_per_row1 plants_per_row2 : ℕ)
  (hours1 hours2 : ℕ)
  (h1 : rows1 = 200)
  (h2 : rows2 = 200)
  (h3 : plants_per_row1 = 275)
  (h4 : plants_per_row2 = 325)
  (h5 : hours1 = 15)
  (h6 : hours2 = 25) :
  ((rows1 * plants_per_row1 + rows2 * plants_per_row2) / (hours1 + hours2) = 3000) :=
  by
  sorry

end heath_average_carrots_per_hour_l2300_230071


namespace beads_probability_l2300_230023

/-
  Four red beads, three white beads, and two blue beads are placed in a line in random order.
  Prove that the probability that no two neighboring beads are the same color is 1/70.
-/
theorem beads_probability :
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18 -- conservative estimate from the solution
  (valid_permutations : ℚ) / total_permutations = 1 / 70 :=
by
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18
  show (valid_permutations : ℚ) / total_permutations = 1 / 70
  -- skipping proof details
  sorry

end beads_probability_l2300_230023


namespace part1_part2_l2300_230017

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - (a / 2) * x^2

-- Define the line l
noncomputable def l (k : ℤ) (x : ℝ) : ℝ := (k - 2) * x - k + 1

-- Theorem for part (1)
theorem part1 (x : ℝ) (a : ℝ) (h₁ : e ≤ x) (h₂ : x ≤ e^2) (h₃ : f a x > 0) : a < 2 / e :=
sorry

-- Theorem for part (2)
theorem part2 (k : ℤ) (h₁ : a = 0) (h₂ : ∀ (x : ℝ), 1 < x → f 0 x > l k x) : k ≤ 4 :=
sorry

end part1_part2_l2300_230017


namespace solve_four_tuple_l2300_230080

-- Define the problem conditions
theorem solve_four_tuple (a b c d : ℝ) : 
    (ab + c + d = 3) → 
    (bc + d + a = 5) → 
    (cd + a + b = 2) → 
    (da + b + c = 6) → 
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by
  intros h1 h2 h3 h4
  sorry

end solve_four_tuple_l2300_230080


namespace sum_of_cubes_consecutive_integers_l2300_230004

theorem sum_of_cubes_consecutive_integers (x : ℕ) (h1 : 0 < x) (h2 : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 :=
by
  -- proof will go here
  sorry

end sum_of_cubes_consecutive_integers_l2300_230004


namespace smallest_positive_integer_satisfying_conditions_l2300_230087

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (N : ℕ), N = 242 ∧
    ( ∃ (i : Fin 4), (N + i) % 8 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 9 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 25 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 121 = 0 ) :=
sorry

end smallest_positive_integer_satisfying_conditions_l2300_230087


namespace parabola_equation_l2300_230059

variables (a b c p : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : p > 0)
variables (h_eccentricity : c / a = 2)
variables (h_b : b = Real.sqrt (3) * a)
variables (h_c : c = Real.sqrt (a^2 + b^2))
variables (d : ℝ) (h_distance : d = 2) (h_d_formula : d = (a * p) / (2 * c))

theorem parabola_equation (h : (a > 0) ∧ (b > 0) ∧ (p > 0) ∧ (c / a = 2) ∧ (b = (Real.sqrt 3) * a) ∧ (c = Real.sqrt (a^2 + b^2)) ∧ (d = 2) ∧ (d = (a * p) / (2 * c))) : x^2 = 16 * y :=
by {
  -- Lean does not require an actual proof here, so we use sorry.
  sorry
}

end parabola_equation_l2300_230059


namespace intersection_eq_M_l2300_230011

-- Define the sets M and N according to the given conditions
def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | |x| < 2}

-- The 'theorem' statement to prove M ∩ N = M
theorem intersection_eq_M : M ∩ N = M :=
  sorry

end intersection_eq_M_l2300_230011


namespace find_x_l2300_230013

theorem find_x {x y : ℝ} (h1 : 3 * x - 2 * y = 7) (h2 : x^2 + 3 * y = 17) : x = 3.5 :=
sorry

end find_x_l2300_230013


namespace machine_value_after_two_years_l2300_230094

theorem machine_value_after_two_years (initial_value : ℝ) (decrease_rate : ℝ) (years : ℕ) (value_after_two_years : ℝ) :
  initial_value = 8000 ∧ decrease_rate = 0.30 ∧ years = 2 → value_after_two_years = 3200 := by
  intros h
  sorry

end machine_value_after_two_years_l2300_230094


namespace range_of_a_l2300_230053

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ a ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by
  sorry

end range_of_a_l2300_230053


namespace difference_of_numbers_l2300_230078

theorem difference_of_numbers :
  ∃ (a b : ℕ), a + b = 36400 ∧ b = 100 * a ∧ b - a = 35640 :=
by
  sorry

end difference_of_numbers_l2300_230078


namespace initial_glass_bottles_count_l2300_230056

namespace Bottles

variable (G P : ℕ)

/-- The weight of some glass bottles is 600 g. 
    The total weight of 4 glass bottles and 5 plastic bottles is 1050 g.
    A glass bottle is 150 g heavier than a plastic bottle.
    Prove that the number of glass bottles initially weighed is 3. -/
theorem initial_glass_bottles_count (h1 : G * (P + 150) = 600)
  (h2 : 4 * (P + 150) + 5 * P = 1050)
  (h3 : P + 150 > P) :
  G = 3 :=
  by sorry

end Bottles

end initial_glass_bottles_count_l2300_230056


namespace infinite_solutions_congruence_l2300_230014

theorem infinite_solutions_congruence (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ᶠ x in at_top, a ^ x + x ≡ b [MOD c] :=
sorry

end infinite_solutions_congruence_l2300_230014


namespace doubled_container_volume_l2300_230029

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l2300_230029


namespace sin_value_l2300_230067

theorem sin_value (alpha : ℝ) (h1 : -π / 6 < alpha ∧ alpha < π / 6)
  (h2 : Real.cos (alpha + π / 6) = 4 / 5) :
  Real.sin (2 * alpha + π / 12) = 17 * Real.sqrt 2 / 50 :=
by
    sorry

end sin_value_l2300_230067


namespace flea_never_lands_on_all_points_l2300_230099

noncomputable def a_n (n : ℕ) : ℕ := (n * (n + 1) / 2) % 300

theorem flea_never_lands_on_all_points :
  ∃ k : ℕ, k < 300 ∧ ∀ n : ℕ, a_n n ≠ k :=
sorry

end flea_never_lands_on_all_points_l2300_230099


namespace sqrt_of_four_is_pm_two_l2300_230052

theorem sqrt_of_four_is_pm_two (y : ℤ) : y * y = 4 → y = 2 ∨ y = -2 := by
  sorry

end sqrt_of_four_is_pm_two_l2300_230052


namespace bug_total_distance_l2300_230054

def total_distance_bug (start : ℤ) (pos1 : ℤ) (pos2 : ℤ) (pos3 : ℤ) : ℤ :=
  abs (pos1 - start) + abs (pos2 - pos1) + abs (pos3 - pos2)

theorem bug_total_distance :
  total_distance_bug 3 (-4) 6 2 = 21 :=
by
  -- We insert a sorry here to indicate the proof is skipped.
  sorry

end bug_total_distance_l2300_230054


namespace ratio_of_length_to_width_l2300_230024

variable (L W : ℕ)
variable (H1 : W = 50)
variable (H2 : 2 * L + 2 * W = 240)

theorem ratio_of_length_to_width : L / W = 7 / 5 := 
by sorry

end ratio_of_length_to_width_l2300_230024


namespace average_salary_all_workers_l2300_230060

-- Definitions based on the conditions
def technicians_avg_salary := 16000
def rest_avg_salary := 6000
def total_workers := 35
def technicians := 7
def rest_workers := total_workers - technicians

-- Prove that the average salary of all workers is 8000
theorem average_salary_all_workers :
  (technicians * technicians_avg_salary + rest_workers * rest_avg_salary) / total_workers = 8000 := by
  sorry

end average_salary_all_workers_l2300_230060


namespace quoted_price_of_shares_l2300_230095

theorem quoted_price_of_shares (investment : ℝ) (face_value : ℝ) (rate_dividend : ℝ) (annual_income : ℝ) (num_shares : ℝ) (quoted_price : ℝ) :
  investment = 4455 ∧ face_value = 10 ∧ rate_dividend = 0.12 ∧ annual_income = 648 ∧ num_shares = annual_income / (rate_dividend * face_value) →
  quoted_price = investment / num_shares :=
by sorry

end quoted_price_of_shares_l2300_230095


namespace smallest_multiple_of_18_all_digits_9_or_0_l2300_230006

theorem smallest_multiple_of_18_all_digits_9_or_0 :
  ∃ (m : ℕ), (m > 0) ∧ (m % 18 = 0) ∧ (∀ d ∈ (m.digits 10), d = 9 ∨ d = 0) ∧ (m / 18 = 5) :=
sorry

end smallest_multiple_of_18_all_digits_9_or_0_l2300_230006


namespace arithmetic_sequence_sum_l2300_230084

theorem arithmetic_sequence_sum (c d : ℕ) (h₁ : 3 + 5 = 8) (h₂ : 8 + 5 = 13) (h₃ : c = 13 + 5) (h₄ : d = 18 + 5) (h₅ : d + 5 = 28) : c + d = 41 :=
by
  sorry

end arithmetic_sequence_sum_l2300_230084


namespace find_quarters_l2300_230074

-- Define the conditions
def quarters_bounds (q : ℕ) : Prop :=
  8 < q ∧ q < 80

def stacks_mod4 (q : ℕ) : Prop :=
  q % 4 = 2

def stacks_mod6 (q : ℕ) : Prop :=
  q % 6 = 2

def stacks_mod8 (q : ℕ) : Prop :=
  q % 8 = 2

-- The theorem to prove
theorem find_quarters (q : ℕ) (h_bounds : quarters_bounds q) (h4 : stacks_mod4 q) (h6 : stacks_mod6 q) (h8 : stacks_mod8 q) : 
  q = 26 :=
by
  sorry

end find_quarters_l2300_230074


namespace find_BC_distance_l2300_230050

-- Definitions of constants as per problem conditions
def ACB_angle : ℝ := 120
def AC_distance : ℝ := 2
def AB_distance : ℝ := 3

-- The theorem to prove the distance BC
theorem find_BC_distance (BC : ℝ) (h : AC_distance * AC_distance + (BC * BC) - 2 * AC_distance * BC * Real.cos (ACB_angle * Real.pi / 180) = AB_distance * AB_distance) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end find_BC_distance_l2300_230050


namespace average_income_correct_l2300_230091

def incomes : List ℕ := [250, 400, 750, 400, 500]

noncomputable def average : ℕ := (incomes.sum) / incomes.length

theorem average_income_correct : average = 460 :=
by 
  sorry

end average_income_correct_l2300_230091


namespace base_conversion_correct_l2300_230090

def convert_base_9_to_10 (n : ℕ) : ℕ :=
  3 * 9^2 + 6 * 9^1 + 1 * 9^0

def convert_base_13_to_10 (n : ℕ) (C : ℕ) : ℕ :=
  4 * 13^2 + C * 13^1 + 5 * 13^0

theorem base_conversion_correct :
  convert_base_9_to_10 361 + convert_base_13_to_10 4 12 = 1135 :=
by
  sorry

end base_conversion_correct_l2300_230090


namespace num_squares_less_than_1000_with_ones_digit_2_3_or_4_l2300_230000

-- Define a function that checks if the one's digit of a number is one of 2, 3, or 4.
def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

-- Define the main theorem to prove
theorem num_squares_less_than_1000_with_ones_digit_2_3_or_4 : 
  ∃ n, n = 6 ∧ ∀ m < 1000, ∃ k, m = k^2 → ends_in m 2 ∨ ends_in m 3 ∨ ends_in m 4 :=
sorry

end num_squares_less_than_1000_with_ones_digit_2_3_or_4_l2300_230000


namespace degree_difference_l2300_230012

variable (S J : ℕ)

theorem degree_difference :
  S = 150 → S + J = 295 → S - J = 5 :=
by
  intros h₁ h₂
  sorry

end degree_difference_l2300_230012


namespace delores_initial_money_l2300_230031

theorem delores_initial_money (cost_computer : ℕ) (cost_printer : ℕ) (money_left : ℕ) (initial_money : ℕ) :
  cost_computer = 400 → cost_printer = 40 → money_left = 10 → initial_money = cost_computer + cost_printer + money_left → initial_money = 450 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end delores_initial_money_l2300_230031


namespace unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l2300_230097

-- Definitions of points and lines
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a : α) (b : α) -- Represented as ax + by = 0

-- Given conditions
variables {α : Type*} [Field α]
variables (P Q : Point α)
variables (L1 L2 : Line α) -- L1 and L2 are perpendicular

-- Proof problem statement
theorem unique_ellipse_through_points_with_perpendicular_axes (P Q : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
(P ≠ Q) → 
∃! (E : Set (Point α)), -- E represents the ellipse as a set of points
(∀ (p : Point α), p ∈ E → (p = P ∨ p = Q)) ∧ -- E passes through P and Q
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

theorem infinite_ellipses_when_points_coincide (P : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
∃ (E : Set (Point α)), -- E represents an ellipse
(∀ (p : Point α), p ∈ E → p = P) ∧ -- E passes through P
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

end unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l2300_230097


namespace infinite_points_with_sum_of_squares_condition_l2300_230042

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle centered at origin with given radius
def isWithinCircle (P : Point2D) (r : ℝ) :=
  P.x^2 + P.y^2 ≤ r^2

-- Define the distance squared from a point to another point
def dist2 (P Q : Point2D) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the problem
theorem infinite_points_with_sum_of_squares_condition :
  ∃ P : Point2D, isWithinCircle P 1 → (dist2 P ⟨-1, 0⟩ + dist2 P ⟨1, 0⟩ = 3) :=
by  
  sorry

end infinite_points_with_sum_of_squares_condition_l2300_230042


namespace remainder_when_divided_by_296_l2300_230003

theorem remainder_when_divided_by_296 (N : ℤ) (Q : ℤ) (R : ℤ)
  (h1 : N % 37 = 1)
  (h2 : N = 296 * Q + R)
  (h3 : 0 ≤ R) 
  (h4 : R < 296) :
  R = 260 := 
sorry

end remainder_when_divided_by_296_l2300_230003


namespace find_m_value_l2300_230009

theorem find_m_value
    (x y m : ℝ)
    (hx : x = -1)
    (hy : y = 2)
    (hxy : m * x + 2 * y = 1) :
    m = 3 :=
by
  sorry

end find_m_value_l2300_230009


namespace no_quad_term_l2300_230044

theorem no_quad_term (x m : ℝ) : 
  (2 * x^2 - 2 * (7 + 3 * x - 2 * x^2) + m * x^2) = -6 * x - 14 → m = -6 := 
by 
  sorry

end no_quad_term_l2300_230044


namespace inverse_of_h_l2300_230025

def h (x : ℝ) : ℝ := 3 + 6 * x

noncomputable def k (x : ℝ) : ℝ := (x - 3) / 6

theorem inverse_of_h : ∀ x, h (k x) = x :=
by
  intro x
  unfold h k
  sorry

end inverse_of_h_l2300_230025


namespace sequence_sqrt_l2300_230058

theorem sequence_sqrt (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a n > 0)
  (h₃ : ∀ n, a (n+1 - 1) ^ 2 = a (n+1) ^ 2 + 4) :
  ∀ n, a n = Real.sqrt (4 * n - 3) :=
by
  sorry

end sequence_sqrt_l2300_230058


namespace find_c_l2300_230041

theorem find_c (y c : ℝ) (h : y > 0) (h₂ : (8*y)/20 + (c*y)/10 = 0.7*y) : c = 6 :=
by
  sorry

end find_c_l2300_230041


namespace mudit_age_l2300_230019

theorem mudit_age :
    ∃ x : ℤ, x + 16 = 3 * (x - 4) ∧ x = 14 :=
by
  use 14
  sorry -- Proof goes here

end mudit_age_l2300_230019


namespace loaned_books_count_l2300_230063

variable (x : ℕ) -- x is the number of books loaned out during the month

theorem loaned_books_count 
  (initial_books : ℕ) (returned_percentage : ℚ) (remaining_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : returned_percentage = 0.80)
  (h3 : remaining_books = 66) :
  x = 45 :=
by
  -- Proof can be inserted here
  sorry

end loaned_books_count_l2300_230063


namespace david_biology_marks_l2300_230001

theorem david_biology_marks
  (english math physics chemistry avg_marks num_subjects : ℕ)
  (h_english : english = 86)
  (h_math : math = 85)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 87)
  (h_avg_marks : avg_marks = 85)
  (h_num_subjects : num_subjects = 5) :
  ∃ (biology : ℕ), biology = 85 :=
by
  -- Total marks for all subjects
  let total_marks_for_all_subjects := avg_marks * num_subjects
  -- Total marks in English, Mathematics, Physics, and Chemistry
  let total_marks_in_other_subjects := english + math + physics + chemistry
  -- Marks in Biology
  let biology := total_marks_for_all_subjects - total_marks_in_other_subjects
  existsi biology
  sorry

end david_biology_marks_l2300_230001


namespace range_of_a_l2300_230086

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l2300_230086


namespace P_subset_M_l2300_230061

def P : Set ℝ := {x | x^2 - 6 * x + 9 = 0}
def M : Set ℝ := {x | x > 1}

theorem P_subset_M : P ⊂ M := by sorry

end P_subset_M_l2300_230061


namespace square_pyramid_sum_l2300_230018

-- Define the number of faces, edges, and vertices of a square pyramid.
def faces_square_base : Nat := 1
def faces_lateral : Nat := 4
def edges_base : Nat := 4
def edges_lateral : Nat := 4
def vertices_base : Nat := 4
def vertices_apex : Nat := 1

-- Summing the faces, edges, and vertices
def total_faces : Nat := faces_square_base + faces_lateral
def total_edges : Nat := edges_base + edges_lateral
def total_vertices : Nat := vertices_base + vertices_apex

theorem square_pyramid_sum : (total_faces + total_edges + total_vertices = 18) :=
by
  sorry

end square_pyramid_sum_l2300_230018


namespace distinct_prime_factors_count_l2300_230057

theorem distinct_prime_factors_count :
  ∀ (a b c d : ℕ),
  (a = 79) → (b = 3^4) → (c = 5 * 17) → (d = 3 * 29) →
  (∃ s : Finset ℕ, ∀ n ∈ s, Nat.Prime n ∧ 79 * 81 * 85 * 87 = s.prod id) :=
sorry

end distinct_prime_factors_count_l2300_230057


namespace map_representation_l2300_230015

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l2300_230015


namespace total_numbers_l2300_230069

theorem total_numbers (m j c : ℕ) (h1 : m = j + 20) (h2 : j = c - 40) (h3 : c = 80) : m + j + c = 180 := 
by sorry

end total_numbers_l2300_230069


namespace simple_interest_time_l2300_230082

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem simple_interest_time (SI CI : ℝ) (SI_given CI_given P_simp P_comp r_simp r_comp t_comp : ℝ) :
  SI = CI / 2 →
  CI = compound_interest P_comp r_comp 1 t_comp - P_comp →
  SI = simple_interest P_simp r_simp t_comp →
  P_simp = 1272 →
  r_simp = 0.10 →
  P_comp = 5000 →
  r_comp = 0.12 →
  t_comp = 2 →
  t_comp = 5 :=
by
  intros
  sorry

end simple_interest_time_l2300_230082


namespace wendy_time_correct_l2300_230027

variable (bonnie_time wendy_difference : ℝ)

theorem wendy_time_correct (h1 : bonnie_time = 7.80) (h2 : wendy_difference = 0.25) : 
  (bonnie_time - wendy_difference = 7.55) :=
by
  sorry

end wendy_time_correct_l2300_230027


namespace abc_equal_l2300_230002

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 - ab - bc - ac = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l2300_230002


namespace largest_six_consecutive_nonprime_under_50_l2300_230007

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def consecutiveNonPrimes (m : ℕ) : Prop :=
  ∀ i : ℕ, i < 6 → ¬ isPrime (m + i)

theorem largest_six_consecutive_nonprime_under_50 (n : ℕ) :
  (n < 50 ∧ consecutiveNonPrimes n) →
  n + 5 = 35 :=
by
  intro h
  sorry

end largest_six_consecutive_nonprime_under_50_l2300_230007


namespace betty_needs_more_flies_l2300_230008

def betty_frog_food (daily_flies: ℕ) (days_per_week: ℕ) (morning_catch: ℕ) 
  (afternoon_catch: ℕ) (flies_escaped: ℕ) : ℕ :=
  days_per_week * daily_flies - (morning_catch + afternoon_catch - flies_escaped)

theorem betty_needs_more_flies :
  betty_frog_food 2 7 5 6 1 = 4 :=
by
  sorry

end betty_needs_more_flies_l2300_230008


namespace conditional_probability_P_B_given_A_l2300_230022

-- Let E be an enumeration type with exactly five values, each representing one attraction.
inductive Attraction : Type
| dayu_yashan : Attraction
| qiyunshan : Attraction
| tianlongshan : Attraction
| jiulianshan : Attraction
| sanbaishan : Attraction

open Attraction

-- Define A and B's choices as random variables.
axiom A_choice : Attraction
axiom B_choice : Attraction

-- Event A is that A and B choose different attractions.
def event_A : Prop := A_choice ≠ B_choice

-- Event B is that A and B each choose Chongyi Qiyunshan.
def event_B : Prop := A_choice = qiyunshan ∧ B_choice = qiyunshan

-- Calculate the conditional probability P(B|A)
theorem conditional_probability_P_B_given_A : 
  (1 - (1 / 5)) * (1 - (1 / 5)) = 2 / 5 :=
sorry

end conditional_probability_P_B_given_A_l2300_230022


namespace avg_age_decrease_l2300_230089

/-- Define the original average age of the class -/
def original_avg_age : ℕ := 40

/-- Define the number of original students -/
def original_strength : ℕ := 17

/-- Define the average age of the new students -/
def new_students_avg_age : ℕ := 32

/-- Define the number of new students joining -/
def new_students_strength : ℕ := 17

/-- Define the total original age of the class -/
def total_original_age : ℕ := original_strength * original_avg_age

/-- Define the total age of the new students -/
def total_new_students_age : ℕ := new_students_strength * new_students_avg_age

/-- Define the new total strength of the class after joining of new students -/
def new_total_strength : ℕ := original_strength + new_students_strength

/-- Define the new total age of the class after joining of new students -/
def new_total_age : ℕ := total_original_age + total_new_students_age

/-- Define the new average age of the class -/
def new_avg_age : ℕ := new_total_age / new_total_strength

/-- Prove that the average age decreased by 4 years when the new students joined -/
theorem avg_age_decrease : original_avg_age - new_avg_age = 4 := by
  sorry

end avg_age_decrease_l2300_230089


namespace initial_principal_amount_l2300_230030

theorem initial_principal_amount
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ)
  (hA : A = 8400) 
  (hr : r = 0.05)
  (hn : n = 1) 
  (ht : t = 1) 
  (hformula : A = P * (1 + r / n) ^ (n * t)) : 
  P = 8000 :=
by
  rw [hA, hr, hn, ht] at hformula
  sorry

end initial_principal_amount_l2300_230030


namespace part1_inequality_part2_range_of_a_l2300_230010

-- Definitions and conditions
def f (x a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- First proof problem for a = 1
theorem part1_inequality (x : ℝ) : f x 1 > 1 ↔ x > 1/2 :=
by sorry

-- Second proof problem for range of a when f(x) > x in (0, 1)
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → f x a > x) → 0 < a ∧ a ≤ 2 :=
by sorry

end part1_inequality_part2_range_of_a_l2300_230010


namespace square_chord_length_eq_l2300_230036

def radius1 := 10
def radius2 := 7
def centers_distance := 15
def chord_length (x : ℝ) := 2 * x

theorem square_chord_length_eq :
    ∀ (x : ℝ), chord_length x = 15 →
    (10 + x)^2 - 200 * (Real.sqrt ((1 + 19.0 / 35.0) / 2)) = 200 - 200 * Real.sqrt (27.0 / 35.0) :=
sorry

end square_chord_length_eq_l2300_230036


namespace findValuesForFibSequence_l2300_230092

noncomputable def maxConsecutiveFibonacciTerms (A B C : ℝ) : ℝ :=
  if A ≠ 0 then 4 else 0

theorem findValuesForFibSequence :
  maxConsecutiveFibonacciTerms (1/2) (-1/2) 2 = 4 ∧ maxConsecutiveFibonacciTerms (1/2) (1/2) 2 = 4 :=
by
  -- This statement will follow from the given conditions and the solution provided.
  sorry

end findValuesForFibSequence_l2300_230092


namespace time_for_plastic_foam_drift_l2300_230055

def boat_speed_in_still_water : ℝ := sorry
def speed_of_water_flow : ℝ := sorry
def distance_between_docks : ℝ := sorry

theorem time_for_plastic_foam_drift (x y s t : ℝ) 
(hx : 6 * (x + y) = s)
(hy : 8 * (x - y) = s)
(t_eq : t = s / y) : 
t = 48 := 
sorry

end time_for_plastic_foam_drift_l2300_230055


namespace find_element_in_A_l2300_230077

def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

def f (p : A) : B := (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem find_element_in_A : ∃ p : A, f p = (3, 1) ∧ p = (1, 1) := by
  sorry

end find_element_in_A_l2300_230077


namespace probability_wife_selection_l2300_230005

theorem probability_wife_selection (P_H P_only_one P_W : ℝ)
  (h1 : P_H = 1 / 7)
  (h2 : P_only_one = 0.28571428571428575)
  (h3 : P_only_one = (P_H * (1 - P_W)) + (P_W * (1 - P_H))) :
  P_W = 1 / 5 :=
by
  sorry

end probability_wife_selection_l2300_230005


namespace mixed_gender_appointment_schemes_l2300_230083

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 
  else n * factorial (n - 1)

noncomputable def P (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

theorem mixed_gender_appointment_schemes : 
  let total_students := 9
  let total_permutations := P total_students 3
  let male_students := 5
  let female_students := 4
  let male_permutations := P male_students 3
  let female_permutations := P female_students 3
  total_permutations - (male_permutations + female_permutations) = 420 :=
by 
  sorry

end mixed_gender_appointment_schemes_l2300_230083


namespace problem_statement_l2300_230047

open Real

theorem problem_statement (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * π)
  (h₁ : 2 * cos x ≤ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))
  ∧ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x)) ≤ sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := sorry

end problem_statement_l2300_230047


namespace line_passes_point_l2300_230051

theorem line_passes_point (k : ℝ) :
  ((1 + 4 * k) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k)) = 0 :=
by
  sorry

end line_passes_point_l2300_230051


namespace closest_point_exists_l2300_230038

def closest_point_on_line_to_point (x : ℝ) (y : ℝ) : Prop :=
  ∃(p : ℝ × ℝ), p = (3, 1) ∧ ∀(q : ℝ × ℝ), q.2 = (q.1 + 3) / 3 → dist p (3, 2) ≤ dist q (3, 2)

theorem closest_point_exists :
  closest_point_on_line_to_point 3 2 :=
sorry

end closest_point_exists_l2300_230038


namespace valid_exponent_rule_l2300_230066

theorem valid_exponent_rule (a : ℝ) : (a^3)^2 = a^6 :=
by
  sorry

end valid_exponent_rule_l2300_230066


namespace part_a_l2300_230034

theorem part_a (a b c : ℝ) (m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) :=
by
  sorry

end part_a_l2300_230034


namespace sin_cos_identity_l2300_230064

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) + Real.cos (20 * Real.pi / 180) * Real.sin (140 * Real.pi / 180)) =
  (Real.sqrt 3 / 2) := by
  sorry

end sin_cos_identity_l2300_230064


namespace sector_area_l2300_230033

theorem sector_area (s θ r : ℝ) (hs : s = 4) (hθ : θ = 2) (hr : r = s / θ) : (1/2) * r^2 * θ = 4 := by
  sorry

end sector_area_l2300_230033


namespace range_of_m_for_point_in_second_quadrant_l2300_230070

theorem range_of_m_for_point_in_second_quadrant (m : ℝ) :
  (m - 3 < 0) ∧ (m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  -- The proof will be inserted here.
  sorry

end range_of_m_for_point_in_second_quadrant_l2300_230070


namespace rods_in_one_mile_l2300_230085

-- Definitions of the conditions
def mile_to_furlong := 10
def furlong_to_rod := 50

-- Theorem statement corresponding to the proof problem
theorem rods_in_one_mile : mile_to_furlong * furlong_to_rod = 500 := 
by sorry

end rods_in_one_mile_l2300_230085


namespace recurring_decimal_to_fraction_l2300_230037

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end recurring_decimal_to_fraction_l2300_230037


namespace angle_rotation_l2300_230035

theorem angle_rotation (initial_angle : ℝ) (rotation : ℝ) :
  initial_angle = 30 → rotation = 450 → 
  ∃ (new_angle : ℝ), new_angle = 60 :=
by
  sorry

end angle_rotation_l2300_230035


namespace sum_of_coordinates_l2300_230021

theorem sum_of_coordinates (C D : ℝ × ℝ) (hC : C = (0, 0)) (hD : D.snd = 6) (h_slope : (D.snd - C.snd) / (D.fst - C.fst) = 3/4) : D.fst + D.snd = 14 :=
sorry

end sum_of_coordinates_l2300_230021


namespace find_a_l2300_230049

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 2 ↔ |a * x + 2| < 6) → a = -4 :=
by
  intro h
  sorry

end find_a_l2300_230049


namespace no_real_roots_for_pair_2_2_3_l2300_230079

noncomputable def discriminant (A B : ℝ) : ℝ :=
  let a := 1 - 2 * B
  let b := -B
  let c := -A + A * B
  b ^ 2 - 4 * a * c

theorem no_real_roots_for_pair_2_2_3 : discriminant 2 (2 / 3) < 0 := by
  sorry

end no_real_roots_for_pair_2_2_3_l2300_230079


namespace Jackson_money_is_125_l2300_230072

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end Jackson_money_is_125_l2300_230072


namespace find_retail_price_l2300_230048

-- Define the wholesale price
def wholesale_price : ℝ := 90

-- Define the profit as 20% of the wholesale price
def profit (w : ℝ) : ℝ := 0.2 * w

-- Define the selling price as the wholesale price plus the profit
def selling_price (w p : ℝ) : ℝ := w + p

-- Define the selling price as 90% of the retail price t
def discount_selling_price (t : ℝ) : ℝ := 0.9 * t

-- Prove that the retail price t is 120 given the conditions
theorem find_retail_price :
  ∃ t : ℝ, wholesale_price + (profit wholesale_price) = discount_selling_price t → t = 120 :=
by
  sorry

end find_retail_price_l2300_230048


namespace red_balls_in_bag_l2300_230032

theorem red_balls_in_bag : 
  ∃ (r : ℕ), (r * (r - 1) = 22) ∧ (r ≤ 12) :=
by { sorry }

end red_balls_in_bag_l2300_230032


namespace total_time_equals_l2300_230046

-- Define the distances and speeds
def first_segment_distance : ℝ := 50
def first_segment_speed : ℝ := 30
def second_segment_distance (b : ℝ) : ℝ := b
def second_segment_speed : ℝ := 80

-- Prove that the total time is equal to (400 + 3b) / 240 hours
theorem total_time_equals (b : ℝ) : 
  (first_segment_distance / first_segment_speed) + (second_segment_distance b / second_segment_speed) 
  = (400 + 3 * b) / 240 := 
by
  sorry

end total_time_equals_l2300_230046


namespace remaining_structure_volume_and_surface_area_l2300_230016

-- Define the dimensions of the large cube and the small cubes
def large_cube_volume := 12 * 12 * 12
def small_cube_volume := 2 * 2 * 2

-- Define the number of smaller cubes in the large cube
def num_small_cubes := (12 / 2) * (12 / 2) * (12 / 2)

-- Define the number of smaller cubes removed (central on each face and very center)
def removed_cubes := 7

-- The volume of a small cube after removing its center unit
def single_small_cube_remaining_volume := small_cube_volume - 1

-- Calculate the remaining volume after all removals
def remaining_volume := (num_small_cubes - removed_cubes) * single_small_cube_remaining_volume

-- Initial surface area of a small cube and increase per removal of central unit
def single_small_cube_initial_surface_area := 6 * 4 -- 6 faces of 2*2*2 cube, each face has 4 units
def single_small_cube_surface_increase := 6

-- Calculate the adjusted surface area considering internal faces' reduction
def single_cube_adjusted_surface_area := single_small_cube_initial_surface_area + single_small_cube_surface_increase
def total_initial_surface_area := single_cube_adjusted_surface_area * (num_small_cubes - removed_cubes)
def total_internal_faces_area := (num_small_cubes - removed_cubes) * 2 * 4
def final_surface_area := total_initial_surface_area - total_internal_faces_area

theorem remaining_structure_volume_and_surface_area :
  remaining_volume = 1463 ∧ final_surface_area = 4598 :=
by
  -- Proof logic goes here
  sorry

end remaining_structure_volume_and_surface_area_l2300_230016


namespace total_boxes_count_l2300_230062

theorem total_boxes_count
  (initial_boxes : ℕ := 2013)
  (boxes_per_operation : ℕ := 13)
  (operations : ℕ := 2013)
  (non_empty_boxes : ℕ := 2013)
  (total_boxes : ℕ := initial_boxes + boxes_per_operation * operations) :
  non_empty_boxes = operations → total_boxes = 28182 :=
by
  sorry

end total_boxes_count_l2300_230062


namespace find_constants_l2300_230076

def equation1 (x p q : ℝ) : Prop := (x + p) * (x + q) * (x + 5) = 0
def equation2 (x p q : ℝ) : Prop := (x + 2 * p) * (x + 2) * (x + 3) = 0

def valid_roots1 (p q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ p q ∧ equation1 x₂ p q ∧
  x₁ = -5 ∨ x₁ = -q ∨ x₁ = -p

def valid_roots2 (p q : ℝ) : Prop :=
  ∃ x₃ x₄ : ℝ, x₃ ≠ x₄ ∧ equation2 x₃ p q ∧ equation2 x₄ p q ∧
  (x₃ = -2 * p ∨ x₃ = -2 ∨ x₃ = -3)

theorem find_constants (p q : ℝ) (h1 : valid_roots1 p q) (h2 : valid_roots2 p q) : 100 * p + q = 502 :=
by
  sorry

end find_constants_l2300_230076
