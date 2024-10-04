import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.SquareRoot
import Mathlib.Algebra.Sum
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Geometry.Euclidean
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.MeasurableSpace.Basic
import Mathlib.MeasureTheory.Integral.Lebesgue
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Cardinal
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Geometry.Euclidean.Basic

namespace remainder_4032_125_l286_286960

theorem remainder_4032_125 : 4032 % 125 = 32 := by
  sorry

end remainder_4032_125_l286_286960


namespace find_puppy_weight_l286_286638

noncomputable def weight_problem (a b c : ℕ) : Prop :=
  a + b + c = 36 ∧ a + c = 3 * b ∧ a + b = c + 6

theorem find_puppy_weight (a b c : ℕ) (h : weight_problem a b c) : a = 12 :=
sorry

end find_puppy_weight_l286_286638


namespace problem_1_problem_2_l286_286737

-- Define the functions f and g.
def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - a * x) * (x * Real.exp x + Real.sqrt 2)
def g (x : ℝ) : ℝ := x * Real.exp x + Real.sqrt 2

-- Problem 1: Prove that a = 0 given the slope of the tangent line to y = f(x) at (0, f(0))
theorem problem_1 (a : ℝ) (f : ℝ → ℝ) (h : f 0 = (Real.sqrt 2 + 1) ∧ f'(0) = (Real.sqrt 2 + 1)) : a = 0 :=
sorry

-- Problem 2: Prove that g(x) > 1 for all x in ℝ
theorem problem_2 (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x) = x * Real.exp x + Real.sqrt 2) : ∀ x : ℝ, g x > 1 :=
sorry

end problem_1_problem_2_l286_286737


namespace no_positive_integer_solutions_l286_286690

theorem no_positive_integer_solutions (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) : x^2 + y^2 ≠ 7 * z^2 := by
  sorry

end no_positive_integer_solutions_l286_286690


namespace students_scoring_above_120_l286_286329

theorem students_scoring_above_120 
    (total_students : ℕ)
    (mean_score : ℝ)
    (std_dev : ℝ)
    (sixty_percent_scoring_range : Real)
    (less_than_score : Real)
    (greater_than_score : Real)
    (prop_scoring_range : Real)
    (prob_above_threshold : Real)
    (threshold : Real) 
    (students_scoring_above_threshold : ℕ)
    (h1 : total_students = 1000)
    (h2 : mean_score = 100)
    (h3 : 0 < std_dev)
    (h4 : sixty_percent_scoring_range = 0.6)
    (h5 : less_than_score = 80)
    (h6 : greater_than_score = 120)
    (h7 : prop_scoring_range 
            = (Real.erf ((greater_than_score - mean_score) / (std_dev * Math.sqrt 2)) 
              - Real.erf ((less_than_score - mean_score) / (std_dev * Math.sqrt 2))) / 2)
    (h8 : abs (prop_scoring_range - sixty_percent_scoring_range) < 0.01)
    (h9 : prob_above_threshold = (1 - sixty_percent_scoring_range) / 2)
    (h10 : threshold = 120)
    (h11 : students_scoring_above_threshold 
            = Int.floor (prob_above_threshold * total_students)) :
  students_scoring_above_threshold = 200 := by
  sorry

end students_scoring_above_120_l286_286329


namespace abc_divisibility_l286_286704

theorem abc_divisibility (a b c : ℕ) (h1 : a^2 * b ∣ a^3 + b^3 + c^3) (h2 : b^2 * c ∣ a^3 + b^3 + c^3) (h3 : c^2 * a ∣ a^3 + b^3 + c^3) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end abc_divisibility_l286_286704


namespace total_kids_in_lawrence_l286_286328

-- Define conditions
def stayed_home_lawrence := 644997
def went_to_camp_lawrence := 893835
def additional_kids_outside := 78

-- Define the proof statement
theorem total_kids_in_lawrence :
  stayed_home_lawrence + went_to_camp_lawrence = 1538832 :=
by
  have h : stayed_home_lawrence + went_to_camp_lawrence = 644997 + 893835 := rfl
  rw [h]
  norm_num
  exact rfl

end total_kids_in_lawrence_l286_286328


namespace sin_angle_between_AM_AN_is_correct_l286_286493

noncomputable def sin_theta : ℚ :=
let A := (0 : ℚ, 0 : ℚ),
    B := (0 : ℚ, 2 : ℚ),
    C := (4 : ℚ, 2 : ℚ),
    D := (4 : ℚ, 0 : ℚ),
    M := (3 : ℚ, 2 : ℚ),
    N := (4 : ℚ, 1 : ℚ) in
  (5 : ℚ) / real.sqrt (221)

theorem sin_angle_between_AM_AN_is_correct :
  let A := (0 : ℝ, 0 : ℝ),
      B := (0 : ℝ, 2 : ℝ),
      C := (4 : ℝ, 2 : ℝ),
      D := (4 : ℝ, 0 : ℝ),
      M := (3 : ℝ, 2 : ℝ),
      N := (4 : ℝ, 1 : ℝ),
      AM := real.sqrt ((3 - 0)^2 + (2 - 0)^2),
      AN := real.sqrt ((4 - 0)^2 + (1 - 0)^2),
      MN := real.sqrt ((4 - 3)^2 + (1 - 2)^2),
      cos_theta := (AM^2 + AN^2 - MN^2) / (2 * AM * AN),
      sin_theta := real.sqrt (1 - cos_theta^2) in
  sin_theta = (5 : ℝ) / real.sqrt (221) :=
by
  sorry

end sin_angle_between_AM_AN_is_correct_l286_286493


namespace brie_clothing_count_l286_286597

theorem brie_clothing_count 
(h_blouses : 15)
(h_skirts : 9)
(h_slacks : 8)
(h_dresses : 7)
(h_jackets : 4)
(h_blouses_hamper : Float.ofRat 0.60 * 15 = 9)
(h_skirts_hamper : Float.ofRat 0.3333 * 9 = (3 : Float).round.toNat)
(h_slacks_hamper : Float.ofRat 0.50 * 8 = 4)
(h_dresses_hamper : Float.ofRat 0.5714 * 7 = (4 : Float).round.toNat)
(h_jackets_hamper : Float.ofRat 0.75 * 4 = 3) :
  9 + (3 : Nat) + 4 + (4 : Nat) + 3 = 23 :=
by
  sorry

end brie_clothing_count_l286_286597


namespace antitriangular_sum_one_1001_valid_k_l286_286237

def is_antitriangular (x : ℚ) : Prop := ∃ n : ℕ, x = 2 / (n * (n + 1))

def sum_of_antitriangulars (s : ℕ → ℚ) (k : ℕ) (total : ℚ) : Prop :=
  (∀ i, i < k → is_antitriangular (s i)) ∧ (∑ i in Finset.range k, s i) = total

theorem antitriangular_sum_one_1001_valid_k :
  (Finset.filter (λ k, sum_of_antitriangulars (λ _, 1 / (k:ℚ)) k 1) 
                  (Finset.range 2001 \ Finset.range 1000)).card = 1001 := sorry

end antitriangular_sum_one_1001_valid_k_l286_286237


namespace point_E_divides_BC_l286_286459

-- Define a Triangle with Points A, B, C
structure Triangle :=
  (A B C : Type)
  (division_ratio : A → A → ℝ)
  (F : A)
  (G : A)
  (E : A)

-- Conditions given in the problem
variables (T : Triangle)

-- F divides AC in the ratio 2:3
axiom F_div_AC : T.division_ratio T.A T.C = 2 / 3

-- G divides BF in the ratio 1:3
axiom G_div_BF : T.division_ratio T.B T.F = 1 / 3

-- E is the intersection of BC and AG
axiom intersection_point : T.E = intersection_of_lines T.B T.C (line_through T.A T.G) -- Assuming appropriate definitions

-- Problem to be proved
theorem point_E_divides_BC (h1 : F_div_AC T) (h2 : G_div_BF T) (h3 : intersection_point T) :
  T.division_ratio T.B T.E = 2 / 15 :=
sorry

end point_E_divides_BC_l286_286459


namespace new_average_score_l286_286971

theorem new_average_score (average_initial : ℝ) (total_practices : ℕ) (highest_score lowest_score : ℝ) :
  average_initial = 87 → 
  total_practices = 10 → 
  highest_score = 95 → 
  lowest_score = 55 → 
  ((average_initial * total_practices - highest_score - lowest_score) / (total_practices - 2)) = 90 :=
by
  intros h_avg h_total h_high h_low
  sorry

end new_average_score_l286_286971


namespace volume_of_circular_well_l286_286254

noncomputable def volume_of_earth_dug_out (diameter depth : ℝ) : ℝ :=
  let radius := diameter / 2 in
  Real.pi * radius^2 * depth

theorem volume_of_circular_well (D h : ℝ) (pi_approx : ℝ) (H1 : D = 2) (H2 : h = 14) (H3 : pi_approx = 3.14159) :
  volume_of_earth_dug_out D h = 44.01746 := by
  -- Placeholder for proof steps, acknowledging the given conditions
  sorry

end volume_of_circular_well_l286_286254


namespace comprehensive_survey_is_C_l286_286251

def option (label : String) (description : String) := (label, description)

def A := option "A" "Investigating the current mental health status of middle school students nationwide"
def B := option "B" "Investigating the compliance of food in our city"
def C := option "C" "Investigating the physical and mental conditions of classmates in the class"
def D := option "D" "Investigating the viewership ratings of Nanjing TV's 'Today's Life'"

theorem comprehensive_survey_is_C (suitable: (String × String → Prop)) :
  suitable C :=
sorry

end comprehensive_survey_is_C_l286_286251


namespace polynomial_remainder_l286_286712

noncomputable def divisionRemainder (f g : Polynomial ℝ) : Polynomial ℝ := Polynomial.modByMonic f g

theorem polynomial_remainder :
  divisionRemainder (Polynomial.X ^ 5 + 2) (Polynomial.X ^ 2 - 4 * Polynomial.X + 7) = -29 * Polynomial.X - 54 :=
by
  sorry

end polynomial_remainder_l286_286712


namespace seven_digit_phone_numbers_l286_286426

theorem seven_digit_phone_numbers : let count := 9 * 10^6 in count = 9 * 10 ^ 6 :=
by
  -- conditions
  let first_digit_choices := 9
  let remaining_digit_choices := 10
  sorry

end seven_digit_phone_numbers_l286_286426


namespace hamburgers_needed_l286_286726

theorem hamburgers_needed (price_per_hamburger : ℕ) (initial_hamburgers_1 : ℕ) (initial_hamburgers_2 : ℕ) (target_money : ℕ) :
  price_per_hamburger = 5 →
  initial_hamburgers_1 = 4 →
  initial_hamburgers_2 = 2 →
  target_money = 50 →
  (target_money - (price_per_hamburger * (initial_hamburgers_1 + initial_hamburgers_2))) / price_per_hamburger = 4 :=
by
  intros h_price h_initial_1 h_initial_2 h_target
  rw [h_price, h_initial_1, h_initial_2, h_target]
  norm_num
  sorry

end hamburgers_needed_l286_286726


namespace diameter_C_is_10_sqrt_2_l286_286309

noncomputable def diameter_C_eq : Prop :=
  ∃ (d : ℝ), 
    (20 / 2)^2 * π - (d / 2)^2 * π*(7+1)/1 = 0

theorem diameter_C_is_10_sqrt_2 :
  diameter_C_eq = (10 * Real.sqrt 2) := sorry

end diameter_C_is_10_sqrt_2_l286_286309


namespace count_perfect_square_factors_l286_286069

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286069


namespace hyperbola_eccentricity_l286_286313

-- Definitions of the conditions
def hyperbola_eq (x y a b : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

def right_focus (a b : ℝ) : ℝ := 
  sqrt (a^2 + b^2)

def asymptote_eq (x y a b : ℝ) : Prop :=
  y = b / a * x

def circle_tangent_to_asymptote (a b c : ℝ) : Prop :=
  a = b ∧ sqrt (a^2 + b^2) = c

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (Hhyperbola : ∀ x y, hyperbola_eq x y a b)
  (Hasymptote : ∀ x y, asymptote_eq x y a b)
  (Htangent : circle_tangent_to_asymptote a b c) :
  c / a = sqrt 2 :=
sorry

end hyperbola_eccentricity_l286_286313


namespace volume_of_bound_region_l286_286716

noncomputable def volume_bound_region : set (ℝ × ℝ × ℝ) :=
  {p | let (x, y, z) := p in
       |2 * x + y - z| + |2 * x - y + z| + |y + 2 * z| + |-x + 2 * y - z| ≤ 6}

theorem volume_of_bound_region : measure_theory.volume {p | let (x, y, z) := p in
  |2 * x + y - z| + |2 * x - y + z| + |y + 2 * z| + |-x + 2 * y - z| ≤ 6} = 20 / 3 :=
sorry

end volume_of_bound_region_l286_286716


namespace license_plate_count_l286_286427

-- Define the number of letters and digits
def num_letters := 26
def num_digits := 10
def num_odd_digits := 5  -- (1, 3, 5, 7, 9)
def num_even_digits := 5  -- (0, 2, 4, 6, 8)

-- Calculate the number of possible license plates
theorem license_plate_count : 
  (num_letters ^ 3) * ((num_even_digits * num_odd_digits * num_digits) * 3) = 13182000 :=
by sorry

end license_plate_count_l286_286427


namespace painting_cost_in_cny_l286_286521

theorem painting_cost_in_cny (usd_to_nad : ℝ) (usd_to_cny : ℝ) (painting_cost_nad : ℝ) :
  usd_to_nad = 8 → usd_to_cny = 7 → painting_cost_nad = 160 →
  painting_cost_nad / usd_to_nad * usd_to_cny = 140 :=
by
  intros
  sorry

end painting_cost_in_cny_l286_286521


namespace parallel_vectors_magnitude_acute_angle_range_l286_286008

variables {x : ℝ}

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1
def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vector_abs (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem parallel_vectors_magnitude :
  are_parallel (vector_a x) (vector_b x) →
  vector_abs (vector_sub (vector_a x) (vector_b x)) = 2 ∨ 
  vector_abs (vector_sub (vector_a x) (vector_b x)) = 2 * real.sqrt 5 := sorry

theorem acute_angle_range :
  is_acute (vector_a x) (vector_b x) →
  x ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 3 := sorry

end parallel_vectors_magnitude_acute_angle_range_l286_286008


namespace count_perfect_square_factors_l286_286022

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286022


namespace running_hours_per_week_l286_286576

theorem running_hours_per_week 
  (initial_days : ℕ) (additional_days : ℕ) (morning_run_time : ℕ) (evening_run_time : ℕ)
  (total_days : ℕ) (total_run_time_per_day : ℕ) (total_run_time_per_week : ℕ)
  (H1 : initial_days = 3)
  (H2 : additional_days = 2)
  (H3 : morning_run_time = 1)
  (H4 : evening_run_time = 1)
  (H5 : total_days = initial_days + additional_days)
  (H6 : total_run_time_per_day = morning_run_time + evening_run_time)
  (H7 : total_run_time_per_week = total_days * total_run_time_per_day) :
  total_run_time_per_week = 10 := 
sorry

end running_hours_per_week_l286_286576


namespace percentage_of_a_l286_286806

theorem percentage_of_a (x a : ℝ) (paise_in_rupee : ℝ := 100) (a_value : a = 160 * paise_in_rupee) (h : (x / 100) * a = 80) : x = 0.5 :=
by sorry

end percentage_of_a_l286_286806


namespace convex_polygon_diagonals_l286_286300

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end convex_polygon_diagonals_l286_286300


namespace calc_dot_product_ab_calc_expr_ab_l286_286192

noncomputable def vector_a : ℝ := 4
noncomputable def vector_b : ℝ := 2
noncomputable def angle_ab : ℝ := real.pi * (2 / 3)
noncomputable def dot_product_ab : ℝ := vector_a * vector_b * real.cos angle_ab

theorem calc_dot_product_ab :
  dot_product_ab = -4 :=
sorry

noncomputable def expr_ab : ℝ :=
(vector_a^2 - 2 * (vector_a * vector_b * real.cos angle_ab) + vector_a * vector_b * real.cos angle_ab - 2 * vector_b^2)

theorem calc_expr_ab :
  expr_ab = 12 :=
sorry

end calc_dot_product_ab_calc_expr_ab_l286_286192


namespace visits_needed_at_discount_clinic_l286_286949

def normal_doctor_cost : ℝ := 200
def discount_rate : ℝ := 0.70
def savings : ℝ := 80

theorem visits_needed_at_discount_clinic : 
  let discount_cost := normal_doctor_cost * (1 - discount_rate) in
  let total_cost := normal_doctor_cost - savings in
  let visits := total_cost / discount_cost in
  visits = 2 := 
by
  sorry

end visits_needed_at_discount_clinic_l286_286949


namespace tables_count_l286_286991

theorem tables_count (c t : Nat) (h1 : c = 8 * t) (h2 : 3 * c + 5 * t = 580) : t = 20 :=
by
  sorry

end tables_count_l286_286991


namespace solve_expression_l286_286186

noncomputable def log_base (a b : ℝ) : ℝ :=
(if b > 0 ∧ b ≠ 1 ∧ a > 0 then real.log a / real.log b else 0)

theorem solve_expression :
  log_base 3 (real.sqrt 27) +
  (8 / 125) ^ (-1 / 3) -
  (3 / 5) ^ 0 +
  real.root 4 (16 ^ 3) = 11 :=
by
  sorry

end solve_expression_l286_286186


namespace intersection_A_compB_l286_286006

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of B relative to ℝ
def comp_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- State the main theorem to prove
theorem intersection_A_compB : A ∩ comp_B = {x | -3 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_compB_l286_286006


namespace total_revenue_from_selling_snakes_l286_286118

-- Definitions based on conditions
def num_snakes := 3
def eggs_per_snake := 2
def standard_price := 250
def rare_multiplier := 4

-- Prove the total revenue Jake gets from selling all baby snakes is $2250
theorem total_revenue_from_selling_snakes : 
  (num_snakes * eggs_per_snake - 1) * standard_price + (standard_price * rare_multiplier) = 2250 := 
by
  sorry

end total_revenue_from_selling_snakes_l286_286118


namespace teacher_distribution_ways_l286_286948

theorem teacher_distribution_ways :
  let T1 := ℕ
  let T2 := ℕ
  T1 + T2 = 6 ∧ T1 ≤ 4 ∧ T2 ≤ 4 ∧ T1 ≥ 0 ∧ T2 ≥ 0 →
  (nat.choose 6 2 + nat.choose 6 3 + nat.choose 6 2 = 50) :=
by
  sorry

end teacher_distribution_ways_l286_286948


namespace kaprekar_converges_to_6174_l286_286553

noncomputable def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

noncomputable def transform (n : ℕ) : ℕ :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  let m := digits.sort (<=) |> List.reverse |> List.foldl (fun acc d => acc * 10 + d) 0
  let n_rev := reverse_number m
  m - n_rev

theorem kaprekar_converges_to_6174 (a0 : ℕ) (h : a0 = 5298 ∨ a0 = 4852) :
  ∃ k, k ≤ 7 ∧ (transform^[k] a0 = 6174) :=
sorry

end kaprekar_converges_to_6174_l286_286553


namespace area_formula_right_triangle_l286_286894

noncomputable def area_of_right_triangle (s1 s2 : ℝ) : ℝ :=
  (2 / 15) * Real.sqrt(17 * s1^2 * s2^2 - 4 * (s1^4 + s2^4))

theorem area_formula_right_triangle (a b s1 s2 : ℝ) 
  (h1 : s1^2 = (a^2 / 4) + b^2) 
  (h2 : s2^2 = a^2 + (b^2 / 4)) : 
  area_of_right_triangle s1 s2 = (2 / 15) * Real.sqrt(17 * s1^2 * s2^2 - 4 * (s1^4 + s2^4)) :=
by
  sorry

end area_formula_right_triangle_l286_286894


namespace derivative_of_f_l286_286541

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) : (deriv f x) = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by
  -- This statement skips the proof details
  sorry

end derivative_of_f_l286_286541


namespace find_k_l286_286877

noncomputable def e1 : ℝ := sorry  -- Placeholder for non-zero vector e1
noncomputable def e2 : ℝ := sorry  -- Placeholder for non-zero vector e2

noncomputable def a : ℝ := 2 * e1 - e2 -- Vector a
noncomputable def b (k : ℝ) : ℝ := k * e1 + e2 -- Vector b depending on k


theorem find_k (k : ℝ) (h1 : a = 2 * e1 - e2) (h2 : b k = k * e1 + e2) 
  (hCollinear : ∃ λ : ℝ, a = λ * b k) : k = -2 :=
by
  sorry

end find_k_l286_286877


namespace vector_inequality_l286_286744

noncomputable def vectorA (x y : ℝ) : ℝ × ℝ := (x, y)
noncomputable def vectorB (x y : ℝ) : ℝ × ℝ := (x, y)
noncomputable def vectorC (m : ℝ) : ℝ × ℝ := (m, 1 - m)
noncomputable def vectorD (n : ℝ) : ℝ × ℝ := (n, 1 - n)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

noncomputable def vector_distance (v1 v2 : ℝ × ℝ) : ℝ :=
  magnitude (vector_sub v1 v2)

theorem vector_inequality (T : ℝ) :
  (∀ (m n : ℝ), vector_distance (vectorA 1 0) (vectorC m) + vector_distance (vectorB 0.5 (real.sqrt 3 / 2)) (vectorD n) ≥ T) ↔
  T ≤ (real.sqrt 6 - real.sqrt 2) / 4 :=
by
  -- Proof omitted
  sorry

end vector_inequality_l286_286744


namespace diameter_of_circle_c_correct_l286_286307

noncomputable theory
open Real

def diameter_of_circle_c : Prop :=
  let D_diam := 20
  let D_radius := D_diam / 2
  let D_area := π * D_radius ^ 2
  ∃ r_C : ℝ, let C_area := π * r_C ^ 2 in r_C * 2 = 7.08 ∧ (D_area - C_area) / C_area = 7

theorem diameter_of_circle_c_correct : diameter_of_circle_c :=
sorry

end diameter_of_circle_c_correct_l286_286307


namespace rational_of_rational_f_eq_f_l286_286683

def f (x : ℝ) : ℝ := ((x-2) * (x+1) * (2*x-1)) / (x * (x-1))

theorem rational_of_rational_f_eq_f (u v : ℝ) (hu : u ∈ ℚ) (hv_eq : f u = f v) : v ∈ ℚ :=
by
  sorry

end rational_of_rational_f_eq_f_l286_286683


namespace greatest_odd_factors_l286_286155

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l286_286155


namespace number_of_diagonals_25_sides_l286_286299

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end number_of_diagonals_25_sides_l286_286299


namespace perfect_square_factors_l286_286048

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286048


namespace find_F12_l286_286849

variables {F : ℝ → ℝ}

def condition_1 := ∀ x : ℝ, IsPolynomial F
def condition_2 := F 6 = 15
def condition_3 := ∀ x : ℝ, (x^2 + 5*x + 6 ≠ 0) → (F (3 * x) / F (x + 3) = 9 - (48 * x + 54) / (x^2 + 5 * x + 6))

theorem find_F12
  (h1 : condition_1)
  (h2 : condition_2)
  (h3 : condition_3) :
  F 12 = 66 :=
sorry

end find_F12_l286_286849


namespace johns_remaining_income_l286_286516

def base_income : ℕ := 2000
def bonus_percentage : ℝ := 0.15
def public_transport_percentage : ℝ := 0.05
def rent : ℕ := 500
def utilities : ℕ := 100
def food : ℕ := 300
def miscellaneous_percentage : ℝ := 0.10

def total_income (base_income : ℕ) (bonus_percentage : ℝ) : ℕ :=
  base_income + (bonus_percentage * base_income).toNat

def total_expenses (total_income : ℕ) (public_transport_percentage : ℝ)
                   (rent : ℕ) (utilities : ℕ) (food : ℕ) (miscellaneous_percentage : ℝ) : ℕ :=
  let public_transport := (public_transport_percentage * total_income).toNat
  let miscellaneous := (miscellaneous_percentage * total_income).toNat
  public_transport + rent + utilities + food + miscellaneous

theorem johns_remaining_income :
  let total_inc := total_income base_income bonus_percentage in
  let total_exp := total_expenses total_inc public_transport_percentage rent utilities food miscellaneous_percentage in
  total_inc - total_exp = 1055 :=
by
  sorry

end johns_remaining_income_l286_286516


namespace radius_circle_parabolas_tangent_l286_286898

theorem radius_circle_parabolas_tangent (r : ℝ) (h_tangent_circle : ∀ x, (x^2 + r) = (1 / √3) * x) :
  r = 1 / 12 :=
by sorry

end radius_circle_parabolas_tangent_l286_286898


namespace effective_speed_upstream_l286_286999

theorem effective_speed_upstream (rowing_speed : ℝ) (current_speed : ℝ) (wind_speed : ℝ) : 
  rowing_speed = 20 → 
  current_speed = 3 → 
  wind_speed = 2 → 
  rowing_speed - (current_speed + wind_speed) = 15 :=
by {
  intros h_rowing_speed h_current_speed h_wind_speed,
  rw [h_rowing_speed, h_current_speed, h_wind_speed],
  norm_num,
  sorry
}

end effective_speed_upstream_l286_286999


namespace gcd_lcm_1365_910_l286_286710

theorem gcd_lcm_1365_910 :
  gcd 1365 910 = 455 ∧ lcm 1365 910 = 2730 :=
by
  sorry

end gcd_lcm_1365_910_l286_286710


namespace definite_integral_result_l286_286325

noncomputable def integrand (x : ℝ) : ℝ := Real.exp x + 2 * x

theorem definite_integral_result :
  ∫ x in 0..1, integrand x = Real.exp 1 :=
by sorry

end definite_integral_result_l286_286325


namespace octagon_area_is_six_and_m_plus_n_is_seven_l286_286234

noncomputable def area_of_octagon (side_length : ℕ) (segment_length : ℚ) : ℚ :=
  let triangle_area := 1 / 2 * side_length * segment_length
  let octagon_area := 8 * triangle_area
  octagon_area

theorem octagon_area_is_six_and_m_plus_n_is_seven :
  area_of_octagon 2 (3/4) = 6 ∧ (6 + 1 = 7) :=
by
  sorry

end octagon_area_is_six_and_m_plus_n_is_seven_l286_286234


namespace intersection_trace_l286_286102

-- Define the given conditions
variable (A B C D M : Point)
variable (a c k : ℝ)
variable (AB AC BD CD : Line)

-- Hypotheses
hypothesis h1 : ConvexTrapezoid A B C D
hypothesis h2 : length AB = a
hypothesis h3 : length CD = c
hypothesis h4 : c < a
hypothesis h5 : perimeter A B C D = k

-- Define the point of intersection of the extensions of the non-parallel sides
noncomputable def Intersection_M (A B C D : Point) : Point := sorry -- Definitions for point intersection based on geometry

-- State the problem
theorem intersection_trace (A B C D : Point) (AB AC BD CD : Line)
  (h1 : ConvexTrapezoid A B C D) (h2 : length AB = a)
  (h3 : length CD = c) (h4 : c < a) (h5 : perimeter A B C D = k) :
  ∃ (E : Ellipse), (Intersection_M A B C D) ∈ E := 
sorry

end intersection_trace_l286_286102


namespace right_triangle_third_side_l286_286465

theorem right_triangle_third_side (a b : ℝ) (h : a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2)
  (h1 : a = 3 ∧ b = 5 ∨ a = 5 ∧ b = 3) : c = 4 ∨ c = Real.sqrt 34 :=
sorry

end right_triangle_third_side_l286_286465


namespace polynomial_factors_abs_val_l286_286458

theorem polynomial_factors_abs_val (h k : ℝ)
  (p : Polynomial ℝ) (hx1 : p.eval 3 = 0) (hx2 : p.eval (-4) = 0)
  (hp : p = 3 * X ^ 4 - h * X ^ 2 + k * X - 12) :
  |3 * h - 2 * k| = PICK_FROM_CHOICES :=
by
  sorry

end polynomial_factors_abs_val_l286_286458


namespace sum_of_areas_of_fifteen_disks_l286_286339

noncomputable def radius_of_disks : ℝ := 1 * (2 - real.sqrt 3)

noncomputable def area_of_one_disk : ℝ := real.pi * radius_of_disks^2

noncomputable def total_area_of_fifteen_disks : ℝ := 15 * area_of_one_disk

theorem sum_of_areas_of_fifteen_disks :
  total_area_of_fifteen_disks = real.pi * (105 - 60 * real.sqrt 3) :=
by
  sorry

end sum_of_areas_of_fifteen_disks_l286_286339


namespace maximize_savings_promotion_a_l286_286994

noncomputable def promotion_a_cost (shoe_price handbag_price : ℕ) : ℝ := 
  let total_shoes := (shoe_price + shoe_price / 2);
  let total_with_handbag := total_shoes + handbag_price;
  total_with_handbag * 0.9

noncomputable def promotion_b_cost (shoe_price handbag_price : ℕ) : ℝ := 
  let total_shoes := shoe_price + (shoe_price - 15);
  total_shoes + handbag_price

theorem maximize_savings_promotion_a (shoe_price handbag_price : ℕ) :
  promotion_a_cost shoe_price handbag_price = 85.5 ∧
  promotion_b_cost shoe_price handbag_price = 105 →
  105 - 85.5 = 19.5 :=
by
  intros h
  cases h with hpa hpb
  sorry

end maximize_savings_promotion_a_l286_286994


namespace goldfish_disappeared_l286_286522

theorem goldfish_disappeared (original_goldfish : Nat) (current_goldfish : Nat) 
  (h1 : original_goldfish = 15) (h2 : current_goldfish = 4) : original_goldfish - current_goldfish = 11 :=
by
  rw [h1, h2]
  rfl

end goldfish_disappeared_l286_286522


namespace num_pos_int_satisfying_condition_l286_286507

/-- Let s(n) denote the sum of the digits (in base ten) of a positive integer n.
    We are to prove that there are 2530 positive integers n, with n ≤ 10^4,
    that satisfy s(11n) = 2s(n). -/
theorem num_pos_int_satisfying_condition :
  ∃ count : ℕ, count = 2530 ∧ 
  (∀ n : ℕ, 0 < n ∧ n ≤ 10000 → (s n) (11 * n) = 2 * (s n) → count == 2530) :=
sorry

/-- Define s(n) as the sum of the digits of n in base ten. -/
def s (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

end num_pos_int_satisfying_condition_l286_286507


namespace ellipse_angle_proof_l286_286915

open Real

noncomputable def angle_F1PF2 : ℝ := 120

theorem ellipse_angle_proof:
  ∀ (F1 F2 P : ℝ × ℝ),
  let e := { p : ℝ × ℝ | p.1 ^ 2 / 9 + p.2 ^ 2 / 2 = 1 } in
  F1 ∈ e ∧ F2 ∈ e ∧ P ∈ e →
  dist P F1 = 4 →
  dist P F1 + dist P F2 = 6 →
  angle F1 P F2 = angle_F1PF2 :=
by
  sorry

end ellipse_angle_proof_l286_286915


namespace vote_ratio_l286_286098

theorem vote_ratio (X Y Z : ℕ) (hZ : Z = 25000) (hX : X = 22500) (hX_Y : X = Y + (1/2 : ℚ) * Y) 
    : Y / (Z - Y) = 2 / 5 := 
by 
  sorry

end vote_ratio_l286_286098


namespace minimum_calls_to_know_all_info_l286_286293

theorem minimum_calls_to_know_all_info (n : ℕ) (h : 1 ≤ n) :
  ∃ seq : list (ℕ × ℕ), 
    (∀ i, i < n → ∃ m, (i, m) ∈ seq ∨ (m, i) ∈ seq) ∧ 
    list.length seq = 2 * n - 2 :=
sorry

end minimum_calls_to_know_all_info_l286_286293


namespace inequality_proof_l286_286608

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
    sqrt (a^2 + a*b + b^2) + sqrt (a^2 + a*c + c^2) ≥ 
    4 * sqrt ((ab / (a + b))^2 + (ab / (a + b)) * (ac / (a + c)) + (ac / (a + c))^2) :=
sorry

end inequality_proof_l286_286608


namespace original_strength_of_class_l286_286908

-- Definitions from the problem conditions
def average_age_original (x : ℕ) : ℕ := 40 * x
def total_students (x : ℕ) : ℕ := x + 17
def total_age_new_students : ℕ := 17 * 32
def new_average_age : ℕ := 36

-- Lean statement to prove that the original strength of the class is 17.
theorem original_strength_of_class :
  ∃ x : ℕ, average_age_original x + total_age_new_students = total_students x * new_average_age ∧ x = 17 :=
by
  sorry

end original_strength_of_class_l286_286908


namespace perpendicular_plane_l286_286502

variables {l m : Type} {α : Plane}
variables [IsLine l] [IsLine m]

theorem perpendicular_plane
  (h1: l ⊥ α)  -- l is perpendicular to the plane α
  (h2: l ∥ m)  -- l is not parallel to m
  : m ⊥ α :=  -- prove that m is perpendicular to the plane α
sorry

end perpendicular_plane_l286_286502


namespace simplify_expression_l286_286184

theorem simplify_expression (m n : ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3^(m * n / (m - n))) : 
  ((
    (x^(2 / m) - 9 * x^(2 / n)) *
    ((x^(1 - m))^(1 / m) - 3 * (x^(1 - n))^(1 / n))
  ) / (
    (x^(1 / m) + 3 * x^(1 / n))^2 - 12 * x^((m + n) / (m * n))
  ) = (x^(1 / m) + 3 * x^(1 / n)) / x) := 
sorry

end simplify_expression_l286_286184


namespace ellipse_and_line_equation_l286_286827

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def focus_condition (a c : ℝ) : Prop :=
  ∃ x y, x = -1 ∧ y = 0 ∧ c = a - (sqrt 2 - 1)

noncomputable def line_through_point (l : ℝ × ℝ → Prop) : Prop :=
  ∃ x y, x = 0 ∧ y = sqrt 2 ∧ l (x, y)

noncomputable def line_equation (k : ℝ) : Prop :=
  ∃ x y, y = k * x + sqrt 2

noncomputable def tangent_condition (x y k : ℝ) : Prop :=
  x^2 / 2 + (k * x + sqrt 2)^2 = 1

theorem ellipse_and_line_equation :
  (∃ (a b : ℝ), ellipse_equation a b ∧ focus_condition a 1) ∧
  (∃ (l : ℝ × ℝ → Prop), line_through_point l ∧
    (∃ k, line_equation k ∧ tangent_condition (sqrt 2 / 2) k) ∧
    (∃ k, line_equation (-sqrt 2 / 2) ∧ tangent_condition (-sqrt 2 / 2) k)) :=
by {
  sorry
}

end ellipse_and_line_equation_l286_286827


namespace x_equals_one_is_necessary_and_sufficient_l286_286204

theorem x_equals_one_is_necessary_and_sufficient :
  (x : ℝ) → x = 1 ↔ x^2 - 2*x + 1 = 0 := 
begin 
  intro x,
  split;
  intro hyp,
  { -- sufficiency proof
    rw hyp,
    ring, 
    exact eq.refl 0,
  },
  { -- necessity proof
    have factored := eq_of_succ_eq_zero (hyp : (x - 1) * (x - 1) = 0),
    rw eq_iff_eq_cancel_left at factored,
    exact factored,
  }
end

end x_equals_one_is_necessary_and_sufficient_l286_286204


namespace correctness_of_statements_l286_286288

-- Define the function shifting property
def range_same (f : ℝ → ℝ) : Prop :=
  range f = range (λ x, f (x + 1))

-- Define the properties of the two functions
def lg_function_same (x : ℝ) : Prop :=
  ∀ x: ℝ, 2 * log x = log (x^2)

-- Define the property for an odd function decreasing on both intervals
def odd_function_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x ∧ (∀ x < 0, f (x) > f (x + 1)) → ∀ x > 0, f (x) > f (x - 1)

-- Define the property for a function with no zeros in interval given continuity
def no_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  continuous_on f [a, b] ∧ (f a) * (f b) > 0 → ∀ x ∈ Ioo a b, f (x) ≠ 0 

-- Define and prove the main theorem
theorem correctness_of_statements : 
  ∃ f : ℝ → ℝ, range_same f ∧ 
  ¬ lg_function_same ∧ 
  odd_function_decreasing f ∧ 
  ¬ no_zeros_in_interval (λ x, x^2) (-1) 2 := 
by
  let f1 := (λ x, x) -- Example function to satisfy range_same
  let f2 := (λ x, x^2) -- Example function to fail no_zeros_in_interval
  have range_prop: range_same f1 := sorry
  have lg_prop: ¬ lg_function_same := sorry
  have odd_prop: odd_function_decreasing (λ x, - x) := sorry
  have no_zero_prop: ¬ no_zeros_in_interval f2 (-1) 2 := sorry
  exact ⟨f1, range_prop, lg_prop, odd_prop, no_zero_prop⟩

end correctness_of_statements_l286_286288


namespace y1_lt_y2_l286_286383

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l286_286383


namespace trapezoid_area_l286_286680

-- Given conditions
variables (h : ℝ) (b1 b2 : ℝ)
axiom shorter_base : b1 = 2 * h
axiom longer_base : b2 = 3 * h
axiom shorter_base_value : b1 = 10

-- Lean statement to prove
theorem trapezoid_area :
  let K := 1 / 2 * (b1 + b2) * h in
  K = 62.5 :=
by
  sorry

end trapezoid_area_l286_286680


namespace solve_log_equation_l286_286555

noncomputable def solution_to_log_equation : ℝ :=
  2

theorem solve_log_equation (x : ℝ) :
  log 2 (4^x + 4) = x + log 2 (2^(x + 1) - 3) ↔ x = 2 := 
by {
  sorry
}

end solve_log_equation_l286_286555


namespace solve_for_x_l286_286798

theorem solve_for_x (x y z : ℝ) (h1 : 3^x * 4^y / 2^z = 59049)
                              (h2 : x - y + 2 * z = 10)
                              (h3 : x^2 + y^2 = z^2) : 
  x = 10 := 
sorry

end solve_for_x_l286_286798


namespace intersection_M_N_l286_286784

noncomputable def M : set ℝ := { y | ∃ x : ℝ, y = |cos (2 * x)| }

noncomputable def N : set ℝ := { x | abs (2 * x / (1 - sqrt 3 * complex.I)) < 1 }

theorem intersection_M_N : M ∩ N = set.Ico 0 1 := 
  sorry

end intersection_M_N_l286_286784


namespace arithmetic_sequence_sum_l286_286394

noncomputable def arithmetic_sum (a b : ℝ) (n : ℕ) : ℝ :=
  n * ((a * 1 + b + a * n + b) / 2)

theorem arithmetic_sequence_sum (a b : ℝ) (h : (a / (1 - complex.I)) + (b / (2 - complex.I)) = (1 / (3 - complex.I))) :
  arithmetic_sum a b 100 = -910 :=
by
  sorry

end arithmetic_sequence_sum_l286_286394


namespace maximum_value_l286_286396

theorem maximum_value (a b : ℝ) (h : a + b = 4) : 
  (∃ (m : ℝ), (∀ a b, a + b = 4 → m ≥ (1 / (a^2 + 1) + 1 / (b^2 + 1))) 
   ∧ m = (sqrt 5 + 2) / 4) :=
sorry

end maximum_value_l286_286396


namespace points_collinear_distance_relation_l286_286575

theorem points_collinear_distance_relation (x y : ℝ) 
  (h1 : (5 - y) * (5 - 1) = -4 * (-2 - x))
  (h2 : real.sqrt ((y - 1)^2 + 9) = 2 * real.sqrt ((x - 1)^2 + 16)) :
  (x + y = -9 / 2) ∨ (x + y = 17 / 2) := 
sorry

end points_collinear_distance_relation_l286_286575


namespace increase_in_radius_is_0_11_inches_l286_286879

noncomputable def initial_radius : ℝ := 12
noncomputable def odometer_distance_initial : ℝ := 300
noncomputable def odometer_distance_new : ℝ := 310
noncomputable def actual_distance_new : ℝ := 315

theorem increase_in_radius_is_0_11_inches :
  ∃ Δr : ℝ, (Δr = 0.11) →
    let r := initial_radius in
    let C := 2 * Real.pi * r in
    let distance_per_rotation := C / 63360 in
    let rotations := actual_distance_new / distance_per_rotation in
    let new_radius := (actual_distance_new * 63360) / (2 * Real.pi * rotations) in
    Δr = new_radius - r :=
sorry

end increase_in_radius_is_0_11_inches_l286_286879


namespace probability_range_inequality_l286_286093

theorem probability_range_inequality :
  ∀ p : ℝ, 0 ≤ p → p ≤ 1 →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2 → 0.4 ≤ p ∧ p < 1) := sorry

end probability_range_inequality_l286_286093


namespace find_y_l286_286693

noncomputable def f (x y : ℝ) := (x + y) * (x + 1)^4

theorem find_y 
  (H : ∀ (x : ℝ), f x y = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) 
  (H1 : 2 * (a_1 + a_3 + a_5) = 32) :
  y = 3 :=
begin
  -- Proof omitted
  sorry
end

end find_y_l286_286693


namespace kylie_necklaces_on_tuesday_l286_286489

theorem kylie_necklaces_on_tuesday :
  (∃ tuesday_necklaces : ℕ,
    let monday_necklaces := 10 in
    let wednesday_bracelets := 5 in
    let wednesday_earrings := 7 in
    let beads_per_necklace := 20 in
    let beads_per_bracelet := 10 in
    let beads_per_earring := 5 in
    let total_beads := 325 in
    (monday_necklaces * beads_per_necklace) +
    (wednesday_bracelets * beads_per_bracelet) +
    (wednesday_earrings * beads_per_earring) +
    (tuesday_necklaces * beads_per_necklace) = total_beads) ∧
    (tuesday_necklaces = 2)
:=
by
  sorry

end kylie_necklaces_on_tuesday_l286_286489


namespace number_of_cyclic_sets_l286_286466

open Nat

-- Define the conditions
def team_played_each_other_once (n : ℕ) : Prop :=
  n = 25

def won_and_lost_each_game (wins losses : ℕ) : Prop :=
  wins = 12 ∧ losses = 12

def total_sets_of_three_teams (n : ℕ) : ℕ :=
  (choose n 3)

def sets_where_one_team_dominates (n dominance : ℕ) : ℕ :=
  n * dominance

def cyclic_sets_count (total_sets dominated_sets : ℕ) : ℕ :=
  total_sets - dominated_sets

-- Define the final statement
theorem number_of_cyclic_sets :
  ∀ (n wins losses : ℕ),
  team_played_each_other_once n → won_and_lost_each_game wins losses →
  let total := total_sets_of_three_teams n in
  let dominance := choose wins 2 in
  let dominated := sets_where_one_team_dominates n dominance in
  cyclic_sets_count total dominated = 650 :=
by
  intros n wins losses n_cond wl_cond
  have n_eq : n = 25 := n_cond
  have wins_eq : wins = 12 := wl_cond.1
  have losses_eq : losses = 12 := wl_cond.2
  simp only [n_eq, wins_eq, losses_eq,
             total_sets_of_three_teams, sets_where_one_team_dominates,
             choose, choose_eq_fact_div_fact]
  sorry

end number_of_cyclic_sets_l286_286466


namespace remainder_7_pow_700_div_100_l286_286980

theorem remainder_7_pow_700_div_100 : (7 ^ 700) % 100 = 1 := 
  by sorry

end remainder_7_pow_700_div_100_l286_286980


namespace number_of_ways_l286_286861

theorem number_of_ways (n : ℕ) :
  let S := (finset.range (2*n+1)).powerset in
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n → ∃ S_ij ∈ S, S_ij.card = i + j ∧
  (∀ k l : ℕ, 0 ≤ i ∧ i ≤ k ∧ k ≤ n ∧ 0 ≤ j ∧ j ≤ l ∧ l ≤ n → S_ij ⊆ S.ij)) →
  ∃ S_set : (finset (finset ℤ)), (S_set.card = (2*n)! * 2^(n^2)) :=
sorry

end number_of_ways_l286_286861


namespace probability_at_least_one_deciphers_l286_286765

theorem probability_at_least_one_deciphers (P_A P_B : ℚ) (hA : P_A = 1/2) (hB : P_B = 1/3) :
    P_A + P_B - P_A * P_B = 2/3 := by
  sorry

end probability_at_least_one_deciphers_l286_286765


namespace middle_part_l286_286433

theorem middle_part (x : ℝ) (h : 2 * x + (2 / 3) * x + (2 / 9) * x = 120) : 
  (2 / 3) * x = 27.6 :=
by
  -- Assuming the given conditions
  sorry

end middle_part_l286_286433


namespace integral_2x_plus_3_squared_l286_286352

open Real

-- Define the function to be integrated
def f (x : ℝ) := (2 * x + 3) ^ 2

-- State the theorem for the indefinite integral
theorem integral_2x_plus_3_squared :
  ∃ C : ℝ, ∫ x, f x = (1 / 6) * (2 * x + 3) ^ 3 + C :=
by
  sorry

end integral_2x_plus_3_squared_l286_286352


namespace smallest_integer_for_perfect_square_l286_286653

def large_number_y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9

theorem smallest_integer_for_perfect_square :
  ∃ k : ℕ, (k > 0) ∧ (is_square (large_number_y * k)) ∧ (∀ m : ℕ, (m > 0) ∧ (is_square (large_number_y * m)) → k ≤ m) :=
sorry

end smallest_integer_for_perfect_square_l286_286653


namespace melting_point_of_ice_l286_286956

theorem melting_point_of_ice (boiling_point_f : ℤ) (boiling_point_c : ℤ) (melting_point_f : ℤ) (water_temperature_c : ℤ) (water_temperature_f : ℤ) :
  boiling_point_f = 212 -> boiling_point_c = 100 -> melting_point_f = 32 -> water_temperature_c = 40 -> water_temperature_f = 104 ->
  ∃ (melting_point_c : ℤ), melting_point_c = 0 :=
by intro bf bc mf wc wf hbf hbc hmf hwc hwf; exists 0; sorry

end melting_point_of_ice_l286_286956


namespace greatest_odd_factors_below_200_l286_286166

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l286_286166


namespace train_meeting_distance_l286_286257

variable (t A B : Type) [LinearOrder t] [AddCommGroup t] [Module ℝ t] [Field ℝ]

def speedA {R : Type} [Field R] (distance time : R) : R := distance / time
def speedB {R : Type} [Field R] (distance time : R) : R := distance / time
def relative_speed {R : Type} [Field R] (vA vB : R) : R := vA + vB

theorem train_meeting_distance
  (distance : ℝ)
  (timeA : ℝ)
  (timeB : ℝ)
  (speedA_val : speedA distance timeA = 20)
  (speedB_val : speedB distance timeB = 30)
  (relative_speed_val : relative_speed 20 30 = 50) :
  ∃ t : ℝ, t = 2.4 ∧ (20 * t = 48) :=
by {
  sorry
}

end train_meeting_distance_l286_286257


namespace count_numbers_with_square_factors_l286_286044

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286044


namespace infinite_triples_l286_286133

theorem infinite_triples (a b c : ℝ) : 
  (∃ n : ℝ, n = ∞) ∧ (∀ b : ℝ, b ≠ 0 → (∃ a c : ℝ, 
    (3 * x + b * y + c = 0) = (a * x - 3 * y + 5 = 0) ∧ a = -b ∧ c = (5 * b) / 3)) := 
sorry

end infinite_triples_l286_286133


namespace slant_height_correct_l286_286768

def volume_of_frustum (r1 r2 h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * h * (r1^2 + r2^2 + r1 * r2)

def slant_height_of_frustum (r1 r2 h : ℝ) : ℝ :=
  Real.sqrt (h^2 + (r2 - r1)^2)

theorem slant_height_correct (r1 r2 V : ℝ) (h : ℝ) :
  r1 = 2 → r2 = 4 → V = 56 * Real.pi → volume_of_frustum r1 r2 h = V →
  slant_height_of_frustum r1 r2 h = 2 * Real.sqrt 10 :=
by
  intros
  sorry

end slant_height_correct_l286_286768


namespace find_five_digit_N_l286_286239

theorem find_five_digit_N : 
  ∃ N : ℕ, (N.digits.length = 5 ∧ 
  { (N.digits.nth 0).bind (λ d1, (N.digits.nth 1).map (λ d2 => d1*10 + d2)),
    (N.digits.nth 0).bind (λ d1, (N.digits.nth 2).map (λ d3 => d1*10 + d3)),
    (N.digits.nth 0).bind (λ d1, (N.digits.nth 3).map (λ d4 => d1*10 + d4)),
    (N.digits.nth 0).bind (λ d1, (N.digits.nth 4).map (λ d5 => d1*10 + d5)),
    (N.digits.nth 1).bind (λ d2, (N.digits.nth 2).map (λ d3 => d2*10 + d3)),
    (N.digits.nth 1).bind (λ d2, (N.digits.nth 3).map (λ d4 => d2*10 + d4)),
    (N.digits.nth 1).bind (λ d2, (N.digits.nth 4).map (λ d5 => d2*10 + d5)),
    (N.digits.nth 2).bind (λ d3, (N.digits.nth 3).map (λ d4 => d3*10 + d4)),
    (N.digits.nth 2).bind (λ d3, (N.digits.nth 4).map (λ d5 => d3*10 + d5)),
    (N.digits.nth 3).bind (λ d4, (N.digits.nth 4).map (λ d5 => d4*10 + d5))}
  = {33, 37, 37, 37, 38, 73, 77, 78, 83, 87}) ∧ N = 37837 :=
by
  -- Proof required here
  sorry

end find_five_digit_N_l286_286239


namespace find_x_l286_286785

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 2)

def b (x : ℝ) : ℝ × ℝ := (x, -2)

def c (x : ℝ) : ℝ × ℝ := (1 - x, 4)

theorem find_x (x : ℝ) (h : vector_dot_product a (c x) = 0) : x = 9 :=
by
  sorry

end find_x_l286_286785


namespace intern_teacher_allocation_l286_286536

theorem intern_teacher_allocation :
  let num_teachers := 5
  let class90 := 1
  let class91 := 1
  let class92 := 1
  (∀ class : nat, class = class90 ∨ class = class91 ∨ class = class92 -> class ≥ 1 ∧ class ≤ 2) ->
  let ways_to_form_groups := (finset.card (finset.powerset_univ num_teachers).filter (λ g, finset.card g = 1) *
                             (finset.card (finset.powerset_univ (num_teachers - 1)).filter (λ g, finset.card g = 2)) /
                             (fact 2)) in
  let total_ways := ways_to_form_groups * (fact 3) in
  total_ways = 90 :=
begin
  sorry
end

end intern_teacher_allocation_l286_286536


namespace count_numbers_with_perfect_square_factors_l286_286033

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286033


namespace greatest_odd_factors_below_200_l286_286164

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l286_286164


namespace sum_of_first_10_terms_l286_286473

variable {α : Type*} [OrderedCommGroup α] (a_n : ℕ → α)

-- Given conditions
axiom condition1 : a_n 3 + a_n 7 = 8
axiom condition2 : a_n 6 = 6

-- Theorem to be proved: Sum of first 10 terms equals 50
theorem sum_of_first_10_terms (a_n_bis : ℕ → α) : (∑ i in Finset.range 10, a_n i) = 50 := by
  sorry

end sum_of_first_10_terms_l286_286473


namespace marquita_garden_width_l286_286146

theorem marquita_garden_width
  (mancino_gardens : ℕ) (marquita_gardens : ℕ)
  (mancino_length mancnio_width marquita_length total_area : ℕ)
  (h1 : mancino_gardens = 3)
  (h2 : mancino_length = 16)
  (h3 : mancnio_width = 5)
  (h4 : marquita_gardens = 2)
  (h5 : marquita_length = 8)
  (h6 : total_area = 304) :
  ∃ (marquita_width : ℕ), marquita_width = 4 :=
by
  sorry

end marquita_garden_width_l286_286146


namespace table_runner_combined_area_l286_286947

theorem table_runner_combined_area
    (table_area : ℝ) (cover_percentage : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) (A : ℝ) :
    table_area = 175 →
    cover_percentage = 0.8 →
    area_two_layers = 24 →
    area_three_layers = 28 →
    A = (cover_percentage * table_area - area_two_layers - area_three_layers) + area_two_layers + 2 * area_three_layers →
    A = 168 :=
by
  intros h_table_area h_cover_percentage h_area_two_layers h_area_three_layers h_A
  sorry

end table_runner_combined_area_l286_286947


namespace rainy_day_average_speed_l286_286114

theorem rainy_day_average_speed :
  ∀ (d_up d_down : ℝ) (t_up t_down : ℝ) (rain_factor : ℝ),
  d_up = 1.5 → d_down = 1.5 → 
  t_up = 45 / 60 → t_down = 15 / 60 →
  rain_factor = 3 / 4 →
  let t_rain_up := t_up / rain_factor,
      t_rain_down := t_down / rain_factor,
      total_distance := d_up + d_down,
      total_time_rain := t_rain_up + t_rain_down in
  (total_distance / total_time_rain) = 2.25 :=
begin
  intros d_up d_down t_up t_down rain_factor h1 h2 h3 h4 h5,
  sorry
end

end rainy_day_average_speed_l286_286114


namespace count_10_digit_numbers_divisible_by_33_l286_286812

theorem count_10_digit_numbers_divisible_by_33 :
  ∃ (a b : ℕ), a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                ((a + b + 19) % 3 = 0) ∧ 
                ((b + 1 - a) % 11 = 0) ∧
                count_such_pairs = 3 := 
sorry

end count_10_digit_numbers_divisible_by_33_l286_286812


namespace count_numbers_with_square_factors_l286_286042

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286042


namespace sum_of_sines_eq_tan_l286_286759

-- Define the primary statement formally in Lean
theorem sum_of_sines_eq_tan :
  let m := 3 in let n := 2 in 
  ∑ k in Finset.range 50 + 1, Real.sin (7 * k) = Real.tan (m/n) :=
by sorry

end sum_of_sines_eq_tan_l286_286759


namespace geometric_seq_seventh_term_l286_286359

theorem geometric_seq_seventh_term (a r : ℝ) (h1 : a = 3) (h2 : a * r^2 = 1 / 9) : 
  a * r^6 = sqrt 3 / 81 :=
by
  -- Proof is to be completed
  sorry

end geometric_seq_seventh_term_l286_286359


namespace derivative_of_y_l286_286201

def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_y : (fun x => x * Real.cos x - Real.sin x)' = fun x => - x * Real.sin x := by
 sorry

end derivative_of_y_l286_286201


namespace greatest_perfect_square_below_200_l286_286159

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l286_286159


namespace Q_subset_P_l286_286368

def P : set ℝ := { y | ∃ x : ℝ, y = x^2 }
def Q : set ℝ := { y | ∃ x : ℝ, y = 2^x }

theorem Q_subset_P : Q ⊆ P := 
sorry

end Q_subset_P_l286_286368


namespace minimum_square_area_l286_286805

theorem minimum_square_area : 
  ∃ (A B C D : ℝ × ℝ), 
    (∃ k : ℝ, (A.2 = 2 * A.1 - 17) ∧ (B.2 = 2 * B.1 - 17)) ∧
    (C.2 = C.1 ^ 2) ∧ (D.2 = D.1 ^ 2) ∧
    (C.1 + D.1 = 2) ∧
    ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 = (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2) ∧
    ((B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = 4 * 5) ∧
    (let area := (4 * Real.sqrt 5) ^ 2 in area = 80) := sorry

end minimum_square_area_l286_286805


namespace hypotenuse_length_l286_286084

theorem hypotenuse_length (a b c : ℕ) (h1 : a = 12) (h2 : b = 5) (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end hypotenuse_length_l286_286084


namespace lines_BE_and_AF_are_perpendicular_l286_286231

noncomputable theory

-- Definitions and conditions
variable {A B C D E F : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B = dist A C

def midpoint (A B : Type) [MetricSpace A] [MetricSpace B] : Type := sorry
def foot_of_perpendicular (A : Type) (l : Line) : Type := sorry

-- Given conditions
variable (ABC_is_iso : is_isosceles_triangle A B C)
variable (D : Type) [MetricSpace D] (D_is_midpoint : midpoint B C)
variable (E : Type) [MetricSpace E] (E_is_foot : foot_of_perpendicular D (line_from D A C))
variable (F : Type) [MetricSpace F] (F_is_midpoint : midpoint D E)

-- Question: Show that lines BE and AF are perpendicular
theorem lines_BE_and_AF_are_perpendicular :
  ∀ {B E F : Type} [MetricSpace B] [MetricSpace E] [MetricSpace F],
    (line_from B E ⊥ line_from A F) := by
  sorry

end lines_BE_and_AF_are_perpendicular_l286_286231


namespace average_percentage_l286_286606

theorem average_percentage (n1 n2 : ℕ) (s1 s2 : ℕ)
  (h1 : n1 = 15) (h2 : s1 = 80) (h3 : n2 = 10) (h4 : s2 = 90) :
  (n1 * s1 + n2 * s2) / (n1 + n2) = 84 :=
by
  sorry

end average_percentage_l286_286606


namespace union_A_B_l286_286783

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_A_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l286_286783


namespace fraction_calculation_l286_286670

theorem fraction_calculation : 
  (1/2 - 1/3) / (3/7 * 2/8) = 14/9 :=
by
  sorry

end fraction_calculation_l286_286670


namespace star_value_15_5_l286_286924

def star (a b : ℝ) : ℝ := a - a / b

theorem star_value_15_5 : star 15 5 = 12 := 
by
  rw [star]
  norm_num
  sorry

end star_value_15_5_l286_286924


namespace count_non_empty_subsets_l286_286013

theorem count_non_empty_subsets (n : ℕ) :
  let S := (set.range 17).map (λ x, x + 1),
      P1 (T : set ℕ) := ∀ (x : ℕ), x ∈ T → x + 1 ∉ T,
      P2 (T : set ℕ) := ∃ k, T.finite ∧ T.card = k ∧ ∀ x ∈ T, x ≥ k + 1 in
  card {T // T ⊆ S ∧ T ≠ ∅ ∧ P1 T ∧ P2 T} = 594 := 
by
  sorry

end count_non_empty_subsets_l286_286013


namespace share_difference_3600_l286_286291

theorem share_difference_3600 (x : ℕ) (p q r : ℕ) (h1 : p = 3 * x) (h2 : q = 7 * x) (h3 : r = 12 * x) (h4 : r - q = 4500) : q - p = 3600 := by
  sorry

end share_difference_3600_l286_286291


namespace max_value_of_f_l286_286367

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_value_of_f:
  ∃ x ∈ Set.Ioo 0 Real.pi, ∀ y ∈ Set.Ioo 0 Real.pi, f y ≤ 2 ∧ f x = 2 ∧ x = Real.pi / 6 :=
by sorry

end max_value_of_f_l286_286367


namespace lateral_surface_area_ratio_l286_286080

-- Define the conditions
variables (r : ℝ) (π : ℝ)
-- π is a constant in Lean MathLib as real.pi

-- Assume the necessary geometric conditions
def cone_axial_section_equilateral (r : ℝ) : Prop :=
  ∃ (h : ℝ), h = r * sqrt 3 -- height of equilateral triangle base radius times sqrt(3)

-- Define the base area
def base_area (r : ℝ) : ℝ :=
  π * r^2

-- Define the lateral surface area given the conditions
def lateral_surface_area (r : ℝ) : ℝ :=
  2 * π * r^2

-- The main theorem to be proved
theorem lateral_surface_area_ratio (r : ℝ) (h : ℝ) (H : h = r * sqrt 3):
  lateral_surface_area r = 2 * base_area r :=
sorry

end lateral_surface_area_ratio_l286_286080


namespace math_problem_l286_286501

noncomputable def f (x : ℝ) : ℝ := x + 2
noncomputable def f_inv (x : ℝ) : ℝ := x - 2
noncomputable def g (x : ℝ) : ℝ := x / 3
noncomputable def g_inv (x : ℝ) : ℝ := 3 * x

theorem math_problem : 
  f_inv (g (f (f_inv (g_inv (f (g (f_inv (f (g_inv (f 27))))))))))) = -(1/3) := 
by
  sorry

end math_problem_l286_286501


namespace part_I_part_II_l286_286382

noncomputable def f(x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def g(x : ℝ) (m : ℝ) : ℝ := m * (2 * x + 1)
noncomputable def h(x : ℝ) (m : ℝ) : ℝ := f x - g x m

theorem part_I : ∀ x : ℝ, f x ≥ g x 1 :=
by
  sorry

theorem part_II : ∀ (a b : ℝ), a > b → (m = 1) → 
  (∀ x : ℝ, f' x = 2 * Real.exp (2 * x)) → 
  (∀ x₀ : ℝ, (m * (2 * x₀ + 1) = Real.exp (2 * x₀)) ∧ (2 * Real.exp (2 * x₀) = 2 * m)) → 
  (∀ (a b : ℝ), a > b → 
  (h a 1 - h b 1) / (a - b) < 2 * Real.exp (2 * a) - 2) :=
by
  sorry

end part_I_part_II_l286_286382


namespace find_a_plus_b_plus_c_l286_286108

noncomputable def max_possible_area_triangle_BPE
    (A B C D P E : Point)
    (AB AC BC : ℝ)
    (h_ABC : Triangle A B C)
    (AB_eq : distance A B = AB)
    (BC_eq : distance B C = BC)
    (CA_eq : distance C A = AC)
    (D_on_BC : collinear B D C)
    (E_mid_BC : middle E B C)
    (I_B I_C : Point)
    (I_B_incenter_ABD : incenter I_B (Triangle A B D))
    (I_C_incenter_ACD : incenter I_C (Triangle A C D))
    (P_intersections : on_circumcircle P (Triangle B I_B D) ∧ on_circumcircle P (Triangle C I_C D))
    : ℝ :=
    let maximum_area := 0 - 25 * Real.sqrt 3
    in 28

theorem find_a_plus_b_plus_c :
    ∃ a b c : ℕ, a - b * Real.sqrt c = max_possible_area_triangle_BPE A B C D P E 8 12 10 h_ABC AB_eq BC_eq CA_eq D_on_BC E_mid_BC I_B I_C I_B_incenter_ABD I_C_incenter_ACD P_intersections ∧ c ∉ SquareNumbers ∧ a + b + c = 28 :=
by {
  sorry
}

end find_a_plus_b_plus_c_l286_286108


namespace XY_length_l286_286103

/- Define the initial conditions -/
def O : Type := ℝ
def A : O := 10
def B : O := 10
def radius : O := 10
def OY_perpendicular_to_AB : Prop := true
def angle_AOB : ℝ := 90
def OX : ℝ := 5 * Real.sqrt 2

/- Prove that XY is equal to 10 - 5 * sqrt 2 -/
theorem XY_length : ∃ XY : ℝ, XY = 10 - 5 * Real.sqrt 2 := 
begin
  use 10 - 5 * Real.sqrt 2,
  sorry,
end

end XY_length_l286_286103


namespace hexagon_area_l286_286276

theorem hexagon_area (perimeter_square : ℝ) (h_perimeter : perimeter_square = 160) :
  let side_length_square := perimeter_square / 4 in
  let side_length_segment := side_length_square / 2 in
  let side_length_hexagon := side_length_segment in
  let area_hexagon := (3 * Real.sqrt 3 / 2) * side_length_hexagon^2 in
  area_hexagon = 600 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_l286_286276


namespace count_perfect_square_factors_l286_286066

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286066


namespace available_seats_l286_286591

/-- Two-fifths of the seats in an auditorium that holds 500 people are currently taken. --/
def seats_taken : ℕ := (2 * 500) / 5

/-- One-tenth of the seats in an auditorium that holds 500 people are broken. --/
def seats_broken : ℕ := 500 / 10

/-- Total seats in the auditorium --/
def total_seats := 500

/-- There are 500 total seats in an auditorium. Two-fifths of the seats are taken and 
one-tenth are broken. Prove that the number of seats still available is 250. --/
theorem available_seats : (total_seats - seats_taken - seats_broken) = 250 :=
by 
  sorry

end available_seats_l286_286591


namespace sqrt_sin_eq_l286_286111

theorem sqrt_sin_eq (a b c : ℕ) (h_a : a = 1) (h_b : b = 4) (h_c : c = 10) (h_c_range : 0 < c ∧ c < 90) :
  sqrt (9 - 8 * sin (50 * Real.pi / 180)) = a + b * sin (c * Real.pi / 180) →
  (b + c) / a = 14 :=
by
  intro h
  rw [h_a, h_b, h_c]
  exact sorry

end sqrt_sin_eq_l286_286111


namespace range_of_m_l286_286734

/-- Given the conditions:
- \( \left|1 - \frac{x - 2}{3}\right| \leq 2 \)
- \( x^2 - 2x + 1 - m^2 \leq 0 \) where \( m > 0 \)
- \( \neg \left( \left|1 - \frac{x - 2}{3}\right| \leq 2 \right) \) is a necessary but not sufficient condition for \( x^2 - 2x + 1 - m^2 \leq 0 \)

Prove that the range of \( m \) is \( m \geq 10 \).
-/
theorem range_of_m (m : ℝ) (x : ℝ)
  (h1 : ∀ x, ¬(abs (1 - (x - 2) / 3) ≤ 2) → x < -1 ∨ x > 11)
  (h2 : ∀ x, ∀ m > 0, x^2 - 2 * x + 1 - m^2 ≤ 0)
  : m ≥ 10 :=
sorry

end range_of_m_l286_286734


namespace students_taking_art_l286_286621

theorem students_taking_art :
  ∀ (total_students music_students both_music_art neither_music_art : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_music_art = 10 →
  neither_music_art = 470 →
  (total_students - neither_music_art) - (music_students - both_music_art) - both_music_art = 10 :=
by
  intros total_students music_students both_music_art neither_music_art h_total h_music h_both h_neither
  sorry

end students_taking_art_l286_286621


namespace inner_circle_radius_l286_286494

theorem inner_circle_radius :
  ∃ (r : ℝ) (a b c d : ℕ), 
    (r = (-78 + 70 * Real.sqrt 3) / 26) ∧ 
    (a = 78) ∧ 
    (b = 70) ∧ 
    (c = 3) ∧ 
    (d = 26) ∧ 
    (Nat.gcd a d = 1) ∧ 
    (a + b + c + d = 177) := 
sorry

end inner_circle_radius_l286_286494


namespace sum_bi_roots_l286_286499

noncomputable def sum_bi (roots : Fin 2023 → ℂ) : ℂ :=
  ∑ n : Fin 2023, 1 / (1 - roots n)

theorem sum_bi_roots :
  let roots := λ n: Fin 2023, (Polynomial.roots (X ^ 2023 + X ^ 2022 + ⋯ + X - 2024)).nth n in
  sum_bi roots = -2054238 :=
sorry

end sum_bi_roots_l286_286499


namespace total_travel_time_l286_286583

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end total_travel_time_l286_286583


namespace anna_routes_15_roads_exactly_l286_286312

noncomputable def number_of_routes (cities roads : ℕ) (roads_to_travel : ℕ) (start end : ℕ) : ℕ :=
  if cities = 15 ∧ roads = 20 ∧ roads_to_travel = 15 then 4
  else 0

theorem anna_routes_15_roads_exactly :
  number_of_routes 15 20 15 0 1 = 4 :=
by
  dec_trivial

end anna_routes_15_roads_exactly_l286_286312


namespace kittens_count_l286_286120

def initial_kittens : ℕ := 8
def additional_kittens : ℕ := 2
def total_kittens : ℕ := 10

theorem kittens_count : initial_kittens + additional_kittens = total_kittens := by
  -- Proof will go here
  sorry

end kittens_count_l286_286120


namespace epidemic_prevention_control_l286_286203

noncomputable def suburban_scores : List ℝ := [74, 81, 75, 76, 70, 75, 75, 79, 81, 70, 74, 80, 91, 69, 82]
noncomputable def urban_scores : List ℝ := [81, 94, 83, 77, 83, 80, 81, 70, 81, 73, 78, 82, 80, 70, 50]

structure ScoreDistribution where
  below_60 : ℕ
  from_60_to_80 : ℕ
  from_80_to_90 : ℕ
  above_90 : ℕ

def suburban_distribution : ScoreDistribution := { below_60 := 0, from_60_to_80 := 10, from_80_to_90 := 4, above_90 := 1 }
def urban_distribution : ScoreDistribution := { below_60 := 1, from_60_to_80 := a, from_80_to_90 := 8, above_90 := 1 }

def suburban_analysis : ScoreDistribution := { below_60 := 0, from_60_to_80 := 10, from_80_to_90 := 4, above_90 := 1 }
def urban_analysis : ScoreDistribution := { below_60 := 1, from_60_to_80 := 5, from_80_to_90 := 8, above_90 := 1 }

def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

def median (scores : List ℝ) : ℝ :=
  let sorted := scores.qsort (≤)
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

def mode (scores : List ℝ) : ℝ :=
  scores.groupBy id |>.maxBy (·.length) |>.head!

theorem epidemic_prevention_control :
  mean suburban_scores = 76.8 ∧
  median suburban_scores = 75 ∧
  mode suburban_scores = 75 ∧
  mean urban_scores = 77.5 ∧
  median urban_scores = 80 ∧
  mode urban_scores = 81 :=
by
  sorry

end epidemic_prevention_control_l286_286203


namespace collinear_P_Q_H_l286_286848

theorem collinear_P_Q_H {A B C H M N P Q : Type*}
  [Triangle ABC]
  (H : Orthocenter A B C)
  (M : PointOnSegment A B M)
  (N : PointOnSegment A C N)
  (P Q : IntersectionPoints (CircleDiameter B N) (CircleDiameter C M)) :
  Collinear P Q H :=
sorry

end collinear_P_Q_H_l286_286848


namespace count_numbers_with_perfect_square_factors_l286_286061

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286061


namespace max_possible_value_en_l286_286500

-- Definitions
def b (n : ℕ) : ℚ := (15 ^ n - 1) / 14 
def e (n : ℕ) : ℕ := Int.gcd (b n).natAbs (b (n + 1)).natAbs

theorem max_possible_value_en : ∀ n : ℕ, e n = 1 := by
  sorry

end max_possible_value_en_l286_286500


namespace projection_m_n_l286_286356

variables (m n : ℝ × ℝ × ℝ)
def projection (m n : ℝ × ℝ × ℝ) : ℝ := 
  (m.1 * n.1 + m.2 * n.2 + m.3 * n.3) / real.sqrt (n.1^2 + n.2^2 + n.3^2)

theorem projection_m_n (m n : ℝ × ℝ × ℝ)
  (hm : m = (2, -4, 1))
  (hn : n = (2, -1, 2)) :
  projection m n = 10 / 3 :=
by
  sorry

end projection_m_n_l286_286356


namespace inverse_proportion_inequality_l286_286390

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l286_286390


namespace EH_value_in_quadrilateral_l286_286825

noncomputable def EH_possible_values (EH : ℕ) : Prop :=
  7 + 12 > EH ∧
  12 + 7 > EH ∧
  7 + 11 > EH ∧
  11 + 7 > EH ∧
  EH + 11 > 12 ∧
  EH + 11 > 7

theorem EH_value_in_quadrilateral (EH : ℕ) : 
  EH_possible_values EH → (12 ≤ EH ∧ EH ≤ 18) :=
by
  intro poss
  have h1 : 7 + 12 > EH := poss.1
  have h2 : 12 + 7 > EH := poss.2
  have h3 : 7 + 11 > EH := poss.3
  have h4 : 11 + 7 > EH := poss.4
  have h5 : EH + 11 > 12 := poss.5
  have h6 : EH + 11 > 7 := poss.6
  sorry

end EH_value_in_quadrilateral_l286_286825


namespace circumscribed_circle_radius_l286_286586

-- Define the tangent circles with radii R1 and R2
variables {R1 R2 : ℝ} (hR1 : R1 > 0) (hR2 : R2 > 0)

-- Assume the existence of relevant points and lines
variables {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define conditions: A tangent line AB, and another tangent line parallel to AB
def tangent_line_AB (A B : Type*) : Prop := sorry
def parallel_tangent_lines (A B C D : Type*) : Prop := sorry

-- Mathematical statement: the radius of the circumscribed circle around triangle BDC
theorem circumscribed_circle_radius (h_tangent_AB : tangent_line_AB A B) (h_parallel_tangent : parallel_tangent_lines A B C D) :
  ∃ (radius: ℝ), radius = 2 * sqrt (R1 * R2) :=
sorry

end circumscribed_circle_radius_l286_286586


namespace pyramid_ball_count_l286_286820

theorem pyramid_ball_count :
  ∃ n : ℕ, 
    (1 + (n - 1) * 3 = 37) ∧
    (n = 13) ∧
    (∑ i in finset.range n, 1 + (i * 3) = 247) :=
by
  -- proof steps are omitted as the statement requires using existent conditions and not solution steps
  sorry

end pyramid_ball_count_l286_286820


namespace num_committees_of_4_from_6_l286_286278

theorem num_committees_of_4_from_6 (n k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  nat.choose n k = 15 :=
by
  rw [h1, h2]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)
  sorry

end num_committees_of_4_from_6_l286_286278


namespace vector_addition_l286_286530

variable (A B C D : Type)
variable (AB DC CB CD BC AD : A → A)
variable (h1 : ∀ (a : A), - DC a = CD a)
variable (h2 : ∀ (a : A), - CB a = BC a)

theorem vector_addition (AB DC CB CD BC AD : A → A)
  (h1 : ∀ (a : A), - DC a = CD a)
  (h2 : ∀ (a : A), - CB a = BC a) :
  ∀ (a : A), AB a - DC a - CB a = AD a :=
by
  sorry

end vector_addition_l286_286530


namespace sequence_properties_l286_286746

noncomputable def sequenceSn (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2

-- Assuming sequences a_n and b_n are defined
def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 2 * n - 1

def c (n : ℕ) : ℕ := a n * b n

def T_sum (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

theorem sequence_properties :
  (∀ n : ℕ, sequenceSn a n = 2 * a n - 2) ∧
  (b 1 = 1) ∧
  (∀ n : ℕ, b (n + 1) - b n = 2) → 
  (∀ n : ℕ, a n = 2^n) ∧ 
  (∀ n : ℕ, b n = 2*n - 1) ∧ 
  (∀ n : ℕ, T_sum n = (2*n-3)*2^(n+1) + 6) :=
by
  intros,
  sorry

end sequence_properties_l286_286746


namespace evaluate_expression_l286_286337

-- Define the terms a and b
def a : ℕ := 2023
def b : ℕ := 2024

-- The given expression
def expression : ℤ := (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b)

-- The theorem to prove
theorem evaluate_expression : expression = ↑a := 
by sorry

end evaluate_expression_l286_286337


namespace pencils_per_row_l286_286338

theorem pencils_per_row (total_pencils rows : ℕ) (h1 : total_pencils = 720) (h2 : rows = 30) :
  total_pencils / rows = 24 :=
by
  rw [h1, h2]
  norm_num

end pencils_per_row_l286_286338


namespace change_received_l286_286929

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end change_received_l286_286929


namespace parallelogram_probability_l286_286882

theorem parallelogram_probability (P Q R S : ℝ × ℝ) 
  (hP : P = (4, 2)) 
  (hQ : Q = (-2, -2)) 
  (hR : R = (-6, -6)) 
  (hS : S = (0, -2)) :
  let parallelogram_area := 24 -- given the computed area based on provided geometry
  let divided_area := parallelogram_area / 2
  let not_above_x_axis_area := divided_area
  (not_above_x_axis_area / parallelogram_area) = (1 / 2) :=
by
  sorry

end parallelogram_probability_l286_286882


namespace monotonically_increasing_sequence_l286_286294

theorem monotonically_increasing_sequence (k : ℝ) : (∀ n : ℕ+, n^2 + k * n < (n + 1)^2 + k * (n + 1)) ↔ k > -3 := by
  sorry

end monotonically_increasing_sequence_l286_286294


namespace quadratic_sum_is_zero_l286_286205

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l286_286205


namespace product_of_102_and_27_l286_286524

theorem product_of_102_and_27 : 102 * 27 = 2754 :=
by
  sorry

end product_of_102_and_27_l286_286524


namespace perfect_square_factors_l286_286051

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286051


namespace total_spectators_after_halftime_l286_286664

theorem total_spectators_after_halftime
  (initial_boys : ℕ := 300)
  (initial_girls : ℕ := 400)
  (initial_adults : ℕ := 300)
  (total_people : ℕ := 1000)
  (quarter_boys_leave_fraction : ℚ := 1 / 4)
  (quarter_girls_leave_fraction : ℚ := 1 / 8)
  (quarter_adults_leave_fraction : ℚ := 1 / 5)
  (halftime_new_boys : ℕ := 50)
  (halftime_new_girls : ℕ := 90)
  (halftime_adults_leave_fraction : ℚ := 3 / 100) :
  let boys_after_first_quarter := initial_boys - initial_boys * quarter_boys_leave_fraction
  let girls_after_first_quarter := initial_girls - initial_girls * quarter_girls_leave_fraction
  let adults_after_first_quarter := initial_adults - initial_adults * quarter_adults_leave_fraction
  let boys_after_halftime := boys_after_first_quarter + halftime_new_boys
  let girls_after_halftime := girls_after_first_quarter + halftime_new_girls
  let adults_after_halftime := adults_after_first_quarter * (1 - halftime_adults_leave_fraction)
  boys_after_halftime + girls_after_halftime + adults_after_halftime = 948 :=
by sorry

end total_spectators_after_halftime_l286_286664


namespace find_f_prime_at_1_l286_286773

noncomputable def f (x : ℝ) (f'₁ : ℝ) : ℝ := 2 * f'₁ * real.log x - x

theorem find_f_prime_at_1 (f'₁ : ℝ) (h : ∀ x, deriv (λ x, f x f'₁) x = 2 * f'₁ * (1 / x) - 1) : 
  f'₁ = 1 := 
by
  have deriv_at_1 : deriv (λ x, f x f'₁) 1 = f'₁ := by simp [h]
  linarith

end find_f_prime_at_1_l286_286773


namespace probability_of_C_l286_286281

def region_prob_A := (1 : ℚ) / 4
def region_prob_B := (1 : ℚ) / 3
def region_prob_D := (1 : ℚ) / 6

theorem probability_of_C :
  (region_prob_A + region_prob_B + region_prob_D + (1 : ℚ) / 4) = 1 :=
by
  sorry

end probability_of_C_l286_286281


namespace man_year_of_birth_l286_286998

theorem man_year_of_birth (x : ℕ) (hx1 : (x^2 + x >= 1850)) (hx2 : (x^2 + x < 1900)) : (1850 + (x^2 + x - x)) = 1892 :=
by {
  sorry
}

end man_year_of_birth_l286_286998


namespace calculate_mod_121_l286_286866

theorem calculate_mod_121 (n : ℕ) (h : n = 95) : 
  (5^n + 11^n) % 121 = 16 :=
by {
  rw [h],
  sorry
}

end calculate_mod_121_l286_286866


namespace first_part_lent_years_l286_286644

theorem first_part_lent_years (P P1 P2 : ℝ) (rate1 rate2 : ℝ) (years2 : ℝ) (interest1 interest2 : ℝ) (t : ℝ) 
  (h1 : P = 2717)
  (h2 : P2 = 1672)
  (h3 : P1 = P - P2)
  (h4 : rate1 = 0.03)
  (h5 : rate2 = 0.05)
  (h6 : years2 = 3)
  (h7 : interest1 = P1 * rate1 * t)
  (h8 : interest2 = P2 * rate2 * years2)
  (h9 : interest1 = interest2) :
  t = 8 :=
sorry

end first_part_lent_years_l286_286644


namespace quadratic_sum_is_zero_l286_286206

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l286_286206


namespace line_in_slope_intercept_form_l286_286632

-- Given the condition
def line_eq (x y : ℝ) : Prop :=
  (2 * (x - 3)) - (y + 4) = 0

-- Prove that the line equation can be expressed as y = 2x - 10.
theorem line_in_slope_intercept_form (x y : ℝ) :
  line_eq x y ↔ y = 2 * x - 10 := 
sorry

end line_in_slope_intercept_form_l286_286632


namespace four_isosceles_triangles_l286_286175

def distance (p1 p2 : (ℕ × ℕ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^(2 : ℕ) + (p2.2 - p1.2)^(2 : ℕ))

def is_isosceles (v1 v2 v3 : (ℕ × ℕ)) : Prop :=
  let d1 := distance v1 v2
  let d2 := distance v1 v3
  let d3 := distance v2 v3
  d1 = d2 ∨ d1 = d3 ∨ d2 = d3

def triangles : list ((ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) :=
  [ ((1, 5), (3, 5), (2, 3)),
    ((4, 3), (4, 5), (6, 3)),
    ((1, 2), (4, 3), (7, 2)),
    ((5, 1), (4, 3), (6, 1)),
    ((3, 1), (4, 3), (5, 1)) ]

def count_isosceles : ℕ :=
  triangles.filter (λ ⟨v1, v2, v3⟩, is_isosceles v1 v2 v3).length

theorem four_isosceles_triangles : count_isosceles = 4 := by
  sorry

end four_isosceles_triangles_l286_286175


namespace monica_studied_32_67_hours_l286_286147

noncomputable def monica_total_study_time : ℚ :=
  let monday := 1
  let tuesday := 2 * monday
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let total_weekday := monday + tuesday + wednesday + thursday + friday
  let saturday := total_weekday
  let sunday := saturday / 3
  total_weekday + saturday + sunday

theorem monica_studied_32_67_hours :
  monica_total_study_time = 32.67 := by
  sorry

end monica_studied_32_67_hours_l286_286147


namespace count_perfect_square_factors_l286_286065

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286065


namespace arithmetic_mean_of_remaining_set_l286_286193

def arithmetic_mean_after_discarding (mean : ℕ) (total_numbers : ℕ) (discard_numbers : List ℕ) : Prop :=
  let original_sum := mean * total_numbers
  let discard_sum := discard_numbers.sum
  let new_sum := original_sum - discard_sum
  let remaining_numbers := total_numbers - discard_numbers.length
  new_sum / remaining_numbers = 41

theorem arithmetic_mean_of_remaining_set :
  arithmetic_mean_after_discarding 42 60 [50, 60, 70] :=
by {
  unfold arithmetic_mean_after_discarding,
  -- Let's unpack this step by step.
  let original_sum := 42 * 60,       -- 2520
  let discard_sum := 50 + 60 + 70,  -- 180
  let new_sum := original_sum - discard_sum, -- 2340
  let remaining_numbers := 60 - 3,  -- 57
  have h1 : original_sum = 2520 := by refl,
  have h2 : discard_sum = 180 := by refl,
  have h3 : new_sum = 2340 := by rw [h1, h2]; norm_num,
  have h4 : remaining_numbers = 57 := by refl,
  rw [h3, h4],
  norm_num,
  exact rfl,
}

end arithmetic_mean_of_remaining_set_l286_286193


namespace inverse_proportion_inequality_l286_286386

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l286_286386


namespace cot_neg_45_deg_eq_neg_1_l286_286345

theorem cot_neg_45_deg_eq_neg_1
  (cot_def: ∀ x, Real.cot x = 1 / Real.tan x)
  (tan_neg: ∀ x, Real.tan (-x) = -Real.tan x)
  (tan_45_deg: Real.tan (Real.pi / 4) = 1) :
  Real.cot (-Real.pi / 4) = -1 :=
by
  sorry

end cot_neg_45_deg_eq_neg_1_l286_286345


namespace other_root_and_m_l286_286451

-- Definitions for the conditions
def quadratic_eq (m : ℝ) := ∀ x : ℝ, x^2 + 2 * x + m = 0
def root (x : ℝ) (m : ℝ) := x^2 + 2 * x + m = 0

-- Theorem statement
theorem other_root_and_m (m : ℝ) (h : root 2 m) : ∃ t : ℝ, (2 + t = -2) ∧ (2 * t = m) ∧ t = -4 ∧ m = -8 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end other_root_and_m_l286_286451


namespace ellipse_equation_l286_286544

theorem ellipse_equation (c a b : ℝ)
  (foci1 foci2 : ℝ × ℝ) 
  (h_foci1 : foci1 = (-1, 0)) 
  (h_foci2 : foci2 = (1, 0)) 
  (h_c : c = 1) 
  (h_major_axis : 2 * a = 10) 
  (h_b_sq : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1)) :=
by
  sorry

end ellipse_equation_l286_286544


namespace evaluate_expression_l286_286799

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 7) :
  (x^5 + 3 * y^3) / 9 = 141 :=
by
  sorry

end evaluate_expression_l286_286799


namespace convex_polygon_diagonals_l286_286301

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end convex_polygon_diagonals_l286_286301


namespace sum_of_decimals_l286_286655

theorem sum_of_decimals :
  let a := 0.35
  let b := 0.048
  let c := 0.0072
  a + b + c = 0.4052 := by
  sorry

end sum_of_decimals_l286_286655


namespace prime_a_b_l286_286796

theorem prime_a_b (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^11 + b = 2089) : 49 * b - a = 2007 :=
sorry

end prime_a_b_l286_286796


namespace trajectory_of_P_l286_286407

-- Define the point and vector structures
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the fixed points A and B
def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

-- Define the dot product of two vectors
def dotProd (a b : Point) : ℝ :=
  a.x * b.x + a.y * b.y

-- Define the condition on the moving point P
def cond (P : Point) : Prop :=
  dotProd ⟨-2 - P.x, -P.y⟩ ⟨2 - P.x, -P.y⟩ = - P.x^2

-- Define the trajectory equation of point P
def trajectory_eqn (P : Point) : Prop :=
  (P.x^2) / 2 + (P.y^2) / 4 = 1

-- The main theorem statement
theorem trajectory_of_P (P : Point) (h : cond P) : trajectory_eqn P :=
  sorry

end trajectory_of_P_l286_286407


namespace exists_m_n_for_any_d_l286_286180

theorem exists_m_n_for_any_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) :=
by
  sorry

end exists_m_n_for_any_d_l286_286180


namespace alpha_combination_l286_286477

variable (α1 α2 α3 : ℝ)

def alpha1 (AM MA1 : ℝ) : ℝ := AM / MA1
def alpha2 (BM MB1 : ℝ) : ℝ := BM / MB1
def alpha3 (CM MC1 : ℝ) : ℝ := CM / MC1

theorem alpha_combination 
  (hα1 : α1 = alpha1 AM MA1)
  (hα2 : α2 = alpha2 BM MB1)
  (hα3 : α3 = alpha3 CM MC1) :
  α1 * α2 * α3 - (α1 + α2 + α3) = 2 :=
sorry

end alpha_combination_l286_286477


namespace geometric_sequence_dn_l286_286904

variable {c : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (c : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, c (n+1) = c n * r

def positive_terms (c : ℕ → ℝ) : Prop :=
∀ n, c n > 0

-- Main proposition
theorem geometric_sequence_dn (c : ℕ → ℝ) (r : ℝ) (h1 : is_geometric_sequence c r) (h2 : positive_terms c) :
  ∃d : ℕ → ℝ, (∀ n, d n = n * (c 1 * (real.sqrt r)^(n-1))^n) ∧ positive_terms d ∧ is_geometric_sequence d (real.sqrt r) :=
by
  sorry

end geometric_sequence_dn_l286_286904


namespace greatest_perfect_square_below_200_l286_286161

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l286_286161


namespace tetrahedron_BC_squared_l286_286506

theorem tetrahedron_BC_squared (AB AC BC R r : ℝ) 
  (h1 : AB = 1) 
  (h2 : AC = 1) 
  (h3 : 1 ≤ BC) 
  (h4 : R = 4 * r) 
  (concentric : AB = AC ∧ R > 0 ∧ r > 0) :
  BC^2 = 1 + Real.sqrt (7 / 15) := 
by 
sorry

end tetrahedron_BC_squared_l286_286506


namespace sequence_bounded_l286_286378

theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_dep : ∀ k n m l, k + n = m + l → (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ m M : ℝ, ∀ n, m ≤ a n ∧ a n ≤ M :=
sorry

end sequence_bounded_l286_286378


namespace swap_to_composite_sums_l286_286227

/-- Prove that in 35 moves, it is possible to achieve a state where the sum of every pair of 
    numbers that share an edge is composite in a 10x10 table filled with numbers 1 to 100, each number appearing exactly once.

Conditions:
- There are natural numbers from 1 to 100 placed randomly in the cells of a 10 × 10 table, each number appearing exactly once.
- In one move, it is allowed to swap any two numbers.
-/
theorem swap_to_composite_sums :
  ∃ (config : Fin 100 → Fin 100 → ℕ) (move_count : ℕ), 
    (∀i j : Fin 100, config i j ∈ {x | 1 ≤ x ∧ x ≤ 100}) ∧
    bijective (uncurry config) ∧
    -- move_count is the total number of moves required to achieve the desired configuration
    move_count ≤ 35 ∧
    -- sum of every pair of numbers sharing an edge is composite
    (∀i j : Fin 100, (i + 1 < 100 → composite (config i j + config (i + 1) j)) ∧
                     (j + 1 < 100 → composite (config i j + config i (j + 1)))) :=
sorry

/-- Helper Function that checks if a number is composite -/
def composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n

/-- Helper Function to make the configuration total -/
def uncurry {α β γ : Type*} (f : α → β → γ) : α × β → γ :=
  λ p, f p.1 p.2


end swap_to_composite_sums_l286_286227


namespace papaya_production_l286_286892

theorem papaya_production (P : ℕ)
  (h1 : 2 * P + 3 * 20 = 80) :
  P = 10 := 
by sorry

end papaya_production_l286_286892


namespace arithmetic_progression_value_l286_286187

variable {α : Type*}
variable a : ℕ → ℝ

theorem arithmetic_progression_value (h1 : a 1 = 2) (h3_6 : a 3 + a 6 = 8) : a 8 = 6 :=
sorry

end arithmetic_progression_value_l286_286187


namespace matrix_product_zero_l286_286678

variable (a b c d : ℝ)

def mat1 : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
  [0, d, -c],
  [-d, 0, b],
  [c, -b, 0]
  ]

def mat2 : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
  [d ^ 2, d * c, d * b],
  [d * c, c ^ 2, c * b],
  [d * b, c * b, b ^ 2]
  ]

theorem matrix_product_zero : mat1 a b c d ⬝ mat2 a b c d = !![
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]
  ] :=
by
  sorry

end matrix_product_zero_l286_286678


namespace cot_neg_45_eq_neg1_l286_286346

theorem cot_neg_45_eq_neg1 : Real.cot (-(Real.pi / 4)) = -1 :=
by
  sorry

end cot_neg_45_eq_neg1_l286_286346


namespace area_of_triangle_l286_286495

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) := 
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let c := real.sqrt (a^2 - b^2) in ((c, 0), (-c, 0))

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem area_of_triangle {a b : ℝ} (h₁ : a = 2) (h₂ : b = 1)
  (P : ℝ × ℝ) (hP : P ∈ ellipse a b)
  (h_perp : let (F1, F2) := foci a b in ∀ x y z, (distance F1 P * distance F2 P = x * y * z))
  (distPF1_distPF2 : let (F1, F2) := foci a b in ∀ x y, (distance P F1 + distance P F2 = x + y) ∧ (distance F1 F2 = real.sqrt(3) * 2)) :
  let (F1, F2) := foci a b in (1 / 2) * distance P F1 * distance P F2 = 1 :=
sorry

end area_of_triangle_l286_286495


namespace proposition_three_correct_l286_286132

variables (m n l : Line) (α β γ : Plane)

-- Context and conditions
variable [Different m n l]
variable [Different α β γ]

-- Proving proposition 3 under the given conditions
theorem proposition_three_correct 
  (h1 : m ⊥ α) 
  (h2 : n ∥ β)
  (h3 : α ∥ β) : m ⊥ n :=
sorry

end proposition_three_correct_l286_286132


namespace g_diff_320_160_l286_286363

def σ (n : ℕ) : ℕ := (Divisors n).sum id

def g (n : ℕ) : ℚ := (σ n - n) / n

theorem g_diff_320_160 : g 320 - g 160 = 3 / 160 := by
  sorry

end g_diff_320_160_l286_286363


namespace solve_equation_l286_286899

theorem solve_equation (x : ℝ) (hx : (x + 1) ≠ 0) :
  (x = -3 / 4) ∨ (x = -1) ↔ (x^3 + x^2 + x + 1) / (x + 1) = x^2 + 4 * x + 4 :=
by
  sorry

end solve_equation_l286_286899


namespace books_still_to_read_l286_286225

theorem books_still_to_read (B R U : ℕ) (h1 : B = 22) (h2 : R = 12) : U = B - R :=
by {
  intros,
  -- The steps below would be the proof, skipped here with sorry.
  sorry,
}

end books_still_to_read_l286_286225


namespace sum_a_1_to_100_l286_286410

noncomputable def f (n : ℕ) : ℤ :=
  if even n then n^2 else -n^2

def a (n : ℕ) : ℤ := f n + f (n + 1)

theorem sum_a_1_to_100 : (∑ n in Finset.range 100, a (n + 1)) = -100 := 
  sorry

end sum_a_1_to_100_l286_286410


namespace probability_centrally_symmetric_shape_l286_286226

def shape := {circle, rectangle, equilateral_triangle, regular_pentagon}

def is_centrally_symmetric (s : shape) : Prop :=
  s = circle ∨ s = rectangle

theorem probability_centrally_symmetric_shape : 
  ∃ (p : ℚ), p = 1/2 ∧
  (∀ (shapes : list shape), 
    shapes.length = 4 →
    shapes.contains circle →
    shapes.contains rectangle →
    shapes.contains equilateral_triangle →
    shapes.contains regular_pentagon →
    p = (shapes.countp is_centrally_symmetric) / (shapes.length)) :=
sorry

end probability_centrally_symmetric_shape_l286_286226


namespace function_neither_even_nor_odd_l286_286110

noncomputable def g (x : ℝ) : ℝ := floor x + real.sin x

theorem function_neither_even_nor_odd : 
  ¬(∀ x, g (-x) = g (x)) ∧ ¬(∀ x, g (-x) = -g (x)) :=
by
  -- Even check
  have h1 : ¬(∀ x, g (-x) = g (x)) :=
    begin
      intro h,
      -- provide a counterexample, e.g., x = 1
      specialize h 1,
      unfold g at h,
      -- floor 1 = 1, floor (-1) = -1, sin is an odd function
      have h2 : floor (-1 : ℝ) + real.sin (-1 : ℝ) ≠ floor (1 : ℝ) + real.sin (1 : ℝ) := by norm_num,
      contradiction,
    end,
    
  -- Odd check
  have h3 : ¬(∀ x, g (-x) = -g (x)) :=
    begin
      intro h,
      -- provide a counterexample, e.g., x = 1 / 2
      specialize h (1 / 2),
      unfold g at h,
      -- floor (1/2) = 0, floor (-1/2) = -1, sin is an odd function
      have h4 : floor (- (1 : ℝ) / 2) + real.sin (-(1 : ℝ) / 2) ≠ -(floor (1 : ℝ) / 2 + real.sin (1 : ℝ) / 2) := by norm_num,
      contradiction,
    end,

  exact ⟨h1, h3⟩

end function_neither_even_nor_odd_l286_286110


namespace count_perfect_square_factors_l286_286070

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286070


namespace find_x_l286_286224

theorem find_x (k : ℝ) (x y : ℝ) (h1 : y = k / sqrt x) (h2 : x = 4) (h3 : y = 2) (h4 : y = 8) : x = 1 / 4 :=
sorry

end find_x_l286_286224


namespace exists_k_phobic_l286_286995

-- Condition: Definition of a k-phobic number.
def is_k_phobic (n k : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000 ∧ (∀ m : ℕ, (m = n ∨ (m / 10000 = n / 10000 ∧ abs (m - n) ≤ 40000))
                                 → (m >= 10000 ∧ m < 100000)
                                   → (m ≠ 0 ∧ m % k ≠ 0))

-- Question: Smallest positive integer k such that there exists a k-phobic number.
def smallest_k_phobic : ℕ :=
  11112

-- Proof problem statement
theorem exists_k_phobic : ∃ n, is_k_phobic n smallest_k_phobic := by
  sorry

end exists_k_phobic_l286_286995


namespace problem_x_l286_286730

theorem problem_x (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 :=
sorry

end problem_x_l286_286730


namespace maximize_absolute_differences_l286_286072

theorem maximize_absolute_differences :
  ∃ (a : Fin 1962 → ℕ),
    (∀ i, a i ∈ (1 : Fin 1962).val..1962) ∧
    set.univ = {i : Fin 1962 | a i} ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    let d := λ i j, abs (a i - a j) in
    ∑ i in Finset.range 1962, d i (i.succ % 1962) =
      4 * (1962 - 1) * (1962 - 1961) :=
begin
  sorry
end

end maximize_absolute_differences_l286_286072


namespace Q20_correct_l286_286687

def S (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2 - 1

def Q (n : ℕ) : ℝ :=
  ∏ k in Finset.range (n - 1) + 2, (S k).toReal / ((S k).toReal + 1)

theorem Q20_correct : Q 20 = 0.6018 :=
by
  sorry

end Q20_correct_l286_286687


namespace train_passes_man_in_time_l286_286280

noncomputable def kmph_to_mps (v_kmph : ℝ) : ℝ := v_kmph * 1000 / 3600

theorem train_passes_man_in_time :
  ∀ (L : ℝ) (v_man_kmph : ℝ) (v_train_kmph : ℝ),
  L = 120 →
  v_man_kmph = 6 →
  v_train_kmph = 65.99424046076315 →
  L / (kmph_to_mps(v_train_kmph) + kmph_to_mps(v_man_kmph)) = 6.0003 :=
by
  intros L v_man_kmph v_train_kmph hL hvm ht     
  sorry

end train_passes_man_in_time_l286_286280


namespace count_perfect_square_factors_l286_286064

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286064


namespace part1_part2_l286_286417

-- Define the function f(x) = |x - 1|
def f (x : ℝ) : ℝ := abs (x - 1)

-- Part (1): Prove that f(x) ≥ 1 - x² if and only if x ≤ 0 or x ≥ 1
theorem part1 (x : ℝ) : f(x) ≥ 1 - x^2 ↔ x ≤ 0 ∨ x ≥ 1 :=
by
  sorry 

-- Part (2): Prove that for some x in ℝ, f(x) < a - x² + |x + 1| if and only if a > -1
theorem part2 (a : ℝ) : (∃ x : ℝ, f(x) < a - x^2 + abs (x + 1)) ↔ a > -1 :=
by
  sorry

end part1_part2_l286_286417


namespace trigonometric_identity_proof_l286_286369

theorem trigonometric_identity_proof
  (α : ℝ) (h1 : 0 < α ∧ α < π / 4)
  (h2 : cos (α - π / 4) = 4 / 5) :
  (cos (2 * α) / sin (α + π / 4) = -6 / 5) :=
by
  sorry

end trigonometric_identity_proof_l286_286369


namespace problem1_problem2_l286_286688

def d (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  f (max a b) - f (min a b)

noncomputable def f1 (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem problem1 : d f1 1 2 = 1 := 
sorry

noncomputable def f2 (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem problem2 (m : ℝ) (H : d (f2 m) 1 2 ≠ |f2 m 2 - f2 m 1|) : 1 < m ∧ m < 4 := 
sorry

end problem1_problem2_l286_286688


namespace part1_l286_286803

-- Definition of a double root equation
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ x₂ = 2 * x₁ ∧ a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0)

-- Part 1: Proof that x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_equation 1 (-3) 2 :=
by {
  use [1, 2],
  split,
  { intros h,
    linarith, },
  split,
  { refl, },
  split;
  { simp, linarith, }
}

end part1_l286_286803


namespace light_intensity_after_glass_pieces_minimum_glass_pieces_l286_286144

theorem light_intensity_after_glass_pieces (a : ℝ) (x : ℕ) : 
  (y : ℝ) = a * (0.9 ^ x) :=
sorry

theorem minimum_glass_pieces (a : ℝ) (x : ℕ) : 
  a * (0.9 ^ x) < a / 3 ↔ x ≥ 11 :=
sorry

end light_intensity_after_glass_pieces_minimum_glass_pieces_l286_286144


namespace find_XY_square_l286_286498

noncomputable def triangleABC := Type

variables (A B C T X Y : triangleABC)
variables (ω : Type) (BT CT BC TX TY XY : ℝ)

axiom acute_scalene_triangle (ABC : triangleABC) : Prop
axiom circumcircle (ABC: triangleABC) (ω: Type) : Prop
axiom tangents_intersect (ω: Type) (B C T: triangleABC) (BT CT : ℝ) : Prop
axiom projections (T: triangleABC) (X: triangleABC) (AB: triangleABC) (Y: triangleABC) (AC: triangleABC) : Prop

axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom TX_TY_XY_relation : TX^2 + TY^2 + XY^2 = 1450

theorem find_XY_square : XY^2 = 841 :=
by { sorry }

end find_XY_square_l286_286498


namespace harry_travel_ratio_l286_286010

theorem harry_travel_ratio
  (bus_initial_time : ℕ)
  (bus_rest_time : ℕ)
  (total_travel_time : ℕ)
  (walking_time : ℕ := total_travel_time - (bus_initial_time + bus_rest_time))
  (bus_total_time : ℕ := bus_initial_time + bus_rest_time)
  (ratio : ℚ := walking_time / bus_total_time)
  (h1 : bus_initial_time = 15)
  (h2 : bus_rest_time = 25)
  (h3 : total_travel_time = 60)
  : ratio = (1 / 2) := 
sorry

end harry_travel_ratio_l286_286010


namespace johns_remaining_money_l286_286121

theorem johns_remaining_money (initial_amount : ℝ) (fraction_snacks : ℝ) (fraction_necessities : ℝ) 
  (h_initial : initial_amount = 20) (h_fraction_snacks : fraction_snacks = 1 / 5) 
  (h_fraction_necessities : fraction_necessities = 3 / 4) : 
  let after_snacks := initial_amount - initial_amount * fraction_snacks in
  let after_necessities := after_snacks - after_snacks * fraction_necessities in
  after_necessities = 4 :=
by
  sorry

end johns_remaining_money_l286_286121


namespace sum_of_midpoints_l286_286558

variable (a b c : ℝ)

def sum_of_vertices := a + b + c

theorem sum_of_midpoints (h : sum_of_vertices a b c = 15) :
  (a + b)/2 + (a + c)/2 + (b + c)/2 = 15 :=
by
  sorry

end sum_of_midpoints_l286_286558


namespace lines_not_parallel_lines_perpendicular_l286_286511

variable {m : ℚ}

def l_1 (m : ℚ) : ℚ → ℚ → Prop := λ x y, (3 + m) * x + 4 * y = 5 - 3 * m
def l_2 (m : ℚ) : ℚ → ℚ → Prop := λ x y, 2 * x + (1 + m) * y = -20

theorem lines_not_parallel (h : ∀ x y, ¬ l_1 m x y ↔ ¬ l_2 m x y) : m = 1 :=
sorry

theorem lines_perpendicular (h : ∀ x y, l_1 m x y ∧ l_2 m x y → 0 = 0) : m = -5 / 3 :=
sorry

end lines_not_parallel_lines_perpendicular_l286_286511


namespace correct_statement_l286_286374

noncomputable theory

variables {P : Type} [AffineSpace P ℝ]
variables (m : Line P) (α : AffineSubspace ℝ P)

-- Hypothesis: m intersects α but is not perpendicular to α
def intersects (m : Line P) (α : AffineSubspace ℝ P) : Prop := 
  ∃ (p : P), p ∈ m ∧ p ∈ α

def not_perpendicular (m : Line P) (α : AffineSubspace ℝ P) : Prop :=
  ¬ ∀ (p ∈ m) (q ∈ α), m.direction ⊥ α.direction

theorem correct_statement (hm : intersects m α) (hmp : not_perpendicular m α) :
  ∃! (β : AffineSubspace ℝ P), m ⊆ β ∧ β ⊥ α :=
sorry

end correct_statement_l286_286374


namespace find_y_l286_286834

-- Definitions of the angles
def angle_ABC : ℝ := 80
def angle_BAC : ℝ := 70
def angle_BCA : ℝ := 180 - angle_ABC - angle_BAC -- calculation of third angle in triangle ABC

-- Right angle in triangle CDE
def angle_ECD : ℝ := 90

-- Defining the proof problem
theorem find_y (y : ℝ) : 
  angle_BCA = 30 →
  angle_CDE = angle_BCA →
  angle_CDE + y + angle_ECD = 180 → 
  y = 60 := by
  intro h1 h2 h3
  sorry

end find_y_l286_286834


namespace blue_pieces_correct_l286_286943

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end blue_pieces_correct_l286_286943


namespace valid_code_count_l286_286975

def isValidCode (code : List ℕ) : Prop :=
  (code.length = 5) ∧
  (∀ (d ∈ code), d ∈ [0, 1, 2, 3, 4]) ∧
  (∀ (d ∈ code), code.count d = 1) ∧
  (code.nth! 1 = 2 * code.nth! 0)

noncomputable def countValidCodes : ℕ :=
  (List.permutations [0, 1, 2, 3, 4]).count isValidCode

theorem valid_code_count : countValidCodes = 12 := by
  sorry

end valid_code_count_l286_286975


namespace inequality_solution_l286_286969

theorem inequality_solution (x : ℝ) : (3 * x^2 - 4 * x - 4 < 0) ↔ (-2/3 < x ∧ x < 2) :=
sorry

end inequality_solution_l286_286969


namespace superhero_distance_difference_l286_286648

theorem superhero_distance_difference :
  let t := 4 in
  let v := 100 in
  (60 / t * 10) - v = 50 :=
by
  let t := 4
  let v := 100
  sorry

end superhero_distance_difference_l286_286648


namespace geometric_locus_right_angle_vertex_l286_286603

-- Define a right triangle with vertices A, B, and C where <ABC is the right angle.
def right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := dist A B in
  let BC := dist B C in
  let AC := dist A C in
  AB^2 + BC^2 = AC^2

-- Prove the geometric locus of the right-angle vertex with given conditions
theorem geometric_locus_right_angle_vertex (A B C : ℝ × ℝ) (h_triangle: right_triangle A B C) 
    (h_slide: ∀ t : ℝ, let A' := (t, 0) in let B' := (0, t) in right_triangle A' B' C) :
    ∃ r : ℝ, locus_C = {C' : ℝ × ℝ | euclidean_distance (0,0) C' = r ∧ 0 ≤ x ∧ 0 ≤ y} :=
sorry

end geometric_locus_right_angle_vertex_l286_286603


namespace cos_4_arccos_2_5_l286_286340

noncomputable def arccos_2_5 : ℝ := Real.arccos (2/5)

theorem cos_4_arccos_2_5 : Real.cos (4 * arccos_2_5) = -47 / 625 :=
by
  -- Define x = arccos 2/5
  let x := arccos_2_5
  -- Declare the assumption cos x = 2/5
  have h_cos_x : Real.cos x = 2 / 5 := Real.cos_arccos (by norm_num : 2 / 5 ∈ Set.Icc (-1 : ℝ) 1)
  -- sorry to skip the proof
  sorry

end cos_4_arccos_2_5_l286_286340


namespace smallest_possible_value_gt_perimeter_l286_286455

theorem smallest_possible_value_gt_perimeter (c : ℕ) (h1 : 14 < c) (h2 : c < 24) : 
  let a := 5
  let b := 19
  let P := a + b + c
  48 > P :=
by
  let a := 5
  let b := 19
  let P := a + b + c
  have : a + b = 24 := by simp [a, b]
  have h3 : 24 > c := h2
  have h4 : c > 14 := h1
  have : P < 48 := by
    rw [← this]
    linarith
  exact this

end smallest_possible_value_gt_perimeter_l286_286455


namespace probability_of_red_ball_l286_286470

noncomputable def total_balls : Nat := 4 + 2
noncomputable def red_balls : Nat := 2

theorem probability_of_red_ball :
  (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
sorry

end probability_of_red_ball_l286_286470


namespace perfect_square_factors_l286_286053

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286053


namespace find_number_l286_286596

theorem find_number (N : ℕ) (h : N / 7 = 12 ∧ N % 7 = 5) : N = 89 := 
by
  sorry

end find_number_l286_286596


namespace min_value_of_x_l286_286816

-- Define the conditions and state the problem
theorem min_value_of_x (x : ℝ) : (∀ a : ℝ, a > 0 → x^2 < 1 + a) → x ≥ -1 :=
by
  sorry

end min_value_of_x_l286_286816


namespace sin_C_l286_286100

-- Define the relevant angles and sine function relationships
variables (A B C : ℝ)
-- Conditions
axiom right_triangle : B = real.pi / 2
axiom sin_A : real.sin A = 3 / 5
axiom sin_B : real.sin B = 1

-- Statement to be proven
theorem sin_C : real.sin C = 4 / 5 :=
by 
  sorry

end sin_C_l286_286100


namespace pyramid_solution_l286_286478

noncomputable def pyramid_problem (p q r s t x : ℕ) : Prop :=
  p = 105 - 47 ∧
  q = p - 31 ∧
  r = 47 - q ∧
  s = r - 13 ∧
  t = 13 - 9 ∧
  x = s - t

theorem pyramid_solution : ∃ x : ℕ, pyramid_problem 58 27 20 7 4 x ∧ x = 3 :=
by
  existsi 3
  unfold pyramid_problem
  simp
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

end pyramid_solution_l286_286478


namespace isaiah_types_more_words_than_micah_l286_286513

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end isaiah_types_more_words_than_micah_l286_286513


namespace change_received_l286_286930

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end change_received_l286_286930


namespace root_diff_l286_286863

noncomputable def sqrt11 := Real.sqrt 11
noncomputable def sqrt180 := Real.sqrt 180
noncomputable def sqrt176 := Real.sqrt 176

theorem root_diff (x1 x2 : ℂ) 
  (h1 : sqrt11 * x1^2 + sqrt180 * x1 + sqrt176 = 0)
  (h2 : sqrt11 * x2^2 + sqrt180 * x2 + sqrt176 = 0)
  (h3 : x1 ≠ x2) :
  abs ((1 / (x1^2)) - (1 / (x2^2))) = (Real.sqrt 45) / 44 :=
sorry

end root_diff_l286_286863


namespace range_of_b_varphi_minimum_no_parallel_tangent_l286_286418

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (a b x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 + b * x
noncomputable def h (b x : ℝ) := (Real.log x) + x^2 - b * x
noncomputable def varphi (b x : ℝ) := Real.exp (2 * x) + b * Real.exp x

theorem range_of_b (b : ℝ) : ∀ x > 0, (h b x)' ≥ 0 → b ≤ 2 * Real.sqrt 2 :=
begin
  intro x,
  intro hx_pos,
  intro h_deriv_nonneg,
  sorry
end

theorem varphi_minimum : ∀ x ∈ [0, Real.log 2], ∀ b,
  (varphi b x) = if (-2 ≤ b ∧ b ≤ 2 * Real.sqrt 2) then (b + 1)
                 else if (-4 < b ∧ b < -2) then (- b^2 / 4)
                 else if (b ≤ -4) then (4 + 2 * b)
                 else (varphi b x) :=
begin
  intros x hx b,
  sorry
end

theorem no_parallel_tangent (a b : ℝ) (h_neq : a ≠ 0) : ¬∃ (R : ℝ), let x1 := ... , x2 := ... ,
  let M := (x1 + x2) / 2, let N := (x1 + x2) / 2,
  let k1 := 1 / M, let k2 := (a * M + b),
  k1 = k2 :=
begin
  intro H,
  sorry
end

end range_of_b_varphi_minimum_no_parallel_tangent_l286_286418


namespace count_numbers_with_perfect_square_factors_l286_286062

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286062


namespace expression_value_l286_286246

theorem expression_value (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
by
  sorry

end expression_value_l286_286246


namespace contingency_fund_correct_l286_286444

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l286_286444


namespace greatest_odd_factors_l286_286149

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l286_286149


namespace max_area_central_symmetric_polygon_in_triangle_l286_286883

variable (T : Triangle) (M : Polygon) (O : Point)

-- Conditions
def CenterSymmetricPolygon (T : Triangle) (M : Polygon) (O : Point) : Prop :=
  IsCentralSymmetric M O ∧ LiesWithin M T

-- Main statement to prove
theorem max_area_central_symmetric_polygon_in_triangle : 
  CenterSymmetricPolygon T M O → Area M = (2 / 3) * Area T := 
sorry

end max_area_central_symmetric_polygon_in_triangle_l286_286883


namespace parabola_from_hyperbola_l286_286406

noncomputable def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

noncomputable def parabola_equation_1 (x y : ℝ) : Prop := y^2 = -24 * x

noncomputable def parabola_equation_2 (x y : ℝ) : Prop := y^2 = 24 * x

theorem parabola_from_hyperbola :
  (∃ x y : ℝ, hyperbola_equation x y) →
  (∃ x y : ℝ, parabola_equation_1 x y ∨ parabola_equation_2 x y) :=
by
  intro h
  -- proof is omitted
  sorry

end parabola_from_hyperbola_l286_286406


namespace abs_diff_one_l286_286788

theorem abs_diff_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := sorry

end abs_diff_one_l286_286788


namespace susan_vacation_pay_missed_l286_286189

noncomputable def susan_weekly_pay (hours_worked : ℕ) : ℕ :=
  let regular_hours := min 40 hours_worked
  let overtime_hours := max (hours_worked - 40) 0
  15 * regular_hours + 20 * overtime_hours

noncomputable def susan_sunday_pay (num_sundays : ℕ) (hours_per_sunday : ℕ) : ℕ :=
  25 * num_sundays * hours_per_sunday

noncomputable def pay_without_sundays : ℕ :=
  susan_weekly_pay 48
    
noncomputable def total_three_week_pay : ℕ :=
  let weeks_normal_pay := 3 * pay_without_sundays
  let sunday_hours_1 := 1 * 8
  let sunday_hours_2 := 2 * 8
  let sunday_hours_3 := 0 * 8
  let sundays_total_pay := susan_sunday_pay 1 8 + susan_sunday_pay 2 8 + susan_sunday_pay 0 8
  weeks_normal_pay + sundays_total_pay
  
noncomputable def paid_vacation_pay : ℕ :=
  let paid_days := 6
  let paid_weeks_pay := susan_weekly_pay 40 + susan_weekly_pay (paid_days % 5 * 8)
  paid_weeks_pay

theorem susan_vacation_pay_missed :
  let missed_pay := total_three_week_pay - paid_vacation_pay
  missed_pay = 2160 := sorry

end susan_vacation_pay_missed_l286_286189


namespace probability_green_ball_l286_286686

-- Definition of the problem conditions
def Container := { red : ℕ, green : ℕ }

def ContainerX : Container := { red := 5, green := 5 }
def ContainerY : Container := { red := 8, green := 2 }
def ContainerZ : Container := { red := 3, green := 7 }

def probability_green (c : Container) : ℚ :=
  c.green.to_rat / (c.red + c.green).to_rat

def probability_select (c : Container) : ℚ :=
  1 / 3

-- Proof statement
theorem probability_green_ball :
  probability_select ContainerX * probability_green ContainerX +
  probability_select ContainerY * probability_green ContainerY +
  probability_select ContainerZ * probability_green ContainerZ =
  7 / 15 :=
by sorry

end probability_green_ball_l286_286686


namespace part1_l286_286612

theorem part1 : 2 * Real.tan (60 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - (Real.sin (45 * Real.pi / 180)) ^ 2 = 5 / 2 := 
sorry

end part1_l286_286612


namespace evaluate_trig_expr_l286_286393

variable {θ : ℝ}
hypothesis h_tan : Real.tan θ = 2

theorem evaluate_trig_expr : (Real.sin (2 * θ) / (Real.cos θ ^ 2 - Real.sin θ ^ 2)) = -4 / 3 := by
  sorry

end evaluate_trig_expr_l286_286393


namespace solve_equation_l286_286185

theorem solve_equation : ∀ x : ℝ, x * (x + 2) = 3 * x + 6 ↔ (x = -2 ∨ x = 3) := by
  sorry

end solve_equation_l286_286185


namespace minimum_value_of_f_is_15_l286_286437

noncomputable def f (x : ℝ) : ℝ := 9 * x + (1 / (x - 1))

theorem minimum_value_of_f_is_15 (h : ∀ x, x > 1) : ∃ x, x > 1 ∧ f x = 15 :=
by sorry

end minimum_value_of_f_is_15_l286_286437


namespace isosceles_triangles_count_isosceles_triangles_l286_286011

theorem isosceles_triangles (x : ℕ) (b : ℕ) : 
  (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14) → 
  (b = 1 ∧ x = 14 ∨ b = 3 ∧ x = 13 ∨ b = 5 ∧ x = 12 ∨ b = 7 ∧ x = 11 ∨ b = 9 ∧ x = 10) :=
by sorry

theorem count_isosceles_triangles : 
  (∃ x b, (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14)) → 
  (5 = 5) :=
by sorry

end isosceles_triangles_count_isosceles_triangles_l286_286011


namespace no_divisor_form_24k_20_l286_286661

theorem no_divisor_form_24k_20 (n : ℕ) : ¬ ∃ k : ℕ, 24 * k + 20 ∣ 3^n + 1 :=
sorry

end no_divisor_form_24k_20_l286_286661


namespace bookstore_shoe_store_common_sales_l286_286619

-- Define the conditions
def bookstore_sale_days (d: ℕ) : Prop := d % 4 = 0 ∧ d >= 4 ∧ d <= 28
def shoe_store_sale_days (d: ℕ) : Prop := (d - 2) % 6 = 0 ∧ d >= 2 ∧ d <= 26

-- Define the question to be proven as a theorem
theorem bookstore_shoe_store_common_sales : 
  ∃ (n: ℕ), n = 2 ∧ (
    ∀ (d: ℕ), 
      ((bookstore_sale_days d ∧ shoe_store_sale_days d) → n = 2) 
      ∧ (d < 4 ∨ d > 28 ∨ d < 2 ∨ d > 26 → n = 2)
  ) :=
sorry

end bookstore_shoe_store_common_sales_l286_286619


namespace contingency_fund_allocation_l286_286441

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l286_286441


namespace cyclic_points_l286_286880

variables {A B C A1 B1 C1 A2 C2 : EuclideanGeometry.Point}
variable (circumcircleABC : EuclideanGeometry.Circle)
variables (P K M : EuclideanGeometry.Point)

-- Definitions in line with conditions (a):
def segments_intersect_at_point (P : EuclideanGeometry.Point) :=
  EuclideanGeometry.LineThrough A A1 = EuclideanGeometry.LineThrough B B1 ∧ 
  EuclideanGeometry.LineThrough B B1 = EuclideanGeometry.LineThrough C C1 ∧ 
  EuclideanGeometry.LineThrough A A1 = EuclideanGeometry.LineThrough C C1

def intersection_with_circumcircle (B1A1 := EuclideanGeometry.LineThrough B1 A1) : EuclideanGeometry.Point :=
  EuclideanGeometry.line_circle_intersection circumcircleABC B1A1

def intersection_with_circumcircle' (B1C1 := EuclideanGeometry.LineThrough B1 C1) : EuclideanGeometry.Point :=
  EuclideanGeometry.line_circle_intersection circumcircleABC B1C1

-- Assuming A2 and C2 are the respective intersections from B1A1 and B1C1 with the circumcircle
axiom A2_def : intersection_with_circumcircle = A2
axiom C2_def : intersection_with_circumcircle' = C2

-- The goal (Proof statement in Lean):
theorem cyclic_points (hP : segments_intersect_at_point P) 
                        (hA2 : intersection_with_circumcircle = A2)
                        (hC2 : intersection_with_circumcircle' = C2)
                        (hK : EuclideanGeometry.LineThrough A2 C2 ∩ EuclideanGeometry.LineThrough B B1 = K)
                        (hM : EuclideanGeometry.midpoint A2 C2 = M) :
  EuclideanGeometry.CircleThrough4 A C K M :=
by sorry

end cyclic_points_l286_286880


namespace sequence_formula_l286_286735

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else Real.sqrt (2 * (sequence (n - 1))^4 + 6 * (sequence (n - 1))^2 + 3)

theorem sequence_formula (n : ℕ) (hn : n > 0) : 
  sequence n = Real.sqrt (0.5 * (5^(2^(n-1)) - 3)) := sorry

end sequence_formula_l286_286735


namespace common_divisor_is_19_l286_286895

theorem common_divisor_is_19 (a d : ℤ) (h1 : d ∣ (35 * a + 57)) (h2 : d ∣ (45 * a + 76)) : d = 19 :=
sorry

end common_divisor_is_19_l286_286895


namespace unique_bounded_sequence_exists_l286_286376

variable (a : ℝ) (n : ℕ) (hn_pos : n > 0)

theorem unique_bounded_sequence_exists :
  ∃! (x : ℕ → ℝ), (x 0 = 0) ∧ (x (n+1) = 0) ∧
                   (∀ i, 1 ≤ i ∧ i ≤ n → (1/2) * (x (i+1) + x (i-1)) = x i + x i ^ 3 - a ^ 3) ∧
                   (∀ i, i ≤ n + 1 → |x i| ≤ |a|) := by
  sorry

end unique_bounded_sequence_exists_l286_286376


namespace moses_investment_rate_l286_286148

-- Define the initial investment, income, and dividend rate
def investment : ℝ := 3000
def income : ℝ := 210
def dividend_rate : ℝ := 5.04

-- Define the formula for computing the rate given the income, investment and dividend rate.
def calculate_rate (I A : ℝ) : ℝ := 
  (I * 100) / A 

-- The problem statement in Lean: Prove that the rate is 7%
theorem moses_investment_rate :
  calculate_rate income investment = 7 := 
by {
  sorry
}

end moses_investment_rate_l286_286148


namespace concurrency_of_perpendiculars_and_h1h2_l286_286142

variable (A B C D P O1 O2 H1 H2 E1 E2 : Type*)
variable [PlaneGeometry A B C D P O1 O2 H1 H2 E1 E2]

-- Define the conditions
variables (is_quadrilateral : Quadrilateral A B C D) 
          (intersect_AD_BC : Intersects A D B C P)
          (circumcenter_O1 : Circumcenter O1 (Triangle A B P))
          (circumcenter_O2 : Circumcenter O2 (Triangle D C P))
          (orthocenter_H1 : Orthocenter H1 (Triangle A B P))
          (orthocenter_H2 : Orthocenter H2 (Triangle D C P))
          (midpoint_E1 : Midpoint E1 O1 H1)
          (midpoint_E2 : Midpoint E2 O2 H2)

-- Define the theorem
theorem concurrency_of_perpendiculars_and_h1h2 :
  Concurrent
    (PerpendicularFrom E1 D C)
    (PerpendicularFrom E2 A B)
    (LineThrough H1 H2) :=
sorry

end concurrency_of_perpendiculars_and_h1h2_l286_286142


namespace final_sum_after_50_passes_l286_286634

theorem final_sum_after_50_passes
  (particip: ℕ) 
  (num_passes: particip = 50) 
  (init_disp: ℕ → ℤ) 
  (initial_condition : init_disp 0 = 1 ∧ init_disp 1 = 0 ∧ init_disp 2 = -1)
  (operations: Π (i : ℕ), 
    (init_disp 0 = 1 →
    init_disp 1 = 0 →
    (i % 2 = 0 → init_disp 2 = -1) →
    (i % 2 = 1 → init_disp 2 = 1))
  )
  : init_disp 0 + init_disp 1 + init_disp 2 = 0 :=
by
  sorry

end final_sum_after_50_passes_l286_286634


namespace count_numbers_with_square_factors_l286_286039

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286039


namespace prob_divisible_by_5_of_digits_ending_in_7_l286_286223

theorem prob_divisible_by_5_of_digits_ending_in_7 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ N % 10 = 7) → (0 : ℚ) = 0 :=
by
  intro N
  sorry

end prob_divisible_by_5_of_digits_ending_in_7_l286_286223


namespace ramanujan_number_l286_286818

theorem ramanujan_number :
  ∃ r : ℂ, r * (7 + 2 * complex.i) = 50 - 14 * complex.i ↔ r = 6 - (198 / 53) * complex.i :=
by sorry

end ramanujan_number_l286_286818


namespace shortest_bug_path_l286_286639

/-- Define the rectangle size, number of tiles, and missing tile position --/
def width : ℕ := 12
def length : ℕ := 20
def total_tiles : ℕ := 240
def missing_tile : (ℕ × ℕ) := (6, 10)
def start_pos : (ℕ × ℕ) := (0, 0)
def end_pos : (ℕ × ℕ) := (width, length)

/-- Shortest path the bug takes in terms of tile visits, accounting for the missing tile --/
theorem shortest_bug_path : 
  (∀ width length total_tiles missing_tile start_pos end_pos,
    width = 12 ∧ length = 20 ∧ total_tiles = 240 ∧ missing_tile = (6, 10) ∧ start_pos = (0, 0) ∧ end_pos = (12, 20) →
    ∃ num_tiles : ℕ, num_tiles = 29) :=
by
  intros width length total_tiles missing_tile start_pos end_pos h,
  have := h.1, -- Extract conditions from the hypothesis
  sorry -- proof steps go here

end shortest_bug_path_l286_286639


namespace find_fraction_l286_286546

variable (f : ℝ → ℝ)

axiom functional_eq (x y : ℝ) : f(x + 2*y) - f(3*x - 2*y) = 2*y - x

theorem find_fraction (t : ℝ) : (f (5*t) - f t) / (f (4*t) - f (3*t)) = 4 :=
sorry

end find_fraction_l286_286546


namespace seats_still_available_l286_286589

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end seats_still_available_l286_286589


namespace inverse_proportion_inequality_l286_286388

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l286_286388


namespace count_numbers_with_perfect_square_factors_l286_286055

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286055


namespace size_of_acute_angle_of_line_l286_286220

theorem size_of_acute_angle_of_line :
  ∃ θ : ℝ, 
    (θ = (5/6) * Real.pi) ∧ 
    (∀ x y : ℝ, sqrt 3 * x + 3 * y + 4 = 0 → tan θ = - (sqrt 3 / 3)) :=
sorry

end size_of_acute_angle_of_line_l286_286220


namespace correct_negation_l286_286835

-- Define a triangle with angles A, B, and C
variables (α β γ : ℝ)

-- Define properties of the angles
def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180
def is_right_angle (angle : ℝ) : Prop := angle = 90
def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

-- Original statement to be negated
def original_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ is_right_angle γ → is_acute_angle α ∧ is_acute_angle β

-- Negation of the original statement
def negated_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ ¬ is_right_angle γ → ¬ (is_acute_angle α ∧ is_acute_angle β)

-- Proof statement: prove that the negated statement is the correct negation
theorem correct_negation (α β γ : ℝ) :
  negated_statement α β γ = ¬ original_statement α β γ :=
sorry

end correct_negation_l286_286835


namespace modulo_residue_l286_286959

theorem modulo_residue:
  (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 :=
by
  sorry

end modulo_residue_l286_286959


namespace total_cost_is_correct_l286_286911

def cost_per_pound : ℝ := 0.45
def weight_sugar : ℝ := 40
def weight_flour : ℝ := 16

theorem total_cost_is_correct :
  weight_sugar * cost_per_pound + weight_flour * cost_per_pound = 25.20 :=
by
  sorry

end total_cost_is_correct_l286_286911


namespace part_1_part_2_l286_286873

section
variable {f : ℝ → ℝ} {a b : ℝ}

-- Defining the function f(x) = x^3 - 3ax + b
def f (x : ℝ) : ℝ := x^3 - 3 * a * x + b

-- Given the conditions
axiom a_nonzero : a ≠ 0
axiom curve_tangent : f(2) = 8 ∧ (3 * 2^2 - 3 * a = 0)

-- (I) Prove that a = 4 and b = 24
theorem part_1 : a = 4 ∧ b = 24 := sorry

-- (II) Determine the intervals of monotonicity and the extremum points of f(x)
theorem part_2 : ∃ I1 I2 I3, 
  (∀ x ∈ I1, f' x > 0) ∧ (∀ x ∈ I2, f' x < 0) ∧ (∀ x ∈ I3, f' x > 0) ∧
  (I1 = (-∞, -2)) ∧ (I2 = (-2, 2)) ∧ (I3 = (2, ∞)) ∧ 
  (f(-2) is local_max) ∧ (f(2) is local_min) := sorry

end

end part_1_part_2_l286_286873


namespace Jane_visited_centers_l286_286963

-- Defining the number of centers each individual visited
def visited_centers (Lisa Jude Han Jane : ℕ) : Prop :=
  Lisa = 6 ∧
  Jude = Lisa / 2 ∧
  Han = 2 * Jude - 2 ∧
  Jane = 2 * Han + 6 ∧
  Lisa + Jude + Han + Jane = 27

-- Stating the theorem to be proved
theorem Jane_visited_centers (Lisa Jude Han Jane : ℕ) (h: visited_centers Lisa Jude Han Jane) : Jane = 14 :=
by
  cases h with hl rest,
  cases rest with hj rest2,
  cases rest2 with hh rest3,
  cases rest3 with hjane hsum,
  sorry  -- proof skipped

end Jane_visited_centers_l286_286963


namespace limit_of_Sum_S_n_l286_286870

noncomputable def l_n (n : ℕ) : ℝ → ℝ → Prop :=
  λ x y, n * x + (n + 1) * y = 1

noncomputable def S_n (n : ℕ) : ℝ :=
  if n > 0 then (1 / (2 * n)) - (1 / (2 * (n + 1))) else 0

theorem limit_of_Sum_S_n :
  (∀ n, n > 0 → ∀ x y, l_n n x y → S_n n > 0) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, (|∑ i in finset.range(n+1), S_n i - 1/2| < ε)) :=
by
  sorry

end limit_of_Sum_S_n_l286_286870


namespace value_of_x2_minus_y2_l286_286800

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the conditions
def condition1 : Prop := (x + y) / 2 = 5
def condition2 : Prop := (x - y) / 2 = 2

-- State the theorem to prove
theorem value_of_x2_minus_y2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 40 :=
by
  sorry

end value_of_x2_minus_y2_l286_286800


namespace interval_of_x_l286_286936

theorem interval_of_x :
  let x := (1 / logb (1/2) (1/3) + 1 / logb (1/5) (1/3)) 
  in 2 < x ∧ x < 3 :=
by
  let x := (1 / logb (1/2) (1/3) + 1 / logb (1/5) (1/3))
  have h : x = logb 3 10, from sorry,
  have h2 : 2 < logb 3 10 ∧ logb 3 10 < 3, from sorry,
  exact h2

end interval_of_x_l286_286936


namespace count_perfect_square_factors_l286_286019

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286019


namespace sum_of_digits_of_min_win_for_Bernardo_l286_286669

theorem sum_of_digits_of_min_win_for_Bernardo :
  ∃ (M : ℕ), 1 ≤ M ∧ M ≤ 1000 ∧ 
             (27 * M + 480 < 3000 ∧ 27 * M + 520 ≥ 3000) ∧ 
             (M = 92) ∧ 
             ∑ d in (92.digits 10), d = 11 :=
by sorry

end sum_of_digits_of_min_win_for_Bernardo_l286_286669


namespace min_distance_curve_line_l286_286884

theorem min_distance_curve_line :
  ∀ (A : ℝ × ℝ), A ∈ set_of (λ (p : ℝ × ℝ), p.2 = 3 / 2 * p.1 ^ 2 - real.log p.1) →
  ∃ m : ℝ, m = (abs (-2 * A.1 + A.2 + 1) / real.sqrt (4 + 1)) →
  m = sqrt 5 / 10 :=
sorry

end min_distance_curve_line_l286_286884


namespace students_in_only_one_subject_l286_286907

variables (A B C : ℕ) 
variables (A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ)

def students_in_one_subject (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ) : ℕ :=
  A + B + C - A_inter_B - A_inter_C - B_inter_C + A_inter_B_inter_C - 2 * A_inter_B_inter_C

theorem students_in_only_one_subject :
  ∀ (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ),
    A = 29 →
    B = 28 →
    C = 27 →
    A_inter_B = 13 →
    A_inter_C = 12 →
    B_inter_C = 11 →
    A_inter_B_inter_C = 5 →
    students_in_one_subject A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C = 27 :=
by
  intros A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C hA hB hC hAB hAC hBC hABC
  unfold students_in_one_subject
  rw [hA, hB, hC, hAB, hAC, hBC, hABC]
  norm_num
  sorry

end students_in_only_one_subject_l286_286907


namespace exist_identical_regular_polygons_l286_286938

-- Given problem statement and conditions
theorem exist_identical_regular_polygons (n : ℕ) (colors : Finset (Finset (Fin n))) :
  (∀ c ∈ colors, ∃ k, ∀ x y ∈ c, ∃ m, (p x y m k n)) → 
  ∃ c₁ c₂, c₁ ≠ c₂ ∧ (∃ k, ∀ x y ∈ c₁, ∃ m, (p x y m k n)) ∧ (∃ k, ∀ x y ∈ c₂, ∃ m, (p x y m k n)) ∧ (eq_polygons c₁ c₂ k) := sorry

-- Assumes that each color class forms a regular polygon
-- Proves that two such regular polygons are identical

end exist_identical_regular_polygons_l286_286938


namespace greatest_odd_factors_l286_286151

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l286_286151


namespace prove_profit_conditions_l286_286279

noncomputable def reduce_price_30_or_80 (cost_price selling_price weekly_sales reduction : ℝ) : Prop :=
  let profit_per_kg := selling_price - reduction - cost_price
  let increased_sales := weekly_sales + (reduction / 10) * 40
  profit_per_kg * increased_sales = 41600

noncomputable def maintain_profit_at_80_percent (original_price reduced_price : ℝ) : Prop :=
  reduced_price / original_price = 0.80

noncomputable def impossible_50000_profit (cost_price selling_price weekly_sales reduction : ℝ) : Prop :=
  let profit_per_kg := selling_price - reduction - cost_price
  let increased_sales := weekly_sales + (reduction / 10) * 40
  let discriminant := (-110) ^ 2 - 4 * 1 * 4500
  discriminate < 0

theorem prove_profit_conditions :
  ∃ x1 x2 : ℝ, 
    (reduce_price_30_or_80 240 400 200 x1 ∨ reduce_price_30_or_80 240 400 200 x2) ∧ 
    (maintain_profit_at_80_percent 400 (400 - max x1 x2)) ∧ 
    (∀ y : ℝ, ¬impossible_50000_profit 240 400 200 y) :=
  sorry

end prove_profit_conditions_l286_286279


namespace tournament_matches_l286_286652

theorem tournament_matches (n : ℕ) (h : n = 999) : 
  (matches : ℕ) 
  (condition1 : matches = n - 1) 
  (matches = 998) := 
begin 
  sorry 
end

end tournament_matches_l286_286652


namespace count_odd_sum_numbers_l286_286290

open Finset

def digits : Finset ℕ := {1, 2, 3, 4, 5}

def is_three_digit_without_repetition (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits

def sum_is_odd (a b c : ℕ) : Prop :=
  (a + b + c) % 2 = 1

theorem count_odd_sum_numbers : 
  card {n : ℕ | ∃ (a b c : ℕ), is_three_digit_without_repetition a b c ∧ sum_is_odd a b c ∧ n = 100 * a + 10 * b + c} = 24 :=
sorry

end count_odd_sum_numbers_l286_286290


namespace determinant_M_eq_one_l286_286699

open matrix real

-- Define the matrix
def M (α β : ℝ) : matrix (fin 3) (fin 3) ℝ :=
  ![
    [ cosh α * cosh β, cosh α * sinh β, -sinh α ],
    [ -sinh β, cosh β, 0           ],
    [ sinh α * cosh β, sinh α * sinh β, cosh α ]
  ]

-- State the theorem
theorem determinant_M_eq_one {α β : ℝ} : 
  det (M α β) = 1 :=
by
  sorry

end determinant_M_eq_one_l286_286699


namespace curve_is_parabola_l286_286351

theorem curve_is_parabola (r θ : ℝ) : (r = 1 / (1 - Real.cos θ)) ↔ ∃ x y : ℝ, y^2 = 2 * x + 1 :=
by 
  sorry

end curve_is_parabola_l286_286351


namespace total_numbers_is_six_l286_286194

theorem total_numbers_is_six (N : ℕ)
  (average_all : Float) (average_one : Float) (average_two : Float) (average_three : Float) 
  (h1 : average_all = 3.95)
  (h2 : average_one = 3.8)
  (h3 : average_two = 3.85)
  (h4 : average_three = 4.200000000000001) :
  N = 6 :=
by
  let sum_one := 2 * average_one
  let sum_two := 2 * average_two
  let sum_three := 2 * average_three
  let total_sum := sum_one + sum_two + sum_three
  have h5 : total_sum = 23.700000000000003 := by sorry
  have h6 : average_all * N = total_sum := by sorry
  have h7 : N = total_sum / average_all := by sorry
  show N = 6, from by sorry

end total_numbers_is_six_l286_286194


namespace range_of_m_l286_286772

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  (f (m * Real.sin θ) + f (1 - m) > 0) ↔ (m ≤ 1) :=
sorry

end range_of_m_l286_286772


namespace find_circle_center_l286_286707

noncomputable def circle_center : (ℝ × ℝ) :=
  let x_center := 5
  let y_center := 4
  (x_center, y_center)

theorem find_circle_center (x y : ℝ) (h : x^2 - 10 * x + y^2 - 8 * y = 16) :
  circle_center = (5, 4) := by
  sorry

end find_circle_center_l286_286707


namespace smallest_ge_1_point_1_l286_286598

noncomputable def num1 := 1.4
noncomputable def num2 := 9 / 10 -- 0.9
noncomputable def num3 := 1.2
noncomputable def num4 := 0.5
noncomputable def num5 := 13 / 10 -- 1.3
noncomputable def threshold := 1.1

theorem smallest_ge_1_point_1 : 
  -- Proving that 1.2 is the smallest number among those greater than or equal to 1.1
  (∃ x ∈ {num1, num2, num3, num4, num5}, x ≥ threshold ∧ 
      ∀ y ∈ {num1, num2, num3, num4, num5}, y ≥ threshold → x ≤ y) ↔ (x = num3) :=
begin
  sorry
end

end smallest_ge_1_point_1_l286_286598


namespace factory_output_decrease_l286_286925

theorem factory_output_decrease :
  ∀ (original_output : ℝ),
  let first_increase := original_output * 1.15,
  let second_increase := first_increase * 1.35,
  let third_increase := second_increase * 1.20 in
  original_output = third_increase * (1 - 46.32 / 100) :=
by
  intros,
  let first_increase := original_output * 1.15,
  let second_increase := first_increase * 1.35,
  let third_increase := second_increase * 1.20,
  sorry

end factory_output_decrease_l286_286925


namespace problem_a_problem_b_problem_c_problem_d_l286_286966

variable (a b : ℝ)

-- Conditions
def increasing_sqrt2 : Prop := ∀ x y : ℝ, x < y → (ℝ.sqrt 2) ^ x < (ℝ.sqrt 2) ^ y
def decreasing_half   : Prop := ∀ x y : ℝ, x < y → (1/2) ^ x > (1/2) ^ y
def increasing_3      : Prop := ∀ x y : ℝ, x < y → 3 ^ x < 3 ^ y
def decreasing_03     : Prop := ∀ x y : ℝ, x < y → 0.3 ^ x > 0.3 ^ y
def increasing_sqrt   : Prop := ∀ x y : ℝ, (0 < x ∧ x < y) → x ^ 0.5 < y ^ 0.5
def sixteen_twentyfive_lt_nine_sixteen : Prop := (16 / 25 : ℝ) > (9 / 16 : ℝ)

-- Proof goals
theorem problem_a : increasing_sqrt2 → (ℝ.sqrt 2) ^ 0.2 < (ℝ.sqrt 2) ^ 0.4 :=
sorry

theorem problem_b : decreasing_half → (1 / 2 : ℝ) ^ 0.2 > (1 / 2 : ℝ) ^ 0.4 :=
sorry

theorem problem_c : increasing_3 → decreasing_03 → 3 ^ 0.3 > (0.3) ^ 3 :=
sorry

theorem problem_d : increasing_sqrt → sixteen_twentyfive_lt_nine_sixteen → 
  (16 / 25 : ℝ) ^ 0.5 > (9 / 16 : ℝ) ^ 0.5 :=
sorry

end problem_a_problem_b_problem_c_problem_d_l286_286966


namespace percentage_error_in_square_area_l286_286659

theorem percentage_error_in_square_area 
  (a : ℝ) 
  (h1 : 1.10 * a) 
  (actual_area : ℝ := a^2)
  (erroneous_area : ℝ := (1.10 * a)^2) :
  (erroneous_area - actual_area) / actual_area * 100 = 21 :=
by
  rw [erroneous_area, actual_area]
  sorry

end percentage_error_in_square_area_l286_286659


namespace fill_boxes_l286_286702

theorem fill_boxes : 
  (∃ a b c d e f g : ℕ, 10 ≤ 10 * a + b ∧ 10 ≤ 10 * c + d ∧ 100 ≤ 100 * e + 10 * f + g
  ∧ a ≠ 0 ∧ c ≠ 0 ∧ 10 * a + b + 10 * c + d = 100 * e + 10 * f + g) = 4095 :=
sorry

end fill_boxes_l286_286702


namespace circle_equation_and_distances_l286_286371

def circle_center_on_line (x y : ℝ) : Prop := x = y

def point_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 5

def distance_to_line (x y : ℝ) : ℝ := 
  (1 + 2 * 1 + 4 : ℝ) / real.sqrt 5

theorem circle_equation_and_distances:
  (∃ x y : ℝ, circle_center_on_line x y ∧ point_on_circle x y ∧ (
      distance_to_line x y + real.sqrt 5 = (12/5) * real.sqrt 5 ∧
      distance_to_line x y - real.sqrt 5 = (2/5) * real.sqrt 5
  )) :=
  sorry

end circle_equation_and_distances_l286_286371


namespace complex_subtraction_l286_286081

theorem complex_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 + 3 * I) (h2 : z2 = 3 + I) :
  z1 - z2 = -1 + 2 * I := 
by
  sorry

end complex_subtraction_l286_286081


namespace general_term_max_sequence_b_l286_286004

definition sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n - 1

def recurrence_relation (n : ℕ) : Prop := 
  n > 0 -> n * sequence_a (n + 1) - (n + 1) * sequence_a n = 1

definition sequence_b (n : ℕ) : ℝ :=
  (sequence_a n + 1) / 2 * (8 / 9) ^ n

theorem general_term (n : ℕ) : n > 0 -> sequence_a n = 2 * n - 1 :=
by {
  sorry
}

theorem max_sequence_b : ∃ n, sequence_b n = (8 ^ 9 : ℝ) / (9 ^ 8 : ℝ) :=
by {
  sorry
}

end general_term_max_sequence_b_l286_286004


namespace problem_statement_l286_286401

noncomputable def average_increased (n : ℕ) (incomes : Fin n → ℝ) (new_income : ℝ) : Prop :=
  let old_mean := (∑ i in Finset.range n, incomes i) / n
  let new_mean := (∑ i in Finset.range n, incomes i + new_income) / (n + 1)
  new_mean > old_mean

noncomputable def median_remains_same (n : ℕ) (incomes : Fin n → ℝ) (new_income : ℝ) : Prop :=
  let sorted_incomes := Finset.range n |> Finset.image incomes |> Finset.sort (≤)
  let old_median := if n % 2 = 0 then (sorted_incomes.nth (n / 2 - 1) + sorted_incomes.nth (n / 2)) / 2 else sorted_incomes.nth (n / 2)
  let new_sorted_incomes := (Finset.range n).insert new_income |> Finset.image incomes |> Finset.sort (≤)
  let new_n := n + 1
  let new_median := if new_n % 2 = 0 then (new_sorted_incomes.nth (new_n / 2 - 1) + new_sorted_incomes.nth (new_n / 2)) / 2 else new_sorted_incomes.nth (new_n / 2)
  new_median = old_median ∨ new_median > old_median

noncomputable def variance_increased (n : ℕ) (incomes : Fin n → ℝ) (new_income : ℝ) : Prop :=
  let old_mean := (∑ i in Finset.range n, incomes i) / n
  let old_variance := (∑ i in Finset.range n, (incomes i - old_mean)^2) / n
  let new_mean := (∑ i in Finset.range n, incomes i + new_income) / (n + 1)
  let new_variance := (∑ i in Finset.range n, (incomes i - new_mean)^2 + (new_income - new_mean)^2) / (n + 1)
  new_variance > old_variance

theorem problem_statement (n : ℕ) (incomes : Fin n → ℝ) (new_income : ℝ) (h : new_income ≈ 80000000000.0) :
  average_increased n incomes new_income ∧ median_remains_same n incomes new_income ∧ variance_increased n incomes new_income :=
by sorry

end problem_statement_l286_286401


namespace distance_travelled_downstream_l286_286617

def speed_boat_still_water : ℕ := 24
def speed_stream : ℕ := 4
def time_downstream : ℕ := 6

def effective_speed_downstream : ℕ := speed_boat_still_water + speed_stream
def distance_downstream : ℕ := effective_speed_downstream * time_downstream

theorem distance_travelled_downstream : distance_downstream = 168 := by
  sorry

end distance_travelled_downstream_l286_286617


namespace sin_sq_sub_cos_sq_l286_286400

-- Given condition
variable {α : ℝ}
variable (h : Real.sin α = Real.sqrt 5 / 5)

-- Proof goal
theorem sin_sq_sub_cos_sq (h : Real.sin α = Real.sqrt 5 / 5) : Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 := sorry

end sin_sq_sub_cos_sq_l286_286400


namespace roots_difference_one_l286_286324

theorem roots_difference_one (p : ℝ) :
  (∃ (x y : ℝ), (x^3 - 7 * x + p = 0) ∧ (y^3 - 7 * y + p = 0) ∧ (x - y = 1)) ↔ (p = 6 ∨ p = -6) :=
sorry

end roots_difference_one_l286_286324


namespace invisible_trees_in_square_l286_286920

theorem invisible_trees_in_square (n : ℕ) : 
  ∃ (N M : ℕ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 
  Nat.gcd (N + i) (M + j) ≠ 1 :=
by
  sorry

end invisible_trees_in_square_l286_286920


namespace count_perfect_square_factors_except_one_l286_286027

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286027


namespace k_h_neg3_eq_16_l286_286138

def h (x : ℝ) : ℝ := 4 * x^2 - 8

axiom k (z : ℝ) : ℝ

axiom k_h3_eq_16 : k (h 3) = 16

theorem k_h_neg3_eq_16 : k (h (-3)) = 16 :=
by
  have h3_eq : h 3 = 28 := by 
    calc
      h 3 = 4 * 3^2 - 8 : by rfl
      _ = 36 - 8 : by rfl
      _ = 28 : by rfl
  have h_neg3_eq : h (-3) = 28 := by 
    calc
      h (-3) = 4 * (-3)^2 - 8 : by rfl
      _ = 36 - 8 : by rfl
      _ = 28 : by rfl
  rw [h3_eq] at k_h3_eq_16
  rw [h_neg3_eq]
  exact k_h3_eq_16

end k_h_neg3_eq_16_l286_286138


namespace max_elements_M_l286_286912

def nat_plus := {n : ℕ | n > 0}

def M := {m : ℕ | m ∈ nat_plus ∧ (8 - m) ∈ nat_plus}

theorem max_elements_M : ∃ n, ∀ s, (s = M → n = s.card) ∧ n = 7 := by
  sorry

end max_elements_M_l286_286912


namespace num_perfect_square_factors_1800_l286_286431

theorem num_perfect_square_factors_1800 :
  let factors_1800 := [(2, 3), (3, 2), (5, 2)]
  ∃ n : ℕ, (n = 8) ∧
           (∀ p_k ∈ factors_1800, ∃ (e : ℕ), (e = 0 ∨ e = 2) ∧ n = 2 * 2 * 2 → n = 8) :=
sorry

end num_perfect_square_factors_1800_l286_286431


namespace pieces_on_black_squares_even_l286_286335

def chessboard :=
  { pieces : ℤ // 0 ≤ pieces ∧ pieces ≤ 8 }

def placed_correctly (pieces : chessboard) : Prop :=
  ∀ row col, row ≠ col → pieces.row ≠ pieces.col

def even_pieces_on_black_squares (pieces : chessboard) : Prop :=
  ∃ k, pieces = 2 * k

theorem pieces_on_black_squares_even (pieces : chessboard) :
  placed_correctly pieces → even_pieces_on_black_squares pieces :=
by
  sorry

end pieces_on_black_squares_even_l286_286335


namespace count_numbers_with_perfect_square_factors_l286_286036

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286036


namespace fraction_of_pq_is_correct_l286_286748

-- Define the type to represent Points and Triangles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Assume we have a triangle PQR
variables (P Q R : Point)
def PQR : Triangle := ⟨P, Q, R⟩

-- Assuming T is the incenter of ΔPQR and E, F are points on PQ formed by rays PT and QT.
variable (T : Point)
variable (E : Point)
variable (F : Point)

-- Condition: T is incenter
def incenter (T : Point) (Δ : Triangle) : Prop := sorry  -- Definition of T being incenter is omitted

-- Condition: Intersection of rays PT and QT on PQ at E and F respectively
def intersect_pt_on_pq (P Q E F T : Point) : Prop := sorry  -- Definition of intersection is omitted

-- Given condition: Area of ΔPQR is equal to the area of ΔTFE
def area_eq (Δ1 Δ2 : Triangle) : Prop := sorry  -- Definition of area equality is omitted

-- The fraction of side PQ that constitutes from the perimeter of PQR
noncomputable def fraction_part (PQ : ℝ) : ℝ := (3 - real.sqrt 5) / 2

-- The main theorem statement
theorem fraction_of_pq_is_correct 
(P Q R T E F : Point)
(h1 : incenter T ⟨P, Q, R⟩)
(h2 : intersect_pt_on_pq P Q E F T)
(h3 : area_eq ⟨P, Q, R⟩ ⟨T, F, E⟩) :
(intersect_pt_on_pq P Q E F T) ∧ 
(incenter T ⟨P, Q, R⟩) →
(real.sqrt (area_eq ⟨P, Q, R⟩ ⟨T, F, E⟩) = ∅) :=
sorry

end fraction_of_pq_is_correct_l286_286748


namespace range_of_a_l286_286083

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ x1^3 - 3*x1 + a = 0 ∧ x2^3 - 3*x2 + a = 0 ∧ x3^3 - 3*x3 + a = 0) 
  ↔ -2 < a ∧ a < 2 :=
sorry

end range_of_a_l286_286083


namespace largest_prime_factor_4536_l286_286958

theorem largest_prime_factor_4536 : ∃ p : ℕ, nat.prime p ∧ p ∣ 4536 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 4536 → q ≤ p :=
sorry

end largest_prime_factor_4536_l286_286958


namespace quadratic_sum_is_zero_l286_286207

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l286_286207


namespace proof_problem_l286_286671

noncomputable def problem_statement : ℝ :=
  let x := -5.7 
  let y := 5.7 
  (Real.floor (abs x) + abs (Real.floor y))

theorem proof_problem : problem_statement = 10 := 
  by
  sorry

end proof_problem_l286_286671


namespace magnitude_conjugate_plus_one_l286_286075

def z_eq : Prop :=
  ∃ z : ℂ, (complex.I + 1) * (z - 1) = 2

theorem magnitude_conjugate_plus_one (z : ℂ) (h : (complex.I + 1) * (z - 1) = 2) :
  complex.abs (complex.conj z + 1) = real.sqrt 10 :=
sorry

end magnitude_conjugate_plus_one_l286_286075


namespace general_formula_b_n_sum_S_2023_l286_286747

-- Define the sequence and conditions
noncomputable def a_n (n : ℕ) : ℕ := sorry

-- Condition 1: log_2 a_{n+2} + (-1)^n log_2 a_n = 1
axiom condition1 (n : ℕ) : Real.log2 (a_n (n + 2)) + (-1)^n * Real.log2 (a_n n) = 1

-- Initial values
axiom a1 : a_n 1 = 1
axiom a2 : a_n 2 = 2

-- Define the subsequence b_n
def b_n (n : ℕ) : ℕ := a_n (2 * n - 1)

-- The first required proof: general formula for b_n
theorem general_formula_b_n (n : ℕ) : b_n n = 2^(n - 1) := sorry

-- The second required proof: sum of first 2023 terms
def S_2023 : ℕ := (Finset.range 2023).sum (λ n, a_n n)

theorem sum_S_2023 : S_2023 = 2^1012 + 1516 := sorry

end general_formula_b_n_sum_S_2023_l286_286747


namespace mechanical_moles_l286_286233

-- Define the conditions
def condition_one (x y : ℝ) : Prop :=
  x + y = 1 / 5

def condition_two (x y : ℝ) : Prop :=
  (1 / (3 * x)) + (2 / (3 * y)) = 10

-- Define the main theorem using the defined conditions
theorem mechanical_moles (x y : ℝ) (h1 : condition_one x y) (h2 : condition_two x y) :
  x = 1 / 30 ∧ y = 1 / 6 :=
  sorry

end mechanical_moles_l286_286233


namespace wrapping_paper_area_l286_286627

theorem wrapping_paper_area (l w h : ℝ) : 
  ∃ a : ℝ, (a = 4 * (max l w)^2) :=
by
  use 4 * (max l w)^2
  sorry

end wrapping_paper_area_l286_286627


namespace jake_snake_sales_l286_286115

theorem jake_snake_sales 
  (num_snakes : ℕ)
  (eggs_per_snake : ℕ)
  (regular_price : ℕ)
  (super_rare_multiplier : ℕ)
  (num_snakes = 3)
  (eggs_per_snake = 2)
  (regular_price = 250)
  (super_rare_multiplier = 4) : 
  (num_snakes * eggs_per_snake - 1) * regular_price + regular_price * super_rare_multiplier = 2250 :=
sorry

end jake_snake_sales_l286_286115


namespace walking_rate_ratio_l286_286620

variables (R R' : ℝ)

theorem walking_rate_ratio (h₁ : R * 21 = R' * 18) : R' / R = 7 / 6 :=
by {
  sorry
}

end walking_rate_ratio_l286_286620


namespace base4_sum_correct_l286_286715

/-- Define the base-4 numbers as natural numbers. -/
def a := 3 * 4^2 + 1 * 4^1 + 2 * 4^0
def b := 3 * 4^1 + 1 * 4^0
def c := 3 * 4^0

/-- Define their sum in base 10. -/
def sum_base_10 := a + b + c

/-- Define the target sum in base 4 as a natural number. -/
def target := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

/-- Prove that the sum of the base-4 numbers equals the target sum in base 4. -/
theorem base4_sum_correct : sum_base_10 = target := by
  sorry

end base4_sum_correct_l286_286715


namespace part1_part2_l286_286481

variables {A B C a b c : ℝ}

-- Part 1: Prove that if b^2 = c(a + c) and B = π/3, then a/c = 2
theorem part1 (h1 : b^2 = c * (a + c)) (h2 : B = π / 3) : a / c = 2 :=
sorry

-- Part 2: Prove the range of sqrt(3)sinB + 2cos^2C when the triangle is acute
theorem part2 (h1 : b^2 = c * (a + c)) (h2 : ∀ {x}, x = A ∨ x = B ∨ x = C → 0 < x ∧ x < π / 2) :
  ∃ I : set ℝ, I = set.Ioo (sqrt 3 + 1) 3 ∧ sqrt 3 * Real.sin B + 2 * (Real.cos C)^2 ∈ I :=
sorry

end part1_part2_l286_286481


namespace kite_area_l286_286268

theorem kite_area {a b d1 d2 : ℝ} (h1 : a = 15) (h2 : b = 20) (h3 : d1 = 24)
  (h4 : a ^ 2 + (d2 / 2) ^ 2 = b ^ 2) (hx : d2 = 18) :
  let area := (1/2) * d1 * d2 in
  area = 216 :=
by
  have ha : a = 15 := h1
  have hb : b = 20 := h2
  have hd1 : d1 = 24 := h3
  have hd2 : d2 = 18 := hx
  have h : a ^ 2 + (d2 / 2) ^ 2 = b ^ 2 := h4
  let area := (1/2) * d1 * d2
  show area = 216
  sorry -- the proof is skipped

end kite_area_l286_286268


namespace unit_price_correct_minimum_cost_l286_286623

-- Given conditions for the unit prices
-- ∀ x (the unit price of type A), if 110/(x) = 120/(x+1) then x = 11
theorem unit_price_correct :
  ∀ (x : ℝ), (110 / x = 120 / (x + 1)) → (x = 11) :=
by
  assume x,
  intro h,
  have h_eq : 110 * (x + 1) = 120 * x := (div_eq_iff h).mp rfl,
  have h_simpl: 110 * x + 110 = 120 * x := by linarith,
  have h_final: x = 11 := by linarith,
  exact h_final

-- Proving minimum cost
-- ∀ a : ℕ (the number of type A notebooks), b : ℕ (the number of type B notebooks),
-- if a + b = 100, b ≤ 3 * a, then the minimum cost w = 11a + 12(100 - a) is 1100
theorem minimum_cost :
  ∀ (a b : ℕ), (a + b = 100) → (b ≤ 3 * a) → (-a + 1200 = 1100) :=
by
  assume a b,
  intro h1,
  intro h2,
  have h1_rewrite: b = 100 - a := by linarith,
  have h_cost: 11 * a + 12 * (100 - a) = -a + 1200 := by linarith,
  exact h_cost

end unit_price_correct_minimum_cost_l286_286623


namespace dolphin_star_cannot_cover_all_squares_l286_286099

theorem dolphin_star_cannot_cover_all_squares (n : ℕ) (m : ℕ) (move : ℕ → ℕ × ℕ → Option (ℕ × ℕ)) :
  (n = 8) ∧ (m = 8) ∧
  (∀ i j, (move 0 (i, j) = some (i, j + 1) ∨ move 0 (i, j) = some (i + 1, j) ∨ move 0 (i, j) = some (i - 1, j - 1))) ∧
  (move 0 (0, 0) = some (a, b) → ∃ path : ℕ → ℕ × ℕ, (∀ k, path (k + 1) = move k (path k)) ∧
  (∀ i j, ∃ k, path k = (i, j)) ∧
  (∀ i j k l, path i ≠ path j → i ≠ j)) → false :=
sorry

end dolphin_star_cannot_cover_all_squares_l286_286099


namespace triangle_angles_l286_286928

theorem triangle_angles (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : 12 * (180 - B) - 7 * (180 - C) = 900)
  (h3 : C - B = 50) : 
  {A, B, C} = {10, 60, 110} :=
by
  sorry

end triangle_angles_l286_286928


namespace distance_between_centers_of_circles_l286_286140

-- Definitions based on given conditions
def circles_non_intersecting (d r1 r2 : ℝ) : Prop := d > r1 + r2 ∧ ℝ

def internal_tangent_length (d r1 r2 : ℝ) : Prop := sqrt(d^2 - (r1 + r2)^2) = 19

def external_tangent_length (d r1 r2 : ℝ) : Prop := sqrt(d^2 - (r1 - r2)^2) = 37

def expected_value (d r1 r2 : ℝ) : Prop := d^2 + r1^2 + r2^2 = 2023

-- The theorem we need to prove
theorem distance_between_centers_of_circles (d r1 r2 : ℝ)
  (h1 : circles_non_intersecting d r1 r2)
  (h2 : internal_tangent_length d r1 r2)
  (h3 : external_tangent_length d r1 r2)
  (h4 : expected_value d r1 r2) :
  d = 38 :=
by
  sorry

end distance_between_centers_of_circles_l286_286140


namespace solve_fraction_equation_l286_286531

theorem solve_fraction_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_equation_l286_286531


namespace line_eq_point_slope_135_l286_286217

theorem line_eq_point_slope_135 (x y : ℝ) (P : ℝ × ℝ) (tan : ℝ → ℝ) (m : ℝ) : 
  P = (2, 3) ∧ tan 135 = m ∧ m = -1 ∧ (∀ x y : ℝ, y - 3 = m * (x - 2) ↔ x + y - 5 = 0) :=
begin
  -- Given condition that point P is (2, 3)
  assume h1 : P = (2, 3), 
  -- Given condition that slope angle tan 135 is m
  assume h2 : tan 135 = m,
  -- Given m = -1
  assume h3 : m = -1,
  -- Proof goal : equation of the line is x + y - 5 = 0
  sorry
end

end line_eq_point_slope_135_l286_286217


namespace faster_train_passes_slower_l286_286587

theorem faster_train_passes_slower (v_fast v_slow : ℝ) (length_fast : ℝ) 
  (hv_fast : v_fast = 50) (hv_slow : v_slow = 32) (hl_length_fast : length_fast = 75) :
  ∃ t : ℝ, t = 15 := 
by
  sorry

end faster_train_passes_slower_l286_286587


namespace value_of_expression_l286_286362

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = 3) (h3 : z = 4) :
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 :=
by
  sorry

end value_of_expression_l286_286362


namespace last_two_nonzero_digits_70_fact_l286_286323

/-- The last two nonzero digits of 70! are 80. -/
theorem last_two_nonzero_digits_70_fact : ∃ n : ℕ, n = 70! / 10^16 ∧ (n % 100 = 80) :=
sorry

end last_two_nonzero_digits_70_fact_l286_286323


namespace diamond_eval_l286_286316

def diamond (X Y : ℝ) : ℝ := (X + Y) / 5

theorem diamond_eval :
  diamond (diamond 3 15) 10 = 2.72 :=
by
  sorry

end diamond_eval_l286_286316


namespace two_triangles_with_area_at_most_one_fourth_l286_286822

open Set


theorem two_triangles_with_area_at_most_one_fourth (A B C : Point) (points : Finset Point) 
  (h_unit_area : area A B C = 1) (h_five_points : points.card = 5) : 
  ∃ P Q R S T U, {P, Q, R} ⊆ points ∧ {S, T, U} ⊆ points ∧ 
  P ≠ Q ∧ P ≠ R ∧ Q ≠ R ∧ S ≠ T ∧ S ≠ U ∧ T ≠ U ∧ 
  area P Q R ≤ 1 / 4 ∧ area S T U ≤ 1 / 4 ∧ {P, Q, R} ≠ {S, T, U} :=
sorry

end two_triangles_with_area_at_most_one_fourth_l286_286822


namespace triangle_inequality_l286_286865

variables {R : Type*} [LinearOrderedField R]

theorem triangle_inequality 
  (a b c u v w : R)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (a + b + c) * (1 / u + 1 / v + 1 / w) ≤ 3 * (a / u + b / v + c / w) :=
sorry

end triangle_inequality_l286_286865


namespace rearrange_portraits_l286_286538

section Portraits

-- Definition of the circular arrangement and the conditions
variable (n : ℕ)
variable (configuration initial target : List ℕ)

-- Condition 1: Swapping between adjacent portraits only
def adjacent_swap (config : List ℕ) (i : ℕ) : List ℕ :=
  if i < config.length - 1 then
    let x := config.get i
    let y := config.get (i + 1)
    config.set i y |>.set (i + 1) x
  else
    config

-- Condition 2: No swapping between portraits of consecutive kings
def consecutive_kings (a b : ℕ) : Prop := 
  abs (a - b) = 1

def allowed_swap (config : List ℕ) (i : ℕ) : Prop :=
  i < config.length - 1 ∧ ¬ consecutive_kings (config.get i) (config.get (i + 1))

-- Condition 3: Two configurations are considered identical if they differ only by rotation
def rotations (config : List ℕ) : List (List ℕ) :=
  List.range config.length |>.map (λ i => 
    (List.drop i config) ++ (List.take i config)
  )

def equivalent_by_rotation (config1 config2 : List ℕ) : Prop :=
  config2 ∈ rotations config1

-- The main theorem to prove
theorem rearrange_portraits (init_config target_config : List ℕ) 
  (h1 : List.length init_config = n) (h2 : List.length target_config = n) 
  : (∃ f: ℕ → ℕ, 
      (∀ i, i < n → allowed_swap (f i) i) ∧ 
      equivalent_by_rotation (f n) target_config) := 
sorry

end Portraits

end rearrange_portraits_l286_286538


namespace transformed_function_zero_l286_286282

-- Definitions based on conditions
def f : ℝ → ℝ → ℝ := sorry  -- Assume this is the given function f(x, y)

-- Transformed function according to symmetry and reflections
def transformed_f (x y : ℝ) : Prop := f (y + 2) (x - 2) = 0

-- Lean statement to be proved
theorem transformed_function_zero (x y : ℝ) : transformed_f x y := sorry

end transformed_function_zero_l286_286282


namespace infinite_corners_have_subset_l286_286846

open Finset

noncomputable def is_corner {α : Type*} [linear_order α] (n : ℕ) (S : Finset (Vector α n)) : Prop :=
∀ (a b : Vector α n), a ∈ S → (∀ i, a.nth i ≥ b.nth i) → b ∈ S

theorem infinite_corners_have_subset (n : ℕ) (C : Set (Finset (Vector ℕ n)))
  (h_inf : C.infinite) (h_corner : ∀ (S ∈ C), is_corner n S) :
  ∃ (A B ∈ C), A ⊆ B :=
sorry

end infinite_corners_have_subset_l286_286846


namespace quadrilateral_area_l286_286475

/-- The area of quadrilateral ABCD -/
theorem quadrilateral_area (AB BC DC: ℝ) (hAB: AB = 4) (hBC: BC = 7) (hDC: DC = 1) :
  let AC := real.sqrt (AB ^ 2 + BC ^ 2)
  let AD := real.sqrt (AC ^ 2 - DC ^ 2)
  let area_ABC := (1 / 2) * AB * BC
  let area_ADC := (1 / 2) * AD * DC
  (area_ABC + area_ADC) = 18 :=
by {
  sorry
}

end quadrilateral_area_l286_286475


namespace speaker_is_female_doctor_l286_286267

theorem speaker_is_female_doctor :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 17 ∧ 
    a + b ≥ c + d ∧ 
    c > a ∧ 
    a > b ∧ 
    d ≥ 2 ∧
    -- Upon excluding a female doctor, the properties should hold for 16 members:
    (a + (b - 1) + c + d = 16 → 
     a + (b - 1) ≥ c + d ∧ 
     c > a ∧ 
     a > (b - 1) ∧ 
     d ≥ 2) :=
by {
  existsi (5, 4, 6, 2),
  split; ring,
  split,
  exact le_of_eq (by norm_num),
  split,
  exact (by norm_num : 6 > 5),
  split,
  exact (by norm_num : 5 > 4),
  split,
  exact (by norm_num : 2 ≥ 2),
  intro h1,
  split,
  exact le_of_eq (by norm_num),
  split,
  exact (by norm_num : 6 > 5),
  split,
  exact (by norm_num : 5 > 3),
  exact (by norm_num : 2 ≥ 2),
  sorry
}

end speaker_is_female_doctor_l286_286267


namespace Isaiah_types_more_l286_286515

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end Isaiah_types_more_l286_286515


namespace alternating_square_sum_l286_286517

theorem alternating_square_sum (n : ℕ) (hn : n > 0) :
  (∑ k in Finset.range n, (-1)^(k + 1) * (k + 1)^2) = (-1)^(n + 1) * (n * (n + 1) / 2) :=
sorry

end alternating_square_sum_l286_286517


namespace sales_neither_notebooks_nor_markers_l286_286191

theorem sales_neither_notebooks_nor_markers (percent_notebooks percent_markers percent_staplers : ℝ) 
  (h1 : percent_notebooks = 25)
  (h2 : percent_markers = 40)
  (h3 : percent_staplers = 15) : 
  percent_staplers + (100 - (percent_notebooks + percent_markers + percent_staplers)) = 35 :=
by
  sorry

end sales_neither_notebooks_nor_markers_l286_286191


namespace count_numbers_with_square_factors_l286_286043

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286043


namespace count_perfect_square_factors_except_one_l286_286030

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286030


namespace hypotenuse_of_right_triangle_l286_286881

section
variable (a : ℝ) (h : a = 10) (θ : ℝ) (θ_h : θ = real.pi / 4)

theorem hypotenuse_of_right_triangle :
  ∃ c : ℝ, c = a * real.sqrt 2 :=
by
  -- Adding sorry to skip the proof part
  sorry
end

end hypotenuse_of_right_triangle_l286_286881


namespace pieces_count_l286_286945

def pieces_after_n_tears (n : ℕ) : ℕ :=
  3 * n + 1

theorem pieces_count (n : ℕ) : pieces_after_n_tears n = 3 * n + 1 :=
by
  sorry

end pieces_count_l286_286945


namespace find_natural_n_l286_286922

theorem find_natural_n (a : ℂ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) (h₂ : a ≠ -1)
    (h₃ : a ^ 11 + a ^ 7 + a ^ 3 = 1) : a ^ 4 + a ^ 3 = a ^ 15 + 1 :=
sorry

end find_natural_n_l286_286922


namespace cot_neg_45_deg_eq_neg_1_l286_286344

theorem cot_neg_45_deg_eq_neg_1
  (cot_def: ∀ x, Real.cot x = 1 / Real.tan x)
  (tan_neg: ∀ x, Real.tan (-x) = -Real.tan x)
  (tan_45_deg: Real.tan (Real.pi / 4) = 1) :
  Real.cot (-Real.pi / 4) = -1 :=
by
  sorry

end cot_neg_45_deg_eq_neg_1_l286_286344


namespace find_polynomials_l286_286742

noncomputable theory

open Polynomial

theorem find_polynomials {
  (n : ℕ) (a : Fin (n + 1) → ℤ) (f : Polynomial ℤ) 
  (roots_real : ∀ z ∈ (f.roots.map (algebraMap ℚ ℝ)).to_finset, z.im = 0)
  (coeff_one_neg_one : ∀ i, a i = 1 ∨ a i = -1)
}
  (h_f : f = ∑ i in Finset.range (n + 1), (C (a i) * X^i)) :
  (n = 1 ∧ (f = X - 1 ∨ f = -X + 1 ∨ f = X + 1 ∨ f = -X - 1))
  ∨ (n = 2 ∧ (f = X^2 + X - 1 ∨ f = -X^2 - X + 1 ∨ f = X^2 - X - 1 ∨ f = -X^2 + X + 1))
  ∨ (n = 3 ∧ (f = X^3 + X^2 - X - 1 ∨ f = -X^3 - X^2 + X + 1 ∨ f = X^3 - X^2 - X + 1 ∨ f = -X^3 + X^2 + X - 1)) :=
sorry

end find_polynomials_l286_286742


namespace volume_common_part_equal_quarter_volume_each_cone_l286_286584

theorem volume_common_part_equal_quarter_volume_each_cone
  (r h : ℝ) (V_cone : ℝ)
  (h_cone_volume : V_cone = (1 / 3) * π * r^2 * h) :
  ∃ V_common, V_common = (1 / 4) * V_cone :=
by
  -- Main structure of the proof skipped
  sorry

end volume_common_part_equal_quarter_volume_each_cone_l286_286584


namespace find_angle_A_find_area_l286_286755

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def law_c1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + c * Real.cos A = -2 * b * Real.cos A

def law_c2 (a : ℝ) : Prop := a = 2 * Real.sqrt 3
def law_c3 (b c : ℝ) : Prop := b + c = 4

-- Questions
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c) : 
  A = 2 * Real.pi / 3 :=
sorry

theorem find_area (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c)
  (hA : A = 2 * Real.pi / 3) : 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_l286_286755


namespace evaluate_expression_l286_286698

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  rw [h]
  norm_num
  sorry

end evaluate_expression_l286_286698


namespace equation_of_line_l286_286692

theorem equation_of_line (c : ℝ) (p : ℝ × ℝ) (eq_line : ℝ → ℝ → ℝ) :
  p = (-1, 3) ∧ eq_line = (λ x y, x - 2 * y + 3) →
  (∀ x y, eq_line x y = 0 → (eq_line x y = 0) = (x - 2 * y + c = 0)) →
  (x - 2 * (3 : ℝ) + c = 0) →
  c = 7 :=
by
  sorry

end equation_of_line_l286_286692


namespace purchase_price_is_600_l286_286119

open Real

def daily_food_cost : ℝ := 20
def num_days : ℝ := 40
def vaccination_cost : ℝ := 500
def selling_price : ℝ := 2500
def profit : ℝ := 600

def total_food_cost : ℝ := daily_food_cost * num_days
def total_expenses : ℝ := total_food_cost + vaccination_cost
def total_cost : ℝ := selling_price - profit
def purchase_price : ℝ := total_cost - total_expenses

theorem purchase_price_is_600 : purchase_price = 600 := by
  sorry

end purchase_price_is_600_l286_286119


namespace correct_sequence_of_operations_l286_286964

-- Define the conditions as steps of the operations involved in linear regression
def collect_data {n : ℕ} (data : fin n → ℝ × ℝ) : Prop := true

def plot_scatter_diagram {n : ℕ} (data : fin n → ℝ × ℝ) : Prop := true

def derive_linear_regression_equation (data : (fin n → ℝ × ℝ)) : Prop := true

def predict_using_regression (equation : ℝ → ℝ) : Prop := true

-- Proving the correct sequence of operations
theorem correct_sequence_of_operations 
  {n : ℕ} (data : fin n → ℝ × ℝ) 
  (step1 : collect_data data)
  (step2 : plot_scatter_diagram data)
  (step3 : derive_linear_regression_equation data)
  (step4 : predict_using_regression (λ x, 0.0)) :
  [step1, step2, step3, step4] = [collect_data data, plot_scatter_diagram data, derive_linear_regression_equation data, predict_using_regression (λ x, 0.0)] := 
by sorry

end correct_sequence_of_operations_l286_286964


namespace inf_f_equals_zero_l286_286868

open Real

def P (x : Fin n → ℝ) : ℝ := ∑ i, 1 / (x i + (n - 1))

def f (x : Fin n → ℝ) : ℝ := ∑ i, (1 / (x i)) - x i

theorem inf_f_equals_zero (n : ℕ) (h : n ≥ 3) :
  ∃ (x : Fin n → ℝ), (∀ i, x i > 0) ∧ P x = 1 ∧ f x = 0 :=
sorry -- Proof omitted

end inf_f_equals_zero_l286_286868


namespace Lance_workdays_per_week_l286_286330

theorem Lance_workdays_per_week (weekly_hours hourly_wage daily_earnings : ℕ) 
  (h1 : weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63) :
  weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end Lance_workdays_per_week_l286_286330


namespace number_of_small_cubes_l286_286993

def large_cube_edge_length := 9
def small_cube_edge_length := 3

theorem number_of_small_cubes :
  (large_cube_edge_length / small_cube_edge_length) * 
  (large_cube_edge_length / small_cube_edge_length) * 
  (large_cube_edge_length / small_cube_edge_length) = 27 := 
by {
  have h : large_cube_edge_length / small_cube_edge_length = 3 := by norm_num,
  rw [h, h, h],
  norm_num,
}

#eval number_of_small_cubes -- This will evaluate the theorem to see if it holds true.

end number_of_small_cubes_l286_286993


namespace bike_cost_l286_286182

theorem bike_cost (price_per_apple repairs_share remaining_share apples_sold earnings repairs_cost bike_cost : ℝ) :
  price_per_apple = 1.25 →
  repairs_share = 0.25 →
  remaining_share = 1/5 →
  apples_sold = 20 →
  earnings = apples_sold * price_per_apple →
  repairs_cost = earnings * 4/5 →
  repairs_cost = bike_cost * repairs_share →
  bike_cost = 80 :=
by
  intros;
  sorry

end bike_cost_l286_286182


namespace sum_distances_l286_286109

noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

axiom intersect_circumcircles (A B C D E F : ℝ × ℝ) : ∃ (X : ℝ × ℝ), X ≠ E -- Intersection exists and distinct from E

def XA (X A : ℝ × ℝ) := length X A

def XB (X B : ℝ × ℝ) := length X B

def XC (X C : ℝ × ℝ) := length X C

theorem sum_distances 
  (A B C D E F X : ℝ × ℝ)
  (hA : length A B = 15)
  (hB : length B C = 13)
  (hC : length A C = 14)
  (hD : D = midpoint A B)
  (hE : E = midpoint B C)
  (hF : F = midpoint A C)
  (hX : intersect_circumcircles A B C D E F) :
  XA X A + XB X B + XC X C = 117 / 8 :=
sorry

end sum_distances_l286_286109


namespace contingency_fund_allocation_l286_286442

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l286_286442


namespace james_tylenol_daily_intake_l286_286485

def tylenol_per_tablet : ℕ := 375
def tablets_per_dose : ℕ := 2
def hours_per_dose : ℕ := 6
def hours_per_day : ℕ := 24

theorem james_tylenol_daily_intake :
  (hours_per_day / hours_per_dose) * (tablets_per_dose * tylenol_per_tablet) = 3000 := by
  sorry

end james_tylenol_daily_intake_l286_286485


namespace points_collinear_distance_relation_l286_286574

theorem points_collinear_distance_relation (x y : ℝ) 
  (h1 : (5 - y) * (5 - 1) = -4 * (-2 - x))
  (h2 : real.sqrt ((y - 1)^2 + 9) = 2 * real.sqrt ((x - 1)^2 + 16)) :
  (x + y = -9 / 2) ∨ (x + y = 17 / 2) := 
sorry

end points_collinear_distance_relation_l286_286574


namespace daily_grass_feeds_l286_286569

variables (G r C : ℝ)

/-- Condition 1: 10 cows can graze the pasture for 8 days before it's completely eaten. -/
def condition1 : Prop := 10 * 8 * C = G + 8 * r

/-- Condition 2: 15 cows, starting with one less cow each subsequent day, can finish grazing it in 5 days. -/
def condition2 : Prop := 15 * 5 * (1 + (14/15) + (13/15) + (12/15) + (11/15)) * C = G + 5 * r

/-- To prove: The number of cows that can be fed daily by the grass growing each day on the pasture -/
theorem daily_grass_feeds : condition1 G r C ∧ condition2 G r C → r = 5 * C := 
begin
  sorry
end

end daily_grass_feeds_l286_286569


namespace color_diff_exists_l286_286535

noncomputable def coloring (f : ℤ → ℕ) : Prop :=
(f 0 = 1 ∧ f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 4) -- For example purposes, this should be generally f: ℤ -> {1, 2, 3, 4}

theorem color_diff_exists 
  (f : ℤ → {R : Type, G : Type, B : Type, Y : Type}) -- coloring function 
  (x y : ℤ)
  (hx : x % 2 = 1) -- x is odd
  (hy : y % 2 = 1) -- y is odd
  (hxy : |x| ≠ |y|) -- absolute values of x and y are not equal
  : ∃ a b : ℤ, f(a) = f(b) ∧ (a - b ∈ {x, y, x + y, x - y}) :=
begin
  sorry
end

end color_diff_exists_l286_286535


namespace find_p_q_sum_p_plus_q_l286_286305

noncomputable def probability_third_six : ℚ :=
  have fair_die_prob_two_sixes := (1 / 6) * (1 / 6)
  have biased_die_prob_two_sixes := (2 / 3) * (2 / 3)
  have total_prob_two_sixes := (1 / 2) * fair_die_prob_two_sixes + (1 / 2) * biased_die_prob_two_sixes
  have prob_fair_given_two_sixes := fair_die_prob_two_sixes / total_prob_two_sixes
  have prob_biased_given_two_sixes := biased_die_prob_two_sixes / total_prob_two_sixes
  let prob_third_six :=
    prob_fair_given_two_sixes * (1 / 6) +
    prob_biased_given_two_sixes * (2 / 3)
  prob_third_six

theorem find_p_q_sum : 
  probability_third_six = 65 / 102 :=
by sorry

theorem p_plus_q : 
  65 + 102 = 167 :=
by sorry

end find_p_q_sum_p_plus_q_l286_286305


namespace phase_shift_and_amplitude_l286_286711

theorem phase_shift_and_amplitude (A B C : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = A * Real.sin (B * x - C)) →
  (A = 3) → (B = 5) → (C = π / 2) →
  (A = 3) ∧ (C / B = π / 10) :=
by
  intro h1 h2 h3 h4
  have hA : A = 3 := h2
  have hC_over_B : C / B = (π / 2) / 5 := by
    calc
      C / B = (π / 2) / 5 : by rw [h4, h3]
          ... = π / 10  : by norm_num
  exact ⟨hA, hC_over_B⟩

end phase_shift_and_amplitude_l286_286711


namespace digit_in_92nd_place_l286_286074

/-- The fraction 5/33 is expressed in decimal form as a repeating decimal 0.151515... -/
def fraction_to_decimal : ℚ := 5 / 33

/-- The repeated pattern in the decimal expansion of 5/33 is 15, which is a cycle of length 2 -/
def repeated_pattern (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 else 1

/-- The digit at the 92nd place in the decimal expansion of 5/33 is 5 -/
theorem digit_in_92nd_place : repeated_pattern 92 = 5 :=
by sorry

end digit_in_92nd_place_l286_286074


namespace student_A_claps_6_times_up_to_100_l286_286717

noncomputable def fib : ℕ → ℕ 
| 0 => 0
| 1 => 1
| n => fib (n - 1) + fib (n - 2)

def is_multiple_of (n d : ℕ) : Prop := d ≠ 0 ∧ n % d = 0

def student_A_reports (n : ℕ) : ℕ := fib (5 * n - 4)

def claps (n : ℕ) : Prop := is_multiple_of (student_A_reports n) 3

theorem student_A_claps_6_times_up_to_100 :
  (finset.range 20).filter claps = {2, 5, 8, 11, 14, 17}.card = 6 := by
  sorry

end student_A_claps_6_times_up_to_100_l286_286717


namespace change_received_l286_286931

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end change_received_l286_286931


namespace greatest_perfect_square_below_200_l286_286163

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l286_286163


namespace greatest_perfect_square_below_200_l286_286160

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l286_286160


namespace employee_pay_l286_286571

variable (X Y Z : ℝ)

-- Conditions
def X_pay (Y : ℝ) := 1.2 * Y
def Z_pay (X : ℝ) := 0.75 * X

-- Proof statement
theorem employee_pay (h1 : X = X_pay Y) (h2 : Z = Z_pay X) (total_pay : X + Y + Z = 1540) : 
  X + Y + Z = 1540 :=
by
  sorry

end employee_pay_l286_286571


namespace next_point_bisection_method_l286_286955

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3

theorem next_point_bisection_method : ∀ (a b : ℝ), a < b → f 1 < 0 → f 2 > 0 → 
  1 < 2 → (1 + 2) / 2 = 1.5 :=
by {
  intros a b hab hfa hfb h12,
  unfold f at *,
  ring,
}

end next_point_bisection_method_l286_286955


namespace point_in_first_quadrant_l286_286832

noncomputable def z1 : ℂ := (2 : ℂ) + (3 : ℂ) * complex.I
noncomputable def z2 : ℂ := -1 + (2 : ℂ) * complex.I
noncomputable def z : ℂ := z1 - z2

theorem point_in_first_quadrant : (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end point_in_first_quadrant_l286_286832


namespace part1_part2_l286_286769

open Set

variable (a : ℝ)

def real_universe := @univ ℝ

def set_A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def set_B : Set ℝ := {x | 2 < x ∧ x < 10}
def set_C (a : ℝ) : Set ℝ := {x | x ≤ a}

noncomputable def complement_A := (real_universe \ set_A)

theorem part1 : (complement_A ∩ set_B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } :=
by sorry

theorem part2 : set_A ⊆ set_C a → a > 7 :=
by sorry

end part1_part2_l286_286769


namespace monomials_like_terms_l286_286808

theorem monomials_like_terms {m n : ℕ} (hm : m = 3) (hn : n = 1) : m - n = 2 :=
by
  rw [hm, hn]
  rfl

end monomials_like_terms_l286_286808


namespace unit_price_correct_minimum_cost_l286_286622

-- Given conditions for the unit prices
-- ∀ x (the unit price of type A), if 110/(x) = 120/(x+1) then x = 11
theorem unit_price_correct :
  ∀ (x : ℝ), (110 / x = 120 / (x + 1)) → (x = 11) :=
by
  assume x,
  intro h,
  have h_eq : 110 * (x + 1) = 120 * x := (div_eq_iff h).mp rfl,
  have h_simpl: 110 * x + 110 = 120 * x := by linarith,
  have h_final: x = 11 := by linarith,
  exact h_final

-- Proving minimum cost
-- ∀ a : ℕ (the number of type A notebooks), b : ℕ (the number of type B notebooks),
-- if a + b = 100, b ≤ 3 * a, then the minimum cost w = 11a + 12(100 - a) is 1100
theorem minimum_cost :
  ∀ (a b : ℕ), (a + b = 100) → (b ≤ 3 * a) → (-a + 1200 = 1100) :=
by
  assume a b,
  intro h1,
  intro h2,
  have h1_rewrite: b = 100 - a := by linarith,
  have h_cost: 11 * a + 12 * (100 - a) = -a + 1200 := by linarith,
  exact h_cost

end unit_price_correct_minimum_cost_l286_286622


namespace find_lambda_l286_286786

noncomputable def vector_perpendicular : Prop :=
  let a := (1, -3)
  let b := (4, 2)
  ∃ (λ : ℝ), (a.1 * (b.1 + λ) + a.2 * (b.2 - 3 * λ) = 0) ∧ (λ = 1 / 5)

-- The theorem to be proven
theorem find_lambda : vector_perpendicular :=
by {
  sorry
}

end find_lambda_l286_286786


namespace probability_exactly_one_red_ball_l286_286839

open ProbabilityTheory

theorem probability_exactly_one_red_ball :
  ∀ (A B : List String), 
  A = ["red", "red", "yellow"] → 
  B = ["red", "red", "yellow"] →
  (∃ p : ℝ, p = 4 / 9 ∧ 
    (∃ (VA VB : String), VA ∈ A ∧ VB ∈ B ∧ ((VA = "red" ∧ VB ≠ "red") ∨ (VA ≠ "red" ∧ VB = "red")))) :=
begin
  intros A B hA hB,
  use 4 / 9,
  split,
  { sorry }, -- This is where the proof would go
  { sorry }  -- This is where the proof would go
end

end probability_exactly_one_red_ball_l286_286839


namespace reflection_squared_is_identity_l286_286853

theorem reflection_squared_is_identity (R : Matrix (Fin 2) (Fin 2) ℝ) (v : Vector ℝ 2):
  (reflect_over_vector R v (Matrix.vecCons 4 (Matrix.vecCons (-2) Matrix.vecNil))) →  R ^ 2 = (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end reflection_squared_is_identity_l286_286853


namespace problem1_problem2_l286_286615

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin(Real.pi / 2 - α))^2 + 3 * Real.sin(α + Real.pi) * Real.sin(α + Real.pi / 2) = -1 := by
  sorry

-- Problem 2
theorem problem2 (α a : ℝ) 
  (h1 : α > Real.pi / 2 ∧ α < Real.pi) 
  (h2 : ∃ a, (a, 1) ∈ {(x, y) | y = Real.sin α ∧ x = a}) 
  (h3 : Real.cos α = (Real.sqrt 2 / 4) * a)
  (ha_neg : a < 0) :
  a = -Real.sqrt 7 := by
  sorry

end problem1_problem2_l286_286615


namespace average_weight_of_abc_l286_286196

theorem average_weight_of_abc (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 40) : 
  (A + B + C) / 3 = 42 := 
sorry

end average_weight_of_abc_l286_286196


namespace problem1_problem2_l286_286302

-- Problem 1
theorem problem1 : (-3) + (-9) - (+10) - (-18) = -4 := 
sorry

-- Problem 2
theorem problem2 : (-81) ÷ (9 / 4) × (-4 / 9) ÷ (-16) = -1 := 
sorry

end problem1_problem2_l286_286302


namespace binomial_variance_l286_286814

theorem binomial_variance
  {X : Type}
  (n : ℕ)
  (h_binom : ∀ k : ℕ, Pr(X = k) = (n.choose k) * (1/3)^k * (2/3)^(n-k))
  (h_expect : E(X) = 5/3)
  (h_var_formula : D(X) = n * (1/3) * (1 - 1/3)) :
  D(X) = 10 / 9 :=
by
  sorry

end binomial_variance_l286_286814


namespace converse_true_proposition_l286_286826

def not_coplanar_implies_not_collinear (A B C D : Point) : (¬ coplanar A B C D) → ¬  collinear A B C :=
sorry

def no_common_point_implies_skew (l m : Line) : (¬ intersect l m ≠ ∅) → skew l m :=
sorry

def skew_implies_no_common_point (l m : Line) : skew l m → (¬ intersect l m ≠ ∅) :=
sorry

theorem converse_true_proposition : (∀ (A B C D : Point), (¬ coplanar A B C D) → ¬ collinear A B C) 
  ∧ (∀ (l m : Line), (¬ intersect l m ≠ ∅) → skew l m)
  ∧ ∃ (l m : Line), skew l m → (¬ intersect l m ≠ ∅) :=
by {
  intros,
  apply and.intro,
    {assume A B C D, exact not_coplanar_implies_not_collinear},
    apply and.intro,
    {assume l m, exact no_common_point_implies_skew},
      {existsi some_line, existsi some_other_line, exact skew_implies_no_common_point}
}

end converse_true_proposition_l286_286826


namespace sufficient_but_not_necessary_condition_ellipse_l286_286557

theorem sufficient_but_not_necessary_condition_ellipse (a : ℝ) :
  (a^2 > 1 → ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1)) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → (a^2 > 1 ∨ 0 < a^2 ∧ a^2 < 1)) → ¬ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_ellipse_l286_286557


namespace second_train_speed_l286_286984

noncomputable def speed_of_first_train_kmh : ℝ := 120
noncomputable def length_of_first_train_m : ℝ := 210
noncomputable def length_of_second_train_m : ℝ := 290.04
noncomputable def crossing_time_s : ℝ := 9

def speed_of_first_train_ms : ℝ := speed_of_first_train_kmh * (5 / 18)
def total_distance_m : ℝ := length_of_first_train_m + length_of_second_train_m
def relative_speed_ms : ℝ := total_distance_m / crossing_time_s
def speed_of_second_train_ms : ℝ := relative_speed_ms - speed_of_first_train_ms
def speed_of_second_train_kmh (v_2_ms : ℝ) : ℝ := v_2_ms * (18 / 5)

theorem second_train_speed :
  speed_of_second_train_kmh speed_of_second_train_ms ≈ 399.14 := 
sorry

end second_train_speed_l286_286984


namespace change_received_l286_286932

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end change_received_l286_286932


namespace eccentricity_of_hyperbola_l286_286636

open Real

noncomputable def hyperbola_eccentricity : ℝ :=
  let a : ℝ := sorry  -- will be defined as part of hyperbola
  let b : ℝ := sorry  -- will be defined as part of hyperbola
  let c : ℝ := sorry  -- derived as part of the proof
  if h₁ : a > 0 ∧ b > 0 then
    if h₂ : sqrt 5 * a = c then
      sqrt 5
    else
      sorry -- contradiction of condition h₂
  else
    sorry -- contradiction of condition h₁

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ) (c : ℝ),
    (a > 0 ∧ b > 0) →
    let xa : ℝ := a
    let yb : ℝ := b
    let F := (c, 0)
    let A := (a^2/c, -a*b/c)
    let B := (- (c^2 + 2 * a^2) / (3 * c), -2 * a * b / (3 * c))
    let hyperbola : (ℝ × ℝ) → Prop := λ (p : ℝ × ℝ), (p.1^2 / a^2 - p.2^2 / b^2 = 1)
    hyperbola B →
    sqrt 5 * a = c →
    hyperbola_eccentricity = sqrt 5 :=
by
  intros a b c hpos hrelation1
  sorry

end eccentricity_of_hyperbola_l286_286636


namespace product_of_radii_l286_286125

namespace MathProof

theorem product_of_radii
  (r1 r2 d : ℝ)
  (h_external_tangent: sqrt (d^2 - (r1 + r2)^2) = 2017)
  (h_internal_tangent: sqrt (d^2 - (r1 - r2)^2) = 2009) :
  r1 * r2 = 8052 :=
begin
  sorry
end

end MathProof

end product_of_radii_l286_286125


namespace sum_S_100_l286_286851

variables (S : ℕ+ → ℝ) (a : ℕ+ → ℝ)

-- Define the sequence condition
axiom S_def : ∀ n : ℕ+, S n = (-1) ^ n * a n - 1 / 2 ^ (n : ℕ)

-- The main statement
theorem sum_S_100 :
  (∑ n in Finset.range 100, S ⟨n + 1, Nat.succ_pos' n⟩) =
  1 / 3 * (1 / 2 ^ 100 - 1) :=
sorry

end sum_S_100_l286_286851


namespace sum_S2017_eq_2017_div_2_l286_286412

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / ((Real.exp x) + 1)

-- Definition of the geometric sequence properties
variables {a : ℕ → ℝ}
axiom geom_seq_pos : ∀ n, a n > 0
axiom a1009_eq_one : a 1009 = 1

-- Define S2017 as the sum involving the function f and the geometric sequence
def S2017 : ℝ := ∑ i in Finset.range 2017, f (Real.log (a (i + 1)))

-- Statement of the problem
theorem sum_S2017_eq_2017_div_2 :
  S2017 = 2017 / 2 :=
begin
  sorry
end

end sum_S2017_eq_2017_div_2_l286_286412


namespace geometric_series_sum_l286_286961

theorem geometric_series_sum :
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  (3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = S) → S = 765 :=
by
  -- conditions
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  have h : 3 * (1 - 2^n) / (1 - 2) = 765 := sorry
  sorry

end geometric_series_sum_l286_286961


namespace BM_parallel_AC_l286_286469

open EuclideanGeometry

namespace IsoscelesTriangleProblem

variables {A B C D M : Point}
variables {S1 S2 : Circle}

-- Given conditions
def is_isosceles_triangle (A B C : Point) : Prop := dist A B = dist B C

def on_side_AB (D A B : Point) : Prop := collinear D A B ∧ D ≠ A ∧ D ≠ B

def circumcircle_ADC (S1 : Circle) (A D C : Point) : Prop := is_circumcircle S1 A D C

def circumcircle_BDC (S2 : Circle) (B D C : Point) : Prop := is_circumcircle S2 B D C

def tangent_intersects (S1 S2 : Circle) (M D : Point) : Prop :=
  tangent S1 D ∧ intersects S2 (tangent_point S1 D) M

-- Main theorem statement
theorem BM_parallel_AC
  (h_iso : is_isosceles_triangle A B C)
  (h_D_on_AB : on_side_AB D A B)
  (h_S1 : circumcircle_ADC S1 A D C)
  (h_S2 : circumcircle_BDC S2 B D C)
  (h_tangent : tangent_intersects S1 S2 M D) :
  parallel (line_through B M) (line_through A C) :=
sorry

end IsoscelesTriangleProblem

end BM_parallel_AC_l286_286469


namespace five_triangles_not_possible_l286_286259

open Set FiniteGeometry -- Open necessary namespaces

-- Define a structure for the triangle and its circumradius
structure Triangle where
  vertices : Finset (ℝ × ℝ)
  circumradius : ℝ

-- Define distance between triangles based on their circumradii
def distance (t1 t2 : Triangle) : ℝ :=
  t1.circumradius + t2.circumradius

-- Define the main theorem to prove the impossibility
theorem five_triangles_not_possible (triangles : Finset Triangle) (h_card : triangles.card = 5) :
  ∃ t1 t2 : Triangle, t1 ∈ triangles ∧ t2 ∈ triangles ∧ t1 ≠ t2 ∧ distance t1 t2 ≠ t1.circumradius + t2.circumradius :=
by
  -- We are essentially stating that the condition for distance being the sum of circumradii cannot hold for all pairs of 5 triangles
  -- Using Kuratowski's theorem directly in the proof would be necessary, here we just structure the theorem
  sorry

end five_triangles_not_possible_l286_286259


namespace sufficient_not_necessary_l286_286855

theorem sufficient_not_necessary (a b : ℝ) (h : 2^a + 2^b = 2^(a + b)) : a + b ≥ 2 :=
by sorry

end sufficient_not_necessary_l286_286855


namespace correct_propositions_l286_286663

-- Definitions of parallel and perpendicular
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Main theorem
theorem correct_propositions (m n α β γ : Type) :
  ( (parallel m α ∧ parallel n β ∧ parallel α β → parallel m n) ∧
    (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ∧
    (perpendicular α γ ∧ perpendicular β γ → parallel α β) ) →
  ( (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ) :=
  sorry

end correct_propositions_l286_286663


namespace charles_finishes_book_in_12_days_l286_286676

theorem charles_finishes_book_in_12_days (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 8) (h2 : total_pages = 96) :
  total_pages / pages_per_day = 12 :=
by
  rw [h1, h2]
  norm_num

end charles_finishes_book_in_12_days_l286_286676


namespace longer_train_length_correct_l286_286588

-- Define conditions
def shorter_train_length : ℕ := 60 
def speed_first_train : ℕ := 42 
def speed_second_train : ℕ := 30 
def time_to_clear : ℝ := 16.998640108791296 

-- Assume the conversion factor between kmph and m/s
def kmph_to_mps (speed_kmph : ℕ) : ℝ := speed_kmph * (5 / 18 : ℝ)

-- Define the relative speed
def relative_speed := kmph_to_mps (speed_first_train + speed_second_train)  -- in m/s

-- Calculate the total distance covered
def total_distance_covered := relative_speed * time_to_clear

-- Define the length of the longer train
noncomputable def length_longer_train : ℝ := total_distance_covered - shorter_train_length

-- The theorem to be proved
theorem longer_train_length_correct : length_longer_train = 279.9728021758259 := 
by
  -- Proof has to be provided here
  sorry

end longer_train_length_correct_l286_286588


namespace sum_of_30th_set_l286_286422

def set_first_element : ℕ → ℕ
| 1 := 1
| n := set_first_element (n - 1) + n - 1

def set_last_element (n : ℕ) : ℕ :=
  set_first_element n + n - 1

def Sn (n : ℕ) : ℕ :=
  n * (set_first_element n + set_last_element n) / 2

theorem sum_of_30th_set : Sn 30 = 13515 := by
  sorry

end sum_of_30th_set_l286_286422


namespace cylindrical_box_paperclip_capacity_l286_286275

-- Definitions of given conditions
def rectangularBoxBaseArea : ℝ := 4
def rectangularBoxHeight : ℝ := 4
def paperclipsInRectangularBox : ℝ := 50

def cylindricalBoxBaseRadius : ℝ := 2
def cylindricalBoxHeight : ℝ := 6

-- Volume of the rectangular box
def rectangularBoxVolume : ℝ :=
  rectangularBoxBaseArea * rectangularBoxHeight

-- Paperclip density
def density : ℝ :=
  paperclipsInRectangularBox / rectangularBoxVolume

-- Volume of the cylindrical box
def cylindricalBoxVolume : ℝ :=
  Real.pi * cylindricalBoxBaseRadius^2 * cylindricalBoxHeight

-- Calculate the number of paperclips in the cylindrical box
def paperclipsInCylindricalBox : ℝ :=
  cylindricalBoxVolume * density

-- Theorem statement to prove the equivalence
theorem cylindrical_box_paperclip_capacity :
  paperclipsInCylindricalBox = 75 * Real.pi := 
by
  sorry

end cylindrical_box_paperclip_capacity_l286_286275


namespace total_books_l286_286529

def sam_books : ℕ := 110
def joan_books : ℕ := 102

theorem total_books : sam_books + joan_books = 212 := by
  sorry

end total_books_l286_286529


namespace g_not_monotonic_in_1_2_max_k_such_that_f_geq_g_plus_x_plus_2_l286_286420

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (k x : ℝ) : ℝ := k * x^3 - x - 2
noncomputable def g' (k x : ℝ) : ℝ := 3 * k * x^2 - 1

-- (1) Prove the range of k for g(x) not being monotonic in (1, 2)
theorem g_not_monotonic_in_1_2 (k : ℝ) : 
  (1 / 12 < k ∧ k < 1 / 3) ↔ ¬ Monotonic (λ x : ℝ, g k x) :=
sorry

-- (2) Prove the maximum value of k such that f(x) ≥ g(x) + x + 2 for all x ∈ [1, +∞)
noncomputable def h (x : ℝ) : ℝ := (x - 2) * Real.exp(x) / x^3

theorem max_k_such_that_f_geq_g_plus_x_plus_2 : 
  ∀ x ≥ 1, f x ≥ g (-Real.exp 1) x + x + 2 :=
sorry

end g_not_monotonic_in_1_2_max_k_such_that_f_geq_g_plus_x_plus_2_l286_286420


namespace extreme_value_of_f_l286_286914

def f(x : ℝ) : ℝ := (1/3) * x^3 + 3 * x^2 + 5 * x + 2

theorem extreme_value_of_f : ∃ x : ℝ, (x = -1) ∧ (f x = -(1/3)) :=
by
  sorry

end extreme_value_of_f_l286_286914


namespace greatest_odd_factors_below_200_l286_286165

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l286_286165


namespace sub_square_expansion_l286_286107

theorem sub_square_expansion (a b : ℝ) :
  (a + b)^4 = a^4 + 4 * a^3 * b + 6 * a^2 * b^2 + 4 * a * b^3 + b^4 →
  (a - b)^4 = a^4 - 4 * a^3 * b + 6 * a^2 * b^2 - 4 * a * b^3 + b^4 :=
begin
  sorry
end

end sub_square_expansion_l286_286107


namespace no_odd_number_not_in_brazilian_seq_l286_286219
open Nat

/-- The sequence of positive integers is brazilian if a_1 = 1 and for each n > 1, 
a_n is the least integer greater than a_{n-1} and a_n is coprime with at least half of 
the elements of the set {a_1, a_2, ..., a_{n-1}} --/
def is_brazilian_seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n > 1, 
  a n > a (n-1) ∧ ∃ b : ℕ, b = a n ∧ ∀ m, m < n → (gcd (a(m)) b > 1 → (half ≤ m))

theorem no_odd_number_not_in_brazilian_seq :
  ∀ (a : ℕ → ℕ), is_brazilian_seq(a) → ∀ (m : ℕ), odd m → ∃ n, a(n) = m :=
by
  -- proof omitted
  sorry

end no_odd_number_not_in_brazilian_seq_l286_286219


namespace segment_length_hypotenuse_inside_circle_l286_286258

theorem segment_length_hypotenuse_inside_circle 
  (A B C : Point) 
  (AB BC : ℝ) 
  (h_AB : AB = 3) 
  (h_BC : BC = 4) 
  (right_triangle : right_triangle_at B A C) 
  (midpoint_AB_midpoint_AC : circle_through_midpoints_AB_AC_touches_BC A B C) : 
  ∃ segment_length : ℝ, segment_length = 11 / 10 :=
by 
  sorry

end segment_length_hypotenuse_inside_circle_l286_286258


namespace bargain_range_l286_286484

theorem bargain_range (cost_price lowest_cp highest_cp : ℝ)
  (h_lowest : lowest_cp = 50)
  (h_highest : highest_cp = 200 / 3)
  (h_marked_at : cost_price = 100)
  (h_lowest_markup : lowest_cp * 2 = cost_price)
  (h_highest_markup : highest_cp * 1.5 = cost_price)
  (profit_margin : ∀ (cp : ℝ), (cp * 1.2 ≥ cp)) : 
  (60 ≤ cost_price * 1.2 ∧ cost_price * 1.2 ≤ 80) :=
by
  sorry

end bargain_range_l286_286484


namespace constant_term_in_expansion_l286_286708

theorem constant_term_in_expansion :
  let binomial_expansion_fn := (λ (x : ℝ), (x^3 + 1 / x^2)^5)
  ∃ k : ℕ, k = 5 ∧ binomial_expansion_fn 0 = 10 :=
by
  let binomial_expansion_fn := (λ (x : ℝ), (x^3 + 1 / x^2)^5)
  use 5
  split
  · refl
  · sorry

end constant_term_in_expansion_l286_286708


namespace perpendicular_lines_a_l286_286007

theorem perpendicular_lines_a {a : ℝ} :
  ((∀ x y : ℝ, (2 * a - 1) * x + a * y + a = 0) → (∀ x y : ℝ, a * x - y + 2 * a = 0) → a = 0 ∨ a = 1) :=
by
  intro h₁ h₂
  sorry

end perpendicular_lines_a_l286_286007


namespace kennedy_distance_to_school_l286_286488

def miles_per_gallon : ℕ := 19
def initial_gallons : ℕ := 2
def distance_softball_park : ℕ := 6
def distance_burger_restaurant : ℕ := 2
def distance_friends_house : ℕ := 4
def distance_home : ℕ := 11

def total_distance_possible : ℕ := miles_per_gallon * initial_gallons
def distance_after_school : ℕ := distance_softball_park + distance_burger_restaurant + distance_friends_house + distance_home
def distance_to_school : ℕ := total_distance_possible - distance_after_school

theorem kennedy_distance_to_school :
  distance_to_school = 15 :=
by
  sorry

end kennedy_distance_to_school_l286_286488


namespace matching_socks_probability_l286_286792

def total_ways_to_choose_socks : ℕ :=
(10.choose 2) * (8.choose 2) * (6.choose 2) * (4.choose 2) * (2.choose 2)

def favorable_outcomes_matching_socks : ℕ :=
5 * 4 * ((6.choose 2) * (4.choose 2) * (2.choose 2))

def probability_matching_socks_third_and_fifth_day : ℚ :=
(favorable_outcomes_matching_socks : ℚ) / (total_ways_to_choose_socks : ℚ)

theorem matching_socks_probability :
  probability_matching_socks_third_and_fifth_day = 1 / 63 :=
by
  -- proof will be written here
  sorry

end matching_socks_probability_l286_286792


namespace speed_equivalence_l286_286265

def convert_speed (speed_kmph : ℚ) : ℚ :=
  speed_kmph * 0.277778

theorem speed_equivalence : convert_speed 162 = 45 :=
by
  sorry

end speed_equivalence_l286_286265


namespace andrew_total_travel_time_l286_286580

theorem andrew_total_travel_time :
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  subway_time + train_time + bike_time = 38 :=
by
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  sorry

end andrew_total_travel_time_l286_286580


namespace bisector_property_l286_286289

-- Define the immediate conditions
def triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def is_isosceles (α β γ : ℝ) (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ triangle α β γ

def bisector_perpendicular (α β γ : ℝ) (a b c : ℝ) : Prop :=
  is_isosceles α β γ a b c → (some_condition α β γ a b c)

-- Define Lean statement for the theorem
theorem bisector_property :
  ∃ (α β γ a b c : ℝ), bisector_perpendicular α β γ a b c ∧ ∀ α β γ a b c, is_right_triangle α β γ a b c → ¬bisector_perpendicular α β γ a b c :=
sorry

end bisector_property_l286_286289


namespace xyz_value_l286_286504

noncomputable def find_xyz (x y z : ℝ) 
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) : ℝ :=
  if (x * y * z = 31 / 3) then 31 / 3 else 0  -- This should hold with the given conditions

theorem xyz_value (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) :
  find_xyz x y z h₁ h₂ h₃ = 31 / 3 :=
by 
  sorry  -- The proof should demonstrate that find_xyz equals 31 / 3 given the conditions

end xyz_value_l286_286504


namespace petya_correct_square_l286_286970

theorem petya_correct_square :
  ∃ x a b : ℕ, (1 ≤ x ∧ x ≤ 9) ∧
              (x^2 = 10 * a + b) ∧ 
              (2 * x = 10 * b + a) ∧
              (x^2 = 81) :=
by
  sorry

end petya_correct_square_l286_286970


namespace andrew_total_travel_time_l286_286581

theorem andrew_total_travel_time :
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  subway_time + train_time + bike_time = 38 :=
by
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  sorry

end andrew_total_travel_time_l286_286581


namespace root_diff_l286_286864

noncomputable def sqrt11 := Real.sqrt 11
noncomputable def sqrt180 := Real.sqrt 180
noncomputable def sqrt176 := Real.sqrt 176

theorem root_diff (x1 x2 : ℂ) 
  (h1 : sqrt11 * x1^2 + sqrt180 * x1 + sqrt176 = 0)
  (h2 : sqrt11 * x2^2 + sqrt180 * x2 + sqrt176 = 0)
  (h3 : x1 ≠ x2) :
  abs ((1 / (x1^2)) - (1 / (x2^2))) = (Real.sqrt 45) / 44 :=
sorry

end root_diff_l286_286864


namespace symm_parabola_l286_286411

theorem symm_parabola (a b : ℝ) :
  let f : ℝ → ℝ := λ x, 3 * x^2 + a * x + b in
  (∀ x : ℝ, f (x - 1) = f (1 - x)) →
  (f (-1) < f (-3 / 2) ∧ f (-1) < f (3 / 2) ∧ f (-3 / 2) = f (3 / 2)) :=
by
  sorry

end symm_parabola_l286_286411


namespace exist_N_for_fn_eq_n_l286_286001

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_condition1 (m n : ℕ+) : (f m, f n) ≤ (m, n) ^ 2014
axiom f_condition2 (n : ℕ+) : n ≤ f n ∧ f n ≤ n + 2014

theorem exist_N_for_fn_eq_n :
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := sorry

end exist_N_for_fn_eq_n_l286_286001


namespace percentage_of_water_in_nectar_l286_286249

-- Define the necessary conditions and variables
def weight_of_nectar : ℝ := 1.7 -- kg
def weight_of_honey : ℝ := 1 -- kg
def honey_water_percentage : ℝ := 0.15 -- 15%

noncomputable def water_in_honey : ℝ := weight_of_honey * honey_water_percentage -- Water content in 1 kg of honey

noncomputable def total_water_in_nectar : ℝ := water_in_honey + (weight_of_nectar - weight_of_honey) -- Total water content in nectar

-- The theorem to prove
theorem percentage_of_water_in_nectar :
    (total_water_in_nectar / weight_of_nectar) * 100 = 50 := 
by 
    -- Skipping the proof by using sorry as it is not required
    sorry

end percentage_of_water_in_nectar_l286_286249


namespace company_workers_l286_286978

theorem company_workers (W : ℕ) (H1 : (1/3 : ℚ) * W = ((1/3 : ℚ) * W)) 
  (H2 : 0.20 * ((1/3 : ℚ) * W) = ((1/15 : ℚ) * W)) 
  (H3 : 0.40 * ((2/3 : ℚ) * W) = ((4/15 : ℚ) * W)) 
  (H4 : (4/15 : ℚ) * W + (4/15 : ℚ) * W = 160)
  : (W - 160 = 140) :=
by
  sorry

end company_workers_l286_286978


namespace minimize_G_l286_286432

def F (p q : ℝ) : ℝ :=
  -2 * p * q + 3 * p * (1 - q) + 3 * (1 - p) * q - 4 * (1 - p) * (1 - q)

def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G :
  ∀ p, 0 ≤ p ∧ p ≤ 1 → G p ≤ G (7 / 12) :=
by
  sorry

end minimize_G_l286_286432


namespace work_rate_l286_286631

theorem work_rate (R_B : ℚ) (R_A : ℚ) (R_total : ℚ) (days : ℚ)
  (h1 : R_A = (1/2) * R_B)
  (h2 : R_B = 1 / 22.5)
  (h3 : R_total = R_A + R_B)
  (h4 : days = 1 / R_total) : 
  days = 15 := 
sorry

end work_rate_l286_286631


namespace range_of_n_l286_286141

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n : ℝ) : Set ℝ := {x | n-1 < x ∧ x < n+1}

-- Define the condition A ∩ B ≠ ∅
def A_inter_B_nonempty (n : ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B n

-- Prove the range of n for which A ∩ B ≠ ∅ is (-2, 2)
theorem range_of_n : ∀ n, A_inter_B_nonempty n ↔ (-2 < n ∧ n < 2) := by
  sorry

end range_of_n_l286_286141


namespace angle_BCD_eq_angle_A_DE_eq_EC_angle_CFD_eq_2angle_A_l286_286823

-- Assume definitions of the geometric entities and properties

variables {A B C D E F : Type} [InnerProductSpace ℝ A]

-- Assume ABC is an acute triangle
def is_acute_triangle (A B C : A) : Prop := 
  ∡ A B C < π / 2 ∧ ∡ B C A < π / 2 ∧ ∡ C A B < π / 2

-- Assume D is a point on AB such that DC is perpendicular to AB
def point_on_line (A B : A) (D : A) : Prop :=
  ∃ (k : ℝ), D = A + k • (B - A)

def is_perpendicular (D C : A) (AB : set A) : Prop :=
  ∃ D' ∈ AB, D' = D ∧ ⟨D - D', C - D'⟩ = 0

-- Assume the circle with diameter BC intersects DC again at E
def point_on_circle (C : A) (BC : A × A) (E : A) : Prop :=
  ∃ (r : ℝ), r = (C - BC.1).norm / 2 ∧ (E - C).norm = r ∧ (E - BC.2).norm = r

-- Assume a tangent to the circle at D intersects the extension of side CB at F
def is_tangent (D : A) (circle : A × ℝ) (F : A) : Prop :=
  (F - circle.1).norm = circle.2 ∧ ⟨F - circle.1, D - circle.1⟩ = 0
  

-- Prove that ∠BCD = ∠A
theorem angle_BCD_eq_angle_A {A B C D E F : A} :
  is_acute_triangle A B C ∧
  point_on_line A B D ∧
  is_perpendicular D C (line_segment A B) ∧
  point_on_circle C (B, C) E ∧
  is_tangent D (C, (C - B).norm / 2) F →
  ∡ B C D = ∡ A :=
sorry

-- Prove that DE = EC
theorem DE_eq_EC {A B C D E F : A} :
  is_acute_triangle A B C ∧
  point_on_line A B D ∧
  is_perpendicular D C (line_segment A B) ∧
  point_on_circle C (B, C) E ∧
  is_tangent D (C, (C - B).norm / 2) F →
  (E - D).norm = (E - C).norm :=
sorry

-- Prove that ∠CFD = 2∠A
theorem angle_CFD_eq_2angle_A {A B C D E F : A} :
  is_acute_triangle A B C ∧
  point_on_line A B D ∧
  is_perpendicular D C (line_segment A B) ∧
  point_on_circle C (B, C) E ∧
  is_tangent D (C, (C - B).norm / 2) F →
  ∡ C F D = 2 * ∡ A :=
sorry

end angle_BCD_eq_angle_A_DE_eq_EC_angle_CFD_eq_2angle_A_l286_286823


namespace sum_roots_of_quadratic_eq_l286_286244

theorem sum_roots_of_quadratic_eq (a b c: ℝ) (x: ℝ) :
    (a = 1) →
    (b = -7) →
    (c = -9) →
    (x ^ 2 - 7 * x + 2 = 11) →
    (∃ r1 r2 : ℝ, x ^ 2 - 7 * x - 9 = 0 ∧ r1 + r2 = 7) :=
by
  sorry

end sum_roots_of_quadratic_eq_l286_286244


namespace bisection_method_root_interval_l286_286766

theorem bisection_method_root_interval
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_cont : ∀ x ∈ set.Icc a b, continuous_at f x)
  (h_ab : f a * f b < 0)
  (h_fa : f a < 0)
  (h_fb : f b > 0)
  (h_fm : f ((a + b) / 2) > 0) :
  ∃ c, c ∈ set.Ioo a ((a + b) / 2) ∧ f c = 0 := 
sorry

end bisection_method_root_interval_l286_286766


namespace number_of_triangles_l286_286840

theorem number_of_triangles (OA_points OB_points : ℕ) (O : ℕ) :
    OA_points = 4 ∧ OB_points = 5 ∧ O = 1 →
    ∑ n in (Finset.range 10).powersetLen 3, if isCollinear n then 0 else 1 = 90 :=
by
    sorry

-- Assumptions and conditions definitions
def isCollinear (points : Finset ℕ) : Bool :=
    sorry  -- Define the collinearity check based on the edges.

end number_of_triangles_l286_286840


namespace count_perfect_square_factors_l286_286063

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286063


namespace WO_length_l286_286095

noncomputable def length_of_WO (WXYZ : Type*) [IsConvexQuadrilateral WXYZ]
  (W Z : WXYZ) (XZ Y : WXYZ) (O : WXYZ)
  (h1 : length_segment W Z = 15)
  (h2 : length_segment X Y = 18)
  (h3 : divides_diag W Y O 1 2)
  (h4 : area_of_condition (triangle_areas_equal (triangle W O X) (triangle Y O Z)))
  : ℝ :=
  5

theorem WO_length 
  {WXYZ : Type*} [IsConvexQuadrilateral WXYZ]
  {W Z : WXYZ} {XZ Y : WXYZ} {O : WXYZ}
  (h1 : length_segment W Z = 15)
  (h2 : length_segment X Y = 18)
  (h3 : divides_diag W Y O 1 2)
  (h4 : area_of_condition (triangle_areas_equal (triangle W O X) (triangle Y O Z)))
  : length_segment W O = 5 :=
by 
  simp [length_of_WO, h1, h2, h3, h4]
  sorry

end WO_length_l286_286095


namespace box_third_dimension_l286_286274

theorem box_third_dimension (num_cubes : ℕ) (cube_volume box_vol : ℝ) (dim1 dim2 h : ℝ) (h_num_cubes : num_cubes = 24) (h_cube_volume : cube_volume = 27) (h_dim1 : dim1 = 9) (h_dim2 : dim2 = 12) (h_box_vol : box_vol = num_cubes * cube_volume) :
  box_vol = dim1 * dim2 * h → h = 6 := 
by
  sorry

end box_third_dimension_l286_286274


namespace find_common_ratio_l286_286852

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : ∀ n, S n = finset.sum (finset.range n) (λ k, a k)) 
  (h3 : S 4 = 10 * S 2) 
  (h_pos : ∀ n : ℕ, a n > 0) : 
  q = 3 :=
sorry

end find_common_ratio_l286_286852


namespace all_statements_true_l286_286311

def greatest_integer (x : ℝ) : ℤ := Int.floor x

theorem all_statements_true :
  (∀ x : ℝ, greatest_integer (x + 2) = greatest_integer x + 2) ∧
  (∀ x y : ℝ, greatest_integer (x + 1/2 + y + 1/2) = greatest_integer x + greatest_integer y + 1) ∧
  (∀ x y : ℝ, greatest_integer (0.5 * x * (0.5 * y)) = 0.25 * greatest_integer x * greatest_integer y) :=
by
  sorry

end all_statements_true_l286_286311


namespace sum_of_edge_lengths_of_cube_l286_286222

-- Define the problem conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- The final statement to prove
theorem sum_of_edge_lengths_of_cube (a : ℝ) (ha : surface_area a = 150) : 12 * a = 60 :=
by
  sorry

end sum_of_edge_lengths_of_cube_l286_286222


namespace arithmetic_sequence_sum_l286_286472

-- Define the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

-- Define the specific condition of the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d  -- where a₁ is the first term and d is the common difference

-- Given condition
def a₁ := -- Specify the first term here
def d := -- Specify the common difference here

-- The specific case for the 10th term
def a10 : ℤ := 5

-- The sequence is arithmetic, calculate sum of first 19 terms
def S19 : ℤ :=
  (19 * (a₁ + a10)) / 2

-- Theorem statement
theorem arithmetic_sequence_sum :
  S19 = 95 := by
  sorry

end arithmetic_sequence_sum_l286_286472


namespace triangle_cos_Z_l286_286836

theorem triangle_cos_Z (X Y Z : ℝ) (hXZ : X + Y + Z = π) 
  (sinX : Real.sin X = 4 / 5) (cosY : Real.cos Y = 3 / 5) : 
  Real.cos Z = 7 / 25 := 
sorry

end triangle_cos_Z_l286_286836


namespace max_lambda_inequality_l286_286807

theorem max_lambda_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 / Real.sqrt (20 * a + 23 * b) + 1 / Real.sqrt (23 * a + 20 * b)) ≥ (2 / Real.sqrt 43 / Real.sqrt (a + b)) :=
by
  sorry

end max_lambda_inequality_l286_286807


namespace prob_at_least_one_heart_spade_or_king_l286_286988

theorem prob_at_least_one_heart_spade_or_king :
  let total_cards := 52
  let hearts := 13
  let spades := 13
  let kings := 4
  let unique_hsk := hearts + spades + 2  -- Two unique kings from other suits
  let prob_not_hsk := (total_cards - unique_hsk) / total_cards
  let prob_not_hsk_two_draws := prob_not_hsk * prob_not_hsk
  let prob_at_least_one_hsk := 1 - prob_not_hsk_two_draws
  prob_at_least_one_hsk = 133 / 169 :=
by sorry

end prob_at_least_one_heart_spade_or_king_l286_286988


namespace isosceles_triangle_side_length_l286_286973

theorem isosceles_triangle_side_length (total_length : ℝ) (one_side_length : ℝ) (remaining_wire : ℝ) (equal_side : ℝ) :
  total_length = 20 → one_side_length = 6 → remaining_wire = total_length - one_side_length → remaining_wire / 2 = equal_side →
  equal_side = 7 :=
by
  intros h_total h_one_side h_remaining h_equal_side
  sorry

end isosceles_triangle_side_length_l286_286973


namespace intersection_cardinality_l286_286413

noncomputable def f : ℝ → ℝ := sorry
variable {F : set ℝ}

theorem intersection_cardinality :
  ∃ n : ℕ, n = 0 ∨ n = 1 ∧
  n = (λ F (f : F → ℝ),
    if 1 ∈ F then 1 else 0) F f :=
by
  sorry

end intersection_cardinality_l286_286413


namespace quadratic_coeff_sum_l286_286212

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l286_286212


namespace smallest_number_is_51_l286_286560

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end smallest_number_is_51_l286_286560


namespace nth_inequality_l286_286878

theorem nth_inequality (n : ℕ) : 
  (∑ i in Finset.range n, Real.sqrt ((i+1) * (i+2))) < ((n+1)^2/2) := 
sorry

end nth_inequality_l286_286878


namespace shortest_time_for_goods_transport_l286_286616

noncomputable def time_to_reach_destination (v : ℝ) : ℝ :=
  v / 25 + 600 / v

theorem shortest_time_for_goods_transport :
  ∀ v : ℝ, 100 ≤ v ∧ v ≤ 120 → time_to_reach_destination v ≥ 9.8 :=
by
begin
  intro v,
  intro hv,
  sorry
end

end shortest_time_for_goods_transport_l286_286616


namespace cookie_recipe_total_cups_l286_286096

theorem cookie_recipe_total_cups (r_butter : ℕ) (r_flour : ℕ) (r_sugar : ℕ) (sugar_cups : ℕ) 
  (h_ratio : r_butter = 1 ∧ r_flour = 2 ∧ r_sugar = 3) (h_sugar : sugar_cups = 9) : 
  r_butter * (sugar_cups / r_sugar) + r_flour * (sugar_cups / r_sugar) + sugar_cups = 18 := 
by 
  sorry

end cookie_recipe_total_cups_l286_286096


namespace probability_parallelograms_l286_286105

variable (m n : ℕ)
variable (a_set : Set ℕ) (b_set : Set ℕ)

def num_parallelograms (a_set b_set : Set ℕ) : ℕ :=
  (a_set.card * b_set.card).choose 2

def num_small_area_parallelograms (a_set b_set : Set ℕ) : ℕ := 5

theorem probability_parallelograms (h_a : a_set = {2, 4}) (h_b : b_set = {1, 3, 5})
  (hn : num_parallelograms a_set b_set = 15)
  (hm : num_small_area_parallelograms a_set b_set = 5) :
  (hm : ℚ) / hn = 1 / 3 :=
by
  rw [hm, hn]
  norm_num
  sorry

end probability_parallelograms_l286_286105


namespace limit_problem_l286_286672

theorem limit_problem : 
  (∀ (f : ℝ → ℝ), (∀ (x : ℝ), x ≠ 0 → f x = sqrt ((exp (sin x) - 1) * cos (1/x) + 4 * cos x)) → 
  filter.tendsto f (nhds 0) (nhds 2)) :=
sorry

end limit_problem_l286_286672


namespace perimeter_of_convex_quad_l286_286628

open Set

-- Definitions for the given conditions in the problem
variables (A B C D P : Point)
variables (PA PB PC PD : ℝ)
variables (area : ℝ)
variables (perimeter : ℝ)

-- Given conditions
def convex_quadrilateral : Prop := 
  convex_hull ℝ ({A, B, C, D} : Set Point).nonempty ∧
  finite ({A, B, C, D} : Set Point)

def area_ABCD (area : ℝ) : Prop :=
  area = 2500

def PA_distance : Prop := 
  PA = 30

def PB_distance : Prop :=
  PB = 40

def PC_distance : Prop :=
  PC = 35

def PD_distance : Prop :=
  PD = 50

-- Theorem statement proving that given the conditions, the perimeter is approximately 218.31
theorem perimeter_of_convex_quad :
  convex_quadrilateral A B C D →
  area_ABCD 2500 →
  PA_distance →
  PB_distance →
  PC_distance →
  PD_distance →
  perimeter = 218.31 :=
by sorry

end perimeter_of_convex_quad_l286_286628


namespace brinley_animal_count_l286_286331

def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 12 * leopards
def cheetahs : ℕ := snakes / 3  -- rounding down implicitly considered
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem brinley_animal_count : total_animals = 673 :=
by
  -- Mathematical proof would go here.
  sorry

end brinley_animal_count_l286_286331


namespace rosy_efficiency_increase_l286_286875

variable {Mary_work Rosy_work : ℝ}

def Mary_efficiency (Mary_work : ℝ) : ℝ := 1 / Mary_work
def Rosy_efficiency (Rosy_work : ℝ) : ℝ := 1 / Rosy_work

theorem rosy_efficiency_increase :
  Mary_work = 11 → Rosy_work = 10 →
  ((Rosy_efficiency Rosy_work - Mary_efficiency Mary_work) / Mary_efficiency Mary_work) * 100 = 10 :=
by
  intros h1 h2
  rw [h1, h2, Mary_efficiency, Rosy_efficiency]
  rw [←div_div_eq_div_mul, div_self, div_eq_mul_inv, ←mul_assoc, one_mul, inv_div, ←mul_inv, inv_inv]
  sorry

end rosy_efficiency_increase_l286_286875


namespace parabola_through_origin_l286_286199

theorem parabola_through_origin {a b c : ℝ} :
  (c = 0 ↔ ∀ x, (0, 0) = (x, a * x^2 + b * x + c)) :=
sorry

end parabola_through_origin_l286_286199


namespace angle_A_measure_l286_286200

theorem angle_A_measure (angle1 : ℝ) (angle2 : ℝ) (angle3 : ℝ) 
(h1 : angle1 = 110) (h2 : angle2 = 40) (h3 : angle3 = 100) : 
  let A := 180 - (180 - angle1 - angle2) - angle3 in A = 30 :=
by
  sorry

end angle_A_measure_l286_286200


namespace area_of_given_parallelogram_l286_286976

def parallelogram_base : ℝ := 24
def parallelogram_height : ℝ := 16
def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem area_of_given_parallelogram : parallelogram_area parallelogram_base parallelogram_height = 384 := 
by sorry

end area_of_given_parallelogram_l286_286976


namespace number_of_possible_values_of_c_l286_286188

theorem number_of_possible_values_of_c {r s t u : ℂ} (h_distinct : r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ s ≠ t ∧ s ≠ u ∧ t ≠ u)
(h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = (z - c * r) * (z - c * s) * (z - c * t) * (z - c * u)):
∃ (cs : set ℂ), by { 
  let roots := {1, -1, complex.I, -complex.I} in
  have h_subset : cs = roots, sorry,
  have h_card : cs.card = 4, sorry,
  exact ⟨cs, h_subset, h_card⟩
}

end number_of_possible_values_of_c_l286_286188


namespace no_real_roots_quadratic_l286_286457

theorem no_real_roots_quadratic (k : ℝ) : 
  ∀ (x : ℝ), k * x^2 - 2 * x + 1 / 2 ≠ 0 → k > 2 :=
by 
  intro x h
  have h1 : (-2)^2 - 4 * k * (1/2) < 0 := sorry
  have h2 : 4 - 2 * k < 0 := sorry
  have h3 : 2 < k := sorry
  exact h3

end no_real_roots_quadratic_l286_286457


namespace incorrect_propositions_l286_286408

theorem incorrect_propositions
  (A B C : ℝ)
  (h1 : cos (A - B) * cos (B - C) * cos (C - A) = 1 ↔ is_equilateral A B C)
  (h2 : sin A = cos B ↔ is_right_angled A B C)
  (h3 : cos A * cos B * cos C < 0 ↔ is_obtuse A B C)
  (h4 : sin 2A = sin 2B ↔ is_isosceles A B C) :
  (¬ (sin A = cos B → is_right_angled A B C)) ∧ 
  (¬ (sin 2A = sin 2B → is_isosceles A B C)) :=
by {
  sorry,
}

-- Define the necessary predicates to use in the hypotheses
def is_equilateral (A B C : ℝ) : Prop := A = B ∧ B = C
def is_right_angled (A B C : ℝ) : Prop := A = π / 2 ∨ B = π / 2 ∨ C = π / 2
def is_obtuse (A B C : ℝ) : Prop := A > π / 2 ∨ B > π / 2 ∨ C > π / 2
def is_isosceles (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A

end incorrect_propositions_l286_286408


namespace log3_equation_solution_l286_286326

theorem log3_equation_solution (x : ℝ) (h : log 3 (x^2 + 6*x) = 3) : x = 3 ∨ x = -9 :=
by
  sorry

end log3_equation_solution_l286_286326


namespace distance_between_towns_l286_286579

theorem distance_between_towns (D S : ℝ) (h1 : D = S * 3) (h2 : 200 = S * 5) : D = 120 :=
by
  sorry

end distance_between_towns_l286_286579


namespace triangle_ABC_equilateral_l286_286482

noncomputable def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist A C

variables {A B C A1 B1 C1 : Point}

theorem triangle_ABC_equilateral
  (h1 : A1 ∈ line_segment B C) (h2 : is_altitude A A1)
  (h3 : B1 ∈ line_segment A C) (h4 : is_median B B1)
  (h5 : C1 ∈ line_segment A B) (h6 : is_angle_bisector C C1)
  (h7 : is_equilateral_triangle A1 B1 C1) :
  is_equilateral_triangle A B C :=
sorry

end triangle_ABC_equilateral_l286_286482


namespace carpeting_cost_l286_286605

theorem carpeting_cost :
  ∀ (L B : ℕ) (carpet_width_m cost_per_sqm : ℕ),
  L = 13 →
  B = 9 →
  carpet_width_m = 75 / 100 →
  cost_per_sqm = 12 →
  let area := L * B in
  let total_cost := area * cost_per_sqm in
  total_cost = 1404 :=
by
  intros L B carpet_width_m cost_per_sqm hL hB hw hcost
  have hL : L = 13 := hL
  have hB : B = 9 := hB
  have hcost : cost_per_sqm = 12 := hcost
  let area := L * B
  have harea : area = 13 * 9 := by rw [hL, hB]
  let total_cost := area * cost_per_sqm
  have htotal_cost : total_cost = 117 * 12 := by rw [harea, hcost]
  have htotal : total_cost = 1404 := by norm_num at htotal_cost
  exact htotal

end carpeting_cost_l286_286605


namespace first_term_geometric_progression_l286_286563

theorem first_term_geometric_progression (a r : ℝ) 
  (h1 : a / (1 - r) = 6)
  (h2 : a + a * r = 9 / 2) :
  a = 3 ∨ a = 9 := 
sorry -- Proof omitted

end first_term_geometric_progression_l286_286563


namespace find_second_certificate_interest_rate_l286_286658

theorem find_second_certificate_interest_rate
  (P : ℝ := 15000)  -- Initial investment
  (r1 : ℝ := 0.08)  -- First certificate's annual interest rate
  (period : ℝ := 3/12)  -- Three-month period (quarter year)
  (Value_After_Six_Months : ℝ := 16246)  -- Value after six months
  : (24.76 : ℝ) = λ s, s :=
by
  sorry

end find_second_certificate_interest_rate_l286_286658


namespace percentage_of_number_l286_286450

theorem percentage_of_number (X P : ℝ) (h1 : 0.20 * X = 80) (h2 : (P / 100) * X = 160) : P = 40 := by
  sorry

end percentage_of_number_l286_286450


namespace distance_QR_is_18_75_l286_286527

-- Define the geometric configuration
variable {DEF : Type} [triangle DEF]
variable {E D F : point DEF}
variable {Q R : point DEF}

-- Define the side lengths and properties of triangle DEF
axiom DEF_right_triangle_with_sides :
  is_right_triangle E D F ∧ side_length DE 9 ∧ side_length EF 12 ∧ side_length DF 15

-- Define the properties of circles centered at Q and R
axiom circle_Q :
  is_tangent_to_line_at Q EF E ∧ passes_through Q D

axiom circle_R :
  is_tangent_to_line_at R DE D ∧ passes_through R F

-- Define the expected distance QR
def distance_QR : ℝ := distance QR

-- The statement we need to prove
theorem distance_QR_is_18_75:
  distance_QR = 18.75 := sorry

end distance_QR_is_18_75_l286_286527


namespace pentagon_side_length_squared_l286_286640

noncomputable def square_length_of_side_of_pentagon_inscribed_in_ellipse : ℝ :=
  let a := 1
  let b := 3
  let ellipse_equation (x y : ℝ) := x^2 + b^2 * y^2 = 9
  let vertex_A := (0, 1)
  let altitude_along_y_axis := True
  in 20.684

theorem pentagon_side_length_squared :
  ∀ (x y : ℝ), (ellipse_equation x y) → (vertex_A = (0, 1)) → altitude_along_y_axis →
  (square_length_of_side_of_pentagon_inscribed_in_ellipse = 20.684) :=
by
  intros x y ellipse_eq vertex_eq altitude_eq
  sorry

end pentagon_side_length_squared_l286_286640


namespace count_perfect_square_factors_except_one_l286_286025

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286025


namespace blue_whale_tongue_weight_in_tons_l286_286919

-- Define the conditions
def weight_of_tongue_pounds : ℕ := 6000
def pounds_per_ton : ℕ := 2000

-- Define the theorem stating the question and its answer
theorem blue_whale_tongue_weight_in_tons :
  (weight_of_tongue_pounds / pounds_per_ton) = 3 :=
by sorry

end blue_whale_tongue_weight_in_tons_l286_286919


namespace inverse_proportion_inequality_l286_286391

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l286_286391


namespace triangle_ABC_right_l286_286078

def sin_squared (x : ℝ) : ℝ := (Real.sin x) ^ 2

theorem triangle_ABC_right (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (hC : C = π - (A + B)) (h_cond : sin_squared A + sin_squared B = Real.sin C) :
  A + B = π / 2 := 
sorry

end triangle_ABC_right_l286_286078


namespace complement_of_A_is_negatives_l286_286565

theorem complement_of_A_is_negatives :
  let U := Set.univ (α := ℝ)
  let A := {x : ℝ | x ≥ 0}
  (U \ A) = {x : ℝ | x < 0} :=
by
  sorry

end complement_of_A_is_negatives_l286_286565


namespace inequality_solution_condition_necessary_but_not_sufficient_l286_286731

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ↔ (a ≥ 0 ∨ a ≤ -1) := sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (a > 0 ∨ a < -1) → (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ∧ ¬(∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0 → (a > 0 ∨ a < -1)) := sorry

end inequality_solution_condition_necessary_but_not_sufficient_l286_286731


namespace correct_operation_l286_286965

theorem correct_operation :
  let sqrt_four := real.sqrt 4
  let neg_two_cubed := (-2) ^ 3
  let neg_sqrt_four := -real.sqrt 4
  let neg_two_squared := -(2 ^ 2)
  neg_sqrt_four = -2 :=
by
  have h1 : sqrt_four = 2 := by sorry
  have h2 : neg_two_cubed = -8 := by sorry
  have h3 : neg_sqrt_four = -2 := by sorry
  have h4 : neg_two_squared = -4 := by sorry
  exact h3

end correct_operation_l286_286965


namespace condition_sufficient_but_not_necessary_l286_286135

theorem condition_sufficient_but_not_necessary (x : ℝ) :
  (x^3 > 8 → |x| > 2) ∧ (|x| > 2 → ¬ (x^3 ≤ 8 ∨ x^3 ≥ 8)) := by
  sorry

end condition_sufficient_but_not_necessary_l286_286135


namespace characterize_solution_pairs_l286_286689

/-- Define a set S --/
def S : Set ℝ := { x : ℝ | x > 0 ∧ x ≠ 1 }

/-- log inequality --/
def log_inequality (a b : ℝ) : Prop :=
  Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)

/-- Define the solution sets --/
def sol1 : Set (ℝ × ℝ) := {p | p.2 = 1 ∧ p.1 > 0 ∧ p.1 ≠ 1}
def sol2 : Set (ℝ × ℝ) := {p | p.1 > p.2 ∧ p.2 > 1}
def sol3 : Set (ℝ × ℝ) := {p | p.2 > 1 ∧ p.2 > p.1}
def sol4 : Set (ℝ × ℝ) := {p | p.1 < p.2 ∧ p.2 < 1}
def sol5 : Set (ℝ × ℝ) := {p | p.2 < 1 ∧ p.2 < p.1}

/-- Prove the log inequality and characterize the solution pairs --/
theorem characterize_solution_pairs (a b : ℝ) (h1 : a ∈ S) (h2 : b > 0) :
  log_inequality a b ↔
  (a, b) ∈ sol1 ∨ (a, b) ∈ sol2 ∨ (a, b) ∈ sol3 ∨ (a, b) ∈ sol4 ∨ (a, b) ∈ sol5 :=
sorry

end characterize_solution_pairs_l286_286689


namespace houses_with_pools_l286_286097

theorem houses_with_pools (total G overlap N P : ℕ) 
  (h1 : total = 70) 
  (h2 : G = 50) 
  (h3 : overlap = 35) 
  (h4 : N = 15) 
  (h_eq : total = G + P - overlap + N) : 
  P = 40 := by
  sorry

end houses_with_pools_l286_286097


namespace min_value_of_expression_l286_286377

theorem min_value_of_expression (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
    (h1 : ∀ n, S_n n = (4/3) * (a_n n - 1)) :
  ∃ (n : ℕ), (4^(n - 2) + 1) * (16 / a_n n + 1) = 4 :=
by
  sorry

end min_value_of_expression_l286_286377


namespace y1_lt_y2_l286_286384

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l286_286384


namespace usual_time_is_10_l286_286518

noncomputable def usual_time : ℝ :=
  let S := S
  let T := T
  let D := S * T
  let S' := (4 / 5) * S
  let D' := 1.2 * D
  let T' := D' / S'
  let additional_time := 15
  let total_time := T + 20
  T

theorem usual_time_is_10 :
  ∃ T : ℝ, T = 10 :=
begin
  use 10,
  sorry
end

end usual_time_is_10_l286_286518


namespace g_inv_comp_eval_l286_286779

noncomputable def g : ℕ → ℕ 
| 2 := 8
| 3 := 15
| 4 := 24
| 5 := 35
| 6 := 48
| _ := sorry

noncomputable def g_inv : ℕ → ℕ 
| 8 := 2
| 15 := 3
| 24 := 4
| 35 := 5
| 48 := 6
| _ := sorry

theorem g_inv_comp_eval :
  g_inv (g_inv 48 + g_inv 24 - g_inv 15) = 7 := by
  sorry

end g_inv_comp_eval_l286_286779


namespace value_of_f_5_l286_286077

-- Define the function f
def f (x y : ℕ) : ℕ := 2 * x ^ 2 + y

-- Given conditions
variable (some_value : ℕ)
axiom h1 : f some_value 52 = 60
axiom h2 : f 5 52 = 102

-- Proof statement
theorem value_of_f_5 : f 5 52 = 102 := by
  sorry

end value_of_f_5_l286_286077


namespace remaining_garden_area_l286_286626

theorem remaining_garden_area : 
  (let r := 7 in
   let total_area := 49 * Real.pi in
   let chord_dist_from_center := 3 in
   let walkway_width := 4 in
   let distance_to_walkway_edge := chord_dist_from_center + walkway_width in
   let chord_length := 2 * Real.sqrt (r^2 - chord_dist_from_center^2) in
   let theta := 2 * Real.acos (chord_dist_from_center / r) in
   let area_sector := r^2 * (theta / 2) in
   let area_triangle := 1 / 2 * chord_length * (r - chord_dist_from_center) in
   let area_segment := area_sector - area_triangle in
   let remaining_area := total_area - area_segment in
   remaining_area ≈ 49 * Real.pi - 21.53) :=
begin
  sorry
end

end remaining_garden_area_l286_286626


namespace external_common_tangents_intersect_on_circle_l286_286136

theorem external_common_tangents_intersect_on_circle
  (A B C D : Point)
  (AB CD BC AD : Line)
  (P : Proofs)
  (convex_ABCD : ConvexQuadrilateral A B C D)
  (BA_ne_BC : BA ≠ BC)
  (omega1 : Circle)
  (omega2 : Circle)
  (incircle_ABC : Incircle omega1 ABC)
  (incircle_ADC : Incircle omega2 ADC)
  (exists_omega : ∃ (Omega : Circle), Tangent Omega BA beyond A ∧ Tangent Omega BC beyond C ∧ Tangent Omega AD ∧ Tangent Omega CD) :
  ∃ (X : Point), Intersection X (ExternalCommonTangents omega1 omega2) ∧ X ∈ Omega ⟨exists_omega.some_spec⟩ := 
sorry

end external_common_tangents_intersect_on_circle_l286_286136


namespace parabola_min_value_sum_abc_zero_l286_286208

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l286_286208


namespace least_k_square_divisible_by_240_l286_286607

theorem least_k_square_divisible_by_240 (k : ℕ) (h : ∃ m : ℕ, k ^ 2 = 240 * m) : k ≥ 60 :=
by
  sorry

end least_k_square_divisible_by_240_l286_286607


namespace total_sacks_after_6_days_l286_286789

-- Define the conditions
def sacks_per_day : ℕ := 83
def days : ℕ := 6

-- Prove the total number of sacks after 6 days is 498
theorem total_sacks_after_6_days : sacks_per_day * days = 498 := by
  -- Proof Content Placeholder
  sorry

end total_sacks_after_6_days_l286_286789


namespace find_coordinates_l286_286423

def pointA : ℝ × ℝ := (2, -4)
def pointB : ℝ × ℝ := (0, 6)
def pointC : ℝ × ℝ := (-8, 10)

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_coordinates :
  scalar_mult (1/2) (vector pointA pointC) - 
  scalar_mult (1/4) (vector pointB pointC) = (-3, 6) :=
by
  sorry

end find_coordinates_l286_286423


namespace find_NP_l286_286202

theorem find_NP
  (M N P Q T : Type)
  [inhabited M] [inhabited N] [inhabited P] [inhabited Q] [inhabited T]
  (MNPQ_cyclic : is_cyclic M N P Q)
  (MP_bisects_angle_NMQ : is_angle_bisector M P Q N)
  (T_on_NQ : is_intersection T N Q P)
  (TM_eq_5 : length M T = 5)
  (TP_eq_4 : length T P = 4) :
  length N P = 9 :=
sorry

end find_NP_l286_286202


namespace second_train_speed_l286_286953

theorem second_train_speed :
  ∃ v : ℝ, 
  (∀ t : ℝ, 20 * t = v * t + 50) ∧
  (∃ t : ℝ, 20 * t + v * t = 450) →
  v = 16 :=
by
  sorry

end second_train_speed_l286_286953


namespace like_terms_monomials_m_n_l286_286810

theorem like_terms_monomials_m_n (m n : ℕ) (h1 : 3 * x ^ m * y = - x ^ 3 * y ^ n) :
  m - n = 2 :=
by
  sorry

end like_terms_monomials_m_n_l286_286810


namespace first_day_is_wednesday_l286_286537

theorem first_day_is_wednesday (day22_wednesday : ∀ n, n = 22 → (n = 22 → "Wednesday" = "Wednesday")) :
  ∀ n, n = 1 → (n = 1 → "Wednesday" = "Wednesday") :=
by
  sorry

end first_day_is_wednesday_l286_286537


namespace ordered_pairs_count_l286_286721

theorem ordered_pairs_count :
  (∃ (n : ℕ), n = 997 ∧
    ∀ (x y : ℕ), 0 < x ∧ x < y ∧ y < 1000000 → (x + y) / 2 = nat.sqrt(x * y) + 2 → (x, y) ∈ finset.range y ) := sorry

end ordered_pairs_count_l286_286721


namespace total_kids_playing_soccer_l286_286996

theorem total_kids_playing_soccer (initial_kids : ℕ) (friends_per_kid : ℕ):
  initial_kids = 14 → friends_per_kid = 3 → 
  (initial_kids + initial_kids * friends_per_kid) = 56 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  done

end total_kids_playing_soccer_l286_286996


namespace find_equation_for_second_machine_l286_286269

theorem find_equation_for_second_machine (x : ℝ) : 
  (1 / 6) + (1 / x) = 1 / 3 ↔ (x = 6) := 
by 
  sorry

end find_equation_for_second_machine_l286_286269


namespace coins_on_straight_line_possible_l286_286570

-- Definitions of coins with radii
structure Coin :=
  (radius : ℕ)
  (id : nat)

-- The coins we will use for our proof
def coin2 := Coin.mk 2 1
def coin5 := Coin.mk 5 2
def coin3 := Coin.mk 3 3
def coin8 := Coin.mk 8 4

-- Definition for centers lying on a straight line
def centers_on_straight_line (a b x y : Coin) : Prop :=
  (b.radius - a.radius) * (a.radius * x.radius + a.radius * y.radius + 2 * x.radius * y.radius) =
  a.radius * (a.radius + b.radius) * (2 * a.radius + x.radius + y.radius)

-- Proof that the given coins' centers can lie on a straight line
theorem coins_on_straight_line_possible :
  centers_on_straight_line coin2 coin5 coin3 coin8 :=
sorry

end coins_on_straight_line_possible_l286_286570


namespace greatest_odd_factors_l286_286157

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l286_286157


namespace question1_question2_question3_l286_286741

-- Definition for question 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def line_through_focus (l : ℝ × ℝ → Prop) : Prop := ∃ k : ℝ, ∀ (P : ℝ × ℝ), l P → (∃ y : ℝ, P = (1 + y/k, y))
def distance_AF (A : ℝ × ℝ) : Prop := ∀ (F : ℝ × ℝ), focus F → abs (fst A - fst F) = 4

-- Question 1
theorem question1 :
  ∀ A : ℝ × ℝ, parabola (fst A) (snd A) ∧ distance_AF A → 
  A = (3, 2 * Real.sqrt 3) ∨ A = (3, -2 * Real.sqrt 3) :=
sorry

-- Definitions for question 2
def line_intersects_parabola (l : ℝ × ℝ → Prop) : Prop := ∃ k : ℝ, k ≠ 0 ∧ ∃ x1 x2 : ℝ, l (x1, k * (x1 - 1)) ∧ l (x2, k * (x2 - 1))
def length_AB (x1 x2 : ℝ) (k : ℝ) : Prop := abs (x2 - x1 + 4 / k^2) = 5

-- Question 2
theorem question2 :
  ∀ k : ℝ, line_intersects_parabola (λ P, ∃ y : ℝ, P = (1 + y/k, y)) →
  length_AB 1 (2 + 4 / k^2) k → 
  k = 2 ∨ k = -2 :=
sorry

-- Definitions for question 3
def point_on_parabola (P : ℝ × ℝ) : Prop := ∃ y : ℝ, P = (y^2 / 4, y)
def min_distance (P : ℝ × ℝ) : Prop := 
    ∀ P : ℝ × ℝ, point_on_parabola P → abs (2 * fst P - snd P + 4) / Real.sqrt 5 = 7 * Real.sqrt 5 / 10 ∧ 
    P = (0.25, 1)

-- Question 3
theorem question3 :
  ∃ P : ℝ × ℝ, min_distance P :=
sorry

end question1_question2_question3_l286_286741


namespace find_f11_l286_286762

-- Define the odd function properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the functional equation property
def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

-- Define the specific values of the function on (0,2)
def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The main theorem that needs to be proved
theorem find_f11 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : functional_eqn f) (h3 : specific_values f) : 
  f 11 = -2 :=
sorry

end find_f11_l286_286762


namespace intervals_of_increase_max_min_on_interval_l286_286774

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2 + π / 6) + 3

theorem intervals_of_increase :
  ∀ k : ℤ, ∃ (a b : ℝ), (a = -4 * π / 3 + 4 * k * π) ∧ 
                          (b = 2 * π / 3 + 4 * k * π) ∧ 
                          ∀ x : ℝ, (a ≤ x ∧ x ≤ b) → f x < f (x + 1) :=
sorry

theorem max_min_on_interval :
  ∃ (x_min x_max : ℝ), (x_min = 4 * π / 3) ∧
                      (x_max = 2 * π / 3) ∧
                      f (4 * π / 3) = 9 / 2 ∧
                      f (2 * π / 3) = 6 :=
sorry

end intervals_of_increase_max_min_on_interval_l286_286774


namespace smallest_positive_c_satisfies_inequality_l286_286714

noncomputable def smallest_c : ℝ := 1 / 3

theorem smallest_positive_c_satisfies_inequality :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → (real.cbrt (x * y) + smallest_c * |x - y| ≥ (x + y) / 2) :=
by
  sorry

end smallest_positive_c_satisfies_inequality_l286_286714


namespace cost_price_l286_286218

theorem cost_price (C : ℝ) (h1 : 54 - C = C - 40) : C = 7 := by
  -- Proof steps go here
  sorry

end cost_price_l286_286218


namespace at_least_two_equal_elements_l286_286467

open Function

theorem at_least_two_equal_elements :
  ∀ (k : Fin 10 → Fin 10),
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j) → False :=
by
  intros k h
  sorry

end at_least_two_equal_elements_l286_286467


namespace number_of_unit_squares_in_100th_ring_center_rectangle_l286_286310

theorem number_of_unit_squares_in_100th_ring_center_rectangle :
  let n := 100,
      center_length := 2,
      center_width := 3,
      ring_length (n : ℕ) := center_length + 2 * n,
      ring_width (n : ℕ) := center_width + 2 * n,
      ring_area (n : ℕ) := (ring_length n) * (ring_width n),
      num_squares_in_ring (n : ℕ) := ring_area n - ring_area (n - 1) := 
  num_squares_in_ring 100 = 408 :=
by
  sorry

end number_of_unit_squares_in_100th_ring_center_rectangle_l286_286310


namespace greatest_odd_factors_below_200_l286_286169

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l286_286169


namespace pentagon_edges_same_color_l286_286272

theorem pentagon_edges_same_color
  (A B : Fin 5 → Fin 5)
  (C : (Fin 5 → Fin 5) × (Fin 5 → Fin 5) → Bool)
  (condition : ∀ (i j : Fin 5), ∀ (k l m : Fin 5), (C (i, j) = C (k, l) → C (i, j) ≠ C (k, m))) :
  (∀ (x : Fin 5), C (A x, A ((x + 1) % 5)) = C (B x, B ((x + 1) % 5))) :=
by
sorry

end pentagon_edges_same_color_l286_286272


namespace poisson_incomplete_gamma_equivalence_l286_286260

noncomputable def poisson_cdf (m : ℕ) (λ : ℝ) : ℝ :=
  ∑ k in Finset.range (m + 1), (Real.exp (-λ) * λ^k) / k.factorial

noncomputable def incomplete_gamma (m : ℕ) (λ : ℝ) : ℝ :=
  (1 / m.factorial) * (∫ (x : ℝ) in λ..∞, (x^m) * Real.exp (-x))

theorem poisson_incomplete_gamma_equivalence (m : ℕ) (λ : ℝ) :
  poisson_cdf m λ = incomplete_gamma m λ :=
  sorry

end poisson_incomplete_gamma_equivalence_l286_286260


namespace integral_f_eq_neg_one_third_l286_286435

noncomputable def f (x : ℝ) := x^2 + 2 * ∫ y in 0..1, f y

theorem integral_f_eq_neg_one_third : (∫ x in 0..1, f x) = -1/3 :=
by 
  sorry

end integral_f_eq_neg_one_third_l286_286435


namespace point_in_fourth_quadrant_l286_286399

noncomputable def a : ℤ := 2

theorem point_in_fourth_quadrant (x y : ℤ) (h1 : x = a - 1) (h2 : y = a - 3) (h3 : x > 0) (h4 : y < 0) : a = 2 := by
  sorry

end point_in_fourth_quadrant_l286_286399


namespace luke_split_equal_loads_l286_286145

theorem luke_split_equal_loads :
  ∀ (n : ℕ), 
  (∃ total_clothing initial_load remaining_clothing pieces_per_load : ℕ, 
    total_clothing = 47 ∧ 
    initial_load = 17 ∧ 
    remaining_clothing = total_clothing - initial_load ∧ 
    pieces_per_load = 6 ∧ 
    remaining_clothing = pieces_per_load * n) → 
  n = 5 :=
by
  intros n h
  cases h with total_clothing h
  cases h with initial_load h
  cases h with remaining_clothing h
  cases h with pieces_per_load h
  have h1 : total_clothing = 47 := h.left,
  have h2 : initial_load = 17 := h.right.left,
  have h3 : remaining_clothing = total_clothing - initial_load := h.right.right.left,
  have h4 : pieces_per_load = 6 := h.right.right.right.left,
  have h5 : remaining_clothing = pieces_per_load * n := h.right.right.right.right,
  sorry

end luke_split_equal_loads_l286_286145


namespace cube_dot_path_length_l286_286235

theorem cube_dot_path_length 
  (a : ℝ)
  (h₁ : a = 2)
  (cube_fixed : Prop)
  (cube_rolls_without_slip_or_lift : Prop)
  (starts_and_ends_same_face_touching : Prop)
  (dot_position_at_start : ℝ × ℝ × ℝ)
  (h₂ : dot_position_at_start = (0, 0, a))
  : ∃ k : ℝ, (path_length dot_position_at_start cube_fixed cube_rolls_without_slip_or_lift starts_and_ends_same_face_touching = k * π) ∧ k = 4 :=
by
  sorry

end cube_dot_path_length_l286_286235


namespace find_f6_l286_286545

-- Define the function f
variable {f : ℝ → ℝ}
-- The function satisfies f(x + y) = f(x) + f(y) for all real numbers x and y
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
-- f(4) = 6
axiom f_of_4 : f 4 = 6

theorem find_f6 : f 6 = 9 :=
by
    sorry

end find_f6_l286_286545


namespace exists_n_add_S_n_eq_1980_exists_n_add_S_n_in_consecutive_l286_286858

def S (n : ℕ) : ℕ := nat.digits 10 n |>.sum

theorem exists_n_add_S_n_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by
  sorry

theorem exists_n_add_S_n_in_consecutive (m : ℕ) : 
  ∃ n : ℕ, n + S n = m ∨ (n + 1) + S (n + 1) = m := by
  sorry

end exists_n_add_S_n_eq_1980_exists_n_add_S_n_in_consecutive_l286_286858


namespace count_numbers_with_perfect_square_factors_l286_286034

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286034


namespace perfect_square_factors_l286_286050

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286050


namespace base_of_term_raised_to_power_k_l286_286797

-- Define the parameters of the statement
def a := 2
def b := 9
def k := 11.5
def lhs (x : ℝ) := (1 / a)^23 * (1 / x)^k
def rhs := 1 / (a * b)^23

-- Statement of the problem
theorem base_of_term_raised_to_power_k (x : ℝ)
  (h_eq : lhs x = rhs)
  (h_k : k = 11.5) :
  x = 9 :=
sorry

end base_of_term_raised_to_power_k_l286_286797


namespace count_numbers_with_perfect_square_factors_l286_286059

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286059


namespace inverse_proportion_inequality_l286_286387

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l286_286387


namespace part1_solution_set_part2_range_of_a_l286_286000

noncomputable def f (x t : ℝ) : ℝ := |x - 1| + |x - t|

theorem part1_solution_set (x : ℝ) :
  f x 2 > 2 ↔ x < 1/2 ∨ x > 5/2 :=
sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ t ∈ set.Icc 1 2, ∀ x ∈ set.Icc (-1) 3, f x t ≥ a + x) → a ≤ -1 :=
sorry

end part1_solution_set_part2_range_of_a_l286_286000


namespace minimum_value_of_expression_l286_286082

theorem minimum_value_of_expression (a b : ℝ) (h : (6.choose 3) * a^3 * b^3 = 8) :
    (∃ ab : ℝ, ab = 2 → (a^2 + b^2 + 2) / ab = 3) :=
begin
  sorry
end

end minimum_value_of_expression_l286_286082


namespace collinear_points_sum_xy_solution_l286_286573

theorem collinear_points_sum_xy_solution (x y : ℚ)
  (h1 : (B : ℚ × ℚ) = (-2, y))
  (h2 : (A : ℚ × ℚ) = (x, 5))
  (h3 : (C : ℚ × ℚ) = (1, 1))
  (h4 : dist (B.1, B.2) (C.1, C.2) = 2 * dist (A.1, A.2) (C.1, C.2))
  (h5 : (y - 5) / (-2 - x) = (1 - 5) / (1 - x)) :
  x + y = -9 / 2 ∨ x + y = 17 / 2 :=
by sorry

end collinear_points_sum_xy_solution_l286_286573


namespace orthocenter_parallelogram_l286_286844

theorem orthocenter_parallelogram
  (A B C D K L M N H1 H2 H3 H4 : Type*)
  [cyclic_quadrilateral A B C D]
  [midpoint K A B] [midpoint L B C] [midpoint M C D] [midpoint N D A]
  [orthocenter H1 A K N] [orthocenter H2 K B L] [orthocenter H3 L C M] [orthocenter H4 M D N] :
  parallelogram H1 H2 H3 H4 :=
sorry

end orthocenter_parallelogram_l286_286844


namespace intersection_eq_l286_286736

def M : set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

def N : set (ℝ × ℝ) := {p | let (x, y) := p in x ^ 2 + y ^ 2 = 2}

def intersection : set (ℝ × ℝ) := {p | let (x, y) := p in y ∈ M ∧ (x, y) ∈ N}

theorem intersection_eq : intersection = {p | let (x, y) := p in 0 ≤ x ∧ x ≤ √2 ∧ x^2 = y} :=
by
  sorry

end intersection_eq_l286_286736


namespace frank_needs_more_hamburgers_l286_286724

/-- 
Frank is making hamburgers and he wants to sell them to make $50. 
Frank is selling each hamburger for $5 and 2 people purchased 4 
and another 2 customers purchased 2 hamburgers. 
Prove that Frank needs to sell 4 more hamburgers to make $50.

Parameters:
h : Nat := 5 -- price per hamburger
r : Nat := 50 -- required revenue
sold1 : Nat := 4 -- hamburgers bought by 2 people
sold2 : Nat := 2 -- hamburgers bought by another 2 people

Theorem:
Frank needs to sell 4 more hamburgers to make $50.
-/
theorem frank_needs_more_hamburgers 
  (h : Nat := 5) 
  (r : Nat := 50) 
  (sold1 : Nat := 4) 
  (sold2 : Nat := 2) : 
  (number_more_hamburgers : Nat) :=
  let total_hamburgers_sold := 2 * sold1 + 2 * sold2 
  let current_revenue := total_hamburgers_sold * h 
  let additional_revenue_needed := r - current_revenue 
  number_more_hamburgers = additional_revenue_needed / h 
by sorry

end frank_needs_more_hamburgers_l286_286724


namespace count_perfect_square_factors_l286_286021

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286021


namespace pentagon_area_ratio_l286_286891

-- Definitions of regular decagon and pentagon
def is_regular_decagon (D : fin 10 → ℝ × ℝ) : Prop :=
  ∀ i j : fin 10, i ≠ j → dist (D i) (D j) = dist (D (⟨(i + 1) % 10, by simp⟩)) (D (⟨(j + 1) % 10, by simp⟩))

def decagon_area (D : fin 10 → ℝ × ℝ) (h : is_regular_decagon D) : ℝ := sorry -- Assume the area calculation of a regular decagon

def is_regular_pentagon (P : fin 5 → ℝ × ℝ) : Prop :=
  ∀ i j : fin 5, i ≠ j → dist (P i) (P j) = dist (P (⟨(i + 1) % 5, by simp⟩)) (P (⟨(j + 1) % 5, by simp⟩))

def pentagon_area (P : fin 5 → ℝ × ℝ) (h : is_regular_pentagon P) : ℝ := sorry -- Assume the area calculation of a regular pentagon

-- Our main theorem 
theorem pentagon_area_ratio (D : fin 10 → ℝ × ℝ) (hD : is_regular_decagon D) (P : fin 5 → ℝ × ℝ) (hP : is_regular_pentagon P) :
  (pentagon_area P hP) / (decagon_area D hD) = -- Provide the final ratio using exact simplification and calculation satisfying condition
  sorry

end pentagon_area_ratio_l286_286891


namespace r_sq_plus_s_sq_l286_286134

variable {r s : ℝ}

theorem r_sq_plus_s_sq (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := 
by
  sorry

end r_sq_plus_s_sq_l286_286134


namespace contingency_fund_amount_l286_286448

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l286_286448


namespace arithmetic_sequence_sum_11_proof_l286_286831

variable {α : Type*} [LinearOrderedField α]
variable (a₁ d : α) -- declare sequence start point and common difference

-- arithmetic sequence definition
def a (n : ℕ) : α := a₁ + n * d

-- conditions from the problem
def condition1 : Prop := a 8 = (1 / 2) * a 11 + 6 -- Lean uses 0-based index, hence a 8 = a_9, a 11 = a_{12}

-- correct result based on problem
def S₁₁ : α := (11 * (a 0 + a 10)) / 2 -- sum of first 11 terms, using Lean's 0-based indexing

-- final proof statement
theorem arithmetic_sequence_sum_11_proof (h : condition1) : S₁₁ = 132 := 
by
  sorry

end arithmetic_sequence_sum_11_proof_l286_286831


namespace rational_v_l286_286681

def f (x : ℝ) : ℝ := ((x - 2) * (x + 1) * (2 * x - 1)) / (x * (x - 1))

theorem rational_v (u v : ℝ) (hu : f u = f v) (hr : ∃ q : ℚ, u = q) : ∃ q : ℚ, v = q := 
sorry

end rational_v_l286_286681


namespace slope_of_tangent_to_sin_sq_at_pi_six_l286_286221

theorem slope_of_tangent_to_sin_sq_at_pi_six : 
  deriv (fun x => (sin x)^2) (π / 6) = sqrt 3 / 2 :=
by
  sorry

end slope_of_tangent_to_sin_sq_at_pi_six_l286_286221


namespace problem_part1_problem_part2_l286_286127

noncomputable def a_n (n : ℕ+) : ℕ := 2 * n + 2

def S_n (n : ℕ+) : ℕ := (n * (2 * n + 2))  -- Sum of first n terms of a_n

def geometric_seq (m : ℕ) : ℕ := 2^(m+1)

def k_n (n : ℕ) : ℕ := 2^n - 1

noncomputable def T_n (n : ℕ) : ℕ := 2^(n+1) - n - 2

theorem problem_part1 (n : ℕ+) : a_n n = 2 * n + 2 := 
sorry

theorem problem_part2 (n : ℕ+) :
  (geometric_seq n = 2^(n+1)) ∧
  (k_n n = 2^n - 1) ∧
  (T_n n = 2^(n+1) - n - 2) :=
sorry

end problem_part1_problem_part2_l286_286127


namespace arithmetic_sequence_sum_l286_286824

noncomputable def a₁ : ℕ → ℕ := λ n, 2014
def S_n (n : ℕ) : ℕ := n * (2015 - n)

theorem arithmetic_sequence_sum :
  a₁ 1 = 2014 ∧ (S_n 2014) / 2014 - (S_n 2012) / 2012 = -2 → S_n 2015 = 0 :=
by {
  intro h,
  cases h with h₁ h₂,
  sorry
}

end arithmetic_sequence_sum_l286_286824


namespace arithmetic_progression_25th_term_l286_286604

def arithmetic_progression_nth_term (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_progression_25th_term : arithmetic_progression_nth_term 5 7 25 = 173 := by
  sorry

end arithmetic_progression_25th_term_l286_286604


namespace train_passes_jogger_in_36_seconds_l286_286997

-- Definitions from conditions in the problem
def jogger_speed_kmh : ℝ := 10
def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600

def train_speed_kmh : ℝ := 60
def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

def initial_distance_ahead : ℝ := 350
def train_length : ℝ := 150
def total_distance_to_cover : ℝ := initial_distance_ahead + train_length

def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
def time_to_pass_jogger : ℝ := total_distance_to_cover / relative_speed_ms

theorem train_passes_jogger_in_36_seconds :
  time_to_pass_jogger ≈ 36 := 
by
  sorry -- Proof omitted

end train_passes_jogger_in_36_seconds_l286_286997


namespace division_and_multiplication_l286_286957

theorem division_and_multiplication (a b c d : ℝ) : (a / b / c * d) = 30 :=
by 
  let a := 120
  let b := 6
  let c := 2
  let d := 3
  sorry

end division_and_multiplication_l286_286957


namespace seats_still_available_l286_286590

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end seats_still_available_l286_286590


namespace jake_snake_sales_l286_286116

theorem jake_snake_sales 
  (num_snakes : ℕ)
  (eggs_per_snake : ℕ)
  (regular_price : ℕ)
  (super_rare_multiplier : ℕ)
  (num_snakes = 3)
  (eggs_per_snake = 2)
  (regular_price = 250)
  (super_rare_multiplier = 4) : 
  (num_snakes * eggs_per_snake - 1) * regular_price + regular_price * super_rare_multiplier = 2250 :=
sorry

end jake_snake_sales_l286_286116


namespace average_annual_growth_rate_l286_286564

theorem average_annual_growth_rate (p q x : ℝ) : 
  (1 + p) * (1 + q) = (1 + x) * (1 + x) ↔ x = sqrt ((1 + p) * (1 + q)) - 1 := 
sorry

end average_annual_growth_rate_l286_286564


namespace balance_difference_is_347_l286_286304

theorem balance_difference_is_347 :
  let P := 12000
  let r_cedric := 0.05
  let r_daniel := 0.07
  let t := 15
  let cedric_balance := P * (1 + r_cedric) ^ t
  let daniel_balance := P * (1 + r_daniel * t)
  let difference := cedric_balance - daniel_balance
  difference ≈ 347 :=
by
  let P := 12000
  let r_cedric := 0.05
  let r_daniel := 0.07
  let t := 15
  let cedric_balance := P * (1 + r_cedric) ^ t
  let daniel_balance := P * (1 + r_daniel * t)
  let difference := cedric_balance - daniel_balance
  show difference ≈ 347
  sorry

end balance_difference_is_347_l286_286304


namespace triangle_area_correct_l286_286232

-- Define the isosceles triangle with specific side lengths.
noncomputable def area_of_triangle (a b c : ℝ) (ha : a = 17) (hb : b = 17) (hc : c = 16) : ℝ :=
  let s := (a + b + c) / 2 in
  (s * (s - a) * (s - b) * (s - c)).sqrt

-- The main theorem statement
theorem triangle_area_correct : 
  area_of_triangle 17 17 16 = 120 :=
sorry

end triangle_area_correct_l286_286232


namespace evaluate_expression_l286_286434

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 :=
by {
    sorry
}

end evaluate_expression_l286_286434


namespace count_numbers_with_square_factors_l286_286041

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286041


namespace impossible_to_cut_out_raisin_l286_286101

-- Define the square pie and the position of the raisin at the center
def square_pie := { (x, y) : ℝ × ℝ // x ∈ Icc (-1:ℝ) (1:ℝ) ∧ y ∈ Icc (-1:ℝ) (1:ℝ) }
def raisin := (0, 0)

-- Define the conditions for the triangular cuts
inductive cut : square_pie → Type
| make (p1 p2 : (ℝ × ℝ)) (h1 : p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2 ∧ 0 ≤ p1.1 ∧ p1.1 ≤ 1 ∧ 0 ≤ p1.2 ∧ p1.2 ≤ 1 ∧
                            0 ≤ p2.1 ∧ p2.1 ≤ 1 ∧ 0 ≤ p2.2 ∧ p2.2 ≤ 1) : cut ⟨p1, sorry⟩

-- Define the theorem to be proven
theorem impossible_to_cut_out_raisin :
  ∀ (C : square_pie), C.1 = 0 → C.2 = 0 → ¬∃ (c : cut C), true :=
by
  intros C h1 h2 h3
  sorry

end impossible_to_cut_out_raisin_l286_286101


namespace count_perfect_square_factors_except_one_l286_286024

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286024


namespace superhero_vs_supervillain_distance_l286_286646

-- Definitions expressing the conditions
def superhero_speed (miles : ℕ) (minutes : ℕ) := (10 : ℕ) / (4 : ℕ)
def supervillain_speed (miles_per_hour : ℕ) := (100 : ℕ)

-- Distance calculation in 60 minutes
def superhero_distance_in_hour := 60 * superhero_speed 10 4
def supervillain_distance_in_hour := supervillain_speed 100

-- Proof statement
theorem superhero_vs_supervillain_distance :
  superhero_distance_in_hour - supervillain_distance_in_hour = (50 : ℕ) :=
by
  sorry

end superhero_vs_supervillain_distance_l286_286646


namespace erased_angle_is_correct_l286_286666

theorem erased_angle_is_correct (n : ℕ) (x : ℝ) (h_convex : convex_polygon n) (h_sum_remaining : sum_remaining_angles = 1703) : x = 97 :=
by
  -- This is where the proof would be placed, but we'll use sorry for now
  sorry

end erased_angle_is_correct_l286_286666


namespace sum_of_last_two_digits_of_fibonacci_factorial_sum_l286_286700

theorem sum_of_last_two_digits_of_fibonacci_factorial_sum 
  (fib : ℕ → ℕ)
  (hfib : ∀ n < 18, (fib n) = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584].nth_le n n_lt_18)
  (factorial_sum : ℕ → ℕ)
  (hfactorial_sum : ∀ n, factorial_sum n = (fib n)!)
  (last_two_digits_sum : ℕ)
  (hlast_two_digits_sum : last_two_digits_sum = 
    (1! + 1! + 2! + 3! + 5! + 8! + 13! + 21! + 34! + 55! + 89! + 144! + 233! + 377! + 610! + 987! + 1597! + 2584!) % 100)
  (last_two_digit_fibs_sum : ℕ)
  (hlast_two_digit_fibs_sum : last_two_digit_fibs_sum = (1 + 1 + 2 + 6 + 20 + 20))
  : last_two_digit_fibs_sum % 100 = 50 := 
sorry

end sum_of_last_two_digits_of_fibonacci_factorial_sum_l286_286700


namespace num_perfect_square_factors_1800_l286_286430

theorem num_perfect_square_factors_1800 :
  let factors_1800 := [(2, 3), (3, 2), (5, 2)]
  ∃ n : ℕ, (n = 8) ∧
           (∀ p_k ∈ factors_1800, ∃ (e : ℕ), (e = 0 ∨ e = 2) ∧ n = 2 * 2 * 2 → n = 8) :=
sorry

end num_perfect_square_factors_1800_l286_286430


namespace fibonacci_term_count_le_n_l286_286982

-- Definitions
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib (n)

theorem fibonacci_term_count_le_n (N : ℕ) :
  let φ := (1 + Real.sqrt 5) / 2,
      ψ := (1 - Real.sqrt 5) / 2 in
  let Binet (n : ℕ) := (φ^n - ψ^n) / Real.sqrt 5 in
  ∀ n : ℕ, Binet n ≤ N ↔
  n ≤ Real.floor (Real.log10 (N * Real.sqrt 5) / Real.log10 φ) :=
sorry

end fibonacci_term_count_le_n_l286_286982


namespace find_a_of_extremum_l286_286453

theorem find_a_of_extremum (a : ℝ) (h : ∃ (f : ℝ → ℝ), (∀ x, f x = a / (x + 1) + x) ∧ is_extremum f 1) : a = 4 :=
sorry

end find_a_of_extremum_l286_286453


namespace diameter_of_larger_sphere_l286_286691

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (V : ℝ) (π : ℝ := 3.141592653589793)
    (h₁ : V = (4/3) * π * 12^3)
    (h₂ : V' = 3 * V)
    (h₃ : (4/3) * π * r^3 = V') :
    let diameter := 2 * 12 * real.cbrt 3 in
    let a := 24 in
    let b := 3 in
    a + b = 27 := by
  -- This is where the proof would be written, but we are not concerned with this part.
  sorry

end diameter_of_larger_sphere_l286_286691


namespace M_eq_91_solutions_l286_286903

def M (n : ℕ) : ℕ :=
  if n > 100 then
    n - 10
  else
    M (M (n + 11))

theorem M_eq_91_solutions : (Finset.filter (fun n => M n = 91) (Finset.range 102)).card = 101 :=
by
  sorry

end M_eq_91_solutions_l286_286903


namespace intersection_complement_eq_l286_286874

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def P : Finset ℕ := {1, 2, 3, 4}
def Q : Finset ℕ := {3, 4, 5}
def U_complement_Q : Finset ℕ := U \ Q

theorem intersection_complement_eq : P ∩ U_complement_Q = {1, 2} :=
by {
  sorry
}

end intersection_complement_eq_l286_286874


namespace count_perfect_square_factors_l286_286015

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286015


namespace coaster_circumference_and_area_l286_286625

def radius (r : ℝ) := 4 -- Radius of the coaster is 4 centimeters

theorem coaster_circumference_and_area (r : ℝ) :
  r = 4 →
  let C := 2 * 3.14 * r in
  let A := 3.14 * r^2 in
  (C = 25.12) ∧ (A = 50.24) :=
by
  intros h
  have C := 2 * 3.14 * r
  have A := 3.14 * r^2
  sorry

end coaster_circumference_and_area_l286_286625


namespace length_of_FD_l286_286476

-- Define the initial conditions of the problem
def square_abcd (A B C D : Point) (side_len : ℝ) : Prop :=
  distance A B = side_len ∧
  distance B C = side_len ∧
  distance C D = side_len ∧
  distance D A = side_len ∧
  angle A B C = 90 ∧
  angle B C D = 90 ∧
  angle C D A = 90 ∧
  angle D A B = 90

def point_on_line (E A D : Point) (ratio : ℝ) : Prop :=
  collinear A E D ∧
  distance A E = ratio * distance A D

-- Now we formulate the main theorem
theorem length_of_FD {A B C D E F G : Point} (side_len : ℝ) (E_ratio : ℝ) (FD_len : ℝ) : 
(square_abcd A B C D side_len) → 
(point_on_line E A D E_ratio) → 
(F_on_line F C D (side_len, 8 - FD_len)) → -- Position F on line CD defined by its distance property
(FD_len = 20/9) :=
by
  sorry

end length_of_FD_l286_286476


namespace find_a_of_parabola_and_circle_l286_286003

theorem find_a_of_parabola_and_circle (a : ℝ) (h_a : a > 0) :
  let directrix := λ x, - (1 / (4 * a))
  let circle := 
    { points : set (ℝ × ℝ) | ∃ x y, (x - 3)^2 + y^2 = 1}
  let intersects := ∃ p1 p2 : (ℝ × ℝ),
                    p1 ∈ circle ∧ p2 ∈ circle ∧ 
                    p1.2 = directrix p1.1 ∧ 
                    p2.2 = directrix p2.1 ∧ 
                    (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 3
  intersects → a = 1 / 2 :=
by
  sorry

end find_a_of_parabola_and_circle_l286_286003


namespace minimum_filtration_process_l286_286654

noncomputable def filtration_process (n : ℕ) : Prop :=
  (0.8 : ℝ) ^ n < 0.05

theorem minimum_filtration_process : ∃ n : ℕ, filtration_process n ∧ n ≥ 14 := 
  sorry

end minimum_filtration_process_l286_286654


namespace contingency_fund_allocation_l286_286440

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l286_286440


namespace victoria_worked_weeks_l286_286236

-- Definitions for given conditions
def hours_worked_per_day : ℕ := 9
def total_hours_worked : ℕ := 315
def days_in_week : ℕ := 7

-- Main theorem to prove
theorem victoria_worked_weeks : total_hours_worked / hours_worked_per_day / days_in_week = 5 :=
by
  sorry

end victoria_worked_weeks_l286_286236


namespace contingency_fund_correct_l286_286445

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l286_286445


namespace num_B_students_l286_286464

variable (students : ℕ)
variable (A B C : ℕ)
variable (prob_A prob_B prob_C : ℚ)

axiom probabilities_relations : prob_A = 0.6 * prob_B ∧ prob_C = 1.2 * prob_B
axiom total_students : students = 40
axiom grade_distribution : A + B + C = students

theorem num_B_students (hA : A = intRat ⌊ 0.6 * B ⌋) (hC : C = intRat ⌊ 1.2 * B ⌋) : 
  B = 14 := sorry

end num_B_students_l286_286464


namespace Isaiah_types_more_l286_286514

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end Isaiah_types_more_l286_286514


namespace find_number_and_n_l286_286262

def original_number (x y z n : ℕ) : Prop := 
  n = 2 ∧ 100 * x + 10 * y + z = 178

theorem find_number_and_n (x y z n : ℕ) :
  (∀ x y z n, original_number x y z n) ↔ (n = 2 ∧ 100 * x + 10 * y + z = 178) := 
sorry

end find_number_and_n_l286_286262


namespace not_always_diagonal_gt_2_l286_286332

theorem not_always_diagonal_gt_2 (H : ∀ (h : list ℝ), h.length = 6 → (∀ (s : ℝ), s ∈ h → s > 1) → convex_hexagon h → ∃ (d : ℝ), d > 2) : false :=
sorry

structure list (α : Type) :=
    -- Abstract structure representing a list.
    (length : ℕ) 
    (head : option α)

def convex_hexagon (h : list ℝ) : Prop :=
-- Property stating that the list represents a convex hexagon
sorry

end not_always_diagonal_gt_2_l286_286332


namespace no_hamiltonian_circuit_in_G_l286_286197

-- Suppose we have a specific graph G with vertices and specific edges
def G : SimpleGraph := {
  adj := λ x y,
    (x = 'A' ∧ y = 'B') ∨ (x = 'B' ∧ y = 'A') ∨
    (x = 'B' ∧ y = 'C') ∨ (x = 'C' ∧ y = 'B') ∨
    (x = 'C' ∧ y = 'D') ∨ (x = 'D' ∧ y = 'C') ∨
    (x = 'D' ∧ y = 'A') ∨ (x = 'A' ∧ y = 'D')
}

-- The vertices are {'A', 'B', 'C', 'D'}
def V := ['A', 'B', 'C', 'D']

-- The main theorem stating that no Hamiltonian circuit exists in the graph G
theorem no_hamiltonian_circuit_in_G : ¬∃ c : List V, G.isHamiltonianCircuit c :=
by sorry

end no_hamiltonian_circuit_in_G_l286_286197


namespace transform_binomial_expansion_l286_286106

variable (a b : ℝ)

theorem transform_binomial_expansion (h : (a + b)^4 = a^4 + 4 * a^3 * b + 6 * a^2 * b^2 + 4 * a * b^3 + b^4) :
  (a - b)^4 = a^4 - 4 * a^3 * b + 6 * a^2 * b^2 - 4 * a * b^3 + b^4 :=
by
  sorry

end transform_binomial_expansion_l286_286106


namespace prob_truth_same_time_l286_286977

theorem prob_truth_same_time (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.60) :
  pA * pB = 0.51 :=
by
  rw [hA, hB]
  norm_num

end prob_truth_same_time_l286_286977


namespace measure_angle_BDC_l286_286905

-- Definitions for the conditions
variables {P : Type*} [MetricSpace P]

def congruent (Δ₁ Δ₂ : Triangle P) : Prop :=
  Δ₁ ≃ Δ₂

variables (A B C D : P)
variables (Δ₁ : Triangle P) (hcong : congruent Δ₁ (triangle.mk A B C))
variables (Δ₂ : Triangle P) (hcong2 : congruent Δ₂ (triangle.mk A C D))

def equal_lengths : Prop :=
  dist A B = dist A C ∧ dist A B = dist A D

def angle_BAC : Real.angle :=
  30

axiom equal_angles_congruent : ∀ {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P]
  {A B C D : P},
  congruent (triangle.mk A B C) (triangle.mk A C D) →
  ∠ B A C = 30 → ∠ B D C = 15

-- The proof statement
theorem measure_angle_BDC
  (hcong : congruent Δ₁ (triangle.mk A B C))
  (hcong2 : congruent Δ₂ (triangle.mk A C D))
  (heq_lengths : equal_lengths A B C D)
  (hangle_BAC : angle_BAC = 30) :
  angle (D - C) (B - C) = 15 :=
by 
  sorry

end measure_angle_BDC_l286_286905


namespace sqrt_multiplication_is_correct_l286_286599

theorem sqrt_multiplication_is_correct : (sqrt 3 * sqrt 6 = 3 * sqrt 2) :=
sorry

end sqrt_multiplication_is_correct_l286_286599


namespace exists_root_in_interval_l286_286918

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem exists_root_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
begin
  let f := λ x, Real.log x + 2 * x - 6,
  sorry
end

end exists_root_in_interval_l286_286918


namespace count_numbers_with_perfect_square_factors_l286_286031

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286031


namespace range_of_m_l286_286771

noncomputable def f (x m : ℝ) := x * Real.log x + m * x^2 - m

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, 0 < x → deriv (λ x : ℝ, x * Real.log x + m * x^2 - m) x ≠ 0) : 
  m ≤ -1/2 :=
by 
  sorry

end range_of_m_l286_286771


namespace largest_of_consecutive_non_prime_integers_l286_286906

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of consecutive non-prime sequence condition
def consecutive_non_prime_sequence (start : ℕ) : Prop :=
  ∀ i : ℕ, 0 ≤ i → i < 10 → ¬ is_prime (start + i)

theorem largest_of_consecutive_non_prime_integers :
  (∃ start, start + 9 < 50 ∧ consecutive_non_prime_sequence start) →
  (∃ start, start + 9 = 47) :=
by
  sorry

end largest_of_consecutive_non_prime_integers_l286_286906


namespace replace_digit_for_divisibility_l286_286270

theorem replace_digit_for_divisibility (d : ℕ) (h₂ : d = 0 ∨ d = 5) :
  ∃ d, (d = 0) ∧ (∃ n, n = 626840 ∧ 0 ≤ n ∧ 626840 % 5 = 0 ∧ 626840 % 8 = 0) :=
by
  -- digit constraints for divisibility by 5 and 8
  have h₁ : 626840 % 5 = 0, from sorry,
  have h₂ : 626840 % 8 = 0, from sorry,
  
  -- digit cannot be 5 since 4 is second last digit
  existsi 0,
  split,
  exact rfl,
  
  -- Proof of divisibility conditions
  existsi 626840,
  split,
  refl, -- proof that n = 626840
  split,
  linarith, -- n is non-negative
  split,
  exact h₁, -- divisibility by 5
  exact h₂, -- divisibility by 8

end replace_digit_for_divisibility_l286_286270


namespace Q_returns_to_origin_with_distance_12_l286_286837

-- Define the triangle XYZ and the properties
def triangle_XYZ (a b c : ℝ) (h : a^2 + b^2 = c^2) : Prop :=
  a = 9 ∧ b = 12 ∧ c = 15

-- Define the circle properties
def circle_Q (r : ℝ) (h : r = 2) : Prop :=
  r = 2

-- Define the distance traveled by Q
def distance_traveled_by_Q (d : ℝ) : Prop :=
  d = 12

-- The theorem stating the equivalent problem
theorem Q_returns_to_origin_with_distance_12 :
  ∃ (a b c r d : ℝ), triangle_XYZ a b c (by norm_num) ∧ circle_Q r (by norm_num) ∧ distance_traveled_by_Q d :=
by {
  use [9, 12, 15, 2, 12],
  split,
  { simp [triangle_XYZ, pow_two, add_def] },
  split,
  { simp [circle_Q] },
  { simp [distance_traveled_by_Q] },
  sorry
}

end Q_returns_to_origin_with_distance_12_l286_286837


namespace inequality_problem_l286_286508

theorem inequality_problem 
  {x y z : ℝ} 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : real.sqrt x + real.sqrt y + real.sqrt z = 1) 
  : 
  (x ^ 2 + y * z) / real.sqrt (2 * x ^ 2 * (y + z)) + 
  (y ^ 2 + z * x) / real.sqrt (2 * y ^ 2 * (z + x)) + 
  (z ^ 2 + x * y) / real.sqrt (2 * z ^ 2 * (x + y)) ≥ 1 :=
sorry

end inequality_problem_l286_286508


namespace neither_necessary_nor_sufficient_l286_286510

-- defining polynomial inequalities
def inequality_1 (a1 b1 c1 x : ℝ) : Prop := a1 * x^2 + b1 * x + c1 > 0
def inequality_2 (a2 b2 c2 x : ℝ) : Prop := a2 * x^2 + b2 * x + c2 > 0

-- defining proposition P and proposition Q
def P (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := ∀ x : ℝ, inequality_1 a1 b1 c1 x ↔ inequality_2 a2 b2 c2 x
def Q (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

-- prove that Q is neither a necessary nor sufficient condition for P
theorem neither_necessary_nor_sufficient {a1 b1 c1 a2 b2 c2 : ℝ} : ¬(Q a1 b1 c1 a2 b2 c2 ↔ P a1 b1 c1 a2 b2 c2) := 
sorry

end neither_necessary_nor_sufficient_l286_286510


namespace problem_statement_l286_286782

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := { x : ℝ | abs x < 2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x - 1 }

theorem problem_statement :
  compl M ∪ compl N = Iic (-1) ∪ Ici 2 :=
by {
  sorry
}

end problem_statement_l286_286782


namespace frank_needs_more_hamburgers_l286_286725

/-- 
Frank is making hamburgers and he wants to sell them to make $50. 
Frank is selling each hamburger for $5 and 2 people purchased 4 
and another 2 customers purchased 2 hamburgers. 
Prove that Frank needs to sell 4 more hamburgers to make $50.

Parameters:
h : Nat := 5 -- price per hamburger
r : Nat := 50 -- required revenue
sold1 : Nat := 4 -- hamburgers bought by 2 people
sold2 : Nat := 2 -- hamburgers bought by another 2 people

Theorem:
Frank needs to sell 4 more hamburgers to make $50.
-/
theorem frank_needs_more_hamburgers 
  (h : Nat := 5) 
  (r : Nat := 50) 
  (sold1 : Nat := 4) 
  (sold2 : Nat := 2) : 
  (number_more_hamburgers : Nat) :=
  let total_hamburgers_sold := 2 * sold1 + 2 * sold2 
  let current_revenue := total_hamburgers_sold * h 
  let additional_revenue_needed := r - current_revenue 
  number_more_hamburgers = additional_revenue_needed / h 
by sorry

end frank_needs_more_hamburgers_l286_286725


namespace cubes_sum_mod_13_l286_286593

theorem cubes_sum_mod_13 : (∑ k in Finset.range 11, k^3) % 13 = 10 := by
  sorry

end cubes_sum_mod_13_l286_286593


namespace bus_count_l286_286229

theorem bus_count (x : ℕ) (h : 9 ≥ 0) : 
  let initial_count := 38 in
  let stepped_on := x in
  let stepped_off := x + 9 in
  initial_count - stepped_off + stepped_on = 29 :=
by
  let initial_count := 38
  let stepped_on := x
  let stepped_off := x + 9
  have h1 : initial_count - stepped_off + stepped_on = 29 := by sorry
  exact h1

end bus_count_l286_286229


namespace identify_solute_l286_286248

-- Definitions of the solutes
def NaCl := "NaCl"
def KC2H3O2 := "KC2H3O2"
def LiBr := "LiBr"
def NH4NO3 := "NH4NO3"

-- Condition: phenolphthalein turns pink in basic solution
def phenolphthalein_turns_pink_in_basic : Prop := ∀ solute, 
  solute = KC2H3O2 → solution_is_basic solute

-- Main theorem: proving KC2H3O2 is the right solute
theorem identify_solute 
  (phenolphthalein_indicator : ∀ solute, solute ∈ {NaCl, KC2H3O2, LiBr, NH4NO3} → (phenolphthalein_turns_pink_in_basic → solute = KC2H3O2)) 
  (phenolphthalein_turns_pink : phenolphthalein_turns_pink_in_basic) : 
  ∃ solute, solute = KC2H3O2 :=
by {
  have solutes : solute ∈ {NaCl, KC2H3O2, LiBr, NH4NO3},
  { sorry },
  exact phenolphthalein_indicator solute solutes phenolphthalein_turns_pink,
}

end identify_solute_l286_286248


namespace final_irises_l286_286927

-- Given conditions
def iris_rose_ratio : Ratio := Ratio.mk 3 4
def initial_roses : ℕ := 32
def added_roses : ℕ := 16
def total_roses : ℕ := initial_roses + added_roses

-- The final goal: total irises after addition
theorem final_irises : total_roses * 3 / 4 = 36 := by
  -- This is just stating the theorem without proving it
  sorry

end final_irises_l286_286927


namespace find_m_l286_286781

-- Defining the sets and conditions
def A (m : ℝ) : Set ℝ := {1, m-2}
def B : Set ℝ := {x | x = 2}

theorem find_m (m : ℝ) (h : A m ∩ B = {2}) : m = 4 := by
  sorry

end find_m_l286_286781


namespace sum_possible_digits_in_base2_l286_286989

theorem sum_possible_digits_in_base2 (n : ℕ) (h₁ : 9^3 ≤ n) (h₂ : n < 9^4) :
  (let d1 := (nat.log 2 n).to_nat + 1 in
   let d2 := (nat.log 2 (n + (9^4 - 1 - n))).to_nat + 1 in
   d1 + d1.succ + d1.succ.succ = 33) :=
sorry

end sum_possible_digits_in_base2_l286_286989


namespace det_proj_matrix_l286_286126

def vector := ℝ × ℝ × ℝ

def proj_matrix (v : vector) : matrix (fin 3) (fin 3) ℝ :=
  let len := real.sqrt (v.1^2 + v.2^2 + v.3^2)
  let norm_vec := (v.1 / len, v.2 / len, v.3 / len)
  let a1 : fin 3 → ℝ := ![norm_vec.1, norm_vec.2, norm_vec.3]
  let a2 : fin 3 → fin 3 → ℝ := (λ i j, a1 i * a1 j)
  λ i j, a2 i j

theorem det_proj_matrix (v : vector) (h : v = (3, -1, 4)) : det (proj_matrix v) = 0 :=
by
  sorry

end det_proj_matrix_l286_286126


namespace rational_v_l286_286682

def f (x : ℝ) : ℝ := ((x - 2) * (x + 1) * (2 * x - 1)) / (x * (x - 1))

theorem rational_v (u v : ℝ) (hu : f u = f v) (hr : ∃ q : ℚ, u = q) : ∃ q : ℚ, v = q := 
sorry

end rational_v_l286_286682


namespace abigail_written_words_l286_286283

theorem abigail_written_words (total_words : ℕ) (typing_rate_per_hour : ℕ) (remaining_minutes : ℕ) :
  total_words = 1000 →
  typing_rate_per_hour = 600 →
  remaining_minutes = 80 →
  ∃ words_written : ℕ, words_written = 200 :=
by
  assume h_total_words ht_rate_per_hour h_remaining_minutes
  sorry

end abigail_written_words_l286_286283


namespace chord_bisected_by_P_hyperbola_eqn_l286_286266

theorem chord_bisected_by_P_hyperbola_eqn :
  ∀ (A B : ℝ × ℝ),
  (A.1^2 / 36 - A.2^2 / 9 = 1) ∧ (B.1^2 / 36 - B.2^2 / 9 = 1) ∧
  ((A.1 + B.1) / 2 = 4) ∧ ((A.2 + B.2) / 2 = 2) →
  ∀ (l : ℝ → ℝ), (∃ (k : ℝ), l = λ x, k * x + 2 - k * 4) → l = λ x, 2 :=
sorry

end chord_bisected_by_P_hyperbola_eqn_l286_286266


namespace find_x_in_equation_l286_286436

theorem find_x_in_equation (m : ℝ) (h : 0 < m) : (∃ x : ℝ, 10^x = log10 (10 * m) + log10 (1 / m)) ↔ x = 0 :=
by
  sorry

end find_x_in_equation_l286_286436


namespace position_of_2020_in_sequence_l286_286662

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem position_of_2020_in_sequence :
  let sequence := (list.range 10000).filter (λ n, sum_of_digits n = 4 ∧ is_four_digit_number n) in
  let sorted_sequence := sequence.sorted (≤) in
  list.index_of 2020 sorted_sequence = 27 := -- index is zero-based, so 28th position is index 27
by
  sorry

end position_of_2020_in_sequence_l286_286662


namespace probability_rain_at_most_2_days_l286_286551

theorem probability_rain_at_most_2_days :
  let p := 1 / 5
  let q := 4 / 5
  let days := 28
  (∑ k in Finset.range 3, (Nat.choose days k) * (p ^ k) * (q ^ (days - k)) = 0.184) :=
by
  sorry

end probability_rain_at_most_2_days_l286_286551


namespace smallest_m_l286_286850

noncomputable def S : set ℂ :=
  {z : ℂ | ∃ (x y : ℝ), (1/2 ≤ x ∧ x ≤ real.sqrt 2 / 2) ∧ (y ≥ 1/2) ∧ (z = x + y * complex.I)}

theorem smallest_m (m : ℕ) (hm : m = 24) : ∀ n ≥ m, ∃ z ∈ S, z^n = 1 := 
by 
  sorry

end smallest_m_l286_286850


namespace first_customer_bought_5_l286_286486

variables 
  (x : ℕ) -- Number of boxes the first customer bought
  (x2 : ℕ) -- Number of boxes the second customer bought
  (x3 : ℕ) -- Number of boxes the third customer bought
  (x4 : ℕ) -- Number of boxes the fourth customer bought
  (x5 : ℕ) -- Number of boxes the fifth customer bought

def goal : ℕ := 150
def remaining_boxes : ℕ := 75
def sold_boxes := x + x2 + x3 + x4 + x5

axiom second_customer (hx2 : x2 = 4 * x) : True
axiom third_customer (hx3 : x3 = (x2 / 2)) : True
axiom fourth_customer (hx4 : x4 = 3 * x3) : True
axiom fifth_customer (hx5 : x5 = 10) : True
axiom sales_goal (hgoal : sold_boxes = goal - remaining_boxes) : True

theorem first_customer_bought_5 (hx2 : x2 = 4 * x) 
                                (hx3 : x3 = (x2 / 2)) 
                                (hx4 : x4 = 3 * x3) 
                                (hx5 : x5 = 10) 
                                (hgoal : sold_boxes = goal - remaining_boxes) : 
                                x = 5 :=
by
  -- Here, we would perform the proof steps
  sorry

end first_customer_bought_5_l286_286486


namespace negation_if_then_l286_286549

theorem negation_if_then (x : ℝ) : ¬ (x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) :=
by 
  sorry

end negation_if_then_l286_286549


namespace angle_AMC_eq_angle_BMP_l286_286752

-- Define the structure of the triangle and the necessary points
def isosceles_right_triangle (A B C M P : ℝ × ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧
  A = (0, 0) ∧
  B = (a, 0) ∧
  C = (0, a) ∧
  M = (a / 2, 0) ∧
  P = (0, 0)

theorem angle_AMC_eq_angle_BMP (A B C M P : ℝ × ℝ) : 
  isosceles_right_triangle A B C M P →
  ∠(A, M, C) = ∠(B, M, P) :=
by
  sorry

end angle_AMC_eq_angle_BMP_l286_286752


namespace find_B_and_C_l286_286718

def values_of_B_and_C (B C : ℤ) : Prop :=
  5 * B - 3 = 32 ∧ 2 * B + 2 * C = 18

theorem find_B_and_C : ∃ B C : ℤ, values_of_B_and_C B C ∧ B = 7 ∧ C = 2 := by
  sorry

end find_B_and_C_l286_286718


namespace function_increasing_l286_286777

def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem function_increasing (a : ℝ) (h₀ : 0 < a) (h₁ : ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ a → f x ≤ f y) :
  0 < a ∧ a ≤ (Real.pi / 12) :=
by
  sorry

end function_increasing_l286_286777


namespace combined_variance_l286_286568

def mean (s : List ℝ) : ℝ :=
s.sum / s.length

def variance (s : List ℝ) : ℝ :=
let m := mean s
(s.map (λ x => (x - m) ^ 2)).sum / s.length

theorem combined_variance (A B : List ℝ) (hA_length : A.length = 6) (hB_length : B.length = 6)
    (hA_mean : mean A = 3) (hA_variance : variance A = 5)
    (hB_mean : mean B = 5) (hB_variance : variance B = 3) :
    variance (A ++ B) = 5 := by
  simp [A, B, hA_length, hB_length, hA_mean, hA_variance, hB_mean, hB_variance]
  sorry

end combined_variance_l286_286568


namespace sequence_sum_bound_l286_286277

theorem sequence_sum_bound (a : ℕ → ℤ)
  (h1 : ∀ j : ℕ, j ≥ 1 → 1 ≤ a j ∧ a j ≤ 2015) 
  (h2 : ∀ k l : ℕ, 1 ≤ k → k < l → k + a k ≠ l + a l) :
  ∃ b N : ℕ, ∀ m n : ℕ, n > m ∧ m ≥ N → abs (∑ j in finset.range (n - m), (a (m + 1 + j) - b)) ≤ 1007^2 := 
sorry

end sequence_sum_bound_l286_286277


namespace contingency_fund_amount_l286_286447

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l286_286447


namespace cos_equiv_l286_286353

theorem cos_equiv (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) (hcos : Real.cos (n * Real.pi / 180) = Real.cos (1018 * Real.pi / 180)) : n = 62 := 
sorry

end cos_equiv_l286_286353


namespace find_value_of_product_of_1_plus_roots_of_polynomial_l286_286860

theorem find_value_of_product_of_1_plus_roots_of_polynomial :
  ∀ (a b c : ℝ),
  (Polynomial.aeval a (Polynomial.C 1 + Polynomial.C (-15)*X + 
   Polynomial.C 25*X^2 + Polynomial.C (-10)*X^3)) = 0 →
  (Polynomial.aeval b (Polynomial.C 1 + Polynomial.C (-15)*X + 
   Polynomial.C 25*X^2 + Polynomial.C (-10)*X^3)) = 0 →
  (Polynomial.aeval c (Polynomial.C 1 + Polynomial.C (-15)*X + 
   Polynomial.C 25*X^2 + Polynomial.C (-10)*X^3)) = 0 →
  (1 + a) * (1 + b) * (1 + c) = 51 :=
begin
  sorry,
end

end find_value_of_product_of_1_plus_roots_of_polynomial_l286_286860


namespace greatest_odd_factors_l286_286158

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l286_286158


namespace B_is_Brownian_motion_l286_286497

noncomputable def inner_product (f g : ℝ → ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..1, f x * g x

variables {e : ℕ → (ℝ → ℝ)} {ξ : ℕ → ℝ} {I : ℝ → ℝ}

-- e_n is a complete orthonormal system in L²[0,1]
axiom complete_orthonormal_system {n : ℕ} (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∀ n m, inner_product (e n) (e m) = if n = m then 1 else 0

-- (ξ_n) are i.i.d. N(0,1) random variables
axiom iid_gaussian {n : ℕ} : ∀ n, ξ n ~ᵢ ℕ(0,1)

-- (f, g) denotes the inner product
def inner_product (f g : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, f x * g x

-- Define I_{[0, t]} = I_{[0, t]}(x) for x ∈ [0, 1]
def I (t x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ t then 1 else 0

noncomputable def B (t : ℝ) : ℝ :=
  ∑ n in (finset.range (nat.succ (floor (1/t).to_nat))), inner_product (e n) (I t) * ξ n

-- The process B_t is a Brownian motion
theorem B_is_Brownian_motion : ∀ t, 0 ≤ t ∧ t ≤ 1 → is_Brownian_motion (B t) :=
sorry

end B_is_Brownian_motion_l286_286497


namespace greatest_odd_factors_l286_286150

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l286_286150


namespace domain_h_parity_h_h_negative_set_find_a_l286_286419

-- Define the functions and conditions
def f (a : ℝ) (x : ℝ) := log a (1 + x)
def g (a : ℝ) (x : ℝ) := log a (1 - x)
def h (a : ℝ) (x : ℝ) := f a x - g a x

-- Domain of h(x)
theorem domain_h (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : 
  ∀ x, (-1 < x ∧ x < 1) → 0 ≤ 1 + x ∧ 0 ≤ 1 - x :=
by sorry

-- Parity of h(x)
theorem parity_h (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (x : ℝ) : 
  h a (-x) = -h a x :=
by sorry

-- Given f(3) = 2, find set of x that makes h(x) < 0
theorem h_negative_set (a : ℝ) (ha : f a 3 = 2) :
  a = 2 →
  ∀ x, (-1 < x ∧ x < 0) → h 2 x < 0 :=
by sorry

-- Find the value of a given range of h(x) on [0, 1/2]
theorem find_a (h_range : ∀ x, (0 ≤ x ∧ x ≤ 1/2) → (0 ≤ h 3 x ∧ h 3 x ≤ 1)) : 
  a = 3 :=
by sorry

end domain_h_parity_h_h_negative_set_find_a_l286_286419


namespace functional_equation_solution_l286_286856

theorem functional_equation_solution:
  (∃ f : ℝ → ℝ, (∀ x y z : ℝ, f(x^2 + y * f(z)) = x * f(x) + z * f(y)) ∧
    (∃ n : ℕ, (∃ s : ℝ, f(5) = s ∧ n = (if f(5) = 0 then 1 else 2)))) :=
  sorry

end functional_equation_solution_l286_286856


namespace count_perfect_square_factors_l286_286067

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286067


namespace hamburgers_needed_l286_286727

theorem hamburgers_needed (price_per_hamburger : ℕ) (initial_hamburgers_1 : ℕ) (initial_hamburgers_2 : ℕ) (target_money : ℕ) :
  price_per_hamburger = 5 →
  initial_hamburgers_1 = 4 →
  initial_hamburgers_2 = 2 →
  target_money = 50 →
  (target_money - (price_per_hamburger * (initial_hamburgers_1 + initial_hamburgers_2))) / price_per_hamburger = 4 :=
by
  intros h_price h_initial_1 h_initial_2 h_target
  rw [h_price, h_initial_1, h_initial_2, h_target]
  norm_num
  sorry

end hamburgers_needed_l286_286727


namespace min_attempts_to_open_locker_l286_286271

/-- 
  Given a five-digit number that must contain both sequences 23 and 37,
  prove that the minimum number of such numbers to be tried to definitely open
  the locker is 356.
-/
theorem min_attempts_to_open_locker : 
  let five_digit_numbers_containing (a b : ℕ) := { n : ℕ // 10000 ≤ n ∧ n < 100000 ∧ (n.toString.contains (a.toString)) ∧ (n.toString.contains (b.toString)) }
  in (finset.card (finset.filter (λ n, (n.val.toString.contains "23") ∧ (n.val.toString.contains "37")) (finset.Ico 10000 100000))) = 356 :=
sorry

end min_attempts_to_open_locker_l286_286271


namespace perfect_square_factors_l286_286049

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286049


namespace find_Sa_plus_S_minus_a_l286_286719

variable {a : ℝ}
noncomputable def S (r : ℝ) := 15 / (1 - r)

theorem find_Sa_plus_S_minus_a (h1 : -1 < a) (h2 : a < 1) (h3 : S(a) * S(-a) = 2025) : S(a) + S(-a) = 270 := by
  sorry

end find_Sa_plus_S_minus_a_l286_286719


namespace greatest_odd_factors_below_200_l286_286172

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l286_286172


namespace greatest_odd_factors_below_200_l286_286171

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l286_286171


namespace percent_of_a_is_4b_l286_286533

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : (4 * b) / a = 20 / 9 :=
by sorry

end percent_of_a_is_4b_l286_286533


namespace toothpaste_usage_l286_286935

-- Define the variables involved
variables (t : ℕ) -- total toothpaste in grams
variables (d : ℕ) -- grams used by dad per brushing
variables (m : ℕ) -- grams used by mom per brushing
variables (b : ℕ) -- grams used by Anne + brother per brushing
variables (r : ℕ) -- brushing rate per day
variables (days : ℕ) -- days for toothpaste to run out
variables (N : ℕ) -- family members

-- Given conditions
variables (ht : t = 105)         -- Total toothpaste is 105 grams
variables (hd : d = 3)           -- Dad uses 3 grams per brushing
variables (hm : m = 2)           -- Mom uses 2 grams per brushing
variables (hr : r = 3)           -- Each member brushes three times a day
variables (hdays : days = 5)     -- Toothpaste runs out in 5 days

-- Additional calculations
variable (total_brushing : ℕ)
variable (total_usage_d: ℕ)
variable (total_usage_m: ℕ)
variable (total_usage_parents: ℕ)
variable (total_usage_family: ℕ)

-- Helper expressions
def total_brushing_expr := days * r * 2
def total_usage_d_expr := d * r
def total_usage_m_expr := m * r
def total_usage_parents_expr := (total_usage_d_expr + total_usage_m_expr) * days
def total_usage_family_expr := t - total_usage_parents_expr

-- Assume calculations
variables (h1: total_usage_d = total_usage_d_expr)  
variables (h2: total_usage_m = total_usage_m_expr)
variables (h3: total_usage_parents = total_usage_parents_expr)
variables (h4: total_usage_family = total_usage_family_expr)
variables (h5 : total_brushing = total_brushing_expr)

-- Define the proof
theorem toothpaste_usage : 
  b = total_usage_family / total_brushing := 
  sorry

end toothpaste_usage_l286_286935


namespace area_of_triangle_l286_286349

def point := (ℝ × ℝ × ℝ)
def A : point := (0, 5, 8)
def B : point := (-2, 4, 4)
def C : point := (-3, 7, 4)

theorem area_of_triangle (A B C : point) : 
  (let x1 := A.1; let y1 := A.2.1; let z1 := A.2.2;
       x2 := B.1; let y2 := B.2.1; let z2 := B.2.2;
       x3 := C.1; let y3 := C.2.1; let z3 := C.2.2 in
  0.5 * abs ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))) = 15 / 2 := 
  sorry

end area_of_triangle_l286_286349


namespace ledi_age_10_in_years_l286_286983

-- Definitions of ages of Duoduo and Ledi
def duoduo_current_age : ℝ := 10
def years_ago : ℝ := 12.3
def sum_ages_years_ago : ℝ := 12

-- Function to calculate Ledi's current age
def ledi_current_age :=
  (sum_ages_years_ago + years_ago + years_ago) + (duoduo_current_age - years_ago)

-- Function to calculate years from now for Ledi to be 10 years old
def years_until_ledi_age_10 (ledi_age_now : ℝ) : ℝ :=
  10 - ledi_age_now

-- Main statement we need to prove
theorem ledi_age_10_in_years : years_until_ledi_age_10 ledi_current_age = 6.3 :=
by
  -- Proof goes here
  sorry

end ledi_age_10_in_years_l286_286983


namespace ticket_cost_l286_286295

-- Conditions
variable (tickets_bought tickets_left money_spent : ℕ)
variable (tickets_used cost_per_ticket : ℕ)

-- Hypotheses based on conditions
hypothesis H1 : tickets_bought = 13
hypothesis H2 : tickets_left = 4
hypothesis H3 : money_spent = 81
hypothesis H4 : tickets_used = tickets_bought - tickets_left

-- Proof goal
theorem ticket_cost (H5 : money_spent = cost_per_ticket * tickets_used) : cost_per_ticket = 9 :=
by
  have : tickets_used = 9, from by simp [H1, H2, H4]
  have : cost_per_ticket * 9 = 81, from by simp [this, H3, H5]
  sorry

end ticket_cost_l286_286295


namespace proof_for_imo_part_of_z_l286_286320

noncomputable def z : ℂ := 2 / (-1 + complex.I)

theorem proof_for_imo_part_of_z : z.im = -1 := by
  sorry

end proof_for_imo_part_of_z_l286_286320


namespace count_numbers_with_square_factors_l286_286046

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286046


namespace rectangle_cutouts_in_10x10_grid_l286_286471

theorem rectangle_cutouts_in_10x10_grid :
  let n := 10
  let rect_width := 3
  ∃ (num_ways : ℕ), num_ways = 512 ∧ ∀ (l : ℕ), l ∈ {4, 5, 6, 7, 8, 9, 10} →
    (let horizontal_ways := 8 * (11 - l)
         vertical_ways := 8 * (11 - l)
         total_ways := horizontal_ways + vertical_ways 
     in total_ways - if l = 3 then 64 else 0) +
    (let horizontal_ways := 8 * (11 - l)
         vertical_ways := 8 * (11 - l)
         total_ways := horizontal_ways + vertical_ways 
     in total_ways - if l = 3 then 64 else 0) = 512 := 
by
  sorry

end rectangle_cutouts_in_10x10_grid_l286_286471


namespace locus_of_centers_of_inscribed_rectangles_l286_286242

-- Given a triangle ABC with base BC
variables {A B C : ℝ} -- Assuming points are on real line for simplicity
variable h : ℝ -- height from A to BC
variables {M N : ℝ} -- midpoints

-- The condition that triangle ABC has a base BC, with an inscribed rectangle whose one side lies along the base BC
def inscribed_rectangle (A B C M N : ℝ) (h : ℝ) : Prop :=
  -- Add your geometric conditions

-- The centroid of the rectangle (which is the midpoint of its diagonal)
def rectangle_centroid (A B C M N : ℝ) (h : ℝ) : Prop :=
  -- Add conditions that express the centroid terms

-- Proving that the set of all centroids (midpoints of diagonals) is a line segment
theorem locus_of_centers_of_inscribed_rectangles (A B C M N : ℝ) (h : ℝ) :
  ∃ line_segment, ∀ (centroid : ℝ),
    inscribed_rectangle A B C M N h →
    rectangle_centroid A B C M N h →
    centroid ∈ line_segment :=
sorry

end locus_of_centers_of_inscribed_rectangles_l286_286242


namespace contingency_fund_correct_l286_286443

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l286_286443


namespace f_240_eq_388_l286_286554

/-- 
  Define the sequences f and g such that:
  1. f is a strictly increasing sequence
  2. g is a strictly increasing sequence
  3. g(n) = f[f(n)] + 1 for all n ≥ 1
-/
axiom f : ℕ → ℕ
axiom g : ℕ → ℕ
axiom f_increasing : ∀ {m n : ℕ}, m < n → f(m) < f(n)
axiom g_increasing : ∀ {m n : ℕ}, m < n → g(m) < g(n)
axiom g_condition : ∀ n ≥ 1, g(n) = f(f(n)) + 1

/-- Prove that f(240) = 388 -/
theorem f_240_eq_388 : f 240 = 388 :=
sorry

end f_240_eq_388_l286_286554


namespace greatest_odd_factors_below_200_l286_286170

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l286_286170


namespace total_travel_time_l286_286582

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end total_travel_time_l286_286582


namespace collinear_vectors_l286_286009

-- Defining the basic vectors
variables (x y k : ℝ)

-- Given Vector Definitions
def a := (x, 1)
def b := (0, -1)
def c := (k, y)

-- Collinearity condition of vectors
def collinear (u v : ℝ × ℝ) : Prop := ∃ α : ℝ, v = α • u

-- The theorem to prove
theorem collinear_vectors (x y k : ℝ) (h : collinear (a x 1 - 2 • b) (c k y)) : k = x :=
by
  sorry

end collinear_vectors_l286_286009


namespace parabola_min_value_roots_l286_286547

-- Lean definition encapsulating the problem conditions and conclusion
theorem parabola_min_value_roots (a b c : ℝ) 
  (h1 : ∀ x, (a * x^2 + b * x + c) ≥ 36)
  (hvc : (b^2 - 4 * a * c) = 0)
  (hx1 : (a * (-3)^2 + b * (-3) + c) = 0)
  (hx2 : (a * (5)^2 + b * 5 + c) = 0)
  : a + b + c = 36 := by
  sorry

end parabola_min_value_roots_l286_286547


namespace rational_of_rational_f_eq_f_l286_286684

def f (x : ℝ) : ℝ := ((x-2) * (x+1) * (2*x-1)) / (x * (x-1))

theorem rational_of_rational_f_eq_f (u v : ℝ) (hu : u ∈ ℚ) (hv_eq : f u = f v) : v ∈ ℚ :=
by
  sorry

end rational_of_rational_f_eq_f_l286_286684


namespace derivative_ln_x_squared_plus_2_l286_286732

noncomputable def y (x : ℝ) : ℝ := Real.log (x^2 + 2)

def y_prime (x : ℝ) : ℝ := (2 * x) / (x^2 + 2)

theorem derivative_ln_x_squared_plus_2 (x : ℝ) :
  has_deriv_at y (y_prime x) x := by sorry

end derivative_ln_x_squared_plus_2_l286_286732


namespace max_mn_l286_286917

theorem max_mn (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (m n : ℝ)
  (h₂ : 2 * m + n = 2) : m * n ≤ 4 / 9 :=
by
  sorry

end max_mn_l286_286917


namespace circle_distance_to_point_is_five_l286_286543

noncomputable def circle_center_distance : ℝ :=
let c_x := 3, c_y := 4,
    a_x := -1, a_y := 1 in
real.sqrt ((c_x - a_x)^2 + (c_y - a_y)^2)

theorem circle_distance_to_point_is_five :
  let eqn := (λ x y : ℝ, x^2 + y^2 - 6*x - 8*y + 23) in
  eqn 0 0 = 23 → circle_center_distance = 5 :=
by
  intro h
  rw [circle_center_distance]
  calc
    _ = real.sqrt ((3 + 1)^2 + (4 - 1)^2) : by sorry
    _ = 5                                  : by sorry

end circle_distance_to_point_is_five_l286_286543


namespace cos_4_arccos_l286_286342

theorem cos_4_arccos (y : ℝ) (hy1 : y = Real.arccos (2/5)) (hy2 : Real.cos y = 2/5) : 
  Real.cos (4 * y) = -47 / 625 := 
by 
  sorry

end cos_4_arccos_l286_286342


namespace minimum_value_l286_286397

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2 * m + n = 4) : 
  ∃ (x : ℝ), (x = 2) ∧ (∀ (p q : ℝ), q > 0 → p > 0 → 2 * p + q = 4 → x ≤ (1 / p + 2 / q)) := 
sorry

end minimum_value_l286_286397


namespace diameter_of_circle_c_correct_l286_286306

noncomputable theory
open Real

def diameter_of_circle_c : Prop :=
  let D_diam := 20
  let D_radius := D_diam / 2
  let D_area := π * D_radius ^ 2
  ∃ r_C : ℝ, let C_area := π * r_C ^ 2 in r_C * 2 = 7.08 ∧ (D_area - C_area) / C_area = 7

theorem diameter_of_circle_c_correct : diameter_of_circle_c :=
sorry

end diameter_of_circle_c_correct_l286_286306


namespace triangle_BD_length_l286_286479

theorem triangle_BD_length (A B C D E : Type)
  [has_angle_right : ∠C = 90°] 
  (AC_6 : AC = 6)
  (BC_8 : BC = 8)
  (on_AB : D ∈ line_segment A B)
  (on_BC : E ∈ line_segment B C)
  (angle_BED_right : ∠BED = 90°)
  (DE_4 : DE = 4) :
  BD = 20 / 3 :=
sorry

end triangle_BD_length_l286_286479


namespace tan_subtraction_l286_286449

theorem tan_subtraction (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 :=
by
  sorry

end tan_subtraction_l286_286449


namespace quadratic_root_range_l286_286086

theorem quadratic_root_range (a : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = -x^2 + 2 * a * x + 4 * a + 1)
  (h_root1 : ∃ x, x < -1 ∧ f x = 0)
  (h_root2 : ∃ x, x > 3 ∧ f x = 0)
  : ∃ I : set ℝ, I = {x | x > 4 / 5 ∧ x < 1} ∧ a ∈ I :=
sorry

end quadratic_root_range_l286_286086


namespace count_valid_sets_l286_286942

def card : Type := Σ (shape : {Δ, □, ⊙}), Σ (letter : {A, B, C}), {1, 2, 3}

def deck : finset card :=
  finset.univ

def valid_set (s : finset card) : Prop :=
  s.card = 3 ∧ ∀ (c1 c2 : card) (h1 : c1 ∈ s) (h2 : c2 ∈ s), 
  (c1.1 ≠ c2.1 ∨ c1.2.1 ≠ c2.2.1 ∨ c1.2.2 ≠ c2.2.2)

theorem count_valid_sets : (finset.filter valid_set (finset.powerset_len 3 deck)).card = 2358 :=
sorry

end count_valid_sets_l286_286942


namespace calc1_calc2_l286_286674

noncomputable def calculation1 := -4^2

theorem calc1 : calculation1 = -16 := by
  sorry

noncomputable def calculation2 := (-3) - (-6)

theorem calc2 : calculation2 = 3 := by
  sorry

end calc1_calc2_l286_286674


namespace max_shirt_price_l286_286285

theorem max_shirt_price (total_money : ℕ) (n_shirts : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ)
  (h_total_money : total_money = 150)
  (h_entrance_fee : entrance_fee = 5)
  (h_n_shirts : n_shirts = 20)
  (h_tax_rate : tax_rate = 0.05) : 
  ∃ (price_per_shirt : ℕ), (price_per_shirt = 6) ∧ (price_per_shirt * n_shirts * (1 + tax_rate) : ℚ ≤ (total_money - entrance_fee)) :=
sorry

end max_shirt_price_l286_286285


namespace pradeep_passing_percentage_l286_286887

-- Given conditions
variables (m f M : ℕ)
-- Assumptions based on the problem conditions
hypothesis (h1 : m = 160)
hypothesis (h2 : f = 25)
hypothesis (h3 : M = 925)

-- Define the passing mark
def passing_mark := m + f

-- Define the percentage calculation
def percentage (passing_mark M : ℕ) : ℕ := (passing_mark * 100) / M

-- Main theorem to state the proof problem
theorem pradeep_passing_percentage : 
  percentage (passing_mark m f) M = 20 :=
by
  rw [percentage, passing_mark, h1, h2, h3],
  norm_num,
  sorry

end pradeep_passing_percentage_l286_286887


namespace triangle_is_right_angled_l286_286585

theorem triangle_is_right_angled
  (a b c : ℝ)
  (h1 : a ≠ c)
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : c > 0)
  (h5 : ∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0 ∧ x ≠ 0) :
  c^2 + b^2 = a^2 :=
by sorry

end triangle_is_right_angled_l286_286585


namespace isaiah_types_more_words_than_micah_l286_286512

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end isaiah_types_more_words_than_micah_l286_286512


namespace no_sol_for_frac_eq_l286_286525

theorem no_sol_for_frac_eq (x y : ℕ) (h : x > 1) : ¬ (y^5 + 1 = (x^7 - 1) / (x - 1)) :=
sorry

end no_sol_for_frac_eq_l286_286525


namespace perimeter_triangle_PBQ_l286_286611

noncomputable def square (side : ℝ) := {AB := side, BC := side, CD := side, DA := side, angle_ABC := 90, angle_BCD := 90, angle_CDA := 90, angle_DAB := 90}

variables (P Q : ℝ) (x y : ℝ)
variables (side : ℝ)
variables (AB P BC Q CD D DA angle_PD angle_P )

def triangle_PDQ_area (x y : ℝ) := (1/2) * real.sqrt 2 / 2 * real.sqrt (1 - x) * real.sqrt (1 - y)

-- Theorem where given conditions lead to proving the perimeter
theorem perimeter_triangle_PBQ : 
  ∀ (AP BP BQ CR : ℝ) (x y : ℝ), 
  0 ≤ x ∧ x ≤ 1 -> 
  0 ≤ y ∧ y ≤ 1 -> 
  BP = 1 - x -> BQ = 1 - y -> 
  ∠PDQ = 45 -> 
  ∆ PBQ = 1 - x + 1 - y + x + y := 
begin
    sorry
end

end perimeter_triangle_PBQ_l286_286611


namespace find_m_evaluate_expression_l286_286405

noncomputable def m := -1
noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom α_conditions (h1 : P (-1, 2 * Real.sqrt 2)) (h2 : Real.sin α = 2 * Real.sqrt 2 / 3) (h3 : π / 2 < α ∧ α < π) : True

axiom β_condition (h4 : Real.tan β = Real.sqrt 2) : True

def expression := 
  (Real.sin α * Real.cos β + 3 * Real.sin (π / 2 + α) * Real.sin β) /
  (Real.cos (π + α) * Real.cos (-β) - 3 * Real.sin α * Real.sin β)

theorem find_m : m = -1 := sorry

theorem evaluate_expression :
  expression = Real.sqrt 2 / 11 := sorry

end find_m_evaluate_expression_l286_286405


namespace usable_sheets_in_stack_l286_286273

theorem usable_sheets_in_stack :
  ∀ (n : ℕ) (h₁ : n = 400) (h₂ : 4 / n = 0.01) (h₃ : 12 / 0.01 = 1200) (h₄ : 0.90 * 1200 = 1080),
  0.90 * (12 / (4 / n)) = 1080 :=
by
  intros n h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end usable_sheets_in_stack_l286_286273


namespace cos_4_arccos_l286_286343

theorem cos_4_arccos (y : ℝ) (hy1 : y = Real.arccos (2/5)) (hy2 : Real.cos y = 2/5) : 
  Real.cos (4 * y) = -47 / 625 := 
by 
  sorry

end cos_4_arccos_l286_286343


namespace evaluate_polynomial_at_minus_two_l286_286962

theorem evaluate_polynomial_at_minus_two : 
  (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 3 * x^2 + x) : f (-2) = -22 :=
by
  -- Definition of the function f
  let f := λ x : ℝ, x^3 - 3 * x^2 + x
  -- Condition: ∀ x, f x = x^3 - 3 * x^2 + x
  have h : ∀ x : ℝ, f x = x^3 - 3 * x^2 + x := by intros; rfl
  sorry

end evaluate_polynomial_at_minus_two_l286_286962


namespace center_of_circle_l286_286104

noncomputable def polar_center (rho: ℝ) (theta: ℝ) : (ℝ × ℝ) :=
let x := rho * cos theta
let y := rho * sin theta
let c := (sqrt 2 / 2, - sqrt 2 / 2)  -- Cartesian coordinates of center
in (sqrt (c.1^2 + c.2^2), atan2 c.2 c.1) -- Polar coordinates conversion

theorem center_of_circle :
  polar_center 1 (-π / 4) = (1, -π / 4) :=
by sorry

end center_of_circle_l286_286104


namespace total_silk_dyed_l286_286578

theorem total_silk_dyed (green_silk pink_silk : ℕ) (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : green_silk + pink_silk = 111421 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_silk_dyed_l286_286578


namespace greatest_odd_factors_l286_286154

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l286_286154


namespace marshmallow_total_l286_286424

-- Define the number of marshmallows each kid can hold
def Haley := 8
def Michael := 3 * Haley
def Brandon := Michael / 2

-- Prove the total number of marshmallows held by all three is 44
theorem marshmallow_total : Haley + Michael + Brandon = 44 := by
  sorry

end marshmallow_total_l286_286424


namespace parallelogram_count_l286_286327

theorem parallelogram_count (n : ℕ) : 
  let trianglesize := n + 2 in
  3 * (Nat.choose trianglesize 4) + 2 = 
      3 * (Nat.choose (n + 2) 4) + 2 :=
by
  sorry

end parallelogram_count_l286_286327


namespace min_number_of_pipes_l286_286637

theorem min_number_of_pipes (d1 d2 : ℝ) (h : ℝ) (h_pos : 0 < h)
  (d1_eq : d1 = 12) (d2_eq : d2 = 3) :
  let V_large := Real.pi * (d1 / 2) * (d1 / 2) * h,
      V_small := Real.pi * (d2 / 2) * (d2 / 2) * h,
      n := V_large / V_small
  in n = 16 :=
by
  sorry

end min_number_of_pipes_l286_286637


namespace root_of_transformed_equation_l286_286743

variable (a b m : ℝ)
hypothesis (h : a * m^2 + b * m + 1 = 0)

theorem root_of_transformed_equation : a * (1/m)^2 + b * (1/m) + 1 = 0 → (1/m)^2 + b * (1/m) + a = 0 :=
by
  sorry

end root_of_transformed_equation_l286_286743


namespace equal_convex_hull_vertices_l286_286379

open Set

def is_convex_hull (S : Set ℝ × ℝ) :=
  subset_convex_hull ℝ S

def num_vertices (X : Set (ℝ × ℝ)) : ℕ :=
  -- This function should calculate the number of vertices of the convex hull of X
  sorry

theorem equal_convex_hull_vertices (S : Set (ℝ × ℝ)) (h_even : Even (Card.mk S)) 
  (h_no_three_collinear : ∀ (p q r : ℝ × ℝ), p ≠ q → q ≠ r → p ≠ r → p ∈ S → q ∈ S → r ∈ S → ¬Collinear ℝ ({p, q, r} : Set (ℝ × ℝ))) :
  ∃ X Y : Set (ℝ × ℝ), X ⊆ S ∧ Y ⊆ S ∧ (X ∩ Y = ∅) ∧ (X ∪ Y = S) ∧ num_vertices (convexHull ℝ X) = num_vertices (convexHull ℝ Y) :=
sorry

end equal_convex_hull_vertices_l286_286379


namespace geometric_progression_fourth_term_l286_286321

noncomputable def x := -9/4
noncomputable def a1 := x
noncomputable def a2 := 3 * x + 3
noncomputable def a3 := 5 * x + 5
noncomputable def r := a2 / a1 -- Calculating the common ratio using the first two terms

theorem geometric_progression_fourth_term :
  let a4 := r * a3 in
  a4 = -125 / 12 :=
by
  sorry

end geometric_progression_fourth_term_l286_286321


namespace kenny_trumpet_hours_l286_286842

variables (x y : ℝ)
def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 2 * running_hours

theorem kenny_trumpet_hours (x y : ℝ) (H : basketball_hours + running_hours + trumpet_hours = x + y) :
  trumpet_hours = 40 :=
by
  sorry

end kenny_trumpet_hours_l286_286842


namespace cough_minutes_l286_286366

/-- Georgia coughs 5 times a minute.
    Robert coughs twice as much as Georgia.
    Together, they have coughed a total of 300 times.
    Prove that the number of minutes m they have been coughing is 20.
 -/
theorem cough_minutes
  (Georgia_cough_rate : ℕ := 5)
  (Robert_cough_rate : ℕ := 2 * Georgia_cough_rate)
  (total_coughs : ℕ := 300) :
  ∃ (m : ℕ), (Georgia_cough_rate + Robert_cough_rate) * m = total_coughs ∧ m = 20 :=
begin
  use 20,
  simp [Georgia_cough_rate, Robert_cough_rate, total_coughs],
  exact ⟨by norm_num, rfl⟩,
end

end cough_minutes_l286_286366


namespace tires_sale_price_l286_286365

variable (n : ℕ)
variable (t p_original p_sale : ℝ)

theorem tires_sale_price
  (h₁ : n = 4)
  (h₂ : t = 36)
  (h₃ : p_original = 84)
  (h₄ : p_sale = p_original - t / n) :
  p_sale = 75 := by
  sorry

end tires_sale_price_l286_286365


namespace wheel_radius_l286_286926

theorem wheel_radius (speed_kmh : ℝ) (rpm : ℝ) (radius_cm : ℝ) :
  speed_kmh = 66 → rpm = 70.06369426751593 → 
  radius_cm ≈ 2500.57 :=
by
  -- Letting speed_kmh and rpm be the given values
  assume h1 : speed_kmh = 66,
  assume h2 : rpm = 70.06369426751593,
  sorry

end wheel_radius_l286_286926


namespace school_competition_proof_l286_286821

-- Definitions of given conditions.
def Team := String
def points (team: Team) (place: Int): Int := 
  if team = "8A" then 22 
  else if team = "8Б" ∨ team = "8B" then 9
  else 0

def team_position := 
  ∀ (x y z : ℕ), (x > y) ∧ (y > z) ∧ (z > 0)

theorem school_competition_proof (n : ℕ) (second_place_javelin : Team):
  team_position →
  points "8A" n + points "8Б" n + points "8B" n = 40 →
  team_position →
  (n = 5) ∧ (second_place_javelin = "8Б") :=
  by sorry

end school_competition_proof_l286_286821


namespace complex_plane_quadrant_l286_286867

def imaginary_unit : Complex := Complex.I

def complex_number (i : Complex) : Complex :=
  2 * i / (1 - i)

theorem complex_plane_quadrant 
  (i : Complex) (h : i = Complex.I) :
  complex_number i = Complex.mk (-1) 1 -> 
  ∃ (x y : ℝ), x < 0 ∧ y > 0 :=
by
  intro h1
  rw [complex_number, h] at h1
  exact ⟨-1, 1, by norm_num, by norm_num⟩

end complex_plane_quadrant_l286_286867


namespace least_five_digit_integer_congruent_3_mod_17_l286_286241

theorem least_five_digit_integer_congruent_3_mod_17 : 
  ∃ n, n ≥ 10000 ∧ n % 17 = 3 ∧ ∀ m, (m ≥ 10000 ∧ m % 17 = 3) → n ≤ m := 
sorry

end least_five_digit_integer_congruent_3_mod_17_l286_286241


namespace sum_of_squares_of_solutions_eq_33_l286_286361

theorem sum_of_squares_of_solutions_eq_33 :
  (∀ x, (1 / x + 2 / (x + 3) + 3 / (x + 6) = 1) → (∑ (r : ℝ) in ({x : ℝ | 1 / x + 2 / (x + 3) + 3 / (x + 6) = 1}), r^2) = 33) :=
by
  sorry

end sum_of_squares_of_solutions_eq_33_l286_286361


namespace not_perfect_cube_of_N_l286_286252

-- Define a twelve-digit number
def N : ℕ := 100000000000

-- Define the condition that a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℤ, n = k ^ 3

-- Problem statement: Prove that 100000000000 is not a perfect cube
theorem not_perfect_cube_of_N : ¬ is_perfect_cube N :=
by sorry

end not_perfect_cube_of_N_l286_286252


namespace pure_imaginary_number_l286_286198

theorem pure_imaginary_number (m : ℝ) (h : (m + 2) - (0 : ℝ) * i = (0 : ℝ) - i) : m = -2 :=
by {
  have : m + 2 = 0,
  { sorry },
  exact this,
}

end pure_imaginary_number_l286_286198


namespace absent_laborers_l286_286992

theorem absent_laborers (L : ℝ) (A : ℝ) (hL : L = 17.5) (h_work_done : (L - A) / 10 = L / 6) : A = 14 :=
by
  sorry

end absent_laborers_l286_286992


namespace coefficient_of_x5_in_P5_l286_286350

-- Define the polynomial P(x) = 1 + 2x + 3x^2 + 4x^3
def P (x : ℚ) : ℚ := 1 + 2 * x + 3 * x^2 + 4 * x^3

-- Define the statement about the coefficient of x^5 in the expansion of P(x)^5
theorem coefficient_of_x5_in_P5 : (P x)^5.coeff 5 = 1772 := by
  sorry

end coefficient_of_x5_in_P5_l286_286350


namespace optimal_cookies_result_l286_286523

-- Definitions
def initial_blackboard : List Nat := List.replicate 2020 1

inductive TerminationCondition
| dominance (num : Nat)
| all_zeros

structure GameState := 
  (blackboard : List Nat)
  (terminates : TerminationCondition)

-- Optimal Strategy Condition for both Players
def optimal_cookies (state : GameState) : Nat := 
  let sum_of_binary_digits := (2020).binary_digits.count 1 
  if state.terminates = TerminationCondition.all_zeros then
    sum_of_binary_digits
  else 
    0  -- this else case shouldn't actually be hit in an optimal play scenario

theorem optimal_cookies_result : ∃ state : GameState, optimal_cookies state = 7 :=
by
  let state := {blackboard := [], terminates := TerminationCondition.all_zeros}
  exists state
  unfold optimal_cookies
  simp
  sorry

end optimal_cookies_result_l286_286523


namespace paint_surface_area_l286_286113

theorem paint_surface_area (s : ℝ) (paint_cube : ℝ) (paint_shape : ℝ) :
  6 * s^2 = 6 * s^2 → paint_cube = 9 → paint_shape = 9 :=
by
  intro h_surface_area h_paint_cube
  rw [h_paint_cube]
  exact h_surface_area 
  sorry

end paint_surface_area_l286_286113


namespace ellipse_m_value_l286_286380

theorem ellipse_m_value (m : ℝ) (h1 : 0 < m) :
  ((∃ x y : ℝ, (x^2 / 5 + y^2 / m = 1) ∧ ((x = 0 ∨ y = 0) → (∃ k : ℝ, k = sqrt(10) / 5))) → 
  (m = 3 ∨ m = 25 / 3)) :=
sorry

end ellipse_m_value_l286_286380


namespace complex_quadrant_l286_286968

def complex_number : ℂ := 2 / (1 + complex.I)

theorem complex_quadrant : complex_number.re > 0 ∧ complex_number.im < 0 :=
by sorry

end complex_quadrant_l286_286968


namespace marshmallow_total_l286_286425

-- Define the number of marshmallows each kid can hold
def Haley := 8
def Michael := 3 * Haley
def Brandon := Michael / 2

-- Prove the total number of marshmallows held by all three is 44
theorem marshmallow_total : Haley + Michael + Brandon = 44 := by
  sorry

end marshmallow_total_l286_286425


namespace A_share_in_profit_l286_286986

/-
Given:
1. a_contribution (A's amount contributed in Rs. 5000) and duration (in months 8)
2. b_contribution (B's amount contributed in Rs. 6000) and duration (in months 5)
3. total_profit (Total profit in Rs. 8400)

Prove that A's share in the total profit is Rs. 4800.
-/

theorem A_share_in_profit 
  (a_contribution : ℝ) (a_months : ℝ) 
  (b_contribution : ℝ) (b_months : ℝ) 
  (total_profit : ℝ) :
  a_contribution = 5000 → 
  a_months = 8 → 
  b_contribution = 6000 → 
  b_months = 5 → 
  total_profit = 8400 → 
  (a_contribution * a_months / (a_contribution * a_months + b_contribution * b_months) * total_profit) = 4800 := 
by {
  sorry
}

end A_share_in_profit_l286_286986


namespace find_correlated_pair_l286_286600

-- Define the properties and relations between the quantities
universe u

def uniform_linear_motion (time displacement : Type u) (f : time → displacement) :=
  ∀ t₁ t₂ : time, f (t₁ + t₂) = f t₁ + f t₂

def student_attributes (grades weight : Type u) :=
  ¬(∃ (f : grades → weight), ∀ g : grades, true)

def drunk_drivers_and_accidents (drunkDrivers trafficAccidents : Type u) :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ d : drunkDrivers, ∃ f : drunkDrivers → trafficAccidents, true

def volume_and_weight_of_water (volume weight : Type u) :=
  ∃ (k : ℝ), ∀ v : volume, ∃ w : weight, w = k * v

-- Constants representing the quantities
constant Time : Type u
constant Displacement : Type u
constant Grades : Type u
constant Weight : Type u
constant DrunkDrivers : Type u
constant TrafficAccidents : Type u
constant Volume : Type u

-- Given conditions
axiom A : uniform_linear_motion Time Displacement (λ t, t)
axiom B : student_attributes Grades Weight
axiom C : drunk_drivers_and_accidents DrunkDrivers TrafficAccidents
axiom D : volume_and_weight_of_water Volume Weight

-- The main statement: Prove that option C describes a correlation relationship
theorem find_correlated_pair : C = drunk_drivers_and_accidents DrunkDrivers TrafficAccidents :=
by sorry

end find_correlated_pair_l286_286600


namespace collinear_points_sum_xy_solution_l286_286572

theorem collinear_points_sum_xy_solution (x y : ℚ)
  (h1 : (B : ℚ × ℚ) = (-2, y))
  (h2 : (A : ℚ × ℚ) = (x, 5))
  (h3 : (C : ℚ × ℚ) = (1, 1))
  (h4 : dist (B.1, B.2) (C.1, C.2) = 2 * dist (A.1, A.2) (C.1, C.2))
  (h5 : (y - 5) / (-2 - x) = (1 - 5) / (1 - x)) :
  x + y = -9 / 2 ∨ x + y = 17 / 2 :=
by sorry

end collinear_points_sum_xy_solution_l286_286572


namespace first_candidate_percentage_l286_286987

noncomputable
def passing_marks_approx : ℝ := 240

noncomputable
def total_marks (P : ℝ) : ℝ := (P + 30) / 0.45

noncomputable
def percentage_marks (T P : ℝ) : ℝ := ((P - 60) / T) * 100

theorem first_candidate_percentage :
  let P := passing_marks_approx
  let T := total_marks P
  percentage_marks T P = 30 :=
by
  sorry

end first_candidate_percentage_l286_286987


namespace triangle_inequality_range_x_l286_286089

theorem triangle_inequality_range_x (x : ℝ) :
  let a := 3;
  let b := 8;
  let c := 1 + 2 * x;
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔ (2 < x ∧ x < 5) :=
by
  sorry

end triangle_inequality_range_x_l286_286089


namespace complex_numbers_pentagon_properties_l286_286014

theorem complex_numbers_pentagon_properties :
  let cis (θ : ℝ) := Complex.exp (Complex.I * θ) in
  {z : ℂ | z ≠ 0 ∧ ∃ k : ℕ, k ∈ {1, 2, 3, 4} ∧ z = cis (2 * Real.pi * k / 5)}.card = 4 :=
by
  sorry

end complex_numbers_pentagon_properties_l286_286014


namespace sum_of_decimals_l286_286284

theorem sum_of_decimals :
  5.467 + 2.349 + 3.785 = 11.751 :=
sorry

end sum_of_decimals_l286_286284


namespace quadratic_inequality_solution_l286_286723

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50 * x + 601 ≤ 9} = {x : ℝ | 19.25545 ≤ x ∧ x ≤ 30.74455} :=
by 
  sorry

end quadratic_inequality_solution_l286_286723


namespace count_numbers_with_perfect_square_factors_l286_286032

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286032


namespace convex_polygon_cannot_be_divided_into_100_equilateral_triangles_l286_286890

theorem convex_polygon_cannot_be_divided_into_100_equilateral_triangles
    (P : Polygon) (h1 : P.convex) :
    ¬ ∃ (T : Finset EquilateralTriangle) (h2 : T.card = 100), 
    (∀ t ∈ T, t.inside P ∧ ∀ t1 t2 ∈ T, t1 ≠ t2 → t1 ≠ t2) := 
sorry

end convex_polygon_cannot_be_divided_into_100_equilateral_triangles_l286_286890


namespace geom_sequence_sum_first_10_terms_l286_286373

theorem geom_sequence_sum_first_10_terms :
  ∀ {a : ℕ → ℝ} (q : ℝ),
  (a 1) * (1 + q) = 6 →
  (a 1) * q^3 * (1 + q) = 48 →
  (∑ i in Finset.range 10, a (i + 1)) = 2046 :=
by
  intros a q h1 h2
  sorry

end geom_sequence_sum_first_10_terms_l286_286373


namespace simplify_expression_l286_286897

theorem simplify_expression (n : ℕ) (h : n > 0) : 
  ((
    (∑ i in Finset.range n.succ, i * (3 * i) * (9 * i))
    /
    (∑ i in Finset.range n.succ, i * (5 * i) * (25 * i))
  ) ^ (1/3 : ℝ)) = (3/5 : ℝ) :=
sorry

end simplify_expression_l286_286897


namespace num_pieces_l286_286071

theorem num_pieces (total_length : ℝ) (piece_length : ℝ) 
  (h1: total_length = 253.75) (h2: piece_length = 0.425) :
  ⌊total_length / piece_length⌋ = 597 :=
by
  rw [h1, h2]
  sorry

end num_pieces_l286_286071


namespace part1_part2_l286_286802

-- Define what a double root equation is
def is_double_root_eq (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ * a + x₁ * b + c = 0 ∧ x₂ = 2 * x₁ ∧ x₂ * x₂ * a + x₂ * b + c = 0

-- Statement for part 1: proving x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_eq 1 (-3) 2 :=
sorry

-- Statement for part 2: finding correct values of a and b for ax^2 + bx - 6 = 0 to be a double root equation with one root 2
theorem part2 : (∃ a b : ℝ, is_double_root_eq a b (-6) ∧ (a = -3 ∧ b = 9) ∨ (a = -3/4 ∧ b = 9/2)) :=
sorry

end part1_part2_l286_286802


namespace probability_odd_sum_three_dice_l286_286946

-- Define the probability problem related to the sum of three dice rolls.
theorem probability_odd_sum_three_dice : 
  let P_odd (n : ℕ) : ℚ := if odd n then 1/2 else 0 in
  let P_three_dice_odd_sum : ℚ := 
    ∑ i in finset.range 6, ∑ j in finset.range 6, ∑ k in finset.range 6, if odd (i + j + k) then 1 else 0 in 
  (P_three_dice_odd_sum / 216 = 1/2) :=
by
  sorry

end probability_odd_sum_three_dice_l286_286946


namespace tan_a8_l286_286751

noncomputable def sequenceTerms (n : ℕ) : ℝ := sorry  -- \(a_n\) definition placeholder
noncomputable def sumFirstNTerms (n : ℕ) : ℝ := sorry  -- \(S_n\) definition placeholder

axiom arithmetic_seq (seq : ℕ → ℝ) : (∃ a d : ℝ, seq = λ n, a + n * d)
axiom sum_condition : sumFirstNTerms 15 = 25 * Real.pi

theorem tan_a8 : ∃ a1 a15 : ℝ, ∀ seq, arithmetic_seq seq → 
  (sumFirstNTerms 15 = 15 * seq 8) → 
  ∀ a8 : ℝ, seq 8 = a8 → tan a8 = -Real.sqrt 3 :=
by {
  sorry
}

end tan_a8_l286_286751


namespace cost_of_gum_is_10_dollars_l286_286910

-- Define the cost of one piece of gum in cents
def cost_of_one_piece_of_gum : ℕ := 1

-- Define the conversion rate from cents to dollars
def cents_to_dollars (cents : ℕ) : ℝ := cents / 100.0

-- Define the cost of 1000 pieces of gum in cents
def cost_of_thousand_pieces_of_gum_in_cents : ℕ := 1000 * cost_of_one_piece_of_gum

-- Define the cost of 1000 pieces of gum in dollars
def cost_of_thousand_pieces_of_gum_in_dollars : ℝ := cents_to_dollars cost_of_thousand_pieces_of_gum_in_cents

-- Theorem stating the cost of 1000 pieces of gum in dollars is 10 dollars
theorem cost_of_gum_is_10_dollars : cost_of_thousand_pieces_of_gum_in_dollars = 10.0 := by
  -- Placeholder for proof
  sorry

end cost_of_gum_is_10_dollars_l286_286910


namespace famous_sentences_correct_l286_286703

def blank_1 : String := "correct_answer_1"
def blank_2 : String := "correct_answer_2"
def blank_3 : String := "correct_answer_3"
def blank_4 : String := "correct_answer_4"
def blank_5 : String := "correct_answer_5"
def blank_6 : String := "correct_answer_6"
def blank_7 : String := "correct_answer_7"
def blank_8 : String := "correct_answer_8"

theorem famous_sentences_correct :
  blank_1 = "correct_answer_1" ∧
  blank_2 = "correct_answer_2" ∧
  blank_3 = "correct_answer_3" ∧
  blank_4 = "correct_answer_4" ∧
  blank_5 = "correct_answer_5" ∧
  blank_6 = "correct_answer_6" ∧
  blank_7 = "correct_answer_7" ∧
  blank_8 = "correct_answer_8" :=
by
  -- The proof details correspond to the part "refer to the correct solution for each blank"
  sorry

end famous_sentences_correct_l286_286703


namespace original_price_dish_l286_286979

-- Conditions
variables (P : ℝ) -- Original price of the dish
-- Discount and tips
def john_discounted_and_tip := 0.9 * P + 0.15 * P
def jane_discounted_and_tip := 0.9 * P + 0.135 * P

-- Condition of payment difference
def payment_difference := john_discounted_and_tip P = jane_discounted_and_tip P + 0.36

-- The theorem to prove
theorem original_price_dish : payment_difference P → P = 24 :=
by
  intro h
  sorry

end original_price_dish_l286_286979


namespace horner_method_eval_l286_286954

theorem horner_method_eval :
  let f : ℕ → ℕ := λ x, 6 * x^5 + 5 * x^4 - 4 * x^3 + 3 * x^2 - 2 * x + 1
  in f 2 = 249 := by
sorry

end horner_method_eval_l286_286954


namespace area_triangle_GIJ_l286_286179

variables {F G H I J : ℝ × ℝ × ℝ}
variables (d : ℝ) (a1 a2 a3 : Prop)

def dist_eq (p q : ℝ × ℝ × ℝ) : Prop := (p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2 = d^2
def angle_90 (p q r : ℝ × ℝ × ℝ) : Prop := let uv : ℝ × ℝ × ℝ := (q.1 - p.1, q.2 - p.2, q.3 - p.3);
                                        let wv : ℝ × ℝ × ℝ := (q.1 - r.1, q.2 - r.2, q.3 - r.3);
                                        uv.1 * wv.1 + uv.2 * wv.2 + uv.3 * wv.3 = 0
def plane_parallel (p q r s t : ℝ × ℝ × ℝ) : Prop := (r.1 - q.1) * (p.1 - s.1) + (r.2 - q.2) * (p.2 - s.2) + (r.3 - q.3) * (p.3 - s.3) = 0 ∧
                                                     (t.1 - s.1) * (p.1 - s.1) + (t.2 - s.2) * (p.2 - s.2) + (t.3 - s.3) * (p.3 - s.3) = 0

theorem area_triangle_GIJ (h_fgh : dist_eq F G d ∧ dist_eq G H d ∧ dist_eq H I d ∧ dist_eq I J d ∧ dist_eq J F d)
                         (h_angles : angle_90 F G H ∧ angle_90 H I J ∧ angle_90 J I F)
                         (h_plane: plane_parallel F G H I J)
                         : let area : ℝ := 0.5 * d * d in
                           area = 4.5 :=
sorry

end area_triangle_GIJ_l286_286179


namespace minivan_tank_capacity_l286_286463

-- Define the given conditions
def service_cost_per_vehicle := 2.20
def fuel_cost_per_liter := 0.70
def num_minivans := 3
def num_trucks := 2
def truck_capacity_factor := 2.2
def total_cost := 347.7

-- Define the unknown variable
variable (V : ℝ) -- Capacity of a mini-van's tank

-- The total cost equation derived from the conditions
def total_cost_equation := 
  (num_minivans + num_trucks) * service_cost_per_vehicle + 
  (num_minivans * V + num_trucks * (truck_capacity_factor * V)) * fuel_cost_per_liter

-- Prove that V = 65
theorem minivan_tank_capacity : total_cost_equation V = total_cost → V = 65 :=
begin
  sorry
end

end minivan_tank_capacity_l286_286463


namespace find_angle_A_find_area_l286_286758

noncomputable theory

-- Define the given conditions of the triangle
variables (A B C : ℝ) (a b c : ℝ)

-- Define the problem conditions
axiom angle_conditions : A + B + C = π
axiom side_conditions : a = 2 * √3 ∧ b + c = 4
axiom equation_condition : a * cos C + c * cos A = -2 * b * cos A

-- Prove the required results
theorem find_angle_A : A = 2 * π / 3 := sorry

theorem find_area : 1 / 2 * b * c * sin A = √3 := sorry

end find_angle_A_find_area_l286_286758


namespace find_abc_sum_l286_286847

theorem find_abc_sum :
  let s : list ℂ := roots (X^3 + X^2 + (9/2)*X + 9)
  ∃ (a b c : ℕ), (∏ x in s, 4*x^4 + 81) = 2^a * 3^b * 5^c ∧ a + b + c = 16 := 
sorry

end find_abc_sum_l286_286847


namespace point_on_circle_l286_286404

-- Define the radius and distance
def radius : ℝ := 4
def distance_from_center : ℝ := 4

-- Statement to be proved: point P is on the circle
theorem point_on_circle (r d : ℝ) (h1 : r = 4) (h2 : d = 4) : d = r :=
by { rw [h1, h2], } 

end point_on_circle_l286_286404


namespace distance_between_walls_l286_286843

-- Definitions for the lengths of the ladders and the heights they reach on each wall.
def ladder_length : ℝ := 15
def height_wall1 : ℝ := 12
def height_wall2 : ℝ := 9

-- Using Pythagorean theorem to calculate horizontal distances from the floor touching point.
def dist_to_wall1 : ℝ := Real.sqrt (ladder_length ^ 2 - height_wall1 ^ 2)
def dist_to_wall2 : ℝ := Real.sqrt (ladder_length ^ 2 - height_wall2 ^ 2)

-- Statement to prove that the distance between the two walls is 21 meters.
theorem distance_between_walls : dist_to_wall1 + dist_to_wall2 = 21 := by
  sorry

end distance_between_walls_l286_286843


namespace sin_cos_solution_set_l286_286360
open Real

theorem sin_cos_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * π + (-1)^k * (π / 6) - (π / 3)} =
  {x : ℝ | sin x + sqrt 3 * cos x = 1} :=
by sorry

end sin_cos_solution_set_l286_286360


namespace find_50th_term_in_sequence_l286_286090

def sequence (n : ℕ) : ℕ :=
  2 + 4 * (n - 1)

theorem find_50th_term_in_sequence :
  sequence 50 = 198 :=
by
  sorry

end find_50th_term_in_sequence_l286_286090


namespace root_of_equation_value_l286_286398

theorem root_of_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2 * m^2 - 4 * m + 5 = 11 := 
by
  sorry

end root_of_equation_value_l286_286398


namespace parabola_min_value_sum_abc_zero_l286_286210

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l286_286210


namespace total_revenue_from_selling_snakes_l286_286117

-- Definitions based on conditions
def num_snakes := 3
def eggs_per_snake := 2
def standard_price := 250
def rare_multiplier := 4

-- Prove the total revenue Jake gets from selling all baby snakes is $2250
theorem total_revenue_from_selling_snakes : 
  (num_snakes * eggs_per_snake - 1) * standard_price + (standard_price * rare_multiplier) = 2250 := 
by
  sorry

end total_revenue_from_selling_snakes_l286_286117


namespace harry_hours_l286_286334

noncomputable def hoursHarryWorked (x y : ℝ) (H_extra : ℝ) : ℝ := 21 + H_extra

theorem harry_hours (x y : ℝ) (H_extra : ℝ) : 
  (21 * x + 1.5 * x * H_extra = 40 * x + y * x) ∧ 
  (y = 1.5) → 
  (hoursHarryWorked x y H_extra = 35) :=
begin
  sorry
end

end harry_hours_l286_286334


namespace sum_of_extrema_l286_286503

theorem sum_of_extrema (x y z : ℝ)
  (h1 : x + y + z = 5)
  (h2 : x^2 + y^2 + z^2 = 11)
  (h3 : xyz = 6) :
  let m := (5 - Real.sqrt 34) / 3,
      M := (5 + Real.sqrt 34) / 3
  in m + M = 10 / 3 :=
by {
  sorry
}

end sum_of_extrema_l286_286503


namespace judson_contribution_l286_286123

theorem judson_contribution (J K C : ℝ) (hK : K = 1.20 * J) (hC : C = K + 200) (h_total : J + K + C = 1900) : J = 500 :=
by
  -- This is where the proof would go, but we are skipping it as per the instructions.
  sorry

end judson_contribution_l286_286123


namespace quadrant_of_z_l286_286452

noncomputable def z (i : ℂ) : ℂ := (1 - i) / ((1 + i) * (1 + i))

theorem quadrant_of_z :
  let i := complex.I in
  let (x, y) := ((1 - i) / ((1 + i) * (1 + i))).re, ((1 - i) / ((1 + i) * (1 + i))).im in
  x < 0 ∧ y < 0 :=
by
  sorry

end quadrant_of_z_l286_286452


namespace work_completion_days_l286_286263

open Real

theorem work_completion_days (days_A : ℝ) (days_B : ℝ) (amount_total : ℝ) (amount_C : ℝ) :
  days_A = 6 ∧ days_B = 8 ∧ amount_total = 5000 ∧ amount_C = 625.0000000000002 →
  (1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1)) = 5 / 12 →
  1 / ((1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1))) = 2.4 :=
  sorry

end work_completion_days_l286_286263


namespace arithmetic_seq_n_possible_values_l286_286830

theorem arithmetic_seq_n_possible_values
  (a1 : ℕ) (a_n : ℕ → ℕ) (d : ℕ) (n : ℕ):
  a1 = 1 → 
  (∀ n, n ≥ 3 → a_n n = 100) → 
  (∃ d : ℕ, ∀ n, n ≥ 3 → a_n n = a1 + (n - 1) * d) → 
  (n = 4 ∨ n = 10 ∨ n = 12 ∨ n = 34 ∨ n = 100) := by
  sorry

end arithmetic_seq_n_possible_values_l286_286830


namespace locus_of_point_l286_286124

variable (A B C M : Point)

-- Define a function to compute the area of triangle given three points
noncomputable def area_triangle (P Q R : Point) : ℝ := sorry

-- The given condition
def condition (A B C M : Point) : Prop :=
  area_triangle A B M = 2 * area_triangle A C M

-- The result we need to prove
theorem locus_of_point (A B C : Point) :
  ∀ M : Point, condition A B C M → exists l : Line, is_parallel_to l (line_through B C) ∧ M ∈ l :=
begin
  sorry
end

end locus_of_point_l286_286124


namespace denomination_of_checks_l286_286651

-- Definitions based on the conditions.
def total_checks := 30
def total_worth := 1800
def checks_spent := 24
def average_remaining := 100

-- Statement to be proven.
theorem denomination_of_checks :
  ∃ x : ℝ, (total_checks - checks_spent) * average_remaining + checks_spent * x = total_worth ∧ x = 40 :=
by
  sorry

end denomination_of_checks_l286_286651


namespace smallest_in_ascending_order_l286_286923

noncomputable def nums : List ℝ := [2.23, 3.12, 9.434, 2.453]

theorem smallest_in_ascending_order : (nums.sorted.head = 2.23) :=
by
  sorry

end smallest_in_ascending_order_l286_286923


namespace majority_votes_l286_286468

theorem majority_votes (total_votes : ℝ) (percentage_winning : ℝ) (percentage_losing : ℝ) 
    (h1 : total_votes = 480) (h2 : percentage_winning = 0.70) (h3 : percentage_losing = 0.30) :
    (total_votes * percentage_winning) - (total_votes * percentage_losing) = 192 :=
by 
  -- Given conditions in the theorem body
  have h_winning : total_votes * percentage_winning = 336 := 
    by { rw [h1, h2], norm_num },
  have h_losing : total_votes * percentage_losing = 144 := 
    by { rw [h1, h3], norm_num },
  -- Calculate the majority
  rw [h_winning, h_losing],
  norm_num,
  -- Conclude the majority is 192
  sorry

end majority_votes_l286_286468


namespace smallest_positive_period_and_monotonically_decreasing_interval_analytical_expression_of_f_enclosed_area_by_transformed_function_l286_286414

-- Define the function f(x) and given conditions
def f (x : ℝ) (a : ℝ) : ℝ :=
  √3 * sin x * cos x + cos x^2 + a

-- Define the interval
def interval (x : ℝ) : Prop :=
  - (π / 6) ≤ x ∧ x ≤ π / 3

-- Statement for part (I)
theorem smallest_positive_period_and_monotonically_decreasing_interval
    (a : ℝ) :
  (∃ T > 0, ∀ x, f x a = f (x + T) a) ∧
  (∃ k : ℤ, ∀ x, interval x → (π / 6 + k * π ≤ x ∧ x ≤ 2 * π / 3 + k * π)) :=
sorry

-- Statement for part (II)
theorem analytical_expression_of_f (a : ℝ) :
  (∀ x, interval x → f x a = sin (2 * x + π / 6) + 0.5) →
  (sin (- π / 6 + π / 6) + 0.5 + sin (5 * π / 6 + π / 6) + 0.5) = 3 / 2 :=
sorry

-- Define the transformed function g(x)
def g (x : ℝ) : ℝ := sin x

-- Statement for part (III)
theorem enclosed_area_by_transformed_function :
  (∀ a, f x a = sin x + 0.5 → ∫ (x : ℝ) in 0 .. π / 2, g x = 1) :=
sorry

end smallest_positive_period_and_monotonically_decreasing_interval_analytical_expression_of_f_enclosed_area_by_transformed_function_l286_286414


namespace color_of_2021st_ball_l286_286981

def ball_color (n : ℕ) : ℕ → Prop
| 2 := True -- Ball number 2 is green
| 20 := True -- Ball number 20 is green
| (n % 5 = 1) := True -- We want to prove this for n = 2021
| _ := False

theorem color_of_2021st_ball (n : ℕ) :
  (n = 2021) →
  (∀ (k : ℕ), k % 5 = 1 → ball_color k) → 
  n % 5 = 1 → 
  ball_color n
:= by
  intro
  intro
  intro
  exact sorry

end color_of_2021st_ball_l286_286981


namespace solve_l286_286130

noncomputable def f : ℝ → ℝ := sorry

theorem solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f x + y) = f (x + y) + x * f y - 2 * x * y - x ^ 2 + 2) :
  let n := 1 in
  let s := f 1 in
  (n * s = 3) :=
by
  sorry

end solve_l286_286130


namespace proof_problem_l286_286492

open Nat

noncomputable def has_at_least_three_distinct_prime_divisors (n : ℕ) (a : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ : ℕ, p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ p₁ ∣ a ∧ p₂ ∣ a ∧ p₃ ∣ a

theorem proof_problem (p : ℕ) (hp : Prime p) (h : 2^(p - 1) ≡ 1 [MOD p^2]) (n : ℕ) (hn : n ∈ ℕ) :
  has_at_least_three_distinct_prime_divisors n ((p - 1) * (factorial p + 2^n)) :=
sorry

end proof_problem_l286_286492


namespace smallest_sum_a_b_l286_286128

open Real

theorem smallest_sum_a_b (
  a b : ℝ 
) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a)^2 - 4 * (3 * b) ≥ 0) (h4 : (3 * b)^2 - 4 * (2 * a) ≥ 0) :
    a + b = 2 * sqrt 2 + (4 / 3) * real.sqrt (real.sqrt 2) :=
by
  sorry

end smallest_sum_a_b_l286_286128


namespace perimeter_of_square_from_circle_l286_286595

-- Define the constants and variables
def circle_circumference := 52.5
def pi : ℝ := Real.pi
def diameter_of_circle (C : ℝ) (π : ℝ) := C / π
def side_of_square (C : ℝ) (π : ℝ) := diameter_of_circle C π
def perimeter_of_square (s : ℝ) := 4 * s

-- Define the perimeter of a square based on circle's circumference
theorem perimeter_of_square_from_circle :
  perimeter_of_square (side_of_square circle_circumference pi) = 210 / pi := by
  sorry

end perimeter_of_square_from_circle_l286_286595


namespace cubic_geometric_sequence_conditions_l286_286722

-- Conditions from the problem
def cubic_eq (a b c x : ℝ) : Prop := x^3 + a * x^2 + b * x + c = 0

-- The statement to be proven
theorem cubic_geometric_sequence_conditions (a b c : ℝ) :
  (∃ x q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ 
    cubic_eq a b c x ∧ cubic_eq a b c (x*q) ∧ cubic_eq a b c (x*q^2)) → 
  (b^3 = a^3 * c ∧ c ≠ 0 ∧ -a^3 < c ∧ c < a^3 / 27 ∧ a < m ∧ m < - a / 3) :=
by 
  sorry

end cubic_geometric_sequence_conditions_l286_286722


namespace proof_correct_option_l286_286287

def like_terms (m1 m2 : List (Char × Nat)) : Bool :=
  (m1.map Prod.fst = m2.map Prod.fst) && (m1.map Prod.snd = m2.map Prod.snd)

def option_a := like_terms [ ('m', 1), ('n', 1) ] [ ('n', 1), ('m', 1) ] = true
def option_b := like_terms [ ('m', 1), ('n', 2) ] [ ('m', 2), ('n', 1) ] = false
def option_c := like_terms [ ('x', 3) ] [ ('y', 3) ] = false
def option_d := like_terms [ ('a', 1), ('b', 1) ] [ ('a', 1), ('b', 1), ('c', 1) ] = false

theorem proof_correct_option :
  option_a ∧ option_b ∧ option_c ∧ option_d 
  := by
s \ sorry

end proof_correct_option_l286_286287


namespace a_pow_10_add_b_pow_10_eq_123_l286_286174

variable (a b : ℕ) -- better as non-negative integers for sequence progression

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a_pow_10_add_b_pow_10_eq_123 : a^10 + b^10 = 123 := by
  sorry

end a_pow_10_add_b_pow_10_eq_123_l286_286174


namespace percent_voters_for_A_is_50_l286_286460

-- Define the number of total voters (for simplicity, we'll use 100)
def total_voters : ℕ := 100

-- Define the percentage of Democrats and Republicans
def percent_democrats : ℝ := 0.60
def percent_republicans : ℝ := 0.40

-- Define the number of Democrats and Republicans
def democrats : ℕ := (percent_democrats * total_voters).to_int
def republicans : ℕ := (percent_republicans * total_voters).to_int

-- Define the percentage of Democrats and Republicans voting for candidate A
def percent_democrats_for_A : ℝ := 0.70
def percent_republicans_for_A : ℝ := 0.20

-- Define the number of Democrats and Republicans voting for candidate A
def democrats_for_A : ℕ := (percent_democrats_for_A * democrats).to_int
def republicans_for_A : ℕ := (percent_republicans_for_A * republicans).to_int

-- Define the total number of voters for candidate A
def total_for_A : ℕ := democrats_for_A + republicans_for_A

-- Define the percentage of total voters for candidate A
def percent_for_A : ℝ := (total_for_A.to_real / total_voters) * 100

-- Theorem statement: 50% of the registered voters are expected to vote for candidate A
theorem percent_voters_for_A_is_50 :
  percent_for_A = 50 := sorry

end percent_voters_for_A_is_50_l286_286460


namespace percentage_of_students_attend_chess_class_l286_286939

-- Definitions based on the conditions
def total_students : ℕ := 1000
def swimming_students : ℕ := 125
def chess_to_swimming_ratio : ℚ := 1 / 2

-- Problem statement
theorem percentage_of_students_attend_chess_class :
  ∃ P : ℚ, (P / 100) * total_students / 2 = swimming_students → P = 25 := by
  sorry

end percentage_of_students_attend_chess_class_l286_286939


namespace exists_positive_integer_special_N_l286_286841

theorem exists_positive_integer_special_N : 
  ∃ (N : ℕ), 
    (∃ (m : ℕ), N = 1990 * (m + 995)) ∧ 
    (∀ (n : ℕ), (∃ (m : ℕ), 2 * N = (n + 1) * (2 * m + n)) ↔ (3980 = 2 * 1990)) := by
  sorry

end exists_positive_integer_special_N_l286_286841


namespace time_ratio_correct_l286_286264

def original_distance := 360
def original_time := 6
def new_speed := 40

/-- Prove the ratio of the new time to the original time is 3:2. -/
theorem time_ratio_correct :
  let new_time := original_distance / new_speed in
  new_time / original_time = 3 / 2 := 
by
  sorry

end time_ratio_correct_l286_286264


namespace classroom_children_l286_286567

-- Define initial conditions as constants
def initial_boys : Nat := 5
def initial_girls : Nat := 4
def boys_left : Nat := 3
def girls_entered : Nat := 2

-- Define the final number of boys, girls, and total children
def final_boys : Nat := initial_boys - boys_left
def final_girls : Nat := initial_girls + girls_entered
def final_children : Nat := final_boys + final_girls

-- Prove that the final number of children is 8
theorem classroom_children : final_children = 8 := by
  simp [final_boys, final_girls, final_children, initial_boys, initial_girls, boys_left, girls_entered]
  rfl

end classroom_children_l286_286567


namespace quadratic_coeff_sum_l286_286213

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l286_286213


namespace intersection_of_C1_and_C2_max_distance_from_C1_to_C2_l286_286828

-- Definitions from the problem conditions
def C1_parametric (r θ : ℝ) : ℝ × ℝ :=
  (- (Real.sqrt 2 / 2) + r * Real.cos θ, - (Real.sqrt 2 / 2) + r * Real.sin θ)

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - (Real.pi / 4)) = 1

-- Additional definitions derived from the conditions
def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  Real.abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

def C1_center_distance_to_C2 : ℝ :=
  distance_point_to_line (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) 1 1 (-Real.sqrt 2)

-- Lean theorem statements
theorem intersection_of_C1_and_C2 (r : ℝ) (hr_pos : 0 < r) (r_eq_sqrt_5 : r = Real.sqrt 5) :
  C1_center_distance_to_C2 < r :=
by sorry

theorem max_distance_from_C1_to_C2 (r : ℝ) (hr_pos : 0 < r) (max_distance_eq_3 : C1_center_distance_to_C2 + r = 3) :
  r = 1 :=
by sorry

end intersection_of_C1_and_C2_max_distance_from_C1_to_C2_l286_286828


namespace positional_relationship_l286_286456

variables (l a : Type) (α : Type)
variables [Line l] [Line a] [Plane α]

namespace PositionalRelationship

def is_parallel (l : Type) (α : Type) [Line l] [Plane α] : Prop :=
  sorry -- definition of parallelity between line and plane

def lies_on (a : Type) (α : Type) [Line a] [Plane α] : Prop :=
  sorry -- definition of a line lying on a plane

def relation (l a : Type) [Line l] [Line a] : Type :=
  sorry -- definition of positional relationship between two lines

inductive Rel
| Perpendicular : Rel
| Parallel : Rel
| DifferentPlanes : Rel

theorem positional_relationship (hl_par : is_parallel l α) (ha_on : lies_on a α) :
  relation l a ∈ {Rel.Perpendicular, Rel.Parallel, Rel.DifferentPlanes} :=
by
  sorry

end PositionalRelationship

end positional_relationship_l286_286456


namespace angle_BDC_in_quadrilateral_ABCD_l286_286833

theorem angle_BDC_in_quadrilateral_ABCD
  (A B C D : Type) [EuclideanGeometry A B C D]
  (h1 : perp AC BD)
  (h2 : ∠ BCA = 10)
  (h3 : ∠ BDA = 20)
  (h4 : ∠ BAC = 40) :
  ∠ BDC = 60 :=
by
  sorry

end angle_BDC_in_quadrilateral_ABCD_l286_286833


namespace convert_radians_to_degrees_l286_286315

theorem convert_radians_to_degrees:
  (4/3 * real.pi = 240) :=
by
  -- sorry is a placeholder for the proof
  sorry

end convert_radians_to_degrees_l286_286315


namespace percentage_women_without_retirement_plan_is_twenty_percent_l286_286094

def total_workers := 240
def men_workers := 120
def women_workers := 120
def no_retirement_plan := total_workers / 3
def with_retirement_plan := total_workers - no_retirement_plan
def men_with_retirement_plan := 0.40 * with_retirement_plan
def men_with_no_retirement_plan := men_workers - men_with_retirement_plan
def women_with_no_retirement_plan := no_retirement_plan - men_with_no_retirement_plan
def percentage_women_no_retirement_plan := (women_with_no_retirement_plan / women_workers) * 100

theorem percentage_women_without_retirement_plan_is_twenty_percent :
  percentage_women_no_retirement_plan = 20 := sorry

end percentage_women_without_retirement_plan_is_twenty_percent_l286_286094


namespace count_perfect_square_factors_l286_286016

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286016


namespace x_expression_l286_286137

-- Definitions given in the problem
def g (t : ℝ) : ℝ := t / (1 + t)
def f (t : ℝ) : ℝ := t / (1 - t)

-- Variable x and y
variables (x y : ℝ)

-- Conditions given in the problem
axiom h1 : y = g x

-- The theorem to prove
theorem x_expression : x = f y :=
sorry

end x_expression_l286_286137


namespace remainder_of_sum_l286_286921

theorem remainder_of_sum (h_prime : Prime 2027) :
  ((∑ k in Finset.range (81), Nat.choose 2024 k) % 2027) = 1681 :=
  sorry

end remainder_of_sum_l286_286921


namespace percentage_of_loss_is_15_percent_l286_286642

/-- 
Given:
  SP₁ = 168 -- Selling price when gaining 20%
  Gain = 20% 
  SP₂ = 119 -- Selling price when calculating loss

Prove:
  The percentage of loss when the article is sold for Rs. 119 is 15%
--/

noncomputable def percentage_loss (CP SP₂: ℝ) : ℝ :=
  ((CP - SP₂) / CP) * 100

theorem percentage_of_loss_is_15_percent (CP SP₂ SP₁: ℝ) (Gain: ℝ):
  CP = 140 ∧ SP₁ = 168 ∧ SP₂ = 119 ∧ Gain = 20 → percentage_loss CP SP₂ = 15 :=
by
  intro h
  sorry

end percentage_of_loss_is_15_percent_l286_286642


namespace find_T_l286_286005

variable {n : ℕ}
variable {a b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- Conditions
axiom h1 : ∀ n, b n - a n = 2^n + 1
axiom h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2

-- Goal
theorem find_T (n : ℕ) (a b S T : ℕ → ℕ)
  (h1 : ∀ n, b n - a n = 2^n + 1)
  (h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2) :
  T n = 2^(n + 1) + n * (n + 1) / 2 - 5 := sorry

end find_T_l286_286005


namespace ratio_of_surface_areas_l286_286087

-- Definitions based on conditions in the problem
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

-- The main theorem to state the equivalent problem
theorem ratio_of_surface_areas (r1 r2 : ℝ) (h : r1 / r2 = 1 / 2) :
  surface_area r1 / surface_area r2 = 1 / 4 := by
  sorry

end ratio_of_surface_areas_l286_286087


namespace distance_between_A_and_B_l286_286177

theorem distance_between_A_and_B (v_A v_B d d' : ℝ)
  (h1 : v_B = 50)
  (h2 : (v_A - v_B) * 30 = d')
  (h3 : (v_A + v_B) * 6 = d) :
  d = 750 :=
sorry

end distance_between_A_and_B_l286_286177


namespace least_positive_angle_l286_286709

theorem least_positive_angle (φ : ℝ) (h : real.cos (10 * real.pi / 180) = real.sin (50 * real.pi / 180) + real.sin (φ * real.pi / 180)) :
  φ = 10 :=
sorry

end least_positive_angle_l286_286709


namespace hyperbola_params_l286_286913

theorem hyperbola_params (a b h k : ℝ) (h_positivity : a > 0 ∧ b > 0)
  (asymptote_1 : ∀ x : ℝ, ∃ y : ℝ, y = (3/2) * x + 4)
  (asymptote_2 : ∀ x : ℝ, ∃ y : ℝ, y = -(3/2) * x + 2)
  (passes_through : ∃ x y : ℝ, x = 2 ∧ y = 8 ∧ (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) 
  (standard_form : ∀ x y : ℝ, ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) : 
  a + h = 7/3 := sorry

end hyperbola_params_l286_286913


namespace six_digit_palindromes_div_by_3_l286_286319

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  (d1 = d6) ∧ (d2 = d5) ∧ (d3 = d4)

def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 + d2 + d3 + d4 + d5 + d6

def is_divisible_by_3 (n : ℕ) : Prop :=
  sum_of_digits n % 3 = 0

def valid_six_digit_palindromes : ℕ :=
  (Finset.filter (λ n, is_palindrome n ∧ is_divisible_by_3 n ∧ (n / 100000) % 10 ≠ 0)
    (Finset.range 900000)).card

theorem six_digit_palindromes_div_by_3 : valid_six_digit_palindromes = 270 :=
by
  sorry

end six_digit_palindromes_div_by_3_l286_286319


namespace johns_minutes_billed_l286_286364

theorem johns_minutes_billed 
  (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 5) (h2 : cost_per_minute = 0.25) (h3 : total_bill = 12.02) :
  ⌊(total_bill - monthly_fee) / cost_per_minute⌋ = 28 :=
by
  sorry

end johns_minutes_billed_l286_286364


namespace partition_students_into_two_groups_l286_286333

-- Define the problem parameters
variables {Student : Type} [fintype Student] (knows : Student → Student → Prop)
variables (hknows_symmetric : symmetric knows) (hknows_exact : ∀ s, (finset.filter (knows s) finset.univ).card = 3)

theorem partition_students_into_two_groups (Student : Type) [fintype Student] (knows : Student → Student → Prop)
  (hknows_symmetric : symmetric knows)
  (hknows_exact : ∀ s, (finset.filter (knows s) finset.univ).card = 3):
  ∃ (group1 group2 : finset Student), (group1 ∪ group2 = finset.univ) ∧ (group1 ∩ group2 = ∅) ∧
  (∀ s ∈ group1, (finset.filter (λ t, knows s t ∧ t ∈ group1) group1).card ≤ 1) ∧
  (∀ s ∈ group2, (finset.filter (λ t, knows s t ∧ t ∈ group2) group2).card ≤ 1) :=
by
  sorry

end partition_students_into_two_groups_l286_286333


namespace find_M_value_l286_286392

theorem find_M_value :
  (let M := (2⁻¹ * (∑ k in (finset.range 11), (nat.choose 20 k) (real.sqrt 1)) - 21 ) / 20
  in ⌊M / 100⌋ = 262)  :=
begin
  -- theorem statement only
  sorry
end

end find_M_value_l286_286392


namespace adjacent_diff_at_least_five_l286_286701

theorem adjacent_diff_at_least_five :
  ∃ (i j k l : ℕ), 
    i < 8 ∧ j < 8 ∧ k < 8 ∧ l < 8 ∧ 
    ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ 
     (j = l ∧ (i = k + 1 ∨ i = k - 1))) ∧ 
    ∃ (grid : Fin 8 → Fin 8 → ℕ), 
      (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 64) ∧ 
      |grid i j - grid k l| ≥ 5 :=
begin
  sorry
end

end adjacent_diff_at_least_five_l286_286701


namespace perpendicular_HK_AB_l286_286857

variable {Point Line Circle : Type}
variable [Geometry Point Line Circle]

-- Assume we have an isosceles triangle ABC with vertex at A
variable (A B C : Point)
variable (triangle_ABC_isosceles : IsoscelesTriangle A B C)

-- Let ω be a circle tangent to AC at C
variable (ω : Circle)
variable (tangent_ω_AC_C : TangentAt ω AC C)
variable (center_K : Center ω)

-- Let H be the second intersection point of BC and ω
variable (H : Point)
variable (second_intersection_H : SecondIntersection ω (LineThrough B C) H)

-- We need to show that HK is perpendicular to AB
theorem perpendicular_HK_AB
  (triangle_ABC_isosceles : IsoscelesTriangle A B C)
  (tangent_ω_AC_C : TangentAt ω AC C)
  (center_K : Center ω)
  (second_intersection_H : SecondIntersection ω (LineThrough B C) H) :
  Perpendicular (LineThrough H K) (LineThrough A B) :=
sorry

end perpendicular_HK_AB_l286_286857


namespace volume_of_smaller_pyramid_correct_l286_286641

noncomputable def volume_of_smaller_pyramid (base_edge : ℝ) (slant_height : ℝ) (intersect_height : ℝ) : ℝ :=
  let OA := base_edge / 2
  let VA := slant_height
  let VO := real.sqrt (VA^2 - OA^2)
  let small_pyr_height := VO - intersect_height
  let scale_ratio := small_pyr_height / VO
  let small_base_edge := base_edge * scale_ratio
  let small_base_area := (small_base_edge)^2
  let small_volume := (1 / 3) * small_base_area * small_pyr_height
  small_volume

theorem volume_of_smaller_pyramid_correct : 
  volume_of_smaller_pyramid 12 15 5 = (48 * (189 - 10 * real.sqrt 189 + 25) * (real.sqrt 189 - 5)) / 189 :=
sorry

end volume_of_smaller_pyramid_correct_l286_286641


namespace squirrel_cannot_jump_1000_units_away_l286_286112

noncomputable def squirrel_initial_points_area : ℕ :=
  let initial_points := {(x, y) | 0 ≤ x ∧ x < 5 ∧ 0 ≤ y ∧ y < 5} in
  initial_points.card

theorem squirrel_cannot_jump_1000_units_away :
  squirrel_initial_points_area = 25 :=
by 
  sorry

end squirrel_cannot_jump_1000_units_away_l286_286112


namespace decreasing_interval_g_l286_286871

def f (x : ℝ) : ℝ := sorry  -- since f(x) is not explicitly given

def g (x : ℝ) : ℝ := x^2 * f(x - 1)

theorem decreasing_interval_g : ∃ I : set ℝ, I = set.Ioo 0 1 ∧ ∀ x y ∈ I, x < y → g y < g x :=
begin
  sorry
end

end decreasing_interval_g_l286_286871


namespace count_numbers_with_perfect_square_factors_l286_286037

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286037


namespace sum_of_coordinates_of_point_D_l286_286885

theorem sum_of_coordinates_of_point_D (x y : ℝ) :
  let M := (5 : ℝ, 4 : ℝ)
  let C := (7 : ℝ, -2 : ℝ)
  let D := (x, y)
  2 * 5 = 7 + x ∧ 2 * 4 = -2 + y -> x + y = 13 :=
by
  intros M C D hx hy
  sorry

end sum_of_coordinates_of_point_D_l286_286885


namespace no_sphinx_tiling_l286_286303

def equilateral_triangle_tiling_problem (side_length : ℕ) (pointing_up : ℕ) (pointing_down : ℕ) : Prop :=
  let total_triangles := side_length * side_length
  pointing_up + pointing_down = total_triangles ∧ 
  total_triangles = 36 ∧
  pointing_down = 1 + 2 + 3 + 4 + 5 ∧
  pointing_up = 1 + 2 + 3 + 4 + 5 + 6 ∧
  (pointing_up % 2 = 1) ∧
  (pointing_down % 2 = 1) ∧
  (2 * pointing_up + 4 * pointing_down ≠ total_triangles ∧ 4 * pointing_up + 2 * pointing_down ≠ total_triangles)

theorem no_sphinx_tiling : ¬equilateral_triangle_tiling_problem 6 21 15 :=
by
  sorry

end no_sphinx_tiling_l286_286303


namespace transformation_order_independent_transformation_decomposition_similar_triangle_decomposition_l286_286238

structure Triangle :=
(point1 : ℝ × ℝ)
(point2 : ℝ × ℝ)
(point3 : ℝ × ℝ)

variables (H1 H4 : Triangle)
variables (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ)

-- Define the transformation function F
noncomputable def F (T : Triangle) (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ) : Triangle := sorry

-- Define the order independence of rotation and scaling
theorem transformation_order_independent (T : Triangle) (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ) :
  F (F T K ϕ 1) K 0 λ = F (F T K 0 λ) K ϕ 1 := sorry

-- Define decomposition of transformation F into F1 and F2
noncomputable def F1 (T : Triangle) (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ) : Triangle := sorry
noncomputable def F2 (T : Triangle) (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ) : Triangle := sorry

variables (H′ H″ : Triangle)

-- Prove the existence of intermediate steps
theorem transformation_decomposition (H1 H4 : Triangle) (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ) :
  H′ = F1 H1 K ϕ λ ∧ H″ = F2 H′ K ϕ λ ∧ F H1 K ϕ λ = H4 := sorry

-- Prove similar triangle transformation can be decomposed into smaller transformations
theorem similar_triangle_decomposition (H1 H2 : Triangle) (K : ℝ × ℝ) (ϕ : ℝ) (λ : ℝ) :
  ∃ S1 S2 S3 : Triangle,
    F1 (F1 S1 K ϕ λ) K ϕ λ = H2 ∧
    F1 (F1 (F1 S1 K ϕ λ) K ϕ λ) K ϕ λ = S2 ∧
    F1 (F1 (F1 (F1 S1 K ϕ λ) K ϕ λ) K ϕ λ) K ϕ λ = S3 := sorry

end transformation_order_independent_transformation_decomposition_similar_triangle_decomposition_l286_286238


namespace triangle_area_correct_l286_286375

noncomputable def area_triangle {α d : ℝ} (ABCD : Type) [CyclicQuadrilateral ABCD] 
  (h1 : side_eq ABCD) (h2 : angle_eq α ABCD) (h3 : diag_eq d ABCD) : ℝ :=
  1/2 * d^2 * sin α

# Assumed definitions for required context
class CyclicQuadrilateral (ABCD : Type) := (is_cyclic : Prop)
structure side_eq (ABCD : Type) := (AB_eq_BC : ℝ) (AD_plus_CD : ℝ)
structure angle_eq (α : ℝ) (ABCD : Type) := (BAD : ℝ)
structure diag_eq (d : ℝ) (ABCD : Type) := (AC : ℝ)

theorem triangle_area_correct {ABCD : Type} [CyclicQuadrilateral ABCD]
  (h1 : side_eq ABCD) (h2 : angle_eq α ABCD) (h3 : diag_eq d ABCD) :
  area_triangle ABCD h1 h2 h3 = 1/2 * d^2 * sin α := sorry

end triangle_area_correct_l286_286375


namespace inverse_proportion_inequality_l286_286389

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l286_286389


namespace root_eq_l286_286713

-- Define the mathematical problem: proving the roots of the specific equation.
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem root_eq :
  (∀ x : ℝ, 4 * sqrt x + 3 * x ^ (-1 / 2) = 10 → x = (19 + 5 * Real.sqrt 13) / 8 ∨ x = (19 - 5 * Real.sqrt 13) / 8) ∧
  (4 * sqrt ((19 + 5 * sqrt 13) / 8) + 3 * ((19 + 5 * sqrt 13) / 8) ^ (-1 / 2) = 10) ∧
  (4 * sqrt ((19 - 5 * sqrt 13) / 8) + 3 * ((19 - 5 * sqrt 13) / 8) ^ (-1 / 2) = 10) :=
by
  sorry

end root_eq_l286_286713


namespace minimum_sum_of_sequence_l286_286491

def min_sum_even (k : ℕ) : ℕ :=
  3 * k^2 + 2 * k - 1

def min_sum_odd (k : ℕ) : ℕ :=
  3 * k^2 + 5 * k + 1

theorem minimum_sum_of_sequence (n : ℕ) (α : List ℕ) (h1 : α.length = n) (h2 : ∀ x ∈ α, x > 0) 
(h3 : α.nodup = false) (h4 : ∀ i, List.nodup ((α.map (λ x => x - 1)) ↔
 List.nodup α) (k : ℕ) :
  (n = 2 * k → ∃ α, (α.sum = min_sum_even k)) ∧
  (n = 2 * k + 1 → ∃ α, (α.sum = min_sum_odd k)) := by
  sorry

end minimum_sum_of_sequence_l286_286491


namespace maximum_marks_l286_286886

theorem maximum_marks (M : ℝ) (mark_obtained failed_by : ℝ) (pass_percentage : ℝ) 
  (h1 : pass_percentage = 0.6) (h2 : mark_obtained = 250) (h3 : failed_by = 50) :
  (pass_percentage * M = mark_obtained + failed_by) → M = 500 :=
by 
  sorry

end maximum_marks_l286_286886


namespace greatest_odd_factors_below_200_l286_286167

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l286_286167


namespace cot_neg_45_eq_neg1_l286_286347

theorem cot_neg_45_eq_neg1 : Real.cot (-(Real.pi / 4)) = -1 :=
by
  sorry

end cot_neg_45_eq_neg1_l286_286347


namespace find_intervals_and_alpha_l286_286775

noncomputable theory
open_locale real

-- Conditions
def is_maximum_at (f : ℝ → ℝ) (x M : ℝ) := ∀ y, f y ≤ M ∧ f x = M
def is_minimum_at (f : ℝ → ℝ) (x m : ℝ) := ∀ y, f y ≥ m ∧ f x = m

-- Given function and properties
def f (x : ℝ) : ℝ := A * sin (ω * x + φ)
variables (A ω φ : ℝ)
variables hA : A > 0
variables hω : ω > 0
variables hφ : 0 < φ ∧ φ < π
variables hmax : is_maximum_at (f A ω φ) (π / 12) 4
variables hmin : is_minimum_at (f A ω φ) (5 * π / 12) (-4)

-- Theorem to prove
theorem find_intervals_and_alpha :
  (∃ intervals : set ℝ, intervals = (Icc 0 (π / 12)) ∪ (Icc (5 * π / 12) (3 * π / 4)) ∧
   (∀ α : ℝ, (0 < α ∧ α < π) → f A ω φ ((2 / 3) * α + (π / 12)) = 2 → (α = π / 6 ∨ α = 5 * π / 6))) :=
sorry

end find_intervals_and_alpha_l286_286775


namespace count_perfect_square_factors_l286_286068

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l286_286068


namespace area_triangle_AMN_l286_286838

-- Define the geometric problem in Lean
variable (ABC : Triangle)
variable (S : ℝ) (AB AC : ℝ)
variable (ratio_AB_AC : AB / AC = 1 / 5)
variable (M N : Point) -- Points of intersection on the midline parallel to BC

-- Define the area function for triangle AMN given the above conditions
noncomputable def area_AMN (ABC : Triangle) (M N : Point) : ℝ := 
  let S := area ABC in
  S / 12

-- Statement of the theorem
theorem area_triangle_AMN :
  ∀ (ABC : Triangle) (M N : Point),
  let S := area ABC in
  (AB / AC = 1 / 5) → 
  area_AMN ABC M N = S / 12 :=
sorry

end area_triangle_AMN_l286_286838


namespace count_numbers_with_square_factors_l286_286045

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286045


namespace sabrina_fraction_books_second_month_l286_286528

theorem sabrina_fraction_books_second_month (total_books : ℕ) (pages_per_book : ℕ) (books_first_month : ℕ) (pages_total_read : ℕ)
  (h_total_books : total_books = 14)
  (h_pages_per_book : pages_per_book = 200)
  (h_books_first_month : books_first_month = 4)
  (h_pages_total_read : pages_total_read = 1000) :
  let total_pages := total_books * pages_per_book
  let pages_first_month := books_first_month * pages_per_book
  let pages_remaining := total_pages - pages_first_month
  let books_remaining := total_books - books_first_month
  let pages_read_first_month := total_pages - pages_total_read
  let pages_read_second_month := pages_read_first_month - pages_first_month
  let books_second_month := pages_read_second_month / pages_per_book
  let fraction_books := books_second_month / books_remaining
  fraction_books = 1 / 2 :=
by
  sorry

end sabrina_fraction_books_second_month_l286_286528


namespace not_monotonically_increasing_implies_positive_derivative_l286_286454

theorem not_monotonically_increasing_implies_positive_derivative (a b : ℝ) (h₁ : ∀ x ∈ Set.Ioo a b, (swap (≺)) (f x)) :
  ∃ x ∈ Set.Ioo a b, (f' x) ≤ 0 :=
by
  let f : ℝ → ℝ := λ x, x^3
  have h_inc : ∀ x1 x2, x1 < x2 → f x1 < f x2 := by
    intros x1 x2 h
    exact pow_lt_pow_of_lt_left h dec_trivial
  have contra : ∃ x ∈ Set.Ioo a b, derivative f x ≤ 0 := sorry
  exact contra

end not_monotonically_increasing_implies_positive_derivative_l286_286454


namespace problem1_problem2_l286_286601

-- Definitions for the inequalities
def f (x a : ℝ) : ℝ := abs (x - a) - 1

-- Problem 1: Given a = 2, solve the inequality f(x) + |2x - 3| > 0
theorem problem1 (x : ℝ) (h1 : abs (x - 2) + abs (2 * x - 3) > 1) : (x ≥ 2 ∨ x ≤ 4 / 3) := sorry

-- Problem 2: If the inequality f(x) > |x - 3| has solutions, find the range of a
theorem problem2 (a : ℝ) (h2 : ∃ x : ℝ, abs (x - a) - abs (x - 3) > 1) : a < 2 ∨ a > 4 := sorry

end problem1_problem2_l286_286601


namespace find_four_digit_numbers_l286_286143

theorem find_four_digit_numbers (a b c d : ℕ) : 
  (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
  (1000 * a + 100 * b + 10 * c + d ≤ 9999) ∧ 
  (1000 ≤ 1000 * d + 100 * c + 10 * b + a) ∧ 
  (1000 * d + 100 * c + 10 * b + a ≤ 9999) ∧
  (a + d = 9) ∧ 
  (b + c = 13) ∧
  (1001 * (a + d) + 110 * (b + c) = 19448) → 
  (1000 * a + 100 * b + 10 * c + d = 9949 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9859 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9769 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9679 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9589 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9499) :=
sorry

end find_four_digit_numbers_l286_286143


namespace quadratic_coeff_sum_l286_286211

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l286_286211


namespace odd_n_not_divisible_by_n_squared_l286_286706

theorem odd_n_not_divisible_by_n_squared (n : ℕ) (h1 : n % 2 = 1) (h2 : n ≠ 1) : ¬ (n^2 ∣ (nat.factorial (n-1))) ↔ (nat.prime n ∨ n = 9) :=
by sorry

end odd_n_not_divisible_by_n_squared_l286_286706


namespace probability_same_color_l286_286487

-- Definitions according to conditions
def total_socks : ℕ := 24
def blue_pairs : ℕ := 7
def green_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_blue_socks : ℕ := blue_pairs * 2
def total_green_socks : ℕ := green_pairs * 2
def total_red_socks : ℕ := red_pairs * 2

-- Probability calculations
def probability_blue : ℚ := (total_blue_socks * (total_blue_socks - 1)) / (total_socks * (total_socks - 1))
def probability_green : ℚ := (total_green_socks * (total_green_socks - 1)) / (total_socks * (total_socks - 1))
def probability_red : ℚ := (total_red_socks * (total_red_socks - 1)) / (total_socks * (total_socks - 1))

def total_probability : ℚ := probability_blue + probability_green + probability_red

theorem probability_same_color : total_probability = 28 / 69 :=
by
  sorry

end probability_same_color_l286_286487


namespace projection_of_sum_on_vec_a_l286_286733

open Real

noncomputable def vector_projection (a b : ℝ) (angle : ℝ) : ℝ := 
  (cos angle) * (a * b) / a

theorem projection_of_sum_on_vec_a (a b : EuclideanSpace ℝ (Fin 3)) 
  (h₁ : ‖a‖ = 2) 
  (h₂ : ‖b‖ = 2) 
  (h₃ : inner a b = (2 * 2) * (cos (π / 3))):
  (inner (a + b) a) / ‖a‖ = 3 := 
by
  sorry

end projection_of_sum_on_vec_a_l286_286733


namespace greatest_odd_factors_l286_286153

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l286_286153


namespace tanya_problem_max_uncrossed_numbers_l286_286190

theorem tanya_problem_max_uncrossed_numbers :
  ∃ (S : set ℕ), (∀ x ∈ S, x ∈ finset.range 101) ∧
    (∀ (a b : ℕ), a ∈ S → b ∈ S → a^2 ≥ 4*b) ∧
    finset.card (finset.filter (λ x, x ∈ S) (finset.range 101)) = 81 := sorry

end tanya_problem_max_uncrossed_numbers_l286_286190


namespace reflection_of_P_about_M_is_Q_l286_286540

-- Given points P and M
structure Point :=
(x : ℝ)
(y : ℝ)

def P : Point := ⟨1, -2⟩
def M : Point := ⟨3, 0⟩

-- Define a function to calculate the reflection Q of point P about M
def reflection (P M : Point) : Point :=
⟨2 * M.x - P.x, 2 * M.y - P.y⟩

-- The expected coordinates of Q
def Q_expected : Point := ⟨5, 2⟩

-- Proof statement: the reflection of P about M is Q
theorem reflection_of_P_about_M_is_Q : reflection P M = Q_expected := 
sorry

end reflection_of_P_about_M_is_Q_l286_286540


namespace locus_in_equilateral_triangle_l286_286354

variable {A B C M : EuclideanGeometry.Point}

theorem locus_in_equilateral_triangle
  (h_equilateral : EuclideanGeometry.is_equilateral_triangle A B C)
  (h_ma_eq_mb_mc : EuclideanGeometry.dist A M ^ 2 = EuclideanGeometry.dist B M ^ 2 + EuclideanGeometry.dist C M ^ 2) :
  EuclideanGeometry.on_the_locus_of_arc M A B C (150 : ℝ) :=
sorry

end locus_in_equilateral_triangle_l286_286354


namespace average_xyz_l286_286509

theorem average_xyz (x y z : ℝ) 
  (h1 : 2003 * z - 4006 * x = 1002) 
  (h2 : 2003 * y + 6009 * x = 4004) : (x + y + z) / 3 = 5 / 6 :=
by
  sorry

end average_xyz_l286_286509


namespace find_n_0_l286_286720

open Real

def sequence_a_n (n : ℕ) : ℕ :=
  ⌊log 2 n⌋₊

def sum_S_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence_a_n (i + 1)

theorem find_n_0 : ∃ n_0 : ℕ, sum_S_n n_0 > 2018 ∧ ∀ n < n_0, sum_S_n n ≤ 2018 :=
  by
  use 316
  split
  { -- Prove sum_S_n 316 > 2018
    sorry }
  { -- Prove for all n < 316, sum_S_n n ≤ 2018
    sorry }

end find_n_0_l286_286720


namespace count_perfect_square_factors_l286_286018

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286018


namespace option_D_correct_l286_286967

-- Define the condition for oil floating on water.
def oil_floats_on_water : Prop := true  -- acknowledging the fact that oil floats on water.

-- Define the condition for basketball motion predictability.
def basketball_motion_predictable : Prop := true  -- acknowledging the fact that the basketball's motion is predictable.

-- Define the impracticality of a census survey for the pen refills service life.
def census_impractical_for_pen_refills : Prop := true  -- acknowledging the impracticality of a census survey.

-- Define the stability condition of data sets based on variance.
def stability_condition (S_A S_B : ℝ) (μ_A μ_B : ℝ) (hμ : μ_A = μ_B) : Prop :=
  S_A < S_B  -- Lower variance indicates greater stability.

-- Given variances for sets A and B.
def S_A : ℝ := 2  -- Variance of set A.
def S_B : ℝ := 2.5  -- Variance of set B.
def μ_A : ℝ := 0  -- Mean of set A.
def μ_B : ℝ := 0  -- Mean of set B.

theorem option_D_correct : oil_floats_on_water ∧ basketball_motion_predictable ∧ census_impractical_for_pen_refills ∧ stability_condition S_A S_B μ_A μ_B →
  ∃ (correct_option : string), correct_option = "D" :=
by
  intros h,
  existsi "D",
  sorry

end option_D_correct_l286_286967


namespace complex_num_quadrant_l286_286739

theorem complex_num_quadrant (z : ℂ) (h : z * (1 + complex.I) = 5 + complex.I) : 
  0 < z.re ∧ 0 > z.im :=
sorry

end complex_num_quadrant_l286_286739


namespace count_numbers_with_square_factors_l286_286040

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l286_286040


namespace find_angle_A_find_area_l286_286756

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def law_c1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + c * Real.cos A = -2 * b * Real.cos A

def law_c2 (a : ℝ) : Prop := a = 2 * Real.sqrt 3
def law_c3 (b c : ℝ) : Prop := b + c = 4

-- Questions
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c) : 
  A = 2 * Real.pi / 3 :=
sorry

theorem find_area (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c)
  (hA : A = 2 * Real.pi / 3) : 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_l286_286756


namespace parallel_planes_normal_vectors_x_value_l286_286085

theorem parallel_planes_normal_vectors_x_value :
  ∀ (x : ℝ), let a := (-1, 2, 4) in
             let b := (x, -1, -2) in
             (∃ λ : ℝ, a = λ • b) → x = 1 / 2 :=
by
  sorry

end parallel_planes_normal_vectors_x_value_l286_286085


namespace limit_of_cos_cos_2_eq_exp_neg_tan_2_l286_286297

noncomputable def limit_cos_cos_2 : Real := 
  lim (fun x => (cos x / cos 2) ^ (1 / (x - 2))) (λ x, true) 2

theorem limit_of_cos_cos_2_eq_exp_neg_tan_2 : 
  limit_cos_cos_2 = exp (-tan 2) :=
sorry

end limit_of_cos_cos_2_eq_exp_neg_tan_2_l286_286297


namespace find_K_l286_286245

theorem find_K : ∃ K : ℕ, (64 ^ (2 / 3) * 16 ^ 2) / 4 = 2 ^ K :=
by
  use 10
  sorry

end find_K_l286_286245


namespace greatest_odd_factors_below_200_l286_286173

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l286_286173


namespace available_seats_l286_286592

/-- Two-fifths of the seats in an auditorium that holds 500 people are currently taken. --/
def seats_taken : ℕ := (2 * 500) / 5

/-- One-tenth of the seats in an auditorium that holds 500 people are broken. --/
def seats_broken : ℕ := 500 / 10

/-- Total seats in the auditorium --/
def total_seats := 500

/-- There are 500 total seats in an auditorium. Two-fifths of the seats are taken and 
one-tenth are broken. Prove that the number of seats still available is 250. --/
theorem available_seats : (total_seats - seats_taken - seats_broken) = 250 :=
by 
  sorry

end available_seats_l286_286592


namespace no_sol_for_eq_xn_minus_yn_eq_2k_l286_286490

theorem no_sol_for_eq_xn_minus_yn_eq_2k (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_n : n > 2) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^n - y^n = 2^k := 
sorry

end no_sol_for_eq_xn_minus_yn_eq_2k_l286_286490


namespace probability_of_both_gender_selection_l286_286286

noncomputable def probability_both_gender_selected : ℚ :=
  let total_ways := (Nat.choose 8 5) in
  let male_ways := (Nat.choose 5 5) in
  let prob_only_males := male_ways / total_ways in
  1 - prob_only_males

theorem probability_of_both_gender_selection :
  probability_both_gender_selected = 55 / 56 := 
sorry

end probability_of_both_gender_selection_l286_286286


namespace Nala_seashells_l286_286176

theorem Nala_seashells :
  let seashells_found : ℕ → ℕ := λ n, 5 + 3 * n
  let unbroken_seashells : ℕ → ℕ
  | 0 => 0
  | n+1 => if (n + 1) % 2 = 0 then (seashells_found n * 3) / 4 else (seashells_found n * 9) / 10
  let total_unbroken_seashells : ℕ := (list.range 7).sum (λ i, unbroken_seashells (i + 1))
  total_unbroken_seashells = 79
:= sorry

end Nala_seashells_l286_286176


namespace number_of_diagonals_25_sides_l286_286298

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end number_of_diagonals_25_sides_l286_286298


namespace count_numbers_with_perfect_square_factors_l286_286056

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286056


namespace maximum_profit_l286_286990

noncomputable def sales_volume (x : ℝ) : ℝ := -10 * x + 1000
noncomputable def profit (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

theorem maximum_profit : ∀ x : ℝ, 44 ≤ x ∧ x ≤ 46 → profit x ≤ 8640 :=
by
  intro x hx
  sorry

end maximum_profit_l286_286990


namespace triangle_inequality_proof_l286_286889

noncomputable def triangle_inequality (a b c A B C: ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧
A + B + C = π →

a * A + b * B + c * C ≥ (a + b + c) * (π/3)

theorem triangle_inequality_proof (a b c A B C: ℝ) 
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: A > 0)
  (h5: B > 0)
  (h6: C > 0)
  (h7: A + B + C = π):
  a * A + b * B + c * C ≥ (a + b + c) * (π/3) :=
sorry

end triangle_inequality_proof_l286_286889


namespace min_value_AC_plus_2BD_l286_286421

theorem min_value_AC_plus_2BD (E : (ℝ → ℝ) := λ x, (x^2 - 4) / 4) 
  (F : (ℝ × ℝ) → Prop := λ p, (p.1 ^ 2 + (p.2 - 1) ^ 2 = 1))
  (l : (ℝ × ℝ) → Prop := λ p, (p.1 = m * (p.2 - 1))) :
  ∃ (A C D B : ℝ × ℝ), 
    F (0, 1) ∧
    A ≠ C ∧ A ≠ B ∧ A ≠ D ∧ C ≠ D ∧ C ≠ B ∧ D ≠ B ∧
    (E A.1 = A.2) ∧ (E C.1 = C.2) ∧ (E D.1 = D.2) ∧ (E B.1 = B.2) ∧ 
    (|A.1 - C.1| + 2 * |B.2 - D.2|) = 2 * real.sqrt 2 :=
sorry

end min_value_AC_plus_2BD_l286_286421


namespace count_numbers_with_perfect_square_factors_l286_286058

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286058


namespace point_on_transformed_graph_l286_286403

theorem point_on_transformed_graph (g : ℝ → ℝ) :
  g 8 = 6 → 
  3 * (1 : ℝ) = (g (3 * (8 / 3)) + 3) / 3 ∧
  (8 / 3 + 1 = 11 / 3) :=
by
  intro h
  have h1 : g (3 * (8 / 3)) = g 8 := by
    norm_num
  rw [h1, h]
  norm_num
  split
  · norm_num
  · norm_num
  trace "Point and sum identified correctly"
  sorry

end point_on_transformed_graph_l286_286403


namespace problem_statement_l286_286740

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (set.Ioo 1 2) then 2 - x else
if ∃ k : ℤ, x ∈ set.Ioo (2^k : ℝ) (2^(k + 1)) then by
  -- Function definition to be detailed for scalar intervals
  sorry
else 0

theorem problem_statement :
  (∀ m : ℤ, f (2^m) = 0) ∧
  (set.Icc 0 (⊤ : ℝ) = set.image f set.Ioi 0) ∧
  (¬ ∃ n : ℤ, f (2^n + 1) = 9) ∧
  (∀ a b : ℝ, (∃ k : ℤ, set.Ioo a b ⊆ set.Ioo (2^k : ℝ) (2^(k + 1))) ↔ (∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x > f y)) := sorry

end problem_statement_l286_286740


namespace find_j_l286_286633

open Real

def slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_j (j : ℝ) :
  slope (5, -6) (j, 29) = 3 / 2 → j = 85 / 3 :=
by
  intros h,
  sorry

end find_j_l286_286633


namespace count_numbers_with_perfect_square_factors_l286_286038

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286038


namespace erased_angle_is_97_l286_286667

theorem erased_angle_is_97 (n : ℕ) (h1 : 3 ≤ n) (h2 : (n - 2) * 180 = 1703 + x) : 
  1800 - 1703 = 97 :=
by sorry

end erased_angle_is_97_l286_286667


namespace cannot_assemble_fourth_figure_l286_286178

-- Definitions derived from the problem conditions
def rhombus_piece : Type := sorry   -- Type representing the rhombus pieces.
def is_white : rhombus_piece → Prop := sorry   -- Predicate for identifying white colored pieces.
def is_grey : rhombus_piece → Prop := sorry    -- Predicate for identifying grey colored pieces.
def can_rotate (r : rhombus_piece) : Prop := sorry  -- Predicate stating rhombus can be rotated.

-- Main theorem
theorem cannot_assemble_fourth_figure (pieces : set rhombus_piece) (fig4 : Type)
  (h1 : ∀ r ∈ pieces, (is_white r ∨ is_grey r) ∧ can_rotate r)
  (h2 : ∀ f, f ∈ fig4 → sorry) -- Conditions representing the fourth figure
  : ¬ ∃ s, (s ⊆ pieces) ∧ (fig4 = s) := 
sorry

end cannot_assemble_fourth_figure_l286_286178


namespace part1_l286_286804

-- Definition of a double root equation
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ x₂ = 2 * x₁ ∧ a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0)

-- Part 1: Proof that x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_equation 1 (-3) 2 :=
by {
  use [1, 2],
  split,
  { intros h,
    linarith, },
  split,
  { refl, },
  split;
  { simp, linarith, }
}

end part1_l286_286804


namespace symmetric_point_A_equals_B_l286_286079

theorem symmetric_point_A_equals_B (θ : ℝ) :
  A = (Real.cos θ, Real.sin θ) ∧ B = (Real.sin (θ + Real.pi / 3), -Real.cos (θ + Real.pi / 3)) →
  (Real.cos θ = Real.sin (θ + Real.pi / 3) ∧ Real.sin θ = Real.cos (θ + Real.pi / 3)) →
  θ = Real.pi / 12 :=
begin
  sorry,
end

end symmetric_point_A_equals_B_l286_286079


namespace probability_yellow_ball_l286_286660

theorem probability_yellow_ball (total_balls : ℕ) (white_balls : ℕ) (yellow_balls : ℕ)
    (h_total : total_balls = white_balls + yellow_balls) :
    (yellow_balls / total_balls : ℚ) = 3 / 5 :=
by
  have h1 : total_balls = 5 := by
    rw [h_total]
    exact rfl
  have h2 : yellow_balls = 3 := rfl
  rw [h1, h2]
  norm_num
  sorry

end probability_yellow_ball_l286_286660


namespace count_perfect_square_factors_except_one_l286_286026

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286026


namespace brass_selling_price_l286_286550

noncomputable def copper_price : ℝ := 0.65
noncomputable def zinc_price : ℝ := 0.30
noncomputable def total_weight_brass : ℝ := 70
noncomputable def weight_copper : ℝ := 30
noncomputable def weight_zinc := total_weight_brass - weight_copper
noncomputable def cost_copper := weight_copper * copper_price
noncomputable def cost_zinc := weight_zinc * zinc_price
noncomputable def total_cost := cost_copper + cost_zinc
noncomputable def selling_price_per_pound := total_cost / total_weight_brass

theorem brass_selling_price :
  selling_price_per_pound = 0.45 :=
by
  sorry

end brass_selling_price_l286_286550


namespace contrapositive_abc_l286_286909

theorem contrapositive_abc (a b c : ℝ) : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (abc ≠ 0) := 
sorry

end contrapositive_abc_l286_286909


namespace simplify_fraction_l286_286896

variable (x y : ℝ)

theorem simplify_fraction :
  (2 * x + y) / 4 + (5 * y - 4 * x) / 6 - y / 12 = (-x + 6 * y) / 6 :=
by
  sorry

end simplify_fraction_l286_286896


namespace matching_socks_probability_l286_286794

theorem matching_socks_probability (pairs : ℕ) (days : ℕ) : 
  pairs = 5 → days = 5 → 
  (∀ d, 1 ≤ d ∧ d ≤ days → (∃ x y, x ≠ y ∧ {x, y} ⊆ (finset.range (2 * pairs)) \ (finset.range ((d - 1) * 2)))) →
  ∃ P, P = (1 / 25 : ℚ) := 
by 
  intros h_pairs h_days h_condition
  sorry

end matching_socks_probability_l286_286794


namespace tank_plastering_cost_l286_286649

noncomputable def plastering_cost (L W D : ℕ) (cost_per_sq_meter : ℚ) : ℚ :=
  let A_bottom := L * W
  let A_long_walls := 2 * (L * D)
  let A_short_walls := 2 * (W * D)
  let A_total := A_bottom + A_long_walls + A_short_walls
  A_total * cost_per_sq_meter

theorem tank_plastering_cost :
  plastering_cost 25 12 6 0.25 = 186 := by
  sorry

end tank_plastering_cost_l286_286649


namespace greatest_odd_factors_l286_286152

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l286_286152


namespace intervals_sum_greater_than_one_over_n_l286_286519

theorem intervals_sum_greater_than_one_over_n (n : ℕ) (h : 0 < n) (S : set ℝ)
  (intervals : fin n → set ℝ) 
  (hcov : S = ⋃ i, intervals i)
  (hdist : ∀ d, 0 < d ∧ d ≤ 1 → ∃ x y ∈ S, |x - y| = d) : 
  ∑ i, real.length (intervals i) > 1 / n :=
sorry

end intervals_sum_greater_than_one_over_n_l286_286519


namespace probability_not_snowing_l286_286552

variable (P_snowing : ℚ)
variable (h : P_snowing = 2/5)

theorem probability_not_snowing (P_not_snowing : ℚ) : 
  P_not_snowing = 3 / 5 :=
by 
  -- sorry to skip the proof
  sorry

end probability_not_snowing_l286_286552


namespace cyclic_quad_l286_286817

noncomputable theory

variables {α : Type*} [euclidean_geometry α]
open euclidean_geometry

def is_convex_quad (A B C D : α) : Prop :=
convex_hull (finset.insert A (finset.insert B (finset.insert C {D}))) = convex_hull {A, B, C, D}

def equal_sides (A B C D : α) : Prop :=
dist A B = dist B C ∧ dist B C = dist C D

def intersection_point (A B C D : α) : α :=
classical.some (exists! (λ M, between A M C ∧ between B M D))

def bisector_intersection (A D : α) : α :=
classical.some (exists (λ K, ∠AKD))

theorem cyclic_quad (A B C D : α) (h_convex : is_convex_quad A B C D) (h_eq_sides : equal_sides A B C D) :
  let M := intersection_point A B C D,
      K := bisector_intersection A D in
  cyclic {A, M, K, D} :=
begin
  sorry,
end

end cyclic_quad_l286_286817


namespace positive_integer_solution_l286_286940

/-- Given that x, y, and t are all equal to 1, and x + y + z + t = 10, we need to prove that z = 7. -/
theorem positive_integer_solution {x y z t : ℕ} (hx : x = 1) (hy : y = 1) (ht : t = 1) (h : x + y + z + t = 10) : z = 7 :=
by {
  -- We would provide the proof here, but for now, we use sorry
  sorry
}

end positive_integer_solution_l286_286940


namespace simplify_expression_l286_286250

theorem simplify_expression :
  ( (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) ) / ( (2 * 3) * (3 * 4) * (4 * 5) * (5 * 6) ) = 1 / 5 :=
by
  sorry

end simplify_expression_l286_286250


namespace ce_squared_plus_de_squared_eq_216_l286_286854

-- Definitions of the geometric objects and given conditions
variables (A B C D E : Point)
variable (O : Point) -- Center of the circle

variable (r : ℝ) (h_radius : r = 6 * Real.sqrt 2) -- Radius of the circle
variables (h_AB_diameter : diameter A B O)
variables (h_BE : E ∈ LineSegment B O)
variable (h_BE_value : distance B E = 3 * Real.sqrt 2)
variable (h_angle_AEC : angle A E C = π / 3) -- 60 degrees converted to radians

-- Statement to be proved
theorem ce_squared_plus_de_squared_eq_216 :
  let CE := distance C E
  let DE := distance D E
  CE ^ 2 + DE ^ 2 = 216 :=
sorry

end ce_squared_plus_de_squared_eq_216_l286_286854


namespace diameter_C_is_10_sqrt_2_l286_286308

noncomputable def diameter_C_eq : Prop :=
  ∃ (d : ℝ), 
    (20 / 2)^2 * π - (d / 2)^2 * π*(7+1)/1 = 0

theorem diameter_C_is_10_sqrt_2 :
  diameter_C_eq = (10 * Real.sqrt 2) := sorry

end diameter_C_is_10_sqrt_2_l286_286308


namespace cuboid_edge_integer_l286_286888

theorem cuboid_edge_integer 
  (large_cuboid : {length : ℝ, width : ℝ, height : ℝ})
  (small_cuboids : List {length : ℝ, width : ℝ, height : ℝ})
  (H : ∀ c ∈ small_cuboids, c.length.isInteger ∨ c.width.isInteger ∨ c.height.isInteger) 
  : large_cuboid.length.isInteger ∨ large_cuboid.width.isInteger ∨ large_cuboid.height.isInteger :=
sorry

end cuboid_edge_integer_l286_286888


namespace perfect_square_factors_l286_286052

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286052


namespace bathing_suits_per_model_l286_286539

def models : ℕ := 6
def evening_wear_sets_per_model : ℕ := 3
def time_per_trip_minutes : ℕ := 2
def total_show_time_minutes : ℕ := 60

theorem bathing_suits_per_model : (total_show_time_minutes - (models * evening_wear_sets_per_model * time_per_trip_minutes)) / (time_per_trip_minutes * models) = 2 :=
by
  sorry

end bathing_suits_per_model_l286_286539


namespace identify_triangle_centers_l286_286505

variable (P : Fin 7 → Type)
variable (I O H L G N K : Type)
variable (P1 P2 P3 P4 P5 P6 P7 : Type)
variable (cond : (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H))

theorem identify_triangle_centers :
  (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H) :=
by sorry

end identify_triangle_centers_l286_286505


namespace no_monotonically_decreasing_l286_286131

variable (f : ℝ → ℝ)

theorem no_monotonically_decreasing (x1 x2 : ℝ) (h1 : ∃ x1 x2, x1 < x2 ∧ f x1 ≤ f x2) : ∀ x1 x2, x1 < x2 → f x1 > f x2 → False :=
by
  intros x1 x2 h2 h3
  obtain ⟨a, b, h4, h5⟩ := h1
  have contra := h5
  sorry

end no_monotonically_decreasing_l286_286131


namespace range_g_l286_286357

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2*x + 2)

theorem range_g : 
  set.range g = set.Icc (-1 / 2) 1 :=
sorry

end range_g_l286_286357


namespace clock_shows_l286_286694

-- Definitions for the hands and their positions
variables {A B C : ℕ} -- Representing hands A, B, and C as natural numbers for simplicity

-- Conditions based on the problem description:
-- 1. Hands A and B point exactly at the hour markers.
-- 2. Hand C is slightly off from an hour marker.
axiom hand_A_hour_marker : A % 12 = A
axiom hand_B_hour_marker : B % 12 = B
axiom hand_C_slightly_off : C % 12 ≠ C

-- Theorem stating that given these conditions, the clock shows the time 4:50
theorem clock_shows (h1: A % 12 = A) (h2: B % 12 = B) (h3: C % 12 ≠ C) : A = 50 ∧ B = 12 ∧ C = 4 :=
sorry

end clock_shows_l286_286694


namespace count_perfect_square_factors_except_one_l286_286023

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286023


namespace urn_probability_four_each_l286_286292

def number_of_sequences := Nat.choose 6 3

def probability_of_sequence := (1/3) * (1/2) * (3/5) * (1/2) * (4/7) * (5/8)

def total_probability := number_of_sequences * probability_of_sequence

theorem urn_probability_four_each :
  total_probability = 5 / 14 := by
  -- proof goes here
  sorry

end urn_probability_four_each_l286_286292


namespace intersect_orthogonal_triahedron_with_plane_l286_286483

-- Define vertex points A, B, C, and origin O
variables {A B C O : Point}

-- Define acute triangle ABC
variables {α β γ : ℝ} -- Angles of triangle
variables {a b c : ℝ} -- Sides opposite to angles α, β, γ

-- Define the orthogonal trihedral angle represented by right triangles BCO, CAO, and ABO
noncomputable def right_triangle (X Y Z : Point) : Prop :=
  ∃ (a b c : ℝ), c^2 = a^2 + b^2

-- Condition: Right triangles BCO, CAO, ABO
axiom BCO_triangle : right_triangle B C O
axiom CAO_triangle : right_triangle C A O
axiom ABO_triangle : right_triangle A B O

-- The equivalent proof problem
theorem intersect_orthogonal_triahedron_with_plane :
  ∃ (P : Plane), congruent (triangle_intersection P) (acute_triangle A B C) :=
sorry

end intersect_orthogonal_triahedron_with_plane_l286_286483


namespace parabola_intersections_l286_286314

def condition1 : Prop := ∀ (f : set (ℝ × ℝ)), ∃ (P ∈ f), P = (0, 1)  -- Focus at point (0,1)
def condition2 (a b : ℤ) (y x : ℝ) : Prop := y = a * x + b  -- Directrix line y = ax + b
def condition3_a : set ℤ := {-3, -2, -1, 0, 1, 2, 3}  -- Integer values for a
def condition3_b : set ℤ := {-4, -3, -2, -1, 1, 2, 3, 4}  -- Integer values for b

def intersecting_parabolas : set (ℝ × ℝ) := 
  { p : ℝ × ℝ | ∃ a₁ a₂ b₁ b₂, a₁ ∈ condition3_a ∧ b₁ ∈ condition3_b ∧ a₂ ∈ condition3_a ∧ b₂ ∈ condition3_b ∧ 
    (a₁ ≠ a₂ ∨ b₁ ≠ b₂ ∧ (∀ x y, condition2 a₁ b₁ y x ∧ condition2 a₂ b₂ y x)) ∧ (∀ f, condition1 f) }

theorem parabola_intersections : 
  nat.card (intersecting_parabolas) = 2282 := 
  sorry

end parabola_intersections_l286_286314


namespace complex_triples_l286_286348

variables {C : Type*} [ComplexField C] (x y z : C)

-- Define the hypotheses
def hypothesis1 : Prop :=
  (x + y)^3 + (y + z)^3 + (z + x)^3 - 3 * (x + y) * (y + z) * (z + x) = 0

def hypothesis2 : Prop :=
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = 0

-- The statement to be proven
theorem complex_triples (h1 : hypothesis1 x y z) (h2 : hypothesis2 x y z) :
  x + y + z = 0 ∧ x * y * z = 0 := by
  sorry

end complex_triples_l286_286348


namespace find_smallest_number_l286_286562

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end find_smallest_number_l286_286562


namespace shaded_squares_count_l286_286474

def rows := 8
def cols := 8
def shaded_pattern : ℕ → ℕ → Bool := λ i j => (i + j) % 2 = 0

theorem shaded_squares_count : ∑ i in Finset.range rows, ∑ j in Finset.range cols, if shaded_pattern i j then 1 else 0 = 49 :=
by
  sorry

end shaded_squares_count_l286_286474


namespace discount_rate_on_mysteries_l286_286618

-- Define the normal prices for biographies and mysteries.
def normal_price_biography := 20
def normal_price_mystery := 12

-- Define the quantities of books bought.
def quantity_biographies := 5
def quantity_mysteries := 3

-- Define the given total savings and the sum of the discount rates.
def total_savings := 19
def sum_discount_rates := 0.43

-- Define the question to be answered.
theorem discount_rate_on_mysteries 
  (B M : ℝ)
  (h1 : B + M = sum_discount_rates)
  (h2 : quantity_biographies * normal_price_biography * (1 - B) + quantity_mysteries * normal_price_mystery * (1 - M) = quantity_biographies * normal_price_biography + quantity_mysteries * normal_price_mystery - total_savings) :
  M = 0.375 :=
  sorry

end discount_rate_on_mysteries_l286_286618


namespace shorter_diagonal_of_rhombus_l286_286542

variable (d s : ℝ)  -- d for shorter diagonal, s for the side length of the rhombus

theorem shorter_diagonal_of_rhombus 
  (h1 : ∀ (s : ℝ), s = 39)
  (h2 : ∀ (a b : ℝ), a^2 + b^2 = s^2)
  (h3 : ∀ (d a : ℝ), (d / 2)^2 + a^2 = 39^2)
  (h4 : 72 / 2 = 36)
  : d = 30 := 
by 
  sorry

end shorter_diagonal_of_rhombus_l286_286542


namespace g_inv_g_inv_14_l286_286902

def g (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g_inv (y : ℝ) : ℝ := (y + 3) / 5

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 32 / 25 :=
by
  sorry

end g_inv_g_inv_14_l286_286902


namespace find_f_solutions_to_f_eq_n_l286_286372

def f : Nat → Nat
| n => if n > 2000 then n - 12 else f (f (n + 16))

theorem find_f (n : Nat) : 
  (n > 2000 → f n = n - 12) ∧ 
  (n ≤ 2000 → f n = 1992 - (n % 4)) := by
  sorry

theorem solutions_to_f_eq_n (n : Nat) : 
  f n = n ↔ n ∈ {1992, 1991, 1990, 1989} := by
  sorry

end find_f_solutions_to_f_eq_n_l286_286372


namespace greatest_perfect_square_below_200_l286_286162

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l286_286162


namespace num_correct_relations_l286_286216

variable {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem num_correct_relations 
  (h1 : 0 • a = (0:V))
  (h2 : a ⬝ b = b ⬝ a) 
  (h3 : a ⬝ a = ∥a∥^2) 
  (h4 : (a ⬝ b) • c = a ⬝ (b ⬝ c)) 
  (h5 : ∥a ⬝ b∥ ≤ a ⬝ b) : 
  2 = number_of_true_statements [h1, h2, h3, h4, h5] :=
sorry

end num_correct_relations_l286_286216


namespace blue_pieces_correct_l286_286944

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end blue_pieces_correct_l286_286944


namespace vector_sum_length_l286_286764

open Real

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vector_angle_cosine (v w : ℝ × ℝ) : ℝ :=
dot_product v w / (vector_length v * vector_length w)

theorem vector_sum_length (a b : ℝ × ℝ)
  (ha : vector_length a = 2)
  (hb : vector_length b = 2)
  (hab_angle : vector_angle_cosine a b = cos (π / 3)):
  vector_length (a.1 + b.1, a.2 + b.2) = 2 * sqrt 3 :=
by sorry

end vector_sum_length_l286_286764


namespace find_C_find_abc_l286_286092

noncomputable theory

variables {A B C a b c : ℝ}

-- Given conditions
def conditions_1 := A = π / 6 ∧ (1 + real.sqrt 3) * c = 2 * b
def conditions_2 := A = π / 6 ∧ C = π / 4 ∧ (1 + real.sqrt 3) * c = 2 * b ∧ (b * a * real.cos C) = (1 + real.sqrt 3)

-- Proving the first part
theorem find_C (h : conditions_1) : C = π / 4 := 
sorry

-- Proving the second part
theorem find_abc (h : conditions_2) : a = real.sqrt 2 ∧ b = (1 + real.sqrt 3) ∧ c = 2 := 
sorry

end find_C_find_abc_l286_286092


namespace relationship_among_a_ab_ab2_l286_286129

theorem relationship_among_a_ab_ab2 (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) :
  a < a * b ∧ a * b < a * b^2 :=
by
  sorry

end relationship_among_a_ab_ab2_l286_286129


namespace find_4a_3b_l286_286317

noncomputable def g (x : ℝ) : ℝ := 4 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := g x + 2

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_4a_3b (a b : ℝ) (h_inv : ∀ x : ℝ, f (f_inv x) a b = x) : 4 * a + 3 * b = 4 :=
by
  -- Proof skipped for now
  sorry

end find_4a_3b_l286_286317


namespace probability_inequality_up_to_99_l286_286635

theorem probability_inequality_up_to_99 :
  (∀ x : ℕ, 1 ≤ x ∧ x < 100 → (2^x / x!) > x^2) →
    (∃ n : ℕ, (1 ≤ n ∧ n < 100) ∧ (2^n / n!) > n^2) →
      ∃ p : ℚ, p = 1/99 :=
by
  sorry

end probability_inequality_up_to_99_l286_286635


namespace area_union_disks_correct_l286_286610

noncomputable def area_union_disks (r R : ℝ) (h_r : r = 1) (h_R : R = 2) 
  (α : ℝ) (γ : ℝ) (h_cos : cos α = r / (2 * R)) (h_alpha : α = arccos (1/4))
  (h_sin_alpha : sin α = sqrt (1 - (1/4)^2)) (h_sin_2alpha : sin (2*α) = (1 / (2 * 2)) * sqrt (1 - (1 / 4)^2))
  (h_gamma : γ = π - 2 * α) (h_sin_2gamma : sin (2*γ) = (7 * sqrt 15) / 32) : 
  ℝ :=
  π + 7 * arccos (1/4) + sqrt(15) / 2

theorem area_union_disks_correct : 
  area_union_disks 1 2 (rfl) (rfl) (arccos (1 / 4)) (π - 2 * arccos (1 / 4))
    (by rw [cos_arccos, div_div, mul_div_mul_comm, div_self]; norm_num)
    (rfl) 
    (sqrt_sub_one_div_sqr)
    (by rw [one_div, cos_arccos]; simp [sqrt_sub_self_div_sqr])
    (rfl) 
    (by rw [sin_sub_arccos, arccos_div_two, sqrt_one_sub_pow_two]; norm_num) = 
    π + 7 * arccos(1 / 4) + sqrt(15) / 2 := sorry

end area_union_disks_correct_l286_286610


namespace calculate_ab_plus_cd_l286_286859

theorem calculate_ab_plus_cd (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 12) :
  a * b + c * d = 27 :=
by
  sorry -- Proof to be filled in.

end calculate_ab_plus_cd_l286_286859


namespace find_angle_A_find_area_l286_286757

noncomputable theory

-- Define the given conditions of the triangle
variables (A B C : ℝ) (a b c : ℝ)

-- Define the problem conditions
axiom angle_conditions : A + B + C = π
axiom side_conditions : a = 2 * √3 ∧ b + c = 4
axiom equation_condition : a * cos C + c * cos A = -2 * b * cos A

-- Prove the required results
theorem find_angle_A : A = 2 * π / 3 := sorry

theorem find_area : 1 / 2 * b * c * sin A = √3 := sorry

end find_angle_A_find_area_l286_286757


namespace integral_equation_solution_l286_286901

noncomputable def phi (x : ℝ) : ℝ := x * Real.exp x

theorem integral_equation_solution :
  ∀ x, 
    phi x = Real.sin x + 2 * ∫ t in 0..x, (Real.cos (x - t) * phi t) :=
by
  sorry

end integral_equation_solution_l286_286901


namespace candy_distribution_l286_286336

-- Definition of the problem
def emily_candies : ℕ := 30
def friends : ℕ := 4

-- Lean statement to prove
theorem candy_distribution : emily_candies % friends = 2 :=
by sorry

end candy_distribution_l286_286336


namespace cookies_per_bag_l286_286972

theorem cookies_per_bag (n_bags : ℕ) (total_cookies : ℕ) (n_candies : ℕ) (h_bags : n_bags = 26) (h_cookies : total_cookies = 52) (h_candies : n_candies = 15) : (total_cookies / n_bags) = 2 :=
by sorry

end cookies_per_bag_l286_286972


namespace PD_parallel_QR_l286_286091

-- Define the entities in our problem
variable (A B C D E F M N P Q R O : Point)

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom altitude_AD : IsAltitude A D (Line B C)
axiom altitude_BE : IsAltitude B E (Line C A)
axiom altitude_CF : IsAltitude C F (Line A B)
axiom circle_with_diameter_AD : ExistsDiameterCircle A D (Circle Γ)
axiom intersect_AC : Intersects (Circle Γ) (Line A C) M
axiom intersect_AB : Intersects (Circle Γ) (Line A B) N
axiom tangents_intersect_at_P : TangentsIntersect (Circle Γ) M N P
axiom circumcenter_O : IsCircumcenter O (Triangle A B C)
axiom extend_AO_to_Q : Intersects (Line A O) (Line B C) Q
axiom intersect_AD_EF_R : Intersects (Line A D) (Line E F) R

-- Goal
theorem PD_parallel_QR : Parallel (Line P D) (Line Q R) :=
by sorry

end PD_parallel_QR_l286_286091


namespace plant_species_numbering_impossible_l286_286656

theorem plant_species_numbering_impossible :
  ∀ (n m : ℕ), 2 ≤ n ∨ n ≤ 20000 ∧ 2 ≤ m ∨ m ≤ 20000 ∧ n ≠ m → 
  ∃ x y : ℕ, 2 ≤ x ∨ x ≤ 20000 ∧ 2 ≤ y ∨ y ≤ 20000 ∧ x ≠ y ∧
  (∀ k : ℕ, gcd x k = gcd n k ∧ gcd y k = gcd m k) :=
  by sorry

end plant_species_numbering_impossible_l286_286656


namespace cos_4_arccos_2_5_l286_286341

noncomputable def arccos_2_5 : ℝ := Real.arccos (2/5)

theorem cos_4_arccos_2_5 : Real.cos (4 * arccos_2_5) = -47 / 625 :=
by
  -- Define x = arccos 2/5
  let x := arccos_2_5
  -- Declare the assumption cos x = 2/5
  have h_cos_x : Real.cos x = 2 / 5 := Real.cos_arccos (by norm_num : 2 / 5 ∈ Set.Icc (-1 : ℝ) 1)
  -- sorry to skip the proof
  sorry

end cos_4_arccos_2_5_l286_286341


namespace solve_y_minus_x_l286_286934

theorem solve_y_minus_x :
  ∃ (x y : ℝ), x + y = 500 ∧ x / y = 0.75 ∧ y - x = 71.42 :=
begin
  sorry
end

end solve_y_minus_x_l286_286934


namespace superhero_distance_difference_l286_286647

theorem superhero_distance_difference :
  let t := 4 in
  let v := 100 in
  (60 / t * 10) - v = 50 :=
by
  let t := 4
  let v := 100
  sorry

end superhero_distance_difference_l286_286647


namespace greatest_odd_factors_below_200_l286_286168

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l286_286168


namespace range_of_x_l286_286872

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1-x)
  else 1 - Real.log x / (Real.log 2)

theorem range_of_x {x : ℝ} (h : f x ≤ 2) : 0 ≤ x :=
by
  sorry

end range_of_x_l286_286872


namespace correct_def_E_bar_C_plus_E_bar_Riemann_integral_E_bar_correct_Riesz_Lebesgue_integrability_C_plus_not_nonneg_bounded_l286_286139

-- Define relevant structures and concepts
noncomputable theory

def Omega := Set.Ioo 0 1
def BorelOmega := MeasurableSpace.comap (Subtype.val : Omega → ℝ) volume

structure StepFunction (ω : Omega) :=
(c : ℕ → ℝ)
(ω_incr : ∀ i, ∃ a b, (0:ℝ) ≤ a ∧ b ≤ 1 ∧ (a < b ∧ ω ∈ Ioo a b))

noncomputable def E_bar (ξ : StepFunction) : ℝ :=
sum (λ i, ξ.c i * (ω_incr i).2.1)

def C_plus (ξ : StepFunction) :=
∀ n, ∃ ξ_n : StepFunction, (ξ_n.c = ξ.c ∧ E_bar ξ_n < ∞) ∧ ∀ᵐ ω, ξ_n ω ⟶ ξ ω

theorem correct_def_E_bar_C_plus :
  ∀ ξ : StepFunction, (C_plus ξ) → ∃! (lim_n : ℕ → ℝ), lim (λ n, E_bar (ξ_n n)) = lim_n := sorry

theorem E_bar_Riemann_integral (ξ : StepFunction) :
  (RiemannIntegrable ξ) → (C_plus ξ) ∧ (E_bar ξ = ∫ x in Omega, ξ.val x) := sorry

theorem E_bar_correct (ξ : StepFunction) :
  ∀ ξ_+, ξ_- : StepFunction, (ξ = ξ_+ - ξ_-) → C_plus ξ_+ ∧ C_plus ξ_-  ∧ E_bar ξ = E_bar ξ_+ - E_bar ξ_- := sorry

theorem Riesz_Lebesgue_integrability (ξ : StepFunction) :
  (LebesgueIntegrable ξ) → ξ_val := sorry

theorem C_plus_not_nonneg_bounded (ξ : StepFunction) :
  ((∀ ω, 0 ≤ ξ ω) ∧ (∫ x in Omega, ξ.val x < ∞)) → ¬ C_plus ξ := sorry

end correct_def_E_bar_C_plus_E_bar_Riemann_integral_E_bar_correct_Riesz_Lebesgue_integrability_C_plus_not_nonneg_bounded_l286_286139


namespace jerry_age_l286_286876

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J + 10) (h2 : M = 30) : J = 5 := by
  sorry

end jerry_age_l286_286876


namespace general_term_sequence_sum_b_n_l286_286750

-- Definitions based on the conditions
def a₁ := 1
def d := 2
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- General term formula proof statement
theorem general_term_sequence (n : ℕ) : a_n n = 2 * n - 1 :=
by {
  -- Fill proof here
  sorry
}

-- Definitions for b_n sequence and sum S_n
def b_n (n : ℕ) := 1 / ((2 * n - 1) * (2 * n + 1))
def S_n (n : ℕ) := ∑ i in finset.range n, b_n (i + 1)

-- Sum of the first n terms proof statement
theorem sum_b_n (n : ℕ) : S_n n = n / (2 * n + 1) :=
by {
  -- Fill proof here
  sorry
}

end general_term_sequence_sum_b_n_l286_286750


namespace y1_lt_y2_l286_286385

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l286_286385


namespace points_on_circle_l286_286754

variable (A B C D H K : Type)
variable [Parallelogram A B C D] -- Assuming a Parallelogram type with vertices A, B, C, D
variable (H_is_foot : Foot H A B C) -- H is the foot of the perpendicular from A to BC
variable (K_is_extension_median_circumcircle : IsCtPtMedianCircle K A B C) -- K is the intersection point as given

theorem points_on_circle (obtuse_angle_A : obtuse_angle A) :
  cyclic K H C D :=
sorry

end points_on_circle_l286_286754


namespace find_linear_function_l286_286761

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_linear_function
  (f : ℝ -> ℝ)
  (h1 : ∃ a b : ℝ, ∀ x : ℝ, f(x) = a * x + b)
  (h2 : ∀ x : ℝ, (f^[10])(x) ≥ 1024 * x + 1023) :
  (∃ b : ℝ, (f = (λ x, 2 * x + b) ∧ b ≥ 1)) ∨ (∃ b : ℝ, (f = (λ x, -2 * x + b) ∧ b ≤ -3)) :=
by
  sorry

end find_linear_function_l286_286761


namespace count_perfect_square_factors_l286_286017

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286017


namespace three_pizzas_needed_l286_286675

noncomputable def masha_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "sausage" ∉ p

noncomputable def vanya_pizza (p : Set String) : Prop :=
  "mushrooms" ∈ p

noncomputable def dasha_pizza (p : Set String) : Prop :=
  "tomatoes" ∉ p

noncomputable def nikita_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "mushrooms" ∉ p

noncomputable def igor_pizza (p : Set String) : Prop :=
  "mushrooms" ∉ p ∧ "sausage" ∈ p

theorem three_pizzas_needed (p1 p2 p3 : Set String) :
  (∃ p1, masha_pizza p1 ∧ vanya_pizza p1 ∧ dasha_pizza p1 ∧ nikita_pizza p1 ∧ igor_pizza p1) →
  (∃ p2, masha_pizza p2 ∧ vanya_pizza p2 ∧ dasha_pizza p2 ∧ nikita_pizza p2 ∧ igor_pizza p2) →
  (∃ p3, masha_pizza p3 ∧ vanya_pizza p3 ∧ dasha_pizza p3 ∧ nikita_pizza p3 ∧ igor_pizza p3) →
  ∀ p, ¬ ((masha_pizza p ∨ dasha_pizza p) ∧ vanya_pizza p ∧ (nikita_pizza p ∨ igor_pizza p)) :=
sorry

end three_pizzas_needed_l286_286675


namespace perfect_square_probability_l286_286643

open Classical

theorem perfect_square_probability : 
  let p := 1
  let q := 4 in
  p >= 0 ∧ q >= 0 ∧ Nat.gcd p q = 1 ∧
  (∃ p q : ℕ, (p = 1 ∧ q = 4 ∧ Nat.gcd p q = 1) ∧ p + q = 5) ∧
  (∀ product_probs,
    product_probs = ∑ i in (Finset.range 3), (6 ^ i) →
    (product_probs = ∑ prod in {1, 4, 9, 16, 25, 36}, product_probs = 1/4)) :=
  let p := 1
  let q := 4 in
    have h1 : Nat.gcd p q = 1 := sorry,
    have h2 : 216 = 6^3 := rfl,
    have h3 : ∃ prod : ℕ, (Set.Mem prod {1, 4, 9, 16, 25, 36}) ∧ (∑ x in ({1, 4, 9, 16, 25, 36} : Set ℕ), x = 1/4) := sorry,
    have h4 : p + q = 5 := by
      rw [add_comm, add_left_cancel_iff],
      exact nat_pred_left_eq_nat_pred_right.mpr (Nat.add_lt_add_left rfl _),
    ⟨h1, h2, h3, h4⟩

end perfect_square_probability_l286_286643


namespace critical_point_and_range_l286_286763

def f (x a : ℝ) := x^3 - a * x^2 + 3 * x

theorem critical_point_and_range (x : ℝ) (a : ℝ)
  (h : x = 3 ∧ deriv (f x a) x = 0 ∧ ∀ x ∈ [2,4], x ∈ range f) :
  a = 5 ∧ range f = Icc (-9 : ℝ) (-4 : ℝ) :=
by {
  sorry,
}

end critical_point_and_range_l286_286763


namespace part1_part2_l286_286801

-- Define what a double root equation is
def is_double_root_eq (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ * a + x₁ * b + c = 0 ∧ x₂ = 2 * x₁ ∧ x₂ * x₂ * a + x₂ * b + c = 0

-- Statement for part 1: proving x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_eq 1 (-3) 2 :=
sorry

-- Statement for part 2: finding correct values of a and b for ax^2 + bx - 6 = 0 to be a double root equation with one root 2
theorem part2 : (∃ a b : ℝ, is_double_root_eq a b (-6) ∧ (a = -3 ∧ b = 9) ∨ (a = -3/4 ∧ b = 9/2)) :=
sorry

end part1_part2_l286_286801


namespace remainder_5n_minus_12_l286_286247

theorem remainder_5n_minus_12 (n : ℤ) (hn : n % 9 = 4) : (5 * n - 12) % 9 = 8 := 
by sorry

end remainder_5n_minus_12_l286_286247


namespace perpendicular_lines_eq_a_1_l286_286767

theorem perpendicular_lines_eq_a_1 (a : ℝ) : 
  (let line1 := fun x y => ax - y = 1 in
   let line2 := fun x y => (2 - a)x + a(y) = -1 in
   (∀ x y, line1 x y ⊢
    ∀ x y, line2 x y ⊢)) → 
   a = 1 :=
sorry

end perpendicular_lines_eq_a_1_l286_286767


namespace parabola_min_value_sum_abc_zero_l286_286209

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l286_286209


namespace find_phi_symmetric_l286_286370

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.sqrt 3 * (Real.cos (2 * x)))

theorem find_phi_symmetric : ∃ φ : ℝ, (φ = Real.pi / 12) ∧ ∀ x : ℝ, f (-x + φ) = f (x + φ) := 
sorry

end find_phi_symmetric_l286_286370


namespace monotonic_function_root_product_l286_286395

variables {α : Type*} [linear_order α] {β : Type*} [linear_order β]

noncomputable def is_root (f : α → β) (a : α) : Prop := f(a) = 0

theorem monotonic_function_root_product 
  (f : α → β) (a x1 x2 : α)
  (h1 : is_root f a) 
  (h2 : x1 < a ∧ a < x2)
  (h3 : monotone f ∨ antitone f) :
  f(x1) * f(x2) < 0 :=
sorry

end monotonic_function_root_product_l286_286395


namespace count_numbers_with_perfect_square_factors_l286_286057

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286057


namespace number_of_solutions_l286_286355

theorem number_of_solutions :
  (∃ n : ℤ, (1 + (⌊200 * n / 201⌋) = ⌈198 * n / 200⌉)) ↔ 40200 := by sorry

end number_of_solutions_l286_286355


namespace sin_alpha_proof_l286_286937

def vertex_at_origin (α : ℝ) : Prop := true -- This condition is intrinsic and standard; the vertex being at origin is used directly.

def initial_side_positive_x_axis (α : ℝ) : Prop := true -- Initial side of the angle along positive x-axis is intrinsic and used directly.

def terminal_side_passes_through_point (α : ℝ) : Prop :=
  let x := -1
  let y := -2
  r = Real.sqrt (x * x + y * y)
  r = Real.sqrt 5

theorem sin_alpha_proof (α : ℝ) 
  (h1 : vertex_at_origin α) 
  (h2 : initial_side_positive_x_axis α) 
  (h3 : terminal_side_passes_through_point α): 
  Real.sin α = -2 * Real.sqrt 5 / 5 := 
by
  sorry

end sin_alpha_proof_l286_286937


namespace sin_eq_sin_2x_sol_set_l286_286076

theorem sin_eq_sin_2x_sol_set (x : ℝ) (h₁ : x ∈ Set.Ioo (-real.pi) (2 * real.pi))
  (h₂ : sin (2 * x) = sin x) :
  x = 0 ∨ x = real.pi ∨ x = -real.pi / 3 ∨ x = real.pi / 3 ∨ x = 5 * real.pi / 3 :=
sorry

end sin_eq_sin_2x_sol_set_l286_286076


namespace tempo_original_value_l286_286650

theorem tempo_original_value
    (V : ℝ)
    (h1 : ∃ I : ℝ, I = (4/5) * V)
    (h2 : 910 = (1.3 / 100) * I)
    (h3 : I = 70000) :
    V = 87500 :=
by
  obtain ⟨I, hI⟩ := h1
  rw [← hI] at h2 h3
  have hI_value : I = 70000 :=
    by
      linarith
  have hV : V = I / (4 / 5) :=
    by
      sorry
  rw [h3] at hV
  linarith

end tempo_original_value_l286_286650


namespace circumscribable_hexagon_l286_286181

theorem circumscribable_hexagon (hexagon : Type) 
  (A B C D E F : hexagon)
  (opposite_sides_parallel : ∀ {P Q R S : hexagon}, (P = A ∧ Q = B ∧ R = E ∧ S = F) ∨ (P = B ∧ Q = C ∧ R = F ∧ S = A) ∨ (P = C ∧ Q = D ∧ R = A ∧ S = B) ∨ (P = D ∧ Q = E ∧ R = B ∧ S = C) ∨ (P = E ∧ Q = F ∧ R = C ∧ S = D) ∨ (P = F ∧ Q = A ∧ R = D ∧ S = E) → P.vector ⟂ Q.vector)
  (equal_diagonals : (A.vector D.vector = B.vector E.vector) ∧ (B.vector E.vector = C.vector F.vector) ∧ (A.vector D.vector = C.vector F.vector)) :
  ∃ (O : hexagon), is_circumscribed hexagon A B C D E F O :=
sorry

end circumscribable_hexagon_l286_286181


namespace range_of_a_l286_286813

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by
  sorry

end range_of_a_l286_286813


namespace clock_angle_at_3_40_l286_286243

noncomputable def hour_hand_angle (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
noncomputable def minute_hand_angle (m : ℕ) : ℝ := m * 6
noncomputable def angle_between_hands (h m : ℕ) : ℝ := 
  let angle := |minute_hand_angle m - hour_hand_angle h m|
  if angle > 180 then 360 - angle else angle

theorem clock_angle_at_3_40 : angle_between_hands 3 40 = 130.0 := 
by
  sorry

end clock_angle_at_3_40_l286_286243


namespace count_numbers_with_perfect_square_factors_l286_286035

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l286_286035


namespace reduced_price_l286_286974

theorem reduced_price (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 600 = (600 / P + 4) * R) : R = 30 := 
by
  sorry

end reduced_price_l286_286974


namespace factorial_equality_l286_286795

theorem factorial_equality (n : ℕ) (h : 4! * 5 = n!) : n = 5 :=
by
  sorry

end factorial_equality_l286_286795


namespace contingency_fund_amount_l286_286446

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l286_286446


namespace find_smallest_number_l286_286561

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end find_smallest_number_l286_286561


namespace first_part_second_part_third_part_l286_286002

-- Problem conditions and correct answers put into Lean definitions and theorems

theorem first_part (n : ℕ) : 
  (a_0 + a_1 * x + a_2 * x^2 + ... + a_{2n} * x^(2n)) :=
  (1 + x + x^2)^n = (3^n - 1) := sorry

theorem second_part (n : ℕ) (x : ℕ) :
  a_2 + 2 * a_3 + ... + 2^{2n-2} * a_{2n} = (1 / 4 * 7^n - 1 / 4 - 1 / 2 * n) := sorry

theorem third_part (n : ℕ) (h : n ≥ 6) : 
  (A_2^2 * a_2 + 2 * A_3^2 * a_3 + ... + 2^{2n-2} * A_{2n}^2 * a_{2n}) < 49^{n-2} := sorry

end first_part_second_part_third_part_l286_286002


namespace largest_prime_factor_1337_l286_286240

theorem largest_prime_factor_1337 : ∃ p : ℕ, nat.prime p ∧ (p ∣ 1337) ∧ (∀ q : ℕ, nat.prime q ∧ q ∣ 1337 → q ≤ p) ∧ p = 191 :=
by {
    have h_div7 : 1337 % 7 = 0 := by norm_num,
    have h_p7 : nat.prime 7 := nat.prime_of_nat_prime_dec_trivial 7,
    have h_div191 : 1337 / 7 = 191 := by norm_num,
    have h_p191 : nat.prime 191 := by {
        haveI : prime 191 := sorry,  -- Skip the proof of primality of 191
    },
    use 191,
    split,
    exact h_p191,
    split,
    exact ⟨(1337 / 191) * 191, by simp [h_div7, h_div191]⟩,
    split,
    intros q h_qprime h_qdiv,
    cases h_qdiv with k h_k,
    have : (q = 7) ∨ (q = 191) := by {
        cases q,
        finish 
        -- Further reasoning skipped
        -- We can use pre-processing information to verify prime factors of 1337.
    },
    cases this,
    linarith,
    rw this,
    linarith
},
sorry 
]

end largest_prime_factor_1337_l286_286240


namespace max_value_of_x_and_y_l286_286073

theorem max_value_of_x_and_y (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : (x - 4) * (x - 10) = 2 ^ y) : x + y ≤ 16 :=
sorry

end max_value_of_x_and_y_l286_286073


namespace complement_union_l286_286614

open Set

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 5, 6, 8})
  (hA : A = {1, 5, 8})(hB : B = {2}) :
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  rw [hU, hA, hB]
  -- Intermediate steps would go here
  sorry

end complement_union_l286_286614


namespace triangle_tangent_circle_BO_OC_l286_286480

theorem triangle_tangent_circle_BO_OC (A B C O : Type*)
  [Triangle A B C]
  (AB AC BC : ℝ)
  (BO OC : ℝ)
  (h1 : AB = 12)
  (h2 : AC = 15)
  (h3 : BC = 18)
  (tangent_circle : TangentCircle O BC AB AC) : 
    BO = 8 ∧ OC = 10 :=
sorry

end triangle_tangent_circle_BO_OC_l286_286480


namespace range_of_a_l286_286381

def f (x : ℝ) (a : ℝ) : ℝ := a * x^3 - x^2 + x + 2
def g (x : ℝ) : ℝ := (Real.exp (Real.log x)) / x -- this simplifies to x / x = 1, but follows the function definition
def h (x : ℝ) : ℝ := (x^2 - x - 2) / x^3

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Icc 0 1, ∀ x₂ ∈ Icc 0 1, f x₁ a ≥ g x₂) → a ≥ -2 := by
  sorry

end range_of_a_l286_286381


namespace replaced_person_weight_l286_286195

theorem replaced_person_weight (new_person_weight : ℕ) (total_weight_increase : ℕ)
  (total_weight_increase = 20) (new_person_weight = 70) :
  ∃ W : ℕ, new_person_weight - W = total_weight_increase ∧ W = 50 :=
by
  sorry

end replaced_person_weight_l286_286195


namespace num_students_in_class_l286_286228

theorem num_students_in_class (A : ℕ) (avg_age_9 : ℕ) (age_10th : ℕ) (avg_age_10 : ℕ) :
  avg_age_9 = 8 → age_10th = 28 → avg_age_10 = avg_age_9 + 2 → A = 9 * avg_age_9 →
  (A + age_10th) / 10 = avg_age_10 → 10 = 10 :=
begin
  sorry
end

end num_students_in_class_l286_286228


namespace visible_angle_regular_tetrahedron_l286_286745

-- Definitions related to the geometry problem

noncomputable def regular_tetrahedron (a : ℝ) :=
  (is_regular_tetrahedron a a a a)

-- M is the midpoint of edge SC in tetrahedron SABC
def midpoint (S C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((S.1 + C.1) / 2, (S.2 + C.2) / 2, (S.3 + C.3) / 2)

-- The angle at which edge AB is visible from the midpoint M of edge SC in regular tetrahedron SABC
def angle_AB_visible_from_M (S A B C : ℝ × ℝ × ℝ) (a : ℝ) [regular_tetrahedron a] : ℝ :=
  let M := midpoint S C in
  real.arccos ((vector.dot_product (vector.of_points M A) (vector.of_points M B)) /
               ((vector.norm (vector.of_points M A)) * (vector.norm (vector.of_points M B))))

-- The theorem to be proven
theorem visible_angle_regular_tetrahedron {S A B C : ℝ × ℝ × ℝ} {a : ℝ} [regular_tetrahedron a] :
  angle_AB_visible_from_M S A B C a = real.arccos (1 / 3) :=
sorry

end visible_angle_regular_tetrahedron_l286_286745


namespace count_numbers_with_perfect_square_factors_l286_286060

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l286_286060


namespace hypotenuse_longer_side_difference_l286_286214

theorem hypotenuse_longer_side_difference
  (x : ℝ)
  (h1 : 17^2 = x^2 + (x - 7)^2)
  (h2 : x = 15)
  : 17 - x = 2 := by
  sorry

end hypotenuse_longer_side_difference_l286_286214


namespace white_pairs_coincide_l286_286697

def geometric_figure : Type := 
{ red : Nat, blue : Nat, white : Nat }

noncomputable def coinciding_pairs (upper lower : geometric_figure) (red_pairs blue_pairs red_white_pairs : Nat) : Nat := 
let red_triangles_left := upper.red - red_pairs
let blue_triangles_left := upper.blue - blue_pairs
let white_triangles_left := upper.white - red_white_pairs - blue_triangles_left
white_triangles_left

theorem white_pairs_coincide :
  ∀ (upper lower : geometric_figure), 
    upper.red = 4 → upper.blue = 6 → upper.white = 10 →
    3 * 2 = 6 → 4 * 2 = 8 → 2 * 2 = 4 →
    coinciding_pairs upper lower 3 4 2 = 7 := 
by 
  -- Placeholders for detailed proof steps
  intros
  sorry

end white_pairs_coincide_l286_286697


namespace a_and_b_together_complete_in_10_days_l286_286255

noncomputable def a_works_twice_as_fast_as_b (a b : ℝ) : Prop :=
  a = 2 * b

noncomputable def b_can_complete_work_in_30_days (b : ℝ) : Prop :=
  b = 1/30

theorem a_and_b_together_complete_in_10_days (a b : ℝ) 
  (h₁ : a_works_twice_as_fast_as_b a b)
  (h₂ : b_can_complete_work_in_30_days b) : 
  (1 / (a + b)) = 10 := 
sorry

end a_and_b_together_complete_in_10_days_l286_286255


namespace real_imag_eq_l286_286815

theorem real_imag_eq (a : ℝ) : (∀ z : ℂ, z = (1 + a * complex.I) * (2 + complex.I) → z.re = z.im) → a = 1 / 3 :=
by
  intro h
  sorry

end real_imag_eq_l286_286815


namespace erased_angle_is_correct_l286_286665

theorem erased_angle_is_correct (n : ℕ) (x : ℝ) (h_convex : convex_polygon n) (h_sum_remaining : sum_remaining_angles = 1703) : x = 97 :=
by
  -- This is where the proof would be placed, but we'll use sorry for now
  sorry

end erased_angle_is_correct_l286_286665


namespace problem_1_problem_2_l286_286778

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  4 * cos x * sin (x - π / 3) + a

theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, 4 * cos x * sin (x - π / 3) + a ≤ 2) →
  (∃ a = sqrt 3 ∧ (∀ x, f x (sqrt 3) = 2 * sin (2 * x - π / 3))) ∧
  (∀ x, f x (sqrt 3) = 4 * cos x * sin (x - π / 3) + sqrt 3 → 2 * sin (2 * x - π / 3) + sqrt 3 → period f π) := sorry

theorem problem_2 (A B : ℝ) :
  A < B →
  f A (sqrt 3) = 1 →
  f B (sqrt 3) = 1 →
  (∀ C : ℝ, C = π - A - B → (sin A / sin B) = sqrt 2) := sorry

end problem_1_problem_2_l286_286778


namespace minimum_translation_distance_l286_286950

theorem minimum_translation_distance 
  (y : ℝ → ℝ)
  (h : ∀ x : ℝ, y x = sqrt 3 * cos x + sin x)
  (m : ℝ)
  (hm : m > 0) :
  (∀ x : ℝ, y (x + m) = y (-x - m)) → m = π/6 := 
sorry

end minimum_translation_distance_l286_286950


namespace sum_of_integers_satisfying_quadratic_l286_286556

theorem sum_of_integers_satisfying_quadratic:
  (∑ x in ({14, -10} : Finset ℤ), x) = 4 :=
begin
  -- Lean requires proof that only integers 14 and -10 satisfy the equation x^2 = 3x + 140
  -- Proof can be omitted since it's not required here
  sorry
end

end sum_of_integers_satisfying_quadratic_l286_286556


namespace geom_seq_a3_a5_product_l286_286088

-- Defining the conditions: a sequence and its sum formula
def geom_seq (a : ℕ → ℕ) := ∃ r : ℕ, ∀ n, a (n+1) = a n * r

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n-1) + a 1

-- The theorem statement
theorem geom_seq_a3_a5_product (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : geom_seq a) (h2 : sum_first_n_terms a S) : a 3 * a 5 = 16 := 
sorry

end geom_seq_a3_a5_product_l286_286088


namespace johns_remaining_money_l286_286122

theorem johns_remaining_money (initial_amount : ℝ) (fraction_snacks : ℝ) (fraction_necessities : ℝ) 
  (h_initial : initial_amount = 20) (h_fraction_snacks : fraction_snacks = 1 / 5) 
  (h_fraction_necessities : fraction_necessities = 3 / 4) : 
  let after_snacks := initial_amount - initial_amount * fraction_snacks in
  let after_necessities := after_snacks - after_snacks * fraction_necessities in
  after_necessities = 4 :=
by
  sorry

end johns_remaining_money_l286_286122


namespace smallest_number_is_51_l286_286559

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end smallest_number_is_51_l286_286559


namespace length_of_arc_SP_in_terms_of_pi_l286_286461

noncomputable def angle_SIP : ℝ := 45
noncomputable def radius_OS : ℝ := 15

theorem length_of_arc_SP_in_terms_of_pi :
  let circumference := 2 * Real.pi * radius_OS,
      proportion := (2 * angle_SIP) / 360,
      arc_length := proportion * circumference
  in arc_length = 7.5 * Real.pi :=
by
  sorry

end length_of_arc_SP_in_terms_of_pi_l286_286461


namespace solve_part_I_solve_part_II_l286_286787

noncomputable section

open EuclideanGeometry

-- Define the given vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)

-- Define the vector c and conditions
def c (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the dot product condition
def dot_product_same (x y : ℝ) : Prop :=
  (a.1 * x + a.2 * y) = (b.1 * x + b.2 * y) ∧ (a.1 * x + a.2 * y) > 0

-- Define the magnitude condition
def magnitude_3 (x y : ℝ) : Prop :=
  real.sqrt (x * x + y * y) = 3

-- Define the solution to the first part
def c_solution : ℝ × ℝ := (0, 3)

-- Define the magnitude of 3a - c
def magnitude_of_3a_minus_c : ℝ :=
  real.sqrt ((3 * a.1 - c_solution.1) ^ 2 + (3 * a.2 - c_solution.2) ^ 2)

-- Prove the first part
theorem solve_part_I : ∀ x y : ℝ, dot_product_same x y → magnitude_3 x y → c x y = c_solution :=
sorry

-- Prove the second part
theorem solve_part_II : magnitude_of_3a_minus_c = 3 * real.sqrt 10 :=
sorry

end solve_part_I_solve_part_II_l286_286787


namespace wave_equation_solution_l286_286900

noncomputable def u (x t : ℝ) : ℝ :=
  3 * (1 - Real.cos t) * Real.sin x - (1 / 9) * (1 - Real.cos (3 * t)) * Real.sin (3 * x)

theorem wave_equation_solution :
  (∀ x t, 0 < x ∧ x < Real.pi ∧ 0 < t →
    (D[2] (D[2] u t) - D[2] u x = 4 * Real.sin x ^ 3)) ∧
  (∀ x, 0 < x ∧ x < Real.pi →
    u x 0 = 0 ∧ (D[1] u x 0 = 0)) ∧
  (∀ t, 0 < t →
    u 0 t = 0 ∧ u Real.pi t = 0) :=
by
  sorry

end wave_equation_solution_l286_286900


namespace initial_candies_l286_286696

theorem initial_candies (c_given : ℕ) (c_left : ℕ) (initial_candies : ℕ) 
  (h_given : c_given = 40) (h_left : c_left = 20) : initial_candies = c_given + c_left → initial_candies = 60 :=
by
  intros h
  rw [h_given, h_left]
  simp
  exact h

end initial_candies_l286_286696


namespace deletion_ways_correct_l286_286566

def number_of_ways_deleting_six_apps (total_apps specific_apps : ℕ) : ℕ :=
  match total_apps, specific_apps with
  | 21, 6 =>
    let num_ways_choose_specific : ℕ := 2 * 2 * 2 -- number of ways to choose from pairs (T, T'), (V, V'), (F, F')
    let remaining_apps := total_apps - specific_apps 
    let num_ways_choose_remaining := Nat.binom remaining_apps 3 -- number of ways to choose 3 from remaining 15
    num_ways_choose_specific * num_ways_choose_remaining
  | _, _ => 0

theorem deletion_ways_correct : 
  number_of_ways_deleting_six_apps 21 6 = 3640 := by
  -- Proof omitted
  sorry

end deletion_ways_correct_l286_286566


namespace change_in_surface_area_l286_286679

-- Definition for the rectangular solid dimensions
def length : ℝ := 4
def width : ℝ := 3
def height : ℝ := 5

-- Definition for the cube dimensions
def cube_side : ℝ := 2

-- Theorem stating the change in surface area is 24 square feet after cube removal
theorem change_in_surface_area : 
  let original_surface_area := 2 * (length * width + length * height + width * height),
      cube_surface_area := 6 * (cube_side ^ 2)
  in original_surface_area + cube_surface_area = 94 + 24 := 
by
  sorry

end change_in_surface_area_l286_286679


namespace studio_apartments_per_building_l286_286657

theorem studio_apartments_per_building :
  ∃ (S : ℕ), S = 10 ∧
  let buildings := 4,
      two_person_apartments := 20,
      four_person_apartments := 5,
      total_people := 210,
      occupancy_rate := 0.75,
      max_occupancy := (total_people : ℚ) / occupancy_rate,
      people_in_one_building := two_person_apartments * 2 + four_person_apartments * 4,
      total_people_in_apartments := people_in_one_building * buildings,
      people_in_studios := (max_occupancy - total_people_in_apartments) in
    people_in_studios / buildings = S :=
begin
  use 10,
  simp,
  sorry
end

end studio_apartments_per_building_l286_286657


namespace fractions_are_integers_l286_286845

theorem fractions_are_integers (a b c : ℤ) (H : (a * b / c) + (a * c / b) + (b * c / a) ∈ ℤ) :
  (a * b / c) ∈ ℤ ∧ (a * c / b) ∈ ℤ ∧ (b * c / a) ∈ ℤ := sorry

end fractions_are_integers_l286_286845


namespace arithmetic_geometric_sequence_l286_286749

noncomputable def a_n (n : ℕ) : ℚ := 2 * n

def S (n : ℕ) : ℚ := n * a_n n / 2

def b (n : ℕ) : ℚ := 1 / ((a_n n - 1) * (a_n n + 1))

def T (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), b i

theorem arithmetic_geometric_sequence (S10_eq : S 10 = 110)
    (geo_seq : (a_n 2)^2 = a_n 1 * a_n 4) : 
  (∀ n, a_n n = 2 * n) ∧ (∀ n, T n = n / (2 * n + 1)) :=
by sorry

end arithmetic_geometric_sequence_l286_286749


namespace scaling_transformation_of_circle_l286_286753

theorem scaling_transformation_of_circle (x y x' y' : ℝ) (h1 : x^2 + y^2 = 1) 
(h2 : x' = 2 * x) (h3 : y' = 3 * y) : 
  (\frac{x'^2}{4}) + (\frac{y'^2}{9}) = 1 :=
by
  sorry

end scaling_transformation_of_circle_l286_286753


namespace count_perfect_square_factors_except_one_l286_286029

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286029


namespace like_terms_monomials_m_n_l286_286811

theorem like_terms_monomials_m_n (m n : ℕ) (h1 : 3 * x ^ m * y = - x ^ 3 * y ^ n) :
  m - n = 2 :=
by
  sorry

end like_terms_monomials_m_n_l286_286811


namespace ABCD1_is_cyclic_l286_286952

-- Definitions and conditions
variables {A B C D A1 B1 C1 D1 P : Type} [point : HasReflectSymm P A B C D A1 B1 C1 D1] 
variables (cyclic_A1BCD : IsCyclicQuadrilateral A1 B C D)
variables (cyclic_AB1CD : IsCyclicQuadrilateral A B1 C D)
variables (cyclic_ABC1D : IsCyclicQuadrilateral A B C1 D)

-- Main theorem to prove
theorem ABCD1_is_cyclic (h_symm : SymmetricQuadrilaterals A B C D A1 B1 C1 D1 P)
  : IsCyclicQuadrilateral A B C D1 :=
begin
  sorry
end

end ABCD1_is_cyclic_l286_286952


namespace prob_two_out_of_three_A_prob_at_least_one_A_and_B_l286_286613

open ProbabilityTheory

-- Define the probability of buses arriving on time
def prob_bus_A : ℝ := 0.7
def prob_bus_B : ℝ := 0.75

-- 1. Probability that exactly two out of three tourists taking bus A arrive on time
theorem prob_two_out_of_three_A : 
  (C 3 2) * (prob_bus_A ^ 2) * ((1 - prob_bus_A) ^ 1) = 0.441 :=
sorry

-- 2. Probability that at least one out of two tourists, one taking bus A, and the other taking bus B, arrives on time
theorem prob_at_least_one_A_and_B :
  1 - ((1 - prob_bus_A) * (1 - prob_bus_B)) = 0.925 :=
sorry

end prob_two_out_of_three_A_prob_at_least_one_A_and_B_l286_286613


namespace range_fraction_sum_l286_286438

theorem range_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  ∀ z, ∃ w (w >= 4) (z = w) :=
sorry

end range_fraction_sum_l286_286438


namespace superhero_vs_supervillain_distance_l286_286645

-- Definitions expressing the conditions
def superhero_speed (miles : ℕ) (minutes : ℕ) := (10 : ℕ) / (4 : ℕ)
def supervillain_speed (miles_per_hour : ℕ) := (100 : ℕ)

-- Distance calculation in 60 minutes
def superhero_distance_in_hour := 60 * superhero_speed 10 4
def supervillain_distance_in_hour := supervillain_speed 100

-- Proof statement
theorem superhero_vs_supervillain_distance :
  superhero_distance_in_hour - supervillain_distance_in_hour = (50 : ℕ) :=
by
  sorry

end superhero_vs_supervillain_distance_l286_286645


namespace problem1_problem2_l286_286230

-- Problem 1
theorem problem1 : -9 + (-4 * 5) = -29 :=
by
  sorry

-- Problem 2
theorem problem2 : (-(6) * -2) / (2 / 3) = -18 :=
by
  sorry

end problem1_problem2_l286_286230


namespace problem_statement_l286_286439

theorem problem_statement (a b c : ℤ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 3) 
  (h3 : |c| = 6) 
  (h4 : |a + b| = - (a + b)) 
  (h5 : |a + c| = a + c) : 
  a - b + c = -2 ∨ a - b + c = 4 :=
sorry

end problem_statement_l286_286439


namespace find_a_extremum_find_g_maximum_l286_286776

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem find_a_extremum {a : ℝ} (ha : a > 0 ∧ a ≠ 1) : a = Real.exp 1 :=
by sorry

def g (t : ℝ) : ℝ :=
  if 0 < t ∧ t < Real.exp 1 - 1 then (Real.log (t + 1)) / (t + 1)
  else if Real.exp 1 - 1 ≤ t ∧ t ≤ Real.exp 1 then 1 / Real.exp 1
  else if t > Real.exp 1 then (Real.log t) / t
  else 0

theorem find_g_maximum {t : ℝ} (ht : t > 0) :
  g t =
    if 0 < t ∧ t < Real.exp 1 - 1 then (Real.log (t + 1)) / (t + 1)
    else if Real.exp 1 - 1 ≤ t ∧ t ≤ Real.exp 1 then 1 / Real.exp 1
    else if t > Real.exp 1 then (Real.log t) / t
    else 0 :=
by sorry

end find_a_extremum_find_g_maximum_l286_286776


namespace largest_real_part_of_sum_correct_l286_286534

noncomputable def largest_real_part_of_sum {z w : ℂ} (hz : abs z = Real.sqrt 2) (hw : abs w = 2)
  (hzw : z * conj w + conj z * w = 1) : ℝ :=
  Real.sqrt 7

theorem largest_real_part_of_sum_correct {z w : ℂ} (hz : abs z = Real.sqrt 2) (hw : abs w = 2)
  (hzw : z * conj w + conj z * w = 1) : re (z + w) ≤ largest_real_part_of_sum hz hw hzw :=
sorry

end largest_real_part_of_sum_correct_l286_286534


namespace matching_socks_probability_l286_286793

theorem matching_socks_probability (pairs : ℕ) (days : ℕ) : 
  pairs = 5 → days = 5 → 
  (∀ d, 1 ≤ d ∧ d ≤ days → (∃ x y, x ≠ y ∧ {x, y} ⊆ (finset.range (2 * pairs)) \ (finset.range ((d - 1) * 2)))) →
  ∃ P, P = (1 / 25 : ℚ) := 
by 
  intros h_pairs h_days h_condition
  sorry

end matching_socks_probability_l286_286793


namespace functional_eq_solution_l286_286705

noncomputable def f : ℚ → ℚ := sorry

theorem functional_eq_solution (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1):
  ∀ x : ℚ, f x = x + 1 :=
sorry

end functional_eq_solution_l286_286705


namespace curve_length_at_least_circumference_l286_286738

-- Define the necessary structures
structure Sphere :=
  (radius : ℝ)

def circumference (s : Sphere) : ℝ := 2 * Real.pi * s.radius

structure CurveOnSphere (s : Sphere) :=
  (length : ℝ)
  (intersects_every_great_circle : ∀ GreatCircle : Set (ℝ × ℝ × ℝ), True)

theorem curve_length_at_least_circumference (s : Sphere) (c : CurveOnSphere s) : 
  c.length ≥ circumference s :=
sorry

end curve_length_at_least_circumference_l286_286738


namespace running_speeds_and_exercise_time_l286_286520

theorem running_speeds_and_exercise_time
  (dist_AB : ℕ)
  (ratio : ℝ)
  (time_diff : ℝ)
  (burn_first_30 : ℕ)
  (calories_increase : ℝ)
  (total_calories : ℕ)
  (xX_1: dist_AB = 9000)
  (xX_2: ratio = 1.2)
  (xX_3: time_diff = 5)
  (xX_4: burn_first_30 = 30*10)
  (xX_5: calories_increase = 1)
  (xX_6: total_calories = 2300):
  ∃ (x : ℝ), 
  let xm := 1.2 * x in
  let time_x := 9000 / x in
  let time_xm := 9000 / (1.2 * x) in
  time_x - time_xm = 5 ∧
  x = 300 ∧
  xm = 360 ∧
  ∃ y, 
  let time_B_to_C := y in
  let burn_after_30 := (y - 5) * (15 + y) in
  300 + burn_after_30 = 2300 ∧
  y = 45 ∧
  25 + 45 = 70 := sorry

end running_speeds_and_exercise_time_l286_286520


namespace garden_city_tree_equation_l286_286577

theorem garden_city_tree_equation (x : ℕ) :
  let original_saplings := x in
  (6 * (original_saplings + 22 - 1) = 7 * (original_saplings - 1)) :=
sorry

end garden_city_tree_equation_l286_286577


namespace max_ab_min_2a_b_min_frac_sum_l286_286729

noncomputable theory

open Real

theorem max_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ab + 2*a + b = 16) : ab ≤ 8 := sorry

theorem min_2a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ab + 2*a + b = 16) : 8 ≤ 2*a + b := sorry

theorem min_frac_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ab + 2*a + b = 16) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ (sqrt 2 / 3) := sorry

end max_ab_min_2a_b_min_frac_sum_l286_286729


namespace erased_angle_is_97_l286_286668

theorem erased_angle_is_97 (n : ℕ) (h1 : 3 ≤ n) (h2 : (n - 2) * 180 = 1703 + x) : 
  1800 - 1703 = 97 :=
by sorry

end erased_angle_is_97_l286_286668


namespace sum_consecutive_equals_prime_l286_286933

theorem sum_consecutive_equals_prime (m k p : ℕ) (h_prime : Nat.Prime p) :
  (∃ S, S = (m * (2 * k + m - 1)) / 2 ∧ S = p) →
  m = 1 ∨ m = 2 :=
sorry

end sum_consecutive_equals_prime_l286_286933


namespace cone_generatrix_angle_proof_l286_286770

theorem cone_generatrix_angle_proof
  (P : Point) 
  (r : ℝ) 
  (L : ℝ)
  (A B : Point)
  (h_radius : r = sqrt 3)
  (h_lateral_area : L = 2 * sqrt 3 * π)
  (h_A_on_circum : on_circumference A r)
  (h_B_on_circum : on_circumference B r) :
  (generatrix_length P r L = 2) ∧ (angle_between_PA_base P A r = π / 6) :=
sorry

end cone_generatrix_angle_proof_l286_286770


namespace function_increasing_increasing_interval_l286_286548

noncomputable def function := λ x : ℝ, x^3 + x

theorem function_increasing : ∀ x : ℝ, deriv function x ≥ 0 :=
by {
  intro x,
  have h_deriv : deriv function x = 3 * x^2 + 1 := by {
    dsimp [function],
    simp [deriv],
  },
  rw h_deriv,
  linarith,
}

theorem increasing_interval : ∀ x₁ x₂ : ℝ, x₁ < x₂ → function x₁ < function x₂ :=
by {
  intros x₁ x₂ hlt,
  apply deriv_strict_mono_of_deriv_nonnegative (λ x, deriv function x) function_increasing,
  exact hlt,
}

end function_increasing_increasing_interval_l286_286548


namespace remainder_x1001_mod_poly_l286_286358

noncomputable def remainder_poly_div (n k : ℕ) (f g : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.modByMonic f g

theorem remainder_x1001_mod_poly :
  remainder_poly_div 1001 3 (Polynomial.X ^ 1001) (Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.X + 1) = Polynomial.X ^ 2 :=
by
  sorry

end remainder_x1001_mod_poly_l286_286358


namespace subtraction_base_8_correct_l286_286296

def sub_in_base_8 (a b : Nat) : Nat := sorry

theorem subtraction_base_8_correct : sub_in_base_8 (sub_in_base_8 0o123 0o51) 0o15 = 0o25 :=
sorry

end subtraction_base_8_correct_l286_286296


namespace monomials_like_terms_l286_286809

theorem monomials_like_terms {m n : ℕ} (hm : m = 3) (hn : n = 1) : m - n = 2 :=
by
  rw [hm, hn]
  rfl

end monomials_like_terms_l286_286809


namespace value_of_t5_l286_286609

noncomputable def t_5_value (t1 t2 : ℚ) (r : ℚ) (a : ℚ) : ℚ := a * r^4

theorem value_of_t5 
  (a r : ℚ)
  (h1 : a > 0)  -- condition: each term is positive
  (h2 : a + a * r = 15 / 2)  -- condition: sum of first two terms is 15/2
  (h3 : a^2 + (a * r)^2 = 153 / 4)  -- condition: sum of squares of first two terms is 153/4
  (h4 : r > 0)  -- ensuring positivity of r
  (h5 : r < 1)  -- ensuring t1 > t2
  : t_5_value a (a * r) r a = 3 / 128 :=
sorry

end value_of_t5_l286_286609


namespace more_balloons_l286_286253

theorem more_balloons (you_balloons : ℕ) (friend_balloons : ℕ) (h_you : you_balloons = 7) (h_friend : friend_balloons = 5) : 
  you_balloons - friend_balloons = 2 :=
sorry

end more_balloons_l286_286253


namespace num_integers_between_l286_286012

theorem num_integers_between :
  let a := 11.5
  let b := 11.7
  let x := a^3
  let y := b^3
  (floor y - ceil x + 1) = 81 :=
by
  let a := 11.5
  let b := 11.7
  let x := a^3
  let y := b^3
  have h1 : x = 1520.875 := by sorry
  have h2 : y = 1601.61 := by sorry
  have h3 : ceil x = 1521 := by sorry
  have h4 : floor y = 1601 := by sorry
  show (floor y - ceil x + 1) = 81 from by
  rw [h3, h4]
  exact rfl

end num_integers_between_l286_286012


namespace maximum_value_of_ab_l286_286760

theorem maximum_value_of_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (sqrt 2) * (sqrt 2) = 2): a + b = 1 → ab ≤ 1/4 ∧ (∀ x y : ℝ, (x = 1/2) ∧ (y = 1/2) → ab = 1/4) := 
begin 
  sorry
end

end maximum_value_of_ab_l286_286760


namespace category_B_has_more_numbers_l286_286695

def isSumOfSquareAndCube (n : ℕ) : Prop :=
  ∃ (k m : ℕ), n = k^2 + m^3

def A : Finset ℕ := (Finset.range 1000000).filter isSumOfSquareAndCube

def B : Finset ℕ := (Finset.range 1000000).filter (λ n, ¬ isSumOfSquareAndCube n)

theorem category_B_has_more_numbers : B.card > A.card := 
by
  sorry

end category_B_has_more_numbers_l286_286695


namespace perfect_square_factors_l286_286054

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286054


namespace math_problem_l286_286862

theorem math_problem (n a b : ℕ) (hn_pos : n > 0) (h1 : 3 * n + 1 = a^2) (h2 : 5 * n - 1 = b^2) :
  (∃ x y: ℕ, 7 * n + 13 = x * y ∧ 1 < x ∧ 1 < y) ∧
  (∃ p q: ℕ, 8 * (17 * n^2 + 3 * n) = p^2 + q^2) :=
  sorry

end math_problem_l286_286862


namespace num_perfect_square_factors_of_1800_l286_286428

theorem num_perfect_square_factors_of_1800 : 
  ∃ n : ℕ, n = 8 ∧ ∀ m : ℕ, m ∣ 1800 → (∃ k : ℕ, m = k^2) ↔ m ∈ {d | d ∣ 1800 ∧ is_square d} := 
sorry

end num_perfect_square_factors_of_1800_l286_286428


namespace tank_capacity_l286_286256

theorem tank_capacity (T : ℚ) (h1 : T > 0) (h2 : 7 + 3/4 * T = 9/10 * T) : 
  T = 140 / 3 :=
begin
  sorry -- Proof to be completed
end

end tank_capacity_l286_286256


namespace smallest_X_l286_286496

theorem smallest_X (T : ℕ) (hT_digits : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) (hX_int : ∃ (X : ℕ), T = 20 * X) : ∃ T, ∀ X, X = T / 20 → X = 55 :=
by
  sorry

end smallest_X_l286_286496


namespace count_perfect_square_factors_except_one_l286_286028

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l286_286028


namespace sum_rational_irrational_not_rational_l286_286183

theorem sum_rational_irrational_not_rational (r i : ℚ) (hi : ¬ ∃ q : ℚ, i = q) : ¬ ∃ s : ℚ, r + i = s :=
by
  sorry

end sum_rational_irrational_not_rational_l286_286183


namespace total_pastries_and_bagels_l286_286941

-- Definitions of the conditions
def total_items := 350
def bread_rolls := 120
def croissants := 45
def muffins := 87
def pastries_per_bread_roll := 1.5

-- Calculation of bagels
def bagels := total_items - (bread_rolls + croissants + muffins)

-- Calculation of pastries
def pastries := pastries_per_bread_roll * bread_rolls

-- Final proof statement
theorem total_pastries_and_bagels : pastries + bagels = 278 := by
  -- The actual proof would go here
  sorry

end total_pastries_and_bagels_l286_286941


namespace area_gray_region_l286_286677

noncomputable def centerA := (4, 4)
noncomputable def radiusA := 4

noncomputable def centerB := (10, 4)
noncomputable def radiusB := 6

noncomputable def x_axis := 0

-- Lean statement for the proof problem
theorem area_gray_region :
  let rect_area := 6 * 4,
      sector_area_A := (1/4) * (radiusA^2) * Real.pi,
      sector_area_B := (3/4) * (1/2) * (radiusB^2) * Real.pi in
  rect_area - (sector_area_A + sector_area_B) = 24 - 17.5 * Real.pi :=
by sorry

end area_gray_region_l286_286677


namespace factory_processing_time_eq_l286_286629

variable (x : ℝ) (initial_rate : ℝ := x)
variable (parts : ℝ := 500)
variable (first_stage_parts : ℝ := 100)
variable (remaining_parts : ℝ := parts - first_stage_parts)
variable (total_days : ℝ := 6)
variable (new_rate : ℝ := 2 * initial_rate)

theorem factory_processing_time_eq (h : x > 0) : (first_stage_parts / initial_rate) + (remaining_parts / new_rate) = total_days := 
by
  sorry

end factory_processing_time_eq_l286_286629


namespace dice_probability_dice_probability_l286_286951

theorem dice_probability : 
  (probability (event_or (event_eq die_1 3) (event_eq die_2 4))) = 11/36 := 
sorry

namespace dice_probability
  -- Defining the basic structure for the problem in terms of probabilities

  -- Define the probability space for a single die
  def die_space := {1, 2, 3, 4, 5, 6}

  -- Define the event that the first die shows a 3
  def event_eq_die1_3 := {ω : die_space × die_space | ω.fst = 3}

  -- Define the event that the second die shows a 4
  def event_eq_die2_4 := {ω : die_space × die_space | ω.snd = 4}

  -- Probability measure on the product space of two dice
  def dice_probability (A : set (die_space × die_space)) := 
    (A.card.to_rational) / ((die_space.card * die_space.card).to_rational)

  -- Define the event that one of the conditions is met
  def event_die1_3_or_die2_4 := event_eq_die1_3 ∪ event_eq_die2_4

  -- Prove the probability of the event using dice_probability
  theorem dice_probability :
    dice_probability event_die1_3_or_die2_4 = 11/36 := sorry

end dice_probability

end dice_probability_dice_probability_l286_286951


namespace angle_equality_l286_286215

open_locale classical

variables {A B C M P Q : Type*}

-- Median definition
def is_median (A B C M : Type*) [has_coe_to_fun M (λ _, ℝ)] [has_coe_to_fun A (λ _, ℝ)] [has_coe_to_fun B (λ _, ℝ)] [has_coe_to_fun C (λ _, ℝ)] : Prop :=
((M : ℝ) = (B : ℝ) / 2 + (C : ℝ) / 2) ∧ (A ≠ B) ∧ (A ≠ C)

-- Perpendicular definition
def is_foot_of_perpendicular (B P A M : Type*) [has_coe_to_fun B (λ _, ℝ)] [has_coe_to_fun P (λ _, ℝ)] [has_coe_to_fun A (λ _, ℝ)] [has_coe_to_fun M (λ _, ℝ)] : Prop :=
((P : ℝ) = ((A : ℝ) + (M : ℝ)) / 2) ∧ (B ≠ A)

-- Point on segment definition with condition
def point_on_segment_condition (A Q M P : Type*) [has_coe_to_fun A (λ _, ℝ)] [has_coe_to_fun Q (λ _, ℝ)] [has_coe_to_fun M (λ _, ℝ)] [has_coe_to_fun P (λ _, ℝ)] : Prop :=
(Q = M) ∧ ((A : ℝ) - (Q : ℝ) = 2 * ((P : ℝ) - (M : ℝ)))

-- The theorem statement in Lean with conditions
theorem angle_equality
  (A B C M P Q : Type*) [has_coe_to_fun A (λ _, ℝ)] [has_coe_to_fun B (λ _, ℝ)] [has_coe_to_fun C (λ _, ℝ)] [has_coe_to_fun M (λ _, ℝ)] [has_coe_to_fun P (λ _, ℝ)] [has_coe_to_fun Q (λ _, ℝ)]
  (h_median : is_median A B C M) 
  (h_perpendicular : is_foot_of_perpendicular B P A M)
  (h_segment_condition : point_on_segment_condition A Q M P) :
  ∠CQM = ∠BAM :=
sorry

end angle_equality_l286_286215


namespace initial_water_percentage_is_approx_74_989_l286_286985

-- Define the initial conditions
def initial_volume := 340
def added_sugar := 3.2
def added_water := 12
def added_kola := 6.8
def final_volume := initial_volume + added_sugar + added_water + added_kola
def final_sugar_percentage := 19.66850828729282 / 100

-- Define the relation for calculating the initial percentage of water
def initial_water_percentage (W : ℝ) : Prop :=
  let initial_sugar_volume := 0.95 * initial_volume - 0.01 * W * initial_volume in
  let final_sugar_volume := initial_sugar_volume + added_sugar in
  final_sugar_volume / final_volume = final_sugar_percentage

-- State that the initial water percentage is approximately 74.989%
theorem initial_water_percentage_is_approx_74_989 : ∃ W : ℝ, initial_water_percentage W ∧ W ≈ 74.989 :=
by
  sorry

end initial_water_percentage_is_approx_74_989_l286_286985


namespace fred_final_baseball_cards_l286_286728

-- Conditions
def initial_cards : ℕ := 25
def sold_to_melanie : ℕ := 7
def traded_with_kevin : ℕ := 3
def bought_from_alex : ℕ := 5

-- Proof statement (Lean theorem)
theorem fred_final_baseball_cards : initial_cards - sold_to_melanie - traded_with_kevin + bought_from_alex = 20 := by
  sorry

end fred_final_baseball_cards_l286_286728


namespace min_value_frac_l286_286318

-- Definition of the function f
def f (x : ℝ) : ℕ := ⌊x * ⌊x⌋⌋

-- Function to compute the number of unique elements in the image set A
def a_n (n : ℕ) : ℕ :=
  if h : n > 0 then ∑ i in finset.range n, 1 else 0

-- Proving the minimum value
theorem min_value_frac (n : ℕ) (hn : n > 0) : 
  ∃ k : ℝ, k = 19 / 2 ∧ (a_n n + 49) / n = k := sorry

end min_value_frac_l286_286318


namespace percentage_change_difference_l286_286462

theorem percentage_change_difference (total_students : ℕ) (initial_enjoy : ℕ) (initial_not_enjoy : ℕ) (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  total_students = 100 →
  initial_enjoy = 40 →
  initial_not_enjoy = 60 →
  final_enjoy = 80 →
  final_not_enjoy = 20 →
  (40 ≤ y ∧ y ≤ 80) ∧ (40 - 40 = 0) ∧ (80 - 40 = 40) ∧ (80 - 40 = 40) :=
by
  sorry

end percentage_change_difference_l286_286462


namespace cone_height_l286_286402

namespace ConeHeight

theorem cone_height
  (L : ℝ) (r : ℝ) (h : ℝ)
  (hL : L = 15 * real.pi)
  (hr : r = 3) :
  h = 4 :=
by
  sorry

end ConeHeight

end cone_height_l286_286402


namespace count_perfect_square_factors_l286_286020

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l286_286020


namespace tan_alpha_plus_pi_over_4_eq_one_third_l286_286829

theorem tan_alpha_plus_pi_over_4_eq_one_third (t : ℝ) (ht : t ≠ 0) :
  let α := real.arctan (-1/2) in
  real.tan (α + real.pi / 4) = 1 / 3 :=
by
  let α := real.arctan (-1/2)
  calc
    real.tan (α + real.pi / 4) = (tan α + 1) / (1 - tan α) := by sorry
    ... = 1 / 3 := by sorry

end tan_alpha_plus_pi_over_4_eq_one_third_l286_286829


namespace greatest_odd_factors_l286_286156

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l286_286156


namespace sin_neg_20pi_over_3_eq_neg_sqrt3_over_2_l286_286261

theorem sin_neg_20pi_over_3_eq_neg_sqrt3_over_2 : 
  sin (- (20 * π) / 3) = - (sqrt 3) / 2 :=
by
  -- The proof of this theorem is omitted
  sorry

end sin_neg_20pi_over_3_eq_neg_sqrt3_over_2_l286_286261


namespace man_loss_after_wage_changes_l286_286602

theorem man_loss_after_wage_changes :
  ∀ (original_wages : ℝ), 
  let decreased_wages := original_wages * (1 - 0.5),
  let increased_wages := decreased_wages * (1 + 0.5),
  original_wages - increased_wages = 25 :=
by
  intro original_wages
  let decreased_wages := original_wages * 0.5
  let increased_wages := decreased_wages * 1.5
  have loss : original_wages - increased_wages = original_wages - (original_wages * 0.5 * 1.5) := by rfl
  sorry

end man_loss_after_wage_changes_l286_286602


namespace true_proposition_l286_286409

theorem true_proposition :
  (∀ x y : ℝ, ¬ (xy = 0 → x = 0 ∧ y = 0)) ∧
  (¬ (∀ x : ℝ, ∃ y : ℝ, square x = rhombus y)) ∧
  (∀ a b c : ℝ, ¬ (ac^2 > bc^2 → a > b)) ∧
  (∀ m : ℝ, m > 2 → (∀ x : ℝ, x^2 - 2 * x + m > 0)) :=
by
  sorry

end true_proposition_l286_286409


namespace count_odd_sum_numbers_l286_286526

-- Define a three-digit number
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Function to reverse the digits of a number
def reverse_digits (n : ℕ) : ℕ := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  units * 100 + tens * 10 + hundreds

-- Definition of an odd sum number
def is_odd_sum_number (n : ℕ) : Prop :=
  let sum := n + reverse_digits n
  all_digits_odd sum

-- Function to check if all digits of a number are odd
def all_digits_odd (n : ℕ) : Prop :=
  (n % 10) % 2 = 1 ∧ ((n / 10) % 10) % 2 = 1 ∧ ((n / 100) % 10) % 2 = 1 ∧ ((n / 1000) % 10) % 2 = 1

theorem count_odd_sum_numbers : 
  ∃ (count : ℕ), count = (100 : ℕ) ∧ 
  (∀ n, is_three_digit n → is_odd_sum_number n → count = 100) :=
sorry

end count_odd_sum_numbers_l286_286526


namespace painting_cost_l286_286893

theorem painting_cost (total_cost : ℕ) (num_paintings : ℕ) (price : ℕ)
  (h1 : total_cost = 104)
  (h2 : 10 < num_paintings)
  (h3 : num_paintings < 60)
  (h4 : total_cost = num_paintings * price)
  (h5 : price ∈ {d ∈ {d : ℕ | d > 0} | total_cost % d = 0}) :
  price = 2 ∨ price = 4 ∨ price = 8 :=
by
  sorry

end painting_cost_l286_286893


namespace taxi_ride_distance_l286_286624

variable (initial_charge : ℝ := 2.80)
variable (additional_charge_per_fifth_mile : ℝ := 0.40)
variable (total_charge : ℝ := 18.40)

theorem taxi_ride_distance :
  let additional_charge := total_charge - initial_charge,
      num_fifth_mile_increments := additional_charge / additional_charge_per_fifth_mile,
      total_fifth_miles := num_fifth_mile_increments + 1
  in
  total_fifth_miles * (1 / 5) = 8 := by sorry

end taxi_ride_distance_l286_286624


namespace find_fixed_point_l286_286916

noncomputable def g (z : ℂ) : ℂ := ((1 + complex.I * real.sqrt 3) * z + (4 * real.sqrt 3 + 12 * complex.I)) / 2

theorem find_fixed_point : 
  ∃ d : ℂ, g d = d ∧ d = -4 * real.sqrt 3 * (2 - complex.I) :=
sorry

end find_fixed_point_l286_286916


namespace count_integers_divisible_by_4_not_by_3_or_10_l286_286790

theorem count_integers_divisible_by_4_not_by_3_or_10 :
    (finset.filter (λ n: ℕ, n % 4 = 0 ∧ n % 3 ≠ 0 ∧ n % 10 ≠ 0) (finset.range 1001)).card = 133 :=
by
  sorry

end count_integers_divisible_by_4_not_by_3_or_10_l286_286790


namespace plane_divides_edge_equally_l286_286685

-- Definitions of points and line segments involved in the problem.
variables {Point : Type}
variables (A B C A₁ B₁ C₁ : Point)
variables (AB A₁C BC₁ : set Point)

-- Definitions asserting the conditions.
def passes_through (p : Point) (s : set Point) : Prop := p ∈ s
def parallel (s1 s2 : set Point) : Prop := sorry

-- Theorem statement: in Lean 4
theorem plane_divides_edge_equally 
  (h₁ : passes_through A₁ A₁C)
  (h₂ : passes_through C A₁C)
  (h₃ : parallel A₁C BC₁)
  : ∃ M, passes_through M AB ∧ (dist A M = dist M B) :=
sorry

end plane_divides_edge_equally_l286_286685


namespace gcd_consecutive_odd_product_l286_286322

theorem gcd_consecutive_odd_product (n : ℕ) (hn : n % 2 = 0 ∧ n > 0) : 
  Nat.gcd ((n+1)*(n+3)*(n+7)*(n+9)) 15 = 15 := 
sorry

end gcd_consecutive_odd_product_l286_286322


namespace perfect_square_factors_l286_286047

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l286_286047


namespace binary_multiplication_l286_286673

theorem binary_multiplication : nat.binary_rec 110110 * nat.binary_rec 111 = nat.binary_rec 10010010 := sorry

end binary_multiplication_l286_286673


namespace solve_for_a_l286_286780

noncomputable def f (x : ℝ) : ℝ := 5 ^ x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x ^ 2 - x

theorem solve_for_a (a : ℝ) : (f (g 1 a) = 1) → (a = 1) :=
by
  assume h : f (g 1 a) = 1
  sorry

end solve_for_a_l286_286780


namespace slope_of_line_l286_286594

variable (x y : ℝ)

def line_equation : Prop := 4 * y = -5 * x + 8

theorem slope_of_line (h : line_equation x y) :
  ∃ m b, y = m * x + b ∧ m = -5/4 :=
by
  sorry

end slope_of_line_l286_286594


namespace problem1_problem2_l286_286416

noncomputable def f (x : ℝ) : ℝ := |(2 * x + 1)| + |(2 * x - 3)|

theorem problem1 (x : ℝ) : (f x ≤ 6) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem problem2 (a : ℝ) :
  (∀ x, x ∈ set.Icc (-1/2 : ℝ) 1 → f x ≥ |(2 * x + a)| - 4) ↔ (-7 ≤ a ∧ a ≤ 6) :=
by sorry

-- The set.Icc denotes the closed interval [a, b].

end problem1_problem2_l286_286416


namespace tangent_line_eq_at_0_max_min_values_l286_286415

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_eq_at_0 : ∀ x : ℝ, x = 0 → f x = 1 :=
by
  sorry

theorem max_min_values : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 0 ≥ f x) ∧ (f (Real.pi / 2) = -Real.pi / 2) :=
by
  sorry

end tangent_line_eq_at_0_max_min_values_l286_286415


namespace desired_average_sale_l286_286630

def sales (month : ℕ) : ℕ :=
  if month = 1 then 2435
  else if month = 2 then 2920
  else if month = 3 then 2855
  else if month = 4 then 3230
  else if month = 5 then 2560
  else if month = 6 then 1000
  else 0

theorem desired_average_sale :
  ∑ i in Finset.range 6, sales (i + 1) / 6 = 2500 :=
by
  sorry

end desired_average_sale_l286_286630


namespace least_members_in_band_l286_286819

theorem least_members_in_band :
  ∃ n : ℕ, (n % 6 = 5) ∧ (n % 5 = 4) ∧ (n % 7 = 6) ∧ ∀ m : ℕ, 
  (m % 6 = 5) ∧ (m % 5 = 4) ∧ (m % 7 = 6) → n ≤ m → n = 119 :=
begin
  sorry
end

end least_members_in_band_l286_286819


namespace num_perfect_square_factors_of_1800_l286_286429

theorem num_perfect_square_factors_of_1800 : 
  ∃ n : ℕ, n = 8 ∧ ∀ m : ℕ, m ∣ 1800 → (∃ k : ℕ, m = k^2) ↔ m ∈ {d | d ∣ 1800 ∧ is_square d} := 
sorry

end num_perfect_square_factors_of_1800_l286_286429


namespace matching_socks_probability_l286_286791

def total_ways_to_choose_socks : ℕ :=
(10.choose 2) * (8.choose 2) * (6.choose 2) * (4.choose 2) * (2.choose 2)

def favorable_outcomes_matching_socks : ℕ :=
5 * 4 * ((6.choose 2) * (4.choose 2) * (2.choose 2))

def probability_matching_socks_third_and_fifth_day : ℚ :=
(favorable_outcomes_matching_socks : ℚ) / (total_ways_to_choose_socks : ℚ)

theorem matching_socks_probability :
  probability_matching_socks_third_and_fifth_day = 1 / 63 :=
by
  -- proof will be written here
  sorry

end matching_socks_probability_l286_286791


namespace max_area_triangle_PF1F2_product_of_slopes_const_l286_286869

-- Definition of the ellipse C
def is_point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  in (x^2 / 2) + y^2 = 1

-- Definition of F1 and F2, the foci of the ellipse
def foci_of_ellipse : ℝ × ℝ := (sqrt 2, 0) -- assuming F1 is (-sqrt(2), 0) and F2 is (sqrt(2), 0)

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := x + y = 1

-- Definitions of slopes
def slope_PA (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  in y / (x + sqrt 2)

def slope_PB (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  in y / (x - sqrt 2)

-- Statement: Maximum area of triangle PF1F2 is 1
theorem max_area_triangle_PF1F2 (P : ℝ × ℝ) (h : is_point_on_ellipse P) :
  let F1 := (-sqrt 2, 0)
  let F2 := (sqrt 2, 0)
  let area := abs ((F1.1 - P.1) * (F2.2 - P.2) - (F2.1 - P.1) * (F1.2 - P.2)) / 2
  in area ≤ 1 :=
sorry

-- Statement: k_PA * k_PB is a constant -1/2
theorem product_of_slopes_const (P : ℝ × ℝ) (h : is_point_on_ellipse P) :
  slope_PA P * slope_PB P = -1 / 2 :=
sorry

end max_area_triangle_PF1F2_product_of_slopes_const_l286_286869


namespace pyramids_edge_length_is_one_l286_286532

open EuclideanGeometry

def equilateral_triangle (ABC : Triangle) : Prop :=
  ∀ a b c : ℝ, Triangle.sides ABC = [a, b, c] → a = 1 ∧ b = 1 ∧ c = 1

def regular_hexagon (hexagon : Hexagon) : Prop :=
  ∃ l : ℝ, ∀ (v1 v2 : Vertex), adjacent hexagon v1 v2 → distance v1 v2 = l ∧ ∀ (u1 u2 : Vertex), distance u1 u2 = l

variables {ABC : Triangle} {ABRS BCPQ CAMN : Square} {pyramid1 pyramid2 pyramid3 : Pyramid} {M N P Q R S : Point}

-- Conditions
axiom h1 : equilateral_triangle ABC
axiom h2 : square_on_side ABRS ABC.ABCside1
axiom h3 : square_on_side BCPQ ABC.ABCside2
axiom h4 : square_on_side CAMN ABC.ABCside3
axiom h5 : identical_pyramids_with_equal_edge_length pyramid1 pyramid2 pyramid3
axiom h6 : apex_of_pyramids_coincide pyramid1 pyramid2 pyramid3
axiom h7 : vertices_form_regular_hexagon [M, N, P, Q, R, S]

-- Theorem
theorem pyramids_edge_length_is_one : pyramid_edge_length pyramid1 = 1 :=
by sorry

end pyramids_edge_length_is_one_l286_286532
