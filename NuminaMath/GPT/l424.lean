import Data.Finset.Basic
import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Algebra.Order
import Mathlib.Algebra.Seq.Lemmas
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trig
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorial.LatticePaths
import Mathlib.Combinatorics.Combination
import Mathlib.Combinatorics.CombinatorialNumbers
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Log
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Algebra.Order
import Mathlib.Probability.DiscreteUniform
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace maximum_value_of_complex_expression_l424_424765

noncomputable def max_complex_expression (z : ℂ) (hz : complex.abs z = 2) : ℝ :=
  complex.abs ((z - 2) * (z + 2)^2)

theorem maximum_value_of_complex_expression :
  ∃ z : ℂ, complex.abs z = 2 ∧ max_complex_expression z (by sorry) = 16 * real.sqrt 2 :=
sorry

end maximum_value_of_complex_expression_l424_424765


namespace arithmetic_sequence_minimum_sum_l424_424659

/-- 
Problem 1: 
Given a sequence {a_n}, where a_n is a positive natural number, and the sum of the first n terms 
S_n is given by  (1/8) * (a_n + 2)^2. Prove that {a_n} is an arithmetic sequence.
-/
theorem arithmetic_sequence (a : ℕ → ℕ) (h_pos : ∀ n, a n > 0) (h_sum : ∀ n, (finset.range n).sum a = (1 / 8) * (a n + 2) ^ 2) : 
  ∃ d, ∀ n, a (n + 1) = a n + d := 
sorry

/-- 
Problem 2: 
Given b_n = (1/2) * a_n - 30, prove that the minimum value of the sum of the first n terms 
of the sequence {b_n} is -225.
-/
theorem minimum_sum (a b : ℕ → ℤ) (h_a_seq : ∀ n, a (n + 1) = a n + 4)
  (h_b_def : ∀ n, b n = (1 / 2 : ℤ) * a n - 30) : 
  ∃ n, (finset.range n).sum b = -225 := 
sorry

end arithmetic_sequence_minimum_sum_l424_424659


namespace correct_statements_l424_424735

section Probability
open MeasureTheory ProbabilityTheory

variables {Ω : Type*} {P : Measure Ω} [ProbabilitySpace Ω]
variables (A B A₁ A₂ : Set Ω)

-- Conditions
def jarA_red_balls : ℕ := 3
def jarA_white_balls : ℕ := 2
def jarB_red_balls : ℕ := 4
def jarB_white_balls : ℕ := 1

-- Events
def event_A₁ : Set Ω := A₁  -- Drawing a red ball from jar A
def event_A₂ : Set Ω := A₂  -- Drawing a white ball from jar A
def event_B : Set Ω := B    -- Drawing a red ball from jar B

-- Probabilities
noncomputable def P_A₁ : ℝ := 3 / 5
noncomputable def P_A₂ : ℝ := 2 / 5
noncomputable def P_B_given_A₁ : ℝ := 5 / 6
noncomputable def P_B_given_A₂ : ℝ := 2 / 3

theorem correct_statements :
  (P (event_B) = 23 / 30) ∧
  Disjoint event_A₁ event_A₂ :=
sorry

end Probability

end correct_statements_l424_424735


namespace problem_l424_424802

theorem problem (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 := by
  sorry

end problem_l424_424802


namespace simplify_expression_l424_424421

theorem simplify_expression (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 5) * (2 * x - 1) - 
  (2 * x - 1) * (x^2 + 2 * x - 8) + 
  (x^2 - 2 * x + 3) * (2 * x - 1) * (x - 2) = 
  8 * x^4 - 2 * x^3 - 5 * x^2 + 32 * x - 15 := 
  sorry

end simplify_expression_l424_424421


namespace prove_monotonicity_solve_inequality_l424_424238

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (3 * x + b) / (a * x^2 + 4)

theorem prove_monotonicity
  (h_odd : ∀ x : ℝ, f 1 0 x = -f 1 0 (-x))
  (h_value : f 1 0 1 = 3 / 5) :
  ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 → f 1 0 x1 < f 1 0 x2 :=
sorry

theorem solve_inequality
  (h_odd : ∀ x : ℝ, f 1 0 x = -f 1 0 (-x))
  (h_value : f 1 0 1 = 3 / 5) :
  ∀ m : ℝ, sqrt(2) - 1 < m ∧ m < 1 → f 1 0 (m^2 + 1) + f 1 0 (2 * m - 2) > 0 :=
sorry

end prove_monotonicity_solve_inequality_l424_424238


namespace find_pairs_l424_424153

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 1 + 5^a = 6^b → (a, b) = (1, 1) := by
  sorry

end find_pairs_l424_424153


namespace find_coordinates_of_Q_l424_424807

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 25

noncomputable def point_on_line (x : ℝ) (p : ℝ) : Prop :=
  x = 11

noncomputable def area_triangle (ax ay bx by cx cy : ℝ) : ℝ :=
  0.5 * abs (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

theorem find_coordinates_of_Q (x y : ℝ) (p : ℝ) :
  point_on_circle x y →
  x * x + y * y = 25 →
  area_triangle (-5) 0 (5) 0 x y = 4 * area_triangle (-5) 0 11 p 11 0 →
  (x = -2.6) ∧ (y = 4.27) :=
sorry

end find_coordinates_of_Q_l424_424807


namespace ratio_of_x_intercepts_l424_424880

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l424_424880


namespace emily_gre_preparation_period_l424_424618

theorem emily_gre_preparation_period : 
  ∀ (months : ℕ), 
  months = 4 ↔ 
  ((months = (November - 1) - June + 1) := 4) := begin
  sorry
end

end emily_gre_preparation_period_l424_424618


namespace probability_correct_l424_424548

noncomputable def probability_two_faces_and_none_face : ℚ := 
  let total_unit_cubes := 64
  let cubes_with_two_painted_faces := 4
  let cubes_with_no_painted_faces := 36
  let total_ways_to_choose_two_cubes := Nat.choose total_unit_cubes 2
  let successful_outcomes := cubes_with_two_painted_faces * cubes_with_no_painted_faces
  successful_outcomes / total_ways_to_choose_two_cubes

theorem probability_correct : probability_two_faces_and_none_face = 1 / 14 := by
  sorry

end probability_correct_l424_424548


namespace find_b2017_l424_424243

noncomputable def a (n : ℕ) := (1 : ℝ) / (n * (n + 1))

noncomputable def b : ℕ → ℝ
| 1 => 0
| (n + 1) => b n + a n

theorem find_b2017 (h₀ : b 1 = 0) (ha : ∀ n, a n = 1 / n - 1 / (n + 1)) (hb : ∀ n ≥ 2, b n = b (n - 1) + a (n - 1)) :
  b 2017 = 2016 / 2017 :=
sorry

end find_b2017_l424_424243


namespace business_transaction_loss_l424_424096

theorem business_transaction_loss (cost_price : ℝ) (final_price : ℝ) (markup_percent : ℝ) (reduction_percent : ℝ) : 
  (final_price = 96) ∧ (markup_percent = 0.2) ∧ (reduction_percent = 0.2) ∧ (cost_price * (1 + markup_percent) * (1 - reduction_percent) = final_price) → 
  (cost_price - final_price = -4) :=
by
sorry

end business_transaction_loss_l424_424096


namespace price_of_other_stock_l424_424779

theorem price_of_other_stock (total_shares : ℕ) (total_spent : ℝ) (share_1_quantity : ℕ) (share_1_price : ℝ) :
  total_shares = 450 ∧ total_spent = 1950 ∧ share_1_quantity = 400 ∧ share_1_price = 3 →
  (750 / 50 = 15) :=
by sorry

end price_of_other_stock_l424_424779


namespace max_permutation_sum_l424_424523

theorem max_permutation_sum (a : Fin 2011 → Fin 2012)
  (h : bijective a) :
  (∑ i in (Finset.range 2010 : Finset ℕ), (| (a ⟨i, Nat.lt_succ_self i⟩ : ℕ) - (a ⟨i+1, _⟩ : ℕ) |)) ≤
  (∑ i in (Finset.range 2010 : Finset ℕ), (| (a' ⟨i, Nat.lt_succ_self i⟩ : ℕ) - (a' ⟨i+1, _⟩ : ℕ) |))
  :=
sorry
where a' : Fin 2011 → Fin 2012 := λ i, ⟨i.1 + 2, by { norm_num [i.is_lt] } ⟩

end max_permutation_sum_l424_424523


namespace ninth_grade_total_score_l424_424095

-- Definition of the problem setup.
def num_events : ℕ := 3
def points(first second third : ℕ) : ℕ := first * 5 + second * 3 + third * 1
def seventh_grade_students : ℕ
def eighth_grade_students : ℕ := 2 * seventh_grade_students
def seventh_grade_score(seventh_first seventh_second seventh_third : ℕ) : ℕ := 
  points seventh_first seventh_second seventh_third
def eighth_grade_score(eighth_first eighth_second eighth_third : ℕ) : ℕ := 
  points eighth_first eighth_second eighth_third

-- Condition: Total scores of the seventh and eighth grades are equal.
axiom score_equality 
  (seventh_first eighth_first : ℕ) 
  (seventh_second eighth_second : ℕ)
  (seventh_third eighth_third : ℕ) :
  seventh_grade_score seventh_first seventh_second seventh_third =
  eighth_grade_score eighth_first eighth_second eighth_third

-- Condition: Number of students from the eighth grade who placed in the top three is twice that of the seventh grade.
axiom student_condition : seventh_grade_students * 1 = eighth_grade_students * 2

-- The final proof statement we need to show.
theorem ninth_grade_total_score (ninth_grade_score : ℕ) : ninth_grade_score = 7 :=
sorry

end ninth_grade_total_score_l424_424095


namespace count_pairs_satisfying_condition_l424_424295

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424295


namespace resulting_polygon_sides_l424_424608

theorem resulting_polygon_sides 
    (triangle_sides : ℕ := 3) 
    (square_sides : ℕ := 4) 
    (pentagon_sides : ℕ := 5) 
    (heptagon_sides : ℕ := 7) 
    (hexagon_sides : ℕ := 6) 
    (octagon_sides : ℕ := 8) 
    (shared_sides : ℕ := 1) :
    (2 * shared_sides + 4 * (shared_sides + 1)) = 16 := by 
  sorry

end resulting_polygon_sides_l424_424608


namespace percentage_error_in_calculated_area_l424_424466

theorem percentage_error_in_calculated_area :
  let initial_length_error := 0.03 -- 3%
  let initial_width_error := -0.02 -- 2% deficit
  let temperature_change := 15 -- °C
  let humidity_increase := 20 -- %
  let length_error_temp_increase := (temperature_change / 5) * 0.01
  let width_error_humidity_increase := (humidity_increase / 10) * 0.005
  let total_length_error := initial_length_error + length_error_temp_increase
  let total_width_error := initial_width_error + width_error_humidity_increase
  let total_percentage_error := total_length_error + total_width_error
  total_percentage_error * 100 = 3 -- 3%
:= by
  sorry

end percentage_error_in_calculated_area_l424_424466


namespace hexagon_perimeter_arithmetic_sequence_l424_424851

theorem hexagon_perimeter_arithmetic_sequence :
  let a₁ := 10
  let a₂ := 12
  let a₃ := 14
  let a₄ := 16
  let a₅ := 18
  let a₆ := 20
  let lengths := [a₁, a₂, a₃, a₄, a₅, a₆]
  let perimeter := lengths.sum
  perimeter = 90 :=
by
  sorry

end hexagon_perimeter_arithmetic_sequence_l424_424851


namespace num_ordered_triples_l424_424562

theorem num_ordered_triples 
  (a b c : ℕ)
  (h_cond1 : 1 ≤ a ∧ a ≤ b ∧ b ≤ c)
  (h_cond2 : a * b * c = 4 * (a * b + b * c + c * a)) : 
  ∃ (n : ℕ), n = 5 :=
sorry

end num_ordered_triples_l424_424562


namespace wendy_total_sales_correct_l424_424886

noncomputable def wendy_total_sales : ℝ :=
  let morning_apples := 40 * 1.50
  let morning_oranges := 30 * 1
  let morning_bananas := 10 * 0.75
  let afternoon_apples := 50 * 1.35
  let afternoon_oranges := 40 * 0.90
  let afternoon_bananas := 20 * 0.675
  let unsold_bananas := 20 * 0.375
  let unsold_oranges := 10 * 0.50
  let total_morning := morning_apples + morning_oranges + morning_bananas
  let total_afternoon := afternoon_apples + afternoon_oranges + afternoon_bananas
  let total_day_sales := total_morning + total_afternoon
  let total_unsold_sales := unsold_bananas + unsold_oranges
  total_day_sales + total_unsold_sales

theorem wendy_total_sales_correct :
  wendy_total_sales = 227 := by
  unfold wendy_total_sales
  sorry

end wendy_total_sales_correct_l424_424886


namespace min_distance_mn_l424_424803

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_mn : ∃ m > 0, ∀ x > 0, |f x - g x| = 1/2 + 1/2 * Real.log 2 :=
by
  sorry

end min_distance_mn_l424_424803


namespace train_travel_distance_l424_424109

-- Define the given conditions
def travel_time_minutes : ℝ := 4 + 30/60 -- 4 minutes and 30 seconds converted to minutes
def travel_distance_miles : ℝ := 3 -- The train travels 3 miles
def duration_hours : ℝ := 2 -- The time duration we are interested in, which is 2 hours
def minutes_per_hour : ℝ := 60 -- Number of minutes in an hour

-- Convert travel_time_minutes to a noncomputable definition if necessary
noncomputable def rate_miles_per_minute : ℝ := travel_distance_miles / travel_time_minutes
noncomputable def total_time_minutes : ℝ := duration_hours * minutes_per_hour

-- Statement of the theorem to be proved
theorem train_travel_distance : 
  (rate_miles_per_minute * total_time_minutes) = 80 :=
by
  sorry -- Proof to be filled in

end train_travel_distance_l424_424109


namespace ratio_u_v_l424_424870

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l424_424870


namespace max_distance_convoy_l424_424083

structure Vehicle :=
  (mpg : ℝ) (min_gallons : ℝ)

def SUV : Vehicle := ⟨12.2, 10⟩
def Sedan : Vehicle := ⟨52, 5⟩
def Motorcycle : Vehicle := ⟨70, 2⟩

def total_gallons : ℝ := 21

def total_distance (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) : ℝ :=
  SUV.mpg * SUV_gallons + Sedan.mpg * Sedan_gallons + Motorcycle.mpg * Motorcycle_gallons

theorem max_distance_convoy (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) :
  SUV_gallons + Sedan_gallons + Motorcycle_gallons = total_gallons →
  SUV_gallons >= SUV.min_gallons →
  Sedan_gallons >= Sedan.min_gallons →
  Motorcycle_gallons >= Motorcycle.min_gallons →
  total_distance SUV_gallons Sedan_gallons Motorcycle_gallons = 802 :=
sorry

end max_distance_convoy_l424_424083


namespace primes_in_factorial_range_l424_424635

theorem primes_in_factorial_range (n : ℕ) (h : n > 2) :
  (if n > 3 then (finset.filter nat.prime (finset.Ico ((nat.factorial (n - 1)) + 2) ((nat.factorial (n - 1)) + n))).card = 0
   else (finset.filter nat.prime (finset.Ico ((nat.factorial (n - 1)) + 2) ((nat.factorial (n - 1)) + n))).card = 1) := sorry

end primes_in_factorial_range_l424_424635


namespace factorial_calculation_l424_424596

theorem factorial_calculation : 7! - (6 * 6! + 6!) = 0 :=
by
  sorry

end factorial_calculation_l424_424596


namespace find_c_add_5_l424_424574

noncomputable def A := (-2, 3) : ℝ × ℝ
noncomputable def B := (-6, -8) : ℝ × ℝ
noncomputable def C := (4, -1) : ℝ × ℝ

noncomputable def line_eq := ∀ (x y : ℝ), 5*x + 4*y + c = 0

theorem find_c_add_5 (c : ℝ) (h : line_eq (-6) (-8)) :
  c + 5 = -155 / 7 := sorry


end find_c_add_5_l424_424574


namespace negation_equivalent_l424_424842

theorem negation_equivalent :
  (¬ ∀ x : ℝ, x > 1 → (1 / 2) ^ x < 1 / 2) ↔ (∃ x₀ : ℝ, x₀ > 1 ∧ (1 / 2) ^ x₀ ≥ 1 / 2) :=
by
  sorry

end negation_equivalent_l424_424842


namespace ratio_u_v_l424_424868

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l424_424868


namespace volume_of_revolution_l424_424759

noncomputable def region_S := { p : ℝ × ℝ | abs(6 - p.1) + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 20 }
def revolve_line (p : ℝ × ℝ) : ℝ := p.2 - (p.1 / 4 + 5)
def solid_volume : ℝ := 48 * real.pi

theorem volume_of_revolution :
  let S := region_S in
  is_triangle S →
  let line := λ p : ℝ × ℝ, revolve_line p in
  volume_of_solid_of_revolution S line = solid_volume :=
by sorry

end volume_of_revolution_l424_424759


namespace probability_three_draws_one_white_l424_424476

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_white_balls + num_black_balls

def probability_one_white_three_draws : ℚ := 
  (num_white_balls / total_balls) * 
  ((num_black_balls - 1) / (total_balls - 1)) * 
  ((num_black_balls - 2) / (total_balls - 2)) * 3

theorem probability_three_draws_one_white :
  probability_one_white_three_draws = 12 / 35 := by sorry

end probability_three_draws_one_white_l424_424476


namespace max_modulus_z_i_l424_424683

open Complex

theorem max_modulus_z_i (z : ℂ) (hz : abs z = 2) : ∃ z₂ : ℂ, abs z₂ = 2 ∧ abs (z₂ - I) = 3 :=
sorry

end max_modulus_z_i_l424_424683


namespace standard_circle_eq_proof_l424_424009

-- Define the problem conditions
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (5, -6)

-- Define the midpoint of A and B
def C : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Calculate the radius squared
def r_squared : ℝ :=
  (A.fst - C.fst)^2 + (A.snd - C.snd)^2

-- Define the standard equation of the circle
def standard_circle_eq (x y : ℝ) : ℝ :=
  (x - C.fst)^2 + (y - C.snd)^2

-- The theorem to prove the standard equation form
theorem standard_circle_eq_proof :
  standard_circle_eq 2 (-2) = 25 :=
by
  -- placeholder for the proof
  sorry

end standard_circle_eq_proof_l424_424009


namespace number_of_increasing_digits_l424_424162

theorem number_of_increasing_digits : 
  (∑ k in finset.range 10, if 2 ≤ k then nat.choose 9 k else 0) = 502 :=
by
  sorry

end number_of_increasing_digits_l424_424162


namespace range_of_x_l424_424996

noncomputable def integer_part (x : ℝ) : ℤ := ⌊x⌋

theorem range_of_x (x : ℝ) (h : integer_part ((1 - 3*x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by sorry

end range_of_x_l424_424996


namespace f_monotonic_increasing_a_neg1_f_min_value_l424_424236

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x + a / x)

-- Monotonicity part for a = -1 on (1, ∞)
theorem f_monotonic_increasing_a_neg1 (x : ℝ) (h : 1 < x) :
  monotone_on (λ x, abs (x - 1 / x)) (Ioi 1) := by
  sorry

-- Minimum value part based on different values of a
theorem f_min_value (a : ℝ) (x : ℝ) (hx : x > 0) :
  (a < 0 → (f x a = 0 ↔ x = real.sqrt (-a))) ∧
  (a = 0 → ¬ (∃ y, ∀ x, y ≤ f x a)) ∧
  (a > 0 → (f x a = 2 * real.sqrt a ↔ x = real.sqrt a)) := by
  sorry

end f_monotonic_increasing_a_neg1_f_min_value_l424_424236


namespace minimum_value_of_expression_l424_424720

theorem minimum_value_of_expression (x y : ℝ) (h : (x - 1) * 4 + 2 * y = 0) :
  ∃ (x y : ℝ), 9^x + 3^y = 6 :=
by
  -- Lean statement just follows the condition to prove the given answer
  sorry

end minimum_value_of_expression_l424_424720


namespace distinct_positive_solutions_conditions_l424_424426

-- Definitions for the given conditions
variables {a b x y z : ℝ}

noncomputable def system_of_equations (x y z a b : ℝ) : Prop :=
  x + y + z = a ∧
  x^2 + y^2 + z^2 = b^2 ∧
  x * y = z^2

-- Statement of the Lean 4 proof problem
theorem distinct_positive_solutions_conditions (h : system_of_equations x y z a b) :
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  x > 0 ∧ y > 0 ∧ z > 0 →
  a > 0 ∧ b > 0 :=
begin
  sorry
end

end distinct_positive_solutions_conditions_l424_424426


namespace quadratic_roots_positive_difference_has_form_l424_424847

noncomputable def positive_diff_between_roots (a b c : ℝ) : ℝ :=
  let discriminant := b ^ 2 - 4 * a * c
  have ha : a ≠ 0 := by linarith
  have hdiscriminant : discriminant ≥ 0 := by linarith
  let sqrt_discriminant := real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  real.abs (root1 - root2)

theorem quadratic_roots_positive_difference_has_form (p q : ℕ) 
  (hp : p = 304) (hq : q = 5) : positive_diff_between_roots 5 (-2) (-15) = real.sqrt p / q :=
by
  rw [hp, hq]
  simp [positive_diff_between_roots]
  sorry

end quadratic_roots_positive_difference_has_form_l424_424847


namespace last_digit_quaternary_389_l424_424609

theorem last_digit_quaternary_389 : ∃ d : ℕ, nat.digits 4 389 = [d] ++ _ ∧ d = 1 :=
by
  sorry

end last_digit_quaternary_389_l424_424609


namespace angle_measure_l424_424730

variable {α : Type} [LinearOrderedField α] [Real α] [Field α]

namespace Geometry

-- Define the structure of the point and triangle
structure Point (α : Type) := (x y : α)
structure Triangle (α : Type) :=
(A B C : Point α)
(variable pAB : α)
(variable pBC : α)
(variable medianAD : Point α)
(variable bisectorCE : Point α)

-- Define the isosceles triangle and perpendicular relationship
variables (A B C AD CE : Point α)
variables (ABC : Triangle α)
include A B C AD CE ABC

-- define the properties in terms of Lean structures
def is_isosceles (ABC : Triangle α) : Prop :=
  (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 = (ABC.B.x - ABC.C.x)^2 + (ABC.B.y - ABC.C.y)^2

def is_perpendicular (AD CE : Point α) : Prop :=
  (AD.x * CE.x + AD.y * CE.y) = 0

-- the main theorem statement
theorem angle_measure :
  is_isosceles ABC ∧ is_perpendicular AD CE → 
  α = 180 - Real.arccos (Real.sqrt 6 / 4) :=
sorry

end Geometry

end angle_measure_l424_424730


namespace moving_point_on_x_axis_Q_l424_424211

open Real EuclideanGeometry

def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1
def AB_length_eq : ℝ := (4 * √2) / 3
def MQ_length := 3
def fixed_point : (ℝ × ℝ) := (0, 3/2)

theorem moving_point_on_x_axis_Q 
  (x : ℝ)
  (circle_eq : ∀ {x y}, circle_eq x y) 
  (AB_length : ∀ A B, |((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ^ (1/2)| = AB_length_eq ) :
  (|√((MQ_length)^2) = MQ_length) ∧ 
  ((Q : Type)(|2 * Q.1 + √5 * Q.2 - 2 * √5 = 0) 
  ∨ (|2 * Q.1 - √5 * Q.2 + 2 * √5 = 0)) ∧
  (line_AB : Type) (∀ (Q A B : Type), ((∀x y, on_line AB x y) →
  ((0, 3/2) ∈ line_AB))) :=
  sorry

end moving_point_on_x_axis_Q_l424_424211


namespace coordinates_of_C_l424_424025

noncomputable def triangle_ABC_reflection (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let C' := (-(C.1), C.2)  -- Reflect over the y-axis
  let C'' := (C'.1, -(C'.2))  -- Reflect over the x-axis
  C''

theorem coordinates_of_C'' :
  triangle_ABC_reflection (7, 3) (3, 6) (3, 1) = (-3, -1) :=
by
  simp [triangle_ABC_reflection]
  rfl

end coordinates_of_C_l424_424025


namespace triangle_area_45_45_90_l424_424437

theorem triangle_area_45_45_90 (BC : ℝ) (BD : ℝ) (h₁ : ∠ B = 90) (h₂ : ∠ A = 45) (h₃ : ∠ C = 45) (h₄ : BD = 5) :
  let AC := BC * real.sqrt 2 in
  (1 / 2) * (BC * BC) = 25 := by
  sorry

end triangle_area_45_45_90_l424_424437


namespace total_sharks_l424_424992

-- Define the number of sharks at each beach.
def N : ℕ := 22
def D : ℕ := 4 * N
def H : ℕ := D / 2

-- Proof that the total number of sharks on the three beaches is 154.
theorem total_sharks : N + D + H = 154 := by
  sorry

end total_sharks_l424_424992


namespace sum_15_pretty_less_2024_div_15_l424_424143

def is_15_pretty (n : ℕ) : Prop :=
  nat_num_divisors n = 15 ∧ n % 15 = 0

theorem sum_15_pretty_less_2024_div_15 :
  ∑ n in finset.filter (λ x, is_15_pretty x) (finset.Ico 1 2024) / 15 = 9.6 :=
sorry

end sum_15_pretty_less_2024_div_15_l424_424143


namespace count9s_in_range_1_to_100_l424_424103

def count9s (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc x => acc + x.digits.count (fun d => d = 9)) 0

theorem count9s_in_range_1_to_100 : count9s 100 = 20 :=
by
  -- This is a placeholder for actual proof which should be filled in
  sorry

end count9s_in_range_1_to_100_l424_424103


namespace rope_length_in_cm_l424_424962

-- Define the given conditions
def num_equal_pieces : ℕ := 150
def length_equal_piece_mm : ℕ := 75
def num_remaining_pieces : ℕ := 4
def length_remaining_piece_mm : ℕ := 100

-- Prove that the total length of the rope in centimeters is 1165
theorem rope_length_in_cm : (num_equal_pieces * length_equal_piece_mm + num_remaining_pieces * length_remaining_piece_mm) / 10 = 1165 :=
by
  sorry

end rope_length_in_cm_l424_424962


namespace ratio_of_shaded_to_white_area_l424_424893

/-- Given a figure where the vertices of all the squares, except for the largest one, 
    are located at the midpoints of the corresponding sides, prove that the ratio 
    of the area of the shaded part to the white part is 5:3. -/
theorem ratio_of_shaded_to_white_area :
  (let total_shaded := 20 in
  let total_white := 12 in
  total_shaded / total_white) = (5 / 3) :=
by {
  sorry
}

end ratio_of_shaded_to_white_area_l424_424893


namespace union_of_A_and_B_l424_424933

open Set

-- Define the sets A and B based on given conditions
def A (x : ℤ) : Set ℤ := {y | y = x^2 ∨ y = 2 * x - 1 ∨ y = -4}
def B (x : ℤ) : Set ℤ := {y | y = x - 5 ∨ y = 1 - x ∨ y = 9}

-- Specific condition given in the problem
def A_intersect_B_condition (x : ℤ) : Prop :=
  A x ∩ B x = {9}

-- Prove problem statement that describes the union of A and B
theorem union_of_A_and_B (x : ℤ) (h : A_intersect_B_condition x) : A x ∪ B x = {-8, -7, -4, 4, 9} :=
sorry

end union_of_A_and_B_l424_424933


namespace milk_added_is_10_l424_424722

variable (x y : ℚ) -- x and y are rational numbers representing initial amounts and added milk respectively.

-- Conditions
def initial_ratio_milk_water (x : ℚ) := (4 * x, 3 * x)
def new_ratio_milk_water (x y : ℚ) := (4 * x + y, 3 * x)
def capacity_of_can : ℚ := 30
def new_ratio : ℚ := 5 / 2

-- Proof Statement
def proof_milk_added (x y : ℚ) : Prop :=
  (initial_ratio_milk_water x).1 + (initial_ratio_milk_water x).2 = capacity_of_can ∧
  new_ratio_milk_water x y = new_ratio ∧
  x = 20 / 7 ∧
  y = 10

theorem milk_added_is_10 : ∃ (x y : ℚ), proof_milk_added x y := by
  sorry

end milk_added_is_10_l424_424722


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424280

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424280


namespace polynomial_remainder_l424_424385

theorem polynomial_remainder (q : ℝ[X])
    (h1 : q.eval 1 = 10)
    (h2 : q.eval (-3) = -8) :
  q % ((X - 1) * (X + 3)) = 4.5 * X + 5.5 :=
by
  sorry

end polynomial_remainder_l424_424385


namespace dogwood_trees_after_5_years_l424_424459

theorem dogwood_trees_after_5_years :
  let current_trees := 39
  let trees_planted_today := 41
  let growth_rate_today := 2 -- trees per year
  let trees_planted_tomorrow := 20
  let growth_rate_tomorrow := 4 -- trees per year
  let years := 5
  let total_planted_trees := trees_planted_today + trees_planted_tomorrow
  let total_initial_trees := current_trees + total_planted_trees
  let total_growth_today := growth_rate_today * years
  let total_growth_tomorrow := growth_rate_tomorrow * years
  let total_growth := total_growth_today + total_growth_tomorrow
  let final_tree_count := total_initial_trees + total_growth
  final_tree_count = 130 := by
  sorry

end dogwood_trees_after_5_years_l424_424459


namespace Egor_possible_numbers_l424_424149

noncomputable def countDistinctNumbers : ℕ := 114240

theorem Egor_possible_numbers : ∀ (letters : Finset Char),
  (∀ (a b : Char) (ha : a ∈ letters) (hb : b ∈ letters), a ≠ b → digit a ≠ digit b) ∧
  (∀ (a b : Char), a = b ↔ digit a = digit b) ∧
  (letters.sum (λ c => digit.to_nat (digit c))) % 5 = 0 →
  (countDistinctNumbers = 114240) := by
  sorry

end Egor_possible_numbers_l424_424149


namespace exists_n_such_that_n_exp_n_div_2_eq_10_l424_424059

theorem exists_n_such_that_n_exp_n_div_2_eq_10 : ∃ n : ℝ, n^ (n / 2) = 10 ∧ n ≈ 4.5287 :=
sorry

end exists_n_such_that_n_exp_n_div_2_eq_10_l424_424059


namespace base12_mod_9_remainder_l424_424511

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 7 * 12^2 + 3 * 12^1 + 2 * 12^0

theorem base12_mod_9_remainder : (base12_to_base10 1732) % 9 = 2 := by
  sorry

end base12_mod_9_remainder_l424_424511


namespace joan_change_received_l424_424356

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end joan_change_received_l424_424356


namespace bar_chart_shows_quantity_l424_424887

-- Definitions based on the given conditions
def barChartCharacteristics := "a bar chart is a statistical chart."

-- Statement to prove the main assertion
theorem bar_chart_shows_quantity (h : barChartCharacteristics) : "it can easily show the quantity" :=
sorry

end bar_chart_shows_quantity_l424_424887


namespace probability_A_wins_series_l424_424764

variable (p q : ℝ)
variable (hp_pos : 0 < p)
variable (hq_pos : 0 < q)
variable (hpq_lt_one : p + q < 1)

theorem probability_A_wins_series :
    (p^2) / (p^2 + q^2) = probability_of_A_winning_series p q :=
sorry

end probability_A_wins_series_l424_424764


namespace count_strictly_increasing_digits_l424_424158

theorem count_strictly_increasing_digits : 
  (∑ k in Finset.range 9, Nat.choose 9 k.succ) = 502 :=
by
  sorry

end count_strictly_increasing_digits_l424_424158


namespace point_in_second_quadrant_l424_424461

-- Definitions based on the conditions
def complex_to_point (z : ℂ) : ℝ × ℝ :=
  (z.re, z.im)

def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- The theorem to prove
theorem point_in_second_quadrant :
  is_in_second_quadrant (complex_to_point (-1 + 2 * Complex.i)) :=
by
  sorry

end point_in_second_quadrant_l424_424461


namespace maximize_profit_l424_424949

noncomputable def fixed_cost := 14000
noncomputable def variable_cost := 210
noncomputable def daily_sales_volume (x : ℕ) : ℝ :=
  if x ≤ 400 then (1 / 625) * (x : ℝ)^2 else 256
noncomputable def selling_price (x : ℕ) : ℝ :=
  if x ≤ 400 then (-5 / 8) * (x : ℝ) + 750 else 500
noncomputable def total_cost (x : ℕ) : ℝ :=
  fixed_cost + variable_cost * (x : ℝ)
noncomputable def daily_sales_profit (x : ℕ) : ℝ :=
  daily_sales_volume x * selling_price x - total_cost x

theorem maximize_profit :
  let x := 400 in
  daily_sales_profit x = 30000 := by
  sorry

end maximize_profit_l424_424949


namespace find_dot_product_l424_424221

variables (a b : EuclideanSpace ℝ (Fin 3))

noncomputable def condition_1 : Prop := ‖a‖ = 4
noncomputable def condition_2 : Prop := ‖b‖ = 3
noncomputable def condition_3 : Prop := (2 • a - 3 • b) ⬝ (2 • a + b) = 61

theorem find_dot_product (h1 : condition_1 a) (h2 : condition_2 b) (h3 : condition_3 a b) : 
  a ⬝ b = -3 / 2 :=
by
  sorry

end find_dot_product_l424_424221


namespace sum_floor_alpha_geq_l424_424064

open Real

theorem sum_floor_alpha_geq (n : ℕ) (x y : ℕ → ℝ) (α : ℝ)
  (hx : ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → x i ≤ x j)
  (hy : ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → y i ≥ y j)
  (hsum : ∑ i in finset.range n, i * x (i + 1) = ∑ i in finset.range n, i * y (i + 1)) :
  (∑ i in finset.range n, x (i + 1) * ⌊(i + 1) * α⌋) ≥ (∑ i in finset.range n, y (i + 1) * ⌊(i + 1) * α⌋) :=
sorry

end sum_floor_alpha_geq_l424_424064


namespace percentage_of_men_l424_424725

theorem percentage_of_men (M W : ℝ) (h1 : M + W = 1) (h2 : 0.60 * M + 0.2364 * W = 0.40) : M = 0.45 :=
by
  sorry

end percentage_of_men_l424_424725


namespace total_candles_l424_424642

theorem total_candles (num_big_boxes : ℕ) (small_boxes_per_big_box : ℕ) (candles_per_small_box : ℕ) :
  num_big_boxes = 50 ∧ small_boxes_per_big_box = 4 ∧ candles_per_small_box = 40 → 
  (num_big_boxes * (small_boxes_per_big_box * candles_per_small_box) = 8000) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  norm_num
  sorry

end total_candles_l424_424642


namespace sin_alpha_terminal_point_l424_424719

theorem sin_alpha_terminal_point :
  let alpha := (2 * Real.cos (120 * (π / 180)), Real.sqrt 2 * Real.sin (225 * (π / 180)))
  α = -π / 4 →
  α.sin = - Real.sqrt 2 / 2
:=
by
  intro α_definition
  sorry

end sin_alpha_terminal_point_l424_424719


namespace repeating_decimal_sum_l424_424903

theorem repeating_decimal_sum :
  let x : ℚ := 45 / 99 in
  let simp_fraction := (5, 11) in
  simp_fraction.fst + simp_fraction.snd = 16 :=
by
  let x : ℚ := 45 / 99
  let simp_fraction := (5, 11)
  have h_gcd : Int.gcd 45 99 = 9 := by norm_num
  have h_simplify : x = simp_fraction.fst / simp_fraction.snd := by
    rw [num_denom, h_gcd]
    norm_cast
    simp
  show simp_fraction.fst + simp_fraction.snd = 16 from
    by norm_num
  simp_fraction.rfl

end repeating_decimal_sum_l424_424903


namespace same_exponent_for_all_bases_l424_424504

theorem same_exponent_for_all_bases {a : Type} [LinearOrderedField a] {C : a} (ha : ∀ (a : a), a ≠ 0 → a^0 = C) : C = 1 :=
by
  sorry

end same_exponent_for_all_bases_l424_424504


namespace total_time_in_pool_is_29_minutes_l424_424975

noncomputable def calculate_total_time_in_pool : ℝ :=
  let jerry := 3             -- Jerry's time in minutes
  let elaine := 2 * jerry    -- Elaine's time in minutes
  let george := elaine / 3    -- George's time in minutes
  let susan := 150 / 60      -- Susan's time in minutes
  let puddy := elaine / 2    -- Puddy's time in minutes
  let frank := elaine / 2    -- Frank's time in minutes
  let estelle := 0.1 * 60    -- Estelle's time in minutes
  let total_excluding_newman := jerry + elaine + george + susan + puddy + frank + estelle
  let newman := total_excluding_newman / 7   -- Newman's average time
  total_excluding_newman + newman

theorem total_time_in_pool_is_29_minutes : 
  calculate_total_time_in_pool = 29 :=
by
  sorry

end total_time_in_pool_is_29_minutes_l424_424975


namespace first_player_avoids_losing_l424_424929

-- Define a type for 2D vectors
structure Vec2D :=
  (x : ℝ)
  (y : ℝ)

-- There are 1992 vectors in the plane
constant vectors : Fin 1992 → Vec2D

-- Sum of a set of vectors
def sum_vectors (vecs : List Vec2D) : Vec2D :=
  vecs.foldr (λ v acc, ⟨acc.x + v.x, acc.y + v.y⟩) ⟨0, 0⟩

-- Game condition: Two players pick unpicked vectors alternately
-- and the winner is the one whose vectors sum to a vector with
-- larger magnitude (or they draw if magnitudes are the same).
/-
The conjecture that needs to be proved:
There exists a strategy for the first player such that the magnitude
of the sum of the vectors picked by the first player is at least as large 
as the magnitude of the sum of the vectors picked by the second player.
-/
theorem first_player_avoids_losing :
  ∃ strategy : (List Vec2D → List Vec2D → List Vec2D),
  ∀ vectors : List Vec2D,
  let first_player_vectors := strategy vectors [] [],
      second_player_vectors := vectors.filter (λ v, ¬ v ∈ first_player_vectors),
      sum1 := sum_vectors first_player_vectors,
      sum2 := sum_vectors second_player_vectors in
  (real.sqrt (sum1.x^2 + sum1.y^2)) ≥ (real.sqrt (sum2.x^2 + sum2.y^2)) :=
sorry

end first_player_avoids_losing_l424_424929


namespace volume_of_pyramid_l424_424097

noncomputable def volume_pyramid : ℝ :=
  let a := 9
  let b := 12
  let s := 15
  let base_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let half_diagonal := diagonal / 2
  let height := Real.sqrt (s^2 - half_diagonal^2)
  (1 / 3) * base_area * height

theorem volume_of_pyramid :
  volume_pyramid = 36 * Real.sqrt 168.75 := by
  sorry

end volume_of_pyramid_l424_424097


namespace ratio_of_x_intercepts_l424_424881

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l424_424881


namespace cos_cofunction_identity_l424_424217

theorem cos_cofunction_identity (α : ℝ) (h : Real.sin (30 * Real.pi / 180 + α) = Real.sqrt 3 / 2) :
  Real.cos (60 * Real.pi / 180 - α) = Real.sqrt 3 / 2 := by
  sorry

end cos_cofunction_identity_l424_424217


namespace carlos_class_size_l424_424600

-- Define the conditions and the question in Lean
  
theorem carlos_class_size (h1 : ∀ n : ℕ, Carlos_is_the_n_best_student n = 75) 
                           (h2 : ∀ n : ℕ, Carlos_is_the_n_worst_student n = 75) 
                           : ∃ n : ℕ, n = 149 :=
sorry

end carlos_class_size_l424_424600


namespace find_f_find_extrema_on_interval_l424_424219

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f(x) = a*x^2 + b*x + c

def condition1 (f : ℝ → ℝ) : Prop :=
  f 0 = 1

def condition2 (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ (a*(-1)^2 + b*(-1) + c - (-1) - 4 = 0) ∧ (a*(3)^2 + b*(3) + c - (3) - 4 = 0)

theorem find_f :
  ∃ f : ℝ → ℝ, is_quadratic f ∧ condition1 f ∧ condition2 f ∧ f(x) = x^2 - x + 1 := sorry

def g (x : ℝ) : ℝ :=
  (x^2 - x + 1) - 3*x - 6

def domain_interval (x : ℝ) : Prop :=
  1/9 ≤ x ∧ x ≤ 27

theorem find_extrema_on_interval :
  let y := g(log 3 x) in
  ∀ x : ℝ, domain_interval x →
  (∀ x y, domain_interval x → y = g(log 3 x) → y ≥ -9 → y ≤ 7) :=
sorry

end find_f_find_extrema_on_interval_l424_424219


namespace largest_sum_of_distinct_factors_l424_424327

theorem largest_sum_of_distinct_factors (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_product : A * B * C = 3003) :
  A + B + C ≤ 105 :=
sorry  -- Proof is not required, just the statement.

end largest_sum_of_distinct_factors_l424_424327


namespace count_valid_pairs_l424_424610

theorem count_valid_pairs : 
  ∃! (S : Finset (ℕ × ℕ)), 
    (∀ (p ∈ S), (p.1 > 0) ∧ (p.2 > 0) ∧ 
    (5 / p.1.to_real + 3 / p.2.to_real = 1)) ∧ 
    (S.card = 4) := sorry

end count_valid_pairs_l424_424610


namespace sum_center_radius_eq_neg2_l424_424761

theorem sum_center_radius_eq_neg2 (c d s : ℝ) (h_eq : ∀ x y : ℝ, x^2 + 14 * x + y^2 - 8 * y = -64 ↔ (x + c)^2 + (y + d)^2 = s^2) :
  c + d + s = -2 :=
sorry

end sum_center_radius_eq_neg2_l424_424761


namespace book_page_count_l424_424978

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end book_page_count_l424_424978


namespace cones_to_cylinder_volume_ratio_l424_424726

theorem cones_to_cylinder_volume_ratio :
  let π := Real.pi
  let r_cylinder := 4
  let h_cylinder := 18
  let r_cone := 4
  let h_cone1 := 6
  let h_cone2 := 9
  let V_cylinder := π * r_cylinder^2 * h_cylinder
  let V_cone1 := (1 / 3) * π * r_cone^2 * h_cone1
  let V_cone2 := (1 / 3) * π * r_cone^2 * h_cone2
  let V_totalCones := V_cone1 + V_cone2
  V_totalCones / V_cylinder = 5 / 18 :=
by
  sorry

end cones_to_cylinder_volume_ratio_l424_424726


namespace bailing_rate_sufficient_l424_424051

-- Define the given conditions as hypotheses
def miles_from_shore : ℝ := 3
def water_inflow_rate : ℝ := 15 -- gallons per minute
def maximum_tolerable_water : ℝ := 60
def rowing_speed : ℝ := 6 -- miles per hour
def time_to_shore : ℝ := (miles_from_shore / rowing_speed) * 60 -- converting hours to minutes

-- Define the goal: the minimum bailing rate
def minimum_bailing_rate : ℝ := 13

theorem bailing_rate_sufficient :
  (miles_from_shore = 3) →
  (water_inflow_rate = 15) →
  (maximum_tolerable_water = 60) →
  (rowing_speed = 6) →
  (time_to_shore = 30) →
  minimum_bailing_rate = (390 / time_to_shore) := 
by 
  intros h1 h2 h3 h4 h5
  have h_time_to_shore_30 : time_to_shore = 30, from by norm_num [time_to_shore, h1, h4]; done
  have h_total_water : 450 = water_inflow_rate * time_to_shore, by norm_num [h2, h_time_to_shore_30]; done
  have h_excess_water : 390 = h_total_water - maximum_tolerable_water, by norm_num [h3]; done
  exact (eq.symm h_excess_water)
suffices minimum_bailing_rate = 390 / 30, by norm_cast; linarith
sorry

end bailing_rate_sufficient_l424_424051


namespace number_of_increasing_digits_l424_424166

theorem number_of_increasing_digits : 
  (∑ k in finset.range 10, if 2 ≤ k then nat.choose 9 k else 0) = 502 :=
by
  sorry

end number_of_increasing_digits_l424_424166


namespace coeff_x5_in_expansion_l424_424611

-- Define the binomial coefficient function
def binomial (n k : ℕ) := Nat.choose n k

-- Define the polynomial expansion and the coefficient of the x^5 term
theorem coeff_x5_in_expansion : 
  let f := (x^2 + x - 1) in coeff (expand (f ^ 5)) (5) = 11 :=
by
  sorry

end coeff_x5_in_expansion_l424_424611


namespace value_op_and_add_10_l424_424005

def op_and (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem value_op_and_add_10 : op_and 8 5 + 10 = 49 :=
by
  sorry

end value_op_and_add_10_l424_424005


namespace books_to_sell_to_reach_goal_l424_424435

-- Definitions for conditions
def initial_savings : Nat := 10
def clarinet_cost : Nat := 90
def book_price : Nat := 5
def halfway_goal : Nat := clarinet_cost / 2

-- The primary theorem to prove
theorem books_to_sell_to_reach_goal : 
  initial_savings + (initial_savings = 0 → clarinet_cost) / book_price = 25 :=
by
  -- Proof steps (skipped in the statement)
  sorry

end books_to_sell_to_reach_goal_l424_424435


namespace estimate_probability_l424_424676

theorem estimate_probability :
  let hits := [1, 2, 3, 4]
  let misses := [5, 6, 7, 8, 9, 0]
  let groups := ["907", "966", "191", "925", "271", "932", "812", "458", "569", "683",
                 "431", "257", "393", "027", "556", "488", "730", "113", "537", "989"]
  (groups.filter (fun group =>
    (group.to_list.count (λ x, x ∈ hits) = 2) ∧ 
    (group.to_list.count (λ x, x ∈ misses) = 1)).length) / (groups.length) = 0.25 :=
by
  let hits := [1, 2, 3, 4]
  let misses := [5, 6, 7, 8, 9, 0]
  let groups := ["907", "966", "191", "925", "271", "932", "812", "458", "569", "683",
                 "431", "257", "393", "027", "556", "488", "730", "113", "537", "989"]
  let valid_groups := groups.filter (fun group =>
    (group.to_list.count (λ x, x ∈ hits) = 2) ∧ 
    (group.to_list.count (λ x, x ∈ misses) = 1))
  have h_eq : valid_groups.length = 5 := sorry
  have total_groups : groups.length = 20 := sorry
  calc 
    valid_groups.length / groups.length = 5 / 20 := by rw [h_eq, total_groups]
    ... = 0.25 := by norm_num

end estimate_probability_l424_424676


namespace step1_eq_step2_step2_eq_step3_step3_eq_step4_step4_ne_finalStep_correct_answer_l424_424985

-- Define the expressions for each step
def step1 (a b c : ℝ) : ℝ := (a + b - c) * (a - b - c)
def step2 (a b c : ℝ) : ℝ := (a - c + b) * (a - c - b)
def step3 (a b c : ℝ) : ℝ := ((a - c) + b) * ((a - c) - b)
def step4 (a b c : ℝ) : ℝ := (a - c)^2 - b^2
def finalStep (a b c : ℝ) : ℝ := a^2 - 2 * a * c + c^2 - b^2

-- Prove that the steps are correct
theorem step1_eq_step2 (a b c : ℝ) : step1 a b c = step2 a b c := by
  sorry

theorem step2_eq_step3 (a b c : ℝ) : step2 a b c = step3 a b c := by
  sorry

theorem step3_eq_step4 (a b c : ℝ) : step3 a b c = step4 a b c := by
  sorry

theorem step4_ne_finalStep (a b c : ℝ) : step4 a b c ≠ finalStep a b c := by
  sorry

-- The correct answer is D
theorem correct_answer : "D" := by
  sorry

end step1_eq_step2_step2_eq_step3_step3_eq_step4_step4_ne_finalStep_correct_answer_l424_424985


namespace sum_of_exterior_angles_of_regular_dodecagon_l424_424853

theorem sum_of_exterior_angles_of_regular_dodecagon :
  ∀ (dodecagon : Type) [polygon dodecagon] (h : sides dodecagon = 12), sum_of_exterior_angles dodecagon = 360 :=
sorry

end sum_of_exterior_angles_of_regular_dodecagon_l424_424853


namespace max_squares_removed_to_prevent_domino_l424_424065

-- Define the chessboard and relevant properties
def is_valid_chessboard (board : list (list bool)) : Prop :=
  board.length = 8 ∧ (∀ row ∈ board, row.length = 8)

def is_valid_2x1_domino_placement (board : list (list bool)) (i j : ℕ) : Prop :=
  if i < 7 then board[i][j] ∧ board[i+1][j] = true
  else if j < 7 then board[i][j] ∧ board[i][j+1] = true
  else false

def can_place_domino (board : list (list bool)) : Prop :=
  ∃ i j, is_valid_2x1_domino_placement board i j

-- Stating the maximum number of squares that can be removed
theorem max_squares_removed_to_prevent_domino (board : list (list bool)) :
  is_valid_chessboard board →
  (∃ removed_board : list (list bool), is_valid_chessboard removed_board ∧
     (removed_board = remove_squares board 48) ∧
     ¬ can_place_domino removed_board ∧
     (∀ i j, ∃ returned_board : list (list bool), returned_board = return_square removed_board i j ∧ can_place_domino returned_board)) :=
sorry

end max_squares_removed_to_prevent_domino_l424_424065


namespace problem_statement_l424_424598

theorem problem_statement : 
  ((Real.sqrt 2 - 1)^0 + abs (-3) - (Real.cbrt 27) + (-1)^(2021) = 0) :=
by
  sorry

end problem_statement_l424_424598


namespace length_of_platform_l424_424052

theorem length_of_platform :
  ∀ (L : ℝ), (∀ (s : ℝ), (425 / 40 = s) → ((425 + L) / 55 = s)) → L = 159 :=
by
  assume L h
  have s := 425 / 40
  have h_s : s = 425 / 40 := by rfl
  specialize h s h_s
  rw [h] at h_s
  field_simp at h_s
  linarith

end length_of_platform_l424_424052


namespace probability_of_green_ball_l424_424338

theorem probability_of_green_ball
  (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 10)
  (h2 : red_balls = 3)
  (h3 : blue_balls = 2)
  (h4 : green_balls = total_balls - (red_balls + blue_balls)) :
  (green_balls : ℚ) / total_balls = 1 / 2 :=
sorry

end probability_of_green_ball_l424_424338


namespace triangle_base_length_l424_424319

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 24) (h_height : height = 8) (h_area_formula : area = (base * height) / 2) :
  base = 6 :=
by
  sorry

end triangle_base_length_l424_424319


namespace proof_problem_l424_424569

-- Define a probability space for a standard die rolled eight times
def die_roll := {1, 2, 3, 4, 5, 6}

-- Define the probability that a single roll of a standard die results in an odd number.
def prob_odd_roll : ℚ := 1 / 2

-- Define the probability that a single roll of a standard die results in a number other than 5.
def prob_not_five : ℚ := 5 / 6

-- Define the probability that all eight rolls are odd
def prob_all_odd : ℚ := (1 / 2) ^ 8

-- Define the probability that at least one of the eight rolls is a 5
def prob_at_least_one_five : ℚ := 1 - (5 / 6) ^ 8

-- Calculate the combined probability
def combined_prob : ℚ := prob_all_odd * prob_at_least_one_five

-- Final calculation
def final_probability : ℚ := 1288991 / 429981696

-- The proof problem statement in Lean:
theorem proof_problem : combined_prob = final_probability := sorry

end proof_problem_l424_424569


namespace average_salary_rest_l424_424334

variable (totalWorkers : ℕ)
variable (averageSalaryAll : ℕ)
variable (numTechnicians : ℕ)
variable (averageSalaryTechnicians : ℕ)

theorem average_salary_rest (h1 : totalWorkers = 28) 
                           (h2 : averageSalaryAll = 8000)
                           (h3 : numTechnicians = 7)
                           (h4 : averageSalaryTechnicians = 14000) : 
                           (averageSalaryAll * totalWorkers - averageSalaryTechnicians * numTechnicians) / (totalWorkers - numTechnicians) = 6000 :=
begin
  -- The proof will be provided here
  sorry
end

end average_salary_rest_l424_424334


namespace find_a_monotonicity_intervals_range_of_b_intersection_l424_424671

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  a * Real.log (1 + x) + x ^ 2 - 10 * x

-- Given condition: x = 3 is an extremum point
theorem find_a (a : ℝ) : (deriv (λ x => f x a) 3 = 0) → (a = 16) := by
  sorry

-- Determine intervals of monotonicity
theorem monotonicity_intervals (a : ℝ) (h : a = 16) :
  (∀ x ∈ Ioo (-1 : ℝ) 1, deriv (λ x => f x a) x > 0) ∧
  (∀ x ∈ Ioo (3 : ℝ) (⊤ : ℝ), deriv (λ x => f x a) x > 0) ∧
  (∀ x ∈ Ioo (1 : ℝ) 3, deriv (λ x => f x a) x < 0) := by
  sorry

-- Find range of b such that y = b intersects the graph of f at three points
theorem range_of_b_intersection (b : ℝ) :
  (∃ a : ℝ, a = 16) →
  (3 : ℝ) ∈ Ioo (32 * Real.log 2 - 21) (16 * Real.log 2 - 9) → true := by
  sorry

end find_a_monotonicity_intervals_range_of_b_intersection_l424_424671


namespace total_time_late_l424_424593

theorem total_time_late
  (charlize_late : ℕ)
  (classmate_late : ℕ → ℕ)
  (h1 : charlize_late = 20)
  (h2 : ∀ n, n < 4 → classmate_late n = charlize_late + 10) :
  charlize_late + (∑ i in Finset.range 4, classmate_late i) = 140 := by
  sorry

end total_time_late_l424_424593


namespace min_minutes_for_B_cheaper_l424_424595

-- Define the relevant constants and costs associated with each plan
def cost_A (x : ℕ) : ℕ := 12 * x
def cost_B (x : ℕ) : ℕ := 2500 + 6 * x
def cost_C (x : ℕ) : ℕ := 9 * x

-- Lean statement for the proof problem
theorem min_minutes_for_B_cheaper : ∃ (x : ℕ), x = 834 ∧ cost_B x < cost_A x ∧ cost_B x < cost_C x := 
sorry

end min_minutes_for_B_cheaper_l424_424595


namespace sin_theta_of_triangle_l424_424111

theorem sin_theta_of_triangle (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (h₁ : A = 24) (h₂ : a = 8) (h₃ : m = 7.2) 
  (h₄ : 24 = 1 / 2 * 8 * 7.2 * sin θ) : sin θ = 5 / 6 := 
by 
  rw [h₁, h₂, h₃] at h₄
  norm_num at h₄
  exact h₄

end sin_theta_of_triangle_l424_424111


namespace overlap_area_is_3_l424_424133

structure Point where
  x : ℝ
  y : ℝ

def PointA : Point := ⟨0, 0⟩
def PointB : Point := ⟨2, 2⟩
def PointC : Point := ⟨4, 0⟩
def PointD : Point := ⟨0, 2⟩
def PointE : Point := ⟨2, 0⟩
def PointF : Point := ⟨4, 2⟩

def Triangle1 : set Point := {p | p = PointA ∨ p = PointB ∨ p = PointC}
def Triangle2 : set Point := {p | p = PointD ∨ p = PointE ∨ p = PointF}

noncomputable def overlapArea : ℝ := 3

theorem overlap_area_is_3 :
  triangleOverlapArea Triangle1 Triangle2 = 3 :=
sorry

end overlap_area_is_3_l424_424133


namespace prob_x_lt_y_is_correct_l424_424959

open Set

noncomputable def prob_x_lt_y : ℝ :=
  let rectangle := Icc (0: ℝ) 4 ×ˢ Icc (0: ℝ) 3
  let area_rectangle := 4 * 3
  let triangle := {p : ℝ × ℝ | p.1 ∈ Icc (0: ℝ) 3 ∧ p.2 ∈ Icc (0: ℝ) 3 ∧ p.1 < p.2}
  let area_triangle := 1 / 2 * 3 * 3
  let probability := area_triangle / area_rectangle
  probability

-- To state as a theorem using Lean's notation
theorem prob_x_lt_y_is_correct : prob_x_lt_y = 3 / 8 := sorry

end prob_x_lt_y_is_correct_l424_424959


namespace cube_shortest_trip_length_is_4_l424_424584

noncomputable def cube_shortest_trip_length (edge_length : ℝ) : ℝ :=
  let mid_length := edge_length / 2
  in mid_length + edge_length + mid_length

theorem cube_shortest_trip_length_is_4 :
  cube_shortest_trip_length 2 = 4 := 
by
  rw [cube_shortest_trip_length, div_eq_mul_inv, ← mul_assoc, ← add_assoc]
  norm_num
  sorry

end cube_shortest_trip_length_is_4_l424_424584


namespace imprint_opposite_face_to_O_l424_424066

-- Define the initial conditions
def conditions : Prop :=
  ∃ cube : list (list (list bool)), cube.length = 3 ∧
  ∀ face, face ∈ cube → 
  (face.length = 3 ∧ ∀ row, row ∈ face → row.size = 3) ∧
  (count True ((flatten cube).sum)) = 16

-- The imprint formed by the opposite face to the one showing "O"
def imprint_opposite_face_O : list (list bool) :=
  [[false, true, false],
   [false, false, false],
   [false, true, false]]

-- The problem statement
theorem imprint_opposite_face_to_O :
  conditions → imprint_of_opposite_face = imprint_opposite_face_O :=
by sorry

end imprint_opposite_face_to_O_l424_424066


namespace geometric_series_sum_l424_424127

theorem geometric_series_sum : 
  ∀ (a r l : ℕ), 
    a = 2 ∧ r = 3 ∧ l = 4374 → 
    ∃ n S, 
      a * r ^ (n - 1) = l ∧ 
      S = a * (r^n - 1) / (r - 1) ∧ 
      S = 6560 :=
by 
  intros a r l h
  sorry

end geometric_series_sum_l424_424127


namespace range_of_x_l424_424995

noncomputable def integer_part (x : ℝ) : ℤ := ⌊x⌋

theorem range_of_x (x : ℝ) (h : integer_part ((1 - 3*x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by sorry

end range_of_x_l424_424995


namespace matching_shoe_probability_l424_424540

theorem matching_shoe_probability 
  (total_shoes : ℕ) (pairs : ℕ) (select : ℕ) 
  (h1 : total_shoes = 24) 
  (h2 : pairs = 12) 
  (h3 : select = 2) : 
  (12 / (24.choose 2)) = (1 / 46) :=
by
  sorry

end matching_shoe_probability_l424_424540


namespace donny_paid_l424_424148

variable (total_capacity initial_fuel price_per_liter change : ℕ)

theorem donny_paid (h1 : total_capacity = 150) 
                   (h2 : initial_fuel = 38) 
                   (h3 : price_per_liter = 3) 
                   (h4 : change = 14) : 
                   (total_capacity - initial_fuel) * price_per_liter + change = 350 := 
by
  sorry

end donny_paid_l424_424148


namespace johns_tour_program_days_l424_424921

/-- John has Rs 360 for his expenses. If he exceeds his days by 4 days, he must cut down daily expenses by Rs 3. Prove that the number of days of John's tour program is 20. -/
theorem johns_tour_program_days
    (d e : ℕ)
    (h1 : 360 = e * d)
    (h2 : 360 = (e - 3) * (d + 4)) : 
    d = 20 := 
  sorry

end johns_tour_program_days_l424_424921


namespace total_pages_in_book_l424_424979

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end total_pages_in_book_l424_424979


namespace guitar_center_shipping_fee_is_zero_l424_424419

-- Conditions
def suggested_retail_price : ℝ := 1000
def discount_guitar_center : ℝ := 0.15
def discount_sweetwater : ℝ := 0.10
def savings : ℝ := 50

-- Definitions derived from conditions
def price_after_discount (price discount : ℝ) : ℝ := (1 - discount) * price
def total_cost_sweetwater : ℝ := price_after_discount suggested_retail_price discount_sweetwater

-- Theorem: the shipping fee of Guitar Center
theorem guitar_center_shipping_fee_is_zero : 
  let total_cost_guitar_center := total_cost_sweetwater - savings,
      price_after_discount_guitar_center := price_after_discount suggested_retail_price discount_guitar_center
  in total_cost_guitar_center = price_after_discount_guitar_center → 
     (total_cost_guitar_center - price_after_discount_guitar_center = 0) := 
by {
  sorry
}

end guitar_center_shipping_fee_is_zero_l424_424419


namespace area_ratio_of_traced_path_l424_424491

-- Definitions for the conditions given in the problem
def Point := (ℝ × ℝ)
def Square (A B C D : Point) : Prop :=
  A = (0, 0) ∧
  B = (1, 0) ∧
  C = (1, 1) ∧
  D = (0, 1)

def Midpoint (P Q : Point) : Point := 
  ((P.1 + Q.1) / 2 , (P.2 + Q.2) / 2)

-- The main theorem to state the problem
theorem area_ratio_of_traced_path (A B C D : Point) (R : ℝ) 
  (h1 : Square A B C D)
  (initial_pos_A : Point)
  (initial_mid_AB : Point)
  (path_traces_square : Area (Square traced by midpoint of line segment joining particles starting from A and midpoint of AB))
  (area_ABCD : ℝ = 1) : 
  
  R / area_ABCD = 1 / 4 := 
  sorry

end area_ratio_of_traced_path_l424_424491


namespace correct_factorization_l424_424581

-- Definitions for the given conditions of different options
def condition_A (a : ℝ) : Prop := 2 * a^2 - 2 * a + 1 = 2 * a * (a - 1) + 1
def condition_B (x y : ℝ) : Prop := (x + y) * (x - y) = x^2 - y^2
def condition_C (x y : ℝ) : Prop := x^2 - 4 * x * y + 4 * y^2 = (x - 2 * y)^2
def condition_D (x : ℝ) : Prop := x^2 + 1 = x * (x + 1 / x)

-- The theorem to prove that option C is correct
theorem correct_factorization (x y : ℝ) : condition_C x y :=
by sorry

end correct_factorization_l424_424581


namespace students_chose_water_l424_424974

/-- Let total be the total number of students. -/
variable (total : ℕ) (juice soda water : ℕ)

/-- Conditions given in the problem -/
variable (h_juice : juice = 0.50 * total)
variable (h_soda : soda = 0.30 * total)
variable (h_soda_count : soda = 90)
variable (h_water : water = total - juice - soda)

/-- The claim to prove -/
theorem students_chose_water : water = 60 :=
by
  /- Proof is omitted -/
  sorry

end students_chose_water_l424_424974


namespace find_angle_A_max_perimeter_l424_424206

-- Definitions for the conditions
def is_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π 

-- Problem 1: Magnitude of angle A
theorem find_angle_A (a b c A B C : ℝ) 
  (h_triangle : triangle a b c A B C) 
  (h_perp : is_perpendicular (a, b + c) (Real.cos C + Real.sqrt 3 * Real.sin C, -1)) :
  A = π / 3 := 
sorry

-- Problem 2: Maximum perimeter
theorem max_perimeter (a b c A B C : ℝ) 
  (h_triangle : triangle a b c A B C) 
  (a_eq_sqrt3 : a = Real.sqrt 3)
  (A_eq_pi_over_3 : A = π / 3) :
  b + c = 2 * Real.sqrt 3 → 
  b + c + a ≤ 3 * Real.sqrt 3 := 
sorry

end find_angle_A_max_perimeter_l424_424206


namespace original_rectangle_area_l424_424960

theorem original_rectangle_area : 
  ∃ (a b : ℤ), (a + b = 20) ∧ (a * b = 96) := by
  sorry

end original_rectangle_area_l424_424960


namespace percent_increase_in_maintenance_time_l424_424515

theorem percent_increase_in_maintenance_time (original_time new_time : ℝ) (h1 : original_time = 25) (h2 : new_time = 30) : 
  ((new_time - original_time) / original_time) * 100 = 20 :=
by
  sorry

end percent_increase_in_maintenance_time_l424_424515


namespace maria_dice_problem_l424_424391

theorem maria_dice_problem :
  let fair_die_prob := 1 / 6
  let biased_die_prob := 1 / 2
  let other_prob := 1 / 10
  let prob_three_with_fair := (fair_die_prob ^ 3)
  let prob_three_with_biased := (biased_die_prob ^ 3)
  let total_prob_fair_given_three := prob_three_with_fair / (prob_three_with_fair + prob_three_with_biased)
  let total_prob_biased_given_three := prob_three_with_biased / (prob_three_with_fair + prob_three_with_biased)
  let fourth_three_prob_fair := total_prob_fair_given_three * fair_die_prob
  let fourth_three_prob_biased := total_prob_biased_given_three * biased_die_prob
  let final_prob := fourth_three_prob_fair + fourth_three_prob_biased
  ∃ (p q : ℕ), nat.gcd p q = 1 ∧ p / q = final_prob ∧ p + q = 125 :=
by
  sorry

end maria_dice_problem_l424_424391


namespace geometric_representation_segment_l424_424818

open Complex

def geometric_representation {θ : ℝ} (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : ℂ :=
  Complex.i * Real.cos θ

theorem geometric_representation_segment :
  ∀ {θ : ℝ} (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi),
    ∃ y : ℝ, 
      (y = Real.cos θ) ∧ y ∈ Set.Icc (-1) 1 :=
by
  intro θ h
  use Real.cos θ
  constructor
  . rfl
  . sorry

end geometric_representation_segment_l424_424818


namespace difference_is_correct_l424_424522

-- Define the digits
def digits : List ℕ := [9, 2, 1, 5]

-- Define the largest number that can be formed by these digits
def largestNumber : ℕ :=
  1000 * 9 + 100 * 5 + 10 * 2 + 1 * 1

-- Define the smallest number that can be formed by these digits
def smallestNumber : ℕ :=
  1000 * 1 + 100 * 2 + 10 * 5 + 1 * 9

-- Define the correct difference
def difference : ℕ :=
  largestNumber - smallestNumber

-- Theorem statement
theorem difference_is_correct : difference = 8262 :=
by
  sorry

end difference_is_correct_l424_424522


namespace shipCargoCalculation_l424_424567

def initialCargo : Int := 5973
def cargoLoadedInBahamas : Int := 8723
def totalCargo (initial : Int) (loaded : Int) : Int := initial + loaded

theorem shipCargoCalculation : totalCargo initialCargo cargoLoadedInBahamas = 14696 := by
  sorry

end shipCargoCalculation_l424_424567


namespace shopping_center_expense_l424_424360

theorem shopping_center_expense
    (films_count : ℕ := 9)
    (films_original_price : ℝ := 7)
    (film_discount : ℝ := 2)
    (books_full_price : ℝ := 10)
    (books_count : ℕ := 5)
    (books_discount_rate : ℝ := 0.25)
    (cd_price : ℝ := 4.50)
    (cd_count : ℕ := 6)
    (tax_rate : ℝ := 0.06)
    (total_amount_spent : ℝ := 109.18) :
    let films_total := films_count * (films_original_price - film_discount)
    let remaining_books := books_count - 1
    let discounted_books_total := remaining_books * (books_full_price * (1 - books_discount_rate))
    let books_total := books_full_price + discounted_books_total
    let cds_paid_count := cd_count - (cd_count / 3)
    let cds_total := cds_paid_count * cd_price
    let total_before_tax := films_total + books_total + cds_total
    let tax := total_before_tax * tax_rate
    let total_with_tax := total_before_tax + tax
    total_with_tax = total_amount_spent :=
by
  sorry

end shopping_center_expense_l424_424360


namespace neighboring_cells_diff_l424_424344

theorem neighboring_cells_diff
  (n : ℕ) (h : 2 ≤ n)
  (board : ℕ → ℕ → ℕ)
  (h_entries : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2)
  (h_all_entries : ∀ k, ∃ i j, board i j = k) :
  ∃ i j i' j', (|i - i'| ≤ 1 ∧ |j - j'| ≤ 1 ∧ (i ≠ i' ∨ j ≠ j')) ∧ (|board i j - board i' j'| ≥ n + 1) :=
sorry

end neighboring_cells_diff_l424_424344


namespace distinct_sums_mod_p_l424_424384

theorem distinct_sums_mod_p (p : ℕ) (hp : p.prime) (r : ℕ) (hr : r < p) (a : Fin r → ℕ) (hl : ∀ i, a i < p) :
  ∃ (s : Fin (r + 1) → ℕ), function.injective (λ i, ∑ j in finset.range r, if i.val ≤ j then a ⟨j, hr.trans_le (nat.le_of_lt_succ i.is_lt)⟩ else 0) :=
begin
  sorry
end

end distinct_sums_mod_p_l424_424384


namespace Rhett_rent_expense_l424_424412

-- Define the problem statement using given conditions
theorem Rhett_rent_expense
  (late_payments : ℕ := 2)
  (no_late_fees : Bool := true)
  (fraction_of_salary : ℝ := 3 / 5)
  (monthly_salary : ℝ := 5000)
  (tax_rate : ℝ := 0.1) :
  let salary_after_taxes := monthly_salary * (1 - tax_rate)
  let total_late_rent := fraction_of_salary * salary_after_taxes
  let monthly_rent_expense := total_late_rent / late_payments
  monthly_rent_expense = 1350 := by
  sorry

end Rhett_rent_expense_l424_424412


namespace abs_eq_zero_iff_l424_424047

theorem abs_eq_zero_iff {x : ℝ} : (|4 * x - 2| = 0) ↔ (x = 1 / 2) :=
by
  have h1 : 4 * x - 2 = 0 ↔ x = 2 / 4 := sorry
  have h2 : 2 / 4 = 1 / 2 := sorry
  rw [abs_eq_zero, h1, h2]
  sorry

end abs_eq_zero_iff_l424_424047


namespace initial_distance_l424_424943

noncomputable def cheetah_speed_mph := 60
noncomputable def gazelle_speed_mph := 40
noncomputable def mph_to_fps := 1.5
noncomputable def time_to_catch := 7

noncomputable def cheetah_speed_fps := cheetah_speed_mph * mph_to_fps
noncomputable def gazelle_speed_fps := gazelle_speed_mph * mph_to_fps
noncomputable def relative_speed := cheetah_speed_fps - gazelle_speed_fps

theorem initial_distance (d : ℝ) :
  d = relative_speed * time_to_catch :=
by
  sorry

end initial_distance_l424_424943


namespace intersection_points_max_distance_l424_424737

-- Definition of curve C
def curve_C (θ : Real) : ℝ × ℝ :=
  (3 * Real.cos θ, Real.sin θ)

-- Parametric equation of line l
def line_l (a : Real) (t : Real) : ℝ × ℝ :=
  (a + 4 * t, 1 - t)

-- Proof Problem 1
theorem intersection_points (a : ℝ) (θ : ℝ) (if a = -1) :
  ∃ t,
  curve_C θ = line_l a t ↔ (curve_C θ = (3, 0) ∨ curve_C θ = (-21/25, 24/25)) :=
sorry

-- Proof Problem 2
theorem max_distance (a : ℝ) (θ : ℝ) :
  (∀ θ, ∃ t, Real.sqrt ((curve_C θ).1 - (line_l a t).1)^2 + ((curve_C θ).2 - (line_l a t).2)^2 = √17) ↔ (a = -16 ∨ a = 8) :=
sorry

end intersection_points_max_distance_l424_424737


namespace valentines_left_l424_424398

def initial_valentines : ℕ := 60
def valentines_given_away : ℕ := 16
def valentines_received : ℕ := 5

theorem valentines_left : (initial_valentines - valentines_given_away + valentines_received) = 49 :=
by sorry

end valentines_left_l424_424398


namespace smallest_quotient_div_l424_424583

theorem smallest_quotient_div (-1, 2, 0, : Set[ℝ]) : ∃ smallest : ℝ, 
  (∀ a b ∈ {−1, 2, 0}, smallest ≤ a / b) ∧ 
  (∀ a b ∈ {−1, 2, 0}, a / b = -2 → smallest = -2) := 
by
  let quotients := {-1 / 2, 2 / -1, 0 / -1, 0 / 2, -1 / 0, 2 / 0}
  let smallest := inf quotients
  use smallest
  apply and.intro
  · intros a ha b hb
    show smallest ≤ a / b
    sorry
  · intros a ha b hb h
    show smallest = a / b
    sorry

end smallest_quotient_div_l424_424583


namespace area_AFC_l424_424346

theorem area_AFC (EF FC : ℝ) (area_AEF : ℝ) :
  (EF / FC = 3 / 5) ∧ (area_AEF = 27) → (let area_AFC := area_AEF * (5 / 3) in area_AFC = 45) :=
by
  intros
  rcases H with ⟨h1, h2⟩
  let area_AFC := area_AEF * (5 / 3)
  have h_area_AFC : area_AFC = 45, by sorry
  exact h_area_AFC

end area_AFC_l424_424346


namespace cycle_efficiency_is_correct_l424_424118

def initial_pressure : ℝ := 3 * P₀
def final_pressure : ℝ := P₀
def initial_density : ℝ := ρ₀
def final_density : ℝ := 3 * ρ₀
def cycle_center : (ℝ × ℝ) := (1, 1)
def T₀ : ℝ := P₀ * V₀ / R
def T₁ : ℝ := P₀ * V₀ / R
def T₂ : ℝ := 3 * P₀ * V₀ / R
def Carnot_efficiency : ℝ := 1 - (T₁ / T₂)
def given_efficiency : ℝ := 1 / 8 * Carnot_efficiency

theorem cycle_efficiency_is_correct (P₀ V₀ ρ₀ R : ℝ) (H₀ : P₀ > 0) (H₁ : V₀ > 0) (H₂ : ρ₀ > 0) (H₃ : R > 0) :
  given_efficiency = 1 / 9 := sorry

end cycle_efficiency_is_correct_l424_424118


namespace seating_arrangement_round_table_l424_424339

open Nat

/-- The number of ways to seat 6 people around a round table with two specific people (A and B)
    sitting next to each other is 48. -/
theorem seating_arrangement_round_table (n : ℕ) (A B : Type) (table : Finset A) (people : Finset B) (h_table : table.card = 1)
  (h_people : people.card = 6) (ab : B) (h_ab : ab ∈ people) : by
  let units := (people.erase ab) ∪ (table)
  have h_units_count : units.card = 5, by
  rw [Finset.card_union, h_people, Finset.card_erase_of_mem h_ab, h_table],
  let ways_unit := (factorial (units.card - 1)),
  let ways_ab := 2,
  have ways_total := ways_unit * ways_ab,
  exact ways_total = 48
sorry

end seating_arrangement_round_table_l424_424339


namespace time_worked_together_l424_424911

noncomputable def work_rate_jane : ℚ := 1 / 4
noncomputable def work_rate_roy : ℚ := 1 / 5
noncomputable def combined_work_rate : ℚ := work_rate_jane + work_rate_roy
noncomputable def additional_work_time : ℚ := 0.4
noncomputable def additional_contribution_jane : ℚ := work_rate_jane * additional_work_time
noncomputable def total_task : ℚ := 1

theorem time_worked_together : 
  ∃ t : ℚ, 
    combined_work_rate * t + additional_contribution_jane = total_task ∧ 
    t = 2 :=
by {
  use 2,
  split,
  {
    unfold combined_work_rate work_rate_jane work_rate_roy additional_contribution_jane total_task,
    norm_num,
  },
  refl
}

end time_worked_together_l424_424911


namespace find_exponent_value_l424_424200

theorem find_exponent_value
  (x y : ℝ)
  (i : ℂ)
  (h1 : i * (x - 1) - y = 2 + i)
  (h2 : i^2 = -1) :
  (1 + i) ^ (x - y) = -4 := by
  -- Proof goes here
  sorry

end find_exponent_value_l424_424200


namespace sin_double_angle_l424_424666

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 4 / 5) : Real.sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l424_424666


namespace total_candles_in_small_boxes_l424_424644

-- Definitions of the conditions
def num_small_boxes_per_big_box := 4
def num_big_boxes := 50
def candles_per_small_box := 40

-- The total number of small boxes
def total_small_boxes : Nat := num_small_boxes_per_big_box * num_big_boxes

-- The statement to prove the total number of candles in all small boxes is 8000
theorem total_candles_in_small_boxes : candles_per_small_box * total_small_boxes = 8000 :=
by 
  sorry

end total_candles_in_small_boxes_l424_424644


namespace field_trip_students_l424_424008

theorem field_trip_students (seats_per_bus : ℕ) (num_buses : ℕ) (total_students : ℕ) :
  seats_per_bus = 2 → num_buses = 7 → total_students = seats_per_bus * num_buses → total_students = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [show 2 * 7 = 14, by norm_num] at h3
  exact h3

end field_trip_students_l424_424008


namespace number_of_pairs_count_number_of_pairs_l424_424302

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424302


namespace shortest_path_inside_triangle_l424_424966

noncomputable def point (α : Type*) := α × α

variables {α : Type*} [linear_ordered_field α]
variables (A B C P Q : point α)

def inside_triangle (A B C P : point α) : Prop :=
  -- Definition of point being inside the triangle using some geometric constraints

def touches_sides (P Q : point α) (A B C : point α) : Prop :=
  -- Definition of the path between P and Q touching all sides of triangle ABC

theorem shortest_path_inside_triangle
  (hP : inside_triangle A B C P)
  (hQ : inside_triangle A B C Q) :
  ∃ Q' : point α, 
  (touches_sides P Q A B C) ∧ 
  (∀ path, touches_sides P Q A B C → length path ≥ length (line_segment P Q')) :=
sorry

end shortest_path_inside_triangle_l424_424966


namespace average_salary_rest_l424_424335

theorem average_salary_rest (number_of_workers : ℕ) 
                            (avg_salary_all : ℝ) 
                            (number_of_technicians : ℕ) 
                            (avg_salary_technicians : ℝ) 
                            (rest_workers : ℕ) 
                            (total_salary_all : ℝ) 
                            (total_salary_technicians : ℝ) 
                            (total_salary_rest : ℝ) 
                            (avg_salary_rest : ℝ) 
                            (h1 : number_of_workers = 28)
                            (h2 : avg_salary_all = 8000)
                            (h3 : number_of_technicians = 7)
                            (h4 : avg_salary_technicians = 14000)
                            (h5 : rest_workers = number_of_workers - number_of_technicians)
                            (h6 : total_salary_all = number_of_workers * avg_salary_all)
                            (h7 : total_salary_technicians = number_of_technicians * avg_salary_technicians)
                            (h8 : total_salary_rest = total_salary_all - total_salary_technicians)
                            (h9 : avg_salary_rest = total_salary_rest / rest_workers) :
  avg_salary_rest = 6000 :=
by {
  -- the proof would go here
  sorry
}

end average_salary_rest_l424_424335


namespace delineate_rectangular_area_possible_l424_424472

noncomputable def regular_triangular_grid_side_length (A B C : ℝ) := 
(A = B) ∧ (B = C) ∧ (C = A)

def vertices_of_rectangle_are_trees (vertices : ℕ) := vertices

def distance_between_trees (d : ℝ) := d

def number_of_trees_inside_equals_boundary_trees
  (interior_trees boundary_trees : ℕ) :=
  interior_trees = boundary_trees 

theorem delineate_rectangular_area_possible
  (d : ℝ)
  (A B C : ℝ)
  (vertices : ℕ)
  (interior_trees boundary_trees : ℕ) :
  (regular_triangular_grid_side_length A B C) ∧
  (vertices_of_rectangle_are_trees vertices) ∧
  (distance_between_trees d) ∧
  (number_of_trees_inside_equals_boundary_trees interior_trees boundary_trees) →
  ∃ (b c : ℕ), 2 * (b + c) = b * c + (b - 1) * (c - 1) :=
sorry

end delineate_rectangular_area_possible_l424_424472


namespace prob_x_lt_y_is_correct_l424_424958

open Set

noncomputable def prob_x_lt_y : ℝ :=
  let rectangle := Icc (0: ℝ) 4 ×ˢ Icc (0: ℝ) 3
  let area_rectangle := 4 * 3
  let triangle := {p : ℝ × ℝ | p.1 ∈ Icc (0: ℝ) 3 ∧ p.2 ∈ Icc (0: ℝ) 3 ∧ p.1 < p.2}
  let area_triangle := 1 / 2 * 3 * 3
  let probability := area_triangle / area_rectangle
  probability

-- To state as a theorem using Lean's notation
theorem prob_x_lt_y_is_correct : prob_x_lt_y = 3 / 8 := sorry

end prob_x_lt_y_is_correct_l424_424958


namespace triangle_side_lengths_l424_424464

theorem triangle_side_lengths (A B C O M : Point) (a b c : ℝ)
    (hBC : segment B C = 12)
    (hRadius : OA = 10)
    (hBisect : is_midpoint M B C)
    (hPerpendicular : is_perpendicular (line O M) (line B C))
    (hOB : O.distance B = OA)
    (hM_coord : |B.coord - M.coord| = 6 ∧ |C.coord - M.coord| = 6)
    (hRightAngle : ∠ OMB = 90°) :
    segment A B = 2 * sqrt 10 ∧ segment A C = 2 * sqrt 10 := by
  sorry

end triangle_side_lengths_l424_424464


namespace anita_total_cartons_l424_424586

-- Defining the conditions
def cartons_of_strawberries : ℕ := 10
def cartons_of_blueberries : ℕ := 9
def additional_cartons_needed : ℕ := 7

-- Adding the core theorem to be proved
theorem anita_total_cartons :
  cartons_of_strawberries + cartons_of_blueberries + additional_cartons_needed = 26 := 
by
  sorry

end anita_total_cartons_l424_424586


namespace sum_digits_of_palindromes_l424_424774

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits in
  digits == digits.reverse

def valid_palindrome (n : ℕ) : Prop :=
  let digits := n.digits in
  n >= 10000 ∧ n <= 99999 ∧
  is_palindrome n ∧
  (( ∃ a b c, digits = [a, b, c, b, a]) ∧ a ≠ 0 ∧ b.even)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_digits_of_palindromes :
  sum_of_digits (Finset.sum (Finset.filter valid_palindrome (Finset.range 100000))) = 45 :=
by sorry

end sum_digits_of_palindromes_l424_424774


namespace height_of_crate_l424_424078

-- Definitions and conditions from the problem
def Crate : Type := {length : ℝ // length = 6} × {width : ℝ // width = 8} × {height : ℝ}
def Cylinder : Type := {radius : ℝ // radius = 4} × {diameter : ℝ // diameter = 2 * 4}

-- Proving that the height of the crate is 6 feet
theorem height_of_crate (c : Crate) (cy : Cylinder) : c.2.2 = 6 := sorry

end height_of_crate_l424_424078


namespace truncated_pyramid_volume_ratio_l424_424332

/-
Statement: Given a truncated triangular pyramid with a plane drawn through a side of the upper base parallel to the opposite lateral edge,
and the corresponding sides of the bases in the ratio 1:2, prove that the volume of the truncated pyramid is divided in the ratio 3:4.
-/

theorem truncated_pyramid_volume_ratio (S1 S2 h : ℝ) 
  (h_ratio : S1 = 4 * S2) :
  (h * S2) / ((7 * h * S2) / 3 - h * S2) = 3 / 4 :=
by
  sorry

end truncated_pyramid_volume_ratio_l424_424332


namespace leftmost_blue_off_mid_red_on_right_red_on_l424_424417

-- We define n as a positive integer, representing the total number of lamps.
noncomputable def probability_specified_configuration : ℚ :=
let total_lamps := 7 in
let red_lamps := 4 in
let blue_lamps := total_lamps - red_lamps in
let on_lamps := 4 in
let total_ways_to_choose_red_lamps := @nat.choose total_lamps red_lamps in
let total_ways_to_turn_on_lamps := @nat.choose total_lamps on_lamps in
let ways_to_choose_remaining_red_lamps := @nat.choose (total_lamps - 2) (red_lamps - 2) in
let ways_to_turn_on_remaining_lamps := @nat.choose (total_lamps - 2) (on_lamps - 2) in
(ways_to_choose_remaining_red_lamps * ways_to_turn_on_remaining_lamps : ℚ) / 
(total_ways_to_choose_red_lamps * total_ways_to_turn_on_lamps)

theorem leftmost_blue_off_mid_red_on_right_red_on :
  probability_specified_configuration = 4 / 49 :=
by sorry

end leftmost_blue_off_mid_red_on_right_red_on_l424_424417


namespace find_solutions_l424_424621

theorem find_solutions : 
    {x : ℝ | (15 * x - x^2) / (x + 1) * (x + (15 - x) / (x + 1)) = 60} = {5, 6, 3 + real.sqrt 2, 3 - real.sqrt 2} :=
sorry

end find_solutions_l424_424621


namespace necessary_and_sufficient_condition_correct_l424_424951

noncomputable def necessary_and_sufficient_condition (f : ℝ → ℝ) :=
  continuous f ∧ (∀ x, deriv (λ x, ∫ t in 0..x, f (x + t)) = 0) →
  (∑' n, f (2^n) = 1 ↔ f (2) = 1/2)

theorem necessary_and_sufficient_condition_correct (f : ℝ → ℝ) :
  necessary_and_sufficient_condition f := sorry

end necessary_and_sufficient_condition_correct_l424_424951


namespace veronica_brown_balls_l424_424885

theorem veronica_brown_balls (yellow_balls total_balls brown_balls : ℕ) 
  (h1 : yellow_balls = 27)
  (h2 : yellow_balls = (45 * total_balls) / 100) :
  brown_balls = total_balls - yellow_balls :=
by
  have h3 : total_balls = 60 := sorry
  show brown_balls = 33 from sorry

end veronica_brown_balls_l424_424885


namespace sum_of_possible_n_values_l424_424108

theorem sum_of_possible_n_values : 
  (∑ n in finset.Icc 5 13, n) = 81 :=
by
  sorry

end sum_of_possible_n_values_l424_424108


namespace chord_property_l424_424204

noncomputable def chord_length (R r k : ℝ) : Prop :=
  k = 2 * Real.sqrt (R^2 - r^2)

theorem chord_property (P O : Point) (R k : ℝ) (hR : 0 < R) (hk : 0 < k) :
  ∃ r, r = Real.sqrt (R^2 - k^2 / 4) ∧ chord_length R r k :=
sorry

end chord_property_l424_424204


namespace card_M_inter_N_l424_424714

-- Define the sets M and N.
def M : set (ℝ × ℝ) := 
{ p | tan (π * p.2) + sin (π * p.1) ^ 2 = 0}

def N : set (ℝ × ℝ) := 
{ p | p.1^2 + p.2^2 ≤ 2 }

-- State the proof problem.
theorem card_M_inter_N : (M ∩ N).to_finset.card = 9 := 
by sorry

end card_M_inter_N_l424_424714


namespace count_pairs_satisfying_condition_l424_424266

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424266


namespace interval_of_monotonic_increase_triangle_area_l424_424695

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6)) + Real.cos (2 * x)

theorem interval_of_monotonic_increase (k : ℤ) :
  ∀ x, (k * Real.pi - (5 * Real.pi / 12)) ≤ x ∧ x ≤ (k * Real.pi + (Real.pi / 12)) → Real.mono_increasing_on (λ x, sqrt 3 * Real.sin (2 * x + Real.pi / 3)) x :=
sorry

theorem triangle_area (A B : ℝ) (a : ℝ) (hA : A = Real.pi / 4) (hB : B = Real.pi / 3) (ha : a = 2) (hfA : f A = sqrt 3 / 2) :
  ∃ b C S, b = sqrt 6 ∧ Real.sin C = (sqrt 2 + sqrt 6) / 4 ∧ S = (3 + sqrt 3) / 2 :=
sorry

end interval_of_monotonic_increase_triangle_area_l424_424695


namespace range_of_a_l424_424814

def function_increasing_on_interval {a : ℝ} (f : ℝ → ℝ) (interval : set ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) (a : ℝ) := -x^2 + 2*(a-1)*x + 2

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ∈ set.Iic 4 → y ∈ set.Iic 4 → x < y → f x a < f y a) ↔ a ≥ 5 := 
by
  sorry

end range_of_a_l424_424814


namespace int_solutions_ineq_l424_424192

theorem int_solutions_ineq (x : ℤ) :
  (x - 2) / 2 ≤ -x / 2 + 2 ∧ 4 - 7 * x < -3 ↔ x = 2 ∨ x = 3 := by
  -- Translations for inequalities
  have h1 : x / 2 - 1 ≤ -x / 2 + 2 := by
    rw [sub_eq_add_neg, add_comm, add_assoc, neg_add, add_assoc, add_right_neg, zero_add,
        ←add_le_add_iff_left (x / 2 + 1)]
    exact (by linarith : (x / 2 + x / 2 + 1 : ℤ) ≤ 3)
  -- Translate the second inequality
  have h2 : -3 < 4 - 7 * x := by
    rw [sub_lt_neg_iff_lt_add, left.neg_add, sub_lt_zero, sub_eq_add_neg, add_comm]
    exact (by linarith : (7 * x : ℤ) > 1)
  sorry  -- Proof to be filled based on translation

end int_solutions_ineq_l424_424192


namespace prism_sphere_surface_area_l424_424805

theorem prism_sphere_surface_area :
  ∀ (a b c : ℝ), (a * b = 6) → (b * c = 2) → (a * c = 3) → 
  4 * Real.pi * ((Real.sqrt ((a ^ 2) + (b ^ 2) + (c ^ 2))) / 2) ^ 2 = 14 * Real.pi :=
by
  intros a b c hab hbc hac
  sorry

end prism_sphere_surface_area_l424_424805


namespace staircase_perimeter_l424_424349

/--
Given a staircase-shaped region with the following properties:
1. Each of the eight congruent sides marked with a tick mark has length 1 foot.
2. The region has an area of 53 square feet.
3. The bottom length of the region is 9 feet.
4. The small staircase removed consists of ten 1 foot by 1 foot squares.
Prove that the perimeter of the region is 32 feet.
-/
theorem staircase_perimeter
  (a b c d e f g h : ℝ) -- lengths of the 8 congruent sides marked with a tick mark
  (h_tick : a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 1 ∧ h = 1)
  (region_area : ℝ)
  (h_area : region_area = 53)
  (base_length : ℝ)
  (h_base : base_length = 9)
  (small_staircase_area : ℝ)
  (h_small_staircase : small_staircase_area = 10) :
  let
    total_height := 7, -- calculated height
    rect_length := base_length,
    rect_area := rect_length * total_height,
    actual_area := rect_area - small_staircase_area
  in
  actual_area = region_area →
  let
    rect_perimeter := 2 * (total_height + rect_length),
    steps_perimeter := 2 * 4, -- each step contributing 1 foot both vertically and horizontally
    ticks_perimeter := 8
  in
  rect_perimeter + steps_perimeter + ticks_perimeter = 32 := by {
    sorry,
  }

end staircase_perimeter_l424_424349


namespace polynomial_identity_l424_424383

def a (x y : ℕ) : ℕ := x^2 * y + x * y^2
def b (x y : ℕ) : ℕ := x^2 + x * y + y^2
def E (n x y : ℕ) : ℕ := (x + y)^n + (-1)^n * (x^n + y^n)

theorem polynomial_identity (n x y : ℕ) : 
  ∃ (p_n : ℕ → ℕ → ℕ → ℕ), E n x y = p_n (a x y) (b x y) :=
by
  sorry

end polynomial_identity_l424_424383


namespace total_cups_for_8_batches_l424_424403

def cups_of_flour (batches : ℕ) : ℝ := 4 * batches
def cups_of_sugar (batches : ℕ) : ℝ := 1.5 * batches
def total_cups (batches : ℕ) : ℝ := cups_of_flour batches + cups_of_sugar batches

theorem total_cups_for_8_batches : total_cups 8 = 44 := 
by
  -- This is where the proof would go
  sorry

end total_cups_for_8_batches_l424_424403


namespace problem1_problem2_problem3_l424_424773

noncomputable def Sn (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of the sequence {a_n}
noncomputable def an (n : ℕ) : ℝ := sorry  -- nth term of the sequence {a_n}
noncomputable def xn (n : ℕ) : ℝ := sorry  -- The other root of the equation x^2 - a_n x - a_n = 0
noncomputable def Tn (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of the sequence \( \frac{1}{2^n x_n} \)

-- Condition: The equation x^2 - a_n x - a_n = 0 has a root S_n - 1
axiom eq1 (n : ℕ) (hn : 1 ≤ n) : (Sn n - 1) ^ 2 - an n * (Sn n - 1) - an n = 0

-- Problem 1: Prove that { \frac{1}{S_n - 1} } is an arithmetic sequence
theorem problem1 : ∃ a d : ℝ, ∀ n : ℕ, 1 ≤ n → 1 / (Sn n - 1) = a + d * (n - 1) := sorry

-- Problem 2: Find 2^2013 * (2 - T_2013)
theorem problem2 : 2 ^ 2013 * (2 - Tn 2013) = 2015 := sorry

-- Problem 3: Prove the existence of positive integers p and q such that S_1, S_p, and S_q form a geometric sequence
theorem problem3 : ∃ p q : ℕ, 1 ≤ p ∧ 1 ≤ q ∧ p ≠ q ∧ (Sn p)^2 = (Sn 1) * (Sn q) :=
  begin
    use [2, 8],
    sorry
  end

end problem1_problem2_problem3_l424_424773


namespace smallest_positive_period_f_max_value_g_l424_424254

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x)
noncomputable def g (x m : ℝ) : ℝ := f x + m

-- Statement 1: Prove that the smallest positive period of f(x) is π.
theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
  sorry

-- Statement 2: Given f(x) = sin(2x + π/6) + 1/2 and x in [-π/6, π/3],
-- prove that the maximum value of g(x) = f(x) + 2 is 7/2.
theorem max_value_g : ∀ x ∈ Icc (-Real.pi / 6) (Real.pi / 3), 
  g x 2 ≤ 7 / 2 ∧ ∃ x_max, x_max = Real.pi / 6 ∧ g x_max 2 = 7 / 2 := 
  sorry

end smallest_positive_period_f_max_value_g_l424_424254


namespace square_division_rectangles_l424_424965

theorem square_division_rectangles (k l : ℕ) (h_square : exists s : ℝ, 0 < s) 
(segment_division : ∀ (p q : ℝ), exists r : ℕ, r = s * k ∧ r = s * l) :
  ∃ n : ℕ, n = k * l :=
sorry

end square_division_rectangles_l424_424965


namespace example_correct_set_exists_count_correct_sets_l424_424990

def correct_set (weights : List ℕ) : Prop :=
  ∀ n ∈ (List.range' 1 200), ∃ unique combination : List ℕ, 
    List.perm (combination) (List.filter (λ w, w ∈ weights) (List.repeat n (List.length combination)))

theorem example_correct_set_exists : 
  ∃ weights : List ℕ, 
    List.sum weights = 200 ∧
    correct_set weights ∧
    (¬∀ w ∈ weights, w = 1) :=
sorry

theorem count_correct_sets : 
  ∃ count : ℕ, 
    count = 3 ∧ 
    ∀ weights : List ℕ, 
      List.sum weights = 200 → 
      correct_set weights → 
      (∀ w₁ w₂ ∈ weights, w₁ = w₂) :=
sorry

end example_correct_set_exists_count_correct_sets_l424_424990


namespace sum_of_squares_of_four_integers_equals_175_l424_424457

theorem sum_of_squares_of_four_integers_equals_175 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a^2 + b^2 + c^2 + d^2 = 175 ∧ a + b + c + d = 23 :=
sorry

end sum_of_squares_of_four_integers_equals_175_l424_424457


namespace rent_expense_l424_424413

theorem rent_expense (salary gross: ℕ) (tax_percentage: ℕ) (rent_months: ℕ) :
  gross = 5000 → tax_percentage = 10 → rent_months = 2 → salary = gross * (100 - tax_percentage) / 100 → 
  (3 * salary / 5) / rent_months = 1350 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end rent_expense_l424_424413


namespace no_lattice_points_on_hyperbola_l424_424844

theorem no_lattice_points_on_hyperbola : ∀ x y : ℤ, x^2 - y^2 ≠ 2022 :=
by
  intro x y
  -- proof omitted
  sorry

end no_lattice_points_on_hyperbola_l424_424844


namespace three_digit_numbers_sum_24_l424_424638

noncomputable def count_valid_numbers : ℕ := 
  let valid_digits (a b c : ℕ) := (a + b + c = 24) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) in
  finset.card { (a, b, c) | valid_digits a b c }

theorem three_digit_numbers_sum_24 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9) (h7 : a + b + c = 24) :
  count_valid_numbers = 5 :=
sorry

end three_digit_numbers_sum_24_l424_424638


namespace value_of_m_maximize_profit_l424_424812

-- Define the sales volume y as a function of x and m
def sales_volume (x : ℝ) (m : ℝ) : ℝ := (m / (x - 3)) + 8 * (x - 6)^2

-- Define the daily profit function 
def daily_profit (x : ℝ) (m : ℝ) : ℝ := (x - 3) * (sales_volume x m)

-- The value of m when x = 5 and y = 11
theorem value_of_m : 
  (x : ℝ) (hx : x = 5) (y : ℝ) (hy : y = 11) :
  sales_volume 5 m = 11 → m = 6 :=
sorry

-- The value of x that maximizes daily profit when cost price is 3 yuan/kg
theorem maximize_profit :
  (m : ℝ) (hm : m = 6) (cost : ℝ) (hc : cost = 3) :
  ∃ (x : ℝ), x = 4 ∧ 
  (∀ (y : ℝ), (3 < y) ∧ (y < 6) → daily_profit y m ≤ daily_profit x m) :=
sorry

end value_of_m_maximize_profit_l424_424812


namespace sum_of_possible_radii_eq_14_l424_424546

theorem sum_of_possible_radii_eq_14 (r : ℝ) (h1 : r > 0) (h2 : (r - 5)^2 + r^2 = (r + 2)^2) : 
  let r1 := 7 + 2 * Real.sqrt 7 in
  let r2 := 7 - 2 * Real.sqrt 7 in
  (r = r1 ∨ r = r2) →
  r1 + r2 = 14 :=
by
  intros hr
  sorry

end sum_of_possible_radii_eq_14_l424_424546


namespace find_principal_l424_424053

variable (P : ℝ)
variable (SI : ℝ) (R : ℝ) (T : ℝ)

noncomputable def simple_interest := P * R * T / 100

theorem find_principal :
  SI = 4016.25 → R = 5 → T = 5 → P = 16065 :=
by
  intros SI_eq R_eq T_eq
  rw [SI_eq, R_eq, T_eq]
  sorry

end find_principal_l424_424053


namespace base_satisfies_equation_l424_424613

namespace BaseProof

def value142_b (b : ℕ) : ℕ := 1 * b^2 + 4 * b + 2
def value243_b (b : ℕ) : ℕ := 2 * b^2 + 4 * b + 3
def value405_b (b : ℕ) : ℕ := 4 * b^2 + 0 * b + 5

theorem base_satisfies_equation : ∃ (b : ℕ), 142_b + 243_b = 405_b →
  (value142_b b + value243_b b = value405_b b) ∧ b = 8 :=
by
  sorry

end BaseProof

end base_satisfies_equation_l424_424613


namespace increasing_function_on_interval_l424_424816

noncomputable def f (a x : ℝ) : ℝ := -x^2 + 2 * (a - 1) * x + 2

theorem increasing_function_on_interval (a : ℝ) :
  (∀ x, x < 4 → f(a, x) < f(a, x + 1)) → a ≥ 5 :=
sorry

end increasing_function_on_interval_l424_424816


namespace ratio_of_intercepts_l424_424873

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l424_424873


namespace diver_descend_rate_l424_424566

theorem diver_descend_rate (depth : ℕ) (time : ℕ) (rate : ℕ) 
  (h1 : depth = 6400) (h2 : time = 200) : rate = 32 :=
by
  sorry

end diver_descend_rate_l424_424566


namespace find_n_l424_424502

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 :=
by {
    -- Let's assume there exists an integer n such that the given condition holds
    use 4,
    -- We now prove the condition and the conclusion
    split,
    -- This simplifies the left-hand side of the condition to 15, achieving the goal
    calc 
        4 + (4 + 1) + (4 + 2) = 4 + 5 + 6 : by rfl
        ... = 15 : by norm_num,
    -- The conclusion directly follows
    rfl
}

end find_n_l424_424502


namespace college_students_freshmen_psych_majors_l424_424590

variable (T : ℕ)
variable (hT : T > 0)

def freshmen (T : ℕ) : ℕ := 40 * T / 100
def lib_arts (F : ℕ) : ℕ := 50 * F / 100
def psych_majors (L : ℕ) : ℕ := 50 * L / 100
def percent_freshmen_psych_majors (P : ℕ) (T : ℕ) : ℕ := 100 * P / T

theorem college_students_freshmen_psych_majors :
  percent_freshmen_psych_majors (psych_majors (lib_arts (freshmen T))) T = 10 := by
  sorry

end college_students_freshmen_psych_majors_l424_424590


namespace bus_speed_in_km_hr_l424_424541

-- Given definitions
def bus_length : ℝ := 15 -- in meters
def time_to_pass : ℝ := 1.125 -- in seconds
def man_speed_km_hr : ℝ := 8 -- in km/hr
def man_speed_m_s : ℝ := man_speed_km_hr * (1000 / 3600) -- converting to m/s

-- Hypothesis/Conditions
theorem bus_speed_in_km_hr : 
  let relative_speed := bus_length / time_to_pass in
  let bus_speed_m_s := relative_speed - man_speed_m_s in
  bus_speed_m_s * (3600 / 1000) = 40 :=
by
  sorry

end bus_speed_in_km_hr_l424_424541


namespace find_number_l424_424916

theorem find_number : ∃ x : ℝ, (x / 5 + 7 = x / 4 - 7) ∧ x = 280 :=
by
  -- Here, we state the existence of a real number x
  -- such that the given condition holds and x = 280.
  sorry

end find_number_l424_424916


namespace MV_length_l424_424347

-- Definitions based on conditions
def rectangle_MNOP : Prop := ∃ (MN NO : ℝ), MN = 3 ∧ NO = 2
def congruent_rectangles : Prop := ∃ (a b : ℝ), a * b = 1 ∧ (3 - a) + b = 3
def total_area_condition : Prop := ∃ (MN NO : ℝ), MN * NO = 6 ∧ (2 * (MN - a) * b) = 2

-- Theorem statement based on the problem question and solution
theorem MV_length (MN NO a b : ℝ) (x : ℝ) :
  rectangle_MNOP →
  congruent_rectangles →
  total_area_condition →
  x = 3 - a →
  x = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end MV_length_l424_424347


namespace routes_from_A_to_B_in_4_by_3_grid_l424_424134

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end routes_from_A_to_B_in_4_by_3_grid_l424_424134


namespace least_days_to_repay_l424_424777

-- Definitions based on the conditions
def amount_borrowed : ℕ := 10
def daily_interest_rate : ℕ := 20 -- as a percentage
def total_multiplier : ℕ := 3

-- The inequality Mark needs to satisfy
def repayment_days (x : ℕ) : Prop := 
  amount_borrowed + (amount_borrowed * daily_interest_rate / 100) * x ≥ amount_borrowed * total_multiplier

-- The proof statement
theorem least_days_to_repay : ∃ x : ℕ, repayment_days x ∧ ∀ y : ℕ, repayment_days y → x ≤ y :=
begin
  -- skip the proof with sorry
  sorry
end

end least_days_to_repay_l424_424777


namespace ratio_of_x_intercepts_l424_424876

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l424_424876


namespace num_relatively_prime_pairs_eq_ten_l424_424968

def count_relatively_prime_pairs : ℕ :=
  finset.card (finset.filter (λ p : ℕ × ℕ, p.1 ≤ p.2 ∧ Nat.gcd p.1 p.2 = 1)
                             (finset.Icc (1, 1) (5, 5)))

theorem num_relatively_prime_pairs_eq_ten : count_relatively_prime_pairs = 10 := by
  sorry

end num_relatively_prime_pairs_eq_ten_l424_424968


namespace num_pairs_nat_nums_eq_l424_424277

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424277


namespace isosceles_triangle_EBD_l424_424135

theorem isosceles_triangle_EBD :
  ∀ (A B C E: Type) [RightTriangle A B C] (AC_diameter_semi: Semicircle AC)
    (E_hyp_intersect: Intersection AC_diameter_semi Hypotenuse(E)) (tangent_at_E : Tangent AC_diameter_semi E),
  is_isosceles_triangle EBD :=
by
  sorry

end isosceles_triangle_EBD_l424_424135


namespace twelve_meal_order_l424_424864

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => n * factorial n

noncomputable def derangement : ℕ → ℕ
| 0     => 1
| 1     => 0
| (n + 1) => (n * (derangement n + derangement (n - 1)))

def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem twelve_meal_order : 
  let n_people := 12
  let meals := ["beef", "beef", "beef", "chicken", "chicken", "chicken", "fish", "fish", "fish", "vegetarian", "vegetarian", "vegetarian"]
  let exact_two_correct := 
    binom n_people 2 * 
    (factorial 10 - binom 10 1 * derangement 9 + binom 10 2 * derangement 8) =
  208581450 
  in 
  exact_two_correct := 208581450
  :=
by
  sorry

end twelve_meal_order_l424_424864


namespace derivative_f_l424_424624

def f (x : ℝ) : ℝ := x * Real.cos x + Real.sin x

theorem derivative_f (x : ℝ) : deriv f x = 2 * Real.cos x - x * Real.sin x := 
by 
  sorry

end derivative_f_l424_424624


namespace probability_y_le_sin_x_l424_424490

open Real

def probability_event_le_sin (x y : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ π ∧ 0 ≤ y ∧ y ≤ π
then (∫ x in 0..π, sin x) / (π * π)
else 0

theorem probability_y_le_sin_x :
  ∀ x y, 0 ≤ x ∧ x ≤ π ∧ 0 ≤ y ∧ y ≤ π → 
  probability_event_le_sin x y = 2 / π^2 :=
sorry

end probability_y_le_sin_x_l424_424490


namespace area_of_rectangle_l424_424341

noncomputable def leanProblem : Prop :=
  let E := 8
  let F := 2.67
  let BE := E -- length from B to E on AB
  let AF := F -- length from A to F on AD
  let BC := E * (Real.sqrt 3) -- from triangle properties CB is BE * sqrt(3)
  let FD := BC - F -- length from F to D on AD
  let CD := FD * (Real.sqrt 3) -- applying the triangle properties again
  (BC * CD = 192 * (Real.sqrt 3) - 64.08)

theorem area_of_rectangle (E : ℝ) (F : ℝ) 
  (hE : E = 8) 
  (hF : F = 2.67) 
  (BC : ℝ) (CD : ℝ) :
  leanProblem :=
by 
  sorry

end area_of_rectangle_l424_424341


namespace log_base_2_six_l424_424650

-- Conditions
variables (a b : ℝ)
-- \(10^a = 3\)
axiom h1 : 10^a = 3
-- \(\lg 2 = b\)
axiom h2 : Real.log 2 = b

-- Required to prove
theorem log_base_2_six :
  Real.log 6 / Real.log 2 = 1 + a / b := by
  sorry

end log_base_2_six_l424_424650


namespace part1_part2_part3_l424_424215

def setA : set ℝ := {x | x^2 + 6*x + 5 < 0}
def setB : set ℝ := {x | -1 ≤ x ∧ x < 1}
def universalSet : set ℝ := {x | -5 < x ∧ x < 5}

theorem part1 : A ∩ B = ∅ :=
by sorry

theorem part2 : (universalSet \ (A ∪ B)) = {x | 1 ≤ x ∧ x < 5} :=
by sorry

theorem part3 (a : ℝ) 
  (C : set ℝ := {x | x < a}) 
  (h : B ∩ C = B) : a ≥ 1 :=
by sorry

end part1_part2_part3_l424_424215


namespace PT_length_expression_l424_424799

-- Definitions based on problem conditions
def square_side_length := Real.sqrt 2
def PT (x : ℝ) := x
def QU (x : ℝ) := x

-- Translated Problem Statement
theorem PT_length_expression (x : ℝ) (a b : ℕ) (h1 : PT x = QU x)
  (h2 : PT x = Real.sqrt 8 - 2) :
  a = 8 ∧ b = 2 → a + b = 10 :=
by
  sorry

end PT_length_expression_l424_424799


namespace number_of_pairs_count_number_of_pairs_l424_424299

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424299


namespace area_of_triangle_l424_424661

-- Define the conditions from the problem
def C : ℝ := 30 * Real.pi / 180 -- converting degrees to radians
def AC : ℝ := 3 * Real.sqrt 3
def AB : ℝ := 3

-- Define the sin function in radians
noncomputable def sin (x : ℝ) : ℝ := Real.sin x

-- Assume B can be either 60 degrees or 120 degrees
def B1 : ℝ := 60 * Real.pi / 180
def B2 : ℝ := 120 * Real.pi / 180

-- Prove the area can be either of the two values
theorem area_of_triangle (B : ℝ) (hB : B = B1 ∨ B = B2) :
  let A := Real.pi - B - C in
  let area1 := (1/2) * AC * AB * sin B1 in
  let area2 := (1/2) * AC * AB * sin (Real.pi - B2 - C) in
  B = B1 → (1/2) * AC * AB * sin A = area1 ∨ 
  B = B2 → (1/2) * AC * AB * sin A = area2 := 
by {
  intros,
  cases hB;
  { 
    subst hB,
    sorry
  }
}

end area_of_triangle_l424_424661


namespace isosceles_triangle_min_perimeter_l424_424406

noncomputable def triangle (a b c : ℝ) : ℝ := a + b + c
noncomputable def isosceles_triangle_perimeter(base height : ℝ) : ℝ := 2 * real.sqrt ((base / 2) ^ 2 + height ^ 2) + base
noncomputable def general_triangle_perimeter (base height x : ℝ) : ℝ := real.sqrt (x ^ 2 + height ^ 2) + real.sqrt ((base - x) ^ 2 + height ^ 2) + base

theorem isosceles_triangle_min_perimeter (b h : ℝ) (A : ℝ := 1 / 2 * b * h)
  : ∀ (x : ℝ), isosceles_triangle_perimeter b h ≤ general_triangle_perimeter b h x :=
sorry

end isosceles_triangle_min_perimeter_l424_424406


namespace find_m_l424_424700

variable (m : ℕ)
def A := {1, 3, m}
def B := {3, 4}
def A_union_B := A ∪ B

theorem find_m (h : A_union_B = {1, 2, 3, 4}) : m = 2 := 
  sorry

end find_m_l424_424700


namespace trigonometric_identity_l424_424668

theorem trigonometric_identity (α : Real) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10) / Real.sin (α - Real.pi / 5) = 3) :=
sorry

end trigonometric_identity_l424_424668


namespace num_increasing_digits_l424_424179

theorem num_increasing_digits :
  let C := λ (n k : ℕ), Nat.choose n k in
  ∑ k in Finset.range 8, C 9 (k + 2) = 502 :=
by
  sorry

end num_increasing_digits_l424_424179


namespace find_a_tangent_line_l424_424804

theorem find_a_tangent_line (a : ℝ) : 
  (∃ (x0 y0 : ℝ), y0 = a * x0^2 + (15/4 : ℝ) * x0 - 9 ∧ 
                  (y0 = 0 ∨ (x0 = 3/2 ∧ y0 = 27/4)) ∧ 
                  ∃ (m : ℝ), (0 - y0) = m * (1 - x0) ∧ (m = 2 * a * x0 + 15/4)) → 
  (a = -1 ∨ a = -25/64) := 
sorry

end find_a_tangent_line_l424_424804


namespace fractions_addition_l424_424495

theorem fractions_addition :
  (1 / 3) * (3 / 4) * (1 / 5) + (1 / 6) = 13 / 60 :=
by 
  sorry

end fractions_addition_l424_424495


namespace cyclic_trapezoid_problem_l424_424017

variables (A B C D E : Type)
variables (r1 r2 : ℝ) (BC AE BE : ℝ)
variables a b c d e : A

-- Given conditions:
-- 1. \ABCD\ is a cyclic trapezoid (inscribed in a circle)
-- 2. The second circle passing through \A and \C touches \CD\ and intersects the extension of \BC\ at \E.
def is_cyclic_trapezoid (a b c d : A) : Prop := sorry
def touches_line (circ : Type) (l : Type) : Prop := sorry
def intersects_extension (circ : Type) (point : A) (l : Type) : Prop := sorry

-- Given values
def BC_length : BC = 7 := sorry
def AE_length : AE = 12 := sorry

-- Required to find BE and the ratio of the radii of the circles
def BE_value : BE = 2 * (Real.sqrt 21) := sorry

def radius_ratio (r1 r2 : ℝ) : Prop :=
  r1 / r2 ∈ (Set.Ioo (1/4) (Real.sqrt 7 / 4) ∪ Set.Ioo (Real.sqrt 7 / 4) (7 / 4))

theorem cyclic_trapezoid_problem 
  (is_cyclic_trapezoid a b c d) 
  (circ1 passes through b and c)
  (circle2 passes through a and c touches_line CD)
  (BC_length : BC = 7)
  (AE_length : AE = 12)
  : BE_value
  ∧ radius_ratio r1 r2 := 
begin
  sorry
end

end cyclic_trapezoid_problem_l424_424017


namespace cosine_square_sum_l424_424605

theorem cosine_square_sum : 
  (∑ k in finset.range 30, (cos ((6 * k + 3) * (real.pi / 180)))^2) = 14.5 :=
by
  sorry

end cosine_square_sum_l424_424605


namespace marks_in_chemistry_l424_424993

-- Define the given conditions
def marks_english := 76
def marks_math := 65
def marks_physics := 82
def marks_biology := 85
def average_marks := 75
def number_subjects := 5

-- Define the theorem statement to prove David's marks in Chemistry
theorem marks_in_chemistry :
  let total_marks := marks_english + marks_math + marks_physics + marks_biology
  let total_marks_all_subjects := average_marks * number_subjects
  let marks_chemistry := total_marks_all_subjects - total_marks
  marks_chemistry = 67 :=
sorry

end marks_in_chemistry_l424_424993


namespace count_valid_pairs_l424_424263

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424263


namespace greatest_divisors_in_range_l424_424400

-- Definition of the set of numbers from 1 to 20
def nums : List ℕ := List.range' 1 21

-- Definition of the set of numbers from 1 to 20 that are divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def nums_div_by_4 : List ℕ := nums.filter divisible_by_4

-- Function to count the number of divisors of a number
def num_of_divisors (n : ℕ) : ℕ :=
  List.length (List.filter (λ d, n % d = 0) (List.range' 1 (n + 1)))

-- Claim: The numbers 12 and 20 have the most divisors among numbers divisible by 4 in the range 1 to 20
theorem greatest_divisors_in_range : (∀ n ∈ nums_div_by_4, num_of_divisors n ≤ 6) ∧ (12 ∈ nums_div_by_4 ∧ num_of_divisors 12 = 6) ∧ (20 ∈ nums_div_by_4 ∧ num_of_divisors 20 = 6) :=
by
  sorry

end greatest_divisors_in_range_l424_424400


namespace area_grazed_by_horse_area_grazed_by_horse_approx_l424_424914

noncomputable def area_grazed (r : ℝ) : ℝ := (π * r^2) / 4

theorem area_grazed_by_horse :
  let r := 14 in
  area_grazed r = 49 * π :=
by
  sorry

theorem area_grazed_by_horse_approx :
  (49 : ℝ) * real.pi ≈ 153.94 :=
by
  sorry

end area_grazed_by_horse_area_grazed_by_horse_approx_l424_424914


namespace probability_at_least_one_different_probability_at_least_one_equal_l424_424927

open scoped ProbabilityTheory

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Statement a: For different probabilities p_i, the probability that at least one of the events A_i occurs is 1 - ∏_{i=1}^{n}(1 - p_i) -/
theorem probability_at_least_one_different (n : ℕ) (A : Fin n → Event Ω) (p : Fin n → ℝ) 
  (h_independent : Pairwise (λ i j, A i ∩ A j = A i ∩ A j))
  (h_probability : ∀ i, ProbabilitySpace.Probability (A i) = p i) : 
  ProbabilitySpace.Probability (⋃ i, A i) = 1 - ∏ i, (1 - p i) :=
sorry

/-- Statement b: For equal probabilities p, the probability that at least one of the events A_i occurs is 1 - (1 - p)^n -/
theorem probability_at_least_one_equal (n : ℕ) (A : Fin n → Event Ω) (p : ℝ)
  (h_independent : Pairwise (λ i j, A i ∩ A j = A i ∩ A j))
  (h_probability : ∀ i, ProbabilitySpace.Probability (A i) = p) : 
  ProbabilitySpace.Probability (⋃ i, A i) = 1 - (1 - p)^n :=
sorry

end probability_at_least_one_different_probability_at_least_one_equal_l424_424927


namespace Triangle_ABC_properties_l424_424337

theorem Triangle_ABC_properties 
  (a b c A B C : ℝ)
  (h_acute : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_c : c = 2)
  (h_sqrt3a : sqrt 3 * a = 2 * c * sin A) :
  (C = π / 3) ∧ ( (area_ABC : ℝ), (h_area : area_ABC = sqrt 3), a = 2 ∧ b = 2) ∧ (max_area_ABC : ℝ), max_area_ABC = sqrt 3 := 
sorry

end Triangle_ABC_properties_l424_424337


namespace MisfortuneProtestMax_l424_424524

universe u

noncomputable def maxProtesters (n reforms dislikes perReform dislikeThreshold : ℕ) : ℕ :=
  if h : dislikes = n * reforms / 2 ∧ threshold = (reforms + 1) / 2 then
    n * reforms / (2 * threshold)
  else
    sorry

theorem MisfortuneProtestMax : maxProtesters 96 5 48 3 = 80 :=
by
  have h1 : 48 = 96 * 5 / 2 := by sorry
  have h2 : 3 = (5 + 1) / 2 := by sorry
  unfold maxProtesters
  split_ifs
  case pos h => sorry
  case neg h => sorry

end MisfortuneProtestMax_l424_424524


namespace roots_of_polynomial_l424_424630

noncomputable def poly (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem roots_of_polynomial :
  ∀ x : ℝ, poly x = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end roots_of_polynomial_l424_424630


namespace cross_product_example_l424_424623

open Matrix

def vector_cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (ux, uy, uz) := u
  let (vx, vy, vz) := v
  (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)

theorem cross_product_example :
  vector_cross_product (-3, 4, 2) (8, -5, 6) = (34, -34, -17) := by
  sorry

end cross_product_example_l424_424623


namespace number_of_ordered_triples_l424_424629

/-- 
Prove the number of ordered triples (x, y, z) of positive integers that satisfy 
  lcm(x, y) = 180, lcm(x, z) = 210, and lcm(y, z) = 420 is 2.
-/
theorem number_of_ordered_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h₁ : Nat.lcm x y = 180) (h₂ : Nat.lcm x z = 210) (h₃ : Nat.lcm y z = 420) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end number_of_ordered_triples_l424_424629


namespace subtracted_number_from_32_l424_424715

theorem subtracted_number_from_32 (x : ℕ) (h : 32 - x = 23) : x = 9 := 
by 
  sorry

end subtracted_number_from_32_l424_424715


namespace is_complex_expression_l424_424507

-- Define z as given in the condition
def z : ℂ := (1 - complex.I) / real.sqrt 2

-- State the theorem, ensuring to use the given condition and final result only
theorem is_complex_expression :
  z ^ 100 + z ^ 50 + 1 = -complex.I :=
by
  sorry

end is_complex_expression_l424_424507


namespace values_satisfy_ffx_eq_5_l424_424445

def f (x : ℝ) : ℝ := 
-- This captures the behavior on given points.
  if x = 3 then 5 else if x = -3 || x = 1 || x = 5 then 3 else 0

theorem values_satisfy_ffx_eq_5 : ∃ (xs : Finset ℝ), xs.card = 3 ∧ ∀ x ∈ xs, f (f x) = 5 :=
by
  -- Define the set of values that meet the condition.
  let xs := { -3, 1, 5 }
  use xs
  -- Prove the set has 3 elements.
  have : xs.card = 3, from Finset.card_mk _,
  have h : ∀ x ∈ xs, f (f x) = 5, 
  { intros x hx,
    finset_cases hx with cases
    { simp [f], }, { simp [f], }, { simp [f], } },
  exact ⟨this, h⟩

end values_satisfy_ffx_eq_5_l424_424445


namespace square_area_l424_424746

-- Definitions of the conditions
def is_square (s : Type) (side_length : ℕ) := 
  ∀ x ∈ s, x ∈ set.univ -- This is just a placeholder to formalize s being a mathematical set that we call a square.

-- Define the problem
theorem square_area (s : Type) (side_length : ℕ) (h1: is_square s side_length) (h2: ∀ s, ∃ side_length = 4) : 
  ∃ area : ℕ, area = 16 :=
sorry

end square_area_l424_424746


namespace base2_to_base8_conversion_l424_424889

theorem base2_to_base8_conversion : 
  (binary_to_base8 110110110_2) = 666_8 := 
by 
  -- Define the function binary_to_base8 if not already existing or adapted to our needs for clarity
  sorry

end base2_to_base8_conversion_l424_424889


namespace bruce_bank_savings_l424_424984

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end bruce_bank_savings_l424_424984


namespace min_steps_to_no_zeros_l424_424475

theorem min_steps_to_no_zeros : 
  ∀ (zeroes ones : ℕ), zeroes = 150 → ones = 151 → 
  (∀ step_count, (step_count ≥ 0) → 
    (∀ (p q r : ℕ), (p = 0 ∨ p = 1) → (q = 0 ∨ q = 1) → (r = 0 ∨ r = 1) →
    (p = q ∧ q = r → step_count = step_count) ∧ (p ≠ q ∨ q ≠ r → (p ∨ q = r) → step_count + 1 ≤ 150)) → 
  step_count = 150) :=
by
  intros zeroes ones h_zeroes h_ones h_steps
  have h₁ : zeroes = 0 := sorry
  have h₂ : ones = 1 := sorry
  exact sorry

end min_steps_to_no_zeros_l424_424475


namespace hoseok_wire_length_l424_424708

theorem hoseok_wire_length (side_length : ℕ) (equilateral : Prop) (leftover_wire : ℕ) (total_wire : ℕ)  
  (eq_side : side_length = 19) (eq_leftover : leftover_wire = 15) 
  (eq_equilateral : equilateral) : total_wire = 72 :=
sorry

end hoseok_wire_length_l424_424708


namespace distinct_tasty_primes_l424_424367

-- Define the problem statement
theorem distinct_tasty_primes (p : ℕ) (hp : prime p) (hp_ge_5 : 5 ≤ p) :
  ∃ (q1 q2 : ℕ), q1 ≠ q2 ∧ prime q1 ∧ prime q2 ∧ 1 < q1 ∧ q1 < p - 1 ∧ 1 < q2 ∧ q2 < p - 1 ∧ 
    (q1^(p - 1) % p^2 ≠ 1) ∧ (q2^(p - 1) % p^2 ≠ 1) :=
sorry

end distinct_tasty_primes_l424_424367


namespace length_RS_l424_424373

theorem length_RS (AB AC BC : ℝ) (h1 : AB = 2024) (h2 : AC = 2023) (h3 : BC = 2022) :
  ∃ (m n : ℕ) (h_rel_prime : Nat.coprime m n), 
  let RS := (m : ℚ) / n 
  ∧ RS = 1 / 4048
  ∧ m + n = 4049 := 
by
  have : m = 1 := sorry
  have : n = 4048 := sorry
  exists m n
  sorry 

end length_RS_l424_424373


namespace problem_statement_l424_424675

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end problem_statement_l424_424675


namespace cottage_pie_ground_mince_amount_l424_424101

theorem cottage_pie_ground_mince_amount :
  ∃ (x : ℕ), let lasagna_count := 100,
                 lasagna_mince := 2,
                 total_mince := 500,
                 cottage_pie_count := 100,
                 mince_for_lasagna := lasagna_count * lasagna_mince,
                 mince_for_cottage_pies := total_mince - mince_for_lasagna,
                 x := mince_for_cottage_pies / cottage_pie_count
             in x = 3 :=
by
  sorry

end cottage_pie_ground_mince_amount_l424_424101


namespace complex_point_in_first_quadrant_l424_424845

-- Definitions
def z := (2 * complex.I) / (1 + complex.I)

-- Statement
theorem complex_point_in_first_quadrant (z : ℂ) (hz : z = (2 * complex.I) / (1 + complex.I)) : (1 : ℝ) > 0 ∧ (1 : ℝ) > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l424_424845


namespace nine_digit_flippable_numbers_l424_424508

open Nat

theorem nine_digit_flippable_numbers : 
  let unchanged_digits : List ℕ := [0, 1, 8]
  let swap_digits : List (ℕ × ℕ) := [(6, 9), (9, 6)]
  ∃ (n : ℕ), n = 1500 ∧
  (∀ x : ℕ, x.dragDigits 9 → 
    let digits := x.digits 10
    digits.length = 9 ∧
    (digits[0] = digits[8] ∧ (digits[0] ∈ unchanged_digits ∨ (digits[0], digits[8]) ∈ swap_digits)) ∧
    (digits[1] = digits[7] ∧ (digits[1] ∈ unchanged_digits ∨ (digits[1], digits[7]) ∈ swap_digits)) ∧
    (digits[2] = digits[6] ∧ (digits[2] ∈ unchanged_digits ∨ (digits[2], digits[6]) ∈ swap_digits)) ∧
    (digits[3] = digits[5] ∧ (digits[3] ∈ unchanged_digits ∨ (digits[3], digits[5]) ∈ swap_digits)) ∧
    (digits[4] ∈ unchanged_digits)) := sorry

end nine_digit_flippable_numbers_l424_424508


namespace sum_of_number_is_8_l424_424010

theorem sum_of_number_is_8 (x v : ℝ) (h1 : 0.75 * x + 2 = v) (h2 : x = 8.0) : v = 8.0 :=
by
  sorry

end sum_of_number_is_8_l424_424010


namespace change_received_l424_424358

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end change_received_l424_424358


namespace problem_statement_l424_424756

noncomputable def M : ℕ :=
  let nums := {n | n ≤ 1500 ∧ let s := n.binary_repr in s.is_palindrome ∧ s.count '1' > s.count '0'} in
  nums.to_finset.card

theorem problem_statement : M % 1000 = 16 :=
  sorry

end problem_statement_l424_424756


namespace num_pairs_nat_nums_eq_l424_424271

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424271


namespace product_of_solutions_l424_424311

theorem product_of_solutions :
  ∀ (x : ℝ), (abs ((18 / x) + 4) = 3) → 
  (x = -18 ∨ x = -18 / 7) →
  ((-18) * (-18 / 7) = 324 / 7) :=
begin
  sorry
end

end product_of_solutions_l424_424311


namespace red_candies_difference_l424_424866

def jar1_ratio_red : ℕ := 7
def jar1_ratio_yellow : ℕ := 3
def jar2_ratio_red : ℕ := 5
def jar2_ratio_yellow : ℕ := 4
def total_yellow : ℕ := 108

theorem red_candies_difference :
  ∀ (x y : ℚ), jar1_ratio_yellow * x + jar2_ratio_yellow * y = total_yellow ∧ jar1_ratio_red + jar1_ratio_yellow = jar2_ratio_red + jar2_ratio_yellow → 10 * x = 9 * y → 7 * x - 5 * y = 21 := 
by sorry

end red_candies_difference_l424_424866


namespace root_of_quadratic_eq_satisfies_value_l424_424614

theorem root_of_quadratic_eq_satisfies_value :
  let x := (-12 - Real.sqrt 400) / 15 in
  ∃ v : ℝ, (3*x^2 + 12*x + v = 0) ↔ (v = 704 / 75) :=
by
  let x := (-12 - Real.sqrt 400) / 15
  use (704 / 75)
  sorry

end root_of_quadratic_eq_satisfies_value_l424_424614


namespace count_pairs_satisfying_condition_l424_424297

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424297


namespace park_area_l424_424463

-- Definitions for the conditions
def length (breadth : ℕ) : ℕ := 4 * breadth
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the proof problem
theorem park_area (breadth : ℕ) (h1 : perimeter (length breadth) breadth = 1600) : 
  let len := length breadth
  len * breadth = 102400 := 
by 
  sorry

end park_area_l424_424463


namespace count_valid_pairs_l424_424259

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424259


namespace proof_ellipse_eq_l424_424969

def ellipse : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 9 * p.2^2 = 9}
def equilateral_triangle (A B C : ℝ × ℝ) : Prop := 
  A.1 = 0 ∧ A.2 = 1 ∧
  let l := (B.1 - A.1)^2 + (B.2 - A.2)^2 in
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = l ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = l ∧
  C.1 = -B.1 ∧ B.2 = C.2

noncomputable def side_length_sq {A B C : ℝ × ℝ} (h₁ : equilateral_triangle A B C) : ℚ :=
  let l := (B.1 - A.1)^2 + (B.2 - A.2)^2 in
  ⟨l, 1⟩

theorem proof_ellipse_eq (p q : ℕ) (h_rel_prime : nat.coprime p q) :
  (∃ A B C : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ C ∈ ellipse ∧ equilateral_triangle A B C ∧ side_length_sq equilateral_triangle = ⟨p, q⟩) → 
  p + q = 601 :=
by
  sorry

end proof_ellipse_eq_l424_424969


namespace hitting_polynomial_R5_l424_424155

/-- Define the hitting polynomial of a 5x5 chessboard with specified forbidden positions.
    The hitting polynomial E(t) is given as: 20 + 39t + 38t^2 + 16t^3 + 6t^4 + t^5. -/
theorem hitting_polynomial_R5 (t : ℤ) (R5 : set (ℕ × ℕ)) : 
  (E : ℤ[X]) = 20 + 39 * t + 38 * t^2 + 16 * t^3 + 6 * t^4 + t^5 := 
by
  sorry

end hitting_polynomial_R5_l424_424155


namespace triangle_inequality_problem_l424_424682

theorem triangle_inequality_problem
  (x : ℝ)
  (side1 side2 : ℝ)
  (h1 : side1 = 5)
  (h2 : side2 = 7)
  (h3 : x = 10) :
  (2 < x ∧ x < 12) := 
by
  sorry

end triangle_inequality_problem_l424_424682


namespace one_integer_segment_from_D_to_EF_l424_424100

-- Defining the right triangle with its sides
def DE := 15
def EF := 36
def DF := Real.sqrt (DE^2 + EF^2)

-- Define what it means to be an integer segment from D to a point on EF
def is_integer_segment (length : ℝ) : Prop := ∃ n : ℤ, (n : ℝ) = length

-- Main theorem statement
theorem one_integer_segment_from_D_to_EF :
  ∃! length : ℝ, is_integer_segment length ∧ length = DE :=
sorry

end one_integer_segment_from_D_to_EF_l424_424100


namespace price_of_shoes_on_Monday_l424_424782

noncomputable def priceOnThursday : ℝ := 50

noncomputable def increasedPriceOnFriday : ℝ := priceOnThursday * 1.2

noncomputable def discountedPriceOnMonday : ℝ := increasedPriceOnFriday * 0.85

noncomputable def finalPriceOnMonday : ℝ := discountedPriceOnMonday * 1.05

theorem price_of_shoes_on_Monday :
  finalPriceOnMonday = 53.55 :=
by
  sorry

end price_of_shoes_on_Monday_l424_424782


namespace walk_distance_l424_424027

def total_distance := 36
def speed_increase := 1.25

variables {v t D_Q D_P : ℝ} (hv : v > 0)

theorem walk_distance (hv : v > 0) (H1 : D_Q + D_P = total_distance) (H2 : D_Q = v * t) (H3 : D_P = speed_increase * v * t) :
  D_P = 20 := 
  sorry

end walk_distance_l424_424027


namespace average_of_hidden_primes_l424_424129

theorem average_of_hidden_primes (p₁ p₂ : ℕ) (h₁ : Nat.Prime p₁) (h₂ : Nat.Prime p₂) (h₃ : p₁ + 37 = p₂ + 53) : 
  (p₁ + p₂) / 2 = 11 := 
by
  sorry

end average_of_hidden_primes_l424_424129


namespace sequence_first_two_elements_l424_424493

theorem sequence_first_two_elements (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) = a (n + 1) / a n)
  (h2 : ∏ i in finset.range 40, a i = 8)
  (h3 : ∏ i in finset.range 80, a i = 8) :
  a 0 = 2 ∧ a 1 = 4 :=
by
  sorry

end sequence_first_two_elements_l424_424493


namespace blake_spent_60_on_mangoes_l424_424981

def spent_on_oranges : ℕ := 40
def spent_on_apples : ℕ := 50
def initial_amount : ℕ := 300
def change : ℕ := 150
def total_spent := initial_amount - change
def total_spent_on_fruits := spent_on_oranges + spent_on_apples
def spending_on_mangoes := total_spent - total_spent_on_fruits

theorem blake_spent_60_on_mangoes : spending_on_mangoes = 60 := 
by
  -- The proof will go here
  sorry

end blake_spent_60_on_mangoes_l424_424981


namespace initial_number_of_friends_l424_424481

theorem initial_number_of_friends (X : ℕ) (H : 3 * (X - 3) = 15) : X = 8 :=
by
  sorry

end initial_number_of_friends_l424_424481


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424279

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424279


namespace platform_length_l424_424054

theorem platform_length (length_train : ℝ) (speed_train_kmph : ℝ) (time_sec : ℝ) (length_platform : ℝ) :
  length_train = 1020 → speed_train_kmph = 102 → time_sec = 50 →
  length_platform = (speed_train_kmph * 1000 / 3600) * time_sec - length_train :=
by
  intros
  sorry

end platform_length_l424_424054


namespace problem_part1_problem_part2_l424_424698

noncomputable def f (x : ℝ) (m : ℝ) := Real.sqrt (|x + 2| + |x - 4| - m)

theorem problem_part1 (m : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 := 
by
  sorry

theorem problem_part2 (a b : ℕ) (n : ℝ) (h1 : (0 < a) ∧ (0 < b)) (h2 : n = 6) 
  (h3 : (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = n) : 
  ∃ (value : ℝ), 4 * a + 7 * b = 3 / 2 := 
by
  sorry

end problem_part1_problem_part2_l424_424698


namespace count_increasing_numbers_l424_424176

-- Define the set of digits we are concerned with
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a natural number type representing numbers with increasing digits
def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits i < n.digits j

-- Define the set of natural numbers with increasing digits and at least two digits
def increasing_numbers : set ℕ :=
  {n | increasing_digits n ∧ 10 ≤ n ∧ n ≤ 987654321}

-- Define the theorem to be proved
theorem count_increasing_numbers : set.card increasing_numbers = 502 :=
by sorry

end count_increasing_numbers_l424_424176


namespace wrapping_paper_area_l424_424081

variable {l w h : ℝ}

theorem wrapping_paper_area (hl : 0 < l) (hw : 0 < w) (hh : 0 < h) :
  (4 * l * h + 2 * l * h + 2 * w * h) = 6 * l * h + 2 * w * h :=
  sorry

end wrapping_paper_area_l424_424081


namespace gcd_372_684_is_12_l424_424447

/-- Define the gcd function -/
def gcd : ℕ → ℕ → ℕ
| 0, b := b
| a, 0 := a
| a, b :=
  if a > b then
    gcd (a % b) b
  else
    gcd a (b % a)

/-- Theorem statement to prove gcd of 372 and 684 is 12 -/
theorem gcd_372_684_is_12 : gcd 372 684 = 12 := by
  sorry

end gcd_372_684_is_12_l424_424447


namespace proof_problem_l424_424684

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := -2 / 3
noncomputable def c : ℝ := 2 -- We pick c = 2 since c^2 = 4 irrespective of the sign of c.

theorem proof_problem : a + b + c^2 = 16 / 3 :=
by
  -- a = 2, b = -2/3, c^2 = 4
  have ha : a = 2 := rfl
  have hb : b = -2 / 3 := rfl
  have hc : c^2 = 4 := by
    simp [c]
  calc
    a + b + c^2
      = 2 + -2 / 3 + 4 : by simp [ha, hb, hc]
  ... = 6 - 2 / 3 : by simp
  ... = 6 - 2 / 3 : by
    have : (6 : ℝ) = 18 / 3 := by norm_num
    rw [this]
  ... = 16 / 3 : by
    linarith

end proof_problem_l424_424684


namespace slant_asymptote_sum_l424_424106

theorem slant_asymptote_sum:
  (∃ (m b : ℝ), ∀ x : ℝ, x ≠ 2 → abs (x - 2) > 1 → 
  (((3 * x^2 + 4 * x - 5) / (x - 2)) - (m * x + b) = 15 / (x - 2)) ∧ (m + b = 13)) :=
begin
  sorry,
end

end slant_asymptote_sum_l424_424106


namespace log_expression_simplifies_l424_424813

theorem log_expression_simplifies (x y : ℝ) (hx : x > 0) (hy : y > 0) (hyn : y ≠ 1) : 
  log (y^6) x * log (x^5) (y^2) * log (y^4) (x^3) * log (x^3) (y^4) * log (y^2) (x^5) = (1/6 : ℝ) * log y x :=
by sorry

end log_expression_simplifies_l424_424813


namespace hyperbola_eccentricity_l424_424665

theorem hyperbola_eccentricity (a b c e : ℝ) (h1 : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1)
  (h2 : ∃ (F1 F2 : ℝ × ℝ), F1 = (-c, 0) ∧ F2 = (c, 0) ∧ |OF2| = c)
  (h3 : ∃ (circle_center : ℝ × ℝ) (radius : ℝ), circle_center = (c, 0) ∧ radius = |OF2|)
  (h4 : ∃ (Q : ℝ × ℝ), tangent_from F1 Q = Q ∧ segment F1 Q is_bisected_by asymptote):
  e = 2 := 
sorry

end hyperbola_eccentricity_l424_424665


namespace total_output_equal_at_20_l424_424085

noncomputable def total_output_A (x : ℕ) : ℕ :=
  200 + 20 * x

noncomputable def total_output_B (x : ℕ) : ℕ :=
  30 * x

theorem total_output_equal_at_20 :
  total_output_A 20 = total_output_B 20 :=
by
  sorry

end total_output_equal_at_20_l424_424085


namespace asymptotes_equation_l424_424137

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  x^2 / 64 - y^2 / 36 = 1

theorem asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y → (y = (3/4) * x ∨ y = - (3/4) * x) :=
by
  intro x y
  intro h
  sorry

end asymptotes_equation_l424_424137


namespace length_AC_eq_length_PQ_l424_424462

variables {Point : Type} [MetricSpace Point]

structure Circle (Point : Type) :=
(center : Point)
(radius : ℝ)

def lies_on_circle (c : Circle Point) (P : Point) : Prop :=
dist c.center P = c.radius

variables (A B C D E P Q : Point) (c : Circle Point)

variables (h1 : lies_on_circle c A)
variables (h2 : lies_on_circle c B)
variables (h3 : lies_on_circle c C)
variables (h4 : lies_on_circle c D)
variables (h5 : lies_on_circle c E)
variables (h6 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E)
variables (h7 : parallel (line_through A B) (line_through E C))
variables (h8 : parallel (line_through A C) (line_through E D))
variables (h9 : tangent_line_through E c P (line_through A B))
variables (h10 : intersect (line_through B D) (line_through E C) Q)

theorem length_AC_eq_length_PQ : dist A C = dist P Q :=
sorry

end length_AC_eq_length_PQ_l424_424462


namespace molly_xanthia_reading_time_difference_in_minutes_l424_424912

theorem molly_xanthia_reading_time_difference_in_minutes :
  ∀ (xanthia_rate molly_rate : ℝ) (book_pages : ℕ) 
    (xanthia_rate_pos : xanthia_rate > 0)
    (molly_rate_pos : molly_rate > 0)
    (xanthia_rate : xanthia_rate = 75) 
    (molly_rate : molly_rate = 45) 
    (book_pages : book_pages = 270),
  ( (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 144) := 
by
  intros xanthia_rate molly_rate book_pages xanthia_rate_pos molly_rate_pos xanthia_rate molly_rate book_pages 
  sorry

end molly_xanthia_reading_time_difference_in_minutes_l424_424912


namespace symmetrical_parabola_eq_l424_424438

/-- 
  Given a parabola y = (x-1)^2 + 3, prove that its symmetrical parabola 
  about the x-axis is y = -(x-1)^2 - 3.
-/
theorem symmetrical_parabola_eq (x : ℝ) : 
  (x-1)^2 + 3 = -(x-1)^2 - 3 ↔ y = -(x-1)^2 - 3 := 
sorry

end symmetrical_parabola_eq_l424_424438


namespace tan_half_sum_sq_l424_424713

theorem tan_half_sum_sq (a b : ℝ) : 
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 → 
  ∃ (x : ℝ), (x = (Real.tan (a / 2) + Real.tan (b / 2))^2) ∧ (x = 6 ∨ x = 26) := 
by
  intro h
  sorry

end tan_half_sum_sq_l424_424713


namespace ratio_of_shaded_to_white_area_l424_424891

/-- Given a figure where the vertices of all the squares, except for the largest one, 
    are located at the midpoints of the corresponding sides, prove that the ratio 
    of the area of the shaded part to the white part is 5:3. -/
theorem ratio_of_shaded_to_white_area :
  (let total_shaded := 20 in
  let total_white := 12 in
  total_shaded / total_white) = (5 / 3) :=
by {
  sorry
}

end ratio_of_shaded_to_white_area_l424_424891


namespace shaded_to_unshaded_area_ratio_l424_424899

-- Define a structure for a large square with its side length.
structure Square :=
  (side_length : ℝ)

-- Define the condition that vertices of smaller squares are at the midpoints.
def midpoint_vertex_property (sq : Square) : Prop :=
  ∀ (sub_sq : Square), sub_sq.side_length = sq.side_length / 2

-- Define areas of regions within one quadrant.
def shaded_area (sq : Square) : ℝ :=
  5 * (sq.side_length / 2) * (sq.side_length / 2) / 2

def unshaded_area (sq : Square) : ℝ :=
  3 * (sq.side_length / 2) * (sq.side_length / 2) / 2

-- Define the ratio of the shaded area to the unshaded area.
def area_ratio (sq : Square) : ℝ :=
  shaded_area sq / unshaded_area sq

-- The theorem stating that the area ratio is 5:3.
theorem shaded_to_unshaded_area_ratio (sq : Square) (h : midpoint_vertex_property sq) :
  area_ratio sq = 5 / 3 := 
  sorry

end shaded_to_unshaded_area_ratio_l424_424899


namespace range_of_k_l424_424237

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, 0 ≤ x → (x - 2) * Real.exp(x) ≥ k * (x^2 - 2*x - 1)) ↔ (2 ≤ k ∧ k ≤ Real.exp(3) / 2) :=
by
  sorry

end range_of_k_l424_424237


namespace inverse_function_application_l424_424801

def f (x : ℝ) : ℝ := 3 * x + 2

def f_inv (y : ℝ) : ℝ := (y - 2) / 3

theorem inverse_function_application :
  f_inv (f_inv 14) = 2 / 3 :=
by
  sorry

end inverse_function_application_l424_424801


namespace m_range_l424_424241

theorem m_range (m : ℝ) :
  (∀ x : ℝ, 1 < x → 2 * x + m + 2 / (x - 1) > 0) ↔ m > -6 :=
by
  -- The proof will be provided later
  sorry

end m_range_l424_424241


namespace min_colors_needed_to_paint_cube_l424_424787

/-!
Pedro wants to paint a cube-shaped box in such a way that faces sharing a common edge are painted 
in different colors. Prove that the minimum number of colors needed to paint the box in this 
manner is 3.
-/

def cube_faces : Finset (Fin 6) := {0, 1, 2, 3, 4, 5}

def shares_edge (f1 f2 : Fin 6) : Prop :=
  (f1 == 0 ∧ (f2 == 1 ∨ f2 == 2 ∨ f2 == 3 ∨ f2 == 4)) ∨
  (f1 == 1 ∧ (f2 == 0 ∨ f2 == 2 ∨ f2 == 3 ∨ f2 == 5)) ∨
  (f1 == 2 ∧ (f2 == 0 ∨ f2 == 1 ∨ f2 == 4 ∨ f2 == 5)) ∨
  (f1 == 3 ∧ (f2 == 0 ∨ f2 == 1 ∨ f2 == 4 ∨ f2 == 5)) ∨
  (f1 == 4 ∧ (f2 == 0 ∨ f2 == 2 ∨ f2 == 3 ∨ f2 == 5)) ∨
  (f1 == 5 ∧ (f2 == 1 ∨ f2 == 2 ∨ f2 == 3 ∨ f2 == 4))

def valid_coloring (coloring : Fin 6 → ℕ) : Prop :=
  ∀ f1 f2, f1 ∈ cube_faces ∧ f2 ∈ cube_faces ∧ shares_edge f1 f2 → coloring f1 ≠ coloring f2

theorem min_colors_needed_to_paint_cube : ∃ (c : Fin 6 → ℕ), valid_coloring c ∧ (∀ (c' : Fin 6 → ℕ), valid_coloring c' → finset.card (finset.image c' cube_faces) ≥ 3) :=
by
  sorry

end min_colors_needed_to_paint_cube_l424_424787


namespace angle_between_a_b_is_60_degrees_l424_424246

open Real

def vector_a : ℝ × ℝ × ℝ := (0, 1, 1)
def vector_b : ℝ × ℝ × ℝ := (1, 0, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ :=
  acos (dot_product u v / (magnitude u * magnitude v))

theorem angle_between_a_b_is_60_degrees : 
  angle_between_vectors vector_a vector_b = π / 3 :=
sorry

end angle_between_a_b_is_60_degrees_l424_424246


namespace A_has_winning_strategy_l424_424019

/-- A proposition that states: given 9 balls, two players A and B take turns drawing 1 to 3 balls,
    and player A goes first, A has a winning strategy (i.e., A will always draw the last ball). -/
theorem A_has_winning_strategy : 
  ∃ strategy_A : ℕ → ℕ, ∃ strategy_B : ℕ → ℕ, (∀ n, n ≥ 0 → n ≤ 9 → 
    (n = 9 → strategy_A(n) < 4) ∧ 
    (n = 9 - strategy_A(n) → 1 ≤ strategy_B(n) ∧ strategy_B(n) ≤ 3) ∧ 
    (n = 9 - strategy_A(n) - strategy_B(n) → 1 ≤ strategy_A(n) ∧ strategy_A(n) ≤ 3)
  ) ∧ 
  (n = 9 - strategy_A(n) - strategy_B(n) - strategy_A(n) - strategy_B(n) … = 0) → 
  A wins :=
begin
  sorry
end

end A_has_winning_strategy_l424_424019


namespace strictly_increasing_seqs_count_l424_424169

theorem strictly_increasing_seqs_count : 
  ∑ k in (finset.range 9).filter (λ k, k ≥ 2), nat.choose 9 k = 502 := by
  sorry

end strictly_increasing_seqs_count_l424_424169


namespace total_weight_of_full_bucket_l424_424048

variable (a b x y : ℝ)

def bucket_weights :=
  (x + (1/3) * y = a) → (x + (3/4) * y = b) → (x + y = (16/5) * b - (11/5) * a)

theorem total_weight_of_full_bucket :
  bucket_weights a b x y :=
by
  intro h1 h2
  -- proof goes here, can be omitted as per instructions
  sorry

end total_weight_of_full_bucket_l424_424048


namespace diamonds_in_G10_l424_424185

theorem diamonds_in_G10 :
  let G : ℕ → ℕ := λ n, if n = 1 then 1
                       else if n = 2 then 6
                       else if n = 3 then 22
                       else G (n-1) + 4 * (n-1)^2
  in G 10 = 1142 :=
by
  let G : ℕ → ℕ := λ n, if n = 1 then 1
                        else if n = 2 then 6
                        else if n = 3 then 22
                        else G (n-1) + 4 * (n-1)^2
  sorry

end diamonds_in_G10_l424_424185


namespace length_of_train_is_150_l424_424918

-- Definition of the speed in km/hr and the time in seconds
def speed_km_hr : ℝ := 90
def time_sec : ℝ := 6

-- Conversion factor from km/hr to m/s
def km_hr_to_m_s (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Speed in m/s
def speed_m_s : ℝ := km_hr_to_m_s speed_km_hr

-- Define the length of the train as Distance = Speed * Time
def length_of_train : ℝ := speed_m_s * time_sec

-- Our goal is to prove that the length of the train is 150 meters
theorem length_of_train_is_150 : length_of_train = 150 := by
  sorry

end length_of_train_is_150_l424_424918


namespace height_formula_correct_l424_424728

noncomputable def height_from_third_vertex (α β R : ℝ) : ℝ :=
  2 * R * sin α * sin β

theorem height_formula_correct (α β R : ℝ) : 
  height_from_third_vertex α β R = 2 * R * sin α * sin β :=
by
  -- To be proven
  sorry

end height_formula_correct_l424_424728


namespace increasing_interval_of_f_l424_424239

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 15 * x ^ 2 + 36 * x - 24

theorem increasing_interval_of_f : (∀ x : ℝ, x = 2 → deriv f x = 0) → ∀ x : ℝ, 3 < x → 0 < deriv f x :=
by
  intro h x hx
  -- We know that the function has an extreme value at x = 2
  have : deriv f 2 = 0 := h 2 rfl
  -- Require to prove the function is increasing in interval (3, +∞)
  sorry

end increasing_interval_of_f_l424_424239


namespace ann_trip_longer_than_mary_l424_424394

-- Define constants for conditions
def mary_hill_length : ℕ := 630
def mary_speed : ℕ := 90
def ann_hill_length : ℕ := 800
def ann_speed : ℕ := 40

-- Define a theorem to express the question and correct answer
theorem ann_trip_longer_than_mary : 
  (ann_hill_length / ann_speed - mary_hill_length / mary_speed) = 13 :=
by
  -- Now insert sorry to leave the proof unfinished
  sorry

end ann_trip_longer_than_mary_l424_424394


namespace common_solution_ys_l424_424147

theorem common_solution_ys : 
  {y : ℝ | ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 + 2*y = 7} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
sorry

end common_solution_ys_l424_424147


namespace cone_frustum_volume_l424_424228

theorem cone_frustum_volume (h r l : ℝ) (hlateral : π * r * l = 2 * π)
  (hl2r : l = 2 * r) (hheight : h = sqrt (l^2 - r^2)) :
  (1 / 3) * π * r^2 * h = sqrt 3 * π / 3 ∧
  ((7 / 8) * ((1 / 3) * π * r^2 * h)) = (7 * sqrt 3 * π) / 24 :=
by
  sorry

end cone_frustum_volume_l424_424228


namespace definite_integral_ln2_l424_424857

theorem definite_integral_ln2 :
  ∫ x in 0..1, (1 / (1 + x)) = Real.log 2 :=
by
  sorry

end definite_integral_ln2_l424_424857


namespace count_valid_pairs_l424_424260

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424260


namespace pattern_value_l424_424824

def alphabet_indices : List ℕ :=
  [1, 0, 2, 0, 1, 0, -1, 0, -2, 0, -1, 0, 1, 0, 2, 0, 1, 0, -1, 0, -2, 0, -1, 0, 1, 0]

def letter_value (c : Char) : Int :=
  let index := (c.toNat - 'A'.toNat) % alphabet_indices.length
  alphabet_indices.getOrElse index 0

def word_value (word : String) : Int :=
  word.foldl (λ acc c => acc + letter_value c) 0

theorem pattern_value :
  word_value "PATTERN" = 3 := by
  -- proof will go here
  sorry

end pattern_value_l424_424824


namespace incorrect_statement_is_3_l424_424910

-- Definitions related to conditions
def is_chord (c : ℝ) (r : ℝ) : Prop := c < 2 * r
def is_diameter (c : ℝ) (r : ℝ) : Prop := c = 2 * r
def is_arc (s : ℝ) (π : ℝ) : Prop := s <= π * r
def is_concentric (c1 c2 r1 r2 : ℝ) : Prop := c1 = c2 ∧ r1 = r2

-- Statements to be evaluated
def statement_1 (c r : ℝ) : Prop := is_chord c r → is_diameter c r
def statement_2 (s π : ℝ) : Prop := is_arc s π
def statement_3 (c r : ℝ) : Prop := is_chord c r ∧ is_diameter c r
def statement_4 (c1 c2 r1 r2 : ℝ) : Prop := is_concentric c1 c2 r1 r2

-- Problem: Which statement is incorrect?
def incorrect_statement := statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ statement_4
theorem incorrect_statement_is_3 (c r : ℝ) : incorrect_statement → statement_3 :=
sorry

end incorrect_statement_is_3_l424_424910


namespace ratio_of_x_intercepts_l424_424877

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l424_424877


namespace max_colored_cells_l424_424606

-- Define the grid size
def grid_size := 10

-- Define the condition for a valid coloring
def valid_coloring (colored_cells : Finset (ℕ × ℕ)) : Prop :=
  ∀ (c ∈ colored_cells),
    (∃ (d ∈ colored_cells), d ≠ c ∧ (d.1 ≥ c.1 ∧ d.2 ≥ c.2)) ∨
    (∀ (d ∈ colored_cells), d = c ∨ d.1 < c.1 ∨ d.2 < c.2)

-- Define the main theorem which states the maximum number of cells that can be colored is 15
theorem max_colored_cells : ∃ (colored_cells : Finset (ℕ × ℕ)), 
  valid_coloring colored_cells ∧ colored_cells.card = 15 := 
sorry

end max_colored_cells_l424_424606


namespace value_of_f_at_5_l424_424443

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_f_at_5 : f 5 = 15 := 
by {
  sorry
}

end value_of_f_at_5_l424_424443


namespace diagonals_perpendicular_l424_424205

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D : V)

-- Defining the main theorem
theorem diagonals_perpendicular 
  (h : dist A B ^ 2 + dist C D ^ 2 = dist A D ^ 2 + dist B C ^ 2) : 
  ⟪A - C, B - D⟫ = 0 :=
sorry

end diagonals_perpendicular_l424_424205


namespace ellipse_line_intersection_length_line_equation_area_ratio_l424_424233

noncomputable def ellipse_contains (P : ℝ × ℝ) : Prop :=
  (P.1)^2/2 + P.2^2 = 1

noncomputable def line_l (P : ℝ × ℝ) : Prop :=
  P.2 = 2 * P.1 - 2

noncomputable def intersection_points (C D : ℝ × ℝ) : Prop :=
  ellipse_contains C ∧ line_l C ∧ ellipse_contains D ∧ line_l D

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def point_E (P : ℝ × ℝ) : Prop :=
  P.1 = 2 ∧ P.2 = 2 * 2 - 2

noncomputable def area_ratio_condition (O E C : ℝ × ℝ) : Prop :=
  3 * (Real.sqrt ((O.1 - E.1)^2 + (O.2 - E.2)^2)) = 1 * (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2))

theorem ellipse_line_intersection_length (C D : ℝ × ℝ) (E : ℝ × ℝ):
  intersection_points C D →
  point_E E →
  distance C D = 10 * Real.sqrt 2 / 9 := 
sorry

theorem line_equation_area_ratio (O E C : ℝ × ℝ):
  area_ratio_condition O E C →
  (∀ x, O = (0,0) ∧ E = (2,2)) →
  (∀ k, C = (k + 1, k - 1)) →
  (∀ x, x = x - 1 ∨ x = - x + 1) :=
sorry

end ellipse_line_intersection_length_line_equation_area_ratio_l424_424233


namespace john_needs_at_least_12_bottles_l424_424748

noncomputable def min_bottles (fl_oz_needed : ℝ) (ml_per_bottle : ℝ) (fl_oz_per_liter : ℝ) : ℕ :=
  let liters_needed := fl_oz_needed / fl_oz_per_liter
  let ml_needed := liters_needed * 1000
  let bottles_needed := ml_needed / ml_per_bottle
  ⌈bottles_needed⌉.to_nat

theorem john_needs_at_least_12_bottles:
  min_bottles 60 150 34 = 12 :=
by
  sorry

end john_needs_at_least_12_bottles_l424_424748


namespace max_value_of_k_l424_424628

theorem max_value_of_k (n : ℕ) (k : ℕ) (h : 3^11 = k * (2 * n + k + 1) / 2) : k = 486 :=
sorry

end max_value_of_k_l424_424628


namespace line_tangent_to_parabola_proof_l424_424829

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l424_424829


namespace hyperbola_eccentricity_l424_424216

variables {a b c : ℝ} (P F1 F2 : ℝ → ℝ)

def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2 - y^2 / b^2 = 1)
def asymptote (x y a b : ℝ) : Prop := (y = b * x / a)
def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def distance_between_foci (a b : ℝ) : ℝ := (a^2 + b^2)^(1/2)

theorem hyperbola_eccentricity (a : ℝ) (h_pos : a > 0) (h : b = 2 * a) :
  eccentricity (distance_between_foci a b) a = Real.sqrt 5 := by
  unfold distance_between_foci
  simp [h]
  sorry

end hyperbola_eccentricity_l424_424216


namespace product_of_solutions_l424_424310

theorem product_of_solutions :
  ∀ (x : ℝ), (abs ((18 / x) + 4) = 3) → 
  (x = -18 ∨ x = -18 / 7) →
  ((-18) * (-18 / 7) = 324 / 7) :=
begin
  sorry
end

end product_of_solutions_l424_424310


namespace arrangements_no_adjacent_dances_arrangements_alternating_order_l424_424972

-- Part (1)
theorem arrangements_no_adjacent_dances (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 43200 := 
by sorry

-- Part (2)
theorem arrangements_alternating_order (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 2880 := 
by sorry

end arrangements_no_adjacent_dances_arrangements_alternating_order_l424_424972


namespace integral_sin_zero_l424_424527

theorem integral_sin_zero : ∫ x in -2..2, sin x = 0 := by
  sorry

end integral_sin_zero_l424_424527


namespace seating_arrangements_total_l424_424742

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

end seating_arrangements_total_l424_424742


namespace closest_vector_l424_424632

theorem closest_vector 
  (s : ℝ)
  (u b d : ℝ × ℝ × ℝ)
  (h₁ : u = (3, -2, 4) + s • (6, 4, 2))
  (h₂ : b = (1, 7, 6))
  (hdir : d = (6, 4, 2))
  (h₃ : (u - b) = (2 + 6 * s, -9 + 4 * s, -2 + 2 * s)) :
  ((2 + 6 * s) * 6 + (-9 + 4 * s) * 4 + (-2 + 2 * s) * 2) = 0 →
  s = 1 / 2 :=
by
  -- Skipping the proof, adding sorry
  sorry

end closest_vector_l424_424632


namespace sum_least_n_divisible_by_5_l424_424636

theorem sum_least_n_divisible_by_5 : 
  ∀ (n : ℕ), n ≥ 2 ∧ (∃ (t : ℕ), t = (n - 1) * n * (n + 1) * (3 * n + 2) / 24 ∧ t % 5 = 0) →
    let ns := [2, 3, 4, 5, 6, 7, 8, 9, 10] in
    (∑ x in (ns.filter (λ m, (m - 1) * m * (m + 1) * (3 * m + 2) ≡ 0 [MOD 5])).take 6, x) = 21 :=
by
  sorry

end sum_least_n_divisible_by_5_l424_424636


namespace F_mean_identity_l424_424380

noncomputable def F (n r : ℕ) : ℚ :=
  (∑ k in finset.range (n - r + 1), (k + 1) * (nat.choose (n - (k + 1)) (r - 1))) / nat.choose n r

theorem F_mean_identity (n r : ℕ) (h₀ : 1 ≤ r) (h₁ : r ≤ n) :
  F n r = (n + 1) / (r + 1) := by
  sorry

end F_mean_identity_l424_424380


namespace distance_covered_by_train_l424_424573

-- Define the average speed and the total duration of the journey
def speed : ℝ := 10
def time : ℝ := 8

-- Use these definitions to state and prove the distance covered by the train
theorem distance_covered_by_train : speed * time = 80 := by
  sorry

end distance_covered_by_train_l424_424573


namespace employee_n_salary_l424_424026

theorem employee_n_salary (x : ℝ) (h : x + 1.2 * x = 583) : x = 265 := sorry

end employee_n_salary_l424_424026


namespace angle_bisector_divides_correctly_l424_424865

variables (α β : ℝ) (A B C D E K : Type)
variables (AB BC DB BE: Type)
variables (angle_ABC angle_DBE angle_AKD : ℝ)

-- Definitions according to given conditions
def isosceles_triangles := (AB = BC ∧ DB = BE)
def common_vertex := AB ∩ BC = B ∧ DB ∩ BE = B
def coplanar_opposite_sides := Opposite_sides_of_line A B D C
def intersection_point := AC ∩ DE = K
def angle_conditions := angle ABC = α ∧ angle DBE = α ∧ α < π / 2 ∧ angle AKD = β ∧ β < α

-- The proof problem
theorem angle_bisector_divides_correctly :
  isosceles_triangles ∧ common_vertex ∧ coplanar_opposite_sides ∧ intersection_point ∧ angle_conditions
  → (angle ABK / angle KBC = (α + β) / (α - β)) := 
sorry

end angle_bisector_divides_correctly_l424_424865


namespace range_of_x_l424_424997

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_l424_424997


namespace gcd_polynomial_l424_424218

-- Define the condition that b is a multiple of 528
def is_multiple_of {x y : ℕ} (h : x ∣ y) := ∃ k, y = x * k

-- Let b be such a multiple
variables (b : ℕ) (h : is_multiple_of 528 b)

-- Goal: Prove that gcd(4b^3 + 2b^2 + 5b + 66, b) = 66
theorem gcd_polynomial (h : is_multiple_of 528 b) : 
  Nat.gcd (4 * b^3 + 2 * b^2 + 5 * b + 66) b = 66 := 
sorry

end gcd_polynomial_l424_424218


namespace line_tangent_to_parabola_proof_l424_424831

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l424_424831


namespace scrabble_champions_l424_424348

theorem scrabble_champions :
  let total_champions := 10
  let men_percentage := 0.40
  let men_champions := total_champions * men_percentage
  let bearded_percentage := 0.40
  let non_bearded_percentage := 0.60

  let bearded_men_champions := men_champions * bearded_percentage
  let non_bearded_men_champions := men_champions * non_bearded_percentage

  let bearded_bald_percentage := 0.60
  let bearded_with_hair_percentage := 0.40
  let non_bearded_bald_percentage := 0.30
  let non_bearded_with_hair_percentage := 0.70

  (bearded_men_champions * bearded_bald_percentage).round = 2 ∧
  (bearded_men_champions * bearded_with_hair_percentage).round = 1 ∧
  (non_bearded_men_champions * non_bearded_bald_percentage).round = 2 ∧
  (non_bearded_men_champions * non_bearded_with_hair_percentage).round = 4 :=
by 
sorry

end scrabble_champions_l424_424348


namespace range_of_a_l424_424691

def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2*a else x^2 - 4*a*x + a

theorem range_of_a (a : ℝ) (h₁ : 0 < a ∧ a ≤ 1/2) (h₂ : 1/4 < a) : ∃ x₁ x₂ x₃ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ :=
begin
  sorry
end

end range_of_a_l424_424691


namespace fraction_product_simplified_l424_424126

theorem fraction_product_simplified :
  (1 / 3) * (4 / 7) * (9 / 11) = 12 / 77 :=
by
  -- Here, we add the proof steps
  sorry

end fraction_product_simplified_l424_424126


namespace initial_blue_balls_l424_424061

-- Define the initial conditions
variables (B : ℕ) (total_balls : ℕ := 15) (removed_blue_balls : ℕ := 3)
variable (prob_after_removal : ℚ := 1 / 3)
variable (remaining_balls : ℕ := total_balls - removed_blue_balls)
variable (remaining_blue_balls : ℕ := B - removed_blue_balls)

-- State the theorem
theorem initial_blue_balls : 
  remaining_balls = 12 → remaining_blue_balls = remaining_balls * prob_after_removal → B = 7 :=
by
  intros h1 h2
  sorry

end initial_blue_balls_l424_424061


namespace count_valid_pairs_l424_424257

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424257


namespace find_x_pos_integer_l424_424379

theorem find_x_pos_integer (x : ℕ) (h : 0 < x) (n d : ℕ)
    (h1 : n = x^2 + 4 * x + 29)
    (h2 : d = 4 * x + 9)
    (h3 : n = d * x + 13) : 
    x = 2 := 
sorry

end find_x_pos_integer_l424_424379


namespace chord_equation_l424_424626

-- Definitions and conditions
def parabola (x y : ℝ) := y^2 = 8 * x
def point_Q := (4, 1)

-- Statement to prove
theorem chord_equation :
  ∃ (m : ℝ) (c : ℝ), m = 4 ∧ c = -15 ∧
    ∀ (x y : ℝ), (parabola x y ∧ x + y = 8 ∧ y + y = 2) →
      4 * x - y = 15 :=
by
  sorry -- Proof elided

end chord_equation_l424_424626


namespace girls_boys_difference_l424_424721

variables (B G : ℕ) (x : ℕ)

-- Condition that relates boys and girls with a ratio
def ratio_condition : Prop := 3 * x = B ∧ 4 * x = G

-- Condition that the total number of students is 42
def total_students_condition : Prop := B + G = 42

-- We want to prove that the difference between the number of girls and boys is 6
theorem girls_boys_difference (h_ratio : ratio_condition B G x) (h_total : total_students_condition B G) : 
  G - B = 6 :=
sorry

end girls_boys_difference_l424_424721


namespace square_area_divided_into_equal_rectangles_l424_424568

theorem square_area_divided_into_equal_rectangles (s y : ℝ) 
  (h1: (s = 5 + 2 * y))
  (h2 : (5 * y = 0.5 * s^2)):
  (s^2 = 400) := 
begin
  sorry
end

end square_area_divided_into_equal_rectangles_l424_424568


namespace pyramid_cross_sections_areas_inequality_l424_424098

noncomputable theory

-- Define the properties and areas
variables (S : ℝ) (h1 h2 h3 : ℝ)
def S1 := (1 / 4) * S
def S2 := (1 / 2) * S
def S3 := S / (4^(1 / 3))

-- Prove the desired inequality
theorem pyramid_cross_sections_areas_inequality 
  (S_pos : 0 < S) :
  S1 S < S2 S ∧ S2 S < S3 S :=
by {
  have h1 : S1 S = (1 / 4) * S := rfl,
  have h2 : S2 S = (1 / 2) * S := rfl,
  have h3 : S3 S = S / (4^(1 / 3)) := rfl,
  have h4 : (1 / 4) * S < (1 / 2) * S, 
  { linarith [by norm_num], },
  have h5 : (1 / 2) * S < S / (4^(1 / 3)),
  { refine mul_lt_mul_of_pos_right _ S_pos,
    refine one_div_lt_one_div_of_lt _,
    norm_num,
  },
  exact ⟨h4, h5⟩,
  sorry
}

end pyramid_cross_sections_areas_inequality_l424_424098


namespace ratio_of_intercepts_l424_424871

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l424_424871


namespace repeating_decimal_sum_l424_424904

theorem repeating_decimal_sum :
  let x : ℚ := 45 / 99 in
  let simp_fraction := (5, 11) in
  simp_fraction.fst + simp_fraction.snd = 16 :=
by
  let x : ℚ := 45 / 99
  let simp_fraction := (5, 11)
  have h_gcd : Int.gcd 45 99 = 9 := by norm_num
  have h_simplify : x = simp_fraction.fst / simp_fraction.snd := by
    rw [num_denom, h_gcd]
    norm_cast
    simp
  show simp_fraction.fst + simp_fraction.snd = 16 from
    by norm_num
  simp_fraction.rfl

end repeating_decimal_sum_l424_424904


namespace percentage_full_merit_scholarship_l424_424942

theorem percentage_full_merit_scholarship (total_students : ℕ) (half_scholarships : ℕ) (no_scholarships : ℕ) (x : ℝ)
  (h1 : total_students = 300)
  (h2 : half_scholarships = 0.1 * 300)
  (h3 : no_scholarships = 255)
  (h4 : ((x / 100) * 300) + half_scholarships + no_scholarships = total_students) :
  x = 5 := 
sorry

end percentage_full_merit_scholarship_l424_424942


namespace pascal_row_12_sum_pascal_row_12_middle_l424_424324

open Nat

/-- Definition of the sum of all numbers in a given row of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ :=
  2^n

/-- Definition of the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Pascal Triangle Row 12 sum -/
theorem pascal_row_12_sum : pascal_sum 12 = 4096 :=
by
  sorry

/-- Pascal Triangle Row 12 middle number -/
theorem pascal_row_12_middle : binomial 12 6 = 924 :=
by
  sorry

end pascal_row_12_sum_pascal_row_12_middle_l424_424324


namespace count_pairs_satisfying_condition_l424_424269

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424269


namespace ef_value_l424_424588

theorem ef_value (A B C D E F : ℝ) (AB EF CD BC : ℝ) 
  (h1 : AB = 20)
  (h2 : EF = 16)
  (h3 : CD = 80)
  (h4 : BC = 100)
  (h5 : AB ∥ EF)
  (h6 : EF ∥ CD) :
  EF = 16 :=
by
  sorry

end ef_value_l424_424588


namespace largest_n_property_l424_424156

open Nat

theorem largest_n_property :
  ∃ n : ℕ, n = 10 ∧
    (∀ p : ℕ, (prime p ∧ p % 2 = 1 ∧ p < n) → prime (n - p)) ∧
    (∀ m : ℕ, (∀ p : ℕ, (prime p ∧ p % 2 = 1 ∧ p < m) → prime (m - p)) → m ≤ 10) := by
  sorry

end largest_n_property_l424_424156


namespace slope_abs_value_l424_424641

-- Define the centers of the circles and the line passing point
def centers : List (ℝ × ℝ) := [(10, 150), (15, 130), (20, 145), (25, 120)]
def radius := 5
def point_on_line := (15, 130)

-- Given question to prove
theorem slope_abs_value :
  ∃(m : ℝ), ∀(line : ℝ × ℝ → ℝ), 
  (line = λ (p : ℝ × ℝ), p.2 - m * p.1 - (130 - m * 15)) → 
  (∃(y1 y2 : ℝ), y1 = 1 ∧ y2 = -1 ∧ ((y1 ≥ 0 ∧ y2 ≤ 0) ∨ 
  (y1 ≤ 0 ∧ y2 ≥ 0))) → 
  abs m = 1 := sorry

end slope_abs_value_l424_424641


namespace meeting_time_coincides_l424_424521

variables (distance_ab : ℕ) (speed_train_a : ℕ) (start_time_train_a : ℕ) (distance_at_9am : ℕ) (speed_train_b : ℕ) (start_time_train_b : ℕ)

def total_distance_ab := 465
def train_a_speed := 60
def train_b_speed := 75
def start_time_a := 8
def start_time_b := 9
def distance_train_a_by_9am := train_a_speed * (start_time_b - start_time_a)
def remaining_distance := total_distance_ab - distance_train_a_by_9am
def relative_speed := train_a_speed + train_b_speed
def time_to_meet := remaining_distance / relative_speed

theorem meeting_time_coincides :
  time_to_meet = 3 → (start_time_b + time_to_meet = 12) :=
by
  sorry

end meeting_time_coincides_l424_424521


namespace problem_1_problem_2_problem_3_l424_424634

def pair_otimes (a b c d : ℚ) : ℚ := b * c - a * d

-- Problem (1)
theorem problem_1 : pair_otimes 5 3 (-2) 1 = -11 := 
by 
  unfold pair_otimes 
  sorry

-- Problem (2)
theorem problem_2 (x : ℚ) (h : pair_otimes 2 (3 * x - 1) 6 (x + 2) = 22) : x = 2 := 
by 
  unfold pair_otimes at h
  sorry

-- Problem (3)
theorem problem_3 (x k : ℤ) (h : pair_otimes 4 (k - 2) x (2 * x - 1) = 6) : 
  k = 8 ∨ k = 9 ∨ k = 11 ∨ k = 12 := 
by 
  unfold pair_otimes at h
  sorry

end problem_1_problem_2_problem_3_l424_424634


namespace area_of_figure1_values_of_a_for_three_solutions_l424_424247

noncomputable def figure1 (x y : ℝ) : Prop :=
  |3 * x| + |4 * y| + |48 - 3 * x - 4 * y| = 48

noncomputable def figure2 (x y a : ℝ) : Prop :=
  (x - 8)^2 + (y + 6 * Real.cos (a * Real.pi / 2))^2 = (a + 4)^2

theorem area_of_figure1 :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), figure1 a b ∧ figure1 a c ∧ figure1 b c ∧ (triangle_area a b c = 96) :=
sorry

theorem values_of_a_for_three_solutions :
  ∀ (a : ℝ), (∀ (x y : ℝ), figure1 x y ∧ figure2 x y a) ↔ (a = 6 ∨ a = -14) :=
sorry

end area_of_figure1_values_of_a_for_three_solutions_l424_424247


namespace right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l424_424967

theorem right_triangle_min_hypotenuse (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) : c ≥ 4 * Real.sqrt 2 := by
  sorry

theorem right_triangle_min_hypotenuse_achieved (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) (h_isosceles : a = b) : c = 4 * Real.sqrt 2 := by
  sorry

end right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l424_424967


namespace speaking_orders_count_l424_424076

theorem speaking_orders_count : 
  let students : Finset String := {"A", "B", "C", "D", "E", "F", "G"}
  let k : ℕ := 4
  let total_count : ℕ :=
    (C (2, 1) * C (5, 3) * A (4, 4)) + 
    (C (4, 2) * A (2, 2) * A (3, 2))
  in 
  (∀ (A B : String) (students_subset : Finset String),
    A ∈ students ∧ B ∈ students ∧ 
    students_subset ⊆ students ∧ 
    k = students_subset.card ∧ 
    (⌊A, B ∈ students_subset ⌋ → "C" ∉ students_subset) ∧ 
    (A ∈ students_subset ∨ B ∈ students_subset) ∧
     ∀ i j, i ≠ j ∧ 
    (A, B ∉ (st2, st3, st4)) → 
    count_total = 552) :=
sorry

end speaking_orders_count_l424_424076


namespace sparrow_population_decline_l424_424120

/--
  Given that the number of sparrows in a reserve diminishes by 40% annually, 
  and the initial count of sparrows on July 1, 2000 is \(N_0\) (where \(N_0 > 0\)),
  prove that by the 6th year (from 2000), the population will fall to 5% or less of the number recorded in the year 2000.
-/
theorem sparrow_population_decline (N_0 : ℕ) (hN : N_0 > 0) : 
  0.6^6 * N_0 ≤ 0.05 * N_0 :=
by
  sorry

end sparrow_population_decline_l424_424120


namespace line_tangent_to_parabola_proof_l424_424832

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l424_424832


namespace three_digit_integers_count_l424_424306

theorem three_digit_integers_count :
  let count := { n : ℤ | (100 ≤ n) ∧ (n < 1000) ∧ (n % 7 = 3) ∧ (n % 10 = 4) ∧ (n % 12 = 8) }.card
  count = 3 :=
by
  sorry

end three_digit_integers_count_l424_424306


namespace product_of_y_coordinates_l424_424789

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem product_of_y_coordinates : 
  let P1 := (1, 2 + 4 * Real.sqrt 2)
  let P2 := (1, 2 - 4 * Real.sqrt 2)
  distance (5, 2) P1 = 12 ∧ distance (5, 2) P2 = 12 →
  (P1.2 * P2.2 = -28) :=
by
  intros
  sorry

end product_of_y_coordinates_l424_424789


namespace circle_equation_tangent_l424_424201

theorem circle_equation_tangent (x y : ℝ) :
  let A := (-1, 2)
  let line := λ x y, x + 2 * y + 7 = 0
  let radius := 2 * (Real.sqrt 5)
  let center_circle := λ A x y, (x - A.1) ^ 2 + (y - A.2) ^ 2 - radius ^ 2 = 0
  center_circle (A.fst, A.snd) (x, y)
  ↔ (x + 1) ^ 2 + (y - 2) ^ 2 = 20 :=
begin
  sorry
end

end circle_equation_tangent_l424_424201


namespace larger_city_cubic_yards_l424_424724

theorem larger_city_cubic_yards
  (people_per_cubic_yard : ℕ)
  (delta_people : ℕ)
  (smaller_city_cubic_yards : ℕ)
  (people_in_smaller_city := people_per_cubic_yard * smaller_city_cubic_yards)
  (people_in_larger_city := people_in_smaller_city + delta_people)
  (larger_city_cubic_yards : ℕ)
  (h : people_in_larger_city = people_per_cubic_yard * larger_city_cubic_yards) :
  larger_city_cubic_yards = 9000 := 
by
  sorry

# Where:
# people_per_cubic_yard represents the number of people living per cubic yard
# delta_people represents the additional 208000 people in the larger city
# smaller_city_cubic_yards represents the 6400 cubic yards in the smaller city
# people_in_smaller_city calculates the number of people in the smaller city 
# people_in_larger_city calculates the number of people in the larger city, given delta_people
# larger_city_cubic_yards represents the number of cubic yards in the larger city (to be proven as 9000)

-- Specific values for our conditions, to be used in the proof:
-- people_per_cubic_yard = 80
-- delta_people = 208000
-- smaller_city_cubic_yards = 6400

end larger_city_cubic_yards_l424_424724


namespace count9s_in_range_1_to_100_l424_424102

def count9s (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc x => acc + x.digits.count (fun d => d = 9)) 0

theorem count9s_in_range_1_to_100 : count9s 100 = 20 :=
by
  -- This is a placeholder for actual proof which should be filled in
  sorry

end count9s_in_range_1_to_100_l424_424102


namespace divisors_49n5_l424_424190

def has_210_divisors (m : ℕ) : Prop :=
  m.divisor_count = 210

theorem divisors_49n5 
  (n : ℕ) (hn_pos : 0 < n) (h : has_210_divisors (210 * n^3)) : 
  (49 * n^5).divisor_count = 1728 := 
sorry

end divisors_49n5_l424_424190


namespace num_multiples_of_30_between_2000_3000_l424_424255

theorem num_multiples_of_30_between_2000_3000 : 
  (finset.card (finset.filter (λ n, 30 ∣ n)
    (finset.Icc 2000 3000))) = 34 :=
by sorry

end num_multiples_of_30_between_2000_3000_l424_424255


namespace average_salary_rest_l424_424336

theorem average_salary_rest (number_of_workers : ℕ) 
                            (avg_salary_all : ℝ) 
                            (number_of_technicians : ℕ) 
                            (avg_salary_technicians : ℝ) 
                            (rest_workers : ℕ) 
                            (total_salary_all : ℝ) 
                            (total_salary_technicians : ℝ) 
                            (total_salary_rest : ℝ) 
                            (avg_salary_rest : ℝ) 
                            (h1 : number_of_workers = 28)
                            (h2 : avg_salary_all = 8000)
                            (h3 : number_of_technicians = 7)
                            (h4 : avg_salary_technicians = 14000)
                            (h5 : rest_workers = number_of_workers - number_of_technicians)
                            (h6 : total_salary_all = number_of_workers * avg_salary_all)
                            (h7 : total_salary_technicians = number_of_technicians * avg_salary_technicians)
                            (h8 : total_salary_rest = total_salary_all - total_salary_technicians)
                            (h9 : avg_salary_rest = total_salary_rest / rest_workers) :
  avg_salary_rest = 6000 :=
by {
  -- the proof would go here
  sorry
}

end average_salary_rest_l424_424336


namespace max_rectangles_in_5x5_square_l424_424072

theorem max_rectangles_in_5x5_square :
  ∀ (a b : ℕ), a + b = 7 → 4 * a + 3 * b = 25 → a ≥ 0 ∧ b ≥ 0 :=
begin
  sorry
end

end max_rectangles_in_5x5_square_l424_424072


namespace range_eq_l424_424848

noncomputable def f (x : ℝ) : ℝ := x - 2 * real.sqrt (x + 1)

theorem range_eq : set.range f = set.Ici (-2) :=
sorry

end range_eq_l424_424848


namespace charlie_brown_lightning_distance_l424_424785

/-- 
Given:
- speed_of_sound = 1100 feet per second
- time_taken = 15 seconds
- feet_per_mile = 5280 feet

Goal:
Compute the distance to the nearest half mile from the flash of lightning 
--/
theorem charlie_brown_lightning_distance
  (speed_of_sound : ℕ := 1100) 
  (time_taken : ℕ := 15) 
  (feet_per_mile : ℕ := 5280) :
  let distance_feet := speed_of_sound * time_taken
  let distance_miles := (distance_feet : ℚ) / feet_per_mile
  distance_miles ≈ 3 :=
  sorry

end charlie_brown_lightning_distance_l424_424785


namespace polygon_angle_sum_l424_424854

theorem polygon_angle_sum (n : ℕ) : 
  let interior_angle_sum := (n - 1) * 180 
  let exterior_angle_sum := 360 
  in interior_angle_sum + exterior_angle_sum = (n + 1) * 180 := 
by 
  let interior_angle_sum := (n - 1) * 180 
  let exterior_angle_sum := 360 
  have h : interior_angle_sum + exterior_angle_sum = (n + 1) * 180 := sorry 
  exact h

end polygon_angle_sum_l424_424854


namespace cost_of_each_fish_is_four_l424_424751

-- Definitions according to the conditions
def number_of_fish_given_to_dog := 40
def number_of_fish_given_to_cat := number_of_fish_given_to_dog / 2
def total_fish := number_of_fish_given_to_dog + number_of_fish_given_to_cat
def total_cost := 240
def cost_per_fish := total_cost / total_fish

-- The main statement / theorem that needs to be proved
theorem cost_of_each_fish_is_four :
  cost_per_fish = 4 :=
by
  sorry

end cost_of_each_fish_is_four_l424_424751


namespace ratio_of_intercepts_l424_424874

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l424_424874


namespace number_of_women_in_company_l424_424920

theorem number_of_women_in_company (W : ℕ) : 
  (W / 3) * 2 * 0.4 + (W / 3) * 0.8 = 112 → 
  W = 210 →
  W - 112 = 98 :=
by sorry

end number_of_women_in_company_l424_424920


namespace hiker_average_speed_l424_424955

theorem hiker_average_speed :
  let trail_length := 3.8 -- km
  let time_flat := 45 // minutes
  let time_uphill := 35 // minutes
  let time_downhill := 25 // minutes
  let total_time := (time_flat + time_uphill + time_downhill : ℝ) / 60 -- in hours
  let average_speed := trail_length / total_time
  average_speed = 2.17 := 
by
  sorry

end hiker_average_speed_l424_424955


namespace even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l424_424377

open Int
open Nat

theorem even_n_square_mod_8 (n : ℤ) (h : n % 2 = 0) : (n^2 % 8 = 0) ∨ (n^2 % 8 = 4) := sorry

theorem odd_n_square_mod_8 (n : ℤ) (h : n % 2 = 1) : n^2 % 8 = 1 := sorry

theorem odd_n_fourth_mod_8 (n : ℤ) (h : n % 2 = 1) : n^4 % 8 = 1 := sorry

end even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l424_424377


namespace find_m_l424_424937

theorem find_m (m : ℕ) (h_pos : 0 < m)
  (h_expect : (7 / (7 + m) * 3 + m / (7 + m) * -1) = 1) : m = 7 :=
sorry

end find_m_l424_424937


namespace constant_g_exists_l424_424793

-- Lean statements for the given problem:
variable {G : Type} [Graph G]
variable {r : ℕ}

theorem constant_g_exists (g : ℕ) (Hcirc : ∀ (G : Type) [Graph G], circ(G) ≥ g)
    (Hchi : ∀ (G : Type) [Graph G], χ(G) ≥ r) :
    ∃ g, (∀ (G : Type) [Graph G], circ(G) ≥ g → (χ(G) ≥ r → G ⊇ TK^r)) :=
begin
  sorry
end

end constant_g_exists_l424_424793


namespace number_of_9s_in_1_to_100_l424_424104

theorem number_of_9s_in_1_to_100 : ∀ n, 1 ≤ n ∧ n ≤ 100 → count_digit 9 n = 19 :=
by
  -- Proof here
  sorry

end number_of_9s_in_1_to_100_l424_424104


namespace lineup_arrangements_l424_424531

theorem lineup_arrangements (n r : ℕ) (h₀ : 0 < 2 * r) (h₁ : 2 * r < n - 3) : 
  2 * (n - 3)! * (n - 2 * r - 2) = 2 * (n - 3)! * (n - 2 * r - 2) := 
by
  sorry

end lineup_arrangements_l424_424531


namespace find_range_a_l424_424696

def f (x a : ℝ) : ℝ := Real.exp x * (Real.sin x + a * Real.cos x)

variable {a : ℝ}

theorem find_range_a :
  (∀ x ∈ Set.Ioc (Real.pi / 4) (Real.pi / 2), has_deriv_at ℝ (f x a) (differentiable_at ℝ _ _)) →
  (∀ x ∈ Set.Ioc (Real.pi / 4) (Real.pi / 2), deriv (f x a) x ≥ 0) →
  a ≤ 1 :=
sorry

end find_range_a_l424_424696


namespace interval_of_decrease_for_f_l424_424144

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3)

def decreasing_interval (s : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem interval_of_decrease_for_f :
  decreasing_interval {x : ℝ | x < -1} f :=
by
  sorry

end interval_of_decrease_for_f_l424_424144


namespace isosceles_triangle_area_ac_l424_424731

theorem isosceles_triangle_area_ac 
  (ABC : Triangle)
  (AB BC AC : ℝ) 
  (y x : ℝ)
  (S1 S2 : ℝ) 
  (h1 : AB = y) 
  (h2 : BC = y) 
  (h3 : AC = x)
  (h4 : triangle_isosceles ABC AB BC)
  (h5 : angle_bisector ABC AD)
  (h_S1 : area_triangle (ABD) = S1)
  (h_S2 : area_triangle (ADC) = S2) :
  AC = (2 * sqrt(S2 * (S1 + S2))) / (real.sqrt4 (4 * S1^2 - S2^2)) :=
begin
  sorry,
end

end isosceles_triangle_area_ac_l424_424731


namespace segment_combination_l424_424733

theorem segment_combination (x y : ℕ) :
  7 * x + 12 * y = 100 ↔ (x, y) = (4, 6) :=
by
  sorry

end segment_combination_l424_424733


namespace proof_problem_l424_424686

noncomputable theory

def point := ℝ × ℝ

def O : point := (0, 0)
def A : point := (2, 9)
def B : point := (6, -3)

def P (λ : ℝ) : point := (14, -7)
def Q : point := (4, 3)
  
-- Condition: Given
def vec(OP : point -> point) (P : point) : point := (P.1, P.2)
def vec(PB : point -> point) (P : point) : point := (P.1 - B.1, P.2 - B.2)

-- First problem: Find λ and coordinated of P
def λ_condition {λ : ℝ} : Prop :=
  let P := vec O (14, λ)
  λ = -7 / 4 ∧ P = (14, -7)
  
-- Second problem: Find coordinates of Q
def perpendicular_condition (P : point) (Q : point) : Prop :=
  let AP : point := (A.1 - P.1, A.2 - P.2)
  Q.1 * AP.2 = Q.2 * AP.1
  
def line_eq_condition (Q : point) : Prop :=
  let AQ := (4 * Q.1, 3 * (Q.2 + 3))
  (3 * Q.1 + Q.2 - 15 = 0) ∧ (Q.1 = 4 ∧ Q.2 = 3)

-- Third problem: Range for dot product
def range_condition (O : point) (Q : point) : set ℝ :=
  let R (t : ℝ) := (4 * t, 3 * t)
  let RO (t : ℝ) := (t - 0.5) * t * 50 - 25 / 2
  {(R t.1, R t.2) | t' ∈ set.Icc (0:ℝ) (1:ℝ)} 

theorem proof_problem :
  λ_condition ∧ perpendicular_condition (P _) Q ∧ line_eq_condition Q ∧ range_condition :=
sorry

end proof_problem_l424_424686


namespace tangent_line_parabola_l424_424828

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l424_424828


namespace Ann_trip_takes_longer_l424_424395

theorem Ann_trip_takes_longer (mary_distance : ℕ) (mary_speed : ℕ)
                              (ann_distance : ℕ) (ann_speed : ℕ)
                              (mary_time : ℕ) (ann_time : ℕ) :
  mary_distance = 630 →
  mary_speed = 90 →
  ann_distance = 800 →
  ann_speed = 40 →
  mary_time = mary_distance / mary_speed →
  ann_time = ann_distance / ann_speed →
  (ann_time - mary_time) = 13 :=
by
  intros
  calculate!
  sorry

end Ann_trip_takes_longer_l424_424395


namespace hyperbola_focus_l424_424718

theorem hyperbola_focus (a : ℝ) (h₁ : 1 = a * sqrt 2) (h₂ : a > 0) : a = sqrt 2 / 2 :=
by
  sorry

end hyperbola_focus_l424_424718


namespace nat_pair_count_eq_five_l424_424285

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424285


namespace book_page_count_l424_424977

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end book_page_count_l424_424977


namespace championship_outcomes_l424_424646

theorem championship_outcomes :
  ∀ (students events : ℕ), students = 4 → events = 3 → students ^ events = 64 :=
by
  intros students events h_students h_events
  rw [h_students, h_events]
  exact rfl

end championship_outcomes_l424_424646


namespace sequence_mod_p_l424_424366

variables (a b c : ℤ)
variable [fact (odd b)]

def sequence_def (x : ℕ → ℤ) : Prop :=
  x 0 = 4 ∧ x 1 = 0 ∧ x 2 = 2 * c ∧ x 3 = 3 * b ∧
  ∀ n ≥ 1, x (n + 3) = a * x (n - 1) + b * x n + c * x (n + 1)

theorem sequence_mod_p (x : ℕ → ℤ) (m : ℕ) (p : ℕ) [fact (nat.prime p)] :
  sequence_def a b c x →
  ∀ m, x (p^m) % p = 0 :=
sorry

end sequence_mod_p_l424_424366


namespace num_pairs_nat_nums_eq_l424_424273

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424273


namespace change_received_l424_424357

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end change_received_l424_424357


namespace smallest_black_edges_l424_424617

noncomputable def min_black_edges : ℕ :=
  6

def rectangular_prism (x y z : ℕ) :=
  x = 2 ∧ y = 3 ∧ z = 4

def each_face_has_two_non_adjacent_black_edges (edges : ℕ) : Prop :=
  edges = 2

theorem smallest_black_edges (x y z edges : ℕ) :
  rectangular_prism x y z →
  (∀ face, each_face_has_two_non_adjacent_black_edges edges) →
  edges * 3 = min_black_edges :=
by sorry

end smallest_black_edges_l424_424617


namespace line_tangent_parabola_unique_d_l424_424836

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l424_424836


namespace geo_shapes_with_rectangle_cross_section_l424_424033

-- Definitions for the geometric shapes
structure RectangularPrism := (height width length : ℝ)
structure Cylinder := (radius height : ℝ)
structure Cone := (radius height : ℝ)
structure Cube := (side : ℝ)

-- Predicate to check if a shape can have a rectangular cross-section when intersected by a plane
def canHaveRectangleCrossSection : Type → Prop
| RectangularPrism := True
| Cylinder := True
| Cone := False
| Cube := True
| _ := False

-- Instances of the shapes
def shape1 : RectangularPrism := ⟨1, 1, 1⟩
def shape2 : Cylinder := ⟨1, 1⟩
def shape3 : Cone := ⟨1, 1⟩
def shape4 : Cube := ⟨1⟩

-- Theorem statement
theorem geo_shapes_with_rectangle_cross_section :
  canHaveRectangleCrossSection shape1 ∧
  canHaveRectangleCrossSection shape2 ∧
  ¬ canHaveRectangleCrossSection shape3 ∧
  canHaveRectangleCrossSection shape4 :=
by
  sorry

end geo_shapes_with_rectangle_cross_section_l424_424033


namespace limit_sum_b_n_l424_424658

/-- Definitions of sequences a_n and b_n as per the given conditions --/

def S (n : ℕ) : ℕ := -n^2 + n

def a (n : ℕ) : ℕ :=
if h : n = 1 then 0
else let m := n - 1 in S n - S m

def b (n : ℕ) : ℕ := 2 ^ (a n)

theorem limit_sum_b_n : limit (λ n, (∑ i in range (n + 1), b i)) (4 / 3) :=
by 
sorry

end limit_sum_b_n_l424_424658


namespace cone_lateral_surface_area_l424_424225

theorem cone_lateral_surface_area (r l : ℝ) (h1 : r = 2) (h2 : l = 5) : 
    0.5 * (2 * Real.pi * r * l) = 10 * Real.pi := by
    sorry

end cone_lateral_surface_area_l424_424225


namespace quadratic_has_real_roots_iff_a_eq_neg_one_l424_424640

theorem quadratic_has_real_roots_iff_a_eq_neg_one
  (a : ℝ) : 
  (∃ x : ℝ, a * (1 + complex.i) * x^2 + (1 + a^2 * complex.i) * x + (a^2 + complex.i) = (0 : ℂ)) ↔ 
  a = -1 :=
by 
  sorry

end quadratic_has_real_roots_iff_a_eq_neg_one_l424_424640


namespace f_increasing_f_range_l424_424653

section
variables {f : ℝ → ℝ}
variables (h_add : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
variables (h_pos : ∀ x : ℝ, x > 0 → f(x) > 0)
variables (h_neg_one : f(-1) = -2)

theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
by
  sorry

theorem f_range : set.range (f : ℝ → ℝ) ∩ set.Icc (-2 : ℝ) 1 = set.Icc (-4 : ℝ) 2 :=
by
  sorry
end

end f_increasing_f_range_l424_424653


namespace strictly_increasing_seqs_count_l424_424167

theorem strictly_increasing_seqs_count : 
  ∑ k in (finset.range 9).filter (λ k, k ≥ 2), nat.choose 9 k = 502 := by
  sorry

end strictly_increasing_seqs_count_l424_424167


namespace toms_average_speed_l424_424863

-- Conditions
def segment1_distance := 10 -- miles
def segment1_speed := 20 -- miles per hour
def segment1_time := segment1_distance / segment1_speed -- hours

def segment2_distance := 15 -- miles
def segment2_speed := 30 -- miles per hour
def segment2_time := segment2_distance / segment2_speed -- hours

def segment3_distance := 25 -- miles
def segment3_speed := 45 -- miles per hour
def segment3_time := segment3_distance / segment3_speed -- hours

def segment4_distance := 40 -- miles
def segment4_speed := 60 -- miles per hour
def segment4_time := segment4_distance / segment4_speed -- hours

def total_distance := segment1_distance + segment2_distance + segment3_distance + segment4_distance -- 90 miles
noncomputable def total_time := segment1_time + segment2_time + segment3_time + segment4_time -- hours

-- Average speed calculation
noncomputable def avg_speed := total_distance / total_time -- miles per hour

-- Theorem statement
theorem toms_average_speed : avg_speed = 40.5 := by
  sorry

end toms_average_speed_l424_424863


namespace intersection_point_value_of_A_l424_424195

theorem intersection_point_value_of_A (a b : ℝ)
  (h1 : b = a - 3)
  (h2 : b = 2 / a) :
  let A := (a - (a ^ 2 / (a + b))) / (a ^ 2 * b ^ 2 / (a ^ 2 - b ^ 2))
  in A = 3 / 2 := by
  -- Proof goes here
  sorry

end intersection_point_value_of_A_l424_424195


namespace probability_F_l424_424954

theorem probability_F :
  let P_D := 2 / 5
  let P_E := 1 / 5
  let P_G := 1 / 10
  let P_F := 1 - P_D - P_E - P_G
  P_F = 3 / 10 :=
by
  let P_D := 2 / 5
  let P_E := 1 / 5
  let P_G := 1 / 10
  have P_F := 1 - P_D - P_E - P_G
  have P_F_eq : P_F = 3 / 10 := sorry
  exact P_F_eq

end probability_F_l424_424954


namespace wire_length_after_cuts_l424_424322

-- Given conditions as parameters
def initial_length_cm : ℝ := 23.3
def first_cut_mm : ℝ := 105
def second_cut_cm : ℝ := 4.6

-- Final statement to be proved
theorem wire_length_after_cuts (ell : ℝ) (c1 : ℝ) (c2 : ℝ) : (ell = 23.3) → (c1 = 105) → (c2 = 4.6) → 
  (ell * 10 - c1 - c2 * 10 = 82) := sorry

end wire_length_after_cuts_l424_424322


namespace radius_squared_proof_l424_424946

-- Define the given conditions about the tangents and distances
variables {r : ℝ}
def ER : ℝ := 25
def RF : ℝ := 35
def GS : ℝ := 40
def SH : ℝ := 20
def ET : ℝ := 45
def RT : ℝ := 25
def ST : ℝ := 40

-- Define the result of squaring the radius of the circle
def r_squared := r * r

-- Prove that the square of the radius of the circle is 3600 given the conditions
theorem radius_squared_proof (r : ℝ) (h1 : RT = ER) (h2 : ST = GS) : r_squared = 3600 :=
sorry

end radius_squared_proof_l424_424946


namespace casey_hula_hoop_difference_l424_424399

theorem casey_hula_hoop_difference :
  ∀ (n c m : ℕ), n = 10 → m = 21 → m = 3 * c → (n - c = 3) :=
begin
  intros n c m hn hm hmc,
  rw [hn, hm, hmc],
  norm_num,
end

end casey_hula_hoop_difference_l424_424399


namespace ratio_c_d_l424_424016

theorem ratio_c_d (a b c d : ℝ) (h_eq : ∀ x, a * x^3 + b * x^2 + c * x + d = 0) 
    (h_roots : ∀ r, r = 2 ∨ r = 4 ∨ r = 5 ↔ (a * r^3 + b * r^2 + c * r + d = 0)) :
    c / d = 19 / 20 :=
by
  sorry

end ratio_c_d_l424_424016


namespace chocolate_bar_cost_l424_424944

variable (cost_per_bar num_bars : ℝ)

theorem chocolate_bar_cost (num_scouts smores_per_scout smores_per_bar : ℕ) (total_cost : ℝ)
  (h1 : num_scouts = 15)
  (h2 : smores_per_scout = 2)
  (h3 : smores_per_bar = 3)
  (h4 : total_cost = 15)
  (h5 : num_bars = (num_scouts * smores_per_scout) / smores_per_bar)
  (h6 : total_cost = cost_per_bar * num_bars) :
  cost_per_bar = 1.50 :=
by
  sorry

end chocolate_bar_cost_l424_424944


namespace find_n_l424_424501

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 :=
by {
    -- Let's assume there exists an integer n such that the given condition holds
    use 4,
    -- We now prove the condition and the conclusion
    split,
    -- This simplifies the left-hand side of the condition to 15, achieving the goal
    calc 
        4 + (4 + 1) + (4 + 2) = 4 + 5 + 6 : by rfl
        ... = 15 : by norm_num,
    -- The conclusion directly follows
    rfl
}

end find_n_l424_424501


namespace icosahedron_painting_distinguishable_ways_l424_424563

theorem icosahedron_painting_distinguishable_ways :
  let num_faces := 20
  let num_colors := 20
  let rotational_symmetries := 60
  num_faces = num_colors →
  -- The number of distinguishable ways to paint the icosahedron is 19! / 60
  ∀ (f : Fin 20 → Fin 20), -- Function f maps each face to a color
  ∃! c, 
    (∀ i j, (¬ adjacent i j) → f i ≠ f j) ∧ -- Non-adjacent faces different colors
    (f ≃₁ f) -- f is in the same equivalence class of rotations as itself
    →
    finset.card (equiv.perm.preimage f) = Nat.factorial 19 / rotational_symmetries :=
sorry

end icosahedron_painting_distinguishable_ways_l424_424563


namespace cos_shift_right_l424_424487

theorem cos_shift_right :
  ∀ (x : ℝ), cos (2 * x + (π / 4)) = cos (2 * (x + π / 8)) :=
begin
  intro x,
  have h : cos (2 * x + π / 4) = cos (2 * (x + π / 8)),
  { sorry },
  exact h,
end

end cos_shift_right_l424_424487


namespace count_valid_pairs_l424_424261

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424261


namespace identify_non_fraction_l424_424580

def is_fraction (num denom : ℤ) (denom_poly : Prop) : Prop :=
  denom_poly

theorem identify_non_fraction :
  (denom_poly_m : Prop) (denom_poly_a : Prop) (denom_poly_y : Prop) →
  ¬ (is_fraction x 5 False) :=
by
  assume denom_poly_m denom_poly_a denom_poly_y
  simp
  sorry

end identify_non_fraction_l424_424580


namespace series_sum_correct_l424_424184

noncomputable def sum_series (n : ℕ) : ℂ :=
  ∑ k in finset.range (n//2+1), (-1)^k * (1 / (3^k)) * (nat.choose n (2*k + 1))

theorem series_sum_correct (n : ℕ) : 
  sum_series n = 2^n * (3:ℂ)^(1 - n/2) * complex.sin (n * real.pi / 6) :=
by
  sorry

end series_sum_correct_l424_424184


namespace determinant_cos_matrix_is_zero_l424_424131

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    [Real.cos 1, Real.cos 2, Real.cos 3],
    [Real.cos 4, Real.cos 5, Real.cos 6],
    [Real.cos 7, Real.cos 8, Real.cos 9]
  ]

theorem determinant_cos_matrix_is_zero : det A = 0 := 
by sorry

end determinant_cos_matrix_is_zero_l424_424131


namespace b_is_geometric_c_is_arithmetic_sum_of_first_n_terms_d_l424_424744

def S (n : ℕ) : ℕ := 4 * a n + 2
def a : ℕ → ℕ
| 1 := 1
| n + 1 := 4 * a n + 2

def b (n : ℕ) := a (n+1) - 2 * a n
def c (n : ℕ) := a n / 2^n
def d (n : ℕ) := 1 / (c n * c (n+1))

theorem b_is_geometric :
  ∃ (r : ℕ), ∃ (b₁ : ℕ), b 1 = b₁ ∧ (∀ n, b (n+1) = r * b n) :=
sorry

theorem c_is_arithmetic :
  ∃ (d : ℕ), ∃ (c₁ : ℕ), c 1 = c₁ ∧ (∀ n, c (n+1) = c n + d) :=
sorry

theorem sum_of_first_n_terms_d (n : ℕ) :
  (∑ i in finset.range n, d i) = 4 / (3 * n + 1) :=
sorry

end b_is_geometric_c_is_arithmetic_sum_of_first_n_terms_d_l424_424744


namespace largest_lcm_value_is_90_l424_424037

def lcm_vals (a b : ℕ) : ℕ := Nat.lcm a b

theorem largest_lcm_value_is_90 :
  max (lcm_vals 18 3)
      (max (lcm_vals 18 9)
           (max (lcm_vals 18 6)
                (max (lcm_vals 18 12)
                     (max (lcm_vals 18 15)
                          (lcm_vals 18 18))))) = 90 :=
by
  -- Use the fact that the calculations of LCMs are as follows:
  -- lcm(18, 3) = 18
  -- lcm(18, 9) = 18
  -- lcm(18, 6) = 18
  -- lcm(18, 12) = 36
  -- lcm(18, 15) = 90
  -- lcm(18, 18) = 18
  -- therefore, the largest value among these is 90
  sorry

end largest_lcm_value_is_90_l424_424037


namespace vessel_base_length_l424_424549

variables (L : ℝ) (edge : ℝ) (W : ℝ) (h : ℝ)
def volume_cube := edge^3
def volume_rise := L * W * h

theorem vessel_base_length :
  (volume_cube 16 = volume_rise L 15 13.653333333333334) →
  L = 20 :=
by sorry

end vessel_base_length_l424_424549


namespace product_of_numbers_l424_424470

theorem product_of_numbers :
  ∃ (x y z : ℚ), (x + y + z = 30) ∧ (x = 3 * (y + z)) ∧ (y = 5 * z) ∧ (x * y * z = 175.78125) :=
by
  sorry

end product_of_numbers_l424_424470


namespace rectangle_square_area_ratio_l424_424454

theorem rectangle_square_area_ratio :
  let s := 20
  let area_S := s * s
  let longer_side_R := 1.05 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  ratio_R_S := area_R / area_S
in 
  ratio_R_S = 357 / 400 := by
  sorry

end rectangle_square_area_ratio_l424_424454


namespace area_XHI_l424_424350

variable (X Y Z G H I : Type) [inhabited X] [inhabited Y] [inhabited Z] [inhabited G] [inhabited H] [inhabited I]

-- Given conditions
variable (midpoint_G : G = midpoint X Y)
variable (midpoint_H : H = midpoint X Z)
variable (midpoint_I : I = midpoint X G)
variable (area_XYZ : ℝ)
variable [h_area_XYZ : (area (triangle X Y Z) = 120)]

-- Goal
theorem area_XHI :
  area (triangle X H I) = 30 := sorry

end area_XHI_l424_424350


namespace value_of_x_l424_424473

theorem value_of_x (z : ℕ) (y : ℕ) (x : ℕ) 
  (h₁ : y = z / 5)
  (h₂ : x = y / 2)
  (h₃ : z = 60) : 
  x = 6 :=
by
  sorry

end value_of_x_l424_424473


namespace euler_line_nine_point_circle_l424_424928

theorem euler_line_nine_point_circle (A B C : Point) (G O H N : Point) 
  (h1 : NinePointCircle A B C passes_through (midpoint A B) ∧ passes_through (midpoint B C) ∧ passes_through (midpoint C A) ∧
        passes_through (foot_of_altitude A) ∧ passes_through (foot_of_altitude B) ∧ passes_through (foot_of_altitude C) ∧
        passes_through (midsegment A H) ∧ passes_through (midsegment B H) ∧ passes_through (midsegment C H))
  (hCircum : CircumCircle A B C passes_through A ∧ passes_through B ∧ passes_through C)
  (hHomothety : Homothety (circumcircle (triangle A B C)) (nine_point_circle (triangle A B C)) (-1 / 2) G)
  (hCentroid : divides (median A) (ratio 2 1) G ∧ divides (median B) (ratio 2 1) G ∧ divides (median C) (ratio 2 1) G)
  (hMidpoint : N = midpoint O H)
  (hEulerLine : Line_through (euler_line (triangle A B C)) H ∧ Line_through (euler_line (triangle A B C)) G ∧ Line_through (euler_line (triangle A B C)) O)
  : Line_through (euler_line (triangle A B C)) N :=
sorry

end euler_line_nine_point_circle_l424_424928


namespace find_other_number_l424_424850

-- Define the conditions
variable (B : ℕ)
variable (HCF : ℕ → ℕ → ℕ)
variable (LCM : ℕ → ℕ → ℕ)

axiom hcf_cond : HCF 24 B = 15
axiom lcm_cond : LCM 24 B = 312

-- The theorem statement
theorem find_other_number (B : ℕ) (HCF : ℕ → ℕ → ℕ) (LCM : ℕ → ℕ → ℕ) 
  (hcf_cond : HCF 24 B = 15) (lcm_cond : LCM 24 B = 312) : 
  B = 195 :=
sorry

end find_other_number_l424_424850


namespace probability_abcd_16_l424_424049

theorem probability_abcd_16 :
  let outcomes := {1, 2, 3, 4, 5, 6}
  let events := {t | (t ∈ outcomes × outcomes × outcomes × outcomes) ∧ (∃ a b c d, t = (a, b, c, d) ∧ a * b * c * d = 16)}
  @Prob (outcomes : Set ℕ) (fun x => discreteUniform outcomes) events = 7 / 1296 := 
sorry

end probability_abcd_16_l424_424049


namespace tile_arrangement_probability_l424_424422

theorem tile_arrangement_probability :
  let X := 4  -- Number of tiles marked X
  let O := 2  -- Number of tiles marked O
  let total := 6  -- Total number of tiles
  let arrangement := [true, true, false, true, false, true]  -- XXOXOX represented as [X, X, O, X, O, X]
  (↑(X / total) * ↑((X - 1) / (total - 1)) * ↑((O / (total - 2))) * ↑((X - 2) / (total - 3)) * ↑((O - 1) / (total - 4)) * 1 : ℚ) = 1 / 15 :=
sorry

end tile_arrangement_probability_l424_424422


namespace gear_ratios_l424_424004

variable (x y z : ℕ)
variable (ω_A ω_B ω_C : ℝ)
variable h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C

theorem gear_ratios (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C) : 
  ω_A / ω_B / ω_C = yz / xz / xy := sorry

end gear_ratios_l424_424004


namespace flea_probability_at_B_after_2019_jumps_l424_424006

variable (p1 p2 p3 p4 : ℝ)

theorem flea_probability_at_B_after_2019_jumps
  (h1 : p1 + p2 + p3 + p4 = 1)
  (h2 : p1 + p3 = 0.2) :
  let P_B_2019 := 1/2 * (p1 + p3)
  in P_B_2019 = 0.1 :=
by sorry

end flea_probability_at_B_after_2019_jumps_l424_424006


namespace count_sum_or_diff_squares_at_least_1500_l424_424711

theorem count_sum_or_diff_squares_at_least_1500 : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 ∧ (∃ (x y : ℕ), n = x^2 + y^2 ∨ n = x^2 - y^2)) → 
  1500 ≤ 2000 :=
by
  sorry

end count_sum_or_diff_squares_at_least_1500_l424_424711


namespace smallest_n_correct_l424_424043

noncomputable def smallest_n (z : ℤ) : ℤ :=
  if h : z = 9 then 26 else sorry

theorem smallest_n_correct : smallest_n 9 = 26 :=
by {
  unfold smallest_n,
  split_ifs,
  { refl },
  { sorry }
}

end smallest_n_correct_l424_424043


namespace midpoint_correct_solution_exists_num_solutions_l424_424248

variables {R : Type*} [Field R] [EuclideanSpace3 R]

-- Define the points A, B, P, and the distance d
variables (A B P : EuclideanSpace3 R) (d : R)

-- Define the plane S and the projections A', B', P'
structure Plane (R : Type*) [Field R] [EuclideanSpace3 R] :=
  (Origin : EuclideanSpace3 R)
  (Normal : EuclideanSpace3 R)

variables (S : Plane R)
variables (A' B' P' M : EuclideanSpace3 R)
variables (ra rb rp : R)

-- Define the conditions for the projections
def conditions_projection : Prop :=
  (∥S.Origin - A'∥ + ∥S.Origin - B'∥ = d) ∧
  (∥S.Origin - P'∥ = d)

-- Define the midpoint of A' and B'
def midpoint (A' B' : EuclideanSpace3 R) : EuclideanSpace3 R :=
  (1 / 2) • (A' + B')

-- Midpoint used
theorem midpoint_correct : M = midpoint A' B' :=
  sorry

-- Define the circles
def circle1 : set (EuclideanSpace3 R) :=
  { x | ∥x - M∥ = d / 2 }

def circle2 : set (EuclideanSpace3 R) :=
  { x | ∥x - P'∥ = d }

-- Implies that the tangents meet the conditions
theorem solution_exists :
  (∃ S : EuclideanSpace3 R, ∥S - A'∥ + ∥S - B'∥ = d ∧ ∥S - P'∥ = d) :=
sorry

-- Conclude the potential number of solutions
theorem num_solutions :
  ∃ n : ℕ, n ≤ 4 ∧
  ∃ (planes : fin n → Plane R),
    ∀ i, conditions_projection A B P d (planes i) :=
sorry

end midpoint_correct_solution_exists_num_solutions_l424_424248


namespace complex_number_in_third_quadrant_l424_424345

open Complex

noncomputable def complex_number : ℂ := (1 - 3 * I) / (1 + 2 * I)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_number_in_third_quadrant : in_third_quadrant complex_number :=
sorry

end complex_number_in_third_quadrant_l424_424345


namespace max_length_OB_l424_424883

theorem max_length_OB (O A B : Type) (r : Real) (h : r = 1) (AOB : angle O A B = 30°) : max (length OB) = 2 :=
sorry

end max_length_OB_l424_424883


namespace inscribe_right_triangle_l424_424032

theorem inscribe_right_triangle
  (O M : Point)
  (α : Angle)
  (C : Circle) :
  (∃ A B : Point, ∠BCA = 90 ∧ Line A C ∋ M ∧ ∠CAB = α) :=
sorry

end inscribe_right_triangle_l424_424032


namespace eighteen_gon_vertex_number_l424_424124

theorem eighteen_gon_vertex_number (a b : ℕ) (P : ℕ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : P = a + b) : P = 38 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end eighteen_gon_vertex_number_l424_424124


namespace quadratic_roots_shifted_l424_424375

theorem quadratic_roots_shifted (a b c : ℝ) (r s : ℝ) 
  (h1 : 4 * r ^ 2 + 2 * r - 9 = 0) 
  (h2 : 4 * s ^ 2 + 2 * s - 9 = 0) :
  c = 51 / 4 := by
  sorry

end quadratic_roots_shifted_l424_424375


namespace f_3_minus_f_4_l424_424227

noncomputable def f : ℝ → ℝ := sorry
axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom initial_condition : f 1 = 1

theorem f_3_minus_f_4 : f 3 - f 4 = -1 :=
by
  sorry

end f_3_minus_f_4_l424_424227


namespace correct_option_is_A_l424_424906

def is_quadratic_equation_with_one_variable (eq : nat → nat → nat → Prop) :=
  ∃ a b c : nat, a ≠ 0 ∧ eq a b c

def Option_A (a b c : nat) : Prop := (-a) = -1 ∧ b = 4 ∧ c = 0

theorem correct_option_is_A :
  is_quadratic_equation_with_one_variable Option_A :=
by
  sorry

end correct_option_is_A_l424_424906


namespace cost_of_dinner_l424_424976

theorem cost_of_dinner (tax_rate service_charge_rate tip_rate total_spent : ℝ) (h1 : tax_rate = 0.12) (h2 : service_charge_rate = 0.05) (h3 : tip_rate = 0.18) (h4 : total_spent = 40.80) :
  let x := total_spent / (1 + service_charge_rate + tax_rate + tip_rate) in
  x = 30.22 :=
by 
  sorry

end cost_of_dinner_l424_424976


namespace part1_part2_l424_424690

noncomputable def f (x : ℝ) : ℝ := 2^x - 1 / 2^|x|

theorem part1 (x : ℝ) (h : f x = 2) : x = Real.log 2 (1 + Real.sqrt 2) :=
sorry

theorem part2 (m : ℝ) (h : ∀ (t : ℝ), 1 ≤ t ∧ t ≤ 2 → 2^t * f (2*t) + m * f t ≥ 0) : m ≥ -5 :=
sorry

end part1_part2_l424_424690


namespace express_x13_in_y_poly_l424_424152

theorem express_x13_in_y_poly (x : ℂ) (hx : x ≠ 0) :
  let y := x + ⅟x,
      a : ℕ → ℂ := λ k, x^k + ⅟x^k in
  y = x + ⅟x ∧
  a 0 = 2 ∧
  a 1 = y ∧
  (∀ k, a (k + 1) = y * a k - a (k - 1)) ∧
  (∀ k, k % 2 = 0 → a (2 * k) = (a k)^2 - 2)
  → a 13 = y^13 - 13 * y^11 + 65 * y^9 - 156 * y^7 + 182 * y^5 - 91 * y^3 + 13 * y :=
begin
  sorry
end

end express_x13_in_y_poly_l424_424152


namespace almond_butter_servings_l424_424950

noncomputable def servings_in_container (total_tbsps : ℚ) (serving_size : ℚ) : ℚ :=
  total_tbsps / serving_size

theorem almond_butter_servings :
  servings_in_container (34 + 3/5) (5 + 1/2) = 6 + 21/55 :=
by
  sorry

end almond_butter_servings_l424_424950


namespace trapezoid_area_l424_424471

-- Definitions of the problem's conditions
def a : ℕ := 4
def b : ℕ := 8
def h : ℕ := 3

-- Lean statement to prove the area of the trapezoid is 18 square centimeters
theorem trapezoid_area : (a + b) * h / 2 = 18 := by
  sorry

end trapezoid_area_l424_424471


namespace number_of_space_diagonals_l424_424952

theorem number_of_space_diagonals (V E F tF qF : ℕ)
    (hV : V = 30) (hE : E = 72) (hF : F = 44) (htF : tF = 34) (hqF : qF = 10) : 
    V * (V - 1) / 2 - E - qF * 2 = 343 :=
by
  sorry

end number_of_space_diagonals_l424_424952


namespace train_speed_with_coaches_l424_424115

theorem train_speed_with_coaches (V₀ : ℝ) (V₉ V₁₆ : ℝ) (k : ℝ) :
  V₀ = 30 → V₁₆ = 14 → V₉ = 30 - k * (9: ℝ) ^ (1/2: ℝ) ∧ V₁₆ = 30 - k * (16: ℝ) ^ (1/2: ℝ) →
  V₉ = 18 :=
by sorry

end train_speed_with_coaches_l424_424115


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424278

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424278


namespace arithmetic_sequence_sum_l424_424503

theorem arithmetic_sequence_sum :
  ∃ d x y, (∀ n, nth_term (3 + n * d) = 3 + n * d) ∧
            d = 6 ∧
            y = 33 - d ∧
            x = y - d ∧
            x + y = 48 :=
by
  sorry

end arithmetic_sequence_sum_l424_424503


namespace monotonic_increasing_iff_l424_424693

noncomputable def f (x b : ℝ) : ℝ := (x - b) * Real.log x + x^2

theorem monotonic_increasing_iff (b : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → 0 ≤ (Real.log x - b/x + 1 + 2*x)) ↔ b ∈ Set.Iic (3 : ℝ) :=
by
  sorry

end monotonic_increasing_iff_l424_424693


namespace bus_stops_per_hour_l424_424517

theorem bus_stops_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h₁ : speed_without_stoppages = 50)
  (h₂ : speed_with_stoppages = 40) :
  ∃ (minutes_stopped : ℝ), minutes_stopped = 12 :=
by
  sorry

end bus_stops_per_hour_l424_424517


namespace sufficient_condition_for_perpendicular_l424_424757

variables {Plane : Type} {Line : Type} 
variables (α β γ : Plane) (m n : Line)

-- Definitions based on conditions
variables (perpendicular : Plane → Plane → Prop)
variables (perpendicular_line : Line → Plane → Prop)
variables (intersection : Plane → Plane → Line)

-- Conditions from option D
variable (h1 : perpendicular_line n α)
variable (h2 : perpendicular_line n β)
variable (h3 : perpendicular_line m α)

-- Statement to prove
theorem sufficient_condition_for_perpendicular (h1 : perpendicular_line n α)
  (h2 : perpendicular_line n β) (h3 : perpendicular_line m α) : 
  perpendicular_line m β := 
sorry

end sufficient_condition_for_perpendicular_l424_424757


namespace line_tangent_parabola_unique_d_l424_424835

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l424_424835


namespace isosceles_triangle_angle_l424_424734

theorem isosceles_triangle_angle (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (a b c : triangle_data A B C) (h₁ : isosceles c ) (h₂ : ∠aBC = 120 °) : 
  angle ( to_vec A B) (to_vec B C) = 150:=
sorry

end isosceles_triangle_angle_l424_424734


namespace nat_pair_count_eq_five_l424_424288

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424288


namespace toothpicks_in_10th_stage_l424_424558

open Nat

theorem toothpicks_in_10th_stage : 
  let initial_toothpicks := 5
  let added_toothpicks (n : ℕ) := 3 * n
  ∀ n : ℕ, 
  (n ≥ 1) → 
  (∀ k : ℕ, k ≥ 2 → ∑ i in range (k - 1), added_toothpicks (i + 2) + initial_toothpicks = (3 * k * (k + 1)) / 2 + 2) := sorry

end toothpicks_in_10th_stage_l424_424558


namespace g_50_equals_279_l424_424142

noncomputable def g : ℕ → ℤ
| x := if (∃ (k : ℕ), 3^k = x) then int.log 3 x else 2 + g (x + 2)

theorem g_50_equals_279 : g 50 = 279 := 
sorry

end g_50_equals_279_l424_424142


namespace equilateral_triangle_side_length_l424_424405

theorem equilateral_triangle_side_length 
  (x1 y1 : ℝ) 
  (hx1y1 : y1 = - (1 / 4) * x1^2)
  (h_eq_tri: ∃ (x2 y2 : ℝ), x2 = -x1 ∧ y2 = y1 ∧ (x2, y2) ≠ (x1, y1) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = x1^2 + y1^2 ∧ (x1 - 0)^2 + (y1 - 0)^2 = (x1 - x2)^2 + (y1 - y2)^2)):
  2 * x1 = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l424_424405


namespace angle_BOE_eq_angle_COD_l424_424530

variable {A B C D E O : Type} [Geometry A]

theorem angle_BOE_eq_angle_COD (h1: angle_bisector A D B C)
                               (h2: is_incenter O A B C)
                               (h3: is_perpendicular E O (line B C)) :
  angle B O E = angle C O D := by
  sorry

end angle_BOE_eq_angle_COD_l424_424530


namespace children_got_off_bus_l424_424537

-- Conditions
def original_number_of_children : ℕ := 43
def children_left_on_bus : ℕ := 21

-- Definition of the number of children who got off the bus
def children_got_off : ℕ := original_number_of_children - children_left_on_bus

-- Theorem stating the number of children who got off the bus
theorem children_got_off_bus : children_got_off = 22 :=
by
  -- This is to indicate where the proof would go
  sorry

end children_got_off_bus_l424_424537


namespace pow_five_2010_mod_seven_l424_424041

theorem pow_five_2010_mod_seven :
  (5 ^ 2010) % 7 = 1 :=
by
  have h : (5 ^ 6) % 7 = 1 := sorry
  sorry

end pow_five_2010_mod_seven_l424_424041


namespace common_point_exists_l424_424408

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end common_point_exists_l424_424408


namespace TrajectoryOfPointP_l424_424342

variable (A B P : Point)
variable (x y : ℝ)
variable (B_symmetric : B = (1, -1))
variable (A : A = (-1, 1))

theorem TrajectoryOfPointP (H : (x ≠ 1) ∧ (x ≠ -1))
    (symmetry : B = (1, -1))
    (slope_product : ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = 1/3) : 
    x^2 - 3 * y^2 = -2 := sorry

end TrajectoryOfPointP_l424_424342


namespace marathon_total_distance_l424_424080

theorem marathon_total_distance 
    (equispaced_checkpoints : ∀ (n : ℕ), n ∈ {1, 2, 3} → distance (C n) (C (n + 1)) = 6)
    (C1_start : distance start C1 = 1)
    (C4_finish : distance C4 finish = 1) : 
    total_distance = 20 :=
by
    sorry

end marathon_total_distance_l424_424080


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424283

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424283


namespace ratio_of_shaded_to_white_area_l424_424892

/-- Given a figure where the vertices of all the squares, except for the largest one, 
    are located at the midpoints of the corresponding sides, prove that the ratio 
    of the area of the shaded part to the white part is 5:3. -/
theorem ratio_of_shaded_to_white_area :
  (let total_shaded := 20 in
  let total_white := 12 in
  total_shaded / total_white) = (5 / 3) :=
by {
  sorry
}

end ratio_of_shaded_to_white_area_l424_424892


namespace count_increasing_numbers_l424_424172

-- Define the set of digits we are concerned with
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a natural number type representing numbers with increasing digits
def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits i < n.digits j

-- Define the set of natural numbers with increasing digits and at least two digits
def increasing_numbers : set ℕ :=
  {n | increasing_digits n ∧ 10 ≤ n ∧ n ≤ 987654321}

-- Define the theorem to be proved
theorem count_increasing_numbers : set.card increasing_numbers = 502 :=
by sorry

end count_increasing_numbers_l424_424172


namespace number_of_pairs_count_number_of_pairs_l424_424301

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424301


namespace find_non_negative_integer_pairs_l424_424154

theorem find_non_negative_integer_pairs (m n : ℕ) :
  3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) := by
  sorry

end find_non_negative_integer_pairs_l424_424154


namespace amanda_earnings_l424_424577

def worked_hours : list ℕ := [90, 45, 135, 40]

def hourly_rate : ℕ := 4

def total_earnings (hours_list : list ℕ) (rate : ℕ) : ℕ :=
  let total_minutes := hours_list.foldl (λ acc x => acc + x) 0
  let total_hours := total_minutes.to_float / 60
  let earnings := total_hours * rate
  earnings.round

theorem amanda_earnings : total_earnings worked_hours hourly_rate = 21 := by
  sorry

end amanda_earnings_l424_424577


namespace sophomores_more_than_first_graders_l424_424020

def total_students : ℕ := 95
def first_graders : ℕ := 32
def second_graders : ℕ := total_students - first_graders

theorem sophomores_more_than_first_graders : second_graders - first_graders = 31 := by
  sorry

end sophomores_more_than_first_graders_l424_424020


namespace hispanic_population_in_west_is_41_percent_l424_424492

def NE_Hispanic : ℕ := 3
def MW_Hispanic : ℕ := 4
def South_Hispanic : ℕ := 10
def West_Hispanic : ℕ := 12

def total_Hispanic : ℕ := NE_Hispanic + MW_Hispanic + South_Hispanic + West_Hispanic

def Hispanic_percentage_West : ℕ := (West_Hispanic * 100) / total_Hispanic

theorem hispanic_population_in_west_is_41_percent :
  round (Hispanic_percentage_West) = 41 :=
by
  dsimp [Hispanic_percentage_West]
  norm_num
  sorry

end hispanic_population_in_west_is_41_percent_l424_424492


namespace count_pairs_satisfying_condition_l424_424267

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424267


namespace line_tangent_to_parabola_proof_l424_424830

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l424_424830


namespace simplify_expression1_simplify_expression2_l424_424526

-- 1. First expression simplification
theorem simplify_expression1 :
  (0.027:ℝ) ^ (2 / 3) + (27 / 125:ℝ) ^ (-1 / 3) - (2 + 7 / 9:ℝ) ^ 0.5 = 0.09 :=
by
  sorry

-- 2. Second expression simplification
theorem simplify_expression2 :
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) * (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) = 5 / 4 :=
by
  sorry

end simplify_expression1_simplify_expression2_l424_424526


namespace sum_of_digits_of_8_pow_1002_l424_424499

theorem sum_of_digits_of_8_pow_1002 :
  let n := 8^1002 in
  let tens_digit := (n / 10) % 10 in
  let units_digit := n % 10 in
  tens_digit + units_digit = 10 := by
  sorry

end sum_of_digits_of_8_pow_1002_l424_424499


namespace remainder_of_polynomial_l424_424182

noncomputable def P (x : ℝ) := 3 * x^5 - 2 * x^3 + 5 * x^2 - 8
noncomputable def D (x : ℝ) := x^2 + 3 * x + 2
noncomputable def R (x : ℝ) := 64 * x + 60

theorem remainder_of_polynomial :
  ∀ x : ℝ, P x % D x = R x :=
sorry

end remainder_of_polynomial_l424_424182


namespace number_of_increasing_digits_l424_424165

theorem number_of_increasing_digits : 
  (∑ k in finset.range 10, if 2 ≤ k then nat.choose 9 k else 0) = 502 :=
by
  sorry

end number_of_increasing_digits_l424_424165


namespace count_valid_pairs_l424_424262

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424262


namespace routes_from_A_to_B_l424_424604

-- Define cities as a type with \( A, B, C, D, C' \)
inductive City
| A | B | C | D | C' 
deriving DecidableEq

-- Define roads as a type with corresponding roads
inductive Road
| AB | AD | AE | BC | BC' | CD | DA
deriving DecidableEq

def usesEachRoadExactlyOnce (path : List Road) : Prop := 
  -- Checks if each road appears exactly once in the path
  List.nodup path ∧ ∀ r : Road, r ∈ path

def immediateTravelToD (path : List City) : Prop := 
  -- Checks if the travel from \( C \) or \( C' \) is immediately followed by \( D \)
  ∀ i, (path.get? i = some City.C ∨ path.get? i = some City.C') → path.get? (i + 1) = some City.D

-- Number of routes from \( A \) to \( B \) using each road exactly once, in compliance with the defined travel restrictions
theorem routes_from_A_to_B : 
  ∃ (path : List Road), 
    (usesEachRoadExactlyOnce path) ∧ (immediateTravelToD (List.map Road.travel path)) ∧ 
    (List.head path = some Road.AB ∨ List.head path = some Road.AD ∨ List.head path = some Road.AE) ∧ 
    -- Ensuring path starts at \( A \) and ends at \( B \),
    List.head ([City.A, City.B]) = City.A ∧ List.last ([City.A, City.B]) = City.B ∧ 
    -- Number of these paths is exactly 9
    (List.length path = 9) := sorry

end routes_from_A_to_B_l424_424604


namespace molecular_weight_of_compound_l424_424039

noncomputable def molecularWeight (Ca_wt : ℝ) (O_wt : ℝ) (H_wt : ℝ) (nCa : ℕ) (nO : ℕ) (nH : ℕ) : ℝ :=
  (nCa * Ca_wt) + (nO * O_wt) + (nH * H_wt)

theorem molecular_weight_of_compound :
  molecularWeight 40.08 15.999 1.008 1 2 2 = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l424_424039


namespace polar_circle_eq_l424_424743

theorem polar_circle_eq {ρ θ : ℝ} : 
  (∀ x y : ℝ, (x = ρ * cos θ ∧ y = ρ * sin θ) → ((x - 1)^2 + y^2 = 1) ↔ (ρ = 2 * cos θ)) :=
by
  sorry

end polar_circle_eq_l424_424743


namespace range_of_x_l424_424998

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_l424_424998


namespace part1_part2_l424_424697

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

theorem part1 (x : ℝ) : f x ≥ -x^2 + x := sorry

theorem part2 (k : ℝ) : (∀ x > 0, f x ≥ k * x) → k ≤ Real.exp 1 - 2 := sorry

end part1_part2_l424_424697


namespace monotonically_increasing_interval_triangle_area_l424_424692

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * cos (π / 2 + x) * cos x + sin x ^ 2

theorem monotonically_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ,
    (k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) ↔
    (derivative f x > 0) := sorry

/-- For triangle ABC, given B = π/4, a = 2, and f(A) = 0,
    the area is (3 + sqrt 3) / 3. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  B = π / 4 →
  a = 2 →
  f A = 0 →
  (area A B C a b c = (3 + sqrt 3) / 3) := sorry

end monotonically_increasing_interval_triangle_area_l424_424692


namespace base_angle_isosceles_l424_424732

-- Define an isosceles triangle with one angle being 100 degrees
def isosceles_triangle (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A) ∧ (angle_A + angle_B + angle_C = 180) ∧ (angle_A = 100)

-- The main theorem statement
theorem base_angle_isosceles {A B C : Type} (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) :
  isosceles_triangle A B C angle_A angle_B angle_C → (angle_B = 40 ∨ angle_C = 40) :=
  sorry

end base_angle_isosceles_l424_424732


namespace find_higher_percentage_l424_424079

-- Definitions based on conditions
def principal : ℕ := 8400
def time : ℕ := 2
def rate_0 : ℕ := 10
def delta_interest : ℕ := 840

-- The proof statement
theorem find_higher_percentage (r : ℕ) :
  (principal * rate_0 * time / 100 + delta_interest = principal * r * time / 100) →
  r = 15 :=
by sorry

end find_higher_percentage_l424_424079


namespace d_share_l424_424516

theorem d_share (x : ℝ) (d c : ℝ)
  (h1 : c = 3 * x + 500)
  (h2 : d = 3 * x)
  (h3 : c = 4 * x) :
  d = 1500 := 
by 
  sorry

end d_share_l424_424516


namespace PF1_plus_PF2_constant_l424_424340

variables (a b c : ℝ)
-- The conditions
def ellipse_eq (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def foci_left := (-c, 0)
def foci_right := (c, 0)

def point_A := (Ax, Ay) -- Assume A has coordinates (Ax, Ay)
def point_B := (Bx, By) -- Assume B has coordinates (Bx, By)
def point_P := (Px, Py) -- Assume P has coordinates (Px, Py)

-- Assume a > b > 0
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom ab_relation : a > b

-- A F1 is parallel to B F2
axiom AF1_parallel_BF2 : true -- This axiom represents the parallel condition

-- A F2 intersects B F1 at P
axiom AF2_intersects_BF1_at_P : true -- This axiom represents the intersection condition

-- Prove that PF1 + PF2 is constant
theorem PF1_plus_PF2_constant : 
  (dist (Px, Py) foci_left) + (dist (Px, Py) foci_right) = some_constant :=
sorry

end PF1_plus_PF2_constant_l424_424340


namespace calculated_area_error_l424_424469

def percentage_error_area (initial_length_error : ℝ) (initial_width_error : ℝ) 
(temperature_change : ℝ) (humidity_change : ℝ) 
(length_error_per_temp : ℝ) (width_error_per_humidity : ℝ) : ℝ :=
let total_length_error := initial_length_error + (temperature_change / 5) * length_error_per_temp in
let total_width_error := initial_width_error + (humidity_change / 10) * width_error_per_humidity in
total_length_error - total_width_error

theorem calculated_area_error :
  percentage_error_area 3 2 15 20 1 0.5 = 3 :=
sorry

end calculated_area_error_l424_424469


namespace min_value_inverse_sum_l424_424810

theorem min_value_inverse_sum {m n : ℝ} (h1 : -2 * m - 2 * n + 1 = 0) (h2 : m * n > 0) : 
  (1 / m + 1 / n) ≥ 8 :=
sorry

end min_value_inverse_sum_l424_424810


namespace number_of_increasing_digits_l424_424164

theorem number_of_increasing_digits : 
  (∑ k in finset.range 10, if 2 ≤ k then nat.choose 9 k else 0) = 502 :=
by
  sorry

end number_of_increasing_digits_l424_424164


namespace find_radius_and_AC_l424_424544

-- Definitions of given conditions
variables (A B C K L M N I : Type) [Point A] [Point B] [Point C] [Point K] [Point L] [Point M] [Point N] [Point I]
variables (Γ : Circle) (r : ℝ)
variables (angleB : B = 90)
variables (AB BC AC : Segment)
variables (cir_center_I : Center I Γ)
variables (inscribed_in_ABC : InscribedCircle Γ ABC)
variables (touches_AB_at_K : Touches Γ AB K)
variables (touches_BC_at_L : Touches Γ BC L)
variables (line_through_I_MN : LineThroughPoints I M N)
variables (MN_parallel_AC : ParallelLine MN AC)
variables (MK_144 : SegmentLength M K = 144)
variables (NL_25 : SegmentLength N L = 25)

-- The theorem to prove the questions given conditions
theorem find_radius_and_AC : r = 60 ∧ SegmentLength A C = 390 :=
begin
  sorry,  -- Proof not required as per the instructions
end

end find_radius_and_AC_l424_424544


namespace books_total_pages_l424_424786

theorem books_total_pages (x y z : ℕ) 
  (h1 : (2 / 3 : ℚ) * x - (1 / 3 : ℚ) * x = 20)
  (h2 : (3 / 5 : ℚ) * y - (2 / 5 : ℚ) * y = 15)
  (h3 : (3 / 4 : ℚ) * z - (1 / 4 : ℚ) * z = 30) : 
  x = 60 ∧ y = 75 ∧ z = 60 :=
by
  sorry

end books_total_pages_l424_424786


namespace correct_option_is_A_l424_424907

def is_quadratic_equation_with_one_variable (eq : nat → nat → nat → Prop) :=
  ∃ a b c : nat, a ≠ 0 ∧ eq a b c

def Option_A (a b c : nat) : Prop := (-a) = -1 ∧ b = 4 ∧ c = 0

theorem correct_option_is_A :
  is_quadratic_equation_with_one_variable Option_A :=
by
  sorry

end correct_option_is_A_l424_424907


namespace Allison_greater_probability_l424_424114

open ProbabilityTheory

/-- The probability problem -/
theorem Allison_greater_probability :
  let AllisonRoll := 6
  let CharlieRoll := {1, 1, 2, 2, 3, 3}
  let EmmaRoll := {3, 3, 3, 3, 5, 5}
  (1 : ℚ) * (4 / 6 : ℚ) = (2 / 3 : ℚ) :=
by
  let AllisonRoll := 6
  let CharlieRoll := {1, 1, 2, 2, 3, 3}
  let EmmaRoll := {3, 3, 3, 3, 5, 5}
  have hCharlie : (1 : ℚ) = 1 := by sorry
  have hEmma : (4 / 6 : ℚ) = (2 / 3 : ℚ) := by sorry
  show (1 : ℚ) * (4 / 6 : ℚ) = (2 / 3 : ℚ) from 
    by rw [hCharlie, hEmma]


end Allison_greater_probability_l424_424114


namespace polynomial_real_roots_l424_424138

theorem polynomial_real_roots (n : ℕ) 
  (h1 : n ≥ 4) 
  (α β : Fin n → ℝ) 
  (h2 : (∑ i, α i ^ 2) < 1)
  (h3 : (∑ i, β i ^ 2) < 1) :
  let A : ℝ := Real.sqrt (1 - ∑ i, α i ^ 2)
  let B : ℝ := Real.sqrt (1 - ∑ i, β i ^ 2)
  let W : ℝ := 1 / 2 * (1 - ∑ i, α i * β i) ^ 2
  ∀ λ : ℝ, 
    (∀ x : ℝ, is_root (x ^ n + λ * (∑ j in Fin.range (n - 3), x ^ j + W * x ^ 2 + A * B * x + 1) = 0) x) 
    ↔ λ = 0 :=
sorry

end polynomial_real_roots_l424_424138


namespace find_N_l424_424739

theorem find_N (N : ℕ) (r : ℝ) (π : ℝ) :
  let A := N * (π * r^2 / 2),
      B := (π * (N * r)^2 / 2) - A in
  A / B = 1 / 24 → N = 25 :=
  sorry

end find_N_l424_424739


namespace rotate_90deg_l424_424908

def Shape := Type

structure Figure :=
(triangle : Shape)
(circle : Shape)
(square : Shape)
(pentagon : Shape)

def rotated_position (fig : Figure) : Figure :=
{ triangle := fig.circle,
  circle := fig.square,
  square := fig.pentagon,
  pentagon := fig.triangle }

theorem rotate_90deg (fig : Figure) :
  rotated_position fig = { triangle := fig.circle,
                           circle := fig.square,
                           square := fig.pentagon,
                           pentagon := fig.triangle } :=
by {
  sorry
}

end rotate_90deg_l424_424908


namespace sum_f_1023_l424_424188

/-- Define highest power of 2 dividing n! -/
def f (n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), n / 2^k

/-- Define the sum f(1) + f(2) + ... + f(1023) -/
theorem sum_f_1023 : (finset.range 1024).sum f = 518656 :=
sorry

end sum_f_1023_l424_424188


namespace max_positive_integers_l424_424784

theorem max_positive_integers (a : ℕ → ℤ) (h_cycle : ∀ k, a (k + 2018) = a k)
  (h_condition : ∀ k, a (k + 1) > a k + a (k - 1)) :
  ∃ (S : ℕ → ℤ) (hS : ∀ k, 0 ≤ S k) (h_sum : (S ∘ a).sum = 1008), True :=
begin
  sorry -- The proof is not required
end

end max_positive_integers_l424_424784


namespace solution_l424_424318

noncomputable def y : ℝ := (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) * ⋯ * (Real.log 32 / Real.log 31)

theorem solution : y = 5 := sorry

end solution_l424_424318


namespace molecular_weight_N2O_l424_424597

theorem molecular_weight_N2O :
  ∀ (atomic_weight_N atomic_weight_O : ℝ),
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  2 * atomic_weight_N + atomic_weight_O = 44.02 :=
by
  intros atomic_weight_N atomic_weight_O hN hO
  rw [hN, hO]
  simp
  norm_num
  sorry

end molecular_weight_N2O_l424_424597


namespace count_strictly_increasing_digits_l424_424161

theorem count_strictly_increasing_digits : 
  (∑ k in Finset.range 9, Nat.choose 9 k.succ) = 502 :=
by
  sorry

end count_strictly_increasing_digits_l424_424161


namespace supervisors_per_bus_l424_424449

theorem supervisors_per_bus (total_supervisors : ℕ) (total_buses : ℕ) (H1 : total_supervisors = 21) (H2 : total_buses = 7) : (total_supervisors / total_buses = 3) :=
by
  sorry

end supervisors_per_bus_l424_424449


namespace intersect_horizontal_asymptote_l424_424146

theorem intersect_horizontal_asymptote (x : ℚ) :
  let g := (3 * x^2 - 8 * x + 4) / (x^2 - 5 * x + 6) in
  g = 3 ↔ x = 2 :=
by
  sorry

end intersect_horizontal_asymptote_l424_424146


namespace projection_b_on_a_l424_424220

variables (a b : ℝ^3) 

def magnitude (x : ℝ^3) : ℝ := real.sqrt (x • x)

noncomputable def projection (u v : ℝ^3) : ℝ :=
  (u • v) / (magnitude v)

theorem projection_b_on_a (h1 : magnitude a = 2) (h2 : magnitude b = 3) (h3 : (2 • a + b) • (a - 2 • b) = 0) :
  projection b a = -5 / 3 :=
sorry

end projection_b_on_a_l424_424220


namespace ratio_of_buttons_to_magnets_per_earring_l424_424795

-- Definitions related to the problem statement
def gemstones_per_button : ℕ := 3
def magnets_per_earring : ℕ := 2
def sets_of_earrings : ℕ := 4
def required_gemstones : ℕ := 24

-- Problem statement translation into Lean 4
theorem ratio_of_buttons_to_magnets_per_earring :
  (required_gemstones / gemstones_per_button / (sets_of_earrings * 2)) = 1 / 2 := by
  sorry

end ratio_of_buttons_to_magnets_per_earring_l424_424795


namespace triangle_geom_problem_l424_424766

theorem triangle_geom_problem
  {A B C D F G H I H' I' Q M P : Type}
  (CA_ne_CB : CA ≠ CB)
  (midpoints : (midpoint D A B) ∧ (midpoint F A C) ∧ (midpoint G B C))
  (circle_Gamma : (passes_through C Γ) ∧ (tangent_to Γ AB D))
  (intersections : (Γ ∩ AF = {H}) ∧ (Γ ∩ BG = {I}))
  (symmetric : (symmetric_to H' H F) ∧ (symmetric_to I' I G))
  (line_intersections : (line H'I' ∩ CD = {Q}) ∧ (line H'I' ∩ FG = {M}))
  (line_CM : line_intersection (line CM) Γ P)
  : distance C Q = distance Q P := 
sorry

end triangle_geom_problem_l424_424766


namespace factory_costs_cost_comparison_200_l424_424021

noncomputable def cost_factory_A (x : ℕ) : ℕ :=
  4.8 * x + 500

noncomputable def cost_factory_B (x : ℕ) : ℕ :=
  6 * x + 200

theorem factory_costs (x : ℕ) :
  cost_factory_A x = 4.8 * x + 500 ∧ cost_factory_B x = 6 * x + 200 := by
  sorry

theorem cost_comparison_200 :
  let cost_A := cost_factory_A 200
  let cost_B := cost_factory_B 200
  cost_A = 1460 ∧ cost_B = 1400 ∧ cost_B < cost_A := by
  sorry

end factory_costs_cost_comparison_200_l424_424021


namespace vector_representation_exists_l424_424582

-- Define the vector a
def vector_a : ℝ × ℝ := (-3, 7)

-- Define groups of vectors
def group_A_e1 : ℝ × ℝ := (0, 1)
def group_A_e2 : ℝ × ℝ := (0, -2)

def group_B_e1 : ℝ × ℝ := (1, 5)
def group_B_e2 : ℝ × ℝ := (-2, -10)

def group_C_e1 : ℝ × ℝ := (-5, 3)
def group_C_e2 : ℝ × ℝ := (-2, 1)

def group_D_e1 : ℝ × ℝ := (7, 8)
def group_D_e2 : ℝ × ℝ := (-7, -8)

-- The proof problem statement
theorem vector_representation_exists : 
  (∃ λ μ : ℝ, vector_a = (λ • group_A_e1 + μ • group_A_e2)) ∨ 
  (∃ λ μ : ℝ, vector_a = (λ • group_B_e1 + μ • group_B_e2)) ∨ 
  (∃ λ μ : ℝ, vector_a = (λ • group_C_e1 + μ • group_C_e2)) ∨ 
  (∃ λ μ : ℝ, vector_a = (λ • group_D_e1 + μ • group_D_e2)) :=
sorry

end vector_representation_exists_l424_424582


namespace jacket_price_after_noon_l424_424552

theorem jacket_price_after_noon :
  ∀ (total_jackets : ℕ) (price_before_noon : ℝ) (jackets_sold_after_noon : ℕ) (total_receipts : ℝ),
  total_jackets = 214 →
  price_before_noon = 31.95 →
  jackets_sold_after_noon = 133 →
  total_receipts = 5108.30 →
  let jackets_sold_before_noon := total_jackets - jackets_sold_after_noon,
      receipts_before_noon := (jackets_sold_before_noon : ℝ) * price_before_noon,
      receipts_after_noon := total_receipts - receipts_before_noon,
      price_after_noon := receipts_after_noon / (jackets_sold_after_noon : ℝ)
  in price_after_noon = 18.95 := by 
  intros total_jackets price_before_noon jackets_sold_after_noon total_receipts 
  sorry

end jacket_price_after_noon_l424_424552


namespace P_ge_n_minus_3_P_le_2n_minus_7_P_le_2n_minus_10_l424_424082

-- Definition of P(n) as the minimum number of transformations required to convert any subdivision into any other
def P (n : ℕ) : ℕ := sorry -- Placeholder definition

-- Condition: We have a convex n-gon and assumptions stated in the problem above (such as non-intersecting diagonals)
-- and P(n) as defined before.

theorem P_ge_n_minus_3 (n : ℕ) (h : n ≥ 3) : P(n) ≥ n - 3 := by sorry

theorem P_le_2n_minus_7 (n : ℕ) (h : n ≥ 3) : P(n) ≤ 2n - 7 := by sorry

theorem P_le_2n_minus_10 (n : ℕ) (h : n ≥ 13) : P(n) ≤ 2n - 10 := by sorry

end P_ge_n_minus_3_P_le_2n_minus_7_P_le_2n_minus_10_l424_424082


namespace quotient_of_sum_of_distinct_remainders_is_three_l424_424429

theorem quotient_of_sum_of_distinct_remainders_is_three :
  let remainders := {1^2 % 13, 2^2 % 13, 3^2 % 13, 4^2 % 13, 5^2 % 13, 6^2 % 13, 7^2 % 13, 8^2 % 13, 
                      9^2 % 13, 10^2 % 13, 11^2 % 13, 12^2 % 13, 13^2 % 13, 14^2 % 13, 15^2 % 13}.toFinset in
  let m := remainders.sum in
  m / 13 = 3 := 
by
  sorry

end quotient_of_sum_of_distinct_remainders_is_three_l424_424429


namespace books_needed_to_buy_clarinet_l424_424434

def cost_of_clarinet : ℕ := 90
def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def halfway_loss : ℕ := (cost_of_clarinet - initial_savings) / 2

theorem books_needed_to_buy_clarinet 
    (cost_of_clarinet initial_savings price_per_book halfway_loss : ℕ)
    (initial_savings_lost : halfway_loss = (cost_of_clarinet - initial_savings) / 2) : 
    ((cost_of_clarinet - initial_savings + halfway_loss) / price_per_book) = 24 := 
sorry

end books_needed_to_buy_clarinet_l424_424434


namespace prob_red_or_black_prob_red_black_or_white_l424_424539

noncomputable def ball_colors := {red: 5, black: 4, white: 2, green: 1}
noncomputable def total_balls := 12
noncomputable def favorable_red_or_black := 9
noncomputable def favorable_red_black_or_white := 11

theorem prob_red_or_black : (favorable_red_or_black : ℚ) / total_balls = 3 / 4 :=
by
  sorry

theorem prob_red_black_or_white : (favorable_red_black_or_white : ℚ) / total_balls = 11 / 12 :=
by
  sorry

end prob_red_or_black_prob_red_black_or_white_l424_424539


namespace op_comm_l424_424994

def op (a b : ℝ) : ℝ := a^2 + a * b + b^2

theorem op_comm (x y : ℝ) : op x y = op y x := 
by
  unfold op
  ring
  sorry

end op_comm_l424_424994


namespace shopping_money_l424_424923

theorem shopping_money (X : ℝ) (h : 0.70 * X = 840) : X = 1200 :=
sorry

end shopping_money_l424_424923


namespace diagonal_difference_l424_424073

def matrix_initial : List (List Nat) := 
[[1, 2, 3, 4, 5], 
 [11, 12, 13, 14, 15], 
 [21, 22, 23, 24, 25], 
 [31, 32, 33, 34, 35], 
 [41, 42, 43, 44, 45]]

def matrix_reversed : List (List Nat) := 
[[1, 2, 3, 4, 5], 
 [15, 14, 13, 12, 11], 
 [25, 24, 23, 22, 21], 
 [31, 32, 33, 34, 35], 
 [45, 44, 43, 42, 41]]

def main_diagonal_sum (m : List (List Nat)) : Nat :=
  (m.get! 0).get! 0 + (m.get! 1).get! 1 + (m.get! 2).get! 2 + (m.get! 3).get! 3 + (m.get! 4).get! 4

def secondary_diagonal_sum (m : List (List Nat)) : Nat :=
  (m.get! 0).get! 4 + (m.get! 1).get! 3 + (m.get! 2).get! 2 + (m.get! 3).get! 1 + (m.get! 4).get! 0

theorem diagonal_difference : 
  abs (main_diagonal_sum matrix_reversed - secondary_diagonal_sum matrix_reversed) = 4 := 
by
  sorry

end diagonal_difference_l424_424073


namespace number_of_9s_in_1_to_100_l424_424105

theorem number_of_9s_in_1_to_100 : ∀ n, 1 ≤ n ∧ n ≤ 100 → count_digit 9 n = 19 :=
by
  -- Proof here
  sorry

end number_of_9s_in_1_to_100_l424_424105


namespace f_properties_l424_424240

noncomputable theory

variable {f : ℝ → ℝ}

-- Define the conditions
axiom f_symmetry_2 : ∀ x, f (2 - x) = f (2 + x)
axiom f_symmetry_7 : ∀ x, f (7 - x) = f (7 + x)
axiom f_at_zero : f 0 = 0

-- State the main proof problem
theorem f_properties :
  (∃ p, ∀ x, f (x + p) = f x) ∧
  (∃ z : list ℝ, z.length ≥ 13 ∧ ∀ x ∈ z, x ∈ Icc (-30 : ℝ) 30 ∧ f x = 0) :=
sorry

end f_properties_l424_424240


namespace nat_pair_count_eq_five_l424_424287

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424287


namespace cubic_poly_prime_l424_424193

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m ∣ p, m = 1 ∨ m = p

theorem cubic_poly_prime (n : ℕ) (h : n > 0) : ¬ is_prime (n^3 - 9 * n^2 + 19 * n - 13) :=
sorry

end cubic_poly_prime_l424_424193


namespace max_real_root_lt_100_l424_424790

theorem max_real_root_lt_100 (k a b c : ℕ) (r : ℝ)
  (ha : ∃ m : ℕ, a = k^m)
  (hb : ∃ n : ℕ, b = k^n)
  (hc : ∃ l : ℕ, c = k^l)
  (one_real_solution : b^2 = 4 * a * c)
  (r_is_root : ∃ r : ℝ, a * r^2 - b * r + c = 0)
  (r_lt_100 : r < 100) :
  r ≤ 64 := sorry

end max_real_root_lt_100_l424_424790


namespace number_of_pairs_count_number_of_pairs_l424_424304

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424304


namespace circumference_of_circle_inscribing_rectangle_l424_424961

theorem circumference_of_circle_inscribing_rectangle (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi := by
  sorry

end circumference_of_circle_inscribing_rectangle_l424_424961


namespace initial_geese_count_l424_424576

theorem initial_geese_count (G : ℕ) (h1 : G / 2 + 4 = 12) : G = 16 := by
  sorry

end initial_geese_count_l424_424576


namespace length_of_path_F_travelled_l424_424121

-- Given certain geometric properties about semi-circles and rolling motion.
def semi_circle_center (D E F : Point) := 
  is_center F (semi_circle D E) 

def initial_position (F : Point) : ℝ := 
  EF_distance = 4 / π -- EF = \frac{4}{\pi} cm

def path_length (F : Point) (RS : Line) : ℝ :=
  rolling_length DEF RS = 8 -- length of path point F travels

theorem length_of_path_F_travelled 
  (D E F : Point) (RS : Line)
  (semi_circle_center D E F) -- Arc DE is a semi-circle with center F
  (initial_position F)  -- EF = \frac{4}{\pi} cm
:
  path_length F RS = 8 := sorry

end length_of_path_F_travelled_l424_424121


namespace polygon_sides_eq_six_l424_424012

theorem polygon_sides_eq_six (n : ℕ) (S_i S_e : ℕ) :
  S_i = 2 * S_e →
  S_e = 360 →
  (n - 2) * 180 = S_i →
  n = 6 :=
by
  sorry

end polygon_sides_eq_six_l424_424012


namespace x_intercept_of_line_is_six_l424_424888

theorem x_intercept_of_line_is_six : ∃ x : ℝ, (∃ y : ℝ, y = 0) ∧ (2*x - 4*y = 12) ∧ x = 6 :=
by {
  sorry
}

end x_intercept_of_line_is_six_l424_424888


namespace no_such_function_exists_l424_424754

noncomputable def is_fixpoint_of_iterates (T : ℤ → ℤ) (n : ℕ) (x : ℤ) : Prop :=
  (nat.iterate T n x = x)

noncomputable def number_of_fixpoints (T : ℤ → ℤ) (n : ℕ) : ℕ :=
  (Finset.univ.filter (λ x, is_fixpoint_of_iterates T n x)).card

theorem no_such_function_exists (P : Polynomial ℤ) (h : ¬ P.degree = 0) :
  ¬ ∃ (T : ℤ → ℤ), ∀ n : ℕ, 1 ≤ n → number_of_fixpoints T n = P.eval (n : ℤ) :=
sorry

end no_such_function_exists_l424_424754


namespace strictly_increasing_seqs_count_l424_424170

theorem strictly_increasing_seqs_count : 
  ∑ k in (finset.range 9).filter (λ k, k ≥ 2), nat.choose 9 k = 502 := by
  sorry

end strictly_increasing_seqs_count_l424_424170


namespace identify_spy_l424_424525

variable (Knight Liar Spy : Type)
variable {A B C : Knight ⊕ Liar ⊕ Spy}

-- Definitions of roles
def is_knight (x : Knight ⊕ Liar ⊕ Spy) : Prop := 
  ∃ (k : Knight), (x = sum.inl k)

def is_liar (x : Knight ⊕ Liar ⊕ Spy) : Prop := 
  ∃ (l : Liar), (x = sum.inr (sum.inl l))

def is_spy (x : Knight ⊕ Liar ⊕ Spy) : Prop := 
  ∃ (s : Spy), (x = sum.inr (sum.inr s))

-- Given conditions encoded
axiom initially_assigned_roles : 
  (is_knight A ∧ is_liar B ∧ is_spy C) ∨ (is_knight A ∧ is_spy B ∧ is_liar C) ∨ 
  (is_liar A ∧ is_knight B ∧ is_spy C) ∨ (is_spy A ∧ is_knight B ∧ is_liar C) ∨ 
  (is_liar A ∧ is_spy B ∧ is_knight C) ∨ (is_spy A ∧ is_liar B ∧ is_knight C)

-- Judge releases one of the defendants proving they are not the spy
axiom judge_releases_not_spy : 
  (¬ is_spy C)

-- After releasing, judge asks if the neighbor is a spy, and concludes one is a spy
theorem identify_spy : 
  (is_spy B) :=
sorry

end identify_spy_l424_424525


namespace ratio_of_x_intercepts_l424_424882

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l424_424882


namespace count_pairs_satisfying_condition_l424_424270

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424270


namespace estimate_probability_of_hitting_target_at_least_3_times_l424_424561

-- Define the condition for a shot hitting the target
def hits_target (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 9

-- Define the condition for a group of 4 shots hitting the target at least 3 times
def hits_target_at_least_3 (group : List ℕ) : Prop :=
  (group.filter hits_target).length ≥ 3

-- Given the 20 groups of results
def groups : List (List ℕ) := [
  [7, 5, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7], [0, 3, 4, 7],
  [4, 3, 7, 3], [8, 6, 3, 6], [6, 9, 4, 7], [1, 4, 1, 7], [4, 6, 9, 8],
  [0, 3, 7, 1], [6, 2, 3, 3], [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1],
  [3, 6, 6, 1], [9, 5, 9, 7], [7, 4, 2, 4], [7, 6, 1, 0], [4, 2, 8, 1]
]

-- State the theorem
theorem estimate_probability_of_hitting_target_at_least_3_times :
  (groups.filter hits_target_at_least_3).length = 15 →
  (groups.filter hits_target_at_least_3).length / groups.length = 0.75 :=
by
  sorry

end estimate_probability_of_hitting_target_at_least_3_times_l424_424561


namespace num_increasing_digits_l424_424181

theorem num_increasing_digits :
  let C := λ (n k : ℕ), Nat.choose n k in
  ∑ k in Finset.range 8, C 9 (k + 2) = 502 :=
by
  sorry

end num_increasing_digits_l424_424181


namespace count_strictly_increasing_digits_l424_424157

theorem count_strictly_increasing_digits : 
  (∑ k in Finset.range 9, Nat.choose 9 k.succ) = 502 :=
by
  sorry

end count_strictly_increasing_digits_l424_424157


namespace solve_system_l424_424196

theorem solve_system : ∀ (x y : ℤ), 2 * x + y = 5 → x + 2 * y = 6 → x - y = -1 :=
by
  intros x y h1 h2
  sorry

end solve_system_l424_424196


namespace people_per_seat_l424_424935

theorem people_per_seat (total_people seats : ℕ) (h1 : total_people = 16) (h2 : seats = 4) : total_people / seats = 4 :=
by 
  rw [h1, h2]
  norm_num

end people_per_seat_l424_424935


namespace fraction_of_males_l424_424591

theorem fraction_of_males (M F : ℚ) (h1 : M + F = 1)
  (h2 : (3/4) * M + (5/6) * F = 7/9) :
  M = 2/3 :=
by sorry

end fraction_of_males_l424_424591


namespace convex_polyhedron_properties_l424_424639

theorem convex_polyhedron_properties
  (b : ℝ)
  (h : ∀ (T : Type) [regular_tetrahedron T], ∀ (U : Type), vertices_divide_each_edge_into_three_equal_parts T U)
  (e : edge_length_polyhedron U = b) :
  (∀ (d : ℝ), body_diagonal_length U d → d = b * sqrt 5) ∧
  (∀ (d : ℝ), distance_to_centroid U d → d = b * sqrt 2 / 4) ∧
  (∃ (n : ℕ), body_diagonal_intersections U = 30) :=
by
  sorry

end convex_polyhedron_properties_l424_424639


namespace julia_more_kids_on_Monday_l424_424750

def kids_played_on_Tuesday : Nat := 14
def kids_played_on_Monday : Nat := 22

theorem julia_more_kids_on_Monday : kids_played_on_Monday - kids_played_on_Tuesday = 8 :=
by {
  sorry
}

end julia_more_kids_on_Monday_l424_424750


namespace log_sum_geometric_sequence_l424_424191

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

theorem log_sum_geometric_sequence {a : ℕ → ℝ} 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, 0 < a n)
  (h_condition : a 5 * a 6 + a 4 * a 7 = 18) :
  ∑ i in finset.range 10, Real.logBase 3 (a (i + 1)) = 10 := 
sorry

end log_sum_geometric_sequence_l424_424191


namespace chessboard_coloring_l424_424183

theorem chessboard_coloring (n : ℕ) (board_size : ℕ) (colored_squares : set (ℕ × ℕ)) :
  board_size = 1000 →
  (∀ (x y : ℕ), x < board_size ∧ y < board_size → (x, y) ∈ colored_squares → x < board_size ∧ y < board_size) →
  colored_squares.card = n →
  n ≥ 1999 →
  ∃ (a b c : ℕ × ℕ),
    a ∈ colored_squares ∧
    b ∈ colored_squares ∧
    c ∈ colored_squares ∧
    (a.1 = b.1 ∨ a.2 = b.2) ∧
    (b.1 = c.1 ∨ b.2 = c.2) ∧
    (a.1 = c.1 ∨ a.2 = c.2) ∧
    ((a.1 = b.1 ∧ b.2 = c.2) ∨ (a.2 = b.2 ∧ b.1 = c.1)) :=
begin
  sorry -- Proof not required as per instructions
end

end chessboard_coloring_l424_424183


namespace distance_between_points_l424_424890

theorem distance_between_points : 
  let p1 : ℝ × ℝ × ℝ := (3, -2, 1)
  let p2 : ℝ × ℝ × ℝ := (8, 4, 3)
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2) = Real.sqrt 65) :=
by
  let p1 : ℝ × ℝ × ℝ := (3, -2, 1)
  let p2 : ℝ × ℝ × ℝ := (8, 4, 3)
  have h : Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2) = Real.sqrt 65 := sorry
  exact h

end distance_between_points_l424_424890


namespace find_q_l424_424207

noncomputable def Sn (n : ℕ) (d : ℚ) : ℚ :=
  d^2 * (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def Tn (n : ℕ) (d : ℚ) (q : ℚ) : ℚ :=
  d^2 * (1 - q^n) / (1 - q)

theorem find_q (d : ℚ) (q : ℚ) (hd : d ≠ 0) (hq : 0 < q ∧ q < 1) :
  Sn 3 d / Tn 3 d q = 14 → q = 1 / 2 :=
by
  sorry

end find_q_l424_424207


namespace range_of_a_l424_424800

theorem range_of_a (m : ℝ) (a : ℝ) : 
  m ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  (∀ x₁ x₂ : ℝ, x₁^2 - m * x₁ - 2 = 0 ∧ x₂^2 - m * x₂ - 2 = 0 → a^2 - 5 * a - 3 ≥ |x₁ - x₂|) ↔ (a ≥ 6 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l424_424800


namespace geometric_sequence_fifth_term_l424_424856

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 16) (h2 : a * r^6 = 2) : a * r^4 = 2 :=
sorry

end geometric_sequence_fifth_term_l424_424856


namespace part1_part2_l424_424769

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (|x - 1| ≤ 2) ∧ ((x + 3) / (x - 2) ≥ 0)

-- Part 1
theorem part1 (h_a : a = 1) (h_p : p a x) (h_q : q x) : 2 < x ∧ x < 3 := sorry

-- Part 2
theorem part2 (h_suff : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := sorry

end part1_part2_l424_424769


namespace count_pairs_satisfying_condition_l424_424294

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424294


namespace area_BCEF_l424_424529

noncomputable def Point := (ℝ × ℝ)
def A : Point := (0, 0)
def B : Point := (2, 0)
def C : Point := (2, 2)
def D : Point := (0, 2)
def E : Point := (1, 0) -- E is on AB such that AE = 1
def F : Point := (1 / (sqrt 2), 1 / (sqrt 2)) -- F is on AC such that AF = 1

def area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

def area_quadrilateral (p1 p2 p3 p4 : Point) : ℝ :=
  area p1 p2 p3 + area p1 p3 p4

theorem area_BCEF : area_quadrilateral B C E F = (5 - sqrt 2) / 2 :=
  by 
  sorry

end area_BCEF_l424_424529


namespace midpoints_form_regular_dodecagon_l424_424991

def point := (ℝ × ℝ)

def square_vertices : List point := [(1, 1), (-1, 1), (-1, -1), (1, -1)]

def equilateral_triangle (A B : point) (k : ℝ) : point :=
  ((A.1 + B.1) / 2, (A.2 - B.2 + k))

def midpoint (P Q : point) : point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def midpoints_for_segments(A B C D K L M N : point) : List point :=
  [midpoint K L, midpoint L M, midpoint M N, midpoint N K,
   midpoint A K, midpoint B K, midpoint B L, midpoint C L,
   midpoint C M, midpoint D M, midpoint D N, midpoint A N]

theorem midpoints_form_regular_dodecagon (A B C D K L M N : point) (k h j : ℝ) :
  A ∈ square_vertices ∧
  B ∈ square_vertices ∧
  C ∈ square_vertices ∧
  D ∈ square_vertices ∧
  K = equilateral_triangle A B k ∧
  L = equilateral_triangle B C k ∧
  M = equilateral_triangle C D k ∧
  N = equilateral_triangle D A k ∧
  midpoints_for_segments A B C D K L M N = [(h, j), (-h, j), (-j, h), (-j, -h),
                                             (-h, -j), (h, -j), (j, -h), (j, h)] ->
  -- Placeholder that the points form a regular dodecahedron
  -- This part of the theorem will need formalization and proof
  sorry

end midpoints_form_regular_dodecagon_l424_424991


namespace compare_abc_l424_424932

noncomputable def a : ℝ := Real.logBase 0.3 2
noncomputable def b : ℝ := Real.ln 2
noncomputable def c : ℝ := 0.25 ^ (-0.5)

theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l424_424932


namespace elena_subtracts_99_to_compute_49_squared_l424_424486

noncomputable def difference_between_squares_50_49 : ℕ := 99

theorem elena_subtracts_99_to_compute_49_squared :
  ∀ (n : ℕ), n = 50 → (n - 1)^2 = n^2 - difference_between_squares_50_49 :=
by
  intro n
  sorry

end elena_subtracts_99_to_compute_49_squared_l424_424486


namespace integral_x2_e3x_l424_424063

theorem integral_x2_e3x : (∫ x in 0..1, x^2 * exp(3 * x)) = (5 * exp(3) - 2) / 27 := by
  sorry

end integral_x2_e3x_l424_424063


namespace count_false_propositions_l424_424843

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

end count_false_propositions_l424_424843


namespace find_m_n_sum_l424_424753

variables (a b : ℝ) (x y : ℝ)
variables (m n : ℕ)

-- Given conditions
def conditions := a > 0 ∧ b > 0 ∧ a ≥ b ∧ 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧
  (a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2)

-- The theorem to prove
theorem find_m_n_sum (h : conditions a b x y) : 
  let ρ := 2 / real.sqrt 3 in
  let ρ2 := ρ * ρ in
  let (m, n) := (4, 3) in
  m.nat_abs.gcd n.nat_abs = 1 ∧ ρ2 = (m : ℝ) / (n : ℝ) →
  m + n = 7 :=
by { simp [ρ, ρ2], sorry }

end find_m_n_sum_l424_424753


namespace compare_exponents_l424_424528

theorem compare_exponents : 
  let a := (1 / 2)^(1 / 4)
  let b := (1 / 3)^(1 / 2)
  let c := (1 / 4)^(1 / 3)
  in b < c ∧ c < a :=
by
  let a := (1 / 2)^(1 / 4)
  let b := (1 / 3)^(1 / 2)
  let c := (1 / 4)^(1 / 3)
  sorry

end compare_exponents_l424_424528


namespace number_of_handshakes_l424_424860

theorem number_of_handshakes :
  ∃ (n : ℕ),
  let num_companies := 3
  let reps_per_company := 5
  let total_people := num_companies * reps_per_company
  let handshakes := (total_people * (total_people - 1)) / 2 - 
                    (reps_per_company * (reps_per_company - 1) / 2 + 
                     reps_per_company * (reps_per_company - 1) / 2 + 
                     reps_per_company * (reps_per_company - 1) / 2) - 
                    (reps_per_company * reps_per_company)
  in n = 63 :=
by
  let num_companies := 3
  let reps_per_company := 5
  let total_people := num_companies * reps_per_company
  let handshakes := (total_people * (total_people - 1)) / 2 - 
                    (reps_per_company * (reps_per_company - 1) / 2 + 
                     reps_per_company * (reps_per_company - 1) / 2 + 
                     reps_per_company * (reps_per_company - 1) / 2) - 
                    (reps_per_company * reps_per_company)
  use handshakes
  sorry

end number_of_handshakes_l424_424860


namespace monotonic_decreasing_interval_l424_424002

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 (Real.sqrt 3 / 3), (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l424_424002


namespace even_sum_probability_l424_424321

theorem even_sum_probability :
  let S := {1, 2, 3, 4, 5}
  let even_pairs := {(2, 4)}
  let odd_pairs := {(1, 3), (1, 5), (3, 5)}
  let total_pairs := 5 * 4 / 2
  Prob (2 ∈ S ∧ 4 ∈ S ∨ 1 ∈ S ∧ 3 ∈ S ∨ 1 ∈ S ∧ 5 ∈ S ∨ 3 ∈ S ∧ 5 ∈ S) = (4 / 10) :=
by
  sorry

end even_sum_probability_l424_424321


namespace continuity_of_f_l424_424768

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
if x ≤ 4 then 2 * x^2 + 5
else if x ≤ 6 then b * x + 3
else c * x^2 - 2 * x + 9

theorem continuity_of_f (b c : ℝ) :
  (f 4 b c = 2 * 4^2 + 5) ∧
  (f 4 b c = b * 4 + 3) ∧
  (f 6 b c = b * 6 + 3) ∧
  (f 6 b c = c * 6^2 - 2 * 6 + 9) → 
  b = 8.5 ∧ c = 19 / 12 :=
begin 
  sorry 
end

end continuity_of_f_l424_424768


namespace machine_transport_equation_l424_424388

theorem machine_transport_equation (x : ℝ) :
  (∀ (rateA rateB : ℝ), rateB = rateA + 60 → (500 / rateA = 800 / rateB) → rateA = x → rateB = x + 60) :=
by
  sorry

end machine_transport_equation_l424_424388


namespace machine_A_produces_1_sprockets_per_hour_l424_424519

namespace SprocketsProduction

variable {A T : ℝ} -- A: sprockets per hour of machine A, T: hours it takes for machine Q to produce 110 sprockets

-- Given conditions
axiom machine_Q_production_rate : 110 / T = 1.10 * A
axiom machine_P_production_rate : 110 / (T + 10) = A

-- The target theorem to prove
theorem machine_A_produces_1_sprockets_per_hour (h1 : 110 / T = 1.10 * A) (h2 : 110 / (T + 10) = A) : A = 1 :=
by sorry

end SprocketsProduction

end machine_A_produces_1_sprockets_per_hour_l424_424519


namespace find_number_l424_424316

def x : ℝ := 33.75

theorem find_number (x: ℝ) :
  (0.30 * x = 0.25 * 45) → x = 33.75 :=
by
  sorry

end find_number_l424_424316


namespace distance_between_parallel_lines_l424_424687

theorem distance_between_parallel_lines (a : ℝ)
  (h1 : (3 : ℝ) * x + 4 * y - 4 = 0)
  (h2 : a * x + 8 * y + 2 = 0)
  (ha : a = 6)
  (parallels : (3 : ℝ) / 4 = a / 8) :
  let l1_c := -4 in
  let l2_c := -(a * 1 / 2) in
  let distance := (abs (l2_c - l1_c)) / (sqrt ((3 : ℝ) ^ 2 + (4 : ℝ) ^ 2)) in
  distance = 1 := by
begin
  sorry
end

end distance_between_parallel_lines_l424_424687


namespace heights_equal_l424_424819

-- Define base areas and volumes
variables {V : ℝ} {S : ℝ}

-- Assume equal volumes and base areas for the prism and cylinder
variables (h_prism h_cylinder : ℝ) (volume_eq : V = S * h_prism) (base_area_eq : S = S)

-- Define a proof goal
theorem heights_equal 
  (equal_volumes : V = S * h_prism) 
  (equal_base_areas : S = S) : 
  h_prism = h_cylinder :=
sorry

end heights_equal_l424_424819


namespace line_through_parabola_no_intersection_l424_424763

-- Definitions of the conditions
def parabola (x : ℝ) : ℝ := x^2 
def point_Q := (10, 5)

-- The main theorem statement
theorem line_through_parabola_no_intersection :
  ∃ r s : ℝ, (∀ (m : ℝ), (r < m ∧ m < s) ↔ ¬ ∃ x : ℝ, parabola x = m * (x - 10) + 5) ∧ r + s = 40 :=
sorry

end line_through_parabola_no_intersection_l424_424763


namespace investment_comparison_l424_424948

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem investment_comparison :
  let P := 100
  let r1 := 0.10
  let r2 := 0.09
  let t := 5
  let SI := simple_interest P r1 t
  let CI := compound_interest P r2 t
  SI = 150 ∧ CI ≈ 153.86 ∧ CI > SI ∧ CI - SI ≈ 3.86 :=
by {
  let P := 100
  let r1 := 0.10
  let r2 := 0.09
  let t := 5
  let SI := simple_interest P r1 t
  let CI := compound_interest P r2 t
  have h1 : SI = 150, from sorry,
  have h2 : CI ≈ 153.86, from sorry,
  have h3 : CI > SI, from sorry,
  have h4 : CI - SI ≈ 3.86, from sorry,
  exact ⟨h1, h2, h3, h4⟩
}

end investment_comparison_l424_424948


namespace inverse_of_307_mod_455_l424_424132

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

noncomputable def euclidean_algorithm (a b : ℕ) : ℕ × ℕ :=
  if b = 0 then (1, 0)
  else
    let (x, y) := euclidean_algorithm b (a % b)
    (y, x - (a / b) * y)

theorem inverse_of_307_mod_455 : ∃ a : ℤ, 0 ≤ a ∧ a < 455 ∧ (307 * a) % 455 = 1 := by
  have h1 : gcd 307 455 = 1 := by
    -- Verification part to check gcd(307, 455) = 1
    have h2 : gcd 148 11 = gcd 148 (307 % 148) := by rw [nat.mod_eq_sub_div, nat.mod_eq_of_lt]; norm_num
    have h3 : gcd 307 455 = gcd 148 (455 % 307) := by rw [nat.mod_eq_sub_div, nat.mod_eq_of_lt]; norm_num
    rw [←h3, ←h2, gcd]
    norm_num
  use 81
  split
  norm_num
  split
  norm_num
  norm_num
  show (307 * 81) % 455 = 1
  sorry

end inverse_of_307_mod_455_l424_424132


namespace problem_part_a_problem_part_b_l424_424509

noncomputable def P_a (n : ℕ) : ℚ := 
  ∑ k in Finset.range (n + 1), (-1) ^ k * (Nat.descFactorial (n - k) n)⁻¹

noncomputable def P_b (n : ℕ) : ℚ := 
  (∑ k in Finset.Ico 2 (n + 1), (-1) ^ k * (k.factorial)⁻¹) ^ 2

theorem problem_part_a (n : ℕ) : 
  P_a n = ∑ k in Finset.range (n + 1), (-1) ^ k * (Nat.descFactorial (n - k) n)⁻¹ := 
sorry

theorem problem_part_b (n : ℕ) : 
  P_b n = (∑ k in Finset.Ico 2 (n + 1), (-1) ^ k * (k.factorial)⁻¹) ^ 2 := 
sorry

end problem_part_a_problem_part_b_l424_424509


namespace solution_l424_424249

noncomputable def exists_point_X (A B : Point) (S : Circle) (MN : Line) : Prop :=
  ∃ X : Point, 
    OnCircle X S ∧ ∃ C D : Point, 
      OnCircle C S ∧ OnCircle D S ∧ 
      Collinear A X C ∧ Collinear B X D ∧ 
      Parallel CD MN

def proof_problem : Prop := 
  ∀ (A B : Point) (S : Circle) (MN : Line), exists_point_X A B S MN

theorem solution : proof_problem := 
  sorry

end solution_l424_424249


namespace graph_shift_sin_l424_424023

theorem graph_shift_sin (x : ℝ) : 2 * sin (x + π / 3) = 2 * sin (x - (-π / 3)) := 
sorry

end graph_shift_sin_l424_424023


namespace chemical_reaction_produces_l424_424256

def balanced_equation : Prop :=
  ∀ {CaCO3 HCl CaCl2 CO2 H2O : ℕ},
    (CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O)

def calculate_final_products (initial_CaCO3 initial_HCl final_CaCl2 final_CO2 final_H2O remaining_HCl : ℕ) : Prop :=
  balanced_equation ∧
  initial_CaCO3 = 3 ∧
  initial_HCl = 8 ∧
  final_CaCl2 = 3 ∧
  final_CO2 = 3 ∧
  final_H2O = 3 ∧
  remaining_HCl = 2

theorem chemical_reaction_produces :
  calculate_final_products 3 8 3 3 3 2 :=
by sorry

end chemical_reaction_produces_l424_424256


namespace exists_natural_number_appended_to_itself_as_perfect_square_l424_424620

theorem exists_natural_number_appended_to_itself_as_perfect_square :
  ∃ (A : ℕ), ∃ (n : ℕ), nat.is_square ((10^n + 1) * A) :=
by
sorry

end exists_natural_number_appended_to_itself_as_perfect_square_l424_424620


namespace cosine_of_angle_between_skew_lines_l424_424736

theorem cosine_of_angle_between_skew_lines
  (A B C D E F A1 : Point)
  (h1 : rhombus A B C D)
  (h2 : angle A D B = 60)
  (h3 : fold_tri_along_BD A B D A1)
  (h4 : dihedral_angle A1 B D C = 60) :
  cos_angle (line_through D A1) (line_through B C) = 1 / 8 :=
sorry

end cosine_of_angle_between_skew_lines_l424_424736


namespace strictly_increasing_seqs_count_l424_424171

theorem strictly_increasing_seqs_count : 
  ∑ k in (finset.range 9).filter (λ k, k ≥ 2), nat.choose 9 k = 502 := by
  sorry

end strictly_increasing_seqs_count_l424_424171


namespace sum_elements_of_B_is_5_l424_424418

theorem sum_elements_of_B_is_5 :
  let A := {2, 0, 1, 3}
  let B := { x | x ∈ A ∧ 2 - x^2 ∉ A }
  (B.sum id) = 5 :=
by
  sorry

end sum_elements_of_B_is_5_l424_424418


namespace product_of_possible_values_of_x_l424_424312

theorem product_of_possible_values_of_x : 
  (∀ x : ℝ, (|18 / x + 4| = 3) → x = -18 ∨ x = -(18 / 7)) →
  ((-18) * (-(18 / 7)) = 324 / 7) :=
by
  intros h
  have hx1 : (-18) * (-(18 / 7)) = 324 / 7, from sorry
  exact hx1

end product_of_possible_values_of_x_l424_424312


namespace water_consumption_per_hour_l424_424362

theorem water_consumption_per_hour 
  (W : ℝ) 
  (initial_water : ℝ := 20) 
  (initial_food : ℝ := 10) 
  (initial_gear : ℝ := 20) 
  (food_consumption_rate : ℝ := 1 / 3) 
  (hours : ℝ := 6) 
  (remaining_weight : ℝ := 34)
  (initial_weight := initial_water + initial_food + initial_gear)
  (consumed_water := W * hours)
  (consumed_food := food_consumption_rate * W * hours)
  (consumed_weight := consumed_water + consumed_food)
  (final_equation := initial_weight - consumed_weight)
  (correct_answer := 2) :
  final_equation = remaining_weight → W = correct_answer := 
by 
  sorry

end water_consumption_per_hour_l424_424362


namespace ratio_PM_MQ_eq_1_l424_424745

theorem ratio_PM_MQ_eq_1
  (A B C D E M P Q : ℝ × ℝ)
  (square_side : ℝ)
  (h_square_side : square_side = 15)
  (hA : A = (0, square_side))
  (hB : B = (square_side, square_side))
  (hC : C = (square_side, 0))
  (hD : D = (0, 0))
  (hE : E = (8, 0))
  (hM : M = ((A.1 + E.1) / 2, (A.2 + E.2) / 2))
  (h_slope_AE : E.2 - A.2 = (E.1 - A.1) * -15 / 8)
  (h_P_on_AD : P.2 = 15)
  (h_Q_on_BC : Q.2 = 0)
  (h_PM_len : dist M P = dist M Q) :
  dist P M = dist M Q :=
by sorry

end ratio_PM_MQ_eq_1_l424_424745


namespace room_volume_l424_424565

theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 :=
by sorry

end room_volume_l424_424565


namespace remainder_difference_l424_424314

theorem remainder_difference :
  ∃ (d r: ℤ), (1 < d) ∧ (1250 % d = r) ∧ (1890 % d = r) ∧ (2500 % d = r) ∧ (d - r = 10) :=
sorry

end remainder_difference_l424_424314


namespace range_of_x_for_sqrt_l424_424740

theorem range_of_x_for_sqrt (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 := 
sorry

end range_of_x_for_sqrt_l424_424740


namespace sum_of_100_and_98_consecutive_diff_digits_l424_424986

def S100 (n : ℕ) : ℕ := 50 * (2 * n + 99)
def S98 (n : ℕ) : ℕ := 49 * (2 * n + 297)

theorem sum_of_100_and_98_consecutive_diff_digits (n : ℕ) :
  ¬ (S100 n % 10 = S98 n % 10) :=
sorry

end sum_of_100_and_98_consecutive_diff_digits_l424_424986


namespace parallel_vectors_m_l424_424705

theorem parallel_vectors_m (m : ℝ) : 
  let a := (1 : ℝ, 2 : ℝ)
  let b := (2 : ℝ, m^2)
  (∀ k : ℝ, b = (k * fst a, k * snd a)) → m = 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_m_l424_424705


namespace count_pairs_satisfying_condition_l424_424264

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424264


namespace distance_from_midpoint_C_equals_14_69_l424_424488

-- Define the conditions
def distance_AB : ℝ := 235
def speed_car1 : ℝ := 70
def speed_car2 : ℝ := 90

-- Define the theorem to prove
theorem distance_from_midpoint_C_equals_14_69 :
  let midpoint_distance := distance_AB / 2 in
  let combined_speed := speed_car1 + speed_car2 in
  let meeting_time := distance_AB / combined_speed in
  let distance_car1 := speed_car1 * meeting_time in
  let distance_car2 := speed_car2 * meeting_time in
  let distance_from_C := |midpoint_distance - distance_car1| in
  distance_from_C = 14.69 :=
sorry

end distance_from_midpoint_C_equals_14_69_l424_424488


namespace convex_polygon_parallelograms_iff_centrally_symmetric_l424_424791

/-- Prove that a convex polygon can be decomposed into a finite number of parallelograms if and only if the polygon is centrally symmetric. -/
theorem convex_polygon_parallelograms_iff_centrally_symmetric
  {K : Type*} [convex K] :
  (∃ F : finset (set K), ∀ P ∈ F, is_parallelogram P ∧ ⋃₀ F = K) ↔ centrally_symmetric K :=
sorry

end convex_polygon_parallelograms_iff_centrally_symmetric_l424_424791


namespace sum_of_integers_mod_59_l424_424484

theorem sum_of_integers_mod_59 (a b c : ℕ) (h1 : a % 59 = 29) (h2 : b % 59 = 31) (h3 : c % 59 = 7)
  (h4 : a^2 % 59 = 29) (h5 : b^2 % 59 = 31) (h6 : c^2 % 59 = 7) :
  (a + b + c) % 59 = 8 :=
by
  sorry

end sum_of_integers_mod_59_l424_424484


namespace max_value_of_a_l424_424673

noncomputable def max_a : ℝ :=
  6 - 6 * Real.log 6

theorem max_value_of_a :
  ∀ a : ℝ, (∀ k : ℝ, -1 ≤ k ∧ k ≤ 1 → ∀ x : ℝ, 0 < x ∧ x ≤ 6 →
  6 * Real.log x + x^2 - 8 * x + a ≤ k * x) → a ≤ max_a :=
begin
  sorry
end

end max_value_of_a_l424_424673


namespace find_x_l424_424250

structure Vector2D where
  x : ℝ
  y : ℝ

def vecAdd (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

def vecScale (c : ℝ) (v : Vector2D) : Vector2D :=
  ⟨c * v.x, c * v.y⟩

def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem find_x (x : ℝ)
  (a : Vector2D := ⟨1, 2⟩)
  (b : Vector2D := ⟨x, 1⟩)
  (h : areParallel (vecAdd a (vecScale 2 b)) (vecAdd (vecScale 2 a) (vecScale (-2) b))) :
  x = 1 / 2 :=
by
  sorry

end find_x_l424_424250


namespace boxes_containing_neither_l424_424415

theorem boxes_containing_neither
  (total_boxes : ℕ) (pencils_boxes : ℕ) (pens_boxes : ℕ) (both_boxes : ℕ) :
  total_boxes = 15 →
  pencils_boxes = 8 →
  pens_boxes = 5 →
  both_boxes = 4 →
  (total_boxes - (pencils_boxes + pens_boxes - both_boxes)) = 6 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end boxes_containing_neither_l424_424415


namespace base8_addition_l424_424575

theorem base8_addition : (234 : ℕ) + (157 : ℕ) = (4 * 8^2 + 1 * 8^1 + 3 * 8^0 : ℕ) :=
by sorry

end base8_addition_l424_424575


namespace ounces_per_pound_l424_424553

variable {days_in_year : ℕ} -- number of days in a year
variable {first_days : ℕ} -- first days where puppy needs 2 ounces/day
variable {second_days : ℕ} -- remaining days where puppy needs 4 ounces/day
variable {total_bags : ℕ} -- total number of 5-pound bags needed to feed the puppy
variable {ounce_per_day_first : ℕ} -- ounces per day for first period
variable {ounce_per_day_second : ℕ} -- ounces per day for second period
variable {total_ounces : ℕ} -- total ounces needed for the entire period
variable {pounds_per_bag : ℕ} -- pounds per bag

axiom h1 : days_in_year = 365
axiom h2 : first_days = 60
axiom h3 : second_days = days_in_year - first_days
axiom h4 : total_bags = 17
axiom h5 : ounces_per_day_first = 2
axiom h6 : ounces_per_day_second = 4
axiom h7 : pounds_per_bag = 5
axiom h8 : total_ounces = (first_days * ounces_per_day_first) + (second_days * ounces_per_day_second)

theorem ounces_per_pound :
  (total_ounces.to_rat / total_bags.to_rat).nat_abs / pounds_per_bag = 16 :=
by sorry

end ounces_per_pound_l424_424553


namespace number_of_children_in_group_l424_424555

-- Definitions based on the conditions
def num_adults : ℕ := 55
def meal_for_adults : ℕ := 70
def meal_for_children : ℕ := 90
def remaining_children_after_adults : ℕ := 81
def num_adults_eaten : ℕ := 7
def ratio_adult_to_child : ℚ := (70 : ℚ) / 90

-- Statement of the problem to prove number of children in the group
theorem number_of_children_in_group : 
  ∃ C : ℕ, 
    (meal_for_adults - num_adults_eaten) * (ratio_adult_to_child) = (remaining_children_after_adults) ∧
    C = remaining_children_after_adults := 
sorry

end number_of_children_in_group_l424_424555


namespace alpha_epsilon_time_difference_l424_424007

def B := 100
def M := 120
def A := B - 10

theorem alpha_epsilon_time_difference : M - A = 30 := by
  sorry

end alpha_epsilon_time_difference_l424_424007


namespace age_sum_proof_l424_424483

noncomputable def leilei_age : ℝ := 30 -- Age of Leilei this year
noncomputable def feifei_age (R : ℝ) : ℝ := 1 / 2 * R + 12 -- Age of Feifei this year defined in terms of R

theorem age_sum_proof (R F : ℝ)
  (h1 : F = 1 / 2 * R + 12)
  (h2 : F + 1 = 2 * (R + 1) - 34) :
  R + F = 57 :=
by 
  -- Proof steps would go here
  sorry

end age_sum_proof_l424_424483


namespace percent_alcohol_new_solution_l424_424538

-- Defining the initial conditions
def original_volume : ℝ := 40
def original_percent_alcohol : ℝ := 0.05
def added_alcohol : ℝ := 4.5
def added_water : ℝ := 5.5

-- Calculating the initial amount of alcohol
def initial_alcohol : ℝ := original_volume * original_percent_alcohol

-- Calculating the new amount of alcohol
def new_alcohol : ℝ := initial_alcohol + added_alcohol

-- Calculating the total new volume
def new_volume : ℝ := original_volume + added_alcohol + added_water

-- Defining the percentage of alcohol in the new solution
def percent_alcohol (amount_alcohol : ℝ) (total_volume : ℝ) : ℝ :=
  (amount_alcohol / total_volume) * 100

-- Statement of the theorem to be proved
theorem percent_alcohol_new_solution :
  percent_alcohol new_alcohol new_volume = 13 :=
by
  sorry

end percent_alcohol_new_solution_l424_424538


namespace gcd_38_23_l424_424031

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

theorem gcd_38_23 : gcd 38 23 = 1 := by
  sorry

end gcd_38_23_l424_424031


namespace determine_optimal_nut_distribution_l424_424752

-- Definitions for the problem
variables (n : ℕ) (h_n : n ≥ 2)

-- Definition of the first stage: Petya divides into two parts
variables (a b : ℕ) (h_div : a + b = 2 * n + 1) (h_a2 : a ≥ 2) (h_b2 : b ≥ 2)

-- Definition of the second stage: Kolya subdivides each part into two subparts
variables (a1 a2 b1 b2 : ℕ) (h_sub_a : a1 + a2 = a) (h_sub_b : b1 + b2 = b) 
          (h_a1 : a1 ≥ 1) (h_a2 : a2 ≥ 1) (h_b1 : b1 ≥ 1) (h_b2 : b2 ≥ 1)

-- Definition of the third stage methods
def method1_kolya_nuts : ℕ := max a1 b1
def method2_kolya_nuts : ℕ := (a1 + a2 + b1 + b2) - (max a1 b1 + min a2 b2)
def method3_kolya_nuts : ℕ := max (max a1 b1 - 1) (method2_kolya_nuts - 1)

-- Theorem statement
theorem determine_optimal_nut_distribution : 
  ∃ method : string, (method = "method1" ∧ method1_kolya_nuts = max method1_kolya_nuts method2_kolya_nuts method3_kolya_nuts) ∨ 
                     (method = "method2" ∧ method2_kolya_nuts = max method1_kolya_nuts method2_kolya_nuts method3_kolya_nuts) ∨ 
                     (method = "method3" ∧ method3_kolya_nuts = max method1_kolya_nuts method2_kolya_nuts method3_kolya_nuts) :=
sorry

end determine_optimal_nut_distribution_l424_424752


namespace possible_area_of_triangle_SIX_l424_424924

theorem possible_area_of_triangle_SIX :
  ∀ (dodecagon : Polygon) 
  (S I X : Point),
  (isDodecagon dodecagon) → -- dodecagon has 12 sides
  (allSidesEqual dodecagon 2) → -- all side lengths are equal to 2
  (isNotSelfIntersecting dodecagon) → -- dodecagon is not self-intersecting
  (interiorAnglesCondition dodecagon) → -- each interior angle is either 90 or 270 degrees
  (pointsFromDodecagon dodecagon S I X) → -- S, I, X are distinct vertices of the dodecagon
  (areaOfTriangle S I X = 2 ∨ areaOfTriangle S I X = 4) :=
sorry

end possible_area_of_triangle_SIX_l424_424924


namespace nth_positive_nonsquare_l424_424187

noncomputable def isNonsquare (x : ℕ) : Prop :=
  ¬ ∃ k : ℕ, k * k = x

noncomputable def nthNonsquare (n : ℕ) : ℕ :=
  (Finset.range ((n + 1)^2)).filter isNonsquare).nth n).getOrElse 0

theorem nth_positive_nonsquare (n : ℕ) (hn : 0 < n) :
  nthNonsquare n = n + (Nat.floor (Real.sqrt n)) := by
  sorry

end nth_positive_nonsquare_l424_424187


namespace evaluate_sqrt_l424_424151

theorem evaluate_sqrt {a : ℝ} (h : a = 5^7) :
    Real.cbrt (a + a + a + a + a) = 25 * Real.cbrt (5^2) :=
by
  -- The steps of the proof would be elaborated here
  sorry

end evaluate_sqrt_l424_424151


namespace tangent_line_parabola_l424_424825

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l424_424825


namespace number_of_binders_l424_424480

theorem number_of_binders (total_sheets : ℕ) (sheets_used : ℕ) (total_colored : ℕ) (split_evenly : ∃ k : ℕ, k * total_colored = total_sheets) :
  total_sheets = 2450 ∧ sheets_used = 245 ∧ total_colored = 490 ∧ split_evenly → total_sheets / total_colored = 5 :=
by
  intros h
  obtain ⟨hsheets, hused, hcolored, hsplit⟩ := h
  rw [←nat.mul_div_cancel_left (total_colored) (ne_zero.intro hcolored)] at hsplit
  rw [hsheets, hcolored, hsplit]
  exact rfl

end number_of_binders_l424_424480


namespace rectangular_to_polar_l424_424139

theorem rectangular_to_polar (x y : ℝ) (r : ℝ) (θ : ℝ) (h₁ : x = 3) (h₂ : y = -3) 
(h₃ : r = sqrt (x^2 + y^2)) (h₄ : θ = if y = 0 then if x ≥ 0 then 0 else π else if x = 0 then if y > 0 then π/2 else 3*π/2 else if x > 0 then atan (y / x) else if x < 0 ∧ y ≥ 0 then atan (y / x) + π else atan (y / x) - π) :
r = 3 * sqrt 2 ∧ θ = 7 * π / 4 := by
  sorry

end rectangular_to_polar_l424_424139


namespace john_candies_correct_l424_424352

variable (Bob_candies : ℕ) (Mary_candies : ℕ)
          (Sue_candies : ℕ) (Sam_candies : ℕ)
          (Total_candies : ℕ) (John_candies : ℕ)

axiom bob_has : Bob_candies = 10
axiom mary_has : Mary_candies = 5
axiom sue_has : Sue_candies = 20
axiom sam_has : Sam_candies = 10
axiom total_has : Total_candies = 50

theorem john_candies_correct : 
  Bob_candies + Mary_candies + Sue_candies + Sam_candies + John_candies = Total_candies → John_candies = 5 := by
sorry

end john_candies_correct_l424_424352


namespace BD_is_diameter_of_circle_l424_424363

variables {A B C D X Y : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] [MetricSpace X] [MetricSpace Y]

-- Assume these four points lie on a circle with certain ordering
variables (circ : Circle A B C D)

-- Given conditions
variables (h1 : circ.AB < circ.AD)
variables (h2 : circ.BC > circ.CD)

-- Points X and Y are where angle bisectors meet the circle again
variables (h3 : circ.bisects_angle_BAD_at X)
variables (h4 : circ.bisects_angle_BCD_at Y)

-- Hexagon sides with four equal lengths
variables (hex_equal : circ.hexagon_sides_equal_length A B X C D Y)

-- Prove that BD is a diameter
theorem BD_is_diameter_of_circle : circ.is_diameter BD := 
by
  sorry

end BD_is_diameter_of_circle_l424_424363


namespace ratio_of_intercepts_l424_424872

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l424_424872


namespace strictly_increasing_seqs_count_l424_424168

theorem strictly_increasing_seqs_count : 
  ∑ k in (finset.range 9).filter (λ k, k ≥ 2), nat.choose 9 k = 502 := by
  sorry

end strictly_increasing_seqs_count_l424_424168


namespace count_pairs_satisfying_condition_l424_424293

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424293


namespace number_of_pairs_count_number_of_pairs_l424_424305

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424305


namespace distilled_water_required_l424_424326

theorem distilled_water_required :
  ∀ (nutrient_concentrate distilled_water : ℝ) (total_solution prep_solution : ℝ), 
    nutrient_concentrate = 0.05 →
    distilled_water = 0.025 →
    total_solution = 0.075 → 
    prep_solution = 0.6 →
    (prep_solution * (distilled_water / total_solution)) = 0.2 :=
by
  intros nutrient_concentrate distilled_water total_solution prep_solution
  sorry

end distilled_water_required_l424_424326


namespace books_needed_to_buy_clarinet_l424_424433

def cost_of_clarinet : ℕ := 90
def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def halfway_loss : ℕ := (cost_of_clarinet - initial_savings) / 2

theorem books_needed_to_buy_clarinet 
    (cost_of_clarinet initial_savings price_per_book halfway_loss : ℕ)
    (initial_savings_lost : halfway_loss = (cost_of_clarinet - initial_savings) / 2) : 
    ((cost_of_clarinet - initial_savings + halfway_loss) / price_per_book) = 24 := 
sorry

end books_needed_to_buy_clarinet_l424_424433


namespace distance_between_l1_l2_l424_424441

def line1 : ℝ → ℝ → ℝ := fun x y => 3 * x - 4 * y + 6
def line2 : ℝ → ℝ → ℝ := fun x y => 6 * x - 8 * y + 9

def distance_between_parallel_lines (c1 c2 a b : ℝ) : ℝ :=
  abs (c1 - c2) / real.sqrt (a^2 + b^2)

theorem distance_between_l1_l2 :
  distance_between_parallel_lines 12 9 6 8 = 3 / 10 :=
by
  -- Proof goes here
  sorry

end distance_between_l1_l2_l424_424441


namespace line_perpendicular_AC_l424_424087

-- Definitions of the conditions
variable {A B C : Type} [EuclideanGeometry A B C]

-- Definitions for the problem
def line_perpendicular_to_sides_of_triangle (l : line) (A B C : point) : Prop :=
  (perpendicular l (line_through_points A B) ∧ perpendicular l (line_through_points B C))

-- The theorem to prove
theorem line_perpendicular_AC {l : line} {A B C : point} (h1 : line_perpendicular_to_sides_of_triangle l A B C) :
  perpendicular l (line_through_points A C) :=
  sorry

end line_perpendicular_AC_l424_424087


namespace polygon_sides_eq_six_l424_424013

theorem polygon_sides_eq_six (n : ℕ) :
  ((n - 2) * 180 = 2 * 360) → n = 6 := by
  intro h
  have : (n - 2) * 180 = 720 := by exact h
  have : n - 2 = 4 := by linarith
  have : n = 6 := by linarith
  exact this

end polygon_sides_eq_six_l424_424013


namespace tracy_initial_candies_l424_424024

theorem tracy_initial_candies : 
  ∃ (x : ℕ), 
    (∃ y : ℕ, y = x - 2 * x / 5 ∧ ∃ z : ℕ, z = y - y / 3 ∧ ∃ w : ℕ, w = z - 40 ∧ ∃ b : ℕ, 3 ≤ b ∧ b ≤ 7 ∧ w - b = 5) ∧ 
    x = 120 :=
begin
  -- The proof will be here
  sorry,
end

end tracy_initial_candies_l424_424024


namespace henry_seashells_l424_424707

theorem henry_seashells (H L : ℕ) (h1 : H + 24 + L = 59) (h2 : H + 24 + (3 * L) / 4 = 53) : H = 11 := by
  sorry

end henry_seashells_l424_424707


namespace profit_difference_l424_424056

variable (a_capital b_capital c_capital b_profit : ℕ)

theorem profit_difference (h₁ : a_capital = 8000) (h₂ : b_capital = 10000) 
                          (h₃ : c_capital = 12000) (h₄ : b_profit = 2000) : 
  c_capital * (b_profit / b_capital) - a_capital * (b_profit / b_capital) = 800 := 
sorry

end profit_difference_l424_424056


namespace arithmetic_sequence_20th_term_l424_424136

theorem arithmetic_sequence_20th_term (a1 d : ℕ) (n : ℕ) (h_a1 : a1 = 3) (h_d : d = 5) (h_n : n = 20) :
  a1 + (n - 1) * d = 98 :=
by
  rw [h_a1, h_d, h_n]
  -- after simplification, we will check if 3 + (20 - 1) * 5 = 98
  sorry

end arithmetic_sequence_20th_term_l424_424136


namespace jellybeans_in_new_bag_l424_424922

theorem jellybeans_in_new_bag (average_per_bag : ℕ) (num_bags : ℕ) (additional_avg_increase : ℕ) (total_jellybeans_old : ℕ) (total_jellybeans_new : ℕ) (num_bags_new : ℕ) (new_bag_jellybeans : ℕ) : 
  average_per_bag = 117 → 
  num_bags = 34 → 
  additional_avg_increase = 7 → 
  total_jellybeans_old = num_bags * average_per_bag → 
  total_jellybeans_new = (num_bags + 1) * (average_per_bag + additional_avg_increase) → 
  new_bag_jellybeans = total_jellybeans_new - total_jellybeans_old → 
  new_bag_jellybeans = 362 := 
by 
  intros 
  sorry

end jellybeans_in_new_bag_l424_424922


namespace perfect_square_probability_l424_424953

noncomputable def total_outcomes : ℕ := 8^6

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfect_square_probability :
  (∀ n : ℕ, n > 0 → n ∈ [1, 2, 3, 4, 6, 8]^6 → is_perfect_square (list.prod n)) →
  (∃ m n : ℕ, m / n = 625 / 32768 ∧ m + n = 33393) := 
by
  sorry

end perfect_square_probability_l424_424953


namespace sum_of_10_smallest_n_l424_424189

def T_n (n : ℕ) : ℚ :=
  (n - 1) * n * (n + 1) * (3 * n + 2) / 24

theorem sum_of_10_smallest_n :
  let ns := (filter (λ n : ℕ, n ≥ 4 ∧ (T_n n).denom = 1 ∧ (T_n n).num % 5 = 0) (list.range 100)).take 10 in
  ns.sum = 85 :=
by sorry

end sum_of_10_smallest_n_l424_424189


namespace stability_comparison_l424_424862

def variance (data : List ℤ) (mean : ℤ) : ℚ :=
  (List.sum (data.map (λ x => (x - mean) ^ 2))) / data.length

def var_A : ℚ :=
  variance [79, 81, 80, 80, 78, 82, 80] 80

def var_B : ℚ :=
  variance [79, 77, 80, 82, 81, 82, 79] 80

theorem stability_comparison :
  var_A < var_B := by
  -- proof steps omitted for brevity
  sorry

end stability_comparison_l424_424862


namespace circle_center_radius_proof_l424_424999

noncomputable def circle_center_radius (x y : ℝ) :=
  x^2 + y^2 - 4*x + 2*y + 2 = 0

theorem circle_center_radius_proof :
  ∀ x y : ℝ, circle_center_radius x y ↔ ((x - 2)^2 + (y + 1)^2 = 3) :=
by
  sorry

end circle_center_radius_proof_l424_424999


namespace shaded_to_unshaded_area_ratio_l424_424897

-- Define a structure for a large square with its side length.
structure Square :=
  (side_length : ℝ)

-- Define the condition that vertices of smaller squares are at the midpoints.
def midpoint_vertex_property (sq : Square) : Prop :=
  ∀ (sub_sq : Square), sub_sq.side_length = sq.side_length / 2

-- Define areas of regions within one quadrant.
def shaded_area (sq : Square) : ℝ :=
  5 * (sq.side_length / 2) * (sq.side_length / 2) / 2

def unshaded_area (sq : Square) : ℝ :=
  3 * (sq.side_length / 2) * (sq.side_length / 2) / 2

-- Define the ratio of the shaded area to the unshaded area.
def area_ratio (sq : Square) : ℝ :=
  shaded_area sq / unshaded_area sq

-- The theorem stating that the area ratio is 5:3.
theorem shaded_to_unshaded_area_ratio (sq : Square) (h : midpoint_vertex_property sq) :
  area_ratio sq = 5 / 3 := 
  sorry

end shaded_to_unshaded_area_ratio_l424_424897


namespace smallest_2214_l424_424044

/-- The smallest positive four-digit number divisible by 18 which has three even digits and one odd digit. -/
def smallest_four_digit_divisible_by_18 : ℕ :=
  2214

theorem smallest_2214 : ∃ n : ℕ, n = smallest_four_digit_divisible_by_18 ∧ (1000 ≤ n ∧ n < 10000) ∧ n % 18 = 0 ∧
                       (even (n / 1000) + even ((n / 100) % 10) + even ((n / 10) % 10) + even (n % 10) >= 3 ∧
                        odd (n / 1000) + odd ((n / 100) % 10) + odd ((n / 10) % 10) + odd (n % 10) = 1) :=
by
  use 2214
  split
  . rfl
  repeat split
  . sorry -- Proof that 2214 is a four-digit number
  . sorry -- Proof that 2214 is divisible by 18
  . sorry -- Proof that 2214 has three even digits and one odd digit


end smallest_2214_l424_424044


namespace smallest_positive_period_intervals_of_monotonic_decrease_range_m_value_l424_424770

noncomputable def f (x m : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + m

theorem smallest_positive_period (m : ℝ) : ∃ T > 0, ∀ x, f x m = f (x + T) m ∧ T = Real.pi := sorry

theorem intervals_of_monotonic_decrease (m : ℝ) : ∀ k : ℤ, is_monotonic_decreasing (f x m) ($$left_bound = ℝ.., interval : [ℝ]) := sorry

theorem range_m_value (x : ℝ) : x ∈ [0, Real.pi / 2] → ∃ m : ℝ, m = 1 / 2 ∧ (∀ y ∈ (range (f x m)), y ∈ [1 / 2, 7 / 2]) := sorry

end smallest_positive_period_intervals_of_monotonic_decrease_range_m_value_l424_424770


namespace reflection_across_x_axis_l424_424738

theorem reflection_across_x_axis :
  let initial_point := (-3, 5)
  let reflected_point := (-3, -5)
  reflected_point = (initial_point.1, -initial_point.2) :=
by
  sorry

end reflection_across_x_axis_l424_424738


namespace molecular_weight_CuCO3_8_moles_l424_424496

-- Definitions for atomic weights
def atomic_weight_Cu : ℝ := 63.55
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Definition for the molecular formula of CuCO3
def molecular_weight_CuCO3 :=
  atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O

-- Number of moles
def moles : ℝ := 8

-- Total weight of 8 moles of CuCO3
def total_weight := moles * molecular_weight_CuCO3

-- Proof statement
theorem molecular_weight_CuCO3_8_moles :
  total_weight = 988.48 :=
  by
  sorry

end molecular_weight_CuCO3_8_moles_l424_424496


namespace alpha_perpendicular_beta_l424_424410

variables {m n : Type} {α β : Type} [LinearOrder m] [LinearOrder n] [OrderBot α] [OrderBot β]

def parallel (l1 l2 : Type) := sorry -- Definition for parallel lines or planes
def perpendicular (l1 l2 : Type) := sorry -- Definition for perpendicular lines or planes
def subset (l1 l2 : Type) := sorry -- Definition for a line being within a plane

-- Hypotheses
variable (hmn : parallel m n)
variable (hmα : subset m α)
variable (hnβ : perpendicular n β)

-- Conclusion
theorem alpha_perpendicular_beta : perpendicular α β :=
sorry

end alpha_perpendicular_beta_l424_424410


namespace clerk_daily_salary_l424_424329

theorem clerk_daily_salary (manager_salary : ℝ) (num_managers num_clerks : ℕ) (total_salary : ℝ) (clerk_salary : ℝ)
  (h1 : manager_salary = 5)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16) :
  clerk_salary = 2 :=
by
  sorry

end clerk_daily_salary_l424_424329


namespace mike_afternoon_seeds_l424_424741

theorem mike_afternoon_seeds :
  let morning_mike := 50
  let morning_ted := 2 * morning_mike
  let afternoon_total := 250 - (morning_mike + morning_ted)
  ∃ X, mike_afternoon = X ∧ ted_afternoon = X - 20 ∧ (morning_mike + morning_ted) + (mike_afternoon + ted_afternoon) = 250 := 
  let mike_afternoon := 60,
  let ted_afternoon := mike_afternoon - 20
  (50 + (2 * 50)) + (60 + (60 - 20)) = 250

by sorry

end mike_afternoon_seeds_l424_424741


namespace max_possible_value_l424_424970

theorem max_possible_value (P Q : ℤ) (hP : P * P ≤ 729 ∧ 729 ≤ -P * P * P)
  (hQ : Q * Q ≤ 729 ∧ 729 ≤ -Q * Q * Q) :
  10 * (P - Q) = 180 :=
by
  sorry

end max_possible_value_l424_424970


namespace KN_length_l424_424409

theorem KN_length (K L M N : Point) (R n : ℝ)
  (h1 : InscribedQuadrilateral K L M N R)
  (h2 : LM = n)
  (h3 : PerpendicularDiagonals K M L N) :
  KN = sqrt (4 * R ^ 2 - n ^ 2) :=
sorry

end KN_length_l424_424409


namespace sin_value_l424_424649

theorem sin_value (theta : ℝ) (h : Real.cos (3 * Real.pi / 14 - theta) = 1 / 3) : 
  Real.sin (2 * Real.pi / 7 + theta) = 1 / 3 :=
by
  -- Sorry replaces the actual proof which is not required for this task
  sorry

end sin_value_l424_424649


namespace num_pairs_nat_nums_eq_l424_424272

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424272


namespace part_one_part_two_l424_424672

variable (t : ℝ) (α β : ℝ)
variable (u1 u2 u3 : ℝ)
variable (φ : Set.Icc α β → ℝ)

-- Part I
theorem part_one (h1 : α ≠ β) (h2 : 4*α^2 - 4*t*α - 1 = 0)
  (h3 : 4*β^2 - 4*(t)*β - 1 = 0) :
  φ = λ x, (2*x - t)/(x^2 + 1) →
  max φ - min φ =
  (8*Real.sqrt(t^2 + 1)*(2*t^2 + 5))/(16*t^2 + 25) :=
sorry

-- Part II
theorem part_two (h1 : ∀ i, 0 < u i ∧ u i < π/2) (h2 : ∑ i in [1, 2, 3], Real.sin (u i) = 1) :
  ∑ i in [1, 2, 3], 1/(φ (u i)) < (3/4) * Real.sqrt 6 :=
sorry

end part_one_part_two_l424_424672


namespace ratio_u_v_l424_424867

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l424_424867


namespace solve_number_l424_424431

noncomputable def find_number : Prop :=
  ∃ x : ℝ, (3/4 * x - 25) / 7 + 50 = 100 ∧ x = 500

theorem solve_number : find_number :=
  sorry

end solve_number_l424_424431


namespace count_pairs_satisfying_condition_l424_424265

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424265


namespace nat_pair_count_eq_five_l424_424291

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424291


namespace sum_of_areas_B_D_l424_424122

theorem sum_of_areas_B_D (area_large_square : ℝ) (area_small_square : ℝ) (B D : ℝ) 
  (h1 : area_large_square = 9) 
  (h2 : area_small_square = 1)
  (h3 : B + D = 4) : 
  B + D = 4 := 
by
  sorry

end sum_of_areas_B_D_l424_424122


namespace part_a_part_b_l424_424371

namespace Fibonacci

def fibonacci : ℕ → ℕ 
| 0     := 1
| 1     := 1
| n + 2 := fibonacci (n + 1) + fibonacci n

noncomputable def solution_a (k : ℕ) : (ℕ × ℕ) :=
(fibonacci (k - 1), fibonacci k)

noncomputable def solution_b : (ℝ × ℝ) :=
(1, 1)

theorem part_a (a b : ℕ) (n : ℕ) : (∃ k, (a, b) = solution_a k) ↔ (a * fibonacci n + b * fibonacci (n + 1) = fibonacci (l n)) :=
by
  sorry

theorem part_b (u v : ℝ) (n : ℕ) : (u = 1 ∧ v = 1) ↔ (u * (fibonacci n)^2 + v * (fibonacci (n + 1))^2 = fibonacci (l n)) :=
by
  sorry

end Fibonacci

end part_a_part_b_l424_424371


namespace rowing_time_l424_424557

def man_speed_still := 10.0
def river_speed := 1.2
def total_distance := 9.856

def upstream_speed := man_speed_still - river_speed
def downstream_speed := man_speed_still + river_speed

def one_way_distance := total_distance / 2
def time_upstream := one_way_distance / upstream_speed
def time_downstream := one_way_distance / downstream_speed

theorem rowing_time :
  time_upstream + time_downstream = 1 :=
by
  sorry

end rowing_time_l424_424557


namespace median_separates_area_l424_424034

theorem median_separates_area (Δ : Type) [triangle Δ] (A B C : Δ) (M : Δ) 
  (hAB : midpoint A B = M) (hCM : median C M) :
  area (triangle C M) / area (triangle A B C) = 1 / 4 := 
sorry

end median_separates_area_l424_424034


namespace joanna_marbles_l424_424747

theorem joanna_marbles (m n : ℕ) (h1 : m * n = 720) (h2 : m > 1) (h3 : n > 1) :
  ∃ (count : ℕ), count = 28 :=
by
  -- Use the properties of divisors and conditions to show that there are 28 valid pairs (m, n).
  sorry

end joanna_marbles_l424_424747


namespace one_coin_tails_up_l424_424925

theorem one_coin_tails_up (n : ℕ) : 
  ∃! k : fin (2*n + 1), (∑i in finset.range(2*n+1), if ((i + (i+1)) % (2*n + 1) = k) then 1 else 0) % 2 = 1 :=
sorry

end one_coin_tails_up_l424_424925


namespace num_pairs_nat_nums_eq_l424_424275

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424275


namespace max_consecutive_sum_to_450_l424_424038

theorem max_consecutive_sum_to_450 :
  ∃ n : ℕ, (∑ k in finset.range (n + 1), if k ≥ 3 then k else 0) < 450 ∧ ∀ m : ℕ, (m > n → (∑ k in finset.range (m + 1), if k ≥ 3 then k else 0) ≥ 450) := 
sorry

end max_consecutive_sum_to_450_l424_424038


namespace foxes_wolves_bears_num_l424_424194

-- Definitions and theorem statement
def num_hunters := 45
def num_rabbits := 2008
def rabbits_per_fox := 59
def rabbits_per_wolf := 41
def rabbits_per_bear := 40

theorem foxes_wolves_bears_num (x y z : ℤ) : 
  x + y + z = num_hunters → 
  rabbits_per_wolf * x + rabbits_per_fox * y + rabbits_per_bear * z = num_rabbits → 
  x = 18 ∧ y = 10 ∧ z = 17 :=
by 
  intro h1 h2 
  sorry

end foxes_wolves_bears_num_l424_424194


namespace unit_digit_sum_cubes_l424_424231

theorem unit_digit_sum_cubes : 
  let unit_digits := [1, 8, 7, 4, 5, 6, 3, 2, 9, 0] in
  let sum_unit_digit := 5 in
  let repeat_count := 200 in
  let last_digit := 1 in
  (repeat_count * sum_unit_digit + last_digit) % 10 = 1 :=
by
  sorry

end unit_digit_sum_cubes_l424_424231


namespace circle_radius_l424_424945

theorem circle_radius (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 10) : r = 20 :=
by
  sorry

end circle_radius_l424_424945


namespace find_value_of_f_l424_424669

axiom f : ℝ → ℝ

theorem find_value_of_f :
  (∀ x : ℝ, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (Real.pi / 9)) = -1 / 2 :=
sorry

end find_value_of_f_l424_424669


namespace ratio_of_x_intercepts_l424_424875

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l424_424875


namespace field_width_calculation_l424_424451

theorem field_width_calculation (w : ℝ) (h_length : length = 24) (h_length_width_relation : length = 2 * w - 3) : w = 13.5 :=
by 
  sorry

end field_width_calculation_l424_424451


namespace pairing_count_l424_424858

theorem pairing_count (n : ℕ) (h : n = 12) :
  let knows := λ (a b : ℕ), 
    ((b = (a + 1) % n) ∨ (b = (a + n - 1) % n) ∨ (b = (a + n/2) % n) ∨ (b = (a + 2) % n))
  in ∃ pairs : list (ℕ × ℕ), 
    (pairs.length = n / 2) ∧ 
    ∀ (p : ℕ × ℕ), p ∈ pairs → knows p.fst p.snd ∧ 
    ∀ x, (∃ p, p ∈ pairs ∧ (x = p.fst ∨ x = p.snd)) := 
    20 :=
sorry

end pairing_count_l424_424858


namespace sum_of_sequences_l424_424370

theorem sum_of_sequences (a b : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) :
  (∀ n, a n = 2^n) →
  (∀ n, b n = 2 * n - 1) →
  (∀ n, S n = (∑ k in finset.range n, a (k + 1) + b (k + 1))) →
  S n = 2^(n+1) + n^2 - 2 :=
by
  intros ha hb hS
  sorry

end sum_of_sequences_l424_424370


namespace nat_pair_count_eq_five_l424_424289

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424289


namespace count_increasing_numbers_l424_424175

-- Define the set of digits we are concerned with
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a natural number type representing numbers with increasing digits
def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits i < n.digits j

-- Define the set of natural numbers with increasing digits and at least two digits
def increasing_numbers : set ℕ :=
  {n | increasing_digits n ∧ 10 ≤ n ∧ n ≤ 987654321}

-- Define the theorem to be proved
theorem count_increasing_numbers : set.card increasing_numbers = 502 :=
by sorry

end count_increasing_numbers_l424_424175


namespace slope_of_AB_zero_l424_424688

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : set (ℝ × ℝ) :=
{ p | ∃ x y, p = (x, y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1 }

noncomputable def parabola (c : ℝ) : set (ℝ × ℝ) :=
{ p | ∃ x y, p = (x, y) ∧ y^2 = 2 * c * (x - c / 2) }

theorem slope_of_AB_zero 
  (a b c x y : ℝ)
  (h : a > b ∧ b > 0)
  (hc : c = sqrt (a^2 - b^2))
  (A_on_ellipse : (x, y) ∈ ellipse a b h)
  (A_on_parabola : (x, y) ∈ parabola c)
  (B_on_ellipse : ∃ (xb yb : ℝ), (xb, yb) ∈ ellipse a b h ∧ xb * x < 0)
  (S_ratio : ∀ (S_ABE S_ABD : ℝ), S_ABE / S_ABD = a / c) :
  (∃ m, m = 0) :=
by
  -- Proof will be here
  sorry

end slope_of_AB_zero_l424_424688


namespace possible_arrangement_l424_424001

def is_parity_even (s : list ℕ) : bool :=
(s.sum % 2 = 0)

def neighboring (i j k l : ℕ) : Prop :=
(abs (i - k) = 1 ∧ j = l) ∨ (abs (j - l) = 1 ∧ i = k)

theorem possible_arrangement :
  let answer_sets := {s : list ℕ // s.length = 6 ∧ ∀ x, x ∈ s → x = 0 ∨ x = 1 } in
  ∃ (f : ℕ → ℕ → answer_sets),
    (∀ i j k l, neighboring i j k l → dist (f i j) (f k l) = 1) ∧
    (∀ i j k l, (i ≠ k ∨ j ≠ l) → f i j ≠ f k l)
: sorry

end possible_arrangement_l424_424001


namespace find_third_number_l424_424439

theorem find_third_number (x : ℝ) 
  (h : (20 + 40 + x) / 3 = (10 + 50 + 45) / 3 + 5) : x = 60 :=
sorry

end find_third_number_l424_424439


namespace least_possible_value_l424_424762

theorem least_possible_value (P : ℤ → ℤ) (h_poly : ∀ (x : ℤ), P x = C(x)) (H : P 0 + P 90 = 2018) : |P 20 + P 70| = 782 :=
sorry

end least_possible_value_l424_424762


namespace find_years_simple_interest_l424_424571

variable (R T : ℝ)
variable (P : ℝ := 6000)
variable (additional_interest : ℝ := 360)
variable (rate_diff : ℝ := 2)
variable (H : P * ((R + rate_diff) / 100) * T = P * (R / 100) * T + additional_interest)

theorem find_years_simple_interest (h : P = 6000) (h₁ : P * ((R + 2) / 100) * T = P * (R / 100) * T + 360) : 
T = 3 :=
sorry

end find_years_simple_interest_l424_424571


namespace binary_sequences_no_three_consecutive_zeros_l424_424612

-- Define the primary theorem to be proven
theorem binary_sequences_no_three_consecutive_zeros :
  let total_sequences := 
      (∑ m in Finset.range 6, 
         (Nat.choose 11 m) * (Nat.choose (11 - m) (10 - 2 * m))) in
  total_sequences = 24068 :=
by {
  -- Using the specific transformation without the need to provide proof steps here
  sorry
}

end binary_sequences_no_three_consecutive_zeros_l424_424612


namespace John_income_l424_424359

noncomputable def J : ℝ := 56067.31

def John_tax_rate : ℝ := 0.30
def Ingrid_tax_rate : ℝ := 0.40
def Ingrid_income : ℝ := 74000
def Combined_tax_rate : ℝ := 0.3569
def Combined_income : ℝ := J + Ingrid_income
def Combined_tax : ℝ := John_tax_rate * J + Ingrid_tax_rate * Ingrid_income

theorem John_income :
  Combined_tax = Combined_tax_rate * Combined_income :=
sorry

end John_income_l424_424359


namespace trader_made_20_percent_profit_l424_424572

variable (P : Real)

-- Define the bought price
def boughtPrice : Real := 0.80 * P

-- Define the sold price
def soldPrice : Real := 1.20 * P

-- Define the profit
def profit : Real := soldPrice - P

-- Define the profit percent on original price
def profitPercentOnOriginalPrice : Real := (profit / P) * 100

-- The theorem to prove the profit percent on the original price
theorem trader_made_20_percent_profit (hP : P > 0) : profitPercentOnOriginalPrice = 20 := 
by
  sorry

end trader_made_20_percent_profit_l424_424572


namespace rectangular_prism_inequality_l424_424771

variable {a b c l : ℝ}

theorem rectangular_prism_inequality (h_diag : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
sorry

end rectangular_prism_inequality_l424_424771


namespace stadium_length_in_feet_l424_424456

theorem stadium_length_in_feet (yard_length : ℕ) (yard_to_feet : ℕ) (length_in_yards : ℕ) : 
  yard_length = 1 → yard_to_feet = 3 → length_in_yards = 61 → length_in_yards * yard_to_feet = 183 :=
by
  intros h1 h2 h3
  rw [h3, h2]
  exact rfl

end stadium_length_in_feet_l424_424456


namespace ratio_of_areas_l424_424896

-- Define the conditions of the problem
variable (SQ : Type) [Square SQ]    -- Assume SQ is the type of all squares, and it satisfies some square properties.

-- Define the vertices and midpoint condition
axiom vertices_at_midpoints (s : SQ) : vertices at midpoints ∧ symmetric across quadrants

-- Define the area function and assume shaded and white area as given in a quadrant
variable (area : SQ → ℕ)
def shaded_area (sq : SQ) := 5 * area sq -- Shaded triangles per quadrant
def white_area (sq : SQ) := 3 * area sq  -- White triangles per quadrant

-- Resulting areas for the whole figure
def total_shaded_area (sq : SQ) := 4 * shaded_area sq
def total_white_area (sq : SQ) := 4 * white_area sq

-- Define the ratio function to compare shaded and white areas
def ratio (a b : ℕ) := a / b

theorem ratio_of_areas (s : SQ) (area_nonzero : area s ≠ 0):
  ratio (total_shaded_area s) (total_white_area s) = 5 / 3 := by
  sorry

end ratio_of_areas_l424_424896


namespace ellipse_foci_coordinates_l424_424440

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (y^2 / 3 + x^2 / 2 = 1) → (x, y) = (0, -1) ∨ (x, y) = (0, 1) :=
by
  sorry

end ellipse_foci_coordinates_l424_424440


namespace solve_for_x_l424_424317

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h_eq : x^4 = 6561) : x = 9 :=
sorry

end solve_for_x_l424_424317


namespace AX_perp_XC_l424_424343

theorem AX_perp_XC
  (A B C D M X : Type*)
  [IsTriangle (Triangle A B C)]
  (h₁ : IsAcute ∠A B C)
  (h₂ : AngleBisector A B D C)
  (h₃ : Midpoint M A D)
  (h₄ : ∃ (X ∈ Segment B M), ∠(M, X, A) = ∠(D, A, C)) :
  Perpendicular A X X C := 
begin
  sorry
end

end AX_perp_XC_l424_424343


namespace clothing_factory_interests_l424_424578

theorem clothing_factory_interests 
  (n : ℕ) (hn : n > 11000) (selected_students : Fin n → ℝ)
  (mean : ℝ) (median : ℝ) (mode : ℝ) (variance : ℝ) :
  clothing_factory_interest selected_students = mode :=
by
  sorry

noncomputable def clothing_factory_interest : (Fin n → ℝ) → ℝ :=
  sorry

end clothing_factory_interests_l424_424578


namespace sum_value_l424_424368

noncomputable def arithmetic_sum (n : ℕ) (a : ℕ → ℚ) : ℚ :=
  n / 2 * (a 1 + a n)

noncomputable def S (n : ℕ) := arithmetic_sum n (λ n, a n)
noncomputable def T (n : ℕ) := arithmetic_sum n (λ n, b n)

axiom ratio (n : ℕ) : (S n) / (T n) = (2 * n + 1) / (4 * n - 2)

theorem sum_value :
  (a 10 / (b 3 + b 18) + a 11 / (b 6 + b 15)) = 41 / 78 := by
  sorry

end sum_value_l424_424368


namespace nat_pair_count_eq_five_l424_424286

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424286


namespace product_of_sums_is_even_l424_424401

-- Definitions of the conditions
def cards := Fin 99 → ℕ -- 99 cards
def numbers_on_sides (a b : Fin 99) := (a + 1) + (b + 1) -- Summing the numbers on each side of the cards

theorem product_of_sums_is_even (cards : Fin 99 → ℕ → ℕ) :
  (∏ i, let a := i; let b := cards i in numbers_on_sides a b) % 2 = 0 :=
sorry

end product_of_sums_is_even_l424_424401


namespace maximize_revenue_l424_424938

theorem maximize_revenue (p : ℝ) (h₁ : p ≤ 30) (h₂ : p = 18.75) : 
  ∃(R : ℝ), R = p * (150 - 4 * p) :=
by
  sorry

end maximize_revenue_l424_424938


namespace cows_count_l424_424090

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end cows_count_l424_424090


namespace total_sum_of_subsets_l424_424244

open Finset

def M : Finset ℕ := {1, 2, 3, 4}

def subset_sum (A : Finset ℕ) : ℤ := 
  A.sum (λ k, (-1)^k * k)

def total_subset_sums : ℕ := 
  (powerset M \ {∅}).sum (λ A, subset_sum A)

theorem total_sum_of_subsets : 
  total_subset_sums = 16 := 
by {
  sorry
}

end total_sum_of_subsets_l424_424244


namespace perpendicularity_transitive_l424_424199

variable {α β γ : Type} [linear_ordered_field α]
variable (l1 l2 l3 : affine_plane α)
variable (plane₁ plane₂ plane₃ : affine_plane α)

-- Assume the conditions
axiom lines_distinct : l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3
axiom planes_distinct : plane₁ ≠ plane₂ ∧ plane₂ ≠ plane₃ ∧ plane₁ ≠ plane₃

-- Define the relationships according to statement
axiom perp₁ : is_perpendicular plane₁ plane₂
axiom parallel : is_parallel plane₂ plane₃

-- The statement to prove
theorem perpendicularity_transitive : is_perpendicular plane₁ plane₃ :=
by
  sorry

end perpendicularity_transitive_l424_424199


namespace field_width_l424_424452

variable width : ℚ -- Define a variable width of type rational

-- Define the conditions
def length_eq_24 : Prop := 24 = 2 * width - 3

-- State the theorem to prove the width is 13.5 meters
theorem field_width :
  length_eq_24 → width = 13.5 :=
by
  intro h,
  -- Proof can be filled out here. For now, we use sorry to skip it
  sorry

end field_width_l424_424452


namespace inverse_proportion_quadrants_l424_424446

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  ∀ (x y : ℝ), y = k^2 / x → (x > 0 → y > 0) ∧ (x < 0 → y < 0) :=
by
  sorry

end inverse_proportion_quadrants_l424_424446


namespace smallest_t_for_temperature_104_l424_424402

theorem smallest_t_for_temperature_104 : 
  ∃ t : ℝ, (-t^2 + 16*t + 40 = 104) ∧ (t > 0) ∧ (∀ s : ℝ, (-s^2 + 16*s + 40 = 104) ∧ (s > 0) → t ≤ s) :=
sorry

end smallest_t_for_temperature_104_l424_424402


namespace painting_area_percentage_l424_424706

-- Define the conditions of the problem
def painting_length : ℚ := 13 / 4
def painting_width : ℚ := 38 / 5

def wall_longest_side : ℚ := 14
def wall_height : ℚ := 9

-- Define the areas based on the conditions
def area_of_painting : ℚ := painting_length * painting_width
def area_of_wall : ℚ := (wall_longest_side * wall_height) / 2

-- Define the percentage calculation
def percentage_of_painting_area_on_wall : ℚ := (area_of_painting / area_of_wall) * 100

-- State the problem statement
theorem painting_area_percentage :
  percentage_of_painting_area_on_wall ≈ 39.21 := by
  sorry

end painting_area_percentage_l424_424706


namespace metal_relative_atomic_mass_is_24_l424_424075

noncomputable def relative_atomic_mass (metal_mass : ℝ) (hcl_mass_percent : ℝ) (hcl_total_mass : ℝ) (mol_mass_hcl : ℝ) : ℝ :=
  let moles_hcl := (hcl_total_mass * hcl_mass_percent / 100) / mol_mass_hcl
  let maximum_molar_mass := metal_mass / (moles_hcl / 2)
  let minimum_molar_mass := metal_mass / (moles_hcl / 2)
  if 20 < maximum_molar_mass ∧ maximum_molar_mass < 28 then
    24
  else
    0

theorem metal_relative_atomic_mass_is_24
  (metal_mass_1 : ℝ)
  (metal_mass_2 : ℝ)
  (hcl_mass_percent : ℝ)
  (hcl_total_mass : ℝ)
  (mol_mass_hcl : ℝ)
  (moles_used_1 : ℝ)
  (moles_used_2 : ℝ)
  (excess : Bool)
  (complete : Bool) :
  relative_atomic_mass 3.5 18.25 50 36.5 = 24 :=
by
  sorry

end metal_relative_atomic_mass_is_24_l424_424075


namespace greatest_third_term_is_16_l424_424852

open Int

theorem greatest_third_term_is_16:
  ∃ (a d : ℤ), 
    (2 * a + 3 * d = 25) ∧ 
    even (a + 2 * d) ∧ 
    (∀ b c, 
      (2 * b + 3 * c = 25) ∧ 
      even (b + 2 * c) →
      (a + 2 * d) ≥ (b + 2 * c)) ∧
    (a + 2 * d = 16) :=
by
  sorry

end greatest_third_term_is_16_l424_424852


namespace union_complements_eq_l424_424369

-- Definitions as per conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define complements
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof statement
theorem union_complements_eq :
  (C_UA ∪ C_UB) = {0, 1, 4} :=
by
  sorry

end union_complements_eq_l424_424369


namespace num_pairs_nat_nums_eq_l424_424276

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424276


namespace tangent_line_parabola_l424_424839

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l424_424839


namespace total_payment_correct_l424_424982

section Discounts

variable (Bob_bill Kate_bill John_bill Sarah_bill : ℝ)
          (Bob_discount Kate_discount John_discount Sarah_discount : ℝ)

def Bob_pay := Bob_bill - Bob_bill * Bob_discount
def Kate_pay := Kate_bill - Kate_bill * Kate_discount
def John_pay := John_bill - John_bill * John_discount
def Sarah_pay := Sarah_bill - Sarah_bill * Sarah_discount

def total_payment := Bob_pay + Kate_pay + John_pay + Sarah_pay

theorem total_payment_correct :
  (let Bob_bill := 35.50;
       Kate_bill := 29.75;
       John_bill := 43.20;
       Sarah_bill := 27.35;
       Bob_discount := 0.0575;
       Kate_discount := 0.0235;
       John_discount := 0.0395;
       Sarah_discount := 0.0945 in
   total_payment Bob_bill Kate_bill John_bill Sarah_bill Bob_discount Kate_discount John_discount Sarah_discount = 128.76945) :=
sorry

end Discounts

end total_payment_correct_l424_424982


namespace seq_arithmetic_exists_N_l424_424069

-- Definition of the sequence {a_n} where a_2 = p and S_n given by a specific formula
variable (p : ℝ) [nontrivial p]

def a : ℕ → ℝ
| 2 := p
| n + 1 := sorry  -- fill in the rule for generating a_{n+1} (details needed)

def S : ℕ → ℝ
| 0 := 0
| n + 1 := S n + a (n + 1)

-- 1. Prove that {a_n} is an arithmetic sequence
theorem seq_arithmetic (p : ℝ) (hp : p ≠ 0) :
  ∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a n + d :=
sorry -- Needs implementation based on problem's conditions and formula

-- 2. Define b_n and find the sum of the first n terms T_n
def b : ℕ → ℝ := sorry -- b_n definition needed from problem

def T : ℕ → ℝ
| 0 := 0
| n + 1 := T n + b (n + 1)

-- 3. Define c_n and prove the existence of N such that c_n belongs to ( , 3 )
def c : ℕ → ℝ := λ n, T n - 2 * n

theorem exists_N (N : ℕ) (hN : ∀ n > N, c n > 0 ∧ c n < 3) : 
  ∃ N, ∀ n > N, (c n ∈ (0, 3)) :=
sorry -- Needs proof and specific value for N (e.g., N = 6)

end seq_arithmetic_exists_N_l424_424069


namespace equation_solved_in_consecutive_integers_l424_424607

theorem equation_solved_in_consecutive_integers :
  ∃ (x y z w : ℕ), (x + y + z + w = 50) ∧ (x + 1 = y) ∧ (y + 1 = z) ∧ (z + 1 = w) :=
begin
  sorry
end

end equation_solved_in_consecutive_integers_l424_424607


namespace hexagon_and_circle_area_equal_l424_424545

-- Circle radius definition and related calculations
def circle_radius : ℝ := 3

-- Definition for the circle's area
def area_circle : ℝ := Real.pi * circle_radius^2

-- Number of sectors
def num_sectors : ℕ := 6

-- Definition for the sectors' area
def area_sector : ℝ := (1/2) * circle_radius^2 * (2 * Real.pi / num_sectors)

-- Total area of the hexagon formed by the sectors
def area_hexagon : ℝ := num_sectors * area_sector

-- The theorem we need to prove
theorem hexagon_and_circle_area_equal :
  area_hexagon = area_circle :=
by
  sorry

end hexagon_and_circle_area_equal_l424_424545


namespace ann_trip_longer_than_mary_l424_424393

-- Define constants for conditions
def mary_hill_length : ℕ := 630
def mary_speed : ℕ := 90
def ann_hill_length : ℕ := 800
def ann_speed : ℕ := 40

-- Define a theorem to express the question and correct answer
theorem ann_trip_longer_than_mary : 
  (ann_hill_length / ann_speed - mary_hill_length / mary_speed) = 13 :=
by
  -- Now insert sorry to leave the proof unfinished
  sorry

end ann_trip_longer_than_mary_l424_424393


namespace marble_problem_l424_424915

theorem marble_problem
  (x : ℕ) (h1 : 144 / x = 144 / (x + 2) + 1) :
  x = 16 :=
sorry

end marble_problem_l424_424915


namespace parabola_equation_l424_424088

theorem parabola_equation (p : ℝ) (h₀ : 0 < p)
  (h₁ : ∃ A B : ℝ × ℝ, ∃ l : ℝ, l = 2 * sqrt 2 * (fst A - p / 2)
    ∧ l = 2 * sqrt 2 * (fst B - p / 2) ∧ ((A.1) ^ 2 = 2 * p * A.2)
    ∧ ((B.1) ^ 2 = 2 * p * B.2) ∧ abs (fst A - p / 2) = 3 ∧ abs (fst A - p / 2) > abs (fst B - p / 2)) :
  2 * p = 4 := 
sorry

end parabola_equation_l424_424088


namespace opqr_shape_l424_424232

-- Definitions of points
variables {x1 y1 x2 y2 : ℝ}
def P := (x1, y1)
def Q := (x2, y2)
def R := (x1 + x2, y1 + y2)
def O := (0, 0)

-- The figure OPQR can be a parallelogram or a straight line, but not a trapezoid.
theorem opqr_shape :
  (let OP := (x1, y1), OQ := (x2, y2), OR := (x1 + x2, y1 + y2) in
    ∃ (config : ℕ), (config = 1 ∨ config = 2) ∧ ¬ (config = 3)) :=
by
  sorry

end opqr_shape_l424_424232


namespace num_pairs_nat_nums_eq_l424_424274

theorem num_pairs_nat_nums_eq (a b : ℕ) (h₁ : a ≥ b) (h₂ : 1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 6) :
  ∃ (p : fin 6 → ℕ × ℕ), (∀ i, (p i).1 ≥ (p i).2) ∧ (∀ i, 1 / (p i).1 + 1 / (p i).2 = 1 / 6) ∧ (∀ i j, i ≠ j → p i ≠ p j) :=
sorry

end num_pairs_nat_nums_eq_l424_424274


namespace ratio_of_wages_l424_424361

def hours_per_day_josh : ℕ := 8
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def wage_per_hour_josh : ℕ := 9
def monthly_total_payment : ℚ := 1980

def hours_per_day_carl : ℕ := hours_per_day_josh - 2

def monthly_hours_josh : ℕ := hours_per_day_josh * days_per_week * weeks_per_month
def monthly_hours_carl : ℕ := hours_per_day_carl * days_per_week * weeks_per_month

def monthly_earnings_josh : ℚ := wage_per_hour_josh * monthly_hours_josh
def monthly_earnings_carl : ℚ := monthly_total_payment - monthly_earnings_josh

def hourly_wage_carl : ℚ := monthly_earnings_carl / monthly_hours_carl

theorem ratio_of_wages : hourly_wage_carl / wage_per_hour_josh = 1 / 2 := by
  sorry

end ratio_of_wages_l424_424361


namespace spinner_even_product_probability_l424_424428

theorem spinner_even_product_probability :
  let A := {1, 2, 3, 4, 5}
  let B := {1, 3, 5}
  (∀ a ∈ A, ∀ b ∈ B, even (a * b)) → 
  ( ∃ z : ℚ, z = 2 / 5 ) :=
by
  sorry

end spinner_even_product_probability_l424_424428


namespace speed_of_first_train_is_correct_l424_424028

-- Define the lengths of the trains
def length_train1 : ℕ := 110
def length_train2 : ℕ := 200

-- Define the speed of the second train in kmph
def speed_train2 : ℕ := 65

-- Define the time they take to clear each other in seconds
def time_clear_seconds : ℚ := 7.695936049253991

-- Define the speed of the first train
def speed_train1 : ℚ :=
  let time_clear_hours : ℚ := time_clear_seconds / 3600
  let total_distance_km : ℚ := (length_train1 + length_train2) / 1000
  let relative_speed_kmph : ℚ := total_distance_km / time_clear_hours 
  relative_speed_kmph - speed_train2

-- The proof problem is to show that the speed of the first train is 80.069 kmph
theorem speed_of_first_train_is_correct : speed_train1 = 80.069 := by
  sorry

end speed_of_first_train_is_correct_l424_424028


namespace inequality_proof_l424_424656

theorem inequality_proof (n : ℕ) (h_n : 2 ≤ n) 
(a : ℕ → ℕ)
(h_inc : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
(h_sum : ∑ i in Finset.range n, (1 : ℚ) / (a i) ≤ 1)
(x : ℝ) :
(∑ i in Finset.range n, 1 / (a i ^ 2 + x^2))^2 ≤ 1 / 2 / (a 1 * (a 1 - 1) + x^2) := by
  sorry

end inequality_proof_l424_424656


namespace triangle_cosine_identity_l424_424797

open Real

variables {A B C a b c : ℝ}

theorem triangle_cosine_identity (h : b = (a + c) / 2) : cos (A - C) + 4 * cos B = 3 :=
sorry

end triangle_cosine_identity_l424_424797


namespace students_before_intersection_equal_l424_424482

-- Define the conditions
def students_after_stop : Nat := 58
def percentage : Real := 0.40
def percentage_students_entered : Real := 12

-- Define the target number of students before stopping
def students_before_stop (total_after : Nat) (entered : Nat) : Nat :=
  total_after - entered

-- State the proof problem
theorem students_before_intersection_equal :
  ∃ (x : Nat), 
  percentage * (x : Real) = percentage_students_entered ∧ 
  students_before_stop students_after_stop x = 28 :=
by
  sorry

end students_before_intersection_equal_l424_424482


namespace oak_trees_cut_down_l424_424478

-- Define the conditions
def initial_oak_trees : ℕ := 9
def final_oak_trees : ℕ := 7

-- Prove that the number of oak trees cut down is 2
theorem oak_trees_cut_down : (initial_oak_trees - final_oak_trees) = 2 :=
by
  -- Proof is omitted
  sorry

end oak_trees_cut_down_l424_424478


namespace hyperbola_asymptotes_proof_l424_424678

noncomputable def focus_coordinate := λ m : ℝ, real.sqrt (9 + m)

def is_on_circle (x y : ℝ) := x^2 + y^2 - 4 * x - 5 = 0

def hyperbola_asymptotes (a b : ℝ) := (λ (x : ℝ), (b/a) * x, λ (x : ℝ), -(b/a) * x)

theorem hyperbola_asymptotes_proof (h : ∀ m, is_on_circle (focus_coordinate m) 0) :
  hyperbola_asymptotes 3 4 = 
    (λ x : ℝ, (4/3) * x, λ x : ℝ, -(4/3) * x) := 
by 
  sorry

end hyperbola_asymptotes_proof_l424_424678


namespace simplify_fraction_l424_424234

theorem simplify_fraction (a b : ℝ) (h : a ≠ b) : 
  (a^(-6) - b^(-6)) / (a^(-3) - b^(-3)) = a^(-6) + a^(-3) * b^(-3) + b^(-6) :=
  sorry

end simplify_fraction_l424_424234


namespace arithmetic_sequence_correct_geometric_sequence_correct_t_n_correct_lambda_max_l424_424208

open Real

noncomputable def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1
noncomputable def geometric_sequence (n : ℕ) : ℕ := 3^(n - 1)
noncomputable def a_n (n : ℕ) : ℕ := arithmetic_sequence n
noncomputable def b_n (n : ℕ) : ℕ := geometric_sequence n
noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

def S_n (n : ℕ) : ℕ := n * (n + 1)

noncomputable def C_n (n : ℕ) : ℚ := ((a_n n + 4 : ℚ) / ((S_n n + n : ℚ) * b_n (n + 1)))
noncomputable def A_n (n : ℕ) : ℚ := (1 - (1 / ((n + 1) * 3^n)))

theorem arithmetic_sequence_correct (n : ℕ) : 
  a_n n = 2 * n - 1 :=
sorry

theorem geometric_sequence_correct (n : ℕ) : 
  b_n n = 3 ^ (n - 1) :=
sorry

noncomputable def T_n (n : ℕ) : ℕ :=
((n - 1) * 3^n).succ

theorem t_n_correct (n : ℕ) : 
  T_n n = (n - 1) * 3^n + 1 :=
sorry

theorem lambda_max (n : ℕ) : 
  ∀ n : ℕ, 1 - (1 / ((n + 1) * 3^n)) ≥ λ / (n + 1) → λ ≤ 5 / 3 :=
sorry

end arithmetic_sequence_correct_geometric_sequence_correct_t_n_correct_lambda_max_l424_424208


namespace ratio_of_x_intercepts_l424_424878

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l424_424878


namespace consecutive_numbers_perfect_square_l424_424407

theorem consecutive_numbers_perfect_square (a : ℕ) (h : a ≥ 1) : 
  (a * (a + 1) * (a + 2) * (a + 3) + 1) = (a^2 + 3 * a + 1)^2 :=
by sorry

end consecutive_numbers_perfect_square_l424_424407


namespace gcd_lcm_sum_l424_424755

theorem gcd_lcm_sum (A B : ℕ) :
  A = Nat.gcd 18 (Nat.gcd 24 30) ∧ B = Nat.lcm 18 (Nat.lcm 24 30) →
  A + B = 366 :=
begin
  sorry
end

end gcd_lcm_sum_l424_424755


namespace tangent_line_parabola_l424_424827

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l424_424827


namespace prob_A_at_least_one_phase_prob_A_more_phases_than_B_l424_424956

-- Conditions: Probabilities of company A obtaining the contracts
def prob_A1 : ℝ := 2 / 3
def prob_A2 : ℝ := 1 / 2
def prob_A3 : ℝ := 1 / 4

-- The questions transformed into proof goals
theorem prob_A_at_least_one_phase :
  (1 - (1 - prob_A1) * (1 - prob_A2) * (1 - prob_A3)) = 7 / 8 :=
by
  sorry

theorem prob_A_more_phases_than_B :
  (prob_A1 * prob_A2 * (1 - prob_A3) +
  prob_A1 * (1 - prob_A2) * prob_A3 +
  (1 - prob_A1) * prob_A2 * prob_A3 +
  prob_A1 * prob_A2 * prob_A3) = 11 / 24 :=
by
  sorry

end prob_A_at_least_one_phase_prob_A_more_phases_than_B_l424_424956


namespace regression_lines_intersect_at_sample_center_l424_424022

variable (a b : ℝ)

-- Define the linear regression lines l₁ and l₂
variable (l₁ l₂ : ℝ → ℝ) 

-- Conditions: Average observed values of x and y
axiom AvgX : ∀ (x_values : Fin 10 → ℝ), average x_values = a
axiom AvgY : ∀ (y_values : Fin 10 → ℝ), average y_values = b
axiom AvgX_2 : ∀ (x_values_2 : Fin 15 → ℝ), average x_values_2 = a
axiom AvgY_2 : ∀ (y_values_2 : Fin 15 → ℝ), average y_values_2 = b

-- The main proof problem
theorem regression_lines_intersect_at_sample_center :
  l₁ a = b ∧ l₂ a = b := 
sorry

end regression_lines_intersect_at_sample_center_l424_424022


namespace gg_points_and_sum_l424_424432

/-- Define the function g with specific points -/
def g : ℕ → ℕ 
| 1 := 6
| 2 := 4
| 4 := 2
| _ := 0  -- g is not fully defined, but we provide a default case

/-- Define the composed function gg(x) = g(g(x)) -/
def gg (x : ℕ) : ℕ := g (g x)

/-- The main proof problem -/
theorem gg_points_and_sum :
  gg 2 = 2 ∧ gg 1 = 2 ∧ (2 * 2 + 1 * 2 = 6) := 
by
  sorry

end gg_points_and_sum_l424_424432


namespace floor_sqrt_47_l424_424619

theorem floor_sqrt_47 : (⌊real.sqrt 47⌋ = 6) :=
by {
  have h1: 6^2 = 36 := by norm_num,
  have h2: 7^2 = 49 := by norm_num,
  have h3: 6 < real.sqrt 47 := by {
    apply real.sqrt_lt.mpr,
    exact ⟨by norm_num, by linarith⟩,
  },
  have h4: real.sqrt 47 < 7 := by {
    apply real.sqrt_lt.mpr,
    exact ⟨by linarith, by norm_num⟩,
  },
  rw nat.floor,
  simp only [real.sqrt_eq_rpow, real.rpow_nat_cast, real.sqrt_eq_rpow, real.rpow_nat_cast],
  linarith,
}

end floor_sqrt_47_l424_424619


namespace problem_1_monotonic_problem_2_inequality_l424_424213

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp (x - 1) * Real.log x
def g (x : ℝ) : ℝ := x^2 - x

-- Problem 1: Prove that f(x) is monotonically increasing on (0, +∞)
theorem problem_1_monotonic :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) < f(y) :=
sorry

-- Problem 2: Prove that for x in (0, 2), f(x) ≤ g(x)
theorem problem_2_inequality :
  ∀ x : ℝ, 0 < x → x < 2 → f(x) ≤ g(x) :=
sorry

end problem_1_monotonic_problem_2_inequality_l424_424213


namespace tangent_line_parabola_l424_424840

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l424_424840


namespace length_of_train_is_correct_l424_424917

noncomputable def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_sec

theorem length_of_train_is_correct (speed_km_hr : ℝ) (time_sec : ℝ) (expected_length : ℝ) :
  speed_km_hr = 60 → time_sec = 21 → expected_length = 350.07 →
  train_length speed_km_hr time_sec = expected_length :=
by
  intros h1 h2 h3
  simp [h1, h2, train_length]
  sorry

end length_of_train_is_correct_l424_424917


namespace tangent_line_parabola_l424_424838

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l424_424838


namespace sum_of_extrema_l424_424855

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem sum_of_extrema : 
  let a := -2
  let b := 3
  (f 1) + (f (-2)) = -1 := 
by
  let a := (-2 : ℝ)
  let c := 3
  let fm := f 1
  let fn := f (-2)
  have h1 : f 1 = 4 := sorry
  have h2 : f (-2) = -5 := sorry
  calc
    (f 1) + (f (-2)) = 4 + -5 : by rw [h1, h2]
    ...             = -1 : by norm_num

end sum_of_extrema_l424_424855


namespace possible_hands_l424_424514

theorem possible_hands (n : ℕ) (h : n ≥ 2) : 
  -- Statement asserting the number of possible hands for the first player
  ∃ k: ℕ, k = 2^n - 2 :=
begin
  use (2^n - 2),
  sorry,
end

end possible_hands_l424_424514


namespace find_H_over_G_l424_424820

variable (G H : ℤ)
variable (x : ℝ)

-- Conditions
def condition (G H : ℤ) (x : ℝ) : Prop :=
  x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 ∧
  (↑G / (x + 7) + ↑H / (x * (x - 6)) = (x^2 - 3 * x + 15) / (x^3 + x^2 - 42 * x))

-- Theorem Statement
theorem find_H_over_G (G H : ℤ) (x : ℝ) (h : condition G H x) : (H : ℝ) / G = 15 / 7 :=
sorry

end find_H_over_G_l424_424820


namespace percentage_error_in_calculated_area_l424_424467

theorem percentage_error_in_calculated_area :
  let initial_length_error := 0.03 -- 3%
  let initial_width_error := -0.02 -- 2% deficit
  let temperature_change := 15 -- °C
  let humidity_increase := 20 -- %
  let length_error_temp_increase := (temperature_change / 5) * 0.01
  let width_error_humidity_increase := (humidity_increase / 10) * 0.005
  let total_length_error := initial_length_error + length_error_temp_increase
  let total_width_error := initial_width_error + width_error_humidity_increase
  let total_percentage_error := total_length_error + total_width_error
  total_percentage_error * 100 = 3 -- 3%
:= by
  sorry

end percentage_error_in_calculated_area_l424_424467


namespace Ramsey_number_bound_l424_424637

noncomputable def Ramsey_number (k : ℕ) : ℕ := sorry

theorem Ramsey_number_bound (k : ℕ) (h : k ≥ 3) : Ramsey_number k > 2^(k / 2) := sorry

end Ramsey_number_bound_l424_424637


namespace mat_cost_per_square_meter_l424_424330

def cost_per_square_meter (length width height total_expenditure : ℝ) : ℝ :=
  let floor_area := length * width
  let long_walls_area := 2 * (length * height)
  let short_walls_area := 2 * (width * height)
  let total_area := floor_area + long_walls_area + short_walls_area
  total_expenditure / total_area

theorem mat_cost_per_square_meter :
  cost_per_square_meter 20 15 5 47500 = 73.08 :=
by
  sorry

end mat_cost_per_square_meter_l424_424330


namespace product_of_possible_values_of_x_l424_424313

theorem product_of_possible_values_of_x : 
  (∀ x : ℝ, (|18 / x + 4| = 3) → x = -18 ∨ x = -(18 / 7)) →
  ((-18) * (-(18 / 7)) = 324 / 7) :=
by
  intros h
  have hx1 : (-18) * (-(18 / 7)) = 324 / 7, from sorry
  exact hx1

end product_of_possible_values_of_x_l424_424313


namespace handshake_problem_l424_424427

theorem handshake_problem : ∃ n : ℕ, (n * (n - 1)) / 2 = 153 ∧ n = 18 :=
by
  have h : ∀ n : ℕ, (n * (n - 1)) / 2 = 153 → n = 18 := by
    intro n hn
    -- Here we would do the actual proof which is omitted
    sorry
  exact ⟨18, by calc
    (18 * 17) / 2 = 306 / 2 : by norm_num
    ... = 153 : by norm_num, h 18⟩

end handshake_problem_l424_424427


namespace polygon_sides_eq_six_l424_424011

theorem polygon_sides_eq_six (n : ℕ) (S_i S_e : ℕ) :
  S_i = 2 * S_e →
  S_e = 360 →
  (n - 2) * 180 = S_i →
  n = 6 :=
by
  sorry

end polygon_sides_eq_six_l424_424011


namespace inequality_sum_squares_products_l424_424536

theorem inequality_sum_squares_products {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_sum_squares_products_l424_424536


namespace monotonicity_f_gteq0_monotonicity_f_lt0_max_value_f_l424_424694

noncomputable def f (a x : ℝ) : ℝ := Real.log x + a * x^2 + (2 * a + 1) * x

theorem monotonicity_f_gteq0 (a : ℝ) (ha : a ≥ 0) : 
  ∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → f a x ≤ f a y := 
sorry

theorem monotonicity_f_lt0 (a : ℝ) (ha : a < 0) : 
  ∀ x y : ℝ, (0 < x ∧ x < -1/(2*a) ∧ x < y ≤ -1/(2*a)) → f a x < f a y ∧
              (-1/(2*a) < x ∧ x < y ∧ 0 < y) → f a x > f a y := 
sorry

theorem max_value_f (a : ℝ) (ha : a < 0) : 
  ∀ x : ℝ, 0 < x → f a x ≤ -3/(4*a) - 2 := 
sorry

end monotonicity_f_gteq0_monotonicity_f_lt0_max_value_f_l424_424694


namespace smallest_model_length_l424_424416

theorem smallest_model_length (full_size : ℕ) (mid_size_factor smallest_size_factor : ℚ) :
  full_size = 240 →
  mid_size_factor = 1 / 10 →
  smallest_size_factor = 1 / 2 →
  (full_size * mid_size_factor) * smallest_size_factor = 12 :=
by
  intros h_full_size h_mid_size_factor h_smallest_size_factor
  sorry

end smallest_model_length_l424_424416


namespace possible_scenarios_count_l424_424723

theorem possible_scenarios_count : 
  ∃ (w d l : ℕ), 
    (w + d + l = 15) ∧ 
    (3 * w + d = 22) ∧ 
    (∃ T : finset (ℕ × ℕ × ℕ), 
       T = { (4, 10, 1), (5, 7, 3), (6, 4, 5), (7, 1, 7) } ∧
       T.card = 4) := 
sorry

end possible_scenarios_count_l424_424723


namespace count_strictly_increasing_digits_l424_424160

theorem count_strictly_increasing_digits : 
  (∑ k in Finset.range 9, Nat.choose 9 k.succ) = 502 :=
by
  sorry

end count_strictly_increasing_digits_l424_424160


namespace sum_of_roots_l424_424631
-- Importing the entire Mathlib for necessary mathematical constructs.

-- Defining the polynomial and establishing the proof of the sum of its roots.
theorem sum_of_roots :
  let p := (λ x : ℝ, (x - 1) ^ 2008 + 2 * (x - 2) ^ 2007 + 3 * (x - 3) ^ 2006 + 4 * (x - 4) ^ 2005 + 5 * (x - 5) ^ 2004 + 6 * (x - 6) ^ 2003) in
  (sum (roots p)) = 4014 :=
by
  sorry

end sum_of_roots_l424_424631


namespace probability_shaded_region_l424_424947

-- Definitions of angles and their properties in the problem context
def total_angle : ℝ := 360
def large_region_angle : ℝ := 140
def small_region_angle : ℝ := 20
def four_regions_angle_sum : ℝ := 200   -- Derived from 4 * 50 where each region has 50 degrees

-- Proving the probability of landing on a shaded region is 2/3
theorem probability_shaded_region :
  let total_degrees := 360 in
  let shaded_regions_sum := 140 + 50 + 50 in
  shaded_regions_sum / total_degrees = 2 / 3 :=
by
  -- Direct computations and solution steps skipped
  sorry

end probability_shaded_region_l424_424947


namespace cricket_bat_profit_percentage_l424_424963

/-- A sells a cricket bat to B at a certain profit percentage. B sells it to C at a profit of 25%.
    C pays $228 for it. The cost price of the cricket bat for A is $152.
    Prove that the profit percentage for A when selling the bat to B is 20%. -/
theorem cricket_bat_profit_percentage:
  let CP_A := 152
  let SP_C := 228
  let profit_BC_percentage := 25 in
  let CP_B := SP_C / (1 + profit_BC_percentage / 100) in
  let SP_AB := CP_B in
  let profit_A := SP_AB - CP_A in
  let profit_A_percentage := (profit_A / CP_A) * 100 in
  profit_A_percentage = 20 :=
by
  -- Define constants
  let CP_A := 152
  let SP_C := 228
  let profit_BC_percentage := 25
  -- Calculate CP_B and SP_AB
  let CP_B := SP_C / (1 + profit_BC_percentage / 100)
  let SP_AB := CP_B
  -- Calculate A's profit
  let profit_A := SP_AB - CP_A
  let profit_A_percentage := (profit_A / CP_A) * 100
  -- Show the expected profit percentage
  show profit_A_percentage = 20 from sorry

end cricket_bat_profit_percentage_l424_424963


namespace ratio_of_areas_l424_424895

-- Define the conditions of the problem
variable (SQ : Type) [Square SQ]    -- Assume SQ is the type of all squares, and it satisfies some square properties.

-- Define the vertices and midpoint condition
axiom vertices_at_midpoints (s : SQ) : vertices at midpoints ∧ symmetric across quadrants

-- Define the area function and assume shaded and white area as given in a quadrant
variable (area : SQ → ℕ)
def shaded_area (sq : SQ) := 5 * area sq -- Shaded triangles per quadrant
def white_area (sq : SQ) := 3 * area sq  -- White triangles per quadrant

-- Resulting areas for the whole figure
def total_shaded_area (sq : SQ) := 4 * shaded_area sq
def total_white_area (sq : SQ) := 4 * white_area sq

-- Define the ratio function to compare shaded and white areas
def ratio (a b : ℕ) := a / b

theorem ratio_of_areas (s : SQ) (area_nonzero : area s ≠ 0):
  ratio (total_shaded_area s) (total_white_area s) = 5 / 3 := by
  sorry

end ratio_of_areas_l424_424895


namespace total_pages_in_book_l424_424980

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end total_pages_in_book_l424_424980


namespace find_n_l424_424500

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 :=
by {
    -- Let's assume there exists an integer n such that the given condition holds
    use 4,
    -- We now prove the condition and the conclusion
    split,
    -- This simplifies the left-hand side of the condition to 15, achieving the goal
    calc 
        4 + (4 + 1) + (4 + 2) = 4 + 5 + 6 : by rfl
        ... = 15 : by norm_num,
    -- The conclusion directly follows
    rfl
}

end find_n_l424_424500


namespace nat_pair_count_eq_five_l424_424290

theorem nat_pair_count_eq_five :
  (∃ n : ℕ, n = {p : ℕ × ℕ | p.1 ≥ p.2 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6}.toFinset.card) ∧ n = 5 :=
by
  sorry

end nat_pair_count_eq_five_l424_424290


namespace analogical_reasoning_l424_424905

theorem analogical_reasoning {a b c : ℝ} (h1 : c ≠ 0) : 
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := 
by 
  sorry

end analogical_reasoning_l424_424905


namespace pyramid_cross_section_area_l424_424465

-- Definitions of the given conditions
variables {Pyramid : Type} [Has_Side_Base (Pyramid) (ABC : Real) 4] 
variables [Has_Lateral_Edge (Pyramid) T (ABC)]
variables {AC BT} (mid_AC : Midpoint (AC)) (mid_BT : Midpoint (BT))

-- Constant distance from vertex T to the intersecting plane
constant vertex_T : Real 
constant distance_T : vertex_T -> Real := 1/2

-- The proof goal stating the cross-section area of the pyramid
theorem pyramid_cross_section_area (P : Pyramid)
  [Is_Perpendicular_Lateral_Edge (P) T (ABC)]
  [Plane_Intersects_Midpoints (P) mid_AC mid_BT]
  [Plane_Parallel_to_Median (P) BD]
  : Area (Cross_Section (P) mid_AC mid_BT) = 6 / Real.sqrt 5 :=
begin
  sorry
end

end pyramid_cross_section_area_l424_424465


namespace reciprocal_of_4_l424_424849

variable (x : ℝ) (h : x ≠ 0)

def reciprocal (x : ℝ) : ℝ := 1 / x

def problem_statement : Prop :=
  reciprocal 4 = 1 / 4

theorem reciprocal_of_4:
  problem_statement :=
by
  unfold problem_statement
  unfold reciprocal
  exact rfl

end reciprocal_of_4_l424_424849


namespace initial_peanuts_l424_424351

-- Definitions based on conditions
def peanuts_added := 8
def total_peanuts_now := 12

-- Statement to prove
theorem initial_peanuts (initial_peanuts : ℕ) (h : initial_peanuts + peanuts_added = total_peanuts_now) : initial_peanuts = 4 :=
sorry

end initial_peanuts_l424_424351


namespace factorial_floor_expression_l424_424989

open Nat

theorem factorial_floor_expression :
  ⇑floor ((2012! + 2008!) / (2011! + 2010!)) = 1 :=
sorry

end factorial_floor_expression_l424_424989


namespace pawns_rearrangement_impossible_l424_424543

theorem pawns_rearrangement_impossible (N : ℕ) (hN : odd N) :
  ∃ (p : ℕ → ℕ) (hp1 : ∀ i, p i < N) (hp2 : ∀ i, ∃ j, p i = j) (hp3 : ∀ i j, adjacent i j → p i ≠ j), false :=
by
  sorry

-- Here, adjacent is a placeholder for the definition of cells being adjacent on a chessboard.
-- The actual definition of adjacency would need to be provided for the theorem to be complete.

end pawns_rearrangement_impossible_l424_424543


namespace final_milk_quantity_l424_424055

theorem final_milk_quantity (initial_milk : ℝ) (removal_1 : ℝ) (addition_1 : ℝ) (removal_2 : ℝ) (vessel_capacity : ℝ) :
  initial_milk = 60 →
  removal_1 = 9 →
  addition_1 = 9 →
  removal_2 = 9 →
  vessel_capacity = 60 →
  let milk_after_first_removal := initial_milk - removal_1 in
  let milk_after_first_addition := milk_after_first_removal in
  let milk_fraction := milk_after_first_addition / vessel_capacity in
  let milk_removed_second := milk_fraction * removal_2 in
  let final_milk_quantity := milk_after_first_addition - milk_removed_second in
  final_milk_quantity = 43.35 :=
by
  intros _ _ _ _ _ 
  let milk_after_first_removal := initial_milk - removal_1 
  let milk_after_first_addition := milk_after_first_removal 
  let milk_fraction := milk_after_first_addition / vessel_capacity 
  let milk_removed_second := milk_fraction * removal_2 
  let final_milk_quantity := milk_after_first_addition - milk_removed_second 
  sorry

end final_milk_quantity_l424_424055


namespace arithmetic_sequence_properties_l424_424662

noncomputable def general_term_formula (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : Prop :=
∀ n, a n = -2.8 * n + 28.4

noncomputable def maximum_sum (S : ℕ → ℝ) (n_max : ℕ) (max_sum : ℝ) : Prop :=
S n_max = max_sum ∧ (∀ n, S n ≤ max_sum)

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 20)
  (h2 : a 10 = 6) :
  general_term_formula a (-2.8) ∧ maximum_sum S 10 144 :=
by
  sorry

end arithmetic_sequence_properties_l424_424662


namespace books_to_sell_to_reach_goal_l424_424436

-- Definitions for conditions
def initial_savings : Nat := 10
def clarinet_cost : Nat := 90
def book_price : Nat := 5
def halfway_goal : Nat := clarinet_cost / 2

-- The primary theorem to prove
theorem books_to_sell_to_reach_goal : 
  initial_savings + (initial_savings = 0 → clarinet_cost) / book_price = 25 :=
by
  -- Proof steps (skipped in the statement)
  sorry

end books_to_sell_to_reach_goal_l424_424436


namespace james_paid_per_shirt_after_discount_l424_424353

theorem james_paid_per_shirt_after_discount : 
  ∀ (num_shirts : ℕ) (total_cost : ℝ) (discount_rate : ℝ), 
  num_shirts = 3 →
  total_cost = 60 →
  discount_rate = 0.40 →
  (total_cost * (1 - discount_rate)) / num_shirts = 12 :=
by
  intros num_shirts total_cost discount_rate h_num_shirts h_total_cost h_discount_rate
  rw [h_num_shirts, h_total_cost, h_discount_rate]
  norm_num
  sorry

end james_paid_per_shirt_after_discount_l424_424353


namespace sum_of_adjacent_triangles_eq_fourth_triangle_l424_424485

-- Define the geometrical setup
variable (A B C O : Point)
variable (A1 B1 C1 : Point)

-- Assume conditions
axiom inside_triangle : InTriangle O A B C
axiom parallel_AA1_BC : Parallel AA1 BC
axiom parallel_BB1_CA : Parallel BB1 CA
axiom parallel_CC1_AB : Parallel CC1 AB

-- Define areas
variable (S_a S_b S_c S : ℝ)

-- Assume the areas are given by some function (the specific function defining area is not important for the statement)
axiom area_triangles : Area A AA1 C1 = S_a ∧ Area B BB1 A1 = S_b ∧ Area C CC1 B1 = S_c
axiom area_fourth_triangle : Area A1 B1 C1 = S

-- The theorem to prove
theorem sum_of_adjacent_triangles_eq_fourth_triangle :
  S_a + S_b + S_c = S :=
sorry

end sum_of_adjacent_triangles_eq_fourth_triangle_l424_424485


namespace boxed_boxed_15_eq_60_l424_424186

/-- The sum of the positive factors of a number. -/
def boxed (n : ℕ) : ℕ :=
  (Divisors n).sum

/-- The statement to be proved: for \( n = 15 \), boxed(boxed(n)) = 60. -/
theorem boxed_boxed_15_eq_60 : boxed (boxed 15) = 60 :=
by 
sorry

end boxed_boxed_15_eq_60_l424_424186


namespace number_of_increasing_digits_l424_424163

theorem number_of_increasing_digits : 
  (∑ k in finset.range 10, if 2 ≤ k then nat.choose 9 k else 0) = 502 :=
by
  sorry

end number_of_increasing_digits_l424_424163


namespace square_nonneg_of_nonneg_l424_424930

theorem square_nonneg_of_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x^2 :=
sorry

end square_nonneg_of_nonneg_l424_424930


namespace problem_statement_l424_424674

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end problem_statement_l424_424674


namespace range_of_a_l424_424245

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (log 2 (x - 1) < 1) ∧ (|x - a| < 2)) ↔ (-1 < a ∧ a < 5) :=
by
  sorry

end range_of_a_l424_424245


namespace Murtha_pebble_collection_sum_l424_424780

def a (n : ℕ) : ℕ := 2 + (n - 1) * 3

theorem Murtha_pebble_collection_sum :
  (∑ i in finset.range (15 + 1), a (i + 1)) = 345 := by
  sorry

end Murtha_pebble_collection_sum_l424_424780


namespace simplify_fraction_l424_424420

variable (x y : ℕ)

theorem simplify_fraction (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 :=
by
  sorry

end simplify_fraction_l424_424420


namespace basketball_game_score_l424_424936

theorem basketball_game_score 
  (a r b d : ℕ)
  (H1 : a = b)
  (H2 : a + a * r + a * r^2 = b + (b + d) + (b + 2 * d))
  (H3 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (H4 : r = 3)
  (H5 : a = 3)
  (H6 : d = 10)
  (H7 : a * (1 + r) = 12)
  (H8 : b * (1 + 3 + (b + d)) = 16) :
  a + a * r + b + (b + d) = 28 :=
by simp [H4, H5, H6, H7, H8]; linarith

end basketball_game_score_l424_424936


namespace range_of_a_l424_424701

theorem range_of_a (P : set ℝ) (M : set ℝ) (a : ℝ) 
  (hP : P = {x : ℝ | -1 ≤ x ∧ x ≤ 1}) 
  (hM : M = {-a, a}) 
  (hUnion : P ∪ M = P) :
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l424_424701


namespace bag_of_balls_problem_l424_424477

theorem bag_of_balls_problem :
  (∀ events, (events.1 = "both are white balls" ∧ events.2 = "both are black balls") → events.1 ∩ events.2 = ∅) ∧
  (∀ events, (events.1 = "at least one white ball" ∧ events.2 = "both are black balls") → events.1ᶜ = events.2) :=
by sorry

end bag_of_balls_problem_l424_424477


namespace lines_parallel_or_concurrent_l424_424331

variables {A B C D P Q E F : Type*}

-- Definitions and Conditions
def is_trapezoid (ABCD : A × B × C × D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 90 ∧ b ≠ 90 ∧ c ≠ 90 ∧ d ≠ 90

def intersect (AC BD : Type*) : Prop :=
  ∃ E : Type*, AC ∩ BD = E

def feet_of_altitude (P Q : Type*) : Prop :=
  ∃ (A B : Type*), P ∈ altitude_foot A B ∧ Q ∈ altitude_foot B A

def circumscribed_circle (T : Type*) : Type* := sorry -- definition of circumscribed circle

def circles_intersect (circle1 circle2 : Type*) (F E : Type*) : Prop :=
  F ≠ E ∧ F ∈ (circle1 ∩ circle2)

-- Theorem Statement
theorem lines_parallel_or_concurrent
  (ABCD : A × B × C × D)
  (AC BD : Type*)
  (P Q E F : Type*)
  (h_trapezoid : is_trapezoid ABCD)
  (h_intersect : intersect AC BD)
  (h_feet : feet_of_altitude P Q)
  (h_circles : circles_intersect (circumscribed_circle (triangle C E Q)) (circumscribed_circle (triangle D E P)) F E) :
  (∃ (P_inter Q_inter : Type*), (P_inter ∈ line_through A P ∧ Q_inter ∈ line_through B Q 
    ∧ line_through P_inter Q_inter ∥ line_through E F)) 
  ∨ ∃ (O : Type*), O ∈ (line_through A P ∩ line_through B Q ∩ line_through E F) :=
sorry

end lines_parallel_or_concurrent_l424_424331


namespace cows_count_l424_424091

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end cows_count_l424_424091


namespace average_speed_is_35_l424_424130

-- Given constants
def distance : ℕ := 210
def speed_difference : ℕ := 5
def time_difference : ℕ := 1

-- Definition of time for planned speed and actual speed
def planned_time (x : ℕ) : ℚ := distance / (x - speed_difference)
def actual_time (x : ℕ) : ℚ := distance / x

-- Main theorem to be proved
theorem average_speed_is_35 (x : ℕ) (h : (planned_time x - actual_time x) = time_difference) : x = 35 :=
sorry

end average_speed_is_35_l424_424130


namespace gcd_38_23_l424_424030

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

theorem gcd_38_23 : gcd 38 23 = 1 := by
  sorry

end gcd_38_23_l424_424030


namespace max_sqrt_sum_eq_sqrt69_l424_424767

noncomputable def max_sqrt_sum (a b c : ℝ) : ℝ :=
  √(3 * a + 2) + √(3 * b + 2) + √(3 * c + 2)

theorem max_sqrt_sum_eq_sqrt69 (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 7) : 
  max_sqrt_sum a b c ≤ √69 :=
sorry

end max_sqrt_sum_eq_sqrt69_l424_424767


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424284

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424284


namespace count_strictly_increasing_digits_l424_424159

theorem count_strictly_increasing_digits : 
  (∑ k in Finset.range 9, Nat.choose 9 k.succ) = 502 :=
by
  sorry

end count_strictly_increasing_digits_l424_424159


namespace range_of_b_l424_424315

theorem range_of_b (b : ℝ) (hb : b > 0) : (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) :=
by
  sorry

end range_of_b_l424_424315


namespace square_inscribed_in_triangle_l424_424107

theorem square_inscribed_in_triangle (ABC : Triangle) (s r : ℝ) 
  (square_inscribed : Square) 
  (square_properties : 
    square_inscribed.two_vertices_on_base ABC ∧ 
    square_inscribed.two_vertices_on_sides ABC ∧ 
    square_inscribed.side_length = s ∧ 
    inscribed_circle_radius ABC = r) :
  sqrt(2) * r < s ∧ s < 2 * r :=
sorry

end square_inscribed_in_triangle_l424_424107


namespace registered_voters_vote_candidate_a_l424_424058

theorem registered_voters_vote_candidate_a 
  (total_voters : ℕ)
  (democrat_percentage : ℕ)
  (republican_percentage : ℕ)
  (democrat_vote_for_candidate_A_percentage : ℕ)
  (republican_vote_for_candidate_A_percentage : ℕ) :
  democrat_percentage = 70 → 
  republican_percentage = 30 → 
  democrat_vote_for_candidate_A_percentage = 80 → 
  republican_vote_for_candidate_A_percentage = 30 → 
  total_voters ≥ 100 → 
  let democrat_voters := total_voters * 70 / 100 in
  let republican_voters := total_voters * 30 / 100 in
  let democrat_votes_for_A := democrat_voters * 80 / 100 in
  let republican_votes_for_A := republican_voters * 30 / 100 in
  let total_votes_for_A := democrat_votes_for_A + republican_votes_for_A in
  (total_votes_for_A * 100 / total_voters) = 65 :=
begin
  intros,
  sorry
end

end registered_voters_vote_candidate_a_l424_424058


namespace remainder_11_power_1995_mod_5_l424_424042

theorem remainder_11_power_1995_mod_5 : (11 ^ 1995) % 5 = 1 := by
  -- We use the fact that 11 ≡ 1 (mod 5)
  have h : 11 % 5 = 1 := by norm_num
  -- Consequently, 11^1995 ≡ 1^1995 (mod 5)
  rw [← Nat.pow_mod, h]
  -- Simplify 1^1995 to 1
  norm_num
  -- Final result
  exact Nat.mod_self 5

end remainder_11_power_1995_mod_5_l424_424042


namespace range_of_a_l424_424815

def function_increasing_on_interval {a : ℝ} (f : ℝ → ℝ) (interval : set ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) (a : ℝ) := -x^2 + 2*(a-1)*x + 2

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ∈ set.Iic 4 → y ∈ set.Iic 4 → x < y → f x a < f y a) ↔ a ≥ 5 := 
by
  sorry

end range_of_a_l424_424815


namespace three_friends_at_least_50_l424_424796

theorem three_friends_at_least_50 :
  ∀ (a b c d e f g : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    a + b + c + d + e + f + g = 100 →
    ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
      x + y + z ≥ 50 ∧
      (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f ∨ x = g) ∧
      (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f ∨ y = g) ∧
      (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e ∨ z = f ∨ z = g) :=
begin
  sorry
end

end three_friends_at_least_50_l424_424796


namespace parallelogram_area_l424_424308

def vertices : list (ℝ × ℝ) := [(0, 0), (5, 0), (1, 10), (6, 10)]

theorem parallelogram_area : 
  let base := 5
  let height := 10
  let area := base * height
  area = 50 := 
by 
  sorry

end parallelogram_area_l424_424308


namespace remainder_for_second_number_l424_424627

theorem remainder_for_second_number (G R1 : ℕ) (first_number second_number : ℕ)
  (hG : G = 144) (hR1 : R1 = 23) (hFirst : first_number = 6215) (hSecond : second_number = 7373) :
  ∃ q2 R2, second_number = G * q2 + R2 ∧ R2 = 29 := 
by {
  -- Ensure definitions are in scope
  exact sorry
}

end remainder_for_second_number_l424_424627


namespace pow_reciprocal_product_l424_424035

theorem pow_reciprocal_product (a b : ℚ) (h : b = a⁻¹) (n : ℕ) :
  (a^n) * (b^n) = 1 := by
  sorry

noncomputable def problem : Prop :=
  pow_reciprocal_product (9/8) (8/9) rfl 4

end pow_reciprocal_product_l424_424035


namespace tan_a_pi_over_six_eq_neg_sqrt_three_l424_424229

theorem tan_a_pi_over_six_eq_neg_sqrt_three (a : ℝ) (h : 3^a = 81) : 
  Real.tan (a * Real.pi / 6) = -Real.sqrt 3 :=
sorry

end tan_a_pi_over_six_eq_neg_sqrt_three_l424_424229


namespace correct_equation_l424_424390

-- Condition 1: Machine B transports 60 kg more per hour than Machine A
def machine_B_transports_more (x : ℝ) : Prop := 
  x + 60

-- Condition 2: Time to transport 500 kg by Machine A equals time 
-- to transport 800 kg by Machine B.
def transportation_time_eq (x : ℝ) : Prop :=
  500 / x = 800 / (x + 60)

-- Theorem statement: Prove the correct equation for given conditions
theorem correct_equation (x : ℝ) (h1 : machine_B_transports_more x) (h2 : transportation_time_eq x) : 
  500 / x = 800 / (x + 60) :=
  by
    sorry

end correct_equation_l424_424390


namespace domain_of_g_l424_424036

noncomputable def g (x : ℝ) : ℝ := (x + 6) / (Real.sqrt (x^2 - 4*x - 5) * (x - 3))

theorem domain_of_g :
  {x : ℝ | g x ∈ ℝ} = {x : ℝ | x < -1 ∨ x > 5} :=
by
  sorry

end domain_of_g_l424_424036


namespace cows_now_l424_424092

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end cows_now_l424_424092


namespace field_width_l424_424453

variable width : ℚ -- Define a variable width of type rational

-- Define the conditions
def length_eq_24 : Prop := 24 = 2 * width - 3

-- State the theorem to prove the width is 13.5 meters
theorem field_width :
  length_eq_24 → width = 13.5 :=
by
  intro h,
  -- Proof can be filled out here. For now, we use sorry to skip it
  sorry

end field_width_l424_424453


namespace num_increasing_digits_l424_424177

theorem num_increasing_digits :
  let C := λ (n k : ℕ), Nat.choose n k in
  ∑ k in Finset.range 8, C 9 (k + 2) = 502 :=
by
  sorry

end num_increasing_digits_l424_424177


namespace leak_empty_tank_time_l424_424788

theorem leak_empty_tank_time (fill_time_A : ℝ) (fill_time_A_with_leak : ℝ) (leak_empty_time : ℝ) :
  fill_time_A = 6 → fill_time_A_with_leak = 9 → leak_empty_time = 18 :=
by
  intros hA hL
  -- Here follows the proof we skip
  sorry

end leak_empty_tank_time_l424_424788


namespace prime_divides_3pplus1_2pplus1_l424_424934

theorem prime_divides_3pplus1_2pplus1 (p : ℕ) [Fact p.Prime] : 
  (p ∣ (3 ^ (p + 1) - 2 ^ (p + 1))) → (p = 5) :=
by {
  sorry,
}

end prime_divides_3pplus1_2pplus1_l424_424934


namespace problem1_problem2_problem3_l424_424210

-- Definitions and assumptions based on conditions in the problem
variable (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b^2 = 3 * c^2)

-- Definition for the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Points and vectors definitions
def B : ℝ × ℝ := (0, b)
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)

-- Given condition
def D : ℝ × ℝ := (-b^2 / c, 0)

-- Problem 1: Prove that triangle BF1F2 is an equilateral triangle
theorem problem1 : ∀ (b c : ℝ), b^2 = 3 * c^2 → angle B F2 F1 = π / 3 := sorry

-- Problem 2: Prove the equation of the ellipse
theorem problem2 : ∀ (x y : ℝ), ellipse x y ↔ ((x^2 / 4) + (y^2 / 3) = 1) := sorry

-- Problem 3: Prove the existence of a fixed point N on the x-axis.
theorem problem3 : ∃ N : ℝ × ℝ, N = (4, 0) := sorry

end problem1_problem2_problem3_l424_424210


namespace correct_statements_l424_424253

open Real

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-3, 4)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (norm v1 * norm v2)

noncomputable def are_collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ μ : ℝ, v1 = (μ • v2)

theorem correct_statements :
  (cos_angle vector_a vector_b = sqrt 2 / 10) ∧ 
  (∀ λ : ℝ, are_collinear (vector_a + λ • vector_b) (vector_a - λ • vector_b) → λ = 0) :=
by
  sorry

end correct_statements_l424_424253


namespace golden_raisins_scoops_l424_424307

theorem golden_raisins_scoops (x y : ℕ) 
    (h1 : 3.45 * x + 2.55 * y = 60) 
    (h2 : x + y = 20) : 
    y = 10 := 
sorry

end golden_raisins_scoops_l424_424307


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424281

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424281


namespace final_state_lamp_C_remain_l424_424479

-- Define the initial states of the lamps
def initial_lamps : Fin 3 → Bool
| 0 => true  -- Lamp A
| 1 => true  -- Lamp B
| 2 => true  -- Lamp C
| _ => false  -- Out of bounds, not matching any lamps

-- Define toggle function for single switch press
def toggle (lamp : Bool) : Bool := not lamp

-- Define red switch action (toggles lamps A and B)
def red_switch (lamps : Fin 3 → Bool) : Fin 3 → Bool
| 0 => toggle (lamps 0) -- Lamp A
| 1 => toggle (lamps 1) -- Lamp B
| 2 => lamps 2          -- Lamp C (unchanged)
| _ => false            -- Out of bounds

-- Define the red switch press action function
def press_red_switch (lamps : Fin 3 → Bool) (n : Nat) : Fin 3 → Bool :=
match n with
| 0 => lamps
| (n + 1) => press_red_switch (red_switch lamps) n

-- Supposing the whole process of remaining 19 presses with specific
-- final condition derivation
-- (additional presses simplified setup structuring on lamps).

theorem final_state_lamp_C_remain (total_red_presses : Nat) (total_presses : Nat) 
(h1 : total_red_presses = 8) (h2 : total_presses = 19):
  (press_red_switch initial_lamps total_red_presses) = λ i, if i == 2 then true else false
:= by
  sorry

end final_state_lamp_C_remain_l424_424479


namespace mark_money_given_l424_424392

noncomputable def total_cost : ℝ := 4.20 + 2.05
noncomputable def change_nickels : ℝ := 8 * 0.05
noncomputable def change_other_coins : ℝ := 0.25 + 0.10
noncomputable def total_change : ℝ := change_nickels + change_other_coins
noncomputable def money_given : ℝ := total_cost + total_change

theorem mark_money_given :
  money_given = 7.00 :=
begin
  sorry
end

end mark_money_given_l424_424392


namespace number_of_pairs_count_number_of_pairs_l424_424300

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424300


namespace total_cost_of_cable_l424_424594

theorem total_cost_of_cable : 
  let east_west_streets := 18 
  let east_west_length_per_street := 2
  let north_south_streets := 10  
  let north_south_length_per_street := 4  
  let cable_per_mile := 5 
  let cost_per_mile := 2000 
  let total_length := (east_west_streets * east_west_length_per_street) + (north_south_streets * north_south_length_per_street)
  let total_cable_needed := total_length * cable_per_mile 
  let total_cost := total_cable_needed * cost_per_mile 
  in total_cost = 760000 := by
  sorry

end total_cost_of_cable_l424_424594


namespace surface_area_of_sphere_with_diameter_4_l424_424015

theorem surface_area_of_sphere_with_diameter_4 :
    let diameter := 4
    let radius := diameter / 2
    let surface_area := 4 * Real.pi * radius^2
    surface_area = 16 * Real.pi :=
by
  -- Sorry is used in place of the actual proof.
  sorry

end surface_area_of_sphere_with_diameter_4_l424_424015


namespace remainder_of_2_pow_87_plus_3_mod_7_l424_424498

theorem remainder_of_2_pow_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end remainder_of_2_pow_87_plus_3_mod_7_l424_424498


namespace parallel_line_plane_intersection_l424_424533

variable (m n : Line) (α β : Plane)

-- Given conditions
variable (h1 : m.parallel α) (h2 : m.in β) (h3 : α.cap β = n)

-- Theorem statement
theorem parallel_line_plane_intersection : m.parallel n :=
sorry

end parallel_line_plane_intersection_l424_424533


namespace find_a_l424_424226

theorem find_a (a : ℝ) : (binom 5 2) + a * (binom 5 1) = 5 → a = -1 := 
by
  intro h
  sorry

end find_a_l424_424226


namespace iron_fence_enough_l424_424564

theorem iron_fence_enough (A_sqr : ℝ) (A_rect : ℝ) (ratio_lw : ℝ) (ratio_wl : ℝ) :
  A_sqr = 400 ∧ A_rect = 300 ∧ ratio_lw / ratio_wl = 5 / 3 → 
  4 * real.sqrt A_sqr ≥ 2 * (real.sqrt (A_rect * (ratio_lw / ratio_wl)) + real.sqrt (A_rect / (ratio_lw / ratio_wl))) := 
by
  sorry

end iron_fence_enough_l424_424564


namespace ratio_u_v_l424_424869

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l424_424869


namespace tangent_of_angle_l424_424448

theorem tangent_of_angle (α : ℝ) (P : ℝ × ℝ) 
  (h1 : P = (-2, 1)) 
  (h2 : ∃ θ : ℝ, α = θ ∧ (∀ θ, θ = α → θ ≠ 0 → tan θ = P.snd / P.fst)) :
  tan α = -1 / 2 :=
by sorry

end tangent_of_angle_l424_424448


namespace conveyor_belts_together_time_l424_424585

def time_to_move_coal (rate_old rate_new : ℕ → ℚ) (output_time_old output_time_new : ℕ) : ℚ :=
  rate_old output_time_old + rate_new output_time_new

theorem conveyor_belts_together_time :
  let old_time := 21,
      new_time := 15,
      combined_rate := time_to_move_coal (λ t, 1 / t) (λ t, 1 / t) old_time new_time in
  (1 / combined_rate = 8.75) := 
sorry

end conveyor_belts_together_time_l424_424585


namespace mystery_shelves_l424_424397

theorem mystery_shelves (M : ℕ) (H : M * 7 + 2 * 7 = 70) : M = 8 :=
begin
  sorry
end

end mystery_shelves_l424_424397


namespace travel_options_l424_424783

/-- 
  There are 10 train departures, 2 flights, and 12 long-distance bus services 
  from City A to City B. If Xiao Zhang chooses only one of these three 
  transportation options, then there are 24 ways for Xiao Zhang to 
  travel from City A to City B.
--/
theorem travel_options (train_deps flights buses : ℕ) 
  (h_train_deps : train_deps = 10) 
  (h_flights : flights = 2) 
  (h_buses : buses = 12) : 
  train_deps + flights + buses = 24 :=
by
  rw [h_train_deps, h_flights, h_buses]
  norm_num
  sorry

end travel_options_l424_424783


namespace min_correct_answers_l424_424727

/-- 
Given:
1. There are 25 questions in the preliminary round.
2. Scoring rules: 
   - 4 points for each correct answer,
   - -1 point for each incorrect or unanswered question.
3. A score of at least 60 points is required to advance to the next round.

Prove that the minimum number of correct answers needed to advance is 17.
-/
theorem min_correct_answers (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 25) (h3 : 4 * x - (25 - x) ≥ 60) : x ≥ 17 :=
sorry

end min_correct_answers_l424_424727


namespace cyclic_heat_engine_efficiency_l424_424117

-- Definitions and conditions
def isochoric_pressure_reduction (P0 V0 T0 R: ℝ) : Prop :=
  T0 = P0 * V0 / R

def isobaric_density_increase (P0 : ℝ) (V0 : ℝ) (T2 : ℝ) (R: ℝ) : Prop :=
  T2 = P0 * (3 * V0) / R

def max_possible_efficiency (T1 T2 : ℝ) : ℝ :=
  1 - T1 / T2

def cycle_efficiency (η_max : ℝ) : ℝ :=
  η_max / 8

-- Theorem statement
theorem cyclic_heat_engine_efficiency (P0 V0 T0 T1 T2 R: ℝ)
  (h_isochoric : isochoric_pressure_reduction P0 V0 T0 R)
  (h_isobaric : isobaric_density_increase P0 V0 T2 R)
  (h_temp_eq : T1 = T0)
  : cycle_efficiency (max_possible_efficiency T1 T2) = 1 / 12 :=
  sorry

end cyclic_heat_engine_efficiency_l424_424117


namespace num_pairs_of_regular_polygons_l424_424145

def num_pairs : Nat := 
  let pairs := [(7, 42), (6, 18), (5, 10), (4, 6)]
  pairs.length

theorem num_pairs_of_regular_polygons : num_pairs = 4 := 
  sorry

end num_pairs_of_regular_polygons_l424_424145


namespace hyperbola_equation_line_equation_l424_424203

-- Part Ⅰ: Determine the equation of the hyperbola C given the conditions
theorem hyperbola_equation (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_tangent_circle : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 + y^2 = 3)) 
  (h_focus_line : ∀ x : ℝ, (1 / 3 * x^2 - 1) = 0 ∧ x > 0) :
  (a = sqrt 3 ∧ b = 1) ∧ (∀ x y : ℝ, (x^2 / 3 - y^2 = 1) = (x^2 / a^2 - y^2 / b^2 = 1)) :=
sorry

-- Part Ⅱ: Find the equation of the line l given area condition
theorem line_equation (P : ℝ × ℝ) (hP : P.1^2 + P.2^2 = 3 ∧ P.1 > 0 ∧ P.2 > 0)
  (h_area : ∃ l : ℝ → ℝ, (∀ x : ℝ, x^2 + (l x)^2 = 3 ∧ 
    (∀ A B : ℝ × ℝ, (A ≠ B ∧ 
    A.1^2 / 3 - A.2^2 = 1 ∧ B.1^2 / 3 - B.2^2 = 1) ∧ 
    (area_triangle A B (0,0) = 3*sqrt 2)) → 
    ∀ x, l x = -x + sqrt 6)) :=
sorry

end hyperbola_equation_line_equation_l424_424203


namespace distance_between_x_intercepts_l424_424089

theorem distance_between_x_intercepts :
  ∀ (l1 l2 : ℝ → ℝ)
    (hl1 : ∃ m1 b1, (∀ x, l1 x = m1 * x + b1) ∧ m1 = 4 ∧ ∃ y, l1 y = 0)
    (hl2 : ∃ m2 b2, (∀ x, l2 x = m2 * x + b2) ∧ m2 = -3 ∧ ∃ y, l2 y = 0)
    (hintersect : ∃ x y, l1 x = y ∧ l2 x = y ∧ x = 8 ∧ y = 20),
  ∃ d : ℝ,
    d = abs(3 - 44 / 3) :=
begin
  sorry
end

end distance_between_x_intercepts_l424_424089


namespace field_width_calculation_l424_424450

theorem field_width_calculation (w : ℝ) (h_length : length = 24) (h_length_width_relation : length = 2 * w - 3) : w = 13.5 :=
by 
  sorry

end field_width_calculation_l424_424450


namespace period_of_y_l424_424497

def y (x : ℝ) := 3 * Real.sin (2 * x) + 4 * Real.cos (2 * x)

theorem period_of_y : ∃ T > 0, ∀ x, y (x + T) = y x := by
  use π
  intro x
  sorry

end period_of_y_l424_424497


namespace minimum_distance_from_P_to_A_and_y_axis_l424_424679

open Real

def distance (p1 p2 : Point) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def point_on_parabola (y : ℝ) : Point := 
  ⟨(1/4) * y^2, y⟩

def distance_to_y_axis (p : Point) : ℝ := 
  abs p.1

def distance_to_point_a (p : Point) : ℝ := 
  distance p ⟨0, 2⟩

noncomputable def total_distance (y : ℝ) : ℝ :=
  let p := point_on_parabola y in
  distance_to_point_a p + distance_to_y_axis p

theorem minimum_distance_from_P_to_A_and_y_axis : 
  (∃ (y : ℝ), total_distance y = (sqrt 5 - 1)) :=
sorry

end minimum_distance_from_P_to_A_and_y_axis_l424_424679


namespace additional_men_joined_l424_424113

def men_to_finish_work (initial_men : ℕ) (initial_days : ℕ) (reduced_days : ℕ) (total_men : ℕ) : ℕ :=
  total_men - initial_men

theorem additional_men_joined (initial_men : ℕ) (initial_days : ℕ) (reduced_days : ℕ) (total_man_days : ℕ) :
  (initial_days - reduced_days) * (initial_men + 4) = total_man_days → men_to_finish_work initial_men initial_days reduced_days (initial_men + 4) = 4 :=
by
  -- Assume 4 men joined additionally
  cases (show initial_men = 10 ∧ initial_days = 12 ∧ reduced_days = 3 ∧ total_man_days = 120) with h₁ h₂
  sorry

end additional_men_joined_l424_424113


namespace find_value_l424_424197

variable (a : ℝ) (h : a + 1/a = 7)

theorem find_value :
  a^2 + 1/a^2 = 47 :=
sorry

end find_value_l424_424197


namespace number_of_pairs_count_number_of_pairs_l424_424303

theorem number_of_pairs (a b : ℕ) (h : a ≥ b) (h₁ : a > 0) (h₂ : b > 0) : 
  (1 / a + 1 / b = 1 / 6) → (a, b) = (42, 7) ∨ (a, b) = (24, 8) ∨ (a, b) = (18, 9) ∨ (a, b) = (15, 10) ∨ (a, b) = (12, 12) :=
sorry

theorem count_number_of_pairs : 
  {p : ℕ × ℕ // p.fst ≥ p.snd ∧ 1 / p.fst + 1 / p.snd = 1 / 6}.to_finset.card = 5 :=
sorry

end number_of_pairs_count_number_of_pairs_l424_424303


namespace solution_set_l424_424425

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

theorem solution_set :
  ∀ x y z : ℝ, system_of_equations x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = -1 ∧ z = -3) ∨ 
  (x = -2 ∧ y = 1 ∧ z = -3) ∨ (x = -2 ∧ y = -1 ∧ z = 3) :=
by
  sorry

end solution_set_l424_424425


namespace complex_addition_l424_424068

-- Defining the imaginary unit i
def i : ℂ := complex.I

theorem complex_addition : (1 + i)^2 + 2 * i = 4 * i := 
by sorry

end complex_addition_l424_424068


namespace count_increasing_numbers_l424_424174

-- Define the set of digits we are concerned with
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a natural number type representing numbers with increasing digits
def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits i < n.digits j

-- Define the set of natural numbers with increasing digits and at least two digits
def increasing_numbers : set ℕ :=
  {n | increasing_digits n ∧ 10 ≤ n ∧ n ≤ 987654321}

-- Define the theorem to be proved
theorem count_increasing_numbers : set.card increasing_numbers = 502 :=
by sorry

end count_increasing_numbers_l424_424174


namespace shaded_to_unshaded_area_ratio_l424_424898

-- Define a structure for a large square with its side length.
structure Square :=
  (side_length : ℝ)

-- Define the condition that vertices of smaller squares are at the midpoints.
def midpoint_vertex_property (sq : Square) : Prop :=
  ∀ (sub_sq : Square), sub_sq.side_length = sq.side_length / 2

-- Define areas of regions within one quadrant.
def shaded_area (sq : Square) : ℝ :=
  5 * (sq.side_length / 2) * (sq.side_length / 2) / 2

def unshaded_area (sq : Square) : ℝ :=
  3 * (sq.side_length / 2) * (sq.side_length / 2) / 2

-- Define the ratio of the shaded area to the unshaded area.
def area_ratio (sq : Square) : ℝ :=
  shaded_area sq / unshaded_area sq

-- The theorem stating that the area ratio is 5:3.
theorem shaded_to_unshaded_area_ratio (sq : Square) (h : midpoint_vertex_property sq) :
  area_ratio sq = 5 / 3 := 
  sorry

end shaded_to_unshaded_area_ratio_l424_424898


namespace polynomial_inequality_l424_424846

-- Define the polynomial P and its condition
def P (a b c : ℝ) (x : ℝ) : ℝ := 12 * x^3 + a * x^2 + b * x + c
-- Define the polynomial Q and its condition
def Q (a b c : ℝ) (x : ℝ) : ℝ := (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c

-- Assumptions
axiom P_has_distinct_roots (a b c : ℝ) : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0
axiom Q_has_no_real_roots (a b c : ℝ) : ¬ ∃ x : ℝ, Q a b c x = 0

-- The goal to prove
theorem polynomial_inequality (a b c : ℝ) (h1 : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0)
  (h2 : ¬ ∃ x : ℝ, Q a b c x = 0) : 2001^3 + a * 2001^2 + b * 2001 + c > 1 / 64 :=
by {
  -- sorry is added to skip the proof part
  sorry
}

end polynomial_inequality_l424_424846


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l424_424902

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.45
  let a := 9 -- GCD of 45 and 99
  let numerator := 5
  let denominator := 11
  numerator + denominator = 16 :=
by { 
  sorry 
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l424_424902


namespace sqrt_mul_neg_eq_l424_424128

theorem sqrt_mul_neg_eq : - (Real.sqrt 2) * (Real.sqrt 7) = - (Real.sqrt 14) := sorry

end sqrt_mul_neg_eq_l424_424128


namespace second_train_speed_l424_424029

theorem second_train_speed:
  ∃ (v : ℕ), 
    (let t := (900 - 100) / (50 + v) in 
      50 * t = v * t + 100 ∧ 
      50 * t + v * t = 900) → 
    v = 40 := 
begin
  use 40,
  intro h,
  simp at h,
  sorry
end

end second_train_speed_l424_424029


namespace solution_set_interval_l424_424670

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = f(x)
def decreasing_F (f : ℝ → ℝ) := ∀ x < 0, (x ^ 2) * f(x)

theorem solution_set_interval (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_diff : differentiable ℝ f)
  (h_condition : ∀ x : ℝ, x < 0 → 2 * f(x) + x * (deriv f x) > x^2) :
  { x : ℝ | (x + 2014) ^ 2 * f(x + 2014) - 4 * f(-2) < 0 } = set.Ioo (-2016) (-2012) :=
sorry

end solution_set_interval_l424_424670


namespace number_of_bird_cages_l424_424559

-- Definitions for the problem conditions
def birds_per_cage : ℕ := 2 + 7
def total_birds : ℕ := 72

-- The theorem to prove the number of bird cages is 8
theorem number_of_bird_cages : total_birds / birds_per_cage = 8 := by
  sorry

end number_of_bird_cages_l424_424559


namespace num_increasing_digits_l424_424180

theorem num_increasing_digits :
  let C := λ (n k : ℕ), Nat.choose n k in
  ∑ k in Finset.range 8, C 9 (k + 2) = 502 :=
by
  sorry

end num_increasing_digits_l424_424180


namespace ratio_of_areas_l424_424894

-- Define the conditions of the problem
variable (SQ : Type) [Square SQ]    -- Assume SQ is the type of all squares, and it satisfies some square properties.

-- Define the vertices and midpoint condition
axiom vertices_at_midpoints (s : SQ) : vertices at midpoints ∧ symmetric across quadrants

-- Define the area function and assume shaded and white area as given in a quadrant
variable (area : SQ → ℕ)
def shaded_area (sq : SQ) := 5 * area sq -- Shaded triangles per quadrant
def white_area (sq : SQ) := 3 * area sq  -- White triangles per quadrant

-- Resulting areas for the whole figure
def total_shaded_area (sq : SQ) := 4 * shaded_area sq
def total_white_area (sq : SQ) := 4 * white_area sq

-- Define the ratio function to compare shaded and white areas
def ratio (a b : ℕ) := a / b

theorem ratio_of_areas (s : SQ) (area_nonzero : area s ≠ 0):
  ratio (total_shaded_area s) (total_white_area s) = 5 / 3 := by
  sorry

end ratio_of_areas_l424_424894


namespace minimum_n_xn_l424_424444

noncomputable def f (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 0 then 2 * x + 1 else 
if 0 ≤ x ∧ x ≤ 2 then -2 * x + 1 else 
if 2 ≤ x ∧ x ≤ 4 then 2 * x - 3 else 
-2 * (x - 4) + 1

def even_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x, f(x) = f(x + p) ∧ f(x) = f(-x)

def condition1 : Prop := even_periodic f 4
def condition2 (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0 → f(x) = 2*x + 1
def condition3 (xs : List ℝ) : Prop := xs.Sorted (<) ∧ (∀ x, 0 ≤ x → x ∈ xs )

def cond_abs_sum (xs : List ℝ) : Prop :=
(xs.pairwise (λ a b, ∃ n m, |f(a) - f(b)| = 4 * (n - m))) ∧ xs.sum = 2016

theorem minimum_n_xn (n : ℕ) (xn : ℝ) (xs : List ℝ)
  (h1 : condition1)
  (h2 : ∀ x, condition2 x)
  (h3 : condition3 xs)
  (h4 : cond_abs_sum xs):
  n + xn = 1513 := 
sorry

end minimum_n_xn_l424_424444


namespace B_and_C_mutually_exclusive_l424_424647

-- Defining events in terms of products being defective or not
def all_not_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, ¬x

def all_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, x

def not_all_defective (products : List Bool) : Prop := 
  ∃ x ∈ products, ¬x

-- Given a batch of three products, define events A, B, and C
def A (products : List Bool) : Prop := all_not_defective products
def B (products : List Bool) : Prop := all_defective products
def C (products : List Bool) : Prop := not_all_defective products

-- The theorem to prove that B and C are mutually exclusive
theorem B_and_C_mutually_exclusive (products : List Bool) (h : products.length = 3) : 
  ¬ (B products ∧ C products) :=
by
  sorry

end B_and_C_mutually_exclusive_l424_424647


namespace incorrect_differentiation_C_l424_424050

theorem incorrect_differentiation_C (x : ℝ) (hA : deriv (λ x, 3^x) x = 3^x * real.log 3)
  (hB : deriv (λ x, real.log x / x) x = (1 - real.log x) / (x ^ 2))
  (hD : deriv (λ x, real.sin x * real.cos x) x = real.cos (2 * x)) :
  deriv (λ x, x + 1 / x) x ≠ 1 + 1 / (x ^ 2) :=
sorry

end incorrect_differentiation_C_l424_424050


namespace min_value_of_exp_sum_l424_424214

theorem min_value_of_exp_sum (a b : ℝ) (h : a + b = 2) : 3^a + 3^b ≥ 6 :=
sorry

end min_value_of_exp_sum_l424_424214


namespace proper_subsets_count_of_M_l424_424458

noncomputable def M : Set ℕ := {x | -1 ≤ Real.logb 10 (x) ∧ Real.logb 10 (x) < -1/2 ∧ x ∈ ℕ}

/-- The number of proper subsets of the set M is 2^90 - 1. -/
theorem proper_subsets_count_of_M : (2^90 - 1) = (2^90 - 1) :=
sorry

end proper_subsets_count_of_M_l424_424458


namespace total_candles_l424_424643

theorem total_candles (num_big_boxes : ℕ) (small_boxes_per_big_box : ℕ) (candles_per_small_box : ℕ) :
  num_big_boxes = 50 ∧ small_boxes_per_big_box = 4 ∧ candles_per_small_box = 40 → 
  (num_big_boxes * (small_boxes_per_big_box * candles_per_small_box) = 8000) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  norm_num
  sorry

end total_candles_l424_424643


namespace cube_volume_eq_pyramid_volume_l424_424550

theorem cube_volume_eq_pyramid_volume :
  let V_cube := 6^3 in
  let V_pyramid := (1 / 3) * 10 * 8 * h in
  V_cube = V_pyramid → h = 8.1 :=
by
  sorry

end cube_volume_eq_pyramid_volume_l424_424550


namespace excircle_collinear_l424_424000

noncomputable def Excircle (A B C T: Type*) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder T] : Prop :=
  let D D' : Type* := sorry -- Definition of point D (tangent to BC)
  let E E' : Type* := sorry -- Definition of point E (tangent to AB extension)
  let F F' : Type* := sorry -- Definition of point F (tangent to AC extension)
  let BF : Line := { E | ∃ F, sorry } -- Intersection of line BF
  let CE : Line := { E | ∃ F, sorry } -- Intersection of line CE
  let T : Type* := { t | exist (BF ∩ CE) } -- Intersection point of lines BF and CE
  Collinear A D T

/-- Prove that the points A, D, and T are collinear given the excircle conditions. -/
theorem excircle_collinear {A B C D E F T: Type*}
  (h1 : Excircle A B C D E F T) :
  Collinear A D T :=
sorry

end excircle_collinear_l424_424000


namespace dogwood_trees_after_5_years_l424_424460

theorem dogwood_trees_after_5_years :
  let current_trees := 39
  let trees_planted_today := 41
  let growth_rate_today := 2 -- trees per year
  let trees_planted_tomorrow := 20
  let growth_rate_tomorrow := 4 -- trees per year
  let years := 5
  let total_planted_trees := trees_planted_today + trees_planted_tomorrow
  let total_initial_trees := current_trees + total_planted_trees
  let total_growth_today := growth_rate_today * years
  let total_growth_tomorrow := growth_rate_tomorrow * years
  let total_growth := total_growth_today + total_growth_tomorrow
  let final_tree_count := total_initial_trees + total_growth
  final_tree_count = 130 := by
  sorry

end dogwood_trees_after_5_years_l424_424460


namespace average_salary_rest_l424_424333

variable (totalWorkers : ℕ)
variable (averageSalaryAll : ℕ)
variable (numTechnicians : ℕ)
variable (averageSalaryTechnicians : ℕ)

theorem average_salary_rest (h1 : totalWorkers = 28) 
                           (h2 : averageSalaryAll = 8000)
                           (h3 : numTechnicians = 7)
                           (h4 : averageSalaryTechnicians = 14000) : 
                           (averageSalaryAll * totalWorkers - averageSalaryTechnicians * numTechnicians) / (totalWorkers - numTechnicians) = 6000 :=
begin
  -- The proof will be provided here
  sorry
end

end average_salary_rest_l424_424333


namespace V1_ne_V2_min_V1_over_V2_l424_424202

noncomputable theory

-- Define the necessary variables
variables 
  (R r h : ℝ)  -- radius of sphere, radius of base of cone, height of cone respectively
  (V1 V2 : ℝ) -- volumes of cone and cylinder

-- Condition: height of cone is greater than twice the radius of the sphere
axiom height_condition : h > 2 * R

-- Equations for volumes
axiom V1_eq : V1 = (1/3) * π * r^2 * h
axiom V2_eq : V2 = 2 * π * R^3

-- Expressions for r and h in terms of apex angle θ
axiom r_eq : r = h * tan θ
axiom h_eq : h = (1 + sin θ) / sin θ * R

-- Theorems to prove
theorem V1_ne_V2 : V1 ≠ V2 := sorry

theorem min_V1_over_V2 : (fraction (V1 / V2) = 4 / 3) ∧ 
                                     (exists θ' : ℝ, θ' = arcsin (1 / 3)) := sorry

end V1_ne_V2_min_V1_over_V2_l424_424202


namespace count_pairs_satisfying_condition_l424_424298

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424298


namespace hyperbola_eccentricity_sqrt2_l424_424822

theorem hyperbola_eccentricity_sqrt2
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = Real.sqrt (a^2 + b^2))
  (h : (c + a)^2 + (b^2 / a)^2 = 2 * c * (c + a)) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_sqrt2_l424_424822


namespace correct_step_l424_424510

-- Define the variables and the system of equations
variables (a b : ℝ)

-- Given conditions
def eq1 : Prop := 2 * a + b = 7
def eq2 : Prop := a - b = 2

-- The proof statement 
theorem correct_step : eq1 ∧ eq2 → 3 * a = 9 :=
by
  intros h,
  have h1 := h.1,
  have h2 := h.2,

  -- Adding equations eq1 and eq2 results
  have add_eq := eq1.add eq2,

  -- Substitute into the resulting statement
  sorry

end correct_step_l424_424510


namespace count_increasing_numbers_l424_424173

-- Define the set of digits we are concerned with
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a natural number type representing numbers with increasing digits
def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits i < n.digits j

-- Define the set of natural numbers with increasing digits and at least two digits
def increasing_numbers : set ℕ :=
  {n | increasing_digits n ∧ 10 ≤ n ∧ n ≤ 987654321}

-- Define the theorem to be proved
theorem count_increasing_numbers : set.card increasing_numbers = 502 :=
by sorry

end count_increasing_numbers_l424_424173


namespace magnitude_range_l424_424252

variables (α : ℝ)

def vector_a := (Real.cos α, 0)
def vector_b := (1, Real.sin α)

def vector_sum := (vector_a α).1 + (vector_b α).1, (vector_a α).2 + (vector_b α).2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_range : 
  ∀ α, 0 ≤ magnitude (vector_sum α) ∧ magnitude (vector_sum α) ≤ 2 := 
begin
  sorry
end

end magnitude_range_l424_424252


namespace english_but_not_russian_l424_424123

noncomputable def teachers_english_not_russian (E R : Set α) [Fintype E] [Fintype R] : ℕ :=
  Fintype.card E - Fintype.card (E ∩ R)

theorem english_but_not_russian (E R : Set α) [Fintype E] [Fintype R]
  (h_union : Fintype.card (E ∪ R) = 110)
  (h_english : Fintype.card E = 75)
  (h_russian : Fintype.card R = 55) :
  teachers_english_not_russian E R = 55 :=
by
  have h_intersect : Fintype.card (E ∩ R) = 20 :=
    by
      rw [←Fintype.card_union_add_card_inter E R, h_english, h_russian, h_union]
      exact Nat.sub_left_inj _ _
  unfold teachers_english_not_russian
  rw [h_intersect, h_english]
  exact Nat.sub _ _
  

end english_but_not_russian_l424_424123


namespace acute_angle_half_l424_424760

variables {A X Y Z B P Q R S O : Point}

/-- 
Let AXYZB be a convex pentagon inscribed in a semicircle with diameter AB,
and let P, Q, R, and S be the feet of the perpendiculars from Y onto lines 
AX, BX, AZ, and BZ respectively.
Prove that the acute angle formed by lines PQ and RS is half the size of ∠XOZ,
where O is the midpoint of segment AB.
-/
theorem acute_angle_half (convex_pentagon_inscribed : ConvexPentagonInscribed A X Y Z B)
    (P_is_perpendicular_foot : PerpendicularFoot Y P A X)
    (Q_is_perpendicular_foot : PerpendicularFoot Y Q B X)
    (R_is_perpendicular_foot : PerpendicularFoot Y R A Z)
    (S_is_perpendicular_foot : PerpendicularFoot Y S B Z)
    (O_is_midpoint : Midpoint O A B)
    : angle (line_through P Q) (line_through R S) = angle_div_two (angle (vertex X O Z)) := 
    sorry

end acute_angle_half_l424_424760


namespace probability_both_selected_l424_424062

def P_X : ℚ := 1 / 3
def P_Y : ℚ := 2 / 7

theorem probability_both_selected : P_X * P_Y = 2 / 21 :=
by
  sorry

end probability_both_selected_l424_424062


namespace tangent_line_parabola_l424_424826

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l424_424826


namespace perpendicular_bisector_of_MN_line_l_parallel_to_MN_l424_424703

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def perpendicular_bisector (M N : ℝ × ℝ) : ℝ → ℝ :=
  let mid := midpoint M N
  let k := slope M N
  let k_perp := -1 / k
  λ x => k_perp * (x - mid.1) + mid.2

noncomputable def equation_of_perpendicular_bisector (M N : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, y = perpendicular_bisector M N x ↔ x + 3 * y - 6 = 0

noncomputable def parallel_line (P : ℝ × ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x => k * (x - P.1) + P.2

noncomputable def equation_of_parallel_line (P : ℝ × ℝ) (M N : ℝ × ℝ) : Prop :=
  let k := slope M N
  ∀ x y : ℝ, y = parallel_line P k x ↔ 3 * x - y - 9 = 0

theorem perpendicular_bisector_of_MN :
  equation_of_perpendicular_bisector (2, -2) (4, 4) :=
sorry

theorem line_l_parallel_to_MN :
  equation_of_parallel_line (2, -3) (2, -2) (4, 4) :=
sorry

end perpendicular_bisector_of_MN_line_l_parallel_to_MN_l424_424703


namespace percentage_of_pure_acid_is_20_l424_424474

-- Definitions of the conditions
def volume_of_pure_acid : ℝ := 1.6
def total_volume_of_solution : ℝ := 8
def percentage_of_pure_acid : ℝ := (volume_of_pure_acid / total_volume_of_solution) * 100

-- The theorem we want to prove
theorem percentage_of_pure_acid_is_20 :
  percentage_of_pure_acid = 20 :=
sorry

end percentage_of_pure_acid_is_20_l424_424474


namespace fib_identity_l424_424988

theorem fib_identity : 
  (∀ n : ℕ, fib (n + 1) * fib (n - 1) - fib n ^ 2 = (-1) ^ n) →
  fib 102 * fib 100 - fib 101 ^ 2 = -1 :=
by
  sorry

end fib_identity_l424_424988


namespace max_investment_at_7_percent_l424_424971

variables (x y : ℝ)

theorem max_investment_at_7_percent 
  (h1 : x + y = 25000)
  (h2 : 0.07 * x + 0.12 * y ≥ 2450) : 
  x ≤ 11000 :=
sorry

end max_investment_at_7_percent_l424_424971


namespace ticket_sales_total_l424_424913

variable (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ)

def total_money_collected (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - child_tickets
  let total_child := child_tickets * price_child
  let total_adult := adult_tickets * price_adult
  total_child + total_adult

theorem ticket_sales_total :
  price_adult = 6 →
  price_child = 4 →
  total_tickets = 21 →
  child_tickets = 11 →
  total_money_collected price_adult price_child total_tickets child_tickets = 104 :=
by
  intros
  unfold total_money_collected
  simp
  sorry

end ticket_sales_total_l424_424913


namespace exists_natural_numbers_l424_424615

theorem exists_natural_numbers (λ : ℕ) (hλ : λ > 10^5) :
  let a := 2012 * (λ^2 - 1), b := 2012 * λ, c := 2012 * λ in
  a > 10^10 ∧ b > 10^10 ∧ c > 10^10
  ∧ (a * b * c) % (a + 2012) = 0
  ∧ (a * b * c) % (b + 2012) = 0
  ∧ (a * b * c) % (c + 2012) = 0 := 
by
  -- Proof goes here
  sorry

end exists_natural_numbers_l424_424615


namespace cycle_efficiency_is_correct_l424_424119

def initial_pressure : ℝ := 3 * P₀
def final_pressure : ℝ := P₀
def initial_density : ℝ := ρ₀
def final_density : ℝ := 3 * ρ₀
def cycle_center : (ℝ × ℝ) := (1, 1)
def T₀ : ℝ := P₀ * V₀ / R
def T₁ : ℝ := P₀ * V₀ / R
def T₂ : ℝ := 3 * P₀ * V₀ / R
def Carnot_efficiency : ℝ := 1 - (T₁ / T₂)
def given_efficiency : ℝ := 1 / 8 * Carnot_efficiency

theorem cycle_efficiency_is_correct (P₀ V₀ ρ₀ R : ℝ) (H₀ : P₀ > 0) (H₁ : V₀ > 0) (H₂ : ρ₀ > 0) (H₃ : R > 0) :
  given_efficiency = 1 / 9 := sorry

end cycle_efficiency_is_correct_l424_424119


namespace find_total_estate_l424_424778

theorem find_total_estate (E : ℝ) (x : ℝ) (y : ℝ) :
  -- Children share in the ratio 5:3:2
  5 * x + 3 * x + 2 * x = (2 / 3) * E →
  -- Wife's share is three times the youngest child's share
  6 * x = 3 * 2 * x →
  -- Charity received $600
  y = 600 →
  -- Total estate is then
  E = 90_000 :=
by
  intros h1 h2 h3
  -- proof to be filled in later
  sorry

end find_total_estate_l424_424778


namespace strawberry_harvest_l424_424534

theorem strawberry_harvest
  (length : ℕ) (width : ℕ)
  (plants_per_sqft : ℕ) (yield_per_plant : ℕ)
  (garden_area : ℕ := length * width) 
  (total_plants : ℕ := plants_per_sqft * garden_area) 
  (expected_strawberries : ℕ := yield_per_plant * total_plants) :
  length = 10 ∧ width = 12 ∧ plants_per_sqft = 5 ∧ yield_per_plant = 8 → 
  expected_strawberries = 4800 := by
  sorry

end strawberry_harvest_l424_424534


namespace expression_equals_negative_seven_l424_424374

variable (a b c : ℝ)

-- Given conditions
def nonzero_real_numbers := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
def sum_is_zero := a + b + c = 0
def sum_products_not_zero := ab + ac + bc ≠ 0

-- Problem statement
theorem expression_equals_negative_seven (h1 : nonzero_real_numbers a b c) (h2 : sum_is_zero a b c) (h3 : sum_products_not_zero a b c) :
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
sorry

end expression_equals_negative_seven_l424_424374


namespace correct_system_of_equations_l424_424931

theorem correct_system_of_equations (x y : ℝ) :
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y) :=
sorry

end correct_system_of_equations_l424_424931


namespace sugar_quantity_l424_424094

theorem sugar_quantity (T : ℝ) (h : 0.08 * (T - 600) + 0.18 * 600 = 0.14 * T) :
  T = 1000 :=
begin
  sorry
end

end sugar_quantity_l424_424094


namespace quad_function_analytic_expression_range_of_k_l424_424657

noncomputable def f (x : ℝ) : ℝ := (x + 2)^2 - 3

theorem quad_function_analytic_expression :
  (∀ x : ℝ, f(x) = (x + 2)^2 - 3) ∧ (f 0 = 1) ∧ (∃ a b, a = -2 + real.sqrt 3 ∧ (f a = 0)) :=
by sorry

theorem range_of_k (k : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, f ((1/2) ^ x) > k) → k < 13 / 4 :=
by sorry

end quad_function_analytic_expression_range_of_k_l424_424657


namespace brenda_initial_peaches_l424_424125

variable (P : ℕ)

def brenda_conditions (P : ℕ) : Prop :=
  let fresh_peaches := P - 15
  (P > 15) ∧ (fresh_peaches * 60 = 100 * 150)

theorem brenda_initial_peaches : ∃ (P : ℕ), brenda_conditions P ∧ P = 250 :=
by
  sorry

end brenda_initial_peaches_l424_424125


namespace chord_intersection_probability_l424_424716

theorem chord_intersection_probability :
  ∀ (A B C D E F : ℕ), 1 ≤ A ∧ A ≤ 2004 → 
                        1 ≤ B ∧ B ≤ 2004 → 
                        1 ≤ C ∧ C ≤ 2004 → 
                        1 ≤ D ∧ D ≤ 2004 → 
                        1 ≤ E ∧ E ≤ 2004 → 
                        1 ≤ F ∧ F ≤ 2004 → 
                        A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ 
                        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ 
                        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ 
                        D ≠ E ∧ D ≠ F ∧ 
                        E ≠ F → 
  (probability_of_chord_intersection A B C D E F) = 1 / 3 := 
sorry

end chord_intersection_probability_l424_424716


namespace parity_f1_parity_f2_parity_f3_parity_piecewise_f_l424_424309

-- Definitions of parity for functions.
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Function definitions
def f1 (x : ℝ) : ℝ := (x ^ 2).cbrt + 1
def f2 (x : ℝ) : ℝ := x ^ 3 + x ^ (-3)
def f3 (f : ℝ → ℝ) (x : ℝ) : ℝ := 1/2 * (f x - f (-x))

def piecewise_f (x : ℝ) : ℝ :=
  if x > 0 then 1 - x else if x = 0 then 0 else 1 + x

-- Proof statements
theorem parity_f1 : is_even f1 := sorry
theorem parity_f2 : is_odd f2 := sorry
theorem parity_f3 (f : ℝ → ℝ) (a : ℝ) (h : a > 0) : is_odd (f3 f) := sorry
theorem parity_piecewise_f : is_even piecewise_f := sorry

end parity_f1_parity_f2_parity_f3_parity_piecewise_f_l424_424309


namespace correct_equation_l424_424389

-- Condition 1: Machine B transports 60 kg more per hour than Machine A
def machine_B_transports_more (x : ℝ) : Prop := 
  x + 60

-- Condition 2: Time to transport 500 kg by Machine A equals time 
-- to transport 800 kg by Machine B.
def transportation_time_eq (x : ℝ) : Prop :=
  500 / x = 800 / (x + 60)

-- Theorem statement: Prove the correct equation for given conditions
theorem correct_equation (x : ℝ) (h1 : machine_B_transports_more x) (h2 : transportation_time_eq x) : 
  500 / x = 800 / (x + 60) :=
  by
    sorry

end correct_equation_l424_424389


namespace total_number_of_girls_in_school_l424_424084

theorem total_number_of_girls_in_school 
  (students_sampled : ℕ) 
  (students_total : ℕ) 
  (sample_girls : ℕ) 
  (sample_boys : ℕ)
  (h_sample_size : students_sampled = 200)
  (h_total_students : students_total = 2000)
  (h_diff_girls_boys : sample_boys = sample_girls + 6)
  (h_stratified_sampling : students_sampled / students_total = 200 / 2000) :
  sample_girls * (students_total / students_sampled) = 970 :=
by
  sorry

end total_number_of_girls_in_school_l424_424084


namespace quadrilateral_MNPQ_is_cyclic_l424_424809

-- Definitions of points and diagonals
variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables (O M N P Q : Point)
variables (OA OB : Point → Point)

-- Given the conditions
def diagonals_are_perpendicular (diag1 diag2 : Point → Point) : Prop :=
∀ p : Point, diag1 p = O + p ∧ diag2 p = O - p ∧ ∀ a b : Point, 0 = (a - O) • (b - O)

def perpendicular_to_side (A B : Point) (C : Point → Point) : Prop :=
∀ p : Point, (p - A) • (C p - B) = 0

def extended_perpendiculars (M N P Q : Point) : Prop :=
∃ OM ON : Point → Point, 
  perpendicular_to_side O OM M ∧ perpendicular_to_side O ON N ∧
  ∃ PQ : Point → Point, PQ = λ p, p + (P - O) ∧ PQ = λ p, p + (Q - O)
  
-- The theorem to prove
theorem quadrilateral_MNPQ_is_cyclic 
  (h1 : diagonals_are_perpendicular OA OB)
  (h2 : extended_perpendiculars M N P Q) :
  cyclic M N P Q :=
sorry

end quadrilateral_MNPQ_is_cyclic_l424_424809


namespace sum_of_731_and_one_fifth_l424_424900

theorem sum_of_731_and_one_fifth :
  (7.31 + (1 / 5) = 7.51) :=
sorry

end sum_of_731_and_one_fifth_l424_424900


namespace part_a_part_b_l424_424749

-- Define Jonas's sequence
def jonas_sequence : List ℕ := List.range (2021 * 13) |>.filter (λ x, x > 0 ∧ x % 13 = 0) 

-- Define a function to get the nth digit of the sequence
def digit_in_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let digits := seq.bind (λ num, num.digits)
  digits.get_or_else (n - 1) 0

-- Part (a): Prove that the 2019th digit in Jonas's sequence is 8
theorem part_a : digit_in_sequence jonas_sequence 2019 = 8 := by
  sorry

-- Part (b): Prove that the number 2019 does not appear in Jonas's sequence
theorem part_b : ((2019 ∈ jonas_sequence) = false) := by
  sorry

end part_a_part_b_l424_424749


namespace problem1_problem2_l424_424599

theorem problem1 : 
  -(3^3) * ((-1 : ℚ)/ 3)^2 - 24 * (3/4 - 1/6 + 3/8) = -26 := 
by 
  sorry

theorem problem2 : 
  -(1^100 : ℚ) - (3/4) / (((-2)^2) * ((-1 / 4) ^ 2) - 1 / 2) = 2 := 
by 
  sorry

end problem1_problem2_l424_424599


namespace tangent_line_at_1_2_l424_424811

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

def tangent_eq (x y : ℝ) : Prop := y = 2 * x

theorem tangent_line_at_1_2 : tangent_eq 1 2 :=
by
  have f_1 := 1
  have f'_1 := 2
  sorry

end tangent_line_at_1_2_l424_424811


namespace quadratic_polynomial_equality_l424_424212

theorem quadratic_polynomial_equality (f g : ℝ → ℝ) (h_f_quad : ∃ b c, ∀ x, f(x) = x^2 + b * x + c)
  (h_g_quad : ∃ d e, ∀ x, g(x) = x^2 + d * x + e) (h_distinct : f ≠ g) (h_leading_coeff : ∀ x, (f x = x ^ 2 + b * x + c) ∧ (g x = x ^ 2 + d * x + e))
  (h_sum_eq : f(1) + f(10) + f(100) = g(1) + g(10) + g(100)) :
  f(37) = g(37) := sorry

end quadratic_polynomial_equality_l424_424212


namespace emily_lives_l424_424150

theorem emily_lives :
  ∃ (lives_gained : ℕ), 
    let initial_lives := 42
    let lives_lost := 25
    let lives_after_loss := initial_lives - lives_lost
    let final_lives := 41
    lives_after_loss + lives_gained = final_lives :=
sorry

end emily_lives_l424_424150


namespace unit_digit_product_7858_1086_4582_9783_l424_424046

-- Define the unit digits of the given numbers
def unit_digit_7858 : ℕ := 8
def unit_digit_1086 : ℕ := 6
def unit_digit_4582 : ℕ := 2
def unit_digit_9783 : ℕ := 3

-- Define a function to calculate the unit digit of a product of two numbers based on their unit digits
def unit_digit_product (a b : ℕ) : ℕ :=
  (a * b) % 10

-- The theorem that states the unit digit of the product of the numbers is 4
theorem unit_digit_product_7858_1086_4582_9783 :
  unit_digit_product (unit_digit_product (unit_digit_product unit_digit_7858 unit_digit_1086) unit_digit_4582) unit_digit_9783 = 4 :=
  by
  sorry

end unit_digit_product_7858_1086_4582_9783_l424_424046


namespace young_li_age_l424_424861

theorem young_li_age (x : ℝ) (old_li_age : ℝ) 
  (h1 : old_li_age = 2.5 * x)  
  (h2 : old_li_age + 10 = 2 * (x + 10)) : 
  x = 20 := 
by
  sorry

end young_li_age_l424_424861


namespace Ann_trip_takes_longer_l424_424396

theorem Ann_trip_takes_longer (mary_distance : ℕ) (mary_speed : ℕ)
                              (ann_distance : ℕ) (ann_speed : ℕ)
                              (mary_time : ℕ) (ann_time : ℕ) :
  mary_distance = 630 →
  mary_speed = 90 →
  ann_distance = 800 →
  ann_speed = 40 →
  mary_time = mary_distance / mary_speed →
  ann_time = ann_distance / ann_speed →
  (ann_time - mary_time) = 13 :=
by
  intros
  calculate!
  sorry

end Ann_trip_takes_longer_l424_424396


namespace inverse_function_log3_l424_424821

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x

theorem inverse_function_log3 :
  ∀ x : ℝ, x > 0 →
  ∃ y : ℝ, f (3 ^ y) = y := 
sorry

end inverse_function_log3_l424_424821


namespace hyperbola_midpoint_distance_l424_424655

-- Define the conditions
noncomputable def hyperbola_conditions (x y : ℝ) : Prop := 
  ∃ F1 F2 : ℝ × ℝ, 
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
    (F1 = (-2, 0) ∧ F2 = (2, 0)) ∧
    (abs (x - F2.1) = 4) ∧
    (2 / (a ^ 2 + b ^ 2).sqrt = 1 / ((x ^ 2 + y ^ 2).sqrt))

-- Define the question
def distance_M_to_O_eq_three : Prop :=
  ∀ (P F1 F2 : ℝ × ℝ) (M O : ℝ × ℝ), 
    (M = midpoint (P,F2)) ∧ (O = midpoint (F1,F2)) ∧
    (|M - O| = 3)

-- Define the problem
theorem hyperbola_midpoint_distance :
  ∀ x y : ℝ, hyperbola_conditions x y → distance_M_to_O_eq_three :=
by
  intros x y h
  sorry

end hyperbola_midpoint_distance_l424_424655


namespace flour_needed_l424_424957

theorem flour_needed (flour_per_recipe : ℕ) (pancakes_per_recipe : ℕ) (desired_pancakes : ℕ) :
  flour_per_recipe = 3 →
  pancakes_per_recipe = 20 →
  desired_pancakes = 180 →
  (desired_pancakes / pancakes_per_recipe * flour_per_recipe) = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end flour_needed_l424_424957


namespace bruce_bank_savings_l424_424983

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end bruce_bank_savings_l424_424983


namespace continuous_square_integral_zero_implies_zero_l424_424794

open Set
open IntervalIntegral

noncomputable section

variable {R : Type*} [MeasureSpace R] [NormedSpace ℝ R]

noncomputable def X := Icc (0 : ℝ) 1 ×ˢ Icc (0 : ℝ) 1

noncomputable def boundary_condition (f : R × R → ℝ) (Y : Set (R × R)) : Prop :=
  Y ⊆ X ∧ (∃ x_boundary, (∀ x ∈ Y, x.fst ∈ Icc (0 : ℝ) 1 ∧ x.snd = 0) ∨ (∀ x ∈ Y, x.fst ∈ Icc (0 : ℝ) 1 ∧ x.snd = 1) ∨ (∀ x ∈ Y, x.snd ∈ Icc (0 : ℝ) 1 ∧ x.fst = 0) ∨ (∀ x ∈ Y, x.snd ∈ Icc (0 : ℝ) 1 ∧ x.fst = 1))

theorem continuous_square_integral_zero_implies_zero (f : R × R → ℝ) (h_cont : ContinuousOn f X)
  (h_zero : ∀ Y : Set (R × R), boundary_condition f Y → ∫ (x : (R × R)) in Y, f x = 0) :
  ∀ p ∈ X, f p = 0 := 
sorry

end continuous_square_integral_zero_implies_zero_l424_424794


namespace find_scalar_d_l424_424372

variables {ℝ : Type*} [innerm : has_inner ℝ E] [has_cross : has_cross ℝ E] [decidable_eq E] 
  [finite_dimensional ℝ E] [inner_product_space ℝ E]

def vector_a (a1 a2 a3 : ℝ) : E := a1 • (1 : E) + a2 • (2 : E) + a3 • (3 : E)

def vector_eq (a : E) (d : ℝ) (v : E) : Prop :=
  (1 : E) × (v × (1 : E)) + (2 : E) × (v × (2 : E)) + (3 : E) × (v × (3 : E)) + a × v = d • v

theorem find_scalar_d (a1 a2 a3 : ℝ) :
  ∀v : E, vector_eq (vector_a a1 a2 a3) 2 v :=
by
  sorry

end find_scalar_d_l424_424372


namespace matrix_condition_l424_424455

theorem matrix_condition (x y z : ℝ) 
  (h : let M := ![
    ![0, 3 * y, 2 * z],
    ![2 * x, 2 * y, -z],
    ![2 * x, -3 * y, 2 * z]
  ] 
  in Mᵀ ⬝ M = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ)) : 
  x^2 + y^2 + z^2 = 179 / 396 := 
sorry

end matrix_condition_l424_424455


namespace hyperbola_equation_range_of_k_l424_424224

-- Problem (Ⅰ)
theorem hyperbola_equation
  (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (c : ℝ) (h₂ : c = 2)
  (h₃ : b / a = sqrt 3 / 3) :
  (c^2 = a^2 + b^2) → (a = sqrt 3) → (b = 1) → (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 = 1) :=
sorry

-- Problem (Ⅱ)
theorem range_of_k
  (k : ℝ)
  (intersects_distinct : ∀ k : ℝ, 
    (∀ x y : ℝ, y = k * x + sqrt 2 → x^2 - 3 * y^2 = 3) →  
    ((1 - 3 * k^2) * x^2 - 6 * sqrt 2 * k * x - 9 = 0) →
    (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ x1 x2 : ℝ, x1 + x2 = (6 * sqrt 2 * k)/(1 - 3 * k^2) ∧
    x1 * x2 = -9 / (1 - 3 * k^2) ∧ 
    (1 + k^2) * x1 * x2 + sqrt 2 * k * (x1 + x2) + 2 > 0))) :
  k^2 < 1 ∧ k^2 ≠ 1 / 3 → k^2 > 1 / 3 → (-1 < k ∧ k < -sqrt 3 / 3 ∨
  sqrt 3 / 3 < k ∧ k < 1) :=
sorry

end hyperbola_equation_range_of_k_l424_424224


namespace Katya_possible_numbers_l424_424926

def divisible_by (n m : ℕ) : Prop := m % n = 0

def possible_numbers (n : ℕ) : Prop :=
  let condition1 := divisible_by 7 n  -- Alyona's condition
  let condition2 := divisible_by 5 n  -- Lena's condition
  let condition3 := n < 9             -- Rita's condition
  (condition1 ∨ condition2) ∧ condition3 ∧ 
  ((condition1 ∧ condition3 ∧ ¬condition2) ∨ (condition2 ∧ condition3 ∧ ¬condition1))

theorem Katya_possible_numbers :
  ∀ n : ℕ, 
    (possible_numbers n) ↔ (n = 5 ∨ n = 7) :=
sorry

end Katya_possible_numbers_l424_424926


namespace max_pairs_anna_can_achieve_l424_424587

theorem max_pairs_anna_can_achieve : 
  ∀ (S : Finset ℕ), S = (Finset.range 2022).map (λ n, n + 1) → 
  ∃ N, N = 3 * 337 - 3 :=
by
  intros S hS
  use (3 * 337 - 3)
  sorry

end max_pairs_anna_can_achieve_l424_424587


namespace line_tangent_to_circle_l424_424689

noncomputable def tangent_line_circle (a b θ : ℝ) (h_distinct_roots : a ≠ b)
  (h_roots : ∀ x : ℝ, x^2 + x / (Real.tan θ) - 1 / (Real.sin θ) = 0 ↔ x = a ∨ x = b)
  : Prop :=
let line_AB := λ x, (a + b) * (x - (a + b) / 2) + (a^2 + b^2) / 2 in
let distance := abs (1 / (Real.sin θ)) / real.sqrt (1 + (1 / (Real.tan θ))^2) in
distance = 1

theorem line_tangent_to_circle (a b θ : ℝ) (h_distinct_roots : a ≠ b)
  (h_roots : ∀ x : ℝ, x^2 + x / (Real.tan θ) - 1 / (Real.sin θ) = 0 ↔ x = a ∨ x = b)
  : tangent_line_circle a b θ h_distinct_roots h_roots :=
by sorry

end line_tangent_to_circle_l424_424689


namespace balance_difference_l424_424775

def compounded_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

def simple_interest_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

/-- Cedric deposits $15,000 into an account that pays 6% interest compounded annually,
    Daniel deposits $15,000 into an account that pays 8% simple annual interest.
    After 10 years, the positive difference between their balances is $137. -/
theorem balance_difference :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 10
  compounded_balance P r_cedric t - simple_interest_balance P r_daniel t = 137 := 
sorry

end balance_difference_l424_424775


namespace complementary_event_l424_424325

def batch_has_more_than_4_defective_products : Prop := sorry
def among_4_selected_products there_is_at_least_one_defective (s: set Product) : Prop := sorry

theorem complementary_event (s: set Product) 
  (h1: batch_has_more_than_4_defective_products)
  (h2: s.card = 4) 
  (h3: among_4_selected_products s) : 
  ∃ (s': set Product), s'.card = 4 ∧ ¬ among_4_selected_products s' :=
sorry

end complementary_event_l424_424325


namespace existence_of_barycentric_coordinates_uniqueness_of_barycentric_coordinates_l424_424382

noncomputable def barycentric_coordinates (X A1 A2 A3 A4 : ℝ^3) : Prop :=
∃ (m1 m2 m3 m4 : ℝ), 
  (m1 + m2 + m3 + m4 = 1) ∧ 
  (X = m1 • A1 + m2 • A2 + m3 • A3 + m4 • A4)

theorem existence_of_barycentric_coordinates (A1 A2 A3 A4 X : ℝ^3) :
  barycentric_coordinates X A1 A2 A3 A4 := 
sorry

theorem uniqueness_of_barycentric_coordinates (A1 A2 A3 A4 X : ℝ^3)
  (m1 m2 m3 m4 n1 n2 n3 n4 : ℝ)
  (h₁ : m1 + m2 + m3 + m4 = 1)
  (h₂ : n1 + n2 + n3 + n4 = 1)
  (hx1 : X = m1 • A1 + m2 • A2 + m3 • A3 + m4 • A4)
  (hx2 : X = n1 • A1 + n2 • A2 + n3 • A3 + n4 • A4) :
  m1 = n1 ∧ m2 = n2 ∧ m3 = n3 ∧ m4 = n4 :=
sorry

end existence_of_barycentric_coordinates_uniqueness_of_barycentric_coordinates_l424_424382


namespace solve_for_x_l424_424423

theorem solve_for_x : ∀ x : ℝ, (x - 27) / 3 = (3 * x + 6) / 8 → x = -234 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l424_424423


namespace percentage_calculation_l424_424505

variable (x : Real)
variable (hx : x > 0)

theorem percentage_calculation : 
  ∃ p : Real, p = (0.18 * x) / (x + 20) * 100 :=
sorry

end percentage_calculation_l424_424505


namespace tan_sum_eq_neg_one_l424_424712

theorem tan_sum_eq_neg_one (α β : ℝ) 
  (h : 2 * sin β * sin (α - π / 4) = sin (α - β + π / 4)) 
  : tan (α + β) = -1 := 
by 
sry

end tan_sum_eq_neg_one_l424_424712


namespace christine_distance_l424_424601

theorem christine_distance (speed time : ℝ) (h_speed : speed = 4) (h_time : time = 5) : 
  speed * time = 20 :=
by
  rw [h_speed, h_time]
  norm_num
  sorry

end christine_distance_l424_424601


namespace painting_area_l424_424506

theorem painting_area (c t A : ℕ) (h1 : c = 15) (h2 : t = 840) (h3 : c * A = t) : A = 56 := 
by
  sorry -- proof to demonstrate A = 56

end painting_area_l424_424506


namespace problem_solution_l424_424841

noncomputable def f (x a : ℝ) : ℝ :=
  2 * (Real.cos x)^2 - 2 * a * Real.cos x - (2 * a + 1)

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a < 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem problem_solution :
  g a = 1 ∨ g a = (-a^2 / 2 - 2 * a - 1) ∨ g a = 1 - 4 * a →
  (∀ a, g a = 1 / 2 → a = -1) ∧ (f x (-1) ≤ 5) :=
sorry

end problem_solution_l424_424841


namespace number_of_lines_mowed_l424_424987

/-- Define the parameters in the problem: mowing time per line, rows, flowers per row, time per flower, and total gardening time. -/
def mowing_time_per_line := 2
def rows := 8
def flowers_per_row := 7
def time_per_flower := 0.5
def total_gardening_time := 108

/-- Prove that the number of lines mowed is 40 given the conditions above. -/
theorem number_of_lines_mowed : 
  let total_flowers := rows * flowers_per_row,
      time_planting := total_flowers * time_per_flower,
      time_left_for_mowing := total_gardening_time - time_planting,
      lines_mowed := time_left_for_mowing / mowing_time_per_line in
  lines_mowed = 40 :=
by
  sorry

end number_of_lines_mowed_l424_424987


namespace ab_value_l424_424884

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end ab_value_l424_424884


namespace ray_dissects_AFB_l424_424551

-- Define the cyclic quadrilateral
variables {A B C D E F I : Type}
variables (cyclic_quadrilateral : CyclicQuadrilateral A B C D)
variables (intersection_1 : ∃ E : Type, Line A B ∩ Line D C = {E})
variables (intersection_2 : ∃ F : Type, Line B C ∩ Line A D = {F})
variables (I_incenter : I = incenter (Triangle A E D))
variables (ray_perp_bisector : ∃ G : Type, Ray F G ⊥ angle_bisector A I D)

-- The theorem statement
theorem ray_dissects_AFB : 
  ratio_angles (angle (Ray F G) (angle A F B)) = 1 / 3 :=
sorry

end ray_dissects_AFB_l424_424551


namespace _l424_424622

noncomputable def area_cos_2ϕ : ℝ :=
  2 * ∫ (θ : ℝ) in -π/4 .. π/4, ∫ (r : ℝ) in 0 .. cos (2*θ), r

noncomputable theorem area_cos_2ϕ_val : area_cos_2ϕ = π / 2 := by
  sorry

noncomputable def area_sinϕ_4_minus_2 : ℝ :=
  ∫ (θ : ℝ) in 0 .. π, ∫ (r : ℝ) in 2 * sin θ .. 4 * sin θ, r

noncomputable theorem area_sinϕ_4_minus_2_val : area_sinϕ_4_minus_2 = 3 * π := by
  sorry

noncomputable def area_sin_3ϕ : ℝ :=
  3 * ∫ (θ : ℝ) in 0 .. π/3, ∫ (r : ℝ) in 0 .. sin (3*θ), r

noncomputable theorem area_sin_3ϕ_val : area_sin_3ϕ = 3 * π / 4 := by
  sorry

end _l424_424622


namespace proposition_A_iff_proposition_B_l424_424386

-- Define propositions
def Proposition_A (A B C : ℕ) : Prop := (A = 60 ∨ B = 60 ∨ C = 60)
def Proposition_B (A B C : ℕ) : Prop :=
  (A + B + C = 180) ∧ 
  (2 * B = A + C)

-- The theorem stating the relationship between Proposition_A and Proposition_B
theorem proposition_A_iff_proposition_B (A B C : ℕ) :
  Proposition_A A B C ↔ Proposition_B A B C :=
sorry

end proposition_A_iff_proposition_B_l424_424386


namespace value_of_a_l424_424702

open Set

variable {α : Type*}

theorem value_of_a 
  {a : α} : 
  let U : Set α := {-1, 2, 3, a}
  let M : Set α := {-1, 3}
  let complement_U_M : Set α := {2, 5}
  M = U \ complement_U_M → 
  a = 5 :=
by
  intros U M complement_U_M h
  exact sorry

end value_of_a_l424_424702


namespace jasper_class_students_received_b_l424_424323

theorem jasper_class_students_received_b (h_thompson_ratio : 12 / 16 = 3 / 4)
                                         (h_jasper_total : 32 = 32) :
  ∃ y : ℕ, (3 / 4 = y / 32) ∧ y = 24 :=
by
  use 24
  split
  sorry

end jasper_class_students_received_b_l424_424323


namespace polynomial_decreases_by_two_l424_424513

noncomputable def polynomial_example (b : ℤ) : Polynomial ℤ :=
Polynomial.C b - Polynomial.C 2 * Polynomial.x

theorem polynomial_decreases_by_two (P : Polynomial ℤ) :
  (∀ x : ℤ, P.eval (x + 1) = P.eval x - 2) →
  ∃ b : ℤ, P = polynomial_example b :=
by
  sorry

end polynomial_decreases_by_two_l424_424513


namespace intersection_eq_35_l424_424664

def setA : Set ℝ := {x | abs (x - 1) ≤ 4}
def setB : Set ℝ := {x | log x / log 3 > 1}

theorem intersection_eq_35 :
  {x | abs (x - 1) ≤ 4} ∩ {x | log x / log 3 > 1}
  = {x | 3 < x ∧ x ≤ 5} :=
by
  sorry

end intersection_eq_35_l424_424664


namespace jane_babysitting_start_l424_424354

-- Definitions based on the problem conditions
def jane_current_age := 32
def years_since_babysitting := 10
def oldest_current_child_age := 24

-- Definition for the starting babysitting age
def starting_babysitting_age : ℕ := 8

-- Theorem statement to prove
theorem jane_babysitting_start (h1 : jane_current_age - years_since_babysitting = 22)
  (h2 : oldest_current_child_age - years_since_babysitting = 14)
  (h3 : ∀ (age_jane age_child : ℕ), age_child ≤ age_jane / 2) :
  starting_babysitting_age = 8 :=
by
  sorry

end jane_babysitting_start_l424_424354


namespace calculated_area_error_l424_424468

def percentage_error_area (initial_length_error : ℝ) (initial_width_error : ℝ) 
(temperature_change : ℝ) (humidity_change : ℝ) 
(length_error_per_temp : ℝ) (width_error_per_humidity : ℝ) : ℝ :=
let total_length_error := initial_length_error + (temperature_change / 5) * length_error_per_temp in
let total_width_error := initial_width_error + (humidity_change / 10) * width_error_per_humidity in
total_length_error - total_width_error

theorem calculated_area_error :
  percentage_error_area 3 2 15 20 1 0.5 = 3 :=
sorry

end calculated_area_error_l424_424468


namespace triangle_is_3_l424_424625

def base6_addition_valid (delta : ℕ) : Prop :=
  delta < 6 ∧ 
  2 + delta + delta + 4 < 6 ∧ -- No carry effect in the middle digits
  ((delta + 3) % 6 = 4) ∧
  ((5 + delta + (2 + delta + delta + 4) / 6) % 6 = 3) ∧
  ((4 + (5 + delta + (2 + delta + delta + 4) / 6) / 6) % 6 = 5)

theorem triangle_is_3 : ∃ (δ : ℕ), base6_addition_valid δ ∧ δ = 3 :=
by
  use 3
  sorry

end triangle_is_3_l424_424625


namespace probability_same_person_given_same_look_l424_424859

-- Definitions based on conditions
def total_students := 36
def group_sizes := [1, 2, 3, 4, 5, 6, 7, 8]
def sum_of_group_sizes := group_sizes.sum
def sum_of_squares := group_sizes.map (λ x => x * x).sum
def total_pairs : ℕ := total_students * total_students
def same_person_pairs := total_students

-- Proof statement
theorem probability_same_person_given_same_look :
  (sum_of_group_sizes = total_students) →
  (sum_of_squares = 204) →
  (total_pairs = 1296) →
  (same_person_pairs = 36) →
  (36 / 204 = (3 / 17 : ℚ)) :=
by sorry

end probability_same_person_given_same_look_l424_424859


namespace num_pairs_of_nat_numbers_satisfying_eq_l424_424282

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l424_424282


namespace vector_problem_l424_424704

-- Define the vector type in this context
def Vector := (ℝ × ℝ)

-- Given conditions
def a : Vector := (1, 2)
def b (m : ℝ) : Vector := (-2, m)

-- Parallel condition
def parallel (v1 v2 : Vector) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Lean statement for the proof problem
theorem vector_problem (m : ℝ) (h : parallel a (b m)) : 2 * a.1 + 3 * b m.1 = -4 ∧ 2 * a.2 + 3 * b m.2 = -8 := 
by
  -- We will use sorry here to acknowledge the proof needs to be done
  sorry

end vector_problem_l424_424704


namespace store_raised_price_l424_424602

variable (P : ℝ) (x : ℝ)

theorem store_raised_price:
  (0.85 * P = 61.2) →
  (P - 67.5 = 4.5) →
  (P = 72) →
  (x = (630 / 61.2)) →
  x ≈ 10.29 :=
by
  sorry

end store_raised_price_l424_424602


namespace football_team_practice_hours_l424_424086

-- Definitions for each day's practice adjusted for weather events
def monday_hours : ℕ := 4
def tuesday_hours : ℕ := 5 - 1
def wednesday_hours : ℕ := 0
def thursday_hours : ℕ := 5
def friday_hours : ℕ := 3 + 2
def saturday_hours : ℕ := 4
def sunday_hours : ℕ := 0

-- Total practice hours calculation
def total_practice_hours : ℕ := 
  monday_hours + tuesday_hours + wednesday_hours + 
  thursday_hours + friday_hours + saturday_hours + 
  sunday_hours

-- Statement to prove
theorem football_team_practice_hours : total_practice_hours = 22 := by
  sorry

end football_team_practice_hours_l424_424086


namespace total_time_late_l424_424592

theorem total_time_late
  (charlize_late : ℕ)
  (classmate_late : ℕ → ℕ)
  (h1 : charlize_late = 20)
  (h2 : ∀ n, n < 4 → classmate_late n = charlize_late + 10) :
  charlize_late + (∑ i in Finset.range 4, classmate_late i) = 140 := by
  sorry

end total_time_late_l424_424592


namespace bridge_angle_sum_l424_424939

theorem bridge_angle_sum 
  (ABC_isosceles : ∀ {A B C : ℝ} (h : Triangle A B C), is_isosceles A B C ∧ angle A B C = angle A C B) 
  (angle_BAC : angle A B C = 25) 
  (DEF_right : ∀ {D E F : ℝ}, (angle D E F) = 90) 
  (angle_EDF : angle E D F = 40)
  (AD_parallel_BC : ∀ {A D B C : ℝ}, parallel A D B C) :
  angle D A C + angle A D E = 127.5 := 
by
  sorry

end bridge_angle_sum_l424_424939


namespace sin_double_angle_l424_424680

theorem sin_double_angle (α : ℝ) (P : ℝ × ℝ) (hP : P.snd = 2 * P.fst) :
  ∃ r : ℝ, r ≠ 0 → |P.fst| = r / sqrt (P.fst ^ 2 + (2 * P.fst) ^ 2) → sin (2 * α) = 4 / 5 :=
by
  sorry

end sin_double_angle_l424_424680


namespace six_people_with_A_not_on_ends_l424_424071

-- Define the conditions and the problem statement
def standing_arrangements (n : ℕ) (A : Type) :=
  {l : List A // l.length = n}

theorem six_people_with_A_not_on_ends : 
  (arr : standing_arrangements 6 ℕ) → 
  (∀ a ∈ arr.val, a ≠ 0 ∧ a ≠ 5) → 
  ∃! (total_arrangements : ℕ), total_arrangements = 480 :=
  by
    sorry

end six_people_with_A_not_on_ends_l424_424071


namespace infinitely_many_sums_of_two_squares_l424_424792

theorem infinitely_many_sums_of_two_squares :
  ∃ᶠ n : ℕ, ∃ a b c d e f : ℕ,
    (n = a^2 + b^2) ∧ (n + 1 = c^2 + d^2) ∧ (n + 2 = e^2 + f^2) :=
begin
  sorry
end

end infinitely_many_sums_of_two_squares_l424_424792


namespace average_price_per_book_l424_424520

-- Definitions of the conditions
def books_shop1 := 65
def cost_shop1 := 1480
def books_shop2 := 55
def cost_shop2 := 920

-- Definition of total values
def total_books := books_shop1 + books_shop2
def total_cost := cost_shop1 + cost_shop2

-- Proof statement
theorem average_price_per_book : (total_cost / total_books) = 20 := by
  sorry

end average_price_per_book_l424_424520


namespace tax_free_items_cost_l424_424140

variable (total_worth : ℝ) (sales_tax : ℝ) (tax_rate : ℝ) (tax_free_items_cost : ℝ)

theorem tax_free_items_cost :
  total_worth = 25 →
  sales_tax = 0.30 →
  tax_rate = 0.10 →
  tax_free_items_cost = total_worth - (sales_tax / tax_rate) → 
  tax_free_items_cost = 22 := by
  intros h_total h_tax h_rate h_calculation
  rw [h_total, h_tax, h_rate] at h_calculation
  exact h_calculation

end tax_free_items_cost_l424_424140


namespace at_least_one_variety_has_27_apples_l424_424018

theorem at_least_one_variety_has_27_apples (total_apples : ℕ) (varieties : ℕ) 
  (h_total : total_apples = 105) (h_varieties : varieties = 4) : 
  ∃ v : ℕ, v ≥ 27 := 
sorry

end at_least_one_variety_has_27_apples_l424_424018


namespace number_of_correct_propositions_l424_424235

structure Planes :=
(α β γ : Type)
(l : Type)
(a b : Type)

-- Propositions
def prop1 (p: Planes) : Prop := (⊥ p.α p.γ) ∧ (⊥ p.β p.γ) ∧ (∃ l, p.α ∩ p.β = l) → ⊥ l p.γ
def prop2 (p: Planes) : Prop := (∃ n_points, ∀ n_points \in p.l, n_points ∉ p.α) → ∥ p.l p.α
def prop3 (p: Planes) : Prop := ¬ ∥ p.a p.b ∧ (p.a ⊆ p.α) ∧ ∥ p.a p.β ∧ (p.b ⊆ p.β) ∧ ∥ p.b p.α → ∥ p.α p.β
def prop4 (p: Planes) : Prop := (∃ line1, line1 ∈ p.α) → (∃ l2 ∈ p.α, ∀ l_other ∈ p.β, ⊥ l2 l_other)

-- Number of correct propositions
theorem number_of_correct_propositions (p: Planes) : 
  ((prop1 p) ∧ (prop2 p = False) ∧ (prop3 p) ∧ (prop4 p)) → (num_true_propositions = 3) :=
by
  sorry  -- Proof details not required


end number_of_correct_propositions_l424_424235


namespace tangent_points_l424_424808

theorem tangent_points (x y : ℝ) (h : y = x^3 - 3 * x) (slope_zero : 3 * x^2 - 3 = 0) :
  (x = -1 ∧ y = 2) ∨ (x = 1 ∧ y = -2) :=
sorry

end tangent_points_l424_424808


namespace count_valid_integers_l424_424709

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def valid_integer (n : ℕ) : Prop := 200 ≤ n ∧ n < 400 ∧ sum_of_digits n = 17

theorem count_valid_integers : (Finset.filter valid_integer (Finset.Ico 200 400)).card = 17 := by sorry

end count_valid_integers_l424_424709


namespace arithmetic_sum_nineteen_nonnegative_l424_424663

-- Define the conditions
variable {a : ℕ → ℝ} -- arithmetic sequence
variable {d : ℝ} (h_d : d ≠ 0) -- non-zero common difference
variable Sn : ℕ → ℝ -- sum of the first n terms

-- The key condition that for all positive integers n, Sn ≤ S10
variable (h_sum : ∀ n : ℕ, 0 < n → Sn n ≤ Sn 10)

-- Prove that S19 ≥ 0
theorem arithmetic_sum_nineteen_nonnegative 
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d) -- arithmetic sequence definition
  (h_sum_def : ∀ n : ℕ, Sn n = n * a 1 + n * (n - 1) / 2 * d) -- sum formula
  :
  Sn 19 ≥ 0 := 
sorry

end arithmetic_sum_nineteen_nonnegative_l424_424663


namespace find_a_l424_424681

def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2

def slope_at_x (a : ℝ) (x : ℝ) : ℝ := 
  let y := curve a x
  derivative (curve a) x

theorem find_a (a : ℝ) (h : slope_at_x a 1 = -4) : a = -2 :=
  by
  sorry

end find_a_l424_424681


namespace volume_of_tetrahedron_l424_424660

variables (R : ℝ) (r : ℝ) (a : ℝ) (V : ℝ)

-- Given conditions
def sphere_radius : Prop := R = sqrt 22 / 2
def intersection_length : Prop := 4 * (2 * π * r) = 8 * π
def intersection_radius : Prop := 2 * π * r = 2 * π
def circumsphere_height_relation : Prop := R ^ 2 - r ^ 2 = (sqrt 6 * a / 12) ^ 2
def volume_formula : Prop := V = sqrt 2 / 12 * a ^ 3

-- Theorem statement
theorem volume_of_tetrahedron (h_radius : sphere_radius R) (h_length : intersection_length r)
    (h_circle_radius : intersection_radius r) (h_relation : circumsphere_height_relation R r a)
    (h_volume : volume_formula V a) : V = 54 * sqrt 6 := by
  sorry

end volume_of_tetrahedron_l424_424660


namespace winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l424_424430

def game (n : ℕ) : Prop :=
  ∃ A_winning_strategy B_winning_strategy neither_winning_strategy,
    (n ≥ 8 → A_winning_strategy) ∧
    (n ≤ 5 → B_winning_strategy) ∧
    (n = 6 ∨ n = 7 → neither_winning_strategy)

theorem winning_strategy_for_A (n : ℕ) (h : n ≥ 8) :
  game n :=
sorry

theorem winning_strategy_for_B (n : ℕ) (h : n ≤ 5) :
  game n :=
sorry

theorem no_winning_strategy (n : ℕ) (h : n = 6 ∨ n = 7) :
  game n :=
sorry

end winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l424_424430


namespace problem_solution_set_l424_424376

open Real

theorem problem_solution_set (f : ℝ → ℝ) (h_deriv : ∀ x, differentiable_at ℝ f x)
  (h_ineq : ∀ x, f(x) + deriv f x < 1) (h_f0 : f(0) = 2015) :
  {x | e^x * f(x) - e^x > 2014} = Iio 0 :=
by
  sorry

end problem_solution_set_l424_424376


namespace tangent_line_parabola_l424_424837

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l424_424837


namespace work_problem_solution_l424_424535

-- Definitions for the given conditions
def persons := ℕ
def days := ℕ
def length := ℕ

-- Given constants and conditions
constant W1 : ℕ
constant persons1 : persons := 15
constant days1 : days := 42
constant length1 : length := 90
constant persons2 : persons := 40
constant length2 : length := 60

-- Work done is directly proportional to the number of workers and the time they work
axiom work_done (p : persons) (d : days) : ℕ

-- Calculate work done in the first case
axiom W1_eq : W1 = work_done persons1 days1

-- Calculate the work done in the second case assuming it's (2/3) of W1
def W2 : ℕ := (2/3) * W1

-- Statement to prove
theorem work_problem_solution : ∃ D2 : ℝ, 
  D2 = 10.5 ∧ 
  work_done persons2 D2.toInt = W2 :=
sorry

end work_problem_solution_l424_424535


namespace radius_of_sphere_with_surface_area_4pi_l424_424685

noncomputable def sphere_radius (surface_area: ℝ) : ℝ :=
  sorry

theorem radius_of_sphere_with_surface_area_4pi :
  sphere_radius (4 * Real.pi) = 1 :=
by
  sorry

end radius_of_sphere_with_surface_area_4pi_l424_424685


namespace number_of_girls_in_first_year_l424_424077

theorem number_of_girls_in_first_year
  (total_students : ℕ)
  (sample_size : ℕ)
  (boys_in_sample : ℕ)
  (girls_in_first_year : ℕ) :
  total_students = 2400 →
  sample_size = 80 →
  boys_in_sample = 42 →
  girls_in_first_year = total_students * (sample_size - boys_in_sample) / sample_size →
  girls_in_first_year = 1140 :=
by 
  intros h1 h2 h3 h4
  sorry

end number_of_girls_in_first_year_l424_424077


namespace radius_C1_l424_424603

open Real

variables (C1 C2 : set (ℝ × ℝ)) (O X Y Z : ℝ × ℝ)
variable [metric_space (ℝ × ℝ)]

-- Define the centers and positions
def center (A : set (ℝ × ℝ)) := {x | ∀ y ∈ A, dist x y = dist x y} -- simplified definition
noncomputable def radius (C : set (ℝ × ℝ)) (O : ℝ × ℝ) : ℝ :=
  classical.some (exists_dist_eq_r _ O)

-- Hypotheses
axiom hC1C2 : C1 ⊆ C2
axiom hC1_inter_C2 : ∃ X Y, X ∈ C1 ∧ X ∈ C2 ∧ Y ∈ C1 ∧ Y ∈ C2
axiom hZ_pos : Z ∉ C1 ∧ Z ∈ C2
axiom hXZ : dist X Z = 15
axiom hOZ : dist O Z = 5
axiom hYZ : dist Y Z = 12

-- Distance properties and proof goal
theorem radius_C1 (hOX : dist O X = dist O Y) : radius C1 O = 5 * sqrt 10 :=
by {
  sorry
}

end radius_C1_l424_424603


namespace correct_statements_l424_424909

theorem correct_statements (x : ℝ) (y : ℝ) (A B : Set ℝ) (P : Set ℝ → ℝ)
  (P_pos_A : 0 < P A) (P_pos_B : 0 < P B)
  (P_cond : P (A ∩ B) / P B = P A)
  (r_reg_eq : y = 0.3 - 0.7 * x) :
  ¬(∃ (s : Fin 10 → ℝ), variance s = 3 ∧ variance (λ i, 2 * s i + 1) = 7) ∧ 
  ∃ (x y : Réel), r_reg_eq → correlation x y < 0 ∧
  (P (A ∩ B) = P A * P B) :=
by
  sorry

end correct_statements_l424_424909


namespace inequality_holds_l424_424364

variable {a b c : ℝ}

theorem inequality_holds (h : a > 0) (h' : b > 0) (h'' : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end inequality_holds_l424_424364


namespace cows_now_l424_424093

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end cows_now_l424_424093


namespace increasing_function_on_interval_l424_424817

noncomputable def f (a x : ℝ) : ℝ := -x^2 + 2 * (a - 1) * x + 2

theorem increasing_function_on_interval (a : ℝ) :
  (∀ x, x < 4 → f(a, x) < f(a, x + 1)) → a ≥ 5 :=
sorry

end increasing_function_on_interval_l424_424817


namespace machine_transport_equation_l424_424387

theorem machine_transport_equation (x : ℝ) :
  (∀ (rateA rateB : ℝ), rateB = rateA + 60 → (500 / rateA = 800 / rateB) → rateA = x → rateB = x + 60) :=
by
  sorry

end machine_transport_equation_l424_424387


namespace problem_statement_l424_424141

variable {f : ℝ → ℝ}

-- Condition 1: The function f satisfies (x - 1)f'(x) ≤ 0
def cond1 (f : ℝ → ℝ) : Prop := ∀ x, (x - 1) * (deriv f x) ≤ 0

-- Condition 2: The function f satisfies f(-x) = f(2 + x)
def cond2 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (2 + x)

theorem problem_statement (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h_cond1 : cond1 f)
  (h_cond2 : cond2 f)
  (h_dist : abs (x₁ - 1) < abs (x₂ - 1)) :
  f (2 - x₁) > f (2 - x₂) :=
sorry

end problem_statement_l424_424141


namespace simplify_and_evaluate_l424_424798

-- Definitions for the variables and the expression
variable (x y z : ℤ)
def expression := 3 * x - 2 * y - [2 * x + 2 * y - (2 * x * y * z + x + 2 * z) - 4 * x + 2 * z] - x * y * z

-- Setting the variable values according to the conditions
def x_val := -1
def y_val := -2
def z_val := 3

-- Prove that simplifying the expression yields 8 for the given values
theorem simplify_and_evaluate : expression x_val y_val z_val = 8 := by
  sorry

end simplify_and_evaluate_l424_424798


namespace find_sum_l424_424110

noncomputable def sumPutAtSimpleInterest (R: ℚ) (P: ℚ) := 
  let I := P * R * 5 / 100
  I + 90 = P * (R + 6) * 5 / 100 → P = 300

theorem find_sum (R: ℚ) (P: ℚ) : sumPutAtSimpleInterest R P := by
  sorry

end find_sum_l424_424110


namespace total_candles_in_small_boxes_l424_424645

-- Definitions of the conditions
def num_small_boxes_per_big_box := 4
def num_big_boxes := 50
def candles_per_small_box := 40

-- The total number of small boxes
def total_small_boxes : Nat := num_small_boxes_per_big_box * num_big_boxes

-- The statement to prove the total number of candles in all small boxes is 8000
theorem total_candles_in_small_boxes : candles_per_small_box * total_small_boxes = 8000 :=
by 
  sorry

end total_candles_in_small_boxes_l424_424645


namespace value_of_a_l424_424518

theorem value_of_a 
  (a : ℝ) 
  (h : 0.005 * a = 0.85) : 
  a = 170 :=
sorry

end value_of_a_l424_424518


namespace count_integers_between_300_and_600_with_digit_sum_17_l424_424710

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

theorem count_integers_between_300_and_600_with_digit_sum_17 : 
  (finset.filter (λ n, sum_of_digits n = 17) (finset.Ico 300 601)).card = 30 :=
by 
  sorry

end count_integers_between_300_and_600_with_digit_sum_17_l424_424710


namespace triangle_perimeter_l424_424112

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 19)
  (ineq1 : a + b > c) (ineq2 : a + c > b) (ineq3 : b + c > a) : a + b + c = 44 :=
by
  -- Proof omitted
  sorry

end triangle_perimeter_l424_424112


namespace domain_of_function_l424_424442

theorem domain_of_function :
  {x : ℝ | x ≥ -1} \ {0} = {x : ℝ | (x ≥ -1 ∧ x < 0) ∨ x > 0} :=
by
  sorry

end domain_of_function_l424_424442


namespace missing_exponent_l424_424532

theorem missing_exponent (x : ℝ) (h : (9^5.6 * 9^x) / 9^2.56256 = 9^13.33744) : x = 4.69944 :=
by sorry

end missing_exponent_l424_424532


namespace a_4_value_general_term_l424_424242

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n+1 => 2 * (sequence n) + 1

theorem a_4_value : sequence 4 = 23 := 
  sorry

theorem general_term (n : ℕ) : sequence n = 3 * 2^(n-1) - 1 := 
  sorry

end a_4_value_general_term_l424_424242


namespace area_inside_S_but_outside_R_l424_424099

-- Define the initial setup and conditions
def side_length_hexagon := 2
def side_length_square := 2
def hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * (side_length_hexagon ^ 2)
def total_area_R := hexagon_area + 18 * (side_length_square ^ 2)
def side_length_S := 4
def hexagon_area_S : ℝ := (3 * Real.sqrt 3 / 2) * (side_length_S ^ 2)
def required_area := hexagon_area_S - total_area_R

-- The theorem to prove 
theorem area_inside_S_but_outside_R:
  required_area = 42 * Real.sqrt 3 - 72 := by
  sorry

end area_inside_S_but_outside_R_l424_424099


namespace arithmetic_sequence_general_term_and_sum_sum_of_reciprocal_of_b_n_sequence_l424_424209

theorem arithmetic_sequence_general_term_and_sum
  (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 4 = 14)
  (h3 : a 1 * (a 1 + 6 * d) = (a 1 + d) ^ 2) :
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, S n = 2 * n ^ 2 - n) :=
begin
  sorry
end

theorem sum_of_reciprocal_of_b_n_sequence
  (S : ℕ → ℕ) (b : ℕ → ℕ → ℤ) (T : ℕ → ℕ → ℤ)
  (k1 k2 : ℤ)
  (hb1 : ∀ n, b n k1 = (S n) / (n + k1))
  (hb2 : ∀ n, b n k2 = (S n /(n + k2))
  (h_arth_seq1: ∀ n, 2 * b 2 k1 = b 1 k1 + b 3 k1)
  (h_arth_seq2: ∀ n, 2 * b 2 k2 = b 1 k2 + b 3 k2)
  (h_S : ∀ n, S n = 2 * n ^ 2 - n) :
  (T n k1 = n / (4 * (n + 1))) ∧ (T n k2 = n / (2 * n + 1)) :=
begin
  sorry
end

end arithmetic_sequence_general_term_and_sum_sum_of_reciprocal_of_b_n_sequence_l424_424209


namespace range_of_k_l424_424222

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 3| + |x - 1| > k) ↔ k < 4 :=
by sorry

end range_of_k_l424_424222


namespace area_division_isosceles_right_triangle_l424_424648

/-- 
Given an isosceles right triangle ABC with AC = BC, and a point P on the hypotenuse AB. 
Perpendiculars dropped from P to AC and BC form two triangles and one rectangle.
We want to show that the area of each part cannot be less than 4/9 of the area of triangle ABC.
-/
theorem area_division_isosceles_right_triangle (a : ℝ) (P : {p : ℝ × ℝ // p.1^2 + p.2^2 = (a * Real.sqrt 2)^2 / 2}) :
  let A := (0, 0) : ℝ × ℝ,
      B := (a, 0) : ℝ × ℝ,
      C := (0, a) : ℝ × ℝ,
      triangle_area (p1 p2 p3 : ℝ × ℝ) := 0.5 * |((p2.1 - p1.1) * (p3.2 - p1.2) - (p2.2 - p1.2) * (p3.1 - p1.1))|,
      total_area := triangle_area A B C
  in total_area = a^2 / 2 →
     let AP := P.1,
         PB := P.2,
         h := P.2 / Real.sqrt 2
     in (0.5 * AP * h < total_area * 4 / 9) ∧ 
        (0.5 * PB * h < total_area * 4 / 9) ∧ 
        (h^2 < total_area * 4 / 9) → False :=
sorry

end area_division_isosceles_right_triangle_l424_424648


namespace polynomial_divisible_by_3_l424_424378

/--
Given q and p are integers where q is divisible by 3 and p+1 is divisible by 3,
prove that the polynomial Q(x) = x^3 - x + (p+1)x + q is divisible by 3 for any integer x.
-/
theorem polynomial_divisible_by_3 (q p : ℤ) (hq : 3 ∣ q) (hp1 : 3 ∣ (p + 1)) :
  ∀ x : ℤ, 3 ∣ (x^3 - x + (p+1) * x + q) :=
by {
  sorry
}

end polynomial_divisible_by_3_l424_424378


namespace ratio_of_x_intercepts_l424_424879

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l424_424879


namespace count_pairs_satisfying_condition_l424_424292

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424292


namespace xiao_zhao_physical_education_grade_l424_424940

def classPerformanceScore : ℝ := 40
def midtermExamScore : ℝ := 50
def finalExamScore : ℝ := 45

def classPerformanceWeight : ℝ := 0.3
def midtermExamWeight : ℝ := 0.2
def finalExamWeight : ℝ := 0.5

def overallGrade : ℝ :=
  (classPerformanceScore * classPerformanceWeight) +
  (midtermExamScore * midtermExamWeight) +
  (finalExamScore * finalExamWeight)

theorem xiao_zhao_physical_education_grade : overallGrade = 44.5 := by
  sorry

end xiao_zhao_physical_education_grade_l424_424940


namespace exists_point_D_in_triangle_l424_424616

open Classical

theorem exists_point_D_in_triangle (A B C : Type*) [EuclideanGeometry A B C]
  (D : Type*) [EuclideanGeometry D] :
  ∃ D : Point, (area (triangle A B D) = 1/2 * area (triangle A B C)) ∧ 
               (area (triangle B C D) = 1/6 * area (triangle A B C)) := 
by
  sorry

end exists_point_D_in_triangle_l424_424616


namespace find_xy_sum_l424_424223

open Nat

theorem find_xy_sum (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + x * y = 8) 
  (h2 : y + z + y * z = 15) 
  (h3 : z + x + z * x = 35) : 
  x + y + z + x * y = 15 := 
sorry

end find_xy_sum_l424_424223


namespace line_tangent_parabola_unique_d_l424_424834

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l424_424834


namespace find_angle_ADE_l424_424729

-- Defining the isosceles triangle and given angles
noncomputable def isosceles_triangle_ABC (A B C : Type*) (angle_ABC : ℝ) (angle_DAC : ℝ) (angle_ECA : ℝ) : Prop :=
  ∀ (ABC_isosceles : true) (angle_ABC_eq : angle_ABC = 20) (angle_DAC_eq : angle_DAC = 60) (angle_ECA_eq : angle_ECA = 50),
  ∃ (angle_ADE : ℝ), angle_ADE = 30

-- The statement to be proven
theorem find_angle_ADE (A B C D E : Type*) :
  isosceles_triangle_ABC A B C 20 60 50 :=
begin
  sorry
end

end find_angle_ADE_l424_424729


namespace basketball_team_wins_44_of_remaining_55_l424_424074

def number_of_remaining_wins (total_games remaining_games : ℕ) (initial_wins : ℕ) (desired_percentage : ℚ) : ℕ :=
  let total_season_games := total_games + remaining_games
  let desired_wins := desired_percentage * total_season_games
  let remaining_wins := desired_wins - initial_wins
  return remaining_wins

theorem basketball_team_wins_44_of_remaining_55 :
  number_of_remaining_wins 75 55 60 (4 / 5) = 44 :=
  by
    sorry

end basketball_team_wins_44_of_remaining_55_l424_424074


namespace count_valid_pairs_l424_424258

-- Definition of the predicate indicating the pairs (a, b) satisfying the conditions
def valid_pair (a b : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (a ≥ b) ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 6)

-- Main theorem statement: there are 5 valid pairs
theorem count_valid_pairs : 
  (finset.univ.filter (λ (ab : ℕ × ℕ), valid_pair ab.1 ab.2)).card = 5 :=
by
  sorry

end count_valid_pairs_l424_424258


namespace lucas_eggs_in_baskets_l424_424776

theorem lucas_eggs_in_baskets :
  ∃ n, n = 6 ∧ n ≥ 5 ∧ 30 % n = 0 ∧ 42 % n = 0 :=
begin
  sorry
end

end lucas_eggs_in_baskets_l424_424776


namespace find_a_l424_424667

theorem find_a 
  (α : ℝ)
  (h_sin_cos_roots : ∀ x, 3 * x^2 - 2 * x + a = 0 → (x = sin α ∨ x = cos α))
  : a = -5 / 6 := 
sorry

end find_a_l424_424667


namespace equation_correct_l424_424941

-- Define the number of sportswear and the initial processed sets, and the increased efficiency.
def total_sets := 400
def initial_sets := 160
def remaining_sets := total_sets - initial_sets
def efficiency_increase := 0.20
def total_days := 18

-- Define the equation to be proven
def equation (x : ℝ) :=
  (initial_sets / x) + (remaining_sets / ((1 + efficiency_increase) * x)) = total_days

theorem equation_correct (x : ℝ) : 
  equation x :=
by
  have h1 : (initial_sets / x) = 160 / x := rfl
  have h2 : (remaining_sets / ((1 + efficiency_increase) * x)) = 240 / (1.2 * x) := rfl
  exact sorry

end equation_correct_l424_424941


namespace new_circumference_to_diameter_ratio_l424_424717

theorem new_circumference_to_diameter_ratio (r : ℝ) : 
  let new_radius := r + 2,
      new_circumference := 2 * Real.pi * new_radius,
      new_diameter := 2 * new_radius in
  new_circumference / new_diameter = Real.pi :=
by
  sorry

end new_circumference_to_diameter_ratio_l424_424717


namespace triangle_area_l424_424060

-- Definitions from conditions
variables (P r : ℝ)
def semiperimeter (P : ℝ) : ℝ := P / 2

-- Given conditions
axiom h1 : P = 35
axiom h2 : r = 4.5

-- The goal to prove
theorem triangle_area (h1 : P = 35) (h2 : r = 4.5) : r * semiperimeter P = 78.75 := by
  sorry

end triangle_area_l424_424060


namespace negation_equiv_l424_424003

noncomputable def negate_existential : Prop :=
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0

noncomputable def universal_negation : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0

theorem negation_equiv : negate_existential = universal_negation :=
by
  -- Proof to be filled in
  sorry

end negation_equiv_l424_424003


namespace cyclic_heat_engine_efficiency_l424_424116

-- Definitions and conditions
def isochoric_pressure_reduction (P0 V0 T0 R: ℝ) : Prop :=
  T0 = P0 * V0 / R

def isobaric_density_increase (P0 : ℝ) (V0 : ℝ) (T2 : ℝ) (R: ℝ) : Prop :=
  T2 = P0 * (3 * V0) / R

def max_possible_efficiency (T1 T2 : ℝ) : ℝ :=
  1 - T1 / T2

def cycle_efficiency (η_max : ℝ) : ℝ :=
  η_max / 8

-- Theorem statement
theorem cyclic_heat_engine_efficiency (P0 V0 T0 T1 T2 R: ℝ)
  (h_isochoric : isochoric_pressure_reduction P0 V0 T0 R)
  (h_isobaric : isobaric_density_increase P0 V0 T2 R)
  (h_temp_eq : T1 = T0)
  : cycle_efficiency (max_possible_efficiency T1 T2) = 1 / 12 :=
  sorry

end cyclic_heat_engine_efficiency_l424_424116


namespace fraction_area_circumcircle_covered_l424_424494

theorem fraction_area_circumcircle_covered (a : ℝ) : 
  let R := (a * Real.sqrt 3) / 3,
      small_circle_radius := a / 2,
      A_circumcircle := π * R^2,
      fraction_covered := (3 * (small_circle_radius^2 * (1.123 / 8))) / A_circumcircle 
  in fraction_covered ≈ 0.914 :=
by
  sorry

end fraction_area_circumcircle_covered_l424_424494


namespace polynomial_zero_l424_424560

noncomputable def P (α β r s t : ℤ) (x: ℂ) : ℂ := (x - r) * (x - s) * (x - t) * (x^2 + α * x + β)

theorem polynomial_zero :
  ∃ (r s t : ℤ) (α β : ℤ), 
    ∀ (x : ℂ), 
      P α β r s t x = 0 →
      - (1 + Complex.i * Real.sqrt (15)) / 2 = x :=
  sorry

end polynomial_zero_l424_424560


namespace freshman_to_sophomore_ratio_l424_424973

variable (f s : ℕ)

-- Define the participants from freshmen and sophomores
def freshmen_participants : ℕ := (3 * f) / 7
def sophomores_participants : ℕ := (2 * s) / 3

-- Theorem: There are 14/9 times as many freshmen as sophomores
theorem freshman_to_sophomore_ratio (h : freshmen_participants f = sophomores_participants s) : 
  9 * f = 14 * s :=
by
  sorry

end freshman_to_sophomore_ratio_l424_424973


namespace graph_transformation_correct_l424_424699

noncomputable def g (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 1 then -x else
if 1 < x ∧ x ≤ 3 then sqrt (4 - (x - 1)^2) + 1 else
if 3 < x ∧ x ≤ 5 then x - 3 else 0

def transformed_graph_correct : Prop :=
∀ x : ℝ, g (x - 3) =  -- Define the transformation that describes the graph E

theorem graph_transformation_correct : transformed_graph_correct :=
sorry

end graph_transformation_correct_l424_424699


namespace total_tin_correct_l424_424070

-- Definitions based on conditions
def weight_A := 225
def weight_B := 175
def weight_C := 150

def ratio_lead_tin_A := (5, 3)
def ratio_tin_copper_B := (4, 3)
def ratio_copper_tin_C := (6, 1)

-- Calculate the amount of tin in each alloy
def tin_in_A := (3 / (ratio_lead_tin_A.1 + ratio_lead_tin_A.2)) * weight_A
def tin_in_B := (4 / (ratio_tin_copper_B.1 + ratio_tin_copper_B.2)) * weight_B
def tin_in_C := (1 / (ratio_copper_tin_C.1 + ratio_copper_tin_C.2)) * weight_C

-- Calculate the total amount of tin in the newly formed alloy
def total_tin := tin_in_A + tin_in_B + tin_in_C

-- Theorem stating the total amount of tin in the newly formed alloy
theorem total_tin_correct : total_tin = 205.80357 := by
  sorry

end total_tin_correct_l424_424070


namespace problem_solution_l424_424589

-- Define the set of characters and their possible values
def chars := ['华', '罗', '庚', '金', '杯']
def digits := {1, 2, 3, 4, 5}

-- Defining the property that each sum of pairs must be one of a given set
def isValidSum (sums : List ℕ) : Prop :=
  sums = [4, 5, 6, 7, 8]

-- Define the main problem
def problem : Prop :=
  ∃ (f : Char → ℕ), 
    (∀ c ∈ chars, f c ∈ digits) ∧ 
    (∃ (edges : List (Char × Char)) (sums : List ℕ),
       edges.length = 5 ∧ 
       (∀ (e ∈ edges), (f e.1 + f e.2) ∈ [4, 5, 6, 7, 8]) ∧
       isValidSum (edges.map (λ e, f e.1 + f e.2))) ∧ 
    ( (List.permutations [1,2,3,4,5]).count p|p = f ) = 10

-- The main theorem that encapsulates the problem
theorem problem_solution : problem :=
sorry

end problem_solution_l424_424589


namespace find_b_and_c_l424_424652

theorem find_b_and_c (n : ℝ) (h₀ : 1 < n)
  (h₁ : ∀ a b : ℝ × ℝ, a.1 * b.1 + a.2 * b.2 = (Real.sqrt (a.1 ^ 2 + a.2 ^ 2)) * (Real.sqrt (b.1 ^ 2 + b.2 ^ 2)) * Real.cos (Real.pi / 4))
  (h₂ : ∀ a c : ℝ × ℝ, c.1 = -2 * a.1 → c.2 = 6 * a.2 → (c - a).1 * a.1 + (c - a).2 * a.2 = 0) :
  let a := (1, 2) in
  let b := (-2, 6) in
  let c := (-1, 3) in
  (a.1 * b.1 + a.2 * b.2 = (Real.sqrt (a.1 ^ 2 + a.2 ^ 2)) * (Real.sqrt (b.1 ^ 2 + b.2 ^ 2)) * Real.cos (Real.pi / 4))
  ∧ (c.1 = b.1 / 2 ∧ c.2 = b.2 / 2 ∧ (c - a).1 * a.1 + (c - a).2 * a.2 = 0) :=
by
  sorry

end find_b_and_c_l424_424652


namespace collinear_points_l424_424489

noncomputable theory

/-- Given: Two circles intersect at points A and B. A common tangent intersects these circles at points P and Q.
The tangents to the circumcircle of the triangle APQ at points P and Q intersect at point S.
Let H be the reflection of B over the line PQ.

To prove: The points A, S, and H are collinear. -/
theorem collinear_points (A B P Q S H : Point) (circle1 circle2 : Circle) (linePQ : Line) :
  circle1.IntersectAt A ∧ circle1.IntersectAt B ∧
  circle2.IntersectAt A ∧ circle2.IntersectAt B ∧
  TangentIntersection circle1 circle2 P Q ∧
  TangentToCircumcircle APQ P Q S ∧
  ReflectionOverLine B linePQ H →
  Collinear A S H :=
  sorry

end collinear_points_l424_424489


namespace b_share_of_earnings_l424_424057

def a_work_rate := 1 / 6
def b_work_rate := 1 / 8
def c_work_rate := 1 / 12
def combined_work_rate := a_work_rate + b_work_rate + c_work_rate
def total_earnings := 2340

theorem b_share_of_earnings :
  let b_share := (b_work_rate / combined_work_rate) * total_earnings in
  b_share = 780 :=
by
  -- proof goes here
  sorry

end b_share_of_earnings_l424_424057


namespace sin_double_angle_given_sum_identity_l424_424651

theorem sin_double_angle_given_sum_identity {α : ℝ} 
  (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 5 / 5) : 
  Real.sin (2 * α) = -3 / 5 := 
by 
  sorry

end sin_double_angle_given_sum_identity_l424_424651


namespace angle_not_in_second_quadrant_l424_424579

theorem angle_not_in_second_quadrant :
  (∀ θ ∈ [160, 480, -960, 1530], θ ≠ 1530 → θ ∈ second_quadrant) →
  ¬ 1530 ∈ second_quadrant :=
by
  sorry

notation "second_quadrant" => { θ : ℝ | 90 < θ ∧ θ < 180 ∨ -270 < θ ∧ θ < -180 }

end angle_not_in_second_quadrant_l424_424579


namespace total_numbers_count_l424_424806

theorem total_numbers_count
  (h_avg : ∀ (nums : List ℝ), nums.Sum / nums.length = 3.95)
  (h_avg_1 : ∀ (nums : List ℝ), nums.length = 2 → nums.Sum / nums.length = 3.6)
  (h_avg_2 : ∀ (nums : List ℝ), nums.length = 2 → nums.Sum / nums.length = 3.85)
  (h_avg_3 : ∀ (nums : List ℝ), nums.length = 2 → nums.Sum / nums.length = 4.400000000000001) :
  ∃ (n : ℕ), 2 + 2 + 2 = n :=
by
  use 6
  sorry

end total_numbers_count_l424_424806


namespace jean_to_shirt_ratio_l424_424547

theorem jean_to_shirt_ratio (shirts_sold jeans_sold shirt_cost total_revenue: ℕ) (h1: shirts_sold = 20) (h2: jeans_sold = 10) (h3: shirt_cost = 10) (h4: total_revenue = 400) : 
(shirt_cost * shirts_sold + jeans_sold * ((total_revenue - (shirt_cost * shirts_sold)) / jeans_sold)) / (total_revenue - (shirt_cost * shirts_sold)) / jeans_sold = 2 := 
sorry

end jean_to_shirt_ratio_l424_424547


namespace arrange_abc_l424_424198

noncomputable def a : ℝ := 0.4^2
noncomputable def b : ℝ := 3^0.4
noncomputable def c : ℝ := Real.log 0.3 / Real.log 4

theorem arrange_abc : c < a ∧ a < b := by
  sorry

end arrange_abc_l424_424198


namespace polygon_sides_eq_six_l424_424014

theorem polygon_sides_eq_six (n : ℕ) :
  ((n - 2) * 180 = 2 * 360) → n = 6 := by
  intro h
  have : (n - 2) * 180 = 720 := by exact h
  have : n - 2 = 4 := by linarith
  have : n = 6 := by linarith
  exact this

end polygon_sides_eq_six_l424_424014


namespace pythagorean_theorem_l424_424381

-- Define the geometric setup
structure RightTriangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  (right_angle : ∀ x y z : Type, (A = x) → (B = y) → (C = z) → right_angle x y z)
  (angle_A_right : ∃ a b c : Type, (A = a) → (B = b) → (C = c) → right_angle B A C)

-- Define the Lean statement to be proven
theorem pythagorean_theorem {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (h : RightTriangle A B C) :
  dist B C ^ 2 = dist A C ^ 2 + dist A B ^ 2 :=
sorry

end pythagorean_theorem_l424_424381


namespace min_distinct_values_l424_424556

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (others_max_count : ℕ) 
    (total_count : ℕ) (mode_unique : ℕ) 
    (h1 : mode_count = 11)
    (h2 : others_max_count = 10)
    (h3 : total_count = 2017)
    (h4 : mode_unique = 1) :
    ∃ k, k = 202 ∧ (∑ i in range k, if i = 0 then mode_count else others_max_count) ≥ total_count :=
by {
  sorry
}

end min_distinct_values_l424_424556


namespace joan_change_received_l424_424355

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end joan_change_received_l424_424355


namespace sphere_cube_volume_ratio_l424_424040

theorem sphere_cube_volume_ratio (d a : ℝ) (h_d : d = 12) (h_a : a = 6) :
  let r := d / 2
  let V_sphere := (4 / 3) * π * r^3
  let V_cube := a^3
  V_sphere / V_cube = (4 * π) / 3 :=
by
  sorry

end sphere_cube_volume_ratio_l424_424040


namespace stretched_rhombus_area_l424_424823

noncomputable def area_of_stretched_rhombus : ℝ :=
  let d1 := 6
  let d2 := 4
  let scale := 3
  let d1_new := scale * d1
  let d2_new := scale * d2
  (d1_new * d2_new) / 2

theorem stretched_rhombus_area :
  area_of_stretched_rhombus = 108 :=
by
  -- We acknowledge that the following proof is computational
  -- and uses numerical calculations directly.
  -- Proof is straightforward as it follows arithmetic steps.
  have h : (3 * 6 : ℝ) * (3 * 4) / 2 = 108, by norm_num
  exact h

end stretched_rhombus_area_l424_424823


namespace circumcircles_common_point_l424_424365

-- Define the right triangle ABC with a right angle at C
variables {A B C I D A1 B1 C1 E F K L: Type} [Inhabited A] [Inhabited B] [Inhabited C] 
          [AffineGeometry.point C] 

-- Let ABC be a right triangle with a right angle at C
def right_triangle_angle_C (A B C : AffineGeometry.point C) : Prop :=
  AffineGeometry.angle A C B = π / 2

-- Define I as the incenter of triangle ABC
def incenter (A B C I : AffineGeometry.point C) : Prop :=
  ∀ (P : AffineGeometry.point C), 
  AffineGeometry.dist I P = AffineGeometry.dist I A ∧ 
  AffineGeometry.dist I P = AffineGeometry.dist I B ∧ 
  AffineGeometry.dist I P = AffineGeometry.dist I C

-- Define D as the foot of the altitude from C to AB
def foot_of_altitude (A B C D: AffineGeometry.point C) : Prop := 
  AffineGeometry.perp D C A ∧ AffineGeometry.perp D C B

-- Define the incircle omega of triangle ABC and tangency points A1, B1, and C1
def tangency_points (A1 B1 C1 A B C: AffineGeometry.point C) : Prop := 
  AffineGeometry.tangency A1 B C ∧ 
  AffineGeometry.tangency B1 A C ∧ 
  AffineGeometry.tangency C1 A B

-- Define E and F as the reflections of C in lines C1A1 and C1B1 respectively
def reflections_C_E_F (C1 A1 B1 C E F : AffineGeometry.point C) : Prop :=
  AffineGeometry.reflection_line C1 A1 E C ∧ 
  AffineGeometry.reflection_line C1 B1 F C

-- Define K and L as the reflections of D in lines C1A1 and C1B1 respectively
def reflections_D_K_L (C1 A1 B1 D K L : AffineGeometry.point C) : Prop :=
  AffineGeometry.reflection_line C1 A1 K D ∧ 
  AffineGeometry.reflection_line C1 B1 L D

-- Prove that the circumcircles of triangles A1EI, B1FI, and C1KL have a common point
theorem circumcircles_common_point 
  (A B C I D A1 B1 C1 E F K L : AffineGeometry.point C) 
  (h1 : right_triangle_angle_C A B C) 
  (h2 : incenter A B C I) 
  (h3 : foot_of_altitude A B C D) 
  (h4 : tangency_points A1 B1 C1 A B C) 
  (h5 : reflections_C_E_F C1 A1 B1 C E F) 
  (h6 : reflections_D_K_L C1 A1 B1 D K L) : 
  ∃ P : AffineGeometry.point C, 
    Circle.is_circumcircle A1 E I P ∧ 
    Circle.is_circumcircle B1 F I P ∧ 
    Circle.is_circumcircle C1 K L P :=
begin
  sorry
end

end circumcircles_common_point_l424_424365


namespace count_pairs_satisfying_condition_l424_424268

theorem count_pairs_satisfying_condition:
  {a b : ℕ} (h₁ : a ≥ b) (h₂ : (1 : ℚ) / a + (1 : ℚ) / b = (1 / 6) : ℚ) :
  ↑((Finset.filter (λ ab : ℕ × ℕ, ab.fst ≥ ab.snd ∧ 
     (1:ℚ)/ab.fst + (1:ℚ)/ab.snd = (1/6)) 
     ((Finset.range 100).product (Finset.range 100))).card) = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424268


namespace line_tangent_parabola_unique_d_l424_424833

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l424_424833


namespace matrix_power_identity_l424_424758

open Matrix

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, 4;
     0, 1]

-- State the required proof as a theorem
theorem matrix_power_identity :
  B^20 - 3 * B^19 = !![-1, 4;
                       0, -2] := by
  sorry

end matrix_power_identity_l424_424758


namespace product_ge_n_minus_1_pow_n_l424_424067

theorem product_ge_n_minus_1_pow_n {n : ℕ} (x : Fin n → ℝ) (hx : ∀ i, 0 < x i)
  (h : (∑ i in Finset.univ, 1 / (1 + x i)) = 1) : (∏ i in Finset.univ, x i) ≥ (n - 1)^n :=
sorry

end product_ge_n_minus_1_pow_n_l424_424067


namespace farmer_ducks_sold_l424_424554

theorem farmer_ducks_sold (D : ℕ) (earnings : ℕ) :
  (earnings = (10 * D) + (5 * 8)) →
  ((earnings / 2) * 2 = 60) →
  D = 2 := by
  sorry

end farmer_ducks_sold_l424_424554


namespace sequence_formula_exists_lambda_arithmetic_sequence_l424_424772

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else (1 / 2)^(n - 1)

noncomputable def sum_sequence (n : ℕ) : ℝ :=
  ∑ i in finset.range n, sequence i.succ

noncomputable def is_on_line (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, n ≠ 0 → 2 * a (n + 1) + S n = 2

theorem sequence_formula :
  ∀ n : ℕ, sequence n = (1 / 2)^(n - 1) := 
sorry

theorem exists_lambda_arithmetic_sequence :
  ∃ λ : ℝ, λ = 2 ∧ ∀ n : ℕ, λ ≠ 0 →
  (sum_sequence n + λ * n + λ / (2^n)) - 
  (sum_sequence (n + 1) + λ * (n + 1) + λ / (2^(n + 1))) = 
  (sum_sequence (n + 1) + λ * (n + 1) + λ / (2^(n + 1))) - 
  (sum_sequence (n + 2) + λ * (n + 2) + λ / (2^(n + 2))) :=
sorry

end sequence_formula_exists_lambda_arithmetic_sequence_l424_424772


namespace hexagon_side_length_l424_424404

-- Define the conditions
variables (h : ℝ) -- side length of the hexagon
variables (d : ℝ) -- distance between opposite sides

-- Define the relation between side length and the distance between opposite sides
def is_valid_side_length (h d : ℝ) : Prop :=
  d = (h * (√3)) / 2

-- Now state the theorem
theorem hexagon_side_length :
  ∀ (h d : ℝ), is_valid_side_length h d → d = 12 → h = 8 * √3 :=
begin
  intros h d h_property d_value,
  simp [is_valid_side_length] at h_property,
  rw d_value at h_property,
  sorry
end

end hexagon_side_length_l424_424404


namespace negated_proposition_l424_424781

theorem negated_proposition : ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0 := by
  sorry

end negated_proposition_l424_424781


namespace fraction_of_girls_is_half_l424_424328

variables (T G B : ℝ)
def fraction_x_of_girls (x : ℝ) : Prop :=
  x * G = (1/5) * T ∧ B / G = 1.5 ∧ T = B + G

theorem fraction_of_girls_is_half (x : ℝ) (h : fraction_x_of_girls T G B x) : x = 0.5 :=
sorry

end fraction_of_girls_is_half_l424_424328


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l424_424901

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.45
  let a := 9 -- GCD of 45 and 99
  let numerator := 5
  let denominator := 11
  numerator + denominator = 16 :=
by { 
  sorry 
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l424_424901


namespace Rhett_rent_expense_l424_424411

-- Define the problem statement using given conditions
theorem Rhett_rent_expense
  (late_payments : ℕ := 2)
  (no_late_fees : Bool := true)
  (fraction_of_salary : ℝ := 3 / 5)
  (monthly_salary : ℝ := 5000)
  (tax_rate : ℝ := 0.1) :
  let salary_after_taxes := monthly_salary * (1 - tax_rate)
  let total_late_rent := fraction_of_salary * salary_after_taxes
  let monthly_rent_expense := total_late_rent / late_payments
  monthly_rent_expense = 1350 := by
  sorry

end Rhett_rent_expense_l424_424411


namespace max_value_geom_seq_l424_424654

/-- Given a geometric sequence {a n} with initial term a₁ and common ratio q,
 satisfying
 - product a₂ * a₅ = 2 * a₃,
 - a₄, 5/4, and 2 * a₇ form an arithmetic sequence,
 prove that the maximum value of the product of the first n terms is 1024. -/
theorem max_value_geom_seq (a q : ℕ) :
  (a * q * (a * q^4) = 2 * (a * q^2)) ∧
  (a * q^3 + 2 * (a * q^6) = 2 * (5 / 4)) →
 max (a * q * (a * q^2) * (a * q^3) * ... * (a * q^(n-1))) = 1024 :=
sorry

end max_value_geom_seq_l424_424654


namespace num_increasing_digits_l424_424178

theorem num_increasing_digits :
  let C := λ (n k : ℕ), Nat.choose n k in
  ∑ k in Finset.range 8, C 9 (k + 2) = 502 :=
by
  sorry

end num_increasing_digits_l424_424178


namespace find_general_term_and_sum_l424_424677

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Given conditions
variables (a1 d : ℤ) (an : ℕ → ℤ) (bn : ℕ → ℝ)
variable (a_sum_condition : 4 * a1 + 6 * d = 10)
variable (geo_condition : (a1 + d) * (a1 + 6 * d) = (a1 + 2 * d) ^ 2)
variable (non_zero_d : d ≠ 0)

-- Conclusion we need to show
theorem find_general_term_and_sum (h_an : an = λ n, 3 * n - 5) (h_bn : bn = λ n, 2 ^ (an n)):
    (∀ n, an n = 3 * n - 5) ∧ (∑ i in finset.range n, bn i = (8^n - 1) / 28) :=
by
  sorry

end find_general_term_and_sum_l424_424677


namespace multiple_of_A_share_l424_424570

theorem multiple_of_A_share (a b c : ℤ) (hC : c = 84) (hSum : a + b + c = 427)
  (hEquality1 : ∃ x : ℤ, x * a = 4 * b) (hEquality2 : 7 * c = 4 * b) : ∃ x : ℤ, x = 3 :=
by {
  sorry
}

end multiple_of_A_share_l424_424570


namespace correct_conclusions_l424_424512

def problem_1_statement : Prop :=
  ¬ (∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0

def problem_2_statement : Prop :=
  (∀ a b : ℝ, ab = 0 → (a = 0 ∨ b = 0)) ↔ (∀ a b : ℝ, ¬ (ab = 0) → (a ≠ 0 ∧ b ≠ 0))

def problem_3_statement (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f(x-1) = f(2-x)) → (∀ x : ℝ, f(x) = f(-x))

def problem_4_statement (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f(x + 1) = f(1 - x)) → False

theorem correct_conclusions (f : ℝ → ℝ) :
  (problem_1_statement = false) ∧
  (problem_2_statement) ∧
  (problem_3_statement f) ∧
  ¬ (problem_4_statement f) :=
by sorry

end correct_conclusions_l424_424512


namespace count_pairs_satisfying_condition_l424_424296

theorem count_pairs_satisfying_condition : 
  {p : ℕ × ℕ // p.1 ≥ p.2 ∧ ((1 : ℚ) / p.1 + (1 : ℚ) / p.2 = 1 / 6)}.to_finset.card = 5 := 
sorry

end count_pairs_satisfying_condition_l424_424296


namespace rent_expense_l424_424414

theorem rent_expense (salary gross: ℕ) (tax_percentage: ℕ) (rent_months: ℕ) :
  gross = 5000 → tax_percentage = 10 → rent_months = 2 → salary = gross * (100 - tax_percentage) / 100 → 
  (3 * salary / 5) / rent_months = 1350 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end rent_expense_l424_424414


namespace monomial_exponent_match_l424_424320

theorem monomial_exponent_match (m : ℤ) (x y : ℂ) : (-x^(2*m) * y^3 = 2 * x^6 * y^3) → m = 3 := 
by 
  sorry

end monomial_exponent_match_l424_424320


namespace banquet_attendance_l424_424542

theorem banquet_attendance : 
  ∃ N : ℕ, (219 * 12.95 + N * 17.95 = 9423.70) ∧ (219 + N = 586) :=
by
  sorry

end banquet_attendance_l424_424542


namespace max_ratio_S_over_a_l424_424230

noncomputable def arithmetic_sequence (a_n : ℕ → ℚ) : Prop :=
  ∃ (a d : ℚ), ∀ (n : ℕ), a_n n = a + n * d

noncomputable def arithmetic_sum (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) : Prop :=
  ∀ (n : ℕ), S_n n = ∑ k in finset.range(n + 1), a_n k

variables {a_n : ℕ → ℚ} {S_n : ℕ → ℚ}

theorem max_ratio_S_over_a
  (arith_seq : arithmetic_sequence a_n)
  (arith_sum : arithmetic_sum a_n S_n)
  (h1 : S_n 15 > 0)
  (h2 : S_n 16 < 0)
  : ∃ n, (1 ≤ n ∧ n ≤ 15) ∧ ∀ m, (1 ≤ m ∧ m ≤ 15) → S_n n / a_n n > S_n m / a_n m :=
sorry

end max_ratio_S_over_a_l424_424230


namespace train_pass_time_l424_424919

theorem train_pass_time
  (v : ℝ) (l_tunnel l_train : ℝ) (h_v : v = 75) (h_l_tunnel : l_tunnel = 3.5) (h_l_train : l_train = 0.25) :
  (l_tunnel + l_train) / v * 60 = 3 :=
by 
  -- Placeholder for the proof
  sorry

end train_pass_time_l424_424919


namespace solve_trigonometric_equation_l424_424424

theorem solve_trigonometric_equation (x : ℝ) (k : ℤ) :
  (sin (3 * x) / sin x - 2 * cos (3 * x) / cos x = 5 * |sin x|) →
  x = arsin (1/4) + k * π ∨ x = -arsin (1/4) + k * π :=
by
  sorry

end solve_trigonometric_equation_l424_424424


namespace find_y_l424_424633

theorem find_y (y : ℝ) (h : real.cbrt (y + 16) - real.cbrt (y - 4) = 2) :
  y = 12 ∨ y = -8 :=
sorry

end find_y_l424_424633


namespace checkerboard_swap_condition_l424_424964

-- Definition of the initial checkerboard pattern and the rules
inductive CellColor
| white
| black
| green
deriving DecidableEq, Inhabited

-- Definition of the transformation rule
def recolor (c : CellColor) : CellColor :=
  match c with
  | CellColor.white => CellColor.black
  | CellColor.black => CellColor.green
  | CellColor.green => CellColor.white

-- Definition of the grid and initial conditions
structure CheckerboardGrid (n : ℕ) :=
(grid : Fin n → Fin n → CellColor)
(condition : grid 0 0 = CellColor.black ∨ grid 0 0 = CellColor.white)

-- Swapping condition
def swapped_checkerboard (grid : Fin n → Fin n → CellColor) : Prop :=
∀ i j : Fin n, if (i + j).val % 2 = 0 then grid i j = CellColor.white else grid i j = CellColor.black

-- Define the Lean statement for the problem
theorem checkerboard_swap_condition (n : ℕ) (g : CheckerboardGrid n) :
  (∃ k : ℕ, 3 * k = n) ↔
  (∃ m : ℕ, 3 * m = n ∧ 
  ∃ f : (Fin n → Fin n → CellColor) -> Fin n → Fin n → CellColor, 
  (f g.grid = swapped_checkerboard g.grid)) :=
sorry

end checkerboard_swap_condition_l424_424964


namespace sum_squares_6_to_14_l424_424045

def sum_of_squares (n : ℕ) := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_squares_6_to_14 :
  (sum_of_squares 14) - (sum_of_squares 5) = 960 :=
by
  sorry

end sum_squares_6_to_14_l424_424045


namespace angle_between_sum_and_diff_eq_2_div_3_pi_l424_424251

variables {V : Type*} [inner_product_space ℝ V]

-- Let e1 and e2 be unit vectors with an angle of 60 degrees (pi/3 radians) between them
variables (e1 e2 : V)
variables (h1 : ∥e1∥ = 1)
variables (h2 : ∥e2∥ = 1)
variables (angle_e1_e2 : real.angle e1 e2 = real.pi / 3)

-- Prove the angle between (e1 + e2) and (e2 - 2 • e1) is 2/3 * π
theorem angle_between_sum_and_diff_eq_2_div_3_pi :
  real.angle (e1 + e2) (e2 - 2 • e1) = 2 * real.pi / 3 :=
sorry

end angle_between_sum_and_diff_eq_2_div_3_pi_l424_424251
